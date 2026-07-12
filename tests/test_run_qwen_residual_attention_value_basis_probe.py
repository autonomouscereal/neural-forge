from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip(
        "Installed Transformers has no Qwen3 implementation.",
        allow_module_level=True,
    )

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_residual_attention_probe as base  # noqa: E402
import run_qwen_residual_attention_value_basis_probe as basis_probe  # noqa: E402
import run_qwen_residual_attention_value_probe as raw_probe  # noqa: E402
from neural_forge.hf_cached_attention_trace import (  # noqa: E402
    HFCachedAttentionTraceLimits,
)
from neural_forge.hf_qwen_attention_surgery import (  # noqa: E402
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self) -> int:
        return 24

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{index}": index for index in range(len(self))}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"


def _runtime():
    torch.manual_seed(9595)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=24,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()
    tokenizer = _Tokenizer()
    return prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=3,
        neuron_prototypes_per_neuron=3,
    )


def _limits() -> HFCachedAttentionTraceLimits:
    return HFCachedAttentionTraceLimits(
        max_prompts=1,
        max_horizon=2,
        max_prompt_tokens=16,
        max_layers=4,
        max_groups=1,
        max_capture_bytes=4_000_000,
        max_projection_bytes=1_000_000,
    )


def _run(runtime):
    return basis_probe.run_qwen_p35_residual_value_basis_probe(
        runtime,
        (torch.tensor([[1, 4, 6]]),),
        (torch.tensor([[1, 5, 7]]),),
        redirect=base.AttentionRedirect(2, 0, 0, 0),
        require_exact_p35=False,
        horizon=2,
        basis_ridge=1e-4,
        o_ridge=1e-4,
        trace_limits=_limits(),
        max_affected_sites=4,
    )


def test_residual_basis_four_control_probe_is_disjoint_and_rolls_back() -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)

    report = _run(runtime)

    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert report["schema"].endswith("-v1")
    assert report["tentative_state_reset_exact"] is True
    assert report["incumbent_rollback_exact"] is True
    assert sorted(report["variants"]) == [
        "basis_v_only",
        "basis_v_then_o",
        "o_only",
        "uncompensated",
    ]
    for variant in report["variants"].values():
        assert set(variant) >= {"calibration", "heldout"}
        assert set(variant["calibration"]) == {
            "forced",
            "greedy",
            "promotion_recommended",
        }
        assert set(variant["heldout"]) == {
            "forced",
            "greedy",
            "promotion_recommended",
        }
    assert report["prompt_splits"]["token_rows_disjoint"] is True
    assert report["prompt_splits"]["heldout"]["used_by_basis_v_fit"] is False
    assert report["prompt_splits"]["heldout"]["used_by_o_fits"] is False
    assert report["basis"]["captured_before_tentative_compaction"] is True
    assert report["basis"]["survivor_source_flat_index"] == 0
    assert report["basis"]["removed_source_flat_index"] == 2
    parent_o_captures = report["basis"]["parent_o_blocks"]
    assert len(parent_o_captures) == report["affected_assignment_count"]
    basis_registry_hashes = {
        entry["source_registry_sha256"] for entry in report["basis"]["entries"]
    }
    assert len(basis_registry_hashes) == 1
    assert {
        capture["source_registry_sha256"] for capture in parent_o_captures.values()
    } == basis_registry_hashes
    assert all(capture["bias_excluded"] for capture in parent_o_captures.values())
    assert report["basis"]["coefficient_policy"] == {
        "prior": {"survivor-source-0": 1.0, "removed-source-2": 0.0},
        "lower": {"survivor-source-0": 0.5, "removed-source-2": -0.5},
        "upper": {"survivor-source-0": 1.5, "removed-source-2": 0.5},
    }
    basis_fit = report["fits"]["residual_targeted_basis_v"]
    assert basis_fit["target_contract"].startswith("parent-selected-group")
    assert basis_fit["runtime_coefficients_retained"] is False
    assert basis_fit["new_runtime_parameters"] == 0
    assert report["fits"]["basis_v_installation"]["parameter_identity_preserved"]
    assert report["fits"]["post_basis_v_o"]
    assert report["fits"]["o_only"]
    assert report["basis_value_trace_audit"][
        "all_trace_hashes_match_final_materialized_v"
    ]
    assert set(report["trace_collection_state_bindings"]) == {
        "parent",
        "uncompensated",
        "o_only",
        "basis_v_only",
        "basis_v_then_o",
    }
    for control in report["trace_collection_state_bindings"].values():
        for split in control.values():
            assert split["state_unchanged_during_collection"] is True
            assert split["before"] == split["after"]
            assert split["attention_registry_content_bound"] is True
            assert split["full_runtime_storage_and_version_bound"] is True
    contract = report["fit_contract"]
    assert contract["raw_parent_v_runner_preserved_and_not_called"] is True
    assert contract["basis_fit_target_is_parent_isolated_group_residual_write"]
    assert contract["basis_fit_design_uses_candidate_fixed_qk_and_current_o"]
    assert contract["post_basis_v_o_fit_recomputed_not_reused"]
    assert contract["post_basis_v_o_design_uses_recollected_actual_child_A_at_V"]
    assert contract["nested_o_then_v_then_incumbent_rollback"]
    assert report["accounting"]["prototype_count_removed"] == 1
    assert report["accounting"]["physical_attention_bytes_saved"] > 0
    assert report["accounting"]["runtime_parameters_added_by_basis"] == 0
    assert report["checkpoint_saved"] is False
    assert report["fold_committed"] is False
    assert report["report_sha256"]


def test_basis_probe_rejects_prompt_leakage_before_mutation() -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)
    prompt = torch.tensor([[1, 4, 6]])

    with pytest.raises(
        basis_probe.QwenResidualAttentionValueBasisProbeError,
        match="must be disjoint",
    ):
        basis_probe.run_qwen_p35_residual_value_basis_probe(
            runtime,
            (prompt,),
            (prompt.clone(),),
            redirect=base.AttentionRedirect(2, 0, 0, 0),
            require_exact_p35=False,
            horizon=2,
            trace_limits=_limits(),
        )

    assert qwen3_attention_runtime_snapshot(runtime) == before


def test_basis_probe_rejects_invalid_ridge_before_mutation() -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)

    with pytest.raises(
        basis_probe.QwenResidualAttentionValueBasisProbeError,
        match="basis_ridge must be",
    ):
        basis_probe.run_qwen_p35_residual_value_basis_probe(
            runtime,
            (torch.tensor([[1, 4, 6]]),),
            (torch.tensor([[1, 5, 7]]),),
            redirect=base.AttentionRedirect(2, 0, 0, 0),
            require_exact_p35=False,
            horizon=2,
            basis_ridge=0.0,
            trace_limits=_limits(),
        )

    assert qwen3_attention_runtime_snapshot(runtime) == before


def test_failure_after_post_basis_o_mutation_restores_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)
    original = base._apply_compensation
    calls = 0

    def fail_after_second_edit(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            raise RuntimeError("synthetic failure after post-basis O mutation")
        return result

    monkeypatch.setattr(base, "_apply_compensation", fail_after_second_edit)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        _run(runtime)

    assert calls == 2
    assert qwen3_attention_runtime_snapshot(runtime) == before


def test_separate_basis_runner_never_calls_raw_value_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("raw V runner was called")

    monkeypatch.setattr(raw_probe, "run_qwen_p35_value_probe", forbidden)
    report = _run(runtime)

    assert report["fit_contract"]["raw_parent_v_runner_preserved_and_not_called"]


def test_cli_and_exact_basis_are_hard_wired_to_p35_lineage() -> None:
    assert basis_probe.EXACT_REDIRECT == base.AttentionRedirect(19, 4, 21, 6)
    assert basis_probe.EXACT_REMOVED_SOURCE == 156
    assert basis_probe.EXACT_SURVIVOR_SOURCE == 174
    assert basis_probe.EXACT_AFFECTED_SITES == ((2, 1), (19, 4), (21, 6))
    assert basis_probe.BASIS_COEFFICIENT_POLICY == {
        "prior": {
            "survivor-source-174-L21G6": 1.0,
            "removed-source-156-L19G4": 0.0,
        },
        "lower": {
            "survivor-source-174-L21G6": 0.5,
            "removed-source-156-L19G4": -0.5,
        },
        "upper": {
            "survivor-source-174-L21G6": 1.5,
            "removed-source-156-L19G4": 0.5,
        },
    }
    args = basis_probe.build_parser().parse_args(
        [
            "--codebook-checkpoint",
            "p35",
            "--tokenizer",
            "qwen",
            "--calibration-prompt",
            "alpha",
            "--heldout-prompt",
            "beta",
            "--output",
            "report.json",
        ]
    )
    assert not hasattr(args, "candidate")
    assert args.codebook_checkpoint == Path("p35")
    assert args.basis_ridge == pytest.approx(1e-4)
    assert args.o_ridge == pytest.approx(1e-6)
