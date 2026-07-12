from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip(
        "Installed Transformers has no Qwen3 implementation.", allow_module_level=True
    )

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_qwen_residual_attention_probe as base  # noqa: E402
import run_qwen_residual_attention_value_probe as value_probe  # noqa: E402
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
    torch.manual_seed(9494)
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


def test_four_control_value_probe_uses_disjoint_splits_and_rolls_back() -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)

    report = value_probe.run_qwen_p35_value_probe(
        runtime,
        (torch.tensor([[1, 4, 6]]),),
        (torch.tensor([[1, 5, 7]]),),
        redirect=base.AttentionRedirect(2, 0, 0, 0),
        require_exact_p35=False,
        horizon=2,
        ridge=1e-4,
        trace_limits=_limits(),
        max_affected_sites=4,
    )

    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert report["schema"].endswith("-v2")
    assert report["tentative_state_reset_exact"] is True
    assert report["incumbent_rollback_exact"] is True
    assert sorted(report["variants"]) == [
        "o_only",
        "uncompensated",
        "v_only",
        "v_then_o",
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
    assert report["prompt_splits"]["heldout"]["used_by_v_fit"] is False
    assert report["prompt_splits"]["heldout"]["used_by_o_fits"] is False
    assert report["fit_contract"]["post_v_o_fit_recomputed_not_reused"] is True
    assert report["fit_contract"][
        "live_v_installation_revalidated_immediately_before_o_refit"
    ]
    assert report["fit_contract"]["v_only_control_is_always_retained_even_if_harmful"]
    assert report["fit_contract"]["raw_parent_v_cache_reconstruction_is_diagnostic"]
    assert report["fit_contract"][
        "promotion_objective_is_heldout_post_v_residual_and_action_behavior"
    ]
    assert (
        report["fit_contract"]["post_v_o_design_uses_recollected_actual_child_A_at_V"]
        is True
    )
    assert (
        report["value_trace_audit"]["all_trace_hashes_match_final_materialized_v"]
        is True
    )
    assert set(report["trace_collection_state_bindings"]) == {
        "parent",
        "uncompensated",
        "o_only",
        "v_only",
        "v_then_o",
    }
    for control in report["trace_collection_state_bindings"].values():
        for split in control.values():
            assert split["state_unchanged_during_collection"] is True
            assert split["before"] == split["after"]
            assert split["attention_registry_content_bound"] is True
            assert split["full_runtime_storage_and_version_bound"] is True
    assert report["fits"]["value"]
    for fit in report["fits"]["value"].values():
        assert fit["calibration_v_sha256_by_layer"]
        assert fit["runtime_contract"]["cross_tokenizer_mapping_supported"] is False
    assert report["fits"]["post_v_o"]
    assert report["fits"]["o_only"]
    assert report["accounting"]["prototype_count_removed"] == 1
    assert report["accounting"]["physical_attention_bytes_saved"] > 0
    assert report["checkpoint_saved"] is False
    assert report["fold_committed"] is False
    assert report["report_sha256"]


def test_value_probe_rejects_prompt_leakage_before_mutation() -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)
    prompt = torch.tensor([[1, 4, 6]])

    with pytest.raises(
        value_probe.QwenResidualAttentionValueProbeError,
        match="must be disjoint",
    ):
        value_probe.run_qwen_p35_value_probe(
            runtime,
            (prompt,),
            (prompt.clone(),),
            redirect=base.AttentionRedirect(2, 0, 0, 0),
            require_exact_p35=False,
            horizon=2,
            trace_limits=_limits(),
        )

    assert qwen3_attention_runtime_snapshot(runtime) == before


def test_failure_after_post_v_o_mutation_still_restores_incumbent(
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
            raise RuntimeError("synthetic failure after post-V O mutation")
        return result

    monkeypatch.setattr(base, "_apply_compensation", fail_after_second_edit)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        value_probe.run_qwen_p35_value_probe(
            runtime,
            (torch.tensor([[1, 4, 6]]),),
            (torch.tensor([[1, 5, 7]]),),
            redirect=base.AttentionRedirect(2, 0, 0, 0),
            require_exact_p35=False,
            horizon=2,
            ridge=1e-4,
            trace_limits=_limits(),
            max_affected_sites=4,
        )

    assert calls == 2
    assert qwen3_attention_runtime_snapshot(runtime) == before


def test_cli_is_hard_wired_to_exact_p35_redirect() -> None:
    assert value_probe.EXACT_REDIRECT == base.AttentionRedirect(19, 4, 21, 6)
    parser = value_probe.build_parser()
    args = parser.parse_args(
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


def test_block_o_layout_refuses_duplicate_layer_targets() -> None:
    with pytest.raises(
        value_probe.QwenResidualAttentionValueProbeError,
        match="refuses to duplicate",
    ):
        value_probe._assert_block_o_layout({7: ((2, 0), (2, 1), (4, 0))})

    value_probe._assert_block_o_layout({7: ((2, 0), (3, 1), (4, 0))})


def test_trace_collection_binding_rejects_runtime_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    registry = base._registry(runtime)

    def mutate_runtime(*_args, **_kwargs):
        with torch.no_grad():
            registry.prototype_q_weight.reshape(-1)[0].add_(1.0)
        return {}

    monkeypatch.setattr(base, "_collect_candidate_traces", mutate_runtime)
    with pytest.raises(
        value_probe.QwenResidualAttentionValueProbeError,
        match="changed its bound registry/runtime state",
    ):
        value_probe._collect_both_splits(
            runtime,
            (torch.tensor([[1, 4, 6]]),),
            (torch.tensor([[1, 5, 7]]),),
            ((0, 0),),
            {0: (8, 9)},
            {0: (8, 9)},
            label="adversarial",
            horizon=2,
            limits=_limits(),
        )
