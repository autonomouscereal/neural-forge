from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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

import run_qwen_residual_attention_probe as probe  # noqa: E402
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
    torch.manual_seed(9191)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=24,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
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
        attention_prototypes_per_group=2,
        neuron_prototypes_per_neuron=2,
    )


def _fake_registry(*, proposed: bool):
    assignments = (
        torch.tensor([[0, 1], [2, 1]], dtype=torch.int32)
        if proposed
        else torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    )
    sources = (
        torch.tensor([0, 1, 2], dtype=torch.int32)
        if proposed
        else torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    )
    count = int(sources.numel())
    return SimpleNamespace(
        assignments=assignments,
        prototype_source_indices=sources,
        prototype_q_weight=torch.zeros(count, 2, 3),
        prototype_o_weight=torch.zeros(count, 5, 2),
        group_permutations=torch.tensor([[1, 0], [0, 1]], dtype=torch.int32),
        role_scales=None,
        key_value_heads=2,
    )


def test_permutation_aware_affected_sites_include_only_exact_shared_survivor() -> None:
    parent = _fake_registry(proposed=False)
    proposed = _fake_registry(proposed=True)
    parent_bindings = probe._materialized_site_bindings(parent)

    assert parent_bindings[(0, 0)].assignment_axis == 1
    assert parent_bindings[(0, 0)].source_flat_index == 1
    assert parent_bindings[(0, 1)].assignment_axis == 0
    assert parent_bindings[(0, 1)].source_flat_index == 0

    affected, by_source = probe._affected_sites(
        parent,
        proposed,
        (probe.AttentionRedirect(1, 1, 0, 1),),
    )

    assert affected == ((0, 0), (1, 1))
    assert by_source[1] == ((0, 0), (1, 1))
    assert (0, 1) not in affected
    assert (1, 0) not in affected


def test_probe_globally_fits_shared_o_slot_and_rolls_back_exactly(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)
    report = probe.run_qwen_residual_attention_probe(
        runtime,
        (torch.tensor([[1, 4, 6]]),),
        (probe.AttentionRedirect(1, 0, 0, 0),),
        horizon=2,
        compensation="both",
        ridge=1e-4,
        trace_limits=HFCachedAttentionTraceLimits(
            max_prompts=1,
            max_horizon=2,
            max_prompt_tokens=16,
            max_layers=4,
            max_groups=1,
            max_capture_bytes=2_000_000,
        ),
        max_affected_sites=4,
    )

    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert report["incumbent_rollback_exact"] is True
    assert report["tentative_state_reset_exact"] is True
    assert report["affected_assignment_count"] == 2
    assert report["survivor_assignment_sets"]["0"] == [[0, 0], [1, 0]]
    assert sorted(report["variants"]) == ["ridge", "scalar", "uncompensated"]
    assert report["fit_contract"] == {
        "global_across_every_survivor_assignment": True,
        "selected_group_contribution_targets": True,
        "full_delta_single_group_regression": False,
        "o_design_uses_candidate_actual_context": True,
        "parent_fixed_attention_readout_is_diagnostic_only": True,
        "qk_averaged_before_softmax": False,
        "v_and_o_fold_remain_coupled": True,
        "compensation_edits_existing_o_prototypes_only": True,
    }
    fit = report["fits"]["0"]
    assert fit["affected_sites"] == [[0, 0], [1, 0]]
    assert fit["full_attention_delta_used_as_regression_target"] is False
    assert fit["design_contract"].startswith("candidate-runtime")
    assert (
        report["variants"]["ridge"]["edit_audit"]["unaffected_o_slots_byte_exact"]
        is True
    )
    forced_damage = report["variants"]["uncompensated"]["forced"]["damage"]
    assert forced_damage["cache_closed_loop"] is True
    assert forced_damage["token_lane"] == "forced-shared-token"
    assert forced_damage["autoregressive_closed_loop"] is False
    assert report["checkpoint_saved"] is False
    assert report["fold_committed"] is False
    output = tmp_path / "probe.json"
    probe._atomic_write_json(output, report)
    assert (
        json.loads(output.read_text(encoding="utf-8"))["report_sha256"]
        == report["report_sha256"]
    )


def test_candidate_parser_is_directional_and_strict() -> None:
    value = probe.parse_attention_redirect("35:7=12:3")
    assert value.source == (35, 7)
    assert value.target == (12, 3)
    assert value.candidate_id == "L35G7->L12G3"

    with pytest.raises(Exception, match="candidate must be"):
        probe.parse_attention_redirect("35,7->12,3")
