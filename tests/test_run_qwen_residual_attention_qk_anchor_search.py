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
import run_qwen_residual_attention_qk_anchor_search as search  # noqa: E402
from neural_forge.hf_cached_attention_trace import (  # noqa: E402
    HFCachedAttentionTraceLimits,
)
from neural_forge.hf_qwen_attention_surgery import (  # noqa: E402
    qwen3_attention_runtime_snapshot,
)
from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
)
from neural_forge.hf_residual_attention_qk_anchor import (  # noqa: E402
    RealQKAnchor,
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
    torch.manual_seed(9797)
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


def _attach_tokenizer_contract(runtime, tokenizer: _Tokenizer) -> None:
    runtime.neural_forge_tokenizer_contract = {
        "schema": "neural-forge-tokenizer-contract-v1",
        "resolved": True,
        "tokenizer_name_or_path": "unit-test-tokenizer",
        "vocab_size": len(tokenizer),
        "vocab_sha256": base._sha256(tokenizer.get_vocab()),
        "special_token_ids": {
            name: getattr(tokenizer, name)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        "parent_token_rows_sha256": base._sha256(
            [int(value) for value in runtime.parent_token_rows.tolist()]
        ),
    }


def test_tokenizer_runtime_binding_rejects_same_size_reordered_vocabulary() -> None:
    runtime = _runtime()
    tokenizer = _Tokenizer()
    _attach_tokenizer_contract(runtime, tokenizer)
    binding = search._tokenizer_runtime_binding(runtime, tokenizer)
    assert binding["exact_vocab_and_special_ids_verified"] is True

    class _ReorderedTokenizer(_Tokenizer):
        def get_vocab(self) -> dict[str, int]:
            result = super().get_vocab()
            result["token-3"], result["token-4"] = (
                result["token-4"],
                result["token-3"],
            )
            return result

    with pytest.raises(
        search.QwenResidualAttentionQKAnchorSearchError,
        match="does not match the checkpoint's exact vocabulary",
    ):
        search._tokenizer_runtime_binding(runtime, _ReorderedTokenizer())


def _lane(value: float, *, comparable: bool = True) -> dict:
    metric_names = (
        "attention_js_divergence",
        "attention_score_geometry_error",
        "residual_transition_error",
        "cache_transition_error",
        "action_margin_damage",
        "topk_damage",
        "action_mismatch",
    )
    return {
        "comparable_internal_evidence": comparable,
        "damage": {
            "risk_score": value,
            "metrics": {
                name: {"quantile": value, "worst": value}
                for name in metric_names
            },
        },
    }


def _candidate_report(
    layer: int,
    *,
    heldout_value: float,
    calibration_value: float,
    greedy_comparable: bool = True,
) -> dict:
    return {
        "anchor_edit": {
            "anchor": {"layer_index": layer, "physical_group": 0}
        },
        "variants": {
            "qk_only": {
                "calibration": {
                    "forced": _lane(calibration_value),
                    "greedy": _lane(calibration_value),
                },
                "heldout": {
                    "forced": _lane(heldout_value),
                    "greedy": _lane(
                        heldout_value,
                        comparable=greedy_comparable,
                    ),
                },
            }
        },
    }


def test_ranking_is_heldout_only_and_prioritizes_greedy_alignment() -> None:
    reports = {
        # Better held-out causal evidence despite intentionally awful
        # calibration evidence.
        "L0G0": _candidate_report(
            0,
            heldout_value=0.1,
            calibration_value=999.0,
        ),
        # Worse held-out evidence despite perfect calibration.
        "L1G0": _candidate_report(
            1,
            heldout_value=0.2,
            calibration_value=0.0,
        ),
        # Numerically best forced evidence, but its autoregressive lane has
        # diverged and must sort after every aligned candidate.
        "L2G0": _candidate_report(
            2,
            heldout_value=0.0,
            calibration_value=0.0,
            greedy_comparable=False,
        ),
    }
    ranking = search._rank_candidates(reports, "qk_only")
    assert [item["anchor_id"] for item in ranking] == ["L0G0", "L1G0", "L2G0"]
    assert all(item["ranking_uses_calibration_rows"] is False for item in ranking)


def test_ranking_rejects_nonfinite_risk_instead_of_emitting_invalid_json() -> None:
    report = _candidate_report(
        0,
        heldout_value=0.1,
        calibration_value=0.1,
    )
    report["variants"]["qk_only"]["heldout"]["forced"]["damage"][
        "risk_score"
    ] = float("nan")
    with pytest.raises(
        search.QwenResidualAttentionQKAnchorSearchError,
        match="finite non-negative risk score",
    ):
        search._rank_candidates({"L0G0": report}, "qk_only")


def test_qk_anchor_search_ranks_heldout_controls_and_rolls_back_exactly() -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)

    report = search.run_qwen_p35_qk_anchor_search(
        runtime,
        (torch.tensor([[1, 4, 6]]),),
        (torch.tensor([[1, 5, 7]]),),
        (RealQKAnchor(0, 0), RealQKAnchor(1, 0)),
        redirect=base.AttentionRedirect(2, 0, 0, 0),
        require_exact_p35=False,
        horizon=2,
        fit_o_ridge=True,
        ridge=1e-4,
        trace_limits=_limits(),
        max_anchors=2,
        max_affected_sites=4,
    )

    assert qwen3_attention_runtime_snapshot(runtime) == before
    assert report["incumbent_rollback_exact"] is True
    assert report["tentative_state_reset_exact"] is True
    assert report["checkpoint_saved"] is False
    assert report["fold_committed"] is False
    assert sorted(report["candidates"]) == ["L0G0", "L1G0"]
    assert len(report["ranking"]["qk_only"]) == 2
    assert len(report["ranking"]["qk_plus_o_ridge"]) == 2
    assert report["ranking"]["objective"]["calibration_used_for_ranking"] is False
    for candidate in report["candidates"].values():
        assert candidate["tentative_reset_exact"] is True
        assert candidate["assignment_audit"][
            "every_shared_destination_assignment_audited"
        ]
        assert set(candidate["variants"]) == {"qk_only", "qk_plus_o_ridge"}
        assert candidate["anchor_edit"]["q_and_k_copied_from_same_real_active_slot"]
        assert candidate["anchor_edit"]["q_or_k_averaged_before_softmax"] is False
        assert candidate["fits"]["o_ridge"]
        for variant in candidate["variants"].values():
            assert set(variant) >= {"calibration", "heldout"}
    assert report["search_contract"]["heldout_only_ranking"] is True
    assert report["search_contract"]["group_permutations_and_role_scales_audited"]


def test_qk_anchor_search_rejects_removed_anchor_before_parent_trace() -> None:
    runtime = _runtime()
    before = qwen3_attention_runtime_snapshot(runtime)

    with pytest.raises(
        search.QwenResidualAttentionQKAnchorSearchError, match="Invalid"
    ):
        search.run_qwen_p35_qk_anchor_search(
            runtime,
            (torch.tensor([[1, 4, 6]]),),
            (torch.tensor([[1, 5, 7]]),),
            (RealQKAnchor(2, 0),),
            redirect=base.AttentionRedirect(2, 0, 0, 0),
            require_exact_p35=False,
            horizon=2,
            fit_o_ridge=False,
            trace_limits=_limits(),
            max_anchors=2,
            max_affected_sites=4,
        )

    assert qwen3_attention_runtime_snapshot(runtime) == before
