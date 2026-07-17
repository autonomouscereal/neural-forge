from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_forge.dynamic_sequence_horizon import (
    SequenceHorizonContractError,
    assert_model_context_preserved,
    build_sequence_horizon_contract,
    evaluation_rollout_contract,
    folding_evidence_horizon,
)


class _Config(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


def _config(*, capacity: int = 128_000):
    return _Config(
        model_type="lfm2",
        max_position_embeddings=capacity,
        rope_parameters={
            "rope_theta": 1_000_000.0,
            "rope_type": "default",
        },
        hidden_size=1024,
    )


def _manifest():
    return {
        "capture": {
            "sample_last_tokens": 8,
            "max_decode_steps": 64,
        },
        "corpus": {
            "prompt_token_counts": [19, 18],
            "forced_decode_token_counts": [9, 58],
        },
    }


def test_real_cache_shape_reports_76_not_architectural_64_cap() -> None:
    report = folding_evidence_horizon(_manifest())
    assert report["configured_decode_ceiling_tokens"] == 64
    assert report["maximum_observed_full_prefix_tokens"] == 76
    assert report["retained_prefill_rows"] == 16
    assert report["retained_decode_rows"] == 67
    assert report["retained_causal_rows"] == 83
    assert report["full_prefill_history_retained"] is False


def test_evaluation_limit_is_a_separate_rollout_contract() -> None:
    report = build_sequence_horizon_contract(
        parent_config=_config(),
        child_config=_config(),
        trace_manifest=_manifest(),
        evaluation_maximum_new_tokens=64,
    )
    assert report["model_context"]["child"]["capacity_tokens"] == 128_000
    assert report["folding_evidence"]["maximum_observed_full_prefix_tokens"] == 76
    assert report["evaluation_rollout"]["maximum_new_tokens"] == 64
    assert report["evaluation_rollout"]["used_as_folding_evidence_horizon"] is False
    assert report["evaluation_rollout"]["used_as_model_context_capacity"] is False
    assert report["evaluation_rollout"]["used_as_consumer_fit_row_limit"] is False
    assert report["evaluation_rollout"][
        "used_to_select_or_limit_sealed_cache_batches"
    ] is False
    assert report["evaluation_rollout"]["evaluation_only"] is True
    assert report["behavioral_prefix_probe_coverage"] == {
        "64": True,
        "256": False,
        "1024": False,
    }
    assert report["uncovered_behavioral_prefixes"] == [256, 1024]
    assert report["long_context_behavior_proven"] is False


def test_evaluation_rollout_report_cannot_claim_folding_scope() -> None:
    report = evaluation_rollout_contract(64)
    assert report == {
        "maximum_new_tokens": 64,
        "purpose": "candidate_behavior_rollout_stopping_condition_only",
        "used_as_model_context_capacity": False,
        "used_as_folding_evidence_horizon": False,
        "used_as_consumer_fit_row_limit": False,
        "used_to_select_or_limit_sealed_cache_batches": False,
        "evaluation_only": True,
    }


def test_fold_contract_has_no_implicit_evaluation_limit() -> None:
    report = build_sequence_horizon_contract(
        parent_config=_config(),
        child_config=_config(),
        trace_manifest=_manifest(),
        evaluation_maximum_new_tokens=None,
    )
    assert report["evaluation_rollout"] is None
    assert report["evaluation_limit_can_cap_folding"] is False
    assert report["limits_are_independently_named_and_bound"] is True


def test_cache_prefix_coverage_is_not_child_behavior_proof() -> None:
    manifest = _manifest()
    manifest["corpus"]["prompt_token_counts"] = [1024]
    manifest["corpus"]["forced_decode_token_counts"] = [1]
    report = build_sequence_horizon_contract(
        parent_config=_config(),
        child_config=_config(),
        trace_manifest=manifest,
        evaluation_maximum_new_tokens=None,
    )
    assert report["uncovered_behavioral_prefixes"] == []
    assert report["folding_evidence_covers_all_requested_prefixes"] is True
    assert report["long_context_behavior_preserved"] is None
    assert report["long_context_behavior_proven"] is False
    assert report["long_context_behavior_requires_explicit_child_probe"] is True


def test_structural_runner_and_evaluator_bind_limits_to_separate_reports() -> None:
    root = Path(__file__).resolve().parents[1]
    runner_tree = ast.parse(
        (root / "experiments" / "run_lfm2_dynamic_structural_compression.py")
        .read_text(encoding="utf-8")
    )
    evaluator_tree = ast.parse(
        (
            root
            / "experiments"
            / "evaluate_lfm2_dynamic_structural_compression.py"
        ).read_text(encoding="utf-8")
    )
    runner_contract_calls = [
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_sequence_horizon_contract"
    ]
    assert len(runner_contract_calls) == 1
    evaluation_keywords = {
        keyword.arg: keyword.value
        for keyword in runner_contract_calls[0].keywords
    }
    assert isinstance(
        evaluation_keywords["evaluation_maximum_new_tokens"],
        ast.Constant,
    )
    assert (
        evaluation_keywords["evaluation_maximum_new_tokens"].value
        is None
    )

    explicit_evaluation_fields = [
        value
        for node in ast.walk(evaluator_tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
        and key.value == "evaluation_rollout"
    ]
    assert len(explicit_evaluation_fields) == 1
    value = explicit_evaluation_fields[0]
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == "evaluation_rollout_contract"


def test_context_capacity_or_rope_drift_fails_closed() -> None:
    with pytest.raises(
        SequenceHorizonContractError,
        match="changed context capacity",
    ):
        assert_model_context_preserved(_config(), _config(capacity=64))

    child = _config()
    child.rope_parameters = {
        "rope_theta": 10_000.0,
        "rope_type": "default",
    }
    with pytest.raises(
        SequenceHorizonContractError,
        match="changed context capacity",
    ):
        assert_model_context_preserved(_config(), child)


def test_misaligned_or_over_ceiling_trace_counts_fail_closed() -> None:
    manifest = _manifest()
    manifest["corpus"]["forced_decode_token_counts"] = [65, 1]
    with pytest.raises(
        SequenceHorizonContractError,
        match="exceeds its configured decode ceiling",
    ):
        folding_evidence_horizon(manifest)

    manifest = _manifest()
    manifest["corpus"]["prompt_token_counts"] = [19]
    with pytest.raises(
        SequenceHorizonContractError,
        match="nonempty and aligned",
    ):
        folding_evidence_horizon(manifest)
