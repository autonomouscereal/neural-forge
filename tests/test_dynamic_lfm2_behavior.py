from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from neural_forge.dynamic_behavior_gate import resolve_behavior_gate_binding
from neural_forge.dynamic_fold_executor import FoldDecision
from neural_forge.dynamic_lfm2_behavior import (
    CombinedLfm2FinalEvaluator,
    DynamicLfm2BehaviorError,
    Lfm2WholeOutputBehaviorEvaluator,
    load_lfm2_cached_whole_output_baseline,
    seal_lfm2_cached_whole_output_envelope,
)


PARENT_CACHE_KEY = "a" * 64
PARENT_STATE = "b" * 64
TOKENIZER = "c" * 64


def _write_rows(path):
    rows = [
        {
            "id": "math",
            "prompt": "What is 6 * 7?",
            "category": "math",
            "expected_response": "42",
            "parent_output_text": "42",
            "parent_output_ids": [4, 2],
        },
        {
            "id": "code",
            "prompt": "Write a tiny Python function named add that returns +.",
            "category": "code",
            "expected_response": "def add(a, b):\n    return a + b\n```",
            "parent_output_text": "def add():\n    return 0",
            "parent_output_ids": [5, 6, 7],
        },
        {
            "id": "empty-parent",
            "prompt": "Unknown control prompt.",
            "category": "instruction",
            "expected_response": None,
            "parent_output_text": "",
            "parent_output_ids": [7],
        },
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def _seal(tmp_path, *, generation_limit=64):
    rows = tmp_path / "parent-rows.json"
    envelope = tmp_path / "parent-envelope.json"
    _write_rows(rows)
    receipt = seal_lfm2_cached_whole_output_envelope(
        rows,
        envelope,
        parent_cache_key=PARENT_CACHE_KEY,
        parent_state_fingerprint=PARENT_STATE,
        tokenizer_fingerprint=TOKENIZER,
        generation_limit=generation_limit,
    )
    return envelope, receipt


def _load(envelope, receipt, *, generation_limit=64):
    return load_lfm2_cached_whole_output_baseline(
        envelope,
        expected_sha256=receipt["file_sha256"],
        expected_parent_cache_key=PARENT_CACHE_KEY,
        expected_parent_state_fingerprint=PARENT_STATE,
        expected_tokenizer_fingerprint=TOKENIZER,
        expected_generation_limit=generation_limit,
    )


def test_cached_lfm2_baseline_is_parent_bound_and_scores_tasks(tmp_path) -> None:
    envelope, receipt = _seal(tmp_path)
    baseline = _load(envelope, receipt)

    assert len(baseline.cases) == 3
    math = baseline.cases[0]
    code = baseline.cases[1]
    assert baseline.task_score(math, "The answer is 42.") is True
    assert baseline.correctness_score(math, "The answer is 42.") is True
    assert baseline.correctness_score(math, "The answer is 42.5.") is False
    assert (
        baseline.task_score(
            code,
            "def add(a, b):\n    return a + b",
        )
        is True
    )
    assert (
        baseline.correctness_score(
            code,
            "def add(a, b):\n    return a + b + 1",
        )
        is False
    )
    assert (
        baseline.task_score(
            code,
            "def ADD(A, B):\n    return A + B",
        )
        is False
    )
    assert baseline.task_score(code, code.parent_output) is True
    assert baseline.correctness_score(code, code.parent_output) is False
    assert baseline.cases[2].parent_empty is True
    assert baseline.as_dict()["parent_forward_passes"] == 0
    assert len(baseline.fingerprint) == 64

    with pytest.raises(DynamicLfm2BehaviorError, match="fingerprint differs"):
        load_lfm2_cached_whole_output_baseline(
            envelope,
            expected_sha256="0" * 64,
            expected_parent_cache_key=PARENT_CACHE_KEY,
            expected_parent_state_fingerprint=PARENT_STATE,
            expected_tokenizer_fingerprint=TOKENIZER,
        )
    with pytest.raises(DynamicLfm2BehaviorError, match="parent_cache_key differs"):
        load_lfm2_cached_whole_output_baseline(
            envelope,
            expected_sha256=receipt["file_sha256"],
            expected_parent_cache_key="d" * 64,
            expected_parent_state_fingerprint=PARENT_STATE,
            expected_tokenizer_fingerprint=TOKENIZER,
        )


def test_nonempty_unknown_case_cannot_enter_semantic_baseline(tmp_path) -> None:
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            [
                {
                    "id": "unknown",
                    "prompt": "Unknown.",
                    "category": "unknown",
                    "expected_response": None,
                    "parent_output_text": "Some unvalidated output.",
                    "parent_output_ids": [1, 2],
                }
            ]
        ),
        encoding="utf-8",
    )
    envelope = tmp_path / "envelope.json"
    receipt = seal_lfm2_cached_whole_output_envelope(
        rows,
        envelope,
        parent_cache_key=PARENT_CACHE_KEY,
        parent_state_fingerprint=PARENT_STATE,
        tokenizer_fingerprint=TOKENIZER,
    )
    with pytest.raises(DynamicLfm2BehaviorError, match="no semantic validator"):
        _load(envelope, receipt)


class _Trajectory:
    def __init__(self, accepted, *, return_bool=False):
        self.accepted = accepted
        self.return_bool = return_bool
        self.calls = []

    def evaluate(self, _context):
        self.calls.append("memory")
        if self.return_bool:
            return self.accepted
        if self.accepted:
            return FoldDecision.accept("trajectory accepted", {"score": 0.9})
        return FoldDecision.reject("trajectory rejected", {"score": 0.1})

    def evaluate_fresh_reload(self, _model):
        self.calls.append("reload")
        return self.evaluate(None)


class _Behavior:
    def __init__(self, accepted):
        self.accepted = accepted
        self.calls = []

    def evaluate_model(self, _model, *, phase):
        self.calls.append(phase)
        if self.accepted:
            return FoldDecision.accept("behavior accepted", {"retention": 1.0})
        return FoldDecision.reject("behavior rejected", {"retention": 0.5})


def test_combined_final_gate_requires_trajectory_and_whole_output() -> None:
    context = SimpleNamespace(execution_context=SimpleNamespace(child_model=object()))
    trajectory = _Trajectory(True)
    behavior = _Behavior(False)
    evaluator = CombinedLfm2FinalEvaluator(
        trajectory_evaluator=trajectory,
        behavior_evaluator=behavior,
    )

    rejected = evaluator.evaluate(context)

    assert rejected.accepted is False
    assert behavior.calls == ["accepted_in_memory_child"]
    assert dict(rejected.metrics)["trajectory_score"] == pytest.approx(0.9)
    assert dict(rejected.metrics)["whole_output_retention"] == pytest.approx(0.5)

    short_circuit_behavior = _Behavior(True)
    short_circuit = CombinedLfm2FinalEvaluator(
        trajectory_evaluator=_Trajectory(False, return_bool=True),
        behavior_evaluator=short_circuit_behavior,
    )
    rejected_trajectory = short_circuit.evaluate(context)
    assert rejected_trajectory.accepted is False
    assert short_circuit_behavior.calls == []


def test_combined_final_evaluator_records_numeric_failures_for_manual_review() -> (
    None
):
    context = SimpleNamespace(execution_context=SimpleNamespace(child_model=object()))
    trajectory = _Trajectory(False)
    behavior = _Behavior(False)
    evaluator = CombinedLfm2FinalEvaluator(
        trajectory_evaluator=trajectory,
        behavior_evaluator=behavior,
        performance_metrics_diagnostic_only=True,
    )

    decision = evaluator.evaluate(context)

    assert decision.accepted is True
    assert behavior.calls == ["accepted_in_memory_child"]
    assert dict(decision.metrics)["counterfactual_trajectory_accepted"] == 0.0
    assert dict(decision.metrics)["counterfactual_whole_output_accepted"] == 0.0
    assert dict(decision.metrics)["numeric_decision_authority"] == 0.0
    event = evaluator.events[-1]["summary"]
    assert event["trajectory"]["accepted"] is False
    assert event["whole_output"]["accepted"] is False
    assert event["performance_metrics_diagnostic_only"] is True
    assert event["manual_intelligence_review_required"] is True


def test_real_generator_accepts_concise_coherent_nonidentical_output(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    rows = tmp_path / "rows.json"
    rows.write_text(
        json.dumps(
            [
                {
                    "id": "math",
                    "prompt": "What is 6 * 7?",
                    "category": "math",
                    "expected_response": "42",
                    "parent_output_text": "The answer is 42.",
                    "parent_output_ids": [1, 2, 3],
                }
            ]
        ),
        encoding="utf-8",
    )
    envelope = tmp_path / "envelope.json"
    receipt = seal_lfm2_cached_whole_output_envelope(
        rows,
        envelope,
        parent_cache_key=PARENT_CACHE_KEY,
        parent_state_fingerprint=PARENT_STATE,
        tokenizer_fingerprint=TOKENIZER,
        generation_limit=4,
    )
    baseline = _load(envelope, receipt, generation_limit=4)

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, _text, *, return_tensors, add_special_tokens):
            assert return_tensors == "pt"
            assert add_special_tokens is True
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

        def decode(self, _tokens, **_kwargs):
            return "42"

    class Model(torch.nn.Module):
        def __init__(self, *, preserve_prefix=True):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.preserve_prefix = preserve_prefix

        def forward(self, input_ids):
            return input_ids * self.weight

        def generate(self, *, input_ids, **_kwargs):
            self(input_ids)
            prefix = input_ids if self.preserve_prefix else input_ids + 1
            suffix = torch.tensor([[4, 2, 5]], device=input_ids.device)
            return torch.cat((prefix, suffix), dim=1)

    evaluator = Lfm2WholeOutputBehaviorEvaluator(
        baseline=baseline,
        tokenizer=Tokenizer(),
    )
    final_binding = resolve_behavior_gate_binding(evaluator.evaluate)
    fresh_binding = resolve_behavior_gate_binding(evaluator.evaluate_fresh_reload)
    callable_binding = resolve_behavior_gate_binding(evaluator)
    assert final_binding.manifest_fingerprint == fresh_binding.manifest_fingerprint
    assert final_binding.manifest_fingerprint == callable_binding.manifest_fingerprint
    assert final_binding.manifest["coverage_policy"]["required_splits"] == ["heldout"]
    decision = evaluator.evaluate_model(Model(), phase="test")

    assert decision.accepted is True
    assert dict(decision.metrics)["task_score_retention"] == pytest.approx(1.0)
    assert dict(decision.metrics)["per_case_task_regression_count"] == 0
    assert evaluator.events[0]["parent_forward_passes"] == 0

    with pytest.raises(DynamicLfm2BehaviorError, match="prefix is invalid"):
        evaluator.evaluate_model(
            Model(preserve_prefix=False),
            phase="bad-prefix",
        )


def test_evaluator_rejects_invalid_fragment_threshold(tmp_path) -> None:
    envelope, receipt = _seal(tmp_path)
    baseline = _load(envelope, receipt)
    with pytest.raises(
        DynamicLfm2BehaviorError,
        match="positive integer",
    ):
        Lfm2WholeOutputBehaviorEvaluator(
            baseline=baseline,
            tokenizer=object(),
            fragment_minimum_characters=-1,
        )


def test_evaluator_binds_parent_relative_ninety_percent_policy(tmp_path) -> None:
    envelope, receipt = _seal(tmp_path)
    baseline = _load(envelope, receipt)
    evaluator = Lfm2WholeOutputBehaviorEvaluator(
        baseline=baseline,
        tokenizer=object(),
        maximum_new_empty_responses=2,
        maximum_new_fragment_responses=2,
        minimum_parent_nonempty_retention=0.9,
        minimum_task_score_retention=0.9,
        maximum_parent_task_score_regressions=2,
        maximum_new_generation_limit_saturations=2,
        maximum_repeated_ngram_share_regressions=2,
    )

    threshold = resolve_behavior_gate_binding(evaluator).manifest[
        "threshold_config"
    ]
    assert threshold["maximum_new_empty_responses"] == 2
    assert threshold["maximum_new_fragment_responses"] == 2
    assert threshold["minimum_parent_nonempty_retention"] == pytest.approx(0.9)
    assert threshold["minimum_task_score_retention"] == pytest.approx(0.9)
    assert threshold["maximum_parent_task_score_regressions"] == 2
    assert threshold["maximum_new_generation_limit_saturations"] == 2
    assert threshold["maximum_repeated_ngram_share_regressions"] == 2


def test_evaluator_honors_configured_per_case_regression_budget(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    envelope, receipt = _seal(tmp_path, generation_limit=4)
    baseline = _load(envelope, receipt, generation_limit=4)

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, _text, *, return_tensors, add_special_tokens):
            assert return_tensors == "pt"
            assert add_special_tokens is True
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

        def decode(self, _tokens, **_kwargs):
            return "not a number"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, input_ids):
            return input_ids * self.weight

        def generate(self, *, input_ids, **_kwargs):
            self(input_ids)
            return torch.cat(
                (
                    input_ids,
                    torch.tensor([[4, 2, 5]], device=input_ids.device),
                ),
                dim=1,
            )

    allowed = Lfm2WholeOutputBehaviorEvaluator(
        baseline=baseline,
        tokenizer=Tokenizer(),
        minimum_task_score_retention=0.0,
        maximum_parent_task_score_regressions=2,
        maximum_repeated_ngram_share_regressions=2,
        maximum_repeated_bigram_share_delta=1.0,
    ).evaluate_model(Model(), phase="allowed-regression")
    strict = Lfm2WholeOutputBehaviorEvaluator(
        baseline=baseline,
        tokenizer=Tokenizer(),
        minimum_task_score_retention=0.0,
        maximum_parent_task_score_regressions=1,
        maximum_repeated_ngram_share_regressions=2,
        maximum_repeated_bigram_share_delta=1.0,
    ).evaluate_model(Model(), phase="strict-regression")

    assert allowed.accepted is True
    assert dict(allowed.metrics)["per_case_task_regression_count"] == 2
    assert strict.accepted is False
    assert "regression budget exceeded" in strict.reason


def test_envelope_file_hash_covers_exact_written_bytes(tmp_path) -> None:
    envelope, receipt = _seal(tmp_path)
    assert hashlib.sha256(envelope.read_bytes()).hexdigest() == (receipt["file_sha256"])
