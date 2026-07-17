from __future__ import annotations

from dataclasses import replace

import pytest

from neural_forge.dynamic_behavior_gate import (
    CachedWholeOutputBehaviorGate,
    CandidateWholeOutput,
    DynamicBehaviorGateError,
    TaskScoreCoveragePolicy,
    WholeOutputCase,
    WholeOutputGateConfig,
    WholeOutputMetadata,
    identify_behavior_callable,
    resolve_behavior_gate_binding,
)


torch = pytest.importorskip("torch")


def _cases():
    return (
        WholeOutputCase(
            "chat",
            "Say hello.",
            "Hello! How can I help?",
            category="conversation",
            parent_eos_margin=2.0,
        ),
        WholeOutputCase(
            "math",
            "What is 2 + 3?",
            "5",
            category="math",
        ),
        WholeOutputCase(
            "parent-blank",
            "An unsupported task.",
            "",
            category="other",
        ),
    )


def test_gate_accepts_coherent_paraphrases_without_exact_text_matching() -> None:
    gate = CachedWholeOutputBehaviorGate(
        _cases(),
        generator=lambda _model, _cases: {
            "chat": CandidateWholeOutput(
                "Hi there—what can I do for you?",
                eos_margin=2.2,
            ),
            "math": "The result is 5.",
            "parent-blank": "",
        },
        task_scorer=lambda case, output: (
            "5" in output if case.case_id == "math" else None
        ),
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)

    assert decision.accepted
    assert dict(decision.metrics)["new_empty_count"] == 0
    assert dict(decision.metrics)["exact_text_equality_is_diagnostic_only"] is True


def test_binding_gate_requires_observed_candidate_model_execution() -> None:
    model = torch.nn.Linear(2, 2).eval()
    cached_replay = CachedWholeOutputBehaviorGate(
        (WholeOutputCase("case", "Prompt", "answer"),),
        generator=lambda _model, _cases: {"case": "answer"},
    )

    with pytest.raises(
        DynamicBehaviorGateError,
        match="without executing the candidate model",
    ):
        cached_replay.evaluate_model(model)

    def live_generator(candidate, _cases):
        with torch.no_grad():
            candidate(torch.ones((1, 2)))
        return {"case": "answer"}

    live_gate = CachedWholeOutputBehaviorGate(
        (WholeOutputCase("case", "Prompt", "answer"),),
        generator=live_generator,
    )
    decision = live_gate.evaluate_model(model)
    metrics = dict(decision.metrics)

    assert decision.accepted
    assert metrics["model_execution_verified"] is True
    assert metrics["observed_child_module_forward_calls"] > 0
    assert metrics["observed_child_root_forward_calls"] > 0


def test_binding_gate_rejects_leaf_only_execution_with_cached_return() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.ReLU(),
    ).eval()

    def leaf_only_generator(candidate, _cases):
        with torch.no_grad():
            candidate[0](torch.ones((1, 2)))
        return {"case": "cached answer"}

    gate = CachedWholeOutputBehaviorGate(
        (WholeOutputCase("case", "Prompt", "cached answer"),),
        generator=leaf_only_generator,
    )

    with pytest.raises(
        DynamicBehaviorGateError,
        match="without executing the candidate model root",
    ):
        gate.evaluate_model(model)


def test_gate_rejects_a_new_empty_response_even_when_traces_look_good() -> None:
    gate = CachedWholeOutputBehaviorGate(
        _cases(),
        generator=lambda _model, _cases: {
            "chat": "",
            "math": "5",
            "parent-blank": "",
        },
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)

    assert not decision.accepted
    assert "new_empty=chat" in decision.reason


def test_gate_rejects_fragment_eos_and_task_correctness_regressions() -> None:
    gate = CachedWholeOutputBehaviorGate(
        _cases(),
        generator=lambda _model, _cases: {
            "chat": CandidateWholeOutput(
                "Hello",
                fragment_like=True,
                eos_margin=0.5,
            ),
            "math": "6",
            "parent-blank": "",
        },
        config=WholeOutputGateConfig(
            minimum_eos_margin_delta=-0.25,
        ),
        task_scorer=lambda case, output: (
            output.strip() == "5" if case.case_id == "math" else None
        ),
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)

    assert not decision.accepted
    assert "new_fragments=chat" in decision.reason
    assert "eos_margin_regression=chat" in decision.reason
    assert "task_score_retention_below_threshold" in decision.reason


def test_per_case_task_loss_cannot_be_masked_by_a_different_task_gain() -> None:
    cases = (
        WholeOutputCase("retained", "First task", "pass"),
        WholeOutputCase("gained", "Second task", "fail"),
    )
    gate = CachedWholeOutputBehaviorGate(
        cases,
        generator=lambda _model, _cases: {
            "retained": "fail",
            "gained": "pass",
        },
        task_scorer=lambda _case, output: output == "pass",
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)
    metrics = dict(decision.metrics)

    assert not decision.accepted
    assert decision.reason.endswith("parent_task_score_regression=retained")
    assert metrics["task_score_retention"] == pytest.approx(1.0)
    assert metrics["parent_task_score_regression_count"] == 1


def test_critical_case_can_require_nonempty_even_when_parent_was_blank() -> None:
    case = WholeOutputCase(
        "safety",
        "What should I do if I smell gas?",
        "",
        category="safety",
        critical_nonempty=True,
    )
    gate = CachedWholeOutputBehaviorGate(
        (case,),
        generator=lambda _model, _cases: {"safety": ""},
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)

    assert not decision.accepted
    assert "critical_empty=safety" in decision.reason


def test_gate_fails_closed_when_generated_case_ids_change() -> None:
    gate = CachedWholeOutputBehaviorGate(
        _cases(),
        generator=lambda _model, _cases: {"chat": "hello"},
    )

    with pytest.raises(DynamicBehaviorGateError, match="case IDs"):
        gate.evaluate_model(object(), verify_model_execution=False)


def _metadata_cases():
    return (
        WholeOutputCase(
            "already-saturated",
            "Write a story.",
            "Parent story",
            parent_metadata=WholeOutputMetadata(
                generated_token_count=80,
                generation_limit=80,
                maximum_repeated_ngram_share=0.05,
            ),
        ),
        WholeOutputCase(
            "short-answer",
            "Explain RAM.",
            "Parent explanation",
            parent_metadata=WholeOutputMetadata(
                generated_token_count=32,
                generation_limit=80,
                maximum_repeated_ngram_share=0.10,
            ),
        ),
        WholeOutputCase(
            "another-short-answer",
            "Explain disk storage.",
            "Another parent explanation",
            parent_metadata=WholeOutputMetadata(
                generated_token_count=28,
                generation_limit=80,
                maximum_repeated_ngram_share=0.08,
            ),
        ),
    )


def _metadata_config() -> WholeOutputGateConfig:
    return WholeOutputGateConfig(
        maximum_new_generation_limit_saturations=1,
        maximum_repeated_ngram_share_regressions=0,
        maximum_repeated_ngram_share_delta=0.02,
    )


def test_gate_accepts_parent_parity_for_saturation_and_repetition() -> None:
    gate = CachedWholeOutputBehaviorGate(
        _metadata_cases(),
        generator=lambda _model, _cases: {
            "already-saturated": CandidateWholeOutput(
                "A different coherent story",
                metadata=WholeOutputMetadata(
                    generated_token_count=60,
                    generation_limit=80,
                    maximum_repeated_ngram_share=0.06,
                ),
            ),
            "short-answer": CandidateWholeOutput(
                "A concise explanation",
                metadata=WholeOutputMetadata(
                    generated_token_count=80,
                    generation_limit=80,
                    maximum_repeated_ngram_share=0.11,
                ),
            ),
            "another-short-answer": CandidateWholeOutput(
                "Another concise explanation",
                metadata=WholeOutputMetadata(
                    generated_token_count=36,
                    generation_limit=80,
                    maximum_repeated_ngram_share=0.09,
                ),
            ),
        },
        config=_metadata_config(),
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)
    metrics = dict(decision.metrics)

    assert decision.accepted
    assert metrics["parent_generation_limit_saturation_count"] == 1
    assert metrics["candidate_generation_limit_saturation_count"] == 1
    assert metrics["new_generation_limit_saturation_count"] == 1
    assert metrics["repeated_ngram_regression_count"] == 0
    assert metrics["exact_text_equality_is_diagnostic_only"] is True


def test_gate_rejects_new_saturation_and_repetition_regressions() -> None:
    gate = CachedWholeOutputBehaviorGate(
        _metadata_cases(),
        generator=lambda _model, _cases: {
            "already-saturated": CandidateWholeOutput(
                "A different coherent story",
                metadata=WholeOutputMetadata(
                    generated_token_count=80,
                    generation_limit=80,
                    maximum_repeated_ngram_share=0.05,
                ),
            ),
            "short-answer": CandidateWholeOutput(
                "RAM is fast.",
                metadata=WholeOutputMetadata(
                    generated_token_count=80,
                    generation_limit=80,
                    maximum_repeated_ngram_share=0.11,
                ),
            ),
            "another-short-answer": CandidateWholeOutput(
                "Disk is durable. Disk is durable. Disk is durable.",
                metadata=WholeOutputMetadata(
                    generated_token_count=80,
                    generation_limit=80,
                    maximum_repeated_ngram_share=0.35,
                ),
            ),
        },
        config=_metadata_config(),
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)
    metrics = dict(decision.metrics)

    assert not decision.accepted
    assert (
        "new_generation_limit_saturation=short-answer,another-short-answer"
        in decision.reason
    )
    assert "repeated_ngram_regression=another-short-answer" in decision.reason
    assert metrics["new_generation_limit_saturation_count"] == 2
    assert metrics["repeated_ngram_regression_count"] == 1


def test_configured_metadata_gates_fail_closed_on_candidate_omission() -> None:
    gate = CachedWholeOutputBehaviorGate(
        _metadata_cases(),
        generator=lambda _model, _cases: {
            "already-saturated": "Candidate without metadata",
            "short-answer": "Candidate without metadata",
            "another-short-answer": "Candidate without metadata",
        },
        config=_metadata_config(),
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)

    assert not decision.accepted
    assert (
        "missing_generation_limit_metadata="
        "already-saturated,short-answer,another-short-answer" in decision.reason
    )
    assert (
        "missing_repeated_ngram_metadata="
        "already-saturated,short-answer,another-short-answer" in decision.reason
    )


def test_configured_metadata_gates_reject_generation_limit_mismatch() -> None:
    gate = CachedWholeOutputBehaviorGate(
        _metadata_cases(),
        generator=lambda _model, _cases: {
            case.case_id: CandidateWholeOutput(
                "Candidate",
                metadata=WholeOutputMetadata(
                    generated_token_count=32,
                    generation_limit=64,
                    maximum_repeated_ngram_share=0.10,
                ),
            )
            for case in _metadata_cases()
        },
        config=_metadata_config(),
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)

    assert not decision.accepted
    assert (
        "generation_limit_mismatch="
        "already-saturated,short-answer,another-short-answer" in decision.reason
    )


def test_configured_metadata_gates_require_parent_metadata() -> None:
    with pytest.raises(
        DynamicBehaviorGateError,
        match="complete parent metadata",
    ):
        CachedWholeOutputBehaviorGate(
            _cases(),
            generator=lambda _model, _cases: {},
            config=WholeOutputGateConfig(
                maximum_new_generation_limit_saturations=0,
            ),
        )


def test_repetition_configuration_must_be_complete() -> None:
    with pytest.raises(DynamicBehaviorGateError, match="configured together"):
        WholeOutputGateConfig(
            maximum_repeated_ngram_share_delta=0.05,
        )


@pytest.mark.parametrize("invalid_score", (-0.01, 1.01, float("nan")))
def test_gate_rejects_unbounded_or_nonfinite_task_scores(
    invalid_score,
) -> None:
    gate = CachedWholeOutputBehaviorGate(
        _cases(),
        generator=lambda _model, _cases: {
            "chat": "Hello!",
            "math": "5",
            "parent-blank": "",
        },
        task_scorer=lambda case, _output: (
            invalid_score if case.case_id == "math" else None
        ),
    )

    with pytest.raises(DynamicBehaviorGateError, match="task score"):
        gate.evaluate_model(object(), verify_model_execution=False)


def _manifest_generator(_model, cases):
    return {case.case_id: case.parent_output for case in cases}


def _manifest_scorer_a(_case, output):
    return output == "pass"


def _manifest_scorer_b(_case, output):
    return bool(output.strip()) and output[0:4] == "pass"


def _registered_gate(*, config=None, scorer=_manifest_scorer_a):
    cases = (
        WholeOutputCase(
            "heldout-math",
            "Solve it.",
            "pass",
            category="math",
            split="heldout",
        ),
    )
    return CachedWholeOutputBehaviorGate(
        cases,
        generator=_manifest_generator,
        config=config,
        task_scorer=scorer,
        gate_id="test.behavior.manifest",
        gate_version=7,
        generator_identity=identify_behavior_callable(
            _manifest_generator,
            implementation_id="test.manifest.generator",
            version=3,
        ),
        task_scorer_identity=identify_behavior_callable(
            scorer,
            implementation_id="test.manifest.scorer",
            version=2,
        ),
        baseline_binding={"cache_sha256": "3" * 64},
    )


def test_manifest_binds_thresholds_and_callable_implementation() -> None:
    default_gate = _registered_gate()
    threshold_gate = _registered_gate(
        config=WholeOutputGateConfig(maximum_new_empty_responses=1)
    )
    scorer_gate = _registered_gate(scorer=_manifest_scorer_b)

    default_binding = resolve_behavior_gate_binding(default_gate)
    threshold_binding = resolve_behavior_gate_binding(threshold_gate)
    scorer_binding = resolve_behavior_gate_binding(scorer_gate)

    assert (
        default_binding.baseline_fingerprint == threshold_binding.baseline_fingerprint
    )
    assert default_binding.manifest_fingerprint != (
        threshold_binding.manifest_fingerprint
    )
    assert default_binding.manifest_fingerprint != scorer_binding.manifest_fingerprint
    assert default_binding.manifest["threshold_config"] == (
        default_gate.config.as_dict()
    )
    assert default_binding.manifest["cases"][0]["split"] == "heldout"
    assert default_binding.manifest["cases"][0]["parent_output"] == "pass"
    assert default_binding.manifest["generator"]["registered"] is True
    assert default_binding.manifest["scorer"]["registered"] is True


def test_gate_recomputes_generator_identity_and_rejects_forged_resources() -> None:
    clean_identity = identify_behavior_callable(
        _manifest_generator,
        implementation_id="test.manifest.generator",
        version=3,
    )
    gate = CachedWholeOutputBehaviorGate(
        (WholeOutputCase("case", "Prompt", "pass"),),
        generator=_manifest_generator,
        generator_identity=clean_identity,
    )
    assert resolve_behavior_gate_binding(gate).manifest["generator"] == (
        clean_identity.as_dict()
    )

    forged = replace(clean_identity, resource_fingerprint="f" * 64)
    with pytest.raises(
        DynamicBehaviorGateError,
        match="compiler-observed callable implementation or resources",
    ):
        CachedWholeOutputBehaviorGate(
            (WholeOutputCase("case", "Prompt", "pass"),),
            generator=_manifest_generator,
            generator_identity=forged,
        )

    with pytest.raises(
        DynamicBehaviorGateError,
        match="resource fingerprint differs",
    ):
        identify_behavior_callable(
            _manifest_generator,
            implementation_id="test.forged.generator",
            resource_fingerprint="e" * 64,
        )


def test_final_and_fresh_entrypoints_resolve_the_exact_same_manifest() -> None:
    gate = _registered_gate()
    bindings = (
        resolve_behavior_gate_binding(gate),
        resolve_behavior_gate_binding(gate.evaluate),
        resolve_behavior_gate_binding(gate.evaluate_model),
        resolve_behavior_gate_binding(gate.__call__),
    )

    assert len({value.manifest_fingerprint for value in bindings}) == 1
    assert gate.binding_fields() == {
        "behavior_gate_id": "test.behavior.manifest",
        "behavior_gate_version": 7,
        "behavior_baseline_fingerprint": bindings[0].baseline_fingerprint,
        "behavior_gate_manifest_fingerprint": bindings[0].manifest_fingerprint,
        "behavior_gate_manifest": bindings[0].manifest,
    }


def test_task_scorer_with_zero_parent_coverage_fails_closed() -> None:
    gate = CachedWholeOutputBehaviorGate(
        (
            WholeOutputCase(
                "heldout",
                "Question.",
                "Parent answer.",
                category="facts",
                split="heldout",
            ),
        ),
        generator=_manifest_generator,
        task_scorer=lambda _case, _output: None,
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)

    assert decision.accepted is False
    assert "task_score_parent_coverage_zero" in decision.reason


def test_production_coverage_requires_parent_correct_heldout_categories() -> None:
    cases = (
        WholeOutputCase(
            "heldout-math",
            "Math.",
            "pass",
            category="math",
            split="heldout",
        ),
        WholeOutputCase(
            "calibration-code",
            "Code.",
            "pass",
            category="code",
            split="calibration",
        ),
    )
    gate = CachedWholeOutputBehaviorGate(
        cases,
        generator=_manifest_generator,
        config=WholeOutputGateConfig(
            task_score_coverage=TaskScoreCoveragePolicy.production(
                required_splits=("heldout",),
                required_categories=("math", "code"),
                require_split_category_cross_product=True,
            )
        ),
        task_scorer=_manifest_scorer_a,
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)

    assert decision.accepted is False
    assert (
        "task_score_required_split_category_parent_correct_coverage_zero="
        "heldout:code" in decision.reason
    )


def test_production_coverage_rejects_zero_parent_correct_heldout() -> None:
    gate = CachedWholeOutputBehaviorGate(
        (
            WholeOutputCase(
                "heldout-fact",
                "Fact.",
                "parent-failed",
                category="facts",
                split="heldout",
            ),
        ),
        generator=_manifest_generator,
        config=WholeOutputGateConfig(
            task_score_coverage=TaskScoreCoveragePolicy.production(
                required_splits=("heldout",),
                required_categories=("facts",),
                require_split_category_cross_product=True,
            )
        ),
        task_scorer=_manifest_scorer_a,
    )

    decision = gate.evaluate_model(object(), verify_model_execution=False)

    assert decision.accepted is False
    assert "task_score_parent_correct_coverage_zero" in decision.reason
    assert "task_score_required_split_parent_correct_coverage_zero=heldout" in (
        decision.reason
    )
