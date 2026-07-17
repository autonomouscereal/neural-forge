from __future__ import annotations

import copy

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_intelligence_gate import (
    BoundManualRatings,
    CandidateOutput,
    CapabilitySpec,
    GenerationEvidence,
    IntelligenceGatePolicy,
    Lfm2IntelligenceGateError,
    PureFunctionExample,
    ReliableValidatorSpec,
    bind_manual_ratings,
    build_manual_review_packet,
    evaluate_cached_intelligence,
    load_bound_manual_ratings,
    load_cached_parent_rows,
    load_manual_review_packet,
    seal_cached_parent_rows,
    verify_intelligence_report,
)


PARENT_CACHE_KEY = "a" * 64
PARENT_STATE = "b" * 64
TOKENIZER = "c" * 64


def _generation(
    token_ids,
    *,
    limit=16,
    eos=True,
    fragment=False,
):
    return GenerationEvidence(
        generated_token_count=len(token_ids),
        generation_limit=limit,
        eos_observed=eos,
        fragment_like=fragment,
        token_ids=tuple(token_ids),
    )


def _capability(*groups, minimum_words=1):
    return CapabilitySpec(
        required_concept_groups=tuple((value,) for value in groups),
        minimum_words=minimum_words,
    )


def _row(
    case_id,
    prompt,
    category,
    parent_output,
    generation,
    *,
    validator,
    capability,
    manual=False,
):
    return {
        "case_id": case_id,
        "prompt": prompt,
        "category": category,
        "split": "heldout_manual",
        "parent_output": parent_output,
        "generation": generation.as_dict(),
        "validator": None if validator is None else validator.as_dict(),
        "capability": None if capability is None else capability.as_dict(),
        "manual_review_required": manual,
    }


def _math_validator(expected="42"):
    return ReliableValidatorSpec(
        kind="arithmetic_exact",
        expected_numeric=expected,
    )


def _code_validator():
    return ReliableValidatorSpec(
        kind="python_pure_function",
        function_name="add",
        examples=(
            PureFunctionExample(arguments=(2, 3), expected=5),
            PureFunctionExample(arguments=(-4, 7), expected=3),
        ),
    )


def _instruction_validator():
    return ReliableValidatorSpec(
        kind="instruction_constraints",
        exact_word="blue",
        maximum_words=1,
    )


def _baseline(rows):
    document = seal_cached_parent_rows(
        rows,
        parent_cache_key=PARENT_CACHE_KEY,
        parent_state_fingerprint=PARENT_STATE,
        tokenizer_fingerprint=TOKENIZER,
    )
    baseline = load_cached_parent_rows(
        document,
        expected_document_sha256=canonical_sha256(document),
        expected_parent_cache_key=PARENT_CACHE_KEY,
        expected_parent_state_fingerprint=PARENT_STATE,
        expected_tokenizer_fingerprint=TOKENIZER,
    )
    return document, baseline


def _four_case_baseline():
    rows = [
        _row(
            "math",
            "What is six times seven?",
            "math",
            "The result is 42.",
            _generation([1, 2, 3, 4]),
            validator=_math_validator(),
            capability=_capability("42"),
        ),
        _row(
            "code",
            "Write add(a, b).",
            "code",
            "def add(a, b):\n    return a + b",
            _generation([5, 6, 7, 8, 9]),
            validator=_code_validator(),
            capability=_capability("def add", "return", minimum_words=3),
        ),
        _row(
            "instruction",
            "Reply with only the color of a clear daytime sky.",
            "instruction",
            "blue",
            _generation([10]),
            validator=_instruction_validator(),
            capability=_capability("blue"),
        ),
        _row(
            "open",
            "Why does rain fall?",
            "factual",
            "Water condenses in clouds, and gravity pulls droplets down.",
            _generation([11, 12, 13, 14, 15, 16, 17, 18]),
            validator=None,
            capability=_capability("water", "cloud", "gravity", minimum_words=7),
            manual=True,
        ),
    ]
    return _baseline(rows)


def _good_candidates():
    return (
        CandidateOutput(
            case_id="math",
            output="Six multiplied by seven equals 42.",
            generation=_generation([21, 22, 23, 24, 25, 26]),
            topical_relevance=True,
        ),
        CandidateOutput(
            case_id="code",
            output="```python\ndef add(a, b):\n    return a + b\n```",
            generation=_generation([31, 32, 33, 34, 35, 36]),
            topical_relevance=True,
        ),
        CandidateOutput(
            case_id="instruction",
            output="blue",
            generation=_generation([41]),
            topical_relevance=True,
        ),
        CandidateOutput(
            case_id="open",
            output=(
                "Cloud water forms droplets, then gravity draws the rain "
                "toward the ground."
            ),
            generation=_generation([51, 52, 53, 54, 55, 56, 57, 58]),
            topical_relevance=True,
        ),
    )


def _perfect_manual_ratings(packet):
    return bind_manual_ratings(
        packet,
        [
            {
                "case_id": "open",
                "correctness": 4,
                "coherence": 4,
                "instruction_following": 4,
                "capability_retention": 4,
            }
        ],
        reviewer_id="reviewer-1",
    )


def test_strict_gate_accepts_semantically_different_coherent_candidate() -> None:
    _document, baseline = _four_case_baseline()
    candidates = _good_candidates()

    pending = evaluate_cached_intelligence(baseline, candidates)
    packet = build_manual_review_packet(
        baseline,
        candidates,
        IntelligenceGatePolicy(),
    )
    ratings = _perfect_manual_ratings(packet)
    report = evaluate_cached_intelligence(
        baseline,
        candidates,
        manual_ratings=ratings,
    )

    assert pending.promoted is False
    assert pending.headline_lower_bound is None
    assert "manual_review:pending" in pending.reasons
    assert report.promoted is True
    assert report.headline_lower_bound == pytest.approx(1.0)
    assert report.component_map["task_correctness"].passed is True
    assert report.component_map["code_behavior"].passed is True
    assert report.component_map["arithmetic_exactness"].passed is True
    assert report.component_map["instruction_compliance"].passed is True
    assert report.manual_review["exact_wording_comparison_used"] is False
    assert report.as_dict()["parent_forward_passes"] == 0
    assert report.as_dict()["optimizer_steps"] == 0
    assert report.as_dict()["backward_calls"] == 0


def _ten_case_fixture():
    rows = []
    candidates = []
    for index in range(7):
        expected = str(index + 10)
        case_id = f"math-{index}"
        rows.append(
            _row(
                case_id,
                f"Return {expected}.",
                "math",
                f"The exact result is {expected}.",
                _generation([100 + index, 200 + index, 300 + index]),
                validator=_math_validator(expected),
                capability=_capability(expected),
            )
        )
        candidates.append(
            CandidateOutput(
                case_id=case_id,
                output=f"A different but correct explanation ends at {expected}.",
                generation=_generation(
                    [400 + index, 500 + index, 600 + index, 700 + index]
                ),
                topical_relevance=True,
            )
        )
    rows.extend(
        [
            _row(
                "code",
                "Write add(a, b).",
                "code",
                "def add(a, b):\n    return a + b",
                _generation([10, 11, 12, 13], limit=8),
                validator=_code_validator(),
                capability=_capability("def add", "return", minimum_words=3),
            ),
            _row(
                "instruction",
                "Reply only blue.",
                "instruction",
                "blue",
                _generation([14]),
                validator=_instruction_validator(),
                capability=_capability("blue"),
            ),
            _row(
                "open",
                "Explain rain.",
                "factual",
                "Cloud water becomes droplets that gravity pulls downward.",
                _generation([15, 16, 17, 18, 19, 20]),
                validator=None,
                capability=_capability("cloud", "water", "gravity", minimum_words=6),
                manual=True,
            ),
        ]
    )
    candidates.extend(
        [
            CandidateOutput(
                case_id="code",
                output="add add add add add add add add",
                generation=_generation(
                    [9, 9, 9, 9, 9, 9, 9, 9],
                    limit=8,
                    eos=False,
                ),
                topical_relevance=True,
            ),
            CandidateOutput(
                case_id="instruction",
                output="blue",
                generation=_generation([31]),
                topical_relevance=True,
            ),
            CandidateOutput(
                case_id="open",
                output="Cloud water gravity words without a coherent explanation.",
                generation=_generation([32, 33, 34, 35, 36, 37, 38]),
                topical_relevance=False,
            ),
        ]
    )
    return _baseline(rows), tuple(candidates)


def test_ninety_percent_topical_relevance_cannot_hide_code_eos_or_coherence() -> None:
    (_document, baseline), candidates = _ten_case_fixture()
    packet = build_manual_review_packet(
        baseline,
        candidates,
        IntelligenceGatePolicy(),
    )
    ratings = bind_manual_ratings(
        packet,
        [
            {
                "case_id": "open",
                "correctness": 2,
                "coherence": 0,
                "instruction_following": 3,
                "capability_retention": 2,
            }
        ],
        reviewer_id="reviewer-2",
    )

    report = evaluate_cached_intelligence(
        baseline,
        candidates,
        manual_ratings=ratings,
    )

    assert report.topical_relevance_diagnostic["rate"] == pytest.approx(0.90)
    assert report.topical_relevance_diagnostic["promotion_eligible"] is False
    assert report.component_map["code_syntax"].passed is False
    assert report.component_map["code_behavior"].passed is False
    assert report.component_map["eos_generation_limit"].passed is False
    assert report.component_map["repetition_degeneration"].passed is False
    assert report.manual_review["passed"] is False
    assert report.promoted is False
    assert "code_behavior:threshold_failed" in report.reasons
    assert "eos_generation_limit:threshold_failed" in report.reasons
    assert "repetition_degeneration:threshold_failed" in report.reasons
    assert "manual_review:threshold_failed" in report.reasons


def test_arithmetic_requires_final_exact_value_not_any_number() -> None:
    _document, baseline = _four_case_baseline()
    candidates = list(_good_candidates())
    candidates[0] = CandidateOutput(
        case_id="math",
        output="The problem mentions 42, but my final answer is 43.",
        generation=_generation([61, 62, 63, 64, 65, 66, 67, 68]),
        topical_relevance=True,
    )

    report = evaluate_cached_intelligence(baseline, candidates)

    arithmetic = report.component_map["arithmetic_exactness"]
    assert arithmetic.retention == pytest.approx(0.0)
    assert arithmetic.failure_ids == ("math",)
    assert report.component_map["task_correctness"].passed is False


def test_code_behavior_uses_restricted_ast_interpreter_and_never_exec() -> None:
    _document, baseline = _four_case_baseline()
    candidates = list(_good_candidates())
    candidates[1] = CandidateOutput(
        case_id="code",
        output=(
            "def add(a, b):\n    return __import__('os').system('should-never-run')"
        ),
        generation=_generation([71, 72, 73, 74, 75, 76]),
        topical_relevance=True,
    )

    report = evaluate_cached_intelligence(baseline, candidates)
    code = report.component_map["code_behavior"]

    assert report.component_map["code_syntax"].retention == pytest.approx(1.0)
    assert code.retention == pytest.approx(0.0)
    assert code.details["python_exec_used"] is False
    per_case = report.component_map["task_correctness"].details["per_case"]
    assert per_case["code"]["candidate_detail"]["failure_reason"].startswith(
        "unsafe_or_failed_example"
    )


def test_baseline_report_packet_and_ratings_reject_tampering() -> None:
    document, baseline = _four_case_baseline()
    candidates = _good_candidates()
    policy = IntelligenceGatePolicy()
    packet = build_manual_review_packet(baseline, candidates, policy)
    ratings = _perfect_manual_ratings(packet)
    report = evaluate_cached_intelligence(
        baseline,
        candidates,
        policy=policy,
        manual_ratings=ratings,
    )
    verified = verify_intelligence_report(
        report.as_dict(),
        expected_report_sha256=report.report_sha256,
        expected_baseline_document_sha256=baseline.document_sha256,
        expected_candidate_outputs_sha256=report.candidate_outputs_sha256,
        expected_policy_sha256=policy.fingerprint,
    )
    assert verified["promoted"] is True

    changed_baseline = copy.deepcopy(document)
    changed_baseline["cases"][0]["parent_output"] = "43"
    with pytest.raises(Lfm2IntelligenceGateError, match="fingerprint differs"):
        load_cached_parent_rows(
            changed_baseline,
            expected_document_sha256=canonical_sha256(document),
            expected_parent_cache_key=PARENT_CACHE_KEY,
            expected_parent_state_fingerprint=PARENT_STATE,
            expected_tokenizer_fingerprint=TOKENIZER,
        )

    changed_packet = copy.deepcopy(packet)
    changed_packet["cases"][0]["candidate_response"] = "different"
    with pytest.raises(Lfm2IntelligenceGateError, match="packet"):
        load_manual_review_packet(
            changed_packet,
            expected_packet_sha256=packet["packet_sha256"],
        )

    changed_ratings = ratings.as_dict()
    changed_ratings["ratings"][0]["coherence"] = 0
    with pytest.raises(Lfm2IntelligenceGateError, match="ratings"):
        load_bound_manual_ratings(
            changed_ratings,
            expected_packet_sha256=ratings.packet_sha256,
            expected_ratings_sha256=ratings.ratings_sha256,
        )

    changed_report = report.as_dict()
    changed_report["promoted"] = False
    with pytest.raises(Lfm2IntelligenceGateError, match="report"):
        verify_intelligence_report(
            changed_report,
            expected_report_sha256=report.report_sha256,
            expected_baseline_document_sha256=baseline.document_sha256,
            expected_candidate_outputs_sha256=report.candidate_outputs_sha256,
            expected_policy_sha256=policy.fingerprint,
        )


def test_manual_review_is_packet_bound_and_forbids_exact_wording_criterion() -> None:
    _document, baseline = _four_case_baseline()
    candidates = _good_candidates()
    packet = build_manual_review_packet(
        baseline,
        candidates,
        IntelligenceGatePolicy(),
    )

    assert packet["rubric"]["exact_wording_comparison_permitted"] is False
    assert "Do not use exact wording" in packet["rubric"]["comparison_standard"]
    with pytest.raises(Lfm2IntelligenceGateError, match="exact wording"):
        bind_manual_ratings(
            packet,
            [
                {
                    "case_id": "open",
                    "correctness": 4,
                    "coherence": 4,
                    "instruction_following": 4,
                    "capability_retention": 4,
                }
            ],
            reviewer_id="reviewer-3",
            used_exact_wording_comparison=True,
        )

    other_candidates = list(candidates)
    other_candidates[-1] = CandidateOutput(
        case_id="open",
        output="A different response about water, cloud, and gravity.",
        generation=_generation([81, 82, 83, 84, 85, 86]),
        topical_relevance=True,
    )
    ratings = _perfect_manual_ratings(packet)
    with pytest.raises(Lfm2IntelligenceGateError, match="different review packet"):
        evaluate_cached_intelligence(
            baseline,
            other_candidates,
            manual_ratings=ratings,
        )


def test_typed_finite_and_coverage_checks_fail_closed() -> None:
    with pytest.raises(Lfm2IntelligenceGateError, match="finite"):
        IntelligenceGatePolicy(minimum_headline_lower_bound=float("nan"))
    with pytest.raises(Lfm2IntelligenceGateError, match="token_ids"):
        GenerationEvidence(
            generated_token_count=2,
            generation_limit=4,
            eos_observed=True,
            fragment_like=False,
            token_ids=(1,),
        )
    with pytest.raises(Lfm2IntelligenceGateError, match="manual review"):
        _baseline(
            [
                _row(
                    "invalid",
                    "Open question.",
                    "open",
                    "Open response.",
                    _generation([1, 2]),
                    validator=None,
                    capability=None,
                    manual=False,
                )
            ]
        )


def test_bound_manual_ratings_type_is_immutable_evidence() -> None:
    _document, baseline = _four_case_baseline()
    packet = build_manual_review_packet(
        baseline,
        _good_candidates(),
        IntelligenceGatePolicy(),
    )
    ratings = _perfect_manual_ratings(packet)

    assert isinstance(ratings, BoundManualRatings)
    with pytest.raises(TypeError):
        ratings.rating_map["open"]["coherence"] = 0
