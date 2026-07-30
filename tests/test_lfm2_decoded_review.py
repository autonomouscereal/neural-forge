from __future__ import annotations

import pytest

from neural_forge.lfm2_decoded_review import (
    Lfm2DecodedReviewError,
    response_diagnostics,
    response_termination_evidence,
)


def test_generation_limit_is_censored_not_an_automatic_failure() -> None:
    evidence = response_termination_evidence(
        ended_with_eos=False,
        generation_limit_reached=True,
    )

    assert evidence == {
        "finish_reason": "length",
        "completion_status": "censored",
        "ended_with_eos": False,
        "generation_limit_reached": True,
        "output_cap_hit": True,
        "censored_by_output_cap": True,
        "output_cap_hit_is_automatic_folding_failure": False,
        "loop_observed": None,
        "loop_observation_source": "not-reviewed",
        "runtime_error": None,
        "automatic_folding_quality_decision": None,
    }


def test_eos_loop_and_runtime_error_remain_separate_observations() -> None:
    complete = response_termination_evidence(
        ended_with_eos=True,
        generation_limit_reached=False,
        loop_observed=True,
    )
    failed_runtime = response_termination_evidence(
        ended_with_eos=False,
        generation_limit_reached=False,
        runtime_error={"error_type": "RuntimeError", "message": "boom"},
    )

    assert complete["completion_status"] == "complete"
    assert complete["finish_reason"] == "eos"
    assert complete["loop_observed"] is True
    assert complete["censored_by_output_cap"] is False
    assert failed_runtime["completion_status"] == "runtime-error"
    assert failed_runtime["finish_reason"] == "error"
    assert failed_runtime["runtime_error"]["message"] == "boom"
    assert failed_runtime["automatic_folding_quality_decision"] is None


def test_response_diagnostics_carries_termination_contract() -> None:
    diagnostics = response_diagnostics(
        [3, 3, 3],
        "continued text",
        maximum_new_tokens=3,
        eos_token_ids=frozenset({9}),
    )

    assert diagnostics["generation_limit_reached"] is True
    assert diagnostics["completion_status"] == "censored"
    assert diagnostics["finish_reason"] == "length"
    assert diagnostics["loop_observed"] is None


def test_contradictory_eos_and_cap_is_rejected() -> None:
    with pytest.raises(Lfm2DecodedReviewError, match="both end with EOS"):
        response_termination_evidence(
            ended_with_eos=True,
            generation_limit_reached=True,
        )
