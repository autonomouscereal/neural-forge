from __future__ import annotations

import math

import pytest
import torch

from neural_forge.stage2_kd_objective import (
    STAGE2_GKD_BETA,
    STAGE2_KD_TEMPERATURE,
    STAGE2_ON_POLICY_CONTINUATION_TOKENS,
    STAGE2_ON_POLICY_FRACTION_AFTER_SWITCH,
    STAGE2_ON_POLICY_SWITCH_TOKENS,
    Stage2DonorVocabulary,
    Stage2KdObjectiveError,
    Stage2ObjectiveSchedule,
    diagnostic_held_out_negative_log_likelihood,
    forward_kl_logit_distillation,
    generalized_jensen_shannon_divergence,
)


VOCABULARY = Stage2DonorVocabulary(defined_row_count=6, physical_row_count=8)


def _logit_pair(seed: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(2, 5, 6, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 5, 6, generator=generator)
    return student, teacher


def test_design_constants_match_the_preregistered_objective() -> None:
    """Section 2.4's numbers are constants, not defaults someone can drift."""

    assert STAGE2_KD_TEMPERATURE == 1.0
    assert STAGE2_GKD_BETA == 0.5
    assert STAGE2_ON_POLICY_SWITCH_TOKENS == 10_000_000
    assert STAGE2_ON_POLICY_FRACTION_AFTER_SWITCH == 0.5
    assert STAGE2_ON_POLICY_CONTINUATION_TOKENS == 256


def test_forward_kl_is_zero_only_when_the_distributions_agree() -> None:
    student, teacher = _logit_pair()
    identical = forward_kl_logit_distillation(
        teacher.clone().requires_grad_(True),
        teacher,
        vocabulary=VOCABULARY,
    )
    different = forward_kl_logit_distillation(
        student,
        teacher,
        vocabulary=VOCABULARY,
    )
    assert float(identical.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(different.detach()) > 0.0


def test_forward_kl_matches_an_independent_reference() -> None:
    student, teacher = _logit_pair(seed=11)
    reference = torch.nn.functional.kl_div(
        torch.log_softmax(student, dim=-1),
        torch.log_softmax(teacher, dim=-1),
        log_target=True,
        reduction="none",
    ).sum(dim=-1).mean()
    observed = forward_kl_logit_distillation(
        student,
        teacher,
        vocabulary=VOCABULARY,
    )
    assert float(observed.detach()) == pytest.approx(
        float(reference.detach()),
        rel=1e-6,
    )


def test_generalized_jsd_matches_an_independent_symmetric_computation() -> None:
    """The on-policy objective is checked against plain probability algebra."""

    student, teacher = _logit_pair(seed=5)
    student_probabilities = torch.softmax(student.detach(), dim=-1)
    teacher_probabilities = torch.softmax(teacher, dim=-1)
    mixture = 0.5 * teacher_probabilities + 0.5 * student_probabilities
    reference = 0.5 * (
        (teacher_probabilities * (teacher_probabilities / mixture).log())
        .sum(dim=-1)
        .mean()
        + (student_probabilities * (student_probabilities / mixture).log())
        .sum(dim=-1)
        .mean()
    )
    observed = generalized_jensen_shannon_divergence(
        student,
        teacher,
        vocabulary=VOCABULARY,
    )
    assert float(observed.detach()) == pytest.approx(float(reference), rel=1e-6)
    assert 0.0 < float(observed.detach()) < math.log(2.0)


def test_generalized_jsd_vanishes_at_both_raw_endpoints() -> None:
    """Pinned so nobody "fixes" beta=1 into a forward KL it never was."""

    student, teacher = _logit_pair(seed=5)
    for beta in (0.0, 1.0):
        assert float(
            generalized_jensen_shannon_divergence(
                student,
                teacher,
                vocabulary=VOCABULARY,
                beta=beta,
            ).detach()
        ) == pytest.approx(0.0, abs=1e-6)
    forward = forward_kl_logit_distillation(
        student,
        teacher,
        vocabulary=VOCABULARY,
    )
    half = generalized_jensen_shannon_divergence(
        student,
        teacher,
        vocabulary=VOCABULARY,
    )
    assert 0.0 < float(half.detach()) < float(forward.detach())


def test_generalized_jsd_is_symmetric_at_the_preregistered_beta() -> None:
    student, teacher = _logit_pair(seed=7)
    forward = generalized_jensen_shannon_divergence(
        student,
        teacher,
        vocabulary=VOCABULARY,
    )
    swapped = generalized_jensen_shannon_divergence(
        teacher.clone().requires_grad_(True),
        student.detach(),
        vocabulary=VOCABULARY,
    )
    assert float(forward.detach()) == pytest.approx(float(swapped.detach()), rel=1e-6)


def test_objectives_carry_gradient_to_the_student_only() -> None:
    student, teacher = _logit_pair(seed=13)
    teacher = teacher.clone().requires_grad_(True)
    generalized_jensen_shannon_divergence(
        student,
        teacher,
        vocabulary=VOCABULARY,
    ).backward()
    assert student.grad is not None
    assert torch.any(student.grad != 0.0)
    assert teacher.grad is None


def test_undefined_donor_rows_are_sliced_away_and_never_scored() -> None:
    """A physical-width teacher readout is restricted, not broadcast."""

    generator = torch.Generator().manual_seed(2)
    student = torch.randn(3, 6, generator=generator, requires_grad=True)
    physical = torch.randn(3, 8, generator=generator)
    physical[..., 6:] = 50.0  # undefined rows would dominate if they were scored
    restricted = forward_kl_logit_distillation(
        student,
        physical,
        vocabulary=VOCABULARY,
    )
    sliced = forward_kl_logit_distillation(
        student,
        physical[..., :6],
        vocabulary=VOCABULARY,
    )
    assert float(restricted.detach()) == pytest.approx(float(sliced.detach()), rel=1e-9)


def test_a_width_the_vocabulary_does_not_recognise_is_refused() -> None:
    generator = torch.Generator().manual_seed(4)
    with pytest.raises(Stage2KdObjectiveError, match="which is neither"):
        forward_kl_logit_distillation(
            torch.randn(3, 7, generator=generator),
            torch.randn(3, 7, generator=generator),
            vocabulary=VOCABULARY,
        )


def test_held_out_nll_refuses_targets_outside_the_defined_rows() -> None:
    generator = torch.Generator().manual_seed(6)
    logits = torch.randn(2, 4, 6, generator=generator)
    with pytest.raises(Stage2KdObjectiveError, match="defined donor rows"):
        diagnostic_held_out_negative_log_likelihood(
            logits,
            torch.full((2, 4), 6, dtype=torch.int64),
            vocabulary=VOCABULARY,
        )


def test_schedule_switches_exactly_at_the_token_threshold() -> None:
    schedule = Stage2ObjectiveSchedule()
    assert schedule.fraction_at(STAGE2_ON_POLICY_SWITCH_TOKENS - 1) == 0.0
    assert schedule.fraction_at(STAGE2_ON_POLICY_SWITCH_TOKENS) == 0.5
    assert schedule.phase_at(0) == "off-policy-forward-kl"
    assert (
        schedule.phase_at(STAGE2_ON_POLICY_SWITCH_TOKENS)
        == "mixed-forward-kl-and-on-policy-gkd"
    )
    assert not schedule.step_is_on_policy(tokens_consumed=0, steps_in_phase=17)


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_on_policy_quota_is_exact_and_rng_free(fraction: float) -> None:
    """The mix is arithmetic, so an arm and its control cannot diverge."""

    schedule = Stage2ObjectiveSchedule(
        switch_tokens=0,
        fraction_after_switch=fraction,
    )
    steps = 400
    selected = [
        schedule.step_is_on_policy(tokens_consumed=1_000, steps_in_phase=index)
        for index in range(steps)
    ]
    assert sum(selected) == math.floor(steps * fraction)
    repeated = [
        schedule.step_is_on_policy(tokens_consumed=1_000, steps_in_phase=index)
        for index in range(steps)
    ]
    assert selected == repeated


def test_schedule_receipt_names_what_it_rejected() -> None:
    payload = Stage2ObjectiveSchedule().as_dict()
    assert payload["gkd_beta"] == 0.5
    assert payload["kd_temperature"] == 1.0
    assert payload["teacher_receives_gradient"] is False
    assert payload["sequence_level_hard_target_kd_used"] is False
    assert payload["reward_shaping_used"] is False
    assert "no-rng" in payload["on_policy_step_selection"]
