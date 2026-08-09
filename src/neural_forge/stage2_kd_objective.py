"""Stage-2 distillation objectives over the donor vocabulary.

``docs/STAGE2_FIRST_MEASUREMENT_DESIGN_2026-08-08.md`` section 2.4 fixes the
objective for all four arms: token-level **forward KL** against the teacher on
the teacher-generated corpus for the first 10M student-consumed donor tokens,
then a **50/50 mix** of that off-policy term and an **on-policy GKD** term --
the student samples a continuation, the teacher scores the student's own
tokens, and the loss is the generalized Jensen-Shannon divergence at
``beta = 0.5`` (Agarwal et al.).  Section 2.5 fixes the space the loss lives
in: donor vocabulary, undefined physical rows masked out of every loss term and
every sample.

Two properties of this module are load-bearing for the measurement rather than
for the mathematics:

* the schedule is a **pure function of the token counter**, and the on-policy
  step selection is deterministic arithmetic rather than a sampled coin, so the
  fold arm and its control consume the identical objective mix in the identical
  order (design section 4's matched-compute definition); and
* nothing here can score an undefined donor row.  The mask is an object with a
  count, not a convention a caller can forget.

No metric defined here is an acceptance signal.  The only rollout-level
progress signals stage 2 may quote are the free-run token match/lcp series in
``stage2_freerun_probe`` (``docs/CURRENT_FOLDING_GOAL.md``, closed-path
addition 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


STAGE2_KD_OBJECTIVE_SCHEMA = "neural-forge-stage2-kd-objective-v1"

# Design section 2.4 argues forward KL against *unbiased* teacher samples, and
# section 2.3 draws those samples at temperature 1.0.  The design names no
# separate distillation temperature, so the KD temperature is the same 1.0 the
# corpus was drawn at; at 1.0 the standard ``T^2`` gradient rescale is exactly
# the identity, so the constant is carried explicitly instead of being folded
# away, and any later temperature sweep changes one number here.
STAGE2_KD_TEMPERATURE = 1.0
# Design section 2.4: "generalized JSD loss with beta = 0.5, per Agarwal et
# al., GKD".
STAGE2_GKD_BETA = 0.5
# Design section 2.4: on-policy mixing is 0% before 10M student-consumed donor
# tokens and 50% afterwards.
STAGE2_ON_POLICY_SWITCH_TOKENS = 10_000_000
STAGE2_ON_POLICY_FRACTION_BEFORE_SWITCH = 0.0
STAGE2_ON_POLICY_FRACTION_AFTER_SWITCH = 0.5
# Design section 2.4: "sample a 256-token continuation from the student on
# corpus prefixes".
STAGE2_ON_POLICY_CONTINUATION_TOKENS = 256


class Stage2KdObjectiveError(RuntimeError):
    """A stage-2 distillation objective, mask, or schedule request is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Stage2KdObjectiveError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Stage2KdObjectiveError(f"{name} must be a nonnegative integer.")
    return value


def _unit_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage2KdObjectiveError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise Stage2KdObjectiveError(f"{name} must be a finite value in [0, 1].")
    return number


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage2KdObjectiveError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise Stage2KdObjectiveError(f"{name} must be a finite positive value.")
    return number


@dataclass(frozen=True, slots=True)
class Stage2DonorVocabulary:
    """The scored donor row set, separated from the physical row capacity.

    ``docs/CURRENT_FOLDING_GOAL.md``: the donor tokenizer has 64,402 contiguous
    defined IDs inside a 65,536-row physical capacity and "the remaining 1,134
    rows are undefined and must never be scored".  Distillation is the first
    stage-2 surface that could quietly score them -- a teacher readout is
    physical-width -- so the restriction is an object every loss call takes,
    and a width this object does not recognise is a refusal rather than a
    silent broadcast.
    """

    defined_row_count: int
    physical_row_count: int

    def __post_init__(self) -> None:
        defined = _positive_int(self.defined_row_count, "defined_row_count")
        physical = _positive_int(self.physical_row_count, "physical_row_count")
        if defined > physical:
            raise Stage2KdObjectiveError(
                "Defined donor rows cannot exceed the physical row capacity."
            )

    @property
    def undefined_row_count(self) -> int:
        return self.physical_row_count - self.defined_row_count

    def restrict(self, logits: Tensor, *, name: str = "logits") -> Tensor:
        """Return the defined-row view of one logit tensor, or refuse."""

        if not isinstance(logits, Tensor) or logits.ndim < 1:
            raise Stage2KdObjectiveError(f"{name} must be a tensor with rows.")
        width = int(logits.shape[-1])
        if width == self.defined_row_count:
            return logits
        if width == self.physical_row_count:
            return logits[..., : self.defined_row_count]
        raise Stage2KdObjectiveError(
            f"{name} has {width} rows, which is neither the defined donor "
            f"count {self.defined_row_count} nor the physical capacity "
            f"{self.physical_row_count}."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "defined_row_count": self.defined_row_count,
            "physical_row_count": self.physical_row_count,
            "undefined_row_count": self.undefined_row_count,
            "undefined_rows_scored": False,
        }


def _validated_logit_pair(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    vocabulary: Stage2DonorVocabulary,
) -> tuple[Tensor, Tensor]:
    if not isinstance(vocabulary, Stage2DonorVocabulary):
        raise Stage2KdObjectiveError(
            "vocabulary must be a Stage2DonorVocabulary."
        )
    student = vocabulary.restrict(student_logits, name="student_logits")
    teacher = vocabulary.restrict(teacher_logits, name="teacher_logits")
    if student.shape != teacher.shape:
        raise Stage2KdObjectiveError(
            "Student and teacher logits must cover the same positions."
        )
    if student.numel() == 0:
        raise Stage2KdObjectiveError("Distillation needs at least one position.")
    if not torch.is_floating_point(student) or not torch.is_floating_point(
        teacher
    ):
        raise Stage2KdObjectiveError("Distillation logits must be floating point.")
    return student, teacher


def forward_kl_logit_distillation(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    vocabulary: Stage2DonorVocabulary,
    temperature: float = STAGE2_KD_TEMPERATURE,
) -> Tensor:
    """Token-level ``KL(teacher || student)`` over the defined donor rows.

    The teacher distribution is the target, so the teacher side is detached:
    the frozen parent contributes no gradient (design section 2.1 pins the
    teacher checkpoint; nothing trains it).
    """

    student, teacher = _validated_logit_pair(
        student_logits,
        teacher_logits,
        vocabulary=vocabulary,
    )
    scale = _positive_float(temperature, "temperature")
    student_log_probabilities = torch.log_softmax(
        student.float() / scale,
        dim=-1,
    )
    with torch.no_grad():
        teacher_log_probabilities = torch.log_softmax(
            teacher.detach().float() / scale,
            dim=-1,
        )
        teacher_probabilities = teacher_log_probabilities.exp()
    per_position = (
        teacher_probabilities
        * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1)
    return per_position.mean() * (scale * scale)


def generalized_jensen_shannon_divergence(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    vocabulary: Stage2DonorVocabulary,
    beta: float = STAGE2_GKD_BETA,
    temperature: float = STAGE2_KD_TEMPERATURE,
) -> Tensor:
    """The GKD divergence with teacher ``P``, student ``Q``, mixture ``M``.

    ``D(beta) = beta * KL(P || M) + (1 - beta) * KL(Q || M)`` with
    ``M = beta * P + (1 - beta) * Q``, exactly the form in Agarwal et al.  Note
    what this form does *not* do, because implementations differ and a wrong
    assumption here would silently change the on-policy objective: the raw
    expression vanishes at **both** endpoints (``beta = 1`` makes ``M = P``,
    ``beta = 0`` makes ``M = Q``), so it does not "reduce to" forward or
    reverse KL without a ``1/beta`` normalization.  The design pins
    ``beta = 0.5``, where it is the symmetric Jensen-Shannon divergence and is
    bounded above by ``ln 2``; that is the value the trainer uses and the value
    the tests check against an independent computation.
    """

    student, teacher = _validated_logit_pair(
        student_logits,
        teacher_logits,
        vocabulary=vocabulary,
    )
    weight = _unit_float(beta, "beta")
    scale = _positive_float(temperature, "temperature")
    student_log_probabilities = torch.log_softmax(
        student.float() / scale,
        dim=-1,
    )
    with torch.no_grad():
        teacher_log_probabilities = torch.log_softmax(
            teacher.detach().float() / scale,
            dim=-1,
        )
    if weight == 1.0:
        mixture_log_probabilities = teacher_log_probabilities
    elif weight == 0.0:
        mixture_log_probabilities = student_log_probabilities
    else:
        mixture_log_probabilities = torch.logaddexp(
            teacher_log_probabilities + math.log(weight),
            student_log_probabilities + math.log1p(-weight),
        )
    divergence = student.new_zeros(())
    if weight > 0.0:
        teacher_probabilities = teacher_log_probabilities.exp()
        divergence = divergence + weight * (
            teacher_probabilities
            * (teacher_log_probabilities - mixture_log_probabilities)
        ).sum(dim=-1).mean()
    if weight < 1.0:
        student_probabilities = student_log_probabilities.exp()
        divergence = divergence + (1.0 - weight) * (
            student_probabilities
            * (student_log_probabilities - mixture_log_probabilities)
        ).sum(dim=-1).mean()
    return divergence * (scale * scale)


def diagnostic_held_out_negative_log_likelihood(
    student_logits: Tensor,
    donor_target_ids: Tensor,
    *,
    vocabulary: Stage2DonorVocabulary,
) -> Tensor:
    """Held-out NLL.  Design section 3.2: **diagnostic only, never gating.**

    It is exported so the receipt can carry the series the design asks for; the
    receipt marks the whole series ``diagnostic_only`` and no acceptance path
    reads it.
    """

    student = vocabulary.restrict(student_logits, name="student_logits")
    if (
        not isinstance(donor_target_ids, Tensor)
        or donor_target_ids.dtype not in {torch.int32, torch.int64}
        or tuple(donor_target_ids.shape) != tuple(student.shape[:-1])
    ):
        raise Stage2KdObjectiveError(
            "donor_target_ids must be one integer id per scored position."
        )
    if bool((donor_target_ids < 0).any()) or bool(
        (donor_target_ids >= vocabulary.defined_row_count).any()
    ):
        raise Stage2KdObjectiveError(
            "Held-out targets must stay inside the defined donor rows."
        )
    log_probabilities = torch.log_softmax(student.float(), dim=-1)
    gathered = log_probabilities.gather(
        -1,
        donor_target_ids.to(torch.int64).unsqueeze(-1),
    ).squeeze(-1)
    return -gathered.mean()


@dataclass(frozen=True, slots=True)
class Stage2ObjectiveSchedule:
    """The design's objective mix as a pure function of the token counter.

    The on-policy *fraction* is a step property, not a per-token property, so a
    fraction has to be turned into a concrete sequence of steps.  A sampled
    coin would make the arm and its control diverge in objective mix even
    under one seed, which section 4 forbids ("identical objective mix and
    schedule").  The rule below spends exactly ``fraction`` of the post-switch
    steps on policy, in a fixed order that depends only on how many steps have
    been taken since the switch.
    """

    switch_tokens: int = STAGE2_ON_POLICY_SWITCH_TOKENS
    fraction_before_switch: float = STAGE2_ON_POLICY_FRACTION_BEFORE_SWITCH
    fraction_after_switch: float = STAGE2_ON_POLICY_FRACTION_AFTER_SWITCH
    continuation_tokens: int = STAGE2_ON_POLICY_CONTINUATION_TOKENS
    beta: float = STAGE2_GKD_BETA
    temperature: float = STAGE2_KD_TEMPERATURE

    def __post_init__(self) -> None:
        _nonnegative_int(self.switch_tokens, "switch_tokens")
        _unit_float(self.fraction_before_switch, "fraction_before_switch")
        _unit_float(self.fraction_after_switch, "fraction_after_switch")
        _positive_int(self.continuation_tokens, "continuation_tokens")
        _unit_float(self.beta, "beta")
        _positive_float(self.temperature, "temperature")

    def fraction_at(self, tokens_consumed: int) -> float:
        """The on-policy fraction that applies to a step starting here."""

        consumed = _nonnegative_int(tokens_consumed, "tokens_consumed")
        if consumed < self.switch_tokens:
            return self.fraction_before_switch
        return self.fraction_after_switch

    def phase_at(self, tokens_consumed: int) -> str:
        consumed = _nonnegative_int(tokens_consumed, "tokens_consumed")
        return (
            "off-policy-forward-kl"
            if consumed < self.switch_tokens
            else "mixed-forward-kl-and-on-policy-gkd"
        )

    def step_is_on_policy(
        self,
        *,
        tokens_consumed: int,
        steps_in_phase: int,
    ) -> bool:
        """Decide one step's branch from counters only -- no RNG, no state.

        ``steps_in_phase`` counts optimizer steps already started in the
        current phase.  The step is on policy exactly when the running exact
        quota ``floor((k + 1) * f)`` crosses an integer, which spends the
        fraction with no drift and no dependence on wall-clock or device.
        """

        fraction = self.fraction_at(tokens_consumed)
        taken = _nonnegative_int(steps_in_phase, "steps_in_phase")
        if fraction <= 0.0:
            return False
        if fraction >= 1.0:
            return True
        return math.floor((taken + 1) * fraction) > math.floor(taken * fraction)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE2_KD_OBJECTIVE_SCHEMA,
            "off_policy_objective": "token-level-forward-kl-teacher-to-student",
            "on_policy_objective": "generalized-jsd-gkd-student-samples",
            "on_policy_switch_tokens": self.switch_tokens,
            "on_policy_fraction_before_switch": self.fraction_before_switch,
            "on_policy_fraction_after_switch": self.fraction_after_switch,
            "on_policy_continuation_tokens": self.continuation_tokens,
            "on_policy_step_selection": (
                "exact-quota-floor-crossing-deterministic-no-rng"
            ),
            "gkd_beta": self.beta,
            "kd_temperature": self.temperature,
            "teacher_scores_student_sampled_tokens": True,
            "teacher_receives_gradient": False,
            "sequence_level_hard_target_kd_used": False,
            "reward_shaping_used": False,
        }


__all__ = [
    "STAGE2_GKD_BETA",
    "STAGE2_KD_OBJECTIVE_SCHEMA",
    "STAGE2_KD_TEMPERATURE",
    "STAGE2_ON_POLICY_CONTINUATION_TOKENS",
    "STAGE2_ON_POLICY_FRACTION_AFTER_SWITCH",
    "STAGE2_ON_POLICY_FRACTION_BEFORE_SWITCH",
    "STAGE2_ON_POLICY_SWITCH_TOKENS",
    "Stage2DonorVocabulary",
    "Stage2KdObjectiveError",
    "Stage2ObjectiveSchedule",
    "diagnostic_held_out_negative_log_likelihood",
    "forward_kl_logit_distillation",
    "generalized_jensen_shannon_divergence",
]
