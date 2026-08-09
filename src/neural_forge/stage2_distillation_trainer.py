"""The stage-2 distillation trainer: a plain torch loop with sealed receipts.

Design section 5.1 chose "the smallest honest implementation": a plain PyTorch
loop, new stage-2 code, no HF Trainer / Lightning / accelerate, no database,
receipts as content-addressed JSON exactly like stage 1.  This module is that
loop.  Everything it does that is *not* obvious loop mechanics comes from a
numbered design decision:

* **objective** (section 2.4) -- forward KL on the teacher-generated corpus
  until 10M student-consumed donor tokens, then a deterministic 50/50 mix with
  on-policy GKD in which the student samples a 256-token continuation and the
  teacher scores the student's own tokens;
* **space** (section 2.5) -- every loss term lives in donor vocabulary through
  the routed readout, undefined donor rows never scored;
* **ladder** (section 3.1) -- 10M/100M/1B token rungs as checkpoints inside one
  run, at 16 sequences x 1,024 tokens per optimizer step;
* **optimizer** (section 3.1) -- AdamW (0.9, 0.95), weight decay 0.1,
  lr 3e-4 cosine to 3e-5, 100-step warmup, grad clip 1.0, bf16 autocast over
  fp32 master weights;
* **curve signal** (section 3.2) -- free-run token match/lcp only, logged every
  1M tokens; held-out NLL carried as diagnostic-only; and
* **matched compute** (section 4) -- token counts, objective mix, schedule,
  hyperparameters and off-policy shard order are properties of the *config*,
  so an arm and its control differ only in initialization.

Two metric classes are absent on purpose and their absence is pinned by a test:
terminal reconstruction (disqualified, ceiling report section 2.3) and
teacher-forced top-1 (anti-correlated with free-run, section 4.3).  Nothing in
this module computes, logs, or gates on either.
"""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
from torch import Tensor

from .stage2_carrier_student import (
    Stage2RoutedStudent,
    assert_route_surface_matches_frozen_runtime,
)
from .stage2_checkpoint_layout import (
    Stage2RouteProvenance,
    write_stage2_rung_checkpoint,
)
from .stage2_freerun_probe import (
    STAGE2_FREERUN_PROBE_INTERVAL_TOKENS,
    STAGE2_FREERUN_PROBE_NEW_TOKENS,
    Stage2FreeRunProbePrompts,
    run_stage2_freerun_probe,
)
from .stage2_kd_objective import (
    Stage2DonorVocabulary,
    Stage2ObjectiveSchedule,
    diagnostic_held_out_negative_log_likelihood,
    forward_kl_logit_distillation,
    generalized_jensen_shannon_divergence,
)
from .stage2_training_receipt import (
    STAGE2_RUNG_TOKENS,
    Stage2CorpusConsumption,
    Stage2RungRecord,
    Stage2ShardConsumption,
    Stage2TrainingReceiptBuilder,
    Stage2TrainingReceiptError,
    optimizer_state_digest,
)


STAGE2_TRAINER_SCHEMA = "neural-forge-stage2-distillation-trainer-v1"
# Design section 3.1: 16 sequences x 1,024 donor tokens = 16,384 tokens/step,
# giving rung checkpoints at steps 610 / 6,104 / 61,036.
STAGE2_BATCH_SEQUENCES = 16
STAGE2_SEQUENCE_LENGTH = 1_024
STAGE2_LEARNING_RATE = 3e-4
STAGE2_MINIMUM_LEARNING_RATE = 3e-5
STAGE2_WARMUP_STEPS = 100
STAGE2_ADAM_BETAS = (0.9, 0.95)
STAGE2_WEIGHT_DECAY = 0.1
STAGE2_GRADIENT_CLIP_NORM = 1.0
# Design section 3.1 states the rung steps as 610 / 6,104 / 61,036.  The token
# budgets are the authority ("the token budgets are exact regardless", section
# 5.2), and 10,000,000 is not a multiple of 16,384: 610 steps consume 9,994,240
# tokens, one step short of the rung.  :meth:`Stage2TrainerConfig.steps_for_tokens`
# therefore rounds *up*, which reproduces the design's 6,104 and 61,036 exactly
# and moves the 10M rung by one step.  The stated tuple is kept so the
# difference is visible in a receipt instead of buried in arithmetic.
STAGE2_DESIGN_STATED_RUNG_STEPS = (610, 6_104, 61_036)


class Stage2TrainerError(RuntimeError):
    """A stage-2 training configuration, batch, or accounting is invalid."""


@dataclass(frozen=True, slots=True)
class Stage2CorpusBatch:
    """One batch plus the shards it came from, in read order.

    Lane 2 owns corpus generation and the loader (design section 10); lane 1
    consumes this shape read-only.  The shard records travel with the tokens so
    the receipt can name its exact data without a second pass over the data
    pipeline.
    """

    donor_ids: Tensor
    shards: tuple[Mapping[str, Any], ...]


class Stage2CorpusSource(Protocol):
    """The read-only corpus contract lane 1 depends on."""

    manifest_path: str
    manifest_sha256: str
    teacher_checkpoint_sha256: str
    shard_order_seed: int

    def next_batch(
        self,
        *,
        sequences: int,
        sequence_length: int,
    ) -> Stage2CorpusBatch: ...


@dataclass(frozen=True, slots=True)
class Stage2CheckpointSettings:
    """Everything the rung writer needs that the loop itself does not own."""

    root: Path
    base: Any
    provenance: Stage2RouteProvenance
    structural_route_tensors: Mapping[str, Any]
    initialization_child_directory: Path
    semantic_source_tokenizer_sha256: str
    lexicon_package_sha256: str
    save_optimizer: Callable[[Any, Path], None] | None = None
    keep_optimizer_state: bool = True
    # The design-declared initialization artifact for this arm (section 1):
    # the A9/A11 saved child state for S2-A, the a0028 published artifact tree
    # for S2-B, and ``None`` for the two random-init controls, which have a
    # recorded seed and no source artifact.  It is sealed into every rung
    # checkpoint so the stage-2 admission list has something to key on.
    init_artifact_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Stage2ProbeSettings:
    """The free-run probe surface and how often to spend it."""

    prompts: Stage2FreeRunProbePrompts
    tokenize: Callable[[str], Sequence[int]]
    teacher_donor_logits: Callable[[Tensor], Tensor]
    maximum_new_tokens: int = STAGE2_FREERUN_PROBE_NEW_TOKENS
    interval_tokens: int = STAGE2_FREERUN_PROBE_INTERVAL_TOKENS
    end_of_sequence_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Stage2TrainerConfig:
    """The pre-registered knobs.  Identical for every arm but ``arm_id``."""

    arm_id: str
    seed: int
    rung_tokens: tuple[int, ...] = STAGE2_RUNG_TOKENS
    batch_sequences: int = STAGE2_BATCH_SEQUENCES
    micro_batch_sequences: int = 1
    sequence_length: int = STAGE2_SEQUENCE_LENGTH
    learning_rate: float = STAGE2_LEARNING_RATE
    minimum_learning_rate: float = STAGE2_MINIMUM_LEARNING_RATE
    warmup_steps: int = STAGE2_WARMUP_STEPS
    adam_betas: tuple[float, float] = STAGE2_ADAM_BETAS
    weight_decay: float = STAGE2_WEIGHT_DECAY
    gradient_clip_norm: float = STAGE2_GRADIENT_CLIP_NORM
    enable_autocast: bool | None = None
    autocast_dtype: torch.dtype = torch.bfloat16
    loss_log_interval_steps: int = 10
    safety_checkpoint_seconds: float = 1_800.0
    safety_checkpoints_kept: int = 2
    # H4 of NOTES-S2B.  On-policy sampling used to re-run the whole student
    # forward for every generated token: 256 full forwards over a growing
    # context per on-policy step.  The cached path advances one token at a time
    # through the child body's own cache -- the mechanism the deployed routed
    # decode already uses (``hf_lfm2_semantic_step_model_runtime``).  The
    # uncached path stays as the correctness reference, and the equivalence
    # test drives both at one seed.
    on_policy_sampling_uses_cache: bool = True

    def __post_init__(self) -> None:
        if not self.arm_id.strip():
            raise Stage2TrainerError("arm_id must be nonempty.")
        if not self.rung_tokens or list(self.rung_tokens) != sorted(
            self.rung_tokens
        ):
            raise Stage2TrainerError("rung_tokens must be ascending and nonempty.")
        if self.batch_sequences <= 0 or self.sequence_length <= 0:
            raise Stage2TrainerError("Batch and sequence sizes must be positive.")
        if (
            self.micro_batch_sequences <= 0
            or self.batch_sequences % self.micro_batch_sequences != 0
        ):
            raise Stage2TrainerError(
                "micro_batch_sequences must divide batch_sequences exactly."
            )
        if self.warmup_steps < 0 or self.loss_log_interval_steps <= 0:
            raise Stage2TrainerError("Schedule intervals are out of range.")

    @property
    def tokens_per_step(self) -> int:
        return self.batch_sequences * self.sequence_length

    @property
    def accumulation_steps(self) -> int:
        return self.batch_sequences // self.micro_batch_sequences

    def steps_for_tokens(self, tokens: int) -> int:
        return math.ceil(int(tokens) / self.tokens_per_step)

    @property
    def total_steps(self) -> int:
        return self.steps_for_tokens(self.rung_tokens[-1])

    def learning_rate_at(self, step_index: int) -> float:
        """Linear warmup then cosine decay to the pre-registered floor."""

        if step_index < self.warmup_steps:
            return self.learning_rate * (step_index + 1) / max(self.warmup_steps, 1)
        span = max(self.total_steps - self.warmup_steps, 1)
        progress = min(max((step_index - self.warmup_steps) / span, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_learning_rate + (
            self.learning_rate - self.minimum_learning_rate
        ) * cosine

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE2_TRAINER_SCHEMA,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "rung_tokens": list(self.rung_tokens),
            "rung_steps": [self.steps_for_tokens(value) for value in self.rung_tokens],
            "design_stated_rung_steps": list(STAGE2_DESIGN_STATED_RUNG_STEPS),
            "rung_step_rounding": "ceil-so-a-rung-never-underspends-its-budget",
            "batch_sequences": self.batch_sequences,
            "micro_batch_sequences": self.micro_batch_sequences,
            "gradient_accumulation_steps": self.accumulation_steps,
            "sequence_length": self.sequence_length,
            "tokens_per_optimizer_step": self.tokens_per_step,
            "total_steps": self.total_steps,
            "loss_log_interval_steps": self.loss_log_interval_steps,
            "safety_checkpoint_seconds": self.safety_checkpoint_seconds,
            "safety_checkpoints_kept": self.safety_checkpoints_kept,
        }

    def optimizer_configuration(self) -> dict[str, Any]:
        return {
            "optimizer": "AdamW",
            "learning_rate": self.learning_rate,
            "minimum_learning_rate": self.minimum_learning_rate,
            "schedule": "linear-warmup-then-cosine",
            "warmup_steps": self.warmup_steps,
            "betas": list(self.adam_betas),
            "weight_decay": self.weight_decay,
            "weight_decay_applies_to": "all-parameters-one-group",
            "gradient_clip_norm": self.gradient_clip_norm,
            "master_weight_dtype": "float32",
            "autocast_dtype": str(self.autocast_dtype).removeprefix("torch."),
        }


class Stage2DistillationTrainer:
    """One arm, one run, one receipt."""

    def __init__(
        self,
        *,
        config: Stage2TrainerConfig,
        student: Stage2RoutedStudent,
        teacher_donor_logits: Callable[[Tensor], Tensor],
        teacher_checkpoint_sha256: str,
        corpus: Stage2CorpusSource,
        donor_vocabulary: Stage2DonorVocabulary,
        objective: Stage2ObjectiveSchedule | None = None,
        preregistration: Mapping[str, Any],
        device: Any = None,
        checkpoints: Stage2CheckpointSettings | None = None,
        probe: Stage2ProbeSettings | None = None,
        held_out_batch: Callable[[], tuple[Tensor, Tensor]] | None = None,
        device_inventory: Sequence[Mapping[str, Any]] | None = None,
        route_surface_equivalence: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(student, Stage2RoutedStudent):
            raise Stage2TrainerError("student must be a Stage2RoutedStudent.")
        if student.initialization is None:
            raise Stage2TrainerError(
                "The student must carry its initialization provenance."
            )
        self.config = config
        self.student = student
        self.teacher_donor_logits = teacher_donor_logits
        self.teacher_checkpoint_sha256 = teacher_checkpoint_sha256
        self.corpus = corpus
        self.vocabulary = donor_vocabulary
        self.objective = objective or Stage2ObjectiveSchedule()
        self.preregistration = dict(preregistration)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.checkpoints = checkpoints
        self.probe = probe
        self.held_out_batch = held_out_batch
        self.device_inventory = tuple(
            dict(row) for row in (device_inventory or ())
        )
        self.route_surface_equivalence = dict(route_surface_equivalence or {})
        self._autocast_enabled = (
            self.device.type == "cuda"
            if config.enable_autocast is None
            else bool(config.enable_autocast)
        )
        # Deterministic seeding: one recorded seed drives parameter-free RNG
        # for the whole run, and on-policy sampling draws from an explicit
        # generator keyed by (seed, step) so a replicate reproduces the exact
        # sampled continuations.
        torch.manual_seed(int(config.seed))
        self._sample_generator = torch.Generator(device="cpu").manual_seed(
            int(config.seed) ^ 0x5F2B
        )
        self.optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.student.parameters()
                if parameter.requires_grad
            ],
            lr=config.learning_rate,
            betas=tuple(config.adam_betas),
            weight_decay=config.weight_decay,
        )
        self.tokens_consumed = 0
        self.optimizer_steps = 0
        self.backward_calls = 0
        self.micro_batch_forward_calls = 0
        self.on_policy_sample_forward_calls = 0
        self.on_policy_steps = 0
        self.off_policy_steps = 0
        self.steps_in_phase = 0
        self.wall_clock_seconds = 0.0
        self._phase = self.objective.phase_at(0)
        self._shards: dict[str, dict[str, Any]] = {}
        self._loss_curve: list[dict[str, Any]] = []
        self._probe_series: list[dict[str, Any]] = []
        self._nll_series: list[dict[str, Any]] = []
        self._teacher_rollout_cache: dict[str, tuple[int, ...]] = {}
        self._next_probe_tokens = 0
        self._phase_switch_step: int | None = None
        self._safety_checkpoints: list[Path] = []
        self._last_safety_checkpoint = time.monotonic()
        self._loss_curve_sealed = 0
        self._nll_series_sealed = 0

    # -- accounting --------------------------------------------------------

    def _record_shards(self, shards: Sequence[Mapping[str, Any]]) -> None:
        for row in shards:
            relative_path = str(row["relative_path"])
            tokens = int(row["tokens_consumed"])
            existing = self._shards.get(relative_path)
            if existing is None:
                self._shards[relative_path] = {
                    "order_index": len(self._shards),
                    "relative_path": relative_path,
                    "sha256": str(row["sha256"]),
                    "tokens_consumed": tokens,
                }
                continue
            if existing["sha256"] != str(row["sha256"]):
                raise Stage2TrainerError(
                    f"Shard {relative_path!r} changed digest mid-run."
                )
            existing["tokens_consumed"] += tokens

    def corpus_consumption(self) -> Stage2CorpusConsumption:
        return Stage2CorpusConsumption(
            manifest_path=str(self.corpus.manifest_path),
            manifest_sha256=str(self.corpus.manifest_sha256),
            teacher_checkpoint_sha256=str(self.corpus.teacher_checkpoint_sha256),
            shard_order_seed=int(self.corpus.shard_order_seed),
            shards=tuple(
                Stage2ShardConsumption(
                    order_index=int(row["order_index"]),
                    relative_path=str(row["relative_path"]),
                    sha256=str(row["sha256"]),
                    tokens_consumed=int(row["tokens_consumed"]),
                )
                for row in sorted(
                    self._shards.values(),
                    key=lambda value: int(value["order_index"]),
                )
            ),
        )

    # -- objective branches -------------------------------------------------

    def _student_logits(self, donor_ids: Tensor) -> Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.config.autocast_dtype,
            enabled=self._autocast_enabled,
        ):
            logits = self.student(donor_ids)
        self.micro_batch_forward_calls += 1
        return logits

    def _teacher_logits(self, donor_ids: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.teacher_donor_logits(donor_ids)
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise Stage2TrainerError(
                "The teacher must return [batch, length, rows] donor logits."
            )
        return logits.detach()

    def _draw(self, donor_logits: Tensor, *, greedy: bool) -> Tensor:
        """One ``[batch, 1]`` donor id from ``[batch, donor_rows]`` logits.

        Both sampling paths funnel through here, so the cached and uncached
        decodes cannot diverge in the *selection* rule -- only in how the
        logits were produced, which is exactly what the equivalence test is
        measuring.
        """

        restricted = self.vocabulary.restrict(donor_logits)
        if greedy:
            return torch.argmax(restricted, dim=-1, keepdim=True)
        probabilities = torch.softmax(
            restricted.float() / self.objective.temperature,
            dim=-1,
        )
        return torch.multinomial(
            probabilities.to(device="cpu"),
            num_samples=1,
            generator=self._sample_generator,
        ).to(device=donor_logits.device)

    def sample_on_policy_sequence(
        self,
        prefix: Tensor,
        *,
        greedy: bool = False,
        use_cache: bool | None = None,
    ) -> Tensor:
        """Ancestral student continuation at T=1.0 over defined donor rows.

        ``use_cache`` defaults to the config's
        ``on_policy_sampling_uses_cache``.  The uncached branch re-runs the
        whole student forward per token (the original H4 cost); the cached
        branch prefills once and advances one token at a time through the
        child body's own cache, scoring only the position that emits.
        """

        cached = (
            self.config.on_policy_sampling_uses_cache
            if use_cache is None
            else bool(use_cache)
        )
        context = prefix
        was_training = self.student.training
        self.student.eval()
        try:
            with torch.no_grad():
                if cached:
                    donor_logits, state = self.student.start_cached_decode(context)
                    self.on_policy_sample_forward_calls += 1
                    for step in range(self.objective.continuation_tokens):
                        drawn = self._draw(donor_logits, greedy=greedy)
                        context = torch.cat((context, drawn), dim=1)
                        if step + 1 == self.objective.continuation_tokens:
                            # The terminal token is never fed back: nothing
                            # scores after it.  The deployed decode makes the
                            # same skip (``terminal_token_not_fed_to_unused_
                            # cache``), and it keeps the forward-call count at
                            # exactly one per emitted token.
                            break
                        donor_logits = self.student.advance_cached_decode(
                            state,
                            drawn,
                        )
                        self.on_policy_sample_forward_calls += 1
                else:
                    for _step in range(self.objective.continuation_tokens):
                        logits = self.student(context)
                        self.on_policy_sample_forward_calls += 1
                        drawn = self._draw(logits[:, -1, :], greedy=greedy)
                        context = torch.cat((context, drawn), dim=1)
        finally:
            self.student.train(was_training)
        return context

    def _sample_on_policy_sequence(self, prefix: Tensor) -> Tensor:
        return self.sample_on_policy_sequence(prefix)

    def _micro_batches(self, donor_ids: Tensor) -> list[Tensor]:
        size = self.config.micro_batch_sequences
        return [
            donor_ids[start : start + size]
            for start in range(0, int(donor_ids.shape[0]), size)
        ]

    def _off_policy_loss(self, donor_ids: Tensor) -> Tensor:
        student_logits = self._student_logits(donor_ids)
        teacher_logits = self._teacher_logits(donor_ids)
        return forward_kl_logit_distillation(
            student_logits,
            teacher_logits,
            vocabulary=self.vocabulary,
            temperature=self.objective.temperature,
        )

    def _on_policy_loss(self, donor_ids: Tensor) -> Tensor:
        continuation = self.objective.continuation_tokens
        prefix_length = int(donor_ids.shape[1]) - continuation
        if prefix_length <= 0:
            raise Stage2TrainerError(
                "On-policy sequences need a prefix shorter than the sequence."
            )
        sequence = self._sample_on_policy_sequence(donor_ids[:, :prefix_length])
        student_logits = self._student_logits(sequence)
        teacher_logits = self._teacher_logits(sequence)
        # The GKD divergence is taken on the distributions that *produced* the
        # student's own tokens: position t-1 emitted token t.
        start = prefix_length - 1
        stop = int(sequence.shape[1]) - 1
        return generalized_jensen_shannon_divergence(
            student_logits[:, start:stop, :],
            teacher_logits[:, start:stop, :],
            vocabulary=self.vocabulary,
            beta=self.objective.beta,
            temperature=self.objective.temperature,
        )

    # -- one optimizer step -------------------------------------------------

    def _step(self, step_index: int) -> dict[str, Any]:
        phase_before = self._phase
        phase_now = self.objective.phase_at(self.tokens_consumed)
        if phase_now != phase_before:
            self._phase = phase_now
            self.steps_in_phase = 0
            self._phase_switch_step = step_index
        on_policy = self.objective.step_is_on_policy(
            tokens_consumed=self.tokens_consumed,
            steps_in_phase=self.steps_in_phase,
        )
        learning_rate = self.config.learning_rate_at(step_index)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        batch = self.corpus.next_batch(
            sequences=self.config.batch_sequences,
            sequence_length=self.config.sequence_length,
        )
        donor_ids = batch.donor_ids.to(device=self.device, dtype=torch.int64)
        if tuple(donor_ids.shape) != (
            self.config.batch_sequences,
            self.config.sequence_length,
        ):
            raise Stage2TrainerError(
                "Corpus returned a batch of the wrong exact shape."
            )
        self._record_shards(batch.shards)
        self.optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        micro_batches = self._micro_batches(donor_ids)
        for micro in micro_batches:
            loss = (
                self._on_policy_loss(micro)
                if on_policy
                else self._off_policy_loss(micro)
            )
            scaled = loss / len(micro_batches)
            scaled.backward()
            self.backward_calls += 1
            accumulated += float(loss.detach())
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.student.parameters()
                    if parameter.requires_grad
                ],
                self.config.gradient_clip_norm,
            )
        )
        self.optimizer.step()
        # AdamW's decoupled weight decay moves parameters that received no
        # gradient, so the protected route anchors are re-pinned after every
        # step rather than trusted to the gradient mask alone.
        self.student.restore_protected_anchor_coefficients()
        self.optimizer_steps += 1
        self.steps_in_phase += 1
        self.tokens_consumed += self.config.tokens_per_step
        if on_policy:
            self.on_policy_steps += 1
        else:
            self.off_policy_steps += 1
        return {
            "step": step_index,
            "tokens_consumed": self.tokens_consumed,
            "phase": phase_now,
            "on_policy": on_policy,
            "objective": (
                "generalized-jsd-gkd" if on_policy else "forward-kl"
            ),
            "loss": accumulated / len(micro_batches),
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
        }

    # -- probes and checkpoints ---------------------------------------------

    def _run_probe(self) -> dict[str, Any] | None:
        if self.probe is None:
            return None
        was_training = self.student.training
        self.student.eval()
        try:
            result = run_stage2_freerun_probe(
                student_donor_logits=self.student,
                teacher_donor_logits=self.probe.teacher_donor_logits,
                probe=self.probe.prompts,
                tokenize=self.probe.tokenize,
                maximum_new_tokens=self.probe.maximum_new_tokens,
                end_of_sequence_ids=self.probe.end_of_sequence_ids,
                device=self.device,
                tokens_consumed=self.tokens_consumed,
                optimizer_steps=self.optimizer_steps,
                teacher_rollout_cache=self._teacher_rollout_cache,
            )
        finally:
            self.student.train(was_training)
        self._probe_series.append(result)
        return result

    def _run_held_out_nll(self) -> dict[str, Any] | None:
        if self.held_out_batch is None:
            return None
        donor_ids, targets = self.held_out_batch()
        was_training = self.student.training
        self.student.eval()
        try:
            with torch.no_grad():
                logits = self.student(donor_ids.to(device=self.device))
                value = float(
                    diagnostic_held_out_negative_log_likelihood(
                        logits,
                        targets.to(device=self.device),
                        vocabulary=self.vocabulary,
                    )
                )
        finally:
            self.student.train(was_training)
        record = {
            "tokens_consumed": self.tokens_consumed,
            "optimizer_steps": self.optimizer_steps,
            "negative_log_likelihood": value,
            "diagnostic_only": True,
        }
        self._nll_series.append(record)
        return record

    def _write_checkpoint(
        self,
        *,
        rung_tokens: int,
        directory_name: str,
    ) -> dict[str, Any] | None:
        if self.checkpoints is None:
            return None
        settings = self.checkpoints
        optimizer = (
            self.optimizer
            if settings.keep_optimizer_state and settings.save_optimizer
            else None
        )
        return write_stage2_rung_checkpoint(
            Path(settings.root) / directory_name,
            student=self.student,
            base=settings.base,
            provenance=settings.provenance,
            structural_route_tensors=settings.structural_route_tensors,
            initialization_child_directory=settings.initialization_child_directory,
            semantic_source_tokenizer_sha256=(
                settings.semantic_source_tokenizer_sha256
            ),
            lexicon_package_sha256=settings.lexicon_package_sha256,
            arm_id=self.config.arm_id,
            rung_tokens=rung_tokens,
            tokens_consumed=self.tokens_consumed,
            optimizer_steps=self.optimizer_steps,
            backward_calls=self.backward_calls,
            init_source_sha256=(
                self.student.initialization.init_source_sha256
            ),
            routes_were_trained=self.student.train_routes,
            initialization_mode=self.student.initialization.mode,
            init_artifact_sha256=settings.init_artifact_sha256,
            optimizer=optimizer,
            save_optimizer=settings.save_optimizer,
        )

    def _maybe_safety_checkpoint(self, step_index: int) -> None:
        if self.checkpoints is None or self.config.safety_checkpoint_seconds <= 0:
            return
        now = time.monotonic()
        if now - self._last_safety_checkpoint < self.config.safety_checkpoint_seconds:
            return
        self._last_safety_checkpoint = now
        manifest = self._write_checkpoint(
            rung_tokens=max(self.tokens_consumed, 1),
            directory_name=f"safety-step-{step_index:09d}",
        )
        if manifest is None:
            return
        self._safety_checkpoints.append(
            Path(self.checkpoints.root) / f"safety-step-{step_index:09d}"
        )
        while len(self._safety_checkpoints) > self.config.safety_checkpoints_kept:
            shutil.rmtree(self._safety_checkpoints.pop(0), ignore_errors=True)

    # -- run ---------------------------------------------------------------

    def run(
        self,
        *,
        maximum_steps: int | None = None,
        verify_route_surface: tuple[Any, Sequence[int], Tensor] | None = None,
    ) -> dict[str, Any]:
        """Train to the last rung, sealing a receipt with every counter."""

        if verify_route_surface is not None:
            runtime, donor_ids, target_scores = verify_route_surface
            self.route_surface_equivalence = (
                assert_route_surface_matches_frozen_runtime(
                    self.student,
                    runtime,
                    donor_ids=donor_ids,
                    target_scores=target_scores,
                )
            )
        started = time.monotonic()
        self.student.train()
        rungs: list[Stage2RungRecord] = []
        rung_targets = list(self.config.rung_tokens)
        total_steps = (
            self.config.total_steps
            if maximum_steps is None
            else min(self.config.total_steps, int(maximum_steps))
        )
        rung_index = 0
        rung_started = started
        rung_start_counters = self._counters()
        for step_index in range(total_steps):
            record = self._step(step_index)
            if (
                step_index % self.config.loss_log_interval_steps == 0
                or step_index + 1 == total_steps
            ):
                self._loss_curve.append(record)
            if self.tokens_consumed >= self._next_probe_tokens:
                self._run_probe()
                self._run_held_out_nll()
                interval = (
                    STAGE2_FREERUN_PROBE_INTERVAL_TOKENS
                    if self.probe is None
                    else self.probe.interval_tokens
                )
                self._next_probe_tokens = self.tokens_consumed + interval
            self._maybe_safety_checkpoint(step_index)
            while (
                rung_index < len(rung_targets)
                and self.tokens_consumed >= rung_targets[rung_index]
            ):
                rungs.append(
                    self._seal_rung(
                        rung_tokens=rung_targets[rung_index],
                        rung_started=rung_started,
                        start_counters=rung_start_counters,
                    )
                )
                rung_index += 1
                rung_started = time.monotonic()
                rung_start_counters = self._counters()
        if rung_index < len(rung_targets):
            # A truncated run (calibration, toy) still seals the rung it
            # actually reached, labelled with the tokens it actually consumed.
            rungs.append(
                self._seal_rung(
                    rung_tokens=max(self.tokens_consumed, 1),
                    rung_started=rung_started,
                    start_counters=rung_start_counters,
                    partial=True,
                )
            )
        self.wall_clock_seconds = time.monotonic() - started
        builder = Stage2TrainingReceiptBuilder(
            arm_id=self.config.arm_id,
            init_source_sha256=self.student.initialization.init_source_sha256,
            initialization=self.student.initialization.as_dict(),
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            corpus=self.corpus_consumption(),
            objective_schedule=self.objective.as_dict(),
            optimizer_configuration=self.config.optimizer_configuration(),
            trainer_configuration={
                **self.config.as_dict(),
                "device": str(self.device),
                "autocast_enabled": self._autocast_enabled,
                "executed_steps": self.optimizer_steps,
                "on_policy_sample_forward_calls": (
                    self.on_policy_sample_forward_calls
                ),
                "phase_switch_step": self._phase_switch_step,
                "run_truncated_by_maximum_steps": maximum_steps is not None,
            },
            preregistration=self.preregistration,
            device_inventory=self.device_inventory,
            seeds={
                "run_seed": int(self.config.seed),
                "on_policy_sample_seed": int(self.config.seed) ^ 0x5F2B,
                "probe_seed": (
                    None if self.probe is None else self.probe.prompts.seed
                ),
                "corpus_shard_order_seed": int(self.corpus.shard_order_seed),
            },
            probe_surface=(
                {} if self.probe is None else self.probe.prompts.as_dict()
            ),
            route_surface_equivalence=self.route_surface_equivalence,
            counted_parameters={
                **self.student.counted_parameter_elements(),
                **self.student.route_training_mask(),
                "route_layouts": self.student.route_layouts(),
            },
            rungs=rungs,
            loss_curve=list(self._loss_curve),
            free_run_probe_series=list(self._probe_series),
            diagnostic_held_out_nll_series=list(self._nll_series),
            wall_clock_seconds=self.wall_clock_seconds,
        )
        try:
            return builder.seal()
        except Stage2TrainingReceiptError as exc:
            raise Stage2TrainerError(
                f"Stage-2 run produced no sealable receipt: {exc}"
            ) from exc

    def _counters(self) -> dict[str, int]:
        return {
            "tokens_consumed": self.tokens_consumed,
            "optimizer_steps": self.optimizer_steps,
            "backward_calls": self.backward_calls,
            "micro_batch_forward_calls": self.micro_batch_forward_calls,
            "on_policy_sample_forward_calls": self.on_policy_sample_forward_calls,
            "on_policy_steps": self.on_policy_steps,
            "off_policy_steps": self.off_policy_steps,
        }

    def _seal_rung(
        self,
        *,
        rung_tokens: int,
        rung_started: float,
        start_counters: Mapping[str, int],
        partial: bool = False,
    ) -> Stage2RungRecord:
        probe = self._run_probe()
        manifest = self._write_checkpoint(
            rung_tokens=rung_tokens,
            directory_name=f"rung-{rung_tokens}",
        )
        # Each rung carries only the curve segment it is responsible for; the
        # full series lives once at the top level of the receipt.
        loss_segment = tuple(self._loss_curve[self._loss_curve_sealed :])
        nll_segment = tuple(self._nll_series[self._nll_series_sealed :])
        self._loss_curve_sealed = len(self._loss_curve)
        self._nll_series_sealed = len(self._nll_series)
        counters = self._counters()
        return Stage2RungRecord(
            rung_tokens=int(rung_tokens),
            tokens_consumed=counters["tokens_consumed"],
            optimizer_steps=counters["optimizer_steps"],
            backward_calls=counters["backward_calls"],
            micro_batch_forward_calls=counters["micro_batch_forward_calls"],
            on_policy_steps=counters["on_policy_steps"]
            - int(start_counters["on_policy_steps"]),
            off_policy_steps=counters["off_policy_steps"]
            - int(start_counters["off_policy_steps"]),
            wall_clock_seconds=time.monotonic() - rung_started,
            checkpoint_manifest_sha256=(
                None if manifest is None else str(manifest["manifest_sha256"])
            ),
            checkpoint_directory=(
                None
                if self.checkpoints is None
                else str(Path(self.checkpoints.root) / f"rung-{rung_tokens}")
            ),
            optimizer_state=optimizer_state_digest(self.optimizer),
            free_run_probe=probe,
            loss_curve=loss_segment,
            diagnostic_held_out_nll=nll_segment,
        )

    @property
    def phase_switch_step(self) -> int | None:
        """The step at which the on-policy phase opened, if it opened."""

        return self._phase_switch_step


__all__ = [
    "STAGE2_ADAM_BETAS",
    "STAGE2_BATCH_SEQUENCES",
    "STAGE2_DESIGN_STATED_RUNG_STEPS",
    "STAGE2_GRADIENT_CLIP_NORM",
    "STAGE2_LEARNING_RATE",
    "STAGE2_MINIMUM_LEARNING_RATE",
    "STAGE2_SEQUENCE_LENGTH",
    "STAGE2_TRAINER_SCHEMA",
    "STAGE2_WARMUP_STEPS",
    "STAGE2_WEIGHT_DECAY",
    "Stage2CheckpointSettings",
    "Stage2CorpusBatch",
    "Stage2CorpusSource",
    "Stage2DistillationTrainer",
    "Stage2ProbeSettings",
    "Stage2TrainerConfig",
    "Stage2TrainerError",
]
