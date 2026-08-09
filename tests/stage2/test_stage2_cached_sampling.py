"""H4: KV-cached on-policy sampling, proved equivalent and timed.

The uncached sampler re-ran the whole student forward for every generated
token -- 256 full forwards over a growing context per on-policy step.  The
cached sampler prefills once and then advances one semantic token at a time
through the child body's own cache, which is the mechanism the deployed routed
decode already uses (``hf_lfm2_semantic_step_model_runtime``: prefill with
``use_cache=True``, then ``past_key_values=cache`` with a full-length attention
mask, one token per call).

Two surfaces are proved, deliberately:

* the **real architecture** -- a genuine ``transformers.Lfm2ForCausalLM`` body,
  hybrid conv + attention, wearing the carrier ABI's untied tables and the
  toy's real packaged routes.  This is where a cache is actually hard, and it
  is the only place a claim about "the LFM2 cache" can be made honestly.
* the **trainer's own sampler** on the CPU toy, so the flag, the RNG stream and
  the forward-call accounting are exercised end to end.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch


transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stage2_cpu_toy_validation_cached_sampling",
    REPOSITORY_ROOT / "experiments" / "stage2_cpu_toy_validation.py",
)
assert SPEC is not None and SPEC.loader is not None
TOY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOY
SPEC.loader.exec_module(TOY)

from neural_forge.lfm2_carrier_child_package import (  # noqa: E402
    load_authenticated_lfm2_carrier_child_package,
)
from neural_forge.stage2_carrier_student import (  # noqa: E402
    STAGE2_FOLD_INITIALIZATION,
    Stage2CarrierStudentError,
    Stage2RoutedStudent,
    Stage2StudentInitialization,
    load_stage2_routed_student,
)
from neural_forge.stage2_distillation_trainer import (  # noqa: E402
    Stage2CorpusBatch,
    Stage2DistillationTrainer,
    Stage2TrainerConfig,
)
from neural_forge.stage2_kd_objective import (  # noqa: E402
    Stage2DonorVocabulary,
    Stage2ObjectiveSchedule,
)


@pytest.fixture(scope="module")
def published_child(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return TOY.build_toy_published_child(
        tmp_path_factory.mktemp("stage2-cached-sampling-fixture")
    )


# ---------------------------------------------------------------------------
# A real LFM2 body wearing the carrier ABI
# ---------------------------------------------------------------------------


def _real_lfm2_child(seed: int = 20260809) -> Any:
    """A tiny but genuine LFM2: conv blocks plus a full-attention block."""

    torch.manual_seed(seed)
    config = transformers.Lfm2Config(
        vocab_size=TOY.TOY_TARGET_ROWS,
        hidden_size=TOY.TOY_HIDDEN_SIZE,
        intermediate_size=2 * TOY.TOY_HIDDEN_SIZE,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        conv_bias=False,
        conv_L_cache=3,
        max_position_embeddings=128,
        block_multiple_of=1,
        block_auto_adjust_ff_dim=False,
        block_ffn_dim_multiplier=1.0,
        norm_eps=1e-5,
        tie_word_embeddings=False,
        layer_types=["conv", "conv", "full_attention", "conv"],
    )
    model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _student_on(child: Any, published: dict[str, Any]) -> Stage2RoutedStudent:
    package = load_authenticated_lfm2_carrier_child_package(
        published["child_directory"]
    )
    tensors = package.routes.tensors
    return Stage2RoutedStudent(
        child,
        input_offsets=tensors["input_offsets"],
        input_target_ids=tensors["input_target_ids"],
        input_coefficients=tensors["input_coefficients"],
        output_offsets=tensors["output_offsets"],
        output_target_ids=tensors["output_target_ids"],
        output_coefficients=tensors["output_coefficients"],
        direct_target_row_to_semantic_id=tensors[
            "direct_target_row_to_semantic_id"
        ],
        donor_row_count=TOY.TOY_DONOR_ROWS,
        target_row_count=TOY.TOY_TARGET_ROWS,
        hidden_size=TOY.TOY_HIDDEN_SIZE,
        train_routes=False,
        initialization=Stage2StudentInitialization(
            mode=STAGE2_FOLD_INITIALIZATION,
            child_directory=str(published["child_directory"]),
            child_package_sha256=package.package_sha256,
            config_sha256="0" * 64,
            source_bridge_artifact_sha256=published["source_artifact_sha256"],
            train_routes=False,
        ),
    )


def _uncached_rollout(
    student: Stage2RoutedStudent,
    prefix: torch.Tensor,
    steps: int,
    *,
    greedy: bool,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    context = prefix
    logits_seen: list[torch.Tensor] = []
    with torch.no_grad():
        for _step in range(steps):
            logits = student(context)[:, -1, :]
            logits_seen.append(logits.clone())
            if greedy:
                drawn = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(logits.float(), dim=-1)
                drawn = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
            context = torch.cat((context, drawn), dim=1)
    return context, logits_seen


def _cached_rollout(
    student: Stage2RoutedStudent,
    prefix: torch.Tensor,
    steps: int,
    *,
    greedy: bool,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    context = prefix
    logits_seen: list[torch.Tensor] = []
    with torch.no_grad():
        logits, state = student.start_cached_decode(context)
        for step in range(steps):
            logits_seen.append(logits.clone())
            if greedy:
                drawn = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(logits.float(), dim=-1)
                drawn = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
            context = torch.cat((context, drawn), dim=1)
            if step + 1 < steps:
                logits = student.advance_cached_decode(state, drawn)
    return context, logits_seen


@pytest.mark.parametrize("greedy", [True, False])
def test_cached_decode_matches_the_uncached_path_on_a_real_lfm2_body(
    published_child: dict[str, Any],
    greedy: bool,
) -> None:
    """The correctness pin, on the architecture the child actually is."""

    student = _student_on(_real_lfm2_child(), published_child)
    prefix = torch.tensor(
        [[3, 9, 1, 7, 4, 2], [8, 8, 0, 5, 6, 3]],
        dtype=torch.int64,
    )
    steps = 12
    uncached, uncached_logits = _uncached_rollout(
        student,
        prefix,
        steps,
        greedy=greedy,
        generator=torch.Generator().manual_seed(4242),
    )
    cached, cached_logits = _cached_rollout(
        student,
        prefix,
        steps,
        greedy=greedy,
        generator=torch.Generator().manual_seed(4242),
    )
    assert torch.equal(uncached, cached)
    assert uncached.shape[1] == int(prefix.shape[1]) + steps
    # The token sequences are identical; the logits agree to float32 noise,
    # which is what a cached attention/conv stack can promise and all a sampler
    # needs.  Recorded rather than asserted loosely.
    deltas = [
        float((left - right).abs().max())
        for left, right in zip(uncached_logits, cached_logits, strict=True)
    ]
    assert max(deltas) < 1e-4, deltas


def test_the_real_lfm2_cache_actually_shortens_the_context_it_reads(
    published_child: dict[str, Any],
) -> None:
    """A cache that silently re-reads the prefix would pass equality alone."""

    student = _student_on(_real_lfm2_child(), published_child)
    prefix = torch.tensor([[3, 9, 1, 7, 4, 2]], dtype=torch.int64)
    seen: list[int] = []
    body = student.child.model
    original = body.forward

    def watched(*args: Any, **kwargs: Any) -> Any:
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None:
            seen.append(int(embeds.shape[1]))
        return original(*args, **kwargs)

    body.forward = watched
    try:
        _cached_rollout(student, prefix, 6, greedy=True)
    finally:
        body.forward = original
    # one prefill of the whole prefix, then five single-token advances (the
    # terminal token is never fed back -- nothing scores after it).
    assert seen == [6, 1, 1, 1, 1, 1]


def test_a_body_that_returns_no_cache_is_refused_rather_than_guessed(
    published_child: dict[str, Any],
) -> None:
    student = _student_on(_real_lfm2_child(), published_child)

    class _CachelessBody(torch.nn.Module):
        """A body that ignores ``use_cache`` -- a real risk with a custom body."""

        def __init__(self, embed_tokens: torch.nn.Module) -> None:
            super().__init__()
            self.embed_tokens = embed_tokens

        def forward(self, **kwargs: Any) -> torch.Tensor:
            return torch.zeros(
                (
                    int(kwargs["inputs_embeds"].shape[0]),
                    int(kwargs["inputs_embeds"].shape[1]),
                    TOY.TOY_HIDDEN_SIZE,
                )
            )

    student.child.model = _CachelessBody(student.child.model.embed_tokens)
    with pytest.raises(Stage2CarrierStudentError, match="returned no cache"):
        student.start_cached_decode(
            torch.tensor([[1, 2, 3]], dtype=torch.int64),
        )


# ---------------------------------------------------------------------------
# The trainer's own sampler, on the CPU toy
# ---------------------------------------------------------------------------


class _StubCorpus:
    """Enough of ``Stage2CorpusSource`` for a sampler test; never stepped."""

    def next_batch(self, *, sequences: int, sequence_length: int) -> Any:
        return Stage2CorpusBatch(
            donor_ids=torch.zeros((sequences, sequence_length), dtype=torch.int64),
            shards=(),
        )


def _trainer(
    published: dict[str, Any],
    *,
    uses_cache: bool,
    seed: int = 31,
    continuation_tokens: int = 8,
) -> Stage2DistillationTrainer:
    student, _decode_model, _package = load_stage2_routed_student(
        published["child_directory"],
        base=TOY._ToyBaseCheckpointIO(None),
        mode=STAGE2_FOLD_INITIALIZATION,
        train_routes=False,
    )
    config = Stage2TrainerConfig(
        arm_id="S2-TOY",
        seed=seed,
        rung_tokens=(TOY.TOY_BATCH_SEQUENCES * TOY.TOY_SEQUENCE_LENGTH,),
        batch_sequences=TOY.TOY_BATCH_SEQUENCES,
        micro_batch_sequences=TOY.TOY_MICRO_BATCH_SEQUENCES,
        sequence_length=TOY.TOY_SEQUENCE_LENGTH,
        warmup_steps=1,
        on_policy_sampling_uses_cache=uses_cache,
    )
    return Stage2DistillationTrainer(
        config=config,
        student=student,
        teacher_donor_logits=lambda donor_ids: torch.zeros(
            (*donor_ids.shape, TOY.TOY_DONOR_ROWS)
        ),
        teacher_checkpoint_sha256="0" * 64,
        corpus=_StubCorpus(),
        donor_vocabulary=Stage2DonorVocabulary(
            defined_row_count=TOY.TOY_DONOR_ROWS,
            physical_row_count=TOY.TOY_DONOR_ROWS,
        ),
        objective=Stage2ObjectiveSchedule(
            switch_tokens=0,
            continuation_tokens=continuation_tokens,
        ),
        preregistration={},
    )


@pytest.mark.parametrize("greedy", [True, False])
def test_the_trainers_sampler_is_identical_cached_and_uncached(
    published_child: dict[str, Any],
    greedy: bool,
) -> None:
    """Same seed, same flag-off/flag-on, same tokens -- the H4 correctness pin."""

    prefix = torch.tensor(
        [[3, 9, 1, 7], [8, 8, 0, 5], [2, 2, 6, 4], [5, 1, 1, 9]],
        dtype=torch.int64,
    )
    uncached = _trainer(published_child, uses_cache=False)
    cached = _trainer(published_child, uses_cache=True)
    without = uncached.sample_on_policy_sequence(prefix, greedy=greedy)
    with_cache = cached.sample_on_policy_sequence(prefix, greedy=greedy)
    assert torch.equal(without, with_cache)
    assert int(without.shape[1]) == int(prefix.shape[1]) + 8
    # One body advance per emitted token either way, so the counter that the
    # receipt reports stays comparable across the flag.
    assert uncached.on_policy_sample_forward_calls == 8
    assert cached.on_policy_sample_forward_calls == 8


def test_the_flag_defaults_to_cached_and_the_override_is_honoured(
    published_child: dict[str, Any],
) -> None:
    trainer = _trainer(published_child, uses_cache=True)
    assert trainer.config.on_policy_sampling_uses_cache is True
    prefix = torch.tensor([[3, 9, 1, 7]] * 4, dtype=torch.int64)
    default = trainer.sample_on_policy_sequence(prefix, greedy=True)
    forced_off = trainer.sample_on_policy_sequence(
        prefix,
        greedy=True,
        use_cache=False,
    )
    assert torch.equal(default, forced_off)
    assert Stage2TrainerConfig(
        arm_id="S2-B",
        seed=1,
    ).on_policy_sampling_uses_cache is True


def test_cached_sampling_is_measurably_faster_on_the_toy(
    published_child: dict[str, Any],
    tmp_path: Path,
) -> None:
    """A timing measurement, recorded to disk, not an assertion about hardware.

    The only hard assertion is that the cached path is not *slower*; the actual
    ratio is written out so the number in NOTES-S2C is a measured one.  The toy
    body is 16-wide over 24 donor rows, so this understates the real gain by a
    lot -- the uncached path re-runs the body over the whole growing context
    *and* takes the routed ``C_out`` readout at every position, including the
    ~1,023 positions that emit nothing.
    """

    prefix_tokens = 256
    continuation_tokens = 64
    long_prefix = torch.randint(
        0,
        TOY.TOY_DONOR_ROWS,
        (2, prefix_tokens),
        generator=torch.Generator().manual_seed(11),
        dtype=torch.int64,
    )
    timings: dict[str, float] = {}
    for name, flag in (("uncached", False), ("cached", True)):
        trainer = _trainer(
            published_child,
            uses_cache=flag,
            continuation_tokens=continuation_tokens,
        )
        trainer.sample_on_policy_sequence(long_prefix, greedy=True, use_cache=flag)
        started = time.perf_counter()
        for _repeat in range(3):
            trainer.sample_on_policy_sequence(
                long_prefix,
                greedy=True,
                use_cache=flag,
            )
        timings[name] = (time.perf_counter() - started) / 3
    timings["speedup"] = timings["uncached"] / max(timings["cached"], 1e-9)
    timings["prefix_tokens"] = float(prefix_tokens)
    timings["continuation_tokens"] = float(continuation_tokens)
    timings["batch_sequences"] = float(long_prefix.shape[0])
    (tmp_path / "cached-sampling-timing.json").write_text(
        json.dumps(timings, indent=2, sort_keys=True),
        encoding="ascii",
    )
    print(json.dumps(timings, sort_keys=True))  # noqa: T201 - the measurement
    assert timings["speedup"] > 1.0
