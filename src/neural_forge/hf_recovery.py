"""Lightweight recovery training for Hugging Face causal LM surgery outputs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from neural_forge.hf_tools import (
    BEHAVIOR_PROBES,
    GENERATION_PROMPTS,
    TEXT_EVALS,
    _category_breakdown_summary,
    _decoder_layer_container,
    _dtype_from_name,
    _generate_text_batch,
    _generation_health,
    _load_eval_suite,
    _passed_category_breakdown,
    _ridge_solve_features,
    _score_behavior_probe,
    _torch_and_transformers,
)


RECOVERY_TEXTS = [
    "A compressed assistant should keep short factual answers stable after its internal layers are edited.",
    "Model surgery changes the geometry of hidden states, so recovery needs held-out behavior checks.",
    "A useful pruning run keeps arithmetic, code syntax, and common-sense answers readable.",
    "Python code blocks usually contain a function name, parameters, indentation, and a return statement.",
    "A cautious merge workflow searches several small coefficients and rejects repetitive decoding.",
    "Calibration text should teach the repaired model local language structure without copying eval prompts.",
    "When a checkpoint is folded too hard, perplexity and generation samples can reveal the damage.",
    "Short recovery training can reduce surgery shock, but it must be validated on unseen prompts.",
]


def _recovery_progress(event: str, **fields: object) -> None:
    values = " ".join(f"{key}={fields[key]}" for key in sorted(fields))
    suffix = f" {values}" if values else ""
    print(f"neural_forge_progress event={event}{suffix}", flush=True)

HIDDEN_KD_METRICS = {
    "projected",
    "cka",
    "both",
    "rsd",
    "projected-rsd",
    "gram",
    "projected-gram",
    "geometry",
    "projected-geometry",
}
PROJECTED_HIDDEN_KD_METRICS = {"projected", "both", "projected-rsd", "projected-gram", "projected-geometry"}
HIDDEN_KD_METRIC_DESCRIPTION = (
    "projected, cka, both, rsd, projected-rsd, gram, projected-gram, geometry, or projected-geometry"
)
MODULE_OUTPUT_TARGETS = {"attention", "mlp", "both"}
MODULE_OUTPUT_TRANSPORTS = {"paired", "sinkhorn", "both"}
SPAN_PAIR_MODES = {"exact", "overlap", "both"}
RESIDUAL_MAGNITUDE_COMPENSATION_MODES = {
    "none",
    "pre",
    "pre-linear",
    "linear-pre",
    "pre-lowrank",
    "lowrank-pre",
    "pre-eora",
    "eora-pre",
    "pre-module-linear",
    "module-linear-pre",
    "pre-module-lowrank",
    "module-lowrank-pre",
    "pre-module-eora",
    "module-eora-pre",
}
MODULE_RESIDUAL_COMPENSATION_MODES = {
    "pre-module-linear",
    "pre-module-lowrank",
    "pre-module-eora",
}


def _validate_blockwise_learning_rate(lr: float, steps_per_layer: int) -> None:
    if steps_per_layer > 0 and lr <= 0:
        raise ValueError("lr must be positive when training steps are positive.")
    if steps_per_layer == 0 and lr < 0:
        raise ValueError("lr must be non-negative when training steps are zero.")


def _recovery_behavior_eval_suite(
    suite: str | Path | dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, Sequence[dict[str, Any]] | None, Sequence[dict[str, Any]] | None]:
    if suite is None:
        return None, None, None
    loaded = _load_eval_suite(suite)
    behavior_probes = loaded["behavior_probes"]
    if not behavior_probes:
        raise ValueError("behavior_eval_suite must contain at least one behavior probe.")
    return loaded, behavior_probes, loaded["generation_prompts"]


def _recovery_behavior_eval_suite_report(suite: dict[str, Any] | None) -> dict[str, Any] | None:
    if suite is None:
        return None
    return {
        "source": suite.get("source"),
        "behavior_probe_count": len(suite.get("behavior_probes", [])),
        "generation_prompt_count": len(suite.get("generation_prompts", [])),
    }


def _freeze_ordered_recovery_texts(texts: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings, not one string.")
    frozen: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"{field_name}[{index}] must be a string.")
        if not text.strip():
            raise ValueError(f"{field_name}[{index}] must not be blank.")
        frozen.append(text)
    return tuple(frozen)


def _ordered_recovery_texts_sha256(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(len(texts).to_bytes(8, byteorder="big", signed=False))
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _freeze_ordered_replay_bank(
    bank: Sequence[Sequence[str]],
    *,
    field_name: str,
) -> tuple[tuple[str, ...], ...]:
    if isinstance(bank, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of text batches.")
    frozen: list[tuple[str, ...]] = []
    for index, batch in enumerate(bank):
        frozen_batch = _freeze_ordered_recovery_texts(
            batch,
            field_name=f"{field_name}[{index}]",
        )
        if not frozen_batch:
            raise ValueError(f"{field_name}[{index}] must contain at least one text.")
        frozen.append(frozen_batch)
    return tuple(frozen)


def _freeze_on_policy_replay_bank(
    bank: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    return _freeze_ordered_replay_bank(bank, field_name="on_policy_replay_bank")


def _on_policy_replay_bank_sha256(bank: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    digest.update(len(bank).to_bytes(8, byteorder="big", signed=False))
    for batch in bank:
        batch_digest = bytes.fromhex(_ordered_recovery_texts_sha256(batch))
        digest.update(len(batch).to_bytes(8, byteorder="big", signed=False))
        digest.update(batch_digest)
    return digest.hexdigest()


_STEPWISE_REPLAY_SCHEMA = "neural-forge.stepwise-token-replay.v1"
_STEPWISE_REPLAY_PREFLIGHT_SCHEMA = "neural-forge.stepwise-token-replay-preflight.v1"


class _FrozenStepwiseReplayRow(NamedTuple):
    prompt_input_ids: tuple[int, ...]
    rollout_input_ids: tuple[int, ...]


class _FrozenStepwiseReplayEvent(NamedTuple):
    step: int
    rows: tuple[_FrozenStepwiseReplayRow, ...]


def _freeze_stepwise_token_ids(value: Any, *, field_name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of integer token IDs.")
    frozen: list[int] = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"{field_name}[{index}] must be an integer token ID.")
        if token_id < 0:
            raise ValueError(f"{field_name}[{index}] must be non-negative.")
        frozen.append(int(token_id))
    if not frozen:
        raise ValueError(f"{field_name} must contain at least one token ID.")
    return tuple(frozen)


def _freeze_stepwise_replay_bank(
    bank: Sequence[Mapping[str, Any]],
) -> tuple[_FrozenStepwiseReplayEvent, ...]:
    if isinstance(bank, (str, bytes, Mapping)) or not isinstance(bank, Sequence):
        raise TypeError("stepwise_replay_bank must be a sequence of token replay event mappings.")
    frozen_events: list[_FrozenStepwiseReplayEvent] = []
    for event_index, event in enumerate(bank):
        event_name = f"stepwise_replay_bank[{event_index}]"
        if not isinstance(event, Mapping):
            raise TypeError(f"{event_name} must be a mapping with step and rows fields.")
        if set(event) != {"step", "rows"}:
            raise ValueError(f"{event_name} must contain exactly the step and rows fields.")
        step = event["step"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(f"{event_name}.step must be an integer.")
        if step <= 0:
            raise ValueError(f"{event_name}.step must be positive.")
        rows = event["rows"]
        if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Sequence):
            raise TypeError(f"{event_name}.rows must be a sequence of token replay row mappings.")
        if not rows:
            raise ValueError(f"{event_name}.rows must contain at least one replay row.")
        frozen_rows: list[_FrozenStepwiseReplayRow] = []
        for row_index, row in enumerate(rows):
            row_name = f"{event_name}.rows[{row_index}]"
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"{row_name} must be a mapping with prompt_input_ids and rollout_input_ids fields."
                )
            if set(row) != {"prompt_input_ids", "rollout_input_ids"}:
                raise ValueError(
                    f"{row_name} must contain exactly prompt_input_ids and rollout_input_ids."
                )
            prompt_input_ids = _freeze_stepwise_token_ids(
                row["prompt_input_ids"],
                field_name=f"{row_name}.prompt_input_ids",
            )
            rollout_input_ids = _freeze_stepwise_token_ids(
                row["rollout_input_ids"],
                field_name=f"{row_name}.rollout_input_ids",
            )
            if len(rollout_input_ids) <= len(prompt_input_ids):
                raise ValueError(f"{row_name} must contain at least one continuation token.")
            if rollout_input_ids[: len(prompt_input_ids)] != prompt_input_ids:
                raise ValueError(f"{row_name}.prompt_input_ids must be an exact rollout prefix.")
            frozen_rows.append(
                _FrozenStepwiseReplayRow(
                    prompt_input_ids=prompt_input_ids,
                    rollout_input_ids=rollout_input_ids,
                )
            )
        frozen_events.append(_FrozenStepwiseReplayEvent(step=int(step), rows=tuple(frozen_rows)))
    return tuple(frozen_events)


def _update_stepwise_replay_digest_ids(digest: Any, token_ids: Sequence[int]) -> None:
    digest.update(len(token_ids).to_bytes(8, byteorder="big", signed=False))
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, byteorder="big", signed=False))


def _stepwise_replay_event_sha256(event: _FrozenStepwiseReplayEvent) -> str:
    digest = hashlib.sha256()
    digest.update(_STEPWISE_REPLAY_SCHEMA.encode("utf-8"))
    digest.update(int(event.step).to_bytes(8, byteorder="big", signed=False))
    digest.update(len(event.rows).to_bytes(8, byteorder="big", signed=False))
    for row in event.rows:
        _update_stepwise_replay_digest_ids(digest, row.prompt_input_ids)
        _update_stepwise_replay_digest_ids(digest, row.rollout_input_ids)
    return digest.hexdigest()


def _stepwise_replay_bank_sha256(bank: Sequence[_FrozenStepwiseReplayEvent]) -> str:
    digest = hashlib.sha256()
    digest.update(_STEPWISE_REPLAY_SCHEMA.encode("utf-8"))
    digest.update(len(bank).to_bytes(8, byteorder="big", signed=False))
    for event in bank:
        digest.update(bytes.fromhex(_stepwise_replay_event_sha256(event)))
    return digest.hexdigest()


def _stepwise_replay_preflight_sha256(summary: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            summary,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _normalize_behavior_checkpoint_steps(
    checkpoint_steps: Sequence[int] | None,
    *,
    steps: int,
) -> tuple[int, ...] | None:
    if checkpoint_steps is None:
        return None
    if isinstance(checkpoint_steps, (str, bytes)):
        raise TypeError("behavior_eval_checkpoint_steps must be a sequence of integers.")
    normalized: list[int] = []
    previous = -1
    for index, step in enumerate(checkpoint_steps):
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(f"behavior_eval_checkpoint_steps[{index}] must be an integer.")
        if step < 0 or step > steps:
            raise ValueError(f"behavior_eval_checkpoint_steps[{index}] must be in [0, {steps}].")
        if step <= previous:
            raise ValueError("behavior_eval_checkpoint_steps must be strictly increasing and unique.")
        normalized.append(int(step))
        previous = int(step)
    return tuple(normalized)


def _deep_readonly_recovery_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_readonly_recovery_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_readonly_recovery_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_readonly_recovery_value(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Recovery checkpoint observer values must be JSON-like, got {type(value).__name__}.")


def _notify_recovery_checkpoint_observer(
    observer: Callable[[Mapping[str, Any]], None] | None,
    *,
    operation: str,
    checkpoint_index: int,
    behavior_entry: Mapping[str, Any],
) -> None:
    if observer is None:
        return
    event = _deep_readonly_recovery_value(
        {
            "schema": "neural-forge.recovery-behavior-checkpoint-observer.v1",
            "operation": operation,
            "checkpoint_index": int(checkpoint_index),
            "step": int(behavior_entry["step"]),
            "behavior": dict(behavior_entry),
        }
    )
    observer(event)


def recover_causal_lm(
    source_model: str | None,
    candidate_model: str,
    output_dir: str | Path,
    steps: int = 80,
    batch_size: int = 1,
    max_length: int = 96,
    lr: float = 5e-5,
    kd_weight: float = 0.5,
    lm_weight: float = 0.5,
    hidden_kd_weight: float = 0.0,
    hidden_kd_metric: str = "cka",
    hidden_layer: str = "last",
    hidden_token_mode: str = "tokens",
    hidden_ridge: float = 1e-3,
    hidden_max_anchors: int = 2048,
    top_k_kd_weight: float = 0.0,
    top_k_kd_k: int = 32,
    first_token_top_k_kd_weight: float = 0.0,
    first_token_kd_weight: float = 0.0,
    first_token_lm_weight: float = 0.0,
    answer_contrastive_weight: float = 0.0,
    answer_contrastive_max_negatives: int = 4,
    unlikelihood_weight: float = 0.0,
    unlikelihood_window: int = 16,
    entropy_weight: float = 0.0,
    temperature: float = 2.0,
    train_scope: str = "all",
    torch_dtype: str = "float32",
    candidate_train_dtype: str | None = None,
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    calibration_texts: Sequence[str] | None = None,
    calibration_order: str = "original",
    response_loss_mode: str = "none",
    append_eos: bool = False,
    grad_clip: float = 1.0,
    report_interval: int | None = None,
    save_best_loss: bool = False,
    save_best_behavior: bool = False,
    behavior_eval_interval: int | None = None,
    behavior_max_new_tokens: int = 96,
    behavior_max_probes: int = 6,
    behavior_eval_suite: str | Path | dict[str, Any] | None = None,
    behavior_max_generation_prompts: int | None = None,
) -> dict[str, Any]:
    """Run a short recovery pass after surgery.

    When `kd_weight` is positive, `source_model` and `candidate_model` must use
    compatible tokenizers because KL is computed over the same vocabulary. When
    `kd_weight` is zero, recovery is teacherless LM repair on the candidate
    tokenizer, which is useful after cross-family/tokenizer folding.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    _validate_blockwise_learning_rate(float(lr), int(steps))
    if (
        kd_weight < 0
        or lm_weight < 0
        or hidden_kd_weight < 0
        or top_k_kd_weight < 0
        or first_token_top_k_kd_weight < 0
        or first_token_kd_weight < 0
        or first_token_lm_weight < 0
        or answer_contrastive_weight < 0
        or unlikelihood_weight < 0
        or entropy_weight < 0
    ):
        raise ValueError("loss weights must be non-negative.")
    if (
        kd_weight == 0
        and lm_weight == 0
        and hidden_kd_weight == 0
        and top_k_kd_weight == 0
        and first_token_top_k_kd_weight == 0
        and first_token_kd_weight == 0
        and first_token_lm_weight == 0
        and answer_contrastive_weight == 0
        and unlikelihood_weight == 0
        and entropy_weight == 0
    ):
        raise ValueError("At least one loss weight must be positive.")
    if answer_contrastive_max_negatives <= 0:
        raise ValueError("answer_contrastive_max_negatives must be positive.")
    if unlikelihood_window <= 0:
        raise ValueError("unlikelihood_window must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    if hidden_kd_metric not in HIDDEN_KD_METRICS:
        raise ValueError(f"hidden_kd_metric must be {HIDDEN_KD_METRIC_DESCRIPTION}.")
    if hidden_layer not in {"last", "middle", "all"}:
        raise ValueError("hidden_layer must be last, middle, or all.")
    if hidden_token_mode not in {"tokens", "response", "pooled"}:
        raise ValueError("hidden_token_mode must be tokens, response, or pooled.")
    if hidden_ridge < 0:
        raise ValueError("hidden_ridge must be non-negative.")
    if hidden_max_anchors <= 0:
        raise ValueError("hidden_max_anchors must be positive.")
    if top_k_kd_k <= 0:
        raise ValueError("top_k_kd_k must be positive.")
    if response_loss_mode == "none" and (
        first_token_kd_weight > 0 or first_token_lm_weight > 0 or first_token_top_k_kd_weight > 0
    ):
        raise ValueError("first-token losses require response_loss_mode to be auto, assistant, or answer.")
    if grad_clip < 0:
        raise ValueError("grad_clip must be non-negative.")
    if report_interval is not None and report_interval <= 0:
        raise ValueError("report_interval must be positive when provided.")
    if behavior_eval_interval is not None and behavior_eval_interval <= 0:
        raise ValueError("behavior_eval_interval must be positive when provided.")
    if behavior_max_new_tokens <= 0:
        raise ValueError("behavior_max_new_tokens must be positive.")
    if behavior_max_probes <= 0:
        raise ValueError("behavior_max_probes must be positive.")
    if behavior_max_generation_prompts is not None and behavior_max_generation_prompts <= 0:
        raise ValueError("behavior_max_generation_prompts must be positive when provided.")
    if calibration_order not in {"original", "domain-balanced"}:
        raise ValueError("calibration_order must be original or domain-balanced.")

    if (
        kd_weight > 0
        or hidden_kd_weight > 0
        or top_k_kd_weight > 0
        or first_token_top_k_kd_weight > 0
        or first_token_kd_weight > 0
    ) and not source_model:
        raise ValueError("source_model is required when teacher losses are positive.")

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    train_dtype = _dtype_from_name(torch, candidate_train_dtype) if candidate_train_dtype else None
    candidate_tokenizer = transformers.AutoTokenizer.from_pretrained(candidate_model, trust_remote_code=trust_remote_code)
    tokenizer = candidate_tokenizer
    source = None
    if source_model:
        source_tokenizer = transformers.AutoTokenizer.from_pretrained(source_model, trust_remote_code=trust_remote_code)
        if (
            kd_weight > 0
            or hidden_kd_weight > 0
            or top_k_kd_weight > 0
            or first_token_top_k_kd_weight > 0
            or first_token_kd_weight > 0
        ):
            _assert_same_vocab(source_tokenizer, candidate_tokenizer)
        tokenizer = candidate_tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if source_model:
        source = transformers.AutoModelForCausalLM.from_pretrained(
            source_model,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        )
    candidate = transformers.AutoModelForCausalLM.from_pretrained(
        candidate_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    if source is not None:
        source.to(device)
        source.eval()
    candidate.to(device)
    if train_dtype is not None:
        candidate.to(dtype=train_dtype)
    candidate.train()
    if hasattr(candidate.config, "use_cache"):
        candidate.config.use_cache = False

    trainable_params = _set_train_scope(candidate, train_scope)
    hidden_projection = None
    hidden_projection_report = None
    if hidden_kd_weight > 0 and hidden_kd_metric in PROJECTED_HIDDEN_KD_METRICS:
        if source is None:
            raise ValueError("source_model is required for projected hidden KD.")
        vocab_size = min(
            int(source.get_input_embeddings().weight.shape[0]),
            int(candidate.get_input_embeddings().weight.shape[0]),
            len(tokenizer),
        )
        anchor_count = min(int(vocab_size), int(hidden_max_anchors))
        anchor_idx = torch.linspace(0, vocab_size - 1, steps=anchor_count, device=device).round().to(torch.long)
        hidden_projection_report = _fit_shared_embedding_hidden_projection(
            torch,
            source,
            candidate,
            anchor_idx,
            anchor_idx,
            hidden_ridge,
            hidden_max_anchors,
        )
        hidden_projection = hidden_projection_report["map"].to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW([p for p in candidate.parameters() if p.requires_grad], lr=lr)
    texts = _order_recovery_texts(calibration_texts or RECOVERY_TEXTS, calibration_order)
    if not texts:
        raise ValueError("Need at least one recovery text.")
    eval_overlap_count = len(set(texts) & set(TEXT_EVALS))
    calibration_domain_counts = _recovery_domain_counts(texts) if calibration_order == "domain-balanced" else {}
    behavior_suite, behavior_probes, generation_prompts = _recovery_behavior_eval_suite(behavior_eval_suite)
    batches = _make_batches(
        tokenizer,
        texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        response_loss_mode=response_loss_mode,
        append_eos=append_eos,
    )

    losses = []
    answer_contrastive_counts = []
    interval = int(report_interval or max(1, steps // 10))
    best_loss = None
    best_step = None
    best_state = None
    behavior_interval = int(behavior_eval_interval or interval)
    behavior_history: list[dict[str, Any]] = []
    best_behavior_summary = None
    best_behavior_loss = None
    best_behavior_step = None
    best_behavior_state = None
    if save_best_behavior:
        was_training = candidate.training
        candidate.eval()
        with torch.no_grad():
            behavior_summary = _heldout_behavior_summary(
                torch,
                candidate,
                candidate_tokenizer,
                device,
                int(behavior_max_new_tokens),
                int(behavior_max_probes),
                behavior_probes,
                generation_prompts,
                behavior_max_generation_prompts,
            )
        if was_training:
            candidate.train()
        behavior_entry = _behavior_history_entry(0, None, behavior_summary)
        behavior_history.append(behavior_entry)
        _print_behavior_checkpoint("recover_behavior", behavior_entry)
        best_behavior_summary = dict(behavior_entry)
        best_behavior_loss = math.inf
        best_behavior_step = 0
        best_behavior_state = {key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()}
    for step in range(1, steps + 1):
        batch = batches[(step - 1) % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        teacher = None
        if source is not None:
            with torch.no_grad():
                teacher = source(
                    **_model_forward_inputs(batch),
                    use_cache=False,
                    output_hidden_states=hidden_kd_weight > 0,
                )
        student = candidate(
            **_model_forward_inputs(batch),
            use_cache=False,
            output_hidden_states=hidden_kd_weight > 0,
        )
        loss = _recovery_loss(
            torch,
            student,
            teacher,
            batch["input_ids"],
            batch.get("attention_mask"),
            batch.get("loss_mask"),
            kd_weight=kd_weight,
            lm_weight=lm_weight,
            unlikelihood_weight=unlikelihood_weight,
            unlikelihood_window=unlikelihood_window,
            entropy_weight=entropy_weight,
            temperature=temperature,
            top_k_kd_weight=top_k_kd_weight,
            top_k_kd_k=top_k_kd_k,
            first_token_top_k_kd_weight=first_token_top_k_kd_weight,
            first_token_kd_weight=first_token_kd_weight,
            first_token_lm_weight=first_token_lm_weight,
        )
        if hidden_kd_weight > 0:
            if teacher is None:
                raise ValueError("teacher output is required when hidden_kd_weight is positive.")
            hidden_loss = _same_tokenizer_hidden_loss(
                torch,
                student,
                teacher,
                batch.get("attention_mask"),
                batch.get("loss_mask"),
                hidden_projection,
                hidden_layer,
                hidden_kd_metric,
                hidden_token_mode,
            )
            loss = loss + float(hidden_kd_weight) * hidden_loss
        if answer_contrastive_weight > 0:
            contrastive, contrastive_count = _answer_contrastive_loss(
                torch,
                candidate,
                candidate_tokenizer,
                batch.get("raw_texts", batch.get("texts", [])),
                response_loss_mode,
                answer_contrastive_max_negatives,
                max_length,
                device,
            )
            answer_contrastive_counts.append(int(contrastive_count))
            loss = loss + float(answer_contrastive_weight) * contrastive
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError(f"Non-finite recovery loss at step {step}; aborting before save.")
        loss.backward()
        _raise_if_nonfinite_gradients(torch, candidate, step)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in candidate.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        _raise_if_nonfinite_parameters(torch, candidate, step)
        if step == 1 or step % interval == 0 or step == steps:
            loss_value = float(loss.detach().cpu())
            losses.append({"step": step, "loss": loss_value})
            print(f"recover step={step} loss={loss_value:.6f}", flush=True)
            if save_best_loss and (best_loss is None or loss_value < best_loss):
                best_loss = loss_value
                best_step = int(step)
                best_state = {key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()}
        if save_best_behavior and (step == 1 or step % behavior_interval == 0 or step == steps):
            loss_value = float(loss.detach().cpu())
            was_training = candidate.training
            candidate.eval()
            with torch.no_grad():
                behavior_summary = _heldout_behavior_summary(
                    torch,
                    candidate,
                    candidate_tokenizer,
                    device,
                    int(behavior_max_new_tokens),
                    int(behavior_max_probes),
                    behavior_probes,
                    generation_prompts,
                    behavior_max_generation_prompts,
                )
            if was_training:
                candidate.train()
            behavior_entry = _behavior_history_entry(step, loss_value, behavior_summary)
            behavior_history.append(behavior_entry)
            _print_behavior_checkpoint("recover_behavior", behavior_entry)
            if _behavior_checkpoint_is_better(behavior_entry, best_behavior_summary, best_behavior_loss):
                best_behavior_summary = dict(behavior_entry)
                best_behavior_loss = loss_value
                best_behavior_step = int(step)
                best_behavior_state = {
                    key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()
                }

    if save_best_behavior and best_behavior_state is not None:
        candidate.load_state_dict(best_behavior_state)
    elif save_best_loss and best_state is not None:
        candidate.load_state_dict(best_state)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    report = {
        "operation": "recover_causal_lm",
        "source_model": source_model,
        "candidate_model": candidate_model,
        "output_dir": str(target),
        "steps": steps,
        "batch_size": batch_size,
        "max_length": max_length,
        "lr": lr,
        "kd_weight": kd_weight,
        "lm_weight": lm_weight,
        "hidden_kd_weight": hidden_kd_weight,
        "hidden_kd_metric": hidden_kd_metric,
        "hidden_layer": hidden_layer,
        "hidden_token_mode": hidden_token_mode,
        "hidden_ridge": hidden_ridge,
        "hidden_max_anchors": hidden_max_anchors,
        "hidden_projection": hidden_projection_report
        and {key: value for key, value in hidden_projection_report.items() if key != "map"},
        "top_k_kd_weight": top_k_kd_weight,
        "top_k_kd_k": top_k_kd_k,
        "first_token_top_k_kd_weight": first_token_top_k_kd_weight,
        "first_token_kd_weight": first_token_kd_weight,
        "first_token_lm_weight": first_token_lm_weight,
        "answer_contrastive_weight": answer_contrastive_weight,
        "answer_contrastive_max_negatives": answer_contrastive_max_negatives,
        "answer_contrastive_count_observed": answer_contrastive_counts,
        "unlikelihood_weight": unlikelihood_weight,
        "unlikelihood_window": unlikelihood_window,
        "entropy_weight": entropy_weight,
        "temperature": temperature,
        "report_interval": interval,
        "save_best_loss": save_best_loss,
        "best_loss": best_loss,
        "best_step": best_step,
        "save_best_behavior": bool(save_best_behavior),
        "behavior_eval_interval": behavior_interval,
        "behavior_max_new_tokens": int(behavior_max_new_tokens),
        "behavior_max_probes": int(behavior_max_probes),
        "behavior_eval_suite": _recovery_behavior_eval_suite_report(behavior_suite),
        "behavior_max_generation_prompts": (
            None if behavior_max_generation_prompts is None else int(behavior_max_generation_prompts)
        ),
        "behavior_history": behavior_history,
        "best_behavior_step": best_behavior_step,
        "best_behavior": best_behavior_summary,
        "teacher_used": source_model is not None
        and (
            kd_weight > 0
            or hidden_kd_weight > 0
            or top_k_kd_weight > 0
            or first_token_top_k_kd_weight > 0
            or first_token_kd_weight > 0
        ),
        "response_loss_mode": response_loss_mode,
        "append_eos": bool(append_eos),
        "train_scope": train_scope,
        "candidate_train_dtype": candidate_train_dtype,
        "trainable_param_count": trainable_params,
        "calibration_order": calibration_order,
        "calibration_domain_counts": calibration_domain_counts,
        "calibration_text_count": len(texts),
        "eval_overlap_count": eval_overlap_count,
        "losses": losses,
        "param_count": sum(param.numel() for param in candidate.parameters()),
    }
    _write_json(target / "neural_forge_recovery_report.json", report)
    return report


def _nonfinite_gradient_names(torch, model, limit: int = 5) -> list[str]:
    bad = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if not bool(torch.isfinite(param.grad).all().detach().cpu()):
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def _raise_if_nonfinite_gradients(torch, model, step: int) -> None:
    bad = _nonfinite_gradient_names(torch, model)
    if bad:
        joined = ", ".join(bad)
        raise RuntimeError(f"Non-finite recovery gradients at step {step}: {joined}")


def _nonfinite_parameter_names(torch, model, limit: int = 5) -> list[str]:
    bad = []
    for name, param in model.named_parameters():
        if not bool(torch.isfinite(param.data).all().detach().cpu()):
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def _raise_if_nonfinite_parameters(torch, model, step: int) -> None:
    bad = _nonfinite_parameter_names(torch, model)
    if bad:
        joined = ", ".join(bad)
        raise RuntimeError(f"Non-finite recovery parameters after step {step}: {joined}")


def _load_cross_tokenizer_recovery_tokenizers(
    transformers,
    *,
    source_model: str | Path,
    candidate_model: str | Path,
    candidate_tokenizer_model: str | Path | None,
    trust_remote_code: bool,
    local_files_only: bool,
):
    """Load recovery tokenizers, allowing a model-independent candidate source."""

    tokenizer_source = (
        candidate_model
        if candidate_tokenizer_model is None
        else candidate_tokenizer_model
    )
    source_tokenizer = transformers.AutoTokenizer.from_pretrained(
        source_model,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    candidate_tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return source_tokenizer, candidate_tokenizer, str(tokenizer_source)


def recover_cross_tokenizer_causal_lm(
    source_model: str,
    candidate_model: str,
    output_dir: str | Path,
    steps: int = 40,
    batch_size: int = 1,
    max_length: int = 96,
    lr: float = 5e-5,
    anchor_kd_weight: float = 0.5,
    lm_weight: float = 0.5,
    first_token_lm_weight: float = 0.0,
    response_end_lm_weight: float = 0.0,
    focused_lm_weight: float = 0.0,
    hidden_kd_weight: float = 0.0,
    span_kd_weight: float = 0.0,
    span_hidden_kd_weight: float = 0.0,
    dual_space_kd_weight: float = 0.0,
    attention_influence_weight: float = 0.0,
    attention_influence_layer: str = "last",
    projected_vocab_kd_weight: float = 0.0,
    focused_projected_vocab_kd_weight: float = 0.0,
    projected_vocab_kd_mode: str = "teacher-to-student",
    projected_vocab_max_source_tokens: int = 8,
    reference_model: str | None = None,
    reference_kd_weight: float = 0.0,
    unlikelihood_weight: float = 0.0,
    unlikelihood_window: int = 16,
    entropy_weight: float = 0.0,
    entropy_aware_kd_weight: float = 0.0,
    entropy_aware_kd_threshold: float = 0.55,
    entropy_aware_kd_sharpness: float = 8.0,
    span_max_pairs: int = 512,
    hidden_ridge: float = 1e-3,
    hidden_layer: str = "last",
    hidden_kd_metric: str = "projected",
    hidden_max_anchors: int = 2048,
    temperature: float = 2.0,
    max_anchors: int = 20_000,
    anchor_selection: str = "stratified",
    anchor_min_chars: int = 1,
    train_scope: str = "last1",
    torch_dtype: str = "float32",
    candidate_torch_dtype: str = "float32",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    calibration_texts: Sequence[str] | None = None,
    calibration_order: str = "original",
    response_loss_mode: str = "none",
    token_kd_weight: float = 0.0,
    token_focus: str = "auto",
    token_max_pairs: int = 512,
    token_teacher_confidence_floor: float = 0.0,
    alm_kd_weight: float = 0.0,
    alm_max_chunks: int = 64,
    alm_min_chars: int = 2,
    answer_contrastive_weight: float = 0.0,
    answer_contrastive_max_negatives: int = 4,
    answer_first_token_margin_weight: float = 0.0,
    answer_first_token_margin: float = 1.0,
    answer_first_token_max_negatives: int = 8,
    answer_first_token_top_negatives: int = 0,
    teacher_rank_weight: float = 0.0,
    teacher_rank_temperature: float = 1.0,
    teacher_rank_max_negatives: int = 4,
    teacher_generate_lm_weight: float = 0.0,
    teacher_generate_max_new_tokens: int = 64,
    teacher_generate_loss_mode: str = "none",
    stepwise_donor_kd_weight: float = 0.0,
    stepwise_donor_max_new_tokens: int = 16,
    on_policy_anchor_kd_weight: float = 0.0,
    on_policy_unlikelihood_weight: float = 0.0,
    on_policy_hidden_kd_weight: float = 0.0,
    on_policy_max_new_tokens: int = 24,
    on_policy_interval: int = 10,
    append_eos: bool = False,
    grad_clip: float = 1.0,
    report_interval: int | None = None,
    save_best_loss: bool = False,
    save_best_behavior: bool = False,
    behavior_eval_interval: int | None = None,
    behavior_max_new_tokens: int = 96,
    behavior_max_probes: int = 6,
    behavior_eval_suite: str | Path | dict[str, Any] | None = None,
    behavior_max_generation_prompts: int | None = None,
    precomputed_teacher_generated_texts: Sequence[str] | None = None,
    on_policy_replay_bank: Sequence[Sequence[str]] | None = None,
    behavior_eval_checkpoint_steps: Sequence[int] | None = None,
    behavior_checkpoint_observer: Callable[[Mapping[str, Any]], None] | None = None,
    adamw_foreach: bool | None = None,
    stepwise_replay_bank: Sequence[Mapping[str, Any]] | None = None,
    candidate_tokenizer_model: str | Path | None = None,
) -> dict[str, Any]:
    """Recover a cross-tokenizer folded child against donor anchor logits.

    Tokenizers are not assumed to align position-by-position. Instead, shared
    decoded-token anchors define a common logit subspace. For each calibration
    text, the teacher and child next-token logits are averaged over valid
    positions in their own tokenization, then KL is computed over the shared
    anchor vocabulary. Candidate-token LM loss remains available as a stabilizer.
    """

    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    _validate_blockwise_learning_rate(float(lr), int(steps))
    if (
        anchor_kd_weight < 0
        or lm_weight < 0
        or first_token_lm_weight < 0
        or response_end_lm_weight < 0
        or focused_lm_weight < 0
        or hidden_kd_weight < 0
        or span_kd_weight < 0
        or span_hidden_kd_weight < 0
        or dual_space_kd_weight < 0
        or attention_influence_weight < 0
        or projected_vocab_kd_weight < 0
        or focused_projected_vocab_kd_weight < 0
        or reference_kd_weight < 0
        or unlikelihood_weight < 0
        or entropy_weight < 0
        or token_kd_weight < 0
        or alm_kd_weight < 0
        or answer_contrastive_weight < 0
        or answer_first_token_margin_weight < 0
        or teacher_rank_weight < 0
        or teacher_generate_lm_weight < 0
        or stepwise_donor_kd_weight < 0
        or on_policy_anchor_kd_weight < 0
        or on_policy_unlikelihood_weight < 0
        or on_policy_hidden_kd_weight < 0
    ):
        raise ValueError("loss weights must be non-negative.")
    if teacher_generate_max_new_tokens <= 0:
        raise ValueError("teacher_generate_max_new_tokens must be positive.")
    if teacher_generate_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("teacher_generate_loss_mode must be none, auto, assistant, or answer.")
    if stepwise_donor_max_new_tokens <= 0:
        raise ValueError("stepwise_donor_max_new_tokens must be positive.")
    if token_focus not in {"auto", "response", "numbers", "code", "all"}:
        raise ValueError("token_focus must be auto, response, numbers, code, or all.")
    if token_max_pairs <= 0:
        raise ValueError("token_max_pairs must be positive.")
    if not 0.0 <= float(token_teacher_confidence_floor) <= 1.0:
        raise ValueError("token_teacher_confidence_floor must be in [0, 1].")
    if alm_max_chunks <= 0:
        raise ValueError("alm_max_chunks must be positive.")
    if alm_min_chars <= 0:
        raise ValueError("alm_min_chars must be positive.")
    if answer_contrastive_max_negatives <= 0:
        raise ValueError("answer_contrastive_max_negatives must be positive.")
    if answer_first_token_max_negatives <= 0:
        raise ValueError("answer_first_token_max_negatives must be positive.")
    if answer_first_token_top_negatives < 0:
        raise ValueError("answer_first_token_top_negatives must be non-negative.")
    if answer_first_token_margin <= 0:
        raise ValueError("answer_first_token_margin must be positive.")
    if teacher_rank_max_negatives <= 0:
        raise ValueError("teacher_rank_max_negatives must be positive.")
    if teacher_rank_temperature <= 0:
        raise ValueError("teacher_rank_temperature must be positive.")
    if on_policy_max_new_tokens <= 0:
        raise ValueError("on_policy_max_new_tokens must be positive.")
    if on_policy_interval <= 0:
        raise ValueError("on_policy_interval must be positive.")
    if not 0.0 <= float(entropy_aware_kd_weight) <= 1.0:
        raise ValueError("entropy_aware_kd_weight must be in [0, 1].")
    if not 0.0 <= float(entropy_aware_kd_threshold) <= 1.0:
        raise ValueError("entropy_aware_kd_threshold must be in [0, 1].")
    if entropy_aware_kd_sharpness <= 0:
        raise ValueError("entropy_aware_kd_sharpness must be positive.")
    if (
        anchor_kd_weight == 0
        and lm_weight == 0
        and first_token_lm_weight == 0
        and response_end_lm_weight == 0
        and focused_lm_weight == 0
        and hidden_kd_weight == 0
        and span_kd_weight == 0
        and span_hidden_kd_weight == 0
        and dual_space_kd_weight == 0
        and attention_influence_weight == 0
        and projected_vocab_kd_weight == 0
        and focused_projected_vocab_kd_weight == 0
        and reference_kd_weight == 0
        and unlikelihood_weight == 0
        and entropy_weight == 0
        and token_kd_weight == 0
        and alm_kd_weight == 0
        and answer_contrastive_weight == 0
        and answer_first_token_margin_weight == 0
        and teacher_rank_weight == 0
        and teacher_generate_lm_weight == 0
        and stepwise_donor_kd_weight == 0
        and on_policy_anchor_kd_weight == 0
        and on_policy_unlikelihood_weight == 0
        and on_policy_hidden_kd_weight == 0
    ):
        raise ValueError("At least one loss weight must be positive.")
    if unlikelihood_window <= 0:
        raise ValueError("unlikelihood_window must be positive.")
    if reference_kd_weight > 0 and not reference_model:
        raise ValueError("reference_model is required when reference_kd_weight is positive.")
    if span_max_pairs <= 0:
        raise ValueError("span_max_pairs must be positive.")
    if projected_vocab_max_source_tokens <= 0:
        raise ValueError("projected_vocab_max_source_tokens must be positive.")
    if projected_vocab_kd_mode not in {"teacher-to-student", "student-to-teacher", "both"}:
        raise ValueError("projected_vocab_kd_mode must be teacher-to-student, student-to-teacher, or both.")
    if hidden_ridge < 0:
        raise ValueError("hidden_ridge must be non-negative.")
    if hidden_layer not in {"last", "middle"}:
        raise ValueError("hidden_layer must be last or middle.")
    if attention_influence_layer not in {"last", "middle"}:
        raise ValueError("attention_influence_layer must be last or middle.")
    if hidden_kd_metric not in HIDDEN_KD_METRICS:
        raise ValueError(f"hidden_kd_metric must be {HIDDEN_KD_METRIC_DESCRIPTION}.")
    if hidden_max_anchors <= 0:
        raise ValueError("hidden_max_anchors must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    if response_loss_mode == "none" and (
        first_token_lm_weight > 0 or response_end_lm_weight > 0 or answer_first_token_margin_weight > 0
    ):
        raise ValueError("response-boundary losses require response_loss_mode to be auto, assistant, or answer.")
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")
    if anchor_selection not in {"shortest", "longest", "stratified"}:
        raise ValueError("anchor_selection must be shortest, longest, or stratified.")
    if anchor_min_chars <= 0:
        raise ValueError("anchor_min_chars must be positive.")
    if grad_clip < 0:
        raise ValueError("grad_clip must be non-negative.")
    if report_interval is not None and report_interval <= 0:
        raise ValueError("report_interval must be positive when provided.")
    if behavior_eval_interval is not None and behavior_eval_interval <= 0:
        raise ValueError("behavior_eval_interval must be positive when provided.")
    if behavior_max_new_tokens <= 0:
        raise ValueError("behavior_max_new_tokens must be positive.")
    if behavior_max_probes <= 0:
        raise ValueError("behavior_max_probes must be positive.")
    if behavior_max_generation_prompts is not None and behavior_max_generation_prompts <= 0:
        raise ValueError("behavior_max_generation_prompts must be positive when provided.")
    if calibration_order not in {"original", "domain-balanced"}:
        raise ValueError("calibration_order must be original or domain-balanced.")
    if behavior_checkpoint_observer is not None and not callable(behavior_checkpoint_observer):
        raise TypeError("behavior_checkpoint_observer must be callable when provided.")
    if adamw_foreach is not None and not isinstance(adamw_foreach, bool):
        raise TypeError("adamw_foreach must be a bool or None.")

    normalized_behavior_checkpoint_steps = _normalize_behavior_checkpoint_steps(
        behavior_eval_checkpoint_steps,
        steps=int(steps),
    )
    frozen_teacher_generated_texts = (
        None
        if precomputed_teacher_generated_texts is None
        else _freeze_ordered_recovery_texts(
            precomputed_teacher_generated_texts,
            field_name="precomputed_teacher_generated_texts",
        )
    )
    frozen_on_policy_replay_bank = (
        None if on_policy_replay_bank is None else _freeze_on_policy_replay_bank(on_policy_replay_bank)
    )
    frozen_stepwise_replay_bank = (
        None
        if stepwise_replay_bank is None
        else _freeze_stepwise_replay_bank(stepwise_replay_bank)
    )
    on_policy_enabled = bool(
        on_policy_anchor_kd_weight > 0
        or on_policy_unlikelihood_weight > 0
        or on_policy_hidden_kd_weight > 0
    )
    scheduled_on_policy_steps = tuple(
        step
        for step in range(1, int(steps) + 1)
        if on_policy_enabled and (step == 1 or step % int(on_policy_interval) == 0)
    )
    if frozen_on_policy_replay_bank is not None and len(frozen_on_policy_replay_bank) != len(
        scheduled_on_policy_steps
    ):
        raise ValueError(
            "on_policy_replay_bank must contain exactly one ordered text batch per scheduled "
            f"on-policy event ({len(scheduled_on_policy_steps)} required, "
            f"got {len(frozen_on_policy_replay_bank)})."
        )
    scheduled_stepwise_steps = tuple(
        range(1, int(steps) + 1)
    ) if stepwise_donor_kd_weight > 0 else ()
    if frozen_stepwise_replay_bank is not None:
        replay_steps = tuple(event.step for event in frozen_stepwise_replay_bank)
        if replay_steps != scheduled_stepwise_steps:
            raise ValueError(
                "stepwise_replay_bank must contain exactly one ordered token replay event for each "
                f"stepwise donor KD optimizer step (expected steps {scheduled_stepwise_steps}, "
                f"got {replay_steps})."
            )

    from neural_forge.hf_xfamily import _canonical_token
    from neural_forge.hf_xfamily import _canonical_token_map

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    candidate_dtype = _dtype_from_name(torch, candidate_torch_dtype)
    source_tokenizer, candidate_tokenizer, candidate_tokenizer_source = (
        _load_cross_tokenizer_recovery_tokenizers(
            transformers,
            source_model=source_model,
            candidate_model=candidate_model,
            candidate_tokenizer_model=candidate_tokenizer_model,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    )
    if source_tokenizer.pad_token_id is None and source_tokenizer.eos_token_id is not None:
        source_tokenizer.pad_token = source_tokenizer.eos_token
    if candidate_tokenizer.pad_token_id is None and candidate_tokenizer.eos_token_id is not None:
        candidate_tokenizer.pad_token = candidate_tokenizer.eos_token

    print(
        f"xfamily_recover phase=anchor_build max_anchors={int(max_anchors)} "
        f"selection={anchor_selection} min_chars={int(anchor_min_chars)}",
        flush=True,
    )
    anchors = _shared_recovery_anchors(
        source_tokenizer,
        candidate_tokenizer,
        max_anchors,
        _canonical_token_map,
        selection=anchor_selection,
        min_chars=anchor_min_chars,
    )
    if not anchors:
        raise ValueError("No shared decoded-token anchors found for cross-tokenizer recovery.")
    print(f"xfamily_recover phase=anchor_build_done anchors={len(anchors)}", flush=True)
    source_anchor_idx = torch.tensor([item[1] for item in anchors], dtype=torch.long, device=device)
    candidate_anchor_idx = torch.tensor([item[2] for item in anchors], dtype=torch.long, device=device)
    projected_vocab_groups = None
    projected_vocab_mask = None
    projected_vocab_weights = None
    projected_vocab_valid = None
    projected_vocab_group_report = None
    if projected_vocab_kd_weight > 0 or focused_projected_vocab_kd_weight > 0 or stepwise_donor_kd_weight > 0:
        print(
            "xfamily_recover phase=projected_vocab_group_build "
            f"max_source_tokens={int(projected_vocab_max_source_tokens)}",
            flush=True,
        )
        (
            projected_vocab_groups,
            projected_vocab_mask,
            projected_vocab_weights,
            projected_vocab_valid,
        ) = _candidate_to_source_token_groups(
            torch,
            source_tokenizer,
            candidate_tokenizer,
            max_source_tokens=projected_vocab_max_source_tokens,
            device=device,
        )
        projected_vocab_group_report = _projected_vocab_group_summary(
            torch,
            candidate_tokenizer,
            projected_vocab_mask,
            projected_vocab_valid,
            projected_vocab_weights,
        )
        print(
            "xfamily_recover phase=projected_vocab_group_build_done "
            f"valid_rows={projected_vocab_group_report['valid_rows']} "
            f"multi_source_rows={projected_vocab_group_report['multi_source_rows']} "
            f"digit_valid_rows={projected_vocab_group_report['digit_valid_rows']} "
            f"code_valid_rows={projected_vocab_group_report['code_valid_rows']}",
            flush=True,
        )

    print("xfamily_recover phase=model_load", flush=True)
    attention_load_kwargs = {"attn_implementation": "eager"} if attention_influence_weight > 0 else {}
    source = transformers.AutoModelForCausalLM.from_pretrained(
        source_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        **attention_load_kwargs,
    )
    candidate = transformers.AutoModelForCausalLM.from_pretrained(
        candidate_model,
        torch_dtype=candidate_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        **attention_load_kwargs,
    )
    reference = None
    if reference_kd_weight > 0:
        reference_tokenizer = transformers.AutoTokenizer.from_pretrained(
            reference_model,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if hasattr(reference_tokenizer, "get_vocab") and hasattr(candidate_tokenizer, "get_vocab"):
            if reference_tokenizer.get_vocab() != candidate_tokenizer.get_vocab():
                raise ValueError("reference_model tokenizer must match candidate tokenizer for reference KD.")
        reference = transformers.AutoModelForCausalLM.from_pretrained(
            reference_model,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    print("xfamily_recover phase=model_load_done", flush=True)
    source.to(device)
    candidate.to(device)
    if reference is not None:
        reference.to(device)
    source.eval()
    if reference is not None:
        reference.eval()
    candidate.train()
    if hasattr(source.config, "use_cache"):
        source.config.use_cache = False
    if reference is not None and hasattr(reference.config, "use_cache"):
        reference.config.use_cache = False
    if hasattr(candidate.config, "use_cache"):
        candidate.config.use_cache = False

    hidden_projection_report = None
    hidden_projection = None
    reverse_hidden_projection_report = None
    reverse_hidden_projection = None
    needs_projected_hidden = (hidden_kd_metric in PROJECTED_HIDDEN_KD_METRICS and (
        hidden_kd_weight > 0 or span_hidden_kd_weight > 0 or on_policy_hidden_kd_weight > 0
    )) or dual_space_kd_weight > 0
    if needs_projected_hidden:
        hidden_projection_report = _fit_shared_embedding_hidden_projection(
            torch,
            source,
            candidate,
            source_anchor_idx,
            candidate_anchor_idx,
            ridge=hidden_ridge,
            max_anchors=hidden_max_anchors,
        )
        hidden_projection = hidden_projection_report["map"].to(device=device, dtype=torch.float32)
    if dual_space_kd_weight > 0:
        reverse_hidden_projection_report = _fit_shared_embedding_hidden_projection(
            torch,
            candidate,
            source,
            candidate_anchor_idx,
            source_anchor_idx,
            ridge=hidden_ridge,
            max_anchors=hidden_max_anchors,
        )
        reverse_hidden_projection = reverse_hidden_projection_report["map"].to(device=device, dtype=torch.float32)

    trainable_params = _set_train_scope(candidate, train_scope)
    optimizer_kwargs: dict[str, Any] = {"lr": lr}
    if adamw_foreach is not None:
        optimizer_kwargs["foreach"] = bool(adamw_foreach)
    optimizer = torch.optim.AdamW(
        [p for p in candidate.parameters() if p.requires_grad],
        **optimizer_kwargs,
    )
    texts = _order_recovery_texts(calibration_texts or RECOVERY_TEXTS, calibration_order)
    if not texts:
        raise ValueError("Need at least one recovery text.")
    eval_overlap_count = len(set(texts) & set(TEXT_EVALS))
    calibration_domain_counts = _recovery_domain_counts(texts) if calibration_order == "domain-balanced" else {}
    behavior_suite, behavior_probes, generation_prompts = _recovery_behavior_eval_suite(behavior_eval_suite)
    batches = _make_paired_batches(
        source_tokenizer,
        candidate_tokenizer,
        texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        include_offsets=(
            span_kd_weight > 0
            or span_hidden_kd_weight > 0
            or dual_space_kd_weight > 0
            or projected_vocab_kd_weight > 0
            or focused_lm_weight > 0
            or focused_projected_vocab_kd_weight > 0
            or token_kd_weight > 0
            or alm_kd_weight > 0
            or attention_influence_weight > 0
        ),
        response_loss_mode=response_loss_mode,
        append_eos=append_eos,
    )
    stepwise_replay_preflight: dict[str, Any] | None = None
    if frozen_stepwise_replay_bank is not None:
        prompt_max_length = max(2, int(max_length) - int(stepwise_donor_max_new_tokens))
        candidate_vocab_limit = _stepwise_replay_vocab_limit(candidate, candidate_tokenizer)
        preflight_events: list[dict[str, Any]] = []
        for event_index, event in enumerate(frozen_stepwise_replay_bank):
            batch_index = (int(event.step) - 1) % len(batches)
            _source_batch, scheduled_candidate_batch = batches[batch_index]
            scheduled_raw_texts = tuple(
                str(text)
                for text in scheduled_candidate_batch.get(
                    "raw_texts",
                    scheduled_candidate_batch.get("texts", ()),
                )
            )
            scheduled_prompts = tuple(
                prompt
                for prompt in (
                    _on_policy_prompt_text(text, response_loss_mode)
                    for text in scheduled_raw_texts
                )
                if prompt.strip()
            )
            generated, generated_mask, prompt_lengths, rollout_lengths = (
                _validated_stepwise_replay_tensors(
                    torch,
                    candidate,
                    candidate_tokenizer,
                    scheduled_prompts,
                    event,
                    prompt_max_length=prompt_max_length,
                    max_new_tokens=int(stepwise_donor_max_new_tokens),
                    max_length=int(max_length),
                    device=device,
                )
            )
            del generated, generated_mask
            continuation_counts = [
                rollout_length - prompt_length
                for prompt_length, rollout_length in zip(
                    prompt_lengths,
                    rollout_lengths,
                    strict=True,
                )
            ]
            preflight_events.append(
                {
                    "event_index": int(event_index),
                    "optimizer_step": int(event.step),
                    "scheduled_batch_index": int(batch_index),
                    "row_count": len(event.rows),
                    "raw_text_count": len(scheduled_raw_texts),
                    "raw_texts_sha256": _ordered_recovery_texts_sha256(scheduled_raw_texts),
                    "prompt_texts_sha256": _ordered_recovery_texts_sha256(scheduled_prompts),
                    "prompt_token_counts": prompt_lengths,
                    "rollout_token_counts": rollout_lengths,
                    "continuation_token_counts": continuation_counts,
                    "kd_position_count": sum(continuation_counts),
                    "event_sha256": _stepwise_replay_event_sha256(event),
                }
            )
        preflight_body = {
            "schema": _STEPWISE_REPLAY_PREFLIGHT_SCHEMA,
            "status": "validated",
            "event_count": len(frozen_stepwise_replay_bank),
            "replay_bank_sha256": _stepwise_replay_bank_sha256(frozen_stepwise_replay_bank),
            "candidate_vocab_limit": int(candidate_vocab_limit),
            "prompt_max_length": int(prompt_max_length),
            "stepwise_donor_max_new_tokens": int(stepwise_donor_max_new_tokens),
            "max_length": int(max_length),
            "events": preflight_events,
        }
        stepwise_replay_preflight = {
            **preflight_body,
            "sha256": _stepwise_replay_preflight_sha256(preflight_body),
        }
    teacher_generated_texts: list[str] = []
    teacher_generated_text_source = "disabled"
    teacher_generated_batches: list[dict[str, Any]] = []
    if teacher_generate_lm_weight > 0:
        if frozen_teacher_generated_texts is not None:
            teacher_generated_texts = list(frozen_teacher_generated_texts)
            teacher_generated_text_source = "precomputed"
        else:
            teacher_generated_texts = _teacher_generated_lm_texts(
                torch,
                source,
                source_tokenizer,
                texts,
                response_loss_mode,
                max_new_tokens=teacher_generate_max_new_tokens,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
            )
            teacher_generated_text_source = "donor-generated"
        if not teacher_generated_texts:
            raise ValueError("teacher_generate_lm_weight requested but no teacher-generated training texts were supplied.")
        teacher_generated_batches = _make_batches(
            candidate_tokenizer,
            teacher_generated_texts,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            response_loss_mode=teacher_generate_loss_mode,
            append_eos=append_eos,
        )

    losses = []
    span_pair_counts = []
    token_focus_counts = []
    alm_chunk_counts = []
    answer_contrastive_counts = []
    answer_first_token_margin_counts = []
    teacher_rank_counts = []
    on_policy_counts = []
    on_policy_hidden_counts = []
    on_policy_replay_consumption: list[dict[str, Any]] = []
    on_policy_replay_event_index = 0
    stepwise_replay_consumption: list[dict[str, Any]] = []
    stepwise_replay_event_index = 0
    interval = int(report_interval or max(1, steps // 10))
    best_loss = None
    best_step = None
    best_state = None
    behavior_interval = int(behavior_eval_interval or interval)
    if normalized_behavior_checkpoint_steps is not None:
        effective_behavior_checkpoint_steps = normalized_behavior_checkpoint_steps
    elif save_best_behavior or behavior_checkpoint_observer is not None:
        effective_behavior_checkpoint_steps = tuple(
            [0]
            + [
                step
                for step in range(1, int(steps) + 1)
                if step == 1 or step % behavior_interval == 0 or step == int(steps)
            ]
        )
    else:
        effective_behavior_checkpoint_steps = ()
    behavior_checkpoint_step_set = set(effective_behavior_checkpoint_steps)
    behavior_history: list[dict[str, Any]] = []
    behavior_checkpoint_observer_calls = 0
    best_behavior_summary = None
    best_behavior_loss = None
    best_behavior_step = None
    best_behavior_state = None
    if 0 in behavior_checkpoint_step_set:
        print(
            f"xfamily_recover phase=behavior_eval step=0 probes={int(behavior_max_probes)} "
            f"generation_prompts={len(generation_prompts) if generation_prompts is not None else len(GENERATION_PROMPTS)}",
            flush=True,
        )
        was_training = candidate.training
        candidate.eval()
        with torch.no_grad():
            behavior_summary = _heldout_behavior_summary(
                torch,
                candidate,
                candidate_tokenizer,
                device,
                int(behavior_max_new_tokens),
                int(behavior_max_probes),
                behavior_probes,
                generation_prompts,
                behavior_max_generation_prompts,
            )
        if was_training:
            candidate.train()
        behavior_entry = _behavior_history_entry(0, None, behavior_summary)
        behavior_history.append(behavior_entry)
        _print_behavior_checkpoint("xfamily_recover_behavior", behavior_entry)
        _notify_recovery_checkpoint_observer(
            behavior_checkpoint_observer,
            operation="recover_cross_tokenizer_causal_lm",
            checkpoint_index=len(behavior_history) - 1,
            behavior_entry=behavior_entry,
        )
        if behavior_checkpoint_observer is not None:
            behavior_checkpoint_observer_calls += 1
        if save_best_behavior:
            best_behavior_summary = dict(behavior_entry)
            best_behavior_loss = math.inf
            best_behavior_step = 0
            best_behavior_state = {
                key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()
            }
    for step in range(1, steps + 1):
        source_batch, candidate_batch = batches[(step - 1) % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher = source(
                **_model_forward_inputs(source_batch),
                use_cache=False,
                output_hidden_states=hidden_kd_weight > 0 or span_hidden_kd_weight > 0 or dual_space_kd_weight > 0,
                output_attentions=attention_influence_weight > 0,
            )
            reference_teacher = None
            if reference is not None:
                reference_teacher = reference(
                    **_model_forward_inputs(candidate_batch),
                    use_cache=False,
                )
        student = candidate(
            **_model_forward_inputs(candidate_batch),
            use_cache=False,
            output_hidden_states=hidden_kd_weight > 0 or span_hidden_kd_weight > 0 or dual_space_kd_weight > 0,
            output_attentions=attention_influence_weight > 0,
        )
        span_pairs = None
        if span_kd_weight > 0 or span_hidden_kd_weight > 0 or dual_space_kd_weight > 0 or projected_vocab_kd_weight > 0:
            span_pairs = _cross_tokenizer_span_pairs(
                torch,
                source_tokenizer,
                candidate_tokenizer,
                source_batch,
                candidate_batch,
                _canonical_token,
                max_pairs=span_max_pairs,
            )
            span_pair_counts.append(int(span_pairs.shape[0]))
        attention_pairs = None
        attention_influence_loss = None
        if attention_influence_weight > 0:
            attention_pairs = _cross_tokenizer_span_pairs(
                torch,
                source_tokenizer,
                candidate_tokenizer,
                source_batch,
                candidate_batch,
                _canonical_token,
                max_pairs=span_max_pairs,
                shift_positions=False,
            )
            attention_influence_loss = _cross_tokenizer_attention_influence_loss(
                torch,
                student,
                teacher,
                attention_pairs,
                layer=attention_influence_layer,
            )
            loss_attention_pair_count = int(attention_pairs.shape[0])
        else:
            loss_attention_pair_count = 0
        token_focus_spans = None
        if token_kd_weight > 0 or focused_lm_weight > 0 or focused_projected_vocab_kd_weight > 0:
            token_focus_spans = _cross_tokenizer_focus_span_pairs(
                torch,
                source_batch,
                candidate_batch,
                response_loss_mode=response_loss_mode,
                token_focus=token_focus,
                max_pairs=token_max_pairs,
            )
            token_focus_counts.append(
                sum(int(item["candidate_positions"].numel()) for item in token_focus_spans)
            )
        alm_count = 0
        loss = _cross_tokenizer_recovery_loss(
            torch,
            student,
            teacher,
            candidate_batch["input_ids"],
            candidate_batch.get("attention_mask"),
            candidate_batch.get("loss_mask"),
            source_batch.get("attention_mask"),
            candidate_anchor_idx,
            source_anchor_idx,
            anchor_kd_weight,
            lm_weight,
            first_token_lm_weight,
            response_end_lm_weight,
            hidden_kd_weight,
            span_kd_weight,
            span_hidden_kd_weight,
            projected_vocab_kd_weight,
            focused_projected_vocab_kd_weight,
            projected_vocab_kd_mode,
            reference_kd_weight,
            unlikelihood_weight,
            unlikelihood_window,
            entropy_weight,
            entropy_aware_kd_weight,
            entropy_aware_kd_threshold,
            entropy_aware_kd_sharpness,
            hidden_projection,
            hidden_layer,
            hidden_kd_metric,
            span_pairs,
            projected_vocab_groups,
            projected_vocab_mask,
            projected_vocab_valid,
            reference_teacher,
            temperature,
            source_input_ids=source_batch["input_ids"],
            token_kd_weight=token_kd_weight,
            token_focus_spans=token_focus_spans,
            token_teacher_confidence_floor=token_teacher_confidence_floor,
            focused_lm_weight=focused_lm_weight,
            projected_vocab_weights=projected_vocab_weights,
        )
        if alm_kd_weight > 0:
            alm_loss, alm_count = _cross_tokenizer_alm_chunk_loss(
                torch,
                shift_student=torch.nan_to_num(student.logits[:, :-1, :].contiguous().to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4),
                teacher=teacher,
                candidate_input_ids=candidate_batch["input_ids"],
                source_input_ids=source_batch["input_ids"],
                source_batch=source_batch,
                candidate_batch=candidate_batch,
                response_loss_mode=response_loss_mode,
                max_chunks=alm_max_chunks,
                min_chars=alm_min_chars,
            )
            alm_chunk_counts.append(int(alm_count))
            loss = loss + float(alm_kd_weight) * alm_loss
        if attention_influence_loss is not None:
            loss = loss + float(attention_influence_weight) * attention_influence_loss
        if dual_space_kd_weight > 0 and span_pairs is not None and int(span_pairs.numel()) > 0:
            dual_space = _cross_tokenizer_dual_space_kd_loss(
                torch,
                student,
                teacher,
                span_pairs,
                hidden_projection,
                reverse_hidden_projection,
                source.get_output_embeddings().weight,
                candidate.get_output_embeddings().weight,
                source_anchor_idx,
                candidate_anchor_idx,
                hidden_layer,
                temperature,
            )
            loss = loss + float(dual_space_kd_weight) * dual_space
        if answer_contrastive_weight > 0:
            contrastive, contrastive_count = _answer_contrastive_loss(
                torch,
                candidate,
                candidate_tokenizer,
                candidate_batch.get("raw_texts", candidate_batch.get("texts", [])),
                response_loss_mode,
                answer_contrastive_max_negatives,
                max_length,
                device,
            )
            answer_contrastive_counts.append(int(contrastive_count))
            loss = loss + float(answer_contrastive_weight) * contrastive
        answer_first_token_margin_loss = None
        if answer_first_token_margin_weight > 0:
            answer_first_token_margin_loss, answer_margin_count = _answer_first_token_margin_loss(
                torch,
                torch.nan_to_num(
                    student.logits[:, :-1, :].contiguous().to(torch.float32),
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                ),
                candidate_batch["input_ids"],
                candidate_batch.get("attention_mask"),
                candidate_batch.get("loss_mask"),
                candidate_tokenizer,
                candidate_batch.get("raw_texts", candidate_batch.get("texts", [])),
                response_loss_mode,
                answer_first_token_max_negatives,
                answer_first_token_margin,
                top_negatives=answer_first_token_top_negatives,
            )
            answer_first_token_margin_counts.append(int(answer_margin_count))
            loss = loss + float(answer_first_token_margin_weight) * answer_first_token_margin_loss
        if teacher_rank_weight > 0:
            teacher_rank, teacher_rank_count = _teacher_ranked_candidate_loss(
                torch,
                source,
                candidate,
                source_tokenizer,
                candidate_tokenizer,
                candidate_batch.get("raw_texts", candidate_batch.get("texts", [])),
                response_loss_mode,
                teacher_rank_max_negatives,
                teacher_rank_temperature,
                max_length,
                device,
            )
            teacher_rank_counts.append(int(teacher_rank_count))
            loss = loss + float(teacher_rank_weight) * teacher_rank
        is_on_policy_step = bool(
            on_policy_enabled and (step == 1 or step % int(on_policy_interval) == 0)
        )
        on_policy_replay_texts: tuple[str, ...] | None = None
        if is_on_policy_step and frozen_on_policy_replay_bank is not None:
            on_policy_replay_texts = frozen_on_policy_replay_bank[on_policy_replay_event_index]
            on_policy_replay_consumption.append(
                {
                    "event_index": int(on_policy_replay_event_index),
                    "step": int(step),
                    "text_count": len(on_policy_replay_texts),
                    "ordered_text_sha256": _ordered_recovery_texts_sha256(on_policy_replay_texts),
                }
            )
            on_policy_replay_event_index += 1
        if (
            (on_policy_anchor_kd_weight > 0 or on_policy_unlikelihood_weight > 0)
            and is_on_policy_step
        ):
            on_policy_loss, on_policy_count = _cross_tokenizer_on_policy_rollout_loss(
                torch,
                source,
                candidate,
                source_tokenizer,
                candidate_tokenizer,
                candidate_batch.get("raw_texts", candidate_batch.get("texts", [])),
                response_loss_mode,
                int(on_policy_max_new_tokens),
                max_length,
                device,
                source_anchor_idx,
                candidate_anchor_idx,
                temperature,
                on_policy_anchor_kd_weight,
                on_policy_unlikelihood_weight,
                unlikelihood_window,
                rollout_texts=on_policy_replay_texts,
            )
            on_policy_counts.append(int(on_policy_count))
            loss = loss + on_policy_loss
        on_policy_hidden_loss = None
        on_policy_hidden_count = 0
        if on_policy_hidden_kd_weight > 0 and is_on_policy_step:
            on_policy_hidden_loss, on_policy_hidden_count = _cross_tokenizer_on_policy_hidden_loss(
                torch,
                source,
                candidate,
                source_tokenizer,
                candidate_tokenizer,
                candidate_batch.get("raw_texts", candidate_batch.get("texts", [])),
                response_loss_mode,
                int(on_policy_max_new_tokens),
                max_length,
                device,
                hidden_projection,
                hidden_layer,
                hidden_kd_metric,
                span_max_pairs,
                rollout_texts=on_policy_replay_texts,
            )
            on_policy_hidden_counts.append(int(on_policy_hidden_count))
            loss = loss + float(on_policy_hidden_kd_weight) * on_policy_hidden_loss
        teacher_generate_loss = None
        if teacher_generated_batches:
            teacher_batch = teacher_generated_batches[(step - 1) % len(teacher_generated_batches)]
            teacher_student = candidate(
                **_model_forward_inputs(teacher_batch),
                use_cache=False,
            )
            teacher_generate_loss = _candidate_lm_stabilizer_loss(
                torch,
                teacher_student,
                teacher_batch["input_ids"],
                teacher_batch.get("attention_mask"),
            )
            loss = loss + float(teacher_generate_lm_weight) * teacher_generate_loss
        stepwise_donor_loss = None
        stepwise_donor_count = 0
        if stepwise_donor_kd_weight > 0:
            if (
                projected_vocab_groups is None
                or projected_vocab_mask is None
                or projected_vocab_weights is None
                or projected_vocab_valid is None
            ):
                raise ValueError("projected vocab maps are required when stepwise donor KD is positive.")
            stepwise_replay_event: _FrozenStepwiseReplayEvent | None = None
            if frozen_stepwise_replay_bank is not None:
                stepwise_replay_event = frozen_stepwise_replay_bank[stepwise_replay_event_index]
                if stepwise_replay_event.step != int(step):
                    raise RuntimeError("Frozen stepwise token replay event order diverged during training.")
                stepwise_replay_consumption.append(
                    {
                        "event_index": int(stepwise_replay_event_index),
                        "step": int(step),
                        "row_count": len(stepwise_replay_event.rows),
                        "prompt_token_counts": [
                            len(row.prompt_input_ids) for row in stepwise_replay_event.rows
                        ],
                        "continuation_token_counts": [
                            len(row.rollout_input_ids) - len(row.prompt_input_ids)
                            for row in stepwise_replay_event.rows
                        ],
                        "event_sha256": _stepwise_replay_event_sha256(stepwise_replay_event),
                    }
                )
                stepwise_replay_event_index += 1
            stepwise_donor_loss, stepwise_donor_count = _cross_tokenizer_stepwise_prefix_kd_loss(
                torch,
                source,
                candidate,
                source_tokenizer,
                candidate_tokenizer,
                candidate_batch.get("raw_texts", candidate_batch.get("texts", [])),
                response_loss_mode,
                int(stepwise_donor_max_new_tokens),
                max_length,
                device,
                projected_vocab_groups,
                projected_vocab_mask,
                projected_vocab_weights,
                projected_vocab_valid,
                temperature,
                replay_event=stepwise_replay_event,
            )
            if stepwise_replay_event is not None:
                expected_stepwise_positions = sum(
                    len(row.rollout_input_ids) - len(row.prompt_input_ids)
                    for row in stepwise_replay_event.rows
                )
                if int(stepwise_donor_count) != int(expected_stepwise_positions):
                    raise RuntimeError(
                        "Frozen stepwise token replay did not produce exactly one KD position per "
                        "continuation token."
                    )
                stepwise_replay_consumption[-1]["kd_position_count"] = int(stepwise_donor_count)
            loss = loss + float(stepwise_donor_kd_weight) * stepwise_donor_loss
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise RuntimeError(f"Non-finite cross-tokenizer recovery loss at step {step}; aborting before save.")
        loss.backward()
        _raise_if_nonfinite_gradients(torch, candidate, step)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in candidate.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        _raise_if_nonfinite_parameters(torch, candidate, step)
        if step == 1 or step % interval == 0 or step == steps:
            loss_value = float(loss.detach().cpu())
            teacher_generate_loss_value = (
                None if teacher_generate_loss is None else float(teacher_generate_loss.detach().cpu())
            )
            stepwise_donor_loss_value = (
                None if stepwise_donor_loss is None else float(stepwise_donor_loss.detach().cpu())
            )
            on_policy_hidden_loss_value = (
                None if on_policy_hidden_loss is None else float(on_policy_hidden_loss.detach().cpu())
            )
            attention_influence_loss_value = (
                None if attention_influence_loss is None else float(attention_influence_loss.detach().cpu())
            )
            answer_first_token_margin_loss_value = (
                None
                if answer_first_token_margin_loss is None
                else float(answer_first_token_margin_loss.detach().cpu())
            )
            losses.append(
                {
                    "step": step,
                    "loss": loss_value,
                    "attention_influence_loss": attention_influence_loss_value,
                    "attention_influence_pairs": int(loss_attention_pair_count),
                    "teacher_generate_lm_loss": teacher_generate_loss_value,
                    "stepwise_donor_kd_loss": stepwise_donor_loss_value,
                    "stepwise_donor_count": int(stepwise_donor_count),
                    "on_policy_hidden_kd_loss": on_policy_hidden_loss_value,
                    "on_policy_hidden_count": int(on_policy_hidden_count),
                    "answer_first_token_margin_loss": answer_first_token_margin_loss_value,
                    "answer_first_token_margin_count": (
                        int(answer_first_token_margin_counts[-1]) if answer_first_token_margin_counts else 0
                    ),
                }
            )
            print(
                f"xfamily_recover step={step} loss={loss_value:.6f} "
                f"span_pairs={span_pair_counts[-1] if span_pair_counts else 0} "
                f"token_focus={token_focus_counts[-1] if token_focus_counts else 0} "
                f"alm_chunks={alm_chunk_counts[-1] if alm_chunk_counts else 0} "
                f"answer_contrastive={answer_contrastive_counts[-1] if answer_contrastive_counts else 0} "
                f"answer_margin={answer_first_token_margin_counts[-1] if answer_first_token_margin_counts else 0} "
                f"teacher_rank={teacher_rank_counts[-1] if teacher_rank_counts else 0} "
                f"on_policy={on_policy_counts[-1] if on_policy_counts else 0} "
                f"on_policy_hidden={on_policy_hidden_counts[-1] if on_policy_hidden_counts else 0} "
                f"attention_influence={attention_influence_loss_value if attention_influence_loss_value is not None else 0.0:.6f} "
                f"attention_pairs={int(loss_attention_pair_count)} "
                f"teacher_generate_lm={teacher_generate_loss_value if teacher_generate_loss_value is not None else 0.0:.6f} "
                f"stepwise_donor={stepwise_donor_loss_value if stepwise_donor_loss_value is not None else 0.0:.6f} "
                f"stepwise_donor_count={int(stepwise_donor_count)}",
                flush=True,
            )
            if save_best_loss and (best_loss is None or loss_value < best_loss):
                best_loss = loss_value
                best_step = int(step)
                best_state = {key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()}
        if step in behavior_checkpoint_step_set:
            loss_value = float(loss.detach().cpu())
            print(
                f"xfamily_recover phase=behavior_eval step={int(step)} probes={int(behavior_max_probes)} "
                f"generation_prompts={len(generation_prompts) if generation_prompts is not None else len(GENERATION_PROMPTS)}",
                flush=True,
            )
            was_training = candidate.training
            candidate.eval()
            with torch.no_grad():
                behavior_summary = _heldout_behavior_summary(
                    torch,
                    candidate,
                    candidate_tokenizer,
                    device,
                    int(behavior_max_new_tokens),
                    int(behavior_max_probes),
                    behavior_probes,
                    generation_prompts,
                    behavior_max_generation_prompts,
                )
            if was_training:
                candidate.train()
            behavior_entry = _behavior_history_entry(step, loss_value, behavior_summary)
            behavior_history.append(behavior_entry)
            _print_behavior_checkpoint("xfamily_recover_behavior", behavior_entry)
            _notify_recovery_checkpoint_observer(
                behavior_checkpoint_observer,
                operation="recover_cross_tokenizer_causal_lm",
                checkpoint_index=len(behavior_history) - 1,
                behavior_entry=behavior_entry,
            )
            if behavior_checkpoint_observer is not None:
                behavior_checkpoint_observer_calls += 1
            if save_best_behavior and _behavior_checkpoint_is_better(
                behavior_entry,
                best_behavior_summary,
                best_behavior_loss,
            ):
                best_behavior_summary = dict(behavior_entry)
                best_behavior_loss = loss_value
                best_behavior_step = int(step)
                best_behavior_state = {
                    key: value.detach().cpu().clone() for key, value in candidate.state_dict().items()
                }

    if frozen_on_policy_replay_bank is not None and on_policy_replay_event_index != len(
        frozen_on_policy_replay_bank
    ):
        raise RuntimeError("Frozen on-policy replay bank was not consumed exactly once in scheduled order.")
    if frozen_stepwise_replay_bank is not None and stepwise_replay_event_index != len(
        frozen_stepwise_replay_bank
    ):
        raise RuntimeError("Frozen stepwise replay bank was not consumed exactly once in scheduled order.")

    if save_best_behavior and best_behavior_state is not None:
        candidate.load_state_dict(best_behavior_state)
    elif save_best_loss and best_state is not None:
        candidate.load_state_dict(best_state)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(target, safe_serialization=True)
    candidate_tokenizer.save_pretrained(target)
    report = {
        "operation": "recover_cross_tokenizer_causal_lm",
        "source_model": source_model,
        "candidate_model": candidate_model,
        "candidate_tokenizer_model": candidate_tokenizer_source,
        "output_dir": str(target),
        "steps": steps,
        "batch_size": batch_size,
        "max_length": max_length,
        "lr": lr,
        "anchor_kd_weight": anchor_kd_weight,
        "lm_weight": lm_weight,
        "first_token_lm_weight": first_token_lm_weight,
        "response_end_lm_weight": response_end_lm_weight,
        "focused_lm_weight": focused_lm_weight,
        "hidden_kd_weight": hidden_kd_weight,
        "span_kd_weight": span_kd_weight,
        "span_hidden_kd_weight": span_hidden_kd_weight,
        "dual_space_kd_weight": dual_space_kd_weight,
        "attention_influence_weight": attention_influence_weight,
        "attention_influence_layer": attention_influence_layer,
        "projected_vocab_kd_weight": projected_vocab_kd_weight,
        "focused_projected_vocab_kd_weight": focused_projected_vocab_kd_weight,
        "projected_vocab_kd_mode": projected_vocab_kd_mode,
        "projected_vocab_max_source_tokens": projected_vocab_max_source_tokens,
        "projected_vocab_group": projected_vocab_group_report,
        "reference_model": reference_model,
        "reference_kd_weight": reference_kd_weight,
        "unlikelihood_weight": unlikelihood_weight,
        "unlikelihood_window": unlikelihood_window,
        "entropy_weight": entropy_weight,
        "response_loss_mode": response_loss_mode,
        "token_kd_weight": token_kd_weight,
        "token_focus": token_focus,
        "token_max_pairs": token_max_pairs,
        "token_teacher_confidence_floor": token_teacher_confidence_floor,
        "alm_kd_weight": alm_kd_weight,
        "alm_max_chunks": alm_max_chunks,
        "alm_min_chars": alm_min_chars,
        "alm_chunk_count_observed": alm_chunk_counts,
        "answer_contrastive_weight": answer_contrastive_weight,
        "answer_contrastive_max_negatives": answer_contrastive_max_negatives,
        "answer_contrastive_count_observed": answer_contrastive_counts,
        "answer_first_token_margin_weight": answer_first_token_margin_weight,
        "answer_first_token_margin": answer_first_token_margin,
        "answer_first_token_max_negatives": answer_first_token_max_negatives,
        "answer_first_token_top_negatives": answer_first_token_top_negatives,
        "answer_first_token_margin_count_observed": answer_first_token_margin_counts,
        "teacher_rank_weight": teacher_rank_weight,
        "teacher_rank_temperature": teacher_rank_temperature,
        "teacher_rank_max_negatives": teacher_rank_max_negatives,
        "teacher_rank_count_observed": teacher_rank_counts,
        "teacher_generate_lm_weight": teacher_generate_lm_weight,
        "teacher_generate_max_new_tokens": teacher_generate_max_new_tokens,
        "teacher_generate_loss_mode": teacher_generate_loss_mode,
        "teacher_generated_text_source": teacher_generated_text_source,
        "teacher_generated_text_count": len(teacher_generated_texts),
        "teacher_generated_text_samples": teacher_generated_texts[:3],
        "teacher_generated_ordered_text_sha256": (
            _ordered_recovery_texts_sha256(teacher_generated_texts)
            if teacher_generated_texts
            else None
        ),
        "calibration_order": calibration_order,
        "calibration_domain_counts": calibration_domain_counts,
        "stepwise_donor_kd_weight": stepwise_donor_kd_weight,
        "stepwise_donor_max_new_tokens": stepwise_donor_max_new_tokens,
        "stepwise_scheduled_steps": list(scheduled_stepwise_steps),
        "stepwise_replay_bank_schema": (
            _STEPWISE_REPLAY_SCHEMA if frozen_stepwise_replay_bank is not None else None
        ),
        "stepwise_replay_bank_supplied": frozen_stepwise_replay_bank is not None,
        "stepwise_replay_bank_event_count": (
            0 if frozen_stepwise_replay_bank is None else len(frozen_stepwise_replay_bank)
        ),
        "stepwise_replay_bank_sha256": (
            None
            if frozen_stepwise_replay_bank is None
            else _stepwise_replay_bank_sha256(frozen_stepwise_replay_bank)
        ),
        "stepwise_replay_preflight": stepwise_replay_preflight,
        "stepwise_replay_consumption": stepwise_replay_consumption,
        "stepwise_dynamic_candidate_generation": bool(
            stepwise_donor_kd_weight > 0 and frozen_stepwise_replay_bank is None
        ),
        "on_policy_anchor_kd_weight": on_policy_anchor_kd_weight,
        "on_policy_unlikelihood_weight": on_policy_unlikelihood_weight,
        "on_policy_hidden_kd_weight": on_policy_hidden_kd_weight,
        "on_policy_max_new_tokens": on_policy_max_new_tokens,
        "on_policy_interval": on_policy_interval,
        "on_policy_scheduled_steps": list(scheduled_on_policy_steps),
        "on_policy_replay_bank_supplied": frozen_on_policy_replay_bank is not None,
        "on_policy_replay_bank_event_count": (
            0 if frozen_on_policy_replay_bank is None else len(frozen_on_policy_replay_bank)
        ),
        "on_policy_replay_bank_sha256": (
            None
            if frozen_on_policy_replay_bank is None
            else _on_policy_replay_bank_sha256(frozen_on_policy_replay_bank)
        ),
        "on_policy_replay_consumption": on_policy_replay_consumption,
        "on_policy_dynamic_candidate_generation": bool(
            on_policy_enabled and frozen_on_policy_replay_bank is None
        ),
        "on_policy_count_observed": on_policy_counts,
        "on_policy_hidden_count_observed": on_policy_hidden_counts,
        "append_eos": bool(append_eos),
        "token_focus_count_observed": token_focus_counts,
        "entropy_aware_kd_weight": entropy_aware_kd_weight,
        "entropy_aware_kd_threshold": entropy_aware_kd_threshold,
        "entropy_aware_kd_sharpness": entropy_aware_kd_sharpness,
        "report_interval": interval,
        "adamw_foreach": adamw_foreach,
        "adamw_foreach_explicit": adamw_foreach is not None,
        "save_best_loss": save_best_loss,
        "best_loss": best_loss,
        "best_step": best_step,
        "save_best_behavior": bool(save_best_behavior),
        "behavior_eval_interval": behavior_interval,
        "behavior_eval_checkpoint_steps": list(effective_behavior_checkpoint_steps),
        "behavior_checkpoint_observer_calls": int(behavior_checkpoint_observer_calls),
        "behavior_checkpoint_observer_read_only_payload": behavior_checkpoint_observer is not None,
        "behavior_max_new_tokens": int(behavior_max_new_tokens),
        "behavior_max_probes": int(behavior_max_probes),
        "behavior_eval_suite": _recovery_behavior_eval_suite_report(behavior_suite),
        "behavior_max_generation_prompts": (
            None if behavior_max_generation_prompts is None else int(behavior_max_generation_prompts)
        ),
        "behavior_history": behavior_history,
        "best_behavior_step": best_behavior_step,
        "best_behavior": best_behavior_summary,
        "span_max_pairs": span_max_pairs,
        "span_pair_count_observed": span_pair_counts,
        "hidden_ridge": hidden_ridge,
        "hidden_layer": hidden_layer,
        "hidden_kd_metric": hidden_kd_metric,
        "hidden_max_anchors": hidden_max_anchors,
        "hidden_projection": hidden_projection_report
        and {key: value for key, value in hidden_projection_report.items() if key != "map"},
        "reverse_hidden_projection": reverse_hidden_projection_report
        and {key: value for key, value in reverse_hidden_projection_report.items() if key != "map"},
        "temperature": temperature,
        "max_anchors": max_anchors,
        "anchor_selection": anchor_selection,
        "anchor_min_chars": anchor_min_chars,
        "anchors_used": len(anchors),
        "train_scope": train_scope,
        "torch_dtype": torch_dtype,
        "candidate_torch_dtype": candidate_torch_dtype,
        "local_files_only": bool(local_files_only),
        "trainable_param_count": trainable_params,
        "calibration_text_count": len(texts),
        "eval_overlap_count": eval_overlap_count,
        "losses": losses,
        "param_count": sum(param.numel() for param in candidate.parameters()),
    }
    _write_json(target / "neural_forge_xfamily_recovery_report.json", report)
    return report


def calibrate_cross_tokenizer_lm_head(
    source_model: str,
    candidate_model: str,
    output_dir: str | Path,
    ridge: float = 1e-2,
    head_blend_alpha: float = 1.0,
    head_calibration_mode: str = "full",
    response_loss_mode: str = "auto",
    answer_first_token_margin: float = 1.0,
    answer_first_token_max_negatives: int = 8,
    answer_first_token_top_negatives: int = 0,
    answer_margin_steps: int = 3,
    max_length: int = 128,
    max_calibration_rows: int = 2048,
    projected_vocab_max_source_tokens: int = 6,
    torch_dtype: str = "float16",
    candidate_torch_dtype: str = "float32",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    calibration_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit a blank/cross-tokenizer child's LM head to donor next-token logits."""

    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not 0.0 <= float(head_blend_alpha) <= 1.0:
        raise ValueError("head_blend_alpha must be in [0, 1].")
    if head_calibration_mode not in {"full", "mean-shift", "answer-margin"}:
        raise ValueError("head_calibration_mode must be full, mean-shift, or answer-margin.")
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    if answer_first_token_margin <= 0:
        raise ValueError("answer_first_token_margin must be positive.")
    if answer_first_token_max_negatives <= 0:
        raise ValueError("answer_first_token_max_negatives must be positive.")
    if answer_first_token_top_negatives < 0:
        raise ValueError("answer_first_token_top_negatives must be non-negative.")
    if answer_margin_steps <= 0:
        raise ValueError("answer_margin_steps must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if max_calibration_rows <= 0:
        raise ValueError("max_calibration_rows must be positive.")
    if projected_vocab_max_source_tokens <= 0:
        raise ValueError("projected_vocab_max_source_tokens must be positive.")

    from neural_forge.hf_xfamily import _canonical_token

    torch, transformers = _torch_and_transformers()
    source_dtype = _dtype_from_name(torch, torch_dtype)
    candidate_dtype = _dtype_from_name(torch, candidate_torch_dtype)
    candidate_tokenizer = transformers.AutoTokenizer.from_pretrained(candidate_model, trust_remote_code=trust_remote_code)
    if candidate_tokenizer.pad_token_id is None and candidate_tokenizer.eos_token_id is not None:
        candidate_tokenizer.pad_token = candidate_tokenizer.eos_token

    candidate = transformers.AutoModelForCausalLM.from_pretrained(
        candidate_model,
        torch_dtype=candidate_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    candidate.to(device)
    candidate.eval()
    if hasattr(candidate.config, "use_cache"):
        candidate.config.use_cache = False

    output_head = candidate.get_output_embeddings()
    if output_head is None:
        raise ValueError("Candidate model has no output embeddings to calibrate.")

    texts = list(calibration_texts or RECOVERY_TEXTS)
    if not texts:
        raise ValueError("Need at least one head calibration text.")
    projected_vocab_group_report = None
    valid_idx = None
    hidden_chunks = []
    logit_chunks = []
    answer_gold_ids: list[int] = []
    answer_negative_ids: list[list[int]] = []
    span_pair_counts = []
    rows_collected = 0

    if head_calibration_mode == "answer-margin":
        batches = _make_batches(
            candidate_tokenizer,
            texts,
            batch_size=1,
            max_length=max_length,
            device=device,
            include_offsets=True,
            response_loss_mode=response_loss_mode,
        )
        with torch.no_grad():
            for candidate_batch in batches:
                if rows_collected >= int(max_calibration_rows):
                    break
                student = candidate(**_model_forward_inputs(candidate_batch), use_cache=False, output_hidden_states=True)
                first_mask = _first_response_shift_mask(
                    torch,
                    candidate_batch.get("loss_mask"),
                    candidate_batch.get("attention_mask"),
                )
                if first_mask is None or int(first_mask.sum().detach().cpu().item()) <= 0:
                    span_pair_counts.append(0)
                    continue
                student_hidden = _select_hidden_state(student.hidden_states, "last")[:, :-1, :].contiguous().to(torch.float32)
                input_ids = candidate_batch["input_ids"]
                batch_texts = list(candidate_batch.get("raw_texts") or candidate_batch.get("texts") or [])
                batch_rows = 0
                batch_size = min(int(first_mask.shape[0]), len(batch_texts))
                vocab_size = int(output_head.weight.shape[0])
                for batch_idx in range(batch_size):
                    if rows_collected >= int(max_calibration_rows):
                        break
                    positions = torch.nonzero(first_mask[batch_idx], as_tuple=False).flatten()
                    if int(positions.numel()) <= 0:
                        continue
                    logit_pos = int(positions[0].detach().cpu().item())
                    label_pos = logit_pos + 1
                    if label_pos < 0 or label_pos >= int(input_ids.shape[1]):
                        continue
                    gold_id = int(input_ids[batch_idx, label_pos].detach().cpu().item())
                    if gold_id < 0 or gold_id >= vocab_size:
                        continue
                    negative_ids = _answer_first_token_negative_ids(
                        candidate_tokenizer,
                        str(batch_texts[batch_idx]),
                        response_loss_mode,
                        int(answer_first_token_max_negatives),
                        gold_id,
                        vocab_size,
                    )
                    if int(answer_first_token_top_negatives) > 0:
                        row_logits = student.logits[batch_idx, logit_pos].detach().to(torch.float32)
                        top_k = min(vocab_size, int(answer_first_token_top_negatives) + 1)
                        top_ids = torch.topk(row_logits, k=top_k, dim=-1).indices.detach().cpu().tolist()
                        for top_id in top_ids:
                            top_id = int(top_id)
                            if top_id == gold_id or top_id in negative_ids:
                                continue
                            negative_ids.append(top_id)
                            if len(negative_ids) >= int(answer_first_token_max_negatives) + int(answer_first_token_top_negatives):
                                break
                    if not negative_ids:
                        continue
                    hidden_chunks.append(student_hidden[batch_idx, logit_pos].detach().cpu().unsqueeze(0))
                    answer_gold_ids.append(int(gold_id))
                    answer_negative_ids.append([int(item) for item in negative_ids])
                    rows_collected += 1
                    batch_rows += 1
                span_pair_counts.append(int(batch_rows))
    else:
        source_tokenizer = transformers.AutoTokenizer.from_pretrained(source_model, trust_remote_code=trust_remote_code)
        if source_tokenizer.pad_token_id is None and source_tokenizer.eos_token_id is not None:
            source_tokenizer.pad_token = source_tokenizer.eos_token
        source = transformers.AutoModelForCausalLM.from_pretrained(
            source_model,
            torch_dtype=source_dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        )
        source.to(device)
        source.eval()
        if hasattr(source.config, "use_cache"):
            source.config.use_cache = False
        (
            projected_vocab_groups,
            projected_vocab_mask,
            projected_vocab_weights,
            projected_vocab_valid,
        ) = _candidate_to_source_token_groups(
            torch,
            source_tokenizer,
            candidate_tokenizer,
            max_source_tokens=projected_vocab_max_source_tokens,
            device=device,
        )
        projected_vocab_group_report = _projected_vocab_group_summary(
            torch,
            candidate_tokenizer,
            projected_vocab_mask,
            projected_vocab_valid,
            projected_vocab_weights,
        )
        valid_vocab = projected_vocab_valid.detach().cpu().bool()
        valid_idx = torch.nonzero(valid_vocab, as_tuple=False).flatten()
        if int(valid_idx.numel()) == 0:
            raise ValueError("No candidate vocab rows can be projected from source tokenizer.")

        batches = _make_paired_batches(
            source_tokenizer,
            candidate_tokenizer,
            texts,
            batch_size=1,
            max_length=max_length,
            device=device,
            include_offsets=True,
        )
        with torch.no_grad():
            for source_batch, candidate_batch in batches:
                if rows_collected >= int(max_calibration_rows):
                    break
                teacher = source(**_model_forward_inputs(source_batch), use_cache=False)
                student = candidate(**_model_forward_inputs(candidate_batch), use_cache=False, output_hidden_states=True)
                span_pairs = _cross_tokenizer_span_pairs(
                    torch,
                    source_tokenizer,
                    candidate_tokenizer,
                    source_batch,
                    candidate_batch,
                    _canonical_token,
                    max_pairs=max(1, int(max_calibration_rows) - rows_collected),
                )
                span_pair_counts.append(int(span_pairs.shape[0]))
                if int(span_pairs.numel()) == 0:
                    continue
                pair_idx = span_pairs.to(device=device, dtype=torch.long)
                batch_idx = pair_idx[:, 0]
                candidate_pos = pair_idx[:, 1]
                source_pos = pair_idx[:, 2]
                student_hidden = _select_hidden_state(student.hidden_states, "last")[:, :-1, :].contiguous().to(torch.float32)
                teacher_logits = teacher.logits[:, :-1, :].contiguous().to(torch.float32)
                rows = student_hidden[batch_idx, candidate_pos]
                projected = _project_source_logits_to_candidate_vocab(
                    torch,
                    teacher_logits[batch_idx, source_pos],
                    projected_vocab_groups,
                    projected_vocab_mask,
                    projected_vocab_valid,
                    projected_vocab_weights=projected_vocab_weights,
                )
                remain = int(max_calibration_rows) - rows_collected
                if int(rows.shape[0]) > remain:
                    rows = rows[:remain]
                    projected = projected[:remain]
                hidden_chunks.append(rows.detach().cpu())
                logit_chunks.append(projected.detach().cpu().index_select(dim=1, index=valid_idx))
                rows_collected += int(rows.shape[0])

    if rows_collected == 0:
        raise ValueError("No aligned token rows found for cross-tokenizer head calibration.")

    hidden_rows = torch.cat(hidden_chunks, dim=0).to(torch.float32)
    if head_calibration_mode == "full":
        if valid_idx is None:
            raise RuntimeError("Internal error: full head calibration requires projected vocab rows.")
        target_logits = torch.cat(logit_chunks, dim=0).to(torch.float32)
        current_head = output_head.weight.data.detach().cpu().to(torch.float32).index_select(dim=0, index=valid_idx)
        fitted_head, fit_report = _fit_lm_head_from_hidden_logits(
            torch,
            hidden_rows,
            target_logits,
            current_head,
            ridge=float(ridge),
            head_blend_alpha=float(head_blend_alpha),
        )
        updated = output_head.weight.data.detach().cpu().to(torch.float32)
        updated[valid_idx] = fitted_head
    elif head_calibration_mode == "mean-shift":
        if valid_idx is None:
            raise RuntimeError("Internal error: mean-shift head calibration requires projected vocab rows.")
        target_logits = torch.cat(logit_chunks, dim=0).to(torch.float32)
        current_head = output_head.weight.data.detach().cpu().to(torch.float32).index_select(dim=0, index=valid_idx)
        fitted_head, fit_report = _fit_lm_head_mean_shift(
            torch,
            hidden_rows,
            target_logits,
            current_head,
            ridge=float(ridge),
            head_blend_alpha=float(head_blend_alpha),
        )
        updated = output_head.weight.data.detach().cpu().to(torch.float32)
        updated[valid_idx] = fitted_head
    else:
        current_head = output_head.weight.data.detach().cpu().to(torch.float32)
        updated, fit_report = _fit_lm_head_answer_margin(
            torch,
            hidden_rows,
            answer_gold_ids,
            answer_negative_ids,
            current_head,
            ridge=float(ridge),
            head_blend_alpha=float(head_blend_alpha),
            margin=float(answer_first_token_margin),
            steps=int(answer_margin_steps),
        )
    output_head.weight.data.copy_(updated.to(device=output_head.weight.device, dtype=output_head.weight.dtype))

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(target, safe_serialization=True)
    candidate_tokenizer.save_pretrained(target)
    report = {
        "operation": "calibrate_cross_tokenizer_lm_head",
        "source_model": source_model,
        "candidate_model": candidate_model,
        "output_dir": str(target),
        "ridge": float(ridge),
        "head_blend_alpha": float(head_blend_alpha),
        "head_calibration_mode": head_calibration_mode,
        "response_loss_mode": response_loss_mode,
        "answer_first_token_margin": float(answer_first_token_margin),
        "answer_first_token_max_negatives": int(answer_first_token_max_negatives),
        "answer_first_token_top_negatives": int(answer_first_token_top_negatives),
        "answer_margin_steps": int(answer_margin_steps),
        "max_length": int(max_length),
        "max_calibration_rows": int(max_calibration_rows),
        "rows_collected": int(rows_collected),
        "projected_vocab_max_source_tokens": int(projected_vocab_max_source_tokens),
        "valid_vocab_rows": int(valid_idx.numel()) if valid_idx is not None else int(output_head.weight.shape[0]),
        "candidate_vocab_size": len(candidate_tokenizer),
        "projected_vocab_group": projected_vocab_group_report,
        "span_pair_count_observed": span_pair_counts,
        "torch_dtype": torch_dtype,
        "candidate_torch_dtype": candidate_torch_dtype,
        "calibration_text_count": len(texts),
        "fit": fit_report,
        "param_count": sum(param.numel() for param in candidate.parameters()),
    }
    _write_json(target / "neural_forge_xfamily_head_calibration_report.json", report)
    return report


def reconstruct_folded_causal_lm(
    source_model: str,
    candidate_model: str,
    output_dir: str | Path,
    layer_merge_report: str | Path | dict[str, Any] | None = None,
    ridge: float = 1e-3,
    reconstruction_alpha: float = 1.0,
    reconstruction_target: str = "consumer-input",
    reconstruction_objective: str = "ridge",
    cosine_steps: int = 80,
    cosine_lr: float = 1e-2,
    cosine_l2: float = 1e-4,
    max_length: int = 128,
    max_tokens_per_layer: int = 512,
    torch_dtype: str = "float16",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    calibration_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply GRAIL-style linear reconstruction after residual layer folding.

    This is a no-backprop compensation pass for same-tokenizer depth folds. For
    each folded layer group, it fits calibration maps from the folded candidate's
    downstream layernorm activations to the source model's corresponding
    activations, then absorbs those maps into the downstream attention and MLP
    input projections.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not 0.0 <= float(reconstruction_alpha) <= 1.0:
        raise ValueError("reconstruction_alpha must be in [0, 1].")
    if reconstruction_target not in {"consumer-input", "survivor-mlp", "both"}:
        raise ValueError("reconstruction_target must be consumer-input, survivor-mlp, or both.")
    if reconstruction_objective not in {"ridge", "cosine"}:
        raise ValueError("reconstruction_objective must be ridge or cosine.")
    if cosine_steps < 0:
        raise ValueError("cosine_steps must be non-negative.")
    if cosine_lr <= 0:
        raise ValueError("cosine_lr must be positive.")
    if cosine_l2 < 0:
        raise ValueError("cosine_l2 must be non-negative.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if max_tokens_per_layer <= 0:
        raise ValueError("max_tokens_per_layer must be positive.")

    report = _load_layer_merge_report(candidate_model, layer_merge_report)
    fold_groups = _fold_groups_from_report(report)
    if not fold_groups:
        raise ValueError("Fold reconstruction requires at least one fold-method layer group.")

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source_model, trust_remote_code=trust_remote_code)
    candidate_tokenizer = transformers.AutoTokenizer.from_pretrained(candidate_model, trust_remote_code=trust_remote_code)
    _assert_same_vocab(tokenizer, candidate_tokenizer)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    source = transformers.AutoModelForCausalLM.from_pretrained(
        source_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    candidate = transformers.AutoModelForCausalLM.from_pretrained(
        candidate_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    source.to(device)
    candidate.to(device)
    source.eval()
    candidate.eval()
    if hasattr(source.config, "use_cache"):
        source.config.use_cache = False
    if hasattr(candidate.config, "use_cache"):
        candidate.config.use_cache = False

    source_layers = _recovery_decoder_layers(source)
    candidate_layers = _recovery_decoder_layers(candidate)
    if not source_layers or not candidate_layers:
        raise ValueError("Both source and candidate need decoder layers.")

    texts = list(calibration_texts or RECOVERY_TEXTS)
    if not texts:
        raise ValueError("Need at least one reconstruction text.")
    eval_overlap_count = len(set(texts) & set(TEXT_EVALS))
    batches = _make_batches(tokenizer, texts, batch_size=1, max_length=max_length, device=device)

    repairs = []
    with torch.no_grad():
        for group in fold_groups:
            repair = _reconstruct_fold_group(
                torch,
                source,
                candidate,
                source_layers,
                candidate_layers,
                group,
                batches,
                ridge=ridge,
                reconstruction_alpha=float(reconstruction_alpha),
                reconstruction_target=reconstruction_target,
                reconstruction_objective=reconstruction_objective,
                cosine_steps=cosine_steps,
                cosine_lr=cosine_lr,
                cosine_l2=cosine_l2,
                fit_gate="none",
                fit_gate_validation_fraction=0.25,
                fit_gate_min_validation_rows=32,
                fit_gate_min_relative_improvement=0.0,
                fit_gate_max_cosine_drop=0.01,
                max_tokens_per_layer=max_tokens_per_layer,
            )
            repairs.append(repair)
            if device.startswith("cuda") and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    reconstruction_report = {
        "operation": "reconstruct_folded_causal_lm",
        "source_model": source_model,
        "candidate_model": candidate_model,
        "output_dir": str(target),
        "layer_merge_report": report,
        "ridge": ridge,
        "reconstruction_alpha": float(reconstruction_alpha),
        "reconstruction_target": reconstruction_target,
        "reconstruction_objective": reconstruction_objective,
        "cosine_steps": int(cosine_steps),
        "cosine_lr": float(cosine_lr),
        "cosine_l2": float(cosine_l2),
        "max_length": max_length,
        "max_tokens_per_layer": max_tokens_per_layer,
        "torch_dtype": torch_dtype,
        "calibration_text_count": len(texts),
        "eval_overlap_count": eval_overlap_count,
        "repairs": repairs,
        "param_count": sum(param.numel() for param in candidate.parameters()),
    }
    _write_json(target / "neural_forge_fold_reconstruction_report.json", reconstruction_report)
    return reconstruction_report


def _load_fold_hidden_projection(
    torch,
    candidate_model: str | Path,
    source_dim: int,
    target_dim: int,
    expected_source_model: str | None = None,
) -> dict[str, Any] | None:
    root = Path(candidate_model)
    if not root.is_dir():
        return None
    report_path = root / "neural_forge_donor_projected_init_report.json"
    report = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
    reported_source_model = report.get("donor_model")
    if (
        expected_source_model is not None
        and reported_source_model is not None
        and str(reported_source_model) != str(expected_source_model)
    ):
        return None

    artifact = report.get("hidden_basis_artifact") or {}
    artifact_name = artifact.get("file") or "neural_forge_hidden_basis.safetensors"
    artifact_path = root / str(artifact_name)
    if artifact_path.is_file():
        from safetensors.torch import load_file

        tensors = load_file(str(artifact_path), device="cpu")
        basis = tensors.get("output_basis")
        basis_key = "output_basis"
        if basis is None:
            basis = tensors.get("basis")
            basis_key = "basis"
        if basis is not None and tuple(basis.shape) == (int(source_dim), int(target_dim)):
            input_basis = tensors.get("input_basis")
            duality_error = None
            if input_basis is not None:
                if tuple(input_basis.shape) != (int(source_dim), int(target_dim)):
                    return None
                identity = torch.eye(int(target_dim), dtype=torch.float32)
                duality_error = float(
                    (input_basis.detach().cpu().to(torch.float32).t().matmul(basis.to(torch.float32)) - identity)
                    .abs()
                    .max()
                    .item()
                )
                if not math.isfinite(duality_error) or duality_error > 1e-3:
                    return None
            return {
                "map": basis.detach().cpu().to(torch.float32).contiguous(),
                "source": "fold-basis-artifact",
                "basis_key": basis_key,
                "artifact": str(artifact_path),
                "source_dim": int(source_dim),
                "target_dim": int(target_dim),
                "reported_source_model": reported_source_model,
                "duality_max_abs_error": duality_error,
            }

    basis_report = report.get("hidden_basis") or {}
    if basis_report.get("mode") == "activation-coordinate":
        indices = [int(item) for item in basis_report.get("selected_indices") or []]
        if len(indices) == int(target_dim) and len(set(indices)) == int(target_dim):
            if min(indices, default=0) >= 0 and max(indices, default=-1) < int(source_dim):
                basis = torch.zeros((int(source_dim), int(target_dim)), dtype=torch.float32)
                basis[
                    torch.tensor(indices, dtype=torch.long),
                    torch.arange(int(target_dim), dtype=torch.long),
                ] = 1.0
                return {
                    "map": basis.contiguous(),
                    "source": "fold-basis-coordinate-report",
                    "basis_key": "selected_indices",
                    "report": str(report_path),
                    "source_dim": int(source_dim),
                    "target_dim": int(target_dim),
                }
    return None


def _copy_fold_basis_artifacts(candidate_model: str | Path, output_dir: Path) -> list[dict[str, Any]]:
    source_root = Path(candidate_model)
    if not source_root.is_dir():
        return []
    names = {
        "neural_forge_hidden_basis.safetensors",
        "neural_forge_donor_projected_init_report.json",
    }
    report_path = source_root / "neural_forge_donor_projected_init_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            artifact_name = (report.get("hidden_basis_artifact") or {}).get("file")
            if artifact_name:
                names.add(str(artifact_name))
        except (OSError, json.JSONDecodeError):
            pass
    copied = []
    for name in sorted(names):
        source = source_root / name
        if not source.is_file():
            continue
        destination = output_dir / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied.append({"file": name, "bytes": int(destination.stat().st_size)})
    return copied


def _load_fold_layer_plan(
    candidate_model: str | Path,
    target_layer_count: int,
    source_layer_count: int,
) -> dict[str, Any] | None:
    report_path = Path(candidate_model) / "neural_forge_donor_projected_init_report.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    by_target: dict[int, dict[str, Any]] = {}
    for entry in report.get("layer_reports") or []:
        if not isinstance(entry, dict) or "target_layer" not in entry:
            continue
        target_idx = int(entry["target_layer"])
        item = {
            "representative": int(entry.get("donor_layer", -1)),
            "group": [int(value) for value in entry.get("donor_layer_group") or []],
            "mode": str(entry.get("donor_layer_fold_mode") or "representative"),
        }
        if target_idx in by_target and by_target[target_idx] != item:
            raise ValueError(f"Inconsistent donor fold plan for target layer {target_idx}.")
        by_target[target_idx] = item
    if set(by_target) != set(range(int(target_layer_count))):
        return None
    group_report = report.get("donor_layer_groups") or {}
    weight_rows = group_report.get("group_layer_weights") or []
    layers = []
    for target_idx in range(int(target_layer_count)):
        item = dict(by_target[target_idx])
        if not item["group"]:
            item["group"] = [int(item["representative"])]
        if item["representative"] not in item["group"]:
            raise ValueError(f"Donor representative is outside target layer {target_idx} group.")
        if any(layer_idx < 0 or layer_idx >= int(source_layer_count) for layer_idx in item["group"]):
            raise ValueError(f"Donor fold group for target layer {target_idx} is out of range.")
        weights = None
        if target_idx < len(weight_rows):
            candidate_weights = [float(value) for value in weight_rows[target_idx]]
            if len(candidate_weights) == len(item["group"]):
                weights = candidate_weights
        item["weights"] = weights
        item["target_layer"] = int(target_idx)
        layers.append(item)
    return {
        "source": str(report_path),
        "strategy": group_report.get("strategy"),
        "layers": layers,
    }


def reconstruct_cross_tokenizer_causal_lm(
    source_model: str,
    candidate_model: str,
    output_dir: str | Path,
    ridge: float = 1e-3,
    reconstruction_alpha: float = 1.0,
    reconstruction_target: str = "both",
    donor_layer_fold_mode: str = "representative",
    donor_layer_plan: dict[str, Any] | None = None,
    reconstruction_objective: str = "ridge",
    cosine_steps: int = 80,
    cosine_lr: float = 1e-2,
    cosine_l2: float = 1e-4,
    max_length: int = 128,
    max_tokens_per_layer: int = 512,
    max_anchors: int = 2048,
    hidden_projection_source: str = "auto",
    hidden_projection_scope: str = "global",
    fit_final_interface: bool = False,
    layer_limit: int | None = None,
    fit_gate: str = "none",
    fit_gate_validation_fraction: float = 0.25,
    fit_gate_min_validation_rows: int = 32,
    fit_gate_min_relative_improvement: float = 0.0,
    fit_gate_max_cosine_drop: float = 0.01,
    fit_gate_min_fitted_cosine: float = -1.0,
    layer_gate_metric: str = "none",
    layer_gate_max_ppl_ratio: float = 1.02,
    layer_gate_max_texts: int = 4,
    layer_gate_max_behavior_drop: float = 0.0,
    layer_gate_behavior_max_new_tokens: int = 96,
    layer_gate_eval_suite: str | Path | dict[str, Any] | None = None,
    layer_gate_max_generation_prompts: int | None = None,
    torch_dtype: str = "float16",
    candidate_torch_dtype: str = "float32",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    calibration_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run cross-tokenizer block reconstruction on a folded child.

    Source hidden rows are projected into child hidden size using shared token
    embeddings. Paired activation rows are then selected from matching decoded
    token spans and used to fit child-space input correction maps and optional
    output projection weights for target block attention/MLP modules.
    """

    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not 0.0 <= float(reconstruction_alpha) <= 1.0:
        raise ValueError("reconstruction_alpha must be in [0, 1].")
    xfamily_targets = {
        "consumer-input",
        "block-interface",
        "attention",
        "attention-output",
        "attention-all",
        "mlp",
        "mlp-output",
        "mlp-all",
        "module-output",
        "module-all",
        "both",
        "all",
    }
    if reconstruction_target not in xfamily_targets:
        raise ValueError(
            "reconstruction_target must be consumer-input, block-interface, attention, attention-output, attention-all, "
            "mlp, mlp-output, mlp-all, module-output, module-all, both, or all."
        )
    if donor_layer_fold_mode not in {"representative", "average", "fold"}:
        raise ValueError("donor_layer_fold_mode must be representative, average, or fold.")
    if reconstruction_objective not in {"ridge", "cosine"}:
        raise ValueError("reconstruction_objective must be ridge or cosine.")
    if cosine_steps < 0:
        raise ValueError("cosine_steps must be non-negative.")
    if cosine_lr <= 0:
        raise ValueError("cosine_lr must be positive.")
    if cosine_l2 < 0:
        raise ValueError("cosine_l2 must be non-negative.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if max_tokens_per_layer <= 0:
        raise ValueError("max_tokens_per_layer must be positive.")
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")
    if hidden_projection_source not in {"auto", "fold-basis", "shared-token"}:
        raise ValueError("hidden_projection_source must be auto, fold-basis, or shared-token.")
    if hidden_projection_scope not in {"global", "layer-input", "layer-input-gated"}:
        raise ValueError("hidden_projection_scope must be global, layer-input, or layer-input-gated.")
    if layer_limit is not None and layer_limit <= 0:
        raise ValueError("layer_limit must be positive when provided.")
    if fit_gate not in {"none", "validation"}:
        raise ValueError("fit_gate must be none or validation.")
    if not 0.0 <= float(fit_gate_validation_fraction) < 1.0:
        raise ValueError("fit_gate_validation_fraction must be in [0, 1).")
    if fit_gate_min_validation_rows <= 0:
        raise ValueError("fit_gate_min_validation_rows must be positive.")
    if fit_gate_min_relative_improvement < 0:
        raise ValueError("fit_gate_min_relative_improvement must be non-negative.")
    if fit_gate_max_cosine_drop < 0:
        raise ValueError("fit_gate_max_cosine_drop must be non-negative.")
    if not -1.0 <= float(fit_gate_min_fitted_cosine) <= 1.0:
        raise ValueError("fit_gate_min_fitted_cosine must be between -1.0 and 1.0.")
    if layer_gate_metric not in {"none", "ppl", "behavior"}:
        raise ValueError("layer_gate_metric must be none, ppl, or behavior.")
    if layer_gate_max_ppl_ratio <= 0:
        raise ValueError("layer_gate_max_ppl_ratio must be positive.")
    if layer_gate_max_texts <= 0:
        raise ValueError("layer_gate_max_texts must be positive.")
    if not 0.0 <= float(layer_gate_max_behavior_drop) <= 1.0:
        raise ValueError("layer_gate_max_behavior_drop must be in [0, 1].")
    if layer_gate_behavior_max_new_tokens <= 0:
        raise ValueError("layer_gate_behavior_max_new_tokens must be positive.")
    if layer_gate_max_generation_prompts is not None and layer_gate_max_generation_prompts <= 0:
        raise ValueError("layer_gate_max_generation_prompts must be positive when provided.")

    from neural_forge.hf_xfamily import _canonical_token
    from neural_forge.hf_xfamily import _canonical_token_map
    from neural_forge.hf_xfamily import _donor_layer_group_for_target

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    candidate_dtype = _dtype_from_name(torch, candidate_torch_dtype)
    source_tokenizer = transformers.AutoTokenizer.from_pretrained(source_model, trust_remote_code=trust_remote_code)
    candidate_tokenizer = transformers.AutoTokenizer.from_pretrained(candidate_model, trust_remote_code=trust_remote_code)
    if source_tokenizer.pad_token_id is None and source_tokenizer.eos_token_id is not None:
        source_tokenizer.pad_token = source_tokenizer.eos_token
    if candidate_tokenizer.pad_token_id is None and candidate_tokenizer.eos_token_id is not None:
        candidate_tokenizer.pad_token = candidate_tokenizer.eos_token

    source = transformers.AutoModelForCausalLM.from_pretrained(
        source_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    candidate = transformers.AutoModelForCausalLM.from_pretrained(
        candidate_model,
        torch_dtype=candidate_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    source.to(device)
    candidate.to(device)
    source.eval()
    candidate.eval()
    if hasattr(source.config, "use_cache"):
        source.config.use_cache = False
    if hasattr(candidate.config, "use_cache"):
        candidate.config.use_cache = False

    source_layers = _recovery_decoder_layers(source)
    candidate_layers = _recovery_decoder_layers(candidate)
    if not source_layers or not candidate_layers:
        raise ValueError("Both source and candidate need decoder layers.")
    if donor_layer_plan is not None:
        plan_layers = list(donor_layer_plan.get("layers") or [])
        if len(plan_layers) < len(candidate_layers):
            raise ValueError("donor_layer_plan does not cover every candidate layer.")
        for item in plan_layers[: len(candidate_layers)]:
            group = [int(value) for value in item.get("group") or []]
            representative = int(item.get("representative", -1))
            if not group or representative not in group:
                raise ValueError("donor_layer_plan needs a representative contained in every group.")
            if any(value < 0 or value >= len(source_layers) for value in group):
                raise ValueError("donor_layer_plan contains an out-of-range source layer.")

    anchors = _shared_recovery_anchors(
        source_tokenizer,
        candidate_tokenizer,
        max_anchors,
        _canonical_token_map,
        selection="stratified",
        min_chars=1,
    )
    if not anchors:
        raise ValueError("No shared decoded-token anchors found for cross-tokenizer reconstruction.")
    source_anchor_idx = torch.tensor([item[1] for item in anchors], dtype=torch.long, device=device)
    candidate_anchor_idx = torch.tensor([item[2] for item in anchors], dtype=torch.long, device=device)
    hidden_projection_report = None
    if hidden_projection_source in {"auto", "fold-basis"}:
        hidden_projection_report = _load_fold_hidden_projection(
            torch,
            candidate_model,
            int(source.get_input_embeddings().weight.shape[1]),
            int(candidate.get_input_embeddings().weight.shape[1]),
            expected_source_model=source_model,
        )
        if hidden_projection_report is None and hidden_projection_source == "fold-basis":
            raise ValueError("candidate_model does not contain a compatible Neural Forge fold basis.")
    if hidden_projection_report is None:
        hidden_projection_report = _fit_shared_embedding_hidden_projection(
            torch,
            source,
            candidate,
            source_anchor_idx,
            candidate_anchor_idx,
            ridge=ridge,
            max_anchors=max_anchors,
        )
        hidden_projection_report["source"] = "shared-token-ridge"
    hidden_projection = hidden_projection_report["map"].to(device=device, dtype=torch.float32)

    texts = list(calibration_texts or RECOVERY_TEXTS)
    if not texts:
        raise ValueError("Need at least one reconstruction text.")
    eval_overlap_count = len(set(texts) & set(TEXT_EVALS))
    paired_batches = _make_paired_batches(
        source_tokenizer,
        candidate_tokenizer,
        texts,
        batch_size=1,
        max_length=max_length,
        device=device,
        include_offsets=True,
    )

    repairs = []
    layer_count = len(candidate_layers) if layer_limit is None else min(int(layer_limit), len(candidate_layers))
    layer_gate: dict[str, Any] = {
        "metric": layer_gate_metric,
        "max_ppl_ratio": float(layer_gate_max_ppl_ratio),
        "max_texts": int(layer_gate_max_texts),
        "max_behavior_drop": float(layer_gate_max_behavior_drop),
        "behavior_max_new_tokens": int(layer_gate_behavior_max_new_tokens),
    }
    layer_gate_suite, layer_gate_behavior_probes, layer_gate_generation_prompts = _recovery_behavior_eval_suite(
        layer_gate_eval_suite
    )
    if layer_gate_suite is not None:
        layer_gate["eval_suite"] = {
            **(_recovery_behavior_eval_suite_report(layer_gate_suite) or {}),
            "max_behavior_probes": int(layer_gate_max_texts),
            "max_generation_prompts": (
                None if layer_gate_max_generation_prompts is None else int(layer_gate_max_generation_prompts)
            ),
        }
    gate_texts = list(TEXT_EVALS[: int(layer_gate_max_texts)])
    previous_gate_ppl: float | None = None
    previous_gate_behavior: dict[str, Any] | None = None
    with torch.no_grad():
        if layer_gate_metric == "ppl":
            previous_gate_ppl = _heldout_ppl_mean(
                torch,
                candidate,
                candidate_tokenizer,
                gate_texts,
                device,
                max_length,
            )
            layer_gate["initial_ppl"] = previous_gate_ppl
            layer_gate["texts"] = gate_texts
        elif layer_gate_metric == "behavior":
            previous_gate_behavior = _heldout_behavior_summary(
                torch,
                candidate,
                candidate_tokenizer,
                device,
                int(layer_gate_behavior_max_new_tokens),
                int(layer_gate_max_texts),
                layer_gate_behavior_probes,
                layer_gate_generation_prompts,
                layer_gate_max_generation_prompts,
            )
            layer_gate["initial_behavior"] = previous_gate_behavior
        _recovery_progress(
            "functional_reconstruction_start",
            layers=layer_count,
            target=reconstruction_target,
            fit_final_interface=bool(fit_final_interface),
        )
        for candidate_idx in range(layer_count):
            _recovery_progress(
                "functional_reconstruction_layer_start",
                current=candidate_idx + 1,
                total=layer_count,
            )
            plan_layers = [] if donor_layer_plan is None else list(donor_layer_plan.get("layers") or [])
            plan_item = plan_layers[candidate_idx] if candidate_idx < len(plan_layers) else None
            if plan_item is None:
                source_idx = _proportional_layer_index(candidate_idx, len(candidate_layers), len(source_layers))
                source_group = (
                    [source_idx]
                    if donor_layer_fold_mode == "representative"
                    else _donor_layer_group_for_target(candidate_idx, len(candidate_layers), len(source_layers))
                )
                source_group_weights = None
            else:
                source_idx = int(plan_item["representative"])
                source_group = [int(item) for item in plan_item["group"]]
                source_group_weights = plan_item.get("weights")
            layer_state = (
                _snapshot_layer_gate_states(candidate_layers, candidate_idx, reconstruction_target)
                if layer_gate_metric in {"ppl", "behavior"}
                else None
            )
            layer_hidden_projection = hidden_projection
            layer_hidden_projection_report: dict[str, Any] | None = None
            if hidden_projection_scope in {"layer-input", "layer-input-gated"}:
                layer_hidden_projection_report = _fit_cross_tokenizer_layer_input_hidden_projection(
                    torch,
                    source,
                    candidate,
                    source_tokenizer,
                    candidate_tokenizer,
                    paired_batches,
                    source_group,
                    candidate_idx,
                    _canonical_token,
                    ridge,
                    max_tokens_per_layer,
                    baseline_projection=hidden_projection if hidden_projection_scope == "layer-input-gated" else None,
                    min_relative_improvement=fit_gate_min_relative_improvement,
                    max_cosine_drop=fit_gate_max_cosine_drop,
                    min_fitted_cosine=fit_gate_min_fitted_cosine,
                )
                if not layer_hidden_projection_report.get("skipped"):
                    layer_hidden_projection = layer_hidden_projection_report["map"].to(device=device, dtype=torch.float32)
            repair = _reconstruct_cross_tokenizer_layer(
                torch,
                source,
                candidate,
                source_layers[source_idx],
                candidate_layers[candidate_idx],
                source_layers,
                candidate_layers,
                candidate_idx,
                source_group,
                source_tokenizer,
                candidate_tokenizer,
                paired_batches,
                layer_hidden_projection,
                _canonical_token,
                ridge=ridge,
                reconstruction_alpha=float(reconstruction_alpha),
                reconstruction_target=reconstruction_target,
                reconstruction_objective=reconstruction_objective,
                cosine_steps=cosine_steps,
                cosine_lr=cosine_lr,
                cosine_l2=cosine_l2,
                fit_gate=fit_gate,
                fit_gate_validation_fraction=fit_gate_validation_fraction,
                fit_gate_min_validation_rows=fit_gate_min_validation_rows,
                fit_gate_min_relative_improvement=fit_gate_min_relative_improvement,
                fit_gate_max_cosine_drop=fit_gate_max_cosine_drop,
                max_tokens_per_layer=max_tokens_per_layer,
                donor_layer_fold_mode=donor_layer_fold_mode,
                source_layer_group_weights=source_group_weights,
            )
            repair["source_layer"] = int(source_idx)
            repair["source_layer_group"] = [int(item) for item in source_group]
            repair["source_layer_group_weights"] = source_group_weights
            repair["candidate_layer"] = int(candidate_idx)
            if layer_hidden_projection_report is not None:
                repair["hidden_projection"] = {
                    key: value for key, value in layer_hidden_projection_report.items() if key != "map"
                }
            if layer_gate_metric == "ppl":
                candidate_ppl = _heldout_ppl_mean(
                    torch,
                    candidate,
                    candidate_tokenizer,
                    gate_texts,
                    device,
                    max_length,
                )
                gate_decision = _layer_gate_decision(
                    previous_gate_ppl if previous_gate_ppl is not None else candidate_ppl,
                    candidate_ppl,
                    layer_gate_max_ppl_ratio,
                )
                if gate_decision["accepted"]:
                    previous_gate_ppl = candidate_ppl
                elif layer_state is not None:
                    _restore_layer_gate_states(candidate_layers, layer_state)
                    reverted_ppl = _heldout_ppl_mean(
                        torch,
                        candidate,
                        candidate_tokenizer,
                        gate_texts,
                        device,
                        max_length,
                    )
                    gate_decision["reverted"] = True
                    gate_decision["reverted_ppl"] = reverted_ppl
                    gate_decision["restored_layers"] = [int(idx) for idx in sorted(layer_state)]
                    previous_gate_ppl = reverted_ppl
                if layer_state is not None:
                    gate_decision["snapshot_layers"] = [int(idx) for idx in sorted(layer_state)]
                repair["layer_gate"] = gate_decision
            elif layer_gate_metric == "behavior":
                candidate_behavior = _heldout_behavior_summary(
                    torch,
                    candidate,
                    candidate_tokenizer,
                    device,
                    int(layer_gate_behavior_max_new_tokens),
                    int(layer_gate_max_texts),
                    layer_gate_behavior_probes,
                    layer_gate_generation_prompts,
                    layer_gate_max_generation_prompts,
                )
                gate_decision = _layer_gate_behavior_decision(
                    previous_gate_behavior or candidate_behavior,
                    candidate_behavior,
                    layer_gate_max_behavior_drop,
                )
                gate_decision["candidate_behavior"] = candidate_behavior
                if gate_decision["accepted"]:
                    previous_gate_behavior = candidate_behavior
                elif layer_state is not None:
                    _restore_layer_gate_states(candidate_layers, layer_state)
                    reverted_behavior = _heldout_behavior_summary(
                        torch,
                        candidate,
                        candidate_tokenizer,
                        device,
                        int(layer_gate_behavior_max_new_tokens),
                        int(layer_gate_max_texts),
                        layer_gate_behavior_probes,
                        layer_gate_generation_prompts,
                        layer_gate_max_generation_prompts,
                    )
                    gate_decision["reverted"] = True
                    gate_decision["reverted_behavior"] = reverted_behavior
                    gate_decision["restored_layers"] = [int(idx) for idx in sorted(layer_state)]
                    previous_gate_behavior = reverted_behavior
                if layer_state is not None:
                    gate_decision["snapshot_layers"] = [int(idx) for idx in sorted(layer_state)]
                repair["layer_gate"] = gate_decision
            repairs.append(repair)
            _recovery_progress(
                "functional_reconstruction_layer_done",
                current=candidate_idx + 1,
                total=layer_count,
            )
            if device.startswith("cuda") and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()

        final_interface_report = None
        if fit_final_interface:
            _recovery_progress("functional_reconstruction_final_interface_start")
            final_interface_report = _reconstruct_cross_tokenizer_final_interface(
                torch,
                source,
                candidate,
                source_tokenizer,
                candidate_tokenizer,
                paired_batches,
                hidden_projection,
                _canonical_token,
                ridge=ridge,
                reconstruction_alpha=float(reconstruction_alpha),
                fit_gate=fit_gate,
                fit_gate_validation_fraction=fit_gate_validation_fraction,
                fit_gate_min_validation_rows=fit_gate_min_validation_rows,
                fit_gate_min_relative_improvement=fit_gate_min_relative_improvement,
                fit_gate_max_cosine_drop=fit_gate_max_cosine_drop,
                max_tokens=max_tokens_per_layer,
            )
            _recovery_progress(
                "functional_reconstruction_final_interface_done",
                applied=bool(final_interface_report.get("applied")),
            )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(target, safe_serialization=True)
    candidate_tokenizer.save_pretrained(target)
    fold_artifacts = _copy_fold_basis_artifacts(candidate_model, target)
    reconstruction_report = {
        "operation": "reconstruct_cross_tokenizer_causal_lm",
        "source_model": source_model,
        "candidate_model": candidate_model,
        "output_dir": str(target),
        "ridge": ridge,
        "reconstruction_alpha": float(reconstruction_alpha),
        "reconstruction_target": reconstruction_target,
        "donor_layer_fold_mode": donor_layer_fold_mode,
        "donor_layer_plan": donor_layer_plan,
        "reconstruction_objective": reconstruction_objective,
        "cosine_steps": int(cosine_steps),
        "cosine_lr": float(cosine_lr),
        "cosine_l2": float(cosine_l2),
        "max_length": max_length,
        "max_tokens_per_layer": max_tokens_per_layer,
        "max_anchors": max_anchors,
        "hidden_projection_source": hidden_projection_source,
        "hidden_projection_scope": hidden_projection_scope,
        "fit_final_interface": bool(fit_final_interface),
        "layer_limit": layer_limit,
        "fit_gate": {
            "mode": fit_gate,
            "validation_fraction": float(fit_gate_validation_fraction),
            "min_validation_rows": int(fit_gate_min_validation_rows),
            "min_relative_improvement": float(fit_gate_min_relative_improvement),
            "max_cosine_drop": float(fit_gate_max_cosine_drop),
            "min_fitted_cosine": float(fit_gate_min_fitted_cosine),
        },
        "layer_gate": layer_gate,
        "torch_dtype": torch_dtype,
        "candidate_torch_dtype": candidate_torch_dtype,
        "calibration_text_count": len(texts),
        "eval_overlap_count": eval_overlap_count,
        "hidden_projection": {key: value for key, value in hidden_projection_report.items() if key != "map"},
        "final_interface": final_interface_report,
        "fold_artifacts": fold_artifacts,
        "repairs": repairs,
        "param_count": sum(param.numel() for param in candidate.parameters()),
    }
    _write_json(target / "neural_forge_xfamily_reconstruction_report.json", reconstruction_report)
    return reconstruction_report


def functional_fold_cross_tokenizer_causal_lm(
    source_model: str,
    candidate_model: str,
    output_dir: str | Path,
    functional_target: str = "module-output",
    alpha: float = 1.0,
    ridge: float = 1e-3,
    max_length: int = 160,
    max_tokens_per_layer: int = 4096,
    max_anchors: int = 20_000,
    fit_gate_validation_fraction: float = 0.20,
    fit_gate_min_validation_rows: int = 64,
    fit_gate_min_relative_improvement: float = 0.0,
    fit_gate_max_cosine_drop: float = 0.01,
    torch_dtype: str = "float16",
    candidate_torch_dtype: str = "float32",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    calibration_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize sequential donor behavior with closed-form maps only."""

    if functional_target not in {"module-output", "module-all"}:
        raise ValueError("functional_target must be module-output or module-all.")
    donor_layer_plan = None
    candidate_root = Path(candidate_model)
    if candidate_root.is_dir():
        try:
            config = json.loads((candidate_root / "config.json").read_text(encoding="utf-8"))
            init_report = json.loads(
                (candidate_root / "neural_forge_donor_projected_init_report.json").read_text(encoding="utf-8")
            )
            entries = [item for item in (init_report.get("layer_reports") or []) if isinstance(item, dict)]
            target_layer_count = int(
                config.get("num_hidden_layers")
                or config.get("n_layer")
                or (1 + max(int(item.get("target_layer", -1)) for item in entries))
            )
            donor_indices = [
                int(index)
                for item in entries
                for index in [item.get("donor_layer", -1), *(item.get("donor_layer_group") or [])]
            ]
            source_layer_count = 1 + max(donor_indices)
            donor_layer_plan = _load_fold_layer_plan(
                candidate_root,
                target_layer_count=target_layer_count,
                source_layer_count=source_layer_count,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            donor_layer_plan = None
    plan_modes = {
        str(item.get("mode"))
        for item in (donor_layer_plan or {}).get("layers", [])
        if item.get("mode")
    }
    planned_fold_mode = next(iter(plan_modes)) if len(plan_modes) == 1 else "representative"
    report = reconstruct_cross_tokenizer_causal_lm(
        source_model=source_model,
        candidate_model=candidate_model,
        output_dir=output_dir,
        ridge=ridge,
        reconstruction_alpha=alpha,
        reconstruction_target=functional_target,
        donor_layer_fold_mode=planned_fold_mode,
        donor_layer_plan=donor_layer_plan,
        reconstruction_objective="ridge",
        cosine_steps=0,
        cosine_lr=1e-2,
        cosine_l2=0.0,
        max_length=max_length,
        max_tokens_per_layer=max_tokens_per_layer,
        max_anchors=max_anchors,
        hidden_projection_source="fold-basis",
        hidden_projection_scope="global",
        fit_final_interface=True,
        fit_gate="validation",
        fit_gate_validation_fraction=fit_gate_validation_fraction,
        fit_gate_min_validation_rows=fit_gate_min_validation_rows,
        fit_gate_min_relative_improvement=fit_gate_min_relative_improvement,
        fit_gate_max_cosine_drop=fit_gate_max_cosine_drop,
        layer_gate_metric="none",
        torch_dtype=torch_dtype,
        candidate_torch_dtype=candidate_torch_dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        calibration_texts=calibration_texts,
    )
    functional_report = {
        **report,
        "operation": "functional_fold_cross_tokenizer_causal_lm",
        "zero_gradient": True,
        "optimizer_steps": 0,
        "functional_target": functional_target,
        "alpha": float(alpha),
        "hidden_projection_source": "fold-basis",
    }
    _write_json(Path(output_dir) / "neural_forge_functional_fold_report.json", functional_report)
    return functional_report


def blockwise_reconstruct_cross_tokenizer_causal_lm(
    source_model: str,
    candidate_model: str,
    output_dir: str | Path,
    steps_per_layer: int = 24,
    batch_size: int = 1,
    max_length: int = 128,
    lr: float = 2e-5,
    lm_weight: float = 0.0,
    response_lm_weight: float = 0.0,
    first_token_lm_weight: float = 0.0,
    response_end_lm_weight: float = 0.0,
    response_loss_mode: str = "none",
    hidden_ridge: float = 1e-3,
    hidden_projection_scope: str = "global",
    hidden_metric: str = "projected",
    span_pair_mode: str = "exact",
    block_target: str = "hidden",
    residual_magnitude_compensation: str = "none",
    residual_compensation_min_scale: float = 0.50,
    residual_compensation_max_scale: float = 1.50,
    residual_compensation_linear_ridge: float = 1e-3,
    residual_compensation_alpha: float = 1.0,
    residual_compensation_linear_rank: int = 32,
    pre_reconstruction_target: str = "none",
    pre_reconstruction_ridge: float = 1e-3,
    pre_reconstruction_alpha: float = 0.05,
    pre_reconstruction_objective: str = "ridge",
    pre_reconstruction_cosine_steps: int = 40,
    pre_reconstruction_cosine_lr: float = 1e-2,
    pre_reconstruction_cosine_l2: float = 1e-4,
    pre_reconstruction_fit_gate: str = "validation",
    pre_reconstruction_validation_fraction: float = 0.25,
    pre_reconstruction_min_validation_rows: int = 32,
    pre_reconstruction_max_tokens_per_layer: int = 512,
    pre_reconstruction_min_relative_improvement: float = 0.0,
    pre_reconstruction_max_cosine_drop: float = 0.01,
    pre_reconstruction_min_fitted_cosine: float = -1.0,
    max_anchors: int = 2048,
    span_max_pairs: int = 512,
    donor_layer_fold_mode: str = "fold",
    donor_layer_group_strategy: str = "depth",
    layer_limit: int | None = None,
    layer_gate_metric: str = "none",
    layer_gate_max_ppl_ratio: float = 1.10,
    layer_gate_max_texts: int = 4,
    layer_gate_max_behavior_drop: float = 0.0,
    layer_gate_behavior_max_new_tokens: int = 96,
    layer_gate_eval_suite: str | Path | dict[str, Any] | None = None,
    layer_gate_max_generation_prompts: int | None = None,
    on_policy_anchor_kd_weight: float = 0.0,
    on_policy_unlikelihood_weight: float = 0.0,
    on_policy_hidden_kd_weight: float = 0.0,
    on_policy_max_new_tokens: int = 24,
    on_policy_interval: int = 10,
    on_policy_response_loss_mode: str = "auto",
    on_policy_unlikelihood_window: int = 16,
    stepwise_donor_kd_weight: float = 0.0,
    stepwise_donor_max_new_tokens: int = 8,
    stepwise_donor_interval: int = 4,
    teacher_generate_lm_weight: float = 0.0,
    teacher_generate_hidden_kd_weight: float = 0.0,
    teacher_generate_projected_vocab_kd_weight: float = 0.0,
    teacher_generate_byte_kd_weight: float = 0.0,
    teacher_generate_max_new_tokens: int = 64,
    teacher_generate_loss_mode: str = "none",
    alm_kd_weight: float = 0.0,
    alm_max_chunks: int = 64,
    alm_min_chars: int = 2,
    projected_vocab_kd_weight: float = 0.0,
    focused_block_hidden_kd_weight: float = 0.0,
    focused_projected_vocab_kd_weight: float = 0.0,
    focused_lm_weight: float = 0.0,
    token_focus: str = "auto",
    projected_vocab_kd_mode: str = "teacher-to-student",
    byte_kd_weight: float = 0.0,
    byte_hidden_kd_weight: float = 0.0,
    byte_hidden_probe_mode: str = "fixed",
    byte_hidden_probe_anchor_weight: float = 0.0,
    block_io_kd_weight: float = 0.0,
    block_io_ridge: float = 1e-3,
    block_io_loss: str = "both",
    train_output_head_with_kd: bool = False,
    untie_output_head_for_kd: bool = False,
    allow_tied_output_head_training: bool = False,
    module_output_kd_weight: float = 0.0,
    module_output_targets: str = "both",
    module_output_transport: str = "paired",
    module_output_temperature: float = 1.0,
    module_output_sinkhorn_iterations: int = 12,
    projected_vocab_max_source_tokens: int = 8,
    temperature: float = 2.0,
    torch_dtype: str = "float16",
    candidate_torch_dtype: str = "float32",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    calibration_texts: Sequence[str] | None = None,
    calibration_order: str = "original",
    grad_clip: float = 1.0,
    nonfinite_step_policy: str = "raise",
    max_nonfinite_steps_per_layer: int = 0,
    report_interval: int | None = None,
    save_best_layer_loss: bool = False,
    save_best_layer_behavior: bool = False,
    layer_behavior_eval_interval: int | None = None,
    layer_checkpoint_interval: int = 0,
) -> dict[str, Any]:
    """Sequentially train each child block to match projected donor block outputs.

    This is a nonlinear parent-to-child fold for cross-tokenizer models. Unlike
    the ridge reconstruction path, it does not only mutate linear maps. It
    freezes the candidate, unfreezes one decoder block at a time, and optimizes
    that block so its span-aligned hidden states match the donor's projected
    hidden states.
    """

    if steps_per_layer < 0:
        raise ValueError("steps_per_layer must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    _validate_blockwise_learning_rate(float(lr), int(steps_per_layer))
    if lm_weight < 0:
        raise ValueError("lm_weight must be non-negative.")
    if response_lm_weight < 0 or first_token_lm_weight < 0 or response_end_lm_weight < 0:
        raise ValueError("response LM weights must be non-negative.")
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    if response_loss_mode == "none" and (
        response_lm_weight > 0 or first_token_lm_weight > 0 or response_end_lm_weight > 0
    ):
        raise ValueError("response LM weights require response_loss_mode to be auto, assistant, or answer.")
    if hidden_ridge < 0:
        raise ValueError("hidden_ridge must be non-negative.")
    if hidden_projection_scope not in {"global", "layer-input", "layer-input-gated"}:
        raise ValueError("hidden_projection_scope must be global, layer-input, or layer-input-gated.")
    if hidden_metric not in HIDDEN_KD_METRICS:
        raise ValueError(f"hidden_metric must be {HIDDEN_KD_METRIC_DESCRIPTION}.")
    if span_pair_mode not in SPAN_PAIR_MODES:
        raise ValueError("span_pair_mode must be exact, overlap, or both.")
    if block_target not in {"hidden", "delta"}:
        raise ValueError("block_target must be hidden or delta.")
    if residual_magnitude_compensation not in RESIDUAL_MAGNITUDE_COMPENSATION_MODES:
        raise ValueError(
            "residual_magnitude_compensation must be none, pre, pre-linear, linear-pre, "
            "pre-lowrank, lowrank-pre, pre-eora, eora-pre, pre-module-linear, "
            "module-linear-pre, pre-module-lowrank, module-lowrank-pre, "
            "pre-module-eora, or module-eora-pre."
        )
    if residual_compensation_min_scale <= 0 or residual_compensation_max_scale <= 0:
        raise ValueError("residual compensation scales must be positive.")
    if residual_compensation_min_scale > residual_compensation_max_scale:
        raise ValueError("residual_compensation_min_scale must be <= residual_compensation_max_scale.")
    if residual_compensation_linear_ridge < 0:
        raise ValueError("residual_compensation_linear_ridge must be non-negative.")
    if not 0 <= residual_compensation_alpha <= 1:
        raise ValueError("residual_compensation_alpha must be between 0 and 1.")
    if residual_compensation_linear_rank <= 0:
        raise ValueError("residual_compensation_linear_rank must be positive.")
    if pre_reconstruction_target not in {"none", "consumer-input", "block-interface", "all"}:
        raise ValueError("pre_reconstruction_target must be none, consumer-input, block-interface, or all.")
    if pre_reconstruction_ridge < 0:
        raise ValueError("pre_reconstruction_ridge must be non-negative.")
    if not 0 <= pre_reconstruction_alpha <= 1:
        raise ValueError("pre_reconstruction_alpha must be between 0 and 1.")
    if pre_reconstruction_objective not in {"ridge", "cosine"}:
        raise ValueError("pre_reconstruction_objective must be ridge or cosine.")
    if pre_reconstruction_cosine_steps < 0:
        raise ValueError("pre_reconstruction_cosine_steps must be non-negative.")
    if pre_reconstruction_cosine_lr <= 0:
        raise ValueError("pre_reconstruction_cosine_lr must be positive.")
    if pre_reconstruction_cosine_l2 < 0:
        raise ValueError("pre_reconstruction_cosine_l2 must be non-negative.")
    if pre_reconstruction_fit_gate not in {"none", "validation"}:
        raise ValueError("pre_reconstruction_fit_gate must be none or validation.")
    if not 0.0 <= float(pre_reconstruction_validation_fraction) < 1.0:
        raise ValueError("pre_reconstruction_validation_fraction must be in [0, 1).")
    if pre_reconstruction_min_validation_rows <= 0:
        raise ValueError("pre_reconstruction_min_validation_rows must be positive.")
    if pre_reconstruction_max_tokens_per_layer <= 0:
        raise ValueError("pre_reconstruction_max_tokens_per_layer must be positive.")
    if pre_reconstruction_min_relative_improvement < 0:
        raise ValueError("pre_reconstruction_min_relative_improvement must be non-negative.")
    if pre_reconstruction_max_cosine_drop < 0:
        raise ValueError("pre_reconstruction_max_cosine_drop must be non-negative.")
    if not -1.0 <= float(pre_reconstruction_min_fitted_cosine) <= 1.0:
        raise ValueError("pre_reconstruction_min_fitted_cosine must be between -1.0 and 1.0.")
    if nonfinite_step_policy not in {"raise", "skip-step"}:
        raise ValueError("nonfinite_step_policy must be raise or skip-step.")
    if max_nonfinite_steps_per_layer < 0:
        raise ValueError("max_nonfinite_steps_per_layer must be non-negative.")
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")
    if span_max_pairs <= 0:
        raise ValueError("span_max_pairs must be positive.")
    if donor_layer_fold_mode not in {"representative", "average", "fold"}:
        raise ValueError("donor_layer_fold_mode must be representative, average, or fold.")
    if donor_layer_group_strategy not in {
        "depth",
        "activation-energy",
        "activation-geometry",
        "activation-ot",
        "activation-cka",
        "activation-cka-weighted",
    }:
        raise ValueError(
            "donor_layer_group_strategy must be depth, activation-energy, activation-geometry, "
            "activation-ot, activation-cka, or activation-cka-weighted."
        )
    if layer_limit is not None and layer_limit <= 0:
        raise ValueError("layer_limit must be positive when provided.")
    if layer_gate_metric not in {"none", "ppl", "behavior"}:
        raise ValueError("layer_gate_metric must be none, ppl, or behavior.")
    if layer_gate_max_ppl_ratio <= 0:
        raise ValueError("layer_gate_max_ppl_ratio must be positive.")
    if layer_gate_max_texts <= 0:
        raise ValueError("layer_gate_max_texts must be positive.")
    if not 0.0 <= float(layer_gate_max_behavior_drop) <= 1.0:
        raise ValueError("layer_gate_max_behavior_drop must be in [0, 1].")
    if layer_gate_behavior_max_new_tokens <= 0:
        raise ValueError("layer_gate_behavior_max_new_tokens must be positive.")
    if layer_gate_max_generation_prompts is not None and layer_gate_max_generation_prompts <= 0:
        raise ValueError("layer_gate_max_generation_prompts must be positive when provided.")
    if on_policy_anchor_kd_weight < 0 or on_policy_unlikelihood_weight < 0 or on_policy_hidden_kd_weight < 0:
        raise ValueError("on-policy weights must be non-negative.")
    if stepwise_donor_kd_weight < 0:
        raise ValueError("stepwise_donor_kd_weight must be non-negative.")
    if (
        teacher_generate_lm_weight < 0
        or teacher_generate_hidden_kd_weight < 0
        or teacher_generate_projected_vocab_kd_weight < 0
        or teacher_generate_byte_kd_weight < 0
    ):
        raise ValueError("teacher-generated recovery weights must be non-negative.")
    if teacher_generate_max_new_tokens <= 0:
        raise ValueError("teacher_generate_max_new_tokens must be positive.")
    if teacher_generate_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("teacher_generate_loss_mode must be none, auto, assistant, or answer.")
    if alm_kd_weight < 0:
        raise ValueError("alm_kd_weight must be non-negative.")
    if alm_max_chunks <= 0:
        raise ValueError("alm_max_chunks must be positive.")
    if alm_min_chars <= 0:
        raise ValueError("alm_min_chars must be positive.")
    if projected_vocab_kd_weight < 0:
        raise ValueError("projected_vocab_kd_weight must be non-negative.")
    if focused_block_hidden_kd_weight < 0:
        raise ValueError("focused_block_hidden_kd_weight must be non-negative.")
    if focused_projected_vocab_kd_weight < 0:
        raise ValueError("focused_projected_vocab_kd_weight must be non-negative.")
    if focused_lm_weight < 0:
        raise ValueError("focused_lm_weight must be non-negative.")
    if token_focus not in {"auto", "response", "numbers", "code", "all"}:
        raise ValueError("token_focus must be auto, response, numbers, code, or all.")
    if projected_vocab_kd_mode not in {"teacher-to-student", "student-to-teacher", "both"}:
        raise ValueError("projected_vocab_kd_mode must be teacher-to-student, student-to-teacher, or both.")
    if byte_kd_weight < 0:
        raise ValueError("byte_kd_weight must be non-negative.")
    if byte_hidden_kd_weight < 0:
        raise ValueError("byte_hidden_kd_weight must be non-negative.")
    if byte_hidden_probe_mode not in {"fixed", "trainable"}:
        raise ValueError("byte_hidden_probe_mode must be fixed or trainable.")
    if byte_hidden_probe_anchor_weight < 0:
        raise ValueError("byte_hidden_probe_anchor_weight must be non-negative.")
    if block_io_kd_weight < 0:
        raise ValueError("block_io_kd_weight must be non-negative.")
    if block_io_ridge < 0:
        raise ValueError("block_io_ridge must be non-negative.")
    if block_io_loss not in {"smooth-l1", "cosine", "both"}:
        raise ValueError("block_io_loss must be smooth-l1, cosine, or both.")
    if (
        train_output_head_with_kd
        and projected_vocab_kd_weight == 0
        and stepwise_donor_kd_weight == 0
        and teacher_generate_lm_weight == 0
        and alm_kd_weight == 0
        and byte_kd_weight == 0
        and teacher_generate_projected_vocab_kd_weight == 0
        and teacher_generate_byte_kd_weight == 0
        and response_lm_weight == 0
        and first_token_lm_weight == 0
        and response_end_lm_weight == 0
        and focused_projected_vocab_kd_weight == 0
        and focused_lm_weight == 0
    ):
        raise ValueError(
            "train_output_head_with_kd requires projected, byte, stepwise, teacher-generated, ALM, or response LM loss."
        )
    if untie_output_head_for_kd and not train_output_head_with_kd:
        raise ValueError("untie_output_head_for_kd requires train_output_head_with_kd.")
    if module_output_kd_weight < 0:
        raise ValueError("module_output_kd_weight must be non-negative.")
    if module_output_targets not in MODULE_OUTPUT_TARGETS:
        raise ValueError("module_output_targets must be attention, mlp, or both.")
    if module_output_transport not in MODULE_OUTPUT_TRANSPORTS:
        raise ValueError("module_output_transport must be paired, sinkhorn, or both.")
    if module_output_temperature <= 0:
        raise ValueError("module_output_temperature must be positive.")
    if module_output_sinkhorn_iterations <= 0:
        raise ValueError("module_output_sinkhorn_iterations must be positive.")
    if on_policy_max_new_tokens <= 0:
        raise ValueError("on_policy_max_new_tokens must be positive.")
    if on_policy_interval <= 0:
        raise ValueError("on_policy_interval must be positive.")
    if stepwise_donor_max_new_tokens <= 0:
        raise ValueError("stepwise_donor_max_new_tokens must be positive.")
    if stepwise_donor_interval <= 0:
        raise ValueError("stepwise_donor_interval must be positive.")
    if projected_vocab_max_source_tokens <= 0:
        raise ValueError("projected_vocab_max_source_tokens must be positive.")
    if on_policy_response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("on_policy_response_loss_mode must be none, auto, assistant, or answer.")
    if on_policy_unlikelihood_window <= 0:
        raise ValueError("on_policy_unlikelihood_window must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if grad_clip < 0:
        raise ValueError("grad_clip must be non-negative.")
    if report_interval is not None and report_interval <= 0:
        raise ValueError("report_interval must be positive when provided.")
    if layer_behavior_eval_interval is not None and layer_behavior_eval_interval <= 0:
        raise ValueError("layer_behavior_eval_interval must be positive when provided.")
    if layer_checkpoint_interval < 0:
        raise ValueError("layer_checkpoint_interval must be non-negative.")
    if calibration_order not in {"original", "domain-balanced"}:
        raise ValueError("calibration_order must be original or domain-balanced.")

    from neural_forge.hf_xfamily import _canonical_token
    from neural_forge.hf_xfamily import _canonical_token_map
    from neural_forge.hf_xfamily import _untie_output_head
    from neural_forge.hf_xfamily import _donor_layer_group_for_target
    from neural_forge.hf_xfamily import _donor_layer_groups_by_energy
    from neural_forge.hf_xfamily import _representative_donor_layer_for_group

    torch, transformers = _torch_and_transformers()
    dtype = _dtype_from_name(torch, torch_dtype)
    candidate_dtype = _dtype_from_name(torch, candidate_torch_dtype)
    source_tokenizer = transformers.AutoTokenizer.from_pretrained(source_model, trust_remote_code=trust_remote_code)
    candidate_tokenizer = transformers.AutoTokenizer.from_pretrained(candidate_model, trust_remote_code=trust_remote_code)
    if source_tokenizer.pad_token_id is None and source_tokenizer.eos_token_id is not None:
        source_tokenizer.pad_token = source_tokenizer.eos_token
    if candidate_tokenizer.pad_token_id is None and candidate_tokenizer.eos_token_id is not None:
        candidate_tokenizer.pad_token = candidate_tokenizer.eos_token

    source = transformers.AutoModelForCausalLM.from_pretrained(
        source_model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    candidate = transformers.AutoModelForCausalLM.from_pretrained(
        candidate_model,
        torch_dtype=candidate_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    source.to(device)
    candidate.to(device)
    source.eval()
    candidate.train()
    if hasattr(source.config, "use_cache"):
        source.config.use_cache = False
    if hasattr(candidate.config, "use_cache"):
        candidate.config.use_cache = False

    source_layers = _recovery_decoder_layers(source)
    candidate_layers = _recovery_decoder_layers(candidate)
    if not source_layers or not candidate_layers:
        raise ValueError("Both source and candidate need decoder layers.")
    candidate_output_embeddings = candidate.get_output_embeddings()
    if byte_hidden_kd_weight > 0 and candidate_output_embeddings is None:
        raise ValueError("byte_hidden_kd_weight requires candidate output embeddings for the fixed byte probe.")
    if train_output_head_with_kd and candidate_output_embeddings is None:
        raise ValueError("train_output_head_with_kd requires an output embedding/lm_head module.")
    output_head_untied_for_kd = False
    if train_output_head_with_kd and untie_output_head_for_kd:
        output_head_untied_for_kd = bool(_untie_output_head(torch, candidate))
        candidate_output_embeddings = candidate.get_output_embeddings()
    output_head = candidate_output_embeddings if train_output_head_with_kd else None
    input_embeddings = candidate.get_input_embeddings() if train_output_head_with_kd else None
    output_head_tied_to_input = _output_head_tied_to_input(output_head, input_embeddings)
    output_head_training_warning = None
    if train_output_head_with_kd and output_head_tied_to_input:
        output_head_training_warning = (
            "output_head_is_tied_to_input_embeddings; training the head also mutates token embeddings"
        )
        if not allow_tied_output_head_training:
            raise ValueError(
                "train_output_head_with_kd would mutate tied input embeddings; "
                "pass allow_tied_output_head_training=True only for explicit tied-head experiments."
            )
        print(f"warning: {output_head_training_warning}", flush=True)

    anchors = _shared_recovery_anchors(
        source_tokenizer,
        candidate_tokenizer,
        max_anchors,
        _canonical_token_map,
        selection="stratified",
        min_chars=1,
    )
    if not anchors:
        raise ValueError("No shared decoded-token anchors found for blockwise reconstruction.")
    source_anchor_idx = torch.tensor([item[1] for item in anchors], dtype=torch.long, device=device)
    candidate_anchor_idx = torch.tensor([item[2] for item in anchors], dtype=torch.long, device=device)
    hidden_projection_report = _fit_shared_embedding_hidden_projection(
        torch,
        source,
        candidate,
        source_anchor_idx,
        candidate_anchor_idx,
        ridge=hidden_ridge,
        max_anchors=max_anchors,
    )
    hidden_projection = hidden_projection_report["map"].to(device=device, dtype=torch.float32)
    projected_vocab_groups = None
    projected_vocab_mask = None
    projected_vocab_weights = None
    projected_vocab_valid = None
    projected_vocab_group_report = None
    if (
        stepwise_donor_kd_weight > 0
        or projected_vocab_kd_weight > 0
        or focused_projected_vocab_kd_weight > 0
        or teacher_generate_projected_vocab_kd_weight > 0
    ):
        (
            projected_vocab_groups,
            projected_vocab_mask,
            projected_vocab_weights,
            projected_vocab_valid,
        ) = _candidate_to_source_token_groups(
            torch,
            source_tokenizer,
            candidate_tokenizer,
            max_source_tokens=projected_vocab_max_source_tokens,
            device=device,
        )
        projected_vocab_group_report = _projected_vocab_group_summary(
            torch,
            candidate_tokenizer,
            projected_vocab_mask,
            projected_vocab_valid,
            projected_vocab_weights,
        )
    source_byte_buckets = None
    candidate_byte_buckets = None
    byte_hidden_probe_weight = None
    byte_hidden_probe_anchor = None
    if byte_kd_weight > 0 or byte_hidden_kd_weight > 0 or teacher_generate_byte_kd_weight > 0:
        source_byte_buckets = _tokenizer_first_byte_buckets(torch, source_tokenizer, device=device)
        candidate_byte_buckets = _tokenizer_first_byte_buckets(torch, candidate_tokenizer, device=device)
    if byte_hidden_kd_weight > 0:
        fixed_byte_hidden_probe_weight = _first_byte_head_from_output_embeddings(
            torch,
            candidate_output_embeddings.weight,
            candidate_byte_buckets,
        ).to(device=device, dtype=torch.float32)
        if byte_hidden_probe_mode == "trainable":
            byte_hidden_probe_weight = torch.nn.Parameter(fixed_byte_hidden_probe_weight.detach().clone())
            byte_hidden_probe_anchor = fixed_byte_hidden_probe_weight.detach().clone()
        else:
            byte_hidden_probe_weight = fixed_byte_hidden_probe_weight

    texts = _order_recovery_texts(calibration_texts or RECOVERY_TEXTS, calibration_order)
    if not texts:
        raise ValueError("Need at least one blockwise reconstruction text.")
    eval_overlap_count = len(set(texts) & set(TEXT_EVALS))
    calibration_domain_counts = _recovery_domain_counts(texts) if calibration_order == "domain-balanced" else {}
    paired_batches = _make_paired_batches(
        source_tokenizer,
        candidate_tokenizer,
        texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        include_offsets=True,
        response_loss_mode=response_loss_mode,
    )
    teacher_generated_texts: list[str] = []
    teacher_generated_candidate_batches: list[dict[str, Any]] = []
    teacher_generated_paired_batches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    if (
        teacher_generate_lm_weight > 0
        or teacher_generate_hidden_kd_weight > 0
        or teacher_generate_projected_vocab_kd_weight > 0
        or teacher_generate_byte_kd_weight > 0
    ):
        teacher_generated_texts = _teacher_generated_lm_texts(
            torch,
            source,
            source_tokenizer,
            texts,
            on_policy_response_loss_mode,
            max_new_tokens=teacher_generate_max_new_tokens,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        if not teacher_generated_texts:
            raise ValueError(
                "teacher-generated block recovery requested but donor generation produced no training texts."
            )
        if teacher_generate_lm_weight > 0:
            teacher_generated_candidate_batches = _make_batches(
                candidate_tokenizer,
                teacher_generated_texts,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                response_loss_mode=teacher_generate_loss_mode,
                append_eos=False,
            )
        if (
            teacher_generate_hidden_kd_weight > 0
            or teacher_generate_projected_vocab_kd_weight > 0
            or teacher_generate_byte_kd_weight > 0
        ):
            teacher_generated_paired_batches = _make_paired_batches(
                source_tokenizer,
                candidate_tokenizer,
                teacher_generated_texts,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                include_offsets=True,
                response_loss_mode=teacher_generate_loss_mode,
                append_eos=False,
            )

    full_source_layer_groups = [
        _donor_layer_group_for_target(idx, len(candidate_layers), len(source_layers))
        for idx in range(len(candidate_layers))
    ]
    donor_layer_group_report: dict[str, Any] = {
        "strategy": "depth",
        "groups": [[int(item) for item in group] for group in full_source_layer_groups],
        "delta_scores": None,
    }
    if donor_layer_group_strategy == "activation-energy":
        source_delta_scores = _source_layer_delta_scores_from_batches(
            torch,
            source,
            paired_batches,
            source_layer_count=len(source_layers),
        )
        full_source_layer_groups = _donor_layer_groups_by_energy(source_delta_scores, len(candidate_layers))
        donor_layer_group_report = {
            "strategy": "activation-energy",
            "groups": [[int(item) for item in group] for group in full_source_layer_groups],
            "delta_scores": [float(item) for item in source_delta_scores],
            "total_delta_score": float(sum(source_delta_scores)),
        }
    elif donor_layer_group_strategy == "activation-geometry":
        geometry = _source_layer_geometry_scores_from_batches(
            torch,
            source,
            paired_batches,
            source_layer_count=len(source_layers),
        )
        geometry_scores = geometry["geometry_scores"]
        full_source_layer_groups = _donor_layer_groups_by_energy(geometry_scores, len(candidate_layers))
        donor_layer_group_report = {
            "strategy": "activation-geometry",
            "groups": [[int(item) for item in group] for group in full_source_layer_groups],
            "delta_scores": [float(item) for item in geometry["delta_scores"]],
            "representative_scores": [float(item) for item in geometry_scores],
            "geometry_scores": [float(item) for item in geometry_scores],
            "total_geometry_score": float(sum(geometry_scores)),
            "layer_geometry": geometry["layer_geometry"],
        }
    elif donor_layer_group_strategy == "activation-ot":
        from neural_forge.hf_xfamily import _activation_layer_ot_groups

        geometry = _source_layer_geometry_scores_from_batches(
            torch,
            source,
            paired_batches,
            source_layer_count=len(source_layers),
        )
        donor_layer_group_report = _activation_layer_ot_groups(
            torch,
            geometry["geometry_scores"],
            target_layer_count=len(candidate_layers),
        )
        full_source_layer_groups = donor_layer_group_report["groups"]
        donor_layer_group_report = {
            **donor_layer_group_report,
            "strategy": "activation-ot",
            "delta_scores": [float(item) for item in geometry["delta_scores"]],
            "representative_scores": [float(item) for item in geometry["geometry_scores"]],
            "geometry_scores": [float(item) for item in geometry["geometry_scores"]],
            "total_geometry_score": float(sum(geometry["geometry_scores"])),
            "layer_geometry": geometry["layer_geometry"],
        }
    elif donor_layer_group_strategy in {"activation-cka", "activation-cka-weighted"}:
        cka = _source_layer_cka_groups_from_batches(
            torch,
            source,
            paired_batches,
            source_layer_count=len(source_layers),
            target_layer_count=len(candidate_layers),
        )
        full_source_layer_groups = cka["groups"]
        donor_layer_group_report = cka
        if donor_layer_group_strategy == "activation-cka-weighted":
            from neural_forge.hf_xfamily import _activation_cka_group_layer_weights

            donor_layer_group_report = {
                **cka,
                "strategy": "activation-cka-weighted",
                "group_layer_weights": _activation_cka_group_layer_weights(cka),
                "group_layer_weight_mode": "assignment_similarity_times_delta_score",
            }

    layer_count = len(candidate_layers) if layer_limit is None else min(int(layer_limit), len(candidate_layers))
    interval = int(report_interval or max(1, steps_per_layer // 4)) if steps_per_layer > 0 else 1
    behavior_interval = int(layer_behavior_eval_interval or interval)
    target = Path(output_dir)
    if layer_checkpoint_interval > 0:
        target.mkdir(parents=True, exist_ok=True)
    layer_reports = []
    gate_texts = list(TEXT_EVALS[: int(layer_gate_max_texts)])
    layer_gate: dict[str, Any] = {
        "metric": layer_gate_metric,
        "max_ppl_ratio": float(layer_gate_max_ppl_ratio),
        "max_texts": int(layer_gate_max_texts),
        "max_behavior_drop": float(layer_gate_max_behavior_drop),
        "behavior_max_new_tokens": int(layer_gate_behavior_max_new_tokens),
    }
    layer_gate_behavior_probes = None
    layer_gate_generation_prompts = None
    layer_gate_suite_report = None
    if layer_gate_eval_suite is not None:
        loaded_gate_suite = _load_eval_suite(layer_gate_eval_suite)
        layer_gate_behavior_probes = list(loaded_gate_suite.get("behavior_probes") or [])
        layer_gate_generation_prompts = list(loaded_gate_suite.get("generation_prompts") or [])
        layer_gate_suite_report = {
            "source": loaded_gate_suite.get("source"),
            "behavior_probe_count": int(len(layer_gate_behavior_probes)),
            "generation_prompt_count": int(len(layer_gate_generation_prompts)),
            "max_behavior_probes": int(layer_gate_max_texts),
            "max_generation_prompts": (
                None if layer_gate_max_generation_prompts is None else int(layer_gate_max_generation_prompts)
            ),
        }
        layer_gate["eval_suite"] = layer_gate_suite_report
    previous_gate_ppl: float | None = None
    previous_gate_behavior: dict[str, Any] | None = None
    with torch.no_grad():
        if layer_gate_metric == "ppl":
            candidate.eval()
            previous_gate_ppl = _heldout_ppl_mean(torch, candidate, candidate_tokenizer, gate_texts, device, max_length)
            candidate.train()
            layer_gate["initial_ppl"] = previous_gate_ppl
            layer_gate["texts"] = gate_texts
        elif layer_gate_metric == "behavior":
            candidate.eval()
            previous_gate_behavior = _heldout_behavior_summary(
                torch,
                candidate,
                candidate_tokenizer,
                device,
                int(layer_gate_behavior_max_new_tokens),
                int(layer_gate_max_texts),
                layer_gate_behavior_probes,
                layer_gate_generation_prompts,
                layer_gate_max_generation_prompts,
            )
            candidate.train()
            layer_gate["initial_behavior"] = previous_gate_behavior

    source_layer_group_weights_by_group = _source_layer_group_weight_lookup(donor_layer_group_report)
    for candidate_idx in range(layer_count):
        source_group = [int(item) for item in full_source_layer_groups[int(candidate_idx)]]
        representative_scores = donor_layer_group_report.get("representative_scores")
        if representative_scores is None:
            representative_scores = donor_layer_group_report.get("delta_scores")
        source_idx = _representative_donor_layer_for_group(source_group, representative_scores)
        source_group = (
            [source_idx]
            if donor_layer_fold_mode == "representative"
            else source_group
        )
        source_group_weights = (
            None
            if donor_layer_fold_mode == "representative"
            else source_layer_group_weights_by_group.get(tuple(int(item) for item in source_group))
        )
        layer_hidden_projection = hidden_projection
        layer_hidden_projection_report: dict[str, Any] | None = None
        if hidden_projection_scope in {"layer-input", "layer-input-gated"}:
            layer_hidden_projection_report = _fit_cross_tokenizer_layer_input_hidden_projection(
                torch,
                source,
                candidate,
                source_tokenizer,
                candidate_tokenizer,
                paired_batches,
                source_group,
                candidate_idx,
                _canonical_token,
                hidden_ridge,
                pre_reconstruction_max_tokens_per_layer,
                baseline_projection=hidden_projection if hidden_projection_scope == "layer-input-gated" else None,
                min_relative_improvement=pre_reconstruction_min_relative_improvement,
                max_cosine_drop=pre_reconstruction_max_cosine_drop,
                min_fitted_cosine=pre_reconstruction_min_fitted_cosine,
            )
            if not layer_hidden_projection_report.get("skipped"):
                layer_hidden_projection = layer_hidden_projection_report["map"].to(
                    device=device,
                    dtype=torch.float32,
                )
        candidate_layer = candidate_layers[candidate_idx]
        snapshot_modules = _block_recovery_snapshot_modules(
            candidate_layers,
            candidate_idx,
            candidate_layer,
            output_head,
            pre_reconstruction_target,
        )
        layer_state = (
            _snapshot_named_module_states(snapshot_modules)
            if layer_gate_metric in {"ppl", "behavior"} or save_best_layer_loss or save_best_layer_behavior
            else None
        )
        pre_reconstruction_report = None
        if pre_reconstruction_target != "none":
            pre_reconstruction_report = _reconstruct_cross_tokenizer_layer(
                torch,
                source,
                candidate,
                source_layers[int(source_idx)],
                candidate_layer,
                source_layers,
                candidate_layers,
                int(candidate_idx),
                source_group,
                source_tokenizer,
                candidate_tokenizer,
                paired_batches,
                layer_hidden_projection,
                _canonical_token,
                float(pre_reconstruction_ridge),
                float(pre_reconstruction_alpha),
                pre_reconstruction_target,
                pre_reconstruction_objective,
                int(pre_reconstruction_cosine_steps),
                float(pre_reconstruction_cosine_lr),
                float(pre_reconstruction_cosine_l2),
                pre_reconstruction_fit_gate,
                float(pre_reconstruction_validation_fraction),
                int(pre_reconstruction_min_validation_rows),
                float(pre_reconstruction_min_relative_improvement),
                float(pre_reconstruction_max_cosine_drop),
                int(pre_reconstruction_max_tokens_per_layer),
                donor_layer_fold_mode=donor_layer_fold_mode,
                source_layer_group_weights=source_group_weights,
            )
        trainable_modules = [candidate_layer]
        if train_output_head_with_kd and output_head is not None:
            trainable_modules.append(output_head)
        trainable_params = _set_only_modules_trainable(candidate, trainable_modules)
        optimizer_params = _unique_trainable_parameters(trainable_modules)
        if byte_hidden_probe_mode == "trainable" and byte_hidden_probe_weight is not None:
            optimizer_params.append(byte_hidden_probe_weight)
        optimizer = torch.optim.AdamW(optimizer_params, lr=lr)
        before_alignment = _cross_tokenizer_block_alignment_state(
            torch,
            source,
            candidate,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            source_group,
            candidate_idx,
            layer_hidden_projection,
            _canonical_token,
            donor_layer_fold_mode,
            block_target,
            span_max_pairs,
            source_group_weights,
        )
        before_metrics = _cross_tokenizer_block_alignment_metrics(
            torch,
            before_alignment,
            layer_hidden_projection,
            hidden_metric,
        )
        compensation_report = None
        if residual_magnitude_compensation != "none":
            compensation_report = _apply_layer_residual_magnitude_compensation(
                torch,
                candidate_layer,
                before_alignment,
                before_metrics,
                float(residual_compensation_min_scale),
                float(residual_compensation_max_scale),
                residual_magnitude_compensation,
                float(residual_compensation_linear_ridge),
                float(residual_compensation_alpha),
                int(residual_compensation_linear_rank),
                source=source,
                candidate=candidate,
                source_layers=source_layers,
                source_layer_group=source_group,
                source_attention_layer=source_layers[max(0, min(source_group[0], len(source_layers) - 1))]
                if source_group
                else source_layers[int(source_idx)],
                source_mlp_layer=source_layers[max(0, min(source_group[-1], len(source_layers) - 1))]
                if source_group
                else source_layers[int(source_idx)],
                source_tokenizer=source_tokenizer,
                candidate_tokenizer=candidate_tokenizer,
                paired_batches=paired_batches,
                hidden_projection=layer_hidden_projection,
                canonical_fn=_canonical_token,
                donor_layer_fold_mode=donor_layer_fold_mode,
                source_layer_group_weights=source_group_weights,
                fit_gate=pre_reconstruction_fit_gate,
                fit_gate_validation_fraction=float(pre_reconstruction_validation_fraction),
                fit_gate_min_validation_rows=int(pre_reconstruction_min_validation_rows),
                fit_gate_min_relative_improvement=float(pre_reconstruction_min_relative_improvement),
                fit_gate_max_cosine_drop=float(pre_reconstruction_max_cosine_drop),
                max_tokens_per_layer=int(pre_reconstruction_max_tokens_per_layer),
            )
        losses = []
        skipped_steps = 0
        nonfinite_step_count = 0
        nonfinite_loss_steps = 0
        nonfinite_gradient_steps = 0
        nonfinite_parameter_steps = 0
        nonfinite_events: list[dict[str, Any]] = []
        best_loss = None
        best_step = None
        best_state = None
        best_behavior_summary = None
        best_behavior_step = None
        best_behavior_loss = None
        best_behavior_state = None
        behavior_history: list[dict[str, Any]] = []
        focused_span_pair_total = 0
        focused_span_pair_active_steps = 0
        focused_span_pair_max = 0
        if save_best_layer_behavior:
            with torch.no_grad():
                candidate.eval()
                behavior_summary = _heldout_behavior_summary(
                    torch,
                    candidate,
                    candidate_tokenizer,
                    device,
                    int(layer_gate_behavior_max_new_tokens),
                    int(layer_gate_max_texts),
                    layer_gate_behavior_probes,
                    layer_gate_generation_prompts,
                    layer_gate_max_generation_prompts,
                )
                candidate.train()
            behavior_entry = _behavior_history_entry(0, None, behavior_summary)
            behavior_history.append(behavior_entry)
            _print_behavior_checkpoint(f"xfamily_block_behavior layer={candidate_idx}", behavior_entry)
            best_behavior_summary = dict(behavior_entry)
            best_behavior_step = 0
            best_behavior_loss = math.inf
            best_behavior_state = _snapshot_named_module_states(snapshot_modules)
        for step in range(1, steps_per_layer + 1):
            source_batch, candidate_batch = paired_batches[(step - 1) % len(paired_batches)]
            optimizer.zero_grad(set_to_none=True)
            def handle_nonfinite_step(kind: str, names: Sequence[str] | None = None) -> bool:
                nonlocal nonfinite_step_count
                nonfinite_step_count += 1
                event = {
                    "layer": int(candidate_idx),
                    "step": int(step),
                    "kind": kind,
                    "names": list(names or []),
                }
                nonfinite_events.append(event)
                over_limit = (
                    max_nonfinite_steps_per_layer > 0
                    and nonfinite_step_count > int(max_nonfinite_steps_per_layer)
                )
                if nonfinite_step_policy == "raise" or over_limit:
                    joined = ", ".join(names or [])
                    suffix = f": {joined}" if joined else ""
                    raise RuntimeError(
                        f"Non-finite blockwise reconstruction {kind} at layer {candidate_idx}, "
                        f"step {step}{suffix}; aborting."
                    )
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"xfamily_block_nonfinite_skip layer={candidate_idx} step={step} "
                    f"kind={kind} count={nonfinite_step_count} names={','.join(names or [])}",
                    flush=True,
                )
                return True

            with torch.no_grad():
                teacher = source(
                    **_model_forward_inputs(source_batch),
                    use_cache=False,
                    output_hidden_states=True,
                )
            student = candidate(
                **_model_forward_inputs(candidate_batch),
                use_cache=False,
                output_hidden_states=True,
            )
            block_span_pairs = _cross_tokenizer_span_pairs_for_mode(
                torch,
                source_tokenizer,
                candidate_tokenizer,
                source_batch,
                candidate_batch,
                _canonical_token,
                max_pairs=span_max_pairs,
                shift_positions=False,
                mode=span_pair_mode,
            )
            loss, pair_count = _cross_tokenizer_block_hidden_loss(
                torch,
                student,
                teacher,
                source_tokenizer,
                candidate_tokenizer,
                source_batch,
                candidate_batch,
                source_group,
                candidate_idx,
                layer_hidden_projection,
                _canonical_token,
                donor_layer_fold_mode,
                hidden_metric,
                block_target,
                span_max_pairs,
                source_group_weights=source_group_weights,
                span_pairs=block_span_pairs,
            )
            block_loss_value = float(loss.detach().cpu())
            block_io_loss_value = None
            block_io_pair_count = 0
            if block_io_kd_weight > 0:
                block_io_loss_tensor, block_io_pair_count = _cross_tokenizer_block_io_reconstruction_loss(
                    torch,
                    student,
                    teacher,
                    source_group,
                    candidate_idx,
                    donor_layer_fold_mode,
                    source_group_weights,
                    block_target,
                    block_span_pairs,
                    float(block_io_ridge),
                    block_io_loss,
                )
                block_io_loss_value = float(block_io_loss_tensor.detach().cpu())
                loss = loss + float(block_io_kd_weight) * block_io_loss_tensor
            output_span_pairs = None
            output_span_pair_count = 0
            if projected_vocab_kd_weight > 0 or byte_kd_weight > 0 or byte_hidden_kd_weight > 0:
                output_span_pairs = _cross_tokenizer_span_pairs_for_mode(
                    torch,
                    source_tokenizer,
                    candidate_tokenizer,
                    source_batch,
                    candidate_batch,
                    _canonical_token,
                    max_pairs=span_max_pairs,
                    shift_positions=True,
                    mode=span_pair_mode,
                )
                output_span_pair_count = int(output_span_pairs.shape[0])
            focus_spans = None
            focus_pairs = None
            focus_pair_count = 0
            focused_block_hidden_loss = None
            focused_projected_vocab_loss = None
            focused_lm_loss = None
            if focused_block_hidden_kd_weight > 0 or focused_projected_vocab_kd_weight > 0 or focused_lm_weight > 0:
                focus_spans = _cross_tokenizer_focus_span_pairs(
                    torch,
                    source_batch,
                    candidate_batch,
                    response_loss_mode,
                    token_focus,
                    span_max_pairs,
                )
                focus_pair_count = sum(int(item["candidate_positions"].numel()) for item in focus_spans)
                focus_pairs = _focus_spans_to_span_pairs(torch, focus_spans, device=student.logits.device)
                focused_span_pair_total += int(focus_pair_count)
                focused_span_pair_max = max(int(focused_span_pair_max), int(focus_pair_count))
                if int(focus_pair_count) > 0:
                    focused_span_pair_active_steps += 1
            if focused_block_hidden_kd_weight > 0 and focus_pairs is not None and int(focus_pairs.numel()) > 0:
                focused_block_hidden_loss, _focused_hidden_count = _cross_tokenizer_block_hidden_loss(
                    torch,
                    student,
                    teacher,
                    source_tokenizer,
                    candidate_tokenizer,
                    source_batch,
                    candidate_batch,
                    source_group,
                    candidate_idx,
                    layer_hidden_projection,
                    _canonical_token,
                    donor_layer_fold_mode,
                    hidden_metric,
                    block_target,
                    span_max_pairs,
                    source_group_weights=source_group_weights,
                    span_pairs=focus_pairs,
                )
                loss = loss + float(focused_block_hidden_kd_weight) * focused_block_hidden_loss
            projected_vocab_loss = None
            projected_vocab_pair_count = int(output_span_pair_count) if projected_vocab_kd_weight > 0 else 0
            if projected_vocab_kd_weight > 0:
                if (
                    projected_vocab_groups is None
                    or projected_vocab_mask is None
                    or projected_vocab_weights is None
                    or projected_vocab_valid is None
                ):
                    raise ValueError("projected vocab maps are required when projected_vocab_kd_weight is positive.")
                if projected_vocab_pair_count > 0:
                    shift_student = torch.nan_to_num(
                        student.logits[:, :-1, :].contiguous().to(torch.float32),
                        nan=0.0,
                        posinf=1e4,
                        neginf=-1e4,
                    )
                    projected_vocab_loss = _cross_tokenizer_projected_vocab_loss(
                        torch,
                        shift_student,
                        teacher,
                        output_span_pairs,
                        projected_vocab_groups,
                        projected_vocab_mask,
                        projected_vocab_weights,
                        projected_vocab_valid,
                        float(temperature),
                        projected_vocab_kd_mode,
                    )
                    loss = loss + float(projected_vocab_kd_weight) * projected_vocab_loss
            if focused_projected_vocab_kd_weight > 0:
                if (
                    projected_vocab_groups is None
                    or projected_vocab_mask is None
                    or projected_vocab_weights is None
                    or projected_vocab_valid is None
                ):
                    raise ValueError("projected vocab maps are required when focused_projected_vocab_kd_weight is positive.")
                if focus_pairs is not None and int(focus_pairs.numel()) > 0:
                    shift_student = torch.nan_to_num(
                        student.logits[:, :-1, :].contiguous().to(torch.float32),
                        nan=0.0,
                        posinf=1e4,
                        neginf=-1e4,
                    )
                    focused_projected_vocab_loss = _cross_tokenizer_projected_vocab_loss(
                        torch,
                        shift_student,
                        teacher,
                        focus_pairs,
                        projected_vocab_groups,
                        projected_vocab_mask,
                        projected_vocab_weights,
                        projected_vocab_valid,
                        float(temperature),
                        projected_vocab_kd_mode,
                    )
                    loss = loss + float(focused_projected_vocab_kd_weight) * focused_projected_vocab_loss
            if focused_lm_weight > 0 and focus_spans:
                focus_mask = _focus_spans_to_shift_mask(
                    torch,
                    focus_spans,
                    candidate_batch["input_ids"][:, 1:].shape,
                    device=student.logits.device,
                )
                if candidate_batch.get("loss_mask") is not None:
                    focus_mask = focus_mask & candidate_batch["loss_mask"][:, 1:].contiguous().to(
                        device=student.logits.device
                    ).bool()
                if int(focus_mask.sum().detach().cpu().item()) > 0:
                    focused_lm_loss = _candidate_shift_lm_loss(
                        torch,
                        student,
                        candidate_batch["input_ids"],
                        focus_mask,
                    )
                    loss = loss + float(focused_lm_weight) * focused_lm_loss
            byte_kd_loss = None
            byte_kd_pair_count = int(output_span_pair_count) if byte_kd_weight > 0 else 0
            if byte_kd_weight > 0:
                if source_byte_buckets is None or candidate_byte_buckets is None:
                    raise ValueError("byte buckets are required when byte_kd_weight is positive.")
                if byte_kd_pair_count > 0:
                    shift_student = torch.nan_to_num(
                        student.logits[:, :-1, :].contiguous().to(torch.float32),
                        nan=0.0,
                        posinf=1e4,
                        neginf=-1e4,
                    )
                    byte_kd_loss = _cross_tokenizer_first_byte_loss(
                        torch,
                        shift_student,
                        teacher,
                        output_span_pairs,
                        candidate_byte_buckets,
                        source_byte_buckets,
                        float(temperature),
                    )
                    loss = loss + float(byte_kd_weight) * byte_kd_loss
            byte_hidden_loss = None
            byte_hidden_probe_anchor_loss = None
            byte_hidden_pair_count = int(output_span_pair_count) if byte_hidden_kd_weight > 0 else 0
            if byte_hidden_kd_weight > 0:
                if source_byte_buckets is None or byte_hidden_probe_weight is None:
                    raise ValueError("source byte buckets and a byte hidden probe are required for byte_hidden_kd_weight.")
                if byte_hidden_pair_count > 0:
                    byte_hidden_loss = _cross_tokenizer_hidden_first_byte_loss(
                        torch,
                        student,
                        teacher,
                        output_span_pairs,
                        int(candidate_idx),
                        byte_hidden_probe_weight,
                        source_byte_buckets,
                        float(temperature),
                    )
                    loss = loss + float(byte_hidden_kd_weight) * byte_hidden_loss
                    if (
                        byte_hidden_probe_mode == "trainable"
                        and byte_hidden_probe_anchor is not None
                        and byte_hidden_probe_anchor_weight > 0
                    ):
                        byte_hidden_probe_anchor_loss = _byte_hidden_probe_anchor_loss(
                            torch,
                            byte_hidden_probe_weight,
                            byte_hidden_probe_anchor,
                        )
                        loss = loss + float(byte_hidden_probe_anchor_weight) * byte_hidden_probe_anchor_loss
            alm_loss = None
            alm_count = 0
            if alm_kd_weight > 0:
                shift_student = torch.nan_to_num(
                    student.logits[:, :-1, :].contiguous().to(torch.float32),
                    nan=0.0,
                    posinf=1e4,
                    neginf=-1e4,
                )
                alm_loss, alm_count = _cross_tokenizer_alm_chunk_loss(
                    torch,
                    shift_student=shift_student,
                    teacher=teacher,
                    candidate_input_ids=candidate_batch["input_ids"],
                    source_input_ids=source_batch["input_ids"],
                    source_batch=source_batch,
                    candidate_batch=candidate_batch,
                    response_loss_mode=response_loss_mode,
                    max_chunks=int(alm_max_chunks),
                    min_chars=int(alm_min_chars),
                )
                if int(alm_count) > 0:
                    loss = loss + float(alm_kd_weight) * alm_loss
            module_output_loss = None
            module_output_counts: dict[str, Any] = {
                "attention_pairs": 0,
                "mlp_pairs": 0,
                "targets": module_output_targets,
                "transport": module_output_transport,
            }
            if module_output_kd_weight > 0:
                module_output_loss, module_output_counts = _cross_tokenizer_block_module_output_loss(
                    torch,
                    source,
                    candidate,
                    source_layers,
                    candidate_layers,
                    source_tokenizer,
                    candidate_tokenizer,
                    source_batch,
                    candidate_batch,
                    source_group,
                    candidate_idx,
                    layer_hidden_projection,
                    _canonical_token,
                    donor_layer_fold_mode,
                    source_group_weights,
                    module_output_targets,
                    module_output_transport,
                    float(module_output_temperature),
                    int(module_output_sinkhorn_iterations),
                    span_max_pairs,
                    span_pair_mode,
                )
                loss = loss + float(module_output_kd_weight) * module_output_loss
            if (
                int(pair_count) <= 0
                and int(block_io_pair_count) <= 0
                and int(module_output_counts.get("attention_pairs", 0)) <= 0
                and int(module_output_counts.get("mlp_pairs", 0)) <= 0
                and (alm_kd_weight == 0 or int(alm_count) <= 0)
                and lm_weight == 0
                and response_lm_weight == 0
                and first_token_lm_weight == 0
                and response_end_lm_weight == 0
                and focused_block_hidden_kd_weight == 0
                and focused_projected_vocab_kd_weight == 0
                and focused_lm_weight == 0
                and on_policy_anchor_kd_weight == 0
                and on_policy_unlikelihood_weight == 0
                and on_policy_hidden_kd_weight == 0
                and int(projected_vocab_pair_count) <= 0
                and int(byte_kd_pair_count) <= 0
                and int(byte_hidden_pair_count) <= 0
                and stepwise_donor_kd_weight == 0
                and teacher_generate_lm_weight == 0
                and teacher_generate_hidden_kd_weight == 0
                and teacher_generate_projected_vocab_kd_weight == 0
                and teacher_generate_byte_kd_weight == 0
            ):
                skipped_steps += 1
                continue
            lm_loss = None
            if lm_weight > 0:
                lm_loss = _candidate_lm_stabilizer_loss(
                    torch,
                    student,
                    candidate_batch["input_ids"],
                    candidate_batch.get("attention_mask"),
                )
                loss = loss + float(lm_weight) * lm_loss
            response_lm_loss = None
            if response_lm_weight > 0:
                response_lm_loss = _candidate_lm_stabilizer_loss(
                    torch,
                    student,
                    candidate_batch["input_ids"],
                    candidate_batch.get("attention_mask"),
                    candidate_batch.get("loss_mask"),
                )
                loss = loss + float(response_lm_weight) * response_lm_loss
            first_token_lm_loss = None
            if first_token_lm_weight > 0:
                first_mask = _first_response_shift_mask(
                    torch,
                    candidate_batch.get("loss_mask"),
                    candidate_batch.get("attention_mask"),
                )
                first_token_lm_loss = _candidate_shift_lm_loss(
                    torch,
                    student,
                    candidate_batch["input_ids"],
                    first_mask,
                )
                loss = loss + float(first_token_lm_weight) * first_token_lm_loss
            response_end_lm_loss = None
            if response_end_lm_weight > 0:
                end_mask = _last_response_shift_mask(
                    torch,
                    candidate_batch.get("loss_mask"),
                    candidate_batch.get("attention_mask"),
                )
                response_end_lm_loss = _candidate_shift_lm_loss(
                    torch,
                    student,
                    candidate_batch["input_ids"],
                    end_mask,
                )
                loss = loss + float(response_end_lm_weight) * response_end_lm_loss
            on_policy_loss = None
            on_policy_count = 0
            if (
                (on_policy_anchor_kd_weight > 0 or on_policy_unlikelihood_weight > 0)
                and (step == 1 or step % int(on_policy_interval) == 0)
            ):
                on_policy_texts = candidate_batch.get("raw_texts", candidate_batch.get("texts", []))
                on_policy_loss, on_policy_count = _cross_tokenizer_on_policy_rollout_loss(
                    torch,
                    source,
                    candidate,
                    source_tokenizer,
                    candidate_tokenizer,
                    on_policy_texts,
                    on_policy_response_loss_mode,
                    int(on_policy_max_new_tokens),
                    int(max_length),
                    device,
                    source_anchor_idx,
                    candidate_anchor_idx,
                    float(temperature),
                    float(on_policy_anchor_kd_weight),
                    float(on_policy_unlikelihood_weight),
                    int(on_policy_unlikelihood_window),
                )
                loss = loss + on_policy_loss
            on_policy_hidden_loss = None
            on_policy_hidden_count = 0
            if on_policy_hidden_kd_weight > 0 and (step == 1 or step % int(on_policy_interval) == 0):
                on_policy_texts = candidate_batch.get("raw_texts", candidate_batch.get("texts", []))
                on_policy_hidden_loss, on_policy_hidden_count = _cross_tokenizer_block_on_policy_hidden_loss(
                    torch,
                    source,
                    candidate,
                    source_tokenizer,
                    candidate_tokenizer,
                    on_policy_texts,
                    on_policy_response_loss_mode,
                    int(on_policy_max_new_tokens),
                    int(max_length),
                    device,
                    source_group,
                    candidate_idx,
                    layer_hidden_projection,
                    _canonical_token,
                    donor_layer_fold_mode,
                    hidden_metric,
                    block_target,
                    span_max_pairs,
                    source_layer_group_weights=source_group_weights,
                )
                loss = loss + float(on_policy_hidden_kd_weight) * on_policy_hidden_loss
            teacher_generate_lm_loss = None
            teacher_generate_hidden_loss = None
            teacher_generate_hidden_count = 0
            teacher_generate_projected_vocab_loss = None
            teacher_generate_projected_vocab_count = 0
            teacher_generate_byte_loss = None
            teacher_generate_byte_count = 0
            if teacher_generated_candidate_batches:
                teacher_candidate_batch = teacher_generated_candidate_batches[
                    (step - 1) % len(teacher_generated_candidate_batches)
                ]
                teacher_generated_student_lm = candidate(
                    **_model_forward_inputs(teacher_candidate_batch),
                    use_cache=False,
                )
                teacher_generate_lm_loss = _candidate_lm_stabilizer_loss(
                    torch,
                    teacher_generated_student_lm,
                    teacher_candidate_batch["input_ids"],
                    teacher_candidate_batch.get("attention_mask"),
                )
                loss = loss + float(teacher_generate_lm_weight) * teacher_generate_lm_loss
            if teacher_generated_paired_batches:
                teacher_source_batch, teacher_candidate_batch = teacher_generated_paired_batches[
                    (step - 1) % len(teacher_generated_paired_batches)
                ]
                with torch.no_grad():
                    teacher_generated_teacher = source(
                        **_model_forward_inputs(teacher_source_batch),
                        use_cache=False,
                        output_hidden_states=teacher_generate_hidden_kd_weight > 0,
                    )
                teacher_generated_student = candidate(
                    **_model_forward_inputs(teacher_candidate_batch),
                    use_cache=False,
                    output_hidden_states=teacher_generate_hidden_kd_weight > 0,
                )
                teacher_output_span_pairs = _cross_tokenizer_span_pairs(
                    torch,
                    source_tokenizer,
                    candidate_tokenizer,
                    teacher_source_batch,
                    teacher_candidate_batch,
                    _canonical_token,
                    max_pairs=span_max_pairs,
                    shift_positions=True,
                )
                teacher_generate_projected_vocab_count = (
                    int(teacher_output_span_pairs.shape[0])
                    if teacher_generate_projected_vocab_kd_weight > 0
                    else 0
                )
                if teacher_generate_projected_vocab_kd_weight > 0 and teacher_generate_projected_vocab_count > 0:
                    if (
                        projected_vocab_groups is None
                        or projected_vocab_mask is None
                        or projected_vocab_weights is None
                        or projected_vocab_valid is None
                    ):
                        raise ValueError(
                            "projected vocab maps are required when teacher-generated projected KD is positive."
                        )
                    teacher_shift_student = torch.nan_to_num(
                        teacher_generated_student.logits[:, :-1, :].contiguous().to(torch.float32),
                        nan=0.0,
                        posinf=1e4,
                        neginf=-1e4,
                    )
                    teacher_generate_projected_vocab_loss = _cross_tokenizer_projected_vocab_loss(
                        torch,
                        teacher_shift_student,
                        teacher_generated_teacher,
                        teacher_output_span_pairs,
                        projected_vocab_groups,
                        projected_vocab_mask,
                        projected_vocab_weights,
                        projected_vocab_valid,
                        float(temperature),
                        projected_vocab_kd_mode,
                    )
                    loss = loss + float(teacher_generate_projected_vocab_kd_weight) * teacher_generate_projected_vocab_loss
                teacher_generate_byte_count = (
                    int(teacher_output_span_pairs.shape[0]) if teacher_generate_byte_kd_weight > 0 else 0
                )
                if teacher_generate_byte_kd_weight > 0 and teacher_generate_byte_count > 0:
                    if source_byte_buckets is None or candidate_byte_buckets is None:
                        raise ValueError("byte buckets are required when teacher-generated byte KD is positive.")
                    teacher_shift_student = torch.nan_to_num(
                        teacher_generated_student.logits[:, :-1, :].contiguous().to(torch.float32),
                        nan=0.0,
                        posinf=1e4,
                        neginf=-1e4,
                    )
                    teacher_generate_byte_loss = _cross_tokenizer_first_byte_loss(
                        torch,
                        teacher_shift_student,
                        teacher_generated_teacher,
                        teacher_output_span_pairs,
                        candidate_byte_buckets,
                        source_byte_buckets,
                        float(temperature),
                    )
                    loss = loss + float(teacher_generate_byte_kd_weight) * teacher_generate_byte_loss
                if teacher_generate_hidden_kd_weight > 0:
                    teacher_generate_hidden_loss, teacher_generate_hidden_count = _cross_tokenizer_block_hidden_loss(
                        torch,
                        teacher_generated_student,
                        teacher_generated_teacher,
                        source_tokenizer,
                        candidate_tokenizer,
                        teacher_source_batch,
                        teacher_candidate_batch,
                        source_group,
                        candidate_idx,
                        layer_hidden_projection,
                        _canonical_token,
                        donor_layer_fold_mode,
                        hidden_metric,
                        block_target,
                        span_max_pairs,
                        source_group_weights=source_group_weights,
                    )
                    loss = loss + float(teacher_generate_hidden_kd_weight) * teacher_generate_hidden_loss
            stepwise_donor_loss = None
            stepwise_donor_count = 0
            if stepwise_donor_kd_weight > 0 and (step == 1 or step % int(stepwise_donor_interval) == 0):
                if (
                    projected_vocab_groups is None
                    or projected_vocab_mask is None
                    or projected_vocab_weights is None
                    or projected_vocab_valid is None
                ):
                    raise ValueError("projected vocab maps are required when stepwise donor KD is positive.")
                stepwise_donor_loss, stepwise_donor_count = _cross_tokenizer_stepwise_prefix_kd_loss(
                    torch,
                    source,
                    candidate,
                    source_tokenizer,
                    candidate_tokenizer,
                    candidate_batch.get("raw_texts", candidate_batch.get("texts", [])),
                    on_policy_response_loss_mode,
                    int(stepwise_donor_max_new_tokens),
                    int(max_length),
                    device,
                    projected_vocab_groups,
                    projected_vocab_mask,
                    projected_vocab_weights,
                    projected_vocab_valid,
                    float(temperature),
                )
                loss = loss + float(stepwise_donor_kd_weight) * stepwise_donor_loss
            if not bool(torch.isfinite(loss).detach().cpu()):
                nonfinite_loss_steps += 1
                handle_nonfinite_step("loss")
                continue
            loss.backward()
            bad_gradient_names = _nonfinite_gradient_names(torch, candidate)
            if bad_gradient_names:
                nonfinite_gradient_steps += 1
                handle_nonfinite_step("gradients", bad_gradient_names)
                continue
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(optimizer_params, grad_clip)
            pre_step_state = (
                _snapshot_named_module_states(snapshot_modules)
                if nonfinite_step_policy == "skip-step"
                else None
            )
            optimizer.step()
            bad_parameter_names = _nonfinite_parameter_names(torch, candidate)
            if bad_parameter_names:
                nonfinite_parameter_steps += 1
                if pre_step_state is not None:
                    _restore_named_module_states(snapshot_modules, pre_step_state)
                handle_nonfinite_step("parameters", bad_parameter_names)
                continue
            loss_value = float(loss.detach().cpu())
            lm_loss_value = None if lm_loss is None else float(lm_loss.detach().cpu())
            response_lm_loss_value = None if response_lm_loss is None else float(response_lm_loss.detach().cpu())
            first_token_lm_loss_value = (
                None if first_token_lm_loss is None else float(first_token_lm_loss.detach().cpu())
            )
            response_end_lm_loss_value = (
                None if response_end_lm_loss is None else float(response_end_lm_loss.detach().cpu())
            )
            focused_block_hidden_loss_value = (
                None if focused_block_hidden_loss is None else float(focused_block_hidden_loss.detach().cpu())
            )
            focused_projected_vocab_loss_value = (
                None if focused_projected_vocab_loss is None else float(focused_projected_vocab_loss.detach().cpu())
            )
            focused_lm_loss_value = None if focused_lm_loss is None else float(focused_lm_loss.detach().cpu())
            module_output_loss_value = (
                None if module_output_loss is None else float(module_output_loss.detach().cpu())
            )
            projected_vocab_loss_value = (
                None if projected_vocab_loss is None else float(projected_vocab_loss.detach().cpu())
            )
            byte_kd_loss_value = None if byte_kd_loss is None else float(byte_kd_loss.detach().cpu())
            byte_hidden_loss_value = None if byte_hidden_loss is None else float(byte_hidden_loss.detach().cpu())
            byte_hidden_probe_anchor_loss_value = (
                None if byte_hidden_probe_anchor_loss is None else float(byte_hidden_probe_anchor_loss.detach().cpu())
            )
            alm_loss_value = None if alm_loss is None else float(alm_loss.detach().cpu())
            on_policy_loss_value = None if on_policy_loss is None else float(on_policy_loss.detach().cpu())
            on_policy_hidden_loss_value = (
                None if on_policy_hidden_loss is None else float(on_policy_hidden_loss.detach().cpu())
            )
            teacher_generate_lm_loss_value = (
                None if teacher_generate_lm_loss is None else float(teacher_generate_lm_loss.detach().cpu())
            )
            teacher_generate_hidden_loss_value = (
                None if teacher_generate_hidden_loss is None else float(teacher_generate_hidden_loss.detach().cpu())
            )
            teacher_generate_projected_vocab_loss_value = (
                None
                if teacher_generate_projected_vocab_loss is None
                else float(teacher_generate_projected_vocab_loss.detach().cpu())
            )
            teacher_generate_byte_loss_value = (
                None if teacher_generate_byte_loss is None else float(teacher_generate_byte_loss.detach().cpu())
            )
            stepwise_donor_loss_value = (
                None if stepwise_donor_loss is None else float(stepwise_donor_loss.detach().cpu())
            )
            if save_best_layer_loss and (best_loss is None or loss_value < best_loss):
                best_loss = loss_value
                best_step = int(step)
                best_state = _snapshot_named_module_states(snapshot_modules)
            if save_best_layer_behavior and (step == 1 or step % behavior_interval == 0 or step == steps_per_layer):
                with torch.no_grad():
                    candidate.eval()
                    behavior_summary = _heldout_behavior_summary(
                        torch,
                        candidate,
                        candidate_tokenizer,
                        device,
                        int(layer_gate_behavior_max_new_tokens),
                        int(layer_gate_max_texts),
                        layer_gate_behavior_probes,
                        layer_gate_generation_prompts,
                        layer_gate_max_generation_prompts,
                    )
                    candidate.train()
                behavior_entry = _behavior_history_entry(step, loss_value, behavior_summary)
                behavior_history.append(behavior_entry)
                _print_behavior_checkpoint(f"xfamily_block_behavior layer={candidate_idx}", behavior_entry)
                if _behavior_checkpoint_is_better(behavior_entry, best_behavior_summary, best_behavior_loss):
                    best_behavior_summary = dict(behavior_entry)
                    best_behavior_step = int(step)
                    best_behavior_loss = loss_value
                    best_behavior_state = _snapshot_named_module_states(
                        snapshot_modules
                    )
            if step == 1 or step % interval == 0 or step == steps_per_layer:
                losses.append(
                    {
                        "step": int(step),
                        "loss": loss_value,
                        "block_loss": block_loss_value,
                        "block_io_loss": block_io_loss_value,
                        "block_io_pairs": int(block_io_pair_count),
                        "lm_loss": lm_loss_value,
                        "response_lm_loss": response_lm_loss_value,
                        "first_token_lm_loss": first_token_lm_loss_value,
                        "response_end_lm_loss": response_end_lm_loss_value,
                        "focused_block_hidden_loss": focused_block_hidden_loss_value,
                        "focused_projected_vocab_kd_loss": focused_projected_vocab_loss_value,
                        "focused_lm_loss": focused_lm_loss_value,
                        "focused_span_pairs": int(focus_pair_count),
                        "projected_vocab_kd_loss": projected_vocab_loss_value,
                        "projected_vocab_pairs": int(projected_vocab_pair_count),
                        "byte_kd_loss": byte_kd_loss_value,
                        "byte_pairs": int(byte_kd_pair_count),
                        "byte_hidden_kd_loss": byte_hidden_loss_value,
                        "byte_hidden_pairs": int(byte_hidden_pair_count),
                        "byte_hidden_probe_anchor_loss": byte_hidden_probe_anchor_loss_value,
                        "alm_kd_loss": alm_loss_value,
                        "alm_chunks": int(alm_count),
                        "module_output_loss": module_output_loss_value,
                        "module_output": module_output_counts,
                        "on_policy_loss": on_policy_loss_value,
                        "on_policy_count": int(on_policy_count),
                        "on_policy_hidden_loss": on_policy_hidden_loss_value,
                        "on_policy_hidden_count": int(on_policy_hidden_count),
                        "teacher_generate_lm_loss": teacher_generate_lm_loss_value,
                        "teacher_generate_hidden_loss": teacher_generate_hidden_loss_value,
                        "teacher_generate_hidden_count": int(teacher_generate_hidden_count),
                        "teacher_generate_projected_vocab_kd_loss": teacher_generate_projected_vocab_loss_value,
                        "teacher_generate_projected_vocab_pairs": int(teacher_generate_projected_vocab_count),
                        "teacher_generate_byte_kd_loss": teacher_generate_byte_loss_value,
                        "teacher_generate_byte_pairs": int(teacher_generate_byte_count),
                        "stepwise_donor_kd_loss": stepwise_donor_loss_value,
                        "stepwise_donor_count": int(stepwise_donor_count),
                        "span_pairs": int(pair_count),
                    }
                )
                print(
                    f"xfamily_block_recover layer={candidate_idx} step={step} "
                    f"loss={loss_value:.6f} block_loss={block_loss_value:.6f} "
                    f"block_io={block_io_loss_value if block_io_loss_value is not None else 0.0:.6f} "
                    f"block_io_pairs={int(block_io_pair_count)} "
                    f"lm_loss={lm_loss_value if lm_loss_value is not None else 0.0:.6f} "
                    f"response_lm={response_lm_loss_value if response_lm_loss_value is not None else 0.0:.6f} "
                    f"first_response_lm={first_token_lm_loss_value if first_token_lm_loss_value is not None else 0.0:.6f} "
                    f"response_end_lm={response_end_lm_loss_value if response_end_lm_loss_value is not None else 0.0:.6f} "
                    f"focused_hidden={focused_block_hidden_loss_value if focused_block_hidden_loss_value is not None else 0.0:.6f} "
                    f"focused_projected={focused_projected_vocab_loss_value if focused_projected_vocab_loss_value is not None else 0.0:.6f} "
                    f"focused_lm={focused_lm_loss_value if focused_lm_loss_value is not None else 0.0:.6f} "
                    f"focused_pairs={int(focus_pair_count)} "
                    f"projected_vocab={projected_vocab_loss_value if projected_vocab_loss_value is not None else 0.0:.6f} "
                    f"projected_vocab_pairs={int(projected_vocab_pair_count)} "
                    f"byte_kd={byte_kd_loss_value if byte_kd_loss_value is not None else 0.0:.6f} "
                    f"byte_pairs={int(byte_kd_pair_count)} "
                    f"byte_hidden={byte_hidden_loss_value if byte_hidden_loss_value is not None else 0.0:.6f} "
                    f"byte_hidden_pairs={int(byte_hidden_pair_count)} "
                    f"byte_hidden_probe_anchor={byte_hidden_probe_anchor_loss_value if byte_hidden_probe_anchor_loss_value is not None else 0.0:.6f} "
                    f"alm={alm_loss_value if alm_loss_value is not None else 0.0:.6f} "
                    f"alm_chunks={int(alm_count)} "
                    f"module_output={module_output_loss_value if module_output_loss_value is not None else 0.0:.6f} "
                    f"module_attention_pairs={int(module_output_counts.get('attention_pairs', 0))} "
                    f"module_mlp_pairs={int(module_output_counts.get('mlp_pairs', 0))} "
                    f"on_policy={on_policy_loss_value if on_policy_loss_value is not None else 0.0:.6f} "
                    f"on_policy_count={int(on_policy_count)} "
                    f"on_policy_hidden={on_policy_hidden_loss_value if on_policy_hidden_loss_value is not None else 0.0:.6f} "
                    f"on_policy_hidden_count={int(on_policy_hidden_count)} "
                    f"teacher_generate_lm={teacher_generate_lm_loss_value if teacher_generate_lm_loss_value is not None else 0.0:.6f} "
                    f"teacher_generate_hidden={teacher_generate_hidden_loss_value if teacher_generate_hidden_loss_value is not None else 0.0:.6f} "
                    f"teacher_generate_hidden_count={int(teacher_generate_hidden_count)} "
                    f"teacher_generate_projected_vocab={teacher_generate_projected_vocab_loss_value if teacher_generate_projected_vocab_loss_value is not None else 0.0:.6f} "
                    f"teacher_generate_projected_vocab_pairs={int(teacher_generate_projected_vocab_count)} "
                    f"teacher_generate_byte={teacher_generate_byte_loss_value if teacher_generate_byte_loss_value is not None else 0.0:.6f} "
                    f"teacher_generate_byte_pairs={int(teacher_generate_byte_count)} "
                    f"stepwise_donor={stepwise_donor_loss_value if stepwise_donor_loss_value is not None else 0.0:.6f} "
                    f"stepwise_donor_count={int(stepwise_donor_count)} "
                    f"span_pairs={int(pair_count)}",
                    flush=True,
                )
        if save_best_layer_behavior and best_behavior_state is not None:
            _restore_named_module_states(snapshot_modules, best_behavior_state)
        elif save_best_layer_loss and best_state is not None:
            _restore_named_module_states(snapshot_modules, best_state)

        after_alignment = _cross_tokenizer_block_alignment_state(
            torch,
            source,
            candidate,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            source_group,
            candidate_idx,
            layer_hidden_projection,
            _canonical_token,
            donor_layer_fold_mode,
            block_target,
            span_max_pairs,
            source_group_weights,
        )
        after_metrics = _cross_tokenizer_block_alignment_metrics(
            torch,
            after_alignment,
            layer_hidden_projection,
            hidden_metric,
        )
        report: dict[str, Any] = {
            "candidate_layer": int(candidate_idx),
            "source_layer": int(source_idx),
            "source_layer_group": [int(item) for item in source_group],
            "source_layer_group_weights": (
                None if source_group_weights is None else [float(item) for item in source_group_weights]
            ),
            "trainable_params": int(trainable_params),
            "steps": int(steps_per_layer),
            "skipped_steps": int(skipped_steps),
            "nonfinite_step_policy": nonfinite_step_policy,
            "max_nonfinite_steps_per_layer": int(max_nonfinite_steps_per_layer),
            "nonfinite_steps": int(nonfinite_step_count),
            "nonfinite_loss_steps": int(nonfinite_loss_steps),
            "nonfinite_gradient_steps": int(nonfinite_gradient_steps),
            "nonfinite_parameter_steps": int(nonfinite_parameter_steps),
            "nonfinite_events": nonfinite_events,
            "losses": losses,
            "best_loss": best_loss,
            "best_step": best_step,
            "behavior_history": behavior_history,
            "best_behavior_step": best_behavior_step,
            "best_behavior": best_behavior_summary,
            "focused_span_pair_total": int(focused_span_pair_total),
            "focused_span_pair_active_steps": int(focused_span_pair_active_steps),
            "focused_span_pair_max": int(focused_span_pair_max),
            "hidden_projection": (
                None
                if layer_hidden_projection_report is None
                else {key: value for key, value in layer_hidden_projection_report.items() if key != "map"}
            ),
            "pre_reconstruction": pre_reconstruction_report,
            "residual_magnitude_compensation": compensation_report,
            "before": before_metrics,
            "after": after_metrics,
        }
        if layer_gate_metric == "ppl":
            with torch.no_grad():
                candidate.eval()
                candidate_ppl = _heldout_ppl_mean(torch, candidate, candidate_tokenizer, gate_texts, device, max_length)
                candidate.train()
            gate_decision = _layer_gate_decision(
                previous_gate_ppl if previous_gate_ppl is not None else candidate_ppl,
                candidate_ppl,
                layer_gate_max_ppl_ratio,
            )
            if gate_decision["accepted"]:
                previous_gate_ppl = candidate_ppl
            elif layer_state is not None:
                _restore_named_module_states(snapshot_modules, layer_state)
                with torch.no_grad():
                    candidate.eval()
                    reverted_ppl = _heldout_ppl_mean(
                        torch,
                        candidate,
                        candidate_tokenizer,
                        gate_texts,
                        device,
                        max_length,
                    )
                    candidate.train()
                gate_decision["reverted"] = True
                gate_decision["reverted_ppl"] = reverted_ppl
                previous_gate_ppl = reverted_ppl
            report["layer_gate"] = gate_decision
        elif layer_gate_metric == "behavior":
            with torch.no_grad():
                candidate.eval()
                candidate_behavior = _heldout_behavior_summary(
                    torch,
                    candidate,
                    candidate_tokenizer,
                    device,
                    int(layer_gate_behavior_max_new_tokens),
                    int(layer_gate_max_texts),
                    layer_gate_behavior_probes,
                    layer_gate_generation_prompts,
                    layer_gate_max_generation_prompts,
                )
                candidate.train()
            gate_decision = _layer_gate_behavior_decision(
                previous_gate_behavior or candidate_behavior,
                candidate_behavior,
                layer_gate_max_behavior_drop,
            )
            gate_decision["candidate_behavior"] = candidate_behavior
            if gate_decision["accepted"]:
                previous_gate_behavior = candidate_behavior
            elif layer_state is not None:
                _restore_named_module_states(snapshot_modules, layer_state)
                with torch.no_grad():
                    candidate.eval()
                    reverted_behavior = _heldout_behavior_summary(
                        torch,
                        candidate,
                        candidate_tokenizer,
                        device,
                        int(layer_gate_behavior_max_new_tokens),
                        int(layer_gate_max_texts),
                        layer_gate_behavior_probes,
                        layer_gate_generation_prompts,
                        layer_gate_max_generation_prompts,
                    )
                    candidate.train()
                gate_decision["reverted"] = True
                gate_decision["reverted_behavior"] = reverted_behavior
                previous_gate_behavior = reverted_behavior
            report["layer_gate"] = gate_decision
        layer_reports.append(report)
        completed_layer_count = int(candidate_idx) + 1
        if layer_checkpoint_interval > 0 and (
            completed_layer_count % int(layer_checkpoint_interval) == 0
            or completed_layer_count == int(layer_count)
        ):
            checkpoint_dir = target / "layer_checkpoints" / f"layer_{completed_layer_count:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            candidate.save_pretrained(checkpoint_dir, safe_serialization=True)
            candidate_tokenizer.save_pretrained(checkpoint_dir)
            progress_report = {
                "operation": "blockwise_reconstruct_cross_tokenizer_causal_lm_progress",
                "source_model": source_model,
                "candidate_model": candidate_model,
                "output_dir": str(target),
                "checkpoint_dir": str(checkpoint_dir),
                "layers_completed": completed_layer_count,
                "layer_count": int(layer_count),
                "layer_checkpoint_interval": int(layer_checkpoint_interval),
                "latest_layer": report,
            }
            _write_json(checkpoint_dir / "neural_forge_xfamily_block_recovery_progress.json", progress_report)
            _write_json(target / "neural_forge_xfamily_block_recovery_progress.json", progress_report)
        if device.startswith("cuda") and hasattr(torch, "cuda"):
            torch.cuda.empty_cache()

    for param in candidate.parameters():
        param.requires_grad_(False)
    target.mkdir(parents=True, exist_ok=True)
    candidate.save_pretrained(target, safe_serialization=True)
    candidate_tokenizer.save_pretrained(target)
    focused_span_pair_total = sum(int(item.get("focused_span_pair_total", 0) or 0) for item in layer_reports)
    focused_span_pair_active_steps = sum(
        int(item.get("focused_span_pair_active_steps", 0) or 0) for item in layer_reports
    )
    focused_span_pair_attempted_steps = (
        int(steps_per_layer) * int(len(layer_reports))
        if focused_block_hidden_kd_weight > 0 or focused_projected_vocab_kd_weight > 0 or focused_lm_weight > 0
        else 0
    )
    focused_span_pair_max = max(
        (int(item.get("focused_span_pair_max", 0) or 0) for item in layer_reports),
        default=0,
    )
    report = {
        "operation": "blockwise_reconstruct_cross_tokenizer_causal_lm",
        "source_model": source_model,
        "candidate_model": candidate_model,
        "output_dir": str(target),
        "steps_per_layer": int(steps_per_layer),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "lr": float(lr),
        "lm_weight": float(lm_weight),
        "response_lm_weight": float(response_lm_weight),
        "first_token_lm_weight": float(first_token_lm_weight),
        "response_end_lm_weight": float(response_end_lm_weight),
        "response_loss_mode": response_loss_mode,
        "hidden_ridge": float(hidden_ridge),
        "hidden_projection_scope": hidden_projection_scope,
        "hidden_metric": hidden_metric,
        "span_pair_mode": span_pair_mode,
        "block_target": block_target,
        "residual_magnitude_compensation": residual_magnitude_compensation,
        "residual_compensation_min_scale": float(residual_compensation_min_scale),
        "residual_compensation_max_scale": float(residual_compensation_max_scale),
        "residual_compensation_linear_ridge": float(residual_compensation_linear_ridge),
        "residual_compensation_alpha": float(residual_compensation_alpha),
        "residual_compensation_linear_rank": int(residual_compensation_linear_rank),
        "pre_reconstruction": {
            "target": pre_reconstruction_target,
            "ridge": float(pre_reconstruction_ridge),
            "alpha": float(pre_reconstruction_alpha),
            "objective": pre_reconstruction_objective,
            "cosine_steps": int(pre_reconstruction_cosine_steps),
            "cosine_lr": float(pre_reconstruction_cosine_lr),
            "cosine_l2": float(pre_reconstruction_cosine_l2),
            "fit_gate": pre_reconstruction_fit_gate,
            "validation_fraction": float(pre_reconstruction_validation_fraction),
            "min_validation_rows": int(pre_reconstruction_min_validation_rows),
            "max_tokens_per_layer": int(pre_reconstruction_max_tokens_per_layer),
            "min_relative_improvement": float(pre_reconstruction_min_relative_improvement),
            "max_cosine_drop": float(pre_reconstruction_max_cosine_drop),
            "min_fitted_cosine": float(pre_reconstruction_min_fitted_cosine),
        },
        "max_anchors": int(max_anchors),
        "span_max_pairs": int(span_max_pairs),
        "donor_layer_fold_mode": donor_layer_fold_mode,
        "donor_layer_group_strategy": donor_layer_group_strategy,
        "donor_layer_groups": donor_layer_group_report,
        "layer_limit": layer_limit,
        "layer_gate": layer_gate,
        "on_policy_anchor_kd_weight": float(on_policy_anchor_kd_weight),
        "on_policy_unlikelihood_weight": float(on_policy_unlikelihood_weight),
        "on_policy_hidden_kd_weight": float(on_policy_hidden_kd_weight),
        "on_policy_max_new_tokens": int(on_policy_max_new_tokens),
        "on_policy_interval": int(on_policy_interval),
        "on_policy_response_loss_mode": on_policy_response_loss_mode,
        "on_policy_unlikelihood_window": int(on_policy_unlikelihood_window),
        "stepwise_donor_kd_weight": float(stepwise_donor_kd_weight),
        "stepwise_donor_max_new_tokens": int(stepwise_donor_max_new_tokens),
        "stepwise_donor_interval": int(stepwise_donor_interval),
        "teacher_generate_lm_weight": float(teacher_generate_lm_weight),
        "teacher_generate_hidden_kd_weight": float(teacher_generate_hidden_kd_weight),
        "teacher_generate_projected_vocab_kd_weight": float(teacher_generate_projected_vocab_kd_weight),
        "teacher_generate_byte_kd_weight": float(teacher_generate_byte_kd_weight),
        "teacher_generate_max_new_tokens": int(teacher_generate_max_new_tokens),
        "teacher_generate_loss_mode": teacher_generate_loss_mode,
        "teacher_generated_text_count": len(teacher_generated_texts),
        "teacher_generated_text_samples": teacher_generated_texts[:3],
        "calibration_order": calibration_order,
        "calibration_domain_counts": calibration_domain_counts,
        "alm_kd_weight": float(alm_kd_weight),
        "alm_max_chunks": int(alm_max_chunks),
        "alm_min_chars": int(alm_min_chars),
        "projected_vocab_kd_weight": float(projected_vocab_kd_weight),
        "focused_block_hidden_kd_weight": float(focused_block_hidden_kd_weight),
        "focused_projected_vocab_kd_weight": float(focused_projected_vocab_kd_weight),
        "focused_lm_weight": float(focused_lm_weight),
        "token_focus": token_focus,
        "projected_vocab_kd_mode": projected_vocab_kd_mode,
        "projected_vocab_group": projected_vocab_group_report,
        "byte_kd_weight": float(byte_kd_weight),
        "byte_hidden_kd_weight": float(byte_hidden_kd_weight),
        "byte_hidden_probe_mode": byte_hidden_probe_mode,
        "byte_hidden_probe_anchor_weight": float(byte_hidden_probe_anchor_weight),
        "block_io_kd_weight": float(block_io_kd_weight),
        "block_io_ridge": float(block_io_ridge),
        "block_io_loss": block_io_loss,
        "train_output_head_with_kd": bool(train_output_head_with_kd),
        "untie_output_head_for_kd": bool(untie_output_head_for_kd),
        "output_head_untied_for_kd": bool(output_head_untied_for_kd),
        "allow_tied_output_head_training": bool(allow_tied_output_head_training),
        "output_head_tied_to_input": bool(output_head_tied_to_input),
        "output_head_training_warning": output_head_training_warning,
        "module_output_kd_weight": float(module_output_kd_weight),
        "module_output_targets": module_output_targets,
        "module_output_transport": module_output_transport,
        "module_output_temperature": float(module_output_temperature),
        "module_output_sinkhorn_iterations": int(module_output_sinkhorn_iterations),
        "projected_vocab_max_source_tokens": int(projected_vocab_max_source_tokens),
        "temperature": float(temperature),
        "torch_dtype": torch_dtype,
        "candidate_torch_dtype": candidate_torch_dtype,
        "grad_clip": float(grad_clip),
        "nonfinite_step_policy": nonfinite_step_policy,
        "max_nonfinite_steps_per_layer": int(max_nonfinite_steps_per_layer),
        "nonfinite_steps": sum(int(item.get("nonfinite_steps", 0) or 0) for item in layer_reports),
        "nonfinite_loss_steps": sum(int(item.get("nonfinite_loss_steps", 0) or 0) for item in layer_reports),
        "nonfinite_gradient_steps": sum(
            int(item.get("nonfinite_gradient_steps", 0) or 0) for item in layer_reports
        ),
        "nonfinite_parameter_steps": sum(
            int(item.get("nonfinite_parameter_steps", 0) or 0) for item in layer_reports
        ),
        "report_interval": int(interval),
        "save_best_layer_loss": bool(save_best_layer_loss),
        "save_best_layer_behavior": bool(save_best_layer_behavior),
        "layer_behavior_eval_interval": int(behavior_interval),
        "layer_checkpoint_interval": int(layer_checkpoint_interval),
        "calibration_text_count": len(texts),
        "eval_overlap_count": eval_overlap_count,
        "hidden_projection": {key: value for key, value in hidden_projection_report.items() if key != "map"},
        "layers": layer_reports,
        "focused_span_pair_total": int(focused_span_pair_total),
        "focused_span_pair_active_steps": int(focused_span_pair_active_steps),
        "focused_span_pair_attempted_steps": int(focused_span_pair_attempted_steps),
        "focused_span_pair_active_fraction": (
            float(focused_span_pair_active_steps / focused_span_pair_attempted_steps)
            if focused_span_pair_attempted_steps > 0
            else None
        ),
        "focused_span_pair_max": int(focused_span_pair_max),
        "param_count": sum(param.numel() for param in candidate.parameters()),
    }
    _write_json(target / "neural_forge_xfamily_block_recovery_report.json", report)
    return report


def load_calibration_texts(
    calibration_file: str | Path,
    max_texts: int | None = None,
    min_chars: int = 32,
    max_chars: int = 4096,
    preserve_whitespace: bool = True,
) -> list[str]:
    """Load recovery calibration texts from plain text or JSONL rows."""

    if max_texts is not None and max_texts <= 0:
        raise ValueError("max_texts must be positive when provided.")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative.")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    path = Path(calibration_file)
    if not path.exists():
        raise FileNotFoundError(path)

    texts = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = _calibration_text_from_line(line)
            if text is None:
                continue
            text = _clean_calibration_text(text, preserve_whitespace=preserve_whitespace)
            if len(text) < min_chars:
                continue
            texts.append(text[:max_chars])
            if max_texts is not None and len(texts) >= max_texts:
                break
    if not texts:
        raise ValueError(f"No usable calibration texts found in {path}.")
    return texts


def _clean_calibration_text(text: str, preserve_whitespace: bool = True) -> str:
    if preserve_whitespace:
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return " ".join(text.split())


def _calibration_text_from_line(line: str) -> str | None:
    raw = line.strip()
    if not raw:
        return None
    if not raw.startswith("{"):
        return raw
    try:
        item = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return _calibration_text_from_obj(item)


def _calibration_text_from_obj(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return None
    messages = item.get("messages")
    if isinstance(messages, list):
        return _messages_to_text(messages)
    input_json = item.get("input_json")
    if isinstance(input_json, dict) and isinstance(input_json.get("messages"), list):
        pieces = [_messages_to_text(input_json["messages"])]
        target = item.get("target")
        if isinstance(target, str) and target.strip():
            pieces.append(target)
        target_json = item.get("target_json")
        if isinstance(target_json, dict):
            content = target_json.get("content")
            if isinstance(content, str) and content.strip():
                pieces.append(content)
            else:
                pieces.append(json.dumps(target_json, sort_keys=True))
        return "\n".join(piece for piece in pieces if piece)
    prompt = item.get("prompt")
    target = item.get("target")
    if isinstance(prompt, str) and isinstance(target, str):
        return _join_prompt_response(prompt, target)
    response = item.get("response")
    if isinstance(prompt, str) and isinstance(response, str):
        return _join_prompt_response(prompt, response)
    for key in ("text", "content", "completion", "response"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _messages_to_text(messages: Sequence[Any]) -> str:
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = message.get("role")
        if isinstance(role, str) and role.strip():
            parts.append(f"{role.strip()}: {content.strip()}")
        else:
            parts.append(content.strip())
    return "\n".join(parts)


def _join_prompt_response(prompt: str, response: str) -> str:
    if prompt.endswith((" ", "\n", "\t")):
        return f"{prompt}{response}"
    if prompt.rstrip().endswith(":"):
        return f"{prompt} {response}"
    return f"{prompt}\n{response}"


def _assert_same_vocab(left, right) -> None:
    if len(left) != len(right):
        raise ValueError(f"Tokenizer length mismatch: {len(left)} != {len(right)}")
    if hasattr(left, "get_vocab") and hasattr(right, "get_vocab") and left.get_vocab() != right.get_vocab():
        raise ValueError("Recovery KD requires matching tokenizers/vocabularies.")


def _set_train_scope(model, train_scope: str) -> int:
    scope = train_scope.lower()
    for param in model.parameters():
        param.requires_grad_(False)
    if scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
    elif scope in {"head", "lm_head", "output", "output_head"}:
        output = model.get_output_embeddings()
        if output is None:
            raise ValueError("train_scope head requires an output embedding/lm_head module.")
        for param in output.parameters():
            param.requires_grad_(True)
    elif scope in {"head_norm", "norm_head", "final_norm_head"}:
        output = model.get_output_embeddings()
        if output is None:
            raise ValueError("train_scope head_norm requires an output embedding/lm_head module.")
        for param in output.parameters():
            param.requires_grad_(True)
        for module in _final_norm_modules(model):
            for param in module.parameters():
                param.requires_grad_(True)
    elif scope == "ffn":
        for layer in _recovery_decoder_layers(model):
            for module_name in ("mlp", "feed_forward"):
                module = getattr(layer, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(True)
    elif scope.startswith("last"):
        count_text = scope.replace("last", "").strip("-_") or "1"
        count = int(count_text)
        layers = _recovery_decoder_layers(model)
        if count <= 0 or not layers:
            raise ValueError("last-N train scope requires N > 0 and decoder layers.")
        for layer in layers[-count:]:
            for param in layer.parameters():
                param.requires_grad_(True)
        output = model.get_output_embeddings()
        if output is not None:
            for param in output.parameters():
                param.requires_grad_(True)
    else:
        raise ValueError("train_scope must be all, head, head_norm, ffn, or lastN, e.g. last1.")
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable == 0:
        raise ValueError(f"train_scope {train_scope!r} selected no trainable parameters.")
    return trainable


def _final_norm_modules(model) -> list[Any]:
    modules = []
    for name in ("norm", "ln_f", "final_layer_norm", "final_norm", "norm_f"):
        module = getattr(model, name, None)
        if module is not None:
            modules.append(module)
    for root_name in ("model", "transformer", "gpt_neox", "base_model"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        for name in ("norm", "ln_f", "final_layer_norm", "final_norm", "norm_f"):
            module = getattr(root, name, None)
            if module is not None and module not in modules:
                modules.append(module)
    return modules


def _recovery_decoder_layers(model) -> list[Any]:
    try:
        return list(_decoder_layer_container(model)[2])
    except ValueError:
        return []


def _make_batches(
    tokenizer,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
    include_offsets: bool = False,
    response_loss_mode: str = "none",
    append_eos: bool = False,
) -> list[dict[str, Any]]:
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    batches = []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start : start + batch_size])
        if append_eos:
            chunk = [_append_eos_text(tokenizer, text) for text in chunk]
        kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": max_length,
        }
        if include_offsets:
            kwargs["return_offsets_mapping"] = True
        try:
            encoded = tokenizer(chunk, **kwargs)
        except NotImplementedError as exc:
            raise ValueError("Span KD requires fast tokenizers that support offset mappings.") from exc
        batch = {key: value.to(device) for key, value in encoded.items()}
        batch["texts"] = chunk
        batch["raw_texts"] = list(texts[start : start + batch_size])
        if response_loss_mode != "none":
            batch["loss_mask"] = _response_loss_mask(
                tokenizer,
                chunk,
                batch["input_ids"],
                batch.get("attention_mask"),
                max_length,
                response_loss_mode,
                device,
            )
        batches.append(batch)
    return batches


def _order_recovery_texts(texts: Sequence[str], calibration_order: str) -> list[str]:
    ordered = list(texts)
    if calibration_order == "original":
        return ordered
    if calibration_order != "domain-balanced":
        raise ValueError("calibration_order must be original or domain-balanced.")
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for text in ordered:
        domain = _recovery_text_domain(text)
        if domain not in groups:
            groups[domain] = []
            order.append(domain)
        groups[domain].append(text)
    balanced: list[str] = []
    max_len = max((len(group) for group in groups.values()), default=0)
    for idx in range(max_len):
        for domain in order:
            group = groups[domain]
            if idx < len(group):
                balanced.append(group[idx])
    return balanced


def _recovery_domain_counts(texts: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        domain = _recovery_text_domain(text)
        counts[domain] = counts.get(domain, 0) + 1
    return counts


def _recovery_text_domain(text: str) -> str:
    try:
        from neural_forge.hf_xfamily import _activation_calibration_domain

        return str(_activation_calibration_domain(text))
    except Exception:
        lowered = str(text).lower()
        if any(marker in lowered for marker in ("def ", "python", "function", "return ", "```", "code")):
            return "code"
        if any(marker in lowered for marker in ("tokenizer", "tokenization", "vocab", "merge", "pruning", "neuron")):
            return "model_surgery"
        if any(ch.isdigit() for ch in lowered) or any(marker in lowered for marker in ("math", "calculate", "sum")):
            return "math"
        if any(marker in lowered for marker in ("safe", "unsafe", "risk", "policy")):
            return "safety"
        return "general"


def _append_eos_text(tokenizer, text: str) -> str:
    eos = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos, str) or not eos:
        return text
    return text if text.endswith(eos) else f"{text}{eos}"


def _make_paired_batches(
    source_tokenizer,
    candidate_tokenizer,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
    include_offsets: bool = False,
    response_loss_mode: str = "none",
    append_eos: bool = False,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    source_batches = _make_batches(
        source_tokenizer,
        texts,
        batch_size,
        max_length,
        device,
        include_offsets,
        append_eos=append_eos,
    )
    candidate_batches = _make_batches(
        candidate_tokenizer,
        texts,
        batch_size,
        max_length,
        device,
        include_offsets,
        response_loss_mode=response_loss_mode,
        append_eos=append_eos,
    )
    if len(source_batches) != len(candidate_batches):
        raise RuntimeError("Paired source/candidate batch counts diverged.")
    return list(zip(source_batches, candidate_batches))


def _model_forward_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in batch.items()
        if key not in {"offset_mapping", "loss_mask", "texts", "raw_texts"}
    }


def _response_loss_mask(
    tokenizer,
    texts: Sequence[str],
    input_ids,
    attention_mask,
    max_length: int,
    response_loss_mode: str,
    device: str,
):
    torch, _transformers = _torch_and_transformers()
    mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    if attention_mask is None:
        attention = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    else:
        attention = attention_mask.bool()
    for row, text in enumerate(texts):
        response_start = _infer_response_char_start(text, response_loss_mode)
        if response_start is None:
            mask[row] = attention[row]
            continue
        prefix = text[: int(response_start)]
        prefix_len = _tokenized_length(tokenizer, prefix, max_length)
        seq_len = int(attention[row].sum().detach().cpu().item())
        if seq_len <= 1 or prefix_len >= seq_len:
            mask[row] = attention[row]
            continue
        start_idx = max(0, int(prefix_len))
        mask[row, start_idx:seq_len] = True
    return mask & attention


def _tokenized_length(tokenizer, text: str, max_length: int) -> int:
    if not text:
        return 0
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return int(encoded["input_ids"].shape[-1])


def _infer_response_char_start(text: str, response_loss_mode: str) -> int | None:
    if response_loss_mode == "none":
        return None
    lower = text.lower()
    if response_loss_mode in {"auto", "assistant"}:
        idx = lower.rfind("assistant:")
        if idx >= 0:
            return idx + len("assistant:")
    if response_loss_mode in {"auto", "answer"}:
        idx = lower.rfind("answer:")
        if idx >= 0:
            return idx + len("answer:")
    if response_loss_mode == "auto":
        for marker in ("\n```python", "\n```"):
            idx = lower.rfind(marker)
            if idx >= 0:
                return idx + 1
        newline_response_start = _infer_last_line_response_start(text)
        if newline_response_start is not None:
            return newline_response_start
    return None


def _infer_last_line_response_start(text: str) -> int | None:
    stripped = text.rstrip()
    idx = stripped.rfind("\n")
    if idx < 0:
        return None
    tail_start = idx + 1
    tail = stripped[tail_start:].strip()
    if not tail:
        return None
    if len(tail) > 240:
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", tail):
        return tail_start + len(stripped[tail_start:]) - len(stripped[tail_start:].lstrip())
    if re.fullmatch(r"[A-D](?:[.)])?", tail, flags=re.IGNORECASE):
        return tail_start + len(stripped[tail_start:]) - len(stripped[tail_start:].lstrip())
    if re.search(r"\b(def|return|class|lambda)\b|[{}();=+\-*/]", tail):
        return tail_start + len(stripped[tail_start:]) - len(stripped[tail_start:].lstrip())
    promptish = text[:idx].lower()
    if any(marker in promptish for marker in ("question:", "solve:", "prompt:", "user:")) and len(tail.split()) <= 24:
        return tail_start + len(stripped[tail_start:]) - len(stripped[tail_start:].lstrip())
    return None


def _cross_tokenizer_span_pairs(
    torch,
    source_tokenizer,
    candidate_tokenizer,
    source_batch: dict[str, Any],
    candidate_batch: dict[str, Any],
    canonical_fn,
    max_pairs: int,
    shift_positions: bool = True,
):
    if "offset_mapping" not in source_batch or "offset_mapping" not in candidate_batch:
        raise ValueError("Span KD batches require offset_mapping.")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive.")

    source_ids = source_batch["input_ids"].detach().cpu()
    candidate_ids = candidate_batch["input_ids"].detach().cpu()
    source_offsets = source_batch["offset_mapping"].detach().cpu()
    candidate_offsets = candidate_batch["offset_mapping"].detach().cpu()
    source_mask = source_batch.get("attention_mask")
    candidate_mask = candidate_batch.get("attention_mask")
    if source_mask is not None:
        source_mask = source_mask.detach().cpu().bool()
    if candidate_mask is not None:
        candidate_mask = candidate_mask.detach().cpu().bool()

    pairs = []
    batch_count = min(int(source_ids.shape[0]), int(candidate_ids.shape[0]))
    for batch_idx in range(batch_count):
        source_lookup = {}
        for source_pos in range(1, int(source_ids.shape[1])):
            if source_mask is not None and not bool(source_mask[batch_idx, source_pos]):
                continue
            start = int(source_offsets[batch_idx, source_pos, 0])
            end = int(source_offsets[batch_idx, source_pos, 1])
            if end <= start:
                continue
            token = _decoded_token(source_tokenizer, int(source_ids[batch_idx, source_pos].item()), canonical_fn)
            if not token:
                continue
            source_lookup.setdefault((start, end, token), source_pos - 1 if shift_positions else source_pos)
        for candidate_pos in range(1, int(candidate_ids.shape[1])):
            if candidate_mask is not None and not bool(candidate_mask[batch_idx, candidate_pos]):
                continue
            start = int(candidate_offsets[batch_idx, candidate_pos, 0])
            end = int(candidate_offsets[batch_idx, candidate_pos, 1])
            if end <= start:
                continue
            token = _decoded_token(candidate_tokenizer, int(candidate_ids[batch_idx, candidate_pos].item()), canonical_fn)
            if not token:
                continue
            source_shift_pos = source_lookup.get((start, end, token))
            if source_shift_pos is None:
                continue
            candidate_pair_pos = candidate_pos - 1 if shift_positions else candidate_pos
            pairs.append((batch_idx, candidate_pair_pos, source_shift_pos))
            if len(pairs) >= max_pairs:
                return torch.tensor(pairs, dtype=torch.long, device=candidate_batch["input_ids"].device)
    if not pairs:
        return torch.empty((0, 3), dtype=torch.long, device=candidate_batch["input_ids"].device)
    return torch.tensor(pairs, dtype=torch.long, device=candidate_batch["input_ids"].device)


def _cross_tokenizer_overlap_span_pairs(
    torch,
    source_batch: dict[str, Any],
    candidate_batch: dict[str, Any],
    max_pairs: int,
    shift_positions: bool = True,
):
    if "offset_mapping" not in source_batch or "offset_mapping" not in candidate_batch:
        raise ValueError("Overlap span pairing requires offset_mapping batches.")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive.")

    source_offsets = source_batch["offset_mapping"].detach().cpu()
    candidate_offsets = candidate_batch["offset_mapping"].detach().cpu()
    source_mask = source_batch.get("attention_mask")
    candidate_mask = candidate_batch.get("attention_mask")
    if source_mask is not None:
        source_mask = source_mask.detach().cpu().bool()
    if candidate_mask is not None:
        candidate_mask = candidate_mask.detach().cpu().bool()

    pairs = []
    batch_count = min(int(source_offsets.shape[0]), int(candidate_offsets.shape[0]))
    for batch_idx in range(batch_count):
        source_spans: list[tuple[int, int, int]] = []
        for source_pos in range(1, int(source_offsets.shape[1])):
            if source_mask is not None and not bool(source_mask[batch_idx, source_pos]):
                continue
            start = int(source_offsets[batch_idx, source_pos, 0])
            end = int(source_offsets[batch_idx, source_pos, 1])
            if end <= start:
                continue
            source_spans.append((start, end, source_pos))
        if not source_spans:
            continue
        for candidate_pos in range(1, int(candidate_offsets.shape[1])):
            if candidate_mask is not None and not bool(candidate_mask[batch_idx, candidate_pos]):
                continue
            candidate_start = int(candidate_offsets[batch_idx, candidate_pos, 0])
            candidate_end = int(candidate_offsets[batch_idx, candidate_pos, 1])
            if candidate_end <= candidate_start:
                continue
            best_source_pos = None
            best_overlap = 0
            best_distance = None
            for source_start, source_end, source_pos in source_spans:
                overlap = min(candidate_end, source_end) - max(candidate_start, source_start)
                if overlap <= 0:
                    continue
                distance = abs(candidate_start - source_start) + abs(candidate_end - source_end)
                if (
                    overlap > best_overlap
                    or (overlap == best_overlap and (best_distance is None or distance < best_distance))
                ):
                    best_source_pos = int(source_pos)
                    best_overlap = int(overlap)
                    best_distance = int(distance)
            if best_source_pos is None:
                continue
            candidate_pair_pos = candidate_pos - 1 if shift_positions else candidate_pos
            source_pair_pos = best_source_pos - 1 if shift_positions else best_source_pos
            pairs.append((batch_idx, candidate_pair_pos, source_pair_pos))
            if len(pairs) >= max_pairs:
                return torch.tensor(pairs, dtype=torch.long, device=candidate_batch["input_ids"].device)
    if not pairs:
        return torch.empty((0, 3), dtype=torch.long, device=candidate_batch["input_ids"].device)
    return torch.tensor(pairs, dtype=torch.long, device=candidate_batch["input_ids"].device)


def _cross_tokenizer_span_pairs_for_mode(
    torch,
    source_tokenizer,
    candidate_tokenizer,
    source_batch: dict[str, Any],
    candidate_batch: dict[str, Any],
    canonical_fn,
    max_pairs: int,
    shift_positions: bool = True,
    mode: str = "exact",
):
    if mode not in SPAN_PAIR_MODES:
        raise ValueError("span pair mode must be exact, overlap, or both.")
    if mode == "exact":
        return _cross_tokenizer_span_pairs(
            torch,
            source_tokenizer,
            candidate_tokenizer,
            source_batch,
            candidate_batch,
            canonical_fn,
            max_pairs=max_pairs,
            shift_positions=shift_positions,
        )
    if mode == "overlap":
        return _cross_tokenizer_overlap_span_pairs(
            torch,
            source_batch,
            candidate_batch,
            max_pairs=max_pairs,
            shift_positions=shift_positions,
        )

    exact = _cross_tokenizer_span_pairs(
        torch,
        source_tokenizer,
        candidate_tokenizer,
        source_batch,
        candidate_batch,
        canonical_fn,
        max_pairs=max_pairs,
        shift_positions=shift_positions,
    )
    remaining = int(max_pairs) - int(exact.shape[0])
    if remaining <= 0:
        return exact[: int(max_pairs)]
    overlap = _cross_tokenizer_overlap_span_pairs(
        torch,
        source_batch,
        candidate_batch,
        max_pairs=max_pairs,
        shift_positions=shift_positions,
    )
    pairs: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for tensor in (exact, overlap):
        for row in tensor.detach().cpu().tolist():
            item = (int(row[0]), int(row[1]), int(row[2]))
            if item in seen:
                continue
            seen.add(item)
            pairs.append(item)
            if len(pairs) >= int(max_pairs):
                return torch.tensor(pairs, dtype=torch.long, device=candidate_batch["input_ids"].device)
    if not pairs:
        return torch.empty((0, 3), dtype=torch.long, device=candidate_batch["input_ids"].device)
    return torch.tensor(pairs, dtype=torch.long, device=candidate_batch["input_ids"].device)


def _focus_char_spans(text: str, response_loss_mode: str, token_focus: str) -> list[tuple[int, int]]:
    if token_focus not in {"auto", "response", "numbers", "code", "all"}:
        raise ValueError("token_focus must be auto, response, numbers, code, or all.")
    if not text:
        return []

    scope_start = _infer_response_char_start(text, response_loss_mode)
    if scope_start is None:
        scope_start = 0
    scope_start = max(0, min(int(scope_start), len(text)))
    while scope_start < len(text) and text[scope_start].isspace():
        scope_start += 1
    scope = text[scope_start:]
    if token_focus == "all":
        return [(0, len(text))]
    if token_focus == "response":
        return [(scope_start, len(text))] if scope.strip() else []

    spans: list[tuple[int, int]] = []
    if token_focus in {"auto", "numbers"}:
        for match in re.finditer(
            r"(?<![\w.])-?\d+(?:\.\d+)?(?:\s*[-+*/=]\s*-?\d+(?:\.\d+)?)+(?!(?:[\w.]))",
            scope,
        ):
            spans.append((scope_start + match.start(), scope_start + match.end()))
        for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", scope):
            spans.append((scope_start + match.start(), scope_start + match.end()))
    if token_focus in {"auto", "code"}:
        code_scope = _code_focus_scope(scope)
        if code_scope is not None:
            spans.append((scope_start + code_scope[0], scope_start + code_scope[1]))
        elif token_focus == "code":
            for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|[-+*/%=(),.:]", scope):
                spans.append((scope_start + match.start(), scope_start + match.end()))
    if token_focus == "auto" and not spans and scope.strip():
        spans.append((scope_start, len(text)))
    return _merge_char_spans(spans)


def _code_focus_scope(text: str) -> tuple[int, int] | None:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence is not None:
        return (int(fence.start(1)), int(fence.end(1)))
    code_like = re.search(
        r"(?m)(def\s+\w+\s*\(|return\s+|class\s+\w+|for\s+\w+\s+in\s+|if\s+.+:|[A-Za-z_][A-Za-z0-9_]*\s*=\s*)",
        text,
    )
    if code_like is None:
        return None
    start = int(code_like.start())
    tail = text[start:]
    paragraph_end = tail.find("\n\n")
    end = len(text) if paragraph_end < 0 else start + paragraph_end
    return (start, end)


def _merge_char_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = sorted((max(0, int(start)), max(0, int(end))) for start, end in spans if int(end) > int(start))
    merged: list[tuple[int, int]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            old_start, old_end = merged[-1]
            merged[-1] = (old_start, max(old_end, end))
    return merged


def _cross_tokenizer_focus_span_pairs(
    torch,
    source_batch: dict[str, Any],
    candidate_batch: dict[str, Any],
    response_loss_mode: str,
    token_focus: str,
    max_pairs: int,
):
    if "offset_mapping" not in source_batch or "offset_mapping" not in candidate_batch:
        raise ValueError("Focused token KD batches require offset_mapping.")
    if "texts" not in candidate_batch:
        raise ValueError("Focused token KD batches require source texts.")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive.")

    source_offsets = source_batch["offset_mapping"].detach().cpu()
    candidate_offsets = candidate_batch["offset_mapping"].detach().cpu()
    source_mask = source_batch.get("attention_mask")
    candidate_mask = candidate_batch.get("attention_mask")
    if source_mask is not None:
        source_mask = source_mask.detach().cpu().bool()
    if candidate_mask is not None:
        candidate_mask = candidate_mask.detach().cpu().bool()

    focus_items = []
    used_pairs = 0
    texts = candidate_batch["texts"]
    batch_count = min(len(texts), int(source_offsets.shape[0]), int(candidate_offsets.shape[0]))
    device = candidate_batch["input_ids"].device
    for batch_idx in range(batch_count):
        spans = _focus_char_spans(str(texts[batch_idx]), response_loss_mode, token_focus)
        if not spans:
            continue
        source_positions = _shift_positions_for_char_spans(
            torch,
            source_offsets[batch_idx],
            None if source_mask is None else source_mask[batch_idx],
            spans,
            device,
        )
        candidate_positions = _shift_positions_for_char_spans(
            torch,
            candidate_offsets[batch_idx],
            None if candidate_mask is None else candidate_mask[batch_idx],
            spans,
            device,
        )
        if int(source_positions.numel()) <= 0 or int(candidate_positions.numel()) <= 0:
            continue
        remaining = int(max_pairs) - used_pairs
        if remaining <= 0:
            break
        if int(candidate_positions.numel()) > remaining:
            candidate_positions = candidate_positions[:remaining]
        focus_items.append(
            {
                "batch_idx": int(batch_idx),
                "source_positions": source_positions,
                "candidate_positions": candidate_positions,
            }
        )
        used_pairs += int(candidate_positions.numel())
        if used_pairs >= int(max_pairs):
            break
    return focus_items


def _answer_contrastive_loss(
    torch,
    model,
    tokenizer,
    texts: Sequence[str],
    response_loss_mode: str,
    max_negatives: int,
    max_length: int,
    device: str,
):
    groups = [
        candidates
        for text in texts
        if (candidates := _answer_contrastive_candidates(str(text), response_loss_mode, max_negatives))
    ]
    if not groups:
        param = next(model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0

    flat = [candidate for group in groups for candidate in group]
    encoded = tokenizer(
        flat,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    batch = {key: value.to(device) for key, value in encoded.items()}
    batch["loss_mask"] = _response_loss_mask(
        tokenizer,
        flat,
        batch["input_ids"],
        batch.get("attention_mask"),
        max_length,
        response_loss_mode,
        device,
    )
    output = model(**_model_forward_inputs(batch), use_cache=False)
    logits = torch.nan_to_num(
        output.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    labels = batch["input_ids"][:, 1:].contiguous()
    mask = batch["loss_mask"][:, 1:].contiguous().bool()
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    token_nll = -logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    row_nll = (token_nll * mask.to(token_nll.dtype)).sum(dim=-1) / mask.to(token_nll.dtype).sum(dim=-1).clamp_min(1.0)

    losses = []
    offset = 0
    for group in groups:
        count = len(group)
        scores = -row_nll[offset : offset + count]
        target = torch.zeros(1, dtype=torch.long, device=scores.device)
        losses.append(torch.nn.functional.cross_entropy(scores.unsqueeze(0), target))
        offset += count
    return torch.stack(losses).mean(), len(groups)


def _teacher_ranked_candidate_loss(
    torch,
    source_model,
    candidate_model,
    source_tokenizer,
    candidate_tokenizer,
    texts: Sequence[str],
    response_loss_mode: str,
    max_negatives: int,
    temperature: float,
    max_length: int,
    device: str,
):
    groups = [
        candidates
        for text in texts
        if (candidates := _answer_contrastive_candidates(str(text), response_loss_mode, max_negatives))
    ]
    if not groups:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0

    flat = [candidate for group in groups for candidate in group]
    with torch.no_grad():
        teacher_nll = _sequence_response_nlls(
            torch,
            source_model,
            source_tokenizer,
            flat,
            response_loss_mode,
            max_length,
            device,
        )
    student_nll = _sequence_response_nlls(
        torch,
        candidate_model,
        candidate_tokenizer,
        flat,
        response_loss_mode,
        max_length,
        device,
    )

    losses = []
    offset = 0
    for group in groups:
        count = len(group)
        teacher_scores = teacher_nll[offset : offset + count]
        student_scores = student_nll[offset : offset + count]
        if count >= 2:
            losses.append(_candidate_rank_kl_loss(torch, teacher_scores, student_scores, temperature))
        offset += count
    if not losses:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0
    return torch.stack(losses).mean(), len(losses)


def _answer_first_token_margin_loss(
    torch,
    shift_student,
    input_ids,
    attention_mask,
    loss_mask,
    tokenizer,
    texts: Sequence[str],
    response_loss_mode: str,
    max_negatives: int,
    margin: float,
    top_negatives: int = 0,
):
    if max_negatives <= 0:
        raise ValueError("max_negatives must be positive.")
    if margin <= 0:
        raise ValueError("margin must be positive.")
    if top_negatives < 0:
        raise ValueError("top_negatives must be non-negative.")
    first_mask = _first_response_shift_mask(torch, loss_mask, attention_mask)
    if first_mask is None or int(first_mask.sum().detach().cpu().item()) <= 0:
        param_dtype = shift_student.dtype
        return torch.zeros((), dtype=param_dtype, device=shift_student.device), 0

    losses = []
    count = 0
    batch_size = min(int(first_mask.shape[0]), len(texts))
    vocab_size = int(shift_student.shape[-1])
    for batch_idx in range(batch_size):
        positions = torch.nonzero(first_mask[batch_idx], as_tuple=False).flatten()
        if int(positions.numel()) <= 0:
            continue
        logit_pos = int(positions[0].detach().cpu().item())
        label_pos = logit_pos + 1
        if label_pos < 0 or label_pos >= int(input_ids.shape[1]):
            continue
        gold_id = int(input_ids[batch_idx, label_pos].detach().cpu().item())
        if gold_id < 0 or gold_id >= vocab_size:
            continue
        row_logits = shift_student[batch_idx, logit_pos]
        negative_ids = _answer_first_token_negative_ids(
            tokenizer,
            str(texts[batch_idx]),
            response_loss_mode,
            max_negatives,
            gold_id,
            vocab_size,
        )
        if int(top_negatives) > 0:
            top_k = min(vocab_size, int(top_negatives) + 1)
            top_ids = torch.topk(row_logits.detach().to(torch.float32), k=top_k, dim=-1).indices.detach().cpu().tolist()
            for top_id in top_ids:
                top_id = int(top_id)
                if top_id == gold_id or top_id in negative_ids:
                    continue
                negative_ids.append(top_id)
                if len(negative_ids) >= int(max_negatives) + int(top_negatives):
                    break
        if not negative_ids:
            continue
        gold_logit = row_logits[gold_id]
        negatives = torch.tensor(negative_ids, dtype=torch.long, device=shift_student.device)
        neg_logits = row_logits.index_select(0, negatives)
        losses.append(torch.relu(float(margin) + neg_logits - gold_logit).mean())
        count += 1
    if not losses:
        return torch.zeros((), dtype=shift_student.dtype, device=shift_student.device), 0
    return torch.stack(losses).mean(), int(count)


def _cross_tokenizer_on_policy_rollout_loss(
    torch,
    source_model,
    candidate_model,
    source_tokenizer,
    candidate_tokenizer,
    texts: Sequence[str],
    response_loss_mode: str,
    max_new_tokens: int,
    max_length: int,
    device: str,
    source_anchor_idx,
    candidate_anchor_idx,
    temperature: float,
    anchor_kd_weight: float,
    unlikelihood_weight: float,
    unlikelihood_window: int,
    *,
    rollout_texts: Sequence[str] | None = None,
):
    if anchor_kd_weight < 0 or unlikelihood_weight < 0:
        raise ValueError("on-policy loss weights must be non-negative.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if anchor_kd_weight == 0 and unlikelihood_weight == 0:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0
    if rollout_texts is None:
        prompts = [_on_policy_prompt_text(str(text), response_loss_mode) for text in texts]
        prompts = [prompt for prompt in prompts if prompt.strip()]
        if not prompts:
            param = next(candidate_model.parameters())
            return torch.zeros((), dtype=param.dtype, device=param.device), 0

        prompt_max_length = max(2, int(max_length) - int(max_new_tokens))
        with _temporary_padding_side(candidate_tokenizer, "left"):
            encoded = candidate_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_max_length,
            )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        pad_id = candidate_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = candidate_tokenizer.eos_token_id
        was_training = candidate_model.training
        try:
            candidate_model.eval()
            with _temporary_padding_side(candidate_tokenizer, "left"):
                with torch.no_grad():
                    generated = candidate_model.generate(
                        **encoded,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=candidate_tokenizer.eos_token_id,
                    )
        finally:
            if was_training:
                candidate_model.train()
        resolved_rollout_texts = [
            candidate_tokenizer.decode(
                row.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for row in generated.detach().cpu()
        ]
        resolved_rollout_texts = [text for text in resolved_rollout_texts if text.strip()]
    else:
        resolved_rollout_texts = list(
            _freeze_ordered_recovery_texts(rollout_texts, field_name="rollout_texts")
        )
    if not resolved_rollout_texts:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0

    paired = _make_paired_batches(
        source_tokenizer,
        candidate_tokenizer,
        resolved_rollout_texts,
        batch_size=len(resolved_rollout_texts),
        max_length=max_length,
        device=device,
        include_offsets=False,
        response_loss_mode="none",
        append_eos=False,
    )
    source_batch, candidate_batch = paired[0]
    with torch.no_grad():
        teacher = source_model(**_model_forward_inputs(source_batch), use_cache=False)
    student = candidate_model(**_model_forward_inputs(candidate_batch), use_cache=False)
    total = torch.zeros((), dtype=student.logits.dtype, device=student.logits.device)
    if anchor_kd_weight > 0:
        total = total + _cross_tokenizer_recovery_loss(
            torch,
            student,
            teacher,
            candidate_batch["input_ids"],
            candidate_batch.get("attention_mask"),
            None,
            source_batch.get("attention_mask"),
            candidate_anchor_idx,
            source_anchor_idx,
            float(anchor_kd_weight),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "teacher-to-student",
            0.0,
            0.0,
            unlikelihood_window,
            0.0,
            0.0,
            0.55,
            8.0,
            None,
            "last",
            "projected",
            None,
            None,
            None,
            None,
            None,
            temperature,
            source_input_ids=source_batch["input_ids"],
        )
    if unlikelihood_weight > 0:
        shift_logits = torch.nan_to_num(
            student.logits[:, :-1, :].contiguous().to(torch.float32),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        shift_labels = candidate_batch["input_ids"][:, 1:].contiguous()
        mask = (
            torch.ones_like(shift_labels, dtype=torch.bool)
            if candidate_batch.get("attention_mask") is None
            else candidate_batch["attention_mask"][:, 1:].contiguous().bool()
        )
        total = total + float(unlikelihood_weight) * _recent_token_unlikelihood_loss(
            torch,
            shift_logits,
            candidate_batch["input_ids"],
            shift_labels,
            mask,
            window=unlikelihood_window,
            exclude_current_label=False,
        )
    return total, len(resolved_rollout_texts)


def _cross_tokenizer_on_policy_hidden_loss(
    torch,
    source_model,
    candidate_model,
    source_tokenizer,
    candidate_tokenizer,
    texts: Sequence[str],
    response_loss_mode: str,
    max_new_tokens: int,
    max_length: int,
    device: str,
    hidden_projection,
    hidden_layer: str,
    hidden_kd_metric: str,
    span_max_pairs: int,
    *,
    rollout_texts: Sequence[str] | None = None,
):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if span_max_pairs <= 0:
        raise ValueError("span_max_pairs must be positive.")
    if hidden_layer not in {"last", "middle"}:
        raise ValueError("hidden_layer must be last or middle.")
    if hidden_kd_metric not in HIDDEN_KD_METRICS:
        raise ValueError(f"hidden_kd_metric must be {HIDDEN_KD_METRIC_DESCRIPTION}.")
    if hidden_projection is None and hidden_kd_metric in PROJECTED_HIDDEN_KD_METRICS:
        raise ValueError("hidden_projection is required when on_policy_hidden_kd_weight is positive.")

    from neural_forge.hf_xfamily import _canonical_token

    if rollout_texts is None:
        prompts = [_on_policy_prompt_text(str(text), response_loss_mode) for text in texts]
        prompts = [prompt for prompt in prompts if prompt.strip()]
        if not prompts:
            param = next(candidate_model.parameters())
            return torch.zeros((), dtype=param.dtype, device=param.device), 0

        prompt_max_length = max(2, int(max_length) - int(max_new_tokens))
        with _temporary_padding_side(candidate_tokenizer, "left"):
            encoded = candidate_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_max_length,
            )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        pad_id = candidate_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = candidate_tokenizer.eos_token_id
        was_training = candidate_model.training
        try:
            candidate_model.eval()
            with _temporary_padding_side(candidate_tokenizer, "left"):
                with torch.no_grad():
                    generated = candidate_model.generate(
                        **encoded,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=candidate_tokenizer.eos_token_id,
                    )
        finally:
            if was_training:
                candidate_model.train()

        resolved_rollout_texts = [
            candidate_tokenizer.decode(
                row.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for row in generated.detach().cpu()
        ]
        resolved_rollout_texts = [text for text in resolved_rollout_texts if text.strip()]
    else:
        resolved_rollout_texts = list(
            _freeze_ordered_recovery_texts(rollout_texts, field_name="rollout_texts")
        )
    if not resolved_rollout_texts:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0

    paired = _make_paired_batches(
        source_tokenizer,
        candidate_tokenizer,
        resolved_rollout_texts,
        batch_size=len(resolved_rollout_texts),
        max_length=max_length,
        device=device,
        include_offsets=True,
        response_loss_mode="none",
        append_eos=False,
    )
    source_batch, candidate_batch = paired[0]
    with torch.no_grad():
        teacher = source_model(
            **_model_forward_inputs(source_batch),
            use_cache=False,
            output_hidden_states=True,
        )
    student = candidate_model(
        **_model_forward_inputs(candidate_batch),
        use_cache=False,
        output_hidden_states=True,
    )
    span_pairs = _cross_tokenizer_span_pairs(
        torch,
        source_tokenizer,
        candidate_tokenizer,
        source_batch,
        candidate_batch,
        _canonical_token,
        max_pairs=span_max_pairs,
        shift_positions=False,
    )
    if int(span_pairs.numel()) > 0:
        return (
            _cross_tokenizer_span_hidden_loss(
                torch,
                student,
                teacher,
                span_pairs,
                hidden_projection,
                hidden_layer,
                hidden_kd_metric,
            ),
            int(span_pairs.shape[0]),
        )
    return (
        _cross_tokenizer_hidden_loss(
            torch,
            student,
            teacher,
            candidate_batch.get("attention_mask"),
            source_batch.get("attention_mask"),
            hidden_projection,
            hidden_layer,
            hidden_kd_metric,
        ),
        len(resolved_rollout_texts),
    )


def _on_policy_prompt_text(text: str, response_loss_mode: str) -> str:
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    start = _infer_response_char_start(text, response_loss_mode)
    if start is None or int(start) <= 0:
        return text
    return text[: int(start)]


@contextmanager
def _temporary_padding_side(tokenizer, padding_side: str):
    original = getattr(tokenizer, "padding_side", None)
    if original is None:
        yield
        return
    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        tokenizer.padding_side = original


def _teacher_generated_lm_texts(
    torch,
    source_model,
    source_tokenizer,
    texts: Sequence[str],
    response_loss_mode: str,
    max_new_tokens: int,
    max_length: int,
    batch_size: int,
    device: str,
) -> list[str]:
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    prompts = [_on_policy_prompt_text(str(text), response_loss_mode) for text in texts]
    prompts = [prompt for prompt in prompts if prompt.strip()]
    if not prompts:
        return []

    pad_id = source_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = source_tokenizer.eos_token_id
    prompt_max_length = max(2, int(max_length) - int(max_new_tokens))
    generated_texts: list[str] = []
    seen: set[str] = set()
    was_training = source_model.training
    source_model.eval()
    try:
        for start in range(0, len(prompts), int(batch_size)):
            chunk = prompts[start : start + int(batch_size)]
            with _temporary_padding_side(source_tokenizer, "left"):
                encoded = source_tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=prompt_max_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.no_grad():
                    generated = source_model.generate(
                        **encoded,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=source_tokenizer.eos_token_id,
                    )
            for row in generated.detach().cpu():
                text = source_tokenizer.decode(
                    row.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                if text and text not in seen:
                    seen.add(text)
                    generated_texts.append(text)
    finally:
        if was_training:
            source_model.train()
    return generated_texts


def _stepwise_replay_vocab_limit(candidate_model, candidate_tokenizer) -> int:
    try:
        tokenizer_limit = int(len(candidate_tokenizer))
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate tokenizer must expose a positive vocabulary size for token replay.") from exc
    if tokenizer_limit <= 0:
        raise ValueError("Candidate tokenizer must expose a positive vocabulary size for token replay.")
    embedding_limit = tokenizer_limit
    get_input_embeddings = getattr(candidate_model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        num_embeddings = getattr(embeddings, "num_embeddings", None)
        if num_embeddings is not None:
            embedding_limit = int(num_embeddings)
            if embedding_limit <= 0:
                raise ValueError("Candidate input embedding vocabulary must be positive for token replay.")
    return min(tokenizer_limit, embedding_limit)


def _validated_stepwise_replay_tensors(
    torch,
    candidate_model,
    candidate_tokenizer,
    prompts: Sequence[str],
    replay_event: _FrozenStepwiseReplayEvent,
    *,
    prompt_max_length: int,
    max_new_tokens: int,
    max_length: int,
    device: str,
):
    if len(replay_event.rows) != len(prompts):
        raise ValueError(
            "Frozen stepwise token replay event must contain exactly one row per current prompt "
            f"({len(prompts)} required, got {len(replay_event.rows)})."
        )
    with _temporary_padding_side(candidate_tokenizer, "left"):
        encoded_prompts = candidate_tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        )
    prompt_attention_mask = encoded_prompts.get("attention_mask")
    if prompt_attention_mask is None:
        raise ValueError("Candidate tokenizer must return attention_mask for exact stepwise token replay.")
    prompt_input_ids = encoded_prompts["input_ids"].detach().cpu()
    prompt_attention_mask = prompt_attention_mask.detach().cpu().bool()
    if int(prompt_input_ids.shape[0]) != len(prompts):
        raise ValueError("Candidate tokenizer returned the wrong prompt batch size for token replay.")

    vocab_limit = _stepwise_replay_vocab_limit(candidate_model, candidate_tokenizer)
    prompt_lengths: list[int] = []
    rollout_lengths: list[int] = []
    for row_index, replay_row in enumerate(replay_event.rows):
        runtime_prompt_ids = tuple(
            int(token_id)
            for token_id in prompt_input_ids[row_index][prompt_attention_mask[row_index]].tolist()
        )
        if runtime_prompt_ids != replay_row.prompt_input_ids:
            raise ValueError(
                "Frozen stepwise token replay prompt IDs do not match the current runtime prompt "
                f"at step {replay_event.step}, row {row_index}."
            )
        if len(replay_row.prompt_input_ids) > int(prompt_max_length):
            raise ValueError(
                f"Frozen stepwise token replay prompt at row {row_index} exceeds prompt_max_length."
            )
        continuation_count = len(replay_row.rollout_input_ids) - len(replay_row.prompt_input_ids)
        if continuation_count <= 0:
            raise ValueError(
                f"Frozen stepwise token replay row {row_index} must contain a continuation token."
            )
        if continuation_count > int(max_new_tokens):
            raise ValueError(
                f"Frozen stepwise token replay row {row_index} exceeds max_new_tokens."
            )
        if len(replay_row.rollout_input_ids) > int(max_length):
            raise ValueError(f"Frozen stepwise token replay row {row_index} exceeds max_length.")
        for field_name, token_ids in (
            ("prompt_input_ids", replay_row.prompt_input_ids),
            ("rollout_input_ids", replay_row.rollout_input_ids),
        ):
            for token_index, token_id in enumerate(token_ids):
                if token_id >= vocab_limit:
                    raise ValueError(
                        f"Frozen stepwise token replay row {row_index} {field_name}[{token_index}] "
                        f"is outside the candidate vocabulary [0, {vocab_limit})."
                    )
        prompt_lengths.append(len(replay_row.prompt_input_ids))
        rollout_lengths.append(len(replay_row.rollout_input_ids))

    pad_id = candidate_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = candidate_tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    if isinstance(pad_id, bool) or not isinstance(pad_id, int) or not 0 <= int(pad_id) < vocab_limit:
        raise ValueError("Candidate tokenizer padding ID is outside the candidate vocabulary.")
    rollout_width = max(rollout_lengths)
    generated = torch.full(
        (len(replay_event.rows), rollout_width),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    generated_mask = torch.zeros_like(generated, dtype=torch.long, device=device)
    for row_index, replay_row in enumerate(replay_event.rows):
        row_ids = torch.tensor(replay_row.rollout_input_ids, dtype=torch.long, device=device)
        row_length = len(replay_row.rollout_input_ids)
        generated[row_index, :row_length] = row_ids
        generated_mask[row_index, :row_length] = 1
    return generated, generated_mask, prompt_lengths, rollout_lengths


def _cross_tokenizer_stepwise_prefix_kd_loss(
    torch,
    source_model,
    candidate_model,
    source_tokenizer,
    candidate_tokenizer,
    texts: Sequence[str],
    response_loss_mode: str,
    max_new_tokens: int,
    max_length: int,
    device: str,
    projected_vocab_groups,
    projected_vocab_mask,
    projected_vocab_weights,
    projected_vocab_valid,
    temperature: float,
    *,
    replay_event: _FrozenStepwiseReplayEvent | None = None,
):
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    prompts = [_on_policy_prompt_text(str(text), response_loss_mode) for text in texts]
    prompts = [prompt for prompt in prompts if prompt.strip()]
    if not prompts:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0

    pad_id = candidate_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = candidate_tokenizer.eos_token_id
    prompt_max_length = max(2, int(max_length) - int(max_new_tokens))
    if replay_event is None:
        with _temporary_padding_side(candidate_tokenizer, "left"):
            encoded = candidate_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_max_length,
            )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        prompt_lengths = torch.full(
            (int(encoded["input_ids"].shape[0]),),
            prompt_width,
            dtype=torch.long,
            device=device,
        )
        was_training = candidate_model.training
        try:
            candidate_model.eval()
            with _temporary_padding_side(candidate_tokenizer, "left"):
                with torch.no_grad():
                    generated = candidate_model.generate(
                        **encoded,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=candidate_tokenizer.eos_token_id,
                    )
        finally:
            if was_training:
                candidate_model.train()
        generated = generated.to(device)
        generated_mask = torch.ones_like(generated, dtype=torch.long, device=device)
        prompt_lengths_cpu = [int(item) for item in prompt_lengths.detach().cpu().tolist()]
    else:
        generated, generated_mask, prompt_lengths_cpu, row_lengths = (
            _validated_stepwise_replay_tensors(
                torch,
                candidate_model,
                candidate_tokenizer,
                prompts,
                replay_event,
                prompt_max_length=prompt_max_length,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                device=device,
            )
        )

    if replay_event is None:
        row_lengths: list[int] = []
    generated_cpu = generated.detach().cpu()
    if replay_event is None:
        for batch_idx, row in enumerate(generated_cpu):
            ids = [int(item) for item in row.tolist()]
            length = len(ids)
            if pad_id is not None:
                while (
                    length > prompt_lengths_cpu[batch_idx]
                    and length > 0
                    and ids[length - 1] == int(pad_id)
                ):
                    length -= 1
            row_lengths.append(max(0, length))
            if length < int(generated.shape[1]):
                generated_mask[batch_idx, length:] = 0

    student = candidate_model(
        input_ids=generated,
        attention_mask=generated_mask,
        use_cache=False,
    )
    shift_student = torch.nan_to_num(
        student.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    prefix_texts: list[str] = []
    row_positions: list[tuple[int, int]] = []
    max_shift_pos = int(shift_student.shape[1]) - 1
    for batch_idx, length in enumerate(row_lengths):
        prompt_len = prompt_lengths_cpu[batch_idx]
        if length <= prompt_len or prompt_len <= 0:
            continue
        first_logit_pos = max(0, prompt_len - 1)
        last_logit_pos = min(length - 2, first_logit_pos + int(max_new_tokens) - 1, max_shift_pos)
        for logit_pos in range(first_logit_pos, last_logit_pos + 1):
            prefix_ids = [int(item) for item in generated_cpu[batch_idx, : logit_pos + 1].tolist()]
            prefix = candidate_tokenizer.decode(
                prefix_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not prefix.strip():
                if replay_event is not None:
                    raise ValueError(
                        "Frozen stepwise token replay produced a blank decoded teacher prefix "
                        f"at step {replay_event.step}, row {batch_idx}, position {logit_pos}."
                    )
                continue
            prefix_texts.append(prefix)
            row_positions.append((batch_idx, logit_pos))
    if not row_positions:
        if replay_event is not None:
            raise ValueError(
                f"Frozen stepwise token replay event at step {replay_event.step} produced no KD positions."
            )
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0

    source_batch = source_tokenizer(
        prefix_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    source_batch = {key: value.to(device) for key, value in source_batch.items()}
    source_lengths = (
        source_batch["attention_mask"].sum(dim=1).to(dtype=torch.long)
        if "attention_mask" in source_batch
        else torch.full(
            (len(prefix_texts),),
            int(source_batch["input_ids"].shape[1]),
            dtype=torch.long,
            device=device,
        )
    )
    with torch.no_grad():
        teacher = source_model(**_model_forward_inputs(source_batch), use_cache=False)
    teacher_rows = torch.arange(len(prefix_texts), dtype=torch.long, device=device)
    teacher_pos = (source_lengths - 1).clamp(min=0, max=int(teacher.logits.shape[1]) - 1)
    teacher_logits = torch.nan_to_num(
        teacher.logits[teacher_rows, teacher_pos].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    projected_teacher = _project_source_logits_to_candidate_vocab(
        torch,
        teacher_logits,
        projected_vocab_groups,
        projected_vocab_mask,
        projected_vocab_valid,
        projected_vocab_weights=projected_vocab_weights,
    )
    batch_positions = torch.tensor([item[0] for item in row_positions], dtype=torch.long, device=device)
    token_positions = torch.tensor([item[1] for item in row_positions], dtype=torch.long, device=device)
    student_logits = shift_student[batch_positions, token_positions]
    loss = _entropy_aware_kl_loss(
        torch,
        student_logits,
        projected_teacher,
        temperature,
        0.0,
        0.55,
        8.0,
        vocab_valid=projected_vocab_valid,
    )
    return loss.to(device=student_logits.device, dtype=torch.float32), len(row_positions)


def _sequence_response_nlls(
    torch,
    model,
    tokenizer,
    texts: Sequence[str],
    response_loss_mode: str,
    max_length: int,
    device: str,
):
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    batch = {key: value.to(device) for key, value in encoded.items()}
    batch["loss_mask"] = _response_loss_mask(
        tokenizer,
        list(texts),
        batch["input_ids"],
        batch.get("attention_mask"),
        max_length,
        response_loss_mode,
        device,
    )
    output = model(**_model_forward_inputs(batch), use_cache=False)
    logits = torch.nan_to_num(
        output.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    labels = batch["input_ids"][:, 1:].contiguous()
    mask = batch["loss_mask"][:, 1:].contiguous().bool()
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    token_nll = -logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    denom = mask.to(token_nll.dtype).sum(dim=-1).clamp_min(1.0)
    return (token_nll * mask.to(token_nll.dtype)).sum(dim=-1) / denom


def _candidate_rank_kl_loss(torch, teacher_nlls, student_nlls, temperature: float):
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if int(teacher_nlls.numel()) != int(student_nlls.numel()):
        raise ValueError("teacher and student candidate groups must have matching lengths.")
    if int(teacher_nlls.numel()) < 2:
        return torch.zeros((), dtype=student_nlls.dtype, device=student_nlls.device)
    target = torch.nn.functional.softmax((-teacher_nlls.detach().to(torch.float32)) / float(temperature), dim=0)
    student_logp = torch.nn.functional.log_softmax((-student_nlls.to(torch.float32)) / float(temperature), dim=0)
    return -(target * student_logp).sum()


def _answer_contrastive_candidates(text: str, response_loss_mode: str, max_negatives: int) -> list[str]:
    if max_negatives <= 0:
        raise ValueError("max_negatives must be positive.")
    start = _infer_response_char_start(text, response_loss_mode)
    if start is None:
        return []
    answer_start = int(start)
    while answer_start < len(text) and text[answer_start].isspace():
        answer_start += 1
    prefix = text[:answer_start]
    response = text[answer_start:].strip()
    if not response:
        return []

    number = re.match(r"-?\d+(?:\.\d+)?", response)
    if number is not None:
        gold = number.group(0)
        negatives = _numeric_answer_negatives(gold, max_negatives, prompt=prefix)
        return [f"{prefix}{item}" for item in [gold, *negatives]]

    choice = re.match(r"([A-D])(?:[.)])?", response, flags=re.IGNORECASE)
    if choice is not None:
        gold = choice.group(1).upper()
        negatives = [letter for letter in ("A", "B", "C", "D") if letter != gold][:max_negatives]
        return [f"{prefix}{item}" for item in [gold, *negatives]]
    return []


def _numeric_answer_negatives(gold: str, max_negatives: int, prompt: str = "") -> list[str]:
    if max_negatives <= 0:
        raise ValueError("max_negatives must be positive.")
    negatives: list[str] = []
    if re.fullmatch(r"-?\d+", gold):
        value = int(gold)
        proposals = _arithmetic_prompt_negative_values(prompt, value)
        proposals.extend(
            [
                value - 1,
                value + 1,
                value - 2,
                value + 2,
                value - 3,
                value + 3,
                value - 5,
                value + 5,
                value - 10,
                value + 10,
                value * 2,
                max(0, value // 2),
            ]
        )
        for proposal in proposals:
            item = str(proposal)
            if item != gold and item not in negatives:
                negatives.append(item)
            if len(negatives) >= max_negatives:
                break
        return negatives
    try:
        value = float(gold)
    except ValueError:
        return []
    for delta in (-1.0, 1.0, -0.5, 0.5, 2.0, -2.0):
        item = f"{value + delta:g}"
        if item != gold and item not in negatives:
            negatives.append(item)
        if len(negatives) >= max_negatives:
            break
    return negatives


def _answer_first_token_negative_ids(
    tokenizer,
    text: str,
    response_loss_mode: str,
    max_negatives: int,
    gold_id: int,
    vocab_size: int | None = None,
) -> list[int]:
    if max_negatives <= 0:
        raise ValueError("max_negatives must be positive.")
    start = _infer_response_char_start(text, response_loss_mode)
    if start is None:
        return []
    answer_start = int(start)
    while answer_start < len(text) and text[answer_start].isspace():
        answer_start += 1
    prefix = text[:answer_start]
    response = text[answer_start:].strip()
    if not response:
        return []

    negatives: list[str]
    number = re.match(r"-?\d+(?:\.\d+)?", response)
    if number is not None:
        negatives = _numeric_answer_negatives(number.group(0), max_negatives, prompt=prefix)
    else:
        choice = re.match(r"([A-D])(?:[.)])?", response, flags=re.IGNORECASE)
        if choice is None:
            return []
        gold = choice.group(1).upper()
        negatives = [letter for letter in ("A", "B", "C", "D") if letter != gold][:max_negatives]

    ids: list[int] = []
    for negative in negatives:
        token_id = _first_token_id_for_prefix_option(tokenizer, prefix, negative)
        if token_id is None:
            continue
        if vocab_size is not None and (int(token_id) < 0 or int(token_id) >= int(vocab_size)):
            continue
        if int(token_id) == int(gold_id) or int(token_id) in ids:
            continue
        ids.append(int(token_id))
        if len(ids) >= int(max_negatives):
            break
    return ids


def _first_token_id_for_prefix_option(tokenizer, prefix: str, option: str) -> int | None:
    full_text = f"{prefix}{option}"
    prefix_len = len(prefix)
    try:
        encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
        ids = _flatten_tokenizer_ids(encoded.get("input_ids", []))
        offsets = encoded.get("offset_mapping", [])
        if offsets and len(offsets) == len(ids):
            for token_id, offset in zip(ids, offsets):
                if len(offset) < 2:
                    continue
                start, end = int(offset[0]), int(offset[1])
                if end > prefix_len and end > start:
                    return int(token_id)
    except Exception:
        pass

    fallback_options = [option]
    if prefix and prefix[-1].isspace():
        fallback_options.insert(0, f" {option}")
    for text in fallback_options:
        try:
            ids = _flatten_tokenizer_ids(tokenizer(text, add_special_tokens=False).get("input_ids", []))
        except Exception:
            ids = []
        if ids:
            return int(ids[0])
    return None


def _flatten_tokenizer_ids(ids) -> list[int]:
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(item) for item in ids]


def _arithmetic_prompt_negative_values(prompt: str, gold_value: int) -> list[int]:
    numbers = [int(match) for match in re.findall(r"(?<![\w.])-?\d+(?![\w.])", prompt)]
    if len(numbers) < 2:
        return []
    left, right = numbers[-2], numbers[-1]
    lower = prompt.lower()
    is_multiply = bool(re.search(r"\*|\bx\b|\btimes\b|\bmultipl", lower))
    is_add = bool(re.search(r"\+|\bplus\b|\bsum\b|\badd", lower))
    add_value = left + right
    mul_value = left * right
    diff_value = abs(left - right)
    concat_value = int(f"{abs(left)}{abs(right)}") if left >= 0 and right >= 0 else None
    proposals: list[int | None]
    if is_multiply and not is_add:
        proposals = [
            add_value,
            mul_value - left,
            mul_value + left,
            mul_value - right,
            mul_value + right,
            diff_value,
            concat_value,
        ]
    elif is_add and not is_multiply:
        proposals = [
            mul_value,
            add_value - left,
            add_value + left,
            add_value - right,
            add_value + right,
            diff_value,
            concat_value,
        ]
    else:
        proposals = [
            add_value,
            mul_value,
            diff_value,
            concat_value,
        ]
    clean: list[int] = []
    for proposal in proposals:
        if proposal is None or proposal == gold_value:
            continue
        if proposal not in clean:
            clean.append(int(proposal))
    return clean


def _shift_positions_for_char_spans(torch, offsets, attention_mask, spans: Sequence[tuple[int, int]], device):
    positions = []
    for token_pos in range(1, int(offsets.shape[0])):
        if attention_mask is not None and not bool(attention_mask[token_pos]):
            continue
        start = int(offsets[token_pos, 0])
        end = int(offsets[token_pos, 1])
        if end <= start:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            positions.append(token_pos - 1)
    return torch.tensor(positions, dtype=torch.long, device=device)


def _decoded_token(tokenizer, token_id: int, canonical_fn) -> str:
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if int(token_id) in special_ids:
        return ""
    try:
        decoded = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        decoded = ""
    return canonical_fn(decoded)


def _projected_vocab_surface_category(decoded: str, canonical_fn) -> str:
    canonical = canonical_fn(decoded)
    if not canonical:
        return "empty"
    if canonical.isdigit():
        return "digit"
    has_alpha = any(ch.isalpha() for ch in canonical)
    has_digit = any(ch.isdigit() for ch in canonical)
    if has_alpha and has_digit:
        return "mixed_digit"
    if re.search(r"[{}\[\]().,;:+=*/<>_`'\"\\|-]", canonical):
        return "code"
    if has_alpha:
        return "alpha"
    return "other"


def _projected_vocab_prefers_sparse_surface(decoded: str, canonical_fn, source_token_count: int, max_source_tokens: int) -> bool:
    if source_token_count <= 0:
        return True
    if source_token_count <= max_source_tokens:
        return False
    category = _projected_vocab_surface_category(decoded, canonical_fn)
    return category in {"digit", "mixed_digit", "code"}


def _projected_vocab_group_summary(
    torch,
    candidate_tokenizer,
    projected_vocab_mask,
    projected_vocab_valid,
    projected_vocab_weights=None,
) -> dict[str, Any]:
    from neural_forge.hf_xfamily import _canonical_token

    valid = projected_vocab_valid.detach().cpu().bool()
    counts = projected_vocab_mask.detach().cpu().to(torch.float32).sum(dim=-1)
    summary: dict[str, Any] = {
        "candidate_vocab_size": int(len(candidate_tokenizer)),
        "valid_rows": int(valid.sum().item()),
        "valid_fraction": (
            float(valid.sum().item() / max(1, len(candidate_tokenizer)))
            if len(candidate_tokenizer) > 0
            else 0.0
        ),
        "multi_source_rows": int((counts > 1).sum().item()),
        "max_source_tokens_observed": int(counts.max().item()) if int(counts.numel()) else 0,
    }
    if projected_vocab_weights is not None:
        weights = projected_vocab_weights.detach().cpu().to(torch.float32)
        active_weights = weights * projected_vocab_mask.detach().cpu().to(torch.float32)
        weight_sums = active_weights.sum(dim=-1)
        non_uniform = (active_weights - projected_vocab_mask.detach().cpu().to(torch.float32) / counts.clamp_min(1.0).unsqueeze(-1)).abs()
        summary["weighted_rows"] = int(((non_uniform.sum(dim=-1) > 1e-6) & valid).sum().item())
        summary["min_weight_sum"] = float(weight_sums[valid].min().item()) if bool(valid.any()) else 0.0
        summary["max_weight_sum"] = float(weight_sums[valid].max().item()) if bool(valid.any()) else 0.0
    for category in ("special", "digit", "mixed_digit", "code", "alpha", "other", "empty"):
        summary[f"{category}_rows"] = 0
        summary[f"{category}_valid_rows"] = 0
    special_ids = set(getattr(candidate_tokenizer, "all_special_ids", []) or [])
    for candidate_id in range(len(candidate_tokenizer)):
        if int(candidate_id) in special_ids:
            category = "special"
        else:
            decoded = _decode_token_for_projection(candidate_tokenizer, int(candidate_id))
            category = _projected_vocab_surface_category(decoded, _canonical_token)
        summary[f"{category}_rows"] += 1
        if bool(valid[int(candidate_id)].item()):
            summary[f"{category}_valid_rows"] += 1
    return summary


def _decode_token_for_projection(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _candidate_to_source_token_groups(
    torch,
    source_tokenizer,
    candidate_tokenizer,
    max_source_tokens: int,
    device: str,
):
    from neural_forge.hf_xfamily import _canonical_token
    from neural_forge.hf_xfamily import _canonical_token_map
    from neural_forge.hf_xfamily import _donor_omp_candidate_ids
    from neural_forge.hf_xfamily import _donor_special_ids_for_target_role
    from neural_forge.hf_xfamily import _donor_tokenized_group_weights
    from neural_forge.hf_xfamily import _encode_without_specials
    from neural_forge.hf_xfamily import _omp_token_piece_weights
    from neural_forge.hf_xfamily import _special_token_ids_by_decoded_surface
    from neural_forge.hf_xfamily import _special_token_ids_by_role

    if max_source_tokens <= 0:
        raise ValueError("max_source_tokens must be positive.")
    candidate_special_ids = set(getattr(candidate_tokenizer, "all_special_ids", []) or [])
    source_special_ids = set(getattr(source_tokenizer, "all_special_ids", []) or [])
    source_vocab_size = len(source_tokenizer)
    source_map = _canonical_token_map(source_tokenizer)
    candidate_special_roles = _special_token_ids_by_role(candidate_tokenizer)
    source_special_roles = _special_token_ids_by_role(source_tokenizer)
    source_special_decoded = _special_token_ids_by_decoded_surface(source_tokenizer)
    candidate_roles_by_id: dict[int, list[str]] = {}
    for role, role_ids in candidate_special_roles.items():
        for role_id in role_ids:
            candidate_roles_by_id.setdefault(int(role_id), []).append(role)
    omp_candidate_limit = max(32, int(max_source_tokens) * 8)
    groups = []
    masks = []
    weights = []
    valids = []
    for candidate_id in range(len(candidate_tokenizer)):
        ids: list[int] = []
        token_weights: list[float] = []
        if candidate_id in candidate_special_ids:
            for role in candidate_roles_by_id.get(int(candidate_id), ["all_special_ids"]):
                role_ids = _donor_special_ids_for_target_role(
                    role,
                    int(candidate_id),
                    candidate_tokenizer,
                    source_special_roles,
                    source_special_decoded,
                )
                ids = []
                for source_id in role_ids:
                    source_id = int(source_id)
                    if 0 <= source_id < source_vocab_size and source_id not in ids:
                        ids.append(source_id)
                    if len(ids) >= int(max_source_tokens):
                        break
                if ids:
                    token_weights = [1.0] * len(ids)
                    break
        else:
            decoded = _decode_token_for_projection(candidate_tokenizer, int(candidate_id))
            if decoded and not any(ord(ch) < 32 for ch in decoded):
                encoded = _encode_without_specials(source_tokenizer, decoded)
                direct_ids = [
                    int(item)
                    for item in encoded
                    if int(item) not in source_special_ids and 0 <= int(item) < source_vocab_size
                ]
                ids = direct_ids[: int(max_source_tokens)]
                token_weights = _donor_tokenized_group_weights(source_tokenizer, ids) if ids else []
                if _projected_vocab_prefers_sparse_surface(
                    decoded,
                    _canonical_token,
                    source_token_count=len(direct_ids),
                    max_source_tokens=int(max_source_tokens),
                ):
                    candidate_ids = _donor_omp_candidate_ids(
                        source_tokenizer,
                        source_map,
                        decoded,
                        source_vocab_size,
                        source_special_ids,
                        max_candidates=omp_candidate_limit,
                    )
                    sparse_ids, _weights = _omp_token_piece_weights(
                        torch,
                        decoded,
                        source_tokenizer,
                        candidate_ids,
                        max_selected=max_source_tokens,
                    )
                    if sparse_ids:
                        ids = [int(item) for item in sparse_ids[: int(max_source_tokens)]]
                        token_weights = [float(item) for item in _weights[: len(ids)]]
        if ids:
            if len(token_weights) != len(ids) or sum(token_weights) <= 0:
                token_weights = [1.0 / len(ids)] * len(ids)
            else:
                total_weight = sum(max(0.0, float(item)) for item in token_weights)
                if total_weight <= 0:
                    token_weights = [1.0 / len(ids)] * len(ids)
                else:
                    token_weights = [max(0.0, float(item)) / total_weight for item in token_weights]
        valid = bool(ids)
        padded = ids + [0] * (int(max_source_tokens) - len(ids))
        padded_weights = token_weights + [0.0] * (int(max_source_tokens) - len(token_weights))
        groups.append(padded[: int(max_source_tokens)])
        masks.append([1.0 if idx < len(ids) else 0.0 for idx in range(int(max_source_tokens))])
        weights.append(padded_weights[: int(max_source_tokens)])
        valids.append(valid)
    return (
        torch.tensor(groups, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.float32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
        torch.tensor(valids, dtype=torch.bool, device=device),
    )


def _shared_recovery_anchors(
    source_tokenizer,
    candidate_tokenizer,
    max_anchors: int,
    map_fn,
    selection: str = "stratified",
    min_chars: int = 1,
) -> list[tuple[str, int, int]]:
    source_map = map_fn(source_tokenizer)
    candidate_map = map_fn(candidate_tokenizer)
    shared_all = list(set(source_map) & set(candidate_map))
    shared = [token for token in shared_all if len(token.strip()) >= int(min_chars)]
    if not shared:
        shared = shared_all
    ordered = _select_anchor_tokens(shared, max_anchors=max_anchors, selection=selection)
    return [(token, source_map[token], candidate_map[token]) for token in ordered]


def _select_anchor_tokens(tokens: Sequence[str], max_anchors: int, selection: str = "stratified") -> list[str]:
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")
    unique = sorted(set(tokens))
    if selection == "shortest":
        return sorted(unique, key=lambda item: (len(item), item))[:max_anchors]
    if selection == "longest":
        return sorted(unique, key=lambda item: (-len(item), item))[:max_anchors]
    if selection != "stratified":
        raise ValueError("selection must be shortest, longest, or stratified.")

    buckets: dict[int, list[str]] = {idx: [] for idx in range(7)}
    for token in unique:
        length = len(token.strip())
        if length <= 1:
            bucket = 0
        elif length == 2:
            bucket = 1
        elif length == 3:
            bucket = 2
        elif length <= 5:
            bucket = 3
        elif length <= 8:
            bucket = 4
        elif length <= 12:
            bucket = 5
        else:
            bucket = 6
        buckets[bucket].append(token)
    for bucket_tokens in buckets.values():
        bucket_tokens.sort(key=_anchor_token_priority)

    selected: list[str] = []
    bucket_order = [3, 4, 5, 2, 6, 1, 0]
    while len(selected) < max_anchors:
        changed = False
        for bucket in bucket_order:
            if buckets[bucket]:
                selected.append(buckets[bucket].pop(0))
                changed = True
                if len(selected) >= max_anchors:
                    break
        if not changed:
            break
    return selected


def _anchor_token_priority(token: str) -> tuple[int, int, str]:
    stripped = token.strip()
    has_alnum = any(ch.isalnum() for ch in stripped)
    has_alpha = any(ch.isalpha() for ch in stripped)
    return (0 if has_alpha else 1 if has_alnum else 2, abs(len(stripped) - 6), stripped)


def _fit_shared_embedding_hidden_projection(
    torch,
    source,
    candidate,
    source_anchor_idx,
    candidate_anchor_idx,
    ridge: float,
    max_anchors: int,
) -> dict[str, Any]:
    """Fit a temporary donor-hidden -> child-hidden map from shared token embeddings."""

    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive.")
    available = int(source_anchor_idx.numel())
    if available <= 0:
        raise ValueError("Need at least one shared anchor for hidden KD projection.")
    if available > max_anchors:
        sample = torch.linspace(
            0,
            available - 1,
            steps=int(max_anchors),
            device=source_anchor_idx.device,
            dtype=torch.float32,
        ).round().to(torch.long)
        source_anchor_idx = source_anchor_idx.index_select(0, sample)
        candidate_anchor_idx = candidate_anchor_idx.index_select(0, sample)

    source_embeddings = source.get_input_embeddings()
    candidate_embeddings = candidate.get_input_embeddings()
    if source_embeddings is None or candidate_embeddings is None:
        raise ValueError("Both source and candidate need input embeddings for hidden KD projection.")
    with torch.no_grad():
        source_rows = source_embeddings.weight.index_select(0, source_anchor_idx).detach().to(torch.float32)
        candidate_rows = candidate_embeddings.weight.index_select(0, candidate_anchor_idx).detach().to(torch.float32)
    projection = _ridge_solve_features(torch, source_rows, candidate_rows, ridge)
    predicted = source_rows.matmul(projection)
    return {
        "map": projection.detach(),
        "anchor_count": int(source_rows.shape[0]),
        "available_anchor_count": available,
        "source_hidden_size": int(source_rows.shape[1]),
        "candidate_hidden_size": int(candidate_rows.shape[1]),
        "ridge": float(ridge),
        "relative_error": _relative_reconstruction_error(torch, predicted, candidate_rows),
        "mean_cosine": _mean_row_cosine(torch, predicted, candidate_rows),
    }


def _fit_cross_tokenizer_layer_input_hidden_projection(
    torch,
    source,
    candidate,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    source_layer_group: Sequence[int],
    candidate_layer_idx: int,
    canonical_fn,
    ridge: float,
    max_tokens: int,
    baseline_projection=None,
    min_relative_improvement: float = 0.0,
    max_cosine_drop: float = 0.01,
    min_fitted_cosine: float = -1.0,
) -> dict[str, Any]:
    if not source_layer_group:
        return {"skipped": True, "reason": "missing_source_layer_group"}
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    source_input_idx = max(0, int(min(source_layer_group)))
    candidate_input_idx = max(0, int(candidate_layer_idx))
    source_rows = []
    candidate_rows = []
    token_count = 0
    was_source_training = source.training
    was_candidate_training = candidate.training
    source.eval()
    candidate.eval()
    with torch.no_grad():
        try:
            for source_batch, candidate_batch in paired_batches:
                if token_count >= int(max_tokens):
                    break
                teacher = source(
                    **_model_forward_inputs(source_batch),
                    use_cache=False,
                    output_hidden_states=True,
                )
                student = candidate(
                    **_model_forward_inputs(candidate_batch),
                    use_cache=False,
                    output_hidden_states=True,
                )
                if teacher.hidden_states is None or student.hidden_states is None:
                    continue
                if source_input_idx >= len(teacher.hidden_states) or candidate_input_idx >= len(student.hidden_states):
                    return {
                        "skipped": True,
                        "reason": "layer_input_hidden_state_out_of_range",
                        "source_input_layer": int(source_input_idx),
                        "candidate_input_layer": int(candidate_input_idx),
                    }
                pairs = _cross_tokenizer_span_pairs(
                    torch,
                    source_tokenizer,
                    candidate_tokenizer,
                    source_batch,
                    candidate_batch,
                    canonical_fn,
                    max_pairs=int(max_tokens) - token_count,
                    shift_positions=False,
                )
                candidate_hidden = student.hidden_states[candidate_input_idx]
                source_hidden = teacher.hidden_states[source_input_idx]
                batch_candidate, batch_source, rows = _span_aligned_hidden_rows(
                    torch,
                    candidate_hidden,
                    source_hidden,
                    pairs,
                )
                if int(rows) <= 0:
                    continue
                remaining = int(max_tokens) - token_count
                take = min(int(rows), remaining)
                candidate_rows.append(batch_candidate[:take].detach().cpu())
                source_rows.append(batch_source[:take].detach().cpu())
                token_count += take
        finally:
            if was_source_training:
                source.train()
            if was_candidate_training:
                candidate.train()
    if not source_rows or not candidate_rows:
        return {
            "skipped": True,
            "reason": "no_cross_tokenizer_layer_input_pairs",
            "source_input_layer": int(source_input_idx),
            "candidate_input_layer": int(candidate_input_idx),
        }
    source_matrix = torch.cat(source_rows, dim=0).to(torch.float32)[: int(max_tokens)]
    candidate_matrix = torch.cat(candidate_rows, dim=0).to(torch.float32)[: int(max_tokens)]
    rows = min(int(source_matrix.shape[0]), int(candidate_matrix.shape[0]))
    if rows <= 1:
        return {
            "skipped": True,
            "reason": "insufficient_cross_tokenizer_layer_input_pairs",
            "pairs": int(rows),
            "source_input_layer": int(source_input_idx),
            "candidate_input_layer": int(candidate_input_idx),
        }
    source_matrix = source_matrix[:rows]
    candidate_matrix = candidate_matrix[:rows]
    projection = _ridge_solve_features(torch, source_matrix, candidate_matrix, ridge)
    predicted = source_matrix.matmul(projection)
    report = {
        "map": projection.detach(),
        "skipped": False,
        "scope": "layer-input",
        "source_input_layer": int(source_input_idx),
        "candidate_input_layer": int(candidate_input_idx),
        "source_layer_group": [int(item) for item in source_layer_group],
        "pairs": int(rows),
        "source_hidden_size": int(source_matrix.shape[1]),
        "candidate_hidden_size": int(candidate_matrix.shape[1]),
        "ridge": float(ridge),
        "relative_error": _relative_reconstruction_error(torch, predicted, candidate_matrix),
        "mean_cosine": _mean_row_cosine(torch, predicted, candidate_matrix),
    }
    if baseline_projection is not None:
        gate = _hidden_projection_gate_report(
            torch,
            source_matrix,
            candidate_matrix,
            projection,
            baseline_projection,
            min_relative_improvement=min_relative_improvement,
            max_cosine_drop=max_cosine_drop,
            min_fitted_cosine=min_fitted_cosine,
        )
        report["gate"] = gate
        if not bool(gate["accepted"]):
            return {
                key: value
                for key, value in report.items()
                if key != "map"
            } | {
                "skipped": True,
                "reason": "layer_input_projection_gate_rejected",
            }
    return report


def _hidden_projection_gate_report(
    torch,
    source_matrix,
    candidate_matrix,
    fitted_projection,
    baseline_projection,
    min_relative_improvement: float = 0.0,
    max_cosine_drop: float = 0.01,
    min_fitted_cosine: float = -1.0,
) -> dict[str, Any]:
    if min_relative_improvement < 0:
        raise ValueError("min_relative_improvement must be non-negative.")
    if max_cosine_drop < 0:
        raise ValueError("max_cosine_drop must be non-negative.")
    if not -1.0 <= float(min_fitted_cosine) <= 1.0:
        raise ValueError("min_fitted_cosine must be between -1.0 and 1.0.")
    source_rows = source_matrix.detach().to(torch.float32)
    candidate_rows = candidate_matrix.detach().to(device=source_rows.device, dtype=torch.float32)
    fitted = fitted_projection.detach().to(device=source_rows.device, dtype=torch.float32)
    baseline = baseline_projection.detach().to(device=source_rows.device, dtype=torch.float32)
    fitted_pred = source_rows.matmul(fitted)
    baseline_pred = source_rows.matmul(baseline)
    fitted_error = _relative_reconstruction_error(torch, fitted_pred, candidate_rows)
    baseline_error = _relative_reconstruction_error(torch, baseline_pred, candidate_rows)
    fitted_cosine = _mean_row_cosine(torch, fitted_pred, candidate_rows)
    baseline_cosine = _mean_row_cosine(torch, baseline_pred, candidate_rows)
    denominator = max(abs(float(baseline_error)), 1e-12)
    relative_improvement = (float(baseline_error) - float(fitted_error)) / denominator
    cosine_drop = float(baseline_cosine) - float(fitted_cosine)
    accepted = (
        math.isfinite(relative_improvement)
        and math.isfinite(cosine_drop)
        and relative_improvement >= float(min_relative_improvement)
        and cosine_drop <= float(max_cosine_drop)
        and float(fitted_cosine) >= float(min_fitted_cosine)
    )
    return {
        "mode": "global_baseline",
        "accepted": bool(accepted),
        "min_relative_improvement": float(min_relative_improvement),
        "max_cosine_drop": float(max_cosine_drop),
        "min_fitted_cosine": float(min_fitted_cosine),
        "relative_improvement": float(relative_improvement),
        "cosine_drop": float(cosine_drop),
        "fitted_relative_error": float(fitted_error),
        "baseline_relative_error": float(baseline_error),
        "fitted_mean_cosine": float(fitted_cosine),
        "baseline_mean_cosine": float(baseline_cosine),
    }


def _load_layer_merge_report(candidate_model: str, layer_merge_report: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(layer_merge_report, dict):
        return layer_merge_report
    if layer_merge_report is None:
        path = Path(candidate_model) / "neural_forge_layer_merge_report.json"
    else:
        path = Path(layer_merge_report)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _fold_groups_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("operation") != "merge_decoder_layers":
        raise ValueError("Layer merge report must come from merge_decoder_layers.")
    if report.get("method") != "fold":
        return []
    groups = []
    folded_indices = sorted({int(idx) for item in report.get("groups", []) for idx in item.get("folded", [])})
    for item in report.get("groups", []):
        group = [int(idx) for idx in item.get("group", [])]
        folded = [int(idx) for idx in item.get("folded", [])]
        survivor = int(item.get("survivor", group[0] if group else -1))
        if len(group) < 2 or not folded:
            continue
        groups.append(
            {
                "group": group,
                "survivor": survivor,
                "folded": folded,
                "all_folded_indices": folded_indices,
            }
        )
    return groups


def _candidate_index_after_depth_fold(original_index: int, folded_indices: Sequence[int]) -> int:
    return int(original_index) - sum(1 for idx in folded_indices if int(idx) < int(original_index))


def _reconstruct_fold_group(
    torch,
    source,
    candidate,
    source_layers: Sequence[Any],
    candidate_layers: Sequence[Any],
    group: dict[str, Any],
    batches: Sequence[dict[str, Any]],
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_target: str,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
) -> dict[str, Any]:
    source_consumer_idx = max(group["group"]) + 1
    candidate_consumer_idx = _candidate_index_after_depth_fold(source_consumer_idx, group["all_folded_indices"])
    base_report = {
        "group": group["group"],
        "survivor": group["survivor"],
        "folded": group["folded"],
        "source_consumer_layer": source_consumer_idx,
        "candidate_consumer_layer": candidate_consumer_idx,
    }
    if source_consumer_idx >= len(source_layers) or candidate_consumer_idx >= len(candidate_layers):
        return {**base_report, "skipped": True, "reason": "no_downstream_consumer_layer"}

    source_layer = source_layers[source_consumer_idx]
    candidate_layer = candidate_layers[candidate_consumer_idx]
    repair_report: dict[str, Any] = {**base_report, "skipped": False}

    if reconstruction_target in {"survivor-mlp", "both"}:
        source_survivor_idx = max(group["group"])
        candidate_survivor_idx = _candidate_index_after_depth_fold(group["survivor"], group["all_folded_indices"])
        if source_survivor_idx < len(source_layers) and candidate_survivor_idx < len(candidate_layers):
            repair_report["survivor_mlp_reconstruction"] = _reconstruct_survivor_mlp_output(
                torch,
                source,
                candidate,
                source_layers[source_survivor_idx],
                candidate_layers[candidate_survivor_idx],
                batches,
                ridge=ridge,
                reconstruction_alpha=reconstruction_alpha,
                reconstruction_objective=reconstruction_objective,
                cosine_steps=cosine_steps,
                cosine_lr=cosine_lr,
                cosine_l2=cosine_l2,
                max_tokens_per_layer=max_tokens_per_layer,
            )
        else:
            repair_report["survivor_mlp_reconstruction"] = {
                "skipped": True,
                "reason": "survivor_layer_out_of_range",
            }

    if reconstruction_target == "survivor-mlp":
        return repair_report

    source_attn_norm = _get_layer_module(source_layer, ("input_layernorm", "ln_1"))
    candidate_attn_norm = _get_layer_module(candidate_layer, ("input_layernorm", "ln_1"))
    attn_linears = _attention_input_linears(candidate_layer)
    if source_attn_norm is not None and candidate_attn_norm is not None and attn_linears:
        source_features = _collect_module_activations(torch, source, source_attn_norm, batches, max_tokens_per_layer)
        candidate_features = _collect_module_activations(torch, candidate, candidate_attn_norm, batches, max_tokens_per_layer)
        fitted = _fit_activation_map(
            torch,
            candidate_features,
            source_features,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
        )
        _absorb_input_map_into_linears(torch, attn_linears, fitted["map"])
        repair_report["attention_reconstruction"] = {
            key: value for key, value in fitted.items() if key != "map"
        } | {"linears_updated": [name for name, _module in attn_linears]}
    else:
        repair_report["attention_reconstruction"] = {
            "skipped": True,
            "reason": "missing_layernorm_or_attention_input_linears",
        }

    source_mlp_norm = _get_layer_module(source_layer, ("post_attention_layernorm", "ln_2"))
    candidate_mlp_norm = _get_layer_module(candidate_layer, ("post_attention_layernorm", "ln_2"))
    mlp_linears = _mlp_input_linears(candidate_layer)
    if source_mlp_norm is not None and candidate_mlp_norm is not None and mlp_linears:
        source_features = _collect_module_activations(torch, source, source_mlp_norm, batches, max_tokens_per_layer)
        candidate_features = _collect_module_activations(torch, candidate, candidate_mlp_norm, batches, max_tokens_per_layer)
        fitted = _fit_activation_map(
            torch,
            candidate_features,
            source_features,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
        )
        _absorb_input_map_into_linears(torch, mlp_linears, fitted["map"])
        repair_report["mlp_reconstruction"] = {
            key: value for key, value in fitted.items() if key != "map"
        } | {"linears_updated": [name for name, _module in mlp_linears]}
    else:
        repair_report["mlp_reconstruction"] = {
            "skipped": True,
            "reason": "missing_layernorm_or_mlp_input_linears",
        }

    return repair_report


def _get_layer_module(layer, names: Sequence[str]):
    for name in names:
        module = getattr(layer, name, None)
        if module is not None:
            return module
    return None


def _attention_input_linears(layer) -> list[tuple[str, Any]]:
    attention = _attention_module(layer)
    if attention is None:
        return []
    linears = []
    for name in ("q_proj", "k_proj", "v_proj", "query_key_value", "c_attn"):
        module = getattr(attention, name, None)
        if _is_linear_like(module):
            linears.append((f"self_attn.{name}", module))
    return linears


def _mlp_input_linears(layer) -> list[tuple[str, Any]]:
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    if mlp is None:
        return []
    linears = []
    for name in ("gate_proj", "up_proj", "w1", "w3", "dense_h_to_4h", "fc1"):
        module = getattr(mlp, name, None)
        if _is_linear_like(module):
            linears.append((f"mlp.{name}", module))
    return linears


def _reconstruct_survivor_mlp_output(
    torch,
    source,
    candidate,
    source_last_layer,
    candidate_survivor_layer,
    batches: Sequence[dict[str, Any]],
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
) -> dict[str, Any]:
    spec = _mlp_projection_spec(candidate_survivor_layer)
    if spec is None:
        return {"skipped": True, "reason": "unsupported_mlp_shape"}

    source_output = _collect_layer_output_activations(torch, source, source_last_layer, batches, max_tokens_per_layer)
    hidden, residual, candidate_output = _collect_survivor_mlp_calibration(
        torch,
        candidate,
        candidate_survivor_layer,
        batches,
        max_tokens_per_layer,
    )
    rows = min(int(source_output.shape[0]), int(hidden.shape[0]), int(residual.shape[0]))
    source_rows = source_output[:rows].to(torch.float32)
    hidden_rows = hidden[:rows].to(torch.float32)
    residual_rows = residual[:rows].to(torch.float32)
    candidate_rows = candidate_output[:rows].to(torch.float32)
    fit_source_rows = source_rows
    fit_hidden_rows = hidden_rows
    fit_residual_rows = residual_rows
    validation_source_rows = None
    validation_hidden_rows = None
    validation_residual_rows = None
    validation_candidate_rows = None
    fit_report: dict[str, Any] = {}
    if fit_gate == "validation":
        split = _split_reconstruction_rows(
            [source_rows, hidden_rows, residual_rows, candidate_rows],
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
        )
        if split is None:
            fit_report["fit_gate"] = {"mode": "validation", "accepted": True, "reason": "insufficient_rows"}
        else:
            fit_source_rows, fit_hidden_rows, fit_residual_rows, _fit_candidate_rows = split["train"]
            (
                validation_source_rows,
                validation_hidden_rows,
                validation_residual_rows,
                validation_candidate_rows,
            ) = split["validation"]
            fit_report["fit_gate"] = {
                "mode": "validation",
                "rows": int(split["rows"]),
                "train_rows": int(split["train_rows"]),
                "validation_rows": int(split["validation_rows"]),
            }
    target_mlp_output = fit_source_rows - fit_residual_rows
    solved_report = _solve_linear_reconstruction(
        torch,
        fit_hidden_rows,
        target_mlp_output,
        ridge,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
    )
    solved = solved_report["map"]
    old_weight = spec["output"].weight.data.detach().cpu().to(torch.float32)
    solved_weight = solved.T.contiguous()
    if tuple(old_weight.shape) != tuple(solved_weight.shape):
        return {
            "skipped": True,
            "reason": "solved_weight_shape_mismatch",
            "old_weight_shape": list(old_weight.shape),
            "solved_weight_shape": list(solved_weight.shape),
        }
    blended_weight = (1.0 - float(reconstruction_alpha)) * old_weight + float(reconstruction_alpha) * solved_weight
    before = _relative_reconstruction_error(torch, candidate_rows, source_rows)
    solved_output = residual_rows + hidden_rows.matmul(solved)
    solved_error = _relative_reconstruction_error(torch, solved_output, source_rows)
    blended_output = residual_rows + hidden_rows.matmul(blended_weight.T)
    after = _relative_reconstruction_error(torch, blended_output, source_rows)
    before_cosine = _mean_row_cosine(torch, candidate_rows, source_rows)
    solved_cosine = _mean_row_cosine(torch, solved_output, source_rows)
    after_cosine = _mean_row_cosine(torch, blended_output, source_rows)
    if (
        validation_source_rows is not None
        and validation_hidden_rows is not None
        and validation_residual_rows is not None
        and validation_candidate_rows is not None
    ):
        gate = _validation_reconstruction_gate(
            torch,
            validation_candidate_rows,
            validation_residual_rows + validation_hidden_rows.matmul(blended_weight.T),
            validation_source_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
        )
        fit_report["fit_gate"] = {**fit_report["fit_gate"], **gate}
        if not gate["accepted"]:
            return {
                "skipped": False,
                "matched_pairs": int(rows),
                "tokens": rows,
                "feature_size": int(hidden_rows.shape[1]),
                "hidden_size": int(source_rows.shape[1]),
                "output_projection": spec["output_name"],
                "reconstruction_alpha": float(reconstruction_alpha),
                "reconstruction_objective": reconstruction_objective,
                "relative_error_before": before,
                "relative_error_solved": solved_error,
                "relative_error_after": after,
                "mean_cosine_before": before_cosine,
                "mean_cosine_solved": solved_cosine,
                "mean_cosine_after": after_cosine,
                "solver": {key: value for key, value in solved_report.items() if key != "map"},
                "mutation_skipped": True,
                **fit_report,
            }
    spec["output"].weight.data.copy_(blended_weight.to(device=spec["output"].weight.device, dtype=spec["output"].weight.dtype))
    return {
        "skipped": False,
        "tokens": rows,
        "feature_size": int(hidden_rows.shape[1]),
        "hidden_size": int(source_rows.shape[1]),
        "output_projection": spec["output_name"],
        "reconstruction_alpha": float(reconstruction_alpha),
        "reconstruction_objective": reconstruction_objective,
        "cosine_steps": int(cosine_steps),
        "cosine_lr": float(cosine_lr),
        "cosine_l2": float(cosine_l2),
        "relative_error_before": before,
        "relative_error_solved": solved_error,
        "relative_error_after": after,
        "mean_cosine_before": before_cosine,
        "mean_cosine_solved": solved_cosine,
        "mean_cosine_after": after_cosine,
        "solver": {key: value for key, value in solved_report.items() if key != "map"},
        **fit_report,
    }


def _proportional_layer_index(index: int, target_count: int, source_count: int) -> int:
    if target_count <= 0 or source_count <= 0:
        raise ValueError("layer counts must be positive.")
    return max(0, min(int(math.floor((int(index) + 0.5) * source_count / target_count)), source_count - 1))


def _source_layer_delta_scores_from_batches(
    torch,
    source_model,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    source_layer_count: int,
) -> list[float]:
    if source_layer_count <= 0:
        raise ValueError("source_layer_count must be positive.")
    if not paired_batches:
        raise ValueError("Need at least one paired batch for source layer delta scoring.")
    sums = [0.0 for _ in range(int(source_layer_count))]
    counts = [0 for _ in range(int(source_layer_count))]
    was_training = source_model.training
    source_model.eval()
    with torch.no_grad():
        for source_batch, _candidate_batch in paired_batches:
            outputs = source_model(
                **_model_forward_inputs(source_batch),
                use_cache=False,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states or [])
            usable = min(int(source_layer_count), max(0, len(hidden_states) - 1))
            for layer_idx in range(usable):
                before = hidden_states[layer_idx].detach().to(torch.float32)
                after = hidden_states[layer_idx + 1].detach().to(torch.float32)
                batch = min(int(before.shape[0]), int(after.shape[0]))
                seq = min(int(before.shape[1]), int(after.shape[1])) if before.dim() > 1 and after.dim() > 1 else 0
                dims = min(int(before.shape[-1]), int(after.shape[-1]))
                if batch <= 0 or seq <= 0 or dims <= 0:
                    continue
                delta = after[:batch, :seq, :dims] - before[:batch, :seq, :dims]
                sums[layer_idx] += float(delta.pow(2).mean().sqrt().detach().cpu())
                counts[layer_idx] += 1
    if was_training:
        source_model.train()
    scores = [
        (sums[idx] / counts[idx]) if counts[idx] > 0 else 1.0
        for idx in range(int(source_layer_count))
    ]
    if not any(score > 1e-8 for score in scores):
        return [1.0 for _ in scores]
    return [max(float(score), 1e-12) for score in scores]


def _source_layer_geometry_scores_from_batches(
    torch,
    source_model,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    source_layer_count: int,
    max_rows_per_layer: int = 512,
) -> dict[str, Any]:
    if source_layer_count <= 0:
        raise ValueError("source_layer_count must be positive.")
    if max_rows_per_layer <= 0:
        raise ValueError("max_rows_per_layer must be positive.")
    if not paired_batches:
        raise ValueError("Need at least one paired batch for source layer geometry scoring.")

    from neural_forge.hf_xfamily import _activation_layer_geometry_scores

    input_parts: list[list[Any]] = [[] for _ in range(int(source_layer_count))]
    output_parts: list[list[Any]] = [[] for _ in range(int(source_layer_count))]
    row_counts = [0 for _ in range(int(source_layer_count))]
    was_training = source_model.training
    source_model.eval()
    with torch.no_grad():
        for source_batch, _candidate_batch in paired_batches:
            if all(count >= int(max_rows_per_layer) for count in row_counts):
                break
            outputs = source_model(
                **_model_forward_inputs(source_batch),
                use_cache=False,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states or [])
            usable = min(int(source_layer_count), max(0, len(hidden_states) - 1))
            for layer_idx in range(usable):
                needed = int(max_rows_per_layer) - int(row_counts[layer_idx])
                if needed <= 0:
                    continue
                before = hidden_states[layer_idx].detach().to(torch.float32)
                after = hidden_states[layer_idx + 1].detach().to(torch.float32)
                batch = min(int(before.shape[0]), int(after.shape[0]))
                seq = min(int(before.shape[1]), int(after.shape[1])) if before.dim() > 1 and after.dim() > 1 else 0
                dims = min(int(before.shape[-1]), int(after.shape[-1]))
                if batch <= 0 or seq <= 0 or dims <= 0:
                    continue
                before_rows = before[:batch, :seq, :dims].reshape(-1, dims)
                after_rows = after[:batch, :seq, :dims].reshape(-1, dims)
                take = min(needed, int(before_rows.shape[0]), int(after_rows.shape[0]))
                if take <= 0:
                    continue
                input_parts[layer_idx].append(before_rows[:take].detach().cpu())
                output_parts[layer_idx].append(after_rows[:take].detach().cpu())
                row_counts[layer_idx] += int(take)
    if was_training:
        source_model.train()
    input_rows_by_layer = {
        int(idx): torch.cat(parts, dim=0)
        for idx, parts in enumerate(input_parts)
        if parts
    }
    output_rows_by_layer = {
        int(idx): torch.cat(parts, dim=0)
        for idx, parts in enumerate(output_parts)
        if parts
    }
    report = _activation_layer_geometry_scores(
        torch,
        input_rows_by_layer,
        output_rows_by_layer,
        int(source_layer_count),
    )
    report["rows_per_layer"] = [int(item) for item in row_counts]
    report["max_rows_per_layer"] = int(max_rows_per_layer)
    return report


def _source_layer_cka_groups_from_batches(
    torch,
    source_model,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    source_layer_count: int,
    target_layer_count: int,
    max_rows_per_layer: int = 512,
) -> dict[str, Any]:
    if source_layer_count <= 0 or target_layer_count <= 0:
        raise ValueError("source_layer_count and target_layer_count must be positive.")
    if max_rows_per_layer <= 0:
        raise ValueError("max_rows_per_layer must be positive.")
    if not paired_batches:
        raise ValueError("Need at least one paired batch for source layer CKA grouping.")

    from neural_forge.hf_xfamily import _activation_layer_cka_groups

    input_parts: list[list[Any]] = [[] for _ in range(int(source_layer_count))]
    output_parts: list[list[Any]] = [[] for _ in range(int(source_layer_count))]
    row_counts = [0 for _ in range(int(source_layer_count))]
    was_training = source_model.training
    source_model.eval()
    with torch.no_grad():
        for source_batch, _candidate_batch in paired_batches:
            if all(count >= int(max_rows_per_layer) for count in row_counts):
                break
            outputs = source_model(
                **_model_forward_inputs(source_batch),
                use_cache=False,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states or [])
            usable = min(int(source_layer_count), max(0, len(hidden_states) - 1))
            for layer_idx in range(usable):
                needed = int(max_rows_per_layer) - int(row_counts[layer_idx])
                if needed <= 0:
                    continue
                before = hidden_states[layer_idx].detach().to(torch.float32)
                after = hidden_states[layer_idx + 1].detach().to(torch.float32)
                batch = min(int(before.shape[0]), int(after.shape[0]))
                seq = min(int(before.shape[1]), int(after.shape[1])) if before.dim() > 1 and after.dim() > 1 else 0
                dims = min(int(before.shape[-1]), int(after.shape[-1]))
                if batch <= 0 or seq <= 0 or dims <= 0:
                    continue
                before_rows = before[:batch, :seq, :dims].reshape(-1, dims)
                after_rows = after[:batch, :seq, :dims].reshape(-1, dims)
                take = min(needed, int(before_rows.shape[0]), int(after_rows.shape[0]))
                if take <= 0:
                    continue
                input_parts[layer_idx].append(before_rows[:take].detach().cpu())
                output_parts[layer_idx].append(after_rows[:take].detach().cpu())
                row_counts[layer_idx] += int(take)
    if was_training:
        source_model.train()
    input_rows_by_layer = {
        int(idx): torch.cat(parts, dim=0)
        for idx, parts in enumerate(input_parts)
        if parts
    }
    output_rows_by_layer = {
        int(idx): torch.cat(parts, dim=0)
        for idx, parts in enumerate(output_parts)
        if parts
    }
    report = _activation_layer_cka_groups(
        torch,
        input_rows_by_layer,
        output_rows_by_layer,
        layer_count=int(source_layer_count),
        target_layer_count=int(target_layer_count),
    )
    report["rows_per_layer"] = [int(item) for item in row_counts]
    report["max_rows_per_layer"] = int(max_rows_per_layer)
    return report


def _layer_gate_decision(previous_ppl: float, candidate_ppl: float, max_ppl_ratio: float) -> dict[str, Any]:
    if previous_ppl <= 0:
        raise ValueError("previous_ppl must be positive.")
    if max_ppl_ratio <= 0:
        raise ValueError("max_ppl_ratio must be positive.")
    ratio = float(candidate_ppl) / max(float(previous_ppl), 1e-12)
    return {
        "previous_ppl": float(previous_ppl),
        "candidate_ppl": float(candidate_ppl),
        "ppl_ratio": ratio,
        "max_ppl_ratio": float(max_ppl_ratio),
        "accepted": ratio <= float(max_ppl_ratio),
    }


def _layer_gate_snapshot_indices(candidate_layers: Sequence[Any], candidate_idx: int, reconstruction_target: str) -> list[int]:
    if candidate_idx < 0 or candidate_idx >= len(candidate_layers):
        raise ValueError("candidate_idx is outside candidate layer range.")
    indices = {int(candidate_idx)}
    if reconstruction_target in {"consumer-input", "block-interface", "all"} and int(candidate_idx) + 1 < len(candidate_layers):
        indices.add(int(candidate_idx) + 1)
    return sorted(indices)


def _snapshot_layer_gate_states(
    candidate_layers: Sequence[Any],
    candidate_idx: int,
    reconstruction_target: str,
) -> dict[int, dict[str, Any]]:
    return {
        int(layer_idx): _snapshot_module_state(candidate_layers[int(layer_idx)])
        for layer_idx in _layer_gate_snapshot_indices(candidate_layers, candidate_idx, reconstruction_target)
    }


def _restore_layer_gate_states(candidate_layers: Sequence[Any], states: dict[int, dict[str, Any]]) -> None:
    for layer_idx, state in states.items():
        _restore_module_state(candidate_layers[int(layer_idx)], state)


def _output_head_tied_to_input(output_head, input_embeddings) -> bool:
    output_head_weight = getattr(output_head, "weight", None)
    input_embedding_weight = getattr(input_embeddings, "weight", None)
    return bool(
        output_head_weight is not None
        and input_embedding_weight is not None
        and (
            output_head_weight is input_embedding_weight
            or output_head_weight.data_ptr() == input_embedding_weight.data_ptr()
        )
    )


def _block_recovery_snapshot_modules(
    candidate_layers: Sequence[Any],
    candidate_idx: int,
    candidate_layer,
    output_head,
    pre_reconstruction_target: str,
) -> dict[str, Any]:
    modules: dict[str, Any] = {"block": candidate_layer}
    if output_head is not None:
        modules["output_head"] = output_head
    if (
        pre_reconstruction_target in {"consumer-input", "block-interface", "all"}
        and int(candidate_idx) + 1 < len(candidate_layers)
    ):
        modules["consumer_block"] = candidate_layers[int(candidate_idx) + 1]
    return modules


def _snapshot_module_state(module) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _restore_module_state(module, state: dict[str, Any]) -> None:
    module.load_state_dict(state, strict=True)


def _snapshot_named_module_states(modules: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _snapshot_module_state(module)
        for name, module in modules.items()
        if module is not None
    }


def _restore_named_module_states(modules: dict[str, Any], states: dict[str, dict[str, Any]]) -> None:
    for name, state in states.items():
        module = modules.get(name)
        if module is None:
            raise ValueError(f"Cannot restore missing module state {name!r}.")
        _restore_module_state(module, state)


def _heldout_ppl_mean(torch, model, tokenizer, texts: Sequence[str], device: str, max_length: int) -> float:
    if not texts:
        raise ValueError("Need at least one held-out text for the layer gate.")
    losses = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        outputs = model(
            input_ids=input_ids,
            attention_mask=None if attention_mask is None else attention_mask.to(device),
            labels=input_ids,
            use_cache=False,
        )
        losses.append(float(outputs.loss.detach().cpu()))
    mean_loss = sum(losses) / max(1, len(losses))
    return float(math.exp(min(20.0, mean_loss)))


def _heldout_behavior_summary(
    torch,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    max_probes: int,
    behavior_probes: Sequence[dict[str, Any]] | None = None,
    generation_prompts: Sequence[str | dict[str, Any]] | None = None,
    max_generation_prompts: int | None = None,
) -> dict[str, Any]:
    probes = list((behavior_probes if behavior_probes is not None else BEHAVIOR_PROBES)[: int(max_probes)])
    if not probes:
        raise ValueError("Need at least one held-out behavior probe for the layer gate.")
    generation_source = list(generation_prompts if generation_prompts is not None else GENERATION_PROMPTS)
    if max_generation_prompts is not None:
        generation_source = generation_source[: int(max_generation_prompts)]
    probe_generations = _generate_text_batch(
        torch,
        model,
        tokenizer,
        [str(probe["prompt"]) for probe in probes],
        device,
        int(max_new_tokens),
    )
    items = [
        _score_behavior_probe(probe, generation)
        for probe, generation in zip(probes, probe_generations, strict=True)
    ]

    generation_prompt_texts = [
        str(prompt.get("prompt", "")) if isinstance(prompt, dict) else str(prompt)
        for prompt in generation_source
    ]
    generated_prompts = _generate_text_batch(
        torch,
        model,
        tokenizer,
        generation_prompt_texts,
        device,
        int(max_new_tokens),
    )

    def _generation_from_prompt(prompt, generated):
        if isinstance(prompt, dict):
            generated = dict(generated)
            generated["id"] = prompt.get("id")
            if prompt.get("category") is not None:
                generated["category"] = prompt.get("category")
            if prompt.get("relevance_rules") is not None:
                generated["relevance_rules"] = prompt.get("relevance_rules")
            for key in (
                "kind",
                "answer",
                "function_name",
                "tests",
                "pattern_groups",
                "expected_prefix",
                "expected_prefix_generated_token_count",
                "require_generation_health",
            ):
                if key in prompt:
                    generated[key] = prompt[key]
            return generated
        return generated

    generation_items = [
        _generation_health(
            _generation_from_prompt(prompt, generated),
            min_continuation_chars=24,
            min_unique_token_ratio=0.20,
        )
        for prompt, generated in zip(generation_source, generated_prompts, strict=True)
    ]
    correct = sum(1 for item in items if item["passed"])
    generation_correct = sum(1 for item in generation_items if item["passed"])
    behavior_breakdown = _passed_category_breakdown(items, "behavior")
    generation_breakdown = _passed_category_breakdown(generation_items, "generation")
    return {
        "correct": int(correct),
        "total": int(len(items)),
        "pass_rate": float(correct / max(1, len(items))),
        "generation_health_correct": int(generation_correct),
        "generation_health_total": int(len(generation_items)),
        "generation_health_rate": float(generation_correct / max(1, len(generation_items))),
        "behavior_category_breakdown": behavior_breakdown,
        "generation_category_breakdown": generation_breakdown,
        "behavior_category_summary": _category_breakdown_summary(behavior_breakdown),
        "generation_category_summary": _category_breakdown_summary(generation_breakdown),
        "items": [
            {
                "id": item["id"],
                "category": item.get("category"),
                "passed": bool(item["passed"]),
                "continuation": item.get("continuation", ""),
            }
            for item in items
        ],
        "generation_items": [
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "prompt": item["prompt"],
                "passed": bool(item["passed"]),
                "continuation_chars": int(item.get("continuation_chars", 0)),
                "unique_token_ratio": float(item.get("unique_token_ratio", 0.0)),
                "relevant_enough": bool(item.get("relevant_enough", False)),
            }
            for item in generation_items
        ],
    }


def _behavior_history_entry(step: int, loss_value: float | None, behavior_summary: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "step": int(step),
        "loss": None if loss_value is None else float(loss_value),
        "correct": int(behavior_summary["correct"]),
        "total": int(behavior_summary["total"]),
        "pass_rate": float(behavior_summary["pass_rate"]),
        "generation_health_correct": int(behavior_summary.get("generation_health_correct", 0)),
        "generation_health_total": int(behavior_summary.get("generation_health_total", 0)),
        "generation_health_rate": float(behavior_summary.get("generation_health_rate", 0.0)),
        "behavior_category_breakdown": behavior_summary.get("behavior_category_breakdown", {}),
        "generation_category_breakdown": behavior_summary.get("generation_category_breakdown", {}),
        "behavior_category_summary": behavior_summary.get("behavior_category_summary", {}),
        "generation_category_summary": behavior_summary.get("generation_category_summary", {}),
        "passed_ids": [str(item["id"]) for item in behavior_summary.get("items", []) if bool(item.get("passed"))],
        "generation_passed_prompts": [
            str(item["prompt"]) for item in behavior_summary.get("generation_items", []) if bool(item.get("passed"))
        ],
    }
    return entry


def _print_behavior_checkpoint(prefix: str, behavior_entry: dict[str, Any]) -> None:
    loss = behavior_entry.get("loss")
    loss_text = "nan" if loss is None else f"{float(loss):.6f}"
    print(
        f"{prefix} step={behavior_entry['step']} "
        f"correct={behavior_entry['correct']}/{behavior_entry['total']} "
        f"generation_health={behavior_entry['generation_health_correct']}/"
        f"{behavior_entry['generation_health_total']} "
        f"pass_rate={behavior_entry['pass_rate']:.4f} "
        f"loss={loss_text}",
        flush=True,
    )


def _behavior_checkpoint_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    incumbent_loss: float | None,
) -> bool:
    if incumbent is None:
        return True
    candidate_correct = int(candidate.get("correct", 0))
    incumbent_correct = int(incumbent.get("correct", 0))
    if candidate_correct != incumbent_correct:
        return candidate_correct > incumbent_correct
    candidate_rate = float(candidate.get("pass_rate", 0.0))
    incumbent_rate = float(incumbent.get("pass_rate", 0.0))
    if abs(candidate_rate - incumbent_rate) > 1e-12:
        return candidate_rate > incumbent_rate
    candidate_behavior_categories = _category_summary_for_behavior_entry(candidate, "behavior")
    incumbent_behavior_categories = _category_summary_for_behavior_entry(incumbent, "behavior")
    candidate_hard_correct = int(candidate_behavior_categories.get("hard_category_correct", 0))
    incumbent_hard_correct = int(incumbent_behavior_categories.get("hard_category_correct", 0))
    if candidate_hard_correct != incumbent_hard_correct:
        return candidate_hard_correct > incumbent_hard_correct
    candidate_min_category = float(candidate_behavior_categories.get("min_pass_rate", 0.0))
    incumbent_min_category = float(incumbent_behavior_categories.get("min_pass_rate", 0.0))
    if abs(candidate_min_category - incumbent_min_category) > 1e-12:
        return candidate_min_category > incumbent_min_category
    candidate_generation_correct = int(candidate.get("generation_health_correct", 0))
    incumbent_generation_correct = int(incumbent.get("generation_health_correct", 0))
    if candidate_generation_correct != incumbent_generation_correct:
        return candidate_generation_correct > incumbent_generation_correct
    candidate_generation_rate = float(candidate.get("generation_health_rate", 0.0))
    incumbent_generation_rate = float(incumbent.get("generation_health_rate", 0.0))
    if abs(candidate_generation_rate - incumbent_generation_rate) > 1e-12:
        return candidate_generation_rate > incumbent_generation_rate
    candidate_generation_categories = _category_summary_for_behavior_entry(candidate, "generation")
    incumbent_generation_categories = _category_summary_for_behavior_entry(incumbent, "generation")
    candidate_generation_hard = int(candidate_generation_categories.get("hard_category_correct", 0))
    incumbent_generation_hard = int(incumbent_generation_categories.get("hard_category_correct", 0))
    if candidate_generation_hard != incumbent_generation_hard:
        return candidate_generation_hard > incumbent_generation_hard
    candidate_generation_min = float(candidate_generation_categories.get("min_pass_rate", 0.0))
    incumbent_generation_min = float(incumbent_generation_categories.get("min_pass_rate", 0.0))
    if abs(candidate_generation_min - incumbent_generation_min) > 1e-12:
        return candidate_generation_min > incumbent_generation_min
    candidate_loss = float(candidate.get("loss", math.inf))
    if incumbent_loss is None:
        return True
    return candidate_loss < float(incumbent_loss)


def _category_summary_for_behavior_entry(entry: dict[str, Any], kind: str) -> dict[str, Any]:
    summary_key = f"{kind}_category_summary"
    breakdown_key = f"{kind}_category_breakdown"
    if isinstance(entry.get(summary_key), dict) and entry[summary_key]:
        return entry[summary_key]
    if isinstance(entry.get(breakdown_key), dict) and entry[breakdown_key]:
        return _category_breakdown_summary(entry[breakdown_key])
    item_key = "generation_items" if kind == "generation" else "items"
    return _category_breakdown_summary(_passed_category_breakdown(entry.get(item_key, []), kind))


def _layer_gate_behavior_decision(
    previous_behavior: dict[str, Any],
    candidate_behavior: dict[str, Any],
    max_behavior_drop: float,
) -> dict[str, Any]:
    previous_rate = float(previous_behavior.get("pass_rate", 0.0))
    candidate_rate = float(candidate_behavior.get("pass_rate", 0.0))
    drop = previous_rate - candidate_rate
    previous_passed_ids = {
        str(item.get("id"))
        for item in previous_behavior.get("items", [])
        if bool(item.get("passed"))
    }
    candidate_passed_ids = {
        str(item.get("id"))
        for item in candidate_behavior.get("items", [])
        if bool(item.get("passed"))
    }
    lost_passed_ids = sorted(previous_passed_ids - candidate_passed_ids)
    previous_generation_rate = float(previous_behavior.get("generation_health_rate", 0.0))
    candidate_generation_rate = float(candidate_behavior.get("generation_health_rate", 0.0))
    generation_drop = previous_generation_rate - candidate_generation_rate
    previous_generation_prompts = {
        str(item.get("prompt"))
        for item in previous_behavior.get("generation_items", [])
        if bool(item.get("passed"))
    }
    candidate_generation_prompts = {
        str(item.get("prompt"))
        for item in candidate_behavior.get("generation_items", [])
        if bool(item.get("passed"))
    }
    lost_generation_prompts = sorted(previous_generation_prompts - candidate_generation_prompts)
    accepted = (
        drop <= float(max_behavior_drop) + 1e-12
        and not lost_passed_ids
        and generation_drop <= float(max_behavior_drop) + 1e-12
        and not lost_generation_prompts
    )
    return {
        "metric": "behavior",
        "accepted": bool(accepted),
        "previous_pass_rate": previous_rate,
        "candidate_pass_rate": candidate_rate,
        "behavior_drop": float(drop),
        "previous_generation_health_rate": previous_generation_rate,
        "candidate_generation_health_rate": candidate_generation_rate,
        "generation_health_drop": float(generation_drop),
        "max_behavior_drop": float(max_behavior_drop),
        "previous_passed_ids": sorted(previous_passed_ids),
        "candidate_passed_ids": sorted(candidate_passed_ids),
        "lost_passed_ids": lost_passed_ids,
        "previous_generation_passed_prompts": sorted(previous_generation_prompts),
        "candidate_generation_passed_prompts": sorted(candidate_generation_prompts),
        "lost_generation_passed_prompts": lost_generation_prompts,
        "reason": "accepted" if accepted else "behavior_regression",
    }


def _set_only_module_trainable(model, module) -> int:
    return _set_only_modules_trainable(model, [module])


def _set_only_modules_trainable(model, modules: Sequence[Any]) -> int:
    for param in model.parameters():
        param.requires_grad_(False)
    seen: set[int] = set()
    trainable = 0
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            param.requires_grad_(True)
            trainable += int(param.numel())
    if trainable <= 0:
        raise ValueError("Selected module set has no trainable parameters.")
    return int(trainable)


def _unique_trainable_parameters(modules: Sequence[Any]) -> list[Any]:
    params = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    if not params:
        raise ValueError("No trainable parameters selected.")
    return params


def _hidden_state_after_decoder_layer(hidden_states, layer_idx: int):
    if hidden_states is None:
        raise ValueError("hidden states are required for blockwise reconstruction.")
    state_idx = int(layer_idx) + 1
    if state_idx <= 0 or state_idx >= len(hidden_states):
        raise ValueError(f"Decoder layer index {layer_idx} is outside hidden state range.")
    return hidden_states[state_idx]


def _normalized_source_group_weights(
    source_layer_group: Sequence[int],
    source_layer_group_weights: Sequence[float] | None,
) -> list[float] | None:
    if source_layer_group_weights is None:
        return None
    group = [int(item) for item in source_layer_group]
    weights = [float(item) for item in source_layer_group_weights]
    if len(group) <= 1 or len(weights) != len(group):
        return None
    clean = [max(0.0, weight) if math.isfinite(weight) else 0.0 for weight in weights]
    total = sum(clean)
    if total <= 0:
        return None
    return [weight / total for weight in clean]


def _source_layer_group_weight_lookup(report: dict[str, Any]) -> dict[tuple[int, ...], list[float]]:
    groups = report.get("groups") or []
    weights = report.get("group_layer_weights") or []
    lookup: dict[tuple[int, ...], list[float]] = {}
    for group, group_weights in zip(groups, weights, strict=False):
        key = tuple(int(item) for item in group)
        values = [float(item) for item in group_weights]
        if key and len(key) == len(values):
            lookup[key] = values
    return lookup


def _weighted_state_sum(torch, states: Sequence[Any], weights: Sequence[float], scale: float = 1.0):
    if not states:
        raise ValueError("Need at least one state to fold.")
    values = [state.to(torch.float32) for state in states]
    weight_tensor = torch.tensor(
        [float(item) for item in weights],
        dtype=torch.float32,
        device=values[0].device,
    )
    view_shape = [len(values)] + [1] * values[0].dim()
    return (torch.stack(values, dim=0) * weight_tensor.view(*view_shape)).sum(dim=0) * float(scale)


def _source_group_hidden_state(
    torch,
    hidden_states,
    source_layer_group: Sequence[int],
    donor_layer_fold_mode: str,
    source_layer_group_weights: Sequence[float] | None = None,
):
    if not source_layer_group:
        raise ValueError("Need at least one source layer for blockwise reconstruction.")
    group = [int(item) for item in source_layer_group]
    weights = _normalized_source_group_weights(group, source_layer_group_weights)
    if donor_layer_fold_mode == "average" and len(group) > 1:
        states = [_hidden_state_after_decoder_layer(hidden_states, idx).to(torch.float32) for idx in group]
        if weights is not None:
            return _weighted_state_sum(torch, states, weights)
        return torch.stack(states, dim=0).mean(dim=0)
    if donor_layer_fold_mode == "fold" and len(group) > 1 and weights is not None:
        start_idx = min(group)
        deltas = [
            (_hidden_state_after_decoder_layer(hidden_states, idx) - hidden_states[int(idx)]).to(torch.float32)
            for idx in group
        ]
        return hidden_states[int(start_idx)].to(torch.float32) + _weighted_state_sum(
            torch,
            deltas,
            weights,
            scale=len(group),
        )
    source_idx = group[0] if donor_layer_fold_mode == "representative" else max(group)
    return _hidden_state_after_decoder_layer(hidden_states, source_idx)


def _source_group_input_state(
    torch,
    hidden_states,
    source_layer_group: Sequence[int],
    donor_layer_fold_mode: str,
    source_layer_group_weights: Sequence[float] | None = None,
):
    if not source_layer_group:
        raise ValueError("Need at least one source layer for blockwise reconstruction.")
    group = [int(item) for item in source_layer_group]
    weights = _normalized_source_group_weights(group, source_layer_group_weights)
    if donor_layer_fold_mode == "average" and len(group) > 1:
        states = [hidden_states[int(idx)].to(torch.float32) for idx in group]
        if weights is not None:
            return _weighted_state_sum(torch, states, weights)
        return torch.stack(states, dim=0).mean(dim=0)
    source_idx = group[0] if donor_layer_fold_mode == "representative" else min(group)
    return hidden_states[int(source_idx)]


def _source_group_delta_state(
    torch,
    hidden_states,
    source_layer_group: Sequence[int],
    donor_layer_fold_mode: str,
    source_layer_group_weights: Sequence[float] | None = None,
):
    if not source_layer_group:
        raise ValueError("Need at least one source layer for blockwise reconstruction.")
    group = [int(item) for item in source_layer_group]
    weights = _normalized_source_group_weights(group, source_layer_group_weights)
    if donor_layer_fold_mode == "average" and len(group) > 1:
        deltas = [
            (
                _hidden_state_after_decoder_layer(hidden_states, idx)
                - hidden_states[int(idx)]
            ).to(torch.float32)
            for idx in group
        ]
        if weights is not None:
            return _weighted_state_sum(torch, deltas, weights)
        return torch.stack(deltas, dim=0).mean(dim=0)
    if donor_layer_fold_mode == "fold" and len(group) > 1 and weights is not None:
        deltas = [
            (
                _hidden_state_after_decoder_layer(hidden_states, idx)
                - hidden_states[int(idx)]
            ).to(torch.float32)
            for idx in group
        ]
        return _weighted_state_sum(torch, deltas, weights, scale=len(group))
    source_idx = group[0] if donor_layer_fold_mode == "representative" else max(group)
    start_idx = source_idx if donor_layer_fold_mode == "representative" else min(group)
    return _hidden_state_after_decoder_layer(hidden_states, source_idx) - hidden_states[int(start_idx)]


def _candidate_layer_delta_state(hidden_states, candidate_layer_idx: int):
    return _hidden_state_after_decoder_layer(hidden_states, candidate_layer_idx) - hidden_states[int(candidate_layer_idx)]


def _span_aligned_hidden_rows(torch, candidate_hidden, source_hidden, span_pairs):
    if int(span_pairs.numel()) <= 0:
        empty_candidate = torch.empty((0, int(candidate_hidden.shape[-1])), dtype=torch.float32, device=candidate_hidden.device)
        empty_source = torch.empty((0, int(source_hidden.shape[-1])), dtype=torch.float32, device=candidate_hidden.device)
        return empty_candidate, empty_source, 0
    candidate_values = candidate_hidden.to(torch.float32)
    source_values = source_hidden.to(device=candidate_values.device, dtype=torch.float32)
    pair_idx = span_pairs.to(device=candidate_values.device, dtype=torch.long)
    batch_idx = pair_idx[:, 0]
    candidate_pos = pair_idx[:, 1]
    source_pos = pair_idx[:, 2]
    valid = (
        (batch_idx >= 0)
        & (batch_idx < candidate_values.shape[0])
        & (batch_idx < source_values.shape[0])
        & (candidate_pos >= 0)
        & (candidate_pos < candidate_values.shape[1])
        & (source_pos >= 0)
        & (source_pos < source_values.shape[1])
    )
    if int(valid.detach().cpu().sum().item()) <= 0:
        empty_candidate = torch.empty((0, int(candidate_values.shape[-1])), dtype=torch.float32, device=candidate_values.device)
        empty_source = torch.empty((0, int(source_values.shape[-1])), dtype=torch.float32, device=candidate_values.device)
        return empty_candidate, empty_source, 0
    rows_candidate = candidate_values[batch_idx[valid], candidate_pos[valid]]
    rows_source = source_values[batch_idx[valid], source_pos[valid]]
    rows = min(int(rows_candidate.shape[0]), int(rows_source.shape[0]))
    return rows_candidate[:rows], rows_source[:rows], rows


def _cross_tokenizer_block_hidden_loss(
    torch,
    student,
    teacher,
    source_tokenizer,
    candidate_tokenizer,
    source_batch: dict[str, Any],
    candidate_batch: dict[str, Any],
    source_layer_group: Sequence[int],
    candidate_layer_idx: int,
    hidden_projection,
    canonical_fn,
    donor_layer_fold_mode: str,
    hidden_metric: str,
    block_target: str,
    span_max_pairs: int,
    source_layer_group_weights: Sequence[float] | None = None,
    source_group_weights: Sequence[float] | None = None,
    span_pairs=None,
):
    if block_target not in {"hidden", "delta"}:
        raise ValueError("block_target must be hidden or delta.")
    if source_group_weights is not None:
        if source_layer_group_weights is not None:
            raise ValueError("Pass only one of source_layer_group_weights or source_group_weights.")
        source_layer_group_weights = source_group_weights
    pairs = span_pairs
    if pairs is None:
        pairs = _cross_tokenizer_span_pairs(
            torch,
            source_tokenizer,
            candidate_tokenizer,
            source_batch,
            candidate_batch,
            canonical_fn,
            max_pairs=span_max_pairs,
            shift_positions=False,
        )
    if int(pairs.numel()) <= 0:
        device = student.logits.device
        return torch.zeros((), dtype=student.logits.dtype, device=device), 0
    if block_target == "delta":
        candidate_hidden = _candidate_layer_delta_state(student.hidden_states, candidate_layer_idx)
        source_hidden = _source_group_delta_state(
            torch,
            teacher.hidden_states,
            source_layer_group,
            donor_layer_fold_mode,
            source_layer_group_weights,
        )
    else:
        candidate_hidden = _hidden_state_after_decoder_layer(student.hidden_states, candidate_layer_idx)
        source_hidden = _source_group_hidden_state(
            torch,
            teacher.hidden_states,
            source_layer_group,
            donor_layer_fold_mode,
            source_layer_group_weights,
        )
    student_rows, teacher_rows, row_count = _span_aligned_hidden_rows(torch, candidate_hidden, source_hidden, pairs)
    if int(row_count) <= 0:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device), 0
    loss = _hidden_representation_loss(
        torch,
        student_rows,
        teacher_rows,
        hidden_projection,
        hidden_metric,
    )
    return loss.to(device=student.logits.device, dtype=torch.float32), int(row_count)


def _cross_tokenizer_block_io_reconstruction_loss(
    torch,
    student,
    teacher,
    source_layer_group: Sequence[int],
    candidate_layer_idx: int,
    donor_layer_fold_mode: str,
    source_layer_group_weights: Sequence[float] | None,
    block_target: str,
    span_pairs,
    ridge: float,
    loss_mode: str,
):
    if block_target not in {"hidden", "delta"}:
        raise ValueError("block_target must be hidden or delta.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if loss_mode not in {"smooth-l1", "cosine", "both"}:
        raise ValueError("loss_mode must be smooth-l1, cosine, or both.")
    if int(span_pairs.numel()) <= 0:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device), 0

    candidate_input = student.hidden_states[int(candidate_layer_idx)]
    source_input = _source_group_input_state(
        torch,
        teacher.hidden_states,
        source_layer_group,
        donor_layer_fold_mode,
        source_layer_group_weights,
    )
    candidate_input_rows, source_input_rows, input_count = _span_aligned_hidden_rows(
        torch,
        candidate_input,
        source_input,
        span_pairs,
    )
    if int(input_count) <= 0:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device), 0

    if block_target == "delta":
        candidate_output = _candidate_layer_delta_state(student.hidden_states, candidate_layer_idx)
        source_output = _source_group_delta_state(
            torch,
            teacher.hidden_states,
            source_layer_group,
            donor_layer_fold_mode,
            source_layer_group_weights,
        )
    else:
        candidate_output = _hidden_state_after_decoder_layer(student.hidden_states, candidate_layer_idx)
        source_output = _source_group_hidden_state(
            torch,
            teacher.hidden_states,
            source_layer_group,
            donor_layer_fold_mode,
            source_layer_group_weights,
        )
    candidate_output_rows, source_output_rows, output_count = _span_aligned_hidden_rows(
        torch,
        candidate_output,
        source_output,
        span_pairs,
    )
    rows = min(int(input_count), int(output_count))
    if rows <= 0:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device), 0

    with torch.no_grad():
        adapter = _ridge_solve_features(
            torch,
            candidate_input_rows[:rows].detach().to(torch.float32),
            source_input_rows[:rows].detach().to(torch.float32),
            float(ridge),
        ).detach()
    projected_output = candidate_output_rows[:rows].to(torch.float32).matmul(
        adapter.to(device=candidate_output_rows.device, dtype=torch.float32)
    )
    source_rows = source_output_rows[:rows].to(device=projected_output.device, dtype=torch.float32)
    losses = []
    if loss_mode in {"smooth-l1", "both"}:
        losses.append(torch.nn.functional.smooth_l1_loss(projected_output, source_rows))
    if loss_mode in {"cosine", "both"}:
        projected_norm = torch.nn.functional.normalize(projected_output, dim=-1, eps=1e-8)
        source_norm = torch.nn.functional.normalize(source_rows, dim=-1, eps=1e-8)
        cosine = torch.nn.functional.cosine_similarity(projected_norm, source_norm, dim=-1, eps=1e-8)
        losses.append(1.0 - cosine.mean())
    return torch.stack(losses).mean().to(device=student.logits.device, dtype=torch.float32), int(rows)


def _source_group_module_output_state(
    torch,
    module_outputs: dict[int, Any],
    source_layer_group: Sequence[int],
    donor_layer_fold_mode: str,
    source_layer_group_weights: Sequence[float] | None = None,
):
    if not source_layer_group:
        raise ValueError("Need at least one source layer for module-output folding.")
    group = [int(item) for item in source_layer_group]
    weights = _normalized_source_group_weights(group, source_layer_group_weights)
    missing = [idx for idx in group if idx not in module_outputs]
    if missing:
        raise ValueError(f"Missing module outputs for source layers: {missing}")
    if donor_layer_fold_mode == "average" and len(group) > 1:
        values = [module_outputs[idx].to(torch.float32) for idx in group]
        if weights is not None:
            return _weighted_state_sum(torch, values, weights)
        return torch.stack(values, dim=0).mean(dim=0)
    if donor_layer_fold_mode == "fold" and len(group) > 1:
        values = [module_outputs[idx].to(torch.float32) for idx in group]
        if weights is not None:
            return _weighted_state_sum(torch, values, weights, scale=len(group))
        return torch.stack(values, dim=0).sum(dim=0)
    idx = group[0] if donor_layer_fold_mode == "representative" else max(group)
    return module_outputs[idx].to(torch.float32)


def _source_group_layer_modules(
    source_layers: Sequence[Any] | None,
    source_layer_group: Sequence[int],
    module_getter,
) -> dict[int, Any]:
    if not source_layers or not source_layer_group:
        return {}
    modules: dict[int, Any] = {}
    for raw_idx in source_layer_group:
        idx = int(raw_idx)
        if idx < 0 or idx >= len(source_layers):
            continue
        module = module_getter(source_layers[idx])
        if module is not None:
            modules[idx] = module
    return modules


def _activation_row_transport_loss(
    torch,
    student_rows,
    teacher_rows,
    mode: str,
    temperature: float,
    iterations: int,
):
    if mode not in MODULE_OUTPUT_TRANSPORTS:
        raise ValueError("mode must be paired, sinkhorn, or both.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    rows = min(int(student_rows.shape[0]), int(teacher_rows.shape[0]))
    if rows <= 0:
        return torch.zeros((), dtype=student_rows.dtype, device=student_rows.device)
    student = torch.nan_to_num(student_rows[:rows].to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    teacher = torch.nan_to_num(teacher_rows[:rows].to(device=student.device, dtype=torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    student_norm = _normalize_hidden_rows(torch, student)
    teacher_norm = _normalize_hidden_rows(torch, teacher)
    losses = []
    if mode in {"paired", "both"}:
        cosine = torch.nn.functional.cosine_similarity(student_norm, teacher_norm, dim=-1, eps=1e-8)
        cosine_loss = 1.0 - cosine.mean()
        mse_loss = torch.nn.functional.mse_loss(student_norm, teacher_norm)
        losses.append(0.5 * cosine_loss + 0.5 * mse_loss)
    if mode in {"sinkhorn", "both"}:
        cost = (1.0 - student_norm.matmul(teacher_norm.t())).clamp_min(0.0)
        log_plan = -cost / float(temperature)
        for _ in range(int(iterations)):
            log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
            log_plan = log_plan - torch.logsumexp(log_plan, dim=0, keepdim=True)
        plan = torch.exp(log_plan)
        plan = plan / plan.sum().clamp_min(1e-12)
        losses.append((plan * cost).sum())
    return torch.stack(losses).mean()


def _projected_module_rows_for_pairs(
    torch,
    candidate_output,
    source_output,
    span_pairs,
    hidden_projection,
):
    student_rows, source_rows, row_count = _span_aligned_hidden_rows(
        torch,
        candidate_output,
        source_output,
        span_pairs,
    )
    if int(row_count) <= 0:
        return student_rows, source_rows, 0
    projected = source_rows.to(torch.float32).matmul(
        hidden_projection.to(device=source_rows.device, dtype=torch.float32)
    )
    return student_rows.to(torch.float32), projected.to(device=student_rows.device, dtype=torch.float32), int(row_count)


def _module_output_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _cross_tokenizer_block_module_output_loss(
    torch,
    source_model,
    candidate_model,
    source_layers: Sequence[Any],
    candidate_layers: Sequence[Any],
    source_tokenizer,
    candidate_tokenizer,
    source_batch: dict[str, Any],
    candidate_batch: dict[str, Any],
    source_layer_group: Sequence[int],
    candidate_layer_idx: int,
    hidden_projection,
    canonical_fn,
    donor_layer_fold_mode: str,
    source_layer_group_weights: Sequence[float] | None,
    module_output_targets: str,
    module_output_transport: str,
    module_output_temperature: float,
    module_output_sinkhorn_iterations: int,
    span_max_pairs: int,
    span_pair_mode: str = "exact",
):
    if module_output_targets not in MODULE_OUTPUT_TARGETS:
        raise ValueError("module_output_targets must be attention, mlp, or both.")
    if module_output_transport not in MODULE_OUTPUT_TRANSPORTS:
        raise ValueError("module_output_transport must be paired, sinkhorn, or both.")
    if span_pair_mode not in SPAN_PAIR_MODES:
        raise ValueError("span_pair_mode must be exact, overlap, or both.")
    if hidden_projection is None:
        raise ValueError("hidden_projection is required for module-output KD.")
    if int(candidate_layer_idx) < 0 or int(candidate_layer_idx) >= len(candidate_layers):
        raise ValueError("candidate_layer_idx is outside the candidate layer range.")
    group = [int(item) for item in source_layer_group]
    if not group:
        raise ValueError("Need at least one source layer for module-output KD.")
    target_names = ["attention", "mlp"] if module_output_targets == "both" else [module_output_targets]
    candidate_layer = candidate_layers[int(candidate_layer_idx)]
    source_captured: dict[str, dict[int, Any]] = {"attention": {}, "mlp": {}}
    candidate_captured: dict[str, Any] = {}
    handles = []

    def make_source_hook(target: str, layer_idx: int):
        def hook(_module, _inputs, output):
            source_captured[target][int(layer_idx)] = _module_output_tensor(output).detach().to(torch.float32)

        return hook

    def make_candidate_hook(target: str):
        def hook(_module, _inputs, output):
            candidate_captured[target] = _module_output_tensor(output).to(torch.float32)

        return hook

    try:
        if "attention" in target_names:
            candidate_attention = _attention_module(candidate_layer)
            if candidate_attention is not None:
                handles.append(candidate_attention.register_forward_hook(make_candidate_hook("attention")))
                for idx in group:
                    if idx < 0 or idx >= len(source_layers):
                        continue
                    source_attention = _attention_module(source_layers[idx])
                    if source_attention is not None:
                        handles.append(source_attention.register_forward_hook(make_source_hook("attention", idx)))
        if "mlp" in target_names:
            candidate_mlp_spec = _mlp_projection_spec(candidate_layer)
            if candidate_mlp_spec is not None:
                handles.append(candidate_mlp_spec["mlp"].register_forward_hook(make_candidate_hook("mlp")))
                for idx in group:
                    if idx < 0 or idx >= len(source_layers):
                        continue
                    source_mlp_spec = _mlp_projection_spec(source_layers[idx])
                    if source_mlp_spec is not None:
                        handles.append(source_mlp_spec["mlp"].register_forward_hook(make_source_hook("mlp", idx)))
        if not handles:
            param = next(candidate_model.parameters())
            return (
                torch.zeros((), dtype=param.dtype, device=param.device),
                {
                    "attention_pairs": 0,
                    "mlp_pairs": 0,
                    "targets": module_output_targets,
                    "transport": module_output_transport,
                    "skipped_reason": "no_supported_modules",
                },
            )
        with torch.no_grad():
            source_model(**_model_forward_inputs(source_batch), use_cache=False)
        candidate_model(**_model_forward_inputs(candidate_batch), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    pairs = _cross_tokenizer_span_pairs_for_mode(
        torch,
        source_tokenizer,
        candidate_tokenizer,
        source_batch,
        candidate_batch,
        canonical_fn,
        max_pairs=span_max_pairs,
        shift_positions=False,
        mode=span_pair_mode,
    )
    losses = []
    counts: dict[str, Any] = {
        "attention_pairs": 0,
        "mlp_pairs": 0,
        "targets": module_output_targets,
        "transport": module_output_transport,
    }
    if int(pairs.numel()) <= 0:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), counts

    for target in target_names:
        if target not in candidate_captured:
            counts[f"{target}_skipped"] = "candidate_module_missing"
            continue
        available = source_captured.get(target, {})
        if any(idx not in available for idx in group):
            counts[f"{target}_skipped"] = "source_module_missing"
            continue
        source_output = _source_group_module_output_state(
            torch,
            available,
            group,
            donor_layer_fold_mode,
            source_layer_group_weights,
        )
        student_rows, teacher_rows, row_count = _projected_module_rows_for_pairs(
            torch,
            candidate_captured[target],
            source_output,
            pairs,
            hidden_projection,
        )
        counts[f"{target}_pairs"] = int(row_count)
        if int(row_count) <= 0:
            continue
        losses.append(
            _activation_row_transport_loss(
                torch,
                student_rows,
                teacher_rows,
                module_output_transport,
                module_output_temperature,
                module_output_sinkhorn_iterations,
            )
        )
    if not losses:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), counts
    return torch.stack(losses).mean().to(torch.float32), counts


def _cross_tokenizer_block_on_policy_hidden_loss(
    torch,
    source_model,
    candidate_model,
    source_tokenizer,
    candidate_tokenizer,
    texts: Sequence[str],
    response_loss_mode: str,
    max_new_tokens: int,
    max_length: int,
    device: str,
    source_layer_group: Sequence[int],
    candidate_layer_idx: int,
    hidden_projection,
    canonical_fn,
    donor_layer_fold_mode: str,
    hidden_metric: str,
    block_target: str,
    span_max_pairs: int,
    source_layer_group_weights: Sequence[float] | None = None,
):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    if span_max_pairs <= 0:
        raise ValueError("span_max_pairs must be positive.")
    if hidden_metric not in HIDDEN_KD_METRICS:
        raise ValueError(f"hidden_metric must be {HIDDEN_KD_METRIC_DESCRIPTION}.")
    if hidden_projection is None and hidden_metric in PROJECTED_HIDDEN_KD_METRICS:
        raise ValueError("hidden_projection is required when on_policy_hidden_kd_weight is positive.")
    prompts = [_on_policy_prompt_text(str(text), response_loss_mode) for text in texts]
    prompts = [prompt for prompt in prompts if prompt.strip()]
    if not prompts:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0

    prompt_max_length = max(2, int(max_length) - int(max_new_tokens))
    with _temporary_padding_side(candidate_tokenizer, "left"):
        encoded = candidate_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    pad_id = candidate_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = candidate_tokenizer.eos_token_id
    was_training = candidate_model.training
    try:
        candidate_model.eval()
        with _temporary_padding_side(candidate_tokenizer, "left"):
            with torch.no_grad():
                generated = candidate_model.generate(
                    **encoded,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=candidate_tokenizer.eos_token_id,
                )
    finally:
        if was_training:
            candidate_model.train()

    rollout_texts = [
        candidate_tokenizer.decode(row.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for row in generated.detach().cpu()
    ]
    rollout_texts = [text for text in rollout_texts if text.strip()]
    if not rollout_texts:
        param = next(candidate_model.parameters())
        return torch.zeros((), dtype=param.dtype, device=param.device), 0

    paired = _make_paired_batches(
        source_tokenizer,
        candidate_tokenizer,
        rollout_texts,
        batch_size=len(rollout_texts),
        max_length=max_length,
        device=device,
        include_offsets=True,
        response_loss_mode="none",
        append_eos=False,
    )
    source_batch, candidate_batch = paired[0]
    with torch.no_grad():
        teacher = source_model(
            **_model_forward_inputs(source_batch),
            use_cache=False,
            output_hidden_states=True,
        )
    student = candidate_model(
        **_model_forward_inputs(candidate_batch),
        use_cache=False,
        output_hidden_states=True,
    )
    return _cross_tokenizer_block_hidden_loss(
        torch,
        student,
        teacher,
        source_tokenizer,
        candidate_tokenizer,
        source_batch,
        candidate_batch,
        source_layer_group,
        candidate_layer_idx,
        hidden_projection,
        canonical_fn,
        donor_layer_fold_mode,
        hidden_metric,
        block_target,
        span_max_pairs,
        source_layer_group_weights=source_layer_group_weights,
    )


def _candidate_shift_lm_loss(torch, student, input_ids, shift_mask):
    if int(input_ids.shape[-1]) <= 1:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device)
    shift_logits = torch.nan_to_num(
        student.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    shift_labels = input_ids[:, 1:].contiguous()
    if shift_mask is None:
        mask = torch.ones_like(shift_labels, dtype=torch.bool, device=shift_logits.device)
    else:
        mask = shift_mask.to(device=shift_logits.device).bool()
    if int(mask.sum().detach().cpu().item()) <= 0:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device)
    token_ce = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.to(device=shift_logits.device).view(-1),
        reduction="none",
    ).view_as(shift_labels)
    return (token_ce * mask).sum() / mask.sum().clamp_min(1)


def _candidate_lm_stabilizer_loss(torch, student, input_ids, attention_mask, loss_mask=None):
    if int(input_ids.shape[-1]) <= 1:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device)
    shift_labels = input_ids[:, 1:].contiguous()
    if attention_mask is None:
        mask = torch.ones_like(shift_labels, dtype=torch.bool, device=student.logits.device)
    else:
        mask = attention_mask[:, 1:].contiguous().to(device=student.logits.device).bool()
    if loss_mask is not None:
        mask = mask & loss_mask[:, 1:].contiguous().to(device=student.logits.device).bool()
    return _candidate_shift_lm_loss(torch, student, input_ids, mask)


def _cross_tokenizer_block_alignment_state(
    torch,
    source,
    candidate,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    source_layer_group: Sequence[int],
    candidate_layer_idx: int,
    hidden_projection,
    canonical_fn,
    donor_layer_fold_mode: str,
    block_target: str,
    span_max_pairs: int,
    source_layer_group_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    if block_target not in {"hidden", "delta"}:
        raise ValueError("block_target must be hidden or delta.")
    source_rows = []
    candidate_rows = []
    token_count = 0
    was_training = candidate.training
    source.eval()
    candidate.eval()
    with torch.no_grad():
        for source_batch, candidate_batch in paired_batches:
            if token_count >= int(span_max_pairs):
                break
            teacher = source(
                **_model_forward_inputs(source_batch),
                use_cache=False,
                output_hidden_states=True,
            )
            student = candidate(
                **_model_forward_inputs(candidate_batch),
                use_cache=False,
                output_hidden_states=True,
            )
            pairs = _cross_tokenizer_span_pairs(
                torch,
                source_tokenizer,
                candidate_tokenizer,
                source_batch,
                candidate_batch,
                canonical_fn,
                max_pairs=int(span_max_pairs) - token_count,
                shift_positions=False,
            )
            if block_target == "delta":
                candidate_hidden = _candidate_layer_delta_state(student.hidden_states, candidate_layer_idx)
                source_hidden = _source_group_delta_state(
                    torch,
                    teacher.hidden_states,
                    source_layer_group,
                    donor_layer_fold_mode,
                    source_layer_group_weights,
                )
            else:
                candidate_hidden = _hidden_state_after_decoder_layer(student.hidden_states, candidate_layer_idx)
                source_hidden = _source_group_hidden_state(
                    torch,
                    teacher.hidden_states,
                    source_layer_group,
                    donor_layer_fold_mode,
                    source_layer_group_weights,
                )
            batch_candidate, batch_source, rows = _span_aligned_hidden_rows(torch, candidate_hidden, source_hidden, pairs)
            if int(rows) <= 0:
                continue
            remaining = int(span_max_pairs) - token_count
            take = min(int(rows), remaining)
            candidate_rows.append(batch_candidate[:take].detach().cpu())
            source_rows.append(batch_source[:take].detach().cpu())
            token_count += take
    if was_training:
        candidate.train()
    if not candidate_rows or not source_rows:
        return {
            "skipped": True,
            "reason": "no_cross_tokenizer_block_pairs",
            "span_pairs": 0,
        }
    candidate_matrix = torch.cat(candidate_rows, dim=0).to(torch.float32)
    source_matrix = torch.cat(source_rows, dim=0).to(torch.float32)
    rows = min(int(candidate_matrix.shape[0]), int(source_matrix.shape[0]))
    candidate_matrix = candidate_matrix[:rows]
    source_matrix = source_matrix[:rows]
    projected = source_matrix.to(device=hidden_projection.device, dtype=torch.float32).matmul(
        hidden_projection.to(dtype=torch.float32)
    ).detach().cpu()
    return {
        "skipped": False,
        "block_target": block_target,
        "span_pairs": int(rows),
        "candidate_matrix": candidate_matrix,
        "source_matrix": source_matrix,
        "projected_matrix": projected,
    }


def _cross_tokenizer_block_alignment_metrics(
    torch,
    alignment_state: dict[str, Any],
    hidden_projection,
    hidden_metric: str,
) -> dict[str, Any]:
    if alignment_state.get("skipped"):
        return {
            "skipped": True,
            "reason": alignment_state.get("reason", "alignment_state_skipped"),
            "span_pairs": int(alignment_state.get("span_pairs", 0)),
        }
    candidate_matrix = alignment_state["candidate_matrix"]
    source_matrix = alignment_state["source_matrix"]
    projected = alignment_state["projected_matrix"]
    candidate_rms = _activation_rms(torch, candidate_matrix)
    projected_rms = _activation_rms(torch, projected)
    metrics = {
        "skipped": False,
        "block_target": alignment_state.get("block_target"),
        "span_pairs": int(alignment_state.get("span_pairs", int(candidate_matrix.shape[0]))),
        "candidate_hidden_size": int(candidate_matrix.shape[1]),
        "source_hidden_size": int(source_matrix.shape[1]),
        "candidate_rms": candidate_rms,
        "projected_rms": projected_rms,
        "projected_to_candidate_rms_ratio": projected_rms / max(candidate_rms, 1e-12),
        "projected_relative_error": _relative_reconstruction_error(torch, candidate_matrix, projected),
        "projected_mean_cosine": _mean_row_cosine(torch, candidate_matrix, projected),
        "cka": float(_linear_cka_tensor(torch, candidate_matrix, source_matrix).detach().cpu()),
    }
    metrics["loss"] = float(
        _hidden_representation_loss(
            torch,
            candidate_matrix,
            source_matrix,
            hidden_projection.detach().cpu() if hidden_projection is not None else None,
            hidden_metric,
        )
        .detach()
        .cpu()
    )
    return metrics


def _activation_rms(torch, rows) -> float:
    values = torch.nan_to_num(rows.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if int(values.numel()) <= 0:
        return 0.0
    return float(values.pow(2).mean().sqrt().detach().cpu())


def _apply_layer_residual_magnitude_compensation(
    torch,
    layer,
    alignment_state: dict[str, Any],
    alignment_metrics: dict[str, Any],
    min_scale: float,
    max_scale: float,
    mode: str,
    linear_ridge: float,
    alpha: float,
    linear_rank: int,
    *,
    source=None,
    candidate=None,
    source_layers: Sequence[Any] | None = None,
    source_layer_group: Sequence[int] | None = None,
    source_layer_group_weights: Sequence[float] | None = None,
    source_attention_layer=None,
    source_mlp_layer=None,
    source_tokenizer=None,
    candidate_tokenizer=None,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]] | None = None,
    hidden_projection=None,
    canonical_fn=None,
    donor_layer_fold_mode: str = "fold",
    fit_gate: str = "none",
    fit_gate_validation_fraction: float = 0.25,
    fit_gate_min_validation_rows: int = 32,
    fit_gate_min_relative_improvement: float = 0.0,
    fit_gate_max_cosine_drop: float = 0.01,
    max_tokens_per_layer: int = 512,
) -> dict[str, Any]:
    canonical_mode = {
        "linear-pre": "pre-linear",
        "lowrank-pre": "pre-lowrank",
        "eora-pre": "pre-eora",
        "module-linear-pre": "pre-module-linear",
        "module-lowrank-pre": "pre-module-lowrank",
        "module-eora-pre": "pre-module-eora",
    }.get(mode, mode)
    if canonical_mode in MODULE_RESIDUAL_COMPENSATION_MODES:
        return _apply_layer_module_residual_compensation(
            torch,
            source,
            candidate,
            source_layers,
            source_layer_group,
            source_layer_group_weights,
            source_attention_layer,
            source_mlp_layer,
            layer,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            hidden_projection,
            canonical_fn,
            donor_layer_fold_mode,
            float(linear_ridge),
            float(alpha),
            int(linear_rank),
            canonical_mode,
            fit_gate,
            float(fit_gate_validation_fraction),
            int(fit_gate_min_validation_rows),
            float(fit_gate_min_relative_improvement),
            float(fit_gate_max_cosine_drop),
            int(max_tokens_per_layer),
        )
    if alignment_metrics.get("skipped"):
        return {
            "enabled": True,
            "applied": False,
            "mode": canonical_mode,
            "reason": alignment_metrics.get("reason", "alignment_metrics_skipped"),
        }
    if canonical_mode in {"pre-linear", "pre-lowrank", "pre-eora"}:
        return _apply_layer_residual_linear_compensation(
            torch,
            layer,
            alignment_state,
            float(linear_ridge),
            float(alpha),
            int(linear_rank),
            canonical_mode,
        )
    if canonical_mode != "pre":
        return {
            "enabled": True,
            "applied": False,
            "mode": canonical_mode,
            "reason": "unsupported_residual_compensation_mode",
        }
    candidate_rms = float(alignment_metrics.get("candidate_rms", 0.0))
    projected_rms = float(alignment_metrics.get("projected_rms", 0.0))
    if candidate_rms <= 0 or not math.isfinite(candidate_rms) or not math.isfinite(projected_rms):
        return {
            "enabled": True,
            "applied": False,
            "mode": canonical_mode,
            "reason": "invalid_rms",
            "candidate_rms": candidate_rms,
            "projected_rms": projected_rms,
        }
    raw_scale = projected_rms / max(candidate_rms, 1e-12)
    scale = min(float(max_scale), max(float(min_scale), float(raw_scale)))
    module_reports = _scale_layer_residual_output_modules(torch, layer, scale)
    return {
        "enabled": True,
        "applied": bool(module_reports),
        "mode": canonical_mode,
        "raw_scale": float(raw_scale),
        "scale": float(scale),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "candidate_rms": candidate_rms,
        "projected_rms": projected_rms,
        "modules": module_reports,
        "reason": "applied" if module_reports else "no_supported_output_modules",
    }


def _apply_layer_residual_linear_compensation(
    torch,
    layer,
    alignment_state: dict[str, Any],
    linear_ridge: float,
    alpha: float,
    linear_rank: int,
    mode: str = "pre-linear",
) -> dict[str, Any]:
    if alignment_state.get("skipped"):
        return {
            "enabled": True,
            "applied": False,
            "mode": mode,
            "reason": alignment_state.get("reason", "alignment_state_skipped"),
        }
    candidate_matrix = alignment_state.get("candidate_matrix")
    projected_matrix = alignment_state.get("projected_matrix")
    if candidate_matrix is None or projected_matrix is None:
        return {
            "enabled": True,
            "applied": False,
            "mode": mode,
            "reason": "missing_alignment_matrices",
        }
    if int(candidate_matrix.shape[1]) != int(projected_matrix.shape[1]):
        return {
            "enabled": True,
            "applied": False,
            "mode": mode,
            "reason": "hidden_size_mismatch",
            "candidate_hidden_size": int(candidate_matrix.shape[1]),
            "projected_hidden_size": int(projected_matrix.shape[1]),
        }
    fit_alpha = 1.0 if mode in {"pre-lowrank", "pre-eora"} else float(alpha)
    fit = _fit_activation_map(
        torch,
        candidate_matrix,
        projected_matrix,
        float(linear_ridge),
        reconstruction_alpha=fit_alpha,
        reconstruction_objective="ridge",
        cosine_steps=0,
        cosine_lr=0.0,
        cosine_l2=0.0,
    )
    map_to_apply = fit["map"]
    lowrank_report = None
    if mode == "pre-lowrank":
        lowrank_map, lowrank_report = _low_rank_delta_map(torch, map_to_apply, float(alpha), int(linear_rank))
        map_to_apply = lowrank_map
        fit["reconstruction_alpha"] = float(alpha)
        predicted = candidate_matrix[: int(fit["tokens"])].to(torch.float32).matmul(map_to_apply)
        target = projected_matrix[: int(fit["tokens"])].to(torch.float32)
        fit["relative_error_after"] = _relative_reconstruction_error(torch, predicted, target)
        fit["mean_cosine_after"] = _mean_row_cosine(torch, predicted, target)
    elif mode == "pre-eora":
        lowrank_map, lowrank_report = _eigenspace_low_rank_delta_map(
            torch,
            map_to_apply,
            candidate_matrix,
            float(alpha),
            int(linear_rank),
            ridge=float(linear_ridge),
        )
        map_to_apply = lowrank_map
        fit["reconstruction_alpha"] = float(alpha)
        predicted = candidate_matrix[: int(fit["tokens"])].to(torch.float32).matmul(map_to_apply)
        target = projected_matrix[: int(fit["tokens"])].to(torch.float32)
        fit["relative_error_after"] = _relative_reconstruction_error(torch, predicted, target)
        fit["mean_cosine_after"] = _mean_row_cosine(torch, predicted, target)
    module_reports = _apply_residual_output_map_to_modules(torch, layer, map_to_apply)
    return {
        "enabled": True,
        "applied": bool(module_reports),
        "mode": mode,
        "ridge": float(linear_ridge),
        "alpha": float(alpha),
        "linear_rank": int(linear_rank),
        "tokens": int(fit["tokens"]),
        "hidden_size": int(fit["hidden_size"]),
        "relative_error_before": float(fit["relative_error_before"]),
        "relative_error_solved": float(fit["relative_error_solved"]),
        "relative_error_after": float(fit["relative_error_after"]),
        "mean_cosine_before": float(fit["mean_cosine_before"]),
        "mean_cosine_solved": float(fit["mean_cosine_solved"]),
        "mean_cosine_after": float(fit["mean_cosine_after"]),
        "solver": fit.get("solver"),
        "lowrank": lowrank_report,
        "modules": module_reports,
        "reason": "applied" if module_reports else "no_supported_output_modules",
    }


def _apply_layer_module_residual_compensation(
    torch,
    source,
    candidate,
    source_layers: Sequence[Any] | None,
    source_layer_group: Sequence[int] | None,
    source_layer_group_weights: Sequence[float] | None,
    source_attention_layer,
    source_mlp_layer,
    candidate_layer,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]] | None,
    hidden_projection,
    canonical_fn,
    donor_layer_fold_mode: str,
    linear_ridge: float,
    alpha: float,
    linear_rank: int,
    mode: str,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "mode": mode,
        "ridge": float(linear_ridge),
        "alpha": float(alpha),
        "linear_rank": int(linear_rank),
        "modules": [],
    }
    required = {
        "source": source,
        "candidate": candidate,
        "source_layers": source_layers,
        "source_layer_group": source_layer_group,
        "source_tokenizer": source_tokenizer,
        "candidate_tokenizer": candidate_tokenizer,
        "paired_batches": paired_batches,
        "hidden_projection": hidden_projection,
        "canonical_fn": canonical_fn,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        report["reason"] = "missing_module_compensation_context"
        report["missing"] = missing
        return report
    if fit_gate not in {"none", "validation"}:
        report["reason"] = "unsupported_fit_gate"
        report["fit_gate"] = fit_gate
        return report

    module_reports = [
        _apply_attention_residual_output_compensation(
            torch,
            source,
            candidate,
            source_layers,
            source_layer_group,
            source_layer_group_weights,
            source_attention_layer,
            candidate_layer,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches or [],
            hidden_projection,
            canonical_fn,
            donor_layer_fold_mode,
            linear_ridge,
            alpha,
            linear_rank,
            mode,
            fit_gate,
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
            max_tokens_per_layer,
        ),
        _apply_mlp_residual_output_compensation(
            torch,
            source,
            candidate,
            source_layers,
            source_layer_group,
            source_layer_group_weights,
            source_mlp_layer,
            candidate_layer,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches or [],
            hidden_projection,
            canonical_fn,
            donor_layer_fold_mode,
            linear_ridge,
            alpha,
            linear_rank,
            mode,
            fit_gate,
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
            max_tokens_per_layer,
        ),
    ]
    report["modules"] = module_reports
    report["applied"] = any(bool(item.get("applied")) for item in module_reports)
    report["reason"] = "applied" if report["applied"] else "no_module_compensation_applied"
    return report


def _apply_attention_residual_output_compensation(
    torch,
    source,
    candidate,
    source_layers: Sequence[Any] | None,
    source_layer_group: Sequence[int] | None,
    source_layer_group_weights: Sequence[float] | None,
    source_layer,
    candidate_layer,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    donor_layer_fold_mode: str,
    linear_ridge: float,
    alpha: float,
    linear_rank: int,
    mode: str,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
) -> dict[str, Any]:
    spec = _attention_projection_spec(candidate_layer)
    if spec is None:
        return {
            "kind": "attention",
            "applied": False,
            "skipped": True,
            "reason": "unsupported_attention_output_shape",
        }
    candidate_rows, target_rows, matched = _collect_cross_tokenizer_module_output_alignment_rows(
        torch,
        source,
        candidate,
        source_layers,
        source_layer_group,
        source_layer_group_weights,
        source_layer,
        candidate_layer,
        "attention",
        source_tokenizer,
        candidate_tokenizer,
        paired_batches,
        hidden_projection,
        canonical_fn,
        donor_layer_fold_mode,
        max_tokens_per_layer,
    )
    if int(matched) <= 0:
        return {
            "kind": "attention",
            "applied": False,
            "skipped": True,
            "reason": "no_cross_tokenizer_attention_output_pairs",
        }
    rows = min(int(target_rows.shape[0]), int(candidate_rows.shape[0]))
    fit_report = _fit_gated_residual_output_map(
        torch,
        candidate_rows[:rows].to(torch.float32),
        target_rows[:rows].to(torch.float32),
        None,
        None,
        None,
        linear_ridge,
        alpha,
        linear_rank,
        mode,
        fit_gate,
        fit_gate_validation_fraction,
        fit_gate_min_validation_rows,
        fit_gate_min_relative_improvement,
        fit_gate_max_cosine_drop,
    )
    module_report = {
        "kind": "attention",
        "name": spec["output_name"],
        "matched_pairs": int(matched),
        "tokens": int(rows),
        **{key: value for key, value in fit_report.items() if key != "map"},
    }
    if fit_report.get("mutation_skipped"):
        return {**module_report, "applied": False}
    applied = _apply_residual_output_map_to_module(torch, "attention", spec["output_name"], spec["output"], fit_report["map"])
    return {**module_report, "applied": True, "module": applied}


def _apply_mlp_residual_output_compensation(
    torch,
    source,
    candidate,
    source_layers: Sequence[Any] | None,
    source_layer_group: Sequence[int] | None,
    source_layer_group_weights: Sequence[float] | None,
    source_layer,
    candidate_layer,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    donor_layer_fold_mode: str,
    linear_ridge: float,
    alpha: float,
    linear_rank: int,
    mode: str,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
) -> dict[str, Any]:
    spec = _mlp_projection_spec(candidate_layer)
    if spec is None:
        return {
            "kind": "mlp",
            "applied": False,
            "skipped": True,
            "reason": "unsupported_mlp_shape",
        }
    candidate_rows, target_rows, matched = _collect_cross_tokenizer_module_output_alignment_rows(
        torch,
        source,
        candidate,
        source_layers,
        source_layer_group,
        source_layer_group_weights,
        source_layer,
        candidate_layer,
        "mlp",
        source_tokenizer,
        candidate_tokenizer,
        paired_batches,
        hidden_projection,
        canonical_fn,
        donor_layer_fold_mode,
        max_tokens_per_layer,
    )
    if int(matched) <= 0:
        return {
            "kind": "mlp",
            "applied": False,
            "skipped": True,
            "reason": "no_cross_tokenizer_mlp_output_pairs",
        }
    rows = min(int(target_rows.shape[0]), int(candidate_rows.shape[0]))
    fit_report = _fit_gated_residual_output_map(
        torch,
        candidate_rows[:rows].to(torch.float32),
        target_rows[:rows].to(torch.float32),
        None,
        None,
        None,
        linear_ridge,
        alpha,
        linear_rank,
        mode,
        fit_gate,
        fit_gate_validation_fraction,
        fit_gate_min_validation_rows,
        fit_gate_min_relative_improvement,
        fit_gate_max_cosine_drop,
    )
    module_report = {
        "kind": "mlp",
        "name": spec["output_name"],
        "matched_pairs": int(matched),
        "tokens": int(rows),
        **{key: value for key, value in fit_report.items() if key != "map"},
    }
    if fit_report.get("mutation_skipped"):
        return {**module_report, "applied": False}
    applied = _apply_residual_output_map_to_module(torch, "mlp", spec["output_name"], spec["output"], fit_report["map"])
    return {**module_report, "applied": True, "module": applied}


def _collect_cross_tokenizer_module_output_alignment_rows(
    torch,
    source,
    candidate,
    source_layers: Sequence[Any] | None,
    source_layer_group: Sequence[int] | None,
    source_layer_group_weights: Sequence[float] | None,
    source_layer,
    candidate_layer,
    target: str,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    donor_layer_fold_mode: str,
    max_tokens: int,
):
    if target not in MODULE_OUTPUT_TARGETS - {"both"}:
        raise ValueError("target must be attention or mlp.")
    group = [int(item) for item in (source_layer_group or [])]
    if not group:
        group = [0]
        source_layers = [source_layer]
    elif source_layers is None:
        source_layers = [source_layer]
        group = [0]

    candidate_module = None
    if target == "attention":
        candidate_module = _attention_module(candidate_layer)
    else:
        candidate_spec = _mlp_projection_spec(candidate_layer)
        candidate_module = None if candidate_spec is None else candidate_spec["mlp"]
    if candidate_module is None:
        empty = torch.empty((0, 0), dtype=torch.float32)
        return empty, empty, 0

    candidate_pieces = []
    target_pieces = []
    token_count = 0
    was_source_training = source.training
    was_candidate_training = candidate.training

    def source_module_for_idx(idx: int):
        if source_layers is None or idx < 0 or idx >= len(source_layers):
            return None
        layer = source_layers[idx]
        if target == "attention":
            return _attention_module(layer)
        spec = _mlp_projection_spec(layer)
        return None if spec is None else spec["mlp"]

    try:
        source.eval()
        candidate.eval()
        for source_batch, candidate_batch in paired_batches:
            if token_count >= max_tokens:
                break
            source_captured: dict[int, Any] = {}
            candidate_captured: dict[str, Any] = {}
            handles = []

            def make_source_hook(layer_idx: int):
                def hook(_module, _inputs, output):
                    source_captured[int(layer_idx)] = _module_output_tensor(output).detach().to(torch.float32)

                return hook

            def candidate_hook(_module, _inputs, output):
                candidate_captured["output"] = _module_output_tensor(output).detach().to(torch.float32)

            try:
                handles.append(candidate_module.register_forward_hook(candidate_hook))
                for idx in group:
                    source_module = source_module_for_idx(int(idx))
                    if source_module is not None:
                        handles.append(source_module.register_forward_hook(make_source_hook(int(idx))))
                if len(handles) <= 1:
                    continue
                with torch.no_grad():
                    source(**_model_forward_inputs(source_batch), use_cache=False)
                    candidate(**_model_forward_inputs(candidate_batch), use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()

            if "output" not in candidate_captured or any(idx not in source_captured for idx in group):
                continue
            pairs = _cross_tokenizer_span_pairs(
                torch,
                source_tokenizer,
                candidate_tokenizer,
                source_batch,
                candidate_batch,
                canonical_fn,
                max_pairs=max_tokens - token_count,
                shift_positions=False,
            )
            if int(pairs.numel()) <= 0:
                continue
            source_output = _source_group_module_output_state(
                torch,
                source_captured,
                group,
                donor_layer_fold_mode,
                source_layer_group_weights,
            )
            candidate_rows, projected_rows, matched = _projected_module_rows_for_pairs(
                torch,
                candidate_captured["output"],
                source_output,
                pairs,
                hidden_projection,
            )
            if int(matched) <= 0:
                continue
            remaining = int(max_tokens) - token_count
            take = min(int(matched), remaining)
            if take <= 0:
                break
            candidate_pieces.append(candidate_rows[:take].detach().cpu())
            target_pieces.append(projected_rows[:take].detach().cpu())
            token_count += take
    finally:
        if was_source_training:
            source.train()
        if was_candidate_training:
            candidate.train()

    if not candidate_pieces or not target_pieces:
        empty = torch.empty((0, 0), dtype=torch.float32)
        return empty, empty, 0
    candidate_rows = torch.cat(candidate_pieces, dim=0)[:max_tokens]
    target_rows = torch.cat(target_pieces, dim=0)[:max_tokens]
    rows = min(int(candidate_rows.shape[0]), int(target_rows.shape[0]))
    return candidate_rows[:rows], target_rows[:rows], rows


def _fit_gated_residual_output_map(
    torch,
    candidate_rows,
    target_rows,
    residual_rows,
    full_before_rows,
    full_target_rows,
    linear_ridge: float,
    alpha: float,
    linear_rank: int,
    mode: str,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
) -> dict[str, Any]:
    tensors = [candidate_rows.to(torch.float32), target_rows.to(torch.float32)]
    has_full_gate = residual_rows is not None and full_before_rows is not None and full_target_rows is not None
    if has_full_gate:
        tensors.extend([residual_rows.to(torch.float32), full_before_rows.to(torch.float32), full_target_rows.to(torch.float32)])
    rows = min(int(tensor.shape[0]) for tensor in tensors)
    tensors = [tensor[:rows] for tensor in tensors]
    fit_candidate_rows = tensors[0]
    fit_target_rows = tensors[1]
    validation_tensors = None
    fit_gate_report: dict[str, Any] = {}
    if fit_gate == "validation":
        split = _split_reconstruction_rows(tensors, fit_gate_validation_fraction, fit_gate_min_validation_rows)
        if split is None:
            return {
                "fit_gate": {"mode": "validation", "accepted": False, "reason": "insufficient_rows"},
                "mutation_skipped": True,
            }
        else:
            fit_tensors = split["train"]
            validation_tensors = split["validation"]
            fit_candidate_rows = fit_tensors[0]
            fit_target_rows = fit_tensors[1]
            fit_gate_report["fit_gate"] = {
                "mode": "validation",
                "rows": int(split["rows"]),
                "train_rows": int(split["train_rows"]),
                "validation_rows": int(split["validation_rows"]),
            }

    map_to_apply, fit, lowrank_report = _fit_residual_output_map(
        torch,
        fit_candidate_rows,
        fit_target_rows,
        linear_ridge,
        alpha,
        linear_rank,
        mode,
    )
    if validation_tensors is not None:
        validation_candidate = validation_tensors[0]
        validation_target = validation_tensors[1]
        before = validation_candidate
        after = validation_candidate.matmul(map_to_apply)
        target = validation_target
        if has_full_gate:
            validation_residual = validation_tensors[2]
            before = validation_tensors[3]
            target = validation_tensors[4]
            after = validation_residual + validation_candidate.matmul(map_to_apply)
        gate = _validation_reconstruction_gate(
            torch,
            before,
            after,
            target,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
        )
        fit_gate_report["fit_gate"] = {**fit_gate_report["fit_gate"], **gate}
        if not gate["accepted"]:
            return {
                **_residual_output_fit_report(fit, lowrank_report, mode, linear_ridge, alpha, linear_rank),
                **fit_gate_report,
                "mutation_skipped": True,
            }
    return {
        "map": map_to_apply,
        **_residual_output_fit_report(fit, lowrank_report, mode, linear_ridge, alpha, linear_rank),
        **fit_gate_report,
    }


def _fit_residual_output_map(
    torch,
    candidate_rows,
    target_rows,
    linear_ridge: float,
    alpha: float,
    linear_rank: int,
    mode: str,
):
    fit_alpha = 1.0 if mode in {"pre-module-lowrank", "pre-module-eora"} else float(alpha)
    fit = _fit_activation_map(
        torch,
        candidate_rows,
        target_rows,
        float(linear_ridge),
        reconstruction_alpha=fit_alpha,
        reconstruction_objective="ridge",
        cosine_steps=0,
        cosine_lr=0.0,
        cosine_l2=0.0,
    )
    map_to_apply = fit["map"]
    lowrank_report = None
    if mode == "pre-module-lowrank":
        map_to_apply, lowrank_report = _low_rank_delta_map(torch, map_to_apply, float(alpha), int(linear_rank))
        fit["reconstruction_alpha"] = float(alpha)
        predicted = candidate_rows[: int(fit["tokens"])].to(torch.float32).matmul(map_to_apply)
        target = target_rows[: int(fit["tokens"])].to(torch.float32)
        fit["relative_error_after"] = _relative_reconstruction_error(torch, predicted, target)
        fit["mean_cosine_after"] = _mean_row_cosine(torch, predicted, target)
    elif mode == "pre-module-eora":
        map_to_apply, lowrank_report = _eigenspace_low_rank_delta_map(
            torch,
            map_to_apply,
            candidate_rows,
            float(alpha),
            int(linear_rank),
            ridge=float(linear_ridge),
        )
        fit["reconstruction_alpha"] = float(alpha)
        predicted = candidate_rows[: int(fit["tokens"])].to(torch.float32).matmul(map_to_apply)
        target = target_rows[: int(fit["tokens"])].to(torch.float32)
        fit["relative_error_after"] = _relative_reconstruction_error(torch, predicted, target)
        fit["mean_cosine_after"] = _mean_row_cosine(torch, predicted, target)
    return map_to_apply, fit, lowrank_report


def _residual_output_fit_report(
    fit: dict[str, Any],
    lowrank_report,
    mode: str,
    linear_ridge: float,
    alpha: float,
    linear_rank: int,
) -> dict[str, Any]:
    return {
        "skipped": False,
        "mode": mode,
        "ridge": float(linear_ridge),
        "alpha": float(alpha),
        "linear_rank": int(linear_rank),
        "hidden_size": int(fit["hidden_size"]),
        "relative_error_before": float(fit["relative_error_before"]),
        "relative_error_solved": float(fit["relative_error_solved"]),
        "relative_error_after": float(fit["relative_error_after"]),
        "mean_cosine_before": float(fit["mean_cosine_before"]),
        "mean_cosine_solved": float(fit["mean_cosine_solved"]),
        "mean_cosine_after": float(fit["mean_cosine_after"]),
        "solver": fit.get("solver"),
        "lowrank": lowrank_report,
    }


def _scale_layer_residual_output_modules(torch, layer, scale: float) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[int] = set()
    specs = []
    attention_spec = _attention_projection_spec(layer)
    if attention_spec is not None:
        specs.append(("attention", attention_spec["output_name"], attention_spec["output"]))
    mlp_spec = _mlp_projection_spec(layer)
    if mlp_spec is not None:
        specs.append(("mlp", mlp_spec["output_name"], mlp_spec["output"]))
    with torch.no_grad():
        for kind, name, module in specs:
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            module_id = id(weight)
            if module_id in seen:
                continue
            seen.add(module_id)
            weight.data.mul_(float(scale))
            bias = getattr(module, "bias", None)
            bias_scaled = bool(bias is not None)
            if bias is not None:
                bias.data.mul_(float(scale))
            reports.append(
                {
                    "kind": kind,
                    "name": name,
                    "scale": float(scale),
                    "weight_shape": [int(item) for item in weight.shape],
                    "bias_scaled": bias_scaled,
                }
            )
    return reports


def _low_rank_delta_map(torch, full_map, alpha: float, rank: int):
    map_cpu = full_map.detach().cpu().to(torch.float32)
    if int(map_cpu.shape[0]) != int(map_cpu.shape[1]):
        raise ValueError("Low-rank residual compensation map must be square.")
    hidden = int(map_cpu.shape[0])
    used_rank = max(1, min(int(rank), hidden))
    identity = torch.eye(hidden, dtype=map_cpu.dtype)
    delta = map_cpu - identity
    singular_values = torch.linalg.svdvals(delta)
    total_energy = float(singular_values.pow(2).sum().detach().cpu())
    if used_rank >= hidden:
        truncated = delta
    else:
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        truncated = (u[:, :used_rank] * s[:used_rank]).matmul(vh[:used_rank, :])
    retained_error = float((delta - truncated).norm().detach().cpu())
    retained_energy = float(torch.linalg.svdvals(truncated).pow(2).sum().detach().cpu())
    compensated = identity + float(alpha) * truncated
    return compensated, {
        "rank": int(used_rank),
        "requested_rank": int(rank),
        "hidden_size": hidden,
        "alpha": float(alpha),
        "delta_norm": float(delta.norm().detach().cpu()),
        "truncated_delta_norm": float(truncated.norm().detach().cpu()),
        "discarded_delta_norm": retained_error,
        "retained_energy_ratio": retained_energy / max(total_energy, 1e-12),
    }


def _eigenspace_low_rank_delta_map(torch, full_map, activation_rows, alpha: float, rank: int, ridge: float = 1e-6):
    map_cpu = torch.nan_to_num(full_map.detach().cpu().to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    rows_cpu = torch.nan_to_num(activation_rows.detach().cpu().to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if int(map_cpu.dim()) != 2 or int(map_cpu.shape[0]) != int(map_cpu.shape[1]):
        raise ValueError("Eigenspace residual compensation map must be square.")
    if int(rows_cpu.dim()) != 2 or int(rows_cpu.shape[1]) != int(map_cpu.shape[0]):
        raise ValueError("activation_rows must be rank-2 with width matching full_map.")
    if int(rank) <= 0:
        raise ValueError("rank must be positive.")
    if not 0 <= float(alpha) <= 1:
        raise ValueError("alpha must be in [0, 1].")
    if float(ridge) < 0:
        raise ValueError("ridge must be non-negative.")
    hidden = int(map_cpu.shape[0])
    identity = torch.eye(hidden, dtype=map_cpu.dtype)
    delta = map_cpu - identity
    if float(delta.norm().detach().cpu()) == 0.0:
        return identity, {
            "method": "eigenspace_low_rank_delta",
            "rank": 0,
            "requested_rank": int(rank),
            "hidden_size": hidden,
            "alpha": float(alpha),
            "ridge": float(ridge),
            "activation_rows": int(rows_cpu.shape[0]),
            "reason": "zero_delta",
        }

    covariance = rows_cpu.t().matmul(rows_cpu) / max(int(rows_cpu.shape[0]), 1)
    if float(ridge) > 0:
        covariance = covariance + float(ridge) * torch.eye(hidden, dtype=covariance.dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(float(ridge) if float(ridge) > 0 else 1e-12)
    weighted_delta = eigenvalues.sqrt().unsqueeze(1) * eigenvectors.t().matmul(delta)
    u, singular_values, vh = torch.linalg.svd(weighted_delta, full_matrices=False)
    full_rank = min(int(weighted_delta.shape[0]), int(weighted_delta.shape[1]))
    used_rank = max(1, min(int(rank), full_rank))
    weighted_truncated = (u[:, :used_rank] * singular_values[:used_rank]).matmul(vh[:used_rank, :])
    truncated = eigenvectors.matmul(eigenvalues.sqrt().reciprocal().unsqueeze(1) * weighted_truncated)
    compensated = identity + float(alpha) * truncated
    total_energy = float(singular_values.pow(2).sum().detach().cpu())
    retained_energy = float(singular_values[:used_rank].pow(2).sum().detach().cpu())
    retained_error = float((rows_cpu.matmul(delta - truncated)).norm().detach().cpu())
    return compensated.contiguous(), {
        "method": "eigenspace_low_rank_delta",
        "rank": int(used_rank),
        "requested_rank": int(rank),
        "full_rank": int(full_rank),
        "hidden_size": hidden,
        "alpha": float(alpha),
        "ridge": float(ridge),
        "activation_rows": int(rows_cpu.shape[0]),
        "delta_norm": float(delta.norm().detach().cpu()),
        "truncated_delta_norm": float(truncated.norm().detach().cpu()),
        "activation_weighted_discarded_delta_norm": retained_error,
        "retained_energy_ratio": retained_energy / max(total_energy, 1e-12),
        "top_eigenvalues": [float(item.detach().cpu()) for item in eigenvalues[-min(8, int(eigenvalues.numel())) :]],
        "singular_values": [float(item.detach().cpu()) for item in singular_values[: min(8, int(singular_values.numel()))]],
    }


def _apply_residual_output_map_to_modules(torch, layer, mapping) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    seen: set[int] = set()
    specs = []
    attention_spec = _attention_projection_spec(layer)
    if attention_spec is not None:
        specs.append(("attention", attention_spec["output_name"], attention_spec["output"]))
    mlp_spec = _mlp_projection_spec(layer)
    if mlp_spec is not None:
        specs.append(("mlp", mlp_spec["output_name"], mlp_spec["output"]))
    map_cpu = mapping.detach().cpu().to(torch.float32)
    if int(map_cpu.shape[0]) != int(map_cpu.shape[1]):
        raise ValueError("Residual output compensation map must be square.")
    with torch.no_grad():
        for kind, name, module in specs:
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            module_id = id(weight)
            if module_id in seen:
                continue
            seen.add(module_id)
            old_weight = weight.data.detach().cpu().to(torch.float32)
            if int(old_weight.shape[0]) != int(map_cpu.shape[0]):
                raise ValueError(
                    f"Cannot absorb residual output map with output {map_cpu.shape[0]} into weight {tuple(old_weight.shape)}."
                )
            updated = map_cpu.T.matmul(old_weight)
            weight.data.copy_(updated.to(device=weight.device, dtype=weight.dtype))
            bias = getattr(module, "bias", None)
            bias_updated = bool(bias is not None)
            if bias is not None:
                old_bias = bias.data.detach().cpu().to(torch.float32)
                bias.data.copy_(old_bias.matmul(map_cpu).to(device=bias.device, dtype=bias.dtype))
            reports.append(
                {
                    "kind": kind,
                    "name": name,
                    "map_shape": [int(item) for item in map_cpu.shape],
                    "weight_shape": [int(item) for item in weight.shape],
                    "bias_updated": bias_updated,
                }
            )
    return reports


def _apply_residual_output_map_to_module(torch, kind: str, name: str, module, mapping) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError("Cannot absorb residual output map into a module without a weight.")
    map_cpu = mapping.detach().cpu().to(torch.float32)
    if int(map_cpu.shape[0]) != int(map_cpu.shape[1]):
        raise ValueError("Residual output compensation map must be square.")
    old_weight = weight.data.detach().cpu().to(torch.float32)
    if int(old_weight.shape[0]) != int(map_cpu.shape[0]):
        raise ValueError(
            f"Cannot absorb residual output map with output {map_cpu.shape[0]} into weight {tuple(old_weight.shape)}."
        )
    with torch.no_grad():
        updated = map_cpu.T.matmul(old_weight)
        weight.data.copy_(updated.to(device=weight.device, dtype=weight.dtype))
        bias = getattr(module, "bias", None)
        bias_updated = bool(bias is not None)
        if bias is not None:
            old_bias = bias.data.detach().cpu().to(torch.float32)
            bias.data.copy_(old_bias.matmul(map_cpu).to(device=bias.device, dtype=bias.dtype))
    return {
        "kind": kind,
        "name": name,
        "map_shape": [int(item) for item in map_cpu.shape],
        "weight_shape": [int(item) for item in weight.shape],
        "bias_updated": bias_updated,
    }


def _split_reconstruction_rows(
    tensors: Sequence[Any],
    validation_fraction: float,
    min_validation_rows: int,
):
    rows = min(int(tensor.shape[0]) for tensor in tensors)
    if rows <= int(min_validation_rows):
        return None
    validation_rows = max(int(min_validation_rows), int(round(rows * float(validation_fraction))))
    validation_rows = min(validation_rows, rows - 1)
    if validation_rows <= 0:
        return None
    train_rows = rows - validation_rows
    if train_rows <= 0:
        return None
    train = [tensor[:train_rows] for tensor in tensors]
    validation = [tensor[train_rows:rows] for tensor in tensors]
    return {
        "rows": rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "train": train,
        "validation": validation,
    }


def _validation_reconstruction_gate(
    torch,
    before,
    after,
    target,
    min_relative_improvement: float,
    max_cosine_drop: float,
) -> dict[str, Any]:
    before_error = _relative_reconstruction_error(torch, before, target)
    after_error = _relative_reconstruction_error(torch, after, target)
    before_cosine = _mean_row_cosine(torch, before, target)
    after_cosine = _mean_row_cosine(torch, after, target)
    relative_improvement = (before_error - after_error) / max(before_error, 1e-12)
    accepted = relative_improvement >= float(min_relative_improvement) and after_cosine >= (
        before_cosine - float(max_cosine_drop)
    )
    reason = "accepted"
    if relative_improvement < float(min_relative_improvement):
        reason = "insufficient_relative_error_improvement"
    elif after_cosine < before_cosine - float(max_cosine_drop):
        reason = "cosine_regression"
    return {
        "accepted": bool(accepted),
        "reason": reason,
        "relative_error_before": before_error,
        "relative_error_after": after_error,
        "relative_error_improvement": before_error - after_error,
        "relative_error_improvement_ratio": relative_improvement,
        "min_relative_improvement": float(min_relative_improvement),
        "mean_cosine_before": before_cosine,
        "mean_cosine_after": after_cosine,
        "cosine_drop": before_cosine - after_cosine,
        "max_cosine_drop": float(max_cosine_drop),
    }


def _reconstruct_cross_tokenizer_layer(
    torch,
    source,
    candidate,
    source_layer,
    candidate_layer,
    source_layers: Sequence[Any] | None,
    candidate_layers: Sequence[Any] | None,
    candidate_layer_idx: int,
    source_layer_group: Sequence[int] | None,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_target: str,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
    donor_layer_fold_mode: str = "representative",
    source_layer_group_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    repair_report: dict[str, Any] = {}
    source_group = [int(item) for item in (source_layer_group or [])]
    source_attn_layer = source_layer
    source_mlp_layer = source_layer
    if source_layers is not None and source_group:
        source_attn_layer = source_layers[max(0, min(source_group[0], len(source_layers) - 1))]
        source_mlp_layer = source_layers[max(0, min(source_group[-1], len(source_layers) - 1))]

    attention_transactional = reconstruction_target in {"attention-all", "module-all", "all"}
    attention_state = _snapshot_module_state(candidate_layer) if attention_transactional else None

    if reconstruction_target in {"attention", "attention-all", "module-all", "both", "all"}:
        source_attn_norm = _get_layer_module(source_attn_layer, ("input_layernorm", "ln_1"))
        source_attn_norm_modules = _source_group_layer_modules(
            source_layers,
            source_group,
            lambda layer: _get_layer_module(layer, ("input_layernorm", "ln_1")),
        )
        if len(source_attn_norm_modules) != len(source_group):
            source_attn_norm_modules = {}
        candidate_attn_norm = _get_layer_module(candidate_layer, ("input_layernorm", "ln_1"))
        attn_linears = _attention_input_linears(candidate_layer)
        repair_report["attention_reconstruction"] = _reconstruct_cross_tokenizer_module_pair(
            torch,
            source,
            candidate,
            source_attn_norm,
            candidate_attn_norm,
            attn_linears,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            hidden_projection,
            canonical_fn,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
            fit_gate,
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
            max_tokens_per_layer,
            source_modules_by_layer=source_attn_norm_modules or None,
            source_layer_group=source_group,
            donor_layer_fold_mode=donor_layer_fold_mode,
            source_layer_group_weights=source_layer_group_weights,
        )
    else:
        repair_report["attention_reconstruction"] = {
            "skipped": True,
            "reason": "reconstruction_target_excludes_attention",
        }
    if reconstruction_target in {"attention-output", "attention-all", "module-output", "module-all", "all"}:
        source_attention_modules = _source_group_layer_modules(
            source_layers,
            source_group,
            _attention_module,
        )
        if len(source_attention_modules) != len(source_group):
            source_attention_modules = {}
        repair_report["attention_output_reconstruction"] = _reconstruct_cross_tokenizer_attention_output(
            torch,
            source,
            candidate,
            source_attn_layer,
            candidate_layer,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            hidden_projection,
            canonical_fn,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
            fit_gate,
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
            max_tokens_per_layer,
            source_attentions_by_layer=source_attention_modules or None,
            source_layer_group=source_group,
            donor_layer_fold_mode=donor_layer_fold_mode,
            source_layer_group_weights=source_layer_group_weights,
        )
    else:
        repair_report["attention_output_reconstruction"] = {
            "skipped": True,
            "reason": "reconstruction_target_excludes_attention_output",
        }

    if attention_state is not None:
        attention_input_applied = _reconstruction_mutation_applied(repair_report["attention_reconstruction"])
        attention_output_applied = _reconstruction_mutation_applied(
            repair_report["attention_output_reconstruction"]
        )
        accepted = bool(attention_input_applied and attention_output_applied)
        if not accepted:
            _restore_module_state(candidate_layer, attention_state)
        repair_report["attention_transaction"] = {
            "accepted": accepted,
            "reverted": not accepted,
            "input_applied": attention_input_applied,
            "output_applied": attention_output_applied,
        }

    mlp_transactional = reconstruction_target in {"mlp-all", "module-all", "all"}
    mlp_state = _snapshot_module_state(candidate_layer) if mlp_transactional else None

    if reconstruction_target in {"mlp", "mlp-all", "module-all", "both", "all"}:
        source_mlp_norm = _get_layer_module(source_mlp_layer, ("post_attention_layernorm", "ln_2"))
        source_mlp_norm_modules = _source_group_layer_modules(
            source_layers,
            source_group,
            lambda layer: _get_layer_module(layer, ("post_attention_layernorm", "ln_2")),
        )
        if len(source_mlp_norm_modules) != len(source_group):
            source_mlp_norm_modules = {}
        candidate_mlp_norm = _get_layer_module(candidate_layer, ("post_attention_layernorm", "ln_2"))
        mlp_linears = _mlp_input_linears(candidate_layer)
        repair_report["mlp_reconstruction"] = _reconstruct_cross_tokenizer_module_pair(
            torch,
            source,
            candidate,
            source_mlp_norm,
            candidate_mlp_norm,
            mlp_linears,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            hidden_projection,
            canonical_fn,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
            fit_gate,
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
            max_tokens_per_layer,
            source_modules_by_layer=source_mlp_norm_modules or None,
            source_layer_group=source_group,
            donor_layer_fold_mode=donor_layer_fold_mode,
            source_layer_group_weights=source_layer_group_weights,
        )
    else:
        repair_report["mlp_reconstruction"] = {
            "skipped": True,
            "reason": "reconstruction_target_excludes_mlp",
        }
    if reconstruction_target in {"mlp-output", "mlp-all", "module-output", "module-all", "all"}:
        repair_report["mlp_output_reconstruction"] = _reconstruct_cross_tokenizer_mlp_output(
            torch,
            source,
            candidate,
            source_mlp_layer,
            candidate_layer,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            hidden_projection,
            canonical_fn,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
            fit_gate,
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
            max_tokens_per_layer,
            source_layer_group=source_group,
            donor_layer_fold_mode=donor_layer_fold_mode,
            source_layer_group_weights=source_layer_group_weights,
        )
    else:
        repair_report["mlp_output_reconstruction"] = {
            "skipped": True,
            "reason": "reconstruction_target_excludes_mlp_output",
        }
    if mlp_state is not None:
        mlp_input_applied = _reconstruction_mutation_applied(repair_report["mlp_reconstruction"])
        mlp_output_applied = _reconstruction_mutation_applied(repair_report["mlp_output_reconstruction"])
        accepted = bool(mlp_input_applied and mlp_output_applied)
        if not accepted:
            _restore_module_state(candidate_layer, mlp_state)
        repair_report["mlp_transaction"] = {
            "accepted": accepted,
            "reverted": not accepted,
            "input_applied": mlp_input_applied,
            "output_applied": mlp_output_applied,
        }
    if reconstruction_target in {"consumer-input", "all"}:
        repair_report["consumer_input_reconstruction"] = _reconstruct_cross_tokenizer_consumer_input(
            torch,
            source,
            candidate,
            source_layers,
            candidate_layers,
            source_group,
            int(candidate_layer_idx),
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            hidden_projection,
            canonical_fn,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
            fit_gate,
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
            max_tokens_per_layer,
        )
    else:
        repair_report["consumer_input_reconstruction"] = {
            "skipped": True,
            "reason": "reconstruction_target_excludes_consumer_input",
        }
    if reconstruction_target in {"block-interface", "all"}:
        repair_report["block_interface_reconstruction"] = _reconstruct_cross_tokenizer_block_interface(
            torch,
            source,
            candidate,
            source_layers,
            candidate_layers,
            source_group,
            int(candidate_layer_idx),
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            hidden_projection,
            canonical_fn,
            ridge,
            reconstruction_alpha,
            reconstruction_objective,
            cosine_steps,
            cosine_lr,
            cosine_l2,
            fit_gate,
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
            max_tokens_per_layer,
        )
    else:
        repair_report["block_interface_reconstruction"] = {
            "skipped": True,
            "reason": "reconstruction_target_excludes_block_interface",
        }
    return repair_report


def _snapshot_module_state(module) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _restore_module_state(module, state: dict[str, Any]) -> None:
    module.load_state_dict(state, strict=True)


def _reconstruction_mutation_applied(report: dict[str, Any]) -> bool:
    if bool(report.get("skipped")) or bool(report.get("mutation_skipped")):
        return False
    if "linears_updated" in report:
        return bool(report.get("linears_updated"))
    return bool(report.get("output_projection") or report.get("applied"))


def _reconstruct_cross_tokenizer_block_interface(
    torch,
    source,
    candidate,
    source_layers: Sequence[Any] | None,
    candidate_layers: Sequence[Any] | None,
    source_layer_group: Sequence[int],
    candidate_layer_idx: int,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
) -> dict[str, Any]:
    if not source_layers or not candidate_layers or not source_layer_group:
        return {"skipped": True, "reason": "missing_source_or_candidate_layers"}
    source_consumer_idx = max(int(item) for item in source_layer_group) + 1
    candidate_consumer_idx = int(candidate_layer_idx) + 1
    report: dict[str, Any] = {
        "source_layer_group": [int(item) for item in source_layer_group],
        "source_consumer_layer": int(source_consumer_idx),
        "candidate_layer": int(candidate_layer_idx),
        "candidate_consumer_layer": int(candidate_consumer_idx),
    }
    if source_consumer_idx >= len(source_layers) or candidate_consumer_idx >= len(candidate_layers):
        return {**report, "skipped": True, "reason": "no_downstream_consumer_layer"}
    source_layer = source_layers[source_consumer_idx]
    consumer_layer = candidate_layers[candidate_consumer_idx]
    consumers = [
        (
            "attention",
            _get_layer_module(source_layer, ("input_layernorm", "ln_1")),
            _get_layer_module(consumer_layer, ("input_layernorm", "ln_1")),
            _attention_input_linears(consumer_layer),
        ),
        (
            "mlp",
            _get_layer_module(source_layer, ("post_attention_layernorm", "ln_2")),
            _get_layer_module(consumer_layer, ("post_attention_layernorm", "ln_2")),
            _mlp_input_linears(consumer_layer),
        ),
    ]
    source_pieces = []
    candidate_pieces = []
    consumer_reports = []
    consumer_linears: list[tuple[str, Any]] = []
    for consumer_name, source_module, candidate_module, linears in consumers:
        if source_module is None or candidate_module is None or not linears:
            consumer_reports.append(
                {
                    "consumer": consumer_name,
                    "skipped": True,
                    "reason": "missing_module_or_candidate_input_linears",
                }
            )
            continue
        source_part, candidate_part, matched_part = _collect_cross_tokenizer_module_activations(
            torch,
            source,
            candidate,
            source_module,
            candidate_module,
            source_tokenizer,
            candidate_tokenizer,
            paired_batches,
            canonical_fn,
            max_tokens_per_layer,
        )
        if int(matched_part) <= 0:
            consumer_reports.append(
                {
                    "consumer": consumer_name,
                    "skipped": True,
                    "reason": "no_cross_tokenizer_activation_pairs",
                    "matched_pairs": 0,
                }
            )
            continue
        source_pieces.append(source_part.detach().cpu().to(torch.float32))
        candidate_pieces.append(candidate_part.detach().cpu().to(torch.float32))
        consumer_linears.extend(linears)
        consumer_reports.append(
            {
                "consumer": consumer_name,
                "skipped": False,
                "matched_pairs": int(matched_part),
                "linears": [name for name, _module in linears],
            }
        )
    if not source_pieces or not candidate_pieces or not consumer_linears:
        return {
            **report,
            "skipped": True,
            "reason": "no_usable_downstream_consumer_pairs",
            "consumers": consumer_reports,
        }
    source_rows = torch.cat(source_pieces, dim=0)[:max_tokens_per_layer]
    candidate_rows = torch.cat(candidate_pieces, dim=0)[:max_tokens_per_layer]
    matched = min(int(source_rows.shape[0]), int(candidate_rows.shape[0]))
    if int(matched) <= 0:
        return {
            **report,
            "skipped": True,
            "reason": "no_cross_tokenizer_consumer_pairs",
            "consumers": consumer_reports,
        }
    source_rows = source_rows[:matched]
    candidate_rows = candidate_rows[:matched]
    source_projected = source_rows.to(device=hidden_projection.device, dtype=torch.float32).matmul(
        hidden_projection.to(dtype=torch.float32)
    ).detach().cpu()
    fit_report: dict[str, Any] = {}
    fit_candidate_rows = candidate_rows
    fit_source_rows = source_projected
    validation_candidate_rows = None
    validation_source_rows = None
    if fit_gate == "validation":
        split = _split_reconstruction_rows(
            [candidate_rows.to(torch.float32), source_projected.to(torch.float32)],
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
        )
        if split is None:
            return {
                **report,
                "skipped": False,
                "matched_pairs": int(matched),
                "consumers": consumer_reports,
                "linears_updated": [],
                "mutation_skipped": True,
                "fit_gate": {"mode": "validation", "accepted": False, "reason": "insufficient_rows"},
            }
        else:
            fit_candidate_rows, fit_source_rows = split["train"]
            validation_candidate_rows, validation_source_rows = split["validation"]
            fit_report["fit_gate"] = {
                "mode": "validation",
                "rows": int(split["rows"]),
                "train_rows": int(split["train_rows"]),
                "validation_rows": int(split["validation_rows"]),
            }
    fitted = _fit_activation_map(
        torch,
        fit_candidate_rows,
        fit_source_rows,
        ridge,
        reconstruction_alpha,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
    )
    if validation_candidate_rows is not None and validation_source_rows is not None:
        gate = _validation_reconstruction_gate(
            torch,
            validation_candidate_rows,
            validation_candidate_rows.matmul(fitted["map"]),
            validation_source_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
        )
        fit_report["fit_gate"] = {**fit_report["fit_gate"], **gate}
        if not gate["accepted"]:
            return {key: value for key, value in fitted.items() if key != "map"} | fit_report | report | {
                "matched_pairs": int(matched),
                "linears_updated": [],
                "mutation_skipped": True,
            }
    _absorb_input_map_into_linears(torch, consumer_linears, fitted["map"])
    return {key: value for key, value in fitted.items() if key != "map"} | fit_report | report | {
        "skipped": False,
        "matched_pairs": int(matched),
        "consumers": consumer_reports,
        "linears_updated": [name for name, _module in consumer_linears],
    }


def _reconstruct_cross_tokenizer_consumer_input(
    torch,
    source,
    candidate,
    source_layers: Sequence[Any] | None,
    candidate_layers: Sequence[Any] | None,
    source_layer_group: Sequence[int],
    candidate_layer_idx: int,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
) -> dict[str, Any]:
    if not source_layers or not candidate_layers or not source_layer_group:
        return {"skipped": True, "reason": "missing_source_or_candidate_layers"}
    source_consumer_idx = max(int(item) for item in source_layer_group) + 1
    candidate_consumer_idx = int(candidate_layer_idx) + 1
    report: dict[str, Any] = {
        "source_consumer_layer": int(source_consumer_idx),
        "candidate_consumer_layer": int(candidate_consumer_idx),
    }
    if source_consumer_idx >= len(source_layers) or candidate_consumer_idx >= len(candidate_layers):
        return {**report, "skipped": True, "reason": "no_downstream_consumer_layer"}
    source_layer = source_layers[source_consumer_idx]
    candidate_layer = candidate_layers[candidate_consumer_idx]
    source_attn_norm = _get_layer_module(source_layer, ("input_layernorm", "ln_1"))
    candidate_attn_norm = _get_layer_module(candidate_layer, ("input_layernorm", "ln_1"))
    attn_linears = _attention_input_linears(candidate_layer)
    report["attention_consumer_reconstruction"] = _reconstruct_cross_tokenizer_module_pair(
        torch,
        source,
        candidate,
        source_attn_norm,
        candidate_attn_norm,
        attn_linears,
        source_tokenizer,
        candidate_tokenizer,
        paired_batches,
        hidden_projection,
        canonical_fn,
        ridge,
        reconstruction_alpha,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
        fit_gate,
        fit_gate_validation_fraction,
        fit_gate_min_validation_rows,
        fit_gate_min_relative_improvement,
        fit_gate_max_cosine_drop,
        max_tokens_per_layer,
    )
    source_mlp_norm = _get_layer_module(source_layer, ("post_attention_layernorm", "ln_2"))
    candidate_mlp_norm = _get_layer_module(candidate_layer, ("post_attention_layernorm", "ln_2"))
    mlp_linears = _mlp_input_linears(candidate_layer)
    report["mlp_consumer_reconstruction"] = _reconstruct_cross_tokenizer_module_pair(
        torch,
        source,
        candidate,
        source_mlp_norm,
        candidate_mlp_norm,
        mlp_linears,
        source_tokenizer,
        candidate_tokenizer,
        paired_batches,
        hidden_projection,
        canonical_fn,
        ridge,
        reconstruction_alpha,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
        fit_gate,
        fit_gate_validation_fraction,
        fit_gate_min_validation_rows,
        fit_gate_min_relative_improvement,
        fit_gate_max_cosine_drop,
        max_tokens_per_layer,
    )
    report["skipped"] = False
    return report


def _reconstruct_cross_tokenizer_module_pair(
    torch,
    source,
    candidate,
    source_module,
    candidate_module,
    candidate_linears,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
    source_modules_by_layer: dict[int, Any] | None = None,
    source_layer_group: Sequence[int] | None = None,
    donor_layer_fold_mode: str = "representative",
    source_layer_group_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    if candidate_module is None or not candidate_linears:
        return {
            "skipped": True,
            "reason": "missing_module_or_candidate_input_linears",
        }
    if source_module is None and not source_modules_by_layer:
        return {
            "skipped": True,
            "reason": "missing_module_or_candidate_input_linears",
        }
    source_rows, candidate_rows, matched = _collect_cross_tokenizer_module_activations(
        torch,
        source,
        candidate,
        source_module,
        candidate_module,
        source_tokenizer,
        candidate_tokenizer,
        paired_batches,
        canonical_fn,
        max_tokens_per_layer,
        source_modules_by_layer=source_modules_by_layer,
        source_layer_group=source_layer_group,
        donor_layer_fold_mode=donor_layer_fold_mode,
        source_layer_group_weights=source_layer_group_weights,
    )
    if int(matched) <= 0:
        return {
            "skipped": True,
            "reason": "no_cross_tokenizer_activation_pairs",
        }
    group_report = {}
    if source_modules_by_layer and source_layer_group:
        group_report = {
            "source_layer_group": [int(item) for item in source_layer_group],
            "donor_layer_fold_mode": donor_layer_fold_mode,
        }
        if source_layer_group_weights is not None:
            group_report["source_layer_group_weights"] = [float(item) for item in source_layer_group_weights]
    source_projected = source_rows.to(device=hidden_projection.device, dtype=torch.float32).matmul(
        hidden_projection.to(dtype=torch.float32)
    ).detach().cpu()
    fit_report: dict[str, Any] = {}
    fit_candidate_rows = candidate_rows
    fit_source_rows = source_projected
    validation_candidate_rows = None
    validation_source_rows = None
    if fit_gate == "validation":
        split = _split_reconstruction_rows(
            [candidate_rows.to(torch.float32), source_projected.to(torch.float32)],
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
        )
        if split is None:
            return {
                **group_report,
                "skipped": False,
                "matched_pairs": int(matched),
                "linears_updated": [],
                "mutation_skipped": True,
                "fit_gate": {"mode": "validation", "accepted": False, "reason": "insufficient_rows"},
            }
        else:
            fit_candidate_rows, fit_source_rows = split["train"]
            validation_candidate_rows, validation_source_rows = split["validation"]
            fit_report["fit_gate"] = {
                "mode": "validation",
                "rows": int(split["rows"]),
                "train_rows": int(split["train_rows"]),
                "validation_rows": int(split["validation_rows"]),
            }
    fitted = _fit_activation_map(
        torch,
        fit_candidate_rows,
        fit_source_rows,
        ridge,
        reconstruction_alpha,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
    )
    if validation_candidate_rows is not None and validation_source_rows is not None:
        gate = _validation_reconstruction_gate(
            torch,
            validation_candidate_rows,
            validation_candidate_rows.matmul(fitted["map"]),
            validation_source_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
        )
        fit_report["fit_gate"] = {**fit_report["fit_gate"], **gate}
        if not gate["accepted"]:
            return {key: value for key, value in fitted.items() if key != "map"} | fit_report | group_report | {
                "matched_pairs": int(matched),
                "linears_updated": [],
                "mutation_skipped": True,
            }
    _absorb_input_map_into_linears(torch, candidate_linears, fitted["map"])
    return {key: value for key, value in fitted.items() if key != "map"} | fit_report | group_report | {
        "matched_pairs": int(matched),
        "linears_updated": [name for name, _module in candidate_linears],
    }


def _reconstruct_cross_tokenizer_attention_output(
    torch,
    source,
    candidate,
    source_layer,
    candidate_layer,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
    source_attentions_by_layer: dict[int, Any] | None = None,
    source_layer_group: Sequence[int] | None = None,
    donor_layer_fold_mode: str = "representative",
    source_layer_group_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    spec = _attention_projection_spec(candidate_layer)
    source_attention = _attention_module(source_layer)
    if spec is None or (source_attention is None and not source_attentions_by_layer):
        return {"skipped": True, "reason": "unsupported_attention_output_shape"}

    source_output, hidden, candidate_output, matched = _collect_cross_tokenizer_attention_output_calibration(
        torch,
        source,
        candidate,
        source_attention,
        spec,
        source_tokenizer,
        candidate_tokenizer,
        paired_batches,
        canonical_fn,
        max_tokens_per_layer,
        source_attentions_by_layer=source_attentions_by_layer,
        source_layer_group=source_layer_group,
        donor_layer_fold_mode=donor_layer_fold_mode,
        source_layer_group_weights=source_layer_group_weights,
    )
    if int(matched) <= 0:
        return {"skipped": True, "reason": "no_cross_tokenizer_attention_output_pairs"}
    group_report = {}
    if source_attentions_by_layer and source_layer_group:
        group_report = {
            "source_layer_group": [int(item) for item in source_layer_group],
            "donor_layer_fold_mode": donor_layer_fold_mode,
        }
        if source_layer_group_weights is not None:
            group_report["source_layer_group_weights"] = [float(item) for item in source_layer_group_weights]
    source_projected = source_output.to(device=hidden_projection.device, dtype=torch.float32).matmul(
        hidden_projection.to(dtype=torch.float32)
    ).detach().cpu()
    rows = min(int(source_projected.shape[0]), int(hidden.shape[0]), int(candidate_output.shape[0]))
    source_rows = source_projected[:rows].to(torch.float32)
    hidden_rows = hidden[:rows].to(torch.float32)
    candidate_rows = candidate_output[:rows].to(torch.float32)
    fit_source_rows = source_rows
    fit_hidden_rows = hidden_rows
    validation_source_rows = None
    validation_hidden_rows = None
    validation_candidate_rows = None
    fit_report: dict[str, Any] = {}
    if fit_gate == "validation":
        split = _split_reconstruction_rows(
            [source_rows, hidden_rows, candidate_rows],
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
        )
        if split is None:
            return {
                **group_report,
                "skipped": False,
                "matched_pairs": int(matched),
                "tokens": int(rows),
                "output_projection": spec["output_name"],
                "mutation_skipped": True,
                "fit_gate": {"mode": "validation", "accepted": False, "reason": "insufficient_rows"},
            }
        else:
            fit_source_rows, fit_hidden_rows, _fit_candidate_rows = split["train"]
            validation_source_rows, validation_hidden_rows, validation_candidate_rows = split["validation"]
            fit_report["fit_gate"] = {
                "mode": "validation",
                "rows": int(split["rows"]),
                "train_rows": int(split["train_rows"]),
                "validation_rows": int(split["validation_rows"]),
            }
    solved_report = _solve_linear_reconstruction(
        torch,
        fit_hidden_rows,
        fit_source_rows,
        ridge,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
    )
    solved = solved_report["map"]
    old_weight = spec["output"].weight.data.detach().cpu().to(torch.float32)
    solved_weight = solved.T.contiguous()
    if tuple(old_weight.shape) != tuple(solved_weight.shape):
        return {
            "skipped": True,
            "reason": "solved_weight_shape_mismatch",
            "old_weight_shape": list(old_weight.shape),
            "solved_weight_shape": list(solved_weight.shape),
        }
    blended_weight = (1.0 - float(reconstruction_alpha)) * old_weight + float(reconstruction_alpha) * solved_weight
    before = _relative_reconstruction_error(torch, candidate_rows, source_rows)
    solved_output = hidden_rows.matmul(solved)
    solved_error = _relative_reconstruction_error(torch, solved_output, source_rows)
    blended_output = hidden_rows.matmul(blended_weight.T)
    after = _relative_reconstruction_error(torch, blended_output, source_rows)
    before_cosine = _mean_row_cosine(torch, candidate_rows, source_rows)
    solved_cosine = _mean_row_cosine(torch, solved_output, source_rows)
    after_cosine = _mean_row_cosine(torch, blended_output, source_rows)
    if validation_source_rows is not None and validation_hidden_rows is not None and validation_candidate_rows is not None:
        gate = _validation_reconstruction_gate(
            torch,
            validation_candidate_rows,
            validation_hidden_rows.matmul(blended_weight.T),
            validation_source_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
        )
        fit_report["fit_gate"] = {**fit_report["fit_gate"], **gate}
        if not gate["accepted"]:
            return {
                "skipped": False,
                "matched_pairs": int(matched),
                "tokens": rows,
                "feature_size": int(hidden_rows.shape[1]),
                "hidden_size": int(source_rows.shape[1]),
                "output_projection": spec["output_name"],
                "reconstruction_alpha": float(reconstruction_alpha),
                "reconstruction_objective": reconstruction_objective,
                "relative_error_before": before,
                "relative_error_solved": solved_error,
                "relative_error_after": after,
                "mean_cosine_before": before_cosine,
                "mean_cosine_solved": solved_cosine,
                "mean_cosine_after": after_cosine,
                "solver": {key: value for key, value in solved_report.items() if key != "map"},
                "mutation_skipped": True,
                **fit_report,
                **group_report,
            }
    spec["output"].weight.data.copy_(blended_weight.to(device=spec["output"].weight.device, dtype=spec["output"].weight.dtype))
    return {
        "skipped": False,
        "matched_pairs": int(matched),
        "tokens": rows,
        "feature_size": int(hidden_rows.shape[1]),
        "hidden_size": int(source_rows.shape[1]),
        "output_projection": spec["output_name"],
        "reconstruction_alpha": float(reconstruction_alpha),
        "reconstruction_objective": reconstruction_objective,
        "relative_error_before": before,
        "relative_error_solved": solved_error,
        "relative_error_after": after,
        "mean_cosine_before": before_cosine,
        "mean_cosine_solved": solved_cosine,
        "mean_cosine_after": after_cosine,
        "solver": {key: value for key, value in solved_report.items() if key != "map"},
        **fit_report,
        **group_report,
    }


def _reconstruct_cross_tokenizer_mlp_output(
    torch,
    source,
    candidate,
    source_layer,
    candidate_layer,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    ridge: float,
    reconstruction_alpha: float,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens_per_layer: int,
    source_layer_group: Sequence[int] | None = None,
    donor_layer_fold_mode: str = "representative",
    source_layer_group_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    spec = _mlp_projection_spec(candidate_layer)
    if spec is None:
        return {"skipped": True, "reason": "unsupported_mlp_shape"}

    source_output, hidden, residual, candidate_output, matched = _collect_cross_tokenizer_mlp_output_calibration(
        torch,
        source,
        candidate,
        source_layer,
        candidate_layer,
        spec,
        source_tokenizer,
        candidate_tokenizer,
        paired_batches,
        canonical_fn,
        max_tokens_per_layer,
        source_layer_group=source_layer_group,
        donor_layer_fold_mode=donor_layer_fold_mode,
        source_layer_group_weights=source_layer_group_weights,
    )
    if int(matched) <= 0:
        return {"skipped": True, "reason": "no_cross_tokenizer_mlp_output_pairs"}
    group_report = {}
    if source_layer_group:
        group_report = {
            "source_layer_group": [int(item) for item in source_layer_group],
            "donor_layer_fold_mode": donor_layer_fold_mode,
        }
        if source_layer_group_weights is not None:
            group_report["source_layer_group_weights"] = [float(item) for item in source_layer_group_weights]
    source_projected = source_output.to(device=hidden_projection.device, dtype=torch.float32).matmul(
        hidden_projection.to(dtype=torch.float32)
    ).detach().cpu()
    rows = min(
        int(source_projected.shape[0]),
        int(hidden.shape[0]),
        int(residual.shape[0]),
        int(candidate_output.shape[0]),
    )
    source_rows = source_projected[:rows].to(torch.float32)
    hidden_rows = hidden[:rows].to(torch.float32)
    residual_rows = residual[:rows].to(torch.float32)
    candidate_rows = candidate_output[:rows].to(torch.float32)
    fit_source_rows = source_rows
    fit_hidden_rows = hidden_rows
    fit_residual_rows = residual_rows
    validation_source_rows = None
    validation_hidden_rows = None
    validation_residual_rows = None
    validation_candidate_rows = None
    fit_report: dict[str, Any] = {}
    if fit_gate == "validation":
        split = _split_reconstruction_rows(
            [source_rows, hidden_rows, residual_rows, candidate_rows],
            fit_gate_validation_fraction,
            fit_gate_min_validation_rows,
        )
        if split is None:
            return {
                **group_report,
                "skipped": False,
                "matched_pairs": int(matched),
                "tokens": int(rows),
                "output_projection": spec["output_name"],
                "mutation_skipped": True,
                "fit_gate": {"mode": "validation", "accepted": False, "reason": "insufficient_rows"},
            }
        else:
            fit_source_rows, fit_hidden_rows, fit_residual_rows, _fit_candidate_rows = split["train"]
            (
                validation_source_rows,
                validation_hidden_rows,
                validation_residual_rows,
                validation_candidate_rows,
            ) = split["validation"]
            fit_report["fit_gate"] = {
                "mode": "validation",
                "rows": int(split["rows"]),
                "train_rows": int(split["train_rows"]),
                "validation_rows": int(split["validation_rows"]),
            }
    target_mlp_output = fit_source_rows - fit_residual_rows
    solved_report = _solve_linear_reconstruction(
        torch,
        fit_hidden_rows,
        target_mlp_output,
        ridge,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
    )
    solved = solved_report["map"]
    old_weight = spec["output"].weight.data.detach().cpu().to(torch.float32)
    solved_weight = solved.T.contiguous()
    if tuple(old_weight.shape) != tuple(solved_weight.shape):
        return {
            "skipped": True,
            "reason": "solved_weight_shape_mismatch",
            "old_weight_shape": list(old_weight.shape),
            "solved_weight_shape": list(solved_weight.shape),
        }
    blended_weight = (1.0 - float(reconstruction_alpha)) * old_weight + float(reconstruction_alpha) * solved_weight
    before = _relative_reconstruction_error(torch, candidate_rows, source_rows)
    solved_output = residual_rows + hidden_rows.matmul(solved)
    solved_error = _relative_reconstruction_error(torch, solved_output, source_rows)
    blended_output = residual_rows + hidden_rows.matmul(blended_weight.T)
    after = _relative_reconstruction_error(torch, blended_output, source_rows)
    before_cosine = _mean_row_cosine(torch, candidate_rows, source_rows)
    solved_cosine = _mean_row_cosine(torch, solved_output, source_rows)
    after_cosine = _mean_row_cosine(torch, blended_output, source_rows)
    if (
        validation_source_rows is not None
        and validation_hidden_rows is not None
        and validation_residual_rows is not None
        and validation_candidate_rows is not None
    ):
        gate = _validation_reconstruction_gate(
            torch,
            validation_candidate_rows,
            validation_residual_rows + validation_hidden_rows.matmul(blended_weight.T),
            validation_source_rows,
            fit_gate_min_relative_improvement,
            fit_gate_max_cosine_drop,
        )
        fit_report["fit_gate"] = {**fit_report["fit_gate"], **gate}
        if not gate["accepted"]:
            return {
                "skipped": False,
                "matched_pairs": int(matched),
                "tokens": rows,
                "feature_size": int(hidden_rows.shape[1]),
                "hidden_size": int(source_rows.shape[1]),
                "output_projection": spec["output_name"],
                "reconstruction_alpha": float(reconstruction_alpha),
                "reconstruction_objective": reconstruction_objective,
                "relative_error_before": before,
                "relative_error_solved": solved_error,
                "relative_error_after": after,
                "mean_cosine_before": before_cosine,
                "mean_cosine_solved": solved_cosine,
                "mean_cosine_after": after_cosine,
                "solver": {key: value for key, value in solved_report.items() if key != "map"},
                "mutation_skipped": True,
                **fit_report,
                **group_report,
            }
    spec["output"].weight.data.copy_(blended_weight.to(device=spec["output"].weight.device, dtype=spec["output"].weight.dtype))
    return {
        "skipped": False,
        "matched_pairs": int(matched),
        "tokens": rows,
        "feature_size": int(hidden_rows.shape[1]),
        "hidden_size": int(source_rows.shape[1]),
        "output_projection": spec["output_name"],
        "reconstruction_alpha": float(reconstruction_alpha),
        "reconstruction_objective": reconstruction_objective,
        "relative_error_before": before,
        "relative_error_solved": solved_error,
        "relative_error_after": after,
        "mean_cosine_before": before_cosine,
        "mean_cosine_solved": solved_cosine,
        "mean_cosine_after": after_cosine,
        "solver": {key: value for key, value in solved_report.items() if key != "map"},
        **fit_report,
        **group_report,
    }


def _reconstruct_cross_tokenizer_final_interface(
    torch,
    source,
    candidate,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    hidden_projection,
    canonical_fn,
    ridge: float,
    reconstruction_alpha: float,
    fit_gate: str,
    fit_gate_validation_fraction: float,
    fit_gate_min_validation_rows: int,
    fit_gate_min_relative_improvement: float,
    fit_gate_max_cosine_drop: float,
    max_tokens: int,
) -> dict[str, Any]:
    source_norms = _final_norm_modules(source)
    candidate_norms = _final_norm_modules(candidate)
    output_head = candidate.get_output_embeddings()
    if not source_norms or not candidate_norms or output_head is None:
        return {
            "applied": False,
            "skipped": True,
            "reason": "missing_final_norm_or_output_head",
        }
    input_embeddings = candidate.get_input_embeddings()
    if (
        input_embeddings is not None
        and getattr(input_embeddings, "weight", None) is not None
        and getattr(output_head, "weight", None) is not None
        and input_embeddings.weight.data_ptr() == output_head.weight.data_ptr()
    ):
        return {
            "applied": False,
            "skipped": True,
            "reason": "tied_output_head_would_mutate_input_embeddings",
        }
    source_rows, candidate_rows, matched = _collect_cross_tokenizer_module_activations(
        torch,
        source,
        candidate,
        source_norms[-1],
        candidate_norms[-1],
        source_tokenizer,
        candidate_tokenizer,
        paired_batches,
        canonical_fn,
        max_tokens,
    )
    if int(matched) <= 0:
        return {
            "applied": False,
            "skipped": True,
            "reason": "no_cross_tokenizer_final_norm_pairs",
        }
    target_rows = source_rows.to(device=hidden_projection.device, dtype=torch.float32).matmul(
        hidden_projection.to(dtype=torch.float32)
    ).detach().cpu()
    rows = min(int(candidate_rows.shape[0]), int(target_rows.shape[0]))
    fit = _fit_gated_residual_output_map(
        torch,
        candidate_rows[:rows].to(torch.float32),
        target_rows[:rows].to(torch.float32),
        None,
        None,
        None,
        ridge,
        reconstruction_alpha,
        0,
        "final-interface",
        fit_gate,
        fit_gate_validation_fraction,
        fit_gate_min_validation_rows,
        fit_gate_min_relative_improvement,
        fit_gate_max_cosine_drop,
    )
    report = {
        "skipped": False,
        "matched_pairs": int(matched),
        "tokens": int(rows),
        **{key: value for key, value in fit.items() if key != "map"},
    }
    if fit.get("mutation_skipped"):
        return {**report, "applied": False}
    weight = getattr(output_head, "weight", None)
    if weight is None or int(weight.shape[1]) != int(fit["map"].shape[0]):
        return {
            **report,
            "applied": False,
            "mutation_skipped": True,
            "reason": "output_head_hidden_shape_mismatch",
        }
    with torch.no_grad():
        mapping = fit["map"].to(device=weight.device, dtype=torch.float32)
        updated = weight.data.to(torch.float32).matmul(mapping.t())
        weight.data.copy_(updated.to(dtype=weight.dtype))
    return {
        **report,
        "applied": True,
        "output_head_shape": [int(item) for item in weight.shape],
    }


def _collect_cross_tokenizer_module_activations(
    torch,
    source,
    candidate,
    source_module,
    candidate_module,
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    canonical_fn,
    max_tokens: int,
    source_modules_by_layer: dict[int, Any] | None = None,
    source_layer_group: Sequence[int] | None = None,
    donor_layer_fold_mode: str = "representative",
    source_layer_group_weights: Sequence[float] | None = None,
):
    source_pieces = []
    candidate_pieces = []
    captured: dict[str, Any] = {}
    captured_source_outputs: dict[int, Any] = {}
    source_group = [int(item) for item in (source_layer_group or [])]
    use_source_group = bool(source_modules_by_layer) and bool(source_group)

    def source_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["source"] = tensor.detach().to(torch.float32)

    def make_source_group_hook(layer_idx: int):
        def source_group_hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured_source_outputs[int(layer_idx)] = tensor.detach().to(torch.float32)

        return source_group_hook

    def candidate_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["candidate"] = tensor.detach().to(torch.float32)

    source_handles = []
    if use_source_group:
        for layer_idx in source_group:
            module = source_modules_by_layer.get(int(layer_idx)) if source_modules_by_layer is not None else None
            if module is not None:
                source_handles.append(module.register_forward_hook(make_source_group_hook(int(layer_idx))))
    elif source_module is not None:
        source_handles.append(source_module.register_forward_hook(source_hook))
    candidate_handle = candidate_module.register_forward_hook(candidate_hook)
    token_count = 0
    try:
        source.eval()
        candidate.eval()
        for source_batch, candidate_batch in paired_batches:
            if token_count >= max_tokens:
                break
            captured.clear()
            captured_source_outputs.clear()
            source(**_model_forward_inputs(source_batch), use_cache=False)
            candidate(**_model_forward_inputs(candidate_batch), use_cache=False)
            if "candidate" not in captured:
                continue
            if use_source_group:
                try:
                    source_values = _source_group_module_output_state(
                        torch,
                        captured_source_outputs,
                        source_group,
                        donor_layer_fold_mode,
                        source_layer_group_weights,
                    )
                except ValueError:
                    continue
            elif "source" in captured:
                source_values = captured["source"]
            else:
                continue
            pairs = _cross_tokenizer_span_pairs(
                torch,
                source_tokenizer,
                candidate_tokenizer,
                source_batch,
                candidate_batch,
                canonical_fn,
                max_pairs=max_tokens - token_count,
                shift_positions=False,
            )
            if int(pairs.numel()) <= 0:
                continue
            candidate_values = captured["candidate"]
            if source_values.dim() != 3 or candidate_values.dim() != 3:
                continue
            batch_idx = pairs[:, 0].to(dtype=torch.long, device=source_values.device)
            candidate_pos = pairs[:, 1].to(dtype=torch.long, device=candidate_values.device)
            source_pos = pairs[:, 2].to(dtype=torch.long, device=source_values.device)
            valid = (
                (batch_idx >= 0)
                & (batch_idx < source_values.shape[0])
                & (batch_idx.to(candidate_values.device) < candidate_values.shape[0])
                & (source_pos >= 0)
                & (source_pos < source_values.shape[1])
                & (candidate_pos >= 0)
                & (candidate_pos < candidate_values.shape[1])
            )
            if not bool(valid.any().detach().cpu()):
                continue
            batch_source = batch_idx[valid]
            source_selected = source_values[batch_source, source_pos[valid]].detach().cpu()
            batch_candidate = batch_idx.to(candidate_values.device)[valid.to(candidate_values.device)]
            candidate_selected = candidate_values[batch_candidate, candidate_pos[valid.to(candidate_values.device)]].detach().cpu()
            remaining = int(max_tokens) - token_count
            source_pieces.append(source_selected[:remaining])
            candidate_pieces.append(candidate_selected[:remaining])
            token_count += int(min(source_selected.shape[0], remaining))
    finally:
        for source_handle in source_handles:
            source_handle.remove()
        candidate_handle.remove()
    if not source_pieces or not candidate_pieces:
        return (
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.float32),
            0,
        )
    source_rows = torch.cat(source_pieces, dim=0)[:max_tokens]
    candidate_rows = torch.cat(candidate_pieces, dim=0)[:max_tokens]
    rows = min(int(source_rows.shape[0]), int(candidate_rows.shape[0]))
    return source_rows[:rows], candidate_rows[:rows], rows


def _collect_cross_tokenizer_attention_output_calibration(
    torch,
    source,
    candidate,
    source_attention,
    candidate_attention_spec: dict[str, Any],
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    canonical_fn,
    max_tokens: int,
    source_attentions_by_layer: dict[int, Any] | None = None,
    source_layer_group: Sequence[int] | None = None,
    donor_layer_fold_mode: str = "representative",
    source_layer_group_weights: Sequence[float] | None = None,
):
    source_pieces = []
    hidden_pieces = []
    candidate_output_pieces = []
    captured: dict[str, Any] = {}
    captured_source_outputs: dict[int, Any] = {}
    source_group = [int(item) for item in (source_layer_group or [])]
    use_source_group = bool(source_attentions_by_layer) and bool(source_group)

    def source_attention_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["source_attention_output"] = tensor.detach().to(torch.float32)

    def make_source_attention_hook(layer_idx: int):
        def source_group_attention_hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured_source_outputs[int(layer_idx)] = tensor.detach().to(torch.float32)

        return source_group_attention_hook

    def candidate_output_pre_hook(_module, inputs):
        captured["candidate_output_hidden"] = inputs[0].detach().to(torch.float32)

    def candidate_attention_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["candidate_attention_output"] = tensor.detach().to(torch.float32)

    source_handles = []
    if use_source_group:
        for layer_idx in source_group:
            module = source_attentions_by_layer.get(int(layer_idx)) if source_attentions_by_layer is not None else None
            if module is not None:
                source_handles.append(module.register_forward_hook(make_source_attention_hook(int(layer_idx))))
    elif source_attention is not None:
        source_handles.append(source_attention.register_forward_hook(source_attention_hook))
    hidden_handle = candidate_attention_spec["output"].register_forward_pre_hook(candidate_output_pre_hook)
    candidate_handle = candidate_attention_spec["attention"].register_forward_hook(candidate_attention_hook)
    token_count = 0
    try:
        source.eval()
        candidate.eval()
        for source_batch, candidate_batch in paired_batches:
            if token_count >= max_tokens:
                break
            captured.clear()
            captured_source_outputs.clear()
            source(**_model_forward_inputs(source_batch), use_cache=False)
            candidate(**_model_forward_inputs(candidate_batch), use_cache=False)
            required = {
                "candidate_output_hidden",
                "candidate_attention_output",
            }
            if any(key not in captured for key in required):
                continue
            if use_source_group:
                try:
                    source_values = _source_group_module_output_state(
                        torch,
                        captured_source_outputs,
                        source_group,
                        donor_layer_fold_mode,
                        source_layer_group_weights,
                    )
                except ValueError:
                    continue
            elif "source_attention_output" in captured:
                source_values = captured["source_attention_output"]
            else:
                continue
            pairs = _cross_tokenizer_span_pairs(
                torch,
                source_tokenizer,
                candidate_tokenizer,
                source_batch,
                candidate_batch,
                canonical_fn,
                max_pairs=max_tokens - token_count,
                shift_positions=False,
            )
            if int(pairs.numel()) <= 0:
                continue
            hidden_values = captured["candidate_output_hidden"]
            output_values = captured["candidate_attention_output"]
            if source_values.dim() != 3 or hidden_values.dim() != 3 or output_values.dim() != 3:
                continue
            batch_idx = pairs[:, 0].to(dtype=torch.long, device=source_values.device)
            candidate_pos = pairs[:, 1].to(dtype=torch.long, device=hidden_values.device)
            source_pos = pairs[:, 2].to(dtype=torch.long, device=source_values.device)
            valid = (
                (batch_idx >= 0)
                & (batch_idx < source_values.shape[0])
                & (batch_idx.to(hidden_values.device) < hidden_values.shape[0])
                & (source_pos >= 0)
                & (source_pos < source_values.shape[1])
                & (candidate_pos >= 0)
                & (candidate_pos < hidden_values.shape[1])
                & (candidate_pos < output_values.shape[1])
            )
            if int(valid.detach().cpu().sum().item()) <= 0:
                continue
            batch_source = batch_idx[valid]
            source_selected = source_values[batch_source, source_pos[valid]].detach().cpu()
            valid_candidate = valid.to(hidden_values.device)
            batch_candidate = batch_idx.to(hidden_values.device)[valid_candidate]
            candidate_selected_pos = candidate_pos[valid_candidate]
            hidden_selected = hidden_values[batch_candidate, candidate_selected_pos].detach().cpu()
            output_selected = output_values[batch_candidate, candidate_selected_pos].detach().cpu()
            remaining = int(max_tokens) - token_count
            take = min(int(source_selected.shape[0]), remaining)
            if take <= 0:
                break
            source_pieces.append(source_selected[:take])
            hidden_pieces.append(hidden_selected[:take])
            candidate_output_pieces.append(output_selected[:take])
            token_count += take
    finally:
        for source_handle in source_handles:
            source_handle.remove()
        hidden_handle.remove()
        candidate_handle.remove()
    if not source_pieces or not hidden_pieces or not candidate_output_pieces:
        empty = torch.empty((0, 0), dtype=torch.float32)
        return empty, empty, empty, 0
    source_rows = torch.cat(source_pieces, dim=0)[:max_tokens]
    hidden_rows = torch.cat(hidden_pieces, dim=0)[:max_tokens]
    candidate_rows = torch.cat(candidate_output_pieces, dim=0)[:max_tokens]
    rows = min(int(source_rows.shape[0]), int(hidden_rows.shape[0]), int(candidate_rows.shape[0]))
    return source_rows[:rows], hidden_rows[:rows], candidate_rows[:rows], rows


def _collect_cross_tokenizer_mlp_output_calibration(
    torch,
    source,
    candidate,
    source_layer,
    candidate_layer,
    candidate_mlp_spec: dict[str, Any],
    source_tokenizer,
    candidate_tokenizer,
    paired_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    canonical_fn,
    max_tokens: int,
    source_layer_group: Sequence[int] | None = None,
    donor_layer_fold_mode: str = "representative",
    source_layer_group_weights: Sequence[float] | None = None,
):
    source_pieces = []
    hidden_pieces = []
    residual_pieces = []
    candidate_output_pieces = []
    captured: dict[str, Any] = {}
    source_group = [int(item) for item in (source_layer_group or [])]
    use_source_group = bool(source_group)

    def source_layer_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["source_output"] = tensor.detach().to(torch.float32)

    def candidate_mlp_hook(_module, inputs, output):
        captured["candidate_hidden"] = _mlp_hidden_features(
            torch,
            candidate_mlp_spec,
            inputs[0],
        ).detach().to(torch.float32)
        captured["candidate_mlp_output"] = output.detach().to(torch.float32)

    def candidate_layer_hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["candidate_layer_output"] = tensor.detach().to(torch.float32)

    source_handle = None if use_source_group else source_layer.register_forward_hook(source_layer_hook)
    mlp_handle = candidate_mlp_spec["mlp"].register_forward_hook(candidate_mlp_hook)
    candidate_handle = candidate_layer.register_forward_hook(candidate_layer_hook)
    token_count = 0
    try:
        source.eval()
        candidate.eval()
        for source_batch, candidate_batch in paired_batches:
            if token_count >= max_tokens:
                break
            captured.clear()
            source_outputs = source(
                **_model_forward_inputs(source_batch),
                use_cache=False,
                output_hidden_states=use_source_group,
            )
            candidate(**_model_forward_inputs(candidate_batch), use_cache=False)
            required = {
                "candidate_hidden",
                "candidate_mlp_output",
                "candidate_layer_output",
            }
            if any(key not in captured for key in required):
                continue
            if use_source_group:
                try:
                    source_values = _source_group_hidden_state(
                        torch,
                        source_outputs.hidden_states,
                        source_group,
                        donor_layer_fold_mode,
                        source_layer_group_weights,
                    ).detach().to(torch.float32)
                except (AttributeError, ValueError):
                    continue
            elif "source_output" in captured:
                source_values = captured["source_output"]
            else:
                continue
            pairs = _cross_tokenizer_span_pairs(
                torch,
                source_tokenizer,
                candidate_tokenizer,
                source_batch,
                candidate_batch,
                canonical_fn,
                max_pairs=max_tokens - token_count,
                shift_positions=False,
            )
            if int(pairs.numel()) <= 0:
                continue
            hidden_values = captured["candidate_hidden"]
            mlp_values = captured["candidate_mlp_output"]
            layer_values = captured["candidate_layer_output"]
            if (
                source_values.dim() != 3
                or hidden_values.dim() != 3
                or mlp_values.dim() != 3
                or layer_values.dim() != 3
            ):
                continue
            batch_idx = pairs[:, 0].to(dtype=torch.long, device=source_values.device)
            candidate_pos = pairs[:, 1].to(dtype=torch.long, device=hidden_values.device)
            source_pos = pairs[:, 2].to(dtype=torch.long, device=source_values.device)
            valid = (
                (batch_idx >= 0)
                & (batch_idx < source_values.shape[0])
                & (batch_idx.to(hidden_values.device) < hidden_values.shape[0])
                & (source_pos >= 0)
                & (source_pos < source_values.shape[1])
                & (candidate_pos >= 0)
                & (candidate_pos < hidden_values.shape[1])
                & (candidate_pos < mlp_values.shape[1])
                & (candidate_pos < layer_values.shape[1])
            )
            if int(valid.detach().cpu().sum().item()) <= 0:
                continue
            batch_source = batch_idx[valid]
            source_selected = source_values[batch_source, source_pos[valid]].detach().cpu()
            valid_candidate = valid.to(hidden_values.device)
            batch_candidate = batch_idx.to(hidden_values.device)[valid_candidate]
            candidate_selected_pos = candidate_pos[valid_candidate]
            hidden_selected = hidden_values[batch_candidate, candidate_selected_pos].detach().cpu()
            mlp_selected = mlp_values[batch_candidate, candidate_selected_pos].detach().cpu()
            layer_selected = layer_values[batch_candidate, candidate_selected_pos].detach().cpu()
            residual_selected = layer_selected - mlp_selected
            remaining = int(max_tokens) - token_count
            take = min(int(source_selected.shape[0]), remaining)
            if take <= 0:
                break
            source_pieces.append(source_selected[:take])
            hidden_pieces.append(hidden_selected[:take])
            residual_pieces.append(residual_selected[:take])
            candidate_output_pieces.append(layer_selected[:take])
            token_count += take
    finally:
        if source_handle is not None:
            source_handle.remove()
        mlp_handle.remove()
        candidate_handle.remove()
    if not source_pieces or not hidden_pieces or not residual_pieces or not candidate_output_pieces:
        empty = torch.empty((0, 0), dtype=torch.float32)
        return empty, empty, empty, empty, 0
    source_rows = torch.cat(source_pieces, dim=0)[:max_tokens]
    hidden_rows = torch.cat(hidden_pieces, dim=0)[:max_tokens]
    residual_rows = torch.cat(residual_pieces, dim=0)[:max_tokens]
    candidate_rows = torch.cat(candidate_output_pieces, dim=0)[:max_tokens]
    rows = min(
        int(source_rows.shape[0]),
        int(hidden_rows.shape[0]),
        int(residual_rows.shape[0]),
        int(candidate_rows.shape[0]),
    )
    return source_rows[:rows], hidden_rows[:rows], residual_rows[:rows], candidate_rows[:rows], rows


def _attention_module(layer):
    return (
        getattr(layer, "self_attn", None)
        or getattr(layer, "attention", None)
        or getattr(layer, "attn", None)
        or getattr(layer, "operator", None)
    )


def _attention_projection_spec(layer) -> dict[str, Any] | None:
    attention = _attention_module(layer)
    if attention is None:
        return None
    for name in ("o_proj", "out_proj", "dense", "c_proj"):
        output = getattr(attention, name, None)
        if _is_linear_like(output):
            return {
                "attention": attention,
                "output": output,
                "output_name": f"self_attn.{name}",
            }
    return None


def _mlp_projection_spec(layer) -> dict[str, Any] | None:
    mlp = getattr(layer, "mlp", None)
    feed_forward = getattr(layer, "feed_forward", None)
    if feed_forward is not None and all(hasattr(feed_forward, name) for name in ("w1", "w3", "w2")):
        return {
            "kind": "gated",
            "mlp": feed_forward,
            "gate": feed_forward.w1,
            "up": feed_forward.w3,
            "output": feed_forward.w2,
            "output_name": "feed_forward.w2",
        }
    if feed_forward is not None and all(hasattr(feed_forward, name) for name in ("c_fc", "c_proj")):
        return {
            "kind": "dense",
            "mlp": feed_forward,
            "input": feed_forward.c_fc,
            "output": feed_forward.c_proj,
            "output_name": "feed_forward.c_proj",
        }
    if mlp is None:
        return None
    if all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        return {
            "kind": "gated",
            "mlp": mlp,
            "gate": mlp.gate_proj,
            "up": mlp.up_proj,
            "output": mlp.down_proj,
            "output_name": "mlp.down_proj",
        }
    if all(hasattr(mlp, name) for name in ("c_fc", "c_proj")):
        return {
            "kind": "dense",
            "mlp": mlp,
            "input": mlp.c_fc,
            "output": mlp.c_proj,
            "output_name": "mlp.c_proj",
        }
    if all(hasattr(mlp, name) for name in ("dense_h_to_4h", "dense_4h_to_h")):
        return {
            "kind": "dense",
            "mlp": mlp,
            "input": mlp.dense_h_to_4h,
            "output": mlp.dense_4h_to_h,
            "output_name": "mlp.dense_4h_to_h",
        }
    if all(hasattr(mlp, name) for name in ("fc1", "fc2")):
        return {
            "kind": "dense",
            "mlp": mlp,
            "input": mlp.fc1,
            "output": mlp.fc2,
            "output_name": "mlp.fc2",
        }
    return None


def _mlp_hidden_features(torch, spec: dict[str, Any], inputs):
    module_inputs = inputs.to(device=spec["output"].weight.device, dtype=spec["output"].weight.dtype)
    if spec["kind"] == "gated":
        activation = getattr(spec["mlp"], "act_fn", None) or torch.nn.functional.silu
        return activation(spec["gate"](module_inputs)) * spec["up"](module_inputs)
    activation = getattr(spec["mlp"], "act_fn", None) or getattr(spec["mlp"], "act", None) or torch.nn.functional.gelu
    return activation(spec["input"](module_inputs))


def _collect_layer_output_activations(torch, model, layer, batches: Sequence[dict[str, Any]], max_tokens: int):
    pieces = []
    token_count = 0
    current_mask = None

    def hook(_module, _inputs, output):
        nonlocal token_count
        tensor = output[0] if isinstance(output, tuple) else output
        rows = _select_token_rows(torch, tensor.detach().to(torch.float32), current_mask)
        remaining = max_tokens - token_count
        if remaining > 0:
            pieces.append(rows[:remaining].detach().cpu())
            token_count += int(min(rows.shape[0], remaining))

    handle = layer.register_forward_hook(hook)
    was_training = model.training
    try:
        model.eval()
        for batch in batches:
            if token_count >= max_tokens:
                break
            current_mask = batch.get("attention_mask")
            model(**batch, use_cache=False)
    finally:
        handle.remove()
        if was_training:
            model.train()
    if not pieces:
        raise RuntimeError("No layer output activations were captured.")
    return torch.cat(pieces, dim=0)


def _collect_survivor_mlp_calibration(
    torch,
    model,
    layer,
    batches: Sequence[dict[str, Any]],
    max_tokens: int,
):
    spec = _mlp_projection_spec(layer)
    if spec is None:
        raise ValueError("Unsupported MLP shape.")
    hidden_pieces = []
    mlp_output_pieces = []
    layer_output_pieces = []
    token_count = 0
    current_mask = None

    def mlp_hook(_module, inputs, output):
        if token_count >= max_tokens:
            return
        hidden_rows = _select_token_rows(
            torch,
            _mlp_hidden_features(torch, spec, inputs[0]).detach().to(torch.float32),
            current_mask,
        )
        output_rows = _select_token_rows(torch, output.detach().to(torch.float32), current_mask)
        hidden_pieces.append(hidden_rows.detach().cpu())
        mlp_output_pieces.append(output_rows.detach().cpu())

    def layer_hook(_module, _inputs, output):
        nonlocal token_count
        tensor = output[0] if isinstance(output, tuple) else output
        rows = _select_token_rows(torch, tensor.detach().to(torch.float32), current_mask)
        remaining = max_tokens - token_count
        if remaining > 0:
            layer_output_pieces.append(rows[:remaining].detach().cpu())
            token_count += int(min(rows.shape[0], remaining))

    mlp_handle = spec["mlp"].register_forward_hook(mlp_hook)
    layer_handle = layer.register_forward_hook(layer_hook)
    was_training = model.training
    try:
        model.eval()
        for batch in batches:
            if token_count >= max_tokens:
                break
            current_mask = batch.get("attention_mask")
            model(**batch, use_cache=False)
    finally:
        mlp_handle.remove()
        layer_handle.remove()
        if was_training:
            model.train()
    if not hidden_pieces or not mlp_output_pieces or not layer_output_pieces:
        raise RuntimeError("No survivor MLP calibration activations were captured.")
    hidden = torch.cat(hidden_pieces, dim=0)[:max_tokens]
    mlp_output = torch.cat(mlp_output_pieces, dim=0)[:max_tokens]
    layer_output = torch.cat(layer_output_pieces, dim=0)[:max_tokens]
    rows = min(int(hidden.shape[0]), int(mlp_output.shape[0]), int(layer_output.shape[0]))
    hidden = hidden[:rows]
    mlp_output = mlp_output[:rows]
    layer_output = layer_output[:rows]
    residual = layer_output - mlp_output
    return hidden, residual, layer_output


def _is_linear_like(module) -> bool:
    weight = getattr(module, "weight", None)
    return weight is not None and getattr(weight, "dim", lambda: 0)() == 2


def _collect_module_activations(torch, model, module, batches: Sequence[dict[str, Any]], max_tokens: int):
    pieces = []
    token_count = 0
    current_mask = None

    def hook(_module, _inputs, output):
        nonlocal token_count
        tensor = output[0] if isinstance(output, tuple) else output
        rows = _select_token_rows(torch, tensor.detach().to(torch.float32), current_mask)
        remaining = max_tokens - token_count
        if remaining > 0:
            pieces.append(rows[:remaining].detach().cpu())
            token_count += int(min(rows.shape[0], remaining))

    handle = module.register_forward_hook(hook)
    was_training = model.training
    try:
        model.eval()
        for batch in batches:
            if token_count >= max_tokens:
                break
            current_mask = batch.get("attention_mask")
            model(**batch, use_cache=False)
    finally:
        handle.remove()
        if was_training:
            model.train()
    if not pieces:
        raise RuntimeError("No calibration activations were captured.")
    return torch.cat(pieces, dim=0)


def _select_token_rows(torch, tensor, attention_mask):
    values = tensor
    if values.dim() == 3:
        values = values.reshape(-1, values.shape[-1])
        if attention_mask is not None:
            mask = attention_mask.reshape(-1).detach().to(torch.bool).cpu()
            values = values.cpu()[mask]
            return values
    elif values.dim() == 1:
        values = values.unsqueeze(0)
    elif values.dim() != 2:
        values = values.reshape(-1, values.shape[-1])
    return values.cpu()


def _fit_activation_map(
    torch,
    candidate_features,
    source_features,
    ridge: float,
    reconstruction_alpha: float = 1.0,
    reconstruction_objective: str = "ridge",
    cosine_steps: int = 80,
    cosine_lr: float = 1e-2,
    cosine_l2: float = 1e-4,
) -> dict[str, Any]:
    rows = min(int(candidate_features.shape[0]), int(source_features.shape[0]))
    if rows <= 0:
        raise RuntimeError("Need at least one activation row to fit reconstruction.")
    candidate_rows = candidate_features[:rows].to(torch.float32)
    source_rows = source_features[:rows].to(torch.float32)
    if candidate_rows.shape[1] != source_rows.shape[1]:
        raise ValueError(
            f"Activation hidden-size mismatch: {candidate_rows.shape[1]} != {source_rows.shape[1]}"
        )
    before = _relative_reconstruction_error(torch, candidate_rows, source_rows)
    before_cosine = _mean_row_cosine(torch, candidate_rows, source_rows)
    solved_report = _solve_linear_reconstruction(
        torch,
        candidate_rows,
        source_rows,
        ridge,
        reconstruction_objective,
        cosine_steps,
        cosine_lr,
        cosine_l2,
    )
    solved = solved_report["map"]
    solved_error = _relative_reconstruction_error(torch, candidate_rows.matmul(solved), source_rows)
    solved_cosine = _mean_row_cosine(torch, candidate_rows.matmul(solved), source_rows)
    if reconstruction_alpha < 1.0:
        identity = torch.eye(solved.shape[0], dtype=solved.dtype, device=solved.device)
        solved = (1.0 - float(reconstruction_alpha)) * identity + float(reconstruction_alpha) * solved
    after = _relative_reconstruction_error(torch, candidate_rows.matmul(solved), source_rows)
    after_cosine = _mean_row_cosine(torch, candidate_rows.matmul(solved), source_rows)
    return {
        "map": solved,
        "tokens": rows,
        "hidden_size": int(candidate_rows.shape[1]),
        "reconstruction_alpha": float(reconstruction_alpha),
        "reconstruction_objective": reconstruction_objective,
        "relative_error_before": before,
        "relative_error_solved": solved_error,
        "relative_error_after": after,
        "mean_cosine_before": before_cosine,
        "mean_cosine_solved": solved_cosine,
        "mean_cosine_after": after_cosine,
        "solver": {key: value for key, value in solved_report.items() if key != "map"},
    }


def _solve_linear_reconstruction(
    torch,
    features,
    targets,
    ridge: float,
    reconstruction_objective: str,
    cosine_steps: int,
    cosine_lr: float,
    cosine_l2: float,
) -> dict[str, Any]:
    ridge_map = _ridge_solve_features(torch, features, targets, ridge)
    if reconstruction_objective == "ridge" or cosine_steps == 0:
        return {
            "map": ridge_map,
            "objective": "ridge",
            "ridge": float(ridge),
            "cosine_refined": False,
        }
    if reconstruction_objective != "cosine":
        raise ValueError("reconstruction_objective must be ridge or cosine.")
    refined, refine_report = _refine_linear_map_by_cosine(
        torch,
        features,
        targets,
        ridge_map,
        steps=cosine_steps,
        lr=cosine_lr,
        l2=cosine_l2,
    )
    return {
        "map": refined,
        "objective": "cosine",
        "ridge": float(ridge),
        "cosine_refined": True,
        **refine_report,
    }


def _refine_linear_map_by_cosine(torch, features, targets, initial_map, steps: int, lr: float, l2: float):
    features = torch.nan_to_num(features.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    targets = torch.nan_to_num(targets.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    matrix = torch.nn.Parameter(initial_map.detach().clone().to(torch.float32))
    anchor = initial_map.detach().clone().to(torch.float32)
    optimizer = torch.optim.AdamW([matrix], lr=float(lr), weight_decay=0.0)
    start_cosine = _mean_row_cosine(torch, features.matmul(matrix.detach()), targets)
    start_error = _relative_reconstruction_error(torch, features.matmul(matrix.detach()), targets)
    final_loss = None
    with torch.enable_grad():
        for _step in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            predicted = features.matmul(matrix)
            cosine_loss = 1.0 - torch.nn.functional.cosine_similarity(predicted, targets, dim=1, eps=1e-8).mean()
            if l2 > 0:
                cosine_loss = cosine_loss + float(l2) * torch.mean((matrix - anchor) ** 2)
            if not torch.isfinite(cosine_loss):
                break
            cosine_loss.backward()
            optimizer.step()
            final_loss = float(cosine_loss.detach().cpu())
    refined = matrix.detach()
    return refined, {
        "cosine_steps": int(steps),
        "cosine_lr": float(lr),
        "cosine_l2": float(l2),
        "cosine_loss_final": final_loss,
        "mean_cosine_initial": start_cosine,
        "mean_cosine_final": _mean_row_cosine(torch, features.matmul(refined), targets),
        "relative_error_initial": start_error,
        "relative_error_final": _relative_reconstruction_error(torch, features.matmul(refined), targets),
    }


def _relative_reconstruction_error(torch, predicted, target) -> float:
    return float((predicted - target).norm().detach().cpu() / target.norm().detach().cpu().clamp_min(1e-12))


def _mean_row_cosine(torch, predicted, target) -> float:
    predicted = torch.nan_to_num(predicted.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    target = torch.nan_to_num(target.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    return float(torch.nn.functional.cosine_similarity(predicted, target, dim=1, eps=1e-8).mean().detach().cpu())


def _absorb_input_map_into_linears(torch, linears: Sequence[tuple[str, Any]], mapping) -> None:
    map_cpu = mapping.detach().cpu().to(torch.float32)
    for _name, linear in linears:
        weight = linear.weight.data.detach().cpu().to(torch.float32)
        if _is_hf_conv1d_like(linear) and int(weight.shape[0]) == int(map_cpu.shape[0]):
            updated = map_cpu.matmul(weight)
        elif int(weight.shape[1]) == int(map_cpu.shape[0]):
            updated = weight.matmul(map_cpu.T)
        else:
            raise ValueError(
                f"Cannot absorb reconstruction map with input {map_cpu.shape[0]} into weight {tuple(weight.shape)}."
            )
        linear.weight.data.copy_(updated.to(device=linear.weight.device, dtype=linear.weight.dtype))


def _is_hf_conv1d_like(module: Any) -> bool:
    cls = module.__class__
    return cls.__name__ == "Conv1D" and "transformers" in str(getattr(cls, "__module__", ""))


def _recovery_loss(
    torch,
    student,
    teacher,
    input_ids,
    attention_mask,
    loss_mask,
    kd_weight: float,
    lm_weight: float,
    unlikelihood_weight: float,
    unlikelihood_window: int,
    entropy_weight: float,
    temperature: float,
    top_k_kd_weight: float = 0.0,
    top_k_kd_k: int = 32,
    first_token_top_k_kd_weight: float = 0.0,
    first_token_kd_weight: float = 0.0,
    first_token_lm_weight: float = 0.0,
):
    shift_student = torch.nan_to_num(
        student.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    shift_labels = input_ids[:, 1:].contiguous()
    if attention_mask is None:
        mask = torch.ones_like(shift_labels, dtype=torch.bool)
    else:
        mask = attention_mask[:, 1:].contiguous().bool()
    if loss_mask is not None:
        mask = mask & loss_mask[:, 1:].contiguous().bool()
        if int(mask.sum().detach().cpu().item()) <= 0:
            if attention_mask is None:
                mask = torch.ones_like(shift_labels, dtype=torch.bool)
            else:
                mask = attention_mask[:, 1:].contiguous().bool()
    shift_labels = shift_labels.masked_fill(~mask, -100)
    first_mask = _first_response_shift_mask(torch, loss_mask, attention_mask)
    if first_mask is not None:
        first_mask = first_mask & mask
    total = torch.zeros((), dtype=shift_student.dtype, device=shift_student.device)
    token_kl = None
    top_token_kl = None
    shift_teacher = None
    kd_vocab = None
    if kd_weight > 0 or first_token_kd_weight > 0 or top_k_kd_weight > 0 or first_token_top_k_kd_weight > 0:
        if teacher is None:
            raise ValueError("teacher output is required when teacher loss weights are positive.")
        shift_teacher = torch.nan_to_num(
            teacher.logits[:, :-1, :].contiguous().to(torch.float32),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        kd_vocab = min(int(shift_student.shape[-1]), int(shift_teacher.shape[-1]))
    if kd_weight > 0:
        student_logp = torch.nn.functional.log_softmax(shift_student[..., :kd_vocab] / temperature, dim=-1)
        teacher_p = torch.nn.functional.softmax(shift_teacher[..., :kd_vocab] / temperature, dim=-1)
        token_kl = torch.nn.functional.kl_div(student_logp, teacher_p, reduction="none").sum(dim=-1)
        kd = (token_kl * mask).sum() / mask.sum().clamp_min(1)
        kd = kd * (temperature**2)
        total = total + float(kd_weight) * kd
    if top_k_kd_weight > 0 or first_token_top_k_kd_weight > 0:
        top_token_kl = _same_tokenizer_top_k_kl(
            torch,
            shift_student,
            shift_teacher,
            kd_vocab,
            temperature,
            top_k_kd_k,
        )
    if top_k_kd_weight > 0:
        top_k_kd = (top_token_kl * mask).sum() / mask.sum().clamp_min(1)
        top_k_kd = top_k_kd * (temperature**2)
        total = total + float(top_k_kd_weight) * top_k_kd
    if first_token_kd_weight > 0:
        if token_kl is None:
            student_logp = torch.nn.functional.log_softmax(shift_student[..., :kd_vocab] / temperature, dim=-1)
            teacher_p = torch.nn.functional.softmax(shift_teacher[..., :kd_vocab] / temperature, dim=-1)
            token_kl = torch.nn.functional.kl_div(student_logp, teacher_p, reduction="none").sum(dim=-1)
        if first_mask is not None and int(first_mask.sum().detach().cpu().item()) > 0:
            first_kd = (token_kl * first_mask).sum() / first_mask.sum().clamp_min(1)
            first_kd = first_kd * (temperature**2)
            total = total + float(first_token_kd_weight) * first_kd
    if first_token_top_k_kd_weight > 0 and first_mask is not None and int(first_mask.sum().detach().cpu().item()) > 0:
        first_top_k_kd = (top_token_kl * first_mask).sum() / first_mask.sum().clamp_min(1)
        first_top_k_kd = first_top_k_kd * (temperature**2)
        total = total + float(first_token_top_k_kd_weight) * first_top_k_kd
    token_ce = None
    if lm_weight > 0 or first_token_lm_weight > 0:
        token_ce = torch.nn.functional.cross_entropy(
            shift_student.view(-1, shift_student.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
    if lm_weight > 0:
        lm = (token_ce * mask).sum() / mask.sum().clamp_min(1)
        total = total + float(lm_weight) * lm
    if first_token_lm_weight > 0 and first_mask is not None and int(first_mask.sum().detach().cpu().item()) > 0:
        first_lm = (token_ce * first_mask).sum() / first_mask.sum().clamp_min(1)
        total = total + float(first_token_lm_weight) * first_lm
    if unlikelihood_weight > 0:
        unlikelihood = _recent_token_unlikelihood_loss(
            torch,
            shift_student,
            input_ids,
            shift_labels,
            mask,
            window=unlikelihood_window,
        )
        total = total + float(unlikelihood_weight) * unlikelihood
    if entropy_weight > 0:
        entropy_penalty = _masked_entropy_penalty(torch, shift_student, mask)
        total = total + float(entropy_weight) * entropy_penalty
    return total


def _same_tokenizer_top_k_kl(torch, shift_student, shift_teacher, kd_vocab: int, temperature: float, top_k: int):
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    k = min(int(top_k), int(kd_vocab))
    teacher_vocab = shift_teacher[..., :kd_vocab]
    student_vocab = shift_student[..., :kd_vocab]
    teacher_top_values, teacher_top_indices = torch.topk(teacher_vocab, k=k, dim=-1)
    student_top_values = student_vocab.gather(dim=-1, index=teacher_top_indices)
    student_logp = torch.nn.functional.log_softmax(student_top_values / temperature, dim=-1)
    teacher_p = torch.nn.functional.softmax(teacher_top_values / temperature, dim=-1)
    return torch.nn.functional.kl_div(student_logp, teacher_p, reduction="none").sum(dim=-1)


def _first_response_shift_mask(torch, loss_mask, attention_mask):
    if loss_mask is None or int(loss_mask.shape[-1]) <= 1:
        return None
    label_mask = loss_mask[:, 1:].contiguous().bool()
    if attention_mask is not None:
        label_mask = label_mask & attention_mask[:, 1:].contiguous().bool()
    if int(label_mask.sum().detach().cpu().item()) <= 0:
        return label_mask
    first_mask = torch.zeros_like(label_mask, dtype=torch.bool)
    has_response = label_mask.any(dim=1)
    first_indices = label_mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(label_mask.shape[0], device=label_mask.device)[has_response]
    first_mask[rows, first_indices[has_response]] = True
    return first_mask


def _last_response_shift_mask(torch, loss_mask, attention_mask):
    if loss_mask is None or int(loss_mask.shape[-1]) <= 1:
        return None
    label_mask = loss_mask[:, 1:].contiguous().bool()
    if attention_mask is not None:
        label_mask = label_mask & attention_mask[:, 1:].contiguous().bool()
    if int(label_mask.sum().detach().cpu().item()) <= 0:
        return label_mask
    last_mask = torch.zeros_like(label_mask, dtype=torch.bool)
    has_response = label_mask.any(dim=1)
    reversed_indices = label_mask.flip(dims=[1]).to(torch.int64).argmax(dim=1)
    last_indices = int(label_mask.shape[1]) - 1 - reversed_indices
    rows = torch.arange(label_mask.shape[0], device=label_mask.device)[has_response]
    last_mask[rows, last_indices[has_response]] = True
    return last_mask


def _cross_tokenizer_recovery_loss(
    torch,
    student,
    teacher,
    candidate_input_ids,
    candidate_attention_mask,
    candidate_loss_mask,
    source_attention_mask,
    candidate_anchor_idx,
    source_anchor_idx,
    anchor_kd_weight: float,
    lm_weight: float,
    first_token_lm_weight: float,
    response_end_lm_weight: float,
    hidden_kd_weight: float,
    span_kd_weight: float,
    span_hidden_kd_weight: float,
    projected_vocab_kd_weight: float,
    focused_projected_vocab_kd_weight: float,
    projected_vocab_kd_mode: str,
    reference_kd_weight: float,
    unlikelihood_weight: float,
    unlikelihood_window: int,
    entropy_weight: float,
    entropy_aware_kd_weight: float,
    entropy_aware_kd_threshold: float,
    entropy_aware_kd_sharpness: float,
    hidden_projection,
    hidden_layer: str,
    hidden_kd_metric: str,
    span_pairs,
    projected_vocab_groups,
    projected_vocab_mask,
    projected_vocab_valid,
    reference_teacher,
    temperature: float,
    source_input_ids=None,
    token_kd_weight: float = 0.0,
    token_focus_spans=None,
    token_teacher_confidence_floor: float = 0.0,
    focused_lm_weight: float = 0.0,
    projected_vocab_weights=None,
):
    shift_student = torch.nan_to_num(
        student.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    shift_labels = candidate_input_ids[:, 1:].contiguous()
    if candidate_attention_mask is None:
        candidate_mask = torch.ones_like(shift_labels, dtype=torch.bool)
    else:
        candidate_mask = candidate_attention_mask[:, 1:].contiguous().bool()
    candidate_lm_mask = candidate_mask
    if candidate_loss_mask is not None:
        candidate_lm_mask = candidate_mask & candidate_loss_mask[:, 1:].contiguous().bool()
        if int(candidate_lm_mask.sum().detach().cpu().item()) <= 0:
            candidate_lm_mask = candidate_mask
    shift_labels = shift_labels.masked_fill(~candidate_lm_mask, -100)
    total = torch.zeros((), dtype=shift_student.dtype, device=shift_student.device)
    if anchor_kd_weight > 0:
        shift_teacher = torch.nan_to_num(
            teacher.logits[:, :-1, :].contiguous().to(torch.float32),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        if source_attention_mask is None:
            source_mask = torch.ones(shift_teacher.shape[:2], dtype=torch.bool, device=shift_teacher.device)
        else:
            source_mask = source_attention_mask[:, 1:].contiguous().bool()
        student_anchor_logits = shift_student.index_select(dim=-1, index=candidate_anchor_idx)
        teacher_anchor_logits = shift_teacher.index_select(dim=-1, index=source_anchor_idx)
        student_summary = _masked_mean_logits(torch, student_anchor_logits, candidate_mask)
        teacher_summary = _masked_mean_logits(torch, teacher_anchor_logits, source_mask)
        kd = _entropy_aware_kl_loss(
            torch,
            student_summary,
            teacher_summary,
            temperature,
            entropy_aware_kd_weight,
            entropy_aware_kd_threshold,
            entropy_aware_kd_sharpness,
        )
        total = total + float(anchor_kd_weight) * kd
    if hidden_kd_weight > 0:
        if hidden_projection is None and hidden_kd_metric in PROJECTED_HIDDEN_KD_METRICS:
            raise ValueError("hidden_projection is required when hidden_kd_weight is positive.")
        hidden = _cross_tokenizer_hidden_loss(
            torch,
            student,
            teacher,
            candidate_attention_mask,
            source_attention_mask,
            hidden_projection,
            hidden_layer,
            hidden_kd_metric,
        )
        total = total + float(hidden_kd_weight) * hidden
    if span_kd_weight > 0 and span_pairs is not None and int(span_pairs.numel()) > 0:
        span = _cross_tokenizer_span_loss(
            torch,
            shift_student,
            teacher,
            span_pairs,
            candidate_anchor_idx,
            source_anchor_idx,
            temperature,
            entropy_aware_kd_weight,
            entropy_aware_kd_threshold,
            entropy_aware_kd_sharpness,
        )
        total = total + float(span_kd_weight) * span
    if span_hidden_kd_weight > 0 and span_pairs is not None and int(span_pairs.numel()) > 0:
        if hidden_projection is None and hidden_kd_metric in PROJECTED_HIDDEN_KD_METRICS:
            raise ValueError("hidden_projection is required when span_hidden_kd_weight is positive.")
        span_hidden = _cross_tokenizer_span_hidden_loss(
            torch,
            student,
            teacher,
            span_pairs,
            hidden_projection,
            hidden_layer,
            hidden_kd_metric,
        )
        total = total + float(span_hidden_kd_weight) * span_hidden
    if projected_vocab_kd_weight > 0 and span_pairs is not None and int(span_pairs.numel()) > 0:
        if (
            projected_vocab_groups is None
            or projected_vocab_mask is None
            or projected_vocab_weights is None
            or projected_vocab_valid is None
        ):
            raise ValueError("projected vocab maps are required when projected_vocab_kd_weight is positive.")
        projected = _cross_tokenizer_projected_vocab_loss(
            torch,
            shift_student,
            teacher,
            span_pairs,
            projected_vocab_groups,
            projected_vocab_mask,
            projected_vocab_weights,
            projected_vocab_valid,
            temperature,
            projected_vocab_kd_mode,
            entropy_aware_kd_weight,
            entropy_aware_kd_threshold,
            entropy_aware_kd_sharpness,
        )
        total = total + float(projected_vocab_kd_weight) * projected
    if token_kd_weight > 0 and token_focus_spans:
        if source_input_ids is None:
            raise ValueError("source_input_ids are required when token_kd_weight is positive.")
        token_kd = _cross_tokenizer_sequence_nll_loss(
            torch,
            shift_student,
            teacher,
            candidate_input_ids,
            source_input_ids,
            token_focus_spans,
            token_teacher_confidence_floor,
        )
        total = total + float(token_kd_weight) * token_kd
    if focused_projected_vocab_kd_weight > 0 and token_focus_spans:
        if (
            projected_vocab_groups is None
            or projected_vocab_mask is None
            or projected_vocab_weights is None
            or projected_vocab_valid is None
        ):
            raise ValueError("projected vocab maps are required when focused_projected_vocab_kd_weight is positive.")
        focus_pairs = _focus_spans_to_span_pairs(torch, token_focus_spans, device=shift_student.device)
        if int(focus_pairs.numel()) > 0:
            focused_projected = _cross_tokenizer_projected_vocab_loss(
                torch,
                shift_student,
                teacher,
                focus_pairs,
                projected_vocab_groups,
                projected_vocab_mask,
                projected_vocab_weights,
                projected_vocab_valid,
                temperature,
                projected_vocab_kd_mode,
                entropy_aware_kd_weight,
                entropy_aware_kd_threshold,
                entropy_aware_kd_sharpness,
            )
            total = total + float(focused_projected_vocab_kd_weight) * focused_projected
    if focused_lm_weight > 0 and token_focus_spans:
        focus_mask = _focus_spans_to_shift_mask(
            torch,
            token_focus_spans,
            shift_labels.shape,
            device=shift_student.device,
        )
        focus_mask = focus_mask & candidate_lm_mask
        if int(focus_mask.sum().detach().cpu().item()) > 0:
            focused_labels = shift_labels.masked_fill(~focus_mask, -100)
            focused_lm = torch.nn.functional.cross_entropy(
                shift_student.view(-1, shift_student.shape[-1]),
                focused_labels.view(-1),
                ignore_index=-100,
            )
            total = total + float(focused_lm_weight) * focused_lm
    if reference_kd_weight > 0:
        if reference_teacher is None:
            raise ValueError("reference_teacher is required when reference_kd_weight is positive.")
        reference_kd = _same_tokenizer_reference_loss(
            torch,
            shift_student,
            reference_teacher,
            candidate_lm_mask,
            temperature,
        )
        total = total + float(reference_kd_weight) * reference_kd
    if lm_weight > 0:
        lm = torch.nn.functional.cross_entropy(
            shift_student.view(-1, shift_student.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total = total + float(lm_weight) * lm
    if first_token_lm_weight > 0:
        first_mask = _first_response_shift_mask(torch, candidate_loss_mask, candidate_attention_mask)
        if first_mask is not None and int(first_mask.sum().detach().cpu().item()) > 0:
            first_labels = candidate_input_ids[:, 1:].contiguous().masked_fill(~first_mask, -100)
            first_lm = torch.nn.functional.cross_entropy(
                shift_student.view(-1, shift_student.shape[-1]),
                first_labels.view(-1),
                ignore_index=-100,
            )
            total = total + float(first_token_lm_weight) * first_lm
    if response_end_lm_weight > 0:
        end_mask = _last_response_shift_mask(torch, candidate_loss_mask, candidate_attention_mask)
        if end_mask is not None and int(end_mask.sum().detach().cpu().item()) > 0:
            end_labels = candidate_input_ids[:, 1:].contiguous().masked_fill(~end_mask, -100)
            end_lm = torch.nn.functional.cross_entropy(
                shift_student.view(-1, shift_student.shape[-1]),
                end_labels.view(-1),
                ignore_index=-100,
            )
            total = total + float(response_end_lm_weight) * end_lm
    if unlikelihood_weight > 0:
        unlikelihood = _recent_token_unlikelihood_loss(
            torch,
            shift_student,
            candidate_input_ids,
            shift_labels,
            candidate_lm_mask,
            window=unlikelihood_window,
        )
        total = total + float(unlikelihood_weight) * unlikelihood
    if entropy_weight > 0:
        entropy_penalty = _masked_entropy_penalty(torch, shift_student, candidate_lm_mask)
        total = total + float(entropy_weight) * entropy_penalty
    return total


def _cross_tokenizer_sequence_nll_loss(
    torch,
    shift_student,
    teacher,
    candidate_input_ids,
    source_input_ids,
    focus_spans,
    teacher_confidence_floor: float = 0.0,
):
    if not 0.0 <= float(teacher_confidence_floor) <= 1.0:
        raise ValueError("teacher_confidence_floor must be in [0, 1].")
    shift_teacher = torch.nan_to_num(
        teacher.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    student_logp = torch.nn.functional.log_softmax(shift_student, dim=-1)
    teacher_logp = torch.nn.functional.log_softmax(shift_teacher, dim=-1)
    losses = []
    weights = []
    for item in focus_spans or []:
        batch_idx = int(item["batch_idx"])
        candidate_positions = item["candidate_positions"].to(device=shift_student.device, dtype=torch.long)
        source_positions = item["source_positions"].to(device=shift_teacher.device, dtype=torch.long)
        if int(candidate_positions.numel()) <= 0 or int(source_positions.numel()) <= 0:
            continue
        candidate_labels = candidate_input_ids[batch_idx, candidate_positions + 1].to(
            device=shift_student.device,
            dtype=torch.long,
        )
        source_labels = source_input_ids[batch_idx, source_positions + 1].to(device=shift_teacher.device, dtype=torch.long)
        student_nll = -student_logp[batch_idx, candidate_positions].gather(
            dim=-1,
            index=candidate_labels.unsqueeze(-1),
        ).squeeze(-1)
        teacher_nll = -teacher_logp[batch_idx, source_positions].gather(
            dim=-1,
            index=source_labels.unsqueeze(-1),
        ).squeeze(-1)
        confidence = torch.exp(-teacher_nll.detach().mean()).clamp(
            min=float(teacher_confidence_floor),
            max=1.0,
        )
        losses.append(student_nll.mean() * confidence.to(device=shift_student.device, dtype=student_nll.dtype))
        weights.append(confidence.to(device=shift_student.device, dtype=student_nll.dtype))
    if not losses:
        return torch.zeros((), dtype=shift_student.dtype, device=shift_student.device)
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-6)


def _focus_spans_to_span_pairs(torch, focus_spans, device):
    pairs = []
    for item in focus_spans or []:
        batch_idx = int(item["batch_idx"])
        candidate_positions = item["candidate_positions"].detach().cpu().to(torch.long)
        source_positions = item["source_positions"].detach().cpu().to(torch.long)
        candidate_count = int(candidate_positions.numel())
        source_count = int(source_positions.numel())
        if candidate_count <= 0 or source_count <= 0:
            continue
        if candidate_count != source_count:
            continue
        for idx in range(candidate_count):
            pairs.append(
                (
                    batch_idx,
                    int(candidate_positions[idx].item()),
                    int(source_positions[idx].item()),
                )
            )
    if not pairs:
        return torch.empty((0, 3), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device)


def _focus_spans_to_shift_mask(torch, focus_spans, shape, device):
    rows, cols = int(shape[0]), int(shape[1])
    mask = torch.zeros((rows, cols), dtype=torch.bool, device=device)
    for item in focus_spans or []:
        batch_idx = int(item["batch_idx"])
        if batch_idx < 0 or batch_idx >= rows:
            continue
        candidate_positions = item["candidate_positions"].detach().cpu().to(torch.long)
        for position in candidate_positions.tolist():
            pos = int(position)
            if 0 <= pos < cols:
                mask[batch_idx, pos] = True
    return mask


def _cross_tokenizer_attention_influence_loss(torch, student, teacher, attention_pairs, layer: str = "last"):
    if layer not in {"last", "middle"}:
        raise ValueError("layer must be last or middle.")
    if attention_pairs is None or int(attention_pairs.numel()) == 0:
        logits = getattr(student, "logits", None)
        device = logits.device if logits is not None else "cpu"
        dtype = logits.dtype if logits is not None else torch.float32
        return torch.zeros((), dtype=dtype, device=device)
    student_attn = _select_attention_map(student, layer)
    teacher_attn = _select_attention_map(teacher, layer)
    if student_attn is None or teacher_attn is None:
        raise ValueError("attention_influence_weight requires model outputs with attentions.")
    student_map = _mean_attention_heads(torch, student_attn)
    teacher_map = _mean_attention_heads(torch, teacher_attn).detach()
    pairs = attention_pairs.to(device=student_map.device, dtype=torch.long)
    losses = []
    for batch_idx in torch.unique(pairs[:, 0]).detach().cpu().tolist():
        batch_idx = int(batch_idx)
        local = pairs[pairs[:, 0] == batch_idx]
        if int(local.shape[0]) <= 1:
            continue
        candidate_pos = local[:, 1].clamp(0, int(student_map.shape[-1]) - 1)
        source_pos = local[:, 2].to(device=teacher_map.device, dtype=torch.long).clamp(0, int(teacher_map.shape[-1]) - 1)
        candidate_rows = student_map[batch_idx].index_select(0, candidate_pos).index_select(1, candidate_pos)
        teacher_rows = teacher_map[batch_idx].index_select(0, source_pos).index_select(1, source_pos)
        candidate_rows = _row_probability_normalize(torch, candidate_rows)
        teacher_rows = _row_probability_normalize(torch, teacher_rows)
        losses.append(torch.nn.functional.smooth_l1_loss(candidate_rows, teacher_rows.to(device=candidate_rows.device)))
    if not losses:
        return torch.zeros((), dtype=student_map.dtype, device=student_map.device)
    return torch.stack(losses).mean()


def _select_attention_map(outputs, layer: str):
    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        return None
    if layer == "last":
        return attentions[-1]
    if layer == "middle":
        return attentions[len(attentions) // 2]
    raise ValueError("layer must be last or middle.")


def _mean_attention_heads(torch, attention):
    values = torch.nan_to_num(attention.to(torch.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if values.dim() == 4:
        values = values.mean(dim=1)
    if values.dim() != 3:
        raise ValueError("attention maps must be rank-3 or rank-4.")
    return values


def _row_probability_normalize(torch, values):
    clipped = torch.nan_to_num(values.to(torch.float32), nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    return clipped / clipped.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _cross_tokenizer_alm_chunk_loss(
    torch,
    shift_student,
    teacher,
    candidate_input_ids,
    source_input_ids,
    source_batch: dict[str, Any],
    candidate_batch: dict[str, Any],
    response_loss_mode: str,
    max_chunks: int,
    min_chars: int,
):
    if "offset_mapping" not in source_batch or "offset_mapping" not in candidate_batch:
        raise ValueError("ALM chunk loss requires offset_mapping batches.")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive.")
    if min_chars <= 0:
        raise ValueError("min_chars must be positive.")
    shift_teacher = torch.nan_to_num(
        teacher.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    student_logp = torch.nn.functional.log_softmax(shift_student, dim=-1)
    teacher_logp = torch.nn.functional.log_softmax(shift_teacher, dim=-1)
    source_offsets = source_batch["offset_mapping"].detach().cpu()
    candidate_offsets = candidate_batch["offset_mapping"].detach().cpu()
    source_mask = source_batch.get("attention_mask")
    candidate_mask = candidate_batch.get("attention_mask")
    source_mask_cpu = None if source_mask is None else source_mask.detach().cpu().bool()
    candidate_mask_cpu = None if candidate_mask is None else candidate_mask.detach().cpu().bool()
    texts = candidate_batch.get("raw_texts") or candidate_batch.get("texts") or []
    losses = []
    chunk_count = 0
    batch_count = min(int(candidate_input_ids.shape[0]), int(source_input_ids.shape[0]), len(texts))
    for batch_idx in range(batch_count):
        for start, end in _alm_text_chunks(str(texts[batch_idx]), response_loss_mode, max_chunks, min_chars):
            candidate_positions = _token_positions_for_char_span(
                torch,
                candidate_offsets,
                candidate_mask_cpu,
                batch_idx,
                start,
                end,
                max_label_position=int(candidate_input_ids.shape[1]) - 1,
            )
            source_positions = _token_positions_for_char_span(
                torch,
                source_offsets,
                source_mask_cpu,
                batch_idx,
                start,
                end,
                max_label_position=int(source_input_ids.shape[1]) - 1,
            )
            if int(candidate_positions.numel()) <= 0 or int(source_positions.numel()) <= 0:
                continue
            candidate_pos = candidate_positions.to(device=shift_student.device, dtype=torch.long)
            source_pos = source_positions.to(device=shift_teacher.device, dtype=torch.long)
            candidate_labels = candidate_input_ids[batch_idx, candidate_pos].to(device=shift_student.device, dtype=torch.long)
            source_labels = source_input_ids[batch_idx, source_pos].to(device=shift_teacher.device, dtype=torch.long)
            student_nll = -student_logp[batch_idx, candidate_pos - 1].gather(
                dim=-1,
                index=candidate_labels.unsqueeze(-1),
            ).squeeze(-1)
            teacher_nll = -teacher_logp[batch_idx, source_pos - 1].gather(
                dim=-1,
                index=source_labels.unsqueeze(-1),
            ).squeeze(-1)
            target = teacher_nll.detach().mean().to(device=student_nll.device, dtype=student_nll.dtype)
            losses.append(torch.nn.functional.smooth_l1_loss(student_nll.mean(), target))
            chunk_count += 1
            if chunk_count >= int(max_chunks):
                break
        if chunk_count >= int(max_chunks):
            break
    if not losses:
        return torch.zeros((), dtype=shift_student.dtype, device=shift_student.device), 0
    return torch.stack(losses).mean(), int(chunk_count)


def _alm_text_chunks(text: str, response_loss_mode: str, max_chunks: int, min_chars: int) -> list[tuple[int, int]]:
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive.")
    if min_chars <= 0:
        raise ValueError("min_chars must be positive.")
    scope_start = 0
    if response_loss_mode != "none":
        inferred = _infer_response_char_start(text, response_loss_mode)
        if inferred is not None:
            scope_start = max(0, min(int(inferred), len(text)))
            while scope_start < len(text) and text[scope_start].isspace():
                scope_start += 1
    spans = []
    for match in re.finditer(r"\S+", text[scope_start:]):
        start = scope_start + int(match.start())
        end = scope_start + int(match.end())
        if end - start >= int(min_chars):
            spans.append((start, end))
            if len(spans) >= int(max_chunks):
                break
    return spans


def _token_positions_for_char_span(
    torch,
    offsets,
    attention_mask,
    batch_idx: int,
    start: int,
    end: int,
    max_label_position: int,
):
    positions = []
    if end <= start:
        return torch.empty((0,), dtype=torch.long)
    seq_len = int(offsets.shape[1])
    limit = min(seq_len - 1, int(max_label_position))
    for pos in range(1, limit + 1):
        if attention_mask is not None and not bool(attention_mask[int(batch_idx), pos]):
            continue
        token_start = int(offsets[int(batch_idx), pos, 0])
        token_end = int(offsets[int(batch_idx), pos, 1])
        if token_end <= token_start:
            continue
        if token_start < end and token_end > start:
            positions.append(pos)
    if not positions:
        return torch.empty((0,), dtype=torch.long)
    return torch.tensor(positions, dtype=torch.long)


def _same_tokenizer_reference_loss(
    torch,
    shift_student,
    reference_teacher,
    candidate_mask,
    temperature: float,
):
    shift_reference = torch.nan_to_num(
        reference_teacher.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    student_logp = torch.nn.functional.log_softmax(shift_student / temperature, dim=-1)
    reference_logp = torch.nn.functional.log_softmax(shift_reference / temperature, dim=-1)
    reference_p = reference_logp.exp()
    token_kl = (reference_p * (reference_logp - student_logp)).sum(dim=-1) * (temperature**2)
    return _masked_mean_values(torch, token_kl, candidate_mask)


def _recent_token_unlikelihood_loss(
    torch,
    shift_logits,
    input_ids,
    shift_labels,
    mask,
    window: int = 16,
    exclude_current_label: bool = True,
):
    if window <= 0:
        raise ValueError("window must be positive.")
    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
    vocab_size = int(shift_logits.shape[-1])
    steps = int(shift_logits.shape[1])
    losses = []
    valid_masks = []
    for offset in range(int(window)):
        if offset >= steps:
            break
        neg_ids = torch.zeros_like(shift_labels)
        neg_ids[:, offset:] = input_ids[:, : steps - offset]
        valid = mask.clone()
        if offset > 0:
            valid[:, :offset] = False
        valid = valid & (shift_labels >= 0) & (neg_ids >= 0) & (neg_ids < vocab_size)
        if exclude_current_label:
            valid = valid & (neg_ids != shift_labels)
        if not bool(valid.any().detach().cpu()):
            continue
        neg_probs = probs.gather(dim=-1, index=neg_ids.clamp(0, vocab_size - 1).unsqueeze(-1)).squeeze(-1)
        losses.append(-torch.log1p(-neg_probs.clamp(max=1.0 - 1e-6)))
        valid_masks.append(valid)
    if not losses:
        return torch.zeros((), dtype=shift_logits.dtype, device=shift_logits.device)
    stacked_loss = torch.stack(losses, dim=0)
    stacked_mask = torch.stack(valid_masks, dim=0)
    return _masked_mean_values(torch, stacked_loss, stacked_mask)


def _masked_entropy_penalty(torch, shift_logits, mask):
    logp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    probs = logp.exp()
    entropy = -(probs * logp).sum(dim=-1)
    return -_masked_mean_values(torch, entropy, mask)


def _entropy_aware_kl_loss(
    torch,
    student_logits,
    teacher_logits,
    temperature: float,
    entropy_aware_kd_weight: float = 0.0,
    entropy_aware_kd_threshold: float = 0.55,
    entropy_aware_kd_sharpness: float = 8.0,
    vocab_valid=None,
):
    if int(student_logits.shape[-1]) != int(teacher_logits.shape[-1]):
        raise ValueError("student and teacher logits must share the final vocab dimension.")
    student = torch.nan_to_num(student_logits.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    teacher = torch.nan_to_num(teacher_logits.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    valid_count = int(student.shape[-1])
    if vocab_valid is not None:
        valid = vocab_valid.to(device=student.device, dtype=torch.bool)
        if int(valid.shape[-1]) != int(student.shape[-1]):
            raise ValueError("vocab_valid length must match the logits vocab dimension.")
        valid_count = int(valid.sum().detach().cpu().item())
        if valid_count <= 0:
            raise ValueError("At least one valid vocab row is required for KD.")
        view_shape = [1] * (student.dim() - 1) + [int(valid.shape[-1])]
        invalid = ~valid.view(*view_shape)
        student = student.masked_fill(invalid, -1e4)
        teacher = teacher.masked_fill(invalid, -1e4)
    student_logp = torch.nn.functional.log_softmax(student / temperature, dim=-1)
    teacher_logp = torch.nn.functional.log_softmax(teacher / temperature, dim=-1)
    teacher_p = teacher_logp.exp()
    forward = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1)
    if float(entropy_aware_kd_weight) == 0.0:
        return forward.mean() * (temperature**2)
    student_p = student_logp.exp()
    reverse = (student_p * (student_logp - teacher_logp)).sum(dim=-1)
    entropy = -(teacher_p * teacher_logp).sum(dim=-1)
    entropy_norm = entropy / math.log(max(2, valid_count))
    gate = torch.sigmoid((entropy_norm - float(entropy_aware_kd_threshold)) * float(entropy_aware_kd_sharpness))
    adaptive = gate * forward + (1.0 - gate) * reverse
    mixed = (1.0 - float(entropy_aware_kd_weight)) * forward + float(entropy_aware_kd_weight) * adaptive
    return mixed.mean() * (temperature**2)


def _entropy_aware_kl_from_log_probs(
    torch,
    student_logp,
    teacher_logp,
    temperature: float,
    entropy_aware_kd_weight: float = 0.0,
    entropy_aware_kd_threshold: float = 0.55,
    entropy_aware_kd_sharpness: float = 8.0,
):
    student_lp = torch.nan_to_num(student_logp.to(torch.float32), nan=-1e4, posinf=0.0, neginf=-1e4)
    teacher_lp = torch.nan_to_num(teacher_logp.to(torch.float32), nan=-1e4, posinf=0.0, neginf=-1e4)
    if int(student_lp.shape[-1]) != int(teacher_lp.shape[-1]):
        raise ValueError("student and teacher log-probabilities must share the final vocab dimension.")
    teacher_p = teacher_lp.exp()
    forward = (teacher_p * (teacher_lp - student_lp)).sum(dim=-1)
    if float(entropy_aware_kd_weight) == 0.0:
        return forward.mean() * (temperature**2)
    student_p = student_lp.exp()
    reverse = (student_p * (student_lp - teacher_lp)).sum(dim=-1)
    entropy = -(teacher_p * teacher_lp).sum(dim=-1)
    entropy_norm = entropy / math.log(max(2, int(student_lp.shape[-1])))
    gate = torch.sigmoid((entropy_norm - float(entropy_aware_kd_threshold)) * float(entropy_aware_kd_sharpness))
    adaptive = gate * forward + (1.0 - gate) * reverse
    mixed = (1.0 - float(entropy_aware_kd_weight)) * forward + float(entropy_aware_kd_weight) * adaptive
    return mixed.mean() * (temperature**2)


def _linear_cka_tensor(torch, first, second):
    x = torch.nan_to_num(first.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    y = torch.nan_to_num(second.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    if y.dim() != 2:
        y = y.reshape(-1, y.shape[-1])
    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("CKA rows must be aligned.")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = torch.linalg.norm(x.t().matmul(y), ord="fro") ** 2
    x_norm = torch.linalg.norm(x.t().matmul(x), ord="fro")
    y_norm = torch.linalg.norm(y.t().matmul(y), ord="fro")
    return cross / (x_norm * y_norm).clamp_min(1e-12)


def _row_gram_geometry_loss(torch, first, second, max_rows: int = 384):
    x = torch.nan_to_num(first.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    y = torch.nan_to_num(second.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    if y.dim() != 2:
        y = y.reshape(-1, y.shape[-1])
    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError("Gram geometry rows must be aligned.")
    row_count = int(x.shape[0])
    if row_count <= 1:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    if row_count > int(max_rows):
        idx = torch.linspace(0, row_count - 1, steps=int(max_rows), device=x.device).round().to(torch.long)
        x = x.index_select(dim=0, index=idx)
        y = y.index_select(dim=0, index=idx)
        row_count = int(max_rows)
    x_norm = x / x.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    y_norm = y / y.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    x_gram = x_norm.matmul(x_norm.t())
    y_gram = y_norm.matmul(y_norm.t())
    eye = torch.eye(row_count, dtype=torch.bool, device=x_gram.device)
    diff = (x_gram - y_gram).masked_fill(eye, 0.0)
    return diff.pow(2).sum() / max(1, row_count * (row_count - 1))


def _hidden_representation_loss(torch, student_rows, teacher_rows, hidden_projection, hidden_kd_metric: str):
    if hidden_kd_metric not in HIDDEN_KD_METRICS:
        raise ValueError(f"hidden_kd_metric must be {HIDDEN_KD_METRIC_DESCRIPTION}.")
    student = torch.nan_to_num(student_rows.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    teacher = torch.nan_to_num(teacher_rows.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    losses = []
    if hidden_kd_metric in PROJECTED_HIDDEN_KD_METRICS:
        if hidden_projection is None:
            raise ValueError("hidden_projection is required for projected hidden KD.")
        teacher_projected = teacher.matmul(hidden_projection.to(device=teacher.device, dtype=torch.float32))
        student_norm = _normalize_hidden_rows(torch, student)
        teacher_norm = _normalize_hidden_rows(torch, teacher_projected)
        cosine = torch.nn.functional.cosine_similarity(student_norm, teacher_norm, dim=-1, eps=1e-8)
        cosine_loss = 1.0 - cosine.mean()
        mse_loss = torch.nn.functional.mse_loss(student_norm, teacher_norm)
        losses.append(0.5 * cosine_loss + 0.5 * mse_loss)
    if hidden_kd_metric in {"cka", "both", "rsd", "geometry", "projected-geometry"}:
        losses.append(1.0 - _linear_cka_tensor(torch, student, teacher))
    if hidden_kd_metric in {"gram", "projected-gram", "geometry", "projected-geometry"}:
        losses.append(_row_gram_geometry_loss(torch, student, teacher))
    if hidden_kd_metric in {"rsd", "projected-rsd", "geometry", "projected-geometry"}:
        losses.append(_feature_decorrelation_loss(torch, student))
    return torch.stack(losses).mean()


def _feature_decorrelation_loss(torch, rows, max_features: int = 1024):
    values = torch.nan_to_num(rows.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if values.dim() != 2:
        values = values.reshape(-1, values.shape[-1])
    row_count = int(values.shape[0])
    feature_count = int(values.shape[1])
    if row_count <= 1 or feature_count <= 1:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    if feature_count > int(max_features):
        idx = torch.linspace(0, feature_count - 1, steps=int(max_features), device=values.device).round().to(torch.long)
        values = values.index_select(dim=1, index=idx)
        feature_count = int(values.shape[1])
    centered = values - values.mean(dim=0, keepdim=True)
    scale = centered.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = centered / scale
    corr = normalized.t().matmul(normalized) / float(row_count)
    eye = torch.eye(feature_count, dtype=torch.bool, device=corr.device)
    off_diag = corr.masked_fill(eye, 0.0)
    return off_diag.pow(2).sum() / max(1, feature_count * (feature_count - 1))


def _cross_tokenizer_span_loss(
    torch,
    shift_student,
    teacher,
    span_pairs,
    candidate_anchor_idx,
    source_anchor_idx,
    temperature: float,
    entropy_aware_kd_weight: float = 0.0,
    entropy_aware_kd_threshold: float = 0.55,
    entropy_aware_kd_sharpness: float = 8.0,
):
    shift_teacher = torch.nan_to_num(
        teacher.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    pair_idx = span_pairs.to(device=shift_student.device, dtype=torch.long)
    batch_idx = pair_idx[:, 0]
    candidate_pos = pair_idx[:, 1]
    source_pos = pair_idx[:, 2]
    student_logits = shift_student[batch_idx, candidate_pos].index_select(dim=-1, index=candidate_anchor_idx)
    teacher_logits = shift_teacher[batch_idx, source_pos].index_select(dim=-1, index=source_anchor_idx)
    return _entropy_aware_kl_loss(
        torch,
        student_logits,
        teacher_logits,
        temperature,
        entropy_aware_kd_weight,
        entropy_aware_kd_threshold,
        entropy_aware_kd_sharpness,
    )


def _cross_tokenizer_hidden_loss(
    torch,
    student,
    teacher,
    candidate_attention_mask,
    source_attention_mask,
    hidden_projection,
    hidden_layer: str,
    hidden_kd_metric: str = "projected",
):
    student_hidden = _select_hidden_state(student.hidden_states, hidden_layer)
    teacher_hidden = _select_hidden_state(teacher.hidden_states, hidden_layer)
    student_pooled = _masked_mean_hidden(torch, student_hidden, candidate_attention_mask)
    teacher_pooled = _masked_mean_hidden(torch, teacher_hidden, source_attention_mask)
    return _hidden_representation_loss(torch, student_pooled, teacher_pooled, hidden_projection, hidden_kd_metric)


def _same_tokenizer_hidden_loss(
    torch,
    student,
    teacher,
    attention_mask,
    loss_mask,
    hidden_projection,
    hidden_layer: str,
    hidden_kd_metric: str = "cka",
    hidden_token_mode: str = "tokens",
):
    if hidden_layer == "all":
        if not student.hidden_states or not teacher.hidden_states:
            raise ValueError("hidden states are required for hidden KD.")
        student_count = len(student.hidden_states)
        teacher_count = len(teacher.hidden_states)
        start = 1 if student_count > 1 and teacher_count > 1 else 0
        losses = []
        denom = max(1, student_count - 1)
        for student_idx in range(start, student_count):
            teacher_idx = round((student_idx / denom) * max(1, teacher_count - 1))
            teacher_idx = max(0, min(teacher_count - 1, int(teacher_idx)))
            losses.append(
                _same_tokenizer_hidden_state_loss(
                    torch,
                    student.hidden_states[student_idx],
                    teacher.hidden_states[teacher_idx],
                    attention_mask,
                    loss_mask,
                    hidden_projection,
                    hidden_kd_metric,
                    hidden_token_mode,
                )
            )
        return torch.stack(losses).mean()
    student_hidden = _select_hidden_state(student.hidden_states, hidden_layer)
    teacher_hidden = _select_hidden_state(teacher.hidden_states, hidden_layer)
    return _same_tokenizer_hidden_state_loss(
        torch,
        student_hidden,
        teacher_hidden,
        attention_mask,
        loss_mask,
        hidden_projection,
        hidden_kd_metric,
        hidden_token_mode,
    )


def _same_tokenizer_hidden_state_loss(
    torch,
    student_hidden,
    teacher_hidden,
    attention_mask,
    loss_mask,
    hidden_projection,
    hidden_kd_metric: str = "cka",
    hidden_token_mode: str = "tokens",
):
    if hidden_token_mode == "pooled":
        student_rows = _masked_mean_hidden(torch, student_hidden, attention_mask)
        teacher_rows = _masked_mean_hidden(torch, teacher_hidden, attention_mask)
        return _hidden_representation_loss(torch, student_rows, teacher_rows, hidden_projection, hidden_kd_metric)
    if attention_mask is None:
        mask = torch.ones(student_hidden.shape[:2], dtype=torch.bool, device=student_hidden.device)
    else:
        mask = attention_mask.to(device=student_hidden.device).bool()
    if hidden_token_mode == "response" and loss_mask is not None:
        response_mask = loss_mask.to(device=student_hidden.device).bool()
        scoped = mask & response_mask
        if int(scoped.sum().detach().cpu().item()) > 1:
            mask = scoped
    elif hidden_token_mode != "tokens":
        raise ValueError("hidden_token_mode must be tokens, response, or pooled.")
    if int(mask.sum().detach().cpu().item()) <= 1:
        return torch.zeros((), dtype=student_hidden.dtype, device=student_hidden.device)
    student_rows = student_hidden.to(torch.float32)[mask]
    teacher_rows = teacher_hidden.to(torch.float32)[mask.to(device=teacher_hidden.device)]
    return _hidden_representation_loss(torch, student_rows, teacher_rows, hidden_projection, hidden_kd_metric)


def _cross_tokenizer_span_hidden_loss(
    torch,
    student,
    teacher,
    span_pairs,
    hidden_projection,
    hidden_layer: str,
    hidden_kd_metric: str = "projected",
):
    student_hidden = _select_hidden_state(student.hidden_states, hidden_layer)
    teacher_hidden = _select_hidden_state(teacher.hidden_states, hidden_layer)
    pair_idx = span_pairs.to(device=student_hidden.device, dtype=torch.long)
    batch_idx = pair_idx[:, 0]
    candidate_pos = pair_idx[:, 1]
    source_pos = pair_idx[:, 2]
    student_rows = student_hidden[batch_idx, candidate_pos].to(torch.float32)
    teacher_rows = teacher_hidden[batch_idx, source_pos].to(torch.float32)
    return _hidden_representation_loss(torch, student_rows, teacher_rows, hidden_projection, hidden_kd_metric)


def _cross_tokenizer_dual_space_kd_loss(
    torch,
    student,
    teacher,
    span_pairs,
    source_to_candidate_projection,
    candidate_to_source_projection,
    source_head_weight,
    candidate_head_weight,
    source_anchor_idx,
    candidate_anchor_idx,
    hidden_layer: str,
    temperature: float,
):
    if source_to_candidate_projection is None or candidate_to_source_projection is None:
        raise ValueError("dual-space KD requires source->candidate and candidate->source projections.")
    if int(span_pairs.numel()) <= 0:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device)
    student_hidden = _select_hidden_state(student.hidden_states, hidden_layer)
    teacher_hidden = _select_hidden_state(teacher.hidden_states, hidden_layer)
    pair_idx = span_pairs.to(device=student_hidden.device, dtype=torch.long)
    batch_idx = pair_idx[:, 0]
    candidate_pos = pair_idx[:, 1]
    source_pos = pair_idx[:, 2].to(device=teacher_hidden.device)
    valid = (
        (batch_idx >= 0)
        & (batch_idx < student_hidden.shape[0])
        & (batch_idx.to(teacher_hidden.device) < teacher_hidden.shape[0])
        & (candidate_pos >= 0)
        & (candidate_pos < student_hidden.shape[1])
        & (source_pos >= 0)
        & (source_pos < teacher_hidden.shape[1])
        & (candidate_pos < student.logits.shape[1] - 1)
        & (source_pos < teacher.logits.shape[1] - 1)
    )
    if int(valid.detach().cpu().sum().item()) <= 0:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device)
    student_batch = batch_idx[valid]
    student_pos = candidate_pos[valid]
    teacher_valid = valid.to(device=teacher_hidden.device)
    teacher_batch = batch_idx.to(teacher_hidden.device)[teacher_valid]
    teacher_pos = source_pos[teacher_valid]
    student_rows = student_hidden[student_batch, student_pos].to(torch.float32)
    teacher_rows = teacher_hidden[teacher_batch, teacher_pos].to(torch.float32)

    source_anchor = source_anchor_idx.to(device=source_head_weight.device, dtype=torch.long)
    candidate_anchor = candidate_anchor_idx.to(device=candidate_head_weight.device, dtype=torch.long)
    source_head = source_head_weight.index_select(0, source_anchor).to(device=student_rows.device, dtype=torch.float32)
    candidate_head = candidate_head_weight.index_select(0, candidate_anchor).to(device=student_rows.device, dtype=torch.float32)
    s_to_t = candidate_to_source_projection.to(device=student_rows.device, dtype=torch.float32)
    t_to_s = source_to_candidate_projection.to(device=student_rows.device, dtype=torch.float32)

    student_as_teacher = student_rows.matmul(s_to_t).matmul(source_head.t())
    teacher_anchor_logits = (
        teacher.logits[:, :-1, :]
        .to(torch.float32)[teacher_batch, teacher_pos]
        .index_select(dim=-1, index=source_anchor.to(device=teacher.logits.device))
        .to(device=student_rows.device)
    )
    teacher_space_loss = _entropy_aware_kl_loss(
        torch,
        student_as_teacher,
        teacher_anchor_logits,
        temperature,
        0.0,
        0.55,
        8.0,
    )

    teacher_as_student = teacher_rows.to(device=student_rows.device).matmul(t_to_s).matmul(candidate_head.t())
    student_anchor_logits = (
        student.logits[:, :-1, :]
        .to(torch.float32)[student_batch, student_pos]
        .index_select(dim=-1, index=candidate_anchor.to(device=student.logits.device))
    )
    student_space_loss = _entropy_aware_kl_loss(
        torch,
        student_anchor_logits,
        teacher_as_student,
        temperature,
        0.0,
        0.55,
        8.0,
    )
    return 0.5 * (teacher_space_loss + student_space_loss)


def _cross_tokenizer_projected_vocab_loss(
    torch,
    shift_student,
    teacher,
    span_pairs,
    projected_vocab_groups,
    projected_vocab_mask,
    projected_vocab_weights,
    projected_vocab_valid,
    temperature: float,
    projected_vocab_kd_mode: str = "teacher-to-student",
    entropy_aware_kd_weight: float = 0.0,
    entropy_aware_kd_threshold: float = 0.55,
    entropy_aware_kd_sharpness: float = 8.0,
    teacher_group_mode: str = "first",
):
    if projected_vocab_kd_mode not in {"teacher-to-student", "student-to-teacher", "both"}:
        raise ValueError("projected_vocab_kd_mode must be teacher-to-student, student-to-teacher, or both.")
    if teacher_group_mode not in {"first", "weighted"}:
        raise ValueError("teacher_group_mode must be first or weighted.")
    shift_teacher = torch.nan_to_num(
        teacher.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    pair_idx = span_pairs.to(device=shift_student.device, dtype=torch.long)
    batch_idx = pair_idx[:, 0]
    candidate_pos = pair_idx[:, 1]
    source_pos = pair_idx[:, 2]
    student_logits = shift_student[batch_idx, candidate_pos]
    teacher_logits = shift_teacher[batch_idx, source_pos]
    losses = []
    if projected_vocab_kd_mode in {"teacher-to-student", "both"}:
        projected_teacher = _project_source_logits_to_candidate_vocab(
            torch,
            teacher_logits,
            projected_vocab_groups,
            projected_vocab_mask,
            projected_vocab_valid,
            projected_vocab_weights=projected_vocab_weights,
            group_mode=teacher_group_mode,
        )
        losses.append(
            _entropy_aware_kl_loss(
                torch,
                student_logits,
                projected_teacher,
                temperature,
                entropy_aware_kd_weight,
                entropy_aware_kd_threshold,
                entropy_aware_kd_sharpness,
                vocab_valid=projected_vocab_valid,
            )
        )
    if projected_vocab_kd_mode in {"student-to-teacher", "both"}:
        student_logp = torch.nn.functional.log_softmax(
            student_logits.masked_fill(~projected_vocab_valid.to(device=student_logits.device).view(1, -1), -1e4)
            / temperature,
            dim=-1,
        )
        student_probs = student_logp.exp()
        projected_student, source_valid = _project_candidate_probs_to_source_vocab(
            torch,
            student_probs,
            projected_vocab_groups,
            projected_vocab_mask,
            projected_vocab_valid,
            source_vocab_size=int(teacher_logits.shape[-1]),
            projected_vocab_weights=projected_vocab_weights,
        )
        teacher_valid_logits = teacher_logits.masked_fill(~source_valid.view(1, -1), -1e4)
        teacher_logp = torch.nn.functional.log_softmax(teacher_valid_logits / temperature, dim=-1)
        projected_student = projected_student.masked_fill(~source_valid.view(1, -1), 0.0)
        projected_student = projected_student / projected_student.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        projected_student_logp = projected_student.clamp_min(1e-12).log()
        source_idx = source_valid.nonzero(as_tuple=False).flatten()
        losses.append(
            _entropy_aware_kl_from_log_probs(
                torch,
                projected_student_logp.index_select(dim=-1, index=source_idx),
                teacher_logp.index_select(dim=-1, index=source_idx),
                temperature,
                entropy_aware_kd_weight,
                entropy_aware_kd_threshold,
                entropy_aware_kd_sharpness,
            )
        )
    return torch.stack(losses).mean()


def _tokenizer_first_byte_buckets(torch, tokenizer, device: str):
    other_bucket = 256
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    buckets = []
    for token_id in range(len(tokenizer)):
        bucket = other_bucket
        if int(token_id) not in special_ids:
            try:
                decoded = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
            except Exception:
                decoded = ""
            encoded = decoded.encode("utf-8", errors="ignore") if decoded else b""
            if encoded:
                bucket = int(encoded[0])
        buckets.append(bucket)
    return torch.tensor(buckets, dtype=torch.long, device=device)


def _first_byte_probs_from_logits(torch, logits, byte_buckets, temperature: float):
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if logits.dim() != 2:
        raise ValueError("byte KD logits must be a 2D tensor.")
    vocab = min(int(logits.shape[-1]), int(byte_buckets.numel()))
    if vocab <= 0:
        raise ValueError("Need at least one token bucket for byte KD.")
    values = torch.nan_to_num(
        logits[:, :vocab].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    buckets = byte_buckets[:vocab].to(device=values.device, dtype=torch.long)
    probs = torch.nn.functional.softmax(values / float(temperature), dim=-1)
    byte_probs = torch.zeros((int(values.shape[0]), 257), dtype=probs.dtype, device=values.device)
    byte_probs.scatter_add_(1, buckets.view(1, -1).expand(int(values.shape[0]), -1), probs)
    return byte_probs / byte_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _first_byte_head_from_output_embeddings(torch, output_weight, byte_buckets):
    """Build a fixed hidden-state byte probe by averaging output rows per byte bucket."""

    if output_weight.dim() != 2:
        raise ValueError("output embedding weight must be 2D.")
    vocab = min(int(output_weight.shape[0]), int(byte_buckets.numel()))
    if vocab <= 0:
        raise ValueError("Need at least one token bucket for byte hidden KD.")
    values = torch.nan_to_num(
        output_weight[:vocab].detach().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    buckets = byte_buckets[:vocab].to(device=values.device, dtype=torch.long)
    head = torch.zeros((257, int(values.shape[1])), dtype=torch.float32, device=values.device)
    counts = torch.zeros((257, 1), dtype=torch.float32, device=values.device)
    head.scatter_add_(0, buckets.view(-1, 1).expand(-1, int(values.shape[1])), values)
    counts.scatter_add_(0, buckets.view(-1, 1), torch.ones((vocab, 1), dtype=torch.float32, device=values.device))
    head = head / counts.clamp_min(1.0)
    return torch.nn.functional.normalize(head, dim=-1, eps=1e-8)


def _cross_tokenizer_hidden_first_byte_loss(
    torch,
    student,
    teacher,
    span_pairs,
    candidate_layer_idx: int,
    byte_probe_weight,
    source_byte_buckets,
    temperature: float,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if int(span_pairs.numel()) <= 0:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device)
    if byte_probe_weight.dim() != 2 or int(byte_probe_weight.shape[0]) != 257:
        raise ValueError("byte_probe_weight must have shape [257, hidden].")
    hidden = _hidden_state_after_decoder_layer(student.hidden_states, int(candidate_layer_idx)).to(torch.float32)
    if int(hidden.shape[-1]) != int(byte_probe_weight.shape[-1]):
        raise ValueError("byte probe hidden size must match candidate hidden size.")
    shift_teacher = torch.nan_to_num(
        teacher.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    pair_idx = span_pairs.to(device=hidden.device, dtype=torch.long)
    batch_idx = pair_idx[:, 0]
    candidate_pos = pair_idx[:, 1]
    source_pos = pair_idx[:, 2]
    valid = (
        (batch_idx >= 0)
        & (batch_idx < hidden.shape[0])
        & (batch_idx < shift_teacher.shape[0])
        & (candidate_pos >= 0)
        & (candidate_pos < hidden.shape[1])
        & (source_pos >= 0)
        & (source_pos < shift_teacher.shape[1])
    )
    if int(valid.detach().cpu().sum().item()) <= 0:
        return torch.zeros((), dtype=student.logits.dtype, device=student.logits.device)
    batch_idx = batch_idx[valid]
    candidate_pos = candidate_pos[valid]
    source_pos = source_pos[valid]
    student_rows = torch.nn.functional.normalize(hidden[batch_idx, candidate_pos], dim=-1, eps=1e-8)
    probe = torch.nn.functional.normalize(
        byte_probe_weight.to(device=hidden.device, dtype=torch.float32),
        dim=-1,
        eps=1e-8,
    )
    student_byte_logits = student_rows.matmul(probe.t())
    teacher_logits = shift_teacher.to(device=hidden.device)[batch_idx, source_pos]
    with torch.no_grad():
        teacher_byte_probs = _first_byte_probs_from_logits(torch, teacher_logits, source_byte_buckets, temperature)
    student_logp = torch.nn.functional.log_softmax(student_byte_logits / float(temperature), dim=-1)
    teacher_logp = teacher_byte_probs.clamp_min(1e-12).log()
    token_kl = (teacher_byte_probs * (teacher_logp - student_logp)).sum(dim=-1)
    return token_kl.mean().to(dtype=torch.float32) * (float(temperature) ** 2)


def _byte_hidden_probe_anchor_loss(torch, byte_probe_weight, byte_probe_anchor):
    if byte_probe_weight.shape != byte_probe_anchor.shape:
        raise ValueError("byte probe anchor shape must match byte probe weight shape.")
    weight = torch.nn.functional.normalize(byte_probe_weight.to(torch.float32), dim=-1, eps=1e-8)
    anchor = torch.nn.functional.normalize(
        byte_probe_anchor.to(device=byte_probe_weight.device, dtype=torch.float32),
        dim=-1,
        eps=1e-8,
    )
    return torch.nn.functional.mse_loss(weight, anchor)


def _cross_tokenizer_first_byte_loss(
    torch,
    shift_student,
    teacher,
    span_pairs,
    candidate_byte_buckets,
    source_byte_buckets,
    temperature: float,
):
    if int(span_pairs.numel()) <= 0:
        return torch.zeros((), dtype=shift_student.dtype, device=shift_student.device)
    shift_teacher = torch.nan_to_num(
        teacher.logits[:, :-1, :].contiguous().to(torch.float32),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )
    pair_idx = span_pairs.to(device=shift_student.device, dtype=torch.long)
    batch_idx = pair_idx[:, 0]
    candidate_pos = pair_idx[:, 1]
    source_pos = pair_idx[:, 2]
    student_logits = shift_student[batch_idx, candidate_pos]
    teacher_logits = shift_teacher[batch_idx, source_pos]
    student_byte_probs = _first_byte_probs_from_logits(torch, student_logits, candidate_byte_buckets, temperature)
    with torch.no_grad():
        teacher_byte_probs = _first_byte_probs_from_logits(torch, teacher_logits, source_byte_buckets, temperature)
    student_logp = student_byte_probs.clamp_min(1e-12).log()
    teacher_logp = teacher_byte_probs.clamp_min(1e-12).log()
    token_kl = (teacher_byte_probs * (teacher_logp - student_logp)).sum(dim=-1)
    return token_kl.mean() * (float(temperature) ** 2)


def _project_candidate_probs_to_source_vocab(
    torch,
    candidate_probs,
    projected_vocab_groups,
    projected_vocab_mask,
    projected_vocab_valid,
    source_vocab_size: int,
    projected_vocab_weights=None,
):
    if source_vocab_size <= 0:
        raise ValueError("source_vocab_size must be positive.")
    groups = projected_vocab_groups.to(device=candidate_probs.device, dtype=torch.long)
    group_mask = projected_vocab_mask.to(device=candidate_probs.device, dtype=candidate_probs.dtype)
    if projected_vocab_weights is None:
        group_weights = group_mask
    else:
        group_weights = projected_vocab_weights.to(device=candidate_probs.device, dtype=candidate_probs.dtype) * group_mask
    candidate_valid = projected_vocab_valid.to(device=candidate_probs.device, dtype=torch.bool)
    if int(groups.shape[0]) != int(candidate_probs.shape[-1]):
        raise ValueError("projected vocab group rows must match candidate vocab size.")
    valid_group_mask = group_mask * candidate_valid.to(dtype=candidate_probs.dtype).unsqueeze(-1)
    source_valid = torch.zeros(int(source_vocab_size), dtype=torch.bool, device=candidate_probs.device)
    valid_group_count = int(valid_group_mask.sum().detach().cpu().item())
    if valid_group_count <= 0:
        raise ValueError("At least one valid projected source token is required.")
    source_valid[groups.masked_select(valid_group_mask.bool()).clamp(0, int(source_vocab_size) - 1)] = True
    weights = group_weights * candidate_valid.to(dtype=candidate_probs.dtype).unsqueeze(-1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = weights * valid_group_mask.bool().to(dtype=candidate_probs.dtype)
    contributions = candidate_probs.masked_fill(~candidate_valid.view(1, -1), 0.0).unsqueeze(-1) * weights.unsqueeze(0)
    projected = torch.zeros(
        int(candidate_probs.shape[0]),
        int(source_vocab_size),
        dtype=candidate_probs.dtype,
        device=candidate_probs.device,
    )
    scatter_idx = groups.clamp(0, int(source_vocab_size) - 1).reshape(1, -1).expand(int(candidate_probs.shape[0]), -1)
    projected.scatter_add_(dim=1, index=scatter_idx, src=contributions.reshape(int(candidate_probs.shape[0]), -1))
    return projected, source_valid


def _project_source_logits_to_candidate_vocab(
    torch,
    source_logits,
    projected_vocab_groups,
    projected_vocab_mask,
    projected_vocab_valid,
    projected_vocab_weights=None,
    group_mode: str = "weighted",
):
    if group_mode not in {"weighted", "first"}:
        raise ValueError("group_mode must be weighted or first.")
    groups = projected_vocab_groups.to(device=source_logits.device, dtype=torch.long)
    mask = projected_vocab_mask.to(device=source_logits.device, dtype=source_logits.dtype)
    valid = projected_vocab_valid.to(device=source_logits.device, dtype=torch.bool)
    if group_mode == "first":
        first_valid = valid & mask[:, 0].to(dtype=torch.bool)
        gather_idx = groups[:, 0].clamp(0, int(source_logits.shape[-1]) - 1)
        projected = source_logits.index_select(dim=-1, index=gather_idx)
        return projected.masked_fill(~first_valid.unsqueeze(0), -1e4)
    if projected_vocab_weights is None:
        weights = mask
    else:
        weights = projected_vocab_weights.to(device=source_logits.device, dtype=source_logits.dtype) * mask
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = weights * mask.bool().to(dtype=source_logits.dtype)
    gather_idx = groups.unsqueeze(0).expand(int(source_logits.shape[0]), -1, -1)
    expanded = source_logits.unsqueeze(1).expand(-1, int(groups.shape[0]), -1)
    gathered = expanded.gather(dim=-1, index=gather_idx)
    projected = (gathered * weights.unsqueeze(0)).sum(dim=-1)
    return projected.masked_fill(~valid.unsqueeze(0), -1e4)


def _fit_lm_head_from_hidden_logits(
    torch,
    hidden_rows,
    target_logits,
    current_head,
    ridge: float,
    head_blend_alpha: float,
):
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not 0.0 <= float(head_blend_alpha) <= 1.0:
        raise ValueError("head_blend_alpha must be in [0, 1].")
    hidden = torch.nan_to_num(hidden_rows.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    targets = torch.nan_to_num(target_logits.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    old_head = torch.nan_to_num(current_head.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if hidden.dim() != 2 or targets.dim() != 2 or old_head.dim() != 2:
        raise ValueError("hidden_rows, target_logits, and current_head must be rank-2 tensors.")
    if int(hidden.shape[0]) != int(targets.shape[0]):
        raise ValueError("hidden_rows and target_logits must have the same row count.")
    if int(hidden.shape[1]) != int(old_head.shape[1]):
        raise ValueError("current_head hidden dimension must match hidden_rows.")
    if int(targets.shape[1]) != int(old_head.shape[0]):
        raise ValueError("target_logits vocab dimension must match current_head rows.")

    gram = hidden.t().matmul(hidden)
    if ridge > 0:
        gram = gram + float(ridge) * torch.eye(int(gram.shape[0]), dtype=gram.dtype, device=gram.device)
    rhs = hidden.t().matmul(targets)
    solved = torch.linalg.solve(gram, rhs).t().contiguous()
    if head_blend_alpha < 1.0:
        solved = (1.0 - float(head_blend_alpha)) * old_head + float(head_blend_alpha) * solved
    before = hidden.matmul(old_head.t())
    after = hidden.matmul(solved.t())
    before_error = _relative_reconstruction_error(torch, before, targets)
    after_error = _relative_reconstruction_error(torch, after, targets)
    return solved, {
        "rows": int(hidden.shape[0]),
        "hidden_size": int(hidden.shape[1]),
        "vocab_rows": int(targets.shape[1]),
        "ridge": float(ridge),
        "head_blend_alpha": float(head_blend_alpha),
        "relative_error_before": before_error,
        "relative_error_after": after_error,
    }


def _fit_lm_head_mean_shift(
    torch,
    hidden_rows,
    target_logits,
    current_head,
    ridge: float,
    head_blend_alpha: float,
):
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not 0.0 <= float(head_blend_alpha) <= 1.0:
        raise ValueError("head_blend_alpha must be in [0, 1].")
    hidden = torch.nan_to_num(hidden_rows.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    targets = torch.nan_to_num(target_logits.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    old_head = torch.nan_to_num(current_head.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if hidden.dim() != 2 or targets.dim() != 2 or old_head.dim() != 2:
        raise ValueError("hidden_rows, target_logits, and current_head must be rank-2 tensors.")
    if int(hidden.shape[0]) != int(targets.shape[0]):
        raise ValueError("hidden_rows and target_logits must have the same row count.")
    if int(hidden.shape[1]) != int(old_head.shape[1]):
        raise ValueError("current_head hidden dimension must match hidden_rows.")
    if int(targets.shape[1]) != int(old_head.shape[0]):
        raise ValueError("target_logits vocab dimension must match current_head rows.")

    before = hidden.matmul(old_head.t())
    hidden_mean = hidden.mean(dim=0)
    denom = hidden_mean.dot(hidden_mean).clamp_min(1e-12) + float(ridge)
    mean_error = targets.mean(dim=0) - before.mean(dim=0)
    shifted = old_head + mean_error.unsqueeze(1) * hidden_mean.unsqueeze(0) / denom
    if head_blend_alpha < 1.0:
        shifted = (1.0 - float(head_blend_alpha)) * old_head + float(head_blend_alpha) * shifted
    after = hidden.matmul(shifted.t())
    before_mean_error = (targets.mean(dim=0) - before.mean(dim=0)).abs().mean()
    after_mean_error = (targets.mean(dim=0) - after.mean(dim=0)).abs().mean()
    report = {
        "mode": "mean-shift",
        "rows": int(hidden.shape[0]),
        "hidden_size": int(hidden.shape[1]),
        "vocab_rows": int(targets.shape[1]),
        "ridge": float(ridge),
        "head_blend_alpha": float(head_blend_alpha),
        "relative_error_before": _relative_reconstruction_error(torch, before, targets),
        "relative_error_after": _relative_reconstruction_error(torch, after, targets),
        "mean_abs_logit_error_before": float(before_mean_error.detach().cpu().item()),
        "mean_abs_logit_error_after": float(after_mean_error.detach().cpu().item()),
        "hidden_mean_norm": float(hidden_mean.norm().detach().cpu().item()),
        "delta_norm": float((shifted - old_head).norm().detach().cpu().item()),
    }
    return shifted.contiguous(), report


def _fit_lm_head_answer_margin(
    torch,
    hidden_rows,
    gold_ids: Sequence[int],
    negative_ids: Sequence[Sequence[int]],
    current_head,
    ridge: float,
    head_blend_alpha: float,
    margin: float,
    steps: int,
):
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not 0.0 <= float(head_blend_alpha) <= 1.0:
        raise ValueError("head_blend_alpha must be in [0, 1].")
    if margin <= 0:
        raise ValueError("margin must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    hidden = torch.nan_to_num(hidden_rows.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    old_head = torch.nan_to_num(current_head.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if hidden.dim() != 2 or old_head.dim() != 2:
        raise ValueError("hidden_rows and current_head must be rank-2 tensors.")
    if int(hidden.shape[0]) != len(gold_ids) or int(hidden.shape[0]) != len(negative_ids):
        raise ValueError("hidden_rows, gold_ids, and negative_ids must have the same row count.")
    if int(hidden.shape[1]) != int(old_head.shape[1]):
        raise ValueError("current_head hidden dimension must match hidden_rows.")

    vocab_size = int(old_head.shape[0])
    rows: list[tuple[int, list[int]]] = []
    touched: set[int] = set()
    for gold_id, negatives in zip(gold_ids, negative_ids):
        gold = int(gold_id)
        if gold < 0 or gold >= vocab_size:
            rows.append((-1, []))
            continue
        filtered = []
        for negative_id in negatives:
            negative = int(negative_id)
            if negative < 0 or negative >= vocab_size or negative == gold or negative in filtered:
                continue
            filtered.append(negative)
        if filtered:
            touched.add(gold)
            touched.update(filtered)
        rows.append((gold, filtered))

    def _hinge_stats(head):
        losses = []
        violation_count = 0
        pair_count = 0
        for row_idx, (gold, negatives) in enumerate(rows):
            if gold < 0 or not negatives:
                continue
            h = hidden[row_idx]
            gold_logit = head[gold].dot(h)
            neg_tensor = torch.tensor(negatives, dtype=torch.long, device=head.device)
            neg_logits = head.index_select(0, neg_tensor).matmul(h)
            violations = torch.relu(float(margin) + neg_logits - gold_logit)
            losses.append(violations.mean())
            violation_count += int((violations > 0).sum().detach().cpu().item())
            pair_count += int(violations.numel())
        if not losses:
            return 0.0, 0, 0
        return float(torch.stack(losses).mean().detach().cpu().item()), int(violation_count), int(pair_count)

    before_loss, before_violations, pair_count = _hinge_stats(old_head)
    solved = old_head.clone()
    updates = 0
    for _step in range(int(steps)):
        for row_idx, (gold, negatives) in enumerate(rows):
            if gold < 0 or not negatives:
                continue
            h = hidden[row_idx]
            norm_sq = h.dot(h).clamp_min(1e-12)
            gold_logit = solved[gold].dot(h)
            neg_tensor = torch.tensor(negatives, dtype=torch.long, device=solved.device)
            neg_logits = solved.index_select(0, neg_tensor).matmul(h)
            violations = float(margin) + neg_logits - gold_logit
            active_mask = violations > 0
            if int(active_mask.sum().detach().cpu().item()) <= 0:
                continue
            active_neg = neg_tensor[active_mask]
            active_violations = violations[active_mask]
            active_count = int(active_neg.numel())
            step_size = active_violations.mean() / ((1.0 + 1.0 / max(1, active_count)) * norm_sq + float(ridge))
            delta = step_size * h
            solved[gold] = solved[gold] + delta
            solved[active_neg] = solved[active_neg] - delta.unsqueeze(0) / max(1, active_count)
            updates += active_count

    if head_blend_alpha < 1.0:
        solved = (1.0 - float(head_blend_alpha)) * old_head + float(head_blend_alpha) * solved
    after_loss, after_violations, _after_pair_count = _hinge_stats(solved)
    return solved.contiguous(), {
        "mode": "answer-margin",
        "rows": int(hidden.shape[0]),
        "hidden_size": int(hidden.shape[1]),
        "vocab_rows": int(old_head.shape[0]),
        "ridge": float(ridge),
        "head_blend_alpha": float(head_blend_alpha),
        "margin": float(margin),
        "steps": int(steps),
        "pair_count": int(pair_count),
        "touched_vocab_rows": int(len(touched)),
        "updates": int(updates),
        "hinge_loss_before": float(before_loss),
        "hinge_loss_after": float(after_loss),
        "violations_before": int(before_violations),
        "violations_after": int(after_violations),
        "delta_norm": float((solved - old_head).norm().detach().cpu().item()),
    }


def _select_hidden_state(hidden_states, hidden_layer: str):
    if not hidden_states:
        raise ValueError("Model output does not include hidden states.")
    if hidden_layer == "last":
        return hidden_states[-1]
    if hidden_layer == "middle":
        return hidden_states[len(hidden_states) // 2]
    raise ValueError("hidden_layer must be last or middle.")


def _masked_mean_hidden(torch, hidden, mask):
    hidden = torch.nan_to_num(hidden.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    if mask is None:
        return hidden.mean(dim=1)
    weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _normalize_hidden_rows(torch, rows):
    rows = torch.nan_to_num(rows.to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
    centered = rows - rows.mean(dim=-1, keepdim=True)
    scale = centered.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return centered / scale


def _masked_mean_logits(torch, logits, mask):
    weights = mask.to(logits.dtype).unsqueeze(-1)
    summed = (logits * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _masked_mean_values(torch, values, mask):
    if mask is None:
        return values.mean()
    weights = mask.to(device=values.device, dtype=values.dtype)
    summed = (values * weights).sum()
    count = weights.sum().clamp_min(1.0)
    return summed / count


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
