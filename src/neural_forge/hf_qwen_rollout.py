"""Deterministic, zero-gradient rollout utilities for Qwen composition.

The functions in this module keep rollout token IDs as the source of truth.
Text is used only to derive prompts and tokenize them once with the child;
donor and child tracing can then share batches built directly from those IDs.
"""

from __future__ import annotations

import hashlib
import json
import operator
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any


ROLLOUT_SCHEMA = "neural-forge-qwen-greedy-rollout-v1"

__all__ = [
    "ROLLOUT_SCHEMA",
    "build_exact_trace_batches",
    "build_trace_batches_from_token_ids",
    "derive_prompt_only_records",
    "derive_prompt_only_strings",
    "derive_prompt_only_texts",
    "derive_rollout_prompts",
    "generate_greedy_child_rollouts",
    "greedy_child_rollouts",
    "hash_rollout_provenance",
    "hash_token_id_sequence",
    "hash_token_id_sequences",
    "make_trace_batches_from_token_ids",
    "validate_token_id_sequences",
]


_COLON_BOUNDARY = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?P<label>answer|assistant|response|solution|code)[ \t]*:"
)
_SPECIAL_ASSISTANT_BOUNDARIES = (
    re.compile(r"(?i)<\|im_start\|>[ \t]*assistant[ \t]*(?:\r?\n)?"),
    re.compile(r"(?i)<\|assistant\|>[ \t]*(?:\r?\n)?"),
)
_CODE_FENCE = re.compile(
    r"(?m)^[ \t]*```(?P<language>[A-Za-z0-9_+.-]*)[ \t]*(?:\r?\n|$)"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(normalized)


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _token_id(value: Any, name: str, vocab_size: int | None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must contain integer token IDs, not booleans.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain only integer token IDs.") from exc
    normalized = int(normalized)
    if normalized < 0:
        raise ValueError(f"{name} contains a negative token ID: {normalized}.")
    if vocab_size is not None and normalized >= vocab_size:
        raise ValueError(
            f"{name} contains token ID {normalized}, outside vocab_size={vocab_size}."
        )
    return normalized


def _sequence_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("token_id_sequences", "sequences"):
            if key in value:
                return value[key]
        raise ValueError(
            "A rollout mapping must contain token_id_sequences or sequences."
        )
    return value


def validate_token_id_sequences(
    token_id_sequences: Any,
    *,
    vocab_size: int | None = None,
    max_length: int | None = None,
) -> list[list[int]]:
    """Validate and normalize non-empty, exact token-ID sequences."""

    vocab_limit = _optional_positive_int(vocab_size, "vocab_size")
    length_limit = _optional_positive_int(max_length, "max_length")
    values = _sequence_value(token_id_sequences)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        if hasattr(values, "tolist"):
            values = values.tolist()
        else:
            raise TypeError("token_id_sequences must be a sequence of sequences.")
    if len(values) == 0:
        raise ValueError("token_id_sequences must not be empty.")

    normalized: list[list[int]] = []
    for row_index, row in enumerate(values):
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"token_id_sequences[{row_index}] must be a sequence.")
        if len(row) == 0:
            raise ValueError(f"token_id_sequences[{row_index}] must not be empty.")
        if length_limit is not None and len(row) > length_limit:
            raise ValueError(
                f"token_id_sequences[{row_index}] has length {len(row)}, "
                f"exceeding max_length={length_limit}."
            )
        normalized.append(
            [
                _token_id(
                    item,
                    f"token_id_sequences[{row_index}]",
                    vocab_limit,
                )
                for item in row
            ]
        )
    return normalized


def hash_token_id_sequence(token_ids: Sequence[int]) -> str:
    """Return the canonical SHA-256 hash of one exact token-ID sequence."""

    return _json_sha256(validate_token_id_sequences([token_ids])[0])


def hash_token_id_sequences(token_id_sequences: Any) -> str:
    """Return the canonical SHA-256 hash of ordered token-ID sequences."""

    return _json_sha256(validate_token_id_sequences(token_id_sequences))


def hash_rollout_provenance(provenance: Mapping[str, Any]) -> str:
    """Hash provenance, excluding its self-referential hash field."""

    normalized = dict(provenance)
    normalized.pop("provenance_sha256", None)
    return _json_sha256(normalized)


def _render_messages(messages: Any, index: int) -> str:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError(
            f"calibration_examples[{index}]['messages'] must be a sequence."
        )
    rendered: list[str] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"calibration_examples[{index}]['messages'][{message_index}] "
                "must be a mapping."
            )
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = message.get("content", "")
        if not isinstance(content, str):
            raise TypeError(
                f"calibration_examples[{index}]['messages'][{message_index}] "
                "content must be a string."
            )
        if role == "assistant" and message_index == len(messages) - 1:
            rendered.append("assistant:")
        else:
            rendered.append(f"{role}: {content}")
    return "\n".join(rendered)


def _example_text(example: Any, index: int) -> tuple[str, str]:
    if isinstance(example, str):
        return example, "string"
    if isinstance(example, Mapping):
        for key in ("prompt", "text", "full_text", "content", "input"):
            if key in example:
                value = example[key]
                if not isinstance(value, str):
                    raise TypeError(
                        f"calibration_examples[{index}]['{key}'] must be a string."
                    )
                return value, key
        if "messages" in example:
            return _render_messages(example["messages"], index), "messages"
        raise ValueError(
            f"calibration_examples[{index}] has no prompt, text, content, input, "
            "or messages field."
        )
    raise TypeError(
        f"calibration_examples[{index}] must be a string or mapping, "
        f"not {type(example).__name__}."
    )


def _prompt_boundary(text: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for match in _COLON_BOUNDARY.finditer(text):
        label = match.group("label").lower()
        candidates.append(
            {
                "boundary_kind": label,
                "boundary_marker": match.group(0),
                "boundary_char_index": int(match.start()),
                "response_char_start": int(match.end()),
                "prompt_char_end": int(match.end()),
            }
        )
    for pattern in _SPECIAL_ASSISTANT_BOUNDARIES:
        for match in pattern.finditer(text):
            candidates.append(
                {
                    "boundary_kind": "assistant",
                    "boundary_marker": match.group(0),
                    "boundary_char_index": int(match.start()),
                    "response_char_start": int(match.end()),
                    "prompt_char_end": int(match.end()),
                }
            )

    fence_open = False
    for match in _CODE_FENCE.finditer(text):
        if not fence_open:
            suffix = text[match.end() :]
            if suffix.strip().strip("`").strip():
                candidates.append(
                    {
                        "boundary_kind": "code_fence",
                        "boundary_marker": match.group(0).rstrip("\r\n"),
                        "boundary_char_index": int(match.start()),
                        "response_char_start": int(match.end()),
                        "prompt_char_end": int(match.start()),
                    }
                )
        fence_open = not fence_open

    if not candidates:
        return {
            "boundary_kind": "none",
            "boundary_marker": None,
            "boundary_char_index": None,
            "response_char_start": None,
            "prompt_char_end": len(text),
        }
    return max(
        candidates,
        key=lambda item: (
            int(item["boundary_char_index"]),
            int(item["prompt_char_end"]),
        ),
    )


def derive_prompt_only_records(
    calibration_examples: Sequence[Any],
    *,
    max_prompt_count: int | None = None,
) -> list[dict[str, Any]]:
    """Derive prompt-only text and boundary provenance from calibration examples.

    ``Answer:``, ``assistant:``, response-like line labels, ChatML assistant
    markers, and opening code fences are recognized. The most recent response
    boundary wins, which preserves demonstrations appearing earlier in a text.
    """

    if isinstance(calibration_examples, (str, bytes)) or not isinstance(
        calibration_examples, Sequence
    ):
        raise TypeError("calibration_examples must be a sequence, not one string.")
    limit = _optional_positive_int(max_prompt_count, "max_prompt_count")
    records: list[dict[str, Any]] = []
    for source_index, example in enumerate(calibration_examples):
        text, source_kind = _example_text(example, source_index)
        boundary = _prompt_boundary(text)
        prompt = text[: int(boundary["prompt_char_end"])]
        record = {
            "source_index": int(source_index),
            "source_kind": source_kind,
            "boundary_kind": boundary["boundary_kind"],
            "boundary_marker": boundary["boundary_marker"],
            "boundary_char_index": boundary["boundary_char_index"],
            "response_char_start": boundary["response_char_start"],
            "original_char_count": len(text),
            "prompt_char_count": len(prompt),
            "response_removed": prompt != text,
            "original_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "prompt_text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt": prompt,
        }
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def derive_prompt_only_strings(
    calibration_examples: Sequence[Any],
    *,
    max_prompt_count: int | None = None,
) -> list[str]:
    """Return prompt-only strings in stable calibration order."""

    return [
        record["prompt"]
        for record in derive_prompt_only_records(
            calibration_examples,
            max_prompt_count=max_prompt_count,
        )
    ]


derive_rollout_prompts = derive_prompt_only_strings
derive_prompt_only_texts = derive_prompt_only_strings


def _torch(torch_module: Any | None):
    if torch_module is not None:
        return torch_module
    import torch

    return torch


def _tokenizer_vocab_size(tokenizer: Any) -> int | None:
    try:
        value = len(tokenizer)
    except (AttributeError, TypeError, NotImplementedError):
        value = getattr(tokenizer, "vocab_size", None)
    if value is None:
        return None
    try:
        return _positive_int(value, "tokenizer vocabulary size")
    except ValueError:
        return None


def _eos_token_ids(tokenizer: Any, vocab_size: int | None) -> list[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value]
    if not values:
        return []
    return [_token_id(item, "eos_token_id", vocab_size) for item in values]


def _model_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return next(parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return None


def _gradient_references(model: Any) -> tuple[tuple[int, Any], ...]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return ()
    return tuple(
        (id(parameter), getattr(parameter, "grad", None)) for parameter in parameters()
    )


def _restore_model_mode(model: Any, was_training: bool | None) -> None:
    if was_training is True:
        train = getattr(model, "train", None)
        if callable(train):
            train()
    elif was_training is False:
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()


@contextmanager
def _temporary_padding_side(tokenizer: Any, padding_side: str):
    has_padding_side = hasattr(tokenizer, "padding_side")
    original = getattr(tokenizer, "padding_side", None)
    if has_padding_side:
        tokenizer.padding_side = padding_side
    try:
        yield original
    finally:
        if has_padding_side:
            tokenizer.padding_side = original


def _as_rank_two_long(torch: Any, value: Any, name: str):
    if hasattr(value, "detach"):
        detached = value.detach().cpu()
        raw = detached.tolist()
    else:
        raw = value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError(f"Tokenizer {name} must be a rank-2 sequence or tensor.")
    if raw and not isinstance(raw[0], Sequence):
        raw = [raw]
    normalized = validate_token_id_sequences(raw) if name == "input_ids" else raw
    tensor = torch.as_tensor(normalized, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"Tokenizer {name} must be rank 2.")
    return tensor


def _attention_tensor(torch: Any, value: Any, shape: Any):
    if hasattr(value, "detach"):
        raw = value.detach().cpu().tolist()
    else:
        raw = value
    tensor = torch.as_tensor(raw)
    if tensor.ndim == 1 and int(shape[0]) == 1:
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError("Tokenizer attention_mask must match input_ids shape.")
    flat = tensor.reshape(-1).tolist()
    if any(not isinstance(item, (bool, int)) for item in flat):
        raise ValueError(
            "Tokenizer attention_mask must contain integer zeros and ones."
        )
    if any(item not in (0, 1) for item in flat):
        raise ValueError("Tokenizer attention_mask must contain only zeros and ones.")
    return tensor.to(dtype=torch.long)


def _generated_sequences_tensor(torch: Any, generated: Any):
    value = getattr(generated, "sequences", generated)
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError("child_model.generate must return rank-2 token sequences.")
    return tensor


def generate_greedy_child_rollouts(
    child_model: Any,
    tokenizer: Any,
    calibration_examples: Sequence[Any],
    *,
    max_prompt_count: int = 8,
    max_length: int = 160,
    max_new_tokens: int = 32,
    device: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Generate deterministic child rollouts and return exact token sequences.

    The child is placed in evaluation mode only for generation. Existing
    parameter gradients, the original train/eval mode, and tokenizer padding
    side are preserved. The returned IDs exclude input padding and generation
    padding after the first EOS, but otherwise are never decoded or re-tokenized.
    """

    torch = _torch(torch_module)
    prompt_limit = _positive_int(max_prompt_count, "max_prompt_count")
    total_limit = _positive_int(max_length, "max_length")
    new_token_limit = _positive_int(max_new_tokens, "max_new_tokens")
    if new_token_limit >= total_limit:
        raise ValueError("max_new_tokens must be smaller than max_length.")
    prompt_token_limit = total_limit - new_token_limit

    all_records = derive_prompt_only_records(calibration_examples)
    records = [record for record in all_records if record["prompt"].strip()][
        :prompt_limit
    ]
    if not records:
        raise ValueError("No non-empty rollout prompts were derived.")
    prompts = [str(record["prompt"]) for record in records]

    vocab_size = _tokenizer_vocab_size(tokenizer)
    eos_ids = _eos_token_ids(tokenizer, vocab_size)
    pad_value = getattr(tokenizer, "pad_token_id", None)
    if pad_value is None:
        pad_value = eos_ids[0] if eos_ids else None
    if pad_value is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id.")
    pad_token_id = _token_id(pad_value, "pad_token_id", vocab_size)

    was_training_value = getattr(child_model, "training", None)
    was_training = (
        bool(was_training_value) if isinstance(was_training_value, bool) else None
    )
    padding_side_before = getattr(tokenizer, "padding_side", None)
    gradients_before = _gradient_references(child_model)
    target_device = device if device is not None else _model_device(child_model)

    try:
        evaluate = getattr(child_model, "eval", None)
        if callable(evaluate):
            evaluate()
        with _temporary_padding_side(tokenizer, "left"):
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_token_limit,
            )
            if not isinstance(encoded, Mapping):
                try:
                    encoded = dict(encoded)
                except (TypeError, ValueError) as exc:
                    raise TypeError("Tokenizer output must be mapping-like.") from exc
            if "input_ids" not in encoded:
                raise ValueError("Tokenizer output must contain input_ids.")
            input_ids_cpu = _as_rank_two_long(torch, encoded["input_ids"], "input_ids")
            if int(input_ids_cpu.shape[0]) != len(prompts):
                raise ValueError("Tokenizer returned the wrong number of prompt rows.")
            validate_token_id_sequences(
                input_ids_cpu.tolist(),
                vocab_size=vocab_size,
                max_length=prompt_token_limit,
            )
            if "attention_mask" in encoded:
                attention_cpu = _attention_tensor(
                    torch, encoded["attention_mask"], input_ids_cpu.shape
                )
            else:
                attention_cpu = torch.ones_like(input_ids_cpu, dtype=torch.long)
            if any(int(row.sum().item()) <= 0 for row in attention_cpu):
                raise ValueError(
                    "Every rollout prompt must contain at least one token."
                )

            input_ids = input_ids_cpu
            attention_mask = attention_cpu
            if target_device is not None:
                input_ids = input_ids.to(target_device)
                attention_mask = attention_mask.to(target_device)
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": new_token_limit,
                "do_sample": False,
                "pad_token_id": pad_token_id,
            }
            eos_argument = getattr(tokenizer, "eos_token_id", None)
            if eos_argument is not None:
                generation_kwargs["eos_token_id"] = eos_argument
            inference_context = getattr(torch, "inference_mode", torch.no_grad)
            with inference_context():
                generated = child_model.generate(**generation_kwargs)
    finally:
        _restore_model_mode(child_model, was_training)

    if getattr(tokenizer, "padding_side", None) != padding_side_before:
        raise RuntimeError("Tokenizer padding_side was not restored after generation.")
    if (
        was_training is not None
        and bool(getattr(child_model, "training", None)) != was_training
    ):
        raise RuntimeError(
            "Child model train/eval state was not restored after generation."
        )
    gradients_after = _gradient_references(child_model)
    if len(gradients_before) != len(gradients_after) or any(
        before_id != after_id or before_grad is not after_grad
        for (before_id, before_grad), (after_id, after_grad) in zip(
            gradients_before, gradients_after
        )
    ):
        raise RuntimeError("Greedy rollout generation changed parameter gradients.")

    generated_cpu = _generated_sequences_tensor(torch, generated)
    if int(generated_cpu.shape[0]) != len(prompts):
        raise ValueError("child_model.generate returned the wrong number of rows.")
    generated_rows = validate_token_id_sequences(
        generated_cpu.tolist(),
        vocab_size=vocab_size,
        max_length=total_limit,
    )
    generated_cpu = torch.tensor(generated_rows, dtype=torch.long)
    input_width = int(input_ids_cpu.shape[1])
    if int(generated_cpu.shape[1]) < input_width:
        raise ValueError("Generated rows are shorter than their padded prompt inputs.")
    generated_prefix = generated_cpu[:, :input_width].to(dtype=torch.long)
    if not bool(torch.equal(generated_prefix, input_ids_cpu)):
        raise ValueError("Generated rows do not preserve the exact padded prompt IDs.")

    exact_sequences: list[list[int]] = []
    prompt_sequences: list[list[int]] = []
    generated_sequences: list[list[int]] = []
    rollout_records: list[dict[str, Any]] = []
    sequence_records: list[dict[str, Any]] = []
    eos_set = set(eos_ids)
    for row_index, record in enumerate(records):
        mask = attention_cpu[row_index].to(dtype=torch.bool)
        prompt_ids = input_ids_cpu[row_index][mask].tolist()
        suffix = generated_cpu[row_index, input_width:].tolist()
        suffix = [_token_id(item, "generated token IDs", vocab_size) for item in suffix]
        stopped_on_eos = False
        if eos_set:
            for token_index, token_id in enumerate(suffix):
                if token_id in eos_set:
                    suffix = suffix[: token_index + 1]
                    stopped_on_eos = True
                    break
        if len(suffix) > new_token_limit:
            raise ValueError(
                "child_model.generate returned more tokens than max_new_tokens."
            )
        exact = prompt_ids + suffix
        validate_token_id_sequences(
            [exact], vocab_size=vocab_size, max_length=total_limit
        )
        prompt_sequences.append(prompt_ids)
        generated_sequences.append(suffix)
        exact_sequences.append(exact)
        token_ids_sha256 = _json_sha256(exact)
        trajectory_sha256 = _json_sha256(
            {
                "token_ids": exact,
                "prompt_length": len(prompt_ids),
            }
        )
        rollout_id = f"sha256:{trajectory_sha256}"
        rollout_records.append(
            {
                "rollout_index": int(row_index),
                "source_index": int(record["source_index"]),
                "rollout_id": rollout_id,
                "trajectory_id": rollout_id,
                "trajectory_sha256": trajectory_sha256,
                "token_ids_sha256": token_ids_sha256,
                "token_ids": exact,
                "prompt_length": len(prompt_ids),
                "generated_length": len(suffix),
                "sequence_length": len(exact),
            }
        )
        sequence_records.append(
            {
                "rollout_index": int(row_index),
                "source_index": int(record["source_index"]),
                "rollout_id": rollout_id,
                "trajectory_sha256": trajectory_sha256,
                "boundary_kind": record["boundary_kind"],
                "prompt_text_sha256": record["prompt_text_sha256"],
                "prompt_token_count": len(prompt_ids),
                "generated_token_count": len(suffix),
                "total_token_count": len(exact),
                "stopped_on_eos": stopped_on_eos,
                "prompt_token_ids_sha256": _json_sha256(prompt_ids),
                "generated_token_ids_sha256": _json_sha256(suffix),
                "sequence_token_ids_sha256": token_ids_sha256,
            }
        )

    sequence_hash = _json_sha256(exact_sequences)
    prompt_record_provenance = [
        {key: value for key, value in record.items() if key != "prompt"}
        for record in records
    ]
    provenance: dict[str, Any] = {
        "schema": ROLLOUT_SCHEMA,
        "algorithm": "greedy-child-generate",
        "do_sample": False,
        "zero_gradient": True,
        "autograd_enabled_during_generation": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "child_model_class": type(child_model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "max_prompt_count": prompt_limit,
        "max_length": total_limit,
        "max_new_tokens": new_token_limit,
        "prompt_token_max_length": prompt_token_limit,
        "available_prompt_count": len(all_records),
        "selected_prompt_count": len(records),
        "discarded_prompt_count": len(all_records) - len(records),
        "padding_side_during_generation": "left",
        "padding_side_before": padding_side_before,
        "padding_side_restored": True,
        "model_training_before": was_training,
        "model_training_restored": True,
        "parameters_with_grad_before": sum(
            gradient is not None for _, gradient in gradients_before
        ),
        "parameters_with_grad_after": sum(
            gradient is not None for _, gradient in gradients_after
        ),
        "pad_token_id": pad_token_id,
        "eos_token_ids": eos_ids,
        "vocab_size": vocab_size,
        "prompt_derivations": prompt_record_provenance,
        "sequences": sequence_records,
        "ordered_prompt_texts_sha256": _json_sha256(prompts),
        "ordered_prompt_token_ids_sha256": _json_sha256(prompt_sequences),
        "ordered_generated_token_ids_sha256": _json_sha256(generated_sequences),
        "ordered_sequence_token_ids_sha256": sequence_hash,
        "ordered_rollout_ids_sha256": _json_sha256(
            [record["rollout_id"] for record in rollout_records]
        ),
    }
    provenance["provenance_sha256"] = hash_rollout_provenance(provenance)
    return {
        "prompts": prompts,
        "prompt_token_id_sequences": prompt_sequences,
        "generated_token_id_sequences": generated_sequences,
        "token_id_sequences": exact_sequences,
        "sequences": [list(row) for row in exact_sequences],
        "rollout_records": rollout_records,
        "sequence_token_ids_sha256": sequence_hash,
        "provenance": provenance,
    }


greedy_child_rollouts = generate_greedy_child_rollouts


def build_exact_trace_batches(
    token_id_sequences: Any,
    batch_size: int = 2,
    pad_token_id: int = 0,
    device: Any = "cpu",
    *,
    prompt_lengths: Sequence[int] | None = None,
    padding_side: str = "right",
    max_length: int | None = None,
    vocab_size: int | None = None,
    torch_module: Any | None = None,
) -> list[dict[str, Any]]:
    """Pad exact IDs into the trace-batch contract used by hf_qwen_compose.

    No tokenizer is called and no sequence is truncated. Build the batches
    once and pass the same dictionaries to donor and child tracing to guarantee
    identical token IDs at both model interfaces. When ``prompt_lengths`` is
    supplied, each batch also has a top-level ``trace_mask`` marking the causal
    state immediately before every generated token; the model attention mask
    continues to cover every non-padding token.
    """

    torch = _torch(torch_module)
    normalized_batch_size = _positive_int(batch_size, "batch_size")
    vocab_limit = _optional_positive_int(vocab_size, "vocab_size")
    length_limit = _optional_positive_int(max_length, "max_length")
    padding = str(padding_side).lower()
    if padding not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'.")
    normalized_pad_id = _token_id(pad_token_id, "pad_token_id", vocab_limit)
    sequences = validate_token_id_sequences(
        token_id_sequences,
        vocab_size=vocab_limit,
        max_length=length_limit,
    )
    normalized_prompt_lengths: list[int] | None = None
    if prompt_lengths is not None:
        values: Any = prompt_lengths
        if hasattr(values, "tolist"):
            values = values.tolist()
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("prompt_lengths must be a sequence of positive integers.")
        if len(values) != len(sequences):
            raise ValueError(
                "prompt_lengths must contain exactly one value per token sequence."
            )
        normalized_prompt_lengths = []
        for sequence_index, (value, sequence) in enumerate(zip(values, sequences)):
            prompt_length = _positive_int(value, f"prompt_lengths[{sequence_index}]")
            if prompt_length > len(sequence):
                raise ValueError(
                    f"prompt_lengths[{sequence_index}]={prompt_length} exceeds "
                    f"sequence length {len(sequence)}."
                )
            normalized_prompt_lengths.append(prompt_length)

    batches: list[dict[str, Any]] = []
    for start in range(0, len(sequences), normalized_batch_size):
        chunk = sequences[start : start + normalized_batch_size]
        width = max(len(row) for row in chunk)
        padded_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        trace_rows: list[list[int]] | None = (
            [] if normalized_prompt_lengths is not None else None
        )
        for chunk_index, row in enumerate(chunk):
            padding_count = width - len(row)
            if padding == "left":
                padded_rows.append([normalized_pad_id] * padding_count + row)
                attention_rows.append([0] * padding_count + [1] * len(row))
            else:
                padded_rows.append(row + [normalized_pad_id] * padding_count)
                attention_rows.append([1] * len(row) + [0] * padding_count)
            if trace_rows is not None and normalized_prompt_lengths is not None:
                prompt_length = normalized_prompt_lengths[start + chunk_index]
                sequence_offset = padding_count if padding == "left" else 0
                trace_row = [0] * width
                decision_start = sequence_offset + prompt_length - 1
                decision_stop = sequence_offset + len(row) - 1
                for position in range(decision_start, decision_stop):
                    trace_row[position] = 1
                if sum(trace_row) != len(row) - prompt_length:
                    raise RuntimeError("Trace mask action count is inconsistent.")
                trace_rows.append(trace_row)
        input_ids = torch.tensor(padded_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        batch = {
            "start": int(start),
            "count": len(chunk),
            "attention_mask": attention_mask,
            "model_inputs": model_inputs,
        }
        if trace_rows is not None:
            batch["trace_mask"] = torch.tensor(
                trace_rows, dtype=torch.long, device=device
            )
        batches.append(batch)
    return batches


build_trace_batches_from_token_ids = build_exact_trace_batches
make_trace_batches_from_token_ids = build_exact_trace_batches
