"""Deterministic global autoregressive gates for exact child rollouts.

The parent is teacher-forced on token IDs already visited by a child.  Only
generated child actions are scored, using the parent logit at the preceding
causal position.  Evaluation is inference-only and never decodes or
re-tokenizes a rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from typing import Any


ROLLOUT_GATE_SCHEMA = "neural-forge-hf-global-autoregressive-gate-v1"
ROLLOUT_PROMOTION_SCHEMA = "neural-forge-hf-rollout-promotion-v1"

__all__ = [
    "ROLLOUT_GATE_SCHEMA",
    "ROLLOUT_PROMOTION_SCHEMA",
    "compare_hf_rollout_gate",
    "compare_rollout_gate_reports",
    "decide_rollout_gate_promotion",
    "evaluate_global_autoregressive_gate",
    "evaluate_hf_rollout_gate",
    "evaluate_parent_rollouts",
    "hash_rollout_records",
    "normalize_rollout_records",
]


_TOKEN_ID_KEYS = (
    "token_ids",
    "full_token_ids",
    "sequence_token_ids",
    "token_id_sequence",
    "input_ids",
    "sequence",
)
_PROMPT_LENGTH_KEYS = ("prompt_length", "prompt_token_count")


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


def _torch(torch_module: Any | None):
    if torch_module is not None:
        return torch_module
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("PyTorch is required for HF rollout gate evaluation.") from exc
    return torch


def _index(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean.")
    try:
        normalized = int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer.")
    return normalized


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return normalized


def _list_value(value: Any, name: str) -> list[Any]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence.")
    return list(value)


def _token_ids(value: Any, name: str) -> list[int]:
    values = _list_value(value, name)
    if not values:
        raise ValueError(f"{name} must not be empty.")
    result = []
    for index, item in enumerate(values):
        if isinstance(item, bool):
            raise ValueError(f"{name}[{index}] must be an integer token ID.")
        try:
            token_id = int(operator.index(item))
        except TypeError as exc:
            raise ValueError(
                f"{name}[{index}] must be an integer token ID."
            ) from exc
        if token_id < 0:
            raise ValueError(f"{name}[{index}] must be a non-negative token ID.")
        result.append(token_id)
    return result


def _attention_mask(value: Any, expected_length: int, name: str) -> list[int]:
    values = _list_value(value, name)
    if len(values) != expected_length:
        raise ValueError(f"{name} must have the same length as its token IDs.")
    mask = []
    for index, item in enumerate(values):
        if isinstance(item, bool):
            bit = int(item)
        else:
            try:
                bit = int(operator.index(item))
            except TypeError as exc:
                raise ValueError(f"{name}[{index}] must be zero or one.") from exc
        if bit not in (0, 1):
            raise ValueError(f"{name}[{index}] must be zero or one.")
        mask.append(bit)
    active = [index for index, bit in enumerate(mask) if bit]
    if not active:
        raise ValueError(f"{name} must select at least one token.")
    if active != list(range(active[0], active[-1] + 1)):
        raise ValueError(f"{name} may contain padding only before or after tokens.")
    return mask


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return "", None


def _aggregate_prompt_lengths(
    rollouts: Mapping[str, Any],
    sequences: Sequence[Sequence[int]],
) -> list[int]:
    for key in ("prompt_lengths", "prompt_token_counts"):
        if key in rollouts:
            values = _list_value(rollouts[key], key)
            if len(values) != len(sequences):
                raise ValueError(f"{key} must match the rollout sequence count.")
            return [
                _index(value, f"{key}[{index}]", minimum=1)
                for index, value in enumerate(values)
            ]

    if "prompt_token_id_sequences" in rollouts:
        prompt_rows = _list_value(
            rollouts["prompt_token_id_sequences"], "prompt_token_id_sequences"
        )
        if len(prompt_rows) != len(sequences):
            raise ValueError(
                "prompt_token_id_sequences must match the rollout sequence count."
            )
        lengths = []
        for index, (prompt_value, sequence) in enumerate(
            zip(prompt_rows, sequences, strict=True)
        ):
            prompt_ids = _token_ids(
                prompt_value, f"prompt_token_id_sequences[{index}]"
            )
            if list(sequence[: len(prompt_ids)]) != prompt_ids:
                raise ValueError(
                    f"prompt_token_id_sequences[{index}] is not the exact sequence prefix."
                )
            lengths.append(len(prompt_ids))
        return lengths

    provenance = rollouts.get("provenance")
    if isinstance(provenance, Mapping):
        provenance_records = provenance.get("sequences")
        if provenance_records is not None:
            values = _list_value(provenance_records, "provenance.sequences")
            if len(values) != len(sequences):
                raise ValueError(
                    "provenance.sequences must match the rollout sequence count."
                )
            lengths = []
            for index, value in enumerate(values):
                if not isinstance(value, Mapping):
                    raise TypeError(
                        f"provenance.sequences[{index}] must be a mapping."
                    )
                key, length = _first_present(value, _PROMPT_LENGTH_KEYS)
                if not key:
                    break
                lengths.append(
                    _index(
                        length,
                        f"provenance.sequences[{index}].{key}",
                        minimum=1,
                    )
                )
            if len(lengths) == len(sequences):
                return lengths

    raise ValueError(
        "Aggregate rollouts must include prompt_lengths, prompt_token_id_sequences, "
        "or provenance sequence prompt counts."
    )


def _records_from_aggregate(rollouts: Mapping[str, Any]) -> list[dict[str, Any]]:
    sequence_key = ""
    sequence_values: Any = None
    for key in ("token_id_sequences", "sequences"):
        if key in rollouts:
            candidate = _list_value(rollouts[key], key)
            if candidate and not isinstance(candidate[0], Mapping):
                sequence_key = key
                sequence_values = candidate
                break
    if not sequence_key:
        raise ValueError(
            "A rollout mapping must contain rollout_records or exact token_id_sequences."
        )

    sequences = [
        _token_ids(value, f"{sequence_key}[{index}]")
        for index, value in enumerate(sequence_values)
    ]
    prompt_lengths = _aggregate_prompt_lengths(rollouts, sequences)

    masks: list[Any] | None = None
    for key in ("attention_masks", "attention_mask"):
        if key in rollouts:
            masks = _list_value(rollouts[key], key)
            if len(masks) != len(sequences):
                raise ValueError(f"{key} must match the rollout sequence count.")
            break

    generated_values = rollouts.get("generated_token_id_sequences")
    generated_rows = (
        _list_value(generated_values, "generated_token_id_sequences")
        if generated_values is not None
        else None
    )
    if generated_rows is not None and len(generated_rows) != len(sequences):
        raise ValueError(
            "generated_token_id_sequences must match the rollout sequence count."
        )

    records = []
    for index, (sequence, prompt_length) in enumerate(
        zip(sequences, prompt_lengths, strict=True)
    ):
        record: dict[str, Any] = {
            "token_ids": sequence,
            "prompt_length": prompt_length,
        }
        if masks is not None:
            record["attention_mask"] = masks[index]
        if generated_rows is not None:
            record["generated_token_ids"] = generated_rows[index]
        records.append(record)
    return records


def normalize_rollout_records(rollouts: Any) -> list[dict[str, Any]]:
    """Normalize exact rollout records to token IDs plus active prompt length.

    Direct records accept ``token_ids`` (or a documented full-sequence alias)
    and ``prompt_length``.  The aggregate result returned by
    ``hf_qwen_rollout.generate_greedy_child_rollouts`` is also accepted.
    Per-record attention masks may remove contiguous left or right padding;
    ``prompt_length`` always counts active, non-padding prompt tokens.
    """

    if isinstance(rollouts, Mapping):
        record_key, record_values = _first_present(
            rollouts, ("rollout_records", "records")
        )
        if record_key:
            raw_records = _list_value(record_values, record_key)
        elif any(key in rollouts for key in _TOKEN_ID_KEYS) and any(
            key in rollouts for key in _PROMPT_LENGTH_KEYS
        ):
            raw_records = [rollouts]
        else:
            raw_records = _records_from_aggregate(rollouts)
    else:
        raw_records = _list_value(rollouts, "rollouts")

    if not raw_records:
        raise ValueError("rollouts must contain at least one record.")

    normalized = []
    for record_index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"rollouts[{record_index}] must be a mapping.")
        token_key, token_value = _first_present(raw_record, _TOKEN_ID_KEYS)
        if not token_key:
            raise ValueError(
                f"rollouts[{record_index}] must contain exact full token IDs."
            )
        ids = _token_ids(token_value, f"rollouts[{record_index}].{token_key}")

        if "attention_mask" in raw_record:
            mask = _attention_mask(
                raw_record["attention_mask"],
                len(ids),
                f"rollouts[{record_index}].attention_mask",
            )
            ids = [token_id for token_id, bit in zip(ids, mask, strict=True) if bit]

        prompt_key, prompt_value = _first_present(raw_record, _PROMPT_LENGTH_KEYS)
        prompt_ids_value = raw_record.get("prompt_token_ids")
        if prompt_key:
            prompt_length = _index(
                prompt_value,
                f"rollouts[{record_index}].{prompt_key}",
                minimum=1,
            )
        elif prompt_ids_value is not None:
            prompt_length = len(
                _token_ids(
                    prompt_ids_value,
                    f"rollouts[{record_index}].prompt_token_ids",
                )
            )
        else:
            raise ValueError(
                f"rollouts[{record_index}] must contain prompt_length."
            )
        if prompt_length > len(ids):
            raise ValueError(
                f"rollouts[{record_index}].prompt_length exceeds its active token count."
            )

        if prompt_ids_value is not None:
            prompt_ids = _token_ids(
                prompt_ids_value,
                f"rollouts[{record_index}].prompt_token_ids",
            )
            if len(prompt_ids) != prompt_length or ids[:prompt_length] != prompt_ids:
                raise ValueError(
                    f"rollouts[{record_index}].prompt_token_ids is not the exact prompt prefix."
                )

        generated_value = raw_record.get("generated_token_ids")
        if generated_value is not None:
            generated_ids = _list_value(
                generated_value,
                f"rollouts[{record_index}].generated_token_ids",
            )
            if generated_ids:
                generated_ids = _token_ids(
                    generated_ids,
                    f"rollouts[{record_index}].generated_token_ids",
                )
            else:
                generated_ids = []
            if ids[prompt_length:] != generated_ids:
                raise ValueError(
                    f"rollouts[{record_index}].generated_token_ids is not the exact generated suffix."
                )

        normalized.append({"token_ids": ids, "prompt_length": prompt_length})
    return normalized


def hash_rollout_records(rollouts: Any) -> str:
    """Hash ordered, normalized rollout token IDs and prompt boundaries."""

    return _json_sha256(normalize_rollout_records(rollouts))


def _model_device(model: Any) -> Any | None:
    direct = getattr(model, "device", None)
    if direct is not None:
        return direct
    for accessor_name in ("parameters", "buffers"):
        accessor = getattr(model, accessor_name, None)
        if not callable(accessor):
            continue
        try:
            first = next(iter(accessor()))
        except (StopIteration, TypeError):
            continue
        return getattr(first, "device", None)
    return None


def _gradient_references(model: Any) -> tuple[tuple[int, Any, Any], ...]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return ()
    references = []
    for parameter in parameters():
        gradient = getattr(parameter, "grad", None)
        version = getattr(gradient, "_version", None) if gradient is not None else None
        references.append((id(parameter), gradient, version))
    return tuple(references)


def _restore_model_mode(model: Any, was_training: bool | None) -> None:
    if was_training is None:
        return
    train = getattr(model, "train", None)
    if callable(train):
        train(was_training)
    elif not was_training:
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()


def _model_logits(torch: Any, output: Any):
    logits = output.get("logits") if isinstance(output, Mapping) else None
    if logits is None:
        logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if not torch.is_tensor(logits):
        raise TypeError("parent_model output must contain a logits tensor.")
    if logits.ndim != 3:
        raise ValueError("parent_model logits must have shape [batch, sequence, vocab].")
    if not logits.dtype.is_floating_point:
        raise ValueError("parent_model logits must have a floating-point dtype.")
    return logits


def _longest_run(values: Sequence[int]) -> int:
    longest = 0
    current = 0
    previous: Any = object()
    for value in values:
        if value == previous:
            current += 1
        else:
            previous = value
            current = 1
        longest = max(longest, current)
    return longest


def _report_hash(report: Mapping[str, Any]) -> str:
    hashable = dict(report)
    hashes = dict(hashable.get("hashes", {}))
    hashes.pop("report_sha256", None)
    hashable["hashes"] = hashes
    return _json_sha256(hashable)


def evaluate_parent_rollouts(
    parent_model: Any,
    rollouts: Any,
    *,
    batch_size: int = 8,
    top_k: int = 5,
    pad_token_id: int | None = None,
    padding_side: str = "right",
    device: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Score child-generated actions under a parent causal language model."""

    torch = _torch(torch_module)
    records = normalize_rollout_records(rollouts)
    normalized_batch_size = _index(batch_size, "batch_size", minimum=1)
    requested_top_k = _index(top_k, "top_k", minimum=1)
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'.")

    configured_pad = pad_token_id
    if configured_pad is None:
        config = getattr(parent_model, "config", None)
        configured_pad = getattr(config, "pad_token_id", None)
    if configured_pad is None:
        configured_pad = 0
    normalized_pad_token_id = _index(configured_pad, "pad_token_id")

    generated_rows = [
        record["token_ids"][record["prompt_length"] :] for record in records
    ]
    generated_token_count = sum(len(row) for row in generated_rows)
    if generated_token_count == 0:
        raise ValueError("At least one rollout must contain a generated token action.")

    config = getattr(parent_model, "config", None)
    configured_vocab_size = getattr(config, "vocab_size", None)
    if configured_vocab_size is not None:
        vocab_limit = _index(configured_vocab_size, "parent_model.config.vocab_size", minimum=1)
        all_ids = [token_id for record in records for token_id in record["token_ids"]]
        if normalized_pad_token_id >= vocab_limit or any(
            token_id >= vocab_limit for token_id in all_ids
        ):
            raise ValueError("Rollout or padding token ID is outside the parent vocabulary.")

    training_value = getattr(parent_model, "training", None)
    was_training = bool(training_value) if isinstance(training_value, bool) else None
    gradients_before = _gradient_references(parent_model)
    target_device = device if device is not None else _model_device(parent_model)

    top1_count = 0
    top_k_count = 0
    rank_sum = 0
    action_logprobs: list[float] = []
    detected_vocab_size: int | None = None
    effective_top_k: int | None = None
    forward_call_count = 0

    try:
        evaluate = getattr(parent_model, "eval", None)
        if callable(evaluate):
            evaluate()
        inference_context = getattr(torch, "inference_mode", torch.no_grad)
        with inference_context():
            for start in range(0, len(records), normalized_batch_size):
                batch = records[start : start + normalized_batch_size]
                width = max(len(record["token_ids"]) for record in batch)
                padded_rows = []
                mask_rows = []
                offsets = []
                for record in batch:
                    ids = record["token_ids"]
                    padding_count = width - len(ids)
                    if padding_side == "left":
                        padded_rows.append(
                            [normalized_pad_token_id] * padding_count + ids
                        )
                        mask_rows.append([0] * padding_count + [1] * len(ids))
                        offsets.append(padding_count)
                    else:
                        padded_rows.append(
                            ids + [normalized_pad_token_id] * padding_count
                        )
                        mask_rows.append([1] * len(ids) + [0] * padding_count)
                        offsets.append(0)

                input_ids = torch.tensor(padded_rows, dtype=torch.long)
                attention_mask = torch.tensor(mask_rows, dtype=torch.long)
                if target_device is not None:
                    input_ids = input_ids.to(target_device)
                    attention_mask = attention_mask.to(target_device)

                output = parent_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                forward_call_count += 1
                logits = _model_logits(torch, output)
                if tuple(logits.shape[:2]) != tuple(input_ids.shape):
                    raise ValueError(
                        "parent_model logits batch and sequence dimensions must match input_ids."
                    )
                vocab_size = int(logits.shape[-1])
                if vocab_size <= 0:
                    raise ValueError("parent_model logits vocabulary must not be empty.")
                if detected_vocab_size is None:
                    detected_vocab_size = vocab_size
                    effective_top_k = min(requested_top_k, vocab_size)
                    if normalized_pad_token_id >= vocab_size:
                        raise ValueError("pad_token_id is outside the parent vocabulary.")
                elif vocab_size != detected_vocab_size:
                    raise ValueError("parent_model vocabulary size changed between batches.")

                action_batch_rows = []
                prediction_positions = []
                action_ids = []
                for batch_index, (record, offset) in enumerate(
                    zip(batch, offsets, strict=True)
                ):
                    ids = record["token_ids"]
                    for token_position in range(record["prompt_length"], len(ids)):
                        action_batch_rows.append(batch_index)
                        prediction_positions.append(offset + token_position - 1)
                        action_ids.append(ids[token_position])

                if any(action_id >= vocab_size for action_id in action_ids):
                    raise ValueError("A generated action token ID is outside the parent vocabulary.")
                row_index = torch.tensor(
                    action_batch_rows, dtype=torch.long, device=logits.device
                )
                position_index = torch.tensor(
                    prediction_positions, dtype=torch.long, device=logits.device
                )
                targets = torch.tensor(action_ids, dtype=torch.long, device=logits.device)
                action_logits = logits[row_index, position_index, :]
                if not bool(torch.isfinite(action_logits).all().item()):
                    raise ValueError(
                        "parent_model produced non-finite logits for generated actions."
                    )

                parent_top1 = action_logits.argmax(dim=-1)
                raw_target_scores = action_logits.gather(
                    1, targets.unsqueeze(1)
                ).squeeze(1)
                ranks = 1 + (
                    action_logits > raw_target_scores.unsqueeze(1)
                ).sum(dim=-1)

                compute_dtype = (
                    torch.float64
                    if action_logits.dtype == torch.float64
                    else torch.float32
                )
                scores = action_logits.to(dtype=compute_dtype)
                target_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
                log_probabilities = target_scores - torch.logsumexp(scores, dim=-1)
                if not bool(torch.isfinite(log_probabilities).all().item()):
                    raise ValueError("Parent action log probabilities must be finite.")
                action_logprobs.extend(
                    float(value) for value in log_probabilities.detach().cpu().tolist()
                )

                top1_count += int((parent_top1 == targets).sum().item())
                top_k_count += int((ranks <= int(effective_top_k)).sum().item())
                rank_sum += int(ranks.sum().item())
    finally:
        _restore_model_mode(parent_model, was_training)

    if was_training is not None and bool(getattr(parent_model, "training", None)) != was_training:
        raise RuntimeError("Parent model train/eval state was not restored.")
    gradients_after = _gradient_references(parent_model)
    if len(gradients_before) != len(gradients_after) or any(
        before_id != after_id
        or before_gradient is not after_gradient
        or before_version != after_version
        for (before_id, before_gradient, before_version), (
            after_id,
            after_gradient,
            after_version,
        ) in zip(gradients_before, gradients_after, strict=True)
    ):
        raise RuntimeError("Rollout gate evaluation changed parent parameter gradients.")

    if len(action_logprobs) != generated_token_count:
        raise RuntimeError("Generated action accounting did not match rollout records.")
    if detected_vocab_size is None or effective_top_k is None:
        raise RuntimeError("Parent model was not evaluated.")

    prompt_rows = [
        record["token_ids"][: record["prompt_length"]] for record in records
    ]
    sequence_rows = [record["token_ids"] for record in records]
    prompt_token_count = sum(len(row) for row in prompt_rows)
    total_token_count = sum(len(row) for row in sequence_rows)
    repeated_token_count = sum(len(row) - len(set(row)) for row in generated_rows)
    within_rollout_unique_count = generated_token_count - repeated_token_count
    global_unique_count = len({token_id for row in generated_rows for token_id in row})
    longest_run = max((_longest_run(row) for row in generated_rows), default=0)
    mean_logprob = math.fsum(action_logprobs) / generated_token_count

    metrics = {
        "parent_top1_agreement": top1_count / generated_token_count,
        "mean_parent_action_logprob": mean_logprob,
        "mean_parent_action_rank": rank_sum / generated_token_count,
        "mean_parent_rank": rank_sum / generated_token_count,
        "parent_action_top_k_coverage": top_k_count / generated_token_count,
        "parent_top_k_coverage": top_k_count / generated_token_count,
        "generated_repetition_rate": repeated_token_count / generated_token_count,
        "generated_unique_token_ratio": within_rollout_unique_count
        / generated_token_count,
        "generated_global_unique_token_ratio": global_unique_count
        / generated_token_count,
        "generated_longest_run": longest_run,
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Rollout gate metrics must all be finite.")

    record_summaries = [
        {
            "rollout_index": index,
            "prompt_length": record["prompt_length"],
            "generated_token_count": len(generated_rows[index]),
            "total_token_count": len(record["token_ids"]),
            "prompt_token_ids_sha256": _json_sha256(prompt_rows[index]),
            "generated_token_ids_sha256": _json_sha256(generated_rows[index]),
            "sequence_token_ids_sha256": _json_sha256(record["token_ids"]),
        }
        for index, record in enumerate(records)
    ]
    hashes = {
        "rollout_records_sha256": _json_sha256(records),
        "prompt_token_ids_sha256": _json_sha256(prompt_rows),
        "generated_token_ids_sha256": _json_sha256(generated_rows),
        "sequence_token_ids_sha256": _json_sha256(sequence_rows),
    }
    report: dict[str, Any] = {
        "schema": ROLLOUT_GATE_SCHEMA,
        "algorithm": "parent-causal-lm-on-exact-child-rollouts",
        "zero_gradient": True,
        "gradient_context": "torch.inference_mode",
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameter_gradients_written": False,
        "parent_model_class": type(parent_model).__name__,
        "model_training_before": was_training,
        "model_training_restored": True,
        "padding_side": padding_side,
        "pad_token_id": normalized_pad_token_id,
        "batch_size": normalized_batch_size,
        "forward_call_count": forward_call_count,
        "requested_top_k": requested_top_k,
        "effective_top_k": effective_top_k,
        "vocab_size": detected_vocab_size,
        "prompt_count": len(records),
        "rollout_count": len(records),
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "generated_action_count": generated_token_count,
        "total_token_count": total_token_count,
        "max_sequence_length": max(len(row) for row in sequence_rows),
        "parent_top1_agreement_count": top1_count,
        "parent_action_top_k_count": top_k_count,
        "parent_action_rank_sum": rank_sum,
        "generated_repeated_token_count": repeated_token_count,
        "generated_within_rollout_unique_token_count": within_rollout_unique_count,
        "generated_unique_token_count": global_unique_count,
        **metrics,
        "metrics": dict(metrics),
        "records": record_summaries,
        "hashes": hashes,
    }
    hashes["report_sha256"] = _report_hash(report)
    return report


evaluate_hf_rollout_gate = evaluate_parent_rollouts
evaluate_global_autoregressive_gate = evaluate_parent_rollouts


def _report_metric(report: Mapping[str, Any], key: str) -> float:
    value = report.get(key)
    if value is None:
        metrics = report.get("metrics")
        if isinstance(metrics, Mapping):
            value = metrics.get(key)
    if value is None:
        raise ValueError(f"Rollout gate report is missing {key}.")
    return _finite_float(value, key)


def compare_rollout_gate_reports(
    incumbent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    min_top1_gain: float = 0.0,
    min_action_logprob_gain: float = 0.0,
    max_repetition_regression: float = 0.0,
    max_longest_run_regression: int = 0,
) -> dict[str, Any]:
    """Return a behavior-first, degeneration-vetoed promotion decision.

    A qualifying parent top-1 gain wins even if action log probability falls.
    If top-1 agreement is unchanged or its positive gain is below the configured
    minimum, parent action log probability is the tie-breaker.  Any top-1 loss,
    repetition regression, or longest-run regression beyond its threshold
    reverts the candidate.
    """

    if not isinstance(incumbent, Mapping) or not isinstance(candidate, Mapping):
        raise TypeError("incumbent and candidate must be rollout gate reports.")
    normalized_min_top1_gain = _finite_float(
        min_top1_gain, "min_top1_gain", minimum=0.0
    )
    normalized_min_logprob_gain = _finite_float(
        min_action_logprob_gain, "min_action_logprob_gain", minimum=0.0
    )
    normalized_max_repetition = _finite_float(
        max_repetition_regression, "max_repetition_regression", minimum=0.0
    )
    normalized_max_run = _index(
        max_longest_run_regression, "max_longest_run_regression"
    )

    metric_keys = (
        "parent_top1_agreement",
        "mean_parent_action_logprob",
        "generated_repetition_rate",
        "generated_longest_run",
    )
    incumbent_metrics = {
        key: _report_metric(incumbent, key) for key in metric_keys
    }
    candidate_metrics = {
        key: _report_metric(candidate, key) for key in metric_keys
    }
    for label, metrics in (
        ("incumbent", incumbent_metrics),
        ("candidate", candidate_metrics),
    ):
        if not 0.0 <= metrics["parent_top1_agreement"] <= 1.0:
            raise ValueError(f"{label} parent_top1_agreement must be in [0, 1].")
        if not 0.0 <= metrics["generated_repetition_rate"] <= 1.0:
            raise ValueError(f"{label} generated_repetition_rate must be in [0, 1].")
        if metrics["generated_longest_run"] < 0.0 or not metrics[
            "generated_longest_run"
        ].is_integer():
            raise ValueError(
                f"{label} generated_longest_run must be a non-negative integer."
            )

    deltas = {
        "parent_top1_agreement": candidate_metrics["parent_top1_agreement"]
        - incumbent_metrics["parent_top1_agreement"],
        "mean_parent_action_logprob": candidate_metrics[
            "mean_parent_action_logprob"
        ]
        - incumbent_metrics["mean_parent_action_logprob"],
        "generated_repetition_rate": candidate_metrics[
            "generated_repetition_rate"
        ]
        - incumbent_metrics["generated_repetition_rate"],
        "generated_longest_run": candidate_metrics["generated_longest_run"]
        - incumbent_metrics["generated_longest_run"],
    }
    thresholds = {
        "min_parent_top1_agreement_gain": normalized_min_top1_gain,
        "min_parent_action_logprob_gain": normalized_min_logprob_gain,
        "max_generated_repetition_rate_regression": normalized_max_repetition,
        "max_generated_longest_run_regression": normalized_max_run,
    }

    veto_reasons = []
    if deltas["generated_repetition_rate"] > normalized_max_repetition:
        veto_reasons.append("generated_repetition_regression_above_threshold")
    if deltas["generated_longest_run"] > normalized_max_run:
        veto_reasons.append("generated_longest_run_regression_above_threshold")

    if veto_reasons:
        accepted = False
        reasons = veto_reasons
    elif deltas["parent_top1_agreement"] < 0.0:
        accepted = False
        reasons = ["parent_top1_agreement_regressed"]
    elif (
        deltas["parent_top1_agreement"] > 0.0
        and deltas["parent_top1_agreement"] >= normalized_min_top1_gain
    ):
        accepted = True
        reasons = ["parent_top1_agreement_improved"]
    elif (
        deltas["mean_parent_action_logprob"] > 0.0
        and deltas["mean_parent_action_logprob"] >= normalized_min_logprob_gain
    ):
        accepted = True
        reasons = ["parent_action_logprob_improved_after_top1_tie"]
    elif (
        0.0 < deltas["parent_top1_agreement"] < normalized_min_top1_gain
    ):
        accepted = False
        reasons = ["parent_top1_agreement_gain_below_threshold"]
    elif deltas["mean_parent_action_logprob"] > 0.0:
        accepted = False
        reasons = ["parent_action_logprob_gain_below_threshold"]
    elif deltas["mean_parent_action_logprob"] < 0.0:
        accepted = False
        reasons = ["parent_action_logprob_regressed_after_top1_tie"]
    else:
        accepted = False
        reasons = ["no_parent_alignment_improvement"]

    incumbent_hash = _json_sha256(incumbent)
    candidate_hash = _json_sha256(candidate)
    decision: dict[str, Any] = {
        "schema": ROLLOUT_PROMOTION_SCHEMA,
        "accepted": accepted,
        "reverted": not accepted,
        "decision": "accepted" if accepted else "reverted",
        "reason": reasons[0],
        "reasons": reasons,
        "accepted_reason": reasons[0] if accepted else None,
        "reverted_reason": reasons[0] if not accepted else None,
        "thresholds": thresholds,
        "deltas": deltas,
        "incumbent_metrics": incumbent_metrics,
        "candidate_metrics": candidate_metrics,
        "hashes": {
            "incumbent_report_sha256": incumbent_hash,
            "candidate_report_sha256": candidate_hash,
        },
    }
    decision["hashes"]["decision_sha256"] = _report_hash(decision)
    return decision


compare_hf_rollout_gate = compare_rollout_gate_reports
decide_rollout_gate_promotion = compare_rollout_gate_reports
