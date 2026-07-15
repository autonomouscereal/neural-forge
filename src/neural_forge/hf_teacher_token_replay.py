"""Deterministic teacher-derived replay construction for cross-tokenizer recovery."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any

from neural_forge import hf_recovery
from neural_forge.hf_compiled_child_cache import (
    MODEL_STATE_FINGERPRINT_SCHEMA,
    model_state_fingerprint,
)


TEACHER_TOKEN_REPLAY_SCHEMA = "neural-forge.teacher-token-replay.v1"

_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "status",
        "calibration_texts",
        "stepwise_scheduled_steps",
        "on_policy_scheduled_steps",
        "stepwise_replay_bank",
        "on_policy_replay_bank",
        "precomputed_teacher_generated_texts",
        "tokenizer_fingerprints",
        "source_model_fingerprint",
        "parameters",
        "provenance_sha256",
        "payload_sha256",
    }
)
_TOKENIZER_FINGERPRINT_FIELDS = frozenset(
    {
        "class",
        "tokenizer_length",
        "vocabulary_entry_count",
        "full_vocabulary_stream_sha256",
        "backend_tokenizer_json_sha256",
        "special_token_ids",
        "padding_side",
        "truncation_side",
        "sha256",
    }
)
_SPECIAL_TOKEN_ID_FIELDS = frozenset(
    {
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
        "all_special_ids",
    }
)
_MODEL_STATE_FINGERPRINT_FIELDS = frozenset(
    {
        "schema",
        "model_class",
        "state_dict_sha256",
        "compiler_state_sha256",
        "state_key_sha256",
        "state_manifest_sha256",
        "state_key_count",
        "logical_state_elements",
        "unique_parameter_count",
        "logical_parameter_count",
        "parameter_topology_sha256",
        "buffer_topology_sha256",
        "all_state_tensors_finite",
        "sha256",
    }
)
_MODEL_FINGERPRINT_FIELDS = frozenset(
    {
        "class",
        "model_state_fingerprint",
        "config_sha256",
        "generation_config_sha256",
        "sha256",
    }
)
_PARAMETER_FIELDS = frozenset(
    {
        "steps",
        "batch_size",
        "max_length",
        "response_loss_mode",
        "calibration_order",
        "stepwise_donor_max_new_tokens",
        "on_policy_max_new_tokens",
        "teacher_generate_max_new_tokens",
        "on_policy_interval",
        "device",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "calibration_texts",
        "parameters",
        "source_tokenizer",
        "candidate_tokenizer",
        "source_model",
        "source_model_state",
        "stepwise_replay_bank",
        "on_policy_replay_bank",
        "precomputed_teacher_generated_texts",
    }
)


class TeacherTokenReplayIntegrityError(ValueError):
    """A frozen teacher replay payload failed strict authentication."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _ordered_token_vocabulary_sha256(vocabulary: Mapping[str, int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"neural-forge.tokenizer-vocabulary.v1")
    digest.update(len(vocabulary).to_bytes(8, byteorder="big", signed=False))
    for token, token_id in sorted(
        vocabulary.items(), key=lambda item: item[0].encode("utf-8")
    ):
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
        digest.update(int(token_id).to_bytes(8, byteorder="big", signed=False))
    return digest.hexdigest()


def _backend_tokenizer_json(tokenizer: Any) -> str | None:
    for attribute in ("backend_tokenizer", "_tokenizer"):
        backend = getattr(tokenizer, attribute, None)
        to_str = getattr(backend, "to_str", None)
        if callable(to_str):
            value = to_str()
            if isinstance(value, str):
                return value
    return None


def _special_token_ids(tokenizer: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    ):
        value = getattr(tokenizer, name, None)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise TypeError(f"{name} must be an integer or None.")
        values[name] = None if value is None else int(value)
    all_special_ids = getattr(tokenizer, "all_special_ids", ())
    if isinstance(all_special_ids, (str, bytes)) or not isinstance(
        all_special_ids, Sequence
    ):
        raise TypeError("all_special_ids must be a sequence of integer token IDs.")
    normalized: list[int] = []
    for index, token_id in enumerate(all_special_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"all_special_ids[{index}] must be an integer token ID.")
        normalized.append(int(token_id))
    values["all_special_ids"] = sorted(set(normalized))
    return values


def tokenizer_functional_fingerprint(tokenizer: Any) -> dict[str, Any]:
    """Hash the tokenizer behavior-affecting vocabulary, backend, and special IDs."""

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise TypeError("Tokenizer must expose get_vocab().")
    raw_vocabulary = get_vocab()
    if not isinstance(raw_vocabulary, Mapping) or not raw_vocabulary:
        raise ValueError("Tokenizer get_vocab() must return a nonempty mapping.")
    vocabulary: dict[str, int] = {}
    for token, token_id in raw_vocabulary.items():
        if not isinstance(token, str):
            raise TypeError("Tokenizer vocabulary tokens must be strings.")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("Tokenizer vocabulary IDs must be integers.")
        if token_id < 0:
            raise ValueError("Tokenizer vocabulary IDs must be nonnegative.")
        vocabulary[token] = int(token_id)
    backend_json = _backend_tokenizer_json(tokenizer)
    try:
        tokenizer_length = int(len(tokenizer))
    except (TypeError, ValueError) as error:
        raise ValueError("Tokenizer must expose a positive length.") from error
    if tokenizer_length <= 0:
        raise ValueError("Tokenizer must expose a positive length.")
    body = {
        "class": _class_name(tokenizer),
        "tokenizer_length": tokenizer_length,
        "vocabulary_entry_count": len(vocabulary),
        "full_vocabulary_stream_sha256": _ordered_token_vocabulary_sha256(vocabulary),
        "backend_tokenizer_json_sha256": (
            None
            if backend_json is None
            else hashlib.sha256(backend_json.encode("utf-8")).hexdigest()
        ),
        "special_token_ids": _special_token_ids(tokenizer),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }
    return {**body, "sha256": _canonical_sha256(body)}


def _model_functional_fingerprint(model: Any) -> dict[str, Any]:
    state_body = model_state_fingerprint(model)
    state_fingerprint = {**state_body, "sha256": _canonical_sha256(state_body)}
    body: dict[str, Any] = {
        "class": _class_name(model),
        "model_state_fingerprint": state_fingerprint,
        "config_sha256": None,
        "generation_config_sha256": None,
    }
    for name in ("config", "generation_config"):
        value = getattr(model, name, None)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                body[f"{name}_sha256"] = _canonical_sha256(dict(payload))
    return {**body, "sha256": _canonical_sha256(body)}


@contextmanager
def _temporary_padding_side(tokenizer: Any, padding_side: str):
    original = getattr(tokenizer, "padding_side", None)
    if original is None:
        yield
        return
    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        tokenizer.padding_side = original


def _frozen_texts(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError("calibration_texts must be a sequence of strings.")
    frozen: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"calibration_texts[{index}] must be a string.")
        if not text.strip():
            raise ValueError(f"calibration_texts[{index}] must not be blank.")
        frozen.append(text)
    if not frozen:
        raise ValueError("calibration_texts must contain at least one text.")
    return tuple(frozen)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _normalize_json_value(value: Any, *, name: str) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TeacherTokenReplayIntegrityError(f"{name} contains a non-string key.")
        return {
            key: _normalize_json_value(value[key], name=f"{name}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TeacherTokenReplayIntegrityError(
        f"{name} must contain only finite normalized JSON values."
    )


def _immutable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _immutable_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_immutable_json_value(item) for item in value)
    return value


def _require_exact_fields(
    value: Any,
    fields: frozenset[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TeacherTokenReplayIntegrityError(
            f"{name} must contain exactly the {sorted(fields)!r} fields."
        )
    return dict(value)


def _verified_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TeacherTokenReplayIntegrityError(
            f"{name} must be a lowercase hexadecimal SHA-256 digest."
        )
    return value


def _verified_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TeacherTokenReplayIntegrityError(f"{name} must be a positive integer.")
    return int(value)


def _verify_tokenizer_fingerprint(value: Any, *, name: str) -> dict[str, Any]:
    fingerprint = _require_exact_fields(
        value,
        _TOKENIZER_FINGERPRINT_FIELDS,
        name=name,
    )
    claimed = _verified_sha256(fingerprint.pop("sha256"), name=f"{name}.sha256")
    if _canonical_sha256(fingerprint) != claimed:
        raise TeacherTokenReplayIntegrityError(f"{name} self-hash does not verify.")
    if not isinstance(fingerprint["class"], str) or not fingerprint["class"]:
        raise TeacherTokenReplayIntegrityError(f"{name}.class must be nonblank.")
    _verified_positive_int(
        fingerprint["tokenizer_length"],
        name=f"{name}.tokenizer_length",
    )
    _verified_positive_int(
        fingerprint["vocabulary_entry_count"],
        name=f"{name}.vocabulary_entry_count",
    )
    _verified_sha256(
        fingerprint["full_vocabulary_stream_sha256"],
        name=f"{name}.full_vocabulary_stream_sha256",
    )
    backend_sha = fingerprint["backend_tokenizer_json_sha256"]
    if backend_sha is not None:
        _verified_sha256(backend_sha, name=f"{name}.backend_tokenizer_json_sha256")
    for field in ("padding_side", "truncation_side"):
        if fingerprint[field] not in {None, "left", "right"}:
            raise TeacherTokenReplayIntegrityError(
                f"{name}.{field} must be left, right, or null."
            )
    specials = _require_exact_fields(
        fingerprint["special_token_ids"],
        _SPECIAL_TOKEN_ID_FIELDS,
        name=f"{name}.special_token_ids",
    )
    for field in _SPECIAL_TOKEN_ID_FIELDS - {"all_special_ids"}:
        token_id = specials[field]
        if token_id is not None and (
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        ):
            raise TeacherTokenReplayIntegrityError(
                f"{name}.special_token_ids.{field} must be a nonnegative integer or null."
            )
    all_special_ids = specials["all_special_ids"]
    if not isinstance(all_special_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in all_special_ids
    ):
        raise TeacherTokenReplayIntegrityError(
            f"{name}.special_token_ids.all_special_ids must be nonnegative integers."
        )
    if all_special_ids != sorted(set(all_special_ids)):
        raise TeacherTokenReplayIntegrityError(
            f"{name}.special_token_ids.all_special_ids must be sorted and unique."
        )
    return {**fingerprint, "sha256": claimed}


def _verify_model_fingerprint(value: Any) -> dict[str, Any]:
    fingerprint = _require_exact_fields(
        value,
        _MODEL_FINGERPRINT_FIELDS,
        name="source_model_fingerprint",
    )
    claimed = _verified_sha256(
        fingerprint.pop("sha256"),
        name="source_model_fingerprint.sha256",
    )
    if _canonical_sha256(fingerprint) != claimed:
        raise TeacherTokenReplayIntegrityError(
            "source_model_fingerprint self-hash does not verify."
        )
    if not isinstance(fingerprint["class"], str) or not fingerprint["class"]:
        raise TeacherTokenReplayIntegrityError(
            "source_model_fingerprint.class must be nonblank."
        )
    for field in ("config_sha256", "generation_config_sha256"):
        if fingerprint[field] is not None:
            _verified_sha256(
                fingerprint[field],
                name=f"source_model_fingerprint.{field}",
            )

    state = _require_exact_fields(
        fingerprint["model_state_fingerprint"],
        _MODEL_STATE_FINGERPRINT_FIELDS,
        name="source_model_fingerprint.model_state_fingerprint",
    )
    state_claimed = _verified_sha256(
        state.pop("sha256"),
        name="source_model_fingerprint.model_state_fingerprint.sha256",
    )
    if _canonical_sha256(state) != state_claimed:
        raise TeacherTokenReplayIntegrityError(
            "source model-state fingerprint self-hash does not verify."
        )
    if state["schema"] != MODEL_STATE_FINGERPRINT_SCHEMA:
        raise TeacherTokenReplayIntegrityError(
            "source model-state fingerprint schema is unsupported."
        )
    if state["model_class"] != fingerprint["class"]:
        raise TeacherTokenReplayIntegrityError(
            "source model-state class differs from the functional fingerprint."
        )
    for field in (
        "state_dict_sha256",
        "compiler_state_sha256",
        "state_key_sha256",
        "state_manifest_sha256",
        "parameter_topology_sha256",
        "buffer_topology_sha256",
    ):
        _verified_sha256(
            state[field],
            name=f"source_model_fingerprint.model_state_fingerprint.{field}",
        )
    for field in (
        "state_key_count",
        "logical_state_elements",
        "unique_parameter_count",
        "logical_parameter_count",
    ):
        _verified_positive_int(
            state[field],
            name=f"source_model_fingerprint.model_state_fingerprint.{field}",
        )
    if state["all_state_tensors_finite"] is not True:
        raise TeacherTokenReplayIntegrityError(
            "source model-state fingerprint must attest finite tensors."
        )
    fingerprint["model_state_fingerprint"] = {**state, "sha256": state_claimed}
    return {**fingerprint, "sha256": claimed}


def _verify_parameters(value: Any) -> dict[str, Any]:
    parameters = _require_exact_fields(value, _PARAMETER_FIELDS, name="parameters")
    for field in (
        "steps",
        "batch_size",
        "max_length",
        "stepwise_donor_max_new_tokens",
        "on_policy_max_new_tokens",
        "teacher_generate_max_new_tokens",
        "on_policy_interval",
    ):
        _verified_positive_int(parameters[field], name=f"parameters.{field}")
    if parameters["max_length"] <= 1:
        raise TeacherTokenReplayIntegrityError(
            "parameters.max_length must be greater than one."
        )
    if parameters["response_loss_mode"] not in {
        "none",
        "auto",
        "assistant",
        "answer",
    }:
        raise TeacherTokenReplayIntegrityError(
            "parameters.response_loss_mode is unsupported."
        )
    if parameters["calibration_order"] not in {"original", "domain-balanced"}:
        raise TeacherTokenReplayIntegrityError(
            "parameters.calibration_order is unsupported."
        )
    if not isinstance(parameters["device"], str) or not parameters["device"]:
        raise TeacherTokenReplayIntegrityError("parameters.device must be nonblank.")
    return parameters


def _verify_step_sequence(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(step, bool) or not isinstance(step, int) or step <= 0
        for step in value
    ):
        raise TeacherTokenReplayIntegrityError(
            f"{name} must be an array of positive integer steps."
        )
    return tuple(int(step) for step in value)


def _candidate_vocabulary(tokenizer: Any) -> tuple[frozenset[int], int]:
    raw = tokenizer.get_vocab()
    ids = frozenset(int(token_id) for token_id in raw.values())
    limit = int(len(tokenizer))
    if not ids or limit <= 0:
        raise ValueError("Candidate tokenizer vocabulary must be nonempty.")
    return ids, limit


def _validate_candidate_ids(
    token_ids: Sequence[int],
    *,
    vocabulary_ids: frozenset[int],
    vocabulary_limit: int,
    name: str,
) -> tuple[int, ...]:
    frozen: list[int] = []
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"{name}[{index}] must be an integer token ID.")
        if (
            token_id < 0
            or token_id >= vocabulary_limit
            or token_id not in vocabulary_ids
        ):
            raise ValueError(f"{name}[{index}] is outside the candidate vocabulary.")
        frozen.append(int(token_id))
    if not frozen:
        raise ValueError(f"{name} must contain at least one token ID.")
    return tuple(frozen)


def _batch_prompt_ids(
    torch: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int,
) -> tuple[tuple[int, ...], ...]:
    with _temporary_padding_side(tokenizer, "left"):
        encoded = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        raise ValueError(
            "Tokenizer must return attention_mask for exact prompt construction."
        )
    input_ids = encoded["input_ids"].detach().cpu()
    attention_mask = attention_mask.detach().cpu().bool()
    if int(input_ids.shape[0]) != len(prompts):
        raise ValueError("Tokenizer returned the wrong prompt batch size.")
    return tuple(
        tuple(
            int(token_id)
            for token_id in input_ids[index][attention_mask[index]].tolist()
        )
        for index in range(len(prompts))
    )


def _flat_input_ids(encoded: Mapping[str, Any], name: str) -> tuple[int, ...]:
    raw_ids = encoded.get("input_ids")
    if raw_ids is None:
        raise ValueError(f"{name} tokenizer output is missing input_ids.")
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()
    if (
        raw_ids
        and isinstance(raw_ids[0], Sequence)
        and not isinstance(raw_ids[0], (str, bytes))
    ):
        if len(raw_ids) != 1:
            raise ValueError(f"{name} tokenizer returned an unexpected batch.")
        raw_ids = raw_ids[0]
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        raise TypeError(f"{name} tokenizer input_ids must be a sequence.")
    result: list[int] = []
    for index, token_id in enumerate(raw_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"{name} input_ids[{index}] must be an integer.")
        result.append(int(token_id))
    return tuple(result)


def _eos_ids(tokenizer: Any) -> frozenset[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return frozenset()
    if isinstance(value, int) and not isinstance(value, bool):
        return frozenset((int(value),))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise TypeError("eos_token_id sequence must contain integers.")
        return frozenset(int(item) for item in value)
    raise TypeError("eos_token_id must be an integer, an integer sequence, or None.")


def _source_pad_id(tokenizer: Any, eos_ids: frozenset[int]) -> int | None:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        return min(eos_ids) if eos_ids else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("pad_token_id must be an integer or None.")
    return int(value)


def _source_continuation_ids(
    generated_row: Sequence[int],
    *,
    prompt_width: int,
    max_new_tokens: int,
    eos_ids: frozenset[int],
    pad_id: int | None,
) -> tuple[int, ...]:
    continuation = [int(token_id) for token_id in generated_row[prompt_width:]]
    if len(continuation) > max_new_tokens:
        raise ValueError("Source generation exceeded max_new_tokens.")
    for index, token_id in enumerate(continuation):
        if token_id in eos_ids:
            continuation = continuation[: index + 1]
            break
    if pad_id is not None and int(pad_id) not in eos_ids:
        while continuation and continuation[-1] == int(pad_id):
            continuation.pop()
    if not continuation:
        raise ValueError("Source generation produced no continuation tokens.")
    return tuple(continuation)


def _teacher_rollout_batch(
    *,
    torch: Any,
    source_model: Any,
    source_tokenizer: Any,
    candidate_tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    max_length: int,
    device: str,
    candidate_vocabulary_ids: frozenset[int],
    candidate_vocabulary_limit: int,
) -> list[dict[str, list[int]]]:
    prompt_max_length = max(2, int(max_length) - int(max_new_tokens))
    candidate_prompt_rows = _batch_prompt_ids(
        torch,
        candidate_tokenizer,
        prompts,
        max_length=prompt_max_length,
    )
    with _temporary_padding_side(source_tokenizer, "left"):
        source_batch = source_tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        )
    source_batch = {key: value.to(device) for key, value in source_batch.items()}
    source_prompt_width = int(source_batch["input_ids"].shape[1])
    eos_ids = _eos_ids(source_tokenizer)
    pad_id = _source_pad_id(source_tokenizer, eos_ids)
    with torch.no_grad():
        generated = source_model.generate(
            **source_batch,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=getattr(source_tokenizer, "eos_token_id", None),
        )
    generated = getattr(generated, "sequences", generated).detach().cpu()
    if int(generated.shape[0]) != len(prompts):
        raise ValueError("Source generation returned the wrong batch size.")

    rows: list[dict[str, list[int]]] = []
    for row_index, candidate_prompt_ids in enumerate(candidate_prompt_rows):
        prompt_ids = _validate_candidate_ids(
            candidate_prompt_ids,
            vocabulary_ids=candidate_vocabulary_ids,
            vocabulary_limit=candidate_vocabulary_limit,
            name=f"candidate_prompt_input_ids[{row_index}]",
        )
        if len(prompt_ids) > prompt_max_length:
            raise ValueError(
                "Candidate prompt exceeds the exact runtime prompt maximum."
            )
        source_continuation_ids = _source_continuation_ids(
            generated[row_index].tolist(),
            prompt_width=source_prompt_width,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            pad_id=pad_id,
        )
        continuation_text = source_tokenizer.decode(
            list(source_continuation_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(continuation_text, str) or not continuation_text.strip():
            raise ValueError("Source continuation decodes to blank text.")
        candidate_continuation_ids = _flat_input_ids(
            candidate_tokenizer(
                continuation_text,
                add_special_tokens=False,
                truncation=False,
            ),
            "candidate continuation",
        )
        continuation_ids = _validate_candidate_ids(
            candidate_continuation_ids,
            vocabulary_ids=candidate_vocabulary_ids,
            vocabulary_limit=candidate_vocabulary_limit,
            name=f"candidate_continuation_input_ids[{row_index}]",
        )
        if len(continuation_ids) > max_new_tokens:
            raise ValueError(
                "Candidate continuation exceeds max_new_tokens after re-encoding."
            )
        rollout_ids = prompt_ids + continuation_ids
        if len(rollout_ids) > max_length:
            raise ValueError("Teacher-derived candidate rollout exceeds max_length.")
        if rollout_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                "Teacher-derived rollout lost its exact candidate prompt prefix."
            )
        rows.append(
            {
                "prompt_input_ids": list(prompt_ids),
                "rollout_input_ids": list(rollout_ids),
            }
        )
    return rows


def _teacher_full_text_batch(
    *,
    torch: Any,
    source_model: Any,
    source_tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    max_length: int,
    device: str,
) -> list[str]:
    """Generate and source-decode complete rows for frozen text replay."""

    prompt_max_length = max(2, int(max_length) - int(max_new_tokens))
    eos_ids = _eos_ids(source_tokenizer)
    pad_id = _source_pad_id(source_tokenizer, eos_ids)
    with _temporary_padding_side(source_tokenizer, "left"):
        encoded = source_tokenizer(
            list(prompts),
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
                eos_token_id=getattr(source_tokenizer, "eos_token_id", None),
            )
    generated = getattr(generated, "sequences", generated).detach().cpu()
    if int(generated.shape[0]) != len(prompts):
        raise ValueError("Source generation returned the wrong batch size.")
    full_texts: list[str] = []
    for row in generated:
        text = source_tokenizer.decode(
            row.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not text:
            raise ValueError("Source-generated replay decodes to blank text.")
        full_texts.append(text)
    return full_texts


def build_teacher_token_replay_payload(
    *,
    torch: Any,
    source_model: Any,
    source_tokenizer: Any,
    candidate_tokenizer: Any,
    calibration_texts: Sequence[str],
    steps: int,
    batch_size: int,
    max_length: int,
    response_loss_mode: str,
    calibration_order: str,
    stepwise_donor_max_new_tokens: int,
    on_policy_max_new_tokens: int,
    teacher_generate_max_new_tokens: int,
    on_policy_interval: int,
    device: str,
) -> dict[str, Any]:
    """Construct source-generated, candidate-token-exact replay without loading models."""

    steps = _positive_int(steps, "steps")
    batch_size = _positive_int(batch_size, "batch_size")
    max_length = _positive_int(max_length, "max_length")
    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    stepwise_donor_max_new_tokens = _positive_int(
        stepwise_donor_max_new_tokens,
        "stepwise_donor_max_new_tokens",
    )
    on_policy_max_new_tokens = _positive_int(
        on_policy_max_new_tokens, "on_policy_max_new_tokens"
    )
    teacher_generate_max_new_tokens = _positive_int(
        teacher_generate_max_new_tokens,
        "teacher_generate_max_new_tokens",
    )
    on_policy_interval = _positive_int(on_policy_interval, "on_policy_interval")
    if response_loss_mode not in {"none", "auto", "assistant", "answer"}:
        raise ValueError("response_loss_mode must be none, auto, assistant, or answer.")
    if not isinstance(device, str) or not device:
        raise TypeError("device must be a nonempty string.")
    texts = tuple(
        hf_recovery._order_recovery_texts(
            _frozen_texts(calibration_texts),
            calibration_order,
        )
    )
    source_fingerprint = tokenizer_functional_fingerprint(source_tokenizer)
    candidate_fingerprint = tokenizer_functional_fingerprint(candidate_tokenizer)
    source_model_fingerprint = _model_functional_fingerprint(source_model)
    candidate_vocabulary_ids, candidate_vocabulary_limit = _candidate_vocabulary(
        candidate_tokenizer
    )
    text_batches = tuple(
        texts[start : start + batch_size] for start in range(0, len(texts), batch_size)
    )
    stepwise_steps = tuple(range(1, steps + 1))
    on_policy_steps = tuple(
        step for step in stepwise_steps if step == 1 or step % on_policy_interval == 0
    )
    stepwise_events: list[dict[str, Any]] = []
    on_policy_batches: list[list[str]] = []
    teacher_texts: list[str] = []
    was_training = bool(getattr(source_model, "training", False))
    source_model.eval()
    try:
        for step in stepwise_steps:
            raw_batch = text_batches[(step - 1) % len(text_batches)]
            prompts = tuple(
                prompt
                for prompt in (
                    hf_recovery._on_policy_prompt_text(text, response_loss_mode)
                    for text in raw_batch
                )
                if prompt.strip()
            )
            if len(prompts) != len(raw_batch):
                raise ValueError(
                    "Every scheduled calibration row must produce one nonblank prompt."
                )
            rows = _teacher_rollout_batch(
                torch=torch,
                source_model=source_model,
                source_tokenizer=source_tokenizer,
                candidate_tokenizer=candidate_tokenizer,
                prompts=prompts,
                max_new_tokens=stepwise_donor_max_new_tokens,
                max_length=max_length,
                device=device,
                candidate_vocabulary_ids=candidate_vocabulary_ids,
                candidate_vocabulary_limit=candidate_vocabulary_limit,
            )
            stepwise_events.append({"step": step, "rows": rows})

        for step in on_policy_steps:
            raw_batch = text_batches[(step - 1) % len(text_batches)]
            prompts = tuple(
                prompt
                for prompt in (
                    hf_recovery._on_policy_prompt_text(text, response_loss_mode)
                    for text in raw_batch
                )
                if prompt.strip()
            )
            if len(prompts) != len(raw_batch):
                raise ValueError(
                    "Every scheduled calibration row must produce one nonblank prompt."
                )
            full_texts = _teacher_full_text_batch(
                torch=torch,
                source_model=source_model,
                source_tokenizer=source_tokenizer,
                prompts=prompts,
                max_new_tokens=on_policy_max_new_tokens,
                max_length=max_length,
                device=device,
            )
            on_policy_batches.append(full_texts)

        teacher_texts = hf_recovery._teacher_generated_lm_texts(
            torch,
            source_model,
            source_tokenizer,
            texts,
            response_loss_mode,
            teacher_generate_max_new_tokens,
            max_length,
            batch_size,
            device,
        )
        if not teacher_texts:
            raise ValueError(
                "Teacher generation produced no nonblank language-model texts."
            )
    finally:
        source_model.train(was_training)

    frozen_stepwise = hf_recovery._freeze_stepwise_replay_bank(stepwise_events)
    frozen_on_policy = hf_recovery._freeze_on_policy_replay_bank(on_policy_batches)
    parameters = {
        "steps": steps,
        "batch_size": batch_size,
        "max_length": max_length,
        "response_loss_mode": response_loss_mode,
        "calibration_order": calibration_order,
        "stepwise_donor_max_new_tokens": stepwise_donor_max_new_tokens,
        "on_policy_max_new_tokens": on_policy_max_new_tokens,
        "teacher_generate_max_new_tokens": teacher_generate_max_new_tokens,
        "on_policy_interval": on_policy_interval,
        "device": device,
    }
    body = {
        "schema": TEACHER_TOKEN_REPLAY_SCHEMA,
        "status": "frozen",
        "calibration_texts": list(texts),
        "stepwise_scheduled_steps": list(stepwise_steps),
        "on_policy_scheduled_steps": list(on_policy_steps),
        "stepwise_replay_bank": stepwise_events,
        "on_policy_replay_bank": on_policy_batches,
        "precomputed_teacher_generated_texts": teacher_texts,
        "tokenizer_fingerprints": {
            "source": source_fingerprint,
            "candidate": candidate_fingerprint,
        },
        "source_model_fingerprint": source_model_fingerprint,
        "parameters": parameters,
        "provenance_sha256": {
            "calibration_texts": hf_recovery._ordered_recovery_texts_sha256(texts),
            "parameters": _canonical_sha256(parameters),
            "source_tokenizer": source_fingerprint["sha256"],
            "candidate_tokenizer": candidate_fingerprint["sha256"],
            "source_model": source_model_fingerprint["sha256"],
            "source_model_state": source_model_fingerprint["model_state_fingerprint"][
                "state_dict_sha256"
            ],
            "stepwise_replay_bank": hf_recovery._stepwise_replay_bank_sha256(
                frozen_stepwise
            ),
            "on_policy_replay_bank": hf_recovery._on_policy_replay_bank_sha256(
                frozen_on_policy
            ),
            "precomputed_teacher_generated_texts": hf_recovery._ordered_recovery_texts_sha256(
                teacher_texts
            ),
        },
    }
    return {**body, "payload_sha256": _canonical_sha256(body)}


def verify_teacher_token_replay_payload(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate a frozen replay payload without loading models or tokenizers."""

    if not isinstance(payload, Mapping):
        raise TeacherTokenReplayIntegrityError(
            "Teacher replay payload must be a mapping."
        )
    normalized = _normalize_json_value(payload, name="payload")
    top = _require_exact_fields(
        normalized,
        _PAYLOAD_FIELDS,
        name="teacher replay payload",
    )
    if top["schema"] != TEACHER_TOKEN_REPLAY_SCHEMA or top["status"] != "frozen":
        raise TeacherTokenReplayIntegrityError(
            "Teacher replay payload schema or frozen status is invalid."
        )
    payload_claimed = _verified_sha256(
        top["payload_sha256"],
        name="payload_sha256",
    )
    payload_body = dict(top)
    payload_body.pop("payload_sha256")
    if _canonical_sha256(payload_body) != payload_claimed:
        raise TeacherTokenReplayIntegrityError(
            "Teacher replay payload self-hash does not verify."
        )

    parameters = _verify_parameters(top["parameters"])
    try:
        texts = _frozen_texts(top["calibration_texts"])
    except (TypeError, ValueError) as error:
        raise TeacherTokenReplayIntegrityError(
            "calibration_texts are malformed."
        ) from error
    expected_order = tuple(
        hf_recovery._order_recovery_texts(
            texts,
            parameters["calibration_order"],
        )
    )
    if expected_order != texts:
        raise TeacherTokenReplayIntegrityError(
            "calibration_texts do not satisfy calibration_order."
        )

    tokenizer_fingerprints = _require_exact_fields(
        top["tokenizer_fingerprints"],
        frozenset({"source", "candidate"}),
        name="tokenizer_fingerprints",
    )
    source_tokenizer_fingerprint = _verify_tokenizer_fingerprint(
        tokenizer_fingerprints["source"],
        name="tokenizer_fingerprints.source",
    )
    candidate_tokenizer_fingerprint = _verify_tokenizer_fingerprint(
        tokenizer_fingerprints["candidate"],
        name="tokenizer_fingerprints.candidate",
    )
    source_model_fingerprint = _verify_model_fingerprint(
        top["source_model_fingerprint"]
    )

    stepwise_steps = _verify_step_sequence(
        top["stepwise_scheduled_steps"],
        name="stepwise_scheduled_steps",
    )
    expected_stepwise_steps = tuple(range(1, parameters["steps"] + 1))
    if stepwise_steps != expected_stepwise_steps:
        raise TeacherTokenReplayIntegrityError(
            "stepwise_scheduled_steps differ from parameters.steps."
        )
    on_policy_steps = _verify_step_sequence(
        top["on_policy_scheduled_steps"],
        name="on_policy_scheduled_steps",
    )
    expected_on_policy_steps = tuple(
        step
        for step in expected_stepwise_steps
        if step == 1 or step % parameters["on_policy_interval"] == 0
    )
    if on_policy_steps != expected_on_policy_steps:
        raise TeacherTokenReplayIntegrityError(
            "on_policy_scheduled_steps differ from the configured interval."
        )

    text_batches = tuple(
        texts[start : start + parameters["batch_size"]]
        for start in range(0, len(texts), parameters["batch_size"])
    )
    try:
        frozen_stepwise = hf_recovery._freeze_stepwise_replay_bank(
            top["stepwise_replay_bank"]
        )
    except (TypeError, ValueError) as error:
        raise TeacherTokenReplayIntegrityError(
            "stepwise_replay_bank is malformed."
        ) from error
    if len(frozen_stepwise) != len(stepwise_steps):
        raise TeacherTokenReplayIntegrityError(
            "stepwise_replay_bank event count differs from its schedule."
        )
    prompt_max_length = max(
        2,
        parameters["max_length"] - parameters["stepwise_donor_max_new_tokens"],
    )
    candidate_vocabulary_limit = candidate_tokenizer_fingerprint["tokenizer_length"]
    for event_index, event in enumerate(frozen_stepwise):
        step = stepwise_steps[event_index]
        expected_batch = text_batches[(step - 1) % len(text_batches)]
        if event.step != step or len(event.rows) != len(expected_batch):
            raise TeacherTokenReplayIntegrityError(
                "stepwise_replay_bank does not match its exact scheduled batch shape."
            )
        for row in event.rows:
            continuation_count = len(row.rollout_input_ids) - len(row.prompt_input_ids)
            if (
                len(row.prompt_input_ids) > prompt_max_length
                or len(row.rollout_input_ids) > parameters["max_length"]
                or continuation_count > parameters["stepwise_donor_max_new_tokens"]
                or any(
                    token_id >= candidate_vocabulary_limit
                    for token_id in row.rollout_input_ids
                )
            ):
                raise TeacherTokenReplayIntegrityError(
                    "stepwise_replay_bank violates token length or vocabulary limits."
                )

    try:
        frozen_on_policy = hf_recovery._freeze_on_policy_replay_bank(
            top["on_policy_replay_bank"]
        )
    except (TypeError, ValueError) as error:
        raise TeacherTokenReplayIntegrityError(
            "on_policy_replay_bank is malformed."
        ) from error
    if len(frozen_on_policy) != len(on_policy_steps):
        raise TeacherTokenReplayIntegrityError(
            "on_policy_replay_bank batch count differs from its schedule."
        )
    for batch_index, batch in enumerate(frozen_on_policy):
        step = on_policy_steps[batch_index]
        expected_batch = text_batches[(step - 1) % len(text_batches)]
        if len(batch) != len(expected_batch):
            raise TeacherTokenReplayIntegrityError(
                "on_policy_replay_bank does not match its exact scheduled batch shape."
            )

    try:
        teacher_texts = hf_recovery._freeze_ordered_recovery_texts(
            top["precomputed_teacher_generated_texts"],
            field_name="precomputed_teacher_generated_texts",
        )
    except (TypeError, ValueError) as error:
        raise TeacherTokenReplayIntegrityError(
            "precomputed_teacher_generated_texts are malformed."
        ) from error
    if not teacher_texts:
        raise TeacherTokenReplayIntegrityError(
            "precomputed_teacher_generated_texts must not be empty."
        )
    if len(teacher_texts) != len(set(teacher_texts)) or len(teacher_texts) > len(texts):
        raise TeacherTokenReplayIntegrityError(
            "precomputed_teacher_generated_texts violate recovery dedup semantics."
        )

    provenance = _require_exact_fields(
        top["provenance_sha256"],
        _PROVENANCE_FIELDS,
        name="provenance_sha256",
    )
    for field, value in provenance.items():
        _verified_sha256(value, name=f"provenance_sha256.{field}")
    state = source_model_fingerprint["model_state_fingerprint"]
    expected_provenance = {
        "calibration_texts": hf_recovery._ordered_recovery_texts_sha256(texts),
        "parameters": _canonical_sha256(parameters),
        "source_tokenizer": source_tokenizer_fingerprint["sha256"],
        "candidate_tokenizer": candidate_tokenizer_fingerprint["sha256"],
        "source_model": source_model_fingerprint["sha256"],
        "source_model_state": state["state_dict_sha256"],
        "stepwise_replay_bank": hf_recovery._stepwise_replay_bank_sha256(
            frozen_stepwise
        ),
        "on_policy_replay_bank": hf_recovery._on_policy_replay_bank_sha256(
            frozen_on_policy
        ),
        "precomputed_teacher_generated_texts": (
            hf_recovery._ordered_recovery_texts_sha256(teacher_texts)
        ),
    }
    if provenance != expected_provenance:
        raise TeacherTokenReplayIntegrityError(
            "Teacher replay provenance hashes do not match the frozen content."
        )
    return _immutable_json_value(normalized)
