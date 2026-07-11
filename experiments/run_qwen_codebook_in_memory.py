#!/usr/bin/env python3
"""Build and fully evaluate one zero-gradient partial Qwen3 codebook runtime."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import weakref
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from neural_forge.hf_attention_codebook import GQAAttentionBundle
from neural_forge.hf_codebook_alignment import (
    DEFAULT_ACTIVATION_SKETCH_SIZE,
    DEFAULT_LOCAL_BLOCK_SIZE,
    DEFAULT_MAX_EXACT_GQA_GROUPS,
    DEFAULT_REFINEMENT_PASSES,
    DEFAULT_SIGNATURE_BITS,
    DEFAULT_SIGNATURE_DIMENSIONS,
    DEFAULT_WEIGHT_SKETCH_SIZE,
    PermutationAlignmentResult,
    infer_qwen_gqa_group_permutations,
    infer_swiglu_neuron_permutations,
    invert_permutations,
    verify_permutation_bijections,
)
from neural_forge.hf_neuron_codebook import GatedMLPBundle
from neural_forge.hf_qwen_codebook_model import (
    RUNTIME_SCHEMA,
    load_qwen3_codebook_checkpoint,
    prepare_qwen3_codebook_model,
    qwen3_codebook_parameter_report,
    save_qwen3_codebook_checkpoint,
)
from neural_forge.hf_tied_depth import exact_token_row_mapping
from neural_forge.hf_tools import evaluate_loaded_text_model


RUN_SCHEMA = "neural-forge-qwen3-codebook-memory-run-v1"
ACTIVATION_SCHEMA = "neural-forge-qwen3-bounded-activation-cache-v1"
ALIGNMENT_SCHEMA = "neural-forge-qwen3-codebook-alignment-v1"
ACCOUNTING_SCHEMA = "neural-forge-qwen3-codebook-accounting-v1"
_PROJECTION_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_ROLES = ("gate_proj", "up_proj", "down_proj")


class QwenCodebookRunnerError(RuntimeError):
    """The production runner encountered an invalid or unproven state."""


@contextmanager
def _checkpoint_io_lease(path: Path):
    """Serialize multi-gigabyte checkpoint I/O across sibling GPU jobs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open("a+b") as handle:
        serialized = os.name == "posix"
        if serialized:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        lease = {
            "path": str(path),
            "serialized": serialized,
            "wait_seconds": float(time.monotonic() - started),
        }
        try:
            yield lease
        finally:
            if serialized:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(slots=True)
class ActivationCache:
    """Bounded CPU activation rows captured from every logical layer."""

    neuron_activations: tuple[Tensor, ...]
    attention_activations: tuple[Tensor, ...]
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class QwenBundleSet:
    """Non-copying dense projection bundles used only before conversion."""

    neurons: tuple[GatedMLPBundle, ...]
    attention: tuple[GQAAttentionBundle, ...]
    logical_layers: int
    intermediate_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int


@dataclass(frozen=True, slots=True)
class AlignmentResults:
    """Coupled neuron and GQA permutation inference results."""

    neurons: PermutationAlignmentResult
    attention: PermutationAlignmentResult


@dataclass(frozen=True, slots=True)
class DenseRuntimeSnapshot:
    """Identity-only evidence for objects that conversion must replace."""

    parameter_ids: frozenset[int]
    module_ids: frozenset[int]
    parameter_refs: tuple[weakref.ReferenceType[nn.Parameter], ...]
    module_refs: tuple[weakref.ReferenceType[nn.Module], ...]


@dataclass(slots=True)
class NoopReference:
    """Small target-vocabulary parent trace for the full-depth sanity run."""

    target_inputs: dict[str, Tensor]
    expected_logits: Tensor
    expected_embedding: Tensor | None = None
    vocabulary_scope: str = "mapped-target-vocabulary"
    source_vocab_size: int = 0
    checked_parent_rows: tuple[int, ...] = ()


def _write_json(path: Path, value: Any) -> None:
    _assert_json_finite(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_stage(output: Path, stage: str, status: str, **details: Any) -> None:
    _write_json(
        output / "run_state.json",
        {
            "schema": RUN_SCHEMA,
            "stage": stage,
            "status": status,
            "details": details,
            "final_report": str(output / "run_report.json"),
        },
    )


def _progress(event: str, **values: Any) -> None:
    fields = " ".join(f"{key}={values[key]}" for key in sorted(values))
    print(f"neural_forge_progress event={event} {fields}".rstrip(), flush=True)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_checkpoint_contract(
    tokenizer: Any,
    *,
    tokenizer_name: str,
    parent_rows: Sequence[int],
) -> dict[str, Any]:
    try:
        vocabulary = tokenizer.get_vocab()
    except Exception as exc:
        raise QwenCodebookRunnerError(
            "Target tokenizer cannot provide a vocabulary checkpoint contract."
        ) from exc
    if not isinstance(vocabulary, Mapping) or not vocabulary:
        raise QwenCodebookRunnerError("Target tokenizer vocabulary contract is empty.")
    normalized_vocab = {str(token): int(token_id) for token, token_id in vocabulary.items()}
    return {
        "tokenizer_name_or_path": str(
            getattr(tokenizer, "name_or_path", tokenizer_name)
        ),
        "vocab_size": len(normalized_vocab),
        "vocab_sha256": _canonical_sha256(normalized_vocab),
        "special_token_ids": {
            name: getattr(tokenizer, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        "parent_token_rows_sha256": _canonical_sha256(
            [int(row) for row in parent_rows]
        ),
    }


def _assert_json_finite(value: Any, path: str = "report") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QwenCodebookRunnerError(f"{path} contains a non-finite value.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")
        return
    raise QwenCodebookRunnerError(
        f"{path} contains unsupported report value {type(value).__name__}."
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QwenCodebookRunnerError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QwenCodebookRunnerError(f"{name} must be a positive integer.") from exc
    if result <= 0:
        raise QwenCodebookRunnerError(f"{name} must be a positive integer.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QwenCodebookRunnerError(f"{name} must be a non-negative integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QwenCodebookRunnerError(
            f"{name} must be a non-negative integer."
        ) from exc
    if result < 0:
        raise QwenCodebookRunnerError(f"{name} must be a non-negative integer.")
    return result


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _prototype_depth_argument(value: str) -> int | str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"logical_layers", "all", "full"}:
        return "logical_layers"
    try:
        result = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "prototype depth must be a positive integer or 'logical_layers'"
        ) from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("prototype depth must be positive")
    return result


def _resolve_prototype_depth(value: int | str, layers: int, name: str) -> int:
    result = layers if value == "logical_layers" else _positive_int(value, name)
    if result > layers:
        raise QwenCodebookRunnerError(
            f"{name}={result} exceeds logical_layers={layers}."
        )
    return result


def _messages_to_prompt(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    pieces: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = message.get("role")
        prefix = (
            f"{str(role).strip()}: " if isinstance(role, str) and role.strip() else ""
        )
        pieces.append(prefix + content.strip())
    return "\n".join(pieces) or None


def _calibration_prompt(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if not isinstance(item, Mapping):
        return None
    messages = _messages_to_prompt(item.get("messages"))
    if messages:
        return messages
    input_json = item.get("input_json")
    if isinstance(input_json, Mapping):
        messages = _messages_to_prompt(input_json.get("messages"))
        if messages:
            return messages
    for key in ("text", "prompt", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def load_calibration_prompts(path: Path) -> list[str]:
    """Strictly load non-empty prompt strings from a JSONL corpus."""

    if not path.is_file():
        raise FileNotFoundError(path)
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise QwenCodebookRunnerError(
                    f"Invalid calibration JSONL at {path}:{line_number}."
                ) from exc
            prompt = _calibration_prompt(item)
            if prompt is None:
                raise QwenCodebookRunnerError(
                    f"Calibration row {path}:{line_number} has no nonempty prompt."
                )
            prompts.append(prompt)
    if not prompts:
        raise QwenCodebookRunnerError("Calibration JSONL contains no prompts.")
    return prompts


def _even_indices(length: int, count: int) -> list[int]:
    if count <= 0 or count > length:
        raise QwenCodebookRunnerError(
            f"Cannot select {count} deterministic rows from a population of {length}."
        )
    if count == length:
        return list(range(length))
    return [((2 * index + 1) * length) // (2 * count) for index in range(count)]


def select_calibration_prompts(
    prompts: Sequence[str], count: int
) -> tuple[list[str], dict[str, Any]]:
    """Select a deterministic, corpus-spanning prompt subset."""

    requested = _positive_int(count, "calibration_count")
    if requested > len(prompts):
        raise QwenCodebookRunnerError(
            f"Need {requested} calibration prompts, but corpus has {len(prompts)}."
        )
    indices = _even_indices(len(prompts), requested)
    if len(set(indices)) != requested:
        raise QwenCodebookRunnerError(
            "Deterministic calibration prompt selection produced overlap."
        )
    selected = [str(prompts[index]) for index in indices]
    return selected, {
        "corpus_count": len(prompts),
        "selected_prompt_count": len(selected),
        "selection": "global-evenly-spaced",
        "selected_indices": indices,
        "corpus_sha256": _canonical_sha256(list(prompts)),
        "selected_prompts_sha256": _canonical_sha256(selected),
        "deterministic": True,
    }


def _spanning_fill_indices(length: int, preferred: Sequence[int]) -> list[int]:
    """Extend an evenly spaced seed with progressively denser global grids."""

    seen = set(int(index) for index in preferred)
    result = [int(index) for index in preferred]
    grid_size = max(1, len(result))
    while len(result) < length:
        grid_size = min(length, max(grid_size + 1, grid_size * 2))
        for index in _even_indices(length, grid_size):
            if index not in seen:
                seen.add(index)
                result.append(index)
        if grid_size == length:
            break
    for index in range(length):
        if index not in seen:
            result.append(index)
    if len(result) != length or len(set(result)) != length:
        raise QwenCodebookRunnerError(
            "Deterministic calibration candidate ordering is not a permutation."
        )
    return result


def select_token_disjoint_calibration_prompts(
    prompts: Sequence[str],
    count: int,
    *,
    source_tokenizer: Any,
    target_tokenizer: Any,
    eval_suite: Path | str | dict[str, Any],
    max_length: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Select unique paths disjoint after the exact calibration truncation."""

    requested = _positive_int(count, "calibration_count")
    length_limit = (
        _positive_int(max_length, "calibration_max_length")
        if max_length is not None
        else None
    )
    if requested > len(prompts):
        raise QwenCodebookRunnerError(
            f"Need {requested} calibration prompts, but corpus has {len(prompts)}."
        )
    eval_texts = _eval_split_texts(eval_suite)
    source_eval = {
        _token_ids(source_tokenizer, text, max_length=length_limit)
        for text in eval_texts
    }
    target_eval = {
        _token_ids(target_tokenizer, text, max_length=length_limit)
        for text in eval_texts
    }
    preferred = _even_indices(len(prompts), requested)
    candidates = _spanning_fill_indices(len(prompts), preferred)
    selected: list[str] = []
    selected_indices: list[int] = []
    source_selected: set[tuple[int, ...]] = set()
    target_selected: set[tuple[int, ...]] = set()
    rejected = {
        "source_evaluation_overlap": 0,
        "target_evaluation_overlap": 0,
        "source_duplicate": 0,
        "target_duplicate": 0,
    }
    for index in candidates:
        prompt = str(prompts[index])
        source_tokens = _token_ids(
            source_tokenizer, prompt, max_length=length_limit
        )
        target_tokens = _token_ids(
            target_tokenizer, prompt, max_length=length_limit
        )
        if source_tokens in source_eval:
            rejected["source_evaluation_overlap"] += 1
            continue
        if target_tokens in target_eval:
            rejected["target_evaluation_overlap"] += 1
            continue
        if source_tokens in source_selected:
            rejected["source_duplicate"] += 1
            continue
        if target_tokens in target_selected:
            rejected["target_duplicate"] += 1
            continue
        selected.append(prompt)
        selected_indices.append(index)
        source_selected.add(source_tokens)
        target_selected.add(target_tokens)
        if len(selected) == requested:
            break
    if len(selected) != requested:
        raise QwenCodebookRunnerError(
            "Calibration corpus has only "
            f"{len(selected)} usable unique token paths under both tokenizers; "
            f"{requested} were requested."
        )
    return selected, {
        "corpus_count": len(prompts),
        "selected_prompt_count": len(selected),
        "selection": "global-even-seed-with-progressive-spanning-fill",
        "selected_indices": selected_indices,
        "selected_prompt_ids": [
            f"calibration-{index:08d}-{_canonical_sha256(prompts[index])[:16]}"
            for index in selected_indices
        ],
        "preferred_indices": preferred,
        "candidate_indices_examined": candidates.index(selected_indices[-1]) + 1,
        "rejected": rejected,
        "corpus_sha256": _canonical_sha256(list(prompts)),
        "selected_prompts_sha256": _canonical_sha256(selected),
        "source_token_paths_sha256": _canonical_sha256(sorted(source_selected)),
        "target_token_paths_sha256": _canonical_sha256(sorted(target_selected)),
        "tokenization_max_length": length_limit,
        "token_paths_after_truncation": length_limit is not None,
        "deterministic": True,
    }


def _token_ids(
    tokenizer: Any, text: str, *, max_length: int | None = None
) -> tuple[int, ...]:
    try:
        if hasattr(tokenizer, "encode"):
            values = tokenizer.encode(text, add_special_tokens=True)
        else:
            encoded = tokenizer(text, add_special_tokens=True)
            values = encoded["input_ids"]
    except Exception as exc:
        raise QwenCodebookRunnerError(
            "Tokenizer failed to encode a split item."
        ) from exc
    if isinstance(values, Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    if values and isinstance(values[0], Sequence):
        if len(values) != 1:
            raise QwenCodebookRunnerError("Tokenizer returned a batched split item.")
        values = values[0]
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QwenCodebookRunnerError(
            "Tokenizer returned non-integer token IDs."
        ) from exc
    if not result:
        raise QwenCodebookRunnerError("Tokenizer produced an empty split item.")
    if max_length is not None:
        result = result[: _positive_int(max_length, "calibration_max_length")]
    if not result:
        raise QwenCodebookRunnerError("Tokenizer truncation produced an empty split item.")
    return result


def _eval_split_texts(eval_suite: Path | str | dict[str, Any]) -> list[str]:
    from neural_forge.hf_tools import _load_eval_suite

    suite = _load_eval_suite(eval_suite)
    texts = [str(item) for item in suite.get("text_evals", [])]
    for collection in ("choice_probes", "generation_prompts", "behavior_probes"):
        for item in suite.get(collection, []):
            if isinstance(item, Mapping) and isinstance(item.get("prompt"), str):
                texts.append(str(item["prompt"]))
    return [text for text in texts if text]


def verify_token_split_disjoint(
    tokenizer: Any,
    calibration_prompts: Sequence[str],
    eval_suite: Path | str | dict[str, Any],
    *,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Fail if calibration and evaluation contain an identical token sequence."""

    calibration_tokens = [
        _token_ids(tokenizer, text, max_length=max_length)
        for text in calibration_prompts
    ]
    if len(set(calibration_tokens)) != len(calibration_tokens):
        raise QwenCodebookRunnerError(
            "Calibration token split contains duplicate token sequences."
        )
    eval_texts = _eval_split_texts(eval_suite)
    eval_tokens = [
        _token_ids(tokenizer, text, max_length=max_length) for text in eval_texts
    ]
    overlap = set(calibration_tokens).intersection(eval_tokens)
    if overlap:
        raise QwenCodebookRunnerError(
            "Calibration and evaluation token splits overlap "
            f"on {len(overlap)} exact token sequence(s)."
        )
    return {
        "method": "exact-source-token-id-sequence",
        "calibration_sequence_count": len(calibration_tokens),
        "evaluation_sequence_count": len(eval_tokens),
        "calibration_unique_sequence_count": len(set(calibration_tokens)),
        "evaluation_unique_sequence_count": len(set(eval_tokens)),
        "overlap_count": 0,
        "disjoint": True,
        "calibration_token_split_sha256": _canonical_sha256(calibration_tokens),
        "evaluation_token_split_sha256": _canonical_sha256(eval_tokens),
        "tokenization_max_length": max_length,
        "token_paths_after_truncation": max_length is not None,
    }


def _qwen_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen3":
        raise QwenCodebookRunnerError("Parent model must have model_type='qwen3'.")
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)) or not layers:
        raise QwenCodebookRunnerError("Parent must expose non-empty model.layers.")
    expected = _positive_int(
        getattr(config, "num_hidden_layers", None), "config.num_hidden_layers"
    )
    if len(layers) != expected:
        raise QwenCodebookRunnerError("model.layers does not match configured depth.")
    return tuple(layers)


def build_qwen3_bundles(model: nn.Module) -> QwenBundleSet:
    """Build non-copying coupled projection bundles from one dense Qwen3 parent."""

    layers = _qwen_layers(model)
    config = model.config
    query_heads = _positive_int(
        getattr(config, "num_attention_heads", None), "config.num_attention_heads"
    )
    key_value_heads = _positive_int(
        getattr(config, "num_key_value_heads", None), "config.num_key_value_heads"
    )
    head_dim = _positive_int(getattr(config, "head_dim", None), "config.head_dim")
    neurons: list[GatedMLPBundle] = []
    attention: list[GQAAttentionBundle] = []
    for layer_index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        self_attention = getattr(layer, "self_attn", None)
        if not isinstance(mlp, nn.Module) or not isinstance(self_attention, nn.Module):
            raise QwenCodebookRunnerError(
                f"Layer {layer_index} lacks Qwen3 MLP or attention modules."
            )
        try:
            neurons.append(
                GatedMLPBundle.from_linears(
                    mlp.gate_proj,
                    mlp.up_proj,
                    mlp.down_proj,
                )
            )
            attention.append(
                GQAAttentionBundle.from_linears(
                    self_attention.q_proj,
                    self_attention.k_proj,
                    self_attention.v_proj,
                    self_attention.o_proj,
                    query_heads=query_heads,
                    key_value_heads=key_value_heads,
                    head_dim=head_dim,
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise QwenCodebookRunnerError(
                f"Layer {layer_index} is not a split-QKV Qwen3 GatedMLP layout."
            ) from exc
    intermediate = int(neurons[0].gate_weight.shape[0])
    return QwenBundleSet(
        neurons=tuple(neurons),
        attention=tuple(attention),
        logical_layers=len(layers),
        intermediate_size=intermediate,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
    )


def _assert_finite_tensor(
    value: Tensor, name: str, chunk_size: int = 1_000_000
) -> None:
    if not (value.is_floating_point() or value.is_complex()):
        return
    flattened = value.detach().reshape(-1)
    for start in range(0, int(flattened.numel()), chunk_size):
        block = flattened[start : start + chunk_size]
        if not bool(torch.isfinite(block).all().item()):
            raise QwenCodebookRunnerError(f"{name} contains NaN or infinity.")


def _assert_model_finite(model: nn.Module, stage: str) -> None:
    seen: set[int] = set()
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            if id(value) in seen:
                continue
            seen.add(id(value))
            _assert_finite_tensor(value, f"{stage} {kind} {name}")


def _assert_zero_gradients(model: nn.Module, stage: str) -> None:
    populated = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if populated:
        raise QwenCodebookRunnerError(
            f"{stage} found populated gradients: {populated[:5]}."
        )


@dataclass(frozen=True, slots=True)
class _EncodedPrompt:
    model_inputs: dict[str, Tensor]
    valid_positions: Tensor

    @property
    def token_count(self) -> int:
        return int(self.valid_positions.numel())


def _encode_calibration_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    max_length: int,
) -> tuple[_EncodedPrompt, ...]:
    length_limit = _positive_int(max_length, "calibration_max_length")
    encoded_prompts: list[_EncodedPrompt] = []
    for prompt_index, prompt in enumerate(prompts):
        try:
            encoded = tokenizer(
                str(prompt),
                return_tensors="pt",
                truncation=True,
                max_length=length_limit,
                add_special_tokens=True,
            )
        except Exception as exc:
            raise QwenCodebookRunnerError(
                f"Tokenizer failed on calibration prompt {prompt_index}."
            ) from exc
        if not isinstance(encoded, Mapping) or not isinstance(
            encoded.get("input_ids"), Tensor
        ):
            raise QwenCodebookRunnerError("Tokenizer must return tensor input_ids.")
        input_ids = encoded["input_ids"].detach().cpu()
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise QwenCodebookRunnerError(
                "Streaming calibration requires one rank-2 tokenized prompt."
            )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if (
            not isinstance(attention_mask, Tensor)
            or attention_mask.shape != input_ids.shape
        ):
            raise QwenCodebookRunnerError("Calibration attention mask shape mismatch.")
        attention_mask = attention_mask.detach().cpu()
        valid_positions = torch.nonzero(
            attention_mask.reshape(-1).to(torch.bool), as_tuple=False
        ).flatten()
        if int(valid_positions.numel()) == 0:
            raise QwenCodebookRunnerError(
                f"Calibration prompt {prompt_index} has no unmasked tokens."
            )
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        for name in ("position_ids",):
            value = encoded.get(name)
            if isinstance(value, Tensor):
                model_inputs[name] = value.detach().cpu()
        encoded_prompts.append(
            _EncodedPrompt(
                model_inputs=model_inputs,
                valid_positions=valid_positions.to(torch.long),
            )
        )
    return tuple(encoded_prompts)


def _activation_row_plan(
    prompts: Sequence[_EncodedPrompt], row_count: int
) -> tuple[tuple[Tensor, ...], list[int]]:
    total_rows = sum(prompt.token_count for prompt in prompts)
    requested = _positive_int(row_count, "activation_rows")
    if total_rows < requested:
        raise QwenCodebookRunnerError(
            f"Insufficient activation rows: requested {requested}, observed {total_rows}."
        )
    selected_global = _even_indices(total_rows, requested)
    per_prompt: list[Tensor] = []
    offset = 0
    cursor = 0
    for prompt in prompts:
        end = offset + prompt.token_count
        ordinals: list[int] = []
        while cursor < len(selected_global) and selected_global[cursor] < end:
            ordinals.append(selected_global[cursor] - offset)
            cursor += 1
        if ordinals:
            ordinal_tensor = torch.tensor(ordinals, dtype=torch.long)
            per_prompt.append(prompt.valid_positions.index_select(0, ordinal_tensor))
        else:
            per_prompt.append(torch.empty(0, dtype=torch.long))
        offset = end
    if cursor != requested:
        raise QwenCodebookRunnerError("Activation row planning did not cover all rows.")
    return tuple(per_prompt), selected_global


def collect_bounded_qwen_activations(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int,
    activation_rows: int,
    device: str,
    max_host_activation_bytes: int | None = None,
) -> ActivationCache:
    """Stream prompts and retain only bounded post-SwiGLU/GQA token rows."""

    layers = _qwen_layers(model)
    host_activation_cap = (
        _positive_int(max_host_activation_bytes, "max_host_activation_bytes")
        if max_host_activation_bytes is not None
        else None
    )
    encoded = _encode_calibration_prompts(tokenizer, prompts, max_length)
    local_positions, selected_global = _activation_row_plan(encoded, activation_rows)
    neuron_width = int(layers[0].mlp.down_proj.in_features)
    attention_width = int(layers[0].self_attn.o_proj.in_features)
    if min(neuron_width, attention_width) <= 0:
        raise QwenCodebookRunnerError("Hooked activation widths must be positive.")

    hooked_modules: list[nn.Module] = []
    for layer_index, layer in enumerate(layers):
        down = getattr(getattr(layer, "mlp", None), "down_proj", None)
        output = getattr(getattr(layer, "self_attn", None), "o_proj", None)
        if not isinstance(down, nn.Module) or not isinstance(output, nn.Module):
            raise QwenCodebookRunnerError(
                f"Layer {layer_index} lacks hookable down_proj or o_proj."
            )
        if int(getattr(down, "in_features", -1)) != neuron_width:
            raise QwenCodebookRunnerError(
                "Post-SwiGLU hook width mismatch across depth."
            )
        if int(getattr(output, "in_features", -1)) != attention_width:
            raise QwenCodebookRunnerError(
                "GQA output hook width mismatch across depth."
            )
        hooked_modules.extend((down, output))
    if len({id(module) for module in hooked_modules}) != len(hooked_modules):
        raise QwenCodebookRunnerError(
            "Dense calibration hook modules overlap across depth."
        )

    caches: dict[str, list[list[Tensor]]] = {
        "neuron": [[] for _ in layers],
        "attention": [[] for _ in layers],
    }
    calls = {
        "neuron": [0 for _ in layers],
        "attention": [0 for _ in layers],
    }
    widths = {"neuron": neuron_width, "attention": attention_width}
    active_prompt = -1
    handles: list[Any] = []
    cached_piece_bytes = 0
    retained_result_bytes = 0
    peak_host_activation_bytes = 0
    backbone = getattr(model, "model", None)
    if not isinstance(backbone, nn.Module):
        raise QwenCodebookRunnerError("Parent has no callable Qwen3 backbone.")

    def capture(kind: str, layer_index: int):
        def hook(_module: nn.Module, arguments: tuple[Any, ...]) -> None:
            nonlocal active_prompt, cached_piece_bytes, peak_host_activation_bytes
            if not 0 <= active_prompt < len(encoded):
                raise QwenCodebookRunnerError(
                    "Activation hook fired outside calibration."
                )
            calls[kind][layer_index] += 1
            if calls[kind][layer_index] != active_prompt + 1:
                raise QwenCodebookRunnerError(
                    f"{kind} hook mismatch at layer {layer_index}, "
                    f"prompt {active_prompt}."
                )
            if len(arguments) != 1 or not isinstance(arguments[0], Tensor):
                raise QwenCodebookRunnerError(
                    f"{kind} hook at layer {layer_index} received invalid inputs."
                )
            value = arguments[0]
            width = widths[kind]
            if value.ndim < 2 or int(value.shape[-1]) != width:
                raise QwenCodebookRunnerError(
                    f"{kind} hook width mismatch at layer {layer_index}."
                )
            flattened = value.detach().reshape(-1, width)
            expected_positions = int(
                encoded[active_prompt].model_inputs["input_ids"].numel()
            )
            if int(flattened.shape[0]) != expected_positions:
                raise QwenCodebookRunnerError(
                    f"{kind} hook token-row mismatch at layer {layer_index}."
                )
            _assert_finite_tensor(
                flattened,
                f"{kind} activation layer {layer_index} prompt {active_prompt}",
            )
            positions = local_positions[active_prompt]
            if int(positions.numel()):
                selected = flattened.index_select(0, positions.to(flattened.device))
                retained = selected.to(device="cpu", dtype=torch.float32).contiguous()
                caches[kind][layer_index].append(retained)
                cached_piece_bytes += int(retained.numel()) * retained.element_size()
                peak_host_activation_bytes = max(
                    peak_host_activation_bytes,
                    retained_result_bytes + cached_piece_bytes,
                )

        return hook

    was_training = model.training
    model.eval()
    try:
        for layer_index, layer in enumerate(layers):
            handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    capture("neuron", layer_index)
                )
            )
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    capture("attention", layer_index)
                )
            )
        with torch.inference_mode():
            for prompt_index, prompt in enumerate(encoded):
                active_prompt = prompt_index
                inputs = {
                    name: value.to(device)
                    for name, value in prompt.model_inputs.items()
                }
                backbone(
                    **inputs,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                for kind in calls:
                    mismatched = [
                        layer_index
                        for layer_index, count in enumerate(calls[kind])
                        if count != prompt_index + 1
                    ]
                    if mismatched:
                        raise QwenCodebookRunnerError(
                            f"{kind} hook mismatch after prompt {prompt_index}: "
                            f"layers {mismatched[:8]}."
                        )
                active_prompt = -1
    finally:
        active_prompt = -1
        for handle in handles:
            handle.remove()
        model.train(was_training)

    requested_rows = _positive_int(activation_rows, "activation_rows")

    def finalize(kind: str, width: int) -> tuple[Tensor, ...]:
        nonlocal cached_piece_bytes, retained_result_bytes, peak_host_activation_bytes
        result: list[Tensor] = []
        for layer_index, pieces in enumerate(caches[kind]):
            if not pieces:
                raise QwenCodebookRunnerError(
                    f"Insufficient {kind} activation rows at layer {layer_index}."
                )
            rows = torch.cat(pieces, dim=0)
            row_bytes = int(rows.numel()) * rows.element_size()
            peak_host_activation_bytes = max(
                peak_host_activation_bytes,
                retained_result_bytes + cached_piece_bytes + row_bytes,
            )
            if tuple(rows.shape) != (requested_rows, width):
                raise QwenCodebookRunnerError(
                    f"{kind} activation cache shape mismatch at layer {layer_index}."
                )
            _assert_finite_tensor(rows, f"cached {kind} layer {layer_index}")
            result.append(rows)
            piece_bytes = sum(int(piece.numel()) * piece.element_size() for piece in pieces)
            caches[kind][layer_index] = []
            pieces.clear()
            cached_piece_bytes -= piece_bytes
            retained_result_bytes += row_bytes
        return tuple(result)

    neuron_rows = finalize("neuron", neuron_width)
    attention_rows = finalize("attention", attention_width)
    actual_elements = sum(
        int(value.numel()) for value in (*neuron_rows, *attention_rows)
    )
    element_limit = requested_rows * len(layers) * (neuron_width + attention_width)
    if actual_elements != element_limit:
        raise QwenCodebookRunnerError("Bounded activation element accounting failed.")
    retained_host_activation_bytes = actual_elements * torch.tensor(
        [], dtype=torch.float32
    ).element_size()
    if retained_result_bytes != retained_host_activation_bytes or cached_piece_bytes != 0:
        raise QwenCodebookRunnerError("Activation host-memory accounting failed.")
    if host_activation_cap is not None and peak_host_activation_bytes > host_activation_cap:
        raise QwenCodebookRunnerError(
            "Peak host activation bytes exceed the configured hard cap: "
            f"{peak_host_activation_bytes}>{host_activation_cap}."
        )
    report = {
        "schema": ACTIVATION_SCHEMA,
        "method": "streaming-forward-hooks-global-even-token-rows",
        "logical_layers": len(layers),
        "prompt_count": len(prompts),
        "max_length": int(max_length),
        "streaming_batch_size": 1,
        "observed_token_rows": sum(prompt.token_count for prompt in encoded),
        "requested_rows_per_layer": requested_rows,
        "selected_rows_per_layer": requested_rows,
        "selected_global_token_rows": selected_global,
        "selected_global_token_rows_sha256": _canonical_sha256(selected_global),
        "prompt_token_counts": [prompt.token_count for prompt in encoded],
        "selected_rows_by_prompt": [int(value.numel()) for value in local_positions],
        "neuron_width": neuron_width,
        "attention_width": attention_width,
        "cache_dtype": "float32",
        "cache_device": "cpu",
        "cached_elements": actual_elements,
        "cached_element_limit": element_limit,
        "cached_bytes": retained_host_activation_bytes,
        "retained_host_activation_bytes": retained_host_activation_bytes,
        "peak_host_activation_bytes": peak_host_activation_bytes,
        "max_host_activation_bytes": host_activation_cap,
        "host_activation_cap_enforced": host_activation_cap is not None,
        "post_swiglu_down_projection_inputs": True,
        "coupled_gqa_output_projection_inputs": True,
        "hook_calls_per_neuron_layer": calls["neuron"],
        "hook_calls_per_attention_layer": calls["attention"],
        "materialized_full_depth_activation_tensor": False,
        "materialized_full_prompt_activation_cache": False,
        "bounded": actual_elements <= element_limit,
        "deterministic_row_selection": True,
        "finite": True,
        "fitting_requires_grad": False,
    }
    _assert_json_finite(report)
    return ActivationCache(neuron_rows, attention_rows, report)


collect_post_swiglu_activations = collect_bounded_qwen_activations


def _validate_alignment_result(
    result: PermutationAlignmentResult,
    *,
    layers: int,
    units: int,
    name: str,
) -> None:
    expected_shape = (layers, units)
    if tuple(result.permutations.shape) != expected_shape:
        raise QwenCodebookRunnerError(f"{name} permutation shape mismatch.")
    if tuple(result.inverse_permutations.shape) != expected_shape:
        raise QwenCodebookRunnerError(f"{name} inverse shape mismatch.")
    if not verify_permutation_bijections(result.permutations):
        raise QwenCodebookRunnerError(f"{name} permutations are not bijective.")
    if not torch.equal(
        invert_permutations(result.permutations), result.inverse_permutations
    ):
        raise QwenCodebookRunnerError(f"{name} permutation inverse mismatch.")
    diagnostics = result.diagnostics
    bounded_assignment_shape = (
        diagnostics.no_full_pairwise_matrix or diagnostics.exact_small_group_assignment
    )
    required = (
        diagnostics.deterministic,
        diagnostics.zero_gradient,
        diagnostics.permutation_proof,
        diagnostics.bounded_work,
        bounded_assignment_shape,
        diagnostics.objective_better_than_or_equal_to_identity,
    )
    if not all(required):
        raise QwenCodebookRunnerError(f"{name} alignment invariants failed.")
    _assert_json_finite(diagnostics.as_dict(), f"{name}_alignment")


def infer_codebook_permutations(
    bundles: QwenBundleSet,
    *,
    neuron_activations: Sequence[Tensor] | None,
    attention_activations: Sequence[Tensor] | None,
    reference_layer: int = 0,
    neuron_weight_sketch_size: int = DEFAULT_WEIGHT_SKETCH_SIZE,
    neuron_activation_sketch_size: int = DEFAULT_ACTIVATION_SKETCH_SIZE,
    attention_weight_sketch_size: int = DEFAULT_WEIGHT_SKETCH_SIZE,
    attention_activation_sketch_size: int = DEFAULT_ACTIVATION_SKETCH_SIZE,
    coarse_signature_dimensions: int = DEFAULT_SIGNATURE_DIMENSIONS,
    coarse_bits_per_dimension: int = DEFAULT_SIGNATURE_BITS,
    local_block_size: int = DEFAULT_LOCAL_BLOCK_SIZE,
    refinement_passes: int = DEFAULT_REFINEMENT_PASSES,
    max_exact_gqa_groups: int = DEFAULT_MAX_EXACT_GQA_GROUPS,
) -> AlignmentResults:
    """Infer canonical-to-physical coupled permutations for both codebooks."""

    neuron_result = infer_swiglu_neuron_permutations(
        bundles.neurons,
        activation_sketches=neuron_activations,
        reference_layer=reference_layer,
        weight_sketch_size=neuron_weight_sketch_size,
        activation_sketch_size=neuron_activation_sketch_size,
        coarse_signature_dimensions=coarse_signature_dimensions,
        coarse_bits_per_dimension=coarse_bits_per_dimension,
        local_block_size=local_block_size,
        refinement_passes=refinement_passes,
    )
    attention_result = infer_qwen_gqa_group_permutations(
        bundles.attention,
        query_heads=bundles.query_heads,
        key_value_heads=bundles.key_value_heads,
        head_dim=bundles.head_dim,
        activation_sketches=attention_activations,
        reference_layer=reference_layer,
        weight_sketch_size=attention_weight_sketch_size,
        activation_sketch_size=attention_activation_sketch_size,
        max_exact_groups=max_exact_gqa_groups,
        coarse_signature_dimensions=coarse_signature_dimensions,
        coarse_bits_per_dimension=coarse_bits_per_dimension,
        local_block_size=local_block_size,
        refinement_passes=refinement_passes,
    )
    _validate_alignment_result(
        neuron_result,
        layers=bundles.logical_layers,
        units=bundles.intermediate_size,
        name="neuron",
    )
    _validate_alignment_result(
        attention_result,
        layers=bundles.logical_layers,
        units=bundles.key_value_heads,
        name="attention",
    )
    return AlignmentResults(neuron_result, attention_result)


def _alignment_result_report(result: PermutationAlignmentResult) -> dict[str, Any]:
    permutations = result.permutations.to(torch.int64).cpu().tolist()
    inverses = result.inverse_permutations.to(torch.int64).cpu().tolist()
    return {
        "canonical_to_physical": permutations,
        "physical_to_canonical": inverses,
        "canonical_to_physical_sha256": _canonical_sha256(permutations),
        "physical_to_canonical_sha256": _canonical_sha256(inverses),
        "diagnostics": result.diagnostics.as_dict(),
    }


def build_alignment_report(results: AlignmentResults) -> dict[str, Any]:
    neuron = _alignment_result_report(results.neurons)
    attention = _alignment_result_report(results.attention)
    report = {
        "schema": ALIGNMENT_SCHEMA,
        "neuron": neuron,
        "attention": attention,
        "invariants": {
            "neuron_permutation_proof": results.neurons.permutation_proof,
            "attention_permutation_proof": results.attention.permutation_proof,
            "neuron_bounded": results.neurons.diagnostics.bounded_work,
            "attention_bounded": results.attention.diagnostics.bounded_work,
            "zero_gradient": (
                results.neurons.diagnostics.zero_gradient
                and results.attention.diagnostics.zero_gradient
            ),
            "all_passed": True,
        },
    }
    _assert_json_finite(report)
    return report


def _dense_runtime_snapshot(
    model: nn.Module, bundles: QwenBundleSet
) -> DenseRuntimeSnapshot:
    parameters: dict[int, nn.Parameter] = {}
    for bundle in bundles.neurons:
        for value in (
            bundle.gate_weight,
            bundle.up_weight,
            bundle.down_weight,
            bundle.gate_bias,
            bundle.up_bias,
            bundle.down_bias,
        ):
            if isinstance(value, nn.Parameter):
                parameters[id(value)] = value
    for bundle in bundles.attention:
        for value in (
            bundle.q_weight,
            bundle.k_weight,
            bundle.v_weight,
            bundle.o_weight,
            bundle.q_bias,
            bundle.k_bias,
            bundle.v_bias,
            bundle.o_bias,
        ):
            if isinstance(value, nn.Parameter):
                parameters[id(value)] = value
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    parameters.update({id(parameter): parameter for parameter in input_embedding.parameters()})
    parameters.update({id(parameter): parameter for parameter in output_embedding.parameters()})
    modules: dict[int, nn.Module] = {
        id(input_embedding): input_embedding,
        id(output_embedding): output_embedding,
    }
    for layer in _qwen_layers(model):
        modules[id(layer.mlp)] = layer.mlp
        for role in _PROJECTION_ROLES:
            projection = getattr(layer.self_attn, role)
            if isinstance(projection, nn.Module):
                modules[id(projection)] = projection
    return DenseRuntimeSnapshot(
        frozenset(parameters),
        frozenset(modules),
        tuple(weakref.ref(parameter) for parameter in parameters.values()),
        tuple(weakref.ref(module) for module in modules.values()),
    )


def _capture_noop_reference(
    model: nn.Module,
    target_tokenizer: Any,
    prompt: str,
    parent_rows: Sequence[int],
    *,
    max_length: int,
    device: str,
) -> NoopReference:
    encoded = _encode_calibration_prompts(
        target_tokenizer, [prompt], min(_positive_int(max_length, "max_length"), 32)
    )[0]
    target_ids = encoded.model_inputs["input_ids"].to(torch.long)
    mapping = torch.tensor(parent_rows, dtype=torch.long)
    if int(target_ids.max()) >= int(mapping.numel()):
        raise QwenCodebookRunnerError("No-op target token ID exceeds row mapping.")
    source_ids = mapping.index_select(0, target_ids.reshape(-1)).reshape_as(target_ids)
    source_inputs = {
        "input_ids": source_ids.to(device),
        "attention_mask": encoded.model_inputs["attention_mask"].to(device),
    }
    with torch.inference_mode():
        output = model(
            **source_inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise QwenCodebookRunnerError("Parent no-op trace returned no logits.")
        source_vocab_size = int(logits.shape[-1])
        full_source_scope = (
            int(mapping.numel()) == source_vocab_size
            and torch.equal(mapping.cpu(), torch.arange(source_vocab_size))
        )
        expected = (
            logits
            if full_source_scope
            else logits.index_select(
                -1, mapping.to(device=logits.device, dtype=torch.long)
            )
        ).detach().cpu().clone()
        source_embedding = model.get_input_embeddings().weight
        expected_embedding = (
            source_embedding
            if full_source_scope
            else source_embedding.index_select(
                0, mapping.to(device=source_embedding.device, dtype=torch.long)
            )
        ).detach().cpu().clone()
    target_inputs = {
        name: value.detach().cpu().clone()
        for name, value in encoded.model_inputs.items()
        if name in {"input_ids", "attention_mask", "position_ids"}
    }
    _assert_finite_tensor(expected, "full-depth parent no-op logits")
    return NoopReference(
        target_inputs,
        expected,
        expected_embedding=expected_embedding,
        vocabulary_scope=(
            "full-source-vocabulary-including-padding"
            if full_source_scope
            else "mapped-target-vocabulary"
        ),
        source_vocab_size=source_vocab_size,
        checked_parent_rows=tuple(int(row) for row in parent_rows),
    )


def _optional_tensor_equal(left: Tensor | None, right: Tensor | None) -> bool:
    if left is None or right is None:
        return left is right
    return torch.equal(left, right)


def _verify_full_depth_projections(
    model: nn.Module,
    bundles: QwenBundleSet,
) -> dict[str, Any]:
    tensor_count = 0
    for layer_index in range(bundles.logical_layers):
        actual_attention = model.attention_codebook.registry.materialize_layer(
            layer_index
        )
        expected_attention = bundles.attention[layer_index]
        for name in (
            "q_weight",
            "k_weight",
            "v_weight",
            "o_weight",
            "q_bias",
            "k_bias",
            "v_bias",
            "o_bias",
        ):
            if not _optional_tensor_equal(
                getattr(actual_attention, name), getattr(expected_attention, name)
            ):
                raise QwenCodebookRunnerError(
                    f"Full-depth attention no-op mismatch at layer {layer_index}.{name}."
                )
            if getattr(expected_attention, name) is not None:
                tensor_count += 1
        actual_neuron = model.neuron_codebook.registry.materialize_layer(layer_index)
        expected_neuron = bundles.neurons[layer_index]
        for name in (
            "gate_weight",
            "up_weight",
            "down_weight",
            "gate_bias",
            "up_bias",
            "down_bias",
        ):
            if not _optional_tensor_equal(
                getattr(actual_neuron, name), getattr(expected_neuron, name)
            ):
                raise QwenCodebookRunnerError(
                    f"Full-depth neuron no-op mismatch at layer {layer_index}.{name}."
                )
            if getattr(expected_neuron, name) is not None:
                tensor_count += 1
    return {
        "projection_tensors_verified": tensor_count,
        "projection_tensors_exact": True,
    }


def _verify_noop_logits(
    model: nn.Module,
    reference: NoopReference,
    *,
    device: str,
) -> dict[str, Any]:
    inputs = {name: value.to(device) for name, value in reference.target_inputs.items()}
    with torch.inference_mode():
        output = model(
            **inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise QwenCodebookRunnerError("Runtime no-op trace returned no logits.")
        actual = logits.detach().cpu()
    _assert_finite_tensor(actual, "full-depth runtime no-op logits")
    if actual.shape != reference.expected_logits.shape:
        raise QwenCodebookRunnerError("Full-depth no-op logit shape mismatch.")
    difference = actual.to(torch.float64) - reference.expected_logits.to(torch.float64)
    maximum_error = float(difference.abs().max().item()) if difference.numel() else 0.0
    mean_error = float(difference.abs().mean().item()) if difference.numel() else 0.0
    expected_norm = float(
        torch.linalg.vector_norm(reference.expected_logits.to(torch.float64)).item()
    )
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    relative_l2_error = difference_norm / max(expected_norm, torch.finfo(torch.float64).tiny)
    exact = bool(torch.equal(actual, reference.expected_logits))
    if not actual.dtype.is_floating_point:
        raise QwenCodebookRunnerError("Full-depth no-op logits must be floating-point.")
    dtype_epsilon = float(torch.finfo(actual.dtype).eps)
    absolute_tolerance = 16.0 * dtype_epsilon
    relative_tolerance = 8.0 * dtype_epsilon
    numerically_close = bool(
        torch.allclose(
            actual,
            reference.expected_logits,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )
    relative_l2_tolerance = 8.0 * dtype_epsilon
    probabilities = torch.softmax(actual.to(torch.float64), dim=-1)
    expected_probabilities = torch.softmax(
        reference.expected_logits.to(torch.float64), dim=-1
    )
    maximum_probability_error = float(
        (probabilities - expected_probabilities).abs().max().item()
    )
    probability_tolerance = 32.0 * dtype_epsilon
    probability_close = maximum_probability_error <= probability_tolerance
    top_k = min(5, int(actual.shape[-1]))
    actual_top_k = torch.topk(actual, top_k, dim=-1).indices
    expected_top_k = torch.topk(reference.expected_logits, top_k, dim=-1).indices
    top_k_matches = actual_top_k == expected_top_k
    top_k_match_count = int(top_k_matches.sum().item())
    top_k_count = int(top_k_matches.numel())
    top_k_exact = bool(top_k_matches.all().item())
    actual_top1 = torch.argmax(actual, dim=-1)
    expected_top1 = torch.argmax(reference.expected_logits, dim=-1)
    top1_matches = actual_top1 == expected_top1
    top1_match_count = int(top1_matches.sum().item())
    top1_count = int(top1_matches.numel())
    top1_exact = bool(top1_matches.all().item())
    numerically_equivalent = (
        numerically_close
        and relative_l2_error <= relative_l2_tolerance
        and probability_close
        and top1_exact
        and top_k_exact
    )
    if not numerically_equivalent:
        raise QwenCodebookRunnerError(
            "Full-depth no-op logits are not numerically equivalent; "
            f"max_abs_error={maximum_error}, relative_l2_error={relative_l2_error}, "
            f"top1_matches={top1_match_count}/{top1_count}, "
            f"top_k_matches={top_k_match_count}/{top_k_count}, "
            f"max_probability_error={maximum_probability_error}."
        )
    embedding_exact = True
    embedding_elements_verified = 0
    if reference.expected_embedding is not None:
        actual_embedding = model.get_input_embeddings().weight.detach().cpu()
        embedding_exact = bool(torch.equal(actual_embedding, reference.expected_embedding))
        embedding_elements_verified = int(actual_embedding.numel())
        if not embedding_exact:
            raise QwenCodebookRunnerError("Full-depth no-op embedding rows changed.")
    return {
        "logit_elements_verified": int(actual.numel()),
        "logits_exact": exact,
        "logits_bitwise_exact": exact,
        "logits_numerically_equivalent": numerically_equivalent,
        "top1_exact": top1_exact,
        "top1_match_count": top1_match_count,
        "top1_count": top1_count,
        "maximum_absolute_logit_error": maximum_error,
        "mean_absolute_logit_error": mean_error,
        "relative_l2_logit_error": relative_l2_error,
        "relative_l2_tolerance": relative_l2_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "dtype_epsilon": dtype_epsilon,
        "maximum_probability_error": maximum_probability_error,
        "probability_tolerance": probability_tolerance,
        "probabilities_numerically_equivalent": probability_close,
        "top_k": top_k,
        "top_k_exact": top_k_exact,
        "top_k_match_count": top_k_match_count,
        "top_k_count": top_k_count,
        "vocabulary_scope": reference.vocabulary_scope,
        "source_vocab_size": reference.source_vocab_size,
        "checked_parent_row_count": len(reference.checked_parent_rows),
        "embedding_rows_exact": embedding_exact,
        "embedding_elements_verified": embedding_elements_verified,
    }


def verify_compact_checkpoint_equivalence(
    in_memory: nn.Module,
    reloaded: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_length: int,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    """Prove a standalone compact child is equivalent to its live runtime."""

    encoded = _encode_calibration_prompts(
        tokenizer, [prompt], min(_positive_int(max_length, "max_length"), 32)
    )[0]
    inputs = {name: value.to(device) for name, value in encoded.model_inputs.items()}
    in_memory.eval()
    reloaded.eval()
    with torch.inference_mode():
        expected = in_memory(
            **inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).logits
        actual = reloaded(
            **inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).logits
    _assert_finite_tensor(expected, "in-memory checkpoint equivalence logits")
    _assert_finite_tensor(actual, "reloaded checkpoint equivalence logits")
    logits_exact = bool(torch.equal(expected, actual))
    if not logits_exact:
        raise QwenCodebookRunnerError("Reloaded compact child logits differ from memory.")
    generation_tokens = min(_positive_int(max_new_tokens, "max_new_tokens"), 8)
    prompt_ids = inputs["input_ids"][:, : min(2, int(inputs["input_ids"].shape[1]))]
    with torch.inference_mode():
        expected_generation = in_memory.generate(
            prompt_ids,
            do_sample=False,
            min_new_tokens=generation_tokens,
            max_new_tokens=generation_tokens,
        )
        actual_generation = reloaded.generate(
            prompt_ids,
            do_sample=False,
            min_new_tokens=generation_tokens,
            max_new_tokens=generation_tokens,
        )
    generation_exact = bool(torch.equal(expected_generation, actual_generation))
    if not generation_exact:
        raise QwenCodebookRunnerError(
            "Reloaded compact child generation differs from memory."
        )
    in_memory_report = qwen3_codebook_parameter_report(in_memory)
    reloaded_report = qwen3_codebook_parameter_report(reloaded)
    accounting_exact = in_memory_report == reloaded_report
    if not accounting_exact:
        raise QwenCodebookRunnerError(
            "Reloaded compact child parameter/index accounting differs from memory."
        )
    return {
        "logits_exact": logits_exact,
        "logit_elements_verified": int(expected.numel()),
        "generation_exact": generation_exact,
        "generation_new_tokens": generation_tokens,
        "parameter_index_accounting_exact": accounting_exact,
        "reloaded_parent_runtime_references": reloaded_report.parent_runtime_reference_count,
    }


def verify_runtime_ownership(
    model: nn.Module,
    snapshot: DenseRuntimeSnapshot,
    parent_rows: Sequence[int],
    *,
    require_process_release: bool = False,
) -> dict[str, Any]:
    """Prove the converted runtime retains no replaced dense parent objects."""

    runtime_parameter_ids = {id(parameter) for parameter in model.parameters()}
    retained_parameter_ids = runtime_parameter_ids.intersection(snapshot.parameter_ids)
    runtime_module_ids = {id(module) for module in model.modules()}
    retained_module_ids = runtime_module_ids.intersection(snapshot.module_ids)
    live_replaced_parameters = sum(
        reference() is not None for reference in snapshot.parameter_refs
    )
    live_replaced_modules = sum(
        reference() is not None for reference in snapshot.module_refs
    )
    state_names = set(model.state_dict())
    dense_state_names = sorted(
        name
        for name in state_names
        if any(
            suffix in name
            for suffix in (
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".self_attn.o_proj.weight",
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
                ".mlp.down_proj.weight",
            )
        )
    )
    parent_attributes = [
        name
        for name in ("parent_model", "source_model", "dense_parent", "donor_model")
        if isinstance(getattr(model, name, None), nn.Module)
    ]
    report = qwen3_codebook_parameter_report(model)
    actual_rows = getattr(model, "parent_token_rows", None)
    expected_rows = torch.tensor(tuple(parent_rows), dtype=torch.int32)
    row_mapping_exact = isinstance(actual_rows, Tensor) and torch.equal(
        actual_rows.detach().cpu(), expected_rows
    )
    attention_owners = [
        name
        for name, module in model.named_modules(remove_duplicate=False)
        if module is model.attention_codebook.registry
    ]
    neuron_owners = [
        name
        for name, module in model.named_modules(remove_duplicate=False)
        if module is model.neuron_codebook.registry
    ]
    checks = {
        "replaced_dense_parameter_references": len(retained_parameter_ids),
        "replaced_dense_module_references": len(retained_module_ids),
        "live_replaced_dense_parameter_weakrefs": live_replaced_parameters,
        "live_replaced_dense_module_weakrefs": live_replaced_modules,
        "process_release_required": require_process_release,
        "dense_projection_state_entries": dense_state_names,
        "parent_module_attributes": parent_attributes,
        "parent_runtime_reference_count": report.parent_runtime_reference_count,
        "parent_token_rows_exact": row_mapping_exact,
        "attention_registry_owners": attention_owners,
        "neuron_registry_owners": neuron_owners,
        "parameterless_layer_views": report.parameterless_layer_views,
        "single_registry_owner_per_codebook": report.single_registry_owner_per_codebook,
    }
    if (
        retained_parameter_ids
        or retained_module_ids
        or dense_state_names
        or parent_attributes
        or report.parent_runtime_reference_count != 0
        or not row_mapping_exact
        or attention_owners != ["attention_codebook.registry"]
        or neuron_owners != ["neuron_codebook.registry"]
        or not report.parameterless_layer_views
        or not report.single_registry_owner_per_codebook
        or (
            require_process_release
            and (live_replaced_parameters or live_replaced_modules)
        )
    ):
        raise QwenCodebookRunnerError(
            "Dense parent runtime ownership invariant failed."
        )
    checks["all_passed"] = True
    return checks


def build_parameter_index_accounting(
    model: nn.Module,
    *,
    p_neuron: int,
    p_attention: int,
) -> dict[str, Any]:
    report = qwen3_codebook_parameter_report(model)
    values = report.as_dict()
    actual_unique = sum(int(parameter.numel()) for parameter in model.parameters())
    index_sum = (
        report.attention_index_bytes
        + report.neuron_index_bytes
        + report.token_row_index_bytes
    )
    invariants = {
        "runtime_schema": report.schema == RUNTIME_SCHEMA,
        "unique_count_exact": actual_unique == report.unique_parameter_count,
        "logical_not_smaller_than_unique": (
            report.logical_parameter_count >= report.unique_parameter_count
        ),
        "index_sum_exact": index_sum == report.index_buffer_bytes,
        "requested_neuron_depth_exact": (
            report.neuron_prototypes_per_neuron == p_neuron
        ),
        "requested_attention_depth_exact": (
            report.attention_prototypes_per_group == p_attention
        ),
        "neuron_prototype_count_exact": (
            report.neuron_prototype_count == report.intermediate_size * p_neuron
        ),
        "full_logical_depth": report.full_logical_depth,
        "exact_parent_token_rows": report.exact_parent_token_rows,
        "tied_input_output_embeddings": report.tied_input_output_embeddings,
        "parameterless_layer_views": report.parameterless_layer_views,
        "single_registry_owner_per_codebook": (
            report.single_registry_owner_per_codebook
        ),
        "zero_gradient_fit": not report.fitting_requires_grad,
        "zero_parent_runtime_references": report.parent_runtime_reference_count == 0,
    }
    invariants["all_passed"] = all(invariants.values())
    if not invariants["all_passed"]:
        failed = [name for name, passed in invariants.items() if not passed]
        raise QwenCodebookRunnerError(
            f"Parameter/index accounting invariants failed: {failed}."
        )
    accounting = {
        "schema": ACCOUNTING_SCHEMA,
        "parameters": {
            "source_unique": report.source_unique_parameter_count,
            "source_logical": report.source_logical_parameter_count,
            "runtime_unique": report.unique_parameter_count,
            "runtime_logical": report.logical_parameter_count,
            "runtime_registered_logical": report.registered_logical_parameter_count,
            "attention_unique": report.attention_unique_parameter_count,
            "attention_logical": report.attention_logical_parameter_count,
            "neuron_unique": report.neuron_unique_parameter_count,
            "neuron_logical": report.neuron_logical_parameter_count,
            "retained_fixed": report.retained_fixed_parameter_count,
            "vocabulary_unique": report.vocabulary_unique_parameter_count,
            "vocabulary_logical": report.vocabulary_logical_parameter_count,
            "saved": report.parameters_saved,
        },
        "indices": {
            "attention_bytes": report.attention_index_bytes,
            "neuron_bytes": report.neuron_index_bytes,
            "token_row_bytes": report.token_row_index_bytes,
            "total_bytes": report.index_buffer_bytes,
        },
        "ratios": {
            "logical_to_unique": report.logical_to_unique_ratio,
            "source_to_runtime": report.source_to_runtime_ratio,
        },
        "prototype_depth": {
            "logical_layers": report.logical_layers,
            "p_neuron": p_neuron,
            "p_attention": p_attention,
        },
        "runtime_report": values,
        "invariants": invariants,
    }
    _assert_json_finite(accounting)
    return accounting


def _alignment_summary(report: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        "report": str(path),
        "neuron": {
            "canonical_to_physical_sha256": report["neuron"][
                "canonical_to_physical_sha256"
            ],
            "diagnostics": report["neuron"]["diagnostics"],
        },
        "attention": {
            "canonical_to_physical_sha256": report["attention"][
                "canonical_to_physical_sha256"
            ],
            "diagnostics": report["attention"]["diagnostics"],
        },
        "invariants": report["invariants"],
    }


def build_run_report(
    *,
    source_model: str,
    source_tokenizer: str,
    target_tokenizer: str,
    p_neuron: int,
    p_attention: int,
    calibration: dict[str, Any],
    alignment: dict[str, Any],
    alignment_path: Path,
    compile_report: dict[str, Any],
    accounting: dict[str, Any],
    verification: dict[str, Any],
    candidate_eval: dict[str, Any],
    comparison: dict[str, Any],
    comparison_path: Path,
    manual_sample_count: int,
    execution_configuration: dict[str, Any] | None = None,
    certification_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the stable top-level report consumed by survival runs."""

    report = {
        "schema": RUN_SCHEMA,
        "source_model": source_model,
        "source_tokenizer": source_tokenizer,
        "target_tokenizer": target_tokenizer,
        "configuration": {
            "p_neuron": p_neuron,
            "p_attention": p_attention,
            "manual_sample_count": manual_sample_count,
        },
        "execution_configuration": execution_configuration or {},
        "certification_contract": certification_contract or {},
        "calibration": calibration,
        "alignment": _alignment_summary(alignment, alignment_path),
        "compile": compile_report,
        "accounting": accounting,
        "verification": verification,
        "candidate_eval": candidate_eval,
        "comparison": str(comparison_path),
        "aggregate": {
            "parent": comparison.get("parent"),
            "child": comparison.get("child"),
            "param_compression_ratio": comparison.get("param_compression_ratio"),
            "behavior": (comparison.get("behavior") or {}).get("overall"),
            "generation": (comparison.get("generation") or {}).get("overall"),
            "choice": (comparison.get("choice") or {}).get("overall"),
        },
        "manual_samples": comparison.get("manual_samples", {}),
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "zero_recovery": True,
        "standalone_checkpoint": False,
        "status": "complete",
    }
    validate_run_report(report)
    return report


def validate_run_report(report: Mapping[str, Any]) -> None:
    """Fail closed if a completed report does not carry every proof section."""

    required_sections = (
        "calibration",
        "alignment",
        "compile",
        "accounting",
        "verification",
        "candidate_eval",
        "aggregate",
        "manual_samples",
    )
    if report.get("schema") != RUN_SCHEMA or report.get("status") != "complete":
        raise QwenCodebookRunnerError("Run report schema or status invariant failed.")
    if any(section not in report for section in required_sections):
        raise QwenCodebookRunnerError("Run report is missing a required section.")
    calibration = report["calibration"]
    alignment = report["alignment"]
    accounting = report["accounting"]
    verification = report["verification"]
    if not isinstance(calibration, Mapping) or not calibration.get("bounded"):
        raise QwenCodebookRunnerError("Run report lacks bounded calibration proof.")
    if not isinstance(alignment, Mapping) or not (
        alignment.get("invariants") or {}
    ).get("all_passed"):
        raise QwenCodebookRunnerError("Run report lacks alignment proof.")
    if not isinstance(accounting, Mapping) or not (
        accounting.get("invariants") or {}
    ).get("all_passed"):
        raise QwenCodebookRunnerError("Run report lacks accounting proof.")
    if not isinstance(verification, Mapping) or not verification.get("all_passed"):
        raise QwenCodebookRunnerError("Run report lacks runtime verification proof.")
    manual_samples = report["manual_samples"]
    if not isinstance(manual_samples, Mapping) or not manual_samples.get(
        "all_available_strata_represented", False
    ):
        raise QwenCodebookRunnerError(
            "Run report lacks category-stratified manual-sample coverage."
        )
    strata = manual_samples.get("strata")
    if not isinstance(strata, Mapping) or not strata or any(
        not isinstance(value, Mapping) or int(value.get("selected", 0)) <= 0
        for value in strata.values()
    ):
        raise QwenCodebookRunnerError("Run report manual-sample strata are incomplete.")
    if manual_samples.get("selected_count") != report.get("configuration", {}).get(
        "manual_sample_count"
    ):
        raise QwenCodebookRunnerError("Run report manual-sample count is not exact.")
    selected_count = sum(
        len(manual_samples.get(name, []))
        for name in ("behavior", "generation")
        if isinstance(manual_samples.get(name, []), list)
    )
    if selected_count != manual_samples.get("selected_count"):
        raise QwenCodebookRunnerError(
            "Run report manual-sample payload does not match its count."
        )
    if not isinstance(report.get("certification_contract"), Mapping) or not report[
        "certification_contract"
    ]:
        raise QwenCodebookRunnerError("Run report lacks the locked certification contract.")
    if (
        report.get("zero_optimizer_steps") != 0
        or report.get("backward_called") is not False
    ):
        raise QwenCodebookRunnerError("Run report violates zero-optimization policy.")
    _assert_json_finite(report)


def _compare_evals(
    parent_eval: Path,
    candidate_eval: Path,
    *,
    manual_sample_count: int,
) -> dict[str, Any]:
    experiments_dir = str(Path(__file__).resolve().parent)
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    from compare_eval_retention import compare

    comparison = compare(parent_eval, candidate_eval, max_samples=manual_sample_count)
    parent = _load_eval_report(parent_eval, "cached parent evaluation")
    candidate = _load_eval_report(candidate_eval, "candidate evaluation")
    comparison["manual_samples"] = _stratified_manual_samples(
        parent,
        candidate,
        manual_sample_count,
    )
    return comparison


def _load_eval_report(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenCodebookRunnerError(f"Cannot load {name}: {path}.") from exc
    if not isinstance(value, dict):
        raise QwenCodebookRunnerError(f"{name} must be a JSON object.")
    return value


def _eval_item_map(items: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            continue
        result[str(item["id"])] = dict(item)
    return result


def _stratified_manual_samples(
    parent: Mapping[str, Any], candidate: Mapping[str, Any], requested: int
) -> dict[str, Any]:
    """Sample every available outcome/category stratum before filling breadth."""

    count = _positive_int(requested, "manual_sample_count")
    collections = (
        (
            "behavior",
            _eval_item_map((parent.get("behavior_suite") or {}).get("items")),
            _eval_item_map((candidate.get("behavior_suite") or {}).get("items")),
            "passed",
        ),
        (
            "generation",
            _eval_item_map(parent.get("generations")),
            _eval_item_map(candidate.get("generations")),
            "passed",
        ),
    )
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for collection, parent_items, candidate_items, pass_key in collections:
        parent_diagnostics = _eval_item_map(parent.get("generation_diagnostics"))
        candidate_diagnostics = _eval_item_map(candidate.get("generation_diagnostics"))
        if collection == "generation":
            for item_id, item in parent_items.items():
                item.update(parent_diagnostics.get(item_id, {}))
            for item_id, item in candidate_items.items():
                item.update(candidate_diagnostics.get(item_id, {}))
        for item_id in sorted(set(parent_items) | set(candidate_items)):
            parent_item = parent_items.get(item_id, {})
            candidate_item = candidate_items.get(item_id, {})
            parent_passed = bool(parent_item.get(pass_key, False))
            candidate_passed = bool(candidate_item.get(pass_key, False))
            outcome = (
                "retained_pass"
                if parent_passed and candidate_passed
                else "loss"
                if parent_passed
                else "gain"
                if candidate_passed
                else "shared_failure"
            )
            category = str(candidate_item.get("category", parent_item.get("category", "uncategorized")))
            item = {
                "id": item_id,
                "category": category,
                "kind": candidate_item.get("kind", parent_item.get("kind")),
                "prompt": candidate_item.get("prompt", parent_item.get("prompt")),
                "parent_passed": parent_passed,
                "child_passed": candidate_passed,
                "outcome": outcome,
            }
            strata.setdefault((collection, category, outcome), []).append(item)
    nonempty = {key: values for key, values in strata.items() if values}
    if not nonempty:
        raise QwenCodebookRunnerError("Manual sampling has no behavior or generation items.")
    if count < len(nonempty):
        raise QwenCodebookRunnerError(
            "manual_sample_count cannot cover every available category/outcome stratum: "
            f"requested={count}, required={len(nonempty)}."
        )
    available = sum(len(values) for values in nonempty.values())
    if count > available:
        raise QwenCodebookRunnerError(
            f"manual_sample_count={count} exceeds {available} eligible stratified items."
        )
    selected: dict[str, list[dict[str, Any]]] = {"behavior": [], "generation": []}
    remaining: dict[tuple[str, str, str], list[dict[str, Any]]] = {
        key: list(values) for key, values in sorted(nonempty.items())
    }
    for key, values in remaining.items():
        selected[key[0]].append(values.pop(0))
    while sum(len(values) for values in selected.values()) < count:
        progressed = False
        for key in sorted(remaining):
            if remaining[key] and sum(len(values) for values in selected.values()) < count:
                selected[key[0]].append(remaining[key].pop(0))
                progressed = True
        if not progressed:
            break
    stratum_report = {
        f"{collection}:{category}:{outcome}": {
            "available": len(nonempty[(collection, category, outcome)]),
            "selected": sum(
                1
                for item in selected[collection]
                if item["category"] == category and item["outcome"] == outcome
            ),
        }
        for collection, category, outcome in sorted(nonempty)
    }
    return {
        **selected,
        "sampling": "deterministic-category-outcome-round-robin",
        "requested_count": count,
        "available_count": available,
        "selected_count": sum(len(values) for values in selected.values()),
        "strata": stratum_report,
        "all_available_strata_represented": all(
            values["selected"] > 0 for values in stratum_report.values()
        ),
    }


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "calibration_count",
        "calibration_max_length",
        "activation_rows",
        "neuron_weight_sketch_size",
        "attention_weight_sketch_size",
        "coarse_signature_dimensions",
        "coarse_bits_per_dimension",
        "local_block_size",
        "max_exact_gqa_groups",
        "max_new_tokens",
        "max_text_evals",
        "max_choice_probes",
        "max_generation_prompts",
        "max_behavior_probes",
        "manual_sample_count",
        "max_host_activation_bytes",
        "max_cuda_allocated_bytes",
        "max_cuda_reserved_bytes",
    ):
        _positive_int(getattr(args, name), name)
    for name in (
        "neuron_activation_sketch_size",
        "attention_activation_sketch_size",
        "refinement_passes",
    ):
        _nonnegative_int(getattr(args, name), name)
    if args.neuron_weight_sketch_size == 0 and args.neuron_activation_sketch_size == 0:
        raise QwenCodebookRunnerError("Neuron alignment requires a non-empty sketch.")
    if (
        args.attention_weight_sketch_size == 0
        and args.attention_activation_sketch_size == 0
    ):
        raise QwenCodebookRunnerError(
            "Attention alignment requires a non-empty sketch."
        )


def _limited(values: Sequence[Any], count: int) -> list[Any]:
    return list(values[: _positive_int(count, "evaluation limit")])


def _suite_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Load and lock the candidate/parent evaluation contract before loading a model."""

    from neural_forge.hf_tools import _load_eval_suite

    try:
        suite = _load_eval_suite(args.eval_suite)
    except Exception as exc:
        raise QwenCodebookRunnerError("Cannot load the evaluation suite contract.") from exc
    selected = {
        "text_evals": _limited(suite["text_evals"], args.max_text_evals),
        "choice_probes": _limited(suite["choice_probes"], args.max_choice_probes),
        "generation_prompts": _limited(
            suite["generation_prompts"], args.max_generation_prompts
        ),
        "behavior_probes": _limited(suite["behavior_probes"], args.max_behavior_probes),
    }
    counts = {name: len(values) for name, values in selected.items()}
    if any(count == 0 for count in counts.values()):
        raise QwenCodebookRunnerError(
            f"Evaluation contract has an empty selected collection: {counts}."
        )
    prompt_ids = {
        name: [str(item["id"]) for item in values]
        for name, values in selected.items()
        if name != "text_evals"
    }
    if any(len(ids) != len(set(ids)) for ids in prompt_ids.values()):
        raise QwenCodebookRunnerError("Evaluation contract contains duplicate prompt IDs.")
    return {
        "suite_source": str(suite["source"]),
        "counts": counts,
        "prompt_ids": prompt_ids,
        "text_evals_sha256": _canonical_sha256(selected["text_evals"]),
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "max_text_evals": args.max_text_evals,
            "max_choice_probes": args.max_choice_probes,
            "max_generation_prompts": args.max_generation_prompts,
            "max_behavior_probes": args.max_behavior_probes,
        },
    }


def _validate_eval_report_contract(
    report: Mapping[str, Any], contract: Mapping[str, Any], label: str
) -> dict[str, Any]:
    provenance = report.get("model", report.get("source_model"))
    if not isinstance(provenance, str) or not provenance.strip():
        raise QwenCodebookRunnerError(f"{label} lacks model provenance.")
    suite = report.get("eval_suite")
    if not isinstance(suite, Mapping):
        raise QwenCodebookRunnerError(f"{label} lacks eval_suite metadata.")
    if str(suite.get("source", "")) != str(contract["suite_source"]):
        raise QwenCodebookRunnerError(f"{label} eval-suite source differs from contract.")
    expected_counts = contract["counts"]
    actual_counts = {
        "text_evals": suite.get("text_eval_count"),
        "choice_probes": suite.get("choice_probe_count"),
        "generation_prompts": suite.get("generation_prompt_count"),
        "behavior_probes": suite.get("behavior_probe_count"),
    }
    if actual_counts != expected_counts:
        raise QwenCodebookRunnerError(
            f"{label} eval counts do not match the locked contract: "
            f"expected={expected_counts}, actual={actual_counts}."
        )
    expected_ids = contract["prompt_ids"]
    actual_ids = {
        "choice_probes": [
            str(item.get("id"))
            for item in (report.get("choice_accuracy") or {}).get("items", [])
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        ],
        "generation_prompts": [
            str(item.get("id"))
            for item in report.get("generations", [])
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        ],
        "behavior_probes": [
            str(item.get("id"))
            for item in (report.get("behavior_suite") or {}).get("items", [])
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        ],
    }
    if actual_ids != expected_ids:
        raise QwenCodebookRunnerError(
            f"{label} prompt IDs do not match the locked contract."
        )
    perplexity = (report.get("perplexity") or {}).get("items")
    if not isinstance(perplexity, list) or len(perplexity) != expected_counts["text_evals"]:
        raise QwenCodebookRunnerError(f"{label} text-evaluation count is not provable.")
    return {"model": provenance, "eval_counts": actual_counts, "prompt_ids": actual_ids}


@dataclass(slots=True)
class _CudaPhasePeaks:
    device: str
    allocated_cap: int
    reserved_cap: int
    phases: dict[str, dict[str, int | bool]]

    def start(self, phase: str) -> None:
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        self.phases[phase] = {"cuda_observed": False}

    def finish(self, phase: str) -> None:
        values = self.phases[phase]
        if not (torch.cuda.is_available() and self.device.startswith("cuda")):
            return
        allocated = int(torch.cuda.max_memory_allocated(self.device))
        reserved = int(torch.cuda.max_memory_reserved(self.device))
        values.update(
            {
                "cuda_observed": True,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "max_cuda_allocated_bytes": self.allocated_cap,
                "max_cuda_reserved_bytes": self.reserved_cap,
            }
        )
        if allocated > self.allocated_cap or reserved > self.reserved_cap:
            raise QwenCodebookRunnerError(
                f"CUDA memory cap exceeded in {phase}: allocated={allocated}, reserved={reserved}."
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Load one parent, convert it in place, evaluate it, and write proofs."""

    import transformers

    _validate_args(args)
    contract = _suite_contract(args)
    cached_parent = _load_eval_report(args.cached_parent_eval, "cached parent evaluation")
    parent_contract = _validate_eval_report_contract(
        cached_parent, contract, "Cached parent evaluation"
    )
    output = args.output_dir.expanduser().resolve()
    resumed = output.exists()
    final_report_path = output / "run_report.json"
    if final_report_path.exists():
        existing = _load_eval_report(final_report_path, "existing final run report")
        validate_run_report(existing)
        raise FileExistsError(f"output directory is already certified: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_stage(output, "preflight", "running", resumed=resumed)
    _write_json(
        output / "certification_contract.json",
        {"contract": contract, "cached_parent": parent_contract},
    )
    peaks = _CudaPhasePeaks(
        str(args.device),
        _positive_int(args.max_cuda_allocated_bytes, "max_cuda_allocated_bytes"),
        _positive_int(args.max_cuda_reserved_bytes, "max_cuda_reserved_bytes"),
        {},
    )

    all_prompts = load_calibration_prompts(args.calibration_jsonl)
    source_tokenizer_name = args.source_tokenizer or args.source_model
    _progress(
        "load_tokenizers", source=source_tokenizer_name, target=args.target_tokenizer
    )
    source_tokenizer = transformers.AutoTokenizer.from_pretrained(
        source_tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.target_tokenizer,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    prompts, prompt_report = select_token_disjoint_calibration_prompts(
        all_prompts,
        args.calibration_count,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        eval_suite=args.eval_suite,
        max_length=args.calibration_max_length,
    )
    parent_rows = exact_token_row_mapping(source_tokenizer, target_tokenizer)
    source_token_split = verify_token_split_disjoint(
        source_tokenizer,
        prompts,
        args.eval_suite,
        max_length=args.calibration_max_length,
    )
    target_token_split = verify_token_split_disjoint(
        target_tokenizer,
        prompts,
        args.eval_suite,
        max_length=args.calibration_max_length,
    )
    token_split = {
        "method": "exact-token-id-sequences-under-both-tokenizers",
        "source": source_token_split,
        "target": target_token_split,
        "overlap_count": 0,
        "disjoint": True,
    }
    prompt_report["token_split"] = token_split
    prompt_report["target_vocab_size"] = len(parent_rows)
    prompt_report["parent_token_rows_sha256"] = _canonical_sha256(parent_rows)
    _write_json(output / "prompt_selection.json", prompt_report)

    _write_stage(output, "load_parent", "running", model=str(args.source_model))
    _progress("load_parent", model=args.source_model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.source_model,
        torch_dtype=_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)
    model.eval()
    _assert_zero_gradients(model, "loaded parent")
    _assert_model_finite(model, "loaded parent")
    bundles = build_qwen3_bundles(model)
    snapshot = _dense_runtime_snapshot(model, bundles)
    p_neuron = _resolve_prototype_depth(
        args.p_neuron, bundles.logical_layers, "p_neuron"
    )
    p_attention = _resolve_prototype_depth(
        args.p_attention, bundles.logical_layers, "p_attention"
    )

    _write_stage(output, "collect_activations", "running", prompts=len(prompts))
    _progress(
        "collect_activations",
        prompts=len(prompts),
        rows=args.activation_rows,
    )
    peaks.start("collect_activations")
    activations = collect_bounded_qwen_activations(
        model,
        source_tokenizer,
        prompts,
        max_length=args.calibration_max_length,
        activation_rows=args.activation_rows,
        device=args.device,
        max_host_activation_bytes=args.max_host_activation_bytes,
    )
    peaks.finish("collect_activations")
    calibration_report = {
        **activations.report,
        "prompt_selection": prompt_report,
        "token_split": token_split,
    }
    activation_memory = {
        name: calibration_report[name]
        for name in (
            "retained_host_activation_bytes",
            "peak_host_activation_bytes",
            "max_host_activation_bytes",
        )
    }
    _write_json(output / "calibration_report.json", calibration_report)

    _write_stage(output, "infer_alignment", "running")
    _progress("infer_alignment", reference_layer=args.reference_layer)
    peaks.start("infer_alignment")
    alignments = infer_codebook_permutations(
        bundles,
        neuron_activations=activations.neuron_activations,
        attention_activations=activations.attention_activations,
        reference_layer=args.reference_layer,
        neuron_weight_sketch_size=args.neuron_weight_sketch_size,
        neuron_activation_sketch_size=args.neuron_activation_sketch_size,
        attention_weight_sketch_size=args.attention_weight_sketch_size,
        attention_activation_sketch_size=args.attention_activation_sketch_size,
        coarse_signature_dimensions=args.coarse_signature_dimensions,
        coarse_bits_per_dimension=args.coarse_bits_per_dimension,
        local_block_size=args.local_block_size,
        refinement_passes=args.refinement_passes,
        max_exact_gqa_groups=args.max_exact_gqa_groups,
    )
    alignment_report = build_alignment_report(alignments)
    peaks.finish("infer_alignment")
    alignment_path = output / "alignment_report.json"
    _write_json(alignment_path, alignment_report)

    full_depth_noop = p_neuron == p_attention == bundles.logical_layers
    noop_reference = (
        _capture_noop_reference(
            model,
            target_tokenizer,
            prompts[0],
            parent_rows,
            max_length=args.calibration_max_length,
            device=args.device,
        )
        if full_depth_noop
        else None
    )
    _write_stage(output, "convert_in_place", "running")
    _progress(
        "convert_in_place",
        p_attention=p_attention,
        p_neuron=p_neuron,
    )
    peaks.start("convert_in_place")
    converted = prepare_qwen3_codebook_model(
        model,
        source_tokenizer,
        target_tokenizer,
        neuron_permutations=alignments.neurons.permutations,
        group_permutations=alignments.attention.permutations,
        attention_prototypes_per_group=p_attention,
        neuron_prototypes_per_neuron=p_neuron,
        attention_use_scales=False,
        neuron_use_scales=False,
        attention_sketch_dim=args.attention_weight_sketch_size,
        neuron_weight_sketch_size=args.neuron_weight_sketch_size,
    )
    if converted is not model:
        raise QwenCodebookRunnerError("Qwen3 conversion was not in place.")

    noop_report: dict[str, Any] = {
        "available_at_logical_layers": True,
        "requested": full_depth_noop,
        "verified": None,
    }
    if full_depth_noop:
        if noop_reference is None:
            raise QwenCodebookRunnerError("Full-depth no-op reference is missing.")
        noop_report.update(_verify_full_depth_projections(model, bundles))
        noop_report.update(
            _verify_noop_logits(model, noop_reference, device=args.device)
        )
        noop_report["verified"] = True
    peaks.finish("convert_in_place")

    parameter_report = qwen3_codebook_parameter_report(model).as_dict()
    accounting = build_parameter_index_accounting(
        model, p_neuron=p_neuron, p_attention=p_attention
    )
    _write_json(output / "compile_report.json", parameter_report)
    _write_json(output / "parameter_index_accounting.json", accounting)

    del bundles
    del activations
    del alignments
    del noop_reference
    gc.collect()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    ownership = verify_runtime_ownership(
        model,
        snapshot,
        parent_rows,
        require_process_release=True,
    )
    _assert_zero_gradients(model, "converted runtime")
    _assert_model_finite(model, "converted runtime")
    verification = {
        **ownership,
        "conversion_in_place": converted is model,
        "zero_optimizer_steps": 0,
        "backward_called": False,
        "gradients_absent": True,
        "finite_runtime": True,
        "full_depth_noop": noop_report,
        "all_passed": True,
    }
    _write_json(output / "runtime_verification.json", verification)

    _write_stage(output, "compact_checkpoint", "running")
    peaks.start("compact_checkpoint")
    checkpoint_directory = output / "compact_child"
    checkpoint_lock = output.parent / ".compact_checkpoint_io.lock"
    _progress("compact_checkpoint_io_wait", lock=checkpoint_lock)
    with _checkpoint_io_lease(checkpoint_lock) as checkpoint_io_lease:
        _progress(
            "compact_checkpoint_io_acquired",
            wait_seconds=round(float(checkpoint_io_lease["wait_seconds"]), 6),
        )
        checkpoint_manifest = save_qwen3_codebook_checkpoint(
            model,
            checkpoint_directory,
            tokenizer_contract=_tokenizer_checkpoint_contract(
                target_tokenizer,
                tokenizer_name=str(args.target_tokenizer),
                parent_rows=parent_rows,
            ),
        )
        reloaded_model = load_qwen3_codebook_checkpoint(
            checkpoint_directory,
            device=args.device,
        )
        checkpoint_equivalence = verify_compact_checkpoint_equivalence(
            model,
            reloaded_model,
            target_tokenizer,
            prompts[0],
            max_length=args.calibration_max_length,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    peaks.finish("compact_checkpoint")
    checkpoint_report = {
        "artifact_directory": str(checkpoint_directory),
        "manifest": checkpoint_manifest,
        "equivalence": checkpoint_equivalence,
        "io_lease": checkpoint_io_lease,
        "standalone_parent_free": (
            checkpoint_equivalence["reloaded_parent_runtime_references"] == 0
        ),
    }
    if not checkpoint_report["standalone_parent_free"]:
        raise QwenCodebookRunnerError("Reloaded compact child retains a parent runtime.")
    verification["compact_checkpoint"] = checkpoint_report
    _write_json(output / "runtime_verification.json", verification)
    _write_json(output / "compact_checkpoint_report.json", checkpoint_report)

    candidate_eval_path = output / "candidate_eval.json"
    candidate_work_path = output / ".candidate_eval.work.json"
    _write_stage(output, "candidate_eval", "running")
    _progress("candidate_eval_start")
    peaks.start("candidate_eval")
    candidate_eval = evaluate_loaded_text_model(
        reloaded_model,
        target_tokenizer,
        candidate_work_path,
        model_label=f"qwen3-codebook:pn{p_neuron}:pa{p_attention}",
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )
    peaks.finish("candidate_eval")
    candidate_contract = _validate_eval_report_contract(
        candidate_eval, contract, "Candidate evaluation"
    )
    candidate_eval["certification_context"] = {
        "contract": contract,
        "model_provenance": {
            "source_model": str(args.source_model),
            "loaded_model_name_or_path": str(
                getattr(reloaded_model.config, "_name_or_path", args.source_model)
            ),
            "model_type": str(getattr(reloaded_model.config, "model_type", "")),
        },
        "runtime": {
            "requested_dtype": args.dtype,
            "parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in reloaded_model.parameters()}
            ),
            "device": str(args.device),
            "transformers_version": str(transformers.__version__),
            "torch_version": str(torch.__version__),
        },
        "validated_candidate": candidate_contract,
    }
    _write_json(candidate_eval_path, candidate_eval)
    candidate_work_path.unlink(missing_ok=True)
    comparison_path = output / "parent_candidate_comparison.json"
    comparison = _compare_evals(
        args.cached_parent_eval,
        candidate_eval_path,
        manual_sample_count=args.manual_sample_count,
    )
    _write_json(comparison_path, comparison)
    execution_configuration = {
        "dtype": args.dtype,
        "device": str(args.device),
        "parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in reloaded_model.parameters()}
        ),
        "model_provenance": {
            "source_model": str(args.source_model),
            "source_tokenizer": str(source_tokenizer_name),
            "target_tokenizer": str(args.target_tokenizer),
            "loaded_model_name_or_path": str(
                getattr(reloaded_model.config, "_name_or_path", args.source_model)
            ),
            "model_type": str(getattr(reloaded_model.config, "model_type", "")),
            "cached_parent_model": parent_contract["model"],
        },
        "sketch": {
            "neuron_weight": args.neuron_weight_sketch_size,
            "neuron_activation": args.neuron_activation_sketch_size,
            "attention_weight": args.attention_weight_sketch_size,
            "attention_activation": args.attention_activation_sketch_size,
            "coarse_signature_dimensions": args.coarse_signature_dimensions,
            "coarse_bits_per_dimension": args.coarse_bits_per_dimension,
            "local_block_size": args.local_block_size,
            "refinement_passes": args.refinement_passes,
            "max_exact_gqa_groups": args.max_exact_gqa_groups,
        },
        "evaluation": contract["settings"],
        "generation": {"max_new_tokens": args.max_new_tokens},
        "packages": {
            "transformers": str(transformers.__version__),
            "torch": str(torch.__version__),
            "python": sys.version.split()[0],
        },
        "memory": {
            "activation": {
                **activation_memory,
            },
            "cuda_phase_peaks": peaks.phases,
        },
        "compact_checkpoint": checkpoint_report,
    }
    run_report = build_run_report(
        source_model=str(args.source_model),
        source_tokenizer=str(source_tokenizer_name),
        target_tokenizer=str(args.target_tokenizer),
        p_neuron=p_neuron,
        p_attention=p_attention,
        calibration=calibration_report,
        alignment=alignment_report,
        alignment_path=alignment_path,
        compile_report=parameter_report,
        accounting=accounting,
        verification=verification,
        candidate_eval=candidate_eval,
        comparison=comparison,
        comparison_path=comparison_path,
        manual_sample_count=args.manual_sample_count,
        execution_configuration=execution_configuration,
        certification_contract={"contract": contract, "cached_parent": parent_contract},
    )
    _write_json(output / "run_report.json", run_report)
    _write_stage(output, "complete", "certified", manual_samples=args.manual_sample_count)
    child = comparison.get("child") or {}
    _progress(
        "complete",
        behavior=child.get("behavior_correct"),
        choice=child.get("choice_correct"),
        generation=child.get("generation_correct"),
    )
    return run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument(
        "--source-tokenizer", "--source-tokenizer-model", dest="source_tokenizer"
    )
    parser.add_argument(
        "--target-tokenizer",
        "--target-tokenizer-model",
        dest="target_tokenizer",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-jsonl",
        "--calibration-file",
        dest="calibration_jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument("--eval-suite", type=Path, required=True)
    parser.add_argument("--cached-parent-eval", type=Path, required=True)
    parser.add_argument(
        "--calibration-count",
        "--calibration-prompt-count",
        dest="calibration_count",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--calibration-max-length",
        "--max-length",
        dest="calibration_max_length",
        type=int,
        default=160,
    )
    parser.add_argument(
        "--activation-rows",
        "--max-activation-rows",
        "--calibration-rows",
        dest="activation_rows",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--p-neuron",
        "--neuron-prototypes-per-neuron",
        dest="p_neuron",
        type=_prototype_depth_argument,
        required=True,
    )
    parser.add_argument(
        "--p-attention",
        "--attention-prototypes-per-group",
        dest="p_attention",
        type=_prototype_depth_argument,
        required=True,
    )
    parser.add_argument(
        "--neuron-weight-sketch-size",
        "--neuron-sketch-size",
        dest="neuron_weight_sketch_size",
        type=int,
        default=DEFAULT_WEIGHT_SKETCH_SIZE,
    )
    parser.add_argument(
        "--neuron-activation-sketch-size",
        type=int,
        default=DEFAULT_ACTIVATION_SKETCH_SIZE,
    )
    parser.add_argument(
        "--attention-weight-sketch-size",
        "--attention-sketch-size",
        dest="attention_weight_sketch_size",
        type=int,
        default=DEFAULT_WEIGHT_SKETCH_SIZE,
    )
    parser.add_argument(
        "--attention-activation-sketch-size",
        type=int,
        default=DEFAULT_ACTIVATION_SKETCH_SIZE,
    )
    parser.add_argument("--reference-layer", type=int, default=0)
    parser.add_argument(
        "--coarse-signature-dimensions",
        type=int,
        default=DEFAULT_SIGNATURE_DIMENSIONS,
    )
    parser.add_argument(
        "--coarse-bits-per-dimension", type=int, default=DEFAULT_SIGNATURE_BITS
    )
    parser.add_argument(
        "--local-block-size", type=int, default=DEFAULT_LOCAL_BLOCK_SIZE
    )
    parser.add_argument(
        "--refinement-passes", type=int, default=DEFAULT_REFINEMENT_PASSES
    )
    parser.add_argument(
        "--max-exact-gqa-groups", type=int, default=DEFAULT_MAX_EXACT_GQA_GROUPS
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-text-evals", type=int, default=1000)
    parser.add_argument("--max-choice-probes", type=int, default=250)
    parser.add_argument("--max-generation-prompts", type=int, default=1000)
    parser.add_argument("--max-behavior-probes", type=int, default=1000)
    parser.add_argument("--manual-sample-count", type=int, default=128)
    parser.add_argument(
        "--max-host-activation-bytes", type=int, default=8 * 1024**3
    )
    parser.add_argument(
        "--max-cuda-allocated-bytes", type=int, default=80 * 1024**3
    )
    parser.add_argument(
        "--max-cuda-reserved-bytes", type=int, default=80 * 1024**3
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
