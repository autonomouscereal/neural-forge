"""Authenticated phase-level semantic spans for cross-tokenizer LFM2 replay.

The older LFM2 circuit caches preserve parent-tokenizer prompt IDs and one
forced donor token for every decode phase.  A target tokenizer cannot safely
reuse those numeric IDs, and a donor token need not correspond to exactly one
target token.  This module turns one already-authenticated cache batch into an
immutable target full-prefix schedule:

* decode the complete donor prefix represented by each parent phase;
* encode that exact text with the local target tokenizer;
* require byte-for-byte-equivalent Python text after target decoding; and
* retain the original ``prefill`` / ``decode.NNN`` identity even when target
  tokenization expands, contracts, or rewrites a suffix at a token boundary.

Only cached token tensors and local tokenizers are consulted.  There is no
parent or child model surface, optimizer, autograd, pruning, quantization, or
parameter mutation in this module.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_circuit_trace_cache import (
    BATCH_SCHEMA,
    CACHE_SCHEMA,
    Lfm2CircuitTraceBatch,
    _tensor_descriptor as _cache_tensor_descriptor,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
    _cache_input_sha256,
)
from neural_forge.hf_tokenizer_bridge import (
    StrictTokenizerBridgeError,
    fingerprint_hf_tokenizer_semantics,
)


SEMANTIC_SPAN_PHASE_BRIDGE_SCHEMA = "neural-forge-lfm2-semantic-span-phase-bridge-v1"
SEMANTIC_SPAN_PHASE_SCHEMA = "neural-forge-lfm2-semantic-span-phase-v1"
SEMANTIC_SPAN_EXPANSION_SCHEMA = "neural-forge-lfm2-semantic-span-expansion-report-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECODE_PHASE = re.compile(r"decode\.(\d{3,})\Z")
_INPUT_NAMES = (
    "input.attention_mask",
    "input.forced_decode_ids",
    "input.input_ids",
)
_MODEL_OPERATION_AUDIT = MappingProxyType(
    {
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "model_forward_calls": 0,
        "model_parameters_read": 0,
        "model_parameters_written": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "pruned_parameters": 0,
        "quantized_parameters": 0,
    }
)
_HORIZON_BUCKETS = (
    ("decode.000-007", 0, 8),
    ("decode.008-031", 8, 32),
    ("decode.032-plus", 32, None),
)


class Lfm2SemanticSpanPhaseBridgeError(ValueError):
    """A cache, tokenizer, phase, or semantic roundtrip was not authentic."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Semantic-span phase construction requires PyTorch."
        ) from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2SemanticSpanPhaseBridgeError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2SemanticSpanPhaseBridgeError(f"{name} must be a nonnegative integer.")
    return value


def _integer_tuple(
    value: Any, name: str, *, allow_empty: bool = False
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2SemanticSpanPhaseBridgeError(
            f"{name} must be a sequence of nonnegative token IDs."
        )
    result = tuple(value)
    if (not result and not allow_empty) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in result
    ):
        raise Lfm2SemanticSpanPhaseBridgeError(
            f"{name} must contain nonnegative token IDs."
        )
    return result


def _text_sha256(value: str) -> str:
    return canonical_sha256(
        {
            "schema": "neural-forge-exact-tokenizer-text-v1",
            "text": value,
        }
    )


def _decode_exact(tokenizer: Any, token_ids: Sequence[int], name: str) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if not callable(decoder):
        raise Lfm2SemanticSpanPhaseBridgeError(
            f"{name} tokenizer must expose decode()."
        )
    ids = list(token_ids)
    try:
        value = decoder(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        value = decoder(ids)
    except Exception as exc:
        raise Lfm2SemanticSpanPhaseBridgeError(
            f"{name} tokenizer rejected a phase prefix."
        ) from exc
    if not isinstance(value, str):
        raise Lfm2SemanticSpanPhaseBridgeError(
            f"{name} tokenizer decode() must return text."
        )
    return value


def _encode_exact(tokenizer: Any, text: str, name: str) -> tuple[int, ...]:
    encoder = getattr(tokenizer, "encode", None)
    value: Any
    if callable(encoder):
        try:
            value = encoder(text, add_special_tokens=False)
        except Exception as exc:
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"{name} tokenizer rejected exact semantic text."
            ) from exc
    else:
        caller = tokenizer if callable(tokenizer) else None
        if caller is None:
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"{name} tokenizer must expose encode() or be callable."
            )
        try:
            encoded = caller(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )
        except Exception as exc:
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"{name} tokenizer rejected exact semantic text."
            ) from exc
        if not isinstance(encoded, Mapping):
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"{name} tokenizer returned a non-mapping encoding."
            )
        value = encoded.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        value = value[0]
    return _integer_tuple(value, f"{name} encoded IDs", allow_empty=True)


def _tokenizer_capacity(tokenizer: Any, name: str) -> int:
    try:
        result = int(len(tokenizer))
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SemanticSpanPhaseBridgeError(
            f"{name} tokenizer length must be a positive integer."
        ) from exc
    if result <= 0:
        raise Lfm2SemanticSpanPhaseBridgeError(
            f"{name} tokenizer length must be a positive integer."
        )
    return result


def _live_tokenizer_fingerprint(tokenizer: Any, name: str) -> str:
    try:
        return fingerprint_hf_tokenizer_semantics(tokenizer)
    except StrictTokenizerBridgeError as exc:
        raise Lfm2SemanticSpanPhaseBridgeError(
            f"{name} tokenizer semantics could not be fingerprinted."
        ) from exc


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _validate_source_manifest(
    manifest: Mapping[str, Any],
    batch: Lfm2CircuitTraceBatch,
    *,
    expected_cache_key: str,
    expected_manifest_sha256: str,
    expected_input_sha256: str,
) -> tuple[str, str, str]:
    if not isinstance(manifest, Mapping):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "source_cache_manifest must be a mapping."
        )
    value = dict(manifest)
    seal = _digest(value.get("manifest_sha256"), "source manifest SHA-256")
    if seal != _digest(expected_manifest_sha256, "expected manifest SHA-256"):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache manifest differs from the expected manifest hash."
        )
    body = dict(value)
    body.pop("manifest_sha256", None)
    if canonical_sha256(body) != seal:
        raise Lfm2SemanticSpanPhaseBridgeError("Source cache manifest seal is invalid.")
    if value.get("schema") != CACHE_SCHEMA:
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache manifest schema is unsupported."
        )
    cache_key = _digest(value.get("cache_key"), "source cache key")
    if cache_key != _digest(expected_cache_key, "expected source cache key"):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache key differs from the expected cache key."
        )

    sections = {}
    for name in (
        "parent",
        "corpus",
        "implementation",
        "layer_topology",
        "capture",
        "runtime",
        "execution",
    ):
        section = value.get(name)
        if not isinstance(section, Mapping):
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"Source cache manifest section {name!r} is invalid."
            )
        sections[name] = dict(section)
    identity = {
        "schema": CACHE_SCHEMA,
        "parent_identity_sha256": _digest(
            sections["parent"].get("identity_sha256"),
            "source parent identity",
        ),
        "corpus_sha256": _digest(
            sections["corpus"].get("corpus_sha256"),
            "source corpus identity",
        ),
        "implementation_sha256": _digest(
            sections["implementation"].get("implementation_sha256"),
            "source implementation identity",
        ),
        "layer_topology_sha256": _digest(
            sections["layer_topology"].get("layer_topology_sha256"),
            "source topology identity",
        ),
        "selection_sha256": _digest(
            sections["capture"].get("selection_sha256"),
            "source capture selection identity",
        ),
        "execution_sha256": _digest(
            sections["execution"].get("execution_sha256"),
            "source execution identity",
        ),
        "runtime": sections["runtime"],
    }
    if canonical_sha256(identity) != cache_key:
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache key does not bind its manifest identities."
        )

    input_sha256 = _digest(
        expected_input_sha256,
        "expected source input SHA-256",
    )
    if batch.input_sha256 != input_sha256:
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source batch differs from the expected input hash."
        )
    corpus = sections["corpus"]
    ordered = corpus.get("ordered_batch_input_sha256")
    if (
        not isinstance(ordered, list)
        or batch.index < 0
        or batch.index >= len(ordered)
        or ordered[batch.index] != input_sha256
        or corpus.get("batch_count") != len(ordered)
    ):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source batch input is not bound at its manifest corpus ordinal."
        )
    return cache_key, seal, input_sha256


def _validated_source_batch(
    batch: Lfm2CircuitTraceBatch,
    *,
    cache_key: str,
    expected_input_sha256: str,
) -> tuple[Any, Any, Any, Any | None, str]:
    if not isinstance(batch, Lfm2CircuitTraceBatch):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "source_batch must be an authenticated Lfm2CircuitTraceBatch."
        )
    torch = _torch()
    tensors = batch.tensors
    input_ids = tensors.get("input.input_ids")
    attention_mask = tensors.get("input.attention_mask")
    forced_ids = tensors.get("input.forced_decode_ids")
    position_ids = tensors.get("input.position_ids")
    for name, value in (
        ("input.input_ids", input_ids),
        ("input.attention_mask", attention_mask),
        ("input.forced_decode_ids", forced_ids),
    ):
        if (
            not torch.is_tensor(value)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.ndim != 2
            or value.numel() == 0
            or value.requires_grad
        ):
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"{name} must be a detached nonempty rank-two dense tensor."
            )
        if value.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise Lfm2SemanticSpanPhaseBridgeError(f"{name} must use an integer dtype.")
        if bool((value < 0).any().item()):
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"{name} contains a negative token or mask value."
            )
    if int(input_ids.shape[0]) != 1 or int(forced_ids.shape[0]) != 1:
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Semantic-span schedules require one authenticated cache sequence."
        )
    if tuple(input_ids.shape) != tuple(attention_mask.shape):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source attention mask does not match input IDs."
        )
    if not bool((attention_mask == 1).all().item()):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache sequence must be the canonical unpadded sequence."
        )
    if position_ids is not None:
        if (
            not torch.is_tensor(position_ids)
            or position_ids.ndim != 2
            or tuple(position_ids.shape) != tuple(input_ids.shape)
            or position_ids.dtype
            not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            or bool((position_ids < 0).any().item())
        ):
            raise Lfm2SemanticSpanPhaseBridgeError(
                "Source position IDs must be nonnegative integers matching input IDs."
            )
        expected = position_ids[:, :1] + torch.arange(
            int(position_ids.shape[1]),
            dtype=position_ids.dtype,
            device=position_ids.device,
        ).reshape(1, -1)
        if not torch.equal(position_ids, expected):
            raise Lfm2SemanticSpanPhaseBridgeError(
                "Source position IDs must be a contiguous canonical sequence."
            )

    observed_input_sha256 = _cache_input_sha256(
        input_ids=input_ids,
        attention_mask=attention_mask,
        forced_decode_ids=forced_ids,
        position_ids=position_ids,
    )
    if observed_input_sha256 != expected_input_sha256:
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache input tensors no longer match their input hash."
        )

    record = batch.record
    record_sha256 = _digest(record.get("record_sha256"), "source record hash")
    record_body = dict(record)
    record_body.pop("record_sha256", None)
    if canonical_sha256(record_body) != record_sha256:
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache batch record seal is invalid."
        )
    if (
        record.get("schema") != BATCH_SCHEMA
        or record.get("cache_key") != cache_key
        or record.get("batch_index") != batch.index
        or record.get("input_sha256") != expected_input_sha256
    ):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache batch record binding is invalid."
        )
    inventory = record.get("tensors")
    if not isinstance(inventory, Mapping):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache tensor inventory is invalid."
        )
    if record.get("tensor_count") != len(inventory) or len(inventory) != len(tensors):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache tensor count differs from its authenticated inventory."
        )
    input_names = list(_INPUT_NAMES)
    if position_ids is not None:
        input_names.append("input.position_ids")
    for name in input_names:
        if inventory.get(name) != _cache_tensor_descriptor(torch, tensors[name]):
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"Source cache input tensor {name!r} differs from its inventory."
            )
    capture = record.get("capture")
    if not isinstance(capture, Mapping) or capture.get("decode_steps") != int(
        forced_ids.shape[1]
    ):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache record does not preserve every forced decode phase."
        )
    phase_order = (
        "prefill",
        *(f"decode.{step:03d}" for step in range(int(forced_ids.shape[1]))),
    )
    diagnostics = capture.get("phase_diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(phase_order):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache phase diagnostics do not cover every parent phase."
        )
    for phase_index, phase in enumerate(phase_order):
        logits_name = f"{phase}.logits"
        logits = tensors.get(logits_name)
        if logits is None or inventory.get(logits_name) != _cache_tensor_descriptor(
            torch, logits
        ):
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"Source parent phase {phase!r} lacks authenticated logits."
            )
        if phase_index == 0:
            continue
        forced_name = f"{phase}.forced_token_ids"
        forced_token = tensors.get(forced_name)
        if (
            forced_token is None
            or inventory.get(forced_name)
            != _cache_tensor_descriptor(torch, forced_token)
            or not torch.equal(
                forced_token.detach().to(device="cpu", dtype=torch.long),
                forced_ids[:, phase_index - 1 : phase_index]
                .detach()
                .to(device="cpu", dtype=torch.long),
            )
        ):
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"Source parent phase {phase!r} lost its forced-token identity."
            )
    return (
        input_ids.detach().to(device="cpu", dtype=torch.long).contiguous().clone(),
        attention_mask.detach().to(device="cpu", dtype=torch.long).contiguous().clone(),
        forced_ids.detach().to(device="cpu", dtype=torch.long).contiguous().clone(),
        None
        if position_ids is None
        else position_ids.detach()
        .to(device="cpu", dtype=torch.long)
        .contiguous()
        .clone(),
        record_sha256,
    )


@dataclass(frozen=True, slots=True)
class Lfm2SemanticSpanPhase:
    """One parent phase and its exact target-tokenizer full prefix."""

    phase: str
    parent_decode_step: int | None
    donor_forced_token_id: int | None
    donor_full_prefix_token_count: int
    target_full_prefix_token_ids: tuple[int, ...]
    target_position_origin: int
    target_rewrite_start: int
    target_replaced_suffix_token_count: int
    target_replacement_token_ids: tuple[int, ...]
    semantic_character_count: int
    semantic_text_sha256: str
    exact_semantic_roundtrip: bool = True
    phase_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        step = self.parent_decode_step
        if self.phase == "prefill":
            if step is not None or self.donor_forced_token_id is not None:
                raise Lfm2SemanticSpanPhaseBridgeError(
                    "Prefill cannot declare a parent decode token."
                )
        else:
            match = _DECODE_PHASE.fullmatch(self.phase)
            if (
                match is None
                or step is None
                or int(match.group(1)) != step
                or self.donor_forced_token_id is None
            ):
                raise Lfm2SemanticSpanPhaseBridgeError(
                    "Decode phase identity does not match its parent step."
                )
            _nonnegative_int(self.donor_forced_token_id, "donor forced token ID")
        _nonnegative_int(self.donor_full_prefix_token_count, "donor prefix length")
        ids = _integer_tuple(
            self.target_full_prefix_token_ids,
            "target full-prefix token IDs",
        )
        replacement = _integer_tuple(
            self.target_replacement_token_ids,
            "target replacement token IDs",
            allow_empty=True,
        )
        origin = _nonnegative_int(self.target_position_origin, "position origin")
        rewrite = _nonnegative_int(self.target_rewrite_start, "rewrite start")
        _nonnegative_int(
            self.target_replaced_suffix_token_count,
            "replaced suffix token count",
        )
        _nonnegative_int(
            self.semantic_character_count,
            "semantic character count",
        )
        if rewrite > len(ids) or replacement != ids[rewrite:]:
            raise Lfm2SemanticSpanPhaseBridgeError(
                "Target replacement span does not reconstruct its full prefix."
            )
        if not self.exact_semantic_roundtrip:
            raise Lfm2SemanticSpanPhaseBridgeError(
                "Semantic-span phases must have an exact text roundtrip."
            )
        object.__setattr__(self, "target_full_prefix_token_ids", ids)
        object.__setattr__(self, "target_replacement_token_ids", replacement)
        object.__setattr__(self, "target_position_origin", origin)
        object.__setattr__(
            self,
            "semantic_text_sha256",
            _digest(self.semantic_text_sha256, "semantic text hash"),
        )
        object.__setattr__(self, "phase_sha256", canonical_sha256(self._body()))

    @property
    def target_full_prefix_length(self) -> int:
        return len(self.target_full_prefix_token_ids)

    @property
    def target_span_token_count(self) -> int:
        return len(self.target_replacement_token_ids)

    @property
    def target_attention_mask_ids(self) -> tuple[int, ...]:
        return (1,) * self.target_full_prefix_length

    @property
    def target_position_ids_tuple(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.target_position_origin,
                self.target_position_origin + self.target_full_prefix_length,
            )
        )

    @property
    def target_input_ids(self) -> Any:
        return _torch().tensor(
            [self.target_full_prefix_token_ids],
            dtype=_torch().long,
        )

    @property
    def target_attention_mask(self) -> Any:
        return _torch().ones(
            (1, self.target_full_prefix_length),
            dtype=_torch().long,
        )

    @property
    def target_position_ids(self) -> Any:
        return _torch().tensor(
            [self.target_position_ids_tuple],
            dtype=_torch().long,
        )

    def as_target_full_prefix_batch(self) -> dict[str, Any]:
        return {
            "input_ids": self.target_input_ids,
            "attention_mask": self.target_attention_mask,
            "position_ids": self.target_position_ids,
        }

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_SPAN_PHASE_SCHEMA,
            "phase": self.phase,
            "parent_decode_step": self.parent_decode_step,
            "donor_forced_token_id": self.donor_forced_token_id,
            "donor_full_prefix_token_count": self.donor_full_prefix_token_count,
            "target_full_prefix_token_ids": list(self.target_full_prefix_token_ids),
            "target_attention_mask": list(self.target_attention_mask_ids),
            "target_position_ids": list(self.target_position_ids_tuple),
            "target_rewrite_start": self.target_rewrite_start,
            "target_replaced_suffix_token_count": (
                self.target_replaced_suffix_token_count
            ),
            "target_replacement_token_ids": list(self.target_replacement_token_ids),
            "target_span_token_count": self.target_span_token_count,
            "semantic_character_count": self.semantic_character_count,
            "semantic_text_sha256": self.semantic_text_sha256,
            "exact_semantic_roundtrip": self.exact_semantic_roundtrip,
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "phase_sha256": self.phase_sha256}


def _build_phases(
    *,
    prompt_ids: tuple[int, ...],
    forced_ids: tuple[int, ...],
    position_origin: int,
    donor_tokenizer: Any,
    target_tokenizer: Any,
) -> tuple[Lfm2SemanticSpanPhase, ...]:
    donor_capacity = _tokenizer_capacity(donor_tokenizer, "donor")
    target_capacity = _tokenizer_capacity(target_tokenizer, "target")
    if max((*prompt_ids, *forced_ids)) >= donor_capacity:
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Source cache contains a token outside the donor tokenizer."
        )
    phases: list[Lfm2SemanticSpanPhase] = []
    prior_target_ids: tuple[int, ...] = ()
    for ordinal in range(len(forced_ids) + 1):
        if ordinal == 0:
            phase = "prefill"
            step = None
            token_id = None
            donor_prefix = prompt_ids
        else:
            step = ordinal - 1
            phase = f"decode.{step:03d}"
            token_id = forced_ids[step]
            donor_prefix = (*prompt_ids, *forced_ids[:ordinal])
        semantic_text = _decode_exact(
            donor_tokenizer,
            donor_prefix,
            "donor",
        )
        target_ids = _encode_exact(target_tokenizer, semantic_text, "target")
        if not target_ids:
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"Target tokenizer produced an empty prefix for {phase!r}."
            )
        if max(target_ids) >= target_capacity:
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"Target tokenizer produced an out-of-vocabulary ID for {phase!r}."
            )
        roundtrip = _decode_exact(target_tokenizer, target_ids, "target")
        if roundtrip != semantic_text:
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"Target tokenizer failed the exact semantic roundtrip for {phase!r}."
            )
        if _encode_exact(target_tokenizer, roundtrip, "target") != target_ids:
            raise Lfm2SemanticSpanPhaseBridgeError(
                f"Target tokenizer encoding is not stable for {phase!r}."
            )
        rewrite_start = _longest_common_prefix(prior_target_ids, target_ids)
        phases.append(
            Lfm2SemanticSpanPhase(
                phase=phase,
                parent_decode_step=step,
                donor_forced_token_id=token_id,
                donor_full_prefix_token_count=len(donor_prefix),
                target_full_prefix_token_ids=target_ids,
                target_position_origin=position_origin,
                target_rewrite_start=rewrite_start,
                target_replaced_suffix_token_count=(
                    len(prior_target_ids) - rewrite_start
                ),
                target_replacement_token_ids=target_ids[rewrite_start:],
                semantic_character_count=len(semantic_text),
                semantic_text_sha256=_text_sha256(semantic_text),
            )
        )
        prior_target_ids = target_ids
    return tuple(phases)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _expansion_report(
    phases: Sequence[Lfm2SemanticSpanPhase],
) -> dict[str, Any]:
    if not phases or phases[0].phase != "prefill":
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Expansion reporting requires a prefill-first phase schedule."
        )
    prefill = phases[0]
    decode = tuple(phases[1:])
    bucket_rows = []
    for name, start, stop in _HORIZON_BUCKETS:
        selected = tuple(
            phase
            for phase in decode
            if phase.parent_decode_step is not None
            and phase.parent_decode_step >= start
            and (stop is None or phase.parent_decode_step < stop)
        )
        replacement_count = sum(phase.target_span_token_count for phase in selected)
        rewritten_count = sum(
            phase.target_replaced_suffix_token_count for phase in selected
        )
        lengths = [phase.target_full_prefix_length for phase in selected]
        bucket_rows.append(
            {
                "name": name,
                "start_parent_decode_step": start,
                "stop_parent_decode_step_exclusive": stop,
                "phase_count": len(selected),
                "first_phase": None if not selected else selected[0].phase,
                "last_phase": None if not selected else selected[-1].phase,
                "target_replacement_token_count": replacement_count,
                "target_rewritten_prior_token_count": rewritten_count,
                "target_tokens_per_parent_phase": _ratio(
                    replacement_count,
                    len(selected),
                ),
                "minimum_target_full_prefix_tokens": (
                    None if not lengths else min(lengths)
                ),
                "maximum_target_full_prefix_tokens": (
                    None if not lengths else max(lengths)
                ),
            }
        )
    final = phases[-1]
    total_replacement = sum(phase.target_span_token_count for phase in decode)
    return {
        "schema": SEMANTIC_SPAN_EXPANSION_SCHEMA,
        "prefill": {
            "donor_token_count": prefill.donor_full_prefix_token_count,
            "target_token_count": prefill.target_full_prefix_length,
            "target_to_donor_token_ratio": _ratio(
                prefill.target_full_prefix_length,
                prefill.donor_full_prefix_token_count,
            ),
        },
        "full_horizon": {
            "parent_decode_phase_count": len(decode),
            "last_parent_decode_step": (
                None if not decode else decode[-1].parent_decode_step
            ),
            "final_donor_prefix_token_count": (final.donor_full_prefix_token_count),
            "final_target_prefix_token_count": (final.target_full_prefix_length),
            "final_target_to_donor_token_ratio": _ratio(
                final.target_full_prefix_length,
                final.donor_full_prefix_token_count,
            ),
            "target_replacement_token_count": total_replacement,
            "target_tokens_per_parent_decode_phase": _ratio(
                total_replacement,
                len(decode),
            ),
            "retokenized_suffix_phase_count": sum(
                phase.target_replaced_suffix_token_count > 0 for phase in decode
            ),
            "zero_target_span_phase_count": sum(
                phase.target_span_token_count == 0 for phase in decode
            ),
            "maximum_target_span_token_count": max(
                (phase.target_span_token_count for phase in decode),
                default=0,
            ),
        },
        "horizon_buckets": bucket_rows,
    }


@dataclass(frozen=True, slots=True)
class Lfm2SemanticSpanPhaseBridge:
    """Sealed immutable target prefixes for every phase of one cache batch."""

    source_cache_key: str
    source_manifest_sha256: str
    source_batch_index: int
    source_input_sha256: str
    source_record_sha256: str
    donor_tokenizer_semantics_sha256: str
    target_tokenizer_semantics_sha256: str
    source_prompt_token_ids: tuple[int, ...]
    source_forced_decode_token_ids: tuple[int, ...]
    target_position_origin: int
    phases: tuple[Lfm2SemanticSpanPhase, ...]
    bridge_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_cache_key",
            "source_manifest_sha256",
            "source_input_sha256",
            "source_record_sha256",
            "donor_tokenizer_semantics_sha256",
            "target_tokenizer_semantics_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "source_batch_index",
            _nonnegative_int(self.source_batch_index, "source batch index"),
        )
        object.__setattr__(
            self,
            "source_prompt_token_ids",
            _integer_tuple(self.source_prompt_token_ids, "source prompt IDs"),
        )
        object.__setattr__(
            self,
            "source_forced_decode_token_ids",
            _integer_tuple(
                self.source_forced_decode_token_ids,
                "source forced decode IDs",
            ),
        )
        object.__setattr__(
            self,
            "target_position_origin",
            _nonnegative_int(self.target_position_origin, "target position origin"),
        )
        phases = tuple(self.phases)
        expected = (
            "prefill",
            *(
                f"decode.{step:03d}"
                for step in range(len(self.source_forced_decode_token_ids))
            ),
        )
        if (
            not phases
            or any(not isinstance(value, Lfm2SemanticSpanPhase) for value in phases)
            or tuple(value.phase for value in phases) != expected
        ):
            raise Lfm2SemanticSpanPhaseBridgeError(
                "Phase schedule does not preserve every parent phase identity."
            )
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "bridge_sha256", canonical_sha256(self._body()))

    @property
    def phase_order(self) -> tuple[str, ...]:
        return tuple(value.phase for value in self.phases)

    @property
    def parent_decode_phase_count(self) -> int:
        return len(self.phases) - 1

    @property
    def model_operation_audit(self) -> dict[str, int]:
        return dict(_MODEL_OPERATION_AUDIT)

    @property
    def expansion_report(self) -> dict[str, Any]:
        return _expansion_report(self.phases)

    def get_phase(self, phase: str) -> Lfm2SemanticSpanPhase:
        for value in self.phases:
            if value.phase == phase:
                return value
        raise KeyError(phase)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_SPAN_PHASE_BRIDGE_SCHEMA,
            "source_cache": {
                "cache_key": self.source_cache_key,
                "manifest_sha256": self.source_manifest_sha256,
                "batch_index": self.source_batch_index,
                "input_sha256": self.source_input_sha256,
                "record_sha256": self.source_record_sha256,
            },
            "tokenizers": {
                "donor_tokenizer_semantics_sha256": (
                    self.donor_tokenizer_semantics_sha256
                ),
                "target_tokenizer_semantics_sha256": (
                    self.target_tokenizer_semantics_sha256
                ),
                "numeric_token_id_equality_required": False,
                "alignment": "exact-full-prefix-text-roundtrip",
            },
            "source_prompt_token_ids_sha256": canonical_sha256(
                list(self.source_prompt_token_ids)
            ),
            "source_forced_decode_token_ids_sha256": canonical_sha256(
                list(self.source_forced_decode_token_ids)
            ),
            "source_prompt_token_count": len(self.source_prompt_token_ids),
            "source_forced_decode_token_count": len(
                self.source_forced_decode_token_ids
            ),
            "target_position_policy": {
                "kind": "target-native-contiguous-full-prefix",
                "origin": self.target_position_origin,
            },
            "phase_order": list(self.phase_order),
            "phase_count": len(self.phases),
            "parent_phase_identity_preserved": True,
            "phases": [value.to_evidence_dict() for value in self.phases],
            "expansion": self.expansion_report,
            "model_operations": self.model_operation_audit,
            "runtime": {
                "cached_parent_tokens_only": True,
                "local_tokenizers_only": True,
                "network_access_performed": False,
            },
        }

    def to_evidence_dict(self) -> dict[str, Any]:
        return {**self._body(), "bridge_sha256": self.bridge_sha256}

    def verify(
        self,
        donor_tokenizer: Any,
        target_tokenizer: Any,
    ) -> Lfm2SemanticSpanPhaseBridge:
        if (
            _live_tokenizer_fingerprint(donor_tokenizer, "donor")
            != self.donor_tokenizer_semantics_sha256
            or _live_tokenizer_fingerprint(target_tokenizer, "target")
            != self.target_tokenizer_semantics_sha256
        ):
            raise Lfm2SemanticSpanPhaseBridgeError(
                "Live tokenizer semantics differ from the authenticated bridge."
            )
        rebuilt = _build_phases(
            prompt_ids=self.source_prompt_token_ids,
            forced_ids=self.source_forced_decode_token_ids,
            position_origin=self.target_position_origin,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
        )
        if tuple(value.to_evidence_dict() for value in rebuilt) != tuple(
            value.to_evidence_dict() for value in self.phases
        ):
            raise Lfm2SemanticSpanPhaseBridgeError(
                "Semantic phase schedule no longer matches its tokenizers."
            )
        if canonical_sha256(self._body()) != self.bridge_sha256:
            raise Lfm2SemanticSpanPhaseBridgeError(
                "Semantic-span bridge seal is invalid."
            )
        return self


def build_lfm2_semantic_span_phase_bridge(
    source_batch: Lfm2CircuitTraceBatch,
    source_cache_manifest: Mapping[str, Any],
    donor_tokenizer: Any,
    target_tokenizer: Any,
    *,
    expected_source_cache_key: str,
    expected_source_manifest_sha256: str,
    expected_source_input_sha256: str,
    expected_donor_tokenizer_fingerprint: str,
    expected_target_tokenizer_fingerprint: str,
) -> Lfm2SemanticSpanPhaseBridge:
    """Build a sealed target schedule without executing either model.

    All expected hashes are mandatory out-of-band identities.  The manifest,
    record, tensor inventory, canonical circuit-input digest, corpus ordinal,
    and both live tokenizer semantics must agree before any schedule is
    returned.
    """

    if not isinstance(source_batch, Lfm2CircuitTraceBatch):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "source_batch must be an authenticated Lfm2CircuitTraceBatch."
        )
    cache_key, manifest_sha256, input_sha256 = _validate_source_manifest(
        source_cache_manifest,
        source_batch,
        expected_cache_key=expected_source_cache_key,
        expected_manifest_sha256=expected_source_manifest_sha256,
        expected_input_sha256=expected_source_input_sha256,
    )
    input_ids, _mask, forced_ids, position_ids, record_sha256 = _validated_source_batch(
        source_batch,
        cache_key=cache_key,
        expected_input_sha256=input_sha256,
    )
    donor_fingerprint = _live_tokenizer_fingerprint(donor_tokenizer, "donor")
    target_fingerprint = _live_tokenizer_fingerprint(target_tokenizer, "target")
    if donor_fingerprint != _digest(
        expected_donor_tokenizer_fingerprint,
        "expected donor tokenizer fingerprint",
    ):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Donor tokenizer differs from its expected fingerprint."
        )
    if target_fingerprint != _digest(
        expected_target_tokenizer_fingerprint,
        "expected target tokenizer fingerprint",
    ):
        raise Lfm2SemanticSpanPhaseBridgeError(
            "Target tokenizer differs from its expected fingerprint."
        )
    prompt_tuple = tuple(int(value) for value in input_ids.reshape(-1).tolist())
    forced_tuple = tuple(int(value) for value in forced_ids.reshape(-1).tolist())
    position_origin = (
        0 if position_ids is None else int(position_ids.reshape(-1)[0].item())
    )
    result = Lfm2SemanticSpanPhaseBridge(
        source_cache_key=cache_key,
        source_manifest_sha256=manifest_sha256,
        source_batch_index=source_batch.index,
        source_input_sha256=input_sha256,
        source_record_sha256=record_sha256,
        donor_tokenizer_semantics_sha256=donor_fingerprint,
        target_tokenizer_semantics_sha256=target_fingerprint,
        source_prompt_token_ids=prompt_tuple,
        source_forced_decode_token_ids=forced_tuple,
        target_position_origin=position_origin,
        phases=_build_phases(
            prompt_ids=prompt_tuple,
            forced_ids=forced_tuple,
            position_origin=position_origin,
            donor_tokenizer=donor_tokenizer,
            target_tokenizer=target_tokenizer,
        ),
    )
    return result.verify(donor_tokenizer, target_tokenizer)


__all__ = [
    "Lfm2SemanticSpanPhase",
    "Lfm2SemanticSpanPhaseBridge",
    "Lfm2SemanticSpanPhaseBridgeError",
    "SEMANTIC_SPAN_EXPANSION_SCHEMA",
    "SEMANTIC_SPAN_PHASE_BRIDGE_SCHEMA",
    "SEMANTIC_SPAN_PHASE_SCHEMA",
    "build_lfm2_semantic_span_phase_bridge",
]
