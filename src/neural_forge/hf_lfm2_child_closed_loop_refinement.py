"""Forward-free orchestration for live-child LFM2 structural refinement.

This module joins an immutable parent circuit cache to an in-memory canonical
child replay by exact token-batch identity.  Parent tensors provide source
activations and desired endpoints only.  Every target-side residual, norm
input, operator context, and downstream MLP input is derived from the live
child replay or from an analytically computed candidate endpoint.

No model is called here.  Coordinate projection and norm fitting are injected
callbacks so experiment-specific maps and fitters do not create import cycles.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.hf_closed_form_fit import FitMetrics, measure_fit
from neural_forge.hf_lfm2_circuit_trace_cache import (
    CACHED_VERIFIED_TRACE_MODE,
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2ChildReplay,
    Lfm2ChildReplayBatch,
    Lfm2CircuitTraceBatch,
    Lfm2CircuitTraceCache,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2PairedInputAlignmentEvidence,
    Lfm2ProjectedCachedParentLogits,
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2StrictTokenizerBridgeTraceError,
    authenticate_lfm2_bridge_input_pair,
    project_lfm2_cached_parent_logits,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedAttentionFoldCase,
    Lfm2CachedMlpFoldCase,
    Lfm2CachedShortConvFoldCase,
    cached_attention_cases_from_lfm2_trace_phases,
    cached_short_conv_case_from_lfm2_trace,
)


CHILD_CLOSED_LOOP_REFINEMENT_SCHEMA = (
    "neural-forge-lfm2-child-closed-loop-refinement-v1"
)


class Lfm2ChildClosedLoopRefinementError(ValueError):
    """Parent/child evidence or offline refinement state is inconsistent."""


class Lfm2ChildClosedLoopRollbackError(RuntimeError):
    """A layer could not be restored without changing its runtime contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 child closed-loop refinement."
        ) from exc
    return torch


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2ChildClosedLoopRefinementError(
            "Refinement metadata must be canonical JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Refinement evidence must contain materialized dense tensors."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cpu_tensor(
    value: Any,
    name: str,
    *,
    ndim: int | None = None,
    floating: bool = True,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or (ndim is not None and value.ndim != ndim)
        or any(int(size) <= 0 for size in value.shape)
        or (floating and (not value.is_floating_point() or value.is_complex()))
    ):
        rank = "" if ndim is None else f" rank-{ndim}"
        raise Lfm2ChildClosedLoopRefinementError(
            f"{name} must be a nonempty materialized dense{rank} tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if (result.is_floating_point() or result.is_complex()) and not bool(
        torch.isfinite(result).all().item()
    ):
        raise Lfm2ChildClosedLoopRefinementError(f"{name} contains NaN or infinity.")
    return result


def _rows(value: Any, name: str) -> Any:
    tensor = _cpu_tensor(value, name)
    if tensor.ndim not in {2, 3}:
        raise Lfm2ChildClosedLoopRefinementError(f"{name} must have rank 2 or 3.")
    return tensor.reshape(-1, int(tensor.shape[-1])).contiguous()


def _required(
    tensors: Mapping[str, Any],
    name: str,
) -> Any:
    if name not in tensors:
        raise Lfm2ChildClosedLoopRefinementError(f"Aligned trace is missing {name!r}.")
    return tensors[name]


def _same_tensor(left: Any, right: Any) -> bool:
    torch = _torch()
    return bool(
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left.detach().cpu(), right.detach().cpu())
    )


def _verified_parent_decode_trace_mode(
    batch: Lfm2CircuitTraceBatch,
) -> str:
    """Read the parent trace semantics from its already-verified record.

    Current records state the mode directly.  The sealed 2026-07-16 cache
    predates that field, but cryptographically records successful cached/full
    prefix numerical comparisons for every phase; that exact legacy contract
    is the only permitted compatibility route to ``cached_verified``.  Top-1
    equality is not part of this mode proof because the authenticated legacy
    cache permits numerically close logits to cross an argmax tie.
    """

    record = batch.record
    capture = record.get("capture")
    if not isinstance(capture, Mapping):
        raise Lfm2ChildClosedLoopRefinementError(
            "Verified parent batch has no capture contract."
        )
    mode = capture.get("decode_trace_mode")
    if mode in {
        CACHED_VERIFIED_TRACE_MODE,
        FULL_PREFIX_CANONICAL_TRACE_MODE,
    }:
        return str(mode)
    verifications = capture.get("cached_full_prefix_verifications")
    legacy_cached = (
        mode is None
        and capture.get("all_cached_logits_verified") is True
        and isinstance(verifications, list)
        and bool(verifications)
        and all(
            isinstance(value, Mapping) and value.get("numerically_close") is True
            for value in verifications
        )
    )
    if legacy_cached:
        return CACHED_VERIFIED_TRACE_MODE
    raise Lfm2ChildClosedLoopRefinementError(
        "Verified parent batch does not authenticate a supported decode trace mode."
    )


@dataclass(frozen=True, slots=True)
class Lfm2PairedReplayBatch:
    """One immutable parent batch paired to the exact live-child replay."""

    index: int
    input_sha256: str
    phase_order: tuple[str, ...]
    parent: Lfm2CircuitTraceBatch
    child: Lfm2ChildReplayBatch
    parent_input_sha256: str | None = None
    tokenizer_bridge_record_id: str | None = None
    tokenizer_bridge_input_evidence: Lfm2PairedInputAlignmentEvidence | None = None


@dataclass(frozen=True, slots=True)
class Lfm2PairedReplay:
    """Exact ordered parent/child batch pairing with no model execution."""

    batches: tuple[Lfm2PairedReplayBatch, ...]
    parent_cache_key: str
    parent_identity_sha256: str
    corpus_fingerprint: str
    child_replay_sha256: str
    selected_layer_indices: tuple[int, ...]
    pairing_sha256: str
    tokenizer_bridge_trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter | None = None
    tokenizer_bridge_evidence_sha256: str | None = None

    def __len__(self) -> int:
        return len(self.batches)

    def load_batch(self, index: int) -> Lfm2PairedReplayBatch:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("batch index must be an integer.")
        if index < 0 or index >= len(self.batches):
            raise IndexError(
                f"batch index must be in [0, {len(self.batches)}), got {index}."
            )
        return self.batches[index]


def pair_lfm2_parent_child_replay(
    parent_cache: Lfm2CircuitTraceCache,
    child_replay: Lfm2ChildReplay,
    *,
    tokenizer_bridge_trace_adapter: (
        Lfm2StrictTokenizerBridgeTraceAdapter | None
    ) = None,
) -> Lfm2PairedReplay:
    """Pair sealed parent and live-child batches by authenticated identity.

    With no bridge adapter this retains the original exact token-ID, input-hash,
    and corpus-identity contract.  A supplied strict bridge instead proves that
    every target-token child batch maps to its donor-token parent cache batch;
    raw token-ID equality is neither expected nor accepted as a substitute.
    """

    if not isinstance(parent_cache, Lfm2CircuitTraceCache):
        raise TypeError("parent_cache must be Lfm2CircuitTraceCache.")
    if not isinstance(child_replay, Lfm2ChildReplay):
        raise TypeError("child_replay must be Lfm2ChildReplay.")
    if len(parent_cache) != len(child_replay):
        raise Lfm2ChildClosedLoopRefinementError(
            "Parent cache and child replay batch counts differ."
        )
    if tokenizer_bridge_trace_adapter is None and (
        parent_cache.corpus_fingerprint != child_replay.corpus_fingerprint
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Parent cache and child replay corpus identities differ."
        )
    if tokenizer_bridge_trace_adapter is not None:
        if not isinstance(
            tokenizer_bridge_trace_adapter,
            Lfm2StrictTokenizerBridgeTraceAdapter,
        ):
            raise TypeError(
                "tokenizer_bridge_trace_adapter must be "
                "Lfm2StrictTokenizerBridgeTraceAdapter."
            )
        try:
            tokenizer_bridge_trace_adapter.verify()
        except Lfm2StrictTokenizerBridgeTraceError as exc:
            raise Lfm2ChildClosedLoopRefinementError(
                "Tokenizer bridge trace adapter failed authentication."
            ) from exc
        if tokenizer_bridge_trace_adapter.record_count != len(parent_cache):
            raise Lfm2ChildClosedLoopRefinementError(
                "Tokenizer bridge record count differs from the paired replay."
            )

    replay_capture_mode = child_replay.metadata.get("selection", {}).get(
        "replay_capture_mode"
    )
    expected_trace_sources = {
        CHILD_REPLAY_PREFIX_BY_PREFIX_MODE: "use_cache_false_full_prefix",
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE: (
            "use_cache_false_full_trajectory_slices"
        ),
    }
    expected_trace_source = expected_trace_sources.get(replay_capture_mode)
    if expected_trace_source is None:
        raise Lfm2ChildClosedLoopRefinementError(
            "Child evidence has an unsupported replay capture mode."
        )

    paired: list[Lfm2PairedReplayBatch] = []
    bridge_pair_evidence: list[Lfm2PairedInputAlignmentEvidence] = []
    selected_layers = child_replay.selected_layer_indices
    for index in range(len(parent_cache)):
        parent = parent_cache.load_batch(index)
        child = child_replay.load_batch(index)
        bridge_record_id = None
        bridge_evidence = None
        if tokenizer_bridge_trace_adapter is None:
            if (
                parent.index != index
                or child.index != index
                or parent.input_sha256 != child.input_sha256
            ):
                raise Lfm2ChildClosedLoopRefinementError(
                    f"Parent/child replay order or input hash differs at batch {index}."
                )
            for field in (
                "input.input_ids",
                "input.attention_mask",
                "input.forced_decode_ids",
            ):
                if not _same_tensor(
                    _required(parent.tensors, field),
                    _required(child.tensors, field),
                ):
                    raise Lfm2ChildClosedLoopRefinementError(
                        f"Parent/child exact replay field {field!r} differs at "
                        f"batch {index}."
                    )
            parent_position = parent.tensors.get("input.position_ids")
            child_position = child.tensors.get("input.position_ids")
            if (parent_position is None) != (child_position is None) or (
                parent_position is not None
                and not _same_tensor(parent_position, child_position)
            ):
                raise Lfm2ChildClosedLoopRefinementError(
                    f"Parent/child position IDs differ at batch {index}."
                )
        else:
            if parent.index != index or child.index != index:
                raise Lfm2ChildClosedLoopRefinementError(
                    f"Parent/child replay order differs at batch {index}."
                )
            aligned = tokenizer_bridge_trace_adapter.records[index]
            if (
                aligned.ordinal != index
                or parent.input_sha256 != aligned.donor_cache_input_sha256
                or child.input_sha256 != aligned.target_cache_input_sha256
            ):
                raise Lfm2ChildClosedLoopRefinementError(
                    "Parent/child replay hashes do not match the authenticated "
                    f"tokenizer bridge record at batch {index}."
                )
            parent_input = {
                "input_ids": _required(parent.tensors, "input.input_ids"),
                "attention_mask": _required(parent.tensors, "input.attention_mask"),
                "forced_decode_ids": _required(
                    parent.tensors, "input.forced_decode_ids"
                ),
            }
            child_input = {
                "input_ids": _required(child.tensors, "input.input_ids"),
                "attention_mask": _required(child.tensors, "input.attention_mask"),
                "forced_decode_ids": _required(
                    child.tensors, "input.forced_decode_ids"
                ),
            }
            parent_position = parent.tensors.get("input.position_ids")
            child_position = child.tensors.get("input.position_ids")
            if parent_position is not None:
                parent_input["position_ids"] = parent_position
            if child_position is not None:
                child_input["position_ids"] = child_position
            try:
                bridge_evidence = authenticate_lfm2_bridge_input_pair(
                    tokenizer_bridge_trace_adapter,
                    record_id=aligned.record_id,
                    parent_cache_batch=parent_input,
                    child_replay_batch=child_input,
                )
            except Lfm2StrictTokenizerBridgeTraceError as exc:
                raise Lfm2ChildClosedLoopRefinementError(
                    "Parent/child replay does not satisfy its authenticated "
                    f"tokenizer bridge at batch {index}."
                ) from exc
            bridge_record_id = aligned.record_id
            bridge_pair_evidence.append(bridge_evidence)
        capture = child.capture
        if (
            capture.get("decode_trace_mode") != FULL_PREFIX_CANONICAL_TRACE_MODE
            or capture.get("canonical_trace_source") != expected_trace_source
            or capture.get("child_replay_capture_mode") != replay_capture_mode
            or capture.get("cached_forward_count") != 0
        ):
            raise Lfm2ChildClosedLoopRefinementError(
                "Child evidence is not a canonical causal-phase replay."
            )
        phases = child.phase_order
        for phase in phases:
            for layer_index in selected_layers:
                base = f"{phase}.layer.{layer_index:03d}."
                _required(parent.tensors, base + "residual.pre")
                _required(child.tensors, base + "residual.pre")
        paired.append(
            Lfm2PairedReplayBatch(
                index=index,
                input_sha256=child.input_sha256,
                phase_order=phases,
                parent=parent,
                child=child,
                parent_input_sha256=(
                    None
                    if tokenizer_bridge_trace_adapter is None
                    else parent.input_sha256
                ),
                tokenizer_bridge_record_id=bridge_record_id,
                tokenizer_bridge_input_evidence=bridge_evidence,
            )
        )

    if tokenizer_bridge_trace_adapter is None:
        paired_corpus_fingerprint = parent_cache.corpus_fingerprint
        identity = {
            "schema": CHILD_CLOSED_LOOP_REFINEMENT_SCHEMA,
            "parent_cache_key": parent_cache.cache_key,
            "parent_identity_sha256": parent_cache.parent_fingerprint,
            "corpus_fingerprint": parent_cache.corpus_fingerprint,
            "child_replay_sha256": child_replay.replay_sha256,
            "replay_capture_mode": replay_capture_mode,
            "selected_layer_indices": list(selected_layers),
            "ordered_batches": [
                {
                    "index": batch.index,
                    "input_sha256": batch.input_sha256,
                    "phase_order": list(batch.phase_order),
                }
                for batch in paired
            ],
        }
        bridge_evidence_sha256 = None
    else:
        bridge_evidence_sha256 = _canonical_sha256(
            {
                "adapter_evidence_sha256": (
                    tokenizer_bridge_trace_adapter.evidence_sha256
                ),
                "paired_input_evidence_sha256": [
                    value.evidence_sha256 for value in bridge_pair_evidence
                ],
            }
        )
        paired_corpus_fingerprint = _canonical_sha256(
            {
                "schema": "neural-forge-lfm2-bridged-paired-corpus-v1",
                "parent_corpus_fingerprint": parent_cache.corpus_fingerprint,
                "child_corpus_fingerprint": child_replay.corpus_fingerprint,
                "tokenizer_bridge_evidence_sha256": bridge_evidence_sha256,
            }
        )
        identity = {
            "schema": CHILD_CLOSED_LOOP_REFINEMENT_SCHEMA,
            "parent_cache_key": parent_cache.cache_key,
            "parent_identity_sha256": parent_cache.parent_fingerprint,
            "corpus_fingerprint": paired_corpus_fingerprint,
            "parent_corpus_fingerprint": parent_cache.corpus_fingerprint,
            "child_corpus_fingerprint": child_replay.corpus_fingerprint,
            "child_replay_sha256": child_replay.replay_sha256,
            "replay_capture_mode": replay_capture_mode,
            "selected_layer_indices": list(selected_layers),
            "tokenizer_bridge": {
                "adapter_evidence_sha256": (
                    tokenizer_bridge_trace_adapter.evidence_sha256
                ),
                "bridge_artifact_sha256": (
                    tokenizer_bridge_trace_adapter.bridge_artifact_sha256
                ),
                "transport_profile_fingerprint": (
                    tokenizer_bridge_trace_adapter.transport_profile_fingerprint
                ),
                "pair_evidence_sha256": bridge_evidence_sha256,
                "numeric_token_id_equality_required": False,
            },
            "ordered_batches": [
                {
                    "index": batch.index,
                    "parent_input_sha256": batch.parent_input_sha256,
                    "target_input_sha256": batch.input_sha256,
                    "record_id": batch.tokenizer_bridge_record_id,
                    "input_alignment_evidence_sha256": (
                        batch.tokenizer_bridge_input_evidence.evidence_sha256
                    ),
                    "phase_order": list(batch.phase_order),
                }
                for batch in paired
            ],
        }
    return Lfm2PairedReplay(
        batches=tuple(paired),
        parent_cache_key=parent_cache.cache_key,
        parent_identity_sha256=parent_cache.parent_fingerprint,
        corpus_fingerprint=paired_corpus_fingerprint,
        child_replay_sha256=child_replay.replay_sha256,
        selected_layer_indices=selected_layers,
        pairing_sha256=_canonical_sha256(identity),
        tokenizer_bridge_trace_adapter=tokenizer_bridge_trace_adapter,
        tokenizer_bridge_evidence_sha256=bridge_evidence_sha256,
    )


def project_lfm2_paired_parent_logits(
    paired: Lfm2PairedReplay,
    *,
    batch_index: int,
    phase: str,
    top_k: int = 5,
) -> Lfm2ProjectedCachedParentLogits:
    """Project one paired donor-logit phase into the child vocabulary.

    The bridge, record ordinal, cache input hash, cache key, and phase tensor are
    taken from the already-authenticated pairing.  This helper performs no
    model call and leaves the sealed donor logits unchanged.
    """

    if not isinstance(paired, Lfm2PairedReplay):
        raise TypeError("paired must be Lfm2PairedReplay.")
    adapter = paired.tokenizer_bridge_trace_adapter
    if adapter is None or paired.tokenizer_bridge_evidence_sha256 is None:
        raise Lfm2ChildClosedLoopRefinementError(
            "Paired replay has no authenticated tokenizer bridge."
        )
    expected_pair_evidence_sha256 = _canonical_sha256(
        {
            "adapter_evidence_sha256": adapter.evidence_sha256,
            "paired_input_evidence_sha256": [
                batch.tokenizer_bridge_input_evidence.evidence_sha256
                for batch in paired.batches
                if batch.tokenizer_bridge_input_evidence is not None
            ],
        }
    )
    if expected_pair_evidence_sha256 != paired.tokenizer_bridge_evidence_sha256:
        raise Lfm2ChildClosedLoopRefinementError(
            "Paired replay tokenizer bridge evidence seal is inconsistent."
        )
    batch = paired.load_batch(batch_index)
    record_id = batch.tokenizer_bridge_record_id
    evidence = batch.tokenizer_bridge_input_evidence
    if record_id is None or evidence is None:
        raise Lfm2ChildClosedLoopRefinementError(
            "Paired replay batch has no tokenizer bridge evidence."
        )
    if (
        evidence.record_id != record_id
        or evidence.donor_cache_input_sha256 != batch.parent.input_sha256
        or evidence.target_cache_input_sha256 != batch.child.input_sha256
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Paired replay batch changed after tokenizer bridge authentication."
        )
    if not isinstance(phase, str) or phase not in batch.phase_order:
        raise Lfm2ChildClosedLoopRefinementError(
            "phase must name an authenticated paired replay phase."
        )
    donor_logits = _required(batch.parent.tensors, f"{phase}.logits")
    try:
        return project_lfm2_cached_parent_logits(
            adapter,
            record_id=record_id,
            parent_cache_input_sha256=batch.parent.input_sha256,
            cache_key=paired.parent_cache_key,
            cache_batch_index=batch.index,
            phase=phase,
            donor_logits=donor_logits,
            top_k=top_k,
        )
    except Lfm2StrictTokenizerBridgeTraceError as exc:
        raise Lfm2ChildClosedLoopRefinementError(
            "Paired parent logits failed authenticated tokenizer projection."
        ) from exc


@dataclass(frozen=True, slots=True)
class Lfm2WholeBatchSplit:
    """Deterministic disjoint split whose unit is an entire replay batch."""

    paired: Lfm2PairedReplay
    fit_batch_indices: tuple[int, ...]
    validation_batch_indices: tuple[int, ...]
    split_sha256: str

    @property
    def fit_batches(self) -> tuple[Lfm2PairedReplayBatch, ...]:
        return tuple(self.paired.load_batch(index) for index in self.fit_batch_indices)

    @property
    def validation_batches(self) -> tuple[Lfm2PairedReplayBatch, ...]:
        return tuple(
            self.paired.load_batch(index) for index in self.validation_batch_indices
        )


def split_lfm2_replay_batches(
    paired: Lfm2PairedReplay,
    *,
    validation_batch_indices: Sequence[int] = (),
) -> Lfm2WholeBatchSplit:
    """Hold out complete batches; by default every batch is used for fitting."""

    if not isinstance(paired, Lfm2PairedReplay):
        raise TypeError("paired must be Lfm2PairedReplay.")
    if isinstance(validation_batch_indices, (str, bytes)):
        raise TypeError("validation_batch_indices must contain integers.")
    try:
        validation = tuple(validation_batch_indices)
    except TypeError as exc:
        raise TypeError("validation_batch_indices must contain integers.") from exc
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(paired)
        for index in validation
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "validation_batch_indices contains an invalid batch index."
        )
    if len(set(validation)) != len(validation):
        raise Lfm2ChildClosedLoopRefinementError(
            "validation_batch_indices must not contain duplicates."
        )
    validation = tuple(sorted(validation))
    validation_set = frozenset(validation)
    fit = tuple(index for index in range(len(paired)) if index not in validation_set)
    if not fit:
        raise Lfm2ChildClosedLoopRefinementError(
            "At least one complete batch must remain in the fit split."
        )
    body = {
        "schema": "neural-forge-lfm2-whole-batch-split-v1",
        "pairing_sha256": paired.pairing_sha256,
        "fit_batch_indices": list(fit),
        "validation_batch_indices": list(validation),
        "split_unit": "whole_replay_batch",
    }
    return Lfm2WholeBatchSplit(
        paired=paired,
        fit_batch_indices=fit,
        validation_batch_indices=validation,
        split_sha256=_canonical_sha256(body),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CaseOrigin:
    """Stable location of one phase-level case in an exact replay batch."""

    split: str
    batch_index: int
    input_sha256: str
    phase: str
    prefix_length: int
    sample_start: int
    sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "batch_index": self.batch_index,
            "input_sha256": self.input_sha256,
            "phase": self.phase,
            "prefix_length": self.prefix_length,
            "sample_start": self.sample_start,
            "sample_count": self.sample_count,
        }


def _origin(
    batch: Lfm2PairedReplayBatch,
    phase: str,
    split: str,
) -> Lfm2CaseOrigin:
    rows = batch.child.record.get("phase_rows")
    if not isinstance(rows, list):
        raise Lfm2ChildClosedLoopRefinementError(
            "Child replay has no deterministic phase-row metadata."
        )
    matches = [value for value in rows if value.get("phase") == phase]
    if len(matches) != 1:
        raise Lfm2ChildClosedLoopRefinementError(
            f"Child replay phase metadata is ambiguous for {phase!r}."
        )
    row = matches[0]
    return Lfm2CaseOrigin(
        split=split,
        batch_index=batch.index,
        input_sha256=batch.input_sha256,
        phase=phase,
        prefix_length=int(row["prefix_length"]),
        sample_start=int(row["sample_start"]),
        sample_count=int(row["sample_count"]),
    )


@dataclass(frozen=True, slots=True)
class Lfm2NormFitRequest:
    """Target-side norm fit request over live residual inputs."""

    role: str
    fit_inputs: tuple[Any, ...]
    fit_targets: tuple[Any, ...]
    validation_inputs: tuple[Any, ...]
    validation_targets: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class Lfm2NormMaterializer:
    """Injected fitted norm and its serializable audit report."""

    role: str
    materialize_callback: Callable[[Any], Any]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise Lfm2ChildClosedLoopRefinementError(
                "Norm materializer role must be nonempty."
            )
        if not callable(self.materialize_callback):
            raise TypeError("materialize_callback must be callable.")
        if not isinstance(self.report, Mapping):
            raise TypeError("report must be a mapping.")
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    def materialize(self, residual: Any, name: str) -> Any:
        source = _cpu_tensor(residual, f"{name}.residual")
        before = _tensor_sha256(source)
        result = self.materialize_callback(source)
        if _tensor_sha256(source) != before:
            raise Lfm2ChildClosedLoopRefinementError(
                f"{self.role} norm callback mutated its residual input."
            )
        materialized = _cpu_tensor(result, name)
        if tuple(materialized.shape) != tuple(source.shape):
            raise Lfm2ChildClosedLoopRefinementError(
                f"{self.role} norm callback changed the residual shape."
            )
        return materialized


NormFitCallback = Callable[[Lfm2NormFitRequest], Lfm2NormMaterializer]
CoordinateMapCallback = Callable[[Any], Any]


def _project_parent(
    callback: CoordinateMapCallback,
    value: Any,
    *,
    target_shape: tuple[int, ...],
    name: str,
) -> Any:
    if not callable(callback):
        raise TypeError("coordinate_map_callback must be callable.")
    source = _cpu_tensor(value, f"{name}.source")
    source_before = _tensor_sha256(source)
    projected = callback(source)
    if _tensor_sha256(source) != source_before:
        raise Lfm2ChildClosedLoopRefinementError(
            "Coordinate-map callback mutated its parent input."
        )
    result = _cpu_tensor(projected, name)
    if tuple(result.shape) != target_shape:
        raise Lfm2ChildClosedLoopRefinementError(
            f"{name} projected shape {tuple(result.shape)} does not match "
            f"live-child shape {target_shape}."
        )
    return result


def _fit_norm(
    callback: NormFitCallback,
    request: Lfm2NormFitRequest,
) -> Lfm2NormMaterializer:
    if not callable(callback):
        raise TypeError("norm_fit_callback must be callable.")
    before = tuple(
        _tensor_sha256(value)
        for values in (
            request.fit_inputs,
            request.fit_targets,
            request.validation_inputs,
            request.validation_targets,
        )
        for value in values
    )
    result = callback(request)
    after = tuple(
        _tensor_sha256(value)
        for values in (
            request.fit_inputs,
            request.fit_targets,
            request.validation_inputs,
            request.validation_targets,
        )
        for value in values
    )
    if after != before:
        raise Lfm2ChildClosedLoopRefinementError(
            f"{request.role} norm fitter mutated its evidence."
        )
    if not isinstance(result, Lfm2NormMaterializer):
        raise TypeError("norm_fit_callback must return Lfm2NormMaterializer.")
    if result.role != request.role:
        raise Lfm2ChildClosedLoopRefinementError(
            "Norm materializer role does not match its fit request."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2OperatorProducerCases:
    """Source operator evidence driven by actual live-child target inputs."""

    layer_index: int
    operator_kind: str
    fit_cases: tuple[Lfm2CachedAttentionFoldCase | Lfm2CachedShortConvFoldCase, ...]
    validation_cases: tuple[
        Lfm2CachedAttentionFoldCase | Lfm2CachedShortConvFoldCase, ...
    ]
    fit_origins: tuple[Lfm2CaseOrigin, ...]
    validation_origins: tuple[Lfm2CaseOrigin, ...]
    norm_materializer: Lfm2NormMaterializer
    split_sha256: str
    parent_cache_key: str
    child_replay_sha256: str

    @property
    def all_cases(
        self,
    ) -> tuple[Lfm2CachedAttentionFoldCase | Lfm2CachedShortConvFoldCase, ...]:
        return self.fit_cases + self.validation_cases

    @property
    def all_origins(self) -> tuple[Lfm2CaseOrigin, ...]:
        return self.fit_origins + self.validation_origins


def _split_name(
    split: Lfm2WholeBatchSplit,
    batch_index: int,
) -> str:
    return (
        "validation"
        if batch_index in frozenset(split.validation_batch_indices)
        else "fit"
    )


def build_lfm2_operator_producer_cases(
    split: Lfm2WholeBatchSplit,
    *,
    layer_index: int,
    operator_kind: str,
    coordinate_map_callback: CoordinateMapCallback,
    norm_fit_callback: NormFitCallback,
) -> Lfm2OperatorProducerCases:
    """Build operator producer cases from live-child residuals, never parent inputs."""

    if not isinstance(split, Lfm2WholeBatchSplit):
        raise TypeError("split must be Lfm2WholeBatchSplit.")
    if (
        isinstance(layer_index, bool)
        or not isinstance(layer_index, int)
        or layer_index < 0
        or layer_index not in split.paired.selected_layer_indices
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "layer_index must be a selected live-child replay layer."
        )
    if operator_kind not in {"full_attention", "short_conv"}:
        raise Lfm2ChildClosedLoopRefinementError(
            "operator_kind must be 'full_attention' or 'short_conv'."
        )

    fit_inputs: list[Any] = []
    fit_targets: list[Any] = []
    validation_inputs: list[Any] = []
    validation_targets: list[Any] = []
    projected_norm_targets: dict[tuple[int, str], Any] = {}
    for batch in split.paired.batches:
        bucket_inputs = (
            validation_inputs
            if batch.index in split.validation_batch_indices
            else fit_inputs
        )
        bucket_targets = (
            validation_targets
            if batch.index in split.validation_batch_indices
            else fit_targets
        )
        for phase in batch.phase_order:
            base = f"{phase}.layer.{layer_index:03d}."
            child_residual = _cpu_tensor(
                _required(batch.child.tensors, base + "residual.pre"),
                base + "child.residual.pre",
            )
            projected = _project_parent(
                coordinate_map_callback,
                _required(batch.parent.tensors, base + "operator.input"),
                target_shape=tuple(child_residual.shape),
                name=base + "projected_parent.operator.input",
            )
            projected_norm_targets[(batch.index, phase)] = projected
            bucket_inputs.append(child_residual)
            bucket_targets.append(projected)
    request = Lfm2NormFitRequest(
        role=f"layer.{layer_index:03d}.operator_norm",
        fit_inputs=tuple(fit_inputs),
        fit_targets=tuple(fit_targets),
        validation_inputs=tuple(validation_inputs),
        validation_targets=tuple(validation_targets),
    )
    materializer = _fit_norm(norm_fit_callback, request)

    fit_cases: list[Any] = []
    validation_cases: list[Any] = []
    fit_origins: list[Lfm2CaseOrigin] = []
    validation_origins: list[Lfm2CaseOrigin] = []
    for batch in split.paired.batches:
        split_name = _split_name(split, batch.index)
        target_tensors: dict[str, Any] = {}
        endpoints: dict[str, Any] = {}
        origins = [_origin(batch, phase, split_name) for phase in batch.phase_order]
        for phase, origin in zip(batch.phase_order, origins):
            base = f"{phase}.layer.{layer_index:03d}."
            residual = _cpu_tensor(
                _required(batch.child.tensors, base + "residual.pre"),
                base + "child.residual.pre",
            )
            target_tensors[base + "residual.pre"] = residual
            target_tensors[base + "operator.input"] = materializer.materialize(
                residual,
                base + "materialized_child.operator.input",
            )
            endpoints[phase] = _project_parent(
                coordinate_map_callback,
                _required(
                    batch.parent.tensors,
                    base + "residual.post_operator",
                ),
                target_shape=tuple(residual.shape),
                name=base + "projected_parent.residual.post_operator",
            )
            if tuple(projected_norm_targets[(batch.index, phase)].shape) != tuple(
                target_tensors[base + "operator.input"].shape
            ):
                raise Lfm2ChildClosedLoopRefinementError(
                    "Fitted operator norm materialization changed target topology."
                )
        if operator_kind == "full_attention":
            parent_decode_trace_mode = _verified_parent_decode_trace_mode(batch.parent)
            cases = cached_attention_cases_from_lfm2_trace_phases(
                batch.parent.tensors,
                phases=batch.phase_order,
                layer_index=layer_index,
                decode_trace_mode=parent_decode_trace_mode,
                target_tensors=target_tensors,
                parent_post_attention_residual_target_space_by_phase=endpoints,
            )
        else:
            cases = tuple(
                cached_short_conv_case_from_lfm2_trace(
                    batch.parent.tensors,
                    phase=phase,
                    layer_index=layer_index,
                    target_input=target_tensors[
                        f"{phase}.layer.{layer_index:03d}.operator.input"
                    ],
                    parent_post_operator_residual_target_space=endpoints[phase],
                    target_pre_operator_residual=target_tensors[
                        f"{phase}.layer.{layer_index:03d}.residual.pre"
                    ],
                )
                for phase in batch.phase_order
            )
        if len(cases) != len(origins):
            raise Lfm2ChildClosedLoopRefinementError(
                "Operator adapter changed phase/case cardinality."
            )
        if split_name == "fit":
            fit_cases.extend(cases)
            fit_origins.extend(origins)
        else:
            validation_cases.extend(cases)
            validation_origins.extend(origins)
    return Lfm2OperatorProducerCases(
        layer_index=layer_index,
        operator_kind=operator_kind,
        fit_cases=tuple(fit_cases),
        validation_cases=tuple(validation_cases),
        fit_origins=tuple(fit_origins),
        validation_origins=tuple(validation_origins),
        norm_materializer=materializer,
        split_sha256=split.split_sha256,
        parent_cache_key=split.paired.parent_cache_key,
        child_replay_sha256=split.paired.child_replay_sha256,
    )


@dataclass(frozen=True, slots=True)
class Lfm2AlignedChildContexts:
    """Actual full-prefix child contexts aligned one-to-one with operator cases."""

    layer_index: int
    operator_kind: str
    fit_contexts: tuple[Any, ...]
    validation_contexts: tuple[Any, ...]
    fit_current_post_operator: tuple[Any, ...]
    validation_current_post_operator: tuple[Any, ...]
    fit_origins: tuple[Lfm2CaseOrigin, ...]
    validation_origins: tuple[Lfm2CaseOrigin, ...]
    child_replay_sha256: str
    canonical_full_prefix: bool = True


def _aligned_context(
    paired: Lfm2PairedReplay,
    origin: Lfm2CaseOrigin,
    *,
    layer_index: int,
    operator_kind: str,
) -> tuple[Any, Any, Any]:
    batch = paired.load_batch(origin.batch_index)
    if (
        batch.input_sha256 != origin.input_sha256
        or origin.phase not in batch.phase_order
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Recaptured child batch no longer matches its case origin."
        )
    fresh_origin = _origin(batch, origin.phase, origin.split)
    if fresh_origin != origin:
        raise Lfm2ChildClosedLoopRefinementError(
            "Recaptured child phase rows no longer align with the cases."
        )
    base = f"{origin.phase}.layer.{layer_index:03d}."
    suffix = (
        "attention.context"
        if operator_kind == "full_attention"
        else "short_conv.context"
    )
    context = _cpu_tensor(
        _required(batch.child.tensors, base + suffix),
        base + "actual_child." + suffix,
    )
    current = _cpu_tensor(
        _required(batch.child.tensors, base + "residual.post_operator"),
        base + "actual_child.residual.post_operator",
    )
    residual_pre = _cpu_tensor(
        _required(batch.child.tensors, base + "residual.pre"),
        base + "actual_child.residual.pre",
    )
    query_axis = 2 if context.ndim == 4 else 1
    if int(context.shape[query_axis]) != origin.sample_count:
        raise Lfm2ChildClosedLoopRefinementError(
            "Actual child context rows do not match phase sample metadata."
        )
    return context, current, residual_pre


def extract_lfm2_actual_child_contexts(
    cases: Lfm2OperatorProducerCases,
    producer_replay: Lfm2PairedReplay,
) -> Lfm2AlignedChildContexts:
    """Extract installed-producer contexts using the original exact case order."""

    if not isinstance(cases, Lfm2OperatorProducerCases):
        raise TypeError("cases must be Lfm2OperatorProducerCases.")
    if not isinstance(producer_replay, Lfm2PairedReplay):
        raise TypeError("producer_replay must be Lfm2PairedReplay.")
    if (
        producer_replay.parent_cache_key != cases.parent_cache_key
        or cases.layer_index not in producer_replay.selected_layer_indices
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Producer replay is not aligned to the operator case parent/layer."
        )
    fit = [
        _aligned_context(
            producer_replay,
            origin,
            layer_index=cases.layer_index,
            operator_kind=cases.operator_kind,
        )
        for origin in cases.fit_origins
    ]
    validation = [
        _aligned_context(
            producer_replay,
            origin,
            layer_index=cases.layer_index,
            operator_kind=cases.operator_kind,
        )
        for origin in cases.validation_origins
    ]
    for case, values in zip(cases.fit_cases, fit):
        expected = (
            case.target_pre_attention_residual
            if cases.operator_kind == "full_attention"
            else case.target_pre_operator_residual
        )
        if not _same_tensor(expected, values[2]):
            raise Lfm2ChildClosedLoopRefinementError(
                "Installed-producer replay changed the selected layer's "
                "pre-operator residual."
            )
    for case, values in zip(cases.validation_cases, validation):
        expected = (
            case.target_pre_attention_residual
            if cases.operator_kind == "full_attention"
            else case.target_pre_operator_residual
        )
        if not _same_tensor(expected, values[2]):
            raise Lfm2ChildClosedLoopRefinementError(
                "Installed-producer validation replay changed the selected "
                "layer's pre-operator residual."
            )
    return Lfm2AlignedChildContexts(
        layer_index=cases.layer_index,
        operator_kind=cases.operator_kind,
        fit_contexts=tuple(value[0] for value in fit),
        validation_contexts=tuple(value[0] for value in validation),
        fit_current_post_operator=tuple(value[1] for value in fit),
        validation_current_post_operator=tuple(value[1] for value in validation),
        fit_origins=cases.fit_origins,
        validation_origins=cases.validation_origins,
        child_replay_sha256=producer_replay.child_replay_sha256,
    )


def _metrics_dict(value: FitMetrics | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "rows": value.rows,
        "output_width": value.output_width,
        "rmse": value.rmse,
        "target_rms": value.target_rms,
        "relative_rmse": value.relative_rmse,
        "mean_cosine_similarity": value.mean_cosine_similarity,
        "finite": value.finite,
    }


@dataclass(frozen=True, slots=True)
class Lfm2OfflineEndpointComparison:
    """Candidate and current-child errors against the same parent endpoint."""

    candidate_fit: FitMetrics
    current_fit: FitMetrics
    candidate_validation: FitMetrics | None
    current_validation: FitMetrics | None

    @property
    def fit_relative_rmse_improvement(self) -> float:
        return self.current_fit.relative_rmse - self.candidate_fit.relative_rmse

    @property
    def validation_relative_rmse_improvement(self) -> float | None:
        if self.candidate_validation is None or self.current_validation is None:
            return None
        return (
            self.current_validation.relative_rmse
            - self.candidate_validation.relative_rmse
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_fit": _metrics_dict(self.candidate_fit),
            "current_fit": _metrics_dict(self.current_fit),
            "candidate_validation": _metrics_dict(self.candidate_validation),
            "current_validation": _metrics_dict(self.current_validation),
            "fit_relative_rmse_improvement": (self.fit_relative_rmse_improvement),
            "validation_relative_rmse_improvement": (
                self.validation_relative_rmse_improvement
            ),
            "offline_only": True,
            "model_forward_calls": 0,
        }


def _measure_set(
    predictions: Sequence[Any],
    currents: Sequence[Any],
    targets: Sequence[Any],
) -> tuple[FitMetrics, FitMetrics]:
    torch = _torch()
    if not predictions or not (len(predictions) == len(currents) == len(targets)):
        raise Lfm2ChildClosedLoopRefinementError(
            "Endpoint metrics require aligned nonempty tensors."
        )
    prediction_rows = torch.cat(
        [_rows(value, "candidate endpoint") for value in predictions],
        dim=0,
    ).to(torch.float64)
    current_rows = torch.cat(
        [_rows(value, "current child endpoint") for value in currents],
        dim=0,
    ).to(torch.float64)
    target_rows = torch.cat(
        [_rows(value, "desired parent endpoint") for value in targets],
        dim=0,
    ).to(torch.float64)
    return (
        measure_fit(prediction_rows, target_rows),
        measure_fit(current_rows, target_rows),
    )


def _optional_measure_set(
    predictions: Sequence[Any],
    currents: Sequence[Any],
    targets: Sequence[Any],
) -> tuple[FitMetrics | None, FitMetrics | None]:
    if not predictions and not currents and not targets:
        return None, None
    return _measure_set(predictions, currents, targets)


@dataclass(frozen=True, slots=True)
class Lfm2CandidatePostOperator:
    """Analytical post-operator residual from actual installed-child contexts."""

    fit_values: tuple[Any, ...]
    validation_values: tuple[Any, ...]
    fit_origins: tuple[Lfm2CaseOrigin, ...]
    validation_origins: tuple[Lfm2CaseOrigin, ...]
    comparison: Lfm2OfflineEndpointComparison
    child_replay_sha256: str


def _context_rows_for_case(
    context: Any,
    case: Lfm2CachedAttentionFoldCase | Lfm2CachedShortConvFoldCase,
    operator_kind: str,
) -> Any:
    tensor = _cpu_tensor(context, "actual child operator context")
    if operator_kind == "full_attention":
        endpoint = _cpu_tensor(
            case.parent_post_attention_residual_target_space,
            "attention endpoint",
            ndim=3,
        )
        batch, sequence = int(endpoint.shape[0]), int(endpoint.shape[1])
        if tensor.ndim == 4:
            if int(tensor.shape[0]) != batch or int(tensor.shape[2]) != sequence:
                raise Lfm2ChildClosedLoopRefinementError(
                    "Attention context is not aligned to its endpoint."
                )
            return tensor.permute(0, 2, 1, 3).reshape(batch, sequence, -1)
        if tensor.ndim == 3 and tuple(tensor.shape[:2]) == (
            batch,
            sequence,
        ):
            return tensor
        raise Lfm2ChildClosedLoopRefinementError(
            "Attention context must be [B,H,Q,D] or [B,Q,C]."
        )
    endpoint = _cpu_tensor(
        case.parent_post_operator_residual_target_space,
        "short-conv endpoint",
        ndim=3,
    )
    if tensor.ndim != 3 or tuple(tensor.shape[:2]) != tuple(endpoint.shape[:2]):
        raise Lfm2ChildClosedLoopRefinementError(
            "Short-conv context is not aligned to its endpoint."
        )
    return tensor


def _operator_candidate_values(
    cases: Sequence[Lfm2CachedAttentionFoldCase | Lfm2CachedShortConvFoldCase],
    contexts: Sequence[Any],
    *,
    operator_kind: str,
    output_weight: Any,
    output_bias: Any | None,
) -> tuple[Any, ...]:
    torch = _torch()
    weight = _cpu_tensor(
        output_weight,
        "operator output weight",
        ndim=2,
    ).to(torch.float64)
    bias = (
        None
        if output_bias is None
        else _cpu_tensor(
            output_bias,
            "operator output bias",
            ndim=1,
        ).to(torch.float64)
    )
    if len(cases) != len(contexts):
        raise Lfm2ChildClosedLoopRefinementError(
            "Operator cases and actual contexts are not one-to-one."
        )
    result: list[Any] = []
    for case, context in zip(cases, contexts):
        design = _context_rows_for_case(context, case, operator_kind).to(torch.float64)
        if int(design.shape[-1]) != int(weight.shape[1]):
            raise Lfm2ChildClosedLoopRefinementError(
                "Actual child context width does not match output weight."
            )
        if operator_kind == "full_attention":
            residual = _cpu_tensor(
                case.target_pre_attention_residual,
                "actual child pre-attention residual",
                ndim=3,
            ).to(torch.float64)
        else:
            residual = _cpu_tensor(
                case.target_pre_operator_residual,
                "actual child pre-short-conv residual",
                ndim=3,
            ).to(torch.float64)
        delta = design @ weight.T
        if bias is not None:
            if tuple(bias.shape) != (int(weight.shape[0]),):
                raise Lfm2ChildClosedLoopRefinementError(
                    "Operator output bias shape does not match its weight."
                )
            delta = delta + bias
        if tuple(delta.shape) != tuple(residual.shape):
            raise Lfm2ChildClosedLoopRefinementError(
                "Candidate operator delta does not match the child residual."
            )
        result.append((residual + delta).contiguous())
    return tuple(result)


def compute_lfm2_candidate_post_operator(
    cases: Lfm2OperatorProducerCases,
    contexts: Lfm2AlignedChildContexts,
    *,
    output_weight: Any,
    output_bias: Any | None = None,
) -> Lfm2CandidatePostOperator:
    """Apply a candidate consumer to actual contexts and score it offline."""

    if not isinstance(cases, Lfm2OperatorProducerCases):
        raise TypeError("cases must be Lfm2OperatorProducerCases.")
    if not isinstance(contexts, Lfm2AlignedChildContexts):
        raise TypeError("contexts must be Lfm2AlignedChildContexts.")
    if (
        cases.layer_index != contexts.layer_index
        or cases.operator_kind != contexts.operator_kind
        or cases.fit_origins != contexts.fit_origins
        or cases.validation_origins != contexts.validation_origins
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Operator cases and actual contexts have different origins."
        )
    fit = _operator_candidate_values(
        cases.fit_cases,
        contexts.fit_contexts,
        operator_kind=cases.operator_kind,
        output_weight=output_weight,
        output_bias=output_bias,
    )
    validation = _operator_candidate_values(
        cases.validation_cases,
        contexts.validation_contexts,
        operator_kind=cases.operator_kind,
        output_weight=output_weight,
        output_bias=output_bias,
    )
    fit_targets = tuple(
        (
            case.parent_post_attention_residual_target_space
            if cases.operator_kind == "full_attention"
            else case.parent_post_operator_residual_target_space
        )
        for case in cases.fit_cases
    )
    validation_targets = tuple(
        (
            case.parent_post_attention_residual_target_space
            if cases.operator_kind == "full_attention"
            else case.parent_post_operator_residual_target_space
        )
        for case in cases.validation_cases
    )
    candidate_fit, current_fit = _measure_set(
        fit,
        contexts.fit_current_post_operator,
        fit_targets,
    )
    candidate_validation, current_validation = _optional_measure_set(
        validation,
        contexts.validation_current_post_operator,
        validation_targets,
    )
    return Lfm2CandidatePostOperator(
        fit_values=fit,
        validation_values=validation,
        fit_origins=cases.fit_origins,
        validation_origins=cases.validation_origins,
        comparison=Lfm2OfflineEndpointComparison(
            candidate_fit=candidate_fit,
            current_fit=current_fit,
            candidate_validation=candidate_validation,
            current_validation=current_validation,
        ),
        child_replay_sha256=contexts.child_replay_sha256,
    )


@dataclass(frozen=True, slots=True)
class Lfm2MlpCases:
    """MLP cases whose pre-MLP residual is the candidate operator endpoint."""

    layer_index: int
    fit_cases: tuple[Lfm2CachedMlpFoldCase, ...]
    validation_cases: tuple[Lfm2CachedMlpFoldCase, ...]
    fit_current_post_layer: tuple[Any, ...]
    validation_current_post_layer: tuple[Any, ...]
    fit_origins: tuple[Lfm2CaseOrigin, ...]
    validation_origins: tuple[Lfm2CaseOrigin, ...]
    norm_materializer: Lfm2NormMaterializer
    child_replay_sha256: str


def _origin_batch(
    paired: Lfm2PairedReplay,
    origin: Lfm2CaseOrigin,
) -> Lfm2PairedReplayBatch:
    batch = paired.load_batch(origin.batch_index)
    if (
        batch.input_sha256 != origin.input_sha256
        or _origin(batch, origin.phase, origin.split) != origin
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Candidate endpoint origin no longer matches the replay."
        )
    return batch


def build_lfm2_mlp_cases_from_candidate_post_operator(
    split: Lfm2WholeBatchSplit,
    operator_cases: Lfm2OperatorProducerCases,
    candidate: Lfm2CandidatePostOperator,
    *,
    coordinate_map_callback: CoordinateMapCallback,
    norm_fit_callback: NormFitCallback,
) -> Lfm2MlpCases:
    """Fit the FFN norm and build MLP cases from the candidate residual."""

    if not isinstance(split, Lfm2WholeBatchSplit):
        raise TypeError("split must be Lfm2WholeBatchSplit.")
    if not isinstance(operator_cases, Lfm2OperatorProducerCases):
        raise TypeError("operator_cases must be Lfm2OperatorProducerCases.")
    if not isinstance(candidate, Lfm2CandidatePostOperator):
        raise TypeError("candidate must be Lfm2CandidatePostOperator.")
    if (
        operator_cases.parent_cache_key != split.paired.parent_cache_key
        or candidate.fit_origins != operator_cases.fit_origins
        or candidate.validation_origins != operator_cases.validation_origins
        or candidate.child_replay_sha256 != split.paired.child_replay_sha256
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Candidate post-operator endpoints are not from this replay split."
        )
    if (
        tuple(sorted({origin.batch_index for origin in candidate.fit_origins}))
        != split.fit_batch_indices
        or tuple(
            sorted({origin.batch_index for origin in candidate.validation_origins})
        )
        != split.validation_batch_indices
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Recaptured replay changed the whole-batch fit/validation split."
        )

    def desired_norm_target(origin: Lfm2CaseOrigin, value: Any) -> Any:
        batch = _origin_batch(split.paired, origin)
        base = f"{origin.phase}.layer.{operator_cases.layer_index:03d}."
        return _project_parent(
            coordinate_map_callback,
            _required(batch.parent.tensors, base + "mlp.input"),
            target_shape=tuple(value.shape),
            name=base + "projected_parent.mlp.input",
        )

    fit_targets = tuple(
        desired_norm_target(origin, value)
        for origin, value in zip(candidate.fit_origins, candidate.fit_values)
    )
    validation_targets = tuple(
        desired_norm_target(origin, value)
        for origin, value in zip(
            candidate.validation_origins,
            candidate.validation_values,
        )
    )
    materializer = _fit_norm(
        norm_fit_callback,
        Lfm2NormFitRequest(
            role=f"layer.{operator_cases.layer_index:03d}.ffn_norm",
            fit_inputs=candidate.fit_values,
            fit_targets=fit_targets,
            validation_inputs=candidate.validation_values,
            validation_targets=validation_targets,
        ),
    )

    def build(
        origins: Sequence[Lfm2CaseOrigin],
        values: Sequence[Any],
    ) -> tuple[tuple[Lfm2CachedMlpFoldCase, ...], tuple[Any, ...]]:
        result: list[Lfm2CachedMlpFoldCase] = []
        current_values: list[Any] = []
        for origin, candidate_post_operator in zip(origins, values):
            batch = _origin_batch(split.paired, origin)
            base = f"{origin.phase}.layer.{operator_cases.layer_index:03d}."
            candidate_residual = _cpu_tensor(
                candidate_post_operator,
                base + "candidate.residual.post_operator",
                ndim=3,
            )
            target_input = materializer.materialize(
                candidate_residual,
                base + "materialized_candidate.mlp.input",
            )
            parent_endpoint = _project_parent(
                coordinate_map_callback,
                _required(
                    batch.parent.tensors,
                    base + "residual.post_layer",
                ),
                target_shape=tuple(candidate_residual.shape),
                name=base + "projected_parent.residual.post_layer",
            )
            result.append(
                Lfm2CachedMlpFoldCase(
                    target_input=_rows(
                        target_input,
                        base + "candidate.mlp.input",
                    ),
                    parent_gate=_rows(
                        _required(
                            batch.parent.tensors,
                            base + "mlp.gate",
                        ),
                        base + "parent.mlp.gate",
                    ),
                    parent_up=_rows(
                        _required(
                            batch.parent.tensors,
                            base + "mlp.up",
                        ),
                        base + "parent.mlp.up",
                    ),
                    parent_post_swiglu=_rows(
                        _required(
                            batch.parent.tensors,
                            base + "mlp.post_swiglu",
                        ),
                        base + "parent.mlp.post_swiglu",
                    ),
                    parent_post_mlp_residual_target_space=_rows(
                        parent_endpoint,
                        base + "projected_parent.residual.post_layer",
                    ),
                    target_pre_mlp_residual=_rows(
                        candidate_residual,
                        base + "candidate.residual.post_operator",
                    ),
                )
            )
            current_values.append(
                _cpu_tensor(
                    _required(
                        batch.child.tensors,
                        base + "residual.post_layer",
                    ),
                    base + "current_child.residual.post_layer",
                    ndim=3,
                )
            )
        return tuple(result), tuple(current_values)

    fit_cases, fit_current = build(
        candidate.fit_origins,
        candidate.fit_values,
    )
    validation_cases, validation_current = build(
        candidate.validation_origins,
        candidate.validation_values,
    )
    return Lfm2MlpCases(
        layer_index=operator_cases.layer_index,
        fit_cases=fit_cases,
        validation_cases=validation_cases,
        fit_current_post_layer=fit_current,
        validation_current_post_layer=validation_current,
        fit_origins=candidate.fit_origins,
        validation_origins=candidate.validation_origins,
        norm_materializer=materializer,
        child_replay_sha256=candidate.child_replay_sha256,
    )


@dataclass(frozen=True, slots=True)
class Lfm2CandidatePostLayer:
    fit_values: tuple[Any, ...]
    validation_values: tuple[Any, ...]
    comparison: Lfm2OfflineEndpointComparison
    child_replay_sha256: str


def _mlp_candidate_values(
    cases: Sequence[Lfm2CachedMlpFoldCase],
    *,
    gate_weight: Any,
    up_weight: Any,
    down_weight: Any,
) -> tuple[Any, ...]:
    torch = _torch()
    gate = _cpu_tensor(gate_weight, "MLP gate weight", ndim=2).to(torch.float64)
    up = _cpu_tensor(up_weight, "MLP up weight", ndim=2).to(torch.float64)
    down = _cpu_tensor(down_weight, "MLP down weight", ndim=2).to(torch.float64)
    if tuple(gate.shape) != tuple(up.shape) or tuple(down.shape) != (
        int(gate.shape[1]),
        int(gate.shape[0]),
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "Candidate MLP weights do not form one legal SwiGLU."
        )
    result: list[Any] = []
    for case in cases:
        design = _rows(case.target_input, "candidate MLP input").to(torch.float64)
        residual = _rows(
            case.target_pre_mlp_residual,
            "candidate pre-MLP residual",
        ).to(torch.float64)
        if int(design.shape[1]) != int(gate.shape[1]):
            raise Lfm2ChildClosedLoopRefinementError(
                "Candidate MLP input width differs from its weight."
            )
        post = torch.nn.functional.silu(design @ gate.T) * (design @ up.T)
        endpoint = residual + post @ down.T
        if tuple(endpoint.shape) != tuple(residual.shape):
            raise Lfm2ChildClosedLoopRefinementError(
                "Candidate MLP endpoint shape differs from its residual."
            )
        result.append(endpoint.contiguous())
    return tuple(result)


def compute_lfm2_candidate_post_layer(
    cases: Lfm2MlpCases,
    *,
    gate_weight: Any,
    up_weight: Any,
    down_weight: Any,
) -> Lfm2CandidatePostLayer:
    """Compute and compare a candidate SwiGLU endpoint without a model call."""

    if not isinstance(cases, Lfm2MlpCases):
        raise TypeError("cases must be Lfm2MlpCases.")
    fit = _mlp_candidate_values(
        cases.fit_cases,
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
    )
    validation = _mlp_candidate_values(
        cases.validation_cases,
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
    )
    fit_targets = tuple(
        case.parent_post_mlp_residual_target_space for case in cases.fit_cases
    )
    validation_targets = tuple(
        case.parent_post_mlp_residual_target_space for case in cases.validation_cases
    )
    candidate_fit, current_fit = _measure_set(
        fit,
        cases.fit_current_post_layer,
        fit_targets,
    )
    candidate_validation, current_validation = _optional_measure_set(
        validation,
        cases.validation_current_post_layer,
        validation_targets,
    )
    return Lfm2CandidatePostLayer(
        fit_values=fit,
        validation_values=validation,
        comparison=Lfm2OfflineEndpointComparison(
            candidate_fit=candidate_fit,
            current_fit=current_fit,
            candidate_validation=candidate_validation,
            current_validation=current_validation,
        ),
        child_replay_sha256=cases.child_replay_sha256,
    )


@dataclass(frozen=True, slots=True)
class Lfm2FinalNormPairs:
    """Final-norm pairs driven by the actual live-child final residual."""

    layer_index: int
    fit_inputs: tuple[Any, ...]
    fit_targets: tuple[Any, ...]
    validation_inputs: tuple[Any, ...]
    validation_targets: tuple[Any, ...]
    fit_origins: tuple[Lfm2CaseOrigin, ...]
    validation_origins: tuple[Lfm2CaseOrigin, ...]
    child_replay_sha256: str


def build_lfm2_final_norm_pairs(
    split: Lfm2WholeBatchSplit,
    *,
    final_layer_index: int,
    coordinate_map_callback: CoordinateMapCallback,
    parent_final_norm_callback: Callable[[Any], Any],
) -> Lfm2FinalNormPairs:
    """Pair actual child final residuals with projected parent norm outputs."""

    if not isinstance(split, Lfm2WholeBatchSplit):
        raise TypeError("split must be Lfm2WholeBatchSplit.")
    if (
        isinstance(final_layer_index, bool)
        or not isinstance(final_layer_index, int)
        or final_layer_index < 0
        or final_layer_index not in split.paired.selected_layer_indices
    ):
        raise Lfm2ChildClosedLoopRefinementError(
            "final_layer_index must be present in the child replay."
        )
    if not callable(parent_final_norm_callback):
        raise TypeError("parent_final_norm_callback must be callable.")

    fit_inputs: list[Any] = []
    fit_targets: list[Any] = []
    validation_inputs: list[Any] = []
    validation_targets: list[Any] = []
    fit_origins: list[Lfm2CaseOrigin] = []
    validation_origins: list[Lfm2CaseOrigin] = []
    for batch in split.paired.batches:
        split_name = _split_name(split, batch.index)
        for phase in batch.phase_order:
            base = f"{phase}.layer.{final_layer_index:03d}."
            origin = _origin(batch, phase, split_name)
            child_residual = _cpu_tensor(
                _required(
                    batch.child.tensors,
                    base + "residual.post_layer",
                ),
                base + "actual_child.final_residual",
                ndim=3,
            )
            parent_residual = _cpu_tensor(
                _required(
                    batch.parent.tensors,
                    base + "residual.post_layer",
                ),
                base + "parent.final_residual",
                ndim=3,
            )
            parent_before = _tensor_sha256(parent_residual)
            normalized = parent_final_norm_callback(parent_residual)
            if _tensor_sha256(parent_residual) != parent_before:
                raise Lfm2ChildClosedLoopRefinementError(
                    "Parent final-norm callback mutated its sealed residual."
                )
            target = _project_parent(
                coordinate_map_callback,
                normalized,
                target_shape=tuple(child_residual.shape),
                name=base + "projected_parent.final_norm_output",
            )
            if split_name == "fit":
                fit_inputs.append(_rows(child_residual, "child final residual"))
                fit_targets.append(_rows(target, "parent final norm target"))
                fit_origins.append(origin)
            else:
                validation_inputs.append(_rows(child_residual, "child final residual"))
                validation_targets.append(_rows(target, "parent final norm target"))
                validation_origins.append(origin)
    return Lfm2FinalNormPairs(
        layer_index=final_layer_index,
        fit_inputs=tuple(fit_inputs),
        fit_targets=tuple(fit_targets),
        validation_inputs=tuple(validation_inputs),
        validation_targets=tuple(validation_targets),
        fit_origins=tuple(fit_origins),
        validation_origins=tuple(validation_origins),
        child_replay_sha256=split.paired.child_replay_sha256,
    )


def _state_items(layer: Any) -> tuple[tuple[str, Any], ...]:
    state = layer.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise Lfm2ChildClosedLoopRollbackError(
            "Layer state must be a nonempty mapping."
        )
    return tuple(
        (
            name,
            _cpu_tensor(
                value,
                f"layer state {name}",
                floating=False,
            ),
        )
        for name, value in sorted(state.items())
    )


def _state_sha256(items: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_tensor_sha256(value)))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Lfm2ParameterSnapshot:
    name: str
    parameter_id: int
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool
    gradient_id: int | None
    gradient_shape: tuple[int, ...] | None
    gradient_dtype: str | None
    gradient: Any | None


@dataclass(frozen=True, slots=True)
class Lfm2RefinementLayerSnapshot:
    state_items: tuple[tuple[str, Any], ...]
    state_spec: tuple[tuple[str, tuple[int, ...], str], ...]
    state_sha256: str
    parameters: tuple[Lfm2ParameterSnapshot, ...]


@dataclass(frozen=True, slots=True)
class Lfm2RefinementLayerRestoreReport:
    exact: bool
    state_sha256: str
    parameter_identities_preserved: bool
    parameter_shapes_preserved: bool
    parameter_dtypes_preserved: bool
    gradient_state_preserved: bool


def snapshot_lfm2_refinement_layer(
    layer: Any,
) -> Lfm2RefinementLayerSnapshot:
    """Snapshot state and gradient bytes without replacing any Parameter."""

    torch = _torch()
    items = _state_items(layer)
    parameters: list[Lfm2ParameterSnapshot] = []
    for name, parameter in layer.named_parameters(remove_duplicate=False):
        if not isinstance(parameter, torch.nn.Parameter):
            raise Lfm2ChildClosedLoopRollbackError(
                f"Layer parameter {name!r} is invalid."
            )
        gradient = parameter.grad
        if gradient is not None:
            gradient = _cpu_tensor(
                gradient,
                f"gradient {name}",
                floating=False,
            )
        parameters.append(
            Lfm2ParameterSnapshot(
                name=name,
                parameter_id=id(parameter),
                shape=tuple(int(size) for size in parameter.shape),
                dtype=str(parameter.dtype),
                requires_grad=bool(parameter.requires_grad),
                gradient_id=(None if parameter.grad is None else id(parameter.grad)),
                gradient_shape=(
                    None
                    if parameter.grad is None
                    else tuple(int(size) for size in parameter.grad.shape)
                ),
                gradient_dtype=(
                    None if parameter.grad is None else str(parameter.grad.dtype)
                ),
                gradient=gradient,
            )
        )
    if not parameters:
        raise Lfm2ChildClosedLoopRollbackError("Layer snapshot found no Parameters.")
    return Lfm2RefinementLayerSnapshot(
        state_items=items,
        state_spec=tuple(
            (name, tuple(int(size) for size in value.shape), str(value.dtype))
            for name, value in items
        ),
        state_sha256=_state_sha256(items),
        parameters=tuple(parameters),
    )


def restore_lfm2_refinement_layer(
    layer: Any,
    snapshot: Lfm2RefinementLayerSnapshot,
) -> Lfm2RefinementLayerRestoreReport:
    """Restore state and gradients in place and prove the rollback exactly."""

    torch = _torch()
    if not isinstance(snapshot, Lfm2RefinementLayerSnapshot):
        raise TypeError("snapshot must be Lfm2RefinementLayerSnapshot.")
    current_state = layer.state_dict()
    current_spec = tuple(
        (
            name,
            tuple(int(size) for size in value.shape),
            str(value.dtype),
        )
        for name, value in sorted(current_state.items())
    )
    if current_spec != snapshot.state_spec:
        raise Lfm2ChildClosedLoopRollbackError(
            "Layer state keys, shapes, or dtypes changed before rollback."
        )
    current_parameters = tuple(layer.named_parameters(remove_duplicate=False))
    if tuple(name for name, _value in current_parameters) != tuple(
        value.name for value in snapshot.parameters
    ):
        raise Lfm2ChildClosedLoopRollbackError(
            "Layer Parameter names changed before rollback."
        )
    for (name, parameter), expected in zip(
        current_parameters,
        snapshot.parameters,
    ):
        if (
            id(parameter) != expected.parameter_id
            or tuple(parameter.shape) != expected.shape
            or str(parameter.dtype) != expected.dtype
            or bool(parameter.requires_grad) != expected.requires_grad
        ):
            raise Lfm2ChildClosedLoopRollbackError(
                f"Layer Parameter contract changed for {name!r}."
            )
        gradient = parameter.grad
        if expected.gradient_id is None:
            if gradient is not None:
                raise Lfm2ChildClosedLoopRollbackError(
                    f"Gradient presence changed for {name!r}."
                )
        elif (
            gradient is None
            or id(gradient) != expected.gradient_id
            or tuple(gradient.shape) != expected.gradient_shape
            or str(gradient.dtype) != expected.gradient_dtype
        ):
            raise Lfm2ChildClosedLoopRollbackError(
                f"Gradient identity/contract changed for {name!r}."
            )

    saved_state = dict(snapshot.state_items)
    with torch.no_grad():
        for name, value in sorted(current_state.items()):
            value.copy_(
                saved_state[name].to(
                    device=value.device,
                    dtype=value.dtype,
                )
            )
        for (_name, parameter), expected in zip(
            current_parameters,
            snapshot.parameters,
        ):
            if expected.gradient is not None:
                parameter.grad.copy_(
                    expected.gradient.to(
                        device=parameter.grad.device,
                        dtype=parameter.grad.dtype,
                    )
                )

    restored_items = _state_items(layer)
    restored_sha = _state_sha256(restored_items)
    identities = all(
        id(parameter) == expected.parameter_id
        for (_name, parameter), expected in zip(
            current_parameters,
            snapshot.parameters,
        )
    )
    shapes = all(
        tuple(parameter.shape) == expected.shape
        for (_name, parameter), expected in zip(
            current_parameters,
            snapshot.parameters,
        )
    )
    dtypes = all(
        str(parameter.dtype) == expected.dtype
        for (_name, parameter), expected in zip(
            current_parameters,
            snapshot.parameters,
        )
    )
    gradients = all(
        (
            parameter.grad is None
            if expected.gradient is None
            else (
                parameter.grad is not None
                and id(parameter.grad) == expected.gradient_id
                and _tensor_sha256(parameter.grad) == _tensor_sha256(expected.gradient)
            )
        )
        for (_name, parameter), expected in zip(
            current_parameters,
            snapshot.parameters,
        )
    )
    exact = bool(
        restored_sha == snapshot.state_sha256
        and identities
        and shapes
        and dtypes
        and gradients
    )
    if not exact:
        raise Lfm2ChildClosedLoopRollbackError(
            "Layer rollback did not restore the exact state/runtime contract."
        )
    return Lfm2RefinementLayerRestoreReport(
        exact=True,
        state_sha256=restored_sha,
        parameter_identities_preserved=True,
        parameter_shapes_preserved=True,
        parameter_dtypes_preserved=True,
        gradient_state_preserved=True,
    )


__all__ = [
    "CHILD_CLOSED_LOOP_REFINEMENT_SCHEMA",
    "CoordinateMapCallback",
    "Lfm2AlignedChildContexts",
    "Lfm2CandidatePostLayer",
    "Lfm2CandidatePostOperator",
    "Lfm2CaseOrigin",
    "Lfm2ChildClosedLoopRefinementError",
    "Lfm2ChildClosedLoopRollbackError",
    "Lfm2FinalNormPairs",
    "Lfm2MlpCases",
    "Lfm2NormFitRequest",
    "Lfm2NormMaterializer",
    "Lfm2OfflineEndpointComparison",
    "Lfm2OperatorProducerCases",
    "Lfm2PairedReplay",
    "Lfm2PairedReplayBatch",
    "Lfm2ParameterSnapshot",
    "Lfm2RefinementLayerRestoreReport",
    "Lfm2RefinementLayerSnapshot",
    "Lfm2WholeBatchSplit",
    "NormFitCallback",
    "build_lfm2_final_norm_pairs",
    "build_lfm2_mlp_cases_from_candidate_post_operator",
    "build_lfm2_operator_producer_cases",
    "compute_lfm2_candidate_post_layer",
    "compute_lfm2_candidate_post_operator",
    "extract_lfm2_actual_child_contexts",
    "pair_lfm2_parent_child_replay",
    "project_lfm2_paired_parent_logits",
    "restore_lfm2_refinement_layer",
    "snapshot_lfm2_refinement_layer",
    "split_lfm2_replay_batches",
]
