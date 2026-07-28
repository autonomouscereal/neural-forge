"""Cached-evidence replay for a partial parent-shell QDEIM ShortConv fold.

The adapter consumes already-captured parent phase tensors and one fixed
intact-channel QDEIM selection.  It reconstructs the selected ShortConv's
pre-``out_proj`` context with the causal depthwise recurrence, flattens only
the phase rows aligned to the cached parent local action, and emits evidence
accepted by :mod:`hf_lfm2_parent_shell_writer_catchup`.

No model object is constructed or executed.  The parent is never recaptured.
All inputs are detached CPU tensors, all arithmetic is deterministic, and no
filesystem or network surface exists in this module.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_parent_shell_writer_catchup import (
    Lfm2ParentShellLocalWriterEvidenceCase,
    lfm2_parent_shell_local_writer_evidence_sha256,
)


LFM2_PARTIAL_SHORT_CONV_PHASE_SCHEMA = (
    "neural-forge-lfm2-partial-short-conv-cached-phase-v1"
)
LFM2_PARTIAL_SHORT_CONV_CANDIDATE_SCHEMA = (
    "neural-forge-lfm2-partial-short-conv-qdeim-candidate-v1"
)
LFM2_PARTIAL_SHORT_CONV_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-partial-short-conv-parent-shell-evidence-v1"
)
LFM2_PARTIAL_SHORT_CONV_REPLAY_SCHEMA = (
    "neural-forge-lfm2-partial-short-conv-causal-replay-v1"
)

Lfm2PartialShortConvPartition = Literal["fit", "heldout"]

_DECODE_PHASE = re.compile(r"decode\.(?P<step>\d{3})\Z")
_OPERATION_AUDIT = MappingProxyType(
    {
        "model_objects_constructed": 0,
        "parent_model_forward_calls": 0,
        "candidate_model_forward_calls": 0,
        "model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "new_parent_trace_rows": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "parameters_shared": 0,
        "filesystem_reads": 0,
        "filesystem_writes": 0,
        "network_calls": 0,
    }
)


class Lfm2PartialShortConvEvidenceError(RuntimeError):
    """Cached partial-ShortConv evidence failed authentication or replay."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError(
            "Partial ShortConv evidence replay requires PyTorch."
        ) from exc
    return torch


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2PartialShortConvEvidenceError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Lfm2PartialShortConvEvidenceError(
            f"{label} must be a nonempty canonical identity."
        )
    return value


def _partition(value: str) -> Lfm2PartialShortConvPartition:
    if value not in {"fit", "heldout"}:
        raise Lfm2PartialShortConvEvidenceError(
            "partition must be exactly 'fit' or 'heldout'."
        )
    return value  # type: ignore[return-value]


def _phase(value: Any) -> tuple[str, int | None]:
    if value == "prefill":
        return "prefill", None
    if not isinstance(value, str):
        raise Lfm2PartialShortConvEvidenceError(
            "phase must be 'prefill' or canonical 'decode.NNN'."
        )
    match = _DECODE_PHASE.fullmatch(value)
    if match is None:
        raise Lfm2PartialShortConvEvidenceError(
            "phase must be 'prefill' or canonical 'decode.NNN'."
        )
    return value, int(match.group("step"))


def _row_ids(
    supplied: Sequence[str],
    *,
    row_count: int,
    label: str,
) -> tuple[str, ...]:
    if isinstance(supplied, (str, bytes)):
        raise Lfm2PartialShortConvEvidenceError(
            f"{label} must be an explicit sequence of stable row IDs."
        )
    try:
        values = tuple(supplied)
    except TypeError as exc:
        raise Lfm2PartialShortConvEvidenceError(
            f"{label} must be an explicit sequence of stable row IDs."
        ) from exc
    if (
        len(values) != row_count
        or len(set(values)) != row_count
        or any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in values
        )
    ):
        raise Lfm2PartialShortConvEvidenceError(
            f"{label} must contain one unique nonempty canonical ID per row."
        )
    return values


def _cached_tensor(
    value: Any,
    label: str,
    *,
    ndim: int,
    allowed_dtypes: set[Any],
) -> Any:
    torch = _torch()
    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.ndim != ndim
        or value.numel() == 0
        or value.dtype not in allowed_dtypes
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_leaf
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        expected = "/".join(
            sorted(str(dtype).removeprefix("torch.") for dtype in allowed_dtypes)
        )
        raise Lfm2PartialShortConvEvidenceError(
            f"{label} must be an exact detached finite contiguous CPU {expected} "
            f"rank-{ndim} cached tensor."
        )
    return value.detach().clone(memory_format=torch.contiguous_format)


def _history_tensor(value: Any, label: str) -> Any:
    torch = _torch()
    return _cached_tensor(
        value,
        label,
        ndim=3,
        allowed_dtypes={torch.bfloat16, torch.float32},
    )


def _residual_tensor(value: Any, label: str) -> Any:
    torch = _torch()
    return _cached_tensor(
        value,
        label,
        ndim=3,
        allowed_dtypes={torch.bfloat16, torch.float32},
    )


def _weight_tensor(value: Any, label: str) -> Any:
    torch = _torch()
    return _cached_tensor(
        value,
        label,
        ndim=2,
        allowed_dtypes={torch.float32, torch.float64},
    )


def _kernel_tensor(value: Any, label: str) -> Any:
    torch = _torch()
    return _cached_tensor(
        value,
        label,
        ndim=3,
        allowed_dtypes={torch.float32},
    )


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "data_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


@dataclass(frozen=True, slots=True)
class Lfm2PartialShortConvCachedPhase:
    """One authenticated full-history parent phase and sampled action rows."""

    case_id: str
    phase: str
    parent_pre_operator_residual: Any = field(repr=False, compare=False)
    parent_post_operator_residual: Any = field(repr=False, compare=False)
    parent_short_conv_o: Any = field(repr=False, compare=False)
    parent_b_history: Any = field(repr=False, compare=False)
    parent_c_history: Any = field(repr=False, compare=False)
    parent_x_history: Any = field(repr=False, compare=False)
    parent_context_from_fir: Any = field(repr=False, compare=False)
    parent_pre_bx_state: Any = field(repr=False, compare=False)
    row_sample_weights: Any = field(repr=False, compare=False)
    row_ids: Sequence[str] = field(repr=False, compare=False)
    parent_trajectory_receipt_sha256: str
    cached_phase_receipt_sha256: str
    aligned_row_identity_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2CompiledPartialShortConvEvidence:
    """Writer-ready cases plus an authenticated algebraic replay receipt."""

    partition: Lfm2PartialShortConvPartition
    evidence_cases: tuple[Lfm2ParentShellLocalWriterEvidenceCase, ...]
    report: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        self.verify()
        return copy.deepcopy(dict(self.report))

    def verify(self) -> str:
        body = copy.deepcopy(dict(self.report))
        seal = body.pop("result_sha256", None)
        if seal != canonical_sha256(body):
            raise Lfm2PartialShortConvEvidenceError(
                "Partial ShortConv evidence receipt seal is invalid."
            )
        actual = lfm2_parent_shell_local_writer_evidence_sha256(
            self.evidence_cases,
            partition=self.partition,
        )
        expected = body["output"]["parent_shell_writer_evidence_sha256"]
        if actual != expected:
            raise Lfm2PartialShortConvEvidenceError(
                "Writer-ready partial ShortConv evidence changed after sealing."
            )
        return str(seal)


def _selected_channels(
    supplied: Sequence[int],
    *,
    source_channels: int,
) -> tuple[int, ...]:
    if isinstance(supplied, (str, bytes)):
        raise Lfm2PartialShortConvEvidenceError(
            "selected_source_channels must be a sequence of integers."
        )
    try:
        values = tuple(supplied)
    except TypeError as exc:
        raise Lfm2PartialShortConvEvidenceError(
            "selected_source_channels must be a sequence of integers."
        ) from exc
    if (
        not values
        or any(type(value) is not int for value in values)
        or len(set(values)) != len(values)
        or any(value < 0 or value >= source_channels for value in values)
    ):
        raise Lfm2PartialShortConvEvidenceError(
            "selected_source_channels must be nonempty, unique, and in source range."
        )
    return values


def _normalize_phases(
    phases: Sequence[Lfm2PartialShortConvCachedPhase],
) -> tuple[dict[str, Any], ...]:
    if isinstance(phases, (str, bytes)):
        raise Lfm2PartialShortConvEvidenceError("cached phases must be a sequence.")
    try:
        supplied = tuple(phases)
    except TypeError as exc:
        raise Lfm2PartialShortConvEvidenceError(
            "cached phases must be a sequence."
        ) from exc
    if not supplied:
        raise Lfm2PartialShortConvEvidenceError(
            "At least one cached parent phase is required."
        )
    torch = _torch()
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_rows: set[str] = set()
    source_width: int | None = None
    parent_hidden: int | None = None
    history_dtype: Any | None = None
    residual_dtype: Any | None = None
    for index, item in enumerate(supplied):
        if not isinstance(item, Lfm2PartialShortConvCachedPhase):
            raise Lfm2PartialShortConvEvidenceError(
                f"cached phase {index} has the wrong evidence type."
            )
        case_id = _identity(item.case_id, f"cached phase {index} case_id")
        phase_name, decode_step = _phase(item.phase)
        if case_id in seen_ids:
            raise Lfm2PartialShortConvEvidenceError(
                "Cached phase case IDs must be unique."
            )
        seen_ids.add(case_id)
        parent_pre = _residual_tensor(
            item.parent_pre_operator_residual,
            f"cached phase {case_id} parent_pre_operator_residual",
        )
        parent_post = _residual_tensor(
            item.parent_post_operator_residual,
            f"cached phase {case_id} parent_post_operator_residual",
        )
        short_conv_o = _residual_tensor(
            item.parent_short_conv_o,
            f"cached phase {case_id} parent_short_conv_o",
        )
        b_history = _history_tensor(
            item.parent_b_history,
            f"cached phase {case_id} parent_b_history",
        )
        c_history = _history_tensor(
            item.parent_c_history,
            f"cached phase {case_id} parent_c_history",
        )
        x_history = _history_tensor(
            item.parent_x_history,
            f"cached phase {case_id} parent_x_history",
        )
        context_from_fir = _history_tensor(
            item.parent_context_from_fir,
            f"cached phase {case_id} parent_context_from_fir",
        )
        pre_state = _history_tensor(
            item.parent_pre_bx_state,
            f"cached phase {case_id} parent_pre_bx_state",
        )
        weights = _weight_tensor(
            item.row_sample_weights,
            f"cached phase {case_id} row_sample_weights",
        )
        batch, sampled_tokens, hidden = map(int, parent_pre.shape)
        row_ids = _row_ids(
            item.row_ids,
            row_count=batch * sampled_tokens,
            label=f"cached phase {case_id} row_ids",
        )
        if seen_rows.intersection(row_ids):
            raise Lfm2PartialShortConvEvidenceError(
                "Cached phase stable row IDs must be globally unique."
            )
        seen_rows.update(row_ids)
        history_batch, history_tokens, channels = map(int, b_history.shape)
        fir_length = int(pre_state.shape[-1])
        if (
            tuple(parent_post.shape) != tuple(parent_pre.shape)
            or parent_post.dtype != parent_pre.dtype
            or int(short_conv_o.shape[0]) != batch
            or int(short_conv_o.shape[1]) < sampled_tokens
            or int(short_conv_o.shape[2]) != hidden
            or tuple(c_history.shape) != tuple(b_history.shape)
            or tuple(x_history.shape) != tuple(b_history.shape)
            or c_history.dtype != b_history.dtype
            or x_history.dtype != b_history.dtype
            or history_batch != batch
            or sampled_tokens > history_tokens
            or int(context_from_fir.shape[0]) != batch
            or int(context_from_fir.shape[1]) < sampled_tokens
            or int(context_from_fir.shape[2]) != channels
            or tuple(pre_state.shape) != (batch, channels, fir_length)
            or pre_state.dtype != b_history.dtype
            or tuple(weights.shape) != (batch, sampled_tokens)
            or not bool((weights > 0).all().item())
        ):
            raise Lfm2PartialShortConvEvidenceError(
                f"Cached phase {case_id} tensors violate exact batch/history/action "
                "shape or dtype alignment."
            )
        if phase_name == "prefill":
            if sampled_tokens != history_tokens or not torch.equal(
                pre_state, torch.zeros_like(pre_state)
            ):
                raise Lfm2PartialShortConvEvidenceError(
                    f"Cached prefill {case_id} must sample its full history and "
                    "start from an exactly zero pre-Bx state."
                )
        else:
            prefix_tokens = history_tokens - sampled_tokens
            # Canonical decode cache phases usually contain only their newly
            # executed suffix (normally T=1) and seal the independent cache
            # state from the preceding phases.  When a caller instead supplies
            # a complete prefix, authenticate that redundant relationship.
            if prefix_tokens > 0:
                prefix_bx = (
                    b_history[:, :prefix_tokens, :] * x_history[:, :prefix_tokens, :]
                ).transpose(1, 2)
                if int(prefix_bx.shape[-1]) < fir_length:
                    prefix_bx = torch.nn.functional.pad(
                        prefix_bx,
                        (fir_length - int(prefix_bx.shape[-1]), 0),
                    )
                expected_state = prefix_bx[..., -fir_length:].contiguous()
                if not torch.equal(pre_state, expected_state):
                    raise Lfm2PartialShortConvEvidenceError(
                        f"Cached decode {case_id} pre-Bx state is not the exact tail "
                        "of its supplied full parent B*x prefix."
                    )
        if source_width is None:
            source_width = channels
            parent_hidden = hidden
            history_dtype = b_history.dtype
            residual_dtype = parent_pre.dtype
        elif (
            channels != source_width
            or hidden != parent_hidden
            or b_history.dtype != history_dtype
            or parent_pre.dtype != residual_dtype
            or fir_length != int(normalized[0]["pre_state"].shape[-1])
        ):
            raise Lfm2PartialShortConvEvidenceError(
                "All cached phases must share source width, parent hidden width, "
                "FIR length, and cache dtypes."
            )
        normalized.append(
            {
                "case_id": case_id,
                "phase": phase_name,
                "decode_step": decode_step,
                "parent_pre": parent_pre,
                "parent_post": parent_post,
                "short_conv_o": short_conv_o,
                "b_history": b_history,
                "c_history": c_history,
                "x_history": x_history,
                "context_from_fir": context_from_fir,
                "pre_state": pre_state,
                "weights": weights,
                "row_ids": row_ids,
                "batch": batch,
                "sampled_tokens": sampled_tokens,
                "history_tokens": history_tokens,
                "source_channels": channels,
                "parent_hidden": hidden,
                "fir_length": fir_length,
                "receipts": {
                    "parent_trajectory_receipt_sha256": _digest(
                        item.parent_trajectory_receipt_sha256,
                        f"cached phase {case_id} parent trajectory receipt",
                    ),
                    "cached_phase_receipt_sha256": _digest(
                        item.cached_phase_receipt_sha256,
                        f"cached phase {case_id} phase receipt",
                    ),
                    "aligned_row_identity_sha256": _digest(
                        item.aligned_row_identity_sha256,
                        f"cached phase {case_id} row identity",
                    ),
                },
            }
        )
    return tuple(normalized)


def _phase_evidence_sha256_from_normalized(
    phases: Sequence[Mapping[str, Any]],
    *,
    partition: Lfm2PartialShortConvPartition,
) -> str:
    return canonical_sha256(
        {
            "schema": LFM2_PARTIAL_SHORT_CONV_PHASE_SCHEMA,
            "partition": partition,
            "case_count": len(phases),
            "phases": [
                {
                    "case_index": index,
                    "case_id": phase["case_id"],
                    "phase": phase["phase"],
                    "decode_step": phase["decode_step"],
                    "batch": phase["batch"],
                    "sampled_tokens": phase["sampled_tokens"],
                    "history_tokens": phase["history_tokens"],
                    "source_channels": phase["source_channels"],
                    "parent_hidden": phase["parent_hidden"],
                    "fir_length": phase["fir_length"],
                    "row_ids": list(phase["row_ids"]),
                    "receipts": dict(phase["receipts"]),
                    "tensors": {
                        name: {
                            "sha256": _tensor_sha256(phase[name]),
                            "shape": [int(size) for size in phase[name].shape],
                            "dtype": str(phase[name].dtype).removeprefix("torch."),
                        }
                        for name in (
                            "parent_pre",
                            "parent_post",
                            "short_conv_o",
                            "b_history",
                            "c_history",
                            "x_history",
                            "context_from_fir",
                            "pre_state",
                            "weights",
                        )
                    },
                }
                for index, phase in enumerate(phases)
            ],
            "parent_bcx_phase_histories_required": True,
            "direct_parent_context_from_fir_required": True,
            "direct_parent_short_conv_o_action_required": True,
            "explicit_stable_row_ids_required": True,
            "decode_pre_bx_state_authenticated_against_prefix_when_redundant": True,
            "operation_audit": dict(_OPERATION_AUDIT),
        }
    )


def lfm2_partial_short_conv_cached_phase_evidence_sha256(
    phases: Sequence[Lfm2PartialShortConvCachedPhase],
    *,
    partition: Lfm2PartialShortConvPartition,
) -> str:
    """Authenticate cached parent phase tensors and their partition."""

    resolved = _partition(partition)
    normalized = _normalize_phases(phases)
    return _phase_evidence_sha256_from_normalized(
        normalized,
        partition=resolved,
    )


def lfm2_partial_short_conv_qdeim_candidate_sha256(
    *,
    selected_source_channels: Sequence[int],
    candidate_fir_kernel: Any,
    source_internal_channels: int,
) -> str:
    """Authenticate the ordered intact-channel selection and candidate FIR."""

    if type(source_internal_channels) is not int or source_internal_channels <= 0:
        raise Lfm2PartialShortConvEvidenceError(
            "source_internal_channels must be a positive integer."
        )
    selected = _selected_channels(
        selected_source_channels,
        source_channels=source_internal_channels,
    )
    kernel = _kernel_tensor(candidate_fir_kernel, "candidate_fir_kernel")
    if tuple(kernel.shape[:2]) != (len(selected), 1):
        raise Lfm2PartialShortConvEvidenceError(
            "candidate_fir_kernel must be [selected_channels,1,fir_length]."
        )
    return canonical_sha256(
        {
            "schema": LFM2_PARTIAL_SHORT_CONV_CANDIDATE_SCHEMA,
            "source_internal_channels": source_internal_channels,
            "selected_source_channels": list(selected),
            "target_internal_channels": len(selected),
            "candidate_fir_kernel_sha256": _tensor_sha256(kernel),
            "candidate_fir_kernel_shape": [int(size) for size in kernel.shape],
            "candidate_fir_kernel_dtype": "float32",
            "intact_source_channel_order_preserved": True,
            "producer_or_fir_channel_averaging_used": False,
        }
    )


def _replay_phase_context(
    phase: Mapping[str, Any],
    *,
    selected: tuple[int, ...],
    kernel: Any,
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    index = torch.tensor(selected, dtype=torch.long)
    sample_count = int(phase["sampled_tokens"])
    history_count = int(phase["history_tokens"])
    start = 0 if phase["phase"] == "prefill" else history_count - sample_count
    selected_b = phase["b_history"].index_select(2, index)
    selected_c = phase["c_history"].index_select(2, index)
    selected_x = phase["x_history"].index_select(2, index)
    selected_state = phase["pre_state"].index_select(1, index).to(torch.float32)
    kernel_rows = kernel[:, 0, :]
    replayed_contexts: list[Any] = []
    state = selected_state.contiguous().clone()
    with torch.inference_mode():
        for token in range(start, history_count):
            # Preserve the cached producer dtype for B*x, then match the
            # candidate FP32 FIR/runtime writer boundary.
            bx = (selected_b[:, token, :] * selected_x[:, token, :]).to(torch.float32)
            state = torch.cat((state[..., 1:], bx.unsqueeze(-1)), dim=-1)
            filtered = (state * kernel_rows.unsqueeze(0)).sum(dim=-1)
            context = selected_c[:, token, :].to(torch.float32) * filtered
            replayed_contexts.append(context.contiguous())
    if len(replayed_contexts) != sample_count:
        raise Lfm2PartialShortConvEvidenceError(
            f"Phase {phase['case_id']} replay did not produce every sampled row."
        )
    replayed_context = torch.stack(replayed_contexts, dim=1).contiguous().clone()
    # The canonical cache already seals the executed parent context reconstructed
    # from its FIR.  Because QDEIM retains complete channels, selecting those
    # exact rows is the non-ambiguous compact pre-writer design.  The local
    # recurrence remains a diagnostic check, never the published design source.
    direct_context = (
        phase["context_from_fir"][:, -sample_count:, :]
        .index_select(2, index)
        .to(torch.float32)
        .contiguous()
        .clone()
    )
    if (
        tuple(direct_context.shape)
        != (int(phase["batch"]), sample_count, len(selected))
        or direct_context.dtype != torch.float32
        or not bool(torch.isfinite(direct_context).all().item())
    ):
        raise Lfm2PartialShortConvEvidenceError(
            f"Phase {phase['case_id']} compact context reconstruction failed."
        )
    replay_error = (replayed_context - direct_context).abs()
    replay_maximum_absolute_error = float(replay_error.max().item())
    replay_matches_direct = bool(
        torch.allclose(
            replayed_context,
            direct_context,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
    )
    body = {
        "schema": LFM2_PARTIAL_SHORT_CONV_REPLAY_SCHEMA,
        "case_id": phase["case_id"],
        "phase": phase["phase"],
        "decode_step": phase["decode_step"],
        "cached_phase_receipt_sha256": phase["receipts"]["cached_phase_receipt_sha256"],
        "selected_source_channels": list(selected),
        "sampled_history_start": start,
        "sampled_history_stop": history_count,
        "sampled_row_count": int(phase["batch"]) * sample_count,
        "recurrence": (
            "state=concat(state[...,1:],(B*x)[...,None]); context=C*sum(state*FIR,lag)"
        ),
        "prefill_zero_state_used": phase["phase"] == "prefill",
        "decode_authenticated_pre_bx_state_used": phase["phase"] != "prefill",
        "producer_product_computed_in_cached_dtype": True,
        "fir_and_context_compute_dtype": "float32",
        "published_compact_context_source": (
            "authenticated-parent-context-from-fir-selected-intact-channels"
        ),
        "direct_parent_context_from_fir_sha256": _tensor_sha256(direct_context),
        "diagnostic_local_replay_context_sha256": _tensor_sha256(replayed_context),
        "diagnostic_local_replay_matches_direct_context": replay_matches_direct,
        "diagnostic_local_replay_maximum_absolute_error": (
            replay_maximum_absolute_error
        ),
        "diagnostic_local_replay_can_replace_direct_context": False,
        "compact_context_sha256": _tensor_sha256(direct_context),
        "compact_context_shape": [int(size) for size in direct_context.shape],
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    receipt = {**body, "result_sha256": canonical_sha256(body)}
    return direct_context, receipt


def compile_lfm2_partial_short_conv_parent_shell_evidence(
    *,
    cached_phases: Sequence[Lfm2PartialShortConvCachedPhase],
    partition: Lfm2PartialShortConvPartition,
    expected_cached_phase_evidence_sha256: str,
    selected_source_channels: Sequence[int],
    candidate_fir_kernel: Any,
    expected_qdeim_candidate_sha256: str,
) -> Lfm2CompiledPartialShortConvEvidence:
    """Reconstruct and seal writer-ready partial-ShortConv phase evidence."""

    torch = _torch()
    resolved_partition = _partition(partition)
    expected_phases = _digest(
        expected_cached_phase_evidence_sha256,
        "expected cached phase evidence SHA-256",
    )
    expected_candidate = _digest(
        expected_qdeim_candidate_sha256,
        "expected QDEIM candidate SHA-256",
    )
    normalized = _normalize_phases(cached_phases)
    source_channels = int(normalized[0]["source_channels"])
    fir_length = int(normalized[0]["fir_length"])
    selected = _selected_channels(
        selected_source_channels,
        source_channels=source_channels,
    )
    kernel = _kernel_tensor(candidate_fir_kernel, "candidate_fir_kernel")
    if tuple(kernel.shape) != (len(selected), 1, fir_length):
        raise Lfm2PartialShortConvEvidenceError(
            "Candidate FIR shape must exactly match selected channels and cached "
            "pre-Bx state length."
        )
    actual_phases_before = _phase_evidence_sha256_from_normalized(
        normalized,
        partition=resolved_partition,
    )
    actual_candidate_before = lfm2_partial_short_conv_qdeim_candidate_sha256(
        selected_source_channels=selected,
        candidate_fir_kernel=candidate_fir_kernel,
        source_internal_channels=source_channels,
    )
    if actual_phases_before != expected_phases:
        raise Lfm2PartialShortConvEvidenceError(
            "Cached parent phase evidence authentication failed."
        )
    if actual_candidate_before != expected_candidate:
        raise Lfm2PartialShortConvEvidenceError(
            "QDEIM ShortConv candidate authentication failed."
        )

    cases: list[Lfm2ParentShellLocalWriterEvidenceCase] = []
    phase_reports: list[dict[str, Any]] = []
    for phase_value in normalized:
        sampled_context, replay_receipt = _replay_phase_context(
            phase_value,
            selected=selected,
            kernel=kernel,
        )
        rows = int(phase_value["batch"]) * int(phase_value["sampled_tokens"])
        context_rows = sampled_context.reshape(rows, len(selected)).contiguous().clone()
        parent_pre_rows = (
            phase_value["parent_pre"]
            .to(torch.float32)
            .reshape(rows, int(phase_value["parent_hidden"]))
            .contiguous()
            .clone()
        )
        parent_post_rows = (
            phase_value["parent_post"]
            .to(torch.float32)
            .reshape(rows, int(phase_value["parent_hidden"]))
            .contiguous()
            .clone()
        )
        parent_local_action_rows = (
            phase_value["short_conv_o"][:, -int(phase_value["sampled_tokens"]) :, :]
            .to(torch.float32)
            .reshape(rows, int(phase_value["parent_hidden"]))
            .contiguous()
            .clone()
        )
        row_weights = (
            phase_value["weights"].to(torch.float32).reshape(rows).contiguous().clone()
        )
        evidence_case = Lfm2ParentShellLocalWriterEvidenceCase(
            case_id=str(phase_value["case_id"]),
            compact_writer_design=context_rows,
            parent_local_action=parent_local_action_rows,
            parent_pre_operator_residual=parent_pre_rows,
            parent_post_operator_residual=parent_post_rows,
            row_sample_weights=row_weights,
            row_ids=phase_value["row_ids"],
            parent_trajectory_receipt_sha256=phase_value["receipts"][
                "parent_trajectory_receipt_sha256"
            ],
            compact_replay_receipt_sha256=replay_receipt["result_sha256"],
            aligned_row_identity_sha256=phase_value["receipts"][
                "aligned_row_identity_sha256"
            ],
        )
        cases.append(evidence_case)
        phase_reports.append(
            {
                "case_id": phase_value["case_id"],
                "phase": phase_value["phase"],
                "decode_step": phase_value["decode_step"],
                "batch": phase_value["batch"],
                "sampled_tokens": phase_value["sampled_tokens"],
                "flattened_rows": rows,
                "flatten_order": "batch-major-then-sequence-major",
                "compact_context_rows_sha256": _tensor_sha256(context_rows),
                "parent_short_conv_o_action_rows_sha256": _tensor_sha256(
                    parent_local_action_rows
                ),
                "parent_pre_rows_sha256": _tensor_sha256(parent_pre_rows),
                "parent_post_rows_sha256": _tensor_sha256(parent_post_rows),
                "row_ids": list(phase_value["row_ids"]),
                "row_sample_weights_sha256": _tensor_sha256(row_weights),
                "causal_replay_receipt": replay_receipt,
            }
        )

    frozen_cases = tuple(cases)
    output_evidence_sha256 = lfm2_parent_shell_local_writer_evidence_sha256(
        frozen_cases,
        partition=resolved_partition,
    )
    # Re-authenticate caller-owned sources after all arithmetic.
    normalized_after = _normalize_phases(cached_phases)
    actual_phases_after = _phase_evidence_sha256_from_normalized(
        normalized_after,
        partition=resolved_partition,
    )
    actual_candidate_after = lfm2_partial_short_conv_qdeim_candidate_sha256(
        selected_source_channels=selected_source_channels,
        candidate_fir_kernel=candidate_fir_kernel,
        source_internal_channels=source_channels,
    )
    if (
        actual_phases_after != actual_phases_before
        or actual_candidate_after != actual_candidate_before
    ):
        raise Lfm2PartialShortConvEvidenceError(
            "Cached phases or QDEIM candidate changed during evidence replay."
        )
    for case in frozen_cases:
        for tensor in (
            case.compact_writer_design,
            case.parent_local_action,
            case.parent_pre_operator_residual,
            case.parent_post_operator_residual,
            case.row_sample_weights,
        ):
            if (
                type(tensor) is not torch.Tensor
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise Lfm2PartialShortConvEvidenceError(
                    "Writer-ready evidence failed detached dense CPU FP32 contracts."
                )

    total_rows = sum(int(report["flattened_rows"]) for report in phase_reports)
    report_body = {
        "schema": LFM2_PARTIAL_SHORT_CONV_EVIDENCE_SCHEMA,
        "status": "parent-shell-qdeim-short-conv-cached-evidence-ready",
        "partition": resolved_partition,
        "method": "cached-intact-channel-causal-fir-replay",
        "authentication": {
            "cached_phase_evidence_sha256_expected": expected_phases,
            "cached_phase_evidence_sha256_before": actual_phases_before,
            "cached_phase_evidence_sha256_after": actual_phases_after,
            "cached_phase_evidence_immutable": True,
            "qdeim_candidate_sha256_expected": expected_candidate,
            "qdeim_candidate_sha256_before": actual_candidate_before,
            "qdeim_candidate_sha256_after": actual_candidate_after,
            "qdeim_candidate_immutable": True,
        },
        "candidate": {
            "source_internal_channels": source_channels,
            "target_internal_channels": len(selected),
            "selected_source_channels": list(selected),
            "fir_length": fir_length,
            "candidate_fir_kernel_sha256": _tensor_sha256(kernel),
            "intact_channel_selection_only": True,
            "producer_or_fir_channel_averaging_used": False,
        },
        "input": {
            "phase_count": len(normalized),
            "case_ids": [phase["case_id"] for phase in normalized],
            "phase_order": [phase["phase"] for phase in normalized],
            "parent_bcx_phase_histories_consumed": True,
            "decode_suffix_only_bcx_histories_supported": True,
            "direct_parent_context_from_fir_consumed": True,
            "direct_parent_short_conv_o_consumed": True,
            "parent_short_conv_o_has_local_target_authority": True,
            "parent_residual_pre_post_consumed": True,
            "parent_residual_pre_post_are_diagnostic_only": True,
            "pre_bx_state_consumed": True,
            "cached_detached_cpu_tensors_only": True,
        },
        "phase_replays": phase_reports,
        "output": {
            "evidence_case_count": len(frozen_cases),
            "flattened_row_count": total_rows,
            "parent_hidden_width": int(normalized[0]["parent_hidden"]),
            "compact_writer_design_width": len(selected),
            "parent_shell_writer_evidence_sha256": output_evidence_sha256,
            "emitted_type": "Lfm2ParentShellLocalWriterEvidenceCase",
            "row_alignment": "phase action rows paired with final causal history rows",
            "flatten_order": "batch-major-then-sequence-major",
            "dense_independent_fp32_tensors": True,
            "local_action_target_source": "canonical-cached-short-conv-o",
            "residual_endpoint_subtraction_used_for_target": False,
            "stable_row_ids": [
                row_id for phase in normalized for row_id in phase["row_ids"]
            ],
            "preferred_design_source": (
                "authenticated-parent-context-from-fir-selected-intact-channels"
            ),
        },
        "diagnostics": {
            "prefill_phase_count": sum(
                phase["phase"] == "prefill" for phase in normalized
            ),
            "decode_phase_count": sum(
                phase["phase"] != "prefill" for phase in normalized
            ),
            "numeric_diagnostics_have_candidate_selection_authority": False,
            "manual_whole_output_review_remains_authoritative": True,
        },
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    report = {**report_body, "result_sha256": canonical_sha256(report_body)}
    result = Lfm2CompiledPartialShortConvEvidence(
        partition=resolved_partition,
        evidence_cases=frozen_cases,
        report=MappingProxyType(report),
    )
    result.verify()
    return result


__all__ = [
    "LFM2_PARTIAL_SHORT_CONV_CANDIDATE_SCHEMA",
    "LFM2_PARTIAL_SHORT_CONV_EVIDENCE_SCHEMA",
    "LFM2_PARTIAL_SHORT_CONV_PHASE_SCHEMA",
    "LFM2_PARTIAL_SHORT_CONV_REPLAY_SCHEMA",
    "Lfm2CompiledPartialShortConvEvidence",
    "Lfm2PartialShortConvCachedPhase",
    "Lfm2PartialShortConvEvidenceError",
    "compile_lfm2_partial_short_conv_parent_shell_evidence",
    "lfm2_partial_short_conv_cached_phase_evidence_sha256",
    "lfm2_partial_short_conv_qdeim_candidate_sha256",
]
