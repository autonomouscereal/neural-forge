"""Model-wide autoregressive Hankel statistics from sealed LFM2 traces.

This module is deliberately a cache consumer, not a model runner.  It extracts
the residual stream at every layer boundary plus the final LM-head input from
one already-authenticated trace batch, then forms lagged future-behaviour
cross-moments in deterministic CPU float64 arithmetic.  No logits, weights,
model objects, forwards, gradients, optimizers, pruning, or quantization are
part of this path.

The non-zero future-lag requirement is intentional.  A horizon-zero-only
artifact would merely reproduce the historical static Gram lane, while a
ShortConv-only realization would reproduce FHVR.  These statistics instead
cover all residual boundaries and at least one autoregressive future lag.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256


LFM2_AUTOREGRESSIVE_HANKEL_BATCH_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-batch-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_STATISTICS_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-sufficient-statistics-v1"
)
LFM2_AUTOREGRESSIVE_HANKEL_PAYLOAD_SCHEMA = (
    "neural-forge-lfm2-autoregressive-hankel-tensor-payload-v1"
)

_PREFILL_LAYER_PRE = re.compile(
    r"prefill\.layer\.(?P<layer>[0-9]+)\.residual\.pre\Z"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2AutoregressiveHankelStatisticsError(ValueError):
    """A sealed trajectory cannot support model-wide Hankel statistics."""


def _zero_execution_audit() -> dict[str, Any]:
    return {
        "model_objects_constructed": 0,
        "parent_forward_calls": 0,
        "child_forward_calls": 0,
        "generate_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "network_accesses": 0,
        "logit_tensors_loaded": 0,
        "gradients_used": False,
    }


def _sealed_receipt(body: Mapping[str, Any]) -> Mapping[str, Any]:
    value = dict(body)
    value["receipt_sha256"] = canonical_sha256(value)
    return MappingProxyType(value)


def _tensor_descriptor(value: Any) -> dict[str, Any]:
    """Hash one detached CPU tensor without allocating a second full payload."""

    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(
        b"neural-forge-lfm2-autoregressive-hankel-tensor-v1\0"
    )
    dtype = str(tensor.dtype).removeprefix("torch.")
    shape = [int(size) for size in tensor.shape]
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(shape)).encode("ascii"))
    digest.update(b"\0")
    byte_view = tensor.view(torch.uint8).reshape(-1)
    chunk_bytes = 16 * 1024 * 1024
    for start in range(0, int(byte_view.numel()), chunk_bytes):
        digest.update(
            byte_view[start : start + chunk_bytes].numpy().tobytes()
        )
    return {
        "dtype": dtype,
        "shape": shape,
        "numel": int(tensor.numel()),
        "size_bytes": int(tensor.numel() * tensor.element_size()),
        "sha256": digest.hexdigest(),
    }


def _tensor_payload_receipt(tensors: Mapping[str, Any]) -> dict[str, Any]:
    inventory = {
        name: _tensor_descriptor(tensors[name]) for name in sorted(tensors)
    }
    body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_PAYLOAD_SCHEMA,
        "tensor_hash_schema": "dtype-nul-shape-nul-raw-v1",
        "tensor_count": len(inventory),
        "tensors": inventory,
    }
    return {**body, "payload_sha256": canonical_sha256(body)}


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised without the HF extra
        raise RuntimeError(
            "LFM2 autoregressive Hankel statistics require the 'hf' extra."
        ) from exc
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"{name} must be a nonnegative integer."
        )
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _cpu_f64(value: Any, name: str) -> Any:
    torch = _torch()
    if not torch.is_tensor(value) or value.is_sparse:
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"{name} must be a dense torch tensor."
        )
    if not torch.is_floating_point(value):
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"{name} must have a floating-point dtype."
        )
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if result.requires_grad or result.grad_fn is not None:
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"{name} retained autograd state."
        )
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"{name} contains a non-finite value."
        )
    return result


def _cpu_i64(value: Any, name: str) -> Any:
    torch = _torch()
    if (
        not torch.is_tensor(value)
        or value.is_sparse
        or value.dtype == torch.bool
        or torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"{name} must be a dense integer tensor."
        )
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
        .clone()
    )
    if result.requires_grad or result.grad_fn is not None:  # pragma: no cover
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"{name} retained autograd state."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2AutoregressiveHankelBatch:
    """One authenticated, model-wide, single-trajectory residual sequence.

    ``boundary_rows`` has shape ``[J, N, H]``.  Boundary zero is layer zero's
    residual input; boundary ``l + 1`` is layer ``l``'s residual output.
    ``behavior_rows`` is the final normalized LM-head input, shape ``[N, H]``.
    The final state has no observed successor action, so
    ``observed_action_token_ids`` has shape ``[N - 1]``.
    """

    boundary_rows: Any
    behavior_rows: Any
    state_token_ids: Any
    observed_action_token_ids: Any
    prefix_positions: Any
    boundary_names: tuple[str, ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Lfm2AutoregressiveHankelStatistics:
    """Lag-separable sufficient statistics for one trajectory.

    Shapes for ``J`` boundaries, hidden width ``H``, and ``Q`` retained lags are:

    * ``lag_pair_counts`` and ``right_censor_counts``: ``[Q]``
    * ``state_sum``: ``[J, Q, H]``
    * ``future_sum``: ``[Q, H]``
    * ``state_second_moment``: ``[J, Q, H, H]``
    * ``future_second_moment``: ``[Q, H, H]``
    * ``state_future_cross``: ``[J, Q, H, H]``
    """

    lag_pair_counts: Any
    right_censor_counts: Any
    state_sum: Any
    future_sum: Any
    state_second_moment: Any
    future_second_moment: Any
    state_future_cross: Any
    receipt: Mapping[str, Any]


def _resolve_future_lags(
    *,
    max_horizon: int | None,
    future_lags: Sequence[int] | None,
) -> tuple[int, ...]:
    if (max_horizon is None) == (future_lags is None):
        raise Lfm2AutoregressiveHankelStatisticsError(
            "Supply exactly one of max_horizon or future_lags."
        )
    if future_lags is None:
        horizon = _positive_int(max_horizon, "max_horizon")
        return tuple(range(horizon + 1))
    if isinstance(future_lags, (str, bytes)) or not isinstance(
        future_lags, Sequence
    ):
        raise Lfm2AutoregressiveHankelStatisticsError(
            "future_lags must be a sequence of integers."
        )
    lags = tuple(future_lags)
    if (
        not lags
        or any(isinstance(value, bool) or not isinstance(value, int) for value in lags)
        or lags != tuple(sorted(set(lags)))
        or lags[0] != 0
        or lags[-1] <= 0
    ):
        raise Lfm2AutoregressiveHankelStatisticsError(
            "future_lags must be sorted, unique, start at zero, and contain "
            "at least one nonzero lag."
        )
    return lags


def accumulate_lfm2_ahb_trajectory(
    boundary_rows: Any,
    behavior_rows: Any,
    *,
    max_horizon: int | None = None,
    future_lags: Sequence[int] | None = None,
    source_receipt_sha256: str | None = None,
) -> Lfm2AutoregressiveHankelStatistics:
    """Form exact lagged moments without models, gradients, or zero padding.

    A lag ``h`` consumes ``boundary_rows[:, :N-h]`` and
    ``behavior_rows[h:]``.  Consequently exactly ``h`` terminal rows are
    right-censored instead of being silently padded.  Supply ``max_horizon``
    for the dense schedule ``0..K`` or explicit ``future_lags`` for a sparse
    schedule.  Either form must retain a nonzero lag, so this API cannot
    publish a disguised static Gram artifact.
    """

    torch = _torch()
    lags = _resolve_future_lags(
        max_horizon=max_horizon,
        future_lags=future_lags,
    )
    source_digest = (
        None
        if source_receipt_sha256 is None
        else _digest(source_receipt_sha256, "source_receipt_sha256")
    )
    with torch.inference_mode():
        states = _cpu_f64(boundary_rows, "boundary_rows")
        future = _cpu_f64(behavior_rows, "behavior_rows")
        if states.ndim != 3:
            raise Lfm2AutoregressiveHankelStatisticsError(
                "boundary_rows must have shape [boundary, row, hidden]."
            )
        if future.ndim != 2:
            raise Lfm2AutoregressiveHankelStatisticsError(
                "behavior_rows must have shape [row, hidden]."
            )
        boundary_count, row_count, hidden_size = (
            int(value) for value in states.shape
        )
        if boundary_count < 2:
            raise Lfm2AutoregressiveHankelStatisticsError(
                "At least the model input and one post-layer boundary are required."
            )
        if tuple(future.shape) != (row_count, hidden_size):
            raise Lfm2AutoregressiveHankelStatisticsError(
                "Boundary and behavior trajectory geometry disagree."
            )
        if lags[-1] >= row_count:
            raise Lfm2AutoregressiveHankelStatisticsError(
                "The maximum future lag must be smaller than the trajectory row count."
            )

        lag_count = len(lags)
        lag_pair_counts = torch.tensor(
            [row_count - lag for lag in lags],
            dtype=torch.int64,
            device="cpu",
        )
        right_censor_counts = torch.tensor(
            lags,
            dtype=torch.int64,
            device="cpu",
        )
        state_sum = torch.empty(
            (boundary_count, lag_count, hidden_size), dtype=torch.float64
        )
        future_sum = torch.empty(
            (lag_count, hidden_size), dtype=torch.float64
        )
        future_second_moment = torch.empty(
            (lag_count, hidden_size, hidden_size), dtype=torch.float64
        )
        state_future_cross = torch.empty(
            (boundary_count, lag_count, hidden_size, hidden_size),
            dtype=torch.float64,
        )
        state_second_moment = torch.empty(
            (boundary_count, lag_count, hidden_size, hidden_size),
            dtype=torch.float64,
        )

        for lag_index, lag in enumerate(lags):
            pair_count = row_count - lag
            lag_states = states[:, :pair_count, :]
            lag_future = future[lag:, :]
            state_sum[:, lag_index, :] = lag_states.sum(dim=1)
            future_sum[lag_index, :] = lag_future.sum(dim=0)
            state_second_moment[:, lag_index, :, :] = torch.einsum(
                "jnh,jnk->jhk", lag_states, lag_states
            )
            future_second_moment[lag_index, :, :] = lag_future.T @ lag_future
            state_future_cross[:, lag_index, :, :] = torch.einsum(
                "jnh,nk->jhk", lag_states, lag_future
            )

        outputs = (
            state_sum,
            future_sum,
            state_second_moment,
            future_second_moment,
            state_future_cross,
        )
        if any(
            value.device.type != "cpu"
            or value.dtype != torch.float64
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
            for value in outputs
        ):
            raise Lfm2AutoregressiveHankelStatisticsError(
                "A Hankel sufficient statistic violated the CPU FP64 contract."
            )

        payload_receipt = _tensor_payload_receipt(
            {
                "lag_pair_counts": lag_pair_counts,
                "right_censor_counts": right_censor_counts,
                "state_sum": state_sum,
                "future_sum": future_sum,
                "state_second_moment": state_second_moment,
                "future_second_moment": future_second_moment,
                "state_future_cross": state_future_cross,
            }
        )
        receipt = _sealed_receipt(
            {
                "schema": LFM2_AUTOREGRESSIVE_HANKEL_STATISTICS_SCHEMA,
                "source_receipt_sha256": source_digest,
                "boundary_count": boundary_count,
                "trajectory_row_count": row_count,
                "hidden_size": hidden_size,
                "future_lags": list(lags),
                "future_lag_count": lag_count,
                "maximum_horizon": lags[-1],
                "dense_future_lag_schedule": lags == tuple(range(lags[-1] + 1)),
                "lag_pair_counts": lag_pair_counts.tolist(),
                "right_censor_counts": right_censor_counts.tolist(),
                "arithmetic_device": "cpu",
                "accumulation_dtype": "float64",
                "terminal_rows_zero_padded": 0,
                "contains_nonzero_future_lags": True,
                "model_wide_residual_boundaries_required": True,
                "static_gram_only_method": False,
                "short_conv_only_fhvr_method": False,
                "tensor_payload": payload_receipt,
                "operation_audit": _zero_execution_audit(),
            }
        )
        return Lfm2AutoregressiveHankelStatistics(
            lag_pair_counts=lag_pair_counts,
            right_censor_counts=right_censor_counts,
            state_sum=state_sum,
            future_sum=future_sum,
            state_second_moment=state_second_moment,
            future_second_moment=future_second_moment,
            state_future_cross=state_future_cross,
            receipt=receipt,
        )


def _layer_indices(inventory: Sequence[str]) -> tuple[int, ...]:
    indices = sorted(
        {
            int(match.group("layer"))
            for name in inventory
            if (match := _PREFILL_LAYER_PRE.fullmatch(name)) is not None
        }
    )
    if not indices or indices != list(range(len(indices))):
        raise Lfm2AutoregressiveHankelStatisticsError(
            "The sealed trace does not expose contiguous model-wide layers."
        )
    return tuple(indices)


def _phase_names(decode_steps: int) -> tuple[str, ...]:
    return (
        "prefill",
        *(f"decode.{index:03d}" for index in range(decode_steps)),
    )


def _required_tensor_names(
    *,
    phases: Sequence[str],
    layer_indices: Sequence[int],
) -> tuple[str, ...]:
    selected = {
        "input.input_ids",
        "input.attention_mask",
        "input.forced_decode_ids",
    }
    for phase in phases:
        selected.add(f"{phase}.layer.000.residual.pre")
        selected.update(
            f"{phase}.layer.{layer:03d}.residual.post_layer"
            for layer in layer_indices
        )
        selected.add(f"{phase}.output.lm_head_input")
    names = tuple(sorted(selected))
    if any("logits" in name for name in names):  # pragma: no cover - construction guard
        raise AssertionError("Hankel extraction must never select logits.")
    return names


def _verify_sealed_residual_chain(
    *,
    record_inventory: Mapping[str, Any],
    phases: Sequence[str],
    layer_indices: Sequence[int],
) -> None:
    """Verify adjacent residual identity from authenticated tensor descriptors."""

    for phase in phases:
        for left, right in zip(layer_indices, layer_indices[1:]):
            post_name = f"{phase}.layer.{left:03d}.residual.post_layer"
            pre_name = f"{phase}.layer.{right:03d}.residual.pre"
            post = record_inventory.get(post_name)
            pre = record_inventory.get(pre_name)
            if (
                not isinstance(post, Mapping)
                or not isinstance(pre, Mapping)
                or dict(post) != dict(pre)
            ):
                raise Lfm2AutoregressiveHankelStatisticsError(
                    "The sealed residual chain is discontinuous at "
                    f"{phase} layer {left}->{right}."
                )


def extract_lfm2_autoregressive_hankel_batch(
    *,
    session: Any,
    reference: Any,
) -> Lfm2AutoregressiveHankelBatch:
    """Load and immediately release one authenticated single-row trace batch.

    The first implementation slice intentionally fails closed for ``B != 1``.
    Production LFM2 evidence uses independent prompt trajectories; preserving
    one trajectory per result also makes terminal censoring and future lags
    unambiguous.  A later aggregate can add these per-batch moments exactly.
    """

    torch = _torch()
    inventory = tuple(getattr(reference, "ordered_tensor_names", ()))
    if (
        not inventory
        or inventory != tuple(sorted(set(inventory)))
        or any(not isinstance(name, str) or not name for name in inventory)
    ):
        raise Lfm2AutoregressiveHankelStatisticsError(
            "reference has no canonical authenticated tensor inventory."
        )
    record = getattr(reference, "record", None)
    if not isinstance(record, Mapping):
        raise Lfm2AutoregressiveHankelStatisticsError(
            "reference has no authenticated batch record."
        )
    capture = record.get("capture")
    record_inventory = record.get("tensors")
    if not isinstance(capture, Mapping) or not isinstance(record_inventory, Mapping):
        raise Lfm2AutoregressiveHankelStatisticsError(
            "The authenticated record lacks capture or tensor descriptors."
        )
    decode_steps = _nonnegative_int(capture.get("decode_steps"), "decode_steps")
    sample_start = _nonnegative_int(
        capture.get("prefill_sample_start"), "prefill_sample_start"
    )
    sample_count = _positive_int(
        capture.get("prefill_sample_count"), "prefill_sample_count"
    )
    layers = _layer_indices(inventory)
    phases = _phase_names(decode_steps)
    names = _required_tensor_names(phases=phases, layer_indices=layers)
    missing = sorted(set(names).difference(inventory))
    if missing:
        raise Lfm2AutoregressiveHankelStatisticsError(
            f"The sealed trace lacks required Hankel endpoints: {missing}."
        )
    _verify_sealed_residual_chain(
        record_inventory=record_inventory,
        phases=phases,
        layer_indices=layers,
    )

    batch_index = _nonnegative_int(getattr(reference, "index", None), "batch index")
    input_sha256 = _digest(
        getattr(reference, "input_sha256", None), "trace input_sha256"
    )
    record_sha256 = _digest(
        getattr(reference, "record_sha256", None), "trace record_sha256"
    )
    session_sha256 = _digest(
        getattr(session, "evidence_sha256", None), "trace session_sha256"
    )
    cache_key = _digest(getattr(session, "cache_key", None), "trace cache_key")

    payload = session.load_selected_tensors(batch_index, names)
    try:
        if (
            getattr(payload, "batch_index", None) != batch_index
            or getattr(payload, "input_sha256", None) != input_sha256
            or tuple(getattr(payload, "ordered_tensor_names", ())) != names
            or getattr(payload, "session_sha256", None) != session_sha256
        ):
            raise Lfm2AutoregressiveHankelStatisticsError(
                "The selected payload is not bound to the requested trace batch."
            )
        selection_sha256 = _digest(
            getattr(payload, "selection_sha256", None),
            "selected payload selection_sha256",
        )
        selected = payload.tensors
        with torch.inference_mode():
            prompt = _cpu_i64(selected["input.input_ids"], "input.input_ids")
            attention = _cpu_i64(
                selected["input.attention_mask"], "input.attention_mask"
            )
            forced = _cpu_i64(
                selected["input.forced_decode_ids"], "input.forced_decode_ids"
            )
            if (
                prompt.ndim != 2
                or tuple(attention.shape) != tuple(prompt.shape)
                or int(prompt.shape[0]) != 1
                or forced.ndim != 2
                or tuple(forced.shape) != (1, decode_steps)
                or not bool((attention == 1).all().item())
            ):
                raise Lfm2AutoregressiveHankelStatisticsError(
                    "Only one canonical unpadded trajectory per batch is supported."
                )
            prompt_length = int(prompt.shape[1])
            if sample_start + sample_count != prompt_length:
                raise Lfm2AutoregressiveHankelStatisticsError(
                    "The sealed prefill sample does not end at the prompt boundary."
                )

            boundary_phase_rows: list[list[Any]] = [
                [] for _ in range(len(layers) + 1)
            ]
            behavior_phase_rows: list[Any] = []
            hidden_size: int | None = None
            for phase_index, phase in enumerate(phases):
                expected_rows = sample_count if phase_index == 0 else 1
                endpoint_names = (
                    f"{phase}.layer.000.residual.pre",
                    *(
                        f"{phase}.layer.{layer:03d}.residual.post_layer"
                        for layer in layers
                    ),
                )
                for boundary_index, endpoint_name in enumerate(endpoint_names):
                    endpoint = _cpu_f64(selected[endpoint_name], endpoint_name)
                    if (
                        endpoint.ndim != 3
                        or int(endpoint.shape[0]) != 1
                        or int(endpoint.shape[1]) != expected_rows
                    ):
                        raise Lfm2AutoregressiveHankelStatisticsError(
                            f"Trace endpoint {endpoint_name!r} has invalid geometry."
                        )
                    width = int(endpoint.shape[2])
                    hidden_size = width if hidden_size is None else hidden_size
                    if width != hidden_size:
                        raise Lfm2AutoregressiveHankelStatisticsError(
                            "Residual boundary hidden widths disagree."
                        )
                    boundary_phase_rows[boundary_index].append(endpoint[0])

                behavior_name = f"{phase}.output.lm_head_input"
                behavior = _cpu_f64(selected[behavior_name], behavior_name)
                if (
                    behavior.ndim != 3
                    or int(behavior.shape[0]) != 1
                    or int(behavior.shape[1]) != expected_rows
                    or int(behavior.shape[2]) != hidden_size
                ):
                    raise Lfm2AutoregressiveHankelStatisticsError(
                        f"Trace endpoint {behavior_name!r} has invalid geometry."
                    )
                behavior_phase_rows.append(behavior[0])

            boundary_rows = torch.stack(
                [torch.cat(rows, dim=0) for rows in boundary_phase_rows], dim=0
            ).contiguous()
            behavior_rows = torch.cat(behavior_phase_rows, dim=0).contiguous()
            state_token_ids = torch.cat(
                (prompt[0, sample_start:], forced[0]), dim=0
            ).contiguous()
            observed_action_token_ids = torch.cat(
                (prompt[0, sample_start + 1 :], forced[0]), dim=0
            ).contiguous()
            prefix_positions = torch.arange(
                sample_start,
                prompt_length + decode_steps,
                dtype=torch.int64,
                device="cpu",
            )
            trajectory_rows = sample_count + decode_steps
            if (
                tuple(boundary_rows.shape)
                != (len(layers) + 1, trajectory_rows, hidden_size)
                or tuple(behavior_rows.shape) != (trajectory_rows, hidden_size)
                or tuple(state_token_ids.shape) != (trajectory_rows,)
                or tuple(observed_action_token_ids.shape)
                != (trajectory_rows - 1,)
                or tuple(prefix_positions.shape) != (trajectory_rows,)
            ):
                raise Lfm2AutoregressiveHankelStatisticsError(
                    "The reconstructed autoregressive trajectory is inconsistent."
                )
            float_outputs = (boundary_rows, behavior_rows)
            if any(
                value.dtype != torch.float64
                or value.device.type != "cpu"
                or value.requires_grad
                or value.grad_fn is not None
                for value in float_outputs
            ):
                raise Lfm2AutoregressiveHankelStatisticsError(
                    "Extracted Hankel rows violated the detached CPU FP64 contract."
                )

            boundary_names = (
                "layer.000.residual.pre",
                *(
                    f"layer.{layer:03d}.residual.post_layer" for layer in layers
                ),
            )
            payload_receipt = _tensor_payload_receipt(
                {
                    "boundary_rows": boundary_rows,
                    "behavior_rows": behavior_rows,
                    "state_token_ids": state_token_ids,
                    "observed_action_token_ids": observed_action_token_ids,
                    "prefix_positions": prefix_positions,
                }
            )
            receipt = _sealed_receipt(
                {
                    "schema": LFM2_AUTOREGRESSIVE_HANKEL_BATCH_SCHEMA,
                    "trace_cache_key": cache_key,
                    "trace_session_sha256": session_sha256,
                    "trace_input_sha256": input_sha256,
                    "trace_record_sha256": record_sha256,
                    "selected_payload_sha256": selection_sha256,
                    "batch_index": batch_index,
                    "selected_tensor_names": list(names),
                    "selected_tensor_count": len(names),
                    "layer_indices": list(layers),
                    "boundary_names": list(boundary_names),
                    "boundary_count": len(boundary_names),
                    "trajectory_row_count": trajectory_rows,
                    "observed_action_count": trajectory_rows - 1,
                    "terminal_rows_without_observed_action": 1,
                    "hidden_size": hidden_size,
                    "prefill_sample_start": sample_start,
                    "prefill_sample_count": sample_count,
                    "decode_steps": decode_steps,
                    "complete_prompt_prefix_coverage": sample_start == 0,
                    "storage_device": "cpu",
                    "storage_dtype": "float64",
                    "model_wide_residual_boundaries_present": True,
                    "future_lag_statistics_required": True,
                    "static_gram_only_method": False,
                    "short_conv_only_fhvr_method": False,
                    "tensor_payload": payload_receipt,
                    "operation_audit": _zero_execution_audit(),
                }
            )
            return Lfm2AutoregressiveHankelBatch(
                boundary_rows=boundary_rows,
                behavior_rows=behavior_rows,
                state_token_ids=state_token_ids,
                observed_action_token_ids=observed_action_token_ids,
                prefix_positions=prefix_positions,
                boundary_names=boundary_names,
                receipt=receipt,
            )
    finally:
        session.release_selected_tensors(payload)


__all__ = [
    "LFM2_AUTOREGRESSIVE_HANKEL_BATCH_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_PAYLOAD_SCHEMA",
    "LFM2_AUTOREGRESSIVE_HANKEL_STATISTICS_SCHEMA",
    "Lfm2AutoregressiveHankelBatch",
    "Lfm2AutoregressiveHankelStatistics",
    "Lfm2AutoregressiveHankelStatisticsError",
    "accumulate_lfm2_ahb_trajectory",
    "extract_lfm2_autoregressive_hankel_batch",
]
