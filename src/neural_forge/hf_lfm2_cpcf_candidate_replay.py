"""Authenticated scheduling seam for LFM2 CPCF target-dtype replay.

The v4 CPCF solver and its closed-form fitting stages remain CPU-only.  This
module isolates the independent *evaluation* of already-materialized
candidate states as :class:`ReplayWorkItem` values.  A caller may evaluate
those work items with the CPU reference implementation or pass them through
the authenticated deterministic two-CUDA-device scheduler.

Scheduled arithmetic is deliberately not presented as the legacy v4 compiler
path.  The caller receives a sealed backend receipt which must be bound into
the explicit v5 fold report by the orchestration layer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from neural_forge.dynamic_replay_scheduler import (
    DeterministicReplayScheduler,
    DynamicReplaySchedulingError,
    ReplayExecutionResult,
    ReplayWorkItem,
    fingerprint_replay_value,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
    Lfm2CpcfShortConvV4Error,
    Lfm2CpcfShortConvV4SolverPolicy,
    Lfm2CpcfV4EvaluationCase,
    Lfm2CpcfV4ProjectionCase,
    _projection_case_values,
    _seal_report,
    _state_sha256,
    _tensor_sha256,
    _torch_dtype,
    evaluate_target_dtype_predictions,
)


LFM2_CPCF_CANDIDATE_REPLAY_WORK_SCHEMA = (
    "neural-forge-lfm2-cpcf-candidate-replay-work-v1"
)
LFM2_CPCF_CANDIDATE_REPLAY_OUTPUT_SCHEMA = (
    "neural-forge-lfm2-cpcf-candidate-replay-output-v1"
)
LFM2_CPCF_CANDIDATE_REPLAY_BACKEND_SCHEMA = (
    "neural-forge-lfm2-cpcf-candidate-replay-backend-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPUTE_DEVICE = re.compile(r"^(?:cpu|cuda:[0-9]+)$")
_STATE_ROLES = (
    "in_proj_weight",
    "in_proj_bias",
    "fir_kernel",
    "fir_bias",
    "writer_weight",
    "writer_bias",
)
_REPLAY_STATE_ROLES = (
    "b_history",
    "c_history",
    "x_history",
    "context_history",
    "write_history",
)


class Lfm2CpcfCandidateReplayError(RuntimeError):
    """Candidate replay work, output, or backend authentication failed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF candidate replay."
        ) from exc
    return torch


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfCandidateReplayError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CpcfCandidateReplayError(f"{name} must be nonempty text.")
    return value.strip()


def _sealed_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfCandidateReplayError(f"{name} must be a mapping.")
    result = dict(value)
    body = dict(result)
    supplied = body.pop("report_sha256", None)
    if supplied != _sha(body):
        raise Lfm2CpcfCandidateReplayError(f"{name} seal is invalid.")
    return result


def _snapshot_target_state(
    state: Mapping[str, Any],
    *,
    target_dtype: str,
) -> Mapping[str, Any]:
    torch = _torch()
    if not isinstance(state, Mapping) or set(state) != set(_STATE_ROLES):
        raise Lfm2CpcfCandidateReplayError(
            "A candidate state must contain exactly the six CPCF parameter roles."
        )
    dtype = _torch_dtype(target_dtype)
    result: dict[str, Any] = {}
    for role in _STATE_ROLES:
        value = state[role]
        if value is None:
            if role not in {"in_proj_bias", "fir_bias", "writer_bias"}:
                raise Lfm2CpcfCandidateReplayError(
                    f"{role} may not be absent."
                )
            result[role] = None
            continue
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type == "meta"
            or value.numel() == 0
            or not torch.is_floating_point(value)
            or value.is_complex()
            or value.requires_grad
            or value.grad_fn is not None
        ):
            raise Lfm2CpcfCandidateReplayError(
                f"{role} must be a detached dense real tensor."
            )
        tensor = value.detach().to(device="cpu", dtype=dtype).contiguous()
        if not bool(torch.isfinite(tensor).all().item()):
            raise Lfm2CpcfCandidateReplayError(
                f"{role} contains NaN or infinity."
            )
        result[role] = tensor
    return MappingProxyType(result)


def _case_payload(
    case: Lfm2CpcfV4ProjectionCase,
    *,
    required_split: str,
) -> Mapping[str, Any]:
    child, _context, _write, prefix_mask, _suffix_mask = (
        _projection_case_values(case, required_split=required_split)
    )
    return MappingProxyType(
        {
            "case_id": case.case_id,
            "split": case.split,
            "suffix_start": case.suffix_start,
            "projection_case_authentication_sha256": (
                case.authentication_sha256
            ),
            "child_input_history": child,
            "valid_prefix_mask": prefix_mask,
        }
    )


def build_lfm2_cpcf_candidate_replay_work_items(
    candidate_states: Mapping[str, Mapping[str, Any]],
    *,
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    required_split: str,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> tuple[ReplayWorkItem, ...]:
    """Seal unique candidate states and shared cases into canonical work."""

    if not isinstance(policy, Lfm2CpcfShortConvV4SolverPolicy):
        raise TypeError("policy must be Lfm2CpcfShortConvV4SolverPolicy.")
    if not policy.verify_authentication():
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay policy authentication failed."
        )
    split = _text(required_split, "required_split").lower()
    if split not in {"fit", "selection", "validation"}:
        raise Lfm2CpcfCandidateReplayError(
            "required_split must be fit, selection, or validation."
        )
    if (
        not isinstance(candidate_states, Mapping)
        or not candidate_states
        or any(
            not isinstance(candidate_id, str) or not candidate_id.strip()
            for candidate_id in candidate_states
        )
    ):
        raise Lfm2CpcfCandidateReplayError(
            "candidate_states must be a nonempty mapping with text IDs."
        )
    ordered_cases = tuple(sorted(cases, key=lambda value: value.case_id))
    if (
        not ordered_cases
        or len({case.case_id for case in ordered_cases}) != len(ordered_cases)
    ):
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay requires unique projection cases."
        )
    case_payloads = tuple(
        _case_payload(case, required_split=split) for case in ordered_cases
    )
    work = []
    for ordinal, candidate_id in enumerate(sorted(candidate_states)):
        state = _snapshot_target_state(
            candidate_states[candidate_id],
            target_dtype=policy.target_dtype,
        )
        state_sha256 = _state_sha256(state, "target_dtype_candidate")
        payload = MappingProxyType(
            {
                "schema": LFM2_CPCF_CANDIDATE_REPLAY_WORK_SCHEMA,
                "candidate_id": candidate_id,
                "candidate_state_sha256": state_sha256,
                "policy_sha256": policy.authentication_sha256,
                "target_dtype": policy.target_dtype,
                "required_split": split,
                "state": state,
                "cases": case_payloads,
            }
        )
        work.append(
            ReplayWorkItem.create(
                ordinal=ordinal,
                work_id=f"lfm2-cpcf-candidate::{candidate_id}",
                shard_key=f"lfm2-cpcf-candidate::{candidate_id}",
                payload=payload,
            )
        )
    return tuple(work)


def _same_device_tensor(value: Any, *, device: Any, name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device != device
        or value.numel() == 0
        or value.requires_grad
        or value.grad_fn is not None
    ):
        raise Lfm2CpcfCandidateReplayError(
            f"{name} must be a nonempty detached dense tensor on {device}."
        )
    return value


def replay_lfm2_cpcf_candidate_work_payload(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Execute one candidate on its payload device without fitting anything."""

    torch = _torch()
    if not isinstance(payload, Mapping):
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay payload must be a mapping."
        )
    if payload.get("schema") != LFM2_CPCF_CANDIDATE_REPLAY_WORK_SCHEMA:
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay payload schema is unsupported."
        )
    candidate_id = _text(payload.get("candidate_id"), "candidate_id")
    candidate_sha = _digest(
        payload.get("candidate_state_sha256"),
        "candidate_state_sha256",
    )
    policy_sha = _digest(payload.get("policy_sha256"), "policy_sha256")
    target_dtype = _text(payload.get("target_dtype"), "target_dtype")
    split = _text(payload.get("required_split"), "required_split")
    dtype = _torch_dtype(target_dtype)
    state = payload.get("state")
    cases = payload.get("cases")
    if not isinstance(state, Mapping) or set(state) != set(_STATE_ROLES):
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay state roles are incomplete."
        )
    if not isinstance(cases, (tuple, list)) or not cases:
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay payload has no cases."
        )
    required = state["in_proj_weight"]
    if not isinstance(required, torch.Tensor):
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay in_proj_weight is not a tensor."
        )
    device = required.device
    normalized_state: dict[str, Any] = {}
    for role in _STATE_ROLES:
        value = state[role]
        if value is None:
            normalized_state[role] = None
            continue
        tensor = _same_device_tensor(value, device=device, name=role)
        if tensor.dtype != dtype:
            raise Lfm2CpcfCandidateReplayError(
                f"{role} is not materialized at target dtype {target_dtype}."
            )
        normalized_state[role] = tensor
    if (
        _state_sha256(normalized_state, "target_dtype_candidate")
        != candidate_sha
    ):
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay state differs from its authenticated digest."
        )

    in_weight = normalized_state["in_proj_weight"]
    in_bias = normalized_state["in_proj_bias"]
    kernel = normalized_state["fir_kernel"]
    fir_bias = normalized_state["fir_bias"]
    writer = normalized_state["writer_weight"]
    writer_bias = normalized_state["writer_bias"]
    if (
        in_weight.ndim != 2
        or int(in_weight.shape[0]) != 3 * int(in_weight.shape[1])
        or kernel.ndim != 2
        or writer.ndim != 2
    ):
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay parameter geometry is invalid."
        )
    target_hidden = int(in_weight.shape[1])
    if (
        int(kernel.shape[0]) != target_hidden
        or int(writer.shape[1]) != target_hidden
    ):
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay parameter widths disagree."
        )

    output_cases = []
    seen_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise Lfm2CpcfCandidateReplayError(
                "Candidate replay case payload must be a mapping."
            )
        case_id = _text(case.get("case_id"), "case_id")
        if case_id in seen_case_ids:
            raise Lfm2CpcfCandidateReplayError(
                "Candidate replay case IDs must be unique."
            )
        seen_case_ids.add(case_id)
        if case.get("split") != split:
            raise Lfm2CpcfCandidateReplayError(
                "Candidate replay case split differs from the work item."
            )
        projection_sha = _digest(
            case.get("projection_case_authentication_sha256"),
            "projection_case_authentication_sha256",
        )
        suffix_start = case.get("suffix_start")
        if (
            isinstance(suffix_start, bool)
            or not isinstance(suffix_start, int)
            or suffix_start < 0
        ):
            raise Lfm2CpcfCandidateReplayError(
                "suffix_start must be a nonnegative integer."
            )
        child_source = _same_device_tensor(
            case.get("child_input_history"),
            device=device,
            name=f"{case_id}.child_input_history",
        )
        mask = _same_device_tensor(
            case.get("valid_prefix_mask"),
            device=device,
            name=f"{case_id}.valid_prefix_mask",
        )
        if (
            child_source.ndim != 3
            or int(child_source.shape[-1]) != target_hidden
            or mask.dtype != torch.bool
            or tuple(mask.shape) != tuple(child_source.shape[:2])
            or suffix_start >= int(child_source.shape[1])
        ):
            raise Lfm2CpcfCandidateReplayError(
                f"{case_id} replay geometry is invalid."
            )
        child = child_source.to(dtype=dtype)
        child = child * mask.unsqueeze(-1).to(dtype=dtype)
        projection = child @ in_weight.T
        if in_bias is not None:
            projection = projection + in_bias
        b_value, c_value, x_value = projection.chunk(3, dim=-1)
        fir_length = int(kernel.shape[1])
        gated = (b_value * x_value).transpose(1, 2).contiguous()
        padded = torch.nn.functional.pad(gated, (fir_length - 1, 0))
        filtered = torch.nn.functional.conv1d(
            padded,
            kernel.reshape(target_hidden, 1, fir_length),
            bias=fir_bias,
            groups=target_hidden,
        )
        context = (c_value * filtered.transpose(1, 2)).contiguous()
        write = context @ writer.T
        if writer_bias is not None:
            write = write + writer_bias
        replay_state = {
            "b_history": b_value.contiguous(),
            "c_history": c_value.contiguous(),
            "x_history": x_value.contiguous(),
            "context_history": context,
            "write_history": write.contiguous(),
        }
        if any(
            not bool(torch.isfinite(value).all().item())
            for value in replay_state.values()
        ):
            raise Lfm2CpcfCandidateReplayError(
                f"{case_id} replay produced NaN or infinity."
            )
        output_cases.append(
            MappingProxyType(
                {
                    "case_id": case_id,
                    "split": split,
                    "suffix_start": suffix_start,
                    "projection_case_authentication_sha256": projection_sha,
                    **replay_state,
                }
            )
        )
    return MappingProxyType(
        {
            "schema": LFM2_CPCF_CANDIDATE_REPLAY_OUTPUT_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_state_sha256": candidate_sha,
            "policy_sha256": policy_sha,
            "target_dtype": target_dtype,
            "required_split": split,
            "cases": tuple(output_cases),
        }
    )


def lfm2_cpcf_cuda_replay_worker(
    _item: ReplayWorkItem,
    _replica: Any,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Exact source-bound CUDA worker accepted by the production v5 seam."""

    return replay_lfm2_cpcf_candidate_work_payload(payload)


def lfm2_cpcf_cpu_parity_worker(
    _item: ReplayWorkItem,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Exact source-bound CPU parity worker accepted by production v5."""

    return replay_lfm2_cpcf_candidate_work_payload(payload)


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCandidateReplayBatchResult:
    """Canonical replay outputs plus their authenticated backend receipt."""

    outputs: tuple[Mapping[str, Any], ...]
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        outputs = tuple(self.outputs)
        report = _sealed_mapping(self.report, "candidate replay backend report")
        work_ids = report.get("canonical_output_work_ids")
        output_hashes = report.get("canonical_output_sha256")
        if (
            not outputs
            or not isinstance(work_ids, list)
            or len(work_ids) != len(outputs)
            or not isinstance(output_hashes, list)
            or output_hashes
            != [fingerprint_replay_value(value) for value in outputs]
        ):
            raise Lfm2CpcfCandidateReplayError(
                "Candidate replay outputs differ from their backend receipt."
            )
        devices = report.get("compute_device_by_work_id")
        if (
            not isinstance(devices, Mapping)
            or set(devices) != set(work_ids)
            or any(
                not isinstance(value, str)
                or _COMPUTE_DEVICE.fullmatch(value) is None
                for value in devices.values()
            )
        ):
            raise Lfm2CpcfCandidateReplayError(
                "Candidate replay compute-device receipt is invalid."
            )
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "report", MappingProxyType(report))

    @classmethod
    def create(
        cls,
        *,
        items: Sequence[ReplayWorkItem],
        outputs: Sequence[Mapping[str, Any]],
        compute_device_by_work_id: Mapping[str, str],
        backend_kind: str,
        execution_report: Mapping[str, Any],
        representative_cpu_cuda_parity_passed: bool,
    ) -> Lfm2CpcfCandidateReplayBatchResult:
        ordered_items = tuple(sorted(items, key=lambda value: value.ordinal))
        values = tuple(outputs)
        work_ids = [item.work_id for item in ordered_items]
        if (
            not ordered_items
            or tuple(item.ordinal for item in ordered_items)
            != tuple(range(len(ordered_items)))
            or len(values) != len(ordered_items)
            or set(compute_device_by_work_id) != set(work_ids)
            or not isinstance(representative_cpu_cuda_parity_passed, bool)
        ):
            raise Lfm2CpcfCandidateReplayError(
                "Candidate replay batch is incomplete or noncanonical."
            )
        for item, output in zip(ordered_items, values, strict=True):
            if not isinstance(output, Mapping):
                raise Lfm2CpcfCandidateReplayError(
                    "Candidate replay outputs must be mappings."
                )
            if (
                output.get("schema")
                != LFM2_CPCF_CANDIDATE_REPLAY_OUTPUT_SCHEMA
                or output.get("candidate_id")
                != item.payload.get("candidate_id")
                or output.get("candidate_state_sha256")
                != item.payload.get("candidate_state_sha256")
                or output.get("policy_sha256")
                != item.payload.get("policy_sha256")
                or output.get("required_split")
                != item.payload.get("required_split")
            ):
                raise Lfm2CpcfCandidateReplayError(
                    f"Replay output does not match work item {item.work_id!r}."
                )
        body = {
            "schema": LFM2_CPCF_CANDIDATE_REPLAY_BACKEND_SCHEMA,
            "kind": _text(backend_kind, "backend_kind"),
            "canonical_output_work_ids": work_ids,
            "canonical_input_payload_sha256": [
                item.payload_sha256 for item in ordered_items
            ],
            "canonical_output_sha256": [
                fingerprint_replay_value(value) for value in values
            ],
            "compute_device_by_work_id": {
                work_id: compute_device_by_work_id[work_id]
                for work_id in work_ids
            },
            "execution_report": json.loads(
                _canonical_json(execution_report).decode("ascii")
            ),
            "representative_cpu_cuda_parity_passed": (
                representative_cpu_cuda_parity_passed
            ),
            "canonical_candidate_order_restored": True,
            "closed_form_solve_math_changed": False,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }
        return cls(
            outputs=values,
            report=MappingProxyType({**body, "report_sha256": _sha(body)}),
        )


class Lfm2CpcfCandidateReplayEvaluator(Protocol):
    """Injected evaluator accepted by the v5 CPCF orchestration path."""

    def evaluate(
        self,
        items: Sequence[ReplayWorkItem],
    ) -> Lfm2CpcfCandidateReplayBatchResult: ...


class CpuReferenceLfm2CpcfCandidateReplayEvaluator:
    """CPU evaluator used for equivalence tests and explicit diagnostics."""

    def evaluate(
        self,
        items: Sequence[ReplayWorkItem],
    ) -> Lfm2CpcfCandidateReplayBatchResult:
        ordered = tuple(sorted(items, key=lambda value: value.ordinal))
        outputs = tuple(
            replay_lfm2_cpcf_candidate_work_payload(item.payload)
            for item in ordered
        )
        work_ids = [item.work_id for item in ordered]
        execution = {
            "schema": "neural-forge-lfm2-cpcf-injected-cpu-reference-v1",
            "work_item_count": len(ordered),
            "canonical_output_work_ids": work_ids,
            "no_cuda_claimed": True,
        }
        return Lfm2CpcfCandidateReplayBatchResult.create(
            items=ordered,
            outputs=outputs,
            compute_device_by_work_id={
                work_id: "cpu" for work_id in work_ids
            },
            backend_kind="injected_cpu_reference",
            execution_report=execution,
            representative_cpu_cuda_parity_passed=False,
        )


class AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator:
    """Adapter from CPCF work items to the deterministic replay scheduler."""

    def __init__(self, scheduler: DeterministicReplayScheduler) -> None:
        if not isinstance(scheduler, DeterministicReplayScheduler):
            raise TypeError("scheduler must be DeterministicReplayScheduler.")
        self.scheduler = scheduler

    def evaluate(
        self,
        items: Sequence[ReplayWorkItem],
    ) -> Lfm2CpcfCandidateReplayBatchResult:
        ordered = tuple(sorted(items, key=lambda value: value.ordinal))
        if not ordered:
            raise Lfm2CpcfCandidateReplayError(
                "Scheduled candidate replay cannot be empty."
            )
        parity_item = ReplayWorkItem.create(
            ordinal=0,
            work_id="lfm2-cpcf-candidate::representative-parity",
            shard_key="lfm2-cpcf-candidate::representative-parity",
            payload=ordered[0].payload,
        )
        try:
            execution = self.scheduler.execute(
                ordered,
                parity_item=parity_item,
            )
        except DynamicReplaySchedulingError as exc:
            raise Lfm2CpcfCandidateReplayError(
                "Authenticated two-GPU candidate replay failed closed."
            ) from exc
        if not isinstance(execution, ReplayExecutionResult):
            raise Lfm2CpcfCandidateReplayError(
                "Replay scheduler returned an unsupported result."
            )
        records = execution.report.get("scheduled_records")
        if not isinstance(records, list):
            raise Lfm2CpcfCandidateReplayError(
                "Replay scheduler omitted assignment records."
            )
        devices = {
            str(record["work_id"]): f"cuda:{int(record['device_index'])}"
            for record in records
        }
        return Lfm2CpcfCandidateReplayBatchResult.create(
            items=ordered,
            outputs=execution.outputs,
            compute_device_by_work_id=devices,
            backend_kind="authenticated_two_cuda_scheduler",
            execution_report=execution.as_dict(),
            representative_cpu_cuda_parity_passed=True,
        )


def _evaluation_arithmetic_report(
    *,
    replay_case: Mapping[str, Any],
    candidate_state_sha256: str,
    projection_case_authentication_sha256: str,
    suffix_start: int,
    target_dtype: str,
    compute_device: str,
) -> Mapping[str, Any]:
    replay_state = {
        role: replay_case[role] for role in _REPLAY_STATE_ROLES
    }
    prediction = replay_case["write_history"][
        :, suffix_start:, :
    ].contiguous()
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
        "kind": "sampled_cp_target_dtype_replay",
        "compute_device": compute_device,
        "parameter_storage_dtype": target_dtype,
        "intermediate_arithmetic_dtype": target_dtype,
        "parameters_materialized_before_replay": True,
        "intermediates_computed_at_declared_dtype": True,
        "final_output_cast_only": False,
        "valid_prefix_mask_applied_before_projection": True,
        "invalid_projected_history_zeroed_after_affine": False,
        "biased_projection_matches_lfm2_mask_before_linear": True,
        "parameter_state_sha256": candidate_state_sha256,
        "replay_state_sha256": _state_sha256(replay_state, "replay"),
        "projection_case_authentication_sha256": (
            projection_case_authentication_sha256
        ),
        "evaluated_suffix_start": suffix_start,
        "evaluated_prediction_sha256": _tensor_sha256(
            prediction,
            f"{replay_case['case_id']}.evaluated_prediction",
        ),
        "evaluated_context_prediction_sha256": _tensor_sha256(
            replay_case["context_history"],
            f"{replay_case['case_id']}.evaluated_context_prediction",
        ),
    }
    return _seal_report(body)


def finalize_lfm2_cpcf_candidate_replay_reports(
    *,
    items: Sequence[ReplayWorkItem],
    result: Lfm2CpcfCandidateReplayBatchResult,
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    required_split: str,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> Mapping[str, Mapping[str, Any]]:
    """Turn authenticated raw replay trees into ordinary v4 metric reports."""

    if not isinstance(result, Lfm2CpcfCandidateReplayBatchResult):
        raise TypeError(
            "result must be Lfm2CpcfCandidateReplayBatchResult."
        )
    if not isinstance(policy, Lfm2CpcfShortConvV4SolverPolicy):
        raise TypeError("policy must be Lfm2CpcfShortConvV4SolverPolicy.")
    if not policy.verify_authentication():
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay policy authentication failed."
        )
    split = _text(required_split, "required_split").lower()
    ordered_items = tuple(sorted(items, key=lambda value: value.ordinal))
    if (
        len(ordered_items) != len(result.outputs)
        or result.report.get("canonical_output_work_ids")
        != [item.work_id for item in ordered_items]
    ):
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay result order differs from its work items."
        )
    ordered_cases = tuple(sorted(cases, key=lambda value: value.case_id))
    cases_by_id = {case.case_id: case for case in ordered_cases}
    if len(cases_by_id) != len(ordered_cases) or not ordered_cases:
        raise Lfm2CpcfCandidateReplayError(
            "Candidate replay finalization requires unique cases."
        )
    reports: dict[str, Mapping[str, Any]] = {}
    device_by_work = result.report["compute_device_by_work_id"]
    for item, output in zip(
        ordered_items,
        result.outputs,
        strict=True,
    ):
        if (
            fingerprint_replay_value(
                item.payload,
                require_cpu_tensors=True,
            )
            != item.payload_sha256
        ):
            raise Lfm2CpcfCandidateReplayError(
                f"Candidate work item {item.work_id!r} changed."
            )
        candidate_id = output.get("candidate_id")
        candidate_sha = _digest(
            output.get("candidate_state_sha256"),
            "candidate_state_sha256",
        )
        raw_cases = output.get("cases")
        if (
            candidate_id != item.payload.get("candidate_id")
            or output.get("policy_sha256") != policy.authentication_sha256
            or output.get("target_dtype") != policy.target_dtype
            or output.get("required_split") != split
            or not isinstance(raw_cases, (tuple, list))
            or len(raw_cases) != len(ordered_cases)
        ):
            raise Lfm2CpcfCandidateReplayError(
                f"Candidate replay output {item.work_id!r} is incomplete."
            )
        evaluation_cases = []
        for replay_case in raw_cases:
            if not isinstance(replay_case, Mapping):
                raise Lfm2CpcfCandidateReplayError(
                    "Candidate replay case output must be a mapping."
                )
            case_id = _text(replay_case.get("case_id"), "case_id")
            case = cases_by_id.get(case_id)
            if case is None:
                raise Lfm2CpcfCandidateReplayError(
                    f"Candidate replay returned unknown case {case_id!r}."
                )
            (
                _child,
                desired_context,
                desired_write,
                prefix_mask,
                suffix_mask,
            ) = _projection_case_values(case, required_split=split)
            if (
                replay_case.get("split") != split
                or replay_case.get("suffix_start") != case.suffix_start
                or replay_case.get(
                    "projection_case_authentication_sha256"
                )
                != case.authentication_sha256
                or any(role not in replay_case for role in _REPLAY_STATE_ROLES)
            ):
                raise Lfm2CpcfCandidateReplayError(
                    f"Candidate replay case {case_id!r} lost its evidence binding."
                )
            prediction = replay_case["write_history"][
                :, case.suffix_start:, :
            ].contiguous()
            arithmetic = _evaluation_arithmetic_report(
                replay_case=replay_case,
                candidate_state_sha256=candidate_sha,
                projection_case_authentication_sha256=(
                    case.authentication_sha256
                ),
                suffix_start=case.suffix_start,
                target_dtype=policy.target_dtype,
                compute_device=device_by_work[item.work_id],
            )
            evaluation_cases.append(
                Lfm2CpcfV4EvaluationCase(
                    case_id=case.case_id,
                    split=case.split,
                    prediction=prediction,
                    target=desired_write,
                    valid_rows=suffix_mask,
                    candidate_state_sha256=candidate_sha,
                    projection_case_authentication_sha256=(
                        case.authentication_sha256
                    ),
                    context_prediction=replay_case["context_history"],
                    context_target=desired_context,
                    valid_context_rows=prefix_mask,
                    arithmetic_report=arithmetic,
                )
            )
        reports[str(candidate_id)] = evaluate_target_dtype_predictions(
            evaluation_cases,
            target_dtype=policy.target_dtype,
            required_split=split,
            policy_sha256=policy.authentication_sha256,
        )
    return MappingProxyType(reports)


def evaluate_lfm2_cpcf_candidate_states_with_backend(
    candidate_states: Mapping[str, Mapping[str, Any]],
    *,
    cases: Sequence[Lfm2CpcfV4ProjectionCase],
    required_split: str,
    policy: Lfm2CpcfShortConvV4SolverPolicy,
    evaluator: Lfm2CpcfCandidateReplayEvaluator,
) -> tuple[
    Mapping[str, Mapping[str, Any]],
    Mapping[str, Any],
    tuple[ReplayWorkItem, ...],
]:
    """Build, execute, authenticate, and finalize one candidate frontier."""

    if evaluator is None or not callable(getattr(evaluator, "evaluate", None)):
        raise TypeError(
            "evaluator must implement the candidate replay evaluator protocol."
        )
    items = build_lfm2_cpcf_candidate_replay_work_items(
        candidate_states,
        cases=cases,
        required_split=required_split,
        policy=policy,
    )
    try:
        result = evaluator.evaluate(items)
    except (
        Lfm2CpcfCandidateReplayError,
        DynamicReplaySchedulingError,
        Lfm2CpcfShortConvV4Error,
    ):
        raise
    except BaseException as exc:
        raise Lfm2CpcfCandidateReplayError(
            "Injected candidate replay backend failed before publication."
        ) from exc
    reports = finalize_lfm2_cpcf_candidate_replay_reports(
        items=items,
        result=result,
        cases=cases,
        required_split=required_split,
        policy=policy,
    )
    return reports, result.report, items


__all__ = [
    "LFM2_CPCF_CANDIDATE_REPLAY_BACKEND_SCHEMA",
    "LFM2_CPCF_CANDIDATE_REPLAY_OUTPUT_SCHEMA",
    "LFM2_CPCF_CANDIDATE_REPLAY_WORK_SCHEMA",
    "AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator",
    "CpuReferenceLfm2CpcfCandidateReplayEvaluator",
    "Lfm2CpcfCandidateReplayBatchResult",
    "Lfm2CpcfCandidateReplayError",
    "Lfm2CpcfCandidateReplayEvaluator",
    "build_lfm2_cpcf_candidate_replay_work_items",
    "evaluate_lfm2_cpcf_candidate_states_with_backend",
    "finalize_lfm2_cpcf_candidate_replay_reports",
    "lfm2_cpcf_cpu_parity_worker",
    "lfm2_cpcf_cuda_replay_worker",
    "replay_lfm2_cpcf_candidate_work_payload",
]
