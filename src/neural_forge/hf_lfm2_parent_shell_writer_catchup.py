"""Pure parent-shell writer folding from authenticated cached tensors.

This module intentionally has no model-facing API.  It answers two narrowly
separated diagnostic questions with deterministic CPU FP64 algebra:

* can a compact ShortConv context write the parent-width *local action*; and
* after that writer is replayed, can exactly one downstream writer absorb the
  remaining *absolute endpoint* defect?

Both compilers solve a prior-centred weighted ridge problem.  FIT evidence is
the only construction input.  Held-out evidence is evaluated only after the
dense FP32 writer has been materialized and cannot select, reject, scale, or
otherwise change it.  The functions construct no model, execute no forward,
perform no iterative optimization, and touch neither the network nor the
filesystem.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from neural_forge.dynamic_circuit_ir import canonical_sha256


LFM2_PARENT_SHELL_LOCAL_WRITER_SCHEMA = "neural-forge-lfm2-parent-shell-local-writer-v1"
LFM2_PARENT_SHELL_CATCHUP_WRITER_SCHEMA = (
    "neural-forge-lfm2-parent-shell-one-downstream-catchup-writer-v1"
)
LFM2_PARENT_SHELL_WRITER_STATE_SCHEMA = "neural-forge-lfm2-parent-shell-writer-state-v1"
LFM2_PARENT_SHELL_LOCAL_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-parent-shell-local-writer-evidence-v1"
)
LFM2_PARENT_SHELL_CATCHUP_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-parent-shell-catchup-writer-evidence-v1"
)
LFM2_PARENT_SHELL_CATCHUP_REPLAY_BINDING_SCHEMA = (
    "neural-forge-lfm2-parent-shell-catchup-replay-binding-v1"
)

Lfm2ParentShellEvidencePartition = Literal["fit", "heldout"]
Lfm2ParentShellWriterRole = Literal[
    "compact-short-conv-local-output",
    "one-downstream-absolute-endpoint-catchup",
]

_LOCAL_ROLE: Lfm2ParentShellWriterRole = "compact-short-conv-local-output"
_CATCHUP_ROLE: Lfm2ParentShellWriterRole = "one-downstream-absolute-endpoint-catchup"
_OPERATION_AUDIT = MappingProxyType(
    {
        "model_objects_constructed": 0,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "parent_trace_recaptures": 0,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parameters_sparsified": 0,
        "parameters_shared": 0,
        "network_calls": 0,
        "filesystem_reads": 0,
        "filesystem_writes": 0,
    }
)


class Lfm2ParentShellWriterCatchupError(RuntimeError):
    """Authenticated parent-shell writer compilation failed closed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("Parent-shell writer compilation requires PyTorch.") from exc
    return torch


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ParentShellWriterCatchupError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Lfm2ParentShellWriterCatchupError(
            f"{label} must be a nonempty canonical identity."
        )
    return value


def _cached_tensor(value: Any, label: str, *, ndim: int) -> Any:
    """Copy one exact, detached, CPU, dense cached tensor."""

    torch = _torch()
    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.ndim != ndim
        or value.numel() == 0
        or value.dtype not in {torch.float32, torch.float64}
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_leaf
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise Lfm2ParentShellWriterCatchupError(
            f"{label} must be an exact detached finite contiguous CPU FP32/FP64 "
            f"rank-{ndim} cached tensor."
        )
    return value.detach().clone(memory_format=torch.contiguous_format)


def _fp32_prior(value: Any, label: str, *, ndim: int) -> Any:
    torch = _torch()
    tensor = _cached_tensor(value, label, ndim=ndim)
    if tensor.dtype != torch.float32:
        raise Lfm2ParentShellWriterCatchupError(
            f"{label} must be a dense FP32 structural prior."
        )
    return tensor


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


def _validate_partition(value: str) -> Lfm2ParentShellEvidencePartition:
    if value not in {"fit", "heldout"}:
        raise Lfm2ParentShellWriterCatchupError(
            "Evidence partition must be exactly 'fit' or 'heldout'."
        )
    return value  # type: ignore[return-value]


def _validate_ridge(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise Lfm2ParentShellWriterCatchupError(
            "ridge must be a finite strictly positive scalar."
        )
    return float(value)


def _row_ids(
    supplied: Sequence[str],
    *,
    row_count: int,
    label: str,
) -> tuple[str, ...]:
    if isinstance(supplied, (str, bytes)):
        raise Lfm2ParentShellWriterCatchupError(
            f"{label} must be an explicit sequence of stable row IDs."
        )
    try:
        values = tuple(supplied)
    except TypeError as exc:
        raise Lfm2ParentShellWriterCatchupError(
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
        raise Lfm2ParentShellWriterCatchupError(
            f"{label} must contain one unique nonempty canonical ID per row."
        )
    return values


def _assert_independent_tensors(tensors: Mapping[str, Any]) -> None:
    torch = _torch()
    values = list(tensors.items())
    for name, tensor in values:
        if (
            type(tensor) is not torch.Tensor
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not tensor.is_leaf
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise Lfm2ParentShellWriterCatchupError(
                f"Result tensor {name} is not an independent detached dense CPU tensor."
            )
    for left_index, (left_name, left) in enumerate(values):
        for right_name, right in values[left_index + 1 :]:
            if left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr():
                raise Lfm2ParentShellWriterCatchupError(
                    f"Result tensors {left_name} and {right_name} alias storage."
                )


@dataclass(frozen=True, slots=True)
class Lfm2ParentShellLocalWriterEvidenceCase:
    """One row-aligned cached trajectory for the compact local writer."""

    case_id: str
    compact_writer_design: Any = field(repr=False, compare=False)
    parent_local_action: Any = field(repr=False, compare=False)
    parent_pre_operator_residual: Any = field(repr=False, compare=False)
    parent_post_operator_residual: Any = field(repr=False, compare=False)
    row_sample_weights: Any = field(repr=False, compare=False)
    row_ids: Sequence[str] = field(repr=False, compare=False)
    parent_trajectory_receipt_sha256: str
    compact_replay_receipt_sha256: str
    aligned_row_identity_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2ParentShellCatchupEvidenceCase:
    """One post-local-replay trajectory for one downstream endpoint writer."""

    case_id: str
    downstream_writer_design: Any = field(repr=False, compare=False)
    live_pre_downstream_residual: Any = field(repr=False, compare=False)
    parent_absolute_endpoint: Any = field(repr=False, compare=False)
    row_sample_weights: Any = field(repr=False, compare=False)
    row_ids: Sequence[str] = field(repr=False, compare=False)
    parent_trajectory_receipt_sha256: str
    post_local_replay_receipt_sha256: str
    aligned_row_identity_sha256: str


@dataclass(frozen=True, slots=True)
class Lfm2CompiledParentShellWriter:
    """One independent dense FP32 writer plus its authenticated receipt."""

    writer_id: str
    writer_role: Lfm2ParentShellWriterRole
    weight_structural_prior: Any = field(repr=False, compare=False)
    bias_structural_prior: Any | None = field(repr=False, compare=False)
    weight_compiled: Any = field(repr=False, compare=False)
    bias_compiled: Any | None = field(repr=False, compare=False)
    weight_delta_fp64: Any = field(repr=False, compare=False)
    bias_delta_fp64: Any | None = field(repr=False, compare=False)
    report: Mapping[str, Any]

    def state_dict(self) -> Mapping[str, Any]:
        self.verify()
        state = {"weight": self.weight_compiled.detach().cpu().contiguous().clone()}
        if self.bias_compiled is not None:
            state["bias"] = self.bias_compiled.detach().cpu().contiguous().clone()
        return MappingProxyType(state)

    def structural_prior_state_dict(self) -> Mapping[str, Any]:
        self.verify()
        state = {
            "weight": self.weight_structural_prior.detach().cpu().contiguous().clone()
        }
        if self.bias_structural_prior is not None:
            state["bias"] = (
                self.bias_structural_prior.detach().cpu().contiguous().clone()
            )
        return MappingProxyType(state)

    def as_dict(self) -> dict[str, Any]:
        self.verify()
        return copy.deepcopy(dict(self.report))

    def verify(self) -> str:
        body = copy.deepcopy(dict(self.report))
        seal = body.pop("result_sha256", None)
        if seal != canonical_sha256(body):
            raise Lfm2ParentShellWriterCatchupError(
                "Compiled parent-shell writer receipt seal is invalid."
            )
        if (
            body.get("writer_id") != self.writer_id
            or body.get("writer_role") != self.writer_role
        ):
            raise Lfm2ParentShellWriterCatchupError(
                "Compiled parent-shell writer identity or role changed after sealing."
            )
        authentication = body.get("authentication")
        topology = body.get("topology")
        if not isinstance(authentication, Mapping) or not isinstance(topology, Mapping):
            raise Lfm2ParentShellWriterCatchupError(
                "Compiled writer receipt lacks authentication or topology."
            )
        actual = lfm2_parent_shell_writer_state_sha256(
            writer_id=self.writer_id,
            writer_role=self.writer_role,
            weight=self.weight_compiled,
            bias=self.bias_compiled,
        )
        expected = authentication.get("compiled_writer_state_sha256")
        if actual != expected:
            raise Lfm2ParentShellWriterCatchupError(
                "Compiled parent-shell writer tensors changed after sealing."
            )
        actual_prior = lfm2_parent_shell_writer_state_sha256(
            writer_id=self.writer_id,
            writer_role=self.writer_role,
            weight=self.weight_structural_prior,
            bias=self.bias_structural_prior,
        )
        if actual_prior != authentication.get("prior_state_sha256_before"):
            raise Lfm2ParentShellWriterCatchupError(
                "Structural writer prior tensors changed after sealing."
            )
        torch = _torch()
        if (
            type(self.weight_delta_fp64) is not torch.Tensor
            or self.weight_delta_fp64.dtype != torch.float64
            or tuple(self.weight_delta_fp64.shape)
            != tuple(self.weight_structural_prior.shape)
        ):
            raise Lfm2ParentShellWriterCatchupError(
                "Compiled writer FP64 weight delta has invalid topology."
            )
        tensors = {
            "weight_structural_prior": self.weight_structural_prior,
            "weight_compiled": self.weight_compiled,
            "weight_delta_fp64": self.weight_delta_fp64,
        }
        if self.bias_structural_prior is None:
            if self.bias_compiled is not None or self.bias_delta_fp64 is not None:
                raise Lfm2ParentShellWriterCatchupError(
                    "Compiled writer changed its absent bias topology."
                )
        else:
            if (
                self.bias_compiled is None
                or type(self.bias_delta_fp64) is not torch.Tensor
                or self.bias_delta_fp64.dtype != torch.float64
                or tuple(self.bias_delta_fp64.shape)
                != tuple(self.bias_structural_prior.shape)
            ):
                raise Lfm2ParentShellWriterCatchupError(
                    "Compiled writer FP64 bias delta has invalid topology."
                )
            tensors.update(
                {
                    "bias_structural_prior": self.bias_structural_prior,
                    "bias_compiled": self.bias_compiled,
                    "bias_delta_fp64": self.bias_delta_fp64,
                }
            )
        _assert_independent_tensors(tensors)
        if _tensor_sha256(self.weight_delta_fp64) != authentication.get(
            "weight_delta_fp64_sha256"
        ) or (
            self.bias_delta_fp64 is not None
            and _tensor_sha256(self.bias_delta_fp64)
            != authentication.get("bias_delta_fp64_sha256")
        ):
            raise Lfm2ParentShellWriterCatchupError(
                "Compiled writer FP64 deltas changed after sealing."
            )
        expected_weight = (
            self.weight_structural_prior.to(torch.float64) + self.weight_delta_fp64
        ).to(torch.float32)
        if not torch.equal(expected_weight, self.weight_compiled):
            raise Lfm2ParentShellWriterCatchupError(
                "Compiled writer is not the authenticated prior plus FP64 delta."
            )
        if self.bias_structural_prior is not None:
            expected_bias = (
                self.bias_structural_prior.to(torch.float64) + self.bias_delta_fp64
            ).to(torch.float32)
            if not torch.equal(expected_bias, self.bias_compiled):
                raise Lfm2ParentShellWriterCatchupError(
                    "Compiled bias is not the authenticated prior plus FP64 delta."
                )
        parameter_elements = int(self.weight_compiled.numel()) + (
            0 if self.bias_compiled is None else int(self.bias_compiled.numel())
        )
        if (
            list(self.weight_compiled.shape) != topology.get("weight_shape")
            or (None if self.bias_compiled is None else list(self.bias_compiled.shape))
            != topology.get("bias_shape")
            or parameter_elements != topology.get("parameter_elements")
            or topology.get("standalone_independent_storage") is not True
            or topology.get("aliases_structural_prior") is not False
        ):
            raise Lfm2ParentShellWriterCatchupError(
                "Compiled writer shape, count, or alias receipt changed."
            )
        return str(seal)


@dataclass(frozen=True, slots=True)
class Lfm2ParentShellCatchupReplayBinding:
    """One sealed post-local replay binding for exactly one downstream writer."""

    report: Mapping[str, Any]

    def verify(self) -> str:
        body = copy.deepcopy(dict(self.report))
        seal = body.pop("binding_sha256", None)
        if (
            body.get("schema") != LFM2_PARENT_SHELL_CATCHUP_REPLAY_BINDING_SCHEMA
            or seal != canonical_sha256(body)
        ):
            raise Lfm2ParentShellWriterCatchupError(
                "Downstream catch-up replay binding seal is invalid."
            )
        _digest(
            body.get("upstream_local_writer_result_sha256"),
            "binding upstream local writer result SHA-256",
        )
        _identity(body.get("downstream_writer_id"), "binding downstream writer ID")
        for partition in ("fit", "heldout"):
            value = body.get(partition)
            if not isinstance(value, Mapping) or not isinstance(
                value.get("rows"), list
            ):
                raise Lfm2ParentShellWriterCatchupError(
                    "Downstream catch-up binding lacks partition row seals."
                )
            _digest(
                value.get("evidence_sha256"),
                f"binding {partition} evidence SHA-256",
            )
            seen: set[str] = set()
            for row in value["rows"]:
                if not isinstance(row, Mapping):
                    raise Lfm2ParentShellWriterCatchupError(
                        "Downstream catch-up binding row seal is malformed."
                    )
                row_id = _identity(row.get("row_id"), "binding row ID")
                if row_id in seen:
                    raise Lfm2ParentShellWriterCatchupError(
                        "Downstream catch-up binding repeats a row ID."
                    )
                seen.add(row_id)
                for name in (
                    "downstream_writer_design_sha256",
                    "live_pre_downstream_residual_sha256",
                    "parent_absolute_endpoint_sha256",
                    "row_sample_weight_sha256",
                ):
                    _digest(row.get(name), f"binding row {name}")
        fit_ids = {row["row_id"] for row in body["fit"]["rows"]}
        heldout_ids = {row["row_id"] for row in body["heldout"]["rows"]}
        if (
            fit_ids & heldout_ids
            or body.get("fit_heldout_row_ids_disjoint") is not True
        ):
            raise Lfm2ParentShellWriterCatchupError(
                "Downstream catch-up binding FIT/heldout rows overlap."
            )
        return str(seal)

    def as_dict(self) -> dict[str, Any]:
        self.verify()
        return copy.deepcopy(dict(self.report))


def _writer_state(
    weight: Any,
    bias: Any | None,
    *,
    label: str,
) -> tuple[Any, Any | None]:
    prior_weight = _fp32_prior(weight, f"{label} weight", ndim=2)
    prior_bias = None if bias is None else _fp32_prior(bias, f"{label} bias", ndim=1)
    if prior_bias is not None and int(prior_bias.shape[0]) != int(
        prior_weight.shape[0]
    ):
        raise Lfm2ParentShellWriterCatchupError(
            f"{label} bias must match the writer output width."
        )
    return prior_weight, prior_bias


def lfm2_parent_shell_writer_state_sha256(
    *,
    writer_id: str,
    writer_role: Lfm2ParentShellWriterRole,
    weight: Any,
    bias: Any | None = None,
) -> str:
    """Seal one named, role-bound dense FP32 parent-shell writer."""

    identity = _identity(writer_id, "writer_id")
    if writer_role not in {_LOCAL_ROLE, _CATCHUP_ROLE}:
        raise Lfm2ParentShellWriterCatchupError("Unknown parent-shell writer role.")
    state_weight, state_bias = _writer_state(weight, bias, label="writer state")
    return canonical_sha256(
        {
            "schema": LFM2_PARENT_SHELL_WRITER_STATE_SCHEMA,
            "writer_id": identity,
            "writer_role": writer_role,
            "weight_sha256": _tensor_sha256(state_weight),
            "bias_sha256": (None if state_bias is None else _tensor_sha256(state_bias)),
            "weight_shape": [int(size) for size in state_weight.shape],
            "bias_shape": (
                None if state_bias is None else [int(size) for size in state_bias.shape]
            ),
            "dtype": "float32",
            "parameter_elements": int(state_weight.numel())
            + (0 if state_bias is None else int(state_bias.numel())),
            "dense": True,
            "parameter_sharing": False,
        }
    )


def _normalize_local_cases(
    cases: Sequence[Lfm2ParentShellLocalWriterEvidenceCase],
    *,
    allow_empty: bool,
) -> tuple[dict[str, Any], ...]:
    if isinstance(cases, (str, bytes)):
        raise Lfm2ParentShellWriterCatchupError("Local evidence must be a sequence.")
    try:
        supplied = tuple(cases)
    except TypeError as exc:
        raise Lfm2ParentShellWriterCatchupError(
            "Local evidence must be a sequence."
        ) from exc
    if not supplied and not allow_empty:
        raise Lfm2ParentShellWriterCatchupError(
            "Local writer compilation requires nonempty FIT evidence."
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_rows: set[str] = set()
    for index, case in enumerate(supplied):
        if not isinstance(case, Lfm2ParentShellLocalWriterEvidenceCase):
            raise Lfm2ParentShellWriterCatchupError(
                f"Local evidence case {index} has the wrong type."
            )
        case_id = _identity(case.case_id, f"local evidence case {index} ID")
        if case_id in seen:
            raise Lfm2ParentShellWriterCatchupError(
                "Local evidence case IDs must be unique within a partition."
            )
        seen.add(case_id)
        design = _cached_tensor(
            case.compact_writer_design,
            f"local evidence {case_id} compact_writer_design",
            ndim=2,
        )
        parent_local_action = _cached_tensor(
            case.parent_local_action,
            f"local evidence {case_id} parent_local_action",
            ndim=2,
        )
        parent_pre = _cached_tensor(
            case.parent_pre_operator_residual,
            f"local evidence {case_id} parent_pre_operator_residual",
            ndim=2,
        )
        parent_post = _cached_tensor(
            case.parent_post_operator_residual,
            f"local evidence {case_id} parent_post_operator_residual",
            ndim=2,
        )
        weights = _cached_tensor(
            case.row_sample_weights,
            f"local evidence {case_id} row_sample_weights",
            ndim=1,
        )
        rows = int(design.shape[0])
        row_ids = _row_ids(
            case.row_ids,
            row_count=rows,
            label=f"local evidence {case_id} row_ids",
        )
        if seen_rows.intersection(row_ids):
            raise Lfm2ParentShellWriterCatchupError(
                "Local evidence row IDs must be globally unique within a partition."
            )
        seen_rows.update(row_ids)
        if (
            int(parent_local_action.shape[0]) != rows
            or int(parent_pre.shape[0]) != rows
            or tuple(parent_post.shape) != tuple(parent_pre.shape)
            or parent_post.dtype != parent_pre.dtype
            or int(parent_local_action.shape[1]) != int(parent_pre.shape[1])
            or tuple(weights.shape) != (rows,)
            or not bool((weights > 0).all().item())
        ):
            raise Lfm2ParentShellWriterCatchupError(
                f"Local evidence {case_id} is not exactly row aligned."
            )
        normalized.append(
            {
                "case_id": case_id,
                "design": design,
                "target": parent_local_action,
                "parent_local_action": parent_local_action,
                "parent_pre": parent_pre,
                "parent_post": parent_post,
                "weights": weights,
                "row_ids": row_ids,
                "receipts": {
                    "parent_trajectory_receipt_sha256": _digest(
                        case.parent_trajectory_receipt_sha256,
                        f"local evidence {case_id} parent trajectory receipt",
                    ),
                    "compact_replay_receipt_sha256": _digest(
                        case.compact_replay_receipt_sha256,
                        f"local evidence {case_id} compact replay receipt",
                    ),
                    "aligned_row_identity_sha256": _digest(
                        case.aligned_row_identity_sha256,
                        f"local evidence {case_id} aligned row identity",
                    ),
                },
            }
        )
    return tuple(normalized)


def _normalize_catchup_cases(
    cases: Sequence[Lfm2ParentShellCatchupEvidenceCase],
    *,
    allow_empty: bool,
) -> tuple[dict[str, Any], ...]:
    if isinstance(cases, (str, bytes)):
        raise Lfm2ParentShellWriterCatchupError("Catch-up evidence must be a sequence.")
    try:
        supplied = tuple(cases)
    except TypeError as exc:
        raise Lfm2ParentShellWriterCatchupError(
            "Catch-up evidence must be a sequence."
        ) from exc
    if not supplied and not allow_empty:
        raise Lfm2ParentShellWriterCatchupError(
            "Catch-up compilation requires nonempty FIT evidence."
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_rows: set[str] = set()
    for index, case in enumerate(supplied):
        if not isinstance(case, Lfm2ParentShellCatchupEvidenceCase):
            raise Lfm2ParentShellWriterCatchupError(
                f"Catch-up evidence case {index} has the wrong type."
            )
        case_id = _identity(case.case_id, f"catch-up evidence case {index} ID")
        if case_id in seen:
            raise Lfm2ParentShellWriterCatchupError(
                "Catch-up evidence case IDs must be unique within a partition."
            )
        seen.add(case_id)
        design = _cached_tensor(
            case.downstream_writer_design,
            f"catch-up evidence {case_id} downstream_writer_design",
            ndim=2,
        )
        live_pre = _cached_tensor(
            case.live_pre_downstream_residual,
            f"catch-up evidence {case_id} live_pre_downstream_residual",
            ndim=2,
        )
        parent_endpoint = _cached_tensor(
            case.parent_absolute_endpoint,
            f"catch-up evidence {case_id} parent_absolute_endpoint",
            ndim=2,
        )
        weights = _cached_tensor(
            case.row_sample_weights,
            f"catch-up evidence {case_id} row_sample_weights",
            ndim=1,
        )
        rows = int(design.shape[0])
        row_ids = _row_ids(
            case.row_ids,
            row_count=rows,
            label=f"catch-up evidence {case_id} row_ids",
        )
        if seen_rows.intersection(row_ids):
            raise Lfm2ParentShellWriterCatchupError(
                "Catch-up evidence row IDs must be globally unique within a partition."
            )
        seen_rows.update(row_ids)
        if (
            int(live_pre.shape[0]) != rows
            or tuple(parent_endpoint.shape) != tuple(live_pre.shape)
            or parent_endpoint.dtype != live_pre.dtype
            or tuple(weights.shape) != (rows,)
            or not bool((weights > 0).all().item())
        ):
            raise Lfm2ParentShellWriterCatchupError(
                f"Catch-up evidence {case_id} is not exactly row aligned."
            )
        normalized.append(
            {
                "case_id": case_id,
                "design": design,
                "target": (
                    parent_endpoint.to(_torch().float64) - live_pre
                ).contiguous(),
                "live_pre": live_pre,
                "parent_endpoint": parent_endpoint,
                "weights": weights,
                "row_ids": row_ids,
                "receipts": {
                    "parent_trajectory_receipt_sha256": _digest(
                        case.parent_trajectory_receipt_sha256,
                        f"catch-up evidence {case_id} parent trajectory receipt",
                    ),
                    "post_local_replay_receipt_sha256": _digest(
                        case.post_local_replay_receipt_sha256,
                        f"catch-up evidence {case_id} post-local replay receipt",
                    ),
                    "aligned_row_identity_sha256": _digest(
                        case.aligned_row_identity_sha256,
                        f"catch-up evidence {case_id} aligned row identity",
                    ),
                },
            }
        )
    return tuple(normalized)


def _evidence_sha256(
    cases: Sequence[Mapping[str, Any]],
    *,
    schema: str,
    partition: Lfm2ParentShellEvidencePartition,
    target_kind: str,
) -> str:
    return canonical_sha256(
        {
            "schema": schema,
            "partition": partition,
            "target_kind": target_kind,
            "case_count": len(cases),
            "cases": [
                {
                    "case_index": index,
                    "case_id": case["case_id"],
                    "row_count": int(case["design"].shape[0]),
                    "row_ids": list(case["row_ids"]),
                    "receipts": dict(case["receipts"]),
                    "tensors": {
                        name: {
                            "sha256": _tensor_sha256(tensor),
                            "shape": [int(size) for size in tensor.shape],
                            "dtype": str(tensor.dtype).removeprefix("torch."),
                        }
                        for name, tensor in sorted(case.items())
                        if name not in {"case_id", "receipts", "row_ids"}
                    },
                }
                for index, case in enumerate(cases)
            ],
            "cached_detached_cpu_tensors_only": True,
            "operation_audit": dict(_OPERATION_AUDIT),
        }
    )


def lfm2_parent_shell_local_writer_evidence_sha256(
    cases: Sequence[Lfm2ParentShellLocalWriterEvidenceCase],
    *,
    partition: Lfm2ParentShellEvidencePartition,
) -> str:
    """Authenticate local-action evidence with an explicit partition."""

    resolved = _validate_partition(partition)
    normalized = _normalize_local_cases(cases, allow_empty=resolved == "heldout")
    return _evidence_sha256(
        normalized,
        schema=LFM2_PARENT_SHELL_LOCAL_EVIDENCE_SCHEMA,
        partition=resolved,
        target_kind="canonical-cached-short-conv-o-action",
    )


def lfm2_parent_shell_catchup_evidence_sha256(
    cases: Sequence[Lfm2ParentShellCatchupEvidenceCase],
    *,
    partition: Lfm2ParentShellEvidencePartition,
) -> str:
    """Authenticate absolute-endpoint evidence with an explicit partition."""

    resolved = _validate_partition(partition)
    normalized = _normalize_catchup_cases(cases, allow_empty=resolved == "heldout")
    return _evidence_sha256(
        normalized,
        schema=LFM2_PARENT_SHELL_CATCHUP_EVIDENCE_SCHEMA,
        partition=resolved,
        target_kind="parent-absolute-endpoint-minus-live-pre-downstream-residual",
    )


def _assert_partition_disjoint(
    fit: Sequence[Mapping[str, Any]], heldout: Sequence[Mapping[str, Any]]
) -> None:
    fit_rows = {row_id for case in fit for row_id in case["row_ids"]}
    heldout_rows = {row_id for case in heldout for row_id in case["row_ids"]}
    if fit_rows & heldout_rows:
        raise Lfm2ParentShellWriterCatchupError(
            "FIT and held-out stable row ID sets must be disjoint."
        )


def _catchup_binding_rows(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for row_index, row_id in enumerate(case["row_ids"]):
            row_slice = slice(row_index, row_index + 1)
            rows.append(
                {
                    "case_index": case_index,
                    "case_id": case["case_id"],
                    "row_index": row_index,
                    "row_id": row_id,
                    "downstream_writer_design_sha256": _tensor_sha256(
                        case["design"][row_slice].contiguous()
                    ),
                    "live_pre_downstream_residual_sha256": _tensor_sha256(
                        case["live_pre"][row_slice].contiguous()
                    ),
                    "parent_absolute_endpoint_sha256": _tensor_sha256(
                        case["parent_endpoint"][row_slice].contiguous()
                    ),
                    "row_sample_weight_sha256": _tensor_sha256(
                        case["weights"][row_slice].contiguous()
                    ),
                }
            )
    return rows


def build_lfm2_parent_shell_catchup_replay_binding(
    *,
    upstream_local_writer_result_sha256: str,
    downstream_writer_id: str,
    fit_cases: Sequence[Lfm2ParentShellCatchupEvidenceCase],
    heldout_cases: Sequence[Lfm2ParentShellCatchupEvidenceCase] = (),
) -> Lfm2ParentShellCatchupReplayBinding:
    """Seal the post-local replay rows consumed by one downstream writer."""

    upstream = _digest(
        upstream_local_writer_result_sha256,
        "upstream local writer result SHA-256",
    )
    writer_id = _identity(downstream_writer_id, "downstream_writer_id")
    fit = _normalize_catchup_cases(fit_cases, allow_empty=False)
    heldout = _normalize_catchup_cases(heldout_cases, allow_empty=True)
    _assert_partition_disjoint(fit, heldout)
    fit_sha256 = _evidence_sha256(
        fit,
        schema=LFM2_PARENT_SHELL_CATCHUP_EVIDENCE_SCHEMA,
        partition="fit",
        target_kind="parent-absolute-endpoint-minus-live-pre-downstream-residual",
    )
    heldout_sha256 = _evidence_sha256(
        heldout,
        schema=LFM2_PARENT_SHELL_CATCHUP_EVIDENCE_SCHEMA,
        partition="heldout",
        target_kind="parent-absolute-endpoint-minus-live-pre-downstream-residual",
    )
    body = {
        "schema": LFM2_PARENT_SHELL_CATCHUP_REPLAY_BINDING_SCHEMA,
        "upstream_local_writer_result_sha256": upstream,
        "downstream_writer_id": writer_id,
        "fit": {
            "evidence_sha256": fit_sha256,
            "rows": _catchup_binding_rows(fit),
        },
        "heldout": {
            "evidence_sha256": heldout_sha256,
            "rows": _catchup_binding_rows(heldout),
        },
        "fit_heldout_row_ids_disjoint": True,
        "exactly_one_downstream_writer": True,
        "row_tensor_seals_complete": True,
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    result = Lfm2ParentShellCatchupReplayBinding(
        report=MappingProxyType({**body, "binding_sha256": canonical_sha256(body)})
    )
    result.verify()
    return result


def _stack_cases(
    cases: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, Any]:
    torch = _torch()
    design = torch.cat([case["design"] for case in cases]).to(torch.float64)
    target = torch.cat([case["target"] for case in cases]).to(torch.float64)
    weights = torch.cat([case["weights"] for case in cases]).to(torch.float64)
    return design.contiguous(), target.contiguous(), weights.contiguous()


def _weighted_diagnostics(
    *,
    design: Any,
    target: Any,
    weights: Any,
    prior_weight: Any,
    prior_bias: Any | None,
    compiled_weight: Any,
    compiled_bias: Any | None,
) -> dict[str, Any]:
    torch = _torch()
    prior = design @ prior_weight.to(torch.float64).T
    compiled = design @ compiled_weight.to(torch.float64).T
    if prior_bias is not None:
        prior = prior + prior_bias.to(torch.float64)
    if compiled_bias is not None:
        compiled = compiled + compiled_bias.to(torch.float64)
    denominator = (weights.sum() * int(target.shape[1])).clamp_min(
        torch.finfo(torch.float64).tiny
    )

    def metrics(prediction: Any) -> dict[str, float]:
        error = prediction - target
        weighted_sse = (error.square() * weights.unsqueeze(1)).sum()
        weighted_target = (target.square() * weights.unsqueeze(1)).sum()
        return {
            "weighted_rmse": float(torch.sqrt(weighted_sse / denominator).item()),
            "weighted_relative_rmse": float(
                torch.sqrt(
                    weighted_sse
                    / weighted_target.clamp_min(torch.finfo(torch.float64).tiny)
                ).item()
            ),
            "maximum_absolute_error": float(error.abs().max().item()),
        }

    return {
        "row_count": int(design.shape[0]),
        "prior": metrics(prior),
        "compiled": metrics(compiled),
        "diagnostic_only": True,
        "has_candidate_selection_authority": False,
        "has_candidate_rejection_authority": False,
    }


def _compile_writer(
    *,
    schema: str,
    status: str,
    writer_id: str,
    writer_role: Lfm2ParentShellWriterRole,
    prior_weight_input: Any,
    prior_bias_input: Any | None,
    expected_prior_sha256: str,
    fit_cases_input: Sequence[Any],
    heldout_cases_input: Sequence[Any],
    normalize: Any,
    evidence_schema: str,
    target_kind: str,
    expected_fit_sha256: str,
    expected_heldout_sha256: str,
    ridge: float,
    fit_unpenalized_bias: bool,
    extra_receipt: Mapping[str, Any],
) -> Lfm2CompiledParentShellWriter:
    torch = _torch()
    identity = _identity(writer_id, "writer_id")
    prior_weight, prior_bias = _writer_state(
        prior_weight_input, prior_bias_input, label="structural prior"
    )
    expected_prior = _digest(expected_prior_sha256, "expected prior state SHA-256")
    expected_fit = _digest(expected_fit_sha256, "expected FIT evidence SHA-256")
    expected_heldout = _digest(
        expected_heldout_sha256, "expected held-out evidence SHA-256"
    )
    ridge_value = _validate_ridge(ridge)
    if type(fit_unpenalized_bias) is not bool:
        raise Lfm2ParentShellWriterCatchupError(
            "fit_unpenalized_bias must be an explicit boolean."
        )
    if fit_unpenalized_bias and prior_bias is None:
        raise Lfm2ParentShellWriterCatchupError(
            "Unpenalized bias fitting was requested without an existing bias topology."
        )

    fit = normalize(fit_cases_input, allow_empty=False)
    heldout = normalize(heldout_cases_input, allow_empty=True)
    _assert_partition_disjoint(fit, heldout)
    actual_prior_before = lfm2_parent_shell_writer_state_sha256(
        writer_id=identity,
        writer_role=writer_role,
        weight=prior_weight_input,
        bias=prior_bias_input,
    )
    actual_fit_before = _evidence_sha256(
        fit,
        schema=evidence_schema,
        partition="fit",
        target_kind=target_kind,
    )
    actual_heldout_before = _evidence_sha256(
        heldout,
        schema=evidence_schema,
        partition="heldout",
        target_kind=target_kind,
    )
    if actual_prior_before != expected_prior:
        raise Lfm2ParentShellWriterCatchupError(
            "Structural writer prior authentication failed."
        )
    if actual_fit_before != expected_fit:
        raise Lfm2ParentShellWriterCatchupError("FIT evidence authentication failed.")
    if actual_heldout_before != expected_heldout:
        raise Lfm2ParentShellWriterCatchupError(
            "Held-out evidence authentication failed."
        )

    design, target, weights = _stack_cases(fit)
    output_width, input_width = map(int, prior_weight.shape)
    if tuple(design.shape[1:]) != (input_width,) or tuple(target.shape[1:]) != (
        output_width,
    ):
        raise Lfm2ParentShellWriterCatchupError(
            "FIT evidence shape does not match the structural writer prior."
        )
    for case in heldout:
        if tuple(case["design"].shape[1:]) != (input_width,) or tuple(
            case["target"].shape[1:]
        ) != (output_width,):
            raise Lfm2ParentShellWriterCatchupError(
                "Held-out evidence shape does not match the structural writer prior."
            )

    prior_t = prior_weight.to(torch.float64).T.contiguous()
    prior_prediction = design @ prior_t
    if prior_bias is not None:
        prior_prediction = prior_prediction + prior_bias.to(torch.float64)
    residual = (target - prior_prediction).contiguous()
    root = torch.sqrt(weights).unsqueeze(1)
    identity_matrix = torch.eye(input_width, dtype=torch.float64)
    with torch.inference_mode():
        if fit_unpenalized_bias:
            mass = weights.sum()
            design_mean = (design * weights.unsqueeze(1)).sum(dim=0) / mass
            residual_mean = (residual * weights.unsqueeze(1)).sum(dim=0) / mass
            centred_design = design - design_mean
            centred_residual = residual - residual_mean
            weighted_design = centred_design * root
            weighted_residual = centred_residual * root
        else:
            design_mean = torch.zeros(input_width, dtype=torch.float64)
            residual_mean = torch.zeros(output_width, dtype=torch.float64)
            weighted_design = design * root
            weighted_residual = residual * root
        gram = weighted_design.T @ weighted_design
        rhs = weighted_design.T @ weighted_residual
        delta_t = torch.linalg.solve(gram + ridge_value * identity_matrix, rhs)
        delta_bias = (
            residual_mean - design_mean @ delta_t
            if fit_unpenalized_bias
            else torch.zeros(output_width, dtype=torch.float64)
        )
        corrected_t = prior_t + delta_t
        corrected_bias_fp64 = (
            None if prior_bias is None else prior_bias.to(torch.float64) + delta_bias
        )

    compiled_weight = corrected_t.T.to(torch.float32).contiguous().clone()
    compiled_bias = (
        None
        if corrected_bias_fp64 is None
        else corrected_bias_fp64.to(torch.float32).contiguous().clone()
    )
    weight_delta = delta_t.T.contiguous().clone()
    bias_delta = None if prior_bias is None else delta_bias.contiguous().clone()

    actual_prior_after = lfm2_parent_shell_writer_state_sha256(
        writer_id=identity,
        writer_role=writer_role,
        weight=prior_weight_input,
        bias=prior_bias_input,
    )
    # Re-normalizing authenticates the caller-owned evidence, not merely our snapshots.
    fit_after = normalize(fit_cases_input, allow_empty=False)
    heldout_after = normalize(heldout_cases_input, allow_empty=True)
    actual_fit_after = _evidence_sha256(
        fit_after,
        schema=evidence_schema,
        partition="fit",
        target_kind=target_kind,
    )
    actual_heldout_after = _evidence_sha256(
        heldout_after,
        schema=evidence_schema,
        partition="heldout",
        target_kind=target_kind,
    )
    if (
        actual_prior_after != actual_prior_before
        or actual_fit_after != actual_fit_before
        or actual_heldout_after != actual_heldout_before
        or compiled_weight.dtype != torch.float32
        or compiled_weight.requires_grad
        or compiled_weight.grad_fn is not None
        or not compiled_weight.is_contiguous()
        or not bool(torch.isfinite(compiled_weight).all().item())
        or (
            compiled_bias is not None
            and (
                compiled_bias.dtype != torch.float32
                or compiled_bias.requires_grad
                or compiled_bias.grad_fn is not None
                or not compiled_bias.is_contiguous()
                or not bool(torch.isfinite(compiled_bias).all().item())
            )
        )
    ):
        raise Lfm2ParentShellWriterCatchupError(
            "Writer materialization failed immutable-input or dense FP32 contracts."
        )
    # Distinct clones are part of the physical result contract.
    if compiled_weight.data_ptr() == prior_weight.data_ptr() or (
        compiled_bias is not None
        and prior_bias is not None
        and compiled_bias.data_ptr() == prior_bias.data_ptr()
    ):
        raise Lfm2ParentShellWriterCatchupError(
            "Compiled writer aliases its structural prior."
        )

    compiled_sha256 = lfm2_parent_shell_writer_state_sha256(
        writer_id=identity,
        writer_role=writer_role,
        weight=compiled_weight,
        bias=compiled_bias,
    )
    fit_diagnostics = _weighted_diagnostics(
        design=design,
        target=target,
        weights=weights,
        prior_weight=prior_weight,
        prior_bias=prior_bias,
        compiled_weight=compiled_weight,
        compiled_bias=compiled_bias,
    )
    heldout_diagnostics = None
    if heldout:
        heldout_design, heldout_target, heldout_weights = _stack_cases(heldout)
        heldout_diagnostics = _weighted_diagnostics(
            design=heldout_design,
            target=heldout_target,
            weights=heldout_weights,
            prior_weight=prior_weight,
            prior_bias=prior_bias,
            compiled_weight=compiled_weight,
            compiled_bias=compiled_bias,
        )

    parameter_elements = int(compiled_weight.numel()) + (
        0 if compiled_bias is None else int(compiled_bias.numel())
    )
    report_body = {
        "schema": schema,
        "status": status,
        "writer_id": identity,
        "writer_role": writer_role,
        **dict(extra_receipt),
        "authentication": {
            "prior_state_sha256_expected": expected_prior,
            "prior_state_sha256_before": actual_prior_before,
            "prior_state_sha256_after": actual_prior_after,
            "prior_immutable": True,
            "fit_evidence_sha256_expected": expected_fit,
            "fit_evidence_sha256_before": actual_fit_before,
            "fit_evidence_sha256_after": actual_fit_after,
            "fit_evidence_immutable": True,
            "heldout_evidence_sha256_expected": expected_heldout,
            "heldout_evidence_sha256_before": actual_heldout_before,
            "heldout_evidence_sha256_after": actual_heldout_after,
            "heldout_evidence_immutable": True,
            "compiled_writer_state_sha256": compiled_sha256,
            "weight_delta_fp64_sha256": _tensor_sha256(weight_delta),
            "bias_delta_fp64_sha256": (
                None if bias_delta is None else _tensor_sha256(bias_delta)
            ),
        },
        "topology": {
            "output_width": output_width,
            "input_width": input_width,
            "weight_shape": [output_width, input_width],
            "bias_shape": None if compiled_bias is None else [output_width],
            "weight_parameter_elements": int(compiled_weight.numel()),
            "bias_parameter_elements": (
                0 if compiled_bias is None else int(compiled_bias.numel())
            ),
            "parameter_elements": parameter_elements,
            "exact_shape_preserved": True,
            "dense_fp32": True,
            "standalone_independent_storage": True,
            "aliases_structural_prior": False,
            "parameter_sharing_used": False,
        },
        "solver": {
            "method": "prior-centred-weighted-ridge-closed-form",
            "compute_device": "cpu",
            "compute_dtype": "float64",
            "materialization_dtype": "float32",
            "ridge": ridge_value,
            "unpenalized_bias_fit_explicitly_enabled": fit_unpenalized_bias,
            "bias_topology_present": prior_bias is not None,
            "weight_delta_penalized": True,
            "bias_delta_penalized": False if fit_unpenalized_bias else None,
            "closed_form_linear_solve_count": 1,
            "iterative_parameter_optimization_used": False,
            "fit_row_count": int(design.shape[0]),
            "fit_case_count": len(fit),
            "fit_gram_sha256": _tensor_sha256(gram),
            "fit_rhs_sha256": _tensor_sha256(rhs),
        },
        "evidence_policy": {
            "fit_partition_used_for_construction": True,
            "heldout_partition_used_for_construction": False,
            "heldout_partition_diagnostic_only": True,
            "heldout_metrics_can_select_candidate": False,
            "heldout_metrics_can_reject_candidate": False,
            "fit_and_heldout_stable_row_id_sets_disjoint": True,
            "row_identity_authority": "explicit-stable-row-ids",
            "fit_case_ids": [case["case_id"] for case in fit],
            "heldout_case_ids": [case["case_id"] for case in heldout],
            "fit_row_ids": [row_id for case in fit for row_id in case["row_ids"]],
            "heldout_row_ids": [
                row_id for case in heldout for row_id in case["row_ids"]
            ],
        },
        "diagnostics": {
            "fit": fit_diagnostics,
            "heldout": heldout_diagnostics,
            "numeric_metrics_have_installation_authority": False,
            "manual_whole_output_review_remains_authoritative": True,
        },
        "operation_audit": dict(_OPERATION_AUDIT),
    }
    report = {**report_body, "result_sha256": canonical_sha256(report_body)}
    result = Lfm2CompiledParentShellWriter(
        writer_id=identity,
        writer_role=writer_role,
        weight_structural_prior=prior_weight,
        bias_structural_prior=prior_bias,
        weight_compiled=compiled_weight,
        bias_compiled=compiled_bias,
        weight_delta_fp64=weight_delta,
        bias_delta_fp64=bias_delta,
        report=MappingProxyType(report),
    )
    result.verify()
    return result


def compile_lfm2_parent_shell_short_conv_local_writer(
    *,
    writer_id: str,
    structural_out_weight: Any,
    structural_out_bias: Any | None = None,
    expected_structural_writer_state_sha256: str,
    fit_cases: Sequence[Lfm2ParentShellLocalWriterEvidenceCase],
    expected_fit_evidence_sha256: str,
    heldout_cases: Sequence[Lfm2ParentShellLocalWriterEvidenceCase] = (),
    expected_heldout_evidence_sha256: str,
    ridge: float = 1.0e-6,
    fit_unpenalized_bias: bool = False,
) -> Lfm2CompiledParentShellWriter:
    """Fit a compact ShortConv context to its parent-width local action."""

    return _compile_writer(
        schema=LFM2_PARENT_SHELL_LOCAL_WRITER_SCHEMA,
        status="parent-shell-compact-short-conv-local-writer-ready",
        writer_id=writer_id,
        writer_role=_LOCAL_ROLE,
        prior_weight_input=structural_out_weight,
        prior_bias_input=structural_out_bias,
        expected_prior_sha256=expected_structural_writer_state_sha256,
        fit_cases_input=fit_cases,
        heldout_cases_input=heldout_cases,
        normalize=_normalize_local_cases,
        evidence_schema=LFM2_PARENT_SHELL_LOCAL_EVIDENCE_SCHEMA,
        target_kind="canonical-cached-short-conv-o-action",
        expected_fit_sha256=expected_fit_evidence_sha256,
        expected_heldout_sha256=expected_heldout_evidence_sha256,
        ridge=ridge,
        fit_unpenalized_bias=fit_unpenalized_bias,
        extra_receipt={
            "target": {
                "kind": "canonical-cached-short-conv-o-action",
                "formula": "direct cached short_conv.o rows",
                "cached_short_conv_o_has_target_authority": True,
                "residual_pre_post_difference_has_target_authority": False,
                "residual_pre_post_rows_are_diagnostic_only": True,
                "absolute_endpoint_used": False,
                "upstream_accumulated_defect_absorbed": False,
            },
            "parent_shell_output_width_preserved": True,
            "compact_context_is_writer_input": True,
        },
    )


def compile_lfm2_parent_shell_one_downstream_catchup_writer(
    *,
    downstream_writer_id: str,
    structural_downstream_weight: Any,
    structural_downstream_bias: Any | None = None,
    expected_structural_writer_state_sha256: str,
    replay_binding: Lfm2ParentShellCatchupReplayBinding,
    fit_cases: Sequence[Lfm2ParentShellCatchupEvidenceCase],
    expected_fit_evidence_sha256: str,
    heldout_cases: Sequence[Lfm2ParentShellCatchupEvidenceCase] = (),
    expected_heldout_evidence_sha256: str,
    ridge: float = 1.0e-6,
    fit_unpenalized_bias: bool = False,
) -> Lfm2CompiledParentShellWriter:
    """Fit exactly one downstream writer to a parent absolute endpoint.

    The supplied trajectories must exactly match the sealed replay binding
    created after the upstream local writer.  This function cannot capture
    that replay and cannot alter any second writer.
    """

    if not isinstance(replay_binding, Lfm2ParentShellCatchupReplayBinding):
        raise Lfm2ParentShellWriterCatchupError(
            "Downstream catch-up requires a sealed replay binding object."
        )
    binding = replay_binding.as_dict()
    if binding.get("downstream_writer_id") != downstream_writer_id:
        raise Lfm2ParentShellWriterCatchupError(
            "Downstream catch-up replay binding names a different writer."
        )
    expected_binding = build_lfm2_parent_shell_catchup_replay_binding(
        upstream_local_writer_result_sha256=binding[
            "upstream_local_writer_result_sha256"
        ],
        downstream_writer_id=downstream_writer_id,
        fit_cases=fit_cases,
        heldout_cases=heldout_cases,
    )
    if expected_binding.as_dict() != binding:
        raise Lfm2ParentShellWriterCatchupError(
            "Downstream catch-up replay binding does not match supplied rows/tensors."
        )
    local_result = binding["upstream_local_writer_result_sha256"]
    return _compile_writer(
        schema=LFM2_PARENT_SHELL_CATCHUP_WRITER_SCHEMA,
        status="one-downstream-absolute-endpoint-catchup-writer-ready",
        writer_id=downstream_writer_id,
        writer_role=_CATCHUP_ROLE,
        prior_weight_input=structural_downstream_weight,
        prior_bias_input=structural_downstream_bias,
        expected_prior_sha256=expected_structural_writer_state_sha256,
        fit_cases_input=fit_cases,
        heldout_cases_input=heldout_cases,
        normalize=_normalize_catchup_cases,
        evidence_schema=LFM2_PARENT_SHELL_CATCHUP_EVIDENCE_SCHEMA,
        target_kind="parent-absolute-endpoint-minus-live-pre-downstream-residual",
        expected_fit_sha256=expected_fit_evidence_sha256,
        expected_heldout_sha256=expected_heldout_evidence_sha256,
        ridge=ridge,
        fit_unpenalized_bias=fit_unpenalized_bias,
        extra_receipt={
            "upstream_local_writer_result_sha256": local_result,
            "catchup_replay_binding_sha256": binding["binding_sha256"],
            "catchup_replay_binding": binding,
            "target": {
                "kind": "absolute-parent-endpoint-catchup-action",
                "formula": ("parent_absolute_endpoint-live_pre_downstream_residual"),
                "absolute_endpoint_used": True,
                "accumulated_upstream_defect_may_be_absorbed": True,
            },
            "downstream_scope": {
                "writer_count": 1,
                "exactly_one_downstream_writer": True,
                "other_downstream_writers_changed": 0,
                "fresh_post_local_cached_replay_required": True,
            },
        },
    )


__all__ = [
    "LFM2_PARENT_SHELL_CATCHUP_EVIDENCE_SCHEMA",
    "LFM2_PARENT_SHELL_CATCHUP_REPLAY_BINDING_SCHEMA",
    "LFM2_PARENT_SHELL_CATCHUP_WRITER_SCHEMA",
    "LFM2_PARENT_SHELL_LOCAL_EVIDENCE_SCHEMA",
    "LFM2_PARENT_SHELL_LOCAL_WRITER_SCHEMA",
    "LFM2_PARENT_SHELL_WRITER_STATE_SCHEMA",
    "Lfm2CompiledParentShellWriter",
    "Lfm2ParentShellCatchupEvidenceCase",
    "Lfm2ParentShellCatchupReplayBinding",
    "Lfm2ParentShellLocalWriterEvidenceCase",
    "Lfm2ParentShellWriterCatchupError",
    "build_lfm2_parent_shell_catchup_replay_binding",
    "compile_lfm2_parent_shell_one_downstream_catchup_writer",
    "compile_lfm2_parent_shell_short_conv_local_writer",
    "lfm2_parent_shell_catchup_evidence_sha256",
    "lfm2_parent_shell_local_writer_evidence_sha256",
    "lfm2_parent_shell_writer_state_sha256",
]
