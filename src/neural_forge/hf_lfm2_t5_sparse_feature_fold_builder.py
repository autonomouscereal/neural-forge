"""Authenticated production construction of full-axis LFM2 T5 bundles.

This module is deliberately target-aware and deliberately separate from the
trace collector.  It consumes only rows copied by the authenticated selective
row reader, invokes the existing T5 route solver, and returns the existing
compile-time sparse-feature bundle.  Importers that do not configure this
factory never import the T5 implementation or alter the technique registry.
"""

from __future__ import annotations

import hashlib
import math
import re
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import ModelCircuitGraph, canonical_sha256
from .dynamic_fold_executor import fingerprint_model_state
from .dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
    validate_lfm2_dense_budgeted_child,
)
from .dynamic_model_inspector import inspect_model_circuits
from .dynamic_target_synthesis import (
    DynamicTargetSynthesisResult,
    verify_compiler_authenticated_target,
)
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCacheError,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
    _layer_topology,
    fingerprint_lfm2_parent,
)
from .hf_lfm2_circuit_trace_row_selection import (
    lfm2_circuit_trace_selected_rows_payload_sha256,
    load_lfm2_circuit_trace_selected_rows,
    release_lfm2_circuit_trace_selected_rows,
)
from .hf_lfm2_sparse_feature_fold_bundle import (
    AXIS_MLP_POST_SWIGLU,
    AXIS_SHORT_CONV_CONTEXT,
    Lfm2SparseFeatureAxisAdmission,
    Lfm2SparseFeatureFoldBundle,
    admit_lfm2_sparse_feature_axis,
    build_lfm2_sparse_feature_fold_bundle,
)
from .hf_lfm2_runtime_semantics_receipt import (
    LFM2_RUNTIME_SEMANTICS_RECEIPT_SCHEMA,
    lfm2_runtime_semantics_receipt,
    lfm2_runtime_semantics_receipt_source_sha256,
    load_fresh_source_module,
    runtime_callable_record,
    source_callable_semantics_record,
    verify_lfm2_runtime_semantics_receipt,
)
from .matrix_technique_body_routes import (
    build_t5_body_route_source_witness,
    fit_body_component_routes,
    t5_tensor_sha256,
)


LFM2_T5_SPARSE_FEATURE_FOLD_BUILDER_SCHEMA = (
    "neural-forge-lfm2-t5-sparse-feature-fold-builder-v1"
)
LFM2_T5_EVIDENCE_PARTITION_SCHEMA = "neural-forge-lfm2-t5-evidence-partition-v1"
LFM2_T5_EVIDENCE_SELECTION_SCHEMA = "neural-forge-lfm2-t5-evidence-selection-v1"
LFM2_T5_EVIDENCE_SELECTION_MANIFEST_SCHEMA = (
    "neural-forge-lfm2-t5-evidence-selection-manifest-v1"
)
LFM2_T5_EVIDENCE_SPLIT_RESOLUTION_SCHEMA = (
    "neural-forge-lfm2-t5-evidence-split-resolution-v1"
)
LFM2_T5_EVIDENCE_PATH_PLAN_SCHEMA = "neural-forge-lfm2-t5-evidence-path-plan-v1"
_PRODUCTION_SEALED_CACHE_SPLIT_SCHEMA = (
    "neural-forge-production-lfm2-sealed-cache-evidence-splits-v1"
)

_BUILDER_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).resolve().read_bytes()
).hexdigest()
_PATH = re.compile(
    r"^batch\.(?P<batch>[0-9]{6})/"
    r"(?P<phase>prefill|decode\.[0-9]{3})/"
    r"row\.(?P<row>[0-9]{4})$"
)
_EXPECTED_LAYER_COUNT = 14
_EXPECTED_SHORT_CONV_LAYERS = (0, 1, 3, 5, 7, 9, 11, 13)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Lfm2T5SparseFeatureFoldBuilderError(RuntimeError):
    """Raised before a bundle can be attributed to exact sealed evidence."""


def _verify_selective_session(
    session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
) -> None:
    try:
        session.verify()
    except Lfm2CircuitTraceCacheError as exc:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "T5 evidence cache runtime was replaced or lost its canonical source "
            "receipt."
        ) from exc


_INTERNAL_RUNTIME_FUNCTION_CONTRACTS: Mapping[str, Any] = MappingProxyType({})
_INTERNAL_RUNTIME_VALUE_CONTRACTS: Mapping[str, Any] = MappingProxyType({})
_CANONICAL_RUNTIME_RECEIPT: Mapping[str, Any] = MappingProxyType({})
_CANONICAL_RUNTIME_RECEIPT_SHA256 = ""


def _capture_internal_runtime_method_contracts() -> Mapping[
    type[Any], Mapping[str, Any]
]:
    """Capture installed native/dynamic callable objects before target admission."""

    import torch
    from transformers.models.lfm2 import modeling_lfm2

    global _INTERNAL_RUNTIME_FUNCTION_CONTRACTS
    global _INTERNAL_RUNTIME_VALUE_CONTRACTS
    global _CANONICAL_RUNTIME_RECEIPT
    global _CANONICAL_RUNTIME_RECEIPT_SHA256
    native_capture_receipt = lfm2_runtime_semantics_receipt()
    native_capture_receipt_sha256 = verify_lfm2_runtime_semantics_receipt()
    native_capture_receipt_source_sha256 = (
        lfm2_runtime_semantics_receipt_source_sha256()
    )
    from . import hf_lfm2_dynamic_blank_materializer as dynamic_blank

    canonical_runtime_body = {
        "schema": "neural-forge-lfm2-t5-composed-runtime-receipt-v1",
        "native_capture_runtime": {
            "schema": native_capture_receipt["schema"],
            "receipt_sha256": native_capture_receipt_sha256,
            "verifier_source_sha256": native_capture_receipt_source_sha256,
        },
        "dynamic_target_runtime": _independent_dynamic_runtime_receipt(dynamic_blank),
    }
    _CANONICAL_RUNTIME_RECEIPT = MappingProxyType(canonical_runtime_body)
    _CANONICAL_RUNTIME_RECEIPT_SHA256 = canonical_sha256(canonical_runtime_body)

    _INTERNAL_RUNTIME_FUNCTION_CONTRACTS = MappingProxyType(
        {
            "torch.nn.functional.silu": torch.nn.functional.silu,
            "torch.nn.functional.linear": torch.nn.functional.linear,
            "torch.nn.functional.conv1d": torch.nn.functional.conv1d,
            "torch.nn.Module.__call__": torch.nn.Module.__call__,
            "torch.nn.Module._wrapped_call_impl": (torch.nn.Module._wrapped_call_impl),
            "torch.nn.Module._call_impl": torch.nn.Module._call_impl,
            "torch.nn.Module._slow_forward": torch.nn.Module._slow_forward,
            "torch.nn.Conv1d._conv_forward": torch.nn.Conv1d._conv_forward,
            "Lfm2ShortConv.slow_forward": (modeling_lfm2.Lfm2ShortConv.slow_forward),
            "Lfm2ShortConv.cuda_kernels_forward": (
                modeling_lfm2.Lfm2ShortConv.cuda_kernels_forward
            ),
        }
    )
    _INTERNAL_RUNTIME_VALUE_CONTRACTS = MappingProxyType(
        {
            "modeling_lfm2.torch": modeling_lfm2.torch,
            "modeling_lfm2.nn": modeling_lfm2.nn,
            "modeling_lfm2.apply_rotary_pos_emb": (modeling_lfm2.apply_rotary_pos_emb),
            "modeling_lfm2.eager_attention_forward": (
                modeling_lfm2.eager_attention_forward
            ),
            "modeling_lfm2.apply_mask_to_padding_states": (
                modeling_lfm2.apply_mask_to_padding_states
            ),
            "modeling_lfm2.rotate_half": modeling_lfm2.rotate_half,
            "modeling_lfm2.repeat_kv": modeling_lfm2.repeat_kv,
            "modeling_lfm2.is_torchdynamo_compiling": (
                modeling_lfm2.is_torchdynamo_compiling
            ),
            "modeling_lfm2.is_fast_path_available": (
                modeling_lfm2.is_fast_path_available
            ),
            "modeling_lfm2.causal_conv1d_fn": modeling_lfm2.causal_conv1d_fn,
            "modeling_lfm2.causal_conv1d_update": (modeling_lfm2.causal_conv1d_update),
            "modeling_lfm2.ALL_ATTENTION_FUNCTIONS": (
                modeling_lfm2.ALL_ATTENTION_FUNCTIONS
            ),
            "attention_interface.eager": (
                modeling_lfm2.ALL_ATTENTION_FUNCTIONS.get_interface(
                    "eager",
                    modeling_lfm2.eager_attention_forward,
                )
            ),
            "attention_interface.sdpa": (
                modeling_lfm2.ALL_ATTENTION_FUNCTIONS.get_interface(
                    "sdpa",
                    modeling_lfm2.eager_attention_forward,
                )
            ),
        }
    )

    dynamic_mlp = dynamic_blank._dynamic_mlp_class()
    dynamic_conv = dynamic_blank._dynamic_short_conv_class()
    dynamic_attention = dynamic_blank._dynamic_attention_class()
    contracts: dict[type[Any], Mapping[str, Any]] = {
        modeling_lfm2.Lfm2DecoderLayer: MappingProxyType(
            {"forward": modeling_lfm2.Lfm2DecoderLayer.forward}
        ),
        modeling_lfm2.Lfm2MLP: MappingProxyType(
            {"forward": modeling_lfm2.Lfm2MLP.forward}
        ),
        modeling_lfm2.Lfm2ShortConv: MappingProxyType(
            {
                "forward": modeling_lfm2.Lfm2ShortConv.forward,
                "slow_forward": modeling_lfm2.Lfm2ShortConv.slow_forward,
                "cuda_kernels_forward": (
                    modeling_lfm2.Lfm2ShortConv.cuda_kernels_forward
                ),
            }
        ),
        modeling_lfm2.Lfm2Attention: MappingProxyType(
            {"forward": modeling_lfm2.Lfm2Attention.forward}
        ),
        modeling_lfm2.Lfm2RMSNorm: MappingProxyType(
            {"forward": modeling_lfm2.Lfm2RMSNorm.forward}
        ),
        dynamic_mlp: MappingProxyType(
            dict(dynamic_blank._DYNAMIC_MLP_RUNTIME_METHODS or {})
        ),
        dynamic_conv: MappingProxyType(
            {
                **dict(dynamic_blank._DYNAMIC_SHORT_CONV_RUNTIME_METHODS or {}),
                "slow_forward": dynamic_conv.slow_forward,
                "cuda_kernels_forward": dynamic_conv.cuda_kernels_forward,
            }
        ),
        dynamic_attention: MappingProxyType(
            dict(dynamic_blank._DYNAMIC_ATTENTION_RUNTIME_METHODS or {})
        ),
        torch.nn.Linear: MappingProxyType({"forward": torch.nn.Linear.forward}),
        torch.nn.Conv1d: MappingProxyType(
            {
                "forward": torch.nn.Conv1d.forward,
                "_conv_forward": torch.nn.Conv1d._conv_forward,
            }
        ),
        torch.nn.Sequential: MappingProxyType({"forward": torch.nn.Sequential.forward}),
    }
    contracts = {
        runtime_class: MappingProxyType(
            {
                **dict(methods),
                "__call__": runtime_class.__call__,
                "_call_impl": runtime_class._call_impl,
            }
        )
        for runtime_class, methods in contracts.items()
    }
    if any(not methods for methods in contracts.values()):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Installed LFM2 runtime method contracts are unavailable."
        )
    for runtime_class, methods in contracts.items():
        for method_name, method in methods.items():
            observed = (
                modeling_lfm2.Lfm2MLP.forward
                if runtime_class is dynamic_mlp and method_name == "__native_forward__"
                else modeling_lfm2.Lfm2ShortConv.forward
                if runtime_class is dynamic_conv and method_name == "__native_forward__"
                else modeling_lfm2.Lfm2Attention.forward
                if runtime_class is dynamic_attention
                and method_name == "__native_forward__"
                else getattr(runtime_class, method_name, None)
            )
            if observed is not method:
                raise Lfm2T5SparseFeatureFoldBuilderError(
                    "Installed LFM2 runtime method identity drifted during import."
                )
    return MappingProxyType(contracts)


_INTERNAL_RUNTIME_METHOD_CONTRACTS: Mapping[type[Any], Mapping[str, Any]] = (
    MappingProxyType({})
)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2T5SparseFeatureFoldBuilderError(f"{name} must be a positive integer.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} must be finite and non-negative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} must be finite and non-negative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} must be finite and non-negative."
        )
    return result


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


@dataclass(frozen=True, slots=True)
class Lfm2T5SparseFeatureFoldSolveConfig:
    """The exact closed-form T5 solve policy used by production."""

    active_k: int = 8
    refit_ridge_relative: float = 1e-10
    write_ridge_relative: float = 1e-10
    write_solve_sweeps: int = 4
    pivot_tolerance: float = 1e-10
    residual_floor: float = 1e-12
    channels_per_chunk: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_k", _positive_int(self.active_k, "active_k"))
        object.__setattr__(
            self,
            "channels_per_chunk",
            _positive_int(self.channels_per_chunk, "channels_per_chunk"),
        )
        if (
            isinstance(self.write_solve_sweeps, bool)
            or not isinstance(self.write_solve_sweeps, int)
            or self.write_solve_sweeps < 0
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "write_solve_sweeps must be a non-negative integer."
            )
        for name in (
            "refit_ridge_relative",
            "write_ridge_relative",
            "pivot_tolerance",
            "residual_floor",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": LFM2_T5_SPARSE_FEATURE_FOLD_BUILDER_SCHEMA,
            "component_level": "X4",
            "block_count": 1,
            "active_k": self.active_k,
            "channel_selection": "evidence-energy",
            "kept_channel_ids": None,
            "write_metric": None,
            "install_coefficients": "write",
            "refit_ridge_relative": self.refit_ridge_relative,
            "write_ridge_relative": self.write_ridge_relative,
            "write_solve_sweeps": self.write_solve_sweeps,
            "pivot_tolerance": self.pivot_tolerance,
            "residual_floor": self.residual_floor,
            "channels_per_chunk": self.channels_per_chunk,
            "solve_dtype": "float64",
            "solve_device": "cpu",
        }

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class _ParsedPath:
    global_index: int
    path: str
    split: str
    batch_index: int
    phase: str
    row_index: int


@dataclass(frozen=True, slots=True)
class _AxisSpec:
    layer_index: int
    axis_kind: str
    component: str
    endpoint: str
    writer_role: str
    graph_site: str
    parameter_paths: Mapping[str, str]
    source_row_slices: Mapping[str, tuple[int, int]]
    target_row_slices: Mapping[str, tuple[int, int]]
    source_width: int
    target_width: int
    parent_writer: Any

    @property
    def key(self) -> tuple[int, str]:
        return (self.layer_index, self.axis_kind)

    def tensor_name(self, phase: str) -> str:
        return f"{phase}.layer.{self.layer_index:03d}.{self.endpoint}"


@dataclass(frozen=True, slots=True)
class _WriterSnapshot:
    tensor: Any
    tensor_sha256: str
    snapshot_tensor_version: int
    object_id: int
    tensor_version: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    parameter_path: str


def _writer_tensor_version(value: Any) -> int:
    version = getattr(value, "_version", None)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "A source writer does not expose a stable tensor version."
        )
    return version


def _verify_writer_snapshot_sources(
    *,
    parent_model: Any,
    specs: Sequence[_AxisSpec],
    snapshots: Mapping[tuple[int, str], _WriterSnapshot],
    parent_state_sha256: str,
) -> None:
    parameters = _named_parameters(parent_model, "parent_model")
    if set(snapshots) != {spec.key for spec in specs}:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "The all-axis writer snapshot inventory changed."
        )
    for spec in specs:
        snapshot = snapshots[spec.key]
        writer = spec.parent_writer
        path = spec.parameter_paths[spec.writer_role]
        if (
            parameters.get(path) is not writer
            or id(writer) != snapshot.object_id
            or _writer_tensor_version(writer) != snapshot.tensor_version
            or _writer_tensor_version(snapshot.tensor)
            != snapshot.snapshot_tensor_version
            or _tensor_sha256(snapshot.tensor, path) != snapshot.tensor_sha256
            or tuple(int(size) for size in writer.shape) != snapshot.shape
            or str(writer.dtype) != snapshot.dtype
            or str(writer.device) != snapshot.device
            or path != snapshot.parameter_path
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Source writer {path!r} changed around the all-axis solve."
            )
    if fingerprint_model_state(parent_model) != parent_state_sha256:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Parent state changed around the all-axis writer snapshot."
        )


def _snapshot_axis_writers(
    *,
    parent_model: Any,
    specs: Sequence[_AxisSpec],
    parent_state_sha256: str,
) -> Mapping[tuple[int, str], _WriterSnapshot]:
    import torch

    snapshots: dict[tuple[int, str], _WriterSnapshot] = {}
    for spec in specs:
        writer = spec.parent_writer
        path = spec.parameter_paths[spec.writer_role]
        version = _writer_tensor_version(writer)
        snapshot = (
            writer.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
        )
        if (
            snapshot.requires_grad
            or not bool(torch.isfinite(snapshot).all().item())
            or _writer_tensor_version(writer) != version
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Source writer {path!r} changed during its owned snapshot."
            )
        snapshots[spec.key] = _WriterSnapshot(
            tensor=snapshot,
            tensor_sha256=_tensor_sha256(snapshot, path),
            snapshot_tensor_version=_writer_tensor_version(snapshot),
            object_id=id(writer),
            tensor_version=version,
            shape=tuple(int(size) for size in writer.shape),
            dtype=str(writer.dtype),
            device=str(writer.device),
            parameter_path=path,
        )
    result = MappingProxyType(snapshots)
    _verify_writer_snapshot_sources(
        parent_model=parent_model,
        specs=specs,
        snapshots=result,
        parent_state_sha256=parent_state_sha256,
    )
    return result


def _canonical_indices(value: Any, name: str, *, row_count: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2T5SparseFeatureFoldBuilderError(f"{name} must be an index sequence.")
    result = tuple(value)
    if (
        not result
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < row_count
            for index in result
        )
        or result != tuple(sorted(set(result)))
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} must be nonempty, canonical, unique, and in range."
        )
    return result


def _parse_paths(
    paths: Sequence[str],
    path_splits: Sequence[str],
) -> tuple[_ParsedPath, ...]:
    path_values = tuple(paths)
    split_values = tuple(str(value).strip().lower() for value in path_splits)
    if not path_values or len(path_values) != len(split_values):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Trace paths and split lineage must be nonempty and aligned."
        )
    result = []
    for index, (path, split) in enumerate(zip(path_values, split_values, strict=True)):
        if not isinstance(path, str):
            raise Lfm2T5SparseFeatureFoldBuilderError("Trace paths must be strings.")
        match = _PATH.fullmatch(path)
        if match is None or split not in {"selection", "fit", "validation"}:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Trace paths or split lineage are not canonical production evidence."
            )
        result.append(
            _ParsedPath(
                global_index=index,
                path=path,
                split=split,
                batch_index=int(match.group("batch")),
                phase=str(match.group("phase")),
                row_index=int(match.group("row")),
            )
        )
    if tuple(value.path for value in result) != path_values or len(
        set(path_values)
    ) != len(path_values):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Trace paths must be unique and unchanged by canonical parsing."
        )
    return tuple(result)


def _tensor_sha256(value: Any, name: str) -> str:
    import torch

    if (
        not torch.is_tensor(value)
        or value.layout != torch.strided
        or value.device.type != "cpu"
        or value.is_complex()
        or not value.is_floating_point()
        or value.ndim != 2
        or value.numel() <= 0
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} must be a nonempty dense real CPU matrix."
        )
    snapshot = value.detach().contiguous().clone()
    if snapshot.requires_grad or not bool(torch.isfinite(snapshot).all().item()):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} must be detached and finite."
        )
    payload = snapshot.view(torch.uint8).numpy().tobytes()
    return canonical_sha256(
        {
            "dtype": str(snapshot.dtype).removeprefix("torch."),
            "shape": [int(size) for size in snapshot.shape],
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )


def _cache_layer_topology_evidence(
    cache: Lfm2CircuitTraceCache,
) -> tuple[str, str]:
    """Return the cache-authenticated topology body and document digests."""

    if type(cache) is not Lfm2CircuitTraceCache:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Layer-topology evidence requires the exact sealed cache type."
        )
    topology = cache.manifest.get("layer_topology")
    if not isinstance(topology, Mapping):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "The sealed cache has no canonical layer-topology document."
        )
    topology_document = dict(topology)
    supplied = topology_document.pop("layer_topology_sha256", None)
    topology_sha256 = _digest(supplied, "cache.layer_topology_sha256")
    if (
        topology_sha256 != cache.layer_topology_fingerprint
        or canonical_sha256(topology_document) != topology_sha256
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "The sealed cache layer-topology body is inconsistent."
        )
    return topology_sha256, canonical_sha256(dict(topology))


def _authenticate_live_lfm2_topology(
    *,
    model: Any,
    graph: ModelCircuitGraph,
    label: str,
    expected_layer_topology_sha256: str | None = None,
    expected_layer_topology_document_sha256: str | None = None,
) -> tuple[str, str]:
    """Reinspect one live LFM2 graph and its exact canonical module topology."""

    if not isinstance(graph, ModelCircuitGraph):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{label} graph is not a ModelCircuitGraph."
        )
    try:
        _layers, _contracts, topology = _layer_topology(model)
        live_graph = inspect_model_circuits(
            model,
            model_name=graph.model_name,
            tokenizer_fingerprint=graph.tokenizer_fingerprint,
        )
    except Exception as exc:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{label} failed live canonical topology inspection."
        ) from exc
    topology_sha256 = _digest(
        topology.get("layer_topology_sha256"),
        f"{label}.layer_topology_sha256",
    )
    topology_body = dict(topology)
    topology_body.pop("layer_topology_sha256", None)
    topology_document_sha256 = canonical_sha256(dict(topology))
    if (
        canonical_sha256(topology_body) != topology_sha256
        or live_graph.fingerprint != graph.fingerprint
        or (
            expected_layer_topology_sha256 is not None
            and topology_sha256
            != _digest(
                expected_layer_topology_sha256,
                f"{label}.expected_layer_topology_sha256",
            )
        )
        or (
            expected_layer_topology_document_sha256 is not None
            and topology_document_sha256
            != _digest(
                expected_layer_topology_document_sha256,
                f"{label}.expected_layer_topology_document_sha256",
            )
        )
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{label} differs from its authenticated graph or layer topology."
        )
    return topology_sha256, topology_document_sha256


def _dynamic_runtime_receipt_body(
    dynamic_blank: types.ModuleType,
    *,
    expected_module: str,
    trusted_unloaded_globals: bool,
) -> dict[str, Any]:
    source_value = getattr(dynamic_blank, "__file__", None)
    if not isinstance(source_value, str):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Dynamic LFM2 materializer has no defining source."
        )
    source_path = Path(source_value).resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Dynamic LFM2 materializer source is not a regular file."
        )
    specifications = (
        (
            "mlp",
            dynamic_blank._dynamic_mlp_class(),
            dynamic_blank._DYNAMIC_MLP_RUNTIME_METHODS,
        ),
        (
            "short_conv",
            dynamic_blank._dynamic_short_conv_class(),
            dynamic_blank._DYNAMIC_SHORT_CONV_RUNTIME_METHODS,
        ),
        (
            "attention",
            dynamic_blank._dynamic_attention_class(),
            dynamic_blank._DYNAMIC_ATTENTION_RUNTIME_METHODS,
        ),
    )
    class_records = {}
    for name, runtime_class, methods in specifications:
        if not isinstance(methods, Mapping) or not methods:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Dynamic LFM2 {name} runtime method receipt is unavailable."
            )
        method_records = {}
        for method_name, recorded_method in sorted(methods.items()):
            observed_method = (
                getattr(runtime_class.__bases__[0], "forward", None)
                if method_name == "__native_forward__"
                else getattr(runtime_class, method_name, None)
            )
            if observed_method is not recorded_method:
                raise Lfm2T5SparseFeatureFoldBuilderError(
                    f"Dynamic LFM2 {name} method {method_name!r} differs from its "
                    "factory-owned receipt."
                )
            method_records[method_name] = source_callable_semantics_record(
                observed_method,
                f"dynamic {name}.{method_name}",
                trusted_unloaded_globals=trusted_unloaded_globals,
            )
        call_override_names = (
            "__call__",
            "_wrapped_call_impl",
            "_call_impl",
            "_slow_forward",
        )
        bases = [
            f"{base.__module__}.{base.__qualname__}" for base in runtime_class.__bases__
        ]
        class_records[name] = {
            "module": expected_module,
            "qualname": runtime_class.__qualname__,
            "bases": bases,
            "local_call_overrides": [
                method_name
                for method_name in call_override_names
                if method_name in vars(runtime_class)
            ],
            "resolved_call": source_callable_semantics_record(
                runtime_class.__call__,
                f"dynamic {name}.__call__",
                trusted_unloaded_globals=trusted_unloaded_globals,
            ),
            "resolved_call_impl": source_callable_semantics_record(
                runtime_class._call_impl,
                f"dynamic {name}._call_impl",
                trusted_unloaded_globals=trusted_unloaded_globals,
            ),
            "factory_methods": method_records,
        }
        if runtime_class.__module__ != dynamic_blank.__name__:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Dynamic LFM2 {name} class has a foreign defining module."
            )
    return {
        "schema": "neural-forge-lfm2-t5-dynamic-runtime-receipt-v1",
        "materializer_source_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "classes": class_records,
    }


def _independent_dynamic_runtime_receipt(
    dynamic_blank: types.ModuleType,
) -> dict[str, Any]:
    expected_module = dynamic_blank.__name__
    live_body = _dynamic_runtime_receipt_body(
        dynamic_blank,
        expected_module=expected_module,
        trusted_unloaded_globals=False,
    )
    fresh_dynamic, source_sha256 = load_fresh_source_module(
        dynamic_blank,
        "neural_forge._neural_forge_runtime_auth_dynamic_blank_materializer",
        "dynamic LFM2 materializer runtime",
    )
    fresh_body = _dynamic_runtime_receipt_body(
        fresh_dynamic,
        expected_module=expected_module,
        trusted_unloaded_globals=True,
    )
    if (
        source_sha256 != fresh_body["materializer_source_sha256"]
        or live_body != fresh_body
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Dynamic LFM2 runtime differs from its independently loaded source."
        )
    return fresh_body


def _verify_canonical_runtime_receipt() -> str:
    native_capture_sha256 = verify_lfm2_runtime_semantics_receipt()
    native_capture_source_sha256 = lfm2_runtime_semantics_receipt_source_sha256()
    receipt = _CANONICAL_RUNTIME_RECEIPT
    claimed = _digest(
        _CANONICAL_RUNTIME_RECEIPT_SHA256,
        "canonical_runtime_receipt_sha256",
    )
    from . import hf_lfm2_dynamic_blank_materializer as dynamic_blank

    live_dynamic_target = _dynamic_runtime_receipt_body(
        dynamic_blank,
        expected_module=dynamic_blank.__name__,
        trusted_unloaded_globals=False,
    )
    if (
        not receipt
        or canonical_sha256(dict(receipt)) != claimed
        or receipt.get("schema") != "neural-forge-lfm2-t5-composed-runtime-receipt-v1"
        or receipt.get("native_capture_runtime")
        != {
            "schema": LFM2_RUNTIME_SEMANTICS_RECEIPT_SCHEMA,
            "receipt_sha256": native_capture_sha256,
            "verifier_source_sha256": native_capture_source_sha256,
        }
        or live_dynamic_target != receipt.get("dynamic_target_runtime")
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Live runtime differs from its independent composed source receipt."
        )
    return claimed


def _canonical_runtime_receipt_sha256() -> str:
    """Return the reauthenticated installed-runtime receipt identity."""

    return _verify_canonical_runtime_receipt()


def _verify_cache_runtime_receipt(cache: Lfm2CircuitTraceCache) -> str:
    if type(cache) is not Lfm2CircuitTraceCache:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Canonical runtime receipt requires the exact trace-cache type."
        )
    receipt_sha256 = _verify_canonical_runtime_receipt()
    native_capture_receipt = lfm2_runtime_semantics_receipt()
    native_capture_receipt_sha256 = verify_lfm2_runtime_semantics_receipt()
    native_capture_receipt_source_sha256 = (
        lfm2_runtime_semantics_receipt_source_sha256()
    )
    implementation = cache.manifest.get("implementation")
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("schema") != "neural-forge-lfm2-circuit-implementation-v2"
        or implementation.get("transformers_lfm2_source_sha256")
        != native_capture_receipt.get("lfm2_source_sha256")
        or implementation.get("torch_version")
        != native_capture_receipt.get("torch_distribution_version")
        or implementation.get("transformers_version")
        != native_capture_receipt.get("transformers_distribution_version")
        or implementation.get("runtime_semantics_receipt_schema")
        != LFM2_RUNTIME_SEMANTICS_RECEIPT_SCHEMA
        or implementation.get("runtime_semantics_receipt") != native_capture_receipt
        or implementation.get("runtime_semantics_receipt_sha256")
        != native_capture_receipt_sha256
        or implementation.get("runtime_semantics_receipt_source_sha256")
        != native_capture_receipt_source_sha256
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Trace-cache implementation differs from the canonical runtime receipt."
        )
    return receipt_sha256


_INTERNAL_RUNTIME_METHOD_CONTRACTS = _capture_internal_runtime_method_contracts()


def _internal_runtime_semantics_sha256(model: Any, label: str) -> str:
    """Authenticate exact internal callables without inspecting root forward."""

    import torch
    from transformers.models.lfm2 import modeling_lfm2
    from torch.nn.modules import module as torch_module_runtime

    canonical_runtime_receipt_sha256 = _verify_canonical_runtime_receipt()

    live_function_contracts = {
        "torch.nn.functional.silu": torch.nn.functional.silu,
        "torch.nn.functional.linear": torch.nn.functional.linear,
        "torch.nn.functional.conv1d": torch.nn.functional.conv1d,
        "torch.nn.Module.__call__": torch.nn.Module.__call__,
        "torch.nn.Module._wrapped_call_impl": torch.nn.Module._wrapped_call_impl,
        "torch.nn.Module._call_impl": torch.nn.Module._call_impl,
        "torch.nn.Module._slow_forward": torch.nn.Module._slow_forward,
        "torch.nn.Conv1d._conv_forward": torch.nn.Conv1d._conv_forward,
        "Lfm2ShortConv.slow_forward": modeling_lfm2.Lfm2ShortConv.slow_forward,
        "Lfm2ShortConv.cuda_kernels_forward": (
            modeling_lfm2.Lfm2ShortConv.cuda_kernels_forward
        ),
    }
    if modeling_lfm2.F is not torch.nn.functional or any(
        live_function_contracts[name] is not _INTERNAL_RUNTIME_FUNCTION_CONTRACTS[name]
        for name in live_function_contracts
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{label} internal activation/projection runtime was replaced."
        )
    live_value_contracts = {
        "modeling_lfm2.torch": modeling_lfm2.torch,
        "modeling_lfm2.nn": modeling_lfm2.nn,
        "modeling_lfm2.apply_rotary_pos_emb": (modeling_lfm2.apply_rotary_pos_emb),
        "modeling_lfm2.eager_attention_forward": (
            modeling_lfm2.eager_attention_forward
        ),
        "modeling_lfm2.apply_mask_to_padding_states": (
            modeling_lfm2.apply_mask_to_padding_states
        ),
        "modeling_lfm2.rotate_half": modeling_lfm2.rotate_half,
        "modeling_lfm2.repeat_kv": modeling_lfm2.repeat_kv,
        "modeling_lfm2.is_torchdynamo_compiling": (
            modeling_lfm2.is_torchdynamo_compiling
        ),
        "modeling_lfm2.is_fast_path_available": (modeling_lfm2.is_fast_path_available),
        "modeling_lfm2.causal_conv1d_fn": modeling_lfm2.causal_conv1d_fn,
        "modeling_lfm2.causal_conv1d_update": (modeling_lfm2.causal_conv1d_update),
        "modeling_lfm2.ALL_ATTENTION_FUNCTIONS": (
            modeling_lfm2.ALL_ATTENTION_FUNCTIONS
        ),
        "attention_interface.eager": (
            modeling_lfm2.ALL_ATTENTION_FUNCTIONS.get_interface(
                "eager",
                modeling_lfm2.eager_attention_forward,
            )
        ),
        "attention_interface.sdpa": (
            modeling_lfm2.ALL_ATTENTION_FUNCTIONS.get_interface(
                "sdpa",
                modeling_lfm2.eager_attention_forward,
            )
        ),
    }
    if any(
        live_value_contracts[name] is not _INTERNAL_RUNTIME_VALUE_CONTRACTS[name]
        for name in live_value_contracts
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{label} internal LFM2 semantic indirection was replaced."
        )
    global_forward_hook_names = (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
        "_global_forward_pre_hooks_with_kwargs",
    )
    if any(
        bool(getattr(torch_module_runtime, name, {}))
        for name in global_forward_hook_names
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{label} cannot authenticate a runtime with global forward hooks."
        )

    layer_container = getattr(getattr(model, "model", None), "layers", None)
    if layer_container is None:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{label} does not expose model.layers for runtime authentication."
        )
    layers = tuple(layer_container)
    if not layers:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{label} has no decoder layers for runtime authentication."
        )
    ordered_modules: list[tuple[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        prefix = f"model.layers.{layer_index}"
        operator = (
            layer.self_attn
            if bool(getattr(layer, "is_attention_layer", False))
            else layer.conv
        )
        feed_forward = layer.feed_forward
        if "act_fn" in vars(feed_forward):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"{label} layer {layer_index} carries a foreign activation override."
            )
        ordered_modules.extend(
            (
                (prefix, layer),
                (f"{prefix}.operator_norm", layer.operator_norm),
                (f"{prefix}.ffn_norm", layer.ffn_norm),
                (f"{prefix}.operator", operator),
                (f"{prefix}.feed_forward", feed_forward),
            )
        )
        for projection_name in ("w1", "w3", "w2"):
            ordered_modules.append(
                (
                    f"{prefix}.feed_forward.{projection_name}",
                    getattr(feed_forward, projection_name),
                )
            )
        if bool(getattr(layer, "is_attention_layer", False)):
            attention_backend = getattr(operator.config, "_attn_implementation", None)
            if attention_backend not in {"eager", "sdpa"} or (
                modeling_lfm2.ALL_ATTENTION_FUNCTIONS.get_interface(
                    attention_backend,
                    modeling_lfm2.eager_attention_forward,
                )
                is not _INTERNAL_RUNTIME_VALUE_CONTRACTS[
                    f"attention_interface.{attention_backend}"
                ]
            ):
                raise Lfm2T5SparseFeatureFoldBuilderError(
                    f"{label} layer {layer_index} attention backend is unauthenticated."
                )
            ordered_modules.extend(
                (
                    (f"{prefix}.self_attn.q_layernorm", operator.q_layernorm),
                    (f"{prefix}.self_attn.k_layernorm", operator.k_layernorm),
                )
            )
            for projection_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                ordered_modules.append(
                    (
                        f"{prefix}.self_attn.{projection_name}",
                        getattr(operator, projection_name),
                    )
                )
        else:
            ordered_modules.extend(
                (
                    (f"{prefix}.conv.in_proj", operator.in_proj),
                    (f"{prefix}.conv.out_proj", operator.out_proj),
                    (f"{prefix}.conv.conv", operator.conv),
                )
            )
            if type(operator.out_proj) is torch.nn.Sequential:
                ordered_modules.extend(
                    (
                        (f"{prefix}.conv.out_proj.0", operator.out_proj[0]),
                        (f"{prefix}.conv.out_proj.1", operator.out_proj[1]),
                    )
                )

    records = []
    seen: set[int] = set()
    for path, module in ordered_modules:
        if id(module) in seen:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"{label} internal runtime module {path!r} aliases another module."
            )
        seen.add(id(module))
        runtime_class = type(module)
        methods = _INTERNAL_RUNTIME_METHOD_CONTRACTS.get(runtime_class)
        if methods is None:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"{label} internal runtime module {path!r} has a foreign class."
            )
        override_names = {"__call__", "forward", *methods}
        if any(name in vars(module) for name in override_names):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"{label} internal runtime module {path!r} has an instance "
                "callable override."
            )
        if (
            getattr(module, "_compiled_call_impl", None) is not None
            or bool(getattr(module, "_forward_pre_hooks", {}))
            or bool(getattr(module, "_forward_hooks", {}))
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"{label} internal runtime module {path!r} has a compiled call "
                "override or forward hooks."
            )
        method_records = {}
        for method_name, expected_method in methods.items():
            if method_name == "__native_forward__":
                native_class = (
                    modeling_lfm2.Lfm2MLP
                    if "MLP" in runtime_class.__qualname__
                    else modeling_lfm2.Lfm2ShortConv
                    if "ShortConv" in runtime_class.__qualname__
                    else modeling_lfm2.Lfm2Attention
                )
                observed_method = native_class.forward
            else:
                observed_method = getattr(runtime_class, method_name, None)
            if observed_method is not expected_method:
                raise Lfm2T5SparseFeatureFoldBuilderError(
                    f"{label} internal runtime class {runtime_class.__qualname__} "
                    f"method {method_name!r} was replaced."
                )
            method_records[method_name] = runtime_callable_record(
                observed_method,
                f"{label}.{path}.{method_name}",
            )
        records.append(
            {
                "path": path,
                "class": (f"{runtime_class.__module__}.{runtime_class.__qualname__}"),
                "methods": method_records,
                "instance_callable_overrides": False,
                "forward_hooks": False,
            }
        )
    return canonical_sha256(
        {
            "schema": LFM2_T5_SPARSE_FEATURE_FOLD_BUILDER_SCHEMA,
            "kind": "live_internal_runtime_semantics",
            "canonical_runtime_receipt_sha256": (canonical_runtime_receipt_sha256),
            "root_forward_consumed_or_authenticated": False,
            "activation": "torch.nn.functional.silu",
            "activation_sha256": canonical_sha256(
                runtime_callable_record(
                    _INTERNAL_RUNTIME_FUNCTION_CONTRACTS["torch.nn.functional.silu"],
                    f"{label}.torch.nn.functional.silu",
                )
            ),
            "delegated_runtime_methods": {
                name: runtime_callable_record(
                    method,
                    f"{label}.{name}",
                )
                for name, method in sorted(live_function_contracts.items())
            },
            "lfm2_semantic_indirections": {
                name: (
                    runtime_callable_record(value, f"{label}.{name}")
                    if callable(value)
                    else {
                        "type": f"{type(value).__module__}.{type(value).__qualname__}",
                        "value": value
                        if type(value) in {bool, int, float, str}
                        else None,
                    }
                )
                for name, value in sorted(live_value_contracts.items())
                if name
                not in {
                    "modeling_lfm2.torch",
                    "modeling_lfm2.nn",
                    "modeling_lfm2.ALL_ATTENTION_FUNCTIONS",
                }
            },
            "ordered_modules": records,
        }
    )


def _model_layers(model: Any, name: str) -> tuple[Any, ...]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} does not expose model.layers."
        )
    result = tuple(layers)
    if len(result) != _EXPECTED_LAYER_COUNT:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} must have exactly {_EXPECTED_LAYER_COUNT} layers."
        )
    return result


def _named_parameters(model: Any, name: str) -> dict[str, Any]:
    callback = getattr(model, "named_parameters", None)
    if not callable(callback):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} does not expose named_parameters()."
        )
    result = dict(callback())
    if not result:
        raise Lfm2T5SparseFeatureFoldBuilderError(f"{name} has no named parameters.")
    return result


def _parameter(
    parameters: Mapping[str, Any],
    path: str,
    *,
    name: str,
) -> Any:
    import torch

    value = parameters.get(path)
    if (
        not torch.is_tensor(value)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.is_complex()
        or not value.is_floating_point()
        or value.ndim not in {2, 3}
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} parameter {path!r} is missing, nonfinite, sparse, or invalid."
        )
    return value


def _mlp_paths(layer_index: int) -> tuple[str, dict[str, str]]:
    site = f"model.layers.{layer_index}.feed_forward"
    return site, {
        "w1": f"{site}.w1.weight",
        "w3": f"{site}.w3.weight",
        "w2": f"{site}.w2.weight",
    }


def _short_conv_paths(layer_index: int) -> tuple[str, dict[str, str]]:
    site = f"model.layers.{layer_index}.conv"
    packed = f"{site}.in_proj.weight"
    return site, {
        "in_proj_b": packed,
        "in_proj_c": packed,
        "in_proj_x": packed,
        "fir_kernel": f"{site}.conv.weight",
        "out_proj": f"{site}.out_proj.weight",
    }


def _short_conv_slices(width: int) -> dict[str, tuple[int, int]]:
    return {
        "in_proj_b": (0, width),
        "in_proj_c": (width, 2 * width),
        "in_proj_x": (2 * width, 3 * width),
    }


def _layer_is_attention(layer: Any, name: str) -> bool:
    value = getattr(layer, "is_attention_layer", None)
    if not isinstance(value, bool):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name}.is_attention_layer must be boolean."
        )
    return value


def _axis_specs(
    *,
    parent_model: Any,
    target_model: Any,
    selected: Lfm2DenseBudgetedSelectedTarget,
) -> tuple[_AxisSpec, ...]:
    parent_layers = _model_layers(parent_model, "parent_model")
    target_layers = _model_layers(target_model, "target.model")
    parent_parameters = _named_parameters(parent_model, "parent_model")
    target_parameters = _named_parameters(target_model, "target.model")
    if (
        selected.target_depth != _EXPECTED_LAYER_COUNT
        or selected.ordered_source_spans
        != tuple((index,) for index in range(_EXPECTED_LAYER_COUNT))
        or len(selected.layer_types) != _EXPECTED_LAYER_COUNT
        or selected.fir_length != 3
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "T5 full-axis construction requires the exact equal-depth FIR-3 target."
        )

    specs: list[_AxisSpec] = []
    observed_short_conv_layers: list[int] = []
    for layer_index, (parent_layer, target_layer, selected_kind) in enumerate(
        zip(parent_layers, target_layers, selected.layer_types, strict=True)
    ):
        source_attention = _layer_is_attention(
            parent_layer, f"parent_model.model.layers.{layer_index}"
        )
        target_attention = _layer_is_attention(
            target_layer, f"target.model.model.layers.{layer_index}"
        )
        selected_attention = selected_kind == "full_attention"
        if (
            source_attention != target_attention
            or source_attention != selected_attention
            or selected_kind not in {"conv", "full_attention"}
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Layer {layer_index} source, target, and selected kinds differ."
            )

        mlp_site, mlp_paths = _mlp_paths(layer_index)
        source_w1 = _parameter(parent_parameters, mlp_paths["w1"], name="parent_model")
        source_w3 = _parameter(parent_parameters, mlp_paths["w3"], name="parent_model")
        source_w2 = _parameter(parent_parameters, mlp_paths["w2"], name="parent_model")
        target_w1 = _parameter(target_parameters, mlp_paths["w1"], name="target.model")
        target_w3 = _parameter(target_parameters, mlp_paths["w3"], name="target.model")
        target_w2 = _parameter(target_parameters, mlp_paths["w2"], name="target.model")
        source_width = int(source_w2.shape[1])
        target_width = int(target_w2.shape[1])
        if (
            source_w1.ndim != 2
            or source_w3.ndim != 2
            or source_w2.ndim != 2
            or target_w1.ndim != 2
            or target_w3.ndim != 2
            or target_w2.ndim != 2
            or int(source_w1.shape[0]) != source_width
            or tuple(source_w3.shape) != tuple(source_w1.shape)
            or int(source_w2.shape[0]) != int(source_w1.shape[1])
            or int(target_w1.shape[0]) != target_width
            or tuple(target_w3.shape) != tuple(target_w1.shape)
            or int(target_w2.shape[0]) != int(target_w1.shape[1])
            or target_width > source_width
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Layer {layer_index} MLP geometry is not a dense contraction."
            )
        specs.append(
            _AxisSpec(
                layer_index=layer_index,
                axis_kind=AXIS_MLP_POST_SWIGLU,
                component="C13",
                endpoint="mlp.post_swiglu",
                writer_role="w2",
                graph_site=mlp_site,
                parameter_paths=MappingProxyType(mlp_paths),
                source_row_slices=MappingProxyType({}),
                target_row_slices=MappingProxyType({}),
                source_width=source_width,
                target_width=target_width,
                parent_writer=source_w2,
            )
        )

        if source_attention:
            continue
        observed_short_conv_layers.append(layer_index)
        conv_site, conv_paths = _short_conv_paths(layer_index)
        source_in = _parameter(
            parent_parameters, conv_paths["in_proj_b"], name="parent_model"
        )
        source_fir = _parameter(
            parent_parameters, conv_paths["fir_kernel"], name="parent_model"
        )
        source_out = _parameter(
            parent_parameters, conv_paths["out_proj"], name="parent_model"
        )
        target_in = _parameter(
            target_parameters, conv_paths["in_proj_b"], name="target.model"
        )
        target_fir = _parameter(
            target_parameters, conv_paths["fir_kernel"], name="target.model"
        )
        target_out = _parameter(
            target_parameters, conv_paths["out_proj"], name="target.model"
        )
        source_conv_width = int(source_out.shape[1])
        target_conv_width = int(target_out.shape[1])
        if (
            source_in.ndim != 2
            or source_fir.ndim != 3
            or source_out.ndim != 2
            or target_in.ndim != 2
            or target_fir.ndim != 3
            or target_out.ndim != 2
            or tuple(source_in.shape)
            != (3 * source_conv_width, int(source_out.shape[0]))
            or tuple(source_fir.shape) != (source_conv_width, 1, 3)
            or tuple(target_in.shape)
            != (3 * target_conv_width, int(target_out.shape[0]))
            or tuple(target_fir.shape) != (target_conv_width, 1, 3)
            or target_conv_width > source_conv_width
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Layer {layer_index} ShortConv geometry is not dense FIR-3."
            )
        specs.append(
            _AxisSpec(
                layer_index=layer_index,
                axis_kind=AXIS_SHORT_CONV_CONTEXT,
                component="C05",
                endpoint="short_conv.context",
                writer_role="out_proj",
                graph_site=conv_site,
                parameter_paths=MappingProxyType(conv_paths),
                source_row_slices=MappingProxyType(
                    _short_conv_slices(source_conv_width)
                ),
                target_row_slices=MappingProxyType(
                    _short_conv_slices(target_conv_width)
                ),
                source_width=source_conv_width,
                target_width=target_conv_width,
                parent_writer=source_out,
            )
        )

    if tuple(observed_short_conv_layers) != _EXPECTED_SHORT_CONV_LAYERS:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "The full-axis production source must expose ShortConv at the exact "
            "eight authenticated layers."
        )
    expected_keys = tuple(
        (layer_index, axis_kind)
        for layer_index in range(_EXPECTED_LAYER_COUNT)
        for axis_kind in (
            AXIS_MLP_POST_SWIGLU,
            *(
                (AXIS_SHORT_CONV_CONTEXT,)
                if layer_index in _EXPECTED_SHORT_CONV_LAYERS
                else ()
            ),
        )
    )
    if len(specs) != 22 or tuple(value.key for value in specs) != expected_keys:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Full-axis T5 construction must resolve exactly 14 MLP and 8 ShortConv axes."
        )
    return tuple(specs)


def _ordered_batch_input_sha256(
    cache: Lfm2CircuitTraceCache,
) -> tuple[str, ...]:
    try:
        values = tuple(
            _digest(value, "ordered_batch_input_sha256")
            for value in cache.manifest["corpus"]["ordered_batch_input_sha256"]
        )
    except (KeyError, TypeError) as exc:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "The sealed cache lacks its ordered input identity."
        ) from exc
    if len(values) != len(cache) or not values:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "The sealed cache ordered input identity is incomplete."
        )
    return values


_SPLIT_RESOLUTION_FACTORY_TOKEN = object()


class Lfm2T5EvidenceSplitResolution:
    """Factory-sealed projection of production's cache split resolution."""

    __slots__ = (
        "_binding_sha256",
        "cache_key",
        "calibration_source_sha256",
        "corpus_fingerprint",
        "evidence_split_sha256",
        "evidence_split_source",
        "maximum_fit_rows",
        "maximum_selection_rows",
        "maximum_validation_rows",
        "on_policy_evidence_identity_sha256",
        "on_policy_extension_cache_keys",
        "ordered_batch_input_sha256",
        "ordered_splits",
        "parent_cache_build_called",
        "parent_cache_reused",
        "source_mode",
        "supplied_calibration_matches_cache",
    )

    def __init__(
        self,
        *,
        cache_key: str,
        corpus_fingerprint: str,
        ordered_batch_input_sha256: tuple[str, ...],
        ordered_splits: tuple[str, ...],
        evidence_split_source: str,
        evidence_split_sha256: str,
        calibration_source_sha256: str | None,
        supplied_calibration_matches_cache: bool | None,
        source_mode: str,
        parent_cache_reused: bool,
        parent_cache_build_called: bool,
        maximum_selection_rows: int,
        maximum_fit_rows: int,
        maximum_validation_rows: int,
        on_policy_evidence_identity_sha256: str | None,
        on_policy_extension_cache_keys: tuple[str, ...],
        _factory_token: object,
    ) -> None:
        if _factory_token is not _SPLIT_RESOLUTION_FACTORY_TOKEN:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Split resolution must be created by its authenticated factory."
            )
        self.cache_key = cache_key
        self.corpus_fingerprint = corpus_fingerprint
        self.ordered_batch_input_sha256 = ordered_batch_input_sha256
        self.ordered_splits = ordered_splits
        self.evidence_split_source = evidence_split_source
        self.evidence_split_sha256 = evidence_split_sha256
        self.calibration_source_sha256 = calibration_source_sha256
        self.supplied_calibration_matches_cache = supplied_calibration_matches_cache
        self.source_mode = source_mode
        self.parent_cache_reused = parent_cache_reused
        self.parent_cache_build_called = parent_cache_build_called
        self.maximum_selection_rows = maximum_selection_rows
        self.maximum_fit_rows = maximum_fit_rows
        self.maximum_validation_rows = maximum_validation_rows
        self.on_policy_evidence_identity_sha256 = on_policy_evidence_identity_sha256
        self.on_policy_extension_cache_keys = on_policy_extension_cache_keys
        self._binding_sha256 = canonical_sha256(self._binding_body())

    def _binding_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_T5_EVIDENCE_SPLIT_RESOLUTION_SCHEMA,
            "cache_key": self.cache_key,
            "corpus_fingerprint": self.corpus_fingerprint,
            "ordered_batch_input_sha256": list(self.ordered_batch_input_sha256),
            "ordered_splits": list(self.ordered_splits),
            "evidence_split_source": self.evidence_split_source,
            "evidence_split_sha256": self.evidence_split_sha256,
            "calibration_source_sha256": self.calibration_source_sha256,
            "supplied_calibration_matches_cache": (
                self.supplied_calibration_matches_cache
            ),
            "source_mode": self.source_mode,
            "parent_cache_reused": self.parent_cache_reused,
            "parent_cache_build_called": self.parent_cache_build_called,
            "maximum_selection_rows": self.maximum_selection_rows,
            "maximum_fit_rows": self.maximum_fit_rows,
            "maximum_validation_rows": self.maximum_validation_rows,
            "on_policy_evidence_identity_sha256": (
                self.on_policy_evidence_identity_sha256
            ),
            "on_policy_extension_cache_keys": list(self.on_policy_extension_cache_keys),
            "base_cache_only": True,
        }

    @property
    def binding_sha256(self) -> str:
        return self._binding_sha256

    def verify(
        self,
        cache: Lfm2CircuitTraceCache,
    ) -> "Lfm2T5EvidenceSplitResolution":
        if type(self) is not Lfm2T5EvidenceSplitResolution:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Split resolution must have the exact authenticated type."
            )
        if type(cache) is not Lfm2CircuitTraceCache:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Split resolution requires the exact sealed cache type."
            )
        ordered_inputs = _ordered_batch_input_sha256(cache)
        splits = tuple(self.ordered_splits)
        if (
            self.cache_key != cache.cache_key
            or self.corpus_fingerprint != cache.corpus_fingerprint
            or tuple(self.ordered_batch_input_sha256) != ordered_inputs
            or len(splits) != len(cache)
            or set(splits) != {"selection", "fit", "validation"}
            or any(value != str(value).strip().lower() for value in splits)
            or self.source_mode != "sealed_existing"
            or self.parent_cache_reused is not True
            or self.parent_cache_build_called is not False
            or self.on_policy_evidence_identity_sha256 is not None
            or tuple(self.on_policy_extension_cache_keys)
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Split resolution is not the exact base-only sealed-cache authority."
            )
        for name in (
            "maximum_selection_rows",
            "maximum_fit_rows",
            "maximum_validation_rows",
        ):
            _positive_int(getattr(self, name), name)
        if (
            self.evidence_split_source != "sealed_cache_deterministic"
            or self.calibration_source_sha256 is not None
            or self.supplied_calibration_matches_cache is not None
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "T5 construction only admits the cache-derived deterministic split."
            )
        ranked = sorted(
            range(len(ordered_inputs)),
            key=lambda index: (ordered_inputs[index], index),
        )
        expected_splits = ["fit"] * len(ordered_inputs)
        expected_splits[ranked[0]] = "selection"
        expected_splits[ranked[-1]] = "validation"
        expected_split_sha256 = canonical_sha256(
            {
                "schema": _PRODUCTION_SEALED_CACHE_SPLIT_SCHEMA,
                "parent_cache_key": cache.cache_key,
                "corpus_fingerprint": cache.corpus_fingerprint,
                "ordered_batch_input_sha256": list(ordered_inputs),
                "ordered_splits": expected_splits,
                "assignment": "content_digest_rank_extrema_v1",
            }
        )
        if (
            splits != tuple(expected_splits)
            or _digest(
                self.evidence_split_sha256,
                "evidence_split_sha256",
            )
            != expected_split_sha256
            or canonical_sha256(self._binding_body())
            != _digest(self._binding_sha256, "split_resolution.binding_sha256")
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Split resolution preimage or immutable binding changed."
            )
        return self


def build_lfm2_t5_evidence_split_resolution(
    *,
    cache: Lfm2CircuitTraceCache,
    ordered_splits: Sequence[str],
    evidence_split_source: str,
    evidence_split_sha256: str,
    source_mode: str = "sealed_existing",
    parent_cache_reused: bool = True,
    parent_cache_build_called: bool = False,
    maximum_selection_rows: int = 256,
    maximum_fit_rows: int = 512,
    maximum_validation_rows: int = 128,
    on_policy_evidence_identity_sha256: str | None = None,
    on_policy_extension_cache_keys: Sequence[str] = (),
) -> Lfm2T5EvidenceSplitResolution:
    """Seal the exact split fields emitted by production cache resolution."""

    if type(cache) is not Lfm2CircuitTraceCache:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Split resolution requires the exact sealed cache type."
        )
    value = Lfm2T5EvidenceSplitResolution(
        cache_key=cache.cache_key,
        corpus_fingerprint=cache.corpus_fingerprint,
        ordered_batch_input_sha256=_ordered_batch_input_sha256(cache),
        ordered_splits=tuple(str(item).strip().lower() for item in ordered_splits),
        evidence_split_source=str(evidence_split_source),
        evidence_split_sha256=evidence_split_sha256,
        calibration_source_sha256=None,
        supplied_calibration_matches_cache=None,
        source_mode=str(source_mode),
        parent_cache_reused=parent_cache_reused,
        parent_cache_build_called=parent_cache_build_called,
        maximum_selection_rows=maximum_selection_rows,
        maximum_fit_rows=maximum_fit_rows,
        maximum_validation_rows=maximum_validation_rows,
        on_policy_evidence_identity_sha256=(on_policy_evidence_identity_sha256),
        on_policy_extension_cache_keys=tuple(on_policy_extension_cache_keys),
        _factory_token=_SPLIT_RESOLUTION_FACTORY_TOKEN,
    )
    return value.verify(cache)


@dataclass(frozen=True, slots=True)
class _EvidenceContext:
    cache: Lfm2CircuitTraceCache
    session: Lfm2CircuitTraceSelectiveAuthenticatedSession
    paths: tuple[str, ...]
    path_splits: tuple[str, ...]
    parsed_paths: tuple[_ParsedPath, ...]
    path_plan_sha256: str
    source_layer_topology_sha256: str
    source_layer_topology_document_sha256: str
    canonical_runtime_receipt_sha256: str
    selection_indices: tuple[int, ...]
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    partition_sha256: str
    split_resolution_sha256: str
    session_evidence: Mapping[str, Any]


_EVIDENCE_AUTHORITY_FACTORY_TOKEN = object()


def _path_plan_sha256(
    *,
    cache: Lfm2CircuitTraceCache,
    session_evidence: Mapping[str, Any],
    paths: Sequence[str],
    path_splits: Sequence[str],
    split_resolution_sha256: str,
    source_layer_topology_sha256: str,
    source_layer_topology_document_sha256: str,
    canonical_runtime_receipt_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema": LFM2_T5_EVIDENCE_PATH_PLAN_SCHEMA,
            "cache_key": cache.cache_key,
            "corpus_fingerprint": cache.corpus_fingerprint,
            "manifest_sha256": session_evidence["manifest_sha256"],
            "completion_sha256": session_evidence["completion_sha256"],
            "selective_session_sha256": session_evidence["session_sha256"],
            "paths_sha256": canonical_sha256(list(paths)),
            "path_splits_sha256": canonical_sha256(list(path_splits)),
            "path_count": len(paths),
            "split_resolution_sha256": split_resolution_sha256,
            "source_layer_topology_sha256": source_layer_topology_sha256,
            "source_layer_topology_document_sha256": (
                source_layer_topology_document_sha256
            ),
            "canonical_runtime_receipt_sha256": canonical_runtime_receipt_sha256,
            "trace_row_index_consumed": False,
            "selected_payloads_are_numeric_authority": True,
        }
    )


class Lfm2T5SparseFeatureEvidenceAuthority:
    """Factory-owned proof of one immutable base-cache row partition."""

    __slots__ = (
        "_authority_sha256",
        "cache",
        "canonical_runtime_receipt_sha256",
        "fit_indices",
        "path_plan_sha256",
        "path_splits",
        "paths",
        "partition_sha256",
        "selection_indices",
        "session",
        "session_evidence",
        "source_layer_topology_document_sha256",
        "source_layer_topology_sha256",
        "split_resolution",
        "validation_indices",
    )

    def __init__(
        self,
        *,
        cache: Lfm2CircuitTraceCache,
        session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
        split_resolution: Lfm2T5EvidenceSplitResolution,
        paths: tuple[str, ...],
        path_splits: tuple[str, ...],
        path_plan_sha256: str,
        source_layer_topology_sha256: str,
        source_layer_topology_document_sha256: str,
        canonical_runtime_receipt_sha256: str,
        selection_indices: tuple[int, ...],
        fit_indices: tuple[int, ...],
        validation_indices: tuple[int, ...],
        partition_sha256: str,
        session_evidence: Mapping[str, Any],
        _factory_token: object,
    ) -> None:
        if _factory_token is not _EVIDENCE_AUTHORITY_FACTORY_TOKEN:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Evidence authority must be created by its authenticated factory."
            )
        self.cache = cache
        self.session = session
        self.split_resolution = split_resolution
        self.paths = paths
        self.path_splits = path_splits
        self.path_plan_sha256 = path_plan_sha256
        self.source_layer_topology_sha256 = source_layer_topology_sha256
        self.source_layer_topology_document_sha256 = (
            source_layer_topology_document_sha256
        )
        self.canonical_runtime_receipt_sha256 = canonical_runtime_receipt_sha256
        self.selection_indices = selection_indices
        self.fit_indices = fit_indices
        self.validation_indices = validation_indices
        self.partition_sha256 = partition_sha256
        self.session_evidence = MappingProxyType(dict(session_evidence))
        self._authority_sha256 = canonical_sha256(self._authority_body())
        self.verify()

    def _authority_body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_T5_SPARSE_FEATURE_FOLD_BUILDER_SCHEMA,
            "cache_key": self.cache.cache_key,
            "corpus_fingerprint": self.cache.corpus_fingerprint,
            "selective_session_sha256": self.session.evidence_sha256,
            "path_plan_sha256": self.path_plan_sha256,
            "source_layer_topology_sha256": (self.source_layer_topology_sha256),
            "source_layer_topology_document_sha256": (
                self.source_layer_topology_document_sha256
            ),
            "canonical_runtime_receipt_sha256": (self.canonical_runtime_receipt_sha256),
            "paths_sha256": canonical_sha256(list(self.paths)),
            "path_splits_sha256": canonical_sha256(list(self.path_splits)),
            "split_resolution_sha256": self.split_resolution.binding_sha256,
            "selection_indices": list(self.selection_indices),
            "fit_indices": list(self.fit_indices),
            "validation_indices": list(self.validation_indices),
            "partition_sha256": self.partition_sha256,
            "session_evidence": dict(self.session_evidence),
        }

    @property
    def authority_sha256(self) -> str:
        self.verify()
        return self._authority_sha256

    def verify(self) -> "Lfm2T5SparseFeatureEvidenceAuthority":
        if type(self) is not Lfm2T5SparseFeatureEvidenceAuthority:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Evidence authority must have the exact factory-owned type."
            )
        if (
            type(self.cache) is not Lfm2CircuitTraceCache
            or type(self.session) is not Lfm2CircuitTraceSelectiveAuthenticatedSession
            or type(self.split_resolution) is not Lfm2T5EvidenceSplitResolution
            or self.session.closed
            or self.session.cache_key != self.cache.cache_key
            or self.session.path != self.cache.path
            or len(self.session) != len(self.cache)
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "T5 evidence authority lost its base cache or selective session."
            )
        _verify_selective_session(self.session)
        self.split_resolution.verify(self.cache)
        expected_runtime_receipt_sha256 = _verify_cache_runtime_receipt(self.cache)
        (
            expected_topology_sha256,
            expected_topology_document_sha256,
        ) = _cache_layer_topology_evidence(self.cache)
        phases = _phases_by_batch(self.session)
        expected_paths, expected_splits = _paths_from_cache_inventory(
            self.session,
            phases,
            self.split_resolution.ordered_splits,
        )
        if (
            self.paths != expected_paths
            or self.path_splits != expected_splits
            or self.source_layer_topology_sha256 != expected_topology_sha256
            or self.source_layer_topology_document_sha256
            != expected_topology_document_sha256
            or self.canonical_runtime_receipt_sha256 != expected_runtime_receipt_sha256
            or self.session.as_dict() != dict(self.session_evidence)
            or self.path_plan_sha256
            != _path_plan_sha256(
                cache=self.cache,
                session_evidence=self.session_evidence,
                paths=expected_paths,
                path_splits=expected_splits,
                split_resolution_sha256=(self.split_resolution.binding_sha256),
                source_layer_topology_sha256=expected_topology_sha256,
                source_layer_topology_document_sha256=(
                    expected_topology_document_sha256
                ),
                canonical_runtime_receipt_sha256=(expected_runtime_receipt_sha256),
            )
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Evidence authority paths, splits, or session seal changed."
            )
        selection, fit, validation, partition_sha256 = _canonical_partition(
            expected_paths,
            expected_splits,
            path_plan_sha256=self.path_plan_sha256,
            cache=self.cache,
            maximum_selection_rows=(self.split_resolution.maximum_selection_rows),
            maximum_fit_rows=self.split_resolution.maximum_fit_rows,
            maximum_validation_rows=(self.split_resolution.maximum_validation_rows),
        )
        if (
            selection != self.selection_indices
            or fit != self.fit_indices
            or validation != self.validation_indices
            or partition_sha256 != self.partition_sha256
            or canonical_sha256(self._authority_body())
            != _digest(self._authority_sha256, "authority_sha256")
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Evidence authority partition or immutable seal changed."
            )
        return self


def _phases_by_batch(
    session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
) -> tuple[tuple[str, ...], ...]:
    result = []
    for reference in session.batch_references:
        inventory = reference.record.get("tensors")
        if not isinstance(inventory, Mapping):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Cache batch {reference.index} has no tensor inventory."
            )
        phases = {
            name.removesuffix(".layer.000.residual.pre")
            for name in inventory
            if name.endswith(".layer.000.residual.pre")
        }
        ordered = tuple(
            sorted(
                phases,
                key=lambda phase: (
                    0 if phase == "prefill" else 1,
                    -1 if phase == "prefill" else int(phase.removeprefix("decode.")),
                ),
            )
        )
        expected = (
            "prefill",
            *tuple(f"decode.{index:03d}" for index in range(len(ordered) - 1)),
        )
        if not ordered or ordered != expected:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Cache batch {reference.index} phase inventory is not canonical."
            )
        result.append(ordered)
    return tuple(result)


def _paths_from_cache_inventory(
    session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
    phases_by_batch: Sequence[Sequence[str]],
    calibration_splits: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    splits = tuple(str(value).strip().lower() for value in calibration_splits)
    if len(splits) != len(session) or set(splits) != {"selection", "fit", "validation"}:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Base-cache calibration splits must cover selection, fit, and validation."
        )
    paths: list[str] = []
    row_splits: list[str] = []
    for reference, phases, split in zip(
        session.batch_references, phases_by_batch, splits, strict=True
    ):
        inventory = reference.record["tensors"]
        for phase in phases:
            descriptor = inventory.get(f"{phase}.layer.000.residual.pre")
            shape = descriptor.get("shape") if isinstance(descriptor, Mapping) else None
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or shape[0] != 1
                or isinstance(shape[1], bool)
                or not isinstance(shape[1], int)
                or shape[1] <= 0
            ):
                raise Lfm2T5SparseFeatureFoldBuilderError(
                    f"Cache batch {reference.index} phase {phase!r} anchor is invalid."
                )
            paths.extend(
                f"batch.{reference.index:06d}/{phase}/row.{row:04d}"
                for row in range(shape[1])
            )
            row_splits.extend(split for _ in range(shape[1]))
    return tuple(paths), tuple(row_splits)


def _bounded_pool(pool: Sequence[int], maximum: int) -> tuple[int, ...]:
    if not pool or maximum <= 0:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Evidence pools and bounds must be positive."
        )
    if len(pool) <= maximum:
        return tuple(pool)
    return tuple(pool[(index * len(pool)) // maximum] for index in range(maximum))


def _canonical_partition(
    paths: Sequence[str],
    path_splits: Sequence[str],
    *,
    path_plan_sha256: str,
    cache: Lfm2CircuitTraceCache,
    maximum_selection_rows: int,
    maximum_fit_rows: int,
    maximum_validation_rows: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], str]:
    parsed = _parse_paths(paths, path_splits)
    selection_pool = [
        value.global_index for value in parsed if value.split == "selection"
    ]
    fit_pool = [value.global_index for value in parsed if value.split == "fit"]
    validation_pool = [
        value.global_index for value in parsed if value.split == "validation"
    ]
    selection = _bounded_pool(selection_pool, maximum_selection_rows)
    fit = _bounded_pool(fit_pool, maximum_fit_rows)
    validation = _bounded_pool(validation_pool, maximum_validation_rows)
    partition_body = {
        "schema": LFM2_T5_EVIDENCE_PARTITION_SCHEMA,
        "cache_key": cache.cache_key,
        "corpus_fingerprint": cache.corpus_fingerprint,
        "path_plan_sha256": path_plan_sha256,
        "paths_sha256": canonical_sha256(list(paths)),
        "path_splits_sha256": canonical_sha256(list(path_splits)),
        "selection_indices": list(selection),
        "fit_indices": list(fit),
        "validation_indices": list(validation),
        "selection_rows_sha256": canonical_sha256(list(selection)),
        "fit_rows_sha256": canonical_sha256(list(fit)),
        "validation_rows_sha256": canonical_sha256(list(validation)),
        "maximum_selection_rows": maximum_selection_rows,
        "maximum_fit_rows": maximum_fit_rows,
        "maximum_validation_rows": maximum_validation_rows,
        "row_sets_pairwise_disjoint": True,
        "selection_rows_used_for_fit": False,
    }
    return selection, fit, validation, canonical_sha256(partition_body)


def build_lfm2_t5_sparse_feature_evidence_authority(
    *,
    cache: Lfm2CircuitTraceCache,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
    split_resolution: Lfm2T5EvidenceSplitResolution,
) -> Lfm2T5SparseFeatureEvidenceAuthority:
    """Seal a cache-derived path plan; numeric rows remain selected-payload owned."""

    if (
        type(cache) is not Lfm2CircuitTraceCache
        or type(selective_session) is not Lfm2CircuitTraceSelectiveAuthenticatedSession
        or type(split_resolution) is not Lfm2T5EvidenceSplitResolution
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "T5 evidence inputs must have their exact authenticated types."
        )
    if selective_session.closed:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "The selective cache session is closed."
        )
    _verify_selective_session(selective_session)
    split_resolution.verify(cache)
    canonical_runtime_receipt_sha256 = _verify_cache_runtime_receipt(cache)
    if (
        selective_session.cache_key != cache.cache_key
        or selective_session.path != cache.path
        or len(selective_session) != len(cache)
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Cache and selective session have different identities."
        )
    phases = _phases_by_batch(selective_session)
    paths, path_splits = _paths_from_cache_inventory(
        selective_session,
        phases,
        split_resolution.ordered_splits,
    )
    session_evidence = selective_session.as_dict()
    for name in ("session_sha256", "manifest_sha256", "completion_sha256"):
        _digest(session_evidence.get(name), f"selective_session.{name}")
    (
        source_layer_topology_sha256,
        source_layer_topology_document_sha256,
    ) = _cache_layer_topology_evidence(cache)
    path_plan_sha256 = _path_plan_sha256(
        cache=cache,
        session_evidence=session_evidence,
        paths=paths,
        path_splits=path_splits,
        split_resolution_sha256=split_resolution.binding_sha256,
        source_layer_topology_sha256=source_layer_topology_sha256,
        source_layer_topology_document_sha256=(source_layer_topology_document_sha256),
        canonical_runtime_receipt_sha256=canonical_runtime_receipt_sha256,
    )
    selection, fit, validation, partition_sha256 = _canonical_partition(
        paths,
        path_splits,
        path_plan_sha256=path_plan_sha256,
        cache=cache,
        maximum_selection_rows=split_resolution.maximum_selection_rows,
        maximum_fit_rows=split_resolution.maximum_fit_rows,
        maximum_validation_rows=split_resolution.maximum_validation_rows,
    )
    return Lfm2T5SparseFeatureEvidenceAuthority(
        cache=cache,
        session=selective_session,
        split_resolution=split_resolution,
        paths=paths,
        path_splits=path_splits,
        path_plan_sha256=path_plan_sha256,
        source_layer_topology_sha256=source_layer_topology_sha256,
        source_layer_topology_document_sha256=(source_layer_topology_document_sha256),
        canonical_runtime_receipt_sha256=canonical_runtime_receipt_sha256,
        selection_indices=selection,
        fit_indices=fit,
        validation_indices=validation,
        partition_sha256=partition_sha256,
        session_evidence=session_evidence,
        _factory_token=_EVIDENCE_AUTHORITY_FACTORY_TOKEN,
    )


def _evidence_context(
    authority: Lfm2T5SparseFeatureEvidenceAuthority,
) -> _EvidenceContext:
    if type(authority) is not Lfm2T5SparseFeatureEvidenceAuthority:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "evidence_authority has the wrong type."
        )
    authority.verify()
    paths = authority.paths
    path_splits = authority.path_splits
    return _EvidenceContext(
        cache=authority.cache,
        session=authority.session,
        paths=paths,
        path_splits=path_splits,
        parsed_paths=_parse_paths(paths, path_splits),
        path_plan_sha256=authority.path_plan_sha256,
        source_layer_topology_sha256=(authority.source_layer_topology_sha256),
        source_layer_topology_document_sha256=(
            authority.source_layer_topology_document_sha256
        ),
        canonical_runtime_receipt_sha256=(authority.canonical_runtime_receipt_sha256),
        selection_indices=authority.selection_indices,
        fit_indices=authority.fit_indices,
        validation_indices=authority.validation_indices,
        partition_sha256=authority.partition_sha256,
        split_resolution_sha256=authority.split_resolution.binding_sha256,
        session_evidence=authority.session_evidence,
    )


@dataclass(frozen=True, slots=True)
class _OwnedMatrixSeal:
    object_id: int
    tensor_version: int
    tensor_sha256: str
    t5_tensor_sha256: str
    shape: tuple[int, ...]
    dtype: str
    device: str


def _owned_matrix_seal(value: Any, name: str) -> _OwnedMatrixSeal:
    return _OwnedMatrixSeal(
        object_id=id(value),
        tensor_version=_writer_tensor_version(value),
        tensor_sha256=_tensor_sha256(value, name),
        t5_tensor_sha256=t5_tensor_sha256(value),
        shape=tuple(int(size) for size in value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
    )


def _verify_owned_matrix(
    value: Any,
    seal: _OwnedMatrixSeal,
    name: str,
) -> None:
    if (
        id(value) != seal.object_id
        or _writer_tensor_version(value) != seal.tensor_version
        or tuple(int(size) for size in value.shape) != seal.shape
        or str(value.dtype) != seal.dtype
        or str(value.device) != seal.device
        or _tensor_sha256(value, name) != seal.tensor_sha256
        or t5_tensor_sha256(value) != seal.t5_tensor_sha256
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{name} changed after its authenticated selected-row snapshot."
        )


@dataclass(frozen=True, slots=True)
class _SelectedAxisRows:
    fit_source: Any
    heldout_source: Any
    fit_float64: Any
    heldout_float64: Any
    fit_source_seal: _OwnedMatrixSeal
    heldout_source_seal: _OwnedMatrixSeal
    fit_float64_seal: _OwnedMatrixSeal
    heldout_float64_seal: _OwnedMatrixSeal
    selection_sha256: str
    selection_body: Mapping[str, Any]


def _verify_selected_axis_rows(
    rows: _SelectedAxisRows,
    axis_name: str,
) -> None:
    pairs = (
        ("fit_source", rows.fit_source, rows.fit_source_seal),
        ("heldout_source", rows.heldout_source, rows.heldout_source_seal),
        ("fit_float64", rows.fit_float64, rows.fit_float64_seal),
        ("heldout_float64", rows.heldout_float64, rows.heldout_float64_seal),
    )
    if len({id(value) for _name, value, _seal in pairs}) != len(pairs):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{axis_name} selected-row matrices unexpectedly alias."
        )
    for name, value, seal in pairs:
        _verify_owned_matrix(value, seal, f"{axis_name}.{name}")
    body = dict(rows.selection_body)
    expected_hashes = {
        "fit_source_dtype_rows_sha256": rows.fit_source_seal.tensor_sha256,
        "heldout_source_dtype_rows_sha256": (rows.heldout_source_seal.tensor_sha256),
        "fit_float64_rows_sha256": rows.fit_float64_seal.tensor_sha256,
        "heldout_float64_rows_sha256": (rows.heldout_float64_seal.tensor_sha256),
        "fit_source_dtype_t5_rows_sha256": (rows.fit_source_seal.t5_tensor_sha256),
        "heldout_source_dtype_t5_rows_sha256": (
            rows.heldout_source_seal.t5_tensor_sha256
        ),
        "fit_float64_t5_rows_sha256": rows.fit_float64_seal.t5_tensor_sha256,
        "heldout_float64_t5_rows_sha256": (rows.heldout_float64_seal.t5_tensor_sha256),
    }
    if any(
        body.get(key) != value for key, value in expected_hashes.items()
    ) or canonical_sha256(body) != _digest(
        rows.selection_sha256, f"{axis_name}.selection_sha256"
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            f"{axis_name} selected-row selection seal changed."
        )


def _release_selected_rows_payload(
    session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
    payload: Any,
) -> None:
    """Clear owned row tensors even when session reauthentication rejects release."""

    release_error: BaseException | None = None
    try:
        release_lfm2_circuit_trace_selected_rows(session, payload)
    except BaseException as exc:
        release_error = exc
    try:
        payload._release()
    except BaseException as exc:  # pragma: no cover - exact payload has trivial clear
        if release_error is not None:
            release_error.add_note(
                "Owned selected-row payload cleanup also failed: "
                f"{type(exc).__qualname__}: {exc}"
            )
        else:
            raise
    if not bool(getattr(payload, "released", False)):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Selected-row payload could not be unconditionally released."
        )
    if release_error is not None:
        raise release_error


def _selected_axis_rows(
    *,
    evidence: _EvidenceContext,
    specs: Sequence[_AxisSpec],
    parent_identity_sha256: str,
    parent_state_sha256: str,
    source_graph_sha256: str,
    target_blank_state_sha256: str,
    target_topology_sha256: str,
    target_live_layer_topology_sha256: str,
    target_live_layer_topology_document_sha256: str,
    source_runtime_semantics_sha256: str,
    target_runtime_semantics_sha256: str,
) -> tuple[Mapping[tuple[int, str], _SelectedAxisRows], tuple[Mapping[str, Any], ...]]:
    import torch

    selected_indices = tuple(
        sorted((*evidence.fit_indices, *evidence.validation_indices))
    )
    selected_set = set(selected_indices)
    if selected_set & set(evidence.selection_indices):  # defensive duplicate boundary
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Selection rows entered the T5 row-read plan."
        )
    by_batch: dict[int, list[_ParsedPath]] = {}
    for index in selected_indices:
        value = evidence.parsed_paths[index]
        by_batch.setdefault(value.batch_index, []).append(value)

    rows_by_axis: dict[tuple[int, str], dict[int, Any]] = {
        spec.key: {} for spec in specs
    }
    payload_records: list[Mapping[str, Any]] = []
    references = evidence.session.batch_references
    for batch_index in sorted(by_batch):
        batch_paths = tuple(
            sorted(by_batch[batch_index], key=lambda value: value.global_index)
        )
        row_plan: dict[str, set[int]] = {}
        name_to_spec: dict[str, _AxisSpec] = {}
        for path in batch_paths:
            for spec in specs:
                name = spec.tensor_name(path.phase)
                row_plan.setdefault(name, set()).add(path.row_index)
                name_to_spec[name] = spec
        canonical_plan = {
            name: tuple(sorted(indices)) for name, indices in sorted(row_plan.items())
        }
        reference = references[batch_index]
        record = reference.record
        inventory = record.get("tensors")
        if not isinstance(inventory, Mapping):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Cache batch {batch_index} has no authenticated tensor inventory."
            )
        for name, indices in canonical_plan.items():
            descriptor = inventory.get(name)
            shape = descriptor.get("shape") if isinstance(descriptor, Mapping) else None
            width = name_to_spec[name].source_width
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or shape[0] != 1
                or shape[2] != width
                or any(index < 0 or index >= shape[1] for index in indices)
            ):
                raise Lfm2T5SparseFeatureFoldBuilderError(
                    f"Cache tensor {name!r} has foreign row geometry."
                )
        payload = load_lfm2_circuit_trace_selected_rows(
            evidence.session,
            batch_index,
            canonical_plan,
        )
        primary_error: BaseException | None = None
        try:
            recomputed = lfm2_circuit_trace_selected_rows_payload_sha256(
                session_sha256=evidence.session.evidence_sha256,
                cache_key=evidence.cache.cache_key,
                batch_index=batch_index,
                reference=reference,
                payload=payload,
            )
            if (
                recomputed != payload.selection_sha256
                or payload.session_sha256 != evidence.session.evidence_sha256
                or payload.batch_index != batch_index
                or payload.whole_tensor_cpu_clone_calls != 0
                or payload.unselected_rows_loaded is not False
            ):
                raise Lfm2T5SparseFeatureFoldBuilderError(
                    "Authenticated selected-row payload failed its live seal."
                )
            tensors = payload.tensors
            for path in batch_paths:
                for spec in specs:
                    name = spec.tensor_name(path.phase)
                    positions = canonical_plan[name]
                    try:
                        position = positions.index(path.row_index)
                    except (
                        ValueError
                    ) as exc:  # pragma: no cover - plan construction proves
                        raise AssertionError(
                            "selected path missing from row plan"
                        ) from exc
                    tensor = tensors[name]
                    if (
                        tensor.ndim != 2
                        or tuple(tensor.shape) != (len(positions), spec.source_width)
                        or tensor.device.type != "cpu"
                        or tensor.layout != torch.strided
                        or tensor.is_complex()
                        or not tensor.is_floating_point()
                        or not bool(torch.isfinite(tensor).all().item())
                    ):
                        raise Lfm2T5SparseFeatureFoldBuilderError(
                            f"Selected cache tensor {name!r} changed after authentication."
                        )
                    rows_by_axis[spec.key][path.global_index] = (
                        tensor[position].detach().contiguous().clone()
                    )
            payload_records.append(
                MappingProxyType(
                    {
                        "batch_index": batch_index,
                        "input_sha256": reference.input_sha256,
                        "record_sha256": reference.record_sha256,
                        "selected_row_payload_sha256": payload.selection_sha256,
                        "ordered_tensor_count": len(payload.ordered_tensor_names),
                        "selected_row_count": payload.selected_row_count,
                        "source_row_count": payload.source_row_count,
                        "source_logical_bytes": payload.source_logical_bytes,
                        "selected_logical_bytes": payload.logical_bytes,
                        "maximum_materialized_source_rows": (
                            payload.maximum_materialized_source_rows
                        ),
                        "whole_tensor_cpu_clone_calls": 0,
                        "unselected_rows_loaded": False,
                    }
                )
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                _release_selected_rows_payload(evidence.session, payload)
            except BaseException as cleanup_exc:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "Selected-row authenticated release failed after the primary "
                    f"error; owned tensors were still cleared: {cleanup_exc}"
                )
        if not payload.released:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Selected-row payload was not deterministically released."
            )

    fit_paths = tuple(evidence.paths[index] for index in evidence.fit_indices)
    heldout_paths = tuple(
        evidence.paths[index] for index in evidence.validation_indices
    )
    payload_selection_sha256s = tuple(
        str(value["selected_row_payload_sha256"]) for value in payload_records
    )
    result: dict[tuple[int, str], _SelectedAxisRows] = {}
    for spec in specs:
        source_rows = rows_by_axis[spec.key]
        if set(source_rows) != selected_set:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                f"Axis {spec.key!r} did not receive every planned fit/heldout row."
            )
        fit_source = torch.stack(
            [source_rows[index] for index in evidence.fit_indices], dim=0
        ).contiguous()
        heldout_source = torch.stack(
            [source_rows[index] for index in evidence.validation_indices], dim=0
        ).contiguous()
        fit_float64 = (
            fit_source.to(device="cpu", dtype=torch.float64).contiguous().clone()
        )
        heldout_float64 = (
            heldout_source.to(device="cpu", dtype=torch.float64).contiguous().clone()
        )
        fit_source_seal = _owned_matrix_seal(fit_source, f"{spec.key}.fit_source")
        heldout_source_seal = _owned_matrix_seal(
            heldout_source, f"{spec.key}.heldout_source"
        )
        fit_float64_seal = _owned_matrix_seal(fit_float64, f"{spec.key}.fit_float64")
        heldout_float64_seal = _owned_matrix_seal(
            heldout_float64, f"{spec.key}.heldout_float64"
        )
        body = {
            "schema": LFM2_T5_EVIDENCE_SELECTION_SCHEMA,
            "cache_key": evidence.cache.cache_key,
            "cache_parent_identity_sha256": parent_identity_sha256,
            "corpus_fingerprint": evidence.cache.corpus_fingerprint,
            "manifest_sha256": evidence.session_evidence["manifest_sha256"],
            "completion_sha256": evidence.session_evidence["completion_sha256"],
            "selective_session_sha256": evidence.session.evidence_sha256,
            "source_graph_sha256": source_graph_sha256,
            "parent_state_sha256": parent_state_sha256,
            "target_blank_state_sha256": target_blank_state_sha256,
            "target_topology_sha256": target_topology_sha256,
            "source_layer_topology_sha256": (evidence.source_layer_topology_sha256),
            "source_layer_topology_document_sha256": (
                evidence.source_layer_topology_document_sha256
            ),
            "target_live_layer_topology_sha256": (target_live_layer_topology_sha256),
            "target_live_layer_topology_document_sha256": (
                target_live_layer_topology_document_sha256
            ),
            "source_runtime_semantics_sha256": source_runtime_semantics_sha256,
            "target_runtime_semantics_sha256": target_runtime_semantics_sha256,
            "canonical_runtime_receipt_sha256": (
                evidence.canonical_runtime_receipt_sha256
            ),
            "path_plan_sha256": evidence.path_plan_sha256,
            "partition_sha256": evidence.partition_sha256,
            "split_resolution_sha256": evidence.split_resolution_sha256,
            "layer_index": spec.layer_index,
            "axis_kind": spec.axis_kind,
            "component": spec.component,
            "evidence_endpoint": spec.endpoint,
            "source_parameter_path": spec.parameter_paths[spec.writer_role],
            "source_width": spec.source_width,
            "ordered_fit_indices": list(evidence.fit_indices),
            "ordered_fit_paths": list(fit_paths),
            "ordered_heldout_indices": list(evidence.validation_indices),
            "ordered_heldout_paths": list(heldout_paths),
            "selection_indices_sha256": canonical_sha256(
                list(evidence.selection_indices)
            ),
            "fit_indices_sha256": canonical_sha256(list(evidence.fit_indices)),
            "heldout_indices_sha256": canonical_sha256(
                list(evidence.validation_indices)
            ),
            "selected_row_payload_sha256s": list(payload_selection_sha256s),
            "fit_source_dtype_rows_sha256": fit_source_seal.tensor_sha256,
            "heldout_source_dtype_rows_sha256": (heldout_source_seal.tensor_sha256),
            "fit_float64_rows_sha256": fit_float64_seal.tensor_sha256,
            "heldout_float64_rows_sha256": (heldout_float64_seal.tensor_sha256),
            "fit_source_dtype_t5_rows_sha256": (fit_source_seal.t5_tensor_sha256),
            "heldout_source_dtype_t5_rows_sha256": (
                heldout_source_seal.t5_tensor_sha256
            ),
            "fit_float64_t5_rows_sha256": (fit_float64_seal.t5_tensor_sha256),
            "heldout_float64_t5_rows_sha256": (heldout_float64_seal.t5_tensor_sha256),
            "fit_row_count": int(fit_source.shape[0]),
            "heldout_row_count": int(heldout_source.shape[0]),
            "selection_rows_read": 0,
            "fit_heldout_disjoint": True,
            "row_selection_precedes_whole_tensor_clone": True,
            "whole_tensor_cpu_clone_calls": 0,
            "unselected_rows_loaded": False,
        }
        selected_axis_rows = _SelectedAxisRows(
            fit_source=fit_source,
            heldout_source=heldout_source,
            fit_float64=fit_float64,
            heldout_float64=heldout_float64,
            fit_source_seal=fit_source_seal,
            heldout_source_seal=heldout_source_seal,
            fit_float64_seal=fit_float64_seal,
            heldout_float64_seal=heldout_float64_seal,
            selection_sha256=canonical_sha256(body),
            selection_body=MappingProxyType(body),
        )
        _verify_selected_axis_rows(selected_axis_rows, repr(spec.key))
        result[spec.key] = selected_axis_rows
    return MappingProxyType(result), tuple(payload_records)


def _source_artifacts(
    *,
    witness: Any,
    evidence: _EvidenceContext,
    parent_identity_sha256: str,
    solve_config_sha256: str,
    target_blank_state_sha256: str,
    target_topology_sha256: str,
    target_live_layer_topology_sha256: str,
    target_live_layer_topology_document_sha256: str,
    source_runtime_semantics_sha256: str,
    target_runtime_semantics_sha256: str,
) -> dict[str, str]:
    return {
        "parent_state": witness.parent_state_fingerprint,
        "component_evidence_cache": witness.evidence_cache_sha256,
        "component_evidence_selection": witness.evidence_selection_sha256,
        "body_route_record": witness.body_route_record_sha256,
        "body_route_source_witness": witness.witness_sha256,
        "parent_lfm2_identity": parent_identity_sha256,
        "parent_trace_corpus": evidence.cache.corpus_fingerprint,
        "parent_trace_manifest": evidence.session_evidence["manifest_sha256"],
        "parent_trace_completion": evidence.session_evidence["completion_sha256"],
        "parent_trace_selective_session": evidence.session.evidence_sha256,
        "parent_trace_path_plan": evidence.path_plan_sha256,
        "parent_trace_layer_topology": (evidence.source_layer_topology_sha256),
        "parent_trace_layer_topology_document": (
            evidence.source_layer_topology_document_sha256
        ),
        "component_evidence_partition": evidence.partition_sha256,
        "component_evidence_split_resolution": (evidence.split_resolution_sha256),
        "t5_builder_source": _BUILDER_SOURCE_SHA256,
        "t5_solve_config": solve_config_sha256,
        "target_blank_state": target_blank_state_sha256,
        "target_topology": target_topology_sha256,
        "target_live_layer_topology": target_live_layer_topology_sha256,
        "target_live_layer_topology_document": (
            target_live_layer_topology_document_sha256
        ),
        "source_runtime_semantics": source_runtime_semantics_sha256,
        "target_runtime_semantics": target_runtime_semantics_sha256,
        "canonical_runtime_receipt": evidence.canonical_runtime_receipt_sha256,
    }


def build_authenticated_lfm2_t5_sparse_feature_fold_bundle(
    *,
    parent_model: Any,
    source_graph: ModelCircuitGraph,
    selected: Lfm2DenseBudgetedSelectedTarget,
    target: DynamicTargetSynthesisResult,
    evidence_authority: Lfm2T5SparseFeatureEvidenceAuthority,
    solve_config: Lfm2T5SparseFeatureFoldSolveConfig | None = None,
) -> Lfm2SparseFeatureFoldBundle:
    """Build all 22 target-bound T5 axes from exact sealed fit/heldout rows."""

    import torch

    if not isinstance(source_graph, ModelCircuitGraph):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "source_graph must be a ModelCircuitGraph."
        )
    if not isinstance(selected, Lfm2DenseBudgetedSelectedTarget):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "selected must be Lfm2DenseBudgetedSelectedTarget."
        )
    if not isinstance(target, DynamicTargetSynthesisResult):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "target must be DynamicTargetSynthesisResult."
        )
    if target.adapter_payload is not selected:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Target adapter payload is not the selected target instance."
        )
    if (
        target.adapter_id != Lfm2TargetSynthesisAdapter.adapter_id
        or target.adapter_version != Lfm2TargetSynthesisAdapter.version
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Target was not synthesized by the authenticated LFM2 adapter."
        )
    if source_graph.fingerprint != selected.source_graph_fingerprint:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Source graph differs from the selected parent graph."
        )
    if dict(target.parent_graph_fingerprints) != {
        selected.source_parent: selected.source_graph_fingerprint
    }:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Target parent graph binding differs from the selected source."
        )
    try:
        validate_lfm2_dense_budgeted_child(selected, target.model, target.graph)
        verify_compiler_authenticated_target(target)
    except Exception as exc:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Target failed live dense-child/compiler authentication."
        ) from exc
    config = (
        Lfm2T5SparseFeatureFoldSolveConfig() if solve_config is None else solve_config
    )
    if type(config) is not Lfm2T5SparseFeatureFoldSolveConfig:
        raise Lfm2T5SparseFeatureFoldBuilderError("solve_config has the wrong type.")

    parent_state_before = fingerprint_model_state(parent_model)
    target_state_before = fingerprint_model_state(target.model)
    parent_identity_before = fingerprint_lfm2_parent(parent_model).identity_sha256
    if parent_state_before != selected.source_state_fingerprint:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Live parent state differs from selected source state."
        )
    if target_state_before != target.blank_state_fingerprint:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Live target is not the authenticated compiler blank."
        )
    evidence = _evidence_context(evidence_authority)
    if parent_identity_before != evidence.cache.parent_fingerprint:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Live parent identity differs from the sealed trace cache."
        )
    source_live_topology_before = _authenticate_live_lfm2_topology(
        model=parent_model,
        graph=source_graph,
        label="source parent",
        expected_layer_topology_sha256=(evidence.source_layer_topology_sha256),
        expected_layer_topology_document_sha256=(
            evidence.source_layer_topology_document_sha256
        ),
    )
    target_live_topology_before = _authenticate_live_lfm2_topology(
        model=target.model,
        graph=target.graph,
        label="compiler target",
    )
    source_runtime_semantics_before = _internal_runtime_semantics_sha256(
        parent_model,
        "source parent",
    )
    target_runtime_semantics_before = _internal_runtime_semantics_sha256(
        target.model,
        "compiler target",
    )
    specs = _axis_specs(
        parent_model=parent_model,
        target_model=target.model,
        selected=selected,
    )
    selected_rows, payload_records = _selected_axis_rows(
        evidence=evidence,
        specs=specs,
        parent_identity_sha256=parent_identity_before,
        parent_state_sha256=parent_state_before,
        source_graph_sha256=source_graph.fingerprint,
        target_blank_state_sha256=target.blank_state_fingerprint,
        target_topology_sha256=target.topology_fingerprint,
        target_live_layer_topology_sha256=(target_live_topology_before[0]),
        target_live_layer_topology_document_sha256=(target_live_topology_before[1]),
        source_runtime_semantics_sha256=source_runtime_semantics_before,
        target_runtime_semantics_sha256=target_runtime_semantics_before,
    )
    writer_snapshots = _snapshot_axis_writers(
        parent_model=parent_model,
        specs=specs,
        parent_state_sha256=parent_state_before,
    )

    axes: list[Lfm2SparseFeatureAxisAdmission] = []
    selection_manifest_rows = []
    for spec in specs:
        rows = selected_rows[spec.key]
        _verify_selected_axis_rows(rows, repr(spec.key))
        writer_snapshot = writer_snapshots[spec.key]
        normalized_writer = writer_snapshot.tensor.clone()
        normalized_writer_sha256 = writer_snapshot.tensor_sha256
        normalized_writer_version = _writer_tensor_version(normalized_writer)
        solver_fit_rows = rows.fit_float64.clone()
        solver_heldout_rows = rows.heldout_float64.clone()
        solver_fit_seal = _owned_matrix_seal(
            solver_fit_rows, f"{spec.key}.solver_fit_rows"
        )
        solver_heldout_seal = _owned_matrix_seal(
            solver_heldout_rows, f"{spec.key}.solver_heldout_rows"
        )
        fold = fit_body_component_routes(
            parent_weight=normalized_writer,
            component_rows=solver_fit_rows,
            component=spec.component,
            level="X4",
            site=f"L{spec.layer_index:02d}",
            kept_width=spec.target_width,
            block_count=1,
            active_k=config.active_k,
            channel_selection="evidence-energy",
            kept_channel_ids=None,
            write_metric=None,
            install_coefficients="write",
            refit_ridge_relative=config.refit_ridge_relative,
            write_ridge_relative=config.write_ridge_relative,
            write_solve_sweeps=config.write_solve_sweeps,
            pivot_tolerance=config.pivot_tolerance,
            residual_floor=config.residual_floor,
            channels_per_chunk=config.channels_per_chunk,
            heldout_component_rows=solver_heldout_rows,
            solve_dtype=torch.float64,
            solve_device="cpu",
        )
        if (
            _writer_tensor_version(normalized_writer) != normalized_writer_version
            or _tensor_sha256(normalized_writer, writer_snapshot.parameter_path)
            != normalized_writer_sha256
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "The T5 solver mutated its owned parent-writer snapshot."
            )
        _verify_owned_matrix(
            solver_fit_rows,
            solver_fit_seal,
            f"{spec.key}.solver_fit_rows",
        )
        _verify_owned_matrix(
            solver_heldout_rows,
            solver_heldout_seal,
            f"{spec.key}.solver_heldout_rows",
        )
        _verify_selected_axis_rows(rows, repr(spec.key))
        if (
            fold.record.get("component_rows_sha256")
            != rows.fit_float64_seal.t5_tensor_sha256
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "The T5 route row digests differ from selected-row authority."
            )
        witness_fit_rows = rows.fit_float64.clone()
        witness_heldout_rows = rows.heldout_float64.clone()
        witness_fit_seal = _owned_matrix_seal(
            witness_fit_rows, f"{spec.key}.witness_fit_rows"
        )
        witness_heldout_seal = _owned_matrix_seal(
            witness_heldout_rows, f"{spec.key}.witness_heldout_rows"
        )
        witness = build_t5_body_route_source_witness(
            fold,
            parent_weight=normalized_writer,
            component_rows=witness_fit_rows,
            heldout_component_rows=witness_heldout_rows,
            parameter_path=spec.parameter_paths[spec.writer_role],
            graph_fingerprint=source_graph.fingerprint,
            parent_state_fingerprint=parent_state_before,
            component_id=fold.component_id,
            layer_index=spec.layer_index,
            site=f"L{spec.layer_index:02d}",
            role="writer",
            evidence_cache_sha256=evidence.cache.cache_key,
            evidence_selection_sha256=rows.selection_sha256,
            evidence_endpoint=spec.endpoint,
            coefficient_set="write",
        )
        _verify_owned_matrix(
            witness_fit_rows,
            witness_fit_seal,
            f"{spec.key}.witness_fit_rows",
        )
        _verify_owned_matrix(
            witness_heldout_rows,
            witness_heldout_seal,
            f"{spec.key}.witness_heldout_rows",
        )
        _verify_selected_axis_rows(rows, repr(spec.key))
        if witness.record.get("component_rows_sha256") != rows.selection_body.get(
            "fit_float64_t5_rows_sha256"
        ) or witness.record.get(
            "heldout_component_rows_sha256"
        ) != rows.selection_body.get("heldout_float64_t5_rows_sha256"):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "The T5 witness row digests differ from selected-row authority."
            )
        admission_kwargs: dict[str, Any] = {
            "source_graph": source_graph,
            "target_graph": target.graph,
            "source_graph_sha256": source_graph.fingerprint,
            "target_graph_sha256": target.graph.fingerprint,
            "layer_index": spec.layer_index,
            "site": spec.graph_site,
            "axis_kind": spec.axis_kind,
            "source_parameter_paths": spec.parameter_paths,
            "target_parameter_paths": spec.parameter_paths,
            "body_route_fold": fold,
            "source_witness": witness,
            "coefficient_set": "write",
            "source_artifact_sha256s": _source_artifacts(
                witness=witness,
                evidence=evidence,
                parent_identity_sha256=parent_identity_before,
                solve_config_sha256=config.config_sha256,
                target_blank_state_sha256=target.blank_state_fingerprint,
                target_topology_sha256=target.topology_fingerprint,
                target_live_layer_topology_sha256=(target_live_topology_before[0]),
                target_live_layer_topology_document_sha256=(
                    target_live_topology_before[1]
                ),
                source_runtime_semantics_sha256=(source_runtime_semantics_before),
                target_runtime_semantics_sha256=(target_runtime_semantics_before),
            ),
        }
        if spec.axis_kind == AXIS_SHORT_CONV_CONTEXT:
            admission_kwargs.update(
                source_parameter_row_slices=spec.source_row_slices,
                target_parameter_row_slices=spec.target_row_slices,
            )
        axis = admit_lfm2_sparse_feature_axis(**admission_kwargs)
        if (
            _writer_tensor_version(normalized_writer) != normalized_writer_version
            or _tensor_sha256(normalized_writer, writer_snapshot.parameter_path)
            != normalized_writer_sha256
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Witness/admission mutated its owned parent-writer snapshot."
            )
        _verify_owned_matrix(
            witness_fit_rows,
            witness_fit_seal,
            f"{spec.key}.witness_fit_rows",
        )
        _verify_owned_matrix(
            witness_heldout_rows,
            witness_heldout_seal,
            f"{spec.key}.witness_heldout_rows",
        )
        _verify_selected_axis_rows(rows, repr(spec.key))
        axes.append(axis)
        selection_manifest_rows.append(
            {
                "layer_index": spec.layer_index,
                "axis_kind": spec.axis_kind,
                "source_width": spec.source_width,
                "target_width": spec.target_width,
                "selection_sha256": rows.selection_sha256,
                "body_route_record_sha256": fold.record["record_sha256"],
                "source_witness_sha256": witness.witness_sha256,
                "axis_admission_sha256": axis.admission_sha256,
            }
        )

    selection_manifest_sha256 = canonical_sha256(
        {
            "schema": LFM2_T5_EVIDENCE_SELECTION_MANIFEST_SCHEMA,
            "cache_key": evidence.cache.cache_key,
            "source_graph_sha256": source_graph.fingerprint,
            "target_graph_sha256": target.graph.fingerprint,
            "target_blank_state_sha256": target.blank_state_fingerprint,
            "target_topology_sha256": target.topology_fingerprint,
            "source_layer_topology_sha256": (source_live_topology_before[0]),
            "source_layer_topology_document_sha256": (source_live_topology_before[1]),
            "target_live_layer_topology_sha256": (target_live_topology_before[0]),
            "target_live_layer_topology_document_sha256": (
                target_live_topology_before[1]
            ),
            "source_runtime_semantics_sha256": (source_runtime_semantics_before),
            "target_runtime_semantics_sha256": (target_runtime_semantics_before),
            "canonical_runtime_receipt_sha256": (
                evidence.canonical_runtime_receipt_sha256
            ),
            "parent_state_sha256": parent_state_before,
            "partition_sha256": evidence.partition_sha256,
            "split_resolution_sha256": evidence.split_resolution_sha256,
            "ordered_axes": selection_manifest_rows,
            "ordered_selected_row_payloads": [dict(value) for value in payload_records],
            "axis_count": len(selection_manifest_rows),
            "selection_rows_read": 0,
        }
    )
    bundle = build_lfm2_sparse_feature_fold_bundle(
        source_graph=source_graph,
        target_graph=target.graph,
        source_graph_sha256=source_graph.fingerprint,
        target_graph_sha256=target.graph.fingerprint,
        axes=tuple(axes),
        source_artifact_sha256s={
            "parent_state": parent_state_before,
            "parent_lfm2_identity": parent_identity_before,
            "component_evidence_cache": evidence.cache.cache_key,
            "parent_trace_corpus": evidence.cache.corpus_fingerprint,
            "parent_trace_manifest": evidence.session_evidence["manifest_sha256"],
            "parent_trace_completion": evidence.session_evidence["completion_sha256"],
            "parent_trace_selective_session": evidence.session.evidence_sha256,
            "parent_trace_path_plan": evidence.path_plan_sha256,
            "parent_trace_layer_topology": (evidence.source_layer_topology_sha256),
            "parent_trace_layer_topology_document": (
                evidence.source_layer_topology_document_sha256
            ),
            "component_evidence_partition": evidence.partition_sha256,
            "component_evidence_split_resolution": (evidence.split_resolution_sha256),
            "component_evidence_selection_manifest": selection_manifest_sha256,
            "t5_builder_source": _BUILDER_SOURCE_SHA256,
            "t5_solve_config": config.config_sha256,
            "target_blank_state": target.blank_state_fingerprint,
            "target_topology": target.topology_fingerprint,
            "target_live_layer_topology": target_live_topology_before[0],
            "target_live_layer_topology_document": (target_live_topology_before[1]),
            "source_runtime_semantics": source_runtime_semantics_before,
            "target_runtime_semantics": target_runtime_semantics_before,
            "canonical_runtime_receipt": (evidence.canonical_runtime_receipt_sha256),
        },
    ).verify()
    if (
        len(bundle.axes) != 22
        or tuple((axis.layer_index, axis.axis_kind) for axis in bundle.axes)
        != tuple(spec.key for spec in specs)
        or any(axis.coefficient_set != "write" for axis in bundle.axes)
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Constructed bundle is not the exact ordered 22-axis write bundle."
        )

    _verify_selective_session(evidence.session)
    evidence_authority.verify()
    for spec in specs:
        _verify_selected_axis_rows(selected_rows[spec.key], repr(spec.key))
    _verify_writer_snapshot_sources(
        parent_model=parent_model,
        specs=specs,
        snapshots=writer_snapshots,
        parent_state_sha256=parent_state_before,
    )
    try:
        validate_lfm2_dense_budgeted_child(selected, target.model, target.graph)
        verify_compiler_authenticated_target(target)
    except Exception as exc:
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Target authentication changed during T5 construction."
        ) from exc
    source_live_topology_after = _authenticate_live_lfm2_topology(
        model=parent_model,
        graph=source_graph,
        label="source parent",
        expected_layer_topology_sha256=(evidence.source_layer_topology_sha256),
        expected_layer_topology_document_sha256=(
            evidence.source_layer_topology_document_sha256
        ),
    )
    target_live_topology_after = _authenticate_live_lfm2_topology(
        model=target.model,
        graph=target.graph,
        label="compiler target",
    )
    source_runtime_semantics_after = _internal_runtime_semantics_sha256(
        parent_model,
        "source parent",
    )
    target_runtime_semantics_after = _internal_runtime_semantics_sha256(
        target.model,
        "compiler target",
    )
    if (
        fingerprint_model_state(parent_model) != parent_state_before
        or fingerprint_lfm2_parent(parent_model).identity_sha256
        != parent_identity_before
        or fingerprint_model_state(target.model) != target_state_before
        or evidence.session.cache_key != evidence.cache.cache_key
        or evidence.session.evidence_sha256
        != evidence.session_evidence["session_sha256"]
        or source_graph.fingerprint != selected.source_graph_fingerprint
        or target.graph.fingerprint != bundle.target_graph_sha256
        or dict(target.parent_graph_fingerprints)
        != {selected.source_parent: selected.source_graph_fingerprint}
        or evidence_authority.paths != evidence.paths
        or evidence_authority.path_splits != evidence.path_splits
        or source_live_topology_after != source_live_topology_before
        or target_live_topology_after != target_live_topology_before
        or source_runtime_semantics_after != source_runtime_semantics_before
        or target_runtime_semantics_after != target_runtime_semantics_before
    ):
        raise Lfm2T5SparseFeatureFoldBuilderError(
            "Parent, target, cache, session, or row-plan authority changed during solve."
        )
    return bundle


@dataclass(frozen=True, slots=True)
class Lfm2AuthenticatedT5SparseFeatureFoldBuilder:
    """Callable resource captured by production wiring before target synthesis."""

    parent_model: Any
    source_graph: ModelCircuitGraph
    evidence_authority: Lfm2T5SparseFeatureEvidenceAuthority
    solve_config: Lfm2T5SparseFeatureFoldSolveConfig = (
        Lfm2T5SparseFeatureFoldSolveConfig()
    )
    _sealed_builder_binding_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        body = self._validated_binding_body()
        object.__setattr__(
            self,
            "_sealed_builder_binding_sha256",
            canonical_sha256(body),
        )
        self.verify()

    def _validated_binding_body(self) -> dict[str, Any]:
        if not isinstance(self.source_graph, ModelCircuitGraph):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "source_graph must be a ModelCircuitGraph."
            )
        if type(self.evidence_authority) is not (Lfm2T5SparseFeatureEvidenceAuthority):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "evidence_authority has the wrong type."
            )
        self.evidence_authority.verify()
        if type(self.solve_config) is not Lfm2T5SparseFeatureFoldSolveConfig:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "solve_config has the wrong type."
            )
        parent_identity_sha256 = fingerprint_lfm2_parent(
            self.parent_model
        ).identity_sha256
        if parent_identity_sha256 != self.evidence_authority.cache.parent_fingerprint:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Builder parent differs from its sealed evidence authority."
            )
        live_layer_topology = _authenticate_live_lfm2_topology(
            model=self.parent_model,
            graph=self.source_graph,
            label="builder source parent",
            expected_layer_topology_sha256=(
                self.evidence_authority.source_layer_topology_sha256
            ),
            expected_layer_topology_document_sha256=(
                self.evidence_authority.source_layer_topology_document_sha256
            ),
        )
        live_runtime_semantics_sha256 = _internal_runtime_semantics_sha256(
            self.parent_model,
            "builder source parent",
        )
        return {
            "schema": LFM2_T5_SPARSE_FEATURE_FOLD_BUILDER_SCHEMA,
            "builder_source_sha256": _BUILDER_SOURCE_SHA256,
            "solve_config_sha256": self.solve_config.config_sha256,
            "source_graph_sha256": self.source_graph.fingerprint,
            "live_parent_state_sha256": fingerprint_model_state(self.parent_model),
            "live_parent_identity_sha256": parent_identity_sha256,
            "live_source_layer_topology_sha256": live_layer_topology[0],
            "live_source_layer_topology_document_sha256": (live_layer_topology[1]),
            "live_source_runtime_semantics_sha256": (live_runtime_semantics_sha256),
            "canonical_runtime_receipt_sha256": (
                self.evidence_authority.canonical_runtime_receipt_sha256
            ),
            "cache_key": self.evidence_authority.cache.cache_key,
            "corpus_fingerprint": (self.evidence_authority.cache.corpus_fingerprint),
            "selective_session_sha256": (
                self.evidence_authority.session.evidence_sha256
            ),
            "path_plan_sha256": self.evidence_authority.path_plan_sha256,
            "cache_source_layer_topology_sha256": (
                self.evidence_authority.source_layer_topology_sha256
            ),
            "cache_source_layer_topology_document_sha256": (
                self.evidence_authority.source_layer_topology_document_sha256
            ),
            "partition_sha256": self.evidence_authority.partition_sha256,
            "split_resolution_sha256": (
                self.evidence_authority.split_resolution.binding_sha256
            ),
            "evidence_authority_sha256": (self.evidence_authority.authority_sha256),
        }

    @property
    def builder_binding_sha256(self) -> str:
        self.verify()
        return self._sealed_builder_binding_sha256

    def verify(self) -> "Lfm2AuthenticatedT5SparseFeatureFoldBuilder":
        if type(self) is not Lfm2AuthenticatedT5SparseFeatureFoldBuilder:
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Builder must have the exact factory-owned type."
            )
        body = self._validated_binding_body()
        if canonical_sha256(body) != _digest(
            self._sealed_builder_binding_sha256,
            "builder_binding_sha256",
        ):
            raise Lfm2T5SparseFeatureFoldBuilderError(
                "Builder source, config, parent, graph, or evidence binding changed."
            )
        return self

    def __call__(
        self,
        selected: Lfm2DenseBudgetedSelectedTarget,
        target: DynamicTargetSynthesisResult,
    ) -> Lfm2SparseFeatureFoldBundle:
        return build_authenticated_lfm2_t5_sparse_feature_fold_bundle(
            parent_model=self.parent_model,
            source_graph=self.source_graph,
            selected=selected,
            target=target,
            evidence_authority=self.evidence_authority,
            solve_config=self.solve_config,
        )


__all__ = [
    "LFM2_T5_EVIDENCE_PARTITION_SCHEMA",
    "LFM2_T5_EVIDENCE_PATH_PLAN_SCHEMA",
    "LFM2_T5_EVIDENCE_SELECTION_MANIFEST_SCHEMA",
    "LFM2_T5_EVIDENCE_SELECTION_SCHEMA",
    "LFM2_T5_EVIDENCE_SPLIT_RESOLUTION_SCHEMA",
    "LFM2_T5_SPARSE_FEATURE_FOLD_BUILDER_SCHEMA",
    "Lfm2AuthenticatedT5SparseFeatureFoldBuilder",
    "Lfm2T5EvidenceSplitResolution",
    "Lfm2T5SparseFeatureEvidenceAuthority",
    "Lfm2T5SparseFeatureFoldBuilderError",
    "Lfm2T5SparseFeatureFoldSolveConfig",
    "build_authenticated_lfm2_t5_sparse_feature_fold_bundle",
    "build_lfm2_t5_evidence_split_resolution",
    "build_lfm2_t5_sparse_feature_evidence_authority",
]
