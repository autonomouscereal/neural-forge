"""Authenticated compile-time sparse feature-map bundles for LFM2.

This module does not solve a route and does not install or retain child runtime
state.  It admits already-compiled :class:`SparseFeatureFoldMap` values only
when their own binding names an exact source graph, target graph, layer, site,
atomic component path set, and authenticated upstream authority.

The component rules are deliberately coupled.  One post-SwiGLU map owns all
``w1``/``w3`` producer rows and the ``w2`` consumer columns.  One ShortConv
context map owns all three input-projection branches, the depthwise FIR rows,
and the output-projection columns.  Attention either keeps all Q/K/V/O
internal feature maps at one sealed identity when topology is unchanged, or
delegates the complete changed geometry to the existing coupled GQA/RoPE
authority.  Independent C06/C07/C08/C09 feature maps are never admissible.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import (
    CircuitKind,
    ModelCircuitGraph,
    ParameterSpec,
    canonical_sha256,
)
from .hf_causal_gqa_attention_fold import (
    SCHEMA as COUPLED_GQA_ROPE_AUTHORITY_SCHEMA,
    HFCausalCoupledAttentionFold,
)
from .hf_feature_fold import SparseFeatureFoldMap
from .matrix_technique_body_routes import (
    T5_BODY_ROUTE_RECORD_SCHEMA,
    BodyRouteFold,
    T5BodyRouteAuthenticationError,
    T5BodyRouteSourceWitness,
    sparse_feature_fold_map_from_body_route,
    verify_sealed_body_route_record,
    verify_t5_body_route_source_witness,
)


LFM2_SPARSE_FEATURE_FOLD_BUNDLE_SCHEMA = (
    "neural-forge-lfm2-sparse-feature-fold-bundle-v1"
)
LFM2_SPARSE_FEATURE_AXIS_ADMISSION_SCHEMA = (
    "neural-forge-lfm2-sparse-feature-axis-admission-v1"
)
LFM2_SPARSE_FEATURE_PROVENANCE_SCHEMA = "neural-forge-lfm2-sparse-feature-provenance-v1"
LFM2_ATTENTION_FEATURE_ADMISSION_SCHEMA = (
    "neural-forge-lfm2-attention-feature-admission-v1"
)
LFM2_IDENTITY_INTERNAL_FEATURE_MAP_SCHEMA = (
    "neural-forge-lfm2-identity-internal-feature-map-v1"
)
LFM2_IDENTITY_INTERNAL_FEATURE_MAP_SHA256 = canonical_sha256(
    {
        "schema": LFM2_IDENTITY_INTERNAL_FEATURE_MAP_SCHEMA,
        "operator": "identity",
        "compile_time_only": True,
        "runtime_parameter_count": 0,
        "runtime_buffer_count": 0,
        "runtime_sidecar_count": 0,
    }
)

AXIS_MLP_POST_SWIGLU = "mlp_post_swiglu"
AXIS_SHORT_CONV_CONTEXT = "short_conv_context"
AXIS_RESIDUAL = "residual"

_AXIS_KINDS = frozenset({AXIS_MLP_POST_SWIGLU, AXIS_SHORT_CONV_CONTEXT})
_MLP_ROLES = ("w1", "w3", "w2")
_SHORT_CONV_ROLES = (
    "in_proj_b",
    "in_proj_c",
    "in_proj_x",
    "fir_kernel",
    "out_proj",
)
_ATTENTION_ROLES = ("q", "k", "v", "o")
_ATTENTION_COMPONENTS = ("C06", "C07", "C08", "C09")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STABLE_ID = re.compile(r"[A-Za-z0-9_.:/\[\]-]+\Z")


class Lfm2SparseFeatureFoldBundleError(ValueError):
    """A sparse feature map or bundle failed its compile-time contract."""


def _plain_json(value: Any, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} must be finite canonical JSON."
        ) from exc


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _stable_id(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or _STABLE_ID.fullmatch(value.strip()) is None
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} must be a nonempty stable graph identifier or parameter path."
        )
    return value.strip()


def _layer_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2SparseFeatureFoldBundleError(
            "layer_index must be a nonnegative integer."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SparseFeatureFoldBundleError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2SparseFeatureFoldBundleError(f"{name} must be positive and finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} must be positive and finite."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise Lfm2SparseFeatureFoldBundleError(f"{name} must be positive and finite.")
    return result


def _path_map(
    value: Mapping[str, str],
    name: str,
    *,
    required_roles: Sequence[str] | None = None,
    paths_must_be_distinct: bool = True,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise Lfm2SparseFeatureFoldBundleError(f"{name} must be a nonempty mapping.")
    result = {
        _stable_id(role, f"{name} role"): _stable_id(path, f"{name}[{role!r}]")
        for role, path in value.items()
    }
    if paths_must_be_distinct and len(set(result.values())) != len(result):
        raise Lfm2SparseFeatureFoldBundleError(f"{name} paths must be distinct.")
    if required_roles is not None and set(result) != set(required_roles):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} must contain exactly {tuple(required_roles)}."
        )
    return dict(sorted(result.items()))


def _graph(
    value: Any,
    name: str,
    *,
    expected_sha256: str | None = None,
) -> ModelCircuitGraph:
    if not isinstance(value, ModelCircuitGraph):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} must be a live ModelCircuitGraph."
        )
    if expected_sha256 is not None and value.fingerprint != _digest(
        expected_sha256, f"{name}_sha256"
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} fingerprint does not match its supplied digest."
        )
    return value


def _parameter_spec_at_site(
    graph: ModelCircuitGraph,
    path: str,
    *,
    layer_index: int,
    site: str,
    name: str,
) -> ParameterSpec:
    matches = [
        (node, parameter)
        for node in graph.nodes
        for parameter in node.parameters
        if parameter.path == path
    ]
    if len(matches) != 1:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} must name exactly one parameter in the supplied graph."
        )
    node, parameter = matches[0]
    if node.layer_index != layer_index or node.path != site:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} does not belong to the admitted layer/site."
        )
    return parameter


def _short_conv_slices(
    value: Mapping[str, Sequence[int]] | None,
    name: str,
    *,
    width: int,
) -> dict[str, list[int]]:
    expected = {
        "in_proj_b": [0, width],
        "in_proj_c": [width, 2 * width],
        "in_proj_x": [2 * width, 3 * width],
    }
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} must contain exact B/C/x packed-row slices."
        )
    result: dict[str, list[int]] = {}
    for role, expected_slice in expected.items():
        raw = value[role]
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or len(raw) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in raw)
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                f"{name}[{role!r}] must be a two-integer row slice."
            )
        resolved = [int(raw[0]), int(raw[1])]
        if resolved != expected_slice:
            raise Lfm2SparseFeatureFoldBundleError(
                f"{name} must partition packed in_proj rows without gaps or overlap."
            )
        result[role] = resolved
    return dict(sorted(result.items()))


def _artifact_digests(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise Lfm2SparseFeatureFoldBundleError(
            "source_artifact_sha256s must be a mapping."
        )
    return dict(
        sorted(
            (
                _stable_id(name, "source artifact name"),
                _digest(digest, f"source_artifact_sha256s[{name!r}]"),
            )
            for name, digest in value.items()
        )
    )


def _verify_sealed_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
    seal_field: str,
    expected_schema: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SparseFeatureFoldBundleError(f"{name} must be a sealed mapping.")
    result = _plain_json(dict(value), name)
    supplied = result.pop(seal_field, None)
    if _digest(supplied, f"{name}.{seal_field}") != canonical_sha256(result):
        raise Lfm2SparseFeatureFoldBundleError(f"{name} seal is invalid.")
    if expected_schema is not None and result.get("schema") != expected_schema:
        raise Lfm2SparseFeatureFoldBundleError(f"{name} schema is unsupported.")
    return {**result, seal_field: supplied}


def _verify_t5_axis_authority(
    record: Mapping[str, Any],
    *,
    axis_kind: str,
    layer_index: int,
    source_width: int,
    target_width: int,
) -> None:
    expected = {
        AXIS_MLP_POST_SWIGLU: ("C13", "mlp_down"),
        AXIS_SHORT_CONV_CONTEXT: ("C05", "conv_out_proj"),
    }.get(axis_kind)
    if expected is None:
        return
    component_id, component = expected
    if (
        record.get("schema") != T5_BODY_ROUTE_RECORD_SCHEMA
        or record.get("technique_id") != "T5"
        or record.get("technique") != "sparse_omp_routes"
        or record.get("component_id") != component_id
        or record.get("component") != component
        or record.get("role") != "writer"
        or record.get("site") != f"L{layer_index:02d}"
        or record.get("block_count") != 1
        or record.get("parent_width") != source_width
        or record.get("kept_width") != target_width
        or record.get("dropped_width") != source_width - target_width
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{axis_kind} requires the exact admitted T5 {component_id} writer route."
        )
    if any(
        _SHA256.fullmatch(str(record.get(field, ""))) is None
        for field in ("support_offsets_sha256", "expansion_route_sha256")
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            "T5 authority predates sealed CSR offsets/expansion and is not admissible."
        )


def _t5_origin_binding(
    record: Mapping[str, Any], coefficient_set: str
) -> dict[str, Any]:
    if coefficient_set not in {"read", "write"}:
        raise Lfm2SparseFeatureFoldBundleError(
            "coefficient_set must be explicitly 'read' or 'write'."
        )
    return {
        "body_route_record_sha256": record["record_sha256"],
        "coefficient_set": coefficient_set,
        "component_id": record["component_id"],
        "site": record["site"],
        "level": record["level"],
        "role": record["role"],
        "installed_coefficients": record["solve"]["installed_coefficients"],
        "source_width": record["parent_width"],
        "target_width": record["kept_width"],
    }


def _required_axis_roles(axis_kind: str) -> Sequence[str] | None:
    if axis_kind == AXIS_MLP_POST_SWIGLU:
        return _MLP_ROLES
    if axis_kind == AXIS_SHORT_CONV_CONTEXT:
        return _SHORT_CONV_ROLES
    if axis_kind == AXIS_RESIDUAL:
        return _ATTENTION_ROLES
    return None


def _axis_paths_and_slices(
    *,
    axis_kind: str,
    paths: Mapping[str, str],
    row_slices: Mapping[str, Sequence[int]] | None,
    width: int,
    name: str,
) -> tuple[dict[str, str], dict[str, list[int]]]:
    required = _required_axis_roles(axis_kind)
    resolved_paths = _path_map(
        paths,
        f"{name}_parameter_paths",
        required_roles=required,
        paths_must_be_distinct=axis_kind != AXIS_SHORT_CONV_CONTEXT,
    )
    if axis_kind != AXIS_SHORT_CONV_CONTEXT:
        if row_slices not in (None, {}):
            raise Lfm2SparseFeatureFoldBundleError(
                f"{name}_parameter_row_slices is only valid for ShortConv."
            )
        return resolved_paths, {}
    branch_paths = {resolved_paths[role] for role in _SHORT_CONV_ROLES[:3]}
    if len(branch_paths) != 1:
        raise Lfm2SparseFeatureFoldBundleError(
            "ShortConv B/C/x roles must share one packed in_proj.weight path."
        )
    packed_path = next(iter(branch_paths))
    if (
        resolved_paths["fir_kernel"] == packed_path
        or resolved_paths["out_proj"] == packed_path
        or resolved_paths["fir_kernel"] == resolved_paths["out_proj"]
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            "ShortConv packed input, FIR, and output paths must be distinct."
        )
    return resolved_paths, _short_conv_slices(
        row_slices,
        f"{name}_parameter_row_slices",
        width=width,
    )


def _validate_axis_graph_geometry(
    *,
    graph: ModelCircuitGraph,
    layer_index: int,
    site: str,
    axis_kind: str,
    paths: Mapping[str, str],
    row_slices: Mapping[str, Sequence[int]],
    width: int,
    name: str,
) -> None:
    nodes = [
        node
        for node in graph.nodes
        if node.layer_index == layer_index and node.path == site
    ]
    if len(nodes) != 1:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} graph must contain exactly one node at the admitted site."
        )
    node = nodes[0]
    expected_node_kinds = (
        {CircuitKind.GATED_MLP.value, CircuitKind.DENSE_MLP.value}
        if axis_kind == AXIS_MLP_POST_SWIGLU
        else {CircuitKind.SHORT_CONV.value, CircuitKind.RECURRENT_MIXER.value}
    )
    if node.kind not in expected_node_kinds:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{axis_kind} {name} graph node kind is invalid."
        )
    specs = {
        role: _parameter_spec_at_site(
            graph,
            path,
            layer_index=layer_index,
            site=site,
            name=f"{name}_parameter_paths[{role!r}]",
        )
        for role, path in paths.items()
    }
    shapes = {role: tuple(spec.shape) for role, spec in specs.items()}
    expected_parameter_semantics = (
        {
            "w1": (("mlp_gate", "mlp.w1_weight"), ".w1.weight"),
            "w3": (("mlp_up", "mlp.w3_weight"), ".w3.weight"),
            "w2": (("mlp_down", "mlp.w2_weight"), ".w2.weight"),
        }
        if axis_kind == AXIS_MLP_POST_SWIGLU
        else {
            "in_proj_b": (
                ("conv_in_proj", "operator.in_proj_weight"),
                ".in_proj.weight",
            ),
            "in_proj_c": (
                ("conv_in_proj", "operator.in_proj_weight"),
                ".in_proj.weight",
            ),
            "in_proj_x": (
                ("conv_in_proj", "operator.in_proj_weight"),
                ".in_proj.weight",
            ),
            "fir_kernel": (
                ("conv_kernel", "operator.conv_weight"),
                ".conv.weight",
            ),
            "out_proj": (
                ("conv_out_proj", "operator.out_proj_weight"),
                ".out_proj.weight",
            ),
        }
    )
    if any(
        spec.role not in allowed_roles or spec.path != f"{site}{path_suffix}"
        for role, spec in specs.items()
        for allowed_roles, path_suffix in (expected_parameter_semantics[role],)
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{axis_kind} {name} graph parameter roles are not the canonical "
            "atomic component path set."
        )
    if axis_kind == AXIS_MLP_POST_SWIGLU:
        valid = (
            len(shapes["w1"]) == 2
            and len(shapes["w3"]) == 2
            and len(shapes["w2"]) == 2
            and shapes["w1"][0] == width
            and shapes["w3"][0] == width
            and shapes["w2"][1] == width
            and shapes["w1"][1] == shapes["w3"][1] == shapes["w2"][0]
        )
    elif axis_kind == AXIS_SHORT_CONV_CONTEXT:
        in_shape = shapes["in_proj_b"]
        valid = (
            shapes["in_proj_b"] == shapes["in_proj_c"] == shapes["in_proj_x"]
            and len(in_shape) == 2
            and in_shape[0] == 3 * width
            and len(shapes["fir_kernel"]) == 3
            and shapes["fir_kernel"][0] == width
            and shapes["fir_kernel"][1] == 1
            and shapes["fir_kernel"][2] == 3
            and len(shapes["out_proj"]) == 2
            and shapes["out_proj"][1] == width
            and in_shape[1] == shapes["out_proj"][0]
            and dict(row_slices)
            == {
                "in_proj_b": [0, width],
                "in_proj_c": [width, 2 * width],
                "in_proj_x": [2 * width, 3 * width],
            }
        )
    else:
        valid = (
            all(len(shapes[role]) == 2 for role in _ATTENTION_ROLES)
            and all(shapes[role][1] == width for role in ("q", "k", "v"))
            and shapes["o"][0] == width
        )
    if not valid:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{axis_kind} {name} graph parameter geometry is invalid."
        )


def _writer_geometry_binding(
    *,
    source_graph: ModelCircuitGraph,
    target_graph: ModelCircuitGraph,
    layer_index: int,
    site: str,
    axis_kind: str,
    source_paths: Mapping[str, str],
    target_paths: Mapping[str, str],
    source_width: int,
    target_width: int,
) -> dict[str, Any]:
    """Bind the T5 writer tensor to both graph-declared geometries.

    T5 contracts the component feature axis only.  Its intermediate writer
    therefore retains the source residual-output rows; the existing residual
    map subsequently transports those rows into the target residual width.
    """

    writer_role = "w2" if axis_kind == AXIS_MLP_POST_SWIGLU else "out_proj"
    source_spec = _parameter_spec_at_site(
        source_graph,
        source_paths[writer_role],
        layer_index=layer_index,
        site=site,
        name=f"source_parameter_paths[{writer_role!r}]",
    )
    target_spec = _parameter_spec_at_site(
        target_graph,
        target_paths[writer_role],
        layer_index=layer_index,
        site=site,
        name=f"target_parameter_paths[{writer_role!r}]",
    )
    source_shape = tuple(int(value) for value in source_spec.shape)
    target_shape = tuple(int(value) for value in target_spec.shape)
    if (
        len(source_shape) != 2
        or len(target_shape) != 2
        or source_shape[1] != source_width
        or target_shape[1] != target_width
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{axis_kind} writer geometry disagrees with its feature widths."
        )
    route_child_shape = (source_shape[0], target_shape[1])
    return {
        "writer_role": writer_role,
        "source_writer_shape": list(source_shape),
        "target_writer_shape": list(target_shape),
        "route_child_weight_shape": list(route_child_shape),
        "source_residual_width": source_shape[0],
        "target_residual_width": target_shape[0],
        "residual_interface_transport_required": source_shape[0] != target_shape[0],
        "residual_input_consumer_transport_required": source_shape[0]
        != target_shape[0],
        "residual_output_producer_transport_required": source_shape[0]
        != target_shape[0],
        "residual_interface_transport_bound": False,
        "residual_interface_transport_expected_authority_kind": (
            "existing-residual-fold-map"
        ),
        "residual_interface_transport_required_at_materialization": (
            source_shape[0] != target_shape[0]
        ),
        "residual_interface_transport_operations": [
            "fold_consumer_weight(input_columns)",
            "fold_producer_weight(output_rows)",
        ],
        "admission_scope": "intermediate-feature-axis-only",
        "complete_target_parameter_authority": False,
    }


def lfm2_sparse_feature_axis_binding(
    *,
    source_graph: ModelCircuitGraph,
    target_graph: ModelCircuitGraph,
    source_graph_sha256: str,
    target_graph_sha256: str,
    layer_index: int,
    site: str,
    axis_kind: str,
    source_parameter_paths: Mapping[str, str],
    target_parameter_paths: Mapping[str, str],
    source_parameter_row_slices: Mapping[str, Sequence[int]] | None = None,
    target_parameter_row_slices: Mapping[str, Sequence[int]] | None = None,
    source_width: int,
    target_width: int,
    authority_schema: str,
    authority_record_sha256: str,
) -> dict[str, Any]:
    """Return the namespaced caller binding accepted by the T5 map adapter."""

    if axis_kind not in _AXIS_KINDS:
        raise Lfm2SparseFeatureFoldBundleError("axis_kind is not supported.")
    source_graph = _graph(
        source_graph,
        "source_graph",
        expected_sha256=source_graph_sha256,
    )
    target_graph = _graph(
        target_graph,
        "target_graph",
        expected_sha256=target_graph_sha256,
    )
    source_width = _positive_int(source_width, "source_width")
    target_width = _positive_int(target_width, "target_width")
    source_paths, source_slices = _axis_paths_and_slices(
        axis_kind=axis_kind,
        paths=source_parameter_paths,
        row_slices=source_parameter_row_slices,
        width=source_width,
        name="source",
    )
    target_paths, target_slices = _axis_paths_and_slices(
        axis_kind=axis_kind,
        paths=target_parameter_paths,
        row_slices=target_parameter_row_slices,
        width=target_width,
        name="target",
    )
    if set(source_paths) != set(target_paths):
        raise Lfm2SparseFeatureFoldBundleError(
            "Source and target parameter roles must match exactly."
        )
    layer_index = _layer_index(layer_index)
    site = _stable_id(site, "site")
    _validate_axis_graph_geometry(
        graph=source_graph,
        layer_index=layer_index,
        site=site,
        axis_kind=axis_kind,
        paths=source_paths,
        row_slices=source_slices,
        width=source_width,
        name="source",
    )
    _validate_axis_graph_geometry(
        graph=target_graph,
        layer_index=layer_index,
        site=site,
        axis_kind=axis_kind,
        paths=target_paths,
        row_slices=target_slices,
        width=target_width,
        name="target",
    )
    writer_geometry = _writer_geometry_binding(
        source_graph=source_graph,
        target_graph=target_graph,
        layer_index=layer_index,
        site=site,
        axis_kind=axis_kind,
        source_paths=source_paths,
        target_paths=target_paths,
        source_width=source_width,
        target_width=target_width,
    )
    axis = {
        "schema": LFM2_SPARSE_FEATURE_AXIS_ADMISSION_SCHEMA,
        "source_graph_sha256": _digest(source_graph_sha256, "source_graph_sha256"),
        "target_graph_sha256": _digest(target_graph_sha256, "target_graph_sha256"),
        "layer_index": layer_index,
        "site": site,
        "axis_kind": axis_kind,
        "source_parameter_paths": source_paths,
        "target_parameter_paths": target_paths,
        "source_parameter_row_slices": source_slices,
        "target_parameter_row_slices": target_slices,
        "source_width": source_width,
        "target_width": target_width,
        **writer_geometry,
        "authority_schema": _stable_id(authority_schema, "authority_schema"),
        "authority_record_sha256": _digest(
            authority_record_sha256, "authority_record_sha256"
        ),
        "atomic_shared_map": True,
        "compile_time_only": True,
    }
    # The T5 adapter owns top-level origin keys including ``site`` and widths.
    # Namespacing prevents caller metadata from colliding with or overriding
    # those authenticated origin fields.
    return {"lfm2_axis": axis}


def _axis_operations(axis_kind: str, roles: Sequence[str]) -> dict[str, str]:
    if axis_kind == AXIS_MLP_POST_SWIGLU:
        return {
            "w1": "fold_producer_weight",
            "w3": "fold_producer_weight",
            "w2": "fold_consumer_weight",
        }
    if axis_kind == AXIS_SHORT_CONV_CONTEXT:
        return {
            "in_proj_b": "fold_packed_producer_row_slice",
            "in_proj_c": "fold_packed_producer_row_slice",
            "in_proj_x": "fold_packed_producer_row_slice",
            "fir_kernel": "select_depthwise_kernel_rows",
            "out_proj": "fold_consumer_weight",
        }
    return {role: "residual_axis_use" for role in roles}


def _t5_witness_contract(axis_kind: str) -> tuple[str, str]:
    if axis_kind == AXIS_MLP_POST_SWIGLU:
        return "w2", "mlp.post_swiglu"
    if axis_kind == AXIS_SHORT_CONV_CONTEXT:
        return "out_proj", "short_conv.context"
    raise Lfm2SparseFeatureFoldBundleError(
        "Only MLP and ShortConv axes have a T5 source-witness contract."
    )


def _required_witness_artifacts(
    witness: T5BodyRouteSourceWitness,
) -> dict[str, str]:
    return {
        "parent_state": witness.parent_state_fingerprint,
        "component_evidence_cache": witness.evidence_cache_sha256,
        "component_evidence_selection": witness.evidence_selection_sha256,
        "body_route_record": witness.body_route_record_sha256,
        "body_route_source_witness": witness.witness_sha256,
    }


def _axis_provenance_body(
    *,
    authority: Mapping[str, Any],
    authority_seal_field: str,
    coefficient_set: str | None,
    feature_map: SparseFeatureFoldMap,
    source_witness: T5BodyRouteSourceWitness,
    source_artifact_sha256s: Mapping[str, str] | None,
) -> dict[str, Any]:
    return {
        "schema": LFM2_SPARSE_FEATURE_PROVENANCE_SCHEMA,
        "authority_schema": authority["schema"],
        "authority_record_sha256": authority[authority_seal_field],
        "authority_seal_field": authority_seal_field,
        "coefficient_set": coefficient_set,
        "feature_map_sha256": feature_map.map_sha256,
        "admission_binding_sha256": feature_map.admission_binding_sha256,
        "source_witness_sha256": source_witness.witness_sha256,
        "source_artifact_sha256s": _artifact_digests(source_artifact_sha256s),
        "compile_time_only": True,
    }


@dataclass(frozen=True, slots=True)
class Lfm2SparseFeatureAxisAdmission:
    """One sealed map shared atomically by every path on one LFM2 axis."""

    source_graph: ModelCircuitGraph
    target_graph: ModelCircuitGraph
    source_graph_sha256: str
    target_graph_sha256: str
    layer_index: int
    site: str
    axis_kind: str
    source_parameter_paths: Mapping[str, str]
    target_parameter_paths: Mapping[str, str]
    source_parameter_row_slices: Mapping[str, Sequence[int]]
    target_parameter_row_slices: Mapping[str, Sequence[int]]
    body_route_fold: BodyRouteFold
    source_witness: T5BodyRouteSourceWitness
    feature_map: SparseFeatureFoldMap
    authority_record: Mapping[str, Any]
    authority_seal_field: str
    coefficient_set: str | None
    provenance: Mapping[str, Any]
    report: Mapping[str, Any]
    admission_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.body_route_fold, BodyRouteFold):
            raise Lfm2SparseFeatureFoldBundleError(
                "Axis admission requires the live authenticated BodyRouteFold."
            )
        if not isinstance(self.source_witness, T5BodyRouteSourceWitness):
            raise Lfm2SparseFeatureFoldBundleError(
                "Axis admission requires an authenticated T5 source witness."
            )
        if not isinstance(self.feature_map, SparseFeatureFoldMap):
            raise TypeError("feature_map must be SparseFeatureFoldMap.")
        report = self.feature_map.verify()
        if self.authority_seal_field != "record_sha256":
            raise Lfm2SparseFeatureFoldBundleError(
                "T5 body-route authority must use record_sha256."
            )
        authority = _verify_sealed_mapping(
            self.authority_record,
            name="authority_record",
            seal_field=self.authority_seal_field,
        )
        authority_sha = authority[self.authority_seal_field]
        live_authority = verify_sealed_body_route_record(self.body_route_fold.record)
        if live_authority != authority:
            raise Lfm2SparseFeatureFoldBundleError(
                "Axis authority differs from its live BodyRouteFold."
            )
        caller_binding = lfm2_sparse_feature_axis_binding(
            source_graph=self.source_graph,
            target_graph=self.target_graph,
            source_graph_sha256=self.source_graph_sha256,
            target_graph_sha256=self.target_graph_sha256,
            layer_index=self.layer_index,
            site=self.site,
            axis_kind=self.axis_kind,
            source_parameter_paths=self.source_parameter_paths,
            target_parameter_paths=self.target_parameter_paths,
            source_parameter_row_slices=self.source_parameter_row_slices,
            target_parameter_row_slices=self.target_parameter_row_slices,
            source_width=report.source_width,
            target_width=report.target_width,
            authority_schema=authority["schema"],
            authority_record_sha256=authority_sha,
        )
        axis = caller_binding["lfm2_axis"]
        if authority["schema"] == T5_BODY_ROUTE_RECORD_SCHEMA:
            if self.authority_seal_field != "record_sha256":
                raise Lfm2SparseFeatureFoldBundleError(
                    "T5 body-route authority must use record_sha256."
                )
            expected_binding = {
                **caller_binding,
                **_t5_origin_binding(authority, self.coefficient_set or ""),
            }
            _verify_t5_axis_authority(
                authority,
                axis_kind=self.axis_kind,
                layer_index=axis["layer_index"],
                source_width=report.source_width,
                target_width=report.target_width,
            )
        else:
            if self.axis_kind != AXIS_RESIDUAL or self.coefficient_set is not None:
                raise Lfm2SparseFeatureFoldBundleError(
                    "MLP/ShortConv axes require an authenticated T5 BodyRouteFold."
                )
            expected_binding = caller_binding
        if self.feature_map.admission_binding_sha256 != canonical_sha256(
            expected_binding
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "SparseFeatureFoldMap admission binding does not match this exact axis."
            )
        expected_map = sparse_feature_fold_map_from_body_route(
            self.body_route_fold,
            coefficient_set=self.coefficient_set or "",
            admission_binding=caller_binding,
        )
        if (
            expected_map.map_sha256 != self.feature_map.map_sha256
            or expected_map.admission_binding_sha256
            != self.feature_map.admission_binding_sha256
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "SparseFeatureFoldMap differs from the live BodyRouteFold payload."
            )
        writer_role, evidence_endpoint = _t5_witness_contract(self.axis_kind)
        if self.source_witness.evidence_endpoint != evidence_endpoint:
            raise Lfm2SparseFeatureFoldBundleError(
                "T5 source witness names the wrong component evidence endpoint."
            )
        try:
            authenticated_witness = verify_t5_body_route_source_witness(
                self.source_witness,
                self.body_route_fold,
                parent_weight=self.source_witness.parent_weight,
                component_rows=self.source_witness.component_rows,
                heldout_component_rows=self.source_witness.heldout_component_rows,
                parameter_path=axis["source_parameter_paths"][writer_role],
                graph_fingerprint=axis["source_graph_sha256"],
                parent_state_fingerprint=self.source_witness.parent_state_fingerprint,
                component_id=authority["component_id"],
                layer_index=axis["layer_index"],
                site=authority["site"],
                role="writer",
                evidence_cache_sha256=self.source_witness.evidence_cache_sha256,
                evidence_selection_sha256=(
                    self.source_witness.evidence_selection_sha256
                ),
                evidence_endpoint=evidence_endpoint,
                coefficient_set=self.coefficient_set or "",
            )
        except T5BodyRouteAuthenticationError as exc:
            raise Lfm2SparseFeatureFoldBundleError(
                "T5 source witness failed live route/value authentication."
            ) from exc
        source_writer_shape = tuple(axis["source_writer_shape"])
        route_child_shape = tuple(axis["route_child_weight_shape"])
        if (
            tuple(authenticated_witness.parent_weight.shape) != source_writer_shape
            or authority.get("parent_parameters") != math.prod(source_writer_shape)
            or authority.get("child_parameters") != math.prod(route_child_shape)
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "T5 source/child writer tensors do not match the graph-bound "
                "feature-axis geometry."
            )
        provenance = _verify_sealed_mapping(
            self.provenance,
            name="provenance",
            seal_field="provenance_sha256",
            expected_schema=LFM2_SPARSE_FEATURE_PROVENANCE_SCHEMA,
        )
        if (
            provenance.get("authority_schema") != authority["schema"]
            or provenance.get("authority_record_sha256") != authority_sha
            or provenance.get("feature_map_sha256") != self.feature_map.map_sha256
            or provenance.get("admission_binding_sha256")
            != self.feature_map.admission_binding_sha256
            or provenance.get("coefficient_set") != self.coefficient_set
            or provenance.get("source_witness_sha256")
            != authenticated_witness.witness_sha256
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "Axis provenance does not bind its authority and sparse map."
            )
        artifacts = provenance.get("source_artifact_sha256s")
        if not isinstance(artifacts, Mapping) or any(
            artifacts.get(key) != digest
            for key, digest in _required_witness_artifacts(
                authenticated_witness
            ).items()
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "Axis provenance does not bind the T5 source witness artifacts."
            )
        expected_provenance_body = _axis_provenance_body(
            authority=authority,
            authority_seal_field=self.authority_seal_field,
            coefficient_set=self.coefficient_set,
            feature_map=self.feature_map,
            source_witness=authenticated_witness,
            source_artifact_sha256s=artifacts,
        )
        if provenance != {
            **expected_provenance_body,
            "provenance_sha256": canonical_sha256(expected_provenance_body),
        }:
            raise Lfm2SparseFeatureFoldBundleError(
                "Axis provenance is not the exact authenticated factory contract."
            )
        body = self._report_body(axis, provenance)
        supplied_report = _verify_sealed_mapping(
            self.report,
            name="axis report",
            seal_field="admission_sha256",
            expected_schema=LFM2_SPARSE_FEATURE_AXIS_ADMISSION_SCHEMA,
        )
        admission_sha = _digest(self.admission_sha256, "admission_sha256")
        if (
            supplied_report["admission_sha256"] != admission_sha
            or canonical_sha256(body) != admission_sha
            or {k: v for k, v in supplied_report.items() if k != "admission_sha256"}
            != body
        ):
            raise Lfm2SparseFeatureFoldBundleError("Axis admission seal is invalid.")
        object.__setattr__(
            self,
            "source_parameter_paths",
            MappingProxyType(axis["source_parameter_paths"]),
        )
        object.__setattr__(
            self,
            "target_parameter_paths",
            MappingProxyType(axis["target_parameter_paths"]),
        )
        object.__setattr__(
            self,
            "source_parameter_row_slices",
            MappingProxyType(
                {
                    role: tuple(bounds)
                    for role, bounds in axis["source_parameter_row_slices"].items()
                }
            ),
        )
        object.__setattr__(
            self,
            "target_parameter_row_slices",
            MappingProxyType(
                {
                    role: tuple(bounds)
                    for role, bounds in axis["target_parameter_row_slices"].items()
                }
            ),
        )
        object.__setattr__(self, "authority_record", MappingProxyType(authority))
        object.__setattr__(self, "source_witness", authenticated_witness)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        object.__setattr__(self, "report", MappingProxyType(supplied_report))

    def _report_body(
        self, axis: Mapping[str, Any], provenance: Mapping[str, Any]
    ) -> dict[str, Any]:
        map_report = self.feature_map.report
        roles = tuple(axis["source_parameter_paths"])
        return {
            **dict(axis),
            "authority_site": self.authority_record["site"],
            "feature_map_sha256": self.feature_map.map_sha256,
            "admission_binding_sha256": self.feature_map.admission_binding_sha256,
            "coefficient_set": self.coefficient_set,
            "operations_by_role": _axis_operations(self.axis_kind, roles),
            "provenance_sha256": provenance["provenance_sha256"],
            "source_witness_sha256": self.source_witness.witness_sha256,
            "source_parent_state_fingerprint": (
                self.source_witness.parent_state_fingerprint
            ),
            "source_parent_weight_sha256": self.source_witness.record[
                "parent_weight_sha256"
            ],
            "source_component_rows_sha256": self.source_witness.record[
                "component_rows_sha256"
            ],
            "source_value_reauthentication_required_at_materialization": True,
            "map_stored_elements_diagnostic": map_report.map_stored_elements,
            "map_stored_bytes_diagnostic": map_report.map_stored_bytes,
            "stored_counts_authoritative": False,
            "target_parameter_count_authority": "materialized-target-state-dict",
            "runtime_parameter_count": 0,
            "runtime_buffer_count": 0,
            "runtime_sidecar_count": 0,
        }

    def verify(self) -> "Lfm2SparseFeatureAxisAdmission":
        self.__post_init__()
        return self

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def admit_lfm2_sparse_feature_axis(
    *,
    source_graph: ModelCircuitGraph,
    target_graph: ModelCircuitGraph,
    source_graph_sha256: str,
    target_graph_sha256: str,
    layer_index: int,
    site: str,
    axis_kind: str,
    source_parameter_paths: Mapping[str, str],
    target_parameter_paths: Mapping[str, str],
    source_parameter_row_slices: Mapping[str, Sequence[int]] | None = None,
    target_parameter_row_slices: Mapping[str, Sequence[int]] | None = None,
    body_route_fold: BodyRouteFold | None = None,
    source_witness: T5BodyRouteSourceWitness | None = None,
    coefficient_set: str | None = None,
    feature_map: SparseFeatureFoldMap | None = None,
    authority_record: Mapping[str, Any] | None = None,
    authority_seal_field: str = "record_sha256",
    source_artifact_sha256s: Mapping[str, str] | None = None,
) -> Lfm2SparseFeatureAxisAdmission:
    """Authenticate one route and admit its adapter-produced sparse map.

    MLP and ShortConv admissions require the live
    :class:`BodyRouteFold`; the existing adapter reauthenticates its complete
    CSR/expansion and constructs the only map this bundle will accept.
    Residual/attention admission is deliberately fail-closed in this bundle.
    """

    if (
        axis_kind not in _AXIS_KINDS
        or not isinstance(body_route_fold, BodyRouteFold)
        or not isinstance(source_witness, T5BodyRouteSourceWitness)
        or feature_map is not None
        or authority_record is not None
        or authority_seal_field != "record_sha256"
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            "Sparse feature-axis admission requires an authenticated live "
            "BodyRouteFold and source witness; arbitrary generic residual "
            "authorities are disabled."
        )
    if coefficient_set not in {"read", "write"}:
        raise Lfm2SparseFeatureFoldBundleError(
            "coefficient_set must be explicitly 'read' or 'write'."
        )
    authority = verify_sealed_body_route_record(body_route_fold.record)
    source_width = body_route_fold.parent_width
    target_width = body_route_fold.kept_width
    caller_binding = lfm2_sparse_feature_axis_binding(
        source_graph=source_graph,
        target_graph=target_graph,
        source_graph_sha256=source_graph_sha256,
        target_graph_sha256=target_graph_sha256,
        layer_index=layer_index,
        site=site,
        axis_kind=axis_kind,
        source_parameter_paths=source_parameter_paths,
        target_parameter_paths=target_parameter_paths,
        source_parameter_row_slices=source_parameter_row_slices,
        target_parameter_row_slices=target_parameter_row_slices,
        source_width=source_width,
        target_width=target_width,
        authority_schema=authority["schema"],
        authority_record_sha256=authority[authority_seal_field],
    )
    axis = caller_binding["lfm2_axis"]
    _verify_t5_axis_authority(
        authority,
        axis_kind=axis_kind,
        layer_index=axis["layer_index"],
        source_width=source_width,
        target_width=target_width,
    )
    feature_map = sparse_feature_fold_map_from_body_route(
        body_route_fold,
        coefficient_set=coefficient_set,
        admission_binding=caller_binding,
    )
    assert feature_map is not None
    artifacts = _artifact_digests(source_artifact_sha256s)
    if any(
        artifacts.get(key) != digest
        for key, digest in _required_witness_artifacts(source_witness).items()
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            "source_artifact_sha256s must bind the exact T5 source witness, "
            "parent state, evidence, and route record."
        )
    provenance_body = _axis_provenance_body(
        authority=authority,
        authority_seal_field=authority_seal_field,
        coefficient_set=coefficient_set,
        feature_map=feature_map,
        source_witness=source_witness,
        source_artifact_sha256s=artifacts,
    )
    provenance = {
        **provenance_body,
        "provenance_sha256": canonical_sha256(provenance_body),
    }
    temporary = object.__new__(Lfm2SparseFeatureAxisAdmission)
    for name, value in {
        "source_graph_sha256": axis["source_graph_sha256"],
        "target_graph_sha256": axis["target_graph_sha256"],
        "layer_index": axis["layer_index"],
        "site": axis["site"],
        "axis_kind": axis_kind,
        "source_parameter_paths": axis["source_parameter_paths"],
        "target_parameter_paths": axis["target_parameter_paths"],
        "source_parameter_row_slices": axis["source_parameter_row_slices"],
        "target_parameter_row_slices": axis["target_parameter_row_slices"],
        "source_witness": source_witness,
        "feature_map": feature_map,
        "authority_record": authority,
        "authority_seal_field": authority_seal_field,
        "coefficient_set": coefficient_set,
        "provenance": provenance,
    }.items():
        object.__setattr__(temporary, name, value)
    body = temporary._report_body(axis, provenance)
    admission_sha = canonical_sha256(body)
    return Lfm2SparseFeatureAxisAdmission(
        source_graph=source_graph,
        target_graph=target_graph,
        source_graph_sha256=axis["source_graph_sha256"],
        target_graph_sha256=axis["target_graph_sha256"],
        layer_index=axis["layer_index"],
        site=axis["site"],
        axis_kind=axis_kind,
        source_parameter_paths=axis["source_parameter_paths"],
        target_parameter_paths=axis["target_parameter_paths"],
        source_parameter_row_slices=axis["source_parameter_row_slices"],
        target_parameter_row_slices=axis["target_parameter_row_slices"],
        body_route_fold=body_route_fold,
        source_witness=source_witness,
        feature_map=feature_map,
        authority_record=authority,
        authority_seal_field=authority_seal_field,
        coefficient_set=coefficient_set,
        provenance=provenance,
        report={**body, "admission_sha256": admission_sha},
        admission_sha256=admission_sha,
    )


def _attention_topology(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2SparseFeatureFoldBundleError(f"{name} must be a mapping.")
    allowed = {
        "query_heads",
        "key_value_heads",
        "head_dim",
        "rope_theta",
        "attention_scaling",
        "query_key_norm",
        "rms_norm_epsilon",
        "attention_window",
        "position_encoding",
    }
    required = allowed - {"attention_window", "position_encoding"}
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} fields do not match HFCausalGQATopology."
        )
    query_heads = _positive_int(value["query_heads"], f"{name}.query_heads")
    kv_heads = _positive_int(value["key_value_heads"], f"{name}.key_value_heads")
    head_dim = _positive_int(value["head_dim"], f"{name}.head_dim")
    if query_heads % kv_heads:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name}.query_heads must be divisible by key_value_heads."
        )
    position = str(value.get("position_encoding", "rope")).strip().lower()
    if position not in {"rope", "nope"} or (position == "rope" and head_dim % 2):
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} has invalid position encoding/head dimension."
        )
    norm = str(value["query_key_norm"]).strip().lower()
    if norm not in {"identity", "rms_norm"}:
        raise Lfm2SparseFeatureFoldBundleError(f"{name}.query_key_norm is invalid.")
    result = {
        "query_heads": query_heads,
        "key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rope_theta": _positive_float(value["rope_theta"], f"{name}.rope_theta"),
        "attention_scaling": _positive_float(
            value["attention_scaling"], f"{name}.attention_scaling"
        ),
        "query_key_norm": norm,
        "rms_norm_epsilon": _positive_float(
            value["rms_norm_epsilon"], f"{name}.rms_norm_epsilon"
        ),
    }
    if value.get("attention_window") is not None:
        result["attention_window"] = _positive_int(
            value["attention_window"], f"{name}.attention_window"
        )
    if position != "rope":
        result["position_encoding"] = position
    return result


def _attention_tensor_sha256(value: Any) -> str:
    tensor = value.detach().to(device="cpu").contiguous().clone()
    raw = tensor.view(__import__("torch").uint8).numpy().tobytes()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _finite_report_number(
    authority: Mapping[str, Any], name: str, *, nonnegative: bool = False
) -> float:
    value = authority.get(name)
    if isinstance(value, bool):
        raise Lfm2SparseFeatureFoldBundleError(
            f"coupled attention authority {name} is invalid."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2SparseFeatureFoldBundleError(
            f"coupled attention authority {name} is invalid."
        ) from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise Lfm2SparseFeatureFoldBundleError(
            f"coupled attention authority {name} is invalid."
        )
    return result


def _verify_coupled_attention_authority(
    fold: HFCausalCoupledAttentionFold,
    source_topology: Mapping[str, Any],
    target_topology: Mapping[str, Any],
    *,
    target_residual_width: int,
    target_parameter_specs: Mapping[str, ParameterSpec],
) -> dict[str, Any]:
    if not isinstance(fold, HFCausalCoupledAttentionFold):
        raise Lfm2SparseFeatureFoldBundleError(
            "Changed attention requires a live HFCausalCoupledAttentionFold."
        )
    authority = _verify_sealed_mapping(
        fold.report,
        name="coupled_attention_authority",
        seal_field="report_sha256",
        expected_schema=COUPLED_GQA_ROPE_AUTHORITY_SCHEMA,
    )
    required_true = (
        "all_source_heads_covered_exactly_once",
        "qk_installed_before_vo_fit",
        "target_softmax_probabilities_replayed",
        "complete_target_residual_endpoint_consumed",
    )
    live_source = _attention_topology(
        fold.source_topology.as_dict(), "live source topology"
    )
    live_target = _attention_topology(
        fold.target_topology.as_dict(), "live target topology"
    )
    tensors = {
        "query_weight": fold.query_weight.detach()
        .to(device="cpu")
        .contiguous()
        .clone(),
        "key_weight": fold.key_weight.detach().to(device="cpu").contiguous().clone(),
        "value_weight": fold.value_weight.detach()
        .to(device="cpu")
        .contiguous()
        .clone(),
        "output_weight": fold.output_weight.detach()
        .to(device="cpu")
        .contiguous()
        .clone(),
    }
    target_query_width = int(target_topology["query_heads"]) * int(
        target_topology["head_dim"]
    )
    target_kv_width = int(target_topology["key_value_heads"]) * int(
        target_topology["head_dim"]
    )
    expected_shapes = {
        "query_weight": (target_query_width, target_residual_width),
        "key_weight": (target_kv_width, target_residual_width),
        "value_weight": (target_kv_width, target_residual_width),
        "output_weight": (target_residual_width, target_query_width),
    }
    graph_shapes = {
        "query_weight": tuple(target_parameter_specs["q"].shape),
        "key_weight": tuple(target_parameter_specs["k"].shape),
        "value_weight": tuple(target_parameter_specs["v"].shape),
        "output_weight": tuple(target_parameter_specs["o"].shape),
    }
    tensor_hashes = {
        name: _attention_tensor_sha256(value) for name, value in tensors.items()
    }
    metric_names = (
        "fit_residual_relative_rmse",
        "fit_residual_retained_energy_fraction",
        "validation_residual_relative_rmse",
        "validation_residual_retained_energy_fraction",
        "fit_probability_relative_rmse",
        "validation_probability_relative_rmse",
    )
    for name in metric_names:
        _finite_report_number(
            authority,
            name,
            nonnegative=name.endswith("relative_rmse"),
        )
    _finite_report_number(authority, "ridge", nonnegative=True)
    if (
        _attention_topology(
            authority.get("source_topology"), "authority source topology"
        )
        != dict(source_topology)
        or _attention_topology(
            authority.get("target_topology"), "authority target topology"
        )
        != dict(target_topology)
        or live_source != dict(source_topology)
        or live_target != dict(target_topology)
        or authority.get("component") != "coupled_gqa_rope_qk_softmax_vo_fold"
        or authority.get("method")
        != "barycentric_complete_head_rope_pair_preimage_then_fixed_qk_vo"
        or authority.get("assignment") != fold.assignment.as_dict()
        or authority.get("source_query_head_coverage") != source_topology["query_heads"]
        or authority.get("source_key_value_head_coverage")
        != source_topology["key_value_heads"]
        or any(authority.get(field) is not True for field in required_true)
        or authority.get("numeric_weight_matching_objective") is not False
        or any(
            authority.get(field) != 0
            for field in ("training_steps", "optimizer_steps", "backward_calls")
        )
        or any(
            tuple(tensors[name].shape) != shape
            for name, shape in expected_shapes.items()
        )
        or graph_shapes != expected_shapes
        or any(
            authority.get(f"{name}_sha256") != tensor_hashes[name] for name in tensors
        )
    ):
        raise Lfm2SparseFeatureFoldBundleError(
            "Changed attention geometry lacks the exact coupled GQA/RoPE authority."
        )
    normalized = target_topology["query_key_norm"] == "rms_norm"
    query_norm = fold.query_norm_weight
    key_norm = fold.key_norm_weight
    if normalized:
        head_dim = int(target_topology["head_dim"])
        if (
            query_norm is None
            or key_norm is None
            or tuple(query_norm.shape) != (head_dim,)
            or tuple(key_norm.shape) != (head_dim,)
            or authority.get("target_query_key_norm_jointly_fitted") is not True
            or authority.get("target_head_norm_fitted_against_installed_projection")
            is not True
            or authority.get("target_query_key_norm_epsilon")
            != target_topology["rms_norm_epsilon"]
            or authority.get("query_norm_weight_sha256")
            != _attention_tensor_sha256(query_norm)
            or authority.get("key_norm_weight_sha256")
            != _attention_tensor_sha256(key_norm)
            or _SHA256.fullmatch(str(authority.get("query_norm_report_sha256", "")))
            is None
            or _SHA256.fullmatch(str(authority.get("key_norm_report_sha256", "")))
            is None
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "Changed attention Q/K normalization authority is incomplete."
            )
        _finite_report_number(
            authority, "query_norm_gain_relative_rmse", nonnegative=True
        )
        _finite_report_number(
            authority, "key_norm_gain_relative_rmse", nonnegative=True
        )
    elif query_norm is not None or key_norm is not None:
        raise Lfm2SparseFeatureFoldBundleError(
            "Identity-Q/K topology cannot carry live normalization weights."
        )
    return authority


def _attention_parameter_specs(
    graph: ModelCircuitGraph,
    paths: Mapping[str, str],
    *,
    layer_index: int,
    site: str,
    name: str,
) -> dict[str, ParameterSpec]:
    return {
        role: _parameter_spec_at_site(
            graph,
            paths[role],
            layer_index=layer_index,
            site=site,
            name=f"{name}_parameter_paths[{role!r}]",
        )
        for role in _ATTENTION_ROLES
    }


def _validate_attention_graph_topology(
    specs: Mapping[str, ParameterSpec],
    topology: Mapping[str, Any],
    *,
    residual_width: int,
    name: str,
) -> None:
    query_width = int(topology["query_heads"]) * int(topology["head_dim"])
    kv_width = int(topology["key_value_heads"]) * int(topology["head_dim"])
    expected = {
        "q": (query_width, residual_width),
        "k": (kv_width, residual_width),
        "v": (kv_width, residual_width),
        "o": (residual_width, query_width),
    }
    if {role: tuple(spec.shape) for role, spec in specs.items()} != expected:
        raise Lfm2SparseFeatureFoldBundleError(
            f"{name} attention graph topology/parameter geometry mismatches."
        )


@dataclass(frozen=True, slots=True)
class Lfm2AttentionFeatureAdmission:
    """Sealed identity-or-coupled admission for one complete Q/K/V/O site."""

    source_graph: ModelCircuitGraph
    target_graph: ModelCircuitGraph
    source_graph_sha256: str
    target_graph_sha256: str
    layer_index: int
    site: str
    source_parameter_paths: Mapping[str, str]
    target_parameter_paths: Mapping[str, str]
    residual_axis: Lfm2SparseFeatureAxisAdmission
    source_topology: Mapping[str, Any]
    target_topology: Mapping[str, Any]
    internal_map_sha256s: Mapping[str, str]
    coupled_attention_fold: HFCausalCoupledAttentionFold | None
    report: Mapping[str, Any]
    admission_sha256: str

    def __post_init__(self) -> None:
        raise Lfm2SparseFeatureFoldBundleError(
            "Attention/residual feature admission is disabled until a production "
            "schema-specific residual witness is available."
        )
        source_graph = _graph(
            self.source_graph,
            "source_graph",
            expected_sha256=self.source_graph_sha256,
        )
        target_graph = _graph(
            self.target_graph,
            "target_graph",
            expected_sha256=self.target_graph_sha256,
        )
        if not isinstance(self.residual_axis, Lfm2SparseFeatureAxisAdmission):
            raise Lfm2SparseFeatureFoldBundleError(
                "Attention requires a live residual-axis admission."
            )
        residual = self.residual_axis.verify()
        source_paths = _path_map(
            self.source_parameter_paths,
            "attention source_parameter_paths",
            required_roles=_ATTENTION_ROLES,
        )
        target_paths = _path_map(
            self.target_parameter_paths,
            "attention target_parameter_paths",
            required_roles=_ATTENTION_ROLES,
        )
        source_topology = _attention_topology(self.source_topology, "source_topology")
        target_topology = _attention_topology(self.target_topology, "target_topology")
        layer_index = _layer_index(self.layer_index)
        site = _stable_id(self.site, "site")
        if (
            residual.source_graph_sha256 != source_graph.fingerprint
            or residual.target_graph_sha256 != target_graph.fingerprint
            or residual.layer_index != layer_index
            or residual.site != site
            or residual.axis_kind != AXIS_RESIDUAL
            or dict(residual.source_parameter_paths) != source_paths
            or dict(residual.target_parameter_paths) != target_paths
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "Attention does not match its live residual-axis graph/layer/site/paths."
            )
        source_specs = _attention_parameter_specs(
            source_graph,
            source_paths,
            layer_index=layer_index,
            site=site,
            name="source",
        )
        target_specs = _attention_parameter_specs(
            target_graph,
            target_paths,
            layer_index=layer_index,
            site=site,
            name="target",
        )
        _validate_attention_graph_topology(
            source_specs,
            source_topology,
            residual_width=residual.feature_map.source_width,
            name="source",
        )
        _validate_attention_graph_topology(
            target_specs,
            target_topology,
            residual_width=residual.feature_map.target_width,
            name="target",
        )
        same_topology = source_topology == target_topology
        maps = dict(self.internal_map_sha256s)
        authority: dict[str, Any] | None = None
        if same_topology:
            expected = {
                component: LFM2_IDENTITY_INTERNAL_FEATURE_MAP_SHA256
                for component in _ATTENTION_COMPONENTS
            }
            if maps != expected or self.coupled_attention_fold is not None:
                raise Lfm2SparseFeatureFoldBundleError(
                    "Same-topology attention requires one declared identity for every C06/C07/C08/C09 map and no coupled authority."
                )
        else:
            if maps:
                raise Lfm2SparseFeatureFoldBundleError(
                    "Changed attention geometry forbids independent C06/C07/C08/C09 maps."
                )
            if self.coupled_attention_fold is None:
                raise Lfm2SparseFeatureFoldBundleError(
                    "Changed attention geometry requires coupled GQA/RoPE authority."
                )
            authority = _verify_coupled_attention_authority(
                self.coupled_attention_fold,
                source_topology,
                target_topology,
                target_residual_width=residual.feature_map.target_width,
                target_parameter_specs=target_specs,
            )
        body = {
            "schema": LFM2_ATTENTION_FEATURE_ADMISSION_SCHEMA,
            "source_graph_sha256": _digest(
                self.source_graph_sha256, "source_graph_sha256"
            ),
            "target_graph_sha256": _digest(
                self.target_graph_sha256, "target_graph_sha256"
            ),
            "layer_index": layer_index,
            "site": site,
            "source_parameter_paths": source_paths,
            "target_parameter_paths": target_paths,
            "residual_axis_admission_sha256": residual.admission_sha256,
            "source_topology": source_topology,
            "target_topology": target_topology,
            "same_topology": same_topology,
            "internal_map_sha256s": dict(sorted(maps.items())),
            "coupled_authority_report_sha256": (
                None if authority is None else authority["report_sha256"]
            ),
            "independent_qkvo_maps_allowed": False,
            "compile_time_only": True,
            "stored_counts_authoritative": False,
            "target_parameter_count_authority": "materialized-target-state-dict",
            "runtime_parameter_count": 0,
            "runtime_buffer_count": 0,
            "runtime_sidecar_count": 0,
        }
        supplied = _verify_sealed_mapping(
            self.report,
            name="attention report",
            seal_field="admission_sha256",
            expected_schema=LFM2_ATTENTION_FEATURE_ADMISSION_SCHEMA,
        )
        seal = _digest(self.admission_sha256, "admission_sha256")
        if (
            supplied["admission_sha256"] != seal
            or canonical_sha256(body) != seal
            or {k: v for k, v in supplied.items() if k != "admission_sha256"} != body
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "Attention admission seal is invalid."
            )
        object.__setattr__(
            self, "source_parameter_paths", MappingProxyType(source_paths)
        )
        object.__setattr__(
            self, "target_parameter_paths", MappingProxyType(target_paths)
        )
        object.__setattr__(self, "source_topology", MappingProxyType(source_topology))
        object.__setattr__(self, "target_topology", MappingProxyType(target_topology))
        object.__setattr__(
            self, "internal_map_sha256s", MappingProxyType(dict(sorted(maps.items())))
        )
        object.__setattr__(self, "report", MappingProxyType(supplied))

    def verify(self) -> "Lfm2AttentionFeatureAdmission":
        self.__post_init__()
        return self

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def admit_lfm2_attention_feature_site(
    *,
    source_graph: ModelCircuitGraph,
    target_graph: ModelCircuitGraph,
    source_graph_sha256: str,
    target_graph_sha256: str,
    layer_index: int,
    site: str,
    source_parameter_paths: Mapping[str, str],
    target_parameter_paths: Mapping[str, str],
    residual_axis: Lfm2SparseFeatureAxisAdmission,
    source_topology: Mapping[str, Any],
    target_topology: Mapping[str, Any],
    internal_map_sha256s: Mapping[str, str] | None = None,
    coupled_attention_fold: HFCausalCoupledAttentionFold | None = None,
    coupled_authority_report: Mapping[str, Any] | None = None,
) -> Lfm2AttentionFeatureAdmission:
    """Fail closed until a production residual-axis witness exists.

    ``HFCausalCoupledAttentionFold`` authenticates the changed internal
    attention geometry, but it does not authenticate the residual-width map
    consumed by the site.  Admitting either same- or changed-topology attention
    here would therefore leave an unauthenticated residual lane.
    """

    raise Lfm2SparseFeatureFoldBundleError(
        "Attention/residual feature admission is disabled until a production "
        "schema-specific residual witness is available."
    )


def _bundle_provenance_body(
    *,
    source_graph_sha256: str,
    target_graph_sha256: str,
    axes: Sequence[Lfm2SparseFeatureAxisAdmission],
    attention_sites: Sequence[Lfm2AttentionFeatureAdmission],
    source_artifact_sha256s: Mapping[str, str] | None,
) -> dict[str, Any]:
    return {
        "schema": LFM2_SPARSE_FEATURE_PROVENANCE_SCHEMA,
        "source_graph_sha256": source_graph_sha256,
        "target_graph_sha256": target_graph_sha256,
        "source_artifact_sha256s": _artifact_digests(source_artifact_sha256s),
        "axis_admission_sha256s": [value.admission_sha256 for value in axes],
        "attention_admission_sha256s": [
            value.admission_sha256 for value in attention_sites
        ],
        "compile_time_only": True,
    }


@dataclass(frozen=True, slots=True)
class Lfm2SparseFeatureFoldBundle:
    """Pure compile-time bundle; only a materialized child owns runtime state."""

    source_graph: ModelCircuitGraph
    target_graph: ModelCircuitGraph
    source_graph_sha256: str
    target_graph_sha256: str
    axes: tuple[Lfm2SparseFeatureAxisAdmission, ...]
    attention_sites: tuple[Lfm2AttentionFeatureAdmission, ...]
    provenance: Mapping[str, Any]
    report: Mapping[str, Any]
    bundle_sha256: str

    def __post_init__(self) -> None:
        source_graph_value = _graph(
            self.source_graph,
            "source_graph",
            expected_sha256=self.source_graph_sha256,
        )
        target_graph_value = _graph(
            self.target_graph,
            "target_graph",
            expected_sha256=self.target_graph_sha256,
        )
        source_graph = source_graph_value.fingerprint
        target_graph = target_graph_value.fingerprint
        axes = tuple(self.axes)
        attention = tuple(self.attention_sites)
        if attention:
            raise Lfm2SparseFeatureFoldBundleError(
                "This sparse bundle admits MLP and ShortConv axes only; "
                "attention/residual admission is fail-closed."
            )
        if not axes or not all(
            isinstance(value, Lfm2SparseFeatureAxisAdmission) for value in axes
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "axes must contain at least one sparse feature admission."
            )
        if not all(
            isinstance(value, Lfm2AttentionFeatureAdmission) for value in attention
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "attention_sites contains an invalid admission."
            )
        for value in (*axes, *attention):
            value.verify()
            if (
                value.source_graph_sha256 != source_graph
                or value.target_graph_sha256 != target_graph
                or value.source_graph.fingerprint != source_graph
                or value.target_graph.fingerprint != target_graph
            ):
                raise Lfm2SparseFeatureFoldBundleError(
                    "A bundle admission names different source/target graphs."
                )
        axis_keys = [(value.layer_index, value.axis_kind) for value in axes]
        attention_keys = [(value.layer_index, value.site) for value in attention]
        if len(set(axis_keys)) != len(axis_keys) or len(set(attention_keys)) != len(
            attention_keys
        ):
            raise Lfm2SparseFeatureFoldBundleError(
                "Bundle layer/semantic-axis admission keys must be unique."
            )
        axes_by_sha = {value.admission_sha256: value for value in axes}
        if len(axes_by_sha) != len(axes):
            raise Lfm2SparseFeatureFoldBundleError(
                "Axis admission seals must be unique."
            )
        provenance = _verify_sealed_mapping(
            self.provenance,
            name="bundle provenance",
            seal_field="provenance_sha256",
            expected_schema=LFM2_SPARSE_FEATURE_PROVENANCE_SCHEMA,
        )
        expected_provenance_body = _bundle_provenance_body(
            source_graph_sha256=source_graph,
            target_graph_sha256=target_graph,
            axes=axes,
            attention_sites=attention,
            source_artifact_sha256s=provenance.get("source_artifact_sha256s"),
        )
        if provenance != {
            **expected_provenance_body,
            "provenance_sha256": canonical_sha256(expected_provenance_body),
        }:
            raise Lfm2SparseFeatureFoldBundleError(
                "Bundle provenance does not bind its exact graphs and admissions."
            )
        body = self._report_body(
            source_graph, target_graph, axes, attention, provenance
        )
        supplied = _verify_sealed_mapping(
            self.report,
            name="bundle report",
            seal_field="bundle_sha256",
            expected_schema=LFM2_SPARSE_FEATURE_FOLD_BUNDLE_SCHEMA,
        )
        bundle_sha = _digest(self.bundle_sha256, "bundle_sha256")
        if (
            supplied["bundle_sha256"] != bundle_sha
            or canonical_sha256(body) != bundle_sha
            or {k: v for k, v in supplied.items() if k != "bundle_sha256"} != body
        ):
            raise Lfm2SparseFeatureFoldBundleError("Bundle seal is invalid.")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "attention_sites", attention)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        object.__setattr__(self, "report", MappingProxyType(supplied))

    @staticmethod
    def _report_body(
        source_graph: str,
        target_graph: str,
        axes: Sequence[Lfm2SparseFeatureAxisAdmission],
        attention: Sequence[Lfm2AttentionFeatureAdmission],
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        unique_maps = {
            value.feature_map.map_sha256: value.feature_map for value in axes
        }
        return {
            "schema": LFM2_SPARSE_FEATURE_FOLD_BUNDLE_SCHEMA,
            "source_graph_sha256": source_graph,
            "target_graph_sha256": target_graph,
            "axis_admission_sha256s": [value.admission_sha256 for value in axes],
            "attention_admission_sha256s": [
                value.admission_sha256 for value in attention
            ],
            "provenance_sha256": provenance["provenance_sha256"],
            "axis_count": len(axes),
            "attention_site_count": len(attention),
            "unique_sparse_map_count": len(unique_maps),
            "map_stored_elements_diagnostic": sum(
                value.stored_elements for value in unique_maps.values()
            ),
            "map_stored_bytes_diagnostic": sum(
                value.stored_bytes for value in unique_maps.values()
            ),
            "stored_counts_authoritative": False,
            "target_parameter_count_authority": "materialized-target-state-dict",
            "compile_time_only": True,
            "runtime_parameter_count": 0,
            "runtime_buffer_count": 0,
            "runtime_sidecar_count": 0,
        }

    def verify(self) -> "Lfm2SparseFeatureFoldBundle":
        self.__post_init__()
        return self

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def build_lfm2_sparse_feature_fold_bundle(
    *,
    source_graph: ModelCircuitGraph,
    target_graph: ModelCircuitGraph,
    source_graph_sha256: str,
    target_graph_sha256: str,
    axes: Sequence[Lfm2SparseFeatureAxisAdmission],
    attention_sites: Sequence[Lfm2AttentionFeatureAdmission] = (),
    source_artifact_sha256s: Mapping[str, str] | None = None,
) -> Lfm2SparseFeatureFoldBundle:
    """Seal admitted component maps; perform no solve, install, or model load."""

    source_graph_value = _graph(
        source_graph,
        "source_graph",
        expected_sha256=source_graph_sha256,
    )
    target_graph_value = _graph(
        target_graph,
        "target_graph",
        expected_sha256=target_graph_sha256,
    )
    source_graph_digest = source_graph_value.fingerprint
    target_graph_digest = target_graph_value.fingerprint
    axis_values = tuple(axes)
    attention_values = tuple(attention_sites)
    if attention_values:
        raise Lfm2SparseFeatureFoldBundleError(
            "This sparse bundle admits MLP and ShortConv axes only; "
            "attention/residual admission is fail-closed."
        )
    provenance_body = _bundle_provenance_body(
        source_graph_sha256=source_graph_digest,
        target_graph_sha256=target_graph_digest,
        axes=axis_values,
        attention_sites=attention_values,
        source_artifact_sha256s=source_artifact_sha256s,
    )
    provenance = {
        **provenance_body,
        "provenance_sha256": canonical_sha256(provenance_body),
    }
    body = Lfm2SparseFeatureFoldBundle._report_body(
        source_graph_digest,
        target_graph_digest,
        axis_values,
        attention_values,
        provenance,
    )
    bundle_sha = canonical_sha256(body)
    return Lfm2SparseFeatureFoldBundle(
        source_graph=source_graph_value,
        target_graph=target_graph_value,
        source_graph_sha256=source_graph_digest,
        target_graph_sha256=target_graph_digest,
        axes=axis_values,
        attention_sites=attention_values,
        provenance=provenance,
        report={**body, "bundle_sha256": bundle_sha},
        bundle_sha256=bundle_sha,
    )


__all__ = [
    "AXIS_MLP_POST_SWIGLU",
    "AXIS_RESIDUAL",
    "AXIS_SHORT_CONV_CONTEXT",
    "COUPLED_GQA_ROPE_AUTHORITY_SCHEMA",
    "LFM2_ATTENTION_FEATURE_ADMISSION_SCHEMA",
    "LFM2_IDENTITY_INTERNAL_FEATURE_MAP_SHA256",
    "LFM2_IDENTITY_INTERNAL_FEATURE_MAP_SCHEMA",
    "LFM2_SPARSE_FEATURE_AXIS_ADMISSION_SCHEMA",
    "LFM2_SPARSE_FEATURE_FOLD_BUNDLE_SCHEMA",
    "LFM2_SPARSE_FEATURE_PROVENANCE_SCHEMA",
    "Lfm2AttentionFeatureAdmission",
    "Lfm2SparseFeatureAxisAdmission",
    "Lfm2SparseFeatureFoldBundle",
    "Lfm2SparseFeatureFoldBundleError",
    "T5_BODY_ROUTE_RECORD_SCHEMA",
    "admit_lfm2_attention_feature_site",
    "admit_lfm2_sparse_feature_axis",
    "build_lfm2_sparse_feature_fold_bundle",
    "lfm2_sparse_feature_axis_binding",
]
