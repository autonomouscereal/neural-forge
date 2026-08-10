from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    CircuitKind,
    CircuitNode,
    ModelCircuitGraph,
    ParameterSpec,
    canonical_sha256,
)
from neural_forge.hf_causal_gqa_attention_fold import (  # noqa: E402
    HFCausalGQATopology,
    compile_hf_causal_gqa_attention_fold,
)
from neural_forge.hf_feature_fold import SparseFeatureFoldMap  # noqa: E402
from neural_forge.hf_lfm2_sparse_feature_fold_bundle import (  # noqa: E402
    AXIS_MLP_POST_SWIGLU,
    AXIS_RESIDUAL,
    AXIS_SHORT_CONV_CONTEXT,
    Lfm2SparseFeatureFoldBundle,
    Lfm2SparseFeatureFoldBundleError,
    admit_lfm2_attention_feature_site,
    admit_lfm2_sparse_feature_axis,
    build_lfm2_sparse_feature_fold_bundle,
    lfm2_sparse_feature_axis_binding,
)
from neural_forge.matrix_technique_body_routes import (  # noqa: E402
    build_t5_body_route_source_witness,
    fit_body_component_routes,
)


MLP_SITE = "model.layers.0.feed_forward"
CONV_SITE = "model.layers.0.conv"
ATTN_SITE = "model.layers.0.self_attn"


def _parameter(path: str, role: str, shape: tuple[int, ...]) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        role=role,
        shape=shape,
        dtype="float64",
        storage_group=path,
        trainable=True,
    )


def _paths(axis_kind: str, *, site: str | None = None) -> dict[str, str]:
    if axis_kind == AXIS_MLP_POST_SWIGLU:
        mlp_site = MLP_SITE if site is None else site
        return {
            "w1": f"{mlp_site}.w1.weight",
            "w3": f"{mlp_site}.w3.weight",
            "w2": f"{mlp_site}.w2.weight",
        }
    if axis_kind == AXIS_SHORT_CONV_CONTEXT:
        conv_site = CONV_SITE if site is None else site
        packed = f"{conv_site}.in_proj.weight"
        return {
            "in_proj_b": packed,
            "in_proj_c": packed,
            "in_proj_x": packed,
            "fir_kernel": f"{conv_site}.conv.weight",
            "out_proj": f"{conv_site}.out_proj.weight",
        }
    attention_site = ATTN_SITE if site is None else site
    return {
        "q": f"{attention_site}.q_proj.weight",
        "k": f"{attention_site}.k_proj.weight",
        "v": f"{attention_site}.v_proj.weight",
        "o": f"{attention_site}.o_proj.weight",
    }


def _short_slices(width: int) -> dict[str, tuple[int, int]]:
    return {
        "in_proj_b": (0, width),
        "in_proj_c": (width, 2 * width),
        "in_proj_x": (2 * width, 3 * width),
    }


def _graph(
    *,
    width: int,
    model_name: str,
    malformed_packed: bool = False,
    fir_length: int = 3,
    residual_width: int = 3,
):
    mlp_paths = _paths(AXIS_MLP_POST_SWIGLU)
    conv_paths = _paths(AXIS_SHORT_CONV_CONTEXT)
    attention_paths = _paths(AXIS_RESIDUAL)
    packed_rows = 3 * width + int(malformed_packed)
    nodes = (
        CircuitNode(
            node_id="layer.000.mlp",
            kind=CircuitKind.GATED_MLP,
            order=0,
            path=MLP_SITE,
            layer_index=0,
            inputs=(),
            outputs=(),
            parameters=(
                _parameter(mlp_paths["w1"], "mlp_gate", (width, residual_width)),
                _parameter(mlp_paths["w3"], "mlp_up", (width, residual_width)),
                _parameter(mlp_paths["w2"], "mlp_down", (residual_width, width)),
            ),
        ),
        CircuitNode(
            node_id="layer.000.conv",
            kind=CircuitKind.SHORT_CONV,
            order=1,
            path=CONV_SITE,
            layer_index=0,
            inputs=(),
            outputs=(),
            parameters=(
                _parameter(
                    conv_paths["in_proj_b"],
                    "conv_in_proj",
                    (packed_rows, residual_width),
                ),
                _parameter(
                    conv_paths["fir_kernel"],
                    "conv_kernel",
                    (width, 1, fir_length),
                ),
                _parameter(
                    conv_paths["out_proj"],
                    "conv_out_proj",
                    (residual_width, width),
                ),
            ),
        ),
        CircuitNode(
            node_id="layer.000.attention",
            kind=CircuitKind.ATTENTION,
            order=2,
            path=ATTN_SITE,
            layer_index=0,
            inputs=(),
            outputs=(),
            parameters=(
                _parameter(attention_paths["q"], "attn_q", (width, residual_width)),
                _parameter(attention_paths["k"], "attn_k", (width, residual_width)),
                _parameter(attention_paths["v"], "attn_v", (width, residual_width)),
                _parameter(attention_paths["o"], "attn_o", (residual_width, width)),
            ),
        ),
    )
    parameter_count = sum(node.logical_parameter_count for node in nodes)
    return ModelCircuitGraph(
        model_name=model_name,
        family="lfm2-fixture",
        model_class="tests.Lfm2Fixture",
        nodes=nodes,
        edges=(),
        interfaces=(),
        logical_parameter_count=parameter_count,
        unique_parameter_count=parameter_count,
        unique_parameter_bytes=parameter_count * 8,
        config_fingerprint=hashlib.sha256(model_name.encode()).hexdigest(),
    )


def _with_parameter_shape(
    graph: ModelCircuitGraph,
    *,
    path: str,
    shape: tuple[int, ...],
) -> ModelCircuitGraph:
    found = False
    changed_nodes = []
    delta = 0
    for node in graph.nodes:
        parameters = []
        for parameter in node.parameters:
            if parameter.path == path:
                found = True
                delta += math.prod(shape) - math.prod(parameter.shape)
                parameter = replace(parameter, shape=shape)
            parameters.append(parameter)
        changed_nodes.append(replace(node, parameters=tuple(parameters)))
    assert found
    return replace(
        graph,
        nodes=tuple(changed_nodes),
        logical_parameter_count=graph.logical_parameter_count + delta,
        unique_parameter_count=graph.unique_parameter_count + delta,
        unique_parameter_bytes=graph.unique_parameter_bytes + delta * 8,
    )


def _with_mlp_path_decoys(graph: ModelCircuitGraph) -> ModelCircuitGraph:
    node = next(value for value in graph.nodes if value.path == MLP_SITE)
    suffixes = ("w1", "w3", "w2")
    decoys = tuple(
        _parameter(
            f"{MLP_SITE}.decoy.{suffix}.weight",
            parameter.role,
            parameter.shape,
        )
        for suffix, parameter in zip(suffixes, node.parameters, strict=True)
    )
    changed = replace(node, parameters=(*node.parameters, *decoys))
    added = sum(math.prod(parameter.shape) for parameter in decoys)
    return replace(
        graph,
        nodes=tuple(changed if value is node else value for value in graph.nodes),
        logical_parameter_count=graph.logical_parameter_count + added,
        unique_parameter_count=graph.unique_parameter_count + added,
        unique_parameter_bytes=graph.unique_parameter_bytes + added * 8,
    )


def _with_second_mlp(graph: ModelCircuitGraph, *, width: int) -> ModelCircuitGraph:
    site = "model.layers.0.second_feed_forward"
    paths = _paths(AXIS_MLP_POST_SWIGLU, site=site)
    node = CircuitNode(
        node_id="layer.000.second_mlp",
        kind=CircuitKind.GATED_MLP,
        order=max(value.order for value in graph.nodes) + 1,
        path=site,
        layer_index=0,
        inputs=(),
        outputs=(),
        parameters=(
            _parameter(paths["w1"], "mlp_gate", (width, 3)),
            _parameter(paths["w3"], "mlp_up", (width, 3)),
            _parameter(paths["w2"], "mlp_down", (3, width)),
        ),
    )
    added = node.logical_parameter_count
    return replace(
        graph,
        nodes=(*graph.nodes, node),
        logical_parameter_count=graph.logical_parameter_count + added,
        unique_parameter_count=graph.unique_parameter_count + added,
        unique_parameter_bytes=graph.unique_parameter_bytes + added * 8,
    )


@pytest.fixture
def graphs() -> tuple[ModelCircuitGraph, ModelCircuitGraph]:
    return _graph(width=4, model_name="parent"), _graph(width=2, model_name="child")


def _parent_weight() -> torch.Tensor:
    return torch.arange(12, dtype=torch.float64).reshape(3, 4) / 17.0


def _component_rows() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 0.2, -0.4, 0.7],
            [0.3, -0.8, 0.9, 0.5],
            [-0.6, 0.4, 0.2, 1.1],
            [0.8, 0.6, -0.3, -0.2],
            [0.1, -0.5, 0.7, 0.9],
            [-0.2, 1.0, 0.5, -0.7],
        ],
        dtype=torch.float64,
    )


def _body_route(component: str, *, site: str = "L00"):
    return fit_body_component_routes(
        parent_weight=_parent_weight(),
        component_rows=_component_rows(),
        component=component,
        level="X4",
        site=site,
        kept_width=2,
        active_k=2,
        channel_selection="explicit",
        kept_channel_ids=(0, 2),
        install_coefficients="write",
    )


def _source_witness(
    fold,
    source_graph: ModelCircuitGraph,
    axis_kind: str,
    *,
    graph_site: str | None = None,
):
    writer_role = "w2" if axis_kind == AXIS_MLP_POST_SWIGLU else "out_proj"
    endpoint = (
        "mlp.post_swiglu" if axis_kind == AXIS_MLP_POST_SWIGLU else "short_conv.context"
    )
    return build_t5_body_route_source_witness(
        fold,
        parent_weight=_parent_weight(),
        component_rows=_component_rows(),
        parameter_path=_paths(axis_kind, site=graph_site)[writer_role],
        graph_fingerprint=source_graph.fingerprint,
        parent_state_fingerprint="a" * 64,
        component_id=fold.component_id,
        layer_index=0,
        site="L00",
        role="writer",
        evidence_cache_sha256="b" * 64,
        evidence_selection_sha256="c" * 64,
        evidence_endpoint=endpoint,
        coefficient_set="write",
    )


def _source_artifacts(witness) -> dict[str, str]:
    return {
        "parent_state": witness.parent_state_fingerprint,
        "component_evidence_cache": witness.evidence_cache_sha256,
        "component_evidence_selection": witness.evidence_selection_sha256,
        "body_route_record": witness.body_route_record_sha256,
        "body_route_source_witness": witness.witness_sha256,
    }


def _axis(
    axis_kind: str,
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
):
    source_graph, target_graph = graphs
    fold = _body_route("C13" if axis_kind == AXIS_MLP_POST_SWIGLU else "C05")
    witness = _source_witness(fold, source_graph, axis_kind)
    kwargs = {
        "source_graph": source_graph,
        "target_graph": target_graph,
        "source_graph_sha256": source_graph.fingerprint,
        "target_graph_sha256": target_graph.fingerprint,
        "layer_index": 0,
        "site": MLP_SITE if axis_kind == AXIS_MLP_POST_SWIGLU else CONV_SITE,
        "axis_kind": axis_kind,
        "source_parameter_paths": _paths(axis_kind),
        "target_parameter_paths": _paths(axis_kind),
        "body_route_fold": fold,
        "source_witness": witness,
        "coefficient_set": "write",
        "source_artifact_sha256s": _source_artifacts(witness),
    }
    if axis_kind == AXIS_SHORT_CONV_CONTEXT:
        kwargs.update(
            source_parameter_row_slices=_short_slices(4),
            target_parameter_row_slices=_short_slices(2),
        )
    return admit_lfm2_sparse_feature_axis(**kwargs)


def _topology(*, query_heads: int, kv_heads: int, head_dim: int = 2):
    return {
        "query_heads": query_heads,
        "key_value_heads": kv_heads,
        "head_dim": head_dim,
        "rope_theta": 10_000.0,
        "attention_scaling": head_dim**-0.5,
        "query_key_norm": "identity",
        "rms_norm_epsilon": 1e-6,
    }


def _live_changed_attention_fold():
    generator = torch.Generator().manual_seed(41)
    source = HFCausalGQATopology(**_topology(query_heads=2, kv_heads=1))
    target = HFCausalGQATopology(**_topology(query_heads=1, kv_heads=1))
    child = torch.randn((1, 4, 3), generator=generator, dtype=torch.float64)
    valid = torch.ones((1, 4), dtype=torch.bool)
    fit = torch.tensor([[True, True, False, False]])
    validation = torch.tensor([[False, False, True, True]])
    return compile_hf_causal_gqa_attention_fold(
        child,
        source_query_sequences=torch.randn(
            (1, 4, 4), generator=generator, dtype=torch.float64
        ),
        source_key_sequences=torch.randn(
            (1, 4, 2), generator=generator, dtype=torch.float64
        ),
        source_value_sequences=torch.randn(
            (1, 4, 2), generator=generator, dtype=torch.float64
        ),
        target_residual_delta_sequences=torch.randn(
            (1, 4, 3), generator=generator, dtype=torch.float64
        ),
        attention_mask=valid,
        position_ids=torch.arange(4).reshape(1, 4),
        valid_token_mask=valid,
        fit_token_mask=fit,
        validation_token_mask=validation,
        source_topology=source,
        target_topology=target,
        ridge=1e-4,
    )


def test_mlp_admits_one_live_c13_map_and_matches_writer_child(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    axis = _axis(AXIS_MLP_POST_SWIGLU, graphs)
    fold = _body_route("C13")

    assert axis.report["authority_site"] == "L00"
    assert axis.report["site"] == MLP_SITE
    assert axis.report["operations_by_role"] == {
        "w1": "fold_producer_weight",
        "w2": "fold_consumer_weight",
        "w3": "fold_producer_weight",
    }
    assert torch.equal(
        axis.feature_map.fold_consumer_weight(_parent_weight()),
        fold.child_weight,
    )
    assert axis.report["runtime_parameter_count"] == 0
    assert axis.report["source_witness_sha256"] == axis.source_witness.witness_sha256
    assert axis.report["source_value_reauthentication_required_at_materialization"]
    assert (
        axis.source_witness.report["authority"]["solver_origin"]
        == "external-artifact-bound-not-attested"
    )
    assert axis.verify() is axis
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="BodyRouteFold"):
        replace(axis, body_route_fold=None)
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="source witness"):
        replace(axis, source_witness=None)
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="differs"):
        replace(axis, body_route_fold=_body_route("C05"))


def test_mlp_binds_witness_geometry_and_composes_changed_residual_width() -> None:
    source_graph = _graph(width=4, model_name="parent", residual_width=3)
    target_graph = _graph(width=2, model_name="child", residual_width=2)
    axis = _axis(AXIS_MLP_POST_SWIGLU, (source_graph, target_graph))

    assert axis.report["source_writer_shape"] == [3, 4]
    assert axis.report["route_child_weight_shape"] == [3, 2]
    assert axis.report["target_writer_shape"] == [2, 2]
    assert axis.report["residual_interface_transport_required"]
    assert axis.report["residual_input_consumer_transport_required"]
    assert axis.report["residual_output_producer_transport_required"]
    assert not axis.report["residual_interface_transport_bound"]
    assert axis.report["residual_interface_transport_required_at_materialization"]
    assert (
        axis.report["residual_interface_transport_expected_authority_kind"]
        == "existing-residual-fold-map"
    )
    assert axis.report["residual_interface_transport_operations"] == [
        "fold_consumer_weight(input_columns)",
        "fold_producer_weight(output_rows)",
    ]
    assert axis.report["admission_scope"] == "intermediate-feature-axis-only"
    assert not axis.report["complete_target_parameter_authority"]

    foreign_weight = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 23.0
    foreign_fold = fit_body_component_routes(
        parent_weight=foreign_weight,
        component_rows=_component_rows(),
        component="C13",
        level="X4",
        site="L00",
        kept_width=2,
        active_k=2,
        channel_selection="explicit",
        kept_channel_ids=(0, 2),
        install_coefficients="write",
    )
    foreign_witness = build_t5_body_route_source_witness(
        foreign_fold,
        parent_weight=foreign_weight,
        component_rows=_component_rows(),
        parameter_path=_paths(AXIS_MLP_POST_SWIGLU)["w2"],
        graph_fingerprint=source_graph.fingerprint,
        parent_state_fingerprint="a" * 64,
        component_id="C13",
        layer_index=0,
        site="L00",
        role="writer",
        evidence_cache_sha256="b" * 64,
        evidence_selection_sha256="c" * 64,
        evidence_endpoint="mlp.post_swiglu",
        coefficient_set="write",
    )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="graph-bound"):
        admit_lfm2_sparse_feature_axis(
            source_graph=source_graph,
            target_graph=target_graph,
            source_graph_sha256=source_graph.fingerprint,
            target_graph_sha256=target_graph.fingerprint,
            layer_index=0,
            site=MLP_SITE,
            axis_kind=AXIS_MLP_POST_SWIGLU,
            source_parameter_paths=_paths(AXIS_MLP_POST_SWIGLU),
            target_parameter_paths=_paths(AXIS_MLP_POST_SWIGLU),
            body_route_fold=foreign_fold,
            source_witness=foreign_witness,
            coefficient_set="write",
            source_artifact_sha256s=_source_artifacts(foreign_witness),
        )


def test_mlp_rejects_partial_roles_wrong_component_and_wrong_authority_site(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    source_graph, target_graph = graphs
    fold = _body_route("C13")
    witness = _source_witness(fold, source_graph, AXIS_MLP_POST_SWIGLU)
    common = {
        "source_graph": source_graph,
        "target_graph": target_graph,
        "source_graph_sha256": source_graph.fingerprint,
        "target_graph_sha256": target_graph.fingerprint,
        "layer_index": 0,
        "site": MLP_SITE,
        "axis_kind": AXIS_MLP_POST_SWIGLU,
        "target_parameter_paths": _paths(AXIS_MLP_POST_SWIGLU),
        "body_route_fold": fold,
        "source_witness": witness,
        "coefficient_set": "write",
        "source_artifact_sha256s": _source_artifacts(witness),
    }
    partial = _paths(AXIS_MLP_POST_SWIGLU)
    partial.pop("w3")
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="exactly"):
        admit_lfm2_sparse_feature_axis(
            **common,
            source_parameter_paths=partial,
        )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="C13"):
        admit_lfm2_sparse_feature_axis(
            **{
                **common,
                "source_parameter_paths": _paths(AXIS_MLP_POST_SWIGLU),
                "body_route_fold": _body_route("C05"),
            }
        )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="C13"):
        admit_lfm2_sparse_feature_axis(
            **{
                **common,
                "source_parameter_paths": _paths(AXIS_MLP_POST_SWIGLU),
                "body_route_fold": _body_route("C13", site="L05"),
            }
        )
    no_coefficient_set = dict(common)
    no_coefficient_set.pop("coefficient_set")
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="explicitly"):
        admit_lfm2_sparse_feature_axis(
            **no_coefficient_set,
            source_parameter_paths=_paths(AXIS_MLP_POST_SWIGLU),
        )


def test_graph_layer_site_path_and_fingerprint_are_live_and_exact(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    source_graph, target_graph = graphs
    authority = _body_route("C13").record
    common = {
        "source_graph": source_graph,
        "target_graph": target_graph,
        "source_graph_sha256": source_graph.fingerprint,
        "target_graph_sha256": target_graph.fingerprint,
        "layer_index": 0,
        "site": MLP_SITE,
        "axis_kind": AXIS_MLP_POST_SWIGLU,
        "source_parameter_paths": _paths(AXIS_MLP_POST_SWIGLU),
        "target_parameter_paths": _paths(AXIS_MLP_POST_SWIGLU),
        "source_width": 4,
        "target_width": 2,
        "authority_schema": authority["schema"],
        "authority_record_sha256": authority["record_sha256"],
    }
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="fingerprint"):
        lfm2_sparse_feature_axis_binding(**{**common, "source_graph_sha256": "f" * 64})
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="node kind"):
        lfm2_sparse_feature_axis_binding(**{**common, "site": CONV_SITE})
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="exactly one node"):
        lfm2_sparse_feature_axis_binding(**{**common, "site": "model.layers.0"})
    wrong_paths = _paths(AXIS_MLP_POST_SWIGLU)
    wrong_paths["w1"] = f"{MLP_SITE}.missing.weight"
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="exactly one"):
        lfm2_sparse_feature_axis_binding(
            **{**common, "source_parameter_paths": wrong_paths}
        )
    source_mlp = source_graph.nodes[0]
    unrelated = replace(source_mlp.parameters[0], role="unrelated_a")
    wrong_role_graph = replace(
        source_graph,
        nodes=(
            replace(
                source_mlp,
                parameters=(unrelated, *source_mlp.parameters[1:]),
            ),
            *source_graph.nodes[1:],
        ),
    )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="atomic component"):
        lfm2_sparse_feature_axis_binding(
            **{
                **common,
                "source_graph": wrong_role_graph,
                "source_graph_sha256": wrong_role_graph.fingerprint,
            }
        )
    decoy_graph = _with_mlp_path_decoys(source_graph)
    decoy_paths = {
        role: f"{MLP_SITE}.decoy.{role}.weight" for role in ("w1", "w3", "w2")
    }
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="atomic component"):
        lfm2_sparse_feature_axis_binding(
            **{
                **common,
                "source_graph": decoy_graph,
                "source_graph_sha256": decoy_graph.fingerprint,
                "source_parameter_paths": decoy_paths,
            }
        )


def test_short_conv_models_one_packed_input_partition_plus_fir_and_output(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    axis = _axis(AXIS_SHORT_CONV_CONTEXT, graphs)
    fold = _body_route("C05")

    packed_paths = {
        axis.source_parameter_paths[role]
        for role in ("in_proj_b", "in_proj_c", "in_proj_x")
    }
    assert packed_paths == {f"{CONV_SITE}.in_proj.weight"}
    assert dict(axis.report["source_parameter_row_slices"]) == {
        key: list(value) for key, value in _short_slices(4).items()
    }
    assert dict(axis.report["target_parameter_row_slices"]) == {
        key: list(value) for key, value in _short_slices(2).items()
    }
    assert {
        axis.report["operations_by_role"][role]
        for role in ("in_proj_b", "in_proj_c", "in_proj_x")
    } == {"fold_packed_producer_row_slice"}
    assert torch.equal(
        axis.feature_map.fold_consumer_weight(_parent_weight()),
        fold.child_weight,
    )


def test_short_conv_rejects_distinct_branch_paths_overlap_gap_and_bad_shape(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    source_graph, target_graph = graphs
    fold = _body_route("C05")
    witness = _source_witness(fold, source_graph, AXIS_SHORT_CONV_CONTEXT)
    common = {
        "source_graph": source_graph,
        "target_graph": target_graph,
        "source_graph_sha256": source_graph.fingerprint,
        "target_graph_sha256": target_graph.fingerprint,
        "layer_index": 0,
        "site": CONV_SITE,
        "axis_kind": AXIS_SHORT_CONV_CONTEXT,
        "source_parameter_paths": _paths(AXIS_SHORT_CONV_CONTEXT),
        "target_parameter_paths": _paths(AXIS_SHORT_CONV_CONTEXT),
        "source_parameter_row_slices": _short_slices(4),
        "target_parameter_row_slices": _short_slices(2),
        "body_route_fold": fold,
        "source_witness": witness,
        "coefficient_set": "write",
        "source_artifact_sha256s": _source_artifacts(witness),
    }
    distinct = _paths(AXIS_SHORT_CONV_CONTEXT)
    distinct["in_proj_c"] = f"{CONV_SITE}.in_proj_c.weight"
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="share one packed"):
        admit_lfm2_sparse_feature_axis(**{**common, "source_parameter_paths": distinct})
    overlap = _short_slices(4)
    overlap["in_proj_c"] = (3, 7)
    with pytest.raises(
        Lfm2SparseFeatureFoldBundleError, match="without gaps or overlap"
    ):
        admit_lfm2_sparse_feature_axis(
            **{**common, "source_parameter_row_slices": overlap}
        )
    malformed_target = _graph(
        width=2,
        model_name="malformed-child",
        malformed_packed=True,
    )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="geometry"):
        admit_lfm2_sparse_feature_axis(
            **{
                **common,
                "target_graph": malformed_target,
                "target_graph_sha256": malformed_target.fingerprint,
            }
        )
    wrong_fir_target = _graph(
        width=2,
        model_name="wrong-fir-child",
        fir_length=2,
    )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="geometry"):
        admit_lfm2_sparse_feature_axis(
            **{
                **common,
                "target_graph": wrong_fir_target,
                "target_graph_sha256": wrong_fir_target.fingerprint,
            }
        )
    mismatched_residual_target = _with_parameter_shape(
        target_graph,
        path=f"{CONV_SITE}.in_proj.weight",
        shape=(6, 5),
    )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="geometry"):
        admit_lfm2_sparse_feature_axis(
            **{
                **common,
                "target_graph": mismatched_residual_target,
                "target_graph_sha256": mismatched_residual_target.fingerprint,
            }
        )


def test_generic_residual_and_arbitrary_schema_are_fail_closed(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    source_graph, target_graph = graphs
    authority = _body_route("C13").record
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="not supported"):
        lfm2_sparse_feature_axis_binding(
            source_graph=source_graph,
            target_graph=target_graph,
            source_graph_sha256=source_graph.fingerprint,
            target_graph_sha256=target_graph.fingerprint,
            layer_index=0,
            site=ATTN_SITE,
            axis_kind=AXIS_RESIDUAL,
            source_parameter_paths=_paths(AXIS_RESIDUAL),
            target_parameter_paths=_paths(AXIS_RESIDUAL),
            source_width=4,
            target_width=2,
            authority_schema=authority["schema"],
            authority_record_sha256=authority["record_sha256"],
        )

    fake_body = {"schema": "self-rehashed-generic-residual-v1", "site": ATTN_SITE}
    fake = {**fake_body, "record_sha256": canonical_sha256(fake_body)}
    feature_map = SparseFeatureFoldMap(
        torch.tensor([0, 2]),
        torch.tensor([1, 3]),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1]),
        torch.tensor([0.25, 0.5], dtype=torch.float64),
        source_width=4,
        target_width=2,
    )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="generic residual"):
        admit_lfm2_sparse_feature_axis(
            source_graph=source_graph,
            target_graph=target_graph,
            source_graph_sha256=source_graph.fingerprint,
            target_graph_sha256=target_graph.fingerprint,
            layer_index=0,
            site=ATTN_SITE,
            axis_kind=AXIS_RESIDUAL,
            source_parameter_paths=_paths(AXIS_RESIDUAL),
            target_parameter_paths=_paths(AXIS_RESIDUAL),
            feature_map=feature_map,
            authority_record=fake,
        )


def test_attention_rejects_self_rehashed_mapping_and_even_live_internal_fold(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    source_graph, target_graph = graphs
    placeholder_axis = _axis(AXIS_MLP_POST_SWIGLU, graphs)
    source_topology = _topology(query_heads=2, kv_heads=1)
    target_topology = _topology(query_heads=1, kv_heads=1)
    fake_body = {
        "schema": "neural-forge-hf-causal-coupled-gqa-rope-fold-v1",
        "source_topology": source_topology,
        "target_topology": target_topology,
    }
    fake = {**fake_body, "report_sha256": canonical_sha256(fake_body)}
    common = {
        "source_graph": source_graph,
        "target_graph": target_graph,
        "source_graph_sha256": source_graph.fingerprint,
        "target_graph_sha256": target_graph.fingerprint,
        "layer_index": 0,
        "site": ATTN_SITE,
        "source_parameter_paths": _paths(AXIS_RESIDUAL),
        "target_parameter_paths": _paths(AXIS_RESIDUAL),
        "residual_axis": placeholder_axis,
        "source_topology": source_topology,
        "target_topology": target_topology,
    }
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="disabled"):
        admit_lfm2_attention_feature_site(
            **common,
            coupled_authority_report=fake,
        )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="disabled"):
        admit_lfm2_attention_feature_site(
            **common,
            coupled_attention_fold=_live_changed_attention_fold(),
        )


def test_bundle_is_deterministic_graph_bound_and_tamper_evident(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    source_graph, target_graph = graphs
    axes = (
        _axis(AXIS_MLP_POST_SWIGLU, graphs),
        _axis(AXIS_SHORT_CONV_CONTEXT, graphs),
    )
    kwargs = {
        "source_graph": source_graph,
        "target_graph": target_graph,
        "source_graph_sha256": source_graph.fingerprint,
        "target_graph_sha256": target_graph.fingerprint,
        "axes": axes,
        "source_artifact_sha256s": {"graph_pair": "4" * 64},
    }
    bundle = build_lfm2_sparse_feature_fold_bundle(**kwargs)
    repeated = build_lfm2_sparse_feature_fold_bundle(**kwargs)

    assert bundle.bundle_sha256 == repeated.bundle_sha256
    assert bundle.report["axis_count"] == 2
    assert bundle.report["attention_site_count"] == 0
    assert bundle.report["runtime_sidecar_count"] == 0
    assert bundle.report["unique_sparse_map_count"] == 2
    assert hashlib.sha256(
        json.dumps(bundle.as_dict(), sort_keys=True).encode()
    ).hexdigest()
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="fingerprint"):
        build_lfm2_sparse_feature_fold_bundle(
            **{**kwargs, "source_graph_sha256": "f" * 64}
        )

    tampered = dict(bundle.report)
    tampered["axis_count"] = 99
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="seal"):
        replace(bundle, report=tampered)

    axes[0].feature_map.coefficients[0] += 1.0
    with pytest.raises(Exception, match="authentication"):
        axes[0].verify()


def test_direct_axis_construction_cannot_reseal_forged_provenance(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    axis = _axis(AXIS_MLP_POST_SWIGLU, graphs)
    authority = axis.authority_record
    axis_binding = lfm2_sparse_feature_axis_binding(
        source_graph=axis.source_graph,
        target_graph=axis.target_graph,
        source_graph_sha256=axis.source_graph_sha256,
        target_graph_sha256=axis.target_graph_sha256,
        layer_index=axis.layer_index,
        site=axis.site,
        axis_kind=axis.axis_kind,
        source_parameter_paths=axis.source_parameter_paths,
        target_parameter_paths=axis.target_parameter_paths,
        source_parameter_row_slices=axis.source_parameter_row_slices,
        target_parameter_row_slices=axis.target_parameter_row_slices,
        source_width=axis.feature_map.source_width,
        target_width=axis.feature_map.target_width,
        authority_schema=authority["schema"],
        authority_record_sha256=authority["record_sha256"],
    )["lfm2_axis"]

    forged_body = {
        key: value
        for key, value in axis.provenance.items()
        if key != "provenance_sha256"
    }
    forged_body["compile_time_only"] = False
    forged_body["source_artifact_sha256s"] = {
        **forged_body["source_artifact_sha256s"],
        "unrelated": "d" * 64,
    }
    forged = {
        **forged_body,
        "provenance_sha256": canonical_sha256(forged_body),
    }
    report_body = axis._report_body(axis_binding, forged)
    admission_sha = canonical_sha256(report_body)
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="factory contract"):
        replace(
            axis,
            provenance=forged,
            report={**report_body, "admission_sha256": admission_sha},
            admission_sha256=admission_sha,
        )

    invalid_artifact_body = {
        **forged_body,
        "compile_time_only": True,
        "source_artifact_sha256s": {
            **axis.provenance["source_artifact_sha256s"],
            "invalid": "not-a-digest",
        },
    }
    invalid_artifact = {
        **invalid_artifact_body,
        "provenance_sha256": canonical_sha256(invalid_artifact_body),
    }
    invalid_report_body = axis._report_body(axis_binding, invalid_artifact)
    invalid_sha = canonical_sha256(invalid_report_body)
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="SHA-256"):
        replace(
            axis,
            provenance=invalid_artifact,
            report={**invalid_report_body, "admission_sha256": invalid_sha},
            admission_sha256=invalid_sha,
        )

    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="record_sha256"):
        replace(axis, authority_seal_field="wrong")


def test_direct_bundle_construction_cannot_reseal_forged_provenance(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    source_graph, target_graph = graphs
    axes = (
        _axis(AXIS_MLP_POST_SWIGLU, graphs),
        _axis(AXIS_SHORT_CONV_CONTEXT, graphs),
    )
    bundle = build_lfm2_sparse_feature_fold_bundle(
        source_graph=source_graph,
        target_graph=target_graph,
        source_graph_sha256=source_graph.fingerprint,
        target_graph_sha256=target_graph.fingerprint,
        axes=axes,
        source_artifact_sha256s={"graph_pair": "4" * 64},
    )
    forged_body = {
        "schema": bundle.provenance["schema"],
        "source_graph_sha256": "0" * 64,
        "target_graph_sha256": "1" * 64,
        "source_artifact_sha256s": dict(bundle.provenance["source_artifact_sha256s"]),
        "axis_admission_sha256s": [],
        "attention_admission_sha256s": [],
        "compile_time_only": False,
    }
    forged = {
        **forged_body,
        "provenance_sha256": canonical_sha256(forged_body),
    }
    report_body = Lfm2SparseFeatureFoldBundle._report_body(
        source_graph.fingerprint,
        target_graph.fingerprint,
        axes,
        (),
        forged,
    )
    bundle_sha = canonical_sha256(report_body)
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="provenance"):
        replace(
            bundle,
            provenance=forged,
            report={**report_body, "bundle_sha256": bundle_sha},
            bundle_sha256=bundle_sha,
        )


def test_bundle_rejects_two_live_sites_for_one_layer_semantic_axis(
    graphs: tuple[ModelCircuitGraph, ModelCircuitGraph],
) -> None:
    source_graph = _with_second_mlp(graphs[0], width=4)
    target_graph = _with_second_mlp(graphs[1], width=2)
    graph_pair = (source_graph, target_graph)
    first = _axis(AXIS_MLP_POST_SWIGLU, graph_pair)
    second_site = "model.layers.0.second_feed_forward"
    second_paths = _paths(AXIS_MLP_POST_SWIGLU, site=second_site)
    second_fold = _body_route("C13")
    second_witness = _source_witness(
        second_fold,
        source_graph,
        AXIS_MLP_POST_SWIGLU,
        graph_site=second_site,
    )
    second = admit_lfm2_sparse_feature_axis(
        source_graph=source_graph,
        target_graph=target_graph,
        source_graph_sha256=source_graph.fingerprint,
        target_graph_sha256=target_graph.fingerprint,
        layer_index=0,
        site=second_site,
        axis_kind=AXIS_MLP_POST_SWIGLU,
        source_parameter_paths=second_paths,
        target_parameter_paths=second_paths,
        body_route_fold=second_fold,
        source_witness=second_witness,
        coefficient_set="write",
        source_artifact_sha256s=_source_artifacts(second_witness),
    )
    with pytest.raises(Lfm2SparseFeatureFoldBundleError, match="semantic-axis"):
        build_lfm2_sparse_feature_fold_bundle(
            source_graph=source_graph,
            target_graph=target_graph,
            source_graph_sha256=source_graph.fingerprint,
            target_graph_sha256=target_graph.fingerprint,
            axes=(first, second),
        )
