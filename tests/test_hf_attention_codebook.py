from __future__ import annotations

import gc

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_attention_codebook import (  # noqa: E402
    AttentionCodebookLayerView,
    AttentionCodebookRegistry,
    AttentionCodebookShapeError,
    GQAAttentionBundle,
    GlobalAttentionCodebook,
    RaggedAttentionCodebookRegistry,
    attention_codebook_registry_from_state_dict,
    fit_global_attention_codebook,
)


QUERY_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 2
QUERY_ROWS_PER_GROUP = (QUERY_HEADS // KV_HEADS) * HEAD_DIM


def _attention_bundles(
    *,
    layers: int = 3,
    input_width: int = 6,
    output_width: int = 6,
    bias: bool = True,
    seed: int = 100,
):
    generator = torch.Generator().manual_seed(seed)
    bundles = []
    linears = []
    for _ in range(layers):
        q_proj = torch.nn.Linear(
            input_width,
            QUERY_HEADS * HEAD_DIM,
            bias=bias,
            dtype=torch.float64,
        )
        k_proj = torch.nn.Linear(
            input_width,
            KV_HEADS * HEAD_DIM,
            bias=bias,
            dtype=torch.float64,
        )
        v_proj = torch.nn.Linear(
            input_width,
            KV_HEADS * HEAD_DIM,
            bias=bias,
            dtype=torch.float64,
        )
        o_proj = torch.nn.Linear(
            QUERY_HEADS * HEAD_DIM,
            output_width,
            bias=bias,
            dtype=torch.float64,
        )
        with torch.no_grad():
            for projection in (q_proj, k_proj, v_proj, o_proj):
                for parameter in projection.parameters():
                    parameter.copy_(
                        torch.randn(
                            parameter.shape,
                            generator=generator,
                            dtype=parameter.dtype,
                        )
                    )
        bundles.append(
            GQAAttentionBundle.from_linears(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                query_heads=QUERY_HEADS,
                key_value_heads=KV_HEADS,
                head_dim=HEAD_DIM,
            )
        )
        linears.append((q_proj, k_proj, v_proj, o_proj))
    return bundles, linears


def _bundle_tensors(bundle: GQAAttentionBundle):
    return {
        name: value
        for name in (
            "q_weight",
            "k_weight",
            "v_weight",
            "o_weight",
            "q_bias",
            "k_bias",
            "v_bias",
            "o_bias",
        )
        if (value := getattr(bundle, name)) is not None
    }


def _assert_bundle_equal(
    actual: GQAAttentionBundle,
    expected: GQAAttentionBundle,
) -> None:
    assert _bundle_tensors(actual).keys() == _bundle_tensors(expected).keys()
    for name, actual_tensor in _bundle_tensors(actual).items():
        torch.testing.assert_close(
            actual_tensor,
            _bundle_tensors(expected)[name],
            atol=0.0,
            rtol=0.0,
        )


def _group_blocks(bundle: GQAAttentionBundle, group: int):
    return {
        "q_weight": bundle.q_weight.reshape(KV_HEADS, QUERY_ROWS_PER_GROUP, -1)[group],
        "k_weight": bundle.k_weight.reshape(KV_HEADS, HEAD_DIM, -1)[group],
        "v_weight": bundle.v_weight.reshape(KV_HEADS, HEAD_DIM, -1)[group],
        "o_weight": bundle.o_weight.reshape(
            bundle.o_weight.shape[0],
            KV_HEADS,
            QUERY_ROWS_PER_GROUP,
        )[:, group],
        "q_bias": (
            bundle.q_bias.reshape(KV_HEADS, QUERY_ROWS_PER_GROUP)[group]
            if bundle.q_bias is not None
            else None
        ),
        "k_bias": (
            bundle.k_bias.reshape(KV_HEADS, HEAD_DIM)[group]
            if bundle.k_bias is not None
            else None
        ),
        "v_bias": (
            bundle.v_bias.reshape(KV_HEADS, HEAD_DIM)[group]
            if bundle.v_bias is not None
            else None
        ),
    }


def _permute_groups(
    bundle: GQAAttentionBundle,
    physical_order,
) -> GQAAttentionBundle:
    order = torch.as_tensor(physical_order, dtype=torch.long)
    q_weight = bundle.q_weight.reshape(KV_HEADS, QUERY_ROWS_PER_GROUP, -1).index_select(
        0, order
    )
    k_weight = bundle.k_weight.reshape(KV_HEADS, HEAD_DIM, -1).index_select(0, order)
    v_weight = bundle.v_weight.reshape(KV_HEADS, HEAD_DIM, -1).index_select(0, order)
    o_weight = bundle.o_weight.reshape(
        bundle.o_weight.shape[0],
        KV_HEADS,
        QUERY_ROWS_PER_GROUP,
    ).index_select(1, order)

    def permute_bias(value, width):
        if value is None:
            return None
        return value.reshape(KV_HEADS, width).index_select(0, order).reshape(-1)

    return GQAAttentionBundle(
        q_weight=q_weight.reshape_as(bundle.q_weight),
        k_weight=k_weight.reshape_as(bundle.k_weight),
        v_weight=v_weight.reshape_as(bundle.v_weight),
        o_weight=o_weight.reshape_as(bundle.o_weight),
        q_bias=permute_bias(bundle.q_bias, QUERY_ROWS_PER_GROUP),
        k_bias=permute_bias(bundle.k_bias, HEAD_DIM),
        v_bias=permute_bias(bundle.v_bias, HEAD_DIM),
        o_bias=bundle.o_bias,
        query_heads=QUERY_HEADS,
        key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )


@pytest.mark.parametrize("bias", [False, True])
def test_full_prototype_codebook_is_exact_and_preserves_all_heads(bias) -> None:
    bundles, _ = _attention_bundles(layers=3, bias=bias, seed=110)
    logical_groups = len(bundles) * KV_HEADS
    codebook = fit_global_attention_codebook(
        bundles,
        prototype_count=logical_groups,
        sketch_dim=11,
    )
    inputs = torch.randn(2, 5, 6, dtype=torch.float64)
    contexts = torch.randn(
        2,
        5,
        QUERY_HEADS * HEAD_DIM,
        dtype=torch.float64,
    )

    for layer_index, source in enumerate(bundles):
        reconstructed = codebook[layer_index].materialize()
        _assert_bundle_equal(reconstructed, source)
        expected_qkv = source.project_qkv(inputs)
        actual_qkv = codebook[layer_index](inputs)
        for actual, expected in zip(actual_qkv, expected_qkv, strict=True):
            torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            codebook[layer_index].o_proj(contexts),
            source.project_output(contexts),
            atol=0.0,
            rtol=0.0,
        )

    expected_assignments = torch.arange(3, dtype=torch.int32).repeat(KV_HEADS, 1).t()
    expected_sources = torch.tensor(
        [[0, 2, 4], [1, 3, 5]],
        dtype=torch.int32,
    )
    assert torch.equal(codebook.registry.assignments, expected_assignments)
    assert torch.equal(
        codebook.registry.prototype_source_indices,
        expected_sources,
    )
    assert codebook.report.logical_group_executions == logical_groups
    assert codebook.report.logical_query_head_executions == 3 * QUERY_HEADS
    assert (
        codebook.report.unique_learned_parameters
        == codebook.report.logical_dense_parameters
    )
    assert codebook.report.schema_metadata_bytes == 8
    assert codebook.report.index_buffer_bytes == 3 * logical_groups * 4 + 8
    assert codebook.diagnostics is not None
    assert codebook.diagnostics.distance_feature_operations == 0
    assert all(isinstance(view, AttentionCodebookLayerView) for view in codebook.layers)
    assert all(list(view.parameters()) == [] for view in codebook.layers)
    assert all(view.registry is codebook.registry for view in codebook.layers)
    assert all(name.startswith("registry.") for name, _ in codebook.named_parameters())


def test_real_medoid_gqa_groups_keep_q_k_v_and_o_slices_coupled() -> None:
    bundles, _ = _attention_bundles(layers=4, seed=120)
    codebook = fit_global_attention_codebook(
        bundles,
        prototypes_per_group=1,
        sketch_dim=13,
    )
    registry = codebook.registry

    for group in range(KV_HEADS):
        source_index = int(registry.prototype_source_indices[group, 0])
        layer_index, physical_group = divmod(source_index, KV_HEADS)
        source = _group_blocks(bundles[layer_index], physical_group)
        torch.testing.assert_close(
            registry.prototype_q_weight[group, 0],
            source["q_weight"],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            registry.prototype_k_weight[group, 0],
            source["k_weight"],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            registry.prototype_v_weight[group, 0],
            source["v_weight"],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            registry.prototype_o_weight[group, 0],
            source["o_weight"],
            atol=0.0,
            rtol=0.0,
        )

    for layer_index in range(registry.logical_layers):
        reconstructed = codebook[layer_index].materialize()
        for group in range(KV_HEADS):
            slot = int(registry.assignments[layer_index, group])
            actual = _group_blocks(reconstructed, group)
            for role in ("q", "k", "v"):
                torch.testing.assert_close(
                    actual[f"{role}_weight"],
                    getattr(registry, f"prototype_{role}_weight")[group, slot],
                    atol=0.0,
                    rtol=0.0,
                )
            torch.testing.assert_close(
                actual["o_weight"],
                registry.prototype_o_weight[group, slot],
                atol=0.0,
                rtol=0.0,
            )


def test_compressed_scaled_projection_views_are_finite_and_count_honestly() -> None:
    bundles, source_linears = _attention_bundles(layers=4, seed=130)
    codebook = fit_global_attention_codebook(
        bundles,
        prototypes_per_group=1,
        use_scales=True,
        sketch_dim=9,
    )
    inputs = torch.randn(3, 6, dtype=torch.float64)
    contexts = torch.randn(
        3,
        QUERY_HEADS * HEAD_DIM,
        dtype=torch.float64,
    )

    for view in codebook.layers:
        q_result, k_result, v_result = view(inputs)
        assert q_result.shape == (3, QUERY_HEADS * HEAD_DIM)
        assert k_result.shape == (3, KV_HEADS * HEAD_DIM)
        assert v_result.shape == (3, KV_HEADS * HEAD_DIM)
        assert view.o_proj(contexts).shape == (3, 6)
        assert all(
            bool(torch.isfinite(value).all())
            for value in (q_result, k_result, v_result, view.o_proj(contexts))
        )
        assert not view.state_dict()

    report = codebook.report
    assert report.prototype_count == KV_HEADS
    assert report.scale_parameters == 4 * KV_HEADS * 4
    assert report.unique_learned_parameters == sum(
        parameter.numel() for parameter in codebook.parameters()
    )
    assert report.unique_learned_parameters < report.logical_dense_parameters
    assert report.index_buffer_bytes == (
        report.assignment_index_bytes
        + report.prototype_source_index_bytes
        + report.permutation_index_bytes
        + report.schema_metadata_bytes
    )
    assert all(
        parameter.grad is None
        for layer in source_linears
        for projection in layer
        for parameter in projection.parameters()
    )


def test_supplied_bijective_group_permutations_are_deterministically_equivalent() -> (
    None
):
    bundles, _ = _attention_bundles(layers=4, seed=140)
    physical_orders = ([0, 1], [1, 0], [1, 0], [0, 1])
    permuted = [
        _permute_groups(bundle, order)
        for bundle, order in zip(bundles, physical_orders, strict=True)
    ]
    canonical_to_physical = [
        torch.argsort(torch.tensor(order)).tolist() for order in physical_orders
    ]
    kwargs = {"prototypes_per_group": 2, "sketch_dim": 12}

    baseline = fit_global_attention_codebook(bundles, **kwargs)
    aligned = fit_global_attention_codebook(
        permuted,
        group_permutations=canonical_to_physical,
        **kwargs,
    )
    repeated = fit_global_attention_codebook(
        permuted,
        group_permutations=canonical_to_physical,
        **kwargs,
    )

    assert torch.equal(
        baseline.registry.assignments,
        aligned.registry.assignments,
    )
    for name in (
        "prototype_q_weight",
        "prototype_k_weight",
        "prototype_v_weight",
        "prototype_o_weight",
        "prototype_q_bias",
        "prototype_k_bias",
        "prototype_v_bias",
    ):
        assert torch.equal(
            getattr(baseline.registry, name),
            getattr(aligned.registry, name),
        )
    for layer_index, order in enumerate(physical_orders):
        _assert_bundle_equal(
            aligned[layer_index].materialize(),
            _permute_groups(
                baseline[layer_index].materialize(),
                order,
            ),
        )
    assert aligned.diagnostics is not None
    assert aligned.diagnostics.used_group_permutations is True
    assert aligned.diagnostics.permutations_provided is True
    assert aligned.diagnostics.permutations_bijective is True
    assert aligned.state_dict().keys() == repeated.state_dict().keys()
    for key, value in aligned.state_dict().items():
        assert torch.equal(value, repeated.state_dict()[key])


def test_aligned_medoid_fitting_has_the_claimed_depth_quadratic_work_bound() -> None:
    layers = 5
    bundles, _ = _attention_bundles(layers=layers, seed=150)
    codebook = fit_global_attention_codebook(
        bundles,
        prototypes_per_group=2,
        sketch_dim=7,
    )
    diagnostics = codebook.diagnostics

    assert diagnostics is not None
    assert diagnostics.deterministic is True
    assert diagnostics.real_coupled_group_medoids is True
    assert diagnostics.bounded_work is True
    assert diagnostics.fitting_requires_grad is False
    assert diagnostics.materialized_global_pairwise_matrix is False
    assert diagnostics.pairwise_distance_values == KV_HEADS * layers * layers
    assert diagnostics.distance_values_computed == KV_HEADS * layers * layers
    assert diagnostics.exact_distance_work_bound == KV_HEADS * layers * layers
    assert diagnostics.exact_distance_coordinate_work_bound == (
        KV_HEADS * layers * layers * diagnostics.sketch_dimensions
    )
    assert diagnostics.maximum_pairwise_block_elements == layers * layers
    assert diagnostics.distance_feature_operations == (
        KV_HEADS * layers * layers * diagnostics.sketch_dimensions
    )
    assert diagnostics.distance_feature_operations <= KV_HEADS * layers * layers * 7
    assert diagnostics.distance_feature_operations <= diagnostics.distance_work_bound
    assert diagnostics.sketch_elements == (
        layers * KV_HEADS * diagnostics.sketch_dimensions
    )


@pytest.mark.parametrize(
    "permutations",
    [
        torch.tensor([[0.0, 1.0]] * 3),
        [[0, 1], [0, 1], [0, False]],
        [[0, 1], [0, 0], [0, 1]],
        [[0, 1], [0, 2], [0, 1]],
    ],
    ids=("floating", "boolean", "duplicate", "out-of-range"),
)
def test_group_permutations_fail_closed_unless_every_row_is_bijective(
    permutations,
) -> None:
    bundles, _ = _attention_bundles(layers=3, seed=155)

    with pytest.raises(AttentionCodebookShapeError, match="group_permutations"):
        fit_global_attention_codebook(
            bundles,
            prototypes_per_group=1,
            group_permutations=permutations,
        )


def test_all_reconstructed_projection_roles_receive_finite_gradients() -> None:
    bundles, source_linears = _attention_bundles(layers=3, seed=160)
    codebook = fit_global_attention_codebook(
        bundles,
        prototypes_per_group=1,
        use_scales=True,
        sketch_dim=10,
    )
    inputs = torch.randn(
        2,
        4,
        6,
        dtype=torch.float64,
        requires_grad=True,
    )
    contexts = torch.randn(
        2,
        4,
        QUERY_HEADS * HEAD_DIM,
        dtype=torch.float64,
        requires_grad=True,
    )

    loss = torch.zeros((), dtype=torch.float64)
    for view in codebook.layers:
        loss = loss + sum(value.square().mean() for value in view(inputs))
        loss = loss + view.o_proj(contexts).square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert contexts.grad is not None
    assert bool(torch.isfinite(inputs.grad).all())
    assert bool(torch.isfinite(contexts.grad).all())
    assert all(parameter.grad is not None for parameter in codebook.parameters())
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in codebook.parameters()
    )
    assert all(
        parameter.grad is None
        for layer in source_linears
        for projection in layer
        for parameter in projection.parameters()
    )


def test_tensor_only_state_roundtrip_is_exact_and_views_keep_registry_alive() -> None:
    bundles, _ = _attention_bundles(layers=3, seed=170)
    supplied_scales = torch.linspace(
        0.75,
        1.25,
        steps=3 * KV_HEADS * 4,
        dtype=torch.float64,
    ).reshape(3, KV_HEADS, 4)
    codebook = fit_global_attention_codebook(
        bundles,
        prototypes_per_group=1,
        role_scales=supplied_scales,
        sketch_dim=8,
    )
    state = {
        key: value.detach().clone() for key, value in codebook.state_dict().items()
    }

    assert state
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert not any("extra_state" in key for key in state)
    restored = GlobalAttentionCodebook.from_state_dict(state)
    direct_registry = AttentionCodebookRegistry.from_state_dict(
        restored.registry.state_dict()
    )
    inputs = torch.randn(4, 6, dtype=torch.float64)
    contexts = torch.randn(
        4,
        QUERY_HEADS * HEAD_DIM,
        dtype=torch.float64,
    )

    assert restored.report == codebook.report
    assert restored.state_dict().keys() == state.keys()
    for key, value in state.items():
        assert torch.equal(restored.state_dict()[key], value)
    for layer_index in range(3):
        _assert_bundle_equal(
            restored[layer_index].materialize(),
            codebook[layer_index].materialize(),
        )
        for actual, expected in zip(
            direct_registry(inputs, layer_index),
            codebook[layer_index](inputs),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            restored[layer_index].o_proj(contexts),
            codebook[layer_index].o_proj(contexts),
            atol=0.0,
            rtol=0.0,
        )
        assert not restored[layer_index].state_dict()

    view = restored[1]
    registry = restored.registry
    del direct_registry
    del restored
    gc.collect()
    assert view.registry is registry
    assert all(bool(torch.isfinite(value).all()) for value in view(inputs))
    assert bool(torch.isfinite(view.o_proj(contexts)).all())


def test_ragged_registry_layout_is_explicit_and_publicly_reloadable() -> None:
    bundles, _ = _attention_bundles(layers=3, seed=177)
    dense = fit_global_attention_codebook(
        bundles,
        prototypes_per_group=3,
        use_scales=True,
    ).registry

    def flattened(name: str):
        value = getattr(dense, name)
        return None if value is None else value.detach().flatten(0, 1)

    groups, slots = dense.prototype_q_weight.shape[:2]
    ragged = RaggedAttentionCodebookRegistry(
        flattened("prototype_q_weight"),
        flattened("prototype_k_weight"),
        flattened("prototype_v_weight"),
        flattened("prototype_o_weight"),
        dense.assignments + torch.arange(groups, dtype=torch.int32) * slots,
        dense.prototype_source_indices.flatten(),
        dense.group_permutations,
        prototype_q_bias=flattened("prototype_q_bias"),
        prototype_k_bias=flattened("prototype_k_bias"),
        prototype_v_bias=flattened("prototype_v_bias"),
        layer_o_bias=dense.layer_o_bias,
        role_scales=dense.role_scales,
    )
    state = {name: value.detach().clone() for name, value in ragged.state_dict().items()}
    restored = attention_codebook_registry_from_state_dict(state)

    assert isinstance(restored, RaggedAttentionCodebookRegistry)
    assert int(restored._registry_layout) == 1
    assert int(restored._registry_schema_version) == 1
    assert restored.report == ragged.report
    assert restored.state_dict().keys() == state.keys()
    for name, value in state.items():
        assert torch.equal(restored.state_dict()[name], value)
    for layer in range(3):
        _assert_bundle_equal(restored.materialize_layer(layer), ragged.materialize_layer(layer))
