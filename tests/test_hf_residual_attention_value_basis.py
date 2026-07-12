import hashlib
from dataclasses import replace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_residual_attention_value_basis as value_basis  # noqa: E402
from neural_forge.hf_attention_codebook import (  # noqa: E402
    AttentionCodebookRegistry,
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_cached_attention_trace import (  # noqa: E402
    HFCachedAttentionStep,
    HFCachedAttentionTrace,
)
from neural_forge.hf_residual_attention_value_basis import (  # noqa: E402
    ResidualAttentionValueBasisError,
    build_residual_value_basis_site_calibration,
    capture_parent_output_block_basis_entry,
    capture_value_prototype_basis_entry,
    fit_residual_targeted_value_basis,
    install_residual_targeted_value_basis,
)


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(
        contiguous.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _registries() -> tuple[AttentionCodebookRegistry, AttentionCodebookRegistry]:
    generator = torch.Generator().manual_seed(71011)
    layers = 3
    groups = 2
    slots = 2
    head_dim = 2
    q_rows = 2
    width = 2

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    q = random(groups, slots, q_rows, width)
    k = random(groups, slots, head_dim, width)
    v = random(groups, slots, head_dim, width)
    o = random(groups, slots, width, q_rows)
    v[0, 0] = torch.tensor([[1.0, 0.2], [-0.3, 0.8]], dtype=torch.float64)
    v[0, 1] = torch.tensor([[-0.5, 1.1], [0.9, 0.4]], dtype=torch.float64)
    o[0, 0] = torch.tensor(
        [[1.0, 0.7], [-0.4, 1.3]],
        dtype=torch.float64,
    )
    v_bias = random(groups, slots, head_dim)
    v_bias[0, 0] = torch.tensor([0.1, -0.2], dtype=torch.float64)
    v_bias[0, 1] = torch.tensor([-0.3, 0.25], dtype=torch.float64)
    permutations = torch.tensor(
        [[1, 0], [0, 1], [1, 0]],
        dtype=torch.int32,
    )
    sources = torch.tensor(
        [[1, 2], [0, 3]],
        dtype=torch.int32,
    )
    scales = torch.ones(layers, groups, 4, dtype=torch.float64)
    scales[0, 0, 2] = 2.0
    scales[0, 0, 3] = 0.5
    scales[1, 0, 2] = 0.5
    scales[1, 0, 3] = 1.5
    incumbent = AttentionCodebookRegistry(
        q,
        k,
        v,
        o,
        torch.tensor([[0, 0], [1, 1], [1, 0]], dtype=torch.int32),
        sources,
        permutations,
        prototype_v_bias=v_bias,
        role_scales=scales,
    )
    proposed = AttentionCodebookRegistry(
        q,
        k,
        v,
        o,
        torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.int32),
        sources,
        permutations,
        prototype_v_bias=v_bias,
        role_scales=scales,
    )
    return incumbent, proposed


def _manual_trace(
    *,
    sequence_id: str,
    layer_index: int,
    physical_group: int,
    input_chunks: tuple[torch.Tensor, ...],
    probabilities: tuple[torch.Tensor, ...],
    selected_contexts: tuple[torch.Tensor, ...],
    v_weight_sha256: str,
    v_bias_sha256: str | None,
) -> HFCachedAttentionTrace:
    steps = []
    history_rows = 0
    heads = 2
    head_dim = 2
    for index, (chunk, attention, selected_context) in enumerate(
        zip(input_chunks, probabilities, selected_contexts, strict=True)
    ):
        history_rows += int(chunk.shape[1])
        cache = torch.zeros(1, heads, history_rows, head_dim, dtype=torch.float64)
        selected_cache = cache[:, physical_group : physical_group + 1]
        delta = torch.zeros(1, 2, dtype=torch.float64)
        steps.append(
            HFCachedAttentionStep(
                step_index=index,
                prefix_length=history_rows,
                selected_token_id=7 + index,
                pre_attention_residual=torch.zeros_like(delta),
                value_projection_input=chunk,
                query=torch.zeros(1, 1, 1, head_dim, dtype=torch.float64),
                key=torch.zeros(1, 1, 1, head_dim, dtype=torch.float64),
                value=torch.zeros(1, 1, 1, head_dim, dtype=torch.float64),
                attention_scores=torch.log(attention),
                attention_probabilities=attention,
                full_attention_probabilities=attention.repeat(1, heads, 1, 1),
                selected_attention_context=selected_context,
                full_attention_context=torch.zeros(1, 4, dtype=torch.float64),
                attention_delta=delta,
                post_attention_residual=delta,
                layer_key_cache=torch.zeros_like(cache),
                layer_value_cache=cache,
                selected_key_cache=torch.zeros_like(selected_cache),
                selected_value_cache=selected_cache,
                action_logits=torch.zeros(1, 17, dtype=torch.float64),
                cached_full_prefix_max_abs_error=0.0,
                selected_context_max_abs_error=0.0,
                value_projection_cache_max_abs_error=0.0,
                captured_byte_count=1,
            )
        )
    horizon = len(steps)
    return HFCachedAttentionTrace(
        sequence_id=sequence_id,
        lane="forced",
        layer_index=layer_index,
        group_indices=(physical_group,),
        prompt_token_ids=(1, 2),
        consumed_token_ids=tuple(7 + index for index in range(horizon - 1)),
        selected_token_ids=tuple(7 + index for index in range(horizon)),
        steps=tuple(steps),
        query_heads=2,
        key_value_heads=2,
        query_heads_per_group=1,
        head_dim=head_dim,
        capture_bytes=1,
        maximum_cached_full_prefix_error=0.0,
        maximum_selected_context_error=0.0,
        maximum_value_projection_cache_error=0.0,
        value_projection_weight_sha256=v_weight_sha256,
        value_projection_bias_sha256=v_bias_sha256,
        transformers_version=None,
        torch_version=torch.__version__,
        model_class="synthetic-fixed-attention",
        attention_backend="eager",
        cache_class="synthetic",
        logit_rtol=1e-8,
        logit_atol=1e-8,
        context_rtol=1e-8,
        context_atol=1e-8,
        attention_source="synthetic-certified",
    )


def _calibrations(
    parent_registry: AttentionCodebookRegistry,
    registry: AttentionCodebookRegistry,
    basis,
    coefficients=(0.65, 0.35),
):
    return _calibrations_for_sites(
        registry,
        basis,
        ((0, 0), (1, 0)),
        coefficients,
        parent_registry,
    )


def _calibrations_for_sites(
    registry: AttentionCodebookRegistry,
    basis,
    sites,
    coefficients=(0.65, 0.35),
    parent_registry=None,
):
    if parent_registry is None:
        raise AssertionError("Tests require explicit parent registry provenance.")
    input_chunks = (
        torch.tensor([[[1.0, -0.4], [0.2, 0.8]]], dtype=torch.float64),
        torch.tensor([[[0.7, 1.1]]], dtype=torch.float64),
        torch.tensor([[[-0.6, 0.3]]], dtype=torch.float64),
    )
    probabilities = (
        torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float64),
        torch.tensor([[[[0.1, 0.35, 0.55]]]], dtype=torch.float64),
        torch.tensor([[[[0.4, 0.1, 0.2, 0.3]]]], dtype=torch.float64),
    )
    values = []
    for layer, canonical in sites:
        physical = int(registry.group_permutations[layer, canonical].item())
        v_weight, v_bias = registry.materialize_projection(layer, "v")
        o_weight, _o_bias = registry.materialize_projection(layer, "o")
        q_rows = registry.query_rows_per_group
        o_block = o_weight[:, physical * q_rows : (physical + 1) * q_rows]
        parent_o = capture_parent_output_block_basis_entry(
            parent_registry,
            layer_index=layer,
            canonical_group=canonical,
        )
        value_scale = float(registry.role_scales[layer, canonical, 2].item())
        history = []
        targets = []
        for chunk, attention in zip(input_chunks, probabilities, strict=True):
            history.append(chunk[0].numpy())
            inputs = np.concatenate(history, axis=0)
            design = []
            for entry in basis:
                projected = inputs @ entry.weight.T + entry.bias
                context = attention.numpy()[0, 0, 0] @ (projected * value_scale)
                design.append(context @ o_block.detach().numpy().T)
            targets.append(
                sum(
                    coefficient * contribution
                    for coefficient, contribution in zip(
                        coefficients, design, strict=True
                    )
                )
            )
        candidate = _manual_trace(
            sequence_id=f"site-{layer}",
            layer_index=layer,
            physical_group=physical,
            input_chunks=input_chunks,
            probabilities=probabilities,
            selected_contexts=tuple(
                torch.zeros(1, 2, dtype=torch.float64) for _ in targets
            ),
            v_weight_sha256=_tensor_sha256(v_weight),
            v_bias_sha256=None if v_bias is None else _tensor_sha256(v_bias),
        )
        parent = _manual_trace(
            sequence_id=f"site-{layer}",
            layer_index=layer,
            physical_group=physical,
            input_chunks=input_chunks,
            probabilities=probabilities,
            selected_contexts=tuple(
                torch.tensor(
                    np.linalg.solve(parent_o.block, target),
                    dtype=torch.float64,
                ).reshape(1, 2)
                for target in targets
            ),
            v_weight_sha256=f"parent-v-{layer}",
            v_bias_sha256=None,
        )
        values.append(
            build_residual_value_basis_site_calibration(
                candidate,
                parent,
                registry,
                basis,
                parent_o,
                canonical_group=canonical,
            )
        )
    return tuple(values)


def _ragged_registries():
    generator = torch.Generator().manual_seed(771011)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    incumbent_v = random(3, 2, 2)
    incumbent_v[0] = torch.tensor([[1.0, 0.2], [-0.3, 0.8]], dtype=torch.float64)
    incumbent_v[1] = torch.tensor([[-0.5, 1.1], [0.9, 0.4]], dtype=torch.float64)
    incumbent_bias = random(3, 2)
    incumbent_bias[0] = torch.tensor([0.1, -0.2], dtype=torch.float64)
    incumbent_bias[1] = torch.tensor([-0.3, 0.25], dtype=torch.float64)
    incumbent_o = random(3, 2, 2)
    incumbent_o[0] = torch.eye(2, dtype=torch.float64)
    permutations = torch.tensor([[1, 0], [1, 0]], dtype=torch.int32)
    scales = torch.ones(2, 2, 4, dtype=torch.float64)
    scales[0, 0, 2:] = torch.tensor([1.7, 0.6], dtype=torch.float64)
    scales[1, 1, 2:] = torch.tensor([0.4, 1.8], dtype=torch.float64)
    incumbent = RaggedAttentionCodebookRegistry(
        random(3, 2, 2),
        random(3, 2, 2),
        incumbent_v,
        incumbent_o,
        torch.tensor([[0, 2], [1, 0]], dtype=torch.int32),
        torch.tensor([1, 2, 0], dtype=torch.int32),
        permutations,
        prototype_v_bias=incumbent_bias,
        role_scales=scales,
    )
    proposed = RaggedAttentionCodebookRegistry(
        incumbent.prototype_q_weight.detach()[[0, 2]],
        incumbent.prototype_k_weight.detach()[[0, 2]],
        incumbent.prototype_v_weight.detach()[[0, 2]],
        incumbent.prototype_o_weight.detach()[[0, 2]],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
        permutations,
        prototype_v_bias=incumbent.prototype_v_bias.detach()[[0, 2]],
        role_scales=scales,
    )
    return incumbent, proposed


def _clone_state(registry):
    return {
        name: value.detach().clone() for name, value in registry.state_dict().items()
    }


def test_residual_targeted_basis_recovers_mixture_across_scaled_permuted_sites() -> (
    None
):
    incumbent, proposed = _registries()
    survivor = capture_value_prototype_basis_entry(
        incumbent,
        source_flat_index=1,
        basis_id="survivor-L0G1",
        role="survivor",
    )
    removed = capture_value_prototype_basis_entry(
        incumbent,
        source_flat_index=2,
        basis_id="removed-L1G0",
        role="removed-source",
    )
    basis = (survivor, removed)
    calibrations = _calibrations(incumbent, proposed, basis)
    for calibration in calibrations:
        np.testing.assert_allclose(
            calibration.target_contribution_rows,
            np.tensordot(
                calibration.basis_contribution_rows,
                np.array([0.65, 0.35]),
                axes=(1, 0),
            ),
            rtol=1e-12,
            atol=1e-12,
        )

    fit = fit_residual_targeted_value_basis(
        proposed,
        calibrations,
        basis,
        survivor_basis_id=survivor.basis_id,
        survivor_group=0,
        survivor_slot=0,
        lower_bounds={survivor.basis_id: 0.0, removed.basis_id: 0.0},
        upper_bounds={survivor.basis_id: 1.0, removed.basis_id: 1.0},
        ridge=1e-12,
    )

    assert fit.report.coefficients == pytest.approx((0.65, 0.35), abs=1e-8)
    assert fit.report.after_relative_rmse < 1e-9
    assert fit.report.after_relative_rmse < fit.report.before_relative_rmse
    assert fit.report.scalar_observation_count == 12
    assert fit.report.enumerated_active_sets == 9
    assert {
        (site.layer_index, site.canonical_group, site.physical_group)
        for site in fit.report.affected_sites
    } == {(0, 0, 1), (1, 0, 0)}
    assert {
        (site.value_role_scale, site.output_role_scale)
        for site in fit.report.affected_sites
    } == {(2.0, 0.5), (0.5, 1.5)}
    payload = fit.report.as_dict()
    assert payload["design_contract"].startswith("candidate-fixed-QK")
    assert payload["runtime_coefficients_retained"] is False
    assert payload["new_runtime_parameters"] == 0
    assert payload["required_next_step"].startswith("recollect actual child")
    assert fit.installable_fit.report.fitted_weight_sha256 == (
        fit.report.fitted_weight_sha256
    )
    assert all(parameter.grad is None for parameter in proposed.parameters())


def test_basis_install_uses_existing_rollback_handle_and_adds_no_parameters() -> None:
    incumbent, proposed = _registries()
    basis = (
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=1,
            basis_id="survivor",
            role="survivor",
        ),
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=2,
            basis_id="removed",
            role="removed-source",
        ),
    )
    fit = fit_residual_targeted_value_basis(
        proposed,
        _calibrations(incumbent, proposed, basis),
        basis,
        survivor_basis_id="survivor",
        survivor_group=0,
        survivor_slot=0,
        ridge=1e-12,
    )
    before = _clone_state(proposed)
    parameter_ids = tuple(id(value) for value in proposed.parameters())
    parameter_count = sum(value.numel() for value in proposed.parameters())

    installation = install_residual_targeted_value_basis(proposed, fit)

    assert installation.report.parameter_identity_preserved is True
    assert tuple(id(value) for value in proposed.parameters()) == parameter_ids
    assert sum(value.numel() for value in proposed.parameters()) == parameter_count
    assert not torch.equal(
        proposed.prototype_v_weight[0, 0], before["prototype_v_weight"][0, 0]
    )
    for name in (
        "prototype_q_weight",
        "prototype_k_weight",
        "prototype_o_weight",
        "assignments",
        "group_permutations",
        "role_scales",
    ):
        assert torch.equal(proposed.state_dict()[name], before[name])
    installation.rollback()
    for name, value in before.items():
        assert torch.equal(proposed.state_dict()[name], value)


def test_ragged_compacted_registry_uses_pre_removal_basis_across_canonical_groups() -> (
    None
):
    incumbent, proposed = _ragged_registries()
    survivor = capture_value_prototype_basis_entry(
        incumbent,
        source_flat_index=1,
        basis_id="survivor",
        role="survivor",
    )
    removed = capture_value_prototype_basis_entry(
        incumbent,
        source_flat_index=2,
        basis_id="removed-before-compaction",
        role="removed-source",
    )
    assert 2 not in proposed.prototype_source_indices.tolist()
    basis = (survivor, removed)
    calibrations = _calibrations_for_sites(
        proposed,
        basis,
        ((0, 0), (1, 1)),
        coefficients=(0.7, 0.3),
        parent_registry=incumbent,
    )

    fit = fit_residual_targeted_value_basis(
        proposed,
        calibrations,
        basis,
        survivor_basis_id="survivor",
        survivor_slot=0,
        ridge=1e-12,
    )

    assert fit.report.layout == "ragged"
    assert fit.report.survivor_group is None
    assert fit.report.coefficients == pytest.approx((0.7, 0.3), abs=1e-8)
    assert {
        (site.layer_index, site.canonical_group, site.physical_group)
        for site in fit.report.affected_sites
    } == {(0, 0, 1), (1, 1, 0)}
    assert fit.report.after_relative_rmse < 1e-9
    installation = install_residual_targeted_value_basis(proposed, fit)
    installation.rollback()


def test_global_fit_rejects_missing_survivor_site() -> None:
    incumbent, proposed = _registries()
    basis = (
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=1,
            basis_id="survivor",
            role="survivor",
        ),
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=2,
            basis_id="removed",
            role="removed-source",
        ),
    )
    calibrations = _calibrations(incumbent, proposed, basis)

    with pytest.raises(ResidualAttentionValueBasisError, match="every and only"):
        fit_residual_targeted_value_basis(
            proposed,
            calibrations[:1],
            basis,
            survivor_basis_id="survivor",
            survivor_group=0,
            survivor_slot=0,
        )

    forged = replace(
        calibrations[0],
        value_role_scale=calibrations[0].value_role_scale + 1.0,
    )
    with pytest.raises(ResidualAttentionValueBasisError, match="stale or forged"):
        fit_residual_targeted_value_basis(
            proposed,
            (forged, *calibrations[1:]),
            basis,
            survivor_basis_id="survivor",
            survivor_group=0,
            survivor_slot=0,
        )

    calibrations[0].basis_contribution_rows.setflags(write=True)
    calibrations[0].basis_contribution_rows[0, 0, 0] += 1.0
    with pytest.raises(ResidualAttentionValueBasisError, match="changed after capture"):
        fit_residual_targeted_value_basis(
            proposed,
            calibrations,
            basis,
            survivor_basis_id="survivor",
            survivor_group=0,
            survivor_slot=0,
        )


def test_install_fails_closed_if_qko_registry_state_changes_after_fit() -> None:
    incumbent, proposed = _registries()
    basis = (
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=1,
            basis_id="survivor",
            role="survivor",
        ),
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=2,
            basis_id="removed",
            role="removed-source",
        ),
    )
    fit = fit_residual_targeted_value_basis(
        proposed,
        _calibrations(incumbent, proposed, basis),
        basis,
        survivor_basis_id="survivor",
        survivor_group=0,
        survivor_slot=0,
    )
    before = proposed.prototype_o_weight.detach().clone()
    with torch.no_grad():
        proposed.prototype_o_weight[0, 0, 0, 0].add_(1.0)

    with pytest.raises(ResidualAttentionValueBasisError, match="changed after"):
        install_residual_targeted_value_basis(proposed, fit)

    with torch.no_grad():
        proposed.prototype_o_weight.copy_(before)
    installation = install_residual_targeted_value_basis(proposed, fit)
    installation.rollback()


def test_builder_rejects_greedy_or_stale_v_trace() -> None:
    incumbent, proposed = _registries()
    basis = (
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=1,
            basis_id="survivor",
            role="survivor",
        ),
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=2,
            basis_id="removed",
            role="removed-source",
        ),
    )
    calibration = _calibrations(incumbent, proposed, basis)[0]
    # The immutable calibration proves the real builder accepted a physical
    # group selected through a non-identity permutation.
    assert calibration.physical_group == 1

    input_chunks = (torch.ones(1, 2, 2, dtype=torch.float64),)
    probabilities = (torch.tensor([[[[0.5, 0.5]]]], dtype=torch.float64),)
    selected = (torch.zeros(1, 2, dtype=torch.float64),)
    child = _manual_trace(
        sequence_id="invalid",
        layer_index=0,
        physical_group=1,
        input_chunks=input_chunks,
        probabilities=probabilities,
        selected_contexts=selected,
        v_weight_sha256="stale",
        v_bias_sha256=None,
    )
    parent = replace(child, value_projection_weight_sha256="parent")
    parent_o = capture_parent_output_block_basis_entry(
        incumbent,
        layer_index=0,
        canonical_group=0,
    )
    with pytest.raises(ResidualAttentionValueBasisError, match="V hashes"):
        build_residual_value_basis_site_calibration(
            child,
            parent,
            proposed,
            basis,
            parent_o,
            canonical_group=0,
        )
    with pytest.raises(ResidualAttentionValueBasisError, match="forced lane"):
        build_residual_value_basis_site_calibration(
            replace(child, lane="greedy"),
            replace(parent, lane="greedy"),
            proposed,
            basis,
            parent_o,
            canonical_group=0,
        )


def test_basis_and_parent_o_capture_tamper_or_foreign_registry_fail_closed() -> None:
    incumbent, proposed = _registries()
    survivor = capture_value_prototype_basis_entry(
        incumbent,
        source_flat_index=1,
        basis_id="survivor",
        role="survivor",
    )
    removed = capture_value_prototype_basis_entry(
        incumbent,
        source_flat_index=2,
        basis_id="removed",
        role="removed-source",
    )
    basis = (survivor, removed)

    with pytest.raises(ResidualAttentionValueBasisError, match="one incumbent"):
        _calibrations(proposed, proposed, basis)

    removed.weight.setflags(write=True)
    removed.weight[0, 0] += 1.0
    with pytest.raises(ResidualAttentionValueBasisError, match="changed after capture"):
        _calibrations(incumbent, proposed, basis)

    parent_o = capture_parent_output_block_basis_entry(
        incumbent,
        layer_index=0,
        canonical_group=0,
    )
    assert parent_o.source_site == (0, 0, 1)
    parent_o.block.setflags(write=True)
    parent_o.block[0, 0] += 1.0
    with pytest.raises(ResidualAttentionValueBasisError, match="changed after capture"):
        parent_o.assert_untampered()


def test_installable_shared_report_is_exactly_bound_to_basis_fit() -> None:
    incumbent, proposed = _registries()
    basis = (
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=1,
            basis_id="survivor",
            role="survivor",
        ),
        capture_value_prototype_basis_entry(
            incumbent,
            source_flat_index=2,
            basis_id="removed",
            role="removed-source",
        ),
    )
    fit = fit_residual_targeted_value_basis(
        proposed,
        _calibrations(incumbent, proposed, basis),
        basis,
        survivor_basis_id="survivor",
        survivor_group=0,
        survivor_slot=0,
    )
    wrong_report = replace(fit.installable_fit.report, survivor_slot=1)
    wrong_installable = replace(fit.installable_fit, report=wrong_report)

    with pytest.raises(ResidualAttentionValueBasisError, match="exactly bound"):
        replace(fit, installable_fit=wrong_installable)

    wrong_coefficients = replace(fit.report, coefficients=(1.0, 0.0))
    with pytest.raises(ResidualAttentionValueBasisError, match="do not materialize"):
        replace(fit, report=wrong_coefficients)


def test_box_anchored_ridge_finds_boundary_optimum() -> None:
    coefficients, _ridge, _condition, active_set, enumerated = (
        value_basis._solve_box_anchored_ridge(
            np.eye(2, dtype=np.float64),
            np.array([2.0, -1.0], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            ridge=1e-12,
            max_active_sets=9,
        )
    )

    assert coefficients == pytest.approx((1.0, 0.0), abs=1e-10)
    assert active_set == ("upper", "lower")
    assert enumerated == 9


def test_bfloat16_basis_and_parent_o_capture_round_trip_source_hashes() -> None:
    incumbent, _proposed = _registries()
    incumbent = incumbent.to(dtype=torch.bfloat16)

    basis = capture_value_prototype_basis_entry(
        incumbent,
        source_flat_index=1,
        basis_id="bf16-survivor",
        role="survivor",
    )
    parent_o = capture_parent_output_block_basis_entry(
        incumbent,
        layer_index=0,
        canonical_group=0,
    )

    assert basis.source_dtype == "torch.bfloat16"
    assert parent_o.source_dtype == "torch.bfloat16"
    basis.assert_untampered()
    parent_o.assert_untampered()
