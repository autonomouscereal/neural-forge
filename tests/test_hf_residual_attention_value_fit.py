from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_attention_codebook import (  # noqa: E402
    AttentionCodebookRegistry,
)
from neural_forge.hf_cached_attention_trace import (  # noqa: E402
    HFCachedAttentionStep,
    HFCachedAttentionTrace,
)
from neural_forge.hf_residual_attention_value_fit import (  # noqa: E402
    ResidualAttentionValueFitError,
    ValueSiteCalibration,
    build_value_site_calibration,
    fit_shared_codebook_value_prototype,
    install_shared_codebook_value_prototype,
    refit_output_after_value_installation,
)


def _registry() -> AttentionCodebookRegistry:
    generator = torch.Generator().manual_seed(4099)
    groups = 2
    slots = 2
    head_dim = 2
    width = 3
    q_rows = 2
    layers = 3

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    role_scales = torch.ones(layers, groups, 4, dtype=torch.float64)
    role_scales[0, 0, 2] = 2.0
    role_scales[1, 0, 2] = 0.5
    role_scales[2, 0, 2] = 1.25
    return AttentionCodebookRegistry(
        random(groups, slots, q_rows, width),
        random(groups, slots, head_dim, width),
        random(groups, slots, head_dim, width),
        random(groups, slots, width, q_rows),
        torch.tensor([[0, 0], [0, 1], [1, 1]], dtype=torch.int32),
        torch.tensor([[1, 5], [0, 3]], dtype=torch.int32),
        torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int32),
        prototype_v_bias=random(groups, slots, head_dim),
        role_scales=role_scales,
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous().reshape(-1)
    return hashlib.sha256(
        contiguous.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _calibrations(
    registry: AttentionCodebookRegistry,
) -> tuple[ValueSiteCalibration, ...]:
    generator = np.random.default_rng(901)
    true_weight = np.array([[0.7, -0.4, 0.2], [-0.3, 0.6, 0.9]])
    true_bias = np.array([0.15, -0.25])
    values = []
    for layer in (0, 1):
        inputs = generator.normal(size=(12, 3))
        scale = float(registry.role_scales[layer, 0, 2].item())
        targets = scale * (inputs @ true_weight.T + true_bias)
        materialized_weight, materialized_bias = registry.materialize_projection(
            layer, "v"
        )
        values.append(
            ValueSiteCalibration(
                layer_index=layer,
                canonical_group=0,
                physical_group=int(registry.group_permutations[layer, 0].item()),
                input_rows=inputs,
                target_value_rows=targets,
                sequence_id=f"sequence-{layer}",
                child_trace_sha256=f"child-{layer}",
                parent_trace_sha256=f"parent-{layer}",
                child_v_weight_sha256=_tensor_sha256(materialized_weight),
                child_v_bias_sha256=(
                    None
                    if materialized_bias is None
                    else _tensor_sha256(materialized_bias)
                ),
            )
        )
    return tuple(values)


def _clone_state(registry: AttentionCodebookRegistry) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone() for name, value in registry.state_dict().items()
    }


def test_global_scaled_v_fit_installs_only_survivor_and_rolls_back_exactly() -> None:
    registry = _registry()
    before = _clone_state(registry)
    parameter_ids = tuple(id(parameter) for parameter in registry.parameters())
    fit = fit_shared_codebook_value_prototype(
        registry,
        _calibrations(registry),
        survivor_group=0,
        survivor_slot=0,
        ridge=1e-10,
    )

    assert fit.report.after_relative_rmse < 1e-8
    assert fit.report.after_relative_rmse < fit.report.before_relative_rmse
    assert {
        (site.layer_index, site.canonical_group) for site in fit.report.affected_sites
    } == {
        (0, 0),
        (1, 0),
    }
    assert {site.physical_group for site in fit.report.affected_sites} == {0, 1}
    assert {site.role_scale for site in fit.report.affected_sites} == {0.5, 2.0}
    assert fit.report.as_dict()["global_all_affected_site_fit"] is True
    assert fit.report.as_dict()["schema"].endswith("-v2")
    assert fit.report.as_dict()["experimental_not_compaction_proof"] is True
    assert [
        layer for layer, _weight, _bias in fit.report.calibration_v_sha256_by_layer
    ] == [0, 1]
    assert (
        fit.report.as_dict()["runtime_contract"][
            "external_full_runtime_snapshot_binding_required"
        ]
        is True
    )

    installation = install_shared_codebook_value_prototype(registry, fit)
    installed = installation.report
    assert installation.assert_live() is installed
    assert installed.exact_unaffected_v_identity is True
    assert installed.exact_non_v_identity is True
    assert installed.parameter_identity_preserved is True
    assert tuple(id(parameter) for parameter in registry.parameters()) == parameter_ids
    assert not torch.equal(
        registry.prototype_v_weight[0, 0], before["prototype_v_weight"][0, 0]
    )
    assert torch.equal(
        registry.prototype_v_weight[0, 1], before["prototype_v_weight"][0, 1]
    )
    assert torch.equal(registry.prototype_v_weight[1], before["prototype_v_weight"][1])
    for name in (
        "prototype_q_weight",
        "prototype_k_weight",
        "prototype_o_weight",
        "assignments",
        "group_permutations",
        "role_scales",
    ):
        assert torch.equal(registry.state_dict()[name], before[name])
    assert all(parameter.grad is None for parameter in registry.parameters())

    installation.rollback()
    assert installation.is_applied is False
    for name, value in before.items():
        assert torch.equal(registry.state_dict()[name], value)


def test_install_rejects_fit_payload_mutated_after_report_creation() -> None:
    registry = _registry()
    before = _clone_state(registry)
    fit = fit_shared_codebook_value_prototype(
        registry,
        _calibrations(registry),
        survivor_group=0,
        survivor_slot=0,
        ridge=1e-10,
    )
    fit.weight.setflags(write=True)
    fit.weight[...] = 777.0

    with pytest.raises(ResidualAttentionValueFitError, match="payload changed"):
        install_shared_codebook_value_prototype(registry, fit)

    for name, value in before.items():
        assert torch.equal(registry.state_dict()[name], value)


def test_install_rejects_materialized_v_changed_after_calibration_fit() -> None:
    registry = _registry()
    fit = fit_shared_codebook_value_prototype(
        registry,
        _calibrations(registry),
        survivor_group=0,
        survivor_slot=0,
        ridge=1e-10,
    )
    with torch.no_grad():
        registry.prototype_v_weight[1, 0, 0, 0].add_(1.0)

    with pytest.raises(ResidualAttentionValueFitError, match="changed after"):
        install_shared_codebook_value_prototype(registry, fit)


def test_rollback_preflights_unrelated_state_and_remains_retryable() -> None:
    registry = _registry()
    before = _clone_state(registry)
    fit = fit_shared_codebook_value_prototype(
        registry,
        _calibrations(registry),
        survivor_group=0,
        survivor_slot=0,
        ridge=1e-10,
    )
    installation = install_shared_codebook_value_prototype(registry, fit)
    installed_v = registry.prototype_v_weight[0, 0].detach().clone()

    with torch.no_grad():
        registry.prototype_q_weight[0, 0, 0, 0].add_(1.0)
    with pytest.raises(ResidualAttentionValueFitError, match="Unrelated registry"):
        installation.rollback()

    assert installation.is_applied is True
    assert torch.equal(registry.prototype_v_weight[0, 0], installed_v)

    with torch.no_grad():
        registry.prototype_q_weight.copy_(before["prototype_q_weight"])
    installation.rollback()
    assert installation.is_applied is False
    for name, value in before.items():
        assert torch.equal(registry.state_dict()[name], value)


def test_global_v_fit_rejects_partial_affected_site_evidence() -> None:
    registry = _registry()
    calibrations = _calibrations(registry)

    with pytest.raises(ResidualAttentionValueFitError, match="every and only"):
        fit_shared_codebook_value_prototype(
            registry,
            calibrations[:1],
            survivor_group=0,
            survivor_slot=0,
        )


def test_global_v_fit_rejects_calibration_stale_to_live_materialized_v() -> None:
    registry = _registry()
    calibrations = _calibrations(registry)
    with torch.no_grad():
        registry.prototype_v_weight[0, 0, 0, 0].add_(1.0)

    with pytest.raises(ResidualAttentionValueFitError, match="stale or foreign"):
        fit_shared_codebook_value_prototype(
            registry,
            calibrations,
            survivor_group=0,
            survivor_slot=0,
        )


def _manual_trace(
    *,
    sequence_id: str,
    layer_index: int,
    group_indices: tuple[int, ...],
    value_input: torch.Tensor,
    value_cache: torch.Tensor,
    v_weight_sha256: str,
    v_bias_sha256: str | None = None,
    context: torch.Tensor | None = None,
    delta: torch.Tensor | None = None,
) -> HFCachedAttentionTrace:
    prefix = int(value_input.shape[1])
    heads = int(value_cache.shape[1])
    head_dim = int(value_cache.shape[-1])
    context = torch.zeros(1, 4, dtype=torch.float64) if context is None else context
    delta = torch.zeros(1, 3, dtype=torch.float64) if delta is None else delta
    pre = torch.zeros_like(delta)
    selected_cache = value_cache[:, list(group_indices)]
    step = HFCachedAttentionStep(
        step_index=0,
        prefix_length=prefix,
        selected_token_id=7,
        pre_attention_residual=pre,
        value_projection_input=value_input,
        query=torch.zeros(1, len(group_indices), 1, head_dim),
        key=torch.zeros(1, len(group_indices), 1, head_dim),
        value=selected_cache[..., -1:, :],
        attention_scores=torch.zeros(1, len(group_indices), 1, prefix),
        attention_probabilities=torch.full(
            (1, len(group_indices), 1, prefix), 1.0 / prefix
        ),
        full_attention_probabilities=torch.full((1, heads, 1, prefix), 1.0 / prefix),
        selected_attention_context=torch.zeros(1, len(group_indices) * head_dim),
        full_attention_context=context,
        attention_delta=delta,
        post_attention_residual=pre + delta,
        layer_key_cache=torch.zeros_like(value_cache),
        layer_value_cache=value_cache,
        selected_key_cache=torch.zeros_like(selected_cache),
        selected_value_cache=selected_cache,
        action_logits=torch.zeros(1, 11),
        cached_full_prefix_max_abs_error=0.0,
        selected_context_max_abs_error=0.0,
        value_projection_cache_max_abs_error=0.0,
        captured_byte_count=1,
    )
    return HFCachedAttentionTrace(
        sequence_id=sequence_id,
        lane="forced",
        layer_index=layer_index,
        group_indices=group_indices,
        prompt_token_ids=(1, 2),
        consumed_token_ids=(),
        selected_token_ids=(7,),
        steps=(step,),
        query_heads=heads,
        key_value_heads=heads,
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
        model_class="toy",
        attention_backend="eager",
        cache_class="toy",
        logit_rtol=1e-5,
        logit_atol=1e-5,
        context_rtol=1e-5,
        context_atol=1e-5,
        attention_source="toy-certified",
    )


def test_trace_pair_maps_canonical_group_to_physical_parent_v_rows() -> None:
    registry = _registry()
    child_u = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float64)
    parent_cache = torch.tensor(
        [[[[10.0, 11.0], [12.0, 13.0]], [[20.0, 21.0], [22.0, 23.0]]]],
        dtype=torch.float64,
    )
    materialized_weight, materialized_bias = registry.materialize_projection(0, "v")
    materialized_weight_sha256 = _tensor_sha256(materialized_weight)
    materialized_bias_sha256 = (
        None if materialized_bias is None else _tensor_sha256(materialized_bias)
    )
    child = _manual_trace(
        sequence_id="mapped",
        layer_index=0,
        group_indices=(1,),
        value_input=child_u,
        value_cache=torch.zeros_like(parent_cache),
        v_weight_sha256=materialized_weight_sha256,
        v_bias_sha256=materialized_bias_sha256,
    )
    parent = _manual_trace(
        sequence_id="mapped",
        layer_index=0,
        group_indices=(1,),
        value_input=child_u,
        value_cache=parent_cache,
        v_weight_sha256="parent-v",
    )

    calibration = build_value_site_calibration(
        child, parent, registry, canonical_group=0
    )

    assert calibration.physical_group == 1
    np.testing.assert_array_equal(calibration.input_rows, child_u.numpy()[0])
    np.testing.assert_array_equal(
        calibration.target_value_rows, parent_cache.numpy()[0, 1]
    )
    assert calibration.as_dict()["design_source"] == "exact-child-v-proj-input"
    assert calibration.as_dict()["target_source"] == "parent-post-update-v-cache"
    assert calibration.child_v_weight_sha256 == materialized_weight_sha256
    assert calibration.child_v_bias_sha256 == materialized_bias_sha256
    assert (
        calibration.as_dict()["runtime_contract"]["cross_tokenizer_mapping_supported"]
        is False
    )

    stale_child = replace(child, value_projection_weight_sha256="stale-v")
    with pytest.raises(ResidualAttentionValueFitError, match="live registry"):
        build_value_site_calibration(
            stale_child,
            parent,
            registry,
            canonical_group=0,
        )


def test_o_refit_rejects_pre_v_trace_and_uses_post_v_actual_context() -> None:
    registry = _registry()
    fit = fit_shared_codebook_value_prototype(
        registry,
        _calibrations(registry),
        survivor_group=0,
        survivor_slot=0,
        ridge=1e-10,
    )
    installation = install_shared_codebook_value_prototype(registry, fit)
    expected_hash, expected_bias_hash = dict(
        (layer, (weight, bias))
        for layer, weight, bias in (installation.report.materialized_v_sha256_by_layer)
    )[0]
    value_input = torch.ones(1, 2, 3, dtype=torch.float64)
    value_cache = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    context = torch.tensor([[1.0, 2.0, -1.0, 0.5]], dtype=torch.float64)
    delta = torch.tensor([[0.5, -0.25, 1.0]], dtype=torch.float64)
    stale = _manual_trace(
        sequence_id="alternating",
        layer_index=0,
        group_indices=(1,),
        value_input=value_input,
        value_cache=value_cache,
        v_weight_sha256="pre-v-stale",
        context=context,
        delta=delta,
    )
    post_v = replace(
        stale,
        value_projection_weight_sha256=expected_hash,
        value_projection_bias_sha256=expected_bias_hash,
    )
    parent = replace(stale, value_projection_weight_sha256="parent-v")

    with pytest.raises(ResidualAttentionValueFitError, match="not recollected"):
        refit_output_after_value_installation(
            installation,
            (stale,),
            (parent,),
            np.zeros((3, 4)),
        )

    alternating = refit_output_after_value_installation(
        installation,
        (post_v,),
        (parent,),
        np.zeros((3, 4)),
    )
    assert alternating.output_fit.report.method.startswith("post-V-actual-child")
    assert alternating.as_dict()["actual_child_attention_context_used"] is True
    assert alternating.as_dict()["experimental_not_compaction_proof"] is True
    assert alternating.as_dict()["shared_codebook_o_installable"] is False
    assert alternating.as_dict()["requires_block_sparse_shared_o_fitter"] is True
    installation.rollback()


def test_o_refit_rejects_stale_trace_after_live_v_target_changes() -> None:
    registry = _registry()
    fit = fit_shared_codebook_value_prototype(
        registry,
        _calibrations(registry),
        survivor_group=0,
        survivor_slot=0,
        ridge=1e-10,
    )
    installation = install_shared_codebook_value_prototype(registry, fit)
    installed_weight = registry.prototype_v_weight[0, 0].detach().clone()
    expected_hash, expected_bias_hash = dict(
        (layer, (weight, bias))
        for layer, weight, bias in (installation.report.materialized_v_sha256_by_layer)
    )[0]
    stale = _manual_trace(
        sequence_id="stale-live-installation",
        layer_index=0,
        group_indices=(1,),
        value_input=torch.ones(1, 2, 3, dtype=torch.float64),
        value_cache=torch.zeros(1, 2, 2, 2, dtype=torch.float64),
        v_weight_sha256=expected_hash,
        context=torch.tensor([[1.0, 2.0, -1.0, 0.5]], dtype=torch.float64),
        delta=torch.tensor([[0.5, -0.25, 1.0]], dtype=torch.float64),
    )
    child = replace(stale, value_projection_bias_sha256=expected_bias_hash)
    parent = replace(stale, value_projection_weight_sha256="parent-v")

    with torch.no_grad():
        registry.prototype_v_weight[0, 0].add_(123.0)
    with pytest.raises(ResidualAttentionValueFitError, match="live registry"):
        installation.assert_live()
    with pytest.raises(ResidualAttentionValueFitError, match="live registry"):
        refit_output_after_value_installation(
            installation,
            (child,),
            (parent,),
            np.zeros((3, 4)),
        )

    with torch.no_grad():
        registry.prototype_v_weight[0, 0].copy_(installed_weight)
    installation.rollback()
