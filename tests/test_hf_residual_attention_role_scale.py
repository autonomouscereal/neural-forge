from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_attention_codebook import (  # noqa: E402
    AttentionCodebookRegistry,
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_residual_attention_role_scale import (  # noqa: E402
    AttentionRoleScales,
    ResidualAttentionRoleScaleError,
    install_attention_role_scales,
)


def _registry(
    *, with_scales: bool = True, dtype: torch.dtype = torch.float32
) -> RaggedAttentionCodebookRegistry:
    torch.manual_seed(4401)
    prototypes = 4
    q_rows = 2
    head_dim = 2
    width = 4
    return RaggedAttentionCodebookRegistry(
        torch.randn(prototypes, q_rows, width, dtype=dtype),
        torch.randn(prototypes, head_dim, width, dtype=dtype),
        torch.randn(prototypes, head_dim, width, dtype=dtype),
        torch.randn(prototypes, width, q_rows, dtype=dtype),
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
        role_scales=(torch.ones(2, 2, 4, dtype=dtype) if with_scales else None),
    )


def _rectangular_registry() -> AttentionCodebookRegistry:
    torch.manual_seed(4402)
    groups = 3
    slots = 2
    layers = 2
    q_rows = 2
    head_dim = 2
    width = 4
    return AttentionCodebookRegistry(
        torch.randn(groups, slots, q_rows, width),
        torch.randn(groups, slots, head_dim, width),
        torch.randn(groups, slots, head_dim, width),
        torch.randn(groups, slots, width, q_rows),
        torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int64),
        torch.tensor([[2, 4], [0, 5], [1, 3]], dtype=torch.int64),
        torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.int64),
        role_scales=torch.ones(layers, groups, 4),
    )


def _state(registry) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone() for name, value in registry.state_dict().items()
    }


def test_physical_group_mapping_edit_and_exact_rollback() -> None:
    registry = _registry()
    before = _state(registry)
    parameter_ids = tuple(id(value) for value in registry.parameters())
    before_v, _ = registry.materialize_projection(0, "v")

    installation = install_attention_role_scales(
        registry,
        layer_index=0,
        physical_group=1,
        scales={"q": 1.0, "k": 1.0, "v": 0.25, "o": -0.5},
    )

    assert installation.report.canonical_group == 0
    assert installation.report.physical_group == 1
    assert installation.report.changed_scalar_count == 2
    assert registry.role_scales is not None
    assert torch.equal(registry.role_scales[0, 0], torch.tensor([1.0, 1.0, 0.25, -0.5]))
    assert torch.equal(registry.role_scales[0, 1], torch.ones(4))
    after_v, _ = registry.materialize_projection(0, "v")
    assert not torch.equal(before_v, after_v)
    assert tuple(id(value) for value in registry.parameters()) == parameter_ids
    assert installation.report.as_dict()["weights_required_to_match_parent"] is False
    assert installation.assert_live() == installation.report

    installation.rollback()
    assert installation.is_applied is False
    assert all(
        torch.equal(registry.state_dict()[name], value)
        for name, value in before.items()
    )
    assert tuple(id(value) for value in registry.parameters()) == parameter_ids


def test_context_manager_rolls_back_uncommitted_edit() -> None:
    registry = _registry()
    before = _state(registry)
    with install_attention_role_scales(
        registry,
        layer_index=1,
        physical_group=0,
        scales=AttentionRoleScales(1.0, 1.0, -1.0, -1.0),
    ) as installation:
        assert installation.is_applied
    assert all(
        torch.equal(registry.state_dict()[name], value)
        for name, value in before.items()
    )


def test_commit_keeps_edit_and_disallows_handle_rollback() -> None:
    registry = _registry()
    installation = install_attention_role_scales(
        registry,
        layer_index=1,
        physical_group=1,
        scales={"q": 0.9, "k": 1.1, "v": 0.5, "o": 0.25},
    )
    installation.commit()
    assert installation.is_committed
    with pytest.raises(ResidualAttentionRoleScaleError, match="committed"):
        installation.rollback()


def test_committed_context_keeps_edit_and_propagates_later_body_exception() -> None:
    registry = _registry()
    installation = None
    with pytest.raises(RuntimeError, match="after commit"):
        with install_attention_role_scales(
            registry,
            layer_index=1,
            physical_group=1,
            scales={"q": 0.9, "k": 1.1, "v": 0.5, "o": 0.25},
        ) as installation:
            installation.commit()
            raise RuntimeError("after commit")

    assert installation is not None
    assert installation.is_committed
    assert installation.assert_live() == installation.report


def test_external_tamper_blocks_rollback_without_overwrite() -> None:
    registry = _registry()
    installation = install_attention_role_scales(
        registry,
        layer_index=0,
        physical_group=1,
        scales={"q": 1.0, "k": 1.0, "v": 0.5, "o": 0.5},
    )
    assert registry.role_scales is not None
    with torch.no_grad():
        registry.role_scales[1, 1, 0] = 9.0
    target_before = registry.role_scales[0, 0].detach().clone()
    with pytest.raises(ResidualAttentionRoleScaleError, match="no longer matches"):
        installation.rollback()
    assert torch.equal(registry.role_scales[0, 0], target_before)


def test_tamper_blocked_rollback_is_retryable_after_external_state_is_repaired() -> (
    None
):
    registry = _registry()
    before = _state(registry)
    installation = install_attention_role_scales(
        registry,
        layer_index=0,
        physical_group=1,
        scales={"q": 1.0, "k": 1.0, "v": 0.5, "o": 0.5},
    )
    assert registry.role_scales is not None
    installed = registry.role_scales.detach().clone()
    with torch.no_grad():
        registry.role_scales[1, 1, 0] = 9.0
    with pytest.raises(ResidualAttentionRoleScaleError, match="no longer matches"):
        installation.rollback()
    with torch.no_grad():
        registry.role_scales.copy_(installed)
    installation.rollback()
    assert installation.is_applied is False
    assert all(
        torch.equal(registry.state_dict()[name], value)
        for name, value in before.items()
    )


def test_requires_existing_scales_and_valid_bounded_values() -> None:
    with pytest.raises(ResidualAttentionRoleScaleError, match="existing"):
        install_attention_role_scales(
            _registry(with_scales=False),
            layer_index=0,
            physical_group=0,
            scales={"q": 1, "k": 1, "v": 1, "o": 1},
        )
    registry = _registry()
    with pytest.raises(ResidualAttentionRoleScaleError, match="exactly"):
        install_attention_role_scales(
            registry,
            layer_index=0,
            physical_group=0,
            scales={"v": 1, "o": 1},
        )
    with pytest.raises(ResidualAttentionRoleScaleError, match="absolute"):
        install_attention_role_scales(
            registry,
            layer_index=0,
            physical_group=0,
            scales={"q": 1, "k": 1, "v": 17, "o": 1},
        )
    with pytest.raises(ResidualAttentionRoleScaleError, match="finite"):
        install_attention_role_scales(
            registry,
            layer_index=0,
            physical_group=0,
            scales={"q": 1, "k": 1, "v": float("nan"), "o": 1},
        )


def test_noop_scale_install_is_audited_and_reversible() -> None:
    registry = _registry()
    installation = install_attention_role_scales(
        registry,
        layer_index=0,
        physical_group=0,
        scales={"q": 1, "k": 1, "v": 1, "o": 1},
    )
    assert installation.report.canonical_group == 1
    assert installation.report.changed_scalar_count == 0
    installation.rollback()


def test_float16_requested_and_materialized_scales_are_both_audited() -> None:
    registry = _registry(dtype=torch.float16)
    before = _state(registry)
    installation = install_attention_role_scales(
        registry,
        layer_index=0,
        physical_group=1,
        scales={"q": 1.0, "k": 1.0, "v": 1.0 / 3.0, "o": -1.0 / 3.0},
    )

    assert installation.report.requested_scales.v == pytest.approx(1.0 / 3.0)
    assert installation.report.installed_scales.v == float(
        torch.tensor(1.0 / 3.0, dtype=torch.float16)
    )
    assert installation.report.role_scales_dtype == "torch.float16"
    assert (
        installation.report.as_dict()["requested_values_exactly_representable"] is False
    )
    installation.rollback()
    assert all(
        torch.equal(registry.state_dict()[name], value)
        for name, value in before.items()
    )


def test_rectangular_registry_maps_physical_group_and_preserves_other_blocks() -> None:
    registry = _rectangular_registry()
    before = _state(registry)
    before_v, _ = registry.materialize_projection(0, "v")
    installation = install_attention_role_scales(
        registry,
        layer_index=0,
        physical_group=0,
        scales={"q": 1.0, "k": 1.0, "v": 0.25, "o": 1.0},
    )

    assert installation.report.canonical_group == 1
    after_v, _ = registry.materialize_projection(0, "v")
    assert not torch.equal(after_v[:2], before_v[:2])
    assert torch.equal(after_v[2:], before_v[2:])
    installation.rollback()
    assert all(
        torch.equal(registry.state_dict()[name], value)
        for name, value in before.items()
    )


def test_incomplete_permutation_is_rejected_even_when_target_appears_once() -> None:
    registry = _rectangular_registry()
    registry.group_permutations[0].copy_(torch.tensor([0, 1, 1]))
    with pytest.raises(ResidualAttentionRoleScaleError, match="complete.*bijection"):
        install_attention_role_scales(
            registry,
            layer_index=0,
            physical_group=0,
            scales={"q": 1.0, "k": 1.0, "v": 0.5, "o": 0.5},
        )

    registry = _rectangular_registry()
    registry.group_permutations = registry.group_permutations.to(torch.float32)
    with pytest.raises(ResidualAttentionRoleScaleError, match="contain integers"):
        install_attention_role_scales(
            registry,
            layer_index=0,
            physical_group=0,
            scales={"q": 1.0, "k": 1.0, "v": 0.5, "o": 0.5},
        )


def test_installation_failure_restores_full_scale_tensor_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    before = _state(registry)
    parameter_ids = tuple(id(value) for value in registry.parameters())

    def fail_materialization(*_args, **_kwargs):
        raise RuntimeError("synthetic materialization failure")

    monkeypatch.setattr(registry, "materialize_projection", fail_materialization)
    with pytest.raises(RuntimeError, match="synthetic materialization failure"):
        install_attention_role_scales(
            registry,
            layer_index=0,
            physical_group=1,
            scales={"q": 1.0, "k": 1.0, "v": 0.5, "o": 0.5},
        )
    assert all(
        torch.equal(registry.state_dict()[name], value)
        for name, value in before.items()
    )
    assert tuple(id(value) for value in registry.parameters()) == parameter_ids


def test_requires_grad_tamper_is_part_of_parameter_identity_audit() -> None:
    registry = _registry()
    installation = install_attention_role_scales(
        registry,
        layer_index=0,
        physical_group=1,
        scales={"q": 1.0, "k": 1.0, "v": 0.5, "o": 0.5},
    )
    parameter = registry.prototype_q_weight
    parameter.requires_grad_(False)
    with pytest.raises(ResidualAttentionRoleScaleError, match="no longer matches"):
        installation.assert_live()
    parameter.requires_grad_(True)
    installation.rollback()
