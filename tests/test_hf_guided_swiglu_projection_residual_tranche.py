from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import neural_forge.hf_dense_swiglu_projection_residual_adapter as guided_gud
from neural_forge.hf_dense_swiglu_projection_residual_adapter import (
    DenseSwiGLUProjectionResidualError,
    DenseSwiGLUProjectionResidualTopologyError,
    DenseSwiGLUProjectionResidualTrainingError,
    DenseSwiGLUResidualProjection,
    GuidedSparseDenseSwiGLUProjectionResidualRegistry,
    install_dense_swiglu_projection_residual_adapter,
    install_guided_dense_swiglu_projection_residual_tranche,
)


ROLES = ("gate", "up", "down")


class _ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=True)
        self.up_proj = nn.Linear(4, 6, bias=True)
        self.down_proj = nn.Linear(6, 4, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(11, 4)
        self.mlps = nn.ModuleList(_ToyMLP() for _ in range(2))
        self.lm_head = nn.Linear(4, 7, bias=False)

    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        for mlp in self.mlps:
            hidden = hidden + mlp(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def _parameter_count(module: nn.Module) -> int:
    return sum(value.numel() for value in module.parameters())


def _install_inherited(model: _ToyModel):
    return install_dense_swiglu_projection_residual_adapter(
        model,
        model.mlps,
        layer_indices=(0, 1),
        hidden_size=4,
        intermediate_size=6,
        rank=2,
        initialization_seed=17,
    )


def _activate_inherited(handle) -> None:
    with torch.no_grad():
        for role_index, role in enumerate(ROLES, start=1):
            getattr(handle.registry, f"{role}_left").fill_(role_index * 0.02)
            getattr(handle.registry, f"{role}_right").fill_(role_index * 0.03)


def _guided_sources() -> tuple[dict[str, Tensor], dict[str, str]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    rows = {
        "gate": torch.randn(2, 2, 4, generator=generator),
        "up": torch.randn(2, 2, 4, generator=generator),
        "down": torch.randn(2, 2, 6, generator=generator),
    }
    hashes = {role: guided_gud._tensor_sha256(rows[role]) for role in ROLES}
    return rows, hashes


def _install_guided(handle):
    rows, hashes = _guided_sources()
    combined = install_guided_dense_swiglu_projection_residual_tranche(
        handle,
        guided_right_rows=rows,
        source_basis_tensor_sha256s=hashes,
        expected_audit_report_sha256="1" * 64,
        expected_basis_envelope_file_sha256="2" * 64,
        expected_basis_payload_sha256="3" * 64,
        expected_basis_manifest_sha256="4" * 64,
    )
    return combined, rows, hashes


def test_real_guided_gud_parameter_count_is_exact() -> None:
    accounting = guided_gud.DenseSwiGLUProjectionResidualAccounting.build(
        hidden_size=1536,
        intermediate_size=5120,
        rank=8,
        site_count=30,
    )

    assert accounting.gate_adapter_parameters == 1_597_440
    assert accounting.up_adapter_parameters == 1_597_440
    assert accounting.down_adapter_parameters == 1_597_440
    assert accounting.adapter_parameters == 4_792_320


def test_guided_registry_authenticates_order_shape_dtype_contiguity_and_hash() -> None:
    rows, hashes = _guided_sources()
    registry = GuidedSparseDenseSwiGLUProjectionResidualRegistry(
        (0, 1),
        hidden_size=4,
        intermediate_size=6,
        rank=2,
        guided_right_rows=rows,
        source_basis_tensor_sha256s=hashes,
        audit_report_sha256="1" * 64,
        basis_envelope_file_sha256="2" * 64,
        basis_payload_sha256="3" * 64,
        basis_manifest_sha256="4" * 64,
        dtype=torch.float32,
        device="cpu",
    )
    assert registry.starts_as_exact_zero_correction
    assert registry.learned_parameter_count == 120
    assert registry.as_dict()["old_and_guided_factors_concatenated"] is False

    reordered = {"up": rows["up"], "gate": rows["gate"], "down": rows["down"]}
    with pytest.raises(DenseSwiGLUProjectionResidualTopologyError, match="ordered"):
        GuidedSparseDenseSwiGLUProjectionResidualRegistry(
            (0, 1),
            hidden_size=4,
            intermediate_size=6,
            rank=2,
            guided_right_rows=reordered,
            source_basis_tensor_sha256s=hashes,
            audit_report_sha256="1" * 64,
            basis_envelope_file_sha256="2" * 64,
            basis_payload_sha256="3" * 64,
            basis_manifest_sha256="4" * 64,
            dtype=torch.float32,
            device="cpu",
        )

    noncontiguous = dict(rows)
    noncontiguous["gate"] = torch.randn(2, 4, 2).transpose(1, 2)
    assert not noncontiguous["gate"].is_contiguous()
    with pytest.raises(DenseSwiGLUProjectionResidualTopologyError, match="contiguous"):
        GuidedSparseDenseSwiGLUProjectionResidualRegistry(
            (0, 1),
            hidden_size=4,
            intermediate_size=6,
            rank=2,
            guided_right_rows=noncontiguous,
            source_basis_tensor_sha256s=hashes,
            audit_report_sha256="1" * 64,
            basis_envelope_file_sha256="2" * 64,
            basis_payload_sha256="3" * 64,
            basis_manifest_sha256="4" * 64,
            dtype=torch.float32,
            device="cpu",
        )

    with pytest.raises(
        DenseSwiGLUProjectionResidualTopologyError, match="authentication"
    ):
        GuidedSparseDenseSwiGLUProjectionResidualRegistry(
            (0, 1),
            hidden_size=4,
            intermediate_size=6,
            rank=2,
            guided_right_rows=rows,
            source_basis_tensor_sha256s={**hashes, "gate": "f" * 64},
            audit_report_sha256="1" * 64,
            basis_envelope_file_sha256="2" * 64,
            basis_payload_sha256="3" * 64,
            basis_manifest_sha256="4" * 64,
            dtype=torch.float32,
            device="cpu",
        )


def test_install_is_exact_identity_separate_and_strictly_accounted() -> None:
    torch.manual_seed(37)
    model = _ToyModel()
    inherited = _install_inherited(model)
    _activate_inherited(inherited)
    input_ids = torch.tensor([[1, 3, 5]])
    inherited_output = model(input_ids).logits.detach().clone()
    inherited_state = {
        name: value.detach().clone()
        for name, value in inherited.registry.named_parameters(recurse=False)
    }
    before_guided = _parameter_count(model)

    combined, rows, _hashes = _install_guided(inherited)

    assert torch.equal(model(input_ids).logits, inherited_output)
    assert combined.guided_registry is not combined.registry
    assert combined.report.model_parameter_count_before_guided_installation == (
        before_guided
    )
    assert combined.report.guided_adapter_parameter_count == 120
    assert combined.report.adapter_parameter_count == 240
    assert _parameter_count(model) == before_guided + 120
    assert all(
        torch.equal(value, inherited_state[name])
        for name, value in inherited.registry.named_parameters(recurse=False)
    )
    assert all(
        torch.equal(getattr(combined.guided_registry, f"{role}_right"), rows[role])
        for role in ROLES
    )
    assert all(
        isinstance(getattr(mlp, f"{role}_proj"), DenseSwiGLUResidualProjection)
        and getattr(mlp, f"{role}_proj").guided_registry is combined.guided_registry
        for mlp in model.mlps
        for role in ROLES
    )
    paths = [
        name
        for name, module in model.named_modules(remove_duplicate=False)
        if module is combined.guided_registry
    ]
    assert paths == [combined.report.owner_name]
    binding = combined.factor_source_binding()
    assert binding["audit_report_sha256"] == "1" * 64
    with pytest.raises(TypeError):
        binding["audit_report_sha256"] = "5" * 64
    combined.rollback()


def test_training_stages_and_guided_only_right_basis_lock() -> None:
    model = _ToyModel()
    combined, _rows, _hashes = _install_guided(_install_inherited(model))

    initial = combined.output_trainable_parameters()
    assert len(initial) == 3
    assert all(
        name.endswith("_left")
        for name, value in combined.guided_registry.named_parameters(recurse=False)
        if any(value is selected for selected in initial)
    )
    assert all(not value.requires_grad for value in combined.registry.parameters())

    guided = combined.set_training_stage("guided")
    assert len(guided) == 6
    assert all(value.requires_grad for value in combined.guided_registry.parameters())
    assert all(not value.requires_grad for value in combined.registry.parameters())

    joint = combined.set_training_stage("joint")
    assert len(joint) == 12
    assert all(value.requires_grad for value in joint)
    with pytest.raises(DenseSwiGLUProjectionResidualTrainingError, match="backward"):
        combined.set_training_stage("guided")
    combined.rollback()

    combined, _rows, _hashes = _install_guided(_install_inherited(_ToyModel()))
    with torch.no_grad():
        combined.guided_registry.gate_right[0, 0, 0].add_(0.5)
    with pytest.raises(DenseSwiGLUProjectionResidualTopologyError, match="right basis"):
        combined.assert_live()
    combined.rollback()


def test_output_training_scope_rejects_unselected_mutation() -> None:
    model = _ToyModel()
    combined, _rows, _hashes = _install_guided(_install_inherited(model))
    base = model.embedding.weight

    with pytest.raises(
        DenseSwiGLUProjectionResidualTrainingError, match="unselected parameter"
    ):
        with combined.output_training_scope() as trainable:
            assert len(trainable) == 3
            with torch.no_grad():
                base.add_(1.0)
    combined.rollback()


def test_guided_only_and_full_rollback_restore_exact_products() -> None:
    torch.manual_seed(41)
    model = _ToyModel()
    native_count = _parameter_count(model)
    inherited = _install_inherited(model)
    _activate_inherited(inherited)
    input_ids = torch.tensor([[2, 4]])
    inherited_output = model(input_ids).logits.detach().clone()
    combined, _rows, _hashes = _install_guided(inherited)
    with torch.no_grad():
        combined.guided_registry.up_left.fill_(0.3)
    assert not torch.equal(model(input_ids).logits, inherited_output)

    restored = combined.rollback_guided_only()
    assert restored is inherited
    assert inherited.is_applied
    assert torch.equal(model(input_ids).logits, inherited_output)
    assert not hasattr(model, combined.report.owner_name)
    inherited.rollback()
    assert _parameter_count(model) == native_count

    model = _ToyModel()
    native_count = _parameter_count(model)
    combined, _rows, _hashes = _install_guided(_install_inherited(model))
    with torch.no_grad():
        combined.guided_registry.down_left[0, 0, 0] = float("nan")
    with pytest.raises(DenseSwiGLUProjectionResidualError):
        combined.assert_live()
    combined.rollback()
    assert _parameter_count(model) == native_count
    assert all(
        type(getattr(mlp, f"{role}_proj")) is nn.Linear
        for mlp in model.mlps
        for role in ROLES
    )

    model = _ToyModel()
    native_count = _parameter_count(model)
    combined, _rows, _hashes = _install_guided(_install_inherited(model))
    combined.set_training_stage("joint")
    with torch.no_grad():
        combined.registry.gate_left[0, 0, 0] = float("nan")
    with pytest.raises(DenseSwiGLUProjectionResidualError):
        combined.assert_live()
    combined.rollback()
    assert _parameter_count(model) == native_count


def test_sequential_bake_matches_live_and_native_restore_releases_clones() -> None:
    torch.manual_seed(43)
    model = _ToyModel()
    native_count = _parameter_count(model)
    native_weights = {
        (layer, role): getattr(model.mlps[layer], f"{role}_proj")
        .weight.detach()
        .clone()
        for layer in range(2)
        for role in ROLES
    }
    inherited = _install_inherited(model)
    _activate_inherited(inherited)
    combined, _rows, _hashes = _install_guided(inherited)
    with torch.no_grad():
        for role_index, role in enumerate(ROLES, start=1):
            getattr(combined.guided_registry, f"{role}_left").fill_(role_index * 0.04)
    input_ids = torch.tensor([[1, 2, 6]])
    live_output = model(input_ids).logits.detach().clone()

    bake = combined.bake_for_publication()

    torch.testing.assert_close(
        model(input_ids).logits, live_output, rtol=1e-5, atol=1e-5
    )
    assert bake.adapter_parameters_removed == 240
    assert bake.inherited_adapter_parameters_removed == 120
    assert bake.guided_adapter_parameters_removed == 120
    assert bake.affected_projection_count == 12
    assert _parameter_count(model) == native_count
    assert all(
        type(getattr(mlp, f"{role}_proj")) is nn.Linear
        for mlp in model.mlps
        for role in ROLES
    )
    assert bake.as_dict()["bake_rule"] == [
        "dense_weight += inherited_left @ inherited_right",
        "dense_weight += guided_left @ guided_right",
    ]
    replacement_references = tuple(
        weakref.ref(projection) for projection in combined._published.values()
    )
    combined.restore_prepublication_native(expected_model_parameter_count=native_count)
    gc.collect()
    assert combined._published == {}
    assert inherited._published == {}
    assert all(reference() is None for reference in replacement_references)
    assert all(
        torch.equal(
            getattr(model.mlps[layer], f"{role}_proj").weight,
            native_weights[(layer, role)],
        )
        for layer in range(2)
        for role in ROLES
    )


def test_suspension_restores_same_owner_and_parameter_identity() -> None:
    model = _ToyModel()
    combined, _rows, _hashes = _install_guided(_install_inherited(model))
    before = _parameter_count(model)
    identities = tuple(
        (id(value), value.data_ptr()) for value in combined.guided_registry.parameters()
    )

    with combined.suspend_guided_registry_owner_for_nested_publication():
        assert not hasattr(model, combined.report.owner_name)
        assert _parameter_count(model) == (
            before - combined.report.guided_adapter_parameter_count
        )

    assert getattr(model, combined.report.owner_name) is combined.guided_registry
    assert (
        tuple(
            (id(value), value.data_ptr())
            for value in combined.guided_registry.parameters()
        )
        == identities
    )
    combined.rollback()


def test_failed_install_and_bake_restore_live_inherited_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyModel()
    inherited = _install_inherited(model)
    before = _parameter_count(model)
    requires_grad = {
        name: value.requires_grad
        for name, value in inherited.registry.named_parameters(recurse=False)
    }
    rows, hashes = _guided_sources()
    real_count = guided_gud._unique_parameter_count
    calls = 0

    def wrong_install_count(module: nn.Module) -> int:
        nonlocal calls
        calls += 1
        result = real_count(module)
        return result + (1 if calls == 2 else 0)

    monkeypatch.setattr(guided_gud, "_unique_parameter_count", wrong_install_count)
    with pytest.raises(DenseSwiGLUProjectionResidualError):
        install_guided_dense_swiglu_projection_residual_tranche(
            inherited,
            guided_right_rows=rows,
            source_basis_tensor_sha256s=hashes,
            expected_audit_report_sha256="1" * 64,
            expected_basis_envelope_file_sha256="2" * 64,
            expected_basis_payload_sha256="3" * 64,
            expected_basis_manifest_sha256="4" * 64,
        )
    monkeypatch.setattr(guided_gud, "_unique_parameter_count", real_count)
    inherited.assert_live()
    assert _parameter_count(model) == before
    assert all(
        value.requires_grad == requires_grad[name]
        for name, value in inherited.registry.named_parameters(recurse=False)
    )
    assert all(
        installed.guided_registry is None for installed in inherited._installed.values()
    )

    combined, _rows, _hashes = _install_guided(inherited)
    original_weights = {
        key: value.weight.detach().clone()
        for key, value in inherited._originals.items()
    }
    calls = 0

    def wrong_bake_count(module: nn.Module) -> int:
        nonlocal calls
        calls += 1
        result = real_count(module)
        return result + (1 if calls >= 2 else 0)

    monkeypatch.setattr(guided_gud, "_unique_parameter_count", wrong_bake_count)
    with pytest.raises(DenseSwiGLUProjectionResidualError):
        combined.bake_for_publication()
    monkeypatch.setattr(guided_gud, "_unique_parameter_count", real_count)
    combined.assert_live()
    assert all(
        torch.equal(value.weight, original_weights[key])
        for key, value in inherited._originals.items()
    )
    combined.rollback()


def test_duplicate_owner_and_projection_identity_tamper_fail_closed() -> None:
    model = _ToyModel()
    combined, _rows, _hashes = _install_guided(_install_inherited(model))
    model.duplicate_guided_owner = combined.guided_registry
    with pytest.raises(DenseSwiGLUProjectionResidualTopologyError, match="exactly one"):
        combined.assert_live()
    del model.duplicate_guided_owner

    original_view = model.mlps[0].gate_proj
    model.mlps[0].gate_proj = nn.Linear(4, 6)
    with pytest.raises(DenseSwiGLUProjectionResidualTopologyError, match="replaced"):
        combined.assert_live()
    model.mlps[0].gate_proj = original_view
    combined.rollback()
