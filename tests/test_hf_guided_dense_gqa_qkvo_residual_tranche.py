from __future__ import annotations

import copy
import gc
import hashlib
import json
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

import neural_forge.hf_dense_gqa_qkvo_residual_adapter as guided_qkvo
import neural_forge.hf_progressive_target_bridge as progressive_bridge
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    GUIDED_QKVO_RIGHT_BASIS_V2_SCHEMA,
    QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA,
    DenseGQAGeometry,
    DenseGQAQKVOResidualAccounting,
    DenseGQAQKVOResidualError,
    DenseGQAQKVOResidualProjection,
    DenseGQAQKVOResidualTopologyError,
    DenseGQAQKVOResidualTrainingError,
    DenseGQASite,
    guided_qkvo_basis_payload_sha256,
    install_dense_gqa_qkvo_residual_adapter,
    install_guided_dense_gqa_qkvo_residual_tranche,
)
from neural_forge.hf_dense_lexical_residual_adapter import (
    install_dense_lexical_residual_adapter,
)


TOY_GEOMETRY = DenseGQAGeometry(
    logical_layers=3,
    hidden_size=8,
    query_heads=4,
    key_value_heads=2,
    head_dim=2,
)
TOY_SITES = tuple(
    DenseGQASite(layer, group) for layer in range(3) for group in range(2)
)
ROLES = ("q", "k", "v", "o")
BASIS_NAMES = {
    "q": "q_orthogonal_right",
    "k": "k_orthogonal_right",
    "v": "v_orthogonal_right",
    "o": "o_orthogonal_right",
}


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=True)
        self.k_proj = nn.Linear(8, 4, bias=True)
        self.v_proj = nn.Linear(8, 4, bias=True)
        self.o_proj = nn.Linear(8, 8, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        query = self.q_proj(hidden)
        key = self.k_proj(hidden).repeat_interleave(2, dim=-1)
        value = self.v_proj(hidden).repeat_interleave(2, dim=-1)
        return self.o_proj(query + key + value)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attentions = nn.ModuleList(_ToyAttention() for _ in range(3))
        self.embedding = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 7, bias=False)

    def forward(self, input_ids: Tensor) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        for attention in self.attentions:
            hidden = attention(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


@pytest.fixture
def toy_stage1(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "_GUIDED_STAGE1_LOGICAL_LAYERS": 3,
        "_GUIDED_STAGE1_HIDDEN_SIZE": 8,
        "_GUIDED_STAGE1_QUERY_HEADS": 4,
        "_GUIDED_STAGE1_KEY_VALUE_HEADS": 2,
        "_GUIDED_STAGE1_HEAD_DIM": 2,
        "_GUIDED_STAGE1_RANK": 1,
        "_GUIDED_STAGE1_SITE_COUNT": 6,
        "_GUIDED_STAGE1_RIGHT_PARAMETER_COUNT": 168,
        "_GUIDED_STAGE1_PARAMETER_COUNT": 264,
    }
    for name, value in values.items():
        monkeypatch.setattr(guided_qkvo, name, value)


def _install_inherited(model: _ToyModel):
    return install_dense_gqa_qkvo_residual_adapter(
        model,
        model.attentions,
        geometry=TOY_GEOMETRY,
        sites=TOY_SITES,
        rank=1,
        initialization_seed=9,
    )


def _activate_inherited(handle) -> None:
    with torch.no_grad():
        for role_index, role in enumerate(ROLES, start=1):
            getattr(handle.adapter_registry, f"{role}_left").fill_(role_index * 0.02)
            getattr(handle.adapter_registry, f"{role}_right").fill_(role_index * 0.03)


def _source_payload(handle):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(71)
    factor_tensors = {
        "q_orthogonal_right": torch.randn(6, 1, 8, generator=generator),
        "k_orthogonal_right": torch.randn(6, 1, 8, generator=generator),
        "v_orthogonal_right": torch.randn(6, 1, 8, generator=generator),
        "o_orthogonal_right": torch.randn(6, 1, 4, generator=generator),
    }
    manifest = guided_qkvo._guided_basis_tensor_manifest(factor_tensors)
    manifest_sha256 = guided_qkvo._canonical_sha256(manifest)
    site_indices = [[site.layer_index, site.physical_group] for site in TOY_SITES]
    site_sha256 = guided_qkvo._canonical_sha256(site_indices)
    inherited_sha256 = guided_qkvo._qkvo_factor_state_sha256(handle.adapter_registry)
    audit = {
        "schema": QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA,
        "passed": True,
        "site_count": 6,
        "site_indices": site_indices,
        "site_indices_sha256": site_sha256,
        "factor_state_sha256_before": "2" * 64,
        "inherited_qkvo_state_sha256_before": inherited_sha256,
        "inherited_qkvo_state_sha256_after": inherited_sha256,
        "diagnostic_passed_roles": list(ROLES),
        "authorized_roles": list(ROLES),
        "installable_roles": list(ROLES),
        "per_role": {role: {"passed": True} for role in ROLES},
        "installer_must_ignore_unauthorized_role_tensors": True,
        "all_four_role_tensors_retained_for_diagnostics": True,
        "basis_manifest": manifest,
        "basis_manifest_sha256": manifest_sha256,
    }
    audit_sha256 = guided_qkvo._canonical_sha256(audit)
    audit["report_sha256"] = audit_sha256
    pack = {
        "schema": GUIDED_QKVO_RIGHT_BASIS_V2_SCHEMA,
        "factor_state_sha256": "2" * 64,
        "site_indices": site_indices,
        "site_indices_sha256": site_sha256,
        "basis_manifest_sha256": manifest_sha256,
        "tensor_count": 4,
        "total_numel": 168,
        "total_nbytes": 672,
        "factors": manifest,
        "factor_tensors": factor_tensors,
        "diagnostic_passed_roles": list(ROLES),
        "authorized_roles": list(ROLES),
        "installable_roles": list(ROLES),
        "authorized_for_training": True,
        "installer_must_ignore_unauthorized_role_tensors": True,
        "all_four_role_tensors_retained_for_diagnostics": True,
        "contains_model_state": False,
        "contains_parent_state": False,
        "contains_optimizer_state": False,
        "contains_gradients": False,
        "contains_activations": False,
        "contains_hidden_states": False,
    }
    return (
        audit,
        pack,
        {
            "expected_audit_report_sha256": audit_sha256,
            "expected_basis_envelope_file_sha256": "a" * 64,
            "expected_basis_payload_sha256": guided_qkvo_basis_payload_sha256(pack),
            "expected_basis_manifest_sha256": manifest_sha256,
        },
    )


def _install_guided(handle):
    audit, pack, hashes = _source_payload(handle)
    combined = install_guided_dense_gqa_qkvo_residual_tranche(
        handle,
        audit_report=audit,
        basis_pack=pack,
        **hashes,
    )
    return combined, audit, pack, hashes


def test_canonical_json_role_key_sorting_preserves_audit_authorization(
    toy_stage1: None,
) -> None:
    """Exercise the real JSON boundary, which alphabetizes ``per_role``."""

    del toy_stage1
    model = _ToyModel()
    inherited = _install_inherited(model)
    audit, pack, hashes = _source_payload(inherited)
    published_audit = json.loads(json.dumps(audit, sort_keys=True))

    assert tuple(published_audit["per_role"]) == ("k", "o", "q", "v")
    combined = install_guided_dense_gqa_qkvo_residual_tranche(
        inherited,
        audit_report=published_audit,
        basis_pack=pack,
        **hashes,
    )
    combined.assert_live()
    combined.rollback()


@pytest.mark.parametrize(
    "per_role",
    (
        {role: {"passed": True} for role in ROLES if role != "q"},
        {
            **{role: {"passed": True} for role in ROLES},
            "unauthorized": {"passed": True},
        },
        {
            **{role: {"passed": True} for role in ROLES if role != "q"},
            "unauthorized": {"passed": True},
        },
    ),
    ids=("missing-role", "extra-role", "same-length-role-substitution"),
)
def test_audit_per_role_requires_exact_semantic_role_set(
    toy_stage1: None,
    per_role: dict[str, dict[str, bool]],
) -> None:
    del toy_stage1
    model = _ToyModel()
    inherited = _install_inherited(model)
    audit, pack, hashes = _source_payload(inherited)
    audit["per_role"] = per_role
    audit.pop("report_sha256")
    audit_sha256 = guided_qkvo._canonical_sha256(audit)
    audit["report_sha256"] = audit_sha256

    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="all four"):
        install_guided_dense_gqa_qkvo_residual_tranche(
            inherited,
            audit_report=audit,
            basis_pack=pack,
            **{
                **hashes,
                "expected_audit_report_sha256": audit_sha256,
            },
        )
    inherited.rollback()


def test_tensor_hash_is_byte_compatible_and_uses_bounded_staging() -> None:
    element_count = guided_qkvo._TENSOR_HASH_STAGING_BYTES // 4 + 17
    value = torch.arange(element_count, dtype=torch.float32)
    expected = hashlib.sha256()
    expected.update(str(value.dtype).encode("ascii"))
    expected.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    )
    expected.update(value.view(torch.uint8).numpy().tobytes())
    observed: list[int] = []

    actual = guided_qkvo._tensor_sha256(
        value,
        staging_observer=observed.append,
    )

    assert actual == expected.hexdigest()
    assert len(observed) == 2
    assert max(observed) == guided_qkvo._TENSOR_HASH_STAGING_BYTES
    assert sum(observed) == value.numel() * value.element_size()


def test_real_stage_one_guided_factor_count_and_shapes_are_exact() -> None:
    accounting = DenseGQAQKVOResidualAccounting.build(
        logical_layers=30,
        hidden_size=1536,
        query_heads=16,
        key_value_heads=4,
        head_dim=128,
        rank=8,
        site_count=120,
        checkpoint_element_size=4,
    )

    assert accounting.q_adapter_parameters == 1_966_080
    assert accounting.k_adapter_parameters == 1_597_440
    assert accounting.v_adapter_parameters == 1_597_440
    assert accounting.o_adapter_parameters == 1_966_080
    assert accounting.adapter_parameters == 7_127_040
    assert 120 * 8 * (1536 + 1536 + 1536 + 512) == 4_915_200


def test_guided_install_is_exact_identity_and_keeps_products_separate(
    toy_stage1: None,
) -> None:
    del toy_stage1
    torch.manual_seed(31)
    model = _ToyModel()
    inherited = _install_inherited(model)
    _activate_inherited(inherited)
    input_ids = torch.tensor([[1, 2, 3]])
    inherited_output = model(input_ids).logits.detach().clone()
    inherited_state = {
        name: value.detach().clone()
        for name, value in inherited.adapter_registry.named_parameters(recurse=False)
    }

    combined, _audit, pack, _hashes = _install_guided(inherited)

    assert torch.equal(model(input_ids).logits, inherited_output)
    assert combined.guided_registry is not combined.adapter_registry
    assert combined.report.guided_adapter_parameter_count == 264
    assert combined.report.adapter_parameter_count == 528
    assert all(
        torch.equal(value, inherited_state[name])
        for name, value in inherited.adapter_registry.named_parameters(recurse=False)
    )
    assert all(
        torch.count_nonzero(getattr(combined.guided_registry, f"{role}_left")).item()
        == 0
        for role in ROLES
    )
    assert all(
        torch.equal(
            getattr(combined.guided_registry, f"{role}_right"),
            pack["factor_tensors"][BASIS_NAMES[role]],
        )
        for role in ROLES
    )
    assert all(
        isinstance(attention.q_proj, DenseGQAQKVOResidualProjection)
        and attention.q_proj.guided_adapter_registry is combined.guided_registry
        for attention in model.attentions
    )
    binding = combined.factor_source_binding()
    assert binding["basis_envelope_file_sha256"] == "a" * 64
    assert binding["basis_payload_sha256"] != binding["basis_envelope_file_sha256"]
    assert binding["basis_manifest_sha256"] == pack["basis_manifest_sha256"]
    combined.rollback()


def test_staged_trainability_is_guided_left_then_guided_then_joint(
    toy_stage1: None,
) -> None:
    del toy_stage1
    model = _ToyModel()
    combined, _audit, _pack, _hashes = _install_guided(_install_inherited(model))

    initial = combined.output_trainable_parameters()
    initial_ids = {id(value) for value in initial}
    assert len(initial) == 4
    assert all(
        name.endswith("_left")
        for name, value in combined.guided_registry.named_parameters(recurse=False)
        if id(value) in initial_ids
    )
    assert all(
        not value.requires_grad for value in combined.adapter_registry.parameters()
    )

    guided = combined.set_training_stage("guided")
    assert len(guided) == 8
    assert all(value.requires_grad for value in combined.guided_registry.parameters())
    assert all(
        not value.requires_grad for value in combined.adapter_registry.parameters()
    )

    joint = combined.set_training_stage("joint")
    assert len(joint) == 16
    assert all(value.requires_grad for value in joint)
    with pytest.raises(DenseGQAQKVOResidualTrainingError, match="backward"):
        combined.set_training_stage("guided")
    combined.rollback()


def test_guided_only_and_full_rollback_are_exact(toy_stage1: None) -> None:
    del toy_stage1
    torch.manual_seed(41)
    model = _ToyModel()
    native_count = sum(value.numel() for value in model.parameters())
    inherited = _install_inherited(model)
    _activate_inherited(inherited)
    input_ids = torch.tensor([[2, 4]])
    inherited_output = model(input_ids).logits.detach().clone()
    combined, _audit, _pack, _hashes = _install_guided(inherited)
    with torch.no_grad():
        combined.guided_registry.q_left.fill_(0.3)
    assert not torch.equal(model(input_ids).logits, inherited_output)

    restored = combined.rollback_guided_only()
    assert restored is inherited
    assert inherited.is_applied
    assert torch.equal(model(input_ids).logits, inherited_output)
    assert not hasattr(model, combined.report.owner_name)

    inherited.rollback()
    assert sum(value.numel() for value in model.parameters()) == native_count

    model = _ToyModel()
    native_count = sum(value.numel() for value in model.parameters())
    combined, _audit, _pack, _hashes = _install_guided(_install_inherited(model))
    with torch.no_grad():
        combined.guided_registry.k_left[0, 0, 0] = float("nan")
    with pytest.raises(DenseGQAQKVOResidualError):
        combined.assert_live()
    combined.rollback()
    assert sum(value.numel() for value in model.parameters()) == native_count
    assert all(type(attention.q_proj) is nn.Linear for attention in model.attentions)


def test_sequential_bake_matches_live_products_and_restores_native(
    toy_stage1: None,
) -> None:
    del toy_stage1
    torch.manual_seed(53)
    model = _ToyModel()
    native_count = sum(value.numel() for value in model.parameters())
    native_weights = {
        (layer, role): getattr(model.attentions[layer], f"{role}_proj")
        .weight.detach()
        .clone()
        for layer in range(3)
        for role in ROLES
    }
    inherited = _install_inherited(model)
    _activate_inherited(inherited)
    combined, _audit, _pack, _hashes = _install_guided(inherited)
    with torch.no_grad():
        for role_index, role in enumerate(ROLES, start=1):
            getattr(combined.guided_registry, f"{role}_left").fill_(role_index * 0.04)
    input_ids = torch.tensor([[1, 3, 5]])
    live_output = model(input_ids).logits.detach().clone()

    bake = combined.bake_for_publication()

    torch.testing.assert_close(
        model(input_ids).logits,
        live_output,
        rtol=2e-5,
        atol=2e-5,
    )
    assert bake.adapter_parameters_removed == 528
    assert bake.inherited_adapter_parameters_removed == 264
    assert bake.guided_adapter_parameters_removed == 264
    assert bake.affected_slice_count == 48
    assert sum(value.numel() for value in model.parameters()) == native_count
    assert not hasattr(model, combined.report.owner_name)
    assert not hasattr(model, inherited.report.owner_name)
    assert all(
        type(getattr(attention, f"{role}_proj")) is nn.Linear
        for attention in model.attentions
        for role in ROLES
    )
    assert bake.as_dict()["factor_concatenation_used"] is False
    assert bake.as_dict()["bake_rule"][0].endswith("inherited_left @ inherited_right")

    combined.restore_prepublication_native(expected_model_parameter_count=native_count)
    assert all(
        torch.equal(
            getattr(model.attentions[layer], f"{role}_proj").weight,
            native_weights[(layer, role)],
        )
        for layer in range(3)
        for role in ROLES
    )


def test_real_transaction_bakes_nested_lexical_then_guided_composite(
    toy_stage1: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the accepted publication order with the real adapter bakes."""

    del toy_stage1

    class _LexicalToyModel(_ToyModel):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = nn.Linear(8, 16, bias=False)

    class _NoopSwiGLUHandle:
        def __init__(self) -> None:
            self.registry = SimpleNamespace(starts_as_exact_zero_correction=True)
            self._state = "live"
            self.bake_report = None

        @property
        def is_applied(self) -> bool:
            return self._state == "live"

        @property
        def is_baked(self) -> bool:
            return self._state == "baked"

        def assert_live(self) -> None:
            assert self.is_applied

        assert_structurally_live = assert_live

        def bake_for_publication(self):
            self._state = "baked"
            report = SimpleNamespace(
                as_dict=lambda: {
                    "schema": "test-noop-swiglu-bake",
                    "published_adapter_parameter_count": 0,
                }
            )
            self.bake_report = report
            return report

        def rollback(self) -> None:
            self._state = "rolled-back"

        def restore_prepublication_native(self) -> None:
            self._state = "rolled-back"

    class _Certificate:
        structure_sha256 = "1" * 64
        state_key_sha256 = "2" * 64
        state_key_count = 1

        @staticmethod
        def as_dict():
            return {
                "schema": "test-native-certificate",
                "structure_sha256": "1" * 64,
                "state_key_sha256": "2" * 64,
                "state_key_count": 1,
            }

    torch.manual_seed(59)
    model = _LexicalToyModel()
    native_count = sum(value.numel() for value in model.parameters())
    native_child_state_sha256 = progressive_bridge._dense_model_state_sha256(
        model
    )
    native_tensor_guards = progressive_bridge._capture_native_tensor_guards(
        model
    )
    inherited = _install_inherited(model)
    lexical = install_dense_lexical_residual_adapter(
        model,
        model.embedding,
        model.lm_head,
        rank=1,
    )
    combined, _audit, _pack, _hashes = _install_guided(inherited)
    swiglu = _NoopSwiGLUHandle()
    certificate = _Certificate()
    transaction = progressive_bridge.ExactTargetCompilerTransaction(
        model=model,
        exact_unique_parameters=native_count,
        topology=SimpleNamespace(layout=object()),
        native_before=certificate,
        native_child_state_sha256=native_child_state_sha256,
        native_tensor_guards=native_tensor_guards,
        qkvo_handle=combined,
        swiglu_handle=swiglu,
        lexical_handle=lexical,
        stage=None,
        parent_compiler_report_sha256="3" * 64,
    )
    with torch.no_grad():
        lexical.registry.embedding_left.fill_(0.01)
        lexical.registry.lm_head_left.fill_(0.02)
        combined.guided_registry.q_left.fill_(0.03)
    monkeypatch.setattr(
        progressive_bridge,
        "certify_native_dense_gqa_target",
        lambda *_args, **_kwargs: certificate,
    )
    receipt = progressive_bridge.ExternalBehaviorGateReceipt(
        evidence_sha256="4" * 64,
        evaluation_contract_sha256="5" * 64,
        heldout_action_count=1,
        free_running_rollout_count=1,
        passed=True,
        internal_similarity_only=False,
    )

    publication = transaction.bake_for_publication(receipt)

    assert publication["guided_registry_suspended_for_lexical_bake"] is True
    assert publication["dense_qkvo_bake"]["schema"] == (
        guided_qkvo.GUIDED_DENSE_GQA_QKVO_BAKE_SCHEMA
    )
    assert sum(value.numel() for value in model.parameters()) == native_count
    assert not hasattr(model, combined.report.owner_name)
    assert not hasattr(model, inherited.report.owner_name)
    assert not hasattr(model, lexical.report.owner_name)
    assert all(
        type(getattr(attention, f"{role}_proj")) is nn.Linear
        for attention in model.attentions
        for role in ROLES
    )


def test_nested_lexical_then_composite_rejection_restores_exact_native(
    toy_stage1: None,
) -> None:
    """Mirror the guided transaction's reverse-order prepublication rollback."""

    del toy_stage1

    class _LexicalToyModel(_ToyModel):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = nn.Linear(8, 16, bias=False)

    torch.manual_seed(61)
    model = _LexicalToyModel()
    native_count = sum(value.numel() for value in model.parameters())
    native_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    inherited = _install_inherited(model)
    lexical = install_dense_lexical_residual_adapter(
        model,
        model.embedding,
        model.lm_head,
        rank=1,
    )
    combined, _audit, _pack, _hashes = _install_guided(inherited)
    with torch.no_grad():
        lexical.registry.embedding_left.fill_(0.2)
        combined.guided_registry.v_left.fill_(0.3)

    with combined.suspend_guided_registry_owner_for_nested_publication():
        lexical.rollback()
    combined.rollback()

    assert sum(value.numel() for value in model.parameters()) == native_count
    assert set(model.state_dict()) == set(native_state)
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in native_state.items()
    )
    assert not hasattr(model, combined.report.owner_name)
    assert not hasattr(model, inherited.report.owner_name)
    assert not hasattr(model, lexical.report.owner_name)


def test_native_restore_releases_failed_bake_projection_clones(
    toy_stage1: None,
) -> None:
    del toy_stage1
    model = _ToyModel()
    combined, _audit, _pack, _hashes = _install_guided(_install_inherited(model))

    combined.bake_for_publication()
    replacement_references = tuple(
        weakref.ref(projection)
        for projection in combined._published_projections.values()
    )
    assert replacement_references
    assert all(reference() is not None for reference in replacement_references)

    combined.restore_prepublication_native()
    gc.collect()

    assert combined._published_projections == {}
    assert combined.inherited_handle._published_projections == {}
    assert all(reference() is None for reference in replacement_references)


def test_failed_bake_keeps_both_products_live_and_base_exact(
    toy_stage1: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del toy_stage1
    model = _ToyModel()
    inherited = _install_inherited(model)
    combined, _audit, _pack, _hashes = _install_guided(inherited)
    originals = {
        key: projection.weight.detach().clone()
        for key, projection in inherited._original_projections.items()
    }
    real_count = guided_qkvo._unique_parameter_count
    calls = 0

    def wrong_after_materialization(module: nn.Module) -> int:
        nonlocal calls
        calls += 1
        result = real_count(module)
        return result + (1 if calls >= 2 else 0)

    monkeypatch.setattr(
        guided_qkvo, "_unique_parameter_count", wrong_after_materialization
    )
    with pytest.raises(DenseGQAQKVOResidualError):
        combined.bake_for_publication()
    monkeypatch.setattr(guided_qkvo, "_unique_parameter_count", real_count)

    combined.assert_live()
    assert all(
        isinstance(
            getattr(model.attentions[layer], f"{role}_proj"),
            DenseGQAQKVOResidualProjection,
        )
        for layer in range(3)
        for role in ROLES
    )
    assert all(
        torch.equal(projection.weight, originals[key])
        for key, projection in inherited._original_projections.items()
    )
    combined.rollback()


def test_audit_basis_and_unauthorized_tensor_tampering_fail_closed(
    toy_stage1: None,
) -> None:
    del toy_stage1
    model = _ToyModel()
    inherited = _install_inherited(model)
    audit, pack, hashes = _source_payload(inherited)

    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="audit v2"):
        install_guided_dense_gqa_qkvo_residual_tranche(
            inherited,
            audit_report=audit,
            basis_pack=pack,
            **{**hashes, "expected_audit_report_sha256": "f" * 64},
        )

    tampered_pack = copy.deepcopy(pack)
    tampered_pack["factor_tensors"]["q_orthogonal_right"][0, 0, 0] += 1.0
    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="payload"):
        install_guided_dense_gqa_qkvo_residual_tranche(
            inherited,
            audit_report=audit,
            basis_pack=tampered_pack,
            **hashes,
        )

    unauthorized_audit = copy.deepcopy(audit)
    unauthorized_audit["authorized_roles"] = ["q"]
    unauthorized_audit["installable_roles"] = ["q"]
    unauthorized_audit.pop("report_sha256")
    unauthorized_sha256 = guided_qkvo._canonical_sha256(unauthorized_audit)
    unauthorized_audit["report_sha256"] = unauthorized_sha256
    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="all four"):
        install_guided_dense_gqa_qkvo_residual_tranche(
            inherited,
            audit_report=unauthorized_audit,
            basis_pack=pack,
            **{
                **hashes,
                "expected_audit_report_sha256": unauthorized_sha256,
            },
        )

    extra_tensor = copy.deepcopy(pack)
    extra_tensor["factor_tensors"]["unauthorized_right"] = torch.ones(1)
    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="tensor names"):
        install_guided_dense_gqa_qkvo_residual_tranche(
            inherited,
            audit_report=audit,
            basis_pack=extra_tensor,
            **hashes,
        )

    reordered_audit = copy.deepcopy(audit)
    reordered_audit["site_indices"] = list(reversed(reordered_audit["site_indices"]))
    reordered_audit["site_indices_sha256"] = guided_qkvo._canonical_sha256(
        reordered_audit["site_indices"]
    )
    reordered_audit.pop("report_sha256")
    reordered_sha256 = guided_qkvo._canonical_sha256(reordered_audit)
    reordered_audit["report_sha256"] = reordered_sha256
    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="ordered 120-site"):
        install_guided_dense_gqa_qkvo_residual_tranche(
            inherited,
            audit_report=reordered_audit,
            basis_pack=pack,
            **{
                **hashes,
                "expected_audit_report_sha256": reordered_sha256,
            },
        )
    inherited.rollback()


def test_live_inherited_factor_tamper_breaks_audit_binding(
    toy_stage1: None,
) -> None:
    del toy_stage1
    model = _ToyModel()
    inherited = _install_inherited(model)
    audit, pack, hashes = _source_payload(inherited)
    with torch.no_grad():
        inherited.adapter_registry.q_right[0, 0, 0].add_(0.5)

    with pytest.raises(DenseGQAQKVOResidualTopologyError, match="audited product"):
        install_guided_dense_gqa_qkvo_residual_tranche(
            inherited,
            audit_report=audit,
            basis_pack=pack,
            **hashes,
        )
    inherited.rollback()
