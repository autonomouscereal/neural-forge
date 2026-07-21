from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack  # noqa: E402
from neural_forge.hf_tied_causal_interface_anchor_fold import (  # noqa: E402
    TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
)
from neural_forge.hf_tied_causal_interface_fold import (  # noqa: E402
    TiedCausalInterfaceFold,
    whiten_tied_causal_interface,
)
from neural_forge.hf_tied_residual_congruence import (  # noqa: E402
    TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE,
    TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    TiedResidualCongruenceError,
    compile_tied_residual_congruence_candidate,
)


_BASE_INTERFACE_SCHEMA = "neural-forge-tied-causal-interface-fold-v1"


def _interface(
    target,
    encoder,
    *,
    schema: str = _BASE_INTERFACE_SCHEMA,
) -> TiedCausalInterfaceFold:
    tensor_sha256 = (
        stack._anchor_interface_tensor_sha256
        if schema == TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA
        else postfold._interface_tensor_sha256
    )
    body = {
        "schema": schema,
        "target_weight_sha256": tensor_sha256(target),
        "producer_encoder_sha256": tensor_sha256(encoder),
    }
    return TiedCausalInterfaceFold(
        target_weight=target,
        producer_encoder=encoder,
        report={**body, "report_sha256": canonical_sha256(body)},
    )


def _synthetic_inputs():
    generator = torch.Generator().manual_seed(720_021)
    source = torch.randn((13, 7), generator=generator, dtype=torch.float64)
    target = torch.randn((13, 4), generator=generator, dtype=torch.float64)
    encoder = torch.randn((7, 4), generator=generator, dtype=torch.float64)
    return source, _interface(target, encoder)


def test_strong_candidate_enforces_global_tied_congruence_and_keeps_control():
    source, interface = _synthetic_inputs()

    control = compile_tied_residual_congruence_candidate(
        interface,
        source,
        mode=TIED_RESIDUAL_CURRENT_SELF_DUAL_MODE,
    )
    strong = compile_tied_residual_congruence_candidate(
        interface,
        source,
        mode=TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    )

    torch.testing.assert_close(
        control.interface.producer_encoder,
        strong.interface.producer_encoder,
        rtol=0.0,
        atol=0.0,
    )
    basis = strong.interface.producer_encoder
    expected = source @ basis
    torch.testing.assert_close(
        strong.interface.target_weight,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        basis.T @ basis,
        torch.eye(4, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    assert strong.report["fp64_congruence_exact"] is True
    assert strong.report["bf16_congruence_exact"] is True
    assert strong.report["producer_basis_orthogonality_authenticated"] is True
    assert strong.report["selected_target_is_strong_congruence"] is True
    assert control.report["selected_target_is_current_selfdual_control"] is True
    assert control.report["selected_vs_old_output_factor_relative_error"] == 0.0
    assert strong.report["current_target_to_congruent_relative_error"] > 0.01
    assert strong.report["congruent_vs_old_output_factor_relative_error"] > 0.01
    assert torch.equal(
        strong.materialized_target_weight,
        expected.to(torch.bfloat16),
    )
    assert all(value == 0 for value in strong.report["operation_audit"].values())


def test_source_interface_tensor_lineage_is_authenticated_before_reissue():
    source, interface = _synthetic_inputs()
    self_dual = whiten_tied_causal_interface(interface)

    valid = compile_tied_residual_congruence_candidate(self_dual, source)
    assert (
        valid.report["source_interface_report_sha256"]
        == self_dual.report["report_sha256"]
    )

    dishonest_body = dict(self_dual.report)
    dishonest_body.pop("report_sha256")
    dishonest_body["producer_encoder_sha256"] = "0" * 64
    dishonest = TiedCausalInterfaceFold(
        target_weight=self_dual.target_weight,
        producer_encoder=self_dual.producer_encoder,
        report={
            **dishonest_body,
            "report_sha256": canonical_sha256(dishonest_body),
        },
    )
    with pytest.raises(
        TiedResidualCongruenceError,
        match="Source interface tensor lineage is not authentic",
    ):
        compile_tied_residual_congruence_candidate(dishonest, source)


def test_anchor_source_interface_uses_its_schema_specific_tensor_seal():
    source, interface = _synthetic_inputs()
    anchor = _interface(
        interface.target_weight,
        interface.producer_encoder,
        schema=TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
    )

    candidate = compile_tied_residual_congruence_candidate(anchor, source)

    assert (
        candidate.report["source_interface_report_sha256"]
        == anchor.report["report_sha256"]
    )


class _NativeTiedModel(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.weight)


def test_strong_candidate_installs_transactionally_without_topology_change(
    monkeypatch,
):
    source, interface = _synthetic_inputs()
    model = _NativeTiedModel(interface.target_weight.to(torch.bfloat16))
    model.weight.grad = torch.full_like(model.weight, 0.125)
    parameter_id = id(model.weight)
    storage_pointer = int(model.weight.untyped_storage().data_ptr())
    gradient_id = id(model.weight.grad)
    parameter_count = sum(int(value.numel()) for value in model.parameters())
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda value: SimpleNamespace(
            identity_sha256=canonical_sha256(postfold._state_tensor_hashes(value))
        ),
    )

    selected, receipt = postfold.install_lfm2_tied_residual_congruence_gauge(
        model,
        interface,
        source,
        mode=TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    )

    expected = (source @ selected.producer_encoder).to(torch.bfloat16)
    assert torch.equal(model.weight.detach(), expected)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert id(model.weight) == parameter_id
    assert int(model.weight.untyped_storage().data_ptr()) == storage_pointer
    assert id(model.weight.grad) == gradient_id
    assert sum(int(value.numel()) for value in model.parameters()) == parameter_count
    assert receipt["mode"] == TIED_RESIDUAL_STRONG_CONGRUENCE_MODE
    assert receipt["fp64_congruence_exact"] is True
    assert receipt["bf16_congruence_exact"] is True
    assert receipt["native_tie_preserved"] is True
    assert receipt["native_tied_alias_set_preserved"] is True
    assert receipt["parameter_topology_preserved"] is True
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["unique_parameter_count_preserved"] is True
    assert all(value == 0 for value in receipt["operation_audit"].values())


def test_terminal_recomputes_congruence_and_binds_real_layer_lineage() -> None:
    source, interface = _synthetic_inputs()
    candidate = compile_tied_residual_congruence_candidate(
        interface,
        source,
        mode=TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    )
    model = _NativeTiedModel(candidate.materialized_target_weight)
    lineage = stack._validated_interface_lineage(model, candidate.interface)

    receipt = postfold._authenticate_fused_terminal_tied_interface(
        target_model=model,
        interface=candidate.interface,
        mapped_parent_tied_table=source,
        layer_receipt={"tied_interface_lineage": lineage},
    )

    assert receipt["interface_report_schema"] == (candidate.interface.report["schema"])
    assert receipt["strong_congruence_recomputed_exact"] is True
    assert receipt["global_residual_gauge_frozen_through_terminal"] is True

    spoofed_lineage = dict(lineage)
    spoofed_lineage["interface_report_schema"] = (
        "neural-forge-tied-causal-interface-self-dual-gauge-v1"
    )
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="differs from the frozen layer-stack congruence",
    ):
        postfold._authenticate_fused_terminal_tied_interface(
            target_model=model,
            interface=candidate.interface,
            mapped_parent_tied_table=source,
            layer_receipt={"tied_interface_lineage": spoofed_lineage},
        )
