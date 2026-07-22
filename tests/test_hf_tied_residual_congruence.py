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
    TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
    TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    TiedResidualAnchorTransportWitness,
    TiedResidualCongruenceError,
    authenticate_tied_residual_congruence_interface,
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


def _anchor_witness(selected, base, residual):
    return TiedResidualAnchorTransportWitness.from_source_interface(
        selected,
        anchor_base_target_weight=base,
        anchor_source_residual_weight=residual,
    )


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


def test_candidate_seven_anchor_residual_is_transported_after_strong_base():
    source, base_interface = _synthetic_inputs()
    generator = torch.Generator().manual_seed(720_077)
    residual = 0.25 * torch.randn(
        base_interface.target_weight.shape,
        generator=generator,
        dtype=torch.float64,
    )
    selected = _interface(
        base_interface.target_weight + residual,
        base_interface.producer_encoder,
        schema=TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
    )

    pure_strong = compile_tied_residual_congruence_candidate(
        selected,
        source,
        mode=TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    )
    hybrid = compile_tied_residual_congruence_candidate(
        selected,
        source,
        mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
        anchor_base_target_weight=base_interface.target_weight,
        anchor_source_residual_weight=residual,
    )

    basis = pure_strong.interface.producer_encoder
    right_stretch = basis.T @ selected.producer_encoder
    expected_residual = residual @ right_stretch
    expected = (source @ basis) + expected_residual
    torch.testing.assert_close(
        hybrid.interface.producer_encoder,
        basis,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        hybrid.interface.target_weight,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        hybrid.interface.target_weight,
        pure_strong.interface.target_weight,
    )
    assert hybrid.report["selected_target_is_strong_base_plus_anchor_residual"] is True
    assert hybrid.report["fp64_congruence_exact"] is False
    assert hybrid.report["fp64_anchor_residual_formula_exact"] is True
    authentication = authenticate_tied_residual_congruence_interface(
        hybrid.interface,
        mapped_parent_tied_table=source,
        anchor_transport_witness=_anchor_witness(
            selected,
            base_interface.target_weight,
            residual,
        ),
    )
    assert authentication["strong_base_plus_anchor_residual_claimed"] is True
    assert authentication["strong_congruence_recomputed_exact"] is False
    assert authentication["anchor_residual_formula_recomputed_exact"] is True
    assert hybrid.report["fp64_selected_formula_relative_error"] == 0.0
    assert hybrid.report["fp64_congruence_relative_error"] > 0.0


def test_candidate_zero_anchor_residual_reduces_exactly_to_pure_strong():
    source, base_interface = _synthetic_inputs()
    residual = torch.zeros_like(base_interface.target_weight)
    selected = _interface(
        base_interface.target_weight + residual,
        base_interface.producer_encoder,
        schema=TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
    )

    pure_strong = compile_tied_residual_congruence_candidate(
        selected,
        source,
        mode=TIED_RESIDUAL_STRONG_CONGRUENCE_MODE,
    )
    hybrid = compile_tied_residual_congruence_candidate(
        selected,
        source,
        mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
        anchor_base_target_weight=base_interface.target_weight,
        anchor_source_residual_weight=residual,
    )

    assert torch.equal(
        hybrid.interface.producer_encoder,
        pure_strong.interface.producer_encoder,
    )
    assert torch.equal(
        hybrid.interface.target_weight,
        pure_strong.interface.target_weight,
    )
    assert torch.equal(
        hybrid.materialized_target_weight,
        pure_strong.materialized_target_weight,
    )
    assert hybrid.report["anchor_residual_transport_applied"] is True
    assert hybrid.report["fp64_congruence_exact"] is True
    assert hybrid.report["bf16_congruence_exact"] is True
    authentication = authenticate_tied_residual_congruence_interface(
        hybrid.interface,
        mapped_parent_tied_table=source,
        anchor_transport_witness=_anchor_witness(
            selected,
            base_interface.target_weight,
            residual,
        ),
    )
    assert authentication["strong_base_plus_anchor_residual_claimed"] is True
    assert authentication["strong_congruence_recomputed_exact"] is True
    assert authentication["anchor_residual_formula_recomputed_exact"] is True


def test_distinct_anchor_residuals_produce_distinct_layer_body_input_lineage():
    source, base_interface = _synthetic_inputs()
    residuals = (
        torch.full_like(base_interface.target_weight, 0.125),
        torch.full_like(base_interface.target_weight, -0.25),
    )
    lineages = []
    for residual in residuals:
        selected = _interface(
            base_interface.target_weight + residual,
            base_interface.producer_encoder,
            schema=TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
        )
        candidate = compile_tied_residual_congruence_candidate(
            selected,
            source,
            mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
            anchor_base_target_weight=base_interface.target_weight,
            anchor_source_residual_weight=residual,
        )
        model = _NativeTiedModel(candidate.materialized_target_weight)
        lineages.append(
            stack._validated_interface_lineage(
                model,
                candidate.interface,
                mapped_parent_tied_table=source,
                anchor_transport_witness=_anchor_witness(
                    selected,
                    base_interface.target_weight,
                    residual,
                ),
            )
        )

    assert lineages[0]["target_weight_sha256"] != lineages[1][
        "target_weight_sha256"
    ]
    assert lineages[0]["interface_report_sha256"] != lineages[1][
        "interface_report_sha256"
    ]


def test_resealed_arbitrary_hybrid_target_fails_exact_transport_recomputation():
    source, base_interface = _synthetic_inputs()
    residual = torch.full_like(base_interface.target_weight, 0.125)
    selected = _interface(
        base_interface.target_weight + residual,
        base_interface.producer_encoder,
        schema=TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
    )
    candidate = compile_tied_residual_congruence_candidate(
        selected,
        source,
        mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
        anchor_base_target_weight=base_interface.target_weight,
        anchor_source_residual_weight=residual,
    )
    forged_target = candidate.interface.target_weight.clone()
    forged_target[0, 0] += 0.5
    strong_base = source @ candidate.interface.producer_encoder
    realized = forged_target - strong_base
    body = dict(candidate.interface.report)
    body.pop("report_sha256")
    body["target_weight_sha256"] = postfold._interface_tensor_sha256(forged_target)
    body["anchor_realized_selfdual_residual_sha256"] = (
        postfold._interface_tensor_sha256(realized)
    )
    body["fp64_congruence_exact"] = bool(torch.equal(forged_target, strong_base))
    body["bf16_congruence_exact"] = bool(
        torch.equal(forged_target.to(torch.bfloat16), strong_base.to(torch.bfloat16))
    )
    body["fp64_congruence_relative_error"] = float(
        torch.linalg.vector_norm(realized).item()
        / torch.linalg.vector_norm(strong_base).item()
    )
    body["fp64_congruence_maximum_absolute_error"] = float(
        realized.abs().max().item()
    )
    forged = TiedCausalInterfaceFold(
        target_weight=forged_target,
        producer_encoder=candidate.interface.producer_encoder,
        report={**body, "report_sha256": canonical_sha256(body)},
    )

    with pytest.raises(
        TiedResidualCongruenceError,
        match="delta @ \\(B.T @ E\\)",
    ):
        authenticate_tied_residual_congruence_interface(
            forged,
            mapped_parent_tied_table=source,
            anchor_transport_witness=_anchor_witness(
                selected,
                base_interface.target_weight,
                residual,
            ),
        )


def test_resealed_orthonormal_basis_fails_recomputed_whitening_identity():
    source, base_interface = _synthetic_inputs()
    residual = torch.full_like(base_interface.target_weight, 0.125)
    selected = _interface(
        base_interface.target_weight + residual,
        base_interface.producer_encoder,
        schema=TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
    )
    candidate = compile_tied_residual_congruence_candidate(
        selected,
        source,
        mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
        anchor_base_target_weight=base_interface.target_weight,
        anchor_source_residual_weight=residual,
    )
    rotation = torch.eye(4, dtype=torch.float64)
    rotation[0, 0] = -1.0
    forged_basis = candidate.interface.producer_encoder @ rotation
    strong_base = source @ forged_basis
    transported = residual @ (forged_basis.T @ selected.producer_encoder)
    forged_target = strong_base + transported
    realized = forged_target - strong_base
    body = dict(candidate.interface.report)
    body.pop("report_sha256")
    body.update(
        {
            "producer_encoder_sha256": postfold._interface_tensor_sha256(
                forged_basis
            ),
            "strong_base_producer_encoder_sha256": (
                postfold._interface_tensor_sha256(forged_basis)
            ),
            "strong_base_target_weight_sha256": (
                postfold._interface_tensor_sha256(strong_base)
            ),
            "congruent_target_weight_sha256": (
                postfold._interface_tensor_sha256(strong_base)
            ),
            "target_weight_sha256": postfold._interface_tensor_sha256(
                forged_target
            ),
            "anchor_selfdual_residual_sha256": (
                postfold._interface_tensor_sha256(transported)
            ),
            "anchor_realized_selfdual_residual_sha256": (
                postfold._interface_tensor_sha256(realized)
            ),
            "fp64_congruence_exact": bool(
                torch.equal(forged_target, strong_base)
            ),
            "bf16_congruence_exact": bool(
                torch.equal(
                    forged_target.to(torch.bfloat16),
                    strong_base.to(torch.bfloat16),
                )
            ),
            "fp64_congruence_relative_error": float(
                torch.linalg.vector_norm(realized).item()
                / torch.linalg.vector_norm(strong_base).item()
            ),
            "fp64_congruence_maximum_absolute_error": float(
                realized.abs().max().item()
            ),
        }
    )
    forged = TiedCausalInterfaceFold(
        target_weight=forged_target,
        producer_encoder=forged_basis,
        report={**body, "report_sha256": canonical_sha256(body)},
    )

    with pytest.raises(
        TiedResidualCongruenceError,
        match="delta @ \\(B.T @ E\\)",
    ):
        authenticate_tied_residual_congruence_interface(
            forged,
            mapped_parent_tied_table=source,
            anchor_transport_witness=_anchor_witness(
                selected,
                base_interface.target_weight,
                residual,
            ),
        )


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


def test_hybrid_terminal_requires_exact_publication_time_transport_proof() -> None:
    source, base_interface = _synthetic_inputs()
    residual = torch.full_like(base_interface.target_weight, 0.125)
    selected = _interface(
        base_interface.target_weight + residual,
        base_interface.producer_encoder,
        schema=TIED_CAUSAL_ANCHOR_RESIDUAL_SCHEMA,
    )
    candidate = compile_tied_residual_congruence_candidate(
        selected,
        source,
        mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
        anchor_base_target_weight=base_interface.target_weight,
        anchor_source_residual_weight=residual,
    )
    witness = candidate.anchor_transport_witness
    assert witness is not None
    model = _NativeTiedModel(candidate.materialized_target_weight)
    lineage = stack._validated_interface_lineage(
        model,
        candidate.interface,
        mapped_parent_tied_table=source,
        anchor_transport_witness=witness,
    )
    authentication = authenticate_tied_residual_congruence_interface(
        candidate.interface,
        mapped_parent_tied_table=source,
        anchor_transport_witness=witness,
    )
    layer_receipt = {
        "tied_interface_lineage": lineage,
        "anchor_residual_transport_authentication": dict(authentication),
        "anchor_residual_transport_authentication_sha256": authentication[
            "result_sha256"
        ],
    }

    receipt = postfold._authenticate_fused_terminal_tied_interface(
        target_model=model,
        interface=candidate.interface,
        mapped_parent_tied_table=source,
        layer_receipt=layer_receipt,
        anchor_transport_witness=witness,
    )
    assert receipt["interface_authentication_sha256"] == authentication[
        "result_sha256"
    ]

    stale_source = source.clone()
    stale_source[0, 0] += 0.25
    stale_candidate = compile_tied_residual_congruence_candidate(
        selected,
        stale_source,
        mode=TIED_RESIDUAL_STRONG_ANCHOR_RESIDUAL_MODE,
        anchor_base_target_weight=base_interface.target_weight,
        anchor_source_residual_weight=residual,
    )
    stale_authentication = authenticate_tied_residual_congruence_interface(
        stale_candidate.interface,
        mapped_parent_tied_table=stale_source,
        anchor_transport_witness=stale_candidate.anchor_transport_witness,
    )
    resealed_body = dict(authentication)
    resealed_body.pop("result_sha256")
    resealed_body["mapped_parent_tied_table_supplied"] = False
    resealed_authentication = {
        **resealed_body,
        "result_sha256": canonical_sha256(resealed_body),
    }
    invalid_publication_proofs = (
        {},
        {
            "anchor_residual_transport_authentication": dict(
                stale_authentication
            ),
            "anchor_residual_transport_authentication_sha256": (
                stale_authentication["result_sha256"]
            ),
        },
        {
            "anchor_residual_transport_authentication": resealed_authentication,
            "anchor_residual_transport_authentication_sha256": (
                resealed_authentication["result_sha256"]
            ),
        },
    )
    for invalid in invalid_publication_proofs:
        with pytest.raises(
            postfold.Lfm2SemanticPostfoldError,
            match="publication-time layer-stack proof",
        ):
            postfold._authenticate_fused_terminal_tied_interface(
                target_model=model,
                interface=candidate.interface,
                mapped_parent_tied_table=source,
                layer_receipt={"tied_interface_lineage": lineage, **invalid},
                anchor_transport_witness=witness,
            )
