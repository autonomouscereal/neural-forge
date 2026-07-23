from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as functional

from neural_forge.hf_lfm2_swiglu_tangent_bank import (
    Lfm2SwiGluTangentAuthenticationError,
    Lfm2SwiGluTangentBankError,
    build_lfm2_swiglu_tangent_bank,
    fingerprint_lfm2_swiglu_tangent_tensor,
)


def _cached_rows(
    w1: torch.Tensor,
    w3: torch.Tensor,
    normalized_inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = normalized_inputs @ w1.transpose(0, 1)
    up = normalized_inputs @ w3.transpose(0, 1)
    return gate, up, functional.silu(gate) * up


def _build(
    *,
    w1: torch.Tensor,
    w3: torch.Tensor,
    normalized_inputs: torch.Tensor,
    decoder: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    coordinate_rms: torch.Tensor | None = None,
    anchor_count: int = 3,
    compute_device: str = "cpu",
):
    gate, up, post = _cached_rows(w1, w3, normalized_inputs)
    return build_lfm2_swiglu_tangent_bank(
        site_id="layer.000.mlp",
        parent_w1=w1,
        parent_w1_sha256=fingerprint_lfm2_swiglu_tangent_tensor(w1),
        parent_w3=w3,
        parent_w3_sha256=fingerprint_lfm2_swiglu_tangent_tensor(w3),
        cached_gate_rows=gate,
        cached_gate_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(gate),
        cached_up_rows=up,
        cached_up_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(up),
        cached_post_rows=post,
        cached_post_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(post),
        target_to_parent_decoder=decoder,
        target_to_parent_decoder_sha256=(
            fingerprint_lfm2_swiglu_tangent_tensor(decoder)
        ),
        sample_weights=sample_weights,
        sample_weights_sha256=(
            None
            if sample_weights is None
            else fingerprint_lfm2_swiglu_tangent_tensor(sample_weights)
        ),
        child_input_coordinate_rms=coordinate_rms,
        child_input_coordinate_rms_sha256=(
            None
            if coordinate_rms is None
            else fingerprint_lfm2_swiglu_tangent_tensor(coordinate_rms)
        ),
        anchor_count=anchor_count,
        compute_device=compute_device,
    )


def test_exact_analytic_jvp_matches_centered_finite_difference_diagnostic() -> None:
    torch.manual_seed(11)
    w1 = torch.randn(5, 3, dtype=torch.float64)
    w3 = torch.randn(5, 3, dtype=torch.float64)
    source = torch.randn(1, 3, dtype=torch.float64)
    decoder = torch.randn(2, 3, dtype=torch.float64)
    bank = _build(
        w1=w1,
        w3=w3,
        normalized_inputs=source,
        decoder=decoder,
        anchor_count=1,
    )

    epsilon = 1.0e-6
    diagnostics = []
    for direction in decoder:
        plus_gate, plus_up, plus_post = _cached_rows(
            w1, w3, source + epsilon * direction
        )
        minus_gate, minus_up, minus_post = _cached_rows(
            w1, w3, source - epsilon * direction
        )
        del plus_gate, plus_up, minus_gate, minus_up
        diagnostics.append(((plus_post - minus_post) / (2.0 * epsilon))[0])
    finite_difference_diagnostic = torch.stack(diagnostics)
    assert torch.allclose(
        bank.tangent_rows,
        finite_difference_diagnostic,
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    assert bank.operation_audit["backward_calls"] == 0
    assert bank.operation_audit["autograd_grad_calls"] == 0


def test_decoder_rows_are_directions_with_no_parent_identity_fallback() -> None:
    w1 = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]], dtype=torch.float64)
    w3 = torch.tensor([[0.5, -1.0], [2.0, 1.0], [-0.5, 0.25]], dtype=torch.float64)
    source = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    decoder = torch.eye(2, dtype=torch.float64)
    bank = _build(
        w1=w1,
        w3=w3,
        normalized_inputs=source,
        decoder=decoder,
        anchor_count=1,
    )
    gate, up, _post = _cached_rows(w1, w3, source)
    sigmoid = torch.sigmoid(gate[0])
    derivative = sigmoid * (1.0 + gate[0] * (1.0 - sigmoid))
    expected = w1.transpose(0, 1) * (derivative * up[0]).unsqueeze(0)
    expected += w3.transpose(0, 1) * functional.silu(gate[0]).unsqueeze(0)
    assert torch.equal(bank.tangent_rows, expected)
    assert bank.operation_audit["authenticated_decoder_direction_rows"] == 2
    assert bank.operation_audit["synthesized_parent_identity_direction_rows"] == 0

    wrong_orientation = torch.ones(2, 1, dtype=torch.float64)
    with pytest.raises(Lfm2SwiGluTangentBankError, match="child_width, parent_hidden"):
        _build(
            w1=w1,
            w3=w3,
            normalized_inputs=source,
            decoder=wrong_orientation,
            anchor_count=1,
        )


def test_authentication_is_strict_for_all_optional_and_required_sources() -> None:
    w1 = torch.eye(2, dtype=torch.float64)
    w3 = torch.tensor([[1.0, 0.5], [0.25, 1.0]], dtype=torch.float64)
    source = torch.tensor([[0.5, 1.0]], dtype=torch.float64)
    decoder = torch.eye(2, dtype=torch.float64)
    gate, up, post = _cached_rows(w1, w3, source)
    authenticated_gate = fingerprint_lfm2_swiglu_tangent_tensor(gate)
    gate[0, 0] += 1.0
    with pytest.raises(
        Lfm2SwiGluTangentAuthenticationError,
        match="cached_gate_rows differs",
    ):
        build_lfm2_swiglu_tangent_bank(
            site_id="layer.000.mlp",
            parent_w1=w1,
            parent_w1_sha256=fingerprint_lfm2_swiglu_tangent_tensor(w1),
            parent_w3=w3,
            parent_w3_sha256=fingerprint_lfm2_swiglu_tangent_tensor(w3),
            cached_gate_rows=gate,
            cached_gate_rows_sha256=authenticated_gate,
            cached_up_rows=up,
            cached_up_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(up),
            cached_post_rows=post,
            cached_post_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(post),
            target_to_parent_decoder=decoder,
            target_to_parent_decoder_sha256=(
                fingerprint_lfm2_swiglu_tangent_tensor(decoder)
            ),
            anchor_count=1,
        )

    coordinate_rms = torch.ones(2, dtype=torch.float64)
    with pytest.raises(
        Lfm2SwiGluTangentAuthenticationError,
        match="coordinate_rms_sha256 is required",
    ):
        build_lfm2_swiglu_tangent_bank(
            site_id="layer.000.mlp",
            parent_w1=w1,
            parent_w1_sha256=fingerprint_lfm2_swiglu_tangent_tensor(w1),
            parent_w3=w3,
            parent_w3_sha256=fingerprint_lfm2_swiglu_tangent_tensor(w3),
            cached_gate_rows=gate,
            cached_gate_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(gate),
            cached_up_rows=up,
            cached_up_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(up),
            cached_post_rows=post,
            cached_post_rows_sha256=fingerprint_lfm2_swiglu_tangent_tensor(post),
            target_to_parent_decoder=decoder,
            target_to_parent_decoder_sha256=(
                fingerprint_lfm2_swiglu_tangent_tensor(decoder)
            ),
            child_input_coordinate_rms=coordinate_rms,
            anchor_count=1,
        )


def test_build_is_deterministic_and_preserves_every_source_tensor() -> None:
    torch.manual_seed(19)
    w1 = torch.randn(4, 3)
    w3 = torch.randn(4, 3)
    source = torch.randn(5, 3)
    decoder = torch.randn(2, 3)
    weights = torch.tensor([1000.0, 1.0, 2.0, 3.0, 0.001])
    coordinate_rms = torch.tensor([0.25, 2.0])
    inputs = (w1, w3, source, decoder, weights, coordinate_rms)
    before = tuple(value.clone() for value in inputs)
    first = _build(
        w1=w1,
        w3=w3,
        normalized_inputs=source,
        decoder=decoder,
        sample_weights=weights,
        coordinate_rms=coordinate_rms,
        anchor_count=3,
    )
    second = _build(
        w1=w1,
        w3=w3,
        normalized_inputs=source,
        decoder=decoder,
        sample_weights=weights,
        coordinate_rms=coordinate_rms,
        anchor_count=3,
    )
    assert first.anchor_indices == (0, 2, 4)
    assert first.bank_sha256 == second.bank_sha256
    assert first.as_dict() == second.as_dict()
    assert torch.equal(first.tangent_rows, second.tangent_rows)
    assert first.tangent_rows.device.type == "cpu"
    assert first.tangent_rows.dtype == torch.float64
    assert not first.tangent_rows.requires_grad
    assert first.tangent_row_weights.tolist() == pytest.approx(
        [1000.0, 1000.0, 2.0, 2.0, 0.001, 0.001]
    )
    assert all(torch.equal(value, snapshot) for value, snapshot in zip(inputs, before))
    audit = first.operation_audit
    for name in (
        "parent_forward_calls",
        "model_forward_calls",
        "generation_calls",
        "backward_calls",
        "autograd_grad_calls",
        "optimizer_constructions",
        "optimizer_steps",
        "training_steps",
        "source_tensor_mutations",
    ):
        assert audit[name] == 0
    assert audit["numeric_diagnostics_used_for_semantic_decision"] is False
    assert audit["numeric_diagnostics_used_for_auto_veto"] is False
    copied_receipt = copy.deepcopy(first.as_dict())
    assert copied_receipt["bank_sha256"] == first.bank_sha256
    compute = copied_receipt["compute_execution"]
    compute_body = dict(compute)
    compute_receipt_sha256 = compute_body.pop("receipt_sha256")
    assert compute["requested_device"] == "cpu"
    assert compute["resolved_device"] == "cpu"
    assert compute["authenticated_source_archive_device"] == "cpu"
    assert compute["materialized_tangent_bank_device"] == "cpu"
    assert compute["active_layer_staged_to_compute_device"] is False
    assert len(compute_receipt_sha256) == 64


@pytest.mark.parametrize(
    "compute_device",
    (
        "cuda",
        "cuda:",
        "cuda:00",
        "cuda:-1",
        "CUDA:0",
        "cuda:\N{ARABIC-INDIC DIGIT ZERO}",
    ),
)
def test_compute_device_rejects_noncanonical_cuda_names(
    compute_device: str,
) -> None:
    w1 = torch.eye(2, dtype=torch.float64)
    w3 = torch.eye(2, dtype=torch.float64)
    source = torch.ones(1, 2, dtype=torch.float64)
    decoder = torch.eye(2, dtype=torch.float64)
    with pytest.raises(Lfm2SwiGluTangentBankError, match="explicitly indexed"):
        _build(
            w1=w1,
            w3=w3,
            normalized_inputs=source,
            decoder=decoder,
            anchor_count=1,
            compute_device=compute_device,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compute_stages_arithmetic_and_returns_authenticated_cpu_bank() -> None:
    torch.manual_seed(23)
    w1 = torch.randn(5, 3, dtype=torch.float64)
    w3 = torch.randn(5, 3, dtype=torch.float64)
    source = torch.randn(3, 3, dtype=torch.float64)
    decoder = torch.randn(2, 3, dtype=torch.float64)
    cpu_bank = _build(
        w1=w1,
        w3=w3,
        normalized_inputs=source,
        decoder=decoder,
        anchor_count=2,
    )
    cuda_bank = _build(
        w1=w1,
        w3=w3,
        normalized_inputs=source,
        decoder=decoder,
        anchor_count=2,
        compute_device="cuda:0",
    )
    assert cuda_bank.tangent_rows.device.type == "cpu"
    assert cuda_bank.tangent_rows.dtype == torch.float64
    assert torch.allclose(
        cuda_bank.tangent_rows,
        cpu_bank.tangent_rows,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    compute = cuda_bank.as_dict()["compute_execution"]
    assert compute["requested_device"] == "cuda:0"
    assert compute["resolved_device"] == "cuda:0"
    assert compute["backend"] == "cuda"
    assert compute["active_layer_staged_to_compute_device"] is True
    assert compute["gpu_is_ephemeral_dense_algebra_workspace"] is True
    assert compute["authenticated_source_archive_device"] == "cpu"
    assert compute["materialized_tangent_bank_device"] == "cpu"
    assert compute["silent_device_fallback_performed"] is False


def test_recommended_scales_equalize_blocks_without_changing_raw_bank() -> None:
    w1 = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    w3 = torch.tensor([[0.5, 0.0], [0.0, 1.5]], dtype=torch.float64)
    source = torch.tensor([[0.5, 1.0], [1.0, 0.25], [0.25, 0.5]], dtype=torch.float64)
    decoder = torch.eye(2, dtype=torch.float64)
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    coordinate_rms = torch.tensor([0.25, 2.0], dtype=torch.float64)
    bank = _build(
        w1=w1,
        w3=w3,
        normalized_inputs=source,
        decoder=decoder,
        sample_weights=weights,
        coordinate_rms=coordinate_rms,
        anchor_count=3,
    )
    _gate, _up, post = _cached_rows(w1, w3, source)
    empirical_norm = torch.linalg.vector_norm(post * torch.sqrt(weights).unsqueeze(1))
    balanced_tangents = (
        bank.tangent_rows
        * torch.sqrt(bank.tangent_row_weights).unsqueeze(1)
        * bank.recommended_tangent_row_scales.unsqueeze(1)
    )
    assert torch.allclose(
        torch.linalg.vector_norm(balanced_tangents),
        empirical_norm,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    diagnostics = bank.advisory_diagnostics
    assert len(diagnostics["raw_tangent_frobenius_norm_by_anchor"]) == 3
    assert len(diagnostics["raw_tangent_frobenius_norm_by_decoder_direction"]) == 2
    assert diagnostics["used_for_semantic_gate"] is False
    assert diagnostics["snapshot_balance_recommendation"][
        "recommendation_is_not_automatically_applied"
    ]

    from neural_forge.hf_lfm2_mlp_qdeim import (
        Lfm2MlpQdeimFunctionalSnapshotBlock,
        lfm2_mlp_qdeim_functional_snapshot_evidence_sha256,
    )

    kwargs = bank.qdeim_functional_snapshot_kwargs(
        block_id="layer.000.child_lifted_tangent"
    )
    block = Lfm2MlpQdeimFunctionalSnapshotBlock(**kwargs)
    assert torch.equal(
        block.functional_rows,
        bank.tangent_rows * bank.coordinate_rms_tangent_row_scales.unsqueeze(1),
    )
    assert (
        fingerprint_lfm2_swiglu_tangent_tensor(block.functional_rows)
        == bank.coordinate_scaled_tangent_rows_sha256
    )
    assert tuple(block.row_sample_weights) == tuple(bank.tangent_row_weights.tolist())
    qdeim_weighted = block.functional_rows * torch.sqrt(
        torch.tensor(block.row_sample_weights, dtype=torch.float64) * block.block_weight
    ).unsqueeze(1)
    assert torch.allclose(
        torch.linalg.vector_norm(qdeim_weighted),
        empirical_norm,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert lfm2_mlp_qdeim_functional_snapshot_evidence_sha256(
        [block]
    ) == lfm2_mlp_qdeim_functional_snapshot_evidence_sha256([block])


def test_tangents_lift_snapshot_rank_as_an_advisory_observation() -> None:
    w1 = torch.eye(2, dtype=torch.float64)
    w3 = torch.eye(2, dtype=torch.float64)
    source = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    decoder = torch.eye(2, dtype=torch.float64)
    bank = _build(
        w1=w1,
        w3=w3,
        normalized_inputs=source,
        decoder=decoder,
        anchor_count=1,
    )
    _gate, _up, post = _cached_rows(w1, w3, source)
    assert int(torch.linalg.matrix_rank(post)) == 1
    assert int(torch.linalg.matrix_rank(bank.tangent_rows)) == 2
    spectrum = bank.advisory_diagnostics["spectrum"]
    assert spectrum["covers_full_tangent_bank"] is True
    assert spectrum["rank_estimate"] == 2
    assert spectrum["used_for_semantic_gate"] is False


def test_gradient_sources_and_nonpositive_weights_fail_closed() -> None:
    w1 = torch.eye(2, dtype=torch.float64)
    w3 = torch.eye(2, dtype=torch.float64)
    source = torch.ones(1, 2, dtype=torch.float64)
    decoder = torch.eye(2, dtype=torch.float64)
    with pytest.raises(Lfm2SwiGluTangentBankError, match="cannot require gradients"):
        _build(
            w1=w1.requires_grad_(True),
            w3=w3,
            normalized_inputs=source,
            decoder=decoder,
            anchor_count=1,
        )
    w1.requires_grad_(False)
    with pytest.raises(Lfm2SwiGluTangentBankError, match="strictly positive"):
        _build(
            w1=w1,
            w3=w3,
            normalized_inputs=source,
            decoder=decoder,
            sample_weights=torch.tensor([0.0], dtype=torch.float64),
            anchor_count=1,
        )
