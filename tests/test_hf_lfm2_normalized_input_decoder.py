from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

import neural_forge.hf_lfm2_structural_residual_attention_fold as legacy_module
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_normalized_input_decoder import (
    LFM2_NORMALIZED_INPUT_DECODER_SCHEMA,
    Lfm2NormalizedInputDecoderError,
    compile_lfm2_normalized_input_decoder,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2CachedNormalizedInputMlpFoldCase,
    compile_lfm2_normalized_input_real_medoid_structural_mlp_fold,
    lfm2_authenticated_tensor_sha256,
)

torch = pytest.importorskip("torch")


def _completion_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _completion_inputs(
    raw_structural_decoder: torch.Tensor,
) -> dict[str, object]:
    body = {
        "schema": "neural-forge-analytic-coordinate-completion-candidate-v1",
        "recipe": "test-authenticated-null-completion",
        "evidence_sha256": "1" * 64,
        "coordinate_map_shape": [int(value) for value in raw_structural_decoder.shape],
        "coordinate_map_sha256": _completion_tensor_sha256(raw_structural_decoder),
        "materialized_rank": int(
            torch.linalg.matrix_rank(raw_structural_decoder).item()
        ),
        "full_rank_materialized": True,
    }
    receipt_sha256 = canonical_sha256(body)
    return {
        "raw_structural_decoder": raw_structural_decoder,
        "raw_structural_decoder_sha256": lfm2_authenticated_tensor_sha256(
            raw_structural_decoder
        ),
        "coordinate_completion_receipt": {
            **body,
            "result_sha256": receipt_sha256,
        },
        "coordinate_completion_receipt_sha256": receipt_sha256,
    }


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    parent_gamma = torch.tensor([10.0, 0.1, 5.0], dtype=torch.float64)
    target_gamma = torch.tensor([0.2, 4.0], dtype=torch.float64)
    normalized_orientation = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    residual_chart = (
        parent_gamma.unsqueeze(1)
        * normalized_orientation
        * target_gamma.reciprocal().unsqueeze(0)
    ).contiguous()
    source_gate = torch.tensor(
        [
            [1.0, 0.25, -0.5],
            [-0.75, 1.5, 0.4],
            [0.3, -2.0, 1.25],
            [1.75, 0.6, -0.8],
        ],
        dtype=torch.float64,
    )
    source_up = torch.tensor(
        [
            [0.5, -1.0, 0.75],
            [1.25, 0.4, -0.6],
            [-0.2, 1.1, 1.5],
            [0.9, -0.7, 0.3],
        ],
        dtype=torch.float64,
    )
    source_down = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75],
            [0.4, 1.25, -0.8, 0.2],
            [-0.3, 0.6, 1.1, -1.0],
        ],
        dtype=torch.float64,
    )
    target_input = torch.tensor([[1.0, 0.0], [-2.0, 0.0]], dtype=torch.float64)
    parent_input = target_input @ normalized_orientation.T
    parent_gate = parent_input @ source_gate.T
    parent_up = parent_input @ source_up.T
    parent_post = torch.nn.functional.silu(parent_gate) * parent_up
    target_residual = torch.tensor(
        [[0.1, -0.2], [0.3, 0.15]],
        dtype=torch.float64,
    )
    parent_delta = parent_post @ source_down.T
    endpoint = target_residual + parent_delta @ residual_chart
    case = Lfm2CachedNormalizedInputMlpFoldCase(
        target_input=target_input,
        parent_normalized_input=parent_input,
        parent_gate=parent_gate,
        parent_up=parent_up,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=endpoint,
        target_pre_mlp_residual=target_residual,
        parent_contribution=parent_post.abs() + 0.125,
    )
    shared = {
        "residual_chart": residual_chart,
        "residual_chart_sha256": lfm2_authenticated_tensor_sha256(residual_chart),
        "parent_ffn_norm_weight": parent_gamma,
        "parent_ffn_norm_weight_sha256": lfm2_authenticated_tensor_sha256(parent_gamma),
        "target_ffn_norm_weight": target_gamma,
        "target_ffn_norm_weight_sha256": lfm2_authenticated_tensor_sha256(target_gamma),
        "cases": (case,),
        "case_sample_weights": (2.5,),
        "normalized_decoder_ridge": 1e-8,
    }
    legacy = {
        **shared,
        "source_gate_weight": source_gate,
        "source_up_weight": source_up,
        "source_down_weight": source_down,
        "target_gate_weight_template": torch.empty(
            (2, 2),
            dtype=torch.float64,
        ),
        "target_up_weight_template": torch.empty(
            (2, 2),
            dtype=torch.float64,
        ),
        "target_down_weight_template": torch.empty(
            (2, 2),
            dtype=torch.float64,
        ),
        "anchors_per_source": 2,
        "max_candidate_targets": 2,
        "ridge": 0.0,
        "cluster_ridge": 1e-8,
    }
    return shared, legacy


def test_decoder_only_compiler_is_bitwise_equivalent_to_legacy_decoder() -> None:
    shared, legacy = _fixture()
    expected = compile_lfm2_normalized_input_real_medoid_structural_mlp_fold(**legacy)
    actual = compile_lfm2_normalized_input_decoder(**shared)

    assert torch.equal(
        actual.normalized_input_decoder,
        expected.normalized_input_decoder,
    )
    assert actual.normalized_input_decoder_sha256 == (
        expected.normalized_input_decoder_sha256
    )
    assert actual.normalized_input_decoder_rank == (
        expected.normalized_input_decoder_rank
    )
    assert actual.target_hidden_size == expected.target_hidden_size
    assert actual.source_hidden_size == expected.source_hidden_size
    assert actual.normalized_input_decoder.device.type == "cpu"
    assert actual.normalized_input_decoder.dtype == torch.float64
    assert actual.normalized_input_decoder.requires_grad is False
    assert actual.normalized_input_decoder.grad_fn is None

    receipt = actual.as_dict()
    expected_receipt = expected.as_dict()
    assert receipt["schema"] == LFM2_NORMALIZED_INPUT_DECODER_SCHEMA
    assert receipt["fit_evidence_sha256"] == expected_receipt["fit_evidence_sha256"]
    assert (
        receipt["fit_row_weights_sha256"]
        == (expected_receipt["fit_row_weights_sha256"])
    )
    assert receipt["producer_bridge"] == expected_receipt["producer_bridge"]
    assert receipt["residual_chart_rank"] == expected.residual_chart_rank
    assert receipt["residual_chart_decoder_method"] == (
        expected.residual_chart_decoder_method
    )
    unsealed = dict(receipt)
    assert unsealed.pop("receipt_sha256") == actual.receipt_sha256
    assert canonical_sha256(unsealed) == actual.receipt_sha256
    assert actual.verify()["verified"] is True


def test_decoder_accepts_authenticated_full_rank_completion_as_structural_prior() -> (
    None
):
    shared, _legacy = _fixture()
    raw_structural_decoder = torch.tensor(
        [[0.02, 0.0, 0.0], [0.0, 40.0, 7.0]],
        dtype=torch.float64,
    )
    legacy = compile_lfm2_normalized_input_decoder(**shared)
    compiled = compile_lfm2_normalized_input_decoder(
        **shared,
        **_completion_inputs(raw_structural_decoder),
    )

    assert compiled.normalized_input_decoder_rank == 2
    assert not torch.equal(
        compiled.normalized_input_decoder,
        legacy.normalized_input_decoder,
    )
    target_rows = shared["cases"][0].target_input
    parent_rows = shared["cases"][0].parent_normalized_input
    assert torch.allclose(
        target_rows @ compiled.normalized_input_decoder,
        parent_rows,
        rtol=1e-12,
        atol=1e-12,
    )
    assert abs(float(compiled.normalized_input_decoder[1, 2].item())) > 1.0

    receipt = compiled.as_dict()
    provenance = receipt["coordinate_completion_provenance"]
    assert provenance["source"] == "authenticated_coordinate_completion_candidate"
    assert provenance["stable_seal_field"] == "result_sha256"
    assert provenance["raw_structural_decoder_sha256"] == (
        lfm2_authenticated_tensor_sha256(raw_structural_decoder)
    )
    assert provenance["coordinate_map_sha256"] == _completion_tensor_sha256(
        raw_structural_decoder
    )
    assert (
        provenance["receipt"]
        == _completion_inputs(raw_structural_decoder)["coordinate_completion_receipt"]
    )
    bridge = receipt["producer_bridge"]
    assert bridge["structural_prior_source"] == (
        "authenticated_coordinate_completion_candidate"
    )
    assert bridge["structural_decoder_left_inverse_required"] is False
    assert receipt["residual_chart_decoder_method"] == (
        "authenticated_coordinate_completion_structural_decoder"
    )
    assert receipt["operation_audit"]["residual_chart_decoder_recomputed"] is False
    assert compiled.verify()["verified"] is True


def test_decoder_preserves_completion_scale_across_negative_fit_alignment() -> None:
    shared, _legacy = _fixture()
    raw_structural_decoder = torch.tensor(
        [[-0.02, 0.0, 0.0], [0.0, 40.0, 7.0]],
        dtype=torch.float64,
    )

    compiled = compile_lfm2_normalized_input_decoder(
        **shared,
        **_completion_inputs(raw_structural_decoder),
    )

    parent_gamma = shared["parent_ffn_norm_weight"]
    target_gamma = shared["target_ffn_norm_weight"]
    structural_prior = (
        target_gamma.reciprocal().unsqueeze(1)
        * raw_structural_decoder
        * parent_gamma.unsqueeze(0)
    )
    target_rows = shared["cases"][0].target_input
    rowspace = torch.linalg.pinv(target_rows) @ target_rows
    nullspace = torch.eye(2, dtype=torch.float64) - rowspace

    assert compiled.normalized_input_decoder_rank == 2
    assert torch.allclose(
        nullspace @ compiled.normalized_input_decoder,
        nullspace @ structural_prior,
        rtol=1e-12,
        atol=1e-12,
    )
    bridge = compiled.as_dict()["producer_bridge"]
    assert bridge["fit_signed_scalar_gain_advisory"] < 0.0
    assert bridge["positive_global_gain"] == pytest.approx(
        torch.linalg.matrix_norm(structural_prior, ord=2).item()
    )
    assert bridge["structural_scale_policy"] == (
        "authenticated-coordinate-completion-prior-scale-preserved"
    )
    assert bridge["supported_relative_cap_rho"] == 0.25
    assert bridge["normalized_input_correction_rho"] == 0.25
    assert bridge["correction_norm_cap_scale"] < 1.0
    assert bridge["fit_structural_gain_used_for_acceptance_or_veto"] is False
    assert bridge["identified_action_policy"] == (
        "rho-capped-fit-rowspace-tikhonov-residual-correction"
    )
    assert compiled.as_dict()["normalized_input_correction_rho"] == 0.25
    assert bridge["structural_null_prior_preserved_exactly"] is True
    assert bridge["rank_safe_selection_is_structural_not_semantic"] is True
    assert compiled.verify()["verified"] is True


def test_completion_correction_rho_traces_historical_fit_rowspace_frontier() -> None:
    shared, _legacy = _fixture()
    raw_structural_decoder = torch.tensor(
        [[0.01, 0.0, 0.0], [0.0, 40.0, 7.0]],
        dtype=torch.float64,
    )
    parent_gamma = shared["parent_ffn_norm_weight"]
    target_gamma = shared["target_ffn_norm_weight"]
    structural_prior = (
        target_gamma.reciprocal().unsqueeze(1)
        * raw_structural_decoder
        * parent_gamma.unsqueeze(0)
    )
    target_rows = shared["cases"][0].target_input
    rowspace = torch.linalg.pinv(target_rows) @ target_rows
    nullspace = torch.eye(2, dtype=torch.float64) - rowspace
    supported_prior_norm = torch.linalg.vector_norm(rowspace @ structural_prior).item()
    compiled_by_rho = {}

    for rho in (0.0, 0.25, 0.5, 1.0):
        compiled = compile_lfm2_normalized_input_decoder(
            **shared,
            normalized_input_correction_rho=rho,
            **_completion_inputs(raw_structural_decoder),
        )
        bridge = compiled.as_dict()["producer_bridge"]
        applied_correction_norm = torch.linalg.vector_norm(
            compiled.normalized_input_decoder - structural_prior
        ).item()

        assert compiled.normalized_input_decoder_rank == 2
        assert compiled.as_dict()["normalized_input_correction_rho"] == rho
        assert bridge["normalized_input_correction_rho"] == rho
        assert bridge["supported_relative_cap_rho"] == rho
        assert applied_correction_norm <= rho * supported_prior_norm + 1e-12
        assert torch.allclose(
            nullspace @ compiled.normalized_input_decoder,
            nullspace @ structural_prior,
            rtol=1e-12,
            atol=1e-12,
        )
        assert compiled.verify()["verified"] is True
        compiled_by_rho[rho] = compiled

    assert torch.equal(
        compiled_by_rho[0.0].normalized_input_decoder,
        structural_prior,
    )
    first_coordinate_actions = [
        float(
            (target_rows @ compiled_by_rho[rho].normalized_input_decoder)[0, 0].item()
        )
        for rho in (0.0, 0.25, 0.5, 1.0)
    ]
    assert first_coordinate_actions == sorted(first_coordinate_actions)
    assert (
        len(set(compiled.receipt_sha256 for compiled in compiled_by_rho.values())) == 4
    )

    default = compile_lfm2_normalized_input_decoder(
        **shared,
        **_completion_inputs(raw_structural_decoder),
    )
    assert default.receipt_sha256 == compiled_by_rho[0.25].receipt_sha256


def test_completion_rowspace_replacement_uses_rank_safe_structural_homotopy() -> None:
    shared, _legacy = _fixture()
    case = shared["cases"][0]
    shared["cases"] = (
        replace(
            case,
            parent_normalized_input=torch.zeros_like(case.parent_normalized_input),
        ),
    )
    shared["normalized_decoder_ridge"] = 0.0
    raw_structural_decoder = torch.tensor(
        [[0.02, 0.0, 0.0], [0.0, 40.0, 7.0]],
        dtype=torch.float64,
    )

    compiled = compile_lfm2_normalized_input_decoder(
        **shared,
        normalized_input_correction_rho=1.0,
        **_completion_inputs(raw_structural_decoder),
    )

    bridge = compiled.as_dict()["producer_bridge"]
    assert compiled.normalized_input_decoder_rank == 2
    assert bridge["fit_signed_scalar_gain_advisory"] == 0.0
    assert 0.0 < bridge["rank_safe_correction_scale"] < 1.0
    assert bridge["rank_safe_frontier_evaluations"] == 2
    assert bridge["rank_safe_prior_minimum_singular_value"] > 0.0
    assert bridge["rank_safe_correction_operator_norm"] > 0.0
    assert bridge["rank_safe_selection_is_structural_not_semantic"] is True
    assert compiled.verify()["verified"] is True


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda values: values.pop("coordinate_completion_receipt_sha256"),
            "must be supplied together",
        ),
        (
            lambda values: values.__setitem__(
                "raw_structural_decoder_sha256",
                "0" * 64,
            ),
            "tensor authentication failed",
        ),
        (
            lambda values: values.__setitem__(
                "coordinate_completion_receipt_sha256",
                "0" * 64,
            ),
            "receipt authentication failed",
        ),
        (
            lambda values: values["coordinate_completion_receipt"].__setitem__(
                "coordinate_map_sha256",
                "0" * 64,
            ),
            "receipt authentication failed",
        ),
    ),
)
def test_decoder_fails_closed_on_incomplete_or_mismatched_completion_provenance(
    mutation,
    match: str,
) -> None:
    shared, _legacy = _fixture()
    raw_structural_decoder = torch.tensor(
        [[0.02, 0.0, 0.0], [0.0, 40.0, 7.0]],
        dtype=torch.float64,
    )
    completion = _completion_inputs(raw_structural_decoder)
    mutation(completion)
    with pytest.raises(Lfm2NormalizedInputDecoderError, match=match):
        compile_lfm2_normalized_input_decoder(**shared, **completion)


def test_decoder_rejects_self_sealed_receipt_for_a_different_coordinate_map() -> None:
    shared, _legacy = _fixture()
    raw_structural_decoder = torch.tensor(
        [[0.02, 0.0, 0.0], [0.0, 40.0, 7.0]],
        dtype=torch.float64,
    )
    completion = _completion_inputs(raw_structural_decoder)
    receipt = dict(completion["coordinate_completion_receipt"])
    receipt.pop("result_sha256")
    receipt["coordinate_map_sha256"] = "2" * 64
    resealed = canonical_sha256(receipt)
    completion["coordinate_completion_receipt"] = {
        **receipt,
        "result_sha256": resealed,
    }
    completion["coordinate_completion_receipt_sha256"] = resealed
    with pytest.raises(
        Lfm2NormalizedInputDecoderError,
        match="does not bind the supplied raw_structural_decoder",
    ):
        compile_lfm2_normalized_input_decoder(**shared, **completion)


def test_decoder_rejects_self_sealed_completion_with_wrong_topology() -> None:
    shared, _legacy = _fixture()
    raw_structural_decoder = torch.tensor(
        [[0.02, 0.0, 0.0], [0.0, 40.0, 7.0]],
        dtype=torch.float64,
    )
    completion = _completion_inputs(raw_structural_decoder)
    receipt = dict(completion["coordinate_completion_receipt"])
    receipt.pop("result_sha256")
    receipt["coordinate_map_shape"] = [3, 2]
    resealed = canonical_sha256(receipt)
    completion["coordinate_completion_receipt"] = {
        **receipt,
        "result_sha256": resealed,
    }
    completion["coordinate_completion_receipt_sha256"] = resealed
    with pytest.raises(
        Lfm2NormalizedInputDecoderError,
        match="coordinate_map_shape does not match",
    ):
        compile_lfm2_normalized_input_decoder(**shared, **completion)


def test_decoder_only_compiler_executes_zero_signature_cluster_or_medoid_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared, _legacy = _fixture()
    calls = {
        "weight_signature": 0,
        "multi_anchor_cluster": 0,
        "activation_cluster": 0,
    }

    def _forbidden(name: str):
        def invoke(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"decoder-only compiler called {name}")

        return invoke

    monkeypatch.setattr(
        legacy_module,
        "build_gated_mlp_weight_complete_signature",
        _forbidden("weight_signature"),
    )
    monkeypatch.setattr(
        legacy_module,
        "build_gated_mlp_multi_anchor_cluster_plan",
        _forbidden("multi_anchor_cluster"),
    )
    monkeypatch.setattr(
        legacy_module,
        "build_gated_mlp_activation_cluster_plan",
        _forbidden("activation_cluster"),
    )

    compiled = compile_lfm2_normalized_input_decoder(**shared)
    assert calls == {
        "weight_signature": 0,
        "multi_anchor_cluster": 0,
        "activation_cluster": 0,
    }
    operations = compiled.as_dict()["operation_audit"]
    assert operations["weight_complete_signature_builds"] == 0
    assert operations["activation_cluster_plan_builds"] == 0
    assert operations["multi_anchor_cluster_plan_builds"] == 0
    assert operations["medoid_selection_operations"] == 0
    assert operations["source_mlp_weight_tensor_reads"] == 0
    assert compiled.as_dict()["cluster_or_medoid_state_constructed"] is False


def test_decoder_only_compiler_fails_closed_on_authentication_and_fit_geometry() -> (
    None
):
    shared, _legacy = _fixture()
    corrupt = dict(shared)
    corrupt["parent_ffn_norm_weight"] = shared["parent_ffn_norm_weight"] + 0.25
    with pytest.raises(
        Lfm2NormalizedInputDecoderError,
        match="parent_ffn_norm_weight tensor authentication failed",
    ):
        compile_lfm2_normalized_input_decoder(**corrupt)

    invalid_weight = dict(shared)
    invalid_weight["case_sample_weights"] = (0.0,)
    with pytest.raises(
        Lfm2NormalizedInputDecoderError,
        match="positive finite",
    ):
        compile_lfm2_normalized_input_decoder(**invalid_weight)

    case = shared["cases"][0]
    zero_fit_case = replace(
        case,
        target_input=torch.zeros_like(case.target_input),
    )
    zero_fit = dict(shared)
    zero_fit["cases"] = (zero_fit_case,)
    with pytest.raises(
        Lfm2NormalizedInputDecoderError,
        match="positive normalized-input structural gain|zero rank",
    ):
        compile_lfm2_normalized_input_decoder(**zero_fit)


def test_decoder_receipt_rejects_rank_deficient_chart() -> None:
    shared, _legacy = _fixture()
    rank_deficient_chart = torch.tensor(
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
        dtype=torch.float64,
    )
    invalid = dict(shared)
    invalid["residual_chart"] = rank_deficient_chart
    invalid["residual_chart_sha256"] = lfm2_authenticated_tensor_sha256(
        rank_deficient_chart
    )
    with pytest.raises(
        Lfm2NormalizedInputDecoderError,
        match="full target-column rank",
    ):
        compile_lfm2_normalized_input_decoder(**invalid)
