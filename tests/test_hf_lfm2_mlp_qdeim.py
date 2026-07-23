from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_mlp_qdeim import (  # noqa: E402
    LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE,
    LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE,
    Lfm2MlpQdeimError,
    Lfm2MlpQdeimFitCase,
    Lfm2MlpQdeimFunctionalSnapshotBlock,
    Lfm2MlpQdeimInsufficientRankError,
    _deterministic_pivoted_qr_indices,
    compile_lfm2_mlp_nonlinear_qdeim,
    lfm2_mlp_qdeim_fit_evidence_sha256,
    lfm2_mlp_qdeim_functional_snapshot_evidence_sha256,
    lfm2_mlp_qdeim_source_state_sha256,
    lfm2_mlp_qdeim_target_template_sha256,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


def _reference_full_residual_pivoted_qr_indices(
    basis: Any,
) -> tuple[tuple[int, ...], Any]:
    """The original full-residual implementation, retained as a test oracle."""

    source_neurons, modes = map(int, basis.shape)
    matrix = basis.T.contiguous()
    selected: list[int] = []
    q_columns = torch.empty((modes, 0), dtype=torch.float64)
    pivot_residual_norms: list[float] = []
    epsilon = torch.finfo(torch.float64).eps
    for _step in range(modes):
        if q_columns.shape[1]:
            residual = matrix - q_columns @ (q_columns.T @ matrix)
            residual = residual - q_columns @ (q_columns.T @ residual)
        else:
            residual = matrix
        scores = residual.square().sum(dim=0)
        if selected:
            scores[torch.tensor(selected, dtype=torch.long)] = -1.0
        maximum = float(scores.max().item())
        tie_tolerance = 128.0 * epsilon * max(1.0, abs(maximum))
        candidates = [
            index
            for index in range(source_neurons)
            if index not in selected
            and float(scores[index].item()) >= maximum - tie_tolerance
        ]
        pivot = min(candidates)
        vector = residual[:, pivot].contiguous()
        norm = float(torch.linalg.vector_norm(vector).item())
        q_columns = torch.cat(
            (q_columns, (vector / norm).unsqueeze(1)),
            dim=1,
        ).contiguous()
        selected.append(pivot)
        pivot_residual_norms.append(norm)
    return tuple(selected), torch.tensor(pivot_residual_norms, dtype=torch.float64)


def _fixture(*, target_intermediate: int = 2) -> dict[str, Any]:
    source_gate = torch.tensor(
        [
            [1.0, -0.5],
            [0.25, 1.5],
            [-1.0, 0.75],
            [2.0, 0.125],
        ],
        dtype=torch.float64,
    )
    source_up = torch.tensor(
        [
            [0.5, 1.0],
            [-1.25, 0.5],
            [0.75, -0.25],
            [1.5, 2.0],
        ],
        dtype=torch.float64,
    )
    source_down = torch.tensor(
        [[0.5, -1.0, 2.0, 0.25], [1.5, 0.75, -0.5, 2.25]],
        dtype=torch.float64,
    )
    coefficients = torch.tensor(
        [
            [1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    cases = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-0.5, 1.25],
            [1.5, 0.25],
        ],
        dtype=torch.float64,
    )
    parent_post = cases @ coefficients
    target_input = torch.tensor(
        [
            [0.5, -0.25],
            [1.0, 0.75],
            [-0.5, 1.25],
            [1.5, -1.0],
            [0.25, 0.5],
            [-1.25, -0.75],
        ],
        dtype=torch.float64,
    )
    target_pre = torch.tensor(
        [[0.1, -0.2], [0.0, 0.3], [-0.2, 0.4], [0.5, 0.1], [0.2, 0.2], [0.0, -0.1]],
        dtype=torch.float64,
    )
    endpoint = target_pre + parent_post @ source_down.T
    fit_case = Lfm2MlpQdeimFitCase(
        target_normalized_input=target_input,
        parent_post_swiglu=parent_post,
        parent_post_mlp_residual_target_space=endpoint,
        target_pre_mlp_residual=target_pre,
    )
    return {
        "source_gate": source_gate,
        "source_up": source_up,
        "source_down": source_down,
        "target_gate_template": torch.zeros(
            (target_intermediate, 2), dtype=torch.bfloat16
        ),
        "target_up_template": torch.zeros(
            (target_intermediate, 2), dtype=torch.bfloat16
        ),
        "target_down_template": torch.zeros(
            (2, target_intermediate), dtype=torch.bfloat16
        ),
        "decoder": torch.eye(2, dtype=torch.float64),
        "chart": torch.eye(2, dtype=torch.float64),
        "fit_cases": (fit_case,),
        "case_weights": (1.75,),
        "functional_blocks": (),
    }


def _compile(
    fixture: dict[str, Any],
    *,
    compute_fit_residual_correction: bool = False,
    **overrides: Any,
):
    blocks = fixture["functional_blocks"]
    arguments = {
        "source_gate_weight": fixture["source_gate"],
        "source_up_weight": fixture["source_up"],
        "source_down_weight": fixture["source_down"],
        "expected_source_state_sha256": lfm2_mlp_qdeim_source_state_sha256(
            fixture["source_gate"],
            fixture["source_up"],
            fixture["source_down"],
        ),
        "target_gate_weight_template": fixture["target_gate_template"],
        "target_up_weight_template": fixture["target_up_template"],
        "target_down_weight_template": fixture["target_down_template"],
        "expected_target_template_sha256": (
            lfm2_mlp_qdeim_target_template_sha256(
                fixture["target_gate_template"],
                fixture["target_up_template"],
                fixture["target_down_template"],
            )
        ),
        "normalized_input_decoder": fixture["decoder"],
        "expected_normalized_input_decoder_sha256": (
            lfm2_authenticated_tensor_sha256(fixture["decoder"])
        ),
        "residual_output_chart": fixture["chart"],
        "expected_residual_output_chart_sha256": (
            lfm2_authenticated_tensor_sha256(fixture["chart"])
        ),
        "fit_cases": fixture["fit_cases"],
        "case_sample_weights": fixture["case_weights"],
        "expected_fit_evidence_sha256": lfm2_mlp_qdeim_fit_evidence_sha256(
            fixture["fit_cases"], fixture["case_weights"]
        ),
        "functional_snapshot_blocks": blocks,
        "expected_functional_snapshot_evidence_sha256": (
            lfm2_mlp_qdeim_functional_snapshot_evidence_sha256(blocks)
        ),
        "compute_fit_residual_correction": compute_fit_residual_correction,
    }
    arguments.update(overrides)
    return compile_lfm2_mlp_nonlinear_qdeim(**arguments)


def test_exact_low_rank_recovery_and_deterministic_qdeim_selection() -> None:
    fixture = _fixture()

    first = _compile(fixture)
    second = _compile(fixture)

    parent_post = fixture["fit_cases"][0].parent_post_swiglu
    selected = list(first.selected_source_neurons)
    reconstructed = parent_post[:, selected] @ first.interpolation_reconstruction
    torch.testing.assert_close(reconstructed, parent_post, atol=1e-12, rtol=1e-12)
    assert first.selected_source_neurons == second.selected_source_neurons
    assert torch.equal(
        first.interpolation_reconstruction,
        second.interpolation_reconstruction,
    )
    assert first.report["result_sha256"] == second.report["result_sha256"]
    assert first.report["minimum_snapshot_rank"] == 2
    assert first.report["observed_snapshot_rank"] == 2
    assert first.report["rank_completion"]["random_completion_modes"] == 0
    assert not first.report["rank_completion"]["silent_completion_performed"]
    compute = first.report["selection_compute_diagnostic"]
    assert compute == {
        "schema": "neural-forge-lfm2-mlp-qdeim-selection-compute-v1",
        "basis_transpose_shape": [2, 4],
        "preallocated_q_shape": [2, 2],
        "pivot_step_count": 2,
        "initial_full_score_pass_count": 1,
        "rank_one_score_update_count": 2,
        "full_residual_matrix_reconstruction_count": 0,
        "selected_pivot_reorthogonalization_pass_count": 2,
        "asymptotic_time_bound": "O(pivot_steps*pod_modes*source_neurons)",
        "asymptotic_auxiliary_storage_bound": (
            "O(pod_modes*source_neurons+pod_modes^2+source_neurons)"
        ),
        "wall_clock_timing_recorded": False,
        "timing_used_for_candidate_selection": False,
        "timing_used_for_semantic_acceptance_or_veto": False,
        "performance_diagnostics_have_semantic_authority": False,
    }


def test_compute_device_is_explicit_receipted_and_returns_cpu_candidates() -> None:
    fixture = _fixture()

    result = _compile(fixture, compute_device="cpu")

    execution = result.report["compute_execution"]
    execution_body = dict(execution)
    receipt_sha256 = execution_body.pop("receipt_sha256")
    assert execution["requested_device"] == "cpu"
    assert execution["resolved_device"] == "cpu"
    assert execution["backend"] == "cpu"
    assert execution["authenticated_source_archive_device"] == "cpu"
    assert execution["materialized_candidate_device"] == "cpu"
    assert execution["silent_device_fallback_performed"] is False
    assert receipt_sha256 == canonical_sha256(execution_body)
    assert all(
        value.device.type == "cpu"
        for value in result.state_dict(
            LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE
        ).values()
    )

    for invalid in ("cuda", "cuda:00", "cuda:\u0660"):
        with pytest.raises(Lfm2MlpQdeimError, match="explicitly indexed"):
            _compile(fixture, compute_device=invalid)
    unavailable = f"cuda:{torch.cuda.device_count()}"
    with pytest.raises(Lfm2MlpQdeimError, match="unavailable"):
        _compile(fixture, compute_device=unavailable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_stages_fp64_qdeim_and_returns_standalone_cpu_bf16() -> None:
    result = _compile(_fixture(), compute_device="cuda:0")

    execution = result.report["compute_execution"]
    assert execution["requested_device"] == "cuda:0"
    assert execution["resolved_device"] == "cuda:0"
    assert execution["backend"] == "cuda"
    assert execution["active_layer_staged_to_compute_device"] is True
    assert execution["gpu_is_ephemeral_dense_algebra_workspace"] is True
    assert execution["compute_dtype"] == "float64"
    assert all(
        value.device.type == "cpu" and value.dtype == torch.bfloat16
        for value in result.state_dict(
            LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE
        ).values()
    )


@pytest.mark.parametrize("source_neurons,modes", ((64, 4), (64, 8), (96, 12)))
def test_incremental_qrcp_preserves_original_deterministic_pivots(
    source_neurons: int,
    modes: int,
) -> None:
    rows = torch.arange(source_neurons, dtype=torch.float64).unsqueeze(1) + 0.5
    columns = torch.arange(modes, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(math.pi * rows * columns / source_neurons)
    basis[:, 0] /= math.sqrt(source_neurons)
    if modes > 1:
        basis[:, 1:] *= math.sqrt(2.0 / source_neurons)

    expected, expected_norms = _reference_full_residual_pivoted_qr_indices(basis)
    actual, actual_norms = _deterministic_pivoted_qr_indices(basis)

    assert actual == expected
    torch.testing.assert_close(actual_norms, expected_norms, atol=1e-14, rtol=1e-14)


def test_incremental_qrcp_preserves_stable_low_index_ties() -> None:
    basis = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )

    selected, norms = _deterministic_pivoted_qr_indices(basis)

    assert selected == (0, 2)
    torch.testing.assert_close(norms, torch.ones(2, dtype=torch.float64))


def test_selected_nonlinear_producers_remain_intact_and_are_not_averaged() -> None:
    fixture = _fixture()
    result = _compile(fixture)
    selected = list(result.selected_source_neurons)

    expected_gate = (fixture["source_gate"][selected] @ fixture["decoder"].T).to(
        torch.bfloat16
    )
    expected_up = (fixture["source_up"][selected] @ fixture["decoder"].T).to(
        torch.bfloat16
    )
    assert torch.equal(result.gate_weight, expected_gate)
    assert torch.equal(result.up_weight, expected_up)
    assert not result.report["producer_transport"]["producer_averaging"]


def test_dense_writer_bakes_in_an_omitted_source_neuron_contribution() -> None:
    fixture = _fixture()
    baseline = _compile(fixture)
    reconstruction = baseline.interpolation_reconstruction
    expected = (fixture["chart"].T @ fixture["source_down"] @ reconstruction.T).to(
        torch.bfloat16
    )
    assert torch.equal(baseline.down_weight_structural_prior, expected)

    omitted = sorted(set(range(4)).difference(baseline.selected_source_neurons))
    contributing = next(
        index
        for index in omitted
        if float(torch.linalg.vector_norm(reconstruction[:, index]).item()) > 0.0
    )
    only_omitted = torch.zeros_like(fixture["source_down"])
    only_omitted[:, contributing] = torch.tensor([2.0, -3.0])
    fixture["source_down"] = only_omitted

    folded = _compile(fixture)

    assert bool(torch.count_nonzero(folded.down_weight_structural_prior).item())
    assert folded.report["writer_transport"]["all_source_writer_columns_consumed"]
    assert (
        folded.report["writer_transport"]["omitted_reconstruction_nonzero_column_count"]
        > 0
    )


def test_optional_fit_only_w2_correction_is_a_separate_dense_candidate() -> None:
    fixture = _fixture()
    selected_probe = _compile(fixture)
    selected = list(selected_probe.selected_source_neurons)
    materialized_gate = fixture["source_gate"][selected].to(torch.bfloat16)
    materialized_up = fixture["source_up"][selected].to(torch.bfloat16)
    target_input = fixture["fit_cases"][0].target_normalized_input
    target_features = torch.nn.functional.silu(
        target_input @ materialized_gate.to(torch.float64).T
    ) * (target_input @ materialized_up.to(torch.float64).T)
    desired_down = torch.tensor([[1.25, -0.75], [0.5, 1.5]], dtype=torch.float64)
    fit_case = fixture["fit_cases"][0]
    fixture["fit_cases"] = (
        replace(
            fit_case,
            parent_post_mlp_residual_target_space=(
                fit_case.target_pre_mlp_residual + target_features @ desired_down.T
            ),
        ),
    )

    result = _compile(fixture, compute_fit_residual_correction=True)

    assert result.candidate_ids == (
        LFM2_MLP_QDEIM_STRUCTURAL_PRIOR_CANDIDATE,
        LFM2_MLP_QDEIM_FIT_CORRECTED_CANDIDATE,
    )
    assert result.down_weight_fit_corrected is not None
    assert not torch.equal(
        result.down_weight_structural_prior,
        result.down_weight_fit_corrected,
    )
    for state in result.candidate_state_dicts().values():
        assert state["w1.weight"].shape == (2, 2)
        assert state["w3.weight"].shape == (2, 2)
        assert state["w2.weight"].shape == (2, 2)
        assert all(tensor.dtype == torch.bfloat16 for tensor in state.values())
        assert all(not tensor.requires_grad for tensor in state.values())
    correction = result.report["fit_only_w2_residual_correction"]
    assert correction["fit_rows_only"]
    assert correction["training_steps"] == 0
    assert result.report["automatic_candidate_selection_performed"] is False


def test_rank_deficit_is_typed_and_authenticated_tangent_rows_complete_it() -> None:
    fixture = _fixture(target_intermediate=3)

    with pytest.raises(Lfm2MlpQdeimInsufficientRankError) as caught:
        _compile(fixture)

    deficit = caught.value.report
    assert deficit["minimum_snapshot_rank"] == 3
    assert deficit["observed_snapshot_rank"] == 2
    assert deficit["snapshot_rank_deficit"] == 1
    assert deficit["selected_source_neurons"] == []
    assert deficit["interpolation_matrix_materialized"] is False
    assert deficit["random_mode_completion_permitted"] is False
    assert deficit["compute_execution"]["requested_device"] == "cpu"
    assert deficit["compute_execution"]["resolved_device"] == "cpu"
    assert (
        deficit["structural_completion_input_contract"]["public_type"]
        == "Lfm2MlpQdeimFunctionalSnapshotBlock"
    )

    tangent_rows = torch.tensor([[0.0, 0.0, 0.0, 0.5]], dtype=torch.float64)
    block = Lfm2MlpQdeimFunctionalSnapshotBlock(
        block_id="layer-0-child-lifted-tangent",
        snapshot_kind="analytic_child_lifted_parent_swiglu_tangent",
        functional_rows=tangent_rows,
        generator_receipt_sha256="a" * 64,
        block_weight=4.0,
        row_sample_weights=(2.25,),
    )
    fixture["functional_blocks"] = (block,)

    completed = _compile(fixture)

    assert len(completed.selected_source_neurons) == 3
    assert completed.report["observed_snapshot_rank"] == 3
    assert completed.report["rank_completion"]["analytic_tangent_snapshot_rows"] == 1
    block_report = completed.report["functional_snapshot_blocks"][0]
    assert block_report["block_weight"] == 4.0
    assert block_report["effective_row_weights_sha256"] == (
        lfm2_authenticated_tensor_sha256(torch.tensor([9.0], dtype=torch.float64))
    )
    assert (
        completed.report["functional_snapshot_blocks_enter_fit_w2_correction"] is False
    )
    # A requested row-amplitude scale of 2 is represented by energy weight 2**2.
    assert torch.equal(
        tangent_rows * 2.0 * 1.5,
        tangent_rows * (block.block_weight * block.row_sample_weights[0]) ** 0.5,
    )


def test_source_is_immutable_and_compilation_never_builds_autograd_graphs() -> None:
    fixture = _fixture()
    fixture["source_gate"].requires_grad_(True)
    fixture["source_up"].requires_grad_(True)
    fixture["source_down"].requires_grad_(True)
    before = lfm2_mlp_qdeim_source_state_sha256(
        fixture["source_gate"],
        fixture["source_up"],
        fixture["source_down"],
    )

    result = _compile(fixture)

    after = lfm2_mlp_qdeim_source_state_sha256(
        fixture["source_gate"],
        fixture["source_up"],
        fixture["source_down"],
    )
    assert before == after
    assert all(
        tensor.grad is None
        for tensor in (
            fixture["source_gate"],
            fixture["source_up"],
            fixture["source_down"],
        )
    )
    for tensor in (
        result.gate_weight,
        result.up_weight,
        result.down_weight_structural_prior,
        result.functional_pod_basis,
        result.interpolation_reconstruction,
    ):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
    assert result.report["autograd_grad_calls"] == 0
    assert result.report["backward_calls"] == 0
    assert result.report["optimizer_steps"] == 0
    assert result.report["training_steps"] == 0


def test_tampered_and_malformed_inputs_fail_closed() -> None:
    fixture = _fixture()
    with pytest.raises(Lfm2MlpQdeimError, match="source W1/W3/W2 state changed"):
        _compile(fixture, expected_source_state_sha256="0" * 64)

    expected_fit = lfm2_mlp_qdeim_fit_evidence_sha256(
        fixture["fit_cases"], fixture["case_weights"]
    )
    fixture["fit_cases"][0].parent_post_swiglu[0, 0] += 0.125
    with pytest.raises(Lfm2MlpQdeimError, match="FIT Q-DEIM evidence changed"):
        _compile(fixture, expected_fit_evidence_sha256=expected_fit)

    malformed = _fixture()
    malformed["target_gate_template"] = torch.zeros((2, 2), dtype=torch.float32)
    with pytest.raises(Lfm2MlpQdeimError, match="BF16"):
        _compile(malformed)

    nonfinite = _fixture()
    nonfinite["fit_cases"][0].parent_post_swiglu[0, 0] = float("nan")
    with pytest.raises(Lfm2MlpQdeimError, match="non-finite"):
        _compile(nonfinite)


def test_tampered_structural_snapshot_and_wrong_width_fail_closed() -> None:
    fixture = _fixture(target_intermediate=3)
    rows = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    block = Lfm2MlpQdeimFunctionalSnapshotBlock(
        block_id="tangent",
        snapshot_kind="analytic_child_lifted_parent_swiglu_tangent",
        functional_rows=rows,
        generator_receipt_sha256="b" * 64,
        block_weight=1.0,
        row_sample_weights=(1.0,),
    )
    fixture["functional_blocks"] = (block,)
    expected = lfm2_mlp_qdeim_functional_snapshot_evidence_sha256((block,))
    rows[0, 3] = 2.0
    with pytest.raises(
        Lfm2MlpQdeimError,
        match="functional snapshot evidence changed",
    ):
        _compile(
            fixture,
            expected_functional_snapshot_evidence_sha256=expected,
        )

    wrong_width = replace(
        block,
        functional_rows=torch.ones((1, 5), dtype=torch.float64),
    )
    fixture["functional_blocks"] = (wrong_width,)
    with pytest.raises(Lfm2MlpQdeimError, match="parent post-SwiGLU"):
        _compile(fixture)
