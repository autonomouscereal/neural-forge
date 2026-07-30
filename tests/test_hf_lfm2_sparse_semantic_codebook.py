from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_sparse_semantic_codebook import (
    LFM2_SPARSE_SEMANTIC_CODEBOOK_SCHEMA,
    Lfm2SparseSemanticCodebook,
    Lfm2SparseSemanticCodebookError,
    compile_lfm2_sparse_semantic_codebook,
    discover_lfm2_sparse_semantic_candidates,
    load_lfm2_sparse_semantic_candidate_chunk,
    load_lfm2_sparse_semantic_candidate_inventory,
    load_lfm2_sparse_semantic_codebook,
    load_lfm2_sparse_semantic_coefficient_chunk,
    reclose_lfm2_sparse_semantic_codebook_norms,
    save_lfm2_sparse_semantic_candidate_chunk,
    save_lfm2_sparse_semantic_candidate_inventory,
    save_lfm2_sparse_semantic_codebook,
    save_lfm2_sparse_semantic_coefficient_chunk,
)


def _case(dtype: torch.dtype = torch.float32):
    anchors = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    donor = torch.cat(
        (
            anchors,
            torch.tensor(
                [
                    [0.75, -0.25, 0.0, 0.0],
                    [0.0, 0.6, 0.0, 0.8],
                    [-0.5, 0.0, 0.5, 0.0],
                ],
                dtype=dtype,
            ),
        )
    )
    return donor, anchors, torch.tensor([0, 1, 2, 3], dtype=torch.int64)


def _compile(dtype: torch.dtype = torch.float32):
    donor, anchors, mapping = _case(dtype)
    candidates = discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        mapping,
        maximum_candidates=4,
        donor_rows_per_chunk=2,
        selection_dtype=torch.float32,
    )
    codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=candidates,
        maximum_candidates=4,
        active_k=2,
        candidate_ridge=1e-8,
        refit_ridge=1e-8,
        donor_rows_per_chunk=2,
        solve_device="cpu",
        solve_dtype=torch.float64,
    )
    return donor, anchors, mapping, candidates, codebook


def _projection_case(
    dtype: torch.dtype = torch.float32,
    *,
    extra_row: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    anchors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype
    )
    donor = torch.cat((anchors, torch.tensor([extra_row], dtype=dtype)))
    mapping = torch.tensor([0, 1], dtype=torch.int64)
    codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        maximum_candidates=2,
        active_k=2,
        candidate_ridge=1e-8,
        refit_ridge=1e-8,
        donor_rows_per_chunk=2,
        selection_dtype=torch.float64,
        solve_dtype=torch.float64,
    )
    return donor, anchors, mapping, codebook


def test_direct_anchors_are_exact_one_hot_and_coefficients_are_counted_child_state() -> None:
    donor, anchors, mapping, candidates, codebook = _compile()

    assert codebook.semantic_code_offsets.device.type == "cpu"
    assert codebook.semantic_code_offsets.dtype == torch.int64
    assert codebook.semantic_code_target_ids.dtype == torch.int64
    assert codebook.semantic_code_coefficients.dtype == torch.float32
    assert torch.equal(codebook.target_to_donor, mapping)
    for target_id, donor_id in enumerate(mapping.tolist()):
        start = int(codebook.semantic_code_offsets[donor_id])
        stop = int(codebook.semantic_code_offsets[donor_id + 1])
        assert stop - start == 1
        assert int(codebook.semantic_code_target_ids[start]) == target_id
        assert codebook.semantic_code_coefficients[start].item() == 1.0

    reconstructed = codebook.reconstruct(anchors)
    assert torch.allclose(reconstructed, donor, atol=2e-5, rtol=2e-5)
    receipt = codebook.as_dict()
    expected = int(mapping.numel()) + (int(donor.shape[0]) - int(mapping.numel())) * 2
    assert receipt["schema"] == LFM2_SPARSE_SEMANTIC_CODEBOOK_SCHEMA
    assert receipt["continuous_fold_coefficient_count"] == expected
    assert receipt["continuous_fold_coefficient_bytes"] == expected * 4
    assert receipt["effective_child_parameter_addition_from_semantic_bridge"] == expected
    assert receipt["continuous_coefficients_count_toward_effective_child_parameters"]
    assert receipt["representation_is_fixed_bridge_state_not_model_sparsification"]
    assert receipt["semantic_code_offset_bytes"] == (int(donor.shape[0]) + 1) * 8
    assert receipt["semantic_code_target_id_bytes"] == expected * 8
    assert receipt["candidate_artifact_sha256"] == candidates.artifact_sha256
    assert len(receipt["row_diagnostics"]) == int(donor.shape[0])
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["global_map_solve_calls"] == 0


def test_absolute_cosine_candidate_search_keeps_anti_aligned_linear_anchor() -> None:
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    donor = torch.cat((anchors, torch.tensor([[-1.0, 0.01]])))
    candidates = discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        [0, 1],
        maximum_candidates=1,
        donor_rows_per_chunk=1,
    )
    start = int(candidates.candidate_offsets[2])
    stop = int(candidates.candidate_offsets[3])
    assert candidates.candidate_target_ids[start:stop].tolist() == [0]
    diagnostic = candidates.as_dict()["row_diagnostics"][2]
    assert diagnostic["nearest_candidate_cosine"] < -0.99
    assert diagnostic["nearest_candidate_absolute_cosine"] > 0.99
    assert candidates.as_dict()["cosine_definition"].startswith("absolute-dot")


@pytest.mark.parametrize(
    "input_dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
)
@pytest.mark.parametrize("solve_dtype", (torch.float32, torch.float64))
def test_all_supported_input_dtypes_use_explicit_compute_and_cpu_fp32_publication(
    input_dtype: torch.dtype,
    solve_dtype: torch.dtype,
) -> None:
    donor, anchors, mapping = _case(input_dtype)
    codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        maximum_candidates=3,
        active_k=2,
        donor_rows_per_chunk=3,
        selection_device="cpu",
        selection_dtype=torch.float32,
        solve_device="cpu",
        solve_dtype=solve_dtype,
    )
    receipt = codebook.as_dict()
    assert receipt["selection_placement"] == {
        "device": "cpu",
        "dtype": "float32",
    }
    assert receipt["solve_placement"] == {
        "device": "cpu",
        "dtype": str(solve_dtype).removeprefix("torch."),
    }
    assert receipt["publication_placement"] == {
        "device": "cpu",
        "dtype": "float32",
    }
    assert codebook.semantic_code_coefficients.dtype == torch.float32
    assert bool(torch.isfinite(codebook.semantic_code_coefficients).all())


def test_candidate_chunks_are_reusable_independently_of_active_k() -> None:
    donor, anchors, mapping = _case()
    published = []
    first = discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        mapping,
        maximum_candidates=4,
        donor_rows_per_chunk=2,
        candidate_chunk_publisher=published.append,
    )
    by_request = {
        checkpoint.as_dict()["request_sha256"]: checkpoint
        for checkpoint in published
    }
    requests = []

    def loader(request):
        requests.append(dict(request))
        return by_request[request["request_sha256"]]

    second = discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        mapping,
        maximum_candidates=4,
        donor_rows_per_chunk=2,
        candidate_chunk_loader=loader,
    )
    assert torch.equal(first.candidate_offsets, second.candidate_offsets)
    assert torch.equal(first.candidate_target_ids, second.candidate_target_ids)
    assert first.content_sha256 == second.content_sha256
    assert len(requests) == len(published) == 4
    assert second.as_dict()["computed_chunk_count"] == 0
    assert second.as_dict()["checkpoint_reused_chunk_count"] == 4

    k1 = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=first,
        maximum_candidates=4,
        active_k=1,
        donor_rows_per_chunk=2,
    )
    k2 = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=first,
        maximum_candidates=4,
        active_k=2,
        donor_rows_per_chunk=2,
    )
    assert k1.as_dict()["candidate_artifact_sha256"] == first.artifact_sha256
    assert k2.as_dict()["candidate_artifact_sha256"] == first.artifact_sha256
    assert k1.as_dict()["candidate_phase_cache_key"] == k2.as_dict()[
        "candidate_phase_cache_key"
    ]
    assert k1.as_dict()["phase_cache_key"] != k2.as_dict()["phase_cache_key"]


def test_coefficient_chunks_are_exactly_resumable() -> None:
    donor, anchors, mapping = _case()
    candidates = discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        mapping,
        maximum_candidates=4,
        donor_rows_per_chunk=2,
    )
    published = []
    first = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=candidates,
        maximum_candidates=4,
        active_k=2,
        donor_rows_per_chunk=2,
        coefficient_chunk_publisher=published.append,
    )
    by_request = {
        checkpoint.as_dict()["request_sha256"]: checkpoint
        for checkpoint in published
    }
    second = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=candidates,
        maximum_candidates=4,
        active_k=2,
        donor_rows_per_chunk=2,
        coefficient_chunk_loader=lambda request: by_request[
            request["request_sha256"]
        ],
    )
    assert torch.equal(first.semantic_code_offsets, second.semantic_code_offsets)
    assert torch.equal(first.semantic_code_target_ids, second.semantic_code_target_ids)
    assert torch.equal(
        first.semantic_code_coefficients, second.semantic_code_coefficients
    )
    assert len(published) == 4
    assert second.as_dict()["computed_chunk_count"] == 0
    assert second.as_dict()["checkpoint_reused_chunk_count"] == 4


def test_progress_is_hash_chained_and_row_chunks_are_bounded() -> None:
    donor, anchors, mapping = _case()
    records = []
    codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        maximum_candidates=3,
        active_k=2,
        donor_rows_per_chunk=2,
        progress_callback=lambda record: records.append(dict(record)),
    )
    assert records
    phases = {record["phase"] for record in records}
    assert phases == {"candidate-discovery", "active-k-local-ridge-refit"}
    for phase in phases:
        phase_records = [record for record in records if record["phase"] == phase]
        previous = None
        for index, record in enumerate(phase_records):
            assert record["progress_index"] == index
            assert record["previous_progress_sha256"] == previous
            body = {key: value for key, value in record.items() if key != "progress_sha256"}
            assert record["progress_sha256"] == canonical_sha256(body)
            previous = record["progress_sha256"]
    chunk_records = [record for record in records if record["event"] == "row-chunk-complete"]
    assert all(record["donor_row_stop"] - record["donor_row_start"] <= 2 for record in chunk_records)
    assert codebook.as_dict()["donor_rows_per_chunk"] == 2


def test_artifacts_and_individual_checkpoints_roundtrip_with_authentication(
    tmp_path,
) -> None:
    donor, anchors, mapping = _case()
    candidate_chunks = []
    candidates = discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        mapping,
        maximum_candidates=4,
        donor_rows_per_chunk=3,
        candidate_chunk_publisher=candidate_chunks.append,
    )
    coefficient_chunks = []
    codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=candidates,
        maximum_candidates=4,
        active_k=2,
        donor_rows_per_chunk=3,
        coefficient_chunk_publisher=coefficient_chunks.append,
    )

    save_lfm2_sparse_semantic_candidate_inventory(candidates, tmp_path / "candidates")
    loaded_candidates = load_lfm2_sparse_semantic_candidate_inventory(
        tmp_path / "candidates",
        expected_artifact_sha256=candidates.artifact_sha256,
    )
    assert torch.equal(loaded_candidates.candidate_target_ids, candidates.candidate_target_ids)

    save_lfm2_sparse_semantic_codebook(codebook, tmp_path / "codebook")
    loaded_codebook = load_lfm2_sparse_semantic_codebook(
        tmp_path / "codebook",
        expected_artifact_sha256=codebook.artifact_sha256,
    )
    assert torch.equal(
        loaded_codebook.semantic_code_coefficients,
        codebook.semantic_code_coefficients,
    )

    save_lfm2_sparse_semantic_candidate_chunk(
        candidate_chunks[0], tmp_path / "candidate-chunk"
    )
    loaded_candidate_chunk = load_lfm2_sparse_semantic_candidate_chunk(
        tmp_path / "candidate-chunk",
        expected_checkpoint_sha256=candidate_chunks[0].as_dict()[
            "checkpoint_sha256"
        ],
    )
    assert torch.equal(
        loaded_candidate_chunk.candidate_target_ids,
        candidate_chunks[0].candidate_target_ids,
    )

    save_lfm2_sparse_semantic_coefficient_chunk(
        coefficient_chunks[0], tmp_path / "coefficient-chunk"
    )
    loaded_coefficient_chunk = load_lfm2_sparse_semantic_coefficient_chunk(
        tmp_path / "coefficient-chunk",
        expected_checkpoint_sha256=coefficient_chunks[0].as_dict()[
            "checkpoint_sha256"
        ],
    )
    assert torch.equal(
        loaded_coefficient_chunk.semantic_code_coefficients,
        coefficient_chunks[0].semantic_code_coefficients,
    )

    binary = tmp_path / "codebook" / "semantic_code_coefficients.bin"
    payload = bytearray(binary.read_bytes())
    payload[0] ^= 1
    binary.write_bytes(payload)
    with pytest.raises(Lfm2SparseSemanticCodebookError, match="authentication"):
        load_lfm2_sparse_semantic_codebook(tmp_path / "codebook")


def test_invalid_mapping_compute_dtype_and_tampered_receipt_fail_closed() -> None:
    donor, anchors, _mapping, _candidates, codebook = _compile()
    with pytest.raises(Lfm2SparseSemanticCodebookError, match="injective"):
        discover_lfm2_sparse_semantic_candidates(
            donor, anchors, [0, 0, 2, 3]
        )
    with pytest.raises(Lfm2SparseSemanticCodebookError, match="FP32 or FP64"):
        compile_lfm2_sparse_semantic_codebook(
            donor,
            anchors,
            [0, 1, 2, 3],
            active_k=2,
            selection_dtype=torch.float16,
        )
    with pytest.raises(Lfm2SparseSemanticCodebookError, match="FP32 or FP64"):
        compile_lfm2_sparse_semantic_codebook(
            donor,
            anchors,
            [0, 1, 2, 3],
            active_k=2,
            solve_dtype=torch.bfloat16,
        )
    tampered = deepcopy(codebook.as_dict())
    tampered["continuous_fold_coefficient_count"] -= 1
    tampered["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "artifact_sha256"}
    )
    with pytest.raises(Lfm2SparseSemanticCodebookError, match="binding"):
        Lfm2SparseSemanticCodebook(
            codebook.semantic_code_offsets,
            codebook.semantic_code_target_ids,
            codebook.semantic_code_coefficients,
            codebook.target_to_donor,
            tampered,
        )


def test_explicit_narrow_compute_overflow_fails_closed_instead_of_publishing_nan() -> None:
    anchors = torch.tensor([[1e100, 0.0], [0.0, 1e100]], dtype=torch.float64)
    donor = torch.cat((anchors, torch.tensor([[1e100, 1e100]], dtype=torch.float64)))
    with pytest.raises(Lfm2SparseSemanticCodebookError, match="became non-finite"):
        discover_lfm2_sparse_semantic_candidates(
            donor,
            anchors,
            [0, 1],
            maximum_candidates=2,
            selection_dtype=torch.float32,
        )


@pytest.mark.parametrize("gamma", (0.5, 1.0))
def test_norm_reclosure_bakes_expected_power_scale_without_adding_state(
    gamma: float,
) -> None:
    donor, anchors, mapping, base = _projection_case()
    before = base.reconstruct(anchors)
    rho = float(
        torch.linalg.vector_norm(before[2])
        / torch.linalg.vector_norm(donor[2])
    )
    result = reclose_lfm2_sparse_semantic_codebook_norms(
        base,
        donor,
        anchors,
        gamma=gamma,
        max_scale=10.0,
        donor_rows_per_chunk=2,
    )
    after = result.reconstruct(anchors)
    expected_scale = rho ** (-gamma)
    expected_after_ratio = rho ** (1.0 - gamma)

    assert torch.equal(result.semantic_code_offsets, base.semantic_code_offsets)
    assert torch.equal(
        result.semantic_code_target_ids, base.semantic_code_target_ids
    )
    assert torch.equal(result.target_to_donor, mapping)
    assert result.semantic_code_coefficients.numel() == (
        base.semantic_code_coefficients.numel()
    )
    start, stop = int(result.semantic_code_offsets[2]), int(
        result.semantic_code_offsets[3]
    )
    assert torch.allclose(
        result.semantic_code_coefficients[start:stop],
        base.semantic_code_coefficients[start:stop] * expected_scale,
        atol=2e-6,
        rtol=2e-6,
    )
    observed_after_ratio = float(
        torch.linalg.vector_norm(after[2])
        / torch.linalg.vector_norm(donor[2])
    )
    assert observed_after_ratio == pytest.approx(expected_after_ratio, rel=2e-6)

    receipt = result.as_dict()
    closure = receipt["norm_reclosure"]
    assert closure["gamma"] == gamma
    assert closure["continuous_coefficient_count_added"] == 0
    assert closure["effective_child_parameter_count_added"] == 0
    assert closure["routing_metadata_values_added"] == 0
    assert closure["unresolved_non_direct_row_count"] == 0
    assert closure["before_norm_ratio_quantiles"]["p50"] == pytest.approx(
        rho, rel=2e-6
    )
    assert closure["after_norm_ratio_quantiles"]["p50"] == pytest.approx(
        expected_after_ratio, rel=2e-6
    )
    assert closure["applied_scale_quantiles"]["p50"] == pytest.approx(
        expected_scale, rel=2e-6
    )
    assert closure["row_reclosure_diagnostics_sha256"] == canonical_sha256(
        closure["row_reclosure_diagnostics"]
    )
    assert closure["row_reclosure_diagnostics"] == [
        {
            "donor_id": 2,
            "semantic_code_edge_count": 2,
            "before_norm_ratio": pytest.approx(rho, rel=2e-6),
            "after_norm_ratio": pytest.approx(expected_after_ratio, rel=2e-6),
            "applied_scale": pytest.approx(expected_scale, rel=2e-6),
            "before_cosine": pytest.approx(rho, rel=2e-6),
            "after_cosine": pytest.approx(rho, rel=2e-6),
            "before_relative_l2_error": pytest.approx(
                (1.0 - rho**2) ** 0.5, rel=2e-6
            ),
            "after_relative_l2_error": pytest.approx(
                (
                    1.0
                    + expected_after_ratio**2
                    - 2.0 * expected_after_ratio * rho
                )
                ** 0.5,
                rel=2e-6,
            ),
            "amplified": True,
            "scale_cap_applied": False,
            "unresolved_low_cosine": False,
        }
    ]
    assert closure["maximum_absolute_cosine_change"] < 1e-6
    assert closure["reported_cosines_clamped_to_closed_unit_interval"]
    assert closure["cosine_roundoff_clamp_count"] >= 0
    assert closure["maximum_absolute_cosine_roundoff_clamp"] >= 0.0
    assert closure["operation_audit"]["training_steps"] == 0
    assert receipt["norm_reclosure_sha256"] == closure[
        "norm_reclosure_sha256"
    ]


def test_exact_norm_reclosure_preserves_direct_bits_and_reports_l2_worsening() -> None:
    donor, anchors, mapping, base = _projection_case()
    result = reclose_lfm2_sparse_semantic_codebook_norms(
        base,
        donor,
        anchors,
        gamma=1.0,
        max_scale=10.0,
        donor_rows_per_chunk=1,
    )
    for target_id, donor_id in enumerate(mapping.tolist()):
        start, stop = int(result.semantic_code_offsets[donor_id]), int(
            result.semantic_code_offsets[donor_id + 1]
        )
        assert stop - start == 1
        assert int(result.semantic_code_target_ids[start]) == target_id
        assert torch.equal(
            result.semantic_code_coefficients[start:stop],
            base.semantic_code_coefficients[start:stop],
        )
        assert result.semantic_code_coefficients[start].item() == 1.0
    closure = result.as_dict()["norm_reclosure"]
    assert closure["protected_direct_rows_preserved_bit_exact"]
    assert closure["l2_worsened_non_direct_row_count"] == 1
    assert closure["l2_worsening_is_reported_not_hidden"]
    assert (
        closure["after_relative_l2_error_quantiles"]["p50"]
        > closure["before_relative_l2_error_quantiles"]["p50"]
    )


def test_norm_reclosure_cap_and_minimum_cosine_leave_bounded_honest_rows() -> None:
    donor, anchors, _mapping, base = _projection_case()
    capped = reclose_lfm2_sparse_semantic_codebook_norms(
        base,
        donor,
        anchors,
        gamma=1.0,
        max_scale=1.05,
    )
    capped_closure = capped.as_dict()["norm_reclosure"]
    assert capped_closure["scale_cap_applied_non_direct_row_count"] == 1
    assert capped_closure["applied_scale_quantiles"]["maximum"] == pytest.approx(
        1.05
    )
    assert capped_closure["after_norm_ratio_quantiles"]["p50"] < 1.0

    low_donor, low_anchors, _mapping, low_base = _projection_case(
        extra_row=(1.0, 0.0, 10.0)
    )
    thresholded = reclose_lfm2_sparse_semantic_codebook_norms(
        low_base,
        low_donor,
        low_anchors,
        gamma=1.0,
        max_scale=10.0,
        minimum_cosine_to_amplify=0.2,
    )
    threshold_closure = thresholded.as_dict()["norm_reclosure"]
    assert threshold_closure["unresolved_non_direct_row_count"] == 1
    assert threshold_closure["amplified_non_direct_row_count"] == 0
    assert threshold_closure["applied_scale_quantiles"]["p50"] == 1.0
    assert torch.equal(
        thresholded.semantic_code_coefficients,
        low_base.semantic_code_coefficients,
    )


@pytest.mark.parametrize(
    ("input_dtype", "compute_dtype"),
    (
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float64),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ),
)
def test_norm_reclosure_supports_all_storage_dtypes_and_persists(
    input_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    tmp_path,
) -> None:
    donor, anchors, _mapping, base = _projection_case(input_dtype)
    result = reclose_lfm2_sparse_semantic_codebook_norms(
        base,
        donor,
        anchors,
        gamma=0.5,
        max_scale=2.0,
        compute_dtype=compute_dtype,
    )
    assert result.semantic_code_coefficients.dtype == torch.float32
    assert bool(torch.isfinite(result.semantic_code_coefficients).all())
    directory = tmp_path / str(input_dtype).replace("torch.", "")
    save_lfm2_sparse_semantic_codebook(result, directory)
    loaded = load_lfm2_sparse_semantic_codebook(
        directory, expected_artifact_sha256=result.artifact_sha256
    )
    assert torch.equal(
        loaded.semantic_code_coefficients,
        result.semantic_code_coefficients,
    )
    assert loaded.as_dict()["norm_reclosure_sha256"] == result.as_dict()[
        "norm_reclosure_sha256"
    ]


def test_norm_reclosure_all_direct_codebook_is_a_persistable_identity(
    tmp_path,
) -> None:
    anchors = torch.eye(3, dtype=torch.float32)
    donor = anchors.clone()
    mapping = torch.arange(3, dtype=torch.int64)
    base = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        maximum_candidates=3,
        active_k=2,
        candidate_ridge=1e-8,
        refit_ridge=1e-8,
        solve_dtype=torch.float64,
    )

    result = reclose_lfm2_sparse_semantic_codebook_norms(
        base,
        donor,
        anchors,
        gamma=0.5,
        max_scale=3.0,
    )

    assert torch.equal(result.semantic_code_offsets, base.semantic_code_offsets)
    assert torch.equal(
        result.semantic_code_target_ids, base.semantic_code_target_ids
    )
    assert torch.equal(
        result.semantic_code_coefficients,
        base.semantic_code_coefficients,
    )
    closure = result.as_dict()["norm_reclosure"]
    assert closure["amplified_non_direct_row_count"] == 0
    assert closure["applied_scale_quantiles"] == {
        "count": 0,
        "empty_population": True,
        "minimum": None,
        "p10": None,
        "p25": None,
        "p50": None,
        "p75": None,
        "p90": None,
        "p99": None,
        "maximum": None,
        "mean": None,
    }
    assert closure["maximum_absolute_cosine_change"] == 0.0
    assert closure["mean_absolute_cosine_change"] == 0.0
    assert closure["row_reclosure_diagnostics"] == []
    assert closure["row_reclosure_diagnostics_sha256"] == canonical_sha256([])

    destination = tmp_path / "all-direct"
    save_lfm2_sparse_semantic_codebook(result, destination)
    loaded = load_lfm2_sparse_semantic_codebook(
        destination,
        expected_artifact_sha256=result.artifact_sha256,
    )
    assert torch.equal(
        loaded.semantic_code_coefficients,
        base.semantic_code_coefficients,
    )


@pytest.mark.parametrize(
    "extra_row",
    ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)
def test_norm_reclosure_fails_closed_on_zero_source_or_reconstruction_norm(
    extra_row: tuple[float, float, float],
) -> None:
    donor, anchors, _mapping, base = _projection_case(extra_row=extra_row)
    with pytest.raises(Lfm2SparseSemanticCodebookError, match="zero source"):
        reclose_lfm2_sparse_semantic_codebook_norms(
            base,
            donor,
            anchors,
            gamma=0.5,
            max_scale=2.0,
        )


def test_norm_reclosure_rejects_nonfinite_or_invalid_controls() -> None:
    donor, anchors, _mapping, base = _projection_case()
    for kwargs in (
        {"gamma": 0.0, "max_scale": 2.0},
        {"gamma": 0.5, "max_scale": 0.5},
        {
            "gamma": 0.5,
            "max_scale": 2.0,
            "minimum_cosine_to_amplify": 2.0,
        },
        {"gamma": float("nan"), "max_scale": 2.0},
    ):
        with pytest.raises(Lfm2SparseSemanticCodebookError):
            reclose_lfm2_sparse_semantic_codebook_norms(
                base, donor, anchors, **kwargs
            )
    nonfinite = donor.clone()
    nonfinite[2, 0] = float("inf")
    with pytest.raises(Lfm2SparseSemanticCodebookError, match="finite"):
        reclose_lfm2_sparse_semantic_codebook_norms(
            base,
            nonfinite,
            anchors,
            gamma=0.5,
            max_scale=2.0,
        )
