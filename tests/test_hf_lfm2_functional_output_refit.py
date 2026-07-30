from __future__ import annotations

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_functional_output_refit import (
    LFM2_FUNCTIONAL_OUTPUT_REFIT_PROGRESS_SCHEMA,
    LFM2_FUNCTIONAL_OUTPUT_REFIT_REQUEST_SCHEMA,
    LFM2_FUNCTIONAL_OUTPUT_REFIT_RESULT_SCHEMA,
    Lfm2FunctionalOutputRefitError,
    authenticate_lfm2_functional_output_refit_result,
    build_lfm2_functional_output_sparse_design,
    lfm2_functional_output_refit_request,
    refit_lfm2_functional_output_coefficients,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (
    compile_lfm2_sparse_semantic_codebook,
)


def _codebook_case(
    *,
    active_k: int,
    storage_dtype: torch.dtype = torch.float32,
):
    generator = torch.Generator().manual_seed(7)
    target_count = max(active_k, 4)
    hidden_size = 8
    anchors = torch.randn(
        target_count, hidden_size, generator=generator, dtype=torch.float32
    )
    mixture = torch.linspace(-0.75, 0.9, target_count, dtype=torch.float32)
    mixture = mixture / mixture.abs().sum()
    extra = mixture @ anchors
    donor = torch.cat((anchors, extra.unsqueeze(0)), dim=0).to(storage_dtype)
    anchors = anchors.to(storage_dtype)
    mapping = torch.arange(target_count, dtype=torch.int64)
    codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        maximum_candidates=target_count,
        active_k=active_k,
        candidate_ridge=1e-5,
        refit_ridge=1e-5,
        donor_rows_per_chunk=target_count + 1,
        selection_device="cpu",
        selection_dtype=torch.float64,
        solve_device="cpu",
        solve_dtype=torch.float64,
    )
    metric = torch.diag(torch.linspace(0.25, 2.0, hidden_size))
    return donor, anchors, codebook, metric


def test_k32_refit_derives_support_width_and_keeps_input_channel_immutable() -> None:
    donor, anchors, codebook, metric = _codebook_case(
        active_k=32, storage_dtype=torch.bfloat16
    )
    before = codebook.semantic_code_coefficients.clone()
    events: list[dict[str, object]] = []

    result = refit_lfm2_functional_output_coefficients(
        codebook=codebook,
        donor_table=donor,
        target_table=anchors,
        functional_metric=metric,
        functional_weight=0.001,
        solve_ridge=1e-4,
        solve_device="cpu",
        solve_dtype=torch.float64,
        rows_per_chunk=1,
        progress=lambda event: events.append(dict(event)),
    )

    assert torch.equal(codebook.semantic_code_coefficients, before)
    assert (
        result.output_semantic_code_coefficients.data_ptr()
        != codebook.semantic_code_coefficients.data_ptr()
    )
    assert result.output_semantic_code_coefficients.dtype == torch.float32
    receipt = result.as_dict()
    assert receipt["schema"] == LFM2_FUNCTIONAL_OUTPUT_REFIT_RESULT_SCHEMA
    assert receipt["maximum_support_width"] == 32
    assert receipt["support_width_histogram"] == {"1": 32, "32": 1}
    assert receipt["input_output_channels_separate"] is True
    assert receipt["input_coefficients_mutated"] is False
    assert receipt["fixed_authenticated_supports_reused"] is True
    assert receipt["solve_failure_count"] == 0
    assert receipt["dtype_contract"] == {
        "donor_table_storage_dtype": "bfloat16",
        "target_table_storage_dtype": "bfloat16",
        "functional_metric_storage_dtype": "float32",
        "solve_dtype": "float64",
        "publication_dtype": "float32",
        "solve_device": "cpu",
        "half_and_bfloat_inputs_promoted_before_linear_algebra": True,
    }
    assert all(value == 0 for value in receipt["operation_audit"].values())

    event_names = [event["event"] for event in events]
    assert event_names == [
        "refit-start",
        "cache-identity-ready",
        "support-group-start",
        "solve-chunk-complete",
        "support-group-complete",
        "refit-complete",
    ]
    assert all(
        event["schema"] == LFM2_FUNCTIONAL_OUTPUT_REFIT_PROGRESS_SCHEMA
        for event in events
    )
    assert len({event["cache_key"] for event in events}) == 1
    chunk = next(event for event in events if event["event"] == "solve-chunk-complete")
    assert chunk["support_width"] == 32
    assert chunk["chunk_row_start"] == 0
    assert chunk["chunk_row_stop"] == 1


@pytest.mark.parametrize("active_k", [2, 4, 32])
def test_authenticated_support_width_is_not_hard_coded(active_k: int) -> None:
    donor, anchors, codebook, metric = _codebook_case(active_k=active_k)

    result = refit_lfm2_functional_output_coefficients(
        codebook=codebook,
        donor_table=donor,
        target_table=anchors,
        functional_metric=metric,
        functional_weight=0.001,
        solve_ridge=1e-4,
        solve_dtype="fp64",
        rows_per_chunk=3,
    )

    assert result.as_dict()["maximum_support_width"] == active_k
    assert result.as_dict()["support_width_histogram"][str(active_k)] == 1
    design = build_lfm2_functional_output_sparse_design(
        codebook, result.output_semantic_code_coefficients
    )
    assert design.layout == torch.sparse_coo
    assert tuple(design.shape) == (int(donor.shape[0]), int(anchors.shape[0]))
    reconstructed = torch.sparse.mm(design, anchors.float())
    torch.testing.assert_close(
        reconstructed[: int(anchors.shape[0])], anchors.float(), rtol=0, atol=0
    )


def test_cache_identity_excludes_chunking_but_binds_all_numeric_math(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor, anchors, codebook, metric = _codebook_case(active_k=4)
    arguments = {
        "codebook": codebook,
        "donor_table": donor,
        "target_table": anchors,
        "functional_metric": metric,
        "functional_weight": 0.001,
        "solve_ridge": 1e-4,
        "solve_device": "cpu",
        "solve_dtype": "fp64",
        "rows_per_chunk": 2,
    }

    first = lfm2_functional_output_refit_request(**arguments)
    different_chunking = lfm2_functional_output_refit_request(
        **{**arguments, "rows_per_chunk": 17}
    )
    changed_weight = lfm2_functional_output_refit_request(
        **{**arguments, "functional_weight": 0.01}
    )
    changed_ridge = lfm2_functional_output_refit_request(
        **{**arguments, "solve_ridge": 2e-4}
    )
    changed_dtype = lfm2_functional_output_refit_request(
        **{**arguments, "solve_dtype": "fp32"}
    )
    changed_metric_value = metric.clone()
    changed_metric_value[0, 0] += 0.125
    changed_metric = lfm2_functional_output_refit_request(
        **{**arguments, "functional_metric": changed_metric_value}
    )
    changed_donor_value = donor.clone()
    changed_donor_value[-1, 0] += 0.125
    changed_donor = lfm2_functional_output_refit_request(
        **{**arguments, "donor_table": changed_donor_value}
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    changed_device = lfm2_functional_output_refit_request(
        **{**arguments, "solve_device": "cuda:0"}
    )

    assert first["schema"] == LFM2_FUNCTIONAL_OUTPUT_REFIT_REQUEST_SCHEMA
    assert first["cache_key"] == different_chunking["cache_key"]
    assert first["execution_contract"]["rows_per_chunk"] == 2
    assert different_chunking["execution_contract"]["rows_per_chunk"] == 17
    assert (
        first["execution_contract"]["progress_logging_changes_math_identity"]
        is False
    )
    assert first["cache_key"] != changed_weight["cache_key"]
    assert first["cache_key"] != changed_ridge["cache_key"]
    assert first["cache_key"] != changed_dtype["cache_key"]
    assert first["cache_key"] != changed_metric["cache_key"]
    assert first["cache_key"] != changed_donor["cache_key"]
    assert first["cache_key"] != changed_device["cache_key"]
    assert first["support_width_histogram"] == {"1": 4, "4": 1}
    assert first["operation_audit"]["model_forward_calls"] == 0
    assert first["operation_audit"]["training_steps"] == 0


def test_loaded_result_must_match_expected_math_cache_key() -> None:
    donor, anchors, codebook, metric = _codebook_case(active_k=4)
    result = refit_lfm2_functional_output_coefficients(
        codebook=codebook,
        donor_table=donor,
        target_table=anchors,
        functional_metric=metric,
        functional_weight=0.001,
        solve_ridge=1e-4,
    )

    loaded = authenticate_lfm2_functional_output_refit_result(
        output_semantic_code_coefficients=(
            result.output_semantic_code_coefficients.clone()
        ),
        receipt=result.as_dict(),
        expected_cache_key=result.cache_key,
    )

    assert loaded.result_sha256 == result.result_sha256
    with pytest.raises(
        Lfm2FunctionalOutputRefitError,
        match="belongs to a different math cache key",
    ):
        authenticate_lfm2_functional_output_refit_result(
            output_semantic_code_coefficients=(
                result.output_semantic_code_coefficients.clone()
            ),
            receipt=result.as_dict(),
            expected_cache_key="f" * 64,
        )

    tampered = result.as_dict()
    tampered["request"]["functional_weight"] = 0.5
    tampered_body = {
        key: value for key, value in tampered.items() if key != "result_sha256"
    }
    tampered["result_sha256"] = canonical_sha256(tampered_body)
    with pytest.raises(
        Lfm2FunctionalOutputRefitError,
        match="request math identity is invalid",
    ):
        authenticate_lfm2_functional_output_refit_result(
            output_semantic_code_coefficients=(
                result.output_semantic_code_coefficients.clone()
            ),
            receipt=tampered,
            expected_cache_key=result.cache_key,
        )


def test_solve_dtype_fails_closed_instead_of_silently_using_half_precision() -> None:
    donor, anchors, codebook, metric = _codebook_case(active_k=2)

    with pytest.raises(
        Lfm2FunctionalOutputRefitError,
        match="solve_dtype must be explicit FP32 or FP64",
    ):
        lfm2_functional_output_refit_request(
            codebook=codebook,
            donor_table=donor,
            target_table=anchors,
            functional_metric=metric,
            functional_weight=0.001,
            solve_ridge=1e-4,
            solve_dtype=torch.bfloat16,
        )
