from __future__ import annotations

import copy

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge import hf_lfm2_exact4x_causal_diagnostic as causal


def _sha(value: str) -> str:
    return __import__("hashlib").sha256(value.encode("utf-8")).hexdigest()


def _fake_v3_request() -> dict[str, object]:
    return {
        "request_sha256": _sha("v3-request"),
        "child_package_sha256": _sha("child-package"),
        "child_state_sha256": _sha("child-state"),
        "trace_cache_key": _sha("trace-cache"),
        "trace_selective_session_sha256": _sha("trace-session"),
        "basis_phase_key_sha256": _sha("basis-phase"),
        "basis_completion_sha256": _sha("basis-completion"),
        "basis_map_sha256": _sha("basis-map"),
        "parent_logit_domain_contract_sha256": _sha("logit-domain"),
        "parent_logit_domain_contract": {
            "defined_semantic_row_count": 6,
            "parent_physical_row_capacity": 8,
        },
        "layer_count": 13,
        "runtime_device": "cpu",
        "runtime_dtype": "float32",
    }


def _fake_v3_ledger(request: dict[str, object]) -> dict[str, object]:
    body = {
        "schema": causal.LFM2_EXACT4X_FIRST_DIVERGENCE_LEDGER_SCHEMA,
        "status": "complete-observation",
        "request": request,
        "ordered_batch_audit_sha256s": [_sha("batch-0"), _sha("batch-1")],
        "batch_count": 2,
        "split_summaries": {
            "fit": {"batch_count": 1},
            "heldout": {"batch_count": 1},
        },
        "first_discrete_c_out_parent_winner_mismatch": None,
        "component_first_divergence_is_descriptive_not_thresholded": True,
        "numeric_metrics_are_diagnostic_only": True,
        "numeric_acceptance_decision": None,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "safety_observations_are_separate_not_failures": True,
        "generation_cap_saturation_is_censored_not_failure": True,
        "free_generation_performed": False,
        "operation_audit": {},
    }
    return {**body, "ledger_sha256": canonical_sha256(body)}


def _records() -> tuple[causal.Lfm2CausalRecordBinding, ...]:
    return (
        causal.Lfm2CausalRecordBinding(
            batch_index=0,
            trace_input_sha256=_sha("input-0"),
            trace_record_sha256=_sha("record-0"),
            split="fit",
            v3_batch_audit_sha256=_sha("batch-0"),
        ),
        causal.Lfm2CausalRecordBinding(
            batch_index=1,
            trace_input_sha256=_sha("input-1"),
            trace_record_sha256=_sha("record-1"),
            split="heldout",
            v3_batch_audit_sha256=_sha("batch-1"),
        ),
    )


def _fake_v3_batch_results() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "batch_index": index,
            "batch_audit_sha256": _sha(f"batch-{index}"),
            "trace_input_sha256": _sha(f"input-{index}"),
            "trace_record_sha256": _sha(f"record-{index}"),
            "split": split,
        }
        for index, split in enumerate(("fit", "heldout"))
    )


def _request(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(
        causal,
        "authenticate_lfm2_exact4x_first_divergence_request",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        causal,
        "authenticate_lfm2_exact4x_first_divergence_batch",
        lambda value, **_kwargs: dict(value),
    )
    v3_request = _fake_v3_request()
    return causal.build_lfm2_exact4x_causal_diagnostic_request(
        first_divergence_request=v3_request,
        first_divergence_ledger=_fake_v3_ledger(v3_request),
        first_divergence_batch_results=_fake_v3_batch_results(),
        record_bindings=_records(),
        implementation_identity_sha256=_sha("implementation"),
        checkpoint_hmac_key_id="causal-test-key",
        bootstrap_seed="causal-test-bootstrap",
        bootstrap_resample_count=64,
    )


def _subset_metrics(
    groups: tuple[str, ...] = ("C", "S", "M"),
    *,
    offset: float = 0.0,
) -> dict[tuple[str, ...], dict[str, float]]:
    result = {}
    for row in causal.canonical_intervention_subsets(groups):
        coalition = tuple(row["members"])
        score = float(
            sum(index + 1 for index, group in enumerate(groups) if group in coalition)
        )
        result[coalition] = {
            causal.TERMINAL_ERROR_METRIC: 10.0 + offset - score,
            causal.DEFINED_VOCAB_KL_METRIC: 5.0 + offset - 0.5 * score,
            causal.PARENT_WINNER_MARGIN_METRIC: -2.0 - offset + score,
        }
    return result


def test_default_and_downstream_operator_specs_share_generic_contract() -> None:
    specs = causal.default_lfm2_l0_csm_operator_specs()
    assert [spec.group_id for spec in specs] == ["C", "S", "M"]
    assert specs[1].consumer_module_path == "model.layers.0.conv.out_proj"
    assert specs[2].feature_endpoint == "mlp.post_swiglu"

    downstream = causal.canonical_operator_specs(
        (
            causal.Lfm2CausalOperatorSpec(
                group_id="A10",
                stage_order=0,
                layer_index=10,
                component_kind="attention_write",
                parent_endpoint="operator.output",
                target_module_path="model.layers.10.self_attn",
                hook_kind="module_output",
            ),
            causal.Lfm2CausalOperatorSpec(
                group_id="A12",
                stage_order=1,
                layer_index=12,
                component_kind="attention_write",
                parent_endpoint="operator.output",
                target_module_path="model.layers.12.self_attn",
                hook_kind="module_output",
            ),
            causal.Lfm2CausalOperatorSpec(
                group_id="O",
                stage_order=2,
                layer_index=12,
                component_kind="terminal_output",
                parent_endpoint="residual.post_layer",
                target_module_path="model.layers.12",
                hook_kind="module_output",
            ),
        )
    )
    assert [spec.layer_index for spec in downstream] == [10, 12, 12]
    assert [
        row["mask"]
        for row in causal.canonical_intervention_subsets(
            [spec.group_id for spec in downstream]
        )
    ] == list(range(8))


def test_exact_shapley_and_recursive_harsanyi_reconstruct_three_player_game() -> None:
    groups = ("C", "S", "M")
    values = {}
    for row in causal.canonical_intervention_subsets(groups):
        members = tuple(row["members"])
        value = 5.0
        value += sum({"C": 1.0, "S": 2.0, "M": 3.0}[group] for group in members)
        if len(members) == 3:
            value += 6.0
        values[members] = value

    shapley = causal.exact_three_player_shapley(values, groups)
    assert shapley == pytest.approx({"C": 3.0, "S": 4.0, "M": 5.0})
    dividends = causal.recursive_harsanyi_dividends(values, groups)
    assert dividends[()] == pytest.approx(5.0)
    assert dividends[("C", "S", "M")] == pytest.approx(6.0)
    assert dividends[("C", "S")] == pytest.approx(0.0)
    decomposition = causal.three_player_causal_decomposition(values, groups)
    assert decomposition["shapley_efficiency_remainder"] == pytest.approx(0.0)
    assert decomposition["harsanyi_reconstruction_remainder"] == pytest.approx(0.0)


def test_metric_benefits_are_all_higher_is_better() -> None:
    metrics = _subset_metrics()
    benefits = causal.higher_is_better_benefits(metrics, ("C", "S", "M"))
    grand = ("C", "S", "M")
    assert benefits[causal.TERMINAL_ERROR_BENEFIT][grand] == pytest.approx(6.0)
    assert benefits[causal.DEFINED_VOCAB_KL_BENEFIT][grand] == pytest.approx(3.0)
    assert benefits[causal.PARENT_WINNER_MARGIN_BENEFIT][grand] == pytest.approx(6.0)
    assert all(values[()] == 0.0 for values in benefits.values())


def test_defined_vocab_kl_excludes_trailing_capacity_rows() -> None:
    parent = torch.tensor([[2.0, 0.0, 1000.0, -1000.0]])
    candidate = torch.tensor([[1.0, 0.0, -1000.0, 1000.0]])
    reference = causal.defined_vocab_kl_rows(
        parent[:, :2], candidate[:, :2], defined_semantic_row_count=2
    )
    padded = causal.defined_vocab_kl_rows(
        parent, candidate, defined_semantic_row_count=2
    )
    assert torch.equal(reference, padded)
    assert float(padded.item()) > 0.0


def test_record_weights_and_clustered_bootstrap_use_whole_records() -> None:
    weights = causal.equal_record_mass_weights(("record-a", "record-a", "record-b"))
    assert weights.tolist() == pytest.approx([0.25, 0.25, 0.5])
    first = causal.deterministic_record_clustered_bootstrap(
        [0.0, 2.0, 10.0],
        ["record-a", "record-a", "record-b"],
        seed="bootstrap-v1",
        resample_count=128,
    )
    permuted = causal.deterministic_record_clustered_bootstrap(
        [10.0, 2.0, 0.0],
        ["record-b", "record-a", "record-a"],
        seed="bootstrap-v1",
        resample_count=128,
    )
    assert first == permuted
    assert first["point_estimate"] == pytest.approx(5.5)
    assert first["record_count"] == 2
    assert first["phase_rows_are_not_bootstrap_units"] is True
    assert first["one_sided_bound_confidence_level"] == pytest.approx(0.95)
    assert first["one_sided_lower_percentile"] >= first["lower_percentile"]
    assert first["one_sided_upper_percentile"] <= first["upper_percentile"]
    assert first["one_sided_bounds_are_diagnostic_not_gates"] is True


def test_request_marks_absent_transport_unavailable_without_blocking_csm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(monkeypatch)
    authenticated = causal.authenticate_lfm2_exact4x_causal_diagnostic_request(request)
    availability = authenticated["transported_feature_availability"]
    assert availability["status"] == "unavailable"
    assert availability["transported_feature_oracle_available"] is False
    assert availability["producer_mismatch_classification_available"] is False
    assert availability["live_feature_consumer_oracle_available"] is True
    assert availability["csm_causal_replay_available"] is True
    assert availability["legacy_v1_lineage_supported"] is False
    assert availability["future_authority_signed_receipt_required"] is True
    assert availability["runner_artifact_rehash_required"] is True
    assert authenticated["canonical_subsets"] == list(
        causal.canonical_intervention_subsets(("C", "S", "M"))
    )


def test_request_rejects_relabelled_v3_trace_identity_or_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        causal,
        "authenticate_lfm2_exact4x_first_divergence_request",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        causal,
        "authenticate_lfm2_exact4x_first_divergence_batch",
        lambda value, **_kwargs: dict(value),
    )
    v3_request = _fake_v3_request()
    relabelled = list(_records())
    relabelled[0] = causal.Lfm2CausalRecordBinding(
        batch_index=0,
        trace_input_sha256=_sha("invented-input"),
        trace_record_sha256=relabelled[0].trace_record_sha256,
        split="fit",
        v3_batch_audit_sha256=relabelled[0].v3_batch_audit_sha256,
    )
    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError,
        match="authenticated v3 batch identities",
    ):
        causal.build_lfm2_exact4x_causal_diagnostic_request(
            first_divergence_request=v3_request,
            first_divergence_ledger=_fake_v3_ledger(v3_request),
            first_divergence_batch_results=_fake_v3_batch_results(),
            record_bindings=relabelled,
            implementation_identity_sha256=_sha("implementation"),
            checkpoint_hmac_key_id="causal-test-key",
            bootstrap_seed="causal-test-bootstrap",
            bootstrap_resample_count=64,
        )


def test_legacy_self_consistent_transport_lineage_is_unverifiable_and_rejected() -> (
    None
):
    specs = causal.default_lfm2_l0_csm_operator_specs()
    legacy_body = {
        "schema": causal.LFM2_EXACT4X_TRANSPORTED_FEATURE_LINEAGE_SCHEMA,
        "trace_cache_key": _sha("trace-cache"),
        "trace_selective_session_sha256": _sha("trace-session"),
        "basis_map_sha256": _sha("basis-map"),
        "operator_specs_sha256": canonical_sha256([spec.as_dict() for spec in specs]),
        "derivation_implementation_sha256": _sha("fabricated-code"),
        "feature_bindings": [],
        "cache_derived_only": True,
        "parent_forward_calls_during_causal_diagnostic": 0,
        "fitted_during_causal_diagnostic": False,
        "coordinates_invented_when_absent": False,
    }
    fabricated = {**legacy_body, "lineage_sha256": canonical_sha256(legacy_body)}
    for operation in (
        lambda: causal.authenticate_lfm2_transported_feature_lineage(
            fabricated,
            operator_specs=specs,
            trace_cache_key=_sha("trace-cache"),
            trace_selective_session_sha256=_sha("trace-session"),
            basis_map_sha256=_sha("basis-map"),
        ),
        lambda: causal.transported_feature_availability(
            fabricated,
            operator_specs=specs,
            trace_cache_key=_sha("trace-cache"),
            trace_selective_session_sha256=_sha("trace-session"),
            basis_map_sha256=_sha("basis-map"),
        ),
    ):
        with pytest.raises(
            causal.Lfm2Exact4xCausalDiagnosticError,
            match="unsupported and unverifiable",
        ):
            operation()
    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError,
        match="unsupported and unverifiable",
    ):
        causal.build_lfm2_transported_feature_lineage(
            trace_cache_key=_sha("trace-cache"),
            trace_selective_session_sha256=_sha("trace-session"),
            basis_map_sha256=_sha("basis-map"),
            operator_specs=specs,
            derivation_implementation_sha256=_sha("transport-code"),
            feature_bindings=(),
        )


def test_resealed_request_cannot_enable_fabricated_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(monkeypatch)
    fabricated_body = {
        "schema": causal.LFM2_EXACT4X_TRANSPORTED_FEATURE_LINEAGE_SCHEMA,
        "trace_cache_key": _sha("trace-cache"),
        "cache_derived_only": True,
    }
    fabricated = {
        **fabricated_body,
        "lineage_sha256": canonical_sha256(fabricated_body),
    }
    resealed = copy.deepcopy(request)
    resealed.pop("request_sha256")
    resealed["transported_feature_lineage"] = fabricated
    resealed["request_sha256"] = canonical_sha256(resealed)
    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError,
        match="unsupported and unverifiable",
    ):
        causal.authenticate_lfm2_exact4x_causal_diagnostic_request(resealed)


def test_transport_binding_type_does_not_enable_legacy_builder() -> None:
    binding = causal.Lfm2TransportedFeatureBinding(
        group_id="S",
        source_feature_tensor_name="layer.000.short_conv.context",
        target_consumer_module_path="model.layers.0.conv.out_proj",
        source_row_selection_sha256=_sha("shortconv-rows"),
        transport_tensor_sha256=_sha("shortconv-map"),
        source_width=1536,
        target_width=816,
    )
    assert binding.group_id == "S"
    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError,
        match="authority-signed receipt",
    ):
        causal.authenticate_lfm2_transported_feature_lineage(
            {"schema": causal.LFM2_EXACT4X_TRANSPORTED_FEATURE_LINEAGE_SCHEMA},
            operator_specs=causal.default_lfm2_l0_csm_operator_specs(),
            trace_cache_key=_sha("trace-cache"),
            trace_selective_session_sha256=_sha("trace-session"),
            basis_map_sha256=_sha("basis-map"),
        )


def test_batch_result_checkpoint_hmac_resume_and_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(monkeypatch)
    result0 = causal.build_lfm2_exact4x_causal_batch_result(
        request,
        batch_index=0,
        subset_metrics=_subset_metrics(offset=0.0),
        child_forward_calls=8,
    )
    result1 = causal.build_lfm2_exact4x_causal_batch_result(
        request,
        batch_index=1,
        subset_metrics=_subset_metrics(offset=0.25),
        child_forward_calls=8,
    )
    authority = causal.Lfm2CausalCheckpointAuthority(
        key_id="causal-test-key",
        secret=b"causal-checkpoint-test-secret-material!",
    )
    checkpoint0 = causal.build_lfm2_exact4x_causal_batch_checkpoint(
        request, result0, authority=authority
    )
    checkpoint1 = causal.build_lfm2_exact4x_causal_batch_checkpoint(
        request, result1, authority=authority
    )
    assert (
        causal.authenticate_lfm2_exact4x_causal_resume(
            [checkpoint0, checkpoint1], request=request, authority=authority
        )[1]["batch_index"]
        == 1
    )
    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError, match="canonical batch prefix"
    ):
        causal.authenticate_lfm2_exact4x_causal_resume(
            [checkpoint1], request=request, authority=authority
        )

    tampered = copy.deepcopy(checkpoint0)
    tampered["result"]["oracle_evidence"]["tampered"] = True
    with pytest.raises(causal.Lfm2Exact4xCausalDiagnosticError):
        causal.authenticate_lfm2_exact4x_causal_batch_checkpoint(
            tampered, request=request, authority=authority
        )

    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError, match="Plain batch results"
    ):
        causal.aggregate_lfm2_exact4x_causal_results(request, [result0, result1])
    ledger = causal.aggregate_lfm2_exact4x_causal_checkpoints(
        request,
        [checkpoint0, checkpoint1],
        authority=authority,
    )
    assert ledger["batch_count"] == 2
    assert ledger["numeric_acceptance_decision"] is None
    assert len(ledger["heldout_record_clustered_shapley_bootstraps"]) == 9
    assert ledger["operation_audit"]["child_forward_calls"] == 16
    assert ledger["checkpoint_chain_authenticated"] is True
    assert len(ledger["ledger_hmac_sha256"]) == 64
    assert (
        causal.authenticate_lfm2_exact4x_causal_ledger(
            ledger,
            request=request,
            checkpoints=[checkpoint0, checkpoint1],
            authority=authority,
        )["ledger_hmac_sha256"]
        == ledger["ledger_hmac_sha256"]
    )
    forged = copy.deepcopy(ledger)
    forged["split_summaries"]["heldout"]["numeric_acceptance_decision"] = "pass"
    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError, match="does not verify"
    ):
        causal.authenticate_lfm2_exact4x_causal_ledger(
            forged,
            request=request,
            checkpoints=[checkpoint0, checkpoint1],
            authority=authority,
        )

    # Even a party holding the runtime key cannot relabel derived statistics
    # while preserving the authenticated checkpoint inputs.  Verification
    # reconstructs the complete ledger rather than trusting a valid MAC alone.
    resigned = copy.deepcopy(ledger)
    resigned.pop("ledger_hmac_sha256")
    resigned.pop("ledger_sha256")
    resigned["split_summaries"]["heldout"]["numeric_acceptance_decision"] = "pass"
    resigned["ledger_sha256"] = canonical_sha256(resigned)
    resigned["ledger_hmac_sha256"] = authority.sign_ledger(resigned)
    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError, match="does not verify"
    ):
        causal.authenticate_lfm2_exact4x_causal_ledger(
            resigned,
            request=request,
            checkpoints=[checkpoint0, checkpoint1],
            authority=authority,
        )


def test_unavailable_transport_forbids_producer_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(monkeypatch)
    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError,
        match="authenticated transport lineage",
    ):
        causal.build_lfm2_exact4x_causal_batch_result(
            request,
            batch_index=0,
            subset_metrics=_subset_metrics(),
            child_forward_calls=8,
            oracle_evidence={
                "transported_feature_oracle_evaluated": True,
                "producer_mismatch_classification": "supported",
            },
        )

    with pytest.raises(
        causal.Lfm2Exact4xCausalDiagnosticError,
        match="noncanonical or unauthenticated",
    ):
        causal.build_lfm2_exact4x_causal_batch_result(
            request,
            batch_index=0,
            subset_metrics=_subset_metrics(),
            child_forward_calls=8,
            oracle_evidence={"transport_oracle_alias": "evaluated"},
        )


def test_core_reexports_single_reviewed_trace_scaled_ridge_solver() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([[1.0], [2.0], [3.0]])
    anchor = torch.zeros((2, 1))
    weights = torch.tensor([1.0, 1.0, 1.0])
    solution, receipt = causal.solve_lfm2_trace_scaled_weighted_anchored_ridge(
        features,
        targets,
        anchor,
        weights,
        output_dtype=torch.float32,
    )
    assert solution.shape == (2, 1)
    assert receipt["solver_path"] == "primal_weighted_normal_equation_cholesky"
    assert receipt["output_dtype_cast_count"] == 1
    assert receipt["optimizer_steps"] == 0
