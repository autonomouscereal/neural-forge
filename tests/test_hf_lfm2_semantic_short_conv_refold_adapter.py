from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge import hf_lfm2_cpcf_short_conv_fold as short_conv_fold
from neural_forge import hf_lfm2_semantic_short_conv_refold_adapter as adapter
from neural_forge import hf_lfm2_semantic_layer_stack_refold_adapter as stack_adapter
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import causal_depthwise_context
from neural_forge.hf_lfm2_semantic_short_conv_refold_adapter import (
    Lfm2SemanticShortConvRefoldError,
    Lfm2SemanticShortConvReplayEvidence,
    compile_install_lfm2_semantic_short_conv_postfold,
)


_CACHE = hashlib.sha256(b"complete-cache").hexdigest()
_BRIDGE = hashlib.sha256(b"semantic-bridge").hexdigest()
_SOURCE_MODEL = hashlib.sha256(b"direct-local-parent").hexdigest()
_LINEAGE = hashlib.sha256(b"short-conv-compiler-lineage").hexdigest()


def test_compiler_and_writer_share_one_reentrant_torch_state_lock() -> None:
    lock = short_conv_fold._SHORT_CONV_TORCH_STATE_LOCK
    assert adapter._SHORT_CONV_TORCH_STATE_LOCK is lock

    assert lock.acquire(blocking=False) is True
    try:
        assert lock.acquire(blocking=False) is True
        lock.release()
        acquired_from_other_thread: list[bool] = []

        def try_acquire() -> None:
            acquired = lock.acquire(blocking=False)
            acquired_from_other_thread.append(acquired)
            if acquired:
                lock.release()

        worker = threading.Thread(target=try_acquire)
        worker.start()
        worker.join()
        assert acquired_from_other_thread == [False]
    finally:
        lock.release()


def test_cpu_fhvr_context_serializes_process_global_torch_state() -> None:
    acquired_from_other_thread: list[bool] = []

    with adapter._fhvr_compute_device_context("cpu") as execution:
        assert execution == {
            "backend": "cpu",
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "device_index": None,
        }

        def try_acquire() -> None:
            acquired = adapter._SHORT_CONV_TORCH_STATE_LOCK.acquire(blocking=False)
            acquired_from_other_thread.append(acquired)
            if acquired:
                adapter._SHORT_CONV_TORCH_STATE_LOCK.release()

        worker = threading.Thread(target=try_acquire)
        worker.start()
        worker.join()

    assert acquired_from_other_thread == [False]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _ToyShortConv(torch.nn.Module):
    def __init__(self, *, bias: bool = True) -> None:
        super().__init__()
        self.in_proj = torch.nn.Linear(
            2,
            6,
            bias=bias,
            dtype=torch.bfloat16,
        )
        self.conv = torch.nn.Conv1d(
            2,
            2,
            kernel_size=2,
            groups=2,
            bias=bias,
            dtype=torch.bfloat16,
        )
        self.out_proj = torch.nn.Linear(
            2,
            2,
            bias=bias,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            self.in_proj.weight.zero_()
            self.in_proj.weight[:2].copy_(torch.eye(2, dtype=torch.bfloat16))
            self.in_proj.weight[4:].copy_(torch.eye(2, dtype=torch.bfloat16))
            if self.in_proj.bias is not None:
                self.in_proj.bias.zero_()
                self.in_proj.bias[2:4].fill_(1.0)
            self.conv.weight.copy_(
                torch.tensor(
                    [[[0.25, 0.75]], [[-0.125, 0.625]]],
                    dtype=torch.bfloat16,
                )
            )
            if self.conv.bias is not None:
                self.conv.bias.zero_()
            self.out_proj.weight.zero_()
            if self.out_proj.bias is not None:
                self.out_proj.bias.zero_()


class _ToyLayer(torch.nn.Module):
    def __init__(self, *, bias: bool = True) -> None:
        super().__init__()
        self.conv = _ToyShortConv(bias=bias)


class _ToyBackbone(torch.nn.Module):
    def __init__(self, *, bias: bool = True) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer(bias=bias)])


class _ToyModel(torch.nn.Module):
    def __init__(self, *, bias: bool = True) -> None:
        super().__init__()
        self.model = _ToyBackbone(bias=bias)
        self.forward_calls = 0

    def forward(self, *_args: object, **_kwargs: object) -> object:
        self.forward_calls += 1
        raise AssertionError("The semantic adapter must never execute the model.")


def _fingerprint(model: _ToyModel) -> SimpleNamespace:
    body = {
        name: adapter._tensor_sha256(parameter)
        for name, parameter in model.named_parameters()
    }
    return SimpleNamespace(identity_sha256=canonical_sha256(body))


@pytest.fixture(autouse=True)
def _patch_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "fingerprint_lfm2_parent", _fingerprint)
    monkeypatch.setattr(stack_adapter, "fingerprint_lfm2_parent", _fingerprint)


def _operator_context(model: _ToyModel, inputs: torch.Tensor) -> torch.Tensor:
    operator = model.model.layers[0].conv
    projected = (
        inputs.to(torch.float64) @ operator.in_proj.weight.detach().to(torch.float64).T
    )
    if operator.in_proj.bias is not None:
        projected = projected + operator.in_proj.bias.detach().to(torch.float64)
    b_value, c_value, x_value = projected.chunk(3, dim=-1)
    gated = (b_value * x_value).transpose(1, 2)
    history = torch.nn.functional.pad(gated, (1, 0))
    filtered = torch.nn.functional.conv1d(
        history,
        operator.conv.weight.detach().to(torch.float64),
        bias=(
            operator.conv.bias.detach().to(torch.float64)
            if operator.conv.bias is not None
            else None
        ),
        groups=2,
    )
    return c_value * filtered.transpose(1, 2)


def _evidence_bank(
    model: _ToyModel,
    *,
    validation_matches_incumbent: bool = False,
) -> tuple[Lfm2SemanticShortConvReplayEvidence, ...]:
    desired_weight = torch.tensor(
        [[1.0, 0.25], [-0.125, 0.75]],
        dtype=torch.float64,
    )
    desired_bias = torch.tensor([0.125, -0.25], dtype=torch.float64)
    captured = _fingerprint(model).identity_sha256
    result = []
    for batch_index in range(8):
        inputs = torch.tensor(
            [
                [
                    [0.25 + 0.05 * batch_index, -0.50],
                    [0.75, 0.125 + 0.025 * batch_index],
                    [-0.375, 0.50 + 0.05 * batch_index],
                    [1.00 - 0.025 * batch_index, -0.25],
                ]
            ],
            dtype=torch.bfloat16,
        )
        pre = torch.tensor(
            [
                [
                    [0.05, -0.10],
                    [0.075, -0.05],
                    [0.10, 0.025],
                    [0.125, 0.05 + 0.01 * batch_index],
                ]
            ],
            dtype=torch.float32,
        )
        context = _operator_context(model, inputs)[:, -1:, :]
        if validation_matches_incumbent and batch_index >= 6:
            branch = torch.zeros_like(context)
        else:
            branch = context @ desired_weight.T + desired_bias
        parent_post = pre[:, -1:, :].to(torch.float64) + branch
        generator = torch.Generator().manual_seed(1000 + batch_index)
        parent_b = torch.randn((1, 3, 4), generator=generator).to(torch.bfloat16)
        parent_c = (0.75 + 0.1 * torch.randn((1, 3, 4), generator=generator)).to(
            torch.bfloat16
        )
        parent_x = torch.randn((1, 3, 4), generator=generator).to(torch.bfloat16)
        result.append(
            Lfm2SemanticShortConvReplayEvidence(
                batch_index=batch_index,
                phase=0,
                layer_index=0,
                alignment_sha256=_digest(f"alignment-{batch_index}"),
                cache_key=_CACHE,
                bridge_artifact_sha256=_BRIDGE,
                source_capability_sha256=_digest(f"capability-{batch_index}"),
                source_history_sha256=_digest(f"history-{batch_index}"),
                compiler_lineage_sha256=_LINEAGE,
                captured_child_model_state_sha256=captured,
                target_to_parent_span_owner_positions=(0, 1, 1, 2),
                endpoint_target_positions=(3,),
                actual_child_operator_input_history=inputs,
                actual_child_pre_operator_residual_history=pre,
                parent_b_history=parent_b,
                parent_c_history=parent_c,
                parent_x_history=parent_x,
                parent_post_operator_residual_target_space=parent_post.to(
                    torch.float32
                ),
                endpoint_row_sqrt_weights=torch.tensor(
                    [[1.0 + 0.1 * batch_index]],
                    dtype=torch.float64,
                ),
            )
        )
    return tuple(result)


def _source_fir() -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.tensor(
        [
            [[0.25, 0.75]],
            [[-0.25, 0.50]],
            [[0.625, -0.125]],
            [[0.375, 0.25]],
        ],
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([0.0, 0.1, -0.1, 0.05], dtype=torch.bfloat16)
    return weight, bias


def _fhvr_source_and_evidence(
    model: _ToyModel,
) -> tuple[
    adapter.Lfm2SemanticShortConvFhvrSource,
    tuple[Lfm2SemanticShortConvReplayEvidence, ...],
]:
    generator = torch.Generator().manual_seed(20260723)
    source_hidden = 4
    source_in_weight = (
        torch.randn(
            (3 * source_hidden, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.2
    ).to(torch.bfloat16)
    source_in_bias = (
        torch.randn(
            (3 * source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.03
    ).to(torch.bfloat16)
    source_out_weight = (
        torch.randn(
            (source_hidden, source_hidden),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.2
    ).to(torch.bfloat16)
    source_out_bias = (
        torch.randn(
            (source_hidden,),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.01
    ).to(torch.bfloat16)
    residual_chart = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.25],
            [-0.125, 0.375],
        ],
        dtype=torch.float64,
    )
    source_fir_weight, source_fir_bias = _source_fir()
    source = adapter.Lfm2SemanticShortConvFhvrSource(
        layer_index=0,
        cache_key=_CACHE,
        bridge_artifact_sha256=_BRIDGE,
        source_model_identity_sha256=_SOURCE_MODEL,
        parent_session_open_receipt_sha256=_digest("parent-session"),
        parent_identity_reconstruction_sha256=_digest("parent-identity"),
        residual_chart_sha256=_digest("residual-chart"),
        norm_epsilon=1e-5,
        source_operator_norm_weight=torch.ones((source_hidden,), dtype=torch.bfloat16),
        source_in_proj_weight=source_in_weight,
        source_in_proj_bias=source_in_bias,
        source_fir_weight=source_fir_weight,
        source_fir_bias=source_fir_bias,
        source_out_proj_weight=source_out_weight,
        source_out_proj_bias=source_out_bias,
        residual_output_chart=residual_chart,
    )
    evidence = []
    for item in _evidence_bank(model):
        parent_input = torch.randn(
            tuple(item.parent_b_history.shape),
            generator=generator,
            dtype=torch.float64,
        ).to(torch.bfloat16)
        projected = parent_input.to(torch.float64) @ source_in_weight.to(
            torch.float64
        ).T + source_in_bias.to(torch.float64)
        parent_b, parent_c, parent_x = (
            value.to(torch.bfloat16) for value in projected.chunk(3, dim=-1)
        )
        parent_context = causal_depthwise_context(
            parent_b,
            parent_c,
            parent_x,
            source_fir_weight[:, 0, :],
            source_fir_bias,
        )
        parent_branch = (
            parent_context.to(torch.float64) @ source_out_weight.to(torch.float64).T
            + source_out_bias.to(torch.float64)
        ) @ residual_chart
        endpoint = item.endpoint_target_positions[0]
        parent_post = (
            item.actual_child_pre_operator_residual_history[
                :, endpoint : endpoint + 1
            ].to(torch.float64)
            + parent_branch[:, -1:, :]
        ).to(torch.float32)
        provenance = canonical_sha256(
            {
                "schema": "test-full-parent-operator-input-v1",
                "batch_index": item.batch_index,
                "history_sha256": adapter._tensor_sha256(parent_input),
                "direct_source_sha256": source.source_sha256,
            }
        )
        evidence.append(
            replace(
                item,
                parent_b_history=parent_b,
                parent_c_history=parent_c,
                parent_x_history=parent_x,
                parent_post_operator_residual_target_space=parent_post,
                parent_operator_input_history=parent_input,
                parent_operator_input_provenance_sha256=provenance,
            )
        )
    return source, tuple(evidence)


def _compile(
    model: _ToyModel,
    evidence: tuple[Lfm2SemanticShortConvReplayEvidence, ...],
    *,
    fold_compute_device: str | None = None,
    body_recipe: str = "fit_guided",
) -> dict[str, object]:
    weight, bias = _source_fir()
    return dict(
        compile_install_lfm2_semantic_short_conv_postfold(
            target_model=model,
            layer_index=0,
            evidence=evidence,
            source_fir_weight=weight,
            source_fir_bias=bias,
            cache_key=_CACHE,
            bridge_artifact_sha256=_BRIDGE,
            source_model_identity_sha256=_SOURCE_MODEL,
            body_recipe=body_recipe,
            **(
                {"fold_compute_device": fold_compute_device}
                if fold_compute_device is not None
                else {}
            ),
        )
    )


def _cpu_writer_report(status: str) -> dict[str, object]:
    execution_body = {
        "schema": "neural-forge-lfm2-short-conv-writer-cpu-execution-v1",
        "backend": "cpu",
        "requested_device": "cpu",
        "requested_device_was_explicit": True,
        "resolved_device": "cpu",
        "device_index": None,
        "compute_dtype": "float64",
        "closed_form_linear_solve_count": 1,
        "canonical_input_authentication_device": "cpu",
        "canonical_result_materialization_device": "cpu",
        "cuda_tensor_stage_count": 0,
        "parent_model_forward_calls": 0,
        "child_model_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
    }
    return {
        "status": status,
        "operation_audit": dict(adapter._ZERO_OPERATIONS),
        "compute_execution": {
            **execution_body,
            "execution_sha256": canonical_sha256(execution_body),
        },
    }


def test_fhvr_full_history_construction_installs_one_dense_child_state() -> None:
    model = _ToyModel()
    source, evidence = _fhvr_source_and_evidence(model)
    source_fir_weight, source_fir_bias = _source_fir()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    receipt = dict(
        compile_install_lfm2_semantic_short_conv_postfold(
            target_model=model,
            layer_index=0,
            evidence=evidence,
            source_fir_weight=source_fir_weight,
            source_fir_bias=source_fir_bias,
            cache_key=_CACHE,
            bridge_artifact_sha256=_BRIDGE,
            source_model_identity_sha256=_SOURCE_MODEL,
            short_conv_construction=(
                adapter.LFM2_SEMANTIC_SHORT_CONV_FHVR_FULL_COVERAGE_CONSTRUCTION
            ),
            fhvr_source=source,
            fold_compute_device="cpu",
        )
    )

    seal = receipt.pop("result_sha256")
    assert canonical_sha256(receipt) == seal
    assert receipt["installed"] is True
    assert receipt["selected_candidate"] == ("full-history-fhvr-full-coverage-hybrid")
    assert (
        receipt["short_conv_construction"]["construction_sha256"]
        == (receipt["short_conv_construction_sha256"])
    )
    assert receipt["short_conv_construction_sha256"] == (
        adapter.resolve_lfm2_semantic_short_conv_construction(
            adapter.LFM2_SEMANTIC_SHORT_CONV_FHVR_FULL_COVERAGE_CONSTRUCTION
        ).construction_sha256
    )
    assert receipt["fhvr_source"]["source_sha256"] == source.source_sha256
    assert receipt["fhvr_source_sha256"] == source.source_sha256
    writer_control = receipt["fhvr_candidate_report"]["writer_control_report"]
    assert writer_control["candidate_id"] == (
        "semantic-full-source-structural-control-writer"
    )
    assert writer_control["method_provenance"][
        "numeric_metrics_used_to_choose_writer_control"
    ] is False
    assert receipt["full_history_fhvr_full_coverage_candidate_competed"] is True
    assert receipt["complete_parent_bcx_history_consumed"] is True
    assert receipt["fit_metrics_role"] == "diagnostic-only"
    assert receipt["fit_metrics_used_for_pre_review_candidate_selection"] is False
    assert receipt["numeric_metrics_have_final_intelligence_authority"] is False
    assert receipt["behavioral_intelligence_review_outside_short_conv_compiler"] is True
    assert receipt["parameter_count_changed"] is False
    assert receipt["parameter_element_count"] == parameter_count
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["backward_calls"] == 0
    assert receipt["operation_audit"]["optimizer_steps"] == 0
    assert model.forward_calls == 0


def test_fhvr_replays_parent_fir_before_owner_expansion() -> None:
    model = _ToyModel()
    source, evidence = _fhvr_source_and_evidence(model)
    selected = evidence[:2]
    fit_cases = adapter._fhvr_fit_cases(selected)

    contexts, receipt = adapter._fhvr_parent_contexts(
        selected,
        fit_cases,
        source=source,
    )

    assert receipt["causal_fir_replayed_before_target_owner_expansion"] is True
    assert receipt["complete_donor_causal_history_consumed"] is True
    assert receipt["valid_mask_admits_each_donor_semantic_boundary_once"] is True
    receipt_body = dict(receipt)
    seal = receipt_body.pop("result_sha256")
    assert canonical_sha256(receipt_body) == seal
    for item, case in zip(selected, fit_cases, strict=True):
        owners = torch.tensor(
            item.target_to_parent_span_owner_positions,
            dtype=torch.long,
        )
        donor_context = causal_depthwise_context(
            item.parent_b_history,
            item.parent_c_history,
            item.parent_x_history,
            source.source_fir_weight,
            source.source_fir_bias,
        )
        expected = donor_context.index_select(1, owners)
        expanded_then_filtered = causal_depthwise_context(
            item.parent_b_history.index_select(1, owners),
            item.parent_c_history.index_select(1, owners),
            item.parent_x_history.index_select(1, owners),
            source.source_fir_weight,
            source.source_fir_bias,
        )
        assert torch.equal(contexts[case.case_id], expected)
        assert not torch.allclose(contexts[case.case_id], expanded_then_filtered)
        assert case.valid_prefix_mask.tolist() == [[True, False, True, True]]
        assert int(case.valid_prefix_mask.sum().item()) == int(
            item.parent_b_history.shape[1]
        )


def test_fhvr_source_tensor_tamper_fails_before_child_mutation() -> None:
    model = _ToyModel()
    source, evidence = _fhvr_source_and_evidence(model)
    source_fir_weight, source_fir_bias = _source_fir()
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    source.source_in_proj_weight[0, 0].add_(1.0)

    with pytest.raises(
        Lfm2SemanticShortConvRefoldError,
        match="authenticated direct source",
    ):
        compile_install_lfm2_semantic_short_conv_postfold(
            target_model=model,
            layer_index=0,
            evidence=evidence,
            source_fir_weight=source_fir_weight,
            source_fir_bias=source_fir_bias,
            cache_key=_CACHE,
            bridge_artifact_sha256=_BRIDGE,
            source_model_identity_sha256=_SOURCE_MODEL,
            short_conv_construction=(
                adapter.LFM2_SEMANTIC_SHORT_CONV_FHVR_FULL_COVERAGE_CONSTRUCTION
            ),
            fhvr_source=source,
            fold_compute_device="cpu",
        )

    assert all(
        torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )
    assert model.forward_calls == 0


def test_writer_only_competes_with_full_source_structural_and_installs() -> None:
    model = _ToyModel()
    evidence = _evidence_bank(model)
    operator = model.model.layers[0].conv
    before_in = operator.in_proj.weight.detach().clone()
    before_conv = operator.conv.weight.detach().clone()
    before_out = operator.out_proj.weight.detach().clone()
    count = sum(parameter.numel() for parameter in model.parameters())

    receipt = _compile(model, evidence)

    seal = receipt.pop("result_sha256")
    assert canonical_sha256(receipt) == seal
    assert receipt["installed"] is True
    assert receipt["status"] == "installed-frozen-semantic-short-conv-candidate"
    assert receipt["install_count"] == 1
    assert receipt["parameter_count_changed"] is False
    assert receipt["parameter_element_count"] == count
    assert receipt["full_source_structural_candidate_competed"] is True
    writer_proposal = receipt["writer_fit_proposal"]
    assert writer_proposal["candidate_frontier_count"] == 5
    assert writer_proposal["structurally_eligible_candidate_count"] == sum(
        int(record["structurally_eligible"] is True)
        for record in writer_proposal["candidate_frontier"]
    )
    assert all(
        record["fit_thresholds_used_for_eligibility"] is False
        and record["fit_nonregression_used_for_eligibility"] is False
        and record["relative_magnitude_threshold_used_for_eligibility"] is False
        for record in writer_proposal["candidate_frontier"]
    )
    reports = receipt["candidate_reports_before_validation"]
    assert "full-source-structural" in reports
    assert reports["full-source-structural"]["provenance"]["full_source_coverage"]
    assert (
        reports["full-source-structural"]["provenance"]["source_channels_dropped"] == 0
    )
    assert receipt["inserted_target_rows_parent_exact"] is False
    assert receipt["inserted_target_rows_used_for_endpoint_fit"] is False
    assert receipt["inserted_target_row_count"] == len(evidence)
    assert receipt["parent_exact_endpoint_row_count"] == len(evidence)
    assert receipt["candidate_frozen_before_validation"] is True
    assert receipt["validation_triggered_refit"] is False
    assert receipt["candidate_eligibility_basis"] == (
        "finite_dense_shape_storage_coverage_and_nonincumbent_effect"
    )
    assert receipt["structural_safety_is_only_candidate_veto"] is True
    assert receipt["fit_metrics_used_for_candidate_ordering"] is True
    assert receipt["fit_metrics_used_for_structural_eligibility"] is False
    assert receipt["fit_check_metrics_diagnostic_only"] is True
    assert receipt["selection_metrics_diagnostic_only"] is True
    assert receipt["validation_metrics_diagnostic_only"] is True
    assert receipt["relative_magnitude_metrics_diagnostic_only"] is True
    assert receipt["numeric_metrics_have_final_intelligence_authority"] is False
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["model_forward_calls"] == 0
    assert model.forward_calls == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == count
    assert not torch.equal(operator.out_proj.weight, before_out)
    if receipt["selected_candidate"] == "incumbent-producer-output-writer":
        assert torch.equal(operator.in_proj.weight, before_in)
        assert torch.equal(operator.conv.weight, before_conv)
        assert all("out_proj" in name for name in receipt["changed_state_keys"])
        assert tuple(
            reports["incumbent-producer-output-writer"]["fit_ordering_key"]
        ) <= tuple(reports["full-source-structural"]["fit_ordering_key"])
        assert (
            reports["incumbent-producer-output-writer"]["candidate_ranking_split"]
            == "fit"
        )


def test_five_recipes_materialize_the_declared_effectful_nonblank_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deterministic_tiebreakers_only(
        _metrics: object,
        *tiebreakers: object,
    ) -> tuple[object, ...]:
        return tuple(tiebreakers)

    monkeypatch.setattr(
        adapter,
        "_fit_metric_ordering_key",
        deterministic_tiebreakers_only,
    )
    receipts: dict[str, dict[str, object]] = {}
    for recipe_id in (
        "fit_guided",
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
        "structural_sequence_residual_mlp_writer",
    ):
        model = _ToyModel()
        receipts[recipe_id] = _compile(
            model,
            _evidence_bank(model),
            body_recipe=recipe_id,
        )

    assert receipts["fit_guided"]["selected_candidate"] == (
        "incumbent-producer-output-writer"
    )
    assert receipts["fit_guided"]["writer_fit_proposal"]["selected_alpha"] == 0.125
    assert receipts["full_strength_structural"]["selected_candidate"] == (
        "full-source-structural"
    )
    assert receipts["residual_writer"]["selected_candidate"] == (
        "incumbent-producer-output-writer"
    )
    assert receipts["residual_writer"]["writer_fit_proposal"]["selected_alpha"] == 1.0
    assert receipts["source_producer_residual_writer"]["selected_candidate"] == (
        "full-source-structural"
    )
    assert (
        receipts["source_producer_residual_writer"]["model_state_after_sha256"]
        == receipts["full_strength_structural"]["model_state_after_sha256"]
    )
    assert (
        receipts["structural_sequence_residual_mlp_writer"]["selected_candidate"]
        == "full-source-structural"
    )
    assert (
        receipts["structural_sequence_residual_mlp_writer"]["model_state_after_sha256"]
        == receipts["full_strength_structural"]["model_state_after_sha256"]
    )
    assert (
        len({receipt["model_state_after_sha256"] for receipt in receipts.values()}) == 3
    )
    assert all(
        receipt["installed"] is True
        and receipt["model_state_before_sha256"] != receipt["model_state_after_sha256"]
        and receipt["numeric_metrics_have_final_intelligence_authority"] is False
        for receipt in receipts.values()
    )


def test_blank_residual_writer_is_distinct_complete_full_source_plant() -> None:
    receipts: dict[str, dict[str, object]] = {}
    for recipe_id in (
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
        "structural_sequence_residual_mlp_writer",
        "nonlinear_qdeim_coupled_attention_short_conv_residual_writer",
    ):
        evidence_source = _ToyModel()
        source_evidence = _evidence_bank(evidence_source)
        model = _ToyModel()
        with torch.no_grad():
            for parameter in model.model.layers[0].conv.parameters():
                parameter.zero_()
        captured = _fingerprint(model).identity_sha256
        evidence = tuple(
            replace(item, captured_child_model_state_sha256=captured)
            for item in source_evidence
        )
        receipt = _compile(model, evidence, body_recipe=recipe_id)
        receipts[recipe_id] = receipt
        assert receipt["installed"] is True
        assert receipt["cold_start_blank_noop_forbidden"] is True
        assert (
            receipt["cold_start_selected_state_uses_full_source_structural_producers"]
            is True
        )
        assert all(
            int(torch.count_nonzero(parameter).item()) > 0
            for parameter in model.model.layers[0].conv.parameters()
        )

    assert receipts["full_strength_structural"]["selected_candidate"] == (
        "full-source-structural"
    )
    assert receipts["residual_writer"]["selected_candidate"] == (
        "full-source-producer-output-writer"
    )
    assert receipts[
        "nonlinear_qdeim_coupled_attention_short_conv_residual_writer"
    ]["selected_candidate"] == "full-source-producer-output-writer"
    assert receipts["source_producer_residual_writer"]["selected_candidate"] == (
        "full-source-structural"
    )
    assert (
        receipts["residual_writer"][
            "cold_start_residual_writer_component_equivalent_to_structural"
        ]
        is False
    )
    assert (
        receipts["full_strength_structural"]["model_state_after_sha256"]
        != receipts["residual_writer"]["model_state_after_sha256"]
    )
    assert (
        receipts[
            "nonlinear_qdeim_coupled_attention_short_conv_residual_writer"
        ]["model_state_after_sha256"]
        == receipts["residual_writer"]["model_state_after_sha256"]
    )
    assert (
        receipts["full_strength_structural"]["model_state_after_sha256"]
        == receipts["source_producer_residual_writer"]["model_state_after_sha256"]
    )
    assert (
        receipts["full_strength_structural"]["model_state_after_sha256"]
        == receipts["structural_sequence_residual_mlp_writer"][
            "model_state_after_sha256"
        ]
    )


def test_cpu_fold_device_routes_writer_to_cpu_with_sealed_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A CUDA-capable host exposed the old bug: the outer compiler said "cpu"
    # while the omitted private writer keyword auto-selected cuda:0.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    implicit_model = _ToyModel()
    explicit_model = _ToyModel()

    implicit = _compile(implicit_model, _evidence_bank(implicit_model))
    explicit = _compile(
        explicit_model,
        _evidence_bank(explicit_model),
        fold_compute_device="cpu",
    )

    assert implicit == explicit
    assert explicit["fold_compute_device"] == "cpu"
    assert "structural_cluster_compute_execution" not in explicit
    assert explicit["requested_cpu_short_conv_writer_route_proved"] is True
    execution = explicit["writer_compute_execution"]
    body = dict(execution)
    seal = body.pop("execution_sha256")
    assert canonical_sha256(body) == seal
    assert explicit["writer_compute_execution_sha256"] == seal
    assert execution["backend"] == "cpu"
    assert execution["requested_device"] == "cpu"
    assert execution["requested_device_was_explicit"] is True
    assert execution["resolved_device"] == "cpu"
    assert execution["compute_dtype"] == "float64"
    assert execution["cuda_tensor_stage_count"] == 0
    assert execution["training_steps"] == 0
    assert all(
        torch.equal(value, explicit_model.state_dict()[name])
        for name, value in implicit_model.state_dict().items()
    )


def test_cpu_writer_execution_proof_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyModel()
    before = _fingerprint(model).identity_sha256
    compile_writer = adapter._compile_writer_candidate

    def corrupt_execution(*args: object, **kwargs: object):
        candidate, report = compile_writer(*args, **kwargs)
        corrupted_report = dict(report)
        corrupted_execution = dict(corrupted_report["compute_execution"])
        corrupted_execution["resolved_device"] = "cuda:0"
        corrupted_report["compute_execution"] = corrupted_execution
        return candidate, corrupted_report

    monkeypatch.setattr(adapter, "_compile_writer_candidate", corrupt_execution)

    with pytest.raises(
        Lfm2SemanticShortConvRefoldError,
        match="CPU short-conv writer execution proof is invalid",
    ):
        _compile(model, _evidence_bank(model), fold_compute_device="cpu")
    assert _fingerprint(model).identity_sha256 == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_indexed_cuda_routes_structural_cluster_and_writer_with_sealed_proofs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    first_model = _ToyModel()
    second_model = _ToyModel()

    first = _compile(
        first_model,
        _evidence_bank(first_model),
        fold_compute_device="cuda:0",
    )
    second = _compile(
        second_model,
        _evidence_bank(second_model),
        fold_compute_device="cuda:0",
    )

    assert first == second
    assert first["fold_compute_device"] == "cuda:0"
    assert first["requested_cuda_short_conv_routes_proved"] is True
    for key in (
        "structural_cluster_compute_execution",
        "writer_compute_execution",
    ):
        execution = first[key]
        body = dict(execution)
        seal = body.pop("execution_sha256")
        assert canonical_sha256(body) == seal
        assert execution["backend"] == "cuda"
        assert execution["requested_device"] == "cuda:0"
        assert execution["compute_dtype"] == "float64"
        assert execution["training_steps"] == 0
    assert all(
        torch.equal(value, second_model.state_dict()[name])
        for name, value in first_model.state_dict().items()
    )


def test_blank_multi_weight_plant_emits_live_canonical_layer_stack_diff() -> None:
    source_model = _ToyModel()
    source_evidence = _evidence_bank(source_model)
    model = _ToyModel()
    operator = model.model.layers[0].conv
    with torch.no_grad():
        for parameter in operator.parameters():
            parameter.zero_()
    assert all(
        int(torch.count_nonzero(parameter).item()) == 0
        for parameter in operator.parameters()
    )
    captured = _fingerprint(model).identity_sha256
    evidence = tuple(
        replace(item, captured_child_model_state_sha256=captured)
        for item in source_evidence
    )
    source_weight, source_bias = _source_fir()
    before = stack_adapter._state_tensor_hashes(model)

    receipt = stack_adapter._install_short_conv_operator(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        source_fir_weight=source_weight,
        source_fir_bias=source_bias,
        cache=SimpleNamespace(
            cache_key=_CACHE,
            parent_fingerprint=_SOURCE_MODEL,
        ),
        bridge=SimpleNamespace(artifact_sha256=_BRIDGE),
        installer=compile_install_lfm2_semantic_short_conv_postfold,
        parameter_count=stack_adapter._parameter_count(model),
    )

    live_changed = list(
        stack_adapter._changed_state_keys(
            before,
            stack_adapter._state_tensor_hashes(model),
        )
    )
    assert receipt["installed"] is True
    assert receipt["selected_candidate"] == "full-source-structural"
    assert receipt["exact_zero_incumbent_at_entry"] is True
    assert receipt["cold_start_compulsory_full_source_structural_plant"] is True
    assert receipt["cold_start_heldout_metrics_diagnostic_only"] is True
    assert receipt["cold_start_blank_noop_forbidden"] is True
    assert receipt["cold_start_all_required_operator_weights_finite_nonzero"] is True
    assert receipt["cold_start_optional_biases_finite"] is True
    assert receipt["cold_start_all_required_operator_axes_covered"] is True
    coverage = dict(receipt["cold_start_structural_axis_coverage"])
    coverage_sha256 = coverage.pop("result_sha256")
    assert canonical_sha256(coverage) == coverage_sha256
    assert coverage_sha256 == receipt["cold_start_structural_axis_coverage_sha256"]
    assert coverage["every_output_row_and_input_or_lag_axis_covered"] is True
    assert coverage["blank_or_partially_blank_required_weight_count"] == 0
    assert all(
        role["structurally_nonzero"] is True
        and role["all_output_structural_units_covered"] is True
        and role["all_input_structural_units_covered"] is True
        for role in coverage["role_receipts"].values()
    )
    assert receipt["cold_start_structural_axis_repair_applied"] is False
    assert receipt["operation_audit"]["training_steps"] == 0
    assert (
        receipt["candidate_reports_before_validation"]["full-source-structural"][
            "provenance"
        ]["full_source_coverage"]
        is True
    )
    assert len(live_changed) > 1
    assert receipt["changed_state_keys"] == live_changed
    assert live_changed == sorted(live_changed)
    assert all(
        int(torch.count_nonzero(parameter).item()) > 0
        for parameter in operator.parameters()
    )


def test_blank_biasless_topology_compulsorily_installs_required_weights() -> None:
    source_model = _ToyModel(bias=False)
    source_evidence = _evidence_bank(source_model)
    model = _ToyModel(bias=False)
    operator = model.model.layers[0].conv
    with torch.no_grad():
        for parameter in operator.parameters():
            parameter.zero_()
    captured = _fingerprint(model).identity_sha256
    evidence = tuple(
        replace(item, captured_child_model_state_sha256=captured)
        for item in source_evidence
    )

    receipt = _compile(model, evidence)

    assert receipt["selected_candidate"] == "full-source-structural"
    assert receipt["installed"] is True
    assert receipt["cold_start_compulsory_full_source_structural_plant"] is True
    assert receipt["cold_start_all_required_operator_weights_finite_nonzero"] is True
    assert receipt["cold_start_optional_biases_finite"] is True
    assert receipt["cold_start_all_required_operator_axes_covered"] is True
    assert set(receipt["changed_parameter_roles"]) == {
        "conv.weight",
        "in_proj.weight",
        "out_proj.weight",
    }
    assert all(
        module.bias is None
        for module in (operator.in_proj, operator.conv, operator.out_proj)
    )
    assert all(
        bool(torch.isfinite(parameter).all().item())
        and int(torch.count_nonzero(parameter).item()) > 0
        for parameter in operator.parameters()
    )


def test_cold_start_validator_allows_finite_zero_optional_biases() -> None:
    candidate = adapter._Candidate(
        name="full-source-structural",
        state={
            "in_proj.weight": torch.ones((6, 2), dtype=torch.bfloat16),
            "in_proj.bias": torch.zeros(6, dtype=torch.bfloat16),
            "conv.weight": torch.ones((2, 1, 2), dtype=torch.bfloat16),
            "conv.bias": torch.zeros(2, dtype=torch.bfloat16),
            "out_proj.weight": torch.eye(2, dtype=torch.bfloat16),
            "out_proj.bias": torch.zeros(2, dtype=torch.bfloat16),
        },
        provenance={
            "full_source_coverage": True,
            "source_channels_dropped": 0,
            "structural_compiler_report": {
                "execution": {
                    "training_steps": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                }
            },
        },
    )

    coverage = adapter._validate_cold_start_structural_plant(candidate)

    assert coverage["every_output_row_and_input_or_lag_axis_covered"] is True
    assert coverage["blank_or_partially_blank_required_weight_count"] == 0


def test_nonzero_but_axis_uncovered_cold_start_fails_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model = _ToyModel()
    source_evidence = _evidence_bank(source_model)
    model = _ToyModel()
    operator = model.model.layers[0].conv
    with torch.no_grad():
        for parameter in operator.parameters():
            parameter.zero_()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    captured = _fingerprint(model).identity_sha256
    evidence = tuple(
        replace(item, captured_child_model_state_sha256=captured)
        for item in source_evidence
    )
    bad_in_proj = torch.ones_like(operator.in_proj.weight)
    bad_in_proj[0].zero_()
    candidate = adapter._Candidate(
        name="full-source-structural",
        state={
            "in_proj.weight": bad_in_proj,
            "in_proj.bias": torch.zeros_like(operator.in_proj.bias),
            "conv.weight": torch.ones_like(operator.conv.weight),
            "conv.bias": torch.zeros_like(operator.conv.bias),
            "out_proj.weight": torch.eye(2, dtype=torch.bfloat16),
            "out_proj.bias": torch.zeros_like(operator.out_proj.bias),
        },
        provenance={
            "full_source_coverage": True,
            "source_channels_dropped": 0,
            "structural_compiler_report": {
                "execution": {
                    "training_steps": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                }
            },
        },
    )
    monkeypatch.setattr(
        adapter,
        "_compile_structural_candidate",
        lambda **_kwargs: candidate,
    )
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())
    gradients = tuple(
        parameter.grad.detach().clone() for parameter in model.parameters()
    )

    with pytest.raises(
        Lfm2SemanticShortConvRefoldError,
        match="leaves a required output row, input axis, or FIR lag axis disconnected",
    ):
        _compile(model, evidence)

    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids
    assert all(
        torch.equal(parameter.grad, gradient)
        for parameter, gradient in zip(model.parameters(), gradients)
    )


def test_blank_incumbent_cannot_win_even_when_validation_prefers_blank_noop() -> None:
    evidence_source = _ToyModel()
    source_evidence = _evidence_bank(
        evidence_source,
        validation_matches_incumbent=True,
    )
    model = _ToyModel()
    operator = model.model.layers[0].conv
    with torch.no_grad():
        for parameter in operator.parameters():
            parameter.zero_()
    captured = _fingerprint(model).identity_sha256
    evidence = tuple(
        replace(item, captured_child_model_state_sha256=captured)
        for item in source_evidence
    )

    receipt = _compile(model, evidence)

    assert receipt["selected_candidate"] == "full-source-structural"
    assert receipt["installed"] is True
    assert receipt["validation_rejected_candidate"] is None
    assert receipt["cold_start_heldout_metrics_diagnostic_only"] is True
    assert receipt["cold_start_fit_structural_candidate_forced_before_heldout"] is True
    assert receipt["cold_start_fit_check_used_for_publication_gate"] is False
    assert receipt["cold_start_selection_used_for_publication_gate"] is False
    assert receipt["cold_start_validation_used_for_publication_gate"] is False
    assert receipt["selection_used_for_candidate_choice"] is False
    assert receipt["validation_used_for_candidate_choice"] is False
    assert receipt["validation_used_only_for_acceptance_or_incumbent_fallback"] is False
    assert receipt["incumbent_noop_candidate_competed"] is False
    assert receipt["incumbent_noop_fallback_used"] is False
    assert all(
        int(torch.count_nonzero(parameter).item()) > 0
        for parameter in operator.parameters()
    )


def test_four_partitions_are_trajectory_isolated_and_unit_balanced() -> None:
    model = _ToyModel()
    receipt = _compile(model, _evidence_bank(model))

    partitions = receipt["partition_batch_indices"]
    observed = []
    for name in ("fit", "fit_check", "selection", "validation"):
        assert partitions[name]
        observed.extend(partitions[name])
        groups = receipt["trajectory_row_weight_balance"][name]["groups"]
        assert all(
            group["normalized_row_weight_mass"] == pytest.approx(1.0)
            for group in groups
        )
    assert sorted(observed) == list(range(8))
    assert len(observed) == len(set(observed))


def test_validation_diagnostic_cannot_restore_authenticated_incumbent_noop() -> None:
    model = _ToyModel()
    evidence = _evidence_bank(model, validation_matches_incumbent=True)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())
    gradients = tuple(
        parameter.grad.detach().clone() for parameter in model.parameters()
    )

    receipt = _compile(model, evidence)

    seal = receipt.pop("result_sha256")
    assert canonical_sha256(receipt) == seal
    assert receipt["status"] == "installed-frozen-semantic-short-conv-candidate"
    assert receipt["incumbent_selected_by_heldout_gates"] is False
    assert receipt["validation_nonregression_passed"] is False
    assert receipt["validation_diagnostic_would_have_failed_legacy_gate"] is True
    assert receipt["validation_metrics_diagnostic_only"] is True
    assert receipt["validation_publication_veto_applied"] is False
    assert receipt["rollback_authorized"] is False
    assert receipt["numeric_metrics_have_final_intelligence_authority"] is False
    assert receipt["installed"] is True
    assert receipt["install_count"] == 1
    assert receipt["changed_state_keys"]
    assert receipt["changed_parameter_roles"]
    assert receipt["selected_candidate"] in {
        "full-source-structural",
        "incumbent-producer-output-writer",
    }
    assert receipt["validation_rejected_candidate"] is None
    assert receipt["incumbent_noop_fallback_used"] is False
    assert receipt["selection_used_for_candidate_choice"] is False
    assert receipt["validation_used_for_candidate_choice"] is False
    assert receipt["validation_used_only_for_acceptance_or_incumbent_fallback"] is False
    assert "exact_zero_incumbent_at_entry" not in receipt
    assert "cold_start_compulsory_full_source_structural_plant" not in receipt
    assert receipt["exact_noop_state_proven"] is False
    assert receipt["model_state_before_sha256"] != receipt["model_state_after_sha256"]
    assert (
        receipt["target_topology_sha256_before"]
        == receipt["target_topology_sha256_after"]
    )
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["cache_key"] == _CACHE
    assert receipt["bridge_artifact_sha256"] == _BRIDGE
    assert receipt["source_model_identity_sha256"] == _SOURCE_MODEL
    assert receipt["compiler_lineage_sha256"] == _LINEAGE
    assert receipt["transport_evidence_sha256"] == [
        item.evidence_sha256 for item in evidence
    ]
    assert all(value == 0 for value in receipt["operation_audit"].values())
    assert model.forward_calls == 0
    assert all(
        (not torch.equal(parameter, before[name]))
        == (name in receipt["changed_state_keys"])
        for name, parameter in model.named_parameters()
    )
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids
    assert all(
        torch.equal(parameter.grad, gradient)
        for parameter, gradient in zip(model.parameters(), gradients)
    )


@pytest.mark.parametrize(
    "body_recipe",
    (
        "fit_guided",
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
        "structural_sequence_residual_mlp_writer",
    ),
)
def test_effectful_structural_candidate_survives_fit_diagnostic_miss(
    monkeypatch: pytest.MonkeyPatch,
    body_recipe: str,
) -> None:
    model = _ToyModel()
    evidence = _evidence_bank(model)

    def deliberately_bad_structural(**kwargs: object) -> adapter._Candidate:
        state = {
            name: value.detach().clone()
            for name, value in kwargs["incumbent_state"].items()
        }
        state["out_proj.weight"].fill_(256.0)
        return adapter._Candidate(
            name="full-source-structural",
            state=state,
            provenance={
                "solver": "test-finite-shape-correct-fit-diagnostic-miss",
                "full_source_coverage": True,
                "source_channels_dropped": 0,
                "operation_audit": dict(adapter._ZERO_OPERATIONS),
            },
        )

    def no_writer(*_args: object, **_kwargs: object) -> tuple[None, dict[str, object]]:
        return None, _cpu_writer_report("test-no-writer")

    monkeypatch.setattr(
        adapter,
        "_compile_structural_candidate",
        deliberately_bad_structural,
    )
    monkeypatch.setattr(adapter, "_compile_writer_candidate", no_writer)

    receipt = _compile(model, evidence, body_recipe=body_recipe)
    report = receipt["candidate_reports_before_validation"]["full-source-structural"]

    assert receipt["installed"] is True
    assert receipt["selected_candidate"] == "full-source-structural"
    assert report["structural_safety_passed"] is True
    assert report["effectful_nonincumbent"] is True
    assert report["structurally_eligible"] is True
    assert report["fit_improved"] is False
    assert report["partition_nonregression"]["fit"] is False
    assert report["numeric_diagnostic_miss"] is True
    assert report["numeric_diagnostic_miss_can_exclude_candidate"] is False
    assert receipt["validation_rejected_candidate"] is None
    assert receipt["incumbent_noop_fallback_used"] is False
    assert receipt["body_recipe_id"] == body_recipe


@pytest.mark.parametrize(
    "body_recipe",
    (
        "fit_guided",
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
        "structural_sequence_residual_mlp_writer",
    ),
)
def test_structurally_invalid_only_proposal_uses_authenticated_incumbent_control(
    monkeypatch: pytest.MonkeyPatch,
    body_recipe: str,
) -> None:
    model = _ToyModel()
    evidence = _evidence_bank(model)
    before = _fingerprint(model).identity_sha256

    def invalid_structural(**kwargs: object) -> adapter._Candidate:
        state = {
            name: value.detach().clone()
            for name, value in kwargs["incumbent_state"].items()
        }
        state["out_proj.weight"] = state["out_proj.weight"][:1].clone()
        return adapter._Candidate(
            name="full-source-structural",
            state=state,
            provenance={
                "solver": "test-invalid-shape",
                "full_source_coverage": True,
                "source_channels_dropped": 0,
                "operation_audit": dict(adapter._ZERO_OPERATIONS),
            },
        )

    def no_writer(*_args: object, **_kwargs: object) -> tuple[None, dict[str, object]]:
        return None, _cpu_writer_report("test-no-writer")

    monkeypatch.setattr(adapter, "_compile_structural_candidate", invalid_structural)
    monkeypatch.setattr(adapter, "_compile_writer_candidate", no_writer)

    receipt = _compile(model, evidence, body_recipe=body_recipe)
    report = receipt["candidate_reports_before_validation"]["full-source-structural"]

    assert receipt["status"] == "structurally-identical-or-invalid-incumbent-noop"
    assert receipt["installed"] is False
    assert receipt["selected_candidate"] == "incumbent-noop"
    assert receipt["incumbent_noop_fallback_used"] is True
    assert receipt["incumbent_selected_by_heldout_gates"] is False
    assert receipt["body_recipe_id"] == body_recipe
    assert report["structural_safety_passed"] is False
    assert report["structurally_eligible"] is False
    assert report["partition_metrics"]["fit"] is None
    assert _fingerprint(model).identity_sha256 == before
    assert stack_adapter._authenticated_heldout_incumbent_noop(
        receipt,
        component="short_conv",
        before_sha256=before,
        after_sha256=before,
        changed_state_keys=(),
    )


@pytest.mark.parametrize(
    "body_recipe",
    (
        "fit_guided",
        "full_strength_structural",
        "residual_writer",
        "source_producer_residual_writer",
        "structural_sequence_residual_mlp_writer",
    ),
)
def test_identical_analytic_candidate_resolves_to_sealed_noop(
    monkeypatch: pytest.MonkeyPatch,
    body_recipe: str,
) -> None:
    model = _ToyModel()
    evidence = _evidence_bank(model)
    before = _fingerprint(model).identity_sha256

    def identical_structural(**kwargs: object) -> adapter._Candidate:
        return adapter._Candidate(
            name="full-source-structural",
            state=kwargs["incumbent_state"],
            provenance={
                "solver": "test-identical-structural-candidate",
                "full_source_coverage": True,
                "source_channels_dropped": 0,
                "operation_audit": dict(adapter._ZERO_OPERATIONS),
            },
        )

    def no_writer(*_args: object, **_kwargs: object) -> tuple[None, dict[str, object]]:
        return None, _cpu_writer_report("test-identical-writer-rounded-to-incumbent")

    monkeypatch.setattr(adapter, "_compile_structural_candidate", identical_structural)
    monkeypatch.setattr(adapter, "_compile_writer_candidate", no_writer)
    receipt = _compile(model, evidence, body_recipe=body_recipe)

    seal = receipt.pop("result_sha256")
    assert canonical_sha256(receipt) == seal
    assert receipt["installed"] is False
    assert receipt["install_count"] == 0
    assert receipt["body_recipe_id"] == body_recipe
    assert receipt["status"] == "structurally-identical-or-invalid-incumbent-noop"
    assert receipt["selected_candidate"] == "incumbent-noop"
    assert receipt["prevalidation_selected_candidate"] == "incumbent-noop"
    assert receipt["validation_rejected_candidate"] is None
    assert receipt["incumbent_noop_fallback_used"] is True
    assert receipt["incumbent_selected_by_heldout_gates"] is False
    assert receipt["structurally_eligible_candidate_count"] == 0
    assert (
        receipt["candidate_reports_before_validation"]["full-source-structural"][
            "fit_improved"
        ]
        is False
    )
    assert (
        receipt["candidate_reports_before_validation"]["full-source-structural"][
            "heldout_selection_improved"
        ]
        is False
    )
    assert (
        receipt["candidate_reports_before_validation"]["full-source-structural"][
            "structurally_eligible"
        ]
        is False
    )
    assert receipt["selected_candidate_state_identical_to_incumbent"] is True
    assert receipt["transport_evidence_sha256"] == [
        item.evidence_sha256 for item in evidence
    ]
    assert _fingerprint(model).identity_sha256 == before
    assert stack_adapter._authenticated_heldout_incumbent_noop(
        receipt,
        component="short_conv",
        before_sha256=receipt["model_state_before_sha256"],
        after_sha256=receipt["model_state_after_sha256"],
        changed_state_keys=receipt["changed_state_keys"],
    )


def test_stale_or_mutated_evidence_fails_before_install() -> None:
    model = _ToyModel()
    stale = _evidence_bank(model)
    with torch.no_grad():
        model.model.layers[0].conv.out_proj.bias.add_(1.0)
    changed = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    with pytest.raises(
        Lfm2SemanticShortConvRefoldError,
        match="captured child state diverged",
    ):
        _compile(model, stale)
    assert all(
        torch.equal(parameter, changed[name])
        for name, parameter in model.named_parameters()
    )

    fresh = list(_evidence_bank(model))
    fresh[0].parent_b_history[0, 0, 0] += 1.0
    with pytest.raises(
        Lfm2SemanticShortConvRefoldError,
        match="changed after authentication",
    ):
        _compile(model, tuple(fresh))


def test_transaction_restores_exact_bytes_after_partial_copy_failure() -> None:
    model = _ToyModel()
    parameters = adapter._parameter_bundle(model, 0)
    state = dict(adapter._suffix_state(parameters, 0))
    state["in_proj.weight"] = state["in_proj.weight"] + torch.tensor(
        0.5,
        dtype=torch.bfloat16,
    )
    state["conv.weight"] = state["conv.weight"][:, :, :1]
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    with pytest.raises(
        Lfm2SemanticShortConvRefoldError,
        match="changed the topology or dtype",
    ):
        adapter._install_state_transaction(
            model,
            layer_index=0,
            state=state,
        )

    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )


def test_evidence_rejects_contraction_and_clones_caller_tensors() -> None:
    model = _ToyModel()
    evidence = _evidence_bank(model)[0]
    original = evidence.parent_b_history.clone()
    caller = original.clone()
    copied = Lfm2SemanticShortConvReplayEvidence(
        batch_index=evidence.batch_index,
        phase=evidence.phase,
        layer_index=evidence.layer_index,
        alignment_sha256=evidence.alignment_sha256,
        cache_key=evidence.cache_key,
        bridge_artifact_sha256=evidence.bridge_artifact_sha256,
        source_capability_sha256=evidence.source_capability_sha256,
        source_history_sha256=evidence.source_history_sha256,
        compiler_lineage_sha256=evidence.compiler_lineage_sha256,
        captured_child_model_state_sha256=(evidence.captured_child_model_state_sha256),
        target_to_parent_span_owner_positions=(0, 1, 1, 2),
        endpoint_target_positions=(3,),
        actual_child_operator_input_history=(
            evidence.actual_child_operator_input_history
        ),
        actual_child_pre_operator_residual_history=(
            evidence.actual_child_pre_operator_residual_history
        ),
        parent_b_history=caller,
        parent_c_history=evidence.parent_c_history,
        parent_x_history=evidence.parent_x_history,
        parent_post_operator_residual_target_space=(
            evidence.parent_post_operator_residual_target_space
        ),
        endpoint_row_sqrt_weights=evidence.endpoint_row_sqrt_weights,
    )
    caller.add_(100.0)
    assert torch.equal(copied.parent_b_history, original)
    assert copied.verify() == copied.evidence_sha256

    with pytest.raises(
        Lfm2SemanticShortConvRefoldError,
        match="pure-expansion history",
    ):
        Lfm2SemanticShortConvReplayEvidence(
            batch_index=evidence.batch_index,
            phase=evidence.phase,
            layer_index=evidence.layer_index,
            alignment_sha256=evidence.alignment_sha256,
            cache_key=evidence.cache_key,
            bridge_artifact_sha256=evidence.bridge_artifact_sha256,
            source_capability_sha256=evidence.source_capability_sha256,
            source_history_sha256=evidence.source_history_sha256,
            compiler_lineage_sha256=evidence.compiler_lineage_sha256,
            captured_child_model_state_sha256=(
                evidence.captured_child_model_state_sha256
            ),
            target_to_parent_span_owner_positions=(0, 0, 2, 2),
            endpoint_target_positions=(3,),
            actual_child_operator_input_history=(
                evidence.actual_child_operator_input_history
            ),
            actual_child_pre_operator_residual_history=(
                evidence.actual_child_pre_operator_residual_history
            ),
            parent_b_history=evidence.parent_b_history,
            parent_c_history=evidence.parent_c_history,
            parent_x_history=evidence.parent_x_history,
            parent_post_operator_residual_target_space=(
                evidence.parent_post_operator_residual_target_space
            ),
            endpoint_row_sqrt_weights=evidence.endpoint_row_sqrt_weights,
        )
