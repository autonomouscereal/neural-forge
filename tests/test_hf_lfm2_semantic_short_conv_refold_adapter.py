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


def _compile(
    model: _ToyModel,
    evidence: tuple[Lfm2SemanticShortConvReplayEvidence, ...],
    *,
    fold_compute_device: str | None = None,
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
            **(
                {"fold_compute_device": fold_compute_device}
                if fold_compute_device is not None
                else {}
            ),
        )
    )


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
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["model_forward_calls"] == 0
    assert model.forward_calls == 0
    assert sum(parameter.numel() for parameter in model.parameters()) == count
    assert not torch.equal(operator.out_proj.weight, before_out)
    if receipt["selected_candidate"] == "incumbent-producer-output-writer":
        assert torch.equal(operator.in_proj.weight, before_in)
        assert torch.equal(operator.conv.weight, before_conv)
        assert all("out_proj" in name for name in receipt["changed_state_keys"])
        writer_selection = reports["incumbent-producer-output-writer"][
            "partition_metrics"
        ]["selection"]
        structural_selection = reports["full-source-structural"]["partition_metrics"][
            "selection"
        ]
        assert (
            writer_selection["endpoint_relative_rmse"]
            <= structural_selection["endpoint_relative_rmse"] + 1e-8
        )
        assert (
            writer_selection["endpoint_maximum_absolute_error"]
            <= structural_selection["endpoint_maximum_absolute_error"] + 1e-8
        )
        assert (
            writer_selection["endpoint_cosine_similarity_mean"] + 1e-8
            >= structural_selection["endpoint_cosine_similarity_mean"]
        )
        assert (
            writer_selection["endpoint_cosine_similarity_worst"] + 1e-8
            >= structural_selection["endpoint_cosine_similarity_worst"]
        )


def test_explicit_cpu_fold_device_preserves_default_receipt_and_state_bytes() -> None:
    implicit_model = _ToyModel()
    explicit_model = _ToyModel()

    implicit = _compile(implicit_model, _evidence_bank(implicit_model))
    explicit = _compile(
        explicit_model,
        _evidence_bank(explicit_model),
        fold_compute_device="cpu",
    )

    assert implicit == explicit
    assert "fold_compute_device" not in explicit
    assert "structural_cluster_compute_execution" not in explicit
    assert "writer_compute_execution" not in explicit
    assert all(
        torch.equal(value, explicit_model.state_dict()[name])
        for name, value in implicit_model.state_dict().items()
    )


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


def test_validation_rejection_falls_back_to_authenticated_incumbent_noop() -> None:
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

    receipt = _compile(model, evidence)

    seal = receipt.pop("result_sha256")
    assert canonical_sha256(receipt) == seal
    assert receipt["status"] == "heldout-selected-incumbent-noop"
    assert receipt["incumbent_selected_by_heldout_gates"] is True
    assert receipt["fit_check_nonregression_passed"] is True
    assert receipt["selection_nonregression_passed"] is True
    assert receipt["validation_nonregression_passed"] is True
    assert stack_adapter._authenticated_heldout_incumbent_noop(
        receipt,
        before_sha256=receipt["model_state_before_sha256"],
        after_sha256=receipt["model_state_after_sha256"],
        changed_state_keys=receipt["changed_state_keys"],
    )
    assert receipt["installed"] is False
    assert receipt["install_count"] == 0
    assert receipt["changed_state_keys"] == []
    assert receipt["changed_parameter_roles"] == []
    assert receipt["selected_candidate"] == "incumbent-noop"
    assert receipt["validation_rejected_candidate"] in {
        "full-source-structural",
        "incumbent-producer-output-writer",
    }
    assert receipt["incumbent_noop_fallback_used"] is True
    assert "exact_zero_incumbent_at_entry" not in receipt
    assert "cold_start_compulsory_full_source_structural_plant" not in receipt
    assert receipt["exact_noop_state_proven"] is True
    assert receipt["model_state_before_sha256"] == receipt["model_state_after_sha256"]
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
        torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids


def test_identical_analytic_candidate_resolves_to_sealed_noop(
    monkeypatch: pytest.MonkeyPatch,
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
        return None, {
            "status": "test-identical-writer-rounded-to-incumbent",
            "operation_audit": dict(adapter._ZERO_OPERATIONS),
        }

    monkeypatch.setattr(adapter, "_compile_structural_candidate", identical_structural)
    monkeypatch.setattr(adapter, "_compile_writer_candidate", no_writer)
    receipt = _compile(model, evidence)

    seal = receipt.pop("result_sha256")
    assert canonical_sha256(receipt) == seal
    assert receipt["installed"] is False
    assert receipt["install_count"] == 0
    assert receipt["selected_candidate"] == "incumbent-noop"
    assert receipt["prevalidation_selected_candidate"] == "incumbent-noop"
    assert receipt["validation_rejected_candidate"] is None
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
    assert receipt["selected_candidate_state_identical_to_incumbent"] is True
    assert receipt["transport_evidence_sha256"] == [
        item.evidence_sha256 for item in evidence
    ]
    assert _fingerprint(model).identity_sha256 == before


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
