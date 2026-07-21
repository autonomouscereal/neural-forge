from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _ToyShortConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj = torch.nn.Linear(
            2,
            6,
            bias=True,
            dtype=torch.bfloat16,
        )
        self.conv = torch.nn.Conv1d(
            2,
            2,
            kernel_size=2,
            groups=2,
            bias=True,
            dtype=torch.bfloat16,
        )
        self.out_proj = torch.nn.Linear(
            2,
            2,
            bias=True,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            self.in_proj.weight.zero_()
            self.in_proj.bias.zero_()
            self.in_proj.weight[:2].copy_(torch.eye(2, dtype=torch.bfloat16))
            self.in_proj.bias[2:4].fill_(1.0)
            self.in_proj.weight[4:].copy_(torch.eye(2, dtype=torch.bfloat16))
            self.conv.weight.copy_(
                torch.tensor(
                    [[[0.25, 0.75]], [[-0.125, 0.625]]],
                    dtype=torch.bfloat16,
                )
            )
            self.conv.bias.zero_()
            self.out_proj.weight.zero_()
            self.out_proj.bias.zero_()


class _ToyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = _ToyShortConv()


class _ToyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer()])


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyBackbone()
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


def _operator_context(model: _ToyModel, inputs: torch.Tensor) -> torch.Tensor:
    operator = model.model.layers[0].conv
    projected = (
        inputs.to(torch.float64) @ operator.in_proj.weight.detach().to(torch.float64).T
    )
    projected = projected + operator.in_proj.bias.detach().to(torch.float64)
    b_value, c_value, x_value = projected.chunk(3, dim=-1)
    gated = (b_value * x_value).transpose(1, 2)
    history = torch.nn.functional.pad(gated, (1, 0))
    filtered = torch.nn.functional.conv1d(
        history,
        operator.conv.weight.detach().to(torch.float64),
        bias=operator.conv.bias.detach().to(torch.float64),
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
    assert receipt["exact_noop_state_proven"] is True
    assert receipt["model_state_before_sha256"] == receipt["model_state_after_sha256"]
    assert receipt["target_topology_sha256_before"] == receipt[
        "target_topology_sha256_after"
    ]
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
    assert receipt["candidate_reports_before_validation"][
        "full-source-structural"
    ]["fit_improved"] is False
    assert receipt["candidate_reports_before_validation"][
        "full-source-structural"
    ]["heldout_selection_improved"] is False
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
