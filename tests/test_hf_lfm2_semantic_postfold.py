from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
import neural_forge.hf_lfm2_cpcf_terminal_action_residual as terminal_action  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_artifact import DynamicFoldArtifactError  # noqa: E402


class _TiedModel:
    def __init__(self, weight):
        self.weight = torch.nn.Parameter(weight.clone())
        self.model = SimpleNamespace(
            layers=(),
            rotary_emb=SimpleNamespace(
                inv_freq=torch.tensor([1.0], dtype=torch.float32),
                attention_scaling=1.0,
            ),
        )

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def parameters(self):
        yield self.weight

    def state_dict(self):
        return {"model.embed_tokens.weight": self.weight}


class _GaugeTiedModel(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def get_output_embeddings(self):
        return SimpleNamespace(weight=self.weight)


def test_self_dual_tied_gauge_install_preserves_operator_and_runtime(monkeypatch):
    generator = torch.Generator().manual_seed(1701)
    target = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    encoder = torch.randn((5, 3), generator=generator, dtype=torch.float64)
    source_body = {
        "schema": "synthetic-tied-interface-v1",
        "target_weight_sha256": postfold._interface_tensor_sha256(target),
        "producer_encoder_sha256": postfold._interface_tensor_sha256(encoder),
    }
    interface = postfold.TiedCausalInterfaceFold(
        target_weight=target,
        producer_encoder=encoder,
        report={
            **source_body,
            "report_sha256": canonical_sha256(source_body),
        },
    )
    model = _GaugeTiedModel(target.to(torch.bfloat16))
    model.weight.grad = torch.full_like(model.weight, 0.25)
    parameter_id = id(model.weight)
    storage_pointer = int(model.weight.data_ptr())
    gradient_id = id(model.weight.grad)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda value: SimpleNamespace(
            identity_sha256=canonical_sha256(postfold._state_tensor_hashes(value))
        ),
    )

    self_dual, receipt = postfold.install_lfm2_self_dual_tied_interface_gauge(
        model,
        interface,
    )

    torch.testing.assert_close(
        self_dual.producer_encoder.T @ self_dual.producer_encoder,
        torch.eye(3, dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    assert torch.equal(
        model.weight.detach(),
        self_dual.target_weight.to(torch.bfloat16),
    )
    assert receipt["global_residual_gauge_frozen"] is True
    assert receipt["independent_per_layer_gauge_rotation_allowed"] is False
    assert receipt["materialized_source_to_vocabulary_operator_relative_error"] < (0.01)
    assert receipt["training_steps"] == 0
    assert receipt["optimizer_steps"] == 0
    assert receipt["backward_calls"] == 0
    assert id(model.weight) == parameter_id
    assert int(model.weight.data_ptr()) == storage_pointer
    assert id(model.weight.grad) == gradient_id


def _terminal_evidence(split: str, tag: str):
    return postfold.Lfm2CpcfTerminalActionEvidence(
        split=split,
        row_ids=(f"{tag}-row",),
        mlp_input=torch.ones((1, 2), dtype=torch.bfloat16),
        pre_mlp_residual=torch.zeros((1, 2), dtype=torch.bfloat16),
        parent_final_norm_output=torch.ones((1, 2), dtype=torch.bfloat16),
        sample_weights=torch.ones((1,), dtype=torch.float64),
    )


class _TerminalFeedForward(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.w2 = torch.nn.Linear(
            int(weight.shape[1]),
            int(weight.shape[0]),
            bias=False,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            self.w2.weight.copy_(weight)


class _TerminalLayer(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.feed_forward = _TerminalFeedForward(weight)


class _TerminalBackbone(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.layers = torch.nn.ModuleList([_TerminalLayer(weight)])


class _TerminalTarget(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.model = _TerminalBackbone(weight)
        self.other = torch.nn.Parameter(torch.tensor([3.0], dtype=weight.dtype))


def _terminal_target_state(model) -> str:
    return canonical_sha256(postfold._state_tensor_hashes(model))


def _terminal_result(
    *,
    incumbent,
    selected,
    published,
    selected_candidate_id: str,
    accepted: bool,
    validation_passed: bool,
    rollback_performed: bool,
):
    policy = terminal_action.Lfm2CpcfTerminalActionPolicy()
    body = {
        "policy_sha256": policy.authentication_sha256,
        "selected_candidate_id": selected_candidate_id,
        "accepted": accepted,
        "selected_nonincumbent": selected_candidate_id != "incumbent",
        "published_is_incumbent": torch.equal(published, incumbent),
        "rollback_performed": rollback_performed,
        "rollback_exact": True,
        "source_seed_writer_unchanged": True,
        "parameter_identity_or_topology_mutation": False,
        "parameter_count_before": int(incumbent.numel()),
        "parameter_count_after": int(published.numel()),
        "validation": {"passed": validation_passed},
        "output_state_sha256": {
            "incumbent": terminal_action._tensor_sha256(incumbent),
            "selected": terminal_action._tensor_sha256(selected),
            "published": terminal_action._tensor_sha256(published),
        },
        "operation_audit": dict(terminal_action._ZERO_OPERATION_AUDIT),
    }
    result_sha256 = terminal_action._canonical_sha256(body)
    return terminal_action.Lfm2CpcfTerminalActionResidualResult(
        policy=policy,
        incumbent_down_weight=incumbent,
        selected_down_weight=selected,
        published_down_weight=published,
        selected_candidate_id=selected_candidate_id,
        accepted=accepted,
        report={**body, "result_sha256": result_sha256},
        result_sha256=result_sha256,
    )


def _compiled_exact_zero_terminal_result():
    generator = torch.Generator().manual_seed(1_901)
    hidden = 2
    intermediate = 3
    epsilon = 1e-5
    gate = (torch.randn(intermediate, hidden, generator=generator) * 0.55).to(
        torch.bfloat16
    )
    up = (torch.randn(intermediate, hidden, generator=generator) * 0.55).to(
        torch.bfloat16
    )
    target = torch.tensor(
        [[0.20, -0.16, 0.18], [-0.09, 0.07, -0.08]],
        dtype=torch.bfloat16,
    )
    tied_table = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0]],
        dtype=torch.bfloat16,
    )
    seed = terminal_action.Lfm2CpcfTerminalActionSeed(
        gate_weight=gate,
        up_weight=up,
        down_weight=torch.zeros((hidden, intermediate), dtype=torch.bfloat16),
        final_norm_weight=torch.ones(hidden, dtype=torch.bfloat16),
        producer_encoder=torch.eye(hidden, dtype=torch.float64),
        target_tied_weight=tied_table,
        source_tied_table=tied_table,
        norm_epsilon=epsilon,
    )

    def evidence(split, rows, random_seed):
        split_generator = torch.Generator().manual_seed(random_seed)
        mlp_input = (torch.randn(rows, hidden, generator=split_generator) * 0.75).to(
            torch.bfloat16
        )
        pre_mlp = (torch.randn(rows, hidden, generator=split_generator) * 0.30).to(
            torch.float64
        )
        pre_mlp[:, 0] = torch.where(
            torch.arange(rows) % 2 == 0,
            1.4,
            -1.4,
        )
        features = torch.nn.functional.silu(
            mlp_input.to(torch.float32) @ gate.to(torch.float32).T
        ) * (mlp_input.to(torch.float32) @ up.to(torch.float32).T)
        target_post = pre_mlp + features.to(torch.float64) @ target.to(torch.float64).T
        parent_final_norm = (
            target_post
            / (target_post.square().mean(dim=1, keepdim=True) + epsilon).sqrt()
        )
        return terminal_action.Lfm2CpcfTerminalActionEvidence(
            split=split,
            row_ids=tuple(f"{split}:{index}" for index in range(rows)),
            mlp_input=mlp_input,
            pre_mlp_residual=pre_mlp.to(torch.bfloat16),
            parent_final_norm_output=parent_final_norm.to(torch.bfloat16),
            sample_weights=torch.ones(rows, dtype=torch.float64),
        )

    banks = {
        "fit": evidence("fit", 32, 1),
        "fit_check": evidence("fit_check", 16, 2),
        "selection": evidence("selection", 16, 3),
        "validation": evidence("validation", 16, 4),
    }
    result = terminal_action.compile_lfm2_cpcf_terminal_action_residual(
        seed=seed,
        fit_evidence=banks["fit"],
        fit_check_evidence=banks["fit_check"],
        selection_evidence=banks["selection"],
        validation_evidence=banks["validation"],
    )
    return seed, banks, result


def test_final_w2_incumbent_retention_is_an_exact_zero_copy_decision(
    monkeypatch,
) -> None:
    target = _TerminalTarget(torch.eye(2, dtype=torch.bfloat16))
    parameter = target.model.layers[-1].feed_forward.w2.weight
    parameter.grad = torch.full_like(parameter, 0.25)
    target.other.grad = torch.full_like(target.other, 0.5)
    before_state = _terminal_target_state(target)
    before_parameter_id = id(parameter)
    before_parameter_ptr = int(parameter.data_ptr())
    before_gradient_id = id(parameter.grad)
    before_gradient_ptr = int(parameter.grad.data_ptr())
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_terminal_target_state(model)),
    )

    receipt = postfold._install_final_w2_once(
        target,
        published_down_weight=parameter.detach().cpu().clone(),
    )

    assert receipt["installed"] is False
    assert receipt["install_count"] == 0
    assert receipt["mutation_count"] == 0
    assert receipt["incumbent_retention_count"] == 1
    assert receipt["exact_zero_copy_noop"] is True
    assert receipt["parameter_copy_calls"] == 0
    assert receipt["model_state_before_sha256"] == before_state
    assert receipt["model_state_after_sha256"] == before_state
    assert receipt["changed_state_keys"] == []
    assert receipt["parameter_topology_preserved"] is True
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["published_w2_structurally_nonzero"] is True
    assert receipt["published_w2_structural_coverage"]["structurally_nonzero"] is True
    assert id(parameter) == before_parameter_id
    assert int(parameter.data_ptr()) == before_parameter_ptr
    assert id(parameter.grad) == before_gradient_id
    assert int(parameter.grad.data_ptr()) == before_gradient_ptr


def test_final_w2_changed_publication_mutates_only_the_writer(monkeypatch) -> None:
    target = _TerminalTarget(torch.eye(2, dtype=torch.bfloat16))
    parameter = target.model.layers[-1].feed_forward.w2.weight
    target.other.grad = torch.full_like(target.other, 0.5)
    before_other = target.other.detach().clone()
    published = torch.full_like(parameter.detach().cpu(), 0.75)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_terminal_target_state(model)),
    )

    receipt = postfold._install_final_w2_once(
        target,
        published_down_weight=published,
    )

    assert receipt["installed"] is True
    assert receipt["install_count"] == 1
    assert receipt["mutation_count"] == 1
    assert receipt["incumbent_retention_count"] == 0
    assert receipt["parameter_copy_calls"] == 1
    assert receipt["changed_state_keys"] == ["model.layers.0.feed_forward.w2.weight"]
    assert receipt["model_state_before_sha256"] != receipt["model_state_after_sha256"]
    torch.testing.assert_close(parameter.detach().cpu(), published, rtol=0, atol=0)
    torch.testing.assert_close(target.other.detach(), before_other, rtol=0, atol=0)
    assert receipt["parameter_topology_preserved"] is True
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["published_w2_structurally_nonzero"] is True


def test_exact_zero_fit_plant_is_authenticated_and_installed_transactionally(
    monkeypatch,
) -> None:
    seed, evidence, terminal = _compiled_exact_zero_terminal_result()
    target = _TerminalTarget(seed.down_weight)
    parameter = target.model.layers[-1].feed_forward.w2.weight
    parameter.grad = torch.full_like(parameter, 0.25)
    before_parameter_id = id(parameter)
    before_storage = int(parameter.data_ptr())
    before_gradient_id = id(parameter.grad)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_terminal_target_state(model)),
    )

    published, decision = postfold._terminal_publication_decision(
        terminal,
        expected_seed=seed,
        expected_evidence=evidence,
    )
    receipt = postfold._install_final_w2_once(
        target,
        published_down_weight=published,
    )

    assert torch.count_nonzero(seed.down_weight).item() == 0
    assert decision["publication_decision"] == ("exact-zero-fit-only-structural-plant")
    assert decision["terminal_exact_zero_cold_start"] is True
    assert decision["heldout_metrics_diagnostic_only"] is True
    assert decision["terminal_incumbent_retained"] is False
    assert receipt["exact_zero_cold_start"] is True
    assert receipt["incumbent_all_elements_zero"] is True
    assert receipt["incumbent_noop_publication_forbidden"] is True
    assert receipt["transactional_dense_cold_start_plant"] is True
    assert receipt["installed"] is True
    assert receipt["install_count"] == 1
    assert receipt["incumbent_retention_count"] == 0
    assert receipt["published_w2_all_elements_finite"] is True
    assert receipt["published_w2_all_output_units_covered"] is True
    assert receipt["published_w2_all_input_units_covered"] is True
    assert receipt["parameter_topology_preserved"] is True
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["parameter_count_changed"] is False
    assert set(receipt["operation_audit"].values()) == {0}
    assert torch.equal(parameter.detach().cpu(), published)
    assert id(parameter) == before_parameter_id
    assert int(parameter.data_ptr()) == before_storage
    assert id(parameter.grad) == before_gradient_id


def test_exact_zero_publication_recomputes_fit_and_rejects_dense_forgery() -> None:
    seed, evidence, terminal = _compiled_exact_zero_terminal_result()
    forged_weight = terminal.selected_down_weight.clone()
    forged_weight[0, 0] = forged_weight[0, 0] + 1.0
    assert not torch.equal(forged_weight, terminal.selected_down_weight)
    forged_hash = terminal_action._tensor_sha256(forged_weight)
    forged_coverage = dict(
        terminal_action._terminal_writer_structural_coverage(forged_weight)
    )
    body = copy.deepcopy(terminal.as_dict())
    body.pop("result_sha256")
    body["output_state_sha256"]["selected"] = forged_hash
    body["output_state_sha256"]["published"] = forged_hash
    body["selected_state_sha256"] = forged_hash
    body["selected_selection_key"][-1] = forged_hash
    body["selected_writer_structural_coverage"] = forged_coverage
    body["selected_writer_structural_coverage_sha256"] = forged_coverage[
        "coverage_sha256"
    ]
    body["selected_writer_all_elements_nonzero"] = forged_coverage[
        "all_elements_nonzero"
    ]
    body["validation"]["frozen_candidate_state_sha256"] = forged_hash
    selected_frontier = next(
        value
        for value in body["frontier"]
        if value["candidate_id"] == body["selected_candidate_id"]
    )
    selected_frontier["state_sha256"] = forged_hash
    selected_frontier["selection_key"][-1] = forged_hash
    selected_frontier["structural_coverage"] = forged_coverage
    selected_frontier["structural_coverage_sha256"] = forged_coverage["coverage_sha256"]
    selected_frontier["frozen_candidate_state_sha256"] = forged_hash
    forged_sha256 = terminal_action._canonical_sha256(body)
    forged = terminal_action.Lfm2CpcfTerminalActionResidualResult(
        policy=terminal.policy,
        incumbent_down_weight=terminal.incumbent_down_weight,
        selected_down_weight=forged_weight,
        published_down_weight=forged_weight,
        selected_candidate_id=terminal.selected_candidate_id,
        accepted=True,
        report={**body, "result_sha256": forged_sha256},
        result_sha256=forged_sha256,
        _compiler_input_binding=terminal._compiler_input_binding,
        _compiler_issuer=terminal_action._TERMINAL_RESULT_COMPILER_ISSUER,
    )
    assert forged.verify_authentication() is True

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="published neither",
    ):
        postfold._terminal_publication_decision(
            forged,
            expected_seed=seed,
            expected_evidence=evidence,
        )


@pytest.mark.parametrize(
    "published",
    (
        torch.zeros((2, 2), dtype=torch.bfloat16),
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16),
    ),
)
def test_final_w2_refuses_blank_or_structurally_partial_publication(
    monkeypatch,
    published,
) -> None:
    target = _TerminalTarget(torch.eye(2, dtype=torch.bfloat16))
    before = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda model: SimpleNamespace(identity_sha256=_terminal_target_state(model)),
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="blank or structurally partial W2",
    ):
        postfold._install_final_w2_once(
            target,
            published_down_weight=published,
        )

    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_validation_rejected_selected_writer_publishes_only_incumbent() -> None:
    incumbent = torch.eye(2, dtype=torch.bfloat16)
    selected = torch.ones((2, 2), dtype=torch.bfloat16)
    terminal = _terminal_result(
        incumbent=incumbent,
        selected=selected,
        published=incumbent.clone(),
        selected_candidate_id="gn-00-ridge-00-alpha-01",
        accepted=False,
        validation_passed=False,
        rollback_performed=True,
    )

    published, decision = postfold._terminal_publication_decision(terminal)

    assert torch.equal(published, incumbent)
    assert not torch.equal(published, selected)
    assert decision["publication_decision"] == (
        "frozen-candidate-validation-rollback-to-incumbent"
    )
    assert decision["terminal_incumbent_retained"] is True
    assert decision["terminal_nonincumbent_candidate_accepted"] is False


def test_exact_zero_writer_cannot_use_legacy_validation_rollback() -> None:
    incumbent = torch.zeros((2, 2), dtype=torch.bfloat16)
    selected = torch.ones((2, 2), dtype=torch.bfloat16)
    terminal = _terminal_result(
        incumbent=incumbent,
        selected=selected,
        published=incumbent.clone(),
        selected_candidate_id="gn-00-ridge-00-alpha-01",
        accepted=False,
        validation_passed=False,
        rollback_performed=True,
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="published neither",
    ):
        postfold._terminal_publication_decision(terminal)

    forged_body = terminal.as_dict()
    forged_body.pop("result_sha256")
    forged_body["exact_zero_cold_start"] = True
    forged_sha256 = terminal_action._canonical_sha256(forged_body)
    forged = terminal_action.Lfm2CpcfTerminalActionResidualResult(
        policy=terminal.policy,
        incumbent_down_weight=terminal.incumbent_down_weight,
        selected_down_weight=terminal.selected_down_weight,
        published_down_weight=terminal.published_down_weight,
        selected_candidate_id=terminal.selected_candidate_id,
        accepted=terminal.accepted,
        report={**forged_body, "result_sha256": forged_sha256},
        result_sha256=forged_sha256,
    )
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="published neither",
    ):
        postfold._terminal_publication_decision(forged)


def test_fused_terminal_never_gates_or_installs_the_private_selected_tensor() -> None:
    source = inspect.getsource(
        postfold.compile_install_lfm2_fused_semantic_terminal_refold
    )

    assert "terminal.selected_down_weight" not in source
    assert "published_down_weight=published_down_weight" in source


def test_nonzero_heldout_regression_is_a_sealed_diagnostic_only(
    monkeypatch,
) -> None:
    class Metrics:
        def __init__(self, name: str) -> None:
            self.name = name

        def as_dict(self) -> dict[str, object]:
            return {"name": self.name, "finite": True}

    monkeypatch.setattr(
        postfold,
        "_metric_gates",
        lambda *_args, **_kwargs: {"heldout_not_regressed": False},
    )
    report = postfold._terminal_heldout_diagnostic(
        Metrics("published"),
        Metrics("nonzero-incumbent"),
        terminal_action.Lfm2CpcfTerminalActionPolicy(),
    )

    assert report["passed"] is False
    assert report["role"] == (
        "diagnostic-only-after-frozen-terminal-publication-decision"
    )
    assert report["publication_gate_applied"] is False
    body = dict(report)
    supplied = body.pop("diagnostic_sha256")
    assert supplied == canonical_sha256(body)
    source = inspect.getsource(
        postfold.compile_install_lfm2_fused_semantic_terminal_refold
    )
    assert "separate-heldout-lane-regression" not in source


def test_heldout_selected_incumbent_is_a_valid_terminal_publication() -> None:
    incumbent = torch.eye(2, dtype=torch.bfloat16)
    terminal = _terminal_result(
        incumbent=incumbent,
        selected=incumbent.clone(),
        published=incumbent.clone(),
        selected_candidate_id="incumbent",
        accepted=False,
        validation_passed=True,
        rollback_performed=False,
    )

    published, decision = postfold._terminal_publication_decision(terminal)

    assert torch.equal(published, incumbent)
    assert decision["publication_decision"] == "heldout-selected-incumbent"
    assert decision["terminal_publication_decision_accepted"] is True
    assert decision["terminal_writer_changed"] is False


def test_reconstructs_saved_mean_preserving_interface_without_parent_model() -> None:
    source = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, -1.0, 1.0],
            [0.5, 4.0, -2.0],
            [-1.0, 0.25, 2.0],
        ],
        dtype=torch.bfloat16,
    )
    target = torch.tensor(
        [
            [0.5, 1.0],
            [0.5, -1.0],
            [0.5, 0.5],
            [0.5, -0.5],
        ],
        dtype=torch.bfloat16,
    )
    model = _TiedModel(target)

    interface = postfold.reconstruct_lfm2_saved_tied_interface(model, source)

    assert interface.target_width == 2
    assert interface.source_width == 3
    assert interface.vocabulary_size == 4
    assert interface.report["checkpoint_target_bytes_are_authoritative"] is True
    assert interface.report["parent_model_calls"] == 0
    body = dict(interface.report)
    seal = body.pop("report_sha256")
    assert seal == canonical_sha256(body)
    torch.testing.assert_close(
        interface.target_weight.to(torch.bfloat16),
        model.weight.detach(),
        rtol=0,
        atol=0,
    )


def _seal(value, field):
    return {**value, field: canonical_sha256(value)}


def test_compiler_lineage_accepts_distinct_semantic_cache_key(monkeypatch) -> None:
    compiler_key = "1" * 64
    semantic_key = "2" * 64
    parent = "3" * 64
    bridge_sha = "4" * 64
    target_state = "5" * 64
    artifact = {
        "query_groups": [[0]],
        "query_barycentric_weights": [[1.0]],
    }
    direct = _seal(
        {
            "candidate_id": "direct_monolithic",
            "operator_artifact": artifact,
            "operator_artifact_sha256": canonical_sha256(artifact),
            "selection_context_metrics": {
                "query_groups_sha256": canonical_sha256(
                    {"groups": [[0]], "weights": [[1.0]]}
                )
            },
        },
        "report_sha256",
    )
    endpoint = _seal(
        {
            "source_inventory": {"fixture": True},
            "source_inventory_sha256": "6" * 64,
        },
        "artifact_sha256",
    )
    layer = _seal(
        {
            "layer_index": 2,
            "operator_kind": "full_attention",
            "solver_evidence_policy": {
                "attention_candidate_evaluations": {
                    "direct_monolithic": direct,
                },
                "endpoint_residual_attention_artifact": endpoint,
                "endpoint_residual_attention_artifact_sha256": endpoint[
                    "artifact_sha256"
                ],
            },
        },
        "result_sha256",
    )
    compiled = _seal({"fixture": "compiled"}, "result_sha256")
    diagnostic = {
        "receipt_sha256": "7" * 64,
        "inputs": {
            "cache_key": compiler_key,
            "parent_fingerprint": parent,
            "bridge_artifact_sha256": bridge_sha,
        },
        "compiled_candidate_consumer_seam": {
            "cache_sha256": compiler_key,
            "parent_sha256": parent,
            "bridge_sha256": bridge_sha,
            "target_state_sha256": target_state,
        },
        "compiled_model_fold": compiled,
        "layer_results": [layer],
    }
    inventory = SimpleNamespace(inventory_sha256="6" * 64)
    monkeypatch.setattr(
        postfold,
        "_authenticated_c61_diagnostic",
        lambda *_a, **_k: (MappingProxyType(diagnostic), "8" * 64),
    )
    monkeypatch.setattr(postfold, "_typed_qkvo_inventory", lambda _value: inventory)
    monkeypatch.setattr(
        postfold,
        "Lfm2SemanticCompilerLayerLineage",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    result = postfold.load_lfm2_semantic_compiler_lineage(
        "unused.json",
        expected_diagnostic_sha256=diagnostic["receipt_sha256"],
        cache=SimpleNamespace(cache_key=semantic_key, parent_fingerprint=parent),
        bridge=SimpleNamespace(artifact_sha256=bridge_sha),
        input_metadata={"model_state_sha256": target_state},
    )

    assert result[2].report["compiler_cache_key"] == compiler_key
    assert result[2].report["semantic_evidence_cache_key"] == semantic_key
    assert (
        result[2].report["compiler_and_semantic_cache_identities_bound_separately"]
        is True
    )
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="not the compiler ancestry",
    ):
        postfold.load_lfm2_semantic_compiler_lineage(
            "unused.json",
            expected_diagnostic_sha256=diagnostic["receipt_sha256"],
            cache=SimpleNamespace(
                cache_key=semantic_key,
                parent_fingerprint="f" * 64,
            ),
            bridge=SimpleNamespace(artifact_sha256=bridge_sha),
            input_metadata={"model_state_sha256": target_state},
        )


def test_anchor_active_interface_rehydrates_exact_producer_encoder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compiler_key = "1" * 64
    semantic_key = "2" * 64
    parent = "3" * 64
    bridge_sha = "4" * 64
    final_norm_sha = "5" * 64
    source = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, -1.0, 1.0], [0.5, 4.0, -2.0]],
        dtype=torch.bfloat16,
    )
    source64 = source.to(torch.float64)
    encoder = torch.tensor(
        [[0.2, 0.1], [0.3, -0.4], [0.5, 0.2]],
        dtype=torch.float64,
    )
    base_target = torch.tensor(
        [[0.5, 1.0], [0.5, -1.0], [0.5, 0.25]],
        dtype=torch.float64,
    )
    # Anchor repair intentionally destroys the constant first column.  An
    # inverse of the saved table is therefore invalid, while the producer map
    # remains exactly the base compiler's map.
    active_target = base_target + torch.tensor(
        [[0.125, 0.0], [0.0, 0.25], [-0.125, 0.0]],
        dtype=torch.float64,
    )
    base_report = _seal(
        {
            "source_table_sha256": postfold._interface_tensor_sha256(source64),
            "target_weight_sha256": postfold._interface_tensor_sha256(base_target),
            "producer_encoder_sha256": postfold._interface_tensor_sha256(encoder),
        },
        "report_sha256",
    )
    anchor_report = _seal(
        {
            "base_interface_report_sha256": base_report["report_sha256"],
            "source_tied_table_sha256": postfold._interface_tensor_sha256(source64),
            "target_weight_sha256": postfold._interface_tensor_sha256(active_target),
            "producer_encoder_sha256": postfold._interface_tensor_sha256(encoder),
            "selected_candidate_is_base_control": False,
            "active_anchor_residual_selected": True,
        },
        "report_sha256",
    )
    interface = postfold.TiedCausalInterfaceFold(
        target_weight=active_target,
        producer_encoder=encoder,
        report=anchor_report,
    )
    base_interface = postfold.TiedCausalInterfaceFold(
        target_weight=base_target,
        producer_encoder=encoder,
        report=base_report,
    )
    model = _TiedModel(active_target.to(torch.bfloat16))
    model.config = SimpleNamespace(hidden_size=2)

    store = tmp_path / "catalog-store"
    artifact = store / "artifact"
    artifact.mkdir(parents=True)
    compiler_cache = tmp_path / "compiler-cache"
    compiler_cache.mkdir()
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text("[]", encoding="ascii")
    trajectory_sha = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    preparation_path = tmp_path / "catalog-receipt.json"
    preparation_body = {
        "catalog_receipt_json": str(preparation_path.resolve()),
        "catalog_store_directory": str(store.resolve()),
        "trace_cache_directory": str(compiler_cache.resolve()),
        "artifact_path": str(artifact.resolve()),
        "cache_key": compiler_key,
        "binding_sha256": "6" * 64,
        "artifact_manifest_sha256": "7" * 64,
        "artifact_catalog_sha256": "8" * 64,
        "selective_catalog_sha256": "9" * 64,
        "selective_source_sha256": "a" * 64,
        "selective_provider_sha256": "b" * 64,
    }
    preparation = _seal(preparation_body, "preparation_receipt_sha256")
    preparation_path.write_bytes(
        (
            json.dumps(
                preparation,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )
    lineage = {
        "base_interface_report": base_report,
        "causal_anchor_report": anchor_report,
        "base_interface_report_sha256": base_report["report_sha256"],
        "causal_anchor_report_sha256": anchor_report["report_sha256"],
        "active_interface_report_sha256": anchor_report["report_sha256"],
        "active_interface_is_anchor_repaired": True,
        "active_interface_is_base_control": False,
        "producer_encoder_fixed_by_anchor_selection": True,
    }
    bundle = _seal(
        {
            "interface_report_sha256": anchor_report["report_sha256"],
            "base_interface_report_sha256": base_report["report_sha256"],
            "causal_anchor_report_sha256": anchor_report["report_sha256"],
            "interface_lineage": lineage,
        },
        "bundle_sha256",
    )
    compiled = _seal(
        {
            "selective_catalog_sha256": preparation["selective_catalog_sha256"],
            "selective_source_sha256": preparation["selective_source_sha256"],
            "selective_provider_sha256": preparation["selective_provider_sha256"],
            "final_norm_authentication_sha256": final_norm_sha,
            "boundary_bundle_sha256": bundle["bundle_sha256"],
        },
        "result_sha256",
    )
    install = _seal(
        {
            "interface_report_sha256": anchor_report["report_sha256"],
            "target_weight_sha256": postfold.cpcf_model_compiler._tensor_sha256(
                model.weight.detach()
            ),
        },
        "report_sha256",
    )
    inputs = {
        "cache_key": compiler_key,
        "parent_fingerprint": parent,
        "bridge_artifact_sha256": bridge_sha,
        "trace_cache_directory": str(compiler_cache.resolve()),
        "trajectory_corpus": str(trajectory.resolve()),
        "trajectory_corpus_sha256": trajectory_sha,
        "trace_adapter_sha256": "c" * 64,
        "selective_catalog_preparation_receipt_sha256": preparation[
            "preparation_receipt_sha256"
        ],
        "selective_catalog_binding_sha256": preparation["binding_sha256"],
        "selective_catalog_artifact_manifest_sha256": preparation[
            "artifact_manifest_sha256"
        ],
        "selective_catalog_artifact_sha256": preparation["artifact_catalog_sha256"],
    }
    diagnostic = {
        "receipt_sha256": "d" * 64,
        "inputs": inputs,
        "request": {
            "selective_catalog_store_directory": str(store.resolve()),
        },
        "selective_catalog_artifact": preparation,
        "compiled_model_fold": compiled,
        "boundary_bundle": bundle,
        "interface_install": install,
        "compiled_candidate_consumer_seam": {
            "cache_sha256": compiler_key,
            "parent_sha256": parent,
            "bridge_sha256": bridge_sha,
        },
    }

    class Provider:
        catalog = SimpleNamespace(
            catalog_sha256=preparation["selective_catalog_sha256"],
            source_sha256=preparation["selective_source_sha256"],
            provider_sha256=preparation["selective_provider_sha256"],
        )

        def load_final_norm_payload(self):
            return object()

        def authenticate_final_norm_payload(self, _payload):
            return final_norm_sha

        def release_final_norm_payload(self, _payload):
            return None

        def close(self):
            return None

    class BoundaryCompiler:
        def __init__(self, *_args, **_kwargs):
            self.base_interface = base_interface
            self.anchor_fold = SimpleNamespace(report=anchor_report)

        def fit_interface(self, *_args, **_kwargs):
            return interface

    monkeypatch.setattr(
        postfold,
        "_authenticated_c61_diagnostic",
        lambda *_a, **_k: (MappingProxyType(diagnostic), "e" * 64),
    )
    monkeypatch.setattr(
        postfold,
        "open_lfm2_circuit_trace_cache",
        lambda *_a, **_k: SimpleNamespace(
            cache_key=compiler_key,
            parent_fingerprint=parent,
        ),
    )
    monkeypatch.setattr(postfold.cpcf_resources, "_read_trajectory_rows", lambda _p: ())
    monkeypatch.setattr(
        postfold.cpcf_resources,
        "_build_trace_adapter",
        lambda **_k: SimpleNamespace(evidence_sha256="c" * 64),
    )
    monkeypatch.setattr(
        postfold,
        "open_lfm2_cpcf_selective_evidence_provider_from_catalog",
        lambda *_a, **_k: Provider(),
    )
    monkeypatch.setattr(
        postfold.cpcf_v5_compiler,
        "build_lfm2_cpcf_v5_target_phase_shells",
        lambda *_a, **_k: (),
    )
    monkeypatch.setattr(
        postfold.cpcf_v5_compiler,
        "materialize_lfm2_cpcf_v5_final_norm_phases",
        lambda *_a, **_k: (),
    )
    monkeypatch.setattr(
        postfold.cpcf_v5_compiler,
        "_SelectiveBoundaryCompiler",
        BoundaryCompiler,
    )
    monkeypatch.setattr(
        postfold.cpcf_model_fold, "_final_norm_epsilon", lambda _m: 1e-5
    )

    rehydrated, receipt = postfold.rehydrate_lfm2_c61_tied_interface(
        model,
        source,
        semantic_cache=SimpleNamespace(
            cache_key=semantic_key,
            parent_fingerprint=parent,
        ),
        bridge=SimpleNamespace(artifact_sha256=bridge_sha),
        target_tokenizer=object(),
        compiler_diagnostic_json="unused.json",
        expected_compiler_diagnostic_sha256=diagnostic["receipt_sha256"],
    )

    torch.testing.assert_close(rehydrated.producer_encoder, encoder, rtol=0, atol=0)
    assert receipt["active_interface_is_anchor_repaired"] is True
    assert receipt["producer_encoder_recovered_from_repaired_target_table"] is False
    assert receipt["compiler_cache_key"] == compiler_key
    assert receipt["semantic_evidence_cache_key"] == semantic_key


def test_attention_installation_receipt_requires_a_real_layer_local_change() -> None:
    receipt = postfold.seal_lfm2_semantic_attention_installation_receipt(
        layer_index=2,
        transport_evidence_sha256=("1" * 64, "2" * 64),
        model_state_before_sha256="3" * 64,
        model_state_after_sha256="4" * 64,
        changed_state_keys=("model.layers.2.self_attn.q_proj.weight",),
        parameter_element_count=57_405_368,
    )

    assert receipt["installed"] is True
    assert receipt["parameter_count_changed"] is False
    assert receipt["operation_audit"]["training_steps"] == 0
    body = dict(receipt)
    seal = body.pop("result_sha256")
    assert seal == canonical_sha256(body)


def _normalized_decoder_fixture(
    parent_rows: tuple[torch.Tensor, ...],
    chart: torch.Tensor,
    *,
    parent_epsilon: float,
    target_epsilon: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    parent_gamma = torch.tensor(
        [0.75, 1.25, 1.5, 0.5],
        dtype=torch.float64,
    )
    target_gamma = torch.tensor([1.125, 0.625], dtype=torch.float64)
    child_rows = tuple(
        (
            (row @ chart)
            * torch.rsqrt(
                (row @ chart).square().mean(dim=-1, keepdim=True) + target_epsilon
            )
        )
        * target_gamma
        for row in parent_rows
    )
    decoder, receipt = postfold._fit_normalized_operator_input_decoder(
        parent_raw_rows=parent_rows,
        child_normalized_rows=child_rows,
        parent_operator_norm_weight=parent_gamma,
        target_operator_norm_weight=target_gamma,
        parent_norm_epsilon=parent_epsilon,
        target_norm_epsilon=target_epsilon,
        residual_chart=chart,
        residual_chart_sha256=postfold._tensor_sha256(chart),
        fit_trajectory_count=len(parent_rows) - 1,
        ridge=1e-12,
    )
    return decoder, dict(receipt)


def test_normalized_decoder_width_change_uses_fit_covariance_scale_and_epsilons() -> (
    None
):
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    parent_rows = (
        torch.tensor([[1.0, 2.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[2.0, -1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[-1.0, -2.0, 0.0, 0.0]], dtype=torch.float64),
    )
    parent_epsilon = 1e-5
    target_epsilon = 3e-5
    decoder, receipt = _normalized_decoder_fixture(
        parent_rows,
        chart,
        parent_epsilon=parent_epsilon,
        target_epsilon=target_epsilon,
    )
    source_energy = parent_rows[0].square().mean()
    projected_energy = (parent_rows[0] @ chart).square().mean()
    expected_scale = torch.sqrt(
        (projected_energy + target_epsilon) / (source_energy + parent_epsilon)
    )
    scale = receipt["structural_covariance_scale"]

    assert scale["trajectory_balanced"] is True
    assert scale["actual_child_rows_used_for_scale"] == 0
    assert scale["heldout_rows_used_for_scale"] == 0
    assert scale["target_norm_epsilon"] == target_epsilon
    assert scale["covariance_scale"] == pytest.approx(
        float(expected_scale.item()),
        rel=1e-12,
    )
    assert scale["covariance_scale"] == pytest.approx(2.0**0.5, rel=2e-5)
    parent_gamma = torch.tensor([0.75, 1.25, 1.5, 0.5], dtype=torch.float64)
    target_gamma = torch.tensor([1.125, 0.625], dtype=torch.float64)
    expected_decoder = (
        expected_scale
        * torch.diag(target_gamma.reciprocal())
        @ torch.linalg.pinv(chart)
        @ torch.diag(parent_gamma)
    )
    torch.testing.assert_close(decoder, expected_decoder, rtol=1e-10, atol=1e-10)
    assert (
        receipt["heldout_rowwise_denominator_ratio_diagnostic"]["diagnostic_only"]
        is True
    )


def test_normalized_decoder_isotropic_projection_scale_is_one() -> None:
    chart = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    basis = tuple(
        torch.eye(4, dtype=torch.float64)[index : index + 1] for index in range(4)
    )
    _decoder, receipt = _normalized_decoder_fixture(
        (*basis, torch.ones((1, 4), dtype=torch.float64)),
        chart,
        parent_epsilon=2e-5,
        target_epsilon=2e-5,
    )

    assert receipt["structural_covariance_scale"]["covariance_scale"] == pytest.approx(
        1.0, rel=1e-12, abs=1e-12
    )


def test_normalized_decoder_nonorthogonal_chart_uses_pseudoinverse_prior() -> None:
    chart = torch.tensor(
        [[1.0, 0.0], [0.5, 1.0], [0.25, -0.5], [0.0, 0.75]],
        dtype=torch.float64,
    )
    coefficients = (
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        torch.tensor([[-1.0, -2.0]], dtype=torch.float64),
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
    )
    parent_rows = tuple(value @ chart.T for value in coefficients)
    decoder, receipt = _normalized_decoder_fixture(
        parent_rows,
        chart,
        parent_epsilon=1e-5,
        target_epsilon=1e-5,
    )
    scale = receipt["structural_covariance_scale"]["covariance_scale"]
    parent_gamma = torch.tensor([0.75, 1.25, 1.5, 0.5], dtype=torch.float64)
    target_gamma = torch.tensor([1.125, 0.625], dtype=torch.float64)
    expected = (
        scale
        * torch.diag(target_gamma.reciprocal())
        @ torch.linalg.pinv(chart)
        @ torch.diag(parent_gamma)
    )

    torch.testing.assert_close(decoder, expected, rtol=1e-9, atol=1e-9)
    assert receipt["fit_metrics"]["maximum_absolute_error"] < 1e-9

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="must change model state",
    ):
        postfold.seal_lfm2_semantic_attention_installation_receipt(
            layer_index=2,
            transport_evidence_sha256=("1" * 64,),
            model_state_before_sha256="3" * 64,
            model_state_after_sha256="3" * 64,
            changed_state_keys=("model.layers.2.self_attn.q_proj.weight",),
            parameter_element_count=10,
        )


def test_layer_stack_defers_final_w2_and_fuses_three_equal_mass_lanes() -> None:
    evidence = {
        split: _terminal_evidence(split, "final-mlp")
        for split in ("fit", "fit_check", "selection", "validation")
    }
    evidence_sha = {
        split: evidence[split].authentication_sha256 for split in sorted(evidence)
    }
    body = {
        "final_layer_mlp_w2_deferred_to_fused_terminal": True,
        "final_layer_mlp_w2_install_count": 0,
        "final_layer_mlp_evidence_sha256": evidence_sha,
    }
    installation = postfold.Lfm2SemanticLayerStackInstallation(
        receipt={**body, "result_sha256": canonical_sha256(body)},
        final_layer_mlp_evidence=evidence,
    )

    assert installation.receipt["final_layer_mlp_w2_install_count"] == 0
    merged = postfold._merge_terminal_evidence(
        {
            "final_layer_mlp": evidence["fit"],
            "lexical": _terminal_evidence("fit", "lexical"),
            "parent_endpoint": _terminal_evidence("fit", "parent"),
        }
    )
    assert merged.rows == 3
    assert float(merged.sample_weights.sum().item()) == pytest.approx(1.0)
    assert tuple(merged.sample_weights.tolist()) == pytest.approx((1.0 / 3.0,) * 3)


def test_lexical_coordinates_cover_all_87_inserted_rows() -> None:
    multiplicities = (1,) * 273 + (2,) * 66 + (3,) * 9 + (4,)
    phases = [
        SimpleNamespace(
            phase="prefill",
            target_full_prefix_length=10,
        )
    ]
    length = 10
    token = 100
    for index, multiplicity in enumerate(multiplicities):
        replacement = tuple(range(token, token + multiplicity))
        phases.append(
            SimpleNamespace(
                phase=f"decode_{index:03d}",
                target_replacement_token_ids=replacement,
                target_replaced_suffix_token_count=0,
                target_rewrite_start=length,
                target_full_prefix_length=length + multiplicity,
            )
        )
        length += multiplicity
        token += multiplicity
    semantic = SimpleNamespace(source_batch_index=7, phases=tuple(phases))

    rows, desired, row_ids, observed = (
        postfold._append_only_lexical_supervision_coordinates(semantic)
    )

    assert len(observed) == 349
    assert len(rows) == len(desired) == len(row_ids) == 87
    assert length - 10 == 436
    assert max(observed) == 4
    assert {value: observed.count(value) for value in set(observed)} == {
        1: 273,
        2: 66,
        3: 9,
        4: 1,
    }
    assert len(set(row_ids)) == 87


def test_lexical_teacher_uses_best_finite_witness_when_exact_top1_is_impossible() -> (
    None
):
    tied = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.bfloat16,
    )

    teacher, report = postfold._lexical_teacher_rows(tied, (1, 2, 1))

    winners = (teacher.to(torch.float64) @ tied.to(torch.float64).T).argmax(dim=1)
    assert winners.tolist() == [0, 2, 0]
    assert torch.isfinite(teacher).all()
    assert report["all_teacher_rows_exact_target_top1"] is False
    assert report["exact_unique_desired_token_count"] == 1
    assert report["approximate_unique_desired_token_count"] == 1
    assert report["exact_desired_row_count"] == 1
    assert report["approximate_desired_row_count"] == 2
    assert report["missing_exact_top1_token_ids"] == [1]
    diagnostics = {
        value["target_token_id"]: value
        for value in report["selected_unique_token_diagnostics"]
    }
    assert diagnostics[1]["exact_target_top1"] is False
    assert diagnostics[1]["winner_token_id"] == 0
    assert diagnostics[1]["target_rank"] == 1
    assert diagnostics[1]["target_margin_over_best_competitor"] == 0.0
    body = dict(report)
    supplied = body.pop("teacher_sha256")
    assert supplied == canonical_sha256(body)


def test_lexical_teacher_exact_frontier_reports_exactness_truthfully() -> None:
    tied = torch.eye(3, dtype=torch.bfloat16)

    teacher, report = postfold._lexical_teacher_rows(tied, (2, 0, 1, 2))

    winners = (teacher.to(torch.float64) @ tied.to(torch.float64).T).argmax(dim=1)
    assert winners.tolist() == [2, 0, 1, 2]
    assert report["all_teacher_rows_exact_target_top1"] is True
    assert report["exact_unique_desired_token_count"] == 3
    assert report["approximate_unique_desired_token_count"] == 0
    assert report["exact_desired_row_count"] == 4
    assert report["approximate_desired_row_count"] == 0
    assert report["missing_exact_top1_token_ids"] == []


def test_semantic_alignment_wires_authenticated_selected_input_contract(
    monkeypatch,
) -> None:
    reference = SimpleNamespace(
        index=0,
        input_sha256="1" * 64,
        ordered_tensor_names=(
            "decode.000.forced_token_ids",
            "input.attention_mask",
            "input.forced_decode_ids",
            "input.input_ids",
        ),
        record={"capture": {"decode_steps": 1}},
    )
    payload = object()
    session_receipt = {"session_sha256": "2" * 64}
    released = []

    class Session:
        batch_references = (reference,)

        def as_dict(self):
            return session_receipt

        def load_selected_tensors(self, index, names):
            assert index == 0
            assert names == reference.ordered_tensor_names
            return payload

        def release_selected_tensors(self, value):
            released.append(value)

    cache = SimpleNamespace(
        cache_key="3" * 64,
        batch_count=1,
        manifest={"manifest_sha256": "4" * 64},
    )
    selected_input = object()
    semantic = SimpleNamespace(
        source_prompt_token_ids=(1, 2),
        phase_order=("prefill", "decode.000"),
    )
    alignment = object()
    calls = []
    tokenizer_receipt = {"receipt_sha256": "7" * 64}

    class TokenizerSession:
        donor_tokenizer_semantics_sha256 = "5" * 64
        target_tokenizer_semantics_sha256 = "6" * 64

        def __enter__(self):
            calls.append("tokenizer-open")
            return self

        def __exit__(self, *_args):
            calls.append("tokenizer-close")
            return None

        def as_dict(self):
            return tokenizer_receipt

    tokenizer_session = TokenizerSession()

    class Bridge:
        def authenticated_live_session(
            self,
            donor,
            target,
            *,
            selective_session_sha256,
        ):
            assert donor == "donor" and target == "target"
            assert selective_session_sha256 == session_receipt["session_sha256"]
            return tokenizer_session

    def build_selected(observed_reference, observed_payload, receipt, **kwargs):
        calls.append("selected")
        assert observed_reference is reference
        assert observed_payload is payload
        assert receipt is session_receipt
        assert kwargs == {"expected_cache_key": cache.cache_key}
        return selected_input

    def build_semantic(source, manifest, donor, target, **kwargs):
        calls.append("semantic")
        assert source is selected_input
        assert manifest is cache.manifest
        assert donor == "donor" and target == "target"
        assert kwargs["expected_source_input_sha256"] == reference.input_sha256
        assert kwargs["tokenizer_session"] is tokenizer_session
        return semantic

    def build_alignment(observed, donor, target, **kwargs):
        calls.append("alignment")
        assert observed is semantic
        assert donor == "donor" and target == "target"
        assert kwargs["parent_sampled_suffix_tokens"] == {
            "prefill": 2,
            "decode.000": 1,
        }
        assert kwargs["tokenizer_session"] is tokenizer_session
        return alignment

    monkeypatch.setattr(
        postfold,
        "build_lfm2_semantic_span_selected_input_evidence",
        build_selected,
    )
    monkeypatch.setattr(
        postfold,
        "build_lfm2_semantic_span_phase_bridge",
        build_semantic,
    )
    monkeypatch.setattr(
        postfold,
        "build_lfm2_semantic_span_cpcf_alignment",
        build_alignment,
    )

    observed_alignments, observed_receipt = postfold._semantic_alignments_from_session(
        cache=cache,
        session=Session(),
        bridge=Bridge(),
        donor_tokenizer="donor",
        target_tokenizer="target",
    )
    assert observed_alignments == (alignment,)
    assert dict(observed_receipt) == tokenizer_receipt
    assert calls == [
        "tokenizer-open",
        "selected",
        "semantic",
        "alignment",
        "tokenizer-close",
    ]
    assert released == [payload]


def test_attention_stage_fails_closed_with_exact_missing_compiler(
    monkeypatch,
) -> None:
    class FakeCache:
        cache_key = "a" * 64
        parent_fingerprint = "b" * 64
        batch_count = 1

        def selective_authenticated_session(self):
            class Session:
                batch_references = ()

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def verify(self):
                    return self

                def as_dict(self):
                    return {"session_sha256": "c" * 64}

            return Session()

    class FakeBridge:
        artifact_sha256 = "d" * 64

        def verify_live_tokenizers(self, _donor, _target):
            return self

    alignment = SimpleNamespace(
        alignment_sha256="e" * 64,
        to_evidence_dict=lambda: {
            "alignment_sha256": "e" * 64,
        },
    )
    target = SimpleNamespace(_neural_forge_quarantine_tokenizer=object())
    monkeypatch.setattr(postfold, "Lfm2CircuitTraceCache", FakeCache)
    monkeypatch.setattr(postfold, "StrictTokenizerBridgeArtifact", FakeBridge)
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _value: "f" * 64,
    )
    monkeypatch.setattr(
        postfold,
        "_attention_layer_indices",
        lambda _cache, _model: (2,),
    )
    monkeypatch.setattr(
        postfold,
        "_target_rope",
        lambda _model: (torch.tensor([1.0], dtype=torch.float64), 1.0),
    )
    monkeypatch.setattr(
        postfold,
        "_semantic_alignments_from_session",
        lambda **_kwargs: ((alignment,), {"receipt_sha256": "8" * 64}),
    )
    monkeypatch.setattr(
        postfold,
        "audit_lfm2_semantic_span_cpcf_refinement",
        lambda *_args, **_kwargs: {
            "probability_preserving_attention_transport_go": True,
            "blocking_reasons": [],
            "report_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(postfold, "_resolve_attention_installer", lambda _value: None)
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="9" * 64),
    )
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 57_405_368)
    monkeypatch.setattr(
        postfold,
        "load_lfm2_parent_qkvo_inventory_from_local_safetensors",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldBlockedError) as captured:
        postfold.compile_install_lfm2_semantic_attention_stage(
            cache=FakeCache(),
            bridge=FakeBridge(),
            target_model=target,
            interface=object(),
            compiler_lineage={
                2: SimpleNamespace(source_inventory=object()),
            },
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
        )

    report = dict(captured.value.report)
    assert report["status"] == "blocked-no-checkpoint-published"
    assert report["terminal_refold_called"] is False
    assert report["checkpoint_save_called"] is False
    assert report["missing_callable"] == (
        "neural_forge.hf_lfm2_semantic_attention_refold_adapter."
        "compile_install_lfm2_semantic_attention_postfold"
    )
    assert report["operation_audit"]["training_steps"] == 0
    body = dict(report)
    seal = body.pop("failure_sha256")
    assert seal == canonical_sha256(body)


def _write_minimal_postfold_artifact(
    directory: Path,
    *,
    extra_name: str | None = None,
) -> None:
    config = directory / "config.json"
    weight = directory / "model.safetensors"
    tokenizer = directory / "tokenizer.json"
    config.write_text("{}", encoding="utf-8")
    weight.write_bytes(b"test")
    tokenizer.write_text("{}", encoding="utf-8")
    body = {
        "schema": postfold.LFM2_CPCF_V5_QUARANTINE_ARTIFACT_SCHEMA,
        "dense_safetensor_files": [
            {
                "path": weight.name,
                "size_bytes": weight.stat().st_size,
                "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            }
        ],
        "tokenizer_package_files": [
            {
                "path": tokenizer.name,
                "bytes": tokenizer.stat().st_size,
                "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            }
        ],
        "dense_safetensors_only": True,
        "target_tokenizer_saved_with_child": True,
        "local_reload_only": True,
        "network_access": False,
        "hub_loader_used": False,
    }
    value = {**body, "artifact_sha256": canonical_sha256(body)}
    (directory / postfold.LFM2_CPCF_V5_ARTIFACT_METADATA).write_text(
        json.dumps(value),
        encoding="utf-8",
    )
    if extra_name is not None:
        (directory / extra_name).write_bytes(b"unlisted runtime material")


def test_checkpoint_publication_is_staged_then_reloaded_after_atomic_move(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _TiedModel(torch.full((4, 2), 0.5, dtype=torch.bfloat16))
    tokenizer = object()
    events = []
    model_state = "1" * 64
    inventory_sha = "2" * 64
    tokenizer_sha = "3" * 64

    class Saver:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _model, directory):
            events.append(("save", directory.name))
            _write_minimal_postfold_artifact(directory)

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, directory):
            events.append(("load", directory.name))
            return model

    monkeypatch.setattr(postfold, "Lfm2CpcfV5QuarantineSaver", Saver)
    monkeypatch.setattr(postfold, "Lfm2CpcfV5StrictLocalLoader", Loader)
    monkeypatch.setattr(postfold, "fingerprint_model_state", lambda _model: model_state)
    monkeypatch.setattr(
        postfold,
        "independent_parameter_inventory",
        lambda _model: {"inventory_sha256": inventory_sha},
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: tokenizer_sha,
    )
    monkeypatch.setattr(postfold, "_ffn_width", lambda _model: 7)
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 100)
    output = tmp_path / "standalone-postfold"

    reloaded, receipt = postfold.publish_lfm2_semantic_postfold_checkpoint(
        target_model=model,
        target_tokenizer=tokenizer,
        output_directory=output,
        context_binding_sha256="4" * 64,
        device="cpu",
    )

    assert reloaded is model
    assert output.is_dir()
    assert events[0][0] == "save"
    assert events[1][0] == "load"
    assert events[1][1].startswith(".semantic-postfold-")
    assert events[2] == ("load", "standalone-postfold")
    assert receipt["staged_reload_verified"] is True
    assert receipt["published_reload_verified"] is True
    assert receipt["complete_artifact_tree_hashed"] is True
    assert receipt["forbidden_named_or_referenced_sidecars_present"] is False
    assert receipt["standalone_directory_purity_authenticated"] is True
    assert receipt["artifact_file_count"] == 4
    assert receipt["artifact_tree_sha256"] == canonical_sha256(receipt["artifact_tree"])
    assert receipt["canonical_typed_artifact_inventory_authenticated"] is True
    assert receipt["staged_artifact_tree_purity_verified_before_publication"] is True
    assert receipt["unallowlisted_artifact_files_present"] is False
    assert receipt["artifact_inventory_sha256"] == canonical_sha256(
        receipt["artifact_inventory"]
    )
    assert {value["artifact_role"] for value in receipt["artifact_inventory"]} == {
        "artifact_metadata",
        "model_config",
        "model_tensor",
        "tokenizer_resource",
    }
    assert receipt["operation_audit"]["optimizer_steps"] == 0


@pytest.mark.parametrize("extra_name", ("auxiliary.dat", "payload.pkl"))
def test_checkpoint_publication_rejects_unallowlisted_sidecar_and_removes_output(
    tmp_path: Path,
    monkeypatch,
    extra_name: str,
) -> None:
    model = _TiedModel(torch.full((4, 2), 0.5, dtype=torch.bfloat16))

    class Saver:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _model, directory):
            _write_minimal_postfold_artifact(directory, extra_name=extra_name)

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _directory):
            return model

    monkeypatch.setattr(postfold, "Lfm2CpcfV5QuarantineSaver", Saver)
    monkeypatch.setattr(postfold, "Lfm2CpcfV5StrictLocalLoader", Loader)
    monkeypatch.setattr(postfold, "fingerprint_model_state", lambda _model: "1" * 64)
    monkeypatch.setattr(
        postfold,
        "independent_parameter_inventory",
        lambda _model: {"inventory_sha256": "2" * 64},
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: "3" * 64,
    )
    monkeypatch.setattr(postfold, "_ffn_width", lambda _model: 7)
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 100)
    output = tmp_path / "forbidden-sidecar-publication"

    with pytest.raises(
        DynamicFoldArtifactError,
        match="canonical typed inventory",
    ):
        postfold.publish_lfm2_semantic_postfold_checkpoint(
            target_model=model,
            target_tokenizer=object(),
            output_directory=output,
            context_binding_sha256="4" * 64,
            device="cpu",
        )

    assert not output.exists()


def test_failed_post_move_reload_removes_only_new_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _TiedModel(torch.full((4, 2), 0.5, dtype=torch.bfloat16))

    class Saver:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _model, directory):
            _write_minimal_postfold_artifact(directory)

    class Loader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, directory):
            if directory.name == "failed-publication":
                raise RuntimeError("synthetic final reload failure")
            return model

    monkeypatch.setattr(postfold, "Lfm2CpcfV5QuarantineSaver", Saver)
    monkeypatch.setattr(postfold, "Lfm2CpcfV5StrictLocalLoader", Loader)
    monkeypatch.setattr(postfold, "fingerprint_model_state", lambda _model: "1" * 64)
    monkeypatch.setattr(
        postfold,
        "independent_parameter_inventory",
        lambda _model: {"inventory_sha256": "2" * 64},
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: "3" * 64,
    )
    monkeypatch.setattr(postfold, "_ffn_width", lambda _model: 7)
    monkeypatch.setattr(postfold, "_parameter_count", lambda _model: 100)
    output = tmp_path / "failed-publication"

    with pytest.raises(RuntimeError, match="synthetic final reload failure"):
        postfold.publish_lfm2_semantic_postfold_checkpoint(
            target_model=model,
            target_tokenizer=object(),
            output_directory=output,
            context_binding_sha256="4" * 64,
            device="cpu",
        )

    assert not output.exists()


def test_complete_layer_stack_fails_closed_on_conv_and_mlp_gaps(
    monkeypatch,
) -> None:
    topology = [
        {
            "contract": {
                "kind": "full_attention" if index in {2, 5, 8, 11} else "short_conv"
            }
        }
        for index in range(14)
    ]
    cache = SimpleNamespace(manifest={"layer_topology": {"layers": topology}})
    monkeypatch.setattr(
        postfold,
        "_resolve_layer_stack_installer",
        lambda _value: None,
    )

    with pytest.raises(postfold.Lfm2SemanticPostfoldBlockedError) as captured:
        postfold.compile_install_lfm2_semantic_layer_stack_stage(
            cache=cache,
            bridge=object(),
            target_model=object(),
            interface=object(),
            compiler_lineage={},
            parent_model_directory="unused",
            donor_tokenizer=object(),
            target_tokenizer=object(),
        )

    report = dict(captured.value.report)
    assert report["required_causal_layer_count"] == 14
    assert report["required_attention_operator_count"] == 4
    assert report["required_short_conv_operator_count"] == 10
    assert "long-horizon-short-conv-refold-missing" in report["blocking_reasons"]
    assert "replay-fresh-per-layer-mlp-refold-missing" in report["blocking_reasons"]
    assert report["fused_terminal_refold_called"] is False
    assert report["checkpoint_save_called"] is False
    body = dict(report)
    seal = body.pop("failure_sha256")
    assert seal == canonical_sha256(body)


def _minimal_postfold_run_arguments(tmp_path: Path) -> dict[str, object]:
    return {
        "input_checkpoint_directory": tmp_path / "input",
        "expected_input_artifact_sha256": "1" * 64,
        "output_checkpoint_directory": tmp_path / "output",
        "cache_directory": tmp_path / "cache",
        "expected_cache_key": "2" * 64,
        "bridge_directory": tmp_path / "bridge",
        "expected_bridge_artifact_sha256": "3" * 64,
        "donor_tokenizer": object(),
        "target_tokenizer": object(),
        "parent_model_directory": tmp_path / "parent",
        "compiler_diagnostic_json": tmp_path / "compiler.json",
        "expected_compiler_diagnostic_sha256": "4" * 64,
        "device": "cpu",
    }


def _cached_fused_terminal_evidence() -> dict[str, object]:
    cache_key = "1" * 64
    manifest_sha256 = "2" * 64
    completion_sha256 = "3" * 64
    record_sha256 = "4" * 64
    record_file_sha256 = "5" * 64

    def auth(body: dict[str, object], seal: str) -> dict[str, object]:
        return {**body, seal: canonical_sha256(body)}

    phases = []
    for phase, rows, disagreements, forced_sha in (
        ("cached_prefill", 2, 0, None),
        ("decode.000", 1, 1, "a" * 64),
    ):
        phase_body = {
            "phase": phase,
            "cached_logits_sha256": "6" * 64,
            "full_prefix_logits_sha256": "7" * 64,
            "row_count": rows,
            "maximum_absolute_error": 0.125,
            "numerically_close": True,
            "top1_exact_agreement": disagreements == 0,
            "top1_disagreement_count": disagreements,
            "all_top1_disagreements_within_numeric_error_envelope": True,
            "forced_token_sha256": forced_sha,
        }
        phases.append(auth(phase_body, "authentication_sha256"))
    raw_body = {
        "batch_index": 0,
        "decode_trace_mode": "cached_verified",
        "raw_cached_full_prefix_pair_count": 2,
        "raw_logit_rows_verified": 3,
        "phase_verification": phases,
        "phase_authentication_sha256": [
            value["authentication_sha256"] for value in phases
        ],
        "top1_disagreement_count": 1,
        "all_logits_numerically_close": True,
        "all_top1_disagreements_within_numeric_error_envelope": True,
        "top1_exactness_is_diagnostic_only": True,
    }
    raw = auth(raw_body, "authentication_sha256")
    batch_compatibility = {
        "batch_index": 0,
        "decode_trace_mode": "cached_verified",
        "verification_phase_count": 2,
        "verification_phase_order": ["cached_prefill", "decode.000"],
        "top1_disagreement_count": 1,
    }
    compatibility_body = {
        "schema": postfold.LFM2_LONG_HORIZON_TERMINAL_CACHE_COMPATIBILITY_SCHEMA,
        "cache_key": cache_key,
        "manifest_sha256": manifest_sha256,
        "completion_sha256": completion_sha256,
        "batch_count": 1,
        "decode_trace_mode": "cached_verified",
        "ordered_batch_record_sha256": [record_sha256],
        "ordered_batch_record_file_sha256": [record_file_sha256],
        "batch_compatibility": [batch_compatibility],
        "all_cached_logits_numerically_verified": True,
        "cached_top1_disagreement_count": 1,
        "terminal_trajectory_compatibility_verified": True,
        "forced_decode_tokens_are_caller_supplied": True,
        "top1_exactness_is_diagnostic_only": True,
        "top1_exactness_is_publication_gate": False,
        "all_top1_disagreements_numeric_envelope_verified": True,
        "full_prefix_recache_required": False,
        "preflight_io_contract": {
            "trace_file_sha256_passes": 0,
            "trace_payload_bytes_read": 0,
            "trace_tensor_payloads_materialized": 0,
        },
    }
    compatibility = auth(compatibility_body, "authentication_sha256")
    session_body = {
        "cache_key": cache_key,
        "manifest_sha256": manifest_sha256,
        "completion_sha256": completion_sha256,
        "ordered_authenticated_batches": [
            {
                "batch_index": 0,
                "record_sha256": record_sha256,
                "record_file_sha256": record_file_sha256,
                "tensor_count": 2,
                "logical_bytes": 128,
            }
        ],
    }
    session = auth(session_body, "session_sha256")
    cache_authentication = {
        "session": session,
        "selected_payloads": [
            {
                "batch_index": 0,
                "authenticated_tensor_count": 2,
                "authenticated_logical_bytes": 128,
                "selective_payload": {
                    "selected_tensor_count": 2,
                    "selected_logical_bytes": 128,
                },
            }
        ],
    }
    payload_load_audit = postfold._summarize_terminal_cache_payload_loads(
        cache_authentication
    )
    binding_body = {
        "compatibility_authentication_sha256": compatibility["authentication_sha256"],
        "payload_session_sha256": session["session_sha256"],
        "manifest_completion_and_ordered_records_exact": True,
        "large_trace_file_sha256_authenticated_once_by_consuming_session": True,
    }
    binding = auth(binding_body, "authentication_sha256")
    io_body = {
        "metadata_preflight_trace_file_sha256_passes": 0,
        "metadata_preflight_trace_payload_bytes_read": 0,
        "consuming_trace_file_sha256_passes_per_batch": 1,
        "consuming_trace_file_sha256_passes_total": 1,
        "same_consuming_session_reused_by_semantic_collector": True,
        "same_consuming_session_reused_from_layer_stack": True,
        "fused_terminal_selective_session_constructions": 0,
        "end_to_end_selective_session_constructions": 1,
        "additional_full_trace_file_sha256_passes_performed_by_fused_terminal": 0,
        "selected_tensor_payload_loads_total": 1,
        "selected_tensor_payload_logical_bytes_total": 128,
        "full_tensor_inventory_payload_loads_per_batch": 1,
        "full_tensor_inventory_payload_loads_total": 1,
        "full_tensor_inventory_payload_logical_bytes_total": 128,
    }
    io_audit = auth(io_body, "result_sha256")
    body = {
        "cache_key": cache_key,
        "terminal_cache_compatibility": compatibility,
        "terminal_cache_compatibility_sha256": compatibility["authentication_sha256"],
        "terminal_cache_payload_binding": binding,
        "parent_cache_authentication": cache_authentication,
        "terminal_cache_tensor_payload_load_audit": payload_load_audit,
        "cached_full_prefix_payload_verification": [raw],
        "cached_full_prefix_payload_verification_sha256": [
            raw["authentication_sha256"]
        ],
        "terminal_cache_io_audit": io_audit,
    }
    return auth(body, "result_sha256")


def _reseal(value: dict[str, object], seal: str) -> None:
    value.pop(seal, None)
    value[seal] = canonical_sha256(value)


def test_fused_cache_evidence_rejects_fully_resealed_session_and_phase_drift():
    valid = _cached_fused_terminal_evidence()
    verified = postfold._verify_sealed(
        valid,
        seal_name="result_sha256",
        label="synthetic fused terminal",
    )
    authentication = postfold._authenticate_fused_terminal_cache_evidence(verified)
    assert authentication["every_raw_cached_full_prefix_row_rederived"] is True
    assert authentication["raw_logit_rows_verified"] == 3
    assert authentication["top1_disagreement_count"] == 1

    session_drift = copy.deepcopy(valid)
    binding = session_drift["terminal_cache_payload_binding"]
    assert isinstance(binding, dict)
    binding["payload_session_sha256"] = "f" * 64
    _reseal(binding, "authentication_sha256")
    _reseal(session_drift, "result_sha256")
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="compatibility/session binding is invalid",
    ):
        postfold._authenticate_fused_terminal_cache_evidence(
            postfold._verify_sealed(
                session_drift,
                seal_name="result_sha256",
                label="resealed session drift",
            )
        )

    phase_drift = copy.deepcopy(valid)
    raw = phase_drift["cached_full_prefix_payload_verification"][0]
    phase = raw["phase_verification"][1]
    phase["numerically_close"] = False
    _reseal(phase, "authentication_sha256")
    raw["phase_authentication_sha256"][1] = phase["authentication_sha256"]
    _reseal(raw, "authentication_sha256")
    phase_drift["cached_full_prefix_payload_verification_sha256"][0] = raw[
        "authentication_sha256"
    ]
    _reseal(phase_drift, "result_sha256")
    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="phase receipt is inconsistent",
    ):
        postfold._authenticate_fused_terminal_cache_evidence(
            postfold._verify_sealed(
                phase_drift,
                seal_name="result_sha256",
                label="resealed phase drift",
            )
        )


@pytest.mark.parametrize(
    ("statistics_directory", "statistics_sha256"),
    (
        (Path("unused-statistics"), None),
        (None, "a" * 64),
    ),
)
def test_top_level_rejects_unpaired_interface_statistics_before_dispatch(
    tmp_path: Path,
    monkeypatch,
    statistics_directory: Path | None,
    statistics_sha256: str | None,
) -> None:
    def unexpected_dispatch(*_args, **_kwargs):
        pytest.fail("unpaired statistics reached ancestry or asset dispatch")

    monkeypatch.setattr(
        postfold,
        "_claims_semantic_foundation_schema",
        unexpected_dispatch,
    )

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="must be supplied together",
    ):
        postfold.run_lfm2_semantic_postfold(
            **_minimal_postfold_run_arguments(tmp_path),
            interface_statistics_directory=statistics_directory,
            expected_interface_statistics_sha256=statistics_sha256,
        )


def test_top_level_rejects_statistics_for_nonfoundation_before_output_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    diagnostic = tmp_path / "compiler.json"
    diagnostic.write_text("{}", encoding="ascii")

    def unexpected_output_dispatch(*_args, **_kwargs):
        pytest.fail("nonfoundation statistics reached output or asset dispatch")

    monkeypatch.setattr(postfold, "_new_output_path", unexpected_output_dispatch)

    with pytest.raises(
        postfold.Lfm2SemanticPostfoldError,
        match="only for semantic-foundation ancestry",
    ):
        postfold.run_lfm2_semantic_postfold(
            **_minimal_postfold_run_arguments(tmp_path),
            interface_statistics_directory=tmp_path / "statistics",
            expected_interface_statistics_sha256="a" * 64,
        )


@pytest.mark.parametrize("use_foundation_statistics", (False, True))
def test_top_level_orders_layer_stack_terminal_and_checkpoint(
    tmp_path: Path,
    monkeypatch,
    use_foundation_statistics: bool,
) -> None:
    events = []
    cache = SimpleNamespace(cache_key="1" * 64, parent_fingerprint="2" * 64)
    bridge = SimpleNamespace(artifact_sha256="3" * 64)
    model = object()
    tokenizer = object()
    parent_norm = object()
    source_table = torch.ones((2, 2), dtype=torch.bfloat16)
    interface = SimpleNamespace(report={"report_sha256": "6" * 64})
    self_dual_interface = SimpleNamespace(report={"report_sha256": "f" * 64})
    statistics_artifact_sha256 = "a" * 64
    statistics_manifest_sha256 = "b" * 64
    statistics_replay_sha256 = "c" * 64
    if use_foundation_statistics:
        rehydration_body = {
            "schema": (
                "neural-forge-lfm2-semantic-foundation-interface-rehydration-v2"
            ),
            "compiler_cache_key": "8" * 64,
            "semantic_evidence_cache_key": cache.cache_key,
            "interface_statistics_artifact_sha256": (statistics_artifact_sha256),
            "interface_statistics_manifest_file_sha256": (statistics_manifest_sha256),
            "interface_statistics_replay_sha256": statistics_replay_sha256,
            "compiler_cache_open_calls": 0,
            "trace_safetensor_open_calls": 0,
            "selective_provider_open_calls": 0,
        }
        interface_rehydration = MappingProxyType(
            {
                **rehydration_body,
                "result_sha256": canonical_sha256(rehydration_body),
            }
        )
    else:
        interface_rehydration = MappingProxyType(
            {
                "compiler_cache_key": "8" * 64,
                "semantic_evidence_cache_key": cache.cache_key,
                "result_sha256": "9" * 64,
            }
        )
    w1_sha = "1" * 64
    w3_sha = "2" * 64
    w2_sha = "3" * 64

    def sealed(body):
        return {**body, "result_sha256": canonical_sha256(body)}

    manifest_sha256 = "4" * 64
    completion_sha256 = "5" * 64
    record_sha256 = "6" * 64
    record_file_sha256 = "7" * 64
    session_batch = {
        "batch_index": 0,
        "record_sha256": record_sha256,
        "record_file_sha256": record_file_sha256,
        "tensor_count": 2,
        "logical_bytes": 128,
    }
    session_body = {
        "cache_key": cache.cache_key,
        "manifest_sha256": manifest_sha256,
        "completion_sha256": completion_sha256,
        "ordered_authenticated_batches": [session_batch],
    }
    session_receipt = MappingProxyType(
        {
            **session_body,
            "session_sha256": canonical_sha256(session_body),
        }
    )
    cache_authentication = {
        "session": dict(session_receipt),
        "selected_payloads": [
            {
                "batch_index": 0,
                "authenticated_tensor_count": 2,
                "authenticated_logical_bytes": 128,
                "selective_payload": {
                    "selected_tensor_count": 2,
                    "selected_logical_bytes": 128,
                },
            }
        ],
    }
    payload_load_audit = postfold._summarize_terminal_cache_payload_loads(
        cache_authentication
    )
    compatibility_batch = {
        "batch_index": 0,
        "decode_trace_mode": "full_prefix_canonical",
        "verification_phase_count": 0,
        "top1_disagreement_count": 0,
    }
    compatibility_body = {
        "schema": postfold.LFM2_LONG_HORIZON_TERMINAL_CACHE_COMPATIBILITY_SCHEMA,
        "cache_key": cache.cache_key,
        "manifest_sha256": manifest_sha256,
        "completion_sha256": completion_sha256,
        "batch_count": 1,
        "decode_trace_mode": "full_prefix_canonical",
        "ordered_batch_record_sha256": [record_sha256],
        "ordered_batch_record_file_sha256": [record_file_sha256],
        "batch_compatibility": [compatibility_batch],
        "all_cached_logits_numerically_verified": None,
        "cached_top1_disagreement_count": 0,
        "terminal_trajectory_compatibility_verified": True,
        "forced_decode_tokens_are_caller_supplied": True,
        "top1_exactness_is_diagnostic_only": True,
        "top1_exactness_is_publication_gate": False,
        "all_top1_disagreements_numeric_envelope_verified": True,
        "full_prefix_recache_required": False,
        "preflight_io_contract": {
            "trace_file_sha256_passes": 0,
            "trace_payload_bytes_read": 0,
            "trace_tensor_payloads_materialized": 0,
        },
    }
    terminal_cache_compatibility = MappingProxyType(
        {
            **compatibility_body,
            "authentication_sha256": canonical_sha256(compatibility_body),
        }
    )
    payload_binding_body = {
        "compatibility_authentication_sha256": terminal_cache_compatibility[
            "authentication_sha256"
        ],
        "payload_session_sha256": session_receipt["session_sha256"],
        "manifest_completion_and_ordered_records_exact": True,
        "large_trace_file_sha256_authenticated_once_by_consuming_session": True,
    }
    payload_binding = MappingProxyType(
        {
            **payload_binding_body,
            "authentication_sha256": canonical_sha256(payload_binding_body),
        }
    )
    raw_body = {
        "batch_index": 0,
        "decode_trace_mode": "full_prefix_canonical",
        "raw_cached_full_prefix_pair_count": 0,
        "raw_logit_rows_verified": 0,
        "top1_disagreement_count": 0,
        "all_logits_numerically_close": True,
        "all_top1_disagreements_within_numeric_error_envelope": True,
        "top1_exactness_is_diagnostic_only": True,
    }
    raw_receipt = MappingProxyType(
        {
            **raw_body,
            "authentication_sha256": canonical_sha256(raw_body),
        }
    )
    io_body = {
        "metadata_preflight_trace_file_sha256_passes": 0,
        "metadata_preflight_trace_payload_bytes_read": 0,
        "consuming_trace_file_sha256_passes_per_batch": 1,
        "consuming_trace_file_sha256_passes_total": 1,
        "same_consuming_session_reused_by_semantic_collector": True,
        "same_consuming_session_reused_from_layer_stack": True,
        "fused_terminal_selective_session_constructions": 0,
        "end_to_end_selective_session_constructions": 1,
        "additional_full_trace_file_sha256_passes_performed_by_fused_terminal": 0,
        "selected_tensor_payload_loads_total": 1,
        "selected_tensor_payload_logical_bytes_total": 128,
        "full_tensor_inventory_payload_loads_per_batch": 1,
        "full_tensor_inventory_payload_loads_total": 1,
        "full_tensor_inventory_payload_logical_bytes_total": 128,
    }
    terminal_cache_io = MappingProxyType(sealed(io_body))

    selective_session_calls = []

    class PipelineCacheSession:
        def __enter__(self):
            events.append("cache_session_open")
            return self

        def __exit__(self, *_args):
            events.append("cache_session_close")
            return None

        def verify(self):
            return self

        def as_dict(self):
            return session_receipt

    pipeline_cache_session = PipelineCacheSession()

    def selective_authenticated_session():
        selective_session_calls.append("constructed")
        return pipeline_cache_session

    cache.selective_authenticated_session = selective_authenticated_session

    gauge_receipt = MappingProxyType(
        sealed(
            {
                "schema": "neural-forge-lfm2-self-dual-tied-gauge-install-v1",
                "global_residual_gauge_frozen": True,
            }
        )
    )

    coverage_before_body = {
        "schema": "neural-forge-lfm2-dense-producer-chain-coverage-v1",
        "layer_index": 13,
        "producer_kind": "mlp",
        "deferred_roles": ["feed_forward.w2.weight"],
        "role_receipts": {
            "feed_forward.w1.weight": {"tensor_sha256": w1_sha},
            "feed_forward.w3.weight": {"tensor_sha256": w3_sha},
            "feed_forward.w2.weight": {"tensor_sha256": w2_sha},
        },
        "all_required_roles_structurally_nonzero": True,
        "blank_or_partially_blank_required_role_count": 0,
    }
    coverage_before = sealed(coverage_before_body)
    coverage_after_body = {
        **coverage_before_body,
        "deferred_roles": [],
    }
    coverage_after = sealed(coverage_after_body)
    published_coverage_body = {
        "schema": "neural-forge-dense-weight-structural-coverage-v1",
        "tensor_sha256": w2_sha,
        "structurally_nonzero": True,
    }
    published_coverage = sealed(published_coverage_body)
    heldout_diagnostic_body = {
        "incumbent": {"finite": True},
        "published": {"finite": True},
        "gates": {"diagnostic_threshold": False},
        "passed": False,
        "role": "diagnostic-only-after-frozen-terminal-publication-decision",
        "publication_gate_applied": False,
    }
    heldout_diagnostic = {
        **heldout_diagnostic_body,
        "diagnostic_sha256": canonical_sha256(heldout_diagnostic_body),
    }
    layer_body = {
        "complete_causal_layer_stack_adjudicated": True,
        "complete_causal_layer_stack_installed": False,
        "whole_stage_contains_meaningful_parameter_changes": True,
        "model_state_before_sha256": "b" * 64,
        "model_state_after_sha256": "d" * 64,
        "changed_state_keys": ["model.layers.0.mutable"],
        "attention_operator_decision_count": 6,
        "attention_operator_install_count": 5,
        "attention_operator_incumbent_noop_count": 1,
        "short_conv_operator_decision_count": 8,
        "short_conv_operator_install_count": 8,
        "short_conv_operator_incumbent_noop_count": 0,
        "replay_fresh_mlp_evidence_compile_count": 14,
        "nonterminal_mlp_writer_decision_count": 13,
        "nonterminal_mlp_writer_install_count": 12,
        "nonterminal_mlp_writer_incumbent_noop_count": 1,
        "final_mlp_producer_install_count": 0,
        "final_mlp_producer_incumbent_noop_count": 1,
        "attention_producer_structural_coverage_count": 6,
        "short_conv_producer_structural_coverage_count": 8,
        "nonterminal_mlp_producer_structural_coverage_count": 13,
        "final_mlp_producer_structural_coverage_count": 1,
        "installed_complete_producer_chain_count": 25,
        "authenticated_nonzero_folded_seed_retained_noop_count": 3,
        "blank_or_partially_blank_producer_chain_count": 0,
        "complete_nonzero_structural_producer_coverage": True,
        "all_producer_chains_installed_or_retained_nonzero_seed": True,
        "layer_installations": [
            {
                "mlp_producer_coverage_decision": {
                    "coverage_after_sha256": coverage_before["result_sha256"]
                }
            }
        ],
        "operator_parameter_mutation_count": 13,
        "mlp_parameter_mutation_count": 12,
        "cache_selective_session_sha256": session_receipt["session_sha256"],
    }
    layer_receipt = MappingProxyType(
        {**layer_body, "result_sha256": canonical_sha256(layer_body)}
    )
    layer_stack = SimpleNamespace(receipt=layer_receipt)
    terminal_installation = {
        "decision_count": 1,
        "installed": False,
        "install_count": 0,
        "mutation_count": 0,
        "incumbent_retention_count": 1,
        "exact_zero_copy_noop": True,
        "parameter_copy_calls": 0,
        "changed_state_keys": [],
        "parameter_topology_preserved": True,
        "parameter_identities_preserved": True,
        "parameter_storage_preserved": True,
        "gradient_state_preserved": True,
        "parameter_count_changed": False,
        "model_state_before_sha256": "d" * 64,
        "model_state_after_sha256": "d" * 64,
        "published_w2_sha256": w2_sha,
        "published_w2_structurally_nonzero": True,
        "published_w2_structural_coverage": published_coverage,
        "published_w2_structural_coverage_sha256": published_coverage["result_sha256"],
    }
    terminal_body = {
        "schema": postfold.LFM2_SEMANTIC_LEXICAL_COMPLETION_SCHEMA,
        "semantic_layer_stack_result_sha256": layer_receipt["result_sha256"],
        "accepted": True,
        "terminal_publication_decision_accepted": True,
        "terminal_nonincumbent_candidate_accepted": False,
        "terminal_incumbent_retained": True,
        "terminal_writer_decision_count": 1,
        "terminal_writer_install_count": 0,
        "terminal_writer_mutation_count": 0,
        "terminal_writer_incumbent_retention_count": 1,
        "installation": terminal_installation,
        "final_mlp_producer_coverage_before_terminal": coverage_before,
        "final_mlp_full_coverage_after_terminal": coverage_after,
        "complete_nonzero_final_mlp_structural_coverage": True,
        "cache_key": cache.cache_key,
        "terminal_cache_compatibility": dict(terminal_cache_compatibility),
        "terminal_cache_compatibility_sha256": terminal_cache_compatibility[
            "authentication_sha256"
        ],
        "terminal_cache_payload_binding": dict(payload_binding),
        "parent_cache_authentication": cache_authentication,
        "terminal_cache_tensor_payload_load_audit": payload_load_audit,
        "cached_full_prefix_payload_verification": [dict(raw_receipt)],
        "cached_full_prefix_payload_verification_sha256": [
            raw_receipt["authentication_sha256"]
        ],
        "terminal_cache_io_audit": dict(terminal_cache_io),
        "fused_heldout_diagnostics": {
            "selection": dict(heldout_diagnostic),
            "validation": dict(heldout_diagnostic),
        },
        "separate_heldout_lane_diagnostics": {
            lane: {
                "selection": dict(heldout_diagnostic),
                "validation": dict(heldout_diagnostic),
            }
            for lane in ("parent_endpoint", "lexical", "final_layer_mlp")
        },
        "terminal_heldout_metrics_diagnostic_only": True,
        "terminal_heldout_publication_gate_applied": False,
        "operation_audit": dict(postfold._ZERO_OPERATIONS),
    }
    terminal = MappingProxyType(
        {**terminal_body, "result_sha256": canonical_sha256(terminal_body)}
    )
    cache_open_calls = []

    def open_cache(*_args, **kwargs):
        cache_open_calls.append(kwargs.get("verify"))
        return cache

    monkeypatch.setattr(postfold, "open_lfm2_circuit_trace_cache", open_cache)
    monkeypatch.setattr(
        postfold,
        "preflight_lfm2_long_horizon_terminal_cache",
        lambda observed_cache: (
            events.append("metadata_preflight")
            or (
                terminal_cache_compatibility
                if observed_cache is cache
                else pytest.fail("preflight received the wrong semantic cache")
            )
        ),
    )
    monkeypatch.setattr(
        postfold,
        "load_strict_tokenizer_bridge_artifact",
        lambda *_a, **_k: bridge,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_cpcf_v5_standalone_child",
        lambda *_a, **_k: (
            model,
            tokenizer,
            MappingProxyType(
                {
                    "artifact_sha256": "4" * 64,
                    "model_state_sha256": "5" * 64,
                }
            ),
        ),
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_hf_tokenizer_semantics",
        lambda _tokenizer: "a" * 64,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_parent_final_rmsnorm_from_local_safetensors",
        lambda *_a, **_k: parent_norm,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_mapped_parent_tied_table_from_local_safetensors",
        lambda *_a, **_k: (
            source_table,
            MappingProxyType({"result_sha256": "7" * 64}),
        ),
    )
    monkeypatch.setattr(
        postfold,
        "rehydrate_lfm2_c61_tied_interface",
        lambda *_a, **_k: (interface, interface_rehydration),
    )
    foundation_rehydration_calls = []

    def rehydrate_foundation(*_args, **kwargs):
        foundation_rehydration_calls.append(kwargs)
        return interface, interface_rehydration

    monkeypatch.setattr(
        postfold,
        "rehydrate_lfm2_semantic_foundation_interface",
        rehydrate_foundation,
    )
    monkeypatch.setattr(
        postfold,
        "load_lfm2_semantic_compiler_lineage",
        lambda *_a, **_k: MappingProxyType({}),
    )

    def install_gauge(_model, source_interface, mapped_table, *, mode):
        events.append("tied_residual_gauge")
        assert source_interface is interface
        assert mapped_table is source_table
        assert mode == postfold.TIED_RESIDUAL_STRONG_CONGRUENCE_MODE
        return self_dual_interface, gauge_receipt

    monkeypatch.setattr(
        postfold,
        "install_lfm2_tied_residual_congruence_gauge",
        install_gauge,
    )

    def layer_stack_compiler(**_kwargs):
        events.append("layer_stack")
        assert _kwargs["interface"] is self_dual_interface
        assert _kwargs["fold_compute_device"] == "cuda:1"
        assert _kwargs["authenticated_cache_session"] is pipeline_cache_session
        return layer_stack

    def terminal_compiler(**_kwargs):
        events.append("terminal")
        assert _kwargs["layer_stack"] is layer_stack
        assert _kwargs["interface"] is self_dual_interface
        assert _kwargs["terminal_cache_compatibility"] is (terminal_cache_compatibility)
        assert _kwargs["authenticated_cache_session"] is pipeline_cache_session
        return terminal

    def publish(**_kwargs):
        events.append("checkpoint")
        return object(), MappingProxyType({"result_sha256": "c" * 64})

    monkeypatch.setattr(
        postfold,
        "compile_install_lfm2_semantic_layer_stack_stage",
        layer_stack_compiler,
    )
    attention_authentication_body = {
        "attention_receipt_count": 6,
        "complete_six_tensor_bundle_count": 6,
    }
    attention_authentication = MappingProxyType(
        {
            **attention_authentication_body,
            "result_sha256": canonical_sha256(attention_authentication_body),
        }
    )
    monkeypatch.setattr(
        postfold,
        "authenticate_lfm2_semantic_layer_stack_attention_receipts",
        lambda _receipt: attention_authentication,
    )
    monkeypatch.setattr(
        postfold,
        "publish_lfm2_semantic_postfold_checkpoint",
        publish,
    )
    monkeypatch.setattr(
        postfold,
        "fingerprint_lfm2_parent",
        lambda _model: SimpleNamespace(identity_sha256="d" * 64),
    )
    input_checkpoint = tmp_path / "input"
    input_checkpoint.mkdir()
    parent_directory = tmp_path / "parent"
    parent_directory.mkdir()
    bridge_directory = tmp_path / "bridge"
    bridge_directory.mkdir()
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    compiler_diagnostic = tmp_path / "compiler.json"
    compiler_diagnostic.write_text(
        (
            json.dumps({"schema": postfold.LFM2_SEMANTIC_FOUNDATION_SCHEMA})
            if use_foundation_statistics
            else "{}"
        ),
        encoding="utf-8",
    )

    result = postfold.run_lfm2_semantic_postfold(
        input_checkpoint_directory=input_checkpoint,
        expected_input_artifact_sha256="4" * 64,
        output_checkpoint_directory=tmp_path / "output",
        cache_directory=cache_directory,
        expected_cache_key=cache.cache_key,
        bridge_directory=bridge_directory,
        expected_bridge_artifact_sha256=bridge.artifact_sha256,
        donor_tokenizer=object(),
        target_tokenizer=tokenizer,
        parent_model_directory=parent_directory,
        compiler_diagnostic_json=compiler_diagnostic,
        expected_compiler_diagnostic_sha256="e" * 64,
        device="cpu",
        fold_compute_device="cuda:1",
        fused_terminal_compiler=terminal_compiler,
        **(
            {
                "interface_statistics_directory": tmp_path / "statistics",
                "expected_interface_statistics_sha256": (statistics_artifact_sha256),
            }
            if use_foundation_statistics
            else {}
        ),
    )

    assert cache_open_calls == [False]
    assert selective_session_calls == ["constructed"]
    assert events == [
        "metadata_preflight",
        "cache_session_open",
        "tied_residual_gauge",
        "layer_stack",
        "terminal",
        "cache_session_close",
        "checkpoint",
    ]
    assert result["compiler_cache_key"] == "8" * 64
    assert result["semantic_evidence_cache_key"] == cache.cache_key
    assert result["context_binding"]["fold_compute_device"] == "cuda:1"
    assert (
        result["context_binding"][
            "compiler_and_semantic_cache_identities_bound_separately"
        ]
        is True
    )
    assert result["claims"]["training_performed"] is False
    assert result["claims"]["attention_correction_installed"] is True
    assert result["claims"]["complete_causal_layer_stack_installed"] is False
    assert result["claims"]["attention_correction_install_count"] == 5
    assert result["claims"]["terminal_writer_install_count"] == 0
    assert result["terminal_incumbent_retained"] is True
    assert (
        result["tied_interface_mode"] == postfold.TIED_RESIDUAL_STRONG_CONGRUENCE_MODE
    )
    assert result["tied_residual_interface_gauge"] == dict(gauge_receipt)
    assert result["self_dual_tied_gauge"] == dict(gauge_receipt)
    session_reuse = result["pipeline_cache_session_reuse"]
    assert (
        session_reuse["pipeline_cache_session_sha256"]
        == session_receipt["session_sha256"]
    )
    assert (
        session_reuse["same_pipeline_session_reused_by_layer_stack_and_terminal"]
        is True
    )
    assert session_reuse["trace_file_sha256_passes_per_batch"] == 1
    assert session_reuse["additional_full_trace_file_sha256_passes_in_pipeline"] == 0
    assert session_reuse["terminal_full_tensor_inventory_payload_loads_total"] == 1
    assert (
        session_reuse["terminal_full_tensor_inventory_payload_logical_bytes_total"]
        == 128
    )
    if use_foundation_statistics:
        assert len(foundation_rehydration_calls) == 1
        assert foundation_rehydration_calls[0]["interface_statistics_directory"] == (
            tmp_path / "statistics"
        )
        assert (
            foundation_rehydration_calls[0]["expected_interface_statistics_sha256"]
            == statistics_artifact_sha256
        )
        statistics_binding = result["interface_statistics_replay"]
        assert statistics_binding["artifact_sha256"] == statistics_artifact_sha256
        assert statistics_binding["manifest_file_sha256"] == (
            statistics_manifest_sha256
        )
        assert statistics_binding["replay_sha256"] == statistics_replay_sha256
        assert statistics_binding["semantic_evidence_cache_open_calls"] == 1
        assert statistics_binding["compiler_cache_open_calls"] == 0
        assert statistics_binding["trace_safetensor_open_calls"] == 0
        assert statistics_binding["selective_provider_open_calls"] == 0
        assert result["context_binding"]["interface_statistics_replay"] == (
            statistics_binding
        )
    else:
        assert foundation_rehydration_calls == []
        assert "interface_statistics_replay" not in result
        assert "interface_statistics_replay" not in result["context_binding"]
    body = dict(result)
    seal = body.pop("result_sha256")
    assert seal == canonical_sha256(body)
