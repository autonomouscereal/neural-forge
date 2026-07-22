from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_attention_refold_adapter as adapter  # noqa: E402
import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack_adapter  # noqa: E402
import neural_forge.hf_lfm2_semantic_postfold as postfold  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (  # noqa: E402
    Lfm2EndpointQkMetrics,
    seal_lfm2_dense_qkvo_source_inventory,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_semantic_span_cpcf_alignment import (  # noqa: E402
    Lfm2SemanticSpanCpcfPhaseAlignment,
)
from neural_forge.hf_semantic_attention_lattice import (  # noqa: E402
    transport_lfm2_phase_probability_preserving_attention,
)


class _Config:
    model_type = "lfm2"
    norm_eps = 1e-5

    def to_dict(self) -> dict[str, object]:
        return {"model_type": self.model_type, "norm_eps": self.norm_eps}


class _Norm(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width, dtype=torch.bfloat16))


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 2
        self.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        self.v_proj = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        self.out_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.q_layernorm = _Norm(2)
        self.k_layernorm = _Norm(2)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _Backbone(torch.nn.Module):
    def __init__(self, layer_count: int = 1) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer() for _ in range(layer_count)])


class _Model(torch.nn.Module):
    def __init__(self, layer_count: int = 1) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _Backbone(layer_count)

    def forward(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("The semantic refold must never execute the model.")


class _Evidence:
    def __init__(
        self,
        *,
        batch_index: int,
        alignment: Lfm2SemanticSpanCpcfPhaseAlignment,
        transport: object,
        replay: adapter.Lfm2SemanticAttentionReplayEndpoint,
        source_inventory: object,
        complete_history: bool = True,
        layer_index: int = 0,
    ) -> None:
        self.batch_index = batch_index
        self.layer_index = layer_index
        self.alignment = alignment
        self.alignment_sha256 = alignment.alignment_sha256
        self.transport = transport
        self.actual_child_replay = replay
        self.source_inventory = source_inventory
        self.query_groups = ((0, 1), (2, 3))
        self.query_group_weights = ((0.5, 0.5), (0.5, 0.5))
        self.key_value_groups = ((0, 1),)
        self.key_value_group_weights = ((0.5, 0.5),)
        batch, _source_heads, query_length, key_length = map(
            int,
            transport.probabilities.shape,
        )
        self.folded_parent_query_pre_rope = (
            torch.arange(batch * 2 * query_length * 2, dtype=torch.float64)
            .reshape(batch, 2, query_length, 2)
            .add(1.0)
            .mul(0.01)
        )
        self.folded_parent_key_pre_rope = (
            torch.arange(batch * key_length * 2, dtype=torch.float64)
            .reshape(batch, 1, key_length, 2)
            .add(1.0)
            .mul(0.015)
        )
        activation_prior_body = {
            "schema": "test-parent-pre-rope-activation-prior-v1",
            "query_prior_sha256": adapter._tensor_sha256(
                self.folded_parent_query_pre_rope
            ),
            "key_prior_sha256": adapter._tensor_sha256(self.folded_parent_key_pre_rope),
            "parent_activation_evidence_only": True,
        }
        self.parent_pre_rope_prior_receipt = {
            **activation_prior_body,
            "prior_sha256": canonical_sha256(activation_prior_body),
        }
        self.parent_query_weight_prior = torch.eye(4, dtype=torch.float64)
        self.parent_key_weight_prior = torch.eye(4, dtype=torch.float64)[:2]
        self.parent_value_weight_prior = torch.tensor(
            [[0.5, -0.25, 0.75, 0.125], [-0.5, 0.25, 0.125, 0.75]],
            dtype=torch.float64,
        )
        self.parent_output_weight_prior = torch.eye(4, dtype=torch.float64)
        self.parent_query_norm_weight = torch.tensor(
            [1.25, 0.75],
            dtype=torch.float64,
        )
        self.parent_key_norm_weight = torch.tensor(
            [0.875, 1.375],
            dtype=torch.float64,
        )
        prior_body = {
            "schema": "test-parent-qkvo-weight-prior-v1",
            "source_inventory_sha256": source_inventory.inventory_sha256,
            "query_groups": [list(value) for value in self.query_groups],
            "query_group_weights": [list(value) for value in self.query_group_weights],
            "key_value_groups": [list(value) for value in self.key_value_groups],
            "key_value_group_weights": [
                list(value) for value in self.key_value_group_weights
            ],
            "query_prior_sha256": adapter._tensor_sha256(
                self.parent_query_weight_prior
            ),
            "key_prior_sha256": adapter._tensor_sha256(self.parent_key_weight_prior),
            "value_prior_sha256": adapter._tensor_sha256(
                self.parent_value_weight_prior
            ),
            "output_prior_sha256": adapter._tensor_sha256(
                self.parent_output_weight_prior
            ),
            "query_norm_weight_sha256": adapter._tensor_sha256(
                self.parent_query_norm_weight
            ),
            "key_norm_weight_sha256": adapter._tensor_sha256(
                self.parent_key_norm_weight
            ),
            "direct_parent_weights_used": True,
        }
        self.parent_weight_prior_receipt = {
            **prior_body,
            "prior_sha256": canonical_sha256(prior_body),
        }
        self.compiler_lineage_sha256 = "9" * 64
        capability_sha = canonical_sha256(
            {"batch": batch_index, "layer": layer_index, "kind": "capability"}
        )
        self.capability = {
            "capability_sha256": capability_sha,
            "layer_index": layer_index,
            "exact_history_supported": complete_history,
            "blocking_reasons": [] if complete_history else ["missing"],
            "missing_donor_prefix_positions": [],
        }
        self.history = {
            "history_sha256": canonical_sha256(
                {"batch": batch_index, "layer": layer_index, "kind": "history"}
            ),
            "source_capability_sha256": capability_sha,
            "token_count": alignment.donor_full_prefix_tokens,
        }
        self.evidence_sha256 = canonical_sha256(
            {
                "batch": batch_index,
                "alignment": alignment.alignment_sha256,
                "transport": transport.transport_sha256,
                "replay": replay.replay_sha256,
            }
        )

    def to_evidence_dict(self) -> dict[str, str]:
        return {"evidence_sha256": self.evidence_sha256}


def _alignment() -> Lfm2SemanticSpanCpcfPhaseAlignment:
    return Lfm2SemanticSpanCpcfPhaseAlignment(
        phase="prefill",
        semantic_phase_sha256="1" * 64,
        donor_full_prefix_token_ids=(10, 11),
        target_full_prefix_token_ids=(20, 21, 22),
        target_position_origin=0,
        target_to_donor_boundary_positions=(0, None, 1),
        target_to_donor_span_owner_positions=(0, 1, 1),
        parent_sampled_suffix_tokens=2,
        parent_history_suffix_tokens=2,
        target_rewrite_start=0,
        target_replaced_suffix_tokens=0,
        target_replacement_token_ids=(20, 21, 22),
    )


def _source_inventory() -> object:
    generator = torch.Generator().manual_seed(7)
    return seal_lfm2_dense_qkvo_source_inventory(
        source_query_weight=torch.randn(8, 8, generator=generator).to(torch.bfloat16),
        source_key_weight=torch.randn(4, 8, generator=generator).to(torch.bfloat16),
        source_value_weight=torch.randn(4, 8, generator=generator).to(torch.bfloat16),
        source_output_weight=torch.randn(8, 8, generator=generator).to(torch.bfloat16),
    )


def _evidence(
    model: _Model,
    *,
    complete_history: bool = True,
    layer_index: int = 0,
) -> tuple[_Evidence, ...]:
    alignment = _alignment()
    generator = torch.Generator().manual_seed(11)
    parent_query = torch.randn(1, 4, 2, 2, generator=generator).to(torch.bfloat16)
    parent_key = torch.randn(1, 2, 2, 2, generator=generator).to(torch.bfloat16)
    parent_value = torch.randn(1, 2, 2, 2, generator=generator).to(torch.bfloat16)
    transport = transport_lfm2_phase_probability_preserving_attention(
        alignment,
        parent_query_post_rope=parent_query,
        parent_key_post_rope=parent_key,
        parent_value=parent_value,
        target_rope_inv_freq=torch.tensor([1.0]),
    )
    state_sha = fingerprint_lfm2_parent(model).identity_sha256
    inventory = _source_inventory()
    result = []
    query_rows = len(transport.query_target_positions)
    for batch_index in range(4):
        replay = adapter.Lfm2SemanticAttentionReplayEndpoint(
            batch_index=batch_index,
            layer_index=layer_index,
            alignment_sha256=alignment.alignment_sha256,
            cache_key="2" * 64,
            bridge_artifact_sha256="3" * 64,
            captured_child_model_state_sha256=state_sha,
            target_query_inputs=torch.randn(
                1,
                query_rows,
                4,
                generator=generator,
            ).to(torch.bfloat16),
            target_key_value_inputs=torch.randn(
                1,
                3,
                4,
                generator=generator,
            ).to(torch.bfloat16),
            actual_child_pre_attention_residual=torch.randn(
                1,
                query_rows,
                4,
                generator=generator,
            ),
            parent_post_attention_residual_target_space=torch.randn(
                1,
                query_rows,
                4,
                generator=generator,
            ),
            query_row_sqrt_weights=torch.ones(1, query_rows),
        )
        result.append(
            _Evidence(
                batch_index=batch_index,
                alignment=alignment,
                transport=transport,
                replay=replay,
                source_inventory=inventory,
                complete_history=complete_history,
                layer_index=layer_index,
            )
        )
    return tuple(result)


def _metrics(endpoint: float) -> Lfm2EndpointQkMetrics:
    return Lfm2EndpointQkMetrics(
        case_count=1,
        eligible_route_rows=1,
        endpoint_relative_rmse=endpoint,
        probability_relative_rmse=endpoint,
        js_divergence_mean=endpoint,
        js_divergence_worst=endpoint,
        top1_route_agreement_fraction=1.0,
        maximum_causal_probability_leakage=0.0,
    )


def _install_fake_compiler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation_regresses: bool = False,
) -> None:
    class _Compiled:
        def __init__(self, seed: object) -> None:
            self.query_weight = seed.query_weight.clone()
            self.query_weight[0, 0] += 1.0
            self.key_weight = seed.key_weight.clone()
            self.accepted = True
            self.changed = True
            self.report = {"report_sha256": "4" * 64}

        def verify(self) -> str:
            return self.report["report_sha256"]

    def compile_fake(**kwargs: object) -> _Compiled:
        return _Compiled(kwargs["seed"])

    values = iter(
        (
            _metrics(0.5),
            _metrics(0.4),
            _metrics(0.5),
            _metrics(0.7 if validation_regresses else 0.4),
        )
    )
    monkeypatch.setattr(adapter, "Lfm2CompiledEndpointQkResidualFold", _Compiled)
    monkeypatch.setattr(
        adapter,
        "compile_lfm2_cached_endpoint_qk_residual_fold",
        compile_fake,
    )
    monkeypatch.setattr(
        adapter, "_evaluate_bank", lambda *_args, **_kwargs: next(values)
    )


def _install_fake_noop_qk_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Compiled:
        def __init__(self, seed: object) -> None:
            self.query_weight = seed.query_weight.clone()
            self.key_weight = seed.key_weight.clone()
            self.accepted = False
            self.changed = False
            self.report = {"report_sha256": "5" * 64}

        def verify(self) -> str:
            return self.report["report_sha256"]

    def compile_fake(**kwargs: object) -> _Compiled:
        return _Compiled(kwargs["seed"])

    monkeypatch.setattr(adapter, "Lfm2CompiledEndpointQkResidualFold", _Compiled)
    monkeypatch.setattr(
        adapter,
        "compile_lfm2_cached_endpoint_qk_residual_fold",
        compile_fake,
    )
    monkeypatch.setattr(
        adapter,
        "_evaluate_bank",
        lambda *_args, **_kwargs: _metrics(0.5),
    )


def _force_unchanged_output_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    original = adapter._compile_output_writer

    def compile_unchanged(
        fit_cases: object,
        incumbent_output_weight: torch.Tensor,
        *,
        solver_device: str,
        ridge: float = 1e-4,
    ) -> object:
        proposal = original(
            fit_cases,
            incumbent_output_weight,
            solver_device=solver_device,
            ridge=ridge,
        )
        baseline = adapter._writer_metrics(tuple(fit_cases), incumbent_output_weight)
        return adapter._OutputWriterProposal(
            output_weight=incumbent_output_weight.detach().cpu().clone(),
            changed=False,
            fit_improved=False,
            selected_alpha=None,
            baseline_fit_metrics=baseline,
            candidate_fit_metrics=baseline,
            ridge=proposal.ridge,
            effective_ridge=proposal.effective_ridge,
            fit_context_sha256=proposal.fit_context_sha256,
            prior_preservation_receipt=proposal.prior_preservation_receipt,
        )

    monkeypatch.setattr(adapter, "_compile_output_writer", compile_unchanged)


def _force_unchanged_joint_vo(monkeypatch: pytest.MonkeyPatch) -> None:
    original = adapter._compile_joint_vo

    def compile_unchanged(
        cases: object,
        incumbent_value: torch.Tensor,
        incumbent_output: torch.Tensor,
        *,
        head_dim: int,
        solver_device: str,
        ridge: float = 1e-4,
    ) -> object:
        proposal = original(
            cases,
            incumbent_value,
            incumbent_output,
            head_dim=head_dim,
            solver_device=solver_device,
            ridge=ridge,
        )
        baseline = adapter._joint_vo_metrics(
            tuple(cases),
            incumbent_value,
            incumbent_output,
            head_dim=head_dim,
        )
        return adapter._JointVoProposal(
            value_weight=incumbent_value.detach().cpu().clone(),
            output_weight=incumbent_output.detach().cpu().clone(),
            changed=False,
            fit_improved=False,
            baseline_fit_metrics=baseline,
            candidate_fit_metrics=baseline,
            ridge=proposal.ridge,
            effective_ridge=proposal.effective_ridge,
            dual_dimension=proposal.dual_dimension,
            primal_feature_dimension=proposal.primal_feature_dimension,
            fit_evidence_sha256=proposal.fit_evidence_sha256,
            factorization_receipts=proposal.factorization_receipts,
        )

    monkeypatch.setattr(adapter, "_compile_joint_vo", compile_unchanged)


def test_closed_form_fit_corrections_publish_with_heldout_metrics_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This assertion describes the fixed-V writer branch.  Seed only the model
    # construction because another valid random incumbent can make FIT ranking
    # select the competing joint V/O branch instead.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = _Model()
    evidence = _evidence(model)
    _install_fake_compiler(monkeypatch)
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )
    authenticated = postfold._authenticate_lfm2_semantic_attention_operator_receipt(
        receipt,
        layer_index=0,
    )

    assert receipt["schema"] == "neural-forge-lfm2-semantic-attention-installation-v1"
    assert authenticated["numeric_behavior_gate_used"] is False
    assert receipt["changed_state_keys"] == [
        "model.layers.0.self_attn.out_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    assert not torch.equal(model.model.layers[0].self_attn.q_proj.weight, before)
    report = receipt["analytic_refold"]
    assert report["candidate_frozen_before_validation"] is True
    assert report["validation_triggered_refit"] is False
    assert report["numeric_parent_child_weight_match_objective"] is False
    assert report["analytically_refit_projection_roles"] == [
        "q_proj",
        "out_proj",
    ]
    assert report["fixed_inherited_projection_roles"] == [
        "k_proj",
        "v_proj",
    ]
    assert report["fresh_query_refit_performed"] is True
    assert report["fresh_key_refit_performed"] is False
    assert report["fresh_four_matrix_qkvo_refit_claimed"] is False
    assert report["fresh_output_writer_refit_performed"] is True
    assert report["output_writer"]["accepted"] is True
    assert report["output_writer"]["rejection_partition"] is None
    assert report["output_writer"]["heldout_metrics_are_diagnostic_only"] is True
    assert report["numeric_metric_used_for_publication_or_rollback"] is False
    assert report["all_semantically_exact_endpoint_rows_used_for_attention_fit"] is True
    assert report["terminal_endpoint_only_supervision"] is False
    attention = model.model.layers[0].self_attn
    expected_bundle = {
        "q_proj.weight": adapter._tensor_sha256(attention.q_proj.weight),
        "k_proj.weight": adapter._tensor_sha256(attention.k_proj.weight),
        "v_proj.weight": adapter._tensor_sha256(attention.v_proj.weight),
        "out_proj.weight": adapter._tensor_sha256(attention.out_proj.weight),
        "q_layernorm.weight": adapter._tensor_sha256(attention.q_layernorm.weight),
        "k_layernorm.weight": adapter._tensor_sha256(attention.k_layernorm.weight),
    }
    assert report["frozen_attention_six_tensor_bundle"] == expected_bundle
    assert report["frozen_attention_six_tensor_bundle_roles"] == list(expected_bundle)
    assert report["frozen_attention_six_tensor_bundle_sha256"] == canonical_sha256(
        expected_bundle
    )
    assert (
        report[
            "analytic_fp64_target_construction_distinct_from_bfloat16_runtime_replay"
        ]
        is True
    )
    assert report["runtime_replay_numeric_contract"] == (
        "bfloat16-linears-fp32-qk-softmax-bfloat16-probability-context"
    )
    assert report["cross_head_dimension_bridge_deferred"] is True
    assert report["refold_operation_audit"]["model_forward_calls"] == 0


def test_validation_regression_is_diagnostic_and_cannot_rollback_fit_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    evidence = _evidence(model)
    _install_fake_compiler(monkeypatch, validation_regresses=True)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )
    authenticated = postfold._authenticate_lfm2_semantic_attention_operator_receipt(
        receipt,
        layer_index=0,
    )

    assert receipt["installed"] is True
    assert authenticated["numeric_behavior_gate_used"] is False
    assert not torch.equal(
        model.model.layers[0].self_attn.q_proj.weight,
        before["model.layers.0.self_attn.q_proj.weight"],
    )
    report = receipt["analytic_refold"]
    assert report["validation_nonregression_diagnostic"] is False
    assert report["heldout_metrics_are_diagnostic_only"] is True
    assert report["numeric_metric_used_for_publication_or_rollback"] is False


def test_accepted_output_writer_is_installed_in_the_same_qkvo_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test exercises the fixed-V output-writer transaction specifically.
    # The default module initializer is random, and some valid initial states
    # make the FIT-guided lane prefer the competing joint V/O proposal instead.
    # Isolate a known writer-selecting state without perturbing the suite RNG.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = _Model()
    evidence = _evidence(model)
    _install_fake_compiler(monkeypatch)
    before_query = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    before_output = model.model.layers[0].self_attn.out_proj.weight.detach().clone()

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    assert receipt["changed_state_keys"] == [
        "model.layers.0.self_attn.out_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    assert not torch.equal(model.model.layers[0].self_attn.q_proj.weight, before_query)
    assert not torch.equal(
        model.model.layers[0].self_attn.out_proj.weight, before_output
    )
    report = receipt["analytic_refold"]
    assert report["value_output_candidate_gate"]["published_candidate"] == (
        "fixed_v_output_only"
    )
    assert report["fresh_output_writer_refit_performed"] is True
    assert report["analytically_refit_projection_roles"] == [
        "q_proj",
        "out_proj",
    ]
    assert report["fixed_inherited_projection_roles"] == ["k_proj", "v_proj"]
    assert report["fresh_value_refit_performed"] is False
    assert report["complete_qkvo_bundle_installed_transactionally"] is True


def test_joint_only_vo_candidate_publishes_and_installs_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    evidence = _evidence(model)
    _install_fake_compiler(monkeypatch)
    _force_unchanged_output_writer(monkeypatch)
    before_value = model.model.layers[0].self_attn.v_proj.weight.detach().clone()
    before_output = model.model.layers[0].self_attn.out_proj.weight.detach().clone()

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )
    postfold._authenticate_lfm2_semantic_attention_operator_receipt(
        receipt,
        layer_index=0,
    )

    report = receipt["analytic_refold"]
    gate = report["value_output_candidate_gate"]
    assert gate["published_candidate"] == "joint_value_output"
    assert gate["structurally_eligible_candidate_count"] == 1
    assert report["output_writer"]["proposal"]["changed"] is False
    assert report["output_writer"]["accepted"] is False
    assert report["joint_value_output_candidate_published"] is True
    assert report["fresh_joint_value_output_refit_performed"] is True
    assert (
        report["joint_value_output"]["value_weight_sha256"]
        == report["frozen_attention_six_tensor_bundle"]["v_proj.weight"]
    )
    assert (
        report["joint_value_output"]["output_weight_sha256"]
        == report["frozen_attention_six_tensor_bundle"]["out_proj.weight"]
    )
    assert not torch.equal(model.model.layers[0].self_attn.v_proj.weight, before_value)
    assert not torch.equal(
        model.model.layers[0].self_attn.out_proj.weight,
        before_output,
    )


def test_qk_seed_fallback_can_publish_joint_v_o_factorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    evidence = _evidence(model)
    _install_fake_noop_qk_compiler(monkeypatch)
    _force_unchanged_output_writer(monkeypatch)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    report = receipt["analytic_refold"]
    assert report["qk_compiler_changed"] is False
    assert report["authenticated_qk_seed_fallback_entered_vo_competition"] is True
    assert report["value_output_candidate_gate"]["published_candidate"] == (
        "joint_value_output"
    )
    assert report["joint_value_output_candidate_published"] is True
    assert torch.equal(
        model.model.layers[0].self_attn.q_proj.weight,
        before["model.layers.0.self_attn.q_proj.weight"],
    )
    assert torch.equal(
        model.model.layers[0].self_attn.k_proj.weight,
        before["model.layers.0.self_attn.k_proj.weight"],
    )
    assert not torch.equal(
        model.model.layers[0].self_attn.v_proj.weight,
        before["model.layers.0.self_attn.v_proj.weight"],
    )


def test_total_qkvo_noop_returns_authenticated_receipt_without_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _evidence(model)
    _install_fake_noop_qk_compiler(monkeypatch)
    _force_unchanged_output_writer(monkeypatch)
    _force_unchanged_joint_vo(monkeypatch)
    monkeypatch.setattr(
        adapter,
        "_publish_complete_attention_cold_start",
        lambda **_kwargs: pytest.fail(
            "A fully nonzero incumbent must preserve the legacy publication path."
        ),
    )
    before_identity = fingerprint_lfm2_parent(model).identity_sha256
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())
    monkeypatch.setattr(
        adapter,
        "_copy_candidate",
        lambda *_args, **_kwargs: pytest.fail(
            "The authenticated total-incumbent path must not copy Parameters."
        ),
    )
    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    assert receipt["status"] == adapter.LFM2_SEMANTIC_ATTENTION_STRUCTURAL_NOOP_STATUS
    assert receipt["installed"] is False
    assert receipt["install_count"] == 0
    assert receipt["changed_state_keys"] == []
    assert receipt["model_state_before_sha256"] == before_identity
    assert receipt["model_state_after_sha256"] == before_identity
    assert (
        receipt["target_topology_sha256_before"]
        == receipt["target_topology_sha256_after"]
    )
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["exact_noop_state_proven"] is True
    assert receipt["selected_candidate"] == "incumbent-noop"
    assert receipt["selected_candidate_state_identical_to_incumbent"] is True
    assert receipt["incumbent_noop_fallback_used"] is True
    assert receipt["incumbent_noop_fallback_reason"] == (
        "all-analytic-proposals-structurally-invalid-or-identical"
    )
    assert receipt["structurally_eligible_candidate_count"] == 0
    assert receipt["incumbent_selected_by_heldout_gates"] is False
    assert receipt["fit_check_nonregression_passed"] is None
    assert receipt["selection_nonregression_passed"] is None
    assert receipt["validation_nonregression_passed"] is None
    assert receipt["numeric_metric_used_for_publication_or_rollback"] is False
    report = receipt["analytic_refold"]
    assert "complete_attention_cold_start" not in report
    assert report["analytically_refit_projection_roles"] == []
    assert report["fixed_inherited_projection_roles"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ]
    assert report["complete_qkvo_bundle_installed_transactionally"] is False
    assert report["complete_qkvo_bundle_transaction_verified"] is True
    assert report["authenticated_incumbent_qkvo_retained_without_installation"] is True
    assert report["projection_parameter_copy_calls"] == 0
    assert len(report["frozen_attention_six_tensor_bundle"]) == 6
    assert report["frozen_attention_six_tensor_bundle_sha256"] == canonical_sha256(
        report["frozen_attention_six_tensor_bundle"]
    )
    assert stack_adapter._authenticated_attention_incumbent_noop(
        receipt,
        before_sha256=before_identity,
        after_sha256=before_identity,
        changed_state_keys=(),
    )
    validated = stack_adapter._install_attention_operator(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
        installer=lambda **_kwargs: receipt,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    assert validated["result_sha256"] == receipt["result_sha256"]
    sealed = dict(receipt)
    digest = sealed.pop("result_sha256")
    assert canonical_sha256(sealed) == digest
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids


def test_exact_zero_qk_is_evidence_planted_and_cannot_publish_as_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.k_proj.weight.zero_()
    evidence = _evidence(model)

    class _Compiled:
        def __init__(self, seed: object) -> None:
            self.query_weight = seed.query_weight.clone()
            self.key_weight = seed.key_weight.clone()
            self.accepted = False
            self.changed = False
            self.report = {"report_sha256": "8" * 64}

        def verify(self) -> str:
            return self.report["report_sha256"]

    monkeypatch.setattr(adapter, "Lfm2CompiledEndpointQkResidualFold", _Compiled)
    monkeypatch.setattr(
        adapter,
        "compile_lfm2_cached_endpoint_qk_residual_fold",
        lambda **kwargs: _Compiled(kwargs["seed"]),
    )
    metric_values = iter(
        [
            value
            for _partition in adapter._PARTITIONS
            for value in (_metrics(0.8), _metrics(0.4))
        ]
        + [_metrics(0.4), _metrics(0.4), _metrics(0.4), _metrics(0.4)]
    )
    monkeypatch.setattr(
        adapter,
        "_evaluate_bank",
        lambda *_args, **_kwargs: next(metric_values),
    )

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    assert int(torch.count_nonzero(attention.q_proj.weight).item()) > 0
    assert int(torch.count_nonzero(attention.k_proj.weight).item()) > 0
    assert receipt["installed"] is True
    assert receipt["parameter_count_changed"] is False
    assert receipt["topology_changed"] is False
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["changed_state_keys"] == [
        "model.layers.0.self_attn.k_layernorm.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.out_proj.weight",
        "model.layers.0.self_attn.q_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    cold = receipt["analytic_refold"]["qk_cold_start"]
    assert cold["original_query_exact_zero"] is True
    assert cold["original_key_exact_zero"] is True
    assert cold["fit_strictly_improved"] is True
    assert cold["candidate_frozen_before_fit_check_selection_validation"] is True
    assert cold["heldout_refit_or_reselection"] is False
    assert cold["zero_incumbent_noop_allowed"] is False
    assert cold["plant"]["fit_partition_only_used_for_plant"] is True
    assert cold["plant"]["heldout_partitions_used_for_plant"] is False
    assert cold["plant"]["randomness_used"] is False
    assert cold["plant"]["gradients_used"] is False
    assert cold["operation_audit"]["training_steps"] == 0
    sealed = dict(receipt)
    digest = sealed.pop("result_sha256")
    assert canonical_sha256(sealed) == digest


def test_all_zero_bfloat16_qkvo_is_compulsorily_planted_with_full_axis_coverage() -> (
    None
):
    model = _Model()
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.out_proj,
        ):
            projection.weight.zero_()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _evidence(model)
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )
    authenticated = postfold._authenticate_lfm2_semantic_attention_operator_receipt(
        receipt,
        layer_index=0,
    )

    assert receipt["installed"] is True
    assert authenticated["numeric_behavior_gate_used"] is False
    assert receipt["changed_state_keys"] == [
        "model.layers.0.self_attn.k_layernorm.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.out_proj.weight",
        "model.layers.0.self_attn.q_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    torch.testing.assert_close(
        attention.q_layernorm.weight,
        evidence[0].parent_query_norm_weight.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        attention.k_layernorm.weight,
        evidence[0].parent_key_norm_weight.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids
    cold = receipt["analytic_refold"]["complete_attention_cold_start"]
    assert cold["exact_zero_roles_at_entry"] == [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
    ]
    assert cold["all_qkvo_roles_finite_and_structurally_covered"] is True
    assert cold["fit_partition_only_used_for_plant"] is True
    assert cold["heldout_metrics_are_diagnostics_not_publication_gates"] is True
    assert cold["zero_role_incumbent_noop_allowed"] is False
    assert cold["randomness_used"] is False
    assert cold["training_used"] is False
    assert cold["pruning_used"] is False
    assert cold["quantization_used"] is False
    for role, coverage in cold["published_role_coverage"].items():
        assert coverage["structurally_nonzero"] is True, role
        assert coverage["all_output_structural_units_covered"] is True
        assert coverage["all_input_structural_units_covered"] is True
    for projection in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        attention.out_proj,
    ):
        assert projection.weight.dtype == torch.bfloat16
        assert int(torch.count_nonzero(projection.weight).item()) > 0
    audit = receipt["analytic_refold"]["refold_operation_audit"]
    assert audit["training_steps"] == 0
    assert audit["backward_calls"] == 0
    assert audit["optimizer_steps"] == 0
    assert audit["parameters_pruned"] == 0
    assert audit["parameters_quantized"] == 0
    sealed = dict(receipt)
    digest = sealed.pop("result_sha256")
    assert canonical_sha256(sealed) == digest


def test_blank_structural_and_residual_writer_attention_lanes_are_distinct() -> None:
    state_by_recipe: dict[str, dict[str, str]] = {}
    recipe_ids = ("fit_guided", "full_strength_structural", "residual_writer")
    for recipe_id in recipe_ids:
        model = _Model()
        attention = model.model.layers[0].self_attn
        with torch.no_grad():
            for projection in (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.out_proj,
            ):
                projection.weight.zero_()
        receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
            target_model=model,
            layer_index=0,
            evidence=_evidence(model),
            cache_key="2" * 64,
            bridge_artifact_sha256="3" * 64,
            body_recipe=recipe_id,
        )
        gate = receipt["analytic_refold"]["value_output_candidate_gate"]
        authenticated = (
            postfold._authenticate_lfm2_semantic_attention_operator_receipt(
                receipt,
                layer_index=0,
                body_recipe=recipe_id,
            )
        )
        assert receipt["body_recipe_id"] == recipe_id
        assert authenticated["body_recipe_id"] == recipe_id
        assert (
            authenticated[
                "recipe_bound_publication_with_diagnostic_heldout_authenticated"
            ]
            is True
        )
        assert gate["body_recipe_id"] == recipe_id
        assert gate["fit_metrics_used_for_candidate_ordering"] is (
            recipe_id == "fit_guided"
        )
        assert receipt["heldout_metrics_are_diagnostic_only"] is True
        foreign_recipe = next(value for value in recipe_ids if value != recipe_id)
        with pytest.raises(
            postfold.Lfm2SemanticPostfoldError,
            match="body recipe",
        ):
            postfold._authenticate_lfm2_semantic_attention_operator_receipt(
                receipt,
                layer_index=0,
                body_recipe=foreign_recipe,
            )
        assert all(
            int(torch.count_nonzero(projection.weight).item()) > 0
            for projection in (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.out_proj,
            )
        )
        state_by_recipe[recipe_id] = {
            role: adapter._tensor_sha256(getattr(attention, role).weight)
            for role in ("q_proj", "k_proj", "v_proj", "out_proj")
        }

    assert state_by_recipe["full_strength_structural"] != state_by_recipe[
        "residual_writer"
    ]


def test_all_zero_qkvo_heldout_regression_is_diagnostic_not_blank_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.out_proj,
        ):
            projection.weight.zero_()
    evidence = _evidence(model)
    values = iter(
        (
            _metrics(0.8),
            _metrics(0.4),
            _metrics(0.2),
            _metrics(0.9),
            _metrics(0.2),
            _metrics(0.9),
            _metrics(0.2),
            _metrics(0.9),
        )
    )
    monkeypatch.setattr(
        adapter,
        "_evaluate_bank",
        lambda *_args, **_kwargs: next(values),
    )

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    assert receipt["installed"] is True
    assert receipt["fit_check_nonregression_passed"] is False
    assert receipt["selection_nonregression_passed"] is False
    assert receipt["validation_nonregression_passed"] is False
    assert receipt["heldout_nonregression_is_diagnostic_only"] is True
    cold = receipt["analytic_refold"]["complete_attention_cold_start"]
    qk = cold["qk_cold_start"]
    assert qk["heldout_nonregression_passed"] is False
    assert qk["partitions"]["fit_check"]["nonregressing"] is False
    assert qk["partitions"]["selection"]["nonregressing"] is False
    assert qk["partitions"]["validation"]["nonregressing"] is False
    assert cold["heldout_regression_can_restore_blank_incumbent"] is False
    assert cold["heldout_metrics_are_diagnostics_not_publication_gates"] is True
    for projection in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        attention.out_proj,
    ):
        assert int(torch.count_nonzero(projection.weight).item()) > 0


def test_exact_zero_final_attention_layer_gets_complete_nonzero_plant() -> None:
    final_attention_layer = 12
    model = _Model(layer_count=final_attention_layer + 1)
    attention = model.model.layers[final_attention_layer].self_attn
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.out_proj,
        ):
            projection.weight.zero_()
    evidence = _evidence(model, layer_index=final_attention_layer)

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=final_attention_layer,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )
    postfold._authenticate_lfm2_semantic_attention_operator_receipt(
        receipt,
        layer_index=final_attention_layer,
    )

    prefix = f"model.layers.{final_attention_layer}.self_attn."
    assert receipt["installed"] is True
    assert all(name.startswith(prefix) for name in receipt["changed_state_keys"])
    cold = receipt["analytic_refold"]["complete_attention_cold_start"]
    assert cold["exact_zero_roles_at_entry"] == [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
    ]
    assert all(
        coverage["structurally_nonzero"] is True
        for coverage in cold["published_role_coverage"].values()
    )
    for projection in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        attention.out_proj,
    ):
        assert int(torch.count_nonzero(projection.weight).item()) > 0


def test_joint_only_cold_start_publishes_without_claiming_output_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.out_proj,
        ):
            projection.weight.zero_()
    evidence = _evidence(model)
    original_compile = adapter._compile_output_writer

    def reject_writer(
        fit_cases: object,
        incumbent_output_weight: torch.Tensor,
        *,
        solver_device: str,
        ridge: float = 1e-4,
    ) -> object:
        proposal = original_compile(
            fit_cases,
            incumbent_output_weight,
            solver_device=solver_device,
            ridge=ridge,
        )
        baseline = adapter._writer_metrics(tuple(fit_cases), incumbent_output_weight)
        return adapter._OutputWriterProposal(
            output_weight=incumbent_output_weight.detach().cpu().clone(),
            changed=False,
            fit_improved=False,
            selected_alpha=None,
            baseline_fit_metrics=baseline,
            candidate_fit_metrics=baseline,
            ridge=proposal.ridge,
            effective_ridge=proposal.effective_ridge,
            fit_context_sha256=proposal.fit_context_sha256,
        )

    monkeypatch.setattr(adapter, "_compile_output_writer", reject_writer)
    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    report = receipt["analytic_refold"]
    assert report["output_writer"]["proposal"]["changed"] is False
    assert report["output_writer"]["accepted"] is False
    assert report["fresh_output_writer_refit_performed"] is False
    assert report["fresh_output_projection_refit_performed"] is True
    assert report["joint_value_output_candidate_published"] is True
    assert report["joint_value_output_component_contributed"] is True
    assert report["output_writer_source"] == (
        "fit_only_joint_value_output_residual_factorization"
    )


def test_cold_start_receipt_failure_rolls_back_successful_copy_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.out_proj,
        ):
            projection.weight.zero_()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _evidence(model)
    before_identity = fingerprint_lfm2_parent(model).identity_sha256
    before_topology = adapter._topology_sha256(model)
    before_count = adapter._parameter_count(model)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())
    successful_copy_observed = False

    def fail_receipt(**_kwargs: object) -> dict[str, object]:
        nonlocal successful_copy_observed
        successful_copy_observed = all(
            int(torch.count_nonzero(projection.weight).item()) > 0
            for projection in (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.out_proj,
            )
        )
        raise RuntimeError("injected-post-copy-receipt-failure")

    monkeypatch.setattr(adapter, "_installation_receipt_body", fail_receipt)
    with pytest.raises(RuntimeError, match="injected-post-copy-receipt-failure"):
        adapter.compile_install_lfm2_semantic_attention_postfold(
            target_model=model,
            layer_index=0,
            evidence=evidence,
            cache_key="2" * 64,
            bridge_artifact_sha256="3" * 64,
        )

    assert successful_copy_observed is True
    assert fingerprint_lfm2_parent(model).identity_sha256 == before_identity
    assert adapter._topology_sha256(model) == before_topology
    assert adapter._parameter_count(model) == before_count
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids


@pytest.mark.parametrize(
    "blank_role",
    ("q_proj", "k_proj", "v_proj", "out_proj"),
)
def test_single_zero_attention_role_is_planted_without_blank_noop(
    blank_role: str,
) -> None:
    model = _Model()
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        getattr(attention, blank_role).weight.zero_()
    before = {
        role: getattr(attention, role).weight.detach().clone()
        for role in ("q_proj", "k_proj", "v_proj", "out_proj")
    }
    evidence = _evidence(model)

    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    assert receipt["installed"] is True
    cold = receipt["analytic_refold"]["complete_attention_cold_start"]
    assert cold["exact_zero_roles_at_entry"] == [f"{blank_role}.weight"]
    assert cold["all_qkvo_roles_finite_and_structurally_covered"] is True
    assert int(torch.count_nonzero(getattr(attention, blank_role).weight).item()) > 0
    preserved_roles = (
        {"q_proj", "k_proj"} if blank_role in {"v_proj", "out_proj"} else set()
    )
    for role in preserved_roles:
        assert torch.equal(getattr(attention, role).weight, before[role])


def test_zero_value_role_is_identifiable_from_the_direct_parent_prior() -> None:
    model = _Model()
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        attention.v_proj.weight.zero_()
    evidence = _evidence(model)
    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    assert receipt["installed"] is True
    assert int(torch.count_nonzero(attention.v_proj.weight).item()) > 0
    cold = receipt["analytic_refold"]["complete_attention_cold_start"]
    assert cold["exact_zero_roles_at_entry"] == ["v_proj.weight"]
    assert cold["qk_cold_start"]["plant"]["query_ridge"] is None
    assert cold["qk_cold_start"]["plant"]["key_ridge"] is None


def test_fit_ridge_axis_completion_uses_real_coefficients_and_covers_every_axis() -> (
    None
):
    target = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, -1.0, 0.5, 3.0],
            [-1.0, 0.25, 2.5, 1.5],
            [0.75, 1.25, -2.0, 0.5],
        ],
        dtype=torch.float64,
    )
    inputs = torch.eye(4, dtype=torch.float64)
    outputs = inputs @ target.T
    incumbent = target.to(torch.bfloat16)
    incumbent[1].zero_()
    incumbent[:, 2].zero_()
    original = incumbent.clone()

    repaired, report = adapter._repair_dense_axis_coverage(
        incumbent,
        role="q_proj.weight",
        fit_inputs=inputs,
        fit_targets=outputs,
        fit_sqrt_weights=torch.ones(4, dtype=torch.float64),
    )

    assert report["coverage_before"]["structurally_nonzero"] is False
    assert report["coverage_after"]["structurally_nonzero"] is True
    assert report["repaired_element_count"] > 0
    assert report["fit_ridge"] is not None
    assert report["synthetic_epsilon_scaffolding_used"] is False
    assert report["existing_nonzero_entries_overwritten"] == 0
    torch.testing.assert_close(
        repaired[original.ne(0)],
        original[original.ne(0)],
        rtol=0,
        atol=0,
    )


def test_parent_activation_priors_cover_all_endpoints_in_native_rope_basis() -> None:
    model = _Model()
    item = _evidence(model)[0]
    source = adapter._normalized_case_source(
        item,
        layer_index=0,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )
    case = adapter._case_from_source(
        source,
        layer_index=0,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
        trajectory_row_weight_scale=1.0,
    )
    seed = adapter._seed(
        model,
        layer_index=0,
        source_inventory=item.source_inventory,
        source_artifact_sha256="4" * 64,
    )
    prior_seed = adapter._copy_seed(
        seed,
        query_weight=case.parent_query_weight_prior.to(torch.bfloat16),
        key_weight=case.parent_key_weight_prior.to(torch.bfloat16),
        value_weight=case.parent_value_weight_prior.to(torch.bfloat16),
        output_weight=case.parent_output_weight_prior.to(torch.bfloat16),
        query_norm_weight=case.parent_query_norm_weight.to(torch.bfloat16),
        key_norm_weight=case.parent_key_norm_weight.to(torch.bfloat16),
    )

    gauge, gauge_receipt = adapter._fit_pre_rope_dense_gauge(prior_seed, (case,))
    query_runtime, key_runtime, route_receipt = adapter._cold_route_runtime_targets(
        case,
        prior_seed,
        pre_rope_gauge=gauge,
    )
    query_raw = adapter._cold_projection_preimage(
        query_runtime,
        torch.zeros_like(query_runtime),
        norm_weight=prior_seed.query_norm_weight,
        norm_epsilon=prior_seed.norm_epsilon,
        cosine=case.query_rope_cos,
        sine=case.query_rope_sin,
        role="query-production-contract",
    )
    replay_normalized = query_raw * torch.rsqrt(
        query_raw.square().mean(dim=-1, keepdim=True) + prior_seed.norm_epsilon
    )
    replay_runtime = adapter._rope_forward_rows(
        replay_normalized
        * prior_seed.query_norm_weight.to(torch.float64).reshape(1, 1, 1, -1),
        case.query_rope_cos.to(torch.float64).unsqueeze(1),
        case.query_rope_sin.to(torch.float64).unsqueeze(1),
    )

    assert torch.equal(gauge, torch.eye(prior_seed.head_dim, dtype=torch.float64))
    assert gauge_receipt["arbitrary_dense_householder_used"] is False
    assert route_receipt["arbitrary_dense_pre_rope_gauge_used"] is False
    assert route_receipt["folded_parent_query_activation_prior_consumed"] is True
    assert route_receipt["folded_parent_key_value_activation_prior_consumed"] is True
    assert route_receipt["causal_logit_residual_applied_after_activation_prior"] is True
    assert route_receipt["query_length"] == int(
        case.target_probability_barycenter.shape[2]
    )
    assert route_receipt["query_length"] > 1
    assert route_receipt["explicit_causal_mask_consumed"] is True
    assert route_receipt["silent_logit_clipping_used"] is False
    assert query_runtime.shape[2] == route_receipt["query_length"]
    assert key_runtime.shape[1] == prior_seed.key_value_heads
    torch.testing.assert_close(replay_runtime, query_runtime, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(
        query_raw,
        torch.full_like(query_raw, 2.0**-0.5),
        rtol=0,
        atol=1e-12,
    )


def _obsolete_real_shaped_parent_bound_relative_rope_fixture() -> None:
    head_dim = 64
    query_heads = 8
    key_value_heads = 2
    queries_per_key_value = query_heads // key_value_heads
    key_length = 9
    generator = torch.Generator().manual_seed(101)
    frequencies = torch.linspace(0.01, 0.25, head_dim // 2, dtype=torch.float64)
    positions = torch.arange(key_length, dtype=torch.float64).reshape(-1, 1)
    angles = positions * frequencies.reshape(1, -1)
    key_cos = torch.cat((torch.cos(angles), torch.cos(angles)), dim=-1)
    key_sin = torch.cat((torch.sin(angles), torch.sin(angles)), dim=-1)
    query_cos = key_cos[-1:].reshape(1, 1, head_dim).to(torch.bfloat16)
    query_sin = key_sin[-1:].reshape(1, 1, head_dim).to(torch.bfloat16)
    key_cos = key_cos.reshape(1, key_length, head_dim).to(torch.bfloat16)
    key_sin = key_sin.reshape(1, key_length, head_dim).to(torch.bfloat16)
    head = torch.arange(1, query_heads + 1, dtype=torch.float64).reshape(1, -1, 1, 1)
    token = torch.arange(1, key_length + 1, dtype=torch.float64).reshape(1, 1, 1, -1)
    logits = 0.04 * torch.sin(head * token)
    probabilities = torch.softmax(logits, dim=-1)
    case = SimpleNamespace(
        evidence_sha256="a" * 64,
        target_probability_barycenter=probabilities,
        target_causal_mask=torch.ones_like(probabilities, dtype=torch.bool),
        query_rope_cos=query_cos,
        query_rope_sin=query_sin,
        key_rope_cos=key_cos,
        key_rope_sin=key_sin,
        folded_parent_query_pre_rope=torch.randn(
            1,
            query_heads,
            1,
            head_dim,
            generator=generator,
            dtype=torch.float64,
        ),
        folded_parent_key_pre_rope=torch.randn(
            1,
            query_heads,
            key_length,
            head_dim,
            generator=generator,
            dtype=torch.float64,
        ),
    )
    seed = SimpleNamespace(
        head_dim=head_dim,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        queries_per_key_value=queries_per_key_value,
        query_norm_weight=torch.ones(head_dim, dtype=torch.bfloat16),
        key_norm_weight=torch.ones(head_dim, dtype=torch.bfloat16),
    )

    gauge, gauge_receipt = adapter._fit_pre_rope_dense_gauge(seed, (case,))
    query_runtime, key_runtime, route_receipt = adapter._cold_route_runtime_targets(
        case,
        seed,
        pre_rope_gauge=gauge,
    )
    query_raw = adapter._cold_projection_preimage(
        query_runtime,
        torch.zeros_like(query_runtime),
        norm_weight=torch.ones(head_dim, dtype=torch.bfloat16),
        norm_epsilon=1e-5,
        cosine=query_cos,
        sine=query_sin,
        role="query",
    )
    key_raw = adapter._cold_projection_preimage(
        key_runtime,
        torch.zeros_like(key_runtime),
        norm_weight=torch.ones(head_dim, dtype=torch.bfloat16),
        norm_epsilon=1e-5,
        cosine=key_cos,
        sine=key_sin,
        role="key",
    )
    query_fit_input = torch.randn(
        1,
        512,
        generator=generator,
        dtype=torch.float64,
    )
    key_fit_input = torch.randn(
        key_length,
        512,
        generator=generator,
        dtype=torch.float64,
    )
    query_weight, _query_ridge = adapter._cold_ridge_projection(
        query_fit_input,
        query_raw.permute(0, 2, 1, 3).reshape(1, -1),
        torch.ones(1, dtype=torch.float64),
        role="q_proj.weight-real-shaped-parent-bound",
    )
    key_weight, _key_ridge = adapter._cold_ridge_projection(
        key_fit_input,
        key_raw.permute(0, 2, 1, 3).reshape(key_length, -1),
        torch.ones(key_length, dtype=torch.float64),
        role="k_proj.weight-real-shaped-parent-bound",
    )
    query_coverage = adapter._dense_axis_coverage(
        query_weight,
        role="q_proj.weight",
    )
    key_coverage = adapter._dense_axis_coverage(
        key_weight,
        role="k_proj.weight",
    )

    assert gauge_receipt["gauge_all_coefficients_nonzero"] is True
    assert gauge_receipt["parent_query_prior_rows_used_for_selection"] > 0
    assert route_receipt["parent_query_prior_used"] is True
    assert route_receipt["parent_key_prior_used"] is True
    assert route_receipt["relative_rope_key_residual_solve"] is True
    assert route_receipt["parent_prior_nullspace_vectors"] > 0
    assert route_receipt["clipped_key_route_vectors"] == 0
    assert route_receipt["maximum_score_constraint_absolute_error"] < 1e-10
    assert route_receipt["maximum_target_probability_absolute_error"] < 1e-10
    assert bool(query_raw.ne(0.0).any(dim=(0, 2)).all().item())
    assert bool(key_raw.ne(0.0).any(dim=(0, 2)).all().item())
    assert query_weight.shape == (512, 512)
    assert key_weight.shape == (128, 512)
    assert query_coverage["structurally_nonzero"] is True
    assert key_coverage["structurally_nonzero"] is True


def test_real_shaped_rank_deficient_fit_uses_exact_nullspace_for_input_axes() -> None:
    generator = torch.Generator().manual_seed(131)
    hidden = 512
    output = 512
    rows = 12
    inputs = torch.randn(rows, hidden, generator=generator, dtype=torch.float64)
    inputs[:, -64:].zero_()
    targets = torch.randn(rows, output, generator=generator, dtype=torch.float64)
    weights = torch.ones(rows, dtype=torch.float64)
    incumbent, _receipt = adapter._cold_ridge_projection(
        inputs,
        targets,
        weights,
        role="q_proj.weight-rank-deficient-fixture",
        normalize_element_rms=False,
    )
    assert bool(incumbent[:, -64:].eq(0.0).all().item())

    repaired, report = adapter._repair_dense_axis_coverage(
        incumbent,
        role="q_proj.weight",
        fit_inputs=inputs,
        fit_targets=targets,
        fit_sqrt_weights=weights,
    )

    nullspace = report["fit_nullspace_completion"]
    assert report["coverage_before"]["all_input_structural_units_covered"] is False
    assert (
        report["coverage_after_fit_ridge"]["all_input_structural_units_covered"]
        is False
    )
    assert report["coverage_after"]["structurally_nonzero"] is True
    assert nullspace is not None
    assert nullspace["design_width"] == hidden
    assert nullspace["numerical_rank"] <= rows
    assert nullspace["nullity"] >= hidden - rows
    assert nullspace["missing_input_axis_count"] == 64
    assert nullspace["completion_lies_in_numerical_fit_design_nullspace"] is True
    assert (
        nullspace["fp64_weighted_fit_prediction_maximum_absolute_change"]
        <= nullspace["fp64_nullspace_tolerance"]
    )
    assert bool(repaired[:, -64:].ne(0.0).all().item())


def test_multi_projection_transaction_matches_wrapper_changed_key_order() -> None:
    model = _Model()
    attention = model.model.layers[0].self_attn
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.out_proj,
        ):
            projection.weight.zero_()
    evidence = _evidence(model)
    state = {
        "q_proj.weight": torch.ones_like(attention.q_proj.weight),
        "k_proj.weight": torch.ones_like(attention.k_proj.weight),
        "v_proj.weight": torch.ones_like(attention.v_proj.weight),
        "out_proj.weight": torch.ones_like(attention.out_proj.weight),
    }

    def install(**kwargs: object) -> dict[str, object]:
        target_model = kwargs["target_model"]
        layer_index = int(kwargs["layer_index"])
        topology = adapter._topology_sha256(target_model)
        before, after, changed, parameter_count = adapter._install_qkvo_transaction(
            target_model,
            layer_index=layer_index,
            state=state,
        )
        body = adapter._installation_receipt_body(
            layer_index=layer_index,
            evidence_sha256=[item.evidence_sha256 for item in evidence],
            model_state_before_sha256=before,
            model_state_after_sha256=after,
            changed_state_keys=changed,
            parameter_element_count=parameter_count,
            target_topology_sha256_before=topology,
            target_topology_sha256_after=adapter._topology_sha256(target_model),
            structurally_eligible_candidate_count=1,
        )
        recipe = adapter.resolve_lfm2_semantic_body_recipe("fit_guided")
        candidate_reports = {
            "incumbent": {"eligible": False},
            "fixed_v_output_only": {"eligible": True},
            "joint_value_output": {"eligible": False},
        }
        gate_body = {
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_component_lane": recipe.component_lanes["value_output"],
            "candidate_reports": candidate_reports,
            "selected_candidate": "fixed_v_output_only",
            "published_candidate": "fixed_v_output_only",
            "structurally_eligible_candidate_count": 1,
            "fit_metrics_used_for_candidate_ordering": True,
            "component_selection_policy_equivalent_recipe_ids": [recipe.recipe_id],
            "component_selection_policy_equivalent_to_other_recipe": False,
        }
        gate = {**gate_body, "gate_sha256": canonical_sha256(gate_body)}
        body["analytic_refold"] = {
            "body_recipe": recipe.as_dict(),
            "body_recipe_sha256": recipe.recipe_sha256,
            "body_recipe_id": recipe.recipe_id,
            "recipe_qk_lane": recipe.component_lanes["qk"],
            "recipe_value_output_lane": recipe.component_lanes["value_output"],
            "value_output_candidate_gate": gate,
        }
        return {**body, "result_sha256": canonical_sha256(body)}

    receipt = stack_adapter._install_attention_operator(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
        installer=install,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )

    assert receipt["changed_state_keys"] == [
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.out_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    assert receipt["changed_state_keys"] == sorted(receipt["changed_state_keys"])
    for projection in (
        attention.q_proj,
        attention.k_proj,
        attention.v_proj,
        attention.out_proj,
    ):
        torch.testing.assert_close(
            projection.weight,
            torch.ones_like(projection.weight),
            rtol=0,
            atol=0,
        )


def test_changed_qkvo_transaction_rolls_back_partial_copy_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    attention = model.model.layers[0].self_attn
    state = {
        "q_proj.weight": attention.q_proj.weight.detach().cpu().clone() + 1,
        "k_proj.weight": attention.k_proj.weight.detach().cpu().clone() + 1,
        "v_proj.weight": attention.v_proj.weight.detach().cpu().clone(),
        "out_proj.weight": attention.out_proj.weight.detach().cpu().clone(),
    }
    before_identity = fingerprint_lfm2_parent(model).identity_sha256
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())
    original_copy = adapter._copy_candidate
    copy_calls = 0

    def fail_second_copy(parameter: object, candidate: object) -> None:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 2:
            raise RuntimeError("injected-second-copy-failure")
        original_copy(parameter, candidate)

    monkeypatch.setattr(adapter, "_copy_candidate", fail_second_copy)
    with pytest.raises(RuntimeError, match="injected-second-copy-failure"):
        adapter._install_qkvo_transaction(
            model,
            layer_index=0,
            state=state,
        )

    assert copy_calls == 2
    assert fingerprint_lfm2_parent(model).identity_sha256 == before_identity
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids


def test_changed_qkvo_transaction_rolls_back_wrong_candidate_copy_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    attention = model.model.layers[0].self_attn
    state = {
        "q_proj.weight": attention.q_proj.weight.detach().cpu().clone() + 1,
        "k_proj.weight": attention.k_proj.weight.detach().cpu().clone() + 1,
        "v_proj.weight": attention.v_proj.weight.detach().cpu().clone(),
        "out_proj.weight": attention.out_proj.weight.detach().cpu().clone(),
    }
    before_identity = fingerprint_lfm2_parent(model).identity_sha256
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())

    def copy_wrong_candidate(
        parameter: torch.nn.Parameter,
        candidate: torch.Tensor,
    ) -> None:
        parameter.copy_(candidate + 7)

    monkeypatch.setattr(adapter, "_copy_candidate", copy_wrong_candidate)
    with pytest.raises(
        adapter.Lfm2SemanticAttentionRefoldError,
        match="did not publish the exact frozen candidate",
    ):
        adapter._install_qkvo_transaction(
            model,
            layer_index=0,
            state=state,
        )

    assert fingerprint_lfm2_parent(model).identity_sha256 == before_identity
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == pointers
    assert tuple(id(parameter.grad) for parameter in model.parameters()) == gradient_ids


def test_incomplete_causal_history_fails_before_the_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    evidence = _evidence(model, complete_history=False)
    called = False

    def compiler(**_kwargs: object) -> object:
        nonlocal called
        called = True
        return SimpleNamespace()

    monkeypatch.setattr(
        adapter,
        "compile_lfm2_cached_endpoint_qk_residual_fold",
        compiler,
    )
    with pytest.raises(
        adapter.Lfm2SemanticAttentionRefoldError,
        match="Complete-causal",
    ):
        adapter.compile_install_lfm2_semantic_attention_postfold(
            target_model=model,
            layer_index=0,
            evidence=evidence,
            cache_key="2" * 64,
            bridge_artifact_sha256="3" * 64,
        )
    assert called is False


def test_replay_endpoint_is_behavioral_residual_not_weight_matching() -> None:
    replay = adapter.Lfm2SemanticAttentionReplayEndpoint(
        batch_index=0,
        layer_index=0,
        alignment_sha256="5" * 64,
        cache_key="6" * 64,
        bridge_artifact_sha256="7" * 64,
        captured_child_model_state_sha256="8" * 64,
        target_query_inputs=torch.ones(1, 1, 4, dtype=torch.bfloat16),
        target_key_value_inputs=torch.ones(1, 2, 4, dtype=torch.bfloat16),
        actual_child_pre_attention_residual=torch.ones(1, 1, 4),
        parent_post_attention_residual_target_space=torch.full((1, 1, 4), 3.0),
        query_row_sqrt_weights=torch.ones(1, 1),
    )

    assert torch.equal(replay.parent_endpoint_target, torch.full((1, 1, 4), 2.0))
    assert (
        replay.to_evidence_dict()["numeric_parent_child_weight_match_objective"]
        is False
    )


def test_replay_phase_weights_are_balanced_per_whole_trajectory() -> None:
    model = _Model()
    evidence = _evidence(model)
    sources = tuple(
        adapter._normalized_case_source(
            item,
            layer_index=0,
            cache_key="2" * 64,
            bridge_artifact_sha256="3" * 64,
        )
        for item in evidence[:2]
    )

    cases, receipt = adapter._balanced_endpoint_cases(
        (sources[0], sources[0], sources[1]),
        layer_index=0,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    first_trajectory_mass = sum(
        float(case.query_row_sqrt_weights.square().sum().item()) for case in cases[:2]
    )
    second_trajectory_mass = float(
        cases[2].query_row_sqrt_weights.square().sum().item()
    )
    assert first_trajectory_mass == pytest.approx(1.0)
    assert second_trajectory_mass == pytest.approx(1.0)
    assert receipt["longer_trajectories_receive_more_total_weight"] is False
    assert receipt["relative_replay_phase_weights_preserved"] is True
    body = dict(receipt)
    assert body.pop("balance_sha256") == canonical_sha256(body)


def _writer_case(
    context: torch.Tensor,
    output_target: torch.Tensor,
    *,
    label: str,
) -> adapter._WriterCase:
    return adapter._WriterCase(
        context=context.to(torch.float64).unsqueeze(0),
        endpoint_target=(context @ output_target.T).to(torch.float64).unsqueeze(0),
        row_weights=torch.ones(1, int(context.shape[0]), dtype=torch.float64),
        trajectory_sha256=canonical_sha256({"writer-case": label}),
    )


def _joint_case_from_factors(
    features: torch.Tensor,
    value_weight: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    label: str,
) -> adapter._JointVoCase:
    rows, query_heads, hidden = map(int, features.shape)
    assert query_heads == 2
    key_inputs = torch.cat(
        tuple(features[:, head_index, :] for head_index in range(query_heads)),
        dim=0,
    ).to(torch.bfloat16)
    probabilities = torch.zeros(
        1,
        query_heads,
        rows,
        rows * query_heads,
        dtype=torch.float64,
    )
    for head_index in range(query_heads):
        row_indices = torch.arange(rows)
        probabilities[0, head_index, row_indices, head_index * rows + row_indices] = 1.0
    contexts = torch.cat(
        tuple(
            (
                features[:, head_index, :].to(torch.bfloat16)
                @ value_weight.to(torch.bfloat16).T
            ).to(torch.float64)
            for head_index in range(query_heads)
        ),
        dim=1,
    )
    endpoint = (contexts @ output_weight.to(torch.float64).T).unsqueeze(0)
    return adapter._JointVoCase(
        head_features=features.to(torch.bfloat16).to(torch.float64).unsqueeze(0),
        probabilities=probabilities,
        key_value_inputs=key_inputs.unsqueeze(0),
        endpoint_target=endpoint,
        row_weights=torch.ones(1, rows, dtype=torch.float64),
        trajectory_sha256=canonical_sha256({"joint-vo-case": label}),
        seed_replay_equivalence=adapter._bfloat16_endpoint_equivalence(
            endpoint,
            endpoint,
        ),
    )


def test_seed_endpoint_equivalence_requires_exact_same_device_replay() -> None:
    expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float64)
    accepted = adapter._bfloat16_endpoint_equivalence(expected.clone(), expected)
    one_ulp = expected.clone()
    one_ulp[0, 0, 0] += torch.finfo(torch.bfloat16).eps
    rejected = adapter._bfloat16_endpoint_equivalence(one_ulp, expected)

    assert accepted["equivalent"] is True
    assert accepted["bitwise_identity_required"] is True
    assert rejected["equivalent"] is False
    accepted_body = dict(accepted)
    assert accepted_body.pop("equivalence_sha256") == canonical_sha256(accepted_body)


def test_joint_vo_prediction_matches_live_lfm2_bfloat16_macrocell() -> None:
    transformers = pytest.importorskip("transformers")
    import neural_forge.hf_lfm2_cpcf_macrocell_replay as macrocell_replay

    config = transformers.Lfm2Config(
        vocab_size=37,
        hidden_size=8,
        intermediate_size=13,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["full_attention"],
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260723)
        model = transformers.Lfm2ForCausalLM(config)
    model.to(dtype=torch.bfloat16)
    model.eval()
    layer = model.model.layers[0]
    with torch.no_grad():
        residual = model.model.embed_tokens(
            torch.tensor(((1, 2, 3, 4, 5),), dtype=torch.long)
        ).detach()
        positions = torch.arange(5, dtype=torch.long).unsqueeze(0)
        cosine, sine = model.model.rotary_emb(residual, position_ids=positions)
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    additive_mask = torch.zeros(
        (1, 1, 5, 5),
        dtype=torch.bfloat16,
    ).masked_fill(future, torch.finfo(torch.bfloat16).min)
    live = macrocell_replay.replay_lfm2_cpcf_macrocell(
        layer,
        residual,
        position_embeddings=(cosine, sine),
        additive_causal_mask=additive_mask,
    )
    live_output, details, _topology = macrocell_replay._attention_replay(
        layer,
        live.operator_input,
        position_embeddings=(cosine, sine),
        additive_causal_mask=additive_mask,
    )
    assert torch.equal(live_output, live.operator_delta)

    probabilities = details["probabilities"]
    key_inputs = live.operator_input
    head_features = (
        torch.einsum(
            "bhqk,bkf->bqhf",
            probabilities.to(torch.float64),
            key_inputs.to(torch.float64),
        )
        .detach()
        .contiguous()
    )
    case = adapter._JointVoCase(
        head_features=head_features,
        probabilities=probabilities.detach().to(torch.float64),
        key_value_inputs=key_inputs.detach(),
        endpoint_target=live.operator_delta.detach().to(torch.float64),
        row_weights=torch.ones((1, 5), dtype=torch.float64),
        trajectory_sha256=canonical_sha256({"joint-vo": "live-macrocell"}),
        seed_replay_equivalence=adapter._bfloat16_endpoint_equivalence(
            live.operator_delta,
            live.operator_delta,
        ),
    )
    attention = layer.self_attn
    predicted = adapter._joint_vo_prediction(
        case,
        attention.v_proj.weight,
        attention.out_proj.weight,
        head_dim=4,
    )
    assert predicted.dtype == torch.bfloat16
    assert torch.equal(predicted, live.operator_delta)

    seed = SimpleNamespace(
        value_weight=attention.v_proj.weight.detach().clone(),
        key_value_heads=1,
        head_dim=4,
        queries_per_key_value=2,
        query_heads=2,
    )
    writer_case = adapter._writer_cases_from_joint((case,), seed)[0]
    assert torch.equal(
        writer_case.context.to(torch.bfloat16),
        live.operator_output_input,
    )


def test_joint_vo_dual_fold_recovers_a_missing_value_subspace() -> None:
    generator = torch.Generator().manual_seed(29)
    features = torch.randint(
        -2,
        3,
        (24, 2, 4),
        generator=generator,
    ).to(torch.float64)
    incumbent_value = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    target_value = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.bfloat16,
    )
    output = torch.eye(4, dtype=torch.bfloat16)
    case = _joint_case_from_factors(
        features,
        target_value,
        output,
        label="missing-value-subspace",
    )
    proposal = adapter._compile_joint_vo(
        (case,),
        incumbent_value,
        output,
        head_dim=2,
        solver_device="cpu",
        ridge=1e-10,
    )

    assert proposal.changed is True
    assert proposal.candidate_fit_metrics.endpoint_relative_rmse < 0.01
    assert (
        proposal.candidate_fit_metrics.endpoint_relative_rmse
        < proposal.baseline_fit_metrics.endpoint_relative_rmse * 0.02
    )
    assert not torch.equal(proposal.value_weight, incumbent_value)
    assert tuple(proposal.value_weight.shape) == tuple(incumbent_value.shape)
    assert tuple(proposal.output_weight.shape) == tuple(output.shape)
    report = proposal.as_dict()
    assert report["giant_primal_system_formed"] is False
    assert report["giant_primal_system_inverted"] is False
    assert report["dual_dimension"] == 24
    assert report["primal_feature_dimension"] == 8
    assert report["diagnostic_only"] is False
    assert report["publication_allowed"] is True
    assert report["publication_lane"] == (
        "structural_fit_ordered_value_output_candidates"
    )
    assert report["factorized_candidate_published_in_proposal"] is True
    assert report["numeric_metric_used_for_publication"] is False
    factor = report["factorization_receipts"][0]
    assert factor["exact_topology_rank"] == 2
    assert "rank_truncation_performed" in factor
    assert report["operation_audit"]["model_forward_calls"] == 0


def test_vo_competition_orders_all_structural_candidates_by_fit_only() -> None:
    generator = torch.Generator().manual_seed(31)
    features = torch.randint(
        -2,
        3,
        (24, 2, 4),
        generator=generator,
    ).to(torch.float64)
    value = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    incumbent_output = torch.eye(4, dtype=torch.bfloat16)
    target_output = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 1.75, 0.0],
            [0.0, 0.0, 0.0, 1.25],
        ],
        dtype=torch.bfloat16,
    )
    joint_case = _joint_case_from_factors(
        features,
        value,
        target_output,
        label="output-only-target",
    )
    context = torch.cat(
        tuple(
            (features[:, head_index, :].to(torch.bfloat16) @ value.T).to(torch.float64)
            for head_index in range(2)
        ),
        dim=1,
    )
    writer_case = _writer_case(context, target_output.to(torch.float64), label="vo")
    output_proposal = adapter._compile_output_writer(
        (writer_case,),
        incumbent_output,
        solver_device="cpu",
        ridge=1e-10,
    )
    joint_proposal = adapter._compile_joint_vo(
        (joint_case,),
        value,
        incumbent_output,
        head_dim=2,
        solver_device="cpu",
        ridge=1e-10,
    )
    seed = SimpleNamespace(
        value_weight=value,
        output_weight=incumbent_output,
        head_dim=2,
        seed_sha256="a" * 64,
    )
    published_value, published_output, gate = adapter._select_and_validate_vo_candidate(
        banks={name: (joint_case,) for name in adapter._PARTITIONS},
        seed=seed,
        output_proposal=output_proposal,
        joint_proposal=joint_proposal,
        tolerance=1e-6,
        validation_route_baseline=_metrics(0.1),
        validation_route_candidate=_metrics(0.1),
    )

    assert output_proposal.changed is True
    reports = gate["candidate_reports"]
    eligible = ("fixed_v_output_only", "joint_value_output")

    def fit_key(name: str) -> tuple[object, ...]:
        metrics = reports[name]["partitions"]["fit"]["candidate"]
        return (
            metrics["endpoint_relative_rmse"],
            metrics["endpoint_rmse"],
            metrics["endpoint_maximum_absolute_error"],
            -metrics["endpoint_cosine_similarity_mean"],
            -metrics["endpoint_cosine_similarity_worst"],
            reports[name]["proposal_sha256"],
        )

    expected = min(eligible, key=fit_key)
    assert all(reports[name]["eligible"] is True for name in eligible)
    assert gate["structurally_eligible_candidate_count"] == 2
    assert gate["selected_candidate"] == expected
    assert gate["published_candidate"] == expected
    assert gate["fit_metrics_used_for_candidate_ordering"] is True
    assert gate["fit_metrics_used_for_structural_eligibility"] is False
    assert gate["fit_metric_threshold_used_for_candidate_selection"] is False
    assert gate["numeric_metric_used_for_candidate_selection"] is True
    assert gate["validation_triggered_refit_or_reselection"] is False
    assert gate["alternate_candidate_selected_after_validation"] is False
    if expected == "fixed_v_output_only":
        assert torch.equal(published_value, value)
        assert torch.equal(published_output, output_proposal.output_weight)
    else:
        assert torch.equal(published_value, joint_proposal.value_weight)
        assert torch.equal(published_output, joint_proposal.output_weight)


def test_vo_structural_eligibility_is_the_exact_nonmetric_conjunction() -> None:
    incumbent_value = torch.zeros(2, 4, dtype=torch.bfloat16)
    incumbent_output = torch.zeros(4, 4, dtype=torch.bfloat16)
    valid_value = incumbent_value.clone()
    valid_value[0, 0] = 1.0

    eligible = adapter._vo_structural_eligibility(
        value_weight=valid_value,
        output_weight=incumbent_output.clone(),
        incumbent_value_weight=incumbent_value,
        incumbent_output_weight=incumbent_output,
    )
    identical = adapter._vo_structural_eligibility(
        value_weight=incumbent_value.clone(),
        output_weight=incumbent_output.clone(),
        incumbent_value_weight=incumbent_value,
        incumbent_output_weight=incumbent_output,
    )
    nonfinite = valid_value.clone()
    nonfinite[0, 0] = torch.nan
    invalid_rows = {
        "nonfinite": adapter._vo_structural_eligibility(
            value_weight=nonfinite,
            output_weight=incumbent_output,
            incumbent_value_weight=incumbent_value,
            incumbent_output_weight=incumbent_output,
        ),
        "dtype": adapter._vo_structural_eligibility(
            value_weight=valid_value.float(),
            output_weight=incumbent_output,
            incumbent_value_weight=incumbent_value,
            incumbent_output_weight=incumbent_output,
        ),
        "shape": adapter._vo_structural_eligibility(
            value_weight=valid_value[:, :3],
            output_weight=incumbent_output,
            incumbent_value_weight=incumbent_value,
            incumbent_output_weight=incumbent_output,
        ),
        "layout": adapter._vo_structural_eligibility(
            value_weight=valid_value.to_sparse(),
            output_weight=incumbent_output,
            incumbent_value_weight=incumbent_value,
            incumbent_output_weight=incumbent_output,
        ),
    }

    assert eligible["eligible"] is True
    assert eligible["numeric_metric_used_for_eligibility"] is False
    assert identical["effectful"] is False
    assert identical["eligible"] is False
    assert all(row["eligible"] is False for row in invalid_rows.values())
    assert invalid_rows["nonfinite"]["finite"] is False
    assert invalid_rows["dtype"]["bfloat16"] is False
    assert invalid_rows["shape"]["exact_shape"] is False
    assert invalid_rows["layout"]["dense_strided"] is False


@pytest.mark.parametrize(
    "body_recipe",
    ("fit_guided", "full_strength_structural", "residual_writer"),
)
def test_vo_incumbent_is_used_only_when_proposals_are_invalid_or_identical(
    body_recipe: str,
) -> None:
    generator = torch.Generator().manual_seed(41)
    features = torch.randint(-2, 3, (8, 2, 4), generator=generator).to(torch.float64)
    value = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    output = torch.eye(4, dtype=torch.bfloat16)
    case = _joint_case_from_factors(
        features,
        value,
        output,
        label="invalid-or-identical-fallback",
    )
    seed = SimpleNamespace(
        value_weight=value,
        output_weight=output,
        head_dim=2,
        seed_sha256="a" * 64,
    )
    invalid_writer = SimpleNamespace(
        output_weight=output[:, :3].contiguous(),
        changed=True,
        proposal_sha256="b" * 64,
    )
    identical_joint = SimpleNamespace(
        value_weight=value.clone(),
        output_weight=output.clone(),
        changed=False,
        proposal_sha256="c" * 64,
    )

    published_value, published_output, gate = adapter._select_and_validate_vo_candidate(
        banks={partition: (case,) for partition in adapter._PARTITIONS},
        seed=seed,
        output_proposal=invalid_writer,
        joint_proposal=identical_joint,
        tolerance=0.0,
        validation_route_baseline=_metrics(0.1),
        validation_route_candidate=_metrics(0.9),
        body_recipe=body_recipe,
    )

    assert gate["structurally_eligible_candidate_count"] == 0
    assert gate["published_candidate"] == "incumbent"
    assert gate["incumbent_noop_fallback_used"] is True
    assert gate["body_recipe_id"] == body_recipe
    assert gate["incumbent_noop_fallback_reason"] == (
        "all-analytic-proposals-structurally-invalid-or-identical"
    )
    assert gate["candidate_reports"]["fixed_v_output_only"]["exact_shape"] is False
    assert gate["candidate_reports"]["fixed_v_output_only"]["effectful"] is True
    assert gate["candidate_reports"]["joint_value_output"]["effectful"] is False
    assert torch.equal(published_value, value)
    assert torch.equal(published_output, output)


@pytest.mark.parametrize(
    "body_recipe",
    ("fit_guided", "full_strength_structural", "residual_writer"),
)
def test_vo_heldout_metric_regression_cannot_veto_frozen_fit_selection(
    body_recipe: str,
) -> None:
    generator = torch.Generator().manual_seed(37)
    features = torch.randint(-2, 3, (24, 2, 4), generator=generator).to(torch.float64)
    incumbent_value = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    target_value = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.bfloat16,
    )
    output = torch.eye(4, dtype=torch.bfloat16)
    fit_case = _joint_case_from_factors(
        features,
        target_value,
        output,
        label="heldout-regression-fit",
    )
    contradictory_heldout = _joint_case_from_factors(
        features,
        incumbent_value,
        output,
        label="heldout-regression-validation",
    )
    incumbent_context = torch.cat(
        tuple(
            (features[:, head_index, :].to(torch.bfloat16) @ incumbent_value.T).to(
                torch.float64
            )
            for head_index in range(2)
        ),
        dim=1,
    )
    writer_case = adapter._WriterCase(
        context=incumbent_context.unsqueeze(0),
        endpoint_target=fit_case.endpoint_target,
        row_weights=torch.ones(1, int(features.shape[0]), dtype=torch.float64),
        trajectory_sha256=canonical_sha256({"writer-case": "heldout-regression"}),
    )
    output_proposal = adapter._compile_output_writer(
        (writer_case,),
        output,
        solver_device="cpu",
        ridge=1e-10,
    )
    joint_proposal = adapter._compile_joint_vo(
        (fit_case,),
        incumbent_value,
        output,
        head_dim=2,
        solver_device="cpu",
        ridge=1e-10,
    )
    seed = SimpleNamespace(
        value_weight=incumbent_value,
        output_weight=output,
        head_dim=2,
        seed_sha256="a" * 64,
    )

    published_value, published_output, gate = adapter._select_and_validate_vo_candidate(
        banks={
            "fit": (fit_case,),
            "fit_check": (contradictory_heldout,),
            "selection": (contradictory_heldout,),
            "validation": (contradictory_heldout,),
        },
        seed=seed,
        output_proposal=output_proposal,
        joint_proposal=joint_proposal,
        tolerance=0.0,
        validation_route_baseline=_metrics(0.1),
        validation_route_candidate=_metrics(0.9),
        body_recipe=body_recipe,
    )

    selected = gate["published_candidate"]
    assert selected != "incumbent"
    assert gate["body_recipe_id"] == body_recipe
    if body_recipe == "full_strength_structural":
        assert selected == "joint_value_output"
    elif body_recipe == "residual_writer":
        assert selected == "fixed_v_output_only"
    assert gate["validation"]["endpoint_nonregressing"] is False
    assert gate["validation"]["route_nonregressing"] is False
    assert gate["validation"]["diagnostic_only"] is True
    assert gate["validation_publication_veto_applied"] is False
    assert gate["selected_correction_rejected_on_validation"] is False
    assert gate["rollback_authorized"] is False
    if selected == "joint_value_output":
        assert torch.equal(published_value, joint_proposal.value_weight)
        assert torch.equal(published_output, joint_proposal.output_weight)
    else:
        assert torch.equal(published_value, incumbent_value)
        assert torch.equal(published_output, output_proposal.output_weight)


def test_output_writer_closed_form_fit_passes_all_frozen_heldout_gates() -> None:
    incumbent = torch.zeros(4, 4, dtype=torch.bfloat16)
    target = torch.tensor(
        [
            [2.0, 0.5, 0.0, -0.25],
            [0.0, 1.5, -0.5, 0.0],
            [0.25, 0.0, 1.0, 0.5],
            [-0.5, 0.25, 0.0, 1.25],
        ],
        dtype=torch.float64,
    )
    fit = _writer_case(torch.eye(4, dtype=torch.float64), target, label="fit")
    fit_check = _writer_case(
        torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
        target,
        label="fit-check",
    )
    selection = _writer_case(
        torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        target,
        label="selection",
    )
    validation = _writer_case(
        torch.tensor([[1.0, -1.0, 1.0, -1.0]], dtype=torch.float64),
        target,
        label="validation",
    )

    proposal = adapter._compile_output_writer((fit,), incumbent, solver_device="cpu")
    published, gate = adapter._gate_output_writer(
        proposal,
        fit_check_cases=(fit_check,),
        selection_cases=(selection,),
        validation_cases=(validation,),
        incumbent_output_weight=incumbent,
        tolerance=1e-8,
    )

    assert proposal.changed is True
    assert proposal.candidate_fit_metrics.endpoint_relative_rmse < 0.001
    assert gate["accepted"] is True
    assert gate["candidate_frozen_across_heldout_gates"] is True
    assert not torch.equal(published, incumbent)
    gate_body = dict(gate)
    assert gate_body.pop("gate_sha256") == canonical_sha256(gate_body)


def test_output_writer_heldout_regression_is_diagnostic_only() -> None:
    incumbent = torch.zeros(4, 4, dtype=torch.bfloat16)
    target = torch.eye(4, dtype=torch.float64)
    fit = _writer_case(torch.eye(4, dtype=torch.float64), target, label="fit")
    proposal = adapter._compile_output_writer((fit,), incumbent, solver_device="cpu")
    contradictory = adapter._WriterCase(
        context=torch.eye(4, dtype=torch.float64).unsqueeze(0),
        endpoint_target=(-torch.eye(4, dtype=torch.float64)).unsqueeze(0),
        row_weights=torch.ones(1, 4, dtype=torch.float64),
        trajectory_sha256=canonical_sha256({"writer-case": "contradictory"}),
    )

    published, gate = adapter._gate_output_writer(
        proposal,
        fit_check_cases=(contradictory,),
        selection_cases=(fit,),
        validation_cases=(fit,),
        incumbent_output_weight=incumbent,
        tolerance=1e-8,
    )

    assert proposal.changed is True
    assert gate["accepted"] is True
    assert gate["rejection_partition"] is None
    assert gate["fit_check"]["nonregressing"] is False
    assert gate["fit_check"]["diagnostic_only"] is True
    assert gate["selection"] is not None
    assert gate["validation"] is not None
    assert gate["numeric_metric_used_for_publication"] is False
    assert torch.equal(published, proposal.output_weight)


def test_execution_policy_excludes_visible_devices_without_native_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    capabilities = ((8, 6), (6, 1), (7, 5), (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capabilities.__getitem__)
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("wrong API must not be called")
        ),
    )

    policy = adapter._execution_policy(8)

    assert policy.solver_device == "cuda:0"
    assert policy.compute_devices == ("cuda:0", "cuda:3")


def test_execution_policy_falls_back_to_cpu_without_native_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (7, 5))

    policy = adapter._execution_policy(8)

    assert policy.solver_device == "cpu"
    assert policy.compute_devices == ("cpu",)
