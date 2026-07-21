from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_attention_refold_adapter as adapter  # noqa: E402
import neural_forge.hf_lfm2_semantic_layer_stack_refold_adapter as stack_adapter  # noqa: E402
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
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _Backbone()

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
    ) -> None:
        self.batch_index = batch_index
        self.layer_index = 0
        self.alignment = alignment
        self.alignment_sha256 = alignment.alignment_sha256
        self.transport = transport
        self.actual_child_replay = replay
        self.source_inventory = source_inventory
        self.query_groups = ((0, 1), (2, 3))
        self.query_group_weights = ((0.5, 0.5), (0.5, 0.5))
        self.compiler_lineage_sha256 = "9" * 64
        capability_sha = canonical_sha256(
            {"batch": batch_index, "layer": 0, "kind": "capability"}
        )
        self.capability = {
            "capability_sha256": capability_sha,
            "layer_index": 0,
            "exact_history_supported": complete_history,
            "blocking_reasons": [] if complete_history else ["missing"],
            "missing_donor_prefix_positions": [],
        }
        self.history = {
            "history_sha256": canonical_sha256(
                {"batch": batch_index, "layer": 0, "kind": "history"}
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


def _evidence(model: _Model, *, complete_history: bool = True) -> tuple[_Evidence, ...]:
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
            layer_index=0,
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


def test_closed_form_four_way_gate_installs_only_after_frozen_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    assert receipt["schema"] == "neural-forge-lfm2-semantic-attention-installation-v1"
    assert receipt["changed_state_keys"] == ["model.layers.0.self_attn.q_proj.weight"]
    assert not torch.equal(model.model.layers[0].self_attn.q_proj.weight, before)
    report = receipt["analytic_refold"]
    assert report["candidate_frozen_before_validation"] is True
    assert report["validation_triggered_refit"] is False
    assert report["numeric_parent_child_weight_match_objective"] is False
    assert report["analytically_refit_projection_roles"] == ["q_proj"]
    assert report["fixed_inherited_projection_roles"] == [
        "k_proj",
        "v_proj",
        "out_proj",
    ]
    assert report["fresh_query_refit_performed"] is True
    assert report["fresh_key_refit_performed"] is False
    assert report["fresh_four_matrix_qkvo_refit_claimed"] is False
    assert report["fresh_output_writer_refit_performed"] is False
    assert report["output_writer"]["accepted"] is False
    assert report["output_writer"]["rejection_partition"] in {
        "fit",
        "fit_check",
        "selection",
        "validation",
    }
    assert report["inserted_target_query_rows_parent_exact"] is False
    assert report["refold_operation_audit"]["model_forward_calls"] == 0


def test_validation_regression_prevents_any_parameter_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    evidence = _evidence(model)
    _install_fake_compiler(monkeypatch, validation_regresses=True)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(
        adapter.Lfm2SemanticAttentionRefoldError,
        match="failed VALIDATION",
    ):
        adapter.compile_install_lfm2_semantic_attention_postfold(
            target_model=model,
            layer_index=0,
            evidence=evidence,
            cache_key="2" * 64,
            bridge_artifact_sha256="3" * 64,
        )

    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )


def test_accepted_output_writer_is_installed_in_the_same_qkvo_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    evidence = _evidence(model)
    _install_fake_compiler(monkeypatch)
    before_query = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    before_output = model.model.layers[0].self_attn.out_proj.weight.detach().clone()

    def accept_writer_competition(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        proposal = kwargs["output_proposal"]
        seed = kwargs["seed"]
        assert proposal.changed is True
        body = {
            "published_candidate": "fixed_v_output_only",
            "candidate_reports": {
                "fixed_v_output_only": {
                    "rejection_partition": None,
                    "partitions": {
                        "fit_check": {"nonregressing": True},
                        "selection": {"nonregressing": True},
                    },
                }
            },
            "validation": {"endpoint_nonregressing": True},
        }
        return seed.value_weight, proposal.output_weight, body

    monkeypatch.setattr(
        adapter,
        "_select_and_validate_vo_candidate",
        accept_writer_competition,
    )
    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    assert receipt["changed_state_keys"] == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.out_proj.weight",
    ]
    assert not torch.equal(model.model.layers[0].self_attn.q_proj.weight, before_query)
    assert not torch.equal(
        model.model.layers[0].self_attn.out_proj.weight, before_output
    )
    report = receipt["analytic_refold"]
    assert report["fresh_output_writer_refit_performed"] is True
    assert report["analytically_refit_projection_roles"] == [
        "q_proj",
        "out_proj",
    ]
    assert report["fixed_inherited_projection_roles"] == ["k_proj", "v_proj"]
    assert report["fresh_value_refit_performed"] is False
    assert report["complete_qkvo_bundle_installed_transactionally"] is True


def test_accepted_joint_vo_is_installed_in_the_same_qkvo_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    evidence = _evidence(model)
    _install_fake_compiler(monkeypatch)
    before_value = model.model.layers[0].self_attn.v_proj.weight.detach().clone()
    before_output = model.model.layers[0].self_attn.out_proj.weight.detach().clone()

    def accept_joint_competition(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        seed = kwargs["seed"]
        value = seed.value_weight.clone()
        output = seed.output_weight.clone()
        value[0, 0] += 1.0
        output[0, 0] += 1.0
        return (
            value,
            output,
            {
                "published_candidate": "joint_value_output",
                "candidate_reports": {
                    "fixed_v_output_only": {
                        "rejection_partition": "selection",
                        "partitions": {
                            "fit_check": {"nonregressing": True},
                            "selection": {"nonregressing": True},
                        },
                    }
                },
                "validation": {"endpoint_nonregressing": True},
            },
        )

    monkeypatch.setattr(
        adapter,
        "_select_and_validate_vo_candidate",
        accept_joint_competition,
    )
    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    assert not torch.equal(model.model.layers[0].self_attn.v_proj.weight, before_value)
    assert not torch.equal(
        model.model.layers[0].self_attn.out_proj.weight, before_output
    )
    assert "model.layers.0.self_attn.v_proj.weight" in receipt["changed_state_keys"]
    assert "model.layers.0.self_attn.out_proj.weight" in receipt["changed_state_keys"]
    report = receipt["analytic_refold"]
    assert report["joint_value_output_candidate_published"] is True
    assert report["fresh_joint_value_output_refit_performed"] is True
    assert report["fresh_value_refit_performed"] is True
    assert report["fresh_output_writer_refit_performed"] is True
    assert report["fresh_four_matrix_qkvo_refit_claimed"] is False
    assert report["fixed_inherited_projection_roles"] == ["k_proj"]


def test_qk_seed_fallback_can_publish_a_joint_v_o_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    evidence = _evidence(model)
    _install_fake_noop_qk_compiler(monkeypatch)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    reached_competition = False

    def accept_joint_competition(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        nonlocal reached_competition
        reached_competition = True
        seed = kwargs["seed"]
        value = seed.value_weight.clone()
        output = seed.output_weight.clone()
        value[0, 0] += 1.0
        output[0, 0] += 1.0
        return (
            value,
            output,
            {
                "published_candidate": "joint_value_output",
                "candidate_reports": {
                    "fixed_v_output_only": {
                        "rejection_partition": "selection",
                        "partitions": {
                            "fit_check": {"nonregressing": True},
                            "selection": {"nonregressing": True},
                        },
                    }
                },
                "validation": {"endpoint_nonregressing": True},
            },
        )

    monkeypatch.setattr(
        adapter,
        "_select_and_validate_vo_candidate",
        accept_joint_competition,
    )
    receipt = adapter.compile_install_lfm2_semantic_attention_postfold(
        target_model=model,
        layer_index=0,
        evidence=evidence,
        cache_key="2" * 64,
        bridge_artifact_sha256="3" * 64,
    )

    assert reached_competition is True
    assert receipt["changed_state_keys"] == [
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.out_proj.weight",
    ]
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
        if name
        not in {
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.out_proj.weight",
        }
    )
    report = receipt["analytic_refold"]
    assert report["qk_compiler_accepted"] is False
    assert report["qk_compiler_changed"] is False
    assert report["authenticated_qk_seed_fallback_entered_vo_competition"] is True
    assert report["joint_value_output_candidate_published"] is True
    assert report["analytically_refit_projection_roles"] == ["v_proj", "out_proj"]
    assert report["fixed_inherited_projection_roles"] == [
        "q_proj",
        "k_proj",
    ]


def test_total_qkvo_noop_returns_authenticated_receipt_without_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    evidence = _evidence(model)
    _install_fake_noop_qk_compiler(monkeypatch)
    before_identity = fingerprint_lfm2_parent(model).identity_sha256
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    identities = tuple(id(parameter) for parameter in model.parameters())
    pointers = tuple(parameter.data_ptr() for parameter in model.parameters())
    gradient_ids = tuple(id(parameter.grad) for parameter in model.parameters())
    reached_competition = False

    def retain_incumbent(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        nonlocal reached_competition
        reached_competition = True
        seed = kwargs["seed"]
        return (
            seed.value_weight,
            seed.output_weight,
            {
                "published_candidate": "incumbent",
                "candidate_reports": {
                    "incumbent": {
                        "eligible_after_fit_fit_check_selection": True,
                        "partitions": {
                            "fit": {"nonregressing": True},
                            "fit_check": {"nonregressing": True},
                            "selection": {"nonregressing": True},
                        },
                    },
                    "fixed_v_output_only": {
                        "rejection_partition": "fit",
                        "partitions": {
                            "fit_check": {"nonregressing": True},
                            "selection": {"nonregressing": True},
                        },
                    }
                },
                "selected_candidate": "incumbent",
                "selected_candidate_frozen_before_validation": True,
                "validation_triggered_refit_or_reselection": False,
                "validation": {
                    "endpoint_nonregressing": True,
                    "route_nonregressing": True,
                },
            },
        )

    monkeypatch.setattr(
        adapter,
        "_select_and_validate_vo_candidate",
        retain_incumbent,
    )
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

    assert reached_competition is True
    assert receipt["status"] == "heldout-selected-incumbent-noop"
    assert receipt["installed"] is False
    assert receipt["install_count"] == 0
    assert receipt["changed_state_keys"] == []
    assert receipt["model_state_before_sha256"] == before_identity
    assert receipt["model_state_after_sha256"] == before_identity
    assert receipt["target_topology_sha256_before"] == receipt[
        "target_topology_sha256_after"
    ]
    assert receipt["parameter_identities_preserved"] is True
    assert receipt["parameter_storage_preserved"] is True
    assert receipt["gradient_state_preserved"] is True
    assert receipt["exact_noop_state_proven"] is True
    assert receipt["incumbent_selected_by_heldout_gates"] is True
    assert receipt["fit_check_nonregression_passed"] is True
    assert receipt["selection_nonregression_passed"] is True
    assert receipt["validation_nonregression_passed"] is True
    report = receipt["analytic_refold"]
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
    assert stack_adapter._authenticated_heldout_incumbent_noop(
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
    factor = report["factorization_receipts"][0]
    assert factor["exact_topology_rank"] == 2
    assert "rank_truncation_performed" in factor
    assert report["operation_audit"]["model_forward_calls"] == 0


def test_vo_competition_prefers_fixed_value_writer_when_joint_is_not_stronger() -> None:
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
    assert gate["selected_candidate"] == "fixed_v_output_only"
    assert gate["published_candidate"] == "fixed_v_output_only"
    assert gate["validation_triggered_refit_or_reselection"] is False
    assert gate["alternate_candidate_selected_after_validation"] is False
    assert torch.equal(published_value, value)
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


def test_output_writer_heldout_regression_explicitly_rejects_candidate() -> None:
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
    assert gate["accepted"] is False
    assert gate["rejection_partition"] == "fit_check"
    assert gate["selection"] is None
    assert gate["validation"] is None
    assert torch.equal(published, incumbent)


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
