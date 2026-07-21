from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import neural_forge.hf_lfm2_semantic_attention_refold_adapter as adapter  # noqa: E402
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
    assert report["analytically_refit_projection_roles"] == ["q_proj", "k_proj"]
    assert report["fixed_inherited_projection_roles"] == ["v_proj", "out_proj"]
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

    def accept_writer(
        proposal: adapter._OutputWriterProposal,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, object]:
        assert proposal.changed is True
        body = {
            "status": "accepted-after-frozen-validation",
            "accepted": True,
            "proposal": proposal.as_dict(),
            "rejection_partition": None,
            "fit_check": {"nonregressing": True},
            "selection": {"nonregressing": True},
            "validation": {"nonregressing": True},
            "candidate_frozen_across_heldout_gates": True,
        }
        return proposal.output_weight, adapter._seal_writer_gate(body)

    monkeypatch.setattr(adapter, "_gate_output_writer", accept_writer)
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
        "k_proj",
        "out_proj",
    ]
    assert report["fixed_inherited_projection_roles"] == ["v_proj"]
    assert report["fresh_value_refit_performed"] is False
    assert report["complete_qkvo_bundle_installed_transactionally"] is True


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
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda index: index in {0, 3},
    )

    policy = adapter._execution_policy(8)

    assert policy.solver_device == "cuda:0"
    assert policy.compute_devices == ("cuda:0", "cuda:3")


def test_execution_policy_falls_back_to_cpu_without_native_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda _index: False)

    policy = adapter._execution_policy(8)

    assert policy.solver_device == "cpu"
    assert policy.compute_devices == ("cpu",)
