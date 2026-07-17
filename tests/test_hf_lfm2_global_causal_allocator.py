from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn
pytest.importorskip("transformers")

from neural_forge import (  # noqa: E402
    hf_lfm2_cached_causal_evidence as cached_causal_module,
)
from experiments import build_lfm2_cached_causal_evidence as producer  # noqa: E402
from experiments import (  # noqa: E402
    finalize_lfm2_cached_causal_merge_canonical as canonical_merge,
)
from experiments import run_lfm2_global_causal_allocator as runner  # noqa: E402
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_runtime_state,
    fingerprint_model_state,
    run_fold_only_callback,
)
from neural_forge.hf_lfm2_cached_causal_evidence import (  # noqa: E402
    Lfm2CachedCausalEvidenceError,
    build_lfm2_cached_causal_split,
    build_lfm2_causal_physical_source_evidence,
    generate_lfm2_full_prefix_rollouts,
    lfm2_large_target_width_frontier,
    seal_lfm2_cached_causal_frontier_document,
    verify_lfm2_cached_causal_frontier_document,
    verify_lfm2_full_prefix_child_trace_cache,
)
from neural_forge.hf_lfm2_canonical_frontier_replay import (  # noqa: E402
    FINALIZER_SCHEMA as CANONICAL_FINALIZER_SCHEMA,
    REPLAY_SCHEMA as CANONICAL_REPLAY_SCHEMA,
    Lfm2CanonicalFrontierReplayError,
    replay_lfm2_cached_causal_frontier_from_canonical_packs,
)
from neural_forge.hf_lfm2_canonical_proposal_pack import (  # noqa: E402
    Lfm2CanonicalProposalPack,
    load_lfm2_canonical_proposal_pack,
    write_lfm2_canonical_proposal_pack,
)
from neural_forge.hf_lfm2_global_causal_allocator import (  # noqa: E402
    CANDIDATE_SCHEMA,
    Lfm2CausalDamage,
    Lfm2CausalDamageWeights,
    Lfm2CausalRemovalCandidate,
    Lfm2GlobalCausalExecutionError,
    Lfm2GlobalCausalInfeasible,
    Lfm2GlobalCausalStageDecision,
    audit_lfm2_parent_forward_calls,
    execute_lfm2_global_causal_plan,
    lfm2_global_causal_budget,
    plan_lfm2_global_causal_target,
    prepare_lfm2_causal_ffn_proposal,
    prepare_lfm2_causal_ffn_proposal_bundle,
    seal_lfm2_global_causal_evidence,
)
from neural_forge.hf_lfm2_interaction_beam_allocator import (  # noqa: E402
    MATERIALIZATION_SCHEMA,
    Lfm2CumulativeInteractionContext,
    Lfm2InteractionAcceptanceThresholds,
    Lfm2InteractionBeamExecutionError,
    Lfm2InteractionBeamInfeasible,
    Lfm2InteractionMaterialization,
    bind_lfm2_interaction_beam_plan,
    enumerate_lfm2_interaction_beam_plan,
    evaluate_lfm2_interaction_beam,
    materialize_lfm2_canonical_multi_pack_allocation,
    plan_lfm2_global_causal_target_from_canonical_packs,
    seal_lfm2_global_causal_evidence_from_canonical_packs,
    seal_lfm2_cumulative_interaction_quality,
    verify_lfm2_interaction_beam_plan,
)
from neural_forge.hf_lfm2_variable_width_fold import (  # noqa: E402
    Lfm2LayerWidthCurve,
    Lfm2WidthCostPoint,
    install_lfm2_variable_width_fold_plan,
    plan_lfm2_variable_width_fold_for_widths,
    seal_lfm2_variable_width_evidence,
    verify_lfm2_variable_width_fixed_plan_bundle,
)


_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"
_LAYERS = 14
_HIDDEN = 4
_INTERMEDIATE = 8


class Lfm2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float64))


class Lfm2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=torch.float64,
        )
        self.w3 = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=torch.float64,
        )
        self.w2 = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            dtype=torch.float64,
        )


class Lfm2Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.float64,
        )
        self.q_layernorm = Lfm2RMSNorm(config.head_dim)
        self.k_layernorm = Lfm2RMSNorm(config.head_dim)


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.is_attention_layer = True
        self.self_attn = Lfm2Attention(config, layer_idx)
        self.feed_forward = Lfm2MLP(
            config.hidden_size,
            config.intermediate_size,
        )
        self.operator_norm = Lfm2RMSNorm(config.hidden_size)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size)


for _class in (Lfm2RMSNorm, Lfm2MLP, Lfm2Attention, Lfm2DecoderLayer):
    _class.__module__ = _LFM2_MODULE


class _Body(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=torch.float64,
        )
        self.layers = nn.ModuleList(
            [Lfm2DecoderLayer(config, index) for index in range(_LAYERS)]
        )
        self.norm = Lfm2RMSNorm(config.hidden_size)


class _TinyLfmCausal(nn.Module):
    def __init__(self, config: SimpleNamespace, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = _Body(config)
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                dtype=torch.float64,
            )
        self.lm_head.weight = self.model.embed_tokens.weight
        self.config = config

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class _Config(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


def _config():
    return _Config(
        model_type="lfm2",
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        block_ff_dim=_INTERMEDIATE,
        block_auto_adjust_ff_dim=False,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_heads=2,
        head_dim=2,
        layer_types=["full_attention"] * _LAYERS,
        conv_L_cache=3,
        conv_bias=False,
        vocab_size=11,
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _activation_rows(parent):
    generator = torch.Generator().manual_seed(901)
    hidden = torch.randn(
        12,
        _HIDDEN,
        generator=generator,
        dtype=torch.float64,
    )
    result = {}
    with torch.no_grad():
        for index, layer in enumerate(parent.model.layers):
            gate = layer.feed_forward.w1(hidden)
            up = layer.feed_forward.w3(hidden)
            result[index] = {
                "gate": gate,
                "up": up,
                "post_swiglu": torch.nn.functional.silu(gate) * up,
            }
    return result


def _curves():
    return tuple(
        Lfm2LayerWidthCurve(
            layer_index=index,
            points=(
                Lfm2WidthCostPoint(8, 0.0, 0.0),
                Lfm2WidthCostPoint(4, 0.4 + index / 100.0, 0.2),
                Lfm2WidthCostPoint(3, 0.5 + index / 100.0, 0.3),
            ),
        )
        for index in range(_LAYERS)
    )


@pytest.fixture(scope="module")
def prepared():
    parent = _TinyLfmCausal(_config(), seed=10)
    variable = seal_lfm2_variable_width_evidence(
        parent,
        _activation_rows(parent),
        _curves(),
        cache_key=_digest("parent-cache"),
        heldout_corpus_sha256=_digest("parent-corpus"),
        evidence_builder_id="test.global-causal-width-frontier.v1",
    )
    candidates = []
    for layer_index in range(_LAYERS):
        for target_width in (4, 3):
            proposal = prepare_lfm2_causal_ffn_proposal(
                parent,
                variable,
                layer_index=layer_index,
                target_width=target_width,
                cluster_max_iterations=4,
            )
            scale = 0.01 * (layer_index + 1) + (0.005 if target_width == 3 else 0.0)
            candidates.append(
                Lfm2CausalRemovalCandidate.from_ffn_proposal(
                    proposal,
                    damage=Lfm2CausalDamage(
                        final_logit_margin=scale,
                        eos=scale / 2,
                        repetition=scale / 3,
                        task_anchor=scale,
                        residual_path=scale / 4,
                        attention_path=scale / 5,
                    ),
                    measured_child_trace_sha256=_digest(
                        f"trace-{layer_index}-{target_width}"
                    ),
                    output_event_sha256=_digest(f"events-{layer_index}-{target_width}"),
                    validation_case_ids_sha256=_digest("validation-cases"),
                    measurement_case_count=12,
                )
            )
    candidates.append(
        Lfm2CausalRemovalCandidate(
            schema=CANDIDATE_SCHEMA,
            option_id="diagnostic-attention-head-option",
            site_kind="attention_head",
            layer_index=0,
            source_units=8,
            target_units=4,
            parameters_saved=10_000_000,
            damage=Lfm2CausalDamage(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            measured_child_trace_sha256=_digest("attention-trace"),
            output_event_sha256=_digest("attention-events"),
            validation_case_ids_sha256=_digest("validation-cases"),
            measurement_case_count=12,
            retained_source_units=(0, 1, 2, 3),
        )
    )
    evidence = seal_lfm2_global_causal_evidence(
        parent,
        variable,
        candidates,
        parent_output_envelope_sha256=_digest("parent-outputs"),
        task_anchor_manifest_sha256=_digest("task-anchors"),
        cluster_max_iterations=4,
    )
    return parent, variable, evidence


def test_real_large_target_budget_and_ffn_only_50_percent_limit() -> None:
    thirty_five = lfm2_global_causal_budget(
        0.35,
        parent_unique_parameters=229_693_184,
        hidden_size=1024,
    )
    fifty = lfm2_global_causal_budget(
        0.50,
        parent_unique_parameters=229_693_184,
        hidden_size=1024,
    )

    assert thirty_five.parameters_saved == 80_391_168
    assert thirty_five.target_unique_parameters == 149_302_016
    assert thirty_five.realized_fraction == pytest.approx(0.34999370290413145)
    assert fifty.parameters_saved == 114_846_720
    maximum_ffn_only = (14 * 2560 - 14) * (3 * 1024)
    assert maximum_ffn_only == 110_057_472
    assert maximum_ffn_only < fifty.parameters_saved


def test_fixed_width_bundle_uses_real_installer_and_exact_rollback(
    prepared,
) -> None:
    parent, variable, _evidence = prepared
    widths = [8] * 14
    widths[0] = 4
    bundle = plan_lfm2_variable_width_fold_for_widths(
        parent,
        variable,
        widths,
        cluster_max_iterations=4,
    )
    verify_lfm2_variable_width_fixed_plan_bundle(bundle, parent=parent)
    child = _TinyLfmCausal(_config(), seed=20)
    blank = fingerprint_model_state(child)

    transaction = install_lfm2_variable_width_fold_plan(
        parent,
        child,
        bundle.fixed_evidence,
        bundle.plan,
    )

    assert child.model.layers[0].feed_forward.w1.weight.shape == (4, 4)
    assert transaction.report.parameters_saved == 4 * 3 * _HIDDEN
    assert transaction.rollback() is True
    assert fingerprint_model_state(child) == blank


def test_canonical_proposal_pack_rebuilds_without_clustering(
    prepared,
    tmp_path,
    monkeypatch,
) -> None:
    parent, variable, _evidence = prepared
    (proposal, canonical), _prepare_guard = run_fold_only_callback(
        lambda: prepare_lfm2_causal_ffn_proposal_bundle(
            parent,
            variable,
            layer_index=0,
            target_width=4,
            cluster_max_iterations=4,
        ),
        phase="test-canonical-pack-prepare",
        protected_models=(parent,),
    )
    published, _publish_guard = run_fold_only_callback(
        lambda: write_lfm2_canonical_proposal_pack(
            tmp_path / "canonical-proposals",
            proposal,
            canonical,
            parent=parent,
            evidence=variable,
        ),
        phase="test-canonical-pack-publish",
        protected_models=(parent,),
    )
    assert published.proposal == proposal
    assert published.bundle.bundle_sha256 == canonical.bundle_sha256

    def forbidden_clustering(*_args, **_kwargs):
        raise AssertionError("canonical workers must not invoke clustering")

    monkeypatch.setattr(
        "neural_forge.hf_lfm2_variable_width_fold."
        "build_gated_mlp_activation_cluster_plan",
        forbidden_clustering,
    )
    loaded, _load_guard = run_fold_only_callback(
        lambda: load_lfm2_canonical_proposal_pack(
            tmp_path / "canonical-proposals",
            {
                "option_id": proposal.option_id,
                "layer_index": proposal.layer_index,
                "target_width": proposal.target_width,
                "proposal_sha256": proposal.proposal_sha256,
            },
            parent=parent,
            evidence=variable,
            expected_cluster_ridge=1e-8,
            expected_cluster_max_iterations=4,
            expected_cluster_distance_chunk_size=512,
        ),
        phase="test-canonical-pack-load",
        protected_models=(parent,),
    )
    assert loaded.proposal == proposal
    assert loaded.bundle.plan.plan_sha256 == canonical.plan.plan_sha256
    assert loaded.bundle.bundle_sha256 == canonical.bundle_sha256
    assert (
        loaded.bundle.plan.layer_plans[0].medoid_source_neurons
        == canonical.plan.layer_plans[0].medoid_source_neurons
    )


def test_parent_forward_audit_does_not_clone_hooks_into_installed_child(
    prepared,
) -> None:
    parent, variable, _evidence = prepared
    widths = [8] * 14
    widths[0] = 4
    bundle = plan_lfm2_variable_width_fold_for_widths(
        parent,
        variable,
        widths,
        cluster_max_iterations=4,
    )
    child = _TinyLfmCausal(_config(), seed=21)
    blank = fingerprint_model_state(child)
    blank_runtime = fingerprint_model_runtime_state(child)
    events = []

    def install_and_call_child():
        transaction = install_lfm2_variable_width_fold_plan(
            parent,
            child,
            bundle.fixed_evidence,
            bundle.plan,
        )
        try:
            return child.model.layers[0].feed_forward.w1(
                torch.ones((1, _HIDDEN), dtype=torch.float64)
            )
        finally:
            transaction.rollback()

    output, calls = audit_lfm2_parent_forward_calls(
        parent,
        install_and_call_child,
        phase="installed-child-regression",
        event_sink=events,
    )

    assert output.shape == (1, 4)
    assert calls == 0
    assert events == []
    assert fingerprint_model_state(child) == blank
    assert fingerprint_model_runtime_state(child) == blank_runtime
    assert all(
        not module._forward_pre_hooks and not module._forward_hooks
        for module in child.modules()
    )


def test_parent_forward_audit_ignores_a_deepcopied_parent_module(
    prepared,
) -> None:
    parent, _variable, _evidence = prepared
    events = []

    def copy_and_call():
        copied_embedding = copy.deepcopy(parent.model.embed_tokens)
        return copied_embedding(torch.tensor([[1, 2]], dtype=torch.long))

    output, calls = audit_lfm2_parent_forward_calls(
        parent,
        copy_and_call,
        phase="deepcopy-regression",
        event_sink=events,
    )

    assert output.shape == (1, 2, _HIDDEN)
    assert calls == 0
    assert events == []


def test_parent_forward_audit_removes_global_hook_before_runtime_authentication(
    prepared,
) -> None:
    parent, _variable, _evidence = prepared
    child = _TinyLfmCausal(_config(), seed=22)
    blank_runtime = fingerprint_model_runtime_state(child)
    observed_inside = []

    _result, calls = audit_lfm2_parent_forward_calls(
        parent,
        lambda: observed_inside.append(fingerprint_model_runtime_state(child)),
        phase="runtime-hook-lifetime-regression",
    )

    assert calls == 0
    assert observed_inside[0] != blank_runtime
    assert fingerprint_model_runtime_state(child) == blank_runtime


def test_global_allocator_builds_monotonic_exact_35_and_refuses_fake_50(
    prepared,
) -> None:
    parent, _variable, evidence = prepared

    plan = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
        maximum_stage_fraction=0.10,
    )

    # The target removes exactly 65 of the toy's 112 FFN neurons.  The global
    # score is free to choose any causally cheaper 3/4/8-width combination.
    assert plan.budget.removed_ffn_neurons == 65
    assert sum(plan.stages[-1].target_widths) == 47
    assert set(plan.stages[-1].target_widths) <= {3, 4, 8}
    assert len(plan.stages) > 1
    assert all(
        left.cumulative_parameters_saved < right.cumulative_parameters_saved
        for left, right in zip(plan.stages, plan.stages[1:])
    )
    assert "diagnostic-attention-head-option" in plan.unsupported_option_ids

    with pytest.raises(Lfm2GlobalCausalInfeasible, match="No authorized"):
        plan_lfm2_global_causal_target(
            parent,
            evidence,
            target_fraction=0.50,
        )


def test_real_child_runner_rejects_and_restores_exactly(prepared) -> None:
    parent, _variable, evidence = prepared
    plan = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
    )
    children = []
    blanks = []

    def child_factory():
        child = _TinyLfmCausal(_config(), seed=100 + len(children))
        children.append(child)
        blanks.append(fingerprint_model_state(child))
        return child

    def reject(_context):
        return Lfm2GlobalCausalStageDecision(
            accepted=False,
            reason="synthetic strict-gate rejection",
            child_trace_cache_key=_digest("rejected-child-cache"),
            child_output_events_sha256=_digest("rejected-events"),
            trajectory_gate_passed=False,
            intelligence_gate_passed=False,
            coherent_response_gate_passed=False,
        )

    run = execute_lfm2_global_causal_plan(
        parent,
        evidence,
        plan,
        child_factory=child_factory,
        evaluator=reject,
    )

    assert run.status == "rejected"
    assert len(run.attempts) == 1
    assert run.attempts[0].rollback_exact is True
    assert fingerprint_model_state(children[0]) == blanks[0]
    assert run.as_dict()["optimizer_steps"] == 0
    assert run.as_dict()["backward_calls"] == 0


def test_real_child_runner_commits_only_an_accepted_final_candidate(
    prepared,
) -> None:
    parent, _variable, evidence = prepared
    plan = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
        maximum_stage_fraction=1.0,
    )
    assert len(plan.stages) == 1

    def accept(_context):
        return Lfm2GlobalCausalStageDecision(
            accepted=True,
            reason="synthetic complete strict-gate pass",
            child_trace_cache_key=_digest("accepted-child-cache"),
            child_output_events_sha256=_digest("accepted-events"),
            trajectory_gate_passed=True,
            intelligence_gate_passed=True,
            coherent_response_gate_passed=True,
        )

    run = execute_lfm2_global_causal_plan(
        parent,
        evidence,
        plan,
        child_factory=lambda: _TinyLfmCausal(_config(), seed=250),
        evaluator=accept,
    )

    assert run.status == "accepted"
    assert run.committed_child is not None
    assert run.attempts[-1].transaction_committed is True
    assert (
        sum(parameter.numel() for parameter in run.committed_child.parameters())
        == plan.budget.target_unique_parameters
    )
    assert (
        tuple(
            layer.feed_forward.w1.out_features
            for layer in run.committed_child.model.layers
        )
        == plan.stages[-1].target_widths
    )


def test_parent_forward_during_child_gate_fails_closed_and_rolls_back(
    prepared,
) -> None:
    parent, _variable, evidence = prepared
    plan = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
    )
    children = []
    blanks = []

    def child_factory():
        child = _TinyLfmCausal(_config(), seed=300)
        children.append(child)
        blanks.append(fingerprint_model_state(child))
        return child

    def invalid(_context):
        parent.model.embed_tokens(torch.tensor([[1, 2]], dtype=torch.long))
        return Lfm2GlobalCausalStageDecision(
            accepted=True,
            reason="invalid parent-backed evaluation",
            child_trace_cache_key=_digest("invalid-child-cache"),
            child_output_events_sha256=_digest("invalid-events"),
            trajectory_gate_passed=True,
            intelligence_gate_passed=True,
            coherent_response_gate_passed=True,
        )

    with pytest.raises(
        Lfm2GlobalCausalExecutionError,
        match="called the live parent",
    ):
        execute_lfm2_global_causal_plan(
            parent,
            evidence,
            plan,
            child_factory=child_factory,
            evaluator=invalid,
        )

    assert fingerprint_model_state(children[0]) == blanks[0]


def test_causal_candidate_binding_tamper_is_rejected(prepared) -> None:
    parent, variable, evidence = prepared
    original = next(
        value for value in evidence.candidates if value.physically_installable
    )
    tampered = replace(original, layer_binding_sha256="f" * 64)
    candidates = tuple(
        tampered if value.option_id == original.option_id else value
        for value in evidence.candidates
    )

    with pytest.raises(Exception, match="does not bind"):
        seal_lfm2_global_causal_evidence(
            parent,
            variable,
            candidates,
            parent_output_envelope_sha256=_digest("parent-outputs"),
            task_anchor_manifest_sha256=_digest("task-anchors"),
            cluster_max_iterations=4,
        )


def test_prepared_frontier_runner_is_local_fold_only_and_physically_executes(
    prepared,
    canonical_pack_catalog,
    monkeypatch,
) -> None:
    parser = runner.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--target-fractions" in options
    assert "--maximum-stage-fraction" in options
    assert "--train" not in options
    assert "--learning-rate" not in options

    parent, _variable, evidence = prepared
    created = []

    def child_factory(_fraction):
        child = _TinyLfmCausal(_config(), seed=500 + len(created))
        created.append(child)
        return child

    def evaluator_factory(_fraction, _plan):
        def reject(_context):
            return Lfm2GlobalCausalStageDecision(
                accepted=False,
                reason="frontier contract rejection",
                child_trace_cache_key=_digest("frontier-child-cache"),
                child_output_events_sha256=_digest("frontier-events"),
                trajectory_gate_passed=False,
                intelligence_gate_passed=False,
                coherent_response_gate_passed=False,
            )

        return reject

    def forbidden_clustering(*_args, **_kwargs):
        raise AssertionError("canonical prepared runner must not recluster")

    monkeypatch.setattr(
        "neural_forge.hf_lfm2_variable_width_fold."
        "build_gated_mlp_activation_cluster_plan",
        forbidden_clustering,
    )
    result = runner.execute_prepared_global_causal_frontier(
        parent,
        evidence,
        child_factory=child_factory,
        evaluator_factory=evaluator_factory,
        canonical_packs=canonical_pack_catalog,
    )

    assert [value["status"] for value in result.report["targets"]] == [
        "rejected",
        "infeasible",
    ]
    assert len(created) == 1
    assert result.report["new_parent_generation_calls"] == 0
    assert result.report["canonical_payloads_used"] is True
    assert result.report["reclustering_used"] is False
    assert result.report["unsupported_head_or_conv_options_counted"] is False


def test_cached_causal_producer_cli_has_no_training_or_parent_build_path() -> None:
    parser = producer.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}

    assert "--parent-cache" in options
    assert "--expected-parent-cache-key" in options
    assert "--parent-output-envelope" in options
    assert "--frontier-document" in options
    assert "--mode" in options
    assert "--run-contract-document" in options
    assert "--canonical-proposal-root" in options
    assert "--probe-layer-index" in options
    assert "--probe-target-width" in options
    assert "--probe-report" in options
    assert "--shard-count" in options
    assert "--shard-index" in options
    assert "--shard-weights" in options
    assert "--adopt-candidate-record-root" in options
    assert "--merge-shard-root" in options
    assert "--train" not in options
    assert "--learning-rate" not in options
    assert "--build-parent-cache" not in options
    assert "--generate-parent-responses" not in options


class _FullPrefixProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        use_cache,
        return_dict,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.detach().cpu().clone(),
                "attention_mask": attention_mask.detach().cpu().clone(),
                "position_ids": position_ids.detach().cpu().clone(),
                "use_cache": use_cache,
                "return_dict": return_dict,
            }
        )
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            7,
            device=input_ids.device,
        )
        lengths = attention_mask.sum(dim=1)
        for row, length in enumerate(lengths.tolist()):
            token = 2 if int(length) <= (3 + row) else 3
            logits[row, -1, token] = 10.0
        return SimpleNamespace(logits=logits)


class _ProbeTokenizer:
    pad_token_id = 0

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return " ".join(str(value) for value in token_ids if value != 3)


class _ProbeBaseline:
    def __init__(self):
        self.cases = tuple(
            SimpleNamespace(
                case_id=f"recovery_math_{index:02d}",
                parent_output="2",
                parent_metadata=SimpleNamespace(
                    maximum_repeated_ngram_share=0.0,
                ),
            )
            for index in range(2)
        )

    @staticmethod
    def task_score(_case, output):
        return "2" in output

    @staticmethod
    def correctness_score(_case, output):
        return output.strip() == "2"


def test_cached_causal_rollout_is_exact_prompt_full_prefix_and_uncached() -> None:
    model = _FullPrefixProbe()
    rows = (
        {
            "id": "recovery_math_00",
            "input_ids": [4, 5, 6],
            "attention_mask": [1, 1, 1],
            "parent_output_ids": [2, 3],
        },
        {
            "id": "recovery_math_01",
            "input_ids": [1, 4, 5, 6],
            "attention_mask": [1, 1, 1, 1],
            "parent_output_ids": [2, 3],
        },
    )

    rollouts = generate_lfm2_full_prefix_rollouts(
        model,
        rows,
        validation_case_ids=(
            "recovery_math_00",
            "recovery_math_01",
        ),
        tokenizer=_ProbeTokenizer(),
        baseline=_ProbeBaseline(),
        eos_token_ids=(3,),
        generation_limit=4,
        rollout_batch_size=2,
    )

    assert [value.token_ids for value in rollouts] == [(2, 3), (2, 3)]
    assert all(value.eos_reached for value in rollouts)
    assert len(model.calls) == 2
    assert all(call["use_cache"] is False for call in model.calls)
    # The exact sealed prompt suffixes survive left-padding unchanged.
    assert model.calls[0]["input_ids"][0].tolist() == [0, 4, 5, 6]
    assert model.calls[0]["input_ids"][1].tolist() == [1, 4, 5, 6]


def test_full_prefix_cache_reads_nested_batch_capture_audit() -> None:
    manifest = {
        "capture": {
            "decode_trace_mode": "full_prefix_canonical",
            "cached_logits_accepted": False,
            "forced_decode_use_cache": False,
        }
    }
    valid = SimpleNamespace(
        manifest=manifest,
        iter_batches=lambda: iter(
            (SimpleNamespace(record={"capture": {"cached_forward_count": 0}}),)
        ),
    )
    stale_shape = SimpleNamespace(
        manifest=manifest,
        iter_batches=lambda: iter(
            (SimpleNamespace(record={"cached_forward_count": 0}),)
        ),
    )

    assert verify_lfm2_full_prefix_child_trace_cache(valid) is valid
    with pytest.raises(
        Lfm2CachedCausalEvidenceError,
        match="recurrent cached forward",
    ):
        verify_lfm2_full_prefix_child_trace_cache(stale_shape)


def _canonical_digest(value) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class _CachedActivationRows:
    def __init__(self, values, *, trace_rows_sha256: str):
        self._values = values
        self.trace_rows_sha256 = trace_rows_sha256

    def field(self, layer_index: int, name: str):
        return self._values[layer_index][name]


@pytest.fixture(scope="module")
def cached_frontier():
    parent = _TinyLfmCausal(_config(), seed=610)
    activation = _activation_rows(parent)
    rows = _CachedActivationRows(
        {
            layer_index: {
                "mlp.gate": value["gate"],
                "mlp.up": value["up"],
                "mlp.post_swiglu": value["post_swiglu"],
            }
            for layer_index, value in activation.items()
        },
        trace_rows_sha256=_digest("cached-causal-parent-rows"),
    )
    descriptors = tuple(
        {
            "path": f"batch.{index:06d}/prefill/row.0000",
            "case_id": (
                f"recovery_math_{index:02d}"
                if index % 2 == 0
                else f"recovery_code_{index:02d}"
            ),
            "predicted_token_id": index % 11,
        }
        for index in range(12)
    )
    profile = SimpleNamespace(
        name="balanced",
        profile_sha256=_digest("cached-causal-profile"),
        selection_indices=(0, 1, 2, 3),
        fit_indices=(4, 5, 6, 7),
        validation_indices=(8, 9, 10, 11),
    )
    split = build_lfm2_cached_causal_split(
        profile,
        descriptors,
        parent_trace_rows_sha256=rows.trace_rows_sha256,
    )
    parent_parameters = sum(int(parameter.numel()) for parameter in parent.parameters())
    widths = lfm2_large_target_width_frontier(
        parent_unique_parameters=parent_parameters,
        hidden_size=_HIDDEN,
        source_width=_INTERMEDIATE,
        coarse_widths=(2,),
    )
    variable = build_lfm2_causal_physical_source_evidence(
        parent,
        rows,
        split,
        widths,
        cache_key=_digest("cached-causal-cache"),
        heldout_corpus_sha256=_digest("cached-causal-corpus"),
    )
    candidates = []
    reports = []
    for layer_index, layer_widths in enumerate(widths):
        for target_width in layer_widths[1:]:
            proposal = prepare_lfm2_causal_ffn_proposal(
                parent,
                variable,
                layer_index=layer_index,
                target_width=target_width,
                cluster_max_iterations=2,
            )
            scale = 0.005 * (layer_index + 1) + 0.001 * (_INTERMEDIATE - target_width)
            report_body = {
                "schema": "test-cached-causal-measurement-v1",
                "option_id": proposal.option_id,
                "split_sha256": split.split_sha256,
                "validation_case_ids": list(split.validation_case_ids),
                "damage_scale": scale,
                "parent_forward_calls": 0,
            }
            event_sha = _canonical_digest(report_body)
            report = {
                **report_body,
                "output_event_sha256": event_sha,
            }
            reports.append(report)
            candidates.append(
                Lfm2CausalRemovalCandidate.from_ffn_proposal(
                    proposal,
                    damage=Lfm2CausalDamage(
                        final_logit_margin=scale,
                        eos=scale / 2,
                        repetition=scale / 3,
                        task_anchor=scale,
                        residual_path=scale / 4,
                        attention_path=scale / 5,
                    ),
                    measured_child_trace_sha256=_digest(
                        f"cached-trace-{layer_index}-{target_width}"
                    ),
                    output_event_sha256=event_sha,
                    validation_case_ids_sha256=(split.validation_case_ids_sha256),
                    measurement_case_count=len(split.validation_case_ids),
                )
            )
    task_anchor_manifest = {
        "schema": "test-task-anchor-manifest-v1",
        "split_sha256": split.split_sha256,
        "manual_cases_used": False,
    }
    task_anchor_sha = _canonical_digest(task_anchor_manifest)
    evidence = seal_lfm2_global_causal_evidence(
        parent,
        variable,
        candidates,
        parent_output_envelope_sha256=_digest("cached-parent-output-envelope"),
        task_anchor_manifest_sha256=task_anchor_sha,
        cluster_max_iterations=2,
    )
    plan = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
        maximum_stage_fraction=0.10,
    )
    with pytest.raises(Lfm2GlobalCausalInfeasible) as infeasible:
        plan_lfm2_global_causal_target(
            parent,
            evidence,
            target_fraction=0.50,
        )
    document = seal_lfm2_cached_causal_frontier_document(
        evidence,
        split,
        widths,
        reports,
        plan,
        parent_output_file_sha256=_digest("cached-parent-output-file"),
        tokenizer_fingerprint=_digest("cached-tokenizer"),
        task_anchor_manifest=task_anchor_manifest,
        fifty_percent_infeasible_reason=str(infeasible.value),
    )
    parent_cache = SimpleNamespace(
        cache_key=variable.cache_key,
        corpus_fingerprint=variable.heldout_corpus_sha256,
    )
    return parent, parent_cache, rows, document, split, variable


def test_cached_causal_split_rejects_selection_validation_leakage() -> None:
    descriptors = tuple(
        {
            "path": f"batch.{index:06d}/prefill/row.0000",
            "case_id": f"recovery_math_{index:02d}",
        }
        for index in range(6)
    )
    leaking = SimpleNamespace(
        name="leaking",
        profile_sha256=_digest("leaking-profile"),
        selection_indices=(0, 1, 2),
        fit_indices=(3, 4),
        validation_indices=(2, 5),
    )

    with pytest.raises(
        Lfm2CachedCausalEvidenceError,
        match="row-disjoint",
    ):
        build_lfm2_cached_causal_split(
            leaking,
            descriptors,
            parent_trace_rows_sha256=_digest("leaking-rows"),
        )


def test_cached_causal_frontier_replays_exact_35_and_refuses_50(
    cached_frontier,
) -> None:
    parent, parent_cache, rows, document, _split, _variable = cached_frontier

    evidence, plan = runner.authenticate_cached_global_causal_frontier(
        parent,
        parent_cache,
        rows,
        document,
    )

    assert evidence.evidence_sha256 == document["global_evidence"]["evidence_sha256"]
    assert plan.plan_sha256 == document["targets"]["0.35"]["plan_sha256"]
    assert plan.budget.requested_fraction == 0.35
    assert plan.stages[-1].cumulative_parameters_saved == plan.budget.parameters_saved
    assert document["targets"]["0.50"]["status"] == "infeasible"


@pytest.fixture(scope="module")
def cached_frontier_canonical_packs(cached_frontier, tmp_path_factory):
    parent, _parent_cache, _rows, document, _split, variable = cached_frontier
    expected_by_coordinate = {
        (candidate["layer_index"], candidate["target_units"]): candidate
        for candidate in document["candidates"]
    }
    root = tmp_path_factory.mktemp("cached-frontier-canonical-packs")
    packs = {}
    for layer_index, layer_widths in enumerate(document["widths_by_layer"]):
        for target_width in layer_widths[1:]:
            proposal, bundle = prepare_lfm2_causal_ffn_proposal_bundle(
                parent,
                variable,
                layer_index=layer_index,
                target_width=target_width,
                cluster_max_iterations=2,
            )
            expected = expected_by_coordinate[(layer_index, target_width)]
            assert proposal.option_id == expected["option_id"]
            assert proposal.proposal_sha256 == expected["proposal_sha256"]
            pack_root = root / proposal.option_id
            packs[proposal.option_id] = Lfm2CanonicalProposalPack(
                root=pack_root,
                manifest_path=pack_root / "manifest.json",
                tensor_path=pack_root / "changed-layer.safetensors",
                manifest=MappingProxyType(
                    {
                        "manifest_sha256": _digest(
                            f"cached-pack-manifest-{proposal.option_id}"
                        ),
                        "tensor_file_sha256": _digest(
                            f"cached-pack-tensor-{proposal.option_id}"
                        ),
                    }
                ),
                proposal=proposal,
                bundle=bundle,
            )
    return packs


def _canonical_finalizer_report(document, packs):
    pack_binding = {
        option_id: {
            "manifest_sha256": pack.manifest["manifest_sha256"],
            "tensor_file_sha256": pack.manifest["tensor_file_sha256"],
            "proposal_sha256": pack.proposal.proposal_sha256,
            "layer_binding_sha256": pack.proposal.layer_binding_sha256,
        }
        for option_id, pack in sorted(packs.items())
    }
    body = {
        "schema": CANONICAL_FINALIZER_SCHEMA,
        "run_contract_sha256": _digest("canonical-replay-contract"),
        "frontier_sha256": document["frontier_sha256"],
        "evidence_sha256": document["global_evidence"]["evidence_sha256"],
        "plan_sha256": document["targets"]["0.35"]["plan_sha256"],
        "finalizer_implementation_sha256": _digest(
            "canonical-replay-finalizer"
        ),
        "canonical_allocator_implementation_sha256": _digest(
            "canonical-replay-allocator"
        ),
        "canonical_pack_binding_sha256": _canonical_digest(pack_binding),
        "canonical_pack_count": len(pack_binding),
        "guard_report_sha256": _digest("canonical-replay-guard"),
        "clustering_calls": 0,
        "parent_forward_calls": 0,
        "parent_cache_build_calls": 0,
        "parent_generation_calls": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "exact_token_match_objective_used": False,
    }
    return {**body, "finalizer_report_sha256": _canonical_digest(body)}


def test_canonical_frontier_replay_uses_exact_packs_and_zero_reclustering(
    cached_frontier,
    cached_frontier_canonical_packs,
) -> None:
    parent, parent_cache, rows, document, _split, _variable = cached_frontier
    finalizer = _canonical_finalizer_report(
        document,
        cached_frontier_canonical_packs,
    )

    replay = replay_lfm2_cached_causal_frontier_from_canonical_packs(
        parent,
        parent_cache,
        rows,
        document,
        cached_frontier_canonical_packs,
        finalizer,
    )

    assert (
        replay.evidence.evidence_sha256
        == document["global_evidence"]["evidence_sha256"]
    )
    assert replay.plan.plan_sha256 == document["targets"]["0.35"]["plan_sha256"]
    assert replay.receipt["schema"] == CANONICAL_REPLAY_SCHEMA
    assert replay.receipt["canonical_pack_count"] == len(
        cached_frontier_canonical_packs
    )
    assert replay.receipt["clustering_calls"] == 0
    assert replay.receipt["parent_forward_calls"] == 0
    assert replay.receipt["optimizer_steps"] == 0
    assert replay.receipt["backward_calls"] == 0
    receipt_body = dict(replay.receipt)
    receipt_seal = receipt_body.pop("replay_receipt_sha256")
    assert receipt_seal == _canonical_digest(receipt_body)


def test_prepared_runner_exposes_canonical_frontier_authentication(
    cached_frontier,
    cached_frontier_canonical_packs,
) -> None:
    parent, parent_cache, rows, document, _split, _variable = cached_frontier
    finalizer = _canonical_finalizer_report(
        document,
        cached_frontier_canonical_packs,
    )

    replay = runner.authenticate_canonical_cached_global_causal_frontier(
        parent,
        parent_cache,
        rows,
        document,
        cached_frontier_canonical_packs,
        finalizer,
    )

    assert replay.receipt["schema"] == CANONICAL_REPLAY_SCHEMA
    assert replay.plan.plan_sha256 == document["targets"]["0.35"]["plan_sha256"]


def test_canonical_merge_wrapper_end_to_end_uses_exact_packs(
    cached_frontier,
    cached_frontier_canonical_packs,
    tmp_path,
    monkeypatch,
) -> None:
    parent, _parent_cache, _rows, document, split, variable = cached_frontier
    candidates = tuple(
        cached_causal_module._candidate_from_dict(value)
        for value in document["candidates"]
    )
    contract_sha = _digest("canonical-merge-e2e-contract")
    fake_contract = {
        "run_contract_sha256": contract_sha,
        "frontier": {
            "candidate_inventory": [
                {"option_id": candidate.option_id}
                for candidate in candidates
            ]
        },
    }
    canonical_root = tmp_path / "canonical-packs"
    canonical_root.mkdir()
    contract_path = tmp_path / "run-contract.json"
    report_path = tmp_path / "canonical-finalizer-report.json"
    monkeypatch.setattr(
        canonical_merge,
        "_required_environment",
        lambda: (canonical_root, contract_path, report_path),
    )
    monkeypatch.setattr(
        canonical_merge,
        "_load_contract",
        lambda path: fake_contract if path == contract_path else None,
    )

    def fake_load(_root, expected, **_kwargs):
        assert _root == canonical_root
        return cached_frontier_canonical_packs[expected["option_id"]]

    monkeypatch.setattr(
        canonical_merge,
        "load_lfm2_canonical_proposal_pack",
        fake_load,
    )
    weights = document["global_evidence"]["damage_weights"]
    cluster = document["global_evidence"]["cluster"]
    evidence, plan, replayed_document = canonical_merge._canonical_replacement(
        parent,
        variable,
        candidates,
        split,
        document["widths_by_layer"],
        document["measurement_reports"],
        parent_output_envelope_sha256=(
            document["parent_binding"]["parent_output_envelope_sha256"]
        ),
        parent_output_file_sha256=(
            document["parent_binding"]["parent_output_file_sha256"]
        ),
        task_anchor_manifest_sha256=(
            document["parent_binding"]["task_anchor_manifest_sha256"]
        ),
        task_anchor_manifest=document["task_anchor_manifest"],
        tokenizer_fingerprint=(
            document["parent_binding"]["tokenizer_fingerprint"]
        ),
        damage_weights=Lfm2CausalDamageWeights(
            final_logit_margin=weights["final_logit_margin"],
            eos=weights["eos"],
            repetition=weights["repetition"],
            task_anchor=weights["task_anchor"],
            residual_path=weights["residual_path"],
            attention_path=weights["attention_path"],
        ),
        cluster_ridge=cluster["ridge"],
        cluster_max_iterations=cluster["max_iterations"],
        cluster_distance_chunk_size=cluster["distance_chunk_size"],
        maximum_stage_fraction=0.10,
    )

    assert evidence.evidence_sha256 == document["global_evidence"]["evidence_sha256"]
    assert plan.plan_sha256 == document["targets"]["0.35"]["plan_sha256"]
    assert replayed_document == document
    with report_path.open("r", encoding="ascii") as handle:
        report = json.load(handle)
    assert report["run_contract_sha256"] == contract_sha
    assert report["frontier_sha256"] == document["frontier_sha256"]
    assert report["canonical_pack_count"] == len(
        cached_frontier_canonical_packs
    )
    assert report["clustering_calls"] == 0
    assert report["parent_forward_calls"] == 0
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0


def test_canonical_frontier_replay_rejects_finalizer_pack_drift(
    cached_frontier,
    cached_frontier_canonical_packs,
) -> None:
    parent, parent_cache, rows, document, _split, _variable = cached_frontier
    finalizer = _canonical_finalizer_report(
        document,
        cached_frontier_canonical_packs,
    )
    tampered = dict(finalizer)
    tampered["canonical_pack_count"] -= 1
    body = dict(tampered)
    body.pop("finalizer_report_sha256")
    tampered["finalizer_report_sha256"] = _canonical_digest(body)

    with pytest.raises(
        Lfm2CanonicalFrontierReplayError,
        match="does not bind this frontier",
    ):
        replay_lfm2_cached_causal_frontier_from_canonical_packs(
            parent,
            parent_cache,
            rows,
            document,
            cached_frontier_canonical_packs,
            tampered,
        )


def test_cached_causal_frontier_tamper_is_rejected(cached_frontier) -> None:
    _parent, _parent_cache, _rows, document, _split, _variable = cached_frontier
    tampered = json.loads(json.dumps(document))
    tampered["measurement_reports"][0]["damage_scale"] += 1.0

    with pytest.raises(
        Lfm2CachedCausalEvidenceError,
        match="frontier seal",
    ):
        verify_lfm2_cached_causal_frontier_document(tampered)


def test_cached_causal_parent_forward_audit_covers_proposal_preparation(
    cached_frontier,
) -> None:
    parent, _parent_cache, _rows, _document, _split, variable = cached_frontier

    proposal, calls = audit_lfm2_parent_forward_calls(
        parent,
        lambda: prepare_lfm2_causal_ffn_proposal(
            parent,
            variable,
            layer_index=0,
            target_width=4,
            cluster_max_iterations=2,
        ),
    )
    assert proposal.target_width == 4
    assert calls == 0

    events = []
    _value, invalid_calls = audit_lfm2_parent_forward_calls(
        parent,
        lambda: parent.model.embed_tokens(torch.tensor([[1, 2]], dtype=torch.long)),
        phase="known-live-parent-regression",
        event_sink=events,
    )
    assert invalid_calls > 0
    assert events[0]["phase"] == "known-live-parent-regression"
    assert events[0]["module_paths"] == ("model.embed_tokens",)
    assert any(
        frame["function"]
        == "test_cached_causal_parent_forward_audit_covers_proposal_preparation"
        for frame in events[0]["call_stack"]
    )


@pytest.fixture(scope="module")
def interaction_bound(prepared):
    parent, variable, evidence = prepared
    beam = enumerate_lfm2_interaction_beam_plan(
        evidence,
        target_fraction=0.35,
        top_k=3,
        per_budget_beam=8,
    )
    bundles = {
        alternative.alternative_sha256: (
            plan_lfm2_variable_width_fold_for_widths(
                parent,
                variable,
                alternative.target_widths,
                cluster_max_iterations=4,
            )
        )
        for alternative in beam.alternatives
    }

    def materialize(alternative):
        return Lfm2InteractionMaterialization(
            schema=MATERIALIZATION_SCHEMA,
            bundle=bundles[alternative.alternative_sha256],
            clustering_calls=0,
        )

    bound = bind_lfm2_interaction_beam_plan(
        parent,
        evidence,
        beam,
        materialize=materialize,
        require_canonical_payload=False,
    )
    return parent, variable, evidence, beam, bound, materialize


@pytest.fixture(scope="module")
def canonical_pack_catalog(prepared, tmp_path_factory):
    parent, variable, evidence = prepared
    root = tmp_path_factory.mktemp("canonical-pack-catalog")
    packs = {}
    for candidate in evidence.candidates:
        if not candidate.physically_installable:
            continue
        proposal, isolated = prepare_lfm2_causal_ffn_proposal_bundle(
            parent,
            variable,
            layer_index=candidate.layer_index,
            target_width=candidate.target_units,
            cluster_max_iterations=4,
        )
        assert proposal.option_id == candidate.option_id
        pack_root = root / candidate.option_id
        packs[candidate.option_id] = Lfm2CanonicalProposalPack(
            root=pack_root,
            manifest_path=pack_root / "manifest.json",
            tensor_path=pack_root / "changed-layer.safetensors",
            manifest=MappingProxyType(
                {
                    "manifest_sha256": _digest(
                        f"catalog-manifest-{candidate.option_id}"
                    ),
                    "tensor_file_sha256": _digest(
                        f"catalog-tensor-{candidate.option_id}"
                    ),
                }
            ),
            proposal=proposal,
            bundle=isolated,
        )
    return packs


def test_interaction_beam_preserves_additive_control_and_exact_topk(
    prepared,
) -> None:
    parent, _variable, evidence = prepared
    control = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
        maximum_stage_fraction=1.0,
    )
    beam = enumerate_lfm2_interaction_beam_plan(
        evidence,
        target_fraction=0.35,
        top_k=5,
        per_budget_beam=8,
    )

    assert len(beam.alternatives) == 5
    assert beam.control.target_widths == control.stages[-1].target_widths
    assert beam.control.option_ids == control.selected_option_ids
    assert all(
        alternative.parameters_saved == control.budget.parameters_saved
        for alternative in beam.alternatives
    )
    assert len(
        {
            alternative.target_widths
            for alternative in beam.alternatives
        }
    ) == len(beam.alternatives)
    assert verify_lfm2_interaction_beam_plan(beam, evidence) is beam


def test_interaction_beam_refuses_unphysical_55_percent_ffn_target(
    prepared,
) -> None:
    _parent, _variable, evidence = prepared

    with pytest.raises(
        Lfm2InteractionBeamInfeasible,
        match="No exact FFN allocation",
    ):
        enumerate_lfm2_interaction_beam_plan(
            evidence,
            target_fraction=0.55,
            top_k=2,
            per_budget_beam=4,
        )


def test_production_interaction_binding_requires_canonical_payloads(
    interaction_bound,
) -> None:
    parent, _variable, evidence, beam, _bound, materialize = interaction_bound

    with pytest.raises(
        Exception,
        match="requires every selected canonical proposal pack",
    ):
        bind_lfm2_interaction_beam_plan(
            parent,
            evidence,
            beam,
            materialize=materialize,
            require_canonical_payload=True,
        )


def test_canonical_multi_pack_materializer_reproduces_full_payload_without_clustering(
    prepared,
    tmp_path,
    monkeypatch,
) -> None:
    parent, variable, evidence = prepared
    beam = enumerate_lfm2_interaction_beam_plan(
        evidence,
        target_fraction=0.35,
        top_k=1,
        per_budget_beam=2,
    )
    alternative = beam.control
    candidate_by_id = {
        candidate.option_id: candidate
        for candidate in evidence.candidates
        if candidate.physically_installable
    }
    packs = {}
    for option_id in alternative.option_ids:
        candidate = candidate_by_id[option_id]
        proposal, isolated = prepare_lfm2_causal_ffn_proposal_bundle(
            parent,
            variable,
            layer_index=candidate.layer_index,
            target_width=candidate.target_units,
            cluster_max_iterations=4,
        )
        assert proposal.option_id == option_id
        pack_root = tmp_path / option_id
        packs[option_id] = Lfm2CanonicalProposalPack(
            root=pack_root,
            manifest_path=pack_root / "manifest.json",
            tensor_path=pack_root / "changed-layer.safetensors",
            manifest=MappingProxyType(
                {
                    "manifest_sha256": _digest(
                        f"multi-pack-manifest-{option_id}"
                    ),
                    "tensor_file_sha256": _digest(
                        f"multi-pack-tensor-{option_id}"
                    ),
                }
            ),
            proposal=proposal,
            bundle=isolated,
        )
    expected = plan_lfm2_variable_width_fold_for_widths(
        parent,
        variable,
        alternative.target_widths,
        cluster_max_iterations=4,
    )

    def forbidden_clustering(*_args, **_kwargs):
        raise AssertionError("multi-pack materialization must not cluster")

    monkeypatch.setattr(
        "neural_forge.hf_lfm2_variable_width_fold."
        "build_gated_mlp_activation_cluster_plan",
        forbidden_clustering,
    )
    materialization = materialize_lfm2_canonical_multi_pack_allocation(
        parent,
        evidence,
        alternative,
        packs,
    )

    assert materialization.clustering_calls == 0
    assert materialization.canonical_option_ids == alternative.option_ids
    assert (
        materialization.bundle.plan.plan_sha256
        == expected.plan.plan_sha256
    )
    assert (
        materialization.bundle.bundle_sha256
        == expected.bundle_sha256
    )
    assert (
        materialization.bundle.layer_binding_sha256
        == expected.layer_binding_sha256
    )


def test_canonical_finalizer_seals_evidence_and_stages_without_reclustering(
    prepared,
    canonical_pack_catalog,
    monkeypatch,
) -> None:
    parent, variable, evidence = prepared
    expected = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
        maximum_stage_fraction=0.10,
    )

    def forbidden_clustering(*_args, **_kwargs):
        raise AssertionError("canonical finalization must not recluster")

    monkeypatch.setattr(
        "neural_forge.hf_lfm2_variable_width_fold."
        "build_gated_mlp_activation_cluster_plan",
        forbidden_clustering,
    )
    canonical_evidence = (
        seal_lfm2_global_causal_evidence_from_canonical_packs(
            parent,
            variable,
            evidence.candidates,
            canonical_pack_catalog,
            parent_output_envelope_sha256=(
                evidence.parent_output_envelope_sha256
            ),
            task_anchor_manifest_sha256=(
                evidence.task_anchor_manifest_sha256
            ),
            damage_weights=evidence.damage_weights,
            cluster_ridge=evidence.cluster_ridge,
            cluster_max_iterations=evidence.cluster_max_iterations,
            cluster_distance_chunk_size=(
                evidence.cluster_distance_chunk_size
            ),
        )
    )
    canonical_plan = plan_lfm2_global_causal_target_from_canonical_packs(
        parent,
        canonical_evidence,
        canonical_pack_catalog,
        target_fraction=0.35,
        maximum_stage_fraction=0.10,
    )

    assert canonical_evidence.evidence_sha256 == evidence.evidence_sha256
    assert canonical_plan.plan_sha256 == expected.plan_sha256
    assert tuple(
        stage.physical_bundle.bundle_sha256
        for stage in canonical_plan.stages
    ) == tuple(
        stage.physical_bundle.bundle_sha256
        for stage in expected.stages
    )


def test_canonical_merge_wrapper_audits_and_seals_exact_pack_binding(
    prepared,
    canonical_pack_catalog,
    tmp_path,
    monkeypatch,
) -> None:
    parent, variable, evidence = prepared
    physical_candidates = tuple(
        candidate
        for candidate in evidence.candidates
        if candidate.physically_installable
    )
    expected_plan = plan_lfm2_global_causal_target(
        parent,
        evidence,
        target_fraction=0.35,
        maximum_stage_fraction=0.10,
    )
    contract_sha = _digest("canonical-wrapper-contract")
    frontier_sha = _digest("canonical-wrapper-frontier")
    contract_path = tmp_path / "run-contract.json"
    report_path = tmp_path / "canonical-finalizer-report.json"
    canonical_root = tmp_path / "canonical-packs"
    canonical_root.mkdir()
    expected_inventory = tuple(
        {
            "option_id": candidate.option_id,
            "layer_index": candidate.layer_index,
            "target_width": candidate.target_units,
        }
        for candidate in physical_candidates
    )
    fake_contract = {
        "run_contract_sha256": contract_sha,
        "frontier": {"candidate_inventory": list(expected_inventory)},
    }
    original_clustering = (
        canonical_merge.variable_width_fold_module
        .build_gated_mlp_activation_cluster_plan
    )
    finalizer_calls = []

    monkeypatch.setattr(
        canonical_merge,
        "_required_environment",
        lambda: (canonical_root, contract_path, report_path),
    )
    monkeypatch.setattr(
        canonical_merge,
        "_load_contract",
        lambda path: fake_contract if path == contract_path else None,
    )

    def fake_load(_root, expected, **_kwargs):
        assert _root == canonical_root
        return canonical_pack_catalog[expected["option_id"]]

    monkeypatch.setattr(
        canonical_merge,
        "load_lfm2_canonical_proposal_pack",
        fake_load,
    )

    def fake_finalize(
        selected_parent,
        selected_variable,
        selected_candidates,
        _split,
        _widths_by_layer,
        _measurement_reports,
        **kwargs,
    ):
        assert selected_parent is parent
        assert selected_variable is variable
        assert tuple(selected_candidates) == physical_candidates
        assert (
            canonical_merge.variable_width_fold_module
            .build_gated_mlp_activation_cluster_plan
            is not original_clustering
        )
        assert tuple(sorted(kwargs["canonical_packs"])) == tuple(
            sorted(canonical_pack_catalog)
        )
        finalizer_calls.append(kwargs)
        return (
            evidence,
            expected_plan,
            {"frontier_sha256": frontier_sha},
        )

    monkeypatch.setattr(
        canonical_merge.canonical_finalizer,
        "build_and_seal_lfm2_cached_causal_frontier_from_canonical_packs",
        fake_finalize,
    )

    result = canonical_merge._canonical_replacement(
        parent,
        variable,
        physical_candidates,
        object(),
        tuple((8, 4, 3) for _index in range(_LAYERS)),
        (),
        parent_output_envelope_sha256=(
            evidence.parent_output_envelope_sha256
        ),
        parent_output_file_sha256=_digest("canonical-wrapper-output-file"),
        task_anchor_manifest_sha256=(
            evidence.task_anchor_manifest_sha256
        ),
        task_anchor_manifest={"schema": "test-task-anchor-manifest"},
        tokenizer_fingerprint=_digest("canonical-wrapper-tokenizer"),
        damage_weights=evidence.damage_weights,
        cluster_ridge=evidence.cluster_ridge,
        cluster_max_iterations=evidence.cluster_max_iterations,
        cluster_distance_chunk_size=(
            evidence.cluster_distance_chunk_size
        ),
        maximum_stage_fraction=0.10,
    )

    assert result == (
        evidence,
        expected_plan,
        {"frontier_sha256": frontier_sha},
    )
    assert len(finalizer_calls) == 1
    assert (
        canonical_merge.variable_width_fold_module
        .build_gated_mlp_activation_cluster_plan
        is original_clustering
    )
    with report_path.open("r", encoding="ascii") as handle:
        report = json.load(handle)
    report_body = dict(report)
    report_seal = report_body.pop("finalizer_report_sha256")
    assert report_seal == canonical_merge._sha256_json(report_body)
    assert report["schema"] == canonical_merge.FINALIZER_SCHEMA
    assert report["run_contract_sha256"] == contract_sha
    assert report["frontier_sha256"] == frontier_sha
    assert report["evidence_sha256"] == evidence.evidence_sha256
    assert report["plan_sha256"] == expected_plan.plan_sha256
    assert report["canonical_pack_count"] == len(canonical_pack_catalog)
    assert report["clustering_calls"] == 0
    assert report["parent_forward_calls"] == 0
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0
    assert report["exact_token_match_objective_used"] is False


def test_interaction_beam_reranks_complete_children_and_rolls_all_back(
    interaction_bound,
) -> None:
    parent, _variable, evidence, _beam, bound, _materialize = interaction_bound
    children = []

    def child_factory(_alternative):
        child = _TinyLfmCausal(_config(), seed=500 + len(children))
        children.append(child)
        return child

    def evaluate(context: Lfm2CumulativeInteractionContext):
        rank = context.alternative.alternative.isolated_rank
        retention = {0: 0.91, 1: 0.97}.get(rank, 0.86)
        return seal_lfm2_cumulative_interaction_quality(
            child_trace_cache_key=_digest(f"beam-cache-{rank}"),
            child_output_events_sha256=_digest(f"beam-events-{rank}"),
            intelligence_retention=retention,
            coherent_response=retention,
            trajectory_retention=retention,
            residual_path_retention=retention,
            attention_path_retention=retention,
            termination_quality=retention,
            repetition_quality=retention,
            evaluator_id="test.cached-whole-response.v1",
            metrics=(("manual_whole_response_score", retention),),
            exact_token_match_diagnostic=0.0,
        )

    run = evaluate_lfm2_interaction_beam(
        parent,
        evidence,
        bound,
        child_factory=child_factory,
        evaluator=evaluate,
        thresholds=Lfm2InteractionAcceptanceThresholds(
            minimum_intelligence_retention=0.90,
            minimum_coherent_response=0.90,
        ),
    )

    assert run.status == "accepted_candidate_found"
    assert (
        run.best_bound_alternative_sha256
        == bound.alternatives[1].bound_alternative_sha256
    )
    assert len(run.attempts) == len(bound.alternatives)
    assert all(attempt.rollback_exact for attempt in run.attempts)
    assert all(
        fingerprint_model_state(child)
        == attempt.blank_child_state_sha256
        and fingerprint_model_runtime_state(child)
        == attempt.blank_child_runtime_sha256
        for child, attempt in zip(
            children,
            run.attempts,
            strict=True,
        )
    )
    assert run.as_dict()["optimizer_steps"] == 0
    assert run.as_dict()["backward_calls"] == 0
    assert run.as_dict()["exact_token_match_objective_used"] is False


def test_interaction_beam_rejects_live_parent_evaluation_and_restores_child(
    interaction_bound,
) -> None:
    parent, _variable, evidence, _beam, bound, _materialize = interaction_bound
    children = []

    def child_factory(_alternative):
        child = _TinyLfmCausal(_config(), seed=600)
        children.append(child)
        return child

    def invalid(_context):
        parent.model.embed_tokens(torch.tensor([[1, 2]], dtype=torch.long))
        return seal_lfm2_cumulative_interaction_quality(
            child_trace_cache_key=_digest("invalid-beam-cache"),
            child_output_events_sha256=_digest("invalid-beam-events"),
            intelligence_retention=1.0,
            coherent_response=1.0,
            trajectory_retention=1.0,
            residual_path_retention=1.0,
            attention_path_retention=1.0,
            termination_quality=1.0,
            repetition_quality=1.0,
            evaluator_id="test.invalid-parent-backed.v1",
        )

    with pytest.raises(
        Lfm2InteractionBeamExecutionError,
        match="called the live parent",
    ):
        evaluate_lfm2_interaction_beam(
            parent,
            evidence,
            bound,
            child_factory=child_factory,
            evaluator=invalid,
        )

    assert tuple(
        layer.feed_forward.w1.out_features
        for layer in children[0].model.layers
    ) == (_INTERMEDIATE,) * _LAYERS
    assert all(
        parameter.requires_grad
        for parameter in children[0].parameters()
    )


def test_interaction_beam_rejects_optimizer_construction_and_restores_child(
    interaction_bound,
) -> None:
    parent, _variable, evidence, _beam, bound, _materialize = interaction_bound
    children = []

    def child_factory(_alternative):
        child = _TinyLfmCausal(_config(), seed=700)
        children.append(child)
        return child

    def invalid(context):
        torch.optim.SGD(context.child.parameters(), lr=0.1)
        return seal_lfm2_cumulative_interaction_quality(
            child_trace_cache_key=_digest("invalid-optimizer-cache"),
            child_output_events_sha256=_digest("invalid-optimizer-events"),
            intelligence_retention=1.0,
            coherent_response=1.0,
            trajectory_retention=1.0,
            residual_path_retention=1.0,
            attention_path_retention=1.0,
            termination_quality=1.0,
            repetition_quality=1.0,
            evaluator_id="test.invalid-optimizer.v1",
        )

    with pytest.raises(Exception, match="(?i)(optimizer|fold-only|training)"):
        evaluate_lfm2_interaction_beam(
            parent,
            evidence,
            bound,
            child_factory=child_factory,
            evaluator=invalid,
        )

    assert tuple(
        layer.feed_forward.w1.out_features
        for layer in children[0].model.layers
    ) == (_INTERMEDIATE,) * _LAYERS
