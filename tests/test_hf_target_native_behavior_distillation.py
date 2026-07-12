from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import neural_forge.hf_target_native_behavior_distillation as behavior_distillation
from neural_forge.hf_arch_layout import (
    AttentionLayout,
    DecoderBlockLayout,
    DecoderLayout,
    EmbeddingLayout,
    MLPLayout,
    NormLayout,
    PositionLayout,
    SERIAL_PRE_NORM_RESIDUAL,
)
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import DenseGQASite
from neural_forge.hf_factorized_vocab import exact_unique_parameter_count
from neural_forge.hf_progressive_target_bridge import (
    install_exact_target_compiler_transaction,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BehaviorCheckpoint,
    BehaviorDistillationSchedule,
    CandidateBehaviorThresholds,
    CausalBehaviorMetrics,
    TargetNativeBehaviorDistillationError,
    capture_target_native_factor_state,
    load_behavior_domain_suite,
    restore_target_native_factor_state,
    select_behavior_checkpoint,
    train_target_native_behavior_distillation,
)


class _ToyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        query = self.q_proj(hidden)
        key = self.k_proj(hidden).repeat_interleave(2, dim=-1)
        value = self.v_proj(hidden).repeat_interleave(2, dim=-1)
        return self.o_proj(query + key + value)


class _ToySwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 12, bias=False)
        self.up_proj = nn.Linear(8, 12, bias=False)
        self.down_proj = nn.Linear(12, 8, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _ToyAttention()
        self.mlp = _ToySwiGLU()

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.self_attn(hidden)
        return hidden + self.mlp(hidden)


class _StrictToyTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(19, 8)
        self.layers = nn.ModuleList([_ToyBlock(), _ToyBlock()])
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 19, bias=False)
        self.internal_output_request_count = 0

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        use_cache: bool = False,
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        logits_to_keep: int | None = None,
    ) -> Tensor:
        del attention_mask, return_dict, logits_to_keep
        if use_cache or output_attentions or output_hidden_states:
            self.internal_output_request_count += 1
            raise AssertionError("Output-only trainer requested model internals.")
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.norm(hidden))


def _layout() -> DecoderLayout:
    attention_norm = NormLayout(
        path="norm",
        kind="rms_norm",
        placement="pre_attention",
        epsilon=1e-5,
        has_bias=False,
    )
    mlp_norm = NormLayout(
        path="norm",
        kind="rms_norm",
        placement="pre_mlp",
        epsilon=1e-5,
        has_bias=False,
    )
    blocks = []
    for index in range(2):
        block = f"layers.{index}"
        attention = f"{block}.self_attn"
        mlp = f"{block}.mlp"
        blocks.append(
            DecoderBlockLayout(
                index=index,
                path=block,
                attention_input_norm=attention_norm,
                attention=AttentionLayout(
                    module_path=attention,
                    projection_style="split_qkv",
                    fused_qkv_path=None,
                    query_path=f"{attention}.q_proj",
                    key_path=f"{attention}.k_proj",
                    value_path=f"{attention}.v_proj",
                    output_path=f"{attention}.o_proj",
                    query_heads=4,
                    key_value_heads=2,
                    head_dim=2,
                    hidden_size=8,
                    projection_weight_layout="output_input",
                    projection_bias=False,
                    attention_dropout=0.0,
                ),
                mlp_input_norm=mlp_norm,
                mlp=MLPLayout(
                    module_path=mlp,
                    kind="gated",
                    input_projection_path=None,
                    gate_projection_path=f"{mlp}.gate_proj",
                    up_projection_path=f"{mlp}.up_proj",
                    down_projection_path=f"{mlp}.down_proj",
                    activation_path=f"{mlp}.act_fn",
                    activation_name="silu",
                    activation_module_type="torch.nn.SiLU",
                    hidden_size=8,
                    intermediate_size=12,
                    projection_weight_layout="output_input",
                    projection_bias=False,
                ),
                residual=SERIAL_PRE_NORM_RESIDUAL,
            )
        )
    return DecoderLayout(
        family="llama",
        model_type="strict_output_only_toy",
        model_class="_StrictToyTarget",
        hidden_size=8,
        intermediate_size=12,
        blocks_path="layers",
        embeddings=EmbeddingLayout(
            token_path="embedding",
            position_path=None,
            dropout_path=None,
            vocab_size=19,
            hidden_size=8,
        ),
        position=PositionLayout(
            kind="rope",
            operator_paths=(),
            max_positions=64,
            rotary_dim=2,
            rope_theta=10_000.0,
            rope_type="default",
        ),
        blocks=tuple(blocks),
        final_norm=NormLayout(
            path="norm",
            kind="rms_norm",
            placement="final",
            epsilon=1e-5,
            has_bias=True,
        ),
        lm_head_path="lm_head",
        tie_word_embeddings=False,
    )


def _domain(split: str, prompts: list[list[int]], artifact: str):
    return load_behavior_domain_suite(
        split,
        {"prompts": [{"token_ids": value} for value in prompts]},
        artifact_sha256=artifact,
    )


def _transaction(model: _StrictToyTarget):
    return install_exact_target_compiler_transaction(
        model,
        exact_unique_parameters=exact_unique_parameter_count(model),
        qkvo_sites=(DenseGQASite(0, 0), DenseGQASite(1, 1)),
        qkvo_rank=2,
        swiglu_layer_indices=(0, 1),
        swiglu_rank=2,
        layout=_layout(),
        initialization_seed=31,
    )


def _selection_metrics(
    *,
    exact_rollouts: int,
    action_agreements: int,
    token_matches: int,
    lcp_tokens: int,
    mean_logprob: float,
) -> CausalBehaviorMetrics:
    return CausalBehaviorMetrics(
        split="calibration-selection",
        action_count=10,
        action_agreement_count=action_agreements,
        action_accuracy=action_agreements / 10,
        parent_action_mean_logprob=mean_logprob,
        parent_action_topk_fraction=0.4 if mean_logprob > -5.0 else 0.3,
        parent_action_mean_rank=4.0 if mean_logprob > -5.0 else 5.0,
        parent_action_worst_rank=8 if mean_logprob > -5.0 else 9,
        rollout_count=2,
        exact_rollout_count=exact_rollouts,
        exact_rollout_fraction=exact_rollouts / 2,
        rollout_token_count=8,
        token_match_count=token_matches,
        token_match_fraction=token_matches / 8,
        lcp_token_count=lcp_tokens,
        lcp_fraction=lcp_tokens / 8,
        repeat_fraction_excess=0.0 if mean_logprob > -5.0 else 0.25,
        adjacent_repeat_fraction_excess=0.0 if mean_logprob > -5.0 else 0.25,
        longest_run_excess=0 if mean_logprob > -5.0 else 1,
        report_sha256=("a" if mean_logprob > -5.0 else "b") * 64,
    )


def test_loader_derives_prompt_identity_and_rejects_false_binding() -> None:
    suite = _domain("fit", [[1, 2, 3], [4, 5]], "a" * 64)
    assert suite.prompts[0].prompt_id == (f"prompt-{suite.prompts[0].token_ids_sha256}")
    assert suite.as_dict()["raw_prompt_text_retained"] is False
    assert _domain("candidate", [[6, 7]], "b" * 64).split_name == "candidate"
    with pytest.raises(TargetNativeBehaviorDistillationError, match="does not match"):
        load_behavior_domain_suite(
            "fit",
            {"prompts": [{"token_ids": [1, 2], "prompt_id": "invented"}]},
            artifact_sha256="a" * 64,
        )


def test_rollout_first_policy_acquires_behavior_without_strict_dominance() -> None:
    baseline = BehaviorCheckpoint(
        step=0,
        metrics=_selection_metrics(
            exact_rollouts=0,
            action_agreements=2,
            token_matches=2,
            lcp_tokens=1,
            mean_logprob=-1.0,
        ),
        factor_state_sha256="1" * 64,
        pareto_safe_vs_step0=True,
    )
    acquired = BehaviorCheckpoint(
        step=1,
        metrics=_selection_metrics(
            exact_rollouts=1,
            action_agreements=3,
            token_matches=3,
            lcp_tokens=2,
            mean_logprob=-10.0,
        ),
        factor_state_sha256="2" * 64,
        pareto_safe_vs_step0=False,
    )
    assert (
        select_behavior_checkpoint(
            (baseline, acquired), selection_policy="strict-nonregression"
        ).step
        == 0
    )
    assert (
        select_behavior_checkpoint(
            (baseline, acquired), selection_policy="rollout-first-acquisition"
        ).step
        == 1
    )
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=1e-3,
        selection_policy="rollout-first-acquisition",
    )
    assert schedule.as_dict()["selection_policy"] == "rollout-first-acquisition"

    first_action_only = BehaviorCheckpoint(
        step=1,
        metrics=_selection_metrics(
            exact_rollouts=0,
            action_agreements=4,
            token_matches=2,
            lcp_tokens=1,
            mean_logprob=-0.5,
        ),
        factor_state_sha256="3" * 64,
        pareto_safe_vs_step0=False,
    )
    sequential_progress = BehaviorCheckpoint(
        step=2,
        metrics=_selection_metrics(
            exact_rollouts=0,
            action_agreements=3,
            token_matches=4,
            lcp_tokens=3,
            mean_logprob=-2.0,
        ),
        factor_state_sha256="4" * 64,
        pareto_safe_vs_step0=False,
    )
    assert (
        select_behavior_checkpoint(
            (baseline, first_action_only, sequential_progress),
            selection_policy="rollout-first-acquisition",
        ).step
        == 2
    )


def test_factor_state_capture_restore_is_exact_and_factor_only() -> None:
    torch.manual_seed(101)
    model = _StrictToyTarget()
    transaction = _transaction(model)
    state = capture_target_native_factor_state(transaction)
    assert len(state.names) == 14
    assert all(name.startswith(("qkvo.", "gud.")) for name in state.names)
    assert state.as_manifest()["native_child_state_included"] is False
    with torch.no_grad():
        transaction.qkvo_handle.adapter_registry.q_left.add_(0.125)
        transaction.swiglu_handle.registry.gate_left.sub_(0.25)
    assert capture_target_native_factor_state(transaction).factor_state_sha256 != (
        state.factor_state_sha256
    )
    restore_target_native_factor_state(transaction, state)
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        state.factor_state_sha256
    )
    transaction.rollback()


def test_output_only_training_audits_all_factors_and_retains_step_zero() -> None:
    torch.manual_seed(102)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 5, dims=0))
        parent.layers[0].self_attn.q_proj.weight.mul_(1.35)
    transaction = _transaction(child)
    domains = {
        "fit": _domain("fit", [[1, 2, 3], [4, 5, 6]], "1" * 64),
        "calibration": _domain("calibration", [[7, 8], [9, 10]], "2" * 64),
        "sealed": _domain("sealed", [[11, 12], [13, 14]], "3" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=3,
        learning_rate=0.04,
        checkpoint_steps=(1, 2),
        fit_horizon=2,
        evaluation_horizon=2,
        fit_prompts_per_step=2,
        top_k=3,
        gradient_clip_norm=5.0,
    )
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        seed=77,
        optimizer_factory=lambda parameters: torch.optim.SGD(
            parameters, lr=schedule.learning_rate
        ),
    )
    assert [value.step for value in result.checkpoints] == [0, 1, 2, 3]
    assert result.checkpoints[0].factor_state_sha256
    assert result.selected_step in {0, 1, 2, 3}
    selected_checkpoint = next(
        value for value in result.checkpoints if value.step == result.selected_step
    )
    assert result.factor_pack.factor_state_sha256 == (
        selected_checkpoint.factor_state_sha256
    )
    assert result.training_audit.parent_greedy_prefix_count == 12
    assert result.training_audit.child_on_policy_prefix_count == 12
    assert result.training_audit.gradient_activation.complete is True
    assert len(result.training_audit.gradient_activation.factor_names) == 14
    assert result.candidate_metrics.action_count == 4
    assert result.candidate_metrics.rollout_count == 2
    assert result.fit_metrics.split == "calibration-exploratory"
    assert result.sealed_metrics is not None
    assert result.sealed_metrics.split == "heldout-evaluation"
    assert result.factor_pack_payload()["optimizer_state_included"] is False
    evidence = result.evidence_payload()
    assert behavior_distillation._canonical_sha256(evidence) == result.evidence_sha256
    assert evidence["factor_pack_manifest"] == result.factor_pack.as_manifest()
    assert (
        evidence["training_audit"]["gradient_activation"][
            "every_qkvo_gud_factor_gradient_activated"
        ]
        is True
    )
    assert evidence["internal_numeric_matching_used"] is False
    assert result.as_dict()["evidence_payload"] == evidence
    assert parent.internal_output_request_count == 0
    assert child.internal_output_request_count == 0
    transaction.rollback()


def test_disjointness_and_candidate_gate_keep_sealed_locked(monkeypatch) -> None:
    torch.manual_seed(103)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 7, dims=0))
    transaction = _transaction(child)
    shared = list(range(1, 9))
    overlap = {
        "fit": _domain("fit", [shared + [9]], "4" * 64),
        "calibration": _domain("calibration", [shared + [10]], "5" * 64),
        "sealed": _domain("sealed", [[11, 12]], "6" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=2,
        learning_rate=0.02,
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )
    with pytest.raises(TargetNativeBehaviorDistillationError, match="prefix overlaps"):
        train_target_native_behavior_distillation(
            transaction, parent, overlap, schedule
        )

    disjoint = {
        "fit": _domain("fit", [[1, 2]], "7" * 64),
        "calibration": _domain("calibration", [[3, 4]], "8" * 64),
        "sealed": _domain("sealed", [[5, 6]], "9" * 64),
    }
    evaluated_domains: list[str] = []
    evaluator = behavior_distillation.evaluate_behavior_domain

    def audited_evaluator(*args, **kwargs):
        evaluated_domains.append(args[2].split_name)
        return evaluator(*args, **kwargs)

    monkeypatch.setattr(
        behavior_distillation, "evaluate_behavior_domain", audited_evaluator
    )
    result = train_target_native_behavior_distillation(
        transaction,
        parent,
        disjoint,
        schedule,
        candidate_thresholds=CandidateBehaviorThresholds(
            minimum_action_accuracy=1.0,
            minimum_exact_rollout_fraction=1.0,
            minimum_token_match_fraction=1.0,
            minimum_lcp_fraction=1.0,
        ),
    )
    assert result.candidate_gate_passed is False
    assert result.sealed_evaluation_unlocked is False
    assert result.sealed_metrics is None
    assert result.as_dict()["sealed_used_for_selection"] is False
    assert "sealed" not in evaluated_domains
    transaction.rollback()
