from __future__ import annotations

import copy
from dataclasses import replace

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
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAQKVOResidualError,
    DenseGQASite,
)
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
    restore_target_native_factor_payload,
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
        lcp_at_least_4_count=int(lcp_tokens >= 4),
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
        == 0
    )
    safe_acquired = replace(
        acquired,
        metrics=_selection_metrics(
            exact_rollouts=1,
            action_agreements=3,
            token_matches=3,
            lcp_tokens=2,
            mean_logprob=-2.0,
        ),
    )
    assert (
        select_behavior_checkpoint(
            (baseline, safe_acquired), selection_policy="rollout-first-acquisition"
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

    repetitive_isolated_matches = BehaviorCheckpoint(
        step=3,
        metrics=_selection_metrics(
            exact_rollouts=0,
            action_agreements=8,
            token_matches=7,
            lcp_tokens=1,
            mean_logprob=-10.0,
        ),
        factor_state_sha256="5" * 64,
        pareto_safe_vs_step0=False,
    )
    assert (
        select_behavior_checkpoint(
            (baseline, repetitive_isolated_matches),
            selection_policy="rollout-first-acquisition",
        ).step
        == 0
    )


def test_divergence_curriculum_and_staged_factor_rates_are_exact() -> None:
    schedule = BehaviorDistillationSchedule(
        steps=32,
        learning_rate=3e-4,
        fit_horizon=8,
        parent_only_steps=8,
        warmup_fit_horizon=4,
        solved_prefix_loss_multiplier=0.5,
        divergence_loss_multiplier=4.0,
        post_divergence_loss_multiplier=2.0,
        post_divergence_positions=1,
        child_lane_starts_at_divergence=True,
        max_child_lane_prefixes=3,
        child_repeat_stop_run_length=3,
        qk_learning_rate_scale=0.5,
        vo_learning_rate_scale=1.0,
        gud_learning_rate_scale=0.5,
        right_factor_learning_rate_scale=0.25,
        learning_rate_decay_step_1=8,
        learning_rate_decay_scale_1=2.0 / 3.0,
        learning_rate_decay_step_2=20,
        learning_rate_decay_scale_2=0.5,
        gud_frozen_steps=8,
    )
    parent = (1, 2, 3, 4, 5, 6, 7, 8)
    child = (1, 9, 9, 9, 9, 9, 9, 9)
    divergence = behavior_distillation._first_divergence(parent, child)
    assert divergence == 1
    assert behavior_distillation._divergence_position_weights(
        len(parent), divergence, schedule
    )[:4] == pytest.approx((0.5, 4.0, 2.0, 1.0))
    assert behavior_distillation._child_lane_indices(child, divergence, schedule) == (
        1,
        2,
        3,
    )
    assert behavior_distillation._factor_learning_rate(
        "qkvo.q_left", schedule, step=1
    ) == pytest.approx(1.5e-4)
    assert behavior_distillation._factor_learning_rate(
        "qkvo.v_right", schedule, step=9
    ) == pytest.approx(5e-5)
    assert behavior_distillation._factor_learning_rate(
        "gud.gate_left", schedule, step=21
    ) == pytest.approx(5e-5)
    legacy_schedule = BehaviorDistillationSchedule(
        steps=2,
        learning_rate=1e-3,
        fit_horizon=8,
    )
    assert behavior_distillation._child_lane_indices(
        child, divergence, legacy_schedule
    ) == tuple(range(8))


def test_gud_freeze_skips_adam_state_until_thaw() -> None:
    torch.manual_seed(109)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 3, dims=0))
    transaction = _transaction(child)
    domains = {
        "fit": _domain("fit", [[1, 2]], "a" * 64),
        "calibration": _domain("calibration", [[3, 4]], "b" * 64),
        "sealed": _domain("sealed", [[5, 6]], "c" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=3,
        learning_rate=1e-3,
        fit_horizon=1,
        evaluation_horizon=1,
        gud_frozen_steps=1,
        top_k=2,
    )
    gud_ids = {
        id(value)
        for value in transaction.swiglu_handle.registry.parameters(recurse=False)
    }
    snapshots: list[dict[int, float | None]] = []

    class _AuditedAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            result = super().step(closure)
            snapshot: dict[int, float | None] = {}
            for group in self.param_groups:
                for parameter in group["params"]:
                    raw = self.state.get(parameter, {}).get("step")
                    snapshot[id(parameter)] = None if raw is None else float(raw.item())
            snapshots.append(snapshot)
            return result

    train_target_native_behavior_distillation(
        transaction,
        parent,
        domains,
        schedule,
        optimizer_factory=lambda parameters: _AuditedAdamW(
            parameters,
            lr=schedule.learning_rate,
            weight_decay=0.0,
        ),
    )
    assert all(snapshots[0][parameter_id] is None for parameter_id in gud_ids)
    assert all(snapshots[1][parameter_id] == 1.0 for parameter_id in gud_ids)
    assert all(snapshots[2][parameter_id] == 2.0 for parameter_id in gud_ids)
    transaction.rollback()


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


def test_factor_only_payload_restores_exact_continuation_and_rejects_tamper() -> None:
    torch.manual_seed(107)
    model = _StrictToyTarget()
    transaction = _transaction(model)
    with torch.no_grad():
        for index, value in enumerate(transaction.output_trainable_parameters()):
            value.add_(float(index + 1) / 1000.0)
    selected = capture_target_native_factor_state(transaction)
    payload = selected.as_payload()
    with torch.no_grad():
        for value in transaction.output_trainable_parameters():
            value.zero_()
    restored = restore_target_native_factor_payload(transaction, payload)
    assert restored.factor_state_sha256 == selected.factor_state_sha256

    tampered = selected.as_payload()
    tampered["factor_tensors"][selected.names[0]].reshape(-1)[0] += 1.0
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="shape/dtype/hash authentication",
    ):
        restore_target_native_factor_payload(transaction, tampered)
    transaction.rollback()


def test_low_precision_child_uses_fp32_factors_and_adam_moments() -> None:
    torch.manual_seed(104)
    model = _StrictToyTarget().to(dtype=torch.float16)
    transaction = _transaction(model)
    factors = transaction.output_trainable_parameters()
    installation = transaction.installation_report()
    assert factors
    assert {value.dtype for value in factors} == {torch.float32}
    assert installation["dense_qkvo_installation"]["factor_dtype"] == ("torch.float32")
    assert installation["dense_gate_up_down_installation"]["factor_dtype"] == (
        "torch.float32"
    )

    optimizer = torch.optim.AdamW(factors, lr=1e-3, weight_decay=0.0, eps=1e-8)
    for index, value in enumerate(factors):
        value.grad = torch.zeros_like(value)
        if index % 2:
            value.grad.reshape(-1)[0] = 1e-4
    optimizer.step()

    assert all(bool(torch.isfinite(value).all()) for value in factors)
    moment_dtypes = {
        state_value.dtype
        for value in factors
        for state_value in optimizer.state[value].values()
        if isinstance(state_value, Tensor) and state_value.is_floating_point()
    }
    assert moment_dtypes == {torch.float32}
    transaction.rollback()


def test_poisoned_factor_snapshot_restore_and_combined_rollback_are_exact() -> None:
    torch.manual_seed(105)
    model = _StrictToyTarget()
    exact = exact_unique_parameter_count(model)
    transaction = _transaction(model)
    step0 = capture_target_native_factor_state(transaction)
    with torch.no_grad():
        transaction.qkvo_handle.adapter_registry.q_left.reshape(-1)[0] = torch.nan
        transaction.swiglu_handle.registry.gate_left.reshape(-1)[0] = torch.inf
    with pytest.raises(DenseGQAQKVOResidualError, match="NaN|infinity"):
        transaction.assert_live()

    restore_target_native_factor_state(transaction, step0)
    transaction.assert_live()
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        step0.factor_state_sha256
    )

    with torch.no_grad():
        transaction.qkvo_handle.adapter_registry.k_left.reshape(-1)[0] = torch.nan
        transaction.swiglu_handle.registry.up_left.reshape(-1)[0] = torch.inf
    transaction.rollback()
    assert exact_unique_parameter_count(model) == exact


def test_poisoning_optimizer_preserves_original_error_and_restores_step_zero() -> None:
    class _PoisoningOptimizer(torch.optim.Optimizer):
        def __init__(self, parameters, *, lr: float) -> None:
            super().__init__(parameters, {"lr": lr})

        @torch.no_grad()
        def step(self, closure=None):
            del closure
            self.param_groups[0]["params"][0].reshape(-1)[0] = torch.nan

    torch.manual_seed(106)
    child = _StrictToyTarget()
    parent = copy.deepcopy(child)
    with torch.no_grad():
        parent.lm_head.weight.copy_(torch.roll(parent.lm_head.weight, 3, dims=0))
    transaction = _transaction(child)
    step0 = capture_target_native_factor_state(transaction)
    domains = {
        "fit": _domain("fit", [[1, 2, 3]], "a" * 64),
        "calibration": _domain("calibration", [[4, 5, 6]], "b" * 64),
        "sealed": _domain("sealed", [[7, 8, 9]], "c" * 64),
    }
    schedule = BehaviorDistillationSchedule(
        steps=1,
        learning_rate=1e-3,
        fit_horizon=1,
        evaluation_horizon=1,
        top_k=2,
    )
    with pytest.raises(
        TargetNativeBehaviorDistillationError,
        match="Optimizer produced non-finite compiler factors",
    ):
        train_target_native_behavior_distillation(
            transaction,
            parent,
            domains,
            schedule,
            optimizer_factory=lambda parameters: _PoisoningOptimizer(
                parameters, lr=schedule.learning_rate
            ),
        )
    transaction.assert_live()
    assert capture_target_native_factor_state(transaction).factor_state_sha256 == (
        step0.factor_state_sha256
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
    initial_factor_payload = capture_target_native_factor_state(
        transaction
    ).as_payload()
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
        initial_factor_payload=initial_factor_payload,
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
    assert result.training_audit.parent_greedy_cache_prompt_count == 2
    assert result.training_audit.parent_greedy_cache_forward_count == 4
    assert result.training_audit.parent_output_forward_count == 16
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
    assert evidence["continued_from_factor_pack"] is True
    assert (
        evidence["initial_factor_state_sha256"]
        == (initial_factor_payload["factor_state_sha256"])
    )
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
