from __future__ import annotations

import copy
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip(
        "Installed Transformers has no Qwen3 implementation.", allow_module_level=True
    )

from neural_forge.hf_residual_attention_oracle import (  # noqa: E402
    HFResidualAttentionOracleShapeError,
    HFResidualAttentionOracleStateError,
    capture_parent_residual_attention_oracle_trace,
    fingerprint_hf_oracle_model,
    fingerprint_hf_oracle_tokenizer,
    replay_residual_attention_oracle_intervention,
    run_residual_attention_oracle_suite,
)


class _Tokenizer:
    def __init__(self, size: int) -> None:
        self.tokens = tuple(f"token-{index}" for index in range(size))
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.unk_token_id = None
        self.mask_token_id = None

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def _tiny_qwen(*, seed: int = 717, vocab_size: int = 24, rope_theta: float = 10_000.0):
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        rope_parameters={"rope_theta": rope_theta, "rope_type": "default"},
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return transformers.Qwen3ForCausalLM(config).eval()


def _hook_counts(model) -> tuple[int, int]:
    return (
        sum(len(module._forward_hooks) for module in model.modules()),
        sum(len(module._forward_pre_hooks) for module in model.modules()),
    )


@pytest.fixture(scope="module")
def parent_evidence():
    parent = _tiny_qwen()
    tokenizer_identity = fingerprint_hf_oracle_tokenizer(_Tokenizer(24))
    trace = capture_parent_residual_attention_oracle_trace(
        parent,
        [1, 4, 6],
        tokenizer_identity_sha256=tokenizer_identity,
        layer_index=0,
        group_indices=(0,),
        horizon=3,
        lane="forced",
        forced_token_ids=(7, 8, 9),
        sequence_id="tiny-forced",
    )
    return parent, tokenizer_identity, trace


def test_parent_capture_is_bound_full_delta_evidence_and_restores_state() -> None:
    parent = _tiny_qwen().train()
    parent.model.layers[0].eval()
    modes = tuple(module.training for module in parent.modules())
    hooks = _hook_counts(parent)
    config_values = (
        parent.config.use_cache,
        parent.config.output_attentions,
        parent.config.output_hidden_states,
    )
    identity_before = fingerprint_hf_oracle_model(parent)
    tokenizer_identity = fingerprint_hf_oracle_tokenizer(_Tokenizer(24))

    trace = capture_parent_residual_attention_oracle_trace(
        parent,
        [1, 3, 5],
        tokenizer_identity_sha256=tokenizer_identity,
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="forced",
        forced_token_ids=(6, 7),
    )

    assert trace.full_attention_deltas[0].shape == (1, 3, 8)
    assert trace.full_attention_deltas[1].shape == (1, 1, 8)
    assert trace.cached_trace.steps[-1].layer_key_cache.shape == (1, 1, 4, 4)
    assert trace.trace_byte_count == (
        trace.cached_trace.capture_bytes
        + sum(
            value.numel() * value.element_size()
            for value in trace.full_attention_deltas
        )
        + trace.vocabulary_chart.child_to_parent_rows.numel()
        * trace.vocabulary_chart.child_to_parent_rows.element_size()
    )
    assert trace.as_dict()["vocabulary_chart_byte_count"] == 24 * 8
    assert trace.attention_coordinate_contract.layer_count == 1
    assert trace.attention_coordinate_contract.layer_index == 0
    assert trace.as_dict()["parent_state_unchanged"] is True
    assert fingerprint_hf_oracle_model(parent) == identity_before
    assert tuple(module.training for module in parent.modules()) == modes
    assert _hook_counts(parent) == hooks
    assert (
        parent.config.use_cache,
        parent.config.output_attentions,
        parent.config.output_hidden_states,
    ) == config_values


@pytest.mark.parametrize(
    ("projection", "intervention"),
    (("k_proj", "k_cache"), ("v_proj", "v_cache")),
)
def test_post_rope_cache_oracle_causally_recovers_corrupted_k_or_v(
    parent_evidence, projection: str, intervention: str
) -> None:
    parent, tokenizer_identity, trace = parent_evidence
    candidate = copy.deepcopy(parent).train()
    hooks = _hook_counts(candidate)
    identity_before = fingerprint_hf_oracle_model(candidate)
    with torch.no_grad():
        weight = getattr(candidate.model.layers[0].self_attn, projection).weight
        weight.add_(2.0 * torch.randn_like(weight))
    corrupted_identity = fingerprint_hf_oracle_model(candidate)
    assert corrupted_identity != identity_before
    modes = tuple(module.training for module in candidate.modules())

    suite = run_residual_attention_oracle_suite(
        candidate,
        trace,
        tokenizer_identity_sha256=tokenizer_identity,
        interventions=("none", intervention),
    )

    baseline = suite.replay("none").summary_metrics()
    recovered = suite.replay(intervention).summary_metrics()
    assert baseline["centered_logit_rmse"] > 0.0
    assert recovered["centered_logit_rmse"] == pytest.approx(0.0, abs=1e-7)
    assert recovered["parent_to_candidate_kl"] == pytest.approx(0.0, abs=1e-7)
    assert suite.comparisons[1].centered_logit_rmse_recovery == pytest.approx(1.0)
    assert all(
        step.injected_key_max_abs_error == 0.0
        for step in suite.replay(intervention).steps
    )
    assert all(
        step.injected_value_max_abs_error == 0.0
        for step in suite.replay(intervention).steps
    )
    assert suite.as_dict()["causal_recovery_comparison_valid"] is True
    assert fingerprint_hf_oracle_model(candidate) == corrupted_identity
    assert tuple(module.training for module in candidate.modules()) == modes
    assert _hook_counts(candidate) == hooks


def test_delta_and_joint_oracles_recover_without_checkpoint_mutation(
    parent_evidence,
) -> None:
    parent, tokenizer_identity, trace = parent_evidence
    candidate = copy.deepcopy(parent).eval()
    with torch.no_grad():
        candidate.model.layers[0].self_attn.k_proj.weight.mul_(3.0)
        candidate.model.layers[0].self_attn.v_proj.weight.mul_(-2.0)
    identity = fingerprint_hf_oracle_model(candidate)

    suite = run_residual_attention_oracle_suite(
        candidate,
        trace,
        tokenizer_identity_sha256=tokenizer_identity,
    )

    assert tuple(replay.intervention for replay in suite.replays) == (
        "none",
        "k_cache",
        "v_cache",
        "attention_delta",
        "kv_cache",
        "kv_cache_and_attention_delta",
    )
    for intervention in ("attention_delta", "kv_cache_and_attention_delta"):
        metrics = suite.replay(intervention).summary_metrics()
        assert metrics["centered_logit_rmse"] == pytest.approx(0.0, abs=1e-7)
        assert metrics["parent_to_candidate_kl"] == pytest.approx(0.0, abs=1e-7)
    assert suite.replay("kv_cache").summary_metrics()[
        "centered_logit_rmse"
    ] == pytest.approx(0.0, abs=1e-7)
    delta = suite.replay("attention_delta")
    joint = suite.replay("kv_cache_and_attention_delta")
    assert all(
        torch.equal(delta_step.candidate_logits, joint_step.candidate_logits)
        for delta_step, joint_step in zip(delta.steps, joint.steps, strict=True)
    )
    joint_comparison = next(
        value
        for value in suite.as_dict()["comparisons"]
        if value["intervention"] == "kv_cache_and_attention_delta"
    )
    assert joint_comparison["independent_causal_evidence"] is False
    assert suite.as_dict()["redundancy_controls"] == ["kv_cache_and_attention_delta"]
    assert fingerprint_hf_oracle_model(candidate) == identity


def test_qwen_codebook_projection_runtime_uses_same_cache_oracle_path() -> None:
    from neural_forge.hf_qwen_codebook_model import prepare_qwen3_codebook_model

    tokenizer = _Tokenizer(24)
    parent = _tiny_qwen()
    parent.config.tie_word_embeddings = True
    parent.tie_weights()
    tokenizer_identity = fingerprint_hf_oracle_tokenizer(tokenizer)
    trace = capture_parent_residual_attention_oracle_trace(
        parent,
        [1, 4, 6],
        tokenizer_identity_sha256=tokenizer_identity,
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="forced",
        forced_token_ids=(7, 8),
    )
    candidate = prepare_qwen3_codebook_model(
        copy.deepcopy(parent),
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=1,
    ).eval()
    assert type(candidate.model.layers[0].self_attn.q_proj).__name__ == (
        "AttentionCodebookProjectionView"
    )

    replay = replay_residual_attention_oracle_intervention(
        candidate,
        trace,
        tokenizer_identity_sha256=tokenizer_identity,
        intervention="none",
    )

    assert replay.full_horizon_valid is True
    assert replay.summary_metrics()["centered_logit_rmse"] == pytest.approx(
        0.0, abs=1e-7
    )


def test_padded_parent_logits_are_canonicalized_through_nonidentity_chart() -> None:
    parent = _tiny_qwen(vocab_size=24)
    rows = torch.arange(1, 24, 2, dtype=torch.int64)
    tokenizer_identity = fingerprint_hf_oracle_tokenizer(_Tokenizer(12))
    trace = capture_parent_residual_attention_oracle_trace(
        parent,
        [0, 2, 4],
        tokenizer_identity_sha256=tokenizer_identity,
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="forced",
        forced_token_ids=(5, 7),
        parent_token_rows=rows,
    )

    assert trace.cached_trace.prompt_token_ids == (0, 2, 4)
    assert trace.parent_prompt_token_ids == (1, 5, 9)
    assert trace.cached_trace.selected_token_ids == (5, 7)
    assert trace.parent_selected_token_ids == (11, 15)
    assert trace.cached_trace.consumed_token_ids == (5,)
    assert trace.cached_trace.steps[0].action_logits.shape == (1, 12)
    assert trace.cached_trace.capture_bytes == sum(
        step.captured_byte_count for step in trace.cached_trace.steps
    )
    assert trace.as_dict()["prompt_rows_identity"] is False
    with torch.inference_mode():
        expected = (
            parent(torch.tensor([[1, 5, 9]])).logits[:, -1, :].index_select(-1, rows)
        )
    assert torch.allclose(
        trace.cached_trace.steps[0].action_logits,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

    candidate = _tiny_qwen(seed=919, vocab_size=12)
    candidate.register_buffer(
        "parent_token_rows", rows.to(dtype=torch.int32), persistent=True
    )
    replay = replay_residual_attention_oracle_intervention(
        candidate,
        trace,
        tokenizer_identity_sha256=tokenizer_identity,
        intervention="none",
    )
    assert replay.full_horizon_valid is True
    assert all(step.candidate_logits.shape == (1, 12) for step in replay.steps)

    wrong = _tiny_qwen(seed=920, vocab_size=12)
    wrong.register_buffer(
        "parent_token_rows", torch.arange(12, dtype=torch.int32), persistent=True
    )
    with pytest.raises(HFResidualAttentionOracleStateError, match="parent_token_rows"):
        replay_residual_attention_oracle_intervention(
            wrong,
            trace,
            tokenizer_identity_sha256=tokenizer_identity,
            intervention="none",
        )


def test_greedy_parent_action_on_unmapped_padding_row_fails_closed() -> None:
    parent = _tiny_qwen(vocab_size=24)
    rows = torch.arange(1, 24, 2, dtype=torch.int64)
    prompt = torch.tensor([[1, 3, 5]])
    with torch.inference_mode():
        hidden = parent.model(prompt).last_hidden_state[:, -1, :].clone()
        parent.lm_head.weight.zero_()
        parent.lm_head.weight[0].copy_(100.0 * hidden[0])
        assert int(torch.argmax(parent(prompt).logits[:, -1, :], dim=-1).item()) == 0

    with pytest.raises(HFResidualAttentionOracleStateError, match="unmapped padded"):
        capture_parent_residual_attention_oracle_trace(
            parent,
            [0, 1, 2],
            tokenizer_identity_sha256=fingerprint_hf_oracle_tokenizer(_Tokenizer(12)),
            layer_index=0,
            group_indices=(0,),
            horizon=1,
            lane="greedy",
            parent_token_rows=rows,
        )


def test_autoregressive_feedback_stops_before_using_wrong_parent_prefix() -> None:
    parent = _tiny_qwen(seed=1719)
    tokenizer_identity = fingerprint_hf_oracle_tokenizer(_Tokenizer(24))
    prompt = [1, 5, 7]
    with torch.inference_mode():
        parent_action = int(
            torch.argmax(parent(torch.tensor([prompt])).logits[:, -1], dim=-1).item()
        )
    if parent_action == 0:
        prompt = [1, 6, 9]
        with torch.inference_mode():
            parent_action = int(
                torch.argmax(
                    parent(torch.tensor([prompt])).logits[:, -1], dim=-1
                ).item()
            )
    assert parent_action != 0
    trace = capture_parent_residual_attention_oracle_trace(
        parent,
        prompt,
        tokenizer_identity_sha256=tokenizer_identity,
        layer_index=0,
        group_indices=(0,),
        horizon=3,
        lane="greedy",
    )
    candidate = copy.deepcopy(parent).eval()
    with torch.no_grad():
        candidate.lm_head.weight.zero_()

    forced = replay_residual_attention_oracle_intervention(
        candidate,
        trace,
        tokenizer_identity_sha256=tokenizer_identity,
        intervention="k_cache",
        feedback="forced-cache-recurrent",
    )
    autoregressive = replay_residual_attention_oracle_intervention(
        candidate,
        trace,
        tokenizer_identity_sha256=tokenizer_identity,
        intervention="k_cache",
        feedback="autoregressive",
    )

    assert forced.completed_horizon == 3
    assert forced.first_action_divergence_step == 0
    assert autoregressive.completed_horizon == 1
    assert autoregressive.first_action_divergence_step == 0
    assert autoregressive.steps[0].candidate_greedy_token_id == 0
    assert autoregressive.steps[0].consumed_token_id is None
    assert (
        autoregressive.as_dict()["feedback_intervention_stopped_at_divergence"] is True
    )


def test_autoregressive_feedback_does_not_claim_a_stop_without_divergence() -> None:
    parent = _tiny_qwen(seed=1771)
    tokenizer_identity = fingerprint_hf_oracle_tokenizer(_Tokenizer(24))
    trace = capture_parent_residual_attention_oracle_trace(
        parent,
        [1, 5, 7],
        tokenizer_identity_sha256=tokenizer_identity,
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="greedy",
    )

    replay = replay_residual_attention_oracle_intervention(
        copy.deepcopy(parent),
        trace,
        tokenizer_identity_sha256=tokenizer_identity,
        intervention="none",
        feedback="autoregressive",
    )

    assert replay.first_action_divergence_step is None
    assert replay.full_horizon_valid is True
    assert replay.as_dict()["feedback_intervention_stopped_at_divergence"] is False


def test_rope_coordinate_mismatch_is_rejected_before_raw_cache_substitution() -> None:
    parent = _tiny_qwen(seed=1881, rope_theta=10_000.0)
    tokenizer_identity = fingerprint_hf_oracle_tokenizer(_Tokenizer(24))
    trace = capture_parent_residual_attention_oracle_trace(
        parent,
        [1, 4, 6],
        tokenizer_identity_sha256=tokenizer_identity,
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="forced",
        forced_token_ids=(7, 8),
    )
    candidate = _tiny_qwen(seed=1882, rope_theta=500_000.0)
    candidate.load_state_dict(parent.state_dict())
    assert all(
        torch.equal(parent_value, candidate_value)
        for parent_value, candidate_value in zip(
            parent.state_dict().values(), candidate.state_dict().values(), strict=True
        )
    )

    with pytest.raises(
        HFResidualAttentionOracleShapeError,
        match="attention-coordinate contract.*rope_parameters",
    ):
        replay_residual_attention_oracle_intervention(
            candidate,
            trace,
            tokenizer_identity_sha256=tokenizer_identity,
            intervention="k_cache",
        )


def test_trace_tokenizer_topology_and_feedback_contracts_fail_closed(
    parent_evidence,
) -> None:
    parent, tokenizer_identity, trace = parent_evidence
    candidate = copy.deepcopy(parent).eval()
    wrong_tokenizer = "f" * 64
    if wrong_tokenizer == tokenizer_identity:
        wrong_tokenizer = "e" * 64

    with pytest.raises(HFResidualAttentionOracleStateError, match="tokenizer"):
        replay_residual_attention_oracle_intervention(
            candidate,
            trace,
            tokenizer_identity_sha256=wrong_tokenizer,
            intervention="none",
        )
    with pytest.raises(HFResidualAttentionOracleShapeError, match="greedy parent"):
        replay_residual_attention_oracle_intervention(
            candidate,
            trace,
            tokenizer_identity_sha256=tokenizer_identity,
            intervention="none",
            feedback="autoregressive",
        )
    incompatible = _tiny_qwen()
    incompatible.config.num_attention_heads = 1
    with pytest.raises(HFResidualAttentionOracleShapeError, match="topology"):
        replay_residual_attention_oracle_intervention(
            incompatible,
            trace,
            tokenizer_identity_sha256=tokenizer_identity,
            intervention="none",
        )


def test_mutated_parent_trace_tensor_is_rejected_before_candidate_forward(
    parent_evidence,
) -> None:
    parent, tokenizer_identity, trace = parent_evidence
    changed_delta = trace.full_attention_deltas[0].clone()
    changed_delta[..., 0] += 1.0
    changed = replace(
        trace,
        full_attention_deltas=(changed_delta, *trace.full_attention_deltas[1:]),
    )

    with pytest.raises(HFResidualAttentionOracleStateError, match="identity"):
        replay_residual_attention_oracle_intervention(
            copy.deepcopy(parent),
            changed,
            tokenizer_identity_sha256=tokenizer_identity,
            intervention="attention_delta",
        )
