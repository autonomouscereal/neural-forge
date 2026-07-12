from __future__ import annotations

from dataclasses import dataclass, replace

import pytest


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from neural_forge.hf_cached_attention_trace import (  # noqa: E402
    HFCachedAttentionTraceLimitError,
    HFCachedAttentionTraceLimits,
    HFCachedAttentionTraceShapeError,
    HFCachedAttentionTraceStateError,
    build_residual_attention_candidate_evidence,
    collect_hf_cached_attention_trace,
)
from neural_forge.hf_residual_attention_fold import (  # noqa: E402
    score_residual_attention_candidate,
)


class _ToyConfig:
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 2
    hidden_size = 8
    use_cache = False
    output_attentions = True
    output_hidden_states = True
    gradient_checkpointing = False


class _Tokenizer:
    def __init__(self, size: int) -> None:
        self.tokens = tuple(f"token-{index}" for index in range(size))
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


class _ToyCacheLayer:
    def __init__(self, *, drift: bool, value_drift: bool) -> None:
        self.keys = None
        self.values = None
        self.drift = drift
        self.value_drift = value_drift

    def update(self, key: torch.Tensor, value: torch.Tensor):
        if self.keys is None:
            self.keys = key.clone()
            self.values = value.clone()
        else:
            self.keys = torch.cat((self.keys, key), dim=-2)
            self.values = torch.cat((self.values, value), dim=-2)
            if self.drift and key.shape[-2] == 1:
                self.keys[..., 0, :] += 4.0
        if self.value_drift:
            self.values[..., -1, :] += 2.0
        return self.keys, self.values


class _ToyCache:
    def __init__(self, layers: int, *, drift: bool, value_drift: bool) -> None:
        self.layers = [
            _ToyCacheLayer(drift=drift, value_drift=value_drift)
            for _ in range(layers)
        ]

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_index: int,
    ):
        return self.layers[layer_index].update(key, value)

    def get_seq_length(self, layer_index: int = 0) -> int:
        keys = self.layers[layer_index].keys
        return 0 if keys is None else int(keys.shape[-2])


class _ToyAttention(nn.Module):
    def __init__(self, config: _ToyConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_index
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **_kwargs,
    ):
        batch, sequence, _hidden = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).view(
                batch, sequence, self.num_heads, self.head_dim
            )
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch, sequence, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        value = (
            self.v_proj(hidden_states)
            .view(batch, sequence, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        cos, sin = position_embeddings
        query = query * cos.unsqueeze(1)
        key = key * cos.unsqueeze(1)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        key_heads = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value_heads = value.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = torch.matmul(query, key_heads.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, value_heads).transpose(1, 2).contiguous()
        context = context.reshape(batch, sequence, -1)
        if bool(getattr(self.config, "context_drift", False)):
            context = context + 3.0
        return self.o_proj(context), probabilities


class _ToyLayer(nn.Module):
    def __init__(self, config: _ToyConfig, layer_index: int) -> None:
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = _ToyAttention(config, layer_index)
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask,
        position_embeddings,
        past_key_values,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        attention, _probabilities = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + attention
        if bool(getattr(self.self_attn.config, "residual_drift", False)):
            hidden_states = hidden_states + 5.0
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class _ToyBackbone(nn.Module):
    def __init__(
        self,
        config: _ToyConfig,
        *,
        drift: bool,
        value_drift: bool,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, config.hidden_size)
        self.layers = nn.ModuleList([_ToyLayer(config, index) for index in range(2)])
        self.norm = nn.Identity()
        self.drift = drift
        self.value_drift = value_drift

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        del attention_mask
        hidden = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = _ToyCache(
                len(self.layers),
                drift=self.drift,
                value_drift=self.value_drift,
            )
        past_length = (
            0 if past_key_values is None else past_key_values.get_seq_length(0)
        )
        query_length = int(input_ids.shape[1])
        key_length = past_length + query_length
        query_positions = torch.arange(query_length) + past_length
        key_positions = torch.arange(key_length)
        allowed = key_positions[None, :] <= query_positions[:, None]
        causal_mask = torch.where(
            allowed,
            torch.tensor(0.0),
            torch.tensor(-1e9),
        ).view(1, 1, query_length, key_length)
        cos = torch.ones(1, query_length, _ToyConfig.head_dim)
        sin = torch.zeros_like(cos)
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_embeddings=(cos, sin),
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
                **kwargs,
            )
        return self.norm(hidden), past_key_values if use_cache else None


@dataclass
class _ToyOutput:
    logits: torch.Tensor
    past_key_values: object | None


class _ToyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        drift: bool = False,
        value_drift: bool = False,
        context_drift: bool = False,
        residual_drift: bool = False,
        cached_logit_drift: bool = False,
    ) -> None:
        super().__init__()
        torch.manual_seed(8181)
        self.config = _ToyConfig()
        self.config.context_drift = context_drift
        self.config.residual_drift = residual_drift
        self.config.cached_logit_drift = cached_logit_drift
        self.model = _ToyBackbone(
            self.config,
            drift=drift,
            value_drift=value_drift,
        )
        self.lm_head = nn.Linear(self.config.hidden_size, 32, bias=False)
        self.is_gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        hidden, cache = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.lm_head(hidden)
        if bool(self.config.cached_logit_drift):
            logits = torch.zeros_like(logits)
            logits[..., 1 if use_cache else 0] = 100.0
        return _ToyOutput(logits, cache)


def _hook_counts(model: nn.Module) -> tuple[int, int]:
    return (
        sum(len(module._forward_hooks) for module in model.modules()),
        sum(len(module._forward_pre_hooks) for module in model.modules()),
    )


def test_forced_cached_lane_captures_real_transition_and_restores_state() -> None:
    model = _ToyCausalLM().train()
    model.model.layers[1].eval()
    modes = tuple(module.training for module in model.modules())
    config_values = (
        model.config.use_cache,
        model.config.output_attentions,
        model.config.output_hidden_states,
    )
    hooks = _hook_counts(model)

    trace = collect_hf_cached_attention_trace(
        model,
        torch.tensor([[1, 4, 7]]),
        layer_index=1,
        group_indices=(0, 1),
        horizon=3,
        lane="forced",
        forced_token_ids=(8, 9, 10),
        sequence_id="forced-a",
    )

    assert trace.selected_token_ids == (8, 9, 10)
    assert trace.consumed_token_ids == (8, 9)
    assert trace.steps[0].prefix_length == 3
    assert trace.steps[-1].prefix_length == 5
    assert trace.steps[-1].layer_key_cache.shape == (1, 2, 5, 2)
    assert trace.steps[-1].attention_probabilities.shape == (1, 4, 1, 5)
    assert trace.maximum_cached_full_prefix_error < 1e-5
    assert trace.maximum_selected_context_error < 1e-5
    assert trace.maximum_value_projection_cache_error < 1e-5
    assert trace.steps[0].value_projection_input.shape == (1, 3, 8)
    assert trace.steps[1].value_projection_input.shape == (1, 1, 8)
    assert trace.steps[2].value_projection_input.shape == (1, 1, 8)
    assert trace.value_projection_weight_sha256
    assert trace.as_dict()["full_prefix_cache_equivalence_verified"] is True
    assert trace.as_dict()["value_projection_cache_identity_verified"] is True
    assert trace.cached_trace_validation_mode == "strict"
    assert trace.as_dict()["finite_validation"]["strict_numeric_enforcement"]
    assert tuple(module.training for module in model.modules()) == modes
    assert _hook_counts(model) == hooks
    assert (
        model.config.use_cache,
        model.config.output_attentions,
        model.config.output_hidden_states,
    ) == config_values


def test_greedy_lane_and_evidence_bridge_score_identical_traces_at_zero() -> None:
    model = _ToyCausalLM().eval()
    trace = collect_hf_cached_attention_trace(
        model,
        [1, 3],
        layer_index=0,
        group_indices=(1,),
        horizon=2,
        lane="greedy",
        sequence_id="greedy-a",
    )
    evidence = build_residual_attention_candidate_evidence(
        (trace,),
        (trace,),
        candidate_ids=("candidate-A",),
    )
    damage = score_residual_attention_candidate(evidence)

    assert trace.selected_token_ids[0] == int(
        torch.argmax(trace.steps[0].action_logits, dim=-1).item()
    )
    assert damage.risk_score == pytest.approx(0.0, abs=1e-12)
    assert damage.metrics["cache_transition_error"].worst == pytest.approx(0.0)
    assert damage.as_dict()["token_lane"] == "greedy-feedback-aligned-prefix"
    assert damage.as_dict()["autoregressive_closed_loop"] is True
    fixed = evidence.rollouts[0].reference.steps[0].fixed_attention_value_readout
    assert fixed.shape == trace.steps[0].full_attention_context.shape == (1, 8)
    assert trace.steps[0].attention_probabilities.shape[1] == 2
    assert trace.steps[0].full_attention_probabilities.shape[1] == 4


def test_evidence_bridge_rejects_mismatched_lane_prompt_and_consumed_path() -> None:
    model = _ToyCausalLM().eval()
    trace = collect_hf_cached_attention_trace(
        model,
        [1, 3],
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="greedy",
        sequence_id="alignment-a",
    )

    for changed in (
        replace(trace, lane="forced"),
        replace(trace, prompt_token_ids=(1, 4)),
        replace(trace, consumed_token_ids=((trace.consumed_token_ids[0] + 1) % 32,)),
    ):
        with pytest.raises(HFCachedAttentionTraceShapeError, match="lane, prompt"):
            build_residual_attention_candidate_evidence(
                (trace,),
                (changed,),
                candidate_ids=("candidate-A",),
            )

    forced = collect_hf_cached_attention_trace(
        model,
        [1, 3],
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="forced",
        forced_token_ids=(7, 8),
        sequence_id="alignment-forced",
    )
    changed_driver = replace(forced, selected_token_ids=(7, 9))
    with pytest.raises(HFCachedAttentionTraceShapeError, match="driving token"):
        build_residual_attention_candidate_evidence(
            (forced,),
            (changed_driver,),
            candidate_ids=("candidate-A",),
        )


def test_adversarial_cache_drift_fails_full_prefix_equivalence_closed() -> None:
    model = _ToyCausalLM(drift=True).eval()

    with pytest.raises(HFCachedAttentionTraceStateError, match="Cached logits differ"):
        collect_hf_cached_attention_trace(
            model,
            [1, 2, 3],
            layer_index=0,
            group_indices=(0,),
            horizon=3,
            lane="forced",
            forced_token_ids=(4, 5, 6),
        )
    assert _hook_counts(model) == (0, 0)


@pytest.mark.parametrize(
    ("model_kwargs", "message"),
    (
        ({"value_drift": True}, "value cache"),
        ({"context_drift": True}, "real pre-O context"),
        ({"residual_drift": True}, "residual \\+ O delta"),
        ({"cached_logit_drift": True}, "top-1 action"),
    ),
)
def test_strict_mode_preserves_each_finite_reconstruction_failure(
    model_kwargs: dict[str, bool],
    message: str,
) -> None:
    model = _ToyCausalLM(**model_kwargs).eval()

    with pytest.raises(HFCachedAttentionTraceStateError, match=message):
        collect_hf_cached_attention_trace(
            model,
            [1, 2, 3],
            layer_index=0,
            group_indices=(0,),
            horizon=2,
            lane="forced",
            forced_token_ids=(4, 5),
        )
    assert _hook_counts(model) == (0, 0)


def test_record_mode_keeps_finite_mismatches_as_explicit_diagnostics() -> None:
    model = _ToyCausalLM(
        value_drift=True,
        context_drift=True,
        residual_drift=True,
        cached_logit_drift=True,
    ).eval()
    limits = HFCachedAttentionTraceLimits(cached_trace_validation_mode="record")

    trace = collect_hf_cached_attention_trace(
        model,
        [1, 2, 3],
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="forced",
        forced_token_ids=(4, 5),
        limits=limits,
    )

    payload = trace.as_dict()
    checks = payload["finite_validation"]["checks"]
    assert trace.cached_trace_validation_mode == "record"
    assert payload["cached_trace_validation_mode"] == "record"
    assert payload["finite_validation"]["strict_numeric_enforcement"] is False
    assert payload["finite_validation"]["finite_mismatches_are_diagnostic_only"]
    assert payload["finite_validation"]["top1_agreement_count"] == 0
    assert payload["finite_validation"]["top1_step_count"] == 2
    assert payload["finite_validation"]["top1_agreement_rate"] == 0.0
    assert not any(
        checks[name]
        for name in (
            "cached_full_prefix_numerically_close",
            "cached_full_prefix_top1_agreement",
            "value_projection_cache_identity_valid",
            "full_context_reconstruction_valid",
            "selected_context_reconstruction_valid",
            "current_value_cache_append_valid",
            "residual_add_valid",
        )
    )
    assert checks["current_key_cache_append_valid"] is True
    assert payload["maximum_cached_full_prefix_error"] > 0.0
    assert payload["maximum_value_projection_cache_error"] > 0.0
    assert payload["maximum_full_context_error"] > 0.0
    assert payload["maximum_current_value_cache_append_error"] > 0.0
    assert payload["maximum_residual_add_error"] > 0.0
    assert payload["full_prefix_cache_equivalence_verified"] is False
    assert payload["value_projection_cache_identity_verified"] is False
    assert payload["attention_context_reconstruction_verified"] is False
    assert payload["current_kv_cache_append_verified"] is False
    assert payload["residual_add_verified"] is False
    assert all(
        step.cached_full_prefix_top1_agreement is False for step in trace.steps
    )
    assert _hook_counts(model) == (0, 0)


def test_record_mode_still_rejects_nonfinite_and_malformed_inputs() -> None:
    limits = HFCachedAttentionTraceLimits(cached_trace_validation_mode="record")
    nonfinite = _ToyCausalLM().eval()
    with torch.no_grad():
        nonfinite.model.layers[0].self_attn.o_proj.weight[0, 0] = float("nan")

    with pytest.raises(HFCachedAttentionTraceShapeError, match="non-finite"):
        collect_hf_cached_attention_trace(
            nonfinite,
            [1, 2],
            layer_index=0,
            group_indices=(0,),
            horizon=1,
            limits=limits,
        )
    with pytest.raises(HFCachedAttentionTraceShapeError, match="unique in-range"):
        collect_hf_cached_attention_trace(
            _ToyCausalLM().eval(),
            [1, 2],
            layer_index=0,
            group_indices=(0, 0),
            horizon=1,
            limits=limits,
        )
    with pytest.raises(HFCachedAttentionTraceShapeError, match="strict or record"):
        HFCachedAttentionTraceLimits(  # type: ignore[arg-type]
            cached_trace_validation_mode="permissive"
        )


def test_capture_memory_and_shape_limits_fail_closed_without_hook_leak() -> None:
    model = _ToyCausalLM().eval()

    with pytest.raises(HFCachedAttentionTraceLimitError, match="capture"):
        collect_hf_cached_attention_trace(
            model,
            [1, 2],
            layer_index=0,
            group_indices=(0,),
            horizon=2,
            limits=HFCachedAttentionTraceLimits(max_capture_bytes=1),
        )
    with pytest.raises(HFCachedAttentionTraceShapeError, match="unique in-range"):
        collect_hf_cached_attention_trace(
            model,
            [1, 2],
            layer_index=0,
            group_indices=(0, 0),
            horizon=1,
        )
    assert _hook_counts(model) == (0, 0)


def test_tiny_transformers_qwen3_sdpa_trace_reconstructs_attention() -> None:
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Installed Transformers has no Qwen3 implementation.")
    config = transformers.Qwen3Config(
        vocab_size=24,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    original_backend = model.config._attn_implementation
    original_hooks = _hook_counts(model)
    trace = collect_hf_cached_attention_trace(
        model,
        [1, 4, 6],
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="forced",
        forced_token_ids=(7, 8),
    )

    assert trace.horizon == 2
    assert trace.maximum_cached_full_prefix_error < 1e-5
    assert trace.maximum_selected_context_error < 1e-5
    assert trace.maximum_value_projection_cache_error < 1e-5
    assert trace.steps[0].value_projection_input.shape == (1, 3, 8)
    assert trace.steps[1].value_projection_input.shape == (1, 1, 8)
    assert trace.attention_backend == original_backend == "sdpa"
    assert model.config._attn_implementation == original_backend
    assert trace.cache_class.endswith(".DynamicCache")
    assert trace.torch_version == torch.__version__
    assert trace.model_class.endswith(".Qwen3ForCausalLM")
    assert _hook_counts(model) == original_hooks
    for step in trace.steps:
        assert step.layer_key_cache.shape[-2] == step.prefix_length
        assert step.layer_value_cache.shape[-2] == step.prefix_length
        assert torch.allclose(
            step.key,
            step.selected_key_cache[..., -1:, :],
            rtol=8e-3,
            atol=8e-3,
        )
        assert torch.allclose(
            step.value,
            step.selected_value_cache[..., -1:, :],
            rtol=8e-3,
            atol=8e-3,
        )


def test_tiny_qwen3_codebook_projection_views_support_cached_trace() -> None:
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Installed Transformers has no Qwen3 implementation.")
    from neural_forge.hf_qwen_codebook_model import prepare_qwen3_codebook_model

    torch.manual_seed(1849)
    config = transformers.Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    tokenizer = _Tokenizer(16)
    runtime = prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        attention_prototypes_per_group=1,
        neuron_prototypes_per_neuron=1,
    ).eval()
    projection = runtime.model.layers[0].self_attn.q_proj
    assert type(projection).__name__ == "AttentionCodebookProjectionView"
    backend = runtime.config._attn_implementation

    trace = collect_hf_cached_attention_trace(
        runtime,
        [1, 5, 7],
        layer_index=0,
        group_indices=(0,),
        horizon=2,
        lane="forced",
        forced_token_ids=(8, 9),
        sequence_id="codebook-a",
    )

    assert trace.horizon == 2
    assert trace.cache_class.endswith(".DynamicCache")
    assert trace.attention_backend == runtime.config._attn_implementation == backend
    assert trace.maximum_cached_full_prefix_error < 1e-5
    assert trace.maximum_selected_context_error < 1e-5
    assert trace.maximum_value_projection_cache_error < 1e-5
    assert trace.value_projection_weight_sha256
    assert trace.steps[0].value_projection_input.shape == (1, 3, 8)
    assert trace.steps[1].value_projection_input.shape == (1, 1, 8)
    assert _hook_counts(runtime) == (0, 0)
