from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_functional_decode_trace import (  # noqa: E402
    FunctionalDecodeLimits,
    FunctionalDecodeTraceError,
    FunctionalDecodeTraceLimitError,
    collect_functional_decode_trace,
    trace_mapping,
)
from neural_forge.hf_residual_attention_functional import (  # noqa: E402
    score_residual_attention_function,
)


class _TinyCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class _TinyCachedLM(torch.nn.Module):
    def __init__(self, *, vocab_size: int = 11, offset: int = 1) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 4)
        self.vocab_size = vocab_size
        self.offset = offset
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
        )

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **_kwargs,
    ):
        del attention_mask, return_dict
        rows = []
        for token in input_ids.reshape(-1).tolist():
            target = (int(token) + self.offset) % self.vocab_size
            row = torch.full(
                (self.vocab_size,),
                -4.0,
                dtype=self.embedding.weight.dtype,
                device=input_ids.device,
            )
            row[target] = 4.0
            rows.append(row)
        logits = torch.stack(rows).view(1, int(input_ids.shape[1]), self.vocab_size)
        seen = 0 if past_key_values is None else int(past_key_values.length)
        cache = _TinyCache(seen + int(input_ids.shape[1])) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


class _NonFiniteLM(_TinyCachedLM):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        result.logits[..., 0] = float("nan")
        return result


class _StaleCacheLM(_TinyCachedLM):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        result.past_key_values = _TinyCache(1)
        return result


class _ExplodingStateMutationLM(_TinyCachedLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 2:
            self.config.use_cache = "tampered"
            self.config.output_attentions = "tampered"
            self.train(True)
            self.embedding.weight.requires_grad_(False)
            self.embedding.weight.grad = torch.zeros_like(self.embedding.weight)
            raise RuntimeError("synthetic cached decode failure")
        return super().forward(*args, **kwargs)


def _limits(**changes) -> FunctionalDecodeLimits:
    values = {
        "max_horizon": 8,
        "max_prompt_tokens": 16,
        "max_vocabulary": 32,
        "max_logit_bytes": 100_000,
        "max_model_devices": 1,
    }
    values.update(changes)
    return FunctionalDecodeLimits(**values)


def test_greedy_trace_records_only_real_output_function_and_restores_state() -> None:
    model = _TinyCachedLM().train()
    model.config.use_cache = False
    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)
    gradient = model.embedding.weight.grad
    gradient_version = gradient._version

    trace = collect_functional_decode_trace(
        model,
        torch.tensor([[1, 2, 3]]),
        horizon=4,
        lane="greedy",
        sequence_id="greedy-0",
        limits=_limits(),
    )

    assert trace.selected_token_ids == (4, 5, 6, 7)
    assert trace.consumed_token_ids == (4, 5, 6)
    assert [step.prefix_length for step in trace.steps] == [3, 4, 5, 6]
    assert [step.cache_length for step in trace.steps] == [3, 4, 5, 6]
    assert all(step.selected_token_id == step.greedy_token_id for step in trace.steps)
    assert model.training is True
    assert model.config.use_cache is False
    assert model.config.output_attentions is True
    assert model.config.output_hidden_states is True
    assert model.embedding.weight.grad is gradient
    assert gradient._version == gradient_version
    payload = trace.as_dict()
    assert payload["output_function_only"] is True
    assert "weights" in payload["excluded_similarity_inputs"]


def test_forced_trace_uses_driver_but_retains_model_greedy_action() -> None:
    trace = collect_functional_decode_trace(
        _TinyCachedLM(),
        [1, 2],
        horizon=3,
        lane="forced",
        forced_token_ids=(9, 8, 7),
        limits=_limits(),
    )
    assert trace.selected_token_ids == (9, 8, 7)
    assert trace.consumed_token_ids == (9, 8)
    assert [step.greedy_token_id for step in trace.steps] == [3, 10, 9]
    assert all(step.selected_token_id != step.greedy_token_id for step in trace.steps)


def test_trace_is_directly_compatible_with_behavior_first_scorer() -> None:
    parent = _TinyCachedLM(offset=1)
    candidate = _TinyCachedLM(offset=2)
    prompt = [1, 2]
    parent_greedy = collect_functional_decode_trace(
        parent, prompt, horizon=3, lane="greedy", sequence_id="p0", limits=_limits()
    )
    drivers = parent_greedy.selected_token_ids
    parent_forced = collect_functional_decode_trace(
        parent,
        prompt,
        horizon=3,
        lane="forced",
        forced_token_ids=drivers,
        sequence_id="p0",
        limits=_limits(),
    )
    candidate_greedy = collect_functional_decode_trace(
        candidate,
        prompt,
        horizon=3,
        lane="greedy",
        sequence_id="p0",
        limits=_limits(),
    )
    candidate_forced = collect_functional_decode_trace(
        candidate,
        prompt,
        horizon=3,
        lane="forced",
        forced_token_ids=drivers,
        sequence_id="p0",
        limits=_limits(),
    )
    report = score_residual_attention_function(
        trace_mapping((parent_forced, parent_greedy)),
        trace_mapping((candidate_forced, candidate_greedy)),
        candidate_id="offset-2",
        split="calibration-selection",
    )
    assert report.forced.parent_driver_top1_rate == 1.0
    assert report.forced.candidate_parent_top1_agreement_rate == 0.0
    assert report.greedy.earliest_divergence_step == 0


def test_nonfinite_output_and_bad_forced_ids_fail() -> None:
    with pytest.raises(FunctionalDecodeTraceError, match="finite"):
        collect_functional_decode_trace(
            _NonFiniteLM(), [1], horizon=2, lane="greedy", limits=_limits()
        )
    with pytest.raises(FunctionalDecodeTraceError, match="out-of-vocabulary"):
        collect_functional_decode_trace(
            _TinyCachedLM(),
            [1],
            horizon=2,
            lane="forced",
            forced_token_ids=(2, 99),
            limits=_limits(),
        )


def test_resource_bounds_and_trace_mapping_are_fail_closed() -> None:
    with pytest.raises(FunctionalDecodeTraceLimitError, match="max_horizon"):
        collect_functional_decode_trace(
            _TinyCachedLM(),
            [1],
            horizon=3,
            lane="greedy",
            limits=_limits(max_horizon=2),
        )
    with pytest.raises(FunctionalDecodeTraceLimitError, match="max_logit_bytes"):
        collect_functional_decode_trace(
            _TinyCachedLM(),
            [1],
            horizon=2,
            lane="greedy",
            limits=_limits(max_logit_bytes=8),
        )
    with pytest.raises(FunctionalDecodeTraceLimitError, match="vocabulary"):
        collect_functional_decode_trace(
            _TinyCachedLM(),
            [1],
            horizon=1,
            lane="greedy",
            limits=_limits(max_vocabulary=10),
        )
    multi_device = _TinyCachedLM()
    multi_device.register_buffer("meta_marker", torch.empty(1, device="meta"))
    with pytest.raises(FunctionalDecodeTraceLimitError, match="device count"):
        collect_functional_decode_trace(
            multi_device, [1], horizon=1, lane="greedy", limits=_limits()
        )
    trace = collect_functional_decode_trace(
        _TinyCachedLM(), [1], horizon=2, lane="greedy", limits=_limits()
    )
    with pytest.raises(FunctionalDecodeTraceError, match="duplicated"):
        trace_mapping((trace, trace))


def test_invalid_input_and_lane_contracts_fail_before_model_mutation() -> None:
    model = _TinyCachedLM().train()
    with pytest.raises(FunctionalDecodeTraceError, match="input_ids"):
        collect_functional_decode_trace(
            model, torch.tensor([[1.0]]), horizon=1, lane="greedy", limits=_limits()
        )
    with pytest.raises(FunctionalDecodeTraceError, match="required"):
        collect_functional_decode_trace(
            model, [1], horizon=1, lane="forced", limits=_limits()
        )
    with pytest.raises(FunctionalDecodeTraceError, match="only"):
        collect_functional_decode_trace(
            model,
            [1],
            horizon=1,
            lane="greedy",
            forced_token_ids=(2,),
            limits=_limits(),
        )
    assert model.training is True


def test_cache_growth_and_declared_vocabulary_are_fail_closed() -> None:
    with pytest.raises(FunctionalDecodeTraceError, match="Cache sequence length"):
        collect_functional_decode_trace(
            _StaleCacheLM(), [1, 2], horizon=2, lane="greedy", limits=_limits()
        )

    mismatched = _TinyCachedLM()
    mismatched.config.vocab_size += 1
    with pytest.raises(FunctionalDecodeTraceError, match="disagrees"):
        collect_functional_decode_trace(
            mismatched, [1], horizon=1, lane="greedy", limits=_limits()
        )
    with pytest.raises(FunctionalDecodeTraceError, match="embedding vocabulary"):
        collect_functional_decode_trace(
            _TinyCachedLM(), [11], horizon=1, lane="greedy", limits=_limits()
        )


def test_forward_exception_restores_config_modes_requires_grad_and_gradient() -> None:
    model = _ExplodingStateMutationLM().train()
    model.embedding.eval()
    model.config.use_cache = False
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)
    gradient = model.embedding.weight.grad
    gradient_version = gradient._version
    modes = tuple(module.training for module in model.modules())

    with pytest.raises(RuntimeError, match="synthetic cached decode failure"):
        collect_functional_decode_trace(
            model, [1, 2], horizon=3, lane="greedy", limits=_limits()
        )

    assert tuple(module.training for module in model.modules()) == modes
    assert model.config.use_cache is False
    assert model.config.output_attentions is True
    assert model.config.output_hidden_states is True
    assert model.embedding.weight.requires_grad is True
    assert model.embedding.weight.grad is gradient
    assert gradient._version == gradient_version


def test_tiny_transformers_qwen_dynamic_cache_length_is_verified() -> None:
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Installed Transformers has no Qwen3 implementation.")
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    ).eval()

    trace = collect_functional_decode_trace(
        model,
        [1, 4, 6],
        horizon=2,
        lane="greedy",
        limits=_limits(),
    )

    assert [step.cache_length for step in trace.steps] == [3, 4]
    assert trace.cache_class.endswith(".DynamicCache")
