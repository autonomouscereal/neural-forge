from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
)
from neural_forge.hf_qwen_sensitivity import (  # noqa: E402
    FINITE_DIFFERENCE_METHOD,
    Qwen3AttentionGroupRef,
    Qwen3NeuronRef,
    Qwen3SensitivityLimitError,
    Qwen3SensitivityLimits,
    Qwen3SensitivitySplitError,
    collect_qwen3_sensitivity_evidence,
)


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, size: int = 24) -> None:
        self.tokens = tuple(f"token-{index}" for index in range(size))

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def _tiny_qwen(seed: int = 1771) -> torch.nn.Module:
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=24,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        max_position_embeddings=32,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    return transformers.Qwen3ForCausalLM(config).eval()


def _calibration_a() -> torch.Tensor:
    return torch.tensor([[1, 3, 4, 5], [2, 6, 7, 8]], dtype=torch.long)


def _calibration_b() -> torch.Tensor:
    return torch.tensor([[1, 13, 14, 15], [2, 16, 17, 18]], dtype=torch.long)


def _gate_a() -> torch.Tensor:
    return torch.tensor([[9, 10, 11, 12]], dtype=torch.long)


def _gate_b() -> torch.Tensor:
    return torch.tensor([[19, 20, 21, 22]], dtype=torch.long)


def _logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return (
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                position_ids=torch.arange(input_ids.shape[1]).repeat(
                    input_ids.shape[0], 1
                ),
                use_cache=False,
                logits_to_keep=torch.tensor([2, 3]),
            )
            .logits.detach()
            .clone()
        )


def _hook_inventory(model: torch.nn.Module) -> tuple[tuple[int, ...], ...]:
    result = []
    for layer in model.model.layers:
        modules = [layer, layer.mlp]
        down = getattr(layer.mlp, "down_proj", None)
        if down is not None:
            modules.append(down)
        modules.extend(
            getattr(layer.self_attn, f"{role}_proj") for role in ("q", "k", "v", "o")
        )
        for module in modules:
            result.append(
                (
                    id(module),
                    *tuple(module._forward_pre_hooks.keys()),
                    -1,
                    *tuple(module._forward_hooks.keys()),
                )
            )
    return tuple(result)


def _collect(
    model: torch.nn.Module,
    calibration: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    **kwargs: object,
):
    return collect_qwen3_sensitivity_evidence(
        model,
        _calibration_a() if calibration is None else calibration,
        _gate_a() if gate is None else gate,
        calibration_attention_mask=torch.ones_like(
            _calibration_a() if calibration is None else calibration
        ),
        gate_attention_mask=torch.ones_like(_gate_a() if gate is None else gate),
        score_positions=(2, 3),
        neuron_candidates=((0, 0), (1, 2)),
        attention_group_candidates=((0, 0), (1, 1)),
        sketch_dim=4,
        **kwargs,
    )


def test_real_post_swiglu_energy_and_evidence_change_with_inputs() -> None:
    model = _tiny_qwen()
    calibration = _calibration_a()
    captured: list[torch.Tensor] = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(capture)
    _logits(model, calibration)
    handle.remove()
    expected = captured[0].float().square().mean(dim=(0, 1))

    first = _collect(model, calibration, _gate_a())
    torch.testing.assert_close(
        torch.tensor(first.neuron_activation_energy[0]),
        expected,
        rtol=1e-6,
        atol=1e-9,
    )
    assert first.neurons[0].derivative_method == FINITE_DIFFERENCE_METHOD
    assert any(
        abs(value) > 0.0 for value in first.neurons[0].finite_difference_controllability
    )
    assert any(
        abs(value) > 0.0 for value in first.neurons[0].finite_difference_observability
    )

    second = _collect(model, _calibration_b(), _gate_b())
    assert second.fingerprints.calibration != first.fingerprints.calibration
    assert second.fingerprints.cache != first.fingerprints.cache
    assert second.neuron_activation_energy != first.neuron_activation_energy
    assert (
        second.neurons[0].finite_difference_controllability
        != first.neurons[0].finite_difference_controllability
    )


def test_neuron_and_atomic_gqa_ablations_restore_model_exactly() -> None:
    model = _tiny_qwen(1772)
    baseline_logits = _logits(model, _calibration_a())
    state_before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    hooks_before = _hook_inventory(model)

    evidence = _collect(model)

    torch.testing.assert_close(
        _logits(model, _calibration_a()),
        baseline_logits,
        rtol=0.0,
        atol=0.0,
    )
    assert _hook_inventory(model) == hooks_before
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0.0, atol=0.0)
    assert all(
        item.atomic_projection_roles == ("q", "k", "v", "o")
        and item.role_ablation_call_counts == (1, 1, 1, 1)
        for item in evidence.attention_groups
    )
    assert evidence.report.hooks_restored is True
    assert evidence.report.model_weights_mutated is False
    assert evidence.report.backward_calls == 0
    assert evidence.report.optimizer_steps == 0
    assert evidence.report.training_steps == 0
    assert evidence.report.recovery_steps == 0


def test_hook_cleanup_is_atomic_when_candidate_forward_raises() -> None:
    model = _tiny_qwen(1773)
    calibration = _calibration_a()
    baseline_logits = _logits(model, calibration)

    def fail_on_expanded_batch(_module, _inputs, output):
        if output.shape[0] > calibration.shape[0]:
            raise RuntimeError("injected candidate failure")

    external = model.model.layers[0].mlp.down_proj.register_forward_hook(
        fail_on_expanded_batch
    )
    inventory_with_external_hook = _hook_inventory(model)
    with pytest.raises(RuntimeError, match="injected candidate failure"):
        _collect(model)
    assert _hook_inventory(model) == inventory_with_external_hook
    external.remove()

    torch.testing.assert_close(
        _logits(model, calibration),
        baseline_logits,
        rtol=0.0,
        atol=0.0,
    )


def test_high_damage_and_user_exact_anchor_masks_are_immutable() -> None:
    model = _tiny_qwen(1774)
    evidence = _collect(
        model,
        explicit_neuron_anchors=((1, 5),),
        explicit_attention_group_anchors=((1, 0),),
        neuron_high_damage_anchor_count=1,
        attention_high_damage_anchor_count=1,
    )
    highest_neuron = max(evidence.neurons, key=lambda item: item.damage_score).node
    highest_attention = max(
        evidence.attention_groups, key=lambda item: item.damage_score
    ).node

    assert evidence.anchors.is_neuron_exact(1, 5)
    assert evidence.anchors.is_neuron_exact(
        highest_neuron.layer_index, highest_neuron.neuron_index
    )
    assert evidence.anchors.is_attention_group_exact(1, 0)
    assert evidence.anchors.is_attention_group_exact(
        highest_attention.layer_index, highest_attention.group_index
    )
    assert Qwen3NeuronRef(1, 5) in evidence.anchors.explicit_neurons
    assert Qwen3AttentionGroupRef(1, 0) in evidence.anchors.explicit_attention_groups
    with pytest.raises(TypeError):
        evidence.anchors.neuron_mask[0][0] = False
    with pytest.raises(FrozenInstanceError):
        evidence.anchors.neuron_mask = ()

    features = evidence.to_adaptive_surgery_features()
    anchored = {
        (feature.layer_index, feature.neuron_index)
        for feature in features
        if feature.anchor is not None and feature.anchor.mode == "exact"
    }
    assert (highest_neuron.layer_index, highest_neuron.neuron_index) in anchored


def test_fingerprints_include_positions_and_splits_and_caps_are_hard() -> None:
    model = _tiny_qwen(1775)
    calibration = _calibration_a()
    gate = _gate_a()
    with pytest.raises(Qwen3SensitivitySplitError, match="must be disjoint"):
        collect_qwen3_sensitivity_evidence(
            model,
            calibration,
            calibration[:1],
            neuron_candidates=((0, 0),),
            attention_group_candidates=((0, 0),),
        )
    with pytest.raises(Qwen3SensitivityLimitError, match="hard cap"):
        collect_qwen3_sensitivity_evidence(
            model,
            calibration,
            gate,
            neuron_candidates=((0, 0),),
            attention_group_candidates=((0, 0),),
            limits=Qwen3SensitivityLimits(max_calibration_rows=1),
        )

    standard = collect_qwen3_sensitivity_evidence(
        model,
        calibration,
        gate,
        calibration_position_ids=torch.arange(4).repeat(2, 1),
        gate_position_ids=torch.arange(4).repeat(1, 1),
        neuron_candidates=((0, 0),),
        attention_group_candidates=((0, 0),),
    )
    shifted = collect_qwen3_sensitivity_evidence(
        model,
        calibration,
        gate,
        calibration_position_ids=torch.arange(4, 8).repeat(2, 1),
        gate_position_ids=torch.arange(8, 12).repeat(1, 1),
        neuron_candidates=((0, 0),),
        attention_group_candidates=((0, 0),),
    )
    assert standard.fingerprints.model == shifted.fingerprints.model
    assert standard.fingerprints.calibration != shifted.fingerprints.calibration
    assert standard.fingerprints.gate != shifted.fingerprints.gate
    assert standard.fingerprints.cache != shifted.fingerprints.cache
    assert standard.report.tracked_peak_collector_bytes <= (
        standard.report.collector_byte_limit
    )


def test_prepared_codebook_runtime_uses_exact_channel_contribution() -> None:
    model = _tiny_qwen(1776)
    tokenizer = _Tokenizer()
    prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        neuron_prototypes_per_neuron=2,
        attention_prototypes_per_group=2,
    )
    baseline_logits = _logits(model, _calibration_a())
    evidence = _collect(model)

    assert evidence.report.mlp_capture_methods == (
        "exact_codebook_recompute_from_real_mlp_input",
        "exact_codebook_recompute_from_real_mlp_input",
    )
    assert all(item.activation_energy >= 0.0 for item in evidence.neurons)
    assert any(item.damage_score > 0.0 for item in evidence.neurons)
    torch.testing.assert_close(
        _logits(model, _calibration_a()),
        baseline_logits,
        rtol=0.0,
        atol=0.0,
    )
