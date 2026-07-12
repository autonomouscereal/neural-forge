from __future__ import annotations

import ctypes
import threading
from dataclasses import FrozenInstanceError, replace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3Config"):
    pytest.skip("Transformers build has no Qwen3 support", allow_module_level=True)

from neural_forge.hf_qwen_codebook_model import (  # noqa: E402
    prepare_qwen3_codebook_model,
)
import neural_forge.hf_qwen_sensitivity as sensitivity  # noqa: E402
from neural_forge.hf_qwen_sensitivity import (  # noqa: E402
    DIRECTIONAL_DIFFERENCE_METHOD,
    Qwen3AttentionGroupRef,
    Qwen3NeuronRef,
    Qwen3SensitivityLimitError,
    Qwen3SensitivityLimits,
    Qwen3SensitivitySplitError,
    Qwen3SensitivityStateError,
    collect_qwen3_sensitivity_evidence,
    fingerprint_qwen3_model_manifest,
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


def _set_first_storage_byte(value: torch.Tensor, byte: int) -> None:
    address = int(value.untyped_storage().data_ptr()) + (
        int(value.storage_offset()) * int(value.element_size())
    )
    ctypes.c_ubyte.from_address(address).value = byte


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
        modules.extend((layer.self_attn.q_norm, layer.self_attn.k_norm))
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


def test_model_manifest_supports_registered_inference_tensors() -> None:
    model = _tiny_qwen(seed=1770)
    with torch.inference_mode():
        inference_buffer = torch.arange(4, dtype=torch.float32)
    model.register_buffer("inference_only_state", inference_buffer)

    manifest = fingerprint_qwen3_model_manifest(model)
    record = next(
        item for item in manifest.tensors if item.name == "inference_only_state"
    )

    assert record.version == -1
    assert record.storage_nbytes == inference_buffer.numel() * inference_buffer.element_size()
    assert record.data_ptr == inference_buffer.untyped_storage().data_ptr()


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
    assert first.neurons[0].evidence_method == DIRECTIONAL_DIFFERENCE_METHOD
    assert any(
        abs(value) > 0.0
        for value in first.neurons[0].residual_ablation_difference_sketch
    )
    assert any(
        abs(value) > 0.0 for value in first.neurons[0].logit_ablation_difference_sketch
    )

    second = _collect(model, _calibration_b(), _gate_b())
    assert second.fingerprints.calibration != first.fingerprints.calibration
    assert second.fingerprints.cache != first.fingerprints.cache
    assert second.neuron_activation_energy != first.neuron_activation_energy
    assert (
        second.neurons[0].residual_ablation_difference_sketch
        != first.neurons[0].residual_ablation_difference_sketch
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


def test_hook_callable_replacement_with_same_handle_is_detected() -> None:
    model = _tiny_qwen(1794)
    layer = model.model.layers[0]
    handle = None

    def replace_self(_module, _inputs, output):
        assert handle is not None
        layer._forward_hooks[handle.id] = lambda *_args: None
        return output

    handle = layer.register_forward_hook(replace_self)
    try:
        with pytest.raises(Qwen3SensitivityStateError, match="hook inventory"):
            _collect(model)
    finally:
        handle.remove()


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
    assert standard.report.process_rss_start_bytes > 0
    assert (
        standard.report.process_rss_peak_bytes
        >= standard.report.process_rss_start_bytes
    )
    assert (
        standard.report.process_rss_peak_increase_bytes
        <= standard.report.process_rss_increase_limit
    )
    assert standard.report.resource_checkpoints > 2
    assert standard.report.attention_backend
    assert standard.report.attention_workspace_bound_bytes > 0


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
        "streamed_direct_codebook_prototype_channels",
        "streamed_direct_codebook_prototype_channels",
    )
    assert all(item.activation_energy >= 0.0 for item in evidence.neurons)
    assert any(item.damage_score > 0.0 for item in evidence.neurons)
    torch.testing.assert_close(
        _logits(model, _calibration_a()),
        baseline_logits,
        rtol=0.0,
        atol=0.0,
    )


def test_prepared_codebook_collector_adds_no_dense_layer_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_qwen(1777)
    tokenizer = _Tokenizer()
    prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        neuron_prototypes_per_neuron=2,
        attention_prototypes_per_group=2,
    )
    registry = model.model.layers[0].mlp.registry
    registry_type = type(registry)
    original = registry_type.materialize_layer
    calls: list[int] = []

    def counted(self, layer_index):
        calls.append(int(layer_index))
        return original(self, layer_index)

    monkeypatch.setattr(registry_type, "materialize_layer", counted)
    evidence = _collect(model)
    expected_runtime_calls = evidence.report.logical_layers * (
        evidence.report.baseline_forward_passes
        + evidence.report.candidate_forward_passes
    )
    assert len(calls) == expected_runtime_calls
    assert calls.count(0) == calls.count(1) == 3


def test_qk_energy_is_captured_after_qwen_normalization() -> None:
    model = _tiny_qwen(1778)
    calibration = _calibration_a()
    captured: dict[str, torch.Tensor] = {}
    layer = model.model.layers[0]

    def capture(role):
        def hook(_module, _inputs, output):
            captured[role] = output.detach().clone()

        return hook

    q_handle = layer.self_attn.q_norm.register_forward_hook(capture("q"))
    k_handle = layer.self_attn.k_norm.register_forward_hook(capture("k"))
    _logits(model, calibration)
    q_handle.remove()
    k_handle.remove()

    def expected(values: torch.Tensor) -> torch.Tensor:
        groups = model.config.num_key_value_heads
        shaped = values.float().reshape(2, 4, groups, -1)
        return shaped.square().mean(dim=(0, 1, 3))

    evidence = _collect(model, calibration, _gate_a())
    measured_q = torch.tensor(
        [group[0] for group in evidence.attention_role_activation_energy[0]]
    )
    measured_k = torch.tensor(
        [group[1] for group in evidence.attention_role_activation_energy[0]]
    )
    torch.testing.assert_close(
        measured_q, expected(captured["q"]), rtol=1e-6, atol=1e-9
    )
    torch.testing.assert_close(
        measured_k, expected(captured["k"]), rtol=1e-6, atol=1e-9
    )


def test_directional_differences_are_not_exported_as_jvp_or_dual() -> None:
    evidence = _collect(_tiny_qwen(1779))
    item = evidence.neurons[0]
    assert item.evidence_method == DIRECTIONAL_DIFFERENCE_METHOD
    assert not hasattr(item, "finite_difference_controllability")
    assert not hasattr(item, "finite_difference_observability")
    assert not hasattr(item, "residual_directional_difference")
    assert not hasattr(item, "logit_directional_difference")
    assert not {
        "jvp",
        "vjp",
        "jacobian",
        "dual",
    } & set(item.evidence_method.split("_"))

    feature = evidence.to_adaptive_surgery_features()[0]
    assert feature.jacobian_features == (0.0,)
    assert feature.dual_features == (0.0,)
    assert feature.domain_features is not None
    directional = (
        item.residual_ablation_difference_sketch + item.logit_ablation_difference_sketch
    )
    assert feature.domain_features[-len(directional) :] == directional


def test_incremental_refresh_matches_full_root_and_detects_undeclared_change() -> None:
    model = _tiny_qwen(1780)
    initial = _collect(model)
    parent = initial.fingerprints.model_manifest
    with torch.no_grad():
        model.model.layers[0].mlp.gate_proj.weight[0, 0].add_(0.125)

    refreshed = _collect(
        model,
        incremental_parent_manifest=parent,
        changed_state_prefixes=("model.layers.0.",),
        accepted_transaction_sha256="a" * 64,
    )
    fresh_full = fingerprint_qwen3_model_manifest(model)
    manifest = refreshed.fingerprints.model_manifest
    assert manifest.mode == "incremental"
    assert manifest.parent_root == parent.root
    assert manifest.root == fresh_full.root
    assert manifest.root != parent.root
    assert manifest.tensors_hashed == len(manifest.tensors)
    assert manifest.tensors_reused > 0
    assert refreshed.report.model_fingerprint_mode == "incremental"
    assert refreshed.report.model_tensors_hashed == manifest.tensors_hashed

    undeclared = model.model.layers[1].mlp.up_proj.weight
    version = undeclared._version
    undeclared.data[0, 0].add_(0.25)
    assert undeclared._version == version
    with pytest.raises(Qwen3SensitivityStateError, match="Undeclared state mutation"):
        fingerprint_qwen3_model_manifest(
            model,
            parent=manifest,
            changed_state_prefixes=("model.layers.0.",),
            accepted_transaction_sha256="b" * 64,
        )


def test_incremental_refresh_detects_direct_raw_storage_mutation() -> None:
    model = _tiny_qwen(1788)
    parent = fingerprint_qwen3_model_manifest(model)
    target = model.model.layers[1].mlp.up_proj.weight
    pointer = target.untyped_storage().data_ptr()
    version = target._version
    original_byte = ctypes.c_ubyte.from_address(pointer).value

    _set_first_storage_byte(target, (original_byte + 1) % 256)

    assert target.untyped_storage().data_ptr() == pointer
    assert target._version == version
    with pytest.raises(Qwen3SensitivityStateError, match="Undeclared state mutation"):
        fingerprint_qwen3_model_manifest(
            model,
            parent=parent,
            changed_state_prefixes=("model.layers.0.",),
            accepted_transaction_sha256="c" * 64,
        )


def test_incremental_collection_finally_rehashes_declared_changed_tensors() -> None:
    model = _tiny_qwen(1795)
    parent = fingerprint_qwen3_model_manifest(model)
    target = model.model.layers[0].mlp.gate_proj.weight
    with torch.no_grad():
        target[0, 0].add_(0.125)
    pointer = target.untyped_storage().data_ptr()
    version = target._version
    original_byte = ctypes.c_ubyte.from_address(pointer).value
    changed_byte = (original_byte + 1) % 256

    def mutate_declared_storage(_module, _inputs, output):
        _set_first_storage_byte(target, changed_byte)
        return output

    handle = model.model.layers[0].register_forward_hook(mutate_declared_storage)
    try:
        with pytest.raises(Qwen3SensitivityStateError, match="exact post-forward"):
            _collect(
                model,
                incremental_parent_manifest=parent,
                changed_state_prefixes=("model.layers.0.",),
                accepted_transaction_sha256="d" * 64,
            )
    finally:
        handle.remove()
    assert target.untyped_storage().data_ptr() == pointer
    assert target._version == version


def test_actual_rss_and_cuda_peak_guards_enforce_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = Qwen3SensitivityLimits(
        max_process_rss_increase_bytes=200,
        max_cuda_allocated_increase_bytes=200,
        max_cuda_reserved_increase_bytes=200,
    )
    rss = [1_000]
    peak = [1_000]
    monkeypatch.setattr(sensitivity, "_process_rss_bytes", lambda: rss[0])
    monkeypatch.setattr(sensitivity, "_process_peak_rss_bytes", lambda: peak[0])
    rss_guard = sensitivity._ResourceGuard(limits, ())
    assert rss_guard._sampler_thread.is_alive()
    rss[0] = 1_201
    peak[0] = 1_201
    with pytest.raises(Qwen3SensitivityLimitError, match="process RSS"):
        rss_guard.close()
    assert rss_guard.stopped
    assert not rss_guard._sampler_thread.is_alive()

    monkeypatch.setattr(sensitivity, "_process_rss_bytes", lambda: 1_000)
    monkeypatch.setattr(sensitivity, "_process_peak_rss_bytes", lambda: 1_000)
    monkeypatch.setattr(sensitivity, "_cuda_devices", lambda _devices: ("cuda:0",))
    monkeypatch.setattr(sensitivity, "_reset_cuda_peak_stats", lambda _devices: None)
    snapshots = iter(
        (
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (110, 110, 301, 110),
        )
    )
    monkeypatch.setattr(
        sensitivity,
        "_cuda_memory_snapshot",
        lambda _devices: next(snapshots),
    )
    cuda_guard = sensitivity._ResourceGuard(limits, ("cuda:0",))
    with pytest.raises(Qwen3SensitivityLimitError, match="CUDA allocated"):
        cuda_guard.close()
    assert cuda_guard.stopped
    assert not cuda_guard._sampler_thread.is_alive()


def test_process_peak_uses_new_absolute_os_high_water_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [1_000]
    lifetime_peak = [5_000]
    limits = Qwen3SensitivityLimits(max_process_rss_increase_bytes=2_000)
    monkeypatch.setattr(sensitivity, "_process_rss_bytes", lambda: current[0])
    monkeypatch.setattr(
        sensitivity,
        "_process_peak_rss_bytes",
        lambda: lifetime_peak[0],
    )
    guard = sensitivity._ResourceGuard(limits, ())
    lifetime_peak[0] = 6_000
    with pytest.raises(Qwen3SensitivityLimitError, match="process RSS increase 5000"):
        guard.checkpoint("new_os_high_water")
    with pytest.raises(Qwen3SensitivityLimitError, match="process RSS increase 5000"):
        guard.close()
    assert guard.process_rss_peak == 6_000
    assert guard.stopped


def test_continuous_rss_sampler_retains_transient_spike_between_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spike = threading.Event()
    sampled = threading.Event()

    def sampled_rss() -> int:
        if (
            threading.current_thread().name == "qwen3-sensitivity-rss-sampler"
            and spike.is_set()
        ):
            sampled.set()
            return 1_301
        return 1_000

    monkeypatch.setattr(sensitivity, "_process_rss_bytes", sampled_rss)
    monkeypatch.setattr(sensitivity, "_process_peak_rss_bytes", lambda: 1_000)
    guard = sensitivity._ResourceGuard(
        Qwen3SensitivityLimits(max_process_rss_increase_bytes=300),
        (),
    )
    spike.set()
    assert sampled.wait(timeout=1.0)
    spike.clear()
    with pytest.raises(Qwen3SensitivityLimitError, match="process RSS"):
        guard.close()
    assert guard.process_rss_peak == 1_301
    assert guard.stopped
    assert not guard._sampler_thread.is_alive()


def test_guard_starts_before_batch_clones_and_stops_after_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = sensitivity._batch
    observations: list[bool] = []
    device_observations: list[bool] = []
    layout_observations: list[bool] = []
    original_devices = sensitivity._guard_model_devices
    original_layout = sensitivity._layout

    def checked_devices(model):
        device_observations.append(
            any(
                thread.name == "qwen3-sensitivity-rss-sampler" and thread.is_alive()
                for thread in sensitivity.threading.enumerate()
            )
        )
        return original_devices(model)

    def checked_layout(*args, **kwargs):
        layout_observations.append(
            any(
                thread.name == "qwen3-sensitivity-rss-sampler" and thread.is_alive()
                for thread in sensitivity.threading.enumerate()
            )
        )
        return original_layout(*args, **kwargs)

    def checked_batch(*args, **kwargs):
        observations.append(
            any(
                thread.name == "qwen3-sensitivity-rss-sampler" and thread.is_alive()
                for thread in sensitivity.threading.enumerate()
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(sensitivity, "_guard_model_devices", checked_devices)
    monkeypatch.setattr(sensitivity, "_layout", checked_layout)
    monkeypatch.setattr(sensitivity, "_batch", checked_batch)
    evidence = _collect(_tiny_qwen(1783))
    assert device_observations == [True]
    assert layout_observations == [True]
    assert observations == [True, True]
    assert evidence.report.process_rss_sampler_stopped


def test_raw_storage_mutation_from_forward_hook_fails_exact_final_rehash() -> None:
    model = _tiny_qwen(1784)
    target = model.model.layers[0].mlp.gate_proj.weight
    version = target._version

    def mutate_raw_storage(_module, _inputs, output):
        target.data[0, 0].add_(0.01)
        return output

    handle = model.model.layers[0].register_forward_hook(mutate_raw_storage)
    try:
        with pytest.raises(Qwen3SensitivityStateError, match="exact post-forward"):
            _collect(model)
    finally:
        handle.remove()
    assert target._version == version


def test_direct_storage_mutation_and_forward_failure_still_run_final_rehash() -> None:
    model = _tiny_qwen(1789)
    target = model.model.layers[0].mlp.gate_proj.weight
    pointer = target.untyped_storage().data_ptr()
    version = target._version
    original_byte = ctypes.c_ubyte.from_address(pointer).value
    changed_byte = (original_byte + 1) % 256

    def mutate_storage_then_raise(_module, _inputs, _output):
        _set_first_storage_byte(target, changed_byte)
        raise RuntimeError("adversarial forward failure")

    handle = model.model.layers[0].register_forward_hook(mutate_storage_then_raise)
    try:
        with pytest.raises(Qwen3SensitivityStateError, match="exact post-forward"):
            _collect(model)
    finally:
        handle.remove()
    assert target.untyped_storage().data_ptr() == pointer
    assert target._version == version


def test_successful_collection_rehashes_every_state_tensor_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_qwen(1790)
    expected = set(sensitivity._model_state_tensors(model))
    observed: list[str] = []
    original = sensitivity._state_tensor_fingerprint

    def counted(name, value, chunk_bytes, checkpoint):
        observed.append(name)
        return original(name, value, chunk_bytes, checkpoint)

    monkeypatch.setattr(sensitivity, "_state_tensor_fingerprint", counted)
    _collect(model)
    assert set(observed) == expected
    assert all(observed.count(name) == 2 for name in expected)


def test_runtime_fingerprint_binds_execution_settings_and_hooks() -> None:
    model = _tiny_qwen(1785)
    first = _collect(model)
    second = _collect(
        model,
        limits=Qwen3SensitivityLimits(max_codebook_channels_per_chunk=64),
    )
    assert first.fingerprints.model == second.fingerprints.model
    assert first.fingerprints.runtime != second.fingerprints.runtime
    assert first.fingerprints.cache != second.fingerprints.cache
    assert first.fingerprints.sketch_basis == second.fingerprints.sketch_basis
    assert first.neurons == second.neurons
    assert first.attention_groups == second.attention_groups
    assert first.report.runtime_fingerprint == first.fingerprints.runtime

    layout = sensitivity._layout(model, Qwen3SensitivityLimits())
    runtime = sensitivity._runtime_identity(model, layout, Qwen3SensitivityLimits())
    assert runtime["attention_backend"]
    assert runtime["state_device_dtypes"] == [["cpu", "torch.float32"]]
    assert runtime["state_tensor_placements"]
    assert runtime["packages"]["transformers"]
    assert runtime["platform"]["version"]
    assert runtime["torch"]["build_config_sha256"]
    assert runtime["execution_implementation_sha256"]
    assert runtime["execution"]["codebook_channels_per_chunk"] == 128
    assert runtime["hooks"]["cross_runtime_reuse"]
    assert runtime["cross_runtime_cache_reuse"]["allowed"]

    alternate_name = "eager" if layout.attention_backend != "eager" else "sdpa"
    alternate_backend = sensitivity._runtime_identity(
        model,
        replace(layout, attention_backend=alternate_name),
        Qwen3SensitivityLimits(),
    )
    assert sensitivity._sha256_json(alternate_backend) != sensitivity._sha256_json(
        runtime
    )

    handle = model.model.layers[0].register_forward_hook(lambda *_args: None)
    try:
        hooked = sensitivity._runtime_identity(model, layout, Qwen3SensitivityLimits())
        hooked_again = sensitivity._runtime_identity(
            model,
            layout,
            Qwen3SensitivityLimits(),
        )
    finally:
        handle.remove()
    assert not hooked["hooks"]["cross_runtime_reuse"]
    assert hooked["hooks"]["cache_nonce"]
    assert not hooked["cross_runtime_cache_reuse"]["allowed"]
    assert hooked["hooks"]["cache_nonce"] != hooked_again["hooks"]["cache_nonce"]
    assert sensitivity._sha256_json(hooked) != sensitivity._sha256_json(hooked_again)


def test_cpu_runtime_identity_does_not_initialize_cuda_or_cudnn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> int:
        raise AssertionError("CPU evidence initialized cuDNN")

    monkeypatch.setattr(torch.backends.cudnn, "version", forbidden)
    evidence = _collect(_tiny_qwen(1787))
    assert evidence.report.model_devices == ("cpu",)
    assert evidence.fingerprints.runtime


def test_codebook_full_materialization_is_preflighted_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_qwen(1786)
    tokenizer = _Tokenizer()
    prepare_qwen3_codebook_model(
        model,
        tokenizer,
        tokenizer,
        neuron_prototypes_per_neuron=2,
        attention_prototypes_per_group=2,
    )
    limits = Qwen3SensitivityLimits()
    layout = sensitivity._layout(model, limits)
    batch = sensitivity._batch(
        _calibration_a(),
        None,
        None,
        split_name="calibration",
        max_rows=limits.max_calibration_rows,
        limits=limits,
    )
    dense, materialization, total = sensitivity._mlp_forward_workspace_bound(
        layout,
        batch,
        1,
    )
    registry = model.model.layers[0].mlp.registry
    scaled = registry.neuron_scales is not None
    matrix_copies = 6 if scaled else 3
    bias_count = int(registry.prototype_gate_bias is not None) + int(
        registry.prototype_up_bias is not None
    )
    bias_copies = bias_count * (2 if scaled else 1)
    element_bytes = max(
        registry.prototype_gate_weight.element_size(),
        registry.prototype_up_weight.element_size(),
        registry.prototype_down_weight.element_size(),
        registry.neuron_scales.element_size() if scaled else 0,
    )
    expected = (
        matrix_copies * layout.hidden_size * layout.intermediate_width
        + bias_copies * layout.intermediate_width
    ) * element_bytes + 2 * layout.intermediate_width * 8
    assert dense == 0
    assert materialization == expected
    assert total > materialization

    forwarded = False

    def forbidden_forward(*_args, **_kwargs):
        nonlocal forwarded
        forwarded = True
        raise AssertionError("preflight allowed a model forward")

    monkeypatch.setattr(model, "forward", forbidden_forward)
    tiny_limit = Qwen3SensitivityLimits(
        max_collector_bytes=1_000,
        max_fingerprint_chunk_bytes=500,
        max_sketch_work_bytes=250,
        max_process_rss_increase_bytes=1024**3,
        max_cuda_allocated_increase_bytes=1024**3,
        max_cuda_reserved_increase_bytes=1024**3,
    )
    with pytest.raises(Qwen3SensitivityLimitError, match="Baseline collector peak"):
        _collect(model, limits=tiny_limit)
    assert not forwarded


def test_dense_mlp_and_individual_resource_caps_are_preflighted_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_qwen(1791)
    base_limits = Qwen3SensitivityLimits()
    layout = sensitivity._layout(model, base_limits)
    batch = sensitivity._batch(
        _calibration_a(),
        None,
        None,
        split_name="calibration",
        max_rows=base_limits.max_calibration_rows,
        limits=base_limits,
    )
    dense, materialization, total = sensitivity._mlp_forward_workspace_bound(
        layout,
        batch,
        1,
    )
    expected_live = (
        batch.rows
        * batch.sequence_length
        * (5 * layout.intermediate_width + 3 * layout.hidden_size)
        * 4
    )
    assert dense >= expected_live + sensitivity._MLP_KERNEL_WORKSPACE_BOUND_BYTES
    assert materialization == 0
    assert total == dense

    forwarded = False

    def forbidden_forward(*_args, **_kwargs):
        nonlocal forwarded
        forwarded = True
        raise AssertionError("resource preflight allowed a model forward")

    monkeypatch.setattr(model, "forward", forbidden_forward)
    limits = Qwen3SensitivityLimits(
        max_collector_bytes=1024**3,
        max_process_rss_increase_bytes=dense - 1,
        max_cuda_allocated_increase_bytes=1024**3,
        max_cuda_reserved_increase_bytes=1024**3,
    )
    with pytest.raises(
        Qwen3SensitivityLimitError,
        match="Predicted forward process RSS peak",
    ):
        _collect(model, limits=limits)
    assert not forwarded


def test_four_b_geometry_bounds_dense_temporaries_and_full_codebook_matrices() -> None:
    limits = Qwen3SensitivityLimits(max_sequence_length=4096)
    dense_model = _tiny_qwen(1792)
    dense_layout = replace(
        sensitivity._layout(dense_model, limits),
        hidden_size=2_560,
        intermediate_width=9_728,
        activation_element_bytes=2,
    )
    ids = torch.ones((2, 2_048), dtype=torch.long)
    large_batch = sensitivity._batch(
        ids,
        torch.ones_like(ids),
        None,
        split_name="calibration",
        max_rows=limits.max_calibration_rows,
        limits=limits,
    )
    dense, _, dense_total = sensitivity._mlp_forward_workspace_bound(
        dense_layout,
        large_batch,
        1,
    )
    ordinary_live_values = (
        2
        * 2_048
        * (5 * dense_layout.intermediate_width + 3 * dense_layout.hidden_size)
        * 4
    )
    assert dense >= ordinary_live_values
    assert dense_total == dense

    codebook_model = _tiny_qwen(1793)
    tokenizer = _Tokenizer()
    prepare_qwen3_codebook_model(
        codebook_model,
        tokenizer,
        tokenizer,
        neuron_prototypes_per_neuron=2,
        attention_prototypes_per_group=2,
    )
    codebook_layout = replace(
        sensitivity._layout(codebook_model, limits),
        hidden_size=2_560,
        intermediate_width=9_728,
        activation_element_bytes=2,
    )
    _, materialization, codebook_total = sensitivity._mlp_forward_workspace_bound(
        codebook_layout,
        large_batch,
        1,
    )
    registry = codebook_model.model.layers[0].mlp.registry
    copies = 6 if registry.neuron_scales is not None else 3
    minimum_full_matrices = (
        copies
        * codebook_layout.hidden_size
        * codebook_layout.intermediate_width
        * registry.prototype_gate_weight.element_size()
    )
    assert materialization >= minimum_full_matrices
    assert codebook_total > materialization


def test_attention_workspace_bound_is_backend_aware() -> None:
    limits = Qwen3SensitivityLimits()
    layout = sensitivity._layout(_tiny_qwen(1782), limits)
    batch = sensitivity._batch(
        _calibration_a(),
        torch.ones_like(_calibration_a()),
        None,
        split_name="calibration",
        max_rows=limits.max_calibration_rows,
        limits=limits,
    )
    eager = sensitivity._attention_workspace_bound(
        replace(layout, attention_backend="eager"),
        batch,
        2,
    )
    flash = sensitivity._attention_workspace_bound(
        replace(layout, attention_backend="flash_attention_2"),
        batch,
        2,
    )
    assert eager > flash > 0


def test_gate_rows_are_only_fingerprinted_and_never_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_qwen(1781)
    calibration = _calibration_a()
    gate = _gate_a()
    observed: list[torch.Tensor] = []
    original = model.forward

    def tracked_forward(*args, **kwargs):
        observed.append(kwargs["input_ids"].detach().to("cpu").clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", tracked_forward)
    _collect(model, calibration, gate)
    assert observed
    assert all(
        torch.equal(
            rows, calibration.repeat((rows.shape[0] // calibration.shape[0], 1))
        )
        for rows in observed
    )
    assert all(not torch.equal(rows[:1], gate) for rows in observed)
