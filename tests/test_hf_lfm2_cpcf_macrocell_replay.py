from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_lfm2_cpcf_macrocell_replay import (  # noqa: E402
    LFM2_CPCF_MACROCELL_REPLAY_SCHEMA,
    Lfm2CpcfMacrocellReplay,
    Lfm2CpcfMacrocellReplayError,
    replay_lfm2_cpcf_macrocell,
)


_DTYPES = (torch.float32, torch.bfloat16)
_KINDS = ("full_attention", "short_conv")


def _tiny_installed_target(
    operator_kind: str,
    dtype: Any,
    *,
    seed: int,
    head_dim: int = 4,
) -> Any:
    config = transformers.Lfm2Config(
        vocab_size=37,
        hidden_size=8,
        intermediate_size=13,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
        layer_types=["full_attention" if operator_kind == "full_attention" else "conv"],
        conv_L_cache=3,
        conv_bias=True,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config)
    model.to(dtype=dtype)
    model.eval()

    # Exercise both legal normalization gauges in every replay: the operator
    # normalization is the compiler's unit gauge and FFN/QK gains are
    # caller-installed nonunit target values.
    layer = model.model.layers[0]
    with torch.no_grad():
        layer.operator_norm.weight.fill_(1.0)
        layer.ffn_norm.weight.copy_(
            torch.linspace(
                0.65,
                1.35,
                config.hidden_size,
                dtype=torch.float32,
            ).to(dtype)
        )
        if operator_kind == "full_attention":
            layer.self_attn.q_layernorm.weight.copy_(
                torch.linspace(
                    0.75,
                    1.25,
                    head_dim,
                    dtype=torch.float32,
                ).to(dtype)
            )
            layer.self_attn.k_layernorm.weight.copy_(
                torch.linspace(
                    1.2,
                    0.8,
                    head_dim,
                    dtype=torch.float32,
                ).to(dtype)
            )
    return model


def _actual_prefill_residual(model: Any) -> Any:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ],
        dtype=torch.long,
        device=model.model.embed_tokens.weight.device,
    )
    with torch.no_grad():
        return model.model.embed_tokens(input_ids).detach().clone()


def _causal_mask(residual: Any) -> Any:
    batch, sequence = int(residual.shape[0]), int(residual.shape[1])
    mask = torch.zeros(
        (batch, 1, sequence, sequence),
        device=residual.device,
        dtype=residual.dtype,
    )
    future = torch.triu(
        torch.ones(
            (sequence, sequence),
            device=residual.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    return mask.masked_fill(future, torch.finfo(residual.dtype).min)


def _position_embeddings(model: Any, residual: Any) -> tuple[Any, Any]:
    positions = torch.arange(
        int(residual.shape[1]),
        device=residual.device,
        dtype=torch.long,
    ).unsqueeze(0)
    positions = positions.expand(int(residual.shape[0]), -1)
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(residual, position_ids=positions)
    return cos.detach().clone(), sin.detach().clone()


def _clone_output(value: Any) -> Any:
    if isinstance(value, tuple):
        value = value[0]
    return value.detach().clone()


def _reference_layer_replay(
    layer: Any,
    residual: Any,
    *,
    position_embeddings: tuple[Any, Any] | None,
    attention_mask: Any,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    handles = []

    def capture(name: str) -> Callable[..., None]:
        def observed(_module: Any, _inputs: Any, output: Any) -> None:
            captured[name] = _clone_output(output)

        return observed

    operator = layer.self_attn if layer.is_attention_layer else layer.conv
    handles.append(
        operator.out_proj.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__(
                "operator_output_input",
                inputs[0].detach().clone(),
            )
        )
    )
    for name, module in (
        ("operator_input", layer.operator_norm),
        ("operator_delta", operator),
        ("mlp_input", layer.ffn_norm),
        ("mlp_delta", layer.feed_forward),
    ):
        handles.append(module.register_forward_hook(capture(name)))
    try:
        with torch.no_grad():
            post_layer = layer(
                residual.detach().clone(),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=None,
            )
    finally:
        for handle in handles:
            handle.remove()

    captured["post_operator_residual"] = residual + captured["operator_delta"]
    captured["post_layer_residual"] = post_layer.detach().clone()
    return captured


def _forbid_module_forwards(layer: Any, monkeypatch: Any) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "CPCF macrocell replay invoked a target module forward method"
        )

    modules = [
        layer,
        layer.operator_norm,
        layer.ffn_norm,
        layer.feed_forward,
        layer.feed_forward.w1,
        layer.feed_forward.w2,
        layer.feed_forward.w3,
    ]
    if layer.is_attention_layer:
        modules.extend(
            [
                layer.self_attn,
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.out_proj,
                layer.self_attn.q_layernorm,
                layer.self_attn.k_layernorm,
            ]
        )
    else:
        modules.extend(
            [
                layer.conv,
                layer.conv.in_proj,
                layer.conv.conv,
                layer.conv.out_proj,
            ]
        )
    for module in modules:
        monkeypatch.setattr(module, "forward", forbidden)


def _gradient_snapshot(layer: Any) -> dict[str, Any | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in layer.named_parameters()
    }


def _assert_gradients_equal(
    expected: dict[str, Any | None],
    layer: Any,
) -> None:
    observed = dict(layer.named_parameters())
    assert set(expected) == set(observed)
    for name, value in expected.items():
        if value is None:
            assert observed[name].grad is None
        else:
            assert observed[name].grad is not None
            assert torch.equal(value, observed[name].grad)


@pytest.mark.parametrize("operator_kind", _KINDS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_replay_matches_real_transformers_lfm2_layer_forward_without_mutation(
    operator_kind: str,
    dtype: Any,
    monkeypatch: Any,
) -> None:
    model = _tiny_installed_target(
        operator_kind,
        dtype,
        seed=9100 + _KINDS.index(operator_kind) * 100 + _DTYPES.index(dtype),
    )
    assert model.config._attn_implementation == "eager"
    layer = model.model.layers[0]
    residual = _actual_prefill_residual(model)
    residual_before = residual.detach().clone()

    # Pre-existing gradients are state too.  Replay must neither produce new
    # gradients nor clear/alter a caller-owned gradient buffer.
    first_parameter = next(layer.parameters())
    first_parameter.grad = torch.full_like(first_parameter, 0.125)
    gradients_before = _gradient_snapshot(layer)
    requires_grad_before = {
        name: bool(parameter.requires_grad)
        for name, parameter in layer.named_parameters()
    }
    modes_before = {
        name: bool(module.training) for name, module in layer.named_modules()
    }

    if operator_kind == "full_attention":
        position_embeddings = _position_embeddings(model, residual)
        attention_mask = _causal_mask(residual)
        replay_kwargs = {
            "position_embeddings": position_embeddings,
            "additive_causal_mask": attention_mask,
        }
    else:
        position_embeddings = None
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
            ],
            device=residual.device,
            dtype=torch.long,
        )
        replay_kwargs = {"padding_mask": attention_mask}

    reference = _reference_layer_replay(
        layer,
        residual,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )
    _forbid_module_forwards(layer, monkeypatch)

    # Call while autograd is enabled to prove the function supplies its own
    # no-grad boundary.
    assert torch.is_grad_enabled()
    replay = replay_lfm2_cpcf_macrocell(
        layer,
        residual,
        **replay_kwargs,
    )

    for name in (
        "operator_input",
        "operator_output_input",
        "operator_delta",
        "post_operator_residual",
        "mlp_input",
        "mlp_delta",
        "post_layer_residual",
    ):
        observed = getattr(replay, name)
        assert torch.equal(observed, reference[name]), (
            name,
            dtype,
            operator_kind,
            float(
                (observed.to(torch.float32) - reference[name].to(torch.float32))
                .abs()
                .max()
                .item()
            ),
        )
        assert not observed.requires_grad
        assert observed.grad_fn is None
        assert observed.device.type == "cpu"
        assert observed.dtype == dtype

    assert torch.equal(residual, residual_before)
    _assert_gradients_equal(gradients_before, layer)
    assert requires_grad_before == {
        name: bool(parameter.requires_grad)
        for name, parameter in layer.named_parameters()
    }
    assert modes_before == {
        name: bool(module.training) for name, module in layer.named_modules()
    }

    report = replay.report
    assert report["schema"] == LFM2_CPCF_MACROCELL_REPLAY_SCHEMA
    assert report["operator_kind"] == operator_kind
    assert report["operator_output_input_source"] == (
        "attention.concatenated_context"
        if operator_kind == "full_attention"
        else "short_conv.output_input"
    )
    source_detail_name = (
        "concatenated_context" if operator_kind == "full_attention" else "output_input"
    )
    assert (
        report["endpoint_sha256"]["operator_output_input"]
        == report["operator_detail_sha256"][source_detail_name]
    )
    assert report["runtime_dtype"] == str(dtype).removeprefix("torch.")
    assert report["target_state_sha256_before"] == report["target_state_sha256_after"]
    assert report["input_sha256_before"] == report["input_sha256_after"]
    assert report["target_state_bytes_preserved"]
    assert report["target_tensor_identities_preserved"]
    assert report["target_gradients_preserved"]
    assert report["target_training_modes_preserved"]
    assert report["returned_tensors_detached"]
    assert report["returned_tensors_archived_on_cpu"]
    assert not report["accelerator_storage_retained_by_returned_tensors"]
    equations = report["residual_equations"]
    assert equations["post_operator_equals_prefill_plus_operator_delta"]
    assert equations["post_operator_maximum_absolute_error"] == 0.0
    assert equations["post_layer_equals_post_operator_plus_mlp_delta"]
    assert equations["post_layer_maximum_absolute_error"] == 0.0
    execution = report["execution"]
    assert execution["entry_grad_enabled"]
    assert not execution["grad_enabled_inside_replay"]
    for name in (
        "parent_forward_count",
        "child_model_forward_count",
        "target_layer_forward_count",
        "cache_generation_count",
        "cache_update_count",
        "training_step_count",
        "backward_call_count",
        "optimizer_step_count",
        "parameters_pruned",
        "parameters_quantized",
    ):
        assert execution[name] == 0
    assert replay.replay_sha256 == report["report_sha256"]


@pytest.mark.parametrize("head_dim", (2, 6))
def test_attention_replay_allows_projection_width_different_from_residual_stream(
    head_dim: int,
) -> None:
    model = _tiny_installed_target(
        "full_attention",
        torch.float32,
        seed=9289,
        head_dim=head_dim,
    )
    layer = model.model.layers[0]
    residual = _actual_prefill_residual(model)
    position_embeddings = _position_embeddings(model, residual)
    attention_mask = _causal_mask(residual)
    reference = _reference_layer_replay(
        layer,
        residual,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )

    replay = replay_lfm2_cpcf_macrocell(
        layer,
        residual,
        position_embeddings=position_embeddings,
        additive_causal_mask=attention_mask,
    )

    residual_shape = (2, 5, 8)
    for name in (
        "operator_input",
        "operator_delta",
        "post_operator_residual",
        "mlp_input",
        "mlp_delta",
        "post_layer_residual",
    ):
        assert tuple(getattr(replay, name).shape) == residual_shape
        assert torch.equal(getattr(replay, name), reference[name])
    projection_width = 2 * head_dim
    assert tuple(replay.operator_output_input.shape) == (
        2,
        5,
        projection_width,
    )
    assert torch.equal(
        replay.operator_output_input,
        reference["operator_output_input"],
    )
    assert replay.report["operator_topology"]["query_heads"] == 2
    assert replay.report["operator_topology"]["head_dim"] == head_dim
    assert projection_width != residual_shape[-1]


@pytest.mark.parametrize(
    ("changed_endpoint", "expected_error"),
    (
        ("operator_output_input", "batch and sequence dimensions"),
        ("operator_delta", "Residual replay endpoint shapes differ"),
    ),
)
def test_replay_archive_rejects_invalid_endpoint_geometry(
    changed_endpoint: str,
    expected_error: str,
) -> None:
    model = _tiny_installed_target(
        "full_attention",
        torch.float32,
        seed=9290,
        head_dim=6,
    )
    residual = _actual_prefill_residual(model)
    replay = replay_lfm2_cpcf_macrocell(
        model.model.layers[0],
        residual,
        position_embeddings=_position_embeddings(model, residual),
        additive_causal_mask=_causal_mask(residual),
    )
    endpoints = {
        "operator_input": replay.operator_input,
        "operator_output_input": replay.operator_output_input,
        "operator_delta": replay.operator_delta,
        "post_operator_residual": replay.post_operator_residual,
        "mlp_input": replay.mlp_input,
        "mlp_delta": replay.mlp_delta,
        "post_layer_residual": replay.post_layer_residual,
    }
    changed = endpoints[changed_endpoint]
    if changed_endpoint == "operator_output_input":
        endpoints[changed_endpoint] = changed[:1].clone()
    else:
        endpoints[changed_endpoint] = changed[:, :, :-1].clone()

    with pytest.raises(Lfm2CpcfMacrocellReplayError, match=expected_error):
        Lfm2CpcfMacrocellReplay(
            **endpoints,
            report=replay.report,
        )


@pytest.mark.parametrize(
    "changed_endpoint",
    ("operator_output_input", "post_layer_residual"),
)
def test_replay_archive_rejects_endpoint_bytes_outside_its_seal(
    changed_endpoint: str,
) -> None:
    model = _tiny_installed_target(
        "short_conv",
        torch.float32,
        seed=9290,
    )
    replay = replay_lfm2_cpcf_macrocell(
        model.model.layers[0],
        _actual_prefill_residual(model),
        padding_mask=torch.ones(2, 5),
    )
    changed = getattr(replay, changed_endpoint).clone()
    changed[0, 0, 0] += 1.0
    endpoints = {
        "operator_input": replay.operator_input,
        "operator_output_input": replay.operator_output_input,
        "operator_delta": replay.operator_delta,
        "post_operator_residual": replay.post_operator_residual,
        "mlp_input": replay.mlp_input,
        "mlp_delta": replay.mlp_delta,
        "post_layer_residual": replay.post_layer_residual,
    }
    endpoints[changed_endpoint] = changed

    with pytest.raises(
        Lfm2CpcfMacrocellReplayError,
        match="authenticated report",
    ):
        Lfm2CpcfMacrocellReplay(
            **endpoints,
            report=replay.report,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to prove accelerator-free returned archives.",
)
def test_cuda_replay_returns_no_cuda_endpoint_storage() -> None:
    model = _tiny_installed_target(
        "full_attention",
        torch.float32,
        seed=9291,
    ).to("cuda")
    residual = _actual_prefill_residual(model)
    replay = replay_lfm2_cpcf_macrocell(
        model.model.layers[0],
        residual,
        position_embeddings=_position_embeddings(model, residual),
        additive_causal_mask=_causal_mask(residual),
    )

    assert residual.device.type == "cuda"
    assert replay.report["runtime_device"].startswith("cuda")
    assert all(
        getattr(replay, name).device.type == "cpu"
        for name in (
            "operator_input",
            "operator_output_input",
            "operator_delta",
            "post_operator_residual",
            "mlp_input",
            "mlp_delta",
            "post_layer_residual",
        )
    )


def test_attention_contract_rejects_noncausal_mask_before_replay() -> None:
    model = _tiny_installed_target(
        "full_attention",
        torch.float32,
        seed=9301,
    )
    layer = model.model.layers[0]
    residual = _actual_prefill_residual(model)
    position_embeddings = _position_embeddings(model, residual)
    noncausal = torch.zeros(
        (
            int(residual.shape[0]),
            1,
            int(residual.shape[1]),
            int(residual.shape[1]),
        ),
        dtype=residual.dtype,
    )
    with pytest.raises(
        Lfm2CpcfMacrocellReplayError,
        match="does not mask every future position",
    ):
        replay_lfm2_cpcf_macrocell(
            layer,
            residual,
            position_embeddings=position_embeddings,
            additive_causal_mask=noncausal,
        )


def test_public_api_has_no_parent_cache_or_training_surface() -> None:
    parameters = inspect.signature(replay_lfm2_cpcf_macrocell).parameters
    assert tuple(parameters) == (
        "target_layer",
        "prefill_residual_states",
        "position_embeddings",
        "additive_causal_mask",
        "padding_mask",
    )
    for forbidden in (
        "parent",
        "parent_model",
        "cache",
        "tokenizer",
        "optimizer",
        "training",
        "quantization",
        "pruning",
    ):
        assert forbidden not in parameters
