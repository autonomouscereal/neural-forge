from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest
import torch

from neural_forge.hf_lfm2_fused_body_diagnostic import (
    Lfm2FusedBodyDiagnosticError,
    Lfm2FusedMlpDiagnosticInput,
    Lfm2FusedShortConvDiagnosticInput,
    authenticate_lfm2_fused_body_candidate,
    compile_lfm2_fused_mlp_diagnostic,
    compile_lfm2_fused_short_conv_diagnostic,
    fingerprint_lfm2_fused_body_diagnostic_input,
    lfm2_fused_mlp_parameter_accounting,
    lfm2_fused_short_conv_parameter_accounting,
    reconstruct_lfm2_fused_body_candidate,
)


_PARENT_HIDDEN = 8
_COMPACT_HIDDEN = 6
_PARENT_INTERMEDIATE = 12
_COMPACT_INTERMEDIATE = 4
_KERNEL = 3
_CONV_LAYER = 1


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=dtype)


def _mlp_source() -> Lfm2FusedMlpDiagnosticInput:
    generator = torch.Generator(device="cpu").manual_seed(411)
    return Lfm2FusedMlpDiagnosticInput(
        parent_hidden_size=_PARENT_HIDDEN,
        parent_intermediate_size=_PARENT_INTERMEDIATE,
        compact_hidden_size=_COMPACT_HIDDEN,
        compact_intermediate_size=_COMPACT_INTERMEDIATE,
        compact_w1_weight=_randn(
            (_COMPACT_INTERMEDIATE, _COMPACT_HIDDEN),
            generator=generator,
        ),
        compact_w2_weight=_randn(
            (_COMPACT_HIDDEN, _COMPACT_INTERMEDIATE),
            generator=generator,
        ),
        compact_w3_weight=_randn(
            (_COMPACT_INTERMEDIATE, _COMPACT_HIDDEN),
            generator=generator,
        ),
        input_encoder=_randn(
            (_PARENT_HIDDEN, _COMPACT_HIDDEN),
            generator=generator,
            dtype=torch.float64,
        ),
        output_decoder=_randn(
            (_COMPACT_HIDDEN, _PARENT_HIDDEN),
            generator=generator,
            dtype=torch.float64,
        ),
    )


def _conv_source() -> Lfm2FusedShortConvDiagnosticInput:
    generator = torch.Generator(device="cpu").manual_seed(733)
    return Lfm2FusedShortConvDiagnosticInput(
        layer_index=_CONV_LAYER,
        parent_hidden_size=_PARENT_HIDDEN,
        compact_hidden_size=_COMPACT_HIDDEN,
        kernel_size=_KERNEL,
        compact_conv_weight=_randn(
            (_COMPACT_HIDDEN, 1, _KERNEL),
            generator=generator,
        ),
        compact_in_proj_weight=_randn(
            (3 * _COMPACT_HIDDEN, _COMPACT_HIDDEN),
            generator=generator,
        ),
        compact_out_proj_weight=_randn(
            (_COMPACT_HIDDEN, _COMPACT_HIDDEN),
            generator=generator,
        ),
        input_encoder=_randn(
            (_PARENT_HIDDEN, _COMPACT_HIDDEN),
            generator=generator,
            dtype=torch.float64,
        ),
        output_decoder=_randn(
            (_COMPACT_HIDDEN, _PARENT_HIDDEN),
            generator=generator,
            dtype=torch.float64,
        ),
    )


def _pins(source: Any) -> tuple[str, dict[str, str]]:
    authentication = fingerprint_lfm2_fused_body_diagnostic_input(source)
    return authentication["input_bundle_sha256"], {
        name: record["sha256"]
        for name, record in authentication["tensor_records"].items()
    }


def _compile_mlp(source: Lfm2FusedMlpDiagnosticInput):
    bundle, pins = _pins(source)
    return compile_lfm2_fused_mlp_diagnostic(
        source,
        expected_input_bundle_sha256=bundle,
        expected_tensor_sha256=pins,
    )


def _compile_conv(source: Lfm2FusedShortConvDiagnosticInput):
    bundle, pins = _pins(source)
    return compile_lfm2_fused_short_conv_diagnostic(
        source,
        expected_input_bundle_sha256=bundle,
        expected_tensor_sha256=pins,
    )


def _parent_config() -> Any:
    transformers = pytest.importorskip("transformers")
    return transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=_PARENT_HIDDEN,
        intermediate_size=_PARENT_INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["full_attention", "conv"],
        conv_L_cache=_KERNEL,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
    )


def _compact_conv_module(source: Lfm2FusedShortConvDiagnosticInput) -> Any:
    transformers = pytest.importorskip("transformers")
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    config = transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=_COMPACT_HIDDEN,
        intermediate_size=_COMPACT_INTERMEDIATE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=3,
        layer_types=["full_attention", "conv"],
        conv_L_cache=_KERNEL,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
    )
    module = Lfm2ShortConv(config, _CONV_LAYER).to(dtype=torch.float32)
    with torch.no_grad():
        module.conv.weight.copy_(source.compact_conv_weight)
        module.in_proj.weight.copy_(source.compact_in_proj_weight)
        module.out_proj.weight.copy_(source.compact_out_proj_weight)
    module.requires_grad_(False)
    module.eval()
    return module


def _reconstruct(candidate: Any) -> Any:
    return reconstruct_lfm2_fused_body_candidate(
        candidate=candidate,
        parent_config=_parent_config(),
        expected_candidate_result_sha256=candidate.receipt["result_sha256"],
        expected_descriptor_sha256=(
            candidate.reconstruction_descriptor["descriptor_sha256"]
        ),
        device="cpu",
    )


def _alias_group_count(module: Any) -> int:
    groups: dict[tuple[str, str, int, int], list[str]] = {}
    for name, tensor in module.named_parameters(remove_duplicate=False):
        identity = (
            str(tensor.device),
            str(tensor.dtype),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        groups.setdefault(identity, []).append(name)
    return sum(len(names) > 1 for names in groups.values())


def test_actual_exact4x_parameter_counts_match_audited_geometry() -> None:
    mlp = lfm2_fused_mlp_parameter_accounting(
        parent_hidden_size=1024,
        parent_intermediate_size=2560,
        compact_hidden_size=776,
        compact_intermediate_size=700,
    )
    assert mlp == {
        "original_parent_component_parameter_elements": 7_864_320,
        "compact_operator_parameter_elements": 1_629_600,
        "current_bridged_wrapper_parameter_elements": 3_218_848,
        "current_input_bridge_parameter_elements": 794_624,
        "current_output_bridge_parameter_elements": 794_624,
        "fused_candidate_parameter_elements": 2_150_400,
        "fused_candidate_stored_tensor_elements": 2_150_400,
        "fused_candidate_parameter_bytes_fp32": 8_601_600,
        "physical_parameter_reduction_from_parent": 5_713_920,
        "physical_parameter_reduction_from_current_wrapper": 1_068_448,
        "runtime_input_bridge_parameter_elements": 0,
        "runtime_output_bridge_parameter_elements": 0,
        "runtime_sidecar_parameter_elements": 0,
        "runtime_shared_parameter_elements": 0,
        "runtime_parameter_alias_group_count": 0,
        "parameter_reduction_attributed_to_sharing": False,
    }
    conv = lfm2_fused_short_conv_parameter_accounting(
        parent_hidden_size=1024,
        compact_hidden_size=776,
        kernel_size=3,
    )
    assert conv["original_parent_component_parameter_elements"] == 4_197_376
    assert conv["compact_operator_parameter_elements"] == 2_411_032
    assert conv["current_bridged_wrapper_parameter_elements"] == 4_000_280
    assert conv["fused_candidate_parameter_elements"] == 3_180_824
    assert conv["physical_parameter_reduction_from_parent"] == 1_016_552
    assert conv["physical_parameter_reduction_from_current_wrapper"] == 819_456
    assert conv["runtime_sidecar_parameter_elements"] == 0


def test_mlp_fp32_cast_lineage_and_fp64_algebraic_equivalence() -> None:
    source = _mlp_source()
    authentication = fingerprint_lfm2_fused_body_diagnostic_input(source)
    input_cast = authentication["runtime_bridge_cast_lineage"]["input_encoder"]
    output_cast = authentication["runtime_bridge_cast_lineage"]["output_decoder"]
    assert input_cast["source"]["dtype"] == "float64"
    assert input_cast["runtime_fp32"]["dtype"] == "float32"
    assert input_cast["cast_performed"] is True
    assert output_cast["cast_performed"] is True

    candidate = _compile_mlp(source)
    receipt = authenticate_lfm2_fused_body_candidate(candidate)
    state = candidate.state_dict()
    encoder = source.input_encoder.float()
    decoder = source.output_decoder.float()
    torch.testing.assert_close(
        state["w1.weight"],
        (source.compact_w1_weight.double() @ encoder.double().T).float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state["w3.weight"],
        (source.compact_w3_weight.double() @ encoder.double().T).float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state["w2.weight"],
        (decoder.double().T @ source.compact_w2_weight.double()).float(),
        rtol=0.0,
        atol=0.0,
    )
    assert receipt["operation_audit"]["training_steps"] == 0
    assert receipt["operation_audit"]["parameters_pruned"] == 0
    assert receipt["operation_audit"]["parameters_quantized"] == 0

    generator = torch.Generator(device="cpu").manual_seed(991)
    x = _randn((2, 5, _PARENT_HIDDEN), generator=generator, dtype=torch.float64)
    w1 = source.compact_w1_weight.double()
    w2 = source.compact_w2_weight.double()
    w3 = source.compact_w3_weight.double()
    e = encoder.double()
    d = decoder.double()
    explicit_hidden = x @ e
    explicit = (
        (torch.nn.functional.silu(explicit_hidden @ w1.T) * (explicit_hidden @ w3.T))
        @ w2.T
        @ d
    )
    fused = (torch.nn.functional.silu(x @ (w1 @ e.T).T) * (x @ (w3 @ e.T).T)) @ (
        d.T @ w2
    ).T
    torch.testing.assert_close(fused, explicit, rtol=1e-12, atol=1e-12)


def test_short_conv_fp32_cast_lineage_and_fp64_algebraic_equivalence() -> None:
    source = _conv_source()
    candidate = _compile_conv(source)
    authenticate_lfm2_fused_body_candidate(candidate)
    state = candidate.state_dict()
    encoder = source.input_encoder.float()
    decoder = source.output_decoder.float()
    torch.testing.assert_close(
        state["in_proj.weight"],
        (source.compact_in_proj_weight.double() @ encoder.double().T).float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state["out_proj.weight"],
        (decoder.double().T @ source.compact_out_proj_weight.double()).float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        state["conv.weight"],
        source.compact_conv_weight,
        rtol=0.0,
        atol=0.0,
    )

    compact = _compact_conv_module(source).to(dtype=torch.float64)
    fused = copy.deepcopy(compact)
    fused.in_proj = torch.nn.Linear(
        _PARENT_HIDDEN,
        3 * _COMPACT_HIDDEN,
        bias=False,
        dtype=torch.float64,
    )
    fused.out_proj = torch.nn.Linear(
        _COMPACT_HIDDEN,
        _PARENT_HIDDEN,
        bias=False,
        dtype=torch.float64,
    )
    with torch.no_grad():
        fused.in_proj.weight.copy_(
            source.compact_in_proj_weight.double() @ encoder.double().T
        )
        fused.out_proj.weight.copy_(
            decoder.double().T @ source.compact_out_proj_weight.double()
        )
    generator = torch.Generator(device="cpu").manual_seed(1091)
    x = _randn((2, 6, _PARENT_HIDDEN), generator=generator, dtype=torch.float64)
    with torch.inference_mode():
        explicit = compact(hidden_states=x @ encoder.double()) @ decoder.double()
        observed = fused(hidden_states=x)
    torch.testing.assert_close(observed, explicit, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["mlp", "short_conv"])
def test_fresh_reconstruction_has_exact_state_count_and_no_sidecars(kind: str) -> None:
    candidate = (
        _compile_mlp(_mlp_source()) if kind == "mlp" else _compile_conv(_conv_source())
    )
    first = _reconstruct(candidate)
    second = _reconstruct(candidate)
    accounting = candidate.receipt["physical_parameter_accounting"]
    assert (
        first.receipt["parameter_elements"]
        == accounting["fused_candidate_parameter_elements"]
    )
    assert (
        first.receipt["stored_tensor_elements"]
        == accounting["fused_candidate_stored_tensor_elements"]
    )
    assert first.receipt["runtime_parameter_alias_group_count"] == 0
    assert _alias_group_count(first.module) == 0
    assert not hasattr(first.module, "input_bridge")
    assert not hasattr(first.module, "output_bridge")
    assert {name for name, _ in first.module.named_parameters()} == set(
        candidate.state_dict()
    )
    assert all(not parameter.requires_grad for parameter in first.module.parameters())
    for name, tensor in first.module.state_dict().items():
        torch.testing.assert_close(
            tensor,
            candidate.state_dict()[name],
            rtol=0.0,
            atol=0.0,
        )
        assert (
            tensor.untyped_storage().data_ptr()
            != second.module.state_dict()[name].untyped_storage().data_ptr()
        )
    if kind == "mlp":
        assert type(first.module).__name__ == "Lfm2MLP"
        assert first.module.w1.in_features == _PARENT_HIDDEN
        assert first.module.w1.out_features == _COMPACT_INTERMEDIATE
    else:
        assert type(first.module).__name__ == "Lfm2BottleneckShortConvRuntime"
        assert first.module.in_proj.in_features == _PARENT_HIDDEN
        assert first.module.in_proj.out_features == 3 * _COMPACT_HIDDEN
        assert first.module.conv.groups == _COMPACT_HIDDEN
        assert first.module.out_proj.out_features == _PARENT_HIDDEN


def _explicit_bridged_conv(
    source: Lfm2FusedShortConvDiagnosticInput,
    hidden_states: torch.Tensor,
    *,
    past_key_values: Any = None,
    attention_mask: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    compact = _compact_conv_module(source)
    # Tests which need persistent cache call their own persistent compact module.
    return (
        compact(
            hidden_states=hidden_states @ source.input_encoder.float(),
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            seq_idx=seq_idx,
        )
        @ source.output_decoder.float()
    )


def test_bottleneck_short_conv_preserves_mask_and_seq_idx_semantics() -> None:
    source = _conv_source()
    module = _reconstruct(_compile_conv(source)).module
    generator = torch.Generator(device="cpu").manual_seed(1201)
    masked_x = _randn((2, 6, _PARENT_HIDDEN), generator=generator)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1]],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        expected_masked = _explicit_bridged_conv(
            source,
            masked_x,
            attention_mask=mask,
        )
        observed_masked = module(masked_x, attention_mask=mask)
    torch.testing.assert_close(observed_masked, expected_masked, rtol=2e-5, atol=2e-5)

    packed_x = _randn((1, 6, _PARENT_HIDDEN), generator=generator)
    seq_idx = torch.tensor([[0, 0, 1, 1, 1, 2]], dtype=torch.int64)
    with torch.inference_mode():
        expected_packed = _explicit_bridged_conv(
            source,
            packed_x,
            seq_idx=seq_idx,
        )
        observed_packed = module(packed_x, seq_idx=seq_idx)
    torch.testing.assert_close(observed_packed, expected_packed, rtol=2e-5, atol=2e-5)


def test_bottleneck_short_conv_dynamic_cache_matches_bridged_compact_and_is_internal_width() -> (
    None
):
    from transformers.cache_utils import DynamicCache

    source = _conv_source()
    module = _reconstruct(_compile_conv(source)).module
    compact = _compact_conv_module(source)
    parent_config = _parent_config()
    compact_config = copy.deepcopy(parent_config)
    compact_config.hidden_size = _COMPACT_HIDDEN
    fused_cache = DynamicCache(config=parent_config)
    compact_cache = DynamicCache(config=compact_config)
    generator = torch.Generator(device="cpu").manual_seed(1301)
    x = _randn((2, 5, _PARENT_HIDDEN), generator=generator)
    with torch.inference_mode():
        observed_prefill = module(x[:, :4], past_key_values=fused_cache)
        expected_prefill = (
            compact(
                x[:, :4] @ source.input_encoder.float(),
                past_key_values=compact_cache,
            )
            @ source.output_decoder.float()
        )
        observed_decode = module(x[:, 4:], past_key_values=fused_cache)
        expected_decode = (
            compact(
                x[:, 4:] @ source.input_encoder.float(),
                past_key_values=compact_cache,
            )
            @ source.output_decoder.float()
        )
    torch.testing.assert_close(observed_prefill, expected_prefill, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(observed_decode, expected_decode, rtol=2e-5, atol=2e-5)
    assert tuple(fused_cache.layers[_CONV_LAYER].conv_states.shape) == (
        2,
        _COMPACT_HIDDEN,
        _KERNEL,
    )


def test_bottleneck_short_conv_rejects_static_or_stale_parent_width_cache() -> None:
    from transformers.cache_utils import DynamicCache
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    source = _conv_source()
    module = _reconstruct(_compile_conv(source)).module
    x = torch.randn(1, 4, _PARENT_HIDDEN)
    with (
        torch.inference_mode(),
        pytest.raises(
            Lfm2FusedBodyDiagnosticError,
            match="static, compileable, or non-Dynamic",
        ),
    ):
        module(x, past_key_values=object())

    parent_config = _parent_config()
    stale_cache = DynamicCache(config=parent_config)
    parent_conv = Lfm2ShortConv(parent_config, _CONV_LAYER).requires_grad_(False)
    parent_conv.eval()
    with torch.inference_mode():
        parent_conv(x, past_key_values=stale_cache)
    assert stale_cache.layers[_CONV_LAYER].conv_states.shape[1] == _PARENT_HIDDEN
    with (
        torch.inference_mode(),
        pytest.raises(
            Lfm2FusedBodyDiagnosticError,
            match="compact_hidden",
        ),
    ):
        module(x[:, -1:], past_key_values=stale_cache)


def test_fail_closed_on_bias_dtype_nonfinite_alias_and_runtime_nonfinite() -> None:
    mlp = _mlp_source()
    with pytest.raises(Lfm2FusedBodyDiagnosticError, match="bias-free"):
        fingerprint_lfm2_fused_body_diagnostic_input(
            replace(mlp, compact_w1_bias=torch.zeros(_COMPACT_INTERMEDIATE))
        )
    with pytest.raises(Lfm2FusedBodyDiagnosticError, match="runtime FP32"):
        fingerprint_lfm2_fused_body_diagnostic_input(
            replace(mlp, compact_w1_weight=mlp.compact_w1_weight.double())
        )
    nonfinite = mlp.compact_w1_weight.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(Lfm2FusedBodyDiagnosticError, match="finite"):
        fingerprint_lfm2_fused_body_diagnostic_input(
            replace(mlp, compact_w1_weight=nonfinite)
        )
    aliased_encoder = torch.randn(_PARENT_HIDDEN, _COMPACT_HIDDEN)
    with pytest.raises(Lfm2FusedBodyDiagnosticError, match="aliased"):
        fingerprint_lfm2_fused_body_diagnostic_input(
            replace(
                mlp,
                input_encoder=aliased_encoder,
                output_decoder=aliased_encoder.T,
            )
        )

    conv = _conv_source()
    with pytest.raises(Lfm2FusedBodyDiagnosticError, match="conv_bias=False"):
        fingerprint_lfm2_fused_body_diagnostic_input(
            replace(conv, compact_conv_bias=torch.zeros(_COMPACT_HIDDEN))
        )
    module = _reconstruct(_compile_conv(conv)).module
    bad_hidden = torch.randn(1, 2, _PARENT_HIDDEN)
    bad_hidden[0, 0, 0] = float("inf")
    with (
        torch.inference_mode(),
        pytest.raises(
            Lfm2FusedBodyDiagnosticError,
            match="finite detached FP32",
        ),
    ):
        module(bad_hidden)


def test_external_pins_and_candidate_mutation_fail_closed() -> None:
    source = _mlp_source()
    bundle, pins = _pins(source)
    bad_pins = dict(pins)
    bad_pins["compact.w1.weight"] = "0" * 64
    with pytest.raises(Lfm2FusedBodyDiagnosticError, match="independent pins"):
        compile_lfm2_fused_mlp_diagnostic(
            source,
            expected_input_bundle_sha256=bundle,
            expected_tensor_sha256=bad_pins,
        )
    candidate = _compile_mlp(source)
    candidate.tensors["w1.weight"][0, 0] += 1.0
    with pytest.raises(Lfm2FusedBodyDiagnosticError, match="product lineage"):
        authenticate_lfm2_fused_body_candidate(candidate)
