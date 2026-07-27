from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_protected_short_conv_diagnostic import (
    Lfm2ProtectedShortConvDiagnosticError,
    Lfm2ProtectedShortConvInput,
    authenticate_lfm2_protected_short_conv_candidate,
    compile_lfm2_protected_short_conv_diagnostic,
    fingerprint_lfm2_protected_short_conv_input,
    lfm2_protected_short_conv_parameter_accounting,
    reconstruct_lfm2_protected_short_conv_candidate,
)


_PARENT = 8
_CHILD = 4
_KERNEL = 3
_LAYER = 1
_PARENT_VARIANTS = (
    "parent-input-roundtrip",
    "parent-output-roundtrip",
    "parent-both-roundtrip",
)
_ALL_VARIANTS = (*_PARENT_VARIANTS, "standalone-child-stem")


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=dtype)


def _source() -> Lfm2ProtectedShortConvInput:
    generator = torch.Generator(device="cpu").manual_seed(2707)
    return Lfm2ProtectedShortConvInput(
        layer_index=_LAYER,
        parent_hidden_size=_PARENT,
        child_hidden_size=_CHILD,
        kernel_size=_KERNEL,
        parent_conv_weight=_randn((_PARENT, 1, _KERNEL), generator=generator),
        parent_in_proj_weight=_randn((3 * _PARENT, _PARENT), generator=generator),
        parent_out_proj_weight=_randn((_PARENT, _PARENT), generator=generator),
        input_encoder=_randn((_PARENT, _CHILD), generator=generator),
        output_decoder=_randn((_CHILD, _PARENT), generator=generator),
    )


def _pins(source: Lfm2ProtectedShortConvInput) -> tuple[str, dict[str, str]]:
    authentication = fingerprint_lfm2_protected_short_conv_input(source)
    return authentication["input_bundle_sha256"], {
        name: row["sha256"] for name, row in authentication["tensor_records"].items()
    }


def _compile(source: Lfm2ProtectedShortConvInput, variant: str) -> Any:
    bundle, pins = _pins(source)
    return compile_lfm2_protected_short_conv_diagnostic(
        source,
        variant=variant,
        expected_input_bundle_sha256=bundle,
        expected_tensor_sha256=pins,
    )


def _reconstruct(candidate: Any) -> Any:
    return reconstruct_lfm2_protected_short_conv_candidate(
        candidate=candidate,
        expected_candidate_result_sha256=candidate.receipt["result_sha256"],
        expected_descriptor_sha256=(
            candidate.reconstruction_descriptor["descriptor_sha256"]
        ),
        device="cpu",
    )


def _short_conv_fp64(
    hidden_states: torch.Tensor,
    *,
    in_proj: torch.Tensor,
    conv: torch.Tensor,
    out_proj: torch.Tensor,
) -> torch.Tensor:
    bcx = torch.nn.functional.linear(hidden_states, in_proj).transpose(-1, -2)
    b, c, x = bcx.chunk(3, dim=-2)
    bx = b * x
    convolved = torch.nn.functional.conv1d(
        bx,
        conv,
        groups=int(conv.shape[0]),
        padding=int(conv.shape[-1]) - 1,
    )[..., : hidden_states.shape[1]]
    return torch.nn.functional.linear(
        (c * convolved).transpose(-1, -2).contiguous(), out_proj
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


@pytest.mark.parametrize("variant", _ALL_VARIANTS)
def test_closed_form_products_are_fp64_solved_and_fp32_materialized(
    variant: str,
) -> None:
    source = _source()
    candidate = _compile(source, variant)
    receipt = authenticate_lfm2_protected_short_conv_candidate(candidate)
    state = candidate.state_dict()
    e = source.input_encoder.double()
    d = source.output_decoder.double()
    # The authenticated round-trip is itself one explicit FP64 -> FP32
    # materialization and that sealed FP32 tensor is the authority for the
    # following projection product.
    p = (e @ d).float().double()
    parent_in = source.parent_in_proj_weight.double()
    parent_out = source.parent_out_proj_weight.double()
    expected_in = {
        "parent-input-roundtrip": parent_in @ p.T,
        "parent-output-roundtrip": parent_in,
        "parent-both-roundtrip": parent_in @ p.T,
        "standalone-child-stem": parent_in @ d.T,
    }[variant].float()
    expected_out = {
        "parent-input-roundtrip": parent_out,
        "parent-output-roundtrip": p.T @ parent_out,
        "parent-both-roundtrip": p.T @ parent_out,
        "standalone-child-stem": e.T @ parent_out,
    }[variant].float()
    torch.testing.assert_close(state["in_proj.weight"], expected_in, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        state["out_proj.weight"], expected_out, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        state["conv.weight"], source.parent_conv_weight, rtol=0.0, atol=0.0
    )
    assert receipt["residual_roundtrip_lineage"]["solve_dtype"] == "float64"
    for name in ("in_proj.weight", "out_proj.weight"):
        lineage = receipt["product_lineage"][name]
        if not lineage["formula"].startswith("identity-copy"):
            assert lineage["solve_dtype"] == "float64"
            assert lineage["materialization_dtype"] == "float32"
    assert (
        receipt["interpretation"]["parent_channelwise_multiplicative_paths_protected"]
        is True
    )
    assert receipt["operation_audit"]["internal_channels_averaged"] == 0
    assert receipt["operation_audit"]["internal_channels_removed"] == 0


@pytest.mark.parametrize("variant", _PARENT_VARIANTS)
def test_parent_shell_input_output_and_both_roundtrips_are_algebraic(
    variant: str,
) -> None:
    source = _source()
    generator = torch.Generator(device="cpu").manual_seed(2801)
    hidden = _randn((2, 6, _PARENT), generator=generator, dtype=torch.float64)
    e = source.input_encoder.double()
    d = source.output_decoder.double()
    p = e @ d
    win = source.parent_in_proj_weight.double()
    conv = source.parent_conv_weight.double()
    wout = source.parent_out_proj_weight.double()
    explicit_input = hidden @ p if variant != "parent-output-roundtrip" else hidden
    explicit = _short_conv_fp64(explicit_input, in_proj=win, conv=conv, out_proj=wout)
    if variant != "parent-input-roundtrip":
        explicit = explicit @ p
    folded_in = win @ p.T if variant != "parent-output-roundtrip" else win
    folded_out = p.T @ wout if variant != "parent-input-roundtrip" else wout
    folded = _short_conv_fp64(
        hidden,
        in_proj=folded_in,
        conv=conv,
        out_proj=folded_out,
    )
    torch.testing.assert_close(folded, explicit, rtol=1e-11, atol=1e-11)


def test_standalone_child_stem_is_exact_external_sandwich_in_fp64() -> None:
    source = _source()
    generator = torch.Generator(device="cpu").manual_seed(2903)
    child_hidden = _randn((2, 7, _CHILD), generator=generator, dtype=torch.float64)
    e = source.input_encoder.double()
    d = source.output_decoder.double()
    win = source.parent_in_proj_weight.double()
    conv = source.parent_conv_weight.double()
    wout = source.parent_out_proj_weight.double()
    explicit = (
        _short_conv_fp64(
            child_hidden @ d,
            in_proj=win,
            conv=conv,
            out_proj=wout,
        )
        @ e
    )
    folded = _short_conv_fp64(
        child_hidden,
        in_proj=win @ d.T,
        conv=conv,
        out_proj=e.T @ wout,
    )
    torch.testing.assert_close(folded, explicit, rtol=1e-11, atol=1e-11)


def test_standalone_dense_parameter_formula_and_actual_exact4x_geometry() -> None:
    accounting = lfm2_protected_short_conv_parameter_accounting(
        parent_hidden_size=1024,
        child_hidden_size=776,
        kernel_size=3,
        variant="standalone-child-stem",
    )
    expected = 4 * 776 * 1024 + 3 * 1024
    assert expected == 3_181_568
    assert accounting["candidate_parameter_elements"] == expected
    assert accounting["candidate_in_proj_parameter_elements"] == 3 * 776 * 1024
    assert accounting["candidate_out_proj_parameter_elements"] == 776 * 1024
    assert accounting["candidate_depthwise_conv_parameter_elements"] == 3 * 1024
    assert accounting["protected_parent_internal_channel_count"] == 1024
    assert accounting["removed_parent_internal_channel_count"] == 0
    assert accounting["runtime_sidecar_parameter_elements"] == 0
    for variant in _PARENT_VARIANTS:
        parent_shell = lfm2_protected_short_conv_parameter_accounting(
            parent_hidden_size=1024,
            child_hidden_size=776,
            kernel_size=3,
            variant=variant,
        )
        assert parent_shell["candidate_parameter_elements"] == 4_197_376
        assert parent_shell["physical_parameter_reduction_from_parent"] == 0


def _child_cache_config() -> Any:
    transformers = pytest.importorskip("transformers")
    return transformers.Lfm2Config(
        vocab_size=32,
        hidden_size=_CHILD,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=_CHILD,
        layer_types=["full_attention", "conv"],
        conv_L_cache=_KERNEL,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
    )


def test_standalone_runtime_cache_on_off_parity_uses_parent_internal_width() -> None:
    from transformers.cache_utils import DynamicCache

    candidate = _compile(_source(), "standalone-child-stem")
    full_module = _reconstruct(candidate).module
    cached_module = _reconstruct(candidate).module
    cache = DynamicCache(config=_child_cache_config())
    generator = torch.Generator(device="cpu").manual_seed(3001)
    hidden = _randn((2, 7, _CHILD), generator=generator)
    with torch.inference_mode():
        full = full_module(hidden)
        pieces = [cached_module(hidden[:, :4], past_key_values=cache)]
        for index in range(4, hidden.shape[1]):
            pieces.append(
                cached_module(hidden[:, index : index + 1], past_key_values=cache)
            )
        cached = torch.cat(pieces, dim=1)
    torch.testing.assert_close(cached, full, rtol=2e-5, atol=2e-5)
    assert tuple(cache.layers[_LAYER].conv_states.shape) == (
        2,
        _PARENT,
        _KERNEL,
    )
    assert cached_module.external_hidden_size == _CHILD
    assert cached_module.internal_hidden_size == _PARENT


@pytest.mark.parametrize("variant", ("parent-both-roundtrip", "standalone-child-stem"))
def test_fresh_reconstruction_is_bridge_free_alias_free_and_deterministic(
    variant: str,
) -> None:
    candidate = _compile(_source(), variant)
    first = _reconstruct(candidate)
    second = _reconstruct(candidate)
    accounting = candidate.receipt["physical_parameter_accounting"]
    assert (
        first.receipt["parameter_elements"]
        == accounting["candidate_parameter_elements"]
    )
    assert (
        first.receipt["stored_tensor_elements"]
        == accounting["candidate_stored_tensor_elements"]
    )
    assert first.receipt["runtime_parameter_alias_group_count"] == 0
    assert _alias_group_count(first.module) == 0
    assert not hasattr(first.module, "input_bridge")
    assert not hasattr(first.module, "output_bridge")
    assert {name for name, _ in first.module.named_parameters()} == {
        "conv.weight",
        "in_proj.weight",
        "out_proj.weight",
    }
    assert all(not value.requires_grad for value in first.module.parameters())
    for name, tensor in first.module.state_dict().items():
        torch.testing.assert_close(
            tensor, candidate.state_dict()[name], rtol=0.0, atol=0.0
        )
        assert (
            tensor.untyped_storage().data_ptr()
            != second.module.state_dict()[name].untyped_storage().data_ptr()
        )
    external = _CHILD if variant == "standalone-child-stem" else _PARENT
    hidden = torch.randn(2, 5, external)
    with torch.inference_mode():
        first_output = first.module(hidden)
        second_output = second.module(hidden)
    torch.testing.assert_close(first_output, second_output, rtol=0.0, atol=0.0)


def test_source_and_lineage_authentication_fail_closed() -> None:
    source = _source()
    bundle, pins = _pins(source)
    wrong_pins = dict(pins)
    wrong_pins["bridge.input_encoder"] = "0" * 64
    with pytest.raises(Lfm2ProtectedShortConvDiagnosticError, match="independent pins"):
        compile_lfm2_protected_short_conv_diagnostic(
            source,
            variant="standalone-child-stem",
            expected_input_bundle_sha256=bundle,
            expected_tensor_sha256=wrong_pins,
        )
    with pytest.raises(Lfm2ProtectedShortConvDiagnosticError, match="FP32"):
        fingerprint_lfm2_protected_short_conv_input(
            replace(source, input_encoder=source.input_encoder.double())
        )
    shared = torch.randn(_PARENT, _PARENT)
    with pytest.raises(Lfm2ProtectedShortConvDiagnosticError, match="aliased"):
        fingerprint_lfm2_protected_short_conv_input(
            replace(
                source,
                parent_out_proj_weight=shared,
                output_decoder=shared[:_CHILD],
            )
        )

    candidate = _compile(source, "parent-both-roundtrip")
    receipt = copy.deepcopy(dict(candidate.receipt))
    receipt["product_lineage"]["conv.weight"]["formula"] = "forged"
    body = dict(receipt)
    body.pop("result_sha256")
    receipt["result_sha256"] = canonical_sha256(body)
    forged = replace(candidate, receipt=receipt)
    with pytest.raises(Lfm2ProtectedShortConvDiagnosticError, match="product lineage"):
        authenticate_lfm2_protected_short_conv_candidate(forged)


def test_operation_receipt_proves_no_training_quantization_pruning_or_sharing() -> None:
    candidate = _compile(_source(), "standalone-child-stem")
    audit = authenticate_lfm2_protected_short_conv_candidate(candidate)[
        "operation_audit"
    ]
    assert audit["training_steps"] == 0
    assert audit["optimizer_constructions"] == 0
    assert audit["optimizer_steps"] == 0
    assert audit["autograd_grad_calls"] == 0
    assert audit["backward_calls"] == 0
    assert audit["parameters_pruned"] == 0
    assert audit["parameters_quantized"] == 0
    assert audit["parameters_sparsified"] == 0
    assert audit["runtime_parameter_aliases_created"] == 0
    assert audit["runtime_parameter_sharing_used"] is False
    source_text = Path(
        "src/neural_forge/hf_lfm2_protected_short_conv_diagnostic.py"
    ).read_text(encoding="utf-8")
    for forbidden_call in (
        "torch.optim.",
        ".backward(",
        "torch.autograd.grad(",
        "prune.",
        "quantize_",
    ):
        assert forbidden_call not in source_text
