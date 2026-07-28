from __future__ import annotations

import hashlib
import json

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

from neural_forge.hf_lfm2_dynamic_physical_layout import (  # noqa: E402
    Lfm2DynamicPhysicalLayoutError,
    Lfm2RaggedSwiGLURuntime,
    authenticate_lfm2_dynamic_physical_layout,
    authenticate_lfm2_physical_materialization,
    build_canonical_lfm2_exact4x_protected_stem_proposal,
    build_lfm2_dynamic_physical_layout,
    lfm2_expected_physical_state_shapes,
)


def _tiny_descriptor(*, budget: int = 1_200):
    return build_lfm2_dynamic_physical_layout(
        hidden_size=8,
        vocab_size=16,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        layer_types=("conv", "full_attention"),
        mlp_intermediate_sizes=(5, 6),
        short_conv_internal_widths=(10, None),
        parent_parameter_elements=2_000,
        requested_dense_parameter_budget=budget,
    )


class _Norm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.float32))


class _ShortConv(nn.Module):
    def __init__(self, hidden: int, internal: int, kernel: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, 3 * internal, bias=False, dtype=torch.float32)
        self.conv = nn.Conv1d(
            internal,
            internal,
            kernel,
            groups=internal,
            bias=False,
            dtype=torch.float32,
        )
        self.out_proj = nn.Linear(internal, hidden, bias=False, dtype=torch.float32)


class _Attention(nn.Module):
    def __init__(self, hidden: int, query: int, key_value: int, head_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, query, bias=False, dtype=torch.float32)
        self.k_proj = nn.Linear(hidden, key_value, bias=False, dtype=torch.float32)
        self.v_proj = nn.Linear(hidden, key_value, bias=False, dtype=torch.float32)
        self.out_proj = nn.Linear(query, hidden, bias=False, dtype=torch.float32)
        self.q_layernorm = _Norm(head_dim)
        self.k_layernorm = _Norm(head_dim)


class _Rotary(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)


class _Layer(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.operator_norm = _Norm(8)
        self.ffn_norm = _Norm(8)
        self.feed_forward = Lfm2RaggedSwiGLURuntime(8, 5 + index)
        if index == 0:
            self.conv = _ShortConv(8, 10, 3)
        else:
            self.self_attn = _Attention(8, 8, 4, 4)


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8, dtype=torch.float32)
        self.rotary_emb = _Rotary(4, 1_000_000.0)
        self.layers = nn.ModuleList([_Layer(0), _Layer(1)])
        self.embedding_norm = _Norm(8)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Core()
        self.lm_head = nn.Linear(8, 16, bias=False, dtype=torch.float32)


def _reseal(value: dict) -> dict:
    body = dict(value)
    body.pop("descriptor_sha256", None)
    payload = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {**body, "descriptor_sha256": hashlib.sha256(payload).hexdigest()}


def _materialization_audit(model, descriptor):
    module_types = {
        name: f"{type(module).__module__}.{type(module).__qualname__}"
        for name, module in model.named_modules(remove_duplicate=False)
    }
    return authenticate_lfm2_physical_materialization(
        model,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
        expected_runtime_root_class=_TinyModel,
        expected_module_type_names=module_types,
    )


def test_canonical_protected_stem_proposal_has_exact_complete_count() -> None:
    descriptor = build_canonical_lfm2_exact4x_protected_stem_proposal()
    accounting = descriptor.accounting

    assert descriptor.hidden_size == 776
    assert tuple(layer.mlp_intermediate_size for layer in descriptor.layers) == (
        700,
        *(674 for _ in range(6)),
        *(675 for _ in range(7)),
    )
    assert descriptor.layers[0].short_conv_internal_width == 1024
    assert descriptor.rope_theta == 1_000_000.0
    assert descriptor.layers[0].cache_contract.as_dict() == {
        "cache_kind": "transformers-DynamicCache-only",
        "layer_index": 0,
        "external_hidden_size": 776,
        "internal_channel_width": 1024,
        "kernel_width": 3,
        "cache_state_shape": ["batch", 1024, 3],
        "static_or_compileable_cache_supported": False,
        "fresh_cache_required_per_materialization": True,
    }
    assert accounting.token_embedding_parameter_elements == 3_178_496
    assert accounting.readout_parameter_elements == 3_178_496
    assert accounting.residual_norm_parameter_elements == 22_504
    assert accounting.attention_projection_parameter_elements == 8_939_520
    assert accounting.attention_qk_norm_parameter_elements == 768
    assert accounting.short_conv_parameter_elements == 20_058_792
    assert accounting.mlp_parameter_elements == 22_043_832
    assert accounting.body_parameter_elements == 51_065_416
    assert accounting.total_parameter_elements == 57_422_408
    assert accounting.total_parameter_bytes_fp32 == 229_689_632
    assert accounting.parent_parameter_elements == 229_693_184
    assert accounting.requested_dense_parameter_budget == 57_423_296
    assert accounting.budget_headroom_parameter_elements == 888
    assert accounting.exact_compaction_ratio == pytest.approx(4.000061857385012)
    assert descriptor.as_dict()["allocation_policy_embedded"] is False

    shapes = lfm2_expected_physical_state_shapes(descriptor)
    assert len(shapes) == 133
    assert shapes["model.layers.0.conv.in_proj.weight"] == (3072, 776)
    assert shapes["model.layers.0.feed_forward.w1.weight"] == (700, 776)
    assert shapes["model.layers.1.feed_forward.w1.weight"] == (674, 776)
    assert shapes["model.layers.13.feed_forward.w2.weight"] == (776, 675)
    assert sum(torch.tensor(shape).prod().item() for shape in shapes.values()) == (
        57_422_408
    )
    assert (
        authenticate_lfm2_dynamic_physical_layout(
            descriptor.as_dict(),
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )
        == descriptor
    )


def test_arbitrary_per_layer_widths_are_accepted_but_budget_is_hard() -> None:
    descriptor = _tiny_descriptor()

    assert tuple(layer.mlp_intermediate_size for layer in descriptor.layers) == (5, 6)
    assert descriptor.accounting.total_parameter_elements == 1_110
    assert descriptor.accounting.budget_headroom_parameter_elements == 90

    other = build_lfm2_dynamic_physical_layout(
        hidden_size=8,
        vocab_size=16,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        layer_types=("conv", "full_attention"),
        mlp_intermediate_sizes=(7, 4),
        short_conv_internal_widths=(9, None),
        parent_parameter_elements=2_000,
        requested_dense_parameter_budget=1_200,
    )
    assert tuple(layer.mlp_intermediate_size for layer in other.layers) == (7, 4)
    assert other.layers[0].short_conv_internal_width == 9
    assert other.descriptor_sha256 != descriptor.descriptor_sha256

    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="exceeds"):
        _tiny_descriptor(budget=1_109)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runtime_parameter_alias_group_count", 1),
        ("cross_layer_parameter_sharing", True),
        ("quantization_kind", "int8"),
        ("sparse_parameter_storage", True),
        ("pruning_masks_present", True),
        ("runtime_sidecar_parameter_elements", 1),
    ),
)
def test_resealed_forbidden_storage_contracts_fail_closed(field: str, value) -> None:
    serialized = _tiny_descriptor().as_dict()
    serialized["materialization_contract"][field] = value
    tampered = _reseal(serialized)

    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="unshared FP32"):
        authenticate_lfm2_dynamic_physical_layout(tampered)


def test_ragged_mlp_keeps_native_state_keys_and_swiglu_action() -> None:
    module = Lfm2RaggedSwiGLURuntime(4, 3)
    with torch.no_grad():
        module.w1.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 20)
        module.w3.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 15)
        module.w2.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10)
    inputs = torch.tensor(
        [[[0.2, -0.5, 0.7, 0.3], [0.1, 0.4, -0.2, 0.8]]],
        dtype=torch.float32,
    )

    result = module(inputs)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(inputs, module.w1.weight))
        * torch.nn.functional.linear(inputs, module.w3.weight),
        module.w2.weight,
    )

    assert tuple(module.state_dict()) == ("w1.weight", "w3.weight", "w2.weight")
    assert sum(parameter.numel() for parameter in module.parameters()) == 36
    assert all(not parameter.requires_grad for parameter in module.parameters())
    torch.testing.assert_close(result, expected)


def test_complete_materialization_audit_accepts_only_exact_dense_unshared_state() -> (
    None
):
    descriptor = _tiny_descriptor()
    model = _TinyModel()

    receipt = _materialization_audit(model, descriptor)

    assert receipt["parameter_state_key_count"] == 22
    assert receipt["parameter_elements"] == 1_110
    assert receipt["parameter_bytes_fp32"] == 4_440
    assert receipt["intrinsic_nonpersistent_buffer_count"] == 2
    assert receipt["intrinsic_nonpersistent_buffer_elements"] == 4
    assert receipt["runtime_parameter_alias_group_count"] == 0
    assert receipt["runtime_sidecar_parameter_elements"] == 0


def test_materialization_audit_rejects_aliases_reduced_dtype_sparse_and_sidecars() -> (
    None
):
    descriptor = _tiny_descriptor()

    aliased = _TinyModel()
    aliased.lm_head.weight = aliased.model.embed_tokens.weight
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="aliases|sharing"):
        _materialization_audit(aliased, descriptor)

    reduced_dtype = _TinyModel()
    reduced_dtype.lm_head.weight = nn.Parameter(
        reduced_dtype.lm_head.weight.detach().to(torch.float16)
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="dense FP32"):
        _materialization_audit(reduced_dtype, descriptor)

    sparse = _TinyModel()
    sparse.lm_head.weight = nn.Parameter(
        sparse.lm_head.weight.detach().to_sparse(), requires_grad=False
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="dense FP32"):
        _materialization_audit(sparse, descriptor)

    sidecar = _TinyModel()
    sidecar.register_parameter(
        "unaccounted_sidecar", nn.Parameter(torch.ones(1, dtype=torch.float32))
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="extra"):
        _materialization_audit(sidecar, descriptor)

    masked = _TinyModel()
    masked.register_buffer("pruning_mask", torch.ones(1, dtype=torch.float32))
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="masks|sidecar"):
        _materialization_audit(masked, descriptor)

    hidden_sidecar = _TinyModel()
    hidden_sidecar.register_buffer(
        "hidden_parent_sidecar",
        torch.ones(100_000, dtype=torch.float32),
        persistent=False,
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="sidecar buffer"):
        _materialization_audit(hidden_sidecar, descriptor)

    expanded = _TinyModel()
    expanded.lm_head.weight = nn.Parameter(
        torch.ones(1, 8, dtype=torch.float32).expand(16, 8),
        requires_grad=False,
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="dense FP32"):
        _materialization_audit(expanded, descriptor)

    hidden_runtime = _TinyModel()
    hidden_runtime.hidden_runtime = [nn.Linear(1_000, 1_000, bias=False)]
    with pytest.raises(
        Lfm2DynamicPhysicalLayoutError,
        match="unregistered runtime state",
    ):
        _materialization_audit(hidden_runtime, descriptor)


def test_odd_head_dim_is_rejected_before_default_rope_materialization() -> None:
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="positive even"):
        build_lfm2_dynamic_physical_layout(
            hidden_size=8,
            vocab_size=16,
            head_dim=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            conv_kernel_size=3,
            rope_theta=1_000_000.0,
            layer_types=("conv", "full_attention"),
            mlp_intermediate_sizes=(5, 6),
            short_conv_internal_widths=(10, None),
            parent_parameter_elements=2_000,
            requested_dense_parameter_budget=1_200,
        )
