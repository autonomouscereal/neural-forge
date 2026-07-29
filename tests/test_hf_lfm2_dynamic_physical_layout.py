from __future__ import annotations

import hashlib
import json

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

from neural_forge.hf_lfm2_dynamic_physical_layout import (  # noqa: E402
    Lfm2DynamicPhysicalLayoutError,
    Lfm2HeterogeneousLayerPhysicalLayout,
    Lfm2HeterogeneousPhysicalLayoutDescriptor,
    Lfm2RaggedSwiGLURuntime,
    authenticate_lfm2_dynamic_physical_layout,
    authenticate_lfm2_physical_materialization,
    build_canonical_lfm2_exact4x_protected_stem_proposal,
    build_lfm2_dynamic_physical_layout,
    build_lfm2_heterogeneous_physical_layout,
    lfm2_expected_heterogeneous_physical_state_shapes,
    lfm2_expected_physical_state_shapes,
    restore_lfm2_heterogeneous_physical_layout,
)


def _tiny_descriptor(
    *,
    budget: int = 1_200,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    pad_token_id: int = 0,
):
    return build_lfm2_dynamic_physical_layout(
        hidden_size=8,
        vocab_size=16,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        block_auto_adjust_ff_dim=False,
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
        self.requires_grad_(False)
        self.eval()


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
    assert descriptor.norm_eps == 1e-5
    assert descriptor.max_position_embeddings == 128_000
    assert descriptor.use_cache is True
    assert descriptor.attention_implementation == "sdpa"
    assert (descriptor.bos_token_id, descriptor.eos_token_id) == (1, 7)
    assert descriptor.pad_token_id == 0
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
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
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


def test_materialization_audit_rejects_training_gradients_and_tensor_subclasses() -> (
    None
):
    descriptor = _tiny_descriptor()

    training = _TinyModel()
    training.train()
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="eval mode"):
        _materialization_audit(training, descriptor)

    gradients = _TinyModel()
    gradients.lm_head.weight.requires_grad_(True)
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="dense FP32"):
        _materialization_audit(gradients, descriptor)

    class EvilParameter(nn.Parameter):
        pass

    executable_parameter = _TinyModel()
    executable_parameter.lm_head.weight = EvilParameter(
        executable_parameter.lm_head.weight.detach().clone(),
        requires_grad=False,
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="dense FP32"):
        _materialization_audit(executable_parameter, descriptor)

    class EvilTensor(torch.Tensor):
        pass

    executable_buffer = _TinyModel()
    original = executable_buffer.model.rotary_emb.inv_freq.detach().clone()
    executable_buffer.model.rotary_emb._buffers["inv_freq"] = (
        torch.Tensor._make_subclass(EvilTensor, original, False)
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="rotary buffer"):
        _materialization_audit(executable_buffer, descriptor)


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
            norm_eps=1e-5,
            max_position_embeddings=128_000,
            use_cache=True,
            attention_implementation="sdpa",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            block_auto_adjust_ff_dim=False,
            layer_types=("conv", "full_attention"),
            mlp_intermediate_sizes=(5, 6),
            short_conv_internal_widths=(10, None),
            parent_parameter_elements=2_000,
            requested_dense_parameter_budget=1_200,
        )


@pytest.mark.parametrize(
    "token_field",
    ("bos_token_id", "eos_token_id", "pad_token_id"),
)
def test_special_token_ids_must_fit_the_physical_vocabulary(
    token_field: str,
) -> None:
    arguments = {token_field: 16}
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="smaller than vocab_size"):
        _tiny_descriptor(**arguments)


def _fixture_digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _heterogeneous_runtime_semantics() -> dict[str, object]:
    return {
        "rope_theta": 1_000_000.0,
        "norm_eps": 1e-5,
        "max_position_embeddings": 128_000,
        "use_cache": True,
        "attention_implementation": "sdpa",
        "bos_token_id": 1,
        "eos_token_id": 7,
        "pad_token_id": 0,
        "block_auto_adjust_ff_dim": False,
    }


def _exact4x_heterogeneous_fixture():
    """Build one evidence-selected feasibility fixture, not an allocation policy."""

    schedule = (
        "conv",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    )
    # These widths are evidence-selected test fixtures.  The module itself has
    # no preferred layers or widths: middle C sums to 4,384, three attention
    # layers use Q16/KV8, KV groups sum to 42, and all I values sum to 8,246.
    middle_conv_widths = iter((731, 731, 731, 731, 730, 730))
    attention_index = 0
    layers = []
    for index, kind in enumerate(schedule):
        evidence = _fixture_digest(f"evidence-selected-layer-{index}")
        if kind == "conv":
            endpoint = index in {0, len(schedule) - 1}
            layers.append(
                Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
                    layer_index=index,
                    mlp_intermediate_size=589,
                    internal_width=1024 if endpoint else next(middle_conv_widths),
                    writer_rank=97 if endpoint else None,
                    selection_evidence_sha256=evidence,
                )
            )
        else:
            high_capacity = attention_index < 3
            layers.append(
                Lfm2HeterogeneousLayerPhysicalLayout.attention(
                    layer_index=index,
                    mlp_intermediate_size=589,
                    query_heads=16 if high_capacity else 12,
                    key_value_heads=8 if high_capacity else 6,
                    head_dim=64,
                    selection_evidence_sha256=evidence,
                )
            )
            attention_index += 1
    return build_lfm2_heterogeneous_physical_layout(
        hidden_size=784,
        vocab_size=4096,
        conv_kernel_size=3,
        **_heterogeneous_runtime_semantics(),
        layers=layers,
        source_topology_sha256=_fixture_digest("source-topology"),
        allocation_evidence_sha256=_fixture_digest("allocation-evidence"),
        token_bridge_evidence_sha256=_fixture_digest("token-bridge-evidence"),
        expected_unique_parameter_count=57_420_144,
    )


def test_evidence_selected_heterogeneous_exact4x_feasibility_witness() -> None:
    descriptor = _exact4x_heterogeneous_fixture()
    accounting = descriptor.accounting
    conv_layers = [layer for layer in descriptor.layers if layer.layer_type == "conv"]
    attention_layers = [
        layer for layer in descriptor.layers if layer.layer_type == "full_attention"
    ]

    assert descriptor.hidden_size == 784
    assert descriptor.vocab_size == 4096
    assert descriptor.layer_count == 14
    assert descriptor.rope_theta == 1_000_000.0
    assert descriptor.norm_eps == 1e-5
    assert descriptor.max_position_embeddings == 128_000
    assert descriptor.use_cache is True
    assert descriptor.attention_implementation == "sdpa"
    assert (
        descriptor.bos_token_id,
        descriptor.eos_token_id,
        descriptor.pad_token_id,
    ) == (1, 7, 0)
    assert descriptor.block_auto_adjust_ff_dim is False
    assert sum(layer.mlp_intermediate_size for layer in descriptor.layers) == 8246
    assert conv_layers[0].short_conv_internal_width == 1024
    assert conv_layers[-1].short_conv_internal_width == 1024
    assert conv_layers[0].short_conv_writer_rank == 97
    assert conv_layers[-1].short_conv_writer_rank == 97
    assert sum(
        int(layer.short_conv_internal_width) for layer in conv_layers[1:-1]
    ) == 4384
    assert sum(int(layer.key_value_heads) for layer in attention_layers) == 42
    assert sum(
        (layer.query_heads, layer.key_value_heads) == (16, 8)
        for layer in attention_layers
    ) == 3

    assert accounting.input_embedding_parameter_elements == 3_211_264
    assert accounting.output_readout_parameter_elements == 3_211_264
    assert accounting.per_layer_residual_norm_parameter_elements == 21_952
    assert accounting.final_norm_parameter_elements == 784
    assert accounting.mlp_parameter_elements == 19_394_592
    assert accounting.attention_projection_parameter_elements == 12_644_352
    assert accounting.attention_qk_norm_parameter_elements == 768
    assert accounting.short_conv_input_projection_parameter_elements == 15_128_064
    assert accounting.short_conv_fir_parameter_elements == 19_296
    assert accounting.short_conv_dense_writer_parameter_elements == 3_437_056
    assert accounting.short_conv_factor_writer_parameter_elements == 350_752
    assert accounting.unique_parameter_elements == 57_420_144
    assert accounting.logical_parameter_elements == 57_420_144
    assert accounting.parameter_bytes_fp32 == 229_680_576
    shapes = lfm2_expected_heterogeneous_physical_state_shapes(descriptor)
    assert shapes["model.layers.0.conv.out_proj.0.weight"] == (97, 1024)
    assert shapes["model.layers.0.conv.out_proj.1.weight"] == (784, 97)
    assert shapes["model.layers.1.conv.out_proj.weight"] == (784, 731)
    assert shapes["model.layers.2.self_attn.q_proj.weight"] == (1024, 784)
    assert shapes["model.layers.8.self_attn.q_proj.weight"] == (768, 784)
    assert sum(torch.tensor(shape).prod().item() for shape in shapes.values()) == (
        57_420_144
    )


def test_heterogeneous_descriptor_is_canonical_serializable_and_restorable() -> None:
    descriptor = _exact4x_heterogeneous_fixture()
    serialized = descriptor.serialize()

    assert serialized == descriptor.serialize()
    assert _exact4x_heterogeneous_fixture().serialize() == serialized
    assert b" " not in serialized
    assert restore_lfm2_heterogeneous_physical_layout(serialized) == descriptor
    assert (
        Lfm2HeterogeneousPhysicalLayoutDescriptor.restore(
            descriptor.as_dict(),
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )
        == descriptor
    )
    assert (
        json.dumps(
            descriptor.as_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        == serialized
    )


def test_heterogeneous_descriptor_rejects_tamper_and_stale_evidence_binding() -> None:
    descriptor = _exact4x_heterogeneous_fixture()
    unsealed = descriptor.as_dict()
    unsealed["layers"][1]["short_conv_internal_width"] += 1
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="seal"):
        restore_lfm2_heterogeneous_physical_layout(unsealed)

    stale_binding = descriptor.as_dict()
    stale_binding["layers"][2]["selection_evidence_sha256"] = _fixture_digest(
        "different-layer-evidence"
    )
    stale_binding = _reseal(stale_binding)
    with pytest.raises(
        Lfm2DynamicPhysicalLayoutError,
        match="canonical reconstruction|binding",
    ):
        restore_lfm2_heterogeneous_physical_layout(stale_binding)

    # Q16/KV8 and Q18/KV6 have the same total projected width at D64, so this
    # changes the selected attention geometry without changing any accounting
    # row.  The inner evidence binding must still reject the stale selection.
    count_preserving_substitution = descriptor.as_dict()
    count_preserving_substitution["layers"][2]["query_heads"] = 18
    count_preserving_substitution["layers"][2]["key_value_heads"] = 6
    count_preserving_substitution = _reseal(count_preserving_substitution)
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="evidence binding"):
        restore_lfm2_heterogeneous_physical_layout(
            count_preserving_substitution
        )

    false_count = descriptor.as_dict()
    false_count["parameter_accounting"]["unique_parameter_elements"] += 1
    false_count = _reseal(false_count)
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="expected.*count"):
        restore_lfm2_heterogeneous_physical_layout(false_count)

    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="canonical JSON"):
        restore_lfm2_heterogeneous_physical_layout(
            descriptor.serialize().decode("ascii") + "\n"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runtime_parameter_alias_group_count", 1),
        ("cross_layer_parameter_sharing", True),
        ("runtime_bridge_parameter_elements", 1),
        ("runtime_sidecar_parameter_elements", 1),
        ("quantization_kind", "int8"),
        ("sparse_parameter_storage", True),
        ("pruning_masks_present", True),
    ),
)
def test_heterogeneous_resealed_unsafe_materialization_contract_fails_closed(
    field: str,
    value,
) -> None:
    serialized = _exact4x_heterogeneous_fixture().as_dict()
    serialized["materialization_contract"][field] = value
    serialized = _reseal(serialized)

    with pytest.raises(
        Lfm2DynamicPhysicalLayoutError,
        match="materialization contract",
    ):
        restore_lfm2_heterogeneous_physical_layout(serialized)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("topology", "rope_parameters", "rope_theta"), 0.0, "rope_theta"),
        (("runtime_semantics", "norm_eps"), 0.0, "norm_eps"),
        (
            ("runtime_semantics", "max_position_embeddings"),
            0,
            "max_position_embeddings",
        ),
        (("runtime_semantics", "use_cache"), 1, "use_cache"),
        (
            ("runtime_semantics", "attention_implementation"),
            "eager",
            "SDPA",
        ),
        (("runtime_semantics", "bos_token_id"), 4096, "smaller than"),
        (("runtime_semantics", "eos_token_id"), 4096, "smaller than"),
        (("runtime_semantics", "pad_token_id"), 4096, "smaller than"),
        (
            ("runtime_semantics", "block_auto_adjust_ff_dim"),
            True,
            "block_auto_adjust_ff_dim=False",
        ),
    ),
)
def test_heterogeneous_resealed_invalid_runtime_semantics_fail_closed(
    path: tuple[str, ...],
    value,
    message: str,
) -> None:
    serialized = _exact4x_heterogeneous_fixture().as_dict()
    target = serialized
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    serialized = _reseal(serialized)

    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match=message):
        restore_lfm2_heterogeneous_physical_layout(serialized)


def test_heterogeneous_runtime_semantics_are_bound_by_external_descriptor_seal() -> (
    None
):
    descriptor = _exact4x_heterogeneous_fixture()
    substituted = descriptor.as_dict()
    substituted["runtime_semantics"]["use_cache"] = False
    substituted = _reseal(substituted)

    replacement = restore_lfm2_heterogeneous_physical_layout(substituted)
    assert replacement.use_cache is False
    assert replacement.descriptor_sha256 != descriptor.descriptor_sha256
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="external seal"):
        restore_lfm2_heterogeneous_physical_layout(
            substituted,
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )


def test_heterogeneous_external_seal_rejects_fully_resealed_substitution() -> None:
    descriptor = _exact4x_heterogeneous_fixture()
    layers = list(descriptor.layers)
    layer = layers[2]
    layers[2] = Lfm2HeterogeneousLayerPhysicalLayout.attention(
        layer_index=layer.layer_index,
        mlp_intermediate_size=layer.mlp_intermediate_size,
        query_heads=int(layer.query_heads),
        key_value_heads=int(layer.key_value_heads),
        head_dim=int(layer.head_dim),
        selection_evidence_sha256=_fixture_digest("replacement-evidence"),
    )
    substitute = build_lfm2_heterogeneous_physical_layout(
        hidden_size=descriptor.hidden_size,
        vocab_size=descriptor.vocab_size,
        conv_kernel_size=descriptor.conv_kernel_size,
        **_heterogeneous_runtime_semantics(),
        layers=layers,
        source_topology_sha256=descriptor.source_topology_sha256,
        allocation_evidence_sha256=descriptor.allocation_evidence_sha256,
        token_bridge_evidence_sha256=descriptor.token_bridge_evidence_sha256,
    )

    assert substitute.descriptor_sha256 != descriptor.descriptor_sha256
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="external seal"):
        restore_lfm2_heterogeneous_physical_layout(
            substitute.serialize(),
            expected_descriptor_sha256=descriptor.descriptor_sha256,
        )


def test_heterogeneous_layout_has_no_fixed_layer_ids_or_operator_geometry() -> None:
    # Attention may lead the schedule, and a factor writer may occur at any
    # conv layer.  This guards against embedding the feasibility fixture's IDs.
    layers = (
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=0,
            mlp_intermediate_size=17,
            query_heads=6,
            key_value_heads=3,
            head_dim=4,
            selection_evidence_sha256=_fixture_digest("arbitrary-attention-0"),
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=1,
            mlp_intermediate_size=19,
            internal_width=13,
            selection_evidence_sha256=_fixture_digest("arbitrary-conv-1"),
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=2,
            mlp_intermediate_size=23,
            internal_width=14,
            writer_rank=3,
            selection_evidence_sha256=_fixture_digest("arbitrary-conv-2"),
        ),
    )
    descriptor = build_lfm2_heterogeneous_physical_layout(
        hidden_size=12,
        vocab_size=31,
        conv_kernel_size=5,
        **_heterogeneous_runtime_semantics(),
        layers=layers,
        source_topology_sha256=_fixture_digest("arbitrary-source"),
        allocation_evidence_sha256=_fixture_digest("arbitrary-allocation"),
        token_bridge_evidence_sha256=_fixture_digest("arbitrary-bridge"),
    )

    assert tuple(layer.layer_type for layer in descriptor.layers) == (
        "full_attention",
        "conv",
        "conv",
    )
    assert descriptor.layers[2].short_conv_writer_rank == 3
    assert descriptor.layers[0].query_heads == 6
    assert restore_lfm2_heterogeneous_physical_layout(descriptor.as_dict()) == (
        descriptor
    )


def test_heterogeneous_layout_strictly_rejects_invalid_geometry() -> None:
    digest = _fixture_digest("geometry")
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="divisible"):
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=0,
            mlp_intermediate_size=8,
            query_heads=5,
            key_value_heads=2,
            head_dim=4,
            selection_evidence_sha256=digest,
        )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="rank"):
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=0,
            mlp_intermediate_size=8,
            internal_width=4,
            writer_rank=5,
            selection_evidence_sha256=digest,
        )
    oversized_rank = Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
        layer_index=0,
        mlp_intermediate_size=8,
        internal_width=12,
        writer_rank=9,
        selection_evidence_sha256=digest,
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="matrix dimension"):
        build_lfm2_heterogeneous_physical_layout(
            hidden_size=8,
            vocab_size=16,
            conv_kernel_size=3,
            **_heterogeneous_runtime_semantics(),
            layers=(oversized_rank,),
            source_topology_sha256=_fixture_digest("strict-source"),
            allocation_evidence_sha256=_fixture_digest("strict-allocation"),
            token_bridge_evidence_sha256=_fixture_digest("strict-bridge"),
        )
    noncontiguous = Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
        layer_index=1,
        mlp_intermediate_size=8,
        internal_width=4,
        selection_evidence_sha256=digest,
    )
    with pytest.raises(Lfm2DynamicPhysicalLayoutError, match="contiguous"):
        build_lfm2_heterogeneous_physical_layout(
            hidden_size=8,
            vocab_size=16,
            conv_kernel_size=3,
            **_heterogeneous_runtime_semantics(),
            layers=(noncontiguous,),
            source_topology_sha256=_fixture_digest("strict-source"),
            allocation_evidence_sha256=_fixture_digest("strict-allocation"),
            token_bridge_evidence_sha256=_fixture_digest("strict-bridge"),
        )
