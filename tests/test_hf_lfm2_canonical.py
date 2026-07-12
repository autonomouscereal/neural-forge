from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

from neural_forge.hf_lfm2_canonical import (  # noqa: E402
    Lfm2CanonicalError,
    Lfm2UnsupportedLayoutError,
    build_lfm2_short_conv_channel_plan,
    inspect_lfm2_layer,
    lfm2_short_conv_reference,
)


_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"


class Lfm2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float64))


class Lfm2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float64)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float64)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float64)


class Lfm2ShortConv(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            config.conv_L_cache,
            groups=hidden_size,
            bias=config.conv_bias,
            padding=config.conv_L_cache - 1,
            dtype=torch.float64,
        )
        self.in_proj = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=config.conv_bias,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=config.conv_bias,
            dtype=torch.float64,
        )


class Lfm2Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.float64,
        )
        self.q_layernorm = Lfm2RMSNorm(self.head_dim)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim)


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config, layer_idx)
        self.feed_forward = Lfm2MLP(config.hidden_size, config.intermediate_size)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size)


for _class in (
    Lfm2RMSNorm,
    Lfm2MLP,
    Lfm2ShortConv,
    Lfm2Attention,
    Lfm2DecoderLayer,
):
    _class.__module__ = _LFM2_MODULE


def _config(
    hidden_size: int = 4,
    *,
    layer_types: list[str] | None = None,
    bias: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_type="lfm2",
        hidden_size=hidden_size,
        intermediate_size=7,
        conv_L_cache=3,
        conv_bias=bias,
        layer_types=layer_types or ["conv", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
    )


def _layer(hidden_size: int = 4, *, bias: bool = False) -> Lfm2DecoderLayer:
    torch.manual_seed(1234 + hidden_size + int(bias))
    return Lfm2DecoderLayer(_config(hidden_size, bias=bias), 0)


def test_inspection_distinguishes_bcx_short_conv_from_qkv_attention() -> None:
    config = _config()
    conv_contract = inspect_lfm2_layer(Lfm2DecoderLayer(config, 0))
    attention_contract = inspect_lfm2_layer(Lfm2DecoderLayer(config, 1))

    assert conv_contract.kind == "short_conv"
    assert conv_contract.operator.as_dict()["input_projection_chunk_order"] == [
        "b",
        "c",
        "x",
    ]
    assert conv_contract.operator.as_dict()["not_attention_qkv"] is True
    assert attention_contract.kind == "full_attention"
    assert attention_contract.operator.as_dict()["projection_roles"] == ["q", "k", "v"]
    json.dumps(conv_contract.as_dict())
    json.dumps(attention_contract.as_dict())


def test_reference_matches_transformers_slow_conv_storage_order() -> None:
    layer = _layer(3, bias=True)
    parameters = build_lfm2_short_conv_channel_plan(
        layer,
        target_hidden_size=3,
        selected_channels=(0, 1, 2),
    ).transplant(layer)
    inputs = torch.randn(2, 5, 3, dtype=torch.float64)

    projected = layer.conv.in_proj(inputs).transpose(1, 2)
    b_values, c_values, x_values = projected.chunk(3, dim=1)
    conv_values = layer.conv.conv(b_values * x_values)[..., : inputs.shape[1]]
    expected = layer.conv.out_proj((c_values * conv_values).transpose(1, 2))

    torch.testing.assert_close(
        lfm2_short_conv_reference(parameters, inputs),
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_full_channel_permutation_is_algebraically_exact() -> None:
    layer = _layer(3, bias=True)
    selected = (2, 0, 1)
    plan = build_lfm2_short_conv_channel_plan(
        layer,
        target_hidden_size=3,
        mode="subset",
        selected_channels=selected,
    )
    target = plan.transplant(layer)
    source = build_lfm2_short_conv_channel_plan(
        layer,
        target_hidden_size=3,
        selected_channels=(0, 1, 2),
    ).transplant(layer)
    inputs = torch.randn(3, 6, 3, dtype=torch.float64)

    source_output = lfm2_short_conv_reference(source, inputs)
    target_output = lfm2_short_conv_reference(target, inputs[..., list(selected)])

    assert plan.exactness.level == "algebraically_exact"
    assert plan.exactness.scope == "full-operator-coordinate-permutation"
    assert plan.invariants.all_hold is True
    torch.testing.assert_close(
        target_output,
        source_output[..., list(selected)],
        atol=1e-12,
        rtol=1e-12,
    )


def test_subset_copy_is_exact_only_for_a_closed_selected_subnetwork() -> None:
    layer = _layer(4, bias=True)
    selected = (2, 0)
    dropped = (1, 3)
    with torch.no_grad():
        for block in layer.conv.in_proj.weight.chunk(3, dim=0):
            block[list(selected)][:, list(dropped)] = 0.0
            for row in selected:
                block[row, list(dropped)] = 0.0
        for row in selected:
            layer.conv.out_proj.weight[row, list(dropped)] = 0.0

    plan = build_lfm2_short_conv_channel_plan(
        layer,
        target_hidden_size=2,
        mode="subset",
        selected_channels=selected,
    )
    target = plan.transplant(layer)
    source = build_lfm2_short_conv_channel_plan(
        layer,
        target_hidden_size=4,
        selected_channels=(0, 1, 2, 3),
    ).transplant(layer)
    inputs = torch.randn(2, 7, 4, dtype=torch.float64)

    assert plan.exactness.level == "algebraically_exact"
    assert plan.exactness.scope == "closed-selected-coordinate-subnetwork"
    torch.testing.assert_close(
        lfm2_short_conv_reference(target, inputs[..., list(selected)]),
        lfm2_short_conv_reference(source, inputs)[..., list(selected)],
        atol=1e-12,
        rtol=1e-12,
    )


def test_nonclosed_subset_and_true_merges_are_classified_approximate() -> None:
    layer = _layer(4)
    subset = build_lfm2_short_conv_channel_plan(
        layer,
        target_hidden_size=2,
        selected_channels=(0, 1),
    )
    first_merge = build_lfm2_short_conv_channel_plan(
        layer,
        target_hidden_size=2,
        mode="merge",
    )
    repeated_merge = build_lfm2_short_conv_channel_plan(
        layer,
        target_hidden_size=2,
        mode="merge",
    )

    assert subset.exactness.level == "approximate"
    assert first_merge == repeated_merge
    assert first_merge.exactness.level == "approximate"
    assert first_merge.exactness.channels_merged is True
    assert all(first_merge.target_groups)
    assert sorted(
        source for group in first_merge.target_groups for source in group
    ) == list(range(4))
    assert first_merge.invariants.bcx_kernel_output_indices_coupled is True
    assert first_merge.transplant(layer).in_proj_weight.shape == (6, 2)
    json.dumps(first_merge.as_dict())


def test_inspection_fails_closed_on_unsupported_or_mislabeled_layouts() -> None:
    wrong_groups = _layer(4)
    wrong_groups.conv.conv.groups = 1
    with pytest.raises(Lfm2UnsupportedLayoutError, match="depthwise"):
        inspect_lfm2_layer(wrong_groups)

    mislabeled = _layer(4)
    mislabeled.is_attention_layer = True
    with pytest.raises(Lfm2UnsupportedLayoutError, match="attributes"):
        inspect_lfm2_layer(mislabeled)

    wrong_family = _layer(4)
    wrong_family.conv.config.model_type = "not_lfm2"
    with pytest.raises(Lfm2UnsupportedLayoutError, match="model_type"):
        inspect_lfm2_layer(wrong_family)

    attention = Lfm2DecoderLayer(_config(), 1)
    with pytest.raises(Lfm2UnsupportedLayoutError, match="short-conv"):
        build_lfm2_short_conv_channel_plan(attention, target_hidden_size=2)


def test_plan_rejects_invalid_selection_and_changed_source_tensors() -> None:
    layer = _layer(3)
    with pytest.raises(Lfm2CanonicalError, match="unique"):
        build_lfm2_short_conv_channel_plan(
            layer,
            target_hidden_size=2,
            selected_channels=(0, 0),
        )
    plan = build_lfm2_short_conv_channel_plan(
        layer,
        target_hidden_size=2,
        selected_channels=(0, 1),
    )
    with torch.no_grad():
        layer.conv.conv.weight[0, 0, 0] += 1.0
    with pytest.raises(Lfm2CanonicalError, match="differ"):
        plan.transplant(layer)
