from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

from neural_forge.dynamic_weight_complete_basis import (  # noqa: E402
    FeatureOrientation,
)
from neural_forge.hf_lfm2_weight_complete_basis import (  # noqa: E402
    Lfm2WeightCompleteBasisError,
    build_lfm2_weight_complete_basis,
    inventory_lfm2_weight_complete_operators,
)


_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"


class Lfm2RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, dtype=torch.float64))


class Lfm2MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate, bias=False, dtype=torch.float64)
        self.w3 = nn.Linear(hidden, intermediate, bias=False, dtype=torch.float64)
        self.w2 = nn.Linear(intermediate, hidden, bias=False, dtype=torch.float64)


class Lfm2Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            dtype=torch.float64,
        )
        self.q_layernorm = Lfm2RMSNorm(config.head_dim)
        self.k_layernorm = Lfm2RMSNorm(config.head_dim)


class Lfm2ShortConv(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias
        hidden = config.hidden_size
        self.in_proj = nn.Linear(
            hidden,
            3 * hidden,
            bias=config.conv_bias,
            dtype=torch.float64,
        )
        self.conv = nn.Conv1d(
            hidden,
            hidden,
            config.conv_L_cache,
            groups=hidden,
            padding=config.conv_L_cache - 1,
            bias=config.conv_bias,
            dtype=torch.float64,
        )
        self.out_proj = nn.Linear(
            hidden,
            hidden,
            bias=config.conv_bias,
            dtype=torch.float64,
        )


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int) -> None:
        super().__init__()
        self.is_attention_layer = (
            config.layer_types[layer_idx] == "full_attention"
        )
        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config, layer_idx)
        self.feed_forward = Lfm2MLP(
            config.hidden_size,
            config.intermediate_size,
        )
        self.operator_norm = Lfm2RMSNorm(config.hidden_size)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size)


for _class in (
    Lfm2RMSNorm,
    Lfm2MLP,
    Lfm2Attention,
    Lfm2ShortConv,
    Lfm2DecoderLayer,
):
    _class.__module__ = _LFM2_MODULE


class _Body(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=torch.float64,
        )
        self.layers = nn.ModuleList(
            [
                Lfm2DecoderLayer(config, layer_index)
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size)


class TinyLfm2(nn.Module):
    def __init__(
        self,
        *,
        tied: bool = True,
        conv_bias: bool = False,
    ) -> None:
        super().__init__()
        layer_types = [
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
        ]
        self.config = SimpleNamespace(
            model_type="lfm2",
            vocab_size=19,
            hidden_size=8,
            intermediate_size=11,
            num_hidden_layers=14,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            layer_types=layer_types,
            conv_L_cache=3,
            conv_bias=conv_bias,
            tie_word_embeddings=tied,
        )
        self.model = _Body(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
            dtype=torch.float64,
        )
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, *_args, **_kwargs):
        raise AssertionError("inventory must never execute the parent model")


def _operator_by_name(inventory, name: str):
    return next(value for value in inventory.operators if value.name == name)


def test_inventory_covers_all_fourteen_layers_and_counts_tied_weight_once() -> None:
    torch.manual_seed(117)
    parent = TinyLfm2(tied=True)

    inventory = inventory_lfm2_weight_complete_operators(parent)
    report = inventory.as_dict()
    coverage = report["coverage"]

    assert inventory.source_width == 8
    assert report["layer_count"] == 14
    assert report["layer_kinds"].count("full_attention") == 6
    assert report["layer_kinds"].count("short_conv") == 8
    assert report["operator_count"] == 120
    assert coverage == {
        "learned_parameter_path_count": 133,
        "unique_parameter_alias_group_count": 132,
        "residual_facing_alias_group_count": 120,
        "excluded_non_residual_alias_group_count": 12,
        "unclassified_alias_group_count": 0,
        "each_residual_facing_alias_group_represented_once": True,
        "each_learned_alias_group_classified_once": True,
        "tied_embedding_head": True,
        "tied_embedding_head_counted_once": True,
        "all_layers_structurally_inspected": True,
    }
    tied = [
        value
        for value in report["parameter_alias_groups"]
        if value["alias_count"] == 2
    ]
    assert len(tied) == 1
    assert tied[0]["paths"] == [
        "lm_head.weight",
        "model.embed_tokens.weight",
    ]
    assert tied[0]["operator_name"] == "lexical.tied_embedding_head.weight"
    assert sum(
        value.name == "lexical.tied_embedding_head.weight"
        for value in inventory.operators
    ) == 1

    source_groups = [
        tuple(value["parameter_paths"])
        for value in report["operator_sources"]
    ]
    assert len(source_groups) == len(set(source_groups)) == 120
    assert report["model_forward_calls"] == 0
    assert report["trace_capture_calls"] == 0
    assert report["training_steps"] == 0
    assert report["filesystem_writes"] == 0


def test_operator_orientations_match_lfm2_attention_conv_swiglu_and_norms() -> None:
    torch.manual_seed(23)
    inventory = inventory_lfm2_weight_complete_operators(TinyLfm2())

    query = _operator_by_name(inventory, "layer.002.attention.query.weight")
    attention_output = _operator_by_name(
        inventory,
        "layer.002.attention.output.weight",
    )
    conv_input = _operator_by_name(
        inventory,
        "layer.000.short_conv.in_proj.weight",
    )
    conv_output = _operator_by_name(
        inventory,
        "layer.000.short_conv.out_proj.weight",
    )
    fir = _operator_by_name(
        inventory,
        "layer.000.short_conv.fir.weight",
    )
    gate = _operator_by_name(inventory, "layer.000.swiglu.gate.weight")
    down = _operator_by_name(inventory, "layer.000.swiglu.down.weight")
    norm = _operator_by_name(inventory, "layer.000.operator_norm.weight")

    assert (query.orientation, query.feature_axis, query.tensor.shape) == (
        FeatureOrientation.INPUT,
        1,
        (8, 8),
    )
    assert (
        attention_output.orientation,
        attention_output.feature_axis,
        attention_output.tensor.shape,
    ) == (FeatureOrientation.OUTPUT, 0, (8, 8))
    assert (conv_input.orientation, conv_input.feature_axis) == (
        FeatureOrientation.INPUT,
        1,
    )
    assert (conv_output.orientation, conv_output.feature_axis) == (
        FeatureOrientation.OUTPUT,
        0,
    )
    assert (fir.orientation, fir.feature_axis, fir.tensor.shape) == (
        FeatureOrientation.OUTPUT,
        0,
        (8, 1, 3),
    )
    assert (gate.orientation, gate.feature_axis, gate.tensor.shape) == (
        FeatureOrientation.INPUT,
        1,
        (11, 8),
    )
    assert (down.orientation, down.feature_axis, down.tensor.shape) == (
        FeatureOrientation.OUTPUT,
        0,
        (8, 11),
    )
    assert norm.diagonal is True
    assert norm.source_width == 8

    excluded = [
        value
        for value in inventory.manifest["parameter_alias_groups"]
        if value["disposition"] == "excluded_non_residual_axis"
    ]
    assert len(excluded) == 12
    assert all(value["shape"] == [2] for value in excluded)
    assert all("head-local" in value["reason"] for value in excluded)


def test_biasful_short_conv_classifies_every_learned_bias_with_hidden_axis() -> None:
    torch.manual_seed(91)
    inventory = inventory_lfm2_weight_complete_operators(
        TinyLfm2(conv_bias=True)
    )
    report = inventory.as_dict()

    assert report["operator_count"] == 144
    assert report["coverage"]["residual_facing_alias_group_count"] == 144
    assert report["coverage"]["excluded_non_residual_alias_group_count"] == 12
    assert report["coverage"]["unique_parameter_alias_group_count"] == 156
    in_bias = _operator_by_name(
        inventory,
        "layer.000.short_conv.in_proj.bias",
    )
    fir_bias = _operator_by_name(
        inventory,
        "layer.000.short_conv.fir.bias",
    )
    out_bias = _operator_by_name(
        inventory,
        "layer.000.short_conv.out_proj.bias",
    )
    assert tuple(in_bias.tensor.shape) == (3, 8)
    assert in_bias.feature_axis == 1
    assert fir_bias.diagonal is True
    assert out_bias.diagonal is True


def test_basis_is_deterministic_and_activation_gram_remains_bounded() -> None:
    torch.manual_seed(508)
    parent = TinyLfm2()
    activation = torch.diag(
        torch.tensor(
            [1e8, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            dtype=torch.float64,
        )
    )

    first = build_lfm2_weight_complete_basis(
        parent,
        5,
        activation_gram=activation,
        activation_weight=0.2,
        maximum_activation_weight=0.25,
    )
    second = build_lfm2_weight_complete_basis(
        parent,
        5,
        activation_gram=activation,
        activation_weight=0.2,
        maximum_activation_weight=0.25,
    )

    assert torch.equal(first.basis, second.basis)
    assert first.report["run_sha256"] == second.report["run_sha256"]
    assert first.inventory.manifest["inventory_sha256"] == (
        second.inventory.manifest["inventory_sha256"]
    )
    assert first.source_width == 8
    assert first.target_width == 5
    assert first.map_sha256 == first.basis_result.map_sha256
    probe = torch.randn(3, 8, dtype=torch.float64)
    assert torch.equal(first(probe), first.encode(probe))
    assert torch.equal(
        first.materialize_dense(max_elements=40),
        first.basis,
    )
    assert first.report["basis"]["coverage_gate"]["passed"] is True
    supplement = first.report["bounded_activation_supplement"]
    assert supplement["supplied"] is True
    assert supplement["actual_weight"] == pytest.approx(0.2)
    assert supplement["structural_objective_weight"] == pytest.approx(0.8)
    assert supplement["activation_cannot_replace_structural_objective"] is True
    assert first.report["model_forward_calls"] == 0
    assert first.report["training_steps"] == 0


def test_inventory_fails_closed_for_unknown_or_unintended_aliases() -> None:
    parent = TinyLfm2()
    parent.extra = nn.Parameter(torch.ones(8, dtype=torch.float64))
    with pytest.raises(
        Lfm2WeightCompleteBasisError,
        match="Unclassified",
    ):
        inventory_lfm2_weight_complete_operators(parent)

    parent = TinyLfm2(tied=False)
    parent.model.layers[0].feed_forward.w3.weight = (
        parent.model.layers[0].feed_forward.w1.weight
    )
    with pytest.raises(
        Lfm2WeightCompleteBasisError,
        match="Unexpected parameter aliases",
    ):
        inventory_lfm2_weight_complete_operators(parent)


def test_adapter_source_contains_no_execution_training_or_write_calls() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "neural_forge"
        / "hf_lfm2_weight_complete_basis.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }

    assert "forward" not in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes
    assert "save" not in called_attributes
    assert "write_text" not in called_attributes
    assert "write_bytes" not in called_attributes
    assert "open" not in {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
