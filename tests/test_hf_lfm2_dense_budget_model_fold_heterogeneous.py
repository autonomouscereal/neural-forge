from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.dynamic_fold_executor import (  # noqa: E402
    fingerprint_model_state,
)
from neural_forge.hf_feature_fold import build_feature_fold_map  # noqa: E402
from neural_forge.hf_lfm2_dense_budget_model_fold import (  # noqa: E402
    Lfm2DenseBudgetFoldError,
    _fold_dense_blank_from_parent,
)
from neural_forge.hf_lfm2_dynamic_blank_materializer import (  # noqa: E402
    materialize_lfm2_dynamic_blank_child,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (  # noqa: E402
    Lfm2HeterogeneousLayerPhysicalLayout,
    build_lfm2_heterogeneous_physical_layout,
)


def _config(
    *,
    hidden: int,
    intermediate: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    layer_types: tuple[str, ...],
):
    config = transformers.Lfm2Config(
        vocab_size=23,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=len(layer_types),
        num_attention_heads=query_heads,
        num_key_value_heads=key_value_heads,
        head_dim=head_dim,
        layer_types=list(layer_types),
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "sdpa"
    return config


def _model(config, *, seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _zero(model) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()


def _residual_map(parent, target_width: int):
    return build_feature_fold_map(
        parent.get_input_embeddings().weight.detach(),
        target_width,
        source_width=int(parent.config.hidden_size),
        chunk_size=16,
        sketch_size=8,
        balance_strength=0.25,
        allow_sign_flip=False,
    )


def _heterogeneous_child(*, factorized_writer: bool = False):
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    layers = (
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=0,
            mlp_intermediate_size=7,
            internal_width=10,
            writer_rank=3 if factorized_writer else None,
            selection_evidence_sha256="0" * 64,
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=1,
            mlp_intermediate_size=8,
            query_heads=2,
            key_value_heads=1,
            head_dim=4,
            selection_evidence_sha256="1" * 64,
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.attention(
            layer_index=2,
            mlp_intermediate_size=9,
            query_heads=4,
            key_value_heads=2,
            head_dim=2,
            selection_evidence_sha256="2" * 64,
        ),
        Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
            layer_index=3,
            mlp_intermediate_size=10,
            internal_width=6,
            selection_evidence_sha256="3" * 64,
        ),
    )
    descriptor = build_lfm2_heterogeneous_physical_layout(
        hidden_size=8,
        vocab_size=23,
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
        layers=layers,
        source_topology_sha256="a" * 64,
        allocation_evidence_sha256="b" * 64,
        token_bridge_evidence_sha256="c" * 64,
    )
    child = _model(
        _config(
            hidden=8,
            intermediate=8,
            query_heads=2,
            key_value_heads=1,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4102,
    )
    materialize_lfm2_dynamic_blank_child(
        child,
        child.config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    _zero(child)
    return child


def test_fold_uses_each_materialized_layer_physical_geometry() -> None:
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4101,
    )
    child = _heterogeneous_child()
    parent_before = fingerprint_model_state(parent)

    report = _fold_dense_blank_from_parent(
        parent,
        child,
        _residual_map(parent, 8),
    )

    assert report["complete_target_parameter_overwrite"] is True
    assert report["training_used"] is False
    assert report["parent_forward_calls"] == 0
    assert fingerprint_model_state(parent) == parent_before
    assert [layer["swiglu"]["target_neurons"] for layer in report["layers"]] == [
        7,
        8,
        9,
        10,
    ]
    first_conv = report["layers"][0]["operator"]
    last_conv = report["layers"][3]["operator"]
    assert first_conv["source_internal_channels"] == 16
    assert first_conv["target_internal_channels"] == 10
    assert first_conv["internal_feature_fold"]["full_source_coverage"] is True
    assert last_conv["target_internal_channels"] == 6
    first_attention = report["layers"][1]["operator"]
    second_attention = report["layers"][2]["operator"]
    assert first_attention["target_physical_geometry"] == {
        "query_heads": 2,
        "key_value_heads": 1,
        "head_dim": 4,
    }
    assert second_attention["target_physical_geometry"] == {
        "query_heads": 4,
        "key_value_heads": 2,
        "head_dim": 2,
    }
    assert first_attention["query"]["target_heads"] == 2
    assert first_attention["query"]["target_head_dim"] == 4
    assert second_attention["query"]["target_heads"] == 4
    assert second_attention["query"]["target_head_dim"] == 2
    assert tuple(child.model.layers[0].conv.in_proj.weight.shape) == (30, 8)
    assert tuple(child.model.layers[3].conv.out_proj.weight.shape) == (8, 6)
    assert tuple(child.model.layers[2].self_attn.q_proj.weight.shape) == (8, 8)
    assert all(
        int(torch.count_nonzero(parameter).item()) > 0
        for parameter in child.parameters()
    )
    with torch.no_grad():
        logits = child(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            use_cache=False,
        ).logits
    assert tuple(logits.shape) == (1, 3, 23)
    assert bool(torch.isfinite(logits).all())


def test_same_geometry_dense_child_remains_supported() -> None:
    layer_types = ("conv", "full_attention")
    parent = _model(
        _config(
            hidden=8,
            intermediate=12,
            query_heads=2,
            key_value_heads=1,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4201,
    )
    child = _model(
        _config(
            hidden=4,
            intermediate=6,
            query_heads=1,
            key_value_heads=1,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4202,
    )
    _zero(child)

    report = _fold_dense_blank_from_parent(
        parent,
        child,
        _residual_map(parent, 4),
    )

    assert report["complete_target_parameter_overwrite"] is True
    conv = report["layers"][0]["operator"]
    assert conv["source_internal_channels"] == 8
    assert conv["target_internal_channels"] == 4
    attention = report["layers"][1]["operator"]
    assert attention["target_physical_geometry"] == {
        "query_heads": 1,
        "key_value_heads": 1,
        "head_dim": 4,
    }


def test_factorized_short_conv_writer_fails_before_target_write() -> None:
    layer_types = (
        "conv",
        "full_attention",
        "full_attention",
        "conv",
    )
    parent = _model(
        _config(
            hidden=16,
            intermediate=24,
            query_heads=4,
            key_value_heads=2,
            head_dim=4,
            layer_types=layer_types,
        ),
        seed=4301,
    )
    child = _heterogeneous_child(factorized_writer=True)

    with pytest.raises(
        Lfm2DenseBudgetFoldError,
        match="Factorized nn.Sequential ShortConv writers",
    ):
        _fold_dense_blank_from_parent(
            parent,
            child,
            _residual_map(parent, 8),
        )

    assert all(
        int(torch.count_nonzero(parameter).item()) == 0
        for parameter in child.parameters()
    )
