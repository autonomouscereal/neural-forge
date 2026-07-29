from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    capture_lfm2_child_replay,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_dynamic_blank_materializer import (  # noqa: E402
    materialize_lfm2_dynamic_blank_child,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (  # noqa: E402
    Lfm2HeterogeneousLayerPhysicalLayout,
    build_lfm2_heterogeneous_physical_layout,
)


def _materialized_child():
    config = transformers.Lfm2Config(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=6,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        layer_types=["conv", "full_attention", "full_attention"],
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
    descriptor = build_lfm2_heterogeneous_physical_layout(
        hidden_size=8,
        vocab_size=19,
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
        layers=(
            Lfm2HeterogeneousLayerPhysicalLayout.short_conv(
                layer_index=0,
                mlp_intermediate_size=5,
                internal_width=10,
                selection_evidence_sha256="0" * 64,
            ),
            Lfm2HeterogeneousLayerPhysicalLayout.attention(
                layer_index=1,
                mlp_intermediate_size=7,
                query_heads=4,
                key_value_heads=2,
                head_dim=2,
                selection_evidence_sha256="1" * 64,
            ),
            Lfm2HeterogeneousLayerPhysicalLayout.attention(
                layer_index=2,
                mlp_intermediate_size=9,
                query_heads=2,
                key_value_heads=1,
                head_dim=4,
                selection_evidence_sha256="2" * 64,
            ),
        ),
        source_topology_sha256="a" * 64,
        allocation_evidence_sha256="b" * 64,
        token_bridge_evidence_sha256="c" * 64,
    )
    model = transformers.Lfm2ForCausalLM(config)
    materialize_lfm2_dynamic_blank_child(
        model,
        config,
        descriptor,
        expected_descriptor_sha256=descriptor.descriptor_sha256,
    )
    return model.eval(), descriptor


def test_prefix_replay_reads_sealed_per_layer_attention_and_conv_widths() -> None:
    child, descriptor = _materialized_child()
    before = fingerprint_lfm2_parent(child)

    replay = capture_lfm2_child_replay(
        child,
        (
            {
                "input_ids": torch.tensor([[1, 4, 5]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
                "forced_decode_ids": torch.tensor([[6, 7]], dtype=torch.long),
            },
        ),
        selected_layer_indices=(0, 1, 2),
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=3,
            max_decode_steps=4,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        replay_capture_mode=CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    )

    assert fingerprint_lfm2_parent(child) == before
    assert replay.selected_layer_indices == (0, 1, 2)
    assert replay.metadata["execution"]["sealed_heterogeneous_dynamic_runtime"] is True
    assert replay.metadata["execution"]["fast_short_conv_selected"] is False
    topology = replay.metadata["layer_topology"]
    assert topology["layers"][0]["contract"]["operator"]["internal_channel_width"] == 10
    assert topology["layers"][1]["contract"]["operator"]["query_heads"] == 4
    assert topology["layers"][1]["contract"]["operator"]["key_value_heads"] == 2
    assert topology["layers"][1]["contract"]["operator"]["head_dim"] == 2
    assert topology["layers"][2]["contract"]["operator"]["query_heads"] == 2
    assert topology["layers"][2]["contract"]["operator"]["key_value_heads"] == 1
    assert topology["layers"][2]["contract"]["operator"]["head_dim"] == 4
    batch = replay.load_batch(0)
    for phase in batch.phase_order:
        assert batch.tensors[f"{phase}.layer.000.short_conv.b"].shape[-1] == 10
        assert batch.tensors[f"{phase}.layer.000.short_conv.o"].shape[-1] == 8
        assert batch.tensors[f"{phase}.layer.001.attention.q"].shape[1:] == (
            4,
            3 if phase == "prefill" else 1,
            2,
        )
        assert batch.tensors[f"{phase}.layer.002.attention.q"].shape[1:] == (
            2,
            3 if phase == "prefill" else 1,
            4,
        )
        assert batch.tensors[f"{phase}.logits"].shape[-1] == 19
    assert descriptor.accounting.unique_parameter_elements == sum(
        parameter.numel() for parameter in child.parameters()
    )
