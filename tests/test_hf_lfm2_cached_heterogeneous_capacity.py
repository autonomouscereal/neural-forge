from __future__ import annotations

import hashlib

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from neural_forge.dynamic_behavior_gate import (  # noqa: E402
    observe_model_execution,
)
from neural_forge.dynamic_model_inspector import (  # noqa: E402
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (  # noqa: E402
    Lfm2HeterogeneousLayoutSelectionInput,
)
from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (  # noqa: E402
    build_lfm2_cached_heterogeneous_capacity_option_bank,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (  # noqa: E402
    build_lfm2_heterogeneous_target_pins,
)
from neural_forge.lfm2_native_exact4x_target import (  # noqa: E402
    lfm2_identity_tokenizer_evidence_sha256,
)
from test_dynamic_lfm2_heterogeneous_budgeted_materialization import (  # noqa: E402
    _parent,
)
from test_hf_lfm2_dynamic_closed_loop_adapter import _batch  # noqa: E402


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def test_cached_capacity_bank_selects_high_capacity_by_trace_pressure(
    tmp_path,
) -> None:
    parent = _parent()
    tokenizer_semantics = _digest("cached-capacity-tokenizer")
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer_semantics,
    )
    cache = build_lfm2_circuit_trace_cache(
        tmp_path / "parent-cache",
        parent,
        [
            _batch((1, 2, 3, 4), (5,)),
            _batch((6, 7, 8, 9), (10,)),
        ],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )

    with observe_model_execution(parent) as parent_execution:
        first = build_lfm2_cached_heterogeneous_capacity_option_bank(
            graph,
            cache,
            short_conv_internal_widths=(12, 14),
            attention_head_options=((2, 1, 4), (4, 2, 8)),
            mlp_intermediate_sizes=(18, 20),
        )
        second = build_lfm2_cached_heterogeneous_capacity_option_bank(
            graph,
            cache,
            short_conv_internal_widths=(12, 14),
            attention_head_options=((2, 1, 4), (4, 2, 8)),
            mlp_intermediate_sizes=(18, 20),
        )

    assert parent_execution["module_forward_calls"] == 0
    assert parent_execution["root_forward_calls"] == 0
    assert first == second
    assert first.source_graph_fingerprint == graph.fingerprint
    assert first.parent_cache_key == cache.cache_key
    assert len(first.layer_evidence) == 4
    assert len(first.layer_options) == 16
    assert first.as_dict()["selection_contract"] == {
        "cached_parent_traces_only": True,
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "global_exact_budget_solver_selects_layout": True,
        "hardcoded_layer_indices": False,
    }

    pins = build_lfm2_heterogeneous_target_pins(
        hidden_size=16,
        vocab_size=64,
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
        target_vocab_evidence_sha256=tokenizer_semantics,
        runtime_evidence_sha256=_digest("cached-capacity-runtime"),
        token_bridge_evidence_sha256=(
            lfm2_identity_tokenizer_evidence_sha256(
                tokenizer_semantics_sha256=tokenizer_semantics,
                vocabulary_size=64,
            )
        ),
    )
    selection = Lfm2HeterogeneousLayoutSelectionInput(
        target_pins=pins,
        layer_options=first.layer_options,
    ).select(
        graph,
        # All-low is 8,040. This exact count adds one high-attention
        # allocation (+1,160) and one high-MLP allocation (+96).
        exact_target_unique_parameters=9_296,
    )
    descriptor = selection.descriptor
    attention_evidence = tuple(
        value for value in first.layer_evidence if value.layer_type == "full_attention"
    )
    expected_high_attention = max(
        attention_evidence,
        key=lambda value: (
            value.operator_delta_energy_fraction,
            -value.layer_index,
        ),
    ).layer_index
    expected_high_mlp = max(
        first.layer_evidence,
        key=lambda value: (
            value.mlp_delta_energy_fraction,
            -value.layer_index,
        ),
    ).layer_index

    assert descriptor.accounting.unique_parameter_elements == 9_296
    assert [
        layer.layer_index
        for layer in descriptor.layers
        if layer.layer_type == "full_attention" and layer.query_heads == 4
    ] == [expected_high_attention]
    assert [
        layer.layer_index
        for layer in descriptor.layers
        if layer.mlp_intermediate_size == 20
    ] == [expected_high_mlp]
