from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

from experiments import run_lfm2_residual_attention_fold as residual  # noqa: E402
from neural_forge.dynamic_closed_loop_fold import (  # noqa: E402
    CapturePoint,
    ClosedLoopAudit,
    ClosedLoopCompilerConfig,
    CompileStage,
    CompiledFoldStage,
    FoldScope,
)
import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache  # noqa: E402
import neural_forge.hf_lfm2_dynamic_closed_loop_adapter as adapter_module  # noqa: E402
from neural_forge.dynamic_behavior_gate import (  # noqa: E402
    observe_model_execution,
)
from neural_forge.dynamic_fold_executor import (  # noqa: E402
    FoldExecutionContext,
    fingerprint_model_state,
    run_fold_only_callback,
)
from neural_forge.dynamic_fold_planner import DynamicFoldPlanner  # noqa: E402
from neural_forge.dynamic_fold_registry import ExperimentLedger  # noqa: E402
from neural_forge.dynamic_lfm2_adapter import (  # noqa: E402
    Lfm2TargetSynthesisAdapter,
    encode_lfm2_layer_types,
    encode_lfm2_source_spans,
)
from neural_forge.dynamic_model_inspector import (  # noqa: E402
    inspect_model_circuits,
)
from neural_forge.dynamic_parameter_frontier import (  # noqa: E402
    Lfm2DenseTargetTopologyCatalog,
    Lfm2HeterogeneousLayoutSelectionInput,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_dynamic_blank_materializer import (  # noqa: E402
    extract_lfm2_dynamic_physical_layout_from_config,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    load_lfm2_direct_local_model,
)
from neural_forge.hf_lfm2_dynamic_closed_loop_adapter import (  # noqa: E402
    LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3,
    Lfm2DynamicClosedLoopAdapter,
    Lfm2DynamicClosedLoopAdapterConfig,
    Lfm2DynamicClosedLoopAdapterError,
    _OutputInterfacePayload,
    _tensor_sha256,
)
from neural_forge.hf_lfm2_dense_budget_model_fold import (  # noqa: E402
    Lfm2DenseBudgetEvidenceProvider,
    Lfm2DenseBudgetFoldConfig,
    Lfm2DenseBudgetModelOperator,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (  # noqa: E402
    build_lfm2_attention_layer_option,
    build_lfm2_heterogeneous_target_pins,
    build_lfm2_short_conv_layer_option,
    discover_lfm2_source_layers,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    compile_lfm2_structural_short_conv_consumer_fold,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (  # noqa: E402
    Lfm2TargetTokenizedRecord,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    seal_strict_tokenizer_bridge_artifact,
)
from test_dynamic_lfm2_heterogeneous_budgeted_materialization import (  # noqa: E402
    _heterogeneous_case,
    _parent,
)
from test_hf_lfm2_child_closed_loop_token_bridge import (  # noqa: E402
    _Tokenizer,
    _target_files,
)
from test_hf_lfm2_dynamic_closed_loop_adapter import (  # noqa: E402
    _DenseMap,
    _batch,
)
from test_hf_lfm2_structural_split_install import (  # noqa: E402
    _short_conv_case,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _capture_config() -> Lfm2CircuitTraceCaptureConfig:
    return Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=2,
        max_decode_steps=1,
        max_capture_bytes=64 * 1024 * 1024,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )


def _materialize_case():
    parent, graph, tokenizer, synthesis, candidate, request = _heterogeneous_case()
    materialized = synthesis.materialize(
        request,
        {"parent": parent},
        {"parent": graph},
        None,
    )
    return parent, graph, candidate, materialized


def _bridged_heterogeneous_case(
    tmp_path: Path,
    *,
    factor_writer: bool = False,
):
    donor_surfaces = ["<pad>", "<bos>", "<eos>"] + [
        f"<token-{index:02d}>" for index in range(3, 64)
    ]
    target_surfaces = donor_surfaces[:17]
    donor = _Tokenizer(
        donor_surfaces,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    target_tokenizer = _Tokenizer(
        target_surfaces,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    target_root = tmp_path / "bridged-target-tokenizer"
    _target_files(target_root)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target_tokenizer,
        target_root,
        donor_row_capacity=64,
        target_row_capacity=17,
    )
    parent = _parent()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
    )
    per_layer = {
        0: (18, 12, None),
        1: (20, None, (2, 1, 8)),
        2: (22, 14, None),
        3: (24, None, (4, 2, 8)),
    }
    options = []
    for discovered in discover_lfm2_source_layers(graph):
        intermediate, internal, attention = per_layer[discovered.ordinal]
        if discovered.layer_type == "conv":
            assert internal is not None
            options.append(
                build_lfm2_short_conv_layer_option(
                    source_layer_sha256=discovered.source_layer_sha256,
                    mlp_intermediate_size=intermediate,
                    internal_width=internal,
                    writer_rank=(
                        3
                        if factor_writer and discovered.ordinal == 0
                        else None
                    ),
                    selection_evidence_sha256=_digest(
                        f"bridged-layer-{discovered.ordinal}-evidence"
                    ),
                    structural_risk=0.2 + discovered.ordinal / 100,
                )
            )
        else:
            assert attention is not None
            options.append(
                build_lfm2_attention_layer_option(
                    source_layer_sha256=discovered.source_layer_sha256,
                    mlp_intermediate_size=intermediate,
                    query_heads=attention[0],
                    key_value_heads=attention[1],
                    head_dim=attention[2],
                    selection_evidence_sha256=_digest(
                        f"bridged-layer-{discovered.ordinal}-evidence"
                    ),
                    structural_risk=0.1 + discovered.ordinal / 100,
                )
            )
    pins = build_lfm2_heterogeneous_target_pins(
        hidden_size=16,
        vocab_size=17,
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
        target_vocab_evidence_sha256=(bridge.target_tokenizer_semantics_sha256),
        runtime_evidence_sha256=_digest("bridged-runtime-evidence"),
        token_bridge_evidence_sha256=bridge.artifact_sha256,
    )
    factor_writer_reduction = (
        16 * 12 - 3 * (16 + 12) if factor_writer else 0
    )
    exact_target_parameters = 8_798 - factor_writer_reduction
    catalog = Lfm2DenseTargetTopologyCatalog(
        tokenizer_bridge_artifact=bridge,
        target_token_interface_mode="untied",
        heterogeneous_layout_selection_input=(
            Lfm2HeterogeneousLayoutSelectionInput(
                target_pins=pins,
                layer_options=tuple(options),
            )
        ),
    )
    frontier = catalog.enumerate_frontier(
        graph,
        requested_unique_parameters=exact_target_parameters,
        target_vocabulary_size=17,
        match_policy="exact",
        maximum_candidates=1,
    )
    candidate = frontier.candidates[0]
    synthesis = Lfm2TargetSynthesisAdapter(
        terminal_topology=None,
        evidence_fingerprint=_digest("bridged-parent-evidence"),
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge_artifact=bridge,
    )
    request = candidate.to_target_request(
        child_name="bridged-heterogeneous-exact-blank",
        tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        adapter_id=synthesis.adapter_id,
        blank_seed=9302,
    )
    materialized = synthesis.materialize(
        request,
        {"parent": parent},
        {"parent": graph},
        None,
    )
    records = (
        Lfm2TargetTokenizedRecord(
            record_id="fit-0",
            split="fit",
            input_ids=torch.tensor([[1, 3, 4, 5]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            forced_decode_ids=torch.tensor([[6]], dtype=torch.long),
        ),
        Lfm2TargetTokenizedRecord(
            record_id="heldout-0",
            split="heldout",
            input_ids=torch.tensor([[1, 7, 8, 9]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            forced_decode_ids=torch.tensor([[10]], dtype=torch.long),
        ),
    )
    trace_adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
        expected_artifact_sha256=bridge.artifact_sha256,
    )
    trace_cache = build_lfm2_circuit_trace_cache(
        tmp_path / "bridged-parent-cache",
        parent,
        trace_adapter.parent_cache_batches(),
        config=_capture_config(),
        input_device="cpu",
    )
    return {
        "bridge": bridge,
        "candidate": candidate,
        "graph": graph,
        "materialized": materialized,
        "parent": parent,
        "request": request,
        "trace_adapter": trace_adapter,
        "trace_cache": trace_cache,
    }


@pytest.fixture(scope="module")
def heterogeneous_adapter_case(tmp_path_factory):
    parent, graph, candidate, materialized = _materialize_case()
    replay_config = _capture_config()
    parent_cache = build_lfm2_circuit_trace_cache(
        tmp_path_factory.mktemp("heterogeneous-adapter-parent"),
        parent,
        [
            _batch((1, 2, 3, 4), (5,)),
            _batch((6, 7, 8, 9), (10,)),
        ],
        config=replay_config,
        input_device="cpu",
    )
    coordinate_map = _DenseMap(32, 16)
    adapter = Lfm2DynamicClosedLoopAdapter(
        source_model=parent,
        child_model=materialized.model,
        parent_cache=parent_cache,
        coordinate_map=coordinate_map,
        initial_child_basis_sha256=coordinate_map.map_sha256,
        replay_config=replay_config,
        input_device="cpu",
        source_graph_fingerprint=graph.fingerprint,
        heterogeneous_layout_selection_receipt=(
            materialized.adapter_payload.heterogeneous_layout_selection_receipt
        ),
        identity_tokenizer_evidence_sha256=(
            materialized.adapter_payload.heterogeneous_layout_descriptor.token_bridge_evidence_sha256
        ),
    )
    return {
        "adapter": adapter,
        "candidate": candidate,
        "child": materialized.model,
        "coordinate_map": coordinate_map,
        "graph": graph,
        "parent": parent,
        "parent_cache": parent_cache,
        "replay_config": replay_config,
    }


def test_dynamic_inspector_accepts_mixed_short_conv_and_attention_widths(
    heterogeneous_adapter_case,
) -> None:
    case = heterogeneous_adapter_case
    adapter = case["adapter"]
    contracts = adapter._child_contracts

    assert case["candidate"].unique_parameters == 9_910
    assert case["graph"].unique_parameter_count >= 4 * 9_910
    assert [contract.ffn_intermediate_size for contract in contracts] == [
        18,
        20,
        22,
        24,
    ]
    assert [
        contract.operator.internal_channel_width
        for contract in contracts
        if contract.kind == "short_conv"
    ] == [12, 14]
    assert [
        (
            contract.operator.query_heads,
            contract.operator.key_value_heads,
            contract.operator.head_dim,
        )
        for contract in contracts
        if contract.kind == "full_attention"
    ] == [(2, 1, 4), (4, 2, 8)]
    assert [
        assignment.target_width
        for _index, assignment in sorted(adapter._short_conv_assignments.items())
    ] == [12, 14]


def test_heterogeneous_child_direct_local_reload_restores_ragged_graph(
    heterogeneous_adapter_case,
    tmp_path: Path,
) -> None:
    child = heterogeneous_adapter_case["child"]
    destination = tmp_path / "ragged-child"
    state_before = fingerprint_model_state(child)
    layout_before = extract_lfm2_dynamic_physical_layout_from_config(child.config)

    residual.atomic_save_local_checkpoint(
        child,
        destination,
        output_root=tmp_path,
    )
    loaded = load_lfm2_direct_local_model(
        destination,
        device="cpu",
        dtype=torch.float32,
    )
    layout_after = extract_lfm2_dynamic_physical_layout_from_config(loaded.config)

    assert layout_before is not None
    assert layout_after is not None
    assert layout_after.descriptor_sha256 == layout_before.descriptor_sha256
    assert fingerprint_model_state(loaded) == state_before
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert [
        int(layer.feed_forward.w1.out_features) for layer in loaded.model.layers
    ] == [18, 20, 22, 24]
    assert [
        int(layer.conv.in_proj.out_features // 3)
        for layer in loaded.model.layers
        if hasattr(layer, "conv")
    ] == [12, 14]


def test_equal_vocab_heterogeneous_child_requires_identity_token_evidence(
    heterogeneous_adapter_case,
) -> None:
    case = heterogeneous_adapter_case

    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="identity_tokenizer_evidence_sha256",
    ):
        Lfm2DynamicClosedLoopAdapter(
            source_model=case["parent"],
            child_model=case["child"],
            parent_cache=case["parent_cache"],
            coordinate_map=case["coordinate_map"],
            initial_child_basis_sha256=(case["coordinate_map"].map_sha256),
            replay_config=case["replay_config"],
            input_device="cpu",
            source_graph_fingerprint=case["graph"].fingerprint,
            heterogeneous_layout_selection_receipt=(
                case["adapter"].heterogeneous_layout_selection_receipt
            ),
        )


def test_heterogeneous_child_rejects_different_source_graph(
    heterogeneous_adapter_case,
) -> None:
    case = heterogeneous_adapter_case

    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="do not share one authenticated contract",
    ):
        Lfm2DynamicClosedLoopAdapter(
            source_model=case["parent"],
            child_model=case["child"],
            parent_cache=case["parent_cache"],
            coordinate_map=case["coordinate_map"],
            initial_child_basis_sha256=(case["coordinate_map"].map_sha256),
            replay_config=case["replay_config"],
            input_device="cpu",
            source_graph_fingerprint=_digest("different-source-graph"),
            heterogeneous_layout_selection_receipt=(
                case["adapter"].heterogeneous_layout_selection_receipt
            ),
            identity_tokenizer_evidence_sha256=(
                case["adapter"].identity_tokenizer_evidence_sha256
            ),
        )


def test_heterogeneous_prefix_replay_admits_merged_layer_captures(
    heterogeneous_adapter_case,
) -> None:
    case = heterogeneous_adapter_case
    adapter = case["adapter"]

    # Canonical prefix replay never consults the single-pass certification
    # registry, so a heterogeneous child merges exactly when a homogeneous one
    # does.  The decline that remains belongs to single_pass_causal.
    assert (
        adapter.config.replay_capture_mode
        == CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert adapter._child_heterogeneous_layout is not None
    assert adapter.supports_merged_child_captures() is True

    audit = ClosedLoopAudit()
    records = tuple(adapter.parent_trajectory_records(audit))
    post_downstream, pre_producer = adapter.replay_and_capture_child_merged(
        (
            (
                FoldScope("layer", 0),
                CapturePoint.POST_DOWNSTREAM,
                records[:1],
            ),
            (FoldScope("layer", 1), CapturePoint.PRE_PRODUCER, records[1:]),
        ),
        audit,
    )
    try:
        primary = adapter._capture_payload(post_downstream)
        secondary = adapter._capture_payload(pre_producer)
        assert primary is secondary
        assert primary.selected_layer_indices == (0, 1)
        assert primary.causal_certification_sha256 is None
        assert post_downstream.capture_sha256 != pre_producer.capture_sha256

        # Each merged point still retains exactly what its own unmerged
        # capture would have retained; the union only widens the replay.
        profile = primary.child_evidence_profile
        assert profile is not None
        for layer_index, point in (
            (0, CapturePoint.POST_DOWNSTREAM),
            (1, CapturePoint.PRE_PRODUCER),
        ):
            single = adapter._child_evidence_profile(
                FoldScope("layer", layer_index),
                point,
                (layer_index,),
                body_internal_role=None,
            )
            assert profile.suffixes_for_layer(layer_index) == (
                single.suffixes_for_layer(layer_index)
            )
    finally:
        adapter._release_capture_lease(post_downstream)
        adapter._release_capture_lease(pre_producer)


def test_heterogeneous_single_pass_declines_merged_layer_captures(
    heterogeneous_adapter_case,
) -> None:
    case = heterogeneous_adapter_case
    adapter = Lfm2DynamicClosedLoopAdapter(
        source_model=case["parent"],
        child_model=case["child"],
        parent_cache=case["parent_cache"],
        coordinate_map=case["coordinate_map"],
        initial_child_basis_sha256=(case["coordinate_map"].map_sha256),
        replay_config=case["replay_config"],
        input_device="cpu",
        config=Lfm2DynamicClosedLoopAdapterConfig(
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        ),
        source_graph_fingerprint=case["graph"].fingerprint,
        heterogeneous_layout_selection_receipt=(
            case["adapter"].heterogeneous_layout_selection_receipt
        ),
        identity_tokenizer_evidence_sha256=(
            case["adapter"].identity_tokenizer_evidence_sha256
        ),
    )
    assert adapter.supports_merged_child_captures() is False


def test_heterogeneous_child_self_certifies_single_pass_replay(
    heterogeneous_adapter_case,
    monkeypatch,
    request,
) -> None:
    case = heterogeneous_adapter_case
    from transformers import masking_utils
    from transformers.models.lfm2 import modeling_lfm2

    runtime_identity = (
        trace_cache._package_version("transformers"),
        trace_cache._source_file_sha256(
            Path(modeling_lfm2.__file__).resolve(),
            "test Transformers LFM2",
        ),
        trace_cache._source_file_sha256(
            Path(masking_utils.__file__).resolve(),
            "test Transformers masking",
        ),
    )
    monkeypatch.setattr(
        trace_cache,
        "_SINGLE_PASS_CAUSAL_RUNTIME_ALLOWLIST",
        frozenset({runtime_identity}),
    )
    authentic_capture = adapter_module.capture_lfm2_child_replay
    replay_modes: list[str] = []
    replay_policies: list[tuple[float, float, bool]] = []

    def recording_capture(*args, **kwargs):
        replay_modes.append(
            kwargs.get(
                "replay_capture_mode",
                CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
            )
        )
        capture_config = kwargs["config"]
        replay_policies.append(
            (
                capture_config.logit_rtol,
                capture_config.logit_atol,
                capture_config.require_top1_agreement,
            )
        )
        return authentic_capture(*args, **kwargs)

    monkeypatch.setattr(
        adapter_module,
        "capture_lfm2_child_replay",
        recording_capture,
    )
    permissive_replay_config = Lfm2CircuitTraceCaptureConfig(
        sample_last_tokens=case["replay_config"].sample_last_tokens,
        max_decode_steps=case["replay_config"].max_decode_steps,
        max_capture_bytes=case["replay_config"].max_capture_bytes,
        logit_rtol=0.2,
        logit_atol=1.0,
        require_top1_agreement=False,
        decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
    )
    adapter = Lfm2DynamicClosedLoopAdapter(
        source_model=case["parent"],
        child_model=case["child"],
        parent_cache=case["parent_cache"],
        coordinate_map=case["coordinate_map"],
        initial_child_basis_sha256=(case["coordinate_map"].map_sha256),
        replay_config=permissive_replay_config,
        input_device="cpu",
        config=Lfm2DynamicClosedLoopAdapterConfig(
            replay_capture_mode=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        ),
        source_graph_fingerprint=case["graph"].fingerprint,
        heterogeneous_layout_selection_receipt=(
            case["adapter"].heterogeneous_layout_selection_receipt
        ),
        identity_tokenizer_evidence_sha256=(
            case["adapter"].identity_tokenizer_evidence_sha256
        ),
    )
    expected_policy = dict(
        adapter_module._heterogeneous_single_pass_equivalence_policy(
            case["child"],
            "cpu",
        )
    )
    expected_replay_policy = (
        expected_policy["rtol"],
        expected_policy["atol"],
        True,
    )

    audit = ClosedLoopAudit()
    records = tuple(adapter.parent_trajectory_records(audit))
    capture = adapter.replay_and_capture_child(
        FoldScope("layer", 0),
        CapturePoint.PRE_PRODUCER,
        records[:1],
        audit,
    )
    certification = adapter._heterogeneous_single_pass_certification
    assert certification is not None
    assert certification["all_sealed_batches_compared"] is True
    assert certification["equivalence"]["tolerance_equivalent"] is True
    assert certification["equivalence"]["reference_supplied"] is True
    assert certification["equivalence"]["rtol"] == expected_policy["rtol"]
    assert certification["equivalence"]["atol"] == expected_policy["atol"]
    assert certification["equivalence"]["top1_required"] is True
    assert certification["equivalence"]["logit_error_model"] == (
        trace_cache.CHILD_REPLAY_LOGIT_ERROR_MODEL
    )
    assert certification["equivalence"]["non_logit_error_model"] == (
        trace_cache.CHILD_REPLAY_NON_LOGIT_ERROR_MODEL
    )
    assert certification["equivalence"][
        "non_logit_tolerance_unchanged"
    ] is False
    assert certification["strict_equivalence_policy"] == expected_policy
    assert certification["numeric_policy_sha256"] == (
        expected_policy["numeric_policy_sha256"]
    )
    assert certification["replay_numeric_policy_sha256"] == (
        expected_policy["replay_numeric_policy_sha256"]
    )
    assert certification["non_logit_envelope_sha256"] == (
        expected_policy["non_logit_envelope_sha256"]
    )
    assert capture.payload.causal_certification_sha256 == (
        certification["certification_sha256"]
    )
    assert capture.payload.numeric_policy_sha256 == (
        certification["numeric_policy_sha256"]
    )
    assert capture.payload.replay_numeric_policy_sha256 == (
        certification["replay_numeric_policy_sha256"]
    )
    assert capture.payload.non_logit_envelope_sha256 == (
        certification["non_logit_envelope_sha256"]
    )
    replay_forwards = len(case["parent_cache"])
    canonical_forwards = capture.payload.canonical_prefix_child_forwards
    assert capture.payload.replay_child_forwards == replay_forwards
    assert capture.payload.certification_reference_child_forwards == (
        canonical_forwards
    )
    assert capture.payload.actual_child_forwards == (
        canonical_forwards + replay_forwards
    )
    assert capture.payload.saved_child_forwards == -replay_forwards
    assert certification["proof_forward_accounting"] == {
        "canonical_baseline_child_forwards": canonical_forwards,
        "canonical_reference_child_forwards": canonical_forwards,
        "single_pass_child_forwards": replay_forwards,
        "total_certification_child_forwards": (
            canonical_forwards + replay_forwards
        ),
        "net_certification_saved_child_forwards": -replay_forwards,
    }
    assert replay_modes == [
        CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    ]
    assert replay_policies == [expected_replay_policy] * 2

    first_certification_sha256 = certification["certification_sha256"]
    with pytest.raises(TypeError):
        certification["equivalence"]["tolerance_equivalent"] = False

    replay_modes.clear()
    replay_policies.clear()
    same_state_capture = adapter.replay_and_capture_child(
        FoldScope("layer", 0),
        CapturePoint.PRE_PRODUCER,
        records[:1],
        audit,
    )
    assert replay_modes == [CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE]
    assert replay_policies == [expected_replay_policy]
    assert same_state_capture.payload.replay_child_forwards == replay_forwards
    assert (
        same_state_capture.payload.certification_reference_child_forwards
        == 0
    )
    assert same_state_capture.payload.actual_child_forwards == replay_forwards
    assert same_state_capture.payload.saved_child_forwards == (
        canonical_forwards - replay_forwards
    )
    assert same_state_capture.payload.causal_certification_sha256 == (
        first_certification_sha256
    )
    false_savings_capture = replace(
        same_state_capture,
        payload=adapter_module._CapturePayloadLease(
            replace(
                same_state_capture.payload.get(),
                saved_child_forwards=(
                    same_state_capture.payload.saved_child_forwards + 1
                ),
            ),
        ),
    )
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="canonical all-batch child replay",
    ):
        adapter._capture_payload(false_savings_capture)

    # The same child under a different retained-hook selection needs a new
    # exact proof.  The earlier payload remains independently authentic even
    # while both captures and both immutable registry entries coexist.
    replay_modes.clear()
    replay_policies.clear()
    other_selection_capture = adapter.replay_and_capture_child(
        FoldScope("layer", 1),
        CapturePoint.PRE_PRODUCER,
        records[:1],
        audit,
    )
    assert replay_modes == [
        CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    ]
    assert other_selection_capture.payload.selected_layer_indices == (1,)
    assert other_selection_capture.payload.causal_certification_sha256 != (
        first_certification_sha256
    )
    assert len(
        adapter._heterogeneous_single_pass_certification_registry
    ) == 2
    assert adapter._capture_payload(same_state_capture) is (
        same_state_capture.payload.get()
    )

    first_parameter = next(case["child"].parameters())
    original_first_parameter = first_parameter.detach().clone()

    def restore_first_parameter() -> None:
        with torch.no_grad():
            first_parameter.copy_(original_first_parameter)

    request.addfinalizer(restore_first_parameter)
    with torch.no_grad():
        first_parameter.view(-1)[0].add_(0.125)
    changed_child_identity = trace_cache.fingerprint_lfm2_parent(
        case["child"]
    ).identity_sha256

    replay_modes.clear()
    replay_policies.clear()
    changed_state_capture = adapter.replay_and_capture_child(
        FoldScope("layer", 0),
        CapturePoint.PRE_PRODUCER,
        records[:1],
        audit,
    )
    changed_certification = (
        adapter._heterogeneous_single_pass_certification
    )
    assert changed_certification is not None
    assert replay_modes == [
        CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    ]
    assert replay_policies == [expected_replay_policy] * 2
    assert changed_certification["proof_child_identity_sha256"] == (
        changed_child_identity
    )
    assert changed_certification["certification_sha256"] != (
        first_certification_sha256
    )
    assert changed_state_capture.payload.causal_certification_sha256 == (
        changed_certification["certification_sha256"]
    )
    assert (
        changed_state_capture.payload.certification_reference_child_forwards
        == canonical_forwards
    )
    assert changed_state_capture.payload.actual_child_forwards == (
        canonical_forwards + replay_forwards
    )
    assert changed_state_capture.payload.saved_child_forwards == (
        -replay_forwards
    )

    semantic_forged = adapter_module._canonical_json_plain_copy(
        changed_certification,
        "test certification",
    )
    semantic_forged.pop("certification_sha256")
    semantic_forged["equivalence"]["top1_required"] = False
    adapter._heterogeneous_single_pass_certification = (
        adapter_module._seal_immutable_json_mapping(
            semantic_forged,
            "certification_sha256",
            "test certification",
        )
    )
    replay_modes.clear()
    replay_policies.clear()
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="proof semantics are invalid",
    ):
        adapter.replay_and_capture_child(
            FoldScope("layer", 0),
            CapturePoint.PRE_PRODUCER,
            records[:1],
            audit,
        )
    assert replay_modes == []
    assert replay_policies == []

    accounting_forged = adapter_module._canonical_json_plain_copy(
        changed_certification,
        "test certification",
    )
    accounting_forged.pop("certification_sha256")
    forged_accounting = accounting_forged["proof_forward_accounting"]
    forged_accounting["single_pass_child_forwards"] += 1
    forged_accounting["total_certification_child_forwards"] += 1
    forged_accounting["net_certification_saved_child_forwards"] -= 1
    adapter._heterogeneous_single_pass_certification = (
        adapter_module._seal_immutable_json_mapping(
            accounting_forged,
            "certification_sha256",
            "test certification",
        )
    )
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="changed after admission",
    ):
        adapter.replay_and_capture_child(
            FoldScope("layer", 0),
            CapturePoint.PRE_PRODUCER,
            records[:1],
            audit,
        )
    assert replay_modes == [CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE]
    assert replay_policies == [expected_replay_policy]

    adapter._heterogeneous_single_pass_certification = changed_certification

    forged = dict(changed_certification)
    forged_equivalence = dict(changed_certification["equivalence"])
    forged_equivalence["tolerance_equivalent"] = False
    forged["equivalence"] = forged_equivalence
    adapter._heterogeneous_single_pass_certification = MappingProxyType(
        forged
    )
    replay_modes.clear()
    replay_policies.clear()
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="canonical seal does not match",
    ):
        adapter.replay_and_capture_child(
            FoldScope("layer", 0),
            CapturePoint.PRE_PRODUCER,
            records[:1],
            audit,
        )
    assert replay_modes == []
    assert replay_policies == []


def test_untied_output_install_changes_only_output_parameter(
    heterogeneous_adapter_case,
) -> None:
    adapter = heterogeneous_adapter_case["adapter"]
    child = heterogeneous_adapter_case["child"]
    child_input = child.get_input_embeddings().weight
    child_output = child.get_output_embeddings().weight
    input_before = child_input.detach().clone()
    output_before = child_output.detach().clone()
    input_id = id(child_input)
    output_id = id(child_output)

    candidate = output_before.clone()
    candidate[0, 0] = candidate[0, 0] + 0.25
    transform = torch.eye(int(candidate.shape[1]), dtype=torch.float64)
    payload = _OutputInterfacePayload(
        base_weight=output_before,
        candidate_weight=candidate,
        transform=transform,
        transform_sha256=_tensor_sha256(transform),
        base_weight_sha256=_tensor_sha256(output_before),
        candidate_weight_sha256=_tensor_sha256(candidate),
        source_binding_sha256=_digest("heterogeneous-source-binding"),
        split_sha256=_digest("heterogeneous-output-split"),
        ridge_effective=0.0,
        interpolation_alpha=1.0,
        materialized_delta_ratio=0.25,
        transform_condition_number=1.0,
        transform_min_singular_value=1.0,
        output_subspace_relative_rmse=0.0,
        body_latent_relative_rmse=0.0,
        identity_skip=False,
    )
    adapter._output_payload = payload
    scope = FoldScope(kind="output_interface")
    compiled = CompiledFoldStage(
        scope=scope,
        stage=CompileStage.OUTPUT_INTERFACE,
        fit_record_keys=(adapter._records[0].key,),
        artifact_sha256=_digest("heterogeneous-output-artifact"),
        payload=payload,
    )

    receipt = adapter.install_output_interface(
        scope,
        compiled,
        ClosedLoopAudit(),
    )

    installed_input = child.get_input_embeddings().weight
    installed_output = child.get_output_embeddings().weight
    metadata = dict(receipt.metadata)
    assert id(installed_input) == input_id
    assert id(installed_output) == output_id
    assert installed_input is not installed_output
    assert torch.equal(installed_input, input_before)
    assert _tensor_sha256(installed_input) == _tensor_sha256(input_before)
    assert not torch.equal(installed_output, output_before)
    assert torch.equal(installed_output, candidate)
    assert receipt.state_sha256_before != receipt.state_sha256_after
    assert receipt.topology_preserved is True
    assert metadata["untied_output_parameter_only"] is True
    assert metadata["input_embedding_bytes_unchanged"] is True


def test_heterogeneous_descriptor_rejects_different_token_bridge(
    tmp_path: Path,
) -> None:
    parent, _graph, _candidate, materialized = _materialize_case()
    surfaces = ["<pad>", "<bos>", "<eos>"] + [
        f"<token-{index:02d}>" for index in range(3, 64)
    ]
    donor = _Tokenizer(
        surfaces,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    target = _Tokenizer(
        surfaces,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    target_root = tmp_path / "target-tokenizer"
    _target_files(target_root)
    bridge = seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        target_root,
        donor_row_capacity=64,
        target_row_capacity=64,
    )
    records = (
        Lfm2TargetTokenizedRecord(
            record_id="fit-0",
            split="fit",
            input_ids=torch.tensor([[1, 3, 4, 5]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            forced_decode_ids=torch.tensor([[6]], dtype=torch.long),
        ),
        Lfm2TargetTokenizedRecord(
            record_id="heldout-0",
            split="heldout",
            input_ids=torch.tensor([[1, 7, 8, 9]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            forced_decode_ids=torch.tensor([[10]], dtype=torch.long),
        ),
    )
    trace_adapter = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        records,
        expected_artifact_sha256=bridge.artifact_sha256,
    )
    replay_config = _capture_config()
    parent_cache = build_lfm2_circuit_trace_cache(
        tmp_path / "parent-cache",
        parent,
        trace_adapter.parent_cache_batches(),
        config=replay_config,
        input_device="cpu",
    )
    descriptor = extract_lfm2_dynamic_physical_layout_from_config(
        materialized.model.config
    )

    assert trace_adapter.donor_row_capacity == 64
    assert trace_adapter.target_row_capacity == 64
    assert trace_adapter.record_count == len(parent_cache) == 2
    assert trace_adapter.bridge_artifact_sha256 == bridge.artifact_sha256
    assert descriptor.token_bridge_evidence_sha256 != bridge.artifact_sha256
    coordinate_map = _DenseMap(32, 16)
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="capacities, seals, child layout, or record count differ",
    ):
        Lfm2DynamicClosedLoopAdapter(
            source_model=parent,
            child_model=materialized.model,
            parent_cache=parent_cache,
            coordinate_map=coordinate_map,
            initial_child_basis_sha256=coordinate_map.map_sha256,
            replay_config=replay_config,
            input_device="cpu",
            tokenizer_bridge_artifact=bridge,
            tokenizer_bridge_trace_adapter=trace_adapter,
            source_graph_fingerprint=_graph.fingerprint,
            heterogeneous_layout_selection_receipt=(
                materialized.adapter_payload.heterogeneous_layout_selection_receipt
            ),
        )


def test_factor_consumer_artifact_preserves_dtype_and_rejects_tamper() -> None:
    case, actual_context = _short_conv_case()
    torch.manual_seed(9501)
    writer = torch.nn.Sequential(
        torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16),
        torch.nn.Linear(1, 2, bias=False, dtype=torch.bfloat16),
    )
    effective = (
        writer[1].weight.detach().to(torch.float64)
        @ writer[0].weight.detach().to(torch.float64)
    ).to(torch.bfloat16)
    compiled = compile_lfm2_structural_short_conv_consumer_fold(
        target_out_proj_weight_template=effective,
        actual_child_contexts=(actual_context,),
        cases=(case,),
        out_proj_weight_prior=effective,
        ridge=0.0,
        max_prior_delta_ratio=None,
    )
    view = adapter_module._ShortConvWriterView(
        topology="factorized",
        writer_rank=1,
        effective_weight=effective,
        bias=None,
        contract_sha256=_digest("factor-consumer-contract"),
        physical_state_sha256=_digest("factor-consumer-state"),
    )

    artifact = adapter_module._compile_factor_short_conv_consumer_artifact(
        compiled,
        view,
        writer,
    )
    repeated = adapter_module._compile_factor_short_conv_consumer_artifact(
        compiled,
        view,
        writer,
    )
    report = artifact.as_dict()

    assert artifact.encoder_weight.dtype == torch.bfloat16
    assert artifact.decoder_weight.dtype == torch.bfloat16
    assert artifact.encoder_weight.requires_grad is False
    assert artifact.decoder_weight.requires_grad is False
    assert artifact.encoder_weight.grad_fn is None
    assert artifact.decoder_weight.grad_fn is None
    assert torch.equal(artifact.encoder_weight, repeated.encoder_weight)
    assert torch.equal(artifact.decoder_weight, repeated.decoder_weight)
    assert artifact.artifact_sha256 == repeated.artifact_sha256
    assert report["method"] == (
        "deterministic-structural-cpu-fp64-exact-svd-truncation"
    )
    assert report["parent_forward_calls"] == 0
    assert report["model_generate_calls"] == 0
    assert report["training_steps"] == 0
    assert report["backward_calls"] == 0
    assert report["optimizer_steps"] == 0

    artifact.encoder_weight[0, 0] += torch.tensor(1, dtype=torch.bfloat16)
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="changed after sealing",
    ):
        artifact.verify()


def test_moving_gauge_factor_publication_is_deterministic_and_tamper_atomic() -> (
    None
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9307)
        encoder = torch.nn.Parameter(
            torch.randn(2, 5, dtype=torch.bfloat16),
            requires_grad=False,
        )
        decoder = torch.nn.Parameter(
            torch.randn(4, 2, dtype=torch.bfloat16),
            requires_grad=False,
        )
        requested = torch.randn(4, 5, dtype=torch.bfloat16)
    parameter_id = "model.layers.0.conv.out_proj.weight"
    physical_ids = (
        "model.layers.0.conv.out_proj.0.weight",
        "model.layers.0.conv.out_proj.1.weight",
    )
    site = adapter_module._MovingGaugeWriterSite(
        parameter_id=parameter_id,
        topology="factorized",
        writer_rank=2,
        writer_contract_sha256=_digest("moving-gauge-factor-contract"),
        seed_physical_state_sha256=_digest("moving-gauge-factor-seed"),
        physical_parameter_ids=physical_ids,
        physical_parameters=(encoder, decoder),
    )
    artifact = adapter_module._compile_moving_gauge_factor_writer_artifact(
        site,
        requested,
    )
    repeated = adapter_module._compile_moving_gauge_factor_writer_artifact(
        site,
        requested,
    )
    report = artifact.as_dict()

    assert artifact.encoder_weight.dtype == torch.bfloat16
    assert artifact.decoder_weight.dtype == torch.bfloat16
    assert artifact.encoder_weight.requires_grad is False
    assert artifact.decoder_weight.requires_grad is False
    assert artifact.encoder_weight.grad_fn is None
    assert artifact.decoder_weight.grad_fn is None
    assert torch.equal(artifact.encoder_weight, repeated.encoder_weight)
    assert torch.equal(artifact.decoder_weight, repeated.decoder_weight)
    assert artifact.artifact_sha256 == repeated.artifact_sha256
    assert report["parent_forward_calls"] == 0
    assert report["training_steps"] == 0
    adapter = object.__new__(Lfm2DynamicClosedLoopAdapter)
    encoder_id = id(encoder)
    decoder_id = id(decoder)
    exact_count = int(encoder.numel() + decoder.numel())
    changed = adapter._publish_moving_gauge_writer_replacements(
        MappingProxyType({parameter_id: site}),
        MappingProxyType({parameter_id: requested}),
        (parameter_id,),
        MappingProxyType({parameter_id: artifact}),
    )

    assert changed == physical_ids
    assert id(encoder) == encoder_id
    assert id(decoder) == decoder_id
    assert int(encoder.numel() + decoder.numel()) == exact_count
    assert encoder.dtype == torch.bfloat16
    assert decoder.dtype == torch.bfloat16
    assert torch.equal(encoder, artifact.encoder_weight)
    assert torch.equal(decoder, artifact.decoder_weight)

    tampered = adapter_module._compile_moving_gauge_factor_writer_artifact(
        site,
        requested,
    )
    dense_id = "model.layers.0.feed_forward.w2.weight"
    dense_parameter = torch.nn.Parameter(
        torch.zeros_like(requested),
        requires_grad=False,
    )
    dense_site = adapter_module._MovingGaugeWriterSite(
        parameter_id=dense_id,
        topology="dense",
        writer_rank=None,
        writer_contract_sha256=_digest("moving-gauge-dense-contract"),
        seed_physical_state_sha256=_digest("moving-gauge-dense-seed"),
        physical_parameter_ids=(dense_id,),
        physical_parameters=(dense_parameter,),
    )
    dense_replacement = torch.ones_like(requested)
    before_dense = dense_parameter.detach().clone()
    before_encoder = encoder.detach().clone()
    before_decoder = decoder.detach().clone()
    tampered.encoder_weight[0, 0] += torch.tensor(1, dtype=torch.bfloat16)
    with pytest.raises(
        Lfm2DynamicClosedLoopAdapterError,
        match="changed after sealing",
    ):
        adapter._publish_moving_gauge_writer_replacements(
            MappingProxyType(
                {
                    dense_id: dense_site,
                    parameter_id: site,
                }
            ),
            MappingProxyType(
                {
                    dense_id: dense_replacement,
                    parameter_id: requested,
                }
            ),
            (dense_id, parameter_id),
            MappingProxyType({parameter_id: tampered}),
        )
    assert torch.equal(dense_parameter, before_dense)
    assert torch.equal(encoder, before_encoder)
    assert torch.equal(decoder, before_decoder)


@pytest.mark.parametrize("factor_writer", [False, True], ids=["dense", "factor"])
def test_dense_operator_prepares_complete_bridged_heterogeneous_closed_loop(
    tmp_path: Path,
    factor_writer: bool,
) -> None:
    case = _bridged_heterogeneous_case(
        tmp_path,
        factor_writer=factor_writer,
    )
    parent = case["parent"]
    graph = case["graph"]
    materialized = case["materialized"]
    child = materialized.model
    selected = materialized.adapter_payload
    trace_cache = case["trace_cache"]
    child_state_before = fingerprint_model_state(child)
    parent_state_before = fingerprint_model_state(parent)
    parameter_topology_before = tuple(
        (name, id(parameter), tuple(parameter.shape))
        for name, parameter in child.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    )
    planner = DynamicFoldPlanner(ledger=ExperimentLedger())
    action = planner._action(
        index=0,
        method_id=Lfm2DenseBudgetModelOperator.method_id,
        parent_graphs=(graph,),
        parent_state_fingerprints={"parent": parent_state_before},
        source_groups=((graph, graph.nodes),),
        target_nodes=materialized.graph.nodes,
        child_graph=materialized.graph,
        child_state_fingerprint=child_state_before,
        evidence_fingerprint=trace_cache.cache_key,
        dependencies=(),
        configuration={
            "catalog_id": selected.catalog_id,
            "source_depth": len(graph.layer_indices),
            "target_depth": selected.target_depth,
            "source_spans": encode_lfm2_source_spans(selected.ordered_source_spans),
            "target_layer_types": encode_lfm2_layer_types(selected.layer_types),
            "target_hidden_size": selected.hidden_size,
            "target_intermediate_size": selected.intermediate_size,
            "target_query_heads": selected.query_heads,
            "target_key_value_heads": selected.key_value_heads,
            "target_head_dim": selected.head_dim,
            "tokenizer_bridge_required": True,
            "tokenizer_bridge_artifact_sha256": (
                selected.tokenizer_bridge_artifact_sha256
            ),
            "tokenizer_transport_profile_sha256": (
                selected.tokenizer_transport_profile_sha256
            ),
            "parent_tokenizer_semantics_sha256": (
                selected.parent_tokenizer_semantics_sha256
            ),
            "child_tokenizer_semantics_sha256": (
                selected.child_tokenizer_semantics_sha256
            ),
            "fit_objective": "cached_closed_loop_heterogeneous_fold",
            "selection_policy": "authenticated_heterogeneous_exact_budget",
            "promotion_gate": "external_whole_output_review",
            "zero_gradient": True,
        },
    )
    context = FoldExecutionContext(
        parent_models={"parent": parent},
        child_model=child,
        parent_graphs={"parent": graph},
        child_graph=materialized.graph,
    )
    provider = Lfm2DenseBudgetEvidenceProvider(
        trace_cache,
        trace_cache.cache_key,
    )
    operator = Lfm2DenseBudgetModelOperator(
        selected,
        trace_cache,
        config=Lfm2DenseBudgetFoldConfig(
            residual_basis_mode="weight_complete",
            run_closed_loop=True,
            body_reclosure_mode=(
                LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3
                if factor_writer
                else "disabled"
            ),
            closed_loop_adapter_config=(
                Lfm2DynamicClosedLoopAdapterConfig(
                    ridge=1e-5,
                    norm_ridge=1e-5,
                    max_prior_delta_ratio=4.0,
                    cluster_max_iterations=4,
                    cluster_distance_chunk_size=16,
                    minimum_layer_relative_rmse_improvement=-1_000_000.0,
                    minimum_final_norm_relative_rmse_improvement=(-1_000_000.0),
                    minimum_output_kl_improvement=-1_000_000.0,
                    minimum_output_known_sequence_nll_improvement=(-1_000_000.0),
                    output_max_prior_delta_ratio=4.0,
                    output_validation_tolerance=1_000_000.0,
                    replay_capture_mode=(CHILD_REPLAY_PREFIX_BY_PREFIX_MODE),
                )
            ),
            closed_loop_compiler_config=ClosedLoopCompilerConfig(
                validation_fraction=0.5,
                split_seed="bridged-heterogeneous-closed-loop-v1",
                fold_body_reclosure=factor_writer,
            ),
        ),
        tokenizer_bridge_artifact=case["bridge"],
        tokenizer_bridge_trace_adapter=case["trace_adapter"],
    )
    evidence = provider.prepare(action, context)

    with observe_model_execution(parent) as parent_execution:
        transaction, fold_only_audit = run_fold_only_callback(
            lambda: operator.prepare(action, context, evidence),
            phase="test bridged heterogeneous dense operator prepare",
            protected_models=(parent,),
        )

    report = transaction.plan.as_dict()
    closed_loop = report["closed_loop"]
    closed_loop_audit = closed_loop["audit"]

    def nested_mappings(value):
        if isinstance(value, dict):
            yield value
            for nested in value.values():
                yield from nested_mappings(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from nested_mappings(nested)

    factor_install_markers = [
        value["short_conv_factor_consumer_installed"]
        for value in nested_mappings(closed_loop)
        if "short_conv_factor_consumer_installed" in value
    ]
    assert transaction.active is True
    assert fingerprint_model_state(child) != child_state_before
    assert fingerprint_model_state(parent) == parent_state_before
    assert parent_execution["module_forward_calls"] == 0
    assert parent_execution["root_forward_calls"] == 0
    assert fold_only_audit.clean is True
    assert fold_only_audit.optimizer_constructions == 0
    assert fold_only_audit.optimizer_steps == 0
    assert fold_only_audit.backward_calls == 0
    assert fold_only_audit.autograd_grad_calls == 0
    assert report["closed_loop_enabled"] is True
    assert report["tokenizer_bridge_required"] is True
    assert report["tokenizer_bridge_artifact_sha256"] == (
        case["bridge"].artifact_sha256
    )
    assert report["complete_target_parameter_overwrite"] is True
    assert report["parameter_topology_preserved"] is True
    assert report["parameter_count_preserved"] is True
    assert report["training_used"] is False
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0
    assert closed_loop_audit["parent_forward_calls"] == 0
    assert closed_loop_audit["parent_trace_regeneration_calls"] == 0
    assert closed_loop_audit["training_steps"] == 0
    assert closed_loop_audit["backward_calls"] == 0
    assert closed_loop_audit["optimizer_steps"] == 0
    assert any(factor_install_markers) is factor_writer
    if factor_writer:
        body_reclosure = closed_loop["body_reclosure"]
        assert body_reclosure["compiled"]["metadata"][
            "body_reclosure_mode"
        ] == LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3
        assert body_reclosure["compiled"]["metadata"][
            "factor_writer_count"
        ] == 1
        assert body_reclosure["compiled"]["metadata"][
            "factor_writer_artifact_sha256"
        ]
        assert body_reclosure["compiled"]["metadata"][
            "causal_action_aware_factor_writer_count"
        ] == 1
        assert body_reclosure["install"]["metadata"][
            "factor_writer_count"
        ] == 1
        assert body_reclosure["install"]["metadata"][
            "action_aware_factor_writer_count"
        ] == 1
        assert body_reclosure["install"]["metadata"][
            "physical_writer_parameter_count"
        ] == 9
        assert body_reclosure["install"]["metadata"][
            "physical_contract_preserved"
        ] is True
        assert body_reclosure["validation"]["metrics"][
            "factor_writer_count"
        ] == 1
        assert body_reclosure["validation"]["metrics"][
            "action_aware_factor_writer_count"
        ] == 1
    assert operator.config.closed_loop_adapter_config.replay_capture_mode == (
        CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
    )
    assert child.get_input_embeddings().weight is not (
        child.get_output_embeddings().weight
    )
    assert tuple(child.get_input_embeddings().weight.shape) == (17, 16)
    assert tuple(child.get_output_embeddings().weight.shape) == (17, 16)
    expected_parameters = 8_690 if factor_writer else 8_798
    assert selected.exact_target_unique_parameters == expected_parameters
    assert graph.unique_parameter_count > 4 * expected_parameters
    assert materialized.graph.unique_parameter_count == expected_parameters
    assert (
        sum(
            parameter.numel()
            for parameter in {
                id(parameter): parameter for parameter in child.parameters()
            }.values()
        )
        == expected_parameters
    )
    assert (
        tuple(
            (name, id(parameter), tuple(parameter.shape))
            for name, parameter in child.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        )
        == parameter_topology_before
    )

    commit = transaction.commit()
    assert commit["accepted"] is True
    assert commit["parameter_topology_preserved"] is True
    descriptor = extract_lfm2_dynamic_physical_layout_from_config(child.config)
    writer = child.model.layers[0].conv.out_proj
    if factor_writer:
        assert descriptor.layers[0].short_conv_writer_rank == 3
        assert type(writer) is torch.nn.Sequential
        assert tuple(writer[0].weight.shape) == (3, 12)
        assert tuple(writer[1].weight.shape) == (16, 3)
        assert torch.count_nonzero(writer[0].weight).item() > 0
        assert torch.count_nonzero(writer[1].weight).item() > 0
        reload_root = tmp_path / "factor-closed-loop-reload"
        inputs = torch.tensor([[1, 3, 4, 5]], dtype=torch.long)
        with torch.inference_mode():
            expected_logits = child(inputs, use_cache=False).logits
        expected_state = fingerprint_model_state(child)
        residual.atomic_save_local_checkpoint(
            child,
            reload_root,
            output_root=tmp_path,
        )
        loaded = load_lfm2_direct_local_model(
            reload_root,
            device="cpu",
            dtype=torch.float32,
        )
        loaded_descriptor = extract_lfm2_dynamic_physical_layout_from_config(
            loaded.config
        )
        with torch.inference_mode():
            actual_logits = loaded(inputs, use_cache=False).logits
        assert loaded_descriptor == descriptor
        assert fingerprint_model_state(loaded) == expected_state
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        assert type(loaded.model.layers[0].conv.out_proj) is torch.nn.Sequential
    else:
        assert descriptor.layers[0].short_conv_writer_rank is None
        assert type(writer) is torch.nn.Linear


def test_heterogeneous_replay_policy_uses_complete_numeric_inventory() -> None:
    _parent_model, _graph, _candidate, materialized = _materialize_case()
    child = materialized.model
    child.register_buffer(
        "numeric_policy_float64_probe",
        torch.ones(1, dtype=torch.float64),
    )

    numeric = trace_cache.compile_lfm2_child_replay_numeric_policy(
        child,
        input_device="cpu",
    )
    policy = adapter_module._heterogeneous_single_pass_equivalence_policy(
        child,
        "cpu",
    )

    target_inventory = numeric["numeric_execution_policy"]["target_inventory"]
    assert target_inventory["floating_parameter_dtype_counts"] == {
        "float32": target_inventory["parameter_binding_count"]
    }
    assert target_inventory["floating_buffer_dtype_counts"]["float64"] == 1
    assert target_inventory["mixed_floating_storage_dtypes"] is True
    strict_product = 8 * (2.0**-24)
    assert policy["rtol"] == strict_product / (1.0 - strict_product)
    assert policy["atol"] == 0.0
    assert policy["top1_required"] is True
    assert policy["byte_exact_required"] is False
    assert policy["logit_error_model"] == trace_cache.CHILD_REPLAY_LOGIT_ERROR_MODEL
    assert policy["non_logit_error_model"] == (
        trace_cache.CHILD_REPLAY_NON_LOGIT_ERROR_MODEL
    )
    assert policy["non_logit_tolerance_unchanged"] is False
    assert policy["numeric_policy_sha256"] == numeric["numeric_policy_sha256"]
    assert policy["replay_numeric_policy_sha256"] == (
        numeric["replay_numeric_policy_sha256"]
    )
    assert policy["non_logit_envelope_sha256"] == (
        numeric["non_logit_envelope_sha256"]
    )
