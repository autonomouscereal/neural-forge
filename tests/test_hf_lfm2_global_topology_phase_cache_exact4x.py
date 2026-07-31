from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_dynamic_lfm2_heterogeneous_budgeted_materialization import (  # noqa: E402
    _parent,
)
from test_hf_lfm2_dynamic_closed_loop_adapter import _batch  # noqa: E402
from test_hf_lfm2_heterogeneous_layout_selection import (  # noqa: E402
    _SCHEDULE,
    _source_graph,
)

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache  # noqa: E402
import neural_forge.hf_lfm2_cached_global_topology_frontier as global_frontier  # noqa: E402
from neural_forge.hf_lfm2_cached_functional_width import (  # noqa: E402
    Lfm2CachedFunctionalWidthEvidence,
    Lfm2FunctionalWidthScore,
    enumerate_lfm2_global_hidden_sizes,
)
from neural_forge.hf_lfm2_cached_global_topology_frontier import (  # noqa: E402
    GLOBAL_TOPOLOGY_CAPACITY_PAYLOAD_SCHEMA,
    GLOBAL_TOPOLOGY_FUNCTIONAL_PAYLOAD_SCHEMA,
    Lfm2GlobalTargetRuntimePins,
    Lfm2GlobalTopologyPhaseCacheContext,
    build_lfm2_cached_global_topology_frontier,
)
from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (  # noqa: E402
    ATTENTION_GEOMETRY_ENABLED_POLICY,
    ATTENTION_GEOMETRY_RISK_METHOD,
    EXPLICIT_MLP_OPTION_POLICY,
    MLP_CAPACITY_RISK_METHOD,
    Lfm2CachedAttentionGeometryRisk,
    Lfm2CachedHeterogeneousCapacityError,
    Lfm2CachedHeterogeneousCapacityOptionBank,
    Lfm2CachedLayerCapacityEvidence,
    Lfm2CachedMlpCapacityRisk,
    derive_lfm2_automatic_mlp_width_policy,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    CACHED_VERIFIED_TRACE_MODE,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceCache,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (  # noqa: E402
    build_lfm2_attention_layer_option,
    build_lfm2_short_conv_layer_option,
    discover_lfm2_source_layers,
)


# Historical no-carrier whole/body target used only as a production-shaped
# cache-control fixture.  The integrated carrier run reserves 3,867,776
# parameters and therefore plans its dense body against 53,552,368 instead.
NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS = 57_420_144


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _runtime() -> Lfm2GlobalTargetRuntimePins:
    return Lfm2GlobalTargetRuntimePins(
        vocab_size=4_096,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        target_vocab_evidence_sha256=_digest("exact4x-target-vocabulary"),
        token_bridge_evidence_sha256=_digest("exact4x-token-bridge"),
    )


def _explicit_union_bank(
    *,
    graph,
    parent_cache,
    policies,
) -> Lfm2CachedHeterogeneousCapacityOptionBank:
    discovered = discover_lfm2_source_layers(graph)
    union_widths = tuple(
        sorted(
            {
                width
                for policy in policies.values()
                for widths in policy.widths_by_layer
                for width in widths
            }
        )
    )
    mlp_risks = tuple(
        Lfm2CachedMlpCapacityRisk(
            intermediate_size=width,
            source_intermediate_size=2_560,
            fold_method_id=MLP_CAPACITY_RISK_METHOD,
            post_swiglu_tail_fraction=((2_560 - width) / 2_560) ** 2,
            signed_regime_tail_fraction=0.0,
            tangent_tail_fraction=0.0,
            downstream_weighted_tail_fraction=0.0,
            collision_fraction=0.0,
            predicted_defect=(2_560 - width) / 2_560,
        )
        for width in union_widths
    )
    attention_risks = tuple(
        Lfm2CachedAttentionGeometryRisk(
            query_heads=query,
            key_value_heads=key_value,
            head_dim=64,
            source_query_heads=16,
            source_key_value_heads=8,
            source_head_dim=64,
            fold_method_id=ATTENTION_GEOMETRY_RISK_METHOD,
            coupled_group_tail_fraction=defect,
            query_offset_tail_fraction=0.0,
            causal_score_tail_fraction=0.0,
            context_tail_fraction=0.0,
            key_value_memory_tail_fraction=0.0,
            predicted_defect=defect,
            causal_query_row_count=256,
            cached_key_value_row_count=256,
        )
        for query, key_value, defect in ((12, 6, 0.05), (16, 8, 0.0))
    )
    evidence = tuple(
        Lfm2CachedLayerCapacityEvidence(
            layer_index=layer.ordinal,
            layer_type=layer.layer_type,
            source_layer_sha256=layer.source_layer_sha256,
            operator_delta_energy_fraction=1.0,
            mlp_delta_energy_fraction=1.0,
            sampled_scalar_count=256,
            nonlinear_sampled_row_count=256,
            source_mlp_intermediate_size=2_560,
            mlp_capacity_risks=mlp_risks,
            attention_geometry_risks=(
                attention_risks
                if layer.layer_type == "full_attention"
                else ()
            ),
            selection_evidence_sha256=_digest(
                f"exact4x-union-layer-{layer.ordinal}"
            ),
        )
        for layer in discovered
    )
    options = []
    for layer in discovered:
        for width in union_widths:
            structural_risk = (2_560 - width) / 2_560
            if layer.layer_type == "conv":
                for internal in (704, 1_024):
                    options.append(
                        build_lfm2_short_conv_layer_option(
                            source_layer_sha256=layer.source_layer_sha256,
                            mlp_intermediate_size=width,
                            internal_width=internal,
                            selection_evidence_sha256=_digest(
                                f"exact4x-union-{layer.ordinal}-{width}-{internal}"
                            ),
                            structural_risk=structural_risk,
                        )
                    )
            else:
                for query, key_value, defect in ((12, 6, 0.05), (16, 8, 0.0)):
                    options.append(
                        build_lfm2_attention_layer_option(
                            source_layer_sha256=layer.source_layer_sha256,
                            mlp_intermediate_size=width,
                            query_heads=query,
                            key_value_heads=key_value,
                            head_dim=64,
                            selection_evidence_sha256=_digest(
                                f"exact4x-union-{layer.ordinal}-{width}-{query}"
                            ),
                            structural_risk=structural_risk + defect,
                        )
                    )
    values = {
        "source_graph_fingerprint": graph.fingerprint,
        "parent_cache_key": parent_cache.cache_key,
        "mlp_option_policy": EXPLICIT_MLP_OPTION_POLICY,
        "attention_geometry_policy": ATTENTION_GEOMETRY_ENABLED_POLICY,
        "exact_budget_lattice_sha256": _digest("exact4x-union-lattice"),
        "layer_evidence": evidence,
        "layer_options": tuple(options),
    }
    provisional = object.__new__(Lfm2CachedHeterogeneousCapacityOptionBank)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    return Lfm2CachedHeterogeneousCapacityOptionBank(
        **values,
        evidence_sha256=canonical_sha256(provisional._body()),
    )


def _functional_width_evidence(
    *,
    graph,
    parent_cache,
    widths,
) -> Lfm2CachedFunctionalWidthEvidence:
    eigenspectrum = _digest("exact4x-functional-eigenspectrum")
    scores = tuple(
        Lfm2FunctionalWidthScore.build(
            hidden_size=width,
            retained=1.0 if width == 816 else 0.0,
            eigenspectrum_sha256=eigenspectrum,
            selection_evidence_sha256=_digest(
                f"exact4x-functional-width-{width}"
            ),
        )
        for width in widths
    )
    roles = (
        "nonlinear_mlp_write",
        "operator_write",
        "residual_context",
        "terminal_context",
    )
    values = {
        "source_graph_fingerprint": graph.fingerprint,
        "parent_cache_key": parent_cache.cache_key,
        "selective_session_sha256": _digest("exact4x-selective-session"),
        "source_hidden_size": 1_024,
        "maximum_rows_per_role": 2_048,
        "sampled_rows_by_role": tuple((role, 256) for role in roles),
        "candidate_scores": scores,
        "role_gram_sha256": tuple(
            (role, _digest(f"exact4x-role-gram-{role}")) for role in roles
        ),
        "functional_eigenspectrum_sha256": eigenspectrum,
        "selection_manifest_sha256": _digest(
            "exact4x-functional-selection-manifest"
        ),
    }
    provisional = object.__new__(Lfm2CachedFunctionalWidthEvidence)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    return Lfm2CachedFunctionalWidthEvidence(
        **values,
        evidence_sha256=canonical_sha256(provisional._body()),
    )


def test_no_carrier_exact4x_control_h816_b3_cold_hit_frontier_equality(
    tmp_path,
    monkeypatch,
) -> None:
    parent = _parent()
    parent.config._attn_implementation = "eager"
    parent_cache = build_lfm2_circuit_trace_cache(
        tmp_path / "tiny-parent-cache",
        parent,
        [_batch((1, 2, 3, 4), (5,))],
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=2,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=CACHED_VERIFIED_TRACE_MODE,
            capture_causal_circuit_evidence=True,
        ),
        input_device="cpu",
    )
    graph = _source_graph(_SCHEDULE, intermediate=2_560)
    runtime = _runtime()
    widths = enumerate_lfm2_global_hidden_sizes(
        graph,
        exact_target_unique_parameters=(
            NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS
        ),
        target_vocabulary_size=runtime.vocab_size,
        minimum_hidden_size=816,
        hidden_size_multiple=8,
    )
    policies = {}
    for hidden_size in widths:
        try:
            policies[hidden_size] = derive_lfm2_automatic_mlp_width_policy(
                layer_types=_SCHEDULE,
                source_widths=(2_560,) * len(_SCHEDULE),
                short_conv_internal_widths=(704, 1_024),
                attention_head_options=((12, 6, 64), (16, 8, 64)),
                target_hidden_size=hidden_size,
                target_vocab_size=runtime.vocab_size,
                exact_target_unique_parameters=(
                    NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS
                ),
                conv_kernel_size=runtime.conv_kernel_size,
            )
        except Lfm2CachedHeterogeneousCapacityError as error:
            if not global_frontier._policy_is_unattainable(error):
                raise
    assert tuple(policies) == (816, 840)
    union_widths = tuple(
        sorted(
            {
                width
                for policy in policies.values()
                for row in policy.widths_by_layer
                for width in row
            }
        )
    )
    bank = _explicit_union_bank(
        graph=graph,
        parent_cache=parent_cache,
        policies=policies,
    )
    functional = _functional_width_evidence(
        graph=graph,
        parent_cache=parent_cache,
        widths=widths,
    )
    progress = []
    context = Lfm2GlobalTopologyPhaseCacheContext(
        cache=DynamicFoldPhaseCache(tmp_path / "derived-phases"),
        evidence_split_sha256=_digest("exact4x-evidence-splits"),
        progress=lambda stage, details: progress.append((stage, dict(details))),
    )
    capacity_key = global_frontier._capacity_phase_key(
        context,
        graph,
        parent_cache,
        runtime=runtime,
        exact_target_unique_parameters=(
            NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS
        ),
        candidate_hidden_sizes=widths,
        union_mlp_widths=union_widths,
        union_writer_ranks=(),
        short_conv_internal_widths=(704, 1_024),
        attention_head_options=((12, 6, 64), (16, 8, 64)),
        discover_exact_short_conv_writer_ranks=False,
    )
    global_frontier._publish_phase_payload(
        context,
        capacity_key,
        schema=GLOBAL_TOPOLOGY_CAPACITY_PAYLOAD_SCHEMA,
        body={"capacity_option_bank": dict(bank.as_dict())},
        result={
            "evidence_sha256": bank.evidence_sha256,
            "layer_count": len(bank.layer_evidence),
            "layer_option_count": len(bank.layer_options),
        },
    )
    functional_key = global_frontier._functional_phase_key(
        context,
        graph,
        parent_cache,
        runtime=runtime,
        candidate_hidden_sizes=widths,
        maximum_rows_per_role=2_048,
    )
    global_frontier._publish_phase_payload(
        context,
        functional_key,
        schema=GLOBAL_TOPOLOGY_FUNCTIONAL_PAYLOAD_SCHEMA,
        body={"functional_width_evidence": dict(functional.as_dict())},
        result={
            "evidence_sha256": functional.evidence_sha256,
            "candidate_score_count": len(functional.candidate_scores),
        },
    )

    def forbidden_session(_self):
        raise AssertionError(
            "seeded no-carrier control evidence must not reopen trace payloads"
        )

    # ``_trace_cache_topology_interface_fingerprint`` binds
    # ``Lfm2CircuitTraceCache.selective_authenticated_session`` by source, so
    # stubbing that reader would otherwise move every derived phase key and turn
    # the seeded hit into a cold miss.  Pin the real reader ABI first, exactly as
    # ``test_global_frontier_phase_cache_resumes_without_evidence_rescan_and_fails_closed``
    # does, so the stub only proves that no trace payload is reopened.
    trace_cache_interface_sha256 = (
        global_frontier._trace_cache_topology_interface_fingerprint()
    )
    monkeypatch.setattr(
        global_frontier,
        "_trace_cache_topology_interface_fingerprint",
        lambda: trace_cache_interface_sha256,
    )
    monkeypatch.setattr(
        Lfm2CircuitTraceCache,
        "selective_authenticated_session",
        forbidden_session,
    )
    kwargs = {
        "runtime": runtime,
        "exact_target_unique_parameters": (
            NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS
        ),
        "minimum_hidden_size": 816,
        "short_conv_internal_widths": (704, 1_024),
        "attention_head_options": ((12, 6, 64), (16, 8, 64)),
        "maximum_functional_rows_per_role": 2_048,
        "max_states": 100_000,
        "phase_cache_context": context,
    }
    cold = build_lfm2_cached_global_topology_frontier(
        graph,
        parent_cache,
        **kwargs,
    )
    hit_progress = []
    hot = build_lfm2_cached_global_topology_frontier(
        graph,
        parent_cache,
        **{
            **kwargs,
            "phase_cache_context": Lfm2GlobalTopologyPhaseCacheContext(
                cache=DynamicFoldPhaseCache(tmp_path / "derived-phases"),
                evidence_split_sha256=_digest("exact4x-evidence-splits"),
                progress=lambda stage, details: hit_progress.append(
                    (stage, dict(details))
                ),
            ),
        },
    )

    assert cold.selected_input.target_pins.hidden_size == 816
    assert [value.hidden_size for value in cold.candidates] == [816, 840]
    assert all(
        value.exact_unique_parameters
        == NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS
        for value in cold.candidates
    )
    assert len(cold.selected_layout.descriptor.layers) == 14
    assert cold.selected_layout.descriptor.accounting.unique_parameter_elements == (
        NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS
    )
    assert dict(hot.as_dict()) == dict(cold.as_dict())
    assert hot.frontier_sha256 == cold.frontier_sha256
    reconstructed = next(
        details
        for stage, details in hit_progress
        if stage == "global_topology_selected_width_reconstructed"
    )
    assert reconstructed["selector_width_count"] == 1
    assert reconstructed["selected_hidden_size"] == 816
