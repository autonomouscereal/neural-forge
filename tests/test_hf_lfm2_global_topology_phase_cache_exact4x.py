from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

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

from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from neural_forge.dynamic_fold_phase_cache import (  # noqa: E402
    DynamicFoldPhaseCache,
    FoldPhaseKey,
)
import neural_forge.dynamic_fold_phase_cache as phase_cache_module  # noqa: E402
import neural_forge.hf_lfm2_cached_global_topology_frontier as global_frontier  # noqa: E402
import neural_forge.hf_lfm2_heterogeneous_layout_selection as layout_selection_module  # noqa: E402
from neural_forge.hf_lfm2_cached_functional_width import (  # noqa: E402
    Lfm2CachedFunctionalWidthEvidence,
    Lfm2FunctionalWidthScore,
    enumerate_lfm2_global_hidden_sizes,
)
from neural_forge.hf_lfm2_cached_global_topology_frontier import (  # noqa: E402
    GLOBAL_TOPOLOGY_CAPACITY_PHASE,
    GLOBAL_TOPOLOGY_CAPACITY_PAYLOAD_SCHEMA,
    GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
    GLOBAL_TOPOLOGY_FUNCTIONAL_PAYLOAD_SCHEMA,
    GLOBAL_TOPOLOGY_SELECTION_PHASE,
    Lfm2ExistingGlobalTopologyPhaseReuseRequest,
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
_PUBLISHED_GLOBAL_ALGORITHM_SHA256 = (
    "3e4531740792b3abe7ed54c02a49ca99caabc3ae94dd181c167913cbb5dbcab1"
)
_A0007_FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "lfm2_global_topology_da5b9da6"
)
_A0007_PHASE_IDENTITIES = {
    GLOBAL_TOPOLOGY_CAPACITY_PHASE: {
        "phase_key_sha256": (
            "ba5a3c048a3799f16b5af72f9fad41237eac6592641f542bd609360e035bedbb"
        ),
        "completion_sha256": (
            "df71a12d8c197047e8803a1626921da00b66ff14cc32e63c8e149c914bea1bb4"
        ),
        "completion_file_sha256": (
            "ceff5a63da365a40d388e6c8d814e1dbd965c6120849df78b72cfaeb0190ae7a"
        ),
        "payload_sha256": (
            "31331fe7900f2e38191ca55a9dcfa90be386b880794fbdf718db4b818b47680e"
        ),
        "payload_file_sha256": (
            "14aa3093bcb3fbfa6473485f90d9b5b128fb572f1c294de1c2330fa9579f1c26"
        ),
        "evidence_sha256": (
            "64ae4e0f61ae8ea94f0468c23792216244ac72724fa88867cf107ee545165d8d"
        ),
        "evidence_field": "capacity_option_bank",
    },
    GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE: {
        "phase_key_sha256": (
            "bcabf47e97c1fe6e9ee6cdcbbb3c7b8c7183ae9df29c3c55faacb868e4cd0348"
        ),
        "completion_sha256": (
            "6f1671d9b76616341863d5b5dab4093faa5eb59c0cabfe011840bbdaf3a0c4c3"
        ),
        "completion_file_sha256": (
            "86b56914963835e1de3d1a3a0140eb78320baa03cb015ba54f77815dbe58a1bd"
        ),
        "payload_sha256": (
            "b264bb7cd9caa9d8476deaeed0c10c781579d1d36216724721126b38937ef7b4"
        ),
        "payload_file_sha256": (
            "aa3b3cdc568717521b8797169b549edb8db2318230c4470eaf67128b9fd88740"
        ),
        "evidence_sha256": (
            "eac6f7ab2518de7226efe183be38c70cb53b3d09f5f841ee9d037bcbbffe965e"
        ),
        "evidence_field": "functional_width_evidence",
    },
}


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


def _copy_platform_normalized_a0007_phase(
    phase_root: Path,
    phase_name: str,
) -> tuple[FoldPhaseKey, dict, dict]:
    """Copy golden bytes, changing only copied mode inventory/seals if needed."""

    identity = _A0007_PHASE_IDENTITIES[phase_name]
    phase_key_sha256 = identity["phase_key_sha256"]
    source = _A0007_FIXTURE_ROOT / phase_name / phase_key_sha256
    destination = phase_root / phase_name / phase_key_sha256
    shutil.copytree(source, destination)
    payload_path = destination / "payload" / "evidence.json"
    completion_path = destination / phase_cache_module.COMPLETE_FILENAME
    payload = json.loads(payload_path.read_text(encoding="ascii"))
    completion = json.loads(completion_path.read_text(encoding="ascii"))

    # Git/NTFS does not preserve the fixture's Linux 0664 payload mode.  The
    # committed source bytes and their original completion are asserted
    # separately; only this temporary copy receives a platform-local inventory
    # and completion seal.  Its phase key and evidence bytes remain unchanged.
    local_inventory = phase_cache_module._payload_inventory(
        destination / phase_cache_module.PAYLOAD_DIRECTORY_NAME
    )
    source_inventory = completion["payload_inventory"]
    normalized_inventory = json.loads(json.dumps(source_inventory))
    local_modes = {
        record["path"]: record["mode"] for record in local_inventory["files"]
    }
    for record in normalized_inventory["files"]:
        record["mode"] = local_modes[record["path"]]
    inventory_body = {
        key: value
        for key, value in normalized_inventory.items()
        if key != "payload_tree_sha256"
    }
    normalized_inventory["payload_tree_sha256"] = canonical_sha256(inventory_body)
    assert normalized_inventory == local_inventory
    completion["payload_inventory"] = normalized_inventory
    completion_body = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    completion["completion_sha256"] = canonical_sha256(completion_body)
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")
    phase_key = FoldPhaseKey.from_dict(completion["phase_key"])
    return phase_key, completion, payload


def _payload_inventory_without_derived_tree_or_modes(value: dict) -> dict:
    normalized = json.loads(json.dumps(value))
    normalized.pop("payload_tree_sha256")
    for record in normalized["files"]:
        record.pop("mode")
    return normalized


def _fixture_parent_cache(
    tmp_path: Path,
    historical_completion: dict,
) -> Lfm2CircuitTraceCache:
    dependencies = historical_completion["phase_key"]["dependency_sha256s"]
    manifest = {
        "cache_key": dependencies["parent_cache_key"],
        "manifest_sha256": dependencies["parent_manifest"],
        "parent": {"identity_sha256": dependencies["parent_identity"]},
        "corpus": {
            "corpus_sha256": dependencies["parent_corpus"],
            "batch_count": 1,
        },
        "layer_topology": {"layer_topology_sha256": dependencies["parent_topology"]},
        "implementation": {
            "implementation_sha256": dependencies["parent_implementation"],
            "collector_source_sha256": dependencies["collector_source"],
            "canonical_contract_source_sha256": dependencies[
                "canonical_contract_source"
            ],
            "transformers_lfm2_source_sha256": dependencies[
                "transformers_lfm2_runtime_source"
            ],
        },
        "execution": {"execution_sha256": dependencies["parent_execution"]},
        "capture": {"selection_sha256": dependencies["parent_capture_selection"]},
    }
    return Lfm2CircuitTraceCache(
        path=tmp_path / "parent-cache-must-not-be-read",
        _manifest=MappingProxyType(manifest),
        _completion=MappingProxyType(
            {"completion_sha256": dependencies["parent_completion"]}
        ),
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
                attention_risks if layer.layer_type == "full_attention" else ()
            ),
            selection_evidence_sha256=_digest(f"exact4x-union-layer-{layer.ordinal}"),
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
            selection_evidence_sha256=_digest(f"exact4x-functional-width-{width}"),
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
        "selection_manifest_sha256": _digest("exact4x-functional-selection-manifest"),
    }
    provisional = object.__new__(Lfm2CachedFunctionalWidthEvidence)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    return Lfm2CachedFunctionalWidthEvidence(
        **values,
        evidence_sha256=canonical_sha256(provisional._body()),
    )


def test_authenticated_a0007_da5_phases_restore_and_recompute_current_selection(
    tmp_path,
    monkeypatch,
) -> None:
    golden_documents = {}
    golden_raw_sha256s = {}
    for phase_name, identity in _A0007_PHASE_IDENTITIES.items():
        source_entry = _A0007_FIXTURE_ROOT / phase_name / identity["phase_key_sha256"]
        completion_path = source_entry / phase_cache_module.COMPLETE_FILENAME
        payload_path = source_entry / "payload" / "evidence.json"
        completion_raw = completion_path.read_bytes()
        payload_raw = payload_path.read_bytes()
        completion = json.loads(completion_raw.decode("ascii"))
        payload = json.loads(payload_raw.decode("ascii"))
        assert completion_raw == canonical_json_bytes(completion) + b"\n"
        completion_body = {
            key: value
            for key, value in completion.items()
            if key != "completion_sha256"
        }
        assert completion["completion_sha256"] == canonical_sha256(completion_body)
        golden_inventory = completion["payload_inventory"]
        golden_inventory_body = {
            key: value
            for key, value in golden_inventory.items()
            if key != "payload_tree_sha256"
        }
        assert golden_inventory["payload_tree_sha256"] == canonical_sha256(
            golden_inventory_body
        )
        assert (
            hashlib.sha256(completion_raw).hexdigest()
            == identity["completion_file_sha256"]
        )
        assert (
            hashlib.sha256(payload_raw).hexdigest() == identity["payload_file_sha256"]
        )
        assert completion["phase_key_sha256"] == identity["phase_key_sha256"]
        assert completion["completion_sha256"] == identity["completion_sha256"]
        assert (
            completion["phase_key"]["dependency_sha256s"]["global_topology_algorithms"]
            == _PUBLISHED_GLOBAL_ALGORITHM_SHA256
        )
        assert payload["payload_sha256"] == identity["payload_sha256"]
        assert (
            payload[identity["evidence_field"]]["evidence_sha256"]
            == identity["evidence_sha256"]
        )
        golden_documents[phase_name] = (completion, payload)
        golden_raw_sha256s[phase_name] = (
            hashlib.sha256(completion_raw).hexdigest(),
            hashlib.sha256(payload_raw).hexdigest(),
        )

    phase_root = tmp_path / "a0007-platform-local-phase-cache"
    local_phases = {
        phase_name: _copy_platform_normalized_a0007_phase(phase_root, phase_name)
        for phase_name in _A0007_PHASE_IDENTITIES
    }
    capacity_historical_key, capacity_completion, capacity_payload = local_phases[
        GLOBAL_TOPOLOGY_CAPACITY_PHASE
    ]
    functional_historical_key, functional_completion, functional_payload = local_phases[
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE
    ]
    for phase_name in _A0007_PHASE_IDENTITIES:
        golden_completion = golden_documents[phase_name][0]
        local_completion = local_phases[phase_name][1]
        golden_inventory = golden_completion["payload_inventory"]
        local_inventory = local_completion["payload_inventory"]
        golden_non_inventory = {
            key: value
            for key, value in golden_completion.items()
            if key not in {"completion_sha256", "payload_inventory"}
        }
        local_non_inventory = {
            key: value
            for key, value in local_completion.items()
            if key not in {"completion_sha256", "payload_inventory"}
        }
        assert local_non_inventory == golden_non_inventory
        assert _payload_inventory_without_derived_tree_or_modes(
            local_inventory
        ) == _payload_inventory_without_derived_tree_or_modes(golden_inventory)
        golden_files = {record["path"]: record for record in golden_inventory["files"]}
        local_files = {record["path"]: record for record in local_inventory["files"]}
        assert set(local_files) == set(golden_files)
        for relative_path in golden_files:
            assert {
                key: value
                for key, value in local_files[relative_path].items()
                if key != "mode"
            } == {
                key: value
                for key, value in golden_files[relative_path].items()
                if key != "mode"
            }
    graph = _source_graph(_SCHEDULE, intermediate=2_560)
    fixture_graph_sha256 = capacity_payload["capacity_option_bank"][
        "source_graph_fingerprint"
    ]
    fixture_layer_evidence = capacity_payload["capacity_option_bank"]["layer_evidence"]
    discovered = discover_lfm2_source_layers(graph)
    assert tuple(value.layer_type for value in discovered) == tuple(
        value["layer_type"] for value in fixture_layer_evidence
    )
    fixture_discovered = tuple(
        replace(
            source_layer,
            source_layer_sha256=evidence["source_layer_sha256"],
        )
        for source_layer, evidence in zip(
            discovered,
            fixture_layer_evidence,
            strict=True,
        )
    )
    historical_capacity_document = golden_documents[GLOBAL_TOPOLOGY_CAPACITY_PHASE][0]
    historical_dependencies = historical_capacity_document["phase_key"][
        "dependency_sha256s"
    ]
    historical_numeric_runtime = historical_capacity_document["phase_key"][
        "numeric_contract"
    ]["numeric_runtime"]
    parent_cache = _fixture_parent_cache(tmp_path, historical_capacity_document)
    runtime = _runtime()
    current_algorithm_sha256 = global_frontier._global_algorithm_fingerprint()
    compatibility_receipt = (
        global_frontier._current_historical_phase_compatibility_receipt(
            historical_algorithm_sha256=_PUBLISHED_GLOBAL_ALGORITHM_SHA256,
            current_algorithm_sha256=current_algorithm_sha256,
        )
    )
    compatibility_receipt_sha256 = str(
        compatibility_receipt["compatibility_receipt_sha256"]
    )

    monkeypatch.setattr(
        type(graph),
        "fingerprint",
        property(lambda _self: fixture_graph_sha256),
    )
    monkeypatch.setattr(
        Lfm2GlobalTargetRuntimePins,
        "fingerprint",
        property(lambda _self: historical_dependencies["target_runtime"]),
    )
    monkeypatch.setattr(
        global_frontier,
        "discover_lfm2_source_layers",
        lambda _graph: fixture_discovered,
    )
    monkeypatch.setattr(
        layout_selection_module,
        "_discover_lfm2_source_layers",
        lambda _graph: fixture_discovered,
    )
    monkeypatch.setattr(
        global_frontier,
        "_torch_numeric_runtime_contract",
        lambda: json.loads(json.dumps(historical_numeric_runtime)),
    )
    monkeypatch.setattr(
        global_frontier,
        "_global_algorithm_fingerprint",
        lambda: current_algorithm_sha256,
    )
    monkeypatch.setattr(
        global_frontier,
        "build_lfm2_cached_heterogeneous_capacity_option_bank",
        lambda *_args, **_kwargs: pytest.fail(
            "authenticated A0007 capacity hit attempted a current builder"
        ),
    )
    monkeypatch.setattr(
        global_frontier,
        "build_lfm2_cached_functional_width_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "authenticated A0007 functional hit attempted a current builder"
        ),
    )
    monkeypatch.setattr(
        Lfm2CircuitTraceCache,
        "selective_authenticated_session",
        lambda *_args, **_kwargs: pytest.fail(
            "authenticated A0007 hits attempted to open a trace session"
        ),
    )
    monkeypatch.setattr(
        torch.nn.Module,
        "_call_impl",
        lambda *_args, **_kwargs: pytest.fail(
            "authenticated A0007 selection attempted model execution"
        ),
    )
    capacity_reuse = Lfm2ExistingGlobalTopologyPhaseReuseRequest(
        phase_name=GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        phase_key_sha256=capacity_historical_key.phase_key_sha256,
        expected_completion_sha256=capacity_completion["completion_sha256"],
        expected_evidence_sha256=_A0007_PHASE_IDENTITIES[
            GLOBAL_TOPOLOGY_CAPACITY_PHASE
        ]["evidence_sha256"],
        expected_compatibility_receipt_sha256=compatibility_receipt_sha256,
    )
    functional_reuse = Lfm2ExistingGlobalTopologyPhaseReuseRequest(
        phase_name=GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        phase_key_sha256=functional_historical_key.phase_key_sha256,
        expected_completion_sha256=functional_completion["completion_sha256"],
        expected_evidence_sha256=_A0007_PHASE_IDENTITIES[
            GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE
        ]["evidence_sha256"],
        expected_compatibility_receipt_sha256=compatibility_receipt_sha256,
    )
    runner_script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "run_lfm2_dynamic_catalog.py"
    )
    runner_spec = importlib.util.spec_from_file_location(
        "run_lfm2_dynamic_catalog_exact4x_reuse",
        runner_script,
    )
    assert runner_spec is not None and runner_spec.loader is not None
    runner = importlib.util.module_from_spec(runner_spec)
    sys.modules[runner_spec.name] = runner
    runner_spec.loader.exec_module(runner)
    progress_stream = io.StringIO()
    progress_journal = runner._ProgressJournal(
        "0" * 64,
        stream=progress_stream,
    )
    relay_config = SimpleNamespace(
        global_topology_phase_cache_root=phase_root,
        global_topology_existing_capacity_phase_reuse=capacity_reuse,
        global_topology_existing_functional_phase_reuse=functional_reuse,
    )
    context = runner._global_topology_phase_cache_context(
        relay_config,
        evidence_split_sha256=historical_dependencies["evidence_split"],
        progress=progress_journal,
        selective_session_factory=lambda: pytest.fail(
            "authenticated A0007 hits attempted a lazy trace session"
        ),
    )
    result = build_lfm2_cached_global_topology_frontier(
        graph,
        parent_cache,
        runtime=runtime,
        exact_target_unique_parameters=53_552_368,
        required_target_hidden_size=816,
        minimum_hidden_size=816,
        short_conv_internal_widths=(704, 1_024),
        discover_exact_short_conv_writer_ranks=True,
        attention_head_options=((12, 6, 64), (16, 8, 64)),
        maximum_functional_rows_per_role=2_048,
        max_states=100_000,
        phase_cache_context=context,
    )

    assert result.selected_input.target_pins.hidden_size == 816
    assert result.exact_target_unique_parameters == 53_552_368
    assert [value.hidden_size for value in result.candidates] == [816]
    progress = [
        json.loads(line)
        for line in progress_stream.getvalue().splitlines()
        if line.strip()
    ]
    reuse_receipts = (
        progress_journal.global_topology_existing_phase_reuse_receipts()
    )
    assert [value["phase_name"] for value in reuse_receipts] == [
        GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
    ]
    for receipt in reuse_receipts:
        identity = _A0007_PHASE_IDENTITIES[receipt["phase_name"]]
        local_completion = local_phases[receipt["phase_name"]][1]
        assert receipt["historical_phase_key_sha256"] == identity["phase_key_sha256"]
        assert (
            receipt["historical_completion_sha256"]
            == local_completion["completion_sha256"]
        )
        assert receipt["historical_payload_sha256"] == identity["payload_sha256"]
        assert receipt["historical_evidence_sha256"] == identity["evidence_sha256"]
        assert receipt["compatibility_receipt_sha256"] == compatibility_receipt_sha256
    terminal_reuse = runner._global_topology_existing_phase_reuse_receipt_fields(
        relay_config,
        progress=progress_journal,
        require_all_observed=True,
    )
    assert terminal_reuse["existing_phase_reuse_observed_live_receipts"] == (
        reuse_receipts
    )
    assert terminal_reuse["existing_phase_reuse_all_requested_hits_observed"] is True
    selection_entries = list((phase_root / GLOBAL_TOPOLOGY_SELECTION_PHASE).iterdir())
    assert len(selection_entries) == 1
    selection_completion = json.loads(
        (selection_entries[0] / phase_cache_module.COMPLETE_FILENAME).read_text(
            encoding="ascii"
        )
    )
    assert (
        selection_completion["phase_key"]["dependency_sha256s"][
            "global_topology_algorithms"
        ]
        == current_algorithm_sha256
    )
    assert any(
        row["stage"] == "global_topology_phase_cache_published"
        and row["phase_name"] == GLOBAL_TOPOLOGY_SELECTION_PHASE
        for row in progress
    )
    for phase_name, identity in _A0007_PHASE_IDENTITIES.items():
        source_entry = _A0007_FIXTURE_ROOT / phase_name / identity["phase_key_sha256"]
        assert (
            hashlib.sha256(
                (source_entry / phase_cache_module.COMPLETE_FILENAME).read_bytes()
            ).hexdigest(),
            hashlib.sha256(
                (source_entry / "payload" / "evidence.json").read_bytes()
            ).hexdigest(),
        ) == golden_raw_sha256s[phase_name]


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
        exact_target_unique_parameters=(NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS),
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
    phase_root = tmp_path / "derived-phases"
    current_algorithm_sha256 = global_frontier._global_algorithm_fingerprint()
    with monkeypatch.context() as historical_patch:
        historical_patch.setattr(
            global_frontier,
            "_global_algorithm_fingerprint",
            lambda: _PUBLISHED_GLOBAL_ALGORITHM_SHA256,
        )
        historical_context = Lfm2GlobalTopologyPhaseCacheContext(
            cache=DynamicFoldPhaseCache(phase_root),
            evidence_split_sha256=_digest("exact4x-evidence-splits"),
        )
        capacity_key = global_frontier._capacity_phase_key(
            historical_context,
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
        capacity_entry, _capacity_payload, _capacity_published = (
            global_frontier._publish_phase_payload(
                historical_context,
                capacity_key,
                schema=GLOBAL_TOPOLOGY_CAPACITY_PAYLOAD_SCHEMA,
                body={"capacity_option_bank": dict(bank.as_dict())},
                result={
                    "evidence_sha256": bank.evidence_sha256,
                    "layer_count": len(bank.layer_evidence),
                    "layer_option_count": len(bank.layer_options),
                },
            )
        )
        functional_key = global_frontier._functional_phase_key(
            historical_context,
            graph,
            parent_cache,
            runtime=runtime,
            candidate_hidden_sizes=widths,
            maximum_rows_per_role=2_048,
        )
        functional_entry, _functional_payload, _functional_published = (
            global_frontier._publish_phase_payload(
                historical_context,
                functional_key,
                schema=GLOBAL_TOPOLOGY_FUNCTIONAL_PAYLOAD_SCHEMA,
                body={"functional_width_evidence": dict(functional.as_dict())},
                result={
                    "evidence_sha256": functional.evidence_sha256,
                    "candidate_score_count": len(functional.candidate_scores),
                },
            )
        )
    compatibility_receipt = (
        global_frontier._current_historical_phase_compatibility_receipt(
            historical_algorithm_sha256=_PUBLISHED_GLOBAL_ALGORITHM_SHA256,
            current_algorithm_sha256=current_algorithm_sha256,
        )
    )
    compatibility_receipt_sha256 = str(
        compatibility_receipt["compatibility_receipt_sha256"]
    )
    capacity_reuse = Lfm2ExistingGlobalTopologyPhaseReuseRequest(
        phase_name=GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        phase_key_sha256=capacity_key.phase_key_sha256,
        expected_completion_sha256=capacity_entry.completion_sha256,
        expected_evidence_sha256=bank.evidence_sha256,
        expected_compatibility_receipt_sha256=compatibility_receipt_sha256,
    )
    functional_reuse = Lfm2ExistingGlobalTopologyPhaseReuseRequest(
        phase_name=GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
        phase_key_sha256=functional_key.phase_key_sha256,
        expected_completion_sha256=functional_entry.completion_sha256,
        expected_evidence_sha256=functional.evidence_sha256,
        expected_compatibility_receipt_sha256=compatibility_receipt_sha256,
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
        global_frontier,
        "_global_algorithm_fingerprint",
        lambda: current_algorithm_sha256,
    )
    monkeypatch.setattr(
        Lfm2CircuitTraceCache,
        "selective_authenticated_session",
        forbidden_session,
    )
    monkeypatch.setattr(
        global_frontier,
        "build_lfm2_cached_heterogeneous_capacity_option_bank",
        lambda *_args, **_kwargs: pytest.fail(
            "historical capacity hit attempted a cold build"
        ),
    )
    monkeypatch.setattr(
        global_frontier,
        "build_lfm2_cached_functional_width_evidence",
        lambda *_args, **_kwargs: pytest.fail(
            "historical functional hit attempted a cold build"
        ),
    )
    progress = []
    context = Lfm2GlobalTopologyPhaseCacheContext(
        cache=DynamicFoldPhaseCache(phase_root),
        evidence_split_sha256=_digest("exact4x-evidence-splits"),
        progress=lambda stage, details: progress.append((stage, dict(details))),
        existing_capacity_phase_reuse=capacity_reuse,
        existing_functional_phase_reuse=functional_reuse,
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
                existing_capacity_phase_reuse=capacity_reuse,
                existing_functional_phase_reuse=functional_reuse,
            ),
        },
    )

    assert cold.selected_input.target_pins.hidden_size == 816
    assert [value.hidden_size for value in cold.candidates] == [816, 840]
    assert all(
        value.exact_unique_parameters == NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS
        for value in cold.candidates
    )
    assert len(cold.selected_layout.descriptor.layers) == 14
    assert cold.selected_layout.descriptor.accounting.unique_parameter_elements == (
        NO_CARRIER_EXACT4X_CONTROL_TARGET_PARAMETERS
    )
    assert dict(hot.as_dict()) == dict(cold.as_dict())
    assert hot.frontier_sha256 == cold.frontier_sha256
    reuse_receipts = [
        details
        for stage, details in progress
        if stage == "global_topology_existing_phase_reuse_hit"
    ]
    assert [value["phase_name"] for value in reuse_receipts] == [
        GLOBAL_TOPOLOGY_CAPACITY_PHASE,
        GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE,
    ]
    for receipt in reuse_receipts:
        body = dict(receipt)
        seal = body.pop("reuse_receipt_sha256")
        assert seal == canonical_sha256(body)
        assert receipt["semantic_phase_contract_exact"] is True
        assert receipt["historical_artifact_rewritten"] is False
        assert receipt["cache_miss_build_allowed"] is False
        assert receipt["current_algorithm_sha256"] == current_algorithm_sha256
        assert receipt["historical_algorithm_sha256"] == (
            _PUBLISHED_GLOBAL_ALGORITHM_SHA256
        )
    assert any(
        stage == "global_topology_phase_cache_published"
        and details["phase_name"] == GLOBAL_TOPOLOGY_SELECTION_PHASE
        for stage, details in progress
    )
    assert [
        child.name for child in (phase_root / GLOBAL_TOPOLOGY_CAPACITY_PHASE).iterdir()
    ] == [capacity_key.phase_key_sha256]
    assert [
        child.name
        for child in (phase_root / GLOBAL_TOPOLOGY_FUNCTIONAL_PHASE).iterdir()
    ] == [functional_key.phase_key_sha256]
    selection_entries = list((phase_root / GLOBAL_TOPOLOGY_SELECTION_PHASE).iterdir())
    assert len(selection_entries) == 1
    selection_completion = json.loads(
        (selection_entries[0] / "complete.json").read_text(encoding="ascii")
    )
    assert (
        selection_completion["phase_key"]["dependency_sha256s"][
            "global_topology_algorithms"
        ]
        == current_algorithm_sha256
    )
    assert context.cache.lookup(capacity_key).completion_sha256 == (
        capacity_entry.completion_sha256
    )
    assert context.cache.lookup(functional_key).completion_sha256 == (
        functional_entry.completion_sha256
    )
    assert (
        sum(
            stage == "global_topology_existing_phase_reuse_hit"
            for stage, _details in hit_progress
        )
        == 2
    )
    assert any(
        stage == "global_topology_phase_cache_hit"
        and details["phase_name"] == GLOBAL_TOPOLOGY_SELECTION_PHASE
        for stage, details in hit_progress
    )
    reconstructed = next(
        details
        for stage, details in hit_progress
        if stage == "global_topology_selected_width_reconstructed"
    )
    assert reconstructed["selector_width_count"] == 1
    assert reconstructed["selected_hidden_size"] == 816
