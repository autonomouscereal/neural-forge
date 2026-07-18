from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from neural_forge.dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_registry import ExperimentLedger
from neural_forge.dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from neural_forge.dynamic_lfm2_cpcf_adapter import (
    Lfm2CpcfCatalogError,
    Lfm2CpcfExternalBehaviorGateContract,
    Lfm2CpcfStackResources,
)
from neural_forge.dynamic_lfm2_cpcf_production import (
    Lfm2CpcfProductionBundleError,
    build_lfm2_cpcf_exact_production_bundle,
)
from neural_forge.dynamic_lfm2_cpcf_resources import (
    build_lfm2_cpcf_cache_phase_coverage_binding,
)
from neural_forge.dynamic_lfm2_cpcf_reviewed_gate import (
    Lfm2CpcfReviewedPromotionGate,
    build_cpcf_reviewed_promotion_report,
    load_lfm2_cpcf_reviewed_promotion_evidence,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import inspect_model_circuits
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
)
from neural_forge.dynamic_production_catalog import (
    ProductionDynamicFoldCatalog,
    ProductionDynamicFoldCatalogError,
    RegisteredExactTargetPortfolio,
)
from neural_forge.dynamic_search import DynamicFoldSearchObjective
from neural_forge.dynamic_target_synthesis import plan_budgeted_dynamic_target
from neural_forge.hf_lfm2_cpcf_installability import (
    LFM2_CPCF_INSTALLABLE_LAYER_SCHEMA,
    LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA,
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_circuit_trace_cache import Lfm2CircuitTraceCache
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    build_lfm2_cpcf_target_phases,
)
from neural_forge.hf_lfm2_cpcf_model_fold import Lfm2CpcfModelFoldPolicy
from neural_forge.hf_lfm2_dense_budget_model_fold import (
    DENSE_BUDGET_MODEL_METHOD,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2TargetTokenizedRecord,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (
    strict_tokenizer_bridge_configuration,
    seal_strict_tokenizer_bridge_artifact,
)
from tests.test_dynamic_lfm2_cpcf_reviewed_gate import (
    _artifact,
    _candidate_file_sha,
    _fixture,
    _rebuild_cpcf_compiler_chain,
    _seal,
)


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _parent() -> Any:
    config = transformers.Lfm2Config(
        vocab_size=81,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4201)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _installability_tensor_sha256(value: Any) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _field_inventory(tensors: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": [int(size) for size in value.shape],
            "sha256": _installability_tensor_sha256(value),
        }
        for name, value in sorted(tensors.items())
    }


def _trace_and_phases(
    bridge: Any,
    cache_key: str,
    *,
    operator_kinds: tuple[str, ...] = ("short_conv", "full_attention"),
) -> tuple[Any, tuple[Any, ...]]:
    trace = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        bridge,
        (
            Lfm2TargetTokenizedRecord(
                record_id="fit-0",
                split="fit",
                input_ids=torch.tensor([[1, 2, 3, 4]]),
                attention_mask=torch.ones((1, 4), dtype=torch.long),
                forced_decode_ids=torch.tensor([[5]]),
            ),
        ),
    )
    aligned = trace.records[0]
    layers = []
    for index, operator_kind in enumerate(operator_kinds):
        tensors = {"residual.pre": torch.zeros(1, 1, 8)}
        body = {
            "schema": LFM2_CPCF_INSTALLABLE_LAYER_SCHEMA,
            "cache_key": cache_key,
            "batch_index": 0,
            "phase": "prefill",
            "layer_index": index,
            "operator_kind": operator_kind,
            "source_width": 8,
            "field_inventory": _field_inventory(tensors),
            "functional_identities_verified": True,
        }
        layers.append(
            Lfm2CpcfInstallableLayerEvidence(
                layer_index=index,
                operator_kind=operator_kind,
                source_width=8,
                tensors=tensors,
                evidence_sha256=canonical_sha256(body),
            )
        )
    parent_phases = []
    for phase_name in ("prefill", "decode.000"):
        phase_body = {
            "schema": LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA,
            "cache_key": cache_key,
            "batch_index": 0,
            "input_sha256": aligned.donor_cache_input_sha256,
            "phase": phase_name,
            "layer_evidence_sha256": [
                layer.evidence_sha256 for layer in layers
            ],
            "final_norm_field_inventory": {},
            "parent_side_compiler_inputs_ready": True,
        }
        parent_phases.append(
            Lfm2CpcfInstallablePhaseEvidence(
                cache_key=cache_key,
                batch_index=0,
                input_sha256=aligned.donor_cache_input_sha256,
                phase=phase_name,
                layers=tuple(layers),
                final_norm_tensors={},
                evidence_sha256=canonical_sha256(phase_body),
            )
        )
    return trace, build_lfm2_cpcf_target_phases(
        trace,
        tuple(parent_phases),
    )


def _reviewed_gate_bound_to_bridge(
    tmp_path: Path,
    bridge: Any,
    *,
    target_plan: Any,
    parent_cache_key: str,
    trace_adapter_sha256: str,
    fold_policy: Lfm2CpcfModelFoldPolicy,
    phase_sha256: tuple[str, ...],
    provenance_overrides: dict[str, Any] | None = None,
) -> tuple[Lfm2CpcfReviewedPromotionGate, Any]:
    tmp_path.mkdir()
    (tmp_path / "reviewed").mkdir()
    fixture = _fixture(tmp_path / "reviewed")
    documents = fixture["documents"]

    cpcf_body = copy.deepcopy(documents["cpcf"])
    cpcf_body.pop("receipt_sha256")
    cpcf_body["inputs"]["bridge_artifact_sha256"] = bridge.artifact_sha256
    cpcf_body["inputs"]["bridge_profile_sha256"] = bridge.profile_fingerprint
    cpcf_body["inputs"]["cache_key"] = parent_cache_key
    cpcf_body["inputs"]["trace_adapter_sha256"] = trace_adapter_sha256
    certificate = target_plan.target_certificate
    selected = target_plan.frontier.candidates[
        certificate.selected_candidate_rank
    ]
    run_plan = {
        "parent_graph_fingerprint": certificate.parent_graph_fingerprint,
        "exact_target_unique_parameters": (
            target_plan.target_request.exact_unique_parameters
        ),
        "target_requested_budget": target_plan.frontier.request.as_dict(),
        "target_frontier_fingerprint": target_plan.frontier.fingerprint,
        "target_certificate_sha256": certificate.fingerprint,
        "selected_target_candidate_fingerprint": selected.fingerprint,
        "selected_target_candidate_rank": certificate.selected_candidate_rank,
    }
    if provenance_overrides is not None:
        run_plan.update(provenance_overrides)
    blank = cpcf_body["blank_target"]
    blank["parent_graph_fingerprint"] = run_plan[
        "parent_graph_fingerprint"
    ]
    blank["target_unique_parameters"] = run_plan[
        "exact_target_unique_parameters"
    ]
    blank["requested_budget"] = run_plan["target_requested_budget"]
    blank["frontier_fingerprint"] = run_plan[
        "target_frontier_fingerprint"
    ]
    blank["target_certificate_sha256"] = run_plan[
        "target_certificate_sha256"
    ]
    blank["selected_candidate_fingerprint"] = run_plan[
        "selected_target_candidate_fingerprint"
    ]
    blank["selected_candidate_rank"] = run_plan[
        "selected_target_candidate_rank"
    ]
    target_parameters = run_plan["exact_target_unique_parameters"]
    source_parameters = target_plan.frontier.source_unique_parameters
    cpcf_body["parent_parameter_inventory"]["unique_parameters"] = (
        source_parameters
    )
    for name in (
        "blank_parameter_inventory",
        "compiled_parameter_inventory",
        "reloaded_parameter_inventory",
    ):
        cpcf_body[name]["unique_parameters"] = target_parameters
    compression = cpcf_body["compression"]
    compression["source_unique_parameters"] = source_parameters
    compression["target_unique_parameters"] = target_parameters
    compression["planned_target_unique_parameters"] = target_parameters
    compiled_provenance = _rebuild_cpcf_compiler_chain(
        cpcf_body,
        policy=fold_policy,
        phase_sha256=phase_sha256,
    )
    blank = cpcf_body["blank_target"]
    cpcf = _seal(cpcf_body, "receipt_sha256")

    capture_body = copy.deepcopy(documents["capture"])
    capture_body.pop("capture_receipt_sha256")
    receipt_file_sha256 = _candidate_file_sha(cpcf)
    capture_provenance = capture_body["cpcf_provenance"]
    capture_provenance["receipt_sha256"] = cpcf["receipt_sha256"]
    capture_provenance["external_receipt_file_sha256"] = receipt_file_sha256
    capture_provenance["internal_receipt_file_sha256"] = receipt_file_sha256
    capture_provenance["reloaded_parameter_inventory"] = cpcf[
        "reloaded_parameter_inventory"
    ]
    capture_bridge = capture_body["strict_tokenizer_bridge"]
    capture_bridge["artifact_sha256"] = bridge.artifact_sha256
    capture_bridge["profile_sha256"] = bridge.profile_fingerprint
    capture = _seal(capture_body, "capture_receipt_sha256")

    reviewed = copy.deepcopy(documents["composite"]["reviewed_evidence"])
    target_plan_binding = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-cpcf-target-plan-binding-v1",
            "parent_graph_fingerprint": run_plan[
                "parent_graph_fingerprint"
            ],
            "exact_target_unique_parameters": run_plan[
                "exact_target_unique_parameters"
            ],
            "requested_budget": run_plan["target_requested_budget"],
            "frontier_fingerprint": run_plan[
                "target_frontier_fingerprint"
            ],
            "target_certificate_sha256": run_plan[
                "target_certificate_sha256"
            ],
            "selected_candidate_fingerprint": run_plan[
                "selected_target_candidate_fingerprint"
            ],
            "selected_candidate_rank": run_plan[
                "selected_target_candidate_rank"
            ],
            "target_topology_fingerprint": blank[
                "target_topology_fingerprint"
            ],
            "blank_state_fingerprint": blank["blank_state_fingerprint"],
        }
    )
    reviewed.update(
        {
            "cpcf_run_receipt_sha256": cpcf["receipt_sha256"],
            "cpcf_run_receipt_document_sha256": canonical_sha256(cpcf),
            "parent_cache_key": parent_cache_key,
            "trace_adapter_sha256": trace_adapter_sha256,
            "blank_target_parent_graph_fingerprint": run_plan[
                "parent_graph_fingerprint"
            ],
            "exact_target_unique_parameters": run_plan[
                "exact_target_unique_parameters"
            ],
            "target_requested_budget": run_plan[
                "target_requested_budget"
            ],
            "target_frontier_fingerprint": run_plan[
                "target_frontier_fingerprint"
            ],
            "target_certificate_sha256": run_plan[
                "target_certificate_sha256"
            ],
            "selected_target_candidate_fingerprint": run_plan[
                "selected_target_candidate_fingerprint"
            ],
            "selected_target_candidate_rank": run_plan[
                "selected_target_candidate_rank"
            ],
            "target_plan_binding_sha256": target_plan_binding,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "bridge_profile_sha256": bridge.profile_fingerprint,
            "capture_receipt_sha256": capture["capture_receipt_sha256"],
            "capture_receipt_document_sha256": canonical_sha256(capture),
            **compiled_provenance,
        }
    )
    inference_contract = BehaviorInferenceExecutionContract.from_dict(
        documents["composite"]["inference_execution"]
    )
    composite = build_cpcf_reviewed_promotion_report(
        reviewed_evidence=reviewed,
        inference_execution_contract=inference_contract,
    )
    artifacts = replace(
        fixture["artifacts"],
        cpcf_run_receipt=_artifact(cpcf),
        capture_receipt=_artifact(capture),
        composite_promotion_report=_artifact(composite),
    )
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    return gate, evidence


def _production_case(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "tokenizer-source").mkdir()
    fixture = _fixture(tmp_path / "tokenizer-source")
    bridge_directory = tmp_path / "bridge-tokenizer"
    shutil.copytree(fixture["tokenizer_directory"], bridge_directory)
    bridge = seal_strict_tokenizer_bridge_artifact(
        fixture["tokenizer"],
        fixture["tokenizer"],
        bridge_directory,
        donor_row_capacity=81,
        target_row_capacity=65,
    )
    parent = _parent()
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
    )
    catalogs = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                minimum_hidden_size=4,
                hidden_size_multiple=4,
                minimum_head_dim=4,
                tokenizer_bridge_artifact=bridge,
            ),
        )
    )
    target_plan = plan_budgeted_dynamic_target(
        parent_graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=2,
            target_vocabulary_size=65,
            maximum_candidates=64,
        ),
        child_name="cpcf-production-child",
        adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
        fold_method_id=DENSE_BUDGET_MODEL_METHOD,
        catalogs=catalogs,
    )
    cache_key = _sha("production-cache")
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 4),
        evidence_fingerprint=cache_key,
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge_artifact=bridge,
    )
    trace, phases = _trace_and_phases(bridge, cache_key)
    fold_policy = Lfm2CpcfModelFoldPolicy(
        minimum_predictive_energy_fraction=0.0,
    )
    gate, evidence = _reviewed_gate_bound_to_bridge(
        tmp_path / "gate",
        bridge,
        target_plan=target_plan,
        parent_cache_key=cache_key,
        trace_adapter_sha256=trace.evidence_sha256,
        fold_policy=fold_policy,
        phase_sha256=tuple(phase.phase_sha256 for phase in phases),
    )
    cache_manifest_body = {
        "cache_key": cache_key,
        "parent": {
            "identity_sha256": evidence.cache_live_parent_fingerprint,
        },
        "corpus": {
            "corpus_sha256": _sha("production-cache-corpus"),
            "batch_count": trace.record_count,
        },
    }
    cache = Lfm2CircuitTraceCache(
        path=tmp_path,
        _manifest=MappingProxyType(
            {
                **cache_manifest_body,
                "manifest_sha256": canonical_sha256(cache_manifest_body),
            }
        ),
        _completion=MappingProxyType({}),
    )
    coverage = build_lfm2_cpcf_cache_phase_coverage_binding(
        cache=cache,
        trace_adapter=trace,
        phases=phases,
        parent_graph=parent_graph,
        trajectory_corpus_sha256=evidence.trajectory_corpus_sha256,
    )
    tied = (
        parent.get_input_embeddings()
        .weight.detach()
        .index_select(
            0,
            torch.tensor(trace.mapping, dtype=torch.long),
        )
        .clone()
    )
    reviewed_map = evidence.reviewed_evidence
    contract = Lfm2CpcfExternalBehaviorGateContract(
        independent_holdout_manifest_sha256=(
            reviewed_map["manifest_sha256"]
        ),
        independent_parent_envelope_sha256=(
            reviewed_map["parent_envelope_sha256"]
        ),
        parent_relative_manual_packet_sha256=(
            reviewed_map["parent_relative_manual_packet_sha256"]
        ),
    )
    resources = Lfm2CpcfStackResources(
        phases=phases,
        trace_adapter=trace,
        coverage_binding=coverage,
        parent_graph_fingerprint=parent_graph.fingerprint,
        expected_full_reviewed_evidence_fingerprint=(
            evidence.full_reviewed_evidence_fingerprint
        ),
        source_tied_table=tied,
        policy=fold_policy,
        external_behavior_contract=contract,
        final_evaluator=gate,
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: None,
        fresh_reload_evaluator=gate,
        saver_id="production-test-saver-v1",
        loader_id="production-test-loader-v1",
    )
    return {
        "bridge": bridge,
        "cache_key": cache_key,
        "cache": cache,
        "coverage_binding": coverage,
        "gate": gate,
        "tokenizer": fixture["tokenizer"],
        "tokenizer_directory": fixture["tokenizer_directory"],
        "trace_adapter_sha256": trace.evidence_sha256,
        "target_adapter": target_adapter,
        "target_plan": target_plan,
        "resources": resources,
    }


def _resources_with_reviewed_plan_override(
    case: dict[str, Any],
    tmp_path: Path,
    **overrides: Any,
) -> Lfm2CpcfStackResources:
    gate, evidence = _reviewed_gate_bound_to_bridge(
        tmp_path,
        case["bridge"],
        target_plan=case["target_plan"],
        parent_cache_key=case["cache_key"],
        trace_adapter_sha256=case["trace_adapter_sha256"],
        fold_policy=case["resources"].policy,
        phase_sha256=tuple(
            phase.phase_sha256 for phase in case["resources"].phases
        ),
        provenance_overrides=overrides,
    )
    return replace(
        case["resources"],
        expected_full_reviewed_evidence_fingerprint=(
            evidence.full_reviewed_evidence_fingerprint
        ),
        final_evaluator=gate,
        fresh_reload_evaluator=gate,
    )


def test_exact_cpcf_bundle_binds_size_bridge_cache_method_and_one_attempt(
    tmp_path: Path,
) -> None:
    case = _production_case(tmp_path)
    plan = case["target_plan"]
    request = plan.target_request
    bundle = build_lfm2_cpcf_exact_production_bundle(
        target_plan=plan,
        target_adapter=case["target_adapter"],
        resources=case["resources"],
    )
    exact_size = request.exact_unique_parameters
    candidate = bundle.candidate_specs[0]
    planned = plan.target_certificate.planned_graph
    portfolio_candidates = RegisteredExactTargetPortfolio(bundle).candidates(
        DynamicFoldSearchObjective(
            target=request,
            maximum_attempts=5,
            portfolio_id=bundle.portfolio_id,
        )
    )

    assert bundle.capability.minimum_target_unique_parameters == exact_size
    assert bundle.capability.maximum_target_unique_parameters == exact_size
    assert bundle.capability.supports(request)
    assert not bundle.capability.supports(
        replace(request, exact_unique_parameters=exact_size + 1)
    )
    assert bundle.maximum_attempts == 1
    assert len(bundle.candidate_specs) == 1
    assert len(portfolio_candidates) == 1
    assert portfolio_candidates[0].target.fold_method_id == (
        DENSE_BUDGET_MODEL_METHOD
    )
    assert portfolio_candidates[0].evidence_fingerprint == case["cache_key"]
    assert planned.ordered_source_spans == tuple(
        (index,) for index in range(planned.layer_count)
    )
    assert all(
        len(phase.parent_evidence.layers) == planned.layer_count
        for phase in case["resources"].phases
    )
    assert case["resources"].parent_graph_fingerprint == (
        plan.target_certificate.parent_graph_fingerprint
    )
    assert case["resources"].coverage_binding is case["coverage_binding"]
    assert case["coverage_binding"].cache_key == case["cache_key"]
    assert case["coverage_binding"].phase_count == len(
        case["resources"].phases
    )
    assert case["coverage_binding"].parent_graph_fingerprint == (
        plan.target_certificate.parent_graph_fingerprint
    )
    assert case["coverage_binding"].binding_fingerprint == canonical_sha256(
        case["coverage_binding"].body()
    )
    assert candidate.method_id == DENSE_BUDGET_MODEL_METHOD
    assert candidate.evidence_fingerprint == case["cache_key"]
    assert candidate.target_adapter_id == Lfm2TargetSynthesisAdapter.adapter_id
    assert dict(candidate.configuration) == dict(request.configuration)
    assert all(
        dict(candidate.configuration).get(name) == value
        for name, value in strict_tokenizer_bridge_configuration(
            case["bridge"]
        ).items()
    )
    metadata = dict(candidate.metadata)
    assert metadata["target_plan_fingerprint"] == plan.fingerprint
    assert metadata["cpcf_resource_fingerprint"] == (
        case["resources"].resource_fingerprint
    )
    assert metadata["tokenizer_bridge_artifact_sha256"] == (
        case["bridge"].artifact_sha256
    )
    evidence = case["gate"].evidence
    assert metadata["reviewed_target_plan_binding_sha256"] == (
        evidence.target_plan_binding_sha256
    )
    assert metadata["reviewed_cpcf_run_receipt_sha256"] == (
        evidence.cpcf_run_receipt_sha256
    )
    assert metadata["reviewed_evidence_fingerprint"] == (
        evidence.full_reviewed_evidence_fingerprint
    )
    assert metadata["reviewed_target_frontier_fingerprint"] == (
        plan.frontier.fingerprint
    )
    assert metadata["reviewed_target_certificate_sha256"] == (
        plan.target_certificate.fingerprint
    )
    assert metadata["reviewed_selected_target_candidate_fingerprint"] == (
        plan.frontier.candidates[
            plan.target_certificate.selected_candidate_rank
        ].fingerprint
    )


@pytest.mark.parametrize(
    ("field", "replacement_factory"),
    (
        (
            "exact_target_unique_parameters",
            lambda case: (
                case["target_plan"].target_request.exact_unique_parameters + 1
            ),
        ),
        (
            "target_requested_budget",
            lambda case: {
                **case["target_plan"].frontier.request.as_dict(),
                "maximum_candidates": (
                    case["target_plan"].frontier.request.maximum_candidates + 1
                ),
            },
        ),
        (
            "target_frontier_fingerprint",
            lambda case: _sha("reviewed-wrong-frontier"),
        ),
        (
            "target_certificate_sha256",
            lambda case: _sha("reviewed-wrong-certificate"),
        ),
        (
            "selected_target_candidate_fingerprint",
            lambda case: _sha("reviewed-wrong-candidate"),
        ),
        (
            "selected_target_candidate_rank",
            lambda case: (
                case["target_plan"].target_certificate.selected_candidate_rank
                + 1
            ),
        ),
    ),
)
def test_exact_cpcf_bundle_rejects_each_reviewed_target_plan_identity_drift(
    tmp_path: Path,
    field: str,
    replacement_factory: Any,
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    case = _production_case(base)
    resources = _resources_with_reviewed_plan_override(
        case,
        tmp_path / f"mismatch-{field}",
        **{field: replacement_factory(case)},
    )

    with pytest.raises(
        Lfm2CpcfProductionBundleError,
        match="does not describe the exact parent graph, budget, frontier",
    ):
        build_lfm2_cpcf_exact_production_bundle(
            target_plan=case["target_plan"],
            target_adapter=case["target_adapter"],
            resources=resources,
        )


def test_stack_resources_reject_reviewed_parent_graph_coverage_drift(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    case = _production_case(base)

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="coverage cache, parent identity, graph",
    ):
        _resources_with_reviewed_plan_override(
            case,
            tmp_path / "mismatch-reviewed-parent-graph",
            parent_graph_fingerprint=_sha("reviewed-wrong-parent-graph"),
        )


def test_stack_resources_reject_tampered_coverage_binding(
    tmp_path: Path,
) -> None:
    case = _production_case(tmp_path)
    mismatched = replace(
        case["coverage_binding"],
        trajectory_corpus_sha256=_sha("wrong-trajectory-corpus"),
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="coverage cache, parent identity, graph",
    ):
        replace(
            case["resources"],
            coverage_binding=mismatched,
        )


def test_exact_cpcf_bundle_rejects_resource_bridge_cache_and_method_drift(
    tmp_path: Path,
) -> None:
    case = _production_case(tmp_path)
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="coverage cache, parent identity, graph",
    ):
        replace(
            case["resources"],
            parent_graph_fingerprint=_sha("wrong-parent-graph"),
        )

    wrong_cache_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 4),
        evidence_fingerprint=_sha("wrong-cache"),
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge_artifact=case["bridge"],
    )
    with pytest.raises(
        Lfm2CpcfProductionBundleError,
        match="evidence fingerprint differs",
    ):
        build_lfm2_cpcf_exact_production_bundle(
            target_plan=case["target_plan"],
            target_adapter=wrong_cache_adapter,
            resources=case["resources"],
        )

    wrong_bridge_directory = tmp_path / "wrong-bridge-tokenizer"
    shutil.copytree(
        case["tokenizer_directory"],
        wrong_bridge_directory,
    )
    wrong_bridge = seal_strict_tokenizer_bridge_artifact(
        case["tokenizer"],
        case["tokenizer"],
        wrong_bridge_directory,
        donor_row_capacity=82,
        target_row_capacity=65,
    )
    wrong_bridge_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 4),
        evidence_fingerprint=case["cache_key"],
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge_artifact=wrong_bridge,
    )
    with pytest.raises(
        Lfm2CpcfProductionBundleError,
        match="configuration differs",
    ):
        build_lfm2_cpcf_exact_production_bundle(
            target_plan=case["target_plan"],
            target_adapter=wrong_bridge_adapter,
            resources=case["resources"],
        )

    plan = case["target_plan"]
    wrong_request = replace(
        plan.target_request,
        fold_method_id="neural_forge.test.wrong_method",
    )
    wrong_certificate = replace(
        plan.target_certificate,
        target_request=wrong_request,
    )
    wrong_method_plan = replace(
        plan,
        target_certificate=wrong_certificate,
    )
    with pytest.raises(
        Lfm2CpcfProductionBundleError,
        match="dense residual-attention method",
    ):
        build_lfm2_cpcf_exact_production_bundle(
            target_plan=wrong_method_plan,
            target_adapter=case["target_adapter"],
            resources=case["resources"],
        )


def test_exact_cpcf_bundle_rejects_depth_span_and_vocabulary_drift(
    tmp_path: Path,
) -> None:
    case = _production_case(tmp_path)
    _trace, shallow_phases = _trace_and_phases(
        case["bridge"],
        case["cache_key"],
        operator_kinds=("short_conv",),
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="coverage does not bind the exact ordered resource phase",
    ):
        replace(
            case["resources"],
            phases=shallow_phases,
        )

    truncated_table_resources = replace(
        case["resources"],
        source_tied_table=case["resources"].source_tied_table[:-1].clone(),
    )
    with pytest.raises(
        Lfm2CpcfProductionBundleError,
        match="target-row capacity",
    ):
        build_lfm2_cpcf_exact_production_bundle(
            target_plan=case["target_plan"],
            target_adapter=case["target_adapter"],
            resources=truncated_table_resources,
        )


def test_exact_cpcf_capability_cannot_overlap_another_lfm2_text_lane(
    tmp_path: Path,
) -> None:
    case = _production_case(tmp_path)
    bundle = build_lfm2_cpcf_exact_production_bundle(
        target_plan=case["target_plan"],
        target_adapter=case["target_adapter"],
        resources=case["resources"],
    )
    exact_size = bundle.capability.minimum_target_unique_parameters
    catalog = ProductionDynamicFoldCatalog(
        (bundle,),
        ledger=ExperimentLedger(),
    )
    disjoint = replace(
        bundle,
        capability=replace(
            bundle.capability,
            capability_id="lfm2-text-cpcf-next-size",
            minimum_target_unique_parameters=exact_size + 1,
            maximum_target_unique_parameters=exact_size + 1,
        ),
        portfolio_id="neural_forge.production.lfm2-cpcf-next-size",
    )
    catalog.register(disjoint)

    overlapping = replace(
        bundle,
        capability=replace(
            bundle.capability,
            capability_id="lfm2-text-cpcf-overlap",
        ),
        portfolio_id="neural_forge.production.lfm2-cpcf-overlap",
    )
    with pytest.raises(
        ProductionDynamicFoldCatalogError,
        match="ranges overlap",
    ):
        catalog.register(overlapping)
