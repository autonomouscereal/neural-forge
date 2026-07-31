from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

import neural_forge.dynamic_lfm2_cpcf_adapter as cpcf_adapter_module
from neural_forge.dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import (
    FoldDecision,
    FoldExecutionContext,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from neural_forge.dynamic_fold_registry import ExperimentLedger
from neural_forge.dynamic_fold_planner import DynamicFoldPlanner
from neural_forge.dynamic_lfm2_adapter import (
    DynamicLfm2AdapterError,
    Lfm2TargetSynthesisAdapter,
)
from neural_forge.dynamic_lfm2_cpcf_adapter import (
    Lfm2CpcfCatalogError,
    Lfm2CpcfEvidenceProvider,
    Lfm2CpcfExternalBehaviorGateContract,
    Lfm2CpcfFoldStackAdapter,
    Lfm2CpcfModelOperator,
    Lfm2CpcfSealedEvidence,
    Lfm2CpcfStackResources,
)
from neural_forge.dynamic_lfm2_cpcf_reviewed_gate import (
    Lfm2CpcfReviewedPromotionGate,
    build_cpcf_reviewed_promotion_report,
    load_lfm2_cpcf_reviewed_promotion_evidence,
)
from neural_forge.dynamic_lfm2_cpcf_resources import (
    Lfm2CpcfCachePhaseCoverageBinding,
    build_lfm2_cpcf_cache_phase_coverage_binding,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import (
    default_inspector_registry,
    inspect_model_circuits,
)
from neural_forge.dynamic_system import DynamicFoldStackAdapterRegistry
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesizer,
    plan_budgeted_dynamic_target,
)
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
from neural_forge.hf_lfm2_cpcf_model_fold import (
    LFM2_CPCF_MODEL_FOLD_SCHEMA,
    Lfm2CpcfModelFoldPolicy,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2TargetTokenizedRecord,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (
    seal_strict_tokenizer_bridge_artifact,
)
from tests.test_dynamic_lfm2_cpcf_reviewed_gate import (
    _artifact,
    _candidate_file_sha,
    _fixture as _reviewed_artifact_fixture,
    _rebuild_cpcf_compiler_chain,
    _seal,
)


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _installability_tensor_sha256(value) -> str:
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
    digest.update(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    )
    return digest.hexdigest()


def _field_inventory(tensors) -> dict[str, Any]:
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": [int(size) for size in value.shape],
            "sha256": _installability_tensor_sha256(value),
        }
        for name, value in sorted(tensors.items())
    }


class _Tokenizer:
    def __init__(self, surfaces: list[str]) -> None:
        self.surfaces = list(surfaces)
        self.all_special_ids: list[int] = [0]
        self.bos_token_id = None
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.unk_token_id = None

    def __len__(self) -> int:
        return len(self.surfaces)

    def get_vocab(self) -> dict[str, int]:
        return {
            f"raw-{index}-{surface}": index
            for index, surface in enumerate(self.surfaces)
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"raw-{token_id}-{self.surfaces[token_id]}"

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.surfaces.index(text)]


def _parent() -> Any:
    config = transformers.Lfm2Config(
        vocab_size=41,
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
        torch.manual_seed(3101)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _bridge(tmp_path):
    donor = _Tokenizer([f"<{index}>" for index in range(41)])
    target = _Tokenizer(donor.surfaces[:17])
    root = tmp_path / "target-tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=41,
        target_row_capacity=17,
    )


def _target(parent, artifact, cache_key: str):
    parent_graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=(
            artifact.donor_tokenizer_semantics_sha256
        ),
    )
    catalogs = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                minimum_hidden_size=4,
                hidden_size_multiple=4,
                minimum_head_dim=4,
                tokenizer_bridge_artifact=artifact,
            ),
        )
    )
    plan = plan_budgeted_dynamic_target(
        parent_graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            compression_ratio=2,
            target_vocabulary_size=17,
            maximum_candidates=64,
        ),
        child_name="cpcf-catalog-child",
        adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
        catalogs=catalogs,
    )
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 4),
        evidence_fingerprint=cache_key,
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge_artifact=artifact,
    )
    target = DynamicTargetSynthesizer(
        inspectors=default_inspector_registry(),
        adapters=DynamicTargetAdapterRegistry((adapter,)),
    ).synthesize(
        request=plan.target_request,
        parent_models={"parent": parent},
        parent_tokenizer_fingerprints={
            "parent": artifact.donor_tokenizer_semantics_sha256
        },
    )
    return parent_graph, plan, target


def _trace_and_phases(artifact, cache_key: str):
    trace = build_lfm2_strict_tokenizer_bridge_trace_adapter(
        artifact,
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
    parent_phases = []
    phase_names = (
        "prefill",
        *(
            f"decode.{index:03d}"
            for index in range(aligned.forced_decode_steps)
        ),
    )
    for phase_name in phase_names:
        layers = []
        for index in range(2):
            tensors = {"residual.pre": torch.zeros(1, 1, 8)}
            operator_kind = (
                "short_conv" if index == 0 else "full_attention"
            )
            body = {
                "schema": LFM2_CPCF_INSTALLABLE_LAYER_SCHEMA,
                "cache_key": cache_key,
                "batch_index": 0,
                "phase": phase_name,
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
        final_norm_tensors: dict[str, Any] = {}
        phase_body = {
            "schema": LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA,
            "cache_key": cache_key,
            "batch_index": 0,
            "input_sha256": aligned.donor_cache_input_sha256,
            "phase": phase_name,
            "layer_evidence_sha256": [
                layer.evidence_sha256 for layer in layers
            ],
            "final_norm_field_inventory": _field_inventory(
                final_norm_tensors
            ),
            "parent_side_compiler_inputs_ready": True,
        }
        parent_phases.append(
            Lfm2CpcfInstallablePhaseEvidence(
                cache_key=cache_key,
                batch_index=0,
                input_sha256=aligned.donor_cache_input_sha256,
                phase=phase_name,
                layers=tuple(layers),
                final_norm_tensors=final_norm_tensors,
                evidence_sha256=canonical_sha256(phase_body),
            )
        )
    return trace, build_lfm2_cpcf_target_phases(
        trace,
        tuple(parent_phases),
    )


def _contract_for_gate(
    gate: Lfm2CpcfReviewedPromotionGate,
) -> Lfm2CpcfExternalBehaviorGateContract:
    reviewed = gate.evidence.reviewed_evidence
    return Lfm2CpcfExternalBehaviorGateContract(
        independent_holdout_manifest_sha256=reviewed["manifest_sha256"],
        independent_parent_envelope_sha256=reviewed[
            "parent_envelope_sha256"
        ],
        parent_relative_manual_packet_sha256=reviewed[
            "parent_relative_manual_packet_sha256"
        ],
    )


def _authentic_reviewed_gate(
    tmp_path,
    artifact,
    *,
    label: str,
    parent_graph: Any,
    target_plan: Any,
    target: Any,
    cache_key: str,
    trace_adapter_sha256: str,
    phases: tuple[Any, ...],
    policy: Lfm2CpcfModelFoldPolicy,
    trajectory_corpus_sha256: str,
    provenance_overrides: dict[str, Any] | None = None,
    compiled_policy: Lfm2CpcfModelFoldPolicy | None = None,
    compiled_phase_sha256: tuple[str, ...] | None = None,
    bridge_artifact_sha256: str | None = None,
    bridge_profile_sha256: str | None = None,
    tokenizer_surfaces: list[str] | None = None,
    extra_tokenizer_config: bool = False,
) -> Lfm2CpcfReviewedPromotionGate:
    surfaces = (
        [f"<{index}>" for index in range(17)]
        if tokenizer_surfaces is None
        else list(tokenizer_surfaces)
    )
    tokenizer = _Tokenizer(surfaces)
    tokenizer_directory = artifact.root_dir
    if extra_tokenizer_config:
        tokenizer_directory = tmp_path / f"reviewed-tokenizer-{label}"
        tokenizer_directory.mkdir()
        (tokenizer_directory / "tokenizer.json").write_bytes(
            (artifact.root_dir / "tokenizer.json").read_bytes()
        )
        (tokenizer_directory / "tokenizer_config.json").write_text(
            '{"variant":true}',
            encoding="utf-8",
        )
    fixture = _reviewed_artifact_fixture(
        tmp_path / f"reviewed-chain-{label}",
        bridge_artifact_sha256=(
            artifact.artifact_sha256
            if bridge_artifact_sha256 is None
            else bridge_artifact_sha256
        ),
        bridge_profile_sha256=(
            artifact.profile_fingerprint
            if bridge_profile_sha256 is None
            else bridge_profile_sha256
        ),
        tokenizer=tokenizer,
        tokenizer_directory=tokenizer_directory,
    )
    documents = fixture["documents"]
    cpcf_body = copy.deepcopy(documents["cpcf"])
    cpcf_body.pop("receipt_sha256")
    cpcf_body["inputs"].update(
        {
            "cache_key": cache_key,
            "trace_adapter_sha256": trace_adapter_sha256,
            "trajectory_corpus_sha256": trajectory_corpus_sha256,
        }
    )
    certificate = target_plan.target_certificate
    selected_candidate = target_plan.frontier.candidates[
        certificate.selected_candidate_rank
    ]
    selected_target = target.adapter_payload
    run_provenance = {
        "parent_graph_fingerprint": selected_target.source_graph_fingerprint,
        "parent_state_fingerprint": selected_target.source_state_fingerprint,
        "target_unique_parameters": (
            selected_target.exact_target_unique_parameters
        ),
        "requested_budget": target_plan.frontier.request.as_dict(),
        "frontier_fingerprint": target_plan.frontier.fingerprint,
        "target_certificate_sha256": certificate.fingerprint,
        "selected_candidate_fingerprint": selected_candidate.fingerprint,
        "selected_candidate_rank": certificate.selected_candidate_rank,
        "target_topology_fingerprint": target.topology_fingerprint,
        "blank_state_fingerprint": target.blank_state_fingerprint,
    }
    if provenance_overrides is not None:
        input_fields = {
            "cache_key",
            "trace_adapter_sha256",
            "trajectory_corpus_sha256",
        }
        cpcf_body["inputs"].update(
            {
                name: value
                for name, value in provenance_overrides.items()
                if name in input_fields
            }
        )
        run_provenance.update(
            {
                name: value
                for name, value in provenance_overrides.items()
                if name not in input_fields
            }
        )
    blank = cpcf_body["blank_target"]
    blank.pop("report_sha256", None)
    blank.update(run_provenance)
    cpcf_body["parent_parameter_inventory"]["unique_parameters"] = (
        target_plan.frontier.source_unique_parameters
    )
    compiled_provenance = _rebuild_cpcf_compiler_chain(
        cpcf_body,
        policy=policy if compiled_policy is None else compiled_policy,
        phase_sha256=(
            tuple(phase.phase_sha256 for phase in phases)
            if compiled_phase_sha256 is None
            else compiled_phase_sha256
        ),
    )
    cpcf = _seal(cpcf_body, "receipt_sha256")

    capture_body = copy.deepcopy(documents["capture"])
    capture_body.pop("capture_receipt_sha256")
    receipt_file_sha256 = _candidate_file_sha(cpcf)
    capture_provenance = capture_body["cpcf_provenance"]
    capture_provenance.update(
        {
            "receipt_sha256": cpcf["receipt_sha256"],
            "external_receipt_file_sha256": receipt_file_sha256,
            "internal_receipt_file_sha256": receipt_file_sha256,
            "reloaded_parameter_inventory": cpcf[
                "reloaded_parameter_inventory"
            ],
        }
    )
    capture = _seal(capture_body, "capture_receipt_sha256")

    blank = cpcf["blank_target"]
    inputs = cpcf["inputs"]
    target_plan_binding_sha256 = canonical_sha256(
        {
            "schema": "neural-forge-lfm2-cpcf-target-plan-binding-v1",
            "parent_graph_fingerprint": blank[
                "parent_graph_fingerprint"
            ],
            "exact_target_unique_parameters": blank[
                "target_unique_parameters"
            ],
            "requested_budget": blank["requested_budget"],
            "frontier_fingerprint": blank["frontier_fingerprint"],
            "target_certificate_sha256": blank[
                "target_certificate_sha256"
            ],
            "selected_candidate_fingerprint": blank[
                "selected_candidate_fingerprint"
            ],
            "selected_candidate_rank": blank[
                "selected_candidate_rank"
            ],
            "target_topology_fingerprint": blank[
                "target_topology_fingerprint"
            ],
            "blank_state_fingerprint": blank[
                "blank_state_fingerprint"
            ],
        }
    )
    reviewed = copy.deepcopy(
        documents["composite"]["reviewed_evidence"]
    )
    reviewed.update(
        {
            "cpcf_run_receipt_sha256": cpcf["receipt_sha256"],
            "cpcf_run_receipt_document_sha256": canonical_sha256(cpcf),
            "parent_cache_key": inputs["cache_key"],
            "cache_live_parent_fingerprint": inputs[
                "parent_fingerprint"
            ],
            "live_parent_state_fingerprint": blank[
                "parent_state_fingerprint"
            ],
            "trace_adapter_sha256": inputs["trace_adapter_sha256"],
            "trajectory_corpus_sha256": inputs[
                "trajectory_corpus_sha256"
            ],
            "blank_target_parent_graph_fingerprint": blank[
                "parent_graph_fingerprint"
            ],
            "exact_target_unique_parameters": blank[
                "target_unique_parameters"
            ],
            "target_requested_budget": blank["requested_budget"],
            "target_frontier_fingerprint": blank[
                "frontier_fingerprint"
            ],
            "target_certificate_sha256": blank[
                "target_certificate_sha256"
            ],
            "selected_target_candidate_fingerprint": blank[
                "selected_candidate_fingerprint"
            ],
            "selected_target_candidate_rank": blank[
                "selected_candidate_rank"
            ],
            "target_topology_fingerprint": blank[
                "target_topology_fingerprint"
            ],
            "blank_target_state_fingerprint": blank[
                "blank_state_fingerprint"
            ],
            "target_plan_binding_sha256": target_plan_binding_sha256,
            **compiled_provenance,
            "capture_receipt_sha256": capture[
                "capture_receipt_sha256"
            ],
            "capture_receipt_document_sha256": canonical_sha256(capture),
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
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(
        artifacts
    )
    return Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=tokenizer,
        tokenizer_directory=tokenizer_directory,
    )


def _coverage_binding(
    tmp_path,
    *,
    label: str,
    gate: Lfm2CpcfReviewedPromotionGate,
    trace: Any,
    phases: tuple[Any, ...],
    parent_graph: Any,
    cache_key: str,
) -> Lfm2CpcfCachePhaseCoverageBinding:
    manifest_body = {
        "cache_key": cache_key,
        "parent": {
            "identity_sha256": (
                gate.evidence.cache_live_parent_fingerprint
            )
        },
        "corpus": {
            "corpus_sha256": _sha(f"{label}:cache-corpus"),
            "batch_count": trace.record_count,
        },
    }
    cache = Lfm2CircuitTraceCache(
        path=tmp_path,
        _manifest=MappingProxyType(
            {
                **manifest_body,
                "manifest_sha256": canonical_sha256(manifest_body),
            }
        ),
        _completion=MappingProxyType({}),
    )
    return build_lfm2_cpcf_cache_phase_coverage_binding(
        cache=cache,
        trace_adapter=trace,
        phases=phases,
        parent_graph=parent_graph,
        trajectory_corpus_sha256=(
            gate.evidence.trajectory_corpus_sha256
        ),
    )


def _stack_case(tmp_path, label: str) -> SimpleNamespace:
    parent = _parent()
    artifact = _bridge(tmp_path)
    cache_key = _sha(label)
    parent_graph, target_plan, target = _target(
        parent,
        artifact,
        cache_key,
    )
    trace, phases = _trace_and_phases(artifact, cache_key)
    policy = Lfm2CpcfModelFoldPolicy(
        minimum_predictive_energy_fraction=0.0,
    )
    trajectory_corpus_sha256 = _sha(f"{label}:trajectory-corpus")
    source_tied_table = (
        parent.get_input_embeddings()
        .weight.detach()
        .index_select(
            0,
            torch.tensor(trace.mapping, dtype=torch.long),
        )
        .clone()
    )
    gate = _authentic_reviewed_gate(
        tmp_path,
        artifact,
        label=f"{label}-reviewed",
        parent_graph=parent_graph,
        target_plan=target_plan,
        target=target,
        cache_key=cache_key,
        trace_adapter_sha256=trace.evidence_sha256,
        phases=phases,
        policy=policy,
        trajectory_corpus_sha256=trajectory_corpus_sha256,
    )
    coverage = _coverage_binding(
        tmp_path,
        label=label,
        gate=gate,
        trace=trace,
        phases=phases,
        parent_graph=parent_graph,
        cache_key=cache_key,
    )
    resources = Lfm2CpcfStackResources(
        phases=phases,
        trace_adapter=trace,
        coverage_binding=coverage,
        parent_graph_fingerprint=parent_graph.fingerprint,
        source_tied_table=source_tied_table,
        policy=policy,
        external_behavior_contract=_contract_for_gate(gate),
        expected_full_reviewed_evidence_fingerprint=(
            gate.evidence.full_reviewed_evidence_fingerprint
        ),
        final_evaluator=gate,
        model_saver=lambda _model, _path: None,
        model_loader=lambda _path: None,
        fresh_reload_evaluator=gate,
        saver_id="test-cpcf-saver-v1",
        loader_id="test-cpcf-loader-v1",
    )
    return SimpleNamespace(
        parent=parent,
        parent_graph=parent_graph,
        target_plan=target_plan,
        target=target,
        artifact=artifact,
        cache_key=cache_key,
        trace=trace,
        phases=phases,
        policy=policy,
        trajectory_corpus_sha256=trajectory_corpus_sha256,
        gate=gate,
        coverage=coverage,
        resources=resources,
        adapter=Lfm2CpcfFoldStackAdapter(resources),
    )


def _reviewed_gate_for_stack_case(
    tmp_path,
    case: SimpleNamespace,
    *,
    label: str,
    provenance_overrides: dict[str, Any] | None = None,
    compiled_policy: Lfm2CpcfModelFoldPolicy | None = None,
    compiled_phase_sha256: tuple[str, ...] | None = None,
) -> Lfm2CpcfReviewedPromotionGate:
    return _authentic_reviewed_gate(
        tmp_path,
        case.artifact,
        label=label,
        parent_graph=case.parent_graph,
        target_plan=case.target_plan,
        target=case.target,
        cache_key=case.cache_key,
        trace_adapter_sha256=case.trace.evidence_sha256,
        phases=case.phases,
        policy=case.policy,
        trajectory_corpus_sha256=case.trajectory_corpus_sha256,
        provenance_overrides=provenance_overrides,
        compiled_policy=compiled_policy,
        compiled_phase_sha256=compiled_phase_sha256,
    )


def _resources_with_reviewed_gate(
    case: SimpleNamespace,
    gate: Lfm2CpcfReviewedPromotionGate,
) -> Lfm2CpcfStackResources:
    return replace(
        case.resources,
        external_behavior_contract=_contract_for_gate(gate),
        expected_full_reviewed_evidence_fingerprint=(
            gate.evidence.full_reviewed_evidence_fingerprint
        ),
        final_evaluator=gate,
        fresh_reload_evaluator=gate,
    )


def _fake_compiled_report(parent, target):
    parent_unique = sum(
        parameter.numel() for parameter in parent.parameters()
    )
    target_unique = target.graph.unique_parameter_count
    body = {
        "schema": LFM2_CPCF_MODEL_FOLD_SCHEMA,
        "all_layers_compiled_in_order": True,
        "actual_folded_upstream_replay_used": True,
        "source_state_preserved": True,
        "target_parameter_count_preserved": True,
        "parameter_receipt": {
            "source_unique_parameters": parent_unique,
            "target_unique_parameters": target_unique,
            "compression_ratio": parent_unique / target_unique,
            "parameter_count_compaction_only": True,
            "checkpoint_byte_ratio_used": False,
            "quantization_used": False,
            "pruning_used": False,
        },
        "operation_audit": {
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "generation_calls": 0,
            "parent_cache_generation_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "fallback_paths": 0,
        },
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _install_fake_compile(monkeypatch, parent, target):
    report = _fake_compiled_report(parent, target)
    live_blank = fingerprint_model_state(target.model)

    def fake_compile(_source, child, **_kwargs):
        assert child is not target.model
        assert fingerprint_model_state(target.model) == live_blank
        with torch.no_grad():
            for index, parameter in enumerate(child.parameters()):
                parameter.fill_(0.01 * (index + 1))
        return SimpleNamespace(
            as_dict=lambda: dict(report),
            result_sha256=report["result_sha256"],
        )

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_compile,
    )


def _operator_case(tmp_path, label: str) -> SimpleNamespace:
    parent = _parent()
    artifact = _bridge(tmp_path)
    cache_key = _sha(label)
    parent_graph, _target_plan, target = _target(
        parent,
        artifact,
        cache_key,
    )
    trace, phases = _trace_and_phases(artifact, cache_key)
    indices = torch.tensor(trace.mapping, dtype=torch.long)
    tied = (
        parent.get_input_embeddings()
        .weight.detach()
        .index_select(0, indices)
        .clone()
    )
    selected = target.adapter_payload
    sealed = Lfm2CpcfSealedEvidence(
        phases=phases,
        trace_adapter=trace,
        source_tied_table=tied,
        blank_target_report=cpcf_adapter_module._blank_report(
            target,
            selected,
        ),
        policy=Lfm2CpcfModelFoldPolicy(
            minimum_predictive_energy_fraction=0.0,
        ),
    )
    blank = fingerprint_model_state(target.model)
    compiled_plan = DynamicFoldPlanner(
        ledger=ExperimentLedger()
    ).plan(
        (parent_graph,),
        target.graph,
        child_state_fingerprint=blank,
        evidence_fingerprint=cache_key,
        parent_state_fingerprints={
            "parent": fingerprint_model_state(parent)
        },
        exact_target_unique_parameters=(
            selected.exact_target_unique_parameters
        ),
    )
    action = compiled_plan.actions[0]
    context = FoldExecutionContext(
        parent_models={"parent": parent},
        child_model=target.model,
        parent_graphs={"parent": parent_graph},
        child_graph=target.graph,
    )
    provider = Lfm2CpcfEvidenceProvider(sealed)
    evidence = provider.prepare(action, context)
    return SimpleNamespace(
        parent=parent,
        artifact=artifact,
        cache_key=cache_key,
        parent_graph=parent_graph,
        target=target,
        blank=blank,
        sealed=sealed,
        operator=Lfm2CpcfModelOperator(selected, sealed),
        action=action,
        context=context,
        evidence=evidence,
    )


def test_model_operator_compiles_only_staging_then_commits_live_copy(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "staged-commit-cache")
    _install_fake_compile(monkeypatch, case.parent, case.target)

    transaction = case.operator.prepare(
        case.action,
        case.context,
        case.evidence,
    )

    assert fingerprint_model_state(case.target.model) != case.blank
    assert transaction.report[
        "compiler_received_disposable_staging_blank"
    ]
    assert transaction.report["compiler_received_live_child"] is False
    assert transaction.report["live_install_controlled_copy_only"]
    assert transaction.report["staged_install_receipt"][
        "staging_and_live_candidate_states_equal"
    ]
    assert transaction.report["compile_sealed_storage_write_guard"][
        "all_protected_bindings_unchanged"
    ]
    committed = transaction.commit()
    assert committed["accepted"] is True
    assert committed["candidate_state_fingerprint"] == (
        fingerprint_model_state(case.target.model)
    )
    compiler_report = dict(committed["compiled_model_fold"])
    compiler_result_sha256 = compiler_report.pop("result_sha256")
    assert canonical_sha256(compiler_report) == compiler_result_sha256
    assert "catalog_id" not in compiler_report
    assert "parameter_path_write_receipt" not in compiler_report
    catalog_report = dict(committed["catalog_compile_report"])
    catalog_report_sha256 = catalog_report.pop("report_sha256")
    assert canonical_sha256(catalog_report) == catalog_report_sha256
    assert (
        catalog_report["compiled_model_fold_result_sha256"]
        == compiler_result_sha256
    )
    assert catalog_report["catalog_id"] == (
        case.target.adapter_payload.catalog_id
    )


def test_invalid_compiler_result_seal_is_rejected_before_live_install(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "invalid-compiler-seal")
    report = _fake_compiled_report(case.parent, case.target)
    report["operation_audit"]["training_steps"] = 1

    def fake_compile(_source, child, **_kwargs):
        with torch.no_grad():
            for index, parameter in enumerate(child.parameters()):
                parameter.fill_(0.01 * (index + 1))
        return SimpleNamespace(
            as_dict=lambda: copy.deepcopy(report),
            result_sha256=report["result_sha256"],
        )

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_compile,
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="result_sha256 seal is invalid",
    ):
        case.operator.prepare(
            case.action,
            case.context,
            case.evidence,
        )
    assert fingerprint_model_state(case.target.model) == case.blank


def test_compiler_object_and_serialized_report_seals_must_match(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "compiler-object-seal-mismatch")
    report = _fake_compiled_report(case.parent, case.target)

    def fake_compile(_source, child, **_kwargs):
        with torch.no_grad():
            for index, parameter in enumerate(child.parameters()):
                parameter.fill_(0.01 * (index + 1))
        return SimpleNamespace(
            as_dict=lambda: copy.deepcopy(report),
            result_sha256=_sha("different-compiler-object-seal"),
        )

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_compile,
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="result object and serialized report seals differ",
    ):
        case.operator.prepare(
            case.action,
            case.context,
            case.evidence,
        )
    assert fingerprint_model_state(case.target.model) == case.blank


def test_compiler_report_is_detached_and_reauthenticated_before_commit(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "compiler-report-immutable")
    _install_fake_compile(monkeypatch, case.parent, case.target)
    transaction = case.operator.prepare(
        case.action,
        case.context,
        case.evidence,
    )

    detached = transaction.compiled_model_fold_report
    detached["operation_audit"]["training_steps"] = 99
    assert (
        transaction.compiled_model_fold_report["operation_audit"][
            "training_steps"
        ]
        == 0
    )
    committed = transaction.commit()
    compiler_report = dict(committed["compiled_model_fold"])
    compiler_result_sha256 = compiler_report.pop("result_sha256")
    assert canonical_sha256(compiler_report) == compiler_result_sha256
    assert compiler_report["operation_audit"]["training_steps"] == 0


def test_catalog_wrapper_tamper_is_rejected_before_commit(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "catalog-wrapper-tamper")
    _install_fake_compile(monkeypatch, case.parent, case.target)
    transaction = case.operator.prepare(
        case.action,
        case.context,
        case.evidence,
    )

    forged_wrapper = dict(transaction.report)
    forged_wrapper.pop("report_sha256")
    forged_wrapper["catalog_id"] = "forged-catalog"
    forged_wrapper["report_sha256"] = canonical_sha256(forged_wrapper)
    transaction._catalog_compile_report_canonical = (
        cpcf_adapter_module._canonical_json(forged_wrapper)
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="no longer matches its bound seal",
    ):
        transaction.commit()
    assert transaction.active is True
    rolled_back = transaction.rollback()
    assert rolled_back["rollback_exact"] is True
    assert fingerprint_model_state(case.target.model) == case.blank


def test_transaction_rollback_and_evidence_tamper_are_exact(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "transaction-cache")
    _install_fake_compile(monkeypatch, case.parent, case.target)
    transaction = case.operator.prepare(
        case.action,
        case.context,
        case.evidence,
    )
    assert fingerprint_model_state(case.target.model) != case.blank
    write_receipt = transaction.report["parameter_path_write_receipt"]
    assert write_receipt[
        "all_unique_parameter_storages_changed_from_compiler_zero"
    ]
    assert write_receipt["resulting_path_hashes_verified"]
    assert write_receipt["target_unique_parameters"] == (
        case.target.graph.unique_parameter_count
    )
    assert any(
        group["tied_alias"]
        for group in write_receipt["alias_groups"]
    )
    report = transaction.rollback()
    assert report["rollback_exact"] is True
    assert fingerprint_model_state(case.target.model) == case.blank

    sealed = case.sealed
    assert isinstance(sealed, Lfm2CpcfSealedEvidence)
    with torch.no_grad():
        sealed.source_tied_table[0, 0].add_(1)
    with pytest.raises(Lfm2CpcfCatalogError, match="tied table changed"):
        sealed.verify()


def test_parent_evidence_normal_in_place_mutation_breaks_version_seal(
    tmp_path,
) -> None:
    case = _operator_case(tmp_path, "version-seal-cache")
    sealed = case.sealed
    tensor = (
        sealed.phases[0]
        .parent_evidence.layers[0]
        .tensors["residual.pre"]
    )
    with torch.no_grad():
        tensor.add_(1)
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="storage/object/version seal changed",
    ):
        sealed.verify()


def test_parent_evidence_data_mutation_breaks_authenticated_rehash(
    tmp_path,
) -> None:
    case = _operator_case(tmp_path, "data-seal-cache")
    sealed = case.sealed
    tensor = (
        sealed.phases[0]
        .parent_evidence.layers[0]
        .tensors["residual.pre"]
    )
    original_version = tensor._version
    with torch.no_grad():
        tensor.data.add_(1)
    assert tensor._version == original_version
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="authentication changed",
    ):
        sealed.verify_parent_tensor_integrity()


def test_compiler_data_mutation_is_caught_after_compile_and_rolls_back(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "post-compile-data-seal-cache")
    sealed = case.sealed
    tensor = (
        sealed.phases[0]
        .parent_evidence.layers[0]
        .tensors["residual.pre"]
    )
    original = tensor.detach().clone()
    original_version = tensor._version
    report = _fake_compiled_report(case.parent, case.target)

    def fake_mutating_compile(_source, child, **_kwargs):
        with torch.no_grad():
            tensor.data.add_(1)
            for index, parameter in enumerate(child.parameters()):
                parameter.fill_(0.01 * (index + 1))
        return SimpleNamespace(as_dict=lambda: dict(report))

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_mutating_compile,
    )
    try:
        with pytest.raises(
            Lfm2CpcfCatalogError,
            match="sealed-storage write",
        ):
            case.operator.prepare(
                case.action,
                case.context,
                case.evidence,
            )
        assert tensor._version == original_version
        assert fingerprint_model_state(case.target.model) == case.blank
    finally:
        with torch.no_grad():
            tensor.data.copy_(original)
    sealed.verify_parent_tensor_integrity()


def test_swallowed_transient_data_mutate_restore_attempt_still_rejects(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "swallowed-transient-write-cache")
    tensor = (
        case.sealed.phases[0]
        .parent_evidence.layers[0]
        .tensors["residual.pre"]
    )
    before = tensor.detach().clone()
    before_version = tensor._version
    report = _fake_compiled_report(case.parent, case.target)

    def fake_swallowing_compile(_source, child, **_kwargs):
        for mutation in (
            lambda: tensor.data.add_(1),
            lambda: tensor.data.copy_(before),
        ):
            try:
                with torch.no_grad():
                    mutation()
            except Lfm2CpcfCatalogError:
                pass
        with torch.no_grad():
            for index, parameter in enumerate(child.parameters()):
                parameter.fill_(0.02 * (index + 1))
        return SimpleNamespace(as_dict=lambda: dict(report))

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_swallowing_compile,
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="even if the compiler swallowed",
    ):
        case.operator.prepare(
            case.action,
            case.context,
            case.evidence,
        )
    assert tensor._version == before_version
    assert torch.equal(tensor, before)
    assert fingerprint_model_state(case.target.model) == case.blank


def test_compiler_result_serialization_remains_inside_write_guard(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "result-serialization-write")
    tied = case.sealed.source_tied_table
    tied_before = tied.detach().clone()
    report = _fake_compiled_report(case.parent, case.target)

    class _MutatingResult:
        def as_dict(self):
            with torch.no_grad():
                tied.data.add_(1)
            return dict(report)

    def fake_compile(_source, child, **_kwargs):
        with torch.no_grad():
            for index, parameter in enumerate(child.parameters()):
                parameter.fill_(0.025 * (index + 1))
        return _MutatingResult()

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_compile,
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="sealed-storage write",
    ):
        case.operator.prepare(
            case.action,
            case.context,
            case.evidence,
        )
    assert torch.equal(tied, tied_before)
    assert fingerprint_model_state(case.target.model) == case.blank


@pytest.mark.parametrize(
    "attack",
    ("phase-view", "tied-foreach", "parent-parameter-out"),
)
def test_dispatch_guard_rejects_view_foreach_and_out_alias_writes(
    tmp_path,
    monkeypatch,
    attack: str,
) -> None:
    case = _operator_case(tmp_path, f"alias-write-{attack}")
    parent_before = fingerprint_model_state(case.parent)
    phase_ids = case.sealed.phases[0].input_ids
    tied = case.sealed.source_tied_table
    phase_before = phase_ids.detach().clone()
    tied_before = tied.detach().clone()
    report = _fake_compiled_report(case.parent, case.target)

    def fake_alias_write(source, child, **_kwargs):
        del child
        with torch.no_grad():
            if attack == "phase-view":
                phase_ids.view(-1).add_(1)
            elif attack == "tied-foreach":
                torch._foreach_add_([tied], 1)
            else:
                parameter = next(source.parameters())
                torch.add(parameter, 1, out=parameter)
        return SimpleNamespace(as_dict=lambda: dict(report))

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_alias_write,
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="sealed-storage write",
    ):
        case.operator.prepare(
            case.action,
            case.context,
            case.evidence,
        )
    assert torch.equal(phase_ids, phase_before)
    assert torch.equal(tied, tied_before)
    assert fingerprint_model_state(case.parent) == parent_before
    assert fingerprint_model_state(case.target.model) == case.blank


@pytest.mark.parametrize(
    "replacement",
    ("parameter-object", "parameter-storage"),
)
def test_staging_replacement_cannot_corrupt_live_blank(
    tmp_path,
    monkeypatch,
    replacement: str,
) -> None:
    case = _operator_case(tmp_path, f"staging-{replacement}")
    live_bindings = tuple(
        case.target.model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    )
    live_identity = tuple(
        (
            name,
            id(parameter),
            int(parameter.untyped_storage()._cdata),
        )
        for name, parameter in live_bindings
    )
    report = _fake_compiled_report(case.parent, case.target)

    def fake_replacing_compile(_source, child, **_kwargs):
        name, parameter = next(
            iter(
                child.named_parameters(
                    recurse=True,
                    remove_duplicate=False,
                )
            )
        )
        replacement_value = torch.full_like(parameter, 0.5)
        if replacement == "parameter-object":
            owner = child
            components = name.split(".")
            for component in components[:-1]:
                owner = getattr(owner, component)
            setattr(
                owner,
                components[-1],
                torch.nn.Parameter(
                    replacement_value,
                    requires_grad=parameter.requires_grad,
                ),
            )
        else:
            parameter.data = replacement_value
        return SimpleNamespace(as_dict=lambda: dict(report))

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_replacing_compile,
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="staging writable tensor",
    ):
        case.operator.prepare(
            case.action,
            case.context,
            case.evidence,
        )
    assert fingerprint_model_state(case.target.model) == case.blank
    assert tuple(
        (
            name,
            id(parameter),
            int(parameter.untyped_storage()._cdata),
        )
        for name, parameter in case.target.model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    ) == live_identity


def test_explicitly_written_zero_folded_storage_is_authenticated(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "written-zero-storage")
    report = _fake_compiled_report(case.parent, case.target)

    def fake_zero_result_compile(_source, child, **_kwargs):
        with torch.no_grad():
            for index, parameter in enumerate(child.parameters()):
                if index == 0:
                    parameter.zero_()
                else:
                    parameter.fill_(0.03 * (index + 1))
        return SimpleNamespace(as_dict=lambda: dict(report))

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_zero_result_compile,
    )

    transaction = case.operator.prepare(
        case.action,
        case.context,
        case.evidence,
    )
    receipt = transaction.report["parameter_path_write_receipt"]
    assert receipt[
        "all_unique_parameter_storages_authentically_written"
    ]
    assert not receipt[
        "all_unique_parameter_storages_changed_from_compiler_zero"
    ]
    assert receipt["zero_valued_folded_storage_count"] >= 1
    assert any(
        group["zero_valued_folded_result"]
        and group["authenticated_compiler_write"]
        and not group["changed_from_compiler_zero"]
        for group in receipt["alias_groups"]
    )
    transaction.rollback()
    assert fingerprint_model_state(case.target.model) == case.blank


def test_failed_write_operations_do_not_forge_staging_write_set(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "failed-write-set-forgery")
    report = _fake_compiled_report(case.parent, case.target)

    def fake_failed_writes(_source, child, **_kwargs):
        with torch.no_grad():
            for parameter in child.parameters():
                incompatible = torch.empty(
                    (int(parameter.numel()) + 1,),
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                try:
                    parameter.copy_(incompatible)
                except RuntimeError:
                    pass
        return SimpleNamespace(as_dict=lambda: dict(report))

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_failed_writes,
    )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="did not write every unique staging target parameter storage",
    ):
        case.operator.prepare(
            case.action,
            case.context,
            case.evidence,
        )
    assert fingerprint_model_state(case.target.model) == case.blank


@pytest.mark.parametrize("compiler_raises", (False, True))
def test_parent_training_modes_restore_when_compiler_changes_them(
    tmp_path,
    monkeypatch,
    compiler_raises: bool,
) -> None:
    case = _operator_case(
        tmp_path,
        f"parent-mode-restore-{compiler_raises}",
    )
    parent_runtime_before = fingerprint_model_runtime_state(case.parent)
    parent_modes_before = tuple(
        bool(module.training) for module in case.parent.modules()
    )
    report = _fake_compiled_report(case.parent, case.target)

    def fake_mode_mutating_compile(source, child, **_kwargs):
        source.train()
        if compiler_raises:
            raise RuntimeError("compiler failed after changing parent mode")
        with torch.no_grad():
            for index, parameter in enumerate(child.parameters()):
                parameter.fill_(0.04 * (index + 1))
        return SimpleNamespace(as_dict=lambda: dict(report))

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_mode_mutating_compile,
    )

    with pytest.raises(
        (Lfm2CpcfCatalogError, RuntimeError),
        match="runtime topology|compiler failed",
    ):
        case.operator.prepare(
            case.action,
            case.context,
            case.evidence,
        )
    assert fingerprint_model_runtime_state(case.parent) == (
        parent_runtime_before
    )
    assert tuple(
        bool(module.training) for module in case.parent.modules()
    ) == parent_modes_before
    assert fingerprint_model_state(case.target.model) == case.blank


def test_claimed_complete_report_cannot_hide_partial_parameter_write(
    tmp_path,
    monkeypatch,
) -> None:
    case = _operator_case(tmp_path, "partial-write-cache")
    report = _fake_compiled_report(case.parent, case.target)
    report.update(
        {
            "complete_target_parameter_overwrite": True,
            "parameter_path_write_receipt": {
                "receipt_sha256": _sha("forged")
            },
        }
    )

    def fake_partial_compile(_source, child, **_kwargs):
        with torch.no_grad():
            next(child.parameters()).fill_(0.25)
        return SimpleNamespace(as_dict=lambda: dict(report))

    monkeypatch.setattr(
        "neural_forge.dynamic_lfm2_cpcf_adapter."
        "compile_lfm2_cpcf_model_fold",
        fake_partial_compile,
    )
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="did not write every unique staging target parameter storage",
    ):
        case.operator.prepare(
            case.action,
            case.context,
            case.evidence,
        )
    assert fingerprint_model_state(case.target.model) == case.blank


def test_stack_resources_reject_mismatched_or_non_strict_behavior(
    tmp_path,
) -> None:
    parent = _parent()
    artifact = _bridge(tmp_path)
    cache_key = _sha("gate-cache")
    parent_graph, target_plan, target = _target(
        parent,
        artifact,
        cache_key,
    )
    trace, phases = _trace_and_phases(artifact, cache_key)
    policy = Lfm2CpcfModelFoldPolicy(
        minimum_predictive_energy_fraction=0.0,
    )
    trajectory_corpus_sha256 = _sha("gate-cache:trajectory-corpus")
    tied = (
        parent.get_input_embeddings()
        .weight.detach()
        .index_select(
            0,
            torch.tensor(trace.mapping, dtype=torch.long),
        )
        .clone()
    )
    authentic_gate = _authentic_reviewed_gate(
        tmp_path,
        artifact,
        label="matching",
        parent_graph=parent_graph,
        target_plan=target_plan,
        target=target,
        cache_key=cache_key,
        trace_adapter_sha256=trace.evidence_sha256,
        phases=phases,
        policy=policy,
        trajectory_corpus_sha256=trajectory_corpus_sha256,
    )
    coverage = _coverage_binding(
        tmp_path,
        label="matching",
        gate=authentic_gate,
        trace=trace,
        phases=phases,
        parent_graph=parent_graph,
        cache_key=cache_key,
    )
    contract = _contract_for_gate(authentic_gate)
    common = {
        "phases": phases,
        "trace_adapter": trace,
        "coverage_binding": coverage,
        "parent_graph_fingerprint": parent_graph.fingerprint,
        "source_tied_table": tied,
        "policy": policy,
        "external_behavior_contract": contract,
        "model_saver": lambda _model, _path: None,
        "model_loader": lambda _path: None,
        "saver_id": "test-cpcf-saver-v1",
        "loader_id": "test-cpcf-loader-v1",
    }
    common["expected_full_reviewed_evidence_fingerprint"] = (
        authentic_gate.evidence.full_reviewed_evidence_fingerprint
    )
    resources = Lfm2CpcfStackResources(
        **common,
        final_evaluator=authentic_gate,
        fresh_reload_evaluator=authentic_gate,
    )
    adapter = Lfm2CpcfFoldStackAdapter(resources)
    assert adapter.supports(target) is True
    assert contract.body()["lifecycle"] == "reviewed-promotion-rerun"
    assert contract.body()["first_fold_quarantine_required"] is True
    assert contract.body()["first_fold_publication_allowed"] is False
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="parent_relative_manual_packet_sha256",
    ):
        Lfm2CpcfExternalBehaviorGateContract(
            independent_holdout_manifest_sha256=_sha("holdout"),
            independent_parent_envelope_sha256=_sha("parent-envelope"),
            parent_relative_manual_packet_sha256=None,
        )
    forged = object.__new__(Lfm2CpcfReviewedPromotionGate)
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="constructor-authenticated",
    ):
        Lfm2CpcfStackResources(
            **common,
            final_evaluator=forged,
            fresh_reload_evaluator=forged,
        )

    class _UntypedGate:
        def evaluate(self, _context):
            return FoldDecision.accept("untyped")

    bad_final = _UntypedGate()
    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="one exact typed reviewed-promotion gate",
    ):
        Lfm2CpcfStackResources(
            **common,
            final_evaluator=bad_final,
            fresh_reload_evaluator=bad_final,
        )

    for _name, gate in (
        (
            "artifact",
            _authentic_reviewed_gate(
                tmp_path,
                artifact,
                label="wrong-artifact",
                parent_graph=parent_graph,
                target_plan=target_plan,
                target=target,
                cache_key=cache_key,
                trace_adapter_sha256=trace.evidence_sha256,
                phases=phases,
                policy=policy,
                trajectory_corpus_sha256=(
                    trajectory_corpus_sha256
                ),
                bridge_artifact_sha256=_sha("wrong-artifact"),
            ),
        ),
        (
            "profile",
            _authentic_reviewed_gate(
                tmp_path,
                artifact,
                label="wrong-profile",
                parent_graph=parent_graph,
                target_plan=target_plan,
                target=target,
                cache_key=cache_key,
                trace_adapter_sha256=trace.evidence_sha256,
                phases=phases,
                policy=policy,
                trajectory_corpus_sha256=(
                    trajectory_corpus_sha256
                ),
                bridge_profile_sha256=_sha("wrong-profile"),
            ),
        ),
        (
            "semantics",
            _authentic_reviewed_gate(
                tmp_path,
                artifact,
                label="wrong-semantics",
                parent_graph=parent_graph,
                target_plan=target_plan,
                target=target,
                cache_key=cache_key,
                trace_adapter_sha256=trace.evidence_sha256,
                phases=phases,
                policy=policy,
                trajectory_corpus_sha256=(
                    trajectory_corpus_sha256
                ),
                tokenizer_surfaces=[
                    f"<different-{index}>" for index in range(17)
                ],
            ),
        ),
        (
            "package",
            _authentic_reviewed_gate(
                tmp_path,
                artifact,
                label="wrong-package",
                parent_graph=parent_graph,
                target_plan=target_plan,
                target=target,
                cache_key=cache_key,
                trace_adapter_sha256=trace.evidence_sha256,
                phases=phases,
                policy=policy,
                trajectory_corpus_sha256=(
                    trajectory_corpus_sha256
                ),
                extra_tokenizer_config=True,
            ),
        ),
    ):
        with pytest.raises(
            Lfm2CpcfCatalogError,
            match="reviewed promotion evidence is not bound",
        ):
            Lfm2CpcfStackResources(
                **{
                    **common,
                    "expected_full_reviewed_evidence_fingerprint": (
                        gate.evidence.full_reviewed_evidence_fingerprint
                    ),
                    "external_behavior_contract": (
                        _contract_for_gate(gate)
                    ),
                },
                final_evaluator=gate,
                fresh_reload_evaluator=gate,
            )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="independently pinned",
    ):
        Lfm2CpcfStackResources(
            **{
                **common,
                "expected_full_reviewed_evidence_fingerprint": _sha(
                    "wrong-reviewed-evidence-pin"
                ),
            },
            final_evaluator=authentic_gate,
            fresh_reload_evaluator=authentic_gate,
        )

    with pytest.raises(
        Lfm2CpcfCatalogError,
        match="coverage cache, parent identity, graph",
    ):
        Lfm2CpcfStackResources(
            **{
                **common,
                "parent_graph_fingerprint": _sha("wrong-parent-graph"),
            },
            final_evaluator=authentic_gate,
            fresh_reload_evaluator=authentic_gate,
        )


def test_supports_uses_exact_build_validation_and_never_hijacks_fallback(
    tmp_path,
) -> None:
    case = _stack_case(tmp_path, "exact-supports-contract")
    target = case.target
    selected = target.adapter_payload
    assert case.adapter.supports(target) is True

    def assert_rejected(
        label: str,
        adapter: Lfm2CpcfFoldStackAdapter,
        candidate,
    ) -> None:
        assert adapter.supports(candidate) is False, label
        with pytest.raises(
            (Lfm2CpcfCatalogError, DynamicLfm2AdapterError),
        ):
            adapter.build(candidate)
        fallback = SimpleNamespace(
            adapter_id=f"test.fallback.{label}",
            version=1,
            priority=-1,
            target_families=("lfm2",),
            target_modalities=("text",),
            method_ids=adapter.method_ids,
            supports=lambda _target: True,
            build=lambda _target: None,
        )
        registry = DynamicFoldStackAdapterRegistry((adapter, fallback))
        assert registry.select(candidate) is fallback

    wrong_graph_identity = replace(
        target,
        graph=replace(
            target.graph,
            model_name=f"{target.graph.model_name}-forged",
        ),
    )
    assert_rejected(
        "graph-identity",
        case.adapter,
        wrong_graph_identity,
    )

    # A request whose exact count disagrees with its authenticated
    # component-allocation receipt can no longer be constructed at all, so
    # the adapter guard for that inconsistency is unreachable through a real
    # request object: construction itself fails closed.
    with pytest.raises(DynamicTargetSynthesisError):
        replace(
            target,
            request=replace(
                target.request,
                exact_unique_parameters=(
                    target.request.exact_unique_parameters + 1
                ),
            ),
        )

    wrong_profile_count = replace(
        target,
        resource_profile=replace(
            target.resource_profile,
            unique_parameters=(
                target.resource_profile.unique_parameters + 1
            ),
        ),
    )
    assert_rejected(
        "profile-exact-count",
        case.adapter,
        wrong_profile_count,
    )

    wrong_graph_count = replace(
        target,
        graph=replace(
            target.graph,
            unique_parameter_count=(
                target.graph.unique_parameter_count + 1
            ),
            unique_parameter_bytes=(
                target.graph.unique_parameter_bytes + 4
            ),
        ),
    )
    assert_rejected(
        "graph-exact-count",
        case.adapter,
        wrong_graph_count,
    )

    # Payload tampers now fail closed at construction: the adapter payload
    # authenticates its fields against the sealed catalog configuration, so
    # a forged payload can no longer exist to reach the adapter guards.
    wrong_vocabulary_size = selected.vocabulary_size - 1
    with pytest.raises(DynamicLfm2AdapterError):
        replace(
            selected,
            vocabulary_size=wrong_vocabulary_size,
            exact_target_logical_parameters=(
                selected.exact_target_unique_parameters
                + (
                    wrong_vocabulary_size * selected.hidden_size
                    if selected.tie_word_embeddings
                    else 0
                )
            ),
        )

    source_depth = sum(
        len(span) for span in selected.ordered_source_spans
    )
    with pytest.raises(DynamicLfm2AdapterError):
        replace(
            selected,
            layer_types=(selected.layer_types[-1],),
            ordered_source_spans=(tuple(range(source_depth)),),
        )

    wrong_tied_table_adapter = Lfm2CpcfFoldStackAdapter(
        replace(
            case.resources,
            source_tied_table=(
                case.resources.source_tied_table[:-1].clone()
            ),
        )
    )
    assert_rejected(
        "tied-table-row-count",
        wrong_tied_table_adapter,
        target,
    )

    topology_marker = "_neural_forge_dense_topology_catalog"
    original_marker = getattr(target.model.config, topology_marker)
    setattr(target.model.config, topology_marker, "forged-catalog")
    try:
        assert_rejected(
            "live-blank-topology",
            case.adapter,
            target,
        )
    finally:
        setattr(target.model.config, topology_marker, original_marker)


@pytest.mark.parametrize(
    "identity",
    ("target_count", "topology", "blank_state", "parent_state"),
)
def test_supports_rejects_each_reviewed_target_identity_drift(
    tmp_path,
    identity: str,
) -> None:
    case = _stack_case(tmp_path, f"reviewed-{identity}")
    overrides = {
        "target_count": {
            "target_unique_parameters": (
                case.target.resource_profile.unique_parameters + 1
            )
        },
        "topology": {
            "target_topology_fingerprint": _sha("wrong-target-topology")
        },
        "blank_state": {
            "blank_state_fingerprint": _sha("wrong-blank-state")
        },
        "parent_state": {
            "parent_state_fingerprint": _sha("wrong-parent-state")
        },
    }[identity]
    gate = _reviewed_gate_for_stack_case(
        tmp_path,
        case,
        label=f"reviewed-{identity}-drift",
        provenance_overrides=overrides,
    )
    resources = _resources_with_reviewed_gate(case, gate)
    adapter = Lfm2CpcfFoldStackAdapter(resources)

    assert adapter.supports(case.target) is False
    with pytest.raises(Lfm2CpcfCatalogError):
        adapter.build(case.target)


def test_stack_resources_reject_reviewed_cache_trace_policy_and_phase_drift(
    tmp_path,
) -> None:
    case = _stack_case(tmp_path, "reviewed-resource-drift")
    phase_sha256 = tuple(phase.phase_sha256 for phase in case.phases)
    wrong_policy = replace(
        case.policy,
        ridge=case.policy.ridge + 1e-5,
    )
    bad_gates = (
        (
            "cache",
            _reviewed_gate_for_stack_case(
                tmp_path,
                case,
                label="reviewed-wrong-cache",
                provenance_overrides={"cache_key": _sha("wrong-cache")},
            ),
            "coverage cache, parent identity, graph",
        ),
        (
            "trace",
            _reviewed_gate_for_stack_case(
                tmp_path,
                case,
                label="reviewed-wrong-trace",
                provenance_overrides={
                    "trace_adapter_sha256": _sha("wrong-trace")
                },
            ),
            "coverage cache, parent identity, graph",
        ),
        (
            "policy",
            _reviewed_gate_for_stack_case(
                tmp_path,
                case,
                label="reviewed-wrong-policy",
                compiled_policy=wrong_policy,
            ),
            "resource policy differs",
        ),
        (
            "phase",
            _reviewed_gate_for_stack_case(
                tmp_path,
                case,
                label="reviewed-wrong-phase",
                compiled_phase_sha256=(
                    _sha("wrong-compiled-phase"),
                    *phase_sha256[1:],
                ),
            ),
            "resource phases differ",
        ),
    )
    for _label, gate, message in bad_gates:
        with pytest.raises(
            Lfm2CpcfCatalogError,
            match=message,
        ):
            _resources_with_reviewed_gate(case, gate)
