from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_lfm2_adapter import Lfm2TargetSynthesisAdapter
from neural_forge.dynamic_lfm2_cpcf_resources import (
    Lfm2CpcfResourceError,
    build_lfm2_cpcf_cache_phase_coverage_binding,
)
from neural_forge.dynamic_lfm2_cpcf_reviewed_gate import (
    ORDERED_PHASE_BINDING_SCHEMA,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_model_inspector import inspect_model_circuits
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
)
from neural_forge.dynamic_target_synthesis import plan_budgeted_dynamic_target
from neural_forge.hf_lfm2_circuit_trace_cache import Lfm2CircuitTraceCache
from neural_forge.hf_lfm2_cpcf_installability import (
    LFM2_CPCF_INSTALLABLE_LAYER_SCHEMA,
    LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA,
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    build_lfm2_cpcf_target_phases,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    LFM2_CPCF_MODEL_FOLD_SCHEMA,
    Lfm2CpcfModelFoldPolicy,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfShortConvV4SolverPolicy,
)
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2TargetTokenizedRecord,
    build_lfm2_strict_tokenizer_bridge_trace_adapter,
)
from neural_forge.hf_tokenizer_bridge import (
    seal_strict_tokenizer_bridge_artifact,
)

import neural_forge.dynamic_lfm2_cpcf_resources as resources_module
import neural_forge.dynamic_lfm2_cpcf_adapter as cpcf_adapter_module


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _tensor_sha256(value: Any) -> str:
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
            "sha256": _tensor_sha256(value),
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
        torch.manual_seed(991)
        model = transformers.Lfm2ForCausalLM(config)
    model.eval()
    return model


def _bridge(
    tmp_path: Path,
    *,
    donor_rows: int = 41,
    target_rows: int = 17,
) -> Any:
    donor = _Tokenizer([f"<{index}>" for index in range(donor_rows)])
    target = _Tokenizer(donor.surfaces[:target_rows])
    root = tmp_path / "target-tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=donor_rows,
        target_row_capacity=target_rows,
    )


def _phase_evidence(
    *,
    cache_key: str,
    donor_input_sha256: str,
    phase_name: str,
) -> Lfm2CpcfInstallablePhaseEvidence:
    layers = []
    for index, operator_kind in enumerate(("short_conv", "full_attention")):
        tensors = {"residual.pre": torch.zeros(1, 1, 8)}
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
    body = {
        "schema": LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA,
        "cache_key": cache_key,
        "batch_index": 0,
        "input_sha256": donor_input_sha256,
        "phase": phase_name,
        "layer_evidence_sha256": [layer.evidence_sha256 for layer in layers],
        "final_norm_field_inventory": {},
        "parent_side_compiler_inputs_ready": True,
    }
    return Lfm2CpcfInstallablePhaseEvidence(
        cache_key=cache_key,
        batch_index=0,
        input_sha256=donor_input_sha256,
        phase=phase_name,
        layers=tuple(layers),
        final_norm_tensors={},
        evidence_sha256=canonical_sha256(body),
    )


def _coverage_case(tmp_path: Path) -> dict[str, Any]:
    bridge = _bridge(tmp_path)
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
    cache_key = _sha("cache")
    donor_input = trace.records[0].donor_cache_input_sha256
    phases = build_lfm2_cpcf_target_phases(
        trace,
        (
            _phase_evidence(
                cache_key=cache_key,
                donor_input_sha256=donor_input,
                phase_name="prefill",
            ),
            _phase_evidence(
                cache_key=cache_key,
                donor_input_sha256=donor_input,
                phase_name="decode.000",
            ),
        ),
    )
    parent = _parent()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
    )
    manifest_body = {
        "cache_key": cache_key,
        "parent": {"identity_sha256": _sha("parent-model")},
        "corpus": {
            "corpus_sha256": _sha("cache-corpus"),
            "batch_count": 1,
        },
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    cache = Lfm2CircuitTraceCache(
        path=tmp_path,
        _manifest=MappingProxyType(manifest),
        _completion=MappingProxyType({}),
    )
    return {
        "cache": cache,
        "trace": trace,
        "phases": phases,
        "graph": graph,
    }


def test_cache_phase_binding_requires_complete_ordered_decode_coverage(
    tmp_path: Path,
) -> None:
    case = _coverage_case(tmp_path)
    binding = build_lfm2_cpcf_cache_phase_coverage_binding(
        cache=case["cache"],
        trace_adapter=case["trace"],
        phases=case["phases"],
        parent_graph=case["graph"],
        trajectory_corpus_sha256=_sha("trajectory"),
    )

    assert binding.phase_count == 2
    assert binding.records[0].phase_names == ("prefill", "decode.000")
    assert binding.records[0].layer_count == 2
    assert binding.as_dict()["installability_audit_rematerialized"] is False
    assert binding.binding_fingerprint == canonical_sha256(binding.body())

    with pytest.raises(Lfm2CpcfResourceError, match="exactly.*phases"):
        build_lfm2_cpcf_cache_phase_coverage_binding(
            cache=case["cache"],
            trace_adapter=case["trace"],
            phases=case["phases"][:1],
            parent_graph=case["graph"],
            trajectory_corpus_sha256=_sha("trajectory"),
        )
    with pytest.raises(Lfm2CpcfResourceError, match="exactly.*phases"):
        build_lfm2_cpcf_cache_phase_coverage_binding(
            cache=case["cache"],
            trace_adapter=case["trace"],
            phases=tuple(reversed(case["phases"])),
            parent_graph=case["graph"],
            trajectory_corpus_sha256=_sha("trajectory"),
        )


def test_cache_phase_binding_rejects_manifest_seal_drift(
    tmp_path: Path,
) -> None:
    case = _coverage_case(tmp_path)
    manifest = case["cache"].manifest
    manifest["corpus"]["corpus_sha256"] = _sha("changed")
    drifted = Lfm2CircuitTraceCache(
        path=tmp_path,
        _manifest=MappingProxyType(manifest),
        _completion=MappingProxyType({}),
    )

    with pytest.raises(Lfm2CpcfResourceError, match="manifest canonical seal"):
        build_lfm2_cpcf_cache_phase_coverage_binding(
            cache=drifted,
            trace_adapter=case["trace"],
            phases=case["phases"],
            parent_graph=case["graph"],
            trajectory_corpus_sha256=_sha("trajectory"),
        )


def _reviewed_for_coverage(
    case: dict[str, Any],
    *,
    coverage: Any,
    policy: Lfm2CpcfModelFoldPolicy,
) -> SimpleNamespace:
    phase_sha256 = tuple(phase.phase_sha256 for phase in case["phases"])
    return SimpleNamespace(
        parent_cache_key=coverage.cache_key,
        cache_live_parent_fingerprint=coverage.cache_parent_fingerprint,
        blank_target_parent_graph_fingerprint=case["graph"].fingerprint,
        trace_adapter_sha256=case["trace"].evidence_sha256,
        trajectory_corpus_sha256=coverage.trajectory_corpus_sha256,
        compiled_model_fold_result_sha256=_sha("compiled-result"),
        compiled_model_fold_policy=policy.as_dict(),
        compiled_model_fold_policy_sha256=canonical_sha256(policy.as_dict()),
        compiled_model_fold_phase_count=len(phase_sha256),
        compiled_model_fold_phase_sha256=phase_sha256,
        compiled_model_fold_phase_binding_sha256=canonical_sha256(
            {
                "schema": ORDERED_PHASE_BINDING_SCHEMA,
                "phase_count": len(phase_sha256),
                "ordered_phase_sha256": list(phase_sha256),
            }
        ),
        compiled_blank_target_report_sha256=_sha("blank-report"),
        compiled_parameter_receipt_sha256=_sha("parameter-receipt"),
    )


def _authenticated_coverage_case(
    tmp_path: Path,
) -> tuple[dict[str, Any], Any, Any, Lfm2CpcfModelFoldPolicy]:
    case = _coverage_case(tmp_path)
    coverage = build_lfm2_cpcf_cache_phase_coverage_binding(
        cache=case["cache"],
        trace_adapter=case["trace"],
        phases=case["phases"],
        parent_graph=case["graph"],
        trajectory_corpus_sha256=_sha("trajectory"),
    )
    policy = Lfm2CpcfModelFoldPolicy(
        minimum_predictive_energy_fraction=0.0,
    )
    reviewed = _reviewed_for_coverage(
        case,
        coverage=coverage,
        policy=policy,
    )
    return case, coverage, reviewed, policy


def test_stack_resource_coverage_binds_reviewed_compiler_chain(
    tmp_path: Path,
) -> None:
    case, coverage, reviewed, policy = _authenticated_coverage_case(tmp_path)

    fields = cpcf_adapter_module._authenticate_cpcf_resource_coverage(
        coverage_binding=coverage,
        phases=case["phases"],
        trace_adapter=case["trace"],
        parent_graph_fingerprint=case["graph"].fingerprint,
        policy=policy,
        reviewed=reviewed,
    )

    assert fields["coverage_binding_fingerprint"] == (coverage.binding_fingerprint)
    assert fields["coverage_cache_manifest_sha256"] == (coverage.cache_manifest_sha256)
    assert fields["coverage_phase_count"] == len(case["phases"])
    assert fields["reviewed_compiled_model_fold_result_sha256"] == (
        reviewed.compiled_model_fold_result_sha256
    )
    assert fields["reviewed_compiled_model_fold_policy_sha256"] == (
        reviewed.compiled_model_fold_policy_sha256
    )
    assert (
        fields["reviewed_compiled_model_fold_phase_binding_sha256"]
        == reviewed.compiled_model_fold_phase_binding_sha256
    )
    assert fields["reviewed_compiled_blank_target_report_sha256"] == (
        reviewed.compiled_blank_target_report_sha256
    )
    assert fields["reviewed_compiled_parameter_receipt_sha256"] == (
        reviewed.compiled_parameter_receipt_sha256
    )


def test_stack_resource_coverage_rejects_each_identity_domain_drift(
    tmp_path: Path,
) -> None:
    case, coverage, reviewed, policy = _authenticated_coverage_case(tmp_path)
    common = {
        "phases": case["phases"],
        "trace_adapter": case["trace"],
        "parent_graph_fingerprint": case["graph"].fingerprint,
        "policy": policy,
        "reviewed": reviewed,
    }

    with pytest.raises(Lfm2CpcfResourceError):
        replace(
            coverage,
            cache_batch_count=coverage.cache_batch_count + 1,
        )
    with pytest.raises(
        cpcf_adapter_module.Lfm2CpcfCatalogError,
        match="exact typed.*coverage",
    ):
        cpcf_adapter_module._authenticate_cpcf_resource_coverage(
            coverage_binding=SimpleNamespace(),
            **common,
        )

    for field_name in (
        "cache_key",
        "cache_parent_fingerprint",
        "parent_graph_fingerprint",
        "trace_adapter_sha256",
        "trajectory_corpus_sha256",
    ):
        drifted = replace(coverage, **{field_name: _sha(field_name)})
        with pytest.raises(
            cpcf_adapter_module.Lfm2CpcfCatalogError,
            match="cache, parent identity, graph, trace adapter, or trajectory",
        ):
            cpcf_adapter_module._authenticate_cpcf_resource_coverage(
                coverage_binding=drifted,
                **common,
            )

    seal_drift = replace(coverage)
    object.__setattr__(
        seal_drift,
        "binding_fingerprint",
        _sha("forged-coverage-seal"),
    )
    with pytest.raises(
        cpcf_adapter_module.Lfm2CpcfCatalogError,
        match="canonical seal",
    ):
        cpcf_adapter_module._authenticate_cpcf_resource_coverage(
            coverage_binding=seal_drift,
            **common,
        )


def test_stack_resource_coverage_rejects_phase_and_policy_drift(
    tmp_path: Path,
) -> None:
    case, coverage, reviewed, policy = _authenticated_coverage_case(tmp_path)
    common = {
        "coverage_binding": coverage,
        "trace_adapter": case["trace"],
        "parent_graph_fingerprint": case["graph"].fingerprint,
        "policy": policy,
        "reviewed": reviewed,
    }

    with pytest.raises(
        cpcf_adapter_module.Lfm2CpcfCatalogError,
        match="exact ordered resource phase",
    ):
        cpcf_adapter_module._authenticate_cpcf_resource_coverage(
            phases=tuple(reversed(case["phases"])),
            **common,
        )

    phase_review_drift = SimpleNamespace(**vars(reviewed))
    phase_review_drift.compiled_model_fold_phase_count += 1
    with pytest.raises(
        cpcf_adapter_module.Lfm2CpcfCatalogError,
        match="reviewed compiler phase",
    ):
        cpcf_adapter_module._authenticate_cpcf_resource_coverage(
            phases=case["phases"],
            **{**common, "reviewed": phase_review_drift},
        )

    policy_review_drift = SimpleNamespace(**vars(reviewed))
    policy_review_drift.compiled_model_fold_policy_sha256 = _sha("wrong-policy")
    with pytest.raises(
        cpcf_adapter_module.Lfm2CpcfCatalogError,
        match="reviewed compiler policy",
    ):
        cpcf_adapter_module._authenticate_cpcf_resource_coverage(
            phases=case["phases"],
            **{**common, "reviewed": policy_review_drift},
        )


def test_receipt_policy_is_reconstructed_exactly_and_seal_bound() -> None:
    expected = Lfm2CpcfModelFoldPolicy(
        ridge=3e-8,
        state_reconstruction_weight=0.07,
        minimum_predictive_energy_fraction=0.91,
        minimum_validation_block_retention=0.12,
        enforce_validation_block_floor=True,
        promotion_compression_ratio=100.0,
        require_promotion_ratio=True,
    )
    body = {
        "schema": LFM2_CPCF_MODEL_FOLD_SCHEMA,
        "policy": expected.as_dict(),
    }
    document = {
        "compiled_model_fold": {
            **body,
            "result_sha256": canonical_sha256(body),
        }
    }

    observed = resources_module._policy_from_receipt(document)
    assert observed.as_dict() == expected.as_dict()

    document["compiled_model_fold"]["policy"]["ridge"] = 0.5
    with pytest.raises(Lfm2CpcfResourceError, match="schema or seal"):
        resources_module._policy_from_receipt(document)


def test_receipt_reconstructs_authenticated_v4_solver_policy() -> None:
    expected = Lfm2CpcfModelFoldPolicy(
        ridge=3e-8,
        short_conv_v4_solver_policy=Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-8, 1e-4),
            variable_projection_passes=2,
            context_objective_weight=0.075,
            target_dtype="bfloat16",
        ),
        promotion_compression_ratio=100.0,
        require_promotion_ratio=True,
    )
    body = {
        "schema": LFM2_CPCF_MODEL_FOLD_SCHEMA,
        "policy": expected.as_dict(),
    }
    document = {
        "compiled_model_fold": {
            **body,
            "result_sha256": canonical_sha256(body),
        }
    }

    observed = resources_module._policy_from_receipt(document)

    assert observed.as_dict() == expected.as_dict()
    assert observed.short_conv_v4_solver_policy is not None
    assert observed.short_conv_v4_solver_policy.verify_authentication()


def test_budget_receipt_round_trips_typed_ratio() -> None:
    budget = resources_module._budget_from_receipt(
        {
            "schema": "neural-forge-dense-parameter-frontier-v1",
            "kind": "budget_request",
            "target_family": "lfm2",
            "target_modalities": ["text"],
            "target_unique_parameters": None,
            "compression_ratio": {"numerator": 100, "denominator": 1},
            "target_vocabulary_size": 4096,
            "match_policy": "at_or_below",
            "maximum_candidates": 8,
        }
    )

    assert budget.compression_ratio == "100"
    assert budget.target_vocabulary_size == 4096


def test_reviewed_plan_reconstruction_materializes_the_exact_zero_blank(
    tmp_path: Path,
) -> None:
    bridge = _bridge(tmp_path, donor_rows=81, target_rows=65)
    config = transformers.Lfm2Config(
        vocab_size=81,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    parent = transformers.Lfm2ForCausalLM(config)
    parent.eval()
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
    )
    budget = DenseParameterBudgetRequest(
        target_family="lfm2",
        target_modalities=("text",),
        compression_ratio=2,
        target_vocabulary_size=65,
        maximum_candidates=64,
    )
    catalogs = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                minimum_hidden_size=32,
                tokenizer_bridge_artifact=bridge,
            ),
        )
    )
    expected = plan_budgeted_dynamic_target(
        graph,
        budget,
        child_name="reviewed-child",
        selected_candidate_rank=0,
        tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        adapter_id=Lfm2TargetSynthesisAdapter.adapter_id,
        catalogs=catalogs,
    )
    candidate = expected.frontier.candidates[0]
    reviewed = SimpleNamespace(
        target_requested_budget=budget.as_dict(),
        selected_target_candidate_rank=0,
        target_frontier_fingerprint=expected.frontier.fingerprint,
        target_certificate_sha256=expected.target_certificate.fingerprint,
        selected_target_candidate_fingerprint=candidate.fingerprint,
        blank_target_parent_graph_fingerprint=graph.fingerprint,
        exact_target_unique_parameters=(
            expected.target_request.exact_unique_parameters
        ),
    )
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=Lfm2Topology(1, 1, 32),
        evidence_fingerprint=_sha("cache"),
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge_artifact=bridge,
    )

    reconstructed, blank = resources_module._plan_from_reviewed_receipt(
        parent_model=parent,
        parent_graph=graph,
        bridge=bridge,
        evidence=reviewed,
        parent_name="parent",
        child_name="reviewed-child",
        blank_seed=0,
        minimum_hidden_size=32,
        target_adapter=adapter,
    )

    assert reconstructed.fingerprint == expected.fingerprint
    assert blank.resource_profile.unique_parameters == (
        expected.target_request.exact_unique_parameters
    )
    assert blank.blank_provenance.compiler_created_blank is True
    assert blank.blank_provenance.all_parameters_zero is True


def test_resource_factory_source_has_no_remote_or_legacy_model_loader_path() -> None:
    source = Path(resources_module.__file__).read_text(encoding="utf-8")
    assert "from_" + "pretrained" not in source
    assert "Auto" + "Model" not in source
    assert "requests." not in source
    assert "http" + "x." not in source


def test_resource_factory_requires_every_independent_production_pin() -> None:
    signature = inspect.signature(resources_module.build_lfm2_cpcf_production_resources)
    for name in (
        "expected_full_reviewed_evidence_fingerprint",
        "expected_cpcf_run_receipt_sha256",
        "expected_cpcf_run_receipt_document_sha256",
        "expected_bridge_artifact_sha256",
        "expected_cache_key",
        "expected_cache_manifest_sha256",
        "expected_trajectory_corpus_sha256",
        "expected_parent_graph_fingerprint",
        "expected_exact_target_unique_parameters",
    ):
        assert signature.parameters[name].default is inspect.Parameter.empty
    resource_signature = inspect.signature(cpcf_adapter_module.Lfm2CpcfStackResources)
    assert (
        resource_signature.parameters["coverage_binding"].default
        is inspect.Parameter.empty
    )
