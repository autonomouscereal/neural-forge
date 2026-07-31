from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from experiments import audit_lfm2_exact4x_first_divergence as runner  # noqa: E402
from neural_forge.dynamic_counted_child_state import (  # noqa: E402
    ContinuousChildStateAllocation,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.dynamic_fold_phase_cache import DynamicFoldPhaseCache  # noqa: E402
from neural_forge.dynamic_model_inspector import inspect_model_circuits  # noqa: E402
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (  # noqa: E402
    LFM2_BALANCED_BASIS_PHASE,
    Lfm2BalancedBasisPhaseBuildRequest,
    Lfm2BalancedBasisPhaseCacheContext,
    PhaseAnchorAuthority,
    load_or_build_lfm2_circuit_balanced_basis,
)
from neural_forge.hf_lfm2_exact4x_first_divergence import (  # noqa: E402
    LFM2_EXACT4X_BUDGET_OBSERVATION_SCHEMA,
    LFM2_EXACT4X_FIRST_DIVERGENCE_BATCH_SCHEMA,
    LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA,
    Lfm2Exact4xFirstDivergenceError,
    authenticate_lfm2_exact4x_first_divergence_batch,
    build_lfm2_exact4x_first_divergence_request,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (  # noqa: E402
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    SELECTIVE_BATCH_PAYLOAD_SCHEMA,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_direct_local import (  # noqa: E402
    Lfm2DirectLocalCheckpointIO,
    Lfm2CanonicalFp32RuntimeRelocationReceipt,
    count_lfm2_direct_local_unique_parameter_elements,
    load_lfm2_direct_local_tokenizer,
)
from neural_forge.hf_lfm2_semantic_step_token_fold import (  # noqa: E402
    bind_lfm2_separate_sparse_semantic_code_runtime,
    seal_lfm2_semantic_step_span_evidence,
)
from neural_forge.hf_tokenizer_bridge import (  # noqa: E402
    fingerprint_hf_tokenizer_semantics,
    seal_strict_tokenizer_bridge_artifact,
)
from neural_forge.lfm2_carrier_bridge_publication import (  # noqa: E402
    PACKAGED_ROUTE_SCHEMA,
    STANDALONE_BRIDGE_SCHEMA,
    AuthenticatedPackagedLfm2CarrierRoutes,
    Lfm2PortableCarrierRuntimeReceipt,
    carrier_materialization_tensor_sha256,
)
from neural_forge.lfm2_carrier_child_package import (  # noqa: E402
    Lfm2CarrierChildCheckpointIO,
    Lfm2CarrierChildPackageRegistration,
    load_authenticated_lfm2_carrier_child_package,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _reference(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        input_sha256=_digest(f"input-{index}"),
        record_sha256=_digest(f"record-{index}"),
    )


def _implementation_identity() -> dict[str, object]:
    names = (
        "balanced_basis_phase",
        "canonical_lfm2",
        "carrier_child_package",
        "carrier_relocation_package",
        "core",
        "direct_local_loader",
        "dynamic_blank_execution",
        "routing_runtime",
        "runner",
        "teacher_forced_readout",
        "trace_cache",
        "transformers_lfm2_runtime",
    )
    body = {
        "schema": LFM2_EXACT4X_FIRST_DIVERGENCE_IMPLEMENTATION_SCHEMA,
        "files": {
            name: {"path": f"{name}.py", "sha256": _digest(name)} for name in names
        },
        "packages": {
            "python": "test",
            "safetensors": "test",
            "torch": "test",
            "transformers": "test",
        },
        "numeric_contract": {"runtime_device": "cpu", "runtime_dtype": "float32"},
        "execution_contract": {
            "network_allowed": False,
            "parent_model_loaded": False,
            "parent_forward_calls": 0,
            "training_steps": 0,
        },
    }
    return {**body, "implementation_sha256": canonical_sha256(body)}


def _request() -> dict[str, object]:
    budget_body = {
        "schema": LFM2_EXACT4X_BUDGET_OBSERVATION_SCHEMA,
        "one_complete_child_observed": True,
    }
    budget = {
        **budget_body,
        "budget_observation_sha256": canonical_sha256(budget_body),
    }
    return build_lfm2_exact4x_first_divergence_request(
        child_package_sha256=_digest("child-package"),
        child_state_sha256=_digest("child-state"),
        route_package_sha256=_digest("route-package"),
        route_runtime_sha256=_digest("route-runtime"),
        portable_runtime_receipt_sha256=_digest("portable-runtime"),
        trace_cache_key=_digest("trace-cache"),
        trace_selective_session_sha256=_digest("trace-session"),
        basis_phase_key_sha256=_digest("phase-key"),
        basis_completion_sha256=_digest("phase-completion"),
        basis_map_sha256=_digest("basis-map"),
        budget_observation=budget,
        layer_count=1,
        focus_layer_indices=(0,),
        runtime_device="cpu",
        runtime_dtype="float32",
        maximum_selected_parent_bytes=1024,
        implementation_identity=_implementation_identity(),
    )


def _metrics() -> dict[str, float]:
    return {
        "candidate_norm": 1.0,
        "cosine": 1.0,
        "cosine_distance": 0.0,
        "l2": 0.0,
        "projected_parent_norm": 1.0,
        "relative_l2": 0.0,
        "rmse": 0.0,
    }


def _rank_margin() -> dict[str, object]:
    return {
        "best_competing_score": 0.0,
        "best_competing_token_id": 1,
        "parent_winner_margin": 1.0,
        "parent_winner_rank": 1,
        "parent_winner_score": 1.0,
        "parent_winner_token_id": 0,
        "top1_matches_parent": True,
        "top1_token_id": 0,
    }


def _batch_result(request: dict[str, object], reference, split: str):
    descriptor = {
        "dtype": "float32",
        "shape": [1, 1, 1],
        "numel": 1,
        "size_bytes": 4,
        "sha256": _digest(f"tensor-{reference.index}"),
    }
    selection_body = {
        "schema": SELECTIVE_BATCH_PAYLOAD_SCHEMA,
        "session_sha256": request["trace_selective_session_sha256"],
        "cache_key": request["trace_cache_key"],
        "batch_index": int(reference.index),
        "input_sha256": str(reference.input_sha256),
        "record_sha256": str(reference.record_sha256),
        "ordered_tensor_names": ["prefill.logits"],
        "ordered_tensor_descriptors": [descriptor],
        "logical_bytes": 4,
    }
    selection_sha256 = canonical_sha256(selection_body)
    selected_parent_authentication = {
        **selection_body,
        "selection_sha256": selection_sha256,
        "retained_tensor_descriptors": [descriptor],
        "retained_tensor_set_sha256": canonical_sha256(
            {
                "ordered_tensor_names": ["prefill.logits"],
                "ordered_tensor_descriptors": [descriptor],
            }
        ),
        "typed_selective_session_authenticated": True,
        "typed_selected_payload_authenticated": True,
    }
    state_path = [
        {
            "component_id": "c_in",
            "layer_index": 0,
            "endpoint": "residual.pre",
            "metrics": _metrics(),
            "incremental_relative_l2_delta": None,
        },
        {
            "component_id": "layer.000.residual.post_operator",
            "layer_index": 0,
            "endpoint": "residual.post_operator",
            "metrics": _metrics(),
            "incremental_relative_l2_delta": 0.0,
        },
        {
            "component_id": "layer.000.residual.post_layer",
            "layer_index": 0,
            "endpoint": "residual.post_layer",
            "metrics": _metrics(),
            "incremental_relative_l2_delta": 0.0,
        },
    ]
    branch_components = [
        {
            "component_id": "layer.000.operator.output",
            "layer_index": 0,
            "endpoint": "operator.output",
            "metrics": _metrics(),
        },
        {
            "component_id": "layer.000.mlp.down",
            "layer_index": 0,
            "endpoint": "mlp.down",
            "metrics": _metrics(),
        },
    ]
    body = {
        "schema": LFM2_EXACT4X_FIRST_DIVERGENCE_BATCH_SCHEMA,
        "request_sha256": request["request_sha256"],
        "batch_index": int(reference.index),
        "trace_input_sha256": str(reference.input_sha256),
        "trace_record_sha256": str(reference.record_sha256),
        "selected_payload_sha256": selection_sha256,
        "selected_parent_authentication": selected_parent_authentication,
        "child_embedding_replay_sha256": _digest(f"replay-{reference.index}"),
        "route_execution_context": {
            "route_package_sha256": request["route_package_sha256"],
            "route_runtime_sha256": request["route_runtime_sha256"],
            "portable_runtime_receipt_sha256": request[
                "portable_runtime_receipt_sha256"
            ],
            "input_route_role": "input",
            "output_route_role": "output",
            "typed_live_sessions_authenticated": True,
        },
        "replay_input_authentication": {
            "attention_mask_sha256": _digest(f"mask-{reference.index}"),
            "position_ids_sha256": _digest(f"positions-{reference.index}"),
            "position_ids_present": True,
            "attention_mask_authenticated": True,
            "position_ids_authenticated": True,
        },
        "split": split,
        "complete_child_observed_layer_indices": [0],
        "focused_branch_layer_indices": [0],
        "focus_layers_limit_branch_diagnostics_only": True,
        "noninitial_residual_pre_parent_reads": 0,
        "phase_count": 1,
        "phases": [
            {
                "phase": "prefill",
                "phase_index": 0,
                "consumed_forced_token_count": 0,
                "state_path": state_path,
                "branch_components": branch_components,
                "complete_state_layer_indices": [0],
                "focused_branch_layer_indices": [0],
                "focus_layers_limit_branch_diagnostics_only": True,
                "descriptive_largest_state_error_jump": {
                    "component_id": "layer.000.residual.post_operator",
                    "incremental_relative_l2_delta": 0.0,
                    "not_an_acceptance_threshold": True,
                },
                "c_out": {
                    "cached_parent": _rank_margin(),
                    "folded_child": _rank_margin(),
                    "exact_parent_winner_mismatch_is_discrete_diagnostic": True,
                },
            }
        ],
        "numeric_decision": None,
        "numeric_metrics_are_diagnostic_only": True,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "safety_observations": [],
        "safety_observations_are_separate_not_failures": True,
        "generation_cap_events": [],
        "generation_cap_saturation_is_censored_not_failure": True,
        "operation_audit": {
            "parent_forward_calls": 0,
            "child_forward_calls": 1,
            "child_forward_call_scope": "batch-producing-run",
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "model_mutations": 0,
            "free_generation_calls": 0,
        },
    }
    return {**body, "batch_audit_sha256": canonical_sha256(body)}


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _checkpoint_authority() -> runner.FirstDivergenceCheckpointAuthority:
    return runner.FirstDivergenceCheckpointAuthority(
        key_id="test-checkpoint-key",
        secret=b"test-checkpoint-secret-material-32-bytes!",
    )


def _lfm2_model(*, hidden_size: int, seed: int):
    head_dim = 2 if hidden_size == 2 else 4
    heads = 1 if hidden_size == 2 else hidden_size // head_dim
    config = transformers.Lfm2Config(
        vocab_size=31,
        hidden_size=hidden_size,
        intermediate_size=max(3, hidden_size + hidden_size // 2),
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=1 if hidden_size == 2 else max(1, heads // 2),
        head_dim=head_dim,
        layer_types=["conv", "full_attention"],
        conv_L_cache=3,
        conv_bias=False,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return transformers.Lfm2ForCausalLM(config)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runner_tokenizer_package(path: Path, *, row_count: int):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path.mkdir()
    vocabulary = {"<unk>": 0, "<eos>": 1}
    vocabulary.update({f"token_{index}": index for index in range(2, row_count)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps({"unk_token": "<unk>", "eos_token": "<eos>"}),
        encoding="utf-8",
    )
    return load_lfm2_direct_local_tokenizer(path)


def _runner_identity_carrier_publication(
    root: Path,
    *,
    row_count: int,
    hidden_size: int,
) -> tuple[Path, str]:
    publication = root / "carrier-publication"
    publication.mkdir()
    offsets = torch.arange(row_count + 1, dtype=torch.int64)
    target_ids = torch.arange(row_count, dtype=torch.int64)
    input_coefficients = torch.ones(row_count, dtype=torch.float32)
    output_coefficients = torch.ones(row_count, dtype=torch.float32)
    tensors = {
        "target_table": torch.arange(
            row_count * hidden_size,
            dtype=torch.float32,
        ).reshape(row_count, hidden_size),
        "target_to_donor": target_ids.clone(),
        "input_offsets": offsets,
        "input_target_ids": target_ids,
        "input_coefficients": input_coefficients,
        "output_offsets": offsets.clone(),
        "output_target_ids": target_ids.clone(),
        "output_coefficients": output_coefficients,
    }
    tensor_path = publication / "bridge.safetensors"
    safetensors_torch.save_file(tensors, str(tensor_path))
    coefficient_count = row_count * 2
    operation_audit = {
        "parent_forward_calls": 0,
        "parent_trace_recaptures": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_pruning_calls": 0,
        "parameter_quantization_calls": 0,
        "parameter_sparsification_calls": 0,
        "parameter_sharing_calls": 0,
        "runtime_sidecar_parameters": 0,
    }
    body = {
        "schema": STANDALONE_BRIDGE_SCHEMA,
        "status": "complete",
        "bridge_tensor_file": tensor_path.name,
        "bridge_tensor_file_sha256": _file_sha256(tensor_path),
        "runtime_requires_parent": False,
        "runtime_requires_external_sidecar": False,
        "all_continuous_bridge_state_is_in_bridge_tensor_file": True,
        "integer_csr_routes_are_structural_metadata": True,
        "continuous_semantic_coefficient_count": coefficient_count,
        "physical_target_table_parameter_count": int(tensors["target_table"].numel()),
        "target_table_dtype": str(tensors["target_table"].dtype),
        "operation_audit": operation_audit,
        "materialization_receipt": {
            "input_coefficients_sha256": (
                carrier_materialization_tensor_sha256(input_coefficients)
            ),
            "output_coefficients_sha256": (
                carrier_materialization_tensor_sha256(output_coefficients)
            ),
            "continuous_fold_coefficient_count": coefficient_count,
        },
    }
    receipt = {**body, "artifact_sha256": canonical_sha256(body)}
    (publication / "bridge-receipt.json").write_text(
        json.dumps(receipt, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return publication, str(receipt["artifact_sha256"])


def _publish_real_runner_child(
    root: Path,
    child,
):
    row_count = int(child.config.vocab_size)
    hidden_size = int(child.config.hidden_size)
    semantic_directory = root / "semantic-tokenizer"
    semantic_tokenizer = _runner_tokenizer_package(
        semantic_directory,
        row_count=row_count,
    )
    target_directory = root / "target-tokenizer"
    target_tokenizer = _runner_tokenizer_package(
        target_directory,
        row_count=row_count,
    )
    bridge = seal_strict_tokenizer_bridge_artifact(
        semantic_tokenizer,
        target_tokenizer,
        target_directory,
        donor_row_capacity=row_count,
        target_row_capacity=row_count,
    )
    publication, artifact_sha256 = _runner_identity_carrier_publication(
        root,
        row_count=row_count,
        hidden_size=hidden_size,
    )
    spans = seal_lfm2_semantic_step_span_evidence(
        [[index] for index in range(row_count)],
        target_row_count=row_count,
        lineage={"real_runner_integration": True},
    )
    span_path = root / "semantic-spans.json"
    span_path.write_text(
        json.dumps(spans, allow_nan=False, sort_keys=True) + "\n",
        encoding="ascii",
    )
    registration = Lfm2CarrierChildPackageRegistration(
        publication_directory=publication,
        expected_artifact_sha256=artifact_sha256,
        span_evidence_json=span_path,
        expected_span_evidence_sha256=str(spans["evidence_sha256"]),
        expected_span_evidence_file_sha256=_file_sha256(span_path),
        expected_target_rows=row_count,
        expected_hidden_size=hidden_size,
        expected_input_route_elements=row_count,
        expected_output_route_elements=row_count,
    )
    writer = Lfm2CarrierChildCheckpointIO(
        base=Lfm2DirectLocalCheckpointIO(
            bridge,
            device="cpu",
            dtype=torch.float32,
        ),
        registration=registration,
        semantic_source_tokenizer_directory=semantic_directory,
        semantic_source_tokenizer_sha256=(
            fingerprint_hf_tokenizer_semantics(semantic_tokenizer)
        ),
        in_memory_semantic_source_tokenizer=semantic_tokenizer,
        in_memory_target_tokenizer=target_tokenizer,
    )
    child_directory = root / "real-child"
    child_directory.mkdir()
    writer.save(child, child_directory)
    return load_authenticated_lfm2_carrier_child_package(child_directory)


def _runner_route_context(child):
    donor_count = int(child.config.vocab_size)
    spans = seal_lfm2_semantic_step_span_evidence(
        [[index] for index in range(donor_count)],
        target_row_count=donor_count,
    )
    offsets = torch.arange(donor_count + 1, dtype=torch.int64)
    ids = torch.arange(donor_count, dtype=torch.int64)
    coefficients = torch.ones(donor_count, dtype=torch.float32)
    routing = bind_lfm2_separate_sparse_semantic_code_runtime(
        spans,
        child.get_input_embeddings().weight,
        live_child_output_table=child.get_output_embeddings().weight,
        input_semantic_code_offsets=offsets,
        input_semantic_code_target_ids=ids,
        input_semantic_code_coefficients=coefficients,
        output_semantic_code_offsets=offsets,
        output_semantic_code_target_ids=ids,
        output_semantic_code_coefficients=coefficients.clone(),
        target_to_donor=ids,
    )
    artifact_sha256 = _digest("runner-route-artifact")
    allocations = (
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_in",
            parameter_elements=donor_count,
            tensor_sha256=_digest("runner-c-in"),
            artifact_sha256=artifact_sha256,
        ),
        ContinuousChildStateAllocation(
            state_id="semantic-carrier-route.c_out",
            parameter_elements=donor_count,
            tensor_sha256=_digest("runner-c-out"),
            artifact_sha256=artifact_sha256,
        ),
    )
    route_body = {
        "schema": PACKAGED_ROUTE_SCHEMA,
        "status": "complete",
        "input_route_allocation": allocations[0].as_dict(),
        "output_route_allocation": allocations[1].as_dict(),
        "continuous_semantic_coefficient_count": donor_count * 2,
    }
    route_receipt = {
        **route_body,
        "package_sha256": canonical_sha256(route_body),
    }
    routes = AuthenticatedPackagedLfm2CarrierRoutes(
        root=Path("."),
        receipt=route_receipt,
        tensors={},
        allocations=allocations,
    )
    dense_count = count_lfm2_direct_local_unique_parameter_elements(child)
    direct = Lfm2CanonicalFp32RuntimeRelocationReceipt(
        provenance_resource_fingerprint=_digest("runner-implementation-placeholder"),
        canonical_device="cpu",
        runtime_device="cpu",
        runtime_dtype="float32",
        canonical_state_sha256=_digest("runner-canonical-state"),
        expected_runtime_state_sha256=_digest("runner-runtime-state"),
        runtime_state_sha256=_digest("runner-runtime-state"),
        canonical_state_alias_abi_sha256=_digest("runner-alias"),
        runtime_state_alias_abi_sha256=_digest("runner-alias"),
        canonical_unique_parameter_elements=dense_count,
        runtime_unique_parameter_elements=dense_count,
        executable_forward_smoke={"logits_finite": True, "use_cache": False},
    )
    portable = Lfm2PortableCarrierRuntimeReceipt(
        route_package_sha256=routes.package_sha256,
        source_bridge_artifact_sha256=artifact_sha256,
        direct_local_relocation=direct,
        canonical_input_table_binding={},
        canonical_output_table_binding={},
        runtime_input_table_binding={},
        runtime_output_table_binding={},
        route_allocations=allocations,
        carrier_runtime_sha256=routing.as_dict()["runtime_sha256"],
    )
    return routing, routes, portable


def test_resume_authenticates_hits_and_processes_only_missing_batches(
    tmp_path,
) -> None:
    request = _request()
    authority = _checkpoint_authority()
    references = tuple(_reference(index) for index in range(3))
    split_by_batch = {0: "fit", 1: "fit", 2: "heldout"}
    calls = []

    def process(reference, split):
        calls.append(int(reference.index))
        return _batch_result(request, reference, split)

    first = runner.resume_first_divergence_batches(
        request=request,
        references=references,
        split_by_batch=split_by_batch,
        checkpoint_directory=tmp_path / "batches",
        processor=process,
        checkpoint_authority=authority,
    )
    assert calls == [0, 1, 2]
    assert [value["batch_index"] for value in first.results] == [0, 1, 2]
    assert first.current_run_child_forward_calls == 3
    assert first.historical_checkpoint_child_forward_calls == 0

    calls.clear()
    second = runner.resume_first_divergence_batches(
        request=request,
        references=references,
        split_by_batch=split_by_batch,
        checkpoint_directory=tmp_path / "batches",
        processor=process,
        checkpoint_authority=authority,
    )
    assert calls == []
    assert second.results == first.results
    assert second.current_run_child_forward_calls == 0
    assert second.historical_checkpoint_child_forward_calls == 3

    (tmp_path / "batches" / "batch-000001.json").unlink()
    third = runner.resume_first_divergence_batches(
        request=request,
        references=references,
        split_by_batch=split_by_batch,
        checkpoint_directory=tmp_path / "batches",
        processor=process,
        checkpoint_authority=authority,
    )
    assert calls == [1]
    assert third.results == first.results
    assert third.current_run_child_forward_calls == 1
    assert third.historical_checkpoint_child_forward_calls == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("numeric_metrics_are_diagnostic_only", False),
        ("operation_audit", {"child_forward_calls": 1}),
    ),
)
def test_batch_authentication_rejects_resealed_policy_or_operation_change(
    field,
    value,
) -> None:
    request = _request()
    batch = _batch_result(request, _reference(0), "fit")
    batch[field] = value
    body = dict(batch)
    body.pop("batch_audit_sha256")
    batch["batch_audit_sha256"] = canonical_sha256(body)
    with pytest.raises(
        Lfm2Exact4xFirstDivergenceError,
        match="policy|operation audit",
    ):
        authenticate_lfm2_exact4x_first_divergence_batch(
            batch,
            expected_request=request,
        )


def test_resume_rejects_fully_resealed_cross_batch_result(tmp_path) -> None:
    request = _request()
    authority = _checkpoint_authority()
    reference = _reference(0)
    directory = tmp_path / "batches"
    runner.resume_first_divergence_batches(
        request=request,
        references=(reference,),
        split_by_batch={0: "fit"},
        checkpoint_directory=directory,
        processor=lambda current, split: _batch_result(
            request,
            current,
            split,
        ),
        checkpoint_authority=authority,
    )
    checkpoint_path = directory / "batch-000000.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="ascii"))
    stale_hmac = checkpoint.pop("checkpoint_hmac_sha256")
    result = checkpoint["result"]
    result["trace_input_sha256"] = _digest("foreign-input")
    result_body = dict(result)
    result_body.pop("batch_audit_sha256")
    result["batch_audit_sha256"] = canonical_sha256(result_body)
    checkpoint["batch_audit_sha256"] = result["batch_audit_sha256"]
    checkpoint_body = dict(checkpoint)
    checkpoint_body.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint_body)
    checkpoint["checkpoint_hmac_sha256"] = stale_hmac
    checkpoint_path.write_bytes(_canonical_bytes(checkpoint))

    with pytest.raises(
        runner.Lfm2Exact4xFirstDivergenceRunnerError,
        match="HMAC does not verify",
    ):
        runner.resume_first_divergence_batches(
            request=request,
            references=(reference,),
            split_by_batch={0: "fit"},
            checkpoint_directory=directory,
            processor=lambda *_args: pytest.fail("tamper invoked processor"),
            checkpoint_authority=authority,
        )


def test_resume_rejects_processor_result_for_another_trace(tmp_path) -> None:
    request = _request()
    reference = _reference(0)
    foreign = _reference(1)
    with pytest.raises(
        runner.Lfm2Exact4xFirstDivergenceRunnerError,
        match="differs from its trace reference",
    ):
        runner.resume_first_divergence_batches(
            request=request,
            references=(reference,),
            split_by_batch={0: "fit"},
            checkpoint_directory=tmp_path / "batches",
            processor=lambda _reference, split: _batch_result(
                request,
                foreign,
                split,
            ),
            checkpoint_authority=_checkpoint_authority(),
        )
    assert not (tmp_path / "batches" / "batch-000000.json").exists()


def test_run_uses_real_trace_phase_hmac_and_checkpoint_wiring(
    tmp_path,
    monkeypatch,
) -> None:
    parent = _lfm2_model(hidden_size=16, seed=3101)
    child = _lfm2_model(hidden_size=2, seed=3102)
    graph = inspect_model_circuits(
        parent,
        model_name="runner-first-divergence-parent",
        tokenizer_fingerprint=_digest("runner-parent-tokenizer"),
    )
    calibration = []
    for index in range(4):
        row = {
            "input_ids": torch.tensor(
                [[1 + index, 5 + index, 9 + index]], dtype=torch.long
            ),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[20 + index]], dtype=torch.long),
        }
        if index != 3:
            row["position_ids"] = torch.arange(3, dtype=torch.long).unsqueeze(0)
        calibration.append(row)
    trace_cache = build_lfm2_circuit_trace_cache(
        tmp_path / "trace-root",
        parent,
        calibration,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=1,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    secret = b"runner-integration-hmac-secret-material!"
    key_id = "runner-integration-key"
    phase_root = tmp_path / "phase-root"
    phase_cache = DynamicFoldPhaseCache(phase_root)
    phase_authority = PhaseAnchorAuthority(key_id=key_id, secret=secret)
    context = Lfm2BalancedBasisPhaseCacheContext(
        cache=phase_cache,
        anchor_authority=phase_authority,
        dependency_provider=lambda: {"carrier.route": _digest("runner-route")},
    )
    target = {
        "schema": "test-selected-target-v1",
        "source_graph_fingerprint": graph.fingerprint,
        "source_state_fingerprint": _digest("runner-source-state"),
        "hidden_size": 2,
        "exact_target_unique_parameters": (
            count_lfm2_direct_local_unique_parameter_elements(child)
        ),
    }
    tokenizer = {
        "schema": "test-tokenizer-binding-v1",
        "parent_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "child_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "bridge_required": False,
    }
    with trace_cache.selective_authenticated_session() as session:
        load_or_build_lfm2_circuit_balanced_basis(
            context,
            parent,
            trace_cache,
            2,
            circuit_graph=graph,
            selective_session=session,
            build_request=Lfm2BalancedBasisPhaseBuildRequest(
                adapter_source_sha256=_digest("runner-basis-adapter"),
                validation_fraction=0.25,
                split_seed="runner-first-divergence-v1",
                max_rows_per_block=8,
            ),
            binding_provider=lambda: {
                "selected_target": target,
                "tokenizer": tokenizer,
            },
            live_vocabulary_candidate_gate_required=False,
        )
    assert (phase_root / LFM2_BALANCED_BASIS_PHASE).is_dir()
    phase_key_sha256 = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir()).name
    routing, routes, portable = _runner_route_context(child)
    package_sha256 = _digest("runner-child-package")
    package = SimpleNamespace(
        package_sha256=package_sha256,
        receipt={"hidden_size": 2, "target_rows": 31},
        routes=routes,
    )
    child_directory = tmp_path / "child"
    child_directory.mkdir()
    key_path = tmp_path / "phase.key"
    key_path.write_bytes(secret)
    key_path.chmod(0o600)
    output = tmp_path / "output"
    args = SimpleNamespace(
        child_directory=child_directory,
        child_package_sha256=package_sha256,
        trace_cache=trace_cache.path,
        basis_phase_cache_root=phase_root,
        basis_phase_key_sha256=phase_key_sha256,
        basis_phase_hmac_key_file=key_path,
        basis_phase_hmac_key_id=key_id,
        output_directory=output,
        source_width=16,
        target_hidden_size=2,
        focus_layer_indices="1",
        runtime_device="cpu",
        runtime_dtype="float32",
        maximum_selected_parent_bytes=64 * 1024 * 1024,
        maximum_donor_rows_per_score_chunk=16,
    )
    package_loads = []
    smoke_forwards = []

    def load_package(_directory, *, child_model=None, expected_package_sha256=None):
        if (
            expected_package_sha256 is not None
            and expected_package_sha256 != package_sha256
        ):
            raise runner.Lfm2Exact4xFirstDivergenceRunnerError(
                "test package SHA mismatch"
            )
        if child_model is not None and child_model is not child:
            raise runner.Lfm2Exact4xFirstDivergenceRunnerError(
                "test package bound to foreign child"
            )
        package_loads.append(child_model is not None)
        return package

    def relocate(
        canonical_child,
        packaged_routes,
        *,
        provenance_resource_fingerprint,
        runtime_device,
        runtime_dtype,
        expected_target_rows,
        expected_hidden_size,
    ):
        assert canonical_child is child
        assert packaged_routes is routes
        assert len(provenance_resource_fingerprint) == 64
        assert str(runtime_device) == "cpu"
        assert runtime_dtype is torch.float32
        assert (expected_target_rows, expected_hidden_size) == (31, 2)
        with torch.inference_mode():
            result = child(
                input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
                attention_mask=torch.ones((1, 3), dtype=torch.long),
                use_cache=False,
            )
        assert bool(torch.isfinite(result.logits).all())
        smoke_forwards.append(1)
        return child, routing, portable

    monkeypatch.setattr(
        runner,
        "load_authenticated_lfm2_carrier_child_package",
        load_package,
    )
    monkeypatch.setattr(
        runner,
        "load_lfm2_direct_local_model",
        lambda *_args, **_kwargs: child,
    )
    monkeypatch.setattr(
        runner,
        "relocate_and_bind_packaged_lfm2_carrier_routes",
        relocate,
    )
    parent.forward = lambda *_args, **_kwargs: pytest.fail(
        "run() performed a parent forward"
    )
    state_before = {
        name: value.detach().clone() for name, value in child.state_dict().items()
    }
    completion = runner.run(args)
    assert completion["status"] == "complete-observation"
    assert completion["parent_forward_calls"] == 0
    assert completion["training_steps"] == 0
    assert completion["batch_count"] == 4
    assert completion["completion_scope"] == (
        "stable-persisted-evidence-across-invocations"
    )
    assert completion["invocation_relative_resume_accounting_is_in_progress_journal"]
    assert completion["checkpoint_hmac_algorithm"] == runner.CHECKPOINT_HMAC_ALGORITHM
    assert package_loads == [False, True]
    assert smoke_forwards == [1]
    for name, value in child.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)
    checkpoint_path = output / "batches" / "batch-000000.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="ascii"))
    assert checkpoint["checkpoint_hmac_key_id"] == key_id
    assert len(checkpoint["checkpoint_hmac_sha256"]) == 64

    resumed_completion = runner.run(args)
    assert resumed_completion == completion
    assert package_loads == [False, True, False, True]
    assert smoke_forwards == [1, 1]
    progress_rows = [
        json.loads(row)
        for row in (output / "progress.jsonl").read_text(encoding="ascii").splitlines()
    ]
    resumed_event = progress_rows[-1]
    assert resumed_event["event"] == "run-complete"
    assert resumed_event["details"]["batch_resume_audit"] == {
        "checkpoint_hit_count": 4,
        "computed_batch_count": 0,
        "current_run_child_forward_calls": 0,
        "historical_checkpoint_child_forward_calls": 4,
        "persisted_batch_evidence_child_forward_calls": 4,
    }
    assert resumed_event["details"]["current_run_child_forward_calls"] == 1
    assert resumed_event["details"]["historical_checkpoint_child_forward_calls"] == 4

    original_identity = runner._implementation_identity
    original = original_identity(
        runtime_device=torch.device("cpu"), runtime_dtype_name="float32"
    )
    changed_body = dict(original)
    changed_body.pop("implementation_sha256")
    changed_body["packages"] = {**changed_body["packages"], "torch": "changed"}
    changed = {
        **changed_body,
        "implementation_sha256": canonical_sha256(changed_body),
    }
    monkeypatch.setattr(runner, "_implementation_identity", lambda **_kwargs: changed)
    with pytest.raises(
        runner.Lfm2Exact4xFirstDivergenceRunnerError,
        match="Existing diagnostic request differs",
    ):
        runner.run(args)
    monkeypatch.setattr(runner, "_implementation_identity", original_identity)

    checkpoint = json.loads(checkpoint_path.read_text(encoding="ascii"))
    stale_hmac = checkpoint.pop("checkpoint_hmac_sha256")
    result = checkpoint["result"]
    result["numeric_metrics_are_diagnostic_only"] = False
    result_body = dict(result)
    result_body.pop("batch_audit_sha256")
    result["batch_audit_sha256"] = canonical_sha256(result_body)
    checkpoint["batch_audit_sha256"] = result["batch_audit_sha256"]
    checkpoint_body = dict(checkpoint)
    checkpoint_body.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint_body)
    checkpoint["checkpoint_hmac_sha256"] = stale_hmac
    checkpoint_path.write_bytes(_canonical_bytes(checkpoint))
    with pytest.raises(
        runner.Lfm2Exact4xFirstDivergenceRunnerError,
        match="HMAC does not verify",
    ):
        runner.run(args)

    wrong_key = tmp_path / "wrong-phase.key"
    wrong_key.write_bytes(b"wrong-runner-integration-secret-material!")
    wrong_key.chmod(0o600)
    wrong_args = SimpleNamespace(
        **{
            **vars(args),
            "basis_phase_hmac_key_file": wrong_key,
            "output_directory": tmp_path / "wrong-key-output",
        }
    )
    with pytest.raises(RuntimeError, match="HMAC does not verify"):
        runner.run(wrong_args)


def test_run_uses_real_published_child_package_and_all_hit_resume(
    tmp_path,
) -> None:
    parent = _lfm2_model(hidden_size=16, seed=3201)
    source_child = _lfm2_model(hidden_size=2, seed=3202).eval()
    package = _publish_real_runner_child(tmp_path, source_child)
    child_directory = package.root
    file_hashes_before = {
        str(row["relative_path"]): str(row["sha256"])
        for row in package.receipt["child_file_records"]
    }
    graph = inspect_model_circuits(
        parent,
        model_name="real-package-runner-parent",
        tokenizer_fingerprint=_digest("real-package-runner-tokenizer"),
    )
    calibration = []
    for index in range(4):
        row = {
            "input_ids": torch.tensor(
                [[1 + index, 5 + index, 9 + index]], dtype=torch.long
            ),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "forced_decode_ids": torch.tensor([[20 + index]], dtype=torch.long),
        }
        if index != 3:
            row["position_ids"] = torch.arange(
                3,
                dtype=torch.long,
            ).unsqueeze(0)
        calibration.append(row)
    trace_cache = build_lfm2_circuit_trace_cache(
        tmp_path / "real-trace-root",
        parent,
        calibration,
        config=Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=1,
            max_decode_steps=1,
            max_capture_bytes=64 * 1024 * 1024,
            decode_trace_mode=FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        input_device="cpu",
    )
    secret = b"real-package-runner-hmac-secret-material!"
    key_id = "real-package-runner-key"
    phase_root = tmp_path / "real-phase-root"
    context = Lfm2BalancedBasisPhaseCacheContext(
        cache=DynamicFoldPhaseCache(phase_root),
        anchor_authority=PhaseAnchorAuthority(key_id=key_id, secret=secret),
        dependency_provider=lambda: {"carrier.route": package.routes.package_sha256},
    )
    target = {
        "schema": "test-selected-target-v1",
        "source_graph_fingerprint": graph.fingerprint,
        "source_state_fingerprint": _digest("real-package-source-state"),
        "hidden_size": 2,
        "exact_target_unique_parameters": (
            count_lfm2_direct_local_unique_parameter_elements(source_child)
        ),
    }
    tokenizer = {
        "schema": "test-tokenizer-binding-v1",
        "parent_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "child_tokenizer_semantics_sha256": graph.tokenizer_fingerprint,
        "bridge_required": False,
    }
    with trace_cache.selective_authenticated_session() as session:
        load_or_build_lfm2_circuit_balanced_basis(
            context,
            parent,
            trace_cache,
            2,
            circuit_graph=graph,
            selective_session=session,
            build_request=Lfm2BalancedBasisPhaseBuildRequest(
                adapter_source_sha256=_digest("real-package-basis-adapter"),
                validation_fraction=0.25,
                split_seed="real-package-runner-v1",
                max_rows_per_block=8,
            ),
            binding_provider=lambda: {
                "selected_target": target,
                "tokenizer": tokenizer,
            },
            live_vocabulary_candidate_gate_required=False,
        )
    phase_key_sha256 = next((phase_root / LFM2_BALANCED_BASIS_PHASE).iterdir()).name
    key_path = tmp_path / "real-package-phase.key"
    key_path.write_bytes(secret)
    key_path.chmod(0o600)
    output = tmp_path / "real-package-output"
    args = SimpleNamespace(
        child_directory=child_directory,
        child_package_sha256=package.package_sha256,
        trace_cache=trace_cache.path,
        basis_phase_cache_root=phase_root,
        basis_phase_key_sha256=phase_key_sha256,
        basis_phase_hmac_key_file=key_path,
        basis_phase_hmac_key_id=key_id,
        output_directory=output,
        source_width=16,
        target_hidden_size=2,
        focus_layer_indices="1",
        runtime_device="cpu",
        runtime_dtype="float32",
        maximum_selected_parent_bytes=64 * 1024 * 1024,
        maximum_donor_rows_per_score_chunk=16,
    )

    cold_completion = runner.run(args)
    resumed_completion = runner.run(args)

    assert resumed_completion == cold_completion
    assert cold_completion["status"] == "complete-observation"
    assert cold_completion["child_package_sha256"] == package.package_sha256
    assert cold_completion["batch_count"] == 4
    assert cold_completion["parent_forward_calls"] == 0
    assert cold_completion["training_steps"] == 0
    assert cold_completion["persisted_batch_evidence_child_forward_calls"] == 4
    assert cold_completion["completion_scope"] == (
        "stable-persisted-evidence-across-invocations"
    )
    assert (
        cold_completion["portable_runtime_receipt"][
            "dtype_or_device_parameter_compaction_credit"
        ]
        == 0
    )
    package_after = load_authenticated_lfm2_carrier_child_package(
        child_directory,
        expected_package_sha256=package.package_sha256,
    )
    assert {
        str(row["relative_path"]): _file_sha256(
            child_directory / str(row["relative_path"])
        )
        for row in package_after.receipt["child_file_records"]
    } == file_hashes_before
    progress_rows = [
        json.loads(row)
        for row in (output / "progress.jsonl").read_text(encoding="ascii").splitlines()
    ]
    assert progress_rows[-1]["event"] == "run-complete"
    assert progress_rows[-1]["details"]["batch_resume_audit"] == {
        "checkpoint_hit_count": 4,
        "computed_batch_count": 0,
        "current_run_child_forward_calls": 0,
        "historical_checkpoint_child_forward_calls": 4,
        "persisted_batch_evidence_child_forward_calls": 4,
    }
