from __future__ import annotations

import contextlib
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import run_lfm2_sparse_semantic_codebook as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseEventLog,
    FoldPhaseKey,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (
    Lfm2SparseSemanticCandidateChunkCheckpoint,
    Lfm2SparseSemanticCoefficientChunkCheckpoint,
    compile_lfm2_sparse_semantic_codebook,
    discover_lfm2_residual_omp_semantic_candidates,
    discover_lfm2_sparse_semantic_candidates,
    load_lfm2_sparse_semantic_codebook,
    reclose_lfm2_sparse_semantic_codebook_norms,
    save_lfm2_sparse_semantic_codebook,
)


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_candidate_phase_identity_is_reusable_across_k_and_ridge() -> None:
    kwargs = {
        "parent_table_sha256": _sha("parent"),
        "target_table_sha256": _sha("target"),
        "row_start": 0,
        "row_stop": 256,
        "neighbour_pool_size": 32,
        "solve_dtype": "float32",
        "algorithm_sha256": _sha("algorithm"),
    }
    first = runner.build_neighbour_chunk_phase_key(**kwargs)
    second = runner.build_neighbour_chunk_phase_key(**kwargs)
    assert first.phase_key_sha256 == second.phase_key_sha256
    encoded = json.dumps(first.as_dict(), sort_keys=True)
    assert '"k"' not in encoded
    assert "ridge" not in first.as_dict()["numeric_contract"]

    coefficient = runner.build_coefficient_chunk_phase_key(
        parent_table_sha256=_sha("parent"),
        target_table_sha256=_sha("target"),
        neighbour_completion_sha256=_sha("neighbours"),
        row_start=0,
        row_stop=256,
        k=8,
        ridge=1e-5,
        solve_dtype="float64",
        algorithm_sha256=_sha("algorithm"),
    )
    changed_k = runner.build_coefficient_chunk_phase_key(
        parent_table_sha256=_sha("parent"),
        target_table_sha256=_sha("target"),
        neighbour_completion_sha256=_sha("neighbours"),
        row_start=0,
        row_stop=256,
        k=4,
        ridge=1e-5,
        solve_dtype="float64",
        algorithm_sha256=_sha("algorithm"),
    )
    changed_ridge = runner.build_coefficient_chunk_phase_key(
        parent_table_sha256=_sha("parent"),
        target_table_sha256=_sha("target"),
        neighbour_completion_sha256=_sha("neighbours"),
        row_start=0,
        row_stop=256,
        k=8,
        ridge=1e-4,
        solve_dtype="float64",
        algorithm_sha256=_sha("algorithm"),
    )
    assert (
        len(
            {
                coefficient.phase_key_sha256,
                changed_k.phase_key_sha256,
                changed_ridge.phase_key_sha256,
            }
        )
        == 3
    )

    omp_kwargs = {
        "parent_table_sha256": _sha("parent"),
        "target_table_sha256": _sha("target"),
        "target_to_donor_sha256": _sha("mapping"),
        "active_k": 8,
        "orthogonality_tolerance": 1e-6,
        "numeric_backend_receipt_sha256": _sha("numeric-backend"),
        "donor_rows_per_chunk": 256,
        "selection_placement_receipt_sha256": _sha("placement"),
        "algorithm_sha256": _sha("algorithm"),
    }
    omp = runner.build_residual_omp_candidate_inventory_phase_key(**omp_kwargs)
    omp_deeper = runner.build_residual_omp_candidate_inventory_phase_key(
        **{**omp_kwargs, "active_k": 16}
    )
    assert omp.phase_name == runner.RESIDUAL_OMP_CANDIDATE_INVENTORY_PHASE
    assert omp.phase_key_sha256 != omp_deeper.phase_key_sha256
    omp_other_backend = runner.build_residual_omp_candidate_inventory_phase_key(
        **{
            **omp_kwargs,
            "numeric_backend_receipt_sha256": _sha("other-numeric-backend"),
        }
    )
    assert omp_other_backend.phase_key_sha256 != omp.phase_key_sha256
    assert omp.as_dict()["abi"]["support_prefixes_reusable_across_active_k"] is True


def test_tensor_phase_is_immutable_and_resumable(tmp_path: Path) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    key = runner.build_neighbour_chunk_phase_key(
        parent_table_sha256=_sha("parent"),
        target_table_sha256=_sha("target"),
        row_start=0,
        row_stop=2,
        neighbour_pool_size=3,
        solve_dtype="float32",
        algorithm_sha256=_sha("algorithm"),
    )
    calls = 0

    def compute(progress):
        nonlocal calls
        calls += 1
        progress({"row_stop": 2})
        return {
            "candidate_offsets": torch.tensor([0, 3, 6], dtype=torch.int64),
            "candidate_target_ids": torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.int64),
        }, {"row_start": 0, "row_stop": 2}

    first = runner.cache_or_compute_tensor_phase(
        cache=cache,
        events=FoldPhaseEventLog(tmp_path / "first.jsonl", run_id="first"),
        phase_key=key,
        compute_call=compute,
    )
    second = runner.cache_or_compute_tensor_phase(
        cache=cache,
        events=FoldPhaseEventLog(tmp_path / "second.jsonl", run_id="second"),
        phase_key=key,
        compute_call=compute,
    )
    assert calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.completion_sha256 == second.completion_sha256
    torch.testing.assert_close(
        first.tensors["candidate_target_ids"],
        second.tensors["candidate_target_ids"],
    )


class _FlushTrackingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def test_tensor_phase_reports_every_cache_and_compute_transition(
    tmp_path: Path,
) -> None:
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    key = runner.build_neighbour_chunk_phase_key(
        parent_table_sha256=_sha("parent"),
        target_table_sha256=_sha("target"),
        row_start=0,
        row_stop=1,
        neighbour_pool_size=1,
        solve_dtype="float32",
        algorithm_sha256=_sha("math-abi"),
    )
    stream = _FlushTrackingStream()
    reporter = runner.CompileProgressReporter(stream=stream)

    def compute(progress):
        progress({"stage": "chunk-complete", "row_stop": 1})
        return {"value": torch.ones(1)}, {"row_stop": 1}

    first = runner.cache_or_compute_tensor_phase(
        cache=cache,
        events=FoldPhaseEventLog(tmp_path / "first.jsonl", run_id="first"),
        phase_key=key,
        compute_call=compute,
        console_progress_call=reporter,
    )
    second = runner.cache_or_compute_tensor_phase(
        cache=cache,
        events=FoldPhaseEventLog(tmp_path / "second.jsonl", run_id="second"),
        phase_key=key,
        compute_call=compute,
        console_progress_call=reporter,
    )
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    stages = [line["details"]["stage"] for line in lines]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert stages == [
        "cache-lookup-start",
        "cache-miss",
        "compute-start",
        "compute-progress",
        "publication-complete",
        "cache-lookup-start",
        "cache-hit",
    ]
    assert all(
        line["schema"] == runner.COMPILE_CONSOLE_PROGRESS_SCHEMA for line in lines
    )
    assert stream.flush_count == len(lines)


def test_phase_math_abis_are_explicit_and_ignore_source_observation_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = runner.sparse_semantic_phase_math_abi_receipts()
    monkeypatch.setattr(runner, "_file_sha256", lambda _path: _sha("logging-only-edit"))
    second = runner.sparse_semantic_phase_math_abi_receipts()
    assert second == first
    assert set(first) == {
        "top_cosine_candidate_selection",
        "residual_omp_support_selection",
        "coefficient_ridge_refit",
        "coefficient_norm_reclosure",
    }
    assert len({receipt["math_abi_sha256"] for receipt in first.values()}) == 4
    assert all(receipt["descriptor"]["version"] == 1 for receipt in first.values())


def test_deferred_runtime_relocation_preserves_state_and_dtype_contract() -> None:
    child = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    child.requires_grad_(False)
    expected = runner.fingerprint_lfm2_direct_local_state(child)
    receipt = runner.relocate_child_for_runtime(
        child,
        runtime_device=torch.device("cpu"),
        forward_dtype=torch.float32,
        expected_state_sha256=expected,
    )
    assert receipt["compile_staging_device"] == "cpu"
    assert receipt["runtime_device"] == "cpu"
    assert receipt["forward_dtype"] == "float32"
    assert receipt["state_sha256_before_relocation"] == expected
    assert receipt["state_sha256_after_relocation"] == expected
    assert receipt["analytic_compilation_completed_before_runtime_relocation"] is True
    assert runner.resolve_runtime_accumulator_dtype(torch.float16) == torch.float32
    assert runner.resolve_runtime_accumulator_dtype(torch.bfloat16) == torch.float32
    assert runner.resolve_runtime_accumulator_dtype(torch.float32) == torch.float32
    assert runner.resolve_runtime_accumulator_dtype(torch.float64) == torch.float64

    contract = runner.build_numeric_dtype_contract(
        parent_source_receipt={
            "source_storage_dtype": "bfloat16",
            "publication_dtype": "float32",
        },
        portable_load_receipt={
            "provenance_dtype": "torch.bfloat16",
            "runtime_dtype": "torch.bfloat16",
        },
        runtime_dtype_receipt={
            "requested_dtype": "bfloat16",
            "resolved_dtype": "bfloat16",
        },
        selection_receipt={"resolved_dtype": "float32"},
        solve_receipt={"resolved_dtype": "float64"},
        forward_dtype=torch.bfloat16,
        norm_reclosure_receipt={"resolved_dtype": "float64"},
    )
    assert contract["parent_source_storage_dtype"] == "bfloat16"
    assert contract["forward_dtype"] == "bfloat16"
    assert contract["routing_accumulator_dtype"] == "float32"
    assert contract["candidate_selection_dtype"] == "float32"
    assert contract["coefficient_solve_dtype"] == "float64"
    assert contract["norm_reclosure_dtype"] == "float64"
    assert contract["low_precision_forward_accumulates_in_float32"] is True


def test_run_stages_child_on_cpu_until_runtime_setup() -> None:
    source = inspect.getsource(runner.run)
    load_index = source.index('runtime_device=torch.device("cpu")')
    relocation_index = source.index("runtime_relocation_receipt = relocate_child")
    candidate_index = source.index("candidate_phase = cache_or_compute_artifact_phase")
    assert load_index < candidate_index < relocation_index
    assert "runtime_device=torch.device(device)" not in source[:relocation_index]


def test_core_chunk_adapter_reuses_candidates_and_content_bound_coefficients(
    tmp_path: Path,
) -> None:
    anchors = torch.eye(4, dtype=torch.float32)
    donor = torch.cat(
        (
            anchors,
            torch.tensor(
                [[0.75, -0.25, 0.0, 0.0], [0.0, 0.6, 0.0, 0.8]],
                dtype=torch.float32,
            ),
        )
    )
    mapping = torch.arange(4, dtype=torch.int64)
    cache = DynamicFoldPhaseCache(tmp_path / "cache")

    first_adapter = runner.CoreChunkPhaseAdapter(
        cache=cache,
        events_directory=tmp_path / "events-first",
        candidate_checkpoint_type=Lfm2SparseSemanticCandidateChunkCheckpoint,
        coefficient_checkpoint_type=Lfm2SparseSemanticCoefficientChunkCheckpoint,
        support_math_abi_sha256=_sha("support-math-abi"),
        coefficient_math_abi_sha256=_sha("coefficient-math-abi"),
    )
    first_candidates = discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        mapping,
        maximum_candidates=4,
        donor_rows_per_chunk=2,
        candidate_chunk_loader=first_adapter.load_candidate,
        candidate_chunk_publisher=first_adapter.publish_candidate,
    )
    first_adapter.assert_drained()
    assert first_adapter.audit["candidate_chunk_publications"] == 3

    second_adapter = runner.CoreChunkPhaseAdapter(
        cache=cache,
        events_directory=tmp_path / "events-second",
        candidate_checkpoint_type=Lfm2SparseSemanticCandidateChunkCheckpoint,
        coefficient_checkpoint_type=Lfm2SparseSemanticCoefficientChunkCheckpoint,
        support_math_abi_sha256=_sha("support-math-abi"),
        coefficient_math_abi_sha256=_sha("coefficient-math-abi"),
    )
    reused_candidates = discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        mapping,
        maximum_candidates=4,
        donor_rows_per_chunk=2,
        candidate_chunk_loader=second_adapter.load_candidate,
        candidate_chunk_publisher=second_adapter.publish_candidate,
    )
    second_adapter.assert_drained()
    assert second_adapter.audit["candidate_chunk_cache_hits"] == 3
    assert reused_candidates.content_sha256 == first_candidates.content_sha256

    changed_support_adapter = runner.CoreChunkPhaseAdapter(
        cache=cache,
        events_directory=tmp_path / "events-changed-support",
        candidate_checkpoint_type=Lfm2SparseSemanticCandidateChunkCheckpoint,
        coefficient_checkpoint_type=Lfm2SparseSemanticCoefficientChunkCheckpoint,
        support_math_abi_sha256=_sha("changed-support-math-abi"),
        coefficient_math_abi_sha256=_sha("coefficient-math-abi"),
    )
    discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        mapping,
        maximum_candidates=4,
        donor_rows_per_chunk=2,
        candidate_chunk_loader=changed_support_adapter.load_candidate,
        candidate_chunk_publisher=changed_support_adapter.publish_candidate,
    )
    changed_support_adapter.assert_drained()
    assert changed_support_adapter.audit["candidate_chunk_cache_misses"] == 3

    first_codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=first_candidates,
        maximum_candidates=4,
        active_k=2,
        donor_rows_per_chunk=2,
        coefficient_chunk_loader=first_adapter.load_coefficient,
        coefficient_chunk_publisher=first_adapter.publish_coefficient,
    )
    first_adapter.assert_drained()
    assert first_adapter.audit["coefficient_chunk_publications"] == 3

    reused_codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=reused_candidates,
        maximum_candidates=4,
        active_k=2,
        donor_rows_per_chunk=2,
        coefficient_chunk_loader=second_adapter.load_coefficient,
        coefficient_chunk_publisher=second_adapter.publish_coefficient,
    )
    second_adapter.assert_drained()
    assert second_adapter.audit["coefficient_chunk_cache_hits"] == 3
    torch.testing.assert_close(
        reused_codebook.semantic_code_coefficients,
        first_codebook.semantic_code_coefficients,
    )

    changed_solve_adapter = runner.CoreChunkPhaseAdapter(
        cache=cache,
        events_directory=tmp_path / "events-changed-solve",
        candidate_checkpoint_type=Lfm2SparseSemanticCandidateChunkCheckpoint,
        coefficient_checkpoint_type=Lfm2SparseSemanticCoefficientChunkCheckpoint,
        support_math_abi_sha256=_sha("support-math-abi"),
        coefficient_math_abi_sha256=_sha("changed-coefficient-math-abi"),
    )
    compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=reused_candidates,
        maximum_candidates=4,
        active_k=2,
        donor_rows_per_chunk=2,
        coefficient_chunk_loader=changed_solve_adapter.load_coefficient,
        coefficient_chunk_publisher=changed_solve_adapter.publish_coefficient,
    )
    changed_solve_adapter.assert_drained()
    assert changed_solve_adapter.audit["coefficient_chunk_cache_misses"] == 3


def _sealed_request(body: dict[str, object]) -> dict[str, object]:
    return {**body, "request_sha256": canonical_sha256(body)}


def test_top_cosine_candidate_request_phase_key_remains_byte_stable() -> None:
    body = {
        "schema": runner._TOP_COSINE_CANDIDATE_REQUEST_SCHEMA,
        "phase_cache_key": _sha("top-cosine-phase"),
        "donor_row_start": 0,
        "donor_row_stop": 256,
        "donor_table_sha256": _sha("donor"),
        "anchor_table_sha256": _sha("anchor"),
        "target_to_donor_sha256": _sha("mapping"),
        "maximum_candidates": 32,
        "selection_device": "cuda:0",
        "selection_dtype": "float32",
    }
    request = _sealed_request(body)
    expected = FoldPhaseKey(
        phase_name=runner.NEIGHBOUR_PHASE,
        dependency_sha256s={
            "anchor_table_sha256": _sha("anchor"),
            "candidate_phase_cache_key": _sha("top-cosine-phase"),
            "donor_table_sha256": _sha("donor"),
            "target_to_donor_sha256": _sha("mapping"),
        },
        numeric_contract={
            "donor_row_start": 0,
            "donor_row_stop": 256,
            "maximum_candidates": 32,
            "selection_device": "cuda:0",
            "selection_dtype": "float32",
        },
        abi={
            "schema": runner._TOP_COSINE_CANDIDATE_REQUEST_SCHEMA,
            "core_request_sha256": request["request_sha256"],
            "active_k_or_ridge_in_identity": False,
            "reusable_across_k_and_ridge": True,
        },
    )
    actual = runner._candidate_request_phase_key(request)
    assert actual.as_dict() == expected.as_dict()
    assert actual.phase_key_sha256 == expected.phase_key_sha256
    assert "implementation_sha256" not in actual.as_dict()["dependency_sha256s"]


def test_residual_omp_prefix_phase_key_binds_depth_and_numeric_contract() -> None:
    base_body = {
        "schema": runner._RESIDUAL_OMP_CANDIDATE_REQUEST_SCHEMA,
        "algorithm": "full-anchor-residual-omp-test",
        "base_phase_cache_key": _sha("base-phase"),
        "donor_row_start": 256,
        "donor_row_stop": 512,
        "donor_table_sha256": _sha("donor"),
        "anchor_table_sha256": _sha("anchor"),
        "implementation_sha256": _sha("implementation"),
        "numeric_backend_receipt_sha256": _sha("numeric-backend"),
        "target_to_donor_sha256": _sha("mapping"),
        "orthogonality_tolerance": 1e-6,
        "selection_device": "cuda:0",
        "selection_dtype": "float32",
        "final_active_k_intentionally_excluded": True,
    }
    base_request_sha = canonical_sha256(base_body)
    request = _sealed_request(
        {**base_body, "base_request_sha256": base_request_sha, "support_depth": 8}
    )
    key = runner._candidate_request_phase_key(request)
    assert key.phase_name == runner.RESIDUAL_OMP_PREFIX_PHASE
    assert key.as_dict()["numeric_contract"] == {
        "donor_row_start": 256,
        "donor_row_stop": 512,
        "support_depth": 8,
        "orthogonality_tolerance": 1e-6,
        "selection_device": "cuda:0",
        "selection_dtype": "float32",
    }
    assert key.as_dict()["dependency_sha256s"]["base_phase_cache_key"] == _sha(
        "base-phase"
    )
    assert key.as_dict()["dependency_sha256s"][
        "numeric_backend_receipt_sha256"
    ] == _sha("numeric-backend")
    deeper = runner._candidate_request_phase_key(
        _sealed_request(
            {**base_body, "base_request_sha256": base_request_sha, "support_depth": 16}
        )
    )
    assert deeper.phase_key_sha256 != key.phase_key_sha256
    assert (
        deeper.as_dict()["dependency_sha256s"]["base_request_sha256"]
        == key.as_dict()["dependency_sha256s"]["base_request_sha256"]
    )
    changed_implementation_body = {
        **base_body,
        "implementation_sha256": _sha("changed-implementation"),
    }
    changed_implementation_base_sha = canonical_sha256(changed_implementation_body)
    changed_implementation = runner._candidate_request_phase_key(
        _sealed_request(
            {
                **changed_implementation_body,
                "base_request_sha256": changed_implementation_base_sha,
                "support_depth": 8,
            }
        )
    )
    assert changed_implementation.phase_key_sha256 != key.phase_key_sha256
    assert changed_implementation.as_dict()["dependency_sha256s"][
        "implementation_sha256"
    ] == _sha("changed-implementation")
    changed_backend_body = {
        **base_body,
        "numeric_backend_receipt_sha256": _sha("changed-numeric-backend"),
    }
    changed_backend_base_sha = canonical_sha256(changed_backend_body)
    changed_backend = runner._candidate_request_phase_key(
        _sealed_request(
            {
                **changed_backend_body,
                "base_request_sha256": changed_backend_base_sha,
                "support_depth": 8,
            }
        )
    )
    assert changed_backend.phase_key_sha256 != key.phase_key_sha256


def test_residual_omp_adapter_reuses_and_extends_support_prefixes(
    tmp_path: Path,
) -> None:
    anchors = torch.eye(4, dtype=torch.float32)
    donor = torch.cat(
        (
            anchors,
            torch.tensor(
                [[0.75, -0.25, 0.2, 0.1], [0.1, 0.6, -0.3, 0.8]],
                dtype=torch.float32,
            ),
        )
    )
    mapping = torch.arange(4, dtype=torch.int64)
    cache = DynamicFoldPhaseCache(tmp_path / "cache")

    first_adapter = runner.CoreChunkPhaseAdapter(
        cache=cache,
        events_directory=tmp_path / "events-k2",
        candidate_checkpoint_type=Lfm2SparseSemanticCandidateChunkCheckpoint,
        coefficient_checkpoint_type=Lfm2SparseSemanticCoefficientChunkCheckpoint,
        support_math_abi_sha256=_sha("support-math-abi"),
        coefficient_math_abi_sha256=_sha("coefficient-math-abi"),
    )
    first = discover_lfm2_residual_omp_semantic_candidates(
        donor,
        anchors,
        mapping,
        active_k=2,
        implementation_sha256=_sha("core-implementation"),
        donor_rows_per_chunk=2,
        selection_dtype=torch.float64,
        candidate_chunk_loader=first_adapter.load_candidate,
        candidate_chunk_publisher=first_adapter.publish_candidate,
    )
    first_adapter.assert_drained()
    assert first.as_dict()["active_k"] == 2
    assert first_adapter.audit["candidate_chunk_publications"] == 6

    extending_adapter = runner.CoreChunkPhaseAdapter(
        cache=cache,
        events_directory=tmp_path / "events-k3",
        candidate_checkpoint_type=Lfm2SparseSemanticCandidateChunkCheckpoint,
        coefficient_checkpoint_type=Lfm2SparseSemanticCoefficientChunkCheckpoint,
        support_math_abi_sha256=_sha("support-math-abi"),
        coefficient_math_abi_sha256=_sha("coefficient-math-abi"),
    )
    extended = discover_lfm2_residual_omp_semantic_candidates(
        donor,
        anchors,
        mapping,
        active_k=3,
        implementation_sha256=_sha("core-implementation"),
        donor_rows_per_chunk=2,
        selection_dtype=torch.float64,
        candidate_chunk_loader=extending_adapter.load_candidate,
        candidate_chunk_publisher=extending_adapter.publish_candidate,
    )
    extending_adapter.assert_drained()
    assert extended.as_dict()["active_k"] == 3
    assert extending_adapter.audit["candidate_chunk_cache_hits"] == 3
    assert extending_adapter.audit["candidate_chunk_cache_misses"] == 3
    assert extending_adapter.audit["candidate_chunk_publications"] == 3
    assert extending_adapter.audit["candidate_support_prefix_cache_hits_by_depth"] == {
        "2": 3
    }
    assert extending_adapter.audit[
        "candidate_support_prefix_publications_by_depth"
    ] == {"3": 3}


def test_residual_omp_depth16_legacy_domain_extends_without_researching_prefix(
    tmp_path: Path,
) -> None:
    target_count = 17
    anchors = torch.eye(target_count, dtype=torch.float64)
    donor = torch.cat(
        (
            anchors,
            torch.linspace(-0.7, 0.9, target_count, dtype=torch.float64).reshape(1, -1),
        )
    )
    mapping = torch.arange(target_count, dtype=torch.int64)
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    pending: dict[str, tuple[FoldPhaseKey, FoldPhaseEventLog]] = {}

    def legacy_load(request):
        value = runner._verified_core_request(request, "legacy candidate chunk")
        key = runner._candidate_request_phase_key(value)
        events = FoldPhaseEventLog(
            tmp_path / f"legacy-{value['request_sha256']}.jsonl",
            run_id=f"legacy-{value['request_sha256']}",
        )
        hit = cache.lookup(key, events=events)
        if hit is None:
            pending[value["request_sha256"]] = (key, events)
            return None
        tensors, receipt = runner._load_tensor_phase_entry(hit)
        return Lfm2SparseSemanticCandidateChunkCheckpoint(
            tensors["candidate_offsets"],
            tensors["candidate_target_ids"],
            receipt["metadata"]["checkpoint_receipt"],
        )

    def legacy_publish(checkpoint):
        receipt = checkpoint.as_dict()
        key, events = pending.pop(receipt["request_sha256"])
        runner.cache_or_compute_tensor_phase(
            cache=cache,
            events=events,
            phase_key=key,
            compute_call=lambda _progress: (
                {
                    "candidate_offsets": checkpoint.candidate_offsets,
                    "candidate_target_ids": checkpoint.candidate_target_ids,
                },
                {"checkpoint_receipt": receipt},
            ),
        )

    legacy = discover_lfm2_residual_omp_semantic_candidates(
        donor,
        anchors,
        mapping,
        active_k=16,
        implementation_sha256=(
            runner._LEGACY_CORE_MATH_CHECKPOINT_COMPATIBILITY_SHA256
        ),
        donor_rows_per_chunk=target_count + 1,
        selection_dtype=torch.float64,
        candidate_chunk_loader=legacy_load,
        candidate_chunk_publisher=legacy_publish,
    )
    assert not pending
    assert legacy.as_dict()["active_k"] == 16
    assert (
        legacy.as_dict()["analytic_operation_audit"]["newly_searched_support_steps"]
        == 16
    )

    math_abis = runner.sparse_semantic_phase_math_abi_receipts()
    extending_adapter = runner.CoreChunkPhaseAdapter(
        cache=cache,
        events_directory=tmp_path / "new-abi-events",
        candidate_checkpoint_type=Lfm2SparseSemanticCandidateChunkCheckpoint,
        coefficient_checkpoint_type=Lfm2SparseSemanticCoefficientChunkCheckpoint,
        support_math_abi_sha256=math_abis["residual_omp_support_selection"][
            "math_abi_sha256"
        ],
        coefficient_math_abi_sha256=math_abis["coefficient_ridge_refit"][
            "math_abi_sha256"
        ],
    )
    extended = discover_lfm2_residual_omp_semantic_candidates(
        donor,
        anchors,
        mapping,
        active_k=17,
        implementation_sha256=(
            runner._LEGACY_CORE_MATH_CHECKPOINT_COMPATIBILITY_SHA256
        ),
        donor_rows_per_chunk=target_count + 1,
        selection_dtype=torch.float64,
        candidate_chunk_loader=extending_adapter.load_candidate,
        candidate_chunk_publisher=extending_adapter.publish_candidate,
    )
    extending_adapter.assert_drained()
    assert extended.as_dict()["active_k"] == 17
    extended_audit = extended.as_dict()["analytic_operation_audit"]
    assert (
        extended.as_dict()["base_phase_cache_key"]
        == legacy.as_dict()["base_phase_cache_key"]
    )
    assert extended_audit["loaded_prefix_steps_skipping_dictionary_search"] == 16
    assert extended_audit["newly_searched_support_steps"] == 1
    assert extended_audit["full_target_dictionary_residual_matmul_calls"] == 1
    assert extending_adapter.audit["candidate_chunk_cache_hits"] == 1
    assert extending_adapter.audit["candidate_chunk_legacy_compatible_cache_hits"] == 1
    assert extending_adapter.audit["candidate_chunk_cache_misses"] == 1
    assert extending_adapter.audit["candidate_support_prefix_cache_hits_by_depth"] == {
        "16": 1
    }
    assert extending_adapter.audit[
        "candidate_support_prefix_publications_by_depth"
    ] == {"17": 1}


def test_effective_accounting_counts_every_continuous_c_scalar() -> None:
    receipt = runner.effective_child_parameter_accounting(
        physical_child_parameters=100,
        semantic_code_coefficients=torch.ones(24, dtype=torch.float32),
        parent_parameters=248,
    )
    assert receipt["continuous_semantic_code_coefficient_count"] == 24
    assert receipt["effective_child_parameter_count"] == 124
    assert receipt["effective_parameter_compaction_ratio"] == 2.0
    assert receipt["integer_routing_metadata_counted_as_parameters"] == 0
    with pytest.raises(runner.Lfm2SparseSemanticRunnerError, match="FP32"):
        runner.effective_child_parameter_accounting(
            physical_child_parameters=100,
            semantic_code_coefficients=torch.ones(24, dtype=torch.bfloat16),
            parent_parameters=248,
        )


def _tiny_sparse_codebook():
    anchors = torch.eye(4, dtype=torch.float32)
    donor = torch.cat(
        (
            anchors,
            torch.tensor(
                [
                    [0.6, 0.6, 0.6, 0.0],
                    [0.5, 0.5, 0.5, 0.5],
                ],
                dtype=torch.float32,
            ),
        )
    )
    mapping = torch.arange(4, dtype=torch.int64)
    candidates = discover_lfm2_sparse_semantic_candidates(
        donor,
        anchors,
        mapping,
        maximum_candidates=4,
        donor_rows_per_chunk=2,
    )
    codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        anchors,
        mapping,
        candidate_inventory=candidates,
        maximum_candidates=4,
        active_k=2,
        candidate_ridge=1e-5,
        refit_ridge=1e-5,
        donor_rows_per_chunk=2,
    )
    return donor, anchors, mapping, codebook


def test_base_codebook_override_authenticates_without_recompilation(
    tmp_path: Path,
) -> None:
    donor, anchors, mapping, codebook = _tiny_sparse_codebook()
    source = tmp_path / "base-codebook"
    source.mkdir()
    receipt = runner.authenticate_base_codebook_override(
        codebook,
        source_directory=source,
        expected_artifact_sha256=codebook.artifact_sha256,
        donor_table=donor,
        target_table=anchors,
        target_to_donor=mapping,
        maximum_candidates=4,
        active_k=2,
        candidate_ridge=1e-5,
        refit_ridge=1e-5,
    )
    assert receipt["candidate_discovery_calls"] == 0
    assert receipt["coefficient_solve_calls"] == 0
    assert receipt["tensor_content"]["continuous_coefficient_count"] == int(
        codebook.semantic_code_coefficients.numel()
    )
    with pytest.raises(
        runner.Lfm2SparseSemanticRunnerError,
        match="differs from requested",
    ):
        runner.authenticate_base_codebook_override(
            codebook,
            source_directory=source,
            expected_artifact_sha256=codebook.artifact_sha256,
            donor_table=donor,
            target_table=anchors,
            target_to_donor=mapping,
            maximum_candidates=4,
            active_k=3,
            candidate_ridge=1e-5,
            refit_ridge=1e-5,
        )


def test_norm_reclosure_phase_key_binds_controls_and_base_provenance() -> None:
    _, _, _, codebook = _tiny_sparse_codebook()
    content = runner.sparse_codebook_tensor_content(codebook)
    kwargs = {
        "base_tensor_content": content,
        "base_codebook_artifact_sha256": codebook.artifact_sha256,
        "donor_table_sha256": _sha("donor"),
        "target_table_sha256": _sha("target"),
        "target_to_donor_sha256": _sha("mapping"),
        "gamma": 1.0,
        "max_scale": 4.0,
        "minimum_cosine_to_amplify": 0.25,
        "donor_rows_per_chunk": 2,
        "compute_device": "cpu",
        "compute_dtype": "float64",
        "algorithm_sha256": _sha("algorithm"),
    }
    key = runner.build_norm_reclosure_phase_key(**kwargs)
    assert key.phase_name == runner.NORM_RECLOSURE_PHASE
    assert codebook.artifact_sha256 in json.dumps(key.as_dict(), sort_keys=True)
    assert (
        runner.build_norm_reclosure_phase_key(
            **{**kwargs, "gamma": 0.5}
        ).phase_key_sha256
        != key.phase_key_sha256
    )
    assert (
        runner.build_norm_reclosure_phase_key(
            **{**kwargs, "base_codebook_artifact_sha256": _sha("other-base")}
        ).phase_key_sha256
        != key.phase_key_sha256
    )
    assert (
        runner.build_norm_reclosure_phase_key(
            **{**kwargs, "compute_dtype": "float32"}
        ).phase_key_sha256
        != key.phase_key_sha256
    )


def test_norm_reclosure_is_separately_cached_and_adds_no_parameters(
    tmp_path: Path,
) -> None:
    donor, anchors, mapping, base = _tiny_sparse_codebook()
    content = runner.sparse_codebook_tensor_content(base)
    key = runner.build_norm_reclosure_phase_key(
        base_tensor_content=content,
        base_codebook_artifact_sha256=base.artifact_sha256,
        donor_table_sha256=runner._tensor_sha256(donor),
        target_table_sha256=runner._tensor_sha256(anchors),
        target_to_donor_sha256=runner._tensor_sha256(mapping),
        gamma=1.0,
        max_scale=4.0,
        minimum_cosine_to_amplify=None,
        donor_rows_per_chunk=2,
        compute_device="cpu",
        compute_dtype="float64",
        algorithm_sha256=_sha("norm-algorithm"),
    )
    cache = DynamicFoldPhaseCache(tmp_path / "cache")
    calls = 0

    def compute(progress):
        nonlocal calls
        calls += 1
        return reclose_lfm2_sparse_semantic_codebook_norms(
            base,
            donor,
            anchors,
            gamma=1.0,
            max_scale=4.0,
            donor_rows_per_chunk=2,
            compute_device="cpu",
            compute_dtype=torch.float64,
            progress_callback=progress,
        )

    def execute(event_name: str):
        return runner.cache_or_compute_artifact_phase(
            cache=cache,
            events=FoldPhaseEventLog(
                tmp_path / f"{event_name}.jsonl", run_id=event_name
            ),
            phase_key=key,
            compute_call=compute,
            save_call=save_lfm2_sparse_semantic_codebook,
            load_call=load_lfm2_sparse_semantic_codebook,
            artifact_sha256_call=lambda artifact: artifact.artifact_sha256,
        )

    first = execute("first")
    second = execute("second")
    assert calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.artifact_sha256 == second.artifact_sha256
    reclosure = first.artifact.as_dict()["norm_reclosure"]
    assert reclosure["continuous_coefficient_count_added"] == 0
    assert reclosure["effective_child_parameter_count_added"] == 0
    assert reclosure["protected_direct_rows_preserved_bit_exact"] is True
    rows, rows_sha = runner.authenticated_reclosure_row_diagnostics(reclosure)
    assert len(rows) == 2
    assert rows_sha == reclosure["row_reclosure_diagnostics_sha256"]
    assert int(first.artifact.semantic_code_coefficients.numel()) == int(
        base.semantic_code_coefficients.numel()
    )
    assert torch.equal(
        first.artifact.semantic_code_offsets,
        base.semantic_code_offsets,
    )
    assert torch.equal(
        first.artifact.semantic_code_target_ids,
        base.semantic_code_target_ids,
    )
    assert not torch.equal(
        first.artifact.semantic_code_coefficients,
        base.semantic_code_coefficients,
    )

    promoted_key = runner.build_norm_reclosure_phase_key(
        base_tensor_content=content,
        base_codebook_artifact_sha256=base.artifact_sha256,
        donor_table_sha256=runner._tensor_sha256(donor),
        target_table_sha256=runner._tensor_sha256(anchors),
        target_to_donor_sha256=runner._tensor_sha256(mapping),
        gamma=1.0,
        max_scale=4.0,
        minimum_cosine_to_amplify=None,
        donor_rows_per_chunk=2,
        compute_device="cpu",
        compute_dtype="float64",
        algorithm_sha256=runner.sparse_semantic_phase_math_abi_receipts()[
            "coefficient_norm_reclosure"
        ]["math_abi_sha256"],
    )

    def must_not_recompute(_progress):  # pragma: no cover - asserted unreachable
        raise AssertionError("compatible norm reclosure was recomputed")

    promoted = runner.cache_or_compute_artifact_phase(
        cache=cache,
        events=FoldPhaseEventLog(tmp_path / "promoted.jsonl", run_id="promoted"),
        phase_key=promoted_key,
        compute_call=must_not_recompute,
        save_call=save_lfm2_sparse_semantic_codebook,
        load_call=load_lfm2_sparse_semantic_codebook,
        artifact_sha256_call=lambda artifact: artifact.artifact_sha256,
        compatible_phase_keys=(key,),
    )
    assert promoted.cache_hit is True
    assert promoted.phase_key.phase_key_sha256 == promoted_key.phase_key_sha256
    assert promoted.artifact_sha256 == first.artifact_sha256
    promoted_entry = cache.lookup(promoted_key)
    assert promoted_entry is not None
    assert (
        promoted_entry.result["compatible_cache_source_phase_key_sha256"]
        == key.phase_key_sha256
    )


def test_unindexed_cuda_compile_placement_is_bound_to_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    device, dtype, receipt = runner.resolve_compile_placement(
        role="coefficient-solve",
        requested_device="cuda",
        requested_dtype="auto",
    )
    assert device == torch.device("cuda:2")
    assert dtype == torch.float32
    assert receipt["resolved_device"] == "cuda:2"


class _FakeSafeSlice:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def get_shape(self):
        return list(self.tensor.shape)

    def __getitem__(self, value):
        return self.tensor[value]


class _FakeSafeOpen:
    def __init__(self, tensor: torch.Tensor, calls: list[tuple[str, str]]) -> None:
        self.tensor = tensor
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return (runner.PARENT_TIED_TENSOR_KEY, "unrelated.weight")

    def get_slice(self, name: str):
        self.calls.append(("get_slice", name))
        return _FakeSafeSlice(self.tensor)

    def get_tensor(self, name: str):  # pragma: no cover - must never be called
        self.calls.append(("get_tensor", name))
        raise AssertionError("selective loader used get_tensor")


def test_parent_loader_reads_only_defined_tied_rows_without_parent_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "parent"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "lfm2",
                "tie_word_embeddings": True,
                "hidden_size": 3,
                "vocab_size": 7,
            }
        ),
        encoding="utf-8",
    )
    (model / "model.safetensors").write_bytes(b"fixture")
    source = torch.arange(21, dtype=torch.bfloat16).reshape(7, 3)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        runner,
        "safe_open",
        lambda *_args, **_kwargs: _FakeSafeOpen(source, calls),
    )
    loaded = runner.load_parent_tied_rows_from_local_safetensors(
        model,
        expected_parent_identity_sha256=_sha("parent"),
        donor_row_count=5,
        hidden_size=3,
    )
    assert calls == [("get_slice", runner.PARENT_TIED_TENSOR_KEY)]
    assert loaded.table.shape == (5, 3)
    assert loaded.table.dtype == torch.float32
    assert loaded.receipt["defined_rows_loaded"] == 5
    assert loaded.receipt["undefined_capacity_rows_loaded"] == 0
    assert loaded.receipt["unrelated_parent_tensors_loaded"] == 0
    assert loaded.receipt["parent_model_objects_constructed"] == 0


class _FakeSelectedPayload:
    def __init__(self, tensors):
        self.tensors = tensors
        self.logical_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )


class _FakeTraceSession:
    def __init__(self) -> None:
        self.batch_references = (
            SimpleNamespace(
                index=0,
                ordered_tensor_names=(
                    "prefill.logits",
                    "prefill.output.lm_head_input",
                ),
            ),
        )
        self.released = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def load_selected_tensors(self, _index, names):
        assert names == (
            "prefill.logits",
            "prefill.output.lm_head_input",
        )
        hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 1.0, 4.0]]])
        return _FakeSelectedPayload(
            {
                "prefill.logits": logits,
                "prefill.output.lm_head_input": hidden,
            }
        )

    def release_selected_tensors(self, _payload):
        self.released = True

    def verify(self):
        return self

    def as_dict(self):
        body = {"schema": "fake-selective-session", "batch_count": 1}
        return {**body, "session_sha256": canonical_sha256(body)}


class _FakeCache:
    cache_key = _sha("cache")
    parent_fingerprint = _sha("parent")

    def __init__(self) -> None:
        self.session = _FakeTraceSession()

    def selective_authenticated_session(self):
        return self.session


class _FakeRouting:
    donor_row_count = 3
    target_row_count = 2
    hidden_size = 2

    def authenticated_live_table_session(self, table):
        return contextlib.nullcontext(table)

    def score_target_tokens(self, _session, hidden):
        return hidden

    def compute_all_donor_logits_from_target_scores(
        self, _session, scores, *, maximum_donor_rows_per_chunk
    ):
        assert maximum_donor_rows_per_chunk == 4096
        # C rows: [1,0], [0,1], [0,2].  The second cached hidden chooses the
        # multi-code row 2, proving that this diagnostic does not reduce to S.
        code = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]], device=scores.device)
        return scores @ code.T

    def span_for_donor(self, donor_id: int):
        return ((0,), (1,), (0, 1))[donor_id]


class _FakeAllDirectRouting(_FakeRouting):
    donor_row_count = 2

    def compute_all_donor_logits_from_target_scores(
        self, _session, scores, *, maximum_donor_rows_per_chunk
    ):
        assert maximum_donor_rows_per_chunk == 4096
        return scores

    def span_for_donor(self, donor_id: int):
        return ((0,), (1,))[donor_id]


def test_cached_hidden_diagnostic_separates_direct_and_multi_code_winners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _FakeCache()
    monkeypatch.setattr(
        runner.Lfm2CircuitTraceCache,
        "open",
        lambda *_args, **_kwargs: cache,
    )
    result = runner.evaluate_cached_hidden_top1_recovery(
        trace_cache_directory=tmp_path,
        expected_parent_identity_sha256=_sha("parent"),
        routing=_FakeRouting(),
        live_target_table=torch.eye(2),
        protected_target_to_donor=torch.tensor([0, 1], dtype=torch.int64),
        row_reclosure_diagnostics=(
            {
                "donor_id": 2,
                "semantic_code_edge_count": 2,
                "before_norm_ratio": 1.2,
                "after_norm_ratio": 0.9,
                "applied_scale": 0.75,
                "before_cosine": 1.000000238,
                "after_cosine": 1.0,
                "before_relative_l2_error": 0.4,
                "after_relative_l2_error": 0.3,
                "amplified": False,
                "scale_cap_applied": False,
                "unresolved_low_cosine": False,
            },
        ),
        row_reclosure_max_scale=4.0,
        maximum_hidden_rows_per_score_chunk=1,
    )
    assert result["cached_hidden_row_count"] == 2
    assert result["cached_hidden_top1_match_count"] == 2
    assert result["parent_top1_direct_anchor_count"] == 1
    assert result["top1_match_given_parent_direct_anchor_count"] == 1
    assert result["parent_top1_multi_code_count"] == 1
    assert result["top1_match_given_parent_multi_code_count"] == 1
    assert result["cached_hidden_top5_match_count"] == 2
    assert result["cached_hidden_top10_match_count"] == 2
    assert result["cached_hidden_mean_reciprocal_rank"] == 1.0
    assert result["mean_parent_winner_margin_over_best_other_child_score"] == 1.0
    assert result["positive_parent_winner_margin_count"] == 2
    assert result["zero_parent_winner_margin_count"] == 0
    assert result["negative_parent_winner_margin_count"] == 0
    assert result["reclosure_row_provenance_enabled"] is True
    assert result["reclosure_row_provenance_count"] == 1
    assert result["reclosure_row_provenance_max_scale"] == 4.0
    assert result["predicted_winner_direct_anchor_count"] == 1
    assert result["predicted_winner_multi_code_count"] == 1
    assert result["parent_winner_by_predicted_winner_kind_confusion"] == {
        "parent_direct_predicted_direct": 1,
        "parent_direct_predicted_multi": 0,
        "parent_multi_predicted_direct": 0,
        "parent_multi_predicted_multi": 1,
    }
    assert len(result["row_rank_diagnostics"]) == 2
    assert result["row_rank_diagnostics"][0]["best_other_child_score"] == 0.0
    assert (
        result["row_rank_diagnostics"][0][
            "parent_winner_margin_over_best_other_child_score"
        ]
        == 1.0
    )
    assert (
        result["row_rank_diagnostics"][1][
            "parent_multi_margin_over_best_protected_direct"
        ]
        == 1.0
    )
    assert (
        result["row_rank_diagnostics"][0]["parent_winner_reclosure_provenance"] is None
    )
    assert (
        result["row_rank_diagnostics"][1]["parent_winner_reclosure_provenance"][
            "applied_scale"
        ]
        == 0.75
    )
    assert (
        result["row_rank_diagnostics"][1]["predicted_winner_reclosure_provenance"][
            "donor_id"
        ]
        == 2
    )
    assert (
        result["reclosure_transition_aggregates"]["by_scale_cap_applied"]["not-capped"][
            "top1_match_count"
        ]
        == 1
    )
    assert (
        result["reclosure_transition_aggregates"]["by_before_cosine_band"][
            "0.75-to-1.0"
        ]["row_count"]
        == 1
    )
    assert result["S_emission_span_length_used_for_classification"] is False
    assert result["metrics_are_advisory_not_an_intelligence_gate"] is True
    assert cache.session.released is True


def test_cached_hidden_diagnostic_accepts_authenticated_all_direct_reclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _FakeCache()
    monkeypatch.setattr(
        runner.Lfm2CircuitTraceCache,
        "open",
        lambda *_args, **_kwargs: cache,
    )
    result = runner.evaluate_cached_hidden_top1_recovery(
        trace_cache_directory=tmp_path,
        expected_parent_identity_sha256=_sha("parent"),
        routing=_FakeAllDirectRouting(),
        live_target_table=torch.eye(2),
        protected_target_to_donor=torch.tensor([0, 1], dtype=torch.int64),
        row_reclosure_diagnostics=(),
        row_reclosure_max_scale=3.0,
        maximum_hidden_rows_per_score_chunk=1,
    )
    assert result["cached_hidden_top1_match_count"] == 2
    assert result["reclosure_row_provenance_enabled"] is True
    assert result["reclosure_row_provenance_count"] == 0
    assert result["reclosure_transition_aggregates"] == {
        "by_before_cosine_band": {},
        "by_scale_cap_applied": {},
        "by_unresolved_low_cosine": {},
    }


def test_cli_defaults_to_k8_and_source_forbids_parent_execution() -> None:
    parser = runner._parser()
    actions = {action.dest: action for action in parser._actions}
    assert actions["k"].default == 8
    assert actions["base_codebook_directory"].default is None
    assert actions["expected_base_codebook_artifact_sha256"].default is None
    assert actions["norm_reclosure_gamma"].default is None
    assert actions["norm_reclosure_max_scale"].default is None
    assert actions["support_selection"].default == "top-cosine"
    assert tuple(actions["support_selection"].choices) == (
        "top-cosine",
        "residual-omp",
    )
    assert actions["omp_orthogonality_tolerance"].default is None
    assert actions["selection_dtype"].default == "float32"
    assert actions["solve_device"].default == "cpu"
    assert actions["solve_dtype"].default == "float64"
    source = Path(runner.__file__).read_text(encoding="utf-8")
    forbidden = (
        "AutoModel",
        "from_pretrained",
        ".generate(",
        "optimizer.step",
        ".backward(",
    )
    assert all(value not in source for value in forbidden)
