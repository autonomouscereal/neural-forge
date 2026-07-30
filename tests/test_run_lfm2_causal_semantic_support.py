from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from experiments import run_lfm2_causal_semantic_support as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.hf_lfm2_causal_semantic_support import (
    seal_lfm2_causal_support_feature_pack,
)
from neural_forge.hf_lfm2_functional_output_refit import (
    refit_lfm2_functional_output_coefficients,
)
from neural_forge.hf_lfm2_sparse_semantic_codebook import (
    compile_lfm2_sparse_semantic_codebook,
    save_lfm2_sparse_semantic_codebook,
)
from neural_forge import hf_lfm2_teacher_forced_trajectory_audit as trajectory


_D = "a" * 64


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="ascii",
    )


def _readout(
    *, winner: int, margin: float, matches: bool, competitor: int = 4
) -> dict[str, object]:
    return {
        "top1_token_id": winner if matches else competitor,
        "parent_winner_token_id": winner,
        "parent_winner_rank": 1 if matches else 2,
        "parent_winner_score": margin,
        "best_competing_token_id": competitor,
        "best_competing_score": 0.0,
        "parent_winner_margin": margin,
        "top1_matches_parent": matches,
    }


def _token(
    *,
    prefix: int,
    hidden_relative_l2: float,
    exact_margin: float,
    exact_matches: bool,
    folded_margin: float,
    folded_matches: bool,
) -> dict[str, object]:
    return {
        "prefix_index": prefix,
        "consumed_forced_token_count": prefix,
        "next_forced_token_id": 5 if prefix == 0 else None,
        "parent_winner_token_id": 3,
        "hidden_vs_cached_parent": {
            "cosine": 1.0,
            "cosine_distance": 0.0,
            "l2": hidden_relative_l2,
            "relative_l2": hidden_relative_l2,
            "candidate_norm": 1.0,
            "parent_norm": 1.0,
        },
        "hidden_vs_same_run_exact_exact": {
            "cosine": max(0.0, 1.0 - hidden_relative_l2),
            "cosine_distance": min(1.0, hidden_relative_l2),
            "l2": hidden_relative_l2,
            "relative_l2": hidden_relative_l2,
            "candidate_norm": 1.0,
            "parent_norm": 1.0,
        },
        "readouts": {
            "exact_parent_table": _readout(
                winner=3, margin=exact_margin, matches=exact_matches
            ),
            "folded_c_out": _readout(
                winner=3, margin=folded_margin, matches=folded_matches
            ),
        },
    }


def _audit() -> dict[str, object]:
    exact = [
        _token(
            prefix=0,
            hidden_relative_l2=0.0,
            exact_margin=2.0,
            exact_matches=True,
            folded_margin=-1.0,
            folded_matches=False,
        ),
        _token(
            prefix=1,
            hidden_relative_l2=0.0,
            exact_margin=2.0,
            exact_matches=True,
            folded_margin=-0.5,
            folded_matches=False,
        ),
    ]
    folded_prompt = [
        _token(
            prefix=0,
            hidden_relative_l2=0.8,
            exact_margin=-1.0,
            exact_matches=False,
            folded_margin=-1.0,
            folded_matches=False,
        ),
        _token(
            prefix=1,
            hidden_relative_l2=0.7,
            exact_margin=-0.5,
            exact_matches=False,
            folded_margin=-0.5,
            folded_matches=False,
        ),
    ]
    exact_folded = [
        _token(
            prefix=0,
            hidden_relative_l2=0.0,
            exact_margin=2.0,
            exact_matches=True,
            folded_margin=-1.0,
            folded_matches=False,
        ),
        _token(
            prefix=1,
            hidden_relative_l2=0.0,
            exact_margin=2.0,
            exact_matches=True,
            folded_margin=-0.5,
            folded_matches=False,
        ),
    ]
    prompt_body = {
        "schema": trajectory.LFM2_TEACHER_FORCED_TRAJECTORY_PROMPT_RESULT_SCHEMA,
        "batch_index": 0,
        "prompt_id": "fit-a",
        "prompt_token_count": 2,
        "forced_token_count": 1,
        "state_count": 2,
        "parent_reference_tokens": [
            {"prefix_index": 0, "next_forced_token_id": 5},
            {"prefix_index": 1, "next_forced_token_id": None},
        ],
        "lanes": {
            "exact_exact": {"summary": {}, "tokens": exact},
            "folded_exact": {"summary": {}, "tokens": folded_prompt},
            "exact_folded": {"summary": {}, "tokens": exact_folded},
            "folded_folded": {"summary": {}, "tokens": folded_prompt},
        },
        "same_run_hidden_contrast_contract": dict(
            trajectory.SAME_RUN_HIDDEN_CONTRAST_CONTRACT
        ),
        "target_token_spans_used_as_body_steps": False,
        "operation_audit": dict(trajectory._OPERATION_AUDIT),
    }
    prompt = {**prompt_body, "result_sha256": canonical_sha256(prompt_body)}
    audit_body = {
        "schema": trajectory.LFM2_TEACHER_FORCED_TRAJECTORY_AUDIT_SCHEMA,
        "request_cache_key": "e" * 64,
        "prompt_count": 1,
        "prompt_results": [prompt],
        "operation_audit": dict(trajectory._OPERATION_AUDIT),
    }
    return {**audit_body, "audit_sha256": canonical_sha256(audit_body)}


def _tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    donor = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.7],
            [-0.2, 1.0],
            [0.8, 0.8],
        ],
        dtype=torch.float32,
    )
    return donor, target, torch.tensor([0, 1, 2], dtype=torch.int64)


def _fixture(tmp_path: Path) -> argparse.Namespace:
    donor, target, mapping = _tables()
    codebook = compile_lfm2_sparse_semantic_codebook(
        donor,
        target,
        mapping,
        maximum_candidates=3,
        active_k=2,
        candidate_ridge=1e-8,
        refit_ridge=1e-8,
        donor_rows_per_chunk=3,
        selection_device="cpu",
        selection_dtype=torch.float64,
        solve_device="cpu",
        solve_dtype=torch.float64,
    )
    codebook_directory = tmp_path / "codebook"
    save_lfm2_sparse_semantic_codebook(codebook, codebook_directory)

    output_result = refit_lfm2_functional_output_coefficients(
        codebook=codebook,
        donor_table=donor,
        target_table=target,
        functional_metric=torch.diag(torch.tensor([2.0, 0.5])),
        functional_weight=0.1,
        solve_ridge=1e-8,
        solve_device="cpu",
        solve_dtype=torch.float64,
        rows_per_chunk=3,
    )
    output_path = tmp_path / "output.safetensors"
    save_file(
        {
            "output_semantic_code_coefficients": (
                output_result.output_semantic_code_coefficients
            )
        },
        str(output_path),
    )
    output_receipt_path = tmp_path / "output-receipt.json"
    _write_json(output_receipt_path, output_result.as_dict())

    audit = _audit()
    audit_path = tmp_path / "audit.json"
    _write_json(audit_path, audit)
    completion_body = {
        "schema": runner.TRAJECTORY_COMPLETION_SCHEMA,
        "request_cache_key": audit["request_cache_key"],
        "audit_sha256": audit["audit_sha256"],
        "audit_file_sha256": _file_sha256(audit_path),
        "batch_checkpoint_sha256s": ["f" * 64],
        "original_core_audit_sha256_before_runtime_receipts": "0" * 64,
        "atomic_batch_checkpoints": True,
        "reusable_identical_math": True,
    }
    completion = {
        **completion_body,
        "completion_sha256": canonical_sha256(completion_body),
    }
    completion_path = tmp_path / "completion.json"
    _write_json(completion_path, completion)
    feature_pack = seal_lfm2_causal_support_feature_pack(
        input_feature_transform=torch.eye(2, dtype=torch.float64),
        output_feature_transform=torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        trajectory_audit_sha256=str(audit["audit_sha256"]),
        fit_prompt_ids=["fit-a"],
        manual_review_prompt_ids=["manual-a"],
        topology_sha256=_D,
        numeric_contract_sha256="b" * 64,
        source_artifact_sha256s={"a12-fit-hidden": "c" * 64},
    )
    feature_path = tmp_path / "features.safetensors"
    save_file(
        {
            "input_feature_transform": feature_pack.input_feature_transform,
            "output_feature_transform": feature_pack.output_feature_transform,
        },
        str(feature_path),
    )
    feature_receipt_path = tmp_path / "feature-receipt.json"
    _write_json(feature_receipt_path, feature_pack.as_dict())
    manifest = runner.build_fit_prompt_manifest(
        trajectory_audit_sha256=str(audit["audit_sha256"]),
        feature_pack_artifact_sha256=feature_pack.artifact_sha256,
        fit_prompt_ids=["fit-a"],
        manual_review_prompt_ids=["manual-a"],
        prompt_token_ids_by_prompt={"fit-a": [3, 0]},
        source_trace_batch_receipt_sha256s={"fit-a": "d" * 64},
    )
    manifest_path = tmp_path / "fit-manifest.json"
    _write_json(manifest_path, manifest)

    donor_path = tmp_path / "donor.safetensors"
    target_path = tmp_path / "target.safetensors"
    save_file({"donor": donor}, str(donor_path))
    save_file({"target": target}, str(target_path))
    return argparse.Namespace(
        trajectory_audit_json=audit_path,
        trajectory_completion_json=completion_path,
        expected_trajectory_audit_file_sha256=_file_sha256(audit_path),
        expected_trajectory_audit_sha256=audit["audit_sha256"],
        expected_trajectory_completion_sha256=completion["completion_sha256"],
        feature_tensors_safetensors=feature_path,
        feature_receipt_json=feature_receipt_path,
        expected_feature_tensor_file_sha256=_file_sha256(feature_path),
        expected_feature_artifact_sha256=feature_pack.artifact_sha256,
        fit_prompt_manifest_json=manifest_path,
        expected_fit_prompt_manifest_sha256=manifest["manifest_sha256"],
        donor_table_safetensors=donor_path,
        donor_table_key="donor",
        expected_donor_table_file_sha256=_file_sha256(donor_path),
        target_table_safetensors=target_path,
        target_table_key="target",
        expected_target_table_file_sha256=_file_sha256(target_path),
        input_codebook_directory=codebook_directory,
        expected_input_codebook_artifact_sha256=codebook.artifact_sha256,
        output_coefficients_safetensors=output_path,
        output_coefficients_key="output_semantic_code_coefficients",
        expected_output_coefficients_file_sha256=_file_sha256(output_path),
        output_refit_receipt_json=output_receipt_path,
        expected_output_refit_cache_key=output_result.cache_key,
        expected_output_refit_result_sha256=output_result.result_sha256,
        physical_child_parameter_count=100,
        maximum_combined_coefficient_count=18,
        input_support_width=1,
        output_support_width=1,
        maximum_input_donor_repairs=1,
        maximum_output_donor_repairs=1,
        position_bin_count=2,
        context_hash_bucket_count=16,
        context_weight=0.25,
        solve_ridge=1e-8,
        solve_dtype="float64",
        output_directory=tmp_path / "output",
    )


def test_offline_runner_publishes_and_reuses_authenticated_result(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    first = runner.run(args)
    second = runner.run(args)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["cache_key"] == second["cache_key"]
    assert first["result_sha256"] == second["result_sha256"]
    assert first["continuous_child_state_count"] == 16
    assert first["effective_child_parameter_count"] == 116
    assert first["input_reselected_donor_count"] == 1
    assert first["output_reselected_donor_count"] == 1
    assert first["input_output_support_searches_independent"]
    assert first["parent_forward_calls"] == 0
    assert first["parent_trace_recaptures"] == 0
    assert first["training_steps"] == 0
    cache = Path(first["output_directory"])
    assert {path.name for path in cache.iterdir()} == {
        "complete.json",
        "receipt.json",
        "routes.safetensors",
    }


def test_cached_tensor_tampering_fails_closed(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    result = runner.run(args)
    tensor_path = Path(result["output_directory"]) / "routes.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
    with pytest.raises(runner.Lfm2CausalSupportRunnerError, match="publication"):
        runner.run(args)


def test_manifest_rejects_fit_review_leakage() -> None:
    with pytest.raises(runner.Lfm2CausalSupportRunnerError, match="disjoint"):
        runner.build_fit_prompt_manifest(
            trajectory_audit_sha256=_D,
            feature_pack_artifact_sha256="b" * 64,
            fit_prompt_ids=["same"],
            manual_review_prompt_ids=["same"],
            prompt_token_ids_by_prompt={"same": [1]},
            source_trace_batch_receipt_sha256s={"same": "c" * 64},
        )


def test_table_file_pin_is_mandatory(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.expected_donor_table_file_sha256 = "f" * 64
    with pytest.raises(runner.Lfm2CausalSupportRunnerError, match="source file"):
        runner.run(args)


def test_tracked_live_a12_completion_authenticates() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "reports" / "lfm2-teacher-forced-trajectory-a12-20260730"
    audit, completion = runner.authenticate_completed_trajectory_audit(
        audit_path=directory / "audit.json",
        completion_path=directory / "complete.json",
        expected_audit_file_sha256=(
            "95972ac8e1155e2baf8d6a70fb0b6c4d22657f6fcec162a78b3f525224c3ede2"
        ),
        expected_audit_sha256=(
            "36b83e86bfc03e81e36949c3ee71e720ffbef32cc4ba8f860f8b978416020c3e"
        ),
        expected_completion_sha256=(
            "0851b60f2a4be2c056118e7fe8aaf2d77ac42cff48029831a257735f691279ba"
        ),
    )
    assert audit["prompt_count"] == 16
    assert audit["request_cache_key"] == (
        "9d803837179575e74c3f381deaeff19b210317594ddac71d732a26c9987acd82"
    )
    assert completion["reusable_identical_math"] is True
    assert audit["operation_audit"]["parent_forward_calls"] == 0
    assert audit["operation_audit"]["parent_trace_recapture_calls"] == 0
