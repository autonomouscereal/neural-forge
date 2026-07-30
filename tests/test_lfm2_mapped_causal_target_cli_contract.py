from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from neural_forge.dynamic_circuit_ir import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


producer = _load(
    "mapped_target_producer_cli_test",
    ROOT / "experiments" / "build_lfm2_mapped_causal_target_capsule.py",
)
resume = _load(
    "mapped_target_resume_cli_test",
    ROOT / "experiments" / "resume_lfm2_pre_causal_body.py",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_producer_reference_document_is_accepted_by_resume_cli(tmp_path: Path) -> None:
    raw_reference = {
        "path": str(tmp_path / "capsule"),
        "receipt_sha256": _sha("receipt"),
        "receipt_file_sha256": _sha("receipt-file"),
        "manifest_sha256": _sha("manifest"),
        "tensor_file_sha256": _sha("tensor-file"),
        "evidence_sha256": _sha("evidence"),
        "root_target_logits_sha256": _sha("root-logits"),
    }
    body = {
        "schema": producer.MAPPED_TARGET_REFERENCE_DOCUMENT_SCHEMA,
        "reference": raw_reference,
        "producer_operation_audit": {
            "parent_cache_selective_authenticated_sessions": 1,
            "parent_model_forward_calls": 0,
            "training_steps": 0,
        },
        "exactly_one_selective_cache_session": True,
        "parent_model_execution_performed": False,
        "parent_trace_recapture_performed": False,
        "training_performed": False,
    }
    document = {**body, "reference_sha256": canonical_sha256(body)}
    path = tmp_path / "mapped-reference.json"
    file_sha256 = producer._write_exclusive_canonical_json(path, document)

    loaded = resume._mapped_target_reference_from_document(
        path,
        expected_file_sha256=file_sha256,
        expected_reference_sha256=document["reference_sha256"],
    )

    assert loaded.as_dict() == raw_reference
    assert producer.MAPPED_TARGET_REFERENCE_DOCUMENT_SCHEMA == (
        resume.MAPPED_TARGET_REFERENCE_DOCUMENT_SCHEMA
    )


def test_resume_cli_exposes_explicit_opt_in_phase_cache_and_event_log() -> None:
    actions = {action.dest: action for action in resume._parser()._actions}

    assert actions["phase_cache_root"].default is None
    assert actions["phase_cache_root"].type is Path
    assert actions["phase_event_log"].default is None
    assert actions["phase_event_log"].type is Path
