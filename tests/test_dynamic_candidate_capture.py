from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from neural_forge.dynamic_candidate_capture import (
    CANDIDATE_DIRECTORY_NAME,
    QUARANTINE_CAPTURE_SCHEMA,
    QUARANTINE_RECEIPT_FILENAME,
    DynamicCandidateCaptureError,
    DynamicCandidateCaptureIntegrityError,
    DynamicCandidateCapturePublicationError,
    QuarantineCaptureRequest,
    QuarantineEvaluation,
    QuarantineZeroOperationAudit,
    capture_quarantined_candidate,
    fingerprint_quarantine_callable,
    load_quarantined_candidate,
)
from neural_forge.dynamic_circuit_ir import canonical_json_bytes


class _TinyCandidate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0, -1.0],
                        [0.5, -0.25, 3.0],
                    ],
                    dtype=torch.float32,
                )
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _save_candidate(model: _TinyCandidate, destination: Path) -> None:
    torch.save(model.state_dict(), destination / "weights.pt")
    (destination / "config.json").write_bytes(
        canonical_json_bytes(
            {
                "architecture": "_TinyCandidate",
                "input_width": 3,
                "output_width": 2,
            }
        )
        + b"\n"
    )


def _load_candidate(source: Path) -> _TinyCandidate:
    if not source.is_dir():
        raise FileNotFoundError(source)
    model = _TinyCandidate()
    state = torch.load(source / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def _evaluation_input() -> torch.Tensor:
    return torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)


def _evaluate_candidate(model: _TinyCandidate) -> QuarantineEvaluation:
    value = model(_evaluation_input())
    output = value.detach().cpu().tolist()
    return QuarantineEvaluation(
        output={"rows": output},
        evaluation={
            "finite": bool(torch.isfinite(value).all()),
            "maximum": float(value.max().item()),
            "minimum": float(value.min().item()),
        },
    )


def _request(
    *,
    saver: Any = _save_candidate,
    loader: Any = _load_candidate,
    evaluator: Any = _evaluate_candidate,
    expected_model_state_sha256: str | None = None,
    expected_output_sha256: str | None = None,
    candidate_fingerprint_sha256: str | None = None,
) -> QuarantineCaptureRequest:
    return QuarantineCaptureRequest(
        plan_sha256=_sha("plan"),
        parent_sha256=_sha("parents"),
        cache_sha256=_sha("cache"),
        bridge_sha256=_sha("bridge"),
        corpus_sha256=_sha("corpus"),
        compiler_sha256=_sha("authenticated compiler run"),
        source_sha256=_sha("compiler source"),
        saver_sha256=fingerprint_quarantine_callable(saver, role="saver"),
        loader_sha256=fingerprint_quarantine_callable(loader, role="loader"),
        evaluator_sha256=fingerprint_quarantine_callable(
            evaluator,
            role="evaluator",
        ),
        expected_model_state_sha256=expected_model_state_sha256,
        expected_output_sha256=expected_output_sha256,
        pre_reviewed_candidate_fingerprint_sha256=(
            candidate_fingerprint_sha256
        ),
    )


def _capture(
    destination: Path,
    *,
    model: _TinyCandidate | None = None,
    saver: Any = _save_candidate,
    loader: Any = _load_candidate,
    evaluator: Any = _evaluate_candidate,
    request: QuarantineCaptureRequest | None = None,
    forbidden_parent_models: tuple[Any, ...] = (),
):
    selected_request = (
        _request(saver=saver, loader=loader, evaluator=evaluator)
        if request is None
        else request
    )
    selected_model = _TinyCandidate() if model is None else model
    return capture_quarantined_candidate(
        destination,
        candidate=selected_model,
        request=selected_request,
        saver=saver,
        loader=loader,
        evaluator=evaluator,
        forbidden_parent_models=forbidden_parent_models,
    )


def test_capture_is_immutable_nonpromotable_and_freshly_reloadable(
    tmp_path: Path,
) -> None:
    original = _TinyCandidate()
    request = _request()
    captured = _capture(
        tmp_path / "quarantine",
        model=original,
        request=request,
    )

    assert captured.model is not original
    assert captured.promotable is False
    assert captured.publication_authorized is False
    assert captured.requires_reviewed_rerun is True
    assert captured.receipt.as_dict()["schema"] == QUARANTINE_CAPTURE_SCHEMA
    assert set(path.name for path in captured.path.iterdir()) == {
        CANDIDATE_DIRECTORY_NAME,
        QUARANTINE_RECEIPT_FILENAME,
    }

    receipt = captured.receipt.as_dict()
    assert receipt["status"] == "quarantined-unreviewed-candidate"
    assert receipt["published_compilation"] is False
    assert receipt["accepted_compilation"] is False
    assert receipt["behavior_capability_proven"] is False
    assert receipt["promotable"] is False
    assert receipt["publication_authorized"] is False
    assert receipt["requires_reviewed_rerun"] is True
    assert receipt["pre_reviewed_candidate_fingerprint_supplied"] is False
    assert receipt["bindings"] == {
        "plan_sha256": request.plan_sha256,
        "parent_sha256": request.parent_sha256,
        "cache_sha256": request.cache_sha256,
        "bridge_sha256": request.bridge_sha256,
        "corpus_sha256": request.corpus_sha256,
        "compiler_sha256": request.compiler_sha256,
        "compiler_binding_semantics": (
            "authenticated-compiler-run-receipt-bound-to-source"
        ),
        "source_sha256": request.source_sha256,
        "model_state_sha256": receipt["model_state"]["in_memory_sha256"],
        "output_sha256": receipt["evaluations"]["in_memory_output_sha256"],
    }
    assert (
        receipt["evaluations"]["in_memory_output_sha256"]
        == receipt["evaluations"]["fresh_reload_output_sha256"]
    )
    assert (
        receipt["evaluations"]["in_memory_evaluation_sha256"]
        == receipt["evaluations"]["fresh_reload_evaluation_sha256"]
    )
    assert (
        receipt["standalone_reload_receipt"][
            "loader_returned_disjoint_model"
        ]
        is True
    )
    assert receipt["audit"]["prior_fold"]["clean"] is True
    assert receipt["audit"]["capture_observed"]["clean"] is True
    assert receipt["candidate_artifact"]["inventory"]["file_count"] == 2
    assert {
        row["path"]
        for row in receipt["candidate_artifact"]["inventory"]["files"]
    } == {"config.json", "weights.pt"}

    raw_receipt = (
        captured.path / QUARANTINE_RECEIPT_FILENAME
    ).read_bytes()
    assert raw_receipt == canonical_json_bytes(receipt) + b"\n"

    loaded = load_quarantined_candidate(
        captured.path,
        request=request,
        expected_receipt_sha256=captured.receipt.receipt_sha256,
        loader=_load_candidate,
        evaluator=_evaluate_candidate,
    )
    assert loaded.model is not captured.model
    assert loaded.receipt.receipt_sha256 == captured.receipt.receipt_sha256
    assert (
        loaded.verification_receipt["fresh_final_destination_reload_performed"]
        is True
    )
    assert loaded.verification_receipt["promotable"] is False


@pytest.mark.parametrize("location", ["artifact", "root", "receipt"])
def test_tamper_and_foreign_entries_fail_before_reload(
    tmp_path: Path,
    location: str,
) -> None:
    request = _request()
    captured = _capture(tmp_path / "quarantine", request=request)

    if location == "artifact":
        weights = captured.artifact_path / "weights.pt"
        weights.write_bytes(weights.read_bytes() + b"tamper")
    elif location == "root":
        (captured.path / "foreign.txt").write_text("foreign", encoding="utf-8")
    else:
        receipt_path = captured.path / QUARANTINE_RECEIPT_FILENAME
        payload = json.loads(receipt_path.read_text(encoding="ascii"))
        payload["status"] = "accepted"
        receipt_path.write_bytes(canonical_json_bytes(payload) + b"\n")

    with pytest.raises(DynamicCandidateCaptureIntegrityError):
        load_quarantined_candidate(
            captured.path,
            request=request,
            expected_receipt_sha256=captured.receipt.receipt_sha256,
            loader=_load_candidate,
            evaluator=_evaluate_candidate,
        )


def test_identical_byte_path_replacement_is_rejected(tmp_path: Path) -> None:
    request = _request()
    captured = _capture(tmp_path / "quarantine", request=request)
    weights = captured.artifact_path / "weights.pt"
    replacement = captured.artifact_path / "replacement.tmp"
    replacement.write_bytes(weights.read_bytes())
    os.replace(replacement, weights)

    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="replaced|inventory",
    ):
        load_quarantined_candidate(
            captured.path,
            request=request,
            expected_receipt_sha256=captured.receipt.receipt_sha256,
            loader=_load_candidate,
            evaluator=_evaluate_candidate,
        )


def test_capture_is_atomic_no_clobber_and_cleans_failed_staging(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "quarantine"
    request = _request()
    first = _capture(destination, request=request)
    receipt_before = (
        destination / QUARANTINE_RECEIPT_FILENAME
    ).read_bytes()

    with pytest.raises(
        DynamicCandidateCapturePublicationError,
        match="already exists",
    ):
        _capture(destination, request=request)
    assert (
        destination / QUARANTINE_RECEIPT_FILENAME
    ).read_bytes() == receipt_before
    assert first.receipt.receipt_sha256 in receipt_before.decode("ascii")

    failed_destination = tmp_path / "failed"

    def failing_saver(model: _TinyCandidate, path: Path) -> None:
        torch.save(model.state_dict(), path / "partial.pt")
        raise RuntimeError("injected crash")

    failing_request = _request(saver=failing_saver)
    with pytest.raises(
        DynamicCandidateCapturePublicationError,
        match="saver failed",
    ):
        _capture(
            failed_destination,
            saver=failing_saver,
            request=failing_request,
        )
    assert not failed_destination.exists()
    assert not list(tmp_path.glob(".failed.*.quarantine-staging"))


def test_loader_must_use_path_and_reload_disjoint_state(tmp_path: Path) -> None:
    def ignoring_loader(_source: Path) -> _TinyCandidate:
        return _TinyCandidate()

    request = _request(loader=ignoring_loader)
    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="absent path",
    ):
        _capture(
            tmp_path / "ignored",
            loader=ignoring_loader,
            request=request,
        )

    original = _TinyCandidate()

    def aliasing_loader(source: Path) -> _TinyCandidate:
        if not source.is_dir():
            raise FileNotFoundError(source)
        return original

    alias_request = _request(loader=aliasing_loader)
    with pytest.raises(
        DynamicCandidateCapturePublicationError,
        match="in-memory candidate object|shares modules",
    ):
        _capture(
            tmp_path / "alias",
            model=original,
            loader=aliasing_loader,
            request=alias_request,
        )


def test_forbidden_sidecars_and_nonzero_operation_audit_fail_closed(
    tmp_path: Path,
) -> None:
    def sidecar_saver(model: _TinyCandidate, path: Path) -> None:
        torch.save(model.state_dict(), path / "weights.pt")
        (path / "parent_cache.pt").write_bytes(b"forbidden")

    request = _request(saver=sidecar_saver)
    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="sidecar",
    ):
        _capture(
            tmp_path / "sidecar",
            saver=sidecar_saver,
            request=request,
        )
    assert not (tmp_path / "sidecar").exists()

    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="exactly zero",
    ):
        QuarantineZeroOperationAudit(optimizer_steps=1)


def test_training_attempt_and_forbidden_parent_call_are_observed(
    tmp_path: Path,
) -> None:
    parent = _TinyCandidate()

    def parent_using_evaluator(model: _TinyCandidate) -> QuarantineEvaluation:
        child_value = model(_evaluation_input())
        parent_value = parent(_evaluation_input())
        return QuarantineEvaluation(
            output=(child_value + parent_value).detach().tolist(),
            evaluation={"executed": True},
        )

    parent_request = _request(evaluator=parent_using_evaluator)
    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="parent_forward_calls|exactly zero",
    ):
        _capture(
            tmp_path / "parent-call",
            evaluator=parent_using_evaluator,
            request=parent_request,
            forbidden_parent_models=(parent,),
        )

    def backward_evaluator(model: _TinyCandidate) -> QuarantineEvaluation:
        value = model(_evaluation_input())
        value.sum().backward()
        return QuarantineEvaluation(
            output=value.detach().tolist(),
            evaluation={"executed": True},
        )

    backward_request = _request(evaluator=backward_evaluator)
    with pytest.raises(DynamicCandidateCaptureError):
        _capture(
            tmp_path / "backward",
            evaluator=backward_evaluator,
            request=backward_request,
        )


def test_callable_substitution_global_monkeypatch_and_closure_drift_reject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    captured = _capture(tmp_path / "quarantine", request=request)

    def alternate_loader(source: Path) -> _TinyCandidate:
        if not source.exists():
            raise FileNotFoundError(source)
        result = _load_candidate(source)
        return result

    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="Loader differs",
    ):
        load_quarantined_candidate(
            captured.path,
            request=request,
            expected_receipt_sha256=captured.receipt.receipt_sha256,
            loader=alternate_loader,
            evaluator=_evaluate_candidate,
        )

    def changed_input() -> torch.Tensor:
        return torch.tensor([[9.0, 9.0, 9.0]], dtype=torch.float32)

    monkeypatch.setitem(
        _evaluate_candidate.__globals__,
        "_evaluation_input",
        changed_input,
    )
    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="Evaluator differs",
    ):
        load_quarantined_candidate(
            captured.path,
            request=request,
            expected_receipt_sha256=captured.receipt.receipt_sha256,
            loader=_load_candidate,
            evaluator=_evaluate_candidate,
        )

    def evaluator_factory(scale: float):
        def evaluator(model: _TinyCandidate) -> QuarantineEvaluation:
            value = model(
                torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
            )
            return QuarantineEvaluation(
                output=(value * scale).detach().tolist(),
                evaluation={"scale": scale},
            )

        return evaluator

    first_evaluator = evaluator_factory(1.0)
    second_evaluator = evaluator_factory(2.0)
    closure_request = _request(evaluator=first_evaluator)
    closure_capture = _capture(
        tmp_path / "closure",
        evaluator=first_evaluator,
        request=closure_request,
    )
    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="Evaluator differs",
    ):
        load_quarantined_candidate(
            closure_capture.path,
            request=closure_request,
            expected_receipt_sha256=(
                closure_capture.receipt.receipt_sha256
            ),
            loader=_load_candidate,
            evaluator=second_evaluator,
        )


def test_request_binds_actual_saver_and_optional_expectations(
    tmp_path: Path,
) -> None:
    def alternate_saver(model: _TinyCandidate, destination: Path) -> None:
        torch.save(dict(model.state_dict()), destination / "weights.pt")
        (destination / "config.json").write_text(
            '{"architecture":"_TinyCandidate"}\n',
            encoding="ascii",
        )

    request = _request()
    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="saver differs",
    ):
        _capture(
            tmp_path / "substituted-saver",
            saver=alternate_saver,
            request=request,
        )

    wrong_state_request = _request(
        expected_model_state_sha256=_sha("wrong state")
    )
    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="expected model state",
    ):
        _capture(
            tmp_path / "wrong-state",
            request=wrong_state_request,
        )

    wrong_output_request = _request(
        expected_output_sha256=_sha("wrong output")
    )
    with pytest.raises(
        DynamicCandidateCaptureIntegrityError,
        match="expected candidate output",
    ):
        _capture(
            tmp_path / "wrong-output",
            request=wrong_output_request,
        )
