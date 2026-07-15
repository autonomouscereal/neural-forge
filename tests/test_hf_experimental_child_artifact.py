from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from neural_forge.hf_compiled_child_cache import model_state_fingerprint
from neural_forge.hf_experimental_child_artifact import (
    ARTIFACT_DIRECTORY_NAME,
    ARTIFACT_RECEIPT_FILENAME,
    BRIDGE_REPORT_FILENAME,
    EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA,
    ExperimentalChildArtifactExpectations,
    ExperimentalChildArtifactIntegrityError,
    ExperimentalChildArtifactPublicationError,
    publish_experimental_child_artifact,
    verify_experimental_child_artifact,
)
from neural_forge.hf_progressive_target_bridge import _dense_model_state_sha256


class _TinyModel(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(
            torch.arange(width * 3, dtype=torch.float32).reshape(3, width) / 11.0
        )
        self.register_buffer("scale", torch.linspace(0.25, 1.0, width))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.weight.T

    def save_pretrained(self, destination: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        destination.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), destination / "model.safetensors")
        (destination / "config.json").write_text(
            json.dumps({"width": self.width}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, source: Path) -> _TinyModel:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        model = cls(width=int(config["width"]))
        model.load_state_dict(
            load_file(source / "model.safetensors", device="cpu"), strict=True
        )
        return model


class _TinyTokenizer:
    def save_pretrained(self, destination: Path) -> None:
        (destination / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 128}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "tokenizer.json").write_text(
            json.dumps(
                {"model": {"type": "WordLevel", "vocab": {"<unk>": 0}}},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


class _ParameterlessAdapter(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _native_certificate(model: nn.Module) -> dict[str, Any]:
    fingerprint = model_state_fingerprint(model)
    return {
        "schema": "test-native-dense-certificate-v1",
        "exact_unique_parameters": fingerprint["unique_parameter_count"],
        "logical_parameters": fingerprint["logical_parameter_count"],
        "structure_sha256": fingerprint["parameter_topology_sha256"],
        "state_key_sha256": fingerprint["state_key_sha256"],
        "state_key_count": fingerprint["state_key_count"],
    }


def _zero_audit(_model: nn.Module, _path: Path) -> dict[str, int]:
    return {
        "adapter_or_registry_module_count": 0,
        "optimizer_parameter_count": 0,
        "source_sized_sidecar_parameter_count": 0,
        "parent_runtime_reference_count": 0,
    }


def _expectations(model: nn.Module) -> ExperimentalChildArtifactExpectations:
    fingerprint = model_state_fingerprint(model)
    return ExperimentalChildArtifactExpectations(
        preregistration_sha256=_digest("preregistration"),
        shared_execution_contract_sha256=_digest("shared-execution-contract"),
        config_payload_sha256=_canonical_sha({"width": model.width}),
        state_key_sha256=fingerprint["state_key_sha256"],
        parameter_topology_sha256=fingerprint["parameter_topology_sha256"],
        buffer_topology_sha256=fingerprint["buffer_topology_sha256"],
        exact_unique_parameters=fingerprint["unique_parameter_count"],
        exact_logical_parameters=fingerprint["logical_parameter_count"],
    )


def _bridge_report_bytes(
    model: nn.Module,
    expectations: ExperimentalChildArtifactExpectations,
) -> bytes:
    embedded_body = {
        "schema": "test-experimental-recovery-receipt-v1",
        "status": "experimental-unaccepted",
        "publication_authorized": False,
    }
    embedded_receipt = {
        **embedded_body,
        "receipt_sha256": _canonical_sha(embedded_body),
    }
    body = {
        "schema": EXACT_TARGET_EXPERIMENTAL_RECOVERY_MATERIALIZATION_SCHEMA,
        "status": "experimental-unaccepted",
        "exact_unique_parameters": expectations.exact_unique_parameters,
        "exact_budget_match": True,
        "native_before": _native_certificate(model),
        "native_materialized": _native_certificate(model),
        "experimental_recovery_contract": {
            "schema": "test-experimental-recovery-contract-v1",
            "recovery_ab_preregistration_sha256": (expectations.preregistration_sha256),
            "shared_execution_contract_sha256": (
                expectations.shared_execution_contract_sha256
            ),
            "exact_unique_parameters": expectations.exact_unique_parameters,
            "publication_authorized": False,
            "sealed_data_accessed": False,
        },
        "experimental_recovery_receipt": embedded_receipt,
        "native_child_state_sha256": _digest("blank-native-state"),
        "materialized_native_state_sha256": _dense_model_state_sha256(model),
        "materialized_registered_adapter_parameter_count": 0,
        "materialized_optimizer_parameter_count": 0,
        "materialized_registered_source_sized_sidecar_parameter_count": 0,
        "standalone_structure_materialized": True,
        "serialization_only_context": True,
        "serialized_state_must_match_materialized_state_sha256": True,
        "fresh_reload_and_recovery_required": True,
        "publication_authorized": False,
        "accepted_artifact": False,
        "sealed_data_accessed": False,
        "behavior_capability_proven": False,
    }
    report = {
        **body,
        "materialization_transaction_sha256": _canonical_sha(body),
    }
    return _canonical_bytes(report)


@contextmanager
def _tracked_context(report_bytes: bytes, lifecycle: dict[str, Any]) -> Iterator[bytes]:
    assert lifecycle.get("active") is not True
    lifecycle["active"] = True
    lifecycle["entries"] = int(lifecycle.get("entries", 0)) + 1
    try:
        yield report_bytes
    finally:
        lifecycle["active"] = False
        lifecycle["exits"] = int(lifecycle.get("exits", 0)) + 1


def _publisher_kwargs(
    model: _TinyModel,
    expectations: ExperimentalChildArtifactExpectations,
    lifecycle: dict[str, Any],
    *,
    report_bytes: bytes | None = None,
    model_loader: Any = None,
) -> dict[str, Any]:
    canonical_report = (
        _bridge_report_bytes(model, expectations)
        if report_bytes is None
        else report_bytes
    )

    def load(path: Path) -> _TinyModel:
        assert lifecycle["active"] is False
        lifecycle["loader_calls"] = int(lifecycle.get("loader_calls", 0)) + 1
        if model_loader is not None:
            return model_loader(path)
        return _TinyModel.from_pretrained(path)

    return {
        "model": model,
        "tokenizer": _TinyTokenizer(),
        "materialization_context": _tracked_context(canonical_report, lifecycle),
        "expectations": expectations,
        "model_loader": load,
        "native_certifier": _native_certificate,
        "runtime_auditor": _zero_audit,
    }


def test_atomic_publish_and_independent_verify_reload_after_context_exit(
    tmp_path: Path,
) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / "folded-publication"

    receipt = publish_experimental_child_artifact(
        publication,
        **_publisher_kwargs(model, expectations, lifecycle),
    )

    assert lifecycle == {
        "active": False,
        "entries": 1,
        "exits": 1,
        "loader_calls": 1,
    }
    assert publication.is_dir()
    assert (publication / ARTIFACT_DIRECTORY_NAME).is_dir()
    assert (publication / ARTIFACT_RECEIPT_FILENAME).is_file()
    assert (publication / BRIDGE_REPORT_FILENAME).read_bytes() == (
        _bridge_report_bytes(model, expectations)
    )
    assert not (
        publication / ARTIFACT_DIRECTORY_NAME / ARTIFACT_RECEIPT_FILENAME
    ).exists()
    payload = receipt.as_dict()
    assert payload["status"] == "experimental-unaccepted"
    assert payload["fresh_reload_performed"] is True
    assert payload["artifact_tree_unchanged_by_fresh_reload"] is True
    assert (
        payload["fresh_reload_bridge_state_sha256"]
        == (payload["bridge_report"]["materialized_bridge_state_sha256"])
    )
    assert (
        payload["fresh_reload_state_dict_sha256"]
        == payload["fresh_reload_model_state_fingerprint"]["state_dict_sha256"]
    )
    assert all(payload[name] == 0 for name in _zero_audit(model, publication))

    verify_lifecycle = {"active": False, "loader_calls": 0}

    def verify_loader(path: Path) -> _TinyModel:
        assert verify_lifecycle["active"] is False
        verify_lifecycle["loader_calls"] += 1
        return _TinyModel.from_pretrained(path)

    verified = verify_experimental_child_artifact(
        publication,
        expectations=expectations,
        expected_receipt_sha256=receipt.receipt_sha256,
        model_loader=verify_loader,
        native_certifier=_native_certificate,
        runtime_auditor=_zero_audit,
    )
    assert verified.receipt.as_dict() == payload
    assert verified.artifact_path == (publication / ARTIFACT_DIRECTORY_NAME).resolve()
    assert verify_lifecycle["loader_calls"] == 1


def test_noncanonical_bridge_report_fails_before_serialization_and_cleans(
    tmp_path: Path,
) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    canonical = _bridge_report_bytes(model, expectations)
    noncanonical = (
        json.dumps(json.loads(canonical), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / "noncanonical"
    saver_calls = 0

    def saver(_model: Any, _path: Path) -> None:
        nonlocal saver_calls
        saver_calls += 1

    with pytest.raises(
        ExperimentalChildArtifactIntegrityError, match="uniquely canonical"
    ):
        publish_experimental_child_artifact(
            publication,
            model_saver=saver,
            **_publisher_kwargs(
                model,
                expectations,
                lifecycle,
                report_bytes=noncanonical,
            ),
        )
    assert saver_calls == 0
    assert lifecycle["active"] is False
    assert not publication.exists()
    assert not list(tmp_path.glob("*.staging"))
    assert not list(tmp_path.glob("*.lock"))


def test_reload_state_mismatch_fails_without_atomic_promotion(tmp_path: Path) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / "reload-mismatch"

    def changed_loader(path: Path) -> _TinyModel:
        loaded = _TinyModel.from_pretrained(path)
        with torch.no_grad():
            loaded.weight.add_(1.0)
        return loaded

    with pytest.raises(
        ExperimentalChildArtifactIntegrityError,
        match="does not match materialized bridge state",
    ):
        publish_experimental_child_artifact(
            publication,
            **_publisher_kwargs(
                model,
                expectations,
                lifecycle,
                model_loader=changed_loader,
            ),
        )
    assert lifecycle["active"] is False
    assert lifecycle["loader_calls"] == 1
    assert not publication.exists()
    assert not list(tmp_path.glob("*.staging"))
    assert not list(tmp_path.glob("*.lock"))


def test_reload_tree_mutation_fails_without_atomic_promotion(tmp_path: Path) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / "tree-mutation"

    def mutating_loader(path: Path) -> _TinyModel:
        loaded = _TinyModel.from_pretrained(path)
        (path / "chat_template.jinja").write_text("mutated", encoding="utf-8")
        return loaded

    with pytest.raises(
        ExperimentalChildArtifactPublicationError,
        match="mutated the staged artifact tree",
    ):
        publish_experimental_child_artifact(
            publication,
            **_publisher_kwargs(
                model,
                expectations,
                lifecycle,
                model_loader=mutating_loader,
            ),
        )
    assert not publication.exists()
    assert not list(tmp_path.glob("*.staging"))


@pytest.mark.parametrize("sidecar_name", ["factor_state.bin", "optimizer.pt"])
def test_strict_allowlist_rejects_sidecars(tmp_path: Path, sidecar_name: str) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / f"sidecar-{sidecar_name.split('.')[0]}"

    def tokenizer_saver(tokenizer: _TinyTokenizer, path: Path) -> None:
        tokenizer.save_pretrained(path)
        (path / sidecar_name).write_bytes(b"forbidden")

    with pytest.raises(
        ExperimentalChildArtifactIntegrityError, match="sidecar-like file"
    ):
        publish_experimental_child_artifact(
            publication,
            tokenizer_saver=tokenizer_saver,
            **_publisher_kwargs(model, expectations, lifecycle),
        )
    assert not publication.exists()


def test_parameterless_adapter_module_is_rejected_even_if_auditor_lies(
    tmp_path: Path,
) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / "adapter"

    def adapter_loader(path: Path) -> _TinyModel:
        loaded = _TinyModel.from_pretrained(path)
        loaded.adapter = _ParameterlessAdapter()
        return loaded

    with pytest.raises(
        ExperimentalChildArtifactIntegrityError,
        match="contains an adapter or residual registry",
    ):
        publish_experimental_child_artifact(
            publication,
            **_publisher_kwargs(
                model,
                expectations,
                lifecycle,
                model_loader=adapter_loader,
            ),
        )
    assert not publication.exists()


def test_runtime_auditor_must_prove_all_explicit_zero_counts(tmp_path: Path) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / "audit"
    kwargs = _publisher_kwargs(model, expectations, lifecycle)
    kwargs["runtime_auditor"] = lambda _model, _path: {
        **_zero_audit(model, tmp_path),
        "parent_runtime_reference_count": 1,
    }

    with pytest.raises(
        ExperimentalChildArtifactIntegrityError,
        match="parent_runtime_reference_count=0",
    ):
        publish_experimental_child_artifact(publication, **kwargs)
    assert not publication.exists()


def test_runtime_auditor_cannot_mutate_reloaded_model(tmp_path: Path) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / "mutating-auditor"
    kwargs = _publisher_kwargs(model, expectations, lifecycle)

    def mutating_auditor(loaded: _TinyModel, path: Path) -> dict[str, int]:
        with torch.no_grad():
            loaded.weight.mul_(2.0)
        return _zero_audit(loaded, path)

    kwargs["runtime_auditor"] = mutating_auditor
    with pytest.raises(
        ExperimentalChildArtifactIntegrityError,
        match="certifier or auditor mutated",
    ):
        publish_experimental_child_artifact(publication, **kwargs)
    assert not publication.exists()


def test_serialization_failure_cleans_staging_lock_and_partial_bytes(
    tmp_path: Path,
) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / "saver-failure"

    def failing_saver(_model: _TinyModel, path: Path) -> None:
        (path / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("synthetic saver failure")

    with pytest.raises(
        ExperimentalChildArtifactPublicationError,
        match="Bridge-context serialization failed",
    ):
        publish_experimental_child_artifact(
            publication,
            model_saver=failing_saver,
            **_publisher_kwargs(model, expectations, lifecycle),
        )
    assert lifecycle["active"] is False
    assert not publication.exists()
    assert not list(tmp_path.glob("*.staging"))
    assert not list(tmp_path.glob("*.lock"))


@pytest.mark.parametrize("target", ["weights", "bridge", "receipt", "extra"])
def test_postpublication_tampering_fails_before_verifier_loader(
    tmp_path: Path, target: str
) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / f"tamper-{target}"
    receipt = publish_experimental_child_artifact(
        publication,
        **_publisher_kwargs(model, expectations, lifecycle),
    )
    if target == "weights":
        path = publication / ARTIFACT_DIRECTORY_NAME / "model.safetensors"
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    elif target == "bridge":
        path = publication / BRIDGE_REPORT_FILENAME
        path.write_bytes(path.read_bytes() + b"\n")
    elif target == "receipt":
        path = publication / ARTIFACT_RECEIPT_FILENAME
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status"] = "accepted"
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    else:
        (publication / "unexpected.json").write_text("{}", encoding="utf-8")
    loader_calls = 0

    def forbidden_loader(_path: Path) -> _TinyModel:
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("metadata tampering must fail before model load")

    with pytest.raises(ExperimentalChildArtifactIntegrityError):
        verify_experimental_child_artifact(
            publication,
            expectations=expectations,
            expected_receipt_sha256=receipt.receipt_sha256,
            model_loader=forbidden_loader,
            native_certifier=_native_certificate,
            runtime_auditor=_zero_audit,
        )
    assert loader_calls == 0


def test_wrong_external_topology_binding_fails_before_verifier_loader(
    tmp_path: Path,
) -> None:
    model = _TinyModel()
    expectations = _expectations(model)
    lifecycle: dict[str, Any] = {}
    publication = tmp_path / "wrong-topology"
    receipt = publish_experimental_child_artifact(
        publication,
        **_publisher_kwargs(model, expectations, lifecycle),
    )
    wrong = ExperimentalChildArtifactExpectations(
        preregistration_sha256=expectations.preregistration_sha256,
        shared_execution_contract_sha256=(
            expectations.shared_execution_contract_sha256
        ),
        config_payload_sha256=expectations.config_payload_sha256,
        state_key_sha256=expectations.state_key_sha256,
        parameter_topology_sha256=_digest("wrong-topology"),
        buffer_topology_sha256=expectations.buffer_topology_sha256,
        exact_unique_parameters=expectations.exact_unique_parameters,
        exact_logical_parameters=expectations.exact_logical_parameters,
    )
    loader_calls = 0

    def forbidden_loader(_path: Path) -> _TinyModel:
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError

    with pytest.raises(
        ExperimentalChildArtifactIntegrityError, match="state/count/topology"
    ):
        verify_experimental_child_artifact(
            publication,
            expectations=wrong,
            expected_receipt_sha256=receipt.receipt_sha256,
            model_loader=forbidden_loader,
            native_certifier=_native_certificate,
            runtime_auditor=_zero_audit,
        )
    assert loader_calls == 0
