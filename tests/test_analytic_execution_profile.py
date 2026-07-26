from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_forge.analytic_execution_profile import (
    ANALYTIC_EXECUTION_PROFILE_SCHEMA,
    AnalyticExecutionProfile,
    AnalyticExecutionProfileError,
    resolve_analytic_execution_profile,
)
from neural_forge.dynamic_circuit_ir import canonical_sha256


def _cuda_status(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
    count: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def test_cpu_profile_is_immutable_serializable_and_self_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_status(monkeypatch, available=False, count=0)

    profile = resolve_analytic_execution_profile("cpu")
    receipt = profile.as_dict()

    assert profile.requested_device == "cpu"
    assert profile.resolved_device == "cpu"
    assert profile.torch_device == torch.device("cpu")
    assert profile.cpu_fallback_used is False
    assert profile.require_cuda is False
    assert profile.allow_cpu_fallback is False
    assert receipt["schema"] == ANALYTIC_EXECUTION_PROFILE_SCHEMA
    assert receipt["compute_dtype"] == "float64"
    assert receipt["cpu_thread_count"] == 1
    assert receipt["deterministic_algorithms_required"] is True
    assert receipt["cuda_tf32_allowed"] is False
    assert receipt["cudnn_benchmark_allowed"] is False
    assert receipt["cpu_fallback_reason"] == "none"
    assert profile.result_sha256 == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "result_sha256"}
    )
    assert AnalyticExecutionProfile(json.loads(json.dumps(receipt))) == profile
    with pytest.raises(TypeError):
        profile.receipt["resolved_device"] = "cuda:0"  # type: ignore[index]


@pytest.mark.parametrize(
    "requested",
    (
        "",
        "cuda",
        "cuda:-1",
        "cpu:0",
        "mps",
        object(),
    ),
)
def test_invalid_or_ambiguous_devices_are_rejected(requested: object) -> None:
    with pytest.raises(AnalyticExecutionProfileError):
        resolve_analytic_execution_profile(requested)


def test_unavailable_cuda_requires_explicit_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_status(monkeypatch, available=False, count=0)

    with pytest.raises(AnalyticExecutionProfileError):
        resolve_analytic_execution_profile("cuda:0")
    with pytest.raises(AnalyticExecutionProfileError):
        resolve_analytic_execution_profile("cuda:0", require_cuda=True)

    profile = resolve_analytic_execution_profile(
        "cuda:0",
        allow_cpu_fallback=True,
    )
    receipt = profile.as_dict()
    assert profile.requested_device == "cuda:0"
    assert profile.resolved_device == "cpu"
    assert profile.cpu_fallback_used is True
    assert receipt["cpu_fallback_reason"] == "cuda-runtime-unavailable"
    assert receipt["cuda_runtime_available"] is False
    assert receipt["cuda_device_count"] == 0


def test_require_cuda_resolves_only_an_available_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_status(monkeypatch, available=True, count=2)

    profile = resolve_analytic_execution_profile("cuda:1", require_cuda=True)
    assert profile.requested_device == "cuda:1"
    assert profile.resolved_device == "cuda:1"
    assert profile.torch_device == torch.device("cuda:1")
    assert profile.require_cuda is True
    assert profile.cpu_fallback_used is False

    with pytest.raises(AnalyticExecutionProfileError):
        resolve_analytic_execution_profile("cuda:2", require_cuda=True)
    fallback = resolve_analytic_execution_profile(
        "cuda:2",
        allow_cpu_fallback=True,
    )
    assert fallback.resolved_device == "cpu"
    assert fallback.as_dict()["cpu_fallback_reason"] == (
        "requested-cuda-index-unavailable"
    )


def test_cuda_probe_failure_is_an_authenticated_fallback_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe() -> bool:
        raise RuntimeError("synthetic CUDA probe failure")

    monkeypatch.setattr(torch.cuda, "is_available", fail_probe)
    profile = resolve_analytic_execution_profile(
        "cuda:0",
        allow_cpu_fallback=True,
    )
    assert profile.resolved_device == "cpu"
    assert profile.as_dict()["cpu_fallback_reason"] == (
        "cuda-availability-probe-failed"
    )


def test_profile_policy_flags_are_strict_and_mutually_exclusive() -> None:
    with pytest.raises(AnalyticExecutionProfileError):
        resolve_analytic_execution_profile("cpu", require_cuda=True)
    with pytest.raises(AnalyticExecutionProfileError):
        resolve_analytic_execution_profile(
            "cuda:0",
            require_cuda=True,
            allow_cpu_fallback=True,
        )
    with pytest.raises(AnalyticExecutionProfileError):
        resolve_analytic_execution_profile("cpu", require_cuda=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("resolved_device", "cuda:0"),
        ("compute_dtype", "float32"),
        ("cpu_thread_count", 2),
        ("cpu_fallback_used", True),
        ("result_sha256", "0" * 64),
    ),
)
def test_receipt_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    _cuda_status(monkeypatch, available=False, count=0)
    receipt = resolve_analytic_execution_profile("cpu").as_dict()
    receipt[field] = replacement

    with pytest.raises(AnalyticExecutionProfileError):
        AnalyticExecutionProfile(receipt)


def test_unknown_receipt_fields_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_status(monkeypatch, available=False, count=0)
    receipt = resolve_analytic_execution_profile("cpu").as_dict()
    receipt["unsealed_extension"] = True
    body = {key: value for key, value in receipt.items() if key != "result_sha256"}
    receipt["result_sha256"] = canonical_sha256(body)

    with pytest.raises(AnalyticExecutionProfileError):
        AnalyticExecutionProfile(receipt)


def test_profile_resolution_has_no_execution_training_or_write_surface() -> None:
    source = Path("src/neural_forge/analytic_execution_profile.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        ".forward(",
        ".generate(",
        ".backward(",
        "torch.optim",
        "optimizer.step(",
        "open(",
        "write_text(",
        "write_bytes(",
        "torch.save(",
    ):
        assert forbidden not in source
