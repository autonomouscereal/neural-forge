from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.numeric_execution_policy import (
    NUMERIC_CAST_LEDGER_SCHEMA,
    NUMERIC_EXECUTION_POLICY_SCHEMA,
    NUMERIC_EXECUTION_RECEIPT_SCHEMA,
    NUMERIC_MODEL_INVENTORY_SCHEMA,
    ModelNumericInventory,
    NumericAuthenticationError,
    NumericCastLedger,
    NumericCastLedgerReceipt,
    NumericDeviceLossError,
    NumericDomain,
    NumericExecutionPolicy,
    NumericExecutionPolicyError,
    NumericExecutionRequest,
    NumericFallbackExhaustedError,
    NumericNonFiniteError,
    NumericPrepublicationOutOfMemoryError,
    NumericRankError,
    NumericShapeError,
    NumericUnsupportedBackendError,
    SUPPORTED_FLOAT_DTYPES,
    inventory_model_numeric_state,
    normalize_numeric_dtype,
    require_authentication,
    require_finite_tensor,
    require_full_rank,
    require_tensor_shape,
    resolve_numeric_execution_policy,
    run_with_numeric_fallback,
    verify_numeric_execution_receipt,
)


class _UniformModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.ones((2, 3), dtype=dtype))
        self.register_buffer("counter", torch.arange(3, dtype=torch.int64))


class _MixedModel(torch.nn.Module):
    def __init__(self, first: torch.dtype, second: torch.dtype) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.ones((2, 3), dtype=first))
        self.second = torch.nn.Parameter(torch.ones((3, 2), dtype=second))
        self.register_buffer("floating_buffer", torch.ones(2, dtype=torch.float16))
        self.register_buffer("integer_buffer", torch.arange(2, dtype=torch.int64))


def _cuda_status(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool,
    count: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def _cpu_policy(**request_kwargs: object) -> NumericExecutionPolicy:
    return resolve_numeric_execution_policy(
        source_models={"parent": _UniformModel()},
        target_model=_UniformModel(),
        request=NumericExecutionRequest(**request_kwargs),
    )


def test_inventory_authenticates_mixed_parameters_buffers_and_integer_state() -> None:
    inventory = inventory_model_numeric_state(
        _MixedModel(torch.bfloat16, torch.float32),
        role="source:parent",
    )
    receipt = inventory.as_dict()

    assert receipt["schema"] == NUMERIC_MODEL_INVENTORY_SCHEMA
    assert receipt["mixed_floating_parameter_dtypes"] is True
    assert receipt["mixed_floating_storage_dtypes"] is True
    assert receipt["floating_parameter_dtype_counts"] == {
        "bfloat16": 1,
        "float32": 1,
    }
    assert receipt["floating_buffer_dtype_counts"] == {"float16": 1}
    assert any(entry["dtype"] == "int64" for entry in receipt["entries"])
    assert receipt["inventory_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "inventory_sha256"}
    )
    assert ModelNumericInventory(json.loads(json.dumps(receipt))) == inventory


def test_policy_keeps_mixed_storage_and_forward_domains_explicit() -> None:
    policy = resolve_numeric_execution_policy(
        source_models={
            "language": _MixedModel(torch.bfloat16, torch.float32),
            "vision": _UniformModel(torch.float64),
        },
        target_model=_MixedModel(torch.float16, torch.float32),
    )

    source_storage = policy.placement(NumericDomain.SOURCE_STORAGE)
    source_forward = policy.placement(NumericDomain.SOURCE_FORWARD)
    target_storage = policy.placement(NumericDomain.TARGET_STORAGE)
    publication = policy.placement(NumericDomain.PUBLICATION)

    assert source_storage.resolved_dtype == "mixed"
    assert source_forward.resolved_dtype == "model-native"
    assert target_storage.resolved_dtype == "mixed"
    assert publication.resolved_dtype == "target-native"
    assert publication.is_concrete is False
    assert len(source_forward.inventory_sha256s) == 2
    assert policy.placement(NumericDomain.ACCUMULATION).concrete.dtype == "float32"
    assert policy.placement(NumericDomain.SOLVE).concrete.dtype == "float64"
    assert policy.placement(NumericDomain.FACTORIZATION).concrete.dtype == "float64"
    assert policy.placement(NumericDomain.VERIFICATION).concrete.dtype == "float64"


def test_all_four_float_dtypes_can_be_assigned_to_distinct_domains() -> None:
    policy = _cpu_policy(
        source_forward_device="cpu",
        source_forward_dtype="fp16",
        target_forward_device="cpu",
        target_forward_dtype="bf16",
        accumulation_device="cpu",
        accumulation_dtype="fp32",
        solve_device="cpu",
        solve_dtype="fp64",
        factorization_device="cpu",
        factorization_dtype="float32",
        publication_device="cpu",
        publication_dtype="float16",
        verification_device="cpu",
        verification_dtype="float64",
        target_verification_device="cpu",
        target_verification_dtype="bfloat16",
    )

    assert policy.placement(NumericDomain.SOURCE_FORWARD).concrete.dtype == "float16"
    assert policy.placement(NumericDomain.TARGET_FORWARD).concrete.dtype == "bfloat16"
    assert policy.placement(NumericDomain.ACCUMULATION).concrete.dtype == "float32"
    assert policy.placement(NumericDomain.SOLVE).concrete.dtype == "float64"
    assert policy.placement(NumericDomain.FACTORIZATION).concrete.dtype == "float32"
    assert policy.placement(NumericDomain.PUBLICATION).concrete.dtype == "float16"
    assert policy.placement(NumericDomain.VERIFICATION).concrete.dtype == "float64"
    assert (
        policy.placement(NumericDomain.TARGET_VERIFICATION).concrete.dtype
        == "bfloat16"
    )


def test_policy_is_json_roundtrippable_immutable_and_self_sealed() -> None:
    policy = _cpu_policy()
    receipt = policy.as_dict()

    assert receipt["schema"] == NUMERIC_EXECUTION_POLICY_SCHEMA
    assert receipt["supported_float_dtypes"] == list(SUPPORTED_FLOAT_DTYPES)
    assert receipt["ordinary_dtype_cast_is_quantization"] is False
    assert receipt["deterministic_settings"]["autocast_enabled"] is False
    assert receipt["deterministic_settings"]["cuda_tf32_allowed"] is False
    assert receipt["policy_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "policy_sha256"}
    )
    assert NumericExecutionPolicy(json.loads(json.dumps(receipt))) == policy
    with pytest.raises(TypeError):
        policy.receipt["schema"] = "tampered"  # type: ignore[index]
    with pytest.raises(TypeError):
        policy.receipt["placements"]["solve"]["resolved_dtype"] = "float32"  # type: ignore[index]


@pytest.mark.parametrize(
    "field",
    (
        "schema",
        "supported_float_dtypes",
        "placements",
        "fallback_policy",
        "policy_sha256",
    ),
)
def test_policy_tampering_fails_authentication(field: str) -> None:
    receipt = _cpu_policy().as_dict()
    if field == "schema":
        receipt[field] = "bad"
    elif field == "supported_float_dtypes":
        receipt[field] = ["float32"]
    elif field == "placements":
        receipt[field]["solve"]["resolved_dtype"] = "float32"
    elif field == "fallback_policy":
        receipt[field]["eligible_failure_classes"].append("runtime-error")
    else:
        receipt[field] = "0" * 64

    with pytest.raises(NumericExecutionPolicyError):
        NumericExecutionPolicy(receipt)


def test_unavailable_requested_cuda_resolves_to_cpu_fp64_only_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_status(monkeypatch, available=False, count=0)
    policy = _cpu_policy(solve_device="cuda:0", solve_dtype="float32")
    solve = policy.placement(NumericDomain.SOLVE)

    assert solve.requested_device == "cuda:0"
    assert solve.requested_dtype == "float32"
    assert solve.resolved_device == "cpu"
    assert solve.resolved_dtype == "float64"
    assert solve.resolution_fallback_used is True
    assert solve.execution_candidates == (
        solve.concrete,
    )

    with pytest.raises(NumericExecutionPolicyError):
        _cpu_policy(
            solve_device="cuda:0",
            solve_dtype="float32",
            allow_cpu_fp64_fallback=False,
        )


def test_low_precision_factorization_is_not_attempted_and_uses_cpu_fp64() -> None:
    policy = _cpu_policy(
        solve_device="cpu",
        solve_dtype="bfloat16",
        factorization_device="cpu",
        factorization_dtype="float16",
    )

    assert policy.placement(NumericDomain.SOLVE).concrete.dtype == "float64"
    factorization = policy.placement(NumericDomain.FACTORIZATION)
    assert factorization.concrete.dtype == "float64"
    assert factorization.resolution_fallback_used is True
    assert any(
        record["reason"] == "low-precision-factorization-disallowed"
        for record in factorization.capability_records
    )


def test_explicit_capability_override_is_sealed_and_drives_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_status(monkeypatch, available=True, count=1)
    policy = resolve_numeric_execution_policy(
        source_models={"parent": _UniformModel()},
        target_model=_UniformModel(),
        request=NumericExecutionRequest(
            solve_device="cuda:0",
            solve_dtype="float64",
        ),
        capability_overrides={
            ("linear_solve", "cuda:0", "float64"): {
                "supported": False,
                "reason": "synthetic-kernel-gap",
                "runtime_preflight_required": False,
            }
        },
    )

    solve = policy.placement(NumericDomain.SOLVE)
    assert solve.concrete.device == "cpu"
    assert solve.resolution_fallback_used is True
    assert any(
        record["source"] == "explicit-override"
        and record["reason"] == "synthetic-kernel-gap"
        for record in solve.capability_records
    )


@pytest.mark.parametrize(
    "failure_type",
    (
        NumericUnsupportedBackendError,
        NumericDeviceLossError,
        NumericPrepublicationOutOfMemoryError,
    ),
)
def test_only_typed_backend_failures_retry_on_cpu_fp64(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    _cuda_status(monkeypatch, available=True, count=1)
    policy = _cpu_policy(solve_device="cuda:0", solve_dtype="float32")
    calls: list[tuple[str, str]] = []

    def execute(placement):
        calls.append((placement.device, placement.dtype))
        if placement.device == "cuda:0":
            raise failure_type("synthetic eligible failure")
        return torch.tensor([3.0], dtype=placement.torch_dtype)

    result = run_with_numeric_fallback(
        policy,
        domain=NumericDomain.SOLVE,
        operation="linear_solve",
        execute=execute,
    )
    receipt = result.receipt_dict()

    assert calls == [("cuda:0", "float32"), ("cpu", "float64")]
    assert result.value.dtype is torch.float64
    assert receipt["schema"] == NUMERIC_EXECUTION_RECEIPT_SCHEMA
    assert receipt["runtime_fallback_used"] is True
    assert receipt["fallback_used"] is True
    assert receipt["attempts"][0]["failure_class"] == failure_type.reason_code
    assert receipt["attempts"][1]["status"] == "succeeded"
    assert receipt["execution_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "execution_sha256"}
    )
    assert dict(
        verify_numeric_execution_receipt(
            json.loads(json.dumps(receipt)),
            expected_policy_sha256=policy.policy_sha256,
        )
    )["status"] == "succeeded"

    receipt["attempts"][0]["failure_class"] = "rank-failure"
    body = {key: value for key, value in receipt.items() if key != "execution_sha256"}
    receipt["execution_sha256"] = canonical_sha256(body)
    with pytest.raises(NumericExecutionPolicyError):
        verify_numeric_execution_receipt(receipt)


@pytest.mark.parametrize(
    "failure",
    (
        NumericNonFiniteError("nonfinite"),
        NumericRankError("rank"),
        NumericAuthenticationError("authentication"),
        NumericShapeError("shape"),
        RuntimeError("untyped runtime error"),
    ),
)
def test_numerical_auth_shape_and_untyped_runtime_failures_never_fallback(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    _cuda_status(monkeypatch, available=True, count=1)
    policy = _cpu_policy(solve_device="cuda:0", solve_dtype="float32")
    calls = 0

    def execute(_placement):
        nonlocal calls
        calls += 1
        raise failure

    with pytest.raises(type(failure)):
        run_with_numeric_fallback(
            policy,
            domain="solve",
            operation="linear_solve",
            execute=execute,
        )
    assert calls == 1


def test_nonfinite_success_value_is_rejected_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_status(monkeypatch, available=True, count=1)
    policy = _cpu_policy(solve_device="cuda:0", solve_dtype="float32")
    calls = 0

    def execute(_placement):
        nonlocal calls
        calls += 1
        return torch.tensor([float("nan")])

    with pytest.raises(NumericNonFiniteError):
        run_with_numeric_fallback(
            policy,
            domain="solve",
            operation="linear_solve",
            execute=execute,
        )
    assert calls == 1


def test_exhausted_eligible_fallback_has_a_sealed_failure_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_status(monkeypatch, available=True, count=1)
    policy = _cpu_policy(solve_device="cuda:0", solve_dtype="float32")

    with pytest.raises(NumericFallbackExhaustedError) as captured:
        run_with_numeric_fallback(
            policy,
            domain="solve",
            operation="linear_solve",
            execute=lambda _placement: (_ for _ in ()).throw(
                NumericUnsupportedBackendError("unsupported")
            ),
        )

    receipt = dict(captured.value.receipt)
    assert receipt["status"] == "failed"
    assert len(receipt["attempts"]) == 2
    assert receipt["resolved_device"] is None


def test_cast_ledger_records_real_projection_loss_for_all_four_dtypes() -> None:
    policy = _cpu_policy()
    ledger = NumericCastLedger(policy)
    source = torch.tensor([1.1, 1.0e-8, -3.25], dtype=torch.float64)
    results = {}

    for dtype in SUPPORTED_FLOAT_DTYPES:
        results[dtype] = ledger.cast(
            source,
            tensor_role=f"candidate.{dtype}",
            domain=NumericDomain.PUBLICATION,
            target_dtype=dtype,
            target_device="cpu",
        )
    sealed = ledger.seal()
    receipt = sealed.as_dict()

    assert results["float16"].dtype is torch.float16
    assert results["bfloat16"].dtype is torch.bfloat16
    assert results["float32"].dtype is torch.float32
    assert results["float64"].dtype is torch.float64
    assert receipt["schema"] == NUMERIC_CAST_LEDGER_SCHEMA
    assert receipt["entry_count"] == 4
    assert receipt["lossy_cast_count"] == 3
    assert receipt["underflow_coordinate_count"] >= 1
    assert receipt["ordinary_dtype_cast_is_quantization"] is False
    assert all(entry["source_finite"] for entry in receipt["entries"])
    assert all(entry["target_finite"] for entry in receipt["entries"])
    assert NumericCastLedgerReceipt(json.loads(json.dumps(receipt))) == sealed
    with pytest.raises(NumericExecutionPolicyError):
        ledger.cast(
            source,
            tensor_role="late",
            domain="publication",
            target_dtype="float32",
            target_device="cpu",
        )


def test_model_native_publication_requires_a_concrete_destination() -> None:
    policy = resolve_numeric_execution_policy(
        source_models={"parent": _UniformModel()},
        target_model=_MixedModel(torch.bfloat16, torch.float32),
    )
    ledger = NumericCastLedger(policy)
    source = torch.ones((2, 3), dtype=torch.float64)

    with pytest.raises(NumericExecutionPolicyError):
        ledger.cast(
            source,
            tensor_role="ambiguous",
            domain="publication",
        )
    target_like = torch.empty((2, 3), dtype=torch.bfloat16)
    result = ledger.cast(
        source,
        tensor_role="parameter.first",
        domain="publication",
        target_like=target_like,
    )
    assert result.dtype is torch.bfloat16


def test_cast_and_validation_helpers_raise_nonfallback_error_classes() -> None:
    policy = _cpu_policy()
    ledger = NumericCastLedger(policy)

    with pytest.raises(NumericNonFiniteError):
        ledger.cast(
            torch.tensor([float("inf")]),
            tensor_role="bad",
            domain="publication",
        )
    with pytest.raises(NumericNonFiniteError):
        require_finite_tensor(torch.tensor([float("nan")]))
    with pytest.raises(NumericShapeError):
        require_tensor_shape(torch.ones(2), (3,))
    with pytest.raises(NumericAuthenticationError):
        require_authentication(False)
    with pytest.raises(NumericRankError):
        require_full_rank(2, 3)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("fp16", "float16"),
        (torch.bfloat16, "bfloat16"),
        ("torch.float32", "float32"),
        ("double", "float64"),
    ),
)
def test_dtype_aliases_are_canonical(value: object, expected: str) -> None:
    assert normalize_numeric_dtype(value) == expected


@pytest.mark.parametrize("value", ("float8", "int8", object()))
def test_unsupported_dtype_is_rejected(value: object) -> None:
    with pytest.raises(NumericExecutionPolicyError):
        normalize_numeric_dtype(value)


def test_policy_resolution_has_no_model_execution_training_or_write_surface() -> None:
    source = Path("src/neural_forge/numeric_execution_policy.py").read_text(
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
