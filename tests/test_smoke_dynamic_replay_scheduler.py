from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "smoke_dynamic_replay_scheduler.py"
)
SPEC = importlib.util.spec_from_file_location(
    "smoke_dynamic_replay_scheduler",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _device(index: int) -> dict[str, Any]:
    return {
        "device_index": index,
        "device_name": MODULE.EXPECTED_DEVICE_NAME,
        "device_uuid": f"GPU-exact-{index}",
        "pci_domain_id": 0,
        "pci_bus_id": index + 2,
        "pci_device_id": index + 3,
    }


def _execution() -> dict[str, Any]:
    parity = [
        {
            "device_index": index,
            "passed": True,
            "tensor_reports": [
                {"dtype": "bfloat16", "passed": True},
                {"dtype": "float32", "passed": True},
            ],
        }
        for index in (0, 1)
    ]
    records = [
        {
            "ordinal": ordinal,
            "device_index": ordinal % 2,
        }
        for ordinal in range(MODULE.WORK_ITEM_COUNT)
    ]
    return {
        "parity_reports": parity,
        "scheduled_records": records,
        "parity_passed_before_scheduled_work": True,
        "canonical_result_order_restored": True,
        "backend_receipts_reauthenticated": True,
        "installed_state_reauthenticated": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "folding_math_changed": False,
    }


def test_import_pins_cublas_without_importing_torch_into_runner_globals() -> None:
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "torch" not in vars(MODULE)


def test_exact_device_contract_discloses_two_distinct_uuid_and_pci_ids() -> None:
    validated = MODULE._validate_device_documents((_device(1), _device(0)))
    assert [row["device_index"] for row in validated] == [0, 1]
    assert len({row["device_uuid"] for row in validated}) == 2
    assert len(
        {
            (
                row["pci_domain_id"],
                row["pci_bus_id"],
                row["pci_device_id"],
            )
            for row in validated
        }
    ) == 2

    with pytest.raises(MODULE.SchedulerSmokeError, match="Exactly two"):
        MODULE._validate_device_documents((_device(0),))
    wrong = _device(1)
    wrong["device_name"] = "NVIDIA RTX 6000 Ada Generation"
    with pytest.raises(MODULE.SchedulerSmokeError, match="RTX 3090"):
        MODULE._validate_device_documents((_device(0), wrong))
    duplicate = _device(1)
    duplicate["device_uuid"] = _device(0)["device_uuid"]
    with pytest.raises(MODULE.SchedulerSmokeError, match="UUIDs"):
        MODULE._validate_device_documents((_device(0), duplicate))


def test_execution_contract_requires_both_dtype_parities_and_canonical_shards() -> None:
    MODULE._validate_execution_report(_execution())

    missing_bf16 = _execution()
    missing_bf16["parity_reports"][0]["tensor_reports"] = [
        {"dtype": "float32", "passed": True}
    ]
    with pytest.raises(MODULE.SchedulerSmokeError, match="BF16 and FP32"):
        MODULE._validate_execution_report(missing_bf16)

    noncanonical = _execution()
    noncanonical["scheduled_records"] = list(
        reversed(noncanonical["scheduled_records"])
    )
    with pytest.raises(MODULE.SchedulerSmokeError, match="canonical"):
        MODULE._validate_execution_report(noncanonical)


def test_success_receipt_is_canonical_sealed_and_atomically_create_only(
    tmp_path: Path,
) -> None:
    body = {
        "schema": MODULE.SCHEMA,
        "status": "passed",
        "numeric_parity_tolerances": {
            "absolute_tolerance": MODULE.ABSOLUTE_TOLERANCE,
            "relative_tolerance": MODULE.RELATIVE_TOLERANCE,
        },
    }
    receipt = MODULE._sealed_success_receipt(body)
    seal = receipt.pop("receipt_sha256")
    assert seal == MODULE.canonical_sha256(receipt)
    receipt["receipt_sha256"] = seal

    destination = tmp_path / "smoke.json"
    encoded = MODULE.canonical_json_bytes(receipt) + b"\n"
    MODULE._atomic_write_new(destination, encoded)
    assert destination.read_bytes() == encoded
    assert json.loads(destination.read_text(encoding="ascii")) == receipt

    with pytest.raises(MODULE.SchedulerSmokeError, match="already exists"):
        MODULE._atomic_write_new(destination, b"different")
    assert destination.read_bytes() == encoded
    assert not list(tmp_path.glob(".smoke.json.stage-*"))


def test_failure_path_never_publishes_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "must-not-exist.json"

    def fail(_output: Path) -> dict[str, Any]:
        raise MODULE.SchedulerSmokeError("injected failure")

    monkeypatch.setattr(MODULE, "run_smoke", fail)
    assert MODULE.main(["--output", str(destination)]) == 1
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_output_argument_is_required() -> None:
    with pytest.raises(SystemExit):
        MODULE._parser().parse_args([])
