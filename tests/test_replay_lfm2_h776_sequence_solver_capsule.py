from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from experiments import replay_lfm2_h776_sequence_solver_capsule as replay


_TINY_TOPOLOGY = {
    "target_query_heads": 10,
    "target_key_value_heads": 5,
    "target_query_heads_per_group": 2,
    "target_hidden_size": 2,
    "target_output_size": 2,
    "head_dim": 1,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {
        **body,
        "report_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, torch.Tensor]]:
    monkeypatch.setattr(replay, "H776_TOPOLOGY", dict(_TINY_TOPOLOGY))
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    tensors = {
        "cross_head_gram": torch.arange(
            10 * 10 * 2 * 2,
            dtype=torch.float64,
        ).reshape(10, 10, 2, 2),
        "endpoint_right": torch.arange(
            10 * 2 * 2,
            dtype=torch.float64,
        ).reshape(10, 2, 2),
        "value_factor": torch.arange(
            5 * 2,
            dtype=torch.float64,
        ).reshape(5, 2, 1),
        "output_blocks": torch.arange(
            10 * 1 * 2,
            dtype=torch.float64,
        ).reshape(10, 1, 2),
    }
    policy = replay.sequence_fold.Lfm2FixedQkSequenceEffectiveVoAlsPolicy().as_dict()
    payload = {
        **tensors,
        "endpoint_square_sum": 123.0,
        "row_count": 17,
        "topology": dict(_TINY_TOPOLOGY),
        "policy": policy,
        "iteration": 1,
    }
    payload_path = capsule / "group0-iteration1.pt"
    torch.save(payload, payload_path)
    payload_sha256 = _file_sha256(payload_path)
    payload_bytes = payload_path.stat().st_size
    monkeypatch.setattr(
        replay,
        "H776_CAPSULE_PAYLOAD_SHA256",
        payload_sha256,
    )
    monkeypatch.setattr(replay, "H776_CAPSULE_PAYLOAD_BYTES", payload_bytes)
    manifest_body = {
        "schema": replay.H776_SOLVER_CAPSULE_SCHEMA,
        "diagnostic_only": True,
        "payload_file": payload_path.name,
        "payload_file_sha256": payload_sha256,
        "payload_file_bytes": payload_bytes,
        "tensor_sha256": {
            name: replay.sequence_fold._tensor_sha256(tensor)
            for name, tensor in tensors.items()
        },
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "endpoint_square_sum": payload["endpoint_square_sum"],
        "row_count": payload["row_count"],
        "topology": dict(_TINY_TOPOLOGY),
        "policy": policy,
        "iteration": 1,
        "source_sequence_file_sha256": hashlib.sha256(b"captured-source").hexdigest(),
        "solver_executed": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    capsule_sha256 = hashlib.sha256(_canonical_json(manifest_body)).hexdigest()
    monkeypatch.setattr(replay, "H776_CAPSULE_SHA256", capsule_sha256)
    _write_json(
        capsule / "manifest.json",
        {**manifest_body, "capsule_sha256": capsule_sha256},
    )
    return capsule, tensors


def _production_report(
    *,
    group: int,
    solution: torch.Tensor,
    final_residual: float = 5e-9,
) -> dict[str, Any]:
    role = f"sequence_effective_structured_value_iteration_1_group_{group}"
    solution_sha256 = replay.sequence_fold._tensor_sha256(solution)
    partial_trace_sha256 = hashlib.sha256(
        f"partial-{group}".encode("ascii")
    ).hexdigest()
    construction = _seal(
        {
            "schema": (
                replay.sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_CHOLESKY_SCHEMA
            ),
            "role": role,
            "hidden_size": _TINY_TOPOLOGY["target_hidden_size"],
            "head_dim": _TINY_TOPOLOGY["head_dim"],
            "partial_trace_sha256": partial_trace_sha256,
            "full_operator_materialized": False,
        }
    )
    factor_sha256 = hashlib.sha256(f"factor-{group}".encode("ascii")).hexdigest()
    receipt = _seal(
        {
            "kind": "deterministic_hidden_partial_trace_cholesky",
            "construction_report": construction,
            "construction_report_sha256": construction["report_sha256"],
            "partial_trace_sha256": partial_trace_sha256,
            "factorization": "lower_cholesky_ex",
            "factorization_info": 0,
            "factor_sha256": factor_sha256,
            "factor_shape": [2, 2],
            "factor_dtype": "torch.float64",
            "factor_device": "cpu",
            "hidden_size": 2,
            "head_dim": 1,
            "partial_trace_spd": True,
            "positive_finite_factor_diagonal": True,
            "inverse_orientation_probe_relative_residual": 1e-15,
            "inverse_orientation_probe_tolerance": 1e-10,
            "factorization_materialized_only_after_original_lane_exact_miss": True,
            "full_operator_materialized": False,
        }
    )
    execution_profile = dict(
        replay.sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE
    )
    pcg = _seal(
        {
            "role": role,
            "solver": replay._EXPECTED_PCG_SOLVER,
            "maximum_iterations": 4096,
            "hard_cap_includes_original_lane_iterations": True,
            "original_lane_maximum_iterations": 64,
            "original_lane_iterations_executed": 64,
            "iterations_executed": 121,
            "continuous_hidden_partial_trace_lane_iterations_executed": 57,
            "single_restart_after_original_lane_exact_authentication_miss": True,
            "required_relative_normal_residual": 1e-8,
            "final_relative_normal_residual": final_residual,
            "exact_final_residual_authenticated": True,
            "solution_finite": True,
            "solution_sha256": solution_sha256,
            "periodic_restart_count": 0,
            "extended_after_original_lane_exact_authentication_miss": True,
            "exact_residual_used_to_initialize_continuous_lane_once": True,
            "exact_diagnostic_matvecs_modify_recurrence_state_or_direction": False,
            "execution_profile": execution_profile,
            "execution_profile_sha256": execution_profile["report_sha256"],
            "restart_preconditioner": {
                "kind": "deterministic_hidden_partial_trace_cholesky",
                "supplied": True,
                "authentication_attempted": True,
                "authenticated": True,
                "applied": True,
                "application_attempt_count": 1,
                "application_count": 1,
                "receipt": receipt,
                "authentication_failure_reason": None,
            },
        }
    )
    start = group * 2
    return _seal(
        {
            "role": role,
            "group": group,
            "heads": [start, start + 1],
            "features": 2,
            "ridge_scale": 1e-4,
            "trace_scale": 1.0,
            "ridge": 1e-4,
            "pcg_report": pcg,
            "hidden_partial_trace_construction_report": construction,
            "hidden_partial_trace_supplied_to_restart_lane": True,
            "hidden_partial_trace_pcg_execution_profile_sha256": (
                replay.sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
            ),
            "solution_sha256": solution_sha256,
            "explicit_coefficient_tensor_materialized": False,
            "hidden_head_dim_gram_materialized": False,
        }
    )


def _install_fake_sweep(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]],
    *,
    bad_report: str | None = None,
    fail: bool = False,
) -> torch.Tensor:
    final = torch.arange(10, dtype=torch.float64).reshape(5, 2, 1) + 0.25

    def fake_sweep(**kwargs: Any) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        calls.append(kwargs)
        assert torch.is_grad_enabled() is False
        if fail:
            raise replay.sequence_fold.Lfm2SequenceEffectiveAttentionFoldError(
                "production failure"
            )
        reports = [
            _production_report(group=group, solution=final[group]) for group in range(5)
        ]
        if bad_report == "tampered_seal":
            reports[3]["pcg_report"]["final_relative_normal_residual"] = 9e-9
        elif bad_report == "resealed_bad_residual":
            pcg_body = dict(reports[3]["pcg_report"])
            pcg_body.pop("report_sha256")
            pcg_body["final_relative_normal_residual"] = 1.1e-8
            reports[3]["pcg_report"] = _seal(pcg_body)
            outer_body = dict(reports[3])
            outer_body.pop("report_sha256")
            reports[3] = _seal(outer_body)
        return final.clone(), reports

    monkeypatch.setattr(
        replay.sequence_fold,
        "_structured_value_pcg_sweep",
        fake_sweep,
    )
    return final


def test_replay_calls_the_live_production_sweep_once_and_seals_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule, tensors = _make_capsule(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    final = _install_fake_sweep(monkeypatch, calls)
    expected_final = replay.sequence_fold._tensor_sha256(final)
    destination = tmp_path / "replay.json"

    document = replay.replay_h776_production_capsule(
        replay.H776ProductionCapsuleReplayConfig(
            capsule_directory=capsule,
            destination_json=destination,
            expected_final_value_factor_sha256=expected_final,
        )
    )

    assert len(calls) == 1
    assert calls[0]["iteration"] == 1
    assert replay.sequence_fold._tensor_sha256(
        calls[0]["value_factor"]
    ) == replay.sequence_fold._tensor_sha256(tensors["value_factor"])
    assert replay.sequence_fold._tensor_sha256(
        calls[0]["statistics"]["cross_head_gram"]
    ) == replay.sequence_fold._tensor_sha256(tensors["cross_head_gram"])
    assert document["production_sweep_calls"] == 1
    assert document["production_group_count"] == 5
    assert document["final_value_factor_sha256"] == expected_final
    assert document["execution"]["training_steps"] == 0
    assert document["execution"]["backward_calls"] == 0
    assert document["execution"]["optimizer_steps"] == 0
    assert document["execution"]["parameters_pruned"] == 0
    assert document["execution"]["parameters_quantized"] == 0
    body = dict(document)
    supplied = body.pop("replay_sha256")
    assert hashlib.sha256(_canonical_json(body)).hexdigest() == supplied
    assert destination.read_bytes() == _canonical_json(document) + b"\n"


@pytest.mark.parametrize(
    "bad_report",
    ["tampered_seal", "resealed_bad_residual"],
)
def test_replay_rejects_tampered_or_failed_production_report_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_report: str,
) -> None:
    capsule, _tensors = _make_capsule(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    final = _install_fake_sweep(
        monkeypatch,
        calls,
        bad_report=bad_report,
    )
    destination = tmp_path / "rejected.json"

    with pytest.raises(replay.H776ProductionCapsuleReplayError):
        replay.replay_h776_production_capsule(
            replay.H776ProductionCapsuleReplayConfig(
                capsule_directory=capsule,
                destination_json=destination,
                expected_final_value_factor_sha256=(
                    replay.sequence_fold._tensor_sha256(final)
                ),
            )
        )

    assert len(calls) == 1
    assert not destination.exists()


def test_replay_rejects_tampered_capsule_before_calling_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule, _tensors = _make_capsule(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    final = _install_fake_sweep(monkeypatch, calls)
    with (capsule / "group0-iteration1.pt").open("ab") as handle:
        handle.write(b"tamper")
    destination = tmp_path / "rejected-capsule.json"

    with pytest.raises(replay.H776ProductionCapsuleReplayError):
        replay.replay_h776_production_capsule(
            replay.H776ProductionCapsuleReplayConfig(
                capsule_directory=capsule,
                destination_json=destination,
                expected_final_value_factor_sha256=(
                    replay.sequence_fold._tensor_sha256(final)
                ),
            )
        )

    assert calls == []
    assert not destination.exists()


def test_replay_propagates_production_failure_without_output_or_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capsule, _tensors = _make_capsule(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []
    final = _install_fake_sweep(monkeypatch, calls, fail=True)
    destination = tmp_path / "production-failure.json"

    with pytest.raises(replay.sequence_fold.Lfm2SequenceEffectiveAttentionFoldError):
        replay.replay_h776_production_capsule(
            replay.H776ProductionCapsuleReplayConfig(
                capsule_directory=capsule,
                destination_json=destination,
                expected_final_value_factor_sha256=(
                    replay.sequence_fold._tensor_sha256(final)
                ),
            )
        )

    assert len(calls) == 1
    assert not destination.exists()
