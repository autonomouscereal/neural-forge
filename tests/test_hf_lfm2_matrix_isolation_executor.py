"""Real-kernel integration tests for the canonical-v2 isolation executor."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.hf_lfm2_matrix_cell import (  # noqa: E402
    MatrixCellRequest,
    run_matrix_cell,
)
from neural_forge.hf_lfm2_matrix_isolation_executor import (  # noqa: E402
    Lfm2MatrixIsolationExecutorError,
    t5_isolation_executor,
)
from neural_forge.matrix_cell_receipt import CellId  # noqa: E402
from neural_forge.matrix_technique_registry import (  # noqa: E402
    CellTechniqueRequest,
    ComponentSpec,
)


class _Evidence:
    def __init__(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(23)
        fit_input = torch.randn((64, 8), generator=generator, dtype=torch.float64)
        held_input = torch.randn((24, 8), generator=generator, dtype=torch.float64)
        weight = _parent_weight()
        self._fit = {
            "mlp.post_swiglu": fit_input,
            "mlp.down": fit_input @ weight.T,
            "residual.post_layer": fit_input @ weight.T,
        }
        self._heldout = {
            "mlp.post_swiglu": held_input,
            "mlp.down": held_input @ weight.T,
            "residual.post_layer": held_input @ weight.T,
        }

    @property
    def endpoints(self) -> tuple[str, ...]:
        return tuple(self._fit)

    @property
    def selection_sha256(self) -> str:
        return "e" * 64

    @property
    def fit_row_count(self) -> int:
        return 64

    @property
    def heldout_row_count(self) -> int:
        return 24

    def fit_rows(self, endpoint: str) -> torch.Tensor:
        return self._fit[endpoint]

    def heldout_rows(self, endpoint: str) -> torch.Tensor:
        return self._heldout[endpoint]


def _parent_weight() -> torch.Tensor:
    values = torch.arange(32, dtype=torch.float64).reshape(4, 8)
    return (values - values.mean()) / 32.0


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def _metric_groups(*, fabricated: bool) -> dict[str, Any]:
    measured = {
        "measurement_status": "measured",
        "measurement_source": "actual_model_forward",
        "relative_l2": 0.05,
        "cosine": 0.995,
    }
    groups: dict[str, Any] = {
        "weight": None,
        "component_write": dict(measured),
        "macrocell": dict(measured),
        "terminal": dict(measured),
        "attention_relation": None,
        "mlp_relation": dict(measured),
        "state_mixer_relation": None,
        "interface_relation": None,
        "cache_trajectory": None,
    }
    if fabricated:
        groups["macrocell"]["measurement_source"] = "fabricated_sentinel"
    return groups


class _Session:
    def __init__(
        self,
        request: MatrixCellRequest,
        phase_keys: dict[str, str],
    ) -> None:
        self._request = request
        self._phase_keys = phase_keys
        extra = bool(
            request.execution_policy.get("target_policy", {}).get("extra_owned")
        )
        weight_keys = ("w", "w2") if extra else ("w",)
        parent_weights = {"w": _parent_weight()}
        if extra:
            parent_weights["w2"] = torch.eye(4, dtype=torch.float64)
        component = ComponentSpec(
            component_id="C13",
            name="mlp_down",
            kind="mlp",
            site="L05",
            layer_index=5,
            weight_keys=weight_keys,
            contraction_axis="input",
            parent_output_width=4,
            parent_input_width=8,
            parent_parameter_count=sum(
                value.numel() for value in parent_weights.values()
            ),
            input_endpoint="mlp.post_swiglu",
            write_endpoint="mlp.down",
            macrocell_endpoint="residual.post_layer",
        )
        self.technique_request = CellTechniqueRequest(
            cell=CellId.parse(request.cell_id),
            component=component,
            level=request.level_id,
            level_width=7,
            seed=31,
            parent_weights=parent_weights,
            evidence=_Evidence(),
            knobs={"active_k": 1, "write_solve_sweeps": 1},
            workspace=Path.cwd(),
            log=lambda _message: None,
        )
        self.code_identity = {"backend": "synthetic-real-kernel"}
        self.evidence = {
            "selection_sha256": "e" * 64,
            "fit_rows": 64,
            "heldout_rows": 24,
            "parent_trace_recaptured": False,
        }
        self.numeric_execution = {
            role: {"dtype": "float64", "device": "cpu"}
            for role in (
                "source_storage",
                "source_forward",
                "child_forward",
                "accumulation",
                "factorization",
                "target_publication",
                "runtime_verification",
            )
        }
        self.resources = {"backend": "cpu-test", "parent_model_copies": 1}
        self.operation_audit = {
            "autograd_grad_calls": 0,
            "backward_calls": 0,
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "parent_trace_recaptures": 0,
            "network_access": 0,
        }
        self.checkpoint_reuse = {
            phase: {
                "status": "miss",
                "cache_key_sha256": key,
                "reason": "fresh deterministic test phase",
            }
            for phase, key in phase_keys.items()
        }
        self.closed = False

    def materialize_copy_exact_control(self, phase_key_sha256: str) -> dict[str, Any]:
        assert phase_key_sha256 == self._phase_keys["target_materialization"]
        return {
            "materialized": True,
            "authenticated": True,
            "control_state_sha256": "c" * 64,
        }

    def calibrate(self, phase_key_sha256: str) -> dict[str, Any]:
        assert phase_key_sha256 == self._phase_keys["evidence_preparation"]
        return {"copy_exact_reproduction_rel_l2": 0.0, "passed": True}

    def install(self, result: Any, phase_key_sha256: str) -> dict[str, Any]:
        assert phase_key_sha256 == self._phase_keys["component_install"]
        delivered = sorted(result.installed_weights)
        expected = list(self.technique_request.component.weight_keys)
        return {
            "expected_owned_weight_keys": expected,
            "delivered_weight_keys": delivered,
            "installed_weight_keys": delivered,
            "unchanged_nonowned_state_verified": True,
            "alias_closure_verified": True,
            "installed_tensor_sha256s": {
                key: _tensor_sha256(value)
                for key, value in result.installed_weights.items()
            },
            "pre_state_sha256": "a" * 64,
            "post_state_sha256": "b" * 64,
        }

    def _diagnostics(self) -> dict[str, Any]:
        fabricated = bool(
            self._request.execution_policy.get("behavior_policy", {}).get(
                "fabricated_metrics"
            )
        )
        return {
            "measurement_component_ids": ["C13"],
            "physical_component_ids": ["C13"],
            "metric_groups": _metric_groups(fabricated=fabricated),
            "numeric_decision_authority": False,
        }

    def evaluate_boundary(self) -> dict[str, Any]:
        return {
            "write_heldout_cos": 0.995,
            "write_heldout_rel_l2": 0.05,
            "post_operator_heldout_cos": 0.99,
            "post_operator_heldout_rel_l2": 0.08,
            "component_diagnostics": self._diagnostics(),
        }

    def evaluate_composed(self) -> dict[str, Any]:
        return {
            "terminal_heldout_rel_l2": 0.10,
            "vs_copy_exact_control_delta": 0.10,
        }

    def evaluate_free_run(self, phase_key_sha256: str) -> dict[str, Any]:
        assert phase_key_sha256 == self._phase_keys["behavior_evaluation"]
        return {"token_match_vs_control": 0.75, "mean_lcp": 0.5}

    def evaluate_decode_smoke(self, phase_key_sha256: str) -> list[dict[str, Any]]:
        assert phase_key_sha256 == self._phase_keys["behavior_evaluation"]
        return [
            {
                "prompt_id": "ordinary",
                "termination_reason": "eos",
                "output_verbatim": "A coherent but different sentence.",
                "safety_observation": None,
            }
        ]

    def close(self) -> None:
        self.closed = True


def fake_backend(
    request: MatrixCellRequest,
    _run_directory: Path,
    phase_key_sha256s: dict[str, str],
    _emit: Any,
) -> _Session:
    return _Session(request, phase_key_sha256s)


def _request(
    *,
    evidence_class: str = "isolation_only",
    extra_owned: bool = False,
    fabricated_metrics: bool = False,
) -> MatrixCellRequest:
    return MatrixCellRequest(
        instance_id="real-t5-c13",
        cell_id="C13.T5.M16.L05",
        component_ids=("C13",),
        technique_id="T5",
        level_id="M16",
        site_id="L05",
        evidence_class=evidence_class,
        campaign_wave=1,
        parent_parameter_count=229_693_184,
        target_parameter_count=57_420_144
        if evidence_class == "complete_child"
        else None,
        evidence_binding={"evidence_sha256": "e" * 64},
        execution_policy={
            "isolation_backend": f"{__name__}:fake_backend",
            "numeric_contract": {
                "source_storage_dtype": "float64",
                "solve_dtype": "float64",
                "target_publication_dtype": "float32",
            },
            "evidence_policy": {
                "selected_endpoints": [
                    "mlp.post_swiglu",
                    "mlp.down",
                    "residual.post_layer",
                ]
            },
            "target_policy": {"extra_owned": extra_owned},
            "behavior_policy": {"fabricated_metrics": fabricated_metrics},
        },
        technique_configuration={"active_k": 1, "write_solve_sweeps": 1},
    )


def test_real_t5_kernel_produces_canonical_v2_isolation_receipt(tmp_path: Path) -> None:
    receipt = run_matrix_cell(
        _request(),
        t5_isolation_executor(),
        tmp_path / "cell",
    )

    assert receipt["schema"] == "neural-forge-lfm2-matrix-cell-receipt-v2"
    assert receipt["compaction_claim_permitted"] is False
    result = receipt["result"]
    assert result["artifacts"]["execution_surface"] == (
        "parent_shell_component_substitution"
    )
    assert result["artifacts"]["stored_parameter_count_claim_authority"] is False
    assert result["solve"]["physical_storage_claim_authority"] is False
    assert (
        result["component_diagnostics"]["metric_groups"]["mlp_relation"][
            "measurement_source"
        ]
        == "actual_model_forward"
    )
    assert result["operation_audit"]["pruning_operations"] == 0
    assert result["operation_audit"]["quantization_operations"] == 0
    assert result["operation_audit"]["network_requests"] == 0
    guard_reports = result["operation_audit"]["runtime_fold_only_guard_reports"]
    assert len(guard_reports) == 8
    assert all(row["clean"] is True for row in guard_reports)
    assert all(row["protected_model_count"] == 0 for row in guard_reports)
    assert set(result["checkpoint_reuse"]) == {
        "evidence_preparation",
        "target_materialization",
        "technique_solve",
        "component_install",
        "behavior_evaluation",
    }
    progress = [
        json.loads(line)
        for line in (tmp_path / "cell" / "progress.jsonl").read_text().splitlines()
    ]
    assert "component_solve_completed" in {row["stage"] for row in progress}


def test_executor_rejects_missing_owned_weight_before_install(tmp_path: Path) -> None:
    with pytest.raises(
        Lfm2MatrixIsolationExecutorError,
        match="every and only owned weight",
    ):
        run_matrix_cell(
            _request(extra_owned=True),
            t5_isolation_executor(),
            tmp_path / "missing-owned",
        )


def test_executor_rejects_fabricated_perfect_or_placeholder_metrics(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        Lfm2MatrixIsolationExecutorError,
        match="no authoritative source",
    ):
        run_matrix_cell(
            _request(fabricated_metrics=True),
            t5_isolation_executor(),
            tmp_path / "fabricated",
        )


def test_isolation_executor_cannot_masquerade_as_complete_child(tmp_path: Path) -> None:
    with pytest.raises(
        Lfm2MatrixIsolationExecutorError,
        match="cannot publish complete-child",
    ):
        run_matrix_cell(
            _request(evidence_class="complete_child"),
            t5_isolation_executor(),
            tmp_path / "fake-complete-child",
        )
