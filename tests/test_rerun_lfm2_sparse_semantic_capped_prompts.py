from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from experiments import rerun_lfm2_sparse_semantic_capped_prompts as runner
from neural_forge.dynamic_circuit_ir import canonical_sha256


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sealed(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: canonical_sha256(body)}


def _plan(tmp_path: Path) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": runner.PLAN_SCHEMA,
        "status": "ready",
        "policy": {
            "source_cap": 256,
            "initial_high_cap": 1024,
            "adaptive_maximum_cap": 2048,
            "safety_is_separate_from_folding_quality": True,
            "cap_hit_is_censored_not_failed": True,
            "rerun_only_source_cap_hits": True,
            "reuse_saved_child_and_routes_only": True,
            "parent_trace_regeneration_forbidden": True,
        },
        "shared": {},
        "lanes": [
            {
                "id": "a9",
                "capped_prompt_ids": ["capped"],
                "output_directory": str(tmp_path / "output"),
            }
        ],
        "operation_contract": {
            "parent_model_load_calls": 0,
            "parent_forward_calls": 0,
            "parent_generate_calls": 0,
            "parent_trace_recaptures": 0,
            "fold_compilations": 0,
            "support_discovery_calls": 0,
            "coefficient_solve_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "network_access": False,
        },
    }
    return {**body, "plan_sha256": canonical_sha256(body)}


def test_plan_authenticates_and_requires_strictly_larger_caps(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    plan = _plan(tmp_path)
    _write_json(path, plan)

    assert runner.load_reuse_plan(path) == plan

    invalid = _plan(tmp_path)
    invalid["policy"]["initial_high_cap"] = 256
    body = dict(invalid)
    body.pop("plan_sha256")
    invalid["plan_sha256"] = canonical_sha256(body)
    _write_json(path, invalid)
    with pytest.raises(runner.Lfm2CappedPromptReuseError, match="Caps must increase"):
        runner.load_reuse_plan(path)


def test_lane_source_must_select_exactly_prior_cap_hits(tmp_path: Path) -> None:
    comparisons = [
        {
            "id": "complete",
            "child_semantic_step_response": {
                "diagnostics": {
                    "ended_with_eos": True,
                    "generation_limit_reached": False,
                }
            },
        },
        {
            "id": "capped",
            "child_semantic_step_response": {
                "diagnostics": {
                    "ended_with_eos": False,
                    "generation_limit_reached": True,
                }
            },
        },
    ]
    body = {
        "maximum_new_tokens": 256,
        "comparisons": comparisons,
    }
    dossier = _sealed(body, "dossier_sha256")
    path = tmp_path / "dossier.json"
    _write_json(path, dossier)
    lane = {
        "source_dossier": str(path),
        "source_dossier_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_dossier_sha256": dossier["dossier_sha256"],
        "source_cap": 256,
        "capped_prompt_ids": ["capped"],
    }

    loaded, rows = runner.authenticate_lane_source(lane)
    assert loaded == dossier
    assert [row["id"] for row in rows] == ["capped"]

    lane["capped_prompt_ids"] = ["complete", "capped"]
    with pytest.raises(runner.Lfm2CappedPromptReuseError, match="not exactly"):
        runner.authenticate_lane_source(lane)


@pytest.mark.parametrize(
    ("lane_id", "report_directory", "expected_count"),
    [
        (
            "a9-k32-residual-omp",
            "sparse-semantic-a9-k32-residual-omp-20260730",
            7,
        ),
        (
            "a11-k32-split-cout",
            "sparse-semantic-a11-k32-split-cout-20260730",
            6,
        ),
    ],
)
def test_production_manifest_selects_exactly_observed_cap_hits(
    lane_id: str,
    report_directory: str,
    expected_count: int,
) -> None:
    root = Path(runner.__file__).resolve().parents[1]
    manifest = runner.load_reuse_plan(
        root / "experiments" / "lfm2_a9_a11_high_cap_reuse_manifest_20260730.json"
    )
    lane = next(row for row in manifest["lanes"] if row["id"] == lane_id)
    dossier_path = root / "reports" / report_directory / "whole-response-dossier.json"
    local_lane = {
        **lane,
        "source_dossier": str(dossier_path),
        "source_dossier_file_sha256": hashlib.sha256(
            dossier_path.read_bytes()
        ).hexdigest(),
    }

    _dossier, rows = runner.authenticate_lane_source(local_lane)

    assert len(rows) == expected_count
    assert [row["id"] for row in rows] == lane["capped_prompt_ids"]


def test_runner_has_no_parent_model_fold_solver_training_or_network_call() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    forbidden_names = {
        "compile_lfm2_sparse_semantic_codebook",
        "discover_lfm2_sparse_semantic_candidates",
        "discover_lfm2_residual_omp_semantic_candidates",
        "refit_lfm2_functional_output_coefficients",
        "load_lfm2_direct_local_model",
    }
    assert called_names.isdisjoint(forbidden_names)
    assert "generate" not in called_attributes
    assert "backward" not in called_attributes
    assert "step" not in called_attributes
    assert "from_pretrained" not in called_attributes
    assert "run_lfm2_semantic_step_model_review" in called_names
    assert "load_lfm2_sparse_semantic_codebook" in called_names
