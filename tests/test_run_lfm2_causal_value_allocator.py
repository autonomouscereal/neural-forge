from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from experiments import finalize_lfm2_causal_value_allocator as finalizer
from experiments import run_lfm2_causal_value_allocator as runner


def _intelligence(*, failed: str | None = None) -> dict:
    required = [
        "task_correctness",
        "parent_capability",
        "nonempty_retention",
    ]
    return {
        "policy": {"required_components": required},
        "core_intelligence_gate": {
            "manual_review": {
                "status": "rejected",
                "passed": False,
                "case_pass_rate": 0.0,
            },
            "components": [
                {
                    "name": name,
                    "retention": 0.95,
                    "passed": name != failed,
                }
                for name in required
            ],
        },
        "parent_relative_task_retention": {
            "absolute_accuracy_retention": 0.93,
        },
    }


def test_decision_uses_parent_relative_manual_not_absolute_manual_failure() -> None:
    manual = {
        "passed": True,
        "retention": 1.0,
        "clear_regressions": [],
    }
    result = runner._parent_relative_decision(_intelligence(), manual)
    assert result["promoted"] is True
    assert result["status"] == "promoted"
    assert result["certified_parent_relative_lower_bound"] == pytest.approx(0.93)
    assert result["absolute_manual_scores_are_diagnostics_only"] is True
    assert result["abilities_absent_in_parent_are_neutralized"] is True


def test_decision_withholds_pending_and_rejects_any_clear_manual_regression() -> None:
    pending = runner._parent_relative_decision(_intelligence(), None)
    assert pending["promoted"] is False
    assert pending["status"] == "pending_parent_relative_manual"
    rejected = runner._parent_relative_decision(
        _intelligence(),
        {
            "passed": False,
            "retention": 0.95,
            "clear_regressions": ["manual_05:instruction_following"],
        },
    )
    assert rejected["promoted"] is False
    assert rejected["status"] == "rejected"
    assert (
        "parent_relative_manual:threshold_or_regression_failed" in rejected["reasons"]
    )
    automatic = runner._parent_relative_decision(
        _intelligence(failed="parent_capability"),
        {"passed": True, "retention": 1.0},
    )
    assert automatic["promoted"] is False
    assert "parent_capability:threshold_failed" in automatic["reasons"]


def test_config_requires_sealed_local_frontier_and_paired_optional_inputs(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    tokenizer = tmp_path / "tokenizer"
    parent.mkdir()
    tokenizer.mkdir()
    frontier = tmp_path / "frontier.json"
    envelope = tmp_path / "envelope.json"
    frontier.write_text("{}", encoding="utf-8")
    envelope.write_text("{}", encoding="utf-8")
    digest = "0" * 64
    with pytest.raises(
        runner.Lfm2CausalValueRunnerError,
        match="must be supplied together",
    ):
        runner.CausalValueRunConfig(
            parent_model_directory=parent,
            tokenizer_directory=tokenizer,
            output_root=tmp_path / "output",
            cached_causal_frontier=frontier,
            expected_cached_causal_frontier_file_sha256=digest,
            parent_envelope=envelope,
            expected_parent_envelope_file_sha256=digest,
            expected_parent_envelope_sha256=digest,
            expected_manifest_sha256=digest,
            expected_parent_state_fingerprint=digest,
            expected_parent_model_fingerprint=digest,
            expected_tokenizer_fingerprint=digest,
            calibration_token_selection=frontier,
        )


@pytest.mark.parametrize("module", [runner, finalizer])
def test_runners_have_no_hub_loader_optimizer_backward_or_training_call(module) -> None:
    tree = ast.parse(inspect.getsource(module))
    forbidden = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"from_pretrained", "backward", "step"}
    }
    assert forbidden == set()
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("optim" in value.lower() for value in imports)
