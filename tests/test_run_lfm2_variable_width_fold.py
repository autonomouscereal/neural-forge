from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments import run_lfm2_variable_width_fold as runner


def test_real_15_and_20_percent_budgets_are_physically_exact() -> None:
    fifteen = runner.target_parameter_budget(0.15)
    twenty = runner.target_parameter_budget(0.20)

    assert fifteen == {
        "requested_fraction": 0.15,
        "unit_parameter_savings": 3072,
        "removed_neurons": 11215,
        "parameters_saved": 34_452_480,
        "target_unique_parameters": 195_240_704,
        "realized_fraction": pytest.approx(0.14999347999808302),
        "target_total_intermediate_width": 24_625,
    }
    assert twenty == {
        "requested_fraction": 0.20,
        "unit_parameter_savings": 3072,
        "removed_neurons": 14954,
        "parameters_saved": 45_938_688,
        "target_unique_parameters": 183_754_496,
        "realized_fraction": pytest.approx(0.20000022290604844),
        "target_total_intermediate_width": 20_886,
    }


def test_measured_frontier_contains_exact_budget_feasibility_points() -> None:
    for fraction, vector in runner.PROVISIONAL_FEASIBILITY_WIDTHS.items():
        budget = runner.target_parameter_budget(fraction)
        assert len(vector) == 14
        assert sum(vector) == budget["target_total_intermediate_width"]
        for layer_index, width in enumerate(vector):
            assert width in runner.candidate_widths_for_layer(layer_index)

    assert runner.maximum_curve_step() == 256


def test_parser_is_local_only_folding_and_has_no_training_controls() -> None:
    parser = runner.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--parent-model" in options
    assert "--cached-parent-outputs" in options
    assert "--parent-horizon-cache-root" in options
    assert "--publish-strict-passes" in options
    assert "--train" not in options
    assert "--learning-rate" not in options
    assert parser.get_default("target_fractions") == (0.15, 0.20)
    assert parser.get_default("correction_ridges") == (1e-8, 1e-6, 1e-4)


@pytest.mark.parametrize("value", ("", "0.10", "0.15,0.15", "15,20"))
def test_target_fraction_parser_rejects_noncanonical_frontiers(
    value: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        runner._parse_fractions(value)


def test_target_fraction_parser_accepts_either_or_both_real_targets() -> None:
    assert runner._parse_fractions("0.15") == (0.15,)
    assert runner._parse_fractions("0.20") == (0.20,)
    assert runner._parse_fractions("0.15,0.20") == (0.15, 0.20)


def test_target_attempt_is_sealed_and_preserved_independently(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    target = {
        "fraction": 0.15,
        "status": "rejected",
        "checkpoint_published": False,
    }

    path = runner._write_target_attempt(
        tmp_path / "target-15pct",
        target,
        parent_identity_sha256=digest,
        parent_cache_key=digest,
        parent_trace_rows_sha256=digest,
        evidence_sha256=digest,
    )

    document = json.loads(path.read_text(encoding="ascii"))
    seal = document.pop("target_attempt_sha256")
    assert document["schema"] == runner.TARGET_ATTEMPT_SCHEMA
    assert document["target"] == target
    assert seal == runner._json_sha256(document)
