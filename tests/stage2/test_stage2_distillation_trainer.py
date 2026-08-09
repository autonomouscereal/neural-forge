from __future__ import annotations

import ast
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Any

import pytest


pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("safetensors")
torch = pytest.importorskip("torch")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "stage2_cpu_toy_validation_trainer",
    REPOSITORY_ROOT / "experiments" / "stage2_cpu_toy_validation.py",
)
assert SPEC is not None and SPEC.loader is not None
TOY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOY
SPEC.loader.exec_module(TOY)

from neural_forge.stage2_distillation_trainer import (  # noqa: E402
    STAGE2_ADAM_BETAS,
    STAGE2_BATCH_SEQUENCES,
    STAGE2_DESIGN_STATED_RUNG_STEPS,
    STAGE2_GRADIENT_CLIP_NORM,
    STAGE2_LEARNING_RATE,
    STAGE2_MINIMUM_LEARNING_RATE,
    STAGE2_SEQUENCE_LENGTH,
    STAGE2_WARMUP_STEPS,
    STAGE2_WEIGHT_DECAY,
    Stage2TrainerConfig,
    Stage2TrainerError,
)


# Disqualified proxy classes.  Ceiling report section 2.3 (terminal
# reconstruction swung 0.507 with no behavioural consequence) and section 4.3
# (teacher-forced top-1 moved *against* free-run across four children).  They
# may appear as an explicit "not used" declaration in a receipt; they may not
# appear as anything this code computes.
_DISQUALIFIED_PROXY = re.compile(
    r"teacher[_-]?forced|terminal[_-]?reconstruct|relative[_-]?rmse"
    r"|rel[_-]?l2|top[_-]?1",
    re.IGNORECASE,
)
_STAGE2_SOURCES = tuple(
    sorted(
        (REPOSITORY_ROOT / "src" / "neural_forge").glob("stage2_*.py"),
        key=lambda path: path.name,
    )
) + tuple(
    sorted(
        (REPOSITORY_ROOT / "experiments").glob("stage2_*.py"),
        key=lambda path: path.name,
    )
)


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One short CPU toy run, reused by every assertion in this module."""

    return TOY.run_toy_validation(
        tmp_path_factory.mktemp("stage2-toy-run"),
        steps=36,
        switch_steps=12,
        train_routes=False,
    )


def test_optimizer_and_batch_constants_match_section_three_point_one() -> None:
    assert STAGE2_BATCH_SEQUENCES == 16
    assert STAGE2_SEQUENCE_LENGTH == 1_024
    assert STAGE2_LEARNING_RATE == 3e-4
    assert STAGE2_MINIMUM_LEARNING_RATE == 3e-5
    assert STAGE2_WARMUP_STEPS == 100
    assert STAGE2_ADAM_BETAS == (0.9, 0.95)
    assert STAGE2_WEIGHT_DECAY == 0.1
    assert STAGE2_GRADIENT_CLIP_NORM == 1.0


def test_the_ladder_matches_the_design_except_for_one_rounding_step() -> None:
    """Design section 3.1 states 610 / 6,104 / 61,036.

    The token budgets are the authority (section 5.2), and 10,000,000 is not a
    multiple of 16,384: 610 steps consume 9,994,240 tokens, which is *under*
    the rung.  Rounding up reproduces the other two rungs exactly and moves the
    10M rung by one step; the deviation is pinned here rather than hidden.
    """

    config = Stage2TrainerConfig(arm_id="S2-B", seed=1)
    assert config.tokens_per_step == 16_384
    assert [config.steps_for_tokens(value) for value in config.rung_tokens] == [
        611,
        6_104,
        61_036,
    ]
    assert STAGE2_DESIGN_STATED_RUNG_STEPS == (610, 6_104, 61_036)
    assert 610 * config.tokens_per_step < 10_000_000 <= 611 * config.tokens_per_step
    assert config.total_steps == 61_036


def test_the_learning_rate_schedule_warms_up_then_cosines_to_the_floor() -> None:
    config = Stage2TrainerConfig(arm_id="S2-B", seed=1)
    assert config.learning_rate_at(0) == pytest.approx(3e-4 / 100)
    assert config.learning_rate_at(99) == pytest.approx(3e-4)
    assert config.learning_rate_at(100) == pytest.approx(3e-4)
    assert config.learning_rate_at(config.total_steps - 1) == pytest.approx(
        3e-5,
        rel=1e-3,
    )
    midpoint = config.learning_rate_at((config.total_steps + 100) // 2)
    assert 3e-5 < midpoint < 3e-4


def test_gradient_accumulation_must_divide_the_batch() -> None:
    with pytest.raises(Stage2TrainerError, match="divide batch_sequences"):
        Stage2TrainerConfig(arm_id="S2-B", seed=1, micro_batch_sequences=7)
    config = Stage2TrainerConfig(arm_id="S2-B", seed=1, micro_batch_sequences=2)
    assert config.accumulation_steps == 8


def test_toy_run_loss_decreases(toy_run: dict[str, Any]) -> None:
    assert toy_run["loss_decreased"] is True
    assert toy_run["last_window_mean_loss"] < toy_run["first_window_mean_loss"]


def test_toy_run_reports_the_two_objective_branches_separately(
    toy_run: dict[str, Any],
) -> None:
    """Forward KL is unbounded, the JSD at beta=0.5 is not; one curve lies."""

    per_objective = toy_run["loss_by_objective"]
    assert per_objective["forward-kl"]["points"] > 0
    assert per_objective["generalized-jsd-gkd"]["points"] > 0
    assert per_objective["forward-kl"]["decreased"] is True
    assert per_objective["generalized-jsd-gkd"]["last"] < math.log(2.0)


def test_toy_run_switches_phase_at_the_token_threshold(
    toy_run: dict[str, Any],
) -> None:
    tokens_per_step = TOY.TOY_BATCH_SEQUENCES * TOY.TOY_SEQUENCE_LENGTH
    assert toy_run["phase_switch_tokens"] == tokens_per_step * 12
    assert toy_run["phase_switch_step"] == 12
    assert toy_run["off_policy_steps"] + toy_run["on_policy_steps"] == (
        toy_run["optimizer_steps"]
    )
    assert toy_run["post_switch_on_policy_fraction"] == pytest.approx(0.5)
    assert toy_run["on_policy_steps"] == toy_run["post_switch_steps"] // 2


def test_toy_run_token_accounting_is_exact(toy_run: dict[str, Any]) -> None:
    tokens_per_step = TOY.TOY_BATCH_SEQUENCES * TOY.TOY_SEQUENCE_LENGTH
    assert toy_run["tokens_consumed"] == toy_run["optimizer_steps"] * tokens_per_step
    for rung in toy_run["rungs"]:
        assert rung["tokens_consumed"] == rung["optimizer_steps"] * tokens_per_step
        assert rung["tokens_consumed"] >= rung["rung_tokens"]


def test_toy_run_seals_reloads_and_refuses_a_tampered_receipt(
    toy_run: dict[str, Any],
) -> None:
    assert toy_run["receipt_reload_matches"] is True
    assert toy_run["receipt_tamper_refused"]["refused"] is True
    assert toy_run["corpus_manifest_check"]["corpus_manifest_verified"] is True


def test_toy_run_probes_and_checkpoints_every_rung(
    toy_run: dict[str, Any],
) -> None:
    assert len(toy_run["rungs"]) == 3
    assert toy_run["free_run_probe_points"] > 0
    for rung in toy_run["rungs"]:
        assert rung["checkpoint_manifest_sha256"] is not None
        assert rung["free_run_token_match_fraction"] is not None
        assert 0.0 <= rung["free_run_token_match_fraction"] <= 1.0
        assert rung["free_run_mean_lcp"] >= 0.0


def test_toy_run_checkpoints_round_trip_through_the_review_loader(
    toy_run: dict[str, Any],
) -> None:
    assert toy_run["checkpoint_round_trips"]
    for result in toy_run["checkpoint_round_trips"]:
        assert result["loaded_through_review_loader"] is True
        assert result["refused_by_stage1_guard"] is False
        assert result["semantic_step_generate_installed"] is True


def test_toy_run_verifies_the_route_surface_before_it_trains(
    toy_run: dict[str, Any],
) -> None:
    equivalence = toy_run["route_surface_equivalence"]
    assert equivalence["input_route_bitwise_identical_to_frozen_runtime"] is True
    assert equivalence["output_route_bitwise_identical_to_frozen_runtime"] is True
    assert toy_run["route_training_mask"]["protected_anchors_intact"] is True


def test_a_random_init_control_trains_through_the_same_trainer(
    tmp_path: Path,
) -> None:
    control = TOY.run_toy_validation(
        tmp_path / "control",
        steps=16,
        switch_steps=8,
        mode="random-init",
        train_routes=False,
    )
    assert control["mode"] == "random-init"
    assert control["loss_decreased"] is True
    assert control["optimizer_steps"] == 16
    assert all(
        result["loaded_through_review_loader"] is True
        for result in control["checkpoint_round_trips"]
    )


def test_no_stage2_module_defines_a_disqualified_proxy_metric() -> None:
    """The disqualified metric classes cannot be computed anywhere in stage 2."""

    assert len(_STAGE2_SOURCES) >= 6
    offenders: list[str] = []
    for path in _STAGE2_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
                names.extend(
                    argument.arg
                    for argument in (*node.args.args, *node.args.kwonlyargs)
                )
            elif isinstance(node, ast.ClassDef):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            offenders.extend(
                f"{path.name}:{name}"
                for name in names
                if _DISQUALIFIED_PROXY.search(name)
            )
    assert offenders == []


def test_every_disqualified_key_in_a_receipt_is_an_explicit_denial(
    toy_run: dict[str, Any],
) -> None:
    """Where the names do appear, they appear only as ``... _used: false``."""

    found: list[tuple[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and _DISQUALIFIED_PROXY.search(key):
                    found.append((key, item))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(toy_run)
    assert found, "the receipt should still declare the disqualified classes"
    for key, item in found:
        assert key.endswith("_used"), key
        assert item is False, key
