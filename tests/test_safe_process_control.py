from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "experiments" / "safe_process_control.py"
SPEC = importlib.util.spec_from_file_location("safe_process_control", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
safe_process_control = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = safe_process_control
SPEC.loader.exec_module(safe_process_control)


def test_tag_matching_rejects_self_matching_shell() -> None:
    rows = safe_process_control.parse_ps_rows(
        """
        10 1 Ss bash bash -c PATTERN='hf-xfamily-chain-anchor-compress tag_a'; kill -STOP $(pgrep -f "$PATTERN")
        20 1 Rl python python -m neural_forge.cli hf-xfamily-chain-anchor-compress Qwen/Qwen3-4B-Base /runs/tag_a --stages init
        """
    )

    selected = safe_process_control.select_processes(
        rows,
        pids=set(),
        tag="tag_a",
        expect_contains=(),
        protected_pids=set(),
    )

    assert [row.pid for row in selected] == [20]


def test_protected_current_or_ancestor_pids_are_excluded() -> None:
    rows = safe_process_control.parse_ps_rows(
        """
        10 1 Rl python python -m neural_forge.cli hf-xfamily-chain-anchor-compress Qwen/Qwen3-4B-Base /runs/tag_a
        20 10 Rl python python -m neural_forge.cli hf-xfamily-chain-anchor-compress Qwen/Qwen3-4B-Base /runs/tag_a
        30 1 Rl python python -m neural_forge.cli hf-xfamily-chain-anchor-compress Qwen/Qwen3-4B-Base /runs/tag_a
        """
    )

    selected = safe_process_control.select_processes(
        rows,
        pids=set(),
        tag="tag_a",
        expect_contains=(),
        protected_pids={10},
    )

    assert [row.pid for row in selected] == [30]


def test_exact_pid_matching_can_require_expected_command_fragment() -> None:
    rows = safe_process_control.parse_ps_rows(
        """
        40 1 Sl python python unrelated.py
        41 1 Rl python python -m neural_forge.cli hf-xfamily-chain-anchor-compress donor /runs/tag_b
        """
    )

    selected = safe_process_control.select_processes(
        rows,
        pids={40, 41},
        tag=None,
        expect_contains=("tag_b",),
        protected_pids=set(),
    )

    assert [row.pid for row in selected] == [41]
