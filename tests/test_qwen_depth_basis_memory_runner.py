from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "run_qwen_depth_basis_in_memory.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_depth_basis_memory_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parser_exposes_every_projection_rank_and_full_eval_defaults() -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--source-model",
            "parent",
            "--output-dir",
            "run",
            "--eval-suite",
            "suite.json",
            "--cached-parent-eval",
            "parent.json",
            "--k-rank",
            "6",
            "--v-rank",
            "6",
            "--local-files-only",
        ]
    )

    assert args.q_rank == 1
    assert args.k_rank == 6
    assert args.v_rank == 6
    assert args.o_rank == 1
    assert args.gate_rank == 1
    assert args.up_rank == 1
    assert args.down_rank == 1
    assert args.max_behavior_probes == 1000
    assert args.max_generation_prompts == 1000
    assert args.max_choice_probes == 250
    assert args.local_files_only is True
