from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "experiments" / "run_qwen_basis_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("qwen_basis_checkpoint_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parser_defaults_to_near_cap_full_topology_plan() -> None:
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
            "--local-files-only",
        ]
    )

    assert args.parameter_budget == 134_445_568
    assert (args.q_rank, args.k_rank, args.v_rank, args.o_rank) == (1, 6, 6, 1)
    assert (args.gate_rank, args.up_rank, args.down_rank) == (1, 1, 1)
    assert args.vocab_rank == 46
    assert args.vocab_algorithm == "randomized"
    assert args.max_behavior_probes == 1000
    assert args.max_generation_prompts == 1000
    assert args.max_choice_probes == 250
