from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "experiments" / "launch_cross_family_blank_run.py"
SPEC = importlib.util.spec_from_file_location("cross_family_blank_runner_parser", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parser_accepts_target_tokenizer_model() -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--source-model",
            "donor-model",
            "--target-tokenizer-model",
            "target-tokenizer",
            "--run-root",
            "runs",
            "--tag",
            "cross-token",
            "--calibration-file",
            "calibration.jsonl",
            "--eval-suite",
            "eval.json",
            "--cached-parent-eval",
            "parent.json",
            "--hidden-size",
            "8",
            "--intermediate-size",
            "12",
            "--num-hidden-layers",
            "2",
            "--num-attention-heads",
            "2",
            "--num-key-value-heads",
            "1",
            "--head-dim",
            "4",
            "--max-position-embeddings",
            "32",
            "--trace-pooling",
            "mean",
        ]
    )

    assert args.target_tokenizer_model == "target-tokenizer"
    assert args.trace_pooling == "mean"
