from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "experiments" / "launch_cross_family_blank_run.py"
SPEC = importlib.util.spec_from_file_location(
    "cross_family_blank_runner_parser", SCRIPT
)
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
    assert args.nonlinear_merge_method == "activation_fold"


def test_parser_accepts_dual_orthonormal_controls() -> None:
    args = MODULE.build_parser().parse_args(
        [
            "--source-model",
            "donor-model",
            "--run-root",
            "runs",
            "--tag",
            "dual",
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
            "--residual-map-method",
            "dual_orthonormal",
            "--nonlinear-merge-method",
            "real_medoid",
            "--dual-alpha",
            "0.5",
            "--dual-beta",
            "1.5",
            "--dual-gamma",
            "3.0",
            "--dual-margin-topk",
            "7",
            "--dual-oracle-vocab-rows",
            "128",
        ]
    )

    assert args.residual_map_method == "dual_orthonormal"
    assert args.nonlinear_merge_method == "real_medoid"
    assert args.dual_alpha == 0.5
    assert args.dual_beta == 1.5
    assert args.dual_gamma == 3.0
    assert args.dual_margin_topk == 7
    assert args.dual_oracle_vocab_rows == 128
