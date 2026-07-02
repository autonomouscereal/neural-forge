"""Command-line interface for Neural Forge."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Sequence

from neural_forge.checkpoint import load_model, random_toy_model, save_model
from neural_forge.surgery import (
    expand_depth_identity,
    merge_models,
    parse_widths,
    plan_widths_for_target_params,
    resize_to_widths,
)

RESIDUAL_MAGNITUDE_COMPENSATION_CHOICES = [
    "none",
    "pre",
    "pre-linear",
    "linear-pre",
    "pre-lowrank",
    "lowrank-pre",
    "pre-eora",
    "eora-pre",
    "pre-module-linear",
    "module-linear-pre",
    "pre-module-lowrank",
    "module-lowrank-pre",
    "pre-module-eora",
    "module-eora-pre",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-forge")
    sub = parser.add_subparsers(required=True)

    toy = sub.add_parser("toy", help="Create a random dense toy checkpoint.")
    toy.add_argument("--output", required=True)
    toy.add_argument("--widths", required=True, help="Comma-separated full widths, e.g. 8,16,4.")
    toy.add_argument("--seed", type=int, default=0)
    toy.add_argument("--activation", default="relu")
    toy.set_defaults(func=cmd_toy)

    inspect = sub.add_parser("inspect", help="Inspect a Neural Forge checkpoint.")
    inspect.add_argument("model")
    inspect.set_defaults(func=cmd_inspect)

    shrink = sub.add_parser("shrink", help="Shrink hidden widths using neuron merging.")
    shrink.add_argument("input")
    shrink.add_argument("output")
    shrink.add_argument("--target-params", type=int)
    shrink.add_argument("--widths", help="Comma-separated hidden widths.")
    shrink.add_argument("--min-width", type=int, default=1)
    shrink.set_defaults(func=cmd_shrink)

    expand = sub.add_parser("expand", help="Expand hidden width and/or depth.")
    expand.add_argument("input")
    expand.add_argument("output")
    expand.add_argument("--target-params", type=int)
    expand.add_argument("--widths", help="Comma-separated hidden widths.")
    expand.add_argument("--add-layers", type=int, default=0)
    expand.add_argument("--noise-std", type=float, default=0.0)
    expand.add_argument("--seed", type=int, default=0)
    expand.set_defaults(func=cmd_expand)

    merge = sub.add_parser("merge", help="Merge two or more dense checkpoints into a target shape.")
    merge.add_argument("output")
    merge.add_argument("--models", nargs="+", required=True)
    merge.add_argument("--target-params", type=int)
    merge.add_argument("--widths", help="Comma-separated hidden widths.")
    merge.add_argument("--weights", help="Comma-separated model weights.")
    merge.add_argument("--method", choices=["average", "slerp"], default="average")
    merge.add_argument("--no-align", action="store_true")
    merge.add_argument("--noise-std", type=float, default=0.0)
    merge.add_argument("--seed", type=int, default=0)
    merge.set_defaults(func=cmd_merge)

    hf_merge = sub.add_parser("hf-merge", help="Merge same-shape Hugging Face causal LM checkpoints.")
    hf_merge.add_argument("output")
    hf_merge.add_argument("--models", nargs="+", required=True)
    hf_merge.add_argument("--weights", help="Comma-separated model weights.")
    hf_merge.add_argument("--dtype", default="float16")
    hf_merge.add_argument("--trust-remote-code", action="store_true")
    hf_merge.set_defaults(func=cmd_hf_merge)

    hf_task_merge = sub.add_parser("hf-task-merge", help="Create base + alpha * (tuned - base) for same-shape HF causal LMs.")
    hf_task_merge.add_argument("output")
    hf_task_merge.add_argument("--base-model", required=True)
    hf_task_merge.add_argument("--tuned-model", required=True)
    hf_task_merge.add_argument("--alpha", type=float, default=1.0)
    hf_task_merge.add_argument("--dtype", default="auto")
    hf_task_merge.add_argument("--trust-remote-code", action="store_true")
    hf_task_merge.set_defaults(func=cmd_hf_task_merge)

    hf_random = sub.add_parser("hf-random-init", help="Create an untrained HF causal LM from an existing config/tokenizer.")
    hf_random.add_argument("output")
    hf_random.add_argument("--config-model", required=True)
    hf_random.add_argument("--tokenizer-model")
    hf_random.add_argument("--seed", type=int, default=0)
    hf_random.add_argument("--dtype", default="float16")
    hf_random.add_argument("--trust-remote-code", action="store_true")
    hf_random.set_defaults(func=cmd_hf_random_init)

    hf_donor_init = sub.add_parser(
        "hf-donor-project-init",
        help="Initialize a target HF causal LM by folding donor tensors into the target shape.",
    )
    hf_donor_init.add_argument("output")
    hf_donor_init.add_argument("--target-config-model", required=True)
    hf_donor_init.add_argument("--target-base-model")
    hf_donor_init.add_argument("--target-tokenizer-model")
    hf_donor_init.add_argument("--donor-model", required=True)
    hf_donor_init.add_argument(
        "--donor-weights",
        help=(
            "Optional comma-separated weights when --donor-model contains multiple comma-separated donors. "
            "The donors are folded sequentially into the same target shape with weighted blend alphas."
        ),
    )
    hf_donor_init.add_argument("--seed", type=int, default=0)
    hf_donor_init.add_argument("--ridge", type=float, default=1e-3)
    hf_donor_init.add_argument("--max-anchors", type=int, default=50_000)
    hf_donor_init.add_argument("--fold-mode", choices=["mean", "rms", "transport"], default="mean")
    hf_donor_init.add_argument(
        "--hidden-basis-mode",
        choices=["none", "activation-pca", "shared-token-ridge"],
        default="none",
        help=(
            "Optionally project donor hidden axes through an activation PCA basis or a shared-token ridge "
            "map before ordinary folding."
        ),
    )
    hf_donor_init.add_argument(
        "--hidden-basis-scope",
        choices=["global", "layer"],
        default="global",
        help="Use one donor hidden basis globally or boundary-specific bases for each folded target layer.",
    )
    hf_donor_init.add_argument(
        "--hidden-basis-max-rows",
        type=int,
        default=2048,
        help="Maximum donor hidden activation rows to retain when fitting the activation PCA basis.",
    )
    hf_donor_init.add_argument(
        "--hidden-basis-preprocess",
        choices=["centered-rowrms", "uncentered-rowrms", "uncentered"],
        default="centered-rowrms",
        help="Preprocessing for activation PCA hidden bases.",
    )
    hf_donor_init.add_argument(
        "--activation-fold-assignment",
        choices=["balanced", "similarity", "signed-similarity"],
        default="balanced",
        help=(
            "How activation features assign source axes to folded target axes. signed-similarity groups opposite "
            "activation directions by absolute similarity and applies the recovered sign during folding."
        ),
    )
    hf_donor_init.add_argument(
        "--fold-stat-repair",
        choices=["none", "rms"],
        default="none",
        help=(
            "Optional per-fold RMS repair after many donor axes collapse into one target axis. "
            "rms rescales each folded target axis toward the donor axes that fed it."
        ),
    )
    hf_donor_init.add_argument(
        "--transport-backend",
        choices=["sparse", "sinkhorn"],
        default="sparse",
        help="Backend for fold-mode=transport: sparse centroid routing or dense Sinkhorn transport.",
    )
    hf_donor_init.add_argument(
        "--transport-support",
        choices=["centroid", "importance", "quantile", "medoid", "quantile-medoid"],
        default="centroid",
        help=(
            "Support set for activation transport. centroid preserves the existing clustered-centroid map; "
            "importance routes all source axes onto the highest activation-weight survivor axes; "
            "quantile anchors transport on activation-weight strata; "
            "medoid uses real donor axes nearest to activation clusters; "
            "quantile-medoid picks real donor axes within activation-weight strata."
        ),
    )
    hf_donor_init.add_argument(
        "--donor-layer-fold-mode",
        choices=["representative", "average", "fold"],
        default="representative",
        help="How to fold parent/donor depth spans into each child/target layer.",
    )
    hf_donor_init.add_argument(
        "--donor-layer-group-strategy",
        choices=["depth", "activation-energy", "activation-geometry", "activation-cka"],
        default="depth",
        help=(
            "How donor layers are grouped before depth folding. activation-energy partitions donor layers by "
            "calibrated residual update energy instead of uniform depth bins; activation-geometry also weights "
            "CKA/cosine shift and residual novelty; activation-cka clusters donor residual-update directions "
            "by pairwise activation CKA so similar non-contiguous layers merge before folding."
        ),
    )
    hf_donor_init.add_argument(
        "--attention-fold-mode",
        choices=["independent", "coupled", "activation-coupled"],
        default="independent",
        help="Fold attention Q/K/V/O projection head partitions together; activation-coupled also routes donor heads by calibration features.",
    )
    hf_donor_init.add_argument(
        "--attention-head-dim-transport",
        choices=["none", "qk-vo"],
        default="none",
        help="Use activation transport for per-head dimensions, sharing one map for Q/K and one for V/O.",
    )
    hf_donor_init.add_argument(
        "--neutralize-unmatched-operators",
        action="store_true",
        help=(
            "For target operator tensors with no donor counterpart, initialize safe residual no-ops where supported "
            "(currently LFM-style short-conv output projections)."
        ),
    )
    hf_donor_init.add_argument(
        "--token-init-mode",
        choices=["shared", "donor-tokenized", "donor-tokenized-weighted", "donor-omp"],
        default="shared",
        help=(
            "Initialize only shared decoded-token rows, fill target vocab rows by donor tokenizing each target token, "
            "weight donor-tokenized pieces by decoded span length, or use sparse OMP token reconstruction."
        ),
    )
    hf_donor_init.add_argument(
        "--projection-scale",
        choices=["none", "target-rms", "source-rms", "geomean-rms"],
        default="none",
        help=(
            "Optional RMS rescale for folded parent tensors before loading into the child shape. "
            "target-rms matches the initialized child tensor; source-rms preserves donor tensor scale; "
            "geomean-rms uses the geometric mean of both."
        ),
    )
    hf_donor_init.add_argument(
        "--projection-blend-alpha",
        type=float,
        default=1.0,
        help="Blend folded parent tensors with the target child tensors after projection; 1.0 fully replaces target.",
    )
    hf_donor_init.add_argument(
        "--consumer-hidden-reduction",
        choices=["mean", "sum", "sqrt-sum"],
        default="mean",
        help="How to fold consumer weights over merged hidden axes; mean is safest for arbitrary blank targets.",
    )
    hf_donor_init.add_argument(
        "--activation-fit-linears",
        action="store_true",
        help="Fit target-shaped linear weights from folded donor input/output activations after tensor folding.",
    )
    hf_donor_init.add_argument(
        "--activation-fit-target",
        choices=["all", "attention", "attention-output", "mlp", "mlp-output", "input-projections"],
        default="all",
        help="Limit activation-fitted linear replacements to safer module families.",
    )
    hf_donor_init.add_argument("--activation-fit-max-rows", type=int, default=0)
    hf_donor_init.add_argument("--activation-fit-alpha", type=float, default=1.0)
    hf_donor_init.add_argument("--activation-fit-ridge", type=float, default=1e-1)
    hf_donor_init.add_argument(
        "--activation-fit-rank",
        type=int,
        default=0,
        help="Optional reduced-rank truncation for activation-fitted linears; 0 disables truncation.",
    )
    hf_donor_init.add_argument(
        "--activation-fit-rank-mode",
        choices=["absolute", "delta"],
        default="absolute",
        help="Whether reduced-rank activation fitting truncates the fitted absolute weight or only its delta from the folded reference.",
    )
    hf_donor_init.add_argument(
        "--activation-fit-method",
        choices=["ridge", "lrc"],
        default="ridge",
        help="Use the historical ridge/SVD fit or train a gated low-rank clone delta and materialize it into the folded weight.",
    )
    hf_donor_init.add_argument("--activation-fit-factor-steps", type=int, default=0)
    hf_donor_init.add_argument("--activation-fit-factor-lr", type=float, default=3e-2)
    hf_donor_init.add_argument("--activation-fit-factor-l2", type=float, default=0.0)
    hf_donor_init.add_argument(
        "--activation-fit-gate",
        choices=["none", "validation"],
        default="validation",
        help="Reject activation-fitted linears that do not beat the folded reference on held-out activation rows.",
    )
    hf_donor_init.add_argument("--activation-fit-validation-fraction", type=float, default=0.20)
    hf_donor_init.add_argument("--activation-fit-min-validation-rows", type=int, default=8)
    hf_donor_init.add_argument("--activation-fit-min-relative-improvement", type=float, default=0.0)
    hf_donor_init.add_argument("--activation-fit-max-cosine-drop", type=float, default=0.01)
    hf_donor_init.add_argument(
        "--activation-feature-mode",
        choices=["stats", "subspace", "covariance"],
        default="stats",
        help="Activation feature summary used for transport/similarity assignment on folded axes.",
    )
    hf_donor_init.add_argument(
        "--activation-subspace-rank",
        type=int,
        default=8,
        help="Extra per-channel PCA/SVD signature dimensions when --activation-feature-mode subspace or covariance is used.",
    )
    hf_donor_init.add_argument(
        "--activation-domain-balance",
        choices=["none", "keyword"],
        default="none",
        help="Optionally balance activation transport summaries across inferred calibration domains.",
    )
    hf_donor_init.add_argument(
        "--untie-output-head",
        action="store_true",
        help="Keep a separate donor-projected lm_head instead of tying it back to input embeddings.",
    )
    hf_donor_init.add_argument("--activation-calibration-file")
    hf_donor_init.add_argument("--activation-max-calibration-texts", type=int)
    hf_donor_init.add_argument("--activation-max-length", type=int, default=128)
    hf_donor_init.add_argument("--activation-device", default="cuda:0")
    hf_donor_init.add_argument(
        "--post-init-eval-suite",
        help="Optional eval suite to run immediately after donor-projected initialization.",
    )
    hf_donor_init.add_argument(
        "--post-init-eval-output",
        help="Optional output JSON path for the immediate post-init eval report.",
    )
    hf_donor_init.add_argument("--post-init-max-text-evals", type=int)
    hf_donor_init.add_argument("--post-init-max-choice-probes", type=int)
    hf_donor_init.add_argument("--post-init-max-generation-prompts", type=int)
    hf_donor_init.add_argument("--post-init-max-behavior-probes", type=int)
    hf_donor_init.add_argument("--post-init-max-new-tokens", type=int, default=96)
    hf_donor_init.add_argument("--post-init-eval-device", default="cuda:0")
    hf_donor_init.add_argument("--post-init-eval-dtype")
    hf_donor_init.add_argument("--post-init-min-behavior-pass-rate", type=float)
    hf_donor_init.add_argument("--post-init-min-generation-pass-rate", type=float)
    hf_donor_init.add_argument("--post-init-min-choice-accuracy", type=float)
    hf_donor_init.add_argument("--post-init-max-ppl", type=float)
    hf_donor_init.add_argument("--dtype", default="float16")
    hf_donor_init.add_argument("--trust-remote-code", action="store_true")
    hf_donor_init.set_defaults(func=cmd_hf_donor_project_init)

    hf_resize = sub.add_parser("hf-resize-ffn", help="Shrink or expand gated FFN intermediate size in HF decoder models.")
    hf_resize.add_argument("model")
    hf_resize.add_argument("output")
    hf_resize.add_argument("--ratio", type=float)
    hf_resize.add_argument("--target-intermediate-size", type=int)
    hf_resize.add_argument("--dtype", default="float16")
    hf_resize.add_argument("--trust-remote-code", action="store_true")
    hf_resize.add_argument(
        "--assignment",
        choices=["combined-cosine", "down-cosine", "round-robin", "topk"],
        default="combined-cosine",
    )
    hf_resize.set_defaults(func=cmd_hf_resize)

    hf_repair = sub.add_parser("hf-repair-ffn", help="Ridge-repair a resized gated FFN model against a source model.")
    hf_repair.add_argument("source_model")
    hf_repair.add_argument("resized_model")
    hf_repair.add_argument("output")
    hf_repair.add_argument("--dtype", default="float16")
    hf_repair.add_argument("--device", default="cuda:0")
    hf_repair.add_argument("--trust-remote-code", action="store_true")
    hf_repair.add_argument("--ridge", type=float, default=1e-3)
    hf_repair.add_argument("--max-length", type=int, default=128)
    hf_repair.add_argument("--max-tokens-per-layer", type=int, default=512)
    hf_repair.set_defaults(func=cmd_hf_repair)

    hf_merge_layers = sub.add_parser("hf-merge-layers", help="Merge/fold residual decoder layer groups before reducing depth.")
    hf_merge_layers.add_argument("model")
    hf_merge_layers.add_argument("output")
    hf_merge_layers.add_argument("--groups", required=True, help="Comma-separated groups, e.g. 14+15,20+21.")
    hf_merge_layers.add_argument("--method", choices=["fold", "average"], default="fold")
    hf_merge_layers.add_argument("--dtype", default="float16")
    hf_merge_layers.add_argument("--trust-remote-code", action="store_true")
    hf_merge_layers.set_defaults(func=cmd_hf_merge_layers)

    hf_rank_folds = sub.add_parser(
        "hf-rank-layer-folds",
        help="Rank adjacent residual layer fold candidates by activation CKA/cosine redundancy.",
    )
    hf_rank_folds.add_argument("model")
    hf_rank_folds.add_argument("output")
    hf_rank_folds.add_argument("--window", type=int, default=2)
    hf_rank_folds.add_argument("--top-k", type=int, default=10)
    hf_rank_folds.add_argument("--max-length", type=int, default=128)
    hf_rank_folds.add_argument("--max-tokens-per-layer", type=int, default=512)
    hf_rank_folds.add_argument("--dtype", default="float16")
    hf_rank_folds.add_argument("--device", default="cuda:0")
    hf_rank_folds.add_argument("--trust-remote-code", action="store_true")
    hf_rank_folds.add_argument("--calibration-file")
    hf_rank_folds.add_argument("--max-calibration-texts", type=int)
    hf_rank_folds.set_defaults(func=cmd_hf_rank_layer_folds)

    hf_drop_layers = sub.add_parser("hf-drop-layers", help="Drop selected residual decoder layers from an HF causal LM.")
    hf_drop_layers.add_argument("model")
    hf_drop_layers.add_argument("output")
    hf_drop_layers.add_argument("--layers", required=True, help="Comma-separated zero-based layer indices to drop.")
    hf_drop_layers.add_argument("--dtype", default="float16")
    hf_drop_layers.add_argument("--trust-remote-code", action="store_true")
    hf_drop_layers.set_defaults(func=cmd_hf_drop_layers)

    hf_eval = sub.add_parser("hf-eval", help="Run lightweight behavior probes for an HF causal LM.")
    hf_eval.add_argument("model")
    hf_eval.add_argument("output")
    hf_eval.add_argument("--dtype", default="float16")
    hf_eval.add_argument("--device", default="cuda:0")
    hf_eval.add_argument("--trust-remote-code", action="store_true")
    hf_eval.add_argument("--max-new-tokens", type=int, default=96)
    hf_eval.add_argument("--eval-suite", help="JSON/JSONL eval suite with text, choice, generation, and behavior probes.")
    hf_eval.add_argument("--max-text-evals", type=int)
    hf_eval.add_argument("--max-choice-probes", type=int)
    hf_eval.add_argument("--max-generation-prompts", type=int)
    hf_eval.add_argument("--max-behavior-probes", type=int)
    hf_eval.set_defaults(func=cmd_hf_eval)

    hf_build_eval_suite = sub.add_parser(
        "hf-build-eval-suite",
        help="Create a deterministic broad HF text eval suite for behavior-first gates.",
    )
    hf_build_eval_suite.add_argument("output")
    hf_build_eval_suite.add_argument("--behavior-count", type=int, default=1000)
    hf_build_eval_suite.add_argument("--generation-count", type=int, default=1000)
    hf_build_eval_suite.add_argument("--choice-count", type=int, default=200)
    hf_build_eval_suite.add_argument("--text-count", type=int, default=64)
    hf_build_eval_suite.add_argument("--seed", type=int, default=0)
    hf_build_eval_suite.set_defaults(func=cmd_hf_build_eval_suite)

    hf_build_recovery_calibration = sub.add_parser(
        "hf-build-recovery-calibration",
        help="Create synthetic non-eval recovery prompt/response JSONL for compacted HF models.",
    )
    hf_build_recovery_calibration.add_argument("output")
    hf_build_recovery_calibration.add_argument("--math-count", type=int, default=512)
    hf_build_recovery_calibration.add_argument("--code-count", type=int, default=128)
    hf_build_recovery_calibration.add_argument("--tokenizer-count", type=int, default=64)
    hf_build_recovery_calibration.add_argument("--model-surgery-count", type=int, default=64)
    hf_build_recovery_calibration.add_argument("--commonsense-count", type=int, default=64)
    hf_build_recovery_calibration.add_argument("--seed", type=int, default=0)
    hf_build_recovery_calibration.add_argument(
        "--exclude-eval-suite",
        help="Optional eval suite whose prompts should be excluded from recovery rows.",
    )
    hf_build_recovery_calibration.set_defaults(func=cmd_hf_build_recovery_calibration)

    hf_tie_head = sub.add_parser(
        "hf-tie-head",
        help="Tie an HF causal LM output head to input embeddings and save a smaller checkpoint.",
    )
    hf_tie_head.add_argument("model")
    hf_tie_head.add_argument("output")
    hf_tie_head.add_argument("--dtype", default="float32")
    hf_tie_head.add_argument("--device", default="cpu")
    hf_tie_head.add_argument("--trust-remote-code", action="store_true")
    hf_tie_head.set_defaults(func=cmd_hf_tie_head)

    hf_gate = sub.add_parser("hf-gate", help="Compare an HF eval report to a baseline and return a pass/fail verdict.")
    hf_gate.add_argument("baseline_report")
    hf_gate.add_argument("candidate_report")
    hf_gate.add_argument("--output")
    hf_gate.add_argument("--max-ppl-ratio", type=float, default=1.25)
    hf_gate.add_argument(
        "--ppl-gate-mode",
        choices=["hard", "advisory"],
        default="advisory",
        help="Whether PPL ratio failure is a hard gate or behavior-ranking telemetry.",
    )
    hf_gate.add_argument("--max-accuracy-drop", type=float, default=0.0)
    hf_gate.add_argument("--max-behavior-drop", type=float, default=0.0)
    hf_gate.add_argument("--min-choice-accuracy", type=float, default=0.75)
    hf_gate.add_argument("--min-behavior-pass-rate", type=float, default=0.60)
    hf_gate.add_argument("--min-continuation-chars", type=int, default=24)
    hf_gate.add_argument("--min-unique-token-ratio", type=float, default=0.20)
    hf_gate.set_defaults(func=cmd_hf_gate)

    hf_rank_evals = sub.add_parser(
        "hf-rank-evals",
        help="Rank HF eval reports by behavior, choice, generation health, size, then PPL.",
    )
    hf_rank_evals.add_argument("output")
    hf_rank_evals.add_argument("reports", nargs="+")
    hf_rank_evals.add_argument("--min-continuation-chars", type=int, default=24)
    hf_rank_evals.add_argument("--min-unique-token-ratio", type=float, default=0.20)
    hf_rank_evals.set_defaults(func=cmd_hf_rank_evals)

    hf_sweep = sub.add_parser("hf-sweep-ffn", help="Sweep gated FFN resize ratios, evaluate each candidate, and gate results.")
    hf_sweep.add_argument("source_model")
    hf_sweep.add_argument("output_dir")
    hf_sweep.add_argument("--ratios", required=True, help="Comma-separated ratios, e.g. 0.995,1.05.")
    hf_sweep.add_argument("--baseline-report")
    hf_sweep.add_argument("--dtype", default="float16")
    hf_sweep.add_argument("--device", default="cuda:0")
    hf_sweep.add_argument("--trust-remote-code", action="store_true")
    hf_sweep.add_argument("--max-new-tokens", type=int, default=80)
    hf_sweep.add_argument("--max-ppl-ratio", type=float, default=1.25)
    hf_sweep.add_argument("--max-accuracy-drop", type=float, default=0.0)
    hf_sweep.add_argument("--min-choice-accuracy", type=float, default=0.75)
    hf_sweep.add_argument("--repair-shrinks", action="store_true")
    hf_sweep.add_argument("--repair-ridge", type=float, default=1e-3)
    hf_sweep.add_argument("--repair-max-tokens", type=int, default=512)
    hf_sweep.add_argument(
        "--assignment",
        choices=["combined-cosine", "down-cosine", "round-robin", "topk"],
        default="combined-cosine",
    )
    hf_sweep.set_defaults(func=cmd_hf_sweep)

    hf_search_compress = sub.add_parser(
        "hf-search-compress",
        help="Search merge-first width/depth compression candidates and gate each result.",
    )
    hf_search_compress.add_argument("source_model")
    hf_search_compress.add_argument("output_dir")
    hf_search_compress.add_argument("--width-ratios", help="Comma-separated ratios, e.g. 0.9975,0.995.")
    hf_search_compress.add_argument(
        "--layer-groups",
        action="append",
        default=[],
        help="Repeatable layer group spec, e.g. --layer-groups 13+14 --layer-groups 8+9,19+20.",
    )
    hf_search_compress.add_argument("--depth-methods", default="average,fold")
    hf_search_compress.add_argument("--baseline-report")
    hf_search_compress.add_argument("--dtype", default="float16")
    hf_search_compress.add_argument("--device", default="cuda:0")
    hf_search_compress.add_argument("--trust-remote-code", action="store_true")
    hf_search_compress.add_argument("--max-new-tokens", type=int, default=80)
    hf_search_compress.add_argument("--max-ppl-ratio", type=float, default=1.5)
    hf_search_compress.add_argument("--max-accuracy-drop", type=float, default=0.25)
    hf_search_compress.add_argument("--min-choice-accuracy", type=float, default=0.75)
    hf_search_compress.add_argument(
        "--assignment",
        choices=["combined-cosine", "down-cosine", "round-robin", "topk"],
        default="combined-cosine",
    )
    hf_search_compress.add_argument("--target-shrink", type=float)
    hf_search_compress.add_argument("--no-combinations", action="store_true")
    hf_search_compress.set_defaults(func=cmd_hf_search_compress)

    hf_task_sweep = sub.add_parser("hf-sweep-task", help="Sweep task-arithmetic alphas, evaluate candidates, and gate results.")
    hf_task_sweep.add_argument("output_dir")
    hf_task_sweep.add_argument("--base-model", required=True)
    hf_task_sweep.add_argument("--tuned-model", required=True)
    hf_task_sweep.add_argument("--alphas", required=True, help="Comma-separated alphas, e.g. 0,0.5,1,1.25.")
    hf_task_sweep.add_argument("--baseline-report")
    hf_task_sweep.add_argument("--dtype", default="auto")
    hf_task_sweep.add_argument("--device", default="cuda:0")
    hf_task_sweep.add_argument("--trust-remote-code", action="store_true")
    hf_task_sweep.add_argument("--max-new-tokens", type=int, default=80)
    hf_task_sweep.add_argument("--max-ppl-ratio", type=float, default=2.0)
    hf_task_sweep.add_argument("--max-accuracy-drop", type=float, default=0.25)
    hf_task_sweep.add_argument("--min-choice-accuracy", type=float, default=0.75)
    hf_task_sweep.set_defaults(func=cmd_hf_task_sweep)

    xfamily = sub.add_parser("hf-xfamily-vocab-merge", help="Blend donor embedding/lm-head signal into a host HF architecture.")
    xfamily.add_argument("output")
    xfamily.add_argument("--host-model", required=True)
    xfamily.add_argument("--donor-model", required=True)
    xfamily.add_argument("--alpha", type=float, default=0.05)
    xfamily.add_argument("--ridge", type=float, default=1e-3)
    xfamily.add_argument("--max-anchors", type=int, default=50_000)
    xfamily.add_argument("--dtype", default="float16")
    xfamily.add_argument("--trust-remote-code", action="store_true")
    xfamily.set_defaults(func=cmd_hf_xfamily_vocab_merge)

    xfamily_sweep = sub.add_parser("hf-sweep-xfamily", help="Sweep cross-family vocab-transplant alphas, evaluate, and gate.")
    xfamily_sweep.add_argument("output_dir")
    xfamily_sweep.add_argument("--host-model", required=True)
    xfamily_sweep.add_argument("--donor-model", required=True)
    xfamily_sweep.add_argument("--alphas", required=True, help="Comma-separated alphas, e.g. 0.005,0.01,0.05.")
    xfamily_sweep.add_argument("--baseline-report")
    xfamily_sweep.add_argument("--ridge", type=float, default=1e-3)
    xfamily_sweep.add_argument("--max-anchors", type=int, default=50_000)
    xfamily_sweep.add_argument("--dtype", default="float16")
    xfamily_sweep.add_argument("--device", default="cuda:0")
    xfamily_sweep.add_argument("--trust-remote-code", action="store_true")
    xfamily_sweep.add_argument("--max-new-tokens", type=int, default=80)
    xfamily_sweep.add_argument("--max-ppl-ratio", type=float, default=1.5)
    xfamily_sweep.add_argument("--max-accuracy-drop", type=float, default=0.25)
    xfamily_sweep.add_argument("--min-choice-accuracy", type=float, default=0.5)
    xfamily_sweep.set_defaults(func=cmd_hf_xfamily_sweep)

    xfamily_mlp = sub.add_parser(
        "hf-xfamily-mlp-merge",
        help="Merge donor MLP channels into a host HF architecture through shared-token hidden projection.",
    )
    xfamily_mlp.add_argument("output")
    xfamily_mlp.add_argument("--host-model", required=True)
    xfamily_mlp.add_argument("--donor-model", required=True)
    xfamily_mlp.add_argument("--alpha", type=float, default=0.01)
    xfamily_mlp.add_argument("--vocab-alpha", type=float)
    xfamily_mlp.add_argument("--ridge", type=float, default=1e-3)
    xfamily_mlp.add_argument("--max-anchors", type=int, default=50_000)
    xfamily_mlp.add_argument("--max-host-channels", type=int, default=512)
    xfamily_mlp.add_argument("--max-donor-channels", type=int, default=512)
    xfamily_mlp.add_argument(
        "--channel-assignment",
        choices=["weight-cosine", "activation-correlation", "hybrid"],
        default="weight-cosine",
    )
    xfamily_mlp.add_argument("--activation-calibration-file")
    xfamily_mlp.add_argument("--activation-max-calibration-texts", type=int)
    xfamily_mlp.add_argument("--activation-max-length", type=int, default=128)
    xfamily_mlp.add_argument("--activation-device", default="cuda:0")
    xfamily_mlp.add_argument("--layer-alpha-mode", choices=["constant", "similarity"], default="constant")
    xfamily_mlp.add_argument("--layer-alpha-min-scale", type=float, default=0.0)
    xfamily_mlp.add_argument("--layer-alpha-max-scale", type=float, default=2.0)
    xfamily_mlp.add_argument("--layer-alpha-similarity-floor", type=float, default=0.0)
    xfamily_mlp.add_argument("--dtype", default="float16")
    xfamily_mlp.add_argument("--trust-remote-code", action="store_true")
    xfamily_mlp.set_defaults(func=cmd_hf_xfamily_mlp_merge)

    xfamily_mlp_sweep = sub.add_parser(
        "hf-sweep-xfamily-mlp",
        help="Sweep cross-family MLP-channel transplant alphas, evaluate, and gate.",
    )
    xfamily_mlp_sweep.add_argument("output_dir")
    xfamily_mlp_sweep.add_argument("--host-model", required=True)
    xfamily_mlp_sweep.add_argument("--donor-model", required=True)
    xfamily_mlp_sweep.add_argument("--alphas", required=True, help="Comma-separated alphas, e.g. 0.001,0.005,0.01.")
    xfamily_mlp_sweep.add_argument("--baseline-report")
    xfamily_mlp_sweep.add_argument("--vocab-alpha", type=float)
    xfamily_mlp_sweep.add_argument("--ridge", type=float, default=1e-3)
    xfamily_mlp_sweep.add_argument("--max-anchors", type=int, default=50_000)
    xfamily_mlp_sweep.add_argument("--max-host-channels", type=int, default=512)
    xfamily_mlp_sweep.add_argument("--max-donor-channels", type=int, default=512)
    xfamily_mlp_sweep.add_argument(
        "--channel-assignment",
        choices=["weight-cosine", "activation-correlation", "hybrid"],
        default="weight-cosine",
    )
    xfamily_mlp_sweep.add_argument("--activation-calibration-file")
    xfamily_mlp_sweep.add_argument("--activation-max-calibration-texts", type=int)
    xfamily_mlp_sweep.add_argument("--activation-max-length", type=int, default=128)
    xfamily_mlp_sweep.add_argument("--activation-device", default="cuda:0")
    xfamily_mlp_sweep.add_argument("--layer-alpha-mode", choices=["constant", "similarity"], default="constant")
    xfamily_mlp_sweep.add_argument("--layer-alpha-min-scale", type=float, default=0.0)
    xfamily_mlp_sweep.add_argument("--layer-alpha-max-scale", type=float, default=2.0)
    xfamily_mlp_sweep.add_argument("--layer-alpha-similarity-floor", type=float, default=0.0)
    xfamily_mlp_sweep.add_argument("--dtype", default="float16")
    xfamily_mlp_sweep.add_argument("--device", default="cuda:0")
    xfamily_mlp_sweep.add_argument("--trust-remote-code", action="store_true")
    xfamily_mlp_sweep.add_argument("--max-new-tokens", type=int, default=80)
    xfamily_mlp_sweep.add_argument("--max-ppl-ratio", type=float, default=1.5)
    xfamily_mlp_sweep.add_argument("--max-accuracy-drop", type=float, default=0.25)
    xfamily_mlp_sweep.add_argument("--min-choice-accuracy", type=float, default=0.5)
    xfamily_mlp_sweep.add_argument("--recover", action="store_true", help="Run lightweight recovery on each selected candidate and re-gate it.")
    xfamily_mlp_sweep.add_argument("--recover-only-failed", action="store_true")
    xfamily_mlp_sweep.add_argument("--recover-steps", type=int, default=40)
    xfamily_mlp_sweep.add_argument("--recover-batch-size", type=int, default=1)
    xfamily_mlp_sweep.add_argument("--recover-max-length", type=int, default=96)
    xfamily_mlp_sweep.add_argument("--recover-lr", type=float, default=5e-5)
    xfamily_mlp_sweep.add_argument("--recover-kd-weight", type=float, default=0.5)
    xfamily_mlp_sweep.add_argument("--recover-lm-weight", type=float, default=0.5)
    xfamily_mlp_sweep.add_argument("--recover-temperature", type=float, default=2.0)
    xfamily_mlp_sweep.add_argument("--recover-train-scope", default="last1")
    xfamily_mlp_sweep.add_argument("--recover-dtype", default="float32")
    xfamily_mlp_sweep.add_argument("--recover-grad-clip", type=float, default=1.0)
    xfamily_mlp_sweep.add_argument("--recover-calibration-file")
    xfamily_mlp_sweep.add_argument("--recover-max-calibration-texts", type=int)
    xfamily_mlp_sweep.add_argument("--eval-suite", help="JSON/JSONL eval suite used for baseline, candidate, and recovered gates.")
    xfamily_mlp_sweep.add_argument("--max-text-evals", type=int)
    xfamily_mlp_sweep.add_argument("--max-choice-probes", type=int)
    xfamily_mlp_sweep.add_argument("--max-generation-prompts", type=int)
    xfamily_mlp_sweep.add_argument("--max-behavior-probes", type=int)
    xfamily_mlp_sweep.set_defaults(func=cmd_hf_xfamily_mlp_sweep)

    xfamily_compress = sub.add_parser(
        "hf-xfamily-compress",
        help="Run cross-family MLP merge, merge-first compression, and optional recovery.",
    )
    xfamily_compress.add_argument("output_dir")
    xfamily_compress.add_argument("--host-model", required=True)
    xfamily_compress.add_argument("--donor-model", required=True)
    xfamily_compress.add_argument("--alpha", type=float, default=0.01)
    xfamily_compress.add_argument("--vocab-alpha", type=float)
    xfamily_compress.add_argument("--ridge", type=float, default=1e-3)
    xfamily_compress.add_argument("--max-anchors", type=int, default=50_000)
    xfamily_compress.add_argument("--max-host-channels", type=int, default=512)
    xfamily_compress.add_argument("--max-donor-channels", type=int, default=512)
    xfamily_compress.add_argument(
        "--channel-assignment",
        choices=["weight-cosine", "activation-correlation", "hybrid"],
        default="weight-cosine",
    )
    xfamily_compress.add_argument("--activation-calibration-file")
    xfamily_compress.add_argument("--activation-max-calibration-texts", type=int)
    xfamily_compress.add_argument("--activation-max-length", type=int, default=128)
    xfamily_compress.add_argument("--activation-device", default="cuda:0")
    xfamily_compress.add_argument("--layer-alpha-mode", choices=["constant", "similarity"], default="constant")
    xfamily_compress.add_argument("--layer-alpha-min-scale", type=float, default=0.0)
    xfamily_compress.add_argument("--layer-alpha-max-scale", type=float, default=2.0)
    xfamily_compress.add_argument("--layer-alpha-similarity-floor", type=float, default=0.0)
    xfamily_compress.add_argument("--width-ratios", help="Comma-separated ratios, e.g. 0.9975,0.995.")
    xfamily_compress.add_argument(
        "--layer-groups",
        action="append",
        default=[],
        help="Repeatable layer group spec, e.g. --layer-groups 13+14 --layer-groups 8+9,19+20.",
    )
    xfamily_compress.add_argument("--depth-methods", default="average,fold")
    xfamily_compress.add_argument("--target-shrink", type=float)
    xfamily_compress.add_argument(
        "--assignment",
        choices=["combined-cosine", "down-cosine", "round-robin", "topk"],
        default="combined-cosine",
    )
    xfamily_compress.add_argument("--no-combinations", action="store_true")
    xfamily_compress.add_argument("--baseline-report")
    xfamily_compress.add_argument("--dtype", default="float16")
    xfamily_compress.add_argument("--device", default="cuda:0")
    xfamily_compress.add_argument("--trust-remote-code", action="store_true")
    xfamily_compress.add_argument("--max-new-tokens", type=int, default=80)
    xfamily_compress.add_argument("--max-ppl-ratio", type=float, default=1.5)
    xfamily_compress.add_argument("--max-accuracy-drop", type=float, default=0.25)
    xfamily_compress.add_argument("--min-choice-accuracy", type=float, default=0.75)
    xfamily_compress.add_argument("--recover", action="store_true", help="Run lightweight recovery on selected compressed candidates.")
    xfamily_compress.add_argument("--recover-only-failed", action="store_true")
    xfamily_compress.add_argument("--recover-top-k", type=int, default=1)
    xfamily_compress.add_argument("--recover-steps", type=int, default=40)
    xfamily_compress.add_argument("--recover-batch-size", type=int, default=1)
    xfamily_compress.add_argument("--recover-max-length", type=int, default=96)
    xfamily_compress.add_argument("--recover-lr", type=float, default=5e-5)
    xfamily_compress.add_argument("--recover-kd-weight", type=float, default=0.5)
    xfamily_compress.add_argument("--recover-lm-weight", type=float, default=0.5)
    xfamily_compress.add_argument("--recover-temperature", type=float, default=2.0)
    xfamily_compress.add_argument("--recover-train-scope", default="last1")
    xfamily_compress.add_argument("--recover-dtype", default="float32")
    xfamily_compress.add_argument("--recover-grad-clip", type=float, default=1.0)
    xfamily_compress.add_argument("--recover-calibration-file")
    xfamily_compress.add_argument("--recover-max-calibration-texts", type=int)
    xfamily_compress.add_argument("--eval-suite", help="JSON/JSONL eval suite used for baseline, candidate, compression, and recovered gates.")
    xfamily_compress.add_argument("--max-text-evals", type=int)
    xfamily_compress.add_argument("--max-choice-probes", type=int)
    xfamily_compress.add_argument("--max-generation-prompts", type=int)
    xfamily_compress.add_argument("--max-behavior-probes", type=int)
    xfamily_compress.set_defaults(func=cmd_hf_xfamily_compress)

    recover = sub.add_parser("hf-recover", help="Run lightweight same-tokenizer recovery training after HF surgery.")
    recover.add_argument("source_model")
    recover.add_argument("candidate_model")
    recover.add_argument("output")
    recover.add_argument("--steps", type=int, default=80)
    recover.add_argument("--batch-size", type=int, default=1)
    recover.add_argument("--max-length", type=int, default=96)
    recover.add_argument("--lr", type=float, default=5e-5)
    recover.add_argument("--kd-weight", type=float, default=0.5)
    recover.add_argument("--lm-weight", type=float, default=0.5)
    recover.add_argument("--hidden-kd-weight", type=float, default=0.0)
    recover.add_argument(
        "--hidden-kd-metric",
        choices=[
            "projected",
            "cka",
            "both",
            "rsd",
            "projected-rsd",
            "gram",
            "projected-gram",
            "geometry",
            "projected-geometry",
        ],
        default="cka",
    )
    recover.add_argument("--hidden-layer", choices=["last", "middle", "all"], default="last")
    recover.add_argument("--hidden-token-mode", choices=["tokens", "response", "pooled"], default="tokens")
    recover.add_argument("--hidden-ridge", type=float, default=1e-3)
    recover.add_argument("--hidden-max-anchors", type=int, default=2048)
    recover.add_argument("--top-k-kd-weight", type=float, default=0.0)
    recover.add_argument("--top-k-kd-k", type=int, default=32)
    recover.add_argument("--first-token-top-k-kd-weight", type=float, default=0.0)
    recover.add_argument("--first-token-kd-weight", type=float, default=0.0)
    recover.add_argument("--first-token-lm-weight", type=float, default=0.0)
    recover.add_argument("--unlikelihood-weight", type=float, default=0.0)
    recover.add_argument("--unlikelihood-window", type=int, default=16)
    recover.add_argument("--entropy-weight", type=float, default=0.0)
    recover.add_argument("--response-loss-mode", choices=["none", "auto", "assistant", "answer"], default="none")
    recover.add_argument("--append-eos", action="store_true")
    recover.add_argument("--temperature", type=float, default=2.0)
    recover.add_argument("--train-scope", default="all")
    recover.add_argument("--dtype", default="float32")
    recover.add_argument("--candidate-train-dtype")
    recover.add_argument("--device", default="cuda:0")
    recover.add_argument("--trust-remote-code", action="store_true")
    recover.add_argument("--grad-clip", type=float, default=1.0)
    recover.add_argument("--report-interval", type=int)
    recover.add_argument("--save-best-loss", action="store_true")
    recover.add_argument(
        "--save-best-behavior",
        action="store_true",
        help="Select the saved recovery checkpoint by behavior probes instead of loss.",
    )
    recover.add_argument("--behavior-eval-interval", type=int)
    recover.add_argument("--behavior-max-new-tokens", type=int, default=96)
    recover.add_argument("--behavior-max-probes", type=int, default=6)
    recover.add_argument("--behavior-eval-suite")
    recover.add_argument("--behavior-max-generation-prompts", type=int)
    recover.add_argument("--calibration-file")
    recover.add_argument("--max-calibration-texts", type=int)
    recover.set_defaults(func=cmd_hf_recover)

    xfamily_recover = sub.add_parser(
        "hf-xfamily-recover",
        help="Run cross-tokenizer recovery using donor shared-token anchor logits plus child LM loss.",
    )
    xfamily_recover.add_argument("source_model")
    xfamily_recover.add_argument("candidate_model")
    xfamily_recover.add_argument("output")
    xfamily_recover.add_argument("--steps", type=int, default=40)
    xfamily_recover.add_argument("--batch-size", type=int, default=1)
    xfamily_recover.add_argument("--max-length", type=int, default=96)
    xfamily_recover.add_argument("--lr", type=float, default=5e-5)
    xfamily_recover.add_argument("--anchor-kd-weight", type=float, default=0.5)
    xfamily_recover.add_argument("--lm-weight", type=float, default=0.5)
    xfamily_recover.add_argument("--first-token-lm-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--response-end-lm-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--focused-lm-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--hidden-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--span-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--span-hidden-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument(
        "--dual-space-kd-weight",
        type=float,
        default=0.0,
        help="Cross-tokenizer DSKD-style loss using student hidden through teacher head and teacher hidden through student head.",
    )
    xfamily_recover.add_argument(
        "--attention-influence-weight",
        type=float,
        default=0.0,
        help="AIM-style loss matching teacher/student attention influence over span-aligned tokens.",
    )
    xfamily_recover.add_argument("--attention-influence-layer", choices=["last", "middle"], default="last")
    xfamily_recover.add_argument("--projected-vocab-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--focused-projected-vocab-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument(
        "--projected-vocab-kd-mode",
        choices=["teacher-to-student", "student-to-teacher", "both"],
        default="teacher-to-student",
    )
    xfamily_recover.add_argument("--projected-vocab-max-source-tokens", type=int, default=8)
    xfamily_recover.add_argument("--reference-model")
    xfamily_recover.add_argument("--reference-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--unlikelihood-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--unlikelihood-window", type=int, default=16)
    xfamily_recover.add_argument("--entropy-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--response-loss-mode", choices=["none", "auto", "assistant", "answer"], default="none")
    xfamily_recover.add_argument("--token-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--token-focus", choices=["auto", "response", "numbers", "code", "all"], default="auto")
    xfamily_recover.add_argument("--token-max-pairs", type=int, default=512)
    xfamily_recover.add_argument("--token-teacher-confidence-floor", type=float, default=0.0)
    xfamily_recover.add_argument(
        "--alm-kd-weight",
        type=float,
        default=0.0,
        help="Align teacher/student likelihoods over same-character chunks for cross-tokenizer recovery.",
    )
    xfamily_recover.add_argument("--alm-max-chunks", type=int, default=64)
    xfamily_recover.add_argument("--alm-min-chars", type=int, default=2)
    xfamily_recover.add_argument("--answer-contrastive-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--answer-contrastive-max-negatives", type=int, default=4)
    xfamily_recover.add_argument("--answer-first-token-margin-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--answer-first-token-margin", type=float, default=1.0)
    xfamily_recover.add_argument("--answer-first-token-max-negatives", type=int, default=8)
    xfamily_recover.add_argument("--answer-first-token-top-negatives", type=int, default=0)
    xfamily_recover.add_argument("--teacher-rank-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--teacher-rank-temperature", type=float, default=1.0)
    xfamily_recover.add_argument("--teacher-rank-max-negatives", type=int, default=4)
    xfamily_recover.add_argument(
        "--teacher-generate-lm-weight",
        type=float,
        default=0.0,
        help="Train the child on donor-generated plain-text continuations using the child tokenizer.",
    )
    xfamily_recover.add_argument("--teacher-generate-max-new-tokens", type=int, default=64)
    xfamily_recover.add_argument(
        "--stepwise-donor-kd-weight",
        type=float,
        default=0.0,
        help="Correct child greedy prefixes with donor next-token logits projected into the child vocab.",
    )
    xfamily_recover.add_argument("--stepwise-donor-max-new-tokens", type=int, default=16)
    xfamily_recover.add_argument("--on-policy-anchor-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--on-policy-unlikelihood-weight", type=float, default=0.0)
    xfamily_recover.add_argument(
        "--on-policy-hidden-kd-weight",
        type=float,
        default=0.0,
        help="Align hidden states on child-generated rollouts, using cross-tokenizer span matches when available.",
    )
    xfamily_recover.add_argument("--on-policy-max-new-tokens", type=int, default=24)
    xfamily_recover.add_argument("--on-policy-interval", type=int, default=10)
    xfamily_recover.add_argument("--append-eos", action="store_true")
    xfamily_recover.add_argument("--entropy-aware-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--entropy-aware-kd-threshold", type=float, default=0.55)
    xfamily_recover.add_argument("--entropy-aware-kd-sharpness", type=float, default=8.0)
    xfamily_recover.add_argument("--span-max-pairs", type=int, default=512)
    xfamily_recover.add_argument("--hidden-ridge", type=float, default=1e-3)
    xfamily_recover.add_argument("--hidden-layer", choices=["last", "middle"], default="last")
    xfamily_recover.add_argument(
        "--hidden-kd-metric",
        choices=[
            "projected",
            "cka",
            "both",
            "rsd",
            "projected-rsd",
            "gram",
            "projected-gram",
            "geometry",
            "projected-geometry",
        ],
        default="projected",
    )
    xfamily_recover.add_argument("--hidden-max-anchors", type=int, default=2048)
    xfamily_recover.add_argument("--temperature", type=float, default=2.0)
    xfamily_recover.add_argument("--max-anchors", type=int, default=20_000)
    xfamily_recover.add_argument("--anchor-selection", choices=["shortest", "longest", "stratified"], default="stratified")
    xfamily_recover.add_argument("--anchor-min-chars", type=int, default=1)
    xfamily_recover.add_argument("--train-scope", default="last1")
    xfamily_recover.add_argument("--dtype", default="float32")
    xfamily_recover.add_argument("--candidate-dtype", default="float32")
    xfamily_recover.add_argument("--device", default="cuda:0")
    xfamily_recover.add_argument("--trust-remote-code", action="store_true")
    xfamily_recover.add_argument("--grad-clip", type=float, default=1.0)
    xfamily_recover.add_argument("--report-interval", type=int)
    xfamily_recover.add_argument("--save-best-loss", action="store_true")
    xfamily_recover.add_argument(
        "--save-best-behavior",
        action="store_true",
        help="Select the saved recovery checkpoint by behavior probes instead of loss.",
    )
    xfamily_recover.add_argument("--behavior-eval-interval", type=int)
    xfamily_recover.add_argument("--behavior-max-new-tokens", type=int, default=96)
    xfamily_recover.add_argument("--behavior-max-probes", type=int, default=6)
    xfamily_recover.add_argument("--behavior-eval-suite")
    xfamily_recover.add_argument("--behavior-max-generation-prompts", type=int)
    xfamily_recover.add_argument("--calibration-file")
    xfamily_recover.add_argument("--max-calibration-texts", type=int)
    xfamily_recover.add_argument(
        "--calibration-order",
        choices=["original", "domain-balanced"],
        default="original",
        help="Order recovery calibration texts as provided or round-robin by keyword-inferred domain.",
    )
    xfamily_recover.set_defaults(func=cmd_hf_xfamily_recover)

    reconstruct = sub.add_parser(
        "hf-reconstruct-folded",
        help="Run training-free ridge reconstruction after residual layer folding.",
    )
    reconstruct.add_argument("source_model")
    reconstruct.add_argument("candidate_model")
    reconstruct.add_argument("output")
    reconstruct.add_argument("--layer-merge-report")
    reconstruct.add_argument("--ridge", type=float, default=1e-3)
    reconstruct.add_argument("--reconstruction-alpha", type=float, default=1.0)
    reconstruct.add_argument(
        "--reconstruction-target",
        choices=["consumer-input", "survivor-mlp", "both"],
        default="consumer-input",
    )
    reconstruct.add_argument(
        "--reconstruction-objective",
        choices=["ridge", "cosine"],
        default="ridge",
        help="Fit folded activations with ridge L2 or ridge-initialized cosine refinement.",
    )
    reconstruct.add_argument("--cosine-steps", type=int, default=80)
    reconstruct.add_argument("--cosine-lr", type=float, default=1e-2)
    reconstruct.add_argument("--cosine-l2", type=float, default=1e-4)
    reconstruct.add_argument("--max-length", type=int, default=128)
    reconstruct.add_argument("--max-tokens-per-layer", type=int, default=512)
    reconstruct.add_argument("--dtype", default="float16")
    reconstruct.add_argument("--device", default="cuda:0")
    reconstruct.add_argument("--trust-remote-code", action="store_true")
    reconstruct.add_argument("--calibration-file")
    reconstruct.add_argument("--max-calibration-texts", type=int)
    reconstruct.set_defaults(func=cmd_hf_reconstruct_folded)

    xfamily_reconstruct = sub.add_parser(
        "hf-xfamily-reconstruct",
        help="Run cross-tokenizer blockwise activation reconstruction on a folded HF causal LM.",
    )
    xfamily_reconstruct.add_argument("source_model")
    xfamily_reconstruct.add_argument("candidate_model")
    xfamily_reconstruct.add_argument("output")
    xfamily_reconstruct.add_argument("--ridge", type=float, default=1e-3)
    xfamily_reconstruct.add_argument("--reconstruction-alpha", type=float, default=1.0)
    xfamily_reconstruct.add_argument(
        "--reconstruction-target",
        choices=[
            "consumer-input",
            "block-interface",
            "attention",
            "attention-output",
            "attention-all",
            "mlp",
            "mlp-output",
            "mlp-all",
            "both",
            "all",
        ],
        default="both",
    )
    xfamily_reconstruct.add_argument(
        "--donor-layer-fold-mode",
        choices=["representative", "average", "fold"],
        default="representative",
    )
    xfamily_reconstruct.add_argument(
        "--reconstruction-objective",
        choices=["ridge", "cosine"],
        default="ridge",
    )
    xfamily_reconstruct.add_argument("--cosine-steps", type=int, default=80)
    xfamily_reconstruct.add_argument("--cosine-lr", type=float, default=1e-2)
    xfamily_reconstruct.add_argument("--cosine-l2", type=float, default=1e-4)
    xfamily_reconstruct.add_argument("--max-length", type=int, default=128)
    xfamily_reconstruct.add_argument("--max-tokens-per-layer", type=int, default=512)
    xfamily_reconstruct.add_argument("--max-anchors", type=int, default=2048)
    xfamily_reconstruct.add_argument(
        "--hidden-projection-scope",
        choices=["global", "layer-input"],
        default="global",
        help="Use the global shared-token hidden map or refit donor->child maps from each layer input.",
    )
    xfamily_reconstruct.add_argument("--layer-limit", type=int)
    xfamily_reconstruct.add_argument(
        "--fit-gate",
        choices=["none", "validation"],
        default="none",
        help="Gate each fitted activation map on held-out activation rows before mutating weights.",
    )
    xfamily_reconstruct.add_argument("--fit-gate-validation-fraction", type=float, default=0.25)
    xfamily_reconstruct.add_argument("--fit-gate-min-validation-rows", type=int, default=32)
    xfamily_reconstruct.add_argument("--fit-gate-min-relative-improvement", type=float, default=0.0)
    xfamily_reconstruct.add_argument("--fit-gate-max-cosine-drop", type=float, default=0.01)
    xfamily_reconstruct.add_argument(
        "--layer-gate-metric",
        choices=["none", "ppl", "behavior"],
        default="none",
        help="Optionally evaluate after each repaired layer and revert repairs that fail the gate.",
    )
    xfamily_reconstruct.add_argument(
        "--layer-gate-max-ppl-ratio",
        type=float,
        default=1.02,
        help="Maximum candidate/previous PPL ratio accepted by --layer-gate-metric ppl.",
    )
    xfamily_reconstruct.add_argument(
        "--layer-gate-max-texts",
        type=int,
        default=4,
        help="Number of held-out text or behavior probes used by --layer-gate-metric.",
    )
    xfamily_reconstruct.add_argument(
        "--layer-gate-max-behavior-drop",
        type=float,
        default=0.0,
        help="Maximum behavior pass-rate drop accepted by --layer-gate-metric behavior.",
    )
    xfamily_reconstruct.add_argument(
        "--layer-gate-behavior-max-new-tokens",
        type=int,
        default=96,
        help="Generation length for behavior probes used by --layer-gate-metric behavior.",
    )
    xfamily_reconstruct.add_argument(
        "--layer-gate-eval-suite",
        help="JSON/JSONL eval suite whose behavior probes replace the built-in layer behavior gate probes.",
    )
    xfamily_reconstruct.add_argument(
        "--layer-gate-max-generation-prompts",
        type=int,
        help="Maximum generation prompts used from --layer-gate-eval-suite for each layer behavior gate.",
    )
    xfamily_reconstruct.add_argument("--dtype", default="float16")
    xfamily_reconstruct.add_argument("--candidate-dtype", default="float32")
    xfamily_reconstruct.add_argument("--device", default="cuda:0")
    xfamily_reconstruct.add_argument("--trust-remote-code", action="store_true")
    xfamily_reconstruct.add_argument("--calibration-file")
    xfamily_reconstruct.add_argument("--max-calibration-texts", type=int)
    xfamily_reconstruct.set_defaults(func=cmd_hf_xfamily_reconstruct)

    xfamily_block_recover = sub.add_parser(
        "hf-xfamily-block-recover",
        help="Sequentially train each target block against projected cross-tokenizer donor block activations.",
    )
    xfamily_block_recover.add_argument("source_model")
    xfamily_block_recover.add_argument("candidate_model")
    xfamily_block_recover.add_argument("output")
    xfamily_block_recover.add_argument("--steps-per-layer", type=int, default=24)
    xfamily_block_recover.add_argument("--batch-size", type=int, default=1)
    xfamily_block_recover.add_argument("--max-length", type=int, default=128)
    xfamily_block_recover.add_argument("--lr", type=float, default=2e-5)
    xfamily_block_recover.add_argument(
        "--lm-weight",
        type=float,
        default=0.0,
        help="Candidate-token LM stabilizer weight applied while each block is trained.",
    )
    xfamily_block_recover.add_argument(
        "--response-lm-weight",
        type=float,
        default=0.0,
        help="Response-token-only LM weight applied while each block is trained.",
    )
    xfamily_block_recover.add_argument(
        "--first-token-lm-weight",
        type=float,
        default=0.0,
        help="Extra LM weight on the first response token during blockwise folding.",
    )
    xfamily_block_recover.add_argument(
        "--response-end-lm-weight",
        type=float,
        default=0.0,
        help="Extra LM weight on the last response token during blockwise folding.",
    )
    xfamily_block_recover.add_argument(
        "--response-loss-mode",
        choices=["none", "auto", "assistant", "answer"],
        default="none",
        help="How calibration responses are masked for blockwise response LM losses.",
    )
    xfamily_block_recover.add_argument("--hidden-ridge", type=float, default=1e-3)
    xfamily_block_recover.add_argument(
        "--hidden-projection-scope",
        choices=["global", "layer-input"],
        default="global",
        help="Use one shared-token hidden map or refit donor->child maps from each block input.",
    )
    xfamily_block_recover.add_argument(
        "--hidden-metric",
        choices=[
            "projected",
            "cka",
            "both",
            "rsd",
            "projected-rsd",
            "gram",
            "projected-gram",
            "geometry",
            "projected-geometry",
        ],
        default="projected",
        help="Block activation objective used for span-aligned donor/target hidden rows.",
    )
    xfamily_block_recover.add_argument(
        "--block-target",
        choices=["hidden", "delta"],
        default="hidden",
        help="Train block recovery on absolute hidden states or residual update deltas.",
    )
    xfamily_block_recover.add_argument(
        "--residual-magnitude-compensation",
        choices=RESIDUAL_MAGNITUDE_COMPENSATION_CHOICES,
        default="none",
        help=(
            "Optionally compensate child attention/MLP output projections to match projected donor "
            "residual-delta activations before layer training. The pre-module-* modes fit separate "
            "attention-output and MLP-output maps."
        ),
    )
    xfamily_block_recover.add_argument("--residual-compensation-min-scale", type=float, default=0.50)
    xfamily_block_recover.add_argument("--residual-compensation-max-scale", type=float, default=1.50)
    xfamily_block_recover.add_argument("--residual-compensation-linear-ridge", type=float, default=1e-3)
    xfamily_block_recover.add_argument("--residual-compensation-alpha", type=float, default=1.0)
    xfamily_block_recover.add_argument("--residual-compensation-linear-rank", type=int, default=32)
    xfamily_block_recover.add_argument(
        "--pre-reconstruction-target",
        choices=["none", "consumer-input", "block-interface", "all"],
        default="none",
        help=(
            "Optionally run cross-tokenizer activation reconstruction before each block is trained. "
            "consumer-input and block-interface mutate downstream consumer projections and are reverted by layer gates."
        ),
    )
    xfamily_block_recover.add_argument("--pre-reconstruction-ridge", type=float, default=1e-3)
    xfamily_block_recover.add_argument("--pre-reconstruction-alpha", type=float, default=0.05)
    xfamily_block_recover.add_argument(
        "--pre-reconstruction-objective",
        choices=["ridge", "cosine"],
        default="ridge",
        help="Linear-map objective for pre-reconstruction interface maps.",
    )
    xfamily_block_recover.add_argument("--pre-reconstruction-cosine-steps", type=int, default=40)
    xfamily_block_recover.add_argument("--pre-reconstruction-cosine-lr", type=float, default=1e-2)
    xfamily_block_recover.add_argument("--pre-reconstruction-cosine-l2", type=float, default=1e-4)
    xfamily_block_recover.add_argument(
        "--pre-reconstruction-fit-gate",
        choices=["none", "validation"],
        default="validation",
        help="Gate fitted pre-reconstruction maps on held-out activation rows before mutating weights.",
    )
    xfamily_block_recover.add_argument("--pre-reconstruction-validation-fraction", type=float, default=0.25)
    xfamily_block_recover.add_argument("--pre-reconstruction-min-validation-rows", type=int, default=32)
    xfamily_block_recover.add_argument("--pre-reconstruction-max-tokens-per-layer", type=int, default=512)
    xfamily_block_recover.add_argument("--pre-reconstruction-min-relative-improvement", type=float, default=0.0)
    xfamily_block_recover.add_argument("--pre-reconstruction-max-cosine-drop", type=float, default=0.01)
    xfamily_block_recover.add_argument("--max-anchors", type=int, default=2048)
    xfamily_block_recover.add_argument("--span-max-pairs", type=int, default=512)
    xfamily_block_recover.add_argument(
        "--span-pair-mode",
        choices=["exact", "overlap", "both"],
        default="exact",
        help="How cross-tokenizer positions are paired for block/output KD: exact decoded-token matches, character-overlap matches, or both.",
    )
    xfamily_block_recover.add_argument(
        "--donor-layer-fold-mode",
        choices=["representative", "average", "fold"],
        default="fold",
        help="How multiple donor layers mapped to one target layer define the donor block target.",
    )
    xfamily_block_recover.add_argument(
        "--donor-layer-group-strategy",
        choices=["depth", "activation-energy", "activation-geometry", "activation-cka"],
        default="depth",
        help=(
            "How donor layers are grouped before block recovery. activation-energy uses teacher residual "
            "update norms on calibration batches instead of uniform depth bins; activation-geometry also weights "
            "CKA/cosine shift and residual novelty; activation-cka clusters teacher residual-update directions "
            "by pairwise activation CKA so similar non-contiguous layers merge before folding."
        ),
    )
    xfamily_block_recover.add_argument("--layer-limit", type=int)
    xfamily_block_recover.add_argument(
        "--layer-gate-metric",
        choices=["none", "ppl", "behavior"],
        default="none",
        help="Optionally evaluate after each trained block and revert blocks that fail the gate.",
    )
    xfamily_block_recover.add_argument("--layer-gate-max-ppl-ratio", type=float, default=1.10)
    xfamily_block_recover.add_argument("--layer-gate-max-texts", type=int, default=4)
    xfamily_block_recover.add_argument(
        "--layer-gate-max-behavior-drop",
        type=float,
        default=0.0,
        help="Maximum held-out behavior pass-rate drop accepted by --layer-gate-metric behavior.",
    )
    xfamily_block_recover.add_argument(
        "--layer-gate-behavior-max-new-tokens",
        type=int,
        default=96,
        help="Maximum generated tokens per held-out behavior probe for behavior layer gating.",
    )
    xfamily_block_recover.add_argument(
        "--layer-gate-eval-suite",
        help="Optional eval suite whose behavior and generation probes replace the tiny built-in layer behavior gate set.",
    )
    xfamily_block_recover.add_argument(
        "--layer-gate-max-generation-prompts",
        type=int,
        help="Maximum generation prompts used from --layer-gate-eval-suite for each layer behavior gate.",
    )
    xfamily_block_recover.add_argument("--on-policy-anchor-kd-weight", type=float, default=0.0)
    xfamily_block_recover.add_argument("--on-policy-unlikelihood-weight", type=float, default=0.0)
    xfamily_block_recover.add_argument(
        "--on-policy-hidden-kd-weight",
        type=float,
        default=0.0,
        help="Add block hidden alignment on greedy child rollouts during blockwise recovery.",
    )
    xfamily_block_recover.add_argument("--on-policy-max-new-tokens", type=int, default=24)
    xfamily_block_recover.add_argument("--on-policy-interval", type=int, default=10)
    xfamily_block_recover.add_argument(
        "--on-policy-response-loss-mode",
        choices=["none", "auto", "assistant", "answer"],
        default="auto",
        help="Prompt truncation mode used before block-recovery on-policy rollouts.",
    )
    xfamily_block_recover.add_argument("--on-policy-unlikelihood-window", type=int, default=16)
    xfamily_block_recover.add_argument("--stepwise-donor-kd-weight", type=float, default=0.0)
    xfamily_block_recover.add_argument("--stepwise-donor-max-new-tokens", type=int, default=8)
    xfamily_block_recover.add_argument("--stepwise-donor-interval", type=int, default=4)
    xfamily_block_recover.add_argument(
        "--teacher-generate-lm-weight",
        type=float,
        default=0.0,
        help="Train each active block/output head on donor-generated completions retokenized by the child.",
    )
    xfamily_block_recover.add_argument(
        "--teacher-generate-hidden-kd-weight",
        type=float,
        default=0.0,
        help="Add block hidden alignment on donor-generated completions during block recovery.",
    )
    xfamily_block_recover.add_argument(
        "--teacher-generate-projected-vocab-kd-weight",
        type=float,
        default=0.0,
        help="Apply projected-vocab donor KD on donor-generated completions during block recovery.",
    )
    xfamily_block_recover.add_argument(
        "--teacher-generate-byte-kd-weight",
        type=float,
        default=0.0,
        help="Apply tokenizer-neutral first-byte KD on donor-generated completions during block recovery.",
    )
    xfamily_block_recover.add_argument("--teacher-generate-max-new-tokens", type=int, default=64)
    xfamily_block_recover.add_argument(
        "--alm-kd-weight",
        type=float,
        default=0.0,
        help="ALM-style character-chunk likelihood matching for cross-tokenizer block recovery.",
    )
    xfamily_block_recover.add_argument("--alm-max-chunks", type=int, default=64)
    xfamily_block_recover.add_argument("--alm-min-chars", type=int, default=2)
    xfamily_block_recover.add_argument(
        "--projected-vocab-kd-weight",
        type=float,
        default=0.0,
        help="Static span-aligned donor next-token KD projected into the child vocab during each block step.",
    )
    xfamily_block_recover.add_argument(
        "--focused-block-hidden-kd-weight",
        type=float,
        default=0.0,
        help="Extra block hidden alignment on focused response spans such as numbers or code.",
    )
    xfamily_block_recover.add_argument(
        "--focused-projected-vocab-kd-weight",
        type=float,
        default=0.0,
        help="Extra projected-vocab donor KD on focused response spans such as numbers or code.",
    )
    xfamily_block_recover.add_argument(
        "--focused-lm-weight",
        type=float,
        default=0.0,
        help="Extra child-token LM loss on focused response spans such as numbers or code.",
    )
    xfamily_block_recover.add_argument(
        "--token-focus",
        choices=["auto", "response", "numbers", "code", "all"],
        default="auto",
        help="Focused span selector used by blockwise focused hidden/projected/LM losses.",
    )
    xfamily_block_recover.add_argument(
        "--projected-vocab-kd-mode",
        choices=["teacher-to-student", "student-to-teacher", "both"],
        default="teacher-to-student",
    )
    xfamily_block_recover.add_argument(
        "--byte-kd-weight",
        type=float,
        default=0.0,
        help="Project donor and child next-token logits to first-byte buckets for tokenizer-neutral KD.",
    )
    xfamily_block_recover.add_argument(
        "--byte-hidden-kd-weight",
        type=float,
        default=0.0,
        help="Use a fixed byte probe from child output embeddings to align active block hidden states with donor byte targets.",
    )
    xfamily_block_recover.add_argument(
        "--block-io-kd-weight",
        type=float,
        default=0.0,
        help="Fitted full-block input/output reconstruction loss in donor hidden space.",
    )
    xfamily_block_recover.add_argument("--block-io-ridge", type=float, default=1e-3)
    xfamily_block_recover.add_argument(
        "--block-io-loss",
        choices=["smooth-l1", "cosine", "both"],
        default="both",
        help="Loss used after mapping candidate block outputs/deltas into donor hidden space.",
    )
    xfamily_block_recover.add_argument(
        "--train-output-head-with-kd",
        action="store_true",
        help="Train the output head together with each block when output-space donor KD is enabled.",
    )
    xfamily_block_recover.add_argument(
        "--untie-output-head-for-kd",
        action="store_true",
        help="Clone a tied LM head before KD/head training so input token embeddings are not mutated.",
    )
    xfamily_block_recover.add_argument(
        "--allow-tied-output-head-training",
        action="store_true",
        help="Allow --train-output-head-with-kd to mutate tied input/output embeddings.",
    )
    xfamily_block_recover.add_argument(
        "--module-output-kd-weight",
        type=float,
        default=0.0,
        help="Weight for differentiable attention/MLP module-output reconstruction during block recovery.",
    )
    xfamily_block_recover.add_argument(
        "--module-output-targets",
        choices=["attention", "mlp", "both"],
        default="both",
        help="Which submodule outputs to align for --module-output-kd-weight.",
    )
    xfamily_block_recover.add_argument(
        "--module-output-transport",
        choices=["paired", "sinkhorn", "both"],
        default="paired",
        help="Row matching mode for module-output KD.",
    )
    xfamily_block_recover.add_argument("--module-output-temperature", type=float, default=1.0)
    xfamily_block_recover.add_argument("--module-output-sinkhorn-iterations", type=int, default=12)
    xfamily_block_recover.add_argument("--projected-vocab-max-source-tokens", type=int, default=8)
    xfamily_block_recover.add_argument("--temperature", type=float, default=2.0)
    xfamily_block_recover.add_argument("--dtype", default="float16")
    xfamily_block_recover.add_argument("--candidate-dtype", default="float32")
    xfamily_block_recover.add_argument("--device", default="cuda:0")
    xfamily_block_recover.add_argument("--trust-remote-code", action="store_true")
    xfamily_block_recover.add_argument("--calibration-file")
    xfamily_block_recover.add_argument("--max-calibration-texts", type=int)
    xfamily_block_recover.add_argument(
        "--calibration-order",
        choices=["original", "domain-balanced"],
        default="original",
        help="Order recovery calibration texts as provided or round-robin by keyword-inferred domain.",
    )
    xfamily_block_recover.add_argument("--grad-clip", type=float, default=1.0)
    xfamily_block_recover.add_argument("--report-interval", type=int)
    xfamily_block_recover.add_argument("--save-best-layer-loss", action="store_true")
    xfamily_block_recover.add_argument(
        "--save-best-layer-behavior",
        action="store_true",
        help="Within each trained block, keep the checkpoint selected by held-out behavior probes.",
    )
    xfamily_block_recover.add_argument("--layer-behavior-eval-interval", type=int)
    xfamily_block_recover.add_argument(
        "--layer-checkpoint-interval",
        type=int,
        default=0,
        help="Save a full candidate checkpoint every N completed block-fold layers. Disabled when 0.",
    )
    xfamily_block_recover.set_defaults(func=cmd_hf_xfamily_block_recover)

    xfamily_chain = sub.add_parser(
        "hf-xfamily-chain-anchor-compress",
        help="Run staged donor-to-child updates while preserving a behavior-gated incumbent anchor.",
    )
    xfamily_chain.add_argument("source_model")
    xfamily_chain.add_argument("anchor_model")
    xfamily_chain.add_argument("output")
    xfamily_chain.add_argument(
        "--source-weights",
        "--donor-weights",
        dest="source_weights",
        help=(
            "Optional comma-separated weights when source_model contains multiple comma-separated donors. "
            "The init stage folds all donors; recovery stages use the first donor as the teacher."
        ),
    )
    xfamily_chain.add_argument("--stages", default="block,alm", help="Comma-separated stage list: init,block,alm.")
    xfamily_chain.add_argument("--eval-suite")
    xfamily_chain.add_argument("--max-text-evals", type=int)
    xfamily_chain.add_argument("--max-choice-probes", type=int)
    xfamily_chain.add_argument("--max-generation-prompts", type=int)
    xfamily_chain.add_argument("--max-behavior-probes", type=int)
    xfamily_chain.add_argument("--max-new-tokens", type=int, default=96)
    xfamily_chain.add_argument("--stage-max-ppl-ratio", type=float, default=1.25)
    xfamily_chain.add_argument("--stage-ppl-gate-mode", choices=["hard", "advisory"], default="advisory")
    xfamily_chain.add_argument("--stage-max-accuracy-drop", type=float, default=0.0)
    xfamily_chain.add_argument("--stage-max-behavior-drop", type=float, default=0.0)
    xfamily_chain.add_argument("--stage-min-choice-accuracy", type=float, default=0.0)
    xfamily_chain.add_argument("--stage-min-behavior-pass-rate", type=float, default=0.0)
    xfamily_chain.add_argument("--promote-policy", choices=["gate-and-rank", "gate-only", "always"], default="gate-and-rank")
    xfamily_chain.add_argument(
        "--stage-input-policy",
        choices=["incumbent", "sequential"],
        default="incumbent",
        help="Feed each stage from the promoted incumbent or sequentially from the previous stage output.",
    )
    xfamily_chain.add_argument("--init-fold-mode", choices=["mean", "rms", "transport"], default="mean")
    xfamily_chain.add_argument("--init-hidden-basis-mode", choices=["none", "activation-pca", "shared-token-ridge"], default="none")
    xfamily_chain.add_argument("--init-hidden-basis-scope", choices=["global", "layer"], default="global")
    xfamily_chain.add_argument("--init-hidden-basis-max-rows", type=int, default=2048)
    xfamily_chain.add_argument(
        "--init-hidden-basis-preprocess",
        choices=["centered-rowrms", "uncentered-rowrms", "uncentered"],
        default="centered-rowrms",
    )
    xfamily_chain.add_argument(
        "--init-activation-fold-assignment",
        choices=["balanced", "similarity", "signed-similarity"],
        default="balanced",
    )
    xfamily_chain.add_argument("--init-fold-stat-repair", choices=["none", "rms"], default="none")
    xfamily_chain.add_argument("--init-transport-backend", choices=["sparse", "sinkhorn"], default="sparse")
    xfamily_chain.add_argument(
        "--init-transport-support",
        choices=["centroid", "importance", "quantile", "medoid", "quantile-medoid"],
        default="centroid",
    )
    xfamily_chain.add_argument("--init-donor-layer-fold-mode", choices=["representative", "average", "fold"], default="fold")
    xfamily_chain.add_argument(
        "--init-donor-layer-group-strategy",
        choices=["depth", "activation-energy", "activation-geometry", "activation-cka"],
        default="depth",
    )
    xfamily_chain.add_argument(
        "--init-attention-fold-mode",
        choices=["independent", "coupled", "activation-coupled"],
        default="independent",
    )
    xfamily_chain.add_argument("--init-attention-head-dim-transport", choices=["none", "qk-vo"], default="none")
    xfamily_chain.add_argument(
        "--init-token-init-mode",
        choices=["shared", "donor-tokenized", "donor-tokenized-weighted", "donor-omp"],
        default="shared",
    )
    xfamily_chain.add_argument(
        "--init-projection-scale",
        choices=["none", "target-rms", "source-rms", "geomean-rms"],
        default="none",
    )
    xfamily_chain.add_argument("--init-projection-blend-alpha", type=float, default=1.0)
    xfamily_chain.add_argument("--init-consumer-hidden-reduction", choices=["mean", "sum", "sqrt-sum"], default="mean")
    xfamily_chain.add_argument("--init-activation-fit-linears", action="store_true")
    xfamily_chain.add_argument(
        "--init-activation-fit-target",
        choices=["all", "attention", "attention-output", "mlp", "mlp-output", "input-projections"],
        default="all",
    )
    xfamily_chain.add_argument("--init-activation-fit-max-rows", type=int, default=0)
    xfamily_chain.add_argument("--init-activation-fit-alpha", type=float, default=1.0)
    xfamily_chain.add_argument("--init-activation-fit-ridge", type=float, default=1e-1)
    xfamily_chain.add_argument("--init-activation-fit-rank", type=int, default=0)
    xfamily_chain.add_argument("--init-activation-fit-rank-mode", choices=["absolute", "delta"], default="absolute")
    xfamily_chain.add_argument("--init-activation-fit-method", choices=["ridge", "lrc"], default="ridge")
    xfamily_chain.add_argument("--init-activation-fit-factor-steps", type=int, default=0)
    xfamily_chain.add_argument("--init-activation-fit-factor-lr", type=float, default=3e-2)
    xfamily_chain.add_argument("--init-activation-fit-factor-l2", type=float, default=0.0)
    xfamily_chain.add_argument("--init-activation-fit-gate", choices=["none", "validation"], default="validation")
    xfamily_chain.add_argument("--init-activation-fit-validation-fraction", type=float, default=0.20)
    xfamily_chain.add_argument("--init-activation-fit-min-validation-rows", type=int, default=8)
    xfamily_chain.add_argument("--init-activation-fit-min-relative-improvement", type=float, default=0.0)
    xfamily_chain.add_argument("--init-activation-fit-max-cosine-drop", type=float, default=0.01)
    xfamily_chain.add_argument("--init-activation-feature-mode", choices=["stats", "subspace", "covariance"], default="stats")
    xfamily_chain.add_argument("--init-activation-subspace-rank", type=int, default=8)
    xfamily_chain.add_argument("--init-activation-domain-balance", choices=["none", "keyword"], default="none")
    xfamily_chain.add_argument(
        "--init-untie-output-head",
        action="store_true",
        help="Untie the target lm_head during donor-projected init so output rows can be folded separately from input embeddings.",
    )
    xfamily_chain.add_argument("--init-activation-max-length", type=int, default=128)
    xfamily_chain.add_argument("--block-steps-per-layer", type=int, default=4)
    xfamily_chain.add_argument("--block-batch-size", type=int, default=1)
    xfamily_chain.add_argument("--block-max-length", type=int, default=112)
    xfamily_chain.add_argument("--block-lr", type=float, default=2e-6)
    xfamily_chain.add_argument("--block-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-response-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-first-token-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-response-end-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-response-loss-mode", choices=["none", "auto", "assistant", "answer"], default="none")
    xfamily_chain.add_argument("--block-hidden-projection-scope", choices=["global", "layer-input"], default="layer-input")
    xfamily_chain.add_argument(
        "--block-hidden-metric",
        choices=[
            "projected",
            "cka",
            "both",
            "rsd",
            "projected-rsd",
            "gram",
            "projected-gram",
            "geometry",
            "projected-geometry",
        ],
        default="projected-rsd",
    )
    xfamily_chain.add_argument("--block-span-pair-mode", choices=["exact", "overlap", "both"], default="exact")
    xfamily_chain.add_argument("--block-target", choices=["hidden", "delta"], default="delta")
    xfamily_chain.add_argument(
        "--block-residual-magnitude-compensation",
        choices=RESIDUAL_MAGNITUDE_COMPENSATION_CHOICES,
        default="lowrank-pre",
    )
    xfamily_chain.add_argument("--block-residual-compensation-min-scale", type=float, default=0.50)
    xfamily_chain.add_argument("--block-residual-compensation-max-scale", type=float, default=1.50)
    xfamily_chain.add_argument("--block-residual-compensation-linear-ridge", type=float, default=1e-3)
    xfamily_chain.add_argument("--block-residual-compensation-alpha", type=float, default=0.01)
    xfamily_chain.add_argument("--block-residual-compensation-linear-rank", type=int, default=16)
    xfamily_chain.add_argument("--block-pre-reconstruction-target", choices=["none", "consumer-input", "block-interface", "all"], default="none")
    xfamily_chain.add_argument("--block-pre-reconstruction-ridge", type=float, default=1e-3)
    xfamily_chain.add_argument("--block-pre-reconstruction-alpha", type=float, default=0.05)
    xfamily_chain.add_argument(
        "--block-pre-reconstruction-objective",
        choices=["ridge", "cosine"],
        default="ridge",
    )
    xfamily_chain.add_argument("--block-pre-reconstruction-cosine-steps", type=int, default=40)
    xfamily_chain.add_argument("--block-pre-reconstruction-cosine-lr", type=float, default=1e-2)
    xfamily_chain.add_argument("--block-pre-reconstruction-cosine-l2", type=float, default=1e-4)
    xfamily_chain.add_argument("--block-pre-reconstruction-fit-gate", choices=["none", "validation"], default="validation")
    xfamily_chain.add_argument("--block-pre-reconstruction-validation-fraction", type=float, default=0.25)
    xfamily_chain.add_argument("--block-pre-reconstruction-min-validation-rows", type=int, default=32)
    xfamily_chain.add_argument("--block-pre-reconstruction-max-tokens-per-layer", type=int, default=512)
    xfamily_chain.add_argument("--block-pre-reconstruction-min-relative-improvement", type=float, default=0.0)
    xfamily_chain.add_argument("--block-pre-reconstruction-max-cosine-drop", type=float, default=0.01)
    xfamily_chain.add_argument("--block-donor-layer-fold-mode", choices=["representative", "average", "fold"], default="fold")
    xfamily_chain.add_argument(
        "--block-donor-layer-group-strategy",
        choices=["depth", "activation-energy", "activation-geometry", "activation-cka"],
        default="activation-energy",
    )
    xfamily_chain.add_argument("--block-layer-limit", type=int)
    xfamily_chain.add_argument("--block-layer-gate-metric", choices=["none", "ppl", "behavior"], default="behavior")
    xfamily_chain.add_argument("--block-layer-gate-max-texts", type=int, default=200)
    xfamily_chain.add_argument("--block-layer-gate-max-generation-prompts", type=int)
    xfamily_chain.add_argument("--block-layer-gate-max-behavior-drop", type=float, default=0.0)
    xfamily_chain.add_argument("--block-alm-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-projected-vocab-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-byte-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-block-io-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-on-policy-anchor-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-on-policy-unlikelihood-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-on-policy-hidden-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-on-policy-max-new-tokens", type=int, default=24)
    xfamily_chain.add_argument("--block-on-policy-interval", type=int, default=10)
    xfamily_chain.add_argument(
        "--block-on-policy-response-loss-mode",
        choices=["none", "auto", "assistant", "answer"],
        default="auto",
    )
    xfamily_chain.add_argument("--block-on-policy-unlikelihood-window", type=int, default=16)
    xfamily_chain.add_argument("--block-stepwise-donor-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-stepwise-donor-max-new-tokens", type=int, default=8)
    xfamily_chain.add_argument("--block-stepwise-donor-interval", type=int, default=4)
    xfamily_chain.add_argument("--block-teacher-generate-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-teacher-generate-hidden-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-teacher-generate-projected-vocab-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-teacher-generate-byte-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-teacher-generate-max-new-tokens", type=int, default=64)
    xfamily_chain.add_argument("--block-alm-max-chunks", type=int, default=64)
    xfamily_chain.add_argument("--block-alm-min-chars", type=int, default=2)
    xfamily_chain.add_argument("--block-focused-block-hidden-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-focused-projected-vocab-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-focused-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-token-focus", choices=["auto", "response", "numbers", "code", "all"], default="auto")
    xfamily_chain.add_argument(
        "--block-projected-vocab-kd-mode",
        choices=["teacher-to-student", "student-to-teacher", "both"],
        default="teacher-to-student",
    )
    xfamily_chain.add_argument("--block-byte-hidden-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-block-io-ridge", type=float, default=1e-3)
    xfamily_chain.add_argument("--block-block-io-loss", choices=["smooth-l1", "cosine", "both"], default="both")
    xfamily_chain.add_argument("--block-train-output-head-with-kd", action="store_true")
    xfamily_chain.add_argument("--block-untie-output-head-for-kd", action="store_true")
    xfamily_chain.add_argument("--block-allow-tied-output-head-training", action="store_true")
    xfamily_chain.add_argument("--block-module-output-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--block-module-output-targets", choices=["attention", "mlp", "both"], default="both")
    xfamily_chain.add_argument("--block-module-output-transport", choices=["paired", "sinkhorn", "both"], default="paired")
    xfamily_chain.add_argument("--block-module-output-temperature", type=float, default=1.0)
    xfamily_chain.add_argument("--block-module-output-sinkhorn-iterations", type=int, default=12)
    xfamily_chain.add_argument("--block-projected-vocab-max-source-tokens", type=int, default=8)
    xfamily_chain.add_argument("--block-save-best-layer-loss", action="store_true")
    xfamily_chain.add_argument("--block-save-best-layer-behavior", action="store_true")
    xfamily_chain.add_argument("--block-layer-behavior-eval-interval", type=int)
    xfamily_chain.add_argument("--block-layer-checkpoint-interval", type=int, default=0)
    xfamily_chain.add_argument("--block-calibration-order", choices=["original", "domain-balanced"], default="original")
    xfamily_chain.add_argument("--alm-steps", type=int, default=80)
    xfamily_chain.add_argument("--alm-batch-size", type=int, default=1)
    xfamily_chain.add_argument("--alm-max-length", type=int, default=112)
    xfamily_chain.add_argument("--alm-lr", type=float, default=2e-6)
    xfamily_chain.add_argument("--alm-anchor-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-first-token-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-response-end-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-focused-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-dual-space-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-attention-influence-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-attention-influence-layer", choices=["last", "middle"], default="last")
    xfamily_chain.add_argument("--alm-kd-weight", type=float, default=0.05)
    xfamily_chain.add_argument("--alm-projected-vocab-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-focused-projected-vocab-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-span-hidden-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-token-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-token-focus", choices=["auto", "response", "numbers", "code", "all"], default="auto")
    xfamily_chain.add_argument("--alm-token-max-pairs", type=int, default=512)
    xfamily_chain.add_argument("--alm-token-teacher-confidence-floor", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-answer-contrastive-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-answer-contrastive-max-negatives", type=int, default=4)
    xfamily_chain.add_argument("--alm-answer-first-token-margin-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-answer-first-token-margin", type=float, default=1.0)
    xfamily_chain.add_argument("--alm-answer-first-token-max-negatives", type=int, default=8)
    xfamily_chain.add_argument("--alm-answer-first-token-top-negatives", type=int, default=0)
    xfamily_chain.add_argument("--alm-teacher-rank-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-teacher-rank-temperature", type=float, default=1.0)
    xfamily_chain.add_argument("--alm-teacher-rank-max-negatives", type=int, default=4)
    xfamily_chain.add_argument("--alm-teacher-generate-lm-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-teacher-generate-max-new-tokens", type=int, default=64)
    xfamily_chain.add_argument("--alm-stepwise-donor-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-stepwise-donor-max-new-tokens", type=int, default=16)
    xfamily_chain.add_argument("--alm-on-policy-anchor-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-on-policy-unlikelihood-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-on-policy-hidden-kd-weight", type=float, default=0.0)
    xfamily_chain.add_argument("--alm-on-policy-max-new-tokens", type=int, default=24)
    xfamily_chain.add_argument("--alm-on-policy-interval", type=int, default=10)
    xfamily_chain.add_argument("--alm-train-scope", default="last4")
    xfamily_chain.add_argument("--alm-response-loss-mode", choices=["none", "auto", "assistant", "answer"], default="none")
    xfamily_chain.add_argument("--alm-append-eos", action="store_true")
    xfamily_chain.add_argument("--alm-save-best-behavior", action="store_true")
    xfamily_chain.add_argument("--alm-behavior-eval-interval", type=int)
    xfamily_chain.add_argument("--alm-calibration-order", choices=["original", "domain-balanced"], default="original")
    xfamily_chain.add_argument("--dtype", default="float16")
    xfamily_chain.add_argument("--candidate-dtype", default="float32")
    xfamily_chain.add_argument("--device", default="cuda:0")
    xfamily_chain.add_argument("--trust-remote-code", action="store_true")
    xfamily_chain.add_argument("--calibration-file")
    xfamily_chain.add_argument("--max-calibration-texts", type=int)
    xfamily_chain.set_defaults(func=cmd_hf_xfamily_chain_anchor_compress)

    lm_recover = sub.add_parser("hf-lm-recover", help="Run teacherless LM recovery on a folded HF causal LM.")
    lm_recover.add_argument("candidate_model")
    lm_recover.add_argument("output")
    lm_recover.add_argument("--steps", type=int, default=80)
    lm_recover.add_argument("--batch-size", type=int, default=1)
    lm_recover.add_argument("--max-length", type=int, default=96)
    lm_recover.add_argument("--lr", type=float, default=5e-5)
    lm_recover.add_argument("--first-token-lm-weight", type=float, default=0.0)
    lm_recover.add_argument("--answer-contrastive-weight", type=float, default=0.0)
    lm_recover.add_argument("--answer-contrastive-max-negatives", type=int, default=4)
    lm_recover.add_argument("--unlikelihood-weight", type=float, default=0.0)
    lm_recover.add_argument("--unlikelihood-window", type=int, default=16)
    lm_recover.add_argument("--entropy-weight", type=float, default=0.0)
    lm_recover.add_argument("--response-loss-mode", choices=["none", "auto", "assistant", "answer"], default="none")
    lm_recover.add_argument("--append-eos", action="store_true")
    lm_recover.add_argument("--train-scope", default="all")
    lm_recover.add_argument("--dtype", default="float32")
    lm_recover.add_argument("--device", default="cuda:0")
    lm_recover.add_argument("--trust-remote-code", action="store_true")
    lm_recover.add_argument("--grad-clip", type=float, default=1.0)
    lm_recover.add_argument("--report-interval", type=int)
    lm_recover.add_argument("--save-best-loss", action="store_true")
    lm_recover.add_argument(
        "--save-best-behavior",
        action="store_true",
        help="Select the saved recovery checkpoint by behavior probes instead of loss.",
    )
    lm_recover.add_argument("--behavior-eval-interval", type=int)
    lm_recover.add_argument("--behavior-max-new-tokens", type=int, default=96)
    lm_recover.add_argument("--behavior-max-probes", type=int, default=6)
    lm_recover.add_argument("--behavior-eval-suite")
    lm_recover.add_argument("--behavior-max-generation-prompts", type=int)
    lm_recover.add_argument("--calibration-file")
    lm_recover.add_argument("--max-calibration-texts", type=int)
    lm_recover.set_defaults(func=cmd_hf_lm_recover)

    xfamily_head = sub.add_parser(
        "hf-xfamily-calibrate-head",
        help="Fit a cross-tokenizer child LM head to donor projected next-token logits.",
    )
    xfamily_head.add_argument("source_model")
    xfamily_head.add_argument("candidate_model")
    xfamily_head.add_argument("output")
    xfamily_head.add_argument("--ridge", type=float, default=1e-2)
    xfamily_head.add_argument("--head-blend-alpha", type=float, default=1.0)
    xfamily_head.add_argument("--head-calibration-mode", choices=["full", "mean-shift", "answer-margin"], default="full")
    xfamily_head.add_argument("--response-loss-mode", choices=["none", "auto", "assistant", "answer"], default="auto")
    xfamily_head.add_argument("--answer-first-token-margin", type=float, default=1.0)
    xfamily_head.add_argument("--answer-first-token-max-negatives", type=int, default=8)
    xfamily_head.add_argument("--answer-first-token-top-negatives", type=int, default=0)
    xfamily_head.add_argument("--answer-margin-steps", type=int, default=3)
    xfamily_head.add_argument("--max-length", type=int, default=128)
    xfamily_head.add_argument("--max-calibration-rows", type=int, default=2048)
    xfamily_head.add_argument("--projected-vocab-max-source-tokens", type=int, default=6)
    xfamily_head.add_argument("--dtype", default="float16")
    xfamily_head.add_argument("--candidate-dtype", default="float32")
    xfamily_head.add_argument("--device", default="cuda:0")
    xfamily_head.add_argument("--trust-remote-code", action="store_true")
    xfamily_head.add_argument("--calibration-file")
    xfamily_head.add_argument("--max-calibration-texts", type=int)
    xfamily_head.set_defaults(func=cmd_hf_xfamily_calibrate_head)

    vision = sub.add_parser("vision-probe", help="Run simple image-description/OCR probes for an HF vision-language model.")
    vision.add_argument("model")
    vision.add_argument("output")
    vision.add_argument("--image-dir", default="runs/vision-images")
    vision.add_argument("--dtype", default="float16")
    vision.add_argument("--device", default="cuda:0")
    vision.add_argument("--trust-remote-code", action="store_true")
    vision.add_argument("--max-new-tokens", type=int, default=80)
    vision.set_defaults(func=cmd_vision_probe)

    vision_chain = sub.add_parser("vision-chain", help="Chain a small VLM observation into a text model answer.")
    vision_chain.add_argument("output")
    vision_chain.add_argument("--vision-model", required=True)
    vision_chain.add_argument("--text-model", required=True)
    vision_chain.add_argument("--image-dir", default="runs/vision-chain-images")
    vision_chain.add_argument("--dtype", default="float16")
    vision_chain.add_argument("--device", default="cuda:0")
    vision_chain.add_argument("--trust-remote-code", action="store_true")
    vision_chain.add_argument("--max-new-tokens", type=int, default=80)
    vision_chain.set_defaults(func=cmd_vision_chain)

    native_vlm = sub.add_parser("native-vlm-demo", help="Train/evaluate a tiny native raw-patch + text-token transformer.")
    native_vlm.add_argument("output_dir")
    native_vlm.add_argument("--steps", type=int, default=600)
    native_vlm.add_argument("--batch-size", type=int, default=64)
    native_vlm.add_argument("--eval-batches", type=int, default=16)
    native_vlm.add_argument("--seed", type=int, default=7)
    native_vlm.add_argument("--device", default="cpu")
    native_vlm.add_argument("--d-model", type=int, default=96)
    native_vlm.add_argument("--layers", type=int, default=3)
    native_vlm.add_argument("--heads", type=int, default=4)
    native_vlm.add_argument("--lr", type=float, default=3e-4)
    native_vlm.set_defaults(func=cmd_native_vlm_demo)

    hf_vision_graft = sub.add_parser("hf-vision-graft", help="Train/evaluate a raw-patch vision graft on an HF causal LM parent.")
    hf_vision_graft.add_argument("model")
    hf_vision_graft.add_argument("output_dir")
    hf_vision_graft.add_argument("--steps", type=int, default=300)
    hf_vision_graft.add_argument("--batch-size", type=int, default=8)
    hf_vision_graft.add_argument("--eval-batches", type=int, default=8)
    hf_vision_graft.add_argument("--seed", type=int, default=11)
    hf_vision_graft.add_argument("--device", default="cuda:0")
    hf_vision_graft.add_argument("--dtype", default="float16")
    hf_vision_graft.add_argument("--lr", type=float, default=3e-3)
    hf_vision_graft.add_argument("--train-last-layers", type=int, default=0)
    hf_vision_graft.add_argument("--trust-remote-code", action="store_true")
    hf_vision_graft.add_argument("--max-new-tokens", type=int, default=6)
    hf_vision_graft.add_argument("--task-keys", default=None, help="Comma-separated subset: square_color,circle_color,digit.")
    hf_vision_graft.add_argument("--grad-clip", type=float, default=1.0)
    hf_vision_graft.add_argument("--text-retention-file")
    hf_vision_graft.add_argument("--text-retention-weight", type=float, default=0.0)
    hf_vision_graft.add_argument("--text-retention-batch-size", type=int, default=2)
    hf_vision_graft.add_argument("--text-retention-max-length", type=int, default=128)
    hf_vision_graft.set_defaults(func=cmd_hf_vision_graft)

    return parser


def cmd_toy(args: argparse.Namespace) -> dict[str, object]:
    model = random_toy_model(parse_widths(args.widths), seed=args.seed, activation=args.activation)
    save_model(model, args.output)
    return {"output": str(Path(args.output)), "model": model.summary()}


def cmd_inspect(args: argparse.Namespace) -> dict[str, object]:
    return load_model(args.model).summary()


def cmd_shrink(args: argparse.Namespace) -> dict[str, object]:
    model = load_model(args.input)
    hidden_widths = _resolve_widths(model, args.widths, args.target_params, args.min_width)
    output = resize_to_widths(model, hidden_widths)
    save_model(output, args.output)
    return {"input": model.summary(), "output": output.summary()}


def cmd_expand(args: argparse.Namespace) -> dict[str, object]:
    model = load_model(args.input)
    output = expand_depth_identity(model, args.add_layers)
    if args.widths:
        hidden_widths = parse_widths(args.widths)
    elif args.target_params:
        hidden_widths = plan_widths_for_target_params(output, args.target_params).hidden_widths
    else:
        hidden_widths = output.hidden_widths
    output = resize_to_widths(output, hidden_widths, seed=args.seed, noise_std=args.noise_std)
    save_model(output, args.output)
    return {"input": model.summary(), "output": output.summary()}


def cmd_merge(args: argparse.Namespace) -> dict[str, object]:
    models = [load_model(path) for path in args.models]
    if args.widths:
        hidden_widths = parse_widths(args.widths)
    elif args.target_params:
        base = models[0]
        hidden_widths = plan_widths_for_target_params(base, args.target_params).hidden_widths
    else:
        hidden_widths = None
    weights = None if args.weights is None else [float(part.strip()) for part in args.weights.split(",") if part.strip()]
    output = merge_models(
        models,
        target_hidden_widths=hidden_widths,
        weights=weights,
        method=args.method,
        align=not args.no_align,
        seed=args.seed,
        noise_std=args.noise_std,
    )
    save_model(output, args.output)
    return {"inputs": [model.summary() for model in models], "output": output.summary()}


def cmd_hf_merge(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import same_shape_merge

    weights = None if args.weights is None else [float(part.strip()) for part in args.weights.split(",") if part.strip()]
    return same_shape_merge(
        model_ids=args.models,
        output_dir=args.output,
        weights=weights,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_hf_task_merge(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import task_arithmetic_merge

    return task_arithmetic_merge(
        base_model=args.base_model,
        tuned_model=args.tuned_model,
        output_dir=args.output,
        alpha=args.alpha,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_hf_resize(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import resize_gated_ffn

    return resize_gated_ffn(
        model_id_or_path=args.model,
        output_dir=args.output,
        ratio=args.ratio,
        target_intermediate_size=args.target_intermediate_size,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        assignment=args.assignment,
    )


def cmd_hf_repair(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import repair_gated_ffn_with_calibration

    return repair_gated_ffn_with_calibration(
        source_model=args.source_model,
        resized_model=args.resized_model,
        output_dir=args.output,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        ridge=args.ridge,
        max_length=args.max_length,
        max_tokens_per_layer=args.max_tokens_per_layer,
    )


def cmd_hf_merge_layers(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import merge_decoder_layers

    return merge_decoder_layers(
        model_id_or_path=args.model,
        output_dir=args.output,
        groups=_parse_layer_groups(args.groups),
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        method=args.method,
    )


def cmd_hf_rank_layer_folds(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import rank_layer_fold_candidates

    return rank_layer_fold_candidates(
        model_id_or_path=args.model,
        output_file=args.output,
        window=args.window,
        top_k=args.top_k,
        max_length=args.max_length,
        max_tokens_per_layer=args.max_tokens_per_layer,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
    )


def cmd_hf_drop_layers(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import drop_decoder_layers

    layers = [int(part.strip()) for part in args.layers.split(",") if part.strip()]
    return drop_decoder_layers(
        model_id_or_path=args.model,
        output_dir=args.output,
        drop_layers=layers,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_hf_eval(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import evaluate_text_model

    return evaluate_text_model(
        model_id_or_path=args.model,
        output_file=args.output,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )


def cmd_hf_build_eval_suite(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import build_eval_suite

    return build_eval_suite(
        output_file=args.output,
        behavior_count=args.behavior_count,
        generation_count=args.generation_count,
        choice_count=args.choice_count,
        text_count=args.text_count,
        seed=args.seed,
    )


def cmd_hf_build_recovery_calibration(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import build_recovery_calibration

    return build_recovery_calibration(
        output_file=args.output,
        math_count=args.math_count,
        code_count=args.code_count,
        tokenizer_count=args.tokenizer_count,
        model_surgery_count=args.model_surgery_count,
        commonsense_count=args.commonsense_count,
        seed=args.seed,
        exclude_eval_suite=args.exclude_eval_suite,
    )


def cmd_hf_tie_head(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import tie_output_head_checkpoint

    return tie_output_head_checkpoint(
        model_id_or_path=args.model,
        output_dir=args.output,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_hf_gate(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import gate_text_report

    return gate_text_report(
        baseline_report=args.baseline_report,
        candidate_report=args.candidate_report,
        output_file=args.output,
        max_ppl_ratio=args.max_ppl_ratio,
        ppl_gate_mode=args.ppl_gate_mode,
        max_accuracy_drop=args.max_accuracy_drop,
        max_behavior_drop=args.max_behavior_drop,
        min_choice_accuracy=args.min_choice_accuracy,
        min_behavior_pass_rate=args.min_behavior_pass_rate,
        min_continuation_chars=args.min_continuation_chars,
        min_unique_token_ratio=args.min_unique_token_ratio,
    )


def cmd_hf_rank_evals(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import rank_evaluation_reports

    reports = []
    for item in args.reports:
        matches = sorted(glob.glob(item))
        reports.extend(matches if matches else [item])
    return rank_evaluation_reports(
        reports=reports,
        output_file=args.output,
        min_continuation_chars=args.min_continuation_chars,
        min_unique_token_ratio=args.min_unique_token_ratio,
    )


def cmd_hf_sweep(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import sweep_gated_ffn_resize

    ratios = [float(part.strip()) for part in args.ratios.split(",") if part.strip()]
    return sweep_gated_ffn_resize(
        source_model=args.source_model,
        output_dir=args.output_dir,
        ratios=ratios,
        baseline_report=args.baseline_report,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        max_ppl_ratio=args.max_ppl_ratio,
        max_accuracy_drop=args.max_accuracy_drop,
        min_choice_accuracy=args.min_choice_accuracy,
        repair_shrinks=args.repair_shrinks,
        repair_ridge=args.repair_ridge,
        repair_max_tokens=args.repair_max_tokens,
        assignment=args.assignment,
    )


def cmd_hf_search_compress(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import search_merge_first_compression

    width_ratios = None
    if args.width_ratios:
        width_ratios = [float(part.strip()) for part in args.width_ratios.split(",") if part.strip()]
    layer_group_specs = [_parse_layer_groups(spec) for spec in args.layer_groups]
    depth_methods = [part.strip() for part in args.depth_methods.split(",") if part.strip()]
    return search_merge_first_compression(
        source_model=args.source_model,
        output_dir=args.output_dir,
        width_ratios=width_ratios,
        layer_group_specs=layer_group_specs,
        depth_methods=depth_methods,
        baseline_report=args.baseline_report,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        max_ppl_ratio=args.max_ppl_ratio,
        max_accuracy_drop=args.max_accuracy_drop,
        min_choice_accuracy=args.min_choice_accuracy,
        assignment=args.assignment,
        include_combinations=not args.no_combinations,
        target_shrink=args.target_shrink,
    )


def cmd_hf_task_sweep(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import sweep_task_arithmetic

    alphas = [float(part.strip()) for part in args.alphas.split(",") if part.strip()]
    return sweep_task_arithmetic(
        base_model=args.base_model,
        tuned_model=args.tuned_model,
        output_dir=args.output_dir,
        alphas=alphas,
        baseline_report=args.baseline_report,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        max_ppl_ratio=args.max_ppl_ratio,
        max_accuracy_drop=args.max_accuracy_drop,
        min_choice_accuracy=args.min_choice_accuracy,
    )


def cmd_hf_random_init(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import random_init_causal_lm

    return random_init_causal_lm(
        config_model=args.config_model,
        tokenizer_model=args.tokenizer_model,
        output_dir=args.output,
        seed=args.seed,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_hf_donor_project_init(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_xfamily import donor_projected_init
    from neural_forge.hf_xfamily import multi_donor_projected_init

    activation_calibration_texts = _load_recovery_calibration_texts(
        args.activation_calibration_file,
        args.activation_max_calibration_texts,
    )
    donor_models = [part.strip() for part in str(args.donor_model).split(",") if part.strip()]
    if not donor_models:
        raise ValueError("--donor-model must include at least one donor.")
    donor_weights = None
    if args.donor_weights:
        donor_weights = [float(part.strip()) for part in str(args.donor_weights).split(",") if part.strip()]
    common_kwargs = {
        "target_config_model": args.target_config_model,
        "target_base_model": args.target_base_model,
        "target_tokenizer_model": args.target_tokenizer_model,
        "output_dir": args.output,
        "seed": args.seed,
        "ridge": args.ridge,
        "max_anchors": args.max_anchors,
        "fold_mode": args.fold_mode,
        "hidden_basis_mode": args.hidden_basis_mode,
        "hidden_basis_scope": args.hidden_basis_scope,
        "hidden_basis_max_rows": args.hidden_basis_max_rows,
        "hidden_basis_preprocess": args.hidden_basis_preprocess,
        "activation_fold_assignment": args.activation_fold_assignment,
        "fold_stat_repair": args.fold_stat_repair,
        "transport_backend": args.transport_backend,
        "transport_support": args.transport_support,
        "donor_layer_fold_mode": args.donor_layer_fold_mode,
        "donor_layer_group_strategy": args.donor_layer_group_strategy,
        "attention_fold_mode": args.attention_fold_mode,
        "attention_head_dim_transport": args.attention_head_dim_transport,
        "token_init_mode": args.token_init_mode,
        "projection_scale": args.projection_scale,
        "projection_blend_alpha": args.projection_blend_alpha,
        "consumer_hidden_reduction": args.consumer_hidden_reduction,
        "activation_fit_linears": args.activation_fit_linears,
        "activation_fit_target": args.activation_fit_target,
        "activation_fit_max_rows": args.activation_fit_max_rows,
        "activation_fit_alpha": args.activation_fit_alpha,
        "activation_fit_ridge": args.activation_fit_ridge,
        "activation_fit_rank": args.activation_fit_rank,
        "activation_fit_rank_mode": args.activation_fit_rank_mode,
        "activation_fit_method": args.activation_fit_method,
        "activation_fit_factor_steps": args.activation_fit_factor_steps,
        "activation_fit_factor_lr": args.activation_fit_factor_lr,
        "activation_fit_factor_l2": args.activation_fit_factor_l2,
        "activation_fit_gate": args.activation_fit_gate,
        "activation_fit_validation_fraction": args.activation_fit_validation_fraction,
        "activation_fit_min_validation_rows": args.activation_fit_min_validation_rows,
        "activation_fit_min_relative_improvement": args.activation_fit_min_relative_improvement,
        "activation_fit_max_cosine_drop": args.activation_fit_max_cosine_drop,
        "activation_feature_mode": args.activation_feature_mode,
        "activation_subspace_rank": args.activation_subspace_rank,
        "activation_domain_balance": args.activation_domain_balance,
        "untie_output_head": args.untie_output_head,
        "activation_calibration_texts": activation_calibration_texts,
        "activation_max_length": args.activation_max_length,
        "activation_device": args.activation_device,
        "neutralize_unmatched_operators": args.neutralize_unmatched_operators,
        "torch_dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if len(donor_models) > 1 or donor_weights is not None:
        init_report = multi_donor_projected_init(
            donor_models=donor_models,
            donor_weights=donor_weights,
            **common_kwargs,
        )
    else:
        init_report = donor_projected_init(
            donor_model=donor_models[0],
            **common_kwargs,
        )
    if not args.post_init_eval_suite:
        return init_report

    from neural_forge.hf_tools import evaluate_text_model

    eval_output = (
        Path(args.post_init_eval_output)
        if args.post_init_eval_output
        else Path(args.output) / "neural_forge_post_init_eval_report.json"
    )
    eval_report = evaluate_text_model(
        model_id_or_path=args.output,
        output_file=eval_output,
        torch_dtype=args.post_init_eval_dtype or args.dtype,
        device=args.post_init_eval_device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.post_init_max_new_tokens,
        eval_suite=args.post_init_eval_suite,
        max_text_evals=args.post_init_max_text_evals,
        max_choice_probes=args.post_init_max_choice_probes,
        max_generation_prompts=args.post_init_max_generation_prompts,
        max_behavior_probes=args.post_init_max_behavior_probes,
    )
    eval_summary = _compact_eval_report(eval_report)
    return {
        "operation": "donor_projected_init_with_post_init_eval",
        "init_report": init_report,
        "post_init_eval_report": str(eval_output),
        "post_init_eval_summary": eval_summary,
        "post_init_gate": _post_init_gate_decision(
            eval_summary,
            min_behavior_pass_rate=args.post_init_min_behavior_pass_rate,
            min_generation_pass_rate=args.post_init_min_generation_pass_rate,
            min_choice_accuracy=args.post_init_min_choice_accuracy,
            max_ppl=args.post_init_max_ppl,
        ),
    }


def _compact_eval_report(report: dict[str, object]) -> dict[str, object]:
    choice = report.get("choice_accuracy", {})
    behavior = report.get("behavior_suite", {})
    generation = report.get("generation_health", {})
    perplexity = report.get("perplexity", {})
    if not isinstance(choice, dict):
        choice = {}
    if not isinstance(behavior, dict):
        behavior = {}
    if not isinstance(generation, dict):
        generation = {}
    if not isinstance(perplexity, dict):
        perplexity = {}
    return {
        "param_count": report.get("param_count"),
        "perplexity_mean": perplexity.get("mean"),
        "choice_correct": choice.get("correct"),
        "choice_total": choice.get("total"),
        "choice_accuracy": choice.get("accuracy"),
        "behavior_correct": behavior.get("correct"),
        "behavior_total": behavior.get("total"),
        "behavior_pass_rate": behavior.get("pass_rate"),
        "generation_correct": generation.get("correct"),
        "generation_total": generation.get("total"),
        "generation_pass_rate": generation.get("pass_rate"),
    }


def _post_init_gate_decision(
    summary: dict[str, object],
    min_behavior_pass_rate: float | None = None,
    min_generation_pass_rate: float | None = None,
    min_choice_accuracy: float | None = None,
    max_ppl: float | None = None,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def value_float(name: str) -> float | None:
        value = summary.get(name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def add_min_check(name: str, metric: str, threshold: float | None) -> None:
        if threshold is None:
            return
        value = value_float(metric)
        checks.append(
            {
                "name": name,
                "metric": metric,
                "value": value,
                "threshold": float(threshold),
                "passed": value is not None and value >= float(threshold),
                "direction": "min",
            }
        )

    def add_max_check(name: str, metric: str, threshold: float | None) -> None:
        if threshold is None:
            return
        value = value_float(metric)
        checks.append(
            {
                "name": name,
                "metric": metric,
                "value": value,
                "threshold": float(threshold),
                "passed": value is not None and value <= float(threshold),
                "direction": "max",
            }
        )

    add_min_check("behavior_pass_rate", "behavior_pass_rate", min_behavior_pass_rate)
    add_min_check("generation_pass_rate", "generation_pass_rate", min_generation_pass_rate)
    add_min_check("choice_accuracy", "choice_accuracy", min_choice_accuracy)
    add_max_check("perplexity_mean", "perplexity_mean", max_ppl)
    return {
        "active": bool(checks),
        "passed": all(bool(check["passed"]) for check in checks) if checks else True,
        "checks": checks,
    }


def cmd_hf_xfamily_vocab_merge(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_xfamily import cross_family_vocab_merge

    return cross_family_vocab_merge(
        host_model=args.host_model,
        donor_model=args.donor_model,
        output_dir=args.output,
        alpha=args.alpha,
        ridge=args.ridge,
        max_anchors=args.max_anchors,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_hf_xfamily_sweep(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_xfamily import sweep_cross_family_vocab_merge

    alphas = [float(part.strip()) for part in args.alphas.split(",") if part.strip()]
    return sweep_cross_family_vocab_merge(
        host_model=args.host_model,
        donor_model=args.donor_model,
        output_dir=args.output_dir,
        alphas=alphas,
        baseline_report=args.baseline_report,
        ridge=args.ridge,
        max_anchors=args.max_anchors,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        max_ppl_ratio=args.max_ppl_ratio,
        max_accuracy_drop=args.max_accuracy_drop,
        min_choice_accuracy=args.min_choice_accuracy,
    )


def cmd_hf_xfamily_mlp_merge(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_xfamily import cross_family_mlp_merge

    activation_calibration_texts = _load_recovery_calibration_texts(
        args.activation_calibration_file,
        args.activation_max_calibration_texts,
    )
    return cross_family_mlp_merge(
        host_model=args.host_model,
        donor_model=args.donor_model,
        output_dir=args.output,
        alpha=args.alpha,
        vocab_alpha=args.vocab_alpha,
        ridge=args.ridge,
        max_anchors=args.max_anchors,
        max_host_channels=args.max_host_channels,
        max_donor_channels=args.max_donor_channels,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        channel_assignment=args.channel_assignment,
        activation_calibration_texts=activation_calibration_texts,
        activation_max_length=args.activation_max_length,
        activation_device=args.activation_device,
        layer_alpha_mode=args.layer_alpha_mode,
        layer_alpha_min_scale=args.layer_alpha_min_scale,
        layer_alpha_max_scale=args.layer_alpha_max_scale,
        layer_alpha_similarity_floor=args.layer_alpha_similarity_floor,
    )


def cmd_hf_xfamily_mlp_sweep(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_xfamily import sweep_cross_family_mlp_merge

    alphas = [float(part.strip()) for part in args.alphas.split(",") if part.strip()]
    recovery_calibration_texts = _load_recovery_calibration_texts(
        args.recover_calibration_file,
        args.recover_max_calibration_texts,
    )
    activation_calibration_texts = _load_recovery_calibration_texts(
        args.activation_calibration_file,
        args.activation_max_calibration_texts,
    )
    return sweep_cross_family_mlp_merge(
        host_model=args.host_model,
        donor_model=args.donor_model,
        output_dir=args.output_dir,
        alphas=alphas,
        baseline_report=args.baseline_report,
        vocab_alpha=args.vocab_alpha,
        ridge=args.ridge,
        max_anchors=args.max_anchors,
        max_host_channels=args.max_host_channels,
        max_donor_channels=args.max_donor_channels,
        channel_assignment=args.channel_assignment,
        activation_calibration_texts=activation_calibration_texts,
        activation_max_length=args.activation_max_length,
        activation_device=args.activation_device,
        layer_alpha_mode=args.layer_alpha_mode,
        layer_alpha_min_scale=args.layer_alpha_min_scale,
        layer_alpha_max_scale=args.layer_alpha_max_scale,
        layer_alpha_similarity_floor=args.layer_alpha_similarity_floor,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        max_ppl_ratio=args.max_ppl_ratio,
        max_accuracy_drop=args.max_accuracy_drop,
        min_choice_accuracy=args.min_choice_accuracy,
        recover_candidates=args.recover,
        recover_only_failed=args.recover_only_failed,
        recovery_steps=args.recover_steps,
        recovery_batch_size=args.recover_batch_size,
        recovery_max_length=args.recover_max_length,
        recovery_lr=args.recover_lr,
        recovery_kd_weight=args.recover_kd_weight,
        recovery_lm_weight=args.recover_lm_weight,
        recovery_temperature=args.recover_temperature,
        recovery_train_scope=args.recover_train_scope,
        recovery_dtype=args.recover_dtype,
        recovery_grad_clip=args.recover_grad_clip,
        recovery_calibration_texts=recovery_calibration_texts,
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )


def cmd_hf_xfamily_compress(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_xfamily import cross_family_merge_compress_pipeline

    width_ratios = None
    if args.width_ratios:
        width_ratios = [float(part.strip()) for part in args.width_ratios.split(",") if part.strip()]
    layer_group_specs = [_parse_layer_groups(spec) for spec in args.layer_groups]
    depth_methods = [part.strip() for part in args.depth_methods.split(",") if part.strip()]
    recovery_calibration_texts = _load_recovery_calibration_texts(
        args.recover_calibration_file,
        args.recover_max_calibration_texts,
    )
    activation_calibration_texts = _load_recovery_calibration_texts(
        args.activation_calibration_file,
        args.activation_max_calibration_texts,
    )
    return cross_family_merge_compress_pipeline(
        host_model=args.host_model,
        donor_model=args.donor_model,
        output_dir=args.output_dir,
        alpha=args.alpha,
        vocab_alpha=args.vocab_alpha,
        ridge=args.ridge,
        max_anchors=args.max_anchors,
        max_host_channels=args.max_host_channels,
        max_donor_channels=args.max_donor_channels,
        channel_assignment=args.channel_assignment,
        activation_calibration_texts=activation_calibration_texts,
        activation_max_length=args.activation_max_length,
        activation_device=args.activation_device,
        layer_alpha_mode=args.layer_alpha_mode,
        layer_alpha_min_scale=args.layer_alpha_min_scale,
        layer_alpha_max_scale=args.layer_alpha_max_scale,
        layer_alpha_similarity_floor=args.layer_alpha_similarity_floor,
        width_ratios=width_ratios,
        layer_group_specs=layer_group_specs,
        depth_methods=depth_methods,
        target_shrink=args.target_shrink,
        assignment=args.assignment,
        include_combinations=not args.no_combinations,
        baseline_report=args.baseline_report,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        max_ppl_ratio=args.max_ppl_ratio,
        max_accuracy_drop=args.max_accuracy_drop,
        min_choice_accuracy=args.min_choice_accuracy,
        recover_candidates=args.recover,
        recover_only_failed=args.recover_only_failed,
        recover_top_k=args.recover_top_k,
        recovery_steps=args.recover_steps,
        recovery_batch_size=args.recover_batch_size,
        recovery_max_length=args.recover_max_length,
        recovery_lr=args.recover_lr,
        recovery_kd_weight=args.recover_kd_weight,
        recovery_lm_weight=args.recover_lm_weight,
        recovery_temperature=args.recover_temperature,
        recovery_train_scope=args.recover_train_scope,
        recovery_dtype=args.recover_dtype,
        recovery_grad_clip=args.recover_grad_clip,
        recovery_calibration_texts=recovery_calibration_texts,
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )


def cmd_hf_recover(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_recovery import recover_causal_lm

    return recover_causal_lm(
        source_model=args.source_model,
        candidate_model=args.candidate_model,
        output_dir=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        kd_weight=args.kd_weight,
        lm_weight=args.lm_weight,
        hidden_kd_weight=args.hidden_kd_weight,
        hidden_kd_metric=args.hidden_kd_metric,
        hidden_layer=args.hidden_layer,
        hidden_token_mode=args.hidden_token_mode,
        hidden_ridge=args.hidden_ridge,
        hidden_max_anchors=args.hidden_max_anchors,
        top_k_kd_weight=args.top_k_kd_weight,
        top_k_kd_k=args.top_k_kd_k,
        first_token_top_k_kd_weight=args.first_token_top_k_kd_weight,
        first_token_kd_weight=args.first_token_kd_weight,
        first_token_lm_weight=args.first_token_lm_weight,
        unlikelihood_weight=args.unlikelihood_weight,
        unlikelihood_window=args.unlikelihood_window,
        entropy_weight=args.entropy_weight,
        temperature=args.temperature,
        train_scope=args.train_scope,
        torch_dtype=args.dtype,
        candidate_train_dtype=args.candidate_train_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
        response_loss_mode=args.response_loss_mode,
        append_eos=args.append_eos,
        grad_clip=args.grad_clip,
        report_interval=args.report_interval,
        save_best_loss=args.save_best_loss,
        save_best_behavior=args.save_best_behavior,
        behavior_eval_interval=args.behavior_eval_interval,
        behavior_max_new_tokens=args.behavior_max_new_tokens,
        behavior_max_probes=args.behavior_max_probes,
        behavior_eval_suite=args.behavior_eval_suite,
        behavior_max_generation_prompts=args.behavior_max_generation_prompts,
    )


def cmd_hf_xfamily_recover(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_recovery import recover_cross_tokenizer_causal_lm

    return recover_cross_tokenizer_causal_lm(
        source_model=args.source_model,
        candidate_model=args.candidate_model,
        output_dir=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        anchor_kd_weight=args.anchor_kd_weight,
        lm_weight=args.lm_weight,
        first_token_lm_weight=args.first_token_lm_weight,
        response_end_lm_weight=args.response_end_lm_weight,
        focused_lm_weight=args.focused_lm_weight,
        hidden_kd_weight=args.hidden_kd_weight,
        span_kd_weight=args.span_kd_weight,
        span_hidden_kd_weight=args.span_hidden_kd_weight,
        dual_space_kd_weight=args.dual_space_kd_weight,
        attention_influence_weight=args.attention_influence_weight,
        attention_influence_layer=args.attention_influence_layer,
        projected_vocab_kd_weight=args.projected_vocab_kd_weight,
        focused_projected_vocab_kd_weight=args.focused_projected_vocab_kd_weight,
        projected_vocab_kd_mode=args.projected_vocab_kd_mode,
        projected_vocab_max_source_tokens=args.projected_vocab_max_source_tokens,
        reference_model=args.reference_model,
        reference_kd_weight=args.reference_kd_weight,
        unlikelihood_weight=args.unlikelihood_weight,
        unlikelihood_window=args.unlikelihood_window,
        entropy_weight=args.entropy_weight,
        response_loss_mode=args.response_loss_mode,
        token_kd_weight=args.token_kd_weight,
        token_focus=args.token_focus,
        token_max_pairs=args.token_max_pairs,
        token_teacher_confidence_floor=args.token_teacher_confidence_floor,
        alm_kd_weight=args.alm_kd_weight,
        alm_max_chunks=args.alm_max_chunks,
        alm_min_chars=args.alm_min_chars,
        answer_contrastive_weight=args.answer_contrastive_weight,
        answer_contrastive_max_negatives=args.answer_contrastive_max_negatives,
        answer_first_token_margin_weight=args.answer_first_token_margin_weight,
        answer_first_token_margin=args.answer_first_token_margin,
        answer_first_token_max_negatives=args.answer_first_token_max_negatives,
        answer_first_token_top_negatives=args.answer_first_token_top_negatives,
        teacher_rank_weight=args.teacher_rank_weight,
        teacher_rank_temperature=args.teacher_rank_temperature,
        teacher_rank_max_negatives=args.teacher_rank_max_negatives,
        teacher_generate_lm_weight=args.teacher_generate_lm_weight,
        teacher_generate_max_new_tokens=args.teacher_generate_max_new_tokens,
        stepwise_donor_kd_weight=args.stepwise_donor_kd_weight,
        stepwise_donor_max_new_tokens=args.stepwise_donor_max_new_tokens,
        on_policy_anchor_kd_weight=args.on_policy_anchor_kd_weight,
        on_policy_unlikelihood_weight=args.on_policy_unlikelihood_weight,
        on_policy_hidden_kd_weight=args.on_policy_hidden_kd_weight,
        on_policy_max_new_tokens=args.on_policy_max_new_tokens,
        on_policy_interval=args.on_policy_interval,
        append_eos=args.append_eos,
        entropy_aware_kd_weight=args.entropy_aware_kd_weight,
        entropy_aware_kd_threshold=args.entropy_aware_kd_threshold,
        entropy_aware_kd_sharpness=args.entropy_aware_kd_sharpness,
        span_max_pairs=args.span_max_pairs,
        hidden_ridge=args.hidden_ridge,
        hidden_layer=args.hidden_layer,
        hidden_kd_metric=args.hidden_kd_metric,
        hidden_max_anchors=args.hidden_max_anchors,
        temperature=args.temperature,
        max_anchors=args.max_anchors,
        anchor_selection=args.anchor_selection,
        anchor_min_chars=args.anchor_min_chars,
        train_scope=args.train_scope,
        torch_dtype=args.dtype,
        candidate_torch_dtype=args.candidate_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
        calibration_order=args.calibration_order,
        grad_clip=args.grad_clip,
        report_interval=args.report_interval,
        save_best_loss=args.save_best_loss,
        save_best_behavior=args.save_best_behavior,
        behavior_eval_interval=args.behavior_eval_interval,
        behavior_max_new_tokens=args.behavior_max_new_tokens,
        behavior_max_probes=args.behavior_max_probes,
        behavior_eval_suite=args.behavior_eval_suite,
        behavior_max_generation_prompts=args.behavior_max_generation_prompts,
    )


def cmd_hf_reconstruct_folded(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_recovery import reconstruct_folded_causal_lm

    return reconstruct_folded_causal_lm(
        source_model=args.source_model,
        candidate_model=args.candidate_model,
        output_dir=args.output,
        layer_merge_report=args.layer_merge_report,
        ridge=args.ridge,
        reconstruction_alpha=args.reconstruction_alpha,
        reconstruction_target=args.reconstruction_target,
        reconstruction_objective=args.reconstruction_objective,
        cosine_steps=args.cosine_steps,
        cosine_lr=args.cosine_lr,
        cosine_l2=args.cosine_l2,
        max_length=args.max_length,
        max_tokens_per_layer=args.max_tokens_per_layer,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
    )


def cmd_hf_xfamily_reconstruct(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_recovery import reconstruct_cross_tokenizer_causal_lm

    return reconstruct_cross_tokenizer_causal_lm(
        source_model=args.source_model,
        candidate_model=args.candidate_model,
        output_dir=args.output,
        ridge=args.ridge,
        reconstruction_alpha=args.reconstruction_alpha,
        reconstruction_target=args.reconstruction_target,
        donor_layer_fold_mode=args.donor_layer_fold_mode,
        reconstruction_objective=args.reconstruction_objective,
        cosine_steps=args.cosine_steps,
        cosine_lr=args.cosine_lr,
        cosine_l2=args.cosine_l2,
        max_length=args.max_length,
        max_tokens_per_layer=args.max_tokens_per_layer,
        max_anchors=args.max_anchors,
        hidden_projection_scope=args.hidden_projection_scope,
        layer_limit=args.layer_limit,
        fit_gate=args.fit_gate,
        fit_gate_validation_fraction=args.fit_gate_validation_fraction,
        fit_gate_min_validation_rows=args.fit_gate_min_validation_rows,
        fit_gate_min_relative_improvement=args.fit_gate_min_relative_improvement,
        fit_gate_max_cosine_drop=args.fit_gate_max_cosine_drop,
        layer_gate_metric=args.layer_gate_metric,
        layer_gate_max_ppl_ratio=args.layer_gate_max_ppl_ratio,
        layer_gate_max_texts=args.layer_gate_max_texts,
        layer_gate_max_behavior_drop=args.layer_gate_max_behavior_drop,
        layer_gate_behavior_max_new_tokens=args.layer_gate_behavior_max_new_tokens,
        layer_gate_eval_suite=args.layer_gate_eval_suite,
        layer_gate_max_generation_prompts=args.layer_gate_max_generation_prompts,
        torch_dtype=args.dtype,
        candidate_torch_dtype=args.candidate_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
    )


def cmd_hf_xfamily_block_recover(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_recovery import blockwise_reconstruct_cross_tokenizer_causal_lm

    return blockwise_reconstruct_cross_tokenizer_causal_lm(
        source_model=args.source_model,
        candidate_model=args.candidate_model,
        output_dir=args.output,
        steps_per_layer=args.steps_per_layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        lm_weight=args.lm_weight,
        response_lm_weight=args.response_lm_weight,
        first_token_lm_weight=args.first_token_lm_weight,
        response_end_lm_weight=args.response_end_lm_weight,
        response_loss_mode=args.response_loss_mode,
        hidden_ridge=args.hidden_ridge,
        hidden_projection_scope=args.hidden_projection_scope,
        hidden_metric=args.hidden_metric,
        span_pair_mode=args.span_pair_mode,
        block_target=args.block_target,
        residual_magnitude_compensation=args.residual_magnitude_compensation,
        residual_compensation_min_scale=args.residual_compensation_min_scale,
        residual_compensation_max_scale=args.residual_compensation_max_scale,
        residual_compensation_linear_ridge=args.residual_compensation_linear_ridge,
        residual_compensation_alpha=args.residual_compensation_alpha,
        residual_compensation_linear_rank=args.residual_compensation_linear_rank,
        pre_reconstruction_target=args.pre_reconstruction_target,
        pre_reconstruction_ridge=args.pre_reconstruction_ridge,
        pre_reconstruction_alpha=args.pre_reconstruction_alpha,
        pre_reconstruction_objective=args.pre_reconstruction_objective,
        pre_reconstruction_cosine_steps=args.pre_reconstruction_cosine_steps,
        pre_reconstruction_cosine_lr=args.pre_reconstruction_cosine_lr,
        pre_reconstruction_cosine_l2=args.pre_reconstruction_cosine_l2,
        pre_reconstruction_fit_gate=args.pre_reconstruction_fit_gate,
        pre_reconstruction_validation_fraction=args.pre_reconstruction_validation_fraction,
        pre_reconstruction_min_validation_rows=args.pre_reconstruction_min_validation_rows,
        pre_reconstruction_max_tokens_per_layer=args.pre_reconstruction_max_tokens_per_layer,
        pre_reconstruction_min_relative_improvement=args.pre_reconstruction_min_relative_improvement,
        pre_reconstruction_max_cosine_drop=args.pre_reconstruction_max_cosine_drop,
        max_anchors=args.max_anchors,
        span_max_pairs=args.span_max_pairs,
        donor_layer_fold_mode=args.donor_layer_fold_mode,
        donor_layer_group_strategy=args.donor_layer_group_strategy,
        layer_limit=args.layer_limit,
        layer_gate_metric=args.layer_gate_metric,
        layer_gate_max_ppl_ratio=args.layer_gate_max_ppl_ratio,
        layer_gate_max_texts=args.layer_gate_max_texts,
        layer_gate_max_behavior_drop=args.layer_gate_max_behavior_drop,
        layer_gate_behavior_max_new_tokens=args.layer_gate_behavior_max_new_tokens,
        layer_gate_eval_suite=args.layer_gate_eval_suite,
        layer_gate_max_generation_prompts=args.layer_gate_max_generation_prompts,
        on_policy_anchor_kd_weight=args.on_policy_anchor_kd_weight,
        on_policy_unlikelihood_weight=args.on_policy_unlikelihood_weight,
        on_policy_hidden_kd_weight=args.on_policy_hidden_kd_weight,
        on_policy_max_new_tokens=args.on_policy_max_new_tokens,
        on_policy_interval=args.on_policy_interval,
        on_policy_response_loss_mode=args.on_policy_response_loss_mode,
        on_policy_unlikelihood_window=args.on_policy_unlikelihood_window,
        stepwise_donor_kd_weight=args.stepwise_donor_kd_weight,
        stepwise_donor_max_new_tokens=args.stepwise_donor_max_new_tokens,
        stepwise_donor_interval=args.stepwise_donor_interval,
        teacher_generate_lm_weight=args.teacher_generate_lm_weight,
        teacher_generate_hidden_kd_weight=args.teacher_generate_hidden_kd_weight,
        teacher_generate_projected_vocab_kd_weight=args.teacher_generate_projected_vocab_kd_weight,
        teacher_generate_byte_kd_weight=args.teacher_generate_byte_kd_weight,
        teacher_generate_max_new_tokens=args.teacher_generate_max_new_tokens,
        alm_kd_weight=args.alm_kd_weight,
        alm_max_chunks=args.alm_max_chunks,
        alm_min_chars=args.alm_min_chars,
        projected_vocab_kd_weight=args.projected_vocab_kd_weight,
        focused_block_hidden_kd_weight=args.focused_block_hidden_kd_weight,
        focused_projected_vocab_kd_weight=args.focused_projected_vocab_kd_weight,
        focused_lm_weight=args.focused_lm_weight,
        token_focus=args.token_focus,
        projected_vocab_kd_mode=args.projected_vocab_kd_mode,
        byte_kd_weight=args.byte_kd_weight,
        byte_hidden_kd_weight=args.byte_hidden_kd_weight,
        block_io_kd_weight=args.block_io_kd_weight,
        block_io_ridge=args.block_io_ridge,
        block_io_loss=args.block_io_loss,
        train_output_head_with_kd=args.train_output_head_with_kd,
        untie_output_head_for_kd=args.untie_output_head_for_kd,
        allow_tied_output_head_training=args.allow_tied_output_head_training,
        module_output_kd_weight=args.module_output_kd_weight,
        module_output_targets=args.module_output_targets,
        module_output_transport=args.module_output_transport,
        module_output_temperature=args.module_output_temperature,
        module_output_sinkhorn_iterations=args.module_output_sinkhorn_iterations,
        projected_vocab_max_source_tokens=args.projected_vocab_max_source_tokens,
        temperature=args.temperature,
        torch_dtype=args.dtype,
        candidate_torch_dtype=args.candidate_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
        calibration_order=args.calibration_order,
        grad_clip=args.grad_clip,
        report_interval=args.report_interval,
        save_best_layer_loss=args.save_best_layer_loss,
        save_best_layer_behavior=args.save_best_layer_behavior,
        layer_behavior_eval_interval=args.layer_behavior_eval_interval,
        layer_checkpoint_interval=args.layer_checkpoint_interval,
    )


def cmd_hf_xfamily_chain_anchor_compress(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_xfamily import chain_anchor_staged_compress

    return chain_anchor_staged_compress(
        source_model=args.source_model,
        source_weights=args.source_weights,
        anchor_model=args.anchor_model,
        output_dir=args.output,
        stages=args.stages,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
        max_new_tokens=args.max_new_tokens,
        stage_max_ppl_ratio=args.stage_max_ppl_ratio,
        stage_ppl_gate_mode=args.stage_ppl_gate_mode,
        stage_max_accuracy_drop=args.stage_max_accuracy_drop,
        stage_max_behavior_drop=args.stage_max_behavior_drop,
        stage_min_choice_accuracy=args.stage_min_choice_accuracy,
        stage_min_behavior_pass_rate=args.stage_min_behavior_pass_rate,
        promote_policy=args.promote_policy,
        stage_input_policy=args.stage_input_policy,
        init_fold_mode=args.init_fold_mode,
        init_hidden_basis_mode=args.init_hidden_basis_mode,
        init_hidden_basis_scope=args.init_hidden_basis_scope,
        init_hidden_basis_max_rows=args.init_hidden_basis_max_rows,
        init_hidden_basis_preprocess=args.init_hidden_basis_preprocess,
        init_activation_fold_assignment=args.init_activation_fold_assignment,
        init_fold_stat_repair=args.init_fold_stat_repair,
        init_transport_backend=args.init_transport_backend,
        init_transport_support=args.init_transport_support,
        init_donor_layer_fold_mode=args.init_donor_layer_fold_mode,
        init_donor_layer_group_strategy=args.init_donor_layer_group_strategy,
        init_attention_fold_mode=args.init_attention_fold_mode,
        init_attention_head_dim_transport=args.init_attention_head_dim_transport,
        init_token_init_mode=args.init_token_init_mode,
        init_projection_scale=args.init_projection_scale,
        init_projection_blend_alpha=args.init_projection_blend_alpha,
        init_consumer_hidden_reduction=args.init_consumer_hidden_reduction,
        init_activation_fit_linears=args.init_activation_fit_linears,
        init_activation_fit_target=args.init_activation_fit_target,
        init_activation_fit_max_rows=args.init_activation_fit_max_rows,
        init_activation_fit_alpha=args.init_activation_fit_alpha,
        init_activation_fit_ridge=args.init_activation_fit_ridge,
        init_activation_fit_rank=args.init_activation_fit_rank,
        init_activation_fit_rank_mode=args.init_activation_fit_rank_mode,
        init_activation_fit_method=args.init_activation_fit_method,
        init_activation_fit_factor_steps=args.init_activation_fit_factor_steps,
        init_activation_fit_factor_lr=args.init_activation_fit_factor_lr,
        init_activation_fit_factor_l2=args.init_activation_fit_factor_l2,
        init_activation_fit_gate=args.init_activation_fit_gate,
        init_activation_fit_validation_fraction=args.init_activation_fit_validation_fraction,
        init_activation_fit_min_validation_rows=args.init_activation_fit_min_validation_rows,
        init_activation_fit_min_relative_improvement=args.init_activation_fit_min_relative_improvement,
        init_activation_fit_max_cosine_drop=args.init_activation_fit_max_cosine_drop,
        init_activation_feature_mode=args.init_activation_feature_mode,
        init_activation_subspace_rank=args.init_activation_subspace_rank,
        init_activation_domain_balance=args.init_activation_domain_balance,
        init_untie_output_head=args.init_untie_output_head,
        init_activation_max_length=args.init_activation_max_length,
        block_steps_per_layer=args.block_steps_per_layer,
        block_batch_size=args.block_batch_size,
        block_max_length=args.block_max_length,
        block_lr=args.block_lr,
        block_lm_weight=args.block_lm_weight,
        block_response_lm_weight=args.block_response_lm_weight,
        block_first_token_lm_weight=args.block_first_token_lm_weight,
        block_response_end_lm_weight=args.block_response_end_lm_weight,
        block_response_loss_mode=args.block_response_loss_mode,
        block_hidden_projection_scope=args.block_hidden_projection_scope,
        block_hidden_metric=args.block_hidden_metric,
        block_span_pair_mode=args.block_span_pair_mode,
        block_target=args.block_target,
        block_residual_magnitude_compensation=args.block_residual_magnitude_compensation,
        block_residual_compensation_min_scale=args.block_residual_compensation_min_scale,
        block_residual_compensation_max_scale=args.block_residual_compensation_max_scale,
        block_residual_compensation_linear_ridge=args.block_residual_compensation_linear_ridge,
        block_residual_compensation_alpha=args.block_residual_compensation_alpha,
        block_residual_compensation_linear_rank=args.block_residual_compensation_linear_rank,
        block_pre_reconstruction_target=args.block_pre_reconstruction_target,
        block_pre_reconstruction_ridge=args.block_pre_reconstruction_ridge,
        block_pre_reconstruction_alpha=args.block_pre_reconstruction_alpha,
        block_pre_reconstruction_objective=args.block_pre_reconstruction_objective,
        block_pre_reconstruction_cosine_steps=args.block_pre_reconstruction_cosine_steps,
        block_pre_reconstruction_cosine_lr=args.block_pre_reconstruction_cosine_lr,
        block_pre_reconstruction_cosine_l2=args.block_pre_reconstruction_cosine_l2,
        block_pre_reconstruction_fit_gate=args.block_pre_reconstruction_fit_gate,
        block_pre_reconstruction_validation_fraction=args.block_pre_reconstruction_validation_fraction,
        block_pre_reconstruction_min_validation_rows=args.block_pre_reconstruction_min_validation_rows,
        block_pre_reconstruction_max_tokens_per_layer=args.block_pre_reconstruction_max_tokens_per_layer,
        block_pre_reconstruction_min_relative_improvement=args.block_pre_reconstruction_min_relative_improvement,
        block_pre_reconstruction_max_cosine_drop=args.block_pre_reconstruction_max_cosine_drop,
        block_donor_layer_fold_mode=args.block_donor_layer_fold_mode,
        block_donor_layer_group_strategy=args.block_donor_layer_group_strategy,
        block_layer_limit=args.block_layer_limit,
        block_layer_gate_metric=args.block_layer_gate_metric,
        block_layer_gate_max_texts=args.block_layer_gate_max_texts,
        block_layer_gate_max_generation_prompts=args.block_layer_gate_max_generation_prompts,
        block_layer_gate_max_behavior_drop=args.block_layer_gate_max_behavior_drop,
        block_alm_kd_weight=args.block_alm_kd_weight,
        block_projected_vocab_kd_weight=args.block_projected_vocab_kd_weight,
        block_byte_kd_weight=args.block_byte_kd_weight,
        block_block_io_kd_weight=args.block_block_io_kd_weight,
        block_on_policy_anchor_kd_weight=args.block_on_policy_anchor_kd_weight,
        block_on_policy_unlikelihood_weight=args.block_on_policy_unlikelihood_weight,
        block_on_policy_hidden_kd_weight=args.block_on_policy_hidden_kd_weight,
        block_on_policy_max_new_tokens=args.block_on_policy_max_new_tokens,
        block_on_policy_interval=args.block_on_policy_interval,
        block_on_policy_response_loss_mode=args.block_on_policy_response_loss_mode,
        block_on_policy_unlikelihood_window=args.block_on_policy_unlikelihood_window,
        block_stepwise_donor_kd_weight=args.block_stepwise_donor_kd_weight,
        block_stepwise_donor_max_new_tokens=args.block_stepwise_donor_max_new_tokens,
        block_stepwise_donor_interval=args.block_stepwise_donor_interval,
        block_teacher_generate_lm_weight=args.block_teacher_generate_lm_weight,
        block_teacher_generate_hidden_kd_weight=args.block_teacher_generate_hidden_kd_weight,
        block_teacher_generate_projected_vocab_kd_weight=args.block_teacher_generate_projected_vocab_kd_weight,
        block_teacher_generate_byte_kd_weight=args.block_teacher_generate_byte_kd_weight,
        block_teacher_generate_max_new_tokens=args.block_teacher_generate_max_new_tokens,
        block_alm_max_chunks=args.block_alm_max_chunks,
        block_alm_min_chars=args.block_alm_min_chars,
        block_focused_block_hidden_kd_weight=args.block_focused_block_hidden_kd_weight,
        block_focused_projected_vocab_kd_weight=args.block_focused_projected_vocab_kd_weight,
        block_focused_lm_weight=args.block_focused_lm_weight,
        block_token_focus=args.block_token_focus,
        block_projected_vocab_kd_mode=args.block_projected_vocab_kd_mode,
        block_byte_hidden_kd_weight=args.block_byte_hidden_kd_weight,
        block_block_io_ridge=args.block_block_io_ridge,
        block_block_io_loss=args.block_block_io_loss,
        block_train_output_head_with_kd=args.block_train_output_head_with_kd,
        block_untie_output_head_for_kd=args.block_untie_output_head_for_kd,
        block_allow_tied_output_head_training=args.block_allow_tied_output_head_training,
        block_module_output_kd_weight=args.block_module_output_kd_weight,
        block_module_output_targets=args.block_module_output_targets,
        block_module_output_transport=args.block_module_output_transport,
        block_module_output_temperature=args.block_module_output_temperature,
        block_module_output_sinkhorn_iterations=args.block_module_output_sinkhorn_iterations,
        block_projected_vocab_max_source_tokens=args.block_projected_vocab_max_source_tokens,
        block_save_best_layer_loss=args.block_save_best_layer_loss,
        block_save_best_layer_behavior=args.block_save_best_layer_behavior,
        block_layer_behavior_eval_interval=args.block_layer_behavior_eval_interval,
        block_layer_checkpoint_interval=args.block_layer_checkpoint_interval,
        block_calibration_order=args.block_calibration_order,
        alm_steps=args.alm_steps,
        alm_batch_size=args.alm_batch_size,
        alm_max_length=args.alm_max_length,
        alm_lr=args.alm_lr,
        alm_anchor_kd_weight=args.alm_anchor_kd_weight,
        alm_lm_weight=args.alm_lm_weight,
        alm_first_token_lm_weight=args.alm_first_token_lm_weight,
        alm_response_end_lm_weight=args.alm_response_end_lm_weight,
        alm_focused_lm_weight=args.alm_focused_lm_weight,
        alm_dual_space_kd_weight=args.alm_dual_space_kd_weight,
        alm_attention_influence_weight=args.alm_attention_influence_weight,
        alm_attention_influence_layer=args.alm_attention_influence_layer,
        alm_kd_weight=args.alm_kd_weight,
        alm_projected_vocab_kd_weight=args.alm_projected_vocab_kd_weight,
        alm_focused_projected_vocab_kd_weight=args.alm_focused_projected_vocab_kd_weight,
        alm_span_hidden_kd_weight=args.alm_span_hidden_kd_weight,
        alm_token_kd_weight=args.alm_token_kd_weight,
        alm_token_focus=args.alm_token_focus,
        alm_token_max_pairs=args.alm_token_max_pairs,
        alm_token_teacher_confidence_floor=args.alm_token_teacher_confidence_floor,
        alm_answer_contrastive_weight=args.alm_answer_contrastive_weight,
        alm_answer_contrastive_max_negatives=args.alm_answer_contrastive_max_negatives,
        alm_answer_first_token_margin_weight=args.alm_answer_first_token_margin_weight,
        alm_answer_first_token_margin=args.alm_answer_first_token_margin,
        alm_answer_first_token_max_negatives=args.alm_answer_first_token_max_negatives,
        alm_answer_first_token_top_negatives=args.alm_answer_first_token_top_negatives,
        alm_teacher_rank_weight=args.alm_teacher_rank_weight,
        alm_teacher_rank_temperature=args.alm_teacher_rank_temperature,
        alm_teacher_rank_max_negatives=args.alm_teacher_rank_max_negatives,
        alm_teacher_generate_lm_weight=args.alm_teacher_generate_lm_weight,
        alm_teacher_generate_max_new_tokens=args.alm_teacher_generate_max_new_tokens,
        alm_stepwise_donor_kd_weight=args.alm_stepwise_donor_kd_weight,
        alm_stepwise_donor_max_new_tokens=args.alm_stepwise_donor_max_new_tokens,
        alm_on_policy_anchor_kd_weight=args.alm_on_policy_anchor_kd_weight,
        alm_on_policy_unlikelihood_weight=args.alm_on_policy_unlikelihood_weight,
        alm_on_policy_hidden_kd_weight=args.alm_on_policy_hidden_kd_weight,
        alm_on_policy_max_new_tokens=args.alm_on_policy_max_new_tokens,
        alm_on_policy_interval=args.alm_on_policy_interval,
        alm_train_scope=args.alm_train_scope,
        alm_response_loss_mode=args.alm_response_loss_mode,
        alm_append_eos=args.alm_append_eos,
        alm_save_best_behavior=args.alm_save_best_behavior,
        alm_behavior_eval_interval=args.alm_behavior_eval_interval,
        alm_calibration_order=args.alm_calibration_order,
        torch_dtype=args.dtype,
        candidate_torch_dtype=args.candidate_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_hf_lm_recover(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_recovery import recover_causal_lm

    return recover_causal_lm(
        source_model=None,
        candidate_model=args.candidate_model,
        output_dir=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        kd_weight=0.0,
        lm_weight=1.0,
        first_token_lm_weight=args.first_token_lm_weight,
        answer_contrastive_weight=args.answer_contrastive_weight,
        answer_contrastive_max_negatives=args.answer_contrastive_max_negatives,
        unlikelihood_weight=args.unlikelihood_weight,
        unlikelihood_window=args.unlikelihood_window,
        entropy_weight=args.entropy_weight,
        temperature=1.0,
        train_scope=args.train_scope,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
        response_loss_mode=args.response_loss_mode,
        append_eos=args.append_eos,
        grad_clip=args.grad_clip,
        report_interval=args.report_interval,
        save_best_loss=args.save_best_loss,
        save_best_behavior=args.save_best_behavior,
        behavior_eval_interval=args.behavior_eval_interval,
        behavior_max_new_tokens=args.behavior_max_new_tokens,
        behavior_max_probes=args.behavior_max_probes,
        behavior_eval_suite=args.behavior_eval_suite,
        behavior_max_generation_prompts=args.behavior_max_generation_prompts,
    )


def cmd_hf_xfamily_calibrate_head(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_recovery import calibrate_cross_tokenizer_lm_head

    return calibrate_cross_tokenizer_lm_head(
        source_model=args.source_model,
        candidate_model=args.candidate_model,
        output_dir=args.output,
        ridge=args.ridge,
        head_blend_alpha=args.head_blend_alpha,
        head_calibration_mode=args.head_calibration_mode,
        response_loss_mode=args.response_loss_mode,
        answer_first_token_margin=args.answer_first_token_margin,
        answer_first_token_max_negatives=args.answer_first_token_max_negatives,
        answer_first_token_top_negatives=args.answer_first_token_top_negatives,
        answer_margin_steps=args.answer_margin_steps,
        max_length=args.max_length,
        max_calibration_rows=args.max_calibration_rows,
        projected_vocab_max_source_tokens=args.projected_vocab_max_source_tokens,
        torch_dtype=args.dtype,
        candidate_torch_dtype=args.candidate_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
    )


def cmd_vision_probe(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import vision_probe

    return vision_probe(
        model_id_or_path=args.model,
        output_file=args.output,
        image_dir=args.image_dir,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
    )


def cmd_vision_chain(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import vision_text_chain

    return vision_text_chain(
        vision_model_id_or_path=args.vision_model,
        text_model_id_or_path=args.text_model,
        output_file=args.output,
        image_dir=args.image_dir,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
    )


def cmd_native_vlm_demo(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.native_multimodal import train_native_vlm_sandbox

    return train_native_vlm_sandbox(
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device=args.device,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        lr=args.lr,
    )


def cmd_hf_vision_graft(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_vision_graft import train_hf_native_vision_graft

    return train_hf_native_vision_graft(
        model_id_or_path=args.model,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device=args.device,
        torch_dtype=args.dtype,
        lr=args.lr,
        train_last_layers=args.train_last_layers,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        task_keys=None if args.task_keys is None else [part.strip() for part in args.task_keys.split(",") if part.strip()],
        grad_clip=args.grad_clip,
        text_retention_file=args.text_retention_file,
        text_retention_weight=args.text_retention_weight,
        text_retention_batch_size=args.text_retention_batch_size,
        text_retention_max_length=args.text_retention_max_length,
    )


def _resolve_widths(model, widths_text: str | None, target_params: int | None, min_width: int) -> list[int]:
    if widths_text and target_params:
        raise ValueError("Use --widths or --target-params, not both.")
    if widths_text:
        return parse_widths(widths_text)
    if target_params:
        return plan_widths_for_target_params(model, target_params, min_width=min_width).hidden_widths
    raise ValueError("Provide --widths or --target-params.")


def _parse_layer_groups(text: str) -> list[list[int]]:
    groups = []
    for group_text in text.split(","):
        parts = [part.strip() for part in group_text.split("+") if part.strip()]
        if parts:
            groups.append([int(part) for part in parts])
    if not groups:
        raise ValueError("Need at least one layer merge group.")
    return groups


def _load_recovery_calibration_texts(path: str | None, max_texts: int | None) -> list[str] | None:
    if path is None:
        return None
    from neural_forge.hf_recovery import load_calibration_texts

    return load_calibration_texts(path, max_texts=max_texts)


if __name__ == "__main__":
    raise SystemExit(main())
