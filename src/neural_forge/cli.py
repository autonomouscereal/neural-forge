"""Command-line interface for Neural Forge."""

from __future__ import annotations

import argparse
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
    hf_donor_init.add_argument("--seed", type=int, default=0)
    hf_donor_init.add_argument("--ridge", type=float, default=1e-3)
    hf_donor_init.add_argument("--max-anchors", type=int, default=50_000)
    hf_donor_init.add_argument("--fold-mode", choices=["mean", "rms", "transport"], default="mean")
    hf_donor_init.add_argument("--activation-fold-assignment", choices=["balanced", "similarity"], default="balanced")
    hf_donor_init.add_argument(
        "--donor-layer-fold-mode",
        choices=["representative", "average", "fold"],
        default="representative",
        help="How to fold parent/donor depth spans into each child/target layer.",
    )
    hf_donor_init.add_argument(
        "--attention-fold-mode",
        choices=["independent", "coupled"],
        default="independent",
        help="Fold attention Q/K/V/O projection head partitions together instead of as flat axes.",
    )
    hf_donor_init.add_argument(
        "--token-init-mode",
        choices=["shared", "donor-tokenized"],
        default="shared",
        help="Initialize only shared decoded-token rows or fill target vocab rows by donor tokenizing each target token.",
    )
    hf_donor_init.add_argument(
        "--projection-scale",
        choices=["none", "target-rms"],
        default="none",
        help="Optional RMS rescale for folded parent tensors before loading into the child shape.",
    )
    hf_donor_init.add_argument(
        "--projection-blend-alpha",
        type=float,
        default=1.0,
        help="Blend folded parent tensors with the target child tensors after projection; 1.0 fully replaces target.",
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
    hf_eval.set_defaults(func=cmd_hf_eval)

    hf_gate = sub.add_parser("hf-gate", help="Compare an HF eval report to a baseline and return a pass/fail verdict.")
    hf_gate.add_argument("baseline_report")
    hf_gate.add_argument("candidate_report")
    hf_gate.add_argument("--output")
    hf_gate.add_argument("--max-ppl-ratio", type=float, default=1.25)
    hf_gate.add_argument("--max-accuracy-drop", type=float, default=0.0)
    hf_gate.add_argument("--min-choice-accuracy", type=float, default=0.75)
    hf_gate.add_argument("--min-behavior-pass-rate", type=float, default=0.60)
    hf_gate.add_argument("--min-continuation-chars", type=int, default=24)
    hf_gate.add_argument("--min-unique-token-ratio", type=float, default=0.20)
    hf_gate.set_defaults(func=cmd_hf_gate)

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
    recover.add_argument("--unlikelihood-weight", type=float, default=0.0)
    recover.add_argument("--unlikelihood-window", type=int, default=16)
    recover.add_argument("--entropy-weight", type=float, default=0.0)
    recover.add_argument("--response-loss-mode", choices=["none", "auto", "assistant", "answer"], default="none")
    recover.add_argument("--temperature", type=float, default=2.0)
    recover.add_argument("--train-scope", default="all")
    recover.add_argument("--dtype", default="float32")
    recover.add_argument("--device", default="cuda:0")
    recover.add_argument("--trust-remote-code", action="store_true")
    recover.add_argument("--grad-clip", type=float, default=1.0)
    recover.add_argument("--report-interval", type=int)
    recover.add_argument("--save-best-loss", action="store_true")
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
    xfamily_recover.add_argument("--hidden-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--span-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--span-hidden-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--projected-vocab-kd-weight", type=float, default=0.0)
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
    xfamily_recover.add_argument("--entropy-aware-kd-weight", type=float, default=0.0)
    xfamily_recover.add_argument("--entropy-aware-kd-threshold", type=float, default=0.55)
    xfamily_recover.add_argument("--entropy-aware-kd-sharpness", type=float, default=8.0)
    xfamily_recover.add_argument("--span-max-pairs", type=int, default=512)
    xfamily_recover.add_argument("--hidden-ridge", type=float, default=1e-3)
    xfamily_recover.add_argument("--hidden-layer", choices=["last", "middle"], default="last")
    xfamily_recover.add_argument("--hidden-kd-metric", choices=["projected", "cka", "both"], default="projected")
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
    xfamily_recover.add_argument("--calibration-file")
    xfamily_recover.add_argument("--max-calibration-texts", type=int)
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

    lm_recover = sub.add_parser("hf-lm-recover", help="Run teacherless LM recovery on a folded HF causal LM.")
    lm_recover.add_argument("candidate_model")
    lm_recover.add_argument("output")
    lm_recover.add_argument("--steps", type=int, default=80)
    lm_recover.add_argument("--batch-size", type=int, default=1)
    lm_recover.add_argument("--max-length", type=int, default=96)
    lm_recover.add_argument("--lr", type=float, default=5e-5)
    lm_recover.add_argument("--unlikelihood-weight", type=float, default=0.0)
    lm_recover.add_argument("--unlikelihood-window", type=int, default=16)
    lm_recover.add_argument("--entropy-weight", type=float, default=0.0)
    lm_recover.add_argument("--response-loss-mode", choices=["none", "auto", "assistant", "answer"], default="none")
    lm_recover.add_argument("--train-scope", default="all")
    lm_recover.add_argument("--dtype", default="float32")
    lm_recover.add_argument("--device", default="cuda:0")
    lm_recover.add_argument("--trust-remote-code", action="store_true")
    lm_recover.add_argument("--grad-clip", type=float, default=1.0)
    lm_recover.add_argument("--report-interval", type=int)
    lm_recover.add_argument("--save-best-loss", action="store_true")
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
    )


def cmd_hf_gate(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_tools import gate_text_report

    return gate_text_report(
        baseline_report=args.baseline_report,
        candidate_report=args.candidate_report,
        output_file=args.output,
        max_ppl_ratio=args.max_ppl_ratio,
        max_accuracy_drop=args.max_accuracy_drop,
        min_choice_accuracy=args.min_choice_accuracy,
        min_behavior_pass_rate=args.min_behavior_pass_rate,
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

    activation_calibration_texts = _load_recovery_calibration_texts(
        args.activation_calibration_file,
        args.activation_max_calibration_texts,
    )
    return donor_projected_init(
        target_config_model=args.target_config_model,
        target_base_model=args.target_base_model,
        target_tokenizer_model=args.target_tokenizer_model,
        donor_model=args.donor_model,
        output_dir=args.output,
        seed=args.seed,
        ridge=args.ridge,
        max_anchors=args.max_anchors,
        fold_mode=args.fold_mode,
        activation_fold_assignment=args.activation_fold_assignment,
        donor_layer_fold_mode=args.donor_layer_fold_mode,
        attention_fold_mode=args.attention_fold_mode,
        token_init_mode=args.token_init_mode,
        projection_scale=args.projection_scale,
        projection_blend_alpha=args.projection_blend_alpha,
        untie_output_head=args.untie_output_head,
        activation_calibration_texts=activation_calibration_texts,
        activation_max_length=args.activation_max_length,
        activation_device=args.activation_device,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )


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
        unlikelihood_weight=args.unlikelihood_weight,
        unlikelihood_window=args.unlikelihood_window,
        entropy_weight=args.entropy_weight,
        temperature=args.temperature,
        train_scope=args.train_scope,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        calibration_texts=_load_recovery_calibration_texts(args.calibration_file, args.max_calibration_texts),
        response_loss_mode=args.response_loss_mode,
        grad_clip=args.grad_clip,
        report_interval=args.report_interval,
        save_best_loss=args.save_best_loss,
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
        hidden_kd_weight=args.hidden_kd_weight,
        span_kd_weight=args.span_kd_weight,
        span_hidden_kd_weight=args.span_hidden_kd_weight,
        projected_vocab_kd_weight=args.projected_vocab_kd_weight,
        projected_vocab_kd_mode=args.projected_vocab_kd_mode,
        projected_vocab_max_source_tokens=args.projected_vocab_max_source_tokens,
        reference_model=args.reference_model,
        reference_kd_weight=args.reference_kd_weight,
        unlikelihood_weight=args.unlikelihood_weight,
        unlikelihood_window=args.unlikelihood_window,
        entropy_weight=args.entropy_weight,
        response_loss_mode=args.response_loss_mode,
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
        grad_clip=args.grad_clip,
        report_interval=args.report_interval,
        save_best_loss=args.save_best_loss,
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
        grad_clip=args.grad_clip,
        report_interval=args.report_interval,
        save_best_loss=args.save_best_loss,
    )


def cmd_hf_xfamily_calibrate_head(args: argparse.Namespace) -> dict[str, object]:
    from neural_forge.hf_recovery import calibrate_cross_tokenizer_lm_head

    return calibrate_cross_tokenizer_lm_head(
        source_model=args.source_model,
        candidate_model=args.candidate_model,
        output_dir=args.output,
        ridge=args.ridge,
        head_blend_alpha=args.head_blend_alpha,
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
