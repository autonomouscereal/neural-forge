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

    hf_resize = sub.add_parser("hf-resize-ffn", help="Shrink or expand gated FFN intermediate size in HF decoder models.")
    hf_resize.add_argument("model")
    hf_resize.add_argument("output")
    hf_resize.add_argument("--ratio", type=float)
    hf_resize.add_argument("--target-intermediate-size", type=int)
    hf_resize.add_argument("--dtype", default="float16")
    hf_resize.add_argument("--trust-remote-code", action="store_true")
    hf_resize.add_argument("--assignment", choices=["down-cosine", "round-robin"], default="down-cosine")
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
    hf_sweep.set_defaults(func=cmd_hf_sweep)

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


def _resolve_widths(model, widths_text: str | None, target_params: int | None, min_width: int) -> list[int]:
    if widths_text and target_params:
        raise ValueError("Use --widths or --target-params, not both.")
    if widths_text:
        return parse_widths(widths_text)
    if target_params:
        return plan_widths_for_target_params(model, target_params, min_width=min_width).hidden_widths
    raise ValueError("Provide --widths or --target-params.")


if __name__ == "__main__":
    raise SystemExit(main())
