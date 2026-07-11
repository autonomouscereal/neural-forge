#!/usr/bin/env python3
"""Compare pair and multi-anchor physical neuron folds on one real Qwen layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neural_forge.hf_qwen_compensated_neuron_fold import (
    fit_compensated_neuron_pairs,
)
from neural_forge.hf_qwen_multi_anchor_fold import fit_multi_anchor_neuron_fold


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_text(value: dict[str, Any]) -> str:
    text = value.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    messages = value.get("messages")
    if isinstance(messages, list):
        pieces = [
            item.get("content", "")
            for item in messages
            if isinstance(item, dict) and isinstance(item.get("content"), str)
        ]
        joined = "\n".join(piece.strip() for piece in pieces if piece.strip())
        if joined:
            return joined
    pieces = [
        value.get(name, "")
        for name in ("prompt", "instruction", "input", "response", "output")
    ]
    joined = "\n".join(
        piece.strip() for piece in pieces if isinstance(piece, str) and piece.strip()
    )
    if not joined:
        raise ValueError("A calibration JSONL row contains no usable text.")
    return joined


def _load_disjoint_texts(
    path: Path,
    calibration_count: int,
    validation_count: int,
) -> tuple[list[str], list[str]]:
    unique: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object.")
            text = _record_text(value)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            unique.append(text)
            if len(unique) >= calibration_count + validation_count:
                break
    required = calibration_count + validation_count
    if len(unique) < required:
        raise ValueError(f"Need {required} unique texts; found {len(unique)}.")
    return unique[:calibration_count], unique[calibration_count:required]


def _provenance(texts: list[str], label: str) -> str:
    payload = json.dumps(
        {"label": label, "texts": texts},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _collect_down_inputs(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    layer_index: int,
    max_length: int,
    max_rows: int,
    device: torch.device,
) -> Tensor:
    layer = model.model.layers[layer_index]
    pieces: list[Tensor] = []
    rows = 0
    prompt_budget = 0
    per_prompt_rows = math.ceil(max_rows / len(texts))

    def capture(_module: Any, inputs: tuple[Tensor, ...]) -> None:
        nonlocal prompt_budget, rows
        if rows >= max_rows or prompt_budget <= 0 or not inputs:
            return
        values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        take = min(max_rows - rows, prompt_budget, int(values.shape[0]))
        positions = torch.linspace(
            0,
            int(values.shape[0]) - 1,
            steps=take,
            device=values.device,
        ).round().to(dtype=torch.long)
        selected = values.index_select(0, positions)
        pieces.append(selected.to(device="cpu", dtype=torch.float32).contiguous())
        rows += take
        prompt_budget -= take

    handle = layer.mlp.down_proj.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for text in texts:
                if rows >= max_rows:
                    break
                prompt_budget = per_prompt_rows
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                )
                inputs = {
                    name: value.to(device=device)
                    for name, value in encoded.items()
                    if isinstance(value, Tensor)
                }
                model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if rows != max_rows:
        raise RuntimeError(f"Collected {rows} activation rows; expected {max_rows}.")
    return torch.cat(pieces, dim=0)


def _protected_indices(activations: Tensor, down_weight: Tensor, count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    activation_norm = torch.linalg.vector_norm(activations.to(torch.float64), dim=0)
    consumer_norm = torch.linalg.vector_norm(
        down_weight.detach().to(device="cpu", dtype=torch.float64), dim=0
    )
    importance = activation_norm * consumer_norm
    return tuple(
        sorted(int(index) for index in torch.topk(importance, count).indices.tolist())
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--layer-index", type=int, default=18)
    parser.add_argument("--drop-count", type=int, default=64)
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=64)
    parser.add_argument("--activation-rows", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--protected-fraction", type=float, default=0.05)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--max-anchors", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for name in (
        "drop_count",
        "calibration_count",
        "validation_count",
        "activation_rows",
        "max_length",
        "candidate_pool_size",
        "max_anchors",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not math.isfinite(args.protected_fraction) or not 0.0 <= args.protected_fraction < 0.5:
        raise ValueError("--protected-fraction must be finite and in [0, 0.5).")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    calibration_texts, validation_texts = _load_disjoint_texts(
        args.calibration_jsonl,
        args.calibration_count,
        args.validation_count,
    )
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    print("neural_forge_probe event=load_parent", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.source_model,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    layers = model.model.layers
    if args.layer_index < 0 or args.layer_index >= len(layers):
        raise ValueError("--layer-index is outside the model depth.")
    mlp = layers[args.layer_index].mlp
    width = int(mlp.gate_proj.weight.shape[0])
    if args.drop_count >= width:
        raise ValueError("--drop-count must be smaller than the layer width.")

    print("neural_forge_probe event=collect_calibration", flush=True)
    calibration = _collect_down_inputs(
        model,
        tokenizer,
        calibration_texts,
        layer_index=args.layer_index,
        max_length=args.max_length,
        max_rows=args.activation_rows,
        device=device,
    )
    print("neural_forge_probe event=collect_validation", flush=True)
    validation = _collect_down_inputs(
        model,
        tokenizer,
        validation_texts,
        layer_index=args.layer_index,
        max_length=args.max_length,
        max_rows=args.activation_rows,
        device=device,
    )
    protected_count = min(
        width - args.drop_count - 1,
        int(round(width * args.protected_fraction)),
    )
    protected = _protected_indices(calibration, mlp.down_proj.weight, protected_count)
    target_width = width - args.drop_count

    started = time.perf_counter()
    print("neural_forge_probe event=fit_pair", flush=True)
    pair = fit_compensated_neuron_pairs(
        mlp.gate_proj.weight,
        mlp.up_proj.weight,
        mlp.down_proj.weight,
        calibration,
        target_width=target_width,
        validation_activations=validation,
        protected_indices=protected,
        projection_count=8,
        neighbor_window=4,
        max_pair_candidates=32_768,
        max_generated_pairs=65_536,
        score_chunk_size=512,
        seed=20260710,
    )
    pair_seconds = time.perf_counter() - started
    pair_report = pair.as_dict()
    del pair
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    started = time.perf_counter()
    print("neural_forge_probe event=fit_multi_anchor", flush=True)
    multi = fit_multi_anchor_neuron_fold(
        mlp.gate_proj.weight,
        mlp.up_proj.weight,
        mlp.down_proj.weight,
        calibration,
        validation_activations=validation,
        target_width=target_width,
        calibration_provenance=_provenance(calibration_texts, "calibration"),
        validation_provenance=_provenance(validation_texts, "validation"),
        protected_indices=protected,
        candidate_pool_size=args.candidate_pool_size,
        max_anchors=args.max_anchors,
        similarity_chunk_size=64,
        max_similarity_products=16_000_000,
        max_projection_fits=131_072,
    )
    multi_seconds = time.perf_counter() - started
    multi_report = multi.as_dict()

    pair_error = float(
        pair_report["diagnostics"]["aggregate_validation_relative_l2"]
    )
    multi_error = float(
        multi_report["diagnostics"]["aggregate_validation_relative_l2"]
    )
    result = {
        "schema": "neural-forge-qwen-multi-anchor-layer-probe-v1",
        "status": "diagnostic-only-not-behavior-certified",
        "source_model": args.source_model,
        "layer_index": args.layer_index,
        "source_width": width,
        "target_width": target_width,
        "drop_count": args.drop_count,
        "protected_count": len(protected),
        "calibration_text_count": len(calibration_texts),
        "validation_text_count": len(validation_texts),
        "activation_rows_per_split": args.activation_rows,
        "activation_row_sampling": "uniform-across-each-prompt-token-sequence-v1",
        "pair": {"seconds": pair_seconds, "fold": pair_report},
        "multi_anchor": {"seconds": multi_seconds, "fold": multi_report},
        "heldout_validation_error_ratio_multi_over_pair": (
            multi_error / pair_error if pair_error > 0.0 else None
        ),
        "behavioral_claim": False,
        "optimizer_steps": 0,
        "backward_called": False,
    }
    _write_json(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "pair_validation_relative_l2": pair_error,
        "multi_anchor_validation_relative_l2": multi_error,
        "multi_over_pair": result["heldout_validation_error_ratio_multi_over_pair"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
