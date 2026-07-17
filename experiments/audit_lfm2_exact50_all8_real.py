#!/usr/bin/env python3
"""Audit a real all8 checkpoint against direct parent quantization.

This is a diagnostic, not an evaluator. It verifies that every stored code and
scale equals a fresh direct eight-bit quantization of the bound local parent,
that the standalone tokenizer is behaviorally identical on the sealed prompts,
and that the packed runtime matches a dense model containing those same
quantized values.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

try:
    from experiments import run_lfm2_residual_attention_fold as direct_lfm2
except ImportError:
    import run_lfm2_residual_attention_fold as direct_lfm2

from neural_forge.hf_lfm2_exact50_all8 import (
    load_lfm2_exact50_all8,
    load_lfm2_exact50_all8_tokenizer,
)
from neural_forge.hf_lfm2_value_fold_control import (
    Lfm2ValueFoldTarget,
    _analyze_model,
    _parameter_chunks,
    _quantize_groups,
)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class RealAll8AuditError(RuntimeError):
    """The real checkpoint differs from its direct reference."""


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir():
        raise RealAll8AuditError(f"{name} is not a local directory: {path}.")
    return path


def _read_envelope(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).expanduser().absolute()
    if not source.is_file():
        raise RealAll8AuditError(f"parent_envelope is not a local file: {source}.")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise RealAll8AuditError("Parent envelope is malformed.")
    return value


def execute(args: argparse.Namespace) -> dict[str, Any]:
    parent_directory = _local_directory(args.parent_model, "parent_model")
    checkpoint = _local_directory(args.checkpoint, "checkpoint")
    envelope = _read_envelope(args.parent_envelope)
    target = Lfm2ValueFoldTarget(
        500_000,
        group_size=args.group_size,
        analysis_chunk_groups=args.analysis_chunk_groups,
    )
    parent = direct_lfm2.load_local_lfm2_model(
        parent_directory,
        device="cpu",
        dtype=torch.bfloat16,
    )
    child = load_lfm2_exact50_all8(
        checkpoint,
        device="cpu",
        max_checkpoint_bytes=232_064_014,
        max_packed_runtime_bytes=232_064_014,
    )
    manifest = child.neural_forge_checkpoint_manifest
    names = sorted(manifest["tensors"])
    registry = tuple(child.neural_forge_packed_registry)
    if len(names) != len(registry):
        raise RealAll8AuditError("Packed registry length differs from the manifest.")
    stores = dict(zip(names, registry, strict=True))
    analyses, aliases = _analyze_model(parent, target)
    if aliases != manifest["state"]["logical_to_canonical"]:
        raise RealAll8AuditError("Parent aliases differ from the all8 manifest.")

    code_mismatches = 0
    scale_mismatches = 0
    reconstructed_mismatches = 0
    maximum_reconstruction_delta = 0.0
    checked_codes = 0
    checked_scales = 0
    with torch.no_grad():
        for analysis in analyses:
            store = stores[analysis.canonical_name]
            store_codes = store.codes.reshape(store.group_count, store.group_size)
            store_scales = store.scales.reshape(-1)
            flat = analysis.parameter.reshape(-1)
            for (
                group_start,
                group_end,
                value_start,
                value_end,
                values,
            ) in _parameter_chunks(
                analysis.parameter,
                group_size=target.group_size,
                chunk_groups=target.analysis_chunk_groups,
            ):
                codes, scales, reconstructed = _quantize_groups(values, 8)
                observed_codes = store_codes[group_start:group_end].cpu()
                observed_scales = store_scales[group_start:group_end].cpu()
                code_mismatches += int(torch.count_nonzero(observed_codes != codes))
                scale_mismatches += int(
                    torch.count_nonzero(observed_scales != scales)
                )
                checked_codes += int(codes.numel())
                checked_scales += int(scales.numel())
                observed = (
                    (
                        observed_codes.to(torch.int16)
                        - 127
                    ).to(torch.float32)
                    * observed_scales.to(torch.float32)[:, None]
                )
                delta = (observed - reconstructed).abs()
                reconstructed_mismatches += int(torch.count_nonzero(delta))
                maximum_reconstruction_delta = max(
                    maximum_reconstruction_delta,
                    float(delta.max().item()),
                )
                flat[value_start:value_end].copy_(
                    reconstructed.reshape(-1)[: value_end - value_start].to(
                        dtype=flat.dtype
                    )
                )

    original_tokenizer = direct_lfm2.load_local_tokenizer(parent_directory)
    packed_tokenizer = load_lfm2_exact50_all8_tokenizer(checkpoint)
    tokenizer_mismatch_ids: list[str] = []
    for row in envelope["rows"]:
        case_id = str(row["case_id"])
        prompt = str(row["prompt"])
        original = original_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        packed = packed_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        if (
            not torch.equal(original["input_ids"], packed["input_ids"])
            or not torch.equal(
                original.get(
                    "attention_mask",
                    torch.ones_like(original["input_ids"]),
                ),
                packed.get(
                    "attention_mask",
                    torch.ones_like(packed["input_ids"]),
                ),
            )
        ):
            tokenizer_mismatch_ids.append(case_id)

    probe_rows = envelope["rows"][: args.probe_count]
    probe_reports = []
    parent_buffers = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "minimum": float(value.float().min().item()) if value.numel() else None,
            "maximum": float(value.float().max().item()) if value.numel() else None,
        }
        for name, value in parent.named_buffers(
            recurse=True,
            remove_duplicate=False,
        )
    }
    child_buffers = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "minimum": float(value.float().min().item()) if value.numel() else None,
            "maximum": float(value.float().max().item()) if value.numel() else None,
        }
        for name, value in child.named_buffers(
            recurse=True,
            remove_duplicate=False,
        )
        if not name.startswith("neural_forge_packed_registry.")
    }
    parent.eval()
    child.eval()
    with torch.inference_mode():
        for row in probe_rows:
            encoded = packed_tokenizer(
                str(row["prompt"]),
                return_tensors="pt",
                add_special_tokens=True,
            )
            kwargs = {
                "input_ids": encoded["input_ids"],
                "use_cache": False,
            }
            if "attention_mask" in encoded:
                kwargs["attention_mask"] = encoded["attention_mask"]
            dense_logits = parent(**kwargs).logits
            packed_logits = child(**kwargs).logits
            difference = (dense_logits.float() - packed_logits.float()).abs()
            probe_reports.append(
                {
                    "case_id": str(row["case_id"]),
                    "bitwise_equal": bool(torch.equal(dense_logits, packed_logits)),
                    "maximum_absolute_delta": float(difference.max().item()),
                    "mean_absolute_delta": float(difference.mean().item()),
                    "final_argmax_equal": bool(
                        dense_logits[:, -1].argmax(dim=-1).item()
                        == packed_logits[:, -1].argmax(dim=-1).item()
                    ),
                }
            )

    passed = (
        code_mismatches == 0
        and scale_mismatches == 0
        and reconstructed_mismatches == 0
        and not tokenizer_mismatch_ids
        and all(row["bitwise_equal"] for row in probe_reports)
    )
    return {
        "passed": passed,
        "tensor_count": len(analyses),
        "checked_codes": checked_codes,
        "checked_scales": checked_scales,
        "code_mismatches": code_mismatches,
        "scale_mismatches": scale_mismatches,
        "reconstructed_mismatches": reconstructed_mismatches,
        "maximum_reconstruction_delta": maximum_reconstruction_delta,
        "tokenizer_case_count": len(envelope["rows"]),
        "tokenizer_mismatch_ids": tokenizer_mismatch_ids,
        "parent_runtime_buffers": parent_buffers,
        "child_nonpacked_runtime_buffers": child_buffers,
        "probe_reports": probe_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parent-envelope", required=True)
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--analysis-chunk-groups", type=int, default=4096)
    parser.add_argument("--probe-count", type=int, default=4)
    return parser


def main() -> int:
    result = execute(build_parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
