"""Audit large-compression quantizer geometry directly from a local checkpoint.

This experiment does not load Transformers, generate text, call a parent
runtime, train, optimize Parameters, or write a child checkpoint.  It samples
the local safetensors weights and compares:

* row-aligned group sizes;
* four- through eight-bit symmetric scalar codes;
* exact removal of the largest values in each group; and
* per-group MSE clipping selected from a sealed ratio grid.

The report is diagnostic evidence for the CREVF option builder.  It does not
claim behavioral retention: every selected representation still has to pass
the standalone child and parent-relative intelligence gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


REPORT_SCHEMA = "neural-forge-lfm2-quantizer-geometry-audit-v1"
DEFAULT_GROUP_SIZES = (32, 64, 128, 256)
DEFAULT_BIT_WIDTHS = (4, 5, 6, 7, 8)
DEFAULT_EXACT_OUTLIERS = (0, 1, 2, 4)
DEFAULT_CLIP_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_int_list(raw: str, *, minimum: int, maximum: int) -> tuple[int, ...]:
    try:
        values = tuple(int(piece.strip()) for piece in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if (
        not values
        or len(set(values)) != len(values)
        or any(value < minimum or value > maximum for value in values)
    ):
        raise argparse.ArgumentTypeError(
            f"Values must be unique integers in [{minimum}, {maximum}]."
        )
    return values


def _parse_float_list(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(piece.strip()) for piece in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers.") from exc
    if (
        not values
        or len(set(values)) != len(values)
        or any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in values)
        or 1.0 not in values
    ):
        raise argparse.ArgumentTypeError(
            "Clip ratios must be unique values in (0, 1] and include 1.0."
        )
    return values


def _sample_group_indices(
    group_count: int,
    *,
    sample_stride: int,
    max_groups_per_tensor: int,
) -> torch.Tensor:
    sample_count = min(
        group_count,
        max_groups_per_tensor,
        max(1, math.ceil(group_count / sample_stride)),
    )
    if sample_count == group_count:
        return torch.arange(group_count, dtype=torch.long)
    # Midpoint sampling gives every equal-width interval one representative
    # and avoids systematically over-sampling the beginning of tensors.
    positions = (
        (torch.arange(sample_count, dtype=torch.float64) + 0.5)
        * float(group_count)
        / float(sample_count)
    )
    return positions.floor().clamp_max(group_count - 1).to(torch.long)


def _sample_groups(
    values: torch.Tensor,
    *,
    group_size: int,
    sample_stride: int,
    max_groups_per_tensor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    value_count = int(flat.numel())
    group_count = (value_count + group_size - 1) // group_size
    selected = _sample_group_indices(
        group_count,
        sample_stride=sample_stride,
        max_groups_per_tensor=max_groups_per_tensor,
    )
    offsets = selected[:, None] * group_size + torch.arange(
        group_size,
        dtype=torch.long,
    )[None, :]
    valid = offsets < value_count
    safe_offsets = offsets.clamp_max(value_count - 1)
    groups = flat.index_select(0, safe_offsets.reshape(-1)).reshape(
        -1,
        group_size,
    )
    groups.masked_fill_(~valid, 0.0)
    return groups, valid


def _empty_record() -> dict[str, Any]:
    return {
        "sampled_values": 0,
        "source_square_sum": 0.0,
        "absmax_square_error": 0.0,
        "best_clip_square_error": 0.0,
        "best_clip_ratio_counts": Counter(),
    }


def _quantization_errors(
    residual: torch.Tensor,
    valid: torch.Tensor,
    *,
    bit_width: int,
    clip_ratios: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qmax = (1 << (bit_width - 1)) - 1
    maxima = residual.abs().amax(dim=1)
    best_sse: torch.Tensor | None = None
    best_ratio_index = torch.zeros(
        residual.shape[0],
        dtype=torch.long,
        device=residual.device,
    )
    absmax_sse: torch.Tensor | None = None
    for ratio_index, ratio in enumerate(clip_ratios):
        clipped_maximum = maxima * ratio
        scale32 = torch.where(
            clipped_maximum > 0,
            clipped_maximum / float(qmax),
            torch.ones_like(clipped_maximum),
        )
        # Match the checkpoint codec: scales are persisted as float16.
        scale = scale32.to(torch.float16).to(torch.float32)
        scale = torch.where(scale > 0, scale, torch.full_like(scale, 2**-24))
        signed = torch.round(residual / scale[:, None]).clamp(-qmax, qmax)
        reconstructed = signed * scale[:, None]
        square_error = ((reconstructed - residual) ** 2).masked_fill(~valid, 0.0)
        group_sse = square_error.sum(dim=1, dtype=torch.float64)
        if ratio == 1.0:
            absmax_sse = group_sse
        if best_sse is None:
            best_sse = group_sse
            best_ratio_index.fill_(ratio_index)
        else:
            improved = group_sse < best_sse
            best_sse = torch.where(improved, group_sse, best_sse)
            best_ratio_index = torch.where(
                improved,
                torch.full_like(best_ratio_index, ratio_index),
                best_ratio_index,
            )
    if absmax_sse is None or best_sse is None:
        raise AssertionError("The sealed clipping grid lost its 1.0 control.")
    return absmax_sse, best_sse, best_ratio_index


def _update_records(
    records: dict[tuple[int, int], dict[str, Any]],
    groups: torch.Tensor,
    valid: torch.Tensor,
    *,
    bit_widths: tuple[int, ...],
    exact_outlier_counts: tuple[int, ...],
    clip_ratios: tuple[float, ...],
) -> None:
    sampled_values = int(valid.sum().item())
    source_square_sum = float(
        (groups**2).masked_fill(~valid, 0.0).sum(dtype=torch.float64).item()
    )
    for exact_count in exact_outlier_counts:
        residual = groups.clone()
        if exact_count:
            count = min(exact_count, residual.shape[1])
            exact_indices = residual.abs().topk(count, dim=1).indices
            residual.scatter_(1, exact_indices, 0.0)
        for bit_width in bit_widths:
            absmax_sse, clipped_sse, ratio_indices = _quantization_errors(
                residual,
                valid,
                bit_width=bit_width,
                clip_ratios=clip_ratios,
            )
            record = records[(bit_width, exact_count)]
            record["sampled_values"] += sampled_values
            record["source_square_sum"] += source_square_sum
            record["absmax_square_error"] += float(absmax_sse.sum().item())
            record["best_clip_square_error"] += float(clipped_sse.sum().item())
            counts = torch.bincount(
                ratio_indices.to(device="cpu"),
                minlength=len(clip_ratios),
            )
            for index, count in enumerate(counts.tolist()):
                record["best_clip_ratio_counts"][str(clip_ratios[index])] += int(count)


def audit_quantizer_geometry(
    *,
    model_file: Path,
    device: str,
    group_sizes: tuple[int, ...],
    bit_widths: tuple[int, ...],
    exact_outlier_counts: tuple[int, ...],
    clip_ratios: tuple[float, ...],
    sample_stride: int,
    max_groups_per_tensor: int,
) -> dict[str, Any]:
    if not model_file.is_file() or model_file.is_symlink():
        raise ValueError("model_file must be a regular local safetensors file.")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("A CUDA audit was requested but CUDA is unavailable.")
    if sample_stride < 1 or max_groups_per_tensor < 1:
        raise ValueError("Sampling limits must be positive integers.")

    aggregate = {
        group_size: {
            (bit_width, exact_count): _empty_record()
            for bit_width in bit_widths
            for exact_count in exact_outlier_counts
        }
        for group_size in group_sizes
    }
    tensor_inventory: list[dict[str, Any]] = []
    with safe_open(model_file, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            values = handle.get_tensor(name)
            if not values.is_floating_point() or values.numel() == 0:
                continue
            shape = tuple(int(value) for value in values.shape)
            tensor_entry: dict[str, Any] = {
                "name": name,
                "shape": list(shape),
                "numel": int(values.numel()),
                "source_dtype": str(values.dtype),
                "group_sizes": {},
            }
            for group_size in group_sizes:
                row_aligned = len(shape) <= 1 or shape[-1] % group_size == 0
                if not row_aligned:
                    tensor_entry["group_sizes"][str(group_size)] = {
                        "eligible": False,
                        "reason": "last_dimension_not_divisible",
                    }
                    continue
                groups, valid = _sample_groups(
                    values,
                    group_size=group_size,
                    sample_stride=sample_stride,
                    max_groups_per_tensor=max_groups_per_tensor,
                )
                tensor_entry["group_sizes"][str(group_size)] = {
                    "eligible": True,
                    "sampled_groups": int(groups.shape[0]),
                    "sampled_values": int(valid.sum().item()),
                }
                _update_records(
                    aggregate[group_size],
                    groups.to(target_device),
                    valid.to(target_device),
                    bit_widths=bit_widths,
                    exact_outlier_counts=exact_outlier_counts,
                    clip_ratios=clip_ratios,
                )
                del groups, valid
            tensor_inventory.append(tensor_entry)
            del values

    frontiers: dict[str, Any] = {}
    for group_size, records in aggregate.items():
        group_report: dict[str, Any] = {}
        for (bit_width, exact_count), record in sorted(records.items()):
            sampled_values = int(record["sampled_values"])
            source_square_sum = float(record["source_square_sum"])
            absmax_sse = float(record["absmax_square_error"])
            clip_sse = float(record["best_clip_square_error"])
            group_report[f"b{bit_width}-exact{exact_count}"] = {
                "bit_width": bit_width,
                "exact_outliers_per_sampled_group": exact_count,
                "sampled_values": sampled_values,
                "source_rms": (
                    math.sqrt(source_square_sum / sampled_values)
                    if sampled_values
                    else None
                ),
                "absmax_rmse": (
                    math.sqrt(absmax_sse / sampled_values) if sampled_values else None
                ),
                "best_clip_rmse": (
                    math.sqrt(clip_sse / sampled_values) if sampled_values else None
                ),
                "absmax_relative_rmse": (
                    math.sqrt(absmax_sse / source_square_sum)
                    if source_square_sum
                    else None
                ),
                "best_clip_relative_rmse": (
                    math.sqrt(clip_sse / source_square_sum)
                    if source_square_sum
                    else None
                ),
                "clip_square_error_reduction": (
                    1.0 - clip_sse / absmax_sse if absmax_sse else 0.0
                ),
                "best_clip_ratio_counts": dict(
                    sorted(record["best_clip_ratio_counts"].items())
                ),
            }
        frontiers[str(group_size)] = group_report

    return {
        "schema": REPORT_SCHEMA,
        "method": {
            "kind": "local_weight_geometry_diagnostic_only",
            "training": False,
            "generation": False,
            "parent_runtime_calls": 0,
            "child_checkpoint_written": False,
            "row_alignment_required": True,
            "scale_dtype": "torch.float16",
            "quantizer": "symmetric_scalar_with_exact_outlier_peeling",
            "clip_selection": "per_group_minimum_sampled_weight_mse",
        },
        "source": {
            "file": str(model_file.resolve()),
            "size_bytes": model_file.stat().st_size,
            "sha256": _sha256_file(model_file),
        },
        "sampling": {
            "sample_stride": sample_stride,
            "max_groups_per_tensor": max_groups_per_tensor,
            "device": str(target_device),
        },
        "search_space": {
            "group_sizes": list(group_sizes),
            "bit_widths": list(bit_widths),
            "exact_outlier_counts": list(exact_outlier_counts),
            "clip_ratios": list(clip_ratios),
        },
        "frontiers": frontiers,
        "tensors": tensor_inventory,
        "interpretation_guard": (
            "Weight-space RMSE only diagnoses codec geometry. It cannot promote a "
            "fold; cached causal evidence and coherent standalone behavior remain "
            "mandatory."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--group-sizes",
        type=lambda value: _parse_int_list(value, minimum=8, maximum=4096),
        default=DEFAULT_GROUP_SIZES,
    )
    parser.add_argument(
        "--bit-widths",
        type=lambda value: _parse_int_list(value, minimum=2, maximum=8),
        default=DEFAULT_BIT_WIDTHS,
    )
    parser.add_argument(
        "--exact-outliers",
        type=lambda value: _parse_int_list(value, minimum=0, maximum=64),
        default=DEFAULT_EXACT_OUTLIERS,
    )
    parser.add_argument(
        "--clip-ratios",
        type=_parse_float_list,
        default=DEFAULT_CLIP_RATIOS,
    )
    parser.add_argument("--sample-stride", type=int, default=128)
    parser.add_argument("--max-groups-per-tensor", type=int, default=32768)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = audit_quantizer_geometry(
        model_file=args.model_file,
        device=args.device,
        group_sizes=tuple(args.group_sizes),
        bit_widths=tuple(args.bit_widths),
        exact_outlier_counts=tuple(args.exact_outliers),
        clip_ratios=tuple(args.clip_ratios),
        sample_stride=args.sample_stride,
        max_groups_per_tensor=args.max_groups_per_tensor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "schema": report["schema"]}))


if __name__ == "__main__":
    main()
