#!/usr/bin/env python3
"""Compare compact Qwen attention registries without loading model weights."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


class ComparisonError(RuntimeError):
    """A codebook tensor contract did not support exact comparison."""


def _tensor(checkpoint: Path, name: str) -> torch.Tensor:
    path = checkpoint / "model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        matches = [key for key in handle.keys() if key == name or key.endswith(f".{name}")]
        if len(matches) != 1:
            raise ComparisonError(f"Expected one {name!r} tensor, found {matches}.")
        return handle.get_tensor(matches[0]).detach().cpu().to(torch.long).clone()


def _registry(checkpoint: Path) -> dict[str, Any]:
    values = {
        name: _tensor(checkpoint, f"attention_codebook.registry.{name}")
        for name in ("assignments", "prototype_source_indices", "group_permutations")
    }
    if values["assignments"].ndim != 2:
        raise ComparisonError("Only [layers, groups] assignment registries are supported.")
    groups = int(values["assignments"].shape[1])
    sources = values["prototype_source_indices"]
    if sources.ndim == 1:
        if sources.numel() == 0:
            raise ComparisonError("Ragged prototype provenance is empty.")
        grouped = []
        for canonical_group in range(groups):
            physical = values["group_permutations"][:, canonical_group]
            allowed = {
                layer * groups + int(physical[layer])
                for layer in range(int(physical.numel()))
            }
            grouped.append(
                torch.tensor(
                    [int(value) for value in sources if int(value) in allowed],
                    dtype=torch.long,
                )
            )
        values["grouped_sources"] = tuple(grouped)
    elif sources.ndim == 2 and int(sources.shape[0]) == groups:
        values["grouped_sources"] = tuple(sources[group] for group in range(groups))
    else:
        raise ComparisonError("Prototype provenance has an unsupported shape.")
    return values


def compare(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = _registry(left_path)
    right = _registry(right_path)
    for name in ("assignments", "group_permutations"):
        if left[name].shape != right[name].shape:
            raise ComparisonError(f"{name} shapes differ.")
    groups = int(left["assignments"].shape[1])
    permutations_equal = torch.equal(
        left["group_permutations"], right["group_permutations"]
    )
    results = []
    suggestions = []
    for canonical_group in range(groups):
        left_sources = left["grouped_sources"][canonical_group]
        right_sources = right["grouped_sources"][canonical_group]
        left_set = {int(value) for value in left_sources}
        right_set = {int(value) for value in right_sources}
        removed = sorted(left_set - right_set)
        added = sorted(right_set - left_set)
        group_payload = {
            "canonical_group": canonical_group,
            "left_source_count": len(left_set),
            "right_source_count": len(right_set),
            "removed_source_flat_indices": removed,
            "added_source_flat_indices": added,
            "right_is_source_subset": not added,
        }
        results.append(group_payload)
        if not added:
            for source in removed:
                source_layer, source_physical_group = divmod(source, groups)
                right_slot = int(right["assignments"][source_layer, canonical_group])
                if right["prototype_source_indices"].ndim != 2:
                    continue
                target = int(
                    right["prototype_source_indices"][canonical_group, right_slot]
                )
                target_layer, target_physical_group = divmod(target, groups)
                if target not in left_set:
                    continue
                suggestions.append(
                    {
                        "canonical_group": canonical_group,
                        "source_flat_index": source,
                        "target_flat_index": target,
                        "candidate": (
                            f"{source_layer}:{source_physical_group}="
                            f"{target_layer}:{target_physical_group}"
                        ),
                    }
                )
    return {
        "schema": "neural-forge-qwen-attention-codebook-comparison-v1",
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "group_permutations_equal": permutations_equal,
        "groups": results,
        "suggested_left_to_right_redirects": suggestions,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = compare(args.left, args.right)
    if args.output is not None:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
