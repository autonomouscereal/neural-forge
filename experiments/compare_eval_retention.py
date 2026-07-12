#!/usr/bin/env python3
"""Compare parent and child eval JSON files by shared item ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return data


def _pass_map(items: Any, pass_key: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str):
            continue
        out[item_id] = bool(item.get(pass_key))
    return out


def _item_map(items: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str):
            out[item_id] = item
    return out


def _item_key(item: dict[str, Any], fallback_prefix: str, index: int) -> str:
    item_id = item.get("id")
    if isinstance(item_id, str) and item_id:
        return item_id
    prompt = item.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    return f"{fallback_prefix}_{index}"


def _loose_item_map(items: Any, fallback_prefix: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for index, item in enumerate(items):
        if isinstance(item, dict):
            out[_item_key(item, fallback_prefix, index)] = item
    return out


def _continuation(item: dict[str, Any]) -> str:
    continuation = item.get("continuation")
    if isinstance(continuation, str):
        return continuation.strip()
    output = item.get("output")
    prompt = item.get("prompt")
    if isinstance(output, str) and isinstance(prompt, str) and output.startswith(prompt):
        return output[len(prompt) :].strip()
    if isinstance(output, str):
        return output.strip()
    return ""


def _expected_summary(item: dict[str, Any]) -> dict[str, Any] | None:
    details = item.get("details")
    if isinstance(details, dict):
        return details
    expected = item.get("expected_behavior")
    if isinstance(expected, dict):
        maybe_details = expected.get("details")
        if isinstance(maybe_details, dict):
            return maybe_details
        return expected
    return None


def _sample_pairs(
    parent_items: dict[str, dict[str, Any]],
    child_items: dict[str, dict[str, Any]],
    parent_pass: dict[str, bool],
    child_pass: dict[str, bool],
    max_samples: int,
) -> list[dict[str, Any]]:
    keys = sorted(set(parent_items) | set(child_items))
    keys.sort(key=lambda key: (not parent_pass.get(key, False) or child_pass.get(key, False), key))
    samples = []
    for key in keys:
        if len(samples) >= max_samples:
            break
        parent_item = parent_items.get(key, {})
        child_item = child_items.get(key, {})
        if not parent_pass.get(key, False) and not child_pass.get(key, False):
            continue
        samples.append(
            {
                "id": key,
                "category": child_item.get("category", parent_item.get("category")),
                "kind": child_item.get("kind", parent_item.get("kind")),
                "prompt": child_item.get("prompt", parent_item.get("prompt")),
                "parent_passed": bool(parent_pass.get(key, False)),
                "child_passed": bool(child_pass.get(key, False)),
                "parent_continuation": _continuation(parent_item),
                "child_continuation": _continuation(child_item),
                "parent_details": _expected_summary(parent_item),
                "child_details": _expected_summary(child_item),
            }
        )
    return samples


def _summary(parent: dict[str, bool], child: dict[str, bool]) -> dict[str, Any]:
    ids = sorted(set(parent) | set(child))
    parent_pass = {item_id for item_id, passed in parent.items() if passed}
    child_pass = {item_id for item_id, passed in child.items() if passed}
    shared = set(parent) & set(child)
    retained = sorted(parent_pass & child_pass)
    lost = sorted(parent_pass - child_pass)
    gained = sorted(child_pass - parent_pass)
    shared_parent_pass = sorted(item_id for item_id in shared if parent.get(item_id))
    shared_retained = sorted(item_id for item_id in shared_parent_pass if child.get(item_id))
    return {
        "total_ids": len(ids),
        "shared_ids": len(shared),
        "parent_pass": len(parent_pass),
        "child_pass": len(child_pass),
        "retained_parent_pass": len(retained),
        "lost_parent_pass": len(lost),
        "child_only_or_parent_failed_gain": len(gained),
        "parent_pass_retention_rate": (len(retained) / len(parent_pass)) if parent_pass else None,
        "shared_parent_pass_retention_rate": (len(shared_retained) / len(shared_parent_pass))
        if shared_parent_pass
        else None,
        "sample_lost_parent_pass_ids": lost[:25],
        "sample_gain_ids": gained[:25],
    }


def _bucket_summary(
    parent_items: dict[str, dict[str, Any]],
    child_items: dict[str, dict[str, Any]],
    parent_pass: dict[str, bool],
    child_pass: dict[str, bool],
    field: str,
) -> dict[str, Any]:
    buckets: dict[str, dict[str, dict[str, bool]]] = {}
    for item_id, item in parent_items.items():
        label = item.get(field, "unknown")
        if not isinstance(label, str):
            label = str(label)
        bucket = buckets.setdefault(label, {"parent": {}, "child": {}})
        bucket["parent"][item_id] = parent_pass.get(item_id, False)
        bucket["child"][item_id] = child_pass.get(item_id, False)
    for item_id, item in child_items.items():
        label = item.get(field, "unknown")
        if not isinstance(label, str):
            label = str(label)
        bucket = buckets.setdefault(label, {"parent": {}, "child": {}})
        bucket["parent"].setdefault(item_id, parent_pass.get(item_id, False))
        bucket["child"][item_id] = child_pass.get(item_id, False)
    return {label: _summary(values["parent"], values["child"]) for label, values in sorted(buckets.items())}


def _behavior(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    parent_items = _item_map(parent.get("behavior_suite", {}).get("items"))
    child_items = _item_map(child.get("behavior_suite", {}).get("items"))
    parent_pass = _pass_map(list(parent_items.values()), "passed")
    child_pass = _pass_map(list(child_items.values()), "passed")
    return {
        "overall": _summary(parent_pass, child_pass),
        "by_category": _bucket_summary(parent_items, child_items, parent_pass, child_pass, "category"),
        "by_kind": _bucket_summary(parent_items, child_items, parent_pass, child_pass, "kind"),
    }


def _choice(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    parent_items = _item_map(parent.get("choice_accuracy", {}).get("items"))
    child_items = _item_map(child.get("choice_accuracy", {}).get("items"))
    parent_pass = _pass_map(list(parent_items.values()), "correct")
    child_pass = _pass_map(list(child_items.values()), "correct")
    return {"overall": _summary(parent_pass, child_pass)}


def _generation(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    parent_items = _item_map(parent.get("generation_diagnostics"))
    child_items = _item_map(child.get("generation_diagnostics"))
    parent_pass = _pass_map(list(parent_items.values()), "passed")
    child_pass = _pass_map(list(child_items.values()), "passed")
    return {
        "overall": _summary(parent_pass, child_pass),
        "by_category": _bucket_summary(parent_items, child_items, parent_pass, child_pass, "category"),
    }


def _manual_samples(parent: dict[str, Any], child: dict[str, Any], max_samples: int) -> dict[str, Any]:
    behavior_parent_items = _item_map(parent.get("behavior_suite", {}).get("items"))
    behavior_child_items = _item_map(child.get("behavior_suite", {}).get("items"))
    behavior_parent_pass = _pass_map(list(behavior_parent_items.values()), "passed")
    behavior_child_pass = _pass_map(list(behavior_child_items.values()), "passed")

    parent_generations = _loose_item_map(parent.get("generations"), "generation")
    child_generations = _loose_item_map(child.get("generations"), "generation")
    parent_generation_diagnostics = _loose_item_map(parent.get("generation_diagnostics"), "generation")
    child_generation_diagnostics = _loose_item_map(child.get("generation_diagnostics"), "generation")
    generation_parent_pass = _pass_map(list(parent_generation_diagnostics.values()), "passed")
    generation_child_pass = _pass_map(list(child_generation_diagnostics.values()), "passed")
    for key, generation in parent_generations.items():
        if key in parent_generation_diagnostics:
            generation.update(parent_generation_diagnostics[key])
    for key, generation in child_generations.items():
        if key in child_generation_diagnostics:
            generation.update(child_generation_diagnostics[key])

    return {
        "behavior": _sample_pairs(
            behavior_parent_items,
            behavior_child_items,
            behavior_parent_pass,
            behavior_child_pass,
            max_samples,
        ),
        "generation": _sample_pairs(
            parent_generations,
            child_generations,
            generation_parent_pass,
            generation_child_pass,
            max_samples,
        ),
    }


def _aggregate(data: dict[str, Any]) -> dict[str, Any]:
    behavior = data.get("behavior_suite", {})
    choice = data.get("choice_accuracy", {})
    generation = data.get("generation_health", {})
    perplexity = data.get("perplexity", {})
    return {
        "model": data.get("model"),
        "param_count": data.get("param_count"),
        "behavior_correct": behavior.get("correct"),
        "behavior_total": behavior.get("total"),
        "generation_correct": generation.get("correct"),
        "generation_total": generation.get("total"),
        "choice_correct": choice.get("correct"),
        "choice_total": choice.get("total"),
        "perplexity_mean": perplexity.get("mean"),
    }


def compare(parent_path: Path, child_path: Path, max_samples: int = 8) -> dict[str, Any]:
    parent = _load(parent_path)
    child = _load(child_path)
    parent_params = parent.get("param_count")
    child_params = child.get("param_count")
    compression = None
    if isinstance(parent_params, int | float) and isinstance(child_params, int | float) and child_params:
        compression = parent_params / child_params
    return {
        "parent_eval": str(parent_path),
        "child_eval": str(child_path),
        "parent": _aggregate(parent),
        "child": _aggregate(child),
        "param_compression_ratio": compression,
        "behavior": _behavior(parent, child),
        "generation": _generation(parent, child),
        "choice": _choice(parent, child),
        "manual_samples": _manual_samples(parent, child, max(0, int(max_samples))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_eval", type=Path)
    parser.add_argument("child_eval", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()
    report = compare(args.parent_eval, args.child_eval, max_samples=args.max_samples)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
