"""Frozen 2026 operator-authored basic semantic diagnostics; no model execution.

Admission screens all supplied source rows, including answers, and separates
private oracles from model inputs. This is bounded lexical/source exclusion,
not proof of semantic novelty or general intelligence.
"""
from __future__ import annotations

import argparse
import collections
import json
from fractions import Fraction
from pathlib import Path

from . import evaluation as e
from .corpus import canonical, digest, file_sha256, iter_jsonl, strings, words

DATE = "2026-10-13"


def build_tasks():
    tasks = []
    def add(family, index, route, prompt, oracle):
        tasks.append({"schema": "recovery_eval_task.v1", "id": f"text-eval-{DATE}-{family}-{index}",
            "group_id": f"operator-text-eval-{DATE}-{family}", "split": "test", "route": route,
            "input": {"messages": [{"role": "user", "content": prompt}], "tools": [], "media": []},
            "oracle": oracle, "provenance": {"synthetic_operator_authored": True, "original_authored_date": DATE,
                "source": "explicit diagnostic definitions in text_eval_seed.py", "difficulty": "basic_diagnostic",
                "model_generated": False, "source_code_sha256": file_sha256(Path(__file__))}})
    def arithmetic(family, i, prompt, value):
        add(family, i, "text_arithmetic", prompt + " Give the final numeric answer on the last line.",
            {"kind": "arithmetic", "expected": str(value), "absolute_tolerance": "0"})
    for i, (a, b) in enumerate([(-17, 6), (23, 14), (-29, -8), (37, 19)]):
        arithmetic("signed_product", i, f"Compute the integer product of {a} and {b}.", a * b)
    for i, (a, b, c) in enumerate([(19, 7, 6), (-13, 8, 9), (25, -4, 11), (6, 17, -3)]):
        arithmetic("precedence", i, f"Evaluate {a} + ({b}) * ({c}) using standard operation precedence.", a + b * c)
    for i, (a, b, c, d) in enumerate([(2, 7, 3, 14), (5, 12, 7, 18), (-3, 8, 5, 6), (11, 15, -2, 9)]):
        arithmetic("fraction_sum", i, f"Find the exact sum {a}/{b} + {c}/{d}. A fraction is acceptable.", Fraction(a, b) + Fraction(c, d))
    for i, (budget, count, cost, shipping) in enumerate([(5000, 7, 325, 180), (7200, 11, 415, 275), (6300, 9, 560, 240), (8500, 13, 375, 320)]):
        arithmetic("budget", i, f"A budget has {budget} cents. Buy {count} items costing {cost} cents each and pay {shipping} cents for shipping. How many cents remain?", budget - count * cost - shipping)
    for i, (lo, hi, divisor) in enumerate([(-12, 25, 4), (7, 63, 7), (-31, -2, 5), (11, 89, 6)]):
        count = hi // divisor - (lo - 1) // divisor
        arithmetic("integer_interval", i, f"How many integers in the inclusive interval from {lo} through {hi} are divisible by {divisor}?", count)
    for i, (a, b, total) in enumerate([(7, 9, 72), (-4, 13, 37), (6, -11, 10), (9, 14, 29)]):
        arithmetic("linear_equation", i, f"Solve ({a}) * x + ({b}) = {total}. Give the exact numeric value of x.", Fraction(total - b, a))

    definitions = [
        ("stable_unique", "Return the distinct integers in xs, keeping the order of their first occurrence.",
         [([[4, 2, 4, 3, 2]], [4, 2, 3]), ([[]], []), ([[0, -1, 0, -1, 2]], [0, -1, 2]), ([[8, 8]], [8])]),
        ("even_count", "Return how many integers in xs are even. Zero and negative even integers count.",
         [([[-4, -3, 0, 7, 8]], 3), ([[]], 0), ([[1, 3, 5]], 0), ([[2, 2, -2]], 3)]),
        ("prefix_totals", "Return a list of running sums of xs. The ith result includes xs[i]. Return [] for empty input.",
         [([[2, -5, 4]], [2, -3, 1]), ([[]], []), ([[0, 0, 1]], [0, 0, 1]), ([[-3]], [-3])]),
        ("rotate_left", "Return xs rotated left by k positions. k is a nonnegative integer and can exceed len(xs). Empty input returns [].",
         [([[1, 2, 3, 4], 2], [3, 4, 1, 2]), ([[], 9], []), ([[5, 6, 7], 8], [7, 5, 6]), ([[9], 0], [9])]),
        ("clamp_values", "Return a new list replacing every x in xs by low if x < low, high if x > high, and otherwise x. low <= high.",
         [([[-4, 0, 5, 9], -1, 5], [-1, 0, 5, 5]), ([[], 0, 1], []), ([[3, 1, -1], 2, 2], [2, 2, 2]), ([[1, 2], 0, 4], [1, 2])]),
        ("count_runs", "Return the number of maximal consecutive runs of equal integers in xs. For empty input return 0.",
         [([[1, 1, 2, 2, 2, 1]], 3), ([[]], 0), ([[7, 7]], 1), ([[0, -1, 0, -1]], 4)]),
        ("nondecreasing", "Return True if each integer in xs is at least the preceding one, otherwise False. Empty and one-element lists return True.",
         [([[1, 1, 3]], True), ([[]], True), ([[2, 1]], False), ([[-4, -2, 0]], True)]),
        ("adjacent_differences", "Return [xs[1]-xs[0], xs[2]-xs[1], ...]. Empty and one-element input returns [].",
         [([[5, 2, 9]], [-3, 7]), ([[]], []), ([[1]], []), ([[-2, -2, 3]], [0, 5])]),
        ("positive_count", "Return the number of strictly positive integers in xs. Zero does not count.",
         [([[-2, 0, 3, 5]], 2), ([[]], 0), ([[0, 0]], 0), ([[1, 1, -1]], 2)]),
        ("median_three", "xs always contains exactly three integers. Return its middle value in sorted order, retaining duplicate values when finding the middle.",
         [([[9, 1, 5]], 5), ([[3, 3, 1]], 3), ([[-8, -2, -4]], -4), ([[7, 7, 7]], 7)]),
        ("argmin_first", "Return the index of the first occurrence of the smallest integer in xs. For empty input return -1.",
         [([[4, 1, 1, 3]], 1), ([[]], -1), ([[-3, -8, -5]], 1), ([[0]], 0)]),
        ("intersection_order", "Return the unique integers appearing in both xs and ys, in their first occurrence order in xs.",
         [([[3, 1, 3, 2], [2, 3]], [3, 2]), ([[], [1]], []), ([[1, 2], []], []), ([[4, 4, -1], [-1, 4]], [4, -1])]),
    ]
    for family, description, cases in definitions:
        arguments = {"rotate_left": "xs, k", "clamp_values": "xs, low, high", "intersection_order": "xs, ys"}.get(family, "xs")
        prompt = f"Write a Python function {family}({arguments}). {description} Use only pure Python operations, no imports or file/network access. Include the function in your answer."
        add(family, 0, "text_code", prompt, {"kind": "python_function", "function": family,
            "cases": [{"args": args, "expected": expected} for args, expected in cases]})
    def instruction(family, i, prompt, expected, properties):
        add(family, i, "text_instruction", prompt + " Output only one JSON object, with no markdown or commentary.",
            {"kind": "instruction_json", "expected": expected, "schema": {"type": "object", "properties": properties,
                "required": sorted(properties), "additionalProperties": False}})
    for i, values in enumerate([[7, 4, 4, -2, 3, 0, -2], [5, 8, 1, 8, 6, 0, 6]]):
        expected = list(dict.fromkeys(x for x in values if x % 2 == 0))
        instruction("filter_order", i, f"From {values}, select even integers, remove later duplicates and preserve first occurrence order. Put the result under the key values.",
                    {"values": expected}, {"values": {"type": "array", "items": {"type": "integer"}}})
    for i, values in enumerate([["pear", "fig", "plum", "kiwi", "a"], ["oak", "elm", "pine", "beech", "yew"]]):
        instruction("sort_words", i, f"Sort these words by increasing character count, breaking equal lengths alphabetically: {values}. Put the result under words.",
                    {"words": sorted(values, key=lambda x: (len(x), x))}, {"words": {"type": "array", "items": {"type": "string"}}})
    for i, values in enumerate([[{"id": "c", "score": 4}, {"id": "a", "score": 9}, {"id": "b", "score": 9}],
                               [{"id": "z", "score": -2}, {"id": "y", "score": -1}, {"id": "x", "score": -1}]]):
        chosen = sorted(values, key=lambda x: (-x["score"], x["id"]))[0]
        instruction("record_selection", i, f"Choose the record with the highest score, breaking a tie by alphabetically smallest id: {json.dumps(values)}. Return its id and score as the only object keys.",
                    chosen, {"id": {"type": "string"}, "score": {"type": "integer"}})
    for i, value in enumerate(["  BLUE river BLUE  ", "  GREEN hill GREEN  "]):
        transformed = value.strip().lower().replace("blue" if i == 0 else "green", "red")
        instruction("transform_text", i, f"Take the string {json.dumps(value)}, trim only its outer whitespace, lowercase it, then replace every occurrence of {'blue' if i == 0 else 'green'} with red. Return the string under text and its character count under length.",
                    {"text": transformed, "length": len(transformed)}, {"text": {"type": "string"}, "length": {"type": "integer"}})
    return tasks


def source_records(path):
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("screen source JSON must be an array")
        yield from value


def screen(tasks, paths):
    """Bounded memory candidate-index scan; all source text, not an exact-only test."""
    candidates = {}
    ngram_index = collections.defaultdict(set)
    exact_index = collections.defaultdict(set)
    for task in tasks:
        tokens = tuple(words(task["input"]["messages"][0]["content"]))
        grams = {tokens[i:i + 13] for i in range(len(tokens) - 12)}
        candidates[task["id"]] = (tokens, grams)
        exact_index[tokens].add(task["id"])
        for gram in grams:
            ngram_index[gram].add(task["id"])
    blockers = collections.defaultdict(list)
    pins, counts, source_groups = {}, {}, set()
    for path in paths:
        key = str(path.resolve())
        pins[key] = file_sha256(path)
        count = 0
        for record in source_records(path):
            count += 1
            if isinstance(record, dict):
                prov = record.get("provenance") or {}
                group = record.get("group_id") or prov.get("recovery_group_id") or prov.get("group_id")
                if group:
                    source_groups.add(group)
                texts = list(strings(record.get("input", record.get("messages", record))))
                source_id = record.get("id", record.get("uid", count))
            else:
                texts, source_id = [record], count
            # Each message and full concatenation are screened independently;
            # identical user prompts cannot hide inside a long conversation.
            texts.append("\n".join(texts))
            for text in texts:
                tokens = tuple(words(text))
                for tid in exact_index.get(tokens, ()):
                    blockers[tid].append({"source": key, "source_id": source_id, "reason": "exact_normalized_text"})
                hits = collections.defaultdict(set)
                for i in range(len(tokens) - 12):
                    gram = tokens[i:i + 13]
                    for tid in ngram_index.get(gram, ()):
                        hits[tid].add(gram)
                for tid, overlap in hits.items():
                    total = len(candidates[tid][1])
                    if total and len(overlap) / total > .5:
                        blockers[tid].append({"source": key, "source_id": source_id, "reason": "candidate_13gram_overlap_gt_half", "overlap": len(overlap), "candidate_ngrams": total})
        if pins[key] != file_sha256(path):
            raise ValueError("screen source changed while being inspected")
        counts[key] = count
    for task in tasks:
        if task["group_id"] in source_groups:
            blockers[task["id"]].append({"reason": "source_group_identity_overlap"})
    return dict(blockers), pins, counts, source_groups


def prepare(paths, output):
    if output.exists():
        raise ValueError("retain earlier frozen evaluation artifacts")
    tasks = build_tasks()
    blockers, pins, counts, source_groups = screen(tasks, paths)
    accepted = [task for task in tasks if task["id"] not in blockers]
    if not paths or not accepted:
        raise ValueError("no screen sources or no independent admitted fixtures")
    output.mkdir(parents=True)
    tasks_path, inputs_path = output / "tasks.jsonl", output / "inputs.jsonl"
    tasks_path.write_bytes(b"".join(canonical(t) + b"\n" for t in accepted))
    inputs_path.write_bytes(b"".join(canonical({"id": t["id"], "group_id": t["group_id"], "split": t["split"],
        "input": t["input"], "input_sha256": digest(t["input"])}) + b"\n" for t in accepted))
    (output / "quarantine.json").write_bytes(canonical(blockers) + b"\n")
    manifest = {"schema": "recovery_eval_text_addendum.v1", "authored_date": DATE,
        "source_code_sha256": file_sha256(Path(__file__)), "grader_sha256": file_sha256(Path(e.__file__)),
        "synthetic_operator_authored": True, "model_evaluated": False, "attempted": len(tasks), "accepted": len(accepted),
        "route_counts": dict(collections.Counter(t["route"] for t in accepted)), "groups": sorted({t["group_id"] for t in accepted}),
        "source_groups_disjoint": not bool(source_groups & {t["group_id"] for t in accepted}),
        "screen_sources_sha256": pins, "screen_source_rows": counts, "screen_policy": "exact normalized message text or >50% candidate unique 13grams anywhere in source messages/full text; source-group equality",
        "limits": ["basic diagnostic fixtures; no IQ/general-intelligence claim", "bounded lexical screening does not prove semantic novelty", "safe AST code subset; unsupported valid programs require review", "strict JSON only where task explicitly requires it", "no model-generated outputs or native media performance measured"],
        "tasks": {"path": str(tasks_path.resolve()), "sha256": file_sha256(tasks_path)},
        "inputs": {"path": str(inputs_path.resolve()), "sha256": file_sha256(inputs_path)}}
    manifest["seal_sha256"] = digest(manifest)
    (output / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.screen, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
