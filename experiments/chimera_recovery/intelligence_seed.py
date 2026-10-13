"""2026 training-only arithmetic/finite-reasoning examples with executable proof.

These are authored task specifications, not harvested/model-generated answers.
The checker uses different algorithms/invariants from the answer construction.
Admission scans supplied sources (including frozen inputs) without modifying them.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction
import itertools
import json
from pathlib import Path
import random
import tempfile

from .corpus import canonical, digest, file_sha256
from .text_eval_seed import screen

DATE = "2026-10-13"
FAMILIES = ("invoice_tax", "euclidean_division", "metric_area", "percentage_change",
            "weighted_mean", "clock_duration", "binary_decode", "affine_recurrence",
            "graph_reachability", "dag_path_count", "boolean_models", "interval_selection")
REASONING = frozenset(FAMILIES[8:])


def money_round(value):
    # Nonnegative rational cents, round half upwards without floating point.
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def parameters(family, rng):
    if family == "invoice_tax":
        return dict(cents=rng.randint(50, 20000), quantity=rng.randint(2, 17),
                    discount=rng.randint(1, 35), tax=rng.randint(1, 15), shipping=rng.randint(0, 999))
    if family == "euclidean_division":
        return dict(a=rng.randint(-9999, 9999), b=rng.randint(2, 79))
    if family == "metric_area":
        return dict(width_cm=rng.randint(3, 997), height_cm=rng.randint(3, 997))
    if family == "percentage_change":
        old = rng.randint(2, 999)
        return dict(old=old, new=rng.randint(0, 1999))
    if family == "weighted_mean":
        return dict(values=[rng.randint(-50, 80) for _ in range(4)], weights=[rng.randint(1, 9) for _ in range(4)])
    if family == "clock_duration":
        return dict(start_hour=rng.randrange(24), start_minute=rng.randrange(60), elapsed_minutes=rng.randint(1, 7000))
    if family == "binary_decode":
        return dict(bits="0" * rng.randrange(3) + format(rng.randint(0, 4095), "b"))
    if family == "affine_recurrence":
        return dict(a0=rng.randint(-15, 15), multiplier=rng.randint(-2, 3), increment=rng.randint(-9, 9), n=rng.randint(2, 7))
    if family == "graph_reachability":
        edges = [(a, b) for a in range(6) for b in range(6) if a != b and rng.random() < .23]
        return dict(nodes=list(range(6)), edges=edges, start=rng.randrange(6))
    if family == "dag_path_count":
        edges = [(a, b) for a in range(7) for b in range(a + 1, 7) if rng.random() < .4]
        return dict(nodes=list(range(7)), edges=edges, start=0, end=6)
    if family == "boolean_models":
        planted = [rng.choice([True, False]) for _ in range(5)]
        clauses = []
        for _ in range(9):
            indexes = rng.sample(range(5), rng.randint(1, 3))
            literals = [(index, rng.choice([True, False])) for index in indexes]
            if not any(planted[index] == positive for index, positive in literals):
                index, _ = literals[0]
                literals[0] = (index, planted[index])
            clauses.append(literals)
        return dict(names=list("ABCDE"), clauses=clauses)
    if family == "interval_selection":
        intervals = []
        for i in range(7):
            start = rng.randrange(24)
            intervals.append(dict(id=chr(65 + i), start=start, end=start + rng.randint(1, 9)))
        return dict(intervals=intervals)
    raise ValueError("unknown_authored_family")


def answer(family, p):
    """Answer-construction path; verifier below independently checks the result."""
    if family == "invoice_tax":
        subtotal = p["cents"] * p["quantity"]
        exact = Fraction(subtotal * (100 - p["discount"]) * (100 + p["tax"]), 10000) + p["shipping"]
        return dict(subtotal_cents=subtotal, exact_cents=str(exact), rounded_cents=money_round(exact))
    if family == "euclidean_division":
        q, r = divmod(p["a"], p["b"])
        return dict(quotient=q, remainder=r)
    if family == "metric_area":
        area = p["width_cm"] * p["height_cm"]
        return dict(square_centimeters=area, square_meters=str(Fraction(area, 10000)))
    if family == "percentage_change":
        return dict(change=p["new"] - p["old"], percent=str(Fraction(100 * (p["new"] - p["old"]), p["old"])))
    if family == "weighted_mean":
        total = sum(v * w for v, w in zip(p["values"], p["weights"]))
        return dict(weighted_sum=total, total_weight=sum(p["weights"]), mean=str(Fraction(total, sum(p["weights"]))))
    if family == "clock_duration":
        total = 60 * p["start_hour"] + p["start_minute"] + p["elapsed_minutes"]
        day, minute = divmod(total, 1440)
        hour, minute = divmod(minute, 60)
        return dict(day_offset=day, time=f"{hour:02}:{minute:02}")
    if family == "binary_decode":
        value = 0
        for bit in p["bits"]:
            value = value * 2 + int(bit)
        return dict(decimal=value)
    if family == "affine_recurrence":
        values = [p["a0"]]
        for _ in range(p["n"]):
            values.append(p["multiplier"] * values[-1] + p["increment"])
        return dict(values=values, result=values[-1])
    if family == "graph_reachability":
        adjacency = {node: [] for node in p["nodes"]}
        for a, b in p["edges"]:
            adjacency[a].append(b)
        seen, queue = {p["start"]}, deque([p["start"]])
        while queue:
            for node in adjacency[queue.popleft()]:
                if node not in seen:
                    seen.add(node)
                    queue.append(node)
        return dict(reachable=sorted(seen))
    if family == "dag_path_count":
        counts = [0] * len(p["nodes"])
        counts[p["start"]] = 1
        for node in p["nodes"]:
            for a, b in p["edges"]:
                if a == node:
                    counts[b] += counts[a]
        return dict(paths=counts[p["end"]])
    if family == "boolean_models":
        models = [bits for bits in itertools.product([False, True], repeat=len(p["names"]))
                  if all(any(bits[index] == positive for index, positive in clause) for clause in p["clauses"])]
        forced, undetermined = {}, []
        for i, name in enumerate(p["names"]):
            values = {bits[i] for bits in models}
            if len(values) == 1:
                forced[name] = values.pop()
            else:
                undetermined.append(name)
        return dict(model_count=len(models), forced=forced, undetermined=undetermined)
    if family == "interval_selection":
        chosen, end = [], -1
        for item in sorted(p["intervals"], key=lambda item: (item["end"], item["start"], item["id"])):
            if item["start"] >= end:
                chosen.append(item["id"])
                end = item["end"]
        return dict(maximum=len(chosen), selected=chosen)
    raise ValueError("unknown_authored_family")


def verify_answer(family, p, a):
    if not isinstance(a, dict):
        return False
    integer_fields = {"invoice_tax": ["subtotal_cents", "rounded_cents"], "euclidean_division": ["quotient", "remainder"],
                      "metric_area": ["square_centimeters"], "percentage_change": ["change"],
                      "weighted_mean": ["weighted_sum", "total_weight"], "clock_duration": ["day_offset"],
                      "binary_decode": ["decimal"], "affine_recurrence": ["result"], "dag_path_count": ["paths"],
                      "boolean_models": ["model_count"], "interval_selection": ["maximum"]}
    if any(type(a.get(key)) is not int for key in integer_fields.get(family, [])):
        return False
    if family in {"graph_reachability", "affine_recurrence"}:
        value = a.get("reachable" if family == "graph_reachability" else "values")
        if not isinstance(value, list) or any(type(x) is not int for x in value):
            return False
    if family == "boolean_models" and (not isinstance(a.get("forced"), dict) or any(type(x) is not bool for x in a["forced"].values())):
        return False
    try:
        return _check_answer(family, p, a)
    except (ArithmeticError, KeyError, TypeError, ValueError):
        return False


def _check_answer(family, p, a):
    """Independent executable checks, including actual witnesses and boundaries."""
    if family == "invoice_tax":
        with localcontext() as ctx:
            ctx.prec = 40
            subtotal = Decimal(p["cents"]) * p["quantity"]
            exact = subtotal * (Decimal(1) - Decimal(p["discount"]) / 100) * (Decimal(1) + Decimal(p["tax"]) / 100) + p["shipping"]
            return a == dict(subtotal_cents=int(subtotal), exact_cents=str(Fraction(exact)), rounded_cents=int(exact.quantize(Decimal(1), rounding=ROUND_HALF_UP)))
    if family == "euclidean_division":
        return set(a) == {"quotient", "remainder"} and all(type(a[k]) is int for k in a) and p["a"] == a["quotient"] * p["b"] + a["remainder"] and 0 <= a["remainder"] < p["b"]
    if family == "metric_area":
        return a == dict(square_centimeters=p["width_cm"] * p["height_cm"], square_meters=str(Fraction(p["width_cm"], 100) * Fraction(p["height_cm"], 100)))
    if family == "percentage_change":
        return a.get("change") + p["old"] == p["new"] and set(a) == {"change", "percent"} and Fraction(a["percent"]) * p["old"] == 100 * a["change"]
    if family == "weighted_mean":
        # Repeated addition independently checks the multiplication/sum path.
        total = sum(sum([value] * weight) for value, weight in zip(p["values"], p["weights"]))
        return set(a) == {"weighted_sum", "total_weight", "mean"} and a["weighted_sum"] == total and a["total_weight"] == sum(p["weights"]) and Fraction(a["mean"]) * a["total_weight"] == total
    if family == "clock_duration":
        try:
            hour, minute = [int(x) for x in a["time"].split(":")]
        except (KeyError, ValueError, AttributeError):
            return False
        return set(a) == {"day_offset", "time"} and type(a["day_offset"]) is int and a["day_offset"] >= 0 and 0 <= hour < 24 and 0 <= minute < 60 and a["time"] == f"{hour:02}:{minute:02}" and 1440 * a["day_offset"] + 60 * hour + minute == 60 * p["start_hour"] + p["start_minute"] + p["elapsed_minutes"]
    if family == "binary_decode":
        return a == {"decimal": int(p["bits"], 2)}
    if family == "affine_recurrence":
        m, c, initial = p["multiplier"], p["increment"], p["a0"]
        closed = [initial + n * c if m == 1 else initial * m ** n + c * (m ** n - 1) // (m - 1) for n in range(p["n"] + 1)]
        return a == dict(values=closed, result=closed[-1])
    if family == "graph_reachability":
        n = len(p["nodes"])
        matrix = [[i == j or (i, j) in [tuple(edge) for edge in p["edges"]] for j in range(n)] for i in range(n)]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    matrix[i][j] |= matrix[i][k] and matrix[k][j]
        return a == {"reachable": [i for i in range(n) if matrix[p["start"]][i]]}
    if family == "dag_path_count":
        edges = {tuple(edge) for edge in p["edges"]}
        middle = [n for n in p["nodes"] if p["start"] < n < p["end"]]
        total = 0
        for count in range(len(middle) + 1):
            for subset in itertools.combinations(middle, count):
                path = [p["start"], *subset, p["end"]]
                total += all((x, y) in edges for x, y in zip(path, path[1:]))
        return a == {"paths": total}
    if family == "boolean_models":
        models = []
        for bits in range(1 << len(p["names"])):
            if all(any(bool(bits & (1 << index)) == positive for index, positive in clause) for clause in p["clauses"]):
                models.append(bits)
        if not models:
            return False
        forced, undecided = {}, []
        for index, name in enumerate(p["names"]):
            true_count = sum(bool(bits & (1 << index)) for bits in models)
            if true_count == 0 or true_count == len(models):
                forced[name] = true_count == len(models)
            else:
                undecided.append(name)
        return a == dict(model_count=len(models), forced=forced, undetermined=undecided)
    if family == "interval_selection":
        intervals = p["intervals"]
        def valid(items):
            ordered = sorted(items, key=lambda item: item["start"])
            return all(x["end"] <= y["start"] for x, y in zip(ordered, ordered[1:]))
        best = max(count for count in range(len(intervals) + 1) if any(valid(subset) for subset in itertools.combinations(intervals, count)))
        names = a.get("selected", [])
        selected = [item for item in intervals if item["id"] in names]
        return set(a) == {"maximum", "selected"} and len(set(names)) == len(names) == len(selected) == best == a["maximum"] and valid(selected)
    return False


def prompt(family, p):
    descriptions = {
        "invoice_tax": "Compute invoice subtotal in cents. Apply the percentage discount to subtotal, then percentage tax to the discounted subtotal, then add shipping. Report subtotal_cents, exact_cents as a reduced rational string, and rounded_cents rounded once at the end to nearest integer, half upwards.",
        "euclidean_division": "Find integer quotient and remainder satisfying a=quotient*b+remainder and 0<=remainder<b. Return quotient and remainder. Negative a uses this same rule.",
        "metric_area": "Find a rectangle's area from integer centimeter dimensions. Report square_centimeters as integer and square_meters as a reduced rational string. One meter equals 100 centimeters.",
        "percentage_change": "Compute the signed change new-old and percentage change relative to old. Report change as integer and percent as a reduced rational string. A decrease has a negative percent.",
        "weighted_mean": "The corresponding values and positive integer weights define a weighted mean. Report weighted_sum, total_weight, and mean as a reduced rational string.",
        "clock_duration": "Starting on day zero at start_hour:start_minute on a 24-hour clock, advance elapsed_minutes. Report day_offset and time formatted HH:MM. Every day has 1440 minutes.",
        "binary_decode": "Decode the binary bits string to a nonnegative decimal integer, allowing leading zeroes. Report decimal.",
        "affine_recurrence": "For a(0)=a0 and a(k+1)=multiplier*a(k)+increment, compute through a(n). Report values including a(0) and result=a(n).",
        "graph_reachability": "Edges [a,b] are directed from a to b. Report reachable as sorted nodes reachable from start, allowing a zero-edge path, so start itself is included. Only listed edges exist.",
        "dag_path_count": "Edges [a,b] are directed from a to b and increase node number. Count every distinct directed start-to-end path; if none exists return zero. Report paths.",
        "boolean_models": "Each clause is an OR of literals [index,positive]; positive=true means the indexed name, false means its negation. All clauses must hold. Report model_count, forced mapping names true in every satisfying assignment or false in every satisfying assignment, and undetermined sorted names taking both values. Do not assume unlisted facts.",
        "interval_selection": "Choose a maximum-size subset of intervals with no overlaps. An end equal to another start is allowed. Report maximum and selected interval IDs as one valid witness, with no repeated IDs.",
    }
    return f"2026 recovery practice. {descriptions[family]}\nData: {json.dumps(p, sort_keys=True)}\nGive a brief method, then the requested JSON object on the final line."


METHODS = {
    "invoice_tax": "Keep exact cents through discount and tax; round only the final amount.",
    "euclidean_division": "Check the quotient identity and the nonnegative remainder bound.",
    "metric_area": "Multiply centimeter dimensions; divide square centimeters by the squared length conversion.",
    "percentage_change": "Subtract the old value from the new value, then divide by the old value and multiply by one hundred.",
    "weighted_mean": "Sum each value times its weight, then divide by the sum of weights.",
    "clock_duration": "Add elapsed minutes and separate complete days, hours, and remaining minutes.",
    "binary_decode": "Read each binary digit by doubling the preceding total and adding the digit.",
    "affine_recurrence": "Apply the stated recurrence once per index; keep the complete value sequence.",
    "graph_reachability": "Follow directed edges from the start until no new node can be reached.",
    "dag_path_count": "Propagate the number of paths to each node along increasing directed edges.",
    "boolean_models": "Check every finite truth assignment against all clauses; distinguish forced values from undecided ones.",
    "interval_selection": "Choose compatible intervals by earliest finishing time; the listed IDs witness the maximum count.",
}


def build(cases_per_family=128, random_seed=20261013):
    if not 1 <= cases_per_family <= 1024:
        raise ValueError("cases_per_family_must_be_1_to_1024")
    rng, rows, receipts, seen = random.Random(random_seed), [], [], set()
    code_sha = file_sha256(Path(__file__))
    for family in FAMILIES:
        for index in range(cases_per_family):
            for _ in range(100):
                p = parameters(family, rng)
                identity = digest({"family": family, "parameters": p})
                if identity not in seen:
                    break
            else:
                raise ValueError("authored_parameter_space_exhausted")
            seen.add(identity)
            result = answer(family, p)
            if not verify_answer(family, p, result):
                raise ValueError("independent_oracle_disagreement")
            identifier = f"intelligence-train-{DATE}-{family}-{index:04}-{identity[:12]}"
            group = f"operator-intelligence-train-{DATE}-{family}"
            content = METHODS[family] + "\n" + json.dumps(result, sort_keys=True)
            receipt = {"schema": "recovery_intelligence_oracle.v1", "id": identifier, "group_id": group,
                       "family": family, "parameters": p, "answer": result, "checker_passed": True,
                       "assistant_content_sha256": digest(content), "source_code_sha256": code_sha}
            row = {"schema": "recovery_corpus.v1", "id": identifier,
                   "route": "text_reasoning" if family in REASONING else "text_arithmetic", "split": "train",
                   "messages": [{"role": "user", "content": prompt(family, p)}, {"role": "assistant", "content": content}],
                   "tools": [], "media": [], "supervision": {"assistant_only": True, "message_mask": [False, True]},
                   "provenance": {"source": "operator_authored_executable_oracle", "source_ids": [identifier],
                       "source_dates": [DATE], "original_authored_date": DATE, "source_verified": True,
                       "synthetic_operator_authored": True, "model_generated": False, "group_id": group,
                       "recovery_group_id": group, "split": "train", "template_family": family,
                       "source_code_sha256": code_sha, "oracle_receipt_sha256": digest(receipt)},
                   "quality": {"schema_complete": True, "train_ready": True, "semantic_verified": True,
                       "verification_scope": "authored finite specification and complete assistant target; no model performance", "blockers": []}}
            verify_row(row, receipt)
            rows.append(row)
            receipts.append(receipt)
    return rows, receipts


def verify_row(row, receipt):
    family, p = receipt["family"], receipt["parameters"]
    messages = row["messages"]
    if row["split"] != "train" or row["id"] != receipt["id"] or row["provenance"]["oracle_receipt_sha256"] != digest(receipt):
        raise ValueError("oracle_row_identity_or_split_disagreement")
    group = f"operator-intelligence-train-{DATE}-{family}"
    if (row["provenance"]["group_id"] != group or row["provenance"]["recovery_group_id"] != group or receipt["group_id"] != group
            or row["route"] != ("text_reasoning" if family in REASONING else "text_arithmetic")
            or row["provenance"]["split"] != "train" or row["tools"] or row["media"]):
        raise ValueError("oracle_row_family_or_route_disagreement")
    if row["supervision"] != {"assistant_only": True, "message_mask": [False, True]} or messages[0] != {"role": "user", "content": prompt(family, p)}:
        raise ValueError("input_or_supervision_not_verified")
    if messages[1]["role"] != "assistant" or digest(messages[1]["content"]) != receipt["assistant_content_sha256"]:
        raise ValueError("assistant_target_hash_disagreement")
    method, result = messages[1]["content"].split("\n")
    if method != METHODS[family] or not verify_answer(family, p, json.loads(result)) or json.loads(result) != receipt["answer"]:
        raise ValueError("assistant_target_semantic_failure")
    return True


def prepare(paths, output, cases_per_family=128, random_seed=20261013):
    if not paths or output.exists():
        raise ValueError("screen_sources_required_and_output_must_be_new")
    rows, receipts = build(cases_per_family, random_seed)
    tasks = [{"id": row["id"], "group_id": row["provenance"]["group_id"],
              "input": {"messages": [row["messages"][0]]}} for row in rows]
    blockers, pins, counts, source_groups = screen(tasks, paths)
    # A hit excludes all parameter variants in the authored family, not merely
    # the exact row: changing its numbers must never manufacture a holdout split.
    excluded_groups = {task["group_id"] for task in tasks if task["id"] in blockers}
    selected = [row for row in rows if row["provenance"]["group_id"] not in excluded_groups]
    ids = {row["id"] for row in selected}
    selected_receipts = [receipt for receipt in receipts if receipt["id"] in ids]
    if not selected:
        raise ValueError("all_authored_training_families_excluded")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".", dir=output.parent) as temp:
        root = Path(temp)
        ready, oracle = root / "ready.jsonl", root / "oracle_receipts.jsonl"
        ready.write_bytes(b"".join(canonical(row) + b"\n" for row in selected))
        oracle.write_bytes(b"".join(canonical(receipt) + b"\n" for receipt in selected_receipts))
        report = {"schema": "recovery_intelligence_training_seed.v1", "authored_date": DATE,
                  "created_at": datetime.now(timezone.utc).isoformat(), "synthetic_operator_authored": True,
                  "model_generated": False, "model_evaluated": False, "native_train_ready": False,
                  "source_code_sha256": file_sha256(Path(__file__)), "random_seed": random_seed,
                  "attempted": len(rows), "accepted": len(selected), "excluded": len(rows) - len(selected),
                  "route_counts": dict(Counter(row["route"] for row in selected)),
                  "family_counts": dict(Counter(row["provenance"]["template_family"] for row in selected)),
                  "split_counts": {"train": len(selected), "validation": 0, "test": 0},
                  "groups": sorted({row["provenance"]["group_id"] for row in selected}),
                  "excluded_groups": sorted(excluded_groups), "direct_blockers": blockers,
                  "source_groups_disjoint": not bool(source_groups & {row["provenance"]["group_id"] for row in selected}),
                  "screen_sources_sha256": pins, "screen_source_rows": counts,
                  "screen_policy": "exact normalized text; >50% unique candidate 13grams; source group equality; any hit excludes complete authored family",
                  "verification": "separate arithmetic identities/Decimal/closed form, Floyd closure, subset path enumeration, bit-mask truth evaluation, exhaustive interval subsets",
                  "limits": ["training supplement only; unchanged frozen evaluation", "finite authored tasks do not establish general intelligence", "lexical and explicit source-group screening does not prove semantic novelty", "requires actual parent-tokenizer/template packing and masks before native training", "no harvested self-reported correctness or teacher/model outputs used"],
                  "outputs": {"ready": {"path": str((output / ready.name).resolve()), "sha256": file_sha256(ready), "rows": len(selected)},
                              "oracle_receipts": {"path": str((output / oracle.name).resolve()), "sha256": file_sha256(oracle), "rows": len(selected_receipts)}}}
        report["seal_sha256"] = digest(report)
        (root / "report.json").write_bytes(canonical(report) + b"\n")
        root.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases-per-family", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=20261013)
    args = parser.parse_args()
    result = prepare(args.screen, args.out, args.cases_per_family, args.random_seed)
    print(json.dumps({key: result[key] for key in ("accepted", "excluded", "route_counts", "family_counts", "seal_sha256", "outputs")}, sort_keys=True))


if __name__ == "__main__":
    main()
