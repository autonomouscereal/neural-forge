"""Independent 2026 groundtruth and adversarial contamination checks."""
import collections
import json
from pathlib import Path

from experiments.chimera_recovery import evaluation as e
from experiments.chimera_recovery import text_eval_seed as seed


REFERENCE = {
    "stable_unique": "def stable_unique(xs):\n    return list(dict.fromkeys(xs))",
    "even_count": "def even_count(xs):\n    return sum([1 for x in xs if x % 2 == 0])",
    "prefix_totals": "def prefix_totals(xs):\n    out = []\n    total = 0\n    for x in xs:\n        total += x\n        out.append(total)\n    return out",
    "rotate_left": "def rotate_left(xs, k):\n    if not xs:\n        return []\n    k = k % len(xs)\n    return xs[k:] + xs[:k]",
    "clamp_values": "def clamp_values(xs, low, high):\n    return [min(high, max(low, x)) for x in xs]",
    "count_runs": "def count_runs(xs):\n    n = 0\n    for i in range(len(xs)):\n        if i == 0 or xs[i] != xs[i-1]:\n            n += 1\n    return n",
    "nondecreasing": "def nondecreasing(xs):\n    for i in range(1, len(xs)):\n        if xs[i] < xs[i-1]:\n            return False\n    return True",
    "adjacent_differences": "def adjacent_differences(xs):\n    return [xs[i] - xs[i-1] for i in range(1, len(xs))]",
    "positive_count": "def positive_count(xs):\n    return sum([1 for x in xs if x > 0])",
    "median_three": "def median_three(xs):\n    return sorted(xs)[1]",
    "argmin_first": "def argmin_first(xs):\n    if not xs:\n        return -1\n    best = 0\n    for i in range(1, len(xs)):\n        if xs[i] < xs[best]:\n            best = i\n    return best",
    "intersection_order": "def intersection_order(xs, ys):\n    out = []\n    for x in xs:\n        if x in ys and x not in out:\n            out.append(x)\n    return out",
}


def response(text):
    return {"messages": [{"role": "assistant", "content": text}]}


def test_every_code_oracle_with_independent_pure_reference_program_and_wrong_program():
    tasks = seed.build_tasks()
    assert collections.Counter(t["route"] for t in tasks) == {"text_arithmetic": 24, "text_code": 12, "text_instruction": 8}
    assert len({t["id"] for t in tasks}) == 44
    for t in tasks:
        if t["route"] != "text_code":
            continue
        function = t["oracle"]["function"]
        assert e.grade_semantics(t, response(REFERENCE[function]))["status"] == "pass"
        assert e.grade_semantics(t, response(f"def {function}(*args):\n    return 999"))["status"] in {"fail", "ungraded"}
    # The alternate imperative implementation is a valid answer too.
    t = next(t for t in tasks if t["oracle"].get("function") == "stable_unique")
    alternate = "def stable_unique(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out"
    assert e.grade_semantics(t, response(alternate))["status"] == "pass"
    assert e.grade_semantics(t, response("def stable_unique(xs):\n    return sorted(set(xs))"))["status"] == "fail"


def test_independent_arithmetic_boundary_values_and_instruction_constraints():
    tasks = seed.build_tasks()
    expected_by_family = {
        "signed_product": ["-102", "322", "232", "703"],
        "precedence": ["61", "59", "-19", "-45"],
        "fraction_sum": ["1/2", "29/36", "11/24", "23/45"],
        "budget": ["2545", "2360", "1020", "3305"],
        "integer_interval": ["10", "9", "6", "13"],
        "linear_equation": ["9", "-6", "7/2", "5/3"],
    }
    for family, expected in expected_by_family.items():
        found = [t for t in tasks if t["group_id"].endswith("-" + family)]
        assert [t["oracle"]["expected"] for t in found] == expected
        for t, value in zip(found, expected):
            assert e.grade_semantics(t, response("Calculation explained.\nAnswer: " + value))["status"] == "pass"
            assert e.grade_semantics(t, response("Answer: 999"))["status"] == "fail"
    for t in tasks:
        if t["route"] != "text_instruction":
            continue
        correct = json.dumps(t["oracle"]["expected"], sort_keys=False, indent=2)
        assert e.grade_semantics(t, response(correct))["status"] == "pass"
        assert e.grade_semantics(t, response("Here is the answer: " + correct))["status"] == "fail"
        assert e.grade_semantics(t, response(json.dumps({**t["oracle"]["expected"], "extra": 1})))["status"] == "fail"
    t = next(t for t in tasks if t["group_id"].endswith("record_selection"))
    assert e.grade_semantics(t, response('{"id":"a","score":9,"score":9}'))["status"] == "fail"
    assert e.grade_semantics(t, response('{"id":"a","score":true}'))["status"] == "fail"


def test_full_message_and_source_group_contamination_excluded_and_oracles_private(tmp_path):
    tasks = seed.build_tasks()
    first, second = tasks[0], tasks[-1]
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"id": "long-source", "messages": [{"role": "system", "content": "Long unrelated context"},
        {"role": "user", "content": first["input"]["messages"][0]["content"]}, {"role": "assistant", "content": "source answer"}]}) + "\n" +
        json.dumps({"id": "group-overlap", "messages": [{"role": "user", "content": "different text"}], "provenance": {"group_id": second["group_id"]}}) + "\n", encoding="utf-8")
    report = seed.prepare([source], tmp_path / "frozen")
    assert report["accepted"] == 41  # both examples share the last exclusive family
    assert report["source_groups_disjoint"]
    inputs = [json.loads(line) for line in (tmp_path / "frozen" / "inputs.jsonl").read_text().splitlines()]
    assert all("oracle" not in row and "expected" not in row and "provenance" not in row for row in inputs)
    assert first["id"] not in {r["id"] for r in inputs}
    assert all(r["group_id"] != second["group_id"] for r in inputs)


def test_near_duplicate_candidate_is_excluded_inside_long_source_answer(tmp_path):
    t = next(t for t in seed.build_tasks() if t["route"] == "text_code")
    prompt = t["input"]["messages"][0]["content"]
    # A few added surrounding words do not defeat a near-duplicate screen.
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"messages": [{"role": "assistant", "content": "prefix words " + prompt + " suffix words"}]}) + "\n", encoding="utf-8")
    blockers, _, counts, _ = seed.screen([t], [source])
    assert counts[str(source.resolve())] == 1
    assert any(x["reason"] == "candidate_13gram_overlap_gt_half" for x in blockers[t["id"]])
