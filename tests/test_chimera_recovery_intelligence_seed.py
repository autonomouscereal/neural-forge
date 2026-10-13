"""Independent arithmetic/logic boundaries and contamination/forgery regressions."""
import copy
import json

import pytest

from experiments.chimera_recovery import intelligence_seed as seed
from experiments.chimera_recovery.corpus import validate_record


@pytest.mark.parametrize("family,p,expected", [
    ("invoice_tax", dict(cents=25, quantity=2, discount=50, tax=2, shipping=0), dict(subtotal_cents=50, exact_cents="51/2", rounded_cents=26)),
    ("euclidean_division", dict(a=-13, b=5), dict(quotient=-3, remainder=2)),
    ("metric_area", dict(width_cm=25, height_cm=80), dict(square_centimeters=2000, square_meters="1/5")),
    ("percentage_change", dict(old=80, new=50), dict(change=-30, percent="-75/2")),
    ("weighted_mean", dict(values=[-5, 2], weights=[2, 3]), dict(weighted_sum=-4, total_weight=5, mean="-4/5")),
    ("clock_duration", dict(start_hour=23, start_minute=59, elapsed_minutes=2882), dict(day_offset=3, time="00:01")),
    ("binary_decode", dict(bits="00101001"), dict(decimal=41)),
    ("affine_recurrence", dict(a0=2, multiplier=-2, increment=1, n=4), dict(values=[2, -3, 7, -13, 27], result=27)),
    ("graph_reachability", dict(nodes=[0, 1, 2, 3], edges=[[0, 1], [1, 0], [1, 2]], start=0), dict(reachable=[0, 1, 2])),
    ("dag_path_count", dict(nodes=[0, 1, 2, 3], edges=[[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], start=0, end=3), dict(paths=3)),
    ("boolean_models", dict(names=["A", "B", "C"], clauses=[[[0, True]], [[0, False], [1, True]]]), dict(model_count=2, forced={"A": True, "B": True}, undetermined=["C"])),
    ("interval_selection", dict(intervals=[dict(id="A", start=0, end=10), dict(id="B", start=0, end=2), dict(id="C", start=2, end=4), dict(id="D", start=4, end=6)]), dict(maximum=3, selected=["B", "C", "D"])),
])
def test_independent_hand_calculated_boundary_fixtures_and_wrong_answers(family, p, expected):
    assert seed.answer(family, p) == expected
    assert seed.verify_answer(family, p, expected)
    wrong = copy.deepcopy(expected)
    key = next(iter(wrong))
    wrong[key] = [] if isinstance(wrong[key], list) else 999
    assert not seed.verify_answer(family, p, wrong)


def test_witness_validity_and_strict_integer_types():
    p = dict(intervals=[dict(id="A", start=0, end=2), dict(id="B", start=2, end=4), dict(id="C", start=0, end=4)])
    assert seed.verify_answer("interval_selection", p, dict(maximum=2, selected=["B", "A"]))
    assert not seed.verify_answer("interval_selection", p, dict(maximum=2, selected=["A", "A"]))
    assert not seed.verify_answer("interval_selection", p, dict(maximum=2, selected=["A", "C"]))
    assert not seed.verify_answer("euclidean_division", dict(a=1, b=2), dict(quotient=False, remainder=True))
    assert not seed.verify_answer("percentage_change", dict(old=10, new=11), dict(change=1, percent="not a number"))
    assert seed.verify_answer("dag_path_count", dict(nodes=[0, 1, 2], edges=[], start=0, end=2), dict(paths=0))


def test_all_authored_rows_checked_without_teacher_or_target_in_input():
    rows, receipts = seed.build(8)
    assert len(rows) == 96 and len(receipts) == 96
    assert {row["split"] for row in rows} == {"train"}
    assert len({row["provenance"]["group_id"] for row in rows}) == 12
    for row, receipt in zip(rows, receipts):
        assert seed.verify_row(row, receipt)
        assert validate_record(row)["train_ready"]
        assert row["provenance"]["model_generated"] is False
        assert row["supervision"]["message_mask"] == [False, True]
        assert "answer" not in row["messages"][0] and "oracle" not in row["messages"][0]
    row, receipt = copy.deepcopy(rows[0]), copy.deepcopy(receipts[0])
    row["messages"][0]["content"] += " The expected answer is 999."
    with pytest.raises(ValueError, match="input_or_supervision"):
        seed.verify_row(row, receipt)


def test_resealed_wrong_target_still_fails_independent_semantics():
    rows, receipts = seed.build(1)
    row, receipt = rows[1], receipts[1]  # Euclidean division.
    receipt["answer"]["quotient"] += 1
    row["messages"][1]["content"] = seed.METHODS[receipt["family"]] + "\n" + json.dumps(receipt["answer"], sort_keys=True)
    receipt["assistant_content_sha256"] = seed.digest(row["messages"][1]["content"])
    row["provenance"]["oracle_receipt_sha256"] = seed.digest(receipt)
    with pytest.raises(ValueError, match="semantic_failure"):
        seed.verify_row(row, receipt)


def test_source_hit_excludes_entire_family_and_retains_frozen_source(tmp_path):
    rows, _ = seed.build(2)
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text(json.dumps({"messages": [rows[0]["messages"][0]]}) + "\n", encoding="utf-8")
    before = frozen.read_bytes()
    report = seed.prepare([frozen], tmp_path / "training", 2)
    assert report["accepted"] == 22 and report["excluded"] == 2
    assert "invoice_tax" not in report["family_counts"]
    assert report["split_counts"] == {"train": 22, "validation": 0, "test": 0}
    assert report["native_train_ready"] is False
    assert frozen.read_bytes() == before
    with pytest.raises(ValueError, match="output_must_be_new"):
        seed.prepare([frozen], tmp_path / "training", 2)
