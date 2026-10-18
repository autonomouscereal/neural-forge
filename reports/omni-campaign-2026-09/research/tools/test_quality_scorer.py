"""Tests for quality_scorer.py (research lane, 2026-10-15).  CPU only, fixtures only: no corpus, no model.

    python -m pytest -q test_quality_scorer.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quality_scorer as qs  # noqa: E402


def row(rid, instruction, response, route="text", ds="ds_plain", family="text", **extra):
    out = {"rid": rid, "instruction": instruction, "response": response, "route": route, "dataset_id": ds, "family": family}
    out.update(extra)
    return out


def ctx_for(rows, **kw):
    return qs.build_context(rows, split=kw.get("split"), d2_kept=kw.get("d2_kept"), screen=kw.get("screen"),
                            holdout_rids=kw.get("holdout", ()))


# ------------------------------------------------------------------------------------------- answers
def test_reference_boxed_nested_and_after_think():
    text = "<think>try \\boxed{1}</think>\nSo the result is \\boxed{\\frac{1}{2}}."
    assert qs.extract_reference(text) == ("\\frac{1}{2}", "boxed")


def test_reference_answer_line_variants_and_junk():
    assert qs.extract_reference("work\nTherefore, the answer is: 180.")[0] == "180"
    assert qs.extract_reference("x\nFinal answer: \\sinA=\\frac{1}{2}")[0] == "\\sinA=\\frac{1}{2}"
    # audit4's FINAL regex took "." from prose like this; the strict rule takes nothing
    assert qs.extract_reference("define quantities and state the final result with units.")[0] is None
    # an over-long answer line is not a clean answer
    assert qs.extract_reference("Answer: If the original speed was maintained, it would take 180 days and then some more.")[0] is None


def test_final_answer_unclosed_think_and_last_line_numeric():
    assert qs.extract_final_answer("<think>\nlong thought without end") == (None, "unclosed_think")
    assert qs.extract_final_answer("<think>a</think>\nThe total is 42.") == ("42", "last_line_numeric")
    assert qs.extract_final_answer("<think>a</think>\n\\boxed{7}")[0] == "7"


@pytest.mark.parametrize("a,b,want", [
    ("\\frac{1}{2}", "0.5", True), ("\\dfrac{3}{4}", "3/4", True), ("x = 3", "3", True), ("12\\%", "0.12", True),
    ("\\textbf{(C)}\\ 8", "8", True), ("(B)", "B", True), ("1,000", "1000", True), ("2", "3", False),
    ("-\\ln 2", "-\\ln 2", True), ("-\\ln 2", "\\ln(1/2)", None), ("A", "C", False), ("7", "seven", False),
    # 1.0.2: reference forms read from the pilot's references.jsonl
    ("6\\mathrm{~}", "6", True), ("`2039190`", "2039190", True), ("24 percent", "24\\%", True), ("24 percent", "24", True),
    ("\\textbf{(C) }\\frac{7}{27}", "\\frac{7}{27}", True), ("\\textbf{(C) }\\frac{7}{27}", "C", True),
    ("\\textbf{(C) }\\frac{7}{27}", "B", False), ("180 days", "180", True), ("2x", "2", False),
])
def test_answers_equal(a, b, want):
    assert qs.answers_equal(a, b) is want


def test_answers_equal_none_inputs():
    assert qs.answers_equal(None, "3") is None


def test_arithmetic_claims_exact_chain_and_rounding():
    checked, wrong, ex = qs.arithmetic_claims("We get 3 + 4 = 7 and 6 * 7 = 43. Also 1/3 = 0.33 and 2 + 3 * 4 = 14. Area 18*11/2 = 99.")
    assert checked == 3 and wrong == 1 and "6 * 7 = 43" in ex[0]


def test_arithmetic_thousands_separators_and_percentages():
    # the first registry pass flagged these as wrong (1.0.0 false positives read from the run)
    text = "Cost 33,000/3530 = 9.35 per unit; 420 * 3 = 1,260 total; share 276 / 430 = 64.2% and 1 / 4 = 25 percent."
    checked, wrong, ex = qs.arithmetic_claims(text)
    assert wrong == 0 and checked == 2


def test_arithmetic_letter_x_chains_are_skipped():
    # every one of 20 sampled 1.0.1 "wrong" claims was one of these (templated curie_* rows): all correct chains
    text = ("Total runs are 13 x 3 + 3 = 42. Check: 14 x 7 + 4 = 102. V1 = 0.20 x 255 / 1.75 = 29.1 mL. "
            "C(8, 2) = 8 x 7 / 2 = 28. But 6 * 7 = 43 is wrong.")
    checked, wrong, ex = qs.arithmetic_claims(text)
    assert (checked, wrong) == (1, 1) and "6 * 7 = 43" in ex[0]


def test_arithmetic_latex_chains_refutations_truncation_rounding():
    # read from the 1.0.3 sample of "wrong" claims
    text = ("total 8(n-2)+4=8 \\cdot 11+4=92 cubes; 998 \\times 1002 + 4 = 1000000 people; T = 185 \\times 36/37 = 180. "
            "Mod 7: 1+0=0 impossible unless b. 2052/22 = 93.272..., not an integer. The ratio is 1608/384.2 = 4.18, so. "
            "Yet 7 + 5 = 13 here.")
    checked, wrong, ex = qs.arithmetic_claims(text)
    assert (checked, wrong) == (2, 1) and "7 + 5 = 13" in ex[0]


def test_arithmetic_zero_divisor_on_other_operators():
    # 1.0.0 built the operator table eagerly: "5 + 0 = 5" divided by zero (crashed the first full registry run)
    assert qs.arithmetic_claims("so 5 + 0 = 5 and 3 - 0 = 3 and 4 / 0 = 9") == (2, 0, [])


def test_fraction_index_ignores_boilerplate_and_flags_near_dups():
    idx = qs.FractionIndex(threshold=0.5, boiler_df=3)
    boiler = "you are a careful assistant who always answers the question asked by the user with care"
    for i in range(5):
        idx.add_text("split", f"s{i}", boiler + " " + long_text(100 + i, 60))
    held = long_text(7, 200)
    idx.add_text("split", "held", held)
    assert idx.hits(boiler + " " + long_text(55, 60)) == []                 # shares only boilerplate
    near = held.split()
    near[100] = "zz"
    hit = idx.hits(" ".join(near))
    assert hit and hit[0]["item"] == "held" and hit[0]["rule"].startswith("near13_frac")
    assert idx.hits(" ".join(held.split()[:20]) + " " + long_text(8, 200)) == []   # a small shared fragment is not a near-dup


def test_screen_combines_bank_and_split_hits():
    s = qs.Screen(FakeScreen(), qs.FractionIndex())
    assert s.hits("SECRET BANK PROMPT")[0]["source"] == "bank_v1"
    assert s.hits("nothing here") == []


def test_problem_in_instruction():
    assert qs.problem_in_instruction("Let x be a real number such that x^2 = 4 and x > 0. Find x.")
    assert not qs.problem_in_instruction("Solve the source-grounded math problem. Show the important steps, state assumptions.")
    assert not qs.problem_in_instruction(qs.TEACH + " ... user: Create a compact math target")


# ------------------------------------------------------------------------------------------- tool calls
def test_tool_calls_qwen_xml_hermes_prose_and_malformed():
    xml = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    t = qs.parse_tool_calls(xml)
    assert t["calls"] == 1 and t["formats"] == ["qwen_xml"] and t["parse_errors"] == 0
    hermes = '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'
    assert qs.parse_tool_calls(hermes)["formats"] == ["hermes_json"]
    prose = 'Sure. {"tool_calls": [{"function": {"name": "f", "arguments": "{\\"a\\": 1}"}}]}'
    t = qs.parse_tool_calls(prose)
    assert t["calls"] == 1 and t["args_not_object"] == 0
    bad = "<tool_call>{not json</tool_call><tool_call>"
    assert qs.parse_tool_calls(bad)["parse_errors"] == 2
    assert qs.parse_tool_calls("no calls here") is None


def test_offered_functions_and_schema_check():
    ins = 'Tools: [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {}}, "required": ["a"]}}}]'
    offered = qs.offered_functions(ins)
    assert set(offered) == {"f"}
    got = qs.check_offered([{"name": "f", "arguments": {}}, {"name": "g", "arguments": {"a": 1}}, {"name": "f", "arguments": {"a": 1, "b": 2}}], offered)
    assert got == {"name_not_offered": 1, "missing_required": 1, "unknown_param": 1}


# ------------------------------------------------------------------------------------------- per-row
def test_score_row_classes_and_flags():
    web_cut = row("r1", "Model this high-quality text sample.", "Some web text that stops in the midd" * 10)
    teacher = row("r2", qs.TEACH + " ...", '{"corrected_response": "x"}', route="math")
    empty = row("r3", "Question?", "")
    hexed = row("r4", "Do it", "0123456789abcdef0123456789abcdef user: hello there")
    unclosed = row("r5", "Think about this problem carefully please and answer.", "<think>\n" + "hmm " * 100, route="reasoning")
    rows = [web_cut, teacher, empty, hexed, unclosed]
    c = ctx_for(rows)
    recs = {r["rid"]: qs.finish(qs.score_row(r, c)) for r in rows}
    assert recs["r1"]["class"] == "08" and "ends_mid_word" in recs["r1"]["flags"]
    assert recs["r2"]["class"] == "02" and recs["r2"]["action"] == "exclude_stage_t_gt_mask_d2"
    assert recs["r3"]["action"] == "reject" and recs["r3"]["score"] == 0.0
    assert recs["r4"]["class"] == "03" and recs["r4"]["action"] == "repair_deterministic"
    assert recs["r5"]["class"] == "06" and recs["r5"]["trunc"]["unclosed_think"]


def test_cap_signature_matches_audit2_rule():
    rows = [row(f"c{i}", "q", "x" * (1000 if i < 25 else 300), ds="capped") for i in range(100)]
    rows += [row(f"n{i}", "q", "y" * (100 + i), ds="free") for i in range(100)]
    caps = qs.cap_signatures(rows)
    assert caps == {"capped": 1000}
    rec = qs.score_row(rows[0], ctx_for(rows))
    assert rec["trunc"]["cap_cut"] and "cap_cut" in rec["flags"]


def test_degenerate_excludes_arc_and_json():
    loop = "the same line again\n" * 60
    assert "degenerate_repetition" in qs.score_row(row("d", "q", loop), ctx_for([row("d", "q", loop)]))["flags"]
    arc = row("a", "q", loop, ds="arc_agi_x")
    assert "degenerate_repetition" not in qs.score_row(arc, ctx_for([arc]))["flags"]


def test_math_verify_kinds():
    good = row("m1", "Find the least positive three-digit multiple of 7, and explain.", "...so \\boxed{105}.", route="math")
    generic = row("m2", "Solve the source-grounded math problem. Show the important steps.", "Problem... \\boxed{8}", route="math")
    noref = row("m3", "Find all integers n such that n^2 + 1 divides n^3 + 7.", "We argue... n = 2.", route="math")
    rows = [good, generic, noref]
    c = ctx_for(rows)
    kinds = {r["rid"]: qs.score_row(r, c)["verify"] for r in rows}
    assert kinds["m1"]["kind"] == "math_ref" and kinds["m1"]["reference"] == "105"
    assert kinds["m2"]["kind"] == "math_ref_no_problem"
    assert kinds["m3"]["kind"] == "math_noref"


class FakeScreen:
    def hits(self, text):
        return [{"rule": "ngram13", "source": "bank_v1", "item": "x"}] if "SECRET BANK PROMPT" in text else []


def test_screen_hit_is_hard_reject_and_split_flag():
    rows = [row("s1", "SECRET BANK PROMPT", "answer"), row("s2", "fine", "answer text.")]
    c = ctx_for(rows, screen=FakeScreen(), split={"s2": "eval"})
    recs = [qs.finish(qs.score_row(r, c)) for r in rows]
    assert recs[0]["action"] == "reject" and recs[0]["screen"][0]["source"] == "bank_v1"
    assert "split_eval" in recs[1]["flags"]


# ------------------------------------------------------------------------------------------- dedup
def long_text(seed, n=300):
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(2000)]
    return " ".join(rng.choice(vocab) for _ in range(n))


def test_dedup_exact_near_and_distinct():
    base = long_text(1)
    near = base.split()
    near[150] = "z"          # shorter than the word it replaces: "a" (longest, train, same flags) is the representative
    rows = [row("a", "q", base, split="train"), row("b", "q", base), row("c", "q", " ".join(near)), row("d", "q", long_text(2))]
    c = ctx_for(rows, split={"a": "train", "b": "train", "c": "train", "d": "train"})
    recs, clusters, stats, sigs = qs.score_all(rows, c, progress_every=0)
    by = {r["rid"]: r for r in recs}
    assert "exact_duplicate_member" in by["b"]["flags"] and by["b"]["dedup"]["exact_of"] == "a"
    assert by["a"]["dedup"].get("representative") is True
    assert "near_duplicate_member" in by["c"]["flags"]
    assert "near_cluster" not in by["d"]["dedup"]
    assert stats["near_clusters"] == 1 and len(clusters) == 1 and clusters[0]["representative"] == "a"
    assert sigs.shape == (4, qs.N_PERM)


def test_dedup_against_ledger():
    base = long_text(3)
    led_rows = [row("L1", "q", base)]
    led, _, _, led_sigs = qs.score_all(led_rows, ctx_for(led_rows), progress_every=0)
    near = base.split()
    near[10] = "x"
    batch = [row("b1", "q", base), row("b2", "q", " ".join(near)), row("b3", "q", long_text(4))]
    ledger = {"sigs": led_sigs, "exact16": {led[0]["dedup"]["exact_sha16"]}}
    recs, clusters, stats, _ = qs.score_all(batch, ctx_for(batch), ledger=ledger, progress_every=0)
    by = {r["rid"]: r for r in recs}
    assert by["b1"]["dedup"]["exact_of"] == "ledger" and by["b1"]["action"] == "reject"
    assert "near_duplicate_member" in by["b2"]["flags"]
    assert by["b3"]["action"] == "keep" and stats["ledger_exact_hits"] == 1


# ------------------------------------------------------------------------------------------- commands
def write_batch(tmp_path, rows, *, bad_sha=False):
    bdir = tmp_path / "batch"
    bdir.mkdir()
    with open(bdir / "rows.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    sha = qs.file_sha256(bdir / "rows.jsonl")
    (bdir / "manifest.json").write_text(json.dumps({"schema": qs.INTAKE_SCHEMA, "batch_id": "t-0001", "rows": len(rows),
                                                    "rows_sha256": "0" * 64 if bad_sha else sha}), encoding="utf-8")
    return bdir


def test_cmd_batch_end_to_end_and_messages_rows(tmp_path):
    rows = [row("x1", "What is 2+2?", "2 + 2 = 4. The answer is 4."),
            {"rid": "x2", "dataset_id": "d", "route": "text",
             "messages": [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "Hello!"}]}]
    bdir = write_batch(tmp_path, rows)
    assert qs.main(["batch", "--batch-dir", str(bdir), "--no-screen"]) == 0
    out = bdir / "quality" / qs.TOOL_VERSION
    recs = [json.loads(l) for l in open(out / "scores.jsonl", encoding="utf-8")]
    assert [r["rid"] for r in recs] == ["x1", "x2"] and recs[1]["len"]["resp_chars"] == len("Hello!")
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["outputs"]["scores.jsonl"]["sha256"] == qs.file_sha256(out / "scores.jsonl")
    assert recs[0]["verify"].get("arith_checked") == 1


def test_cmd_batch_refuses_sha_mismatch(tmp_path):
    bdir = write_batch(tmp_path, [row("x1", "q", "a")], bad_sha=True)
    assert qs.main(["batch", "--batch-dir", str(bdir), "--no-screen"]) == 2


def test_cmd_registry_small_fixture(tmp_path):
    rows = [row("000:0:1", "Model this high-quality text sample.", "cut mid sente" * 30),
            row("000:0:2", "Find the least positive three-digit multiple of 7, explain.", "\\boxed{105}", route="math"),
            row("000:0:3", "Find the least positive three-digit multiple of 7, explain.", "\\boxed{105}", route="math")]
    rp = tmp_path / "rows.jsonl"
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cp = tmp_path / "carve.json"
    cp.write_text(json.dumps({"assignment": {"000:0:1": "train", "000:0:2": "train", "000:0:3": "eval"}}), encoding="utf-8")
    dp = tmp_path / "documents.jsonl"
    dp.write_text(json.dumps({"rid": "000:0:1", "kept": 100, "start": 0, "end": 100, "length": 100, "doc": 0}) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    assert qs.main(["registry", "--rows", str(rp), "--carve", str(cp), "--documents", str(dp), "--no-screen", "--no-eos-check",
                    "--audit4", str(tmp_path / "none.json"), "--bfcl-holdout", str(tmp_path / "none.json"), "--out-dir", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 3 and summary["rows_in_d2"] == 1 and summary["d2_tokens"] == 100
    recs = {r["rid"]: r for r in map(json.loads, open(out / "scores.jsonl", encoding="utf-8"))}
    assert recs["000:0:3"]["action"] == "reject"            # exact duplicate of a train row
    assert recs["000:0:2"]["verify"]["kind"] == "math_ref"
    assert summary["by_split"]["eval"]["rows"] == 1


def test_d2_eos_check_reads_packed_tail(tmp_path):
    np = pytest.importorskip("numpy")
    st = pytest.importorskip("safetensors.numpy")
    ids = np.zeros((2, 8), dtype=np.int32)
    ids[0, 2:5] = [qs.IM_END_ID, 198, qs.SEPARATOR_ID]   # doc 0 = positions 0..4
    shard = tmp_path / "text-00000.safetensors"
    st.save_file({"input_ids": ids}, str(shard))
    (tmp_path / "manifest.json").write_text(json.dumps({"sequence_length": 8, "shards": [{"path": str(shard), "rows": [0, 2]}]}), encoding="utf-8")
    (tmp_path / "docs.jsonl").write_text(json.dumps({"rid": "r", "doc": 0, "start": 0, "end": 5, "kept": 5, "length": 5}) + "\n", encoding="utf-8")
    got = qs.d2_eos_check(tmp_path / "docs.jsonl", tmp_path / "manifest.json", {"r": {"class": "08"}}, tmp_path)
    assert got["tails"] == {"im_end+nl+sep": 1} and got["samples"][0]["last_ids"][-1] == qs.SEPARATOR_ID
