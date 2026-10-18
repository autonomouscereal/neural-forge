"""Tests for vmath_pilot.py (research lane, 2026-10-15).  Fixtures only; the prompt file is checked by the real consumer,
selfgen_sampler.load_prompts, and the claim rule is the real d1_seal_lib.claim_state.

    python -m pytest -q test_vmath_pilot.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vmath_pilot as vp  # noqa: E402


def distill_tools() -> Path:
    for cand in (Path(os.environ.get("OC", "/home/cereal/omni-campaign")) / "distill" / "tools",
                 HERE.parent.parent / "distill" / "tools"):
        if (cand / "d1_seal_lib.py").is_file() and (cand / "selfgen_sampler.py").is_file():
            return cand
    pytest.skip("distill/tools (d1_seal_lib.py, selfgen_sampler.py) not found")


def claim_state():
    return vp.load_by_path("d1_seal_lib", distill_tools() / "d1_seal_lib.py").claim_state


class FakeRenderer:
    def render(self, user):
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def count(self, text):
        return len(text.split())

    def report(self):
        return {"fake": True}


def rec(rid, kind, split="train", route="math", flags=(), problem=True, ref=None):
    ver = {"kind": kind, "problem_in_instruction": problem}
    if ref is not None:
        ver["reference"] = ref
    return {"rid": rid, "split": split, "route": route, "flags": list(flags), "verify": ver}


def write_fixture(tmp_path):
    scores = [rec("r1", "math_ref", ref="105"), rec("r2", "math_ref", ref="7"), rec("a1", "math_noref"), rec("a2", "math_noref"),
              rec("e1", "math_ref", split="eval"), rec("d1", "math_ref", flags=["near_duplicate_member"]),
              rec("n1", "math_ref_no_problem", problem=False), rec("x1", "math_noref", route="reasoning"),
              rec("long", "math_noref")]
    rows = [{"rid": "r1", "instruction": "Find the least positive three-digit multiple of 7.", "response": "So \\boxed{105}.", "dataset_id": "d"},
            {"rid": "r2", "instruction": "Compute 3 + 4 and justify every step briefly.", "response": "Answer: 7", "dataset_id": "d"},
            {"rid": "a1", "instruction": "Find all integers n such that n^2 + 1 divides n^3 + 7.", "response": "...", "dataset_id": "d"},
            {"rid": "a2", "instruction": "How many primes are below one hundred? Explain your count.", "response": "...", "dataset_id": "d"},
            {"rid": "long", "instruction": "word " * 500, "response": "...", "dataset_id": "d"}]
    sp, rp = tmp_path / "scores.jsonl", tmp_path / "rows.jsonl"
    sp.write_text("".join(json.dumps(r) + "\n" for r in scores), encoding="utf-8")
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return sp, rp


def test_select_sources_strata_and_exclusions(tmp_path):
    sp, _ = write_fixture(tmp_path)
    got = vp.select_sources(vp.qs().read_jsonl(sp))
    assert sorted(r["rid"] for r in got["R"]) == ["r1", "r2"]
    assert sorted(r["rid"] for r in got["A"]) == ["a1", "a2", "long"]


def test_build_prompts_order_k_adjacent_and_limits(tmp_path):
    sp, rp = write_fixture(tmp_path)
    built = vp.build_prompts(sp, rp, FakeRenderer(), k=2, max_prompts=10, seed=1, max_prompt_tokens=200, max_new_tokens=6144)
    strata = [p["stratum"] for p in built["prompts"]]
    assert strata[:4] == ["R"] * 4 and set(strata[4:]) == {"A"}
    rids = [p["rid"] for p in built["prompts"]]
    assert rids[0].endswith("~k0") and rids[1].endswith("~k1") and rids[0][:-3] == rids[1][:-3]
    assert built["dropped"] == {"prompt_too_long": 1}
    assert all(vp.BOXED_SUFFIX in p["messages"][0]["content"] for p in built["prompts"])
    refs = {r["source_rid"]: r["reference"] for r in built["references"]}
    assert refs["r1"] == "105" and refs["r2"] == "7" and refs["a1"] is None
    limited = vp.build_prompts(sp, rp, FakeRenderer(), k=2, max_prompts=3, seed=1, max_prompt_tokens=200, max_new_tokens=10)
    assert len(limited["references"]) == 3 and len(limited["prompts"]) == 6


class HitScreen:
    def hits(self, text):
        return [{"source": "bank_v1"}] if "primes" in text else []


def test_build_prompts_screen_drop(tmp_path):
    sp, rp = write_fixture(tmp_path)
    built = vp.build_prompts(sp, rp, FakeRenderer(), k=2, max_prompts=10, seed=1, max_prompt_tokens=200, max_new_tokens=10, screen=HitScreen())
    assert built["dropped"]["screen_hit_prompt"] == 1 and "a2" not in {r["source_rid"] for r in built["references"]}


def test_cmd_prompts_output_loads_in_selfgen_sampler(tmp_path, monkeypatch):
    sp, rp = write_fixture(tmp_path)
    monkeypatch.setattr(vp, "Renderer", lambda *a, **k: FakeRenderer())
    out = tmp_path / "prompts"
    assert vp.main(["prompts", "--scores", str(sp), "--rows", str(rp), "--out-dir", str(out), "--no-screen", "--max-prompt-tokens", "200"]) == 0
    sampler = vp.load_by_path("selfgen_sampler", distill_tools() / "selfgen_sampler.py")
    loaded = sampler.load_prompts(out / "prompts.jsonl", out / "manifest.json")
    assert len(loaded["prompts"]) == 8 and loaded["manifest"]["prompts"] == 8
    assert vp.main(["prompts", "--scores", str(sp), "--rows", str(rp), "--out-dir", str(out), "--no-screen"]) == 2   # never overwritten


# ------------------------------------------------------------------------------------------- gate / yield
FREE = {1: {"used_mib": 5, "apps": 0}, 2: {"used_mib": 5, "apps": 0}, 3: {"used_mib": 16000, "apps": 1}}


def run_gate(lines, **kw):
    args = dict(claims_lines=lines, gpus=[1, 2], gpu_state=FREE, locks_free={1: True, 2: True}, mem_gib=95.0,
                mem_base_gib=70.0, mem_per_card_gib=9.5, min_cards=1, yield_file_exists=False, review_pending=False,
                claim_state=claim_state())
    args.update(kw)
    return vp.gate(**args)


TOOLCALL = "2026-10-14T14:28:20Z CLAIM GPU1,GPU2 owner=claude lane=toolcall what=llama-server"
TOOLCALL_REL = "2026-10-15T03:00:00Z RELEASE GPU1,GPU2 owner=claude lane=toolcall (done)"


def test_gate_waits_on_live_claim_then_opens():
    code, usable, why = run_gate([TOOLCALL])
    assert code == 10 and usable == [] and any("GPU1 claimed" in w for w in why)
    code, usable, why = run_gate([TOOLCALL, TOOLCALL_REL])
    assert code == 0 and usable == [1, 2]


def test_gate_ignores_own_stale_claim_and_respects_others():
    own = "2026-10-15T04:00:00Z CLAIM GPU1,GPU2 owner=claude lane=research-vmath job=vmath-pilot pid=1"
    assert run_gate([TOOLCALL, TOOLCALL_REL, own])[0] == 0
    seal = "2026-10-16T01:00:00Z CLAIM GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review pid=9"
    assert run_gate([TOOLCALL, TOOLCALL_REL, seal])[0] == 10


def test_gate_memory_review_yield_lock_and_busy():
    lines = [TOOLCALL, TOOLCALL_REL]
    code, usable, why = run_gate(lines, mem_gib=85.0)                                  # fits one card, not two
    assert code == 0 and usable == [1] and any("memory fits 1 of 2" in w for w in why)
    code, usable, why = run_gate(lines, mem_gib=75.0)
    assert code == 10 and any("MemAvailable 75.0 GiB < 79.5" in w for w in why)
    assert run_gate(lines, mem_gib=85.0, locks_free={1: True, 2: False})[0] == 0      # one card needs 79.5
    assert run_gate(lines, mem_gib=85.0, min_cards=2)[0] == 10
    assert run_gate(lines, review_pending=True)[0] == 10
    assert run_gate(lines, yield_file_exists=True)[0] == 10
    code, usable, _ = run_gate(lines, gpu_state={1: {"used_mib": 3000, "apps": 0}, 2: {"used_mib": 5, "apps": 0}})
    assert code == 0 and usable == [2]


def test_yield_check():
    mine = "2026-10-15T04:00:00Z CLAIM GPU1,GPU2 owner=claude lane=research-vmath job=vmath-pilot pid=1"
    base = [TOOLCALL, TOOLCALL_REL, mine]
    kw = dict(held=[1, 2], since_line=3, mem_gib=80.0, mem_floor_gib=70.0, yield_file_exists=False, review_pending=False)
    assert vp.yield_check(claims_lines=base, **kw)[0] == 0
    other = "2026-10-15T05:00:00Z GPU2 lane=speech owner=codex job=c-s2 pid=5"
    assert vp.yield_check(claims_lines=base + [other], **kw)[0] == 20
    gpu3 = "2026-10-15T05:00:00Z GPU3 lane=k3accel owner=claude job=x pid=5"
    assert vp.yield_check(claims_lines=base + [gpu3], **kw)[0] == 0
    assert vp.yield_check(claims_lines=base, **dict(kw, mem_gib=60.0))[0] == 20
    assert vp.yield_check(claims_lines=base, **dict(kw, review_pending=True))[0] == 20
    assert vp.yield_check(claims_lines=base, **dict(kw, yield_file_exists=True))[0] == 20


def test_mem_and_gpu_parsers():
    assert vp.mem_available_gib("MemTotal: 1 kB\nMemAvailable:   94371840 kB\n") == 90.0
    rows = vp.gpu_rows("1, 18077, GPU-a\n2, 5, GPU-b\n")
    assert rows == {1: {"used_mib": 18077}, 2: {"used_mib": 5}}
    assert vp.apps_by_gpu("GPU-a, 123\nGPU-a, 124\n", {"GPU-a": 1, "GPU-b": 2}) == {1: 2}


# ------------------------------------------------------------------------------------------- verify
def journal(index, rid, text, end="eog", tokens=100):
    return {"index": index, "rid": rid, "text": text, "end": end, "response_tokens": tokens, "seed": 1}


def test_verify_end_to_end(tmp_path, monkeypatch):
    sp, rp = write_fixture(tmp_path)
    monkeypatch.setattr(vp, "Renderer", lambda *a, **k: FakeRenderer())
    pdir = tmp_path / "prompts"
    assert vp.main(["prompts", "--scores", str(sp), "--rows", str(rp), "--out-dir", str(pdir), "--no-screen", "--max-prompt-tokens", "200", "--seed", "3"]) == 0
    prompts = vp.read_jsonl(pdir / "prompts.jsonl")
    refs = {r["source_rid"]: r["reference"] for r in vp.read_jsonl(pdir / "references.jsonl")}
    texts = []
    for i, p in enumerate(prompts):
        src, j = p["source_rid"], p["sample_index"]
        if p["stratum"] == "R":
            ans = refs[src] if j == 0 else "999"                     # one right, one wrong per referenced problem
            texts.append(journal(i, p["rid"], f"reasoning</think>\n\nSo \\boxed{{{ans}}}."))
        elif src == "a1":
            texts.append(journal(i, p["rid"], "work</think>\n\\boxed{\\frac{1}{2}}" if j == 0 else "work</think>\n\\boxed{0.5}"))
        else:
            texts.append(journal(i, p["rid"], "still thinking" if j == 0 else "x</think>\n\\boxed{25}", end="limit" if j == 0 else "eog"))
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "journal.jsonl", "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps(t) + "\n")
        fh.write('{"index": 99, "torn')                                # a torn tail line is skipped
    out = tmp_path / "verify"
    assert vp.main(["verify", "--run-dir", str(run), "--prompts-dir", str(pdir), "--out-dir", str(out)]) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))["strata"]
    assert s["R"]["samples"] == 4 and s["R"]["accepted_samples"] == 2 and s["R"]["accuracy_vs_reference_on_extractable_decided"] == 0.5
    assert s["R"]["problem_yield_any_accepted"] == 1.0
    assert s["A"]["accepted_samples"] == 2 and s["A"]["problem_yield_any_accepted"] == 0.5
    batch = out / "batch"
    man = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    rows = vp.read_jsonl(batch / "rows.jsonl")
    assert man["schema"] == vp.INTAKE_SCHEMA and man["rows"] == 4 and man["rows_sha256"] == vp.file_sha256(batch / "rows.jsonl")
    assert all(r["response"].startswith("<think>\n") and "</think>" in r["response"] for r in rows)
    qs = vp.qs()                                                      # the batch is a valid quality_scorer intake batch
    assert qs.main(["batch", "--batch-dir", str(batch), "--no-screen"]) == 0


def test_verify_refuses_foreign_rid(tmp_path, monkeypatch):
    sp, rp = write_fixture(tmp_path)
    monkeypatch.setattr(vp, "Renderer", lambda *a, **k: FakeRenderer())
    pdir = tmp_path / "prompts"
    vp.main(["prompts", "--scores", str(sp), "--rows", str(rp), "--out-dir", str(pdir), "--no-screen", "--max-prompt-tokens", "200"])
    run = tmp_path / "run"
    run.mkdir()
    (run / "journal.jsonl").write_text(json.dumps(journal(0, "nope~k0", "x</think>\\boxed{1}")) + "\n", encoding="utf-8")
    assert vp.main(["verify", "--run-dir", str(run), "--prompts-dir", str(pdir), "--out-dir", str(tmp_path / "v")]) == 2
