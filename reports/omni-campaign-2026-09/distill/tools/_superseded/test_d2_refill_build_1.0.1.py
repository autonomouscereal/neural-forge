"""Tests for d2_refill_build.py (lane refill, 2026-10-15).  CPU only; a synthetic Codex tree, no real teacher.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_d2_refill_build.py

What is pinned: the rank-3 copy keeps Codex's ids and masks byte for byte and drops exactly the rows the held-out screen
hits (a shared 13-gram with a held-out row; a 274-task answer stored as a nested object, found only through the task's
JSON form), leaving out only the pack's one scaffold system text; a train-pack row that is a held-out id or in a held-out
group is a refusal with nothing written; a corpus is never rewritten with other bytes; the receipt is the one d2_plan
1.4.0 refill_screen_state accepts (the two rules are compared on good and bad receipts whenever d2_plan has it); the
self-gen producer takes only rows not already in D2 that end at an EOG id and writes the D2 self-gen corpus format, or
nothing (exit 3) with a receipt; screen-toolcall writes the C1 receipt d2_plan screen_state reads and reports a hit
without editing the export; totals adds the ranks against d2_plan's span-form target.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d2_refill_build as R  # noqa: E402

SYSTEM = "Use the supplied local tools. Observe recorded tool errors and follow the user's stated recovery instructions. Report values from actual tool results."
HELD_TEXT = ("the quick brown fox jumps over the lazy dog while seven purple elephants dance on a frozen lake under the moon "
             "and nobody watches")
SCHEMA_ANSWER = {"properties": {"operation": {"enum": ["upper", "lower", "reverse", "strip", "words", "decode_hex"]},
                                "text": {"maxLength": 512, "type": "string"}}, "required": ["operation", "text"], "type": "object"}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def rec(rid: str, split: str, user: str, answer: str, *, group: str, system: bool = False, calls=None) -> dict:
    msgs = ([{"role": "system", "content": SYSTEM}] if system else []) + [{"role": "user", "content": user},
                                                                         {"role": "assistant", "content": answer}]
    prov = {"split": split, "group_id": group, "recovery_group_id": group, "source_verified": True, "template_family": group}
    if calls:
        prov["bfcl_schema_target_adaptation"] = {"original_acceptable_targets": {"expected_tool_calls": calls}}
    return {"id": rid, "messages": msgs, "provenance": prov}


def pack(codex: Path, name: str, rows: list[tuple[str, list[int], list[bool]]], split: str = "train") -> Path:
    import torch
    from safetensors.torch import save_file

    width = max(len(ids) for _, ids, _ in rows) + 2  # Codex's own padding past the longest row
    n = len(rows)
    ids = torch.full((n, width), 248044, dtype=torch.int64)
    mask = torch.zeros((n, width), dtype=torch.bool)
    lengths = torch.zeros((n,), dtype=torch.int64)
    for i, (_, x, m) in enumerate(rows):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(m)] = torch.tensor(m)
        lengths[i] = len(x)
    d = codex / name
    d.mkdir(parents=True, exist_ok=True)
    save_file({"input_ids": ids, "lengths": lengths, "target_mask": mask, "assistant_mask": mask.clone()}, str(d / "assistant_text.safetensors"))
    m = {"row_receipts": [{"row_id": rid, "route": "tool_call", "rendered_sha256": "r" * 64} for rid, _, _ in rows], "rows": n,
         "split": split, "quarantined_count": 0, "cross_split_groups": [], "sha256": sha(d / "assistant_text.safetensors"),
         "shard": str(d / "assistant_text.safetensors"), "supervised_tokens": int(mask.sum()), "tokens": int(lengths.sum()),
         "template_sha256": "t" * 64, "admission": "accepted rows only"}
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return d / "manifest.json"


@pytest.fixture()
def world(tmp_path, monkeypatch):
    codex = tmp_path / "codex"
    held = [rec("h-val-0", "validation", HELD_TEXT, "a held answer", group="g-held-0"),
            rec("h-test-0", "test", "what is two plus two", "four", group="g-held-1",
                calls=[{"calc": {"x": [2], "y": [2]}}]),
            rec("h-tool-0", "validation", "tool question zero", "tool answer zero", group="g-held-tool", system=True)]
    jl(codex / "corpus-schema-v2" / "validation.jsonl", [held[0]])
    jl(codex / "corpus-schema-v2" / "test.jsonl", [held[1]])
    jl(codex / "corpus-tool-v3" / "validation.jsonl", [held[2]])
    jl(codex / "corpus-tool-v3" / "test.jsonl", [])
    packs = []
    for name, rid in (("native-text-v3-validation", "h-val-0"), ("native-text-v3-test", "h-test-0"), ("native-text-v7-validation", "h-tool-0")):
        man = pack(codex, name, [(rid, [1, 2, 3], [False, True, True])], split="validation")
        packs.append({"name": name, "manifest": str(man), "manifest_sha256": sha(man), "rows": 1})
    train = [rec("tool-seed-a", "train", "uppercase alpha beta", "ALPHA BETA", group="fam-upper", system=True),
             rec("tool-seed-b", "train", "repeat: " + HELD_TEXT, "done", group="fam-echo", system=True),       # 13-gram hit
             rec("tool-seed-c", "train", "state the schema", json.dumps(SCHEMA_ANSWER), group="fam-schema", system=True),  # task JSON
             rec("tool-seed-d", "train", "lowercase GAMMA", "gamma", group="fam-lower", system=True)]
    jl(codex / "corpus-tool-v3" / "train.jsonl", train)
    rows = [(r["id"], [5, 6, 7, 8 + i, 9][: 4 + (i % 2)], [False, False, True, True, True][: 4 + (i % 2)]) for i, r in enumerate(train)]
    v7 = pack(codex, "native-text-v7-train", rows)
    tasks = [{"id": "bfcl-heldout:h-test-0", "input": {"messages": [{"role": "user", "content": "what is two plus two"}]}, "oracle": {"calls": 1}},
             {"id": "tool-seed-heldout-task", "input": {"messages": [{"role": "user", "content": "State the local_text parameter schema."}]},
              "oracle": {"schema": SCHEMA_ANSWER}}]
    tasks_file = jl(tmp_path / "tasks.jsonl", tasks)
    bfcl = tmp_path / "bfcl62_rids.json"
    bfcl.write_text(json.dumps({"rids": ["reg:bfcl:0"], "uids": ["h-test-0"], "task_ids": ["bfcl-heldout:h-test-0"],
                                "twin_exposure": {"twin_ids_by_source": {"native_train": ["twin-uid-0"], "registry_text": ["reg:twin:0"]}}}))
    hr = tmp_path / "heldout_rids.json"
    hr.write_text(json.dumps({"schema": "omni-d2-heldout-rids-v1", "rids": ["reg:held:0", "reg:held:1"]}))
    decisions = tmp_path / "D2_DECISIONS.json"
    decisions.write_text(json.dumps({"canonical_split": {"heldout_packs": packs},
                                     "refill_policy": {"order": [{"rank": 3, "rows": {
                                         "executed_tools_train": {"manifest_sha256": sha(v7), "rows": 4,
                                                                  "supervised_tokens": json.loads(v7.read_text())["supervised_tokens"]},
                                         "verified_intelligence_train": {"manifest_sha256": "0" * 64, "rows": 0, "supervised_tokens": 0}}}]}}))
    native_rows = tmp_path / "bfcl_native_rows.json"
    native_rows.write_text(json.dumps({"rows": [{"row_id": "some-bfcl-row"}]}))
    import d2_plan as P

    monkeypatch.setattr(R, "DECISIONS_SHA", sha(decisions))
    monkeypatch.setattr(P, "DECISIONS_SHA", sha(decisions))  # d2_plan >= 1.4.0 judges the receipt too (state / totals)
    monkeypatch.setattr(R, "HELDOUT_IDS_TOTAL", 3)
    monkeypatch.setattr(R, "HELDOUT_REGISTRY_RIDS", 2)
    out = tmp_path / "d2_refill"
    common = ["--tools", str(HERE), "--out", str(out), "--codex-dir", str(codex), "--decisions", str(decisions), "--bfcl62", str(bfcl),
              "--tasks", str(tasks_file), "--heldout-rids", str(hr), "--no-expect"]
    return {"codex": codex, "out": out, "common": common, "native_rows": native_rows, "train": train, "tmp": tmp_path}


def frozen_qtext():
    """The qtext_teacher_cache the refill caches are built with: the frozen 1.0.5 (d2clean_recache FROZEN), read from
    its _superseded copy here (same sha f2465a11), else the deployed tool."""
    import importlib.util

    path = HERE / "_superseded" / "qtext_teacher_cache_1.0.5.py"
    if not path.is_file():
        import qtext_teacher_cache as Q

        return Q
    assert sha(path) == "f2465a11b7a2242f167191fca8583a11ff74a50d33d7bba84b45c2d123ce6cb9"
    spec = importlib.util.spec_from_file_location("qtext_frozen_105", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_v7(world, *extra) -> int:
    return R.main(["codex-native", "--pack", "v7", *world["common"], "--bfcl-native-rows", str(world["native_rows"]), *extra])


def test_codex_native_copies_drops_screen_hits_and_writes_the_receipt(world):
    from safetensors.numpy import load_file

    assert build_v7(world) == 0
    out = world["out"] / "codex-native-v7-train.safetensors"
    t = load_file(str(out))
    # rows a and d kept (b shares a 13-gram with a held-out row, c answers the held-out task's schema verbatim)
    assert t["input_ids"].dtype.name == "int32" and t["lengths"].tolist() == [4, 5]
    assert t["input_ids"].shape == (2, 5) and t["input_ids"][0, :4].tolist() == [5, 6, 7, 8] and t["input_ids"][1].tolist() == [5, 6, 7, 11, 9]
    assert t["target_mask"].tolist() == [[False, False, True, True, False], [False, False, True, True, True]]
    assert (t["target_mask"] == t["assistant_mask"]).all()
    rows = json.loads(out.with_suffix(".rows.json").read_text())
    assert [r["row_id"] for r in rows["rows"]] == ["tool-seed-a", "tool-seed-d"] and rows["counts"]["dropped_rows"] == 2
    receipt = json.loads(R.screen_path(out).read_text())
    assert R.receipt_problems(receipt, sha(out)) == []
    assert receipt["hits_in_corpus"] == 0 and receipt["excluded_id_hits"] == 0 and receipt["split"] == "train"
    dropped = {d["row_id"]: d["hits"][0] for d in receipt["dropped_by_screen"]}
    assert dropped["tool-seed-b"]["rule"] == "ngram13" and dropped["tool-seed-b"]["source"].startswith("heldout:")
    assert dropped["tool-seed-c"]["source"] == "combined_oct13_v1" and dropped["tool-seed-c"]["item"] == "tool-seed-heldout-task"
    # the shared system text is scaffolding: left out of the candidate side only, and named
    assert receipt["system_scaffold"]["left_out_of_candidate_screen"] is True
    assert receipt["system_scaffold"]["sha256"] == hashlib.sha256(SYSTEM.encode()).hexdigest()
    assert receipt["system_scaffold"]["held_out_rows_opening_with_it"] == 1
    # the corpus loads through the frozen cache writer's own reader: stored positions = targets shifted by one
    c = frozen_qtext().load_text_corpus(out)
    assert c["store_mask"].sum().item() == int(t["target_mask"].sum()) and "target_mask shifted" in c["mask_source"]
    # idempotent: the same bytes again are fine
    assert build_v7(world) == 0 and sha(out) == rows["output"]["sha256"]


def test_the_system_scaffold_is_what_keeps_the_pack_alive(world, monkeypatch):
    """NEGATIVE control: screening the scaffold system text drops every row (it opens every held-out tool row too)."""
    monkeypatch.setitem(R.PACKS["v7"], "system_scaffold", False)
    assert build_v7(world) == 2
    assert not (world["out"] / "codex-native-v7-train.safetensors").exists()


@pytest.mark.parametrize("mutate,why", [
    (lambda t: t[0].__setitem__("id", "h-val-0"), "excluded ids"),                                   # a held-out id
    (lambda t: t[3]["provenance"].update(group_id="g-held-1", recovery_group_id="g-held-1"), "excluded ids"),  # a held-out group
    (lambda t: t[0].__setitem__("id", "twin-uid-0"), "excluded ids"),                                 # a BFCL-62 twin
    (lambda t: t[0]["provenance"].update(split="validation"), "excluded ids"),                       # not a train record
    (lambda t: t[0]["provenance"].update(registry_origin={"rid": "reg:7"}), "already D2 inputs"),    # a registry document
])
def test_excluded_or_already_used_rows_are_refusals_with_nothing_written(world, capsys, monkeypatch, mutate, why):
    train = [json.loads(json.dumps(r)) for r in world["train"]]
    mutate(train)
    jl(world["codex"] / "corpus-tool-v3" / "train.jsonl", train)
    # keep the manifest's row ids joined to the records
    man = world["codex"] / "native-text-v7-train" / "manifest.json"
    m = json.loads(man.read_text())
    for receipt, r in zip(m["row_receipts"], train):
        receipt["row_id"] = r["id"]
    man.write_text(json.dumps(m))
    d = json.loads(Path(world["common"][world["common"].index("--decisions") + 1]).read_text())
    d["refill_policy"]["order"][0]["rows"]["executed_tools_train"]["manifest_sha256"] = sha(man)
    dec = Path(world["common"][world["common"].index("--decisions") + 1])
    dec.write_text(json.dumps(d))
    monkeypatch.setattr(R, "DECISIONS_SHA", sha(dec))
    import d2_plan as P

    monkeypatch.setattr(P, "DECISIONS_SHA", sha(dec))
    assert build_v7(world) == 2
    assert why in capsys.readouterr().err
    assert not (world["out"] / "codex-native-v7-train.safetensors").exists()


def test_a_corpus_is_never_rewritten_with_other_bytes(world, capsys):
    assert build_v7(world) == 0
    train = [json.loads(json.dumps(r)) for r in world["train"]]
    train[1]["messages"][1]["content"] = "repeat something harmless"  # row b no longer hits: the output would change
    jl(world["codex"] / "corpus-tool-v3" / "train.jsonl", train)
    assert build_v7(world) == 2 and "immutable" in capsys.readouterr().err


GOOD = {"schema": R.SCREEN_SCHEMA, "ngram": 13, "hits_in_corpus": 0, "split": "train", "excluded_id_hits": 0,
        "verified_rows_only": True}
BAD = [{"hits_in_corpus": 1}, {"split": "validation"}, {"excluded_id_hits": 2}, {"verified_rows_only": False},
       {"ngram": 8}, {"schema": "omni-d2x1-toolcall-screen-v1"}, {"decisions_sha256": "0" * 64}, {"corpus_sha256": "1" * 64}]


def test_receipt_rule_and_its_agreement_with_d2_plan(tmp_path):
    corpus = tmp_path / "x.safetensors"
    corpus.write_bytes(b"corpus")
    good = {**GOOD, "decisions_sha256": R.DECISIONS_SHA, "corpus_sha256": sha(corpus)}
    assert R.receipt_problems(good, sha(corpus)) == [] and R.receipt_problems(None, sha(corpus))
    for bad in BAD:
        assert R.receipt_problems({**good, **bad}, sha(corpus)), bad
    import d2_plan as P

    if not hasattr(P, "refill_screen_state"):
        pytest.skip(f"d2_plan {P.TOOL_VERSION} has no refill_screen_state (< 1.4.0): nothing to compare against")
    assert P.REFILL_SCREEN_SCHEMA == R.SCREEN_SCHEMA and P.DECISIONS_SHA == R.DECISIONS_SHA
    for case in [{}] + BAD:
        R.screen_path(corpus).write_text(json.dumps({**good, **case}))
        st = P.refill_screen_state(None, corpus, sha(corpus))
        assert st["ok"] == (R.receipt_problems({**good, **case}, sha(corpus)) == []), case


# ------------------------------------------------------------------------------------------------ self-gen (rank 2)
def sealed_selfgen_run(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    import selfgen_sampler as sg

    prompts = jl(tmp_path / "prompts.jsonl", [{"rid": r["rid"], "messages": [{"role": "user", "content": r["user"]}]} for r in rows])
    run = tmp_path / "run"
    run.mkdir()
    records, prompt_ids = [], {}
    for i, r in enumerate(rows):
        pids = [100 + i, 101, 102]
        prompt_ids[i] = pids
        resp = r["response"]
        labels = sg.end_labels(r["stop"], resp)
        records.append({"index": i, "rid": r["rid"], "prompt_ids_sha256": sg.bytes_sha256(struct.pack(f"<{len(pids)}I", *pids)),
                        "response_ids": resp, "response_tokens": len(resp), "stop_type": r["stop"], **labels, "undefined_ids": 0,
                        "text": r["text"], "text_matches_server": True})
    shard = sg.write_shard(run, 0, records, prompt_ids, (0, len(rows)))
    manifest = sg.seal({"schema": sg.MANIFEST_SCHEMA, "shard_read_order": ["shard-00000"],
                        "shards": [{"name": "shard-00000", "shard_sha256": shard["shard_sha256"]}],
                        "prompts": {"path": str(prompts), "sha256": sha(prompts)}}, "manifest_sha256")
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run, prompts


def test_selfgen_refill_takes_only_new_eog_rows_in_the_d2_selfgen_format(world, tmp_path):
    from safetensors.numpy import load_file

    rows = [{"rid": "reg:in-d2", "user": "q0", "response": [7, 248046], "stop": "eos", "text": "a0"},
            {"rid": "reg:cut", "user": "q1", "response": [7, 8, 9], "stop": "limit", "text": "a1"},
            {"rid": "reg:held:1", "user": "q2", "response": [7, 248046], "stop": "eos", "text": "a2"},
            {"rid": "reg:hit", "user": "q3", "response": [7, 248046], "stop": "eos", "text": HELD_TEXT},
            {"rid": "reg:new", "user": "q4", "response": [7, 8, 248046], "stop": "eos", "text": "a4"}]
    run, _ = sealed_selfgen_run(tmp_path, rows)
    d2 = tmp_path / "d2_rows.json"
    d2.write_text(json.dumps({"source_manifest_sha256": R.SELFGEN_D2_MANIFEST, "row_ids": ["reg:in-d2"]}))
    rc = R.main(["selfgen", "--run", str(run), "--d2-rows", str(d2), *world["common"]])
    assert rc == 0
    out = world["out"] / R.SELFGEN_OUT
    t = load_file(str(out))
    assert sorted(t) == ["eog_terminated", "input_ids", "lengths", "store_mask", "target_mask"]
    assert t["input_ids"].tolist() == [[104, 101, 102, 7, 8, 248046]] and t["lengths"].tolist() == [6]
    assert t["target_mask"].tolist() == [[0, 0, 0, 1, 1, 1]] and t["store_mask"].tolist() == [[0, 0, 1, 1, 1, 0]]
    receipt = json.loads(R.screen_path(out).read_text())
    assert R.receipt_problems(receipt, sha(out)) == [] and receipt["rank"] == 2
    c = receipt["counts"]
    assert (c["already_in_d2"], c["not_eog_terminated"], c["excluded:heldout_or_bfcl62_registry_rid"], c["dropped_by_screen"]) == (1, 1, 1, 1)
    assert "store_mask, checked equal to target_mask shifted by one" == frozen_qtext().load_text_corpus(out)["mask_source"]


def test_selfgen_with_every_row_already_in_d2_writes_no_corpus(world, tmp_path):
    rows = [{"rid": "reg:in-d2", "user": "q0", "response": [7, 248046], "stop": "eos", "text": "a0"}]
    run, _ = sealed_selfgen_run(tmp_path, rows)
    d2 = tmp_path / "d2_rows.json"
    d2.write_text(json.dumps({"source_manifest_sha256": R.SELFGEN_D2_MANIFEST, "row_ids": ["reg:in-d2"]}))
    assert R.main(["selfgen", "--run", str(run), "--d2-rows", str(d2), *world["common"]]) == R.EXIT_NOTHING
    assert not (world["out"] / R.SELFGEN_OUT).exists()
    (receipt,) = list((world["out"] / "receipts").glob("selfgen-refill-none-*.json"))
    assert json.loads(receipt.read_text())["counts"] == {"rows_read": 1, "already_in_d2": 1, "eligible_rows": 0}


# ------------------------------------------------------------------------------------------------ C1 screen (rank 1)
def test_screen_toolcall_writes_the_c1_receipt_and_reports_hits_without_editing(world, tmp_path):
    import toolcall_selfgen as tc

    def run(name: str, convs: list[dict]) -> dict:
        d = tmp_path / name
        d.mkdir()
        conv = jl(d / "conversations.jsonl", convs)
        m = tc.seal({"file": {"path": str(conv), "sha256": sha(conv)}, "run_config_sha256": "c" * 64}, "manifest_sha256")
        (d / "manifest.json").write_text(json.dumps(m))
        return {"out_dir": str(d), "manifest_sha256": m["manifest_sha256"]}

    tool = {"type": "function", "function": {"name": "f", "description": "does f", "parameters": {"properties": {"x": {"description": "an x"}}}}}
    clean = {"id": "tc-0", "category": "single", "tools": [tool], "messages": [{"role": "user", "content": "call f with x"}]}
    leaky = {"id": "tc-1", "category": "single", "tools": [tool], "messages": [{"role": "user", "content": HELD_TEXT}]}
    corpus = tmp_path / "toolcall.safetensors"
    corpus.write_bytes(b"export")
    for convs, rc, hits in (([clean], 0, 0), ([clean, leaky], 2, 1)):
        sources = [run(f"run{len(convs)}", convs)]
        side = tmp_path / f"side{len(convs)}.rows.json"
        side.write_text(json.dumps({"corpus": {"path": str(corpus), "sha256": sha(corpus)}, "source_manifests": sources,
                                    "row_ids": [c["id"] for c in convs]}))
        out = tmp_path / f"heldout_screen{len(convs)}.json"
        args = ["screen-toolcall", "--rows", str(side), *world["common"]]
        args[args.index("--out") + 1] = str(out)
        assert R.main(args) == rc
        body = json.loads(out.read_text())
        assert body["schema"] == R.TOOLCALL_SCREEN_SCHEMA and body["corpus_sha256"] == sha(corpus) and body["ngram"] == 13
        assert body["hits_in_corpus"] == hits and body["decisions_sha256"] == R.DECISIONS_SHA
    assert corpus.read_bytes() == b"export"


# ------------------------------------------------------------------------------------------------ state / totals
def test_state_and_totals_add_the_ranks_against_d2_plans_target(world, tmp_path):
    import d2_plan as P

    assert build_v7(world) == 0
    out = world["out"]
    corpus = out / "codex-native-v7-train.safetensors"
    cache_root = tmp_path / "caches"
    masks = tmp_path / "masks_v2"
    masks.mkdir()
    (masks / "build_receipt.json").write_text(json.dumps({"refill": {"heldout_trained_positions_removed": 3_095_968,
                                                                     "positions_after_v1_exclusion": 19_689_472,
                                                                     "kd_positions_v1": 19_000_000}}))
    (masks / "exclude_rows.json").write_text(json.dumps({"corpora": {"s0": {"rows": [0, 1]}}}))
    text_manifest = tmp_path / "corpus" / "text" / "manifest.json"
    text_manifest.parent.mkdir(parents=True)
    text_manifest.write_text(json.dumps({"shards": [{"sha256": "s0", "rows": [0, 4883]}]}))
    tc = tmp_path / "toolcall"
    (tc / "run-q38-q4km-2026-10-14").mkdir(parents=True)
    (tc / "run-q38-q4km-2026-10-14" / "manifest.json").write_text(json.dumps({"tokens": 1_500_469, "supervised_tokens": 363_784}))
    extra = ["--tools", str(HERE), "--out", str(out), "--cache-root", str(cache_root), "--toolcall-dir", str(tc),
             "--masks-v2", str(masks), "--text-manifest", str(text_manifest)]
    assert R.main(["totals", *extra]) == 0
    body = json.loads((out / "REFILL_TOTALS.json").read_text())
    want = int(round(3_095_968 * min(1.0, (P.REGISTRY_DRAW + P.ONPOLICY_FALLBACK) / 19_689_472)))
    assert body["target_tokens"] == want == 3_095_968
    by = {(r["rank"], r.get("source")): r for r in body["ranks"]}
    assert by[(1, "tool-call self-gen beyond A2 2(a) (C1 runs 2..)")]["supervised"] == int(round(363_784 / 1_500_469 * 1_499_531))
    v7 = by[(3, "refill-codex-v7-train")]
    assert v7["eligible"] and v7["supervised"] == 5 and v7["cache_sealed"] is False and v7["corpus_sha256"] == sha(corpus)
    assert by[(2, "refill-selfgen")]["supervised"] == 0 and body["rank2_next"] is not None
    assert by[(4, "undrawn registry positions (span-mask form)")]["supervised"] == 0  # 4,881 x 4,096 < the 20M draw
    assert body["expected_supervised_total"] == sum(r["supervised"] for r in body["ranks"])
    assert body["expected_shortfall"] == want - body["expected_supervised_total"] and body["target_met"] is False
    assert body["clean_form"] is None  # no text_clean_v1 beside the registry
    clean = tmp_path / "corpus" / "text_clean_v1"
    clean.mkdir()
    (clean / "manifest.json").write_text(json.dumps({"counts": {"kd_positions_clean_total": 100}, "shards": [{"rows": [0, 10]}],
                                                    "manifest_sha256": "m" * 64}))
    (clean / "clean_exclude_rows.json").write_text(json.dumps({"corpora": {"s0": {"rows": [0, 1, 2]}}}))
    assert R.main(["totals", *extra]) == 0
    cf = json.loads((out / "REFILL_TOTALS.json").read_text())["clean_form"]
    f_clean = min(1.0, 20_000_000 / ((4883 - 3 + 10) * 4096))
    assert cf["registry_positions"] == 4890 * 4096 and cf["target_tokens"] == int(round(19_000_000 * 1.0 - 100 * f_clean))
    # state: the runner's view of the same candidates
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert R.main(["state", *extra]) == 0
    line = next(x for x in buf.getvalue().splitlines() if x.startswith("D2REFILL state "))
    st = {c["name"]: c for c in json.loads(line[len("D2REFILL state "):])["candidates"]}
    assert st["refill-codex-v7-train"]["eligible"] and st["refill-codex-v7-train"]["cache_dir"] == str(cache_root / sha(corpus))
    assert st["refill-selfgen"]["why"] == "corpus file does not exist yet"
    # a tampered corpus is no longer eligible (the receipt names the old sha)
    corpus.write_bytes(corpus.read_bytes() + b"\0")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.main(["state", *extra])
    line = next(x for x in buf.getvalue().splitlines() if x.startswith("D2REFILL state "))
    st = {c["name"]: c for c in json.loads(line[len("D2REFILL state "):])["candidates"]}
    assert st["refill-codex-v7-train"]["eligible"] is False and "screened corpus" in st["refill-codex-v7-train"]["why"]
