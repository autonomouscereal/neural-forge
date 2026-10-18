"""Tests for ``toolcall_heldout_screen.py`` (lane toolscreen, 2026-10-15).

What is REAL here: toolcall_selfgen's sealed-run layout and its export sidecar format (written here exactly as
export_command writes them; with the parent tokenizer on the machine, the real `export` and qtext_teacher_cache's
loader run too), toolcall_selfgen.screen_texts, the real toolcall bank (tc.build_bank, sealed), distill_corpus_build's
word rule / harvest_strings / ReviewIndex / screen_liveness, and d2_plan's own screen_state reading the receipt back.
Stand-ins: the held-out packs are small fixture safetensors decoded by a fixture word tokenizer (the real run decodes
with the parent's pinned tokenizer.json), the review surface is a small ReviewIndex, and the pinned shas are the
fixture files' own (monkeypatched constants).

  test_constants_agree_with_d2_plan            schema, decisions sha and receipt path are the consumer's
  test_index_is_verbatim_13gram                13 words inside filler / re-wrapped hit; 12 words do not; a hash
                                               collision never counts (verified verbatim)
  test_liveness_survives_case_mapping          German sharp s in a probe item: the re-wrap probe keeps it live
  test_literal_names_hit_tool_names            a tool named get_flight_status is caught through the names line
  test_planted_heldout_13gram_is_dropped       end to end: planted Codex-pack 13-gram, BFCL-62 twin 13-gram, 274-task
                                               13-gram and a held-out function name are dropped (and listed); a 12-word
                                               fragment row is kept; corpus + sidecar rewritten (backup first); receipt
                                               0 hits naming the new sha; d2_plan.screen_state accepts it; check 0;
                                               re-run keeps it; --force re-screens with 0 hits and the prior drop
                                               carried; a changed corpus -> check 10
  test_clean_export_is_left_untouched          no hit: corpus bytes unchanged, empty dropped list, receipt names it
  test_refusals_change_nothing                 drop fraction above the cap (would-drop list, applied false), decisions
                                               sha, sidecar sha, tampered run, row not bound to its record, decode
                                               garbage (fidelity), a source with no probe (dead), 274-task sha -> rc 2,
                                               corpus bytes unchanged, no receipt
  test_real_export_round_trip                  (parent tokenizer only) real tc export -> screen -> qtext load

Server:
  cd $OC/distill/tools && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OC/distill/tools \\
    $PY -m pytest -o addopts="" -p no:cacheprovider test_toolcall_heldout_screen.py -q
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d2_plan  # noqa: E402
import distill_corpus_build as dcb  # noqa: E402
import toolcall_heldout_screen as hs  # noqa: E402
import toolcall_selfgen as tc  # noqa: E402

PARENT = Path(os.environ.get("TOOLCALL_PARENT", str(tc.DEFAULT_PARENT)))
needs_parent = pytest.mark.skipif(not (PARENT / "tokenizer.json").is_file(), reason="parent tokenizer not on this machine")

# ------------------------------------------------------------------------------------------------ fixture text
CODEX_TEXT = ("the quarterly maintenance schedule requires that every coolant pump at the eastern substation is inspected "
              "for bearing wear before the winter shutdown and that the inspection notes are filed with the regional office")
BFCL_TEXT = ("find out how fast an object was going if it started from rest and traveled a distance of twenty meters over "
             "four seconds due to a constant acceleration")
TASK_TEXT = ("write a python function that returns the distinct integers in the list keeping the order of their first "
             "occurrence and include the function in your answer")
REVIEW_TEXT = ("a lighthouse keeper records the passing ships in a leather ledger every evening and the ledger is "
               "later copied by the harbour clerk into the official shipping register")
FILLER = "Sure, here is what I found after checking the numbers for you, everything adds up nicely."


def words_of(text: str) -> list[str]:
    return dcb.words(text)


class WordTokenizer:
    """Fixture decoder: id k -> the k-th vocabulary word (the real run decodes with the pinned tokenizer.json)."""

    def __init__(self, texts: list[str]) -> None:
        vocab = sorted({w for t in texts for w in t.split()})
        self.vocab = ["<pad>"] + vocab
        self.ids = {w: i for i, w in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        return [self.ids[w] for w in text.split()]

    def decode(self, rows: list[list[int]]) -> list[str]:
        return [" ".join(self.vocab[i] for i in row) for row in rows]


def tool(name: str, description: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "query text"}},
                                                            "required": ["q"]}}}


def conversation(cid: str, answer: str, tool_name: str = "lookup_inventory", category: str = "single") -> dict:
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please help me with request {cid}."},
                {"role": "assistant", "content": answer}]
    return {"id": cid, "category": category, "language": "English", "domain": "test", "messages": messages,
            "tools": [tool(tool_name, "Look up how many items of a product are in stock.")], "checks": {"calls": 0}}


def seal_run(out_dir: Path, records: list[dict], renders: list[tuple[int, int, str]]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for record, (tokens, supervised, text_sha) in zip(records, renders):
        record["render"] = {"tokens": tokens, "supervised": supervised, "text_sha256": text_sha}
    sha, count = tc.write_jsonl(out_dir / "conversations.jsonl", records)
    manifest = tc.seal({"schema": tc.SCHEMA + "/manifest", "run_config_sha256": "f" * 64, "conversations": count,
                        "tokens": sum(r[0] for r in renders), "supervised_tokens": sum(r[1] for r in renders),
                        "file": {"path": str(out_dir / "conversations.jsonl"), "sha256": sha}}, "manifest_sha256")
    tc.write_json_atomic(out_dir / "manifest.json", manifest)
    return manifest


def write_export(dest: Path, runs: list[Path]) -> None:
    """toolcall_selfgen.export_command's corpus + sidecar, with synthetic ids (the screen reads the conversations)."""

    records, sources = [], []
    for out_dir in runs:
        manifest, rows = tc.sealed_conversations(out_dir)
        sources.append({"out_dir": str(out_dir), "manifest_sha256": manifest["manifest_sha256"], "run_config_sha256": manifest["run_config_sha256"],
                        "row_range": [len(records), len(records) + len(rows)], "conversations": len(rows),
                        "tokens": manifest["tokens"], "supervised_tokens": manifest["supervised_tokens"]})
        records += rows
    width = max(r["render"]["tokens"] for r in records)
    ids = np.full((len(records), width), tc.PAD_ID, dtype=np.int32)
    target = np.zeros((len(records), width), dtype=np.uint8)
    lengths = np.zeros((len(records),), dtype=np.int64)
    for n, r in enumerate(records):
        L, S = r["render"]["tokens"], r["render"]["supervised"]
        ids[n, :L] = (np.arange(L) + 7 * n) % 1000 + 10
        target[n, 1 : 1 + S] = 1
        lengths[n] = L
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file({"input_ids": ids, "lengths": lengths, "target_mask": target}, str(dest),
              metadata={"source_manifest_sha256": "+".join(s["manifest_sha256"] for s in sources), "tool": "toolcall_selfgen 1.2.0", "route": tc.ROUTE})
    side = tc.seal({"schema": tc.SCHEMA + "/rescore-corpus", "tool": {"tool": "toolcall_selfgen", "version": "1.2.0"},
                    "corpus": {"path": str(dest.resolve()), "sha256": tc.file_sha256(dest), "bytes": dest.stat().st_size},
                    "source_manifest_sha256": "+".join(s["manifest_sha256"] for s in sources), "source_manifests": sources,
                    "rows": len(records), "row_length": width, "tokens": int(lengths.sum()), "supervised_tokens": int(target.sum()),
                    "pad_id": tc.PAD_ID, "row_ids": [r["id"] for r in records], "categories": [r["category"] for r in records],
                    "route": tc.ROUTE}, "receipt_sha256")
    tc.write_json_atomic(dest.with_suffix(".rows.json"), side)


def build_world(tmp: Path, monkeypatch, *, plant: bool = True, n_rows: int = 30) -> dict:
    """Held-out fixture sources + two sealed runs + their export at d2_plan's C1 path under tmp/oc."""

    oc = tmp / "oc"
    held = tmp / "held"
    held.mkdir(parents=True)
    # -- six held-out packs (one row of real text each + fillers), decoded by the fixture tokenizer
    pack_texts = {"original_short_validation": [CODEX_TEXT, "a second validation row about " + " ".join(f"topic{i}" for i in range(20))],
                  "original_short_test": ["test row " + " ".join(f"word{i}" for i in range(30))],
                  "original_long_validation": ["long validation " + " ".join(f"lv{i}" for i in range(40))],
                  "original_long_test": ["long test " + " ".join(f"lt{i}" for i in range(40))],
                  "executed_tools_validation": ["executed tool validation " + " ".join(f"ev{i}" for i in range(25))],
                  "executed_tools_test": ["executed tool test " + " ".join(f"et{i}" for i in range(25))]}
    every_text = [t for ts in pack_texts.values() for t in ts]
    tok = WordTokenizer(every_text)
    packs, schema_rows = [], {"validation": [], "test": []}
    n = 0
    for name, texts in pack_texts.items():
        d = held / name
        d.mkdir()
        rows = [tok.encode(t) for t in texts]
        width = max(len(r) for r in rows)
        ids = np.zeros((len(rows), width), dtype=np.int32)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = r
        shard = d / "assistant_text.safetensors"
        save_file({"input_ids": ids, "lengths": np.array([len(r) for r in rows], dtype=np.int64)}, str(shard))
        row_ids = [f"heldout-row-{n + i}" for i in range(len(rows))]
        n += len(rows)
        manifest = {"rows": len(rows), "sha256": tc.file_sha256(shard), "shard": str(shard), "row_receipts": [{"row_id": r} for r in row_ids]}
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        packs.append({"name": name, "manifest": str(d / "manifest.json"), "manifest_sha256": tc.file_sha256(d / "manifest.json"),
                      "corpus_sha256": manifest["sha256"], "rows": len(rows)})
        if not name.startswith("executed_tools"):
            for rid, text in zip(row_ids, texts):
                schema_rows["validation" if "validation" in name else "test"].append(
                    {"id": rid, "messages": [{"role": "user", "content": "short"}, {"role": "assistant", "content": text}]})
    lists = []
    for split, rows in schema_rows.items():
        p = held / f"{split}.jsonl"
        tc.write_jsonl(p, rows)
        lists.append({"path": str(p), "sha256": tc.file_sha256(p)})
    # -- BFCL-62: 3 source uids (BFCL_TASKS patched to 3) + 2 twins, records in a normalized.jsonl
    uids = [f"{i:064x}" for i in range(1, 4)]
    twin_uids = [f"{i:064x}" for i in range(10, 12)]
    normalized = held / "normalized.jsonl"
    norm_rows = [{"id": u, "messages": [{"role": "user", "content": f"bfcl source {i} " + " ".join(f"b{i}x{j}" for j in range(15))}],
                  "tools": [tool(f"bfcl_tool_{i}", "A BFCL tool description that is long enough to index well.")]} for i, u in enumerate(uids)]
    norm_rows += [{"id": twin_uids[0], "messages": [{"role": "user", "content": BFCL_TEXT}], "tools": []},
                  {"id": twin_uids[1], "messages": [{"role": "user", "content": "native train twin " + " ".join(f"nt{j}" for j in range(15))}], "tools": []},
                  {"id": "f" * 64, "messages": [{"role": "user", "content": "an unrelated normalized record " + " ".join(f"u{j}" for j in range(15))}]}]
    tc.write_jsonl(normalized, norm_rows)
    lists.append({"path": str(normalized), "sha256": tc.file_sha256(normalized), "role": "normalized"})
    bfcl = {"uids": uids, "twin_exposure": {
        "hits": [{"uid": twin_uids[0], "rid": "547:x:99", "sources": ["text", "native_train"]}, {"uid": twin_uids[1], "rid": "547:x:174", "sources": ["k3"]}],
        "twin_ids_by_source": {"k3": ["547:x:174"], "long": [], "native_train": [twin_uids[0]], "selfgen": [], "text": ["547:x:99"]}}}
    bfcl_path = held / "bfcl62_rids.json"
    bfcl_path.write_text(json.dumps(bfcl), encoding="utf-8")
    # -- 274-task set (patched to 4 tasks, 3 BFCL)
    tasks = [{"id": f"bfcl-heldout:{u}", "input": {"messages": [{"role": "user", "content": f"bfcl task {i} " + " ".join(f"bt{i}y{j}" for j in range(14))}], "tools": []},
              "oracle": {"kind": "tool"}} for i, u in enumerate(uids)]
    tasks.append({"id": "text-eval-stable_unique-0", "input": {"messages": [{"role": "user", "content": TASK_TEXT}], "tools": []}, "oracle": {"expected": [4, 2, 3]}})
    tasks_path = held / "tasks.jsonl"
    tc.write_jsonl(tasks_path, tasks)
    # -- the real toolcall bank
    bank_path = held / "omni_toolcall_bank_v1.json"
    tc.write_json_atomic(bank_path, tc.build_bank())
    decisions = {"canonical_split": {"heldout_packs": packs, "row_id_lists": lists, "cross_checks": {"heldout_ids_total": n}},
                 "exclusion": {"sets": {"bfcl62": {"path": str(bfcl_path), "sha256": tc.file_sha256(bfcl_path)}}}}
    decisions_path = held / "D2_DECISIONS_2026-10-15.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    monkeypatch.setattr(hs, "DECISIONS_SHA", tc.file_sha256(decisions_path))
    monkeypatch.setattr(d2_plan, "DECISIONS_SHA", tc.file_sha256(decisions_path))
    monkeypatch.setattr(hs, "EVAL_TASKS_SHA", tc.file_sha256(tasks_path))
    monkeypatch.setattr(hs, "EVAL_TASKS_ROWS", len(tasks))
    monkeypatch.setattr(hs, "BFCL_TASKS", len(uids))
    monkeypatch.setattr(hs, "pack_decoder", lambda parent_dir: tok.decode)
    monkeypatch.setattr(hs, "ROOT_FLOOR_BYTES", 0)  # the fixture disk is not the server root

    def review_surface(args):
        index = dcb.ReviewIndex()
        index.add_text("bank_v1", "bank_v1#0", REVIEW_TEXT)
        return index, {"review_sources": [{"name": "bank_v1"}], "added_sources": [], "index": index.summary(), "corpus_tool": {}}

    monkeypatch.setattr(hs, "build_review_surface", review_surface)
    # -- two sealed runs; run 2 carries the plants
    answers = [f"{FILLER} Row {i} reports {i * 3} units in aisle {i % 7}." for i in range(n_rows)]
    names = ["lookup_inventory"] * n_rows
    planted = {}
    if plant:
        codex13 = " ".join(words_of(CODEX_TEXT)[10:23])
        bfcl13 = " ".join(words_of(BFCL_TEXT)[3:16])
        task13 = " ".join(words_of(TASK_TEXT)[2:15])
        frag12 = " ".join(words_of(CODEX_TEXT)[0:12])
        answers[n_rows - 5] = f"{FILLER} As noted, {codex13}, so that is done."
        answers[n_rows - 4] = f"Quick answer: {bfcl13}."
        answers[n_rows - 3] = f"Here you go: {task13} -- hope it helps."
        names[n_rows - 2] = "get_flight_status"
        answers[n_rows - 1] = f"Only a fragment: qqzxv {frag12} vxzqq."
        planted = {"codex": f"tc-{n_rows - 5:04d}", "bfcl": f"tc-{n_rows - 4:04d}", "task": f"tc-{n_rows - 3:04d}",
                   "name": f"tc-{n_rows - 2:04d}", "fragment": f"tc-{n_rows - 1:04d}"}
    records = [conversation(f"tc-{i:04d}", answers[i], names[i]) for i in range(n_rows)]
    renders = [(40 + i, 10 + i % 5, "0" * 64) for i in range(n_rows)]
    half = n_rows // 2
    run1, run2 = oc / "distill/toolcall/run-1", oc / "distill/toolcall/run-2"
    seal_run(run1, records[:half], renders[:half])
    seal_run(run2, records[half:], renders[half:])
    corpus = oc / "distill/toolcall/rescore/toolcall-q38-2026-10-14.safetensors"
    write_export(corpus, [run1, run2])
    args = ["screen", "--corpus", str(corpus), "--decisions", str(decisions_path), "--eval-tasks", str(tasks_path),
            "--bank", str(bank_path), "--eval-split", str(held / "unused"), "--parent-dir", str(held), "--backup-root", str(tmp / "unscreened"),
            "--lock-wait", "5"]
    return {"oc": oc, "corpus": corpus, "args": args, "planted": planted, "run1": run1, "run2": run2, "records": records,
            "receipt": corpus.parent / "heldout_screen.json", "dropped": corpus.parent / "heldout_screen.dropped.json"}


# ------------------------------------------------------------------------------------------------ unit tests


def test_constants_agree_with_d2_plan(tmp_path):
    assert hs.SCHEMA == d2_plan.SCREEN_SCHEMA
    assert hs.DECISIONS_SHA == d2_plan.DECISIONS_SHA
    assert hs.NGRAM == 13
    layout = d2_plan.Layout(oc=tmp_path)
    args = hs.parser().parse_args(["screen", "--corpus", str(layout.toolcall_corpus)])
    assert hs.receipt_paths(args)[0] == layout.toolcall_screen
    assert hs.DEFAULT_CORPUS.name == layout.toolcall_corpus.name


def test_index_is_verbatim_13gram():
    index = hs.HeldoutIndex()
    index.add_text("codex_heldout", "row-1", CODEX_TEXT)
    w = words_of(CODEX_TEXT)
    assert index.hits(f"Intro text. {' '.join(w[5:18])} and more.")[0]["source"] == "codex_heldout"
    assert index.hits("LOG:\n" + "\n".join(x.upper() for x in w[5:18]))
    assert index.hits(f"qqzxv {' '.join(w[5:17])} vxzqq") == []
    # a key that collides with a gram the item does not contain is never a hit
    fake = tuple(f"zz{i}" for i in range(13))
    index.grams[hash(fake)] = 0
    assert index.hits(" ".join(fake)) == []


def test_liveness_survives_case_mapping():
    index = hs.HeldoutIndex()
    index.add_text("codex_heldout", "de-1", "die straße führt am großen fluss entlang und biegt dann nach dem alten "
                                            "bahnhof scharf nach links in richtung der mühle ab")
    index.add_literal("toolcall_bank_functions", "get_flight_status", "get_flight_status")
    live = hs.liveness(index)
    assert live["screen_is_live"], live


def test_literal_names_hit_tool_names():
    index = hs.HeldoutIndex()
    index.add_literal("toolcall_bank_functions", "get_flight_status", "get_flight_status")
    clean = conversation("a", FILLER)
    named = conversation("b", FILLER, tool_name="get_flight_status")
    assert not any(index.hits(t) for t in hs.screened_texts(clean))
    assert any(h["rule"] == "literal" for t in hs.screened_texts(named) for h in index.hits(t))


# ------------------------------------------------------------------------------------------------ end to end


def test_planted_heldout_13gram_is_dropped(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch)
    corpus, planted = w["corpus"], w["planted"]
    before = tc.file_sha256(corpus)
    assert hs.main(w["args"] + ["--max-drop-fraction", "0.2"]) == 0
    receipt = json.loads(w["receipt"].read_text(encoding="utf-8"))
    after = tc.file_sha256(corpus)
    assert after != before and receipt["corpus_sha256"] == after and receipt["hits_in_corpus"] == 0 and receipt["ngram"] == 13
    assert receipt["schema"] == "omni-d2x1-toolcall-screen-v1" and tc.check_seal(receipt, "receipt_sha256")
    dropped = json.loads(w["dropped"].read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in dropped["dropped"]}
    assert dropped["applied"] is True and set(by_id) == {planted["codex"], planted["bfcl"], planted["task"], planted["name"]}
    assert by_id[planted["codex"]]["reasons"][0]["source"] == "codex_heldout"
    assert by_id[planted["bfcl"]]["reasons"][0]["source"] == "bfcl62_twin"
    assert by_id[planted["task"]]["reasons"][0]["source"] == "codex_274"
    assert by_id[planted["name"]]["reasons"][0]["rule"] == "literal"
    assert receipt["hits_in_export"] == 4 and receipt["dropped"]["rows"] == 4 and receipt["dropped"]["sha256"] == tc.file_sha256(w["dropped"])
    # the corpus and sidecar hold exactly the kept rows, bound to their records
    side = json.loads(corpus.with_suffix(".rows.json").read_text(encoding="utf-8"))
    assert tc.check_seal(side, "receipt_sha256") and side["corpus"]["sha256"] == after
    assert planted["fragment"] in side["row_ids"] and not set(by_id) & set(side["row_ids"]) and side["rows"] == 26
    assert [s["row_range"] for s in side["source_manifests"]] == [[0, 15], [15, 26]]
    assert sum(s["supervised_tokens_kept"] for s in side["source_manifests"]) == side["supervised_tokens"]
    reloaded = hs.load_export(corpus)
    assert [r["id"] for r in reloaded["records"]] == side["row_ids"]
    assert reloaded["tensors"]["input_ids"].shape[1] == int(reloaded["tensors"]["lengths"].max())
    # the unscreened export was copied first
    backup = Path(receipt["rewrite"]["backup_dir"])
    assert tc.file_sha256(backup / corpus.name) == before
    assert tc.file_sha256(backup / w["dropped"].name) == receipt["dropped"]["sha256"]
    # the consumer: d2_plan's own screen_state accepts the receipt for the corpus as it is now
    layout = d2_plan.Layout(oc=w["oc"])
    assert layout.toolcall_corpus == corpus
    st = d2_plan.screen_state(layout, {"toolcall_c1": {"corpus_sha256": after}})
    assert st["ok"], st
    assert not d2_plan.screen_state(layout, {"toolcall_c1": {"corpus_sha256": before}})["ok"]
    assert hs.main(["check", "--corpus", str(corpus)]) == 0
    # idempotent; --force re-screens the screened corpus (0 hits, same bytes) and carries the drop forward
    receipt_sha = tc.file_sha256(w["receipt"])
    assert hs.main(w["args"]) == 0 and tc.file_sha256(w["receipt"]) == receipt_sha and tc.file_sha256(corpus) == after
    assert hs.main(w["args"] + ["--force"]) == 0
    again = json.loads(w["receipt"].read_text(encoding="utf-8"))
    assert again["corpus_sha256"] == after and again["hits_in_export"] == 0 and again["dropped"]["prior"]["rows"] == 4
    assert len(json.loads((backup / w["dropped"].name).read_text(encoding="utf-8"))["dropped"]) == 4  # the reasons survive
    # a changed corpus no longer checks
    with open(corpus, "ab") as handle:
        handle.write(b"\0")
    assert hs.main(["check", "--corpus", str(corpus)]) == 10


def test_clean_export_is_left_untouched(tmp_path, monkeypatch):
    w = build_world(tmp_path, monkeypatch, plant=False)
    before = tc.file_sha256(w["corpus"])
    assert hs.main(w["args"]) == 0
    receipt = json.loads(w["receipt"].read_text(encoding="utf-8"))
    assert tc.file_sha256(w["corpus"]) == before == receipt["corpus_sha256"] and receipt["rewrite"] is None
    assert json.loads(w["dropped"].read_text(encoding="utf-8"))["dropped"] == [] and receipt["hits_in_export"] == 0
    assert not (tmp_path / "unscreened").exists()


def _refused(w, extra=()) -> None:
    before = tc.file_sha256(w["corpus"])
    assert hs.main(w["args"] + list(extra)) == 2
    assert tc.file_sha256(w["corpus"]) == before and not w["receipt"].exists()


def test_refusals_change_nothing(tmp_path, monkeypatch):
    # drop fraction above the cap: the would-drop list is written (applied false), nothing else
    w = build_world(tmp_path / "a", monkeypatch)
    _refused(w)
    would = json.loads(w["dropped"].read_text(encoding="utf-8"))
    assert would["applied"] is False and len(would["dropped"]) == 4
    # the decisions file is not the pinned one
    w = build_world(tmp_path / "b", monkeypatch)
    monkeypatch.setattr(hs, "DECISIONS_SHA", "0" * 64)
    _refused(w, ["--max-drop-fraction", "0.2"])
    # the sidecar names another corpus
    w = build_world(tmp_path / "c", monkeypatch)
    side_path = w["corpus"].with_suffix(".rows.json")
    side = json.loads(side_path.read_text(encoding="utf-8"))
    side["corpus"]["sha256"] = "0" * 64
    tc.write_json_atomic(side_path, tc.seal({k: v for k, v in side.items() if k != "receipt_sha256"}, "receipt_sha256"))
    _refused(w, ["--max-drop-fraction", "0.2"])
    # a run whose conversations changed after its seal
    w = build_world(tmp_path / "d", monkeypatch)
    with open(w["run2"] / "conversations.jsonl", "a", encoding="utf-8") as handle:
        handle.write("\n")
    _refused(w, ["--max-drop-fraction", "0.2"])
    # a corpus row whose length is not its record's
    w = build_world(tmp_path / "e", monkeypatch)
    from safetensors.numpy import load_file
    t = load_file(str(w["corpus"]))
    t["lengths"][3] += 1
    save_file(t, str(w["corpus"]))
    side = json.loads(w["corpus"].with_suffix(".rows.json").read_text(encoding="utf-8"))
    side["corpus"]["sha256"] = tc.file_sha256(w["corpus"])
    tc.write_json_atomic(w["corpus"].with_suffix(".rows.json"), tc.seal({k: v for k, v in side.items() if k != "receipt_sha256"}, "receipt_sha256"))
    _refused(w, ["--max-drop-fraction", "0.2"])
    # a decode that is not the held-out text (fidelity)
    w = build_world(tmp_path / "f", monkeypatch)
    monkeypatch.setattr(hs, "pack_decoder", lambda parent_dir: (lambda rows: [" ".join(f"garbage{i}" for i in row) for row in rows]))
    _refused(w, ["--max-drop-fraction", "0.2"])
    # a held-out source with nothing long enough to probe: the screen is not provably live
    w = build_world(tmp_path / "g", monkeypatch)
    tasks_path = Path(w["args"][w["args"].index("--eval-tasks") + 1])
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]
    for t in tasks:
        t["input"]["messages"][0]["content"] = "tiny"
    tc.write_jsonl(tasks_path, tasks)
    monkeypatch.setattr(hs, "EVAL_TASKS_SHA", tc.file_sha256(tasks_path))
    _refused(w, ["--max-drop-fraction", "0.2"])
    # the 274-task file is not the pinned one
    w = build_world(tmp_path / "h", monkeypatch)
    monkeypatch.setattr(hs, "EVAL_TASKS_SHA", "0" * 64)
    _refused(w, ["--max-drop-fraction", "0.2"])


@needs_parent
def test_real_export_round_trip(tmp_path, monkeypatch):
    """The real toolcall_selfgen export (parent template) of rows with a planted held-out 13-gram -> screen -> the
    rewritten corpus loads through qtext_teacher_cache exactly as post_d1 C1 reads it."""

    import qtext_teacher_cache as qt

    w = build_world(tmp_path, monkeypatch)
    renderer = tc.Renderer(PARENT)
    for run in (w["run1"], w["run2"]):
        manifest, rows = tc.sealed_conversations(run)
        renders = []
        for r in rows:
            r.pop("render", None)
            x = renderer.supervised(r["messages"], r["tools"])
            renders.append((len(x["ids"]), int(sum(x["target_mask"])), tc.text_sha256(x["text"])))
        seal_run(run, rows, renders)
    w["corpus"].unlink()
    assert tc.main(["export", "--out-dir", str(w["run1"]), "--out-dir", str(w["run2"]), "--dest", str(w["corpus"]), "--parent-dir", str(PARENT)]) == 0
    assert hs.main(w["args"] + ["--max-drop-fraction", "0.2"]) == 0
    side = json.loads(w["corpus"].with_suffix(".rows.json").read_text(encoding="utf-8"))
    loaded = qt.load_text_corpus(w["corpus"])
    assert loaded["rows"] == side["rows"] == 26 and int(loaded["store_mask"].sum()) == side["supervised_tokens"]
    assert hs.main(["check", "--corpus", str(w["corpus"])]) == 0
