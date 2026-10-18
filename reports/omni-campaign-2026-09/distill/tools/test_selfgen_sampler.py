"""Tests for ``selfgen_sampler.py``.

The sampler's mechanics run against an in-process stand-in for llama-server's
``/completion`` (deterministic in (seed, prompt ids), so a killed-and-resumed run must
reproduce a single run byte for byte).  Everything else is REAL: the parent's pinned
tokenizer.json and generation_config.json, the corpus lane's sealed self-gen prompts,
the export lane's ``llama_probe.directory_digest``, and the rescoring hand-off is loaded
through its actual consumer, ``qtext_teacher_cache.load_text_corpus``.  The stand-in is
only the network peer; the GPU path is proven by the sanity/probe receipts on the server.

  test_version_is_pinned
  test_prompt_ids_reproduce_the_corpus_counts           real prompts.jsonl, first 300
  test_prompt_manifest_sha_is_enforced
  test_full_run_seals_shards_and_a_manifest
  test_kill_and_resume_reproduces_the_single_run        tokens.u32 byte-identical
  test_token_target_cuts_at_a_sampled_prefix
  test_torn_journal_line_is_resampled                   torn tail cut; a 2nd resume keeps every row
  test_failed_append_cuts_back_and_closes_the_journal
  test_server_only_eog_stop_is_truncated_and_listed     1.2.0: <|file_sep|> etc. stops
  test_journal_from_1_1_0_relabels_when_sealed
  test_eog_probe_matches_the_server_set
  test_resume_with_changed_semantics_is_refused
  test_out_of_vocab_id_is_a_functional_failure
  test_dead_server_stops_resumably_and_resume_completes
  test_server_joins_from_the_servers_file               and a wrong-GGUF server never joins
  test_verify_detects_a_tampered_shard
  test_export_rescore_loads_through_the_qtext_consumer
  test_eos_record_keeps_the_eog_id_and_text_excludes_it
  test_scratch_delete_guard
  test_export_check_deletes_only_declared_scratch_and_catches_a_moved_parent
  test_card_rules
  test_tensor_map_on_the_real_dryrun_log               skipped until the log exists
  test_server_only_eog_pieces_match_the_pinned_tokenizer

Server:
  cd $OC/distill/tools && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OC/distill/tools \\
      $PY -m pytest -o addopts="" -p no:cacheprovider test_selfgen_sampler.py -q
"""
from __future__ import annotations

import json
import os
import random
import shutil
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import selfgen_sampler as sg  # noqa: E402

PARENT = Path(os.environ.get("SELFGEN_PARENT", str(sg.DEFAULT_PARENT)))
PROMPTS = Path(os.environ.get("SELFGEN_PROMPTS", str(sg.DEFAULT_PROMPTS)))
PROMPTS_MANIFEST = Path(os.environ.get("SELFGEN_PROMPTS_MANIFEST", str(sg.DEFAULT_PROMPTS_MANIFEST)))
EXPORT_TOOLS = Path(os.environ.get("SELFGEN_EXPORT_TOOLS", str(sg.DEFAULT_EXPORT_TOOLS)))
DRYRUN_LOG = sg.OC / "distill" / "selfgen" / "logs" / "tensor_map_dryrun.log"

if not (PARENT / "tokenizer.json").is_file():
    pytest.skip(f"the parent's tokenizer is not at {PARENT}", allow_module_level=True)

N_PROMPTS = 40
MAX_NEW = 48


# ------------------------------------------------------------------ the network peer
class FakeLlama:
    """Deterministic /completion: the sample is a function of (seed, prompt ids) only."""

    def __init__(self, tok: sg.ParentTokenizer, model_path: str) -> None:
        self.tok = tok
        self.model_path = model_path
        self.pool = sorted(set(tok.encode(
            "The quick brown fox jumps over the lazy dog. In 2026 the model answered every question "
            "with a careful, numbered explanation; see section 3 for details and examples.")))
        self.mode = "ok"
        #: what the stand-in ends generation on (llama.cpp b10752: HF EOS + the FIM trio)
        self.server_eog = set(sg.EOG_IDS) | set(sg.SERVER_ONLY_EOG_IDS)
        #: llama-server leaves a server-only stop piece out of `content` (True) or not
        self.omit_stop_piece = True
        self.delay = 0.0
        self.fail_after = None
        self.calls = 0
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401 - silence
                pass

            def _send(self, code: int, payload) -> None:
                blob = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)

            def do_GET(self):
                if self.path == "/health":
                    return self._send(200, {"status": "ok"})
                if self.path == "/props":
                    return self._send(200, {"model_path": fake.model_path, "total_slots": 4, "build_info": "fake",
                                            "default_generation_settings": {"n_ctx": 6144}})
                return self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    return self._send(200, {"tokens": fake.tok.encode(body["content"])})
                with fake.lock:
                    fake.calls += 1
                    failing = fake.fail_after is not None and fake.calls > fake.fail_after
                if failing:
                    return self._send(500, {"error": "injected"})
                return self._send(200, fake.sample(body))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def forced(self, body):
        """logit_bias +100 on one id: it repeats until it ends generation or hits the limit."""
        token_id = int(body["logit_bias"][0][0])
        if token_id in self.server_eog:
            return {"content": "", "tokens": [token_id], "stop_type": "eos"}
        ids = [token_id] * body["n_predict"]
        return {"content": self.tok.decode(ids), "tokens": ids, "stop_type": "limit"}

    def sample(self, body):
        if self.delay:
            time.sleep(self.delay)
        if body.get("logit_bias"):
            return self.forced(body)
        prompt = body["prompt"]
        n_predict = body["n_predict"]
        rng = random.Random(int(body["seed"]) * 1_000_003 + sum(prompt) * 7 + len(prompt))
        length = rng.randint(1, n_predict)
        ids = [rng.choice(self.pool) for _ in range(length)]
        stop = "limit"
        if length < n_predict:
            ids[-1] = sg.EOG_IDS[0]
            if self.mode == "server_only_eog" and length % 2:
                ids[-1] = sg.SERVER_ONLY_EOG_IDS[length % 3]
            stop = "eos"
        if self.mode == "bad_id":
            ids[0] = sg.TEXT_VOCAB_ROWS + 5
        drop_last = ids[-1] in sg.EOG_IDS or (stop == "eos" and self.omit_stop_piece)
        content_ids = ids[:-1] if drop_last else ids
        return {"content": self.tok.decode(content_ids), "tokens": ids, "stop_type": stop,
                "tokens_evaluated": len(prompt),
                "timings": {"prompt_n": len(prompt), "prompt_ms": 1.0, "predicted_n": len(ids),
                            "predicted_ms": 1.0, "predicted_per_second": 1000.0}}

    def close(self) -> None:
        self.httpd.shutdown()


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def tok() -> sg.ParentTokenizer:
    return sg.ParentTokenizer(PARENT)


@pytest.fixture(scope="module")
def prompt_files(tmp_path_factory, tok) -> dict:
    """The first N real self-gen prompts with max_new_tokens cut for speed, + a manifest."""
    work = tmp_path_factory.mktemp("prompts")
    rows = []
    if PROMPTS.is_file():
        with open(PROMPTS, encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
                if len(rows) >= N_PROMPTS:
                    break
    else:  # pragma: no cover - the server always has the corpus
        for i in range(N_PROMPTS):
            text = f"<|im_start|>user\nQuestion {i}: what is {i} + {i}?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            rows.append({"rid": f"r{i}", "route": "math", "prompt_text": text, "prompt_tokens": len(tok.encode(text))})
    for row in rows:
        row["max_new_tokens"] = MAX_NEW
    path = work / "prompts.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = work / "manifest.json"
    manifest.write_text(json.dumps({"file": {"sha256": sg.file_sha256(path)}, "prompts": len(rows),
                                    "manifest_sha256": "fixture"}))
    return {"prompts": path, "manifest": manifest, "rows": rows}


@pytest.fixture()
def fake(tok, tmp_path):
    gguf = tmp_path / "fixture.Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF fixture")
    server = FakeLlama(tok, str(gguf))
    yield server
    server.close()


def run_args(prompt_files, fake, out_dir, *extra):
    return ["run", "--server", fake.url, "--slots-per-server", "4", "--out-dir", str(out_dir),
            "--prompts", str(prompt_files["prompts"]), "--prompts-manifest", str(prompt_files["manifest"]),
            "--parent-dir", str(PARENT), "--gguf", fake.model_path, "--rows-per-shard", "8",
            "--progress-every", "0.5", "--retry-backoff", "0.05", *extra]


def shard_dirs(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.glob("shard-*") if p.is_dir())


# ------------------------------------------------------------------ tests
def test_version_is_pinned(capsys):
    assert sg.TOOL_VERSION == "1.2.0"
    with pytest.raises(SystemExit):
        sg.main(["--version"])
    assert "selfgen_sampler 1.2.0" in capsys.readouterr().out


@pytest.mark.skipif(not PROMPTS.is_file(), reason="the corpus lane's prompts.jsonl is not here")
def test_prompt_ids_reproduce_the_corpus_counts(tok):
    loaded = sg.load_prompts(PROMPTS, PROMPTS_MANIFEST)
    assert len(loaded["prompts"]) == loaded["manifest"]["prompts"] == 2532
    for prompt in loaded["prompts"][:300]:
        ids = sg.encode_prompt(tok, prompt)
        assert ids[-1] != sg.EOG_IDS[0] and len(ids) == prompt["prompt_tokens"]
        assert prompt["prompt_text"].endswith("<think>\n\n</think>\n\n")


def test_prompt_manifest_sha_is_enforced(prompt_files, tmp_path):
    copy = tmp_path / "prompts.jsonl"
    data = prompt_files["prompts"].read_bytes()
    copy.write_bytes(data.replace(b"\"route\"", b"\"route\" ", 1))
    with pytest.raises(sg.SamplerError, match="manifest"):
        sg.load_prompts(copy, prompt_files["manifest"])
    moved = dict(prompt_files["rows"][0], prompt_tokens=prompt_files["rows"][0]["prompt_tokens"] + 1)
    with pytest.raises(sg.SamplerError, match="tokenizes to"):
        sg.encode_prompt(sg.ParentTokenizer(PARENT), moved)


def test_full_run_seals_shards_and_a_manifest(prompt_files, fake, tmp_path, tok):
    out = tmp_path / "run"
    assert sg.main(run_args(prompt_files, fake, out)) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert sg.check_seal(manifest, "manifest_sha256")
    assert manifest["cutoff_index"] == N_PROMPTS and manifest["totals"]["rows"] == N_PROMPTS
    assert [s["name"] for s in manifest["shards"]] == [f"shard-{i:05d}" for i in range(5)]
    journal = sg.JournalState(out / "journal.jsonl")
    assert manifest["totals"]["response_tokens"] == sum(r["response_tokens"] for r in journal.records.values())
    journal.close()
    assert sg.verify_run(out) == 0
    # tokens.u32 = prompt ids + response ids per row, and the response decodes to the stored text
    for shard_dir in shard_dirs(out):
        loaded = sg.read_shard(shard_dir)
        for row, sample in zip(loaded["record"]["rows"], loaded["samples"]):
            ids = loaded["tokens"][row["offset"]: row["offset"] + row["prompt_len"] + row["response_len"]]
            prompt = prompt_files["rows"][row["index"]]
            assert list(ids[: row["prompt_len"]]) == tok.encode(prompt["prompt_text"])
            body = list(ids[row["prompt_len"]:])
            if sample["eog_terminated"]:
                assert body[-1] in sg.EOG_IDS and sample["stop_type"] == "eos"
                body = body[:-1]
            assert tok.decode(body) == sample["text"] and sample["text_matches_server"]
    assert manifest["anomalies"] == {}
    assert sum(manifest["stop_types"].values()) == N_PROMPTS


def test_kill_and_resume_reproduces_the_single_run(prompt_files, fake, tmp_path):
    single = tmp_path / "single"
    assert sg.main(run_args(prompt_files, fake, single)) == 0
    split = tmp_path / "split"
    assert sg.main(run_args(prompt_files, fake, split, "--stop-after", "13")) == 2
    assert not (split / "manifest.json").exists()
    assert 1 <= len(shard_dirs(split)) < 5
    assert sg.main(run_args(prompt_files, fake, split)) == 0
    a = {p.name: sg.file_sha256(p / "tokens.u32") for p in shard_dirs(single)}
    b = {p.name: sg.file_sha256(p / "tokens.u32") for p in shard_dirs(split)}
    assert a == b and len(a) == 5
    ma = json.loads((single / "manifest.json").read_text())
    mb = json.loads((split / "manifest.json").read_text())
    assert ma["totals"] == mb["totals"] and len(mb["sessions"]) == 2


def test_token_target_cuts_at_a_sampled_prefix(prompt_files, fake, tmp_path):
    out = tmp_path / "cut"
    assert sg.main(run_args(prompt_files, fake, out, "--token-target", "300")) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    cutoff = manifest["cutoff_index"]
    assert 0 < cutoff < N_PROMPTS
    assert manifest["totals"]["rows"] == cutoff and manifest["totals"]["response_tokens"] >= 300
    indexes = [r["index"] for d in shard_dirs(out) for r in sg.read_shard(d)["record"]["rows"]]
    assert indexes == list(range(cutoff))
    assert sg.verify_run(out) == 0


def test_torn_journal_line_is_resampled(prompt_files, fake, tmp_path):
    """Reviewer finding 2: the torn tail must be CUT before append, else the next record is
    glued onto it and lost on the replay after.  Resumes twice: session 2 seals shards over
    the rows it appended after the cut; session 3 must still find every sealed row."""
    out = tmp_path / "torn"
    journal = out / "journal.jsonl"
    assert sg.main(run_args(prompt_files, fake, out, "--stop-after", "6")) == 2
    whole = journal.read_bytes()
    # torn inside a multi-byte character: 1.1.0 crashed here (UnicodeDecodeError on replay)
    fragment = '{"index": 39, "rid": "half-writ \u00e9'.encode("utf-8")[:-1]
    with open(journal, "ab") as handle:
        handle.write(fragment)
    assert sg.main(run_args(prompt_files, fake, out, "--stop-after", "20")) == 2
    data = journal.read_bytes()
    assert data.startswith(whole) and data.endswith(b"\n") and fragment not in data
    assert all(json.loads(line) for line in data.splitlines())   # every line whole
    sessions = [json.loads(p.read_text()) for p in sorted((out / "sessions").glob("*.json"))]
    assert sessions[-1]["resumed_with"]["dropped_journal_lines"] == 1
    assert sessions[-1]["resumed_with"]["dropped_journal_tail_bytes"] == len(fragment)
    assert len(shard_dirs(out)) >= 2   # session 2 sealed rows it appended after the cut
    assert sg.main(run_args(prompt_files, fake, out)) == 0   # session 3: no "sealed but rows ... not in the journal"
    sessions = [json.loads(p.read_text()) for p in sorted((out / "sessions").glob("*.json"))]
    assert sessions[-1]["resumed_with"]["dropped_journal_lines"] == 0
    assert json.loads((out / "manifest.json").read_text())["totals"]["rows"] == N_PROMPTS
    assert sg.verify_run(out) == 0


def test_failed_append_cuts_back_and_closes_the_journal(tmp_path, monkeypatch):
    path = tmp_path / "journal.jsonl"
    journal = sg.JournalState(path)
    assert journal.append({"index": 0, "response_tokens": 3})
    good = path.read_bytes()
    real_write = os.write

    def half_then_enospc(fd, data):
        real_write(fd, bytes(data[: len(data) // 2]))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sg.os, "write", half_then_enospc)
    with pytest.raises(sg.SamplerError, match="No space left"):
        journal.append({"index": 1, "response_tokens": 4})
    monkeypatch.setattr(sg.os, "write", real_write)
    assert path.read_bytes() == good                 # the half line was cut back
    with pytest.raises(sg.SamplerError, match="closed for this session"):
        journal.append({"index": 2, "response_tokens": 5})   # never glued after a failure
    assert path.read_bytes() == good and sorted(journal.records) == [0]
    journal.close()
    again = sg.JournalState(path)
    assert sorted(again.records) == [0] and again.dropped_lines == 0
    again.close()


def test_resume_with_changed_semantics_is_refused(prompt_files, fake, tmp_path):
    out = tmp_path / "locked"
    assert sg.main(run_args(prompt_files, fake, out, "--stop-after", "4")) == 2
    args = run_args(prompt_files, fake, out)
    args[args.index("--rows-per-shard") + 1] = "16"
    assert sg.main(args) == 1
    assert sg.main(run_args(prompt_files, fake, out, "--base-seed", "7")) == 1
    assert sg.main(run_args(prompt_files, fake, out)) == 0


def test_out_of_vocab_id_is_a_functional_failure(prompt_files, fake, tmp_path):
    fake.mode = "bad_id"
    out = tmp_path / "bad"
    assert sg.main(run_args(prompt_files, fake, out)) == 1
    progress = json.loads((out / "progress.json").read_text())
    assert "outside [0, 248320)" in progress["failure"]
    assert not (out / "manifest.json").exists()


def test_dead_server_stops_resumably_and_resume_completes(prompt_files, fake, tmp_path):
    fake.fail_after = 10
    out = tmp_path / "dead"
    assert sg.main(run_args(prompt_files, fake, out, "--retries", "1", "--max-consecutive-errors", "3")) == 2
    progress = json.loads((out / "progress.json").read_text())
    assert progress["servers"][fake.url]["dead"] and progress["failure"] is None
    fake.fail_after = None
    assert sg.main(run_args(prompt_files, fake, out)) == 0
    assert sg.verify_run(out) == 0


def test_server_joins_from_the_servers_file(prompt_files, fake, tok, tmp_path):
    fake.delay = 0.2
    second = FakeLlama(tok, fake.model_path)
    wrong = FakeLlama(tok, str(tmp_path / "some-other.Q4_K_M.gguf"))
    servers = tmp_path / "servers.txt"
    servers.write_text(fake.url + "\n")
    out = tmp_path / "join"

    def add_later():
        journal = out / "journal.jsonl"
        while not journal.exists() or journal.read_text().count("\n") < 4:
            time.sleep(0.02)
        servers.write_text("\n".join([fake.url, "# a comment", wrong.url, second.url]) + "\n")

    threading.Thread(target=add_later, daemon=True).start()
    args = run_args(prompt_files, fake, out, "--progress-every", "0.2")
    at = args.index("--server")
    args[at: at + 2] = ["--servers-file", str(servers)]
    try:
        assert sg.main(args) == 0
    finally:
        second.close()
        wrong.close()
    progress = json.loads((out / "progress.json").read_text())
    assert set(progress["servers"]) == {fake.url, second.url}
    assert progress["servers"][second.url]["done"] > 0 and progress["servers"][fake.url]["done"] > 0
    assert sg.verify_run(out) == 0


def test_verify_detects_a_tampered_shard(prompt_files, fake, tmp_path):
    out = tmp_path / "tamper"
    assert sg.main(run_args(prompt_files, fake, out)) == 0
    target = shard_dirs(out)[2] / "tokens.u32"
    blob = bytearray(target.read_bytes())
    blob[-2] ^= 0x01
    target.write_bytes(bytes(blob))
    assert sg.verify_run(out) == 1


def test_export_rescore_loads_through_the_qtext_consumer(prompt_files, fake, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import qtext_teacher_cache as qtc

    out = tmp_path / "rescore"
    assert sg.main(run_args(prompt_files, fake, out)) == 0
    dest = tmp_path / "selfgen_rescore.safetensors"
    assert sg.main(["export-rescore", "--out-dir", str(out), "--dest", str(dest)]) == 0
    corpus = qtc.load_text_corpus(dest)
    sidecar = json.loads(dest.with_suffix(".rows.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert corpus["rows"] == N_PROMPTS == sidecar["rows"]
    assert corpus["tokens"] == manifest["totals"]["prompt_tokens"] + manifest["totals"]["response_tokens"]
    # one stored distribution per sampled token: positions [prompt_len - 1, length - 2]
    assert int(corpus["store_mask"].sum()) == manifest["totals"]["response_tokens"] == sidecar["stored_positions"]
    first = sg.read_shard(shard_dirs(out)[0])
    row = first["record"]["rows"][0]
    mask = corpus["store_mask"][0]
    assert not bool(mask[row["prompt_len"] - 2]) and bool(mask[row["prompt_len"] - 1])
    length = row["prompt_len"] + row["response_len"]
    assert bool(mask[length - 2]) and not bool(mask[length - 1])
    assert corpus["ids"][0, :length].tolist() == list(first["tokens"][:length])


def test_eos_record_keeps_the_eog_id_and_text_excludes_it(tok, prompt_files):
    prompt = prompt_files["rows"][0]
    ids = tok.encode(prompt["prompt_text"])
    body_ids = tok.encode(" Four.")
    body = {"tokens": body_ids + [248046], "content": " Four.", "stop_type": "eos", "timings": {}}
    record = sg.build_record(0, prompt, ids, 1, body, 0.1, "x", 1, tok)
    assert record["eog_terminated"] and record["eog_id"] == 248046 and record["text"] == " Four."
    assert record["response_tokens"] == len(body_ids) + 1 and not record["truncated"]
    assert sg.record_anomalies(record) == []
    assert record["end"] == "eog" and record["server_only_eog_id"] is None
    limit = sg.build_record(0, prompt, ids, 1, {"tokens": body_ids, "content": " Four.", "stop_type": "limit"},
                            0.1, "x", 1, tok)
    assert limit["truncated"] and not limit["eog_terminated"] and limit["end"] == "limit"
    odd = sg.build_record(0, prompt, ids, 1, {"tokens": body_ids, "content": "other", "stop_type": "eos"}, 0.1, "x", 1, tok)
    assert sg.record_anomalies(odd) == ["eos_stop_without_eog_id", "text_differs_from_server"]
    assert odd["server_content"] == "other" and odd["truncated"] and odd["end"] == "eos_without_eog"
    # reviewer finding 1: the server stopped on <|file_sep|>; HF would have sampled past it
    for content in (" Four.", " Four.<|file_sep|>"):   # piece omitted or kept by the server
        cut = sg.build_record(0, prompt, ids, 1, {"tokens": body_ids + [248065], "content": content, "stop_type": "eos"},
                              0.1, "x", 1, tok)
        assert cut["end"] == "server_only_eog" and cut["server_only_eog_id"] == 248065
        assert cut["truncated"] and not cut["eog_terminated"] and cut["eog_id"] is None
        assert cut["text"] == " Four.<|file_sep|>" and cut["text_matches_server"]
        assert sg.record_anomalies(cut) == ["server_only_eog_stop"]
        assert sg.relabel(cut) == cut


def test_scratch_delete_guard(tmp_path):
    lexar = Path("/mnt/lexar/omni-campaign/scratch")
    ok, _ = sg.scratch_delete_allowed(lexar / "selfgen-x" / "a.bf16.gguf", [lexar])
    assert ok
    assert not sg.scratch_delete_allowed(lexar / "a.safetensors", [lexar])[0]
    assert not sg.scratch_delete_allowed(Path("/mnt/lexar/omni-campaign/k3/a.gguf"), [lexar])[0]
    assert not sg.scratch_delete_allowed(Path("/mnt/lexar/omni-campaign/scratchy/a.gguf"), [lexar])[0]
    assert not sg.scratch_delete_allowed(Path("/mnt/ai_data/x/a.gguf"), [Path("/mnt/ai_data/x")])[0]
    assert not sg.scratch_delete_allowed(Path("/home/cereal/models/gguf/a.gguf"), [Path("/home/cereal/models")])[0]
    assert sg.scratch_delete_allowed(tmp_path / "s" / "a.gguf", [tmp_path / "s"])[0]
    assert not sg.scratch_delete_allowed(tmp_path / "s", [tmp_path / "s"])[0]


def test_export_check_deletes_only_declared_scratch_and_catches_a_moved_parent(tmp_path):
    if not (EXPORT_TOOLS / "llama_probe.py").is_file():
        pytest.skip("the export lane's llama_probe.py is not here")
    probe = sg.llama_probe(EXPORT_TOOLS)
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "config.json").write_text("{}")
    (parent / "model.safetensors").write_bytes(b"weights")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    full = scratch / "p.bf16.gguf"
    full.write_bytes(b"full")
    q4 = scratch / "p.Q4_K_M.gguf"
    q4.write_bytes(b"quant")
    receipt = {"outtype": "bf16", "stages": [{"stage": "convert", "rc": 0}, {"stage": "quantize:Q4_K_M", "rc": 0}],
               "source_directory_digest": probe.directory_digest(str(parent)),
               "artifacts": {"bf16": {"path": str(full), "bytes": 4, "sha256": sg.file_sha256(full)},
                             "Q4_K_M": {"path": str(q4), "bytes": 5, "sha256": sg.file_sha256(q4)}}}
    rpath = scratch / "export-receipt.json"
    rpath.write_text(json.dumps(receipt))
    base = ["export-check", "--export-receipt", str(rpath), "--parent-dir", str(parent),
            "--export-tools", str(EXPORT_TOOLS), "--delete-full"]
    # outside every scratch root: refused, nothing deleted
    assert sg.main(base + ["--scratch-root", str(tmp_path / "elsewhere"), "--out", str(tmp_path / "r0.json")]) == 1
    assert full.exists()
    assert sg.main(base + ["--scratch-root", str(scratch), "--out", str(tmp_path / "r1.json")]) == 0
    out = json.loads((tmp_path / "r1.json").read_text())
    assert out["parent_unchanged"] and out["full_gguf"]["deleted"] is True and not full.exists() and q4.exists()
    assert out["q4"]["sha256"] == sg.file_sha256(q4) and sg.check_seal(out, "receipt_sha256")
    (parent / "model.safetensors").write_bytes(b"weightz")
    assert sg.main(base + ["--scratch-root", str(scratch), "--out", str(tmp_path / "r2.json")]) == 1
    q4.write_bytes(b"quanT")
    (parent / "model.safetensors").write_bytes(b"weights")
    assert sg.main(base + ["--scratch-root", str(scratch), "--out", str(tmp_path / "r3.json")]) == 1


def test_card_rules():
    text = ("0, GPU-4ead, NVIDIA GeForce RTX 3090, 24, 24576\n1, GPU-c95f, Tesla P40, 5, 24576\n"
            "2, GPU-a92b, Tesla P40, 5319, 24576\n5, GPU-p405, Tesla P40, 5, 24576\n")
    table = sg.parse_gpu_table(text)
    assert sg.check_card(table, 1, sg.DEFAULT_ALLOWED_GPUS)["uuid"] == "GPU-c95f"
    with pytest.raises(sg.SamplerError, match="not in this lane"):
        sg.check_card(table, 0, sg.DEFAULT_ALLOWED_GPUS)
    with pytest.raises(sg.SamplerError, match="not in this lane"):
        sg.check_card(table, 5, sg.DEFAULT_ALLOWED_GPUS)
    with pytest.raises(sg.SamplerError, match="MiB in use"):
        sg.check_card(table, 2, sg.DEFAULT_ALLOWED_GPUS)
    with pytest.raises(sg.SamplerError, match="not present"):
        sg.check_card(table, 3, sg.DEFAULT_ALLOWED_GPUS)


def as_1_1_0(record: dict) -> dict:
    """The shape 1.1.0 journaled for the same sample (no end class; truncated = limit only;
    a server-only stop counted as a decode mismatch with server_content kept)."""
    old = {k: v for k, v in record.items() if k not in ("end", "server_only_eog_id")}
    old["truncated"] = record["stop_type"] == "limit" and not record["eog_terminated"]
    if record.get("end") == "server_only_eog" and record["text_matches_server"]:
        piece = sg.SERVER_ONLY_EOG_PIECES[record["server_only_eog_id"]]
        old["text_matches_server"] = False
        old["server_content"] = record["text"][: -len(piece)]
    return old


def test_server_only_eog_stop_is_truncated_and_listed(prompt_files, fake, tmp_path):
    pytest.importorskip("safetensors")
    fake.mode = "server_only_eog"
    out = tmp_path / "fimstop"
    assert sg.main(run_args(prompt_files, fake, out)) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    listed = manifest["server_only_eog_rows"]
    totals = manifest["totals"]
    assert listed and totals["server_only_eog"] == len(listed) == manifest["ends"]["server_only_eog"]
    assert {r["stop_id"] for r in listed} <= set(sg.SERVER_ONLY_EOG_IDS)
    assert totals["truncated"] == totals["rows"] - totals["eog_terminated"]
    assert manifest["anomalies"] == {"server_only_eog_stop": len(listed)}   # decode parity holds
    samples = {s["index"]: s for d in shard_dirs(out) for s in sg.read_shard(d)["samples"]}
    for row in listed:
        s = samples[row["index"]]
        assert s["truncated"] and not s["eog_terminated"] and s["response_ids"][-1] == row["stop_id"]
    dest = tmp_path / "fim.safetensors"
    assert sg.main(["export-rescore", "--out-dir", str(out), "--dest", str(dest)]) == 0
    from safetensors.numpy import load_file
    eog = load_file(str(dest))["eog_terminated"]
    sidecar = json.loads(dest.with_suffix(".rows.json").read_text())
    for n, index in enumerate(sidecar["row_index"]):
        assert bool(eog[n]) == samples[index]["eog_terminated"]
        assert sidecar["row_end"][n] == samples[index]["end"]
    assert sidecar["truncated_rows"] == totals["truncated"] and sidecar["eog_terminated_rows"] == totals["eog_terminated"]


def test_journal_from_1_1_0_relabels_when_sealed(prompt_files, fake, tmp_path):
    """The live run journaled with 1.1.0; resuming under 1.2.0 must seal those rows with
    the new labels and list them in the manifest (relabel works from the ids)."""
    fake.mode = "server_only_eog"
    ref = tmp_path / "ref"
    assert sg.main(run_args(prompt_files, fake, ref)) == 0
    out = tmp_path / "old"
    assert sg.main(run_args(prompt_files, fake, out, "--stop-after", "13")) == 2
    journal = out / "journal.jsonl"
    old_lines = [json.dumps(as_1_1_0(json.loads(line)), sort_keys=True, ensure_ascii=False)
                 for line in journal.read_text(encoding="utf-8").splitlines()]
    assert any('"server_content"' in line for line in old_lines)
    journal.write_text("\n".join(old_lines) + "\n", encoding="utf-8")
    for d in shard_dirs(out):          # as if 1.1.0 had sealed nothing yet
        shutil.rmtree(d)
    assert sg.main(run_args(prompt_files, fake, out)) == 0
    a = json.loads((ref / "manifest.json").read_text())
    b = json.loads((out / "manifest.json").read_text())
    assert a["server_only_eog_rows"] == b["server_only_eog_rows"] and a["totals"] == b["totals"]
    assert a["anomalies"] == b["anomalies"] and a["ends"] == b["ends"]
    for d in shard_dirs(out):
        for s in sg.read_shard(d)["samples"]:
            assert s == sg.relabel(s) and "end" in s


def test_eog_probe_matches_the_server_set(prompt_files, fake, tmp_path):
    base = ["eog-probe", "--server", fake.url, "--parent-dir", str(PARENT),
            "--prompts", str(prompt_files["prompts"]), "--prompts-manifest", str(prompt_files["manifest"])]
    assert sg.main(base + ["--out", str(tmp_path / "ok.json")]) == 0
    receipt = json.loads((tmp_path / "ok.json").read_text())
    assert receipt["ok"] and sg.check_seal(receipt, "receipt_sha256")
    assert receipt["servers"][fake.url]["server_eog_ids"] == sorted(sg.EOG_IDS + sg.SERVER_ONLY_EOG_IDS)
    fake.server_eog = set(sg.EOG_IDS)            # a server whose EOG set differs: refused
    assert sg.main(base + ["--out", str(tmp_path / "bad.json")]) == 1
    assert json.loads((tmp_path / "bad.json").read_text())["ok"] is False


def test_server_only_eog_pieces_match_the_pinned_tokenizer(tok):
    for token_id, piece in sg.SERVER_ONLY_EOG_PIECES.items():
        assert tok.piece(token_id) == piece
        assert tok.decode([token_id]) == piece           # HF keeps it as an ordinary piece
    assert set(sg.SERVER_ONLY_EOG_PIECES) == set(sg.SERVER_ONLY_EOG_IDS)
    gen = json.loads((PARENT / "generation_config.json").read_text())
    assert sorted(gen["eos_token_id"]) == sorted(sg.EOG_IDS)
    assert not set(gen["eos_token_id"]) & set(sg.SERVER_ONLY_EOG_IDS)


@pytest.mark.skipif(not DRYRUN_LOG.is_file(), reason="the converter dry-run log has not been written")
def test_tensor_map_on_the_real_dryrun_log():
    out = sg.tensor_map(DRYRUN_LOG, PARENT)
    assert out["gguf_tensor_count"] > 0
    assert {"token_embd.weight", "output.weight", "output_norm.weight"} <= set(out["gguf_non_block"])
    assert out["gguf_blocks"][0] == 0 and out["gguf_block_count"] >= 64
    assert "visual" in out["hf_tensor_classes"]
