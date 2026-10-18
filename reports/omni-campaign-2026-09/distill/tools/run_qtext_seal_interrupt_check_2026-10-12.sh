#!/usr/bin/env bash
# qtext lane, tool 1.0.3: the reviewer's repro on the REAL 27B smoke cache, seal path only.
# Step 5 of run_qtext_cpu_v103 could not resume the scratch copy on the CPU: its cache_config
# locks the fla GDN kernel (built on a P40) and a CPU session resolves torch, so build-text
# refused ("locks different semantics for ['linear_attention_kernel']") -- the right refusal.
# Here seal_manifest runs directly on that scratch copy (real shards, the real complete
# session record of the P40 run), with one planted session record in the exact shape a
# SIGKILL after the "before" digest leaves (the real "before" digest, nothing else):
#   A  planted BEFORE the real session (killed, then resumed to completion): 1.0.2 control
#      and 1.0.3; 1.0.3 must seal and list it as interrupted, vouched by the real session.
#   B  planted AFTER the real session (killed, not yet resumed): 1.0.3 must not seal (exit 1)
#      and must not call it a teacher change.
#   C  planted BEFORE, but its "before" tree altered: 1.0.3 must refuse (exit 2).
# CPU only, no model load, no teacher read.  The 1.0.2 control copy is $W/v102-control.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
W=$OC/distill/parity/qtext-2026-10-12-cpu-v103
C=$(ls -d $W/q27-interrupt/q38-27bu-smoke/*/)
CORPUS=$OC/distill/parity/qtext-2026-10-12/q27-text-corpus.safetensors
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src QTEXT_TPS_PROBE=$OC/distill/tps_probe.py
sha256sum $T/qtext_teacher_cache.py $W/v102-control/qtext_teacher_cache.py
nice -n 15 $PY - "$C" "$CORPUS" "$T" "$W/v102-control" <<'EOF'
import importlib.util, json, sys
from pathlib import Path

cache, corpus_path, new_dir, old_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]


def load(name, folder):
    spec = importlib.util.spec_from_file_location(name, str(Path(folder) / "qtext_teacher_cache.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new, old = load("qc103", new_dir), load("qc102", old_dir)
print("versions", old.TOOL_VERSION, new.TOOL_VERSION)
corpus = new.load_text_corpus(corpus_path)
assert cache.name == corpus["sha256"], (cache.name, corpus["sha256"])
sessions = cache / "sessions"
for p in sessions.glob("session-*-999999.json"):
    p.unlink()
(real_path,) = sorted(sessions.glob("session-*.json"))
real = json.loads(real_path.read_text())
print("real session", real["session"], "before", real["teacher_digest_before"]["tree_sha256"][:16],
      "after", real["teacher_digest_after"]["tree_sha256"][:16], "unchanged", real["teacher_unchanged_in_session"])


def plant(stamp, tree=None):
    for p in sessions.glob("session-*-999999.json"):
        p.unlink()
    (cache / "manifest.json").unlink(missing_ok=True)
    before = dict(real["teacher_digest_before"])
    if tree:
        before["tree_sha256"] = tree
    sid = f"{stamp}-999999"
    record = {"session": sid, "started_utc": "planted", "tool": real["tool"], "engine": real["engine"],
              "lock_taken_over_from": None, "devices_at_start": [], "teacher_digest_before": before}
    (sessions / f"session-{sid}.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return sid


sid = plant("20261013T020500Z")
print("A control 1.0.2 ->", old.seal_manifest(cache, corpus, Path("/home/cereal/models/Qwen3.8-27B-Uncensored"), 0.0))
(cache / "manifest.json").unlink(missing_ok=True)
code = new.seal_manifest(cache, corpus, Path("/home/cereal/models/Qwen3.8-27B-Uncensored"), 0.0)
print("A 1.0.3 ->", code)
m = json.loads((cache / "manifest.json").read_text())
print("A manifest sessions", [s["session"] for s in m["sessions"]],
      "interrupted", [(s["session"], s["shards_written"], s["vouched_by"]) for s in m["sessions_interrupted"]],
      "positions", m["positions_total"], "all_finite", m["all_finite"],
      "teacher", m["teacher"]["tree_sha256_before"][:16], m["teacher"]["tree_sha256_after"][:16],
      m["teacher"]["equal_before_and_after_every_session"], m["teacher"]["interrupted_sessions_vouched_for"])
print("A verify ->", new.verify_cache(cache))
import shutil
shutil.copy2(cache / "manifest.json", cache.parent / "manifest-caseA-v103.json")
plant("20261013T030000Z")
print("B 1.0.3 ->", new.seal_manifest(cache, corpus, Path("/home/cereal/models/Qwen3.8-27B-Uncensored"), 0.0),
      "manifest exists:", (cache / "manifest.json").exists())
plant("20261013T020500Z", tree="f" * 64)
print("C 1.0.3 ->", new.seal_manifest(cache, corpus, Path("/home/cereal/models/Qwen3.8-27B-Uncensored"), 0.0),
      "manifest exists:", (cache / "manifest.json").exists())
EOF
echo "exit $?"
