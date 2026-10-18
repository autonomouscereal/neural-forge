#!/usr/bin/env bash
# qtext lane, tool 1.0.3 (review fixes: a hard-killed session is INTERRUPTED, not a teacher
# change; SIGTERM runs the "after" digest; every crash records its exit code and failure),
# 2026-10-12.  CPU only: GPU1-3 held 13.0-17.6 GiB by other lanes at launch, GPU0/4/6 are
# step 1's, GPU5 is reserved.  Steps:
#   0  the test file (main venv, CPU)
#   1-3  the Qwen3.5-0.8B parity arms (text / two-image VL / 8-frame video), fp32, torch GDN
#   4  verify every cache written here
#   5  the reviewer's repro on the REAL teacher, through the resume path: a scratch copy of the
#      sealed Qwen3.8-27B-U build-text smoke cache, manifest removed, plus one session record
#      in the exact shape a SIGKILL leaves (the real "before" digest, no "after"); then
#      build-text resumes it (no shard pending, so no teacher load) and must seal.  Pascal
#      venv because the smoke cache's locked kernel path is fla; CUDA hidden, --device cpu.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
PYP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
M08=/home/cereal/models/Qwen3.5-0.8B
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
W1=$OC/distill/parity/qtext-2026-10-12
SMOKE=$OC/distill/parity/qtext-2026-10-12-r3/q27-cache/q38-27bu-smoke
W=$OC/distill/parity/qtext-2026-10-12-cpu-v103
R=$OC/distill/tps
mkdir -p "$W"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
RUN="nice -n 15 $PY $T/qtext_teacher_cache.py"
COMMON="--model-dir $M08 --device cpu --dtype float32 --linear-attn-kernel torch"
sha256sum $T/qtext_teacher_cache.py $T/test_qtext_teacher_cache.py $OC/distill/tps_probe.py
echo "===== 0 tests $(date -u +%FT%TZ)"
(cd $T && nice -n 15 $PY -m pytest -o addopts="" -p no:cacheprovider -p no:warnings test_qtext_teacher_cache.py -q 2>&1 | tail -25)
echo "===== 1 parity-text $(date -u +%FT%TZ)"
$RUN parity-text $COMMON --micro-batch 4 --parity-rows 8 --parity-length 1024 --work-dir $W/text \
  --receipt $R/qtext-parity-text-08b-fp32-cpu-v103.json; echo "exit $?"
echo "===== 2 parity-vl two images $(date -u +%FT%TZ)"
$RUN parity-vl $COMMON --micro-batch 2 --work-dir $W/vl --receipt $R/qtext-parity-vl-08b-fp32-cpu-v103.json; echo "exit $?"
echo "===== 3 parity-vl 8-frame video $(date -u +%FT%TZ)"
$RUN parity-vl $COMMON --micro-batch 1 --corpus $W1/video-src/video.jsonl --work-dir $W/video \
  --receipt $R/qtext-parity-video-08b-fp32-cpu-v103.json; echo "exit $?"
echo "===== 4 verify $(date -u +%FT%TZ)"
for d in $W/*/cache/*/*/; do $RUN verify --cache-dir "$d" | tail -1; done
echo "===== 5 real-teacher interrupted-session resume $(date -u +%FT%TZ)"
Q=$W/q27-interrupt
rm -rf "$Q"; mkdir -p "$Q"
cp -a "$SMOKE" "$Q/"
C=$(ls -d $Q/q38-27bu-smoke/*/)
# Guard: the corpus must address THIS cache, or build-text would start a fresh 27B CPU build.
CS=$($PY -c "import sys; sys.path.insert(0, '$T'); import qtext_teacher_cache as qc; print(qc.load_text_corpus(__import__('pathlib').Path(sys.argv[1]))['sha256'])" $W1/q27-text-corpus.safetensors)
[ "$(basename $C)" = "$CS" ] || { echo "corpus sha $CS does not address $C; step 5 skipped"; exit 3; }
grep -c '"shard":' "$C/manifest.json" | sed 's/^/manifest shard lines before removal: /'
rm -f "$C/manifest.json"
$PY - "$C" <<'EOF5'
import json, sys
from pathlib import Path
c = Path(sys.argv[1])
(src,) = sorted((c / "sessions").glob("session-*.json"))
real = json.loads(src.read_text())
# What a SIGKILL after the "before" digest leaves: the record written at line "session
# record with teacher_digest_before", nothing from the finally block.  Named one second
# after the real session so it sorts after it and before the resume.
sid = "20261013T020537Z-999999"
killed = {k: real[k] for k in ("tool", "engine", "teacher_digest_before")}
killed.update({"session": sid, "started_utc": "2026-10-13T02:05:37Z", "lock_taken_over_from": None,
               "devices_at_start": []})
(c / "sessions" / f"session-{sid}.json").write_text(json.dumps(killed, indent=1, sort_keys=True))
print("planted before-only session", sid, "tree", killed["teacher_digest_before"]["tree_sha256"][:16])
EOF5
nice -n 15 $PYP $T/qtext_teacher_cache.py build-text --model-dir $M27 --teacher q38-27bu-smoke \
  --corpus $W1/q27-text-corpus.safetensors --out-root $Q --device cpu --dtype float32 --micro-batch 2 \
  --rows-per-shard 2 --head-chunk-rows 1024 --digest-threads 8 --allocator-receipt $Q/alloc-check.json \
  --purpose "qtext 1.0.3 interrupted-session resume check (CPU, no shard pending)" 2>&1 | grep -v "Triton\|fast path" | tail -12
echo "exit ${PIPESTATUS[0]}"
$PY - "$C" "$Q/alloc-check.json" <<'EOF6'
import json, sys
from pathlib import Path
c = Path(sys.argv[1]); m = json.loads((c / "manifest.json").read_text())
print("sessions", [s["session"] for s in m["sessions"]])
print("sessions_interrupted", [(s["session"], s["shards_written"], s["vouched_by"]) for s in m["sessions_interrupted"]])
print("teacher", {k: m["teacher"][k] for k in ("tree_sha256_before", "tree_sha256_after",
      "equal_before_and_after_every_session", "interrupted_sessions_vouched_for")})
print("positions_total", m["positions_total"], "all_finite", m["all_finite"])
a = json.loads(Path(sys.argv[2]).read_text()); print("allocator outcome", a.get("outcome"), "failure", a.get("failure"))
EOF6
nice -n 15 $PY $T/qtext_teacher_cache.py verify --cache-dir "$C" | tail -1
echo "===== done $(date -u +%FT%TZ)"
