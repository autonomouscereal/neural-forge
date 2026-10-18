#!/usr/bin/env bash
# test_vmath_queue.sh -- drives the REAL vmath_queue.sh against a stub sampler (fixtures/vmath_queue/), a scratch queues
# dir and fake card / memory state.  No GPU, no model, nothing outside a mktemp dir.  Run on the server (needs the real
# d1_seal_lib.py and a python with the pilot's imports):
#   bash test_vmath_queue.sh            -> "ALL n CHECKS PASSED" or the first failure with the queue log
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PYBIN=${PYBIN:-/home/cereal/model-surgery-lab/venv/bin/python}
SEAL_LIB=${SEAL_LIB:-/home/cereal/omni-campaign/distill/tools/d1_seal_lib.py}
TMP=$(mktemp -d)
trap 'pkill -f "$TMP" 2>/dev/null; rm -rf "$TMP"' EXIT
PASS=0
check() { if eval "$2"; then PASS=$((PASS + 1)); echo "ok   $1"; else echo "FAIL $1"; echo "--- queue log"; cat "$CUR/logs/queue.log" 2>/dev/null | tail -30; echo "--- claims"; cat "$Q/p40.claims"; echo "--- stub log"; cat "$STUB_LOG"; exit 1; fi; }
wait_for() { local n=0; until eval "$1"; do sleep 0.5; n=$((n + 1)); [ $n -gt $(( ${2:-60} * 2 )) ] && return 1; done; return 0; }

Q=$TMP/q; T=$TMP/t; Q4DIR=$TMP/q4; mkdir -p "$Q" "$T" "$Q4DIR"
cp "$SEAL_LIB" "$T/d1_seal_lib.py"
cp "$HERE/fixtures/vmath_queue/stub_selfgen_sampler.py" "$T/selfgen_sampler.py"
echo "fake gguf" > "$Q4DIR/fake.Q4_K_M.gguf"
echo "{\"q4\": {\"path\": \"$Q4DIR/fake.Q4_K_M.gguf\", \"sha256\": \"$(sha256sum "$Q4DIR/fake.Q4_K_M.gguf" | cut -d' ' -f1)\"}}" > "$Q4DIR/q4_copy_receipt.json"
echo 'MemAvailable:   100000000 kB' > "$TMP/meminfo"
echo '{"1": {"used_mib": 5, "apps": 0}, "2": {"used_mib": 5, "apps": 0}}' > "$TMP/gpus.json"
export STUB_LOG=$TMP/stub.log
: > "$STUB_LOG"

make_prompts() {  # make_prompts <pilot dir>: 3 problems x k=2 (R "1", R "1", A) with a manifest selfgen would accept
  "$PYBIN" - "$1/prompts" <<'PY'
import hashlib, json, os, sys
d = sys.argv[1]; os.makedirs(d, exist_ok=True)
probs = [("s1", "R", "1"), ("s2", "R", "1"), ("s3", "A", None)]
rows = []
for src, st, ref in probs:
    for j in range(2):
        rows.append({"rid": f"{src}~k{j}", "source_rid": src, "sample_index": j, "stratum": st, "route": "math_verified",
                     "prompt_text": "p", "prompt_tokens": 1, "max_new_tokens": 8, "messages": [{"role": "user", "content": f"problem {src}"}]})
with open(f"{d}/prompts.jsonl", "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
with open(f"{d}/references.jsonl", "w") as fh:
    for src, st, ref in probs: fh.write(json.dumps({"source_rid": src, "stratum": st, "reference": ref}) + "\n")
sha = hashlib.sha256(open(f"{d}/prompts.jsonl", "rb").read()).hexdigest()
json.dump({"file": {"sha256": sha}, "prompts": len(rows), "k": 2, "manifest_sha256": "test"}, open(f"{d}/manifest.json", "w"))
PY
}

start_queue() {  # start_queue <pilot dir> [extra env...]
  CUR=$1; shift
  mkdir -p "$CUR/logs"
  env VMATH_Q="$Q" VMATH_T="$T" VMATH_RT="$HERE" VMATH_PY="$PYBIN" VMATH_DIR="$CUR" VMATH_Q4DIR="$Q4DIR" VMATH_POLL_S=1 \
      VMATH_BACKOFF_S=1 VMATH_CHECK_S=1 VMATH_TOKEN_TARGET=100000 "$@" \
      VMATH_GATE_EXTRA="--meminfo $TMP/meminfo --gpu-state-json $TMP/gpus.json" \
      setsid bash "$HERE/vmath_queue.sh" > "$CUR/logs/queue.log" 2>&1 < /dev/null &
  QPID=$!
}

TOOLCALL="2026-10-14T14:28:20Z CLAIM GPU1,GPU2 owner=claude lane=toolcall what=llama-server"
echo "$TOOLCALL" > "$Q/p40.claims"

# 1. a live claim of another lane: the queue waits and claims nothing
CUR=$TMP/p1; make_prompts "$CUR"; start_queue "$CUR"
wait_for "grep -q 'state=waiting' '$CUR/queue.status' 2>/dev/null" 20
check "waits on another lane's live claim" "grep -q 'GPU1 claimed' '$CUR/queue.status' && ! grep -q 'lane=research-vmath' '$Q/p40.claims'"

# 2. the toolcall lane releases: claim, serve both cards, sample, verify, release, delete the Q4 copy, DONE
echo "2026-10-15T03:00:00Z RELEASE GPU1,GPU2 owner=claude lane=toolcall (done)" >> "$Q/p40.claims"
wait_for "[ -e '$CUR/DONE' ]" 60
check "completes after the release" "[ -e '$CUR/DONE' ] && [ -s '$CUR/verify/summary.json' ]"
check "claim then release lines" "grep -q 'CLAIM GPU1,GPU2 owner=claude lane=research-vmath' '$Q/p40.claims' && grep -q 'RELEASE GPU1,GPU2 owner=claude lane=research-vmath' '$Q/p40.claims'"
check "served and stopped both cards" "[ \$(grep -c '^serve' '$STUB_LOG') -eq 2 ] && [ \$(grep -c '^stop' '$STUB_LOG') -eq 2 ]"
check "transient Q4 removed after success" "[ ! -e '$Q4DIR/fake.Q4_K_M.gguf' ]"
check "verify accepted R by reference and A by agreement" "$PYBIN -c \"import json,sys; s=json.load(open('$CUR/verify/summary.json'))['strata']; sys.exit(0 if s['R']['accepted_samples']==4 and s['A']['accepted_samples']==2 else 1)\""
wait_for "! kill -0 $QPID 2>/dev/null" 20
check "queue exits after DONE" "! kill -0 $QPID 2>/dev/null"

# 3. a YIELD request mid-run: sampler stopped, servers stopped, RELEASE, then the queue waits; rm YIELD -> resumes to DONE
echo "fake gguf" > "$Q4DIR/fake.Q4_K_M.gguf"
: > "$STUB_LOG"
CUR=$TMP/p3; make_prompts "$CUR"; start_queue "$CUR" STUB_SLEEP=3
wait_for "grep -q 'state=running' '$CUR/queue.status' 2>/dev/null" 30
sleep 4
touch "$CUR/YIELD"
wait_for "grep -q 'state=yielding' '$CUR/queue.status'" 30
wait_for "grep -q 'RELEASE GPU1,GPU2 owner=claude lane=research-vmath job=vmath-pilot released (session ended' '$Q/p40.claims'" 30
check "yield released the cards" "grep -q 'session ended rc=20' '$Q/p40.claims'"
check "yield stopped the sampler and the servers" "grep -q 'run got SIGTERM' '$STUB_LOG' && [ \$(grep -c '^stop' '$STUB_LOG') -eq 2 ] && ! pgrep -f '$CUR/run' > /dev/null"
wait_for "grep -q 'YIELD file present' '$CUR/queue.status'" 20
check "re-gate waits while YIELD exists" "grep -q 'YIELD file present' '$CUR/queue.status'"
done_before=$(wc -l < "$CUR/run/journal.jsonl")
rm -f "$CUR/YIELD"
wait_for "[ -e '$CUR/DONE' ]" 90
check "resumes after YIELD is removed and finishes" "[ -e '$CUR/DONE' ]"
check "resume kept the journal (no index sampled twice)" "[ \$(cut -d, -f1 '$CUR/run/journal.jsonl' | sort | uniq -d | wc -l) -eq 0 ] && [ $done_before -ge 1 ]"
wait_for "! kill -0 $QPID 2>/dev/null" 20

# 4. D1 sealed but not reviewed: v1's review may need the P40s -> the queue does not start
echo "fake gguf" > "$Q4DIR/fake.Q4_K_M.gguf"
touch "$Q/d1.sealed"
CUR=$TMP/p4; make_prompts "$CUR"; start_queue "$CUR"
wait_for "grep -q 'state=waiting' '$CUR/queue.status' 2>/dev/null" 20
check "waits while D1's review is pending" "grep -q 'd1.sealed without d1.reviewed' '$CUR/queue.status'"
kill -TERM $QPID; wait $QPID 2>/dev/null
rm -f "$Q/d1.sealed"

# 5. another lane claims a held card mid-run -> yield
: > "$STUB_LOG"
CUR=$TMP/p5; make_prompts "$CUR"; start_queue "$CUR" STUB_SLEEP=3
wait_for "grep -q 'state=running' '$CUR/queue.status' 2>/dev/null" 30
echo "2026-10-15T05:00:00Z GPU2 lane=speech owner=codex job=c-s2 pid=1" >> "$Q/p40.claims"
wait_for "grep -q 'another lane claimed a held card' '$CUR/queue.status'" 30
check "a foreign claim on a held card makes it yield" "grep -q 'another lane claimed a held card' '$CUR/queue.status'"
wait_for "grep -q 'session ended rc=20' <(tail -1 '$Q/p40.claims')" 30
kill -TERM $QPID; wait $QPID 2>/dev/null
check "queue SIGTERM leaves no sampler behind" "! pgrep -f '$CUR/run' > /dev/null"

echo "ALL $PASS CHECKS PASSED"
