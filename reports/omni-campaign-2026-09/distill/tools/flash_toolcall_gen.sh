#!/bin/bash
# flash_toolcall_gen.sh -- the Chimera Omni tool-call dataset, written by Qwen3.8-Flash-Next (abliterated Strata IQ3_S,
# jimmy31chen/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-IQ3_S-Strata, sha cb5958e5..., Apache-2.0 / Qwen Community
# License 1.0) on the four P40s.  Cyril 2026-10-17: "[Cyril quote withheld]".  Flash Next writes EVERYTHING (tool schemas, user requests, simulated backends, assistant turns); the
# sandbox tools execute deterministically; K3 (primary) and Q38 rescore the sealed conversations later.  No Claude/Codex
# output is used anywhere.  Flash Next's tokenizer is identical to Qwen 3.8 27B's (248,077 ids, 247,587 merges, every
# special id), so toolcall_selfgen.py 1.2.0 drives it unchanged; the canonical output is conversations.jsonl (text,
# tokenizer-agnostic -- the token views are derived).
#   setsid nohup bash flash_toolcall_gen.sh > $W/driver.log 2>&1 < /dev/null &
# Steps: verified GGUF -> CLAIM GPU1,2,3,5 + their locks -> one llama-server split over the four cards -> eog-probe ->
# schemas -> plan -> run (token target, resumable) -> finalize.  The server is stopped and the cards RELEASED on every exit.
set -u
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
LLAMA=$OC/export/llamacpp
GGUF=${GGUF:-/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata/RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf}
GGUF_SHA=cb5958e5b0e97804fa2973f7ce36943a1b0d735b1b83fc1e52b8c3f3b97f87e6
W=${W:-$OC/distill/toolcall/flash-gen-2026-10-18}
PORT=${PORT:-18340}
SLOTS=${SLOTS:-4}
CTX_PER_SLOT=${CTX_PER_SLOT:-8192}
ITEMS=${ITEMS:-3000}
TARGET=${TARGET:-3000000}
SEED=${SEED:-20261018}
EXTRA=${EXTRA:-}            # e.g. EXTRA='-ot per_layer=CPU' if the n-gram tables must stay in host RAM
BANK=$OC/calibration/banks/omni_toolcall_bank_v1.json
URL=http://127.0.0.1:$PORT
STATUS=$W/status
mkdir -p "$W"
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src OMP_NUM_THREADS=4
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) $*" | tee -a "$STATUS"; }
SPID=""; CLAIMED=0
cleanup() {
  [ -n "$SPID" ] && kill -0 "$SPID" 2> /dev/null && { kill -TERM "$SPID"; sleep 10; kill -0 "$SPID" 2> /dev/null && kill -KILL "$SPID"; }
  if [ "$CLAIMED" = 1 ]; then
    echo "$(utc) RELEASE GPU1,GPU2,GPU3,GPU5 owner=claude lane=toolcall job=flash-toolcall-gen released ($1)" >> "$Q/p40.claims"
    CLAIMED=0
  fi
  say "ended: $1"
}
trap 'cleanup "TERM"; exit 75' TERM INT
trap 'cleanup "driver exit"' EXIT

say "driver start; gguf $GGUF; work $W; toolcall_selfgen $(sha256sum $T/toolcall_selfgen.py | cut -c1-16)"
[ -s "$GGUF" ] || { say "STOP: $GGUF missing"; exit 3; }
if [ ! -s "$W/gguf.sha256.ok" ]; then
  got=$(nice -n 15 sha256sum "$GGUF" | cut -d" " -f1)
  [ "$got" = "$GGUF_SHA" ] || { say "STOP: gguf sha $got != $GGUF_SHA"; exit 3; }
  echo "$got" > "$W/gguf.sha256.ok"
fi
for g in 1 2 3 5; do
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -dc 0-9)" -ge 2000 ]; do say "waiting: GPU$g busy"; sleep 60; done
done
exec 6>> "$Q/gpu1.lock" 7>> "$Q/gpu2.lock" 8>> "$Q/gpu3.lock" 4>> "$Q/gpu5.lock"
flock -w 600 6 && flock -w 600 7 && flock -w 600 8 && flock -w 600 4 || { say "STOP: a gpu1/2/3/5 lock is held"; exit 5; }
echo "$(utc) CLAIM GPU1,GPU2,GPU3,GPU5 owner=claude lane=toolcall job=flash-toolcall-gen pid=$$ log=$W/driver.log preempt: kill -TERM $$ (run is journaled and resumable)" >> "$Q/p40.claims"
CLAIMED=1

if ! curl -sf "$URL/health" > /dev/null; then
  say "starting llama-server (GPU1,2,3,5, layer split, $SLOTS slots x $CTX_PER_SLOT)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,5 LD_LIBRARY_PATH=$LLAMA/bin:${LD_LIBRARY_PATH:-} \
    nice -n 5 "$LLAMA/bin/llama-server" -m "$GGUF" -ngl 999 --split-mode layer -ts 1,1,1,1 \
    -c $((SLOTS * CTX_PER_SLOT)) -np "$SLOTS" --host 127.0.0.1 --port "$PORT" -t 16 -fa auto $EXTRA \
    > "$W/llama-server.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do
    curl -sf "$URL/health" > /dev/null && break
    kill -0 "$SPID" 2> /dev/null || { say "STOP: llama-server exited (see $W/llama-server.log): $(tail -3 $W/llama-server.log | tr '\n' ' ' | cut -c1-300)"; exit 4; }
    sleep 10
  done
  curl -sf "$URL/health" > /dev/null || { say "STOP: llama-server not healthy after 30 min"; exit 4; }
fi
say "server healthy: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1,2,3,5 | tr '\n' ' ')"
"$PY" "$T/selfgen_sampler.py" eog-probe --server "$URL" --out "$W/eog_probe.json" > "$W/eog_probe.log" 2>&1 \
  || { say "STOP: eog-probe failed (see $W/eog_probe.log)"; exit 5; }
if [ ! -s "$W/schemas/schemas.jsonl" ]; then
  say "schemas (Flash Next writes the training tool schemas)"
  "$PY" "$T/toolcall_selfgen.py" schemas --server "$URL" --out-dir "$W/schemas" --per-domain 3 --workers-per-server "$SLOTS" \
    > "$W/schemas.log" 2>&1 || { say "STOP: schemas failed (see $W/schemas.log)"; exit 6; }
fi
if [ ! -s "$W/plan.jsonl" ]; then
  "$PY" "$T/toolcall_selfgen.py" plan --schemas "$W/schemas/schemas.jsonl" --out "$W/plan.jsonl" --items "$ITEMS" --base-seed "$SEED" \
    > "$W/plan.log" 2>&1 || { say "STOP: plan failed (see $W/plan.log)"; exit 7; }
fi
say "run: target $TARGET tokens over $(wc -l < $W/plan.jsonl) plan items"
"$PY" "$T/toolcall_selfgen.py" run --plan "$W/plan.jsonl" --server "$URL" --out-dir "$W/run" --token-target "$TARGET" \
  --workers-per-server "$SLOTS" --base-seed "$SEED" --gguf "$GGUF" --gguf-sha256 "$GGUF_SHA" > "$W/run.log" 2>&1
rc=$?
say "run rc=$rc $(tr -d '\n ' < $W/run/progress.json 2>/dev/null | cut -c1-240)"
case $rc in 0) fin=() ;; 2|3|4) fin=(--accept-short --cut-at-gap) ;; *) say "STOP: run failed rc=$rc"; exit "$rc" ;; esac
cleanup "run ended rc=$rc; finalize on CPU follows"
"$PY" "$T/toolcall_selfgen.py" finalize --out-dir "$W/run" --bank "$BANK" ${fin[@]+"${fin[@]}"} > "$W/finalize.log" 2>&1 \
  && say "SEALED: $(sha256sum $W/run/manifest.json | cut -c1-16) conversations=$(python3 -c "import json;d=json.load(open('$W/run/manifest.json'));print(d['conversations'],'tokens',d['tokens'])")" \
  || say "finalize failed (see $W/finalize.log); journal kept"
