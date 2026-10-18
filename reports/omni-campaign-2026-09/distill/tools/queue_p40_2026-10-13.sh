#!/usr/bin/env bash
# Lane p40 queue (Claude, wave 2, 2026-10-13). Installed as $OC/distill/queues/p40.sh; run detached:
#   setsid nohup bash $OC/distill/queues/p40.sh > $OC/distill/queues/p40.log 2>&1 < /dev/null &
#
# media_b_teacher_cache 1.0.8 on the P40s only (GPU1-3: one process per card, one card per container, never P2P),
# every teacher cache under /mnt/lexar/omni-campaign/teacher_cache/:
#   CPU   ace-encode ranks 0/2 + 1/2 (CPU-only containers, 8 threads each, nice 15): the 964 ACE x0 stage files
#   GPU1  hy3d-run rank 0/2 (128 shapes x 30 steps)  -> ACE-LM rank claims
#   GPU2  hy3d-run rank 1/2 (128 shapes x 30 steps)  -> ACE-LM rank claims
#   GPU3  ace-dit-run on the first 8 clips (early rate) -> QVL32 ViT block 13 over 7,492 images (once Lexar has room)
#         -> ace-dit-run on all clips (once the CPU encodes exit 0) -> ACE-LM rank claims -> "GPU3 FREE"
# ACE-LM: 6,667 prompts x 300 codes = 2,000,100 code tokens in LM_WORLD ranks, claimed by whichever card is free
# (mkdir claim), one retry per failed rank (resume skips written shards).
# Every GPU job waits for step 1's S8 review (pid 917640) to exit and for its card < 2,000 MiB; `launch` re-checks the
# card, refuses GPU 0/4/5/6, and writes $OC/composite/receipts/allocator/<job>.json. Each cache is sealed only when
# every job that writes it exited 0, then verified and structurally checked (`check`: mesh faces > 0, finite non-zero
# ACE x0, LM codes, QVL rows) with a receipt under $OC/distill/media_b/receipts_p40/.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/media_b_teacher_cache.py
PY=/home/cereal/model-surgery-lab/venv/bin/python
Q=$OC/distill/queues
STATUS=$Q/p40.status
ST=$Q/p40.state
LOGS=$OC/distill/media_b/logs_p40
REC=$OC/distill/media_b/receipts_p40
TC=/mnt/lexar/omni-campaign/teacher_cache
CORP=$OC/distill/media_b/corpora
HY=$CORP/hy3d_stagec_d256.json
AC=$CORP/ace_clips_stagec_d964.json
AP=$CORP/ace_prompts_stagec_d2m.json
QV=$CORP/qvl_vit_stagec_d7492.json
ACE_SECS=12.52          # the corpus lane's pre-cut clip length (313 Oobleck frames)
LM_WORLD=16             # ~125k code tokens per rank
STEP1_PID=917640
HY3D_MIN_LEXAR_GB=30    # the HY3D cache grows ~1.5 GB/h (22.5 GB total)
QVL_MIN_LEXAR_GB=110    # the QVL cache is ~55-70 GB fp32, written in ~1-2 h
export OC CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9 PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$ST" "$LOGS" "$REC"

st() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }
host() { nice -n 15 "$PY" "$T" "$@"; }
addr() { host address -- "$@" 2>> "$LOGS/address.log" | sed -n 's/^CACHE_DIR //p'; }
lexar_gb() { df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9; }
card_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -dc 0-9; }
step1_alive() { [ -r /proc/$STEP1_PID/cmdline ] && tr '\0' ' ' < /proc/$STEP1_PID/cmdline | grep -q run_review_finite_guard; }

wait_step1() {
  local said=0
  while step1_alive; do
    [ $said = 0 ] && st "WAIT $1: step-1 S8 review pid $STEP1_PID still running" && said=1
    sleep 60
  done
}
wait_card() {
  local said=0
  while [ "$(card_mib "$1")" -ge 2000 ]; do
    [ $said = 0 ] && st "WAIT $2: GPU$1 at $(card_mib "$1") MiB (>= 2000)" && said=1
    sleep 60
  done
}
wait_lexar() {  # wait_lexar <GB> <job>
  local said=0
  while [ "$(lexar_gb)" -lt "$1" ]; do
    [ $said = 0 ] && st "WAIT $2: /mnt/lexar has $(lexar_gb) G free (< $1 G)" && said=1
    sleep 300
  done
}

gpu_job() {  # gpu_job <name> <card> <runtime> <worker args...>; returns the launch exit code
  local name=$1 card=$2 runtime=$3 rc
  shift 3
  wait_step1 "$name"
  wait_card "$card" "$name"
  st "START $name gpu$card (lexar $(lexar_gb) G free)"
  host launch --card "$card" --name "$name" --log "$LOGS/$name.log" --runtime "$runtime" --threads 8 -- "$@" >> "$LOGS/launch.log" 2>&1
  rc=$?
  echo $rc > "$ST/$name.rc"
  st "JOB $name gpu$card exit $rc log $LOGS/$name.log allocator $OC/composite/receipts/allocator/$name.json"
  return $rc
}

seal_check() {  # seal_check <teacher> <cache dir> <corpus> <tag>
  local teacher=$1 dir=$2 corpus=$3 tag=$4 rc
  host seal --cache-dir "$dir" --corpus "$corpus" --teacher "$teacher" >> "$LOGS/seal-$tag.log" 2>&1
  rc=$?
  st "SEAL $tag exit $rc $dir"
  [ $rc = 0 ] || return $rc
  host check --cache-dir "$dir" --out "$REC/$tag.check.json" >> "$LOGS/check-$tag.log" 2>&1
  rc=$?
  st "CHECK $tag exit $rc $(grep -h 'CHECK \|REFUSED' "$LOGS/check-$tag.log" | tail -1 | cut -c1-400)"
  return $rc
}

rc_of() { [ -f "$ST/$1.rc" ] && cat "$ST/$1.rc" || echo none; }

# ---------------------------------------------------------------- HY3D (GPU1 + GPU2)

try_seal_hy3d() {
  local a b
  a=$(rc_of p40-hy3d-r0-gpu1); b=$(rc_of p40-hy3d-r1-gpu2)
  [ "$a" != none ] && [ "$b" != none ] || return 0
  mkdir "$ST/hy3d.seal.lock" 2>/dev/null || return 0
  if [ "$a" = 0 ] && [ "$b" = 0 ]; then
    seal_check hy3d_trajectories "$HYD" "$HY" hy3d
  else
    st "GATE hy3d seal skipped: rank exits r0=$a r1=$b"
  fi
}

hy3d_chain() {  # hy3d_chain <card> <rank>
  local card=$1 r=$2
  wait_lexar $HY3D_MIN_LEXAR_GB "p40-hy3d-r$r-gpu$card"
  gpu_job "p40-hy3d-r$r-gpu$card" "$card" docker hy3d-run --corpus "$HY" --cache-root "$TC/hy3d" --rank "$r" --world 2 \
    --replay-every 16 --reference-first 1
  try_seal_hy3d
  lm_claims "$card"
}

# ---------------------------------------------------------------- ACE LM rank claims (any P40)

try_seal_lm() {
  local r rr
  for r in $(seq 0 $((LM_WORLD - 1))); do
    rr=$(printf %02d "$r")
    [ "$(rc_of "lm-r$rr")" = 0 ] || return 0
  done
  mkdir "$ST/lm.seal.lock" 2>/dev/null || return 0
  seal_check ace_lm_codes "$LMD" "$AP" ace-lm
}

claim_lm_rank() {  # claim_lm_rank <card>: run the next unclaimed (or once-failed) LM rank; 1 when none is left
  local card=$1 r rr rc
  for r in $(seq 0 $((LM_WORLD - 1))); do
    rr=$(printf %02d "$r")
    if mkdir "$ST/lm-claim-r$rr" 2>/dev/null; then
      gpu_job "p40-ace-lm-r$rr-gpu$card" "$card" docker ace-lm-run --corpus "$AP" --cache-root "$TC/ace" --rank "$r" --world $LM_WORLD
      rc=$?
      echo $rc > "$ST/lm-r$rr.rc"
      try_seal_lm
      return 0
    fi
  done
  for r in $(seq 0 $((LM_WORLD - 1))); do
    rr=$(printf %02d "$r")
    if [ "$(rc_of "lm-r$rr")" != 0 ] && [ "$(rc_of "lm-r$rr")" != none ] && mkdir "$ST/lm-retry-r$rr" 2>/dev/null; then
      gpu_job "p40-ace-lm-r$rr-retry-gpu$card" "$card" docker ace-lm-run --corpus "$AP" --cache-root "$TC/ace" --rank "$r" --world $LM_WORLD
      rc=$?
      echo $rc > "$ST/lm-r$rr.rc"
      try_seal_lm
      return 0
    fi
  done
  return 1
}

lm_claims() {
  while claim_lm_rank "$1"; do :; done
  st "LANE gpu$1: no ACE-LM rank left to claim"
}

# ---------------------------------------------------------------- ACE x0 encodes (CPU only)

encode_chain() {
  local r
  for r in 0 1; do
    (
      st "START p40-ace-encode-r$r cpu-only (8 threads)"
      host launch --cpu-only --name "p40-ace-encode-r$r" --log "$LOGS/p40-ace-encode-r$r.log" --threads 8 -- \
        ace-encode --corpus "$AC" --cache-root "$TC/ace" --clip-seconds $ACE_SECS --rank "$r" --world 2 >> "$LOGS/launch.log" 2>&1
      rc=$?
      echo $rc > "$ST/p40-ace-encode-r$r.rc"
      st "JOB p40-ace-encode-r$r cpu exit $rc log $LOGS/p40-ace-encode-r$r.log allocator $OC/composite/receipts/allocator/p40-ace-encode-r$r.json"
    ) &
  done
  wait
}

encodes_done() { [ "$(rc_of p40-ace-encode-r0)" != none ] && [ "$(rc_of p40-ace-encode-r1)" != none ]; }
encodes_ok() { [ "$(rc_of p40-ace-encode-r0)" = 0 ] && [ "$(rc_of p40-ace-encode-r1)" = 0 ]; }

wait_staged() {  # wait_staged <n>: up to 90 min for x0 stage files 0..n-1 (so no item is encoded twice at once)
  local n=$1 i k missing
  for k in $(seq 1 90); do
    missing=0
    for i in $(seq 0 $((n - 1))); do
      [ -s "$X0S/$(printf 'x0-%05d.json' "$i")" ] || missing=$((missing + 1))
    done
    [ $missing = 0 ] && return 0
    sleep 60
  done
  st "WAIT early ACE DiT: $missing of $n x0 stage files still missing after 90 min; ace-dit-run encodes them inline"
}

# ---------------------------------------------------------------- GPU3

gpu3_chain() {
  local qvl_done=0 dit_done=0 rc said_blocked=0
  wait_staged 8
  gpu_job p40-ace-dit-early8-gpu3 3 docker ace-dit-run --corpus "$AC" --cache-root "$TC/ace" --clip-seconds $ACE_SECS --limit 8
  while :; do
    if [ $qvl_done = 0 ] && [ "$(lexar_gb)" -ge $QVL_MIN_LEXAR_GB ]; then
      gpu_job p40-qvl-vit-gpu3 3 pascal qvl-run --corpus "$QV" --cache-root "$TC/qvl32_vit"
      rc=$?
      qvl_done=1
      if [ $rc = 0 ]; then seal_check qvl32_vit_l13 "$QVD" "$QV" qvl-vit; else st "GATE qvl-vit seal skipped: run exit $rc"; fi
      continue
    fi
    if [ $dit_done = 0 ] && encodes_done; then
      dit_done=1
      if encodes_ok; then
        gpu_job p40-ace-dit-gpu3 3 docker ace-dit-run --corpus "$AC" --cache-root "$TC/ace" --clip-seconds $ACE_SECS
        rc=$?
        if [ $rc = 0 ] && [ "$(rc_of p40-ace-dit-early8-gpu3)" = 0 ]; then
          seal_check ace_dit_velocity "$DITD" "$AC" ace-dit
        else
          st "GATE ace-dit seal skipped: early8 exit $(rc_of p40-ace-dit-early8-gpu3), full exit $rc"
        fi
      else
        st "GATE ace-dit skipped: CPU encode exits r0=$(rc_of p40-ace-encode-r0) r1=$(rc_of p40-ace-encode-r1)"
      fi
      continue
    fi
    claim_lm_rank 3 && continue
    [ $qvl_done = 1 ] && [ $dit_done = 1 ] && break
    if [ $said_blocked = 0 ]; then
      st "WAIT gpu3: no LM rank left; qvl_done=$qvl_done (lexar $(lexar_gb) G, needs $QVL_MIN_LEXAR_GB) dit_done=$dit_done"
      said_blocked=1
    fi
    sleep 120
  done
  st "GPU3 FREE"
}

# ---------------------------------------------------------------- main

st "QUEUE start p40.sh sha $(sha256sum "$0" | cut -c1-16) tool $(sha256sum "$T" | cut -c1-16) $(host --version 2>&1)"
[ "$(host --version 2>&1)" = "media_b_teacher_cache 1.0.8" ] || { st "PREFLIGHT FAIL: tool is not 1.0.8"; exit 2; }
HYD=$(addr hy3d-run --corpus "$HY" --cache-root "$TC/hy3d")
DITD=$(addr ace-dit-run --corpus "$AC" --cache-root "$TC/ace" --clip-seconds $ACE_SECS)
X0S=$(addr ace-encode --corpus "$AC" --cache-root "$TC/ace" --clip-seconds $ACE_SECS)
LMD=$(addr ace-lm-run --corpus "$AP" --cache-root "$TC/ace")
QVD=$(addr qvl-run --corpus "$QV" --cache-root "$TC/qvl32_vit")
for v in HYD DITD X0S LMD QVD; do
  [ -n "${!v}" ] || { st "PREFLIGHT FAIL: no cache address for $v (see $LOGS/address.log)"; exit 2; }
  st "ADDR $v ${!v}"
done
export HYD DITD X0S LMD QVD

encode_chain &
hy3d_chain 1 0 &
hy3d_chain 2 1 &
gpu3_chain &
wait
st "QUEUE end"
