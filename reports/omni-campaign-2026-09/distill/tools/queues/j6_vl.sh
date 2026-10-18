#!/usr/bin/env bash
# Stage D, repair plan Step 3b: j6 -- the T(i) vision-in + video-in answer caches (DISTILL_DESIGN_2026-10-12
# section 3 "Vision-/video-in answers", consumed by D3d only).  Written 2026-10-14 by Claude, lane vl.
# Installed as $Q/j6_vl.sh.  It REPLACES gpu0.sh's j6 (qtext 1.0.5: pixel-frame reference boxes against a
# 0-1000-grid teacher, out-root .../teacher_cache/qtext): do not run `gpu0.sh j6`.
#
#   bash $Q/j6_vl.sh <gpu index>                 6 = lane gpu6 after its H3 jobs; 0 = GPU0 after D1 seals
#   J6_GPU=<gpu index> bash $Q/j6_vl.sh          the same; this is how queue_gpu6.sh (1.2.0 :704) calls it
#   J6_SELFTEST=plan bash $Q/j6_vl.sh <gpu>      prints the plan and every gate's current value, runs nothing
#   J6_SELFTEST=lock bash $Q/j6_vl.sh <gpu>      takes the lock as a real run would, prints how, runs nothing
#   tests: bash $Q/test_j6_vl.sh                 (J6_LOCK_FILE / J6_DONE_FILE / J6_STATUS_FILE: test paths only)
#
# 1.1.0 (2026-10-14, review of 1.0.0): (a) the gpu index also comes from $J6_GPU -- queue_gpu6 passes no positional
# argument, and 1.0.0 exited 64 on that call; (b) queue_gpu6 holds $Q/j6.lock on its fd 8 across the call and a
# flock belongs to the open file description, so 1.0.0's own open + flock conflicted with its caller (exit 5).  An
# fd inherited from the caller that is open on the lock and flocks without blocking shares the caller's
# description: the caller's hold is ours.  (c) $Q/j6.done is written on success (queue_gpu6 reads it) and a
# present j6.done ends a later run at once; (d) any live `qtext_teacher_cache*.py build-vl` (the frozen copy
# included, which queue_gpu6's pgrep does not match) refuses the run.
#
# What it runs: qtext_teacher_cache 1.0.6 build-vl, FROZEN at $Q/frozen/qtext_teacher_cache-j6.py (sha pinned
# below and re-checked before every attempt), over
#   corpus/vision_in/vision_in.jsonl   5,896 records; 1.0.6 re-states the box references on the 0-1000 grid and
#                                      quarantines the rows it cannot read unambiguously (receipt below)
#   corpus/vision_in/video_in.jsonl    512 clips
# in ONE process (one teacher load, one digest pair, two caches) into $OUT/q38-27bu/<corpus sha>/.
# Gates -- each one WAITS (or refuses); none stops anything else:
#   card    only GPU6 or GPU0.  GPU5 never; GPU4 and the P40s refused.  GPU0 also needs $Q/d1.sealed (written by
#           the D1 seal, plan Step 4d) and no running container named *d1*.  The card must hold < 2,000 MiB
#           (the tool re-checks before it locks anything).
#   RAM     MemAvailable >= 70 GiB before every attempt (HostResidentReader 45.36 GiB + the j3 smoke's peak host
#           RSS 57.42 GB); never beside a live K3 pass (pgrep k3_teacher_cache).
#   disk    / >= 60 GB (the floor), /mnt/lexar >= 20 GB.
#   go      $Q/gpu0.vl_go present and naming the frozen tool's sha256.
#   lock    flock $Q/j6.lock (or the caller's hold on it, inherited); a second runner exits 5.
#   done    $Q/j6.done present: nothing to do, exit 0.  Another qtext build-vl live: refused (exit 3).
# Kernel: fla ("auto") on sm >= 8.0, the pure-torch GDN path on the RTX 8000 (sm 7.5: fla's Triton bf16 path is
# not proven there).  The kernel is locked into each cache's config, so a resume on the other card reuses the
# kernel the cache was started with, or is refused.
# Plan: shard-00000 of vision_in first (--max-new-shards 1: the rate and the memory on this card), then the rest;
# one automatic resume after a non-functional stop (exit 1 / 143); exit 2 is functional and final; then `verify`
# on every sealed cache.  On GPU6 the lane may wrap this script in its thermal watchdog: a SIGTERM is exit 143.
set -u
J6_VERSION="j6_vl.sh 1.1.0"
GPU_INDEX=${1:-${J6_GPU:-}}
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
L=$Q/j6
STATUS=${J6_STATUS_FILE:-$Q/j6.status}
LOCK=${J6_LOCK_FILE:-$Q/j6.lock}
DONE=${J6_DONE_FILE:-$Q/j6.done}
ALLOC=$OC/composite/receipts/allocator
PY=/home/cereal/model-surgery-lab/venv/bin/python
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
CORPUS=$OC/distill/corpus/vision_in
OUT=/mnt/lexar/omni-campaign/teacher_cache/qtext_vl
QT_FROZEN=$Q/frozen/qtext_teacher_cache-j6.py
QT_SHA=4c7badc1669d4044de765b04ed52c86c8096f9bb0b8ba55bc2b88024960c0f2c       # qtext_teacher_cache 1.0.6
ENGINE=$OC/distill/tps_probe.py
ENGINE_SHA=449eecc4acf985556a2454f4af25ead6379d2d515a7d77879f27d9c9d80f9a1a
BOX_RECEIPT=$OC/distill/tps/qtext-vl-boxes-corpus-v106-2026-10-14.json
GO=$Q/gpu0.vl_go
RAM_MIN_GIB=70
CARD_FREE_MIB=2000
ROOT_MIN_GB=60
LEXAR_MIN_GB=20
WAIT_MAX_S=${J6_WAIT_MAX_S:-172800}   # 48 h of waiting on a gate, then give up with a status line

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
status() { echo "$(stamp) $*" >> $STATUS; echo "[$(stamp)] STATUS $*"; }
mkdir -p $L $ALLOC $OUT

case "$GPU_INDEX" in
  6|0) ;;
  *) echo "usage: j6_vl.sh <6|0>  or  J6_GPU=<6|0> j6_vl.sh  (GPU5 never; GPU4 and the P40s are not j6 cards)"; exit 64 ;;
esac
LOG=$L/j6-vl-gpu$GPU_INDEX.log
GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU_INDEX '$1==g{print $2}')
GPU_NAME=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' -v g=$GPU_INDEX '$1==g{print $2}')
GPU_CC=$(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader | awk -F', ' -v g=$GPU_INDEX '$1==g{print $2}')
used_mib() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU_INDEX '$1==g{print $2}'; }
mem_avail_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
k3_live() { pgrep -f k3_teacher_cache >/dev/null 2>&1; }
d1_container_live() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qi d1; }
root_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
lexar_gb() { df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9; }
sha_of() { sha256sum "$1" 2>/dev/null | cut -c1-64; }
# Any qtext build-vl: the mutable tools/ copy, gpu0.sh's 1.0.5 job, or the frozen j6 copy.
live_build_vl() { pgrep -af 'qtext_teacher_cache[^ /]*\.py build-vl' | head -1 | cut -c1-200; }
mark_done() { echo "$(stamp) j6 done on GPU$GPU_INDEX by $J6_VERSION (tool sha $QT_SHA, log $LOG)" > $DONE; }

# The kernel: a cache already started (by either card) decides; otherwise the card does.
locked_kernel() {
  $PY - "$OUT/q38-27bu" <<'PY'
import glob, json, sys
kinds = {json.load(open(p))["linear_attention_kernel"]["requested"] for p in glob.glob(sys.argv[1] + "/*/cache_config.json")}
print(",".join(sorted(kinds)))
PY
}
card_kernel=$(awk -v c="$GPU_CC" 'BEGIN{print (c+0 >= 8.0) ? "auto" : "torch"}')
KERNEL=$(locked_kernel)
case "$KERNEL" in
  "") KERNEL=$card_kernel ;;
  auto|torch) ;;
  *) echo "caches under $OUT/q38-27bu lock different kernels ($KERNEL)"; exit 65 ;;
esac

gate_report() {
  echo "card GPU$GPU_INDEX $GPU_NAME sm$GPU_CC $GPU_UUID used $(used_mib) MiB (need < $CARD_FREE_MIB)"
  echo "kernel $KERNEL (card default $card_kernel; locked by an existing cache: $(locked_kernel || true))"
  echo "MemAvailable $(mem_avail_gib) GiB (need >= $RAM_MIN_GIB); K3 pass live: $(k3_live && echo yes || echo no)"
  echo "disk / $(root_gb) GB (need >= $ROOT_MIN_GB), /mnt/lexar $(lexar_gb) GB (need >= $LEXAR_MIN_GB)"
  echo "GPU0 only: d1.sealed $([ -f $Q/d1.sealed ] && echo present || echo absent); *d1* container running: $(d1_container_live && echo yes || echo no)"
  echo "go file $GO: $([ -f $GO ] && echo present || echo absent); names frozen sha: $(grep -q "$QT_SHA" $GO 2>/dev/null && echo yes || echo no)"
  echo "frozen tool $QT_FROZEN sha $(sha_of $QT_FROZEN) (pinned $QT_SHA); engine $(sha_of $ENGINE) (pinned $ENGINE_SHA)"
  echo "box receipt $BOX_RECEIPT $([ -f $BOX_RECEIPT ] && sha_of $BOX_RECEIPT || echo absent)"
  echo "$J6_VERSION; lock $LOCK; done file $DONE $([ -e $DONE ] && echo present || echo absent); other build-vl live: $(live_build_vl | grep . || echo none)"
}

if [ "${J6_SELFTEST:-}" = "plan" ]; then
  echo "j6 plan for GPU$GPU_INDEX:"
  gate_report
  echo "command: CUDA_VISIBLE_DEVICES=$GPU_UUID QTEXT_TPS_PROBE=$ENGINE $PY $QT_FROZEN build-vl --model-dir $M27 --device cuda:0" \
       "--dtype bfloat16 --attn-implementation sdpa --linear-attn-kernel $KERNEL --digest-threads 8 --teacher q38-27bu" \
       "--corpus $CORPUS/vision_in.jsonl --corpus $CORPUS/video_in.jsonl --out-root $OUT --micro-batch 8" \
       "--max-batch-tokens 32768 --batch-order length --rows-per-shard 64 --head-precision fp32" \
       "--chat-template-kwargs '{\"enable_thinking\": false}' --reference-boxes qwen1000 [--max-new-shards 1 first]"
  exit 0
fi

# The lock.  queue_gpu6 takes $LOCK on its fd 8 and calls this script while it holds it.  A flock belongs to the
# open file description, so opening the file again here would conflict with the caller.  An inherited fd that is
# open on $LOCK and flocks without blocking shares a description that holds the lock (the caller's): use it.  An
# inherited fd whose flock fails is someone else's hold and falls through to the exit 5 below.
LOCK_HOW=""
LOCK_REAL=$(readlink -f "$LOCK")
for fdp in /proc/$$/fd/*; do
  fd=${fdp##*/}
  case "$fd" in ''|*[!0-9]*) continue ;; esac
  [ "$fd" -gt 2 ] || continue
  [ "$(readlink -f "$fdp" 2>/dev/null)" = "$LOCK_REAL" ] || continue
  if flock -n "$fd"; then LOCK_HOW="held by the caller (inherited fd $fd)"; break; fi
done
if [ -z "$LOCK_HOW" ]; then
  exec 9> "$LOCK"
  if ! flock -n 9; then echo "another j6 runner holds $LOCK"; exit 5; fi
  LOCK_HOW="acquired (fd 9)"
fi
OTHER=$(live_build_vl)
if [ "${J6_SELFTEST:-}" = "lock" ]; then
  echo "J6 lock: $LOCK_HOW; gpu $GPU_INDEX; other build-vl: ${OTHER:-none}; done file: $([ -e $DONE ] && echo present || echo absent)"
  exit 0
fi
if [ "${J6_SELFTEST:-}" = "done" ]; then mark_done; echo "J6 done file written: $DONE"; exit 0; fi

refuse() { status "job=j6-vl gpu=$GPU_INDEX state=refused reason=$*"; exit 3; }
if [ -e $DONE ]; then
  status "job=j6-vl gpu=$GPU_INDEX state=skipped reason=already_done ($DONE: $(head -c 200 $DONE | tr '\n' ' '))"
  exit 0
fi
[ -z "$OTHER" ] || refuse "a qtext build-vl is already running: $OTHER"
[ -f $GO ] || refuse "no go file $GO"
grep -q "$QT_SHA" $GO || refuse "$GO does not name the frozen tool sha $QT_SHA"
[ "$(sha_of $QT_FROZEN)" = "$QT_SHA" ] || refuse "frozen tool $QT_FROZEN sha $(sha_of $QT_FROZEN) != pinned $QT_SHA"
[ "$(sha_of $ENGINE)" = "$ENGINE_SHA" ] || refuse "engine $ENGINE sha $(sha_of $ENGINE) != pinned $ENGINE_SHA"
if [ "$GPU_INDEX" = 0 ]; then
  [ -f $Q/d1.sealed ] || refuse "GPU0 is D1's until D1 seals ($Q/d1.sealed absent)"
fi
[ "$KERNEL" = auto ] && awk -v c="$GPU_CC" 'BEGIN{exit !(c+0 < 8.0)}' && \
  refuse "the caches are locked to the fla kernel and GPU$GPU_INDEX is sm$GPU_CC; resume on GPU0"

wait_gates() {  # every gate, re-read each minute, until all hold at once
  local held=0 why
  while :; do
    why=""
    [ "${GPU_INDEX}" = 0 ] && d1_container_live && why="$why d1_container_running"
    [ "$(used_mib)" -ge $CARD_FREE_MIB ] && why="$why card_busy_$(used_mib)MiB"
    [ "$(mem_avail_gib)" -lt $RAM_MIN_GIB ] && why="$why MemAvailable_$(mem_avail_gib)GiB"
    k3_live && why="$why k3_pass_live"
    [ "$(root_gb)" -lt $ROOT_MIN_GB ] && why="$why root_$(root_gb)GB"
    [ "$(lexar_gb)" -lt $LEXAR_MIN_GB ] && why="$why lexar_$(lexar_gb)GB"
    [ -z "$why" ] && return 0
    [ $held -ge $WAIT_MAX_S ] && { status "job=j6-vl gpu=$GPU_INDEX state=gave_up waited_s=$held reason=$why"; exit 4; }
    [ $((held % 1800)) -eq 0 ] && status "job=j6-vl gpu=$GPU_INDEX event=waiting reason=$why"
    sleep 60; held=$((held + 60))
  done
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_UUID QTEXT_TPS_PROBE=$ENGINE
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$OC/distill/tools CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
ARGS=(build-vl --model-dir $M27 --device cuda:0 --dtype bfloat16 --attn-implementation sdpa
      --linear-attn-kernel $KERNEL --digest-threads 8 --teacher q38-27bu
      --corpus $CORPUS/vision_in.jsonl --corpus $CORPUS/video_in.jsonl --out-root $OUT
      --micro-batch 8 --max-batch-tokens 32768 --batch-order length --rows-per-shard 64 --head-precision fp32
      --chat-template-kwargs '{"enable_thinking": false}' --reference-boxes qwen1000)

run_tool() {  # run_tool <name> <extra args...>; one automatic resume after exit 1/143, exit 2 final
  local name=$1 rc attempt; shift
  for attempt in 1 2; do
    wait_gates
    [ "$(sha_of $QT_FROZEN)" = "$QT_SHA" ] || refuse "frozen tool changed under the runner"
    OTHER=$(live_build_vl); [ -z "$OTHER" ] || refuse "a qtext build-vl started while this runner waited: $OTHER"
    echo "[$(stamp)] $name attempt $attempt on GPU$GPU_INDEX ($GPU_NAME) MemAvailable $(mem_avail_gib) GiB" >> $LOG
    $PY $QT_FROZEN "${ARGS[@]}" "$@" --allocator-receipt $ALLOC/j6-qtext-vl-gpu$GPU_INDEX-$name.json \
      --purpose "j6 ($name): T(i) vision_in + video_in answer caches, Q38 bf16 top-64, fp32 head, grid boxes" >> $LOG 2>&1
    rc=$?
    echo "[$(stamp)] $name attempt $attempt exit $rc" >> $LOG
    if [ $rc -eq 0 ] || [ $rc -eq 2 ]; then break; fi
    sleep 60
  done
  return $rc
}

echo "[$(stamp)] j6 runner GPU$GPU_INDEX" >> $LOG; gate_report >> $LOG
status "job=j6-vl gpu=$GPU_INDEX state=started runner=\"$J6_VERSION\" lock=\"$LOCK_HOW\" kernel=$KERNEL tool_sha=${QT_SHA:0:16} out=$OUT"
run_tool first --max-new-shards 1
rc=$?
status "job=j6-vl-first gpu=$GPU_INDEX state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc $(grep 'shard-[0-9]*: rows' $LOG | tail -1 | cut -c1-300)"
[ $rc -eq 0 ] || exit $rc
run_tool all
rc=$?
for c in $(grep -o "SEALED [^ ]*manifest.json" $LOG | sed 's/^SEALED //; s#/manifest.json##' | sort -u); do
  $PY $QT_FROZEN verify --cache-dir $c >> $LOG 2>&1 || rc=2
done
[ $rc -eq 0 ] && mark_done
status "job=j6-vl gpu=$GPU_INDEX state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc sealed=$(grep -c '^\[.*SEALED' $LOG) $(grep -E 'SEALED|verify ' $LOG | tail -4 | tr '\n' ' ' | cut -c1-700)"
exit $rc
