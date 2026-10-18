#!/usr/bin/env bash
# x3_queue.sh -- lane fold-research (Claude, research; L-FOLD stays Codex's), pilot X3 on the Qwen3.5-0.8B ladder.
# Run detached:  setsid nohup bash x3_queue.sh >> $X/queue.log 2>&1 < /dev/null &
#
# WAITS, NEVER COMPETES (reconciled plan 7.4).  It starts a GPU stage only when ALL hold at that moment:
#   - no lane holds GPU3 in $Q/p40.claims (per lane, the LAST line naming GPU3 decides -- "<utc> GPU3 lane=..",
#     "<utc> CLAIM GPU1,GPU3 .." and "<utc> RELEASE GPU3 .." are all read; a line saying released / paused / stopped /
#     done frees it; a claim whose pid= is dead is logged as stale and ignored);
#   - GPU3 holds < 2,000 MiB (nvidia-smi), so an unannounced process also blocks it;
#   - $Q/gpu3.lock can be flocked (1.1.0: created if missing and HELD while the claim is, so d1_seal_queue.sh's
#     lock_free probe sees the card taken);
#   - MemAvailable >= NEED_GIB (default 80 = the K3 fold-identity floor 70 GiB + this job's ~10 GiB host peak: 5.3 GiB measured for one 0.8B load on CPU, two models in the trainer;
#     critique fix 7: the 40 GiB background gate is not enough while a K3 pass is in its RAM phase).
# Then it appends its claim line, runs the stages below (each idempotent: a stage whose output exists is skipped),
# and appends a RELEASE line on every exit path.  1.1.0 claim / release lines are "<utc> CLAIM GPU3 owner=claude
# lane=fold-research job=x3-08b pid=.." / "<utc> RELEASE GPU3 ... released (why)": d1_seal_lib.claim_state (the seal
# queue's p40-free) reads ANY "released" word as a release, and the 1.0.0 claim line said "claim released within
# ~3 min" in its free text, so the seal queue saw GPU3 as free while this job held it.
#
# PREEMPTIBLE.  kill -TERM <queue pid> (named in the claim line): the running trainer finishes its step, saves a
# resume point and exits 75; an eval or build stage is stopped (it reruns from scratch, minutes).  The claim is
# released and the queue goes back to waiting.  touch $X/STOP_QUEUE stops it for good at the next boundary.
# x3_v1_yield_watchdog.py does both when a v1 lane waits on RAM or on GPU3, and relaunches this queue later.
#
# Stages (outputs under $X = /mnt/lexar/omni-campaign/research/fold/x3-08b):
#   build   qf_build (unchanged, via qf_build_p40.py) C2 --mlp-keep 896 (= 3,584 / 4, the C2K2304 recipe at 0.8B)
#           on omni-evidence-v1 (0.8B and 4B share its tokenizer, sha checked by qf_build)
#   eval    parent, c2k896 (the baseline every rung is paired against)
#   kd      arm kd, rungs 0.25/0.5/1/2M; each rung evaluated + graded + paired as soon as the arm ends
#   opd     arm opd, same rungs
#   kd1m-opd  arm opd initialised from kd/rung-1000000, rungs 0.25/0.5/1M (totals 1.25/1.5/2M)
#   summary x3_summary.py -> $X/SUMMARY.{json,md}
set -uo pipefail
VERSION=1.1.0
# 1.1.0 (2026-10-15): CLAIM / RELEASE claim lines without "released" in a live claim's free text (the seal queue's
#   parser read 1.0.0's claim as a release); RELEASE lines of other lanes are parsed; $Q/gpu3.lock created and held
#   with the claim; run_stage starts nothing once a TERM is pending (1.0.0 ran the next stage to its end when the
#   TERM landed during a CPU grade); X3Q_PY / LOCK3 overrides for the tests.  Installed by rename: a running 1.0.0
#   keeps its inode.
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
R=/mnt/lexar/omni-campaign/research/fold
T=$R/tools
X=${X3Q_ROOT:-$R/x3-08b}
PY=${X3Q_PY:-/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python}
P08=/home/cereal/models/Qwen3.5-0.8B
QF=$OC/qwen-fold
EVID=$QF/evidence/omni-evidence-v1.json
CHILD=$X/children/QF08-C2K896
PACK=$X/evalpack/pack.json
GROUPS_FILE=$X/data/training_groups.json
GPU=${GPU:-3}
LANE=fold-research
NEED_GIB=${NEED_GIB:-80}
POLL_S=${POLL_S:-120}
CLAIMS=${CLAIMS:-$Q/p40.claims}
LOCK3=${LOCK3:-$Q/gpu$GPU.lock}
NVSMI=${NVSMI:-nvidia-smi}
STATUS=$X/queue.status
mkdir -p $X/eval $X/arms $X/children
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "$(stamp) $*" | tee -a $STATUS; }

CHILD_PID=""
PREEMPT=0
HOLDING=0
on_term() { PREEMPT=1; say "TERM received: preempting (child pid ${CHILD_PID:-none})"; [ -n "$CHILD_PID" ] && kill -TERM "$CHILD_PID" 2>/dev/null; }
trap on_term TERM
release() {
  if [ $HOLDING = 1 ]; then
    echo "$(stamp) RELEASE GPU$GPU owner=claude lane=$LANE job=x3-08b released ($1)" >> $CLAIMS
    HOLDING=0
    exec 9>&- 2>/dev/null || true
    say "released GPU$GPU ($1)"
  fi
}
trap 'release "queue exit"' EXIT

gpu3_holders() {  # one line per lane whose last GPU$GPU line is a live claim: "<lane>\t<pid or ->\t<line>"
  awk -v g="GPU$GPU" '
    { cards = ($2 == "CLAIM" || $2 == "RELEASE") ? $3 : $2; n = split(cards, a, ","); hit = 0
      for (i = 1; i <= n; i++) if (a[i] == g) hit = 1
      if (!hit) next
      lane = "?"; pid = "-"
      for (i = 1; i <= NF; i++) { if ($i ~ /^lane=/) lane = substr($i, 6); if ($i ~ /^pid=[0-9]+$/) pid = substr($i, 5) }
      last[lane] = $0; lpid[lane] = pid }
    END { for (l in last) if (last[l] !~ /released|paused|stopped|done/) print l "\t" lpid[l] "\t" last[l] }' "$CLAIMS" 2>/dev/null
}

card_used() { $NVSMI --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1==g{print $2}'; }
mem_gib() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }

gpu_free() {  # prints the reason it is NOT free, or nothing
  local live="" lane pid line used
  while IFS=$'\t' read -r lane pid line; do
    [ -z "$lane" ] && continue
    [ "$lane" = "$LANE" ] && continue
    if [ "$pid" != "-" ] && [ ! -d /proc/$pid ]; then
      echo "$(stamp) note: stale GPU$GPU claim (pid $pid dead): $line" >> $X/queue.stale
      continue
    fi
    live="$lane"
  done < <(gpu3_holders)
  [ -n "$live" ] && { echo "claimed by lane $live"; return; }
  used=$(card_used)
  [ "${used:-99999}" -lt 2000 ] 2>/dev/null || { echo "GPU$GPU holds ${used:-?} MiB"; return; }
  [ "$(mem_gib)" -ge "$NEED_GIB" ] || { echo "MemAvailable $(mem_gib) GiB < $NEED_GIB"; return; }
}

acquire() {
  local why last="" waited=0
  while :; do
    [ -f $X/STOP_QUEUE ] && { say "STOP_QUEUE present: exiting"; exit 0; }
    why=$(gpu_free)
    if [ -z "$why" ]; then
      exec 9>>$LOCK3   # created if missing; held (fd 9, inherited by the stages) until release()
      if ! flock -n 9; then why="$(basename $LOCK3) busy"; exec 9>&-; fi
      if [ -z "$why" ]; then
        # no word "released" / "RELEASE" / "paused" / "stopped" / "done" in a live claim: every parser reads them as a release
        echo "$(stamp) CLAIM GPU$GPU owner=claude lane=$LANE job=x3-08b pid=$$ log=$X/queue.log preempt: kill -TERM $$ (trainer checkpoints, the claim ends within ~3 min, then it re-queues); stop for good: touch $X/STOP_QUEUE" >> $CLAIMS
        HOLDING=1
        say "CLAIMED GPU$GPU after ${waited}s wait; MemAvailable $(mem_gib) GiB, card $(card_used) MiB"
        return 0
      fi
    fi
    if [ "$why" != "$last" ] || [ $((waited % 1800)) -lt $POLL_S ]; then say "WAIT GPU$GPU: $why"; last="$why"; fi
    sleep $POLL_S & wait $! ; waited=$((waited + POLL_S))
    [ $PREEMPT = 1 ] && PREEMPT=0
  done
}

run_stage() {  # run_stage <name> <cmd...>: 0 ok, 75 preempted, other = failure
  local name=$1; shift
  # 1.1.0: a TERM that arrived during a CPU grading step (no CHILD_PID to forward it to) must not start the next stage
  if [ $PREEMPT = 1 ]; then say "PREEMPTED before $name"; return 75; fi
  say "START $name"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 \
    nice -n 15 "$@" >> $X/logs/$name.log 2>&1 &
  CHILD_PID=$!
  local rc
  while :; do
    wait $CHILD_PID; rc=$?
    kill -0 $CHILD_PID 2>/dev/null || break
  done
  CHILD_PID=""
  if [ $PREEMPT = 1 ]; then say "PREEMPTED in $name (rc=$rc)"; return 75; fi
  say "END $name rc=$rc $(grep -E '^\{|RUNG|ARM DONE' $X/logs/$name.log | tail -1 | cut -c1-300)"
  return $rc
}

cpu() {  # CPU-only helper steps (grading) run without the card
  CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 nice -n 15 "$@" >> $X/logs/grade.log 2>&1
}

eval_one() {  # eval_one <label> [overlay]
  local label=$1 overlay=${2:-} model=$CHILD dir=$X/eval/$1
  [ "$label" = parent ] && model=$P08
  if [ ! -f $dir/run.json ]; then
    run_stage eval-$label $PY $T/fold_eval.py run --pack $PACK --model-dir $model ${overlay:+--overlay $overlay} \
      --label $label --device cuda:0 --dtype float32 --batch 16 --out $dir || return $?
  fi
  if [ ! -f $dir/grade.json ]; then
    cpu $PY $T/fold_eval.py grade --pack $PACK --run $dir --parent-run $X/eval/parent --training-groups $GROUPS_FILE \
      --out $dir/grade.json || { say "grade $label FAILED"; return 1; }
    say "GRADED $label $(tail -1 $X/logs/grade.log | cut -c1-400)"
  fi
  if [ "$label" != parent ] && [ "$label" != c2k896 ] && [ ! -f $dir/pair_vs_c2k896.json ]; then
    cpu $PY $T/fold_eval.py pair --base $X/eval/c2k896/grade.json --cand $dir/grade.json --out $dir/pair_vs_c2k896.json \
      || { say "pair $label FAILED"; return 1; }
    say "PAIRED $label $(tail -1 $X/logs/grade.log | cut -c1-400)"
  fi
  return 0
}

arm() {  # arm <name> <rungs> [init overlay]
  local name=$1 rungs=$2 init=${3:-}
  if [ ! -f $X/arms/$name/ARM_DONE.json ]; then
    local kind=opd; [ "$name" = kd ] && kind=kd
    run_stage train-$name $PY $T/x3_ladder.py train --arm $kind --parent $P08 --child $CHILD --data $X/data \
      ${init:+--init-overlay $init} --rungs $rungs --out $X/arms/$name || return $?
  fi
  local r
  for r in ${rungs//,/ }; do
    eval_one $name-$r $X/arms/$name/rung-$r.safetensors || return $?
  done
}

pipeline() {
  if [ ! -f $CHILD/qf-build-receipt.json ]; then
    if [ -d $CHILD ]; then mv $CHILD $CHILD.partial-$(date -u +%Y%m%dT%H%M%SZ); say "moved a partial child aside"; fi
    say "qf_build sha $(sha256sum $QF/qf_build.py | cut -c1-16) qf_mlp sha $(sha256sum $QF/qf_mlp.py | cut -c1-16)"
    run_stage build $PY $T/qf_build_p40.py $QF --parent $P08 --out $CHILD --evidence $EVID --tag C2K896 \
      --parent-code Q35-0.8B --device cuda:0 --dtype float32 --mlp-keep 896 || return $?
  fi
  eval_one parent || return $?
  eval_one c2k896 || return $?
  arm kd 250000,500000,1000000,2000000 || return $?
  arm opd 250000,500000,1000000,2000000 || return $?
  arm kd1m-opd 250000,500000,1000000 $X/arms/kd/rung-1000000.safetensors || return $?
  cpu $PY $T/x3_summary.py --root $X || return 1
  return 0
}

mkdir -p $X/logs
[ "${X3Q_SOURCE_ONLY:-0}" = 1 ] && return 0     # test_x3_queue.sh sources the functions only
say "queue x3_queue.sh $VERSION pid $$ start; tools: $(cd $T && sha256sum fold_common.py fold_eval.py x3_ladder.py x3_summary.py qf_build_p40.py | awk '{printf "%s:%s ", $2, substr($1,1,12)}')"
while :; do
  acquire
  pipeline; rc=$?
  if [ $rc = 0 ]; then release "pipeline complete"; say "DONE: $X/SUMMARY.md"; exit 0; fi
  if [ $rc = 75 ]; then release "preempted"; PREEMPT=0; continue; fi
  release "stage failed rc=$rc"; say "FAILED rc=$rc; see $X/logs"; exit $rc
done
