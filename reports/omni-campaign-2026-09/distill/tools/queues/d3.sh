#!/usr/bin/env bash
# Lane d3 queue (wave 2, 2026-10-13, relaunch "b" for d3_hybrid_module 1.0.2): E1, the D3 hybrid module.
#
# The first queue (2026-10-13, tool 1.0.0, pid 1053456) ran tests, synthetic and the OMNI1 parity (PASS)
# and was stopped in its GPU3 wait, before any GPU job, for the 1.0.2 review fixes (D3c head-grad
# aliasing, vacuous D3d rate cell, no host-RAM / Qwen-streamer gate).  Its logs and receipts stay as
# they are (tag 2026-10-13); this queue writes tag 2026-10-13b.
#
#   1 tests          CPU   pytest test_d3_hybrid_module.py, lab venv (torch 2.11)
#   2 tests-pascal   CPU   the same tests in the Pascal venv the GPU3 jobs run in (torch 2.6.0)
#   3 synthetic      CPU   tiny child: hybrid == ChimeraOmniModel.forward, streamed grads == autograd, D3a-d train
#   4 parity         CPU   OMNI1, fp32, every unit's held S5 segment (trimmed): hybrid vs the Chimera reference
#                          (s7_omni1_check); OMNI1 stat before/after (read-only proof).  RAM/streamer gate first.
#   5 rate-d3b       GPU3  P40, Pascal venv, fp32, lr = 0: x_d recapture + D3b fwd+bwd per unit, real OMNI1 weights
#   6 rate-d3d       GPU3  P40, same, full-depth D3d-shaped step on the QVL32 segment + 8 answer-text tokens
#
# Gating: parity runs only if both test runs and synthetic pass; the GPU3 jobs only if parity PASSES.
# Every GPU job waits, INSIDE this script, for step 1's S8 review (pid 917640) to exit, for lane p40 to
# release GPU3 (its queues/p40.status), and for GPU3 to show < 2000 MiB with no compute process.
# Host RAM rule (wave-2 brief): the K3 pass (97-130 GB) and at most ONE Qwen streamer (56 GB) run
# together, nothing else large.  The OMNI1 parity and both rate cells ARE Qwen streamers (they stream
# the 27B trunk), so before each of them this script also waits for MemAvailable >= $MIN_AVAIL_GIB GiB
# and for NO other Qwen-streamer process ($STREAMER_RE; the K3 cache is not one), and logs the wait.
# Allocator receipts under composite/receipts/allocator/.  CPU jobs: nice 15, <= 16 threads.  One status
# line per job is appended to queues/d3.status.  Nothing here writes a weight: OMNI1 is read through mmap
# views and the sha-verified reference loader, rate cells run at lr = 0 and assert parameters unchanged.
set -u
OC=/home/cereal/omni-campaign
EF=$OC/import/rq27b/tree_ef1ffb9
TOOLS=$OC/distill/tools
Q=$OC/distill/queues
STATUS=$Q/d3.status
D3=$OC/distill/d3
LOGS=$D3/logs
RC=$D3/receipts
PY=/home/cereal/model-surgery-lab/venv/bin/python
PYP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
OMNI1=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
PARENT=/home/cereal/models/Qwen3.8-27B-Uncensored
TAP=$OC/composite/s7/tap_rows_s5
ALLOC=$OC/composite/receipts/allocator
STEP1_PID=917640
GPU=3
TAG=2026-10-13b
MIN_AVAIL_GIB=80
# processes that stream a 27B Qwen trunk (path or "python <name>.py"; test_*.py runs do not match)
STREAMER_RE='(/|python[0-9.]* )(qtext_teacher_cache|streamed_kd_trainer|d3_hybrid_module|p11_needle_engine|run_omni_frozen_review|run_review_finite_guard|s7_omni1_check)\.py( |$)'
WAIT_CAP=172800
mkdir -p "$LOGS" "$RC" "$ALLOC"

exec 9> "$Q/d3.lock"
if ! flock -n 9; then echo "another d3 queue holds $Q/d3.lock"; exit 5; fi

status() { echo "$(date -u +%FT%TZ) d3/$*" >> "$STATUS"; }
cpu_env() {
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
  export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src:$TOOLS OC=$OC
}
tool_sha() { sha256sum "$TOOLS/d3_hybrid_module.py" | cut -c1-16; }
tool_version() { grep -m1 '^TOOL_VERSION' "$TOOLS/d3_hybrid_module.py" | cut -d'"' -f2; }

status "queue start pid=$$ tag=$TAG tool_sha=$(tool_sha) version=$(tool_version)"

# ---------------------------------------------------------------- RAM / Qwen-streamer gate
avail_gib() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
streamers() { pgrep -af -- "$STREAMER_RE" | awk '{p=$1; $1=""; print p":"substr($0, 2, 90)}' | tr '\n' ';'; }
ram_ok() { [ "$(avail_gib)" -ge "$MIN_AVAIL_GIB" ] && [ -z "$(streamers)" ]; }
wait_for_ram() {  # $1 = job name
  local waited=0 next_note=0
  status "$1 ram-wait start (needs MemAvailable >= ${MIN_AVAIL_GIB} GiB and no other Qwen streamer)"
  while true; do
    if ram_ok; then
      status "$1 ram-wait done after ${waited}s: MemAvailable $(avail_gib) GiB, streamers none"
      return 0
    fi
    if [ $waited -ge $next_note ]; then
      status "$1 ram-wait waiting ${waited}s: MemAvailable $(avail_gib) GiB streamers=\"$(streamers | cut -c1-400)\""
      next_note=$((next_note + 3600))
    fi
    if [ $waited -ge $WAIT_CAP ]; then
      status "$1 ram-wait gave up after 48 h"
      return 1
    fi
    sleep 60; waited=$((waited + 60))
  done
}

# ---------------------------------------------------------------- 1 tests (CPU, lab venv)
cpu_env
LOG=$LOGS/tests_$TAG.log
status "tests start log=$LOG"
( cd "$TOOLS" && nice -n 15 timeout 3600 $PY -m pytest -q test_d3_hybrid_module.py -p no:cacheprovider ) > "$LOG" 2>&1
RC_TESTS=$?
status "tests done exit=$RC_TESTS log=$LOG line=\"$(grep -E '[0-9]+ (passed|failed)' "$LOG" | tail -1)\""

# ---------------------------------------------------------------- 2 tests (CPU, Pascal venv)
LOG=$LOGS/tests_pascal_venv_$TAG.log
status "tests-pascal start log=$LOG"
( cd "$TOOLS" && nice -n 15 timeout 3600 $PYP -m pytest -q test_d3_hybrid_module.py -p no:cacheprovider ) > "$LOG" 2>&1
RC_TESTS_P=$?
status "tests-pascal done exit=$RC_TESTS_P log=$LOG line=\"$(grep -E '[0-9]+ (passed|failed)' "$LOG" | tail -1)\""

# ---------------------------------------------------------------- 3 synthetic (CPU)
LOG=$LOGS/synthetic_$TAG.log
REC=$RC/synthetic_$TAG.json
status "synthetic start log=$LOG"
( cd "$TOOLS" && nice -n 15 timeout 3600 $PY d3_hybrid_module.py synthetic --receipt "$REC" --steps 40 ) > "$LOG" 2>&1
RC_SYN=$?
status "synthetic done exit=$RC_SYN receipt=$REC line=\"$(grep -E 'D3 SYNTHETIC' "$LOG" | tail -1 | cut -c1-160)\""

# ---------------------------------------------------------------- 4 parity on OMNI1 (CPU fp32)
if [ "$RC_TESTS" -ne 0 ] || [ "$RC_TESTS_P" -ne 0 ] || [ "$RC_SYN" -ne 0 ]; then
  status "parity skipped: tests exit=$RC_TESTS tests-pascal exit=$RC_TESTS_P synthetic exit=$RC_SYN"
  RC_PAR=99
elif ! wait_for_ram parity; then
  status "parity skipped: RAM/streamer gate never opened"
  RC_PAR=98
else
  LOG=$LOGS/parity_omni1_$TAG.log
  REC=$RC/parity_omni1_cpu_fp32_$TAG.json
  stat -c '%n %s %Y' "$OMNI1"/* > "$D3/omni1_stat_before_$TAG.txt"
  status "parity start log=$LOG"
  ( cd "$TOOLS" && nice -n 15 /usr/bin/time -v timeout 21600 $PY d3_hybrid_module.py parity --omni1 "$OMNI1" \
      --parent-config "$PARENT" --tap-rows "$TAP" --receipt "$REC" --max-rows-per-type 24 --threads 16 ) > "$LOG" 2>&1
  RC_PAR=$?
  stat -c '%n %s %Y' "$OMNI1"/* > "$D3/omni1_stat_after_$TAG.txt"
  if cmp -s "$D3/omni1_stat_before_$TAG.txt" "$D3/omni1_stat_after_$TAG.txt"; then SAME=unchanged; else SAME=CHANGED; fi
  status "parity done exit=$RC_PAR omni1_stat=$SAME receipt=$REC rss=\"$(grep 'Maximum resident' "$LOG" | sed 's/^[[:space:]]*//')\" line=\"$(grep -E 'D3 PARITY' "$LOG" | tail -1 | cut -c1-240)\""
fi

# ---------------------------------------------------------------- GPU3 gate (inside the queue)
gpu3_released_by_p40() {
  local st=$Q/p40.status
  [ -f "$st" ] || return 1
  # lane p40's own markers (p40.sh gpu3_chain / main): the exact line "GPU3 FREE", or "QUEUE end" with
  # no p40.sh left running.  Loose patterns are wrong here: its "WAIT gpu3: ... qvl_done=0" line says "done".
  grep -qE '^[0-9TZ:-]+ GPU3 FREE$' "$st" && return 0
  if grep -qE '^[0-9TZ:-]+ QUEUE end$' "$st" && ! pgrep -f "queues/p40.sh" > /dev/null; then return 0; fi
  return 1
}
gpu3_idle() {
  local used uuid apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid")
  [ "${used:-99999}" -lt 2000 ] && [ "$apps" -eq 0 ]
}
wait_for_gpu3() {  # $1 = job name; GPU3 free AND the RAM/streamer gate, checked together
  local waited=0 next_note=0
  status "$1 gpu3-wait start (needs pid $STEP1_PID gone, lane p40 release, GPU$GPU < 2000 MiB idle, MemAvailable >= ${MIN_AVAIL_GIB} GiB, no other Qwen streamer)"
  while true; do
    if ! kill -0 $STEP1_PID 2> /dev/null && gpu3_released_by_p40 && gpu3_idle && ram_ok; then
      status "$1 gpu3-wait done after ${waited}s: MemAvailable $(avail_gib) GiB, streamers none"
      return 0
    fi
    if [ $waited -ge $next_note ]; then
      local s1=running; kill -0 $STEP1_PID 2> /dev/null || s1=exited
      local p4=held; gpu3_released_by_p40 && p4=released
      status "$1 gpu3-wait waiting ${waited}s: step1=$s1 p40=$p4 gpu3_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')MiB MemAvailable=$(avail_gib)GiB streamers=\"$(streamers | cut -c1-400)\""
      next_note=$((next_note + 3600))
    fi
    if [ $waited -ge $WAIT_CAP ]; then
      status "$1 gpu3-wait gave up after 48 h"
      return 1
    fi
    sleep 120; waited=$((waited + 120))
  done
}
alloc_receipt() {  # job used_mib started ended exit log argv...
  local job=$1 used=$2 started=$3 ended=$4 code=$5 log=$6; shift 6
  ALLOC_JOB=$job ALLOC_USED=$used ALLOC_START=$started ALLOC_END=$ended ALLOC_EXIT=$code ALLOC_LOG=$log \
  ALLOC_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $GPU | tr -d ' ') ALLOC_AVAIL=$(avail_gib) \
  ALLOC_CARD="$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)" ALLOC_SHA=$(sha256sum "$TOOLS/d3_hybrid_module.py" | cut -d' ' -f1) \
  $PY - "$@" <<'EOF'
import json, os, sys
e = os.environ
body = {"argv": sys.argv[1:], "by": "claude d3 lane", "card": e["ALLOC_CARD"], "cpu_only": False,
        "started_utc": e["ALLOC_START"], "ended_utc": e["ALLOC_END"] or None,
        "exit_code": None if e["ALLOC_EXIT"] == "" else int(e["ALLOC_EXIT"]), "gpu": 3,
        "lane": "d3: E1 hybrid D3 module, lr=0 rate cells", "log": e["ALLOC_LOG"], "runtime": "venv-pascal-cu118-torch260",
        "tool": "d3_hybrid_module", "tool_sha256": e["ALLOC_SHA"], "used_mib_at_start": int(e["ALLOC_USED"]), "uuid": e["ALLOC_UUID"],
        "host_mem_available_gib_now": int(e["ALLOC_AVAIL"]), "ram_rule": "K3 pass + at most one Qwen streamer; no other streamer at start"}
path = os.path.join("/home/cereal/omni-campaign/composite/receipts/allocator", e["ALLOC_JOB"] + ".json")
with open(path + ".tmp", "w") as fh:
    json.dump(body, fh, indent=1)
os.replace(path + ".tmp", path)
EOF
}
gpu_job() {  # job log receipt args...
  local job=$1 log=$2; shift 2
  local used started ended code
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -d ' ')
  started=$(date -u +%FT%TZ)
  alloc_receipt "$job" "$used" "$started" "" "" "$log" "$@"
  status "$job start gpu=$GPU used_mib=$used log=$log"
  ( cd "$TOOLS" && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src:$TOOLS OC=$OC PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
      nice -n 15 /usr/bin/time -v timeout 14400 $PYP d3_hybrid_module.py "$@" ) > "$log" 2>&1
  code=$?
  ended=$(date -u +%FT%TZ)
  alloc_receipt "$job" "$used" "$started" "$ended" "$code" "$log" "$@"
  status "$job done exit=$code log=$log rss=\"$(grep 'Maximum resident' "$log" | sed 's/^[[:space:]]*//')\" lines=\"$(grep -E '^\[d3_hybrid_module.*(RATE|D3 RATE)' "$log" | sed 's/^\[d3_hybrid_module [0-9:]*\] //' | tr '\n' '|' | cut -c1-600)\""
  return $code
}

if [ "$RC_PAR" -ne 0 ]; then
  status "gpu3 jobs skipped: parity exit=$RC_PAR (a parity failure needs judgement before any GPU spend)"
else
  if wait_for_gpu3 d3-rate-d3b-p40-gpu3; then
    gpu_job d3-rate-d3b-p40-gpu3 "$LOGS/rate_d3b_p40_$TAG.log" rate --omni1 "$OMNI1" --parent-config "$PARENT" \
      --tap-rows "$TAP" --device cuda:0 --dtype float32 --kernels torch --stage d3b \
      --units hy3d_dit_L10,qimg_mmdit_L30_img,ace_dit_L12,h3_dit_L25_curve6 --steps 3 \
      --receipt "$RC/rate_d3b_p40_fp32_$TAG.json"
    # re-check GPU3 and the RAM/streamer gate: another streamer may have started during rate-d3b
    if wait_for_gpu3 d3-rate-d3d-p40-gpu3; then
      gpu_job d3-rate-d3d-p40-gpu3 "$LOGS/rate_d3d_p40_$TAG.log" rate --omni1 "$OMNI1" --parent-config "$PARENT" \
        --tap-rows "$TAP" --device cuda:0 --dtype float32 --kernels torch --stage d3d \
        --units qvl32_vit_L13 --steps 2 --receipt "$RC/rate_d3d_p40_fp32_$TAG.json"
    fi
    status "gpu3 released by d3 (rate cells finished)"
  fi
fi
status "queue end"
