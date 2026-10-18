#!/usr/bin/env bash
# Stage D wave 2, lane gpu0: GPU0 (RTX 3090) only.  Written 2026-10-13.
# Binds DISTILL_DESIGN_2026-10-12 (section 3 Q-text / Q-long / self-gen rescoring / vision-
# and video-in answers, S0 P2) and STAGE_D_RULINGS_2026-10-12.  One job at a time, in order:
#   j0  head probe   qtext 1.0.4 on 8 x 4,096 real rows at mb8, fp32 head vs model head:
#                    decides the head precision every cache below locks (finding L3)
#   j1  rescoring    self-gen 1,008,982 tokens (selfgen run fe1f0784): export-rescore, the
#                    independent target_mask (L2), build-text, verify
#   j2  P2           streamed_kd_trainer 1.3.0 on OMNI1 via P1, lr = 0: L4096 mb8 then
#                    L32768 mb1.  Targets: scratch Qwen3.5-0.8B caches (content is irrelevant
#                    at lr = 0; the trainer needs a sealed cache of the step's width)
#   j3  VL smoke     27B T(i) greedy answers on 2 images + 1 clip, for a human read
#   j4  Q-text       text-00000 one shard (the rate), then the 20 text files and the 4 K3
#                    passes (Qwen side), one sealed cache per corpus file
#   j5  Q-long       one 32,768-token document first (memory + rate), then all 31
#   j6  VL answers   vision_in (5,896 rows) then video_in (512 clips); runs only if
#                    $Q/gpu0.vl_go exists (written after the j3 read), else skipped
# Q-long runs before the VL answers (the brief listed VL first): D2 consumes Q-long, D3d
# consumes the VL caches, and D2 runs before D3.
# Every GPU job waits until step 1's S8 review (pid 917640) has exited and GPU0 holds
# < 2,000 MiB, and records $OC/composite/receipts/allocator/<job>.json.  One status line per
# job event goes to $Q/gpu0.status.  Caches: /mnt/lexar/omni-campaign/teacher_cache/qtext.
# Scratch: /mnt/lexar/omni-campaign/scratch/qtext-gpu0-2026-10-13 (declared transient).
# queue version: gpu0-2026-10-13b
#   b (09:35Z): qtext 1.0.5.  The 27B teacher is host-resident (one cold pass read 50 GB in
#   669 s = 48.97 tok/s: other lanes evict the page cache) and several corpora share one
#   teacher load and one digest pair, so j4 runs as three processes; j0 is done (head =
#   fp32, pinned in gpu0.head_decision.json by judgement); P2 cells cut to 4 and 3 steps.
set -u
LANE=gpu0
QUEUE_VERSION=gpu0-2026-10-13b
OC=/home/cereal/omni-campaign
D=$OC/distill/tools
Q=$OC/distill/queues
L=$Q/$LANE
STATUS=$Q/$LANE.status
ALLOC=$OC/composite/receipts/allocator
TPS=$OC/distill/tps
PY=/home/cereal/model-surgery-lab/venv/bin/python
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
M08=/home/cereal/models/Qwen3.5-0.8B
OMNI1=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
CORPUS=$OC/distill/corpus
SELFGEN=$OC/distill/selfgen/run-q38-27bu-q4km-2026-10-13
TC=/mnt/lexar/omni-campaign/teacher_cache/qtext
IN=$TC/inputs
SCR=/mnt/lexar/omni-campaign/scratch/qtext-gpu0-2026-10-13
WAIT_PID=917640
GPU_INDEX=0
CARD_FREE_MIB=2000
ROOT_MIN_GB=65      # root floor is 60 GB
LEXAR_MIN_GB=20
STAGES="${*:-j1 j2 j3 j4 j5 j6}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=$D CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
mkdir -p $L $SCR $IN $TPS $ALLOC

exec 9> $Q/$LANE.lock
if ! flock -n 9; then echo "another $LANE queue holds $Q/$LANE.lock"; exit 5; fi

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
status() { echo "$(stamp) $*" >> $STATUS; echo "[$(stamp)] STATUS $*"; }
used_mib() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU_INDEX '$1==g{print $2}'; }
GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU_INDEX '$1==g{print $2}')
export CUDA_VISIBLE_DEVICES=$GPU_UUID
QT="$PY $D/qtext_teacher_cache.py"

waited_for_s8=0
wait_gpu() {  # step 1's S8 review must be gone, then the card must be (nearly) empty
  local held=0
  if [ $waited_for_s8 -eq 0 ]; then
    while kill -0 $WAIT_PID 2>/dev/null; do
      [ $((held % 600)) -eq 0 ] && echo "[$(stamp)] waiting for pid $WAIT_PID (step 1 S8 review) to exit; GPU$GPU_INDEX $(used_mib) MiB"
      sleep 60; held=$((held + 60))
    done
    waited_for_s8=1
    status "lane=$LANE event=s8_pid_gone pid=$WAIT_PID waited_s=$held"
  fi
  held=0
  local u=$(used_mib)
  while [ "${u:-99999}" -ge $CARD_FREE_MIB ]; do
    if [ $held -ge 43200 ]; then status "lane=$LANE event=card_wait_gave_up gpu=$GPU_INDEX used_mib=$u"; exit 4; fi
    [ $((held % 600)) -eq 0 ] && echo "[$(stamp)] GPU$GPU_INDEX holds ${u} MiB (>= $CARD_FREE_MIB); waiting"
    sleep 60; held=$((held + 60)); u=$(used_mib)
  done
  echo "[$(stamp)] GPU$GPU_INDEX $u MiB in use: free for the next job"
}
disk_guard() {
  local held=0 r x
  while :; do
    r=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); x=$(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9)
    if [ "$r" -ge $ROOT_MIN_GB ] && [ "$x" -ge $LEXAR_MIN_GB ]; then return 0; fi
    if [ $held -ge 21600 ]; then status "lane=$LANE event=disk_guard_gave_up root_gb=$r lexar_gb=$x"; exit 6; fi
    [ $((held % 1800)) -eq 0 ] && status "lane=$LANE event=disk_wait root_gb=$r lexar_gb=$x (need $ROOT_MIN_GB / $LEXAR_MIN_GB)"
    sleep 300; held=$((held + 300))
  done
}
alloc_open() {  # alloc_open <name> <purpose>   (for jobs whose tool writes no allocator receipt itself)
  cat > $ALLOC/$1.json <<JSON
{"name":"$1","owner":"claude","lane":"$LANE","server":"ai","gpu_index":$GPU_INDEX,"gpu_uuids":["$GPU_UUID"],"card":"RTX 3090","purpose":"$2","python":"$PY","queue":"$Q/$LANE.sh $QUEUE_VERSION","started_utc":"$(stamp)","state":"running","pid":$$,"log":"$L/$1.log"}
JSON
}
alloc_close() { $PY - "$ALLOC/$1.json" "$2" <<'PY'
import json, sys, time
p = sys.argv[1]; d = json.load(open(p))
d.update({"ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "released (process exited)", "outcome": sys.argv[2]})
json.dump(d, open(p, "w"), indent=1)
PY
}
# A teacher build: qtext writes its own allocator receipt.  One automatic resume after a
# non-functional stop (OOM, a crash: exit 1/143); exit 2 is a functional failure and is final.
qt_build() {  # qt_build <job name> <log> <args...>
  local name=$1 log=$2; shift 2
  local rc attempt
  for attempt in 1 2; do
    wait_gpu; disk_guard
    echo "[$(stamp)] $name attempt $attempt" >> $log
    $QT "$@" --allocator-receipt $ALLOC/$LANE-$name.json >> $log 2>&1
    rc=$?
    echo "[$(stamp)] $name attempt $attempt exit $rc" >> $log
    if [ $rc -eq 0 ] || [ $rc -eq 2 ]; then break; fi
    sleep 60
  done
  return $rc
}
shard_lines() { grep -E "shard-[0-9]+: rows|SEALED|PARTIAL|REFUSED|FUNCTIONAL|verify " "$1" | tail -${2:-3}; }
COMMON27="--model-dir $M27 --device cuda:0 --dtype bfloat16 --attn-implementation sdpa --linear-attn-kernel auto --digest-threads 8"

echo "[$(stamp)] queue $QUEUE_VERSION lane $LANE GPU$GPU_INDEX $GPU_UUID stages: $STAGES"
sha256sum $D/qtext_teacher_cache.py $D/streamed_kd_trainer.py $D/p1_qwen35_reader.py $D/selfgen_sampler.py \
  $OC/distill/tps_probe.py $Q/$LANE.sh $Q/gpu0_selfgen_target_corpus.py
$QT --version
status "lane=$LANE event=queue_start version=$QUEUE_VERSION stages=\"$STAGES\" qtext_sha=$(sha256sum $D/qtext_teacher_cache.py | cut -c1-16) trainer_sha=$(sha256sum $D/streamed_kd_trainer.py | cut -c1-16)"

HEAD=fp32
[ -f $Q/$LANE.head_decision.json ] && HEAD=$($PY -c "import json;print(json.load(open('$Q/$LANE.head_decision.json'))['head_precision'])")

# ---------------------------------------------------------------- j0 head probe
job_j0() {
  local log=$L/j0-headprobe.log rc
  mkdir -p $SCR/headprobe
  status "job=j0-headprobe state=started"
  $PY - $CORPUS/text/text-00000.safetensors $SCR/headprobe/probe8.safetensors >> $log 2>&1 <<'PY'
import sys
from safetensors.torch import load_file, save_file
t = load_file(sys.argv[1])
save_file({"input_ids": t["input_ids"][:8].contiguous(), "real_lengths": t["real_lengths"][:8].contiguous()}, sys.argv[2])
print("probe corpus: text-00000 rows [0, 8) x", t["input_ids"].shape[1])
PY
  for H in fp32 model; do
    qt_build headprobe-$H $log build-text $COMMON27 --teacher headprobe-$H --corpus $SCR/headprobe/probe8.safetensors \
      --out-root $SCR/headprobe/cache --micro-batch 8 --rows-per-shard 8 --head-precision $H --teacher-digest stat \
      --purpose "gpu0 j0: head precision probe ($H) at mb8 L4096, scratch"
    rc=$?
    echo "HEADPROBE $H exit $rc" >> $log
  done
  $PY - $SCR/headprobe/cache $Q/$LANE.head_decision.json >> $log 2>&1 <<'PY'
import glob, json, sys, time
root, out = sys.argv[1], sys.argv[2]
arms = {}
for h in ("fp32", "model"):
    sessions = sorted(glob.glob(f"{root}/headprobe-{h}/*/sessions/session-*.json"))
    shards = sorted(glob.glob(f"{root}/headprobe-{h}/*/shard-*/shard.json"))
    if not sessions:
        arms[h] = {"ran": False}
        continue
    s = json.load(open(sessions[-1]))
    mem = (s.get("torch_memory") or {})
    peak = max((v.get("peak_reserved_gb") or 0) for v in mem.values()) if mem else None
    alloc = max((v.get("peak_allocated_gb") or 0) for v in mem.values()) if mem else None
    free = min((s.get("resources") or {}).get("device_minimum_free_gb", {"x": None}).values() or [None])
    rate = json.load(open(shards[-1]))["forward_tokens_per_second"] if shards else None
    secs = json.load(open(shards[-1]))["seconds"] if shards else None
    arms[h] = {"ran": True, "failure": s.get("failure"), "sealed": bool(glob.glob(f"{root}/headprobe-{h}/*/manifest.json")),
               "peak_reserved_gib": peak, "peak_allocated_gib": alloc, "device_minimum_free_gib": free,
               "batch_seconds": secs, "forward_tokens_per_second": rate, "head": (s.get("engine_facts") or {}).get("head"),
               "head_matmul": s.get("head_matmul")}
f = arms.get("fp32", {})
fits = bool(f.get("ran") and f.get("sealed") and not f.get("failure") and f.get("peak_reserved_gib") is not None
            and f["peak_reserved_gib"] <= 21.5 and (f.get("device_minimum_free_gib") or 0) >= 1.0)
decision = {"head_precision": "fp32" if fits else "model", "decided_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rule": "fp32 if the fp32 arm sealed with no failure, torch peak reserved <= 21.5 GiB and the device never had < 1.0 GiB free (mb8, L4096, 27B, one 3090); else model (the bf16 rounding is then stated in every cache_config.logit_precision)",
            "arms": arms}
json.dump(decision, open(out, "w"), indent=1)
print("HEAD DECISION", json.dumps(decision))
PY
  HEAD=$($PY -c "import json;print(json.load(open('$Q/$LANE.head_decision.json'))['head_precision'])")
  local line=$(grep "HEAD DECISION" $log | tail -1 | cut -c1-900)
  status "job=j0-headprobe state=done head=$HEAD decision=$Q/$LANE.head_decision.json"
  echo "$line"
}

# ---------------------------------------------------------------- j1 self-gen rescoring
job_j1() {
  local log=$L/j1-rescore.log rc
  status "job=j1-rescore state=started head=$HEAD"
  mkdir -p $IN/selfgen
  if [ ! -f $IN/selfgen/rescore_export.safetensors ]; then
    nice -n 15 $PY $D/selfgen_sampler.py export-rescore --out-dir $SELFGEN --dest $IN/selfgen/rescore_export.safetensors >> $log 2>&1
    rc=$?; echo "export-rescore exit $rc" >> $log
    [ $rc -eq 0 ] || { status "job=j1-rescore state=failed step=export-rescore rc=$rc"; return $rc; }
  fi
  if [ ! -f $IN/selfgen/rescore_target.safetensors ]; then
    DISTILL_TOOLS=$D nice -n 15 $PY $Q/gpu0_selfgen_target_corpus.py $SELFGEN $IN/selfgen/rescore_export.safetensors \
      $IN/selfgen/rescore_target.safetensors >> $log 2>&1
    rc=$?; echo "target corpus exit $rc" >> $log
    [ $rc -eq 0 ] || { status "job=j1-rescore state=failed step=target_corpus rc=$rc"; return $rc; }
  fi
  qt_build qtext-selfgen-rescore $log build-text $COMMON27 --teacher q38-27bu --corpus $IN/selfgen/rescore_target.safetensors \
    --out-root $TC --micro-batch 8 --max-batch-tokens 32768 --batch-order length --rows-per-shard 256 --head-precision $HEAD \
    --purpose "gpu0 j1: Q38 bf16 rescoring of the self-gen run fe1f0784 (1,008,982 response tokens), top-64"
  rc=$?
  local cache=$(grep -o "SEALED [^ ]*manifest.json" $log | tail -1 | sed 's/^SEALED //; s#/manifest.json##')
  if [ $rc -eq 0 ] && [ -n "$cache" ]; then
    $QT verify --cache-dir $cache >> $log 2>&1; rc=$?
  fi
  status "job=j1-rescore state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc cache=$cache $(shard_lines $log 2 | tr '\n' ' ' | cut -c1-600)"
  return $rc
}

# ---------------------------------------------------------------- j2 P2 (T(ii) rate cell)
p2_cell() {  # p2_cell <tag> <corpus> <seq> <mb> <steps> <rows_per_shard>
  local tag=$1 corpus=$2 seq=$3 mb=$4 steps=$5 per=$6 log=$L/j2-p2-$1.log rc cache name
  qt_build p2-cache-$tag $log build-text --model-dir $M08 --device cuda:0 --dtype bfloat16 --teacher p2-q35-08b \
    --corpus $corpus --out-root $SCR/p2/cache --micro-batch $mb --rows-per-shard $per --head-precision model \
    --teacher-digest stat --purpose "gpu0 j2: scratch Qwen3.5-0.8B targets for the P2 $tag cell (content irrelevant at lr=0)"
  rc=$?
  cache=$(grep -o "SEALED [^ ]*manifest.json" $log | tail -1 | sed 's/^SEALED //; s#/manifest.json##')
  [ $rc -eq 0 ] && [ -n "$cache" ] || { status "job=j2-p2-$tag state=failed step=scratch_cache rc=$rc"; return 1; }
  echo "{\"sources\": [{\"name\": \"p2-$tag\", \"kind\": \"text\", \"qtext_cache\": \"$cache\"}]}" > $SCR/p2/data_$tag.json
  name=$LANE-p2-trainer-$tag
  wait_gpu; disk_guard
  alloc_open $name "S0 P2: streamed_kd_trainer 1.3.0 rate cell on OMNI1 via P1, $tag, lr=0, bf16"
  (echo "== host before $(stamp)"; free -g; ps -eo pid,rss,etime,args --sort=-rss | head -6 | cut -c1-180) >> $log
  /usr/bin/time -v -o $L/j2-p2-$tag.time $PY $D/streamed_kd_trainer.py train --engine streamed \
    --p1-chimera $OMNI1 --p1-parent-config $M27 --devices cuda:0 --dtype bfloat16 --kernels auto \
    --data-config $SCR/p2/data_$tag.json --seq-len $seq --micro-batch $mb --total-steps $steps \
    --lr 0 --gain-lr 0 --warmup-steps 0 --safety-minutes 100000 --log-every 1 \
    --out $SCR/p2/run_$tag --safety-dir $SCR/p2/safety_$tag \
    --receipt $TPS/p2-trainer-v130-omni1-$tag-gpu0-2026-10-13.json >> $log 2>&1
  rc=$?
  (echo "== host after $(stamp)"; free -g) >> $log
  alloc_close $name "exit $rc"
  $PY - $log $L/j2-p2-$tag.time $TPS/p2-trainer-v130-omni1-$tag-gpu0-2026-10-13.json $tag $rc >> $log 2>&1 <<'PY'
import json, re, sys
log, timef, receipt, tag, rc = sys.argv[1:6]
steps = [json.loads(l[5:]) for l in open(log) if l.startswith("STEP ")]
rss = None
for line in open(timef):
    if "Maximum resident set size" in line:
        rss = round(int(line.split(":")[1]) / 1024**2, 2)
r = json.load(open(receipt)) if rc == "0" else {}
steady = [s["tok_per_s"] for s in steps[1:]]
mem = r.get("memory") or {}
out = {"cell": tag, "exit": int(rc), "steps": len(steps), "tok_per_s_by_step": [s["tok_per_s"] for s in steps],
       "steady_tok_per_s_mean_steps_2_on": round(sum(steady) / len(steady), 2) if steady else None,
       "seconds_by_step": [s["seconds"] for s in steps], "peak_host_rss_gib_time_v": rss, "device_memory": mem,
       "receipt": receipt if r else None, "trainable": (r.get("adapter_census") or {}).get("parameters_trainable"),
       "streamed_gb": r.get("streamed_gb"), "layer_load_seconds": r.get("layer_load_seconds"), "verdict": r.get("verdict")}
print("P2 RESULT " + json.dumps(out))
PY
  status "job=j2-p2-$tag state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc $(grep '^P2 RESULT' $log | tail -1 | cut -c1-700)"
  rm -rf $SCR/p2/run_$tag $SCR/p2/safety_$tag
  return $rc
}
job_j2() {
  mkdir -p $SCR/p2
  status "job=j2-p2 state=started"
  $PY - $CORPUS/text/text-00000.safetensors $CORPUS/long/long-00000.safetensors $SCR/p2 >> $L/j2-p2-prep.log 2>&1 <<'PY'
import sys
from safetensors.torch import load_file, save_file
text, long_, out = sys.argv[1:4]
t = load_file(text); l = load_file(long_)
save_file({"input_ids": t["input_ids"][:64].contiguous(), "real_lengths": t["real_lengths"][:64].contiguous()}, f"{out}/p2_L4096.safetensors")
save_file({"input_ids": l["input_ids"][:4].contiguous(), "real_lengths": l["real_lengths"][:4].contiguous()}, f"{out}/p2_L32768.safetensors")
print("p2 corpora: 64 x 4096 from text-00000, 4 x 32768 from long-00000; real lengths", l["real_lengths"][:4].tolist())
PY
  p2_cell L4096-mb8 $SCR/p2/p2_L4096.safetensors 4096 8 4 64
  p2_cell L32768-mb1 $SCR/p2/p2_L32768.safetensors 32768 1 3 4
  $PY - $L/j2-p2-L4096-mb8.log $L/j2-p2-L32768-mb1.log $TPS/p2-summary-gpu0-2026-10-13.json <<'PY'
import json, sys
cells = {}
for path in sys.argv[1:3]:
    lines = [l for l in open(path) if l.startswith("P2 RESULT ")]
    if lines:
        d = json.loads(lines[-1][len("P2 RESULT "):]); cells[d["cell"]] = d
summary = {"probe": "S0 P2 (DISTILL_DESIGN section 4): T(ii) streamed bf16 fwd+bwd rate on one 3090, lr = 0",
           "student": "OC02-F-Q38-27BU-OMNI1 via p1_qwen35_reader (zero-copy)", "engine_rule": "STAGE_D_RULINGS: >= 151 tok/s -> streamed path",
           "targets_note": "scratch Qwen3.5-0.8B Q-text caches: the targets' values do not enter the rate at lr = 0",
           "cells": cells}
json.dump(summary, open(sys.argv[3], "w"), indent=1)
print(json.dumps(summary)[:1500])
PY
  rm -rf $SCR/p2/cache
}

# ---------------------------------------------------------------- j3 VL smoke (human read)
job_j3() {
  local log=$L/j3-vlsmoke.log rc
  status "job=j3-vlsmoke state=started"
  [ -f $SCR/vlsmoke/smoke.jsonl ] || { status "job=j3-vlsmoke state=failed reason=no_smoke_jsonl"; return 1; }
  wait_gpu; disk_guard
  alloc_open $LANE-qtext-vlsmoke "27B T(i) smoke: greedy answers on 2 images + 1 clip (human read)"
  $QT smoke-vl $COMMON27 --corpus $SCR/vlsmoke/smoke.jsonl --rows 0,1,2 --max-new-tokens 80 --head-precision $HEAD \
    --chat-template-kwargs '{"enable_thinking": false}' --receipt $TPS/qtext-smoke-vl-27b-gpu0-v104-2026-10-13.json >> $log 2>&1
  rc=$?
  alloc_close $LANE-qtext-vlsmoke "exit $rc"
  status "job=j3-vlsmoke state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc receipt=$TPS/qtext-smoke-vl-27b-gpu0-v104-2026-10-13.json (read it; write $Q/gpu0.vl_go to release j6)"
  return $rc
}

# ---------------------------------------------------------------- j4 Q-text production
job_j4() {
  local rc g files
  status "job=j4-qtext state=started head=$HEAD groups=text-00..09,text-10..19,k3-pass-0..3 (one shared-teacher process each)"
  # The one-shard rate: the first shard line of the first group, posted as soon as it appears.
  ( log=$L/j4-qtext-g1.log; n=0
    until grep -q "shard-00000: rows" $log 2>/dev/null || [ $n -ge 2880 ]; do sleep 30; n=$((n + 1)); done
    grep -q "shard-00000: rows" $log 2>/dev/null && status "job=j4-qtext-oneshard state=done $(grep -m1 'shard-00000: rows' $log | cut -c1-400)" ) &
  for g in 1 2 3; do
    case $g in
      1) files=$(for i in $(seq -f "%05g" 0 9); do printf -- "--corpus $CORPUS/text/text-$i.safetensors "; done) ;;
      2) files=$(for i in $(seq -f "%05g" 10 19); do printf -- "--corpus $CORPUS/text/text-$i.safetensors "; done) ;;
      3) files=$(for p in 0 1 2 3; do printf -- "--corpus $CORPUS/k3/pass-$p/qwen_input_ids.safetensors "; done) ;;
    esac
    qt_build qtext-g$g $L/j4-qtext-g$g.log build-text $COMMON27 --teacher q38-27bu $files \
      --out-root $TC --micro-batch 8 --max-batch-tokens 32768 --batch-order length --rows-per-shard 64 --head-precision $HEAD \
      --purpose "gpu0 j4: Q-text production group $g (Q38 bf16 top-64, fp32 head, host-resident teacher)"
    rc=$?
    status "job=j4-qtext-g$g state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc sealed=$(grep -c '^\[.*SEALED' $L/j4-qtext-g$g.log) $(grep 'shard-[0-9]*: rows' $L/j4-qtext-g$g.log | tail -1 | cut -c1-300)"
    [ $rc -eq 0 ] || return $rc
  done
  status "job=j4-qtext state=done"
}

# ---------------------------------------------------------------- j5 Q-long
job_j5() {
  local log=$L/j5-qlong.log rc
  status "job=j5-qlong state=started head=$HEAD"
  qt_build qtext-long-first $log build-text $COMMON27 --teacher q38-27bu --corpus $CORPUS/long/long-00000.safetensors \
    --out-root $TC --micro-batch 1 --rows-per-shard 1 --head-precision $HEAD --max-new-shards 1 \
    --purpose "gpu0 j5: Q-long first 32,768-token document (memory + rate), mb1"
  rc=$?
  status "job=j5-qlong-first state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc $(shard_lines $log 2 | tr '\n' ' ' | cut -c1-500)"
  [ $rc -eq 0 ] || return $rc
  qt_build qtext-long $log build-text $COMMON27 --teacher q38-27bu --corpus $CORPUS/long/long-00000.safetensors \
    --out-root $TC --micro-batch 1 --rows-per-shard 1 --head-precision $HEAD \
    --purpose "gpu0 j5: Q-long production, 31 x 32,768, mb1"
  rc=$?
  status "job=j5-qlong state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc $(shard_lines $log 2 | tr '\n' ' ' | cut -c1-500)"
  return $rc
}

# ---------------------------------------------------------------- j6 vision- and video-in answers
job_j6() {
  local rc log=$L/j6-vl.log
  if [ ! -f $Q/$LANE.vl_go ]; then
    status "job=j6-vl state=skipped reason=no_go_file ($Q/$LANE.vl_go absent; j3 smoke not cleared)"
    return 0
  fi
  status "job=j6-vl state=started head=$HEAD go=$(head -c 300 $Q/$LANE.vl_go | tr '\n' ' ')"
  qt_build qtext-vl $log build-vl $COMMON27 --teacher q38-27bu \
    --corpus $CORPUS/vision_in/vision_in.jsonl --corpus $CORPUS/vision_in/video_in.jsonl \
    --out-root $TC --micro-batch 8 --max-batch-tokens 32768 --batch-order length --rows-per-shard 64 --head-precision $HEAD \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --purpose "gpu0 j6: T(i) vision_in + video_in answer caches (parent model.visual + streamed LLM), teacher-forced, top-64"
  rc=$?
  status "job=j6-vl state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc sealed=$(grep -c '^\[.*SEALED' $log) $(grep 'shard-[0-9]*: rows' $log | tail -1 | cut -c1-300)"
  return $rc
}

for STAGE in $STAGES; do
  echo "[$(stamp)] ===== $STAGE"
  job_$STAGE
  echo "[$(stamp)] ===== $STAGE returned $?"
done
status "lane=$LANE event=queue_end"
