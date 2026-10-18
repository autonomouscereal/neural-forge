#!/usr/bin/env bash
# T(ii) 1.3.0 re-run of every GPU proof on ONE P40 (fp32 / fp16, Pascal venv),
# after the review fixes (skip guard, resume identity, xtok provenance):
#   parity   gradient parity arm (streamed + resident-hop gating, CPU arm diagnostic)
#   train    50-step tiny KD run
#   resume   30 steps, process exit, resume to 50, bitwise compare with train
#   split16  split cell fp16, scaler init 2^10, 6 steps (must NOT halt)
#   dp       2 ranks x mb1 vs 1 rank x mb2, streamed fp32, 3 steps
#   halt16   split cell fp16, scaler init 2^40: real fp16 overflow; the skip
#            guard must halt at 8 consecutive skips with exit 3 and no export
# Everything goes under trainer-lane/v130 (the v1.2.0 evidence is left alone);
# the tiny corpus, its Q-text teacher cache and data_tinykd.json are reused
# read-only.  Waits (bounded) for the first candidate P40 under 2000 MiB.
# Usage: run_trainer_p40_v130_2026-10-12.sh "<candidate gpus, e.g. 3 1 2>" [stage...]
# launcher version: v130-2026-10-12b (b: card_check waits instead of refusing between stages)
set -u
CANDIDATES="${1:?candidate gpu indices}"; shift
STAGES="${*:-parity train resume split16 dp halt16}"
OC=/home/cereal/omni-campaign
D=$OC/distill/tools
W=$OC/distill/trainer-lane
V=$W/v130
PYP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
ALLOC=$OC/composite/receipts/allocator
WAIT_MAX_S=${WAIT_MAX_S:-21600}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=$D CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
mkdir -p $V/receipts
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
used_of() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g{print $2}'; }
GPU=""
waited=0
while [ -z "$GPU" ]; do
  for g in $CANDIDATES; do
    u=$(used_of $g)
    if [ "${u:-99999}" -le 2000 ]; then GPU=$g; break; fi
  done
  if [ -z "$GPU" ]; then
    if [ $waited -ge $WAIT_MAX_S ]; then echo "[$(stamp)] GAVE UP: no P40 of [$CANDIDATES] under 2000 MiB after ${waited}s"; exit 4; fi
    [ $((waited % 600)) -eq 0 ] && echo "[$(stamp)] waiting for a free P40 of [$CANDIDATES] ($(for g in $CANDIDATES; do printf 'GPU%s=%sMiB ' $g $(used_of $g); done))"
    sleep 60; waited=$((waited + 60))
  fi
done
export CUDA_VISIBLE_DEVICES=$GPU
echo "[$(stamp)] 1.3.0 proofs on GPU$GPU (waited ${waited}s) stages: $STAGES"
card_check() {  # waits (bounded) while another lane holds the card between stages; never starts above 2000 MiB
  used=$(used_of $GPU); held=0
  while [ "${used:-99999}" -gt 2000 ]; do
    if [ $held -ge $WAIT_MAX_S ]; then echo "REFUSED: GPU$GPU holds ${used} MiB (> 2000) after ${held}s"; exit 3; fi
    [ $((held % 600)) -eq 0 ] && echo "[$(stamp)] GPU$GPU holds ${used} MiB (> 2000); waiting"
    sleep 60; held=$((held + 60)); used=$(used_of $GPU)
  done
  echo "[$(stamp)] GPU$GPU memory.used ${used} MiB"
}
alloc() {  # alloc <name> <purpose>
  cat > $ALLOC/$1.json <<JSON
{"lane":"T(ii) streamed KD trainer 1.3.0 re-proofs","name":"$1","gpu":$GPU,"card":"Tesla P40","by":"claude trainer lane","purpose":"$2","python":"$PYP","started_utc":"$(stamp)","log":"$W/logs/p40_v130.log","pid":$$}
JSON
}
release() { $PYP - "$ALLOC/$1.json" "$2" <<'PY'
import json, sys, time
p = sys.argv[1]; d = json.load(open(p)); d.update({"ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "released", "outcome": sys.argv[2]})
json.dump(d, open(p, "w"), indent=1)
PY
}
sha256sum $D/streamed_kd_trainer.py $D/qtext_teacher_cache.py $W/tiny/data_tinykd.json $W/tiny/corpus_tiny_8x512.safetensors
$PYP $D/streamed_kd_trainer.py --version
$PYP -c "import torch,transformers;print('torch',torch.__version__,torch.cuda.get_arch_list(),torch.cuda.get_device_name(0),'transformers',transformers.__version__)"
COMMON="--model /home/cereal/models/Qwen3.5-0.8B --devices cuda:0 --dtype float32 --kernels torch \
    --data-config $W/tiny/data_tinykd.json --seq-len 512 --micro-batch 2 --total-steps 50 --trunk-ffn-width 1024 \
    --lr 5e-4 --gain-lr 5e-3 --warmup-steps 5 --safety-every-steps 10 --log-every 1 --deterministic --head-chunk-positions 512"
SMOKE="--model /home/cereal/models/Qwen3.5-0.8B --kernels torch --data-config $W/tiny/data_tinykd.json --seq-len 512 \
    --trunk-ffn-width 1024 --lr 5e-4 --gain-lr 5e-3 --warmup-steps 2 --log-every 1 --head-chunk-positions 512"

for STAGE in $STAGES; do
card_check
case $STAGE in
parity)
  alloc trainer-lane-v130-parity-gpu$GPU "T(ii) 1.3.0 gradient parity arm (Qwen3.5-0.8B fp32, streamed + resident-hop vs stock autograd)"
  timeout 5400 nice -n 15 $PYP $D/streamed_kd_trainer.py grad-parity --model /home/cereal/models/Qwen3.5-0.8B \
      --device cuda:0 --dtype float32 --kernels torch --batch 2 --seq-len 256 --trunk-ffn-width 1024 \
      --also-resident --cross-device cpu --cross-device-threshold 1e-3 --threshold 1e-4 --receipt $V/receipts/grad_parity_q35_08b_p40.json
  rc=$?; echo "[$(stamp)] parity exit $rc"; release trainer-lane-v130-parity-gpu$GPU "exit $rc" ;;
train)
  alloc trainer-lane-v130-tinykd-train-gpu$GPU "tiny KD run 1.3.0: Qwen3.5-0.8B LoRA streamed fp32, 50 steps"
  rm -rf $V/tinykd_full $V/tinykd_full_safety
  timeout 7200 nice -n 15 $PYP $D/streamed_kd_trainer.py train $COMMON --out $V/tinykd_full --safety-dir $V/tinykd_full_safety
  rc=$?; echo "[$(stamp)] train exit $rc"; release trainer-lane-v130-tinykd-train-gpu$GPU "exit $rc" ;;
resume)
  alloc trainer-lane-v130-tinykd-resume-gpu$GPU "tiny KD twin 1.3.0: 30 steps, process exit, resume to 50"
  rm -rf $V/tinykd_half $V/tinykd_half_safety
  timeout 7200 nice -n 15 $PYP $D/streamed_kd_trainer.py train $COMMON --out $V/tinykd_half --safety-dir $V/tinykd_half_safety --max-steps-this-run 30
  echo "[$(stamp)] half exit $?"
  timeout 7200 nice -n 15 $PYP $D/streamed_kd_trainer.py train $COMMON --out $V/tinykd_half --safety-dir $V/tinykd_half_safety --resume auto
  rc=$?; echo "[$(stamp)] resume exit $rc"
  $PYP - <<'PY'
import json, torch
from safetensors.torch import load_file
V = "/home/cereal/omni-campaign/distill/trainer-lane/v130"
a = load_file(f"{V}/tinykd_full/adapters.safetensors")
b = load_file(f"{V}/tinykd_half/adapters.safetensors")
equal = sum(torch.equal(a[k], b[k]) for k in a)
worst = max(float((a[k] - b[k]).abs().max() / (a[k].abs().max() + 1e-30)) for k in a)
print(f"RESUME-CHECK tensors {len(a)} bitwise_equal {equal} worst_rel_maxabs {worst:.3e}")
r = json.load(open(f"{V}/tinykd_half/training-receipt.json"))
print("RESUME-SEGMENTS", [(s["segment"], s["start_step"], s["identity_override"]) for s in r["segments"]], "verdict", r["verdict"])
PY
  release trainer-lane-v130-tinykd-resume-gpu$GPU "exit $rc" ;;
split16)
  alloc trainer-lane-v130-split16-gpu$GPU "split cell fp16 + loss scaler smoke 1.3.0 (scale 2^10, 6 steps)"
  rm -rf $V/split16 $V/split16_safety
  timeout 5400 nice -n 15 $PYP $D/streamed_kd_trainer.py train $SMOKE --engine split --devices cuda:0,cuda:0 --split-layers 12 \
      --dtype float16 --initial-loss-scale 1024 --micro-batch 2 --total-steps 6 --out $V/split16 --safety-dir $V/split16_safety
  rc=$?; echo "[$(stamp)] split exit $rc"; release trainer-lane-v130-split16-gpu$GPU "exit $rc" ;;
dp)
  alloc trainer-lane-v130-dp-gpu$GPU "2-rank host-staged DP smoke 1.3.0"
  rm -rf $V/dp1 $V/dp1_safety $V/dp2 $V/dp2_safety $V/dp1_step1_grads.safetensors $V/dp2_step1_grads.safetensors
  timeout 5400 nice -n 15 $PYP $D/streamed_kd_trainer.py train $SMOKE --devices cuda:0 --dtype float32 --micro-batch 2 \
      --total-steps 3 --out $V/dp1 --safety-dir $V/dp1_safety --deterministic --dump-step-grads $V/dp1_step1_grads.safetensors
  echo "[$(stamp)] dp1 exit $?"
  timeout 5400 nice -n 15 $PYP -m torch.distributed.run --standalone --nproc_per_node 2 $D/streamed_kd_trainer.py train $SMOKE \
      --devices cuda:0,cuda:0 --dtype float32 --micro-batch 1 --total-steps 3 --out $V/dp2 --safety-dir $V/dp2_safety --deterministic \
      --dump-step-grads $V/dp2_step1_grads.safetensors
  rc=$?; echo "[$(stamp)] dp2 exit $rc"
  $PYP - <<'PY'
import json
from safetensors.torch import load_file
V = "/home/cereal/omni-campaign/distill/trainer-lane/v130"
ga, gb = load_file(f"{V}/dp1_step1_grads.safetensors"), load_file(f"{V}/dp2_step1_grads.safetensors")
grel = max(float((ga[k] - gb[k]).norm() / (ga[k].norm() + 1e-30)) for k in ga)
print(f"DP-CHECK reduced step-1 gradient (rank 0, 2 ranks x mb1 vs 1 rank x mb2): tensors {len(ga)} max rel-L2 {grel:.3e}")
a, b = load_file(f"{V}/dp1/adapters.safetensors"), load_file(f"{V}/dp2/adapters.safetensors")
rel = max(float((a[k] - b[k]).norm() / (a[k].norm() + 1e-30)) for k in a)
el = max(float((a[k] - b[k]).abs().max() / (a[k].abs().max() + 1e-30)) for k in a)
r1 = json.load(open(f"{V}/dp1/training-receipt.json")); r2 = json.load(open(f"{V}/dp2/training-receipt.json"))
print("DP-CHECK losses single", [round(s["loss"], 6) for s in r1["steps"]], "dp2", [round(s["loss"], 6) for s in r2["steps"]])
print(f"DP-CHECK adapters max rel-L2 {rel:.3e} max element rel {el:.3e} allreduce_s {r2['allreduce_seconds_mean']}")
PY
  release trainer-lane-v130-dp-gpu$GPU "exit $rc" ;;
halt16)
  alloc trainer-lane-v130-halt16-gpu$GPU "skip-guard proof 1.3.0: split fp16 at loss scale 2^40 (real overflow), expect halt + exit 3"
  rm -rf $V/halt16 $V/halt16_safety
  timeout 5400 nice -n 15 $PYP $D/streamed_kd_trainer.py train $SMOKE --engine split --devices cuda:0,cuda:0 --split-layers 12 \
      --dtype float16 --initial-loss-scale 1099511627776 --micro-batch 2 --total-steps 20 --out $V/halt16 --safety-dir $V/halt16_safety
  rc=$?; echo "[$(stamp)] halt16 exit $rc (expected 3)"
  $PYP - <<'PY'
import json, os
V = "/home/cereal/omni-campaign/distill/trainer-lane/v130"
r = json.load(open(f"{V}/halt16/training-receipt.json"))
print("HALT16 verdict", r["verdict"], "step", r["step"], "applied", r["applied_steps"], "skipped", r["skipped_steps"],
      "scaler", r["scaler"], "reason", r["halt_reason"], "adapters_exported", os.path.exists(f"{V}/halt16/adapters.safetensors"),
      "checkpoints", sorted(os.listdir(f"{V}/halt16_safety")))
PY
  release trainer-lane-v130-halt16-gpu$GPU "exit $rc" ;;
esac
done
echo "[$(stamp)] ALL DONE"
