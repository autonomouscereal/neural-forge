#!/usr/bin/env bash
# T(ii) smokes on ONE P40 after run_trainer_p40_proofs: the split-cell engine in
# fp16 with the dynamic loss scaler (both "devices" on the one card: resident
# layers + scaler path; the host hop itself is proven by the parity arm's
# resident-hop arm), and host-staged data parallel with two ranks on the card.
# Usage: run_trainer_p40_extra_2026-10-12.sh <gpu index>
set -u
GPU="${1:?gpu index}"
OC=/home/cereal/omni-campaign
D=$OC/distill/tools
W=$OC/distill/trainer-lane
PYP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
ALLOC=$OC/composite/receipts/allocator
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$D CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1==g{print $2}')
echo "[$(stamp)] GPU$GPU memory.used ${used} MiB"
if [ "${used:-99999}" -gt 2000 ]; then echo "REFUSED: GPU$GPU holds ${used} MiB (> 2000)"; exit 3; fi
cat > $ALLOC/trainer-lane-extra-gpu$GPU.json <<JSON
{"lane":"T(ii) streamed KD trainer smokes","name":"trainer-lane-extra-gpu$GPU","gpu":$GPU,"card":"Tesla P40","by":"claude trainer lane","purpose":"split-cell fp16 + loss scaler smoke; 2-rank host-staged DP smoke","python":"$PYP","started_utc":"$(stamp)","log":"$W/logs/p40_extra_gpu$GPU.log","pid":$$}
JSON
COMMON="--model /home/cereal/models/Qwen3.5-0.8B --kernels torch --data-config $W/tiny/data_tinykd.json --seq-len 512 \
    --trunk-ffn-width 1024 --lr 5e-4 --gain-lr 5e-3 --warmup-steps 2 --log-every 1 --head-chunk-positions 512"
rm -rf $W/split16 $W/split16_safety $W/dp2 $W/dp2_safety $W/dp1 $W/dp1_safety $W/dp1_step1_grads.safetensors $W/dp2_step1_grads.safetensors
echo "[$(stamp)] split cell fp16, scaler init 2^10, 6 steps"
timeout 5400 nice -n 15 $PYP $D/streamed_kd_trainer.py train $COMMON --engine split --devices cuda:0,cuda:0 --split-layers 12 \
    --dtype float16 --initial-loss-scale 1024 --micro-batch 2 --total-steps 6 --out $W/split16 --safety-dir $W/split16_safety
echo "[$(stamp)] split exit $?"
echo "[$(stamp)] DP: 2 ranks x mb1 vs 1 rank x mb2, streamed fp32, 3 steps"
timeout 5400 nice -n 15 $PYP $D/streamed_kd_trainer.py train $COMMON --devices cuda:0 --dtype float32 --micro-batch 2 \
    --total-steps 3 --out $W/dp1 --safety-dir $W/dp1_safety --deterministic --dump-step-grads $W/dp1_step1_grads.safetensors
echo "[$(stamp)] dp1 exit $?"
timeout 5400 nice -n 15 $PYP -m torch.distributed.run --standalone --nproc_per_node 2 $D/streamed_kd_trainer.py train $COMMON \
    --devices cuda:0,cuda:0 --dtype float32 --micro-batch 1 --total-steps 3 --out $W/dp2 --safety-dir $W/dp2_safety --deterministic --dump-step-grads $W/dp2_step1_grads.safetensors
echo "[$(stamp)] dp2 exit $?"
$PYP - <<'PY'
import json, torch
from safetensors.torch import load_file
W = "/home/cereal/omni-campaign/distill/trainer-lane"
ga, gb = load_file(f"{W}/dp1_step1_grads.safetensors"), load_file(f"{W}/dp2_step1_grads.safetensors")
grel = max(float((ga[k] - gb[k]).norm() / (ga[k].norm() + 1e-30)) for k in ga)
print(f"DP-CHECK reduced step-1 gradient (rank 0, 2 ranks x mb1 vs 1 rank x mb2): tensors {len(ga)} max rel-L2 {grel:.3e}")
a, b = load_file(f"{W}/dp1/adapters.safetensors"), load_file(f"{W}/dp2/adapters.safetensors")
rel = max(float((a[k] - b[k]).norm() / (a[k].norm() + 1e-30)) for k in a)
el = max(float((a[k] - b[k]).abs().max() / (a[k].abs().max() + 1e-30)) for k in a)
r1 = json.load(open(f"{W}/dp1/training-receipt.json")); r2 = json.load(open(f"{W}/dp2/training-receipt.json"))
print("DP-CHECK losses single", [round(s["loss"], 6) for s in r1["steps"]], "dp2", [round(s["loss"], 6) for s in r2["steps"]])
print(f"DP-CHECK adapters max rel-L2 {rel:.3e} max element rel {el:.3e} allreduce_s {r2['allreduce_seconds_mean']}")
s = json.load(open(f"{W}/split16/training-receipt.json"))
print("SPLIT16 scaler", s["scaler"], "applied", s["applied_steps"], "skipped", s["skipped_steps"], "losses", [None if x["loss"] is None else round(x["loss"], 4) for x in s["steps"]])
PY
python3 - "$ALLOC/trainer-lane-extra-gpu$GPU.json" <<'PY'
import json, sys, time
p = sys.argv[1]; d = json.load(open(p)); d.update({"ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "released"}); json.dump(d, open(p, "w"), indent=1)
PY
echo "[$(stamp)] EXTRA DONE"
