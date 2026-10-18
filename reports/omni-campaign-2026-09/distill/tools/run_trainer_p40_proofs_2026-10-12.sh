#!/usr/bin/env bash
# T(ii) proofs on ONE P40 (fp32, Pascal venv): gradient parity arm, then the
# 50-step tiny KD run (+ an interrupted/resumed twin).  Detached launcher; poll
# the log.  Usage: run_trainer_p40_proofs_2026-10-12.sh <gpu index> [stage...]
# stages: parity cache train resume   (default: all four, in order)
set -u
GPU="${1:?gpu index}"; shift
STAGES="${*:-parity cache train resume}"
OC=/home/cereal/omni-campaign
D=$OC/distill/tools
W=$OC/distill/trainer-lane
PYP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
ALLOC=$OC/composite/receipts/allocator
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$D CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p $W/receipts $W/tiny $W/logs
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
card_check() {
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1==g{print $2}')
  echo "[$(stamp)] GPU$GPU memory.used ${used} MiB"
  if [ "${used:-99999}" -gt 2000 ]; then echo "REFUSED: GPU$GPU holds ${used} MiB (> 2000)"; exit 3; fi
}
alloc() {  # alloc <name> <purpose>
  cat > $ALLOC/$1.json <<JSON
{"lane":"T(ii) streamed KD trainer proofs","name":"$1","gpu":$GPU,"card":"Tesla P40","by":"claude trainer lane","purpose":"$2","python":"$PYP","started_utc":"$(stamp)","log":"$W/logs/p40_proofs_gpu$GPU.log","pid":$$}
JSON
}
release() { $PYP - "$ALLOC/$1.json" "$2" <<'PY'
import json, sys, time
p = sys.argv[1]; d = json.load(open(p)); d.update({"ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "released", "outcome": sys.argv[2]})
json.dump(d, open(p, "w"), indent=1)
PY
}
echo "[$(stamp)] trainer proofs on GPU$GPU stages: $STAGES"
sha256sum $D/streamed_kd_trainer.py $D/qtext_teacher_cache.py $OC/distill/tps_probe.py
$PYP -c "import torch,transformers;print('torch',torch.__version__,torch.cuda.get_arch_list(),torch.cuda.get_device_name(0),'transformers',transformers.__version__)"

for STAGE in $STAGES; do
case $STAGE in
parity)
  card_check; alloc trainer-lane-parity-gpu$GPU "T(ii) gradient parity arm (Qwen3.5-0.8B fp32, streamed + resident-hop vs stock autograd)"
  timeout 5400 nice -n 15 $PYP $D/streamed_kd_trainer.py grad-parity --model /home/cereal/models/Qwen3.5-0.8B \
      --device cuda:0 --dtype float32 --kernels torch --batch 2 --seq-len 256 --trunk-ffn-width 1024 \
      --also-resident --cross-device cpu --cross-device-threshold 1e-3 --threshold 1e-4 --receipt $W/receipts/grad_parity_q35_08b_p40.json
  rc=$?; echo "[$(stamp)] parity exit $rc"; release trainer-lane-parity-gpu$GPU "exit $rc" ;;
cache)
  $PYP - <<'PY'
import torch
from safetensors import safe_open
from safetensors.torch import save_file
src = "/home/cereal/omni-campaign/composite/corpus/qwen_input_ids.safetensors"
with safe_open(src, "pt") as f:
    ids, split, real = f.get_tensor("input_ids"), f.get_tensor("split"), f.get_tensor("real_lengths")
rows = [i for i in range(ids.shape[0]) if int(split[i]) == 0][:8]
out = ids[rows, :512].clone().to(torch.int64)
lengths = torch.clamp(real[rows], max=512).to(torch.int64)
save_file({"input_ids": out, "lengths": lengths}, "/home/cereal/omni-campaign/distill/trainer-lane/tiny/corpus_tiny_8x512.safetensors",
          metadata={"source": src, "rows": str(rows), "note": "fit-split rows, first 512 tokens; mechanics test only (in-sample fold corpus)"})
print("corpus rows", rows, "lengths", lengths.tolist(), "max id", int(out.max()))
PY
  card_check; alloc trainer-lane-tinykd-cache-gpu$GPU "tiny KD teacher cache: Qwen3.5-4B fp32 top-64 over 8x512 (qtext_teacher_cache build-text)"
  timeout 5400 nice -n 15 $PYP $D/qtext_teacher_cache.py build-text --model-dir /home/cereal/models/Qwen3.5-4B \
      --teacher q35-4b-tinykd --corpus $W/tiny/corpus_tiny_8x512.safetensors --device cuda:0 --dtype float32 \
      --linear-attn-kernel torch --micro-batch 2 --rows-per-shard 8 --out-root $W/teacher_cache --teacher-digest full
  rc=$?; echo "[$(stamp)] cache exit $rc"; release trainer-lane-tinykd-cache-gpu$GPU "exit $rc" ;;
train|resume)
  SHA=$(sha256sum $W/tiny/corpus_tiny_8x512.safetensors | cut -d' ' -f1)
  cat > $W/tiny/data_tinykd.json <<JSON
{"sources": [{"name": "q35-4b-tinykd", "kind": "text", "qtext_cache": "$W/teacher_cache/q35-4b-tinykd/$SHA"}]}
JSON
  COMMON="--model /home/cereal/models/Qwen3.5-0.8B --devices cuda:0 --dtype float32 --kernels torch \
      --data-config $W/tiny/data_tinykd.json --seq-len 512 --micro-batch 2 --total-steps 50 --trunk-ffn-width 1024 \
      --lr 5e-4 --gain-lr 5e-3 --warmup-steps 5 --safety-every-steps 10 --log-every 1 --deterministic --head-chunk-positions 512"
  card_check
  if [ $STAGE = train ]; then
    alloc trainer-lane-tinykd-train-gpu$GPU "tiny KD run: Qwen3.5-0.8B LoRA streamed fp32, 50 steps"
    rm -rf $W/tinykd_full $W/tinykd_full_safety
    timeout 7200 nice -n 15 $PYP $D/streamed_kd_trainer.py train $COMMON --out $W/tinykd_full --safety-dir $W/tinykd_full_safety
    rc=$?; echo "[$(stamp)] train exit $rc"; release trainer-lane-tinykd-train-gpu$GPU "exit $rc"
  else
    alloc trainer-lane-tinykd-resume-gpu$GPU "tiny KD twin: 30 steps, process exit, resume to 50"
    rm -rf $W/tinykd_half $W/tinykd_half_safety
    timeout 7200 nice -n 15 $PYP $D/streamed_kd_trainer.py train $COMMON --out $W/tinykd_half --safety-dir $W/tinykd_half_safety --max-steps-this-run 30
    echo "[$(stamp)] half exit $?"
    timeout 7200 nice -n 15 $PYP $D/streamed_kd_trainer.py train $COMMON --out $W/tinykd_half --safety-dir $W/tinykd_half_safety --resume auto
    rc=$?; echo "[$(stamp)] resume exit $rc"
    $PYP - <<'PY'
import torch
from safetensors.torch import load_file
a = load_file("/home/cereal/omni-campaign/distill/trainer-lane/tinykd_full/adapters.safetensors")
b = load_file("/home/cereal/omni-campaign/distill/trainer-lane/tinykd_half/adapters.safetensors")
equal = sum(torch.equal(a[k], b[k]) for k in a)
worst = max(float((a[k] - b[k]).abs().max() / (a[k].abs().max() + 1e-30)) for k in a)
print(f"RESUME-CHECK tensors {len(a)} bitwise_equal {equal} worst_rel_maxabs {worst:.3e}")
PY
    release trainer-lane-tinykd-resume-gpu$GPU "exit $rc"
  fi ;;
esac
done
echo "[$(stamp)] ALL DONE"
