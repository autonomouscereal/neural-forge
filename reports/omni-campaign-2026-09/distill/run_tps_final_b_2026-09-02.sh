#!/usr/bin/env bash
# Final pass, second attempt: cell 5 on the configuration the evidence points at,
# then cells 0-4 re-measured with the same probe build.
#
# WHAT THE SEVEN FAILED CELL-5 ATTEMPTS ESTABLISHED, in order:
#   0=40,1=20  OOM inside SDPA on the RTX 8000  -> 4 GiB of headroom is not enough
#   0=34,1=18 / 32,19 / 30,20   modules left on META -> the TOTAL budget must
#                               exceed the 51.75 GiB checkpoint by a few GiB,
#                               because accelerate keeps its own margin
#   0=35,1=19  OOM inside the probe's own isfinite() check -> replaced by amin/amax
#   0=39,1=15  OOM inside SDPA on the RTX 8000 -> same headroom lesson
#   0=37,1=17  non-reentrant checkpoint REFUSES every step: the forward runs the
#              fp32 math SDPA backend and the recompute runs a fused fp16 one
#              ([1,24,4096,4096] fp32 saved vs [1,24,4096,256] fp16 + philox
#              recomputed).  Turning autocast off did not change it.
#   0=37,1=17 --checkpoint-reentrant   reached BACKWARD and died 1.50 GiB short
#              on the RTX 8000 -- 1.50 GiB is exactly 24 x 4096 x 4096 x 4 B, the
#              math backend's fp32 score matrix.
# Measured placement at 0=37GiB,1=17GiB: cuda:0 = 18.47e9 params (34.4 GiB),
# cuda:1 = 8.88e9 (16.5 GiB), everything placed.  At 0=34GiB,1=19GiB the lm_head
# (1,271,398,400 parameters, 2.37 GiB) was left on meta, so accelerate keeps a
# margin of roughly 2.6 GiB per device below the stated budget.  The ladder
# therefore asks for MORE than it wants: 0=35GiB lands about 32.4 GiB of weights
# on the RTX 8000, which is the ~1.5 GiB shift the reentrant attempt was short.
# Reentrant checkpointing is kept (the non-reentrant one cannot be made to agree
# with itself here) and the vision tower this text-only probe never calls is
# dropped -- it is first in a sequential map, so its weights come off cuda:0.
#
#   cd /home/cereal/omni-campaign/distill
#   setsid nohup ./run_tps_final_b_2026-09-02.sh > logs/driver-final-b.log 2>&1 < /dev/null &
#
set -u

PY=/home/cereal/model-surgery-lab/venv/bin/python
DISTILL=/home/cereal/omni-campaign/distill
OUT=$DISTILL/tps
LOGS=$DISTILL/logs
GPU4=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
GPU6=GPU-f5a09535-fc85-a027-50f7-734b344747dd

mkdir -p "$OUT" "$LOGS"
cd "$DISTILL" || exit 2

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
used_mib() {
    nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits \
        | awk -F', *' -v u="$1" '$1==u {print $2}'
}
apps_on() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
        | awk -F', *' -v u="$1" '$1==u {print}'
}
free_card() {
    local used apps
    used=$(used_mib "$1"); apps=$(apps_on "$1")
    echo "[$(stamp)] preflight $2: used=${used:-?} MiB apps=[${apps:-none}]"
    [ -n "$used" ] && [ "$used" -le 1024 ] && [ -z "$apps" ]
}

echo "[$(stamp)] driver-final-b start; probe sha256 $(sha256sum "$DISTILL/tps_probe.py" | cut -d' ' -f1)"

DONE5=0
for pair in 35GiB:21GiB 34GiB:22GiB 36GiB:20GiB 33GiB:22GiB; do
    [ "$DONE5" -eq 1 ] && break
    lhs=${pair%%:*}; rhs=${pair##*:}
    name="c5r-${lhs}-${rhs}"
    free_card "$GPU6" GPU6 || { echo "[$(stamp)] GPU6 busy; skipping cell 5"; break; }
    free_card "$GPU4" GPU4 || { echo "[$(stamp)] GPU4 busy; skipping cell 5"; break; }
    echo "[$(stamp)] START $name (reentrant checkpoint, vision dropped)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU6,$GPU4" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        timeout 5400 "$PY" "$DISTILL/tps_probe.py" --out-dir "$OUT" \
        --cell 27b-lora-split --sequence-length 4096 --micro-batch 1 \
        --device-map sequential --max-memory "0=$lhs,1=$rhs" --loss-chunks 32 \
        --checkpoint-reentrant --drop-vision \
        --cell-id 27b-lora-split-L4096-mb1 \
        > "$LOGS/$name.log" 2>&1
    rc=$?
    echo "[$(stamp)] END   $name rc=$rc"
    if [ $rc -eq 0 ]; then
        echo "[$(stamp)] cell 5 SUCCEEDED at max-memory 0=$lhs,1=$rhs"
        DONE5=1
    else
        grep -hE "OutOfMemoryError|ProbeError|CheckpointError" "$LOGS/$name.log" | tail -2
        grep -h "placement:" "$LOGS/$name.log" | tail -1
    fi
done
[ "$DONE5" -eq 0 ] && echo "[$(stamp)] cell 5 did NOT complete on any rung"

cell() {
    local seconds="$1" name="$2"; shift 2
    free_card "$GPU4" GPU4 || { echo "[$(stamp)] GPU4 busy; skipping $name"; return 1; }
    echo "[$(stamp)] START $name"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU4" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        timeout "$seconds" "$PY" "$DISTILL/tps_probe.py" --out-dir "$OUT" "$@" \
        > "$LOGS/$name.log" 2>&1
    local rc=$?
    echo "[$(stamp)] END   $name rc=$rc"
    [ $rc -ne 0 ] && grep -hE "OutOfMemoryError|ProbeError|CheckpointError" "$LOGS/$name.log" | tail -2
    return $rc
}

cell 900  g1-08b-L1024-mb1    --cell 08b-full --sequence-length 1024 --micro-batch 1
cell 1200 g1-08b-L4096-mb1    --cell 08b-full --sequence-length 4096 --micro-batch 1
cell 2400 g1-08b-L1024-mbauto --cell 08b-full --sequence-length 1024 --micro-batch auto --micro-batch-max 32
cell 2400 g1-08b-L4096-mbauto --cell 08b-full --sequence-length 4096 --micro-batch auto --micro-batch-max 16
cell 1800 g2-4b-lora-mb1      --cell 4b-lora --sequence-length 4096 --micro-batch 1
cell 1800 g2-4b-lora-acc4     --cell 4b-lora --sequence-length 4096 --micro-batch 1 --accumulation 4
cell 3600 g3-4b-cpuadam-acc1  --cell 4b-full-cpuadam --sequence-length 1024 --micro-batch 1
cell 3600 g3-4b-cpuadam-acc16 --cell 4b-full-cpuadam --sequence-length 1024 --micro-batch 1 --accumulation 16
cell 5400 g4-27b-stream-mb1   --cell 27b-teacher-stream --sequence-length 4096 --micro-batch 1
cell 5400 g4-27b-stream-mb8   --cell 27b-teacher-stream --sequence-length 4096 --micro-batch 8
cell 900  g0-stream-verify    --cell stream-verify --sequence-length 256

echo "[$(stamp)] driver-final-b done"
sha256sum "$DISTILL/tps_probe.py"
ls -la "$OUT"
