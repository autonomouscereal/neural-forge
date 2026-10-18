#!/usr/bin/env bash
# Third pass: cell 5 only, on a CORRECTED split ladder.
#
# WHY THE FIRST LADDER WAS WRONG.  Pass B walked 34+18, 32+19, 30+20 GiB, whose
# totals are 52, 51 and 50 GiB against a 51.75 GiB fp16 checkpoint: the last two
# rungs could not hold the model at all, and the first left accelerate one
# rounding error of slack, so all three ended on the meta device.  Headroom is
# balanced when the two cards have the same slack:
#
#   44.29 - a  ==  23.56 - (51.75 - a)   =>   a ~= 36.2 GiB on the RTX 8000
#
# so the ladder now walks outward from there and every rung's TOTAL exceeds the
# checkpoint by several GiB.
#
#   cd /home/cereal/omni-campaign/distill
#   setsid nohup ./run_tps_cell5_c_2026-09-02.sh > logs/driver-c.log 2>&1 < /dev/null &
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

preflight() {
    local uuid="$1" label="$2" used apps
    used=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits \
           | awk -F', *' -v u="$uuid" '$1==u {print $2}')
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
           | awk -F', *' -v u="$uuid" '$1==u {print}')
    echo "[$(stamp)] preflight $label ($uuid): used=${used:-?} MiB apps=[${apps:-none}]"
    [ -n "$used" ] && [ "$used" -le 1024 ]
}

echo "[$(stamp)] driver-c start; probe sha256 $(sha256sum "$DISTILL/tps_probe.py" | cut -d' ' -f1)"

DONE=0
for pair in 37GiB:17GiB 35GiB:19GiB 39GiB:15GiB 33GiB:21GiB; do
    [ "$DONE" -eq 1 ] && break
    lhs=${pair%%:*}; rhs=${pair##*:}
    name="c5-27b-lora-split-${lhs}-${rhs}"
    if ! preflight "$GPU6" GPU6 || ! preflight "$GPU4" GPU4; then
        echo "[$(stamp)] SKIP cell 5 rung $pair: a card is busy"
        break
    fi
    echo "[$(stamp)] START $name"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU6,$GPU4" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        timeout 5400 "$PY" "$DISTILL/tps_probe.py" --out-dir "$OUT" \
        --cell 27b-lora-split --sequence-length 4096 --micro-batch 1 \
        --device-map sequential --max-memory "0=$lhs,1=$rhs" --loss-chunks 16 \
        --cell-id 27b-lora-split-L4096-mb1 \
        > "$LOGS/$name.log" 2>&1
    rc=$?
    echo "[$(stamp)] END   $name rc=$rc"
    if [ $rc -eq 0 ]; then
        echo "[$(stamp)] cell 5 SUCCEEDED at max-memory 0=$lhs,1=$rhs"
        DONE=1
    else
        tail -4 "$LOGS/$name.log"
    fi
done
[ "$DONE" -eq 0 ] && echo "[$(stamp)] cell 5 did NOT complete on any rung"

echo "[$(stamp)] driver-c done"
