#!/usr/bin/env bash
# Fourth pass: cell 5, waiting for GPU6 to be released by whoever holds it.
#
# At 14:45:25Z the corrected ladder found GPU6 held by the qwen-fold lane
# (`qf_build.py --tag DRY-T9B --parent Qwen3.5-0.8B`, pid 3378771, 34 s old).
# That process is not ours and is never touched.  This driver waits for the card
# with a hard deadline, then runs the ladder; if the deadline passes it writes
# the refusal and exits, because "wait forever" is how a lane loses a night.
#
#   cd /home/cereal/omni-campaign/distill
#   setsid nohup ./run_tps_cell5_d_2026-09-02.sh > logs/driver-d.log 2>&1 < /dev/null &
#
set -u

PY=/home/cereal/model-surgery-lab/venv/bin/python
DISTILL=/home/cereal/omni-campaign/distill
OUT=$DISTILL/tps
LOGS=$DISTILL/logs
GPU4=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
GPU6=GPU-f5a09535-fc85-a027-50f7-734b344747dd
WAIT_DEADLINE_SECONDS=2700

mkdir -p "$OUT" "$LOGS"
cd "$DISTILL" || exit 2

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

card_used_mib() {
    nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits \
        | awk -F', *' -v u="$1" '$1==u {print $2}'
}

card_apps() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
        | awk -F', *' -v u="$1" '$1==u {print}'
}

echo "[$(stamp)] driver-d start; probe sha256 $(sha256sum "$DISTILL/tps_probe.py" | cut -d' ' -f1)"
echo "[$(stamp)] waiting up to ${WAIT_DEADLINE_SECONDS}s for GPU6; the holder is never signalled"

deadline=$(( $(date +%s) + WAIT_DEADLINE_SECONDS ))
while :; do
    used=$(card_used_mib "$GPU6")
    apps=$(card_apps "$GPU6")
    if [ -n "$used" ] && [ "$used" -le 1024 ] && [ -z "$apps" ]; then
        echo "[$(stamp)] GPU6 free (${used} MiB, no compute apps)"
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[$(stamp)] DEADLINE: GPU6 still held (${used} MiB) by: ${apps:-unknown}"
        echo "[$(stamp)] SKIPPING cell 5 -- the campaign rule is skip and say so"
        echo "[$(stamp)] driver-d done"
        exit 0
    fi
    sleep 20
done

used4=$(card_used_mib "$GPU4")
apps4=$(card_apps "$GPU4")
echo "[$(stamp)] preflight GPU4: used=${used4} MiB apps=[${apps4:-none}]"
if [ -z "$used4" ] || [ "$used4" -gt 1024 ]; then
    echo "[$(stamp)] SKIPPING cell 5 -- GPU4 is busy"
    echo "[$(stamp)] driver-d done"
    exit 0
fi

DONE=0
for pair in 37GiB:17GiB 35GiB:19GiB 39GiB:15GiB 33GiB:21GiB; do
    [ "$DONE" -eq 1 ] && break
    lhs=${pair%%:*}; rhs=${pair##*:}
    name="c5-27b-lora-split-${lhs}-${rhs}"
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
        tail -6 "$LOGS/$name.log"
    fi
done
[ "$DONE" -eq 0 ] && echo "[$(stamp)] cell 5 did NOT complete on any rung"

echo "[$(stamp)] driver-d done"
