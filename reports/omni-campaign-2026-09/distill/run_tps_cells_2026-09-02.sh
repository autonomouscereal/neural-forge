#!/usr/bin/env bash
# Runs the tps_probe cells for the 2026-09-02 stage-D budget receipt, strictly
# serially on one card at a time.
#
# WHY A DRIVER AND NOT NINE SSH CALLS.  The estate lost hours to client-side
# retry loops and to jobs that died with their SSH session (NOTES-S2A).  So the
# whole sequence is one detached process on the server, each cell wrapped in
# `timeout`, each cell preceded by a preflight that REFUSES the card if anyone
# else is on it.  Nothing here ever stops a process it did not start.
#
#   cd /home/cereal/omni-campaign/distill
#   setsid nohup ./run_tps_cells_2026-09-02.sh > logs/driver.log 2>&1 < /dev/null &
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

# preflight <uuid> <label> -- 0 when the card is free enough to take a cell.
preflight() {
    local uuid="$1" label="$2" used apps
    used=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits \
           | awk -F', *' -v u="$uuid" '$1==u {print $2}')
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
           | awk -F', *' -v u="$uuid" '$1==u {print}')
    echo "[$(stamp)] preflight $label ($uuid): used=${used:-?} MiB apps=[${apps:-none}]"
    if [ -z "$used" ]; then
        echo "[$(stamp)] REFUSE $label: nvidia-smi did not report this UUID"
        return 1
    fi
    if [ "$used" -gt 1024 ]; then
        echo "[$(stamp)] REFUSE $label: $used MiB already in use and it is not ours"
        return 1
    fi
    return 0
}

# cell <timeout-seconds> <log-name> <env-visible-devices> <args...>
cell() {
    local seconds="$1" name="$2" visible="$3"
    shift 3
    echo "[$(stamp)] START $name  (timeout ${seconds}s)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$visible" \
        timeout "$seconds" "$PY" "$DISTILL/tps_probe.py" --out-dir "$OUT" "$@" \
        > "$LOGS/$name.log" 2>&1
    local rc=$?
    echo "[$(stamp)] END   $name rc=$rc"
    if [ $rc -ne 0 ]; then
        echo "----- tail of $LOGS/$name.log -----"
        tail -25 "$LOGS/$name.log"
        echo "-----------------------------------"
    fi
    return $rc
}

echo "[$(stamp)] driver start; probe sha256 $(sha256sum "$DISTILL/tps_probe.py" | cut -d' ' -f1)"

# ---- cell 1: Qwen3.5-0.8B, full parameter, fp32 master under bf16 autocast ----
preflight "$GPU4" GPU4 || exit 3
cell 900  c1-08b-L1024-mb1   "$GPU4" --cell 08b-full --sequence-length 1024 --micro-batch 1
preflight "$GPU4" GPU4 || exit 3
cell 1200 c1-08b-L4096-mb1   "$GPU4" --cell 08b-full --sequence-length 4096 --micro-batch 1
preflight "$GPU4" GPU4 || exit 3
cell 2400 c1-08b-L1024-mbauto "$GPU4" --cell 08b-full --sequence-length 1024 --micro-batch auto --micro-batch-max 32
preflight "$GPU4" GPU4 || exit 3
cell 2400 c1-08b-L4096-mbauto "$GPU4" --cell 08b-full --sequence-length 4096 --micro-batch auto --micro-batch-max 16

# ---- cell 2: Qwen3.5-4B, hand-rolled LoRA r=64 ----
preflight "$GPU4" GPU4 || exit 3
cell 1800 c2-4b-lora-L4096-mb1 "$GPU4" --cell 4b-lora --sequence-length 4096 --micro-batch 1

# ---- cell 3: Qwen3.5-4B, full parameter, host-offloaded AdamW ----
preflight "$GPU4" GPU4 || exit 3
cell 3600 c3-4b-cpuadam-L1024-mb1 "$GPU4" --cell 4b-full-cpuadam --sequence-length 1024 --micro-batch 1

# ---- cell 4: Qwen3.8-27B-Uncensored, layer-streamed teacher forward ----
preflight "$GPU4" GPU4 || exit 3
cell 5400 c4-27b-stream-L4096-mb1 "$GPU4" --cell 27b-teacher-stream --sequence-length 4096 --micro-batch 1
preflight "$GPU4" GPU4 || exit 3
cell 5400 c4-27b-stream-L4096-mb8 "$GPU4" --cell 27b-teacher-stream --sequence-length 4096 --micro-batch 8

# ---- cell 5: Qwen3.8-27B-Uncensored LoRA split over the RTX 8000 and a 3090 ----
if preflight "$GPU6" GPU6 && preflight "$GPU4" GPU4; then
    cell 5400 c5-27b-lora-split-L4096-mb1 "$GPU6,$GPU4" \
        --cell 27b-lora-split --sequence-length 4096 --micro-batch 1 \
        --device-map sequential --max-memory 0=40GiB,1=20GiB
else
    echo "[$(stamp)] SKIP cell 5: GPU6 or GPU4 is busy; the campaign says skip and say so"
fi

echo "[$(stamp)] driver done"
ls -la "$OUT"
