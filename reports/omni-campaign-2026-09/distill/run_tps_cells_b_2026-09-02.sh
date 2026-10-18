#!/usr/bin/env bash
# Second pass: the cell-5 retry ladder and the gradient-accumulation arms.
#
# WHY THERE IS A SECOND PASS.  Cell 5's first attempt OOMed inside SDPA on the
# RTX 8000 with `--max-memory 0=40GiB,1=20GiB`: accelerate happily filled the
# 44.29 GiB card with 40 GiB of fp16 weights and left 4 GiB for activations, and
# a full-attention layer at L=4,096 wants more than that on sm75, where torch's
# flash SDPA backend does not exist and the math backend materialises a
# [1, 24, 4096, 4096] score matrix (1.5 GiB in fp16 -- exactly the allocation
# that failed).  So the retry walks a ladder of splits with more headroom and
# stops at the first one that runs.
#
# The accumulation arms exist because cell 3's number (23 tok/s) is measured at
# ONE optimizer step per 1,024 tokens, and no real schedule runs that way: the
# stage-D batch is 16 x 4,096.  Measuring at accumulation 16 shows how much of
# the 42-second host Adam step amortises.
#
#   cd /home/cereal/omni-campaign/distill
#   setsid nohup ./run_tps_cells_b_2026-09-02.sh > logs/driver-b.log 2>&1 < /dev/null &
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

# cell <timeout> <log-name> <visible-devices> <alloc-conf> <args...>
cell() {
    local seconds="$1" name="$2" visible="$3" alloc="$4"
    shift 4
    echo "[$(stamp)] START $name  (timeout ${seconds}s)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$visible" \
        PYTORCH_CUDA_ALLOC_CONF="$alloc" \
        timeout "$seconds" "$PY" "$DISTILL/tps_probe.py" --out-dir "$OUT" "$@" \
        > "$LOGS/$name.log" 2>&1
    local rc=$?
    echo "[$(stamp)] END   $name rc=$rc"
    [ $rc -ne 0 ] && tail -4 "$LOGS/$name.log"
    return $rc
}

echo "[$(stamp)] driver-b start; probe sha256 $(sha256sum "$DISTILL/tps_probe.py" | cut -d' ' -f1)"

# ---- cell 3 with a realistic optimizer-step cadence ----
if preflight "$GPU4" GPU4; then
    cell 3600 c3-4b-cpuadam-L1024-mb1-acc16 "$GPU4" expandable_segments:True \
        --cell 4b-full-cpuadam --sequence-length 1024 --micro-batch 1 --accumulation 16
fi

# ---- cell 2 with accumulation, to show the LoRA optimizer step is not the cost ----
if preflight "$GPU4" GPU4; then
    cell 1800 c2-4b-lora-L4096-mb1-acc4 "$GPU4" expandable_segments:True \
        --cell 4b-lora --sequence-length 4096 --micro-batch 1 --accumulation 4
fi

# ---- cell 5 retry ladder: first split that runs wins ----
LADDER="34GiB:18GiB 32GiB:19GiB 30GiB:20GiB"
DONE=0
for pair in $LADDER; do
    lhs=${pair%%:*}; rhs=${pair##*:}
    if [ "$DONE" -eq 1 ]; then break; fi
    if preflight "$GPU6" GPU6 && preflight "$GPU4" GPU4; then
        if cell 5400 "c5-27b-lora-split-L4096-mb1-${lhs}-${rhs}" "$GPU6,$GPU4" \
            expandable_segments:True \
            --cell 27b-lora-split --sequence-length 4096 --micro-batch 1 \
            --device-map sequential --max-memory "0=$lhs,1=$rhs" --loss-chunks 16 \
            --cell-id "27b-lora-split-L4096-mb1"; then
            echo "[$(stamp)] cell 5 succeeded at max-memory 0=$lhs,1=$rhs"
            DONE=1
        fi
    else
        echo "[$(stamp)] SKIP cell 5 rung $pair: a card is busy"
        break
    fi
done
[ "$DONE" -eq 0 ] && echo "[$(stamp)] cell 5 did NOT complete on any rung of the ladder"

echo "[$(stamp)] driver-b done"
ls -la "$OUT"
