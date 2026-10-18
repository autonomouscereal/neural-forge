#!/usr/bin/env bash
# Final pass: cell 5 on the split that got furthest, then every other cell
# re-measured with the SAME probe build so one sha256 covers the whole table.
#
# WHY EVERYTHING IS RE-RUN.  Passes A and B ran the LoRA cells under
# `torch.autocast`.  On the 27B split that is not survivable: the model's own
# rope table is forced to fp32 (`maybe_autocast(enabled=False)`), so
# `apply_rotary_pos_emb` promotes q and k to fp32, and then whether SDPA runs the
# fp32 MATH backend or a fused fp16 backend depends on whether autocast is in
# scope.  The forward and the checkpoint's recompute disagreed about that, so
# `torch.utils.checkpoint` refused every step with "Recomputed values ... have
# different metadata" -- saved [1,24,4096,4096] fp32 against recomputed
# [1,24,4096,256] fp16 plus a philox seed.  The LoRA cells now run with no
# autocast at all and the adapters cast themselves, which makes the two passes
# the same function.  That is a change to the thing being measured, and a table
# whose rows were taken under two different policies is not a table, so every
# cell is re-measured.  About an hour, and it buys one probe sha256 for the
# whole table.
#
# The old receipts are tarred first; nothing is deleted.
#
#   cd /home/cereal/omni-campaign/distill
#   setsid nohup ./run_tps_final_2026-09-02.sh > logs/driver-final.log 2>&1 < /dev/null &
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
wait_for_card() {   # wait_for_card <uuid> <label> <deadline-seconds>
    local deadline=$(( $(date +%s) + $3 ))
    while ! free_card "$1" "$2"; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "[$(stamp)] DEADLINE waiting for $2; the holder is never signalled"
            return 1
        fi
        sleep 20
    done
    return 0
}

BACKUP=$DISTILL/tps-backup-passAB-$(date -u +%Y%m%dT%H%M%SZ).tar.gz
tar -czf "$BACKUP" -C "$DISTILL" tps
echo "[$(stamp)] backed up the pass-A/B receipts to $BACKUP"
echo "[$(stamp)] driver-final start; probe sha256 $(sha256sum "$DISTILL/tps_probe.py" | cut -d' ' -f1)"

# ------------------------------------------------------------------ cell 5 --
DONE5=0
for rung in "37GiB:17GiB:" "37GiB:17GiB:--checkpoint-reentrant" "37GiB:17GiB:--drop-vision" "36GiB:18GiB:--drop-vision"; do
    [ "$DONE5" -eq 1 ] && break
    lhs=$(echo "$rung" | cut -d: -f1); rhs=$(echo "$rung" | cut -d: -f2); extra=$(echo "$rung" | cut -d: -f3)
    name="c5-${lhs}-${rhs}$(echo "${extra:-}" | tr -d '-')"
    wait_for_card "$GPU6" GPU6 1800 || break
    free_card "$GPU4" GPU4 || { echo "[$(stamp)] GPU4 busy; skipping cell 5"; break; }
    echo "[$(stamp)] START $name ${extra:-(vision resident)}"
    # shellcheck disable=SC2086
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU6,$GPU4" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        timeout 5400 "$PY" "$DISTILL/tps_probe.py" --out-dir "$OUT" \
        --cell 27b-lora-split --sequence-length 4096 --micro-batch 1 \
        --device-map sequential --max-memory "0=$lhs,1=$rhs" --loss-chunks 16 \
        --cell-id 27b-lora-split-L4096-mb1 $extra \
        > "$LOGS/$name.log" 2>&1
    rc=$?
    echo "[$(stamp)] END   $name rc=$rc"
    if [ $rc -eq 0 ]; then
        echo "[$(stamp)] cell 5 SUCCEEDED at 0=$lhs,1=$rhs ${extra:-with the vision tower resident}"
        DONE5=1
    else
        grep -E "OutOfMemoryError|ProbeError|Error:" "$LOGS/$name.log" | tail -3
    fi
done
[ "$DONE5" -eq 0 ] && echo "[$(stamp)] cell 5 did NOT complete on any rung of the final ladder"

# ------------------------------------------------- cells 1-4 on one policy --
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
    [ $rc -ne 0 ] && grep -E "OutOfMemoryError|ProbeError|Error:" "$LOGS/$name.log" | tail -3
    return $rc
}

cell 900  f1-08b-L1024-mb1    --cell 08b-full --sequence-length 1024 --micro-batch 1
cell 1200 f1-08b-L4096-mb1    --cell 08b-full --sequence-length 4096 --micro-batch 1
cell 2400 f1-08b-L1024-mbauto --cell 08b-full --sequence-length 1024 --micro-batch auto --micro-batch-max 32
cell 2400 f1-08b-L4096-mbauto --cell 08b-full --sequence-length 4096 --micro-batch auto --micro-batch-max 16
cell 1800 f2-4b-lora-mb1      --cell 4b-lora --sequence-length 4096 --micro-batch 1
cell 1800 f2-4b-lora-acc4     --cell 4b-lora --sequence-length 4096 --micro-batch 1 --accumulation 4
cell 3600 f3-4b-cpuadam-acc1  --cell 4b-full-cpuadam --sequence-length 1024 --micro-batch 1
cell 3600 f3-4b-cpuadam-acc16 --cell 4b-full-cpuadam --sequence-length 1024 --micro-batch 1 --accumulation 16
cell 5400 f4-27b-stream-mb1   --cell 27b-teacher-stream --sequence-length 4096 --micro-batch 1
cell 5400 f4-27b-stream-mb8   --cell 27b-teacher-stream --sequence-length 4096 --micro-batch 8
cell 900  f0-stream-verify    --cell stream-verify --sequence-length 256

echo "[$(stamp)] driver-final done"
sha256sum "$DISTILL/tps_probe.py"
ls -la "$OUT"
