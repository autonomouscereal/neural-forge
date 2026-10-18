#!/usr/bin/env bash
# Final pass: cell 5 inside the feasible split window, then cells 0-4 on one
# probe build.  This is the script that produced the shipped receipts.
#
# WHAT THE CELL-5 ATTEMPTS ESTABLISHED, in order (all at L=4,096, micro 1, fp16,
# `device_map="sequential"` over the RTX 8000 (44.29 GiB) and a 3090 (23.56 GiB),
# hand-rolled LoRA r=64, 50.95 GiB of fp16 weights):
#   0=40,1=20   OOM inside SDPA on the RTX 8000        -> 4 GiB headroom is not enough
#   0=34,1=18 | 0=32,1=19 | 0=30,1=20   modules left on META
#                                        -> accelerate keeps ~2.65 GiB of margin
#                                           per device below the stated budget
#   0=35,1=19   OOM inside the probe's own isfinite() -> replaced by amin/amax
#   0=39,1=15   OOM inside SDPA on the RTX 8000
#   0=37,1=17   non-reentrant checkpointing REFUSES every step: the forward runs
#               the fp32 MATH SDPA backend and the recompute a fused fp16 one
#               (saved [1,24,4096,4096] fp32 vs recomputed [1,24,4096,256] fp16
#               plus a philox seed).  Turning autocast off did not change it.
#   0=37,1=17 --checkpoint-reentrant   reached BACKWARD, died 1.50 GiB short on
#               the RTX 8000 (= 24 x 4096 x 4096 x 4 B, the fp32 score matrix).
#               Placement: cuda:0 18.47e9 params (34.4 GiB), cuda:1 8.88e9.
#   0=35,1=21 --checkpoint-reentrant --drop-vision   the other wall: the 3090's
#               18.7 GiB share died 384 MiB short.  Placement: 17.34e9 / 10.02e9.
#
# So the window is: RTX 8000 <= ~34.2 GiB AND 3090 <= ~18.0 GiB of a 50.95 GiB
# model -- about 1.2 GiB wide, centred on 0=36GiB,1=21GiB.  The shorter-sequence
# rungs bracket it: the fp32 score matrix the Turing half is forced into scales
# as L^2 (1.5 GiB at 4,096, 384 MiB at 2,048), so if the window is empty at
# 4,096 the campaign still gets a measured 27B-split rate with an explicit L^2
# note rather than nothing at all.
#
#   cd /home/cereal/omni-campaign/distill
#   cp run_tps_final_c_2026-09-02.sh .running-driver-c.sh   # never overwrite a
#   setsid nohup ./.running-driver-c.sh > logs/driver-c.log 2>&1 < /dev/null &  # running script
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
    [ -n "$used" ] && [ "$used" -le 1024 ] && [ -z "$apps" ]
}
# WAIT, NEVER SKIP AND NEVER SIGNAL.  The orchestrator holds GPU4 for its own
# calibration arms and the qwen-fold lane takes GPU6 between builds.  A driver
# that skips a cell because a peer is on the card silently loses the cell; a
# driver that kills the peer is worse.  So each cell waits, printing who holds
# the card, until the card is genuinely free or the deadline passes.
wait_for_card() {   # wait_for_card <uuid> <label> <deadline-seconds>
    local deadline=$(( $(date +%s) + $3 )) shown=""
    while ! free_card "$1" "$2"; do
        local who="$(apps_on "$1")"
        if [ "$who" != "$shown" ]; then
            echo "[$(stamp)] waiting for $2, held by: ${who:-unknown} ($(used_mib "$1") MiB)"
            shown="$who"
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "[$(stamp)] DEADLINE waiting for $2; giving up on this cell"
            return 1
        fi
        sleep 20
    done
    echo "[$(stamp)] $2 free ($(used_mib "$1") MiB)"
    return 0
}

echo "[$(stamp)] driver-c start; probe sha256 $(sha256sum "$DISTILL/tps_probe.py" | cut -d' ' -f1)"

DONE5=0
for rung in 36GiB:21GiB:4096 36GiB:21GiB:2048 37GiB:20GiB:4096 36GiB:21GiB:1024; do
    [ "$DONE5" -eq 1 ] && break
    lhs=$(echo "$rung" | cut -d: -f1)
    rhs=$(echo "$rung" | cut -d: -f2)
    len=$(echo "$rung" | cut -d: -f3)
    name="c5w-${lhs}-${rhs}-L${len}"
    wait_for_card "$GPU6" GPU6 3600 || break
    wait_for_card "$GPU4" GPU4 3600 || break
    echo "[$(stamp)] START $name (reentrant checkpoint, vision dropped)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU6,$GPU4" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        timeout 5400 "$PY" "$DISTILL/tps_probe.py" --out-dir "$OUT" \
        --cell 27b-lora-split --sequence-length "$len" --micro-batch 1 \
        --device-map sequential --max-memory "0=$lhs,1=$rhs" --loss-chunks 32 \
        --checkpoint-reentrant --drop-vision \
        > "$LOGS/$name.log" 2>&1
    rc=$?
    echo "[$(stamp)] END   $name rc=$rc"
    if [ $rc -eq 0 ]; then
        echo "[$(stamp)] cell 5 SUCCEEDED at 0=$lhs,1=$rhs, L=$len"
        DONE5=1
    else
        grep -hE "OutOfMemoryError|ProbeError|CheckpointError" "$LOGS/$name.log" | tail -2
        grep -h "placement:" "$LOGS/$name.log" | tail -1
    fi
done
[ "$DONE5" -eq 0 ] && echo "[$(stamp)] cell 5 did NOT complete on any rung"

cell() {
    local seconds="$1" name="$2"; shift 2
    wait_for_card "$GPU4" GPU4 3600 || { echo "[$(stamp)] gave up on $name"; return 1; }
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

cell 900  h1-08b-L1024-mb1    --cell 08b-full --sequence-length 1024 --micro-batch 1
cell 1200 h1-08b-L4096-mb1    --cell 08b-full --sequence-length 4096 --micro-batch 1
cell 2400 h1-08b-L1024-mbauto --cell 08b-full --sequence-length 1024 --micro-batch auto --micro-batch-max 32
cell 2400 h1-08b-L4096-mbauto --cell 08b-full --sequence-length 4096 --micro-batch auto --micro-batch-max 16
cell 1800 h2-4b-lora-mb1      --cell 4b-lora --sequence-length 4096 --micro-batch 1
cell 1800 h2-4b-lora-acc4     --cell 4b-lora --sequence-length 4096 --micro-batch 1 --accumulation 4
cell 3600 h3-4b-cpuadam-acc1  --cell 4b-full-cpuadam --sequence-length 1024 --micro-batch 1
cell 3600 h3-4b-cpuadam-acc16 --cell 4b-full-cpuadam --sequence-length 1024 --micro-batch 1 --accumulation 16
cell 5400 h4-27b-stream-mb1   --cell 27b-teacher-stream --sequence-length 4096 --micro-batch 1
cell 5400 h4-27b-stream-mb8   --cell 27b-teacher-stream --sequence-length 4096 --micro-batch 8
cell 900  h0-stream-verify    --cell stream-verify --sequence-length 256

echo "[$(stamp)] driver-c done"
sha256sum "$DISTILL/tps_probe.py"
ls -la "$OUT"
