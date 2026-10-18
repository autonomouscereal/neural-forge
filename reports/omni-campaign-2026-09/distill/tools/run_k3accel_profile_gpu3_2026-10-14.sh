#!/bin/bash
# run_k3accel_profile_gpu3_2026-10-14.sh -- lane k3accel: profile the FROZEN k3_teacher_cache 1.2.4 (2aae4dc3) on GPU3
# (P40) at the production row-group geometry (65,536 positions = 16 real K3 rows x 4,096 from corpus/k3 pass-0, one
# row group; --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 as queue_gpu6.sh 1.3.0 runs the passes), on the
# window L83-L85 (MLA with 7 blocks resident and a non-contiguous block slice; KDA with the L84 block push; KDA with
# all 8 slots, contiguous).  --synthetic-blocks 7 fills the slots (timing / memory only, not teacher numerics).
#   A  baseline: the frozen tool itself under py-spy + per-thread CPU / read MB/s / MemAvailable sampler + dmon
#   B  split:    k3_accel_profile split (stage timers, replays) on the same window
#   C  compare:  prefix_out_sha256 of A vs B per layer (the timed copies compute the same numbers)
# Writes ONLY under /mnt/lexar/omni-campaign/scratch/k3accel/.  Claims GPU3 in $Q/p40.claims and releases it.
# Host-RAM rule: starts at MemAvailable >= START_GIB (75), the sampler SIGTERMs the run below 40 GiB.  nice 15, ionice
# best-effort 7 (the GPU6 B2 pass reads the same Lexar SSD).
set -u
OC=/home/cereal/omni-campaign; Q=$OC/distill/queues; T=$OC/distill/tools
S=/mnt/lexar/omni-campaign/scratch/k3accel
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
R=$S/runs/$STAMP
PP=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
PYSPY=$S/pyspy/bin/py-spy
FROZEN=$Q/frozen/k3_teacher_cache-gpu6.py
WANT=2aae4dc3134dda725e3ee35572584fbf35532e0c4a7a5721a0b0c817fc2c4f9d
TOOLCOPY=$S/src/k3_teacher_cache-1.2.4.${WANT:0:16}.py
START_GIB=${START_GIB:-75}; FLOOR_GIB=${FLOOR_GIB:-40}
GT=${GT:-65536}
mkdir -p "$R" "$S/src" "$S/bench"
exec >> "$R/driver.log" 2>&1
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "== k3accel profile $(utc) run dir $R"

cp -f "$FROZEN" "$TOOLCOPY.tmp" && mv -f "$TOOLCOPY.tmp" "$TOOLCOPY" && chmod a-w "$TOOLCOPY"
GOT=$(sha256sum "$TOOLCOPY" | cut -c1-64)
[ "$GOT" = "$WANT" ] || { echo "REFUSED: tool copy sha $GOT != $WANT"; exit 2; }
echo "tool copy $TOOLCOPY sha $GOT; profiler $(sha256sum "$T/k3_accel_profile.py" | cut -c1-64)"

# GPU3 quiet twice, MemAvailable >= START_GIB
quiet=0
while [ $quiet -lt 2 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
  [ "$used" -le 2000 ] && quiet=$((quiet + 1)) || quiet=0
  [ $quiet -lt 2 ] && sleep 20
done
while :; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge "$START_GIB" ] && break
  echo "$(utc) MemAvailable $avail GiB < $START_GIB; waiting"; sleep 60
done
echo "$(utc) GPU3 ${used} MiB, MemAvailable ${avail} GiB"
echo "$(utc) GPU3 lane=k3accel owner=claude job=k3-profile-1.2.4 pid=$$ log=$R/driver.log" >> "$Q/p40.claims"
release() { echo "$(utc) GPU3 lane=k3accel job=k3-profile-1.2.4 released" >> "$Q/p40.claims"; }
trap release EXIT

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
# windows: tag:layers:rows:truncate:synthetic_blocks:moe_exp_layers:kda_exp_layers
#   w1 = KDA L84 (block push, 7 resident, non-contiguous slice) + KDA L85 (8 resident, contiguous): EXACT production
#        geometry (16 rows x 4,096 = 65,536 positions, one row group)
#   w2 = MLA L83: on the P40 SDPA takes the math kernel (no mem-efficient kernel for sm_61 at qk 192 / v 128) and
#        needs 5.71 GiB of scores at T = 4,096 beside the 17 GB layer -> OOM (run 18:16Z, both A and B); so 32 rows
#        truncated to 2,048 (--truncate-seq, a probe option) = the same 65,536 positions / row group; only the MLA
#        score cost is not the production one
WINDOWS=${WINDOWS:-"w1:84-85:16:0:7:84:85 w2:83:32:2048:7::"}
side() {  # $1 tag: dmon on GPU3 + iostat on the Lexar disk, background
  nvidia-smi dmon -i 3 -s puc -d 2 -o T > "$R/dmon_$1.txt" 2>&1 & DMON=$!
  iostat -dmx sdb 5 > "$R/iostat_$1.txt" 2>&1 & IOST=$!
}
unside() { kill $DMON $IOST 2>/dev/null; wait $DMON $IOST 2>/dev/null; }

for W in $WINDOWS; do
  IFS=: read -r TAG LAYERS ROWS TRUNC SYN MOE_EXP KDA_EXP <<< "$W"
  RUNARGS=(run --input-ids "$OC/distill/corpus/k3/pass-0/k3_input_ids.safetensors" --pass-count 1 --pass-index 0
    --row-limit "$ROWS" --probe --layers "$LAYERS" --synthetic-blocks "$SYN"
    --device cuda:0 --experts-device cuda:0 --kda-backend torch --compute-dtype auto
    --row-chunk-kda 1 --row-chunk-mla 1 --token-chunk 1024 --group-tokens "$GT" --prefix-digests)
  [ "$TRUNC" != 0 ] && RUNARGS+=(--truncate-seq "$TRUNC")
  if [ "${SKIP_A:-0}" != 1 ]; then
    echo "== $TAG A baseline layers $LAYERS rows $ROWS truncate $TRUNC $(utc)"
    side "${TAG}_A"
    "$PP" "$T/k3_accel_profile.py" monitor --out "$R/monitor_${TAG}_A.jsonl" --interval 5 --mem-floor-gib "$FLOOR_GIB" --       nice -n 15 ionice -c2 -n7 "$PYSPY" record --rate 20 --idle --threads --nonblocking --format raw         -o "$R/pyspy_${TAG}_A.txt" -- "$PP" "$TOOLCOPY" "${RUNARGS[@]}" --pass-dir "$R/${TAG}_A-probe" > "$R/${TAG}_A.log" 2>&1
    RCA=$?; unside
    echo "$TAG A exit=$RCA $(utc)"
    [ $RCA -eq 75 ] && { echo "$TAG A stopped by the RAM floor; stopping"; exit 75; }
  fi
  echo "== $TAG B split $(utc)"
  side "${TAG}_B"
  EXP=(--experiments --moe-exp-layers "$MOE_EXP" --kda-exp-layers "$KDA_EXP")
  "$PP" "$T/k3_accel_profile.py" monitor --out "$R/monitor_${TAG}_B.jsonl" --interval 5 --mem-floor-gib "$FLOOR_GIB" --     nice -n 15 ionice -c2 -n7 "$PP" "$T/k3_accel_profile.py" split --tool "$TOOLCOPY" --out "$R/split_${TAG}.json"       "${EXP[@]}" --bench-dir "$S/bench" -- "${RUNARGS[@]}" --pass-dir "$R/${TAG}_B-probe" > "$R/${TAG}_B.log" 2>&1
  RCB=$?; unside
  echo "$TAG B exit=$RCB $(utc)"
  [ $RCB -eq 75 ] && { echo "$TAG B stopped by the RAM floor; stopping"; exit 75; }
  if [ -s "$R/${TAG}_A-probe/probe_receipt.json" ] && [ -s "$R/split_${TAG}.json" ]; then
    "$PP" "$T/k3_accel_profile.py" compare "$R/${TAG}_A-probe/probe_receipt.json" "$R/split_${TAG}.json"       --out "$R/compare_${TAG}_A_B.json"
    echo "$TAG compare exit=$?"
  fi
done
sha256sum "$R"/*.json "$R"/*/probe_receipt.json 2>/dev/null
echo "== done $(utc)"
