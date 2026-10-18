#!/usr/bin/env bash
# Self-gen lane, step 2: the 1.0M-token sampling run over the corpus lane's self-gen
# prompts, on the parent's Q4_K_M served by `selfgen_sampler.py serve` on GPU1 + GPU2
# (ports 18311 / 18312, 4 slots each). Resumable: re-running this script continues the
# same --out-dir from its journal.
#
# Teardown happens ONLY after a complete run whose shards verify: both servers are
# stopped through their launch receipts (allocator receipts -> released) and the Q4
# GGUF is removed from the declared scratch (STAGE_D_RULINGS "Transient scratch": plain
# rm after use). On exit 1 (functional failure) or 2 (servers gone) nothing is torn
# down, so the run can be inspected or resumed.
#
#   setsid nohup bash run_selfgen_sampling_2026-10-13.sh > $OC/distill/selfgen/logs/run.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
T=$OC/distill/tools
SG=$OC/distill/selfgen
OUT=$SG/run-q38-27bu-q4km-2026-10-13
SCRATCH=/mnt/lexar/omni-campaign/scratch/selfgen-q38-2026-10-13
Q4=$SCRATCH/Q38-27BU-parent.Q4_K_M.gguf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

cd "$SG" || exit 3
touch "$SG/servers.txt"
echo "[$(date -u +%FT%TZ)] run start"
nice -n 15 "$PY" "$T/selfgen_sampler.py" run \
  --server http://127.0.0.1:18311 --server http://127.0.0.1:18312 \
  --servers-file "$SG/servers.txt" \
  --out-dir "$OUT" \
  --export-check-receipt "$SG/receipts/export_check_Q38-27BU-parent.json" \
  --token-target 1000000 --rows-per-shard 64
rc=$?
echo "[$(date -u +%FT%TZ)] run rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

nice -n 15 "$PY" "$T/selfgen_sampler.py" verify --out-dir "$OUT"
vrc=$?
echo "[$(date -u +%FT%TZ)] verify rc=$vrc"
[ "$vrc" -eq 0 ] || exit 4

outcome="self-gen run complete: $OUT/manifest.json sha $(sha256sum "$OUT/manifest.json" | cut -c1-16); verify rc 0"
for g in 1 2; do
  "$PY" "$T/selfgen_sampler.py" stop --launch-receipt "$SG/receipts/serve-selfgen-q38-q4km-gpu$g.json" --outcome "$outcome"
done
case "$Q4" in
  /mnt/lexar/omni-campaign/scratch/selfgen-q38-2026-10-13/*.gguf)
    rm -f -- "$Q4" && echo "[$(date -u +%FT%TZ)] removed transient $Q4" ;;
  *) echo "refusing to remove $Q4: not in the declared scratch" ;;
esac
df -BG / /mnt/lexar
echo "[$(date -u +%FT%TZ)] done"
