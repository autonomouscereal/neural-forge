#!/usr/bin/env bash
# media-b lane driver (Claude, 2026-10-12): P4 probes + mini caches on ONE P40 at a time, then the stage-C corpora.
# Run detached:  setsid nohup bash run_media_b_probes_2026-10-12.sh > $OC/distill/media_b/logs/driver.log 2>&1 < /dev/null &
# Every GPU step goes through `media_b_teacher_cache.py launch`, which refuses GPU 0/4/5/6 and any card above 2,000 MiB
# and writes $OC/composite/receipts/allocator/<name>.json. A step waits for a free P40 in the order 3, 1, 2.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/media_b_teacher_cache.py
PY=/home/cereal/model-surgery-lab/venv/bin/python
W=$OC/distill/media_b
MINI=$W/mini_cache
LOGS=$W/logs
CORP=$W/corpora
TPS=$OC/distill/tps
export OC CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9 PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$MINI" "$LOGS" "$CORP" "$TPS"
STATUS=$W/driver_status.txt
echo "started $(date -u +%FT%TZ) tool $(sha256sum $T | cut -c1-16)" >> "$STATUS"

host() { nice -n 15 "$PY" "$T" "$@"; }

free_card() {  # first P40 in 3,1,2 at or under 2000 MiB; waits up to 2 h
  for i in $(seq 1 240); do
    for c in 3 1 2; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$c" | tr -d " ")
      if [ "$used" -le 2000 ]; then echo "$c"; return 0; fi
    done
    sleep 30
  done
  return 1
}

gpu_step() {  # gpu_step <name> <runtime> <worker args...>
  local name=$1 runtime=$2; shift 2
  local card; card=$(free_card) || { echo "STEP $name: no free P40 in 2 h" | tee -a "$STATUS"; return 9; }
  host launch --card "$card" --name "media-b-$name-gpu$card" --log "$LOGS/$name.log" --runtime "$runtime" -- "$@"
  local rc=$?
  echo "STEP $name gpu$card exit $rc $(date -u +%FT%TZ)" | tee -a "$STATUS"
  return $rc
}

# ---- corpora (CPU)
host corpus-hy3d --image-dirs /mnt/lexar/omni-campaign/qimg-broad-calibration-sep07/batch1 \
  --names b1_fit_copper_kettle,b1_fit_toy_robot,b1_held_glass_pitcher,b1_held_stone_turtle --out "$CORP/hy3d_mini4.json"
echo "STEP corpus-hy3d-mini exit $?" >> "$STATUS"
host corpus-ace --max-rows 40 --limit 2 --out "$CORP/ace_clips_mini2.json"
echo "STEP corpus-ace-mini exit $?" >> "$STATUS"
host corpus-ace --kind prompts --max-rows 40 --limit 2 --seconds 10 --out "$CORP/ace_prompts_mini2.json"
echo "STEP corpus-ace-prompts-mini exit $?" >> "$STATUS"
host corpus-qvl --registry --max-rows 300 --limit 4 --out "$CORP/qvl_mini4.json"
echo "STEP corpus-qvl-mini exit $?" >> "$STATUS"

# ---- runtime self-checks
gpu_step selfcheck-docker docker selfcheck --stack all
gpu_step selfcheck-pascal pascal selfcheck --stack qvl

# ---- P4 probes
# skip a probe whose sealed receipt already exists (resume after the 1.0.1 pause)
[ -s "$TPS/p4_hy3d_p40_fp32_2026-10-12.json" ] && echo "STEP p4-hy3d skipped (receipt exists)" >> "$STATUS" || \n  gpu_step p4-hy3d docker hy3d-probe --corpus "$CORP/hy3d_mini4.json" --out "$TPS/p4_hy3d_p40_fp32_2026-10-12.json"
passed() {  # passed <probe receipt>: exists and has no failed phase
  [ -s "$1" ] && python3 -c "import json,sys; sys.exit(1 if json.load(open('$1')).get('failed_phases') else 0)"
}
passed "$TPS/p4_ace_p40_fp32_2026-10-12.json" && echo "STEP p4-ace skipped (passed receipt exists)" >> "$STATUS" || \
  gpu_step p4-ace docker ace-probe --corpus "$CORP/ace_clips_mini2.json" --out "$TPS/p4_ace_p40_fp32_2026-10-12.json"

# ---- mini caches (end to end) + seal + verify
HSHA=$(python3 -c "import json;print(json.load(open('$CORP/hy3d_mini4.json'))['corpus_sha256'])")
if [ -s "$MINI/hy3d_trajectories/$HSHA/manifest.json" ]; then
  echo "STEP mini-hy3d + seal-hy3d skipped (sealed manifest exists)" >> "$STATUS"
else
  gpu_step mini-hy3d docker hy3d-run --corpus "$CORP/hy3d_mini4.json" --cache-root "$MINI" --replay-every 2 --require-mesh
  host seal --cache-dir "$MINI/hy3d_trajectories/$HSHA" --corpus "$CORP/hy3d_mini4.json" --teacher hy3d_trajectories
  echo "STEP seal-hy3d exit $?" >> "$STATUS"
fi
ASHA=$(python3 -c "import json;print(json.load(open('$CORP/ace_clips_mini2.json'))['corpus_sha256'])")
if [ -s "$MINI/ace_dit_velocity/$ASHA/manifest.json" ]; then
  echo "STEP mini-ace-dit + seal skipped (sealed manifest exists)" >> "$STATUS"
else
  gpu_step mini-ace-dit docker ace-dit-run --corpus "$CORP/ace_clips_mini2.json" --cache-root "$MINI"
  host seal --cache-dir "$MINI/ace_dit_velocity/$ASHA" --corpus "$CORP/ace_clips_mini2.json" --teacher ace_dit_velocity
  echo "STEP seal-ace-dit exit $?" >> "$STATUS"
fi
LSHA=$(python3 -c "import json;print(json.load(open('$CORP/ace_prompts_mini2.json'))['corpus_sha256'])")
if [ -s "$MINI/ace_lm_codes/$LSHA/manifest.json" ]; then
  echo "STEP mini-ace-lm + seal skipped (sealed manifest exists)" >> "$STATUS"
else
  gpu_step mini-ace-lm docker ace-lm-run --corpus "$CORP/ace_prompts_mini2.json" --cache-root "$MINI"
  host seal --cache-dir "$MINI/ace_lm_codes/$LSHA" --corpus "$CORP/ace_prompts_mini2.json" --teacher ace_lm_codes
  echo "STEP seal-ace-lm exit $?" >> "$STATUS"
fi
QSHA=$(python3 -c "import json;print(json.load(open('$CORP/qvl_mini4.json'))['corpus_sha256'])")
if [ -s "$MINI/qvl32_vit_l13/$QSHA/manifest.json" ]; then
  echo "STEP mini-qvl + seal skipped (sealed manifest exists)" >> "$STATUS"
else
  gpu_step mini-qvl pascal qvl-run --corpus "$CORP/qvl_mini4.json" --cache-root "$MINI"
  host seal --cache-dir "$MINI/qvl32_vit_l13/$QSHA" --corpus "$CORP/qvl_mini4.json" --teacher qvl32_vit_l13
  echo "STEP seal-qvl exit $?" >> "$STATUS"
fi

# ---- tests again, now with the sealed mini caches as real artefacts
cd $OC/distill/tools
MEDIA_B_MINI_CACHES="$MINI/hy3d_trajectories/$HSHA,$MINI/ace_dit_velocity/$ASHA,$MINI/ace_lm_codes/$LSHA,$MINI/qvl32_vit_l13/$QSHA" \
  nice -n 15 "$PY" -m pytest -o addopts="" -p no:cacheprovider -q test_media_b_teacher_cache.py > "$LOGS/tests_with_mini.log" 2>&1
echo "STEP tests-with-mini exit $? $(tail -1 $LOGS/tests_with_mini.log)" >> "$STATUS"

# ---- stage-C corpora (CPU): what the real jobs read
[ -s "$CORP/ace_clips_stagec_1024.json" ] || host corpus-ace --limit 1024 --out "$CORP/ace_clips_stagec_1024.json"
echo "STEP corpus-ace-1024 exit $?" >> "$STATUS"
[ -s "$CORP/ace_prompts_stagec_6700.json" ] || host corpus-ace --kind prompts --seconds 60 --seeds-per-prompt 3 --limit 6700 --out "$CORP/ace_prompts_stagec_6700.json"
echo "STEP corpus-ace-prompts exit $?" >> "$STATUS"
[ -s "$CORP/qvl_stagec_512.json" ] || host corpus-qvl --registry --limit 512 --out "$CORP/qvl_stagec_512.json"
echo "STEP corpus-qvl-512 exit $?" >> "$STATUS"
echo "finished $(date -u +%FT%TZ)" >> "$STATUS"
