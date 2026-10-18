#!/usr/bin/env bash
# media-b lane driver r6 (Claude, 2026-10-13): media_b_teacher_cache 1.0.5 review-fix rerun. Tests, then every mini cache
# again into a NEW root (1.0.5 addresses carry the run config; the 1.0.4 mini caches stay untouched as evidence), with
# the new reference arms (HY3D upstream pipeline, QVL vs the s7 capture), the CPU-only ace-encode stage path, a resume
# rerun, seal + verify, and the tests again against the sealed 1.0.5 caches.
# Run detached:  setsid nohup bash run_media_b_v105_2026-10-13.sh > $OC/distill/media_b/logs_v105/driver.log 2>&1 < /dev/null &
# GPU steps use ONLY GPU3 (GPU1/2 run the self-gen sampler; 0/4/6 are step 1's; 5 is reserved) via `launch`, which refuses
# any card above 2,000 MiB and writes $OC/composite/receipts/allocator/<name>.json.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/media_b_teacher_cache.py
PY=/home/cereal/model-surgery-lab/venv/bin/python
W=$OC/distill/media_b
MINI=$W/mini_cache_v105
LOGS=$W/logs_v105
CORP=$W/corpora
export OC CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9 PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$MINI" "$LOGS"
STATUS=$W/driver_status_v105.txt
echo "started $(date -u +%FT%TZ) tool $(sha256sum $T | cut -c1-16) driver $(sha256sum $0 | cut -c1-16)" >> "$STATUS"

host() { nice -n 15 "$PY" "$T" "$@"; }
addr() { host address -- "$@" | sed -n 's/^CACHE_DIR //p'; }

free_gpu3() {  # GPU3 at or under 2000 MiB; waits up to 2 h
  for i in $(seq 1 240); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d " ")
    if [ "$used" -le 2000 ]; then return 0; fi
    sleep 30
  done
  return 1
}

gpu_step() {  # gpu_step <name> <runtime> <worker args...>
  local name=$1 runtime=$2; shift 2
  free_gpu3 || { echo "STEP $name: GPU3 not free in 2 h" | tee -a "$STATUS"; return 9; }
  host launch --card 3 --name "media-b-v105-$name-gpu3" --log "$LOGS/$name.log" --runtime "$runtime" -- "$@"
  local rc=$?
  echo "STEP $name gpu3 exit $rc $(date -u +%FT%TZ)" | tee -a "$STATUS"
  return $rc
}

seal_verify() {  # seal_verify <name> <teacher> <corpus> <worker args...>
  local name=$1 teacher=$2 corpus=$3; shift 3
  local d; d=$(addr "$@")
  host seal --cache-dir "$d" --corpus "$corpus" --teacher "$teacher" > "$LOGS/seal-$name.log" 2>&1
  echo "STEP seal-$name exit $? $d" >> "$STATUS"
  host verify --cache-dir "$d" > "$LOGS/verify-$name.log" 2>&1
  echo "STEP verify-$name exit $? $(tail -1 $LOGS/verify-$name.log)" >> "$STATUS"
  echo "$d" >> "$LOGS/sealed_dirs.txt"
}

cd $OC/distill/tools
nice -n 15 "$PY" -m pytest -o addopts="" -p no:cacheprovider -q test_media_b_teacher_cache.py > "$LOGS/tests_before.log" 2>&1
echo "STEP tests-before exit $? $(tail -1 $LOGS/tests_before.log)" >> "$STATUS"

# ---- the QVL reference corpus: 4 batch1 objects (512x512 = an s7 square side)
[ -s "$CORP/qvl_b1ref4.json" ] || host corpus-qvl --image-dirs /mnt/lexar/omni-campaign/qimg-broad-calibration-sep07/batch1 --limit 4 --out "$CORP/qvl_b1ref4.json"
echo "STEP corpus-qvl-b1ref4 exit $?" >> "$STATUS"

gpu_step selfcheck docker selfcheck --stack all

# ---- ACE DiT: CPU-only encode into x0 stage files, then the DiT run must LOAD them, then a resume rerun skips all
ACE=(ace-dit-run --corpus "$CORP/ace_clips_mini2.json" --cache-root "$MINI")
host launch --cpu-only --name media-b-v105-ace-encode-cpu --log "$LOGS/ace-encode.log" --threads 8 -- \
  ace-encode --corpus "$CORP/ace_clips_mini2.json" --cache-root "$MINI"
echo "STEP ace-encode-cpu exit $? $(date -u +%FT%TZ)" >> "$STATUS"
gpu_step mini-ace-dit docker "${ACE[@]}"
gpu_step mini-ace-dit-resume docker "${ACE[@]}"
seal_verify ace-dit ace_dit_velocity "$CORP/ace_clips_mini2.json" "${ACE[@]}"
echo "ADDRESS ace-dit cfg1 $(addr "${ACE[@]}")" >> "$STATUS"
echo "ADDRESS ace-dit cfg2 $(addr "${ACE[@]}" --cfg 2)" >> "$STATUS"
echo "ADDRESS ace-encode $(addr ace-encode --corpus "$CORP/ace_clips_mini2.json" --cache-root "$MINI")" >> "$STATUS"

# ---- ACE LM (reference arm: ComfyUI generate_audio_codes on the first prompt)
LM=(ace-lm-run --corpus "$CORP/ace_prompts_mini2.json" --cache-root "$MINI")
gpu_step mini-ace-lm docker "${LM[@]}"
seal_verify ace-lm ace_lm_codes "$CORP/ace_prompts_mini2.json" "${LM[@]}"

# ---- QVL: the registry mini (determinism replay) and the batch1 reference corpus (vs the s7 capture)
Q1=(qvl-run --corpus "$CORP/qvl_mini4.json" --cache-root "$MINI")
Q2=(qvl-run --corpus "$CORP/qvl_b1ref4.json" --cache-root "$MINI")
gpu_step mini-qvl pascal "${Q1[@]}"
seal_verify qvl qvl32_vit_l13 "$CORP/qvl_mini4.json" "${Q1[@]}"
gpu_step mini-qvl-ref pascal "${Q2[@]}"
seal_verify qvl-ref qvl32_vit_l13 "$CORP/qvl_b1ref4.json" "${Q2[@]}"

# ---- HY3D: 4 shapes, determinism replay every 2nd item, upstream-pipeline reference on the first, meshes required
H=(hy3d-run --corpus "$CORP/hy3d_mini4.json" --cache-root "$MINI" --replay-every 2 --reference-first 1 --require-mesh)
gpu_step mini-hy3d docker "${H[@]}"
seal_verify hy3d hy3d_trajectories "$CORP/hy3d_mini4.json" "${H[@]}"

# ---- tests again with the sealed 1.0.5 caches as real artefacts
cd $OC/distill/tools
MEDIA_B_MINI_CACHES=$(paste -sd, "$LOGS/sealed_dirs.txt") \
  nice -n 15 "$PY" -m pytest -o addopts="" -p no:cacheprovider -q test_media_b_teacher_cache.py > "$LOGS/tests_with_mini.log" 2>&1
echo "STEP tests-with-mini exit $? $(tail -1 $LOGS/tests_with_mini.log)" >> "$STATUS"
echo "finished $(date -u +%FT%TZ)" >> "$STATUS"
