#!/usr/bin/env bash
# media-b lane driver r7 (Claude, 2026-10-13): media_b_teacher_cache 1.0.6 (CPU-only ace-encode import fix). Runs AFTER
# r6 (run_media_b_v105_2026-10-13.sh) has finished. (1) tests; (1b) the QVL reference corpus resumed and sealed under
# the 1.0.6 rel_fro gate; (2) ace-encode in a CPU-only container into a separate
# root; (3) ace-dit-run on GPU3 in that root must LOAD both x0 from the stage files; (4) the CPU-container x0 and the
# resulting shards are compared bitwise with the r6 (1.0.5, in-container encode) stage files and shards; (5) every
# r6-sealed cache and the new one re-verify under 1.0.6; (6) tests with all sealed caches.
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/media_b_teacher_cache.py
PY=/home/cereal/model-surgery-lab/venv/bin/python
W=$OC/distill/media_b
MINI=$W/mini_cache_v105
CPUR=$W/mini_cache_v106_cpuenc
LOGS=$W/logs_v106
CORP=$W/corpora
export OC CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9 PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$CPUR" "$LOGS"
STATUS=$W/driver_status_v106.txt
echo "started $(date -u +%FT%TZ) tool $(sha256sum $T | cut -c1-16) driver $(sha256sum $0 | cut -c1-16)" >> "$STATUS"
host() { nice -n 15 "$PY" "$T" "$@"; }
addr() { host address -- "$@" | sed -n 's/^CACHE_DIR //p'; }

cd $OC/distill/tools
nice -n 15 "$PY" -m pytest -o addopts="" -p no:cacheprovider -q test_media_b_teacher_cache.py > "$LOGS/tests_before.log" 2>&1
echo "STEP tests-before exit $? $(tail -1 $LOGS/tests_before.log)" >> "$STATUS"

wait_gpu3() {
  for i in $(seq 1 240); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d " ")
    [ "$used" -le 2000 ] && return 0; sleep 30
  done
  return 1
}

# ---- QVL reference corpus again: 1.0.5 refused image 4 on max_abs (the s7 capture carries 3090 TF32 drift); 1.0.6 gates
# on rel_fro. Same run config, so the 3 shards 1.0.5 wrote resume (COMPAT 1.0.5 + 1.0.6) and 1.0.6 writes the 4th.
Q2=(qvl-run --corpus "$CORP/qvl_b1ref4.json" --cache-root "$MINI")
wait_gpu3 && host launch --card 3 --name media-b-v106-mini-qvl-ref-gpu3 --log "$LOGS/mini-qvl-ref.log" --runtime pascal -- "${Q2[@]}"
echo "STEP mini-qvl-ref gpu3 exit $? $(date -u +%FT%TZ)" >> "$STATUS"
DQ=$(addr "${Q2[@]}")
host seal --cache-dir "$DQ" --corpus "$CORP/qvl_b1ref4.json" --teacher qvl32_vit_l13 > "$LOGS/seal-qvl-ref.log" 2>&1
echo "STEP seal-qvl-ref exit $? $DQ" >> "$STATUS"

host launch --cpu-only --name media-b-v106-ace-encode-cpu --log "$LOGS/ace-encode.log" --threads 8 -- \
  ace-encode --corpus "$CORP/ace_clips_mini2.json" --cache-root "$CPUR"
echo "STEP ace-encode-cpu exit $? $(date -u +%FT%TZ)" >> "$STATUS"

ACE=(ace-dit-run --corpus "$CORP/ace_clips_mini2.json" --cache-root "$CPUR")
wait_gpu3
host launch --card 3 --name media-b-v106-ace-dit-from-stage-gpu3 --log "$LOGS/ace-dit-from-stage.log" --runtime docker -- "${ACE[@]}"
echo "STEP ace-dit-from-stage gpu3 exit $? $(date -u +%FT%TZ)" >> "$STATUS"
D_NEW=$(addr "${ACE[@]}")
host seal --cache-dir "$D_NEW" --corpus "$CORP/ace_clips_mini2.json" --teacher ace_dit_velocity > "$LOGS/seal-ace-dit-cpuenc.log" 2>&1
echo "STEP seal-ace-dit-cpuenc exit $?" >> "$STATUS"

# bitwise: CPU-container stage x0 vs r6 in-container stage x0, and the resulting shards (identical run config)
D_OLD=$(addr ace-dit-run --corpus "$CORP/ace_clips_mini2.json" --cache-root "$MINI")
S_OLD=$(addr ace-encode --corpus "$CORP/ace_clips_mini2.json" --cache-root "$MINI")
S_NEW=$(addr ace-encode --corpus "$CORP/ace_clips_mini2.json" --cache-root "$CPUR")
nice -n 15 "$PY" - "$S_OLD" "$S_NEW" "$D_OLD" "$D_NEW" > "$LOGS/compare.log" 2>&1 <<'EOF'
import sys, json
from pathlib import Path
import torch
from safetensors.torch import load_file
s_old, s_new, d_old, d_new = map(Path, sys.argv[1:5])
print("same relative address:", d_old.relative_to(d_old.parents[2]) == d_new.relative_to(d_new.parents[2]))
for i in range(2):
    a = load_file(str(s_old / f"x0-{i:05d}.safetensors"))["x0"]; b = load_file(str(s_new / f"x0-{i:05d}.safetensors"))["x0"]
    print(f"x0 stage {i}: shape {list(a.shape)} bitwise {torch.equal(a, b)} max_abs_diff {float((a - b).abs().max()):.3e}")
    sa = load_file(str(d_old / f"shard-{i:05d}.safetensors")); sb = load_file(str(d_new / f"shard-{i:05d}.safetensors"))
    for k in sorted(sa):
        eq = torch.equal(sa[k], sb[k])
        diff = float((sa[k].double() - sb[k].double()).abs().max()) if sa[k].numel() else 0.0
        print(f"shard {i} {k}: bitwise {eq} max_abs_diff {diff:.3e}")
    jb = json.loads((d_new / f"shard-{i:05d}.json").read_text())
    print(f"shard {i} new: x0_stage {jb['x0_stage']} tool {jb['tool_version']}")
EOF
echo "STEP compare exit $? $(grep -c 'bitwise True' $LOGS/compare.log) bitwise-True lines, $(grep -c 'bitwise False' $LOGS/compare.log) bitwise-False lines" >> "$STATUS"

: > "$LOGS/verify_all.txt"
for d in $(sort -u $W/logs_v105/sealed_dirs.txt) "$D_NEW"; do
  host verify --cache-dir "$d" >> "$LOGS/verify_all.txt" 2>&1
  echo "STEP verify-1.0.6 exit $? $d" >> "$STATUS"
done
MEDIA_B_MINI_CACHES="$(sort -u $W/logs_v105/sealed_dirs.txt | paste -sd,),$D_NEW" \
  nice -n 15 "$PY" -m pytest -o addopts="" -p no:cacheprovider -q test_media_b_teacher_cache.py > "$LOGS/tests_with_mini.log" 2>&1
echo "STEP tests-with-mini exit $? $(tail -1 $LOGS/tests_with_mini.log)" >> "$STATUS"
echo "finished $(date -u +%FT%TZ)" >> "$STATUS"
