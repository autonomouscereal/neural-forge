#!/usr/bin/env bash
# media-b lane driver r8 (Claude, 2026-10-13): media_b_teacher_cache 1.0.7 (seeded ACE VAE posterior sample). Proves the
# ACE x0 is now reproducible across processes and across the two encode paths:
#   A = ace-encode in a CPU-only container, then ace-dit-run (GPU3) LOADS those stage files
#   B = ace-dit-run (GPU3) encoding inline (GPU container, VAE on CPU), separate root
#   C = a second CPU-only ace-encode, separate root
# x0 A == C and A == B and the A/B shards must be bitwise equal. Then seal + verify A and B, re-verify the r6/r7 caches,
# and run the tests with every current sealed cache. GPU steps use only GPU3 via `launch` (2,000 MiB start rule).
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/media_b_teacher_cache.py
PY=/home/cereal/model-surgery-lab/venv/bin/python
W=$OC/distill/media_b
RA=$W/mini_cache_v107
RB=$W/mini_cache_v107_inline
RC=$W/mini_cache_v107_cpu2
LOGS=$W/logs_v107
CORP=$W/corpora
C=$CORP/ace_clips_mini2.json
export OC CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9 PYTHONPATH=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$RA" "$RB" "$RC" "$LOGS"
STATUS=$W/driver_status_v107.txt
echo "started $(date -u +%FT%TZ) tool $(sha256sum $T | cut -c1-16) driver $(sha256sum $0 | cut -c1-16)" >> "$STATUS"
host() { nice -n 15 "$PY" "$T" "$@"; }
addr() { host address -- "$@" | sed -n 's/^CACHE_DIR //p'; }
wait_gpu3() {
  for i in $(seq 1 240); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d " ")
    [ "$used" -le 2000 ] && return 0; sleep 30
  done
  return 1
}

cd $OC/distill/tools
nice -n 15 "$PY" -m pytest -o addopts="" -p no:cacheprovider -q test_media_b_teacher_cache.py > "$LOGS/tests_before.log" 2>&1
echo "STEP tests-before exit $? $(tail -1 $LOGS/tests_before.log)" >> "$STATUS"

host launch --cpu-only --name media-b-v107-ace-encode-A --log "$LOGS/ace-encode-A.log" --threads 8 -- ace-encode --corpus "$C" --cache-root "$RA"
echo "STEP ace-encode-A exit $?" >> "$STATUS"
host launch --cpu-only --name media-b-v107-ace-encode-C --log "$LOGS/ace-encode-C.log" --threads 8 -- ace-encode --corpus "$C" --cache-root "$RC"
echo "STEP ace-encode-C exit $?" >> "$STATUS"
wait_gpu3 && host launch --card 3 --name media-b-v107-ace-dit-A-gpu3 --log "$LOGS/ace-dit-A.log" --runtime docker -- ace-dit-run --corpus "$C" --cache-root "$RA"
echo "STEP ace-dit-A exit $?" >> "$STATUS"
wait_gpu3 && host launch --card 3 --name media-b-v107-ace-dit-B-gpu3 --log "$LOGS/ace-dit-B.log" --runtime docker -- ace-dit-run --corpus "$C" --cache-root "$RB"
echo "STEP ace-dit-B exit $?" >> "$STATUS"

DA=$(addr ace-dit-run --corpus "$C" --cache-root "$RA"); DB=$(addr ace-dit-run --corpus "$C" --cache-root "$RB")
SA=$(addr ace-encode --corpus "$C" --cache-root "$RA"); SB=$(addr ace-encode --corpus "$C" --cache-root "$RB"); SC=$(addr ace-encode --corpus "$C" --cache-root "$RC")
for d in "$DA" "$DB"; do
  host seal --cache-dir "$d" --corpus "$C" --teacher ace_dit_velocity >> "$LOGS/seal.log" 2>&1; echo "STEP seal exit $? $d" >> "$STATUS"
  host verify --cache-dir "$d" >> "$LOGS/verify.log" 2>&1; echo "STEP verify exit $? $d" >> "$STATUS"
done
nice -n 15 "$PY" - "$SA" "$SB" "$SC" "$DA" "$DB" > "$LOGS/compare.log" 2>&1 <<'EOF'
import sys, json
from pathlib import Path
import torch
from safetensors.torch import load_file
sa, sb, sc, da, db = map(Path, sys.argv[1:6])
for i in range(2):
    xa, xb, xc = (load_file(str(s / f"x0-{i:05d}.safetensors"))["x0"] for s in (sa, sb, sc))
    print(f"clip {i}: x0 CPU-only A vs CPU-only C bitwise {torch.equal(xa, xc)} | A vs inline B bitwise {torch.equal(xa, xb)} max {float((xa - xb).abs().max()):.3e}")
    print(f"   seeds A {json.loads((sa / f'x0-{i:05d}.json').read_text())['audio']['posterior_seed']} B {json.loads((sb / f'x0-{i:05d}.json').read_text())['audio']['posterior_seed']}")
    ta, tb = load_file(str(da / f"shard-{i:05d}.safetensors")), load_file(str(db / f"shard-{i:05d}.safetensors"))
    for k in sorted(ta):
        print(f"   shard {i} {k}: bitwise {torch.equal(ta[k], tb[k])}")
    print(f"   A x0_stage {json.loads((da / f'shard-{i:05d}.json').read_text())['x0_stage']} | B x0_stage {json.loads((db / f'shard-{i:05d}.json').read_text())['x0_stage']}")
EOF
echo "STEP compare exit $? $(grep -c 'bitwise True' $LOGS/compare.log) True, $(grep -c 'bitwise False' $LOGS/compare.log) False" >> "$STATUS"

ALL=$(grep -v ace_dit_velocity $W/logs_v105/sealed_dirs.txt | sort -u | paste -sd,)
for d in $(echo "$ALL" | tr , " "); do host verify --cache-dir "$d" >> "$LOGS/verify.log" 2>&1; echo "STEP verify-1.0.7 exit $? $d" >> "$STATUS"; done
MEDIA_B_MINI_CACHES="$ALL,$DA,$DB" nice -n 15 "$PY" -m pytest -o addopts="" -p no:cacheprovider -q test_media_b_teacher_cache.py > "$LOGS/tests_with_mini.log" 2>&1
echo "STEP tests-with-mini exit $? $(tail -1 $LOGS/tests_with_mini.log)" >> "$STATUS"
echo "finished $(date -u +%FT%TZ)" >> "$STATUS"
