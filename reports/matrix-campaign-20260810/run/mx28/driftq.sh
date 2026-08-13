#!/bin/bash
# Wave 28, TIER D: bisect the joint-solve code drift wave 27 found and could not
# attribute.
#
# THE FINDING BEING BISECTED.  Wave 27's `bodyq.sh` rebuilt wave 17's sealed
# `ARCH.F1301KV6Q` (blocks 1 and 13 excised, 16->12 joint head merge) and the
# guard refused: identical 200,847,104 parameters, sealed digest
# 90a9e5187feef678..., rebuilt ad406f480a181901....  A second rebuild was
# BYTE-IDENTICAL to the first, so it is code drift and not nondeterminism.  Wave
# 26 had proved the PURE EXCISION path rebuilds bit-exactly across nine waves,
# and wave 27 had quietly generalised that to "excision bodies reproduce".
#
# WHY A `git bisect run` OVER COMMITS IS THE WRONG INSTRUMENT HERE, and this is
# the whole reason this file exists instead of a one-line bisect invocation.
# This campaign does not EDIT one builder; every wave COPIES the previous wave's
# builder to a new path and edits the copy.  `run/mx17/mx17_build.py` is still in
# the tree at HEAD, byte-for-byte as wave 17 ran it.  So the drift is not a
# commit that changed a file -- it is a per-wave FORK, and the bisect variable is
# the WAVE, not the commit.  Every builder from mx17 to mx28 is present in one
# deployment and each takes the same six flags for this body, so the whole
# bisect runs against ONE deployment with no checkout, no rebuild and no
# stashing.  A `git bisect` would have reported "no commit changed
# mx17_build.py since wave 17", which is true and useless.
#
# Each rung is a ~70-115 second GPU build.  Eleven rungs is under twenty minutes
# and the interval is small enough that the linear sweep costs less than the
# bookkeeping of a binary search, so this runs them all and prints the first
# wave whose digest leaves wave 17's.
#
#   driftq.sh <gpu> [wave...]      # default: 17 18 19 20 21 22 23 24 25 26 27 28
#   driftq.sh --check-vocab        # free: the vocabulary path across a wave edge
set -u
. /home/cereal/nf-mx28/env.sh
M=/mnt/ai_data/neural-forge-matrix
SEALED=90a9e5187feef6780a0503dc0757b97b6b18db0bc441fd5a51f2d8811f170228
W27REBUILD=ad406f480a181901db35b7c1af6ede976475ce9eece7b3ec36f0878fbbb76f09

# The vocabulary path's own reproduction test, which costs one CPU pass and no
# GPU: wave 28 rebuilds V18X13X01 because it needs the zero-solve control, and
# wave 27 sealed that child's digest.  A vocabulary fold is a GATHER, so this is
# the control for the claim that only SOLVED paths drift.
if [ "${1:-}" = "--check-vocab" ]; then
  $PY - <<'EOF'
import json
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave28")
sealed = json.loads(
    Path("/mnt/ai_data/neural-forge-matrix/wave27/cells/ARCH.V18X13X01/cell-receipt.json").read_text()
)["architecture"]
got = json.loads((W / "children/MX28-V18X13X01/mx28-build-receipt.json").read_text())
print(f"  wave 27 sealed  {sealed['child_tensors_sha256']}")
print(f"  wave 28 rebuilt {got['child_tensors_sha256']}")
same = sealed["child_tensors_sha256"] == got["child_tensors_sha256"]
print("  VOCABULARY PATH REPRODUCES BIT-EXACTLY" if same else "  *** VOCABULARY PATH DRIFTED TOO ***")
raise SystemExit(0 if same else 9)
EOF
  exit $?
fi

GPU="${1:?gpu}"; shift
WAVES="${*:-17 18 19 20 21 22 23 24 25 26 27 28}"
mkdir -p $W/drift $W/logs

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi

echo "sealed  (wave 17) $SEALED"
echo "rebuilt (wave 27) $W27REBUILD"
echo
for N in $WAVES; do
  B=$NF_DEPLOY/reports/matrix-campaign-20260810/run/mx$N/mx${N}_build.py
  if [ ! -f "$B" ]; then echo "wave $N: NO BUILDER at $B"; continue; fi
  OUT=$W/drift/D$N
  rm -rf $OUT
  # The SAME six flags wave 17 sealed and wave 27 rebuilt with, read off the
  # sealed build receipt's own fields: arm, excise 1, excise 13, kv-groups 6,
  # kv-solve joint, kv-als 0.  Nothing else moves.
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $B \
    --mode build --arm conditioned-local \
    --excise 1 --excise 13 --kv-groups 6 --kv-solve joint --kv-als 0 \
    --child $OUT --device cuda:0 \
    > $W/logs/drift-$N.log 2>&1
  rc=$?
  R=$OUT/mx$N-build-receipt.json
  if [ $rc -ne 0 ] || [ ! -f "$R" ]; then
    echo "wave $N: BUILD FAILED rc=$rc"; tail -4 $W/logs/drift-$N.log; continue
  fi
  SHA=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['child_tensors_sha256'])" $R)
  PAR=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['child_parameters'])" $R)
  MARK="?"
  [ "$SHA" = "$SEALED" ] && MARK="== wave 17 sealed"
  [ "$SHA" = "$W27REBUILD" ] && MARK="== wave 27 rebuild"
  MASK=$(grep -c "layer_types re-derived" $W/logs/drift-$N.log)
  echo "wave $N: $SHA  params $PAR  live-mask-repair=$MASK  $MARK"
  # The weights are 800 MB a rung and the DIGEST is the whole measurement.
  rm -rf $OUT
done
echo "DRIFTQ $GPU DONE"
