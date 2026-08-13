#!/bin/bash
# Wave 27: rebuild a SEALED body that was not retained, so a vocabulary fold has
# something to compose onto.
#
# This wave stacks onto seven bodies and six of them are on the share.  The
# seventh -- wave 17's `ARCH.F1301KV6Q`, blocks 1 and 13 excised plus the 16->12
# joint head merge -- has a sealed cell and no checkpoint, because wave 17's
# retention dropped it.  Wave 26 established that this is safe to do: its
# `--excise 13 --excise 7` rebuild reproduced wave 17's sealed `ARCH.P1307`
# BIT-IDENTICALLY nine waves and one rewritten excision path later.  So the body
# is rebuilt here and the rebuild is CHECKED against the sealed digest rather
# than trusted -- if it does not reproduce, the composed child must not claim the
# sealed cell's provenance and this queue says so instead of continuing.
#
#   bodyq.sh <gpu> <TAG>:<sealed cell receipt to check against> -- <build args...>
set -u
. /home/cereal/nf-mx27/env.sh
GPU="${1:?gpu}"; shift
mkdir -p $W/children $W/logs

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi

TAG=$(echo "$1" | cut -d: -f1)
EXPECT=$(echo "$1" | cut -d: -f2-)
shift
[ "${1:-}" = "--" ] && shift

rm -rf $W/children/MX27-$TAG
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX27/mx27_build.py \
  --mode build --arm conditioned-local \
  --child $W/children/MX27-$TAG --device cuda:0 "$@" \
  > $W/logs/build-$TAG.log 2>&1
rc=$?
echo "$TAG rc=$rc :: $(tail -1 $W/logs/build-$TAG.log)"
if [ $rc -ne 0 ]; then tail -15 $W/logs/build-$TAG.log; exit $rc; fi

# The reproduction check.  A body that does not reproduce is not the sealed body,
# and a composed child built on it may not cite the sealed cell.
# The sealed digest lives at architecture.child_tensors_sha256 and the count at
# architecture.storage.child_parameter_count -- read from the CELL receipt, which
# is the authority, and not from a build receipt that may have been dropped with
# the checkpoint it lived inside.
$PY - "$W/children/MX27-$TAG/mx27-build-receipt.json" "$EXPECT" <<'EOF'
import json, sys
got = json.load(open(sys.argv[1]))
cell = json.load(open(sys.argv[2]))
arch = cell.get("architecture", {})
g, w = got.get("child_tensors_sha256"), arch.get("child_tensors_sha256")
gp = got.get("child_parameters")
wp = (arch.get("storage") or {}).get("child_parameter_count")
print(f"  rebuilt tensors sha256 {g}")
print(f"  sealed  tensors sha256 {w}")
print(f"  parameters rebuilt {gp} sealed {wp}")
if g and w and g == w and gp == wp:
    print("  BIT-IDENTICAL to the sealed body")
else:
    print("  *** DOES NOT REPRODUCE -- do not cite the sealed cell for this body ***")
    raise SystemExit(9)
EOF
echo "BODYQ $GPU DONE"
