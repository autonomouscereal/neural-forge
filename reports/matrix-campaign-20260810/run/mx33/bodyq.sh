#!/bin/bash
# Wave 29: build a BODY that a vocabulary fold can then be composed onto -- one
# that no wave has sealed, or one that was sealed and not retained.
#
# Wave 27 wrote this file to rebuild wave 17's `ARCH.F1301KV6Q` and CHECK the
# rebuild against the sealed digest rather than trust it.  It refused: identical
# parameter count, different digest, byte-identical on a second rebuild.  Wave 29
# attributes that refusal (see `driftq.sh`) and keeps the check exactly as it is,
# with one addition -- a KNOWN-DRIFT registry, so a future wave gets the reason
# with the refusal instead of re-deriving it.
#
# The rule the check enforces is unchanged: a body that does not reproduce is not
# the sealed body, and a composed child built on it may not cite the sealed cell.
#
#   bodyq.sh <gpu> <TAG>:<sealed cell receipt to check against|NEW> -- <build args...>
#
# `NEW` is a body no wave has sealed.  There is nothing to check it against, so
# the queue says so in those words rather than printing a reassuring line.
set -u
. /home/cereal/nf-mx33/env.sh
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

rm -rf $W/children/MX33-$TAG
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX33/mx33_build.py \
  --mode build --arm conditioned-local \
  --child $W/children/MX33-$TAG --device cuda:0 "$@" \
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
if [ "$EXPECT" = "NEW" ]; then
  echo "  NO SEALED CELL: this body has never been sealed, so there is nothing to"
  echo "  check the rebuild against.  It is a new cell, not a reproduction."
  echo "  rebuilt tensors sha256 $($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['child_tensors_sha256'])" $W/children/MX33-$TAG/mx33-build-receipt.json)"
  echo "BODYQ $GPU DONE"
  exit 0
fi

$PY - "$W/children/MX33-$TAG/mx33-build-receipt.json" "$EXPECT" <<'EOF'
import json, sys

# KNOWN DRIFT.  Wave 27 refused here and could not say why; wave 29 bisected it
# across the per-wave builders and the answer is in `mx18_build.py` around the
# `if excised:` branch -- wave 18 re-derives `harness.model.config.layer_types`
# after an excision so the LIVE model's causal mask follows the live blocks.
# Wave 17's builder did not, so on any body that excises blocks which SHIFT a
# later attention block into a conv's slot, wave 17's conditioned capture put
# 28.6-48.0% of the module's attention mass on tokens that had not happened yet
# (measured by `mx18_mask_probe.py`), and every solve that consumed that capture
# consumed different evidence.  The drift is a REPAIR, so the sealed digest is
# the defective one and the rebuild is the correct body; the cell may still not
# be cited, because the bytes are not the bytes that were reviewed.
KNOWN_DRIFT = {
    ("90a9e5187feef6780a0503dc0757b97b6b18db0bc441fd5a51f2d8811f170228",
     "ad406f480a181901db35b7c1af6ede976475ce9eece7b3ec36f0878fbbb76f09"): (
        "wave 17 ARCH.F1301KV6Q: wave 18's live-mask repair (mx18_build.py, the "
        "`if excised:` branch) changed the evidence every conditioned solve reads "
        "on a body whose excision shifts an attention block into a conv slot.  "
        "Bisected in wave 29; the sealed body is the defective one"
    ),
}

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
    reason = KNOWN_DRIFT.get((w, g))
    if reason:
        print(f"  KNOWN DRIFT, attributed: {reason}")
    else:
        print("  UNATTRIBUTED: no wave has bisected this pair.  Run driftq.sh.")
    raise SystemExit(9)
EOF
echo "BODYQ $GPU DONE"
