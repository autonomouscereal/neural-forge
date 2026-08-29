#!/bin/bash
# Wave 43: the XCHANGE GO-GATE CENSUS.  It builds nothing.
#
# WHAT MAKES THIS ONE DIFFERENT FROM EVERY OTHER CENSUS IN THE TREE: its
# pre-registration says it MAY CLOSE THE CLASS.  The three quantities it measures
# are not evidence about how good the substituted operator would be, they are the
# preconditions for that operator being in the class the substitution installs --
# so a site that fails them is a site where no solve can help, and if every
# surviving conv site fails, XCHANGE closes without a build.  That authority is
# stated in `mx45_xchange.py`'s own docstring, in the artifact it writes, and in
# this wave's pre-registration, and it exists in ONE direction only: a pass funds
# a build, it does not seal anything.
#
#   xchangeq.sh <gpu> <out json>
#
# One capture pass over the parent and a page of arithmetic.  No child is written
# and no ledger row is touched.
set -u
. /home/cereal/nf-mx45/env.sh
mkdir -p $W/logs $W/prereads

GPU="${1:?gpu}"; OUT="${2:?out json}"

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX45/mx45_xchange.py \
  --device cuda:0 --out "$OUT" \
  > $W/logs/xchange-gogate.log 2>&1
rc=$?
echo "xchange go-gate rc=$rc"
grep -E "^\[mx45xchange\]" $W/logs/xchange-gogate.log | tail -40
[ $rc -ne 0 ] && tail -20 $W/logs/xchange-gogate.log

# Parse what was written, in the invocation that wrote it.  NOTES-MX-AO §9, and
# it matters more here than anywhere else in the tree: this artifact is the only
# place the consequence of the gate is written down.
$PY -c "import json,sys; a=json.load(open(sys.argv[1])); \
print('GOGATE OK', a['schema']); \
print('  passing sites:', a['passing_sites']); \
print('  consequence :', a['consequence'])" "$OUT"
exit $rc
