#!/bin/bash
# Wave 29: the terminal-read re-solve, carried onto this wave's stack rungs.
#
# Every child here has the SAME parameter count, the same tensor shapes and the
# same config as the sealed child it is built from -- 1,024 values (plus 30,464
# on the 29-diagonal arm) are overwritten in place and nothing is added.  So the
# comparison against its source is controlled to the byte on everything except
# the diagonal, which is the only way a case delta can be attributed to a solve.
#
# WHAT THE ATLAS LOOKS LIKE AFTER WAVE 27, and why this wave spends solves
# differently.  Wave 26 made 0.3 "the rung that never hurts" off four ladders;
# wave 27 built six more arms and 0.3 gained on NONE of them and lost on four,
# twice while IMPROVING its own held-out objective by 23-26%.  So lambda 0.3 is
# demoted from a default to one rung of a ladder, and this wave runs the LADDER
# (1.0 / 0.3 / 0.1) on the summit base rather than a single arm at a rung the
# solver likes.  The point of a ladder is that the objective does not get to
# choose; the review does.
#
# The CL896-family arms are a gate test as much as a solve.  Wave 27 recorded
# that the terminal-gain identity gate REFUSES a width fold, because the
# objective needs the child's token table to BE the parent's rows at the injected
# ids and a width fold projects the embedding.  A vocabulary fold keeps its rows
# VERBATIM (`identity_gate.kind == "bytes"`), so a V*CL896 child may or may not
# pass depending on whether the gate reads the vocabulary fold or the width fold
# underneath it.  Nobody has tried.  If it refuses, the refusal is RECORDED and
# not routed around -- wave 27's rule, kept.
#
#   gainq.sh <gpu> <TAG>:<source child dir>[:<lambda>][:29] ...
set -u
. /home/cereal/nf-mx29/env.sh
GPU="${1:?gpu}"; shift
mkdir -p $W/children $W/logs
export PYTHONPATH="$MX29:$NF_DEPLOY/src:$NF_DEPLOY/experiments"

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi

for spec in "$@"; do
  TAG=$(echo "$spec" | cut -d: -f1)
  SRC=$(echo "$spec" | cut -d: -f2)
  LAM=$(echo "$spec" | cut -d: -f3)
  ARM=$(echo "$spec" | cut -d: -f4)
  EXTRA=""
  [ -n "${LAM:-}" ] && EXTRA="$EXTRA --lambda-relative $LAM"
  [ "${ARM:-}" = "29" ] && EXTRA="$EXTRA --block-gains"
  CELL=""
  case "$SRC" in
    */children/MX*) CELL=$(dirname $(dirname $SRC))/cells/ARCH.$(basename $SRC | sed 's/^MX[0-9]*-//')/cell-receipt.json ;;
  esac
  rm -rf $W/children/MX29-$TAG
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX29/mx29_termgain.py \
    --source "$SRC" --source-cell-receipt "$CELL" \
    --child $W/children/MX29-$TAG --tag $TAG --device cuda:0 $EXTRA \
    > $W/logs/gainbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'final gain:' $W/logs/gainbuild-$TAG.log | tail -1)"
  [ -f $W/children/MX29-$TAG/mx29-build-receipt.json ] || tail -12 $W/logs/gainbuild-$TAG.log
done
echo "GAINQ $GPU DONE"
