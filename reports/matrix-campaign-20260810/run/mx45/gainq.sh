#!/bin/bash
# Wave 33: the terminal-read re-solve, carried onto the ROBUST deep frontier.
#
# THIS FILE WAS RE-AUTHORED, NOT CARRIED.  Wave 29 was the last wave to cut a
# terminal-gain driver; waves 30, 31 and 32 each carried `mx*_termgain.py`
# forward through the tree cut and lost the queue that drives it, so the solver
# has been in every tree for three waves with nothing able to call it.  This is
# wave 29's `gainq.sh` with the wave identifiers moved and the two paragraphs
# below rewritten against this wave's sources; the loop is unchanged.
#
# Every child here has the SAME parameter count, the same tensor shapes and the
# same config as the sealed child it is built from -- 1,024 values (plus 30,464
# on the 29-diagonal arm) are overwritten in place and nothing is added.  So the
# comparison against its source is controlled to the byte on everything except
# the diagonal, which is the only way a case delta can be attributed to a solve.
#
# WHAT THE ATLAS LOOKS LIKE AFTER WAVE 27, and why this wave spends solves the
# way it does.  Wave 26 made 0.3 "the rung that never hurts" off four ladders;
# wave 27 built six more arms and 0.3 gained on NONE of them and lost on four,
# twice while IMPROVING its own held-out objective by 23-26%.  So lambda 0.3 is
# one rung of a ladder and not a default, and the LADDER (1.0 / 0.3 / 0.1) is
# what runs.  The point of a ladder is that the objective does not get to
# choose; the review does.  lambda -> infinity is NOT a build: the solve returns
# the prior exactly, which is the source child in bytes, so its rung of the
# ladder is the source's own sealed count and it costs no GPU.
#
# THE GATE, and why these two sources are eligible.  Wave 27 recorded that the
# terminal-gain identity gate REFUSES a width fold, because the objective needs
# the child's token table to BE the parent's rows at the injected ids and a
# width fold projects the embedding.  Both sources below are a VOCABULARY gather
# composed onto an `intermediate_size` body: the gather keeps its rows VERBATIM
# and the body never touches the embedding, so the table is the parent's rows at
# the kept ids and the gate should read `identity_gate.kind == "bytes"`.  Should.
# If it refuses, the refusal is RECORDED and not routed around -- wave 27's
# rule, kept.
#
#   gainq.sh <gpu> <TAG>:<source child dir>[:<lambda>][:29] ...
set -u
. /home/cereal/nf-mx45/env.sh
GPU="${1:?gpu}"; shift
mkdir -p $W/children $W/logs
export PYTHONPATH="$MX45:$NF_DEPLOY/src:$NF_DEPLOY/experiments"

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
  rm -rf $W/children/MX45-$TAG
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX45/mx45_termgain.py \
    --source "$SRC" --source-cell-receipt "$CELL" \
    --child $W/children/MX45-$TAG --tag $TAG --device cuda:0 $EXTRA \
    > $W/logs/gainbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'final gain:' $W/logs/gainbuild-$TAG.log | tail -1)"
  [ -f $W/children/MX45-$TAG/mx45-build-receipt.json ] || tail -12 $W/logs/gainbuild-$TAG.log
done
echo "GAINQ $GPU DONE"
