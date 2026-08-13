#!/bin/bash
# Wave 27: the terminal-read re-solve, carried onto this wave's stack rungs.
#
# Every child here has the SAME parameter count, the same tensor shapes and the
# same config as the sealed child it is built from -- 1,024 values (plus 30,464
# on the 29-diagonal arm) are overwritten in place and nothing is added.  So the
# comparison against its source is controlled to the byte on everything except
# the diagonal, which is the only way a case delta can be attributed to a solve.
#
# What the lambda atlas looks like after wave 25: 0.3 is the PEAK on both bases
# that have a full ladder (V16X13 reads 5/6/7/6/5 and V8X13 reads 6/6/7/7/4 going
# from no solve through 1.0, 0.3, 0.1, 0.01), and the ladder is FLAT on the third
# base (V32X13 reads 10/10/-/9/10) with 0.3 never tried there.  So wave 27 spends
# its solves at 0.3 and nowhere else: once on this wave's V20X13, where the peak
# either transfers to a fourth base or does not, and once on V32X13, which fills
# the one hole in the only flat ladder the campaign has.
#
#   gainq.sh <gpu> <TAG>:<source child dir>[:<lambda>][:29] ...
set -u
. /home/cereal/nf-mx27/env.sh
GPU="${1:?gpu}"; shift
mkdir -p $W/children $W/logs
export PYTHONPATH="$MX27:$NF_DEPLOY/src:$NF_DEPLOY/experiments"

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
  rm -rf $W/children/MX27-$TAG
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX27/mx27_termgain.py \
    --source "$SRC" --source-cell-receipt "$CELL" \
    --child $W/children/MX27-$TAG --tag $TAG --device cuda:0 $EXTRA \
    > $W/logs/gainbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'final gain:' $W/logs/gainbuild-$TAG.log | tail -1)"
  [ -f $W/children/MX27-$TAG/mx27-build-receipt.json ] || tail -12 $W/logs/gainbuild-$TAG.log
done
echo "GAINQ $GPU DONE"
