#!/bin/bash
# Wave 43: the PLANE axis -- the census FIRST, then at most one build.
#
# THE ORDER IS THE POINT.  `docs/ROUND3_SLATE_2026-08-28.md` section 3.4 requires
# the plane-energy census to run and its retention numbers to be PRE-REGISTERED
# before any child exists, so that a death attributes to the mechanism and not to
# the rung.  A census that runs inside a build has already chosen its band, so
# this driver has two verbs and the build verb refuses without a census artifact
# on disk to have been registered from.
#
#   planeq.sh census <gpu> <out json>
#   planeq.sh build  <gpu> <TAG> <head_dim> <stride> <slide> <expected parameters>
#
# `expected parameters` is not a convenience: it is the wave's pre-registered
# arithmetic, and `mx44_plane.py` recomputes the same number from the parent's
# own config and REFUSES if the two disagree.  A build whose parameter count is
# whatever the build happened to produce is a build with no control on it.
#
# The GPU is pinned by PCI bus id and re-checked in process, `NOTES-MX-E` §2.
set -u
. /home/cereal/nf-mx44/env.sh
mkdir -p $W/children $W/logs $W/prereads

VERB="${1:?census|build}"; GPU="${2:?gpu}"; shift 2

CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))' 2>/dev/null)
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP'" >&2; exit 3
fi

case "$VERB" in
census)
  OUT="${1:?out json}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX44/mx44_plane.py \
    --mode census --device cuda:0 --out "$OUT" \
    > $W/logs/plane-census.log 2>&1
  rc=$?
  echo "plane census rc=$rc"
  grep -E "^\[mx44plane\]" $W/logs/plane-census.log | tail -30
  # Parse what was written, in the invocation that wrote it.  NOTES-MX-AO §9.
  $PY -c "import json,sys; a=json.load(open(sys.argv[1])); \
print('CENSUS OK', a['schema'], len(a['bands']), 'bands;', a['gates'])" "$OUT"
  exit $rc
  ;;
build)
  TAG="${1:?tag}"; D="${2:?head_dim}"; M="${3:?stride}"; S="${4:?slide}"; P="${5:?expected parameters}"
  if [ ! -f "$W/prereads/plane-energy-census-wave44.json" ]; then
    echo "BUILD REFUSED: no census artifact at $W/prereads/plane-energy-census-wave44.json." >&2
    echo "The slate requires the retention numbers registered BEFORE the build." >&2
    exit 4
  fi
  rm -rf $W/children/MX44-$TAG
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX44/mx44_plane.py \
    --mode build --device cuda:0 \
    --head-dim "$D" --stride "$M" --slide "$S" \
    --tag "$TAG" --child $W/children/MX44-$TAG \
    --expect-parameters "$P" \
    > $W/logs/plane-build-$TAG.log 2>&1
  rc=$?
  echo "$TAG rc=$rc"
  grep -E "^\[mx44plane\]" $W/logs/plane-build-$TAG.log | tail -20
  [ -f $W/children/MX44-$TAG/mx44-build-receipt.json ] || tail -20 $W/logs/plane-build-$TAG.log
  exit $rc
  ;;
*)
  echo "unknown verb '$VERB'" >&2; exit 2
  ;;
esac
