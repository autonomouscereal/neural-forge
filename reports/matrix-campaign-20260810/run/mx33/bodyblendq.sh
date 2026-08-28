#!/bin/bash
# Wave 32: BODY-BLEND.  One CPU pass per blend, no GPU, no solve.
#
# The spec is deliberately verbose rather than positional: wave 29's blend queue
# packed source, other and mode into one colon-delimited field, and a three-way
# simplex point does not fit in it without inventing a nesting rule nobody would
# read correctly at 2am.
#
#   bodyblendq.sh <TAG> <align> <source dir> <other dir>=<w> [<other dir>=<w> ...]
#
# `align` is `channel` or `none`; see `mx33_blend_body.py` for why the naive arm
# exists and why it is not the default.
set -u
. /home/cereal/nf-mx33/env.sh
mkdir -p $W/children $W/logs $W/blend

TAG="${1:?tag}"; ALIGN="${2:?align}"; SRC="${3:?source}"; shift 3
[ -d "$SRC" ] || { echo "$TAG REFUSED: no source at $SRC"; exit 2; }

ARGS=""
for spec in "$@"; do
  DIR="${spec%=*}"; WT="${spec##*=}"
  [ -d "$DIR" ] || { echo "$TAG REFUSED: no sibling at $DIR"; exit 2; }
  ARGS="$ARGS --other $DIR --weight $WT"
done

rm -rf $W/children/MX33-$TAG
$PY $MX33/mx33_blend_body.py \
  --source "$SRC" $ARGS \
  --parent $PARENT --align "$ALIGN" \
  --child $W/children/MX33-$TAG --tag $TAG \
  --out-census $W/blend/alignment-$TAG.json \
  > $W/logs/blend-$TAG.log 2>&1
rc=$?
echo "$TAG rc=$rc :: $(grep -h '^blend ' $W/logs/blend-$TAG.log | tail -1)"
[ $rc -eq 0 ] || tail -8 $W/logs/blend-$TAG.log
echo "BODYBLENDQ DONE"
