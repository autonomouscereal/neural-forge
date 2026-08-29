#!/bin/bash
# Wave 41: the ROWSET arms.  One CPU pass per child, no capture, no forward
# call, no GPU -- the same cost as the deletion rung each one is read against,
# on the same body, at the same K, at the same parameter count to the byte.
#
# WHY THE SPEC CARRIES THE ARM AND THE EXPECTED COUNT.  A rowset arm exists only
# as a CONTROLLED comparison against a sealed deletion twin, so the twin's
# parameter count is stated on the command line and the builder REFUSES if this
# child does not land on it.  Wave 40's tie-builder rule, carried to the arm that
# needs it for the same reason: the place to catch a count that drifts is before
# the GPU is asked for a review, not after.
#
# AND THE DIVERGENCE PRE-READ RUNS FIRST, ALWAYS.  An arm whose kept set is the
# sealed child's IS the sealed child, and building it spends a frozen review
# re-reading a sealed result.  `--mode divergence` says how many rows each arm
# actually moves at fixed K before any of them is built.  It publishes either
# way and it gates nothing -- the wave decides, with the number in front of it.
#
#   rowsetq.sh --preread <K> <source dir> <out json> <ARM>...
#   rowsetq.sh <TAG>:<ARM>:<K>:<expected parameters>:<source dir>[:<source cell receipt>] ...
#
# BELOW_FLOOR=1 releases the coverage half of wave 25's census gate for every
# child in the invocation, exactly as it does in vocabq.sh and tieq.sh.  A rowset
# arm holds the mandatory set exactly as sealed, so it loses exactly the
# occurrences its deletion twin loses; the flag is on the command line that
# launched the batch because a flag that dangerous should be visible there.
set -u
. /home/cereal/nf-mx41/env.sh
mkdir -p $W/children $W/logs $W/prereads
FLOOR=""
[ "${BELOW_FLOOR:-0}" = "1" ] && FLOOR="--below-floor"
export PYTHONPATH="$MX41:${PYTHONPATH:-}"

if [ "${1:-}" = "--preread" ]; then
  shift
  K="${1:?k}"; SRC="${2:?source}"; OUT="${3:?out}"; shift 3
  ARGS=""
  for arm in "$@"; do ARGS="$ARGS --arm $arm"; done
  $PY $MX41/mx41_vocabrowset.py --mode divergence --k $K --source "$SRC" \
    --out "$OUT" $ARGS 2>&1 | tail -20
  # Parse what was written, in the invocation that wrote it.  NOTES-MX-AO
  # section 9, and the generator asserts it too -- twice, on purpose, because
  # this is the artifact the wave's build list is decided from.
  $PY -c "import json,sys; b=json.load(open(sys.argv[1])); \
print('PREREAD OK', b['schema'], len(b['rungs']), 'rungs')" "$OUT"
  echo "ROWSETQ PREREAD DONE"
  exit 0
fi

for spec in "$@"; do
  TAG=$(echo "$spec" | cut -d: -f1)
  ARM=$(echo "$spec" | cut -d: -f2)
  K=$(echo "$spec" | cut -d: -f3)
  EXPECT=$(echo "$spec" | cut -d: -f4)
  SRC=$(echo "$spec" | cut -d: -f5)
  CELL=$(echo "$spec" | cut -d: -f6)
  if [ ! -d "$SRC" ]; then echo "$TAG SKIPPED: no source at $SRC"; continue; fi
  rm -rf $W/children/MX41-$TAG
  $PY $MX41/mx41_vocabrowset.py --mode build --arm $ARM --k $K --tag $TAG $FLOOR \
    --expect-parameters $EXPECT \
    --source "$SRC" --source-cell-receipt "${CELL:-}" \
    --child $W/children/MX41-$TAG \
    > $W/logs/rowsetbuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'parameters,' $W/logs/rowsetbuild-$TAG.log | tail -1)"
  grep -h 'identity gate\|rows moved\|moved ' $W/logs/rowsetbuild-$TAG.log | tail -2
  [ -f $W/children/MX41-$TAG/mx41-build-receipt.json ] || tail -12 $W/logs/rowsetbuild-$TAG.log
done
echo "ROWSETQ DONE"
