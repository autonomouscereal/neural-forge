#!/bin/bash
# Wave 40: the ROW-TYING arm, C1's repair.  One CPU pass per child, no capture,
# no forward call, no GPU -- the same cost as the deletion rung it is read
# against, on the same body, at the same parameter count.
#
# WHY THE SPEC IS ONE FIELD LONGER THAN vocabq.sh's.  The tie arm exists only as
# a CONTROLLED comparison against a deletion twin, so the twin's parameter count
# is stated on the command line and the builder REFUSES if this child does not
# land on it to the byte.  A tie arm that misses its twin's count is not a
# comparison, it is a second reading of a different thing, and the place to catch
# that is before the GPU is asked for a review.
#
#   tieq.sh <TAG>:<K>:<tie rows>:<expected parameters>:<source dir>[:<source cell receipt>] ...
#
# BELOW_FLOOR=1 releases the coverage half of wave 25's census gate for every
# child in the invocation, exactly as it does in vocabq.sh and for exactly the
# same reason: the tie child's tokenizer IS the deletion twin's, so it loses
# exactly the occurrences the twin loses and it must take the same decision
# visibly on the command line that launched it.
set -u
. /home/cereal/nf-mx40/env.sh
mkdir -p $W/children $W/logs
FLOOR=""
[ "${BELOW_FLOOR:-0}" = "1" ] && FLOOR="--below-floor"
export PYTHONPATH="$MX40:${PYTHONPATH:-}"

for spec in "$@"; do
  TAG=$(echo "$spec" | cut -d: -f1)
  K=$(echo "$spec" | cut -d: -f2)
  TIE=$(echo "$spec" | cut -d: -f3)
  EXPECT=$(echo "$spec" | cut -d: -f4)
  SRC=$(echo "$spec" | cut -d: -f5)
  CELL=$(echo "$spec" | cut -d: -f6)
  if [ ! -d "$SRC" ]; then echo "$TAG SKIPPED: no source at $SRC"; continue; fi
  rm -rf $W/children/MX40-$TAG
  $PY $MX40/mx40_vocabtie.py --mode build --k $K --tie-rows $TIE --tag $TAG $FLOOR \
    --expect-parameters $EXPECT \
    --source "$SRC" --source-cell-receipt "${CELL:-}" \
    --child $W/children/MX40-$TAG \
    > $W/logs/tiebuild-$TAG.log 2>&1
  echo "$TAG rc=$? :: $(grep -h 'parameters,' $W/logs/tiebuild-$TAG.log | tail -1)"
  [ -f $W/children/MX40-$TAG/mx40-build-receipt.json ] || tail -12 $W/logs/tiebuild-$TAG.log
done

# THE CROSS-PROCESS DECODE CHECK, and it is here rather than in the builder
# because the builder cannot run in two processes.  The many-to-one form of this
# class named a DIFFERENT out-of-vocabulary token on every one of five loads of
# the identical file, which is a nondeterministic id->token inverse; the
# injective form is asserted in-process by the builder across two independently
# constructed tokenizers, and asserted again HERE across two independently
# launched interpreters.  A child whose decode depends on the process cannot be
# frozen-reviewed, and finding that out at review time costs a GPU hour.
for spec in "$@"; do
  TAG=$(echo "$spec" | cut -d: -f1)
  D=$W/children/MX40-$TAG
  [ -f $D/tokenizer.json ] || continue
  A=$($PY - "$D" <<'EOF'
import hashlib, sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1])
n = t.vocab_size
h = hashlib.sha256("\x00".join(t.convert_ids_to_tokens(list(range(n)))).encode("utf-8"))
print(h.hexdigest())
EOF
)
  B=$($PY - "$D" <<'EOF'
import hashlib, sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1])
n = t.vocab_size
h = hashlib.sha256("\x00".join(t.convert_ids_to_tokens(list(range(n)))).encode("utf-8"))
print(h.hexdigest())
EOF
)
  if [ "$A" = "$B" ]; then
    echo "$TAG id-to-token map is a function across processes: $A"
  else
    echo "$TAG DECODE IS NOT A FUNCTION ACROSS PROCESSES: $A vs $B" >&2
    exit 7
  fi
done
echo "TIEQ DONE"
