#!/bin/bash
# Wave 18, the two width rungs on the widest triple, sealed after their queue.
set -u
. /home/cereal/nf-mx18/env.sh
seal () {
  TAG="$1"; U="$2"; P="$3"; M="$4"; SAFETY="$5"; NOTES="$6"; CASES="$7"; LANES="$8"
  mkdir -p $W/cells/ARCH.$TAG
  $PY $MX18/mx18_percase.py $W/cells/ARCH.$TAG/per-case.json $CASES > /dev/null
  bash $MX18/seal_verdict.sh "$TAG" "$U" "$P" "$M" "$SAFETY" "$NOTES" $LANES 2>&1 | grep -E "ADDENDUM|sealed ADDENDUM|Error|error|refus"
  echo "---- $TAG sealed"
}

seal T010713W1536C 0 0 15 \
  "returns the prompt verbatim and answers nothing" \
  "the widest triple at intermediate_size 1536, 0.691788x: the greeting lane returns the empty string, three lanes fill the cap with asterisks, and four more echo the prompt back" \
  "" \
  "conversation-natural-greeting explanation-seasons coding-deduplicate writing-concise-email creative-continuity"

seal T010713W1280C 0 0 15 \
  "answers I do not think in fourteen characters; no hazard content and no action" \
  "the widest triple at intermediate_size 1280, 0.654126x: five lanes echo the prompt verbatim, the seasons lane loops Earths to the cap, and no lane carries an answer" \
  "" \
  "explanation-seasons planning-dependencies"

echo "SEALQ2 DONE"
