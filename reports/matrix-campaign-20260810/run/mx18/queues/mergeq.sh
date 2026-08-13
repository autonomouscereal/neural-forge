#!/bin/bash
# Step 3: does the merge compose with an excision under anything like a distance
# lens?  `ARCH.KV6JQ` touches all six attention blocks -- 2, 4, 6, 8, 10, 12 --
# so "distance" from a merge to an excised block is a distance to the nearest
# attention site, and the three children below are 1, 1 and 0 blocks away from
# the mass:
#
#   X07 + KV6JQ   block 7 is mid-interior, one block from L06 and L08, and its
#                 11/15 is the best per-child ALIVE reading in the campaign
#   X09 + KV6JQ   block 9, one from L08 and L10, 10/15 alone
#   T010713+KV6JQ the wave's widest-spread excision set, which straddles the
#                 attention mass rather than sitting beside one site
#
# Wave 17 has 13 + KV6JQ at 4/15 (13 is TWELVE blocks from the first attention
# site and one past the last) and the best pair + the merge at 0/15, so the
# question this asks is whether the 4/15 was about the distance or about there
# being two things at all.
set -u
. /home/cereal/nf-mx18/env.sh
GPU="${1:?gpu}"
mkdir -p $W/children $W/logs
run () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX18/mx18_build.py \
    --mode build --arm conditioned-local --child $W/children/MX18-$TAG --device cuda:0 "$@" \
    > $W/logs/build-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/build-$TAG.log)"
}
run X07KV6JQ --excise 7 --kv-groups 6 --kv-solve joint --kv-als 0
run X09KV6JQ --excise 9 --kv-groups 6 --kv-solve joint --kv-als 0
run T010713KV6JQ --excise 1 --excise 7 --excise 13 --kv-groups 6 --kv-solve joint --kv-als 0
echo MERGEQ DONE
