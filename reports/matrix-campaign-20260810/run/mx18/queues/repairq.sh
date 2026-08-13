#!/bin/bash
# The blast radius of the mask defect, measured and then rebuilt.
#
# It is much smaller than it first looks, and the reason is in the receipts: a
# PURE excision build runs no solve at all -- one axis, `num_hidden_layers`, no
# per-site write, the conditioned capture taken and never used.  So the live
# stream cannot reach the child, and every pure-excision rebuild came back
# bit-identical: X13C, X00C, X07C, P1301C and T010713C reproduce wave 16's
# X13A `da70dc27...`, wave 17's X00 `90e6e817...`, wave 16's X07 `0ebf6150...`,
# wave 17's P1301 `b90c9609...` and this wave's own triple `c3417c52...` to the
# digit.  **The census, the pair map and the triple map need no re-reading.**
#
# What the defect could reach is a child that excises AND solves against the
# live stream.  In this campaign that is exactly the composed children whose
# excision moves an attention block -- which is every excision except one
# confined to blocks 12 and 13.  Wave 16's frontier `W1536X13` and wave 17's
# `X13KV6J*` excise only block 13 and are clean.  These are not:
#
#   F1301W1536C  wave 17's DEEPEST child, 0.730614x, which read 0/15 and echoed
#                the prompt back in fourteen of sixteen lanes -- solved, all of
#                it, against a model reading the future
#   F1301KV6QC   the best pair plus the merge, 0/15, and the child whose gate
#                refused the value path and told us so
#   X07KV6JQC    the merge on the best single (step 3)
#   X09KV6JQC    the merge on the next-best single (step 3)
#   T010713KV6JQC the merge on the widest triple (step 3)
#   T010713W1536C width on the widest triple (step 5), the deepest attempt yet
#   T010713W1280C the bisection rung below it (step 5)
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
run F1301W1536C --excise 13 --excise 1 --ff 1536
run F1301KV6QC --excise 13 --excise 1 --kv-groups 6 --kv-solve joint --kv-als 0
run X09KV6JQC --excise 9 --kv-groups 6 --kv-solve joint --kv-als 0
run T010713KV6JQC --excise 1 --excise 7 --excise 13 --kv-groups 6 --kv-solve joint --kv-als 0
run T010713W1536C --excise 1 --excise 7 --excise 13 --ff 1536
run T010713W1280C --excise 1 --excise 7 --excise 13 --ff 1280
echo REPAIRQ DONE
