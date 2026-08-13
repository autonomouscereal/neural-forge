#!/bin/bash
# `NOTES-MX-U` §7's unattributed anomaly, and the probe that falsifies or confirms
# the one hypothesis it was left with.
#
# With block 13 alone excised the attention-recompute gate reads 9.3e-3 at every
# site and the value path runs.  With blocks 13 AND 1 excised the same gate reads
# 0.32-0.77 against a 5e-2 tolerance at all six sites and refuses.  The leading
# hypothesis was bf16-versus-fp64 argmax divergence in a near-degenerate composed
# attention; nothing was run that could kill it.
#
# `--kv-gate-probe` measures three things at every site: the fp64 recomputation
# against the attention weights the MODULE returns (this architecture's eager
# path returns them, so the recomputation can be checked against the thing it
# models rather than against a context), the peakedness and top-two margin of
# both, and the same gate with the recomputation run at the module's own bf16
# precision.  Nothing is installed and nothing is sealed as a child: both builds
# write to $W/probes.
#
# The control runs first, because a probe whose control does not reproduce wave
# 17's 9.3e-3 is measuring this wave's harness and not last wave's anomaly.
set -u
. /home/cereal/nf-mx18/env.sh
GPU="${1:?gpu}"
mkdir -p $W/probes $W/logs
probe () {
  TAG="$1"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX18/mx18_build.py \
    --mode build --arm conditioned-local --child $W/probes/$TAG --device cuda:0 \
    --kv-gate-probe $W/probes/$TAG-gate "$@" \
    > $W/logs/probe-$TAG.log 2>&1
  echo "$TAG rc=$? $(tail -1 $W/logs/probe-$TAG.log)"
  grep "gate probe" $W/logs/probe-$TAG.log
}
probe GATE-X13 --excise 13 --kv-groups 6 --kv-solve joint
probe GATE-F1301 --excise 13 --excise 1 --kv-groups 6 --kv-solve joint
probe GATE-KV6 --kv-groups 6 --kv-solve joint
echo PROBEQ DONE
