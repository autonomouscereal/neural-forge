#!/bin/bash
# Seal one basis artifact per (mode, width) this wave builds against.  A basis is
# written ONCE and every child at that width is built against the same bytes and
# quotes the same digest; wave 19's ladder is only a ladder because of that, and
# a wave that compares objectives needs it twice over.
#
#   basisq.sh <gpu> <mode>:<width>[:<beta>] ...
set -u
. /home/cereal/nf-mx20/env.sh
GPU="${1:?gpu}"; shift
mkdir -p $W/basis $W/logs
for spec in "$@"; do
  MODE=${spec%%:*}; REST=${spec#*:}
  WIDTH=${REST%%:*}
  BETA=${REST#*:}; [ "$BETA" = "$WIDTH" ] && BETA=0.5
  case $MODE in
    attribution-weighted) NAME=E-$MODE-b$(echo $BETA | tr -d '.')-$WIDTH ;;
    *)                    NAME=E-$MODE-$WIDTH ;;
  esac
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX20/mx20_build.py \
    --mode basis --device cuda:0 --hidden-size $WIDTH --basis $MODE \
    --basis-beta $BETA --selector-census \
    --basis-file $W/basis/$NAME.safetensors \
    > $W/logs/basis-$NAME.log 2>&1
  echo "$NAME rc=$? $(grep -h '^\[mx20\] basis ->' $W/logs/basis-$NAME.log | tail -1)"
done
echo "BASISQ $GPU DONE"
