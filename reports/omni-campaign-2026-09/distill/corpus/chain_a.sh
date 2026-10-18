#!/bin/bash
set -e
for s in allocate text long k3 selfgen; do ./run_step.sh $s > logs/$s.log 2>&1; done
echo CHAIN_A_DONE > logs/chain_a.done
