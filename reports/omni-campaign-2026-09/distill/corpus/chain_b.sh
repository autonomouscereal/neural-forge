#!/bin/bash
set -e
for s in measure vision media; do ./run_step.sh $s --workers 8 > logs/$s.log 2>&1; done
echo CHAIN_B_DONE > logs/chain_b.done
