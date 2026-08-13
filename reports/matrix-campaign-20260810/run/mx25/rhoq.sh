#!/bin/bash
# P1, the cross-error census (S3, ORTHO-PAIR's free pre-read).  It builds
# nothing and gates nothing; it prices L3's cross term on every sealed two-body
# composition this campaign has a complete triple for, including -- for the
# first time -- one whose two folds do not share a tokenizer.
#
#   rhoq.sh <gpu>
set -u
. /home/cereal/nf-mx25/env.sh
GPU="${1:?gpu}"
M=/mnt/ai_data/neural-forge-matrix
mkdir -p $W/prereads $W/logs
export PYTHONPATH="$MX25:$NF_DEPLOY/src:$NF_DEPLOY/experiments"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $MX25/mx25_rho.py \
  --device cuda:0 --cuts 8 \
  --triple "V16X13:$M/wave16/children/MX16-X13A,$M/wave23/children/MX23-V16,$M/wave23/children/MX23-V16X13" \
  --triple "V8X13:$M/wave16/children/MX16-X13A,$W/children/MX25-V8,$W/children/MX25-V8X13" \
  --triple "V16W896:$M/wave19/children/MX19-W896,$M/wave23/children/MX23-V16,$M/wave23/children/MX23-V16W896" \
  --triple "W896X13:$M/wave19/children/MX19-W896,$M/wave16/children/MX16-X13A,$M/wave19/children/MX19-W896X13" \
  --triple "X07KV6JQ:$M/wave16/children/MX16-X07,$M/wave17/children/MX17-KV6JQ,$M/wave18/children/MX18-X07KV6JQ" \
  --triple "W896LSX13:$M/wave21/children/MX21-W896LS,$M/wave16/children/MX16-X13A,$M/wave21/children/MX21-W896LSX13" \
  --out $W/prereads/P1-cross-error.json \
  > $W/logs/P1.log 2>&1
echo "P1 rc=$?"
grep -h "rho" $W/logs/P1.log | tail -20
