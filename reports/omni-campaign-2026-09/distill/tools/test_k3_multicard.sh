#!/bin/bash
# test_k3_multicard.sh -- k3_multicard.sh's refusals, gate readers and resolved plan (no gate is waited on, no lock is
# taken, nothing is launched; reads GPU0's name and temperature thresholds only).  Usage: bash test_k3_multicard.sh
# [k3_multicard.sh, default beside this file]; K3LIB=<path> tests a staged k3_lib.sh.
#   refuse  GPU5 (never), GPU6 (queue_gpu6's), GPU2 (P40: MLA at T = 4,096 does not fit), no argument: exit 2
#   gates   the go file must name this gpu exactly (gpus=10 is not gpu 0) under line 1 k3-multicard-go; the ruling id
#   plan    GPU0 resolves to the 3090 class, the B2-proven 1.2.4 without a 1.3.0 review, 32,768 group tokens, and
#           exits before any gate
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MC=${1:-$HERE/k3_multicard.sh}
pass=0; fail=0
want_line() {  # $1 name $2 output $3 exact line
  if echo "$2" | grep -qxF -- "$3"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "MISSING ($1): $3"; fi
}
want_re() {  # $1 name $2 output $3 regex
  if echo "$2" | grep -qE -- "$3"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "MISSING ($1): /$3/"; fi
}
for g in 5 6 2 ""; do
  OUT=$(timeout 60 bash "$MC" $g 2>&1); rc=$?
  echo "$OUT" | head -2
  if [ $rc -eq 2 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL refuse gpu '$g': rc $rc"; fi
done
OUT=$(K3MC_SELFTEST=gates timeout 60 bash "$MC" 0 2>&1); echo "$OUT"
for l in "ST go listed: 1" "ST go not-listed: 0" "ST go bad-line1: 0" "ST go substring: 0" "ST go absent: 0" \
         "ST ruling ok: 1" "ST ruling other: 0"; do want_line gates "$OUT" "$l"; done
OUT=$(K3MC_SELFTEST=plan timeout 120 bash "$MC" 0 2>&1); echo "$OUT"
want_re plan "$OUT" "^k3_multicard\.sh 1\.0\.0 gpu 0 \(nvidia-geforce-rtx-3090-sm86\) .*: tool v124 k3_teacher_cache 1\.2\.4 2aae4dc3134dda72, group tokens 32768, B2 dir "
want_re plan "$OUT" "^thermal: stop [0-9]+ C x3, hard [0-9]+ C"
want_re plan "$OUT" "state /mnt/lexar/omni-campaign/scratch/k3cache-state/tool-2aae4dc3134dda72-gpu0;"
want_line plan "$OUT" "SELFTEST plan: exit before any gate"
OUT=$(K3MC_SELFTEST=bogus timeout 60 bash "$MC" 0 2>&1); rc=$?
want_line bogus "$OUT" "refuse: unknown K3MC_SELFTEST=bogus"
echo "K3 MULTICARD TEST ($(sed -n 's/^VERSION="\(.*\)"$/\1/p' "$MC")): $pass passed, $fail failed"
[ $fail -eq 0 ]
