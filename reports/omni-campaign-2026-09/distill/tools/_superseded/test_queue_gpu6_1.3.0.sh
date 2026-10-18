#!/bin/bash
# test_queue_gpu6.sh -- the queue's thermal watchdog, adoption and B2-verdict paths on dummy processes and synthetic
# receipts (no GPU work, no status lines; reads GPU6's temperature only). Usage: bash test_queue_gpu6.sh [queue script,
# default $Q/gpu6.sh]
#   hot        STOP_C=1   held 3 polls -> SIGTERM (rc 143), marker written
#   hard       HARD_C=1   one poll     -> SIGTERM (rc 143), marker written
#   cool       no stop line reached    -> rc 0, no marker
#   fail       a job's own exit 7 is passed through, no marker
#   adopt-hot  a NON-child pid is stopped too; rc comes from its manifest (absent -> 1)
#   adopt-sealed / adopt-other-tool    rc 0 only for a manifest sealed by the frozen tool's sha
#   b2ok (1.2.0)  the K3 passes' gate: 1 only for a passed receipt + the bf16 manifest sealed by the frozen tool (+ the
#                 fp32 manifest in re-baseline mode); 1.1.0 had no such path, and its inline test could never be 1
#   b2ok (1.3.0)  re-baseline mode also needs the fold-identity PASS receipt by the frozen tool over all 93 layers
#   ballast (1.3.0)  ram_ballast holds reserve - RssAnon(job), shrinks as the job grows, releases at its exit, sets the
#                 job's oom_score_adj; below the floor it holds nothing (~5 GiB of host RAM for ~30 s)
QS=${1:-/home/cereal/omni-campaign/distill/queues/gpu6.sh}
OUT=$(GPU6_QUEUE_SELFTEST=thermal timeout 280 bash "$QS" 2>&1)
OUT2=$(GPU6_QUEUE_SELFTEST=b2ok timeout 60 bash "$QS" 2>&1)
OUT3=$(GPU6_QUEUE_SELFTEST=ballast timeout 120 bash "$QS" 2>&1)
echo "$OUT"; echo "$OUT2"; echo "$OUT3"
pass=0; fail=0
for want in "ST hot: rc=143 marker=yes" "ST hard: rc=143 marker=yes" "ST cool: rc=0 marker=no" \
            "ST fail: rc=7 marker=no" "ST adopt-hot: rc=1 marker=yes alive=no" "ST adopt-sealed: rc=0" \
            "ST adopt-other-tool: rc=1"; do
  if echo "$OUT" | grep -qxF "$want"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "MISSING: $want"; fi
done
for want in "ST b2ok pass: 1" "ST b2ok fail: 0" "ST b2ok other-tool: 0" "ST b2ok rebase-no-fp32: 0" \
            "ST b2ok rebase-no-identity: 0" "ST b2ok rebase-identity-fail: 0" "ST b2ok rebase-identity-other-tool: 0" \
            "ST b2ok rebase-identity-incomplete: 0" "ST b2ok rebase-pass: 1" \
            "ST b2ok rebase-identity-pass-parity-fail: 0" "ST b2ok no-receipt: 0"; do
  if echo "$OUT2" | grep -qxF "$want"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "MISSING: $want"; fi
done
for want in "ST ballast: first=3 max=3 last=0 released=1 job_oom_adj=500" "ST ballast-floor: max=0"; do
  if echo "$OUT3" | grep -qxF "$want"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "MISSING: $want"; fi
done
echo "QUEUE SELFTEST: $pass passed, $fail failed"
[ $fail -eq 0 ]
