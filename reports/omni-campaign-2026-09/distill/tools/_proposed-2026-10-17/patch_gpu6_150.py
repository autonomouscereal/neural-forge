#!/usr/bin/env python3
"""Make queue_gpu6.sh 1.5.0 from the installed 1.4.0: provisional K3 passes under ruling k3-b2-bf16-numerics-provisional-2026-10-17.

Only three edits; everything else is byte-identical.  Writes $Q/gpu6.sh.1.5.0.new; the caller installs it by rename.
"""
import hashlib
import sys

Q = "/home/cereal/omni-campaign/distill/queues"
src_path = f"{Q}/gpu6.sh"
src = open(src_path, encoding="utf-8").read()
print("source sha256", hashlib.sha256(src.encode()).hexdigest())

old_version = 'VERSION="queue_gpu6.sh 1.4.0"'
new_version = (
    'VERSION="queue_gpu6.sh 1.5.0"   # 1.5.0 2026-10-17 (Claude, ruling $Q/k3_b2_numerics_ruling.txt): B2 failed ONLY on the\n'
    '                                #   bf16-vs-fp32 parity of one frozen tool while the functional fold-identity gate PASSED\n'
    '                                #   (93/93 layers bit-exact + production closed loop): the K3 passes run PROVISIONALLY into\n'
    '                                #   $K3OUT-provisional-bf16 with their own state base and $Q/k3_pass_tools.provisional.json,\n'
    '                                #   so d2_plan neither accepts nor refuses them until the lead ratifies or revokes the ruling.\n'
    '                                #   No other logic changed (1.4.0 kept as gpu6.sh.1.4.0-live-<utc>).'
)
assert src.count(old_version) == 1, "VERSION line not found exactly once"
src = src.replace(old_version, new_version)

anchor = 'echo "B2 verdict for frozen tool ${K3SHA:0:16} from $B2/parity_b2_receipt.json + $B2_MAN (rebase $REBASE) -> B2_OK=$B2_OK"\n'
assert src.count(anchor) == 1, "B2 verdict echo not found exactly once"
block = anchor + r'''# 1.5.0: provisional K3 passes under the numerics ruling (see VERSION).  Every functional part of b2_ok must hold; only the
# parity verdict may be a FAIL, and then only with a complete, finite receipt (0 non-finite positions).
K3_NUMERICS_RULING=$Q/k3_b2_numerics_ruling.txt
B2_PROVISIONAL=0
if [ "$B2_OK" != 1 ] && [ "$REBASE" = 1 ] && [ -s "$K3_NUMERICS_RULING" ] && [ -s "$B2/parity_b2_receipt.json" ] \
   && manifest_sealed_by "$B2_MAN" "$K3SHA" && manifest_sealed_by "$B2F_MAN" "$K3SHA" \
   && identity_ok "$ID_RCPT" "$K3SHA" \
   && python3 -c "import json,sys; r=json.load(open(sys.argv[1])); g=r.get('gates') or {}; p=r.get('pooled') or {}; sys.exit(0 if g.get('complete') is True and g.get('finite') is True and p.get('non_finite') == 0 else 1)" "$B2/parity_b2_receipt.json"
then
  B2_OK=1; B2_PROVISIONAL=1
  K3_STATE_BASE=$LEXAR/scratch/k3cache-state-provisional-bf16
  export K3_PASS_TOOLS=$Q/k3_pass_tools.provisional.json
  status b2 PROVISIONAL "ruling $(head -1 "$K3_NUMERICS_RULING"): identity PASS + complete/finite parity; bf16-vs-fp32 parity FAIL recorded as a diagnostic; K3 passes go to the -provisional-bf16 dir + $K3_PASS_TOOLS (d2_plan does not read either)"
fi
echo "B2 provisional: $B2_PROVISIONAL (state base $K3_STATE_BASE, pass records ${K3_PASS_TOOLS:-$Q/k3_pass_tools.json})"
'''
src = src.replace(anchor, block)

old_out = "K3OUT=$CACHE_ROOT/k3/$K3SHA16\n"
assert src.count(old_out) == 1, "K3OUT line not found exactly once"
src = src.replace(old_out, old_out + '[ "$B2_PROVISIONAL" = 1 ] && K3OUT=$K3OUT-provisional-bf16   # 1.5.0\n')

out = f"{Q}/gpu6.sh.1.5.0.new"
with open(out, "w", encoding="utf-8") as fh:
    fh.write(src)
print("wrote", out, "sha256", hashlib.sha256(src.encode()).hexdigest())
sys.exit(0)
