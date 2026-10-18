#!/bin/bash
# S5 union child registration, OC02-F-Q38-27BU-C2K4352K3U (2026-10-12, Claude, S5 build lane).
#
# Prerequisites (each is checked below; nothing is typed from memory):
#   * the WRITTEN child $S/child (c10_union_install.py 1.5.1 --resume) with union_receipt.json;
#   * $S/register_counts.json from s5_register_counts.py (header-derived per-layer counts, slack, identity);
#   * $S/vcheck_full_partial.json + $S/s5_metrics_full.json (the built child as a virtual child, GPU6 bf16)
#     and $S/vcheck_written_child.json (the written child's own forward) for the notes.
#
# Lineage / accounting (oc_checkpoint.py rules, R2b):
#   parent checkpoint  OC02-F-Q38-27BU-C2K4352 (the S3 trunk this union was installed into);
#   parent counts      the HF Qwen3.8-27B-Uncensored headers (27,781,427,952), the same denominator S3 used;
#   components         qwen.mlp.layerNN: parent MLP (header) -> the 4,352 Qwen rows kept verbatim in gate/up
#                      (down columns jointly refit), 0.25 each.  ASSERTED form: the parent (HF names) and the
#                      child (Chimera names, one fused union tensor) share no tensor prefix, so the ledger cannot
#                      derive them; s5_register_counts.py derives every number from both headers instead.
#   added mass         the 64 x 12,288 K3 rows (12,079,595,520) and the native-export difference outside the FFNs
#                      are --passed-through and stated as --accounting-slack (re-derived from the headers);
#                      the mtp drop (424,699,392) carries over from S3.
#
# Usage:  s5_register.sh            # dry run, writes nothing
#         s5_register.sh --confirm  # seals the register record + ledger row
# Fix 2026-10-13 (Claude, S5 finish): the notes read u['receipt_sha256'], which c10 1.5.1 computes AFTER writing
# the receipt (so the file holds no such key and the notes said "None"); they now cite the file's own sha256.
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
export OC_ROOT=$OC
S=$OC/composite/c10/S5-C2K4352K3U
CHILD=$S/child
PARENT=/home/cereal/models/Qwen3.8-27B-Uncensored
TOOL=$OC/ledger/oc_checkpoint.py
COUNTS=$S/register_counts.json
CONFIRM=()
if [ "${1:-}" = "--confirm" ]; then CONFIRM=(--confirm); fi
for f in "$CHILD/union_receipt.json" "$COUNTS" "$S/s5_metrics_full.json"; do
  [ -f "$f" ] || { echo "=== REFUSED: missing $f"; exit 3; }
done
echo "=== register with tool: $TOOL sha256 $(sha256sum "$TOOL" | cut -c1-64)  confirm=${CONFIRM[*]:-no}  ($(date -u +%FT%TZ))"
sha256sum "$0" "$COUNTS" "$CHILD/union_receipt.json" "$S/s5_metrics_full.json" "$S"/vcheck_*.json 2>/dev/null
cd $OC
mapfile -t COMPONENTS < <($PY -c 'import json,sys; [print(l) for l in json.load(open(sys.argv[1]))["component_lines"]]' "$COUNTS")
SLACK=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["accounting_slack_argued"])' "$COUNTS")
K3ADD=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["k3_rows_added"])' "$COUNTS")
NATIVE=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["native_export_extra_parameters"])' "$COUNTS")
[ "${#COMPONENTS[@]}" = 64 ] || { echo "=== REFUSED: ${#COMPONENTS[@]} component lines, expected 64"; exit 3; }
ARGS=(register --checkpoint-id OC02-F-Q38-27BU-C2K4352K3U --stage fold --wave oc02 --tag C2K4352K3U)
ARGS+=(--parent-code Q38-27BU --parent-checkpoint-id OC02-F-Q38-27BU-C2K4352)
ARGS+=(--directory "$CHILD" --parent-directory "$PARENT" --parameter-count auto --parent-parameter-count auto)
for c in "${COMPONENTS[@]}"; do ARGS+=(--component "$c"); done
ARGS+=(--passed-through "k3.ffn:ADDED-not-a-component:${K3ADD}-parameters:64x3x5120x12288-rows-of-K3-dense-L1-92-lifted-through-c17-ridge-charts-into-blocks.k.ffn")
ARGS+=(--passed-through "native-export:${NATIVE}-parameters:S3-native-trunk-non-FFN-tensors-vs-parent-non-MLP-non-mtp")
ARGS+=(--passed-through "mtp:dropped-not-folded:424699392-parameters:inherited-from-S3")
ARGS+=(--passed-through gdn_mixer --passed-through attention_mixer --passed-through interface)
ARGS+=(--accounting-slack "$SLACK")
ARGS+=(--recipe "$CHILD/union_receipt.json")
ARGS+=(--recipe-command "c10_union_install.py 1.5.1 (sha $(sha256sum $OC/composite/tools/c10_union_install.py | cut -c1-16)) via $S/launch_s5.sh: --union-target a003=trunk/additive:0.03 --k3-layers 1-92 --k3-channels 12288 --chart-arm ridge --chart-bias-ceiling 0 --engine streamed --group-size 3 --device/--selection-device/--solve-device cuda:0 (GPU4 RTX 3090), closed loop, run1 stopped by the disk floor before L39, --resume --min-free-gib 56 to L63, then --resume child write")
NOTES=$($PY - "$S" "$COUNTS" <<'EOF'
import json, sys, hashlib
s, counts = sys.argv[1], sys.argv[2]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
c = json.load(open(counts))
m = json.load(open(f"{s}/s5_metrics_full.json"))
u = json.load(open(f"{s}/child/union_receipt.json"))
a = m["models"]["a003"]; t = m["models"]["trunk"]; p = m["models"].get("parent", {})
i = m.get("install", {})
w = {}
try:
    w = json.load(open(f"{s}/s5_metrics_written.json"))["models"].get("written", {})
except Exception:
    pass
print(
 "S5 = the C10 K3 union installed into the S3 trunk (OC02-F-Q38-27BU-C2K4352), closed loop, design of record a003 "
 "(Y = 1.0 FFN_trunk + 0.03 TERM_k3) from composite/c10/design_sweep/VERDICT.md (sha d5e1f60be8c96a9c...; RESULTS.json 04d1ad13bfe4723a...). "
 "Inputs: K3 dense L1-92 = k3-run-oct09-claude/fold/dense, pass receipt k3_dense_pass_receipt.json sha 8948bab7feeaa3cd460418f3552600b2d93a6c944143cfedb1b3a9122fa787df; "
 "c17 ridge charts L00-L63, merged receipt composite/c17/charts/receipt.json sha b32bcfd61113cad2... (64 present, 0 invalid), alignment.json sha 4006c1796946202f... "
 "(92 MoE layers -> 64 trunk layers, dense K3 L0 -> nowhere); evidence omni-evidence-27bu-v1.json sha a0a7f0da58e1d1b2...; --chart-bias-ceiling 0 (record only). "
 f"union_receipt.json file sha256 {sha(f'{s}/child/union_receipt.json')}; veto_binding layers {u.get('veto_binding_layers')}/64 "
 "(all 7 are k3_rows_function_absent via a rank-starved pool Gram, not THE LAW: ridge veto 0/64); "
 f"15/64 layers re-solved by the 1.5.1 rung-skipping fallback. Install diagnostics (finite fit numbers are DIAGNOSTICS, owner policy 2026-10-06): "
 f"K3 target energy held mean {i.get('k3_target_energy_held_mean'):.4f}, union held relL2 mean {i.get('fit_held_relL2_mean'):.4f}, update_scale mean {i.get('update_scale_mean'):.4f}. "
 f"Load/identity (c10_virtual_child_check 1.2.0, tree ef1ffb9, GPU6 bf16): virtual child = the partial, structural load strict-equivalent, header {c['child']['header_parameters']:,} "
 f"< parent 27,781,427,952 (headroom 837,209,328; with the five media units 245,772,199), logits finite {a.get('logits_finite')}; "
 f"R-Q top-1 vs S3 trunk {a.get('rq_top1_vs_trunk_count')}, vs 27B parent {a.get('rq_top1_vs_reference'):.4f} (trunk {t.get('rq_top1_vs_reference'):.4f}); "
 f"held NLL {a.get('held_nll'):.4f} (trunk {t.get('held_nll'):.4f}, parent {p.get('held_nll', float('nan')):.4f}); held top-1 vs parent {a.get('held_top1_vs_reference'):.4f}; "
 f"max residual norm ratio {a.get('residual_norm_ratio_max'):.4f} at L{a.get('residual_norm_ratio_max_layer')}. "
 + (f"Written child forward (as the reference beside the virtual child): logits finite {w.get('logits_finite')}, R-Q top-1 vs trunk {w.get('rq_top1_vs_trunk_count')}, held NLL {w.get('held_nll')}. " if w else "")
 + f"Counts: s5_register_counts.py receipt register_counts.json sha {sha(counts)} (per-layer parent MLP 267,386,880 -> 66,846,720 Qwen rows; "
 f"K3 rows added {c['k3_rows_added']:,}; native-export difference {c['native_export_extra_parameters']:,}; mtp {c['parent']['mtp_header_parameters']:,}; identity residual {c['accounting_identity_residual']}). "
 "The K3 held-row gains are in-sample (c17 charts were fitted on taps of the same six held rows); no decoded text has been read; no frozen review yet. Not trained.")
EOF
)
ARGS+=(--notes "$NOTES")
echo "--- notes: $NOTES"
timeout 280 $PY "$TOOL" "${ARGS[@]}" "${CONFIRM[@]}"
echo "=== exit $? ($(date -u +%FT%TZ))"
