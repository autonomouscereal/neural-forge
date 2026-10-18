#!/bin/bash
# OMNI1 registration, OC02-F-Q38-27BU-OMNI1 (2026-10-13, Claude, OMNI1 assembly lane).
#
# Prerequisites (each is checked below; nothing is typed from memory):
#   * the WRITTEN composite $OUT (c16_install_media.py 1.4.0) with composite_manifest.json (+ .sha256 sidecar);
#   * $S/omni1_register_counts.json from omni1_register_counts.py (header-derived trunk + media counts, slack, identity);
#   * $S/omni1_check.json from s7_omni1_check.py 1.0.0 (strict load, identity vs S5, R-Q prompts, media forwards).
#
# Lineage / accounting (oc_checkpoint.py rules; every growth stage of the chain is priced the same way):
#   parent checkpoint  OC02-F-Q38-27BU-C2K4352K3U (the S5 union child the media were installed into);
#   parent counts      the HF Qwen3.8-27B-Uncensored headers (27,781,427,952), the denominator S3 and S5 used;
#   components         the 64 qwen.mlp.layerNN lines (carried from S5, re-derived from the OMNI1 headers) and the five
#                      media block MLPs (<source>.<block>.mlp PARENT:CHILD, each <= 0.25).  ASSERTED form: no tensor
#                      prefix is shared by the accounting parent and the child (the media parents live in their source
#                      model files), so omni1_register_counts.py derives every number from the headers instead;
#   passed through     K3 rows (added), native-export difference, mtp (dropped), the mixers, the native interface,
#                      the media blocks' attention / time / norm tensors and their charts (added), the C15/C11 rows and
#                      the C18 band (written in place, 0 added); the added mass is stated as --accounting-slack.
#
# Usage:  omni1_register.sh            # dry run, writes nothing
#         omni1_register.sh --confirm  # seals the register record + ledger row
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
export OC_ROOT=$OC
S=$OC/composite/s7/omni1
OUT=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
PARENT=/home/cereal/models/Qwen3.8-27B-Uncensored
TOOL=$OC/ledger/oc_checkpoint.py
COUNTS=$S/omni1_register_counts.json
CHECK=$S/omni1_check.json
CONFIRM=()
if [ "${1:-}" = "--confirm" ]; then CONFIRM=(--confirm); fi
for f in "$OUT/composite_manifest.json" "$COUNTS" "$CHECK"; do
  [ -f "$f" ] || { echo "=== REFUSED: missing $f"; exit 3; }
done
[ "$(sha256sum "$OUT/composite_manifest.json" | cut -c1-64)" = "$(cut -c1-64 "$OUT/composite_manifest.json.sha256")" ] || { echo "=== REFUSED: manifest sidecar mismatch"; exit 3; }
echo "=== register with tool: $TOOL sha256 $(sha256sum "$TOOL" | cut -c1-64)  confirm=${CONFIRM[*]:-no}  ($(date -u +%FT%TZ))"
sha256sum "$0" "$COUNTS" "$CHECK" "$OUT/composite_manifest.json" "$S/omni1_lineage_sources.json"
cd $OC
mapfile -t COMPONENTS < <($PY -c 'import json,sys; [print(l) for l in json.load(open(sys.argv[1]))["component_lines"]]' "$COUNTS")
[ "${#COMPONENTS[@]}" = 69 ] || { echo "=== REFUSED: ${#COMPONENTS[@]} component lines, expected 69 (64 qwen + 5 media)"; exit 3; }
mapfile -t PASSED < <($PY - "$COUNTS" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
print(f"k3.ffn:ADDED-not-a-component:{c['k3_rows_added']}-parameters:64x3x5120x12288-rows-of-K3-dense-L1-92-lifted-through-c17-ridge-charts-into-blocks.k.ffn(inherited-from-S5)")
print(f"native-export:{c['native_export_extra_parameters']}-parameters:S3-native-trunk-non-FFN-tensors-vs-parent-non-MLP-non-mtp(inherited)")
print(f"mtp:dropped-not-folded:{c['parent']['mtp_header_parameters']}-parameters:inherited-from-S3")
print("gdn_mixer")
print("attention_mixer")
for m in c["media"]:
    base = m["component"].rsplit(".", 1)[0]
    print(f"{base}.non_mlp:ADDED:{m['non_mlp_block_params']}-parameters:attention/time/norm-of-the-source-block-at-~1x(blocks.{m['omni1_block']}.charted)")
    print(f"{base}.charts:ADDED:{m['chart_params']}-parameters:chart_in/chart_out-ridge-fitted-on-S5-host-rows(blocks.{m['omni1_block']}.charted.chart_*)")
print(f"interface:written-in-place:{c['interface_parameters_written_in_place']}-parameters:C15-latent-rows(interface_latent)+C11-rows-into-native_media_bridge/native_media_head(0-added)")
print(f"music_codes_band:written-in-place:{c['band_parameters_written_in_place']}-parameters:C18-ACE-LM-code-rows-into-embed/lm_head-music_codes-band(0-added)")
EOF
)
SLACK=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["accounting_slack_argued"])' "$COUNTS")
ARGS=(register --checkpoint-id OC02-F-Q38-27BU-OMNI1 --stage fold --wave oc02 --tag OMNI1)
ARGS+=(--parent-code Q38-27BU --parent-checkpoint-id OC02-F-Q38-27BU-C2K4352K3U)
ARGS+=(--directory "$OUT" --parent-directory "$PARENT" --parameter-count auto --parent-parameter-count auto)
for c in "${COMPONENTS[@]}"; do ARGS+=(--component "$c"); done
for p in "${PASSED[@]}"; do ARGS+=(--passed-through "$p"); done
ARGS+=(--accounting-slack "$SLACK")
ARGS+=(--recipe "$OUT/composite_manifest.json")
RC=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["tool"]["recipe_command"])' "$OUT/composite_manifest.json")
ARGS+=(--recipe-command "$RC (c16_install_media 1.4.0 sha $(sha256sum $OC/composite/tools/c16_install_media.py | cut -c1-16), via $S/launch_omni1.sh, CPU, nice 15)")
NOTES=$($PY - "$S" "$OUT" "$COUNTS" "$CHECK" <<'EOF'
import json, sys, hashlib
s, out, counts, check = sys.argv[1:5]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
oc = "/home/cereal/omni-campaign"
c = json.load(open(counts)); k = json.load(open(check)); m = json.load(open(f"{out}/composite_manifest.json"))
t = k["text_forward"]
blocks = []
for j, rep in sorted(k["charted"].items(), key=lambda x: int(x[0])):
    own = rep["segments"][rep["container"]]
    blocks.append(f"{rep['container']} (blocks.{j}, types {rep['type_ids']}): owns {own['rows_owned']} rows, write norm {own['write_norm_owned']:.4g}, "
                  f"write/input median {own.get('write_over_input_norm_median', float('nan')):.4g}, finite {own['write_finite']}, path {own['path']}")
x31 = "; ".join(f"{u} relL2 {r['x31_vs_s5_host_rows']['rel_l2']:.4g}" for u, r in k["media_forward"].items() if r.get("x31_vs_s5_host_rows"))
srcs = ", ".join(m["sources_all"]["names"]) if "sources_all" in m else "?"
print(
 "OMNI1 = the S7 composite: the S5 union child (OC02-F-Q38-27BU-C2K4352K3U: Qwen3.8-27B-Uncensored S3 trunk + Kimi K3 union rows) "
 "plus five charted 4x media blocks inserted after trunk layer 31 (ACE-Step ace_dit_L12, MiniMax H3 h3_dit_L25_curve6, Qwen-Image "
 "qimg_mmdit_L30_img, Hunyuan3D 2.1 hy3d_dit_L10, Qwen3-VL-32B qvl32_vit_L13; charts refitted on S5 host rows), the C15 latent interface "
 "rows + C11 rows of record (D1/D2: composite/s7/interface_latent) and the C18 music_codes band, by c16_install_media 1.4.0 --arm insert "
 f"--charted-dtype match. composite_manifest.json sha {sha(out + '/composite_manifest.json')} names {m['sources_all']['count'] if 'sources_all' in m else '?'} sources ({srcs}); "
 f"lineage file omni1_lineage_sources.json sha {sha(s + '/omni1_lineage_sources.json')}. "
 f"Upstream: S5 register record sha {sha(oc + '/registry/OC02-F-Q38-27BU-C2K4352K3U/OC02-F-Q38-27BU-C2K4352K3U-0001-register.json')}; "
 f"S7-on-S5 media lane summary S7_S5_MEDIA_UNITS.json sha {sha(oc + '/composite/s7/s5media/S7_S5_MEDIA_UNITS.json')}; "
 f"plan check plan_check_S5ALL.json sha {sha(oc + '/composite/s7/s5media/scratch_dryrun_s5/plan_check_S5ALL.json')}; "
 f"units compare units_compare.json sha {sha(oc + '/composite/s7/s5media/logs/units_compare.json')}; "
 f"interface_latent companions.json sha {sha(oc + '/composite/s7/interface_latent/companions.json')}, c11 {sha(oc + '/composite/s7/interface_latent/c11/companions.json')}; "
 f"C18 table sha {sha(oc + '/composite/s7/c18/music_codes_band.safetensors')}; decisions S7_DECISIONS_2026-10-10.md sha {sha(oc + '/composite/s7/S7_DECISIONS_2026-10-10.md')}. "
 f"Counts: omni1_register_counts.json sha {sha(counts)} (header-derived; media parents from the source files' own headers; identity residual "
 f"{c['accounting_identity_residual']}; OMNI1 {c['omni1']['header_parameters']:,} = S5 + {c['omni1_minus_s5']:,}; headroom {c['measured_drop']:,} under the parent). "
 f"Check: s7_omni1_check 1.0.0 receipt omni1_check.json sha {sha(check)} (GPU6 RTX 8000 bf16, tree ef1ffb9): {k['verdict']}; structural strict-equivalent "
 f"{k['structural']['strict_equivalent']}, {k['strict_load_coverage']['tensors_loaded']}/{k['strict_load_coverage']['tensors_expected']} tensors loaded strict and sha-verified; "
 f"trunk blocks bit-identical to S5 {k['identity_trunk_vs_s5']['bit_identical']}/{k['identity_trunk_vs_s5']['blocks']}; "
 f"R-Q top-1 (base vocab, {t['positions']} positions) OMNI1 vs S5 {t['rq_top1_omni1_vs_s5']}/{t['positions']} (max |logit diff| {t['max_abs_logit_diff_omni1_vs_s5_base_vocab']}), "
 f"vs 27B parent {t.get('rq_top1_omni1_vs_parent')}/{t['positions']} (S5 vs parent {t.get('rq_top1_s5_vs_parent')}); logits finite {t['logits_finite']}; "
 f"BAND FINDING: with the ledger bands NOT masked, OMNI1's full-vocab argmax falls in the band at {t['omni1_full_vocab_argmax_in_band_positions']}/{t['positions']} "
 f"R-Q positions (max band logit {t['omni1_max_band_logit']}; the C18 rows written into lm_head) -- text decoding must run with ledger_band_logit_mask until the band head rows are rescaled; "
 f"media forwards (one held native segment per block, whole composite): " + " | ".join(blocks) + f"; x31 vs S5 host rows (diagnostic) {x31}. "
 "Finite fit numbers are DIAGNOSTICS (owner policy 2026-10-06). Recorded limitations: H3 source positions are zeros (no h3_thw_v1 emitter); "
 "HY3D attn2 cross-attention not carried (D3), its time row is; QIMG parent is the FP8 e4m3 distribution widened to BF16; the C18 band is a weak "
 "extrapolation; media non-MLP tensors ride at ~1x (the >=4x rule is MLP-only, D6). No decoded text or media has been read; no frozen review (S8) yet. Not trained.")
EOF
)
ARGS+=(--notes "$NOTES")
echo "--- components: ${#COMPONENTS[@]}  passed-through: ${#PASSED[@]}  slack: $SLACK"
printf -- '--- passed: %s\n' "${PASSED[@]}"
echo "--- notes: $NOTES"
timeout 3000 $PY "$TOOL" "${ARGS[@]}" "${CONFIRM[@]}"
echo "=== exit $? ($(date -u +%FT%TZ))"
