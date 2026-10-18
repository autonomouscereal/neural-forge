#!/bin/sh
# S7-on-S5 media lane (Claude, 2026-10-13): (a) s7_dryrun 1.1.0 with ALL FIVE real units + interface_latent C15 +
# its C11 rows + the C18 band, --charted-dtype match, into a tiny 5120-wide synthetic Chimera child
# (native_feature_dim 6144, music_codes band 65,536 rows, 8 layers), then s7_dryrun_check (load + text forward);
# (b) s5m_plan_check: the same inputs against the WRITTEN S5 child's config and tensors (no write).
#   usage: s5m_dryrun.sh <units root> <tap rows root> <scratch dir> <run name>
set -u
OC=/home/cereal/omni-campaign
S7=$OC/composite/s7
EF=$OC/import/rq27b/tree_ef1ffb9
UR=$1; TAP=$2; SC=$3; RUN=$4
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC=$OC MEDIAFOLD_ROOT=$OC/media OC_LEDGER_ROOT=$OC/ledger CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY="nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python"
mkdir -p $SC
cd $OC/composite/tools
U="$UR/ace/ace_dit_L12 $UR/h3/h3_dit_L25_curve6 $UR/qimg/qimg_mmdit_L30_img $UR/hy3d/hy3d_dit_L10 $UR/qvl32/qvl32_vit_L13"
echo "start $(date -u +%FT%TZ) dryrun sha $(sha256sum s7_dryrun.py | cut -c1-64) c16 sha $(sha256sum c16_install_media.py | cut -c1-64)"
$PY s7_dryrun.py --units $U --scratch $SC --hidden 5120 --rows 1024 --run-name $RUN --build-child --charted-dtype match \
  --feature 6144 --music-band 65536 --layers 8 --intermediate 96 \
  --c15 $S7/interface_latent --c11 $S7/interface_latent/c11 --c18 $S7/c18/music_codes_band.safetensors \
  > $SC/dryrun_$RUN.json 2> $SC/dryrun_$RUN.log
echo "dryrun_exit=$?"
tail -3 $SC/dryrun_$RUN.log
$PY s7_dryrun_check.py --out $SC/out_$RUN --child $SC/tiny_child > $SC/check_$RUN.json 2> $SC/check_$RUN.err
echo "check_exit=$?"
tail -2 $SC/check_$RUN.err
python3 - $SC/check_$RUN.json $SC/out_$RUN/composite_manifest.json <<'P'
import json, sys
r = json.load(open(sys.argv[1]))
print({k: r.get(k) for k in ["loaded_tensors", "parameters", "child_parameters", "layers", "charted_layers", "text_hidden_bit_identical", "text_logits_finite"]})
m = json.load(open(sys.argv[2]))
print("manifest keys", sorted(m))
for k in ("header_count", "components", "register"):
    if k in m:
        print(k, json.dumps(m[k], default=str)[:1500])
for p in (m.get("blocks") or {}).get("placements", []):
    print("placement", p.get("container"), "after", p.get("insert_after", p.get("layer_index")), "types", p.get("type_ids"), "chart", p.get("chart_source"), "dtype", p.get("charted_dtype", p.get("dtype")))
print("interface", [(e["name"], e["type_id"], e["feature_band"], e["emission_band"]) for e in (m.get("interface") or {}).get("entries", [])])
b = m.get("band") or {}
print("band", b.get("band"), b.get("rows_written"), b.get("tied"))
print("warnings", m.get("warnings"))
P
sha256sum $SC/out_$RUN/composite_manifest.json $SC/check_$RUN.json $SC/dryrun_$RUN.json
echo "== plan check vs the S5 child"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $PY $S7/s5media/s5m_plan_check.py --child $OC/composite/c10/S5-C2K4352K3U/child \
  --tools $OC/composite/tools --c15 $S7/interface_latent --c11 $S7/interface_latent/c11 --c18 $S7/c18/music_codes_band.safetensors \
  --units $U --tap-rows $TAP --parent-parameter-count 27781427952 --out $SC/plan_check_$RUN.json 2> $SC/plan_check_$RUN.err
echo "plan_exit=$?"
tail -3 $SC/plan_check_$RUN.err
du -sh $SC; df -h / | tail -1
echo "end $(date -u +%FT%TZ)"
