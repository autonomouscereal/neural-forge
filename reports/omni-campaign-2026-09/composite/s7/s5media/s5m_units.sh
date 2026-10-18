#!/bin/sh
# S7-on-S5 media lane (Claude, 2026-10-13): rebuild the five S7 units with s7_unit_build 1.3.0 into
# $OC/composite/s7/units_s5/<source>/<unit> (the S3-bound units under composite/s7/units are not touched),
# then compare every rebuilt container with the S3 unit's pre-chart state (charted_block.json.pre-m1 + files).
set -u
OC=/home/cereal/omni-campaign
S7=$OC/composite/s7
U5=$S7/units_s5
L=$S7/s5media/logs
EF=$OC/import/rq27b/tree_ef1ffb9
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC=$OC MEDIAFOLD_ROOT=$OC/media CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY="nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python"
mkdir -p $U5/ace $U5/h3 $U5/qimg $U5/hy3d $U5/qvl32 $L
cd $OC/composite/tools
echo "start $(date -u +%FT%TZ) s7_unit_build sha $(sha256sum s7_unit_build.py | cut -c1-64)"
CAP=/mnt/lexar/omni-campaign/qimg-broad-capture-sep07/run/capture_batch1
HY_PARENT=/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors
HY_CHILD=/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/dense-child-blocks/dense-child-block-10-1c1709d9-8afb-47f0-931f-35e01f90f125/dense-child-block10/dense_child.safetensors
B0=/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/initial-gpu4-attempt2/capture-block0/block-00/capture.safetensors
QV_COMP=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['provenance']['source_component'])" $S7/units/qvl32/qvl32_vit_L13/charted_block.json)
QV_CHILD=$(dirname "$QV_COMP")
for x in ace/ace_dit_L12 h3/h3_dit_L25_curve6 qimg/qimg_mmdit_L30_img hy3d/hy3d_dit_L10 qvl32/qvl32_vit_L13; do
  if [ -e $U5/$x ]; then echo "EXISTS $U5/$x -- stop"; exit 2; fi
done
echo "== ace";  timeout 280 $PY s7_unit_build.py ace --src $OC/media/out/c16_L12_v4 --out $U5/ace/ace_dit_L12 > $L/unit_ace.json 2> $L/unit_ace.err; echo "exit=$?"
echo "== h3";   timeout 280 $PY s7_unit_build.py h3 --src $OC/composite/media/h3-native-curve6-sep07 --out $U5/h3/h3_dit_L25_curve6 > $L/unit_h3.json 2> $L/unit_h3.err; echo "exit=$?"
echo "== qimg"; timeout 280 $PY s7_unit_build.py qimg --src /mnt/lexar/omni-campaign/qimg-ownerpolicy-v4-ee1a6166-3e91-405d-8381-3e3ff4c91194 --block 30 \
   --captures $CAP/b1_fit_bicycle_sigma00.safetensors $CAP/b1_fit_bicycle_sigma01.safetensors $CAP/b1_fit_bicycle_sigma02.safetensors $CAP/b1_fit_bicycle_sigma03.safetensors \
   --time-reference $S7/m3/qimg_time_ref_fp32/time_reference.safetensors --out $U5/qimg/qimg_mmdit_L30_img > $L/unit_qimg.json 2> $L/unit_qimg.err; echo "exit=$?"
echo "== hy3d"; timeout 280 $PY s7_unit_build.py hy3d --src $HY_PARENT --dense-child $HY_CHILD --block 10 --out $U5/hy3d/hy3d_dit_L10 --hy3d-block0-capture $B0 > $L/unit_hy3d.json 2> $L/unit_hy3d.err; echo "exit=$?"
echo "== qvl";  timeout 280 $PY s7_unit_build.py qvl --src $QV_CHILD --parent /home/cereal/comfyui/models/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors --block 13 --out $U5/qvl32/qvl32_vit_L13 > $L/unit_qvl.json 2> $L/unit_qvl.err; echo "exit=$?"
for f in $L/unit_*.err; do echo "-- $f"; tail -2 $f; done
$PY $S7/s5media/s5m_units_compare.py $S7/units $U5 > $L/units_compare.json; echo "compare_exit=$?"
python3 -c "import json,sys;r=json.load(open(sys.argv[1]));[print(k,v['verdict'],'missing',v['missing'],'mismatch',v['mismatch'],'hard_diffs',v['json_hard_diffs']) for k,v in r['units'].items()];print('ALL', r['verdict'])" $L/units_compare.json
echo "end $(date -u +%FT%TZ)"
