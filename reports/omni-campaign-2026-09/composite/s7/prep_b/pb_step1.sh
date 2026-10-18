set -u
OC=/home/cereal/omni-campaign
S7=$OC/composite/s7
PB=$S7/prep_b
EF=$OC/import/rq27b/tree_ef1ffb9
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC=$OC MEDIAFOLD_ROOT=$OC/media OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
PY="nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python"
mkdir -p $PB/logs
# 1. keep the old unit / pairs / host rows under .pre-timerow (rename, never delete)
for d in $S7/units/hy3d/hy3d_dit_L10 $S7/pairs/hy3d_dit_L10 $S7/tap_rows_s3/hy3d_dit_L10; do
  if [ -e $d.pre-timerow ]; then echo "EXISTS $d.pre-timerow -- stop"; exit 2; fi
  mv $d $d.pre-timerow && echo "renamed $d -> $d.pre-timerow"
done
( cd $S7/units/hy3d/hy3d_dit_L10.pre-timerow && sha256sum * ) > $PB/pre_timerow_unit_sha256.txt
( cd $S7/pairs/hy3d_dit_L10.pre-timerow && sha256sum * ; cd $S7/tap_rows_s3/hy3d_dit_L10.pre-timerow && sha256sum * ) >> $PB/pre_timerow_unit_sha256.txt
cat $PB/pre_timerow_unit_sha256.txt
# 2. rebuild the unit with s7_unit_build 1.3.0
cd $OC/composite/tools
HY_PARENT=/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors
HY_CHILD=/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/dense-child-blocks/dense-child-block-10-1c1709d9-8afb-47f0-931f-35e01f90f125/dense-child-block10/dense_child.safetensors
B0=/mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/initial-gpu4-attempt2/capture-block0/block-00/capture.safetensors
timeout 250 $PY s7_unit_build.py hy3d --src $HY_PARENT --dense-child $HY_CHILD --block 10 --out $S7/units/hy3d/hy3d_dit_L10 --hy3d-block0-capture $B0 > $PB/logs/unit_build.json 2> $PB/logs/unit_build.err
echo "unit_build_exit=$?"
tail -3 $PB/logs/unit_build.err
python3 - <<PY
import json
new=json.load(open("$S7/units/hy3d/hy3d_dit_L10/charted_block.json"))
old=json.load(open("$S7/units/hy3d/hy3d_dit_L10.pre-timerow/charted_block.json"))
a=new["tensor_sha256"]["charted_block.safetensors"]; b=old["tensor_sha256"]["charted_block.safetensors"]
print("block tensor table identical to pre-timerow:", a==b, len(a))
r=new["runtime_requirements"]["hy3d_time_token_row"]
print("ref table identical:", r["reference_tensor_sha256"]==old["runtime_requirements"]["hy3d_time_token_row"]["reference_tensor_sha256"])
print("block0 diag:", json.dumps(r["reference_vs_block0_capture"]))
print("tool", new["s7_unit"]["tool"], new["s7_unit"]["tool_version"], new["s7_unit"]["component_4x"]["mlp_ratio"], new["s7_unit"]["canonical"]["installed_parameters_at_hidden_5120"])
PY
cmp $S7/units/hy3d/hy3d_dit_L10/charted_block.safetensors $S7/units/hy3d/hy3d_dit_L10.pre-timerow/charted_block.safetensors && echo "charted_block.safetensors byte-identical"
# 3. parent pairs WITH the time row (s7_pairs_parent 1.1.0)
timeout 200 $PY s7_pairs_parent.py hy3d > $PB/logs/pairs_hy3d.log 2>&1; echo "pairs_exit=$?"
tail -3 $PB/logs/pairs_hy3d.log
python3 -c "
import json;p=json.load(open('$S7/pairs/hy3d_dit_L10/parent_capture.json'))
print(p['tool'],p['rows'],p['rows_fit'],p['rows_held'],p['rows_time'],p['rows_time_held'],p['parent']['rows'])
print(json.dumps(p['time_token']['per_sample_diagnostics']))"
sha256sum $S7/pairs/hy3d_dit_L10/* $S7/units/hy3d/hy3d_dit_L10/*
