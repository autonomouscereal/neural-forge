set -u
# PREP-B dry run: tiny synthetic Chimera child at hidden 5120 + the rebuilt hy3d_dit_L10 unit (real charts) through
# c16_install_media; then prove the time row reaches the block (s7_hy3d_timerow_check.py). CPU only.
OC=/home/cereal/omni-campaign
S7=$OC/composite/s7
PB=$S7/prep_b
SC=$PB/scratch_dryrun5120
EF=$OC/import/rq27b/tree_ef1ffb9
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC=$OC MEDIAFOLD_ROOT=$OC/media OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
PY="nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python"
cd $OC/composite/tools
mkdir -p $SC
U=$S7/units/hy3d/hy3d_dit_L10
$PY s7_dryrun.py --units $U --scratch $SC --hidden 5120 --rows 1024 --run-name PBT --build-child --charted-dtype match > $SC/dryrun_PBT.json 2> $SC/dryrun_PBT.log
echo "dryrun_exit=$?"
tail -2 $SC/dryrun_PBT.log
$PY s7_dryrun_check.py --out $SC/out_PBT --child $SC/tiny_child > $SC/check_PBT.json 2> $SC/check_PBT.err
echo "check_exit=$?"
python3 - <<P
import json
r=json.load(open("$SC/check_PBT.json"))
print({k:r[k] for k in ["loaded_tensors","parameters","child_parameters","charted_layers","text_hidden_bit_identical","text_logits_finite"]})
for p in r["blocks"]["placements"]:
    print(p["container"],"layer",p["layer_index"],"types",p["type_ids"],"chart_source",p["chart_source"])
P
sha256sum $SC/out_PBT/composite_manifest.json
$PY s7_hy3d_timerow_check.py --out $SC/out_PBT --unit $U --paired-rows $S7/tap_rows_s3/hy3d_dit_L10/paired_rows.safetensors \
  --parent-capture /mnt/lexar/omni-campaign/hy3d-native-4983c53b-7eee-4f49-980f-ab3788f5db69/dense-child-blocks/dense-child-block-10-1c1709d9-8afb-47f0-931f-35e01f90f125/capture-block10/block-10/capture.safetensors \
  --report $PB/timerow_check.json > $PB/logs/timerow_check.out 2> $PB/logs/timerow_check.err
echo "timerow_check_exit=$?"
cat $PB/logs/timerow_check.out; tail -3 $PB/logs/timerow_check.err
sha256sum $PB/timerow_check.json; cat $PB/timerow_check.json.sha256
echo DRYRUN_DONE
