set -u
# PREP-B close-out: fit the chart of record for the rebuilt hy3d_dit_L10 unit (time rows x DEFAULT weight 4096, all arms). CPU fp64.
OC=/home/cereal/omni-campaign
PB=$OC/composite/s7/prep_b
EF=$OC/import/rq27b/tree_ef1ffb9
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC=$OC OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=
U=$OC/composite/s7/units/hy3d/hy3d_dit_L10
echo "start $(date -u +%FT%TZ)"
sha256sum $U/* $OC/composite/s7/tap_rows_s3/hy3d_dit_L10/paired_rows.safetensors $OC/composite/tools/s7_chart_fit.py
nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python $OC/composite/tools/s7_chart_fit.py --unit hy3d_dit_L10 \
  --weight-choice $PB/time_row_weight_sweep.json --lane "PREP-B 2026-10-11" < /dev/null
echo "fit_exit=$? end $(date -u +%FT%TZ)"
sha256sum $U/*
cat $U/chart_receipt.json.sha256
