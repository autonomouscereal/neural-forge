#!/bin/sh
# S7-on-S5 media lane (Claude, 2026-10-13): refit each unit's charts on the S5 host rows with s7_chart_fit 1.3.0
# (fp64, CPU only, nice 15, 16 BLAS threads, one fit at a time) into composite/s7/units_s5, as each unit's
# tap_rows_s5 receipt lands. Same arguments as the S3 fits (M1 / PREP-B) plus --trunk-label S5.
#   usage: s5m_fit.sh <unit> [<unit> ...]
set -u
OC=/home/cereal/omni-campaign
S7=$OC/composite/s7
L=$S7/s5media/logs
EF=$OC/import/rq27b/tree_ef1ffb9
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src OC=$OC CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
cd $OC
for u in "$@"; do
  while [ ! -f $S7/tap_rows_s5/$u/paired_rows.json ]; do sleep 30; done
  extra=""
  [ "$u" = "qvl32_vit_L13" ] && extra="--type-ids 6"
  [ "$u" = "hy3d_dit_L10" ] && extra="--weight-choice $S7/prep_b/time_row_weight_sweep.json"
  echo "[fit] $(date -u +%FT%TZ) $u start (rows sha $(sha256sum $S7/tap_rows_s5/$u/paired_rows.safetensors | cut -c1-16))"
  nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python composite/tools/s7_chart_fit.py --unit $u $extra \
    --tap-rows $S7/tap_rows_s5 --units-root $S7/units_s5 --chimera-src $EF --trunk-label S5 --lane "S7-on-S5 media 2026-10-13" \
    > $L/fit_$u.log 2>&1 < /dev/null
  echo "[fit] $(date -u +%FT%TZ) $u exit $?"
  tail -2 $L/fit_$u.log
done
echo "[fit] done $(date -u +%FT%TZ)"
