#!/bin/bash
# s7_m1_drive.sh -- lane M1 driver: trunk host rows on ONE GPU (sequential, one unit at a time), then the
# CPU chart fits (sequential) as each unit's paired rows land. Claude, 2026-10-10.
#   usage: s7_m1_drive.sh <gpu-uuid> <unit> [<unit> ...]     (units in GPU order)
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
LOG=$OC/composite/s7/logs
G=$1; shift
UNITS="$*"
cd $OC
gpu_loop() {
  for u in $UNITS; do
    if [ -f composite/s7/tap_rows_s3/$u/paired_rows.json ]; then echo "[gpu] $u already has host rows"; continue; fi
    used=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$G" '$1==g{print $2}')
    while [ "${used:-99999}" -ge 2000 ]; do sleep 20; used=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$G" '$1==g{print $2}'); done
    echo "[gpu] $(date -u +%FT%TZ) $u start (card at ${used} MiB)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      PYTORCH_ALLOC_CONF=expandable_segments:True nice -n 19 $PY composite/tools/s7_host_rows.py --unit $u --device cuda:0 \
      > $LOG/m1_host_$u.log 2>&1 < /dev/null
    echo "[gpu] $(date -u +%FT%TZ) $u exit $?"
  done
}
fit_loop() {
  for u in ace_dit_L12 $UNITS; do
    case " $FITTED " in *" $u "*) continue;; esac
    FITTED="${FITTED:-} $u"
    while [ ! -f composite/s7/tap_rows_s3/$u/paired_rows.json ]; do
      if ! kill -0 $GPU_PID 2>/dev/null && [ ! -f composite/s7/tap_rows_s3/$u/paired_rows.json ]; then echo "[fit] $u: no host rows and the GPU loop ended"; break 2; fi
      sleep 30
    done
    extra=""
    [ "$u" = "qvl32_vit_L13" ] && extra="--type-ids 6"
    echo "[fit] $(date -u +%FT%TZ) $u start"
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 nice -n 19 $PY composite/tools/s7_chart_fit.py --unit $u $extra \
      > $LOG/m1_fit_$u.log 2>&1 < /dev/null
    echo "[fit] $(date -u +%FT%TZ) $u exit $?"
  done
}
FITTED=""
gpu_loop &
GPU_PID=$!
fit_loop
wait $GPU_PID
echo "[drive] $(date -u +%FT%TZ) done"
