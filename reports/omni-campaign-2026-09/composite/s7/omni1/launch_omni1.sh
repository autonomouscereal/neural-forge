#!/bin/sh
# OMNI1 install (S7 step 8), OMNI1 assembly lane, Claude, 2026-10-13.
#   c16_install_media.py 1.4.0 on the WRITTEN S5 child (OC02-F-Q38-27BU-C2K4352K3U) with the five S5-bound units
#   (composite/s7/units_s5, charts refitted on S5 host rows), the interface set of record (D1: interface_latent,
#   D2: its c11 rows), the C18 music_codes band, --charted-dtype match, --lineage-sources (the two sources the S5
#   child already carries).  CPU only (the tool reads lazy views and streams untouched tensors).
#   Output on the Lexar SSD (root keeps its 60 GB floor); --min-free-gib 15 is the Lexar margin after the ~55 GB write.
#   usage: launch_omni1.sh [--overwrite]
set -u
OC=/home/cereal/omni-campaign
S7=$OC/composite/s7
EF=$OC/import/rq27b/tree_ef1ffb9
U=$S7/units_s5
OUT=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src CHIMERA_SRC=$EF/src MEDIAFOLD_ROOT=$OC/media OC_LEDGER_ROOT=$OC/ledger OC=$OC
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=/home/cereal/model-surgery-lab/venv/bin/python
echo "start $(date -u +%FT%TZ) host $(hostname)"
sha256sum $OC/composite/tools/c16_install_media.py $S7/omni1/omni1_lineage.py $S7/omni1/omni1_lineage_sources.json $0
df -h / /mnt/lexar | cat
mkdir -p $(dirname $OUT)
cd $OC/composite/tools
nice -n 15 /usr/bin/time -v $PY c16_install_media.py \
  --child $OC/composite/c10/S5-C2K4352K3U/child --out $OUT \
  --c15 $S7/interface_latent --c11 $S7/interface_latent/c11 --c18 $S7/c18/music_codes_band.safetensors \
  --blocks $U/ace/ace_dit_L12 $U/h3/h3_dit_L25_curve6 $U/qimg/qimg_mmdit_L30_img $U/hy3d/hy3d_dit_L10 $U/qvl32/qvl32_vit_L13 \
  --pair-by depth-fraction --arm insert --charted-dtype match --chimera-src $EF \
  --checkpoint-id OC02-F-Q38-27BU-OMNI1 --parent-checkpoint-id OC02-F-Q38-27BU-C2K4352K3U \
  --accounting-parent-directory /home/cereal/models/Qwen3.8-27B-Uncensored --parent-parameter-count 27781427952 \
  --lineage-sources $S7/omni1/omni1_lineage_sources.json \
  --min-free-gib 15 "$@"
rc=$?
echo "c16_exit=$rc $(date -u +%FT%TZ)"
ls -la $OUT; du -sh $OUT; df -h / /mnt/lexar | cat
[ -f $OUT/composite_manifest.json ] && sha256sum $OUT/composite_manifest.json && cat $OUT/composite_manifest.json.sha256
echo "end $(date -u +%FT%TZ)"
