#!/usr/bin/env bash
# qtext lane, tool 1.0.6 (on top of 1.0.5: reference boxes re-stated on the parent's 0-1000 grid or
# quarantined; session ids unique to the microsecond and session records ordered by a per-cache
# sequence taken under the writer lock -- the root cause of the hard-kill / resume test flakes),
# 2026-10-14.  CPU only: GPU0/GPU4 are D1 r4's (Codex), GPU6 is lane gpu6's.  Steps:
#   0  negative control: the 1.0.6 test file against the 1.0.5 tool (the new tests must fail)
#   1-3  the full suite three times in a row, fresh basetemp each (must be green 3/3)
#   4  the real corpus-lane VL JSONL through load_vl_corpus 1.0.6 (box summary + receipt)
# Detached:  setsid nohup bash run_qtext_cpu_v106_2026-10-14.sh > $LOG 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
S=/mnt/lexar/omni-campaign/scratch/claude-vl-2026-10-14
R=$OC/distill/tps
CORPUS=$OC/distill/corpus/vision_in
mkdir -p "$S"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
export QTEXT_TPS_PROBE=$OC/distill/tps_probe.py TMPDIR=$S
PYTEST="nice -n 15 $PY -m pytest -o addopts= -p no:cacheprovider -p no:warnings -q"
sha256sum $T/qtext_teacher_cache.py $T/test_qtext_teacher_cache.py $T/qtext_vl_box_corpus_check.py \
  $T/_superseded/qtext_teacher_cache_1.0.5.py $OC/distill/tps_probe.py
$PY $T/qtext_teacher_cache.py --version
echo "===== 0 negative control: 1.0.6 tests vs the 1.0.5 tool $(date -u +%FT%TZ)"
N=$S/negctl
rm -rf "$N"; mkdir -p "$N"
cp $T/_superseded/qtext_teacher_cache_1.0.5.py $N/qtext_teacher_cache.py
cp $T/test_qtext_teacher_cache.py $N/
(cd $N && PYTHONPATH=$N $PYTEST test_qtext_teacher_cache.py --basetemp=$S/bt-negctl 2>&1 | grep -E "^FAILED|passed|failed" | cut -c1-200)
rm -rf "$N" $S/bt-negctl
for i in 1 2 3; do
  echo "===== $i full suite run $i/3 $(date -u +%FT%TZ)"
  (cd $T && $PYTEST test_qtext_teacher_cache.py --basetemp=$S/bt-run$i 2>&1 | grep -E "^FAILED|^E  |passed|failed" | head -40 | cut -c1-300)
  rm -rf $S/bt-run$i
done
echo "===== 4 real corpus through load_vl_corpus 1.0.6 $(date -u +%FT%TZ)"
nice -n 15 $PY $T/qtext_vl_box_corpus_check.py --receipt $R/qtext-vl-boxes-corpus-v106-2026-10-14.json \
  $CORPUS/vision_in.jsonl $CORPUS/video_in.jsonl
echo "exit $?"
echo "===== done $(date -u +%FT%TZ)"
