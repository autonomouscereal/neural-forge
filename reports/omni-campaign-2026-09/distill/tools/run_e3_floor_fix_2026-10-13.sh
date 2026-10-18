#!/usr/bin/env bash
# E3 1.0.1 (free-space floor counts the quants) - re-test on the server, CPU only.
#   f1 e3-tests-101   the full chimera_gguf_export suite (1.0.1 + its tests) + the P1 suite
#   f2 negctl-100     the new floor test run against the kept 1.0.0 module: must FAIL
# Launch detached:
#   setsid nohup bash $OC/distill/tools/run_e3_floor_fix_2026-10-13.sh > $OC/distill/logs/export/f_floor_fix.log 2>&1 < /dev/null &
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
EF=$OC/import/rq27b/tree_ef1ffb9
T=$OC/distill/tools
LOGS=$OC/distill/logs/export
SCR=$OC/distill/.scratch/e3
Q=$OC/distill/queues
STATUS=$Q/export.status
mkdir -p $LOGS $SCR/tests $Q
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src:$T
export OC_LEDGER_DIR=$OC/ledger OC_DISTILL_DIR=$OC/distill
NICE="nice -n 15"
status() { echo "$(date -u +%FT%TZ) $1 rc=$2 $3" >> $STATUS; echo "== STATUS $1 rc=$2 $3"; }
keyline() { grep -E "$1" "$2" | tail -1 | cut -c1-400; }
echo "floor-fix start $(date -u +%FT%TZ) host $(hostname) pid $$"
sha256sum $T/chimera_gguf_export.py $T/test_chimera_gguf_export.py $T/p1_qwen35_reader.py $T/test_p1_qwen35_reader.py

# f1 ---------------------------------------------------------------------------
cd $T
E3_TEST_TMP=$SCR/tests $NICE /usr/bin/time -v $PY test_chimera_gguf_export.py -v > $LOGS/f1_e3_tests_101.log 2>&1
rc=$?
$NICE $PY test_p1_qwen35_reader.py > $LOGS/f1_p1_tests.log 2>&1
rc_p1=$?
status f1-e3-tests-101 $rc "$(grep -E '^Ran|^OK|^FAILED' $LOGS/f1_e3_tests_101.log | tr '\n' ' ') | p1 rc=$rc_p1 $(grep -E '^Ran|^OK|^FAILED' $LOGS/f1_p1_tests.log | tr '\n' ' ')"

# f2 ---------------------------------------------------------------------------
# the 1.0.0 module beside the 1.0.1 tests in a scratch dir; only the floor test
NEG=$SCR/negctl-100
rm -rf $NEG && mkdir -p $NEG
cp $T/_superseded/chimera_gguf_export_1.0.0.py $NEG/chimera_gguf_export.py
cp $T/test_chimera_gguf_export.py $T/p1_qwen35_reader.py $T/test_p1_qwen35_reader.py $NEG/
cd $NEG
E3_TEST_TMP=$SCR/tests PYTHONPATH=$EF/src:$NEG $NICE $PY -m unittest -v \
    test_chimera_gguf_export.TestDiskFloor.test_the_floor_counts_the_requested_quants > $LOGS/f2_negctl_100.log 2>&1
rc=$?
status f2-negctl-100 $rc "expect FAIL: $(grep -E '^Ran|^OK|^FAILED|AssertionError' $LOGS/f2_negctl_100.log | head -3 | tr '\n' ' ')"
rm -rf $NEG
echo "floor-fix end $(date -u +%FT%TZ)"
