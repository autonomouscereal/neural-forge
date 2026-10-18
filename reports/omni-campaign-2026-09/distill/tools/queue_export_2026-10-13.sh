#!/bin/bash
# Lane `export` queue (CPU only, nice 15, <= 16 threads; no GPU, no wait on pid 917640).
# Installed as $OC/distill/queues/export.sh.  One status line per job in
# $OC/distill/queues/export.status: "<utc> <job> rc=<rc> <key line>".
#   j1 e3-tests   chimera_gguf_export.py suite + p1_qwen35_reader suite
#   j2 p9-map     S0-P9: E3 converter dry run on OMNI1 (headers only) vs the parent's map
#   j3 e3-l4      convert the first 4 OMNI1 text layers (3 linear + 1 full attention)
#                 through the P1 reader to a bf16 GGUF on root scratch (test artifact)
#   j4 e3-verify  read-back: every tensor byte-equal to the converter on the reader and
#                 the untransformed ones / norms byte-equal to the Chimera shards
#   j5 e3-logits  llama-server (CPU) vs torch fp32 on the same 4 layers: finite logits,
#                 top-1 agreement printed
#   j6 cleanup    rm the transient 4-layer GGUF (receipts keep its sha256)
# j3-j6 gate on the previous job's exit code; j2 runs regardless of j1.
set -u
OC=/home/cereal/omni-campaign
PY=/home/cereal/model-surgery-lab/venv/bin/python
EF=$OC/import/rq27b/tree_ef1ffb9
OMNI1=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
P=/home/cereal/models/Qwen3.8-27B-Uncensored
T=$OC/distill/tools
TPS=$OC/distill/tps
LOGS=$OC/distill/logs/export
SCR=$OC/distill/.scratch/e3
Q=$OC/distill/queues
STATUS=$Q/export.status
L4=$SCR/omni1-l4
mkdir -p $LOGS $SCR $Q
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src:$T
NICE="nice -n 15"
status() { echo "$(date -u +%FT%TZ) $1 rc=$2 $3" >> $STATUS; echo "== STATUS $1 rc=$2 $3"; }
keyline() { grep -E "$1" "$2" | tail -1 | cut -c1-400; }
echo "queue start $(date -u +%FT%TZ) host $(hostname) pid $$"
sha256sum $T/chimera_gguf_export.py $T/test_chimera_gguf_export.py $T/p1_qwen35_reader.py $T/test_p1_qwen35_reader.py

# j1 ---------------------------------------------------------------------------
cd $T
mkdir -p $SCR/tests
E3_TEST_TMP=$SCR/tests $NICE /usr/bin/time -v $PY test_chimera_gguf_export.py > $LOGS/j1_e3_tests.log 2>&1
rc=$?
$NICE $PY test_p1_qwen35_reader.py > $LOGS/j1_p1_tests.log 2>&1
rc_p1=$?
status j1-e3-tests $rc "$(keyline '^Ran|^OK|^FAILED' $LOGS/j1_e3_tests.log | tr '\n' ' ') | p1 rc=$rc_p1 $(grep -E '^Ran|^OK|^FAILED' $LOGS/j1_p1_tests.log | tr '\n' ' ')"

# j2 ---------------------------------------------------------------------------
$NICE /usr/bin/time -v $PY chimera_gguf_export.py map --chimera $OMNI1 --parent-dir $P --threads 16 \
  --work-dir $SCR --tag OC02-F-Q38-27BU-OMNI1 --receipt $TPS/p9_e3_tensor_map_omni1_2026-10-13.json \
  > $LOGS/j2_p9_map.log 2>&1
rc=$?
status j2-p9-map $rc "$(keyline 'P9 MAP' $LOGS/j2_p9_map.log)"

# j3 ---------------------------------------------------------------------------
rm -rf $L4
$NICE /usr/bin/time -v $PY chimera_gguf_export.py convert --chimera $OMNI1 --parent-dir $P --threads 16 \
  --out-dir $L4 --tag OMNI1-L4-e2e --layers 4 --min-free-gib 60 --work-dir $SCR \
  --quantization-stage none-test-artifact --receipt $TPS/e3_convert_omni1_l4_2026-10-13.json \
  > $LOGS/j3_convert_l4.log 2>&1
rc=$?
status j3-e3-l4 $rc "$(keyline 'E3 CONVERT' $LOGS/j3_convert_l4.log) | $(keyline 'Maximum resident' $LOGS/j3_convert_l4.log)"
[ $rc -eq 0 ] || { echo "queue stop after j3 $(date -u +%FT%TZ)"; exit $rc; }

# j4 ---------------------------------------------------------------------------
$NICE /usr/bin/time -v $PY chimera_gguf_export.py verify --gguf $L4/OMNI1-L4-e2e.bf16.gguf --chimera $OMNI1 \
  --parent-dir $P --layers 4 --threads 16 --work-dir $SCR --receipt $TPS/e3_verify_omni1_l4_2026-10-13.json \
  > $LOGS/j4_verify_l4.log 2>&1
rc=$?
status j4-e3-verify $rc "$(keyline 'E3 VERIFY' $LOGS/j4_verify_l4.log)"
[ $rc -eq 0 ] || { echo "queue stop after j4 $(date -u +%FT%TZ)"; exit $rc; }

# j5 ---------------------------------------------------------------------------
$NICE /usr/bin/time -v $PY chimera_gguf_export.py logits --gguf $L4/OMNI1-L4-e2e.bf16.gguf --chimera $OMNI1 \
  --parent-dir $P --layers 4 --threads 16 --tokens 32 --top-k 8 --tps-probe-dir $OC/distill --work-dir $SCR \
  --server-log $LOGS/j5_llama_server.log --receipt $TPS/e3_logits_omni1_l4_2026-10-13.json \
  > $LOGS/j5_logits_l4.log 2>&1
rc=$?
status j5-e3-logits $rc "$(keyline 'E3 LOGITS' $LOGS/j5_logits_l4.log)"
[ $rc -eq 0 ] || { echo "queue stop after j5 $(date -u +%FT%TZ)"; exit $rc; }

# j6 ---------------------------------------------------------------------------
sha=$(sha256sum $L4/OMNI1-L4-e2e.bf16.gguf | cut -c1-64)
rm -f $L4/OMNI1-L4-e2e.bf16.gguf
status j6-cleanup $? "removed transient $L4/OMNI1-L4-e2e.bf16.gguf sha256=$sha (receipts kept)"
echo "queue end $(date -u +%FT%TZ)"
