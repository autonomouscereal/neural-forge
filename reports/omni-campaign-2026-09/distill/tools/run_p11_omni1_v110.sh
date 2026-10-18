#!/usr/bin/env bash
# run_p11_omni1_v110.sh <gpu-index> -- S0 P11 rerun: the OMNI1 256k needle baseline with p11_needle_engine 1.1.x
# (plan Step 5c; needed before the D2 seal, not before the D2 start).  Prepared 2026-10-14 by lane p2p11; NOT
# launched -- the card is decided later.  Launch it detached once its card is free:
#   setsid nohup bash $OC/distill/tools/run_p11_omni1_v110.sh 4 > $OC/distill/tps/p11-v110/driver-gpu4.log 2>&1 < /dev/null &
#
# DRIVER 1.1.0 (2026-10-14, review fixes; 1.0.0 = 83b53bc8 in _superseded): the card gate is p2p11_card_guard.sh 1.0.0,
# run before the MemAvailable wait, AFTER it and before every step; GPU0 / GPU4 need the positive D1 seal marker
# $Q/d1.sealed, not just "no D1 container".  Pins p11_needle_engine 1.1.1 (1.1.0's changelog overstated the rescore).
#
# WHY A RERUN.  The 2026-10-13 baseline (1.0.0) is hit=0 exact=0 everywhere because it scored only the strict argmax.
# `rescore` (receipt $OC/distill/tps/p2p11/p11_omni1_rescore_v110_of_v100.json, sha 2f1213af...) shows what that hid
# and what it did NOT: wrong_digits = 0 in all 7 docs, but the FIRST digit -- the one position not teacher-forced on
# the key prefix -- misses under the content readout in all 6 needle docs (0/6).  At that slot the top-5 is control
# tokens and then a prose opener ("The"; at 256k also "Based", "Let"), and the answer's first digit ranks 9 / 16
# (32k d25 / d75), 12 / 27 (128k), 6 / 7 (256k).  Digits 2-7 are right under teacher forcing; only 256k-d25 is exact,
# and only under the 10-id digit readout.  So the open question is first-digit RETRIEVAL at 32k / 128k, not only a
# control-token preference.  1.1.x saves the top-16 of every scored row, so this read settles it.  Same needle file
# (needles_v1, sha 052e2caf...), so D2 / D4 compare like with like.
#
# JOBS (one status line each in $Q/p11.status):
#   1 recurrence-parity  fp32 on the card (TRITON_F32_DEFAULT=ieee for this job only; the read keeps the defaults of
#                        2026-10-13): tree forward_with_state vs loop, fla-naive-chunk and the TRITON fla kernel,
#                        KDA layers auto (first three + the middle one), L 256 / 4,096, E2 chunks 96 / 1,024.  The fla
#                        kernel had never been checked in fp32 (the CPU arm of 2026-10-14 checked E2's mapping through
#                        fla's own torch references).  A FAIL stops the job: no read on a recurrence that is wrong.
#   2 parity             the E2 gate exactly as 2026-10-13 (bf16, kda fla, L 4,096, chunk 1,024, 256 MiB score budget)
#   3 read               32k (2 depths + no-needle control), 128k, 256k (2 depths each), label p11_omni1_v110
# CARD RULES (refused otherwise; see p2p11_card_guard.sh): 0 / 4 RTX 3090 only after the D1 SEAL (no
# codex-d1-native* container, $Q/d1.sealed written by the D1 sealer, no runtime_leases/D1-* entry newer than it or
# changed in the last 30 min) and the card < 2,000 MiB; 6 RTX 8000 only while $Q/gpu6.slot.p11 (written by the gpu6
# lane owner) exists; P40s refused (no triton on sm61: the fla kernel cannot run); GPU5 never.
# HOST: MemAvailable >= 40 GiB (2026-10-13 peak RSS 17.7 GB).
# ETA: 2026-10-13 on one 3090: parity 20 min, read 1 h 57 min (32k 13.5 min, 128k 27 min, 256k 75 min); + ~20 min
# for job 1 -> about 2 h 40 min.
set -uo pipefail
GPU_INDEX=${1:?usage: run_p11_omni1_v110.sh <gpu-index 0|4|6>}
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
OUT=$OC/distill/tps/p11-v110
PY=/home/cereal/model-surgery-lab/venv/bin/python
T=$OC/distill/tools/p11_needle_engine.py
WANT_T=${WANT_T:-a894b8658b696d395943f02f4240ae018256e9d6aa6294c5b124c206bbc92d36}
G=$OC/distill/tools/p2p11_card_guard.sh
WANT_G=822ebdd5ae0681e8f4c7d9059d751335de808f6bf08df7e09c2c54bad15ddb32
M=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
NEEDLES=$OC/distill/p11/needles_v1
STATUS=$Q/p11.status
LANE=p2p11
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
status() { echo "$(stamp) $*" >> $STATUS; echo "[$(stamp)] STATUS $*"; }
refuse() { echo "REFUSED: $*"; status "lane=$LANE job=p11-v110 gpu=$GPU_INDEX state=refused reason=\"$*\""; exit 2; }

case $GPU_INDEX in
  0|4) LOCK=$Q/gpu$GPU_INDEX.lock; SLOT_FILE= ;;
  6)   LOCK=$Q/gpu6.p11.lock; SLOT_FILE=$Q/gpu6.slot.p11 ;;
  *)   refuse "GPU$GPU_INDEX cannot run the fla kernel P11 reads with (0 / 4 after the D1 seal, 6 with a slot file)" ;;
esac
[ "$(sha256sum $T | cut -c1-64)" = "$WANT_T" ] || refuse "$T sha256 $(sha256sum $T | cut -c1-16) is not the pinned 1.1.1 ${WANT_T:0:16}"
[ "$(sha256sum $G | cut -c1-64)" = "$WANT_G" ] || refuse "$G sha256 $(sha256sum $G | cut -c1-16) is not the pinned guard ${WANT_G:0:16}"
[ "$(sha256sum $NEEDLES/needles.safetensors | cut -c1-64)" = "052e2cafcc6a6a8f517f232e2c2acacd6a77648632bffa77af8b3cfceabd6a5a" ] \
  || refuse "needles_v1 changed: the baseline must read the 2026-10-13 needle file"
. $G
card_clear "p11:start"   # fail fast, before the lock
mkdir -p $OUT
exec 9> $LOCK
flock -n 9 || refuse "another job holds $LOCK"
gate_card 40 p11   # card_clear, the MemAvailable >= 40 GiB wait (up to 6 h), card_clear AGAIN: nothing loads before this
release() { local rc=$?; if [ "$GPU_INDEX" = 6 ]; then rm -f $Q/gpu6.slot.p11; fi; status "lane=$LANE job=p11-v110 gpu=$GPU_INDEX state=end rc=$rc"; }
trap release EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_INDEX   # torch's default order differs from nvidia-smi here
export PYTHONPATH=$OC/distill/tools OC_CALIBRATION_DIR=$OC/calibration OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON=(--model $M --chimera-src $OC/import/rq27b/tree_ef1ffb9 --needles $NEEDLES --device cuda:0)
sha256sum $T $G $0
status "lane=$LANE job=p11-v110 gpu=$GPU_INDEX state=started tool=${WANT_T:0:16} $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader -i $GPU_INDEX)"

TRITON_F32_DEFAULT=ieee nice -n 15 $PY $T recurrence-parity "${COMMON[@]}" --dtype float32 --layers auto --lengths 256,4096 --chunks 96,1024 \
  --arms loop,fla-naive-chunk,fla --out $OUT/p11_omni1_recurrence_parity_gpu${GPU_INDEX}_fp32_fla.json > $OUT/1_recurrence.log 2>&1
rc=$?
line=$(grep '^RECURRENCE PARITY' $OUT/1_recurrence.log | tail -1 | cut -c1-500)
status "lane=$LANE job=p11-v110 step=recurrence-parity rc=$rc $line"
if [ $rc -ne 0 ] && [ "${P11_FORCE:-0}" != 1 ]; then refuse "fp32 recurrence parity rc=$rc: $line"; fi

card_clear "p11:e2-parity"   # the card can change hands between steps (a D1 attempt, the gpu6 slot withdrawn)
nice -n 15 $PY $T parity "${COMMON[@]}" --dtype bfloat16 --kda-kernel fla --length 4096 --chunk 1024 --attn-budget-mib 256 \
  --parity-min-top1 0.90 --out $OUT/p11_omni1_v110_e2_parity_gpu_bf16_fla_L4096.json > $OUT/2_parity.log 2>&1
rc=$?
status "lane=$LANE job=p11-v110 step=e2-parity rc=$rc $(grep '^E2 PARITY' $OUT/2_parity.log | tail -1 | cut -c1-400)"
[ $rc -eq 0 ] || refuse "E2 parity rc=$rc"

card_clear "p11:read"
nice -n 15 $PY $T read "${COMMON[@]}" --dtype bfloat16 --kda-kernel fla --lengths 32768,131072,262144 --chunk 8192 \
  --attn-budget-mib 2048 --parity-length 0 --out-dir $OUT --label p11_omni1_v110 > $OUT/3_read.log 2>&1
rc=$?
status "lane=$LANE job=p11-v110 step=read rc=$rc $(grep '^P11 SUMMARY' $OUT/3_read.log | tail -1 | cut -c1-900)"
exit $rc
