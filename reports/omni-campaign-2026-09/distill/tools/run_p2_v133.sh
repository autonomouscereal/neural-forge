#!/usr/bin/env bash
# run_p2_v133.sh <gpu-index> -- S0 P2 rerun (DISTILL_DESIGN section 4 S0, plan Step 5b): the T(ii) streamed bf16 fwd+bwd
# rate of OMNI1 on ONE card, lr = 0, with streamed_kd_trainer 1.3.3 (the P2 OOM fix).  Prepared 2026-10-14 by lane
# p2p11; NOT launched.  Launch it detached once its card is free:
#   setsid nohup bash $OC/distill/tools/run_p2_v133.sh 0 > $OC/distill/tps/p2-v133/driver-gpu0.log 2>&1 < /dev/null &
#
# DRIVER 1.1.0 (2026-10-14, review fixes; 1.0.0 = bd28b35f in _superseded): (1) the card gate is p2p11_card_guard.sh
# 1.0.0 and runs AGAIN after the MemAvailable wait and before every cell, and GPU0 / GPU4 need a POSITIVE D1 seal
# marker, not just "no D1 container"; (2) the summary is p2_v133_summary.py 1.0.0: every planned cell with its rc and
# OOM phase, "NO READING" when none completed, "long rung failed: <phase>" when L32768-mb1 did not complete; it is
# also written when the driver exits early.  Old cell logs / receipts of the same tag move to $OUT/prev/<UTC>/ first.
#
# CARD RULES (refused otherwise; checked before the MemAvailable wait, after it, and before every cell):
#   0 / 4  RTX 3090 (the T(ii) engine's card): only after the D1 SEAL (Plan Step 5b: "straight after the D1 seal"):
#          no codex-d1-native* container, the marker $Q/d1.sealed written by the D1 sealer, no runtime_leases/D1-*
#          entry newer than it or changed in the last 30 min, the card < 2,000 MiB, and $Q/gpu<N>.lock taken.
#   6      RTX 8000: only while the gpu6 lane owner's $Q/gpu6.slot.p2 exists (a slot between K3 passes); this script
#          never decides GPU6 by itself.  The slot file is removed when the job ends.
#   1/2/3/5  refused (P40s cannot run the bf16 / fla engine P2 measures; GPU5 is never used).
# HOST: waits for MemAvailable >= 60 GiB (pinned boundaries 64 x 32,768 x 5,120 x 2 B = 21.5 GB + OMNI1's 54 GB page
# cache share) and root >= 65 GB / Lexar >= 20 GB free.  Scratch on Lexar only.
#
# CELLS (each lr = 0, gain-lr = 0, bf16 on a 3090 / fp16 on GPU6, kernels auto = fla, 32,768 tokens per optimizer step):
#   L4096-mb8   4 steps, --ffn-chunk-tokens 4096 --recompute-tokens 4096 (row groups of one 4,096-token row)
#   L4096-mb4   4 steps, accumulation 2 (the same 32,768 tokens per step in two micro-batches)
#   L32768-mb1  3 steps, FFN slices of 4,096 tokens (the mixer half of a 32,768-token row is the irreducible term)
# A cell that dies OUT OF MEMORY is retried ONCE at half the FFN slice and head chunk (2,048 tokens / 128 MiB); its
# "OOM {...}" line names the phase.  Each train receipt carries memory_engine (measured per phase) and memory_model
# (predicted), so the model is checked on the real 27B at the same time.
# WATCH L32768-mb1: the GDN mixer half of one 32,768-token row cannot be split (recurrent state) and was measured on
# the P40 only with the torch GDN core; the fla core keeps its inputs in bf16 and is predicted smaller but is
# UNMEASURED.  An OOM there names backward.mixer.linear_attention, and halving the FFN slice will not help: the long
# rung then needs GDN state-carry checkpointing (not built) or a 46 GB card.
# RULE (STAGE_D_RULINGS, engine G5-1): steady tok/s >= 151 -> the streamed path; < 151 -> the measured split cell;
# no depth cut; tokens are cut only below 60 tok/s.  The summary prints the verdict; the ruling is recorded by a person.
set -uo pipefail
GPU_INDEX=${1:?usage: run_p2_v133.sh <gpu-index 0|4|6>}
OC=/home/cereal/omni-campaign
D=$OC/distill/tools
Q=$OC/distill/queues
TPS=$OC/distill/tps
OUT=$TPS/p2-v133
ALLOC=$OC/composite/receipts/allocator
PY=/home/cereal/model-surgery-lab/venv/bin/python
M08=/home/cereal/models/Qwen3.5-0.8B
M27=/home/cereal/models/Qwen3.8-27B-Uncensored
OMNI1=/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1
CORPUS=$OC/distill/corpus
SCR=/mnt/lexar/omni-campaign/scratch/p2-v133-gpu$GPU_INDEX
T=$D/streamed_kd_trainer.py
WANT_T=${WANT_T:-c329953d6afd3901def0e66b2c728ca0d792c6f47ead5617c40491c0b513566e}
G=$D/p2p11_card_guard.sh
WANT_G=822ebdd5ae0681e8f4c7d9059d751335de808f6bf08df7e09c2c54bad15ddb32
SUMMARY=$D/p2_v133_summary.py
WANT_S=9b1ef9672fe8089f737a10f17542bac027934f22a0f520fbea1f44725d97fb5a
STATUS=$Q/p2.status
LANE=p2p11
CELLS=(L4096-mb8 L4096-mb4 L32768-mb1)
LONG_CELL=L32768-mb1
CELLS_BEGUN=0
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
status() { echo "$(stamp) $*" >> $STATUS; echo "[$(stamp)] STATUS $*"; }
refuse() { echo "REFUSED: $*"; status "lane=$LANE job=p2-v133 gpu=$GPU_INDEX state=refused reason=\"$*\""; exit 2; }

case $GPU_INDEX in
  0|4) CARD="RTX 3090"; LOCK=$Q/gpu$GPU_INDEX.lock; DTYPE=bfloat16; SLOT_FILE= ;;
  6)   CARD="Quadro RTX 8000"; LOCK=$Q/gpu6.p2.lock; DTYPE=float16; SLOT_FILE=$Q/gpu6.slot.p2 ;;   # sm75: no bf16 tensor cores
  *)   refuse "GPU$GPU_INDEX is not a P2 card (0 / 4 = 3090 after the D1 seal, 6 = RTX 8000 with a slot file)" ;;
esac
[ "$(sha256sum $T | cut -c1-64)" = "$WANT_T" ] || refuse "$T sha256 $(sha256sum $T | cut -c1-16) is not the pinned 1.3.3 ${WANT_T:0:16}"
[ "$(sha256sum $G | cut -c1-64)" = "$WANT_G" ] || refuse "$G sha256 $(sha256sum $G | cut -c1-16) is not the pinned guard ${WANT_G:0:16}"
[ "$(sha256sum $SUMMARY | cut -c1-64)" = "$WANT_S" ] || refuse "$SUMMARY sha256 $(sha256sum $SUMMARY | cut -c1-16) is not the pinned ${WANT_S:0:16}"
. $G
card_clear "p2:start"   # fail fast, before the lock and any directory is made
mkdir -p $OUT $SCR $ALLOC
exec 9> $LOCK
flock -n 9 || refuse "another job holds $LOCK"
GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU_INDEX '$1==g{print $2}')
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_UUID PYTHONPATH=$D CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gate_card 60 p2   # card_clear, the MemAvailable >= 60 GiB wait (up to 6 h), card_clear AGAIN: nothing loads before this
r=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); x=$(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9)
[ "$r" -ge 65 ] && [ "$x" -ge 20 ] || refuse "disk: root ${r} GB (need 65), lexar ${x} GB (need 20)"

status "lane=$LANE job=p2-v133 gpu=$GPU_INDEX card=\"$CARD\" state=started trainer=${WANT_T:0:16} mem_avail_gib=$(mem_avail_gib)"
cat > $ALLOC/$LANE-p2-v133-gpu$GPU_INDEX.json <<JSON
{"name":"$LANE-p2-v133-gpu$GPU_INDEX","owner":"claude","lane":"$LANE","server":"ai","gpu_index":$GPU_INDEX,"gpu_uuids":["$GPU_UUID"],"card":"$CARD","purpose":"S0 P2 rerun: streamed_kd_trainer 1.3.3 rate cells on OMNI1 via P1, lr=0, bf16","python":"$PY","started_utc":"$(stamp)","state":"running","pid":$$,"log":"$OUT/driver-gpu$GPU_INDEX.log"}
JSON
summarize() {  # every planned cell, with rc / OOM phase, even without a receipt; NO READING when none completed
  local line
  $PY $SUMMARY --out $OUT --gpu $GPU_INDEX --cells "$(IFS=,; echo "${CELLS[*]}")" --long-cell $LONG_CELL > $OUT/summary.log 2>&1
  cat $OUT/summary.log
  line=$(grep '^P2 SUMMARY {' $OUT/summary.log | tail -1 | sed 's/.* -> //' | cut -c1-500)
  status "lane=$LANE job=p2-v133 gpu=$GPU_INDEX state=summary reading=\"${line:-summary failed, see $OUT/summary.log}\" summary=$OUT/p2-summary-v133-gpu$GPU_INDEX.json"
}
release() {
  local rc=$?
  [ "$CELLS_BEGUN" = 1 ] && summarize
  $PY -c 'import json, sys, time; p = sys.argv[1]; d = json.load(open(p)); d.update({"ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "released (process exited)"}); json.dump(d, open(p, "w"), indent=1)' \
    "$ALLOC/$LANE-p2-v133-gpu$GPU_INDEX.json"
  if [ "$GPU_INDEX" = 6 ]; then rm -f $Q/gpu6.slot.p2; fi
  status "lane=$LANE job=p2-v133 gpu=$GPU_INDEX state=end rc=$rc"
}
trap release EXIT
sha256sum $T $G $SUMMARY $D/qtext_teacher_cache.py $D/p1_qwen35_reader.py $0
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader -i $GPU_INDEX

# ---- corpora and scratch targets (content is irrelevant at lr = 0; the trainer needs a sealed cache of the width)
$PY - $CORPUS/text/text-00000.safetensors $CORPUS/long/long-00000.safetensors $SCR <<'PY'
import sys
from safetensors.torch import load_file, save_file
text, long_, out = sys.argv[1:4]
t = load_file(text); l = load_file(long_)
save_file({"input_ids": t["input_ids"][:64].contiguous(), "real_lengths": t["real_lengths"][:64].contiguous()}, f"{out}/p2_L4096.safetensors")
save_file({"input_ids": l["input_ids"][:4].contiguous(), "real_lengths": l["real_lengths"][:4].contiguous()}, f"{out}/p2_L32768.safetensors")
print("p2 corpora: 64 x 4096 from text-00000, 4 x 32768 from long-00000")
PY
cache_for() {  # cache_for <tag> <corpus> <mb> <rows_per_shard>
  local log=$OUT/cache-$1.log
  $PY $D/qtext_teacher_cache.py build-text --model-dir $M08 --device cuda:0 --dtype $DTYPE --teacher p2-q35-08b \
    --corpus $2 --out-root $SCR/cache --micro-batch $3 --rows-per-shard $4 --head-precision model --teacher-digest stat \
    --purpose "P2 v133: scratch Qwen3.5-0.8B targets for $1 (content irrelevant at lr=0)" > $log 2>&1 || return 1
  grep -o "SEALED [^ ]*manifest.json" $log | tail -1 | sed 's/^SEALED //; s#/manifest.json##'
}
card_clear "p2:target-cache"
C4=$(cache_for L4096 $SCR/p2_L4096.safetensors 8 64) || refuse "scratch target cache L4096 failed (see $OUT/cache-L4096.log)"
C32=$(cache_for L32768 $SCR/p2_L32768.safetensors 1 4) || refuse "scratch target cache L32768 failed"
echo "{\"sources\": [{\"name\": \"p2-L4096\", \"kind\": \"text\", \"qtext_cache\": \"$C4\"}]}" > $SCR/data_L4096.json
echo "{\"sources\": [{\"name\": \"p2-L32768\", \"kind\": \"text\", \"qtext_cache\": \"$C32\"}]}" > $SCR/data_L32768.json

cell() {  # cell <tag> <data> <seq> <mb> <acc> <steps> <ffn> <head_mib>
  local tag=$1 data=$2 seq=$3 mb=$4 acc=$5 steps=$6 ffn=$7 head=$8 log=$OUT/$1.log rc f prev=$OUT/prev/$(date -u +%Y%m%dT%H%M%SZ)
  card_clear "p2:cell $tag"   # the card can change hands between cells (a D1 attempt, the gpu6 slot withdrawn)
  for f in $log $OUT/$tag.rc $OUT/$tag.time $OUT/p2-trainer-v133-omni1-$tag-gpu$GPU_INDEX.json; do   # no stale evidence
    [ -e $f ] && { mkdir -p $prev; mv $f $prev/; }
  done
  echo "== $tag $(stamp) ffn $ffn head ${head} MiB" | tee -a $log
  /usr/bin/time -v -o $OUT/$tag.time $PY $T train --engine streamed --p1-chimera $OMNI1 --p1-parent-config $M27 \
    --devices cuda:0 --dtype $DTYPE --kernels auto --data-config $data --seq-len $seq --micro-batch $mb \
    --accumulation $acc --total-steps $steps --lr 0 --gain-lr 0 --warmup-steps 0 --safety-minutes 100000 --log-every 1 \
    --ffn-chunk-tokens $ffn --recompute-tokens 4096 --head-chunk-mib $head \
    --out $SCR/run_$tag --safety-dir $SCR/safety_$tag --receipt $OUT/p2-trainer-v133-omni1-$tag-gpu$GPU_INDEX.json >> $log 2>&1
  rc=$?
  echo $rc > $OUT/$tag.rc   # the attempt RAN: p2_v133_summary.py reports it even without a receipt
  rm -rf $SCR/run_$tag $SCR/safety_$tag
  echo "$tag rc=$rc $(grep -E '^(TRAIN|MEMORY|OOM) ' $log | tail -2 | cut -c1-600)"
  return $rc
}
run_cell() {  # run_cell <tag> <data> <seq> <mb> <acc> <steps>
  cell $1 $2 $3 $4 $5 $6 4096 256
  local rc=$?
  if [ $rc -ne 0 ] && grep -q '^OOM ' $OUT/$1.log; then
    status "lane=$LANE job=p2-v133 cell=$1 state=oom_retry $(grep '^OOM ' $OUT/$1.log | tail -1 | cut -c1-300)"
    cell $1-ffn2048 $2 $3 $4 $5 $6 2048 128; rc=$?
  fi
  status "lane=$LANE job=p2-v133 cell=$1 state=$([ $rc -eq 0 ] && echo done || echo failed) rc=$rc"
  return $rc
}
FAILED=0
CELLS_BEGUN=1   # from here on the EXIT trap writes the summary, whatever ends the driver
run_cell L4096-mb8 $SCR/data_L4096.json 4096 8 1 4 || FAILED=$((FAILED + 1))
run_cell L4096-mb4 $SCR/data_L4096.json 4096 4 2 4 || FAILED=$((FAILED + 1))
run_cell L32768-mb1 $SCR/data_L32768.json 32768 1 1 3 || FAILED=$((FAILED + 1))
rm -rf $SCR/cache
[ $FAILED -eq 0 ] || exit 1   # the summary (EXIT trap) names each failed cell and its phase
