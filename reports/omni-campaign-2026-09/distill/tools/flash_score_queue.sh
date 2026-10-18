#!/bin/bash
# flash_score_queue.sh -- Qwen3.8-Flash-Next teacher logits (top-64 + lse per position) over the Chimera Omni text
# corpora, written as qtext-format caches the full-parameter KD trainer reads as ordinary text sources.
# Cyril 2026-10-17: "[Cyril quote withheld]".
#   setsid nohup bash flash_score_queue.sh > $W/queue.log 2>&1 < /dev/null &
# Tools: gguf_teacher_cache.py 1.0.x + teacher-score (teacher_score.cpp over the export lane's libllama b96806d9).
# Parity 2026-10-18 (Qwen3.5-0.8B f32 GGUF vs qtext HF fp32, 3,188 positions, P40): top-1 100 %, top-64 overlap
# 99.98 %, |dlse| max 0.0026, KL mean 1.8e-4 -- receipts in /mnt/lexar/omni-campaign/distill/fullkd/scorer-parity.
# Waits for: the GGUF verified (OK_MARK holds its sha256; written by flash_stage_chain.sh, or here: once $GGUF exists and
# nothing has it open, it is hashed and the mark written on a match, STOP on a mismatch), flash_toolcall_gen.sh to have released the P40s (it runs
# first: the tool-call dataset is the explicit ask), then holds gpu1/2/3/5 locks and scores the groups in order:
#   G0 the Flash Next tool-call export (if flash_lane.sh exported one), G1 the four K3 pass corpora + tool-bfcl-native (~2.4M tokens), G2 text_clean_v1 (~12M), G3 the registry text (~20M).
# One scorer load and one teacher digest pair per group; every group is resumable shard by shard.
# Caches go to Lexar beside the Q38 caches ($OUT; about 400 B per scored position, ~14 GB for G1-G3); the root disk is full
# (24 GB, 2026-10-18).  Each group starts only with OUT_MIN_GB free on $OUT's filesystem (215: the K3 queue keeps >= 200).
# 2026-10-18 11:40Z: GGUF / OK_MARK / OUT are overridable (the NAS copy of the GGUF is unreadable while the NAS is hung).
set -u
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
T=$OC/distill/tools
PY=/home/cereal/model-surgery-lab/venv/bin/python
GGUF=${GGUF:-/home/cereal/models/gguf/Qwen3.8-Flash-Next-Strata/RVN-Qwen3.8-Flash-Next-IQ3_S-merged.gguf}
SHA=cb5958e5b0e97804fa2973f7ce36943a1b0d735b1b83fc1e52b8c3f3b97f87e6
OK_MARK=${OK_MARK:-/home/cereal/cleanup-20261017/flash_root.sha256.ok}
TEACHER=${TEACHER:-flash-next-strata-iq3s}
OUT=${OUT:-/mnt/lexar/omni-campaign/teacher_cache/qtext}
W=${W:-$OC/distill/flash_score}
CTX=${CTX:-8192}
OUT_MIN_GB=${OUT_MIN_GB:-215}
EXTRA=${EXTRA:-}   # more teacher-score args, e.g. "--cpu-tensor per_layer_token_embd"
STATUS=$W/status
mkdir -p "$W"
export PYTHONPATH=$T CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src OMP_NUM_THREADS=8
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) $*" | tee -a "$STATUS"; }
out_gb() { mkdir -p "$OUT"; df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9; }
C=$OC/distill/corpus
G0=()
[ -s "$C/toolcall_flash_v1/toolcall-flash-v1.safetensors" ] && G0=("$C/toolcall_flash_v1/toolcall-flash-v1.safetensors")
G1=("$C"/k3/pass-0/qwen_input_ids.safetensors "$C"/k3/pass-1/qwen_input_ids.safetensors
    "$C"/k3/pass-2/qwen_input_ids.safetensors "$C"/k3/pass-3/qwen_input_ids.safetensors
    "$C"/bfcl_native_v1/bfcl-native-500.safetensors)
G2=("$C"/text_clean_v1/text-clean-*.safetensors)
G3=("$C"/text/text-*.safetensors)
CLAIMED=0
release() {
  [ "$CLAIMED" = 1 ] && echo "$(utc) RELEASE GPU1,GPU2,GPU3,GPU5 owner=claude lane=flash-score job=flash-score-queue released ($1)" >> "$Q/p40.claims"
  CLAIMED=0
}
trap 'release "TERM"; say "stopped by TERM"; exit 75' TERM INT
trap 'release "exit"' EXIT

say "queue start: teacher $TEACHER, out $OUT, gguf_teacher_cache $(sha256sum $T/gguf_teacher_cache.py | cut -c1-16), teacher-score $(sha256sum $OC/export/llamacpp/bin/teacher-score | cut -c1-16)"
until [ -s "$OK_MARK" ] && grep -q "$SHA" "$OK_MARK"; do
  if [ -s "$GGUF" ] && ! lsof "$GGUF" > /dev/null 2>&1; then
    say "hashing $GGUF"
    got=$(nice -n 10 ionice -c3 sha256sum "$GGUF" | cut -d' ' -f1)
    [ "$got" = "$SHA" ] || { say "STOP: $GGUF sha256 $got != $SHA"; exit 4; }
    echo "$got  $GGUF  verified $(utc)" > "$OK_MARK"; say "GGUF verified: $GGUF"; continue
  fi
  say "waiting: GGUF not verified yet ($GGUF, mark $OK_MARK)"; sleep 600
done
while pgrep -f flash_toolcall_gen.sh > /dev/null; do say "waiting: flash_toolcall_gen.sh holds the P40s"; sleep 600; done
for g in 1 2 3 5; do
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -dc 0-9)" -ge 2000 ]; do say "waiting: GPU$g busy"; sleep 120; done
done
exec 6>> "$Q/gpu1.lock" 7>> "$Q/gpu2.lock" 8>> "$Q/gpu3.lock" 4>> "$Q/gpu5.lock"
flock -w 3600 6 && flock -w 3600 7 && flock -w 3600 8 && flock -w 3600 4 || { say "STOP: a gpu1/2/3/5 lock is held"; exit 5; }
echo "$(utc) CLAIM GPU1,GPU2,GPU3,GPU5 owner=claude lane=flash-score job=flash-score-queue pid=$$ log=$W/queue.log preempt: kill -TERM $$ (shards are resumable)" >> "$Q/p40.claims"
CLAIMED=1
n=0
for group in G0 G1 G2 G3; do
  n=$((n + 1))
  eval "files=(\"\${$group[@]}\")"
  [ ${#files[@]} -gt 0 ] || { say "group $group: empty, skipped"; continue; }
  [ "$(out_gb)" -ge "$OUT_MIN_GB" ] || { say "STOP: $OUT has $(out_gb) GB free < $OUT_MIN_GB"; exit 6; }
  args=()
  for f in "${files[@]}"; do args+=(--corpus "$f"); done
  say "group $group: ${#files[@]} corpora"
  # shellcheck disable=SC2086
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,3,5 nice -n 5 "$PY" "$T/gguf_teacher_cache.py" build \
    --gguf "$GGUF" --gguf-sha256 "$SHA" --teacher "$TEACHER" --out-root "$OUT" "${args[@]}" \
    --rows-per-shard 32 --teacher-digest full --scorer-log "$W/scorer-$group.log" \
    --scorer-arg=-ngl --scorer-arg=999 --scorer-arg=-sm --scorer-arg=layer --scorer-arg=-ts --scorer-arg=1,1,1,1 \
    --scorer-arg=-c --scorer-arg="$CTX" --scorer-arg=-b --scorer-arg=1024 --scorer-arg=-ub --scorer-arg=512 \
    --scorer-arg=-t --scorer-arg=16 --scorer-arg=--post-threads --scorer-arg=16 \
    $(for e in $EXTRA; do printf -- "--scorer-arg=%s " "$e"; done) > "$W/build-$group.log" 2>&1
  rc=$?
  say "group $group exit $rc: $(grep -E 'SEALED|REFUSED|FAILURE|PARTIAL' "$W/build-$group.log" | tail -3 | tr '\n' ' ' | cut -c1-400)"
  [ $rc -eq 0 ] || { say "STOP: group $group failed (see $W/build-$group.log)"; exit "$rc"; }
done
say "DONE: every group sealed under $OUT/$TEACHER"
