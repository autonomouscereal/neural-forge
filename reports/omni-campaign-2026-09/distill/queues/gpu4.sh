#!/usr/bin/env bash
# gpu4.sh -- stage-D wave-2 queue for lane gpu4 (GPU4, RTX 3090). Claude, 2026-10-13.
#
# Card rule (wave 2): GPU4 belongs to this lane only. Every GPU job first waits until step 1's S8 review (pid 917640,
# run_review_finite_guard.py) has exited AND GPU4 shows < 2000 MiB used; that wait lives HERE, server-side.
# Jobs, in order (DISTILL_DESIGN_2026-10-12 §3 QIMG rows, §4 S0 P4 + P11; STAGE_D_WAVE1_STATUS §4.2 C10, §4.3 P11):
#   0  items          CPU   corpus/media/qimg/items.jsonl -> stage_c_media_a items (media_a_corpus_items.py 1.0.0)
#   1a e2-gpu-parity  GPU4  E2 parity arm on OMNI1 at 4,096 tokens: stock layer vs E2 (fla recurrence, bf16, chunk 1024,
#                           256 MiB score budget so chunk boundaries and query sub-chunks are crossed). ~10 min; does NOT
#                           gate the QIMG jobs (it gates only P11, through gpu4.p11.ready). Added in 1.0.2.
#   1  p4-qimg-fp8    GPU4  S0 P4: Qwen-Image FP8 E4M3 storage, ComfyUI manual_cast compute, 512^2, 5 forwards
#   2  p4-qimg-te     GPU4  S0 P4: qwen_2.5_vl_7b_fp8_scaled text encoder
#   3  c-qimg-text    GPU4  stage C: 2,048 prompts + the negative -> TE conditioning cache (Lexar)
#   4  c-qimg-encode  GPU4  stage C: 2,048 images -> Qwen-Image VAE latents (Lexar); velocity needs them
#   5  c-qimg-velocity GPU4 stage C: 2,048 guided targets (1 sigma per item, cfg 4) at 512^2 (Lexar) -- ONLY if job 1
#                           ran clean (exit 0, finite) with a torch peak reservation <= 22 GiB; otherwise a handoff note
#                           to lane gpu6 is appended to handoff.md and this job is skipped (GPU6 is never touched here).
#                           A ComfyUI PARTIAL load (loaded < model size) is recorded, not a handoff: the targets are the
#                           same weights under the same manual_cast compute, only slower (1.0.1; 1.0.0 handed it off).
#   6  p11-omni1      GPU4  S0 P11: OMNI1 teacher-forced needle baseline (E2 engine); starts only when
#                           gpu4.p11.ready names gpu4_p11_job.sh and its sha256 (written after E2's parity arm passed)
# Each job writes its own log + receipts; one status line per job goes to gpu4.status. Jobs 3-5 gate on the
# previous cache job's exit code (the velocity pass needs both sealed caches).
#
# 1.0.3 (2026-10-13 09:50Z): the TE cache path was wrong in 1.0.0-1.0.2 (stage_c_media_a keeps "_" in dir names:
# qimg-te-qwen_2.5_vl_7b_fp8_scaled, not qimg-te-qwen-2.5-...); the sealed cache was fine, only this script's
# QIMG_TE_CACHE pointer was wrong, and job 5 would have failed on it. Resume: a GPU job whose "job=<name> rc=0" line
# is already in gpu4.status is skipped (P4's verdict is re-read from its receipt), and a media-a step whose allocator
# receipt says "running" is ADOPTED (waited on) instead of launched again.
set -uo pipefail
QVERSION="gpu4.sh 1.0.3"
OC=/home/cereal/omni-campaign
Q=$OC/distill/queues
STATUS=$Q/gpu4.status
QLOG=$Q/gpu4.queue.log
WORK=$Q/gpu4
TOOLS=$OC/distill/tools
DRV=$TOOLS/run_media_a_wave2.sh
GPU=4
S8PID=917640
PYLAB=/home/cereal/model-surgery-lab/venv/bin/python
LEX=/mnt/lexar/omni-campaign/teacher_cache
QCACHE=$LEX/qimg
SRC_ITEMS=$OC/distill/corpus/media/qimg/items.jsonl
ITEMS=$OC/distill/teacher_items/qimg/qimg_items.media_a.jsonl
ALLOCD=$OC/composite/receipts/allocator
P4_LIMIT_BYTES=$((22 * 1024 * 1024 * 1024))   # "22 GB" read as 22 GiB of a 24 GiB card: 2 GiB headroom
LEX_MIN_GB=15
mkdir -p "$WORK/receipts" "$ALLOCD"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$QLOG"; }
status() { echo "$(date -u +%FT%TZ) gpu4 $*" >> "$STATUS"; log "STATUS $*"; }

s8_alive() { [ -r /proc/$S8PID/cmdline ] && tr '\0' ' ' < /proc/$S8PID/cmdline | grep -q run_review_finite_guard; }
card_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU | tr -dc 0-9; }

wait_card() {  # $1 = job name; returns when S8 is gone and GPU4 < 2000 MiB
  local waited=0
  while s8_alive; do
    [ $((waited % 900)) -eq 0 ] && log "$1: waiting for step-1 S8 pid $S8PID to exit"
    sleep 60; waited=$((waited + 60))
  done
  while u=$(card_used); [ -z "$u" ] || [ "$u" -ge 2000 ]; do
    [ $((waited % 900)) -eq 0 ] && log "$1: waiting for GPU$GPU < 2000 MiB (now ${u:-?} MiB)"
    sleep 30; waited=$((waited + 30))
  done
  log "$1: card gate passed after ${waited}s (S8 gone, GPU$GPU $(card_used) MiB)"
}

wait_lexar() {  # $1 = job name; Lexar >= LEX_MIN_GB free (the omnicoder purge frees ~845 GB), at most 3 h
  local i=0 free
  while free=$(df -BG --output=avail /mnt/lexar | tail -1 | tr -dc 0-9); [ "$free" -lt $LEX_MIN_GB ]; do
    [ $i -ge 180 ] && { log "$1: Lexar still ${free} GB free after 3 h"; return 1; }
    [ $((i % 15)) -eq 0 ] && log "$1: Lexar ${free} GB free < ${LEX_MIN_GB} GB; waiting (purge log $OC/composite/receipts/disk/oct13/purge_omnicoder_weights.log)"
    sleep 60; i=$((i + 1))
  done
  log "$1: Lexar ${free} GB free, root $(df -BG --output=avail / | tail -1 | tr -dc 0-9) GB free"
}

# run one stage_c_media_a step through the shared driver and block until its allocator receipt says "ended"
# for THIS job file; echoes the exit code. The driver records the allocator receipt + thermal watchdog.
LAST_LOG=""; LAST_ALLOC=""
run_media_a() {
  local step=$1; shift
  local out rc job alloc
  out=$(ALLOW_STEP1_CARDS=1 bash "$DRV" "$step" $GPU "$@" 2>&1); rc=$?
  echo "$out" >> "$QLOG"
  if [ $rc -ne 0 ]; then log "$step: driver refused rc=$rc: $out"; return $rc; fi
  job=$(echo "$out" | sed -n 's/.* job \([^ ]*\) allocator .*/\1/p')
  alloc=$(echo "$out" | sed -n 's/.* allocator \([^ ]*\)$/\1/p')
  LAST_LOG=$(echo "$out" | sed -n 's/.*: log \([^ ]*\) job .*/\1/p'); LAST_ALLOC=$alloc
  [ -n "$job" ] && [ -n "$alloc" ] || { log "$step: could not parse the driver line"; return 90; }
  log "$step: running, log $LAST_LOG"
  wait_alloc "$alloc" "$job"
}

done_ok() { grep -q " gpu4 job=$1 rc=0 " "$STATUS"; }   # resume: this job already ended clean in an earlier queue run

alloc_field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$1" "$2" 2>/dev/null; }

# adopt a media-a step still running from an earlier queue run (its driver job is detached), else launch it
run_or_adopt() {
  local step=$1; shift
  local alloc=$ALLOCD/media-a-${step}-gpu${GPU}-nvidiageforcertx3090-claude.json
  if [ -f "$alloc" ] && [ "$(alloc_field "$alloc" state)" = running ] && pgrep -f "$(alloc_field "$alloc" job)" > /dev/null; then
    LAST_LOG=$(alloc_field "$alloc" log); LAST_ALLOC=$alloc
    log "$step: adopting the in-flight job $(alloc_field "$alloc" job) (started $(alloc_field "$alloc" started_utc)), log $LAST_LOG"
    wait_alloc "$alloc" "$(alloc_field "$alloc" job)"
    return $?
  fi
  wait_card "$step"
  run_media_a "$step" "$@"
}

wait_alloc() {  # $1 allocator receipt, $2 job file; returns the job's exit code once the receipt says ended
  local alloc=$1 job=$2 rc
  while :; do
    rc=$(python3 - "$alloc" "$job" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    print("wait"); raise SystemExit
print(r.get("exit") if r.get("state") == "ended" and r.get("job") == sys.argv[2] else "wait")
PY
)
    [ "$rc" != "wait" ] && break
    sleep 20
  done
  cp -f "$alloc" "$WORK/receipts/$(basename "$alloc" .json)-$(date -u +%Y%m%dT%H%M%SZ).json"
  return "$rc"
}

log "$QVERSION start (pid $$); S8 alive: $(s8_alive && echo yes || echo no); GPU$GPU $(card_used) MiB"
status "queue=start version=\"$QVERSION\" pid=$$"

# ---- job 0: items (CPU) --------------------------------------------------------------------------------------
nice -n 15 "$PYLAB" "$TOOLS/media_a_corpus_items.py" convert --route image --src "$SRC_ITEMS" --out "$ITEMS" \
  --check-files --hash >> "$QLOG" 2>&1
rc=$?
status "job=items rc=$rc items=$ITEMS sha256=$(sha256sum "$ITEMS" 2>/dev/null | cut -c1-64) receipt=$ITEMS.receipt.json"
[ $rc -eq 0 ] || { status "queue=stop reason=items-failed"; exit 1; }
SHA16=$(sha256sum "$ITEMS" | cut -c1-16)
D_TE=$QCACHE/qimg-te-qwen_2.5_vl_7b_fp8_scaled/$SHA16   # stage_c_media_a keeps '_' ([^a-z0-9._-] -> '-')
D_VAE=$QCACHE/qimg-vae/$SHA16
export ITEMS_QIMG=$ITEMS QIMG_VAE_CACHE=$D_VAE QIMG_TE_CACHE=$D_TE   # read by run_media_a_wave2.sh for the c-qimg-* steps

# ---- job 1a: E2 GPU parity arm (OMNI1, 4k) ---------------------------------------------------------------------
if done_ok e2-gpu-parity; then log "e2-gpu-parity: done in an earlier run, skipped"; else
wait_card e2-gpu-parity
E2OUT=$OC/distill/tps/p11_omni1_e2_parity_gpu_bf16_fla_L4096.json
E2LOG=$OC/distill/logs/gpu4/e2_gpu_parity-$(date -u +%Y%m%dT%H%M%SZ).log
E2ALLOC=$ALLOCD/gpu4-e2-gpu-parity-claude.json
STARTED=$(date -u +%FT%TZ)
printf '{"lane":"gpu4 E2 GPU parity arm (OMNI1, 4k)","queue":"%s","gpu":4,"by":"claude lane gpu4","started_utc":"%s","log":"%s","state":"running"}\n' \
  "$QVERSION" "$STARTED" "$E2LOG" > "$E2ALLOC"
( export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$TOOLS OC_CALIBRATION_DIR=$OC/calibration OMP_NUM_THREADS=16 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nice -n 15 "$PYLAB" "$TOOLS/p11_needle_engine.py" parity --model /mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1 \
    --chimera-src $OC/import/rq27b/tree_ef1ffb9 --needles $OC/distill/p11/needles_v1 --length 4096 --device cuda:0 \
    --dtype bfloat16 --kda-kernel fla --chunk 1024 --attn-budget-mib 256 --parity-min-top1 0.90 --out "$E2OUT" ) > "$E2LOG" 2>&1
rc=$?
printf '{"lane":"gpu4 E2 GPU parity arm (OMNI1, 4k)","queue":"%s","gpu":4,"by":"claude lane gpu4","started_utc":"%s","ended_utc":"%s","exit":%s,"log":"%s","state":"ended"}\n' \
  "$QVERSION" "$STARTED" "$(date -u +%FT%TZ)" "$rc" "$E2LOG" > "$E2ALLOC"
status "job=e2-gpu-parity rc=$rc line=\"$(grep -h 'E2 PARITY' "$E2LOG" | tail -1 | cut -c1-260)\" receipt=$E2OUT log=$E2LOG"
fi

# ---- job 1: P4 QIMG FP8 DiT ----------------------------------------------------------------------------------
P4FRESH=1
if done_ok p4-qimg-fp8; then rc=0; P4FRESH=0; log "p4-qimg-fp8: done in an earlier run, verdict re-read from its receipt"
else wait_card p4-qimg-fp8; run_media_a p4-qimg-fp8; rc=$?; fi
P4R=$OC/distill/tps/p4_p4-qimg-fp8-gpu4-nvidiageforcertx3090.json
VERDICT=$(python3 - "$P4R" "$rc" "$P4_LIMIT_BYTES" <<'PY'
import json, sys
path, rc, limit = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    r = json.load(open(path))
except Exception as exc:
    print(f"verdict=no-receipt rc={rc} err={type(exc).__name__}"); raise SystemExit
pk = r.get("vram_peak", {}); ld = r.get("load", {})
res, alloc = pk.get("max_reserved_bytes") or 0, pk.get("max_allocated_bytes") or 0
model, loaded = ld.get("model_size_bytes"), ld.get("loaded_size_bytes")
finite = all(f.get("finite") for f in r.get("forwards", [])) and not r.get("non_finite")
full = loaded is None or model is None or loaded >= model
fits = rc == 0 and finite and res <= limit
spf = r.get("seconds_per_forward_steady")
print(f"verdict={'fits-3090' if fits else 'handoff-gpu6'} rc={rc} s_per_forward={spf} peak_reserved_mib={res >> 20} "
      f"peak_allocated_mib={alloc >> 20} model_mib={(model or 0) >> 20} loaded_mib={(loaded or 0) >> 20} "
      f"weight={ld.get('weight_dtype')} manual_cast={ld.get('manual_cast_dtype')} finite={finite} full_load={full} limit_mib={limit >> 20}")
PY
)
[ $P4FRESH = 1 ] && status "job=p4-qimg-fp8 rc=$rc $VERDICT receipt=$P4R log=$LAST_LOG"
log "p4-qimg-fp8 verdict: $VERDICT"

# ---- job 2: P4 QIMG text encoder ------------------------------------------------------------------------------
if done_ok p4-qimg-te; then log "p4-qimg-te: done in an earlier run, skipped"; else
wait_card p4-qimg-te
run_media_a p4-qimg-te; rc=$?
P4T=$OC/distill/tps/p4_p4-qimg-te-gpu4-nvidiageforcertx3090.json
TEL=$(python3 - "$P4T" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
    pk = r.get("vram_peak", {})
    print(f"s_per_text={r.get('seconds_per_text_steady')} load_device={r.get('load', {}).get('load_device')} "
          f"model_dtype={r.get('load', {}).get('model_dtype')} peak_reserved_mib={(pk.get('max_reserved_bytes') or 0) >> 20} cond_shape={r.get('cond_shape')}")
except Exception as exc:
    print(f"no-receipt err={type(exc).__name__}")
PY
)
status "job=p4-qimg-te rc=$rc $TEL receipt=$P4T log=$LAST_LOG"
fi

# ---- job 3: QIMG text-encoder cache ---------------------------------------------------------------------------
mkdir -p "$QCACHE"
if done_ok c-qimg-text && [ -f "$D_TE/manifest.json" ]; then
  log "c-qimg-text: done in an earlier run, sealed cache $D_TE (manifest $(sha256sum $D_TE/manifest.json | cut -c1-16))"
else
  wait_lexar c-qimg-text || { status "job=c-qimg-text rc=91 reason=lexar-full"; exit 1; }
  run_or_adopt c-qimg-text --cache-root "$QCACHE"; rc=$?
  status "job=c-qimg-text rc=$rc cache=$D_TE manifest=$( [ -f $D_TE/manifest.json ] && sha256sum $D_TE/manifest.json | cut -c1-16 ) log=$LAST_LOG"
  [ $rc -eq 0 ] || { status "queue=stop reason=c-qimg-text-failed"; exit 1; }
fi
[ -f "$D_TE/manifest.json" ] || { status "queue=stop reason=te-cache-missing dir=$D_TE"; exit 1; }

# ---- job 4: QIMG VAE latents ----------------------------------------------------------------------------------
wait_lexar c-qimg-encode || { status "job=c-qimg-encode rc=91 reason=lexar-full"; exit 1; }
ENC_ALLOC=$ALLOCD/media-a-c-qimg-encode-gpu${GPU}-nvidiageforcertx3090-claude.json
if [ -f "$D_VAE/manifest.json" ] && [ "$(alloc_field "$ENC_ALLOC" state)" = ended ] && [ "$(alloc_field "$ENC_ALLOC" exit)" = 0 ]; then
  rc=0; LAST_LOG=$(alloc_field "$ENC_ALLOC" log)   # finished clean while no queue was attached (1.0.2 -> 1.0.3 restart)
  log "c-qimg-encode: ended clean before this queue attached, adopted (allocator $ENC_ALLOC)"
else
  run_or_adopt c-qimg-encode --cache-root "$QCACHE"; rc=$?
fi
status "job=c-qimg-encode rc=$rc cache=$D_VAE manifest=$( [ -f $D_VAE/manifest.json ] && sha256sum $D_VAE/manifest.json | cut -c1-16 ) log=$LAST_LOG"
[ $rc -eq 0 ] || { status "queue=stop reason=c-qimg-encode-failed"; exit 1; }

# ---- job 5: QIMG guided velocity (or the GPU6 handoff) --------------------------------------------------------
VEL_ARGS=(--encode-cache "$D_VAE" --te-cache "$D_TE" --cache-root "$QCACHE" --sigmas-per-item 1)
if echo "$VERDICT" | grep -q "verdict=fits-3090"; then
  wait_lexar c-qimg-velocity || { status "job=c-qimg-velocity rc=91 reason=lexar-full"; exit 1; }
  run_or_adopt c-qimg-velocity "${VEL_ARGS[@]}"; rc=$?
  RATE=$(grep -h "s/forward, peak" "$LAST_LOG" 2>/dev/null | tail -1)
  status "job=c-qimg-velocity rc=$rc cache_root=$QCACHE last_rate=\"$RATE\" log=$LAST_LOG"
else
  {
    echo
    echo "## $(date -u +%FT%TZ) lane gpu4 -> lane gpu6: QIMG guided velocity moved to the END of the gpu6 queue"
    echo
    echo "P4 on GPU4 (RTX 3090) said: \`$VERDICT\` (receipt \`$P4R\`). The 3090 cannot hold the FP8 DiT + activations"
    echo "within 22 GiB with the whole model resident, so per DISTILL_DESIGN §3 the velocity pass falls back to GPU6."
    echo "The TE cache and the VAE cache were built on GPU4 and are sealed; only the velocity pass moves:"
    echo
    echo '```'
    echo "ITEMS_QIMG=$ITEMS QIMG_VAE_CACHE=$D_VAE QIMG_TE_CACHE=$D_TE ALLOW_STEP1_CARDS=1 \\"
    echo "  bash $DRV c-qimg-velocity 6 ${VEL_ARGS[*]}"
    echo '```'
    echo "Lane gpu4 does not touch GPU6. 2,048 targets x 2 forwards (cfg 4)."
  } >> "$Q/handoff.md"
  status "job=c-qimg-velocity rc=skipped reason=handoff-gpu6 note=$Q/handoff.md"
fi

# ---- job 6: P11 OMNI1 needle baseline (E2) --------------------------------------------------------------------
i=0
while [ ! -f "$Q/gpu4.p11.ready" ]; do
  [ $i -ge 1440 ] && { status "job=p11-omni1 rc=92 reason=e2-not-ready-after-24h"; exit 1; }
  [ $((i % 30)) -eq 0 ] && log "p11-omni1: waiting for $Q/gpu4.p11.ready (E2 parity arm)"
  sleep 60; i=$((i + 1))
done
WANT=$(cut -d' ' -f1 "$Q/gpu4.p11.ready"); HAVE=$(sha256sum "$Q/gpu4_p11_job.sh" | cut -c1-64)
[ "$WANT" = "$HAVE" ] || { status "job=p11-omni1 rc=93 reason=job-sha-mismatch want=$WANT have=$HAVE"; exit 1; }
wait_card p11-omni1
P11ALLOC=$ALLOCD/gpu4-p11-omni1-needle-claude.json
STARTED=$(date -u +%FT%TZ)
printf '{"lane":"gpu4 P11 OMNI1 needle baseline","queue":"%s","gpu":4,"card":"%s","by":"claude lane gpu4","started_utc":"%s","job":"%s","job_sha256":"%s","state":"running"}\n' \
  "$QVERSION" "$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)" "$STARTED" "$Q/gpu4_p11_job.sh" "$HAVE" > "$P11ALLOC"
P11LOG=$OC/distill/logs/gpu4/p11-omni1-$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$(dirname "$P11LOG")"
bash "$Q/gpu4_p11_job.sh" > "$P11LOG" 2>&1; rc=$?
printf '{"lane":"gpu4 P11 OMNI1 needle baseline","queue":"%s","gpu":4,"by":"claude lane gpu4","started_utc":"%s","ended_utc":"%s","exit":%s,"job":"%s","job_sha256":"%s","log":"%s","state":"ended"}\n' \
  "$QVERSION" "$STARTED" "$(date -u +%FT%TZ)" "$rc" "$Q/gpu4_p11_job.sh" "$HAVE" "$P11LOG" > "$P11ALLOC"
status "job=p11-omni1 rc=$rc log=$P11LOG summary=\"$(grep -h 'P11 SUMMARY' "$P11LOG" | tail -1 | cut -c1-300)\""
status "queue=end"
