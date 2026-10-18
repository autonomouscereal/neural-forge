#!/usr/bin/env bash
# run_media_a_wave2.sh -- lane media-a, stage_c_media_a 1.0.2 (Claude, 2026-10-13): S0 P4 probes and the stage-C
# H3 / Qwen-Image teacher-cache jobs, one detached docker job per call, each under a thermal watchdog.
#
#   run_media_a_wave2.sh <step> <gpu index> [extra tool args...]
#
# Steps (design §3 card plan: H3 on GPU6 after K3, QIMG on GPU4; P40s 1-3 for anything that fits 24 GB):
#   p4-h3-bf16        H3 bf16 DiT, full 2 s 320x512 shape, 5 forwards              GPU6 (40.2 GB weights) - needs the ruling below
#   p4-h3-fp16        same, --weight-dtype float16 (fp16 weight STORAGE only)       GPU6 - needs the ruling below
#   p4-h3-int8        H3 int8 ConvRot DiT, full shape                               a 3090 or a P40 (loadability check)
#   p4-h3-te          H3 Qwen3-VL-32B text encoder (nvfp4_awq; pass TE=... for bf16) any card that holds it
#   p4-h3-vae         H3 video VAE encode of a 56-frame 320x512 clip                 any
#   p4-qimg-fp8       Qwen-Image FP8 E4M3 DiT at 512^2 (manual_cast)                 GPU4 (3090), GPU6 fallback
#   p4-qimg-te        qwen_2.5_vl_7b_fp8_scaled                                      any
#   c-h3-encode       stage C: curated clips -> H3 VAE latents                       any (VAE 5.8 GB)
#   c-h3-text         stage C: H3 conditioning cache                                  any that holds the TE
#   c-h3-velocity     stage C: 2,048 H3 targets + block-25 pairs                     GPU6 (bf16) after K3 - needs the ruling below
#   c-qimg-encode / c-qimg-text / c-qimg-velocity                                    GPU4 / any
# Environment: ITEMS_H3, ITEMS_QIMG (item JSONL), H3_VAE_CACHE, H3_TE_CACHE, QIMG_VAE_CACHE, QIMG_TE_CACHE (sealed
# cache dirs from the earlier steps), TE (text-encoder override). Every launch refuses on a card above 2000 MiB, on
# GPU5, and when / has < 60 GB free; GPU0/4/6 are refused unless ALLOW_STEP1_CARDS=1 (they belong to the step-1
# workflow until it ends).
#
# 1.0.3 (lane gpu4, 2026-10-13): the Lexar teacher-cache root (LEXAR_TEACHER_CACHE, default
# /mnt/lexar/omni-campaign/teacher_cache; wave-2 rule: all teacher caches live there) is bind-mounted read-write at the
# same path when it exists, so `--cache-root /mnt/lexar/omni-campaign/teacher_cache/<cache>` works inside the container.
# Nothing else changed; a missing directory mounts nothing (the 1.0.2 behaviour).
#
# Thermal (minimax-h3-video skill: a native H3 run on GPU 6 approached that card's thermal shutdown margin and was
# aborted; a retry needs a separate thermal remediation decision): every H3 step on GPU6 is REFUSED unless
# H3_GPU6_THERMAL_RULING=<ruling id> names that decision. Every launch reads the card's `nvidia-smi -q -d
# TEMPERATURE` thresholds (current, slowdown, max operating, shutdown) into the allocator receipt, refuses a card
# already within 5 C of the stop line, and runs a watchdog beside the container that `docker stop`s it when the
# card reaches THERMAL_STOP_C (default: the max operating temperature; else slowdown - 3; else shutdown - 6; never
# at or above shutdown), polling every WATCHDOG_INTERVAL s (default 10). The end receipt records the peak temperature and whether the watchdog fired. Fan, power
# and clock policy are never touched.
set -euo pipefail
VERSION="run_media_a_wave2.sh 1.0.3"
STEP="${1:?step}"; GPU="${2:?gpu index}"; shift 2
OC="${OC:-/home/cereal/omni-campaign}"  # overridable for test_run_media_a_wave2.py only
IMG=comfyui:codex-minimax-h3-runtime-env-r5-20260808
TOOLS=$OC/distill/tools
TOOL=$TOOLS/stage_c_media_a.py
M=/opt/ComfyUI/models
LEXAR_CACHE="${LEXAR_TEACHER_CACHE:-/mnt/lexar/omni-campaign/teacher_cache}"
LEXAR_MOUNT=""
[ -d "$LEXAR_CACHE" ] && LEXAR_MOUNT="-v $LEXAR_CACHE:$LEXAR_CACHE"
TPS=$OC/distill/tps
mkdir -p "$TPS" "$OC/distill/logs" "$OC/composite/receipts/allocator"
[ "$GPU" = 5 ] && { echo "refuse: GPU5 is reserved for another project"; exit 2; }
case "$GPU" in 0|4|6) [ "${ALLOW_STEP1_CARDS:-0}" = 1 ] || { echo "refuse: GPU$GPU belongs to step 1 (set ALLOW_STEP1_CARDS=1 once step 1 has ended)"; exit 2; };; esac
RULING="${H3_GPU6_THERMAL_RULING:-}"
case "$STEP" in
  p4-h3-*|c-h3-*)
    if [ "$GPU" = 6 ] && [ -z "$RULING" ]; then
      echo "refuse: H3 on GPU6 is denied after its native thermal abort (minimax-h3-video skill); set H3_GPU6_THERMAL_RULING=<ruling id> once a thermal remediation decision exists, or use a 3090 / P40 with the int8 DiT"
      exit 2
    fi;;
esac
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU")
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "$USED" -le 2000 ] || { echo "refuse: GPU$GPU has $USED MiB in use"; exit 2; }
[ "$FREE_GB" -ge 60 ] || { echo "refuse: root has ${FREE_GB} GB free (< 60)"; exit 2; }
tq() { nvidia-smi -q -d TEMPERATURE -i "$GPU" | awk -F: -v k="$1" '$1 ~ k && !seen {v=$2; gsub(/[^0-9]/, "", v); print v; seen=1}'; }
T_CUR=$(tq "GPU Current Temp"); T_SLOW=$(tq "GPU Slowdown Temp"); T_MAXOP=$(tq "GPU Max Operating Temp"); T_SHUT=$(tq "GPU Shutdown Temp")
if [ -n "${THERMAL_STOP_C:-}" ]; then STOP_C=$THERMAL_STOP_C; STOP_FROM=THERMAL_STOP_C
elif [ -n "$T_MAXOP" ]; then STOP_C=$T_MAXOP; STOP_FROM=max_operating
elif [ -n "$T_SLOW" ]; then STOP_C=$((T_SLOW - 3)); STOP_FROM=slowdown-3
elif [ -n "$T_SHUT" ]; then STOP_C=$((T_SHUT - 6)); STOP_FROM=shutdown-6
else echo "refuse: GPU$GPU reports no temperature thresholds; set THERMAL_STOP_C"; exit 2; fi
[ -n "$T_CUR" ] || { echo "refuse: GPU$GPU current temperature unreadable"; exit 2; }
if [ -n "$T_SHUT" ] && [ "$STOP_C" -ge "$T_SHUT" ]; then echo "refuse: stop line ${STOP_C} C is not below shutdown ${T_SHUT} C"; exit 2; fi
[ "$T_CUR" -lt $((STOP_C - 5)) ] || { echo "refuse: GPU$GPU is at ${T_CUR} C, within 5 C of the ${STOP_C} C stop line"; exit 2; }
CARD=$(echo "$NAME" | tr -dc 'A-Za-z0-9' | tr 'A-Z' 'a-z')
TAG="${STEP}-gpu${GPU}-${CARD}"
P4=(probe --out "$TPS/p4_${TAG}.json" --label "$TAG" --forwards 5 --expect-device-name "$NAME")
case "$STEP" in
  p4-h3-bf16)    ARGS=("${P4[@]}" --teacher h3 --component dit --path $M/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors --weight-dtype default);;
  p4-h3-fp16)    ARGS=("${P4[@]}" --teacher h3 --component dit --path $M/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors --weight-dtype float16);;
  p4-h3-int8)    ARGS=("${P4[@]}" --teacher h3 --component dit --path $M/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors);;
  p4-h3-te)      ARGS=("${P4[@]}" --teacher h3 --component te --te-load-device gpu --path "${TE:-$M/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}");;
  p4-h3-vae)     ARGS=("${P4[@]}" --teacher h3 --component vae --path $M/vae/minimax_h3_video_vae_fp16.safetensors --forwards 2);;
  p4-qimg-fp8)   ARGS=("${P4[@]}" --teacher qimg --component dit --path $M/diffusion_models/qwen_image_fp8_e4m3fn.safetensors --width 512 --height 512 --text-len 64);;
  p4-qimg-te)    ARGS=("${P4[@]}" --teacher qimg --component te --te-load-device gpu --path $M/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors);;
  c-h3-encode)   ARGS=(h3-encode --items "${ITEMS_H3:?}" --expect-device-name "$NAME");;
  c-h3-text)     ARGS=(h3-text --items "${ITEMS_H3:?}" --te "${TE:-$M/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}" --te-load-device gpu --expect-device-name "$NAME");;
  c-h3-velocity) ARGS=(h3-velocity --items "${ITEMS_H3:?}" --encode-cache "${H3_VAE_CACHE:?}" --te-cache "${H3_TE_CACHE:?}" --expect-device-name "$NAME");;
  c-qimg-encode) ARGS=(qimg-encode --items "${ITEMS_QIMG:?}" --expect-device-name "$NAME");;
  c-qimg-text)   ARGS=(qimg-text --items "${ITEMS_QIMG:?}" --te-load-device gpu --expect-device-name "$NAME");;
  c-qimg-velocity) ARGS=(qimg-velocity --items "${ITEMS_QIMG:?}" --encode-cache "${QIMG_VAE_CACHE:?}" --te-cache "${QIMG_TE_CACHE:?}" --expect-device-name "$NAME");;
  *) echo "unknown step $STEP"; exit 2;;
esac
ARGS+=("$@")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$OC/distill/logs/media-a-${TAG}-${STAMP}.log
JOB=$OC/distill/logs/media-a-${TAG}-${STAMP}.job.sh
ALLOC=$OC/composite/receipts/allocator/media-a-${TAG}-claude.json
CNAME=media_a_${STEP//-/_}_gpu${GPU}
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
THERMAL_JSON=$(printf '{"current_c":"%s","slowdown_c":"%s","max_operating_c":"%s","shutdown_c":"%s","stop_c":%s,"stop_from":"%s","h3_gpu6_thermal_ruling":"%s"}' \
  "$T_CUR" "$T_SLOW" "$T_MAXOP" "$T_SHUT" "$STOP_C" "$STOP_FROM" "$RULING")
printf '{"lane":"media-a %s (stage_c_media_a 1.0.2)","driver":"%s","gpu":%s,"uuid":"%s","card":"%s","by":"claude media-a lane","started_utc":"%s","used_mib_at_start":%s,"thermal":%s,"log":"%s","job":"%s","state":"running"}\n' \
  "$STEP" "$VERSION" "$GPU" "$UUID" "$NAME" "$STARTED" "$USED" "$THERMAL_JSON" "$LOG" "$JOB" > "$ALLOC"
QUOTED=$(printf '%q ' "${ARGS[@]}")
# same-path mounts: $OC and the Lexar teacher-cache root read-write; /home/cereal and the NAS read-only (curated sidecars live there)
cat > "$JOB" <<EOF
#!/usr/bin/env bash
# generated by $VERSION at $STARTED: $STEP on GPU$GPU ($NAME $UUID), thermal stop ${STOP_C} C ($STOP_FROM)
docker run --rm --name $CNAME --user $(id -u):$(id -g) --cpus 16 \\
  --gpus '"device=${UUID}"' -e NVIDIA_VISIBLE_DEVICES=${UUID} -e CUDA_VISIBLE_DEVICES=0 \\
  -e HOME=/tmp -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 -e OC=$OC -w $TOOLS \\
  -v /home/cereal/comfyui/models:/opt/ComfyUI/models:ro -v /home/cereal:/home/cereal:ro -v /mnt/ai_data:/mnt/ai_data:ro \\
  -v $OC:$OC $LEXAR_MOUNT --entrypoint nice $IMG -n 15 /opt/environments/python/comfyui/bin/python $TOOL $QUOTED &
DPID=\$!
PEAK=$T_CUR
FIRED=0
while kill -0 \$DPID 2>/dev/null; do
  T=\$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i $GPU | tr -dc 0-9)
  if [ -n "\$T" ]; then
    [ "\$T" -gt "\$PEAK" ] && PEAK=\$T
    if [ "\$T" -ge $STOP_C ] && [ "\$FIRED" = 0 ]; then
      echo "[thermal watchdog] \$(date -u +%FT%TZ) GPU$GPU at \$T C >= $STOP_C C: docker stop $CNAME"
      FIRED=1
      docker stop -t 20 $CNAME || docker kill $CNAME || true
    fi
  fi
  sleep ${WATCHDOG_INTERVAL:-10}
done
wait \$DPID
RC=\$?
printf '{"lane":"media-a %s (stage_c_media_a 1.0.2)","driver":"%s","gpu":%s,"uuid":"%s","card":"%s","by":"claude media-a lane","started_utc":"%s","ended_utc":"%s","exit":%s,"thermal":%s,"peak_temp_c":%s,"thermal_stop_fired":%s,"log":"%s","job":"%s","state":"ended"}\n' \\
  "$STEP" "$VERSION" "$GPU" "$UUID" "$NAME" "$STARTED" "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$RC" '$THERMAL_JSON' "\$PEAK" "\$FIRED" "$LOG" "$JOB" > "$ALLOC"
EOF
chmod +x "$JOB"
setsid nohup bash "$JOB" > "$LOG" 2>&1 < /dev/null &
echo "launched $STEP on GPU$GPU ($NAME $UUID), thermal stop ${STOP_C} C ($STOP_FROM, now ${T_CUR} C): log $LOG job $JOB allocator $ALLOC"
