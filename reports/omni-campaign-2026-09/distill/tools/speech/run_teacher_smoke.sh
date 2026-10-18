#!/usr/bin/env bash
# run_teacher_smoke.sh -- staged speech-teacher smoke, CPU (GPU3 held by lane k3accel at launch; GPU5 never).
# Claude, 2026-10-15.  One model process at a time; each step gated on MemAvailable >= 40 GiB.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
TH=${SMOKE_THREADS:-16}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/speech/speech_teacher_smoke.py
S=$OC/distill/speech/teachers/smoke
RT=/mnt/lexar/omni-campaign/runtimes
LS=/mnt/lexar/omni-campaign/corpus/external/librispeech/LibriSpeech/test-clean
mkdir -p "$S/tts"
fails=0
memgate() {
  local kb; kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$kb" -lt $((40 * 1024 * 1024)) ]; then echo "MemAvailable $((kb / 1024 / 1024)) GiB < 40: stop"; exit 3; fi
}
step() {  # name venv args...
  local name=$1 venv=$2; shift 2
  memgate
  echo "=== $(date -u +%FT%TZ) $name (MemAvailable $(awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo) GiB)"
  timeout 3h nice -n 15 "$RT/$venv/bin/python" -I "$T" "$@" > "$S/$name.log" 2>&1
  local rc=$?
  grep -v -E 'Warning|warn\(|^\s*$' "$S/$name.log" | grep -E '^\[|REF:|HYP:|pooled|Error|error' | tail -40
  echo "=== rc=$rc $name"
  [ $rc -ne 0 ] && { fails=$((fails + 1)); tail -15 "$S/$name.log"; }
}
python3 "$T" pick-librispeech --librispeech "$LS" --n 4 --out "$S/librispeech_smoke4.json"
step asr-qwen3-1.7b speech-asr asr --model qwen3-asr-1.7b --manifest "$S/librispeech_smoke4.json" --out "$S/asr-qwen3-1.7b.json" --threads $TH
step asr-qwen3-0.6b speech-asr asr --model qwen3-asr-0.6b --manifest "$S/librispeech_smoke4.json" --out "$S/asr-qwen3-0.6b.json" --threads $TH
step asr-granite speech-asr asr --model granite-3.3-2b --manifest "$S/librispeech_smoke4.json" --out "$S/asr-granite.json" --threads $TH
step tts-customvoice speech-qwentts tts --model qwen3-tts-customvoice --out-dir "$S/tts" --threads $TH
step tts-voicedesign speech-qwentts tts --model qwen3-tts-voicedesign --out-dir "$S/tts" --threads $TH
step tts-base-clone speech-qwentts tts --model qwen3-tts-base --out-dir "$S/tts" --clone-ref "$S/librispeech_smoke4.json" --threads $TH
step tts-voxcpm2 speech-voxcpm tts --model voxcpm2 --out-dir "$S/tts" --threads $TH
python3 "$T" merge-manifests "$S"/tts/*.asr_manifest.json --out "$S/roundtrip_manifest.json"
step rt-asr-qwen3-1.7b speech-asr asr --model qwen3-asr-1.7b --manifest "$S/roundtrip_manifest.json" --out "$S/roundtrip-asr-qwen3-1.7b.json" --threads $TH
step rt-asr-granite speech-asr asr --model granite-3.3-2b --manifest "$S/roundtrip_manifest.json" --out "$S/roundtrip-asr-granite.json" --threads $TH
echo "=== SMOKE DONE $(date -u +%FT%TZ) failed_steps=$fails"
echo $fails > "$S/exitcode"
