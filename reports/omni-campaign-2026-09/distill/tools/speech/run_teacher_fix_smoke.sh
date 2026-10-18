#!/usr/bin/env bash
# run_teacher_fix_smoke.sh -- CPU smoke for the 2026-10-15 teacher fix (Claude).  GPU1/2 are held by lane toolcall and
# GPU3 by its own claims at launch; GPU5 never; so everything here runs with CUDA_VISIBLE_DEVICES="".
#   1. two-judge smoke: 4 LibriSpeech dev-clean utterances through asr_two_judge.transcribe_two_judges (judge A
#      Qwen3-ASR-0.6B, judge B granite-speech-4.1-2b): WER of each vs the reference, the A/B agreement verdict;
#   2. family 2: Fun-CosyVoice3-0.5B-2512 (RL LLM) zero-shot from one dev-clean reference clip, the 4 fixed sentences;
#   3. round trip: the 4 CosyVoice3 wavs through both judges -> admission verdicts (both exact = admitted).
# One model process at a time; every step gated on MemAvailable >= 40 GiB.  Outputs are smoke-only (never a corpus).
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
TH=${SMOKE_THREADS:-16}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/speech
S=$OC/distill/speech/teachers/smoke-teacherfix-2026-10-15
RT=/mnt/lexar/omni-campaign/runtimes
LS=/mnt/lexar/omni-campaign/corpus/external/librispeech/LibriSpeech/dev-clean
mkdir -p "$S/tts"
fails=0
memgate() {
  local kb; kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if [ "$kb" -lt $((40 * 1024 * 1024)) ]; then echo "MemAvailable $((kb / 1024 / 1024)) GiB < 40: stop"; exit 3; fi
}
step() {  # name command...
  local name=$1; shift
  memgate
  echo "=== $(date -u +%FT%TZ) $name (MemAvailable $(awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo) GiB)"
  timeout 3h nice -n 15 "$@" > "$S/$name.log" 2>&1
  local rc=$?
  grep -v -E 'Warning|warn\(|^\s*$|it/s\]|s/it\]' "$S/$name.log" | tail -45
  echo "=== rc=$rc $name"
  [ $rc -ne 0 ] && { fails=$((fails + 1)); tail -20 "$S/$name.log"; }
}
step two-judge-devclean /usr/bin/python3 -I "$T/asr_two_judge.py" smoke --librispeech "$LS" --n 4 --out "$S/two_judge_devclean4.json" --threads $TH
/usr/bin/python3 -I "$T/speech_teacher_smoke.py" pick-librispeech --librispeech "$LS" --n 1 --out "$S/clone_ref_devclean.json"
step tts-cosyvoice3 "$RT/speech-cosyvoice3/bin/python" -I "$T/speech_teacher_smoke.py" tts --model cosyvoice3 --out-dir "$S/tts" \
     --clone-ref "$S/clone_ref_devclean.json" --cv3-llm rl --threads $TH
if [ -f "$S/tts/cosyvoice3.asr_manifest.json" ]; then
  step rt-two-judge-cosyvoice3 /usr/bin/python3 -I "$T/asr_two_judge.py" transcribe --manifest "$S/tts/cosyvoice3.asr_manifest.json" \
       --out "$S/roundtrip_two_judge_cosyvoice3.json" --threads $TH
  /usr/bin/python3 - "$S/roundtrip_two_judge_cosyvoice3.json" <<'PYEOF'
import json, sys
for r in json.load(open(sys.argv[1])):
    a = r["admission"]
    print(f"{r['id']}: admission {a['verdict']} (A WER {a['judge_a_wer']}, B WER {a['judge_b_wer']}); "
          f"A/B agreement {r['transcript_status']} {r['agreement_wer']}\n   A: {r['judge_a']['text']}\n   B: {r['judge_b']['text']}")
PYEOF
fi
echo "=== TEACHER-FIX SMOKE DONE $(date -u +%FT%TZ) failed_steps=$fails"
echo $fails > "$S/exitcode"
