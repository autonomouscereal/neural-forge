#!/usr/bin/env bash
# run_teacher_smoke_granite.sh -- rerun of the two Granite steps with speech_teacher_smoke.py 1.0.1 (1.0.0 sent the
# base load to the hub offline; see the tool's 1.0.1 note).  Same env, gates and inputs as run_teacher_smoke.sh.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
TH=${SMOKE_THREADS:-16}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/speech/speech_teacher_smoke.py
S=$OC/distill/speech/teachers/smoke
P=/mnt/lexar/omni-campaign/runtimes/speech-asr/bin/python
fails=0
for spec in "asr-granite:librispeech_smoke4.json:asr-granite.json" "rt-asr-granite:roundtrip_manifest.json:roundtrip-asr-granite.json"; do
  IFS=: read -r name man out <<< "$spec"
  kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  [ "$kb" -lt $((40 * 1024 * 1024)) ] && { echo "MemAvailable < 40 GiB: stop"; exit 3; }
  [ -f "$S/$name.log" ] && mv "$S/$name.log" "$S/$name.v100-failed.log"
  echo "=== $(date -u +%FT%TZ) $name (1.0.1)"
  timeout 3h nice -n 15 $P -I "$T" asr --model granite-3.3-2b --manifest "$S/$man" --out "$S/$out" --threads $TH > "$S/$name.log" 2>&1
  rc=$?
  grep -E '^\[|REF:|HYP:|pooled' "$S/$name.log" | tail -60
  echo "=== rc=$rc $name"
  [ $rc -ne 0 ] && { fails=$((fails + 1)); tail -15 "$S/$name.log"; }
done
echo "=== GRANITE RERUN DONE $(date -u +%FT%TZ) failed_steps=$fails"
echo $fails > "$S/exitcode.granite"
