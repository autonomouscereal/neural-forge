#!/usr/bin/env bash
# stage_teachers_fetch.sh -- SPEECH_LANE_PLAN_v2 decision 4 downloads #1-5 and #7 (root-parent copies by seed), plus
# LibriSpeech test-clean for the teacher smoke.  Claude, 2026-10-15.  Network + disk only; no GPU, no model code.
set -uo pipefail
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/speech
R=$OC/distill/speech/teachers/fetch
mkdir -p "$R"
PY=/usr/bin/python3
CODEX1P7=$OC/distill/codex-speech-1p7-comparison-modality-v1/parents
CODEX06=$OC/distill/codex-preparation-01a1326f-20261013/speech-teachers-v1
rc_all=0
TTS06_REV=$($PY -c "import json,sys;print([r['revision'] for r in json.load(open(sys.argv[1]))['rows'] if r.get('model_id')=='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'][0])" "$CODEX06/upstream-pins.json")
echo "TTS-0.6B-CustomVoice rev $TTS06_REV (from Codex upstream-pins.json)"
run() {  # repo revision [seed]
  local seed=()
  [ -n "${3:-}" ] && seed=(--seed "$3")
  echo "=== $(date -u +%FT%TZ) $1@$2"
  timeout 6h nice -n 10 $PY "$T/speech_fetch.py" fetch --repo "$1" --revision "$2" --receipt-dir "$R" --workers 4 "${seed[@]}"
  local rc=$?
  echo "=== rc=$rc $1"
  [ $rc -ne 0 ] && rc_all=1
}

# LibriSpeech test-clean (openslr 12, CC BY 4.0) -- md5 from the openslr md5sum.txt fetched now
L=/mnt/lexar/omni-campaign/corpus/external/librispeech
mkdir -p "$L"
echo "=== $(date -u +%FT%TZ) librispeech test-clean"
curl -sS --max-time 60 -o "$L/md5sum.txt" https://www.openslr.org/resources/12/md5sum.txt
if [ ! -f "$L/test-clean.tar.gz" ] || ! (cd "$L" && grep ' test-clean.tar.gz$' md5sum.txt | md5sum -c --status); then
  curl -sS -L --retry 5 --max-time 3600 -C - -o "$L/test-clean.tar.gz" https://www.openslr.org/resources/12/test-clean.tar.gz
fi
if (cd "$L" && grep ' test-clean.tar.gz$' md5sum.txt | md5sum -c); then
  [ -d "$L/LibriSpeech/test-clean" ] || tar -xzf "$L/test-clean.tar.gz" -C "$L"
  $PY - "$L" "$R" <<'PYEOF'
import hashlib, json, sys, pathlib, datetime
L, R = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
tgz = L / "test-clean.tar.gz"
sha = hashlib.sha256(tgz.read_bytes()).hexdigest()
md5 = [l.split()[0] for l in (L / "md5sum.txt").read_text().splitlines() if l.endswith(" test-clean.tar.gz")][0]
flacs = sorted((L / "LibriSpeech/test-clean").rglob("*.flac"))
spk = {p.parts[-3] for p in flacs}
json.dump({"schema": "omni-campaign.corpus-fetch-receipt.v1", "name": "librispeech/test-clean",
           "url": "https://www.openslr.org/resources/12/test-clean.tar.gz", "licence": "CC BY 4.0 (openslr 12)",
           "consent": "third_party_licensed", "bytes": tgz.stat().st_size, "sha256": sha, "md5_openslr": md5,
           "md5_verified": True, "flac_count": len(flacs), "speakers": len(spk),
           "extracted": str(L / "LibriSpeech/test-clean"),
           "finished_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
          open(R / "librispeech__test-clean.json", "w"), indent=1)
print("librispeech test-clean", tgz.stat().st_size, "bytes,", len(flacs), "flacs,", len(spk), "speakers")
PYEOF
else
  echo "librispeech test-clean md5 FAILED"; rc_all=1
fi

# #7 root parents, seeded from the Codex on-disk copies (copy + prove, download only what is missing)
run Qwen/Qwen3-ASR-1.7B 7278e1e70fe206f11671096ffdd38061171dd6e5 "$CODEX1P7/Qwen3-ASR-1.7B"
run Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice 0c0e3051f131929182e2c023b9537f8b1c68adfe "$CODEX1P7/Qwen3-TTS-12Hz-1.7B-CustomVoice"
run Qwen/Qwen3-ASR-0.6B 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 "$CODEX06/Qwen3-ASR-0.6B"
run Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice "$TTS06_REV" "$CODEX06/Qwen3-TTS-12Hz-0.6B-CustomVoice"
# #1-#5 new teachers
run Qwen/Qwen3-TTS-12Hz-1.7B-Base fd4b254389122332181a7c3db7f27e918eec64e3
run Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign 5ecdb67327fd37bb2e042aab12ff7391903235d3
run ibm-granite/granite-speech-3.3-2b 4ac2f02f413c6169ae8c0ccc217115a366e552d7
run openbmb/VoxCPM2 32279effe8c19989596f05d353d1447f51d9e915
run hexgrad/Kokoro-82M f3ff3571791e39611d31c381e3a41a3af07b4987
echo "=== ALL DONE $(date -u +%FT%TZ) rc_all=$rc_all"
echo $rc_all > "$R/exitcode"
