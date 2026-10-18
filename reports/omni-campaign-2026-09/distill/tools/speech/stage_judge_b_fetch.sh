#!/usr/bin/env bash
# stage_judge_b_fetch.sh -- teacher-fix order of 2026-10-15 (a): admission judge B becomes ibm-granite/granite-speech-4.1-2b
# (Apache-2.0, ungated, created 2026-04-16, main = de575db6 on 2026-10-15), plus LibriSpeech dev-clean for its CPU smoke.
# Claude, 2026-10-15.  Network + disk only; no GPU, no model code.  Same receipt dir and proof rules as
# stage_teachers_fetch.sh (speech_fetch.py: every file proven against the HF API listing of the pinned revision).
set -uo pipefail
OC=/home/cereal/omni-campaign
T=$OC/distill/tools/speech
R=$OC/distill/speech/teachers/fetch
mkdir -p "$R"
PY=/usr/bin/python3
rc_all=0

echo "=== $(date -u +%FT%TZ) ibm-granite/granite-speech-4.1-2b@de575db6"
timeout 6h nice -n 10 $PY "$T/speech_fetch.py" fetch --repo ibm-granite/granite-speech-4.1-2b \
    --revision de575db64086f84fdc79da4932d1076e965bc546 --receipt-dir "$R" --workers 4
rc=$?; echo "=== rc=$rc granite-speech-4.1-2b"; [ $rc -ne 0 ] && rc_all=1

# LibriSpeech dev-clean (openslr 12, CC BY 4.0) -- md5 from the openslr md5sum.txt fetched now
L=/mnt/lexar/omni-campaign/corpus/external/librispeech
mkdir -p "$L"
echo "=== $(date -u +%FT%TZ) librispeech dev-clean"
curl -sS --max-time 60 -o "$L/md5sum.txt" https://www.openslr.org/resources/12/md5sum.txt
if [ ! -f "$L/dev-clean.tar.gz" ] || ! (cd "$L" && grep ' dev-clean.tar.gz$' md5sum.txt | md5sum -c --status); then
  curl -sS -L --retry 5 --max-time 3600 -C - -o "$L/dev-clean.tar.gz" https://www.openslr.org/resources/12/dev-clean.tar.gz
fi
if (cd "$L" && grep ' dev-clean.tar.gz$' md5sum.txt | md5sum -c); then
  [ -d "$L/LibriSpeech/dev-clean" ] || tar -xzf "$L/dev-clean.tar.gz" -C "$L"
  $PY - "$L" "$R" <<'PYEOF'
import hashlib, json, sys, pathlib, datetime
L, R = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
tgz = L / "dev-clean.tar.gz"
sha = hashlib.sha256(tgz.read_bytes()).hexdigest()
md5 = [l.split()[0] for l in (L / "md5sum.txt").read_text().splitlines() if l.endswith(" dev-clean.tar.gz")][0]
flacs = sorted((L / "LibriSpeech/dev-clean").rglob("*.flac"))
spk = {p.parts[-3] for p in flacs}
json.dump({"schema": "omni-campaign.corpus-fetch-receipt.v1", "name": "librispeech/dev-clean",
           "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz", "licence": "CC BY 4.0 (openslr 12)",
           "consent": "third_party_licensed", "bytes": tgz.stat().st_size, "sha256": sha, "md5_openslr": md5,
           "md5_verified": True, "flac_count": len(flacs), "speakers": len(spk),
           "extracted": str(L / "LibriSpeech/dev-clean"),
           "finished_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
          open(R / "librispeech__dev-clean.json", "w"), indent=1)
print("librispeech dev-clean", tgz.stat().st_size, "bytes,", len(flacs), "flacs,", len(spk), "speakers")
PYEOF
else
  echo "librispeech dev-clean md5 FAILED"; rc_all=1
fi
echo "=== ALL DONE $(date -u +%FT%TZ) rc_all=$rc_all"
echo $rc_all > "$R/exitcode.judge_b"
