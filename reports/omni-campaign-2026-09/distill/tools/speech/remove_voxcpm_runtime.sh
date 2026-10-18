#!/usr/bin/env bash
# remove_voxcpm_runtime.sh -- teacher-fix order of 2026-10-15 (c).  The speech-voxcpm runtime installed pydantic 2.14.1
# (+ pydantic_core 2.50.1) because voxcpm 2.0.3's model code imports pydantic for its config classes.  Cyril's global
# rule: no ORM, no Pydantic, no SQLAlchemy in any code or environment we build.  This removes the runtime (ours, not
# data), purges the shared pip cache that holds the pydantic wheels, and marks the model directories:
#   openbmb/VoxCPM2              weights KEPT on disk, marked unusable under the rule (its loader needs pydantic)
#   ibm-granite/granite-speech-3.3-2b   weights KEPT on disk, marked superseded by granite-speech-4.1-2b (judge B)
# Evidence first (freeze, sizes, counts) to $OC/distill/speech/teachers/runtime/speech-voxcpm.REMOVED-2026-10-15.json;
# the build recipe stays in build_speech_runtimes.sh 1.0.0 (tools/speech/_superseded/), so the env is reproducible.
# Claude, 2026-10-15.  CPU/disk only.
set -uo pipefail
RT=/mnt/lexar/omni-campaign/runtimes
V=$RT/speech-voxcpm
OUT=/home/cereal/omni-campaign/distill/speech/teachers/runtime
EV=$OUT/speech-voxcpm.REMOVED-2026-10-15.json
M=/mnt/lexar/models
if pgrep -f "$V/bin/python" > /dev/null; then echo "a process runs from $V: refusing"; exit 3; fi
if [ -d "$V" ]; then
  "$V/bin/python" -m pip freeze > "$OUT/speech-voxcpm.freeze.at-removal.txt" 2>/dev/null
  files=$(find "$V" -type f | wc -l)
  bytes=$(du -sb --apparent-size "$V" | cut -f1)
  /usr/bin/python3 - "$EV" "$V" "$files" "$bytes" "$OUT/speech-voxcpm.freeze.at-removal.txt" <<'PYEOF'
import datetime, hashlib, json, sys
ev, v, files, size, freeze = sys.argv[1:]
text = open(freeze).read()
json.dump({"schema": "omni-campaign.runtime-removal.v1", "runtime": v, "files": int(files), "bytes": int(size),
           "reason": "pydantic 2.14.1 + pydantic_core 2.50.1 installed for voxcpm 2.0.3 (its model code imports pydantic); "
                     "Cyril's global rule forbids ORM / Pydantic / SQLAlchemy in any environment we build",
           "pydantic_lines": [l for l in text.splitlines() if l.lower().startswith(("pydantic", "sqlalchemy"))],
           "freeze_file": freeze, "freeze_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "rebuild_recipe": "tools/speech/_superseded/build_speech_runtimes_1.0.0.sh (speech-voxcpm section) -- do not rebuild",
           "model_weights_kept": "/mnt/lexar/models/openbmb/VoxCPM2 (marked _STATUS_UNUSABLE_NO_PYDANTIC_RULE.txt)",
           "removed_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
          open(ev, "w"), indent=1)
print(open(ev).read())
PYEOF
  rm -rf "$V"
fi
[ -e "$V" ] && { echo "removal FAILED: $V still exists"; exit 1; }
echo "removed $V"
# The shared pip cache holds the downloaded pydantic wheels: purge it whole (a cache, not data; the torch base it fed is
# installed already, so the cost is re-downloading small wheels for the next build).
cache_before=$(du -sb --apparent-size $RT/.pip-cache 2>/dev/null | cut -f1)
PIP_CACHE_DIR=$RT/.pip-cache $RT/speech-torch/bin/python -m pip cache purge
rm -rf $RT/.pip-cache/http $RT/.pip-cache/http-v2 $RT/.pip-cache/wheels
echo "pip cache purged: ${cache_before:-0} bytes before, $(du -sb --apparent-size $RT/.pip-cache 2>/dev/null | cut -f1) after"
printf '%s\n' \
  "STATUS: UNUSABLE under Cyril's no-ORM / no-Pydantic / no-SQLAlchemy rule (marked 2026-10-15 by Claude)." \
  "openbmb/VoxCPM2 rev 32279effe8c19989596f05d353d1447f51d9e915: the weights are verified and kept, but the only loader" \
  "(pip voxcpm 2.0.3) imports pydantic for its config classes, so no runtime of ours may load it. Its runtime" \
  "/mnt/lexar/omni-campaign/runtimes/speech-voxcpm was removed (evidence: \$OC/distill/speech/teachers/runtime/" \
  "speech-voxcpm.REMOVED-2026-10-15.json). Speech-out family 2 is Fun-CosyVoice3-0.5B-2512 or Qwen3-TTS only; see" \
  "distill/SPEECH_LANE_PLAN_v2_2026-10-14.md revision note 2026-10-15." \
  > "$M/openbmb/VoxCPM2/_STATUS_UNUSABLE_NO_PYDANTIC_RULE.txt"
printf '%s\n' \
  "STATUS: SUPERSEDED (marked 2026-10-15 by Claude)." \
  "ibm-granite/granite-speech-3.3-2b rev 4ac2f02f413c6169ae8c0ccc217115a366e552d7 is kept on disk (verified) but is no" \
  "longer a teacher: admission judge B is ibm-granite/granite-speech-4.1-2b rev de575db64086f84fdc79da4932d1076e965bc546" \
  "(/mnt/lexar/models/ibm-granite/granite-speech-4.1-2b). See distill/SPEECH_LANE_PLAN_v2_2026-10-14.md revision note" \
  "2026-10-15." \
  > "$M/ibm-granite/granite-speech-3.3-2b/_STATUS_SUPERSEDED.txt"
ls -la "$M/openbmb/VoxCPM2/_STATUS_UNUSABLE_NO_PYDANTIC_RULE.txt" "$M/ibm-granite/granite-speech-3.3-2b/_STATUS_SUPERSEDED.txt"
ls -la $RT
echo "=== REMOVAL DONE $(date -u +%FT%TZ)"
