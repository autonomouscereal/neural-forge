#!/usr/bin/env python3
"""write_teacher_status.py -- one file of record for the speech teachers after the 2026-10-15 teacher fix (Claude).

Reads every fetch receipt under $OC/distill/speech/teachers/fetch/ and the runtime freezes, and writes
$OC/distill/speech/teachers/TEACHER_STATUS_2026-10-15.json: role, status (active | superseded | unusable-under-rule |
reference), revision, bytes, receipt sha256, and the forbidden-package count of every speech runtime.  Stdlib only.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

OC = Path("/home/cereal/omni-campaign")
TEACH = OC / "distill/speech/teachers"
RT = Path("/mnt/lexar/omni-campaign/runtimes")
FORBIDDEN = re.compile(r"^(pydantic|pydantic-core|pydantic-settings|pydantic-extra-types|sqlalchemy|sqlalchemy-utils|sqlmodel|"
                       r"alembic|peewee|pony|tortoise-orm|django|mongoengine|odmantic|beanie|ormar|piccolo|fastapi|gradio)$")
ROLES = {
    "Qwen/Qwen3-ASR-1.7B": ("held speech-in judge + C-s5 soft targets", "active", "speech-asr"),
    "Qwen/Qwen3-ASR-0.6B": ("admission judge A", "active", "speech-asr"),
    "ibm-granite/granite-speech-4.1-2b": ("admission judge B (independent vendor)", "active", "speech-asr"),
    "ibm-granite/granite-speech-3.3-2b": ("former judge B", "superseded", "speech-asr (comparison only)"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": ("speech-out family 1", "active", "speech-qwentts"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": ("speech-out family 1 (designed voices)", "active", "speech-qwentts"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": ("owner-voice cloner 1", "active", "speech-qwentts"),
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": ("Codex canary parent (not a v1 teacher)", "reference", "speech-qwentts"),
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512": ("speech-out family 2 + owner-voice cloner 2 (RL LLM)", "active",
                                             "speech-cosyvoice3"),
    "openbmb/VoxCPM2": ("former family 2", "unusable-under-rule", "none (speech-voxcpm removed: it needed pydantic)"),
    "hexgrad/Kokoro-82M": ("former fallback #1", "unusable-under-rule (as packaged)",
                           "none (pip kokoro -> misaki[en] -> spacy/thinc/weasel/confection -> pydantic 2.14.1)"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    models = []
    for receipt in sorted((TEACH / "fetch").glob("*__*.json")):
        body = json.loads(receipt.read_text())
        if "repo" not in body:
            continue  # corpus receipts (librispeech) have no repo
        role, status, runtime = ROLES.get(body["repo"], ("unassigned", "reference", "-"))
        models.append({"repo": body["repo"], "revision": body["revision"], "dest": body["dest"], "role": role,
                       "status": status, "runtime": runtime, "licence": body["licence"]["card_license"],
                       "gated": body["gated"], "created": body.get("api_created_at"), "files": body["file_count"],
                       "bytes": body["total_bytes"], "complete": body["complete"], "receipt": str(receipt),
                       "receipt_sha256": sha256(receipt)})
    runtimes = {}
    for venv in sorted(p for p in RT.glob("speech-*") if (p / "bin/python").exists()):
        freeze = subprocess.run([str(venv / "bin/python"), "-m", "pip", "freeze"], capture_output=True, text=True,
                                check=True).stdout.splitlines()
        names = [re.split(r"[=@ ]", line, 1)[0].lower().replace("_", "-").replace(".", "-") for line in freeze if line]
        runtimes[venv.name] = {"packages": len(names), "forbidden": sorted(n for n in names if FORBIDDEN.match(n)),
                               "freeze_sha256": hashlib.sha256("\n".join(freeze).encode()).hexdigest()}
    out = {"schema": "omni-campaign.speech-teacher-status.v1", "by": "claude",
           "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "plan": "distill/SPEECH_LANE_PLAN_v2_2026-10-14.md section 0 (revision 2026-10-15)",
           "rule": "[Cyril quote withheld]",
           "models": models, "runtimes": runtimes,
           "removed_runtimes": {"speech-voxcpm": str(TEACH / "runtime/speech-voxcpm.REMOVED-2026-10-15.json")}}
    path = TEACH / "TEACHER_STATUS_2026-10-15.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path, sha256(path))
    for m in models:
        print(f"  {m['status']:20s} {m['repo']}@{m['revision'][:8]}  {m['role']}")
    for name, r in runtimes.items():
        print(f"  runtime {name}: {r['packages']} packages, forbidden {r['forbidden'] or 'none'}")
    return 0 if all(not r["forbidden"] for r in runtimes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
