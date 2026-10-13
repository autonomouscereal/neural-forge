"""Audit real authored audio inputs and emit explicit native training blockers.

This CPU-only report validates waveform/answer/source/split identities. It
does not create a clean encoder contract or grant native training admission.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
import wave

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chimera_recovery_audio_preflight_common", HERE / "train_audio.py")
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)
R, Error = A.R, A.Error


def audit(source, source_report, asset_root, composite_config, composite_manifest):
    report = json.loads(source_report.read_text())
    seal = report["artifacts"][source.name]
    all_rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if R.sha(source) != seal["sha256"] or len(all_rows) != seal["rows"] or report.get("cross_split_media_conflicts") != 0:
        raise Error("authored audio source/report/group receipt changed")
    config = json.loads(composite_config.read_text())["config"]
    manifest = json.loads(composite_manifest.read_text())
    h3 = [unit for unit in config["charted_blocks"] if unit["source"] == "h3" and 3 in unit["type_ids"]]
    band = [entry for entry in manifest["interface"]["entries"] if entry["type_id"] == 3]
    if len(h3) != 1 or len(band) != 1 or band[0]["feature_band"] != [480, 512] or band[0]["emission_band"] != [224, 256]:
        raise Error("audio proposal needs the actual unique H3/type3 current32-channel input allocation")
    seen, groups, rows = set(), {}, []
    split_kinds = {split: Counter() for split in ("train", "validation", "test")}
    for record in all_rows:
        if record.get("route") != "audio_understand":
            continue
        provenance, quality = record["provenance"], record.get("quality", {})
        split, group, ident, case = record["split"], provenance["group_id"], record["id"], provenance["case_id"]
        if (ident in seen or split not in split_kinds or provenance.get("split") != split or
                group in groups and groups[group] != split or case in {"", ".", ".."} or "/" in case or "\\" in case or
                quality.get("source_ready") is not True or quality.get("groundtruth_verified") is not True or
                record.get("supervision", {}).get("policy") != "assistant_only" or
                record["supervision"].get("message_mask") != [False, True]):
            raise Error("authored audio source/assistant/split/connected group admission differs")
        seen.add(ident)
        groups[group] = split
        if len(record["media"]) != 1:
            raise Error("audio source must contain its one actual input waveform")
        media = record["media"][0]
        name = str(media["path"]).replace("\\", "/").split("/")[-1]
        if media.get("type") != "audio" or media.get("role") != "input" or name != "sound.wav":
            raise Error("authored audio media input identity differs")
        path = asset_root / case / name
        truth = asset_root / case / "groundtruth.json"
        if (R.sha(path) != R.require_sha(media["sha256"], "audio.asset_sha256") or path.stat().st_size != media["bytes"] or
                R.sha(truth) != R.require_sha(provenance["groundtruth_sha256"], "audio.groundtruth_sha256")):
            raise Error("authored audio waveform or groundtruth bytes changed")
        messages = record["messages"]
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise Error("authored audio needs one terminal assistant reference")
        answer, actual = json.loads(messages[-1]["content"]), json.loads(truth.read_text())["groundtruth"]
        if (set(answer) != {"contains_speech", "count", "events"} or answer["contains_speech"] is not False or
                type(answer["count"]) is not int or answer["count"] != len(answer["events"]) or
                any(actual.get(key) != value for key, value in answer.items())):
            raise Error("audio authored reference differs from actual measured non-speech fixture groundtruth")
        with wave.open(str(path), "rb") as waveform:
            channels, rate, frames, width, compression = (waveform.getnchannels(), waveform.getframerate(), waveform.getnframes(), waveform.getsampwidth(), waveform.getcomptype())
            raw = waveform.readframes(frames)
        duration = frames / rate
        if (channels != 1 or width != 2 or compression != "NONE" or rate != media.get("sample_rate") or
                not math.isclose(duration, media.get("duration_seconds"), rel_tol=0., abs_tol=1e-12) or
                len(raw) != channels * width * frames):
            raise Error("audio source PCM header/sample coverage differs from actual source metadata")
        for event in answer["events"]:
            start, length = event["start_seconds"], event["duration_seconds"]
            if (not math.isfinite(start) or not math.isfinite(length) or start < 0 or length <= 0 or start + length > duration or
                    any(not math.isfinite(event[key]) or not 0 < event[key] < rate / 2 for key in ("frequency_hz", "end_frequency_hz") if key in event)):
                raise Error("audio event labels exceed real waveform/Nyquist bounds")
            split_kinds[split][event["kind"]] += 1
        rows.append({"id": ident, "split": split, "recovery_group_id": group, "waveform_sha256": media["sha256"],
                     "groundtruth_sha256": provenance["groundtruth_sha256"], "source_row_sha256": R.digest(record),
                     "channels": channels, "sample_rate": rate, "samples": frames, "duration_seconds": duration,
                     "authored_answer_sha256": R.digest(answer), "native_train_admitted": False})
    if not rows:
        raise Error("no actual audio source rows")
    return {"schema": "chimera_recovery.audio_understanding_preflight.v1", "source_ready": True,
            "production_training_ready": False, "model_weights_loaded": False, "training_started": False,
            "source_sha256": R.sha(source), "source_report_sha256": R.sha(source_report),
            "config_sha256": R.sha(composite_config), "composite_manifest_sha256": R.sha(composite_manifest),
            "engine_sha256": R.sha(HERE / "train_audio.py"), "audio_rows": len(rows),
            "split_counts": dict(Counter(row["split"] for row in rows)), "event_kind_counts_by_split": split_kinds,
            "speech_rows": 0, "scope": "authored procedural non-speech fixtures only; no ASR/general semantic capability claim",
            "current_native_audio_input": {"type_id": 3, "feature_band": [480, 512], "emission_band": [224, 256],
                                           "channels": 32, "source": "H3", "raw_pcm_or_foreign_speech_codec_compatible": False},
            "proposal": {"objective": A.OBJECTIVE, "read_policy": A.READ_POLICY, "teacher_logits_required": False,
                         "denominator": "actual assistant body/EOS target count", "ownership": ["H3 unit", "type3 input bridge", "all-layer shared r16 LoRA"],
                         "frozen": ["other native units", "text embedding/head", "generative emission"],
                         "source_policy": "train only; all connected validation/test/global held groups excluded"},
            "blockers": ["source-audited standalone full-coverage no-trim clean audio encoder contract absent",
                         "resampling/channels/latent layout and measured audio-only native/source-time positions absent",
                         "current32-channel H3 mapping does not establish any Qwen/Fish speech codec mapping",
                         "dated distinct authored-audio CE objective and prompt-only H3 attention amendment required",
                         "sealed native batch conversion/global group admission and actual parent gradient/answer evaluation required",
                         "sequential shared-adapter composition and production dispatcher binding required"],
            "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-report", "asset-root", "composite-config", "composite-manifest", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise Error("preserve existing audio preflight evidence; select a new output")
    result = audit(args.source, args.source_report, args.asset_root, args.composite_config, args.composite_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(R.canonical(result) + b"\n")
    print(json.dumps({"audio_rows": result["audio_rows"], "source_ready": True, "production_training_ready": False, "blockers": result["blockers"]}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
