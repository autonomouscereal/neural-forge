"""Real PCM coverage/source seals/answer identity; no codec fabrication."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import wave

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/audio_preflight.py"
spec = importlib.util.spec_from_file_location("recovery_audio_preflight_test", PATH)
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)


def fixture(tmp_path):
    assets = tmp_path / "assets"
    records = []
    for index, split in enumerate(("train", "validation", "test")):
        case = f"case{index}"
        folder = assets / case
        folder.mkdir(parents=True)
        audio = folder / "sound.wav"
        with wave.open(str(audio), "wb") as waveform:
            waveform.setnchannels(1)
            waveform.setsampwidth(2)
            waveform.setframerate(16000)
            waveform.writeframes(struct.pack("<16000h", *([20] * 16000)))
        answer = {"contains_speech": False, "count": 1, "events": [{"kind": "fixture", "start_seconds": .25, "duration_seconds": .25}]}
        truth = folder / "groundtruth.json"
        truth.write_bytes(P.R.canonical({"groundtruth": answer}))
        records.append({"id": case, "route": "audio_understand", "split": split,
                        "provenance": {"split": split, "group_id": case, "case_id": case, "groundtruth_sha256": P.R.sha(truth)},
                        "quality": {"source_ready": True, "groundtruth_verified": True},
                        "supervision": {"policy": "assistant_only", "message_mask": [0, 1]},
                        "media": [{"path": str(audio), "type": "audio", "role": "input", "sha256": P.R.sha(audio), "bytes": audio.stat().st_size,
                                   "sample_rate": 16000, "duration_seconds": 1.}],
                        "messages": [{"role": "user", "content": "Identify"}, {"role": "assistant", "content": json.dumps(answer)}]})
    source, report, config, manifest = (tmp_path / name for name in ("source.jsonl", "report.json", "config.json", "composite_manifest.json"))
    def seal():
        source.write_bytes(b"".join(P.R.canonical(row) + b"\n" for row in records))
        report.write_bytes(P.R.canonical({"artifacts": {source.name: {"sha256": P.R.sha(source), "rows": len(records)}}, "cross_split_media_conflicts": 0}))
    seal()
    config.write_bytes(P.R.canonical({"config": {"charted_blocks": [{"source": "h3", "type_ids": [2, 3]}]}}))
    manifest.write_bytes(P.R.canonical({"interface": {"entries": [{"type_id": 3, "feature_band": [480, 512], "emission_band": [224, 256]}]}}))
    return records, seal, (source, report, assets, config, manifest)


def test_actual_pcm_authored_answers_holdouts_and_no_native_admission(tmp_path):
    _, _, args = fixture(tmp_path)
    result = P.audit(*args)
    assert result["audio_rows"] == 3 and result["speech_rows"] == 0
    assert result["source_ready"] and not result["production_training_ready"]
    assert result["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert all(row["samples"] == 16000 and row["duration_seconds"] == 1. and not row["native_train_admitted"] for row in result["rows"])
    assert not result["current_native_audio_input"]["raw_pcm_or_foreign_speech_codec_compatible"]
    assert result["proposal"]["teacher_logits_required"] is False
    assert any("Qwen/Fish" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize("problem,reason", [("group", "connected group"), ("answer", "reference differs"), ("coverage", "coverage differs"), ("band", "allocation")])
def test_source_label_coverage_group_or_foreign_codec_drift_refuses(tmp_path, problem, reason):
    records, seal, args = fixture(tmp_path)
    if problem == "group":
        records[1]["provenance"]["group_id"] = records[0]["provenance"]["group_id"]
    elif problem == "answer":
        answer = json.loads(records[0]["messages"][-1]["content"])
        answer["count"] = 5
        records[0]["messages"][-1]["content"] = json.dumps(answer)
    elif problem == "coverage":
        records[0]["media"][0]["duration_seconds"] = .9
    else:
        args[-1].write_bytes(P.R.canonical({"interface": {"entries": [{"type_id": 3, "feature_band": [480, 2016], "emission_band": [224, 256]}]}}))
    seal()
    with pytest.raises(P.Error, match=reason):
        P.audit(*args)
