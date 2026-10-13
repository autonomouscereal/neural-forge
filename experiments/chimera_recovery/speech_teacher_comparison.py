"""Prepare source-bound speech teacher comparisons; never launch inference here."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import wave


PARENTS = {
    "asr": ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
    "tts": ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "0c0e3051f131929182e2c023b9537f8b1c68adfe"),
}
BASELINE_IDS = {"Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"}
SPEAKERS = {"Ryan", "Aiden", "Vivian", "Serena"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_pin(spec):
    path = Path(spec["path"])
    if file_sha(path) != spec["sha256"]:
        raise ValueError("source file SHA mismatch: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def rows_pin(spec):
    path = Path(spec["path"])
    if file_sha(path) != spec["sha256"]:
        raise ValueError("source rows SHA mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if "rows" in spec and len(rows) != spec["rows"]:
        raise ValueError("source row count mismatch")
    return rows


def download_plan(metadata):
    plans = {}
    for kind, expected in PARENTS.items():
        data = metadata[kind]
        if (data.get("id"), data.get("sha")) != expected:
            raise ValueError("selected 1.7B parent identity/revision mismatch")
        if data.get("cardData", {}).get("license") != "apache-2.0":
            raise ValueError("selected parent license declaration mismatch")
        files = []
        for source in data["siblings"]:
            name = source["rfilename"]
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("unsafe model artifact path")
            lfs = source.get("lfs") or {}
            identity = lfs.get("sha256") or source.get("blobId")
            if not isinstance(source.get("size"), int) or not identity:
                raise ValueError("model artifact lacks upstream size/hash")
            hash_kind = "sha256" if lfs else "git_blob_sha1"
            if len(identity) != (64 if lfs else 40):
                raise ValueError("invalid upstream artifact hash")
            files.append({"path": name, "bytes": source["size"], "hash_kind": hash_kind,
                          "hash": identity,
                          "url": f"https://huggingface.co/{expected[0]}/resolve/{expected[1]}/{name}"})
        paths = {item["path"] for item in files}
        common = {"README.md", "config.json", "generation_config.json", "preprocessor_config.json",
                  "tokenizer_config.json", "merges.txt", "vocab.json"}
        required = common | ({"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                              "model.safetensors.index.json", "chat_template.json"} if kind == "asr" else
                             {"model.safetensors", "speech_tokenizer/model.safetensors",
                              "speech_tokenizer/config.json", "speech_tokenizer/preprocessor_config.json"})
        if not required <= paths:
            raise ValueError("incomplete selected native parent artifact inventory")
        plans[kind] = {"model_id": expected[0], "revision": expected[1], "license": "apache-2.0",
                       "files": files, "bytes": sum(item["bytes"] for item in files)}
    return plans


def verify_download(plan, root):
    """Validate real downloaded bytes offline, including git-blob and LFS hashes."""
    root = Path(root).resolve()
    verified = []
    for item in plan["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("artifact resolves outside model directory")
        if path.stat().st_size != item["bytes"]:
            raise ValueError("artifact byte count mismatch: " + item["path"])
        h = hashlib.sha256() if item["hash_kind"] == "sha256" else hashlib.sha1()
        if item["hash_kind"] == "git_blob_sha1":
            h.update(f"blob {item['bytes']}\0".encode())
        elif item["hash_kind"] != "sha256":
            raise ValueError("unknown upstream hash policy")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != item["hash"]:
            raise ValueError("artifact upstream hash mismatch: " + item["path"])
        verified.append({"path": item["path"], "bytes": item["bytes"], "sha256": file_sha(path)})
    return {"schema": "chimera_recovery.speech_parent_byte_validation.v1", "model_id": plan["model_id"],
            "revision": plan["revision"], "all_source_bytes_verified": True, "files": verified,
            "model_loaded": False, "inference_performed": False}


def sealed_plan(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    seal = value.pop("manifest_sha256")
    if digest(value) != seal:
        raise ValueError("comparison plan seal mismatch")
    value["manifest_sha256"] = seal
    return value


def download_candidates(plan, output, *, reserve_bytes=32 * 1024**3, opener=None):
    """Download only exact selected source files into a new private namespace."""
    opener = opener or urllib.request.urlopen
    models = plan["comparison_parents"]
    for kind, expected in PARENTS.items():
        model = models[kind]
        if (model["model_id"], model["revision"]) != expected or model["license"] != "apache-2.0":
            raise ValueError("download parent selection mismatch")
        prefix = f"https://huggingface.co/{expected[0]}/resolve/{expected[1]}/"
        for item in model["files"]:
            if item["url"] != prefix + item["path"] or Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts:
                raise ValueError("download URL/path is not the selected source artifact")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(m["bytes"] for m in models.values())
    free = shutil.disk_usage(output.parent).free
    if free < required + reserve_bytes:
        raise ValueError("insufficient disk headroom for separate source parents")
    output.mkdir(exist_ok=False)
    receipt = {"schema": "chimera_recovery.speech_parent_download.v1", "comparison_plan_sha256": plan["manifest_sha256"],
               "helper_sha256": file_sha(__file__), "required_bytes": required, "free_before_bytes": free,
               "reserve_bytes": reserve_bytes, "models": {}, "complete": False,
               "model_loaded": False, "inference_performed": False, "training_started": False}
    def save():
        temporary = output / "download.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "download.json")
    save()
    try:
        for kind, model in models.items():
            modeldir = output / Path(model["model_id"]).name
            modeldir.mkdir()
            receipt["active_model"] = model["model_id"]
            for item in model["files"]:
                target = (modeldir / item["path"]).resolve()
                if not target.is_relative_to(modeldir):
                    raise ValueError("download artifact escapes its model directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial")
                h = hashlib.sha256() if item["hash_kind"] == "sha256" else hashlib.sha1()
                if item["hash_kind"] == "git_blob_sha1":
                    h.update(f"blob {item['bytes']}\0".encode())
                elif item["hash_kind"] != "sha256":
                    raise ValueError("unknown artifact identity policy")
                count, saved = 0, time.monotonic()
                receipt["active_file"] = item["path"]
                receipt["active_file_bytes"] = 0
                save()
                request = urllib.request.Request(item["url"], headers={"User-Agent": "ChimeraRecoverySourceDownload/2026-10-13"})
                with opener(request, timeout=60) as response, partial.open("xb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > item["bytes"]:
                            raise ValueError("download exceeds upstream artifact length")
                        stream.write(block)
                        h.update(block)
                        if time.monotonic() - saved > 10:
                            receipt["active_file_bytes"] = count
                            save()
                            saved = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
                if count != item["bytes"] or h.hexdigest() != item["hash"]:
                    raise ValueError("download length/upstream hash mismatch: " + item["path"])
                os.replace(partial, target)
                receipt["active_file_bytes"] = count
                save()
            receipt["models"][kind] = verify_download(model, modeldir)
            save()
        receipt["complete"] = True
        receipt["all_source_bytes_verified"] = True
        receipt["free_after_bytes"] = shutil.disk_usage(output).free
        save()
        return receipt
    except Exception as error:
        receipt["failure"] = type(error).__name__ + ": " + str(error)
        save()
        raise


def baseline_wave(path, expected_sha):
    if file_sha(path) != expected_sha:
        raise ValueError("baseline waveform SHA mismatch")
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 24000):
            raise ValueError("baseline speech is not full 24kHz mono PCM16")
        count = stream.getnframes()
        data = stream.readframes(count)
    if len(data) != count * 2 or not count:
        raise ValueError("baseline speech PCM coverage mismatch")
    return count


def compare_rows(records, admissions):
    by_id = {row["id"]: row for row in admissions}
    if len(by_id) != len(admissions):
        raise ValueError("duplicate global admission identity")
    rows, groups, seen = [], {}, set()
    for source in records:
        identity = source["id"]
        if identity in seen:
            raise ValueError("duplicate speech source")
        seen.add(identity)
        admission = by_id.get(identity)
        prov = source.get("provenance", {})
        if (source.get("route") != "speech_understand" or admission is None or
                admission.get("source_row_sha256") != digest(source)):
            raise ValueError("speech source/global admission join mismatch")
        if prov.get("independent_asr_exact_match") is not True:
            raise ValueError("unverified/quarantined transcript cannot enter comparison")
        speaker = prov.get("speaker")
        if speaker not in SPEAKERS or prov.get("synthetic_official_voice") is not True or prov.get("user_reference_voice_used") is not False:
            raise ValueError("baseline official speaker policy mismatch")
        if source["split"] != admission["split"] or source["split"] not in {"train", "validation", "test"}:
            raise ValueError("original speech partition changed")
        admitted = admission.get("train_admitted") is True
        if (source["split"] == "train") != admitted:
            raise ValueError("non-admitted training source cannot enter comparison")
        group = admission["recovery_group_id"]
        if group in groups and groups[group] != source["split"]:
            raise ValueError("connected speech group crosses partitions")
        groups[group] = source["split"]
        messages = source["messages"]
        if not messages or messages[-1]["role"] != "assistant" or not isinstance(messages[-1]["content"], str):
            raise ValueError("actual authored transcript missing")
        answer = messages[-1]["content"]
        if admission.get("authored_answer_sha256") != digest(answer):
            raise ValueError("source transcript/global admission join mismatch")
        media = source["media"]
        if len(media) != 1 or admission.get("asset_sha256") != media[0]["sha256"]:
            raise ValueError("source asset/global admission join mismatch")
        original = Path(prov["original_audio_path"])
        aligned = Path(media[0]["path"])
        original_samples = baseline_wave(original, prov["original_audio_sha256"])
        aligned_samples = baseline_wave(aligned, media[0]["sha256"])
        tail = aligned_samples - original_samples
        if not 0 <= tail < 600 or aligned_samples % 600:
            raise ValueError("baseline native duration alignment mismatch")
        with wave.open(str(original), "rb") as a, wave.open(str(aligned), "rb") as b:
            pcm = a.readframes(original_samples)
            derived = b.readframes(aligned_samples)
        if derived != pcm + b"\0\0" * tail:
            raise ValueError("baseline aligned audio changed original speech")
        rows.append({"id": identity, "source_row_sha256": digest(source), "split": source["split"],
                     "recovery_group_id": group, "speaker": speaker, "language": "English", "text": answer,
                     "baseline_original_audio": {"path": str(original), "sha256": prov["original_audio_sha256"],
                                                 "sample_rate": 24000, "samples": original_samples},
                     "baseline_aligned_audio": {"path": str(aligned), "sha256": media[0]["sha256"],
                                                "samples": aligned_samples},
                     "reference_source_duration_seconds": original_samples / 24000,
                     "reference_h3_frames": aligned_samples // 600,
                     "reference_requested_native_duration_seconds": aligned_samples / 24000,
                     "baseline_source_train_admitted": admitted, "optimizer_admitted": False})
    return sorted(rows, key=lambda item: item["id"])


def prepare(spec, output):
    prepared = load_pin(spec["speech_preparation"])
    baseline = load_pin(spec["baseline_completion"])
    for key in ("availability", "runtime_packages", "baseline_native_replay_index"):
        if key in spec:
            value = load_pin(spec[key])
            if key == "baseline_native_replay_index" and not all(
                    value.get(flag) is True for flag in ("sealed", "verified", "passed", "complete_source_traversal")):
                raise ValueError("baseline native replay proof incomplete")
    if baseline.get("complete") is not True or {m["model_id"] for m in baseline["models"]} != BASELINE_IDS:
        raise ValueError("immutable 0.6B teacher baseline mismatch")
    admission = load_pin(spec["global_admission"])
    if not all(admission.get(k) is True for k in ("sealed", "complete_source_traversal", "connected_groups_verified", "frozen_exclusions_verified")):
        raise ValueError("complete frozen global admission required")
    records = rows_pin(prepared["outputs"]["source_ready"])
    inventory = rows_pin(prepared["outputs"]["source_inventory"])
    if not {digest(r) for r in records} <= {digest(r) for r in inventory}:
        raise ValueError("ready speech source not in complete inventory")
    admitted = rows_pin(admission["outputs"]["admission"])
    rows = compare_rows(records, admitted)
    downloads = download_plan({kind: load_pin(spec["parents"][kind]) for kind in PARENTS})
    chosen = []
    for speaker in sorted(SPEAKERS):
        found = next((r for r in rows if r["split"] == "train" and r["speaker"] == speaker), None)
        if found is not None:
            chosen.append(found["id"])
    for split in ("validation", "test"):
        for speaker in ("Ryan", "Vivian"):
            found = next((r for r in rows if r["split"] == split and r["speaker"] == speaker), None)
            if found is not None:
                chosen.append(found["id"])
    plan = {"schema": "chimera_recovery.speech_teacher_comparison.v1", "sources": spec,
            "baseline_variants": sorted(BASELINE_IDS), "comparison_parents": downloads,
            "rows": rows, "complete_inventory_rows": len(inventory),
            "source_speech_rows": len(rows), "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")},
            "tts_serial_canary_ids": chosen, "asr_same_waveform_ids": [r["id"] for r in rows],
            "new_cache_namespace": "speech-1p7-comparison-v1", "comparison_only": True,
            "automatic_launch": False, "download_performed": False, "inference_performed": False,
            "optimizer_admitted": False, "native_train_ready": False,
            "runtime": {"asr_package": "qwen-asr==0.0.6", "tts_package": "qwen-tts==0.1.1",
                        "separate_validated_private_runtimes": True, "max_concurrent_models": 1,
                        "cpu_threads": 8, "cpu_memory_cap_gib": 32, "gpu_launch_requires_exclusive_allocator": True},
            "generation_settings": {"do_sample": False, "subtalker_dosample": False, "instruct": "", "max_new_tokens": 256,
                                    "hit_token_cap_is_failure": True, "speaker_and_text_from_original_source": True},
            "duration_policy": {"asr_input_waveforms_and_durations_bitwise_identical_to_baseline": True,
                                "tts_reference_source_and_requested_native_durations_pinned_per_row": True,
                                "native_tts_explicit_duration_argument_available": False,
                                "tts_same_duration_guaranteed": False, "full_generated_audio_preserved": True,
                                "silently_trim_stretch_or_pad_to_baseline_duration": False,
                                "matched_duration_tts_lane_ready": False,
                                "require_measured_duration_delta_and_explicit_new_H3_derivation": True},
            "comparison_metrics": ["per-row ASR exact match and WER on identical baseline original/aligned waveform",
                                   "full 1.7B generated waveform independent ASR agreement without transcript prompt cues",
                                   "actual source/derived duration and delta against pinned 0.6B reference",
                                   "token-cap failures, finite waveform, native H3 encode/reconstruction and decoded-speech ASR",
                                   "held results reported separately, zero optimizer updates"],
            "blockers": ["1.7B parent bytes not verified locally", "1.7B runtime load/canary and resource qualification pending",
                         "new waveform global admission and native proofs pending", "matched-duration TTS control unavailable"]}
    plan["manifest_sha256"] = digest(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--spec", required=True)
    prep.add_argument("--output", required=True)
    verify = sub.add_parser("verify-download")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--kind", choices=tuple(PARENTS), required=True)
    verify.add_argument("--model-dir", required=True)
    download = sub.add_parser("download-candidates")
    download.add_argument("--plan", required=True)
    download.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = prepare(json.loads(Path(args.spec).read_text(encoding="utf-8")), args.output)
        print(json.dumps({k: plan[k] for k in ("schema", "manifest_sha256", "source_speech_rows", "split_counts", "native_train_ready")}))
    elif args.command == "verify-download":
        whole = sealed_plan(args.plan)
        print(json.dumps(verify_download(whole["comparison_parents"][args.kind], args.model_dir), indent=2))
    else:
        receipt = download_candidates(sealed_plan(args.plan), args.output)
        print(json.dumps({k: receipt[k] for k in ("schema", "required_bytes", "complete", "all_source_bytes_verified", "inference_performed")}))


if __name__ == "__main__":
    main()
