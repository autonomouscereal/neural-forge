"""Bounded, comparison-only CPU speech qualification. No dataset admission."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import wave


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pinned(spec):
    if sha(spec["path"]) != spec["sha256"]:
        raise ValueError("source pin mismatch: " + str(spec["path"]))
    return json.loads(Path(spec["path"]).read_text(encoding="utf-8"))


def validate_request(request):
    body = {k: v for k, v in request.items() if k != "request_sha256"}
    if digest(body) != request.get("request_sha256"):
        raise ValueError("canary request seal mismatch")
    plan = pinned(request["plan"])
    if digest({k: v for k, v in plan.items() if k != "manifest_sha256"}) != plan["manifest_sha256"]:
        raise ValueError("comparison plan seal mismatch")
    if request["comparison_plan_sha256"] != plan["manifest_sha256"]:
        raise ValueError("canary/comparison identity mismatch")
    refs = {row["id"]: row for row in plan["rows"]}
    rows = request["rows"]
    if not 1 <= len(rows) <= 4 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("bounded canary must contain one to four distinct sources")
    for row in rows:
        reference = refs.get(row["id"])
        if reference is None or reference["split"] != "train" or reference["optimizer_admitted"] is not False:
            raise ValueError("canary must use comparison-only training references")
        for key in ("speaker", "text", "source_row_sha256", "recovery_group_id", "reference_source_duration_seconds"):
            if row.get(key) != reference[key]:
                raise ValueError("canary changed original reference " + key)
        source = row["baseline_original_audio"]
        original = reference["baseline_original_audio"]
        for key in ("sha256", "samples", "sample_rate"):
            if source.get(key) != original[key]:
                raise ValueError("canary changed full original waveform identity")
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("canary original waveform bytes mismatch")
        with wave.open(str(source["path"]), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) != (1, 2, 24000, source["samples"]):
                raise ValueError("canary full PCM layout/length mismatch")
            if len(stream.readframes(source["samples"])) != 2 * source["samples"]:
                raise ValueError("canary waveform incomplete")
    limits = request["limits"]
    if limits != {"threads": 4, "memory_cap_gib": 24, "task_timeout_seconds": 600, "max_new_tokens": 256}:
        raise ValueError("CPU canary resource/generation policy changed")
    return plan, rows


def words(value):
    return re.findall(r"[a-z0-9]+", value.lower())


def transcript_metrics(expected, actual):
    a, b = words(expected), words(actual)
    if not a:
        raise ValueError("empty reference transcript")
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return {"verbatim_exact": expected == actual, "case_punctuation_exact": a == b,
            "word_edit_distance": previous[-1], "reference_words": len(a), "wer": previous[-1] / len(a)}


def verify_model(request, mode, plan):
    if __package__:
        from .speech_teacher_comparison import verify_download
    else:
        from speech_teacher_comparison import verify_download
    if mode != "asr_baseline":
        kind = "tts" if mode == "tts" else "asr"
        receipt = pinned(request["download"])
        if receipt.get("complete") is not True or receipt.get("all_source_bytes_verified") is not True:
            raise ValueError("candidate source download incomplete")
        if receipt["comparison_plan_sha256"] != plan["manifest_sha256"]:
            raise ValueError("candidate download/comparison identity mismatch")
        modeldir = Path(request["candidate_models"][kind])
        verified = verify_download(plan["comparison_parents"][kind], modeldir)
    else:
        receipt = pinned(request["baseline_completion"])
        if receipt.get("complete") is not True:
            raise ValueError("baseline source download incomplete")
        selected = [r for r in receipt["models"] if r["model_id"] == "Qwen/Qwen3-ASR-0.6B"]
        if len(selected) != 1:
            raise ValueError("baseline ASR variant missing")
        source = selected[0]
        if source["revision"] != "5eb144179a02acc5e5ba31e748d22b0cf3e303b0":
            raise ValueError("baseline ASR revision changed")
        modeldir = Path(source["path"])
        for item in source["files"]:
            path = (modeldir / item["path"]).resolve()
            if not path.is_relative_to(modeldir.resolve()) or path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
                raise ValueError("baseline ASR actual source bytes changed")
        verified = {"model_id": source["model_id"], "revision": source["revision"], "all_source_bytes_verified": True}
    return modeldir, verified


def run(request, mode, output):
    import resource
    import signal
    output = Path(output)
    output.mkdir(exist_ok=False)
    result = {"schema": "chimera_recovery.speech_teacher_cpu_canary.v1", "mode": mode,
              "request_sha256": request.get("request_sha256"), "helper_sha256": sha(__file__),
              "rows": [], "complete": False, "comparison_only": True, "optimizer_admitted": False,
              "native_train_ready": False, "training_started": False, "device": "cpu",
              "dtype": "float32", "attention": "eager", "model_loaded": False,
              "inference_performed": False, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    begin = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("600 second CPU model-task budget exhausted")
    old_alarm = signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    try:
        plan, rows = validate_request(request)
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
        result["host_available_before_bytes"] = available
        if available < 32 * 1024**3:
            raise ValueError("CPU canary requires 32 GiB host availability before launch")
        cap = 24 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        result["memory_limit"] = {"kind": "RLIMIT_AS", "bytes": cap}
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU canary requires explicit CUDA invisibility")
        import numpy as np
        import soundfile as sf
        import torch
        import transformers
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(261013)
        if torch.cuda.is_available():
            raise ValueError("GPU visible in CPU-only qualification")
        kind = "tts" if mode == "tts" else "asr"
        versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                    "speech_package": importlib.metadata.version("qwen-" + kind)}
        expected = {"torch": "2.11.0+cu128", "transformers": "4.57.3" if kind == "tts" else "4.57.6", "speech_package": "0.1.1" if kind == "tts" else "0.0.6"}
        if versions != expected:
            raise ValueError("unchanged private runtime version contract mismatch")
        result["runtime_versions"] = versions
        source_spec = request["runtime_source"][kind]
        module = importlib.import_module(source_spec["module"])
        if sha(module.__file__) != source_spec["sha256"]:
            raise ValueError("actual speech implementation source SHA mismatch")
        result["runtime_source"] = {"path": module.__file__, "sha256": sha(module.__file__)}
        modeldir, verified = verify_model(request, mode, plan)
        result["teacher"] = verified
        save(output / "result.json", result)
        if mode == "tts":
            from qwen_tts import Qwen3TTSModel
            model = Qwen3TTSModel.from_pretrained(str(modeldir), device_map="cpu", dtype=torch.float32, attn_implementation="eager", local_files_only=True)
            result["model_loaded"] = True
            shapes = []
            original = model.model.generate
            def measured(*args, **kwargs):
                value = original(*args, **kwargs)
                shapes.append([list(code.shape) for code in value[0]])
                return value
            model.model.generate = measured
            for row in rows:
                start = time.monotonic()
                waves, rate = model.generate_custom_voice(text=row["text"], language="English", speaker=row["speaker"], instruct="", do_sample=False, subtalker_dosample=False, max_new_tokens=256)
                result["inference_performed"] = True
                values = np.asarray(waves[0], dtype=np.float32)
                if len(waves) != 1 or int(rate) != 24000 or values.ndim != 1 or not values.size or not np.isfinite(values).all() or np.max(np.abs(values)) > 1:
                    raise ValueError("candidate did not produce finite full 24kHz waveform")
                if len(shapes[-1]) != 1 or not 0 < shapes[-1][0][0] < 256:
                    raise ValueError("candidate generation hit token cap")
                path = output / (row["id"] + ".wav")
                sf.write(path, values, 24000, subtype="PCM_16")
                with wave.open(str(path), "rb") as stream:
                    if stream.getnframes() != values.size:
                        raise ValueError("output PCM omitted generated samples")
                duration = values.size / 24000
                item = {"id": row["id"], "text": row["text"], "speaker": row["speaker"], "split": "train",
                        "audio": str(path), "audio_sha256": sha(path), "samples": int(values.size), "sample_rate": 24000,
                        "duration_seconds": duration, "baseline_duration_seconds": row["reference_source_duration_seconds"],
                        "duration_delta_seconds": duration - row["reference_source_duration_seconds"], "codec_shape": shapes[-1][0],
                        "hit_cap": False, "trimmed_samples": 0, "padded_samples": 0, "stretched": False,
                        "rms": float(np.sqrt(np.mean(values.astype(np.float64)**2))), "peak": float(np.max(np.abs(values))),
                        "inference_seconds": time.monotonic() - start}
                result["rows"].append(item)
                save(output / "result.json", result)
                print(json.dumps({"mode": mode, "id": row["id"], "duration_seconds": duration, "inference_seconds": item["inference_seconds"]}), flush=True)
        else:
            from qwen_asr import Qwen3ASRModel
            generated = pinned(request["tts_result"])
            if generated.get("complete") is not True or generated.get("request_sha256") != request["parent_request_sha256"]:
                raise ValueError("matching bounded TTS result incomplete")
            outputs_by_id = {row["id"]: row for row in generated["rows"]}
            if set(outputs_by_id) != {row["id"] for row in rows}:
                raise ValueError("TTS and ASR comparison coverage differs")
            model = Qwen3ASRModel.from_pretrained(str(modeldir), dtype=torch.float32, device_map="cpu", max_inference_batch_size=1, max_new_tokens=256, attn_implementation="eager", local_files_only=True)
            result["model_loaded"] = True
            for row in rows:
                gen = outputs_by_id[row["id"]]
                for name, asset in (("baseline_original", row["baseline_original_audio"]), ("candidate_generated", {"path": gen["audio"], "sha256": gen["audio_sha256"]})):
                    if sha(asset["path"]) != asset["sha256"]:
                        raise ValueError("full ASR waveform changed")
                    values, rate = sf.read(asset["path"], dtype="float32")
                    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
                        raise ValueError("ASR waveform invalid")
                    start = time.monotonic()
                    # No context/prompt transcript cue; the expected answer is used only for scoring.
                    outputs = model.transcribe(audio=(values, int(rate)), language="English")
                    result["inference_performed"] = True
                    if len(outputs) != 1:
                        raise ValueError("ASR returned unexpected output count")
                    actual = outputs[0].text
                    item = {"id": row["id"], "waveform_kind": name, "audio_sha256": asset["sha256"],
                            "input_samples": int(values.size), "sample_rate": int(rate), "duration_seconds": values.size / int(rate),
                            "expected": row["text"], "transcript": actual, "language": outputs[0].language,
                            "answer_cue_in_prompt": False, "inference_seconds": time.monotonic() - start,
                            **transcript_metrics(row["text"], actual)}
                    result["rows"].append(item)
                    save(output / "result.json", result)
                    print(json.dumps({"mode": mode, "id": row["id"], "waveform_kind": name, "wer": item["wer"]}), flush=True)
        result["complete"] = len(result["rows"]) == len(rows) * (1 if mode == "tts" else 2)
        result["inference_performed"] = True
    except Exception as error:
        result["failure"] = type(error).__name__ + ": " + str(error)
        result["gpu_qualification_pending"] = True
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        result["elapsed_seconds"] = time.monotonic() - begin
        result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tts", "asr_candidate", "asr_baseline"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    run(request, args.mode, args.output)


if __name__ == "__main__":
    main()
