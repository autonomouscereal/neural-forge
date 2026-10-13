"""Prepare pinned 2026 media source jobs without importing Torch or launching jobs.

Raw fixtures, teacher-cache jobs, native batches, and trained models are distinct
states. This module never promotes a source fixture to native training readiness.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

try:
    from .memory_harvest import normalize_text, prompt_overlap
except ImportError:
    from memory_harvest import normalize_text, prompt_overlap


ROUTES = {
    "image_edit": ("D3b", "reference-image-conditioned velocity adapter + actual edit teacher cache"),
    "joint_av": ("D3b", "sealed H3 teacher cache + synchronized packed native adapter"),
    "volume_generate": ("D3b", "genuine geometry-to-native HY3D adapter + actual geometry teacher cache"),
    "vision_grounding": ("D3d", "native QVL preprocessing + answer/feature cache + D3d driver"),
    "video_understand": ("D3d", "native frame/time adapter + QVL answer/feature cache + D3d driver"),
    "audio_understand": ("D3d", "native audio-to-text teacher/conditioning adapter + D3d driver"),
}
SPLITS = {"train", "validation", "test"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    return {"path": str(path.absolute()), "sha256": sha(path), "bytes": path.stat().st_size}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def checked_file(entry, fixture_root):
    path = Path(entry["path"]).resolve(strict=True)
    relative = path.relative_to(fixture_root)
    if relative.parts[0] != "assets" or sha(path) != entry["sha256"]:
        raise ValueError("asset path or digest differs from fixture source")
    if "bytes" in entry and path.stat().st_size != entry["bytes"]:
        raise ValueError("asset byte count differs")
    return path, relative


def verify_sources(fixtures, source_code):
    fixtures = Path(fixtures).resolve(strict=True)
    report = read_json(fixtures / "report.json")
    if report.get("schema") != "operator_media_seed_report.v1" or not report.get("generated_at", "").startswith("2026-"):
        raise ValueError("require actual 2026 operator media report")
    code_sha = sha(source_code)
    if report["source_code_sha256"] != code_sha:
        raise ValueError("generator source hash differs")
    rows_path = fixtures / "source_ready.jsonl"
    entry = report["artifacts"][rows_path.name]
    rows = read_rows(rows_path)
    if sha(rows_path) != entry["sha256"] or len(rows) != entry["rows"] or rows_path.stat().st_size != entry["bytes"]:
        raise ValueError("source rows differ from generation receipt")
    ids, groups, media_splits, source_ids, files = set(), {}, {}, set(), {}
    for row in rows:
        prov = row["provenance"]
        split = row["split"]
        if row["schema"] != "recovery_corpus.v1" or row["route"] not in ROUTES or split not in SPLITS:
            raise ValueError("unsupported source row route/schema/split")
        if row["id"] in ids or prov["split"] != split:
            raise ValueError("duplicate row or contradictory split")
        ids.add(row["id"])
        if (prov.get("source_year") != 2026 or prov.get("source_code_sha256") != code_sha or
                prov.get("synthetic_operator_authored") is not True or prov.get("actual_assets_generated") is not True):
            raise ValueError("source provenance differs from authored generator")
        if groups.setdefault(prov["group_id"], split) != split:
            raise ValueError("source family crosses holdout")
        for ident in prov["source_ids"]:
            if ident in source_ids:
                raise ValueError("duplicate source identity")
            source_ids.add(ident)
        if (row["supervision"] != {"policy": "assistant_only", "message_mask": [0, 1]} or
                [m["role"] for m in row["messages"]] != ["user", "assistant"]):
            raise ValueError("source assistant ownership changed")
        json.loads(row["messages"][1]["content"])
        quality = row["quality"]
        if not all(quality.get(k) is True for k in ("source_ready", "assets_verified", "groundtruth_verified")):
            raise ValueError("source or groundtruth is unverified")
        if quality.get("native_cache_status") != "pending":
            raise ValueError("this source planner requires unpromoted raw fixtures")
        for item in row["media"]:
            if item.get("source_verified") is not True or item.get("representation") != "raw" or item["role"] not in {"input", "conditioning", "target"}:
                raise ValueError("unverified media or ambiguous input/target role")
            path, _ = checked_file(item, fixtures)
            files[str(path)] = pin(path)
            if media_splits.setdefault(item["sha256"], split) != split:
                raise ValueError("media bytes cross holdout")
        gt_entry = {"path": prov["groundtruth_path"], "sha256": prov["groundtruth_sha256"]}
        path, _ = checked_file(gt_entry, fixtures)
        if read_json(path)["family"] != prov["template_family"]:
            raise ValueError("groundtruth belongs to another source family")
        files[str(path)] = pin(path)
    if (len(rows) != report["examples"] or Counter(r["route"] for r in rows) != report["routes"] or
            sum(len(r["media"]) for r in rows) != report["verified_assets"] or len(groups) != report["families"]):
        raise ValueError("report coverage differs from actual rows")
    return fixtures, rows, files, report


def source_item(row):
    """Retain conditioning and target separately; no answer enters model input."""
    return {"schema": "recovery_media_source_job_item.v1", "id": row["id"], "route": row["route"],
            "split": row["split"], "group_id": row["provenance"]["group_id"], "source_row_sha256": digest(row),
            "input": {"messages": [row["messages"][0]], "tools": row["tools"],
                      "media": [x for x in row["media"] if x["role"] in {"input", "conditioning"}]},
            "target_assets": [x for x in row["media"] if x["role"] == "target"],
            "groundtruth": {"path": row["provenance"]["groundtruth_path"], "sha256": row["provenance"]["groundtruth_sha256"]},
            "provenance": row["provenance"], "native_train_ready": False,
            "required_native_stage": ROUTES[row["route"]][0], "blockers": [ROUTES[row["route"]][1]]}


def heldout_task(row):
    item = source_item(row)
    if row["split"] == "train":
        raise ValueError("training rows cannot become heldout probes")
    answer = json.loads(row["messages"][1]["content"])
    return {"schema": "recovery_media_probe.v1", "id": row["id"], "route": row["route"], "split": row["split"],
            "group_id": item["group_id"], "input": item["input"],
            "oracle": {"kind": "procedural_media_groundtruth", "expected_answer": answer,
                       "target_assets": item["target_assets"], "groundtruth": item["groundtruth"],
                       "grading_status": "requires native recorded output + modality grader; no score inferred from source"},
            "source_row_sha256": item["source_row_sha256"], "synthetic_operator_authored": True,
            "model_evaluation_run": False}


def h3_item(row, fixtures, remote_root):
    assets = {x["type"]: x for x in row["media"]}
    if row["route"] != "joint_av" or set(assets) != {"video", "audio"} or row["split"] != "train":
        raise ValueError("H3 teacher jobs require train-only paired raw streams")
    video, audio = assets["video"], assets["audio"]
    if (video["role"] != "target" or audio["role"] != "target" or
            video.get("source_media_sha256") != video["sha256"] or audio.get("source_media_sha256") != video["sha256"] or
            any(x.get("start_seconds") != 0.0 or x.get("duration_seconds") != 1.0 for x in (video, audio)) or
            video.get("fps") != 8 or audio.get("sample_rate") != 16000 or not row.get("av_alignment", {}).get("same_origin")):
        raise ValueError("H3 streams lack exact common clock/origin/duration")
    _, video_rel = checked_file(video, fixtures)
    _, audio_rel = checked_file(audio, fixtures)
    window_end = 22 / 24
    frame_map = [min(7, (i * 8) // 24) for i in range(22)]
    truth = read_json(row["provenance"]["groundtruth_path"])["groundtruth"]
    adapted_events = []
    for original in truth["events"]:
        if original["start_seconds"] >= window_end:
            continue
        event = dict(original)
        event["duration_seconds"] = min(event["duration_seconds"], window_end - event["start_seconds"])
        visible = [i for i, source in enumerate(frame_map) if source == event["frame"]]
        adapted_events.append({"source_event": event, "native_visible_frames": visible,
                               "native_visual_onset_seconds": min(visible) / 24 if visible else None})
    prompt = row["messages"][0]["content"] + (
        "\nNative recovery view: generate only the first 22/24 seconds on 22 frames at 24 fps; "
        "preserve the requested source event times and sample each visual event by latest-frame hold.")
    # The real Stage-C parser accepts these id/prompt/video/audio keys. Extra
    # source pins retain provenance; they do not masquerade as latent receipts.
    return {"id": row["id"], "prompt": prompt, "source_prompt_sha256": digest(row["messages"][0]["content"]),
            "video": str(remote_root / video_rel.as_posix()), "audio": str(remote_root / audio_rel.as_posix()),
            "sha256": video["sha256"], "audio_sha256": audio["sha256"], "split": "train",
            "source_group_id": row["provenance"]["group_id"], "source_row_sha256": digest(row),
            "source_window": {"start_seconds": 0., "end_seconds": 22 / 24,
                              "source_duration_seconds": 1., "source_fps": 8, "native_fps": 24,
                              "source_frame_indices": frame_map,
                              "native_frames": 22, "spatial_size": [96, 96], "audio_source_rate": 16000,
                              "temporal_adaptation": "latest source frame at or before native time; native22/24s prefix"},
            "native_window_oracle": {"source_media_sha256": video["sha256"], "window_end_seconds": window_end,
                                     "events": adapted_events, "event_count": len(adapted_events),
                                     "raw_full_second_oracle_reused": False,
                                     "scope": "expected source events; actual teacher decode must still be measured"}}


def check_stage_c(source):
    """Read only: accept the native CLI contract, without importing its module."""
    raw = Path(source).read_text(encoding="utf-8-sig")
    tree = ast.parse(raw)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not {"run_h3_encode", "run_text", "run_velocity", "h3_align_frame_count", "read_items"}.issubset(names):
        raise ValueError("Stage-C H3 functions differ")
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    if any(constants.get(k) != v for k, v in {"H3_FPS": 24, "H3_SHIFT_VIDEO": 12., "H3_SHIFT_AUDIO": 3.}.items()):
        raise ValueError("native H3 clock or schedule differs")
    return pin(source)


def h3_jobs(remote_root, items_sha, source_pin, device):
    python = "/home/cereal/model-surgery-lab/venv/bin/python"
    tool = "/home/cereal/omni-campaign/distill/tools/stage_c_media_a.py"
    models = PurePosixPath("/opt/ComfyUI/models")
    cache = remote_root / "teacher_cache"
    items = str(remote_root / "h3_items.train.jsonl")
    tag, short = "recovery-fixtures-20261013", items_sha[:16]
    encode = cache / ("h3-vae-" + tag) / short
    text = cache / ("h3-te-qwen3vl_32b_minimax_h3_nvfp4_awq-" + tag) / short
    common = ["--items", items, "--cache-root", str(cache), "--cache-tag", tag,
              "--device", device, "--teacher-digest", "full"]
    commands = [
        ("h3-encode", [], ["--width", "96", "--height", "96", "--length", "22", "--start-seconds", "0",
                          "--max-drop-fraction", "0", "--items-per-shard", "8",
                          "--video-vae", str(models / "vae/minimax_h3_video_vae_fp16.safetensors"),
                          "--audio-vae", str(models / "vae/minimax_h3_audio_vae_fp32.safetensors")]),
        ("h3-text", [], ["--te", str(models / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
                        "--no-with-negatives", "--texts-per-shard", "8"]),
        ("h3-velocity", ["h3-encode", "h3-text"], ["--encode-cache", str(encode), "--te-cache", str(text),
             "--dit", str(models / "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"),
             "--weight-dtype", "float16", "--cfg", "1", "--sigmas-per-item", "4", "--shift", "12",
             "--sigma-mode", "stratified-shifted", "--seed", "20261013", "--items-per-shard", "8",
             "--block-pairs", "25", "--block-rows", "256", "--parity-every", "1"]),
    ]
    return [{"id": name, "kind": "teacher_cache_capture", "argv": [python, "-B", tool, name, *common, *extra],
             "environment": {"PYTHONDONTWRITEBYTECODE": "1", "COMFYUI_ROOT": "/opt/ComfyUI"},
             "depends_on": deps, "source_code": source_pin, "items_sha256": items_sha,
             "status": "prepared_not_launched", "training_started": False, "native_train_ready": False,
             "prelaunch_requirements": ["transfer_manifest paths/digests verified on target host",
                 "remote Stage-C source equals pinned local source", "actual teacher weights present + before/after full hashes",
                 "exclusive runtime assignment for selected device", "native parity/finite cache seals"],
             "post_capture_requirement": "native H3 adapter must bind packed positions, both streams, clocks and sigma12/3; no adapter implemented here"}
            for name, deps, extra in commands]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((canonical(value) + "\n").encode())


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(canonical(row) + "\n" for row in rows).encode())
    return {**pin(path), "rows": len(rows)}


def build(fixtures, source_code, stage_c_source, trainer_root, output, remote_root, frozen_prompts, device="cuda:0"):
    remote_root = PurePosixPath(remote_root)
    if not remote_root.is_absolute() or ".." in remote_root.parts or not re.fullmatch(r"cuda:[0-9]+", device):
        raise ValueError("require absolute target staging root and explicit CUDA device name")
    root, rows, files, report = verify_sources(fixtures, source_code)
    prompts = read_json(frozen_prompts)
    if not isinstance(prompts, list) or not all(isinstance(value, str) for value in prompts):
        raise ValueError("frozen prompts must be the actual string list")
    excluded = {normalize_text(value) for value in prompts}
    if report.get("frozen_eval_prompts_excluded") != len(excluded):
        raise ValueError("frozen exclusion bank count differs from fixture generation")
    if any(prompt_overlap(normalize_text(row["messages"][0]["content"]), excluded) for row in rows):
        raise ValueError("source prompt collides with frozen evaluation")
    stage_pin = check_stage_c(stage_c_source)
    runner_pins = {name: pin(Path(trainer_root) / "experiments/chimera_recovery" / name)
                   for name in ("training.py", "train_media.py", "media_cache.py", "ace_cache.py", "train_music.py")}
    output = Path(output)
    if output.exists():
        raise ValueError("retain earlier evidence; output must be new")
    output.mkdir(parents=True)
    artifacts, lanes = {}, {}
    for route, (stage, blocker) in ROUTES.items():
        selected = [r for r in rows if r["route"] == route]
        for split in sorted(SPLITS):
            name = f"sources/{route}.{split}.jsonl"
            artifacts[name] = write_rows(output / name, [source_item(r) for r in selected if r["split"] == split])
        lanes[route] = {"native_stage": stage, "raw_source_count": len(selected),
                        "counts": dict(Counter(r["split"] for r in selected)),
                        "native_cache_count": 0, "native_train_ready": False, "blockers": [blocker]}
    probes = [heldout_task(r) for r in rows if r["split"] != "train"]
    artifacts["heldout_media_probes.jsonl"] = write_rows(output / "heldout_media_probes.jsonl", probes)
    native_items = [h3_item(r, root, remote_root) for r in rows if r["route"] == "joint_av" and r["split"] == "train"]
    if any(prompt_overlap(normalize_text(item["prompt"]), excluded) for item in native_items):
        raise ValueError("adapted H3 conditioning collides with frozen evaluation")
    artifacts["h3_items.train.jsonl"] = write_rows(output / "h3_items.train.jsonl", native_items)
    transfer = []
    for row in rows:
        if row["route"] == "joint_av" and row["split"] == "train":
            for item in row["media"]:
                path, relative = checked_file(item, root)
                transfer.append({**pin(path), "remote_path": str(remote_root / relative.as_posix())})
    transfer.append({**pin(output / "h3_items.train.jsonl"), "remote_path": str(remote_root / "h3_items.train.jsonl")})
    write_json(output / "transfer_manifest.json", {"schema": "recovery_media_transfer.v1", "files": transfer,
        "transferred": False, "remote_verified": False, "never_execute_historical_commands": True})
    manifest = {"schema": "recovery_modality_job_plan.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_report": pin(root / "report.json"), "source_rows": pin(root / "source_ready.jsonl"),
                "generator_source": pin(source_code), "runner_sources": runner_pins, "planner_source": pin(__file__),
                "frozen_prompts": {**pin(frozen_prompts), "raw_count": len(prompts), "normalized_unique_count": len(excluded),
                                   "source_and_adapted_conditioning_rechecked": True,
                                   "screen": "exact/contained/0.85 trigram Jaccard lexical screen; no exhaustive semantic claim"},
                "verified_source_files": list(files.values()), "source_counts": report["routes"], "lanes": lanes,
                "artifacts": artifacts, "jobs": h3_jobs(remote_root, artifacts["h3_items.train.jsonl"]["sha256"], stage_pin, device),
                "heldout_groups_excluded_from_teacher_jobs": True, "heldout_probes": len(probes),
                "training_started": False, "model_performance_claim": False, "thermal_gate_active": False,
                "limitations": ["Synthetic fixtures supplement natural data; no speech/ASR or full 3D claim.",
                    "Source-ready flags are not native cache, runner, trained checkpoint or measured capability readiness.",
                    "H3 uses22 native frames at24fps from a1s8fps source; decode evaluation must apply this pinned prefix/resampling contract.",
                    "Heldout probe oracles are separate from input and still require actual recorded model/media output grading."]}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("fixtures", "source-code", "stage-c-source", "trainer-root", "output", "frozen-prompts"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = build(args.fixtures, args.source_code, args.stage_c_source, args.trainer_root, args.output,
                   args.remote_root, args.frozen_prompts, args.device)
    print(json.dumps({"source_counts": result["source_counts"], "heldout_probes": result["heldout_probes"],
                      "prepared_teacher_jobs": len(result["jobs"]), "native_train_ready": False, "training_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
