"""CPU procedural 2026 media fixtures with genuine assets and measured groundtruth.

Raw generative targets remain pending authentic native teacher caches. This does
not manufacture latent tensors, checkpoint provenance, ASR, or model scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw

try:
    from .memory_harvest import canonical, sha, normalize_text, prompt_overlap
except ImportError:
    from memory_harvest import canonical, sha, normalize_text, prompt_overlap


SIDE, FPS, SAMPLE_RATE, DURATION = 96, 8, 16000, 1.0
COLORS = {"red": (230, 35, 35), "green": (35, 210, 65), "blue": (35, 75, 230), "yellow": (235, 210, 35)}
FAMILIES = [
    ("edit_recolor", "image_edit", "train"), ("edit_translate", "image_edit", "validation"), ("edit_remove", "image_edit", "test"),
    ("av_red_flash", "joint_av", "train"), ("av_blue_flash", "joint_av", "validation"), ("av_green_flash", "joint_av", "test"),
    ("mesh_box", "volume_generate", "train"), ("mesh_octahedron", "volume_generate", "validation"), ("mesh_tetrahedron", "volume_generate", "test"),
    ("sound_tone", "audio_understand", "train"), ("sound_chirp", "audio_understand", "validation"), ("sound_noise", "audio_understand", "test"),
    ("image_rectangles", "vision_grounding", "train"), ("image_circles", "vision_grounding", "validation"), ("image_triangles", "vision_grounding", "test"),
    ("video_red_events", "video_understand", "train"), ("video_blue_events", "video_understand", "validation"), ("video_green_events", "video_understand", "test"),
]
GENERATIVE = {"image_edit", "joint_av", "volume_generate"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def asset(path, kind, role, **metadata):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size == 0:
        raise ValueError("empty_generated_asset")
    return {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size,
            "type": kind, "role": role, "representation": "raw", "source_verified": True, **metadata}


def shape_image(objects, background):
    image = Image.new("RGB", (SIDE, SIDE), background)
    draw = ImageDraw.Draw(image)
    for item in objects:
        x, y, radius, color = item["x"], item["y"], item.get("radius", 10), COLORS[item["color"]]
        if item["shape"] == "circle":
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        elif item["shape"] == "triangle":
            draw.polygon([(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)], fill=color)
        else:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)
        if image.getpixel((x, y)) != color:
            raise ValueError("shape_center_groundtruth_mismatch")
    return image


def image_fixture(directory, family, case_number, background):
    if family.startswith("edit_"):
        before = [{"shape": "triangle" if family == "edit_remove" else "circle" if family == "edit_translate" else "rectangle",
                   "color": "red", "x": 26 + case_number, "y": 38, "radius": 9},
                  {"shape": "circle", "color": "blue", "x": 72, "y": 72, "radius": 8}]
        after = [dict(item) for item in before]
        if family == "edit_recolor":
            after[0]["color"] = "green"
            instruction = f"Change only the red rectangle centered at ({before[0]['x']},38) to green. Keep its position and the blue circle unchanged."
        elif family == "edit_translate":
            after[0]["x"] += 30
            instruction = "Move the red circle 30 pixels right. Keep its color, size, vertical position, and the blue circle unchanged."
        else:
            after = after[1:]
            instruction = "Remove the red triangle and restore the background there. Preserve the blue circle."
        before_path, after_path = directory / "before.png", directory / "after.png"
        source, target = shape_image(before, background), shape_image(after, background)
        source.save(before_path)
        target.save(after_path)
        changed = np.any(np.asarray(source) != np.asarray(target), axis=2)
        if not changed.any() or not np.array_equal(np.asarray(source)[62:83, 62:83], np.asarray(target)[62:83, 62:83]):
            raise ValueError("image_edit_preservation_check_failed")
        points = np.argwhere(changed)
        truth = {"before_objects": before, "after_objects": after, "changed_pixels": int(changed.sum()),
                 "changed_bbox_xyxy": [int(points[:, 1].min()), int(points[:, 0].min()), int(points[:, 1].max()), int(points[:, 0].max())]}
        return instruction, truth, [asset(before_path, "image", "input"), asset(after_path, "image", "target")], {}
    shape = "rectangle" if family == "image_rectangles" else "circle" if family == "image_circles" else "triangle"
    count = 1 + case_number % 3
    objects = [{"shape": shape, "color": color, "x": 20 + i * 27, "y": 30 + case_number * 3, "radius": 8}
               for i, color in enumerate(("red", "green", "blue")[:count])]
    path = directory / "scene.png"
    shape_image(objects, background).save(path)
    truth = {"count": count, "objects": objects, "coordinate_system": "pixels; origin top-left"}
    return "Report the visible object count, shape, color, and center coordinates in pixels.", truth, [asset(path, "image", "input")], {}


def write_wav(path, samples):
    encoded = np.clip(np.rint(samples * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(encoded.tobytes())
    with wave.open(str(path), "rb") as stream:
        if stream.getframerate() != SAMPLE_RATE or stream.getnframes() != int(DURATION * SAMPLE_RATE):
            raise ValueError("waveform_duration_or_rate_mismatch")
        if stream.readframes(stream.getnframes()) != encoded.tobytes():
            raise ValueError("waveform_sample_roundtrip_mismatch")


def sound_samples(events, seed):
    samples = np.zeros(int(DURATION * SAMPLE_RATE), dtype=np.float64)
    generator = np.random.default_rng(seed)
    for event in events:
        first, last = int(event["start_seconds"] * SAMPLE_RATE), int((event["start_seconds"] + event["duration_seconds"]) * SAMPLE_RATE)
        t = np.arange(last - first) / SAMPLE_RATE
        if event["kind"] == "noise_burst":
            signal = generator.uniform(-0.35, 0.35, last - first)
        elif event["kind"] == "chirp":
            slope = (event["end_frequency_hz"] - event["frequency_hz"]) / event["duration_seconds"]
            signal = 0.35 * np.sin(2 * np.pi * (event["frequency_hz"] * t + slope * t * t / 2))
        else:
            signal = 0.35 * np.sin(2 * np.pi * event["frequency_hz"] * t)
        samples[first:last] += signal
    return samples


def verify_sound(samples, events, *, tolerance=0.02):
    occupied = np.zeros(len(samples), dtype=bool)
    measured = []
    for event in events:
        first, last = int(event["start_seconds"] * SAMPLE_RATE), int((event["start_seconds"] + event["duration_seconds"]) * SAMPLE_RATE)
        segment = samples[first:last]
        if np.max(np.abs(segment)) < 0.1:
            raise ValueError("sound_event_missing")
        occupied[max(0, first - 64):min(len(samples), last + 64)] = True
        item = {"start_seconds": event["start_seconds"], "peak": float(np.max(np.abs(segment)))}
        if event["kind"] == "tone":
            power = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
            frequency = float(np.fft.rfftfreq(len(segment), 1 / SAMPLE_RATE)[int(power.argmax())])
            if abs(frequency - event["frequency_hz"]) > 16:
                raise ValueError("observed_tone_frequency_mismatch")
            item["measured_frequency_hz"] = frequency
        measured.append(item)
    if occupied.any() and np.max(np.abs(samples[~occupied]), initial=0) > tolerance:
        raise ValueError("unexpected_sound_outside_events")
    return measured


def audio_fixture(directory, family, case_number, family_number):
    frequency = 440 + family_number * 64 + case_number * 8
    if family == "sound_tone":
        events = [{"kind": "tone", "start_seconds": 0.25, "duration_seconds": 0.25, "frequency_hz": frequency}]
    elif family == "sound_chirp":
        events = [{"kind": "chirp", "start_seconds": start, "duration_seconds": 0.125, "frequency_hz": frequency,
                   "end_frequency_hz": frequency + 320} for start in (0.125, 0.625)]
    else:
        events = [{"kind": "noise_burst", "start_seconds": start, "duration_seconds": 0.0625} for start in (0.125, 0.4375, 0.75)]
    samples = sound_samples(events, family_number * 1000 + case_number)
    measured = verify_sound(samples, events)
    path = directory / "sound.wav"
    write_wav(path, samples)
    truth = {"count": len(events), "events": events, "contains_speech": False, "measured_events": measured}
    prompt = "Identify the non-speech sound events, their count, and onset times. Report tone frequencies or chirp direction when applicable."
    return prompt, truth, [asset(path, "audio", "input", duration_seconds=DURATION, sample_rate=SAMPLE_RATE)], {}


def run(command, *, data=None):
    result = subprocess.run(command, input=data, capture_output=True, timeout=30, check=False)
    if result.returncode != 0:
        raise RuntimeError("media_tool_execution_failed")
    return result.stdout


def av_fixture(directory, family, case_number, family_number, background, ffmpeg, ffprobe):
    color = "red" if "red" in family else "blue" if "blue" in family else "green"
    events = [{"kind": "tone", "frame": frame, "start_seconds": frame / FPS, "duration_seconds": 1 / FPS,
               "frequency_hz": 440 + family_number * 64 + case_number * 8 + index * 32,
               "shape": "circle", "color": color, "x": 25 + index * 22, "y": 42 + case_number * 2}
              for index, frame in enumerate((1, 4, 6))]
    samples = sound_samples(events, family_number * 1000 + case_number)
    original_audio = directory / "source_sound.wav"
    write_wav(original_audio, samples)
    frames = []
    for frame in range(int(DURATION * FPS)):
        active = [event for event in events if event["frame"] == frame]
        frames.append(np.asarray(shape_image(active, background)))
    path = directory / "synchronized.mp4"
    run([ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{SIDE}x{SIDE}", "-framerate", str(FPS),
         "-i", "pipe:0", "-i", str(original_audio), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", str(SAMPLE_RATE), "-shortest", str(path)],
        data=np.stack(frames).astype(np.uint8).tobytes())
    probe = json.loads(run([ffprobe, "-v", "error", "-show_entries", "stream=codec_type,duration,start_time,avg_frame_rate,sample_rate,width,height", "-of", "json", str(path)]))
    streams = {stream["codec_type"]: stream for stream in probe["streams"]}
    if not {"audio", "video"}.issubset(streams) or any(abs(float(streams[kind]["duration"]) - DURATION) > 0.01 for kind in ("audio", "video")):
        raise ValueError("muxed_audio_video_duration_mismatch")
    numerator, denominator = map(int, streams["video"]["avg_frame_rate"].split("/"))
    if numerator / denominator != FPS or int(streams["audio"]["sample_rate"]) != SAMPLE_RATE or any(abs(float(streams[kind]["start_time"])) > 0.01 for kind in ("audio", "video")):
        raise ValueError("muxed_audio_video_clock_mismatch")
    decoded = run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    video = np.frombuffer(decoded, dtype=np.uint8).reshape((-1, SIDE, SIDE, 3))
    if len(video) != int(DURATION * FPS):
        raise ValueError("decoded_frame_count_mismatch")
    for event in events:
        observed = video[event["frame"], event["y"], event["x"]].astype(int)
        if np.max(np.abs(observed - np.asarray(COLORS[color]))) > 20:
            raise ValueError("decoded_visual_event_mismatch")
        for frame in range(len(video)):
            if frame not in {event["frame"] for event in events} and np.max(np.abs(video[frame,event["y"],event["x"]].astype(int) - np.asarray(background))) > 20:
                raise ValueError("unexpected_visual_event")
    extracted = directory / "aligned_sound.wav"
    run([ffmpeg, "-y", "-v", "error", "-i", str(path), "-map", "0:a:0", "-t", str(DURATION), "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(extracted)])
    with wave.open(str(extracted), "rb") as stream:
        decoded_audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(float) / 32767
    if len(decoded_audio) != int(DURATION * SAMPLE_RATE):
        raise ValueError("decoded_audio_length_mismatch")
    measured = verify_sound(decoded_audio, events, tolerance=0.05)
    origin = file_hash(path)
    role = "target" if family.startswith("av_") else "input"
    assets = [asset(path, "video", role, source_media_sha256=origin, start_seconds=0.0, fps=FPS, duration_seconds=DURATION, probe=probe)]
    if role == "target":
        assets.append(asset(extracted, "audio", "target", source_media_sha256=origin, start_seconds=0.0, sample_rate=SAMPLE_RATE, duration_seconds=DURATION))
    truth = {"count": len(events), "events": events, "frame_count": len(video), "measured_audio_events": measured,
             "audio_video_muxed": True, "same_origin_sha256": origin}
    alignment = {"av_alignment": {"source_verified": True, "same_origin": True, "duration_seconds": DURATION,
                                   "fps": FPS, "sample_rate": SAMPLE_RATE, "pairing": "decoded_same_muxed_interval",
                                   "visual_events_verified": True, "audio_events_verified": True}}
    prompt = (f"Create a synchronized one-second clip with {color} circle flashes at frames 1,4,6 (8 fps), at x=25,47,69 and y={42 + case_number * 2}; emit simultaneous 0.125-second tones with the requested event frequencies: " + canonical([event["frequency_hz"] for event in events])) if role == "target" else "Report the circle flash count, colors, center coordinates, and frame/onset time for each visible event."
    return prompt, truth, assets, alignment


def mesh_fixture(directory, family, case_number, background):
    scale = 1 + case_number / 16
    if family == "mesh_box":
        vertices = [(x * scale, y * 0.7, z * 0.9) for x, y, z in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
        faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(3,7,6),(3,6,2),(0,4,7),(0,7,3),(1,2,6),(1,6,5)]
    elif family == "mesh_octahedron":
        vertices = [(scale,0,0),(-scale,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        faces = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    else:
        vertices = [(scale,1,1),(-scale,-1,1),(-scale,1,-1),(scale,-1,-1)]
        faces = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
    path = directory / "geometry.obj"
    path.write_text("# operator authored 2026 procedural mesh\n" + "".join("v " + " ".join(map(str, point)) + "\n" for point in vertices) +
                    "".join("f " + " ".join(str(index + 1) for index in face) + "\n" for face in faces), encoding="utf-8")
    parsed_vertices = [tuple(map(float, line.split()[1:])) for line in path.read_text().splitlines() if line.startswith("v ")]
    parsed_faces = [tuple(int(value) - 1 for value in line.split()[1:]) for line in path.read_text().splitlines() if line.startswith("f ")]
    if parsed_vertices != vertices or parsed_faces != faces or any(not 0 <= index < len(vertices) for face in parsed_faces for index in face):
        raise ValueError("mesh_geometry_roundtrip_mismatch")
    edge_counts = Counter(tuple(sorted((face[i], face[(i + 1) % 3]))) for face in faces for i in range(3))
    if any(count != 2 for count in edge_counts.values()):
        raise ValueError("mesh_not_closed_manifold")
    points = np.asarray(vertices)
    volume = abs(sum(np.dot(points[a], np.cross(points[b], points[c])) for a,b,c in faces)) / 6
    expected_volume = {"mesh_box": 8 * scale * 0.7 * 0.9, "mesh_octahedron": 4 * scale / 3, "mesh_tetrahedron": 8 * scale / 3}[family]
    if not math.isclose(volume, expected_volume, rel_tol=1e-10):
        raise ValueError("mesh_volume_groundtruth_mismatch")
    yaw, elevation = math.radians(30), math.radians(20)
    projected = [(48 + 21 * (x * math.cos(yaw) - y * math.sin(yaw)),
                  48 - 21 * (z * math.cos(elevation) - (x * math.sin(yaw) + y * math.cos(yaw)) * math.sin(elevation))) for x,y,z in vertices]
    image = Image.new("RGB", (SIDE,SIDE), background)
    draw = ImageDraw.Draw(image)
    for number, face in enumerate(sorted(faces, key=lambda face: sum(vertices[index][1] for index in face))):
        draw.polygon([projected[index] for index in face], fill=(60 + number * 11, 120, 180), outline=(235,235,235))
    render = directory / "geometry_render.png"
    image.save(render)
    if np.all(np.asarray(image) == np.asarray(background)):
        raise ValueError("mesh_render_empty")
    truth = {"vertices": vertices, "faces_zero_based": faces, "vertex_count": len(vertices), "triangle_count": len(faces),
             "bbox_min": points.min(axis=0).tolist(), "bbox_max": points.max(axis=0).tolist(), "closed_manifold": True,
             "volume": float(volume), "coordinate_system": "right-handed Cartesian; arbitrary units",
             "render_camera": {"projection": "orthographic", "yaw_degrees": 30, "elevation_degrees": 20, "pixels_per_unit": 21}}
    return (f"Create the closed {family.removeprefix('mesh_')} mesh shown in this orthographic reference, with x scale {scale}.", truth,
            [asset(render, "image", "conditioning"), asset(path, "mesh", "target")], {})


def build_fixture(out, family_number, case_number, ffmpeg, ffprobe):
    family, route, split = FAMILIES[family_number]
    identity = sha(["operator-media-seed-2026.v1", family, case_number])[:24]
    directory = Path(out) / "assets" / identity
    directory.mkdir(parents=True, exist_ok=False)
    background = tuple(12 + int(identity[i:i + 2], 16) % 20 for i in (0,2,4))
    if route in {"image_edit", "vision_grounding"}:
        prompt, truth, assets, extras = image_fixture(directory, family, case_number, background)
    elif route == "audio_understand":
        prompt, truth, assets, extras = audio_fixture(directory, family, case_number, family_number)
    elif route == "volume_generate":
        prompt, truth, assets, extras = mesh_fixture(directory, family, case_number, background)
    else:
        prompt, truth, assets, extras = av_fixture(directory, family, case_number, family_number, background, ffmpeg, ffprobe)
    manifest = directory / "groundtruth.json"
    manifest.write_text(canonical({"family": family, "case": case_number, "groundtruth": truth}) + "\n", encoding="utf-8")
    generative = route in GENERATIVE
    if route == "audio_understand":
        response = {key: truth[key] for key in ("count", "events", "contains_speech")}
    elif route == "video_understand":
        response = {"count": truth["count"], "events": [{key: event[key] for key in ("frame", "start_seconds", "shape", "color", "x", "y")} for event in truth["events"]]}
    elif route == "vision_grounding":
        response = {"count": truth["count"], "objects": [{key: item[key] for key in ("shape", "color", "x", "y")} for item in truth["objects"]]}
    elif route == "image_edit":
        response = {"after_objects": [{key: item[key] for key in ("shape", "color", "x", "y")} for item in truth["after_objects"]]}
    elif route == "volume_generate":
        response = {key: truth[key] for key in ("vertex_count", "triangle_count", "closed_manifold", "bbox_min", "bbox_max")}
    else:
        response = {"count": truth["count"], "events": truth["events"]}
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    row = {"schema": "recovery_corpus.v1", "id": "media-seed-" + identity, "route": route, "split": split,
           "messages": [{"role": "user", "content": f"Procedural fixture {identity}. " + prompt}, {"role": "assistant", "content": canonical(response)}],
           "tools": [], "media": assets,
           "supervision": {"policy": "assistant_only", "message_mask": [0,1]}, "supervised_assistant_messages": [1],
           "provenance": {"source": "operator_authored_procedural_media_2026", "source_verified": True, "source_year": 2026,
                          "source_ids": ["operator-media-seed-2026.v1:" + family + ":" + str(case_number)],
                          "source_code_sha256": source_sha, "synthetic_operator_authored": True,
                          "template_family": family, "case_id": identity, "group_id": "media-seed-family-" + family, "split": split,
                          "groundtruth_path": str(manifest.resolve()), "groundtruth_sha256": file_hash(manifest),
                          "actual_assets_generated": True, "benchmark_claim": False},
           "quality": {"source_ready": True, "assets_verified": True, "groundtruth_verified": True,
                       "train_ready": not generative, "native_cache_status": "pending", "content_review": "procedural_groundtruth_checks",
                       "readiness_scope": "raw assets and authored groundtruth; authentic native parent processing remains required"}, **extras}
    return row


def generate(out, examples=144, excluded_prompts=None):
    if not 128 <= examples <= 256 or datetime.now(timezone.utc).year != 2026:
        raise ValueError("require_128_to_256_examples_and_2026_source_year")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("existing_ffmpeg_and_ffprobe_required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, ownership = [], {}
    for number in range(examples):
        row = build_fixture(out, number % len(FAMILIES), number // len(FAMILIES), ffmpeg, ffprobe)
        if excluded_prompts and prompt_overlap(normalize_text(row["messages"][0]["content"]), excluded_prompts):
            raise ValueError("frozen_eval_prompt_overlap")
        for item in row["media"]:
            previous = ownership.setdefault(item["sha256"], row["split"])
            if previous != row["split"]:
                raise ValueError("cross_split_media_identity")
        rows.append(row)
    artifacts = {}
    for filename, values in (("source_ready.jsonl", rows), ("understanding_ready.jsonl", [row for row in rows if row["quality"]["train_ready"]])):
        data = "".join(canonical(row) + "\n" for row in values).encode()
        (out / filename).write_bytes(data)
        artifacts[filename] = {"sha256": hashlib.sha256(data).hexdigest(), "rows": len(values), "bytes": len(data)}
    report = {"schema": "operator_media_seed_report.v1", "generated_at": datetime.now(timezone.utc).isoformat(),
              "examples": len(rows), "routes": dict(Counter(row["route"] for row in rows)), "split_counts": dict(Counter(row["split"] for row in rows)),
              "families": len(FAMILIES), "verified_assets": sum(len(row["media"]) for row in rows), "cross_split_media_conflicts": 0,
              "generative_native_cache_pending": sum(row["route"] in GENERATIVE for row in rows), "artifacts": artifacts,
              "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "frozen_eval_prompts_excluded": len(excluded_prompts or []),
              "limitations": ["Synthetic procedural scenes and non-speech sound events; no natural-media or ASR coverage claim.",
                              "Actual geometry and muxed synchronized audio/video; no claim of native model capability.",
                              "Generative targets require actual native teacher/cache provenance before training; none is fabricated."]}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=144)
    parser.add_argument("--exclude-prompts", type=Path)
    args = parser.parse_args(argv)
    try:
        excluded = None
        if args.exclude_prompts:
            prompts = json.loads(args.exclude_prompts.read_bytes())
            if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
                raise ValueError("frozen_prompts_must_be_string_list")
            excluded = {normalize_text(prompt) for prompt in prompts}
        print(json.dumps(generate(args.out, args.examples, excluded), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
