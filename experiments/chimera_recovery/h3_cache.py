"""H3 joint cache preflight and a pinned, separate Stage-C writer candidate.

No teacher is loaded by the CLI. The numerical adapter imports Torch lazily and
accepts recorded tensors only. A passing capture preflight is not admission of a
native training batch: parent identity, native prefix and bridge positions must
also be pinned by the production driver. Existing Stage-C v1 seals fail closed.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

STAGE_C_SHA256 = "fa936d72ebb0545acc3d2d2ac8e2bdc4477f370c73947352a0f20bcb23718fdf"
H3_MODEL_SHA256 = "882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024"
CONTRACT = "chimera_recovery.h3_joint_capture.v1"
SOURCE_CONTRACT = "chimera_recovery.h3_same_mux_window.v1"
FEATURE_BANDS = {2: (384, 480), 3: (480, 512)}
EMISSION_BANDS = {2: (128, 224), 3: (224, 256)}
REQUIRED_TENSORS = {"x_t_video", "x_t_audio", "v_video", "v_audio", "v_audio_native",
                    "x0hat_video", "x0hat_audio", "sigma", "sigma_audio", "slope_audio",
                    "seed", "target_item", "item.index", "item.x0_video", "item.x0_audio"}


class Refusal(ValueError):
    pass


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Refusal(f"{label}: expected lowercase SHA256")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise Refusal(f"{label}: finite number required")
    return float(value)


def schedule(sv: float) -> tuple[float, float]:
    sv = finite(sv, "sigma_video")
    if not 0 < sv < 1:
        raise Refusal("sigma_video must be in (0,1)")
    base = sv / (12.0 + sv * (1.0 - 12.0))
    sa = 3.0 * base / (1.0 + 2.0 * base)
    slope = 3.0 * (1.0 + 11.0 * base) ** 2 / (12.0 * (1.0 + 2.0 * base) ** 2)
    return sa, slope


def validate_window(window: dict[str, Any]) -> None:
    if not isinstance(window, dict) or window.get("native_frames") != 22 or window.get("native_fps") != 24:
        raise Refusal("this bounded writer contract supports the actual 22-frame/24-fps fixture window only")
    start = finite(window.get("start_seconds"), "window.start_seconds")
    end = finite(window.get("end_seconds"), "window.end_seconds")
    # This first writer refuses arbitrary origin shifts rather than guessing a
    # relation between the video's rebased clock and a separate audio clock.
    if start != 0 or not math.isclose(end, 22 / 24, rel_tol=0, abs_tol=1e-12):
        raise Refusal("bounded same-mux window must be [0,22/24) seconds")
    if window.get("spatial_size") != [96, 96] or window.get("source_fps") != 8:
        raise Refusal("bounded fixture must retain the measured 96px/8fps source geometry")
    if finite(window.get("source_duration_seconds"), "source_duration") < end:
        raise Refusal("source shorter than requested joint window")
    expected = [int(math.floor((i / 24) * 8 + 1e-10)) for i in range(22)]
    if window.get("source_frame_indices") != expected:
        raise Refusal("source-to-native latest-frame resampling map differs")


def strict_item(item: dict[str, Any]) -> dict[str, Any]:
    validate_window(item.get("source_window"))
    if item.get("split") != "train":
        raise Refusal("held-out item must not enter a teacher training capture")
    source_sha = require_sha(item.get("sha256"), "item.sha256")
    # Use the same physical mux for both streams. An unrelated WAV assignment
    # cannot meet this contract even if its rates and duration happen to match.
    out = dict(item)
    out["audio"] = item["video"]
    out["audio_sha256"] = source_sha
    out["h3_source_contract"] = {
        "schema": SOURCE_CONTRACT, "same_mux": True,
        "video_sha256": source_sha, "audio_sha256": source_sha,
        "source_row_sha256": require_sha(item.get("source_row_sha256"), "source_row_sha256"),
        "source_group_id": item["source_group_id"], "split": "train",
        "window": item["source_window"], "window_sha256": digest(item["source_window"]),
        "audio_source": "decode the audio stream of the identical physical video file",
        "padding_policy": "no waveform padding or silent fallback; record latent padding separately",
    }
    return out


def prepare_items(planner_manifest: Path, out: Path) -> dict[str, Any]:
    manifest = json.loads(planner_manifest.read_text(encoding="utf-8"))
    item_ref = manifest["artifacts"]["h3_items.train.jsonl"]
    item_path = Path(item_ref["path"])
    if sha(item_path) != require_sha(item_ref["sha256"], "planned items SHA"):
        raise Refusal("planned H3 item bytes changed")
    pins = manifest.get("verified_source_files", [])
    # The planner contains the source file seal inventory. Do not infer source
    # authenticity from operator-written SHA fields in an arbitrary input row.
    if not pins:
        pins = manifest.get("source_files_verified", [])
    by_sha = {p["sha256"]: p for p in pins if isinstance(p, dict) and "sha256" in p and "path" in p}
    rows = [json.loads(line) for line in item_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 8:
        raise Refusal("expected the planner's eight training joint fixtures")
    for row in rows:
        pin = by_sha.get(row["sha256"])
        if pin is None or sha(pin["path"]) != row["sha256"]:
            raise Refusal(f"{row['id']}: source media absent from verified planner inventory")
    adapted = [strict_item(row) for row in rows]
    if out.exists() or out.with_suffix(out.suffix + ".receipt.json").exists():
        raise Refusal("refusing to overwrite prepared item file")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"".join(canonical(row) + b"\n" for row in adapted))
    receipt = {"schema": "chimera_recovery.h3_prepared_items.v1", "planner_manifest_sha256": sha(planner_manifest),
               "source_items_sha256": sha(item_path), "prepared_items_sha256": sha(out), "items": len(adapted),
               "same_physical_mux_for_audio": True, "window": "[0,22/24) at 24fps",
               "teacher_jobs_launched": False, "native_train_ready": False}
    out.with_suffix(out.suffix + ".receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        first = stream.read(8)
        if len(first) != 8:
            raise Refusal("short safetensors header")
        n = struct.unpack("<Q", first)[0]
        if n > 8 << 20 or n < 2:
            raise Refusal("unbounded safetensors header")
        raw = stream.read(n)
    if len(raw) != n:
        raise Refusal("short safetensors JSON")
    header = json.loads(raw)
    end = path.stat().st_size - 8 - n
    spans = []
    sizes = {"F32": 4, "I64": 8, "BF16": 2, "F16": 2, "I8": 1, "BOOL": 1}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise Refusal(f"invalid tensor shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(d) is not int for d in offsets):
            raise Refusal(f"invalid tensor offsets: {name}")
        lo, hi = offsets
        if entry.get("dtype") not in sizes or not 0 <= lo <= hi <= end or hi - lo != math.prod(shape) * sizes[entry["dtype"]]:
            raise Refusal(f"invalid tensor byte range: {name}")
        spans.append((lo, hi))
    cursor = 0
    for lo, hi in sorted(spans):
        if lo != cursor:
            raise Refusal("overlapping or unaccounted tensor bytes")
        cursor = hi
    if cursor != end:
        raise Refusal("unaccounted safetensors payload")
    return header


def check_source_record(record: dict[str, Any]) -> None:
    if record.get("schema") != SOURCE_CONTRACT or record.get("same_mux") is not True:
        raise Refusal("missing measured same-mux source record")
    if require_sha(record.get("video_sha256"), "video origin") != require_sha(record.get("audio_sha256"), "audio origin"):
        raise Refusal("joint streams originate from different physical files")
    validate_window(record.get("window"))
    if digest(record["window"]) != record.get("window_sha256"):
        raise Refusal("source window/resampling map moved")
    measured = record.get("measured", {})
    if measured.get("video_frame_indices") != record["window"]["source_frame_indices"]:
        raise Refusal("actual video picks differ from requested resampling map")
    times = measured.get("video_target_times", [])
    if len(times) != 22 or any(not math.isclose(finite(t, "target timestamp"), i / 24, abs_tol=1e-9) for i, t in enumerate(times)):
        raise Refusal("actual native frame times differ")
    source_times = measured.get("video_source_times", [])
    audio_pts = measured.get("audio_source_frame_pts", [])
    if (len(source_times) < 8 or len(audio_pts) < 1 or
            any(finite(b, "source PTS") <= finite(a, "source PTS") for a, b in zip(source_times, source_times[1:])) or
            any(finite(b, "audio PTS") <= finite(a, "audio PTS") for a, b in zip(audio_pts, audio_pts[1:]))):
        raise Refusal("full monotonic source timestamp evidence missing")
    if type(measured.get("audio_source_rate")) is not int or measured["audio_source_rate"] < 1:
        raise Refusal("original audio sample clock missing")
    if measured.get("audio_rate") != 32000 or measured.get("audio_samples") != round(22 / 24 * 32000):
        raise Refusal("actual resampled audio duration differs")
    if measured.get("waveform_padded_samples") != 0 or measured.get("audio") != "decoded":
        raise Refusal("silent fallback or waveform padding cannot certify sync")
    if abs(finite(measured.get("video_first_pts_seconds"), "video first PTS") - finite(measured.get("audio_first_pts_seconds"), "audio first PTS")) > 1 / 32000:
        raise Refusal("audio/video origins differ beyond one resampled sample")
    if measured.get("audio_discontinuities") != 0:
        raise Refusal("audio PTS discontinuities require an explicit alignment implementation")
    if measured.get("minimax_model_sha256") != H3_MODEL_SHA256:
        raise Refusal("current MiniMax parent source differs from chart recipe")


def preflight(directory: Path, writer_sha256: str | None = None) -> dict[str, Any]:
    problems: list[str] = []
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise Refusal("teacher cache has no sealed manifest.json")
    manifest_sha = sha(manifest_path)
    seal = directory / "manifest.json.sha256"
    if not seal.is_file() or seal.read_text().strip() != manifest_sha:
        raise Refusal("teacher manifest seal missing or changed")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if writer_sha256 is None or m.get("tool", {}).get("sha256") != require_sha(writer_sha256, "reviewed writer SHA"):
        problems.append("capture writer is not bound to the reviewed candidate SHA")
    if m.get("kind") != "h3-velocity" or m.get("h3_joint_contract") != CONTRACT:
        problems.append("legacy H3 cache lacks the strict joint capture contract")
    if m.get("all_finite") is not True:
        problems.append("teacher cache lacks all-finite seal")
    integrity = m.get("teacher_integrity", {})
    if integrity.get("mode") != "full" or integrity.get("unchanged") is not True:
        problems.append("full before/after teacher integrity missing")
    before, after = integrity.get("before", {}), integrity.get("after", {})
    if not before or set(before) != set(after):
        problems.append("teacher weight inventory missing or changed")
    else:
        for path, b in before.items():
            a = after[path]
            if (not isinstance(b, dict) or not isinstance(a, dict) or
                    any(b.get(k) != a.get(k) for k in ("sha256", "bytes", "mtime_ns", "inode"))):
                problems.append("teacher weight seal differs: " + path)
            try:
                require_sha(b.get("sha256"), "full teacher weight SHA")
            except (Refusal, AttributeError) as e:
                problems.append(str(e))
    for label in ("encode_cache", "te_cache"):
        try:
            require_sha(m.get(label, {}).get("manifest_sha256"), label + " dependency seal")
        except Refusal as e:
            problems.append(str(e))
    if not m.get("parity_arm") or any(r.get("pass") is not True for r in m.get("parity_arm", [])):
        problems.append("passing actual sampler/direct parity evidence missing")
    shards = m.get("shards", [])
    if not shards or len(shards) > 4096 or m.get("read_order") != [r.get("file") for r in shards]:
        problems.append("bounded unique shard order missing")
    targets = 0
    for row in shards[:4096]:
        try:
            name = row["file"]
            if Path(name).name != name or not name.endswith(".safetensors"):
                raise Refusal("unsafe shard file path")
            p = directory / name
            side = p.with_suffix(".json")
            if sha(p) != require_sha(row.get("sha256"), "tensor shard SHA") or p.stat().st_size != row.get("bytes"):
                raise Refusal("tensor file hash or size moved")
            if sha(side) != require_sha(row.get("sidecar_sha256"), "sidecar SHA"):
                raise Refusal("unsealed target/source shard sidecar")
            s = json.loads(side.read_text(encoding="utf-8"))
            if s.get("sha256") != row["sha256"] or s.get("finite") is not True:
                raise Refusal("sidecar tensor identity/finite seal inconsistent")
            h = safetensors_header(p)
            missing = REQUIRED_TENSORS - h.keys()
            if missing:
                raise Refusal("actual jointly captured tensors missing: " + ",".join(sorted(missing)))
            n = h["sigma"]["shape"][0]
            if n < 1 or len(s.get("targets", [])) != n:
                raise Refusal("target count differs between tensor/sidecar")
            vshape, ashape = h["x_t_video"]["shape"], h["x_t_audio"]["shape"]
            if vshape != [n, 24, 7, 12, 12] or ashape != [n, 32, 2, 37]:
                raise Refusal("captured joint tensor geometry differs")
            ishape = h["item.index"]["shape"]
            if (len(ishape) != 1 or ishape[0] < 1 or
                    h["item.x0_video"]["shape"] != [ishape[0], 24, 7, 12, 12] or
                    h["item.x0_audio"]["shape"] != [ishape[0], 32, 2, 37]):
                raise Refusal("model-space source geometry differs")
            for key in ("v_video", "x0hat_video"):
                if h[key]["shape"] != vshape:
                    raise Refusal("video pair shape mismatch")
            for key in ("v_audio", "v_audio_native", "x0hat_audio"):
                if h[key]["shape"] != ashape:
                    raise Refusal("audio pair shape mismatch")
            for key in ("sigma", "sigma_audio", "slope_audio", "seed", "target_item"):
                if h[key]["shape"] != [n]:
                    raise Refusal("per-target scalar shape mismatch")
            for key in REQUIRED_TENSORS:
                expected_dtype = "I64" if key in {"seed", "target_item", "item.index"} else "F32"
                if h[key]["dtype"] != expected_dtype:
                    raise Refusal("captured dtype differs: " + key)
            cfg = finite(m.get("plan", {}).get("cfg"), "cfg")
            if cfg != 1 and not {"v_cond_video", "v_cond_audio", "v_uncond_video", "v_uncond_audio"}.issubset(h):
                raise Refusal("guided capture lacks actual conditional/unconditional branches")
            for target in s["targets"]:
                require_sha(target.get("noise_sha256"), "actual noise digest")
                schedule(target.get("sigma"))
                check_source_record(target.get("h3_source_contract", {}))
            targets += n
        except (Refusal, KeyError, OSError, TypeError, IndexError) as e:
            problems.append(f"{row.get('file', '?')}: {e}")
    return {"schema": "chimera_recovery.h3_cache_preflight.v1", "cache_manifest_sha256": manifest_sha,
            "capture_contract_ok": not problems, "targets": targets, "problems": problems,
            "native_train_ready": False, "remaining_native_requirements": [
                "numerically validate every captured noisy/velocity pair with the exact model-space clean/noise join",
                "bind genuine native parent identity and authentic chat prefix token IDs",
                "bind measured native bridge positions and full text|audio|video charted_source_positions",
                "seal converted batches, source group split and held-out exclusions through the production admission driver"]}


def tensor_digest(t: Any) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).replace("torch.", "").encode())
    h.update(json.dumps(list(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def validate_pair(tensors: dict[str, Any], target: dict[str, Any], target_row: int, cfg: float) -> dict[str, Any]:
    """Validate actual teacher values; not a replacement for cache/source seals."""
    import torch
    if not REQUIRED_TENSORS.issubset(tensors):
        raise Refusal("missing captured H3 joint tensors")
    if any(not bool(torch.isfinite(t).all()) for t in tensors.values() if t.is_floating_point()):
        raise Refusal("non-finite captured tensor")
    indices = tensors["item.index"].tolist()
    if len(indices) != len(set(indices)):
        raise Refusal("duplicate source item indices")
    item = int(tensors["target_item"][target_row])
    if item != target["item_index"] or item not in indices or int(tensors["seed"][target_row]) != target["seed"]:
        raise Refusal("target/source/noise join differs")
    source_row = indices.index(item)
    xv0 = tensors["item.x0_video"][source_row].detach().cpu().unsqueeze(0).float()
    xa0 = tensors["item.x0_audio"][source_row].detach().cpu().unsqueeze(0).float()
    if xv0.ndim != 5 or xv0.shape[0] != 1 or xv0.shape[1] != 24 or xv0.shape[3] % 2 or xv0.shape[4] % 2:
        raise Refusal("H3 video must be [1,24,T,evenH,evenW]")
    if xa0.ndim != 4 or xa0.shape[:3] != (1, 32, 2):
        raise Refusal("H3 audio must be [1,32,2,T]")
    sv = target["sigma"]  # exact Python sigma, not its rounded float32 tensor
    sa, slope = schedule(sv)
    g = torch.Generator(device="cpu").manual_seed(int(target["seed"]))
    noises = [torch.randn(tuple(x.shape), generator=g, dtype=torch.float32) for x in (xv0, xa0)]
    noise_sha = hashlib.sha256("".join(tensor_digest(n) for n in noises).encode()).hexdigest()
    if noise_sha != target["noise_sha256"]:
        raise Refusal("CPU video-then-audio noise digest differs")
    def observed(name: str):
        return tensors[name][target_row].detach().cpu().unsqueeze(0).float()
    for name, expected in (("x_t_video", (1 - sv) * xv0 + sv * noises[0]),
                           ("x_t_audio", (1 - sa) * xa0 + sa * noises[1])):
        if not torch.equal(observed(name), expected):
            raise Refusal(f"actual {name} differs from sealed source/seed/schedule")
    for name, value in (("sigma", sv), ("sigma_audio", sa), ("slope_audio", slope)):
        if not torch.equal(tensors[name][target_row].float(), torch.tensor(value, dtype=torch.float32)):
            raise Refusal(f"recorded {name} differs from exact dual-clock schedule")
    va = (observed("v_audio").double() / slope).float()
    if not torch.equal(observed("v_audio_native"), va):
        raise Refusal("native audio velocity differs from flat/slope conversion")
    checks = ((observed("x_t_video").double() - sv * observed("v_video").double(), observed("x0hat_video")),
              (observed("x_t_audio").double() - sa * va.double(), observed("x0hat_audio")))
    for reconstructed, captured in checks:
        if not torch.allclose(reconstructed.float(), captured, atol=2e-6, rtol=2e-6):
            raise Refusal("canonical guided velocity does not reproduce actual x0hat")
    if not math.isclose(cfg, 1.0):
        for stream in ("video", "audio"):
            cond, uncond = observed("v_cond_" + stream), observed("v_uncond_" + stream)
            # CFG is performed on denoised in fp32; roundoff means this flat
            # velocity identity is approximate rather than bit-identical.
            if not torch.allclose(observed("v_" + stream), uncond + cfg * (cond - uncond), atol=2e-5, rtol=2e-5):
                raise Refusal("guided target differs from actual CFG branches")
    return {"video_x_t": observed("x_t_video"), "audio_x_t": observed("x_t_audio"),
            "video_velocity": observed("v_video").detach(), "audio_velocity": va.detach(),
            "sigma_video": sv, "sigma_audio": sa, "slope_audio": slope, "noise_sha256": noise_sha}


def chart_positions(prefix: int, tv: int, h: int, w: int, ta: int):
    """Exact s8 PackedLayout transcription, plain t2va/no refs; not timestamps."""
    import torch
    if any(type(n) is not int or n < 1 for n in (prefix, tv, h, w, ta)) or h % 2 or w % 2:
        raise Refusal("invalid full-sequence chart geometry")
    area = math.sqrt(h * w)
    def axis(dim):
        ratio, n = dim / area, dim // 2
        return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1 - ratio) / 2) * 32
    hh, ww = torch.meshgrid(axis(h), axis(w), indexing="ij")
    frame = torch.stack([hh.flatten(), ww.flatten()], dim=-1)
    text = torch.zeros(prefix, 3, dtype=torch.float64)
    text[:, 0] = torch.arange(prefix, dtype=torch.float64)
    audio = torch.zeros(2 * ta, 3, dtype=torch.float64)
    audio[:, 0] = (prefix + torch.arange(ta, dtype=torch.float64)).repeat(2)
    audio[:ta, 2], audio[ta:, 2] = float(axis(w)[0]), float(axis(w)[-1])
    spans = torch.tensor([(5 / 3) * (1, 4, 4, 4, 4)[k % 5] for k in range(tv)], dtype=torch.float64)
    t_grid = prefix + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])
    video = torch.empty(tv, frame.shape[0], 3, dtype=torch.float64)
    video[:, :, 0], video[:, :, 1:] = t_grid[:, None], frame[None]
    return torch.cat([text, audio, video.reshape(-1, 3)]).float().unsqueeze(0)


def pack_pair(pair: dict[str, Any], prefix_ids: Any, bridge_positions: Any, *, media_token_id: int,
              feature_dim: int = 6144) -> dict[str, Any]:
    """Numerical native layout adapter only. Caller must prove/seal provenance.

    Bridge coordinates are deliberately supplied rather than inferred from the
    old pair bank's ASSUMED four-frame temporal stride. They are separate from
    the authentic H3 RoPE chart and must be measured/pinned by the driver.
    """
    import torch
    if prefix_ids.dtype != torch.int64 or prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] < 1:
        raise Refusal("authentic prefix must be int64 [1,P]")
    if feature_dim != 6144 or type(media_token_id) is not int or media_token_id < 0:
        raise Refusal("native parent feature/media-token contract differs")
    def video(x):
        b, c, t, h, w = x.shape
        if (b, c) != (1, 24) or h % 2 or w % 2:
            raise Refusal("invalid H3 video patch geometry")
        return torch.einsum("nctrhpwq->nthwcrpq", x.reshape(b, c, t, 1, h // 2, 2, w // 2, 2)).reshape(-1, 96)
    def audio(x):
        if x.ndim != 4 or x.shape[:3] != (1, 32, 2):
            raise Refusal("invalid H3 stereo audio geometry")
        return x[0].permute(1, 2, 0).reshape(-1, 32)
    av, vv = audio(pair["audio_x_t"]), video(pair["video_x_t"])
    at, vt = audio(pair["audio_velocity"]), video(pair["video_velocity"])
    na, nv, p = len(av), len(vv), prefix_ids.shape[1]
    m = na + nv
    if bridge_positions.shape != (1, m, 4) or not bool(torch.isfinite(bridge_positions).all()):
        raise Refusal("measured native bridge positions must be finite [1,M,4]")
    features = torch.zeros(1, m, feature_dim, dtype=torch.float32)
    targets = torch.zeros_like(features)
    channels = torch.zeros_like(features, dtype=torch.bool)
    features[0, :na, 480:512], features[0, na:, 384:480] = av, vv
    targets[0, :na, 224:256], targets[0, na:, 128:224] = at, vt
    channels[0, :na, 224:256], channels[0, na:, 128:224] = True, True
    types = torch.tensor([[3] * na + [2] * nv], dtype=torch.int64)
    ids = torch.cat([prefix_ids.detach().cpu(), torch.full((1, m), media_token_id, dtype=torch.int64)], dim=1)
    sigma = torch.cat([torch.full((na,), pair["sigma_audio"], dtype=torch.float32),
                       torch.full((nv,), pair["sigma_video"], dtype=torch.float32)]).view(1, m)
    _, _, tv, h, w = pair["video_x_t"].shape
    chart = chart_positions(p, tv, h, w, pair["audio_x_t"].shape[-1])
    if chart.shape != (1, ids.shape[1], 3):
        raise Refusal("chart must cover prefix AND both streams")
    return {"input_ids": ids, "native_media_features": features,
            "native_media_targets": targets.detach(), "native_media_type_ids": types,
            "native_media_positions": bridge_positions.detach().cpu().clone().float(),
            "native_media_indices": torch.arange(p, p + m, dtype=torch.int64).view(1, m),
            "native_media_valid_mask": torch.ones(1, m, dtype=torch.bool),
            "native_media_noise_levels": sigma, "native_media_loss_mask": torch.ones(1, m, dtype=torch.float32),
            "native_media_loss_weights": torch.ones(1, m, dtype=torch.float32),
            "native_media_channel_loss_mask": channels, "charted_source_positions": chart}


# This helper is inserted into a private candidate of the pinned Stage-C source.
# It executes only inside an explicitly launched H3 capture. It refuses clock
# ambiguity rather than declaring an arbitrary decoded pair synchronized.
WRITER_HELPERS = '''
H3_JOINT_CONTRACT = "chimera_recovery.h3_joint_capture.v1"
H3_EXPECTED_MODEL_SHA256 = "882c280b05aa60cd17f231e5e2389b921b52880fb3b4e23574c0aba5f2bd5024"

def h3_joint_source(item, geo, args):
    c = item.get("h3_source_contract", {})
    w = c.get("window", {})
    if c.get("schema") != "chimera_recovery.h3_same_mux_window.v1" or c.get("same_mux") is not True:
        raise SystemExit("H3 strict capture needs a prepared same-mux source contract")
    if str(Path(item["audio"]).resolve()) != str(Path(item["video"]).resolve()):
        raise SystemExit("H3 audio must come from the identical physical mux")
    actual = sha256_file(item["video"])
    if actual != c.get("video_sha256") or actual != c.get("audio_sha256") or actual != item.get("sha256"):
        raise SystemExit("H3 source mux hash differs")
    if w.get("native_frames") != 22 or w.get("native_fps") != 24 or w.get("spatial_size") != [96, 96]:
        raise SystemExit("this bounded H3 writer supports 22 frames at 24fps, 96px only")
    if args.start_seconds != 0 or args.length != 22 or args.width != 96 or args.height != 96:
        raise SystemExit("H3 argv differs from the prepared physical window")
    if w.get("start_seconds") != 0 or not math.isclose(w.get("end_seconds", -1), geo["seconds"], abs_tol=1e-12):
        raise SystemExit("H3 start/end duration differs")
    if sha256_text(canonical_json(w)) != c.get("window_sha256"):
        raise SystemExit("H3 window/resampling contract hash differs")
    # Same physical file is necessary; measured decoder PTS checks below are
    # also necessary. Names/rates/durations alone are not origin proof.
    model_source = Path(comfy.__file__).parent / "ldm" / "minimax" / "model.py"
    if sha256_file(model_source) != H3_EXPECTED_MODEL_SHA256:
        raise SystemExit("MiniMax source differs from the pinned native chart recipe")
    return {**c, "minimax_model_sha256": H3_EXPECTED_MODEL_SHA256}

def h3_joint_measured(c, vm, am, geo):
    if vm["source_frame_indices"] != c["window"]["source_frame_indices"]:
        raise ValueError("measured H3 source frame map differs")
    if am.get("audio") != "decoded" or am.get("padded_samples") != 0 or am.get("discontinuities") != 0:
        raise ValueError("H3 clock proof refuses silence, padding or audio PTS gaps")
    if abs(vm["first_pts_seconds"] - am["first_pts_seconds"]) > 1.0 / H3_AUDIO_SAMPLE_RATE:
        raise ValueError("H3 audio/video PTS origins differ")
    m = {"video_frame_indices": vm["source_frame_indices"], "video_source_times": vm["source_times"],
         "video_target_times": vm["target_times"], "video_first_pts_seconds": vm["first_pts_seconds"],
         "audio_first_pts_seconds": am["first_pts_seconds"], "audio_rate": am["rate"],
         "audio_source_rate": am["source_rate"], "audio_source_frame_pts": am["source_frame_pts"],
         "audio_discontinuities": am["discontinuities"], "audio_samples": am["cropped_samples"],
         "waveform_padded_samples": am["padded_samples"], "audio": am["audio"],
         "minimax_model_sha256": c["minimax_model_sha256"]}
    return {**c, "measured": m}

'''


def writer_candidate(source: Path) -> tuple[str, dict[str, Any]]:
    if sha(source) != STAGE_C_SHA256:
        raise Refusal("Stage-C baseline moved; inspect before regenerating candidate")
    before = source.read_bytes().decode("utf-8").replace("\r\n", "\n")
    after = before
    replacements = [
        ('def run_h3_encode(args: argparse.Namespace) -> int:\n', WRITER_HELPERS + 'def run_h3_encode(args: argparse.Namespace) -> int:\n'),
        ('    corpus = corpus_identity(args.items)\n    directory = cache_dir(args.cache_root, "h3-vae"',
         '    corpus = corpus_identity(args.items)\n    if args.teacher_digest != "full":\n        raise SystemExit("strict H3 capture requires full teacher hashes")\n    directory = cache_dir(args.cache_root, "h3-vae"'),
        ('    before = teacher_identity(teachers, args.teacher_digest)\n    vvae, vinfo = load_vae(args.video_vae)\n',
         '    source_contracts = {i: h3_joint_source(item, geo, args) for i, item in enumerate(items)}\n    before = teacher_identity(teachers, args.teacher_digest)\n    vvae, vinfo = load_vae(args.video_vae)\n'),
        ('            meta["video_sha256"] = sha256_file(path)  # outside the drop guard: an unreadable mount is fatal, not a drop\n',
         '            meta["video_sha256"] = sha256_file(path)  # outside the drop guard: an unreadable mount is fatal, not a drop\n            meta["audio_sha256"] = sha256_file(item["audio"])\n'),
        ('                return frames, vmeta, wave, ameta\n',
         '                meta["h3_source_contract"] = h3_joint_measured(source_contracts[index], vmeta, ameta, geo)\n                return frames, vmeta, wave, ameta\n'),
        ('                                                "picked_source_frames": [picks[0], picks[-1]], "start_seconds": start}',
         '                                                "picked_source_frames": [picks[0], picks[-1]], "start_seconds": start,\n                                                "source_frame_indices": picks, "source_times": rel, "first_pts_seconds": base,\n                                                "target_times": frame_times_for_target(frames, fps, start)}'),
        ('        chunks = []\n        for frame in container.decode(stream):\n',
         '        chunks = []\n        source_pts, source_rate, discontinuities, expected_pts = [], None, 0, None\n        for frame in container.decode(stream):\n            if frame.pts is None or frame.time_base is None or not frame.sample_rate:\n                raise ValueError("H3 audio lacks a measurable sample clock")\n            t = float(frame.pts * frame.time_base)\n            source_pts.append(t)\n            source_rate = int(frame.sample_rate)\n            if expected_pts is not None and abs(t - expected_pts) > 1.0 / source_rate:\n                discontinuities += 1\n            expected_pts = t + frame.samples / source_rate\n'),
        ('    return wave.unsqueeze(0), {"audio": "decoded", "padded_samples": max(0, padded), "rate": rate}',
         '    return wave.unsqueeze(0), {"audio": "decoded", "padded_samples": max(0, padded), "rate": rate,\n                               "cropped_samples": min(need, need - padded), "source_rate": source_rate,\n                               "source_frame_pts": source_pts, "first_pts_seconds": source_pts[0],\n                               "discontinuities": discontinuities}'),
        ('    present = sorted(enc.items)\n',
         '    present = sorted(enc.items)\n    if teacher == "h3":\n        if args.teacher_digest != "full" or args.parity_every < 0:\n            raise SystemExit("strict H3 velocity requires full teacher hashes and actual parity")\n        for entry in enc.items.values():\n            if not entry["meta"].get("h3_source_contract", {}).get("measured"):\n                raise SystemExit("H3 encode cache lacks strict measured joint origins")\n'),
        ('                        put("v_video", tgt["v_video"][0])\n',
         '                        meta["h3_source_contract"] = entry["meta"]["h3_source_contract"]\n                        put("x_t_video", xt[0][0])\n                        put("x_t_audio", xt[1][0])\n                        put("v_audio_native", tgt["v_audio_native"][0])\n                        put("v_video", tgt["v_video"][0])\n'),
        ('        "shards": [{k: r[k] for k in ("file", "sha256", "bytes", "finite") if k in r} for r in records],',
         '        "shards": [{**{k: r[k] for k in ("file", "sha256", "bytes", "finite") if k in r},\n                    "sidecar_sha256": sha256_file(directory / Path(r["file"]).with_suffix(".json"))} for r in records],'),
        ('        "kind": f"{teacher}-velocity", "plan": plan, "plan_sha256": claim["plan_sha256"], "corpus": corpus, "runtime": runtime,',
         '        "kind": f"{teacher}-velocity", "h3_joint_contract": H3_JOINT_CONTRACT if teacher == "h3" else None,\n        "plan": plan, "plan_sha256": claim["plan_sha256"], "corpus": corpus, "runtime": runtime,'),
    ]
    for old, new in replacements:
        if after.count(old) != 1:
            raise Refusal("pinned writer patch anchor missing or ambiguous: " + old[:80])
        after = after.replace(old, new, 1)
    # The original imports comfy lazily inside functions. The helper accesses the
    # loaded module explicitly, after common_runtime has established ComfyUI.
    after = after.replace('    model_source = Path(comfy.__file__).parent', '    import comfy\n    model_source = Path(comfy.__file__).parent', 1)
    ast.parse(after)
    return after, {"schema": "chimera_recovery.h3_writer_candidate.v1", "baseline_sha256": STAGE_C_SHA256,
                   "candidate_sha256": hashlib.sha256(after.encode()).hexdigest(), "patch_anchors": len(replacements),
                   "parent_model_py_sha256": H3_MODEL_SHA256, "teacher_job_launched": False,
                   "native_train_ready": False, "status": "private candidate for review and bounded capture"}


def emit_writer(source: Path, out: Path) -> dict[str, Any]:
    after, receipt = writer_candidate(source)
    if any(p.exists() for p in (out, out.with_suffix(".patch"), out.with_suffix(".receipt.json"))) or out.resolve() == source.resolve():
        raise Refusal("candidate must be a new file; refusing existing source mutation")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(after.encode())
    before = source.read_bytes().decode().replace("\r\n", "\n")
    patch = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                       fromfile="stage_c_media_a.py (pinned baseline)", tofile=out.name))
    out.with_suffix(".patch").write_bytes(patch.encode())
    out.with_suffix(".receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    a = subs.add_parser("patch-writer")
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a = subs.add_parser("prepare-items")
    a.add_argument("--planner-manifest", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a = subs.add_parser("preflight")
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--writer-sha256", required=True)
    a.add_argument("--out", type=Path)
    try:
        if argv is None:
            import sys
            argv = sys.argv[1:]
        args = p.parse_args(argv)
        if args.command == "patch-writer":
            result = emit_writer(args.source, args.out)
        elif args.command == "prepare-items":
            result = prepare_items(args.planner_manifest, args.out)
        else:
            result = preflight(args.cache, args.writer_sha256)
            if args.out:
                if args.out.exists():
                    raise Refusal("refusing to overwrite preflight evidence")
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_bytes(canonical(result) + b"\n")
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("capture_contract_ok", True) else 2
    except (Refusal, OSError, KeyError, json.JSONDecodeError) as e:
        print(json.dumps({"status": "refused", "reason": str(e), "native_train_ready": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
