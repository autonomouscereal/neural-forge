"""Authentic temporal C11 ingress and assistant-answer index conversion.

CPU inspection executes the pinned parent processor. It never fabricates ViT
features or answer logits. New video bridge time positions are an explicit
recovery observation chart, not original folded vision-RoPE parity.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

try:
    from . import audio_admission as A
except ImportError:
    import audio_admission as A

SCHEMA = "recovery_video_ingress.v1"
POLICY = "qvl_temporal_c11_observation_2026-10-13.v1"
PLACEHOLDER, TEXT_VOCAB = 248832, 248320
WRITER_SHA = "f2465a11b7a2242f167191fca8583a11ff74a50d33d7bba84b45c2d123ce6cb9"
VL_ADAPTER_SHA = "fd868f4494384f0dc9c9b964dc84d86752bdefe46833f667fd3bad129cf5d44d"
PYAV_VERSION = "19.0.1"
PYAV_WHEEL_SHA = "1bea5b6134209305199bce7627ac3d33964de2cf2b09c77d08e7f67cf8bd4170"
SOURCE_PINS = {
    "transformers.vision_utils": "71dc126123afc284914755e68f0fc59d0846f1f4f4fd4896ce451ee736ab31bf",
    "transformers.models.qwen3_vl.video_processing_qwen3_vl": "36f6aa5a541172e7e97c904c07bcf233b68eb7d4ff5a0ac71072c1f657a8c7cc",
    "transformers.models.qwen3_vl.processing_qwen3_vl": "add767ca5e9511e1db0a69091d51b1d0b2078c874d59ee3a58c4748b5d9014a4",
    "transformers.models.qwen3_vl.modeling_qwen3_vl": "f39b3621fea3f757a01d2e00ebd3a2be3516b7df3048d862a365988aba9fa994",
}


def module(path, expected, name):
    if A.file_sha(path) != expected:
        raise ValueError("reviewed executable source bytes differ: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def pin(path):
    return {"path": str(Path(path).resolve()), "sha256": A.file_sha(path)}


def full_frame_clock(pts, time_base, fps, count):
    """Require physical decoded PTS to equal the complete nominal frame clock."""
    rate, base = Fraction(fps), Fraction(time_base)
    if rate <= 0 or base <= 0 or len(pts) != count or not count or any(p is None for p in pts):
        raise ValueError("complete finite physical video timestamps required")
    actual = [Fraction(p) * base for p in pts]
    if actual != [Fraction(i, 1) / rate for i in range(count)]:
        raise ValueError("decoded physical PTS differ from the full zero-origin frame clock")
    return [float(t) for t in actual]


def decode_full_pyav(path, num_frames):
    """Explicit alternate decoder; parent processor replay must remain bitwise."""
    import av
    from PIL import Image
    if av.__version__ != PYAV_VERSION:
        raise ValueError("reviewed private decoder version differs")
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError("one physical video stream required")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        rate, base = stream.average_rate, stream.time_base
        decoded = list(container.decode(stream))
        if len(decoded) != num_frames:
            raise ValueError("alternate decoder cannot sample/drop/pad frames")
        clock = full_frame_clock([f.pts for f in decoded], base, rate, len(decoded))
        if (stream.start_time != 0 or stream.duration is None or
                Fraction(stream.duration)*Fraction(base) != Fraction(len(decoded),1)/Fraction(rate)):
            raise ValueError("full physical video duration/start differ from the captured route")
        pixels = [f.to_ndarray(format="rgb24") for f in decoded]
        if any(p.shape != (stream.height,stream.width,3) for p in pixels):
            raise ValueError("decoded frame dimensions differ")
        return [Image.fromarray(p,"RGB") for p in pixels], {
            "fps":float(rate),"total_num_frames":len(decoded),"frames_indices":list(range(len(decoded))),
            "width":stream.width,"height":stream.height,
            "physical_frame_timestamps_seconds":clock,
            "decoder":"explicit private PyAV full decode; original FFmpeg CLI not executed"}


@contextmanager
def decoder_adapter(writer, decoder, wheel):
    if decoder == "original":
        yield {"policy":"original source-pinned FFmpeg CLI", "alternate":False}
        return
    if decoder != "pyav_full" or wheel is None or A.file_sha(wheel) != PYAV_WHEEL_SHA:
        raise ValueError("explicit reviewed private decoder wheel required")
    import av
    if av.__version__ != PYAV_VERSION:
        raise ValueError("private decoder import differs from reviewed wheel")
    original = writer.decode_video
    writer.decode_video = decode_full_pyav
    try:
        yield {"policy":"pyav_full source/PTS preserving; mandatory prior parent tensor replay",
            "alternate":True,"wheel":pin(wheel),"package_version":av.__version__,
            "library_versions":av.library_versions,"original_ffmpeg_cli_executed":False}
    finally:
        writer.decode_video = original


def patchify(frames):
    """Captured parent temporal2/spatial16/merge2 packing, without frame pad."""
    frames = np.asarray(frames)
    if (frames.ndim != 4 or frames.dtype != np.float32 or frames.shape[1] != 3 or
            not frames.shape[0] or frames.shape[0] % 2 or frames.shape[2] % 32 or frames.shape[3] % 32 or
            not np.isfinite(frames).all()):
        raise ValueError("finite complete even-frame FP32 RGB factor32 video required; no silent frame pad")
    t, c, h, w = frames.shape
    gt, gh, gw = t // 2, h // 16, w // 16
    flat = frames.reshape(gt, 2, c, gh // 2, 2, 16, gw // 2, 2, 16).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gt * gh * gw, 1536)
    return np.ascontiguousarray(flat), np.array([[gt, gh, gw]], dtype=np.int64)


def unpack(flat, grid):
    grid = np.asarray(grid)
    if grid.shape != (1, 3) or grid.dtype != np.int64:
        raise ValueError("one complete int64 temporal grid required")
    gt, gh, gw = map(int, grid[0])
    if gt <= 0 or gh <= 0 or gw <= 0 or gh % 2 or gw % 2 or np.asarray(flat).shape != (gt * gh * gw, 1536):
        raise ValueError("temporal patch rows/grid differ")
    return flat.reshape(gt, gh // 2, gw // 2, 2, 2, 3, 2, 16, 16).transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(gt * 2, 3, gh * 16, gw * 16)


def layout(grid, indices, fps, total_frames):
    """Parent mean paired-frame timestamps plus merge-window spatial order."""
    gt, gh, gw = map(int, np.asarray(grid)[0])
    if (type(total_frames) is not int or total_frames <= 0 or len(indices) != gt * 2 or
            indices != list(range(total_frames)) or len(indices) != total_frames or
            not np.isfinite(fps) or fps <= 0 or gh % 2 or gw % 2):
        raise ValueError("this recovery route requires full ordered coverage without frame sampling/padding")
    times = np.array([(indices[i] + indices[i + 1]) / (2 * fps) for i in range(0, len(indices), 2)], dtype=np.float64)
    bh, bw, mh, mw = np.meshgrid(np.arange(gh // 2), np.arange(gw // 2), np.arange(2), np.arange(2), indexing="ij")
    yx = np.stack(((bh * 2 + mh).reshape(-1), (bw * 2 + mw).reshape(-1)), -1).astype(np.int64)
    tyx = np.concatenate((np.repeat(np.arange(gt), gh * gw)[:, None], np.tile(yx, (gt, 1))), axis=1)
    positions = np.zeros((gt * gh * gw, 4), dtype=np.float32)
    positions[:, 0] = np.repeat(times.astype(np.float32), gh * gw)
    positions[:, 1] = tyx[:, 1].astype(np.float32) / max(1, gh - 1)
    positions[:, 2] = tyx[:, 2].astype(np.float32) / max(1, gw - 1)
    cu = np.arange(gt + 1, dtype=np.int32) * (gh * gw)
    return {"tyx": tyx, "native_media_positions": positions, "paired_timestamps_seconds": times, "teacher_cu_seqlens": cu}


def expand_dialogue(ids, types, assistant_mask, *, video_token_id, media_rows):
    """Replace each parent merged video token with its four raw C11 rows.

    Every other genuine parent token (timestamps/headers/assistant/EOS) remains
    unchanged. Answer masks stay target-indexed; prediction masks shift once.
    """
    ids, types, mask = map(np.asarray, (ids, types, assistant_mask))
    if (ids.ndim != 1 or ids.dtype != np.int64 or types.shape != ids.shape or types.dtype != np.int64 or
            mask.shape != ids.shape or mask.dtype != np.bool_ or not mask.any() or mask[0] or
            (ids < 0).any() or (ids >= TEXT_VOCAB).any() or not np.isin(types, [0, 2]).all() or
            np.any(mask & (types != 0)) or media_rows <= 0 or media_rows % 4):
        raise ValueError("authentic parent video history and assistant-only terminal target mask required")
    visual = np.flatnonzero(types == 2)
    target = np.flatnonzero(mask)
    if (len(visual) * 4 != media_rows or not np.array_equal(ids[visual], np.full(len(visual), video_token_id)) or
            not np.array_equal(np.flatnonzero(ids == video_token_id), visual) or not len(visual) or
            visual[-1] >= target[0] or not np.array_equal(target, np.arange(target[0], target[-1] + 1))):
        raise ValueError("parent visual placeholder count/order or assistant contiguity differs")
    result, owned, media, mapping = [], [], [], []
    for i, value in enumerate(ids.tolist()):
        mapping.append(len(result))
        if types[i] == 2:
            media.extend(range(len(result), len(result) + 4))
            result.extend([PLACEHOLDER] * 4)
            owned.extend([False] * 4)
        else:
            result.append(value)
            owned.append(bool(mask[i]))
    new_ids = np.array(result, dtype=np.int64)
    new_mask = np.array(owned, dtype=np.bool_)
    shifted = np.zeros(len(result), dtype=np.bool_)
    shifted[:-1] = new_mask[1:]
    return {"input_ids": new_ids[None], "assistant_token_mask": new_mask,
            "assistant_prediction_mask": shifted, "native_media_indices": np.array(media, dtype=np.int64)[None],
            "parent_to_native_token_index": np.array(mapping, dtype=np.int64),
            "parent_input_ids": ids[None].copy(), "parent_assistant_token_mask": mask.copy()}


def rebase_answer(answer, ingress):
    """Convert already authenticated dense parent K64 targets, never infer them."""
    keys = {"input_ids", "assistant_token_mask", "next_ids", "ce_w", "eos_w", "q_w", "q_ids", "q_lp"}
    if set(answer) != keys:
        raise ValueError("complete authentic vision_answer_source.v1 tensors required")
    a = {k: np.asarray(v) for k, v in answer.items()}
    ids, mask = ingress["parent_input_ids"], ingress["parent_assistant_token_mask"]
    length = ids.shape[1]
    owned = np.zeros(length, dtype=np.bool_); owned[:-1] = mask[1:]
    if (not np.array_equal(a["input_ids"], ids) or not np.array_equal(a["assistant_token_mask"], mask) or
            any(a[k].shape != (length,) for k in ("next_ids", "ce_w", "eos_w", "q_w")) or
            a["q_ids"].shape != (length, 64) or a["q_lp"].shape != (length, 64) or
            any(not np.isfinite(v).all() for v in a.values()) or a["ce_w"].any() or a["eos_w"].any() or
            a["q_ids"].dtype != np.int64 or a["next_ids"].dtype != np.int64 or
            not np.array_equal(a["q_w"], owned.astype(a["q_w"].dtype)) or
            np.any(a["q_ids"] < 0) or np.any(a["q_ids"] >= TEXT_VOCAB) or np.any(a["q_lp"][owned] > 0) or
            any(len(set(row.tolist())) != 64 for row in a["q_ids"][owned]) or
            np.any(np.exp(a["q_lp"][owned].astype(np.float64)).sum(axis=1) > 1.001) or
            not np.array_equal(a["next_ids"][:-1][owned[:-1]], ids[0, 1:][owned[:-1]])):
        raise ValueError("actual parent K64 positions/native vocabulary/assistant ownership differ")
    new_length = ingress["input_ids"].shape[1]
    result = {k: np.zeros((new_length,) + v.shape[1:], dtype=v.dtype) for k, v in a.items()
              if k not in {"input_ids", "assistant_token_mask"}}
    mapping = ingress["parent_to_native_token_index"]
    for parent in np.flatnonzero(owned):
        native = int(mapping[parent + 1]) - 1
        for key in result:
            result[key][native] = a[key][parent]
    result.update(input_ids=ingress["input_ids"].copy(), assistant_token_mask=ingress["assistant_token_mask"].copy())
    if not np.array_equal(result["q_w"] > 0, ingress["assistant_prediction_mask"]):
        raise ValueError("native answer conversion shifted ownership incorrectly")
    return result


def require_training_targets(*, feature_manifest, answer_manifest, source_id, source_row_sha256, asset_sha256):
    """Fail before a model/optimizer load when genuine target seals are absent."""
    for manifest, schema in ((feature_manifest, "recovery_video_vit13_features.v1"),
                             (answer_manifest, "chimera_recovery.vision_answer_source.v1")):
        if (not isinstance(manifest, dict) or manifest.get("schema") != schema or manifest.get("sealed") is not True or
                manifest.get("all_finite") is not True or manifest.get("teacher_weights_unchanged") is not True or
                A.digest({k:v for k,v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256")):
            raise ValueError("genuine sealed temporal ViT13 and assistant K64 caches required before native training")
        integrity = manifest.get("teacher_integrity", {})
        if (integrity.get("mode") != "full" or integrity.get("unchanged") is not True or
                not integrity.get("before") or integrity.get("before") != integrity.get("after")):
            raise ValueError("actual full unchanged teacher file identities are required")
        for value in integrity["before"].values():
            if not A.SHA.fullmatch(str(value.get("sha256"))) or type(value.get("bytes")) is not int or value["bytes"] <= 0:
                raise ValueError("full teacher weight seals are missing")
        if schema == "recovery_video_vit13_features.v1" and (manifest.get("block") != 13 or manifest.get("feature_width") != 1152 or
                manifest.get("model_forward_executed") is not True or manifest.get("source_pins") != SOURCE_PINS):
            raise ValueError("actual source-pinned temporal ViT13 forward targets required")
        matches = [r for r in manifest.get("shards", []) if r.get("source_id") == source_id]
        if (len(matches) != 1 or matches[0].get("asset_sha256") != asset_sha256 or
                matches[0].get("source_row_sha256") != source_row_sha256 or matches[0].get("split") != "train" or
                matches[0].get("held") is not False):
            raise ValueError("temporal features/answer targets must identify the same actual source")
        A.pinned(matches[0], "authentic video target shard")
    return True


def inspect(*, plan, plan_sha256, inspection, inspection_sha256, vl_adapter, writer, model_dir, output,
            decoder="original", decoder_wheel=None):
    if output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("fresh output and hidden CUDA required for CPU-only video inspection")
    import torch
    from transformers.vision_utils import get_vision_cu_seqlens
    if torch.cuda.is_initialized():
        raise ValueError("CPU processor preparation cannot initialize CUDA")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    for name, expected in SOURCE_PINS.items():
        actual = importlib.import_module(name)
        if A.file_sha(actual.__file__) != expected:
            raise ValueError("captured original video processor/model implementation changed")
    V = module(vl_adapter, VL_ADAPTER_SHA, "recovery_video_original_vl_adapter")
    W = V.load_writer(writer, WRITER_SHA)
    p = V.load_plan(plan, plan_sha256)
    old = json.loads(A.pinned({"path": str(inspection), "sha256": inspection_sha256}, "actual original VL inspection").read_text())
    if old.get("passed") is not True or old.get("plan_sha256") != plan_sha256 or old.get("adapter_sha256") != VL_ADAPTER_SHA:
        raise ValueError("actual parent processor source inspection pin differs")
    processor = W.load_processor(model_dir, p["processor_options"]["max_pixels"])
    if W.processor_files(model_dir) != old["processor"]["files_sha256"] or V.R.sha(model_dir / "config.json") != old["model_config_sha256"]:
        raise ValueError("original tokenizer/processor/model configuration changed")
    video = processor.video_processor
    if (video.patch_size, video.temporal_patch_size, video.merge_size, tuple(video.image_mean), tuple(video.image_std)) != (16, 2, 2, (.5,)*3, (.5,)*3):
        raise ValueError("native temporal C11 raw pixel normalization/packing differs")
    previous = {r["row_id"]: r for r in old["rows"]}
    output.mkdir(parents=True); (output / "shards").mkdir()
    receipts = []
    with decoder_adapter(W,decoder,decoder_wheel) as decoder_proof, V.assistant_mask_adapter(W):
        for source in p["sources"]:
            corpus = W.load_vl_corpus(plan.parent / source["path"])
            for record in corpus["rows_data"]:
                if record["route"] != "video_understand":
                    continue
                ident = record["row_id"]
                encoded = W.encode_vl_row(processor, record, plan.parent, {"enable_thinking": False})
                before = previous[ident]
                if (V.R.tensor_sha256(encoded["input_ids"]) != before["input_ids_sha256"] or
                        any(V.R.tensor_sha256(encoded[k]) != sha for k, sha in before["media_tensors_sha256"].items())):
                    raise ValueError("actual repeated video encoding differs from prior parent proof")
                visible = record["messages"][0]["content"][0]
                path = plan.parent / visible["path"]
                if A.file_sha(path) != visible["sha256"]:
                    raise ValueError("source physical video bytes changed")
                frames, metadata = W.decode_video(path, visible["num_frames"])
                flat = encoded["pixel_values_videos"].cpu().numpy().copy()
                grid = encoded["video_grid_thw"].cpu().numpy().astype(np.int64)
                raw = unpack(flat, grid)
                exact_flat, exact_grid = patchify(raw)
                if not np.array_equal(flat, exact_flat) or not np.array_equal(grid, exact_grid):
                    raise ValueError("captured actual processor raw pixels fail exact temporal packing inverse")
                chart = layout(grid, metadata["frames_indices"], metadata["fps"], metadata["total_num_frames"])
                actual_cu = get_vision_cu_seqlens(encoded["video_grid_thw"])
                if not np.array_equal(actual_cu.cpu().numpy(), chart["teacher_cu_seqlens"]):
                    raise ValueError("temporal boundaries differ from actual native vision utility")
                parent_times = processor._calculate_timestamps(metadata["frames_indices"].copy(), metadata["fps"], 2)
                if not np.array_equal(chart["paired_timestamps_seconds"], parent_times):
                    raise ValueError("observation clock differs from original processor frame-pair timestamps")
                dialogue = expand_dialogue(encoded["input_ids"][0].cpu().numpy(), encoded["mm_token_type_ids"][0].cpu().numpy(),
                    encoded["assistant_target_mask"][0].cpu().numpy(), video_token_id=processor.video_token_id, media_rows=len(flat))
                arrays = {**dialogue, **chart, "patch_flat": flat, "grid_thw": grid,
                    "native_media_type_ids": np.full((1, len(flat)), 6, dtype=np.int64),
                    "native_media_valid_mask": np.ones((1, len(flat)), dtype=np.bool_),
                    "native_media_noise_levels": np.zeros((1, len(flat)), dtype=np.float32)}
                shard = output / "shards" / (ident + ".npz"); np.savez_compressed(shard, **arrays)
                receipts.append({"schema": SCHEMA, "id": ident, "source_row_sha256": record["source_row_sha256"],
                    "split": source["split"], "source_group_id": record["recovery_group_id"], "asset_sha256": visible["sha256"],
                    "asset": pin(path), "ingress": pin(shard), "metadata": metadata, "grid_thw": grid[0].tolist(),
                    "paired_timestamps_seconds": parent_times, "parent_prefix_full_ids_reproduced": True,
                    "all_eight_frames_retained": len(frames) == 8 and metadata["frames_indices"] == list(range(8)),
                    "raw_frame_sha256": [hashlib.sha256(np.asarray(f, dtype=np.uint8).tobytes()).hexdigest() for f in frames],
                    "native_input_length": int(dialogue["input_ids"].shape[1]), "media_rows": len(flat),
                    "assistant_target_tokens": int(dialogue["assistant_token_mask"].sum()),
                    "decoder_proof":decoder_proof,"teacher_targets_present": False, "native_train_ready": False})
    path = output / "receipts.jsonl"; path.write_bytes(b"".join(A.canonical(r)+b"\n" for r in receipts))
    result = {"schema": SCHEMA, "sealed": True, "source_year": 2026, "policy": POLICY, "source_pins": SOURCE_PINS,
        "code_sha256": A.file_sha(Path(__file__)), "shared_helpers": {"audio_admission.py":pin(A.__file__)},
        "plan": {"path": str(plan.resolve()), "sha256": plan_sha256}, "decoder_proof":decoder_proof,
        "original_processor_inspection": pin(inspection), "processor_files": old["processor"]["files_sha256"],
        "rows": len(receipts), "raw_pixel_feature_width": 1536, "native_type_id": 6, "native_feature_band": [640,2176],
        "outputs": {"receipts": {**pin(path), "rows": len(receipts)}}, "all_frames_retained": all(r["all_eight_frames_retained"] for r in receipts),
        "parent_vision_spatial_rope_parity": False, "parent_temporal_attention_segmentation_parity": False,
        "full_trunk_parent_parity": False, "model_weights_loaded": False, "teacher_targets_generated": False,
        "native_train_ready": False, "training_started": False,
        "blockers": ["Global frozen/family admission required before optimizer use.",
                     "Actual source-aligned ViT13 features and authentic assistant K64 caches are missing.",
                     "Selected temporal observation/insertion amendment and full native backward/answer evaluation remain required."]}
    result["manifest_sha256"] = A.digest(result)
    (output / "manifest.json").write_bytes(A.canonical(result)+b"\n")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "inspection", "vl-adapter", "writer", "model-dir", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("plan-sha256", "inspection-sha256"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--decoder",choices=("original","pyav_full"),default="original")
    p.add_argument("--decoder-wheel",type=Path)
    a = p.parse_args(argv)
    result = inspect(**vars(a))
    print(json.dumps({"rows":result["rows"], "all_frames_retained":result["all_frames_retained"], "native_train_ready":False}))


if __name__ == "__main__":
    main()
