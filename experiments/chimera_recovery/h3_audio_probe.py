"""Source-pinned standalone H3 clean-audio encode/decode proof, CPU first.

This diagnostic writes actual raw latents, never native XYZ or train admission.
It requires full source waveform coverage and observes the real encoder input.
No waveform/latent trim, fabricated video grid, foreign speech codec or stub is
used. A successful proof is one prerequisite for the distinct audio CE lane.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import wave

STAGE_SHA = "fa936d72ebb0545acc3d2d2ac8e2bdc4477f370c73947352a0f20bcb23718fdf"
VAE_SHA = "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
SOURCE_PINS = {
    "comfy/sd.py": "7efb10f22252446069a0b88f0a8bc2d94713696d9f946d7757a0981ae8101b58",
    "comfy/latent_formats.py": "f13259cbd815a339a474976522eb6aae10fb16c8397f13b41e50ddd107f8caf8",
}
RATE, HOP, CHANNELS = 32000, 800, 32


class Error(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1 << 24), b""):
            result.update(data)
    return result.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Error("a complete lower-case SHA256 is required")
    return value


def wav_contract(path, expected_sha):
    if sha(path) != require_sha(expected_sha):
        raise Error("audio asset SHA differs")
    with wave.open(str(path), "rb") as handle:
        channels, rate, count, width, compression = (handle.getnchannels(), handle.getframerate(),
                                                    handle.getnframes(), handle.getsampwidth(), handle.getcomptype())
        data = handle.readframes(count)
    if channels not in (1, 2) or width != 2 or compression != "NONE" or rate <= 0 or count <= 0 or len(data) != count * width * channels:
        raise Error("probe requires complete real mono/stereo PCM16 WAV samples")
    # Integral target samples and complete native hops are mandatory. Arbitrary
    # durations need their own explicit padding/full-coverage policy.
    samples = count * RATE
    if samples % rate or (samples // rate) % HOP:
        raise Error("full source duration does not end on an exact native800-sample hop")
    return {"asset_sha256": expected_sha, "source_rate": rate, "source_channels": channels,
            "source_samples": count, "pcm_sha256": hashlib.sha256(data).hexdigest(),
            "duration_seconds": count / rate, "target_rate": RATE, "target_samples": samples // rate,
            "expected_latent_frames": samples // rate // HOP, "no_input_padding_or_trim": True}


def verify_sources(stage, comfy_root, audio_source_sha):
    pins = {**SOURCE_PINS, "comfy/ldm/minimax/audio_vae.py": require_sha(audio_source_sha)}
    if sha(stage) != STAGE_SHA:
        raise Error("audited StageC source differs")
    for relative, expected in pins.items():
        if sha(comfy_root / relative) != expected:
            raise Error("audited Comfy source differs: " + relative)
    return {"stage_c_sha256": STAGE_SHA, "comfy_sources": pins}


def tensor_sha(value):
    import torch
    value = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(canonical({"shape": list(value.shape), "dtype": str(value.dtype)}) +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


def decode_full(path, contract):
    """Decode/resample all source frames and flush; reject gaps/padding/trimming."""
    import av
    import numpy as np
    import torch
    chunks, source_samples, starts, ends = [], 0, [], []
    with av.open(str(path)) as container:
        if len(container.streams.audio) != 1:
            raise Error("one actual standalone audio stream required")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=RATE)
        previous_end = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None or frame.sample_rate != contract["source_rate"]:
                raise Error("actual source PCM timing/rate missing or changed")
            start = frame.pts * frame.time_base
            end = start + frame.samples / frame.sample_rate
            if previous_end is not None and not math.isclose(float(start), float(previous_end), rel_tol=0., abs_tol=1e-9):
                raise Error("source audio PTS gap or overlap")
            starts.append(float(start))
            ends.append(float(end))
            previous_end = end
            source_samples += frame.samples
            chunks.extend(out.to_ndarray() for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray() for out in resampler.resample(None))
    if source_samples != contract["source_samples"] or not chunks or not math.isclose(starts[0], 0., abs_tol=1e-9):
        raise Error("decoded source sample coverage/start differs from full WAV")
    audio = torch.from_numpy(np.concatenate(chunks, axis=1)).float().unsqueeze(0)
    if tuple(audio.shape) != (1, 2, contract["target_samples"]) or not torch.isfinite(audio).all():
        raise Error("full resampler output shape/finite contract differs; no corrective pad/trim")
    return audio, {"source_decoded_samples": source_samples, "first_pts_seconds": starts[0],
                   "last_end_seconds": ends[-1], "target_samples": int(audio.shape[-1]),
                   "pyav_version": av.__version__, "libav_versions": av.library_versions,
                   "conversion": "actual libav fltp stereo32k; source mono uses libav stereo conversion",
                   "resampler_flushed": True, "waveform_padded_samples": 0, "waveform_trimmed_samples": 0}


def prove_vae(vae, waveform, contract):
    """Observe exact first-stage input; require full native encode/decode outputs."""
    import torch
    expected_input = waveform.to(device="cpu", dtype=torch.float32)
    if (tuple(expected_input.shape) != (1, 2, contract["target_samples"]) or not torch.isfinite(expected_input).all() or
            str(vae.vae_dtype) != "torch.float32" or str(vae.device) != "cpu" or
            type(vae.first_stage_model).__name__ != "MiniMaxH3AudioVAE" or
            vae.downscale_ratio != HOP or vae.upscale_ratio != HOP or vae.latent_channels != CHANNELS or
            vae.audio_sample_rate != RATE or vae.latent_dim != 2):
        raise Error("actual H3 VAE/input geometry/dtype/CPU contract differs")
    calls = []
    original = vae.first_stage_model.encode
    def observed(value, *args, **kwargs):
        actual = value.detach().float().cpu()
        if not torch.equal(actual, expected_input):
            raise Error("actual first-stage encoder did not receive the full exact waveform")
        calls.append({"shape": list(actual.shape), "sha256": tensor_sha(actual)})
        return original(value, *args, **kwargs)
    vae.first_stage_model.encode = observed
    started = time.monotonic()
    try:
        with torch.inference_mode():
            raw = vae.encode(waveform.movedim(1, -1)).float().cpu()
            repeat = vae.encode(waveform.movedim(1, -1)).float().cpu()
    finally:
        vae.first_stage_model.encode = original
    encode_seconds = time.monotonic() - started
    shape = (1, CHANNELS, 2, contract["expected_latent_frames"])
    if len(calls) != 2 or tuple(raw.shape) != shape or not torch.isfinite(raw).all() or not torch.equal(raw, repeat):
        raise Error("full native latent shape/finite/deterministic encode proof failed; no latent corrective pad/trim")
    started = time.monotonic()
    with torch.inference_mode():
        decoded = vae.decode(raw).float().cpu()
    decode_seconds = time.monotonic() - started
    # Generic Comfy VAE.decode returns [B,samples,2] for audio.
    if tuple(decoded.shape) != (1, contract["target_samples"], 2) or not torch.isfinite(decoded).all():
        raise Error("full native decoded waveform shape/finite proof failed; no corrective pad/trim")
    reconstruction = decoded.movedim(-1, 1)
    packed = raw[0].permute(1, 2, 0).reshape(2 * shape[-1], CHANNELS).contiguous()
    inverse = packed.reshape(2, shape[-1], CHANNELS).permute(2, 0, 1).unsqueeze(0).contiguous()
    if not torch.equal(inverse, raw):
        raise Error("native channel-major stereo latent pack is not lossless")
    error = (reconstruction - expected_input).double().square().mean().item()
    signal = expected_input.double().square().mean().item()
    snr = 10 * math.log10(signal / error) if signal > 0 and error > 0 else None
    return {"x0_raw_audio": raw, "packed_raw_audio": packed, "decoded_waveform": reconstruction}, {
        "encoder_input_calls": calls, "raw_latent_shape": list(raw.shape), "packed_shape": list(packed.shape),
        "raw_latent_sha256": tensor_sha(raw), "decoded_shape": list(reconstruction.shape),
        "decoded_sha256": tensor_sha(reconstruction), "encoder_repeat_bitwise_equal": True,
        "channel_major_pack_bitwise_invertible": True, "latent_padded_frames": 0, "latent_trimmed_frames": 0,
        "measured_samples_per_latent_frame": contract["target_samples"] / raw.shape[-1],
        "nominal_hop_seconds": HOP / RATE, "nominal_frame_interval_seconds": [[i * HOP / RATE, (i + 1) * HOP / RATE] for i in range(shape[-1])],
        "interval_scope": "nominal encoder lattice only; convolution/attention receptive field and native XYZ are not established",
        "waveform_mse": error, "waveform_snr_db": snr, "encode_repeat_seconds": encode_seconds,
        "decode_seconds": decode_seconds}


def load_runtime(stage, comfy_root):
    """Use genuine CPU Comfy imports. Disable optional GPU Triton, no stubs."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise Error("CPU proof requires explicit CUDA_VISIBLE_DEVICES='' before startup")
    import torch
    if torch.cuda.is_initialized():
        raise Error("CPU proof must start before any CUDA initialization")
    os.environ["COMFYUI_ROOT"] = str(comfy_root)
    sys.path.insert(0, str(comfy_root))
    sys.modules["triton"] = None  # genuine optional-import CPU path, not a replacement implementation
    spec = importlib.util.spec_from_file_location("chimera_recovery_h3_probe_original_stage", stage)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    runtime = original.comfy_boot("cpu", vram_guard_mib=0, expect_device_name=None)
    import comfy.sd
    import comfy.ldm.minimax.audio_vae
    import comfy.latent_formats
    for relative, module in (("comfy/sd.py", comfy.sd), ("comfy/ldm/minimax/audio_vae.py", comfy.ldm.minimax.audio_vae),
                             ("comfy/latent_formats.py", comfy.latent_formats)):
        if Path(module.__file__).resolve() != (comfy_root / relative).resolve():
            raise Error("Comfy imported another runtime checkout: " + relative)
    if torch.cuda.is_initialized():
        raise Error("actual CPU imports initialized CUDA")
    runtime.update({"python": platform.python_version(), "cpu_optional_triton_disabled": True,
                    "replacement_modules_used": False})
    return original, runtime


def imported_comfy_sources(comfy_root):
    """Bind actual imported Python implementation bytes, including VAE deps."""
    root, result = comfy_root.resolve(), {}
    for name, module in tuple(sys.modules.items()):
        if name != "comfy" and not name.startswith("comfy."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue  # actual namespace packages have no executable file
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise Error("Comfy imported implementation outside pinned checkout: " + name)
        if path.suffix == ".py":
            result[str(path.relative_to(root)).replace("\\", "/")] = sha(path)
    return result


def run(args):
    result = {"schema": "chimera_recovery.h3_standalone_audio_probe.v1", "production_training_ready": False,
              "training_started": False, "native_xyz_verified": False, "model_space_verified": False,
              "native_admission_granted": False, "adapter_sha256": sha(Path(__file__)), "cpu_only": True}
    weights_before = None
    try:
        from safetensors.torch import save_file
        import torch
        result["sources"] = verify_sources(args.stage_c, args.comfy_root, args.audio_source_sha256)
        contract = wav_contract(args.audio, args.audio_sha256)
        result["input_contract"] = contract
        weights_before = sha(args.audio_vae)
        result["teacher_sha256_before"] = weights_before
        if weights_before != VAE_SHA:
            raise Error("actual H3 audio VAE parent file differs")
        original, result["runtime"] = load_runtime(args.stage_c, args.comfy_root)
        source_before = imported_comfy_sources(args.comfy_root)
        waveform, result["decode"] = decode_full(args.audio, contract)
        reference, metadata = original.read_audio(str(args.audio), seconds=contract["duration_seconds"], start=0., rate=RATE)
        if metadata.get("audio") != "decoded" or metadata.get("padded_samples") != 0 or not torch.equal(reference, waveform):
            raise Error("full no-trim decode differs from actual pinned StageC read_audio")
        result["stage_c_decode_bitwise_equal"] = True
        vae, result["vae"] = original.load_vae(args.audio_vae)
        tensors, result["measurement"] = prove_vae(vae, waveform, contract)
        tensors["input_waveform_stereo32k"] = waveform.contiguous()
        source_after = imported_comfy_sources(args.comfy_root)
        if any(source_after.get(path) != expected for path, expected in source_before.items()):
            raise Error("imported Comfy implementation changed during native proof")
        result["runtime"]["imported_python_sources"] = source_after
        if torch.cuda.is_initialized():
            raise Error("actual encode/decode unexpectedly initialized CUDA")
        result["probe_passed"] = True
        target = args.out.with_suffix(".safetensors")
        if target.exists():
            raise Error("preserve existing native probe tensors; choose a new output")
        save_file(tensors, str(target))
        result["tensors"] = {"path": str(target), "sha256": sha(target), "keys": sorted(tensors)}
    except (Exception, SystemExit) as exc:
        result["probe_passed"] = False
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if weights_before is not None:
            result["teacher_sha256_after"] = sha(args.audio_vae)
            result["teacher_unchanged"] = result["teacher_sha256_after"] == weights_before
            if not result["teacher_unchanged"]:
                result["probe_passed"] = False
                result["failure"] = {"type": "Error", "message": "parent VAE bytes changed during proof"}
    result["remaining_blockers"] = ["audited standalone native/source spatial XYZ policy required; existing H3 helper uses video grid",
        "raw VAE to full H3 parent model-space route parity required", "sealed globally held-excluded source/batch admission required",
        "distinct audio CE/read-policy amendment, production dispatch and sequential shared-adapter composition required"]
    args.out.write_bytes(canonical(result) + b"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio", "audio-vae", "stage-c", "comfy-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--audio-source-sha256", required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.with_suffix(".safetensors").exists():
        raise Error("preserve existing standalone probe evidence; select a new output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    print(json.dumps({"probe_passed": result["probe_passed"], "production_training_ready": False,
                      "failure": result.get("failure"), "receipt": str(args.out)}))
    return 0 if result["probe_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
