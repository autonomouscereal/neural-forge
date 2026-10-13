"""Source-pinned standalone H3 clean-audio encode/decode proof, CPU first.

This diagnostic writes actual raw latents, never native XYZ or train admission.
It requires full source waveform coverage and observes the real encoder input.
No waveform/latent trim, fabricated video grid, foreign speech codec or stub is
used. A successful proof is one prerequisite for the distinct audio CE lane.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
import wave

STAGE_SHA = "fa936d72ebb0545acc3d2d2ac8e2bdc4477f370c73947352a0f20bcb23718fdf"
VAE_SHA = "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
AUDIO_SOURCE_SHA = "3e00d398b25d68dc39ca0eb589fc7549f3693b6bfaf08ef920e7ad2a7a803cba"
SOURCE_PINS = {
    "comfy/sd.py": "7efb10f22252446069a0b88f0a8bc2d94713696d9f946d7757a0981ae8101b58",
    "comfy/latent_formats.py": "f13259cbd815a339a474976522eb6aae10fb16c8397f13b41e50ddd107f8caf8",
    "comfy/model_base.py": "b7e425e0325c632249de429ebfc96d540aa0d87d313fedf89ebd00ed78238722",
    "comfy/supported_models.py": "e638a7bd0964683db583a290be2ca7a7f9e6ff0cd5d8e5103495786a246292ee",
    "comfy/ldm/minimax/audio_vae.py": AUDIO_SOURCE_SHA,
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
    if require_sha(audio_source_sha) != AUDIO_SOURCE_SHA:
        raise Error("audio source differs from actual audited H3 runtime")
    pins = SOURCE_PINS
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
                          value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


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
    import comfy.model_base
    import comfy.supported_models
    for relative, module in (("comfy/sd.py", comfy.sd), ("comfy/ldm/minimax/audio_vae.py", comfy.ldm.minimax.audio_vae),
                             ("comfy/latent_formats.py", comfy.latent_formats), ("comfy/model_base.py", comfy.model_base),
                             ("comfy/supported_models.py", comfy.supported_models)):
        if Path(module.__file__).resolve() != (comfy_root / relative).resolve():
            raise Error("Comfy imported another runtime checkout: " + relative)
    if torch.cuda.is_initialized():
        raise Error("actual CPU imports initialized CUDA")
    runtime.update({"python": platform.python_version(), "cpu_optional_triton_disabled": True,
                    "replacement_modules_used": False})
    return original, runtime


def model_space_transform(raw, *, model_base=None, supported_models=None, latent_formats=None):
    """Call actual inherited H3 transform without constructing the giant DiT.

    Explicit modules allow mathematical protocol fixtures; production imports
    are pinned and checked by verify_sources/load_runtime before this call.
    """
    import torch
    if model_base is None:
        import comfy.model_base as model_base
        import comfy.supported_models as supported_models
        import comfy.latent_formats as latent_formats
    if (supported_models.MiniMaxH3.latent_format is not latent_formats.MiniMaxH3AV or
            model_base.MiniMaxH3.process_latent_in is not model_base.BaseModel.process_latent_in or
            model_base.MiniMaxH3.process_latent_out is not model_base.BaseModel.process_latent_out):
        raise Error("actual H3 model-space method or latent-format ownership differs")
    context = SimpleNamespace(latent_format=supported_models.MiniMaxH3.latent_format())
    if (context.latent_format.scale_factor != 1.0 or
            latent_formats.MiniMaxH3AV.process_in is not latent_formats.LatentFormat.process_in or
            latent_formats.MiniMaxH3AV.process_out is not latent_formats.LatentFormat.process_out):
        raise Error("actual H3 raw/model latent transform is not the audited unit-scale inheritance")
    with torch.inference_mode():
        native = model_base.MiniMaxH3.process_latent_in(context, raw).float().cpu()
        inverse = model_base.MiniMaxH3.process_latent_out(context, native).float().cpu()
        packed = raw[0].permute(1, 2, 0).reshape(2 * raw.shape[-1], CHANNELS).contiguous()
        packed_model = model_base.MiniMaxH3.process_latent_in(context, packed).float().cpu()
    if not torch.equal(native, raw) or not torch.equal(inverse, raw) or not torch.equal(packed_model, packed):
        raise Error("actual parent raw/model-space or packed transform is not bitwise identity")
    return native, packed_model, {"parent_class": "comfy.model_base.MiniMaxH3", "latent_format": "MiniMaxH3AV",
        "method": "actual inherited BaseModel.process_latent_in/out", "scale_factor": 1.0,
        "raw_to_model_bitwise_equal": True, "inverse_bitwise_equal": True, "packed_transform_bitwise_equal": True,
        "full_dit_constructed": False, "scope": "parent's pure latent transform only; no diffusion forward or standalone XYZ parity claim"}


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


class EncoderSession:
    """One actual CPU VAE load; every waveform still receives its own proof.

    Per-wave parent-file hashes remain unchanged. A clean cache must additionally
    seal the session's actual in-memory parameter/buffer identity after traversal.
    No learned output or previous waveform encoding is reused.
    """
    def __init__(self, args):
        import torch
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
            raise Error('reusable encoder requires explicit CUDA-hidden CPU runtime')
        self.paths = {key: str(Path(getattr(args, key)).resolve()) for key in ('stage_c', 'comfy_root', 'audio_vae')}
        self.sources = verify_sources(args.stage_c, args.comfy_root, args.audio_source_sha256)
        if sha(args.audio_vae) != VAE_SHA:
            raise Error('session actual H3 audio VAE parent file differs')
        self.original, self.runtime = load_runtime(args.stage_c, args.comfy_root)
        self.vae, self.metadata = self.original.load_vae(args.audio_vae)
        self.imported_sources = imported_comfy_sources(args.comfy_root)
        self.initial_state = self.state_sha()
        self.rows, self.closed = 0, False

    def state_sha(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.vae.first_stage_model.state_dict().items()):
            digest.update(name.encode('utf-8'))
            digest.update(tensor_sha(value).encode('ascii'))
        return digest.hexdigest()

    def check(self, args):
        if self.closed or {key: str(Path(getattr(args, key)).resolve()) for key in self.paths} != self.paths:
            raise Error('encoder session identity changed or already sealed')
        if verify_sources(args.stage_c, args.comfy_root, args.audio_source_sha256) != self.sources:
            raise Error('encoder session source bytes changed')
        self.check_imports()

    def check_imports(self):
        current = imported_comfy_sources(Path(self.paths['comfy_root']))
        if any(current.get(path) != expected for path, expected in self.imported_sources.items()):
            raise Error('reusable encoder imported implementation changed during traversal')
        # A genuine first encode can import further implementation modules.
        # Retain their exact file digests and reject any later removal/change.
        self.imported_sources.update(current)

    def seal(self):
        if self.closed or self.rows < 1:
            raise Error('encoder session requires a nonempty completed traversal before seal')
        self.check_imports()
        after = self.state_sha()
        if (after != self.initial_state or sha(Path(self.paths['audio_vae'])) != VAE_SHA or
                verify_sources(Path(self.paths['stage_c']), Path(self.paths['comfy_root']), AUDIO_SOURCE_SHA) != self.sources):
            raise Error('actual reusable VAE weights/buffers/imported source changed during traversal')
        self.closed = True
        return {'schema': 'chimera_recovery.h3_cpu_encoder_session.v1', 'sealed': True, 'actual_waveforms': self.rows,
            'one_actual_vae_load': True, 'each_wave_encoded_twice': True, 'each_wave_decoded': True,
            'per_wave_parent_file_sha_verified_before_and_after': True, 'teacher_unchanged': True,
            'initial_state_sha256': self.initial_state, 'final_state_sha256': after, 'teacher_sha256': VAE_SHA,
            'sources': self.sources, 'imported_python_sources': self.imported_sources,
            'cpu_only': True, 'training_started': False}


def run(args, *, session=None):
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
        if session is None:
            original, result['runtime'] = load_runtime(args.stage_c, args.comfy_root)
        else:
            if not isinstance(session, EncoderSession):
                raise Error('reusable runtime must be the actual verified encoder session')
            session.check(args)
            original, result['runtime'] = session.original, copy.deepcopy(session.runtime)
            result['encoder_session_initial_state_sha256'] = session.initial_state
            result['reused_verified_cpu_runtime'] = True
        source_before = imported_comfy_sources(args.comfy_root)
        waveform, result["decode"] = decode_full(args.audio, contract)
        reference, metadata = original.read_audio(str(args.audio), seconds=contract["duration_seconds"], start=0., rate=RATE)
        if metadata.get("audio") != "decoded" or metadata.get("padded_samples") != 0 or not torch.equal(reference, waveform):
            raise Error("full no-trim decode differs from actual pinned StageC read_audio")
        result["stage_c_decode_bitwise_equal"] = True
        if session is None:
            vae, result['vae'] = original.load_vae(args.audio_vae)
        else:
            vae, result['vae'] = session.vae, copy.deepcopy(session.metadata)
        tensors, result["measurement"] = prove_vae(vae, waveform, contract)
        tensors["input_waveform_stereo32k"] = waveform.contiguous()
        native, packed_model, result["model_space_transform"] = model_space_transform(tensors["x0_raw_audio"])
        tensors.update({"x0_model_audio": native.contiguous(), "packed_model_audio": packed_model.contiguous()})
        result["model_space_verified"] = True
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
        if session is not None:
            session.check_imports()
            session.rows += 1
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
        "sealed globally held-excluded source/batch admission required",
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
