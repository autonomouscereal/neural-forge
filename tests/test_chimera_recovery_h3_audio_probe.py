"""Fail-closed full waveform/latents; tiny protocol fixture is not native proof."""
import importlib.util
from pathlib import Path
import struct
import sys
import wave

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/h3_audio_probe.py"
spec = importlib.util.spec_from_file_location("h3_audio_probe_test", PATH)
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)


def wav(tmp_path, samples=16000):
    path = tmp_path / "sound.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack("<" + str(samples) + "h", *([20] * samples)))
    return path


def test_complete_source_and_exact_native_hops(tmp_path):
    path = wav(tmp_path)
    contract = P.wav_contract(path, P.sha(path))
    assert contract["source_samples"] == 16000 and contract["target_samples"] == 32000
    assert contract["expected_latent_frames"] == 40 and contract["no_input_padding_or_trim"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(P.Error, match="asset SHA"):
        P.wav_contract(path, contract["asset_sha256"])


def test_partial_hop_refuses_instead_of_trimming(tmp_path):
    path = wav(tmp_path, samples=15999)
    with pytest.raises(P.Error, match="exact native800"):
        P.wav_contract(path, P.sha(path))


def fixture():
    torch = pytest.importorskip("torch")
    class MiniMaxH3AudioVAE:
        def __init__(self):
            self.crop = False
        def encode(self, value):
            # Deterministic tiny protocol fixture retains all actual input
            # samples by summing each 800-sample native hop.
            return value.reshape(1, 2, 40, 800).mean(-1).unsqueeze(1).repeat(1, 32, 1, 1)
    class VAE:
        vae_dtype, device = torch.float32, torch.device("cpu")
        downscale_ratio, upscale_ratio, latent_channels, audio_sample_rate, latent_dim = 800, 800, 32, 32000, 2
        def __init__(self):
            self.first_stage_model = MiniMaxH3AudioVAE()
            self.trim_latents = False
        def encode(self, value):
            value = value.movedim(-1, 1)
            if self.first_stage_model.crop:
                value = value[..., :-800]
            output = self.first_stage_model.encode(value)
            return output[..., :-1] if self.trim_latents else output
        def decode(self, value):
            return value[:, 0].repeat_interleave(800, dim=-1).movedim(1, -1)
    waveform = torch.arange(32000, dtype=torch.float32).reshape(1, 1, -1).repeat(1, 2, 1) / 32000
    return torch, VAE(), waveform, {"target_samples": 32000, "expected_latent_frames": 40}


def test_exact_observed_encoder_inputs_and_lossless_channel_major_pack():
    torch, vae, waveform, contract = fixture()
    original = vae.first_stage_model.encode
    tensors, result = P.prove_vae(vae, waveform, contract)
    assert vae.first_stage_model.encode == original
    assert result["raw_latent_shape"] == [1, 32, 2, 40] and result["packed_shape"] == [80, 32]
    assert result["measured_samples_per_latent_frame"] == 800 and result["nominal_hop_seconds"] == .025
    assert result["nominal_frame_interval_seconds"][-1] == [.975, 1.]
    assert result["encoder_repeat_bitwise_equal"] and result["channel_major_pack_bitwise_invertible"]
    assert torch.equal(tensors["packed_raw_audio"][:40], tensors["x0_raw_audio"][0, :, 0].T)
    assert torch.equal(tensors["packed_raw_audio"][40:], tensors["x0_raw_audio"][0, :, 1].T)


@pytest.mark.parametrize("problem,reason", [("waveform_trim", "full exact waveform"), ("latent_trim", "latent shape"), ("nan", "geometry")])
def test_real_encoder_input_or_latent_truncation_and_nonfinite_refuse(problem, reason):
    torch, vae, waveform, contract = fixture()
    original = vae.first_stage_model.encode
    if problem == "waveform_trim":
        vae.first_stage_model.crop = True
    elif problem == "latent_trim":
        vae.trim_latents = True
    else:
        waveform[0, 0, -1] = torch.nan
    with pytest.raises(P.Error, match=reason):
        P.prove_vae(vae, waveform, contract)
    assert vae.first_stage_model.encode == original


def test_cpu_startup_requires_explicit_hidden_cuda(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(P.Error, match="explicit CUDA_VISIBLE_DEVICES"):
        P.load_runtime(tmp_path / "stage.py", tmp_path)
