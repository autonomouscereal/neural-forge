"""Pluggable VAE adapters for C15 absorption.

The C15 *science* (pair sampling + closed-form map fitting + composition into Chimera's
bridge / emission head) is VAE-agnostic. Everything a parent VAE has to provide is
behind :class:`VaeAdapter`: a geometry description plus ``encode`` / ``decode``.

Three families ship here:

``SyntheticAudioVae`` / ``SyntheticVideoVae``
    Deterministic, weight-free, nonlinear stand-ins. They exist so the whole pipeline
    (sampler -> fitter -> composition -> proof rung) is runnable and unit-testable
    anywhere, including on a workstation with no model weights, and so the harness has
    a case with a KNOWN round-trip ceiling to validate the metric code against.

``ComfyVaeAdapter``
    The minimum real loader. ``comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))``
    auto-detects each checkpoint and builds the right first-stage model; verified
    branches in ``/opt/ComfyUI/comfy/sd.py`` inside image
    ``comfyui:codex-minimax-h3-runtime-env-r5-20260808``:

    * ``ace_1.5_vae.safetensors`` -> ``AudioOobleckVAE(strides=[2,4,4,6,10])``,
      48 kHz, 64 latent channels, stereo, hop 1920 samples/frame (= 25 Hz),
      latent ``[B, 64, T]``  (sd.py:651-674)
    * ``minimax_h3_audio_vae_fp32.safetensors`` -> ``MiniMaxH3AudioVAE()``, 32 kHz,
      32 latent channels, stereo, hop 800 samples/frame (= 40 Hz), latent
      ``[B, 32, 2, T]``  (sd.py:965-985)
    * ``minimax_h3_video_vae_fp16.safetensors`` -> ``MiniMaxH3VideoVAE``, 24 latent
      channels, 16x spatial / 4x temporal

    This adapter only ever imports ComfyUI, so it must run INSIDE that image. It runs
    the VAE only -- never the H3 DiT. The H3 licence gate (relay ruling 8324ea96) is
    still pending, so no H3 generation happens anywhere in this package.

``PairFileAdapter``
    Replays pairs sampled earlier by someone else, so the fitter never needs weights.

Audio layout convention used throughout: raw ``[B, C, S]`` float32 in [-1, 1];
latent ``[B, Cz, T]`` or ``[B, Cz, C, T]``. Video: raw ``[B, T, C, H, W]``;
latent ``[B, Cz, Tz, Hz, Wz]``.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class VaeGeometry:
    """Everything the pair sampler needs in order to line raw media up with latents."""

    name: str
    modality: str  # "audio" | "video" | "image"
    latent_channels: int
    sample_rate_hz: int = 0
    hop_samples: int = 0  # audio: raw samples per latent frame
    channels: int = 0  # audio: waveform channels
    channel_in_latent: bool = False  # audio: latent carries a channel axis
    spatial_stride: int = 0  # video/image
    temporal_stride: int = 0  # video
    latents_mean: torch.Tensor | None = None
    latents_std: torch.Tensor | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def frame_rate_hz(self) -> float:
        if self.modality == "audio" and self.hop_samples:
            return float(self.sample_rate_hz) / float(self.hop_samples)
        return 0.0

    def to_json(self) -> dict[str, Any]:
        body = {
            "name": self.name,
            "modality": self.modality,
            "latent_channels": int(self.latent_channels),
            "sample_rate_hz": int(self.sample_rate_hz),
            "hop_samples": int(self.hop_samples),
            "channels": int(self.channels),
            "channel_in_latent": bool(self.channel_in_latent),
            "spatial_stride": int(self.spatial_stride),
            "temporal_stride": int(self.temporal_stride),
            "frame_rate_hz": self.frame_rate_hz(),
            "notes": dict(self.notes),
        }
        return body


class VaeAdapter:
    """Minimal contract every C15 parent VAE must satisfy."""

    geometry: VaeGeometry

    def encode(self, raw: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def decode(self, latent: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def roundtrip(self, raw: torch.Tensor) -> torch.Tensor:
        """The parent's own reconstruction. This is the CEILING every fitted map is
        scored against -- absorption is judged relative to it, never in absolute dB."""

        return self.decode(self.encode(raw))


# --------------------------------------------------------------------------------------
# synthetic adapters
# --------------------------------------------------------------------------------------


def _semi_orthonormal(rows: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    matrix = torch.randn(max(rows, columns), min(rows, columns), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(matrix)
    q = q[:, : min(rows, columns)]
    basis = q.transpose(0, 1) if rows <= columns else q
    return basis[:rows, :columns].contiguous()


class SyntheticAudioVae(VaeAdapter):
    """``z = tanh(P x)`` per frame, ``x_hat = P^T atanh(z)``.

    ``P`` is a fixed semi-orthonormal [latent_channels, channels*hop]. The round-trip is
    exactly the orthogonal projection of the frame onto ``P``'s row space, so the
    ceiling PSNR is analytic and the nonlinearity is real -- a linear ``E`` cannot fit it
    perfectly, which is precisely the C15 question in miniature.
    """

    def __init__(
        self,
        *,
        hop_samples: int = 800,
        channels: int = 2,
        latent_channels: int = 32,
        sample_rate_hz: int = 32000,
        seed: int = 20260901,
        name: str = "synthetic_audio",
    ) -> None:
        self.hop = int(hop_samples)
        self.projection = _semi_orthonormal(int(latent_channels), int(channels) * self.hop, seed)
        self.geometry = VaeGeometry(
            name=name,
            modality="audio",
            latent_channels=int(latent_channels),
            sample_rate_hz=int(sample_rate_hz),
            hop_samples=self.hop,
            channels=int(channels),
            channel_in_latent=False,
            notes={"kind": "synthetic", "nonlinearity": "tanh", "seed": int(seed)},
        )

    def _frames(self, raw: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = raw.shape
        count = samples // self.hop
        if count <= 0:
            raise ValueError("waveform is shorter than one latent frame")
        trimmed = raw[..., : count * self.hop]
        return trimmed.reshape(batch, channels, count, self.hop).permute(0, 2, 1, 3).reshape(batch, count, -1)

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        frames = self._frames(raw.to(torch.float64))
        latent = torch.tanh(frames @ self.projection.transpose(0, 1))
        return latent.permute(0, 2, 1).to(raw.dtype)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        rows = latent.to(torch.float64).permute(0, 2, 1)
        frames = torch.atanh(rows.clamp(-0.999999, 0.999999)) @ self.projection
        batch, count, _ = frames.shape
        channels = self.geometry.channels
        waveform = frames.reshape(batch, count, channels, self.hop).permute(0, 2, 1, 3)
        return waveform.reshape(batch, channels, count * self.hop).to(latent.dtype)


class SyntheticVideoVae(VaeAdapter):
    """Same construction on video tubelet patches, so the video geometry is exercised."""

    def __init__(
        self,
        *,
        patch: int = 16,
        tubelet: int = 2,
        channels: int = 3,
        latent_channels: int = 24,
        seed: int = 20260902,
        name: str = "synthetic_video",
    ) -> None:
        self.patch = int(patch)
        self.tubelet = int(tubelet)
        self.channels = int(channels)
        width = self.tubelet * self.channels * self.patch * self.patch
        self.projection = _semi_orthonormal(int(latent_channels), width, seed)
        self.geometry = VaeGeometry(
            name=name,
            modality="video",
            latent_channels=int(latent_channels),
            spatial_stride=self.patch,
            temporal_stride=self.tubelet,
            notes={"kind": "synthetic", "nonlinearity": "tanh", "seed": int(seed), "raw_width": width},
        )

    def _patches(self, raw: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        batch, frames, channels, height, width = raw.shape
        tubelets = frames // self.tubelet
        grid_h = height // self.patch
        grid_w = width // self.patch
        if tubelets <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError("clip is smaller than one tubelet patch")
        block = raw[:, : tubelets * self.tubelet, :, : grid_h * self.patch, : grid_w * self.patch]
        block = block.reshape(batch, tubelets, self.tubelet, channels, grid_h, self.patch, grid_w, self.patch)
        block = block.permute(0, 1, 4, 6, 2, 3, 5, 7).contiguous()
        rows = block.reshape(batch, tubelets * grid_h * grid_w, -1)
        return rows, (batch, tubelets, grid_h, grid_w, channels)

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        rows, shape = self._patches(raw.to(torch.float64))
        latent = torch.tanh(rows @ self.projection.transpose(0, 1))
        batch, tubelets, grid_h, grid_w, _ = shape
        return latent.reshape(batch, tubelets, grid_h, grid_w, -1).permute(0, 4, 1, 2, 3).to(raw.dtype)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch, latent_channels, tubelets, grid_h, grid_w = latent.shape
        rows = latent.to(torch.float64).permute(0, 2, 3, 4, 1).reshape(batch, -1, latent_channels)
        flat = torch.atanh(rows.clamp(-0.999999, 0.999999)) @ self.projection
        block = flat.reshape(batch, tubelets, grid_h, grid_w, self.tubelet, self.channels, self.patch, self.patch)
        block = block.permute(0, 1, 4, 5, 2, 6, 3, 7).contiguous()
        video = block.reshape(
            batch,
            tubelets * self.tubelet,
            self.channels,
            grid_h * self.patch,
            grid_w * self.patch,
        )
        return video.to(latent.dtype)


# --------------------------------------------------------------------------------------
# the real loader (ComfyUI image only)
# --------------------------------------------------------------------------------------

#: Verified branch table read out of /opt/ComfyUI/comfy/sd.py in the runtime image.
COMFY_VAE_PROFILES: dict[str, dict[str, Any]] = {
    "ace_1.5_vae.safetensors": {
        "modality": "audio",
        "class": "AudioOobleckVAE(strides=[2,4,4,6,10])",
        "latent_channels": 64,
        "channels": 2,
        "sample_rate_hz": 48000,
        "hop_samples": 1920,
        "channel_in_latent": False,
        "sd_py_lines": "651-674",
    },
    "minimax_h3_audio_vae_fp32.safetensors": {
        "modality": "audio",
        "class": "comfy.ldm.minimax.audio_vae.MiniMaxH3AudioVAE",
        "latent_channels": 32,
        "channels": 2,
        "sample_rate_hz": 32000,
        "hop_samples": 800,
        "channel_in_latent": True,
        "sd_py_lines": "965-985",
    },
    "minimax_h3_video_vae_fp16.safetensors": {
        "modality": "video",
        "class": "comfy.ldm.minimax.vae.MiniMaxH3VideoVAE",
        "latent_channels": 24,
        "spatial_stride": 16,
        "temporal_stride": 4,
        "sd_py_lines": "942-964",
    },
}


class _PermissiveStub:
    """Stands in for a quantized-kernel symbol we never call."""

    def __getattr__(self, name: str) -> "_PermissiveStub":
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> "_PermissiveStub":
        return self

    def __bool__(self) -> bool:
        return False


def _stub_module_getattr(name: str) -> "_PermissiveStub":
    """Module-level ``__getattr__`` (PEP 562) for the stub packages.

    Dunder lookups MUST raise: ``torchvision`` imports later in the same chain and its
    op registration walks ``inspect.getmodule`` over every loaded module, which asks each
    one for ``__file__`` and calls ``TypeError: is a built-in module`` on anything that
    answers with a non-string. Handing back a stub for ``__file__`` breaks an unrelated
    import three libraries away -- measured, not hypothetical.
    """

    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _PermissiveStub()


def _install_comfy_kitchen_stub() -> None:
    import sys as _sys
    import types

    module = types.ModuleType("comfy_kitchen")
    module.__file__ = "<mediafold.vae_adapters comfy_kitchen stub>"
    module.__path__ = []  # importable as a package
    module.__spec__ = None
    # symbols comfy.quant_ops touches at import time and that must answer honestly:
    # with no kernels available, the backend registry is empty.
    module.list_backends = lambda: {}  # type: ignore[attr-defined]
    module.available_backends = lambda: []  # type: ignore[attr-defined]
    module.__getattr__ = _stub_module_getattr  # type: ignore[attr-defined]
    _sys.modules.setdefault("comfy_kitchen", module)
    for child in ("comfy_kitchen.backends", "comfy_kitchen.backends.triton"):
        sub = types.ModuleType(child)
        sub.__file__ = f"<mediafold.vae_adapters {child} stub>"
        sub.__path__ = []
        sub.__spec__ = None
        sub.__getattr__ = _stub_module_getattr  # type: ignore[attr-defined]
        _sys.modules.setdefault(child, sub)


class ComfyVaeAdapter(VaeAdapter):
    """Load any of the campaign's parent VAEs through ComfyUI's own detector.

    Minimum loader, verified in ``comfy/sd.py`` inside image
    ``comfyui:codex-minimax-h3-runtime-env-r5-20260808``::

        import comfy.sd, comfy.utils
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))

    Nothing else -- no custom node, no pipeline, no sampler. ``comfy.sd.VAE`` picks the
    first-stage class from the state-dict keys and sets ``latent_channels``,
    ``latent_dim``, ``downscale_ratio``, ``output_channels`` and ``audio_sample_rate``,
    which is where this adapter reads its geometry from; the profile table above is a
    cross-check, never the source of truth.

    Two layout facts, read out of ``sd.py`` rather than assumed, because getting either
    one wrong silently produces garbage pairs:

    * ``VAE.encode`` (sd.py:1251-1254) does ``pixel_samples.movedim(-1, 1)`` on its
      argument -- the public API is CHANNELS-LAST, which is why comfy's own
      ``VAEEncodeAudio`` node calls it as ``vae.encode(waveform.movedim(1, -1))``;
    * ``VAE.decode`` returns ``pixel_samples.movedim(1, -1)``, channels-last again.

    This adapter converts on both sides, so the rest of the harness only ever sees
    ``[B, C, S]`` audio and ``[B, T, C, H, W]`` video.
    """

    #: Where ComfyUI lives inside the runtime image. Running a script from /work means
    #: cwd is not on sys.path, so the root is added explicitly rather than relying on
    #: the caller remembering PYTHONPATH.
    COMFY_ROOTS = ("/opt/ComfyUI", "/workspace/ComfyUI")

    def __init__(
        self,
        path: str,
        *,
        device: str = "cpu",
        profile: str | None = None,
        posterior: str = "mean",
        path_mode: str = "auto",
    ) -> None:
        import os
        import sys as _sys

        if posterior not in ("mean", "sample"):
            raise ValueError("posterior must be 'mean' or 'sample'")
        if path_mode not in ("auto", "direct", "wrapper"):
            raise ValueError("path_mode must be 'auto', 'direct' or 'wrapper'")
        self.posterior = str(posterior)
        self._requested_path_mode = str(path_mode)

        for root in (os.environ.get("COMFYUI_ROOT", ""), *self.COMFY_ROOTS):
            if root and os.path.isdir(os.path.join(root, "comfy")) and root not in _sys.path:
                _sys.path.insert(0, root)

        # ComfyUI decides CPU-vs-GPU ONCE, at `comfy.model_management` import time, from
        # `comfy.cli_args.args` (model_management.py:212, `get_torch_device`). Importing
        # `comfy.sd` with no visible CUDA device therefore dies before a single weight is
        # read, with `RuntimeError: Found no NVIDIA driver`. Flipping `args.cpu` first is
        # the supported switch, and it happens here rather than in the caller because the
        # failure is an import-time crash a long way from its cause.
        #
        # GPU note for the r5 runtime image (measured 2026-09-02): it bakes
        # `CUDA_VISIBLE_DEVICES=""` and pins `NVIDIA_VISIBLE_DEVICES` to one GPU UUID, so
        # even `docker run --gpus device=0` leaves torch with zero devices. The working
        # invocation is
        #     docker run --gpus device=<n> -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0
        wants_cpu = str(device) == "cpu" or not torch.cuda.is_available()
        forced_cpu = False
        try:
            import comfy.options  # noqa: F401
            from comfy.cli_args import args as comfy_args

            if wants_cpu:
                comfy_args.cpu = True
                forced_cpu = True
        except Exception:
            comfy_args = None

        shimmed = False
        try:
            import comfy.sd  # noqa: F401  (import fails loudly outside the image, on purpose)
        except Exception as error:
            # `comfy.quant_ops` imports `comfy_kitchen`, whose triton backend enumerates a
            # CUDA driver at IMPORT time; with no driver visible that enumeration raises
            # ("0 active drivers"). None of it is reachable from a VAE encode/decode, so it
            # is stubbed out rather than worked around at the docker level. Nothing on disk
            # is modified.
            if "comfy_kitchen" not in repr(error) and "driver" not in repr(error):
                raise
            _install_comfy_kitchen_stub()
            shimmed = True
            import comfy.sd  # noqa: F401
        import comfy.utils

        state = comfy.utils.load_torch_file(path)
        self.vae = comfy.sd.VAE(sd=state)
        key = profile or path.rsplit("/", 1)[-1]
        table = COMFY_VAE_PROFILES.get(key, {})

        latent_dim = int(getattr(self.vae, "latent_dim", 0) or 0)
        sample_rate = int(getattr(self.vae, "audio_sample_rate", 0) or 0)
        downscale = getattr(self.vae, "downscale_ratio", None)
        # comfy.sd.VAE gives EVERY vae a default audio_sample_rate = 44100 -- the H3
        # VIDEO vae reports 44100 too (measured) -- so the sample rate is not a modality
        # signal. The signals that are: latent_dim == 3, or a tuple downscale_ratio, means
        # a video geometry (the ratio is (frames_fn, 16, 16)); a SCALAR ratio in the
        # hundreds or thousands is a waveform hop, while an image VAE's scalar ratio is
        # 8 / 16 / 32.
        scalar_ratio = isinstance(downscale, (int, float))
        if latent_dim == 3 or isinstance(downscale, (tuple, list)):
            modality = "video"
        elif scalar_ratio and int(downscale) >= 64 and latent_dim in (1, 2):
            modality = "audio"
        else:
            modality = "image"

        hop = 0
        spatial = 0
        if modality == "audio":
            hop = int(downscale) if isinstance(downscale, (int, float)) else 0
        else:
            spatial = int(downscale[-1]) if isinstance(downscale, (tuple, list)) else int(downscale or 0)

        self.latent_dim = latent_dim
        self.modality = modality
        # "auto": the public wrapper for audio (verified correct on both audio VAEs), the
        # first-stage call for video (the wrapper crops the temporal axis -- see
        # `_direct_encode`). Override with path_mode="wrapper"/"direct" to measure both.
        if self._requested_path_mode == "auto":
            self.path_mode = "direct" if modality == "video" else "wrapper"
        else:
            self.path_mode = self._requested_path_mode
        self.geometry = VaeGeometry(
            name=key,
            modality=modality,
            latent_channels=int(getattr(self.vae, "latent_channels", 0) or table.get("latent_channels", 0)),
            sample_rate_hz=sample_rate,
            hop_samples=hop,
            channels=int(getattr(self.vae, "output_channels", 0) or table.get("channels", 0) or 0),
            # `latent_dim == 2` on an audio VAE is comfy's marker for a per-waveform-channel
            # latent axis: MiniMax H3 audio is [B, 32, stereo 2, T] (sd.py:973).
            channel_in_latent=bool(modality == "audio" and latent_dim == 2),
            spatial_stride=spatial,
            temporal_stride=int(table.get("temporal_stride", 0)),
            notes={
                "kind": "comfy",
                "path": path,
                "comfy_kitchen_stubbed": shimmed,
                "forced_cpu": forced_cpu,
                "expected_class": table.get("class"),
                "sd_py_lines": table.get("sd_py_lines"),
                "latent_dim": latent_dim,
                "first_stage_class": type(self.vae.first_stage_model).__name__,
                "vae_dtype": str(getattr(self.vae, "vae_dtype", "")),
                "device": str(getattr(self.vae, "device", device)),
                "posterior": self.posterior,
                "path_mode": self.path_mode,
                "crop_input": bool(getattr(self.vae, "crop_input", False)),
                "downscale_ratio": self._describe_ratio(downscale),
                "upscale_ratio": self._describe_ratio(getattr(self.vae, "upscale_ratio", None)),
            },
        )

    @staticmethod
    def _describe_ratio(value: Any) -> Any:
        """Render comfy's ratio field for JSON: an int for audio, a
        ``(callable, spatial, spatial)`` tuple for video."""

        if isinstance(value, (tuple, list)):
            return [("callable" if callable(item) else item) for item in value]
        if callable(value):
            return "callable"
        return value

    def latent_frame_count(self, frames: int) -> int:
        """Comfy's own frames -> latent-frames formula for this VAE (video only).

        For the MiniMax H3 video VAE that formula is
        ``max(1, (frames - 5) // 17 * 5 + 2)`` (sd.py:949) -- NOT a uniform 4x, so the
        pair sampler is told the alignment explicitly rather than inferring one.
        """

        ratio = getattr(self.vae, "downscale_ratio", None)
        if isinstance(ratio, (tuple, list)) and callable(ratio[0]):
            return int(ratio[0](int(frames)))
        return 0

    def posterior_mean_available(self) -> bool:
        """True when this VAE's sampling step can be pinned to the posterior mean."""

        bottleneck = getattr(self.vae.first_stage_model, "bottleneck", None)
        return bottleneck is not None and hasattr(bottleneck, "encode")

    @contextlib.contextmanager
    def _pinned_posterior(self):
        """Temporarily replace a diagonal-gaussian bottleneck's sample with its mean.

        MEASURED, not assumed: the ACE 1.5 VAE's ``encode`` is NOT deterministic --
        ``comfy/ldm/audio/autoencoder.py:10`` draws ``mean + softplus(scale) * randn``,
        and two encodes of the same clip differ by up to 1.4e-2 in latent units (probe
        receipt ``ace_1.5_vae_probe.json``). Fitting a C15 map against sampled latents
        charges the map for noise no map can predict, so pairs are sampled at the
        posterior mean by default. The patch is scoped to one call and restored in a
        ``finally`` -- comfy's own device / dtype / tiling path still runs.
        """

        bottleneck = getattr(self.vae.first_stage_model, "bottleneck", None)
        if self.posterior != "mean" or bottleneck is None or not hasattr(bottleneck, "encode"):
            yield False
            return
        original = bottleneck.encode

        def mean_only(x, return_info=False, **kwargs):
            mean = x.chunk(2, dim=1)[0]
            if return_info:
                return mean, {"posterior": "mean"}
            return mean

        bottleneck.encode = mean_only
        try:
            yield True
        finally:
            bottleneck.encode = original

    def _load_weights(self, memory_required: float, *, force_full_load: bool = True) -> None:
        """Put the whole first-stage model on the device before a direct call.

        ``force_full_load`` defaults to True because comfy's partial (lowvram) load leaves
        non-parameter module state behind: with another lane holding 9 GiB of the same
        card, the H3 video decoder reached
        ``vae.py:292 torch.cat([h, self.register_tokens.expand(...)]) -> Expected all
        tensors to be on the same device, but found at least two devices, cuda:0 and cpu``.
        A full load either fits or fails loudly; a partial load fails in the middle of
        someone else's module.
        """

        import comfy.model_management as model_management

        model_management.load_models_gpu(
            [self.vae.patcher],
            memory_required=float(memory_required),
            force_full_load=bool(force_full_load or getattr(self.vae, "disable_offload", False)),
        )

    def _direct_encode(self, channels_first: torch.Tensor) -> torch.Tensor:
        """``first_stage_model.encode`` with comfy's memory manager but NOT its cropper.

        MEASURED (probe receipt ``minimax_h3_video_vae_fp16_probe.json``, first run):
        ``VAE.encode`` starts with ``vae_encode_crop_pixels``, which crops every axis of
        ``pixels.shape[1:-1]`` to a multiple of ``spacial_compression_encode()``. On a
        5-D channels-last video that set includes the TEMPORAL axis, so 17, 21, 22 and 26
        frames were all silently cropped to 16 and produced the same 2 latent frames --
        while the VAE's own map (sd.py:949) says 22 frames should give 7. Pair rows built
        on that path would be aligned to frames that were never encoded.

        The first-stage call takes ``[B, C, T, H, W]`` and does the same
        ``process_input`` the wrapper does, so the numbers are the parent's own.
        """

        vae = self.vae
        try:
            memory = float(vae.memory_used_encode(tuple(channels_first.shape), vae.vae_dtype))
        except Exception:
            memory = 0.0
        self._load_weights(memory)
        prepared = vae.process_input(channels_first).to(device=vae.device, dtype=vae.vae_dtype)
        # no_grad is load-bearing, not hygiene. comfy's public VAE.encode/decode wrap
        # themselves; the first-stage call does not, and its parameters carry
        # requires_grad=True, so every returned latent keeps a live autograd graph. Rows are
        # accumulated across a whole corpus here, which pinned one graph per clip: the H3
        # video VAE reached 23.84 GiB on a 24 GiB P40 at 64 px and died (measured twice).
        with torch.no_grad(), self._pinned_posterior():
            if getattr(vae.first_stage_model, "comfy_has_chunked_io", False):
                out = vae.first_stage_model.encode(prepared, device=vae.device)
            else:
                out = vae.first_stage_model.encode(prepared)
        return out.detach().to(device=vae.output_device, dtype=torch.float32)

    def _direct_decode(self, latent: torch.Tensor) -> torch.Tensor:
        vae = self.vae
        try:
            memory = float(vae.memory_used_decode(tuple(latent.shape), vae.vae_dtype))
        except Exception:
            memory = 0.0
        self._load_weights(memory)
        with torch.no_grad():
            out = vae.first_stage_model.decode(latent.to(device=vae.device, dtype=vae.vae_dtype))
        return vae.process_output(out).detach().to(device=vae.output_device, dtype=torch.float32)

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        if self.modality == "audio":
            if self.path_mode == "direct":
                return self._direct_encode(raw)  # [B, C, S] is already channels-first
            with self._pinned_posterior():
                return self.vae.encode(raw.movedim(1, -1))  # comfy's public API is channels-last
        if self.path_mode == "direct":
            return self._direct_encode(raw.movedim(1, 2))  # [B, T, C, H, W] -> [B, C, T, H, W]
        with self._pinned_posterior():
            return self.vae.encode(raw.movedim(2, -1))  # [B, T, C, H, W] -> [B, T, H, W, C]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.path_mode == "direct":
            out = self._direct_decode(latent)
            if self.modality == "audio":
                return out  # [B, C, S]
            return out.movedim(1, 2)  # [B, C, T, H, W] -> [B, T, C, H, W]
        out = self.vae.decode(latent)
        if self.modality == "audio":
            return out.movedim(-1, 1)  # [B, S, C] -> [B, C, S]
        return out.movedim(-1, 2)  # [B, T, H, W, C] -> [B, T, C, H, W]


# --------------------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------------------


class PairFileAdapter(VaeAdapter):
    """Serve pairs sampled earlier; ``encode``/``decode`` are unavailable by design."""

    def __init__(self, geometry: VaeGeometry) -> None:
        self.geometry = geometry

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("PairFileAdapter replays stored pairs; it cannot run the parent VAE")

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("PairFileAdapter replays stored pairs; it cannot run the parent VAE")


def build_adapter(
    spec: str, *, device: str = "cpu", posterior: str = "mean", path_mode: str = "auto"
) -> VaeAdapter:
    """``synthetic-audio``, ``synthetic-video`` or ``comfy:<path>``."""

    if spec == "synthetic-audio":
        return SyntheticAudioVae()
    if spec == "synthetic-video":
        return SyntheticVideoVae()
    if spec.startswith("comfy:"):
        return ComfyVaeAdapter(
            spec.split(":", 1)[1], device=device, posterior=posterior, path_mode=path_mode
        )
    raise ValueError(f"unknown adapter spec {spec!r}")


def analytic_ceiling_psnr(adapter: VaeAdapter, raw: torch.Tensor) -> float:
    """PSNR of the parent's own round-trip -- the number absorption is measured against."""

    from .linalg import psnr

    reconstruction = adapter.roundtrip(raw)
    limit = min(reconstruction.shape[-1], raw.shape[-1])
    return psnr(reconstruction[..., :limit], raw[..., :limit])


def procedural_audio(
    count: int,
    *,
    samples: int,
    channels: int = 2,
    sample_rate_hz: int = 32000,
    seed: int = 0,
) -> torch.Tensor:
    """Procedural stereo corpus: harmonic stacks with onsets, sweeps and noise beds.

    Procedural only -- no model outputs are used as fold data anywhere in this package.
    """

    if int(channels) <= 0:
        raise ValueError("channels must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    time = torch.arange(samples, dtype=torch.float64) / float(sample_rate_hz)
    clips = []
    for _ in range(int(count)):
        base = 110.0 * (2.0 ** (float(torch.randint(0, 24, (1,), generator=generator).item()) / 12.0))
        partials = 1 + int(torch.randint(0, 5, (1,), generator=generator).item())
        sweep = float(torch.rand(1, generator=generator).item()) * 0.5
        onset = int(float(torch.rand(1, generator=generator).item()) * 0.5 * samples)
        mono = torch.zeros(samples, dtype=torch.float64)
        for harmonic in range(1, partials + 1):
            frequency = base * harmonic * (1.0 + sweep * time / max(1e-9, time[-1].item()))
            mono = mono + torch.sin(2.0 * math.pi * frequency * time) / harmonic
        mono = mono / max(1.0, float(partials))
        # each extra channel is the same source delayed by 2 ms, so stereo carries a real
        # inter-channel relationship rather than a duplicate
        lag = max(1, int(0.002 * sample_rate_hz))
        voices = [torch.roll(mono, shifts=index * lag) for index in range(int(channels))]
        noise = 0.02 * torch.randn(int(channels), samples, generator=generator, dtype=torch.float64)
        block = torch.stack(voices, dim=0) * 0.6 + noise
        block[:, :onset] = 0.0
        clips.append(block)
    return torch.stack(clips).to(torch.float32).clamp(-1.0, 1.0)


def procedural_video(
    count: int,
    *,
    frames: int,
    size: int,
    channels: int = 3,
    seed: int = 0,
) -> torch.Tensor:
    """Procedural clips: moving squares, gradients and a drifting sinusoidal texture."""

    generator = torch.Generator().manual_seed(int(seed))
    clips = []
    ys = torch.arange(size, dtype=torch.float64).view(-1, 1)
    xs = torch.arange(size, dtype=torch.float64).view(1, -1)
    for index in range(int(count)):
        speed = 1.0 + float(torch.rand(1, generator=generator).item()) * 2.0
        phase = float(torch.rand(1, generator=generator).item()) * math.pi
        block = max(2, size // 4)
        row = int(torch.randint(0, max(1, size - block), (1,), generator=generator).item())
        clip = torch.zeros(frames, channels, size, size, dtype=torch.float64)
        for frame in range(frames):
            texture = 0.5 + 0.5 * torch.sin(0.4 * xs + 0.3 * ys + speed * frame * 0.5 + phase)
            clip[frame, 0] = texture
            clip[frame, min(1, channels - 1)] = 1.0 - texture
            column = int((frame / max(1, frames - 1)) * (size - block))
            clip[frame, :, row : row + block, column : column + block] = 1.0
        clips.append(clip)
    return torch.stack(clips).to(torch.float32)
