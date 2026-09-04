"""Self-contained, weight-faithful ACE-Step 1.5 turbo DiT.

Transcribed from ComfyUI's own ``/opt/ComfyUI/comfy/ldm/ace/ace_step15.py`` inside image
``comfyui:codex-minimax-h3-runtime-env-r5-20260808`` (read 2026-09-01), so the block
computation is the parent's, not an approximation:

* ``AceStepDiTLayer.forward``      ace_step15.py:249-...  (transcribed line for line)
* ``AceStepAttention.forward``     q_norm/k_norm on the head axis BEFORE transpose;
                                   RoPE only on self-attention; sliding-window additive
                                   bias of ``|i-j| <= window``; GQA by head repeat
* ``RotaryEmbedding``              base 1e6, ``emb = cat(freqs, freqs)``, ``rotate_half``
* ``MLP``                          SwiGLU ``down(silu(gate(x)) * up(x))``
* ``TimestepEmbedding``            flux ``timestep_embedding`` (cos||sin, time_factor
                                   1000) -> linear_1 -> SiLU -> linear_2 -> SiLU ->
                                   time_proj -> [B, 6, hidden]
* ``AceStepDiTModel.forward``      ``x = cat([context_latents, hidden_states], -1)``,
                                   Conv1d patchify (kernel = stride = patch_size),
                                   24 layers, ``norm_out * (1+scale) + shift``,
                                   ConvTranspose1d unpatchify

Why transcribe instead of importing: the parent runs in a python-3.10 / torch-2.6
container behind ``comfy.ops``, while every fold solve in this campaign runs in the
lab venv (python 3.12 / torch 2.11 / fp64 solvers). A standalone module lets the block
be captured, charted, transplanted and coreset-folded in one process, and it keeps the
harness free of ComfyUI imports.

Verified geometry from ``/home/cereal/comfyui/models/ACE-Step/acestep-v15-turbo/config.json``
and the 677-tensor header of ``acestep_v1.5_turbo.safetensors``: hidden 2048, 24 layers
alternating ``sliding_attention(128) / full_attention``, 16 q heads / 8 kv heads x 128,
intermediate 6144 SiLU, in_channels 192 (= 64 source + 64 chunk-mask + 64 noisy),
patch_size 2, audio_acoustic_hidden_dim 64, rope_theta 1e6, rms_norm_eps 1e-6.
Per-block tensors: ``self_attn.{q,k,v,o}_proj``, ``self_attn.{q,k}_norm``,
``cross_attn.*`` (same shapes), ``mlp.{gate,up,down}_proj``,
``{self_attn,cross_attn,mlp}_norm``, ``scale_shift_table [1,6,2048]``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ACE_DEFAULT_CONFIG_PATH = "/home/cereal/comfyui/models/ACE-Step/acestep-v15-turbo/config.json"
ACE_DEFAULT_WEIGHTS_PATH = "/home/cereal/comfyui/models/diffusion_models/acestep_v1.5_turbo.safetensors"
ACE_WEIGHT_PREFIX = "decoder."


@dataclass
class AceStepDiTConfig:
    in_channels: int = 192
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    patch_size: int = 2
    audio_acoustic_hidden_dim: int = 64
    text_hidden_dim: int = 1024
    condition_dim: int = 2048
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 1.0e6
    sliding_window: int = 128
    layer_types: tuple[str, ...] = field(default_factory=lambda: tuple(
        ("sliding_attention", "full_attention") * 12
    ))

    @classmethod
    def from_json(cls, path: str | Path = ACE_DEFAULT_CONFIG_PATH) -> "AceStepDiTConfig":
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            in_channels=int(body["in_channels"]),
            hidden_size=int(body["hidden_size"]),
            num_hidden_layers=int(body["num_hidden_layers"]),
            num_attention_heads=int(body["num_attention_heads"]),
            num_key_value_heads=int(body["num_key_value_heads"]),
            head_dim=int(body["head_dim"]),
            intermediate_size=int(body["intermediate_size"]),
            patch_size=int(body["patch_size"]),
            audio_acoustic_hidden_dim=int(body["audio_acoustic_hidden_dim"]),
            text_hidden_dim=int(body["text_hidden_dim"]),
            condition_dim=int(body["hidden_size"]),
            rms_norm_eps=float(body["rms_norm_eps"]),
            rope_theta=float(body["rope_theta"]),
            sliding_window=int(body["sliding_window"]),
            layer_types=tuple(str(value) for value in body["layer_types"]),
        )

    def layer_type(self, index: int) -> str:
        if not self.layer_types:
            return "full_attention"
        return self.layer_types[index % len(self.layer_types)]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """ComfyUI's RMSNorm == ``F.rms_norm``: ``x * rsqrt(mean(x^2)+eps) * weight``."""

    return F.rms_norm(x, (weight.shape[-1],), weight, eps=eps)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(width)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(length: int, head_dim: int, base: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(length, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def apply_rope(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0, time_factor: float = 1000.0) -> torch.Tensor:
    """Flux-style sinusoidal embedding (cos block first, then sin), as ACE uses it."""

    scaled = time_factor * t
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    arguments = scaled[:, None].float() * frequencies[None]
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
    if dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    if torch.is_floating_point(scaled):
        embedding = embedding.to(scaled)
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, width: int, scale: float = 1000.0) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(int(in_channels), int(width))
        self.linear_2 = nn.Linear(int(width), int(width))
        self.time_proj = nn.Linear(int(width), int(width) * 6)
        self.in_channels = int(in_channels)
        self.scale = float(scale)

    def forward(self, t: torch.Tensor, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = timestep_embedding(t, self.in_channels, time_factor=self.scale)
        temb = self.linear_1(frequencies.to(dtype=dtype or self.linear_1.weight.dtype))
        temb = F.silu(temb)
        temb = self.linear_2(temb)
        projected = self.time_proj(F.silu(temb)).view(-1, 6, temb.shape[-1])
        return temb, projected


class MLP(nn.Module):
    def __init__(self, width: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(int(width), int(intermediate), bias=False)
        self.up_proj = nn.Linear(int(width), int(intermediate), bias=False)
        self.down_proj = nn.Linear(int(intermediate), int(width), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The gated channel activations -- the C16 coreset selects columns of THIS."""

        return F.silu(self.gate_proj(x)) * self.up_proj(x)


class AceStepAttention(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        eps: float,
        *,
        is_cross_attention: bool = False,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = int(heads)
        self.num_kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.is_cross_attention = bool(is_cross_attention)
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(int(width), self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(int(width), self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(int(width), self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, int(width), bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)

        if self.is_cross_attention and encoder_hidden_states is not None:
            source = encoder_hidden_states
            kv_len = source.shape[1]
            key = self.k_proj(source).view(source.shape[0], kv_len, self.num_kv_heads, self.head_dim)
            key = self.k_norm(key).transpose(1, 2)
            value = self.v_proj(source).view(source.shape[0], kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            kv_len = length
            key = self.k_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim)
            key = self.k_norm(key).transpose(1, 2)
            value = self.v_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
            if position_embeddings is not None:
                query, key = apply_rope(query, key, position_embeddings[0], position_embeddings[1])

        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        bias = None
        if self.sliding_window is not None and not self.is_cross_attention:
            index = torch.arange(length, device=query.device)
            in_window = (index.unsqueeze(1) - index.unsqueeze(0)).abs() <= int(self.sliding_window)
            bias = torch.zeros((length, kv_len), device=query.device, dtype=query.dtype)
            bias.masked_fill_(~in_window, torch.finfo(query.dtype).min)
            bias = bias.unsqueeze(0).unsqueeze(0)

        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=bias)
        attended = attended.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.o_proj(attended)


class AceStepDiTLayer(nn.Module):
    """One ACE-Step DiT block. This is the C16 transplant unit."""

    def __init__(
        self,
        width: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        intermediate: int,
        eps: float = 1.0e-6,
        *,
        layer_type: str = "full_attention",
        sliding_window: int = 128,
    ) -> None:
        super().__init__()
        window = int(sliding_window) if layer_type == "sliding_attention" else None
        self.layer_type = layer_type
        self.self_attn_norm = RMSNorm(width, eps)
        self.self_attn = AceStepAttention(width, heads, kv_heads, head_dim, eps, sliding_window=window)
        self.cross_attn_norm = RMSNorm(width, eps)
        self.cross_attn = AceStepAttention(width, heads, kv_heads, head_dim, eps, is_cross_attention=True)
        self.mlp_norm = RMSNorm(width, eps)
        self.mlp = MLP(width, intermediate)
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, int(width)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        modulation = self.scale_shift_table.to(temb.dtype) + temb
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        normed = self.self_attn_norm(hidden_states)
        normed = normed * (1 + scale_msa) + shift_msa
        hidden_states = hidden_states + self.self_attn(normed, position_embeddings=position_embeddings) * gate_msa

        if encoder_hidden_states is not None:
            normed = self.cross_attn_norm(hidden_states)
            hidden_states = hidden_states + self.cross_attn(normed, encoder_hidden_states=encoder_hidden_states)

        normed = self.mlp_norm(hidden_states)
        normed = normed * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + self.mlp(normed) * gate_mlp
        return hidden_states

    def mlp_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.mlp.parameters())

    def block_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())


class AceStepDiTModel(nn.Module):
    def __init__(self, config: AceStepDiTConfig, *, num_layers: int | None = None) -> None:
        super().__init__()
        self.config = config
        depth = int(config.num_hidden_layers if num_layers is None else num_layers)
        self.depth = depth
        self.proj_in = nn.Sequential(
            nn.Identity(),
            nn.Conv1d(config.in_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size),
        )
        self.time_embed = TimestepEmbedding(256, config.hidden_size)
        self.time_embed_r = TimestepEmbedding(256, config.hidden_size)
        self.condition_embedder = nn.Linear(config.condition_dim, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                AceStepDiTLayer(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.intermediate_size,
                    config.rms_norm_eps,
                    layer_type=config.layer_type(index),
                    sliding_window=config.sliding_window,
                )
                for index in range(depth)
            ]
        )
        self.norm_out = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.proj_out = nn.Sequential(
            nn.Identity(),
            nn.ConvTranspose1d(
                config.hidden_size,
                config.audio_acoustic_hidden_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            ),
        )
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        temb_t, projection_t = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_r, projection_r = self.time_embed_r(timestep - timestep_r, dtype=hidden_states.dtype)
        temb = temb_t + temb_r
        timestep_projection = projection_t + projection_r

        x = torch.cat((context_latents, hidden_states), dim=-1)
        original_length = x.shape[1]
        pad = (-x.shape[1]) % self.config.patch_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        x = self.proj_in(x.transpose(1, 2)).transpose(1, 2)

        context = self.condition_embedder(encoder_hidden_states)
        cos, sin = rope_tables(x.shape[1], self.config.head_dim, self.config.rope_theta, x.device, x.dtype)

        for layer in self.layers:
            x = layer(x, timestep_projection, encoder_hidden_states=context, position_embeddings=(cos, sin))

        shift, scale = (self.scale_shift_table.to(temb.dtype) + temb.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm_out(x) * (1 + scale) + shift
        x = self.proj_out(x.transpose(1, 2)).transpose(1, 2)
        return x[:, :original_length, :]

    def block_inputs(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Everything a single block needs, without running the block stack.

        Used by the C16 capture driver so one transplanted block can be exercised
        without materialising all 24.
        """

        _, projection_t = self.time_embed(timestep, dtype=hidden_states.dtype)
        _, projection_r = self.time_embed_r(timestep - timestep_r, dtype=hidden_states.dtype)
        x = torch.cat((context_latents, hidden_states), dim=-1)
        pad = (-x.shape[1]) % self.config.patch_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        x = self.proj_in(x.transpose(1, 2)).transpose(1, 2)
        cos, sin = rope_tables(x.shape[1], self.config.head_dim, self.config.rope_theta, x.device, x.dtype)
        return {
            "hidden": x,
            "temb": projection_t + projection_r,
            "context": self.condition_embedder(encoder_hidden_states),
            "cos": cos,
            "sin": sin,
        }


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def _read_prefixed(path: str | Path, prefix: str, wanted: set[str] | None) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    block: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith(prefix):
                continue
            short = key[len(prefix) :]
            if wanted is not None and short not in wanted:
                continue
            block[short] = handle.get_tensor(key)
    return block


def load_ace_dit_block(
    index: int,
    *,
    weights: str | Path = ACE_DEFAULT_WEIGHTS_PATH,
    config: AceStepDiTConfig | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[AceStepDiTLayer, dict[str, Any]]:
    """Load ONE DiT block plus a structural receipt.

    The receipt asserts every source tensor of that block was consumed exactly once and
    lists each one's sha256, so a transplant can be traced back to the parent bytes.
    """

    from . import receipts as receipt_module

    settings = config or AceStepDiTConfig.from_json()
    block = AceStepDiTLayer(
        settings.hidden_size,
        settings.num_attention_heads,
        settings.num_key_value_heads,
        settings.head_dim,
        settings.intermediate_size,
        settings.rms_norm_eps,
        layer_type=settings.layer_type(int(index)),
        sliding_window=settings.sliding_window,
    )
    prefix = f"{ACE_WEIGHT_PREFIX}layers.{int(index)}."
    source = _read_prefixed(weights, prefix, None)
    if not source:
        raise FileNotFoundError(f"no tensors under {prefix!r} in {weights}")
    expected = set(block.state_dict().keys())
    missing = sorted(expected - set(source))
    unexpected = sorted(set(source) - expected)
    if missing or unexpected:
        raise RuntimeError(f"ACE block {index} tensor mismatch; missing={missing} unexpected={unexpected}")
    block.load_state_dict({name: value.to(torch.float32) for name, value in source.items()}, strict=True)
    block = block.to(device=device, dtype=dtype)
    receipt = {
        "source": str(weights),
        "prefix": prefix,
        "layer_type": block.layer_type,
        "consumed_tensors": len(source),
        "tensor_sha256": {name: receipt_module.hash_tensor(value) for name, value in sorted(source.items())},
        "block_parameters": block.block_parameters(),
        "mlp_parameters": block.mlp_parameters(),
    }
    return block, receipt


def load_ace_dit(
    *,
    weights: str | Path = ACE_DEFAULT_WEIGHTS_PATH,
    config: AceStepDiTConfig | None = None,
    num_layers: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[AceStepDiTModel, dict[str, Any]]:
    """Load the DiT (optionally a truncated depth for cheap plumbing smokes)."""

    from . import receipts as receipt_module

    settings = config or AceStepDiTConfig.from_json()
    model = AceStepDiTModel(settings, num_layers=num_layers)
    wanted = set(model.state_dict().keys())
    source = _read_prefixed(weights, ACE_WEIGHT_PREFIX, wanted)
    missing = sorted(wanted - set(source))
    if missing:
        raise RuntimeError(f"ACE DiT missing tensors: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    model.load_state_dict({name: value.to(torch.float32) for name, value in source.items()}, strict=True)
    model = model.to(device=device, dtype=dtype)
    return model, {
        "source": str(weights),
        "prefix": ACE_WEIGHT_PREFIX,
        "depth": model.depth,
        "consumed_tensors": len(source),
        "state_sha256": receipt_module.hash_state({k: v for k, v in source.items()}),
        "parameters": sum(int(p.numel()) for p in model.parameters()),
        "config": {
            "hidden_size": settings.hidden_size,
            "num_attention_heads": settings.num_attention_heads,
            "num_key_value_heads": settings.num_key_value_heads,
            "head_dim": settings.head_dim,
            "intermediate_size": settings.intermediate_size,
            "patch_size": settings.patch_size,
            "layer_types": list(settings.layer_types),
        },
    }
