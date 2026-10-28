"""``ChartedSourceBlock``: the torch side of the ``charted`` block kind.

See ``chimera_omni.charted`` for the contract. The block is the port of
``mediafold.c16_transplant.TransplantedBlock`` (ACE-Step block computation transcribed
in ``mediafold.ace_dit``) generalised to the five S6B sources by a handful of spec
switches (norm kind, MLP kind, attention bias, q/k norm, RoPE, sliding window,
modulation). Per row it computes

    delta = (block(chart_in(x), temb) - chart_in(x)) * delta_scale
    write = delta @ chart_out.weight.T          # no bias: a zero delta writes nothing

and returns ``write`` scattered onto the rows it owns (native-media rows whose type id
is in ``spec.type_ids``); every other row gets an exact zero. The rows a block owns are
gathered per sample into one packed sequence. Legacy source RoPE positions and the
sliding window use ranks inside that media sequence. H3 additionally reads text
context and requires explicit source THW coordinates for every attended row.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from chimera_omni.charted import H3_FP32_PARAMETERS, MODULATION_CHUNKS, ChartedBlockSpec


def timestep_embedding(t: torch.Tensor, dim: int, *, max_period: float = 10000.0, time_factor: float = 1000.0) -> torch.Tensor:
    """Flux-style sinusoidal embedding (cos block first, then sin), as ACE-Step uses it."""

    scaled = time_factor * t.float()
    half = int(dim) // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    arguments = scaled[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
    if int(dim) % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def rope_tables(length: int, head_dim: int, base: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(length, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos(), embedding.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def h3_plain_source_layout(text_length: int, latent_frames: int, latent_height: int,
                           latent_width: int, audio_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The parent's plain [text | stereo audio | video] THW grid, without refs.

    Returns source positions [tokens,3] and Chimera row types [tokens]. The
    caller must explicitly align/scatter these rows to its actual trunk tokens;
    native packet coordinates and raw-patch row counts are NOT interchangeable.
    Audio uses channel-major packing; video is a 1x2x2 latent patch grid.
    """
    dimensions = (text_length, latent_frames, latent_height, latent_width, audio_frames)
    if any(isinstance(n, bool) or not isinstance(n, int) for n in dimensions):
        raise ValueError("H3 layout dimensions must be integers")
    if text_length < 0 or audio_frames < 0 or latent_frames < 1:
        raise ValueError("invalid H3 text/audio/video extent")
    if min(latent_height, latent_width) < 2 or latent_height % 2 or latent_width % 2:
        raise ValueError("H3 video latent grid must be positive and divisible by its 2x2 patch")
    text = torch.zeros(text_length, 3, dtype=torch.float64)
    text[:, 0] = torch.arange(text_length, dtype=torch.float64)
    area = math.sqrt(latent_height * latent_width)

    def axis(extent: int) -> torch.Tensor:
        ratio, count = extent / area, extent // 2
        return (torch.arange(count, dtype=torch.float64) * ratio / count + (1 - ratio) / 2) * 32

    height, width = axis(latent_height), axis(latent_width)
    hh, ww = torch.meshgrid(height, width, indexing="ij")
    frame = torch.stack((hh.flatten(), ww.flatten()), dim=-1)
    spans = torch.tensor([(1, 4, 4, 4, 4)[i % 5] * 5 / 3 for i in range(latent_frames)], dtype=torch.float64)
    video_times = text_length + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))
    video = torch.empty(latent_frames, len(frame), 3, dtype=torch.float64)
    video[..., 0] = video_times[:, None]
    video[..., 1:] = frame
    audio = torch.zeros(2 * audio_frames, 3, dtype=torch.float64)
    audio[:, 0] = (text_length + torch.arange(audio_frames, dtype=torch.float64)).repeat(2)
    audio[:audio_frames, 2], audio[audio_frames:, 2] = width[0], width[-1]
    types = torch.tensor([0] * text_length + [3] * len(audio) + [2] * (latent_frames * len(frame)))
    return torch.cat((text, audio, video.reshape(-1, 3))), types


class ChartedNorm(nn.Module):
    """RMSNorm (ACE / ViT style) or LayerNorm (MMDiT style), affine or not."""

    def __init__(self, width: int, spec: ChartedBlockSpec) -> None:
        super().__init__()
        self.kind = spec.norm_kind
        self.eps = float(spec.norm_eps)
        self.affine = bool(spec.norm_affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(int(width)))
            if self.kind == "layer":
                self.bias = nn.Parameter(torch.zeros(int(width)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.float()
        if self.kind == "layer":
            values = F.layer_norm(values, (values.shape[-1],), eps=self.eps)
        else:
            values = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.affine:
            values = values * self.weight.float()
            if self.kind == "layer":
                values = values + self.bias.float()
        return values.to(dtype=x.dtype)


class HeadRMSNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(head_dim)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.float()
        values = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (values * self.weight.float()).to(dtype=x.dtype)


class ChartedAttention(nn.Module):
    def __init__(self, spec: ChartedBlockSpec) -> None:
        super().__init__()
        width = int(spec.width)
        self.num_heads = int(spec.heads)
        self.num_kv_heads = int(spec.kv_heads)
        self.head_dim = int(spec.head_dim)
        self.rope_theta = float(spec.rope_theta)
        self.sliding_window = int(spec.sliding_window)
        self.h3_rope = spec.modulation == "h3_curve6"
        if self.h3_rope:
            self.rope_inv_freq = nn.Parameter(torch.zeros(int(spec.h3_rope_freqs)), requires_grad=False)
        bias = bool(spec.attn_bias)
        self.q_proj = nn.Linear(width, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, width, bias=bias)
        if spec.qk_norm:
            self.q_norm = HeadRMSNorm(self.head_dim, spec.norm_eps)
            self.k_norm = HeadRMSNorm(self.head_dim, spec.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, hidden: torch.Tensor, key_valid: torch.Tensor,
                source_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim)
        key = self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)
        if self.h3_rope:
            if source_positions is None or source_positions.shape != (batch, length, 3):
                raise ValueError("H3 attention requires explicit source positions [batch, packed, 3]")
            # The parent fuses per-head RMSNorm with partial split-half 3D RoPE.
            # Its rotation table is rounded to the activation dtype; norm/rotation
            # accumulation is FP32 and the unrotated tail is retained.
            angles = (source_positions.float().unsqueeze(-1) * self.rope_inv_freq.float()).flatten(-2)
            cos = angles.cos().to(query.dtype).float().unsqueeze(2)
            sin = angles.sin().to(query.dtype).float().unsqueeze(2)
            half = angles.shape[-1]

            def norm_rotate(values: torch.Tensor, norm: HeadRMSNorm) -> torch.Tensor:
                wide = values.float()
                wide = wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + norm.eps)
                wide = wide * norm.weight.float()
                left, right = wide[..., :half], wide[..., half:2 * half]
                return torch.cat((left * cos - right * sin, right * cos + left * sin,
                                  wide[..., 2 * half:]), dim=-1).to(values.dtype)

            query = norm_rotate(query, self.q_norm)
            key = norm_rotate(key, self.k_norm)
        elif self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.rope_theta > 0.0:
            cos, sin = rope_tables(length, self.head_dim, self.rope_theta, hidden.device)
            cos = cos.to(query.dtype).unsqueeze(0).unsqueeze(0)
            sin = sin.to(query.dtype).unsqueeze(0).unsqueeze(0)
            query = query * cos + _rotate_half(query) * sin
            key = key * cos + _rotate_half(key) * sin
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        mask = key_valid.to(device=hidden.device, dtype=torch.bool).view(batch, 1, 1, length)
        mask = mask.expand(batch, 1, length, length)
        if self.sliding_window > 0:
            index = torch.arange(length, device=hidden.device)
            in_window = (index.unsqueeze(1) - index.unsqueeze(0)).abs() <= self.sliding_window
            mask = mask & in_window.view(1, 1, length, length)
        # A padded query row attends nothing; let it see itself so SDPA never divides by zero.
        eye = torch.eye(length, dtype=torch.bool, device=hidden.device).view(1, 1, length, length)
        mask = mask | eye
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.o_proj(attended)


class ChartedSwiGLU(nn.Module):
    def __init__(self, width: int, inner: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(int(width), int(inner), bias=False)
        self.up_proj = nn.Linear(int(width), int(inner), bias=False)
        self.down_proj = nn.Linear(int(inner), int(width), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ChartedGeluMLP(nn.Module):
    def __init__(self, width: int, inner: int, *, tanh: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(width), int(inner))
        self.fc2 = nn.Linear(int(inner), int(width))
        self.approximate = "tanh" if tanh else "none"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate=self.approximate))


class ChartedTimestepEmbedding(nn.Module):
    """The source's own timestep path: sinusoid -> linear_1 -> SiLU -> linear_2 -> time_proj."""

    def __init__(self, spec: ChartedBlockSpec) -> None:
        super().__init__()
        hidden = spec.resolved_time_hidden()
        self.freq_dim = int(spec.time_freq_dim)
        self.time_factor = float(spec.time_factor)
        self.linear_1 = nn.Linear(self.freq_dim, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.time_proj = nn.Linear(hidden, MODULATION_CHUNKS * int(spec.width))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        frequencies = timestep_embedding(t.reshape(-1), self.freq_dim, time_factor=self.time_factor)
        temb = self.linear_1(frequencies.to(dtype=self.linear_1.weight.dtype))
        temb = self.linear_2(F.silu(temb))
        return self.time_proj(F.silu(temb))


class H3CurveModulation(nn.Module):
    """Verbatim pruned-H3 time basis and per-modality adaLN projection.

    Inputs are each row's actual sigma, not an unshifted common sampler time.
    Video/audio schedule mapping therefore happens in the caller, exactly once.
    The source interpolates its saved curve at t=1-sigma; there is no SiLU here.
    Frozen coefficients are counted in the checkpoint alongside learned tensors.
    """

    def __init__(self, spec: ChartedBlockSpec) -> None:
        super().__init__()
        self.width = int(spec.width)
        self.curve_table = nn.Parameter(torch.zeros(int(spec.h3_curve_grid), int(spec.h3_curve_dim)),
                                        requires_grad=False)
        self.proj = nn.Linear(int(spec.h3_curve_dim), 3 * MODULATION_CHUNKS * self.width)

    def curve(self, sigma: torch.Tensor) -> torch.Tensor:
        values = sigma.reshape(-1).float()
        if not bool(torch.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
            raise ValueError("H3 sigma must be finite and in [0, 1]")
        table = self.curve_table.float()
        pos = (1.0 - values) * (table.shape[0] - 1)
        lower = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[lower], table[lower + 1], (pos - lower).unsqueeze(1))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        value = F.linear(self.curve(sigma), self.proj.weight.float(), self.proj.bias.float())
        return value.view(-1, 3, MODULATION_CHUNKS, self.width)


class ChartedSourceBlock(nn.Module):
    def __init__(self, spec: ChartedBlockSpec, hidden_size: int) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        width = int(spec.width)
        self.hidden_size = int(hidden_size)
        self.chart_in = nn.Linear(self.hidden_size, width, bias=bool(spec.chart_in_bias))
        self.chart_out = nn.Linear(width, self.hidden_size, bias=False)
        self.attn_norm = ChartedNorm(width, spec)
        self.mlp_norm = ChartedNorm(width, spec)
        self.attn = ChartedAttention(spec)
        if spec.mlp_kind == "swiglu":
            self.mlp = ChartedSwiGLU(width, spec.mlp_width)
        else:
            self.mlp = ChartedGeluMLP(width, spec.mlp_width, tanh=spec.mlp_kind == "gelu_tanh")
        if spec.modulation == "adaln6":
            self.scale_shift_table = nn.Parameter(torch.zeros(MODULATION_CHUNKS, width))
            self.time_embed = ChartedTimestepEmbedding(spec)
            # ACE-Step's second embedding (decoder.time_embed_r): the parent drives it with
            # t - t_r and every capture used t_r = 0, so it sees the same noise level here.
            self.time_embed_r = ChartedTimestepEmbedding(spec) if spec.time_embed_r else None
        else:
            self.scale_shift_table = None
            self.time_embed = None
            self.time_embed_r = None
        self.h3_time = H3CurveModulation(spec) if spec.modulation == "h3_curve6" else None
        self.register_buffer(
            "routed_type_ids", torch.tensor(sorted(int(t) for t in spec.type_ids), dtype=torch.long), persistent=False
        )
        self.delta_scale = float(spec.delta_scale)

    def _apply(self, fn, recurse=True):
        # Module.to()/bfloat16() recurse through _apply, bypassing child .to().
        # Preserve source FP32 coefficient values BEFORE that recursion can round
        # them; widening its result afterward cannot recover those values. All
        # other parameters follow the requested dtype/device normally.
        if self.h3_time is None:
            return super()._apply(fn, recurse=recurse)
        saved = {}
        for name in H3_FP32_PARAMETERS:
            parameter = self.get_parameter(name)
            saved[name] = (None if parameter.is_meta else parameter.detach().clone(),
                           None if parameter.grad is None else parameter.grad.detach().clone())
        result = super()._apply(fn, recurse=recurse)
        for name, (value, gradient) in saved.items():
            parameter = self.get_parameter(name)
            parameter.data = (parameter.data.float() if value is None else
                              value.to(device=parameter.device, dtype=torch.float32))
            if gradient is not None:
                parameter.grad = gradient.to(device=parameter.device, dtype=torch.float32)
        return result

    # -- the source block on a packed media sequence -----------------------------------

    def time_path(self, t: torch.Tensor) -> torch.Tensor:
        """Source modulation: ACE ``[n,6,width]`` or H3 ``[n,3,6,width]``.

        Without the ``scale_shift_table``: this is exactly the ``temb`` a capture records
        (ACE: ``time_embed(t).proj + time_embed_r(t - 0).proj``), so a container's captured
        reference must be reproduced by THIS function before the block is installed.
        """

        if self.h3_time is not None:
            return self.h3_time(t)
        if self.time_embed is None:
            raise ValueError("a block without adaln6 modulation has no timestep path")
        flat = t.reshape(-1)
        modulation = self.time_embed(flat)
        if self.time_embed_r is not None:
            modulation = modulation + self.time_embed_r(flat)
        return modulation.view(flat.shape[0], MODULATION_CHUNKS, int(self.spec.width))

    def block(self, projected: torch.Tensor, noise_levels: torch.Tensor, key_valid: torch.Tensor,
              *, row_types: torch.Tensor | None = None,
              source_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Source forward on packed rows; H3 retains joint text/AV self-attention."""

        hidden = projected
        if self.h3_time is not None:
            if row_types is None or row_types.shape != hidden.shape[:2]:
                raise ValueError("H3 block requires packed row types")
            if bool((key_valid & ~((row_types == 0) | (row_types == 2) | (row_types == 3))).any()):
                raise ValueError("H3 context contains a non-text/non-AV row")
            modulation = self.time_path(noise_levels)
            tags = torch.where(row_types == 2, 0, torch.where(row_types == 3, 2, 1)).reshape(-1)
            selected = modulation[torch.arange(tags.numel(), device=tags.device), tags]
            selected = selected.view(*hidden.shape[:2], MODULATION_CHUNKS, hidden.shape[-1]).to(hidden.dtype)
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = selected.unbind(dim=2)
            normed = self.attn_norm(hidden) * (1.0 + scale_a) + shift_a
            hidden = torch.addcmul(hidden, self.attn(normed, key_valid, source_positions), gate_a)
            normed = self.mlp_norm(hidden) * (1.0 + scale_m) + shift_m
            return torch.addcmul(hidden, self.mlp(normed), gate_m)
        if self.time_embed is not None and self.scale_shift_table is not None:
            batch, length, width = hidden.shape
            # Keep sigma precision through the FP32 sinusoid: BF16 rounding is
            # amplified by time_factor. The embedding casts to weight dtype
            # only after computing the frequencies.
            modulation = self.time_path(noise_levels).view(batch, length, MODULATION_CHUNKS, width)
            modulation = modulation + self.scale_shift_table.to(dtype=hidden.dtype).view(1, 1, MODULATION_CHUNKS, width)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.unbind(dim=2)
            normed = self.attn_norm(hidden) * (1 + scale_msa) + shift_msa
            hidden = hidden + self.attn(normed, key_valid) * gate_msa
            normed = self.mlp_norm(hidden) * (1 + scale_mlp) + shift_mlp
            hidden = hidden + self.mlp(normed) * gate_mlp
            return hidden
        hidden = hidden + self.attn(self.attn_norm(hidden), key_valid)
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden

    def owned_rows(self, row_types: torch.Tensor | None, token_valid: torch.Tensor) -> torch.Tensor | None:
        if row_types is None:
            return None
        if row_types.shape != token_valid.shape:
            raise ValueError("charted row types must be [batch, sequence]")
        owned = torch.isin(row_types.to(device=token_valid.device, dtype=torch.long), self.routed_type_ids.to(token_valid.device))
        owned = owned & token_valid.to(device=token_valid.device, dtype=torch.bool)
        if not bool(owned.any()):
            return None
        return owned

    def forward(
        self,
        x: torch.Tensor,
        row_types: torch.Tensor | None,
        noise_levels: torch.Tensor | None,
        token_valid: torch.Tensor,
        *, source_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The residual write, ``[batch, sequence, hidden]``; exact zeros off the owned rows."""

        token_valid = token_valid.to(device=x.device, dtype=torch.bool)
        if row_types is not None:
            row_types = row_types.to(device=x.device, dtype=torch.long)
        owned = self.owned_rows(row_types, token_valid)
        if owned is None:
            return torch.zeros_like(x)
        batch, sequence, hidden = x.shape
        active = owned
        if self.h3_time is not None:
            # H3 uses joint self-attention, not a detachable cross-attention tower.
            # Read the trunk's text context through the same chart, but never write it.
            active = owned | (row_types.eq(0) & token_valid & owned.any(dim=1, keepdim=True))
            if source_positions is None or source_positions.shape != (batch, sequence, 3):
                raise ValueError("H3 requires source-coordinate evidence [batch, sequence, 3]; native positions are not interchangeable")
            if not bool(torch.isfinite(source_positions.to(x.device)[active]).all()):
                raise ValueError("H3 source positions must be finite on all attended rows")
            if noise_levels is None:
                raise ValueError("H3 requires actual per-type sigma values")
            if noise_levels.shape != (batch, sequence):
                raise ValueError("H3 sigma values must be [batch, sequence]")
            noise_levels = noise_levels.to(device=x.device, dtype=torch.float32).clone()
            # Plain joint AV captures use one sampler time. Text follows video;
            # an audio-only forward maps its sigma back to the video schedule.
            for sample in range(batch):
                if not bool(owned[sample].any()):
                    continue
                video = owned[sample] & row_types[sample].eq(2)
                audio = owned[sample] & row_types[sample].eq(3)
                values = noise_levels[sample, video] if bool(video.any()) else noise_levels[sample, audio]
                sigma_v = values[0]
                if not bool(video.any()):
                    base = sigma_v / (3.0 + sigma_v * (1.0 - 3.0))
                    sigma_v = 12.0 * base / (1.0 + 11.0 * base)
                base = sigma_v / (12.0 + sigma_v * (1.0 - 12.0))
                sigma_a = 3.0 * base / (1.0 + 2.0 * base)
                if (not torch.allclose(noise_levels[sample, video], sigma_v.expand(int(video.sum())), atol=1e-6, rtol=1e-6)
                        or not torch.allclose(noise_levels[sample, audio], sigma_a.expand(int(audio.sum())), atol=1e-6, rtol=1e-6)):
                    raise ValueError("H3 plain AV context requires matching shift-12 video / shift-3 audio schedules")
                noise_levels[sample, row_types[sample].eq(0)] = sigma_v
        counts = active.sum(dim=1)
        packed = int(counts.max().item())
        # gather each sample's owned rows into a packed [batch, packed, hidden] sequence
        order = torch.argsort((~active).to(torch.int8), dim=1, stable=True)[:, :packed]
        key_valid = torch.arange(packed, device=x.device).view(1, -1) < counts.view(-1, 1)
        gather_index = order.unsqueeze(-1).expand(batch, packed, hidden)
        rows = x.gather(1, gather_index)
        rows = torch.where(key_valid.unsqueeze(-1), rows, torch.zeros_like(rows))
        if noise_levels is None:
            levels = torch.zeros((batch, packed), device=x.device, dtype=torch.float32)
        else:
            if noise_levels.shape != owned.shape:
                raise ValueError("charted noise levels must be [batch, sequence]")
            levels = noise_levels.to(device=x.device, dtype=torch.float32).gather(1, order)
            levels = torch.where(key_valid, levels, torch.zeros_like(levels))
        projected = self.chart_in(rows)
        packed_types = row_types.to(x.device).gather(1, order)
        packed_positions = None
        if self.h3_time is not None:
            packed_positions = source_positions.to(x.device).gather(1, order.unsqueeze(-1).expand(batch, packed, 3))
            packed_positions = torch.where(key_valid.unsqueeze(-1), packed_positions, torch.zeros_like(packed_positions))
        delta = self.block(projected, levels, key_valid, row_types=packed_types,
                           source_positions=packed_positions) - projected
        if self.delta_scale != 1.0:
            delta = delta * self.delta_scale
        write = delta @ self.chart_out.weight.to(delta.dtype).transpose(0, 1)
        write_valid = key_valid & owned.gather(1, order)
        write = torch.where(write_valid.unsqueeze(-1), write, torch.zeros_like(write))
        out = torch.zeros_like(x)
        out.scatter_(1, gather_index, write.to(dtype=x.dtype))
        return out

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def mlp_parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.mlp.parameters())


__all__ = ["ChartedSourceBlock", "H3CurveModulation", "h3_plain_source_layout", "rope_tables", "timestep_embedding"]
