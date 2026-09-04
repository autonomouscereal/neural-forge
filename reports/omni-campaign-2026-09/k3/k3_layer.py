#!/usr/bin/env python3
"""One Kimi-K3 decoder layer, loaded from shards and run without the model.

This is a functional transcription of `modeling_kimi_linear.py`:
`KimiDeltaAttention` / `KimiMLAAttention` / `KimiSparseMoeBlock` /
`KimiDecoderLayer._forward_attn_residual`.  It exists because the repo's own
`KimiLinearModel.__init__` forces `flash_attention_2` (line 1110-1117) -- which
is not installed -- and because instantiating 93 layers is 1.42 TiB the box
does not have.  The layer object here owns nothing but its own weights and can
be dropped the moment the streamed pass moves on.

The KDA half has two backends:

* ``fla``   -- ``fla.ops.kda.chunk_kda`` / ``fused_recurrent_kda``, i.e. exactly
              what the repo calls.  Triton: GPU only.
* ``torch`` -- a pure-torch transcription of ``naive_recurrent_kda`` plus the
              gate, the short convolution and the gated output norm.  Runs on
              CPU, is the decode path for the dense child, and is the thing
              ``k3_receipt.py`` measures against ``fla``.

Gate semantics are not guessed: with ``safe_gate=True`` and
``lower_bound=-5.0`` fla's own reference (`fla/ops/kda/gate.py`,
``naive_kda_lowerbound_gate``) is

    g = lower_bound * sigmoid(exp(A_log).view(H, 1) * (g_raw + dt_bias.view(H, K)))

which fixes the A_log axis to ``num_heads`` (96), not head_dim.
"""
from __future__ import annotations

import re
import time
from typing import Any

import torch
import torch.nn.functional as F

from k3_common import (
    K3Index,
    apply_attn_res,
    canonical_device,
    dequant_mxfp4,
    log,
    rms_norm,
    same_device,
    situ,
)

#: fla's `FusedRMSNormGated` normalises first and gates afterwards
#: (`layernorm_gated.rms_norm_ref`, norm_before_gate=True).  `k3_receipt.py
#: --gpu-crosscheck` measures both orders and records which one matched; this
#: constant is what the harness runs.
NORM_BEFORE_GATE = True
#: fla's l2norm epsilon.  q/k are unit-norm afterwards, so a wrong eps here is
#: a ~1e-6 relative perturbation, not a behaviour change -- but it is recorded.
L2NORM_EPS = 1e-6
#: MLA's two LATENT norms are built as bare `KimiRMSNorm(rank)` -- no eps
#: argument -- so they run at the class default 1e-6, NOT at
#: `config.rms_norm_eps` (1e-5 for K3).  Every other norm in the model passes
#: the config value explicitly.  Using 1e-5 here is an algebra error, not a
#: rounding one: measured at fp32 on the synthetic checkpoint it moved the MLA
#: output by relL2 6.06e-2, and it survived the switch to fp32, which is how it
#: was caught (`t08_mla_matches_the_repo_module`).
MLA_LATENT_NORM_EPS = 1e-6


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------
class LayerWeights:
    """The tensors of one decoder layer, grouped by role.

    ``experts`` stays PACKED (uint8) -- 14.8 GB instead of 59.2 GB for a real
    K3 MoE layer -- and is dequantized per touched expert inside ``moe``.
    """

    EXPERT_RE = re.compile(r"^block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.(weight_packed|weight_scale)$")

    def __init__(self, layer_idx: int, tensors: dict[str, torch.Tensor], index: K3Index):
        self.layer_idx = layer_idx
        self.index = index
        self.is_kda = index.is_kda_layer(layer_idx)
        self.is_moe = index.is_moe_layer(layer_idx)
        prefix = index.layer_prefix_for(layer_idx)

        self.attn: dict[str, torch.Tensor] = {}
        self.res: dict[str, torch.Tensor] = {}
        self.body: dict[str, torch.Tensor] = {}
        self.experts: dict[int, dict[str, torch.Tensor]] = {}

        for key, tensor in tensors.items():
            short = key[len(prefix):] if key.startswith(prefix) else key
            match = self.EXPERT_RE.match(short)
            if match:
                expert = int(match.group(1))
                self.experts.setdefault(expert, {})[f"{match.group(2)}.{match.group(3)}"] = tensor
                continue
            if short.startswith("self_attn."):
                self.attn[short[len("self_attn."):]] = tensor
            elif short.startswith(("input_layernorm", "post_attention_layernorm",
                                   "self_attention_res", "mlp_res")):
                self.res[short] = tensor
            else:
                self.body[short] = tensor

    # -- placement ---------------------------------------------------------
    def to(self, device: torch.device | str, experts_device: torch.device | str | None = None) -> "LayerWeights":
        """Move the layer; experts may live on a different device than the body."""
        experts_device = device if experts_device is None else experts_device
        for group in (self.attn, self.res, self.body):
            for key, tensor in group.items():
                group[key] = tensor.to(device)
        for expert in self.experts.values():
            for key, tensor in expert.items():
                expert[key] = tensor.to(experts_device)
        return self

    def device_bytes(self, device: torch.device | str) -> dict[str, int]:
        """BYTES of this layer that are resident on `device`, measured per group.

        Bytes, not parameters, and measured rather than derived: this feeds
        `c14_union_coreset.device_memory_estimate`, whose whole job is to
        refuse a rung before an allocator does.  The estimator used to price
        only the per-chunk transients and ignored the fact that
        `load_layer(index, layer, device, device)` puts all 896 PACKED experts
        on the compute device -- 14.8 GB of `weight_packed` plus 0.93 GB of
        E8M0 scales on a 25.3 GB card, i.e. ~94% of what actually lands there.

        The comparison goes through `same_device`, never a raw `==`: a tensor
        moved to `cuda` reports `cuda:0`, so asking this for `cuda` used to
        answer 0 for a layer that was entirely resident (t13f).
        """
        device = canonical_device(device)

        def total(group) -> int:
            return sum(t.numel() * t.element_size() for t in group
                       if same_device(t.device, device))

        expert_tensors = [t for expert in self.experts.values() for t in expert.values()]
        groups = {
            "attention": total(self.attn.values()),
            "residual": total(self.res.values()),
            "body": total(self.body.values()),
            "experts_packed": total(expert_tensors),
        }
        groups["total"] = sum(groups.values())
        return groups

    def parameter_counts(self) -> dict[str, int]:
        """Parameters, not bytes -- storage is counted in PARAMETERS in this program.

        `weight_scale` (one E8M0 byte per 32 packed values) is counted
        SEPARATELY and is NOT in `experts`.  That is the conservative choice for
        a fold ratio -- it makes the parent look smaller -- but it means the
        `experts` figure is ~0.92 G short of what is on disk per real K3 MoE
        layer, so the number is reported rather than left implicit.
        """
        expert_elements = 0
        scale_elements = 0
        for expert in self.experts.values():
            for key, tensor in expert.items():
                if key.endswith("weight_packed"):
                    expert_elements += tensor.numel() * 2  # a uint8 carries two fp4 values
                elif key.endswith("weight_scale"):
                    scale_elements += tensor.numel()
        return {
            "attention": sum(t.numel() for t in self.attn.values()),
            "residual": sum(t.numel() for t in self.res.values()),
            "body": sum(t.numel() for t in self.body.values()),
            "experts": expert_elements,
            "expert_scale_elements": scale_elements,
            "expert_scale_note": ("E8M0 group scales, one per 32 packed values, NOT counted in "
                                  "`experts`; excluding them understates the parent"),
            "expert_count": len(self.experts),
        }


def load_layer(
    index: K3Index,
    layer_idx: int,
    device: str = "cpu",
    experts_device: str | None = None,
    log_reads: bool = True,
) -> tuple[LayerWeights, dict]:
    """Stream one layer off the shards and return it with its read timing."""
    keys = index.layer_keys(layer_idx)
    if not keys:
        raise SystemExit(f"layer {layer_idx}: the index carries no tensors under {index.layer_prefix_for(layer_idx)}")
    tensors, timing = index.read_keys(keys, progress_tag=f"L{layer_idx:02d}" if log_reads else None)
    weights = LayerWeights(layer_idx, tensors, index)
    weights.to(device, experts_device)
    return weights, timing


# --------------------------------------------------------------------------
# KDA
# --------------------------------------------------------------------------
def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv, fla's `ShortConvolution` with bias=False.

    x       [B, T, D]
    weight  [D, 1, W]
    cache   [B, D, W-1] -- OUR convention (the last W-1 inputs), returned
            updated so a decode step can carry it.  fla stores [N, D, W]; we
            never hand this cache to fla, so the narrower form is used.
    """
    B, T, D = x.shape
    width = weight.shape[-1]
    xt = x.transpose(1, 2)  # [B, D, T]
    if cache is None:
        padded = F.pad(xt, (width - 1, 0))
    else:
        padded = torch.cat((cache.to(xt.dtype), xt), dim=-1)
    y = F.conv1d(padded, weight, bias=None, groups=D)
    new_cache = padded[..., -(width - 1):] if width > 1 else padded[..., :0]
    y = y.transpose(1, 2)
    if activation == "silu":
        y = F.silu(y)
    elif activation is not None:
        raise SystemExit(f"unsupported short-conv activation {activation!r}")
    return y, new_cache


def kda_gate_axis(a_log: torch.Tensor, heads: int, head_dim: int) -> str:
    """Which axis `A_log` indexes -- and on the real K3 it is NOT the one fla assumes.

    fla's reference (`naive_kda_lowerbound_gate`) does ``A_log.view(H, 1)``, and
    `KimiDeltaAttention.__init__` declares ``A_log`` as ``[num_heads]``.  The
    shipped Kimi-K3 checkpoint disagrees: a header-only census of
    `language_model.model.layers.1.self_attn.A_log` gives f32 **[128]**, which
    is `head_dim`, while `num_heads` is 96 (`k3_layer1_shapes.json`; b_proj is
    [96, 7168] and dt_bias is [12288] = 96 x 128, so 96/128 are not swapped).

    Consequences, both real:
      * `load_state_dict(strict=True)` into the repo's own KimiDeltaAttention
        fails on this tensor.
      * `chunk_kda(..., use_gate_in_kernel=True)` cannot consume it, because the
        kernel does the `view(H, 1)` internally.

    So the axis is resolved by shape and the decay is precomputed in torch when
    it is the key-dim one.  WHICH AXIS IS SEMANTICALLY CORRECT IS NOT SETTLED
    BY THIS FILE -- it is a decision line for Cyril (D-a2b) and needs a
    behavioural receipt on real weights.
    """
    if a_log.numel() == heads and heads != head_dim:
        return "heads"
    if a_log.numel() == head_dim and heads != head_dim:
        return "key_dim"
    if a_log.numel() == heads:  # heads == head_dim: ambiguous but equivalent in shape
        return "heads"
    raise SystemExit(
        f"A_log has {a_log.numel()} elements, which is neither num_heads={heads} nor "
        f"head_dim={head_dim}; the KDA gate cannot be formed"
    )


def kda_gate(
    g_raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    axis: str = "heads",
) -> torch.Tensor:
    """fla `naive_kda_gate` / `naive_kda_lowerbound_gate`, fp32.

    g_raw [..., H, K];  dt_bias [H*K];  a_log [H] (axis="heads", fla's form) or
    [K] (axis="key_dim", what the shipped checkpoint actually carries).
    """
    heads = g_raw.shape[-2]
    g = g_raw.float()
    if dt_bias is not None:
        g = g + dt_bias.float().view(heads, -1)
    a = a_log.float().exp()
    a = a.view(heads, 1) if axis == "heads" else a.view(1, -1)
    if lower_bound is None:
        return -a * F.softplus(g)
    return lower_bound * torch.sigmoid(a * g)


def l2norm(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + eps).to(x.dtype)


def rms_norm_gated(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str = "sigmoid",
    norm_before_gate: bool | None = None,
) -> torch.Tensor:
    """fla `FusedRMSNormGated(head_dim, activation='sigmoid')`, in torch.

    `norm_before_gate` defaults to the module constant AT CALL TIME, not at
    definition time: `k3_receipt.py` measures both orders by flipping
    `NORM_BEFORE_GATE`, and a bound default would have made that check a
    no-op that silently reported the two orders as identical.
    """
    if norm_before_gate is None:
        norm_before_gate = NORM_BEFORE_GATE
    act = torch.sigmoid(gate.float()) if activation == "sigmoid" else F.silu(gate.float())
    if norm_before_gate:
        y = rms_norm(x, weight, eps)
        return (y.float() * act).to(x.dtype)
    y = (x.float() * act).to(x.dtype)
    return rms_norm(y, weight, eps)


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fla's `naive_recurrent_kda`, fp32, with the state always returned.

    q,k [B,T,H,K]   v [B,T,H,V]   g [B,T,H,K] (log space)   beta [B,T,H]
    state [B,H,K,V]
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    q = q * scale
    state = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.float().clone()
    out = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        state = state * g_i[..., None].exp()
        delta = v_i - (k_i[..., None] * state).sum(-2)
        state = state + torch.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, delta)
        out[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, state)
    return out.to(dtype), state


class KDAState:
    """Per-layer KDA decode state: three conv caches plus the recurrent state."""

    __slots__ = ("conv_q", "conv_k", "conv_v", "recurrent")

    def __init__(self) -> None:
        self.conv_q = self.conv_k = self.conv_v = None
        self.recurrent = None


def kda_forward(
    weights: LayerWeights,
    x: torch.Tensor,
    cfg: dict,
    backend: str = "torch",
    state: KDAState | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """KimiDeltaAttention.forward on a [B, T, hidden] block. Returns (o, taps)."""
    linear = cfg["linear_attn_config"]
    heads = int(linear["num_heads"])
    head_dim = int(linear["head_dim"])
    lower_bound = linear.get("gate_lower_bound")
    w = weights.attn

    q_lin = F.linear(x, w["q_proj.weight"])
    k_lin = F.linear(x, w["k_proj.weight"])
    v_lin = F.linear(x, w["v_proj.weight"])
    conv_q = state.conv_q if state else None
    conv_k = state.conv_k if state else None
    conv_v = state.conv_v if state else None
    q, conv_q = short_conv(q_lin, w["q_conv1d.weight"], conv_q)
    k, conv_k = short_conv(k_lin, w["k_conv1d.weight"], conv_k)
    v, conv_v = short_conv(v_lin, w["v_conv1d.weight"], conv_v)

    g_raw = F.linear(F.linear(x, w["f_a_proj.weight"]), w["f_b_proj.weight"])
    g_raw = g_raw.view(*g_raw.shape[:-1], heads, head_dim)
    beta_raw = F.linear(x, w["b_proj.weight"]).float()
    q = q.view(*q.shape[:-1], heads, head_dim)
    k = k.view(*k.shape[:-1], heads, head_dim)
    v = v.view(*v.shape[:-1], heads, head_dim)

    a_log = w["A_log"]
    dt_bias = w.get("dt_bias")
    axis = kda_gate_axis(a_log, heads, head_dim)

    initial = state.recurrent if state else None
    gate_log = None
    if backend == "fla":
        from fla.ops.kda import chunk_kda, fused_recurrent_kda

        kernel = fused_recurrent_kda if (state is not None and x.shape[1] == 1) else chunk_kda
        if axis == "heads":
            # the repo's own call: the kernel fuses the gate
            out, recurrent = kernel(
                q=q, k=k, v=v, g=g_raw, beta=beta_raw,
                A_log=a_log, dt_bias=dt_bias,
                initial_state=initial, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=lower_bound is not None, lower_bound=lower_bound,
                transpose_state_layout=True,
            )
        else:
            # the kernel's fused gate hardcodes A_log.view(H, 1); with a key-dim
            # A_log the decay has to be formed outside and handed in already
            # computed (fla's documented use_gate_in_kernel=False path)
            gate_log = kda_gate(g_raw, a_log, dt_bias, lower_bound, axis=axis)
            out, recurrent = kernel(
                q=q, k=k, v=v, g=gate_log.to(q.dtype), beta=beta_raw,
                initial_state=initial, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=True,
                transpose_state_layout=True,
            )
    elif backend == "torch":
        gate_log = kda_gate(g_raw, a_log, dt_bias, lower_bound, axis=axis)
        out, recurrent = naive_recurrent_kda(
            l2norm(q).float(), l2norm(k).float(), v.float(), gate_log,
            torch.sigmoid(beta_raw), scale=head_dim ** -0.5, initial_state=initial,
        )
        out = out.to(v.dtype)
    else:
        raise SystemExit(f"unknown KDA backend {backend!r}")

    gate_out = F.linear(x, w["g_proj.weight"]).view(*x.shape[:-1], heads, head_dim)
    normed = rms_norm_gated(out, gate_out, w["o_norm.weight"], float(cfg["rms_norm_eps"]))
    o = F.linear(normed.reshape(*x.shape[:-1], heads * head_dim), w["o_proj.weight"])

    if state is not None:
        state.conv_q, state.conv_k, state.conv_v = conv_q, conv_k, conv_v
        state.recurrent = recurrent
    taps = {"kda_core": out, "kda_gate_out": gate_out, "kda_gate_axis": axis}
    if gate_log is not None:
        taps["kda_gate_log"] = gate_log
    return o, taps


# --------------------------------------------------------------------------
# MLA
# --------------------------------------------------------------------------
class MLAState:
    """Per-layer MLA decode cache (full K/V, exactly like KimiDynamicCache)."""

    __slots__ = ("key", "value")

    def __init__(self) -> None:
        self.key = None
        self.value = None


def mla_forward(
    weights: LayerWeights,
    x: torch.Tensor,
    cfg: dict,
    state: MLAState | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """KimiMLAAttention.forward, NoPE, optional output gate, SDPA causal."""
    w = weights.attn
    heads = int(cfg["num_attention_heads"])
    nope = int(cfg["qk_nope_head_dim"])
    rope = int(cfg["qk_rope_head_dim"])
    v_dim = int(cfg["v_head_dim"])
    kv_lora = int(cfg["kv_lora_rank"])
    q_head_dim = nope + rope
    B, T, _ = x.shape

    latent_eps = MLA_LATENT_NORM_EPS
    if "q_a_proj.weight" in w:
        q_states = F.linear(
            rms_norm(F.linear(x, w["q_a_proj.weight"]), w["q_a_layernorm.weight"], latent_eps),
            w["q_b_proj.weight"],
        )
    else:
        q_states = F.linear(x, w["q_proj.weight"])
    q_states = q_states.view(B, T, -1, q_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [nope, rope], dim=-1)

    compressed = F.linear(x, w["kv_a_proj_with_mqa.weight"])
    k_lat, k_rot = torch.split(compressed, [kv_lora, rope], dim=-1)
    k_pass = F.linear(rms_norm(k_lat, w["kv_a_layernorm.weight"], latent_eps), w["kv_b_proj.weight"])
    k_pass = k_pass.view(B, T, -1, nope + v_dim).transpose(1, 2)
    k_pass, value = torch.split(k_pass, [nope, v_dim], dim=-1)
    k_rot = k_rot.view(B, 1, T, rope).expand(*k_pass.shape[:-1], -1)

    query = torch.cat((q_pass, q_rot), dim=-1)
    key = torch.cat((k_pass, k_rot), dim=-1)
    if state is not None:
        if state.key is None:
            state.key, state.value = key, value
        else:
            state.key = torch.cat((state.key, key), dim=2)
            state.value = torch.cat((state.value, value), dim=2)
        key, value = state.key, state.value

    # SDPA's `is_causal` aligns the mask TOP-LEFT, so it is only the right mask
    # when the query block starts at position 0.  Every caller today is either a
    # full prefill (state empty) or a single decode step (T == 1); a CHUNKED
    # prefill -- T > 1 appended to a non-empty cache -- would be silently
    # non-causal, so it is refused rather than answered wrongly.
    cached = int(key.shape[2]) - T  # 0 on a fresh prefill, N on a decode step
    if T > 1 and cached > 0:
        raise SystemExit(
            f"mla_forward: chunked prefill (T={T} queries against {cached} cached keys) needs an "
            "explicit bottom-right-aligned mask; SDPA's is_causal would align top-left and run "
            "non-causally.  Prefill in one block or add the mask before using this path."
        )
    causal = T > 1
    attn = F.scaled_dot_product_attention(
        query.contiguous(), key.contiguous(), value.contiguous(),
        is_causal=causal, scale=q_head_dim ** -0.5,
    )
    attn = attn.transpose(1, 2).reshape(B, T, heads * v_dim)
    if "g_proj.weight" in w:
        attn = attn * torch.sigmoid(F.linear(x, w["g_proj.weight"]))
    return F.linear(attn, w["o_proj.weight"]), {"mla_core": attn}


# --------------------------------------------------------------------------
# MoE (latent, SiTU, packed experts)
# --------------------------------------------------------------------------
def route(x: torch.Tensor, weights: LayerWeights, cfg: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KimiMoEGate.forward -- fp32 logits, sigmoid, bias-corrected top-k."""
    gate_w = weights.body["block_sparse_moe.gate.weight"]
    bias = weights.body["block_sparse_moe.gate.e_score_correction_bias"]
    flat = x.reshape(-1, x.shape[-1])
    logits = F.linear(flat.float(), gate_w.float(), None)
    if cfg.get("moe_router_activation_func", "sigmoid") == "sigmoid":
        scores = logits.sigmoid()
    else:
        scores = logits.softmax(dim=1)
    top_k = int(cfg["num_experts_per_token"])
    _, topk_idx = torch.topk(scores + bias.float().unsqueeze(0), k=top_k, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)
    if top_k > 1 and cfg.get("moe_renormalize", True):
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * float(cfg.get("routed_scaling_factor", 1.0))
    return topk_idx, topk_weight, scores


def dequant_expert(weights: LayerWeights, expert: int, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """The 22 MB bf16 scratch for one expert, built on demand from the packed pair."""
    packed = weights.experts[expert]
    out = {}
    for name in ("w1", "w2", "w3"):
        out[name] = dequant_mxfp4(
            packed[f"{name}.weight_packed"], packed[f"{name}.weight_scale"], dtype=dtype
        )
    return out


def moe_forward(
    weights: LayerWeights,
    x: torch.Tensor,
    cfg: dict,
    collect: bool = True,
    dense: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """KimiSparseMoeBlock.forward with the experts kept packed.

    ``dense`` -- when given, the routed half is replaced by the C14 form-E
    child (`w1_s`/`w3_s`/`w2p`); the router is not consulted at all, which is
    the whole point of the fold: 6.4M router parameters are DELETED.
    """
    eps = float(cfg["rms_norm_eps"])
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    body = weights.body
    flat = x.reshape(-1, x.shape[-1])
    latent = F.linear(flat, body["block_sparse_moe.routed_expert_down_proj.weight"])
    taps: dict[str, Any] = {}
    if collect:
        taps["moe_x"] = flat.detach()
        taps["latent_in"] = latent.detach()

    if dense is not None:
        # the coreset ships gate/up rows in the parent's dtype but W2' comes out
        # of an fp64 ridge solve as fp32; match the activation's dtype so the
        # child's arithmetic is the parent's, not a silently promoted one
        gate_h = F.linear(latent, dense["w1_s"].to(latent.dtype))
        up_h = F.linear(latent, dense["w3_s"].to(latent.dtype))
        activated = situ(gate_h, up_h, beta, linear_beta)
        y_lat = F.linear(activated, dense["w2p"].to(activated.dtype)).float()
        taps["dense_channels"] = int(dense["w1_s"].shape[0])
    else:
        topk_idx, topk_weight, _ = route(x, weights, cfg)
        y_lat = torch.zeros(latent.shape[0], latent.shape[1], dtype=torch.float32, device=latent.device)
        counts = torch.bincount(topk_idx.flatten(), minlength=int(cfg["num_experts"]))
        touched = torch.nonzero(counts).flatten().tolist()
        for expert in touched:
            hit = (topk_idx == expert)
            rows = hit.any(dim=1).nonzero().flatten()
            weight = topk_weight[rows].masked_fill(~hit[rows], 0.0).sum(dim=1)
            ex = dequant_expert(weights, expert, dtype=latent.dtype)
            chunk = latent[rows]
            activated = situ(F.linear(chunk, ex["w1"]), F.linear(chunk, ex["w3"]), beta, linear_beta)
            y_lat[rows] += F.linear(activated, ex["w2"]).float() * weight.unsqueeze(-1)
        if collect:
            taps["router_topk_idx"] = topk_idx.detach().to(torch.int16)
            taps["router_topk_w"] = topk_weight.detach().to(torch.bfloat16)
            taps["router_counts"] = counts.detach().cpu()
            taps["router_mass"] = torch.zeros(int(cfg["num_experts"]), dtype=torch.float64).index_add_(
                0, topk_idx.flatten().cpu(), topk_weight.flatten().double().cpu()
            )
    if collect:
        taps["y_lat"] = y_lat.detach()

    y = rms_norm(y_lat.to(latent.dtype), body["block_sparse_moe.routed_expert_norm.weight"], eps)
    routed = F.linear(y, body["block_sparse_moe.routed_expert_up_proj.weight"])
    shared = F.linear(
        situ(
            F.linear(flat, body["block_sparse_moe.shared_experts.gate_proj.weight"]),
            F.linear(flat, body["block_sparse_moe.shared_experts.up_proj.weight"]),
            beta, linear_beta,
        ),
        body["block_sparse_moe.shared_experts.down_proj.weight"],
    )
    out = (routed + shared).view(*x.shape)
    return out, taps


def dense_mlp_forward(weights: LayerWeights, x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """KimiMLP (layer 0's dense replacement), SiTU."""
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    body = weights.body
    return F.linear(
        situ(F.linear(x, body["mlp.gate_proj.weight"]), F.linear(x, body["mlp.up_proj.weight"]),
             beta, linear_beta),
        body["mlp.down_proj.weight"],
    )


# --------------------------------------------------------------------------
# the layer
# --------------------------------------------------------------------------
class StreamedLayer:
    """`KimiDecoderLayer._forward_attn_residual`, weights supplied from outside."""

    def __init__(self, weights: LayerWeights, cfg: dict, kda_backend: str = "torch"):
        self.w = weights
        self.cfg = cfg
        self.kda_backend = kda_backend
        self.eps = float(cfg["rms_norm_eps"])
        self.block_size = cfg.get("attn_res_block_size")

    # -- halves ------------------------------------------------------------
    def attention(self, x: torch.Tensor, state: Any | None = None) -> tuple[torch.Tensor, dict]:
        if self.w.is_kda:
            return kda_forward(self.w, x, self.cfg, backend=self.kda_backend, state=state)
        return mla_forward(self.w, x, self.cfg, state=state)

    def feed_forward(self, x: torch.Tensor, collect: bool = True,
                     dense: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, dict]:
        if self.w.is_moe:
            return moe_forward(self.w, x, self.cfg, collect=collect, dense=dense)
        return dense_mlp_forward(self.w, x, self.cfg), {}

    # -- the layer ---------------------------------------------------------
    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        state: Any | None = None,
        collect: bool = True,
        dense: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """prefix_sum [B, T, H], block_residual [B*T, blocks, H].

        Exactly `_forward_attn_residual`, including the reset of `prefix_sum`
        to None on a snapshot layer -- the attention output BECOMES the prefix
        sum there rather than being added to it.
        """
        if self.block_size is None:
            raise SystemExit("this harness only implements the attn-residual variant (attn_res_block_size set)")
        batch, seq, hidden = prefix_sum.shape
        taps: dict[str, Any] = {}
        hidden_states = prefix_sum
        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = apply_attn_res(
                prefix_sum.view(-1, hidden), block_residual,
                self.w.res["self_attention_res_proj.weight"], self.w.res["self_attention_res_norm.weight"],
                self.eps,
            ).view(batch, seq, hidden)

        carried: torch.Tensor | None = prefix_sum
        if self.w.layer_idx % self.block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, hidden).unsqueeze(1)], dim=1
            )
            carried = None

        normed = rms_norm(hidden_states, self.w.res["input_layernorm.weight"], self.eps)
        attn_out, attn_taps = self.attention(normed, state=state)
        taps.update(attn_taps)
        if collect:
            taps["attn_in"] = normed.detach()
            taps["attn_out"] = attn_out.detach()
        carried = attn_out if carried is None else carried + attn_out

        hidden_states = apply_attn_res(
            carried.view(-1, hidden), block_residual,
            self.w.res["mlp_res_proj.weight"], self.w.res["mlp_res_norm.weight"], self.eps,
        ).view(batch, seq, hidden)
        hidden_states = rms_norm(hidden_states, self.w.res["post_attention_layernorm.weight"], self.eps)
        ffn_out, ffn_taps = self.feed_forward(hidden_states, collect=collect, dense=dense)
        taps.update(ffn_taps)
        if collect:
            taps["moe_out"] = ffn_out.detach()
        carried = carried + ffn_out
        if collect:
            taps["prefix_out"] = carried.detach()
        return carried, block_residual, taps


def new_state(index: K3Index, layer_idx: int):
    """The decode state object a layer needs, or None for a prefill-only pass."""
    return KDAState() if index.is_kda_layer(layer_idx) else MLAState()


def timed(fn, *args, **kwargs):
    """Wall-clock a call, synchronising CUDA so the number means something."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.time() - start
