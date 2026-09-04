"""``qwen3_5`` architecture adapter -- every name the fold units are allowed to know.

The LFM2 units reached into ``model.model.layers[i].feed_forward.w1`` and
``config.norm_eps`` directly, which is why porting them cost a spec.  Nothing in
this campaign does that again: every unit asks this module for the module, the
gain or the count it wants, and this module is the ONE place that knows the
difference between ``Qwen3_5ForCausalLM`` (text-only; ``model.layers``) and
``Qwen3_5ForConditionalGeneration`` (the real checkpoints; ``model.language_model
.layers``), and the one place that knows the two RMSNorm gain conventions.

Verified against transformers 5.12.1's
``transformers/models/qwen3_5/modeling_qwen3_5.py`` on the server
(``/home/cereal/model-surgery-lab/venv``), 2,117 lines, read line by line:

* ``Qwen3_5RMSNorm.forward`` = ``_norm(x) * (1.0 + weight)`` with ``weight``
  initialised to ZEROS (line 740, 749) -- the effective gain is ``1 + w``, not
  ``w``.  LFM2's was ``w``.  Every gain reader and writer in this campaign goes
  through :func:`effective_gain` / :func:`store_gain` so that difference cannot
  be dropped in one call site and kept in another.
* ``Qwen3_5RMSNormGated.forward`` = ``weight * rmsnorm(o) * silu(z)`` (line 199)
  -- ``weight`` used DIRECTLY.  Two conventions in one model.
* ``Qwen3_5DecoderLayer`` carries ``self.layer_type`` (line 760) taken from
  ``config.layer_types[layer_idx]`` and the mixer is named ``linear_attn`` or
  ``self_attn`` accordingly (761-764).
* ``Qwen3_5TextModel.forward`` line 1201 dispatches the mask **by config list
  position**, not by the live module: ``linear_attn_mask if
  self.config.layer_types[i] == "linear_attention" else causal_mask``.  This is
  LFM2's wave-18 trap with a different spelling and it is why
  :func:`assert_layer_types_agree` exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

FAMILY = "qwen3_5"

#: The two values ``config.layer_types`` is allowed to take.
LINEAR = "linear_attention"
FULL = "full_attention"


# ------------------------------------------------------------------ loaders
def force_torch_kernels() -> dict[str, bool]:
    """Disable every optional fused kernel in the qwen3_5 module, in place.

    Three of them are CUDA-only and one of them *constructs* on CUDA:
    ``FusedRMSNormGated.__init__`` calls ``torch.cuda.current_device()``
    (modeling line 413), so a CPU-only instantiation of a GDN layer raises
    before a single weight is read.  ``chunk_gated_delta_rule`` and
    ``fused_recurrent_gated_delta_rule`` are Triton.  The torch fallbacks are in
    the same file (``torch_chunk_gated_delta_rule`` 246, ``torch_recurrent_
    gated_delta_rule`` 327) and are what every fp64 recompute in this campaign
    compares against anyway.

    Returns what was actually switched off so a receipt can say so.  Idempotent.
    """

    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod

    was = {
        "FusedRMSNormGated": mod.FusedRMSNormGated is not None,
        "chunk_gated_delta_rule": mod.chunk_gated_delta_rule is not None,
        "fused_recurrent_gated_delta_rule": mod.fused_recurrent_gated_delta_rule is not None,
        "causal_conv1d_fn": mod.causal_conv1d_fn is not None,
        "causal_conv1d_update": mod.causal_conv1d_update is not None,
    }
    mod.FusedRMSNormGated = None
    mod.chunk_gated_delta_rule = None
    mod.fused_recurrent_gated_delta_rule = None
    mod.causal_conv1d_fn = None
    mod.causal_conv1d_update = None
    mod.is_fast_path_available = False
    return was


def loader_class(config: Any):
    """The class that owns a checkpoint with this config.

    ``Qwen3_5ForConditionalGeneration`` keys are ``model.language_model.layers.*``
    and ``Qwen3_5ForCausalLM``'s are ``model.layers.*``; loading a checkpoint with
    the wrong one silently re-initialises every weight it cannot name.  The
    config's own ``model_type`` decides, because that is what the checkpoint
    declares about itself.
    """

    from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_5":
        return Qwen3_5ForConditionalGeneration
    if model_type == "qwen3_5_text":
        return Qwen3_5ForCausalLM
    raise SystemExit(f"not a qwen3_5 config: model_type={model_type!r}")


def load_config(model_dir: str | Path):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(model_dir))


# ---------------------------------------------------------------- navigation
def text_model(model: nn.Module) -> nn.Module:
    """The ``Qwen3_5TextModel`` under either top-level class."""

    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


def text_config(config: Any):
    """The ``Qwen3_5TextConfig``, whether it is the config or a sub-config."""

    return getattr(config, "text_config", None) or config


def vision_model(model: nn.Module) -> nn.Module | None:
    inner = getattr(model, "model", model)
    return getattr(inner, "visual", None)


def layers(model: nn.Module):
    return text_model(model).layers


def final_norm(model: nn.Module) -> nn.Module:
    return text_model(model).norm


def embed(model: nn.Module) -> nn.Module:
    return text_model(model).embed_tokens


def lm_head(model: nn.Module) -> nn.Module:
    return model.lm_head


def rotary(model: nn.Module) -> nn.Module:
    return text_model(model).rotary_emb


def rms_eps(config: Any) -> float:
    return float(text_config(config).rms_norm_eps)


def layer_kind(layer: nn.Module) -> str:
    """The kind of the LIVE module, never the config's opinion of it.

    ``layer.layer_type`` is set at construction from the config, so on a model
    whose layers have been deleted it is still the module's OWN truth (a full
    layer that moved to position 7 still says ``full_attention``); the config
    list is the thing that goes stale.  The presence check is the tiebreak for a
    layer built by hand.
    """

    if hasattr(layer, "linear_attn"):
        return LINEAR
    if hasattr(layer, "self_attn"):
        return FULL
    raise SystemExit(f"{type(layer).__name__} carries neither linear_attn nor self_attn")


def mixer(layer: nn.Module) -> nn.Module:
    return layer.linear_attn if layer_kind(layer) == LINEAR else layer.self_attn


def mixer_norm(layer: nn.Module) -> nn.Module:
    """The norm whose output the mixer reads (``input_layernorm``)."""

    return layer.input_layernorm


def ffn_norm(layer: nn.Module) -> nn.Module:
    return layer.post_attention_layernorm


def residual_endpoints(model: nn.Module) -> dict[str, nn.Module]:
    """Every norm that reads the residual stream: ``2L + 1`` of them."""

    out: dict[str, nn.Module] = {}
    for index, layer in enumerate(layers(model)):
        out[f"L{index:03d}.operator"] = layer.input_layernorm
        out[f"L{index:03d}.ffn"] = layer.post_attention_layernorm
    out["final"] = final_norm(model)
    return out


# --------------------------------------------------------------------- gains
def is_gated_norm(norm: nn.Module) -> bool:
    """``Qwen3_5RMSNormGated`` (or fla's fused twin) vs ``Qwen3_5RMSNorm``.

    Distinguished by the attribute the two classes do not share:
    ``variance_epsilon`` on the gated one (modeling line 191) against ``eps`` on
    the plain one (739).  A class-name test would break the moment fla's
    ``FusedRMSNormGated`` is the live class, which is the default on this venv.
    """

    return hasattr(norm, "variance_epsilon") or "Gated" in type(norm).__name__


def effective_gain(norm: nn.Module) -> torch.Tensor:
    """The number the forward pass actually multiplies by."""

    weight = norm.weight.data
    return weight if is_gated_norm(norm) else 1.0 + weight


def store_gain(norm: nn.Module, gain: torch.Tensor) -> None:
    """Write an EFFECTIVE gain back into the parameter's own convention."""

    value = gain.to(device=norm.weight.device, dtype=norm.weight.dtype)
    norm.weight.data.copy_(value if is_gated_norm(norm) else value - 1.0)


def transported_gain(gain: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """``diag(E^T diag(g) E)`` -- the prior for a gain in a folded width."""

    return ((basis * gain.unsqueeze(1)) * basis).sum(dim=0)


# ---------------------------------------------------------- rope plane layout
def plane_layout(head_dim: int, rotary_dim: int) -> dict[str, torch.Tensor]:
    """Which head coordinates rotate and which pass through.

    ``apply_rotary_pos_emb`` (modeling 570-605) splits each head at
    ``rotary_dim``: the first ``rotary_dim`` coordinates get
    ``rotate_half`` -- which pairs coordinate ``p`` with ``p + rotary_dim//2``,
    NOT ``p`` with ``p+1`` -- and coordinates ``[rotary_dim:]`` are concatenated
    back untouched.  So a Qwen head is ``rotary_dim/2`` complex planes plus
    ``head_dim - rotary_dim`` REAL coordinates, and the NoPE coordinates are
    where C6 lives.
    """

    if rotary_dim % 2 or rotary_dim > head_dim:
        raise SystemExit(f"rotary_dim {rotary_dim} is not an even part of head_dim {head_dim}")
    half = rotary_dim // 2
    return {
        "lo": torch.arange(half, dtype=torch.long),
        "hi": torch.arange(half, dtype=torch.long) + half,
        "nope": torch.arange(rotary_dim, head_dim, dtype=torch.long),
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
    }


def rotary_dim_of(config: Any) -> int:
    """``head_dim * partial_rotary_factor``, ROUNDED.

    Truncating, exactly as ``compute_default_rope_parameters`` does (``dim =
    int(head_dim * partial_rotary_factor)``, modeling 137), so this reports what
    the model WILL do rather than what the ratio means.  The difference is not
    academic: a C6 child at ``head_dim`` 14 with factor ``8/14`` renders as
    0.5714285714285714 in fp64, and ``int(14 * that)`` is **7** -- which would
    build ``inv_freq`` on a 7-wide ladder, i.e. at frequencies the parent never
    used.  :func:`exact_partial_rotary_factor` is what a fold must write.
    """

    tc = text_config(config)
    params = dict(tc.rope_parameters or {})
    factor = float(params.get("partial_rotary_factor", 1.0))
    return int(int(tc.head_dim) * factor)


def exact_partial_rotary_factor(rotary_dim: int, head_dim: int) -> float:
    """The smallest float ``f`` with ``int(head_dim * f) == rotary_dim``.

    Written rather than assumed because the ratio's nearest double can round
    either way and only one of the two directions preserves the parent's rotary
    frequencies.
    """

    import math

    factor = rotary_dim / head_dim
    for _ in range(64):
        produced = int(head_dim * factor)
        if produced == rotary_dim:
            return factor
        factor = math.nextafter(factor, 1.0 if produced < rotary_dim else 0.0)
    raise SystemExit(
        f"no representable partial_rotary_factor gives rotary_dim {rotary_dim} at "
        f"head_dim {head_dim}"
    )


# ------------------------------------------------------------ parameter count
def text_parameter_count(tc: Any) -> dict[str, int]:
    """Closed-form parameter count of a ``Qwen3_5TextModel`` + its head.

    Replaces LFM2's ``predicted_parameters`` polynomial, which was fitted to one
    model's shape.  This one is read off the module constructors, term by term,
    so every axis in this campaign can price itself without instantiating
    anything -- and :func:`assert_parameter_count` refuses any build whose live
    model disagrees with it.
    """

    H = int(tc.hidden_size)
    V = int(tc.vocab_size)
    F = int(tc.intermediate_size)
    L = int(tc.num_hidden_layers)
    types = list(tc.layer_types)
    if len(types) != L:
        raise SystemExit(f"layer_types has {len(types)} entries for {L} layers")

    head_dim = int(tc.head_dim)
    nh = int(tc.num_attention_heads)
    nkv = int(tc.num_key_value_heads)

    dk = int(tc.linear_key_head_dim)
    dv = int(tc.linear_value_head_dim)
    nk = int(tc.linear_num_key_heads)
    nv = int(tc.linear_num_value_heads)
    kernel = int(tc.linear_conv_kernel_dim)
    Kd, Vd = dk * nk, dv * nv
    conv_dim = 2 * Kd + Vd

    mlp = 3 * H * F
    gdn = (
        conv_dim * H          # in_proj_qkv
        + Vd * H              # in_proj_z
        + 2 * nv * H          # in_proj_b, in_proj_a
        + conv_dim * kernel   # depthwise conv1d, no bias
        + 2 * nv              # dt_bias, A_log
        + dv                  # RMSNormGated weight, shared over value heads
        + H * Vd              # out_proj
    )
    attn = (
        nh * head_dim * 2 * H     # q_proj: query rows AND gate rows
        + 2 * nkv * head_dim * H  # k_proj, v_proj
        + H * nh * head_dim       # o_proj
        + 2 * head_dim            # q_norm, k_norm (one weight shared by all heads)
    )

    n_linear = sum(1 for t in types if t == LINEAR)
    n_full = L - n_linear
    body = n_linear * (gdn + 2 * H + mlp) + n_full * (attn + 2 * H + mlp)
    interface = V * H + H + (0 if bool(tc.tie_word_embeddings) else V * H)
    return {
        "total": body + interface,
        "body": body,
        "interface": interface,
        "per_linear_layer": gdn + 2 * H + mlp,
        "per_full_layer": attn + 2 * H + mlp,
        "mlp_per_layer": mlp,
        "gdn_mixer": gdn,
        "attention_mixer": attn,
        "linear_layers": n_linear,
        "full_layers": n_full,
    }


def vision_parameter_count(vc: Any) -> int:
    """Closed form for ``Qwen3_5VisionModel`` (patch embed, pos embed, blocks, merger)."""

    H = int(vc.hidden_size)
    I = int(vc.intermediate_size)
    depth = int(vc.depth)
    patch = int(vc.patch_size)
    temporal = int(vc.temporal_patch_size)
    channels = int(vc.in_channels)
    merge = int(vc.spatial_merge_size)
    out_hidden = int(vc.out_hidden_size)

    patch_embed = H * channels * temporal * patch * patch + H
    pos_embed = int(vc.num_position_embeddings) * H
    block = (
        3 * H * H + 3 * H     # attn.qkv + bias
        + H * H + H           # attn.proj + bias
        + I * H + I           # mlp.linear_fc1 + bias
        + H * I + H           # mlp.linear_fc2 + bias
        + 4 * H               # norm1, norm2 (LayerNorm: weight + bias)
    )
    merged = H * merge * merge
    merger = 2 * H + merged * merged + merged + out_hidden * merged + out_hidden
    return patch_embed + pos_embed + depth * block + merger


def parameter_count(config: Any) -> dict[str, int]:
    tc = text_config(config)
    report = text_parameter_count(tc)
    vc = getattr(config, "vision_config", None)
    report["vision"] = vision_parameter_count(vc) if vc is not None else 0
    report["total_with_vision"] = report["total"] + report["vision"]
    return report


def live_parameter_count(model: nn.Module) -> int:
    """Distinct parameter tensors, counted once even when the head is tied."""

    seen: dict[int, int] = {}
    for parameter in model.parameters():
        seen[id(parameter)] = parameter.numel()
    return int(sum(seen.values()))


def assert_parameter_count(model: nn.Module, config: Any | None = None) -> dict[str, int]:
    """Refuse a model whose live count disagrees with the closed form.

    This is the receipt's denominator.  A wrong denominator turns every ratio in
    the campaign into a wrong claim, and the two numbers are computed by
    completely different routes, so agreement is evidence.
    """

    config = config if config is not None else model.config
    predicted = parameter_count(config)
    live = live_parameter_count(model)
    expected = predicted["total_with_vision"] if vision_model(model) is not None else predicted["total"]
    if live != expected:
        raise SystemExit(
            f"parameter count disagrees: closed form {expected:,}, live {live:,} "
            f"(delta {live - expected:+,})"
        )
    predicted["live"] = live
    return predicted


# ------------------------------------------------------------- config surgery
def config_field(tc: Any, field: str, value: Any) -> None:
    """Write one field onto a ``@strict`` huggingface dataclass config.

    ``Qwen3_5TextConfig`` is decorated ``@strict`` (configuration file line 28),
    which validates on assignment; a plain ``setattr`` of a wrong-typed value
    raises here rather than three hours later inside a kernel.  Declared
    fields only -- an undeclared name is a typo, and a typo that silently sticks
    to a config is how a child gets exported with a shape nothing reads.
    """

    if not hasattr(tc, field):
        raise SystemExit(f"{type(tc).__name__} has no field {field!r}")
    setattr(tc, field, value)


def config_diff(parent: Any, child: Any) -> list[dict[str, Any]]:
    """Every scalar/list field that differs, flattened over the sub-configs."""

    out: list[dict[str, Any]] = []

    def walk(prefix: str, a: Any, b: Any) -> None:
        a_body = a.to_dict() if hasattr(a, "to_dict") else dict(a)
        b_body = b.to_dict() if hasattr(b, "to_dict") else dict(b)
        for key in sorted(set(a_body) | set(b_body)):
            left, right = a_body.get(key), b_body.get(key)
            if isinstance(left, dict) and isinstance(right, dict):
                walk(f"{prefix}{key}.", left, right)
                continue
            if left != right:
                out.append({"field": f"{prefix}{key}", "parent": left, "child": right})

    walk("", parent, child)
    # `_name_or_path` and `transformers_version` differ on every child ever
    # written and describe nothing about its shape, so they are dropped: a diff
    # that is EMPTY after this filter is the honest statement that the child's
    # config is the parent's, which is true of a values-only axis (C9) and is
    # what the receipt should say.  Nothing in this lane refuses on that -- an
    # earlier version of this comment claimed a sealer refusal that does not
    # exist, and a receipt that describes a check nobody runs is worse than no
    # check at all.
    noise = ("transformers_version", "_name_or_path")
    return [row for row in out if not row["field"].split(".")[-1] in noise]


def derive_layer_types(tc: Any, surviving: list[str]) -> None:
    """Rewrite ``layer_types`` and ``num_hidden_layers`` together.

    They are two spellings of one fact and the forward pass reads BOTH
    (``self.layers[: self.config.num_hidden_layers]`` and ``self.config
    .layer_types[i]``, modeling 1200-1201).  Updating one is the excision bug.
    """

    config_field(tc, "layer_types", list(surviving))
    config_field(tc, "num_hidden_layers", len(surviving))


def assert_layer_types_agree(model: nn.Module) -> None:
    """The live layer at position i must be the kind the config says it is.

    Not a style check.  Line 1201 hands a layer the mask chosen by
    ``config.layer_types[i]``; a full-attention module sitting in a slot the
    config calls linear gets ``linear_attn_mask``, which is ``None`` at batch 1
    with no padding -- and an unmasked softmax attends to the future.  The other
    direction is invisible (``apply_mask_to_padding_states`` ignores a 4D mask at
    batch 1, modeling 210), so this assertion is the only cheap check that sees
    both.
    """

    tc = text_config(model.config)
    live = [layer_kind(layer) for layer in layers(model)]
    declared = list(tc.layer_types)[: int(tc.num_hidden_layers)]
    if len(live) != int(tc.num_hidden_layers):
        raise SystemExit(
            f"model carries {len(live)} layers, config.num_hidden_layers is "
            f"{tc.num_hidden_layers}"
        )
    for position, (a, b) in enumerate(zip(live, declared)):
        if a != b:
            raise SystemExit(
                f"layer {position}: live module is {a}, config.layer_types says {b} -- "
                f"the mask dispatch at modeling_qwen3_5.py:1201 would hand it the "
                f"wrong mask"
            )


def reindex_layers(model: nn.Module) -> None:
    """Make every mixer's ``layer_idx`` its position again.

    ``layer_idx`` is the cache key (GDN conv/recurrent states, KV pages); after a
    deletion two layers can claim one slot, and the model will run and be wrong.
    """

    for position, layer in enumerate(layers(model)):
        mixer(layer).layer_idx = position


# --------------------------------------------------------------- tensor names
def tensor_prefix(model_or_config: Any) -> str:
    """``model.language_model.`` or ``model.`` -- how this class names the text body."""

    config = getattr(model_or_config, "config", model_or_config)
    return "model.language_model." if getattr(config, "model_type", "") == "qwen3_5" else "model."


def carried_prefixes() -> tuple[str, ...]:
    """Checkpoint tensors transformers never owns and export must carry by hand.

    ``_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]`` on
    ``Qwen3_5ForCausalLM`` (modeling 1618) -- and ``^mtp.*`` on every class -- so
    the MTP block is read by nobody and written by nobody.  It is real
    parameters in the parent's index and it is a real part of the child unless a
    receipt says it was dropped.
    """

    return ("mtp.",)


def canonical_json_bytes(value: Any) -> bytes:
    """The estate's canonical encoding (dynamic_circuit_ir.py:82-96), unchanged."""

    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tiny_text_config(
    *,
    hidden: int = 128,
    layers_count: int = 8,
    vocab: int = 512,
    intermediate: int = 256,
    head_dim: int = 32,
    heads: int = 4,
    kv_heads: int = 2,
    linear_key_heads: int = 2,
    linear_value_heads: int = 4,
    linear_key_head_dim: int = 16,
    linear_value_head_dim: int = 16,
    full_attention_interval: int = 4,
):
    """A random-init ``Qwen3_5TextConfig`` with every axis this campaign folds.

    Deliberately NOT a scaled 4B: the tiny model exists to make each fold class's
    mechanics fail loudly on a laptop-sized forward pass, so it keeps the shapes
    that the mechanics depend on -- more value heads than key heads (C5 has
    something to merge), ``head_dim`` strictly larger than the rotary width (C6
    has NoPE coordinates to contract), more query heads than kv heads (C7 has a
    group), and both layer kinds in the ``layer_types`` list (C3/C4 and the mask
    trap are reachable).
    """

    from transformers import Qwen3_5TextConfig

    rotary_dim = head_dim // 4
    planes = rotary_dim // 2
    section = [planes - 2 * (planes // 3), planes // 3, planes // 3]
    # layer_types is written EXPLICITLY rather than derived from
    # full_attention_interval: the derivation runs inside a @strict dataclass's
    # __post_init__ on an undeclared kwarg, and a fixture that depends on that
    # is testing transformers, not this lane.
    types = [
        LINEAR if bool((i + 1) % full_attention_interval) else FULL
        for i in range(layers_count)
    ]
    return Qwen3_5TextConfig(
        layer_types=types,
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers_count,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_key_heads,
        linear_num_value_heads=linear_value_heads,
        linear_conv_kernel_dim=4,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": section,
            "mrope_interleaved": True,
        },
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=None,
    )


def seeded_tiny_model(config: Any, seed: int = 20260901):
    """A tiny model whose weights are reproducible and whose norms are NOT zero.

    ``Qwen3_5RMSNorm`` initialises to zeros, i.e. effective gain exactly 1.  A
    fold that mishandles the ``1 + w`` convention is invisible on such a model --
    it would be tested against the one weight value where both conventions
    agree.  So every norm gets a small nonzero perturbation here, which is the
    whole point of using a random-init model as the test bed.
    """

    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") or ".norm." in name or "layernorm" in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)
            elif parameter.dim() >= 2:
                parameter.mul_(1.0)
        for layer in model.model.layers:
            if layer_kind(layer) == LINEAR:
                # A decay range that neither saturates (state never updates) nor
                # vanishes (state never decays), so a GDN fold is measured on a
                # layer that is actually doing something.
                layer.linear_attn.A_log.copy_(
                    torch.log(torch.empty_like(layer.linear_attn.A_log).uniform_(0.5, 4.0))
                )
                layer.linear_attn.dt_bias.copy_(torch.randn_like(layer.linear_attn.dt_bias) * 0.1)
    return model.eval()


__all__ = [name for name in dir() if not name.startswith("_")]
