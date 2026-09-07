#!/usr/bin/env python3
"""c16_media_fixtures.py -- synthetic S6A/S6B inputs for `c16_install_media.py`.

Generates, at a tiny geometry, one instance of every file format in
`composite/COMPOSITE_S6_INPUT_FORMATS.md`:

    child/                      a Chimera checkpoint (the stand-in for the S5 union child)
    c15/                        an interface row set: companions.json + composed_interface files
    c11/                        a second interface row set (the QVL32 patch/merger rows)
    c18/ace_lm_music_codes.safetensors   the music_codes band table (+ .json sidecar)
    blocks/<name>/              charted block containers (with and without charts; every adaLN
                                container also carries time_reference.safetensors, the modulation
                                the installer must reproduce before it inserts the block)
    tap/<name>/paired_rows.safetensors   paired rows for the fit_charts path

Nothing here is a fold: the tensors are seeded random numbers with the RIGHT SHAPES and
the right file layout, so the installer's accounting, routing and refusals can be proved
before the real S6A/S6B artifacts exist. `test_c16_install_media.py` is the consumer.

    PYTHONPATH=<chimera tree>/src python c16_media_fixtures.py --out DIR [--hidden 32 ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_HIDDEN = 32
DEFAULT_FEATURE = 48
DEFAULT_BASE_VOCAB = 256
DEFAULT_MUSIC_BAND = 24
INTERFACE_SCHEMA = "chimera_omni.composite.interface_set.v1"
BLOCK_SCHEMA = "chimera_omni.composite.charted_block.v1"
TIME_REFERENCE_FILENAME = "time_reference.safetensors"
SMALL_BANDS = (("control", 8), ("media", 16), ("grounding", 8), ("sync", 8), ("tool", 8))


def _torch():
    import torch

    return torch


def save_single(state: dict[str, Any], path: Path) -> dict[str, str]:
    """Write one plain safetensors file and return {name: sha256}.

    The digest is `safetensors_io.TensorView.sha256()` read back from the file (dtype name
    + shape + bytes), i.e. exactly what `c16_install_media.py` recomputes when it verifies a
    declared `tensor_sha256`; a second formula here would only ever agree by accident.
    """

    from chimera_omni.importing.safetensors_io import read_safetensors
    from safetensors.torch import save_file

    payload = {name: value.detach().cpu().contiguous() for name, value in state.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(path))
    return {name: view.sha256() for name, view in sorted(read_safetensors(path, verify=False).items())}


def write_json(path: Path, body: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# the child
# --------------------------------------------------------------------------


def child_config(
    *,
    hidden: int = DEFAULT_HIDDEN,
    feature: int = DEFAULT_FEATURE,
    base_vocab: int = DEFAULT_BASE_VOCAB,
    music_band: int = DEFAULT_MUSIC_BAND,
    num_layers: int = 8,
    residual_mode: str = "block_attnres",
    intermediate: int = 48,
    tie_word_embeddings: bool = False,
    **overrides: Any,
):
    """A 27B-shaped Chimera config (kda,kda,kda,gated_gqa; fixed bands; music band) at toy size."""

    from chimera_omni.config import ChimeraConfig

    bands = SMALL_BANDS + (("music_codes", int(music_band)),)
    settings: dict[str, Any] = {
        "vocab_size": int(base_vocab) + sum(width for _, width in bands),
        "hidden_size": int(hidden),
        "intermediate_size": int(intermediate),
        "num_layers": int(num_layers),
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_sequence_length": 256,
        "block_pattern": ("kda", "kda", "kda", "gated_gqa"),
        "residual_mode": residual_mode,
        "attnres_block_size": 4,
        "attention_head_dim": 8,
        "kda_num_heads": 4,
        "kda_num_key_heads": 2,
        "kda_head_dim": 8,
        "kda_value_head_dim": 8,
        "kda_decay_kind": "gdn",
        "kda_output_gate": "silu",
        "kda_qk_norm_kind": "fla",
        "kda_query_scale": "inv_sqrt_key_dim",
        "kda_conv_bias": False,
        "gqa_rope_dim": 2,
        "native_feature_dim": int(feature),
        "ledger_kind": "fixed_band",
        "ledger_base_vocab_size": int(base_vocab),
        "ledger_bands": bands,
        "tie_word_embeddings": bool(tie_word_embeddings),
        "metadata": {"fixture": "c16_media_fixtures", "profile": "toy_27b_shaped"},
    }
    settings.update(overrides)
    cfg = ChimeraConfig(**settings)
    cfg.validate()
    return cfg


def write_child(out: Path, *, seed: int = 0, dtype: str = "bfloat16", shard_bytes: int | None = None, **config_kwargs: Any) -> dict[str, Any]:
    """Write a synthetic child with ZERO music-band rows (as S0 declares them)."""

    from chimera_omni.importing.safetensors_io import state_sha256, write_safetensors
    from chimera_omni.modeling import ChimeraOmniModel

    torch = _torch()
    cfg = child_config(**config_kwargs)
    torch.manual_seed(int(seed))
    model = ChimeraOmniModel(cfg).to(dtype=getattr(torch, dtype))
    band = cfg.ledger().range("music_codes")
    with torch.no_grad():
        model.embed.weight[band.begin : band.end].zero_()
        if not cfg.tie_word_embeddings:
            model.lm_head.weight[band.begin : band.end].zero_()
    kwargs: dict[str, Any] = {}
    if shard_bytes is not None:
        kwargs["shard_bytes"] = int(shard_bytes)
    index = write_safetensors(
        model.state_dict(),
        out,
        config=cfg.to_manifest(),
        ledger=cfg.ledger().to_manifest(),
        source={"fixture": "c16_media_fixtures.write_child", "seed": int(seed)},
        chimera_commit="f" * 40,
        **kwargs,
    )
    return {
        "directory": str(out),
        "state_sha256": state_sha256(model),
        "parameters": sum(int(p.numel()) for p in model.parameters()),
        "tensors": len(index["weight_map"]),
        "config": cfg.to_manifest(),
    }


# --------------------------------------------------------------------------
# interface row sets (C15 / C11)
# --------------------------------------------------------------------------


def write_interface_set(
    out: Path,
    entries: list[dict[str, Any]],
    *,
    hidden: int,
    feature: int,
    seed: int = 0,
    schema: str = INTERFACE_SCHEMA,
) -> dict[str, Any]:
    """`entries`: [{name, source, type_id, feature_band:[b,e], emission_band?:[b,e], rung?}]."""

    torch = _torch()
    generator = torch.Generator().manual_seed(int(seed))
    manifest_entries: list[dict[str, Any]] = []
    for entry in entries:
        b, e = (int(v) for v in entry["feature_band"])
        eb, ee = (int(v) for v in entry.get("emission_band", entry["feature_band"]))
        state = {
            "bridge.feature_proj.weight": torch.randn(hidden, e - b, generator=generator) * 0.05,
            "bridge.type_embed_delta": torch.randn(hidden, generator=generator) * 0.05,
            "head.out.weight": torch.randn(ee - eb, hidden, generator=generator) * 0.05,
            "head.type_affine_shift": torch.randn(ee - eb, generator=generator) * 0.05,
        }
        rows = Path(entry["name"]) / entry.get("rung", "linear") / "composed_interface.safetensors"
        table = save_single(state, out / rows)
        manifest_entries.append(
            {
                "name": str(entry["name"]),
                "source": str(entry.get("source", entry["name"])),
                "type_id": int(entry["type_id"]),
                "feature_band": [b, e],
                "emission_band": [eb, ee],
                "rows": rows.as_posix(),
                "rung": str(entry.get("rung", "linear")),
                "companion_decoder": bool(entry.get("companion_decoder", True)),
                "psnr_absorbed_db": entry.get("psnr_absorbed_db"),
                "psnr_parent_roundtrip_db": entry.get("psnr_parent_roundtrip_db"),
                "tensor_sha256": table,
            }
        )
    body = {"schema": schema, "hidden": int(hidden), "native_feature_dim": int(feature), "entries": manifest_entries}
    write_json(out / "companions.json", body)
    return body


# --------------------------------------------------------------------------
# music-code band table (C18)
# --------------------------------------------------------------------------


def write_band_table(path: Path, *, width: int, hidden: int, seed: int = 0, tied: bool = False) -> dict[str, Any]:
    torch = _torch()
    generator = torch.Generator().manual_seed(int(seed))
    embed = torch.randn(int(width), int(hidden), generator=generator) * 0.02
    head = embed.clone() if tied else torch.randn(int(width), int(hidden), generator=generator) * 0.02
    table = save_single({"embed.weight": embed, "lm_head.weight": head}, path)
    sidecar = {
        "kind": "c18_ace_lm_music_codes",
        "fixture": True,
        "band_width": int(width),
        "hidden": int(hidden),
        "aligned_rows": int(width),
        "unmapped_rows": 0,
        "tensor_sha256": table,
    }
    write_json(Path(str(path) + ".json"), sidecar)
    return sidecar


# --------------------------------------------------------------------------
# charted block containers (C16)
# --------------------------------------------------------------------------


def block_spec_fields(name: str, **overrides: Any) -> dict[str, Any]:
    presets: dict[str, dict[str, Any]] = {
        "ace_L12": {
            "source": "ace", "component": "ace.dit.block12.mlp", "type_ids": [4], "width": 24, "heads": 2,
            "kv_heads": 1, "head_dim": 8, "mlp_width": 20, "mlp_kind": "swiglu", "modulation": "adaln6",
            "norm_kind": "rms", "norm_affine": True, "norm_eps": 1.0e-6, "qk_norm": True, "attn_bias": False,
            "rope_theta": 1.0e6, "sliding_window": 3, "time_freq_dim": 8, "time_hidden": 24,
            "time_factor": 1000.0, "time_embed_r": True, "delta_scale": 1.0, "chart_in_bias": True,
            "rung": "latents", "source_depth_fraction": 12.0 / 23.0,
            "metadata": {"fixture": "ace-shaped", "timestep_r": "0.0 as captured"},
        },
        "h3_Lmid": {
            # MiniMax H3 audio-video DiT (mediafold.c16_h3): RMSNorm affine, MHA with q/k norm and
            # no bias, fused-fc1 SwiGLU split into gate/up at capture, 6-chunk adaLN. Shape preset:
            # the parent's 3-D RoPE and its adaln_proj(t_emb[8]) time path are the capture
            # script's to fold; the time reference gate refuses a container whose path does not
            # reproduce the captured modulation.
            "source": "h3", "component": "h3.dit.block.mlp", "type_ids": [2, 3], "width": 16, "heads": 2,
            "kv_heads": 2, "head_dim": 8, "mlp_width": 12, "mlp_kind": "swiglu", "modulation": "adaln6",
            "norm_kind": "rms", "norm_affine": True, "norm_eps": 1.0e-6, "qk_norm": True, "attn_bias": False,
            "rope_theta": 0.0, "sliding_window": 0, "time_freq_dim": 8, "time_hidden": 16,
            "time_factor": 1000.0, "delta_scale": 1.0, "chart_in_bias": True, "rung": "latents",
            "source_depth_fraction": 0.5, "metadata": {"fixture": "h3-dit-shaped", "rope": "3-D rope dropped at capture (shape preset)"},
        },
        "hy3d_Lmid": {
            # Hunyuan3D 2.1 shape DiT block (flux-style stream, image half): non-affine LayerNorm,
            # adaLN modulation permuted into the ACE chunk order at capture, q/k RMSNorm, biased
            # attention projections, GELU-tanh MLP, no positional embedding (latent token set).
            "source": "hy3d", "component": "hy3d.dit.block.mlp", "type_ids": [7], "width": 16, "heads": 2,
            "kv_heads": 2, "head_dim": 8, "mlp_width": 12, "mlp_kind": "gelu_tanh", "modulation": "adaln6",
            "norm_kind": "layer", "norm_affine": False, "norm_eps": 1.0e-6, "qk_norm": True, "attn_bias": True,
            "rope_theta": 0.0, "sliding_window": 0, "time_freq_dim": 8, "time_hidden": 16,
            "time_factor": 1000.0, "delta_scale": 1.0, "chart_in_bias": True, "rung": "latents",
            "source_depth_fraction": 0.5, "metadata": {"fixture": "hy3d-dit-shaped"},
        },
        "qvl32_vit_Lmid": {
            "source": "qvl32", "component": "qvl32.vit.block.mlp", "type_ids": [1, 6], "width": 16, "heads": 2,
            "kv_heads": 2, "head_dim": 8, "mlp_width": 12, "mlp_kind": "gelu", "modulation": "none",
            "norm_kind": "layer", "norm_affine": True, "norm_eps": 1.0e-6, "qk_norm": False, "attn_bias": True,
            "rope_theta": 0.0, "sliding_window": 0, "time_freq_dim": 8, "time_hidden": 0,
            "time_factor": 1000.0, "delta_scale": 1.0, "chart_in_bias": True, "rung": "latents",
            "source_depth_fraction": 0.75, "metadata": {"fixture": "vit-shaped", "rope": "2d-dropped"},
        },
        "qimg_Lmid": {
            "source": "qimg", "component": "qimg.mmdit.block.mlp", "type_ids": [1], "width": 16, "heads": 2,
            "kv_heads": 2, "head_dim": 8, "mlp_width": 12, "mlp_kind": "gelu_tanh", "modulation": "adaln6",
            "norm_kind": "layer", "norm_affine": False, "norm_eps": 1.0e-6, "qk_norm": True, "attn_bias": True,
            "rope_theta": 0.0, "sliding_window": 0, "time_freq_dim": 8, "time_hidden": 16,
            "time_factor": 1000.0, "delta_scale": 1.0, "chart_in_bias": True, "rung": "latents",
            "source_depth_fraction": 0.25, "metadata": {"fixture": "mmdit-shaped"},
        },
    }
    fields = dict(presets[name])
    fields.update(overrides)
    return fields


def write_block_container(
    out: Path,
    name: str,
    *,
    hidden: int,
    seed: int = 0,
    with_charts: bool = True,
    insert_after: int | None = None,
    reference_sigmas: tuple[float, ...] = (0.25, 0.5, 0.75),
    **spec_overrides: Any,
) -> dict[str, Any]:
    """A charted block container with seeded random tensors of the exact canonical shapes."""

    from chimera_omni.charted import ChartedBlockSpec

    torch = _torch()
    fields = block_spec_fields(name, **spec_overrides)
    spec = ChartedBlockSpec(layer_index=0, **{**fields, "type_ids": tuple(fields["type_ids"])})
    spec.validate()
    generator = torch.Generator().manual_seed(int(seed))
    shapes = spec.parameter_shapes("", int(hidden))
    state: dict[str, Any] = {}
    charts: dict[str, Any] = {}
    for tensor_name, shape in shapes.items():
        if tensor_name.startswith("chart_"):
            target = charts
            scale = 0.2
        else:
            target = state
            scale = 0.3
        if tensor_name.endswith("norm.weight") or tensor_name.endswith("_norm.weight"):
            value = torch.ones(shape)
        elif tensor_name == "scale_shift_table":
            value = torch.randn(shape, generator=generator) * 0.1
        else:
            value = torch.randn(shape, generator=generator) * scale / (shape[-1] ** 0.5)
        target[tensor_name] = value
    target_dir = out / name
    block_table = save_single(state, target_dir / "charted_block.safetensors")
    chart_table = save_single(charts, target_dir / "charts.safetensors") if with_charts else None
    reference_table = None
    reference_body = None
    if spec.modulation == "adaln6":
        # the "captured" modulation: what the parent's time path emits at these noise levels.
        # A real container takes it from the capture (capture_L*.safetensors:temb); the
        # fixture takes it from the spec's own path so the installer's gate is exercised.
        from chimera_omni.modeling.charted import ChartedSourceBlock

        block = ChartedSourceBlock(spec, int(hidden)).eval()
        block.load_state_dict({**state, **charts}, strict=True)
        sigma = torch.tensor(list(reference_sigmas), dtype=torch.float32)
        with torch.no_grad():
            modulation = block.time_path(sigma).to(torch.float32)
        reference_table = save_single({"sigma": sigma, "modulation": modulation}, target_dir / TIME_REFERENCE_FILENAME)
        reference_body = {"file": TIME_REFERENCE_FILENAME, "source": "fixture: the spec's own time path", "sigma": [float(v) for v in sigma], "timestep_r": 0.0}
    parent_mlp = spec.mlp_parameter_count() * 4
    body = {
        "schema": BLOCK_SCHEMA,
        "spec": {key: value for key, value in fields.items()},
        "provenance": {
            "source_weights": f"fixture://{name}",
            "source_prefix": f"fixture.layers.{name}.",
            "capture_receipt": None,
            "capture_receipt_sha256": None,
            "fixture": True,
        },
        "accounting": {
            "parent_mlp_parameters": int(parent_mlp),
            "child_mlp_parameters": int(spec.mlp_parameter_count()),
            "parent_block_parameters": int(spec.parameter_count(hidden) - sum(v.numel() for v in charts.values()) + 3 * spec.mlp_parameter_count()),
            "child_block_parameters": int(spec.parameter_count(hidden) - sum(v.numel() for v in charts.values())),
        },
        "holdout": {"rel_l2": None, "cos": None, "split_by": "fixture"},
        "time_reference": reference_body,
        "insert_after": insert_after,
        "tensor_sha256": {
            key: value
            for key, value in (
                ("charted_block.safetensors", block_table),
                ("charts.safetensors", chart_table),
                (TIME_REFERENCE_FILENAME, reference_table),
            )
            if value is not None
        },
    }
    write_json(target_dir / "charted_block.json", body)
    return body


def write_tap_rows(out: Path, name: str, *, hidden: int, width: int, rows: int = 96, seed: int = 0) -> dict[str, str]:
    """Paired host/parent rows for the fit_charts path: parent = host @ W + noise."""

    torch = _torch()
    generator = torch.Generator().manual_seed(int(seed))
    host = torch.randn(int(rows), int(hidden), generator=generator)
    mixing = torch.randn(int(hidden), int(width), generator=generator) / (hidden ** 0.5)
    parent = host @ mixing + 0.01 * torch.randn(int(rows), int(width), generator=generator)
    return save_single({"host_rows": host, "parent_rows": parent}, out / name / "paired_rows.safetensors")


# --------------------------------------------------------------------------
# the whole set
# --------------------------------------------------------------------------


def write_fixture_set(
    out: Path,
    *,
    hidden: int = DEFAULT_HIDDEN,
    feature: int = DEFAULT_FEATURE,
    music_band: int = DEFAULT_MUSIC_BAND,
    seed: int = 0,
    residual_mode: str = "block_attnres",
    tie_word_embeddings: bool = False,
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"out": str(out), "hidden": hidden, "native_feature_dim": feature}
    report["child"] = write_child(
        out / "child", seed=seed, hidden=hidden, feature=feature, music_band=music_band,
        residual_mode=residual_mode, tie_word_embeddings=tie_word_embeddings,
    )
    third = feature // 3
    report["c15"] = write_interface_set(
        out / "c15",
        [
            {"name": "music_ace", "source": "ACE", "type_id": 4, "feature_band": [0, third],
             "psnr_absorbed_db": 23.09, "psnr_parent_roundtrip_db": 32.78},
            {"name": "audio_h3", "source": "H3", "type_id": 3, "feature_band": [third, 2 * third],
             "psnr_absorbed_db": 24.13, "psnr_parent_roundtrip_db": 36.34},
        ],
        hidden=hidden,
        feature=feature,
        seed=seed + 1,
    )
    report["c11"] = write_interface_set(
        out / "c11",
        [{"name": "qvl32_patch_merger", "source": "QVL32", "type_id": 1, "feature_band": [2 * third, feature]}],
        hidden=hidden,
        feature=feature,
        seed=seed + 2,
    )
    report["c18"] = write_band_table(
        out / "c18" / "ace_lm_music_codes.safetensors", width=music_band, hidden=hidden, seed=seed + 3, tied=tie_word_embeddings
    )
    report["blocks"] = {
        "ace_L12": write_block_container(out / "blocks", "ace_L12", hidden=hidden, seed=seed + 4),
        "qvl32_vit_Lmid": write_block_container(out / "blocks", "qvl32_vit_Lmid", hidden=hidden, seed=seed + 5),
        "qimg_Lmid": write_block_container(out / "blocks", "qimg_Lmid", hidden=hidden, seed=seed + 6, with_charts=False),
    }
    report["tap"] = {"qimg_Lmid": write_tap_rows(out / "tap", "qimg_Lmid", hidden=hidden, width=16, seed=seed + 7)}
    write_json(out / "fixture_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--feature", type=int, default=DEFAULT_FEATURE)
    parser.add_argument("--music-band", type=int, default=DEFAULT_MUSIC_BAND)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual-mode", default="block_attnres", choices=("block_attnres", "none"))
    parser.add_argument("--tie-word-embeddings", action="store_true")
    args = parser.parse_args(argv)
    report = write_fixture_set(
        Path(args.out), hidden=args.hidden, feature=args.feature, music_band=args.music_band, seed=args.seed,
        residual_mode=args.residual_mode, tie_word_embeddings=args.tie_word_embeddings,
    )
    print(json.dumps({"out": report["out"], "child_parameters": report["child"]["parameters"], "blocks": sorted(report["blocks"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
