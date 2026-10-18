# S6A / S6B input formats for `c16_install_media.py` (S7)

Status: format of record for the files S6A (C15 interface maps), S6B (C16 charted
blocks, C11 QVL32 rows, C18 music-code band) hand to `composite/tools/c16_install_media.py`.
`composite/tools/c16_media_fixtures.py` generates synthetic instances of every file
below (`--out DIR`), and `test_c16_install_media.py` round-trips them into a loadable
Chimera checkpoint. Every tensor file is a single plain safetensors file (no index);
every JSON is UTF-8 with sorted keys. Widths below: `H` = trunk hidden (5,120 at 27B),
`F` = `native_feature_dim` (6,144 at 27B), `V_band` = a declared ledger band width.

## 1. Interface row sets (`--c15 DIR`, `--c11 DIR`)

```
DIR/companions.json
DIR/<entry>/composed_interface.safetensors     (c15_absorb_run.py --hidden H writes this)
```

`companions.json` (schema `chimera_omni.composite.interface_set.v1`):

```json
{"schema": "chimera_omni.composite.interface_set.v1",
 "hidden": 5120, "native_feature_dim": 6144,
 "entries": [
   {"name": "music_ace", "source": "ACE", "type_id": 4,
    "feature_band": [0, 3840], "emission_band": [0, 3840],
    "rows": "music_ace/linear/composed_interface.safetensors",
    "rung": "linear", "companion_decoder": true,
    "psnr_absorbed_db": 23.09, "psnr_parent_roundtrip_db": 32.78}]}
```

* `type_id` is a `native_media.MEDIA_TYPE_IDS` value (image 1, video 2, audio 3, music 4,
  tts 5, ocr 6, volume 7).
* `feature_band = [b, e)` are the columns of the `F`-wide native feature vector this type's
  raw features occupy (the packetizer pads raw features into `F`); `emission_band` the rows
  of the `F`-wide emission the head predicts for it (defaults to `feature_band`). Bands of
  different entries must be DISJOINT on each side and fit inside `F`; the tool refuses
  otherwise (`--allow-band-overlap` sums overlapping bands and records the superposition).
  One `type_id` appears once per install (C15 + C11 together) unless overlap is allowed.
* `composed_interface.safetensors` keys (from `mediafold.c15_fit.ComposedInterface.state`):
  `bridge.feature_proj.weight [H, e-b]`, `bridge.type_embed_delta [H]`,
  `head.out.weight [e'-b', H]`, `head.type_affine_shift [e'-b']`, fp32.
* `tensor_sha256` (per entry, `{tensor name: digest}`) is `safetensors_io.TensorView.sha256()`
  of each tensor (dtype name + shape + bytes). The tool recomputes it from the file and
  refuses a mismatch or a declared name the file lacks; the manifest records `verified`.
* Precondition the tool asserts per type before it writes: the bridge's
  `type_feature_scale[type, b:e]` and the SCALE halves of `native_media_bridge.type_affine[type]`
  and `native_media_head.type_affine[type]` are zero (S0 initialises them zero; the S5 child is
  not S0). A non-zero row would pre-scale the composed rows without a receipt: refusal, or
  `--allow-nonzero-type-affine` to record the counts and proceed.
* `companion_decoder` records the D5b decision (below the absorption line the trunk keeps
  a companion VAE decoder for that band); the tool copies it into the manifest, it does
  not act on it.

What the tool writes: `native_media_bridge.feature_proj.weight[:, b:e] = E`;
`native_media_bridge.type_embed.weight[type_id] += delta`;
`native_media_head.out.weight[b':e', :] = D`; the head's `type_affine` SHIFT half for
`type_id` (`[H:2H]`) = the fp64 least-squares solution of `D s = decoder_bias` (residual
recorded: the head's bias lives in hidden space, the decoder's in feature space).

## 2. Music-code band table (`--c18 FILE`)

One safetensors file: `embed.weight [V_band, H]` and `lm_head.weight [V_band, H]` (fp32 or
the child's dtype). `V_band` must equal the child's declared band (`--band-name`, default
`music_codes`); a tied child requires the two tables equal. The target rows must be zero
in the child (S0 declared them zero) unless `--allow-nonzero-band`. An optional sidecar
`FILE.json` (any content; e.g. C18's aligned/unmapped counts over the full Qwen vocab) is
hashed into the manifest; when it carries `tensor_sha256`, the tool verifies the table
against it (same formula as §1) and refuses a mismatch.

## 3. Charted block containers (`--blocks DIR ...`)

```
DIR/charted_block.json          spec + provenance + coreset accounting + time_reference + tensor_sha256
DIR/charted_block.safetensors   the source block (coreset) in canonical names
DIR/charts.safetensors          chart_in.weight [P, H], chart_in.bias [P], chart_out.weight [H, P]
                                (chart_out.bias may be present; it is NOT installed -- a zero delta
                                must write nothing -- and is recorded as dropped)
DIR/time_reference.safetensors  REQUIRED for modulation "adaln6": sigma [n] and modulation [n, 6, P],
                                the CAPTURED temb (the parent's own timestep path at the capture's
                                noise level(s), WITHOUT the scale_shift_table). The installer rebuilds
                                the spec's time path (`ChartedSourceBlock.time_path`) and refuses the
                                block unless it reproduces this within rel_l2 1e-5.
                                c16 >= 1.2.0: REQUIRED for "h3_curve6" too: sigma [n] and modulation
                                [n, 3, 6, P] (video / text / audio order), same gate. Under
                                --charted-dtype match the four H3_FP32_PARAMETERS stay F32.
                                Modulation "none" must NOT ship this file.
```

Runtime contracts a container may declare in `charted_block.json:runtime_requirements`
(the installer records them; the S7 driver and the R3 host-row tool must honour them):
H3 `source_positions` (explicit h3_thw_v1 coordinates for text and AV rows); HY3D
`hy3d_time_token_row` (each volume segment starts with one type-7 time-token row: zero
feature, the segment's sigma as noise level, thwr_v1 position [-1,-1,-1,-1]; reference
`hy3d_time_token_reference.safetensors` = FP32 t_embedder(sigma)).

`charted_block.json` (schema `chimera_omni.composite.charted_block.v1`): every
`chimera_omni.charted.ChartedBlockSpec` field EXCEPT `layer_index` (the tool assigns it),
plus:

```json
{"schema": "chimera_omni.composite.charted_block.v1",
 "spec": {"source": "ace", "component": "ace.dit.block12.mlp", "type_ids": [4],
          "width": 2048, "heads": 16, "kv_heads": 8, "head_dim": 128, "mlp_width": 1536,
          "mlp_kind": "swiglu", "modulation": "adaln6", "norm_kind": "rms", "norm_affine": true,
          "norm_eps": 1e-6, "qk_norm": true, "attn_bias": false, "rope_theta": 1e6,
          "sliding_window": 128, "time_freq_dim": 256, "time_hidden": 2048, "time_factor": 1000.0,
          "time_embed_r": true, "delta_scale": 1.0, "chart_in_bias": true, "rung": "latents",
          "source_depth_fraction": 0.5, "metadata": {"timestep_r": "0.0 as captured"}},
 "provenance": {"source_weights": "...", "source_prefix": "decoder.layers.12.",
                "capture_receipt": "...c16_receipt.json", "capture_receipt_sha256": "...",
                "capture_file": "capture_L12.safetensors", "capture_sha256": "..."},
 "accounting": {"parent_mlp_parameters": 37748736, "child_mlp_parameters": 9437184,
                "parent_block_parameters": 62933504, "child_block_parameters": 34622464},
 "holdout": {"rel_l2": 0.5442, "cos": 0.8415, "split_by": "clip"},
 "time_reference": {"file": "time_reference.safetensors", "source": "capture_L12.safetensors:temb",
                    "sigma": [0.5], "timestep_r": 0.0, "conversion_rel_l2": 1.2e-07,
                    "chunk_norms": [4.9, 375.8, 32.2, 5.9, 124.8, 33.9]},
 "tensor_sha256": {"charted_block.safetensors": {"attn.q_proj.weight": "...", "...": "..."},
                   "charts.safetensors": {"chart_in.weight": "...", "...": "..."},
                   "time_reference.safetensors": {"sigma": "...", "modulation": "..."}},
 "insert_after": null}
```

* Tensor names in `charted_block.safetensors` are `ChartedBlockSpec.parameter_names("")`
  minus the three chart keys: `attn_norm.weight`, `attn.q_proj.weight`, ..., `attn.q_norm.weight`,
  `mlp.gate_proj.weight` / `mlp.up_proj.weight` / `mlp.down_proj.weight` (swiglu) or
  `mlp.fc1.*` / `mlp.fc2.*` (gelu), `scale_shift_table [6, P]`, `time_embed.linear_1.*`,
  `time_embed.linear_2.*`, `time_embed.time_proj.*` (adaln6 only) and, with
  `time_embed_r: true`, the same six under `time_embed_r.*`. Shapes are exactly
  `ChartedBlockSpec.parameter_shapes`; the tool refuses any other set.
* Modulation chunk order is ACE-Step's `(shift_msa, scale_msa, gate_msa, shift_mlp,
  scale_mlp, gate_mlp)`; capture scripts for MMDiT / Hunyuan / H3 permute into it.
  `time_embed` is the SOURCE's own timestep path (sinusoid `time_freq_dim`, `time_factor`,
  `linear_1 -> SiLU -> linear_2 -> SiLU -> time_proj`); the trunk drives it with the
  per-token noise level. ACE-Step has a SECOND such path, `decoder.time_embed_r`, that the
  parent drives with `t - t_r` and SUMS into the modulation (`temb = proj_t + proj_r`); every
  capture used `t_r = 0`, and it carries ~90 % of the modulation energy (captured chunk norms
  at sigma 0.5 `[4.9, 375.8, 32.2, 5.9, 124.8, 33.9]` vs `time_embed` alone
  `[2.1, 41.4, 5.0, 1.8, 8.3, 6.4]`). It is therefore KEPT (`time_embed_r: true`, evaluated at
  the trunk noise level), never dropped. Whatever a source's time path is, the container's
  `time_reference.safetensors` is the capture's own `temb`, and the installer refuses a spec
  whose `time_path(sigma)` does not reproduce it (rel_l2 <= 1e-5): this gate applies to the
  ACE conversion and to the H3 / QIMG / HY3D capture scripts alike -- a capture script that
  cannot emit a reproducible reference has not folded the parent's time path.
* `tensor_sha256` (`{file name: {tensor name: digest}}`, formula as §1) is verified per file
  on load; a stale digest is a refusal.
* Cross-attention is dropped at capture (the trunk is the context); a ViT block uses
  `modulation: "none"`; a 2-D RoPE source records `rope_theta: 0` (or the 1-D
  approximation it captured with) in `metadata`.
* `source_depth_fraction` = source block index / (source depth - 1). `--pair-by
  depth-fraction` inserts the block right after the trunk's gated-GQA refinement layer
  whose own depth fraction is nearest; `--pair-by explicit` reads `insert_after`.
* `--tap-rows DIR`: when `charts.safetensors` is absent, `DIR/<container name>/paired_rows.safetensors`
  (`host_rows [N, H]`, `parent_rows [N, P]`, paired at the same media frame and sigma) is
  fitted with `mediafold.c16_transplant.fit_charts(method="ridge")`.
* `mediafold`'s `c16_transplant_run.py` output (`folded_block_L{i}_keep{k}.safetensors`,
  `chart_ridge.safetensors`, `c16_receipt.json`, `capture_L{i}.safetensors`) converts into this
  container with `c16_install_media.py --convert-ace-run RUN_DIR --ace-weights
  acestep_v1.5_turbo.safetensors --convert-out DIR` (drops `cross_attn.*`, takes BOTH
  `decoder.time_embed.*` and `decoder.time_embed_r.*` from the ACE weights, writes
  `time_reference.safetensors` from the capture's `temb` at `capture.stimulus.sigma`, and refuses
  the conversion unless the rebuilt time path reproduces it -- the real `c16_L12_v4` run converts
  at rel_l2 ~1e-7).

## 4. What the tool emits (`--out DIR`)

A `chimera_omni.checkpoint.v1` directory (shards + index + config with the charted specs
in the manifest) and `composite_manifest.json` (schema
`chimera_omni.composite_manifest.v1`, sha256 sidecar): per source `mechanism -> rung ->
parameters -> sha256`, per-component accounting in the ledger's `--component
NAME:PARENT:CHILD[:PREFIX]` form (charted MLPs derive from `blocks.<n>.charted.mlp.`;
a component whose ratio exceeds 0.25 -- every blend-arm push of a source narrower than the
trunk -- is carried as `--passed-through`), interface / band writes with the bytes replaced,
block placements with their verified time references, the header count re-derived from the
written shards and asserted equal to input + additions, the disk / RAM accounting (streamed vs
materialised bytes, free space before), and the exact `oc_checkpoint.py register` line with
`register.reachable`: `--parent-checkpoint-id` is the LINEAGE parent (the S5 child) and the fold
accounting is priced against `--accounting-parent-directory` (PARENT-Q38-27BU's directory, read
from its headers by the ledger) because an insert child grows relative to its lineage parent
and `_fold_accounting` refuses `measured_drop <= 0`. With nothing to install the output tensors
are byte-for-byte the input child's (the zero-media control). The output is written by
streaming untouched tensors from the input shards; the tool refuses when the output filesystem
would drop below `--min-free-gib` (60).
