# Native Hunyuan3D 2.1 folding inputs — 2026-09-07

The installed parent is **not** the Flux-like `hy3d_Lmid` fixture. Header and
native-source inspection finds `HunYuanDiTPlain`: 21 blocks, width 2048,
16 heads, 64-channel / 4096-token shape latents, a 1370-token DINO image
condition, affine LayerNorm, exact GELU MLPs, self-attention followed by
cross-attention, and skip inputs on blocks 11–20. Blocks 15–20 contain eight
routed experts plus a shared FFN (top-2 softmax masses are not renormalized).
The timestep MLP produces a **prepended token**, not six-way adaLN. The native
flow pipeline divides scheduler timestep by 1000 before model invocation.
There is no DiT RoPE or positional embedding in this configuration.

## Read-only source preflight

```bash
python c16_hy3d_contract.py \
  --parent-checkpoint /parent/hunyuan_3d_v2.1.safetensors \
  --source-root /source/ComfyUI-Hunyuan3d-2-1
```

Host checkpoint: `/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors`
(7,365,943,290 bytes). It includes `model.*`, `vae.*`, and `conditioner.*`.
Its 196,208-byte header SHA256 is
`0b44c4852a3bd475352accc9db5bc78e678bd4b36c9a313b6592a7e4aabc908a`.
The validator pins the inspected native config and implementation files,
checks tensor geometry and complete offset coverage, and reports separate
component counts. It does **not** hash or authenticate the full weight bytes.
Source changes require another audit, not an override to force a pass.

The separate `vae/Hunyuan3D-vae-v2-1-fp16.ckpt` (655,648,152 bytes) is ZIP/pickle.
It is deliberately not deserialized by preflight. The combined safe checkpoint
already contains the native ShapeVAE **encoder and decoder**. C15 must load
that complete native VAE strictly; decoder presence alone is not absorption.

## Capture receipt and tensor payload

Pass `--capture DIR` to validate `DIR/capture.json` plus its named safetensors
file. The JSON schema is `neural-forge.hy3d-native-capture.v1`; `source_identity`
is the preflight's exact identity object. `file` is a local basename and
`file_sha256` seals all capture bytes. Each `samples` entry has `object_sha256`,
`conditioning_sha256`, a nonnegative integer `seed`, and split `fit` or `held`.
An object stays in one split across all viewpoints/noise levels. `producer`
must explicitly declare `native_forward:true`, `strict_weights:true`,
`eval:true`, and `modified_attention:false`. These declarations are checked
as a contract, **not independently proven by metadata**.

For `kind:dit_block`, include `block_index`, `attention:self_then_image_cross`,
`time_path:sigma_prepended_token`, and
`position_path:sampled_anchors_no_dit_rope`. Preserve full sequences:

- `block_input`, `block_output`, `mlp_input`, `mlp_output`: `[B,4097,2048]`;
- `context`: `[B,1370,1024]`; `sigma` and actual model `timestep`: `[B]`, equal
  and in `[0,1]`; native `time_token`: `[B,1,2048]`;
- `latent_xyz`: `[B,4096,3]`, captured source anchor order (not invented RoPE),
  plus `surface:[B,81920,7]` and I64 `query_source_index:[B,4096]` to verify
  each anchor against the actual source surface;
- `skip_value`: `[B,4097,2048]` for blocks 11–20;
- for blocks 15–20, `router_logits:[B,4097,8]`,
  `router_topk_ids:[B,4097,2]` I64, same-dtype `router_topk_weights`, and
  `shared_output:[B,4097,2048]`. Native top-k masses are checked without
  a tolerance waiver; device/dtype differences need investigation if refused.

For `kind:shape_vae_pairs`, declare `posterior:mode`,
`anchor_source:native_encoder_query_pc`, `feature_kind:anchor_local_sdf`,
and positive `feature_samples:Q`. Preserve:

- `surface:[B,81920,7]` (XYZ + four source features),
  `query_source_index:[B,4096]` I64, `anchor_xyz:[B,4096,3]`, and
  `posterior_mode:[B,4096,64]`;
- `feature_offsets:[Q,3]`, `feature_queries:[B,4096,Q,3]`,
  `feature_sdf:[B,4096,Q]`, `parent_roundtrip_sdf:[B,4096,Q]`.

The native encoder's `sample_points_and_latents` uses random sampling followed
by FPS and returns `pc_infos[0]` query positions. Preserve its actual surface
selection/FPS mapping; merely setting `sample_posterior=False` does not make
its anchors deterministic. Validate every anchor against its exact source
surface index and every SDF query against anchor + offset. The existing D10
voxel/SDF feature decision and measured E/D fit remain required; this contract
does not pick a child feature band or change the trunk width.

## What remains before a folding claim

Real native captures, independent native-forward replay, grouped held-object
coreset scores, a dense union for any chosen MoE block, and counted C15/C16
conversion/install are still missing. Generic C16 currently lacks this exact
self+cross/time-token/skip/MoE contract. Do not install a fixture-shaped block
or drop cross-context as a substitute. A companion decoder cannot establish
that the child's trunk acquired 3D behavior. Validator tests establish only
mechanics; all reports explicitly retain `folded_child_proven:false`.
