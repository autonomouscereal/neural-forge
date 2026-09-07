# Qwen-Image S6 source gap and bounded next capture

Read-only inventory on 2026-09-07. This is not a folded container, capture receipt,
installed child, media result, or permission to use a companion decoder.

## What actually exists

- Parent: `/home/cereal/comfyui/models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors`.
  Size 20,430,635,136 bytes; 1,933 tensors; 20,430,401,088 header-derived elements;
  all stored `F8_E4M3`; 60 blocks (indices 0 through 59). Its raw header SHA256 is
  `bd90f4d6af8d235cb3b46ccf5f426abb351995e28e852b04fc678a541d87158e`.
  **No full-file hash or tensor-body validation was performed.** This is an
  already-FP8 parent, not a new child quantization step.
- VAE: `/home/cereal/comfyui/models/vae/qwen_image_vae.safetensors` exists.
  Source loader identifies the applicable Wan 2.1 path as 16 channels, spatial 8x.
- Text teacher candidate:
  `/home/cereal/comfyui/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`,
  9,384,670,680 bytes. The actual Qwen-Image loader selects its Qwen2.5-7B path.
  This is a **capture-time teacher dependency**, not a final companion text tower.
- Parent implementation exists inside the stopped r5 ComfyUI container
  `c6bdbadf0563` (`comfyui:codex-minimax-h3-runtime-env-r5-20260808`),
  `/opt/ComfyUI/comfy/ldm/qwen_image/model.py`. SHA256:
  `14c805af8da13d31094c2c704e413cc282ef43f4afe17f77de9070bdca301f28`.
  Inspected by `docker cp ... - | tar -xO`; no container was started and no source
  copy or model artifact was persisted. Comfy core is not installed at the host
  path `/home/cereal/comfyui/comfy`.
- No QIMG-named real captures, folded MLPs, containers or C15 pairs were found in
  `/home/cereal/omni-campaign/media`, `/home/cereal/omni-campaign/composite/media`,
  or `/mnt/lexar/omni-campaign/media`. The expected
  `/home/cereal/omni-campaign/composite/corpus/media_stimuli` directory is absent.
  Local campaign source likewise has no `c16_qimg_capture.py`; the runbook labels
  that script NEW. `qimg_Lmid` in `c16_media_fixtures.py` is synthetic, not a capture.

## Exact block contract: generic `adaln6` is insufficient

Block 30 has 339,831,296 parameters, width 3,072, 24 heads of dimension 128.
It has separate image and text Q/K/V projections, four head RMSNorms, separate
attention output projections, separate time modulators and separate GELU-tanh
MLPs. Each stream's MLP has inner width 12,288 and both linear biases. Attention
concatenates text first, then image, and uses **one joint attention operation**.
These source branches must be represented inside the one counted child block;
they cannot be dropped or replaced by an external text model.

The source image/text LayerNorms are non-affine with epsilon 1e-6; Q/K RMSNorms
also use epsilon 1e-6. Image modulation uses fused
`addcmul(shift, normed, 1+scale)` and image residuals use fused `addcmul`.
The text attention residual uses a separate multiply and add; text MLP residual
uses `addcmul`. Those rounding boundaries differ and must not be homogenized.

Time input is the actual sampler sigma: `ModelSamplingFlux.timestep(sigma)`
returns sigma. `Timesteps` forms FP32 `sigma * frequency`, then multiplies by
1,000, concatenates cosine then sine, then casts to the source hidden dtype.
It applies `linear_1 -> SiLU -> linear_2`; each stream then applies
`SiLU -> its own six-chunk projection`. There is no saved scale-shift table.
The four global timestep tensors contain 10,229,760 parameters and must be counted.
Do not reuse an implementation that multiplies sigma by 1,000 **before** the
frequency multiplication without testing the FP32 rounding difference.

The model uses interleaved-pair, three-axis RoPE with axes `(16,56,56)` and
theta 10,000, not H3 split-half RoPE and not scalar token rank. Source rotations
are generated from FP64 frequencies, converted to FP32, then to activation dtype.
Image positions are `(image_index, row - floor(rows/2), col - floor(cols/2))`.
Text positions repeat a token index along all three axes, beginning at the larger
image-grid half extent. Actual text masks and rotation tables must be retained.

Source helper SHA256 values:

| Path relative to `/opt/ComfyUI` | SHA256 |
|---|---|
| `comfy/ldm/flux/math.py` | `ee3473e262894884b12eb3af4caa22d069273eb87db3549fb46c42c37f910ece` |
| `comfy/ldm/flux/layers.py` | `35f2dfbadb8b59de79c306f365f26c2baa0a7d54836414737cd781fc24e6a2bc` |
| `comfy/ldm/lightricks/model.py` | `c9ae1a8505aa0eabefdffe409083141e5e8461dd9d55f452c13f0f7b0c6466d8` |
| `comfy/model_sampling.py` | `8559ac2f700b788babf30cc16f410c93890e1f71485ac8f6a299ea36e2cb8717` |

These pin source math, not numerical GPU parity. The r5 runtime's source storage
dtype, activation dtype, dequantization/manual-cast policy and kernel selection
must be recorded at capture. Explicit dequantization from the FP8 parent is not
the same provenance as an unavailable BF16 original.

## Fourfold accounting trap

One biased GELU MLP has `2*width*inner + inner + width` parameters.

| Per stream | Inner width | Parameters | Qualifies for >=4x |
|---|---:|---:|---|
| Parent | 12,288 | 75,512,832 | baseline |
| Naive quarter width | 3,072 | 18,880,512 | no |
| Largest qualifying width | 3,071 | 18,874,367 | yes, by count only |

The retained down-projection bias makes naive quarter-width slightly too large.
If both stream MLPs are folded, parent/child MLP counts are
151,025,664 / 37,748,734. Whole block becomes 226,554,366, **not 4x**; keeping the
global time path makes 236,784,126 before charts. A shared bidirectional chart at
trunk width 5,120 adds 31,460,352 parameters. Whether one shared chart preserves
both source row types must be measured, not assumed.

## Ready capture plan, not yet launched

1. Prepare an immutable image/prompt manifest from the actual in-scope S1 stimuli
   when available. Preserve image hash, prompt text/hash, prompt-token IDs/mask,
   fit/holdout role, dimensions, seed and parent/VAE/teacher identities. Split by
   original image identity before extracting rows; no cross-sigma or crop leakage.
2. First implement the missing capture adapter around the existing r5 parent,
   not a re-created approximate parent. Initially scope plain text-to-image only:
   no reference images, ControlNet, added time conditioning or runtime patches.
   Capture one selected block (30/59 depth fraction), bounded image size and a
   small sigma grid. The sigma values must be actual values passed to the model,
   not inferred from a proposed `FLOW_TYPE_SHIFTS` constant.
3. Use source hooks for `post_input` to retain the actual image/text IDs and for
   `double_block` / block hooks to capture the complete joint context. Per forward
   retain image/text block input and output, both normalized/modulated MLP inputs,
   both MLP outputs, sigma, `temb [B,3072]`, image/text modulation `[B,6,3072]`,
   text attention mask, source IDs `[B,Ttxt+Timg,3]` and the actual rotation table.
   Include row-to-image, row-to-patch and row-to-prompt-token indices. Sampling
   rows is acceptable for MLP fitting but **not** for full-attention block replay.
4. Preserve one frozen source block plus its global timestep tensors and selected
   tensor digests. Keep compact per-sigma time/position references and grouped
   fit/holdout tensors, rather than accumulating redundant complete runs.
   Read/capture peak CPU RAM, GPU VRAM and disk use. Request a bounded tracked GPU
   unit from root only after script fixture tests and a reviewed resource plan.
5. Reproduce the actual time path and rotation table first; then replay the
   frozen block with both streams and compare both outputs. Keep kernel/dtype
   drift separate from the subsequent MLP fold error. Do not relax the existing
   1e-5 time-reference gate or claim fixtures prove parent parity.
6. Fit per-stream channel coresets at keep <=3,071, keeping first-layer rows and
   bias aligned and fitting the second projection with its bias. Report grouped
   holdout per stream, per sigma and full-block image/text residual errors.
7. Implement a native dual-stream charted representation and converter only
   against those pinned tensors/taps. Route image and text through counted
   source-specific parameters and joint attention, then write only image rows
   to the trunk. Explicit source coordinates and real paired host/source taps
   are required; dropping text, omitting positions or installing the current
   generic synthetic MMDiT fixture is not an alternative.

## Separate image interface gap

The actual VAE spatial stride is 8, and a parent DiT token consumes a 2x2 block
of 16-channel latents: 64 latent values corresponding to a 16x16 RGB patch
(768 raw values), before any chosen larger context. The runbook's 32x32 RGB /
3,072 raw values is therefore a larger grouping, not the native DiT token shape.
Capture must explicitly establish the row grouping and normalization (Wan21
latent format) before fitting C15 maps or host/source charts. No image absorption
quality has been measured here. Final materialization must stay within the single
child; a companion VAE decoder cannot satisfy the user's all-modality requirement.
No feature-band allocation or architecture width is changed by this work.

## Header preflight

`composite/tools/c16_qimg_preflight.py --weights <parent> --block 30` reads only
the header, validates the complete dual-stream tensor geometry, computes counts
including biases, and prints JSON to stdout. It deliberately reports
`fold_ready: false` with the missing evidence above. Tests validate the audit and
its refusals; they are not a replacement for the capture or runtime work.
The exact tested audit also ran successfully against the real parent header via
an in-memory, read-only server invocation: 234,040 header bytes, all 1,933 tensor
ranges accounted for, and the counts above. No source deployment or result file
was created. Local tests: 7 passed.
