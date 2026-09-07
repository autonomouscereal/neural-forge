# Authentic QIMG capture adapter

Implemented by `composite/tools/c16_qimg_capture.py`. This captures the existing
parent, not a substitute implementation. No capture has been launched by the
implementation/tests. The preparation adapter now exists; actual teacher
execution, capture and review remain separate evidence gates.

## Input bundle

The CLI takes `--inputs capture_inputs.json`. Files may be absolute paths or
relative to the manifest. Image paths identify the **original image**, not a
different independently named crop; crops/sigmas of the same original image must
stay in the same split. The validator detects image-byte identity overlap across
fit/holdout; it does not detect visually equivalent files or semantically duplicate
images. No synthetic fallback, random text conditioning or zero-latent generator
exists in this adapter.

Required manifest schema (placeholders below are not a usable capture artifact):

```json
{
  "schema": "neural_forge.qimg_capture_inputs.v1",
  "kind": "real_image_vae_qwen25_context",
  "latent_space": "wan21_model_input",
  "context_kind": "qwen25_vl_7b_hidden_states",
  "encoding_receipt": {"path": "encoding_receipt.json", "sha256": "<actual receipt SHA256>"},
  "entries": [{
    "id": "image001",
    "split": "fit",
    "image": {"path": "original_image.png", "sha256": "<actual image SHA256>"},
    "prompt": "<actual encoded prompt>",
    "prompt_sha256": "<SHA256 of UTF-8 prompt bytes>",
    "tensors": {"path": "image001.safetensors", "sha256": "<actual tensor-file SHA256>"}
  }]
}
```

Each entry's safetensors file contains exactly:

| Tensor | Shape and dtype | Meaning |
|---|---|---|
| `model_inputs` | `[N,16,1,H,W]`, FP32/BF16 | Actual per-sigma model-space noisy inputs; positive even H/W |
| `sigma` | `[N]`, FP32 | Actual timesteps passed to the model, finite in [0,1] |
| `context` | `[1,L,3584]`, FP32/BF16 | Actual Qwen2.5-VL prompt context, reused across this image's sigmas |
| `context_mask` | `[1,L]`, BOOL | Actual valid-context rows; at least one valid row |
| `context_token_ids` | `[1,L]`, I64 | Encoder token IDs aligned to the returned context rows, after any encoder-side cropping |

These are **model inputs**, not merely clean VAE latents. The upstream producer
must record the actual Wan21 normalization and noise/sampler recipe. This adapter
does not infer the schedule, apply another shift, synthesize noise, or encode the
image/text again. A sigma grid must not be fabricated from unrelated observations.

`encoding_receipt.json` must have matching `kind` and `latent_space`, and:

```json
{
  "status": "complete",
  "kind": "real_image_vae_qwen25_context",
  "latent_space": "wan21_model_input",
  "vae": {"path": "<actual VAE weights>", "header_sha256": "<actual raw header SHA256>"},
  "text_encoder": {"path": "<actual Qwen2.5-VL weights>", "header_sha256": "<actual raw header SHA256>"},
  "entries": {
    "image001": {
      "image_sha256": "<same image SHA256>",
      "prompt_sha256": "<same prompt SHA256>",
      "tensor_file_sha256": "<same tensor file SHA256>"
    }
  }
}
```

Producer recipe, software revision, image preprocessing, VAE normalization,
sigma/noise construction and text-token alignment should also be retained in this
hashed upstream receipt. The adapter verifies file identities, source headers and
the exact image/prompt/tensor linkage. It **does not numerically verify the upstream
encoding computation**; that limitation is written into the output receipt.

Manifests/encoding receipts are bounded to 1 MiB each, image files to 128 MiB,
each tensor file to 64 MiB. Full context is bounded to 2,048 text-plus-image tokens
and the whole run to 32 forwards, with smaller limits selectable. Header bounds
are checked before tensor allocation. There must be an explicit split on every
entry; an initial parity-only capture may lack holdout but is not fold evidence.

## Production entry point

### Frozen input preparation

`tools/c16_qimg_prepare_inputs.py` accepts a hashed-image plan with schema
`neural_forge.qimg_encoding_plan.v1`, an explicit `sigmas` list in `(0,1)`, and
`entries` containing `id`, `split`, `seed`, `prompt`, hashed `image`, and
`image_provenance` (`kind` and `description`). Existing teacher-generated images
must be labeled as such; image-pixel input is not automatically a photograph or
quality benchmark. The plan is capped at 32 forwards. Each image must already
be 16-aligned and no larger than 512x512; no implicit resize or crop is performed.
EXIF orientation and white alpha compositing are recorded with the decoded-pixel
digest. `--validate-plan-only` decodes/checks these identities without Comfy or GPU.

The production run takes `--plan`, `--vae`, `--text-encoder`, `--comfy-root`,
`--device cuda:0` and a new `--out`. It hashes eight reviewed r5 code files,
uses the actual Qwen-Image text teacher and Wan21 VAE, and records source header
identities. It does not authenticate all teacher payload bytes. The exact native
system/user-template crop must match returned context rows. FP16 context, if
returned, is stored losslessly as FP32. VAE complete key/shape coverage is checked;
the native encoder returns posterior mu, not a random posterior sample.

Actual VAE latents are normalized once with `Wan21.process_in`. A seeded CPU
Gaussian draw is shared across each image's declared calibration sigmas, using
the native `CONST.noise_scaling` formula. These are deliberately chosen noise
fractions, **not observed sampler states**, and no second timestep shift is
applied. No gradient, optimizer, distillation or fine-tuning is involved. The
resulting tensors, origin/recipe/token alignment and hashes are written to the
upstream receipt; only a complete run emits `capture_inputs.json`. All partial
outputs remain preserved on error. The end-to-end fixture suite checks compatibility
with the downstream identity validator, not source-model fidelity.

### Source block capture

First, on the intended mounted paths, validate the identities and sizes without
loading Comfy, allocating a GPU or writing output:

```text
python c16_qimg_capture.py --weights <existing qwen_image_fp8_e4m3fn.safetensors>
  --inputs <capture_inputs.json> --out <new capture directory>
  --max-forwards 4 --max-tokens 2048 --validate-inputs-only
```

For the separately allocated, root-reviewed capture unit, remove
`--validate-inputs-only` and supply `--comfy-root /opt/ComfyUI --device cuda:0`.
This is an execution interface, not a GPU allocation/deployment command. The caller
must use the project's tracked runtime allocation and mount the exact reviewed
source and inputs. `--device cpu` exists but does **not** make the full 20B parent
a small CPU test; unit tests use explicitly labeled tiny fixtures instead.

The parent header must match the audited FP8 file. The six reviewed source files
are hashed and matched before import. The loader uses the actual Comfy diffusion
loader with FP8 storage, manual-cast inference, FP8 optimizations disabled,
no dynamic patcher, no custom operations and full residency. Selected block/time
runtime tensors must byte-match their original stored FP8 tensors.
Comfy loader uses its default inference reserve: parent bytes are already counted
by its model-memory accounting, not repeated as extra inference scratch. The
reviewed model-management source is pinned with the numerical source files.
The exact
compute dtype, storage dtype counts, TF32 settings, visible-device environment,
source code identities and Python/Torch versions are recorded. Full parent-file
SHA256 is deliberately not recomputed or claimed.

Plain text-to-image only: reference images, added time conditioning, patched
attention/blocks and zero-timestep reference regions are refused. The parent
executes its entire forward. Hooks do not supply or alter its activations.

## Outputs and invariants

The destination must not exist; there is no overwrite/cleanup flag. The adapter
checks a conservative output estimate plus a 4 GiB remaining-disk reserve before
model load. It stores one `source_block_and_time.safetensors` with the selected
block's complete 32-tensor dual-stream state and four global timestep tensors,
verbatim FP8 from the existing source file. It does not save a full parent copy.

One `image001_sigma00.safetensors` is written per actual forward:

- `img_in`, `img_out`, `txt_in`, `txt_out`: full block residual inputs/outputs.
- `img_mlp_in/out`, `txt_mlp_in/out`: source MLP inputs and pre-gate outputs.
- `img_attn_out`, `txt_attn_out`: source attention output projections, pre-gate.
- `temb [1,3072]`; `img_modulation`, `txt_modulation [1,18432]`: the actual raw
  six-chunk modulation outputs, without renaming one stream into the other.
- `source_ids [1,Ttxt+Timg,3]`, `rotation_generated` and `rotation_used`:
  actual position embedder inputs/output and the activation-dtype rotation the
  selected block received. Generated-to-used casting must reproduce exactly.
- `text_attention_mask`: the actual additive source mask received by the block;
  `context_valid` and `context_token_ids` retain the upstream row interpretation.
- `joint_row_kind [Ttxt+Timg]`: 0=text, 1=image; `image_patch_indices [Timg,3]`:
  uncentered time/row/column patch indices. These are validated against the actual
  source's centered image IDs and repeated-axis text IDs, not assumed packet IDs.
- `sigma`, `model_input` and `prediction`: actual complete parent forward input,
  timestep and prediction. Prediction shape must match model input shape.

All attention-context rows are kept, including masked text rows with their mask.
MLP fitting may select rows later, but full-attention replay must not use a sampled
subset. Hooks clone before source mutations and are removed on normal completion,
source exceptions, validation failures and partial hook registration. Missing or
repeated taps are refused, never filled with invented tensors. Non-finite captures
are refused.

`capture_receipt.json` records shapes, dtypes, file/tensor digests, input linkage,
per-forward image identity/split/sigma, casting error for supplied inputs/context,
CPU RSS/GPU peaks, elapsed time and source-header stability. Its digest formula is
explicitly labeled; it is not silently substituted for Chimera's tensor-digest
format. On failure, status remains `incomplete` and available artifacts are
preserved. Only all completed forwards produce status `complete`; neither status
claims a fold or media capability.

## Verification performed locally

25 CPU tests pass across the capture and header-audit suites. These test hook
fidelity in FP32/BF16, source input preservation before in-place writes, complete
context and rotation capture, production rejection of fixture classes, identity
linkage, split leakage refusal, bounds, missing/non-finite taps, exception cleanup
and byte-preserving FP8/FP32/BF16 output serialization. The fixtures are explicitly
labeled. No actual parent inference, real-image encoding, capture, GPU allocation
or deployment was performed during this implementation.
