# Native HY3D parent calibration meshes — 2026-09-07

`tools/c16_hy3d_parent_meshes.py` generates a preregistered set of **source-parent**
meshes from already frozen real QIMG image artifacts. These are calibration
inputs for the native HY3D capture producer, **not folded-child proof**, real-world
object ground truth, metric SDF labels, or evidence of native child rendering.
There is no training, optimization, seed search, retry selection, texture model,
background-removal model, external decoder, or deployment in this tool.

## Input contract

Supply a fresh JSON plan with schema
`neural-forge.hy3d-parent-calibration-images.v1`, `qimg_receipt_file`, its exact
`qimg_receipt_sha256`, a `generation` object, and `images` (2–8 entries).
The referenced receipt must be the actual completed
`neural_forge.qimg_parent_images.v1` receipt. Each image entry contains:

| Field | Requirement |
|---|---|
| `id` | Existing unique QIMG image ID, safe filename characters |
| `conditioning_file` | Actual unmodified 512×512 RGB/RGBA PNG, relative to plan or absolute |
| `conditioning_sha256` | Exact artifact SHA; must also match the QIMG receipt row |
| `split` | `fit` or `held`; original QIMG `fit` / `holdout` mapping cannot change |
| `seed` | One preregistered uint32 native CPU mesh-noise seed |

Both splits are required. Identical image bytes cannot cross splits. Generated
meshes are checked by the exact canonical triangle identity from
`c16_capture_hy3d.py`; changed filenames, winding, vertex order, or file metadata
do not make identical geometry independent. This guard is not semantic or
approximate-geometry duplicate detection. A collision refuses the whole run;
there is no automatic retry or deletion of partial results.

The complete generation object (no omitted or extra fields) can be:

```json
{
  "steps": 20,
  "guidance_scale": 5.0,
  "bounds": 1.01,
  "octree_resolution": 128,
  "query_chunk": 1024,
  "capture_sigma": 0.5,
  "volume_decoder": "VanillaVolumeDecoder",
  "surface_extractor": "MCSurfaceExtractor",
  "mc_level": 0.0,
  "latent_scaling": "none_installed_Hy3D21VAEDecode",
  "seed_device": "cpu",
  "postprocess": "none",
  "dtype": "float16"
}
```

This is an example preregistration, not a searched winning setting. `capture_sigma`
is the subsequent native posterior/DiT capture interpolation parameter; it does
not alter the native parent generation schedule. Bounds: 2–50 steps, resolution
8–256, 1–2048 queries per geo-decoder call, guidance 0–10, spatial bound 0.5–2.
`mc_level` must be JSON floating-point `0.0`, not integer `0`.

## Exact source execution

The complete pinned original `pipelines.py` and `schedulers.py` are imported,
including their real Comfy dependencies. Private package namespaces bypass only
UI-oriented package `__init__` exports; no computational source is edited,
extracted with AST, or replaced. All checkpoint components load exact audited
prefix keys/shapes as stored F16 and run frozen, without gradients.

1. Actual `ImageProcessorV2` alpha-aware recentering, mask and resize, then the
   actual native DINO conditioner. Opaque QIMG PNGs are all foreground; no new
   segmentation model is silently introduced.
2. Actual complete 21-block DiT on both conditional and unconditional streams,
   full 1370-token image context, native timestep token and native attention.
3. Actual native flow scheduler, ascending `linspace(0,1,steps)`, with the source's
   appended final sigma `1` **retained even though it duplicates the endpoint**.
   Model time is cast to F16 *before* division by 1000. The observer returns all
   original forward/scheduler outputs untouched.
4. Actual installed `nodes.py` Hy3D21 decode route: `vae.decode(latents)` then
   `latents2mesh`; **no division by `scale_factor`**. The unused pipeline `_export`
   routine does divide by that value, so it is deliberately not called here.
5. Native `VanillaVolumeDecoder`, dense `(resolution+1)^3` F16 query positions,
   native geo-decoder, F32 grid output, and native `MCSurfaceExtractor` using
   Lewiner marching cubes at binary-logit level zero. Native vertex mapping uses
   `grid_size=(resolution+1)` in its denominator and is not corrected/replaced.
6. The installed node's face-winding reversal is retained. PLY serialization uses
   `Trimesh(process=False)` so it cannot merge vertices or repair geometry. Native
   triangle coordinates are compared to the reparsed output's exact geometry
   identity. No simplification, repair, remeshing, rescale or texture sidecar.

Captured source evidence per image includes processed image/mask, complete CFG
context, each step's full latent input/output, both raw DiT predictions, actual
time token/model time/scheduler time, actual combined CFG prediction, sigma pair,
scheduler indices, and dense **binary field logits, not metric signed distance**.
Hooks require one forward through every block per step and finite values at
each boundary. Trajectory evidence is bounded to 256 MiB per image.

## Output and runtime boundary

Fresh output only. The tool seals a preregistered plan before sampling, then one
PLY, tensor evidence file, and provenance receipt per declared image. On complete
success it emits `objects.json` directly usable by `c16_capture_hy3d.py`, plus
`mesh-run.json`. Both original image identity and exact generated geometry remain
bound into later capture. A partial run is preserved and never labeled complete.
Files/source pins/checkpoint stat/header identity are rechecked; no full-parent
rehash is performed. Parent identity uses the existing audited header/source
contract, not a falsely reported whole-checkpoint SHA.

Example command, **only after root allocates and records the runtime/GPU**:

```sh
/opt/environments/python/comfyui/bin/python /tools/c16_hy3d_parent_meshes.py \
  --plan /inputs/hy3d-parent-plan.json --output /outputs/hy3d-parent-meshes \
  --source-root /native --parent-checkpoint /models/hunyuan_3d_v2.1.safetensors \
  --comfy-root /opt/ComfyUI --device cuda:0
```

The reviewed base is image
`735020f27cb59da7ca4ea62284154ad97fcbc1a765fa8ffc6c0abd9a2d283c0a`,
Torch 2.6.0+cu124 / Python 3.10.12. The pinned FPS extension recipe is useful for
the later VAE **encode** capture, but this generation route only decodes and does
not call FPS. Do not repair xformers or substitute attention/FPS silently.

F16 source storage avoids a BF16 requirement on a P40, but **P40 execution is not
proven by CPU fixtures**. Actual PyTorch native SDPA dispatch, its transient
attention memory, and full native-model finiteness still need a separately
allocated canary. GPU0 after the QIMG job is the lower-risk first real canary;
do not silently change precision or attention kernels for a P40. All three
components are resident for this minimal native route (~7.37 GB stored weights,
plus activations and allocator space). Strict module construction also needs
transient host FP32 model storage before exact F16 assignment; root must leave
host RAM headroom. This tool does not allocate/schedule a GPU itself.

## Tests

`test_c16_hy3d_parent_meshes.py` clearly separates synthetic manifest/model/field
fixtures from real source code. Set `HY3D_NATIVE_TEST_SOURCE` only with the actual
pinned source mounted. The CPU source test executes the genuine scheduler,
native dense volume decoder, MC extractor and PLY parser using a **synthetic
field**, without loading parent weights. This passed in the preserved image.

The complete genuine pipeline import also needs
`HY3D_NATIVE_TEST_PIPELINE=1` in a separately allocated driver-visible runtime.
On this image, CPU-only real Comfy import reaches
`comfy_kitchen -> triton.autotune -> driver.active` and raises
`0 active drivers`, even with Comfy `args.cpu=True`. The CPU-only run explicitly
skips this test; no backend is stubbed and no computational imports are edited.
The optional test itself still uses tiny CPU synthetic components after import.
Neither test establishes parent mesh quality or folded-child capability.
