# Executable native HY3D capture subpath

`c16_capture_hy3d.py` runs the actual pinned Hunyuan3D 2.1 ShapeVAE and,
optionally, the complete image-conditioned source DiT. This is a source evidence
producer, not an installed child, generated-mesh demo, or fidelity acceptance.

The separate contract validator remains necessary. Tiny CPU tests exercise the
producer mechanics with **explicitly synthetic test classes only**. The CLI has
no fixture/fake mode and refuses changed native source/checkpoint headers.

## Inputs and invocation

Run through the tracked runtime allocator, with read-only source, model and
real object/image directories and one fresh writable output directory. This
tool never allocates GPUs itself. For the audited source/checkpoint paths see
`HY3D_NATIVE_INPUT_CONTRACT.md`.

The object manifest has schema `neural-forge.hy3d-capture-objects.v1` and an
`objects` array of 2..8 entries. Each entry requires:

- `object_file`: real single-file `.ply` or `.stl` mesh; relative to manifest or
  absolute inside the runtime's read-only input mounts.
- `object_sha256`: exact mesh file digest (not a filename-derived object ID).
- `conditioning_file`, `conditioning_sha256`: actual corresponding image and
  its exact file digest. Correspondence is curator-provided, not inferred.
- `split`: `fit` or `held`; every occurrence of one object digest must stay in
  the same split, including different views/sigmas. Both splits are required.
- `seed`: integer 0..4294967295; `sigma`: explicit normalized flow time 0..1.

Example argv **inside an already allocated container** (paths must be actual
mounted inputs, not invented placeholders passed to the allocator):

```sh
python /workspace/composite/tools/c16_capture_hy3d.py \
  --source-root /native/ComfyUI-Hunyuan3d-2-1 \
  --parent-checkpoint /models/hunyuan_3d_v2.1.safetensors \
  --objects /objects/hy3d-capture-objects.json \
  --output /run-output/native-hy3d-capture-01 \
  --device cuda:0 --query-chunk 256 --block 15
```

Omit `--block` for the smaller executable ShapeVAE-only subpath. It still uses
the real encoder **and** decoder, not a companion decoder substitution. There
is no resume: preserve incomplete output and use a fresh sibling directory.

Runtime dependencies: the actual native source's PyTorch, safetensors, PyYAML,
NumPy, Pillow, einops, scipy/scikit-image, trimesh, torch-cluster, diffusers,
transformers, torchvision, OpenCV and OmegaConf. `torch-cluster` must match the
runtime PyTorch/CUDA ABI; no fallback FPS is provided. No package installation,
image build, or dependency compatibility check has been performed by this
source-only change. No network model download or pickle checkpoint loading is
used. Full parent weights are read for strict component loading, not hashed.

## What is actually captured

1. The pinned native `SharpEdgeSurfaceLoader(81920, 0)` creates the native
   normalized XYZ + normal + sharpedge-flag surface from each real mesh.
2. `ShapeVAE.encode(surface, sample_posterior=False)` produces posterior modes.
   Scoped observers record the **actual** native `randperm` and `torch_cluster`
   FPS result, preserving arithmetic, including duplicate-coordinate cases.
   Their composed original-surface indices must exactly gather the returned
   native encoder anchors. No grid, nearest-point reconstruction or guessed
   source coordinate is accepted.
3. `ShapeVAE.decode(posterior_mode)` and chunked native `geo_decoder` queries
   evaluate the same anchors. These outputs are explicitly named binary field
   **logits**, not metric signed distances. No SDF feature-band/child-width
   choice is made.
4. With `--block`, the native ImageProcessorV2 and checkpoint-loaded DINO
   conditioner process the real image. The complete native DiT is run at
   `x_sigma=(1-sigma)*(posterior_mode*vae.scale_factor)+sigma*noise`, with the
   **actual representable** sigma sent directly to the native time embedder.
   This is an explicitly recorded object-conditioned flow interpolation probe,
   not a generated scheduler trajectory or an end-to-end generation claim.
5. Hooks preserve selected block inputs/outputs, separate self/cross-attention
   writes, time token, skip value, MLP inputs/outputs and, for layers 15..20,
   actual gate-linear logits, native unnormalized top-k masses/IDs and shared
   expert output. All source experts and the remaining DiT blocks execute.

Only native package **namespace paths** are supplied for the top-level,
`models`, `models.autoencoders`, and `models.denoisers` packages. This skips
unneeded Comfy UI export imports; it does not replace computational modules,
attention, FPS, utilities, or functions with stubs. Real source files execute
under a private namespace. Additional transitive source hashes are pinned.

The explicit torch seed controls encoder randperm/FPS and DiT noise. The native
mesh loader also uses entropy-seeded NumPy generators: the seed is **not** a
claim that mesh resampling reproduces the surface. Exact sampled surfaces and
sampler traces are sealed for replay; the native loader is not patched.

## Outputs and remaining gates

`source.json` records audited source/header identity and extra source pins.
`sample-NNNN.safetensors` plus its JSON receipt seal surface, sampler lineage,
posterior modes, native anchor field queries/logits, and optional image/DiT
input-noise/output and separate attention writes. File digests bind receipts.

With `--block`, `block-NN/capture.safetensors` and `capture.json` contain the
existing strict v1 DiT capture contract across grouped fit/held objects. The
existing validator is run without changing its tolerances or router checks.
In particular, a GPU-vs-CPU router softmax replay mismatch is a real rejection
to investigate, not grounds for silently replacing observed masses.

`capture-run.json` is written only after all samples, the optional strict DiT
contract, and source/checkpoint stability checks pass. Exceptions preserve
partial evidence and never overwrite earlier artifacts.

Still missing: real allocated-runtime execution and native dependency canary;
curated corresponding mesh/image fit/held inputs; replay/fidelity evidence;
mesh-derived D10 local SDF features and semantic alignment to the parent's
binary field logits; faithful native HY3D converter/installer and held-out
folded-child demonstration. The previous `shape_vae_pairs` contract's
`parent_roundtrip_sdf` name must not be populated with uncalibrated binary
logits. This producer deliberately does **not** emit that contract or claim
that D10 pairing is complete.
