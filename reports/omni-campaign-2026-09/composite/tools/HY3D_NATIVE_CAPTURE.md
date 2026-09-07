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

The producer also computes `geometry_identity` from the hash-bound PLY/STL
bytes, using the genuine Trimesh parser with `process=False`. The exact parsed
float64 triangle-set hash ignores metadata, vertex/face order, winding,
duplicate/unused vertices, repeated faces and negative zero. Equal geometry
must remain in one split even when file hashes differ. The computed identity
is sealed in each sample/DiT receipt. An optional declared identity must match
the complete computed object; a declaration can never override the guard.
Mesh/image digests are checked again before sampling after model initialization.

This is **not** a near-duplicate or original-object classifier: transformed,
rounded, remeshed, or otherwise edited versions still require curator-provided
original-object grouping in preparation. No coordinate normalization/rounding
or mesh edits are made by this guard. It refuses more than 1,000,000 faces or
3,000,000 vertices instead of silently reducing them. The native sampler still
consumes the unchanged original input file.

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
runtime PyTorch/CUDA ABI; no fallback FPS is provided. CPU-only inspection of
the existing reviewed image confirmed that `torch_cluster` is absent. No
package installation or image build has been performed by this source change.
No network model download or pickle checkpoint loading is used. Selected
component weights are read for strict loading, not full-parent hashing.

### Isolated FPS runtime extension (recipe, not a deployed runtime)

`Dockerfile.hy3d-capture` extends the reviewed local image
`sha256:735020f27cb59da7ca4ea62284154ad97fcbc1a765fa8ffc6c0abd9a2d283c0a`.
Before a separately authorized build, verify that exact image ID, tag it
`local/hy3d-base:735020f27cb59da7`, and verify the tag resolves to that same ID.
Use this tools directory as build context and `--pull=false`; do not substitute
another base or a registry pull. The recipe itself cannot establish the local
tag's image identity, so the builder must retain that check in its build receipt.

The only installed distribution is the CPython3.10/Linux x86_64 wheel
`torch_cluster-1.6.3+pt26cu124`, compatible by tag with Torch2.6/cu124.
`requirements-hy3d-fps.txt` pins the official URL and observed SHA-256; pip uses
`--require-hashes --no-deps --no-index`, without upgrades or fallback compilation.
The wheel was fetched into local process memory on 2026-09-07, not installed:
3,397,729 bytes, SHA-256
`707bc36cccd885cb18155f3e1718866c4f4e6df4160a00ee17b9457a41bf530b`.
Official index: <https://data.pyg.org/whl/torch-2.6.0%2Bcu124.html>.

The build must pass `hy3d_fps_self_test.py`: genuine two-batch CPU FPS through
the exact native six-argument API and compiled operator, with exact expected
indices, no CUDA initialization, and explicit unproven VAE/child status.
This script has not passed until a separately authorized build actually runs
it. xformers is not repaired/replaced; the source VAE uses native Torch SDPA.
A later allocated GPU FPS/ShapeVAE canary remains required. The recipe allocates
no GPU, loads no parent model, and does not install a companion model/decoder.

The producer, contract and runtime-recipe suites passed **88 tests** in 7.27s
using genuine Trimesh PLY/STL parsing in the existing `735020` image on
2026-09-07. Input meshes and model classes were explicitly synthetic fixtures;
this is parser/guard/receipt coverage, not source-VAE or modality proof.
The CPU-only disposable test container used runc, no network/GPU, a read-only
root filesystem and tmpfs; no image/package changes were made. Exact streamed
test archive SHA-256:
`bf2e4ef0e14fa59a6ebb867a6545d46a473fa3b7cc91f938115aa376ce9f6d38`.

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
