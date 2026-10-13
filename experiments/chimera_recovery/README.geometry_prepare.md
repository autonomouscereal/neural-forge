# Actual geometry preparation, 2026-10-13

`geometry_prepare.py` reuses the original 24 authored closed meshes, preserves
their exact OBJ bytes, and corrects their conditioning images. Each new image
uses pixel-center barycentric rasterization, camera depth testing and exact RGBA
object coverage. The previous renderer sorted faces using original vertex Y,
which did not establish visible-face order for its rotated camera. No chart,
icon, caption, wireframe, or mesh coordinates appear in the new rendered image.

The complete original 144-fixture asset inventory and original groundtruth are
reverified. Global source admission traverses the complete normalized registry,
evaluation index, prior media admission, cross-route exclusions and frozen
prompts. Derived renders retain original source IDs, family groups and target
mesh hashes; view changes cannot evade a held-out family. The geometry manifest
has its own `recovery_geometry_preparation.v1` schema, while its typed admission
rows use `recovery_geometry_admission.v1`. The nested global-source checks reuse
the tested source graph algorithm from `audio_admission.py`; its legacy schema
name does not grant audio or native 3D readiness.

The prepared target surface is `[1,81920,7]` float32: normalized XYZ, face normals
and a zero sharp-edge flag. The recorded operator sampler draws uniformly by
real triangle area with seeded barycentric coordinates. Normalization matches
the installed loader's `.9999` centered-cube scale and retains its FP16
intermediate. This sampler is an explicit reproducible preprocessing policy,
**not parity with the stochastic upstream SharpEdgeSurfaceLoader**. The real
encoder probe recomputes the entire surface from the sealed mesh. Surface XYZ
and the encoder's selected query XYZ are target-side data only.

Generate a new namespace (PowerShell, repository root):

```powershell
C:/Python314/python.exe experiments/chimera_recovery/geometry_prepare.py `
  --pins 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_audio_admission_pins_oct13_v1.json' `
  --asset-root 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_media_seed_oct13_v2' `
  --output 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_geometry_prepare_oct13_v2'
```

`hy3d_surface_probe.py` defaults to preflight. Add `--execute` only for the owned
CPU proof. It verifies the whole original 7,365,943,290-byte checkpoint SHA,
reviewed config and source files, the copied manifest hash, exact mesh/image/
surface seals, and the real CPU FPS binary. It loads only the original 410 VAE
tensors, widened from their actual storage to FP32 with strict keys and shapes.
It calls actual `ShapeVAE.encode(surface, sample_posterior=False)` twice, captures
the real encoder query points with a read-only forward hook, checks bitwise
repeat after explicit Torch/NumPy/Python/libc RNG resets, uses the original
`scale_factor=1.0039506158752403`, and runs the actual decoder and geometry head.

The bounded geometric oracle uses independent convex-mesh half-space containment
on a 5×5×5 grid and reports both logit-sign accuracies. It does not infer native
logit sign, claim dense reconstruction, or accept a poor reconstruction merely
because tensors are finite. The diagnostic latent receipt never grants native
training readiness.

Example CPU argv after copying the v2 preparation and mounting the genuine
source/checkpoint plus root-owned CPU dependency overlay:

```bash
CUDA_VISIBLE_DEVICES='' HY3DGEN_DEBUG=0 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/opt/cpu-deps OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/opt/environments/python/comfyui/bin/python -B \
  experiments/chimera_recovery/hy3d_surface_probe.py \
  --source-root /home/cereal/comfyui/custom_nodes/ComfyUI-Hunyuan3d-2-1 \
  --weights /home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors \
  --prepared-manifest /work/geometry-v2/manifest.json \
  --prepared-manifest-sha256 '<actual v2 manifest file SHA>' \
  --item-id geometry-corrected-media-seed-05ed11dae684f64e9798e8af \
  --surface /work/geometry-v2/assets/media-seed-05ed11dae684f64e9798e8af/target_surface.npy \
  --mesh /work/geometry-v2/assets/media-seed-05ed11dae684f64e9798e8af/target_geometry.obj \
  --conditioning /work/geometry-v2/assets/media-seed-05ed11dae684f64e9798e8af/physical_object_rgba.png \
  --output /work/geometry-proof-v1 --threads 8 --execute
```

The current source audit found the installed parent and runtime intact but FPS
missing. Root separately prepared a read-only private CPU overlay and proved
the genuine CPU FPS operation. Torch seeding alone did not control its libc
random start; resetting both generators repeated the operation in all eight
recorded runs. This probe pins that actual `1.6.3+pt26cpu` binary and the existing
Torch `2.6.0+cu124` runtime, without changing the FPS algorithm or service packages.
Root's actual dependency receipt is
`reviews/2026-10-13-folding/recovery-preparation/hy3d-cpu-fps-v2/probe.json`,
SHA `7557bd55046a8fb202a684e29f2ced3cc13eea2a49fb8540fb1eb2bdd99115b1`.

The VAE has 327,746,177 parameters, about 1.22 GiB in FP32. Full surface cross
attention uses 4,096 query rows against 81,920 source points; CPU runtime and peak
RAM must be measured by the actual owned job. No query count or source geometry
is silently reduced to obtain a quicker proof.

## Remaining native recovery requirements

- Actual per-source VAE targets and acceptable geometric reconstruction evidence.
- Source-bound child QVL image features for these exact corrected RGBA assets.
  The folded native HY3D unit dropped original DINO cross attention; metadata
  declaring an image does not replace actual image conditioning.
- An explicit dated generation query-chart and objective amendment. Original
  HY3D uses no position embeddings and treats its latent set permutation
  equivariantly. Its mesh encoder chooses queries from clean target XYZ. Those
  points cannot be supplied to a generation model that has not observed them.
- A native complete-sequence batch/conditioning proof and actual trained
  reconstruction evaluation, before any native readiness claim.

The existing `hy3d_cache.py` remains an experimental guided-teacher packer and
cannot consume this separate clean-VAE source admission as a sealed trajectory
cache. These assets support a distinct supervised flow proposal; they do not
repair or relabel the earlier planar/icon teacher targets.

Run focused local tests:

```powershell
C:/Python314/python.exe -m unittest tests.test_chimera_geometry_prepare tests.test_chimera_hy3d_surface_probe -v
```
