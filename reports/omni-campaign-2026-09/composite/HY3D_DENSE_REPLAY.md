# Native HY3D dense-block replay — 2026-09-07

`tools/c16_replay_hy3d.py` independently replays one original HY3D dense block
from a sealed `c16_capture_hy3d.py --block N` run. The first useful real capture
is **block 0**. Blocks 0–14 are supported; 15–20 contain MoE and are refused.
No actual parent capture, fitting, GPU allocation, host chart installation,
training, decoder replacement, or new quality gate is performed by this change.

## Existing input sufficiency

The existing producer already captures the required real inputs. No producer
change or reconstruction of missing context is necessary:

| Evidence | Location |
|---|---|
| Complete 4097×2048 block input, including evolving time-token row | Combined block capture |
| All 1370×1024 image-context rows | Combined block capture |
| Native normalized sigma, timestep, initial time-embedder output | Combined block capture |
| Actual skip input, for dense blocks 11–14 | Combined block capture |
| MLP input/output and complete block output | Combined block capture |
| Actual self-attention and cross-attention writes | Per-object sample tensor files |
| Actual initial DiT latent, sampled surface, anchors and query indices | Per-object sample tensor files |

The validator requires the **whole run directory**, not merely
`block-NN/capture.safetensors`. It checks the existing capture contract, exact
source/header identity, combined tensor SHA, ordered run/sample JSON receipts,
each per-object tensor SHA, object/image/seed/split/geometry identity, requested
sigma's F16 conversion, and exact per-object/combined surface/anchor/index
agreement. Geometry cannot cross fit/held partitions. Missing actual attention
writes are refused, not guessed from residual differences.

## Replay semantics

Only one original pinned `HunYuanDiTBlock`, native `TimestepEmbedder`, and native
initial linear embedding are instantiated. Their exact checkpoint prefixes are
strictly loaded as F16; the complete model is not allocated. All computation
uses unchanged native source functions. Models are frozen/inference-only.

Each object is replayed separately, matching the producer's batch size of one,
but retaining its **entire sequence and image context**. No token chunks,
attention kernel substitutions, truncation or stale caches are introduced.
Self-attention, image cross-attention, three native norms, native GELU MLP,
residual additions and (when present) concatenated skip projection/norm remain
inside the original block's `forward`.

The pinned full model passes `c=t_embedder(sigma)` because attention pooling is
disabled; every audited block has `timested_modulate=False`, so the separate `c`
argument is unused in its body. The validator still independently recomputes the
original time token and compares it to the capture. For block 0 it also checks
the complete initial input `cat(time_token, x_embedder(actual_dit_latent))`.
For later blocks, the captured token-zero row has evolved through prior blocks:
**it is never replaced with the original time embedding**. All captured skip
values are consumed unchanged; no skip is fabricated for early layers.

The baseline reports exact equality, maximum absolute error, RMSE and relative
L2 for every captured block boundary. Relative L2 is explicitly `null` if the
reference is zero and the residual is nonzero; there is no invented denominator
floor. Any nonexact baseline produces `baseline_mismatch` and exit code 2;
optional candidate replay is not attempted. This is an identity requirement,
not a new task-quality threshold. Device/kernel differences must be diagnosed,
not absorbed into an unreviewed tolerance.

## Optional already-frozen candidate

An optional `--folded-mlp-receipt` supplies an existing dense candidate, never
asks this validator to fit one. The JSON contract is:

```text
schema: neural-forge.hy3d-frozen-dense-mlp.v1
source_identity: exact existing audited HY3D source identity object
block_index: the captured dense block index
capture_sha256: exact combined capture tensor file SHA256
fit_object_sha256: sorted unique SHA256 list of exactly the capture's fit objects
file: a local basename ending .safetensors
file_sha256: exact frozen candidate file SHA256
```

The tensor file contains exactly `fc1.weight[K,2048]`, `fc1.bias[K]`,
`fc2.weight[2048,K]`, `fc2.bias[2048]`, all finite F16, with `0<K<=8192`.
Both biases are required. Only the native MLP's two linears are replaced; its
original `forward` and exact GELU remain unchanged. The candidate is evaluated
through the entire native block, so F16 residual addition/cast order is included.
The time, attention and MLP-input boundaries must remain baseline-exact.

The report gives per-object fit/held residuals and parameter counts/reduction.
It does not invent a compression or quality acceptance gate. Declared fit
provenance is labeled **receipt declaration, not independently audited fitting**.
No fixture artifact is supplied as a production candidate.

## Execution and remaining runtime gap

Only after root allocates a runtime and an authentic captured run exists:

```sh
/opt/environments/python/comfyui/bin/python /tools/c16_replay_hy3d.py \
  --capture-run /captures/hy3d-native --block 0 \
  --source-root /native --parent-checkpoint /models/hunyuan_3d_v2.1.safetensors \
  --device cuda:0 --output /reports/hy3d-block00-replay.json
```

This is **source-block replay**, not a Chimera host runtime implementation.
The inspected current charted runtime has no HY3D-native self-then-image-cross
block kind, native time-token insertion, three-norm layout or late dense skip
contract. An existing generic single-attention/adaLN or H3 block is not a faithful
HY3D converter target. Host/source paired charts and the corresponding native
charted representation remain separate work, as does authentic geometry-bound
VAE/field integration. This validator does not claim those are installed.

CPU tests use explicit synthetic evidence and tiny weights. With
`HY3D_NATIVE_TEST_SOURCE` pointing at the real pinned source, the optional test
runs original native dense and skip block classes, full fixture context, F16
weights, and changed-MLP residual checks. This establishes replay mechanics,
not parent mesh quality or folded-child modality capability.
