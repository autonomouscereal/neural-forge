# Clean geometry recovery chart proposal — not approved or native-ready

The authored targets are genuine three-dimensional meshes. Actual ShapeVAE
encoding and decoder geometry metrics remain separate proof obligations. The
earlier image-to-shape teacher source was planar; it must not be relabeled as a
physical geometry target or reused as supervision for these new source images.

The original DiT receives 4,096×64 noisy latent rows and a leading timestep row.
Its reviewed config sets `use_pos_emb=false`, and its self attention has no
spatial rotary positions. Original DINO image conditioning uses 1,370×1,024
features. The folded native unit drops that cross attention, so merely attaching
the old DINO tensors to a native batch cannot establish conditioned recovery.

The clean geometry encoder is a different computation: it samples 4,096 queries
from the clean target surface by genuine farthest-point sampling, embeds those
XYZ coordinates, and cross attends to 81,920 observed target surface points.
Its query coordinates are available only while forming labels. The clean mesh,
normals, raw clean latents and captured query XYZ must not enter generation
`input_ids`, native feature rows or generation chart positions.

A candidate native chart can use **categorical latent ordinal slots**, fixed
independently of the mesh and image: slot ordinal0…4095, constant time-channel
ownership and no physical XYZ claim. Image patch positions must instead come
from the actual source-bound child QVL processor. This is a new convention with
`native_parent_xyz_parity=false`, not a statement that every latent is a point
on a fixed physical cube lattice. Do not assign clean mesh coordinates to these
slots, infer them from labels, or copy one object's query chart to another.

Before implementing that candidate, compare a source-blind ordinal chart against
an explicitly position-free native branch using the same pinned clean targets.
The latent set's permutation handling is part of the proposed representation:
seeded target sampling alone gives reproducible labels but does not establish
spatially aligned slots across objects. Any target-only sorting must permute
latents and their diagnostic query metadata together, remain outside inputs,
and prove original decoder permutation invariance numerically. A new ordered
host representation needs its own dated amendment and held-family reconstruction
test; the diagnostic encoder receipt cannot approve it.

If admitted, supervised rectified flow would form `x_t=(1-sigma)*x0+sigma*epsilon`
using the actual encoded model-space latent, and train canonical velocity
`epsilon-x0` from **only x_t** at positive sigma. HY3D type7 feature band576:640
holds noisy64-channel rows; emission band320:384 holds velocity targets. Source
image type6 feature band640:2176 holds actual1,536-wide child image features.
The timestep row has zero loss weight; all intended rows require canonical
`native_media_loss_weights` and full-sequence position metadata. The current
experimental packer exposes only `native_media_loss_mask`, so its eventual
production adapter must bind the canonical alias and prove objective masking.

This would be authored clean-latent supervised recovery, not guided parent KD.
It must use a distinct cache schema, target-kind/policy, dated amendment and
source seals. No existing guided trajectory receipt or D3b seal is rewritten.
