# Clean geometry recovery chart proposal — not approved or native-ready

The authored targets are genuine three-dimensional meshes. Actual ShapeVAE
default-posterior encoding and native MC64 geometry were proved for one box,
including repeated bitwise targets. Mode reconstructed that same parent surface
poorly, so new target preparation must preserve native `posterior.sample()`.
The complete 24-source warm session remains a separate source-level proof. The
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
an explicitly position-free **media bridge** branch using the same pinned clean
targets. The host trunk's sequential RoPE/recurrence still breaks original
parent permutation symmetry in both branches; neither is parent parity.
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

## Concrete proposed batch and preregistration contract

This proposal is `hy3d_categorical_latent_slot_supervised_flow_2026-10-13.v1`.
It remains unselected until the owner records its actual complete-sequence
native proof and dated preregistration amendment. Parent chart parity and parent
latent-set permutation parity are both explicitly false. No source latent XYZ,
surface normals, decoded geometry, reference answer or clean latent enters
generation conditioning or positions.

The genuine native parent template supplies all user→assistant-generation-prefix
IDs. Only actual user instructions are text inputs. Source assistant geometry
metadata is retained in source provenance, never appended to this prefix.
After the prefix come all 1,024 actual source-bound C11 image patch rows, the
original HY3D time row, and 4,096 noisy latent rows. No arbitrary caption-only
substitution or target-derived text is allowed. C11 patches are raw image inputs,
not learned teacher targets; the actual 24-image CPU bitwise parity receipt is
`reviews/2026-10-13-folding/recovery-preparation/geometry-conditioning-cpu-v1/proof.json`,
SHA `61f3f6e03e44bf714fc8f3e82d2d5af50afea216f4fced563c3ef7fd3c459c18`.

The proposed **bridge** coordinate convention is categorical latent slot
`[(k+.5)/4096, 0, 0, 0]`, k0…4095. These coordinates are generated from row count
alone and carry no physical spatial claim. The unchanged parent time row uses
the actual `TIME_ROW_POSITION=[-1,-1,-1,-1]` from pinned s7 source. Image bridge
positions are the actual C11 `[0,y/(gh-1),x/(gw-1),0]`. A full `[1,S,3]` source
position tensor must separately scatter every real prefix, image, time and
latent slot; its ordinal coordinates are a new observation convention, not
clean encoder points. The proposed latent source coordinates are `[P+k,0,0]`
for prefix text count P, with the time row explicitly marked as a sentinel.
The exact image/source scatter must be validated against the native reader
before the amendment is selected. The position-free bridge comparison uses
zero latent bridge coordinates, while retaining the genuine time sentinel and
image positions; it does not pretend to disable the host's sequence positions.

For each admitted clean model-space target x0, a separately prescribed Gaussian
noise draw forms `x_sigma=(1-sigma)*x0+sigma*epsilon`, 0<sigma≤1. Only x_sigma
occupies type7 input band576:640. Emission320:384 receives `epsilon-x0`. This is
authored waveform/geometry-supervised rectified flow, not guided teacher KD.
The source parent time convention is t_native0=noise,1=data, so parent time is
`1-sigma`; the existing folded unit's actual time-input conversion must be
source-audited and numerically proved before the writer/driver is accepted.
The time row and image rows have zero canonical media loss weight. All 4,096
intended latent rows have weight1, with both input and emission bands checked
and targets excluded from input bands. Full masks/indices/typeIDs/noise levels
and `native_media_loss_weights` are mandatory, not just the legacy mask alias.

The proposed training objective is flow weight1, guided/feature KD weight0 and
text answer CE weight0 for this distinct generation stage. It uses the existing
native HY3D unit, image ingress, bridge/emission path and preregistered trunk
adapter scope, without a replacement fold. Source admission intersects full
global group/frozen exclusion, per-source geometry-quality review, actual VAE
target receipt, actual C11 parity and selected amendment; currently that final
native intersection has not been granted. Source-eligible 8 train rows and all
16 held rows stay distinct. New permutation augmentation, if selected, must
permute x0, noise, noisy rows, velocity targets and loss masks together, never
derive generation coordinates from query XYZ, and prove original decoded
geometry invariance with actual parent execution before its own selection.

The retained held families require before/after actual model geometry evaluation
using fixed noise draws and native MC extraction, not cache-replay accuracy or
finite tensors. The source corpus covers only small authored polyhedra; it does
not establish natural-object, texture, arbitrary topology or full 3D competence.

## Actual native bridge/time/permutation investigation

The exact original OMNI1 clone config (file SHA
`3516b587ce80461745b772eb92bf1d3c0ffd40fc5e0d82d83c037ad872c99b02`)
has 69 layers: charted ACE32, H3 33, HY3D34, QIMG35 and QVL36. H3 has one
read/write site only, layer33. Actual native source, original checkpoint,
all 10 loaded bridge tensors and four original time tensors were independently
pinned. `geometry_bridge_probe.py` executed two prescribed real targets (first
box and first held octahedron), full 5,121 image/time/noisy-latent rows and the
original decoder, entirely on CPU. Model states were unchanged after comparison.

The actual proof is
`reviews/2026-10-13-folding/recovery-preparation/geometry-bridge-cpu-v1/probe.json`,
file SHA `087edd5e34e027394fc797be7a466bb641a90a2e791da2fed8def73c6581ad19`.
Measured comparison runtime was 139.258 seconds, excluding initial parent loading.
All reported model arithmetic was an explicit FP32 reference widening of genuine
stored tensors. This was boundary/decoder computation, not a full trunk forward
or a test of trained generation quality.

The original scheduler executed noise2→data5 under its native velocity3 and
progress .25,.5,.75,1, producing2.75,3.5,4.25,5 exactly. The actual flow pipeline
uses native time0=noise,1=data. Its unused `scale_noise` helper has opposite
naming and must not be substituted for the actual flow pipeline. Current OMNI1
passes supplied noise levels directly into Fourier features and noise_proj.
The unchanged S7 pairing time row also used native progress under the name sigma.
Therefore this new recovery path must keep **canonical corruption sigma**
separate from **type7 bridge progress1−sigma**. The actual projection pre-hook
observed that direct progress exactly at canonical .2,.5,.8. Original parent
time-token values at sigma versus1−sigma differed by relative L2 .696668 and
1.135411 off midpoint, so the conversion materially changes the time path.
This does not prove the bridge/trunk reproduces the parent evolved time token.

Original decoder permutation equivariance passed prespecified atol/rtol2e−4
for both real targets (relative L2 1.63e−6 and8.04e−7). The same 125 physical
queries retained logits with max error2.38e−7 and3.93e−6. Equality was numerical,
not bitwise. Actual latent query XYZ was unused in this decoder investigation.

Constant-zero latent bridge coordinates preserved row permutation to max
1.91e−6 and relative L2≤3.02e−9; its unchanged cos(0) contribution is constant
across slots. Actual time sentinel and image coordinates were retained. Fixed
ordinal slot coordinates caused relative L2 .160–.177 and max differences up
to1.022 after permuting noisy rows without their assigned slot identities.
Image/time embeddings matched exactly between policies. No quality experiment
selected a better model: the result demonstrates the additional slot identity
signal introduced by ordinal coordinates.

The recommended pretraining baseline is now **position-free latent bridge
coordinates**, while keeping actual image grid/time sentinel and explicit
source-blind sequence metadata. This recommendation follows measured boundary
symmetry and avoids adding an unproved latent slot identity; it is not a claim
of learned superiority. Host sequence order/RoPE/recurrence still break original
latent-set parity. Any ordinal alternative remains a separately preregistered
ablation requiring actual held geometry evaluation.
