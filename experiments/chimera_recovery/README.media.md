# D3b media recovery driver — 2026-10-13

`train_media.py` now prepares a real production entry point for the existing
HybridD3 engine. It implements D3b guided-velocity training only. D3a refit,
D3c music-code band training, D3d vision/video answer KD, and audio-understanding
teacher/adapter work remain separate requirements; no all-modal readiness claim
is made by this driver.

The exact trainable set is the selected native unit, its interface-out emission
slices/type-affine rows, and r16 LoRA in text layers d through the end. Other units,
layers below d, interface-in, text embedding/head and source-frozen coefficients
stay frozen. The driver calls the real `d3b_forward_backward`, with the sealed
velocity objective callback. It recaptures x_d on the actual base with frozen
lower-layer adapters and stores it using HybridD3's digest-verified 20GB chunk
writer/reader. It does not train on stale S3/S5 prefix captures.

Production preflight requires digest-pinned files and attestations for a
registered immutable clone, native hybrid parity on that clone, a dated recovery
amendment with the original frozen D1-D4 body unchanged, accepted train-only
native batches excluding heldout source groups, actual sealed guided teacher
targets, and an assigned runtime. Each accepted batch has native tensor names
documented in `training.py`, plus a compatible `ReadyObjectiveAdapter` receipt.
Input features and emission targets have explicit allocation layouts; H3 uses
actual packed source positions. Every source group, cache and teacher must remain
the same on resume.

Every sealed cache receipt pins `noisy_feature_bands` and its canonical JSON
SHA256. No loss silently selects the first channels of a conditioning band.
Native ACE packs source64, chunk-mask64 and noisy64 in c*2+k order, so its noisy
suballocation is feature256:384 to emission0:128. Music generation additionally
requires the source and chunk-mask features to be zero under a target-free
generation recipe; the old clean-source S8 stimulus is rejected. The tiny random
fixture explicitly declares its own six noisy channels and never claims native
ACE recipe validation. The optional existing D3 rate hook now also requires
`--noisy-feature-bands <allocation.json>` with its velocity adapter.

The frozen D3 section does not pin its own optimizer/LR defaults. A production
config must explicitly supply unit/emission/LoRA LRs, betas, epsilon, weight decay,
clipping, warmup and minimum LR ratio, recorded in the amendment. This driver has
no guessed production optimizer defaults. The existing section-2 checkpoint
policy is enforced: 30-minute safety snapshots, two rolling safety directories,
and persistent non-finite skip guards. Initial, failure, review and final
checkpoints are preserved separately. A checkpoint contains exact trainables,
Adam state, sampler counter, skip window, CPU/Python/CUDA RNG and all identity
pins. Changed identity or a changed checkpoint digest refuses resume.

At 25% of the budget, the production run requires a decode review tied to this
run identity and trained step, with meaningful prompt-conditioned content. It
halts with an exact checkpoint while that review is unavailable. A small loss
or structurally valid media does not satisfy that content check. The emitted
delta package is reviewable and unregistered; the OC checkpoint composer must
map its unit deltas, emission slices and LoRA sites before checkpoint registration.

```bash
python experiments/chimera_recovery/train_media.py preflight \
  --config /path/to/pinned-d3b-config.json --out /path/to/preflight.json

CUDA_VISIBLE_DEVICES='' CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 \
  STREAMED_KD_TRAINER_PATH=/home/cereal/omni-campaign/distill/tools/streamed_kd_trainer.py \
  python experiments/chimera_recovery/train_media.py cpu-fixture \
  --hybrid-module /home/cereal/omni-campaign/distill/tools/d3_hybrid_module.py \
  --out /path/to/new/cpu-fixture
```

The CPU fixture creates an actual tiny random OMNI-shaped child through the
existing D3 builder, runs three real streamed forward/backward optimizer updates,
compares interrupted/resumed continuation tensor-exactly, verifies changed
identity refusal, and writes a delta. It uses small geometry and labeled fixture
LRs; it never claims production model or teacher evidence. Torch-dependent tests
can be run with `CHIMERA_D3_BASELINE` pointing to the exact D3 module and the
foundation source environment above.

No production job is started as part of this preparation. Sealed native teacher
cache adapters and their accepted batch manifests remain the immediate external
prerequisites. Missing synchronized H3 AV or genuine HY3D targets stay blocked.

`media_cache.py` implements the genuine Stage-C QIMG velocity-cache conversion.
It requires a sealed cache manifest SHA and exact teacher item file, verified
connected-group media admission with frozen eval pins, and the real parent
tokenizer. It reconstructs x_t with the original CPU generator and proves the
recorded noise SHA, denoising identity and CFG branches before packing any
native rows. Inputs use feature512:576, targets use emission256:320, with actual
parent-template prompt tokens and normalized native image-grid positions.
The driver checks placeholder tokens against the actual base ledger. Teacher,
corpus, amendment, native tensor, source cache and source group pins travel into
each receipt. The student raw head and QIMG sign=-1 map to canonical velocity.

```bash
CUDA_VISIBLE_DEVICES='' python experiments/chimera_recovery/media_cache.py \
  --cache /path/to/sealed/qimg-velocity-cache --manifest-sha256 <sha256> \
  --items /path/to/qimg_items.media_a.jsonl \
  --admission /path/to/verified/media-admission/manifest.json \
  --tokenizer /home/cereal/models/Qwen3.8-27B-Uncensored \
  --amendment-sha256 <sha256> --out /path/to/new/native-qimg --limit 2
```

`--limit` produces a bounded CPU preparation proof with an unsealed batch
manifest; the production driver refuses it. Omit it only for an authorized full
cache conversion. ACE needs an additional audited semantic-code conditioning
prefix and its actual teacher source/chunk-mask recipe. H3/HY3D conversion needs
their genuine synchronized and geometry cache contracts. QIMG preparation does
not claim those adapters are available.
