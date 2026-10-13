# D3b media recovery driver — 2026-10-13

`train_media.py` prepares D3b guided-velocity recovery through the existing
HybridD3 engine. `train_stages.py` dispatches the separate D3a, D3c and D3d
engines described below. Source teacher caches, actual modality alignment and
sealed accepted native batches remain requirements; these drivers do not make
an all-modal readiness claim.

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
requires the authentic teacher-code detokenizer source and one-valued chunk
masks, with exact code IDs bound to the native music vocabulary and receipts.
The old clean-source S8 stimulus and earlier zero-source recipe are rejected. The tiny random
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
manifest; the production driver refuses it. Omit it only for a full cache
conversion. Storage is `native_band.v1`: the actual typed feature and emission
bands are saved once and expanded to6144 channels on load. Expansion is tested
bitwise against the original dense tensors, including optional channel masks.

`ace_cache.py condition` uses the exact pinned Comfy ACE classes and32 actual
quantizer/detokenizer tensors from the sealed teacher checkpoint on CPU, without
loading the full DiT. It derives the source latent from the captured audio codes,
using the authentic35847 padding code. `prepare` binds the resulting sealed
conditioning to each admitted teacher shard and packs feature0:384 as
source64/chunk-mask64/noisy64 in c*2+k order, with emission0:128 as velocity.
The actual teacher generated noise on CUDA: the converter consumes its sealed
x_t verbatim and checks the authentic scheduler and item-hash seeds. It never
substitutes a CPU noise approximation. Chunk masks are ones, with the teacher's
right padding set to zero. The final half-token of a313-frame target has a
channel mask and proportional row weight, so invented padding has zero gradient.

```bash
CUDA_VISIBLE_DEVICES='' python experiments/chimera_recovery/ace_cache.py condition \
  --cache /path/to/sealed/ace-velocity-cache --manifest-sha256 <sha256> \
  --items /path/to/ace_clips_stagec_d964.json --admission /path/to/admission/manifest.json \
  --comfy-root /opt/ComfyUI --weights /path/to/acestep_v1.5_turbo.safetensors \
  --out /path/to/new/ace-conditioning --limit 2

CUDA_VISIBLE_DEVICES='' python experiments/chimera_recovery/ace_cache.py prepare \
  --cache /path/to/sealed/ace-velocity-cache --manifest-sha256 <sha256> \
  --items /path/to/ace_clips_stagec_d964.json --admission /path/to/admission/manifest.json \
  --conditioning /path/to/ace-conditioning/manifest.json \
  --tokenizer /home/cereal/models/Qwen3.8-27B-Uncensored \
  --amendment-sha256 <sha256> --out /path/to/new/native-ace --limit 2
```

Teacher semantic codes derived from the clean clip are legitimate declared
conditioning for this D3b lane; the full clean latent never enters student
features. This prepares code-conditioned music denoising. Prompt-to-code
generation still requires D3c. H3/HY3D conversion needs their genuine synchronized
and geometry cache contracts; these adapters do not claim those lanes ready.

## Remaining native stages

`train_refit.py` implements D3a unit-only refit, including chart_in, the source
block and chart_out. The original D3a loss never connected chart_out. The new
outward-chart term requires independent detached host-write targets, exact D2
child recapture pins and a dated `teacher_delta_host_write` amendment; it refuses
to replace a missing target with its own current prediction. CPU tests verify
actual source-block/chart updates, missing-target refusal and exact resume.

`train_music.py` implements D3c full65536 music embedding/head host FP32 masters
and all-layer r16 LoRA. Each step clips the joint gradient and checks every
gradient before any host-table or LoRA update. Ledger/text rows remain frozen.
Checkpoint state includes per-row lazy Adam moments/counters, all-layer LoRA,
RNG and the one-pass sampler. `music_cache.py` converts the actual media_b v2
ACE-LM source: recorded conditional top64 logprobs plus `lse_full - lse_band`
restores code-band normalization; BF16 overmass is quarantined rather than
silently rescaled. Native parent chat tokens encode caption, lyrics, duration
and musical defaults. Only previous sampled codes enter the teacher-forced
input, using native code offset252416. Parent ACE text IDs are never reused.
The current partial ACE-LM cache must acquire its full immutable seal and a new
connected-group/complete-held-inventory admission snapshot before production.

`train_vision.py` implements D3d QVL unit, C11 input bridge and all-layer LoRA.
Assistant-only text KD has weight1 and packed block13 feature MSE has weight0.1.
The auxiliary gradient is attached to the actual native source block during
streamed recomputation, so it reaches the unit, earlier trunk LoRA and C11.
`vision_cache.py` reconstructs actual image bytes with the captured factor32
resize, PIL bicubic, normalization and duplicated-frame merge-window patch
order, then checks the exact sealed teacher grid, positions and feature rows.
The main7492-image ViT cache contains no `patch_flat` and no language answers.
Reconstructing pixels solves the first issue; an independently sealed authentic
assistant-answer teacher cache with exact native vocabulary alignment is still
required. The separate32-image reference cache is diagnostic and excluded from
training. Video inputs and audio-understanding need their own genuine aligned
source/cache contracts; single-image reconstruction does not prepare those.

The remaining-stage dispatcher has the same immutable clone/parity/prereg,
source-group, one-pass, initial checkpoint and30-minute rolling safety checks.
It binds its finite-skip window and decode-review policy to each engine
checkpoint, preserving them across exact resume. At25% of the budget it requires
a run-specific meaningful prompt-conditioned decode review. Final checkpoints
are unregistered and require the production checkpoint composer.

```bash
CUDA_VISIBLE_DEVICES='' python experiments/chimera_recovery/music_cache.py \
  --cache /path/to/fully-sealed/ace-lm-cache --manifest-sha256 <sha256> \
  --items /path/to/ace_prompts_stagec_d2m.json \
  --admission /path/to/new-complete-held-inventory-admission/manifest.json \
  --tokenizer /home/cereal/models/Qwen3.8-27B-Uncensored \
  --amendment-sha256 <sha256> --out /path/to/new/native-music

python experiments/chimera_recovery/train_stages.py preflight \
  --config /path/to/pinned-native-stage-config.json --out /path/to/preflight.json
```

The native-stage config schema is `chimera_recovery.native_stage_config.v1`,
with explicit stage D3a/D3c/D3d, objective, optimizer, seed, steps, one_pass,
safety, r16 LoRA, dtype/device/kernel, runtime assignment, immutable base,
hybrid parity/module and prereg pins. Stage-specific batch manifests use
`refit_batches.v1`, `music_batches.v1` or `vision_batches.v1`. Preflight executes
without Torch/model loading and returns blockers for absent or partial inputs.
No production training job was launched by this preparation.
