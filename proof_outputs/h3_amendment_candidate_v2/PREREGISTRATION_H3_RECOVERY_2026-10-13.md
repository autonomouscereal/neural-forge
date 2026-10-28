# PREREGISTRATION_OMNI_STAGE_D_v1 -- OMNI1 stage D (D1-D4), fixed before the first weight-changing step

Written 2026-10-13 by the trainer-prep lane. Binds `distill/DISTILL_DESIGN_2026-10-12.md` (section 2, 4, 5, 6) and `distill/STAGE_D_RULINGS_2026-10-12.md`; where the two differ, the rulings win (they are later and they are Cyril's).

**How this document is sealed.** The frozen body is every byte from the heading `## 1. Scope` up to and including the newline before the heading `## Amendments`. `distill/tools/stage2_training_receipt.py` (the CAMPAIGN section 1.8 copy of neural-forge's receipt module) pins that body's sha256 and byte length; every stage-D training receipt carries the pin, and a trainer given `--preregistration` refuses to start if the body no longer hashes to it. Changes after sealing go under `## Amendments` (dated, with the reason), never into the body.

## 1. Scope

- The student is OMNI1 = `OC02-F-Q38-27BU-OMNI1` (27,535,655,753 parameters; registered 2026-10-13T05:01:45Z, record `OC02-F-Q38-27BU-OMNI1-0001-register`, `3f38c5cf771fad29`; s7_omni1_check PASS). Stage D starts from its byte clone `OC02-F-Q38-27BU-OMNI1.clone-pre-D1` (ledger step S9), never from OMNI1 itself.
- Teachers: Qwen3.8-27B-Uncensored (the trunk parent, "Q38"), Kimi K3 (teacher-forced logits only), and the media parents (ACE-Step 1.5 DiT + LM `qwen_1.7b_ace15`, Qwen-Image FP8 + `qwen_2.5_vl_7b_fp8_scaled`, Hunyuan3D 2.1, MiniMax H3, the Qwen3-VL-32B ViT). Every teacher output is read from a sealed stage-C cache; no teacher runs inside a training loop except on-policy scoring, which is itself written as a cache first.
- Owner rulings that bind here (STAGE_D_RULINGS 2026-10-12): GO is standing; K3 gets the full 2.0M-token budget in four passes and is never skipped; H3 is included (2,048 targets unless the P4 measurement exceeds 20 s per forward, then 1,024); no depth cut (full-depth LoRA; tokens are cut only if the chosen engine measures below 60 tok/s); on-policy sampler upside path option (ii) Q8_0 proxy on GPU6, planning path option (i) Q4_K_M proxy on the P40s as a recorded transient exception to R3, fraction 0.25 after 20M off-policy tokens.
- Not in scope: stage T (the curated finetune) and the single Q4 export after it.

## 2. Rules common to every D stage

1. **Vocabulary mask.** Every text loss (D1 diagnostic KL, D2, D3d, D4) and every on-policy sample is computed over vocab rows **[0, 248,320)** only: the student's head is sliced to those rows before the log-softmax. Rows [248,320, 252,416) are the ledger bands and [252,416, 317,952) the C18 music band (64,000 written rows); no text loss reads or pushes them. The music band is trained only by D3c, against the ACE LM.
2. **Teacher targets.** Q38: top-64 log-probabilities (bf16) + int32 ids from `qtext_teacher_cache.py`, with the mass outside the top-64 as one tail bucket. K3: top-32 log-probabilities (`logit - lse` over all 163,840 K3 logits) from `k3_teacher_cache.py`, mapped to Qwen ids by IDENTICAL byte string into [0, 248,320) (`xtok`); a K3 id with no identical-byte Qwen token is id -1 and its mass joins the tail bucket with the out-of-top-32 mass; used ONLY at aligned byte boundaries (a Qwen position and a K3 position that have consumed the same bytes and both still have an in-row next token). A corpus file's in-band `real_lengths` bounds every target: padding is never a training position.
3. **Engines.** The text student runs on `streamed_kd_trainer.py` (>= 1.3.1). The upside engine (streamed bf16 fwd+bwd, GPU0 + GPU4 data parallel) is used if its gradient parity arm passes (done: "GRAD-PARITY PASS", worst relative difference 0.000e+00, receipt `distill/trainer-lane/receipts/final-v1.3.0-66b0854f/grad_parity_q35_08b_p40.json`) and S0 P2 measures at least 151 tok/s; otherwise the measured split cell (GPU6 + GPU4, fp16 LoRA, dynamic loss scale init 2^10). D1 runs on `d1_trunk_recovery.py` (>= 1.2.4) two-card data parallel on GPU0 + GPU4.
4. **Hardware.** No P2P anywhere (all reductions host-staged). GPU5 is never used. CPU jobs run at nice 15.
5. **Rolling safety checkpoints.** Every 30 minutes, with optimizer state, keep 2, on root (`--safety-minutes 30 --safety-keep 2`); resume is exact (sampler counters, optimizer, scaler) and refuses a changed base, data or config. D1's per-layer rung files are its rolling checkpoints (one layer is well under 30 minutes).
6. **Skip guard.** A run that keeps skipping steps (non-finite gradients, fp16 overflow) halts with a safety checkpoint and exit 3 after 8 consecutive skips or more than 20% of the last 100 steps, and waits for a ruling (design risk row 2).
7. **Ledger.** Each stage ends in a registered DELTA: `OC02-D-Q38-27BU-OMNI1D<n>` registered with `oc_checkpoint.py register --delta-base <previous clone>` (D1: base `OC02-F-Q38-27BU-OMNI1.clone-pre-D1`), then its byte clone `OMNI1D<n>.clone-pre-D<n+1>`, then the review package. D4 materialises the one merged checkpoint `OC02-D-Q38-27BU-OMNI1D4` and its clone `OMNI1D4.clone-pre-T1`. Parents, registered children and clones are never modified.
8. **Receipts.** Every training receipt records: this document's frozen-body sha256 (the pin), the base identity (directory digest or a delta-compose plan sha), every teacher cache's sealed manifest sha256, the data config sha256, every tool's version and sha256, the exact trainable census, per-step losses, and peak VRAM / host RSS.

## 3. Rungs

### D1 -- per-layer trunk recovery (prefix-fed)

- **Data:** `distill/corpus/text` rows [0, 977) (shards `text-00000` .. `text-00003`, packed 4,096, every row fully real): 977 x 4,096 = **4,001,792 tokens**, the shortest whole-row prefix of at least the design's 4M. The last 61 rows (rows // 16) are held for diagnostics; 916 rows (3,751,936 tokens) are trained.
- **Per text layer i, in order 0..63:** target y_p(i) = the parent layer applied to the PARENT prefix; prediction p = x_m + FFN_trunk(norm2(x_m)) with x_m = h_c(i) + attn_c,i(h_c(i)) from the CHILD prefix, trunk channels [0, 4,352) only. **The K3 term is excluded from the prediction** and kept in the child's prefix advance h_c(i+1) = x_m + FFN_full(norm2(x_m)).
- **Loss (weight 1.0, the only term):** mean over tokens of ||p_t - y_t|| / ||y_t|| + (1 - cos(p_t, y_t)) (`--loss-site residual`).
- **Trainable (one layer at a time):** fused `ffn.up.weight` gate rows [0, 4,352) and up rows [16,640, 20,992), `ffn.down.weight` columns [0, 4,352), `norm2.weight`: 3 x 5,120 x 4,352 + 5,120 = 66,851,840 per layer, **4,278,517,760** over 64 layers. Frozen: attention / KDA, the K3 rows and columns [4,352, 16,640), embed, head, every media tensor.
- **Optimiser (d1 defaults, pinned):** Adam beta (0.9, 0.95), eps 1e-8, lr 2e-5, 10 warm-up steps, cosine to 0.1 x lr, clip 1.0, 2 epochs, 32,768 tokens per step (micro 8,192), fp32 master and training matmuls, bf16 frozen forwards and host streams, seed 0.
- **Diagnostics (never refusals):** per layer on held tokens, before and after: residual relL2 / cos / D1 loss of the trunk-only prediction and of the full layer (with K3) vs y_p(i); FFN-level relL2; K3-term energy relative to the trunk FFN output; the loss curve. After L63: frozen-head KL(parent || child) and top-1 agreement over rows [0, 248,320), for the D1 child and the original child.
- **Output:** `D1_delta.safetensors` (bf16, slice replaces) converted to the ledger recipe `DELTA.json` (op `replace`) and registered as `OC02-D-Q38-27BU-OMNI1D1` on base `OC02-F-Q38-27BU-OMNI1.clone-pre-D1`; clone `OMNI1D1.clone-pre-D2`.

### D2 -- end-to-end LoRA KD, text + K3

- **Data (23.0M short + 1.0M long):** 18.85M registry off-policy tokens (`distill/corpus/text`), 1.15M on-policy (sampler per section 1; if no sampler is ready when D2 reaches its on-policy point, the 1.15M come from the off-policy registry and that is recorded), 1.0M self-generated tokens (sampled by the parent Q4_K_M, sealed `fe1f0784777b8fdf`, 1,008,982 tokens, rescored in bf16), and the 2.0M K3 rows (`distill/corpus/k3`, four passes, 1,991,482 real Qwen tokens, 1,867,899 aligned pairs). Then the long rung: `distill/corpus/long`, 31 documents x 32,768 tokens, KL to Q38 at L = 32,768.
- **Rungs (adapter snapshots):** 5M / 12M / 20M / 23.0M tokens, then the long rung last.
- **Loss, per next-token position p, summed and divided by the step's global position count:**
  - text, self-gen and long rows: w_q38 = **1.0** x KL(Q38 top-64 + tail || student);
  - K3 rows: w_q38 = **lambda_K3 x a** on every stored position and w_k3 = **lambda_K3 x (1 - a)** on aligned positions, with the Q38 anchor **a = 0.5** and **lambda_K3 = 0.3** at the start (G3 fallback: 0.1 with the gain frozen at 1.0 until the 12M rung). w_k3 multiplies KL(K3 top-32 mapped + tail || student);
  - on-policy rows: generalised JSD, beta 0.5, over the same buckets instead of the forward KL;
  - LM cross-entropy floor **0.1** on every position whose next token is a real in-row token;
  - EOS supervision **0.1** (extra CE) where that next token is an EOS id (248046, 248044);
  - unlikelihood **0.1** x sum over up to 8 tokens that would repeat a 3-gram already in the row (the gold token is never a candidate).
- **Trainable (LoRA sites and ranks):** trunk FFN gate / up / down r64 (alpha 64) on the trunk channels [0, 4,352) only = 116,391,936; GDN `linear_attn.in_proj_qkv` and `linear_attn.out_proj` r64; gated-GQA `self_attn.v_proj` and `self_attn.o_proj` r64 (design upper bound for the attention sites: 123,731,968); K3 down columns [4,352, 16,640) LoRA r32 (alpha 32) = 35,651,584; K3 per-column gain, 12,288 per layer = 786,432, **fp32, initialised to exactly 1.0**, applied to the K3 channel activation (equivalently a per-column scale of the K3 down columns). The exact total is the trainer's census at launch and is recorded. Frozen: GQA q/k, RoPE / partial RoPE, KDA decay (`A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`), `in_proj_z`, every norm, embed, head, all media tensors.
- **Optimiser (pinned):** AdamW beta (0.9, 0.95), eps 1e-8, weight decay 0, LoRA lr 1e-4, gain lr 1e-3, 20 warm-up steps, cosine to 0.1 x lr, grad clip 1.0; fp16 split cell only: dynamic loss scale init 2^10, growth interval 200. Seed 20261012.
- **Output:** `OMNI1D2` delta (ops `lora` + `scale`, base digests pinned) on `OMNI1D1.clone-pre-D2`; clone `OMNI1D2.clone-pre-D3`.

### D3 -- media + vision-in / video-in distillation (hybrid module)

- Lanes in order ACE, Qwen-Image, Hunyuan3D, H3 (H3 included by ruling). Engine: the hybrid module (qwen3_5 layers + the Chimera unit at depth d + interface rows + flow head); **it must pass its parity arm (hybrid forward vs the Chimera charted forward on OMNI1 for each unit, zero writes on rows a unit does not own) before any D3 step.**
- **D3a** per-unit refit on host rows recaptured on the D2 child: ACE 102,811,904 + H3 267,889,176 + QIMG 154,985,727 + HY3D 46,156,031 + QVL32 19,594,291 = 591,437,129 parameters.
- **D3b** guided-velocity KD against the cached teacher velocities, backprop through layers d..63 above each unit, x_d cached; trainable the unit + interface-out rows C15 f[0, 640) + LoRA r16 on layers d..63. Loss: mean squared error of the guided velocity, weight 1.0 per lane.
- **D3c** C18 music band rows [252,416, 317,952) in embed + head (2 x 65,536 x 5,120 = 671,088,640, sparse-row lazy Adam on the host) + LoRA r16 on all layers; loss KL(ACE LM top-64 + tail || student band rows), weight 1.0.
- **D3d** vision-in and video-in answer KD through the QVL32 unit + C11 rows (1,536 x 5,120) + the shared LoRA r16; text loss over rows [0, 248,320) as in D2 (weight 1.0); QVL32 feature KD against ViT block 13 at weight **0.1**.
- **Stop-spending line:** a modality whose decodes are still noise after 25% of its D3 budget is paused and reported, not continued.
- Data, token counts and teacher caches as DISTILL_DESIGN section 3 and 4 (D3 row); LTX-rollout v9_base rows excluded. Output: `OMNI1D3` delta (each unit's ratios carried, R2b) -> `OMNI1D3.clone-pre-D4`.

### D4 -- consolidation replay and the one merge

- Replay from existing caches only: 3M Q38 text (including 0.5M long), 1M K3 rows, 300 pairs each for vision, ACE, QIMG, HY3D and H3; text losses over [0, 248,320) with the D2 weights. All D2 + D3 trainables at **lr x 0.3**.
- Then an exact fp32 merge of every LoRA into its base tensor, and the K3 gain folded **once, here, in fp32, cast once**: `down[:, 4352:] <- (down[:, 4352:] + s_k3 * B_k3 @ A_k3) * gain` (after the K3 LoRA merge). Output: the merged `OC02-D-Q38-27BU-OMNI1D4` (root) and its clone `OMNI1D4.clone-pre-T1`.

## 4. Gates: functional vs diagnostic

Owner policy 2026-10-06: finite fit numbers (relL2, KL, NLL, cos, PSNR, velocity MSE) are **diagnostics**, printed beside every gate, never a refusal. Structural, identity and non-finite failures are **functional** and stop the stage.

| Gate | Kind | What is checked | On failure |
|---|---|---|---|
| G0 structural | **functional** | OMNI1 registered and its MANIFEST verifies; loads strict in Chimera; one forward finite; `OMNI1.clone-pre-D1` exists and its digest equals OMNI1's; S0-P1 reads every text tensor into the qwen3_5 layout tensor-exact (uniform FFN 16,640, no gain tensor) | No D spend; fix step 1 (S7 install / clone) |
| G1 text identity | **functional** | R-Q top-1 OMNI1 vs S5 on text-only inputs (expected about 1.0: charted blocks write exact zeros off their rows) ; new failure classes on bank v1 | Below about 0.99, or loops on lanes where S5 had none: diagnose S7 before D1 |
| G2 text start point | diagnostic (recorded) | bank v1 content lanes, artifact-tolerant; R-Q top-1 vs parent | Recorded and read by Cyril; GO is standing (ruling) |
| G3 K3 | diagnostic (recorded) | bank v1 and R-Q with the K3 columns on and zeroed; per-route bits-per-byte probe | Sets lambda_K3 and the route mix (rule in D2); K3 is never skipped |
| G4 media, per modality | structural part **functional**, content diagnostic | a structurally valid decode (audio rms > 0, non-constant pixels, mesh faces > 0, frames > 0, vision answers present, QVL32 features non-zero) | Undecodable: that lane returns to the S7 fix, the others proceed. Noise content is the expected S8 state and is what D3 fixes |
| In-run | **functional** | non-finite loss / gradient / stream; persistent skips (section 2.6); identity drift on resume; a target outside its row or outside [0, 248,320) | Halt with a checkpoint; ruling |

## 5. Promotion between stages (ruling of 2026-10-12)

A stage starts from the previous stage's clone when, against that previous stage: bank v1 content (artifact-tolerant; loops, echo, missing EOS and cap-rides do not count, only content misses do) is **at least** the previous stage's, and **no modality regressed in content**. Otherwise the next stage starts from the earlier clone (CP R2) and the regression is reported. Cyril's read of each package is asynchronous: training does not wait for it. Per-stage success readings (recorded, not gates): D1 bank v1 content >= OMNI1's S8 and R-Q vs parent >= S8; D2 bank v1 >= D1, R-Q >= D1, bank v2 vision lanes not regressed, 256k needle read >= the P11 baseline, K3-zeroed ablation reported both ways; D3 decodes beside the parent's on the same prompt, content no worse than S8 on any modality, text bank v1 >= D2; D4 everything side by side (parent / OMNI1 / D4).

## 6. What may change without an amendment

Paths, card assignments inside the rules of section 2.4, micro-batch / accumulation splits that keep the global tokens per step, and tool patch versions whose tests pass. Everything else in sections 1-5 (data, losses and weights, masks, trainable sites and ranks, optimiser values, rungs, gates, promotion) changes only by a dated amendment below, written before the step it affects.

## Amendments

(none)


# Chimera Omni recovery preparation amendment — 2026-10-13

User direction: prepare recovery training for tool calling, intelligence and all
modalities first. Audit prior folding methods in parallel and backlog changes
while recovering the existing checkpoint. This amendment is a
candidate for the stage-D preregistration. Append it below `## Amendments`
before an affected production step. The frozen 15,339-byte body with SHA-256
`c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370`
must remain byte-identical. Record both the frozen-body digest and the amended
whole-document digest in every run identity and initial receipt.

## A. Corpus integrity and semantic coverage

Keep the sealed D1 input unchanged. For subsequent recovery passes, prepare
`recovery_corpus.v1` rows from the verified registry staging snapshot and
read-only PostgreSQL memory archives. Bind the exact ready-row manifest,
source file digests, original shard digests, source identifiers, collection
dates and split assignment in each cache/run receipt. Unknown original author
dates remain unknown; collection dates are not original author dates.

Tool supervision must retain actual schemas, function names, JSON arguments,
ordered assistant calls and matched results. A final assistant call without a
result may teach call construction and must be labeled as outcome-unobserved.
Do not invent executed tool results. Unambiguous source name-based references
may be canonicalized to deterministic call IDs with that transformation named
in provenance. BFCL acceptable argument alternatives may be selected only from
the recorded target alternatives and validated against the recorded schema.
Missing schemas, context, ambiguous links and redacted targets are quarantine
conditions. Preserve multi-turn, parallel, failure and retry evidence only
when the source explicitly demonstrates it.

Related sessions, source documents, exact and near duplicates belong to one
split. Exclude frozen evaluation sources and prompts before cache creation.
Recompute grouping across merged tranches rather than trusting each tranche's
split independently. Reserve fresh held-out tasks for arithmetic, reasoning,
code, instruction following, tool schemas/calls/results/retries/parallel and
each modality. Never report a coverage label as a behavioral result.

## B. Assistant supervision and teacher caches

The added structured-chat recovery lane uses the actual parent chat template,
thinking off, and authentic `tools` input. Preserve tokens for user/schema/tool
context as input but supervise assistant-owned targets only. Shift ownership
to the next-token target: `assistant_mask[:, 1:]`, intersecting real lengths.
Prompt, tool-result and padding targets have zero CE, EOS, Qwen KD, K3 KD,
unlikelihood and JSD weights. Keep text predictions within `[0, 248320)`.
Never train the music/ledger bands on text examples.

The initial structured tool recovery pass sets `w_ul=0` (`--w-ul 0`). Repeated JSON
keys, function names and schema delimiters are legitimate syntax; generic
repeat-token unlikelihood would penalize correct tool calls. Preserve the
original unlikelihood setting for its original non-tool replay lane. A later
syntax-aware tool unlikelihood objective requires a separate dated amendment
and held-out tool validation.

Caches for this lane must be generated from its new packed token IDs and bound
to their SHA-256; flattened-text caches cannot be reused by label or shape.
Apply the existing D2 trainable sites and optimizer contract. The initial
structured-chat pass is one deterministic pass over accepted training rows
(seed 20261013), with its actual token budget calculated and sealed after
packing. No upsampling may fabricate coverage of absent capabilities. Replay
the existing disjoint text/long/K3 sets in D4 under their original contract.

The structured lane removes right padding per micro-batch without truncating
real context and divides summed objective losses by assistant prediction
positions across the entire accumulation window. It supports one replica
initially; the original replay lane retains its original normalization.
Seal the exclusive row-plan digest and cursor, actual real/compute/supervised
token counts, trainer/helper digests and the full amended document in resume
identity. Record skipped positions separately from applied updates.

## C. Modality repair requirements

Image generation and editing need prompt/conditioning/target provenance and
genuine target images or sealed teacher trajectories; editing additionally
needs explicit source/target roles and a task instruction. Vision and video
understanding require real input media and grounded answer supervision.
Audio/music and speech understanding must have genuine audio references and
appropriate targets; route names alone do not supply them. Joint AV requires
video and audio from the same source interval with timestamps and alignment.
The current H3 deterministic assignment of unrelated audio is excluded from
synchronized AV training. Existing independent video/audio reconstruction data
may be evaluated separately and must not be claimed as synchronized AV.

3D supervision needs geometry or a sealed geometry-teacher target, meaningful
object conditioning, coordinate/normalization metadata and target provenance.
Charts and arbitrary icons without geometry are excluded from claims of 3D
recovery. A teacher-conditioned image-to-3D sample is distinct from a ground
truth geometry sample; record that distinction.

For denoising, clean targets are available to the loss only. They cannot be
copied into ordinary conditioning features. Legitimate inpainting/editing
context is explicitly role/mask limited. D3b keeps the preregistered
guided-velocity MSE objective: production adapters must explicitly select
the reviewed guided-velocity callback. The prepared D3 patch adds
`--flow-objective guided_velocity --objective-adapter /path/to/training.py
--noisy-feature-bands /path/to/native_noisy_feature_bands.json`
to its rate proof; the production driver must bind this same objective to
sealed batches. Endpoint/x0 smoke tests do not establish this objective.
The first D3b recovery pass uses explicitly amended AdamW settings: unit and
interface-out learning rates `2e-5`, LoRA learning rate `1e-4`, betas
`(0.9, 0.95)`, epsilon `1e-8`, weight decay `0`, gradient clip `1`, 20 warm-up
steps and cosine decay to `0.1` of each starting rate, seed `20261013`.
These are initial recovery choices adapted from the adjacent D1/D2 contracts,
not measured optimum rates. Seal one pass over the accepted training batch
manifest as the step budget; preserve the 25% decode review and stop-spending
rule. Holdout batches never enter the optimizer sampler.
Require native/hybrid forward and gradient parity, decoder contracts, finite
targets, actual adapter/driver availability and parameter census before the
affected route trains. Unready routes have executable prerequisite jobs and
explicit blockers, never a `ready` verdict from metadata-only checks.

## D. Receipts, execution and promotion

D1 production launcher passes `--preregistration` into the trainer itself.
The trainer validates before opening checkpoints, includes the observed
document pin in the resume identity, and seals the initial receipt before the
first weight update. Existing resource allocator, disk/RAM floors, clone
protection and exact optimizer resume remain required.

Preparation and CPU contract tests do not launch recovery training or prove
model recovery. Keep original checkpoints and active queues untouched while
preparing private candidates. Publish a route-by-route readiness report with
verified data/caches, code versions, loss objectives, trainable sites, held-out
evaluation and remaining external prerequisites. Preserve stage promotion's
existing content/no-regression rule. Folding changes wait until recovery of
the current model; the historical objective/adoption audit proceeds in parallel.



### 2026-10-13: D1 source-isolated successor after measured source-span audit

This dated correction supersedes the earlier amendment's instruction to keep the sealed D1 input as the active training input. The original sealed prefix and original frozen body are preserved as evidence. The measured original fit prefix includes 1,498,741 tokens across 3,010 documents incompatible with current recovery splits/quality checks, including 371,547 test and 351,492 validation tokens; one document crosses its fit/diagnostic boundary. There were zero original frozen-carve assignment violations. New-split overlap is not retroactive proof the original frozen evaluation leaked; structural answer quarantine is also distinct from hidden-state regression target correctness.

Use only verified complete parent-tokenized admitted train conversations for D1 optimizer positions and admitted validation conversations for diagnostics. Exclude current global cross-route/frozen-source groups from both pools, and prove fit/diagnostic source groups disjoint. No test row enters either pool. Select whole conversations in existing source order until at least 3,751,936 fit and 249,856 diagnostic real tokens; pack the two streams separately at width4,096 with explicit real lengths and an explicit --held-rows count. D1 retains all-real-position prefix-fed hidden-state regression and the original optimizer, layer objectives and immutable base. Whole-document overshoot and final-row padding are reported below; padding is never a training target or claimed real budget. This replaces the exact977-row budget where whole conversations require additional rows. Pin this new dataset/config and amendment before the first update; the original-prefix extractor must not overwrite or substitute this input. Preparation does not launch or establish recovery.

```json
{
  "actual_counts": {
    "diagnostic_documents": 331,
    "diagnostic_real_tokens": 250280,
    "diagnostic_real_tokens_by_route": {
      "text_code": 149556,
      "text_instruction": 47845,
      "text_reasoning": 52879
    },
    "fit_documents": 5855,
    "fit_real_tokens": 3752131,
    "fit_real_tokens_by_route": {
      "text_code": 1111423,
      "text_instruction": 2238591,
      "text_reasoning": 402117
    },
    "fit_rows": 917,
    "held_rows": 62,
    "original_real_budget": 4001792,
    "padded_positions": 7573,
    "total_real_tokens": 4002411,
    "total_rows": 979,
    "whole_document_real_overshoot": 619,
    "width": 4096
  },
  "dataset": {
    "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/d1-source-successor-v2/d1_source_ids.safetensors",
    "sha256": "74f79199297891af208bfcb340a99d6bc846a2337ce08a9599f83bc11f2892ad"
  },
  "source_audit": {
    "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/d1-source-audit-v2/report.json",
    "sha256": "abf51e7283a486e76977f728abb36a7cc77c030033d6c4930381005f0cd07a32"
  },
  "sources": {
    "cross_route_exclusions": {
      "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/d1-admission-v3/cross_route_training_exclusions.jsonl",
      "sha256": "084a7047ee0983d5c4b5dcaf82e041cb3eb1f7107686e2efe0be3c9ae3190fcc"
    },
    "global_admission": {
      "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/d1-admission-v3/manifest.json",
      "sha256": "5b608482942585a66e6b781f16efc431d3a31a1da06b8012f26ebc5cb173fd1d"
    },
    "original_preregistration": {
      "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/preregistration-v1/PREREGISTRATION_OMNI_STAGE_D_v1.md",
      "sha256": "fee68ed636c1455cb02636610029b25ecb7299fe765d0ec9b7d0d550d927e956"
    },
    "train_manifest": {
      "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/native-text-v3-train/manifest.json",
      "sha256": "1ab5ddfc6223eea1321d7122f526018176c0c45862c5732ab96a588050b469ca"
    },
    "train_records": {
      "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/corpus-schema-v2/train.jsonl",
      "sha256": "8ea2050350629783ea5a8d90545de9cbe21db8ac378377c5f9c3223b1fe46a4d"
    },
    "train_shard": {
      "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/native-text-v3-train/assistant_text.safetensors",
      "sha256": "8d815f32b8d4dc82890dd5b9e2e4bcd99dbe9d2d337604f3a5f6f787f49ff107"
    },
    "validation_manifest": {
      "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/native-text-v3-validation/manifest.json",
      "sha256": "5aee591bb50be43e366ce8f8b9b064d2fc213ac4559353b4d36da4d4f58d7f68"
    },
    "validation_records": {
      "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/corpus-schema-v2/validation.jsonl",
      "sha256": "ffba4ea37346b1fa26a7816fa356e16f711f118b6364e4738a5caa0d431b36c4"
    },
    "validation_shard": {
      "path": "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/native-text-v3-validation/assistant_text.safetensors",
      "sha256": "ac82dfadba84ad01989a87b0f9cfc84922253589aa5fce816176c52900190966"
    }
  }
}
```


### 2026-10-13: selected audio observation and assistant-CE recovery amendments

Root integration selects the following exact dated candidate wording under the user-authorized recovery task. Source file hashes are pinned below. Earlier frozen body and existing amendments remain byte-preserved after newline normalization. This selects preparation and training contracts; it grants no native parity, source admission, runtime allocation or trained-model capability. Each must be established by its own actual receipt.

# Standalone clean-audio observation chart, 2026-10-13

This is a new, explicit observation-coordinate convention for recovery of the existing native H3 audio unit. It does not reproduce a proven standalone native-parent PackedLayout. It does not change the existing joint AV cache seals, Stage D preregistration body, pretrained parameters, or folding weights. Its approval and exact file hash must be bound by the parent to the separate audio CE admission and configuration.

The authentic CPU encoder proof at `reviews/2026-10-13-folding/recovery-preparation/h3-audio-proof-v1/probe.json` has SHA256 `caf6961a9d3c99c77326159d48c7b10db79318b79668c1e29e8cb370c0e6c263`. It measured a complete one-second 16k mono source resampled to 32k stereo, 32,000 samples, raw latents `[1,32,2,40]`, channel-major `[80,32]`, bitwise repeat encoding and packing inversion, zero waveform/latent padding or trimming, and full `[0,1]` PTS coverage. The measured nominal hop is 800 samples / .025 seconds. This establishes an encoder lattice, not its convolution/attention receptive-field localization or native rotary XYZ. The v1 proof explicitly does not establish parent model-space parity; production validation requires the successor actual model-space receipt and its tensor seals.

The reviewed `chimera_omni/modeling/charted.py` source has raw Windows SHA256 `0ee1875895a192cc4ce0dcd469a7cc2e9014b29b68ed3d8c4be978f24828ddc3`; identical LF text SHA256 `eaf590abf619e5dafab38b0d940d27dc79ef1d8a3eceb4db72843162fb2391b6`. Its `h3_plain_source_layout` requires video dimensions and derives audio channel X from that video grid (lines 56–92). Its H3 source block supports audio-only observations with supplied coordinates and the clean sigma-zero schedule (lines 428–454). Supplied coordinates affect actual 3D rotary angles (169–183); shape and finite checks alone do not establish parity.

For this one-second, forty-frame scope, channel-major row `r=c*40+i` carries physical metadata `(center_seconds, y, hop_seconds, channel) = ((i+.5)/40, 0, 1/40, c)`. Stereo channel `c` is categorical 0/1 and is not a measured spatial X coordinate. The distinct new source rotary convention is `(prompt_text_count+i, 0, c)`. Prompt and answer text have `(text_ordinal,0,0)` scattered to actual full-sequence slots; only prompt text and clean audio may enter joint H3 attention. Reference-answer text is excluded by the already separate `h3_prompt_and_clean_observation_only` read policy. Media never depends on the number, contents, or positions of reference-answer tokens.

The adapter `h3_audio_chart.py` returns detached-data lists suitable for conversion to FP32: `native_media_positions[1,80,4]`, `charted_source_positions[1,S,3]`, and `h3_prompt_read_mask[1,S]`. It requires exact receipt/amendment/source SHA pins, complete unique sequence-slot coverage, actual channel-major order, and every observation before the first reference-answer slot. It keeps the authentic shared H3 spec types `{2,3}` while supplying only type3 observations and sigma0; it creates no dummy video rows. Its contract deliberately keeps `native_parent_xyz_parity=false` and does not independently grant corpus admission.

Validation consists of actual measured encoder/model-space seals per source, full-sequence scatter and nominal coverage, independent rotary-angle comparison against the exact genuine `ChartedAttention` implementation, and identical earlier attention outputs plus zero answer-input gradient when excluded answer contents/positions change. Parent full-trunk AudioCE tests remain responsible for prompt-prefix logits, optimizer ownership, text-only behavior, and frozen-evaluation group isolation. CPU tests use genuine source attention, not a reimplemented substitute. No GPU or training job is launched by this amendment.

The actual successor model-space proof at `h3-audio-proof-v2/probe.json` has SHA256 `c54f81005456a75b9caa40aac520520b2c541642e3eec3e0506646016ecaf52e`; the adapter's strict proof validator passed against those actual bytes. This upgrades the one-second encoding/model-space evidence, while leaving the new rotary convention explicitly separate from parent parity.

The dated duration extension accepts arbitrary positive measured `T` up to 2048 only when `encoder_proof` validates its own complete `[1,32,2,T]`, packed `[2T,32]`, decoded `[1,2,800T]`, 32k waveform, exact `T/40` PTS duration and full nominal intervals, no internal padding/trimming, bitwise repeat/inverse packing and verified parent model space. A one-second receipt cannot qualify another duration. For source-preserving explicit right-tail padding, the original and derived asset must remain separate immutable records. The derived file must preserve the entire original PCM prefix, append only the minimal known zeros to the 25ms lattice (600 samples at 24k), and include padding count, original duration, waveform-prefix hashes and correspondence in its corpus admission. The encoder must then process that complete derived file without hidden padding or trimming. Frame validity and original-versus-derived time coverage are governed by the caller's sealed admission, not inferred by this chart. Merely specifying `T` does not fabricate a new measurement.


# Candidate amendment: D3d_audio_ce (2026-10-13)

This is a reviewable additive candidate, not the frozen preregistration document.
Preserve the D1–D4 body digest
`c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370`
and its newline-inclusive15,339 bytes.
The root integration must append the accepted exact wording and pin its new
document and amendment digests before any production training.

After vision/video D3d and before D4, introduce **D3d_audio_ce**, objective
`authored_audio_assistant_ce`, weight1.0, text vocabulary[0,248320). This distinct
supervised lane uses actual audio observations and verified authored assistant
answers. It uses no invented audio-teacher logits, Qwen VL audio path, ViT
feature target or foreign speech-codec mapping. It makes no ASR or general sound
capability claim. The initial24 controlled non-speech fixtures have8 train,
8 validation and8 test rows; only the eight admitted training rows enter the
initial one-pass budget. Additional verified speech sources require separately
pinned admission and a new immutable source/budget plan.

Preserve the actual parent chat template and every native text ID. Insert the
native media placeholders inside the user's turn immediately before the native
user-closing EOS. The exact native assistant generation-prefix anchor owns only
the assistant body and completion EOS; prompt, empty-thinking prefix, media and
right padding carry no loss. Normalize CE by the actual supervised positions.
This insertion policy is `native_audio_before_user_close_v1` and must be tested
with the actual parent tokenizer before batch sealing.

Clean observations use the actual source-pinned H3 FP32 audio VAE, full source
WAV coverage, genuine libav stereo32k resampling,800-sample hops, raw
[1,32,2,T] latents and channel-major[2T,32] packing. No waveform or latent trim,
padding or silence fallback is allowed. The actual H3 unit-scale raw/model
transform is source-pinned and checked bitwise. Only native input feature
columns480:512 receive those clean model-space values; sigma is exactly0.
Keep the existing H3 unit's supported type set{2,3}, while this lane supplies
only type3 observations and no fabricated video rows.

Select a **new standalone observation chart**, not parent standalone-XYZ parity.
For stereo branch c∈{0,1}, frame i and measured hop h=800/32000=.025s, native
bridge metadata is `((i+.5)*h,0,h,c)`; the final component is categorical stereo
identity, not a measured spatial coordinate. H3 rotary coordinates are
`(prompt_text_count+i,0,c)`. Actual prompt text receives its ordinal
`(text_ordinal,0,0)` coordinate; reference-answer coordinates are hidden from
H3 by the read policy. All arrays must bind the exact encoded source and actual
native text/media scatter. This chart removes the old dependency on a fabricated
video grid; it is an explicitly amended observation policy requiring source
ownership, angle/scatter and prefix/answer-invariance tests. No equivalence to
the parent's video-relative stereo coordinates is claimed.

The H3 attention policy is `h3_prompt_and_clean_observation_only`: only prompt
text and clean observed audio enter that unit's joint attention. Every reference
assistant-answer row is hidden from H3, preventing future-answer → audio →
earlier-answer leakage. The shared trunk retains its real causal/full-depth
forward and backward. Train only the H3 unit, type3 input bridge columns480:512
and type embedding row3, plus all shared trunk r16/alpha16 adapters. Keep the
other native units, text embedding/head and generative emission frozen.

Initial explicit optimizer: AdamW, H3 unit_lr2e-5, type3 bridge_lr2e-5,
shared LoRA_lr1e-4, betas(.9,.95), eps1e-8, weight_decay0, global grad clip1.0,
20 warmup steps, cosine to minimum ratio.1; FP32 trainable masters, explicit
BF16 or FP32 frozen forwards, seed20261013. These are the initial recovery
media optimizer values and D2's shared-adapter rate, not an optimum claim.
The initial budget is exactly one successful update per accepted training row,
with deterministic source order, no repeated holdouts and exact Adam/RNG/sampler
resume. A source addition changes the pinned one-pass budget and identity.

Require actual native/hybrid parity on the current immutable checkpoint, the
source-pinned full clean-encoder proof, actual parent-tokenizer insertion proof,
globally held-excluded source admission, finite full-depth gradients and
nonempty optimizer ownership before the production entry point opens. Transfer
the cumulative native state and complete shared-r16 bank from the preceding
vision D3d stage before creating the optimizer. Resume the same stage only with
its exact checkpoint identity, optimizer and sampler; stage transition carries
parameters but starts its own optimizer. Export complete replacement shared-r16
state and cumulative native overlays to D4; never sum sequential adapter banks.

Retain30-minute rolling checkpoints with optimizer/RNG/sampler, keep2; halt on
8 consecutive nonfinite skips or more than20 skips in the last100 attempts.
At25% of the initial budget, stop for a pinned actual free-answer review on
held audio fixtures. Record parsed events/count/onsets and transcript metrics
only where genuine speech labels exist. Teacher-forced CE and waveform SNR are
diagnostics and do not establish understanding. D4 replay and learning-rate
multiplier.3 remain as frozen, with this lane's replay inclusion explicitly
pinned in the consolidation source plan.


```json
[
  {
    "path": "C:\\Users\\cereal\\Documents\\ai training\\.agent-worktrees\\neural-forge-a495ffdd2f\\workers\\codex\\chimera-training-recovery-oct13-eae1b72e2c3f\\experiments\\chimera_recovery\\H3_AUDIO_OBSERVATION_CHART_AMENDMENT_2026-10-13.md",
    "sha256": "f27c86a0c3f2b323721e22bf68a787010275cd4c28522295e1c10974f4342248"
  },
  {
    "path": "C:\\Users\\cereal\\Documents\\ai training\\.agent-worktrees\\neural-forge-a495ffdd2f\\workers\\codex\\chimera-training-recovery-oct13-eae1b72e2c3f\\experiments\\chimera_recovery\\AUDIO_CE_AMENDMENT_2026-10-13.md",
    "sha256": "6ec949fe506a49d1b358e0d1f03c083e11a83db0dfaccbc04e2d2580460fd9f3"
  }
]
```


### Selected amendment 2026-10-13: waveform-supervised speaking recovery

Selected before recovery training under the user-authorized existing-model recovery scope. This amendment defines the speech objective and D4 replay addition; it is not evidence of trained capability. Original frozen body and earlier amendment bytes are unchanged.

The added stage is `D3d_speech_flow`, after `D3d_audio_ce` and before `D4`.
Understanding uses authored assistant CE separately. Speaking uses
`supervised_speech_rectified_flow` from genuine waveform latents, rather than
guided H3 teacher-trajectory KD or foreign discrete speech codec IDs.

Source rows remain the unchanged `speech_understand` records from a complete
sealed speech source/admission snapshot. Each accepted target must preserve
the exact official synthetic voice (Ryan, Aiden, Vivian or Serena), full authored
transcript, actual independent exact ASR match, original TTS waveform SHA,
minimal explicitly recorded PCM zero tail, full derived waveform SHA and all
source/plan/teacher/ASR provenance. No user reference voice is used. Exact ASR
does not stand in for human listening; that limitation remains in receipts.
Diagnostic probe row003 (17 times6) is excluded for frozen S8 overlap. The
complete combined training/held source, frozen registry/eval/prior exclusions
and connected speaker/utterance groups must be traversed by the admission
producer. Held rows are encoded only for diagnostics and never admitted.

The actual H3 encoder, model-space transform and source pins from
`h3_audio_probe.py` apply. Every source requires its own full-coverage measured
proof:32kHz stereo,800 samples per nominal frame, raw/model latent
[1,32,2,T], channel-major packed[2T,32], no internal trim/pad, bitwise repeat
encode and verified raw/model/inverse transform. A non-default T requires its
own actual proof; no40-frame assumption is used for speech.

The condition is exactly
`Generate speech in the synthetic {speaker} voice, saying exactly:\n{transcript}`
as one ordinary user message in the pinned parent chat template with
enable_thinking=false and an authentic assistant generation prefix. Its full
rendered/native token IDs must agree. Append noisy type3 placeholder rows after
that complete prefix, using
`native_noisy_audio_after_assistant_generation_prefix_v1`. There is no reference
assistant answer, no codec ID and no clean-latent copy in the model inputs.
The transcript is a legitimate TTS condition and is pinned as such.

Select `audio40hz_generation_prompt_offset_categorical_stereo_v1` and
`h3_generation_prompt_and_noisy_audio_only`. The measured nominal physical
bridge chart is[(i+.5)*.025,0,.025,c]; the separately selected rotary chart is
[prompt_text_count+i,0,c], where c is categorical stereo0/1. Text positions are
[text_ordinal,0,0]. H3 spec types remain[2,3], actual rows are only type3, and
the bounded H3 read policy includes just the generation prefix and noisy audio.
This is an explicit new generation chart derived from proved source-lattice
mathematics; native parent XYZ parity is false and must not be claimed.

Select `speech_sigma_grid=[0.05,0.3,0.7,1.0]`, seed20261013 and the deterministic
`source_row_sigma_sha256_first63bits_v1` noise policy. For each admitted source
and each sigma, sample FP32 epsilon from the sealed seed and form
x_t=(1-sigma)*x0+sigma*epsilon. Only x_t enters the native feature band480:512;
only x0 supervises emission band224:256. Target/noise tensors are detached,
finite and source-reconstructed bitwise. The student type3 emission is explicitly
interpreted as an x0 endpoint for this new supervised objective. Canonical
velocity=(x_t-x0_prediction)/sigma is matched by per-channel MSE to
(x_t-x0_target)/sigma=epsilon-x0_target. Weight1, per-row/channel mean, full-depth
gradient propagation through the shared text trunk. Positive sigma is mandatory.
The sigma weighting is intentional; this is not a declaration that the initial
S8 head already has these endpoint semantics or has learned to speak.

Train the H3 unit, type3 bridge/input embedding slices, type3 emission slices
and shared all-layer r16/alpha16 bank. Freeze text head, other native units,
other bridge/emission bands, music tables, K3 sites and gain. FP32 trainable
masters; unit/bridge/emission learning rates2e-5, shared-r16 learning rate1e-4;
AdamW beta(.9,.95),eps1e-8,weight decay0,clip1,warmup20 successful steps,
cosine minimum ratio.1,seed20261013. Inherit the preceding complete shared bank
as a replacement and complete cumulative native state before optimizer census;
use fresh Adam at stage transition. Exact same-stage resume restores all
trainable masters, Adam,RNG,sampler and accepted review evidence.

The initial budget is one successful update per admitted source/sigma pair,
with the exact manifest/count pinned. Nonfinite skips do not advance the
sampler. Keep the existing eight consecutive /20% of100 skip halt and30-minute
rolling checkpoints,keep2. At25% budget, stop spending until held target-free
sampling has been reviewed with the actual H3 VAE decoder, independent ASR
and explicit intelligibility/content observations. Select
`speech_euler_steps=32`: Gaussian initialization at sigma1,32 equal descending
Euler steps to0,velocity=(x_t-predicted_endpoint)/sigma. No clean target is
supplied to the sampler; physical duration/chart and transcript/voice remain
fixed. Noise/ASR/voice quality are reported, not replaced with teacher-forced
loss. No standalone waveform quality threshold is invented.

D4 keeps its original3M Q38/1M K3/300pairs per original media lane. Add one
replay per accepted source row from the source-admitted audio/speech CE cache,
and one replay per accepted source/sigma speaking pair from the speaking cache,
with exact cache manifests, counts and source-group exclusions pinned. These
are separate explicit replay routes with the same objective definitions and
existing lr multiplier0.3; held diagnostics contribute zero updates. This
addition changes replay scope only through this dated amendment. The D4 runtime
must support these routes before it can claim coverage.

Production additionally requires actual current-base full-depth/read-policy
gradient parity, immutable source identity, complete preceding cumulative
states, sealed full source caches and an exclusive allocator runtime. This
candidate, tiny fixtures and successful parent ingress do not grant those proofs.


Candidate source SHA-256: 6fefe799b0775f3440a77eec222fa7c35e1e286d3b2d4246332df0e8ca66b0a8


### 2026-10-13 selected H3 same-mux guided_velocity preparation successor

This dated selection adds the authentic joint audio/video recovery lane to the
previous selected recovery amendments. The frozen D1-D4 body, earlier dated
audio/speech bodies, original independent preparation receipts and parent model
bytes remain unchanged. This document binds this new H3 route; earlier stage
configurations retain their independently selected preregistration files. No
training launch, accepted stage state, model quality or convergence is claimed.

The source scope is all 24 original joint_av procedural fixtures,8 train and16 held,
screened with the unchanged 225 audio/speech candidates and full 352 original
inventory against 91,984 normalized rows,1,948 frozen evaluation rows,17,203 prior
admission rows and21,081 cross-route exclusions. The immutable new 249 candidate
admission is de642d15eb976cce615264453e20d49b2eb71369cb1870b8eb9ad57638527202.
Its all8 train AV sources are eligible; earlier 194 native CE source admissions and
connected groups are unchanged. Held validation/test sources never enter the
optimizer and have no teacher targets in this initial capture.

The complete actual teacher scope is all 32 targets, 4 sigmas for each of all8 train
sources, with no source, sigma, seed or quality ranking. Initial preparation
selects one successful update per admitted source/target pair, exactly one pass
of the complete eligible capture. The frozen historical 300-pair H3 quota is
preserved as history and does not authorize repeats to fill a quota. Any D4
replay-budget successor must explicitly declare its complete eligible scope.

The teacher is the genuine BF16 MiniMax H3 checkpoint, fullweight SHA
a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4,
unchanged before/after capture. The sealed velocity manifest is
7770cf61b2b88a9dfce802009ca7e3d0f26be05ea9904f2ca47b9f002c90df17;
actual native geometry/packer proof is
2150acf1c6e30ac9b34076df43361b70b86c7dcb6cf52d02dc38db504fa62e77.
CFG 1 means one actual positive branch. For this capture guided_velocity means
matching that actual conditional parent's canonical velocity, without invented
unconditional evaluations, classifier-free amplification or foreign codecs.

Both streams decode the same physical mux and use start 0, end 22/24 seconds,
22 frames at 24 fps, 96x96, latest source-frame hold from all 8 source frames at 8 fps.
Measured audio/video origins agree; no waveform padding, silent fallback or PTS
gaps certify synchronization. The actual video model-space latent is24x7x6x6;
audio is32x2x37. The 29,333 resampled audio samples have a partial final nominal
800-sample hop; all 37 actual VAE latent frames are retained. Nominal hop coverage
is not additional physical source time or an exact receptive-field claim.

The video sigma uses shift 12. Audio sigma uses shift 3 at the same underlying
time; its exact derivative slope from shift 12 to shift 3 is source-pinned per target. Video target
is captured canonical v_video. Audio target is captured v_audio_native equal to
the flat guided audio velocity divided by that exact slope. Native loss compares
audio in its own sigma parameterization, with no additional slope factor. Raw
student parent-head signs are +1 for video and audio, so canonical prediction is
minus raw for both. Input features are noisy video 384:480 and audio 480:512;
emission targets are video 128:224 and audio 224:256. Clean teacher latents and
authored assistant metadata never enter student conditioning.

The actual full sequence is native parent-template text prefix, channel-major
stereo audio 74 rows, then video 63 rows. Authentic MiniMax PackedLayout and packers
are bitwise-proved for all 32 targets. The prefix uses the genuine parent template
and only the user generation request plus its explicit native-window view.

The new native bridge uses a single shared clock from the measured finite
perturbation dominance groups [[0],[1,2,3,4],[5,6,7,8],[9,10,11,12],
[13,14,15,16],[17],[18,19,20,21]], identical across all 8 sources. Video time is
each group's mean frame-center time; spatial grid endpoints are retained. Audio
uses actual 40 Hz channel-major centers and categorical stereo branch identity.
The bridge file SHA is f95cb494ac12fa491667aacc261b976241e3ce7b934b7719a9407fb2ee9f0591.
This measured coordinate selection is an explicit dated bridge convention;
finite perturbation dominance does not prove exclusive receptive fields. The
separate parent rotary PackedLayout parity remains bitwise-established.

The existing D3b h3_dit_L25_curve6 engine, shared-r16 inheritance and cumulative
native-state handoff own this stage. Only audited unit/emission/shared-r16
parameters train, with unit/emission LR 2e-5 and r16 LR 1e-4; AdamW betas .9/.95,
eps 1e-8, weight decay 0, clip 1, warmup 20, min LR ratio .1, seed 20261013. One pass
and exact checkpoint resume preserve all source/target identities. Missing
accepted preceding state, exact-base parity, finite gradients, held generation
evaluation or exclusive runtime keeps production launch closed.
