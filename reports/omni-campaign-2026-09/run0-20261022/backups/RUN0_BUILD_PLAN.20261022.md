# RUN 0 build plan: co2-2b planted from Qwen3.5-2B, ablated, distilled on all modalities on 4 P40s

Status: FINAL plan, 2026-10-22. It replaces the draft. The critic's items 1-27 are each applied or rejected with a reason in §8. The critic text that reached this pass stopped part-way through item 27, so any later items were never received.

Ruling: `cyril-run0-plant-qwen-2026-10-22`, appended verbatim to `distill/STAGE_D_RULINGS_2026-10-12.md`. Branch: `claude/arch-v2-20261021`, HEAD 8d8f064, clean tree. Worktree: `C:/Users/cereal/Documents/ai training/.agent-worktrees/chimera-omni-5d4d4e1dcd/workers/claude/arch-v2-20261021`. Local git only, never push. Commit messages end `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Inputs read for this plan**
- The four surveys in this directory: `survey_code.md`, `survey_data.md`, `survey_hw.md`, `survey_lit.md`.
- `ARCH_CANDIDATES_2026-10-22.md`.
- `ARCHITECTURE_V2_2026-10-21.md` §3-§9 and §14.
- `TOKENIZER_FREE_STUDY_2026-10-22.md` §2.3 and §3e.
- The `ARCH_DRIFT_FINDINGS_2026-10-22.md` summary.
- `H3_HYBRID_AV_FLOW_PROOF_REPORT.md`.
- `OMNI_LAB_GUIDE.md`.
- The branch's `v2_layout.py`, `native_media.py` and `v2_trainer.py`, checked in this pass:
  - `V2Sample`, `MicroBatch`, `save_checkpoint` and `find_fold_layout`/`load_fold_base`;
  - the CLI defaults `--lr 1e-5`, `--warmup 50`, cosine `lr_at`, `--weight-decay 0`, `--dist-timeout 1800`.
- The memory rulings `omni-full-distill-k3-primary`, `omni-corpus-tokenizer-agnostic`, `omni-v2-direction-2026-10-21` and `ai-server-serving-layout`.

Every number below is either measured in a cited survey or marked **(projection)**.

---

## 0. What Cyril must decide before launch (S5)

Four items need Cyril's yes. Build work does not wait on any of them, but S5 (launch) waits on the first two.

| # | Decision | Options | Default if he says "your call" | Blocks |
|---|---|---|---|---|
| **D1** | K3 primacy. The binding 10-17 ruling says "99% kimi k3". K3 has 1.89M cached positions and cannot grow during run 0 without GPU6, which belongs to Strata. | (i) Run 0 uses K3 1.0 wherever it exists, and Strata/Flash Next (same vocab) is primary at volume. (ii) Borrow GPU6 for K3 scoring passes; this needs his approval because GPU6 is Strata's. (iii) Keep a large K3 share with a repetition cap (6 epochs). | (i) plus the (iii) cap | S5 |
| **D2** | GPU5 and the hourly media judge (Qwen3.6-27B Q4, needs GPU5 for 6-35 min each hour). | (a) Pause `media-judge.timer` for the run, so judged media stops reaching intake. (b) Run on 3 cards (GPU1/2/3) and leave the judge on GPU5, at about 30% less throughput **(projection)**. | (a), because he said "all 4 p40s" | S5 |
| D3 | Strata as a data writer through a **standalone client** of `strata-gate :18400` at background priority. It would write the intent bank and edit instructions and capture `n_probs=64` on its own generations. No service is modified. | yes / no | no: run 0 launches on cached signal only | nothing |
| D4 | Qwen3-TTS-1.7B renders of COCO captions through the existing media-gate idle queue, for concept-interleaved samples. No service configuration changes. | yes / no | yes, citing his 10-14 "[Cyril quote withheld]". Recorded as a lead decision in the rulings file. | the concept-interleaved bucket only |

Downloads that need approval and do not block launch: VGGSound or AudioCaps or Clotho, which give real sound effects and real paired audio-video. Until one arrives, run 0 has **no sound effects, no real joint audio-video and no video-understanding text**, and this plan says so wherever "all modalities" would otherwise imply it (§2.5).

The Qwen3.5-2B download itself is covered by his order ("[Cyril quote withheld]"). It is Apache-2.0 and not gated.

Lead (Claude) decisions recorded in the rulings file before S5. These are not Cyril's words:
- **L1: ratify or revoke the provisional bf16 K3 cache** (`teacher_cache/k3/eb35851b2496a628-provisional-bf16/`).
  - Ratify for run 0 only if the cardclass verdict in `$Q/k3_cardclass_provisional/` and `$Q/k3_b2_numerics_ruling.txt` hold, and the cross-map matched mass is at least 0.90 (measured 0.91).
  - If revoked, the K3 bucket goes to 0 and text KD takes its share.
- **L2:** the D4 TTS ruling line.
- **L3:** the lane override notice. A Relay notice goes to the other lanes: GPU1/2/3/5 are held for run 0 for an expected 10 days or more **(projection)**, under `cyril-run0-plant-qwen-2026-10-22`.

---

## 1. Problem

Cyril ordered a first-pass 2B Chimera Omni v2:
- plant Qwen3.5-2B into co2-2b;
- ablate it so it can relearn;
- distill it across all modalities on our architecture on all four P40s, for days, extended as data grows;
- skip the test campaign.

The fold pass comes next and is compared against run 0.

The branch has the geometry, importer, layout, x0 head and a working 2-card trainer. None of the four run-0 steps can run today:
- the weights are not on disk;
- there is no plant or ablate path;
- there are no marker or decision tokens;
- media is toy-only;
- multi-card resume is broken (omni_lab 38324);
- there is no 4-card lane.

This plan fixes every open decision with a default. It splits the build into three areas, **A**, **B** and **C**, with disjoint file ownership and written interfaces (§3), so three builders can work at once.

---

## 2. Design

### 2.1 Model config and plant

**Profile.** `co2-2b` geometry is unchanged:
- 24 layers, 18 GDN : 6 gated GQA, pattern `kda,kda,kda,gated_gqa`, width 2048, FFN 6144.
- GQA: 8/2 heads × 256 with 64 rotary dims, theta 1e7, mRoPE (11,11,10).
- GDN: 16/16 heads × 128, conv 4.
- Block AttnRes, block size 4, `attnres_init="zero"`.
- A6b `native_unit_norm="rms"` on.
- Off: A8, the x0 skip and CFG. All three measured neutral or worse in rounds 5-6.
- Tied vocab 248,320 + 512 control rows. Raw unit 3,072 values. Head GLU 2048→4096→3072.
- Parameter count: 1,923,165,809 [measured, survey_code].

New registered profile `co2-2b-run0` = `co2-2b` plus metadata `init_rule = V2_INIT_RULE_RUN0`:
- `"plant: Qwen/Qwen3.5-2B text trunk, ablated (ruling cyril-run0-plant-qwen-2026-10-22); run 0 only"`
- and `init_source = "Qwen/Qwen3.5-2B@<sha256>"`.

`V2_INIT_RULE` (fold-built) stays for the fold pass. On the server, `/home/cereal/omni-campaign/rulings/arch_v2_init_source.json` records the ruling id, Cyril's verbatim order and "run 0 only; fold pass next".

**Checkpoint.** `Qwen/Qwen3.5-2B` (post-trained, Apache-2.0, 4,548,221,488 B) goes to `/home/cereal/models/Qwen3.5-2B` on the ext4 root, which has 330 GB free. The NAS is full; never write there.

Post-trained, not Base:
- The pass criterion is keeping chat and tool behaviour.
- All 72M cached teacher positions were rendered with Qwen's chat template (`enable_thinking=False`).
- Base would have to relearn chat from about 26M distinct cached tokens.

Before any copy, assert the geometry field by field against co2-2b. The 2B values are web-sourced until this check runs.

**Video unit.** The code wins over the spec table: 2 frames × 3 × 32 × 16 tubelets (`V2_VIDEO_TUBELET=2`, `V2_VIDEO_PATCH=(32,16)`). Tubelets were the round-3 fix that took video pan from 0.500 to 1.000. The ARCH spec §4 row gets corrected.

**Plant mapping.** `importing/qwen35.py` already maps the whole text trunk; the plant tool adds the rest.

| Student tensor | Source | Note |
|---|---|---|
| `embed.weight[0:248320]` (tied head) | Qwen `embed_tokens` | exact fp32 copy |
| `embed.weight[248320:248337]` (17 used control rows) | ids 0-7 existing + 9 new (§2.3) | init per §2.3; never zero (tied table: a zero row is a zero input) |
| `embed.weight[248337:248832]` (495 spare rows) | new | mean text row + N(0, (0.02·rms)²) |
| norms: 24 × input/post-attn, final, q/k | Qwen `(1+w)` folded to a plain weight | exact |
| 24 × `ffn.up` (= gate_proj ‖ up_proj), `ffn.down` | Qwen MLP | exact; the ablation touches only these |
| 18 × GDN: q/k/v split of `in_proj_qkv`, 3 depthwise convs from `conv1d`, `in_proj_z`→gate, `in_proj_b`→write, `in_proj_a`→decay, `dt_bias`, `A_log`, `out_proj`, per-head `norm` | Qwen linear_attn | exact |
| 6 × GQA: `q_proj`/`gate_proj` de-interleaved from the fused q, `k_proj`, `v_proj`, `o_proj` | Qwen full attn | exact; mRoPE is exact for text (t=h=w) |
| `*attention_residual*` (pseudo-queries, norms) | new, zero init | function-preserving; 2.6e-4 logit error measured and test-pinned |
| `native_media_bridge.*` | new, repo init, then the stage-0 alignment (§2.4a) | touches media rows only; text at step 0 is exactly Qwen |
| `native_media_head.*` (RMSNorm + GLU) | new, repo init, last linear zero-init | media output at step 0 is 0, the signal mean |
| `model.visual.*` (297 tensors, about 0.30B) | **not loaded into the student** | used only as the stage-0 alignment **teacher** (§2.4a), then discarded; shapes and hashes go in the receipt |
| `mtp.*` (15 tensors) | **dropped** | `mtp_heads=0`; MTP lost on the phone; hashes stay in the receipt so it can be revived as a training aid |

**Route.**
1. `load_qwen35(residual="none")`.
2. `load_state_dict(strict=False)` into `co2-2b-run0`.
3. Assert the missing keys are exactly `*attention_residual*`, `native_media_bridge.*` and `native_media_head.*`, and that nothing is unexpected.

**Output layout**, mirroring the c21 fold layout so `find_fold_layout` resolves it unchanged:

```
/home/cereal/omni-campaign/run0/plant/receipt.json      schema chimera_omni.run0.plant_receipt.v1
/home/cereal/omni-campaign/run0/plant/child/            chimera safetensors fp32 (~7.7 GB) + config.json
```

After writing, run `chmod -R a-w`. Nothing ever trains on `plant/` in place.

### 2.2 Ablation (literature survey §1.2-1.4)

All magnitudes here are our own engineering inference; run 0 is the first data point for them.

First clone `plant/` → `ablate/` (same layout, `ablate/receipt.json` with schema `chimera_omni.run0.ablate_receipt.v1`, carrying the plant receipt sha). Then, with seed 20261022:

- **Shrink, `ffn.down` only, blocks 7-23:** `down ← 0.9·down`. 17 tensors.
  - This is the critic's item-9 fix. Scaling gate, up and down together would scale each GLU block's output by about 0.9·0.9·0.9 ≈ 0.73. Scaling only `down` scales the FFN output by exactly 0.9.
  - The bottom 7 of 24 blocks (29%) stay untouched, following Spurious Forgetting 2501.13453.
- **Perturb, blocks 7-23:** `w ← w + ε`, with `ε ~ N(0, (0.05·rms(w))²)` on both `ffn.up` and `ffn.down`. The RMS is measured, not an assumed init std.
  - 34 tensors change in total (17 `ffn.up` + 17 `ffn.down`).
- **Untouched:**
  - the embedding/head table and all norms;
  - all GDN and all GQA tensors;
  - blocks 0-6 entirely;
  - AttnRes, the bridge and the x0 head.
- **Probe of the change** (in the receipt, not a gate): 512 mixed text sequences, plant vs ablate. Record:
  - per-block residual-stream RMS ratio;
  - per-block hidden cosine;
  - final top-10 overlap (plant top-10 ∩ ablate top-10).
- **Collapsed-head probe**, diagnostic and bounded:
  - Run the same 512 sequences through the 6 GQA layers (48 heads).
  - A head counts as collapsed if its mean attention entropy is below 0.1 nats or above 0.95·log(ctx).
  - Reinitialise the Q/K/V rows of collapsed heads and zero their O columns only if the count is at most 4. Otherwise record and skip, since the source for this is second-hand. Expected count: 0.
- **Rejected:** a global λ of 0.2-0.4 (it optimises time to reach a from-scratch loss, not retention); FIRE (goes to the fold pass); norm-gain resets.
- **The plasticity lever doing the real work is in training:** decoupled weight decay 0.1 on trunk matrices, plus an LR re-warm (§2.7).

The receipt records, per tensor: RMS and sha256 before and after, λ, σ, seed, the exempt list, the probe result and the head result.

**S2 rollback:** if the 8-prompt read after ablation is visibly broken, delete `ablate/`, re-clone, and apply λ=0.95 to `ffn.down` of blocks 12-23 only.

### 2.3 Marker tokens and the output-decision token

The new tokens are appended to `V2_CONTROL_TOKENS` at ids 248320+index. Existing indexes 0-7 keep their ids, and a test pins this.

| idx | id | token | role |
|---|---|---|---|
| 8 | 248328 | `<|text|>` | input and output piece marker |
| 9 | 248329 | `<|image|>` | marker; opens image spans (replaces `<|media|>` as the opener) |
| 10 | 248330 | `<|audio|>` | marker for speech, music and sound effects; the kind stays in the header text (`speech 38400hz mono`, the round-7 rule) |
| 11 | 248331 | `<|video|>` | marker |
| 12 | 248332 | `<|3d|>` | marker |
| 13 | 248333 | `<|out|>` | opens the decision |
| 14 | 248334 | `<|/out|>` | closes the decision |
| 15 | 248335 | `<|edit|>` | decision modifier; precedes the modality it edits |
| 16 | 248336 | `<|mask|>` | reserved and untrained (housekeeping per ARCH_CANDIDATES, not an adoption) |

`<|media|>` (idx 2) stays in the table but is no longer emitted. Spans open with their modality marker.

**Row init:**
- `<|image|>` ← Qwen `<|vision_start|>`;
- `<|video|>` ← `<|video_pad|>`;
- `<|text|>`, `<|audio|>`, `<|3d|>`, `<|edit|>` ← the mean embedding of the label word's tokens;
- `<|out|>` and `<|/out|>` ← the mean of the marker rows;
- each then gets + N(0, (0.02·row rms)²).

Qwen special ids are resolved from `tokenizer.json` at runtime, never hardcoded.

**Chat format: Qwen `<|im_start|>role\n … <|im_end|>` everywhere.** The band's `<|user|>`, `<|assistant|>` and `<|eos|>` rows are kept but never emitted. The donor and every teacher cache already use im_start.

**Sequence grammar.** `<|/media|>` closes every span.

```
text in       <|im_start|>user\n<|text|>what is this?<|im_end|>
image in      <|im_start|>user\n<|image|>image 512x384<|/hdr|><|raw|>×192<|/media|><|text|>describe it<|im_end|>
text reply    <|im_start|>assistant\n<|out|><|text|><|/out|>It is a ...<|im_end|>
image reply   <|im_start|>assistant\n<|out|><|image|><|/out|><|image|>image 512x512<|/hdr|><|raw|>×256<|/media|><|im_end|>
edit reply    <|out|><|edit|><|image|><|/out|> + span   (the source image is a clean span in the user turn)
mixed reply   <|out|><|text|><|image|><|/out|> + text + span, in the declared order
AV reply      <|out|><|video|><|audio|><|/out|> + video span + audio span (sequential suffixes in run 0; joint = C-7)
```

**Supervision.**
- Decision tokens are cross-entropy targets with weight 1.0. "Output mode comes from understanding the request, never keyword routing" is met by the data, not by code. There is no router anywhere.
- A generation (x0) forward cannot carry text labels. So every generation sample gets a **twin causal sample**: the request plus `<|im_start|>assistant\n<|out|>…<|/out|>`, about 12 supervised tokens, batched into the understanding pass.
- For cached text corpora, a **remap view** inserts `<|text|>` after `user\n` and `<|out|><|text|><|/out|>` after `assistant\n` at load time:
  - inserted positions have no teacher cache; cross-entropy applies only to the decision tokens, and `<|text|>` is unsupervised;
  - every original position keeps its cache index.
- Same-input-different-intent pairs come from the same media item with different requests (§2.5).

### 2.4 Media at real sizes, generation and losses

| Modality | Unit (3,072 values) | Run-0 sizes | Rows | Decode |
|---|---|---|---|---|
| Image | 32×32×3 patch | understanding: long side ≤ 768, aspect-bucketed to multiples of 32; generation: long side 512 (512², 512×384, 384×512, 512×768, 768×512) | 192-576 in, 256-384 out (draft 1/2 = 64-96) | PIL, resize, per-dataset gain constant (`native_media.py:670-715`) |
| Video | 2 frames × 3 × 32 × 16 tubelet, 12.5 fps | understanding: 256×448 up to 2 s at 12.5 fps, or up to 8 s at a declared 2 fps; generation: 192×320, 2 s | 2,688 in; 1,440 out (+360 draft) | ffmpeg/PyAV → 12.5 fps pairs (presence asserted at rank start, §2.9) |
| Audio | 3,072 mono samples at 38.4 kHz = 80 ms | speech 2-12 s; music 10 s mono (stereo deferred; ACE downmixed) | 25-150 | polyphase resample from 48k/24k/22.05k/16k |
| 3D | 8³ voxels × 6 (SDF, R, G, B, rough, metal) at 64³ | 64³ only | 512 | hy3d_voxel64 pipeline; ModelNet PBR = 0 |

1024² images (1,024 rows) are a stage-2 lever once throughput is measured. 512 is the published raw-pixel regime (JiT runs at 256 and 512); it is not a 2×2 toy.

**Generation = joint refinement of every unit of the item.**
- One open suffix holds all rows of the item, and all rows share one σ per sample.
- In the 6 GQA layers, every row attends bidirectionally to the whole span and causally to the prefix. GDN layers stay raster-causal (A8 off).
- σ is drawn from a logit-normal, shifted toward high noise by sqrt(rows/256).
- Draft-then-refine stays at factor 2 for image and video; it is the only mechanism that lifted video-out (0.400 → 0.625).
- Dumps sample with 16 Euler x0 steps. Audio sampling is stochastic per DiTAR.
- **Joint two-span refinement** (video plus its audio on the shared 80 ms clock) is item C-7, a stretch item and not a launch gate.

**Media loss (A-6, `media/training.py`):**
- Keep x0 prediction; JiT shows x-prediction is required at 768+ values per patch.
- Weight the loss in velocity space: `w(σ) = 1/max(σ, 0.2)²`, normalised so E[w]=1 under the training σ schedule, times a type-balanced reduction. This is untested here (A7) and cheap.
- Audio keeps the multi-resolution STFT, level and boundary terms; mel weight 0.05.
- **Off in run 0:**
  - LPIPS: VGG weights in VRAM on a 24 GB card already at a 20.7 GB peak, and the LPIPS package can pull in Triton (§2.9).
  - REPA/QVL32: the evidence is weak.

#### 2.4a Stage-0 bridge alignment for image input (critic item 27, accepted)

The bridge starts random, and image input has no bootstrap. Qwen3.5-2B's own vision tower outputs merged tokens covering 32×32 px, the same granularity as our raw patch, directly in the planted trunk's input space.

Stage 0, run once between S2 and S3, on one P40, about 4 h **(projection)**:
- Freeze the whole trunk. Train only `native_media_bridge.*` on COCO images at the §2.4 understanding sizes.
- Loss: cosine + 0.1·MSE between the bridge's output rows and the tower's post-merger outputs, both RMS-matched to the text embedding rows.
- Discard the tower afterwards; the student never contains it. The end model stays free of encoders and towers.
- Images only. The tower's video tubelet (2 frames × 32×32) does not match ours (2 × 32×16), and the bridge is untyped, so audio, 3D and video rows train in run 0 proper at the new-module learning rate.
- Output: `ablate+align/` (same layout, `receipt.json` with schema `chimera_omni.run0.align_receipt.v1`, chaining the ablate sha). This is the run's `--init`.

The other media teacher caches (the ViT features, Qwen-Image velocity, ACE codes) are **unused in run 0**. Most are in codec or latent space. "Distill across all modalities" is met in run 0 by text KD, teacher-generated media targets, and this stage-0 image alignment.

**Text loss (C-3, `chunked_text_objective`):**
- Cross-entropy:
  - 0.1 on cached text;
  - 0.3 on media answers, raised to 1.0 only after dumps show the reply style holds (critic item 15);
  - 1.0 on decision tokens.
- 3-gram unlikelihood at 0.1.
- Top-K bucket forward KL per same-vocab teacher, with per-position weights:
  - **where K3 exists:** K3 1.0, Strata 0.3, Q38 0.1;
  - **elsewhere:** Strata 1.0, Q38 0.3.
  - Subject to D1.
- KD and CE land only on positions where `loss_mask` is true (§3.2): turns from allowed writers.
- **Metrics:** top-10 overlap (student top-10 ∩ teacher top-10, /10) and the share of the teacher's top-10 probability mass the student captures, per teacher, every logged step. Top-1 is never logged as a gate.
- **Context:**
  - a 4,096 main lane;
  - a 16,384 long lane, which stays **off** until S0 proves the attention kernel at 16k (§2.9).
  - **50k+ is not trained in run 0**, because fp32 per-layer checkpoints alone come to about 12 GB at 50k. This needs C-8.
  - The native positions (theta 1e7, 262,144) carry the capability. Tell Cyril this plainly.

### 2.5 Data mixture

**Targets** are token-position shares per optimizer step. `MixtureAccountant` enforces text + tool ≥ 40%. **Every source has an epoch cap, counted per 7 days of run (rolling).** Budget freed by a capped or empty bucket goes, in order, to:
1. COCO image understanding and generation;
2. procedural edits (only if D3 = yes);
3. MAESTRO;
4. ModelNet;
5. text KD, subject to its own cap.

Realised shares are logged every 10 steps.

| Target % | Bucket | Sources (provenance tag) | Cap (epochs / 7 days) |
|---|---|---|---|
| 22 | text KD | `distill/corpus/text_clean_v1` (12.35M, Q38 + Strata top-64), row-filtered [registry] | 6 |
| 10 | K3 rows (oversampled) | `corpus/k3/pass-0..3` + xtok caches [k3]; 0 if L1 revokes | 6 |
| 8 | tool calls | `toolcall_flash_v1` (Strata-written), `bfcl_native_v1`, intake `strata-toolcall-*` [strata-gen/public]; loss only on Strata or public turns | 6 |
| 3 | long (16k lane, once enabled) | `corpus/long`, 31 docs [public] | 8 |
| 4 | decision twins | twins of every generation sample; intent variants when D3 = yes | n/a (derived) |
| 14 | image understanding | COCO 2017 captions (118k × about 5) with **caption-style requests only** ("one-line caption"), WIT subset [public] | 4 |
| 12 | image generation + edit | COCO caption → image at 512 (bulk); judged t2i (278+); judged edit triples (45+); procedural edits (mirror/crop/grayscale/hue) **only with Strata-written instructions (D3)** [public/media-teacher/procedural] | 4 (edit triples 4 → token presence) |
| 4 | video generation | 18 H3 SOL + 5 LTX with their prompts; the 512 H3 clips video-only, with their prompts as **conditioning only, no CE**, after the B-0 authorship check [media-teacher] | 4 → **token presence only** |
| 12 | speech in/out | LJSpeech 24 h (punctuated); LibriSpeech dev/test 10.8 h (normalised case, ASR target only, below); Cyril's 186/194 items (cleared) [public/cyril] | 4 |
| 6 | music in/out | ACE renders 236 + 120 with prompts (authorship check, B-0); MAESTRO (CC BY-NC-SA, **recorded on the run**), staged off the NAS [media-teacher/public] | 4 |
| 3 | 3D in/out | hy3d_voxel64 v1 (18), the 141 HY3D GLBs voxelised, ModelNet10 (4.9k) [media-teacher/public] | 4 |
| 2 | concept-interleaved | COCO image + caption + TTS of the caption (Qwen3-TTS-1.7B, D4) in one context, order randomised; the only three-modality set | 4 |

**Absent from run 0, and said so to Cyril:**
- video understanding: there is no usable video→text;
- sound effects / environmental audio;
- real joint audio-video;
- 50k-context training.

Dropped from the draft:
- **vision_in** (critic item 7): its answers are excluded, so the bucket supervises nothing.
- the **mismatched H3 LibriSpeech audio**.

**Caption style (critic item 15):**
- LibriSpeech is uppercase with no punctuation. It becomes sentence-case and is used only with "transcribe exactly" requests. Use LibriTTS-R or LJSpeech punctuated text where available; B-0 checks for LibriTTS-R on disk.
- COCO captions pair only with "one-line caption" style requests, never with "describe it".

**Prompt authorship (B-0, critic item 4).** Before use, check who wrote each prompt set:
- ACE v9 prompts;
- the H3 clip prompts;
- the procedural-edit and intent templates;
- the music prompt sets.

A set written by Claude or Codex may only be **unsupervised conditioning in the generation direction** (`loss_mask` false over it). It is never a CE or understanding target. Edit and intent phrasings come only from COCO human captions or from Strata (D3). Claude and Codex never author training text.

**Intent bank** (only if D3 = yes):
- Strata writes requests over COCO items, using their human captions, and over judged generation items, using their generation prompts subject to B-0.
- Six intents: describe, caption-joke, edit, redraw, animate, sound.
- The decision label comes from the template slot, never from a model.

**Row-level exclusion guard** (`training/run0_data.py`, critic item 5):
- Every manifest source carries `provenance ∈ {public, cyril, registry, strata-gen, k3, media-teacher, procedural}` and a sha.
- **Path-level refusals:**
  - `selfgen`, `tool-selfgen`, `toolcall/rescore`, `qtext_vl`;
  - vision_in answers;
  - `strata-capture-*`, `capture-export`;
  - media-judge captions;
  - `d2_refill`, `refill-codex-v7/v8`, `toolcall-q38`;
  - anything Sage-derived.
- **Row-level:** each record's source tag is checked against the same denylist. Text inside `text`/`text_clean` that comes from d2_refill (88k) or selfgen (1.0M) is dropped by row, not by file.
- **Turn-level:** in tool-call rows, turns from Claude, Codex or cloud models are context only, with `loss_mask` false. "Only Strata's own turns become training targets."
- The tokenized view's dataset identity is `(corpus_version, tokenizer_sha + template, teacher)`, because the canonical corpus is tokenizer-agnostic.

**Data off the NAS (critic item 19).** MAESTRO and anything else on `/mnt/ai_data` is staged to the Lexar (exFAT, no symlinks, 435 GB free) or to root before launch. The training loop never reads the NAS. Every loader read has a 30 s timeout, and timeouts are counted per source.

**Streaming intake.** `scripts/run0_intake_sync.py`, run by rank 0 at each checkpoint boundary:
1. scan `/mnt/lexar/corpus-store/chimera-intake/*/<batch>/READY.json`;
2. run the row-level guard;
3. dedupe against hold-out registry v3;
4. write `run0_data_v{N+1}.json`;
5. log an omni_lab `dataset_version` + `run_dataset`.

Rank 0 broadcasts the manifest sha, and all ranks switch together at the checkpoint barrier (critic item 26).

### 2.6 Parallelism and memory on 4 P40s (no P2P)

**DDP, one process per card, fp32.** Gradients go device → pinned host → gloo all-reduce → device; this is proven at 2 cards (230 tok/s).

Rejected:
- FSDP/ZeRO: weights + grads (15.4 GB) fit per card.
- NCCL: if ever used, `NCCL_P2P_DISABLE=1` is mandatory.
- adamw-offload: about 110 GB host.

**Optimizer:** Adafactor fp32 with new decoupled WD 0.1 on trunk matrices only.

**Memory and kernels:**
- Per-layer non-reentrant checkpointing, chunked GDN scan 64, head chunk 256, `expandable_segments`.
- **GQA attention** (critic item 10): today `layers.py:872-885` does `repeat_interleave` + SDPA fp32 with an explicit mask.
  - A-8 wraps it in `torch.nn.attention.sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])`, so an unsupported shape raises instead of silently using the math path.
  - The math path would cost per layer: 512 MB at 4k, 2.1 GB at 8,192 and 8.6 GB at 16,384.
  - If the efficient kernel refuses fp32 / head_dim 256 / an arbitrary mask on sm_61 at any planned length, A-8 implements query-chunked attention with an online softmax (chunk 1,024) as the fallback. The 16k lane stays off until S0 passes at 16,384.

**Per micro-step, per rank:**
- understanding pass ≤ 8,192 positions;
- generation pass ≤ 4,096 positions via the new `--gen-tokens` (bucketed by rows, left-collated; one 2 s video or about 12 images at 512²);
- a long-lane row (16,384) takes a micro-step alone.

`--accum 4` gives one host all-reduce per 4 micro-steps, about 196k positions per optimizer step.

**Batch building is sharded by rank:** rank r builds micro-batches r, r+4, … from a seeded RNG, with a 2-worker decode prefetch per rank. Each rank uses 10-12 CPU threads. The GPU5 rank is pinned with `numactl --cpunodebind=1 --preferred=1`; the GPU1/2/3 ranks run on node 0.

**Host RAM (critic items 11-12):**
- Steady state is about 11 GB RSS + 3 GB decode per rank, about 56 GB total **(projection)**.
- At resume:
  - rank loads are staggered 60 s apart;
  - safetensors load by mmap straight to the device, and CPU copies are freed;
  - the resume broadcast goes tensor by tensor, never as one blob.
- `--mem-floor-gib 60` applies at launch, and **every step**: rank 0 reads MemAvailable, and below 60 GiB it finishes the step, checkpoints, writes `ALERT` and pauses. The supervisor resumes when there is 75 GiB free.
- `oom_score_adj +500` on every rank, so an OOM kills the trainer, never Strata.

**Throughput (projection from the 1- and 2-card measurements):** about 450-535 positions/s, or 39-46M positions/day at accum 4, at about 330-400 s per optimizer step. Re-measure in the first hour and log `tok_per_s`. The ARCH §9 figure of 86M/day is retired.

### 2.7 Schedule, checkpoints, dumps, logging, resume, supervision

**Schedule** (set explicitly; the trainer defaults are cosine and lr 1e-5):
- Warmup: 150 optimizer steps, about 29M positions, about 14-16 h **(projection)**.
- Trunk peak 1.5e-4. New modules (bridge, x0 head, control rows, AttnRes queries) peak at 8e-4.
- Then **stable** indefinitely (WSD; decay only when branching a release checkpoint, for a 10% cooldown).
- WD 0.1 on trunk matrices.
- `--grad-clip 1.0`.

**Loss weights:** text KD 1.0, flow 1.0, CE per §2.4.

Response rule, not a default: if a dump shows text drifting, lower the trunk LR to 1e-4 and restart flow at 0.2, ramping to 1.0 over 300 steps.

**Checkpoints:**
- Every 50 steps, about 4.6-5.5 h **(projection)**, atomic, `--keep 3`, plus a kept milestone every 500 steps.
- They go to `/home/cereal/omni-campaign/run0/ckpt/` on ext4, about 8 GB each.
- `plant/`, `ablate/` and `ablate+align/` are immutable. Bad checkpoints are deleted, not archived.
- Root-disk floor: refuse to write below 60 GB free; alert and pause.

**Replica integrity (critic item 23):**
- At every checkpoint, each rank hashes its parameters and optimizer state.
- On a mismatch, rank 0's state is broadcast to all ranks tensor by tensor, and the event is logged to omni_lab and alerted.
- The first 3 steps after any resume also compare a gradient-slice sha across ranks.

**Sample dumps (critic item 22):**
- Every checkpoint. Items are **split across the 4 ranks**:
  - 8 fixed text prompts (greedy, 256 tokens, including 2 tool calls);
  - 2 decision probes: the same image with "describe it" vs "redraw it as a sketch";
  - 4 images at 512, 16 steps, plus 1 edit;
  - 2 TTS wavs, 2 music clips, 1 video of 2 s as mp4, 1 3D as slices PNG + marching-cubes GLB;
  - a contact sheet.
- The dump is timed in S3 and must finish in under 900 s per rank, against the 1,800 s gloo timeout.
- Output goes to `run/samples/step-N/`, synced every 6 h through the server-manager client to `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/run0-20261022/samples/`.

**Alarms** (critic items 20, 24, 25). These are alarms, not pass/fail gates: pass/fail is behaviour read by a person. Each one writes `run0/ALERT` and sends a Relay message to the lead.

| Alarm | Trigger | Action |
|---|---|---|
| top-10 overlap | relative drop > 20% from step 0 on the fixed held-out slice, against Strata, Q38 and K3, measured at each checkpoint | pause at that checkpoint until the lead reads the dumps and resumes |
| media head collapse | generated-sample std < 10% of data std | alert |
| modality starvation | per-modality decode failures counted; a bucket's realised share < 50% of target for 20 optimizer steps (after caps are applied) | alert; at 60 steps, halt with status `halted-data`, which the supervisor does not auto-restart |
| GPU health | per-rank step time, clocks and temperature; throttling on the passive cards | alert |
| foreign process | `nvidia-smi --query-compute-apps` shows any process on GPU1/2/3/5 that is not the trainer (for example the unlocked `comfyui_qwen21_gpu*` containers) | alert |
| ECC | retired pages at preflight | preflight refuses |

**Logging:**
- Every metric is mirrored to `run/metrics.jsonl` first, then sent to omni_lab with `|| true`, so a database outage loses nothing.
- omni_lab:
  - `log-run start --kind distill`;
  - `run_model` (base = the align checkpoint, teachers);
  - `run_dataset` per manifest version;
  - `metric` every 10 steps: every loss term, top-10 overlap per teacher, tok/s, peak VRAM, all-reduce seconds, realised shares, epochs used per source;
  - `run_event` for checkpoints, dumps, manifest bumps, alarms and restarts;
  - `log-run end`.
- Raw psycopg2 through the existing intake venv (vault key `omni_lab_pg`).

**Resume (C-1, a hard launch gate):**
- Root-cause the 38324 divergence from the CPU repro. `test_two_ranks_through_host_memory_…`: `blocks.1.attention_residual.norm.weight` differs by 5.7e-5, which points at reduction order or the Adafactor factored-state restore.
- Then add belt and braces:
  - after load, rank 0 broadcasts the model and optimizer state tensor by tensor;
  - all ranks assert an equal sha before step 1.
- `world` stays in `training_identity`. `--allow-world-change` is explicit, logged, and rescales the accountant, so a dead card does not strand the run.

**Supervisor (C-9, critic item 21).** `scripts/run0_supervisor.py` holds the lane claim and spawns the 4 ranks.
- If any rank exits, it kills the whole process group, then auto-resumes from the last good checkpoint with backoff (2, 5, 15 min).
- After 4 restarts within 6 h it gives up: status `failed-restarts`, `ALERT`, Relay message.
- The lane claim is kept across restarts and released only on a final stop.

### 2.8 Smoke checks before the long run

These are launch safety, not a proxy test campaign; Cyril said to skip testing.

| Check | What it does | Passes when |
|---|---|---|
| **S0 attention kernel** | one P40 micro-test of the GQA path at L = 4,096, 8,192 and 16,384, with the document mask and the refinement mask, fp32, head_dim 256, forward + backward | the efficient kernel is selected or the chunked fallback is active; peak VRAM is recorded |
| **S1 plant speaks** | the import receipt on the prompt pack, plus 8 greedy prompts (chat, 2 tool calls) | max logit diff < 1e-3, argmax agreement ≥ 99.9% against HF transformers, and the lead reads all 8 as coherent |
| **S2 after ablation** | the same 8 prompts, plus the §2.2 probe in the receipt | still coherent when read by a person; otherwise roll back per §2.2 |
| **Stage 0** | bridge alignment (§2.4a) | the 8 prompts are unchanged, since the trunk is frozen; 4 COCO images captioned through the planted trunk; the lead reads them |
| **S3 4-GPU 100-step smoke** | real data, accum 1, plus one 16k long-lane micro-step (if S0 passed at 16k) and one 2 s video generation step | peak VRAM < 22.5 GB on every rank; every loss term finite; text KD and flow lower at step 100 than at step 1; top-10 overlap and tok/s logged; peak host RSS per rank logged; dump produced, timed (< 900 s per rank) and opened |
| **S4 resume** | STOP at 100, then `--resume auto` for 20 steps; then `kill -9` one rank mid-step and let the supervisor resume | identical replica sha at step 120; the supervisor brings the group back; peak host RSS during resume < 90 GB total |
| **S5 launch** | accum 4, after D1 and D2 | n/a |

### 2.9 Runtime environment guards

- **Python:** `/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260` (torch 2.6.0+cu118, sm_60 code). The cu128 and cu130 envs lack Pascal kernels.
- **Rank startup asserts** (critic item 13):
  - `torch._dynamo` is disabled;
  - no `triton` module is imported;
  - `pydantic`, `sqlalchemy` and other ORM modules are absent from `sys.modules` after our imports. The venv has pydantic installed as a dependency; our code must not import it.
  - ffmpeg/PyAV are present.
- **K3 bf16 cache:** converted to fp32 on CPU before transfer to the card.
- **Environment:** `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Never touch:** GPU0/4/6 and their services (strata-llama, strata-gate, media-gate, speech-worker@*, gen-worker-*). No P2P, no NCCL.

---

## 3. Interfaces between areas

A, B and C build in parallel against these written contracts. **A lands commit A-2 (tokens + layout API) first, within the first working block.** B and C code against the signatures below from the start.

### 3.1 File ownership

No file has two owners. Tests are one file per owner.

| Area | Owns (create or edit) |
|---|---|
| **A** arch, plant, ablate, tokens, losses, kernels | `src/chimera_omni/config.py`, `src/chimera_omni/v2_layout.py`, `src/chimera_omni/importing/qwen35.py`, `src/chimera_omni/media/training.py`, `src/chimera_omni/modeling/layers.py`, `src/chimera_omni/modeling/chimera.py`, new `scripts/plant_qwen35_2b.py`, `scripts/ablate_run0.py`, `scripts/align_bridge_stage0.py`, `scripts/attn_kernel_check.py`, `tests/test_run0_arch_torch_optional.py` |
| **B** data | new `src/chimera_omni/training/run0_data.py`, `src/chimera_omni/training/media_sources.py`, `src/chimera_omni/training/run0_samples.py`, `src/chimera_omni/training/text_remap.py`, `scripts/run0_build_manifest.py`, `scripts/run0_intake_sync.py`, `scripts/run0_stage_data.sh`, `scripts/strata_writer_client.py` (built only if D3 = yes), `tests/test_run0_data_torch_optional.py` |
| **C** trainer, parallelism, ops | `src/chimera_omni/training/v2_trainer.py`, new `src/chimera_omni/training/run0_dump.py`, `src/chimera_omni/training/run0_watch.py`, `scripts/run0_queue.sh`, `scripts/run0_supervisor.py`, `tests/test_run0_trainer_torch_optional.py` |

### 3.2 A → B, C: layout API (`v2_layout.py`, A-2)

```python
V2_CONTROL_TOKENS  # ids 0..7 unchanged; 8..16 = <|text|> <|image|> <|audio|> <|video|> <|3d|> <|out|> <|/out|> <|edit|> <|mask|>
MODALITIES = ("text", "image", "audio", "video", "3d")
MARKER = {"text": "<|text|>", "image": "<|image|>", "audio": "<|audio|>", "video": "<|video|>", "3d": "<|3d|>"}

class V2Sample:
    def text(self, token_ids, *, supervised=False) -> V2Sample                      # unchanged
    def control_token(self, name, *, supervised=False) -> V2Sample                  # unchanged
    def turn_open(self, role: str, enc) -> V2Sample      # <|im_start|>role\n  (ids from enc / tokenizer.json, never hardcoded)
    def turn_close(self, enc, *, supervised: bool) -> V2Sample                       # <|im_end|>
    def marker(self, modality: str, *, supervised=False) -> V2Sample
    def decision(self, outputs: Sequence[str], *, edit: bool = False, supervised=True) -> V2Sample
        # emits <|out|> [<|edit|>] <|m1|> <|m2|> ... <|/out|> ; outputs ⊆ MODALITIES, order = reply order
    def media(self, packet, *, modality: str, row=0, header=(), open_suffix=False) -> V2Sample
        # NOW emits MARKER[modality] (not <|media|>) + header + <|/hdr|> + <|raw|>×R [+ <|/media|>]
    loss_mask: list[bool]   # NEW, parallel to ids; True = allowed-writer position (KD/CE may apply). Default True
                            # for supervised=True text, False for conditioning text. B sets it per turn.
    modalities_in: set[str]; modalities_out: list[str]   # NEW, filled by marker()/decision()
```

`collate()` adds `loss_mask: BoolTensor[B, T]` beside `labels` and `attention_mask`. Everything else in its output is unchanged.

Config keys (`config.py`, A-2):
- `V2_INIT_RULE_RUN0`;
- `PLANT_RULING = "cyril-run0-plant-qwen-2026-10-22"`;
- profile `co2-2b-run0` (metadata `profile`, `init_rule`, `init_source`);
- `native_unit_norm="rms"`, `attnres_init="zero"`, `a8=False`, `x0_skip=False`, `cfg_drop=0.0`.

Media loss (A-6) signature:

```python
flow_loss(pred_x0, target_x0, sigma, type_ids, *, weighting="velocity", sigma_floor=0.2, mel_weight=0.05)
    -> (loss, terms: dict[str, Tensor])
```

C passes `--flow-weighting velocity`.

### 3.3 B → C: data contract (`run0_samples.py`, `run0_data.py`)

```python
@dataclass
class Run0Item:
    kind: str                 # "text_kd" | "understand" | "generate" | "twin" | "long"
    bucket: str               # mixture bucket name from §2.5 (accountant key)
    sample: V2Sample          # built with the §3.2 API; generate items end in an open suffix
    positions: int            # len(sample.ids); budget unit
    teacher: dict | None      # text_kd/long only:
        # {"positions": LongTensor[P]  (indexes into sample.ids),
        #  "ids": {teacher: LongTensor[P, K]}, "logprobs": {teacher: FloatTensor[P, K]}, "lse": {teacher: FloatTensor[P]},
        #  "weight": {teacher: FloatTensor[P]}}   # §2.4 per-position weights already applied by B
    source_id: str; provenance: str; record_sha: str
    epoch: int                # times this record has been served (cap accounting)

class Run0Source(Protocol):                       # one per bucket
    def next(self, rng: random.Random, budget_positions: int) -> list[Run0Item]
    def state(self) -> dict; def load_state(self, d: dict) -> None   # resume-exact
    def exhausted(self) -> bool                   # cap reached
    stats: dict                                   # served, decode_failures, timeouts

def build_sources(manifest: dict, *, cfg, enc, rank: int, world: int, seed: int) -> dict[str, Run0Source]
    # per-rank sharded; refuses (Run0DataError naming the path/row) anything the guard rejects
```

Media packets are the existing `NativeMediaPacket`:
- `features` float32 [1, R, 3072], with the per-dataset gain applied;
- `type_ids` and `positions` per `native_media.py`;
- `valid_mask` all true.

The rows match §2.4 exactly. For example, a 512×384 image gives 192 rows, and 2 s of video at 256×448 gives 2,688.

**Manifest** `/home/cereal/omni-campaign/run0/data/run0_data_vN.json`, schema `chimera_omni.run0.data.v1`:

```json
{"schema": "chimera_omni.run0.data.v1", "version": N, "parent_sha": "...",
 "tokenizer": {"path": "...", "sha256": "...", "template": "qwen35-chat enable_thinking=False"},
 "buckets": {"<name>": {"share": 0.22, "cap_epochs_per_7d": 6, "kind": "text_kd",
             "sources": [{"path": "...", "sha256": "...", "provenance": "registry", "rows": 0,
                          "teacher_caches": {"strata": "...", "q38": "...", "k3": "..."},
                          "corpus_version": "...", "license": "...", "authorship_checked": true}]}},
 "denylist_patterns": ["selfgen", "..."], "holdout_registry": "v3"}
```

This becomes the trainer's `--data-config`. The new `DATA_CONFIG_SCHEMA_RUN0` is accepted beside the existing v1.

### 3.4 A → C: init layout and lineage

`--init /home/cereal/omni-campaign/run0/ablate+align` contains `receipt.json` + `child/` (chimera safetensors + `config.json`), and `find_fold_layout` resolves it unchanged. The receipt schemas form a chain:

```
chimera_omni.run0.align_receipt.v1  ->  chimera_omni.run0.ablate_receipt.v1  ->  chimera_omni.run0.plant_receipt.v1
```

Each link carries `parent_receipt_sha256`, `ruling: "cyril-run0-plant-qwen-2026-10-22"` and `checkpoint.files` (name → sha256).

C-0 changes `load_fold_base` / `build_base_model` (in C's file) to accept `kind="plant"`, provided all of the following hold:
- the receipt schema is one of the three run0 schemas;
- the ruling equals `PLANT_RULING`;
- `cfg.metadata.init_rule == V2_INIT_RULE_RUN0`;
- the profile is `co2-2b-run0`.

The lineage is `{"kind": "plant", "chain": [...]}`, and trainer checkpoints whose base kind is `plant` are accepted for resume.

### 3.5 C: checkpoint layout (extends the existing `save_checkpoint`)

```
/home/cereal/omni-campaign/run0/run/checkpoints/step-XXXXXXXX/
  model/                  safetensors + config.json (source.init = lineage)
  optimizer.pt            Adafactor (+ decoupled WD state)
  trainer_state.pt        step, per-rank builder/source states (rank_k), RNGs
  trainer_state.json
  replica_sha.json        NEW: {rank: {params_sha, optim_sha}} — all equal or the checkpoint is not COMPLETE
  manifest.json           NEW: the run0_data_vN.json in force (sha recorded)
  COMPLETE.json           schema bumped to chimera_omni.v2_trainer.checkpoint.v2 (v1 still readable)
/home/cereal/omni-campaign/run0/run/{train.log, metrics.jsonl, samples/step-N/, ALERT, STOP, status.json}
```

**New trainer flags (C):**
- `--gen-tokens 4096`
- `--understand-tokens 8192`
- `--long-lane-tokens 16384` (0 = off)
- `--schedule wsd --warmup 150 --lr 1.5e-4 --new-module-lr 8e-4`
- `--weight-decay 0.1`
- `--flow-weighting velocity`
- `--media-ce 0.3`
- `--teacher-weights k3present:k3=1,strata=0.3,q38=0.1;else:strata=1,q38=0.3`
- `--mem-floor-gib 60`
- `--disk-floor-gb 60`
- `--allow-world-change`
- `--dump-every-ckpt 1`
- `--alarm-top10-drop 0.2`

---

## 4. Build plan

### Area A: arch, plant, ablate, tokens, losses, kernels

- **A-1** Download `Qwen/Qwen3.5-2B` to `/home/cereal/models/Qwen3.5-2B` through the server-manager client (`--script` with `huggingface-cli download`; no token needed). Record the sha256 of the safetensors and confirm the licence file is present.
  - Accept: `scripts/plant_qwen35_2b.py --check-only` passes the field-by-field geometry check against `co2-2b`.
- **A-2** **First commit.** `config.py` (`V2_INIT_RULE_RUN0`, `PLANT_RULING`, profile `co2-2b-run0`, spec §4 video row corrected in `ARCHITECTURE_V2_2026-10-21.md`) and `v2_layout.py` (the 9 tokens and the §3.2 API: `turn_open/close`, `marker`, `decision`, `media(modality=…)`, `loss_mask`).
  - Accept: ids 248320-248327 unchanged and 248328-248336 pinned (tests); `decision(["text","image"])` emits the exact sequence; collate returns `loss_mask`.
- **A-3** `scripts/plant_qwen35_2b.py`: import; strict=False load; assert missing/unexpected keys; control-row init (§2.3); zero-init the head's last linear; receipt; immutable `plant/`.
  - Accept: S1.
- **A-4** `scripts/ablate_run0.py`: clone; λ on `ffn.down` only; perturbation; residual-RMS/cosine/top-10 probe; head probe; receipt; immutable `ablate/`.
  - Accept: S2; the receipt lists exactly 34 changed tensors (17 down + 17 up), with sha before and after.
- **A-5** On the server, write `/home/cereal/omni-campaign/rulings/arch_v2_init_source.json` (ruling id, verbatim order, "run 0 only; fold pass next").
- **A-6** `media/training.py` (+ `modeling/chimera.py` call site): velocity-space weighting with σ floor 0.2, E[w]=1 normalisation, mel 0.05.
  - Accept: a proxy-config unit test shows E[w]=1 under the schedule and a finite loss at σ ∈ {0.01, 0.5, 0.99}.
- **A-7** `scripts/align_bridge_stage0.py`: stage-0 alignment of the bridge to the tower (§2.4a) on one P40, taken through `lanes.py claim-run`; output `ablate+align/`.
  - Accept: the trunk sha is unchanged against `ablate/`; the bridge-to-tower cosine on held-out COCO is logged; 4 captions are read.
- **A-8** `modeling/layers.py`: enforce `sdpa_kernel([EFFICIENT_ATTENTION])` on the GQA path; query-chunked online-softmax fallback (chunk 1,024) selected automatically when the kernel refuses. `scripts/attn_kernel_check.py` is S0.
  - Accept: S0; chunked and full attention agree to 1e-5 on CPU.

### Area B: data

- **B-0** Provenance audit. Prompt authorship for ACE v9, the H3 clip prompts, the music prompt sets and every template. Check whether LibriTTS-R is on disk. Inventory MAESTRO and the subset to stage.
  - Output: `run0-20261022/data_provenance_audit.md`, with a line per source.
  - Accept: every manifest source has `authorship_checked: true` or is conditioning-only.
- **B-1** `training/run0_data.py`: manifest schema; path + row + turn guard; `(corpus_version, tokenizer+template, teacher)` identity; epoch-cap accounting.
  - Accept: a manifest naming `selfgen` or `strata-capture-*` is refused with the offending path named; on a real `text_clean_v1` shard, rows tagged refill or selfgen are dropped and counted; a tool row with a Claude turn yields `loss_mask` false on that turn.
- **B-2** `training/media_sources.py`: image, video, audio and 3D decoders at §2.4 sizes; aspect buckets; polyphase resample; tubelets; per-dataset gain; 30 s read timeouts; per-rank sharding; 2-worker prefetch; failure counters.
  - Accept: one sample of each modality decodes to the stated rows (512×384 → 192; 2 s 256×448 video → 2,688; 10 s audio → 125; 64³ voxels → 512).
- **B-3** `training/run0_samples.py`: builders for understanding, generation + twin, edit (judged; procedural only with D3), video then audio (AV), concept-interleaved (D4), intent variants (D3). All use the §3.2 API; COCO goes only with caption-style requests; LibriSpeech is sentence-cased and ASR-only.
  - Accept: each builder emits a `Run0Item` whose ids decode to the §2.3 grammar; every generation item has a twin.
- **B-4** `training/text_remap.py`: a `TokenCorpus` wrapper that inserts `<|text|>` and decision tokens and preserves the cache index.
  - Accept: at every original position, a row's teacher-cache lookup matches before and after remap.
- **B-5** `scripts/run0_stage_data.sh` + `scripts/run0_build_manifest.py`:
  - stage MAESTRO and any other NAS-resident subset to the Lexar or root;
  - voxelise the 141 HY3D GLBs with the `hy3d-voxel` venv;
  - (D4) submit COCO-caption TTS jobs to the media-gate **idle queue only**, changing no service configuration;
  - build `run0_data_v1.json`.
  - Accept: the manifest passes B-1; staged files' sha matches their source.
- **B-6** (D3 only) `scripts/strata_writer_client.py`: a standalone client of `strata-gate :18400` at background priority. B-0 confirms which vault key the gate treats as background, rather than assuming it. It writes the intent bank and edit instructions, and seals the `n_probs=64` top-n beside each canonical record. It does not modify any gen-worker or other service.
  - Accept: 100 rows written, judged by the guard, with logprobs attached.
- **B-7** `scripts/run0_intake_sync.py`: guard, dedupe against hold-out v3, manifest bump, omni_lab `dataset_version`.

### Area C: trainer, parallelism, ops

- **C-0** `v2_trainer.py`: accept the `plant` lineage (§3.4) and `DATA_CONFIG_SCHEMA_RUN0`; consume `Run0Source` buckets through `MixtureAccountant` with caps and reallocation.
- **C-1** Fix the 38324 resume divergence from the CPU repro. Add tensor-by-tensor broadcast-on-resume, the post-load sha assert, the 3-step gradient-slice check, the per-checkpoint `replica_sha.json`, staggered rank loads and `--allow-world-change`.
  - Accept: the two-rank CPU test passes at 1e-6, and S4 passes.
- **C-2** Per-rank micro-batch sharding (`v2_trainer.py` ≈ l.1739); `--understand-tokens`, `--gen-tokens` with row bucketing and a split backward; long lane.
- **C-3** Per-position teacher weights; top-10 overlap and captured-mass metrics; decoupled WD in the factored Adafactor; two-group LR (trunk / new modules); WSD schedule; `loss_mask` respected by CE and KD; media CE 0.3.
- **C-4** `training/run0_dump.py`: items split across ranks; `v2_refine.sample_suffix`; materialise to PNG, WAV, MP4 and GLB; contact sheet; timed.
- **C-5** `training/run0_watch.py`: MemAvailable and disk floors, the alarms table (§2.7), `metrics.jsonl` mirror, omni_lab `run_model`/`run_dataset`/`metric`/`run_event`, manifest switch at the barrier.
- **C-6** `scripts/run0_queue.sh`, wrapping:

  ```
  lanes.py claim-run --lane L-TRAIN --gpus 1,2,3,5 --borrow "run0 2B distill on all 4 P40s (cyril-run0-plant-qwen-2026-10-22)" --wait 3600 --mem-floor-gib 60 --omni-lab-kind distill
  ```

  - It waits for locks and never takes held ones; it uses `--gpus 1,2,3` if D2 = (b).
  - Pascal venv, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, numactl for the GPU5 rank, `oom_score_adj +500`.
  - Preflight: plant, ablate and align receipts present and chained; manifest guard passes; ECC clean; §2.9 asserts.
  - Deploy code to `/home/cereal/omni-campaign/chimera-omni-v2/`, sha-matched to the commit.
- **C-9** `scripts/run0_supervisor.py` (§2.7): holds the claim, restarts the group, gives up after 4 restarts in 6 h, sends Relay and ALERT notices.
- **C-7** (stretch, not a launch gate) A multi-span open suffix for joint audio-video refinement, porting the August proof's per-type σ.
- **C-8** (stretch) Activation offload / 16-bit checkpoint storage for the 16k → 50k lane.

### Launch order

```
A-1, A-2 (first commit)
  → in parallel: A-3..A-6, A-8 | B-0..B-4 | C-0..C-3
  → S0, S1 (A-3), S2 (A-4), A-7 stage 0
  → B-5, C-4, C-5, C-6, C-9
  → S3, S4
  → D1 + D2 answered; L1-L3 recorded; Relay lane notice
  → S5 claim-run on 4 cards (or 3 per D2)
  → first-hour throughput re-measured and logged
  → while training: B-6 (D3), B-7, C-7, C-8
```

---

## 5. Schema

No new tables. Run 0 uses the existing omni_lab tables (`run`, `run_device`, `run_model`, `run_dataset`, `dataset_version`, `metric`, `run_event`) through `omni_lab.py log-run`, with raw psycopg2 and parameterized SQL. No ORM, Pydantic or SQLAlchemy anywhere.

- Each manifest bump inserts one `dataset_version` row and one `run_dataset` link.
- The manifest is JSON on disk, mirrored to omni_lab.
- An omni_lab DB backup (agent-memory-backup pattern) is taken before launch.

---

## 6. Trade-offs

| Chose | Rejected | Why |
|---|---|---|
| Post-trained Qwen3.5-2B | Base | behaviour retention is the gate; caches are chat-templated |
| λ 0.9 on `ffn.down` only + σ 0.05·rms on up/down, blocks 7-23, + WD 0.1 | λ on gate, up and down (≈0.73× output); global 0.2-0.4; FIRE | retention, not time to scratch loss; FIRE goes to the fold pass |
| Stage-0 bridge alignment from Qwen's own tower, then discarded | random bridge | the tower lands in exactly the planted trunk's input space at our 32 px granularity; the end model stays tower-free |
| Qwen im_start format everywhere | band `<|user|>` roles | the donor and caches already use it; one convention |
| One `<|audio|>` + kind header | separate speech/music tokens | Cyril: one token per modality |
| `<|out|>…<|/out|>` + `<|edit|>` | one token per combination | combinations compose; 9 rows, not 2^5 |
| 512 long side for generation, 768 for understanding | 1024² on day 1 | budget at about 46M positions/day **(projection)** |
| x-prediction + velocity-space weighting | x0-MSE | the §3e lever; cheap |
| LPIPS/REPA off | on | VRAM, Triton risk, weak evidence |
| K3 1.0 where present; Strata primary at volume (pending D1) | trainer defaults Q38 1/Strata 1/K3 0.3 | binding ruling; K3 cannot grow without GPU6 |
| DDP + Adafactor fp32 through a gloo host all-reduce | FSDP/ZeRO, NCCL, adamw-offload | fits per card; no P2P; RAM headroom for Strata |
| Pause the judge (D2a) or 3 cards (D2b) | judge on CPU | a 27B Q4 on CPU takes about 16 GB of RAM and memory bandwidth from Strata's offloaded layers |
| Strata through a standalone gate client (D3) | editing gen-worker-text | standing rule: never touch gen-worker-* |
| Epoch caps + reallocation | fixed shares | 23 video clips at a 6% share would be about 40 epochs/day |
| 150-step warmup, stable phase | 1-2k steps, cosine | about 330-400 s/step; the run extends as data grows |
| Sequential video-then-audio suffixes | joint two-span refinement at launch | 18 same-origin clips; joint mode is C-7 |
| Supervisor auto-resume | trainer alone | a dead rank would otherwise idle 4 cards for days |

---

## 7. Risks and what we watch

- **Touches production / needs approval:**
  - D1-D4;
  - the 4.5 GB download to root;
  - the server ruling file;
  - pausing `media-judge.timer` (D2a), after which no new judged media reaches intake;
  - TTS jobs in the media-gate idle queue (D4), which compete with Sprite's idle generation;
  - holding GPU1/2/3/5 for days (other lanes are refused; Relay notice L3);
  - about 56 GB host RAM next to Strata's 46 GB and ComfyUI's 17-34 GB swing.
  - Never GPU0/4/6, never their services, never NCCL or P2P.
- **Backups first:** `plant/`, `ablate/` and `ablate+align/` are immutable with chained receipts; omni_lab is backed up before launch.
- **Text retention is the pass/fail**, read in the per-checkpoint dumps. The top-10 overlap alarm (> 20% relative drop) pauses for a read. Response: trunk LR 1e-4 and the flow ramp from 0.2. Show-o2's collapse on raw web text is the failure mode; our text share is curated KD only.
- **Generation may stay blobby at 512 for days.** Rounds 4-7 failed edit, video and speech on toys. Grade by looking at the dumps. LPIPS and 1024² are the stage-2 levers.
- **Throughput is projected**, not measured, at 4 cards, and GPU5 crosses QPI. Below 350 positions/s in the first hour: accum 8, or 3 cards.
- **Resume** is a hard gate (C-1). A lost card needs `--allow-world-change`.
- **Attention memory:** if the efficient kernel refuses on sm_61, the chunked fallback is slower. Its cost is measured in S0/S3, and the long lane stays off until proven.
- **Teacher coverage:**
  - cached text runs about 1.1 days per epoch at a 22% share;
  - with the 6-epoch/week cap, text KD saturates around day 6 **(projection)** unless D3 adds Strata signal;
  - K3 is fixed at 1.89M positions without D1(ii).
- **Data thinness:**
  - video-out is a token presence only;
  - image edit is 45 judged triples (procedural edits need D3);
  - concept-interleaved needs D4;
  - there are no sound effects, no video understanding and no real joint audio-video until approved downloads arrive.
- **Licences:** MAESTRO is CC BY-NC-SA (recorded on the run); COCO is mixed Flickr (Cyril ruled licences fine on 10-14); Qwen3.5-2B is Apache-2.0.
- **Unlocked containers:** `comfyui_qwen21_gpu1/2/5` bypass the locks. The foreign-process alarm catches them.

---

## 8. Critic findings: applied and rejected

| # | Finding | Verdict | Where |
|---|---|---|---|
| 1 | B-6 edits gen-worker-text (standing-rule breach) | **Applied.** B-6 is now a standalone gate client, gated on D3; run 0 launches without it. | §0 D3, B-6 |
| 2 | K3 weighting contradicts the binding ruling; ratify provisional bf16 | **Applied.** D1 is launch-blocking with options (i)/(ii)/(iii); L1 records ratify or revoke. | §0 |
| 3 | A CPU judge eats Strata's RAM headroom | **Applied.** No CPU judge; D2 = pause or 3 cards. | §0, §6 |
| 4 | Claude-authored templates and unknown-author prompts | **Applied.** B-0 audit; conditioning-only rule; phrasings from COCO or Strata only. | §2.5, B-0 |
| 5 | Path-level guard misses rows and turns | **Applied.** Row- and turn-level guard plus `loss_mask`, with an acceptance test on a real shard. | §2.5, B-1, §3.2 |
| 6 | TTS rendering vs the small-model rule | **Applied.** D4 + L2 ruling line citing 10-14; idle queue only. | §0 |
| 7 | vision_in bucket has nothing to train on | **Applied.** Dropped. | §2.5 |
| 8 | Lane override not on record | **Applied.** Recorded in the ruling entry; L3 Relay notice. | §0, rulings file |
| 9 | FFN shrink is about 0.73×, not 0.9× | **Applied.** λ on `ffn.down` only; residual-RMS probe in the receipt. | §2.2, A-4 |
| 10 | Attention fallback OOM at 8k/16k | **Applied.** Kernel enforced, chunked fallback, S0; 16k lane off until proven; S3 adds 16k and video steps. | §2.6, A-8, §2.8 |
| 11 | Host RAM at resume/broadcast | **Applied.** Staggered mmap loads, tensor-wise broadcast, RSS in S4. | §2.6, C-1 |
| 12 | Memory floor only at launch | **Applied.** Per-step MemAvailable watchdog → checkpoint and pause. | §2.6, C-5 |
| 13 | Triton, bf16 and ffmpeg traps | **Applied.** Rank-startup asserts; K3 bf16 → fp32 on CPU. | §2.9 |
| 14 | Tiny sets repeated many times | **Applied.** Per-source epoch caps (per rolling 7 days) with reallocation; video-out stated as token presence. Modification: caps are per 7 days rather than per run, because the run has no fixed end. | §2.5 |
| 15 | Caption style damages text | **Applied.** LibriSpeech normalised and ASR-only; COCO with caption-style requests only; media CE 0.3. | §2.4, §2.5 |
| 16 | No sound effects, joint AV or video understanding | **Applied.** Stated as absent; downloads listed for approval. | §0, §2.5 |
| 17 | Intent bank has no captions Strata can see | **Applied.** COCO + generation prompts only. | §2.5 |
| 18 | Tokenizer-agnostic dataset identity | **Applied.** | §2.5, §3.3 |
| 19 | NAS reads can hang the loader | **Applied.** Staged off the NAS before launch; read timeouts. | §2.5, B-5 |
| 20 | A modality silently vanishes | **Applied, modified.** Alert at 20 steps, halt at 60 (not at 20): a halt idles 4 cards, so a short transient should not trip it. | §2.7 |
| 21 | Dead rank → idle GPUs for days | **Applied.** Supervisor C-9. | §2.7 |
| 22 | Dump on rank 0 hits the gloo timeout | **Applied.** Split across ranks, timed in S3, < 900 s budget. | §2.7, C-4 |
| 23 | Replicas drift after the first 3 steps | **Applied.** Per-checkpoint replica sha with broadcast-on-mismatch. | §2.7, §3.5 |
| 24 | Nobody reads the dumps in time | **Applied.** Top-10 overlap alarm pauses for a read; media-collapse alarm. These are alarms only; pass/fail stays behavioural. | §2.7 |
| 25 | metrics.jsonl, disk floor, thermals, ECC, foreign processes | **Applied.** | §2.7 |
| 26 | Mid-run intake bumps | **Applied.** Guard + hold-out dedupe + synchronised switch at the barrier. | §2.5 |
| 27 | Image input has no bootstrap | **Applied as stage 0 (A-7), images only.** Video alignment rejected: the tower's 2×32×32 tubelet does not match our 2×32×16. Unused teacher caches stated plainly. | §2.4a |
| 28+ | not received (the critic text was cut off in item 27) | n/a | n/a |

Rejected outright: none. Two were applied with a modification (14 and 20), and one with a narrower scope (27), for the reasons given in the table.

---

## 9. Architecture-candidate verdicts (from `ARCH_CANDIDATES_2026-10-22.md`)

No candidate is adopted in run 0.

| Item | Run 0 | Fold pass | Later |
|---|---|---|---|
| MemT-1.0 | no | no | no: conflicts with "memory is tools", slows decode, assumes a frozen trunk |
| Sage-A1.5 | no | no | no, as architecture, teacher or data; coded exclusion in B-1 |
| DeepInfer engine / DSpark / half-depth prefill | no | no | FP4 KV trained in: a small arm inside S6 QAT, judged on 32k behaviour; half-depth prefill only if a phone measurement shows media prefill is the bottleneck |
| TiDAR one-pass layout | no | no | no |
| NLD-style self-speculation (A-DIFF) | no | no | only after the phone microbenchmark on the existing Qwen3.5-2B passes (cost(k=8) < 2.5× cost(k=1)) |
| Fused commit + next-chunk media refinement | no | no | phone inference-scheduling experiment, no training change |
| `<|mask|>` row | reserved, id 248336, untrained | n/a | used only if A-DIFF ships |

**Key paths:**
- Repo: `src/chimera_omni/{config.py, v2_layout.py, importing/qwen35.py, media/training.py, modeling/{layers.py, chimera.py}, training/{v2_trainer.py, run0_data.py, media_sources.py, run0_samples.py, text_remap.py, run0_dump.py, run0_watch.py}}` and `scripts/{plant_qwen35_2b.py, ablate_run0.py, align_bridge_stage0.py, attn_kernel_check.py, run0_build_manifest.py, run0_stage_data.sh, run0_intake_sync.py, strata_writer_client.py, run0_queue.sh, run0_supervisor.py}`.
- Server:
  - `/home/cereal/models/Qwen3.5-2B`
  - `/home/cereal/omni-campaign/run0/{plant, ablate, ablate+align, data, run}`
  - `/home/cereal/omni-campaign/rulings/arch_v2_init_source.json`
  - `/home/cereal/omni-campaign/chimera-omni-v2/`
- Reports: `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/run0-20261022/`.

---

## Addendum, 2026-10-22: Cyril's decisions on §0 (ruling `cyril-run0-decisions-2026-10-22`)

| # | His answer | What runs |
|---|---|---|
| D1 | "[Cyril quote withheld]" | Option (i) plus the (iii) repetition cap. K3 cache only; Strata is the volume teacher. |
| D2 | "[Cyril quote withheld]" | The media judge moves to Strata at background priority (vision enabled first if missing), then GPU5 joins run 0. Until the swap lands, `media-judge.timer` is paused for the run, which is option (a). |
| D3 | "[Cyril quote withheld]" | YES. A standalone strata-gate client at background priority writes the intent bank and edit instructions and captures n_probs=64. Its output goes through the versioned intake. |
| D4 | "4 sure" | YES, TTS caption renders through the media-gate idle queue. |
| Audio gap | "fix the audio gap" | Downloads APPROVED for real sound effects and paired audio-video: VGGSound, AudioCaps, Clotho or equivalents. |
| Corpus | "[Cyril quote withheld]" | Our canonical corpus, the curated set, the tool-call corpora, Cyril's speech and the NAS PDF corpus are added as versioned derived views. |

S5 launch is unblocked on D1 and D2. Data from D3, D4, the audio gap and our corpus streams into the running job through the versioned intake as it lands.
