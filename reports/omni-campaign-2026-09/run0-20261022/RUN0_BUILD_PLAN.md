# RUN 0 build plan: co2-2b planted from Qwen3.5-2B, ablated, distilled on all modalities on 4 P40s

Status: FINAL plan, revision 2, 2026-10-28. It replaces revision 1 (2026-10-22, backed up as `backups/RUN0_BUILD_PLAN.20261022.md`, sha256 prefix `a5cbc3c7`). It folds in the 10-28 surveys (`survey_code.md`, `survey_data.md`, `survey_hw.md`, `survey_lit.md`), `ARCH_CANDIDATES_2026-10-22.md` rev 2, the 10-28 draft and the 10-28 critic. Each critic item (1-29) is applied, modified or rejected with a reason in §8. The critic text that reached this pass stopped part-way through item 29, so any later items were never received.

Rulings in force: `cyril-run0-plant-qwen-2026-10-22`, `cyril-run0-decisions-2026-10-22`, and Cyril's operating words of 2026-10-28, recorded as `cyril-run0-ops-2026-10-28`. All three are in `distill/STAGE_D_RULINGS_2026-10-12.md`. Code: the three run-0 branches (`claude/run0-a-20261022` 4a53d3a, `-b` 96ed2b9, `-c` e52fa93), all forked from `claude/arch-v2-20261021` 8d8f064, are merged into one integration branch in Step 0. Worktrees are under `C:/Users/cereal/Documents/ai training/.agent-worktrees/chimera-omni-5d4d4e1dcd/workers/claude/`. Local git only, never push. Commit messages end `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

Every number below is either measured in a cited survey or marked **(proj)**. Pass/fail of the model is intelligence retention judged on behaviour by a person reading dumps. Numbers about the model are recorded diagnostics. Only ops numbers (VRAM, RAM, disk, tok/s, finiteness, replica identity) can stop or gate the run, because they decide whether the run finishes, not how good it is.

---

## 0. What is settled, and what still needs Cyril

**Cyril, 2026-10-28 (relayed verbatim):** "[Cyril quote withheld]"

What this settles:
- **10 days on GPU1/2/3/5** under `lanes.py claim-run`. GPU0, GPU4 and GPU6 stay Sprite's and Strata's.
- **The NAS cleanup is authorised.** O-1 does it by category (§4, Ops).
- **Strata RAM levers are authorised "if necessary".** The order and the measurement are in §2.6.
- **"Make damn sure it completes" is the design goal of §2.8 and §2.10.** Every way the run can die is either proven in the smokes before the 10 days start, or caught by an alarm within the first 24 h.

Settled earlier, on 10-22 (`cyril-run0-decisions-2026-10-22`): D1 K3 as cached, D2 judge to Strata in the back lane, D3 Strata generates data at background priority, D4 TTS renders, the audio-gap downloads, and "[Cyril quote withheld]".

**Items Cyril must hear before S5.** None of them blocks the build, and each has a default.

| # | Item | Default | Why he must hear it |
|---|---|---|---|
| Q1 | **K3 share is below 1% of KD signal.** K3 has 1.89M cached positions; run 0 trains about 330-460M positions **(proj)**. "[Cyril quote withheld]" (D1) allows this. | Proceed. Record it as an explicit deviation from `cyril-full-distill-no-lora-2026-10-17` ("K3 primary"), for run 0 only. | It is not K3-primary, and the plan must not pretend it is. |
| Q2 | **The media judge on Strata may not fit.** GPU0 is at 24.1 of 24.6 GiB, and no vision projector is loaded (critic 27). | O-4 measures whether a projector exists and fits without slowing Sprite. Until then `media-judge.timer` is paused for the run, and judged media stops reaching intake (2,341 groups already wait). | His D2 said "fix that first". If it costs Sprite VRAM, he decides, because "[Cyril quote withheld]". |
| Q3 | **Strata prompt-cache change.** The 48 GiB RAM prompt cache moves to a disk tier, or gets capped (§2.6 R3). | Disk tier if the build supports it. Otherwise cap at 16-24 GiB. | Sprite's 66k-context prompt reuse gets slower. |
| Q4 | **Docker restart policies** on P40-bound containers (`comfyui_qwen21_gpu1/2/5`, `chimera-p40-*`, `ice_roster`, `nervous_galileo`) set to `no` for the run (critic 21). | Ask. If he says no, the foreign-process alarm is the only defence after a reboot. | It changes container configuration. |
| Q5 | **Root-disk deletions:** the two retired 27B GGUFs (32 GB), and optionally the 83 GB IQ3_S Strata rollback. | Not needed: §2.7 fits without them. Offer them as margin. | Irreversible. |
| Q6 | **No 50k-context training in run 0.** It needs C-8 activation offload. 16k is trained once S0 passes. The native positions (theta 1e7, 262,144) carry the capability. | State it. | His 50k minimum is a phone requirement, and run 0 does not train it. |
| Q7 | **Data that run 0 does not have:** no sound effects and no real joint audio-video until the approved downloads (VGGSound / AudioCaps / Clotho) land; video understanding only from Strata-written generation prompts (§2.5); edit triples in the hundreds. | State it. | "All modalities" would otherwise overclaim. |

Lead decisions (Claude, not Cyril's words), recorded in the rulings file before S5:
- **L1:** ratify the provisional bf16 K3 cache for run 0 if the cardclass verdict and `k3_b2_numerics_ruling.txt` hold and the cross-map matched mass is at least 0.90 (measured 0.91). If revoked, the K3 bucket goes to 0 and text KD takes its share.
- **L2:** the D4 TTS ruling line.
- **L3:** the Relay lane notice. GPU1/2/3/5 are held for about 10 days under `cyril-run0-plant-qwen-2026-10-22`.

---

## 1. Problem

Cyril ordered a first-pass 2B Chimera Omni v2:
1. plant Qwen3.5-2B into co2-2b;
2. ablate it mildly;
3. distill all parameters across every modality on our architecture, on all four P40s, for about 10 days, extended as data streams in;
4. no proxy test campaign.

The fold pass comes next and is compared against run 0.

On 2026-10-28 none of that can run:
- **Code:** it sits on three unmerged branches that have never run together or on a GPU. Merged, they have two contract breaks (`turn_open` tokenizer argument).
- **Weights:** Qwen3.5-2B is not on disk.
- **Data:** there is no sealed `run0_data_v1.json`. The 10-22 data parts miss the 10× intake growth and 460 h of LibriSpeech.
- **RAM:** host MemAvailable is 28 GiB. Four ranks need about 44 GB, because each rank pins a full gradient mirror.
- **Disk:** the NAS is 100% full; root has 113 GB.
- **GPU5:** the hourly judge claims it.
- **Resume:** the resume fix has never run on a GPU.

This plan makes the run start, and makes it provable that it will finish before the 10 days are committed. The build splits into three areas, **A**, **B** and **C**, with disjoint file ownership and written interfaces (§3), plus ops items **O** done by the lead.

---

## 2. Design

### 2.1 Model config and plant (area A)

**Profile `co2-2b-run0`** (on run0-a). The geometry is identical to `co2-2b`. Metadata: `init_rule = V2_INIT_RULE_RUN0`, `PLANT_RULING = "cyril-run0-plant-qwen-2026-10-22"`, `init_source = "Qwen/Qwen3.5-2B@<safetensors sha256>"`.

| Field | co2-2b-run0 | Qwen3.5-2B (HF config, read 10-28) | Change |
|---|---|---|---|
| Layers | 24: 18 GDN + 6 gated GQA, pattern `kda,kda,kda,gated_gqa` | 24, `full_attention_interval` 4 | none |
| Width / FFN | 2048 / 6144 SiLU-GLU | same | none |
| GQA | 8 q / 2 kv × 256, QK-norm, sigmoid out-gate, partial RoPE 64 dims, theta 1e7, mRoPE (11,11,10) | same | none |
| GDN | 16/16 × 128, conv 4, scalar decay | same | none |
| Vocab | 248,320 + 512 control = 248,832, tied | 248,320, tied | +512 rows |
| Context | 262,144 | 262,144 | none |
| Depth mixing | Block AttnRes, block 4, `attnres_init="zero"` (function-preserving, 2.6e-4 logit error, test-pinned) | n/a | new tensors |
| Raw boundary | 3,072-value untyped units; A6b `native_unit_norm="rms"` | n/a | new |
| x0 head | RMSNorm → GLU 2048→4096→3072, last linear zero-init | n/a | new |
| CFG condition drop | **0.1 on generation samples** (critic 24; done in data, §2.4) | n/a | changed from 0.0 |
| Off | A8 reverse GDN scan, x0 skip, MTP heads, aux heads | | |
| Params | 1,923,165,809 (measured build) | 2,274,069,824 in the repo | |

**Donor: post-trained `Qwen/Qwen3.5-2B`** (sha `15852e8c`, Apache-2.0, not gated, one BF16 file of 4,548,221,488 B), not `-Base`. Reasons:
- Retaining chat and tool behaviour is the pass criterion.
- All 72M cached teacher positions were rendered with the chat template.

Download to `/home/cereal/models/Qwen3.5-2B` on root. Never the NAS.

**Plant mapping** (`importing/qwen35.py` + `importing/run0_plant.py`):

| Student tensor | Source | Rule |
|---|---|---|
| `embed.weight[0:248320]` (tied LM head) | `model.language_model.embed_tokens` | exact fp32 copy; tie asserted |
| `embed.weight[248320:248337]` (17 used control rows) | ids 0-7 existing + 9 new, seeded per §2.3 | + N(0,(0.02·rms)²); never zero (in a tied table a zero row is a zero input and a zero logit) |
| `embed.weight[248337:248832]` (495 spare rows) | mean text row + noise | **never emitted:** masked to −inf at the head (§2.3, critic 29) |
| norms: 24 × input/post-attn, final, q/k | Qwen `(1+w)` | folded to a plain weight, exact |
| 24 × `ffn.up` (= gate_proj ‖ up_proj), `ffn.down` | Qwen MLP | exact; the only tensors the ablation touches |
| 18 × GDN: q/k/v split of `in_proj_qkv`; 3 depthwise convs from `conv1d`; `in_proj_z`→gate, `in_proj_b`→write, `in_proj_a`→decay; `dt_bias`, `A_log`, `out_proj`; per-head norm | Qwen `linear_attn` | exact |
| 6 × GQA: `q_proj`/out-gate de-interleaved from the fused q; k, v, o | Qwen `self_attn` | exact; text mRoPE is t=h=w, so the import is exact |
| `*.attention_residual.*` | new | zero init |
| `native_media_bridge.*` (10.66M) | new | repo init; trained first in stage 1 (§2.4a) |
| `native_media_head.*` (29.36M) | new | repo init; last linear zeroed, so media output at step 0 is the signal mean |
| `model.visual.*` (297 tensors, about 0.30B) | **not loaded into the student** | the end model is free of encoders and towers; used only as a frozen stage-1 alignment teacher for image input, then discarded; shapes and hashes go in the receipt |
| `mtp.*` (15 tensors) | **dropped** | `mtp_heads=0`; MTP and speculative decode measured a net loss on the phone; hashes go in the receipt |

**Route:**
1. `load_qwen35(residual="none")`. The importer refuses AttnRes.
2. `load_state_dict(strict=False)` into `co2-2b-run0`.
3. Assert the missing keys are exactly AttnRes + bridge + head, and nothing is unexpected.
4. `plant_function_drift` against HF transformers on 512 prompts (recorded).
5. Write `plant/{receipt.json, child/}`.

The first action after download is `scripts/plant_qwen35_2b.py --check-only`, a header-only geometry check. The 10-28 survey already matched the real config field by field.

**Run-0 tag (critic 2).** Every checkpoint config and receipt carries `init_rule = V2_INIT_RULE_RUN0`. The fold-pass loader (`load_fold_base(kind="fold")`) **refuses** any layout carrying that tag, so planted weights can never become a fold base by accident. A test pins this.

### 2.2 Ablation recipe (survey_lit §1.1-1.4)

First clone `plant/` → `ablate/`, with a chained receipt carrying the parent sha. Seed 20261022. Script `scripts/ablate_run0.py` (on run0-a; its defaults match).

| Step | Tensors | Magnitude | Why |
|---|---|---|---|
| Shrink | `ffn.down` only, blocks 7-23 (17 tensors) | w ← 0.90·w | Scaling only `down` scales the GLU output by exactly 0.9; scaling gate+up+down would give about 0.73×. Published λ 0.2-0.4 (Verwimp 2025, Mallik 2026) optimises time to scratch loss, not retention. Springer 2025 (ICML) shows heavily overtrained donors are the most perturbation-sensitive. |
| Perturb | `ffn.up` and `ffn.down`, blocks 7-23 (34 tensors) | w ← w + N(0,(0.05·rms(w))²) | low end of 0.05-0.1; FFNs hold most parameters and are where new modality features get written |
| Untouched | embedding/head table, every norm, all 18 GDN and 6 GQA tensors, blocks 0-6 (29%), AttnRes, bridge, x0 head | | Spurious Forgetting (ICLR 2025): bottom layers and embeddings carry alignment |
| Collapsed-head probe | 48 GQA heads on 512 mixed sequences | reinit Q/K/V rows and zero O columns only if entropy < 0.1 nats or > 0.95·log(ctx), and only if the count is ≤ 4; otherwise record and skip | the source is second-hand; expected count 0 |
| Receipt probe | per-block residual RMS ratio, hidden cosine, final top-10 overlap, plant vs ablate | **recorded, not a gate** | pass/fail is the S2 read |
| Rejected | global λ ≤ 0.5, FIRE, DASH, norm resets, dead-neuron resets | | FIRE and DASH are fold-pass candidates; the rest harms retention |

The plasticity levers that do the real work are in training: decoupled weight decay 0.1 on trunk matrices (ICML 2026) and the LR re-warm (§2.7).

**S2 rollback:** if the 8-prompt read after ablation is visibly broken, delete `ablate/`, re-clone, and apply λ 0.95 on `ffn.down` of blocks 12-23 only, with no noise (`--rollback`).

### 2.3 Marker tokens, the output-decision group, and the head mask

Control band (run0-a `V2_CONTROL_TOKENS`). Ids 0-7 are unchanged and test-pinned.

| id | token | role | row init |
|---|---|---|---|
| 248328 | `<|text|>` | marks every text piece, in and out | mean embedding of the tokens of "text" |
| 248329 | `<|image|>` | marker and span opener | Qwen `<|vision_start|>` |
| 248330 | `<|audio|>` | marker for speech, music and sfx; the kind goes in the header text (`speech 38400hz mono`) | mean of "audio" |
| 248331 | `<|video|>` | marker and opener | Qwen `<|video_pad|>` |
| 248332 | `<|3d|>` | marker and opener | mean of "3d" |
| 248333 | `<|out|>` | opens the decision | mean of the five marker rows |
| 248334 | `<|/out|>` | closes it | same |
| 248335 | `<|edit|>` | decision modifier; precedes the modality it edits | mean of "edit" |
| 248336 | `<|mask|>` | reserved, untrained | mean text row |

Each seeded row gets + N(0,(0.02·rms)²). Qwen special ids are resolved from `tokenizer.json` at runtime, never hardcoded.

**Chat format:** Qwen `<|im_start|>role\n … <|im_end|>` everywhere.

**Head mask (critic 29).** `EMITTABLE_CONTROL` = the 5 markers, `<|out|>`, `<|/out|>`, `<|edit|>`, `<|/hdr|>` and `<|/media|>`. Every other control id gets logit −inf in the head, in training (CE, KD normalisation) and at sampling. That covers the 495 spare rows, `<|mask|>`, `<|raw|>`, `<|media|>`, and the band's `<|user|>`, `<|assistant|>` and `<|eos|>`. Because the table is tied, those rows would otherwise each give a mid-sized logit and together pull probability from text at step 0. Masked, their init cannot affect text behaviour. They still work as inputs where used: `<|raw|>` is the placeholder input row.

Grammar (`<|/hdr|>` ends a header, `<|/media|>` closes every span):

```
text in     <|im_start|>user\n<|text|>what is this?<|im_end|>
image in    <|im_start|>user\n<|image|>image 512x384<|/hdr|><|raw|>x192<|/media|><|text|>caption it<|im_end|>
text out    <|im_start|>assistant\n<|out|><|text|><|/out|>It is ...<|im_end|>
image out   <|im_start|>assistant\n<|out|><|image|><|/out|><|image|>image 512x512<|/hdr|><|raw|>x256<|/media|><|im_end|>
edit out    <|out|><|edit|><|image|><|/out|> + span      (the source image is a clean span in the user turn)
mixed out   <|out|><|text|><|image|><|/out|> + text + span, in declared order
AV out      <|out|><|video|><|audio|><|/out|> + video span + audio span  (sequential suffixes; joint = C-7 stretch)
```

**Supervision:**
- Every decision token is a CE target at weight 1.0.
- A generation (x0) forward carries no text labels, so every generation sample gets a **twin causal sample**: the request plus `<|out|>…<|/out|>`, about 12 supervised tokens, in the understanding pass.
- Cached text corpora get `<|text|>` and `<|out|><|text|><|/out|>` inserted at load by `text_remap`, with cache positions preserved.
- The output mode is learned from data. There is no router.

### 2.4 Media at real sizes, generation, losses

| Modality | Unit (3,072 values) | Run-0 sizes | Rows per item | Decode |
|---|---|---|---|---|
| Image | 32×32×3 patch | understanding: long side ≤ 768, aspect-bucketed to multiples of 32 (COCO 640×480 = 300 rows); generation: long side 512 in 5 buckets (512², 512×384, 384×512, 512×768, 768×512) | 192-576 in; 256-384 out (+64-96 draft at 1/2) | PIL, per-dataset gain |
| Video | 2 frames × 3 × 32 × 16 tubelet at 12.5 fps | understanding 256×448, 2 s = 24 frames; generation 192×320, 2 s | 2,688 in; 1,440 out (+360 draft) | ffmpeg/PyAV, 30 s timeout |
| Audio | 3,072 mono samples at 38.4 kHz = 80 ms | speech 2-12 s; music 10 s mono (ACE downmixed; stereo deferred) | 25-150 | polyphase resample: 48k → 4:5; 16k → 12:5; 22.05k (LJSpeech) → 256:147 |
| 3D | 8³ voxels × 6 (SDF, RGB, rough, metal) at 64³ | 64³ only | 512 | hy3d_voxel64 pipeline |

1024² images (1,024 rows) are a stage-2 lever, to use after throughput is measured. 512 is the published raw-pixel regime (JiT); it is not a toy.

**Generation = joint refinement of every unit of the item:**
- One open suffix holds all rows. Each item gets one σ (`item_flow_noise_levels`), logit-normal, shifted by sqrt(rows/256).
- In the 6 GQA layers every row attends bidirectionally within the span and causally to the prefix. The 18 GDN layers stay raster-causal (A8 off).
- Draft-then-refine runs at factor 2 for image and video.
- Dumps sample with 16 Euler x0 steps, at CFG scale 1 and 3. Audio sampling is stochastic per DiTAR.
- The number of sequential passes equals the number of flow steps, not the number of units.
- **Joint two-span refinement** (video plus its audio on the shared 80 ms clock) is C-7, a stretch item and not a launch gate.

**Condition dropout (critic 24).** With probability 0.1, a generation sample keeps its turn structure and decision but drops the request text, so CFG becomes available at sampling time. The twin sample is never dropped. This is a B-area data feature (manifest `generation.cond_drop`), not model code.

**Media loss** (`media/training.py` `flow_loss`, run0-a):
- x0 prediction, weighted in velocity space: `w(σ) = 1/max(σ,0.2)²`, normalised to E[w]=1, with a type-balanced reduction.
- Audio adds the multi-resolution STFT, mel 0.05, level and chunk-boundary terms.
- **Band limit (critic 26):** STFT and mel terms are masked above each item's source Nyquist (`meta.source_sr/2`: 8 kHz for LibriSpeech, about 11 kHz for LJSpeech). The model therefore never learns "speech has no highs" from upsampled sources.
- **The launch line must set `--flow-weighting velocity --sigma-sampler logit-normal`.** The defaults are `none` and `uniform`.
- Off in run 0: LPIPS (VGG in VRAM at a 20.7 GB peak, Triton risk) and REPA/QVL32 (weak evidence).

**Text loss** (`chunked_text_objective`; the head runs on supervised positions only, in 256-position chunks; head mask §2.3):
- CE 0.1 on cached text, 0.3 on media answers, 1.0 on decision tokens.
- 3-gram unlikelihood 0.1.
- Top-K bucket forward KL (top-64 + a residual-mass bucket, a top-n target) per same-vocab teacher, with per-position weights:
  - **where K3 exists:** K3 1.0, Strata 0.3, Q38 0.1;
  - **elsewhere:** Strata 1.0, Q38 0.3.
  - This makes Strata the main teacher by volume (Q1).
- Strata top-n from new gen-worker output is labelled teacher `strata-q4kxl-huihui-v2`. It is never conflated with the IQ3_S cache, and is used only after the C-15 pre/post-sampling check (critic 9).
- Logged every step: top-10 overlap and captured top-10 mass per teacher. Top-1 is never a gate.

#### 2.4a Stage 1: new-modules warm start on every modality (critic 25)

This replaces revision 1's one-card image-only "stage 0". It is the first phase of the real run on all 4 cards.

- **Trunk frozen.** Trainable: `native_media_bridge.*`, `native_media_head.*`, the 512 control rows, and the AttnRes pseudo-queries and norms (about 40M parameters, so the all-reduce is small).
- **Data:** the full §2.5 mixture, every modality in and out, including text KD at ≥ 40%, so the decision tokens and control rows learn.
- **Image alignment target:** for image-input rows, cosine + 0.1·MSE between the bridge's output rows and the frozen Qwen vision tower's post-merge features. The tower outputs 32 px merged tokens in the planted trunk's input space; both sides are RMS-matched to the text rows. Weight 1.0. The tower runs on the card (fp32, about 1.2 GB) during stage 1 only and is then dropped from memory.
- **LR:** new modules peak at 8e-4 with a 50-step warmup. Flow weight 1.0. That is safe here because no trunk weight moves.
- **Budget:** 400 optimizer steps or 18 h, whichever comes first **(proj)**. Its dump at the end is read; captions through a bridge alone may be weak, and that read is recorded, not a gate.
- Stage 1 doubles as the first long soak of the 4-rank job (RSS, gloo, dumps, checkpoints).
- `scripts/align_bridge_stage0.py` stays as a one-card fallback and is not on the launch path.

**Stage 2** is the full distill. It unfreezes the trunk, re-warms the trunk LR (§2.7), and ramps the flow weight from 0.2 to 1.0 over 300 steps, so untrained-head gradients never hit the trunk at full weight.

### 2.5 Data mixture (area B)

Shares are token-position shares per optimizer step. `MixtureAccountant` enforces **text + tool ≥ 40%**, and **the text floor wins every conflict** (critic 8).
- Each source has an epoch cap per rolling 7 days.
- Above its cap, a text source keeps being served as **KD-only repeats** (CE weight 0 after epoch 4; hard ceiling 12 epochs per 7 days). Soft targets overfit less than CE.
- Budget freed by a capped or empty media bucket goes, in order, to COCO understanding, COCO generation, procedural edits, MAESTRO, ModelNet, then text KD.

| Share | Bucket | Sources (provenance) | Cap |
|---|---|---|---|
| 24 | text KD | `distill/corpus/text_clean_v1` (12.35M, Q38 + Flash top-64); the canonical `staging/rows.jsonl` derived view (Cyril: "[Cyril quote withheld]"); intake `strata-reasoning-math-*` (879), `strata-instruction-code-*` (535), `strata-long-context-*` (333); everything after the authorship guard below | 6, then KD-only to 12 |
| 8 | K3 rows | `corpus/k3/pass-0..3` + xtok caches (1.89M K3 tokens, provisional bf16, L1) | 6, then KD-only to 12 |
| 8 | tool calls | `toolcall_flash_v1`, `bfcl_native_v1` (kept per Cyril's 10-17 ruling, minus the BFCL-62 held-out rows and twins), intake `strata-toolcall-*` (2,904); loss only on Strata or public turns | 6 |
| 2 | long lane 16k | `corpus/long` (31 docs); off until S0 passes at 16,384; its share goes to text KD meanwhile | 8 |
| 4 | decision twins + intent variants | derived from every generation sample; the Strata-written intent bank (B-5) | n/a |
| 12 | image understanding | COCO 2017 captions (118k × about 5), caption-style requests only; WIT only if B-4 proves it loads | 4 |
| 12 | image generation + edit | COCO caption → 512 image (bulk); judged t2i + edit intake (4,071 png now); procedural edits with Strata-written instructions | 4 |
| 2 | video understanding | judged intake video whose prompt is confirmed Strata-written by B-0, used as the caption target (CE 0.3); a temporal-order task (which of two halves came first, request phrasing from the Strata bank, answer from the data) | 4 |
| 4 | video generation (+ sequential AV) | judged intake video (224 mp4 + 69 webp) with prompts; H3 set prompts conditioning-only | 4, token presence |
| 12 | speech in/out | **LibriSpeech clean train-100/360 (460 h, 64 parquet blobs on Lexar) for ASR and TTS**: normalised sentence-case text, speaker tag in the header (`speech 38400hz mono spk=ls-<id>`), band-limited loss; LibriSpeech dev/test; LJSpeech 24 h (punctuated, ASR + TTS); Cyril's 382 records | 4 |
| 6 | music in/out | ACE intake (1,206 mp3 + 112 flac), MAESTRO subset staged off the NAS (CC BY-NC-SA, recorded on the run) | 4 |
| 3 | 3D in/out | hy3d_voxel64; 414 judged GLBs voxelised; ModelNet10 | 4 |
| 3 | concept-interleaved | COCO image + caption + a Qwen3-TTS render of the caption in one context, order randomised (D4) | 4 |

Text + tool = 42%. Decision twins add text CE on top.

**10-day supply projection (B-4 acceptance).** `run0_build_manifest.py --project-days 10 --positions-per-day 33e6,46e6` prints, per bucket and day, the share it can still serve under its cap. Launch requires that the text + tool floor holds to day 10 at both rates, counting KD-only repeats. Every media bucket that starves earlier is listed for Cyril (Q7) and is not a launch blocker.

**Exclusion guard** (`training/run0_data.py`, path, row, turn and **authorship** levels; critic 6):
- **Path:** `selfgen`, `tool-selfgen`, `toolcall/rescore`, `qtext_vl`, vision_in answers, `strata-capture-*`, `capture-export`, media-judge captions, `d2_refill`, `refill-codex-v7/v8`, `toolcall-q38`, anything Sage-derived.
- **Row:** the source tag of each record is checked against the same denylist. Leftover `selfgen` (1.0M) and `rescore` (0.7M) inside `text`/`text_clean_v1` are dropped by row, not by file.
- **Authorship, on every turn** of `text_clean_v1`, the `text` registry part (20M positions) and `staging/rows.jsonl`:
  - a turn whose author or model field is Claude, Codex, any Anthropic/OpenAI cloud model, or unknown is context-only (`loss_mask` false) in tool rows, and is dropped from text-KD rows;
  - this is matched by field, never by a known-row list;
  - the harvested memory `tool_call` events (674k Codex, 206k Claude) are excluded wholesale.
- **Prompts:** every image, music, 3D and video intake prompt dated before the Strata gen-workers existed, and every H3/ACE v9 prompt set, is conditioning-only unless B-0 proves a non-agent author.
- The tokenized view's identity is `(corpus_version, tokenizer_sha + template, teacher)`.
- Before sealing, fix the stale "RVN IQ3_S" model label in the post-10-27 intake `_meta` files by relabelling in the manifest, not by editing sealed batches.

**Strata-written data (D3, critic 5).** No new client. Intent-bank requests, edit instructions and temporal-order phrasings go as ordinary jobs into the **existing gen-worker-text queue**, using only job types it already supports. It runs at background priority, and its output arrives as screened, provenance-tagged intake. No gen-worker code, unit or config changes. If the existing job types cannot carry `n_probs`, that capture is deferred and recorded. Run 0 holds its text floor with KD-only repeats instead.

**Streaming intake (critic 20).**
- `scripts/run0_intake_sync.py` runs **outside the trainer**, at most once per 24 h:
  1. scan `/mnt/lexar/corpus-store/chimera-intake/*/<batch>/READY.json`;
  2. guard;
  3. dedupe against hold-out registry v3;
  4. write and verify `run0_data_v{N+1}.json` plus `NEXT_MANIFEST`;
  5. log omni_lab `dataset_version` + `run_dataset`.
- The supervisor adopts it at the next checkpoint with a controlled STOP and `--resume auto --data-config vN+1`. That is the resume path S4 proves; there is no in-process manifest switch.
- On resume, source states are keyed by source sha: new sources start fresh, and removed sources are dropped and logged.
- **Media intake is effectively frozen while the judge is paused (Q2).** Only already-judged items stream in.

### 2.6 Parallelism and memory on 4 P40s, no P2P (area C)

**DDP, one process per card, fp32, gloo all-reduce through pinned host buffers.**
- Adafactor is factored and row-blocked with **`beta1=None` (no first moment)**, asserted at start. Momentum would add 7.7 GB per card.
- Proven at 2 cards (230 tok/s, run 38325).
- Rejected: FSDP/ZeRO (weights + grads = 15.4 GB fit per card), NCCL (it would pick P2P on GPU1-3), adamw-offload (about 110 GB host).

**Card isolation (critic 13).**
- Each rank runs with `CUDA_VISIBLE_DEVICES=<one GPU UUID from the claim>` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, set by the launcher (`--isolate-ranks on`, run0-c), not by Python.
- At start, each rank asserts `device_count()==1`, `get_device_name()=="Tesla P40"`, and that its UUID equals the claimed card's UUID from `nvidia-smi`.
- A card-to-card copy is then physically impossible.

**C-10, bucketed host all-reduce (launch gate).** Today `HostAllReduce` pins a full fp32 mirror of every parameter per rank (7.7 GB). It is replaced by one reusable 256 MB pinned bucket per rank: reduce bucket by bucket, copy back. Expected RSS is about 3.5 GB per rank + about 3 GB decode, about 26 GB for 4 ranks **(proj)**, measured in S3.

**Two passes, two backwards (critic 14).** Per micro-step per rank:
1. the understanding pass (≤ 8,192 positions) runs forward and **backward**, accumulating gradients;
2. then the generation pass (≤ 4,096, row-bucketed; one 2 s video or about 12 images at 512) does the same.

A 16k long-lane row takes a micro-step alone. The main lane is seq-len 4,096. `--accum 4` gives one all-reduce per 4 micro-steps, about 146k positions per optimizer step.

**Attention on sm_61 (critic 15):**
- GQA uses `sdpa_kernel([EFFICIENT_ATTENTION])` with `torch.backends.cuda.enable_math_sdp(False)` at rank start, so a fallback raises instead of silently allocating 8+ GiB per layer.
- The query-chunked online-softmax fallback (chunk 1,024, A-8) builds each chunk's mask from segment ids and span ids. It never builds a dense L×L tensor.
- S0 runs forward + backward with the real document + refinement masks at 4k / 8k / 16k.

Other memory settings: per-layer non-reentrant checkpointing, chunked GDN scan 64, head chunk 256, `expandable_segments`. Batch building is sharded by rank, with a 2-worker decode prefetch and 10 threads per rank. The GPU5 rank is pinned with `numactl --cpunodebind=1 --preferred=1`. Every rank gets `oom_score_adj +500`. Peak VRAM was 20.7 of 24 GB at about 9k positions; S3 must hold under 22.5 GB with the worst micro-step.

**Load-time RAM (critic 11).**
- Init, resume and broadcast all stream **tensor by tensor**: safetensors mmap → device, CPU copy freed.
- Replica hashes are computed on the GPU, one tensor at a time.
- Rank loads are staggered 60 s apart (`--load-stagger-s 60`).
- S4 asserts the peak summed RSS during resume.
- `vm.swappiness` is read and recorded. Changing it is a system setting and needs Cyril's yes. Swap already holds 6 of 7 GiB, 5.2 GiB of it Strata's.

**Host RAM plan** (Cyril's item 2; apply in order and stop as soon as launch MemAvailable ≥ measured 4-rank peak RSS from S3/S4 + the 30 GiB floor, about 56-60 GiB **(proj)**):

| # | Lever | Frees | Touches | Gate |
|---|---|---|---|---|
| R1 | C-10 bucketed all-reduce | about 30 GB vs the old plan | trainer code only | none |
| R2 | Trainer floors from measurement: `--mem-floor-gib 30 --mem-resume-gib 40`, claim floor 30 | n/a | trainer flags | none |
| R3 | **Strata prompt cache to a disk tier** (critic 4; Cyril 10-27 called a disk-backed prompt cache fine): keep RAM tier ≤ 16 GiB, spill to `--slot-save-path` (or the xcache build's equivalent) on root. If the build has no disk tier, cap `STRATA_CACHE_RAM` at 16-24 GiB instead and tell Cyril (Q3) | about 24-32 GiB | `strata-llama` restart on GPU0/6 | gated; measured before/after |
| R4 | **Only if still short: some cold experts read from disk** (Cyril: "[Cyril quote withheld]"). Not a plain mmap. The coldest experts by the xcache's own hit counts leave the resident set. On a miss they are read with `O_DIRECT` `pread` into a pinned bounce buffer the xcache already uses for host→VRAM copies (pageable memory would make `cudaMemcpyAsync` synchronous). This is a change to `scripts/sprite/patches/xcache.patch` with its own Sprite-lane test, not a flag flip. | sized to the shortfall | `strata-llama` rebuild and restart | gated; Sprite decode measured; revert if single-stream decode falls more than 20% from 24 tok/s or prompt read more than 20% from 258 tok/s |

ComfyUI on GPU4 is never touched.

**Page-cache hygiene (critic 10).**
- The trainer reads teacher caches (28.6 GB), LibriSpeech parquet (30 GB) and media with `posix_fadvise(POSIX_FADV_DONTNEED)` after use.
- Checkpoint writes use `fadvise DONTNEED` + fsync under `ionice -c3`.
- So run-0 I/O does not evict Strata's working set.
- If R4 is applied, rolling checkpoints also move off the root SSD (the GGUF and the expert misses live there) to the Lexar.

**Expected throughput (proj; a 4-rank gloo job has never run):**
- stage 2: about 380 pos/s at accum 1 and about 535 at accum 4, i.e. 33-46M positions/day and 330-460M over 10 days;
- stage 1 is faster: no trunk weight gradients, 40M-parameter all-reduce.

Rules:
- If the first hour of stage 2 measures below 350 pos/s, go to accum 8.
- If GPU5 across QPI costs more than 15%, fall back to 3 cards (GPU1/2/3, about 410 pos/s).
- Log `tok_per_s` in omni_lab.
- The ARCH §9 figure of 86M/day is retired.

### 2.7 Schedule, checkpoints, dumps, logging, resume

**Schedule (critic 28):**
- **Stage 1:** §2.4a.
- **Stage 2:**
  - warmup 150 optimizer steps to a **trunk peak of 5e-5**; new modules (bridge, x0 head, control rows, AttnRes queries) at 8e-4;
  - then WSD stable indefinitely, with a 10% cooldown only when branching a release checkpoint;
  - WD 0.1 on trunk matrices only: **never on the tied embedding/head table**, norms, or new modules;
  - grad clip 1.0;
  - loss weights: text KD 1.0, flow ramped 0.2 → 1.0 over the first 300 stage-2 steps.
- Raise the trunk peak to 1e-4 only if the 24 h read shows text intact and media not learning. Re-warming is reversible; collapse is not.

**Control-row LR (critic 16).** A gradient scale cannot give the control rows 16× the trunk rate, because Adafactor normalises per element. Instead the trainer holds the tied table as **two parameters**: `text_rows [248320, 2048]` in the trunk group and `control_rows [512, 2048]` in the new-module group. They are concatenated in forward (input and head), and saved as the single `embed.weight`, so the checkpoint layout is unchanged.

**NaN guard (critic 18).**
- Before every optimizer step, the all-reduced gradients are checked for finiteness on every rank. A non-finite step is skipped with a `run_event`; after 3 consecutive skips the run stops (`--max-consecutive-nonfinite 3`).
- A checkpoint is COMPLETE only if every parameter and optimizer tensor is finite **and** `replica_sha.json` agrees across ranks.

**Checkpoints:**
- Every 50 steps (about 3.8-5.5 h **(proj)**), atomic, sha256 per file.
- `--keep 2` rolling on `/home/cereal/omni-campaign/run0/run/checkpoints/` (about 8 GB each).
- A milestone every 500 steps is **copied** (exFAT: no links) to `/mnt/lexar/omni-campaign/run0/milestones/` and sha-verified.
- Disk floor 60 GB on `/`.
- Bad checkpoints are deleted, not archived.

**Inits on the Lexar (critic 17).** Once its receipt is written and verified, each of `plant/`, `ablate/` and `ablate+align` is moved to `/mnt/lexar/omni-campaign/run0/inits/`. `chmod a-w` is meaningless on exFAT, so immutability is enforced by receipt sha checks at every load. Root keeps only the Qwen download, the data parts and the rolling checkpoints.

Root budget **(proj)**: 113 − 4.6 (Qwen) − 16 (2 rolling) − about 25 (refreshed data parts) − at most 8 (one init in flight) = about 59 GB. That is just under the 60 GB floor, so either the data parts move to the Lexar (B-4 decides by measured size), or Q5's 32 GB GGUF deletion is used. B-4 acceptance is ≥ 75 GB free on root after staging.

A Lexar I/O-error / unmount alarm is added (§2.7 alarms).

**Replica integrity:** a per-checkpoint `replica_sha.json` (params + optimizer per rank, hashed on the GPU), broadcast-on-mismatch from rank 0, and a 3-step gradient-slice compare after any resume.

**Sample dumps (critic 12):**
- Every checkpoint, after the checkpoint is written, outside any collective. Each rank runs its share, with a budget of **600 s per rank**, then one barrier.
- The share: 8 fixed text prompts (greedy 256, 2 tool calls); 2 decision probes ("caption it" vs "redraw it as a sketch" on the same image); 4 images at 512 (16 steps, CFG 1 and 3) + 1 edit; 2 TTS wavs; 2 music clips; 1 video 2 s mp4; 1 3D (slices PNG + GLB); a contact sheet.
- Synced every 6 h through the server-manager client to `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/run0-20261022/samples/`.

**gloo and heartbeat (critic 12):**
- `init_process_group("gloo", timeout=1200 s)` (`--dist-timeout 1200`), which is longer than one dump share + checkpoint write.
- The supervisor kills the whole group when `metrics.jsonl` stops growing for longer than max(3 × median step, 1,500 s), then resumes. A rank killed by OOM or `kill -9` therefore never leaves the others hung.

**Logging (critic 23):**
- The trainer writes `run/metrics.jsonl` and `run/events.jsonl` only.
- A separate `scripts/run0_lab_replay.py` process replays them into omni_lab with psycopg2 `connect_timeout=5` and `statement_timeout=10000`, idempotent by (run, step, key). A database stall can never stall a step.
- Recorded in omni_lab:
  - `log-run start --kind distill` with notes naming both deviations (Q1 K3 share; plant under `cyril-run0-plant-qwen-2026-10-22`, run 0 only);
  - `run_model`, `run_dataset` per manifest;
  - `metric` every 10 steps: every loss term, top-10 overlap per teacher, tok/s, peak VRAM, all-reduce s, realised shares, epochs per source, per-rank RSS;
  - `run_event` for stages, checkpoints, dumps, manifest bumps, alarms and restarts.
- Raw parameterized psycopg2 (vault key: `omni_lab_pg`).

**Resume:**
- The C-1 fix: CPU-mapped loads, rank-0 tensor-by-tensor broadcast, sha assert before step 1, staggered loads, `--allow-world-change`.
- `run0_supervisor.py` holds the claim and restarts the group from the last COMPLETE checkpoint, with 2 / 5 / 15 min backoff. It gives up after 4 restarts in 6 h, with `ALERT` + Relay.

**Reboot survival (critic 21).**
- The supervisor runs as a systemd unit (`run0-supervisor.service`; installing it is a gated system change). It re-acquires the claim at boot and keeps the claim's flock FD open across group restarts, so the media judge has no window to grab GPU5.
- Container restart policies are covered by Q4.
- The 3090 tensor-rate unlock is armed for the next full power-off. A power cycle mid-run therefore also changes Strata's cards, and the soak probes re-baseline after any reboot.

**Alarms** (never model-quality gates). Each one writes `run0/ALERT` and sends a Relay note.

| Alarm | Trigger | Action (critic 19) |
|---|---|---|
| top-10 overlap | relative drop > 20% from step 0 on the fixed held-out slice | protect the last good checkpoint (exempt from rolling deletion, copied to the Lexar, at most 3 protected); keep training with the trunk LR halved; roll back only on the reader's verdict |
| media head collapse | sample std < 10% of data std | same as top-10 |
| modality starvation | realised share < 50% of target for 20 steps after caps | alert; halt (`halted-data`) at 60 |
| non-finite | 3 consecutive skipped steps | hard stop; supervisor resumes from the last COMPLETE checkpoint once, then stops |
| foreign process | any non-trainer process on GPU1/2/3/5 | hard pause |
| RAM / disk floors | MemAvailable < 30 GiB; `/` < 60 GB; Lexar I/O error or unmount | checkpoint and pause; supervisor resumes at 40 GiB free |
| GPU thermals / ECC | throttling; retired pages | alert; preflight refuses on ECC |
| RSS leak | per-rank RSS growth > 1 GB per 24 h | alert at 24 h; the soak decision (§2.10) |

**Strata impact probe (critic 22).** Every 30 min, `run0_watch.py` reads `strata-llama`'s `/metrics` and `/slots` timings, and the gate's latency log, with read-only GETs. It never sends a request, so it neither queues behind the gen-workers nor preempts Sprite.

### 2.8 Smoke checks before the long run (launch safety, not a test campaign)

| Check | What | Passes when |
|---|---|---|
| S0 attention kernel | one P40, GQA path at 4,096 / 8,192 / 16,384, document + refinement masks, fp32, head 256, fwd + bwd, math SDP disabled | the efficient kernel is selected or the chunked fallback is active; peak VRAM recorded; the 16k lane is enabled only if it passes at 16,384 |
| S1 plant speaks (before ablation) | import receipt + 8 greedy prompts (chat, 2 tool calls) vs HF transformers on the Pascal venv | the lead reads all 8 as coherent and the greedy text is identical to HF. A text difference is treated as an import (plumbing) bug and fixed, the same functional standing as G0/G1 in the 10-12 rulings. Logit diff and argmax agreement are recorded diagnostics |
| S2 after ablation | the same 8 prompts + the receipt probe | read coherent; otherwise roll back (§2.2) |
| S3 4-GPU smoke | `run0_queue.sh --dry-run`, then `--smoke` on a scratch copy of the init: 10 stage-1 steps, then 100 stage-2 steps at accum 1. It includes one micro-step with a max-row video generation item **and** a full 8,192 understanding batch, one 16k step if S0 passed, then 5 steps at accum 4. Discarded afterwards | peak VRAM < 22.5 GB on every rank; every term finite; host RSS per rank measured (sets the floors); tok/s measured; one dump produced and opened within 600 s per rank. Loss trends are recorded, not judged |
| S4 resume | STOP at 100, `--resume auto` for 20 steps, then `kill -9` one rank mid-step; the heartbeat or exit triggers the supervisor; it resumes. Plus one controlled manifest bump v1 → v1b through `NEXT_MANIFEST` | identical `replica_sha.json` at step 120 on every rank; summed host RSS during resume < 60 GB; no hung rank after the kill (group killed within the heartbeat window); the bump adopted |
| S5 launch | stage 1 on all 4 cards, then stage 2 at accum 4 | n/a |

### 2.9 Runtime guards

- **Python:** `/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260` only.
- **Rank-start asserts** (`--runtime-guards on`):
  - `torch._dynamo` is off;
  - no `triton`;
  - no `pydantic` / `sqlalchemy` / ORM module in `sys.modules`;
  - ffmpeg/PyAV are present;
  - `enable_math_sdp(False)`;
  - Adafactor `beta1 is None`;
  - single visible card, a P40, UUID matched.
- **K3 bf16 cache:** converted to fp32 on CPU.
- **Never touch:** GPU0/4/6 and their services. No NCCL, no P2P.
- **Claims:** through `lanes.py claim-run --gpus 1,2,3,5 --borrow`, never taking a held lock.

### 2.10 Completion proof: why the 10 days will not be wasted

Cyril: "[Cyril quote withheld]". Each way to fail is closed before day 1 or detected by hour 24.

| Failure | Closed by | When |
|---|---|---|
| Code never ran together | Step 0 merge + full pytest count line on the server venv | before anything |
| Wrong weights / import bug | S1 identical greedy text vs HF | before ablation |
| OOM on a card (video, 8k + gen, 16k) | S0 + S3 worst micro-step under 22.5 GB | before S5 |
| Host OOM / Strata starved | R1-R4 + S3/S4 measured RSS + per-step watchdog + `oom_score_adj` | before S5, watched for 24 h |
| Resume divergence / hung group | C-1 + S4 (replica sha, kill -9, heartbeat) | before S5 |
| NaN spread | finite-grad check + COMPLETE rule | every step |
| Disk fills | Lexar inits + milestones, `--keep 2`, B-4 root ≥ 75 GB, the 10-day disk projection printed by `run0_watch.py --project` from the measured checkpoint size | before S5 |
| Data runs dry | B-4 10-day supply projection with the text floor holding | before S5 |
| Reboot / rogue container | systemd supervisor + Q4 | before S5 |
| Too slow to matter | first-hour stage-2 tok/s ≥ 350, or accum 8 / 3 cards | hour 1 of stage 2 |
| Slow leak / Strata degradation | 24 h soak: RSS growth < 1 GB per rank, Strata timings within 20% of pre-launch, MemAvailable never below the floor | **at 24 h of stage 2; any failure stops the run on day 2, not day 9** |
| Retention collapse | per-checkpoint behaviour read; top-10 alarm protects and halves the LR | every checkpoint |

---

## 3. Interfaces between areas

A, B and C build in parallel against these contracts. The contracts are the code already on the three branches, as merged in Step 0. Where this plan adds something, the addition is marked **NEW**.

### 3.1 File ownership

No file has two owners. Tests are one file per owner.

| Area | Owns (create or edit) |
|---|---|
| **A** arch, plant, ablate, tokens, losses, kernels | `src/chimera_omni/config.py`, `src/chimera_omni/v2_layout.py`, `src/chimera_omni/importing/{qwen35,run0_plant}.py`, **NEW** `src/chimera_omni/importing/qwen35_vision_teacher.py`, `src/chimera_omni/media/training.py`, `src/chimera_omni/modeling/{layers,chimera}.py`, `scripts/{plant_qwen35_2b,ablate_run0,align_bridge_stage0,attn_kernel_check}.py`, `tests/test_run0_arch_torch_optional.py` |
| **B** data | `src/chimera_omni/training/{run0_data,media_sources,run0_samples,text_remap}.py`, `scripts/{run0_build_manifest,run0_intake_sync}.py`, `scripts/run0_stage_data.sh`, `scripts/final_steps.sh`, **NEW** `scripts/run0_genworker_jobs.py` (writes jobs to the existing gen-worker-text queue; no service change), `tests/test_run0_data_torch_optional.py` |
| **C** trainer, parallelism, ops code | `src/chimera_omni/training/{v2_trainer,run0_dump,run0_watch}.py`, `scripts/{run0_queue.sh,run0_supervisor.py}`, **NEW** `scripts/run0_lab_replay.py`, **NEW** `scripts/run0-supervisor.service`, `tests/test_run0_trainer_torch_optional.py` |
| **O** lead (ops, gated) | server-side only: Strata config / `scripts/sprite/patches/xcache.patch` (Sprite lane), NAS inventory and cleanup, `media-judge.timer`, docker restart policies, the rulings file, Relay notice |

### 3.2 A → B, C: layout API (`v2_layout.py`, on run0-a)

```python
V2_CONTROL_TOKENS     # ids 0..7 unchanged; 8..16 = <|text|> <|image|> <|audio|> <|video|> <|3d|> <|out|> <|/out|> <|edit|> <|mask|>
MODALITIES = ("text", "image", "audio", "video", "3d");  MARKER = {modality: "<|...|>"}
EMITTABLE_CONTROL     # NEW: the 5 markers, <|out|> <|/out|> <|edit|> <|/hdr|> <|/media|>; all other band ids are -inf at the head

def special_token_id(enc, token) -> int   # tokenizers.Tokenizer | HF tokenizer | dict token->id
def encode_text(enc, text) -> list[int]
def control_ids(cfg) -> dict[str, int]

class V2Sample:
    def text(ids, *, supervised=False, loss_mask=None) -> V2Sample
    def control_token(name, *, supervised=False, loss_mask=None) -> V2Sample
    def turn_open(role: str, enc) -> V2Sample                       # enc = a TOKENIZER OBJECT or dict, never an encode function
    def turn_close(enc, *, supervised: bool, newline=True) -> V2Sample
    def marker(modality, *, supervised=False, loss_mask=None) -> V2Sample
    def decision(outputs: Sequence[str], *, edit=False, supervised=True) -> V2Sample   # <|out|> [<|edit|>] <|m1|>.. <|/out|>
    def media(packet, *, modality: str | None = None, row=0, header=(), open_suffix=False) -> V2Sample
    loss_mask: list[bool]; modalities_in: set[str]; modalities_out: list[str]

def collate(samples, *, align="right") -> dict   # + labels, attention_mask, loss_mask (strip before forward)
```

Config keys (`config.py`):
- `V2_INIT_RULE_RUN0`, `PLANT_RULING`, profile `co2-2b-run0`;
- `native_unit_norm="rms"`, `attnres_init="zero"`, `a8=False`, `x0_skip=False`;
- **NEW** `head_mask_control=True`.

Losses (`media/training.py`):

```python
flow_loss(pred_x0, target_x0, sigma, type_ids, *, weighting="velocity", sigma_floor=0.2, mel_weight=0.05,
          nyquist_hz: Tensor[B] | None = None)          # NEW nyquist_hz: STFT/mel masked above it per item
    -> (loss, terms: dict[str, Tensor])
item_flow_noise_levels(levels, base, *, sampler="logit-normal", rows=...) -> Tensor[B]
align_loss(bridge_rows: Tensor[N, 2048], tower_rows: Tensor[N, 2048]) -> (loss, terms)   # NEW: cosine + 0.1*MSE, RMS-matched
```

Vision teacher (**NEW**, `importing/qwen35_vision_teacher.py`):

```python
load_vision_teacher(model_dir, device) -> nn.Module        # read-only from the Qwen safetensors, frozen, fp32, eval
teacher_rows(module, images: Tensor[B,3,H,W]) -> Tensor[B, (H/32)*(W/32), 2048]   # same raster order as our 32x32 patches
```

Model (`modeling/chimera.py`), **NEW**:
- `split_tied_embedding(model) -> (text_rows, control_rows)` and `merge_tied_embedding(model)`. The state-dict key stays `embed.weight`.
- The head applies `EMITTABLE_CONTROL` when `head_mask_control`.

### 3.3 B → C: data contract (`run0_samples.py`, `run0_data.py`, on run0-b)

```python
@dataclass
class Run0Item:
    kind: str           # "text_kd" | "understand" | "generate" | "twin" | "long"
    bucket: str         # §2.5 bucket name (accountant key)
    sample: V2Sample    # generate items end in an open suffix
    positions: int      # budget unit
    teacher: dict | None   # text_kd/long: {"positions": LongTensor[P], "ids": {t: LongTensor[P,K]},
                           #  "logprobs": {t: FloatTensor[P,K]}, "lse": {t: FloatTensor[P]}, "weight": {t: FloatTensor[P]}}
    source_id: str; provenance: str; record_sha: str; epoch: int
    meta: dict          # decision_positions, task, mode, target_ref; NEW: source_sr (audio), cond_dropped (bool),
                        #   kd_only (bool: served over its CE cap, CE weight 0)

class Run0Source(Protocol):
    stats: dict
    def next(rng, budget_positions) -> list[Run0Item]; def state() -> dict; def load_state(d) -> None; def exhausted() -> bool

def build_sources(manifest, *, cfg, enc, tok, chat=None, rank, world, seed, ledger=None, only=None) -> dict[str, BucketSource]
    # tok = tokenizers.Tokenizer (turn markers), enc = its text encoder; refuses (Run0DataError naming path/row) what the guard rejects
```

Media packets are the existing `NativeMediaPacket`:
- `features` float32 [1, R, 3072] with the per-dataset gain;
- `type_ids` and `positions` per `native_media.py`;
- `valid_mask` all true.

Rows match §2.4 exactly. For example, 512×384 gives 192 rows, and 2 s of video at 256×448 gives 2,688.

**Manifest** `/home/cereal/omni-campaign/run0/data/run0_data_vN.json`, schema `chimera_omni.run0.data.v1` (`DATA_CONFIG_SCHEMA_RUN0`); parts use `chimera_omni.run0.part.v1`, records `chimera_omni.run0.record.v1`:

```json
{"schema": "chimera_omni.run0.data.v1", "version": N, "parent_sha": "...",
 "tokenizer": {"path": "...", "sha256": "...", "template": "qwen35-chat enable_thinking=False"},
 "teacher_weights": {"k3present": {"k3": 1.0, "strata": 0.3, "q38": 0.1}, "else": {"strata": 1.0, "q38": 0.3}},
 "text_floor": 0.40, "kd_only_after_epoch": 4, "kd_only_ceiling_epochs_per_7d": 12,
 "generation": {"cond_drop": 0.1},
 "buckets": {"<name>": {"share": 0.24, "cap_epochs_per_7d": 6, "kind": "text_kd",
             "sources": [{"path": "...", "sha256": "...", "provenance": "registry", "rows": 0,
                          "teacher_caches": {"strata": "...", "q38": "...", "k3": "..."},
                          "corpus_version": "...", "license": "...", "authorship_checked": true,
                          "author_field": "author|model|..."}]}},
 "denylist_patterns": ["selfgen", "..."], "row_deny_patterns": ["..."], "holdout_registry": "v3"}
```

New keys (`text_floor`, `kd_only_*`, `generation.cond_drop`, `author_field`) are **NEW**. `validate_manifest` accepts their absence with the defaults shown, so older manifests still load.

### 3.4 A → C: init layout and lineage

`--init` = a directory with `receipt.json` + `child/` (chimera safetensors + `config.json`), which `find_fold_layout` resolves. The receipt chain:

```
chimera_omni.run0.align_receipt.v1 -> chimera_omni.run0.ablate_receipt.v1 -> chimera_omni.run0.plant_receipt.v1
```

Each link carries `parent_receipt_sha256`, `ruling`, `init_rule`, `state_sha256` and `checkpoint.files` (name → sha256). The run-0 launch init is `ablate/`. The align receipt is produced only by the fallback script; stage 1 is part of the run.

C's `load_fold_base(kind="plant")` accepts the layout only if all of these hold:
- the schema is one of the three;
- the ruling equals `PLANT_RULING`;
- `init_rule == V2_INIT_RULE_RUN0`;
- the profile is `co2-2b-run0`;
- every file sha matches the receipt. **NEW:** checked at every load, because of exFAT.

`kind="fold"` **refuses** `V2_INIT_RULE_RUN0`.

Server locations: `/mnt/lexar/omni-campaign/run0/inits/{plant,ablate}/` after their receipts are verified. The trainer's working state lives only in its checkpoints.

### 3.5 C: checkpoint layout

```
/home/cereal/omni-campaign/run0/run/checkpoints/step-XXXXXXXX/
  model/              safetensors + config.json (source.init = lineage; metadata.init_rule = V2_INIT_RULE_RUN0; single embed.weight)
  optimizer.pt        Adafactor factored state (no first moment), param groups trunk / new / control_rows
  trainer_state.pt    step, stage (1|2), per-rank source states keyed by source sha, epoch ledger, RNGs
  trainer_state.json
  replica_sha.json    {rank: {params_sha, optim_sha}}: all equal, or the checkpoint is not COMPLETE
  finite.json         NEW: every tensor finite
  manifest.json       the run0_data_vN.json in force (sha recorded)
  COMPLETE.json       chimera_omni.v2_trainer.checkpoint.v2
/home/cereal/omni-campaign/run0/run/{train.log, metrics.jsonl, events.jsonl, samples/step-N/, ALERT, STOP, NEXT_MANIFEST, PROTECTED/, status.json}
/mnt/lexar/omni-campaign/run0/{inits/, milestones/step-XXXXXXXX/, protected/step-XXXXXXXX/}
```

### 3.6 Launch line (C-11, in `run0_queue.sh`)

```
lanes.py claim-run --lane L-TRAIN --agent claude --gpus 1,2,3,5 --borrow \
  --job "run0-2b (cyril-run0-plant-qwen-2026-10-22)" --wait 3600 --mem-floor-gib 30 --omni-lab-kind distill
v2_trainer.py --profile co2-2b-run0 --init /mnt/lexar/omni-campaign/run0/inits/ablate --data-config $R0/data/run0_data_v1.json \
  --stage1-steps 400 --stage1-hours 18 --freeze-trunk-stage1 --align-teacher /home/cereal/models/Qwen3.5-2B --align-weight 1.0 \
  --optimizer adafactor --accum 4 --seq-len 4096 --understand-tokens 8192 --gen-tokens 4096 --long-lane-tokens 16384|0 \
  --schedule wsd --warmup 150 --lr 5e-5 --new-module-lr 8e-4 --weight-decay 0.1 --weight-decay-scope trunk-matrices \
  --grad-clip 1.0 --flow-weighting velocity --sigma-sampler logit-normal --flow-ramp 0.2:1.0:300 --media-ce 0.3 \
  --teacher-weights "k3present:k3=1,strata=0.3,q38=0.1;else:strata=1,q38=0.3" --top-n 10 \
  --ckpt-every 50 --keep 2 --keep-every 500 --milestone-dir /mnt/lexar/omni-campaign/run0/milestones \
  --mem-floor-gib 30 --mem-resume-gib 40 --disk-floor-gb 60 --max-consecutive-nonfinite 3 \
  --dist-timeout 1200 --dump-spec $R0/dump_spec.json --dump-every-ckpt 1 --dump-budget-s 600 \
  --load-stagger-s 60 --isolate-ranks on --rank-cpu-nodes 0,0,0,1 --oom-score-adj 500 --runtime-guards on \
  --allow-world-change --health-every 10 --relay-outbox $R0/run/relay --alarm-top10-drop 0.2 --alarm-action protect-halve
```

**NEW** flags C adds: `--stage1-steps`, `--stage1-hours`, `--freeze-trunk-stage1`, `--align-teacher`, `--align-weight`, `--flow-ramp`, `--milestone-dir`, `--alarm-action`. The flags that exist on run0-c keep their names. The existing `--omni-lab-*` flags are replaced by the replay process.

---

## 4. Build plan

Paths: worktrees under `C:/Users/cereal/Documents/ai training/.agent-worktrees/chimera-omni-5d4d4e1dcd/workers/claude/`; on the server, `$R0=/home/cereal/omni-campaign/run0`.

**Step 0: integration branch (blocks everything; lead).**
- Create `claude/run0-int-20261028` from `8d8f064`. Merge `-a` 4a53d3a, `-b` 96ed2b9 and `-c` e52fa93 (the merge-tree is clean).
- Fix the two contract breaks:
  - `run0_dump` passes the tokenizer object, not an encode function (C's file);
  - the byte tokenizer stub in `run0_samples` supplies an `<|im_start|>`/`<|im_end|>` mapping (B's file).
- Re-run `tests/test_run0_{arch,data,trainer}_torch_optional.py` plus the v2 trainer / architecture / all-modalities files on the server Pascal venv, and capture pytest's final count line. The two fp32-tolerance failures in the v2 files get a tolerance review with the measured values recorded, not a silent loosening.
- Deploy one pinned commit to `/home/cereal/omni-campaign/chimera-omni-v2/` with `DEPLOYED.json` (sha-matched; `run0_queue.sh` checks it).
- Accept: the count line is recorded; `DEPLOYED.json` sha equals the commit.

Areas A, B and C then branch from `run0-int` (`claude/run0-int-a/b/c-20261028`) and merge back by area. File ownership (§3.1) keeps the merges conflict-free.

### Area A: arch, plant, ablate, tokens, losses, kernels

- **A-1** Download `Qwen/Qwen3.5-2B` to `/home/cereal/models/Qwen3.5-2B` via the server-manager `--script` (`huggingface-cli download`, no token). Record the safetensors sha256 and the LICENSE.
  - Accept: `plant_qwen35_2b.py --check-only` passes.
- **A-2** `EMITTABLE_CONTROL` + head mask; `split/merge_tied_embedding`; the fold loader's refusal of the RUN0 tag.
  - Accept: the masked ids get −inf logits in CE, KD and sampling; split → merge round-trips bit-exactly; `kind="fold"` refuses a RUN0 layout (tests).
- **A-3** Plant → `$R0/plant/`.
  - Accept: S1. The receipt lists missing keys exactly AttnRes + bridge + head.
- **A-4** Ablate → `$R0/ablate/`.
  - Accept: S2. The receipt lists exactly 34 changed tensors with sha before and after. Then move both inits to the Lexar, sha-verified (§2.7).
- **A-5** S0 `attn_kernel_check.py` on one P40 via `claim-run` (math SDP disabled; chunk masks from segment ids).
  - Accept: §2.8 S0; chunked and full attention agree to 1e-5 on CPU.
- **A-6** `flow_loss(nyquist_hz=…)`, `align_loss`, `qwen35_vision_teacher.py`.
  - Accept: E[w]=1 under the schedule; finite loss at σ ∈ {0.01, 0.5, 0.99}; zero STFT/mel gradient above `nyquist_hz` on a synthetic item; `teacher_rows` raster order matches the 32×32 patch order on a 64×96 test image.

### Area B: data

- **B-0** Provenance audit → `run0-20261022/data_provenance_audit.md`:
  - authorship field per turn across `text_clean_v1`, the `text` part and `staging/rows.jsonl`;
  - prompt authorship per intake batch and date;
  - WIT load check;
  - BFCL-62 held-out rows in `bfcl_native_v1`;
  - which gen-worker-text job types exist and whether any carries `n_probs`;
  - the stale IQ3_S label list.
  - Accept: every manifest source has `authorship_checked: true` or is conditioning-only; counts of excluded turns per source are recorded.
- **B-1** `run0_data.py`: the authorship-field guard, the text floor, KD-only repeats and the ceiling, and the new manifest keys with defaults.
  - Accept: a manifest naming `selfgen` or `strata-capture-*` is refused with the path named; on a real `text_clean_v1` shard, Claude/Codex/unknown-author turns are dropped from KD rows and counted; a tool row with a Claude turn yields `loss_mask` false on that turn; at epoch 5 a text item comes out `kd_only=True` with CE weight 0.
- **B-2** New part `librispeech-train`: map the 64 orphaned parquet blobs in `/mnt/lexar/omnicoder_0.1.0/models/hf/hub/datasets--librispeech_asr/blobs` to splits via the Hub tree; normalise the text; speaker tag; `source_sr=16000`. Used for ASR and TTS.
  - Accept: 10 samples decode to 80 ms rows with the stated counts; split hours sum to about 460 h.
- **B-3** Refresh `intake-image-*`, `intake-music-ace`, `intake-video`, `intake-3d` from the current judged intake. Add the canonical-corpus derived view, the video-understanding and temporal-order builders, and condition dropout in the generation builder. Stage MAESTRO off the NAS to the Lexar; voxelise the 414 GLBs; submit the D4 TTS jobs to the media-gate idle queue.
  - Accept: the guard refuses `strata-capture-*` and trace rows by name; `cond_dropped` occurs at 0.10 ± 0.01 over 10k generation items; twins are never dropped.
- **B-4** `final_steps.sh`: manifest → per-source verify → `verify-kd` on the real caches after remap → intake register → seal `$R0/data/run0_data_v1.json`; the 10-day supply projection; the root-space check.
  - Accept: the trainer loads the manifest via `--data-config`; `verify-kd` matches at every original position; the text + tool floor holds to day 10 at 33M and 46M/day; root ≥ 75 GB free after staging (otherwise data parts go to the Lexar).
- **B-5** `run0_genworker_jobs.py`: intent bank, edit instructions and temporal-order phrasings as existing-type jobs on the gen-worker-text queue at background priority. It runs during training and feeds intake.
  - Accept: 100 rows arrive through intake with Strata provenance and pass the guard.
- **B-6** `run0_intake_sync.py`: outside the trainer, writes `NEXT_MANIFEST`, at most once per 24 h.
  - Accept: S4's controlled bump.

### Area C: trainer, parallelism, ops code

- **C-10** Bucketed pinned all-reduce (256 MB per rank).
  - Accept: a 2-rank CPU test equals the single-rank accumulating step at 1e-6; RSS per rank < 5 GB in S3.
- **C-11** The §3.6 launch line in `run0_queue.sh`; `dump_spec.json` written; milestone copy with sha verify; inits read from the Lexar with receipt checks.
- **C-12** `run0_watch.py`: the alarm table (§2.7) with protect-and-halve; RSS-leak and Strata `/metrics` + `/slots` probes; Lexar I/O alarm; `--project` 10-day disk projection; fadvise and ionice on its readers and writers.
- **C-13** `run0_lab_replay.py` (timeouts, idempotent) and removal of the in-step omni_lab calls.
- **C-14** Stage 1 / stage 2 in `v2_trainer.py`: `--freeze-trunk-stage1`, the align teacher on the card in stage 1 only, the flow ramp, three param groups (trunk / new / control_rows via split embedding), WD scope excluding the tied table, the `beta1 is None` assert, the finite-grad check after the all-reduce, `finite.json`, the backward after each pass, `enable_math_sdp(False)`, the UUID / device assert, `--dist-timeout 1200`, and dumps outside the collective.
  - Accept: the CPU tests for each; S3.
- **C-15** Strata `n_probs` check, used only if gen-worker output carries `n_probs`: the same prompt at temp 0.6 and 1.0 with `top_k=0` must give identical top-64 logprobs, and strata-gate must pass `n_probs` through. Otherwise that teacher is not used.
- **C-9** `run0_supervisor.py`: heartbeat kill, `NEXT_MANIFEST` adoption at checkpoints, claim FD held across restarts. `run0-supervisor.service` (install gated, O-6).
- **C-3** S3, then S4, on GPU1/2/3/5 via `run0_queue.sh --dry-run` and then `--smoke`.
- Stretch, not launch gates: **C-7** joint AV spans; **C-8** activation offload for the 50k lane.

### Ops (lead; each item gated where marked)

- **O-1 NAS cleanup** (authorised by Cyril 10-28):
  1. Inventory per directory on the NAS host itself through the server-manager client (CIFS `du` from the AI server times out). If no NAS entry exists, run a background `du` on the AI server with no timeout, written to a file.
  2. Delete these categories, with evidence kept:
     - unproven Neural Forge checkpoints, ruled deletable 2026-08-19;
     - omnicoder weights, ruled deletable 2026-08-19;
     - exact duplicates whose sha matches a verified copy on the Lexar or root;
     - temp and scratch.
  3. Never delete fold artifacts (protected 10-21), Cyril's speech, sealed corpora, the PDF corpus or MAESTRO's source.
  4. List anything else for Cyril first, because a delete cannot be undone.
  - Target ≥ 10% free. Run 0 never writes to the NAS.
- **O-2** Root disk: Q5 only if B-4 needs it.
- **O-3 RAM:** R3, then R4 only if necessary (§2.6), each with measured before/after Strata timings and a `run_event`. Gated; Sprite lane.
- **O-4 Judge:**
  - Check whether a vision projector exists for the Strata model and its VRAM cost on GPU0.
  - If it fits with Sprite decode within 10%, move the judge to Strata (D2).
  - Otherwise tell Cyril (Q2).
  - Pause `media-judge.timer` from S3 until then.
- **O-5** Relay notice L3 to the other lanes.
- **O-6** Q4 restart policies and the systemd unit install, with Cyril's yes.
- **O-7** omni_lab backup (agent-memory-backup pattern) before S5.

### Launch order

```
Step 0 (merge, fixes, pytest count, deploy)
  -> in parallel: A-1..A-2, A-5, A-6 | B-0..B-4 | C-10..C-15, C-9 | O-1, O-3 (R3), O-4, O-5, O-6
  -> A-3 (S1) -> A-4 (S2) -> inits to Lexar
  -> S3 -> S4 -> (R4 only if S3/S4 RAM says so) -> O-7 omni_lab backup
  -> S5: claim-run on 4 cards -> stage 1 (<= 400 steps / 18 h) -> stage 2
  -> stage-2 hour 1: tok/s rule -> 24 h soak decision (§2.10)
  -> while training: B-5, B-6 (daily), C-7, C-8
```

---

## 5. Schema

No new tables. Run 0 uses the existing omni_lab tables `run`, `run_device`, `run_model`, `run_dataset`, `dataset_version`, `metric` and `run_event`, through `omni_lab.py log-run` and `run0_lab_replay.py`, with raw parameterized psycopg2 (vault key: `omni_lab_pg`). No ORM, Pydantic or SQLAlchemy anywhere.

- Each manifest bump writes one `dataset_version` row and one `run_dataset` link.
- The on-disk JSON (manifest, `metrics.jsonl`, `events.jsonl`) is the source of truth; omni_lab mirrors it.
- The run's `notes` name the two deviations (Q1, plant run-0 only).

---

## 6. Trade-offs

| Chose | Rejected | Why |
|---|---|---|
| Post-trained Qwen3.5-2B | Base | retention of chat/tool behaviour is the gate; caches are chat-templated |
| λ 0.9 on `ffn.down` + σ 0.05·rms on up/down, blocks 7-23, WD 0.1 | global 0.2-0.4; FIRE/DASH; norm resets | retention, not time to scratch loss |
| Trunk peak 5e-5, new modules 8e-4 | 1.5e-4 trunk | most perturbation-sensitive donor; re-warming is reversible, collapse is not |
| Stage 1 on all 4 cards, trunk frozen, every modality + ViT image target | one-card bridge-only stage 0 | heads and bridge warm up before their gradients reach the trunk; cheap; doubles as soak |
| Drop the vision tower and MTP; tower as a stage-1 teacher only | keep as aids | end model encoder-free; MTP lost on the phone |
| Head mask on non-emittable control ids | special spare-row init | tied table: spare rows would steal mass at step 0; a mask makes their init irrelevant |
| Split tied embedding into text/control params | gradient scaling | Adafactor normalises per element; scaling does nothing |
| CFG condition drop 0.1 | 0.0 | guidance cannot be added later without it; the earlier "neutral" result was on toys |
| One `<|audio|>` + kind header; composable `<|out|>…<|/out|>` + `<|edit|>` | per-kind tokens; one token per combination | Cyril: one token per modality |
| 512 generation / 768 understanding | 1024² on day 1 | 33-46M positions/day |
| Text floor wins; KD-only repeats above cap | media first on reallocation | Show-o2 collapse; retention is the gate |
| Strata data through the existing gen-worker-text queue | a new strata-gate client | background priority and screened intake already exist; no new capture path |
| Prompt cache to a disk tier, then cold experts to disk only if necessary | experts first; or plain mmap | lowest Sprite cost first; plain mmap thrashes against run-0 I/O and breaks xcache's pinned copies |
| Bucketed all-reduce | full pinned mirror | 44 GB → about 26 GB for 4 ranks **(proj)** |
| Manifest bumps by controlled stop + resume | in-process barrier switch | reuses the S4-proven path |
| Protect-and-halve on alarms | pause for a read | nobody reads at night; 4 idle cards for hours |
| Telemetry via a replay process | in-step DB writes | a DB stall cannot stall training |
| `--keep 2` on root + Lexar milestones and inits | 3 rolling + inits on root | root would cross its floor on day 0 |
| 4 cards with the judge paused (unless O-4 fits it on Strata) | 3 cards | Cyril: "[Cyril quote withheld]"; more trained tokens |

---

## 7. Risks and what we watch

**Touches production, credentials or running services (each gated):**
- `strata-llama` restart for R3 and, if necessary, the R4 xcache rebuild. Cyril authorised it "if necessary"; the measurement decides. Sprite keeps GPU0/6 throughout.
- Pausing `media-judge.timer`: judged media stops for the run (Q2).
- NAS deletions by category (O-1); root deletions only per Q5.
- Container restart policies and the systemd unit (Q4, O-6).
- Holding GPU1/2/3/5 for about 10 days. Every other lane is refused. Unlocked containers bound to the P40s bypass the locks, and the foreign-process alarm hard-pauses.
- gen-worker jobs and TTS jobs at background or idle priority only; no service configuration changes.
- Never GPU4/ComfyUI, `speech-worker@*` or `gen-worker-*` config. Never NCCL or P2P.

**Backups first:** inits immutable by receipt sha (Lexar); omni_lab backup before S5; this plan's and the rulings file's previous versions are in `backups/`; the vault is untouched.

**What can still go wrong:**
- **Text retention** is the pass/fail, read in every dump. Failure mode: Show-o2-style collapse. Defences: curated KD ≥ 40% with the floor winning, trunk 5e-5, WD 0.1, stage 1 before unfreezing, the flow ramp, the head mask.
- **4-rank gloo, C-1 resume and C-10 have never run on GPUs.** S3/S4 are the proof; any failure is found before day 1.
- **Throughput is projected.** Below 350 pos/s: accum 8, or 3 cards.
- **RAM:** the Strata prompt cache regrows to its cap; the per-step watchdog checkpoints and pauses; `oom_score_adj +500` makes the OOM killer take a rank before Strata.
- **R4 Sprite slowdown:** reverted if decode or prompt read drops more than 20%.
- **Disk:** the root margin is thin (§2.7); the Lexar is exFAT (copies, sha checks, I/O alarm).
- **Generation may stay blobby at 512 for days.** Rounds 4-7 failed edit, video and speech on toys. CFG at sampling is now available. 1024², LPIPS and joint AV are stage-2 levers.
- **Teacher coverage:** K3 is fixed at 1.89M positions (Q1). Strata `n_probs` may be unavailable without a gen-worker change, so the text signal is mostly the cached Q38/Flash top-64 repeated as KD-only after epoch 4.
- **Data thinness** (Q7): video-out is token presence; video understanding rests on Strata-written prompts only; no sfx or joint AV until the downloads land; media intake frozen while the judge is paused.
- **Contamination:** `bfcl_native_v1` is trained (Cyril's 10-17 ruling), so BFCL-based tool scores on run 0 are not held-out evidence beyond the BFCL-62 split. Reports must say so.
- **Licences recorded on the run:** MAESTRO CC BY-NC-SA, HY3D non-commercial, COCO Flickr (ruled fine 10-14), LibriSpeech CC BY 4.0, Qwen3.5-2B Apache-2.0.

---

## 8. Critic findings (10-28): applied, modified, rejected

| # | Finding | Verdict | Where |
|---|---|---|---|
| 1 | "K3 primary" is not met | **Applied.** Stated as a deviation (Q1), recorded in the 10-28 ruling entry and in the omni_lab notes. | §0, §5, rulings |
| 2 | Plant is a narrow exception; tag checkpoints; fold loader refuses | **Applied.** | §2.1, §3.4, A-2 |
| 3 | Numeric gates contradict "behaviour, never a number" | **Applied, one modification.** S1-S3 now pass on behaviour reads and ops facts. S1 keeps "greedy text identical to HF" as a functional plumbing check (an import bug is a plumbing fault, the same standing as G0/G1 in the 10-12 rulings); logit numbers are diagnostics. Stage-1 captions are read, not gated. | §2.8 |
| 4 | RAM lever order differs from Cyril's | **Applied, modified.** The prompt cache goes to a disk tier first (his 10-27 words), then some cold experts to disk only if necessary (his 10-28 words), with the Sprite impact stated (Q3). | §2.6 |
| 5 | Strata writer client is risky | **Applied, modified.** No new client; jobs go to the existing gen-worker-text queue using existing job types only. `n_probs` capture is deferred if no existing job type carries it, because changing gen-worker code would touch `gen-worker-*`. | §2.5, B-5 |
| 6 | Claude/Codex exclusion by known rows only | **Applied.** Authorship-field guard on every turn of three corpora; memory `tool_call` events excluded wholesale; selfgen and rescore dropped by row. | §2.5, B-0, B-1 |
| 7 | Unverified or contaminating sources | **Applied, partly rejected.** WIT verified or dropped. Pre-gen-worker prompts are conditioning-only. **Rejected:** dropping `bfcl_native_v1`, because Cyril ruled on 10-17 that "tool-bfcl-native stays as data"; the BFCL-62 held-out rows stay excluded and the contamination is stated. | §2.5, §7 |
| 8 | Text budget runs out; caps conflict with the floor | **Applied.** The floor wins; KD-only repeats above the cap (CE 0 after epoch 4, 12-epoch ceiling); 10-day supply projection as a B-4 gate. Strata `n_probs` from day 1 only if available (item 5). | §2.5, B-4 |
| 9 | Verify Strata `n_probs` semantics | **Applied.** C-15. | §2.4, C-15 |
| 10 | Expert mmap thrashes against trainer I/O | **Applied, modified.** No plain mmap: cold experts via O_DIRECT into pinned bounce buffers as an xcache patch with its own test; fadvise/ionice on all run-0 I/O. **Partly rejected:** keeping checkpoints off the GGUF SSD unconditionally. There is no other fast ext4 disk, and an 8 GB write every 4-5 h fits the BX500's SLC cache. Checkpoints move to the Lexar only if R4 is applied. | §2.6 |
| 11 | RAM spikes at load | **Applied.** Tensor-wise streaming everywhere, GPU hashing, staggered loads, S4 RSS assert. `vm.swappiness` read; changing it is gated. | §2.6, S4 |
| 12 | A dead rank hangs gloo | **Applied.** Timeout 1200 s, heartbeat kill, dumps outside the collective with a 600 s share. (15 min would be shorter than the old 900 s dump budget, so the dump budget dropped instead.) | §2.7 |
| 13 | Make P2P and wrong-card placement impossible | **Applied.** One UUID per rank via the launcher; device/name/UUID asserts. | §2.6, §2.9 |
| 14 | Two passes' graphs summed | **Applied.** Backward after each pass; S3 worst-case micro-step. | §2.6, S3 |
| 15 | Attention masks on sm_61 | **Applied.** Math SDP disabled; per-chunk masks from segment ids; S0 fwd+bwd with real masks. | §2.6, A-5 |
| 16 | Adafactor details | **Applied.** `beta1=None` asserted; control rows as a separate parameter (split tied embedding; layout unchanged); tied table excluded from WD. | §2.6, §2.7, C-14 |
| 17 | Root disk crosses its floor on day 0 | **Applied.** Inits and milestones on the Lexar with sha checks at load, `--keep 2`, B-4 root ≥ 75 GB, Lexar I/O alarm. | §2.7 |
| 18 | NaN passes the replica check | **Applied.** Finite-grad check after all-reduce, skip + stop after 3, COMPLETE requires finite. | §2.7 |
| 19 | Pause-for-read idles cards at night | **Applied.** Protect-and-halve; hard pause only on hard failures. | §2.7 |
| 20 | Live manifest switching is untested | **Applied.** Controlled stop + resume, proven in S4. | §2.5, B-6, C-9 |
| 21 | Reboot survival | **Applied, gated.** systemd supervisor holding the claim FD; restart policies via Q4. | §2.7, O-6 |
| 22 | Strata probe measures the wrong thing | **Applied.** Read-only `/metrics` and `/slots` plus the gate log. | §2.7 |
| 23 | omni_lab must never block | **Applied.** Replay process with timeouts. | §2.7, C-13 |
| 24 | CFG off forever without dropout | **Applied.** cond-drop 0.1 in data; dumps at CFG 1 and 3. | §2.4 |
| 25 | Image-only stage 0 cannot meet its acceptance; other modalities have no warm start | **Applied.** Stage 1 on all 4 cards, trunk frozen, every modality, ViT target for images, then the flow ramp. The caption read is recorded, not gated. | §2.4a |
| 26 | Video-in untrained; LibriSpeech TTS; 16 kHz band | **Applied.** Video understanding from Strata-authored prompts + temporal order; LibriSpeech for TTS with speaker tags; loss band-limited to the source Nyquist. | §2.4, §2.5 |
| 27 | The judge cannot move to Strata | **Applied, modified.** Cyril's D2 said "fix that first", so O-4 measures whether a projector fits without slowing Sprite before giving up. Meanwhile the judge is paused and Cyril is told (Q2). | §0, O-4 |
| 28 | Trunk LR too aggressive | **Applied.** Trunk 5e-5, new modules 8e-4; raise only on the 24 h read. | §2.7 |
| 29 | Spare control rows steal probability mass (text cut off mid-item) | **Applied, as understood from the partial text.** A head mask on every non-emittable control id makes their init irrelevant to text behaviour. | §2.3 |
| 30+ | not received | n/a | n/a |

Rejected outright: none. Partly rejected, with reasons: 7 (BFCL kept per Cyril's ruling) and 10 (checkpoints stay on root unless R4 is applied). Modified: 3, 4, 5, 12, 27.

---

## 9. Architecture-candidate verdicts (ARCH_CANDIDATES rev 2, 2026-10-28)

No candidate was marked INCLUDE IN RUN 0, so run 0 adopts none, and none goes in the fold pass (that would muddy the run-0-vs-fold comparison).

| Item | Run 0 | Fold pass | Where it lands |
|---|---|---|---|
| MemT-1.0 | no | no | no: conflicts with "memory is tools"; Metis-4B retention failure; 5-7% slower decode; needs a frozen trunk |
| Sage-A1.5 | no | no | no as architecture, teacher or data; coded exclusion in the B-1 guard |
| DeepInfer engine / DSpark / CSA2 / Engram / window replay | no | no | no |
| FP4 KV trained in + cross-layer KV sharing ("KV diet") | no | no | small arm inside S6 QAT on a post-run-0 clone, judged on 32k behaviour; check the bf16-KV assumption first |
| Encoder/decoder split | no | no | only if a phone measurement shows media prefill/TTFT is the bottleneck |
| TiDAR one-pass layout; diffusion chain-of-thought | no | no | no |
| NLD-style linear self-speculation (A-DIFF) | no | no | only after P-SPEC (phone: cost(k=8) < 2.5× cost(k=1)) and ACC-0 (CPU, ≥ 1.3× at 4k and 32k with behaviour held); neither uses run 0's P40s |
| Fused media commit + next-chunk refine | no | no | phone inference-scheduling experiment (P1/P2), no training change |
| `<|mask|>` row | reserved id 248336, untrained, head-masked | n/a | used only if A-DIFF ships |

---

## 10. Key paths

- Reports: `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/run0-20261022/{RUN0_BUILD_PLAN.md, survey_code.md, survey_data.md, survey_hw.md, survey_lit.md, data_provenance_audit.md (B-0), samples/}`, plus `.../omni-campaign-2026-09/{ARCHITECTURE_V2_2026-10-21.md, ARCH_CANDIDATES_2026-10-22.md, distill/STAGE_D_RULINGS_2026-10-12.md}`.
- Worktrees: `C:/Users/cereal/Documents/ai training/.agent-worktrees/chimera-omni-5d4d4e1dcd/workers/claude/{arch-v2-20261021, run0-a-20261022, run0-b-20261022, run0-c-20261022}`; integration branch `claude/run0-int-20261028`. The 10-28 scratch overlay used for the contract check is `C:/Users/cereal/AppData/Local/Temp/claude/C--Users-cereal/0c189f21-3de3-4c40-b42c-db54c9d84462/scratchpad/run0-int`.
- Server:
  - `/home/cereal/models/Qwen3.5-2B`
  - `/home/cereal/omni-campaign/run0/{plant, ablate, data, run}` (`plant` and `ablate` move to the Lexar after their receipts)
  - `/home/cereal/omni-campaign/chimera-omni-v2/` + `DEPLOYED.json`
  - `/home/cereal/omni-campaign/rulings/arch_v2_init_source.json`
  - `/mnt/lexar/omni-campaign/run0/{inits, milestones, protected}/`
  - `/mnt/lexar/omnicoder_0.1.0/models/hf/hub/datasets--librispeech_asr/blobs`

---

## Addendum B2 (2026-10-28): the support products in run 0, lead decisions and the mixture

Claude leads Omni (Cyril, 10-14). Each item below is a **lead call** that Cyril may revise. Cyril ordered the products themselves: "fix the audio gap", TTS "4 sure", "[Cyril quote withheld]", "[Cyril quote withheld]", and on 10-28 "[Cyril quote withheld]". The data is manifest **v4** (area B branch `claude/run0-b-20261022`). Details are in `data_manifest.md`.

### Lead calls

**Audio and joint audio-video**
- **New `sfx` bucket (3 %).** It holds `sfx_caption_v1` (Clotho + AudioCaps: sound captioning, and sound generation from Clotho only) and the `av_label_v1` part of `ext-av-understand` (VGGSound: sound and AV class labels).
- **New `av_joint` bucket (3 %).** It holds the `av_caption_v1` part of `ext-av-understand` (AVCaps, AudioCaps-on-AudioSet and VALOR: AV captions plus audio captions) and `ext-av-gen` (joint video-then-audio generation, half of the bucket). One built source is split between `sfx` and `av_joint` by the staging product of its clips (`media_origin_products`); nothing is rebuilt.
- **`video_understand` keeps `video_caption_v1`.** That is MSR-VTT human captions at 0.75 of the bucket, with the judged intake video taking the rest.
- **AVCaps is counted once.** Its 2,061 clips are in both `video_caption_v1` and `av_caption_v1`. They train only in `av_joint`, and `ext-video-caption` refuses every `video_caption_v1:avcaps:` clip.
- **AudioCaps is 24 kHz.** It trains understanding at full weight. Generation is never trained on upsampled audio: a generation record whose target audio was recorded below 38.4 kHz is not served (`gen_min_source_sr 38400` on `ext-sound` and `ext-av-gen`, and future builds stop writing them). The twin of such a record has no audio target and is still served. Speech TTS is unchanged: plan §2.5 trains it with the band-limited loss.
- **The hold-out stays strict.** VALOR clips lost to the VGGSound test set stay held. Every product's held clip ids, YouTube ids and captions are folded into run 0's hold-out set **r1** (below), and a train clip whose YouTube video is held is refused even when its own split says train.
- **The music bucket is music only (4 %).** ACE intake, MAESTRO and MTG-Jamendo at 0.10.
- **New `audio_edit` bucket (0.5 %).** SonicMaster restore pairs, degraded in and clean out as an audio edit.
- **Tier-2 audio (FSD50K first) waits until after launch.**

**Concept, intent, edits**
- **`coco-tts-captions`:** the concept-interleaved bucket at 2 %. It reads `records.jsonl`, takes a clip only once the product manifest lists its FLAC with a sha256, and is re-synced by the daily intake as new blocks land.
- **`strata-intent-bank`:** twins only. The clarify rows (no ruling yet) are excluded and never read. The ambiguous rows (label-check disagreements) are held in r1.
- **`strata-edit-instructions` and `coco-procedural-edits`:** the crop mismatch is fixed. An edit's source image is now cut exactly like its result (same size, same centre crop; the source used to get 512x416 uncropped against the result's 512x384). `coco-procedural-edits` was rebuilt and the old source retired; it trains at full weight (0.05 of `image_gen`). The instructions themselves train inside the edit records.
- **`coco-edit-pairs`** (unjudged Qwen-Image renders): only pairs the Strata media judge passed (`judged: true`, verdict pass) enter training. Today that is none, and all 13 wait in intake. Any record that says `judged: false` is refused at load as well.

**Text, KD, K3**
- **`strata-kd-gen`:** teacher id `strata-q4kxl-abl-v2` (omni_lab model 2571), as its own teacher key `strata_q4`. `flash-next-strata-iq3s` (model 2572) keeps the key `strata`. The two are separate identities sharing the plan's strata KD weight (1.0, or 0.3 where K3 is present). Where the same prompt exists in both, the Q4 row is served and the IQ3_S-era `intake-strata-text` copy is denied.
- **`registry-text/v1` replaces the allowlisted part of `text_clean_v1` (never both).** It is CE until a teacher cache exists. `text_clean_v1` stays in the manifest disabled, because the top-10 alarm's `holdout_text` rows still come from it with both caches. Every registry-text document that any held row touches is denied (`--text-clean-mode kd` restores the old behaviour). The cost is that the Q38 + Strata KD on about 5.4M allowlisted tokens is replaced by CE on the same documents until a cache pass.
- **`pdf-corpus/v1`:** CE text, cut to run 0's 4,096-position main lane by the real Qwen3.5 token count (at most 3,600 tokens a chunk, request included in the margin). Before this, 3.6 % of the 10,000-character chunks did not fit. The PDF and registry-text documents were rebuilt and the old sources retired.
- **`registry-media/v2`:** Jamendo goes to the music bucket (the track in, its public title / artist / human tags out; tags in, music out where tags exist). SonicMaster restore pairs go to `audio_edit`: two aligned 10 s windows per pair. Degraded-only clips have no target and are skipped.
- **K3** (lead ruling `k3-cache-ratify-run0-L1-2026-10-28`, Relay 204083): the provisional bf16 cache is ratified for run 0. Every loss term is masked on the listed documents: all 521 spans of `k3_authorship_exclude_v1.json` (sha256 prefix f10d4750), which are 500 exclude plus the 21 review documents. The B-0 audit of the 21: all are `manual_cleanup_batch_000156_v10_memory_harness` rows, an agent's memory harness, so they are masked too. The authorship field guard already reads `corpus/k3` by rid through `author_allow_v1.json` (no listed rid is on the allowlist). The spans are now also masked by their own positions, pinned by sha256, and the leakcheck proves each span has no target position.
- **Exclusions stay.** No Claude- or Codex-authored text is ever a target, and there is no small-model data. The DeepSeek-R1 solutions in OpenR1-Math and DeepMath stay.

**Memory harvest** (Cyril, 10-28: "[Cyril quote withheld]")
- **New `memory_tool` bucket (2 %, a tool-use share)** fed from `run0-intake-staging/memory-toolcall/v<N>/`. Every version and every sealed batch is read, and the daily intake sync picks up new batches.
- Only records under ruling `cyril-memory-harvest-2026-10-28` are read. A record must have Strata on every assistant turn, Cyril on the first user turn, tool output on later user turns and a lead template as the system turn. Anything else is refused whole.
- Cyril's prompt and the real tool results are context (`loss_mask` false). Only Strata's turns are targets. No batch has landed yet, so the bucket's share goes to `tool` until one does. Launch does not wait for it.

**Hold-out set r1.** `/mnt/lexar/omni-campaign/run0/holdout/r1` is registry v3 copied byte for byte (the v3 file hashes match `HOLDOUT_REGISTRY_v3.json`). Added to it: the hold-out ids and texts of `sfx_caption_v1`, `av_caption_v1`, `av_label_v1` and `video_caption_v1`, and the intent bank's ambiguous requests.
- It sits beside run 0, not in `corpus/holdouts`, because `intake.py` takes the newest `HOLDOUT_REGISTRY_v<K>.json` there. A registry v4 belongs to the intake owner, and these banks should be folded into it.
- The v4 manifest screens every served record against r1: media sha256, held clip ids, held YouTube ids, exact text and 13-word shingles over every non-template text. It also screens every trained corpus row against r1's additions. Every hit is denied through `deny_records` / `deny_rows`.

### The B2 mixture (manifest v4)

_Filled in when v4 is sealed._
