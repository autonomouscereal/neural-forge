# Chimera Omni architecture review (2026-10-21)

Requested by Cyril on 2026-10-21 (memory event 1228464). This document is a review only. No training job, queue,
cache, ledger entry, registration or model file was started, stopped or changed to write it.

**Evidence key.** Every claim carries one of these tags.
- **[V]** I printed it myself on 2026-10-21 while writing this.
- **[A]** An independent verification pass confirmed it (claims C1 to C7).
- **[A-partly]** The verification pass confirmed only part of it; the correction is given next to the claim.
- **[R]** It comes from an audit report and was not re-verified.
- **UNVERIFIED** is stated outright.

**Abbreviations.**
- `CO` = `C:/Users/cereal/Documents/ai training/chimera-omni`
- `OCL` = `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09`
- `OMNI1` = `/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1` (AI server)
- `STUDENT` = `/mnt/lexar/omni-campaign/distill/fullkd/run1/out/student` (AI server)
- `ev` = an `agent_memory_events.id`

---

## 1. Bottom line

**Yes, we drifted, and the drift is large. It is not quite the drift your note describes.**

**(a) "A rip of K3".**
- **The code repo's design is K3-shaped.** On 07-27 Codex rebuilt the reference profile around K3's mechanisms: exact KDA, gated MLA, depth Block AttnRes, SiTU, and a 93-layer K3-shape test. It also briefly carried K3's MoE. [A]
- **The model we actually built and trained is not K3-shaped.** OMNI1 and the run-1 student are Qwen3.8-27B's topology (Gated DeltaNet + gated attention, plain residual, SiLU), with K3's mechanisms switched off. [V]
- **K3 is in the weights only weakly.** It appears as 12,288 extra FFN channels per layer, added at strength 0.03, with "no measurable held-out text benefit". [A]
- So the repo copied K3's architecture, while the model kept Qwen's architecture and took almost none of K3's knowledge. That is the reverse of what you asked for: K3 knowledge as the main source, K3 architecture as optional inspiration.

**(b) Every tensor carrying every modality.** No, and this is the most serious drift.
- In OMNI1 each media source got its own small transplanted block (layers 32-36). Each block writes only to the rows of its own modality, chosen by type id (`torch.isin(row_types, routed_type_ids)`, `CO/src/chimera_omni/modeling/charted.py:400`). [V]
- The input bridge and output head give each modality its own disjoint feature band and per-type parameters. [A/R]
- Training is split by modality too. The run-1 student is text only: 851 tensors, 0 vision, 0 MTP [V]. Media training (M1) trains only the five media blocks and heads, with the trunk frozen [V].
- This is modality routing plus a miniature tower per source.

**(c) Raw audio, video and images.**
- **In the repo:** yes. Raw patch and waveform code exists, and a 3.19M-parameter proof learned audio and video in and out with one shared trunk [R].
- **In OMNI1:** only vision input is raw pixels. Image, video, audio, speech, music and 3D all live in the parent models' VAE latent spaces and decode through the parents' own VAEs, which sit outside the checkpoint [R].
- **In the run-1 student:** no media path at all [V].

**Where it started.**
1. 07-27: Codex adopted K3's architecture. [A]
2. 09-01: Claude's architecture-budget plan made Qwen the skeleton, made media "a runtime, not a weight", and proposed transplanting a block from each source. [R]
3. 09-06: Claude implemented the type-routed `charted` block. [R]

Each step after that built on these three: the frozen-trunk M1 and the text-only full-KD engine. Section 7 gives the timeline.

**What is still good.**
- The repo's core contract (one trunk, encoder-free boundary, shared head) still matches your intent.
- The tiny joint AV proof is the closest thing we have to your design.
- Run 1 showed that full-parameter KD on the whole trunk recovers intelligence fast (section 8).

---

## 2. What you asked for, verbatim and dated

The source is your own prompts in the agent-memory database. Each quote below was checked to be an exact substring.
The full list (40 fragments, 12 requirements) is in
`C:/Users/cereal/AppData/Local/Temp/claude/C--Users-cereal/0c189f21-3de3-4c40-b42c-db54c9d84462/scratchpad/lab/intended_arch.json`.
It is also in the new database as the `directive` table (136 rows). [A for C1]

| Date (UTC) | ev | Your words |
|---|---|---|
| 2026-05-21 | 23665 | "[Cyril quote withheld]" |
| 2026-05-22 | 27555 | "[Cyril quote withheld]" · input across "[Cyril quote withheld]" and output "[Cyril quote withheld]" · "[Cyril quote withheld]" · "omnimodal at every layer" |
| 2026-05-28 | 54058 | "[Cyril quote withheld]" (masking means "[Cyril quote withheld]") |
| 2026-05-28 | 54885 / 55172 | "[Cyril quote withheld]" · "[Cyril quote withheld]" |
| 2026-06-02 | 90541 | "[Cyril quote withheld]" |
| 2026-06-03 | 94800 | "[Cyril quote withheld]" |
| 2026-06-26 | 163441 | "[Cyril quote withheld]" |
| 2026-06-28 | 192368 | "[Cyril quote withheld]" |
| 2026-07-27 | 390516 | K3: "[Cyril quote withheld]" |
| 2026-08-04 | 506681 | K3 "99% … takes precedence" as weight source; Qwen "a smidge"; "[Cyril quote withheld]"; "[Cyril quote withheld]" |
| 2026-08-12 | 571831 | "[Cyril quote withheld]" |
| 2026-08-28 | 605703 | "[Cyril quote withheld]" |
| 2026-09-01 | 609454 | "[Cyril quote withheld]" |
| 2026-09-02 | transcript `0c189f21…jsonl:440` | "[Cyril quote withheld]" |
| 2026-10-14 (sync date; original date UNVERIFIED) | 904489 / 960655 | "[Cyril quote withheld]" · "[Cyril quote withheld]" |
| 2026-10-21 | 1228464 | "[Cyril quote withheld]" · "[Cyril quote withheld]" · "[Cyril quote withheld]" |

**Requirements that never changed:**
- one model
- dense, no MoE
- the same weights and tokenspace at every layer for every modality
- no encoders, decoders, adapters or codecs
- raw media in and out
- no modality towers and no modality routing
- intelligence before speed
- standard GGUF weights

**Corrections from verification [A-partly, C1].**
- You said the literal words "no moe" **3 times** (ev 90541, 605703, 609454). The audit's "five times" was wrong: 8 of its 11 regex hits were agent-written briefs that had been filed as prompts.
- Counting paraphrases, you rejected MoE **at least 6 times**.
- "Codec-free" rests partly on a question (ev 54885) rather than an order. Ev 94800 and 163441 back it directly.

**Instructions that changed over time.** Please confirm the readings marked **(confirm)**.

| Topic | How it changed |
|---|---|
| Size | ~20B (05-23) → 18-20B (05-28) → 40/20/2B, all modalities at each size (08-04) → "dense … q4 … rtx 3090 with 256k context" (09-01, ev 608920). |
| Context | 1M → 256k (09-01). |
| Video source | LTX → H3 default (08-04) → H3 hybrid, encoder-free (08-28) → "h3 fold not ltx" (09-01). |
| K3 | Weights are the main source (08-04, 09-01, 09-02). Architecture is "inspiration, not adopt" (10-21). **(confirm)** |
| Quantization | "trained at q4" (05-22) → "[Cyril quote withheld]" (08-12, ev 571486, said about folding). Whether you still want Q4-aware training is UNVERIFIED. **(confirm)** |
| MoE as inspiration | ev 892347 lets us "consider" Strata's MoE design for speed-ups. Its original date is UNVERIFIED. |

---

## 3. What we actually built and trained

### 3.1 The three artifacts

**OMNI1** (built 2026-10-13 by `c16_install_media` 1.4.0 from chimera commit `ef1ffb9c`) [V/A]
- 27,535,655,753 parameters, 1,120 tensors.
- 69 layers: 48 `kda` + 16 `gated_gqa` + 5 `charted`, hidden 5120, FFN 16,640.
- `kda_decay_kind=gdn`, `residual_mode=none`, `activation_kind=silu`, `mtp_heads=0`, `mla_latent_size=0`, `gqa_rope_dim=64`, `native_bidirectional_refinement=False`.
- The `kda` layers are Qwen Gated DeltaNet stored under K3's name (`CO/src/chimera_omni/modeling/layers.py:249`, "Qwen gated-delta form") [A].

**Run-1 full-KD student** [V]
- `Qwen3_5ForConditionalGeneration`.
- 851 tensors, of which 0 are `visual` and 0 are MTP.
- Step 611, 19,560,968 tokens seen [R].
- It is a zero-copy qwen3_5-named view of OMNI1's 64 non-charted blocks (`OCL/distill/tools/p1_qwen35_reader.py:6-41`) [R].
- It has not been reassembled into the Chimera layout. Nothing in the ledger records the student or any reassembled child [R].
- It was launched with `--allow-identity-change --no-preregistration` [R].

**Media M1** (`/mnt/lexar/omni-campaign/distill/media_m1/run1/receipt-train-007.json`)
- Verdict PASS, `trainable_parameters` 1,278,333,233, `"trunk": "frozen at the base (stage M1: run 1 owns the trunk)"` [V].
- 300 steps on an RTX 3090.
- Exports the five charted blocks, the music embed and head, and the media head as F32 tensors (5.11 GB). The bridge is not exported [R].
- Training loss fell on six of seven routes. `speech_in` stayed flat (5.365 → 5.352) [R].
- No held-out decode exists, so media capability after M1 is **UNVERIFIED**.

**M2** (the joint text + media stage) is design only (`OCL/distill/MEDIA_FULLKD_DESIGN_2026-10-18.md:67-99`). Only selftests have run [R].

### 3.2 Where the parameters sit (OMNI1) [R; parts sum exactly to the printed total]

| Part | Params | Share |
|---|---|---|
| 64 shared trunk blocks + final norm | 23,598,515,712 | 85.70% |
| ↳ of which K3-derived FFN channels (64 × 3 × 5120 × 12288) | 12,079,595,520 [A] | 43.9% of the model |
| Text vocab rows (embed + lm_head) | 2,542,796,800 | 9.23% |
| Control and tool band rows | 9,175,040 | 0.03% |
| **Modality-specific, total** | **1,385,168,201** | **5.03%** |
| ↳ 5 charted blocks | 591,437,129 | 2.15% |
| ↳ music_codes band (65,536 rows × 2) | 671,088,640 | 2.44% |
| ↳ media / grounding / sync band rows | 32,768,000 | 0.12% |
| ↳ native_media_bridge | 58,284,032 | 0.21% |
| ↳ native_media_head | 31,544,320 | 0.11% |
| ↳ flow / grounding / sync aux heads | 46,080 | ~0% |

### 3.3 Per modality

"Shared trunk" below means the modality's rows pass through the 64 shared blocks. **The trunk has never been trained on media.** M1 froze it, and stage D trained it on text only.

| Modality (type id) | Input form | Output form | Weights specific to it | Routing / masking by modality | Trained? |
|---|---|---|---|---|---|
| Text (0) | tokens | tokens | text vocab rows | the charted blocks skip text rows ("exact identity on text"); band mask at decode | full-KD run 1 (student, text only) |
| Vision-in / "ocr" (6) | **raw** 2×16×16×3 pixel tubelets (1,536 values) in feature columns 640-2176, through a linear map fitted to imitate Qwen3-VL-32B's ViT and merger [R] | one zero output row (input only) | `qvl32.vit.block13.mlp` charted block (layer 36, 19.6M), bridge band, per-type affine | block 36 writes only type-6 rows | M1 `vision_in` route |
| Image (1) | Qwen-Image **VAE latents**, columns 512-576 [R] | latents → parent VAE decoder (companion) | `qimg.mmdit.block30.img_mlp` (layer 35, 155.0M) | block 35 writes only type-1 rows | M1 `image_generate` |
| Video (2) / audio (3) | H3 **VAE latents**, columns 384-480 / 480-512 [R] | latents → H3 VAE (companion) | `h3.dit.block.mlp` (layer 33, 267.9M); it reads text context but writes only AV rows [R] | block 33 writes only type-2/3 rows | M1 `joint_av`, `speech_in/out` |
| Speech in/out | H3 audio latents [R] | H3 audio latents | as above; TTS type 5 has no unit [R] | as above | M1 (speech_in flat) |
| Music (4) | ACE VAE **latents**, columns 0-384, plus a 65,536-row code band [R] | latents/codes → ACE decoder | `ace.dit.block12.mlp` (layer 32, 102.8M) + 671M band | block 32 writes only type-4 rows; the band must be masked for text decode (unmasked it wins argmax at 146/210 text positions [R]) | M1 `music_*` |
| 3D (7) | HY3D ShapeVAE **latents**, columns 576-640 [R] | latents → HY3D decoder | `hy3d.dit.block10.mlp` (layer 34, 46.2M) | block 34 writes only type-7 rows | not a distinct M1 route [R] |

Block positions and type ids were printed today [V]: ace [4], h3 [2,3], hy3d [7], qimg [1], qvl32 [6]. The
per-block parameter counts come from the audit [R].

**Qwen 27B's own vision tower** was never folded into OMNI1 (no `model.visual` tensors), which matches your note ev 904489 [R]. The "vision-in" capability comes from a borrowed Qwen3-VL-32B **ViT block**, which is an encoder component [R].

Repo self-checks that flag the problem:
- `CO/src/chimera_omni/eval/native_frozen.py:160-161` refuses companion decoders: "companion decoders cannot provide native outputs" [V].
- `CO/docs/ARCHITECTURE.md:12` says "One trunk: no modality-specific reasoning towers" [V].
- OMNI1 was built in conflict with both.

### 3.4 How a token flows (code)

- **Text:** `embed` → 64 shared blocks (norm → mixer → norm → dense FFN; no block reads type ids) → final norm → `lm_head`. The charted blocks are skipped on text-only passes (`modeling/chimera.py:497-504`). The S7 check measured `max_abs_logit_diff_omni1_vs_s5 0.0` [R].
- **Media:**
  1. A feature vector enters `NativeContinuousMediaBridge` (`chimera.py:24-127`). The `feature_proj` is shared, but each modality uses disjoint columns and has its own `type_embed`, `type_feature_scale` and `type_affine` [R].
  2. The rows pass the shared trunk, but no media gradient ever reached it.
  3. At layers 32-36 each media row is written to **only** by its own source's block (`charted.py:11-14`, `:400`, `:479-483` exact-zero scatter) [V/A].
  4. Output goes through the shared `UnifiedNativeMediaEmissionHead`, with a per-type affine and a per-modality output band [R], and is then decoded by the parent model's VAE outside the checkpoint [R].

---

## 4. K3: adopted or inspired, component by component

**Your 07-27 instruction.** It said "[Cyril quote withheld]" So changing the model was within the letter of the instruction [A-partly, C2]. What went wrong is how "adjust if needed" was read: Codex read it as "match K3", not "take what is clearly better".
- The proof offered was unit tests ("12 passed focused K3 tests", "35 passed") and a CUDA forward/backward smoke test.
- No ablation compared K3's mechanisms with what we had [A].
- Its own doc calls every pre-K3 deviation "architecture errors or material omissions" (`CO/docs/KIMI_K3_ALIGNMENT.md:39`) [V].

**History of the K3 code.** [A, C3]
- The K3 work, including `StableLatentMoE`, sat uncommitted from 07-27 to 08-28. Memory ev 390959 shows the MoE in the working tree on 07-27.
- It was removed after your 08-28 "no moe" (ev 605703, 605959).
- It landed in commit `3e2623e` with the MoE already gone. `git log --all -S StableLatentMoE` finds nothing in any repo.

| Component | Kimi K3 | Repo reference profile (`k3_aligned_test_config`, `config.py:485-521`) | OMNI1 / run-1 student (what was trained) | Verdict |
|---|---|---|---|---|
| Linear mixer | KDA (per-channel decay, low-rank lift) | exact KDA | Qwen Gated DeltaNet (`kda_decay_kind=gdn`, rank 0) [V] | Repo: **adopted**. Model: Qwen |
| Full attention | gated MLA, NoPE + 64 rope | gated MLA (scaled down) | Qwen gated GQA, all `mla_*`=0, rope 64 [V] | Repo: **adopted**. Model: Qwen |
| Layer pattern | 3 KDA : 1 MLA + forced final MLA, 93 layers | adopted; 93-layer, 7168-wide shape test (`tests/test_kimi_k3_alignment.py:31-60`) [A] | 3:1, 64 layers, no forced final [V] | 3:1 is also Qwen's native interval; the forced final layer is K3-only and unused |
| Depth mixing | Block AttnRes | adopted | `residual_mode=none` [V] | Repo: **adopted**. Model: off |
| Activation | SiTU | adopted | SiLU [V] (β values stored but inert) | Repo: **adopted**. Model: Qwen |
| Width mixing | Stable LatentMoE (896 experts, top-16) | added 07-27, removed 08-28 | dense | rejected, per your ruling |
| Positions | NoPE | NoPE, hard-coded in `architecture.py:28` [V] | partial mRoPE, θ 1e7 [A] | Model: Qwen |
| Tokenizer | 163,840 tiktoken | n/a | Qwen 248,320 (+ Chimera ledger bands in OMNI1) [R] | Qwen |
| MTP | yes (per report) | optional | 0 heads; the student config declares 1 MTP layer but has 0 MTP tensors [V] | none |
| Vision | MoonViT encoder | not copied (encoder-free boundary, original) | per-modality charted blocks (section 3) | not K3, but not your design either |
| **K3 weights** | n/a | n/a | 12,288 of 16,640 FFN channels (73.8%) per layer, added at **0.03** [A, C6]; R-Q agreement 110/210 with K3 vs 108/210 for the S3 trunk without it (sweep figure; the w=0 control scored 107) [A]; "no measurable held-out text benefit" (`OCL/composite/c10/design_sweep/VERDICT.md:17`) [A] | the opposite of "99% K3" |

**The repo still presents the K3 shape as "the architecture".** [A-partly, C4]
- `CO/src/chimera_omni/architecture.py:15` defaults to `k3_aligned_test_config()`.
- Lines 27-28 hard-code `"depth_mixing": "block_attnres"` and the NoPE KDA/gated-MLA policy, whatever the config says [V].
- So `architecture_manifest.json` and `CO/docs/ARCHITECTURE.md:41-57` describe a K3-shaped model.
- That profile was built and trained only as a 5-layer, 128-wide toy, in the AV-flow proof (`proof/h3_hybrid_av_flow.py:150-158, 222-292`) and in the behaviour scripts. It was never built at production scale.

**One more confusion: the 09-01 plan's wording.**
- On 09-01 you chose "the chimera omni architecture" (transcript `a763fdb3…jsonl:8636`).
- The plan recorded that choice as "(KDA, gated-MLA, Block-AttnRes, SiTU…) … export = `kimi-k3`-shaped GGUF" (`neural-forge/docs/…/CHIMERA_OMNI_FOLD_PLAN_2026-09-01.md:434`) [R].
- So "our architecture" had silently come to mean "K3's architecture".

**K3-inspired variants that were planned but never tested.** Fold tests T2 (GDN→KDA decay), T3 (gated attention→MLA) and T4 (finite SiTU β) have no receipts. **UNVERIFIED** whether they ever ran. These are exactly the ablations an "inspiration" approach would need.

---

## 5. The vision on/off switch and every other modality switch

The model config `ChimeraConfig` has **no** vision on/off flag (`CO/src/chimera_omni/config.py:47-114`) [R]. Modality presence is structural, set by `native_type_count`, `MEDIA_TYPE_IDS` (`native_media.py:10-18`), `charted_blocks[].type_ids`, `ledger_bands`, `aux_heads` and `native_bidirectional_refinement`.

The switches that actually turn vision or media off are in tooling and configs:

| Where | What it does | Evidence |
|---|---|---|
| `OCL/distill/tools/streamed_kd_trainer.py:1894-1896` | `visual = getattr(inner, "visual", None)` … `inner.visual = None`. The stage-D/full-KD engine strips the vision tower | [V] |
| `OCL/distill/tps_probe.py:1344, 1351` | `if args.drop_vision and vision is not None:` … `facts["vision_tower_dropped"]`. Used by `run_tps_final_2026-09-02.sh:72` and `run_tps_final_b_2026-09-02.sh:75` | [V] lines; [R] scripts |
| `STUDENT/config.json` | Declares `Qwen3_5ForConditionalGeneration`, `language_model_only: false` and a full `vision_config`, while the weights hold **0** vision tensors. The parent's config was passed through (`p1_qwen35_reader.py:940-947` → `streamed_full_kd_trainer.py:722`) | [V] values; [R] lines |
| `OCL/composite/tools/run_omni_frozen_review.py:2021` | `--vision-cards` is opt-in per run | [R] |
| `oc_run_frozen_bank.py:138-236` | `--text-only` flag | [R] |
| `student_test.py:6` | "no vision tower" | [R] |
| OMNI1 config `native_bidirectional_refinement=False` | the joint media attention from the AV proof is off | [V] |
| `charted.py:400` type-id routing; `chimera.py:198-210` band mask at decode | modality routing in the model itself | [V]/[R] |

**Which note you saw is UNVERIFIED.** The most likely candidates:
- The student `config.json` above (vision declared, none present).
- This session's own transcript (`C:/Users/cereal/.claude/projects/C--Users-cereal/0c189f21-3de3-4c40-b42c-db54c9d84462.jsonl`, around lines 6397-6425): "if the student includes the vision path … Skip this if it doesn't" and "It has no vision weights, so I skipped the image questions" [R].

Either way the substance is the same. The trained student cannot see, and the only vision weights live in a type-routed block that the text training never touched.

---

## 6. Raw audio, video and image support today

| Layer | Raw image | Raw video | Raw audio | Evidence |
|---|---|---|---|---|
| Repo code | yes: `image_to_native_patches` (`native_media.py:140`) | yes: `video_to_native_patches` (`:201`, "Patchify raw video tubelets") | yes: `waveform_to_native_segments` (`:324`) | [R] |
| Tiny AV proof (08-28, 3.19M params) | n/a | **in and out, raw tubelets** | **in and out, raw 32 kHz waveform** | `CO/docs/H3_HYBRID_AV_FLOW_PROOF_REPORT.md:31-38, 89-94` [R] |
| OMNI1 | input only (vision-in/ocr type 6); image generation runs in Qwen-Image VAE latents | H3 VAE latents (frame-by-frame vision-in is planned) | H3 VAE latents | [R] |
| Run-1 student | none | none | none | [V] |

What did not carry over from the AV proof into the 27B [R]:
- **Raw input and output:** no; replaced by H3 latents.
- **Shared trunk learning AV:** no; one H3 block learns it, with the trunk frozen.
- **Joint media attention:** off.
- **Text and AV trained together:** not yet (M2 is design only).

Quality on record [R]:
- At step 1, OMNI1's H3 output was **worse than the noisy input** it was given, by 3.5-4.4 dB on video and 9.5-14.3 dB on audio (`OCL/composite/s7/CHIMERA_OMNI_STEP1_RESULTS.md:160-164`).
- Vision answers were not grounded.
- After M1, held-out media quality is **UNVERIFIED**.

The repo also records its own capacity gap. The latent bands already need 7,712 feature slots, more than the bridge's 6,144 (`H3_NATIVE_CHARTED_CONTRACT.md:114-125`) [R].

---

## 7. Drift timeline and root causes

| Date | Decision | Who | Against your words? | Evidence |
|---|---|---|---|---|
| 05-23 | Research DeepSeek CSA/HCA, MiniMax, Kimi KDA, residual attention | **Cyril** | — (your inspiration list) | ev 28239, 28337 [R] |
| 06-28 | Repo foundation: one dense trunk, encoder-free patch/segment boundary, shared head, tests forbidding towers | Codex | **matches** | commit `6350b28`; `ARCHITECTURE.md:3-16` [R] |
| 07-27 | K3 alignment: exact KDA, gated MLA, Block AttnRes, SiTU, **Stable LatentMoE**, 93-layer K3 shape test | Codex (gpt-5.6-sol) | adopts rather than borrows; the MoE breaks 06-02 | ev 391598; [A] C2/C3 |
| 08-28 | MoE removed after "no moe"; K3 profile committed in `3e2623e` | Codex | MoE fixed; the K3-shaped reference stays | [A] C3 |
| 08-28 | Dense H3-inspired raw AV path; 3.19M proof learns AV in and out in one trunk | Codex + Claude | **matches; the closest we have been** | `3e2623e`, `cc67c49` [R] |
| 08-28 | Media models kept as edge-only teachers; the plan benchmarks the parents | Claude draft, Codex | fine for teachers; benchmarking parents went against "[Cyril quote withheld]" | `MEDIA_TEACHER_STACK_PLAN.md:11,29,49` [R] |
| 09-01 | Architecture-budget plan: **Qwen3.8-27B skeleton**; "Media generation is a runtime, not a weight"; companion `-media.gguf`; Qwen ViT kept as fallback; a transplanted DiT block per source | Claude | **yes: the root of the towers.** Your "[Cyril quote withheld]" came the same evening (ev 609454) | `neural-forge/docs/chimera-omni-plan-inputs-2026-09-01/8_plan_architecture_budget.md:7-11,148-149` [R] |
| 09-01 | The plan was built from **one session's prompts** (07-31 to 09-01), so the 05-22/05-28 founding spec was missing | Claude | process failure | `0_read1_prompts.md:1` [R] |
| 09-01 → 02 | Import switches so the Chimera layer types reproduce Qwen exactly (GDN decay, gated GQA, identity AttnRes) | Claude | neutral mechanism; the result is Qwen topology under K3 names | `4be3b0a`, `59a5e6d` [A, C5] |
| 09-02 / 09-03 | Throughput probes run `--drop-vision`; 27B import keeps vision "as side records"; Qwen ViT never absorbed | Claude | yes (your ev 904489) | `TPS_PROBE_RESULTS_2026-09-02.md:42`; `CAMPAIGN.md:281` [R] |
| 09-06 | **`charted` block type: writes only on rows whose type id it owns** | Claude (`94c42c6`); Codex H3 variant (`61b82b1`) | **yes: modality routing implemented** | `charted.py:11-14,400` [V] |
| 09-07 → 10-10 | S3: 4x fold of all 64 Qwen MLPs (17,408 → 4,352). Text column B fell 12 → 1 | Claude runbook | not architecture, but it damaged intelligence; the plan's own estimate said the net fold should be ~1.0x | `COMPOSITE_STEP1_RUNBOOK.md:18,97`; foldhist [R] |
| 10-08 → 11 | K3 added at additive 0.03, "not because K3 improves the text" | Claude | yes (against "99% K3") | `VERDICT.md:1-22` [A, C6] |
| 10-12 | Stage-D design: the text engine "drops `model.visual`"; media in separate per-modality lanes | Claude-led (authorship split UNVERIFIED) | yes | `DISTILL_DESIGN_2026-10-12.md:50,54,113,284` [R] |
| 10-13 → 14 | Codex branch `native-unified-modalities` (model-owned inputs, QVL raw-patch frontend) | Codex | **closer to your intent; never merged** | `33dd188`, `8deb4ea` [R] |
| 10-14 | The vision-in/video-in answer cache was dropped without anyone deciding it | agent gap | yes (omission) | `STATE_2026-10-14_CODEX_HANDOFF.md:120` [R] |
| 10-17 | Realignment: full-parameter KD, no LoRA. It read prompts **from 07-01 only**, so "same weights / every layer" fell out again; it lists "Qwen VL tower (vision-in)" as a teacher | Claude, approved by you | intent right, but incomplete | `REALIGNMENT_2026-10-17.md:5,17-19,59` [R] |
| 10-18 | Run 1: text-only qwen3_5 student. M1: the five charted blocks trained with the trunk frozen | Claude | yes: modalities trained on disjoint tensors | [V] receipt + student index |

### Root causes

1. **Two meanings of "K3" got merged.** You wanted K3 *knowledge* to dominate and K3 *architecture* reviewed. We did the opposite: adopted the architecture in the repo, and gave the knowledge 0.03.
2. **Plans were not checked against your full history.** Both realignment passes (09-01 and 10-17) read only part of your prompts. Neither saw the 05-22/05-28 founding spec, which is where "same weights at every layer" and "no masking into separate models" live. REALIGNMENT itself admits the agent plans "were never checked against his words".
3. **The fold method works one source at a time.** Closed-form chart fits take one source block at a time, so each source naturally became its own module. Folding everything into shared weights was declared infeasible (`8_plan:11`), and per-source transplants won by default.
4. **Proof-campaign rules were reused on the production model.**
   - The ≥4x-per-component rule ruled out the shared-weight "ARMBLEND" option, whose rows "cannot meet >= 4x for ACE (0.625) or H3/QIMG (0.417)" (`COMPOSITE_STEP1_RUNBOOK.md:102`) [R]. ARMBLEND was never built (UNVERIFIED beyond the inventory).
   - The bit-identical-text gate forced media to stay off text rows.
   - The 4x rule also drove the S3 trunk fold that cost the intelligence.
5. **Tooling gravity.** Every engine with measured throughput is HF `Qwen3_5TextModel` (fla kernels, llama.cpp `qwen35` export). That pulled training into a text-only Qwen layout, and vision kept being dropped "to save memory".
6. **Resource limits.** 251 GB host RAM, 24 GB cards and a 1.37 TB K3 pushed toward separate text and media lanes and a frozen trunk.
7. **The schedule rewarded "all seven sources present"** (the 09-02 step-1 ruling) over shared-weight design.

---

## 8. What the run-1 recovery tells us

Printed today from the new database, on the same bank_v1 cases (`omni_lab.py compare 30658 30666 30676`, 0.48 s wall) [V]. Grades are reader grades by Claude; C = correct, P = partial, M = miss, N = no content:

| Case | Parent (S0 import) | OMNI1 | Run-1 student |
|---|---|---|---|
| conversation-natural-greeting | C | N | P |
| instruction-three-actions | C | M | P |
| explanation-seasons | C | M | M |
| arithmetic-multi-step | C | M | M |
| logic-ordering | C | M | C |
| context-retrieval | C | N | C |
| summarization-priorities | C | P | C |
| multi-turn-correction | C | N | P |
| **Totals on these 8** | **8 C** | **1 P, 4 M, 3 N** | **3 C, 3 P, 2 M** |

On the student's 4 new cases: tool-call C; word problem, Python and factual all M.

**What this tells us:**
- **Full-parameter KD on the whole trunk works, and works fast.** About 19.6M tokens (611 steps) [R] took the model from mostly broken to mostly coherent. So the S3 4x fold's damage is largely recoverable, and the 27B weights still carry the parent's capability.
- **The recovery was text-only.** The student has no vision or media tensors, so it says nothing about media. Note too that the parent and OMNI1 rows ran through the Chimera engine, while the student ran through the qwen35-student engine.
- **The sample is small.** 8 cases, one grader (Claude). Several OMNI1 answers ran into the cap. Treat this as a direction, not a score.
- **This is the piece that carries over best into any realignment.** A truly shared all-modality trunk is only reachable by training (KD) rather than closed-form folding, and run 1 is our only evidence that full-parameter KD on this trunk is affordable on our hardware.

---

## 9. Realignment options (no decisions made, no actions taken)

**The question:** how close do we want to get to "one dense trunk, every layer's weights processing every modality as raw tokens, no routing or towers"?

### Option 1: Keep the current shape and finish M2

What it involves:
- Reassemble the run-1 text trunk with the M1 media blocks.
- Run M2's joint training.

Trade-offs:
- **Pro:** fastest; reuses everything.
- **Con:** it keeps the five type-routed blocks, the per-type bands, parent VAE latents and companion decoders. It still breaks your 09-01 and 10-21 rules.
- **Con:** media quality is capped by one transplanted block per source.

**Not recommended as the destination.** It could serve as a short checkpoint to measure media at all.

### Option 2: Un-route what exists (a middle path)

What it involves:
- Remove the type-id gating so every row passes through every layer, including former blocks 32-36. They become ordinary shared layers.
- Collapse the disjoint bridge and head bands into one shared projection.
- Unfreeze the trunk for media.
- Train text and media in the same batches (full-param KD, as in run 1).

Trade-offs:
- **Pro:** reuses OMNI1, the run-1 weights, M1's learned blocks and the existing trainers. It satisfies "no routing" on paper.
- **Con:** the former blocks are still single-source DiT/ViT pieces inside the trunk, so they behave as narrow sub-towers in practice.
- **Con:** text will move. The bit-identical-text gate has to be dropped and replaced by behaviour grading.
- **Con:** I/O is still parent VAE latents unless the bridge is changed.

### Option 3: A truly shared all-modality trunk (what you described)

**What it requires:**
1. **One raw boundary.** Raw patches and waveform segments go through one shared projection: the repo's `native_media.py` patchify code plus a single typed position scheme, with no disjoint per-modality bands and no VAE. This fits the 08-28 AV proof and Codex's unmerged `native-unified-modalities` branch.
2. **Every trunk layer trained on mixed text, image, video, audio, music and 3D sequences**, with full-param KD.
3. **The parent models become teachers, not blocks.**
   - Qwen3.8-27B's own vision, Qwen3-VL-32B, Qwen-Image, H3, ACE, HY3D and an ASR model provide targets (answers, latents decoded to raw, captions).
   - K3 provides text and tool knowledge at real strength. That means K3 as the main text teacher; REALIGNMENT already sets `lambda_k3 1.0`.
4. **Raw output through the shared head with a flow objective** (as in the AV proof), and the decoder distilled into the trunk.
5. **A training engine that runs the Chimera layout with media rows.** Today's measured engines are Qwen3_5 text-only. This is the biggest engineering item.
6. **A raw-media data pipeline and the omni_lab suites for every modality**, so progress is measured per modality from day one.

**What carries over:**
- The run-1 trunk weights, as the init. They carry the recovered intelligence.
- The full-KD trainer design, teacher caches and the corpus.
- The tiny AV proof.
- `native_media.py`.
- The Codex native-unified branch.
- The omni_lab database and runner.

**What gets discarded:**
- The five charted blocks and their chart fits.
- The latent interface bands.
- The music code band (UNVERIFIED whether it is needed).
- The companion-decoder plan and the `-media.gguf` idea.

**Risks:**
- Raw-pixel and raw-audio generation quality at this scale with our compute is unproven beyond the 3.19M proof.
- Media training data volume and GPU time are much larger than for text.
- Q4 + 256k context on a 3090 at ~27.5B has not been measured. It is UNVERIFIED that it fits.

### Option 4: The K3 decision, independent of options 1-3

Choose one:
- **(a) Keep Qwen3.5 topology.** It is what was trained and what llama.cpp exports as standard GGUF.
- **(b) Treat K3's mechanisms as inspiration and test each one.** Run T2 (KDA decay), T3 (MLA) and T4 (SiTU) on a small model through omni_lab, and adopt only what wins on behaviour.

Either way, the repo's `architecture.py` defaults and `ARCHITECTURE.md` should stop describing the K3 toy profile as "the architecture". That is a doc change; it has not been made.

### Open questions for you

1. Do you confirm the K3 split: knowledge is the main source, architecture is inspiration only?
2. Target size and context: is "dense, Q4 on a 3090, 256k" still the goal, and is ~27.5B acceptable?
3. Is Q4-aware training still wanted (05-22), or is quantization post-hoc (08-12)?
4. Option 3's raw output means distilling the parent decoders into the trunk. Is a temporary latent rung acceptable while that trains, or must raw be there from the start?
5. The music code band (65,536 vocab rows): keep it or drop it?

---

### Appendix: key files

- `CO/docs/ARCHITECTURE.md`, `CO/docs/KIMI_K3_ALIGNMENT.md`, `CO/docs/H3_NATIVE_CHARTED_CONTRACT.md`, `CO/docs/H3_HYBRID_AV_FLOW_PROOF_REPORT.md`
- `CO/src/chimera_omni/{config.py, architecture.py, modeling/charted.py, modeling/chimera.py, modeling/layers.py, native_media.py, importing/profiles.py}`
- `OCL/composite/s7/CHIMERA_OMNI_STEP1_RESULTS.md`, `OCL/composite/c10/design_sweep/VERDICT.md`
- `OCL/distill/{DISTILL_DESIGN_2026-10-12.md, MEDIA_FULLKD_DESIGN_2026-10-18.md, REALIGNMENT_2026-10-17.md}`
- `C:/Users/cereal/Documents/ai training/neural-forge/docs/chimera-omni-plan-inputs-2026-09-01/8_plan_architecture_budget.md`
- Server: `OMNI1/config.json`, `STUDENT/{config.json, model.safetensors.index.json}`, `/mnt/lexar/omni-campaign/distill/media_m1/run1/receipt-train-007.json`
- Database: `omni_lab` (tables `directive`, `ruling`, `document`, `eval_*`). See `OCL/OMNI_LAB_GUIDE.md`.
