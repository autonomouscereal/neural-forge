# Architecture and folding drift findings, 2026-10-22

This is a read-only audit. Nothing was trained, folded, deleted, started or stopped. It answers Cyril's 2026-10-22 request: "[Cyril quote withheld]" It reports only what the record shows. It proposes no designs.

**Evidence tags**
- **[C-nn]** The workflow's verification pass confirmed this (claim id nn).
- **[P-nn]** The verification pass partly confirmed this. The correction it recorded is already applied below.
- **[S]** I checked this in the final pass by reading the cited line myself.
- Anything else from the research extracts is left out of the body. Claims that were refuted, not found, or not re-checked are listed in Appendices A and B so nobody reuses them.

**Path keys**
- `OCL` = `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/`
- `NF` = `C:/Users/cereal/Documents/ai training/neural-forge/`
- `CO` = `C:/Users/cereal/Documents/ai training/chimera-omni/`
- `BR` = branch `claude/arch-v2-20261021` at `C:/Users/cereal/Documents/ai training/.agent-worktrees/chimera-omni-5d4d4e1dcd/workers/claude/arch-v2-20261021`
- `MEM` = `C:/Users/cereal/.claude/projects/C--Users-cereal/memory/`
- `ARCH` = `OCL/ARCHITECTURE_V2_2026-10-21.md`
- `FOLD` = `OCL/V2_FOLD_INIT_PLAN_2026-10-21.md`
- `ev` = an `agent_memory_events.id`; `d` = an omni_lab `directive.id`

---

## Summary

1. **The "August version" is the 2026-08-28 joint audio/video x0-flow proof** (chimera-omni `443cf0f` → `cc67c49`): 3.19M parameters, dense, trained from random init on synthetic data. **Nothing in it was folded.**
   - It proved raw video and raw audio going in, and joint raw video plus audio coming out. It did not test image edit, 3D, or natural speech and music.
   - It also used per-type tables and a typed head. [C-01, C-02, P-03]
2. **v2 changed almost everything at once compared with August:**
   - trunk kinds, positions and activation;
   - unit size (1,536 → 3,072) and head (linear → GLU MLP);
   - typing, and the data and tasks.

   Then it failed. **Image edit, video out and speech out never passed in any of 15 proxy arms, including the typed control.** The planned rerun of the August proof under v2 (A6) was never run. [C-06, C-08, P-09]
3. **Most v2 choices were introduced by agents**, under Cyril's 10-21 delegation of "all of the details". That covers the Qwen3.5-shaped 3 GDN : 1 gated-GQA trunk, the widths, the vocabulary, RoPE/mRoPE, Block AttnRes with block 4, A6b, the 3,072 unit and the GLU head. None was A/B-tested against the August components. [P-06]
4. **Block AttnRes was mapped onto Cyril's "[Cyril quote withheld]" by an agent (ARCH line 287).**
   - His "residual attention folding" is a July *folding technique* (coupled QKVO, residual-aware), not a model layer.
   - The v2 fold plan does not use that technique. AttnRes's own ablation, A1, never ran. [C-07]
5. **The v2 fold plan takes a trained 27B-topology trunk, "T", as its default primary source** (FOLD line 22). Cyril said to throw that trunk away (ev 1241896). [S]
   - The plan's proxy fold of T collapsed: R2 0/9 in every arm, worse than a unigram model (FOLD lines 382-389). [S]
   - Its media boundary and head were zero placeholders, not folds. [S]
   - Its only token metric is held top-1, used as a diagnostic (FOLD line 160). Cyril asked for top-5 on 10-13 and top-10+ on 10-22. [S]
6. **The record proves the following about folding:**
   - Single-layer excision works (LFM2).
   - Per-site MLP coresets at 4x work (LFM2).
   - Hidden-width folds die once composed with other axes, or below roughly 0.86x of width on LFM2.
   - No proof exists for any multi-layer merge, any kind translation (KDA→GDN, MLA→GQA, SiTU→SiLU), or any residual-attention fold at 4x or more. [S, C-06]

---

## 1. The August all-modality proof

### 1.1 Which work Cyril means

- **His words.** On 2026-10-21 Cyril wrote "[Cyril quote withheld]" (ev 1232245, 20:27Z; ruling at `OCL/distill/STAGE_D_RULINGS_2026-10-12.md:113-120`). [C-01, S]
- **The "demonstration".** It is Claude's message at 2026-10-21T16:18:25Z (transcript `C:/Users/cereal/.claude/projects/C--Users-cereal/0c189f21-3de3-4c40-b42c-db54c9d84462.jsonl`, line 6572): "The 3.2M-parameter August proof, where one shared trunk learned raw audio and video in and out, never carried over to the 27B." Claude's only other August mentions before his prompt (L6581, L6624) name the same "August AV proof". [C-01]
- **Commits.** All are in `CO` and dated 2026-08-28. [C-01]
  - `3e2623e`: dense H3-inspired native media architecture (Codex).
  - `443cf0f`: evidence-grade proof.
  - `c5a5c18`: schedule fix; this is the proof's source tree.
  - `cc67c49`: passing GPU proof recorded.
  - `c1ca0b2`: v3 diagnostics, "Kept for lineage; not evidence for the current model".
- **Acceptance.** Cyril on 2026-08-28: "[Cyril quote withheld]" (`MEM/chimera-omni-h3-media-goal.md:34`). [P-03]
- **omni_lab has no row** for this proof or any other August fold (zero `run` matches). [C-04 evidence, P-09]

### 1.2 What the model was

Sources: `CO/src/chimera_omni/config.py:225-261` (`k3_aligned_test_config`); `CO/src/chimera_omni/proof/h3_hybrid_av_flow.py`; `CO/docs/H3_HYBRID_AV_FLOW_PROOF_REPORT.md`, all at `c5a5c18`/`cc67c49`. [C-01, C-02]

- **Size and shape:** 3,185,932 parameters, dense. Hidden 128, FFN 384, 5 layers.
- **Mixers:** pattern `(kda, kda, kda, gated_mla)` with `force_final_gated_mla=True`.
- **Other trunk settings:** `block_attnres` (block 2), SiTU activation, NoPE (`architecture.py:29`, `layers.py:429`), dense FFN.
- **Weights:** random init. `build_model` loads no weights, and the script has no fold or load path.
- **No H3 at all:** "No MiniMax H3 weights, code, inference, or outputs were used" (proof L791).

### 1.3 How each modality went in

- **Raw video:** 32x32 RGB, 8 frames. 16x16 patches over 2-frame tubelets give rows of 1,536 values (`FEATURE_DIM = 2*3*16*16`, proof L41-56).
- **Raw audio:** 32 kHz stereo waveform in 1,536-sample segments.
- **The bridge.** Both go through `NativeContinuousMediaBridge` (`chimera.py:24-126`), which has shared and per-type parts:
  - shared `feature_proj`, `position_proj`, `position_fourier_proj` and `noise_proj`;
  - **per-type tables:** `type_embed`, `type_feature_scale` and `type_affine`.
- **Encoders:** none. There is no encoder, VAE or codec.
- **Not a pure untyped design.** The report itself says "One packed typed video/audio suffix and one shared media head" (L33). The alignment doc lists "Type-conditioned boundary modulation" as an edge boundary. [P-01 correction]

### 1.4 How each modality came out

- **Head:** `UnifiedNativeMediaEmissionHead` (`chimera.py:129-148`): RMSNorm, then a per-type affine, then **one linear layer, 128 → 1,536**.
- **Generation:** rectified flow predicting x0, sampled with `rectified_flow_x0_sample` in 16 Euler steps, with a video shift of 12.0 and an audio shift of 3.0 (report L34-37).
- **Refinement:** bidirectional only in the gated-MLA layers (`config.py:53`, `layers.py:583-637`). Media queries attend to the text prefix and the media set; text stays causal.
- **Reconstruction:** native inverse patch and segment functions write PNG, MP4 and WAV, with no decoder (report L38). [C-02]

### 1.5 Results (v4, 64 held-out examples)

Sources: report L41-58; `CO/proof_outputs/h3_hybrid_av_flow_v4/report.json`. [P-03, confirmed numbers]

| Metric | Frozen (untrained) | After | Gate |
|---|---|---|---|
| AV understanding | 0.1875 | **1.0000** | ≥ 0.90 |
| Generated motion direction | 0.1875 | **1.0000** | ≥ 0.85 |
| Generated pitch | 0.5469 | **1.0000** | ≥ 0.85 |
| AV onset sync within 0.15 | 0.0000 | **0.7969** | ≥ 0.75 |

- **Training:** 4,000 steps, batch 16, on synthetic data: a moving orange square, and 440/880 Hz stereo sines.
- **The report's own limits** (L96-99): "a controlled learnability and wiring proof. It does not prove photorealism, natural music or speech, H3 quality parity…"
- **Coverage of other modalities:**
  - Still image, music and TTS-style output were tested only in the **June 2026** tiny synthetic proofs: `cd6c154`, `1c759e0` and `9f24e2b`. The music gate failed in `925a083`.
  - **Image edit and 3D:** no test found in the June or August proof reports. [P-03 correction]

### 1.6 The August folds (separate from the AV proof)

**The sealed, decoded one: neural-forge `a9abb688`, 2026-08-04.** Sources: `NF/reports/chimera-omni-fold-readiness-20260804/README.md`; `NF/docs/CHIMERA_OMNI_FOLD_COVERAGE_2026-08-04.md`. [P-04]

- **Parent:** a 180-step model on 64-dim synthetic features.
- **The fold:** width only, 128 → 32. One global chart, closed form. Depth untouched.
- **Results:**
  - Masked MSE: parent 0.0524, child 0.1221, blank child 0.8523, so the fold closed 91.3% of the gap.
  - Top-1 agreement with the parent: 73.3%. This is **top-1 only**.
  - Audio render: **worse than blank**.
- **Its own caveat:** "It shows the mechanism, not capability retention" (coverage §5 L332-333).

**Other August folds** ran between 2026-08-04 and 2026-08-05: `d7da31f3`, `92faf1c1`, `b9e37374` and `ecb16477` (NOTES-W11A/B, W12C, W12F). [P-04 correction]

- They executed ledgered, exact-4x **width** folds (128 → 34 and 128 → 31) of test-scale `ChimeraOmniModel` instances.
- Those instances included local, csa/hca, kda and gated_mla mixers and Stable LatentMoE.
- Results are per-branch fit numbers, **not behaviour**.

**What none of the August work did:**
- No August fold was ever applied to the AV proof model.
- No August work folded depth.

---

## 2. The v2 architecture, component by component

**Cyril's yardstick** (verbatim, d684-d693 from ev 1230100 on 10-21, plus d5, d28, d29 and d131). [C-05]

| Id | Cyril's words |
|---|---|
| d684 | "[Cyril quote withheld]" |
| d685 | "[Cyril quote withheld]" |
| d686 | "[Cyril quote withheld]" |
| d687 | "[Cyril quote withheld]" |
| d692 | "[Cyril quote withheld]" |
| d688 | "[Cyril quote withheld]" |
| d689 | "[Cyril quote withheld]" |
| d690 | "[Cyril quote withheld]" |
| d691 | "[Cyril quote withheld]" |
| d131 | "[Cyril quote withheld]" |
| d5 | "[Cyril quote withheld]" |
| d28 | "[Cyril quote withheld]" |
| d29 | "[Cyril quote withheld]" |

**Later rulings.** The 10-21 evening ruling adds four more points (ev 1232245 / 1232260; `STAGE_D_RULINGS:113-120`):
- "[Cyril quote withheld]"
- "no text only tables"
- "[Cyril quote withheld]"
- "[Cyril quote withheld]"

**None of these four is in omni_lab.** The `directive` table stops at id 693 (2026-10-21 17:10Z), and arch_requirement R1-R16 do not carry them. [C-05]

**Delegation.** In the same 17:10Z prompt Cyril wrote "[Cyril quote withheld]" (transcript line 6590). So the agent choices below were made under that delegation, inside his MoE, sparse and engram limits. [P-06]

| Component (as built in BR) | Origin | Conformance | Test result |
|---|---|---|---|
| **Trunk: 3 GDN : 1 gated GQA** (`kda_decay_kind="gdn"`) | **Agent.** ARCH L66: "Design 0's trunk: the Qwen3.5 small-model hybrid"; Judge A L77 [P-06]. It replaced August's KDA + gated-MLA/NoPE + SiTU [C-02, P-06] | d687 is open: ARCH §13 Q2 (L496) asks whether "Qwen's hybrid" counts as our own architecture. Cyril's answer on topology: **not found in sources** [P-06] | No A/B against August. REVIEW: "T2 (KDA decay), T3 (MLA) and T4 (finite SiTU β) have no receipts" [P-06] |
| **Block AttnRes, block 4, zero init** | Cyril asked for "full residual attention" in May. The specific block-4, zero-init layer is agent (ARCH L152-160) [P-06]. ARCH L287 maps his phrase onto it [C-07] | **Agent interpretation.** In d689 Cyril contrasts "[Cyril quote withheld]" with *sparse attention*, not with depth mixing [C-07] | **Never tested.** A1 (ARCH L453) has no omni_lab experiment row. ARCH L290: "small-dense gains are UNVERIFIED, so A1 decides". In the fold it is "zero by construction" (FOLD L122) [C-07] |
| **Raw boundary in, untyped** (shared `feature_proj` + positions + noise) | The encoder-free boundary is August/Codex [C-02]. Removing the type tables is agent [P-06] | R3/R4/R5 pass on the proxy, and the type-id relabel is bit-identical (ARCH R5 row) [S]. Whether a one-linear "bridge" is what d29 rejected: **open** (§6) | All six "in" behaviours pass on both untyped seeds in rounds 5 and 7 [C-08] |
| **A6b per-row RMS norm** | Agent (commit 48e8f2d; "A6b on in every profile", ARCH L661) [P-06] | Shared op, no table. Doc conflict: ARCH L531 still says "off in every profile" [S] | Audio-in 0.773 vs 0.747 for the typed control (ARCH L587, L655) [P-06] |
| **3,072-value raw unit** (32 px; 38.4 kHz) | **Agent.** ARCH L86: "Judge B, critique 8" [P-06]. August used 1,536 [C-02] | Raw; conforms to "no encoder decoder" | Never compared with 1,536 |
| **x0 flow head, untyped GLU MLP** (width → 2×width → 3,072) | x0 flow and one shared head are August [C-02]. GLU in place of linear and no type affine are agent (ARCH L209) [P-06] | Whether a learned 2-layer output MLP counts as a "decoder" (d28): **open** (§6) | Generation fails (below) [C-08] |
| **Positions: partial RoPE 64/256, interleaved mRoPE** | Agent, copied from the Qwen3.5 config (ARCH L184, L234) [P-06]. August was NoPE [C-01] | Serves 256k on paper (d686) | Video pan in 0.995-1.000 after mRoPE/P1 (ARCH L695) [P-06] |
| **Vocab 248,320 + 512 control, tied** | Agent (ARCH L160; follows Qwen3.5) [P-06] | n/a | In the fold, tied rows have median cos **0.027** with T's input rows (FOLD L372) [S] |
| **Text headers inside control markers** (kind-naming audio headers) | Agent (commit f37327b). v2 keeps the 512 control band; the headers sit inside `<|media|>`…`<|/hdr|>` (ARCH L160, L211-214) [P-06] | A data-level type label, not a weight | — |
| **Raw 3D: SDF + PBR channels** | Agent (ARCH L199) [P-06] | No learned encoder | See the results table |
| **Sizes: co2-2b and co2-4b** | Agent; Qwen3.5 2B/4B shapes (ARCH L184) [P-06] | The sizes fit d684 | No 2B or 4B weights exist |

**Proxy results** (`co2-proxy-media`, 9,847,105 params, 6,000 steps, random init, real COCO / LibriSpeech / MAESTRO-ACE / ModelNet10). [C-08]

**Overall verdict:** "FAIL in every arm of every round (4, 5, 6 and 7; 15 arms)" (ARCH L784). omni_lab eval runs 38060-38063, 38110-38113, 38156-38159 and 38238-38240 all have status `fail`.

| Modality | In | Out |
|---|---|---|
| Text | pass | pass |
| Image | pass | image class out does not reliably pass (see Appendix B for the per-seed numbers) |
| Image mirror edit | n/a | **never passed**, best 0.650 |
| Video | pan pass, 0.995-1.000 | **never passed**, best 0.625 |
| Speech | speaker pass on both untyped seeds in rounds 5 and 7 | **never passed**, best 0.375 (typed, round 4); every untyped arm ≤ 0.125 |
| Music | pass | not stable (see Appendix B) |
| 3D | pass | not stable (see Appendix B) |

**What the record measured about the failures** (ARCH "What the measurements say", L794 on). [C-08]
- **Image and video:** "the generated rows are independent per-row blobs" (L698).
- **Edit:** "the model never uses its source" (L795).
- **Speech:** raw-audio x0 MSE is 0.314 for speech and 0.351 for music against a signal variance of 0.25 (L730, L799).
  - In round 6 the x0 skip cut it to 0.146-0.184, but "the content still does not follow the request" (L753, L799).
  - So the MSE gap is a measurement, not a proven cause.
- **Typed control:** "Because the typed control fails the same three, the defect is not the untyped sharing" (L792).
- **The cause the doc names:** proxy scale and budget (L792). It marks that cause "Not established" (L808-811).

---

## 3. Architecture drift list

Each item below says where v2 departed from Cyril's words or from the August proof, and what followed.

1. **Changed many variables at once against the only working reference.**
   - What changed: trunk kinds (KDA → GDN decay, gated MLA → gated GQA, SiTU → SiLU), NoPE → RoPE/mRoPE, typed → untyped, linear head → GLU head, 1,536 → 3,072 unit, control tokens → text headers inside markers, synthetic AV → real data, joint-AV suffix → separate tasks. [C-02, P-06]
   - Consequence: when edit, video out and speech out failed, nothing isolated which change caused it. The planned A6 rerun of `h3_hybrid_av_flow.py` was never done. No commit on BR touches the file (`git log 11adb32..HEAD`), and omni_lab has no run for it (ARCH L458, L473). [P-09]
2. **The architecture was built Qwen-shaped while the question was still open.**
   - ARCH L66 takes "the Qwen3.5 small-model hybrid" as the base. Cyril's d687 asks for "our own custom architecture", and §13 Q2 asks him whether Qwen topology qualifies. No answer is on record. [P-06]
   - Consequence: the whole layout depends on an unanswered question.
3. **Weights were planned from Qwen, then reversed.**
   - ARCH L68 says "Initialised from the Qwen3.5 text weights". ARCH §9 L381-383 says to download Qwen3.5-2B and use the Qwen3.5-4B or OC01 children. [S]
   - Cyril overruled this ("[Cyril quote withheld]"), and FOLD L48 withdraws it. [S, P-06]
   - Consequence: **the stale text is still in ARCH §1 and §9.**
4. **His phrase was reinterpreted as a layer.**
   - ARCH L287 says Block AttnRes "is the mechanism behind Cyril's '[Cyril quote withheld]'". [C-07]
   - Cyril's "residual attention folding" (ev 243398, ev 252138, ev 273038; and on 10-22, ev 1241896) is the July folding technique in `NF/docs/RESIDUAL_ATTENTION_FOLDING_2026-07-11.md`. [C-07]
   - Consequence: the technique he asked for is absent from the v2 fold plan, and a layer he did not specify sits in the model untested.
5. **The 50k phone minimum was not carried into the spec.**
   - ARCH L348 still reads "Phone v1 decodes exact attention up to 32k". [S]
   - The ruling says "phone 50k context is a minimum requirement" (`STAGE_D_RULINGS:116`). [S, C-05]
   - Consequence: the spec sits below a stated minimum.
6. **Sparse attention is planned despite "hot garbage".**
   - ARCH L369 plans kernel K5 `BLOCK_SPARSE_FA` for far-context decode, gated by A3. [S]
   - d689 calls sparse attention "usually hot garbage". [C-05]
   - Consequence: the plan conflicts with his words unless he rules otherwise. It is runtime-only and not built.
7. **A text-only table is still listed after his "no".**
   - ARCH L461 keeps A9(a), a "text-keyed PLE (needs Cyril's ruling: a text-only path)". He ruled "no text only tables" (`STAGE_D_RULINGS:118`). [S, C-05]
   - Consequence: stale; it should not be run.
8. **Rulings are missing from the lab DB.**
   - The 10-21 evening rulings are in no omni_lab table. [C-05]
   - Consequence: the conformance views (`v_arch_conformance`, `v_directives_by_topic`) do not check 50k, fold-only init, no-plant or no-text-only-tables.
9. **Stale doc text conflicts with the code.**
   - ARCH L197 says `tubelet=1`; the code uses 2-frame tubelets. [S]
   - ARCH L531 says A6b is "off in every profile"; L661 says it is on. [S]
   - Consequence: readers of the spec get the wrong configuration.
10. **The trainer was built and smoke-run after the fold proof failed.**
    - ARCH L640-643 lists the v2 trainer T2 as a blocker. The task brief and Appendix B record T2 smoke and probe runs on 10-22.
    - Their timing relative to Cyril's realignment message: **not established** (Appendix B).

---

## 4. Fold compatibility

### 4.1 What "the 27B trunk" is in the plan

- **FOLD L22:** T = "run-1 full-KD student (trained trunk)", `Qwen3_5ForConditionalGeneration`, 64 layers, hidden 5120, FFN 16,640, vocab 248,320 untied, marked "**Default primary trunk source** (mixers, FFN channels, embed/head)". [S]
- **T is trained, not folded:** 851 tensors, "0 are `visual` and 0 are MTP"; step 611, 19,560,968 tokens (`OCL/ARCHITECTURE_REVIEW_2026-10-21.md:123-124`). [S]
- **This matches Cyril's 10-22 complaint** (ev 1241896: "[Cyril quote withheld]"). T carries no media.
- **The Qwen3.8 attribution is agent-written.** The ruling's "Effect" line names "the K3 fold, **Qwen3.8**, and the media parents" (`STAGE_D_RULINGS:123`). Cyril's verbatim lines above it name only "[Cyril quote withheld]" (L115). `MEM/omni-v2-direction-2026-10-21.md:41` repeats "Qwen3.8 trunk" as if it were his. [S]

### 4.2 Parents

- **Qwen3.8-27B** (from `MEM/neural-forge-matrix-campaign.md:1153-1170`, config read on 2026-09-01). [S]
  - Dense; hidden 5120; 64 layers, of which 48 are Gated-DeltaNet and 16 are gated full attention (24 q / 4 kv × 256).
  - FFN 17,408; vocab 248,320, untied; 262,144 context; 1 MTP layer; a SigLIP-class vision encoder.
- **Kimi K3 and the media parents** (H3, LTX 2.3, Qwen-Image 2.1, ACE-Step 1.5, Hunyuan3D 2.1, speech models): the extracts give geometry and on-disk presence, but this pass did not re-verify them. See Appendix B.
- **Qwen3.5-2B/4B weights:** "Not a source, by ruling" (FOLD L48). [S]

### 4.3 Surviving fold artifacts (verified subset)

**K3 dense fold, `k3-run-oct09-claude`.** Source: `MEM/neural-forge-matrix-campaign.md`, entry dated 2026-10-10. [S]
- All 92 MoE layers were folded to dense form; per-layer gates accepted 24 and refused 68.
- Teacher-forced numbers, end of stack:
  - held: top-1 0.692, top-5 set 0.596, KL 1.000;
  - review prompt rows: top-1 0.30-0.42.
- The memory notes it "generalizes to held evidence text but poorly to chat-style prompts".
- No decoded or free-running behaviour result was found.

**S3, OC02-F-Q38-27BU-C2K4352** (same memory file). [S]
- 14,522,158,320 parameters, net 0.5227.
- "Review sealed partially_capable".

**S5 (S3 plus the K3 union at weight 0.03).** [S]
- `OCL/composite/c10/design_sweep/VERDICT.md:17-20`: "K3 adds **no measurable held-out text benefit at the fold stage**".
- R-Q agreement: 110/210 against 108/210 for S3, which is within noise.

**OMNI1 and T.** [S]
- REVIEW (`C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/REVIEW.md:34`): the published A=5/15 and B=0/15 are "machine upper-bound annotations without attached per-case human reading".
- T is described in 4.1.

### 4.4 What the proven techniques can build, and what they cannot

**Proven on record (behaviour-graded, frozen 15-case review).** Sources: `MEM/neural-forge-matrix-campaign.md` lines cited. [S]

- **Single-layer excision (LFM2).**
  - Results: L13 13/15 (above the parent), L07 11/15, L09 10/15 (L329-331).
  - Pairs and triples: one pair 6/15; "triple = 0/15" (L317).
  - Width plus excision: "compose terribly" (L310-311).
- **Hidden-width folding (LFM2).**
  - W896 4/15 at 0.859x; W768 3/15; W640 and W512 0/15 (L363-364).
  - The cliff is a step near 896 (L285-286).
  - Width combined with a second axis is fatal (L370). "Every composed child reads 0-1/15 (the width family kills)" (L830).
- **Best compressed LFM2 child overall:** a vocabulary-coreset plus excision composite, not a width fold.
  - ARCH.V20X13G03 = 11/15 at 0.7466x (L498).
  - ARCH.B75 ≥ 11/15 at 0.742x by zero-solve blend (L623).
  - The floor below which nothing holds is about 0.4116x (L874).
- **Residual-attention folding** (`NF/docs/RESIDUAL_ATTENTION_FOLDING_2026-07-11.md`). [S, C-07]
  - The completed p34 control is a saved checkpoint at **1.0134x**, i.e. no compression (L314).
  - It lost 92 parent-passing generation cases, against 16 at p35 (L327-328).
  - The doc's next step is to fit the child's own `A_child @ V_child @ O_child` contribution (L367-369).
  - Three live value-probe attempts "failed closed" with no scores (L391-394).
  - No residual-attention fold at 4x or more exists.
- **Multi-token objectives.**
  - The 10-13 folding review added free-running consecutive 5/8/16/32 agreement as a *measurement* (commit `67b32b54`). It states "not a new folding result" and "Historical top5 rank is not five consecutive correct tokens" (`reviews/2026-10-13-folding/CLAUDE_RECOVERY_UPDATE_2026-10-13_1946.md:17`). [S]
  - No fold builder in the record uses a top-5 or top-10 objective. FOLD records held top-1 only, as a diagnostic (L160). [S]

**v2 components against the proven techniques:**

| v2 component | Parent pieces of the same kind | What proven techniques can do | What they cannot do (no receipt on record) |
|---|---|---|---|
| GDN mixers | Qwen3.8's 48 GDN layers | Exact head transplant is mechanics only (FOLD §12.1, per the extract; not re-checked) | K3 KDA → GDN; attention → GDN; 5120 → 2048 width; multi-layer merges. The proxy merge of 4:1 plus width gave R2 0/9 (FOLD L382-384) [S]. FOLD L339 calls 2-3 → 1 layers plus 5120 → 2048 "unprecedented in this campaign" [S] |
| Gated GQA | Qwen3.8's 16 full-attention layers | — | K3 MLA → GQA: no receipt (REVIEW: T3 has none) [P-06] |
| FFN (SiLU-GLU) | Qwen3.8 SwiGLU; K3 expert union | Per-site 4x MLP coreset (LFM2) | All-site 4x at 27B: S3 "partially_capable". K3 SiTU → SiLU: no receipt (T4) [P-06]. K3 lift at 0.03: no benefit [S] |
| Embedding and head, tied | Qwen 248,320 rows, untied | Vocabulary coreset on LFM2 (part of the 11/15 composite) | Untied → tied: median cos 0.027 (FOLD L372) [S] |
| Raw boundary in | **No parent has one** | **None** | Built as `zero_placeholder` in the proxy (FOLD L388) [S] |
| x0 raw head out | **No parent emits raw signal** | **None** | `zero_placeholder` (FOLD L388) [S]. Raw-out exists only from random-init training (August; v2 proxy) |
| Block AttnRes | K3 has the mechanism; Qwen has none | — | Not folded: "zero by construction" (FOLD L122) [C-07] |

**Plain reading of the table.**
- The only raw-media in/out ever shown working came from **training from random init**: August, and partly the v2 proxy.
- No fold on record produces a raw-media boundary or head.
- The folding methods the record proves are single-layer excision, per-site MLP coresets and vocabulary coresets, all on LFM2. None has been shown at v2's width ratios (5120 or 7168 → 2048) or with layer merging.

---

## 5. Runtime and resource constraints (verified subset)

**Phone (Sprite).**
- Qualified at 32k: "Larger contexts require their own device qualification" (`D:/SpriteBuild/claude-20261019/docs/SPRITE_PHONE_QUALIFICATION_20261018.md:15, 66-67`). [S]
- No phone measurement at 50k or above was found. Cyril's minimum is 50k. [S, C-05]
- **ARCH §7 phone speeds are projections**, fitted to Bonsai-4B Q1 at 39.5 tok/s and a 2B Q4 at 33 tok/s (ARCH L333). [S]
  - "Only the 2B at 1-2 bits reaches Cyril's 50 tok/s target, and only at short context" (L344).
  - At 256k, exact decode is "under 10 tok/s" (L346).
  - Prefilling 256k "takes about 3-11 minutes and hits the 95 °C trip" (L347).
- **On-device ops today** (ARCH L354-359, native `292a145`): MUL_MAT Q1/Q2/Q4/Q8, GATED_DELTA_NET, SSM_CONV, FLASH_ATTN_EXT, imrope, norms, GLU, softmax, top-k. [S]
  - FLASH_ATTN_EXT handles "head 128 and KV up to 32k only" (K2, L366).
  - MLA is "Not used" (L372).
  - So an MLA-based design would need new kernels, and every 32k+ number depends on K2.
- **Multi-session HTP GDN corruption** means one session or GDN on the CPU (ARCH L327, L370). [S]

**Training hardware.** ARCH L391-397. [S]
- The training lane is P40 fp32: no bf16, no Triton, and it needs a cu124/cu126 torch.
- A 2B with fp32 Adam needs about 31 GB against 24 GB per card.
- P2P between P40s is broken.
- "86M tok/day for the 2B" is UNVERIFIED (L398).

**Live server state, parent weight sizes and disk free** were reported by the constraints pass but not re-verified here. See Appendix B.

---

## 6. Open questions (the record is genuinely ambiguous)

1. **Does the Qwen3.5/3.8 hybrid topology count as "our own custom architecture" (d687)?** ARCH §13 Q2 asked this. Cyril answered the *weights* half ("[Cyril quote withheld]") but not the topology half. [P-06]
2. **Does "throw away the 27b trunk" (ev 1241896) also drop Qwen3.8-27B as a minor parent, or only the trained T/S5/OMNI1 line?** His 10-21 verbatim names only "[Cyril quote withheld]". Qwen3.8 appears only in the agent-written Effect line. [S]
3. **Are the August proof's per-type tables acceptable?** He calls the August version the proof that "we already did this", but it was typed (type_embed, type_affine, a typed suffix) [P-01]. His verbatim table rule is "no text only tables". The phrase "no type tables" in the agents' R3/R5 is an agent reading (extract; see Appendix B).
4. **Is v2's single shared linear `NativeContinuousMediaBridge` the "bridge" d29 rejects?** Is the x0 GLU head a "decoder" under d28? Not addressed by Cyril in the sources.
5. **What did "[Cyril quote withheld]" (d689/d690) mean as an architecture property?** Every layer attending to every earlier layer? Plain full attention instead of sparse? The record holds only the agent's AttnRes mapping. [C-07]
6. **What are the phone bars beyond the 50k minimum** (tok/s at which context, and the memory budget)? ARCH §13 Q1 asks this. No answer was found.

---

## Appendix A: refuted or corrected claims (do not reuse)

- **"The 2026-08-04 toy fold is the only August fold of a chimera-omni model."** Refuted. Four later commits ran ledgered exact-4x width folds of test-scale ChimeraOmniModel instances (`d7da31f3`, `92faf1c1`, `b9e37374`, `ecb16477`). [P-04]
- **"Still image, speech and music were never tested."** Wrong for the repo as a whole. June 2026 tiny synthetic proofs tested image, music and TTS-style output (`cd6c154`, `1c759e0`, `9f24e2b`). Only the August v4 proof skipped them. [P-03]
- **"The August model was free of type tables and edge components."** Wrong. It had a typed suffix, per-type tables and a typed head. [P-01]
- **"REVIEW line 291's 'Qwen topology' refers to v2."** Wrong. It refers to the September import switches (`4be3b0a`, `59a5e6d`). The v2 topology statement is ARCH §13 Q2 (L496). [P-06]
- **"v2 replaced control tokens."** Only half right. The 512 control band stays, and text headers sit inside its markers. [P-06]
- **"These agent choices are not proven at all."** Too broad. They pass on small proxies for every input modality. They were never A/B-tested against the August components. [P-06]
- **"Agent choices were not directed by Cyril."** Needs context. He delegated "all of the details" on 10-21 (transcript L6590). [P-06]
- **"Calling the residual-attention technique simply 'coupled QKVO folding'."** Undersells it. It also covers attention distributions on decode trajectories and the KV-cache state, with Q/K as an anchor and V/O ridge-refit. [C-07 correction]

## Appendix B: extract statements not re-verified in this pass (check before use)

- **"The 256-wide trunk cannot carry a 3,072-value audio row" (ARCH §14.6) is contradicted by August's 12:1 ratio.** This came from the August extract, and the C-09 verdict text was truncated. The arithmetic (1,536 / 128 = 12; 3,072 / 256 = 12) holds. That August's synthetic sines make it a fair comparison is not established.
- **The per-seed out-numbers:**
  - image out passed on one seed only (0.475 / 0.700);
  - music out passed only in round 4;
  - 3D out passed in rounds 4-5, collapsing in rounds 6-7.
- **The p35 "4-token rollouts 5/8 → 8/8 with 16 Adam updates" result.** It is not in `RESIDUAL_ATTENTION_FOLDING_2026-07-11.md` lines 296-395, and a grep for "8/8" and "Adam" in that doc found nothing.
- **K3 geometry:** 93 layers = 69 KDA + 24 gated MLA, 896 experts, `mla_use_nope`. Also the "deadliest class" quote and the "Qwen GDN vs KDA not representable exactly" quote.
- **S3 column B 1/15, OMNI1 S8 column B 0/15,** and the s7 unit cosines (ACE 0.84 / H3 0.95 / HY3D 0.41 / QVL32 0.36), including the dangling H3 symlink.
- **Media parents:** geometry and on-disk byte sizes for H3, LTX 2.3, Qwen-Image 2.1, ACE-Step, Hunyuan3D and the speech models, and the K3 shard count.
- **Live server state at 12:33Z:** GPU occupancy, 161 GiB RAM available with swap full, NAS 16 GB free, Lexar 438 GB free, `/` 338 GB free.
- **Measured Sprite speeds:** Bonsai 8B NPU 29.66-30.26 tok/s, the 27B GPU at 2 tok/s, and so on. Also the memory figures and the app's 8k-64k context options.
- **Three `v2trainer-smoke*` jobs on GPU1/2 on 10-22 (11:33-12:16Z)** and omni_lab runs 38296-38325. Their timing relative to ev 1241896 is not established.
- **The extract's statement that the verbatim phrase "type tables" is absent from Cyril's directive rows.**
- **The statement that the v2 fold plan fits the raw boundary by ridge to QVL32 ViT and speech-encoder activations (FOLD §4.5).**
