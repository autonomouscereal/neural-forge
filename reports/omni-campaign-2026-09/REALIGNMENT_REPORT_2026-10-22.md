# Realignment report, 2026-10-22

For Cyril. Written after your 2026-10-22 12:15Z prompt (agent_memory ev 1241896; transcript 0c189f21 line 7583): "[Cyril quote withheld]".

How this was made: read-only. No training, fold, experiment, deletion or service change was run. Every statement cites its source. Tags:
- **[V Cn]**: a second pass re-opened the source and confirmed it.
- **[P Cn]**: the second pass partly confirmed it. The correction is written into the text.
- No tag: taken from the read-only extraction passes, with the source cited, but not re-opened a second time.
- "Not found in sources" means it was looked for and is not there.

Path short forms:
- OCL = `neural-forge/reports/omni-campaign-2026-09`
- NF = `neural-forge`
- MEM = `C:/Users/cereal/.claude/projects/C--Users-cereal/memory`
- ev = an agent_memory event id
- D# = an omni_lab `directive.id`
- lab# = an omni_lab `experiment.id`

---

## 1. Bottom line

**Where we are**
- The architecture is not finished. The v2 all-modality proof FAILED: mirror edit, video out and speech out never passed in any arm (`V2_FOLD_INIT_PLAN_2026-10-21.md` §12.4 line 398, citing `ARCHITECTURE_V2_2026-10-21.md` §14.5-14.9).
- No v2 trunk exists. The 2B fold was not started (same line).
- The only v2 fold that ran was a proxy, folded from **T**. T is the run-1 full-KD student of S5, and S5 is the Qwen3.8-27BU C2 coreset trunk plus 12,288 K3 channels [P C2: the plan calls T "full-KD on the S5 fold", line 338].
- That proxy is dead:
  - all 27 continuations were an immediate `<|im_end|>`;
  - held NLL was 9.7-10.3 against 8.36 for a unigram;
  - K3 was not in the build;
  - sources: §12.3, omni_lab runs 38262/38272/38273 [V C2, V C3].

**How far we drifted**
- From about 10-11 to today, every trunk artifact has been the Qwen 27B line: S3 → S5 → OMNI1 → T → the v2 proxy.
- You said K3 is "99%" (D63, 08-04) [V C3].
- K3's weight in the fold design has been 0.03 (`V2_FOLD_INIT_PLAN` lines 74, 141, 287) [V C3].
- No Omni fold used:
  - your multi-token objective (ev 887577, 10-13) [V C4];
  - residual-attention folding (ev 243398 / 273038) [V C8].
- The v2 fold stacked four fold axes at once: width, depth merge, head selection and FFN coreset. The record's strongest finding is "one structural axis is survivable; two is fatal" (`NF/docs/PHASE1_FOLD_ONLY_FRONTIER_2026-08-12.md` §2.3, §4.4).

**What is stopped**
- The v2 workflow `wsbi268e5` stopped 10-22 12:16Z (transcript #7591).
- No v2 job is on the server.
- Also stopped earlier:
  - D1/D2 LoRA, stopped 10-17 20:19Z (`REALIGNMENT_2026-10-17.md` §2);
  - the Qwen self-generated data, dropped from every mix;
  - the 10-21 cleanup, stopped, with fold sources now protected (`CLEANUP_REPORT_2026-10-21.md`).

**Still running** (drift audit §5, transcript #7591-#7712)
- Strata on GPU0+6
- the media gate on GPU4
- the P40 lane runners

**Still on disk and still encoding the drift**
- `V2_FOLD_INIT_PLAN_2026-10-21.md`
- tools `OCL/composite/tools/c19-c23`
- branch `claude/arch-v2-20261021`

---

## 2. Your rules, verbatim and dated

### 2a. What counts as a successful fold (grading)

| Date | Words | Source |
|---|---|---|
| 07-09 | "[Cyril quote withheld]" | D41, ev 220444 |
| 07-16 | "[Cyril quote withheld]" | D48, ev 273660 |
| 07-17 | "[Cyril quote withheld]" | D50 |
| 07-21 | "[Cyril quote withheld]" | D56 |
| 08-10 | "[Cyril quote withheld]" | D72 |
| 08-11 | "[Cyril quote withheld]" | D74, ev 563942 |
| 08-11 | "[Cyril quote withheld]" / "[Cyril quote withheld]" | D75, D77 |
| 08-11 | "[Cyril quote withheld]" | D78 |
| 08-11 | "[Cyril quote withheld]" | D80 |
| 08-12 | "[Cyril quote withheld]" / "[Cyril quote withheld]" | D84, D86 |
| 10-06 | "[Cyril quote withheld]" | D114, ev 892366 |

### 2b. Top-k and multi-token matching

| Date | Words | Source |
|---|---|---|
| 07-16 | "[Cyril quote withheld]" | ev 273038 [V C8] |
| 08-07 | "[Cyril quote withheld]" | ev 534617 |
| 10-13 | "[Cyril quote withheld]" | D110, ev 887577 [V C4] |
| 10-22 | "[Cyril quote withheld]" | ev 1241896 [V C4] |

- Only ev 887577 and ev 1241896 use top-5 or top-10 wording about fold matching. That comes from a grep of all prompt events [V C4].
- Your 10-13 wording reads "top N+" as tokens "in a row", meaning consecutive. Codex's audit read it that way too ("five-or-more-consecutive-token"). Source: `reviews/2026-10-13-folding/OBJECTIVE_HISTORY_AND_RECOVERY_BACKLOG.md` lines 3 and 44 [P C5].
- The audit keeps two measures apart:
  - **top-k coverage**: the parent's next token is among the child's k best (line 13);
  - **LCP / exact h-token rollout**: how many tokens match in a row (line 16).
- No source defines "top 10+" literally [P C5].

### 2c. Residual-attention folding

| Date | Words | Source |
|---|---|---|
| 06-26 | "[Cyril quote withheld]" | ev 165352 |
| 07-11 | "[Cyril quote withheld]" | ev 243398 [V C8] |
| 07-12 | "[Cyril quote withheld]" | ev 252138 [V C8] |
| 07-16 | "[Cyril quote withheld]" | ev 273038 [V C8] |
| 10-22 | "[Cyril quote withheld]" | ev 1241896 [V C8] |

Separately, "full residual attention" is also an architecture requirement (section 2g).

### 2d. Layer and width folding

| Date | Words | Source |
|---|---|---|
| 05-28 | "[Cyril quote withheld]" | D18 |
| 08-10 | "[Cyril quote withheld]" | D73, ev 557028 |
| 10-22 | "[Cyril quote withheld]" | ev 1241896 |

### 2e. Fold from the parents, not the 27B trunk

| Date | Words | Source |
|---|---|---|
| 08-04 | "[Cyril quote withheld]" | D63, ev 506681 [P C3: Qwen as "[Cyril quote withheld]", not "only a smidge"] |
| 10-07 / 10-13 | "[Cyril quote withheld]" | D116, ev 893504 |
| 10-17 | "[Cyril quote withheld]" | D121, ev 1170637 |
| 10-21 20:27 | "[Cyril quote withheld]" | transcript line 6836 [V C3] |
| 10-21 20:30 | "[Cyril quote withheld]" | transcript line 6883 [V C3] |
| 10-22 | "[Cyril quote withheld]" | ev 1241896 [V C1] |

### 2f. K3: all experts, all layers

| Date | Words | Source |
|---|---|---|
| 09-01 | "[Cyril quote withheld]" / "[Cyril quote withheld]" | D99, D103 |
| 09-02 | "[Cyril quote withheld]" | D105, transcript line 440 [V C3] |
| 09-07 | "[Cyril quote withheld]" | D106 |

### 2g. Architecture and modalities

| Date | Words | Source |
|---|---|---|
| 05-22 | "[Cyril quote withheld]"; "[Cyril quote withheld]"; "omnimodal at every layer" | D4, D5, D9 |
| 05-28 | "[Cyril quote withheld]" | D17 |
| 06-02 / 08-28 | "[Cyril quote withheld]" / "[Cyril quote withheld]" | D23; MEM `chimera-omni-h3-media-goal.md` L484 |
| 09-01 | "[Cyril quote withheld]" | D101 |
| 10-14 | "[Cyril quote withheld]" | D118 |
| 10-21 | "[Cyril quote withheld]"; "[Cyril quote withheld]" | D131, D132 |
| 10-21 | K3: take inspiration from it, "not adopt it"; "[Cyril quote withheld]" | D130, D687 |
| 10-21 | "[Cyril quote withheld]"; "[Cyril quote withheld]" | D689, D690 |
| 10-21 | MoE "a maybe"; engrams only "[Cyril quote withheld]"; "[Cyril quote withheld]" | D688, D691, D693 |
| 10-21 20:27 | all modalities in and out; raw video with no encoder or decoder; "no text only tables" | ev 1232245 |

### 2h. On-device

| Date | Words | Source |
|---|---|---|
| 05-21 | "[Cyril quote withheld]" | D1 |
| 09-01 | Q4 must fit a 3090 at 256k context | D97 |
| 10-21 | 4B, starting with 2B, Sprite-compatible, quantized later; "256k context"; "[Cyril quote withheld]"; "[Cyril quote withheld]" | D684, D686, D685, ev 1232245 |

### 2i. Process

| Date | Words | Source |
|---|---|---|
| 08-06 | prove the folding ceiling before distillation | D68 |
| 10-14 | "[Cyril quote withheld]" | transcript line 5484 |
| 10-21 | one DB with test / configs / results, one test script, "[Cyril quote withheld]" | D128, D129 |
| 10-22 | "[Cyril quote withheld]" / "[Cyril quote withheld]" | ev 1241896 [V C1] |

---

## 3. What our testing proved about folding

Unless stated otherwise, the scores below come from the LFM2-230M matrix campaign. They are frozen 15-prompt reviews, and the parent scores 12/15. Sources: MEM `neural-forge-matrix-campaign.md` (cited "Mem:line") and `NF/docs/PHASE1_FOLD_ONLY_FRONTIER_2026-08-12.md` (cited "FRONTIER").

### 3a. Layer (depth) folding: works for one or two blocks

- Single-block excision splices in the parent's write. Removing L13 scored 13/15, above the parent's 12/15. Other blocks: L07 11, L09 10, L01/L11 9, L06 8, L08/L10 7, L03/L05 6, L04 2, L00/L02/L12 0 (Mem:327-342).
- The excision ladder went 1 block 13/15 → 2 blocks 7/15 → 3 blocks 3/15 → 4 blocks 0/15 (FRONTIER §2.3). Pair cost tracks block distance; "Depth stops at 2 excisions" (Mem:316, 327-339).
- Qwen3.5-4B single-layer excision held up: X03/X08/X15/X16 scored 15/15 (column B 12-13), X31 12/15 and X00 10/15 (`OCL/SCORECARD.md`; lab#23-28).
- Producer-first depth merging failed. LFM2 merged from 14 to 6 layers gave TF top-1 .009, free-run 0 and 0 tasks (`NF/docs/DYNAMIC_LFM2_RESULTS_2026-07-16.md`:213-240).
- Theorem T9: for ridge, progressive folding equals direct folding (Mem:23-24).

### 3b. Width folding: hidden width is the killer, MLP width works

- Hidden width scores: W896 4/15 at 0.859x, W768 3/15, W736 1/15, W704/640/512 0/15 (Mem:361-370, 413-414).
- Behaviour tracks retained read energy, with a floor of about 0.92. Write cosine of 0.9999 still dies (Mem:365-370, 384-396).
- The proven width method uses the top-k eigenvectors of the pooled reader moment, with an expected floor near 0.90 retained energy (FRONTIER §4.3).
- MLP intermediate width beats hidden width at a matched ratio. Config-standard CL1792 scored 7/15 at 0.856x against W896 4/15 (Mem:647-652).
- A per-site MLP channel coreset at true 4x holds 12/15 on three components: C05.T8 conv_out, C07.T5 attn_k and C13.T8 mlp_down. About three folded components are tolerated (Mem:59-66, 75-93; lab#19).
- Other true-4x survivors: C13.T5.S4.L05 11/15 and C11.T1.S4.L00 10/15. The "MLP wall was an artifact of token-match gating" (Mem:37-46).
- Folding MLP width at all 14 sites at once fails: ff640 (4x) scored 0/15 and ff1280 about 5/15 (Mem:258-264).
- Coreset beats low-rank on 40 of 42 SwiGLU writes. Low-rank wins on conv (Mem:268-269).
- Larger models:
  - Qwen3.5-4B C2 MLP coreset on all layers: 4x 5/15 (column B 2), 8x 1/15 (`SCORECARD.md`; lab#30-31).
  - 27B S3 C2K4352 at 4x on all 64 layers: 8/15 sealed, column B 1, against the parent's 15/15 and B 12 (lab#33).

### 3c. Attention, heads and vocabulary

- Attention Q/K: T1 ridge is strong; attn_k holds 12/15 at 4x. V/O "won't follow q/k down", and three stacked attention folds collapse the decoder (Mem:17-24, 59-66, 84-86).
- Head merges:
  - 16→12 averaged: 5/15;
  - joint relation solve KV6JQ: 8/15 at 0.979x;
  - 16→8: dead.
  - Source: Mem:318-338.
- Head dimension: HD16 0/15, HD48 1/15, because of the RoPE-plane constraint (Mem:286-289).
- Vocabulary is the cheapest axis:
  - V19 14/15 at 0.7946x;
  - robust deep frontier 7/15 at 0.6188x;
  - intelligence floor 0.4116x (Mem:518-531, 697-707, 1066-1101).
  - You called the roughly 0.48x 2/15 child "a dead model" (MEM `folding-behavior-first-grading.md` 127-134).

### 3d. What composes

- **One structural axis survives; two is fatal** (FRONTIER §2.3, §4.4).
- Width plus excision "compose terribly": W896X13 scored 1/15 (Mem:310-311). Excision plus MLP width ff1280 scored 0/15 (Mem:308-311).
- "WIDTH x MLP REGION CLOSED ... every composed child reads 0-1/15" (Mem:826-830).
- Measured interference: depth×vocab 4.44 cases, depth×mlp 1.39, mlp×vocab 1.17 (Mem:653).
- Whole-model exact-4x by dense contraction scored 0/15 in r12, in r13 arm A with every remedy, and in the 2x arm. The stage-1 cliff is bracketed at (1.0x, 1.559x] body (MEM `neural-forge-folding-project.md`:37-41; `CURRENT_FOLDING_GOAL.md`:49-53; lab#10-13).
- The C10 K3 union in the 27B trunk at full weight collapsed to 0/210. At w=0.03 it showed "no measurable held-out text benefit" (`OCL/composite/c10/design_sweep/VERDICT.md`).

### 3e. Residual-attention folding: what it actually did

- **The only positive multi-token result** is p35 QKVO, a one-site, rank-8 residual on a Qwen3-4B codebook child (L19/G4→L21/G6) [P C9]:
  - held-out top-1 went 56/64 → 64/64;
  - exact 4-token rollouts went 5/8 → 8/8;
  - calibration went 38/48 → 45/48;
  - it was **gradient-trained**: manual Adam, hard-action CE 1.0 plus 0.1 distribution KL, horizon 4, selected at step 16 of 32 updates;
  - only a factor pack exists, no checkpoint;
  - sources: `NF/docs/CURRENT_RESULT_AND_NEXT_EXECUTION_2026-07-11.md`:113-135; lab#2 "the only positive multi-token recipe found by the audit".
- **Scaled-up versions failed** [P C9]:
  - MLP residual: exact-4 went 2/8 → 1/8;
  - guided QKVO: 0/24 exact 8-token;
  - tournament: trained at horizon 16, evaluated at horizon 8, 0/24 → 0/24, though rank improved 3172 → 1378 ("small, real residual-attention progress", no trajectory gain);
  - sources: `OBJECTIVE_HISTORY…`:28-30; `RESIDUAL_ATTENTION_TOURNAMENT_AND_GUIDED_GUD_2026-07-15.md`:37-60.
- **Zero-training residual-attention compiler on LFM2** (producer-first, MLP-only, residual refinements on 10 layers) [P C9]:
  - TF top-1 .953, 8-token free-run .8125;
  - "partially coherent", but manual review 0/7 and lab#6 "not intelligence-retaining by review";
  - source: `DYNAMIC_LFM2_RESULTS_2026-07-16.md`:264-330.
- **Closed-form tries all failed:**
  - O-only ridge was not committed;
  - raw V/O regression lowered V RMSE but raised held-out risk from 0.84 to 7.15 ("clearest false-positive");
  - causal oracle: parent K+V made KL 44.97% worse, while the parent residual write helped by +25.42%;
  - source: `RESIDUAL_ATTENTION_FOLDING_2026-07-11.md`:278-376.
- **Not found in sources:** any zero-gradient residual-attention fold that passed a behaviour review.

### 3f. Objectives: top-1 and top-k against behaviour

- Top-k and token match have repeatedly coexisted with dead behaviour [V C7]:
  - Qwen head isolation: TF top-1 rose from 11.67% to 65.75% and top-1-in-top-5 from 30.55% to 85.67%, while behaviour stayed at 0 (`RESIDUAL_ATTENTION_FOLDING_2026-07-11.md`:10-14).
  - LFM2 35% vocab fold: TF top-1 .934 and top-16 overlap .868, yet 0/11 parent-correct tasks (`DYNAMIC_LFM2_RESULTS`:617-637).
  - Teacher-forced top-1 ran against free-run across four children: 1.86% → 33.10% TF while free-run went 0.44% → 0.195%, with 0/15 throughout (`STAGE1_CEILING_REPORT_2026-08-08.md`:22).
- Standing rule: "Never report teacher-forced top-1 without its calibration band (below ~65% it has only ever coexisted with incoherence; ~80%+ is necessary, never sufficient ...). Never in an acceptance path or headline." (`CURRENT_FOLDING_GOAL.md`:74-76) [V C7].
- Fits with a multi-token objective [P C6]:
  - horizon 4: the p35 positive;
  - horizon 16: NF-05 tournament, lab#5, failed.
- **No fold was ever fitted with a top-5 or top-10 objective**; top-5 appears only as a metric [P C6].
- **No positive result exists at 5 or more consecutive tokens** (`OBJECTIVE_HISTORY…`:44) [P C6].
- The one top-10 number on record is A9 k32 residual-OMP: top-10 605/648 (advisory). The same child was rebuilt at 7/15, and a frontier re-score gave 8-9/15 (`NF/reports/sparse-semantic-a9-k32-residual-omp-20260730/manual-review.md`:68-76).
- `folding_retention.py`, a free-run evaluator for 5, 8, 16 and 32 consecutive tokens, exists at commit 67b32b54 (`OBJECTIVE_HISTORY…`:94-96).
- X2 was designed and never run (`OCL/research/FOLDING_PROGRAM_v2_2026-10-14.md`:169-182; lab#43) [V C6]. It covers H=4/8, growing to 16, with endpoint residual plus top-64 KL.

### 3g. Chimera-scale folds (Sept-Oct)

- **K3 dense pass** (all 896 experts plus 2 shared per layer, union coreset, folded prefix, record mode):
  - 24 layers accepted, 68 refused;
  - end of stack, held split: top-1 .692, top-5 .596, KL 1.000, NLL 2.157 against the teacher's 1.261;
  - review rows: top-1 .30-.42;
  - **never decoded; no behaviour review** (Mem:1865-1874; lab#34).
- **S3 / S5 / OMNI1** (27B trunk plus five per-modality charted units):
  - text column B 0;
  - "The text intelligence is gone, and the cause is the trunk fold" (`CHIMERA_OMNI_STEP1_RESULTS.md` §6);
  - media outputs worse than the noisy input, except QIMG at +0.18 to +0.39 dB against the parent's +10.5 to +16.8 dB (STEP1 §4-5);
  - OMNI1 was built with `mtp_heads=0` (`composite/s7/omni1/manifest/config.json`:434) [V C6].
- **0.8B X3** fold plus recovery: fold alone 0/162; on-policy recovery 14/162 against the parent's 54 (lab#41).

---

## 4. Drift audit

Each departure is listed with what it contradicts, its consequence, and whether it is still active. Source for the list: the drift extraction, with each underlying source cited.

| # | Departure | Contradicts | Consequence | Still active? |
|---|---|---|---|---|
| D1 | The 27B Qwen trunk stayed the base: S3 → S5 → OMNI1 → T. In the v2 plan, T is the "**Default primary trunk source**" (line 22). S5 is the control arm (line 23), and S5 is also 27B lineage. Claude recommended T at 10-22 11:50 (#7569). | D63; 10-21 line 6836; 10-22 ev 1241896 | Every v2 fold so far is a fold of the Qwen 27B trunk [V C2]. | Plan, code and branch remain. Workflow stopped. |
| D2 | K3 at 0.03 in the fold; K3 rows at fraction 0.07 in run-1 (`STATE_2026-10-18_FULLKD_LAUNCH.md`:50-51); K3 absent from the proxy | D63 "99%"; D121 | "no measurable held-out text benefit" (VERDICT); "the opposite of '99% K3'" (`ARCHITECTURE_REVIEW`:213) [V C3] | Yes, in the v2 plan. |
| D3 | LoRA D-stages D1/D2 | D116; D121 "no lora" | "The base weights of Chimera Omni never moved" (`REALIGNMENT_2026-10-17` §2) | No. Stopped 10-17. |
| D4 | Qwen3.8-27B self-generated data: 1M-token self-sample, 3M tool-call target | D63; D121 | 4.15M of D2's tokens were Qwen self-generated | No. Dropped; files kept as evidence. |
| D5 | Per-modality routed "charted" units; OMNI1 built with type-routed writes | D101; D131; D132 | Media no better than the noisy input; modalities share no trained weights (`ARCHITECTURE_REVIEW` §1b) | Partly. v2 retires routing but still lifts unit FFN channels and heads as fold sources (§4.4 item 3). |
| D6 | Text-only student: run-1 had 851 tensors and 0 media tensors | D106; 10-22 "[Cyril quote withheld]" | T cannot carry any modality | Yes, as T in the v2 plan. |
| D7 | Qwen3.5-2B import and the Qwen3.5 layout: `ARCHITECTURE_V2` §1 "Initialised from the Qwen3.5 text weights"; the download started | D687; D130; line 6836 | Download stopped 10-21 20:27 (#6840). Whether the files remain is not found in sources. | Yes, as the layout: 24 layers, 3:1 GDN:GQA, 248,320 Qwen vocab, `qwen35` importer (`V2_FOLD_INIT_PLAN` §2). |
| D8 | v2 proxy used a new, unproven stack: global PCA chart + head selection + 4:1 period merge + coreset/ridge, at about 140x (26.1B → 191M) | D73 "[Cyril quote withheld]"; ev 243398/273038; ev 1241896 "[Cyril quote withheld]" | Dead (section 1). The plan calls the merge plus width "unprecedented in this campaign" (§11), ignoring LFM2 14→6 = 0 and W512 = 0/15. Retained energy was 0.799, below the ~0.92 floor. Residual attention / p35 is mentioned 0 times. | Stopped. Code and plan remain. |
| D9 | Top-1 / NLL used to choose and grade folds: C10 chose w on top-1/KL/NLL; S5 reported as "181/210 top-1"; v2 check R1 is next-token NLL (`c23_v2_fold_checks.py`:12, 67); `folding_retention.py` not used | ev 887577; ev 1241896; `CURRENT_FOLDING_GOAL.md`:74-76 | Fold choices were made on the metric the record says is anti-correlated with behaviour | Yes, in c23. |
| D10 | The 10-21 cleanup deleted fold artifacts: C17 charts (18.8 GB), trunk taps (32.2 GB), the OC01 4B ladder, K3 folds L00-04, and the H3 L25 unit source (6 dangling symlinks). Near miss: the only copy of K3 `fold/body` L61-L92. | "[Cyril quote withheld]"; line 6836 | Refold cost ~1 day (C17), ~4.4 h (ladder), ~1 GPU-h (H3) (`CLEANUP_REPORT_2026-10-21.md`:623, 639-660) | Stopped; fold sources now protected. |
| D11 | Plans built from partial prompt histories. The omni_lab directive table lacks ev 243398, 252138 and 273038 (residual attention), and ev 1241896 is not yet a row [V C4]. | D22; ev 534617 "rereview ... every prompt" | The v2 plan was written from that table and never mentions residual attention | Yes. |
| D12 | Proof-campaign rules (≥4x per component, `clears_4x` gate F1) applied to production assembly | not ruled by you for production (`ARCHITECTURE_REVIEW` §7 root cause 4) | Drove the S3 all-layer 4x fold (B 12 → 1) | Yes, in the v2 plan. |
| D13 | Your 10-13 multi-token objective was deferred to "arm A5, only after Codex agrees". The first pilot was a C2 4x fold + KD on Qwen3.5-0.8B: 0/162, paused (`research/DENSITY_AND_DATA_PROGRAM_2026-10-14.md`:99, 153, 273). | ev 887577; 10-14 "taking the lead" | No fold with your objective was ever run on Omni | Status after 10-15 not found in sources. |
| D14 | K3 dense fold judged teacher-forced only, never decoded | MEM `omni-k3-fold-all-experts-all-layers.md` ("judged ONLY on the whole model"); D41 | We do not know whether the K3 fold speaks | Yes. The v2 plan calls it the "Primary knowledge source" without a decode. |
| D15 | v2 attention schedule (3 GDN : 1 full attention + Block AttnRes) inherited from the Qwen3.5 layout | D689/D690 "[Cyril quote withheld]" | Approval not found in sources | Yes. |

**Contradictions inside Claude's own memory**
1. `folding-behavior-first-grading.md` says token match is "not an optimization target. Ever." Your 10-13 and 10-22 prompts ask to aim folding at multi-token / top-10+ matching. The memory over-generalised your 08-11 "[Cyril quote withheld]" (D74).
2. `omni-full-distill-k3-primary.md` lists Codex and Claude `tool_call` events as training text, while `omni-corpus-tokenizer-agnostic.md` says those are never used.
3. `omni-v2-direction-2026-10-21.md` lists the "Qwen3.8 trunk" as a fold source. That is not in your words.

---

## 5. What is still valid, and what goes

### Still valid (reuse)

**Parents (unfolded)**
- Kimi K3: the primary source (D63, D105).
- The media parents behind the S7 units: H3, Qwen-Image (QIMG), Hunyuan3D (HY3D), QVL32 and ACE (`composite/s7/units/*`).
- These are the distillation teachers per D116 and D121.

**K3 fold artifacts**
- `k3-run-oct09-claude` dense pass: all experts, all layers, folded prefix; 24 accepted / 68 refused in record mode (Mem:1865-1874).
- The only copy of K3 `fold/body` L61-L92, now protected (`CLEANUP_REPORT`:639-660).
- K3 folds L00-04 were deleted (D10).
- None of these has been decoded (D14).

**Proven fold techniques and their code (section 3)**
- single-block excision;
- the pooled reader-moment width basis (FRONTIER §4.3);
- the per-site MLP channel coreset at 4x;
- the T1 Q/K ridge;
- the vocab coreset;
- the p35 QKVO residual recipe (factor pack only).

**Measurement**
- `experiments/chimera_recovery/folding_retention.py` (commit 67b32b54): consecutive-token retention at 5/8/16/32.
- The frozen 15-prompt review, which you score (D80).
- omni_lab: one DB, `omni_lab.py run/compare` (MEM `omni-lab-db.md`). Its `directive` table needs the missing residual-attention rows (D11).

**Architecture v2 code and proof**
- The architecture and its proof harness exist in `ARCHITECTURE_V2_2026-10-21.md` and on branch `claude/arch-v2-20261021`.
- The proof result is a valid finding: mirror edit, video out and speech out never passed (§14.5-14.9). That is the work left to finish.

**Data**
- The canonical text/structured corpus (MEM `omni-corpus-tokenizer-agnostic.md`).
- Your own speech recordings, cleared for training (MEM `omni-speech-data-is-cyrils.md`).
- The 34 GiB training corpus on the NAS (MEM `ai-training-corpus-location.md`).

**Serving** (MEM `ai-server-serving-layout.md`)
- Strata on GPU0+6 at :18400
- the media gate on GPU4 at :28400

### Goes, per your 10-22 order (retire as fold or trunk source; nothing is deleted without your word)

**The 27B trunk line, all of it, as trunk or fold source:**
- S3 (`OC02-F-Q38-27BU-C2K4352`)
- S5 (`…K3U`)
- OMNI1
- T, the run-1 student
- the D1 LoRA child
- the v2 proxy built from T

S5 is 27B lineage too, so it cannot be the fallback [V C2 correction].

**Also retired:**
- The v2 plan sections that depend on that line: T/S5 as mixer source, `w_k3 = 0.03` (lines 22-23, 74, 141, 283-287, 338, 351).
- The per-modality charted units as a routing design (D5). Whether their folded FFN channels are reused is question 6.
- The Qwen3.5-2B import (D7). The layout question stays open (question 3).
- Qwen self-generated data. Already dropped.

---

## 6. Proposed realigned plan (for you to approve or change; nothing has been run)

Every step cites the record. Where the record shows a technique failed, the step says so instead of assuming it works.

### Step 0. Fix the record before building on it
- Add ev 243398, 252138, 273038 and 1241896 to the omni_lab `directive` table (gap D11; [V C4] confirms 1241896 is missing).
- Correct the three memory contradictions in section 4.
- Mark `V2_FOLD_INIT_PLAN_2026-10-21.md` superseded on trunk source and K3 weight.
- Source: D22, ev 534617, D128.

### Step 1. Finish the all-modalities architecture and its testing
- **Requirements:** D4, D5, D9, D14, D17, D23/D101 (no MoE / no towers; D688 leaves MoE "a maybe"), D97, D118, D131, D132, D684-D693, ev 1232245.
  - every tensor carries all modalities;
  - raw video in and out with no encoder or decoder;
  - 2B then 4B;
  - 256k context target, 50k minimum on the phone;
  - Sprite-compatible.
- **Open failures to fix:** mirror edit, video out and speech out (`ARCHITECTURE_V2` §14.5-14.9).
- **Rule:** the architecture proof passes before any fold into it (`V2_FOLD_INIT_PLAN` line 398 already set this condition).
- **Decisions needed from you first:** the layout and the attention schedule (questions 3 and 4).

### Step 2. Fold the v2 trunk directly from the parents
- **Sources:** K3 first (D63 "99%"), then the media parents (D99, D106). No 27B trunk (ev 1241896). No direct placement of any model (line 6883).
- **K3:**
  - fold all experts through all layers on the folded prefix, no MoE (D105; D73 "[Cyril quote withheld]");
  - reuse the oct09 dense pass and the L61-92 body where they fit the new shape;
  - refold where needed (line 6883: "[Cyril quote withheld]");
  - **decode it first**: the K3 fold has never produced text (D14).
- **Techniques, only from section 3, one at a time, proven before composing (D73):**
  - Layer folding: single-block excision. The record supports 1-2 blocks; 3+ is fatal (FRONTIER §2.3). Producer-first span merging failed (14→6 = 0).
  - Width folding: the pooled reader-moment basis, held at the ~0.92 retained-read-energy floor (Mem:365-370; FRONTIER §4.3). Prefer MLP-intermediate width over hidden width (Mem:647-652). Hidden widths of 640 and below were dead on LFM2.
  - Residual-attention folding: the only positive recipe is the p35 trained QKVO residual (4-token, one site) [P C9]. Every scaled version failed. The record supports it as a per-site repair after a fold, not as a whole-model method.
  - Composition: the record says a second structural axis is fatal (FRONTIER §2.3, §4.4) and has no successful multi-axis composite. Any stacking needs your call (question 2).
- **Grading:**
  - Diagnostics:
    - top-10+ token matching [V C4];
    - consecutive-token retention from `folding_retention.py` at 5/8/16/32 [P C5];
    - teacher-forced numbers reported only with their calibration band (`CURRENT_FOLDING_GOAL.md`:74-76) [V C7].
  - The grade itself: the frozen 15-prompt review, scored by you (D41, D78, D80). Numbers never fail a Chimera fold on their own (D114).
- **Fit objective:** whether top-10+ multi-token matching is also what the fold is fitted to (your 10-13 "aiming for") is question 1. The designed-but-unrun X2 (H=4/8→16 plus top-64 KL, `FOLDING_PROGRAM_v2`:169-182) is the only such design on record. The horizon-16 tournament failed [P C6].
- **Recording:** every arm goes into omni_lab through the one runner (D128, D129).

### Step 3. Distill from the non-folded parents
- Full-parameter distillation into the folded v2 trunk, from the original parents with K3 primary (D116, D121; MEM `omni-full-distill-k3-primary.md`).
- No LoRA. No training on folded-child outputs.
- No small-model data generation: no Qwen3.8-27B, and nothing under it (D121).
- No Claude or Codex outputs as training text (MEM `omni-corpus-tokenizer-agnostic.md`).
- Gate: the same behaviour review as step 2.

### Step 4. Finetune
- Tool calling, all modalities and speech. These are your binding asks (MEM `omni-codex-handoff-2026-10-14.md`).
- Data:
  - your own recordings (MEM `omni-speech-data-is-cyrils.md`);
  - versioned intake, never editing sealed corpora (MEM `omni-data-grows-over-time.md`).
- Quantize only after this step (D684, D84).

### Exactly what is reused, and what the record says each one is good for

| Item | Use | Record |
|---|---|---|
| K3 oct09 dense fold + L61-92 body | K3 fold source | Never decoded (D14) |
| Media parents (H3, QIMG, HY3D, QVL32, ACE) | Fold sources and distill teachers | S7 charted units failed in OMNI1 |
| Excision / reader-moment width / MLP coreset / T1 Q-K / vocab coreset code | Fold techniques | LFM2 + Qwen3.5-4B results, section 3 |
| p35 QKVO residual recipe | Per-site repair | 4-token positive; one site; no checkpoint |
| `folding_retention.py` | Multi-token diagnostic | Commit 67b32b54 |
| omni_lab + `omni_lab.py` | Single record and runner | MEM `omni-lab-db.md` |
| `ARCHITECTURE_V2` code + proof harness | Step 1 base | Proof failed on 3 tasks |
| Canonical corpus + your speech data | Steps 3-4 | MEM files cited above |

---

## 7. Questions where the record is genuinely ambiguous

1. **What "top 10+" means and where it applies.**
   - Your 10-13 prompt says "[Cyril quote withheld]" (ev 887577). That reads as 10+ consecutive tokens matching.
   - The alternative is the parent's token appearing in the child's top-10 at each position. Codex's audit treats these as different measures (`OBJECTIVE_HISTORY…`:13, 16) [P C5].
   - Separately, "aiming for" (10-13) makes it a fit objective, while D48, D74 and the memory rule say token match is diagnostic only.
   - Which is it: diagnostic, fit objective, or both?
2. **How much stacking you accept.**
   - A 2B/4B trunk from K3 is far past one axis.
   - The record shows a second structural axis is fatal (FRONTIER §2.3), and you set "4x minimum per component" for proving (D75).
   - Is step 2 judged as fold-init quality to be recovered by step-3 distillation (D116 "initialize ... from the folds then ... distillation"), or must the folded trunk pass the review on its own (D41, D106)?
3. **Layout.** The v2 layout is Qwen3.5's: 24 layers, 3:1 GDN:GQA, 248,320 vocab (`V2_FOLD_INIT_PLAN` §2). Does that count as "our own custom architecture" (D687), or must the layout change too?
4. **Attention schedule.** Is 3 GDN : 1 full attention + Block AttnRes acceptable against "[Cyril quote withheld]" (D689/D690)? Your approval was not found in sources.
5. **Qwen's remaining role.**
   - D63 allows Qwen "[Cyril quote withheld]".
   - Run-1 used Q38 as a teacher at 0.2-0.25 (`STATE_2026-10-18_FULLKD_LAUNCH.md`:51).
   - Does "throw it away" cover only the 27B trunk, or also the unfolded Qwen3.8-27B as a minor fold source or distill teacher?
6. **MoE and the S7 media folds.**
   - D688 calls MoE "a maybe" and D691 allows a "fixed" MoE for engrams. That conflicts with D23, D86, D101 and the 08-28 "[Cyril quote withheld]". Which holds?
   - Separately, may the S7 units' folded FFN channels be reused as fold sources in a shared-tensor model, or refolded from the media parents?

---

Not found in sources (listed so nobody fills them in by guessing):
- any decode of the K3 dense fold;
- any Omni fold that used residual-attention folding or a top-k / multi-token fit objective;
- any built excision ladder on the 27B or K3;
- the status of the density fold pilot after 10-15;
- whether the Qwen3.5-2B files were removed;
- any successful multi-axis composite fold.
