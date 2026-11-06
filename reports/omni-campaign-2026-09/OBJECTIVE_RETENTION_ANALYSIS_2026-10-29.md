# Objective vs retention analysis, 2026-10-29

Question from Cyril (D 821, ev 1246301, 2026-10-22): "[Cyril quote withheld]" of intelligence retention. This file measures how well each numeric metric on record tracked behaviour, answers that question with a direct paired measurement, and compares training/fit objectives at equal ratio.

Sources: omni_lab (read as `omni_lab_ro`; tables `fold_arm`, `fold_behaviour`, `fold_diagnostic`, schema 7, after ingest batch 274), the receipts named inline, and one new CPU measurement (section 3b) on 58 existing LFM2.5-230M matrix children. Nothing was folded, trained or deleted. Nothing was written to omni_lab. Scripts, per-child data and query text are in `objective-retention-20261029/` next to this file.

## 0. Answers

1. **Is top-n a better predictor than top-1? Not on the children we can measure, if top-1 is measured teacher-forced on the parent's own text.** On 58 LFM2 children with human frozen-15 reads, teacher-forced top-1 agreement correlates with the review count at Spearman rho = 0.890. Top-10 set overlap gives 0.883, and "parent's top-1 inside the child's top-5" gives 0.898. The paired differences against top-1 are -0.007 (95% CI -0.047 to +0.029) and +0.008 (-0.017 to +0.037). The two metrics rank the children almost identically: rho(top-1, top-10) across children is 0.981.
2. **Among living children (3-13/15, n = 31) top-n is slightly ahead, and the gap is not resolved.** rho is 0.804 for top-1, 0.851 for top-10 overlap (+0.047, CI -0.016 to +0.142) and 0.867 for top-1-in-top-10 (+0.063, CI -0.008 to +0.182). Only one of the 24 alive-band comparisons has a CI that clears zero (top-1-in-top-5 on the held prompt set, +0.048, CI +0.005 to +0.131), and with that many comparisons it should be read as a lead, not a result. So top-n helps by about 0.05 rho at most, and only when ranking children that are already alive.
3. **What was misleading before was free-running greedy match and partial-model top-1, not top-1 as such.** On the same 58 children, the recorded free-run greedy token match (`fold_diagnostic` name `free_run_match`, mostly `token_match_vs_control`) gives rho = 0.685, against 0.890 for teacher-forced top-1 (difference +0.205, CI +0.101 to +0.341). In the alive band the gap is 0.546 vs 0.804. The record's top-1 false positives (Qwen head isolation, TF top-1 65.75% with behaviour 0; LFM2 35% vocab, TF top-1 0.934 with 0/11) were measured on isolated sites or selected traces, and top-5 / top-16 failed on the same children too (85.67% and 0.868 with dead behaviour). On the five record children where both numbers exist on the same rows (LFM2 sparse-semantic A2/A3/A4/A9/A11), top-1 separates alive from dead perfectly (AUC 1.0) and top-5/top-10 do not (AUC 0.83), because A3 has the second-highest top-10 and is dead.
4. **Whole-model distribution metrics are good screens; local fit metrics are not.** Across the matrix campaign (489 children with human reads), lambada perplexity (rho 0.83), drift against the copy-exact control (0.81), terminal hidden cosine (0.78), mean LCP (0.78) and free-run match (0.77) all track the review count. Held-out local write cosine has rho 0.06 (AUC 0.48 alive vs dead): it carries no information. Most of the strength comes from separating broken children from working ones. Inside the alive band every metric drops to rho 0.4-0.7.
5. **No metric decides anything.** Even the best one (rho about 0.9) is near the ceiling set by the outcome itself: sealed counts move by ±2 between reviewers, and the robust re-read keeps only rho 0.60 of the sealed ordering (wave 33). These are screens and optimizer targets, which is what D 821 asks for. They are not gates.
6. **Objectives at equal ratio:** top-n has never been used as a fit target. No fold on record was fitted to top-k, and X2 was designed but never run. Among closed-form per-site fits on LFM2 at 4x, the output-relation objective (T8) was best or tied at 6 of 8 sites (+0.6 cases of 15 over the site mean); activation/write ridge (T1) was best at 1 of 8 (-0.4). Those gaps are inside the ±2 reviewer band. Trajectory objectives hurt (X05T 0/15 vs X05A 6/15). A full-vocab output-KL refinement cut held-out KL by 57% and left correctness unchanged (13/36 both). For recovery training on the same folded child, on-policy distillation beat top-64 off-policy KD (12 vs 2 of 162, parent 54), and KD followed by OPD did best (14). No clean gradient-vs-closed-form pair at equal ratio exists on record.

## 1. Data and rules

**Outcome.** A behaviour read is a `fold_behaviour` row scored by claude, codex, mixed or cyril with a numeric `score_value`/`score_denominator`, or with useful/partial/total (outcome = (useful+partial)/total). Rows scored `machine` are excluded from the outcome everywhere in sections 2-3. That removes the OC01/OC02 machine upper bounds, the 1000/1000/250 suites, the /6 probes and the 162-content oracle. Counts are the frozen-15 useful-or-partial counts in almost all cases (2,365 of 2,401 numeric human rows have denominator 15).
- 645 arms carry such a read. They collapse to **604 children** after normalising the arm key (strip `MX.`, strip `@…`/`#…` suffixes, uppercase, within program), so re-scores and re-ingests of one child count once. Each child's outcome is the median of its reads. By program: matrix-campaign 489, lfm2-dynamic 32, residual-attention 23, sparse-semantic 20, stage1 19, neural-forge 13, chimera-omni 8.
- Alive = median ≥ 5/15 and dead = median ≤ 2/15 (Cyril called a 2/15 child "a dead model", ev 575550). The 3-4/15 band is left out of the binary tests and kept in the rank tests. "Alive band" means ≥ 3/15.
- Sensitivity check: the ADDENDUM 6 verdict words (alive-parent-band / alive-degraded / impaired vs dead) as the outcome. 410 children carry one, 330 of them dead, and the dead ones include 13/15 and 11/15 children marked dead under the degeneracy clause. Results agree in direction (section 2c).

**Metrics.** `fold_diagnostic` rows are mapped to families by `name` plus note text (`analyze.py:family`). Each child gets the median of its values for each family:

| Family | DB rows it comes from | Direction |
|---|---|---|
| free-run top-1 match | `free_run_match` (token_match_vs_control: greedy, 8 probe prompts × 64 tokens vs the copy-exact control) | higher |
| mean LCP | `lcp` (mean longest common prefix, h = 64) | higher |
| TF top-1 | `tf_top1`, `top1`, `top1_agreement_vs_*` | higher |
| top-k | `topk_overlap`, `top5`, `top10` | higher |
| KL / NLL / MSE | `kl`, `nll`, `mse` | lower |
| lambada / wikitext PPL | `ppl` with lambada / wiki in the note (lm-eval limit 200) | lower |
| terminal relL2 / cos vs parent | `relL2`/`cos` "terminal_…_vs_parent" (child index) | lower / higher |
| drift vs control | `relL2` "vs_copy_exact_control_delta", "global drift" | lower |
| held-out local relL2 | `relL2` post_operator_heldout / held split | lower |
| write held-out cos, worst-site cos | `cos` write_heldout / worst_site | higher |
| retained energy | `retained_energy` | higher |
| benchmark accuracy | `other` rows naming hellaswag/arc/piqa/winogrande acc | higher |

**Statistics.** Spearman rho with average ranks, a 2,000-resample bootstrap 95% CI and a permutation p; AUC (Mann-Whitney) alive vs dead; a leave-one-out single-threshold classifier (balanced accuracy); paired rho differences with joint bootstrap over children. Pure Python (`analyze_stats.py`), because C:/Python314 has no scipy.

**Coverage, which limits everything below.** Arms that carry both a metric and a human read: free-run 514, LCP 497, relL2 444, cos 438, PPL 208, retained energy 34, TF top-1 9, top-5 2, top-10 3, **KL 0**, NLL 6, top-k overlap 0. The DB cannot answer top-n vs top-1 or KL vs behaviour. Hence the probe in section 3b.

## 2. How well each recorded metric tracked behaviour

### 2a. All children with a human read (record only)

| Metric | n | rho | 95% CI | alive / dead | AUC | LOO bal. acc. |
|---|--:|--:|---|--:|--:|--:|
| lambada PPL | 192 | 0.830 | 0.778-0.868 | 77 / 81 | 0.966 | 0.923 |
| drift vs copy-exact control | 415 | 0.811 | 0.775-0.841 | 206 / 152 | 0.958 | 0.910 |
| terminal cos vs parent | 291 | 0.784 | 0.734-0.822 | 100 / 144 | 0.951 | 0.871 |
| mean LCP (h 64) | 416 | 0.783 | 0.748-0.814 | 207 / 152 | 0.961 | 0.906 |
| wikitext word PPL | 192 | 0.764 | 0.699-0.814 | 77 / 81 | 0.951 | 0.911 |
| free-run top-1 match | 429 | 0.736 | 0.678-0.782 | 211 / 161 | 0.940 | 0.887 |
| terminal relL2 vs parent | 293 | 0.724 | 0.667-0.773 | 100 / 146 | 0.916 | 0.870 |
| held-out local relL2 | 205 | 0.715 | 0.637-0.776 | 115 / 67 | 0.904 | 0.856 |
| TF top-1 (non-matrix) | 12 | 0.594 | 0.222-0.912 | 3 / 9 | 0.889 | 0.944 |
| retained energy | 34 | 0.542 | 0.273-0.751 | 0 / 30 | — | — |
| benchmark accuracy (lm-eval, limit 200) | 161 | 0.271 | 0.129-0.399 | 55 / 75 | 0.672 | 0.664 |
| worst-site cos | 80 | 0.135 | -0.086-0.337 | 15 / 50 | 0.527 | 0.637 |
| write held-out cos (local) | 195 | 0.089 | -0.076-0.247 | 114 / 59 | 0.503 | 0.591 |

Every row with n ≥ 192 is matrix-campaign (LFM2.5-230M). Outside it, the only metric with n ≥ 10 is TF top-1 on 12 children, and three of those are A9/A9.K32/A11, two of which are the same A9 build. The "alive" side therefore has two distinct children. One dead child, the LFM2 zero-training residual-attention 10% child, has TF top-1 0.953 and free-run 0.81.

### 2b. Where the strength comes from (matrix campaign)

| Subset | n (free-run) | free-run | LCP | drift | lambada PPL | terminal cos | held-out local relL2 | write cos |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| all matrix | 420 | 0.770 | 0.783 | 0.811 | 0.830 | 0.784 | 0.687 | 0.055 |
| alive band ≥ 3/15 | 268 | 0.575 | 0.582 | 0.620 | 0.680 | 0.550 | 0.529 | 0.104 |
| alive ≥ 5/15 | 211 | 0.427 | 0.425 | 0.502 | 0.545 | 0.419 | 0.452 | 0.223 |
| ARCH children (real structural folds) | 291 | 0.660 | 0.700 | 0.727 | 0.842 | 0.784 | 0.131 | 0.131 |
| cells + COMP (parent-shaped, about 0.99x) | 124 | 0.690 | 0.720 | 0.824 | 0.802 (n 22) | — | 0.424 | 0.439 |

Paired, on the same children:
- LCP vs free-run: +0.016 (CI -0.008 to +0.043); alive band +0.015 (-0.027 to +0.054). Consecutive-token agreement adds nothing over single-token agreement when both are measured free-running.
- lambada PPL vs free-run: +0.112 (+0.054 to +0.170); alive band +0.160 (+0.032 to +0.288).
- terminal cos vs free-run: +0.124 (+0.066 to +0.187).
- drift vs free-run: +0.045 (+0.012 to +0.080).
- write held-out cos vs free-run: -0.80. A local fit number says nothing about the whole child, consistent with `ST.five-4` / `S1C.the-law`.

Held-out local relL2 tracks behaviour on parent-shaped single-component children (0.42) and not on real structural folds (0.13). That is the composition law again: local fit is informative only while one component is installed.

### 2c. Sensitivity: verdict words instead of counts

With the ADDENDUM 6 verdict as the outcome (80 alive, 330 dead), the AUCs are: free-run 0.937, LCP 0.936, drift 0.975, lambada PPL 0.977 (12 alive), terminal cos 0.967 (9 alive), held-out local relL2 0.884, write cos 0.477. The x/15 count itself predicts the verdict at AUC 0.90. Verdict-dead children include 4 at 13/15 and 6 at 11/15 (degeneracy clause). Cyril's later guidance ("[Cyril quote withheld]", ev 605159) favours the count, so the count stays primary.

## 3. Top-n vs top-1

### 3a. What the record holds

| Children (same rows for both metrics) | top-1 | top-5 | top-10/16 | behaviour | source |
|---|---|---|---|---|---|
| LFM2 sparse-semantic A2 (k16) | 533/648 | 579 | 584 | rejected (empty, repetition) | `reports/sparse-semantic-a2-k16-rank-retro-20260730/cached-hidden-diagnostic.json` |
| A3 (k16 norm γ0.5) | 513 | 601 | 611 | rejected, worse overall | `…a3-k16-norm-gamma05-cap3…/cached-hidden-diagnostic.json` |
| A4 (k16 residual OMP) | 544 | 582 | 588 | 0/16 completed; 3 empty, 6 looped | `…a4-k16-residual-omp…` |
| A9 (k32 residual OMP) | 551 | 589 | 605 | 3u/7p/6m of 16; V5 8/15 | `…a9-k32-residual-omp…` |
| A11 (k32 split C_out) | 585 | 632 | 639 | 3u/9p/4m of 16; V5 9/15 | `…a11-k32-split-cout…` |
| Qwen3-4B head isolation | 65.75% TF | 85.67% (teacher top-1 in child top-5) | — | 0/1000 behaviour | `NF/docs/RESIDUAL_ATTENTION_FOLDING_2026-07-11.md:11-14` |
| LFM2 35% vocab (rank 128 + 1,135 rows) | 0.934 TF | — | 0.868 top-16 | 0/11 parent-correct tasks | `NF/docs/DYNAMIC_LFM2_RESULTS_2026-07-16.md:617-637` |

On the sparse-semantic five, AUC alive vs dead is 1.0 for top-1 and 0.83 for both top-5 and top-10. A3 is the only child that changes order: it has the lowest top-1 and the second-highest top-10, and it is dead. In the two single-point cases, top-1 and top-k failed together. The record therefore contains **no evidence that top-n beat top-1**. It contains n = 5 against, measured at one token-interface site on cached hidden rows. The K3 dense pass (top-1 0.692, top-5 set 0.596) and the K3 harness review proxies carry both numbers but have never been decoded or read.

### 3b. New measurement: 58 existing LFM2 children, teacher-forced on parent text (CPU)

**Protocol.** `topk_probe.py` ran on the AI server CPU only (`CUDA_VISIBLE_DEVICES=` empty; the script refuses to run if a GPU is visible), at nice 19 with 12-16 threads and MemAvailable observed at 160-178 GiB at each check. It used no P40 or GPU and touched no run-0 resources. Interpreter: `/home/cereal/model-surgery-lab/venv/bin/python` (torch 2.11.0, transformers 5.12.1). Parent: `/home/cereal/model-surgery-lab/models/LFM2.5-230M`. Children: every shared-vocabulary child under `/mnt/ai_data/neural-forge-matrix/wave*/children/` that has a human frozen-15 read: 64 checkpoints, 61 tags. Three tags exist as two builds (CL896, CL896X13, CL1792X13); because their reads cannot be assigned to a build, they are excluded, leaving **58 children**. The 77 vocab-coreset (V*) children were not probed: their token ids differ from the parent's, so token-level agreement is undefined for them (Cyril, D 1010: "[Cyril quote withheld]").
- Text set A: the 16 frozen-review prompts with the parent's sealed verbatim responses from `wave43/cells/ARCH.HD48/frozen-review-dossier.json`, first 384 response tokens (2,745 scored positions).
- Text set B: the 30 extension-bank-v2 prompts (`NF/research/lfm2_extension_bank_prompts_v2.json`). These are not the review prompts. The parent's own greedy continuation was generated here, up to 256 tokens (3,760 positions). Sequences are frozen in `objective-retention-20261029/probe_seqs.json`.
- At each response position, over the full 65,536 vocab, for child q vs parent p: top-1 agreement; top-k set overlap |topk(q) ∩ topk(p)|/k for k = 5, 10, 20, 64; parent argmax inside child top-5/10/20; KL(p‖q); child NLL of the text token; child probability mass on the parent's top-10 set.
- Sanity check: the parent probed as a child gives top-1 1.000, KL 0.000 and NLL equal to its own. 65 rows, 0 errors (`probe_all.jsonl`).

**Results (text sets A+B; 19 alive, 27 dead, 31 in the alive band):**

| Metric | mean | rho (all 58) | 95% CI | AUC | rho alive band | Δrho vs top-1 (all) | Δrho vs top-1 (alive band) |
|---|--:|--:|---|--:|--:|---|---|
| **top-1 agreement** | 0.523 | **0.890** | 0.814-0.932 | 0.990 | 0.804 | — | — |
| top-5 set overlap | 0.434 | 0.884 | 0.795-0.930 | 0.990 | 0.847 | -0.006 (-0.041, +0.027) | +0.043 (-0.018, +0.137) |
| top-10 set overlap | 0.427 | 0.883 | 0.793-0.931 | 0.992 | 0.851 | -0.007 (-0.047, +0.029) | +0.047 (-0.016, +0.142) |
| top-20 set overlap | 0.428 | 0.880 | 0.786-0.930 | 0.990 | 0.857 | -0.010 | +0.053 |
| top-64 set overlap | 0.437 | 0.877 | 0.781-0.929 | 0.990 | 0.847 | -0.013 | +0.043 |
| parent top-1 in child top-5 | 0.744 | 0.898 | 0.817-0.938 | 1.000 | 0.861 | +0.008 (-0.017, +0.037) | +0.057 (-0.009, +0.168) |
| parent top-1 in child top-10 | 0.806 | 0.895 | 0.810-0.937 | 1.000 | 0.867 | +0.005 (-0.026, +0.039) | +0.063 (-0.008, +0.182) |
| KL(parent ‖ child), full vocab | 1.820 | 0.876 | 0.786-0.922 | 0.990 | 0.805 | -0.014 (-0.053, +0.017) | +0.001 (-0.106, +0.106) |
| NLL of parent text | 2.507 | 0.886 | 0.802-0.929 | 0.996 | 0.836 | -0.005 (-0.032, +0.020) | +0.032 (-0.022, +0.113) |
| child mass on parent top-10 | 0.584 | 0.824 | 0.713-0.889 | 0.971 | 0.715 | -0.067 (-0.123, -0.022) | -0.089 (-0.281, +0.066) |
| *DB: free-run greedy match* | — | 0.685 | — | — | 0.546 | -0.205 (-0.341, -0.101) | -0.258 (-0.548, +0.006) |
| *DB: mean LCP* | — | 0.700 | — | — | — | — | — |
| *DB: drift vs control* | — | 0.723 | — | — | — | — | — |
| *DB: terminal cos* | — | 0.782 | — | — | — | — | — |
| *DB: lambada PPL (n 38)* | — | 0.815 | — | — | — | — | — |

Set A alone and set B alone give the same picture (top-1 0.888 / 0.891; top-10 overlap 0.873 / 0.886). The result does not depend on whether the scored text came from the reviewed prompts.

**Calibration on this population** (`probe_results.json`, per-child table): every child with teacher-forced top-1 ≥ 0.70 scored ≥ 6/15 (X13A 0.797 → 13, X07 0.792 → 11, X09 0.759 → 10, X11 0.745 → 9, X05A 0.708 → 6, X06 0.704 → 8). All 23 children below 0.50 scored ≤ 3/15, and 14 of them scored 0. Between 0.55 and 0.70 the read ranges from 1 to 8 (HD48 0.561 → 1, X09KV6JQC 0.555 → 5, CL1792ME 0.599 → 2, X10 0.599 → 7). Inside each fold family the order holds: width W896 → W768 → W736 → W720 → W704 → W512 goes top-1 0.577 → 0.397 → 0.342 → 0.313 → 0.294 → 0.075 with reads 4 → 3 → 1 → 0 → 0 → 0. MLP width CL1792 → CL640 goes 0.660 → 0.466 with reads 7 → 0.

**By how much, in Cyril's terms.** Over all children, top-n is not better: Δrho is -0.007 to +0.008 with CIs about ±0.04. Over living children only, top-n (set overlap or top-1-in-top-n) is ahead by +0.04 to +0.06 rho, which moves rho² from 0.65 to 0.71-0.75 (+7 to +11 points). That is unresolved at n = 31. Cyril's observation holds against what was actually measured before, free-running greedy match: top-n beats free-run by about +0.2 rho on all children and +0.3 in the alive band. Teacher-forced top-1 beats free-run by the same margin. The win comes from measuring teacher-forced over the parent's distribution, not from moving from 1 to n.

**Practical reading for fold objectives.** Teacher-forced top-1, top-k overlap, top-1-in-top-k, KL and NLL are near-interchangeable as screens on LFM2 (pairwise rank correlation about 0.98). Top-k containment is the best in the alive band, and as an optimizer target it carries more gradient signal than top-1. Child mass on the parent's top-10 is clearly worse than the others. None of them should decide a fold.

### 3c. Limits of 3b

- One parent family (LFM2.5-230M), 58 children, almost all structural folds without training. Not tested: Qwen/K3 scale, trained children, vocab-changed children.
- The outcome is a frozen-15 count read by Claude, with ±2 reviewer sensitivity. Its own restatement reliability is modest (section 4). With rho about 0.9, the metrics sit near the ceiling the outcome allows, and that ceiling, not the metrics, may be what stops them being told apart.
- Teacher-forced agreement on the parent's text measures distribution matching. It cannot see EOS/loop failures that appear only in free generation. The reads already include those failures, and the correlation survives them here.

## 4. Confounds

1. **Dead-vs-alive dominates.** In all-children correlations, the metric mostly separates broken children (empty, looping, 0/15) from working ones. Inside the alive band every recorded metric falls to rho 0.42-0.68 (2b), and teacher-forced top-1 falls from 0.89 to 0.80 (3b).
2. **Parent-shaped children.** Cells and COMP children are about 0.99x stored (`MX14.F1`). Local fit metrics look informative there (write cos 0.44) and not on real folds (0.13). Mixing the two populations inflates local metrics.
3. **Machine upper bounds** (OC01 15/15, OC02 column A/B, the 1000/1000/250 suites, the /6 probes, the 162-content oracle) are excluded from the outcome. They appear only in section 5 and are labelled machine there.
4. **Review restatement collapse.** The wave 33 robust re-read keeps only part of the sealed ordering: Spearman(sealed, robust) = 0.60 over 8 children (`MX/wave33/human-review/CAMPAIGN_SCORECARD.md:18-27`); the parent goes 12 → 6 and CL1792M 8 → 0. Every rho here is against sealed counts, which are upper bounds.
5. **Reviewer not blind to metrics.** The frozen-review dossier carries `token_match_vs_control_diagnostic` beside the text, and before ADDENDUM 6 cells under 0.90 token match were not reviewed at all (`MATRIX_CAMPAIGN_DESIGN_2026-08-10.md:325, 416, 688-692`). That is range restriction plus possible anchoring on exactly the free-run metric. It cannot inflate the probe metrics in 3b, which were computed after the fact.
6. **Pseudo-replication.** Children are deduplicated by normalised key, but some different keys are still one build (A9 and A9.K32-RESIDUAL-OMP). The non-matrix TF top-1 row (n 12) is mostly that.
7. **Same quantity under two names.** For ARCH children the copy-exact control is the parent, so "drift vs control" and "terminal relL2 vs parent" are the same number (identical rho 0.727 in 2b).
8. **Program mix.** 81% of the read children are LFM2.5-230M matrix children. Every metric finding is LFM2 evidence until repeated at 2B/4B.

## 5. Which objectives produced better retention at equal ratio

Only within-study comparisons (same parent, same component, same ratio) are used. Cross-study comparisons confound model, site and ratio.

| Comparison (equal ratio) | Objectives | Behaviour | Reader | Source |
|---|---|---|---|---|
| LFM2 single component at S4 (4x), 8 sites with ≥ 3 techniques | T8 output-relation (Gram) ridge vs T5 sparse OMP write fit vs T1/T2 activation/write ridge vs T3 balanced truncation vs T4 weight SVD | T8 best or tied at 6/8 sites, mean +0.61 cases vs site mean; T4 +0.89 (3 sites, conv/attn); T3 -0.11 (3); T5 -0.14 (8, best at 3); T1 = T2 -0.39 (best at 1). T1 and T2 score identically at all 8 sites on separate receipts (same solve at unchanged width, presumably). All gaps are within ±2 cases. | Claude, frozen-15 | omni_lab `C??.T?.S4.L??` arms; `results2.json` `cells` |
| LFM2 excision of block 5 | local re-solve (X05A) vs trajectory objective (X05T) | 6/15 vs 0/15 | Claude | `MX15.F3`, `MX/wave15/NOTES-MX-S.md:344-349` |
| LFM2 hidden width 768 | reader-PCA basis vs evidence-PCA vs MM768 metric | 3/15 vs 1/15 vs 0/15 | Claude | `P1.read-currency`; `PHASE1:328-329` |
| LFM2 heads 16 → 12 | parameter averaging (KV6) vs joint relation solve (KV6JQ) | 5/15 vs 8/15 | Claude | `ARCH.KV6`, `ARCH.KV6JQ` |
| LFM2 MLP 1792 | stream-selected (CL1792) vs reader-metric (M) vs ME / MV / S variants; stale vs conditioned build | 7 / 8 / 2 / 7 / 4; stale 4 vs conditioned 7; robust re-read CL1792M 0 | Claude | `2_read3_fold_record.md:49`; wave 30-33 |
| SmolVLM/SmolLM2 retention studies 1-2 | balanced vs uniform select × fit | identical: 6/48 greedy (text), 11/32 (vision) despite relL2 0.060 vs 0.064 | machine | `retention-text-smollm2-*`, `retention-mm-smolvlm-*` |
| Retention study 3 | response-aware vs uniform shared pool | 5/24 vs 3/24 (relL2 0.094 vs 0.135) | machine | `retention-response-state-v2-*` |
| Retention study 4 | full-vocab output-KL refinement vs native prior ridge (same init) | held-out KL 0.0149 → 0.0064 (-57%), generated 13/36 both | machine | `retention-fourth-*` |
| Qwen3.5-0.8B C2 4x child, recovery (LoRA, not D 121-compliant) | none / top-64 KD / on-policy JSD (OPD) / KD 1M then OPD | 0 / best 2 / best 12 / best 14 of 162 content; parent 54 | machine (Codex oracle) | `x3-08b.*`; `OCL/research/DENSITY_AND_DATA_PROGRAM_2026-10-14.md:273` |
| Qwen3-4B → SmolLM2 blank target, 29.90x, recovery (same child) | dual-space ALM chunk-likelihood KD 240 vs dual-space KD 240 vs on-policy (two settings) | 4/6 vs 3/6 vs 1/6 and 0/6. Separately, on a 28.39x blank-Qwen host (a different child, so not equal ratio): top-k 400/600 last-4 KD 0/6, untied KD 600 0/6 | machine (/6 probe) | `docs-june-july-early:q4b-smol29x-*`, `q4b-blankqwen-*` |
| Qwen3-4B head isolation / LFM2 reclosure | teacher-forced top-1 as the target | TF top-1 11.67% → 65.75% and 1.86% → 33.10%; behaviour 0 | machine + review | `RESIDUAL_ATTENTION_FOLDING_2026-07-11.md:11-14`; `STAGE1_CEILING_REPORT:22` |
| Qwen3-4B p35 L19G4, one site | horizon-4 multi-token gradient fit (Adam) vs action-only selection | 5/8 → 8/8 exact 4-token vs 2/8 → 1/8 | machine | `p35-qkvo-L19G4`; `ra-law-action-only-selection-false-positive` |

What this supports:
- **Closed-form objective choice matters by about one review case at a site.** Output-relation (T8) and spectrum/basis-aware fits beat plain activation/write ridge. That is small and inside reviewer noise.
- **Distribution-level output objectives (full-vocab KL) improved the objective, not the behaviour,** in the one controlled test (study 4). Lower relL2 across selection variants (studies 1-3) did not move behaviour either, except response-aware pooling (+2 of 24, machine).
- **For recovery, on-policy beats off-policy, and off-policy-then-on-policy beats both** (X3). Top-64 KD alone recovered almost nothing (2/162) at budgets up to 2M tokens. That run used LoRA, so it does not test full-parameter KD. Top-k KD on the 28.39x blank target was 0/6.
- **Trajectory/whole-rollout fit objectives in closed form hurt** (X05T). A multi-token objective helped only when gradient-trained at one site (p35, horizon 4).
- **Not measurable from the record:** top-n as a fit target (never run, `gap2:realign-no-topk-fit-objective`); gradient fit vs closed form at equal ratio on the same component (no pair exists); full-parameter KD vs LoRA KD on the same child.

## 6. What follows for the objective design (not a gate)

- Use teacher-forced top-k containment (parent top-1 inside child top-5/10) plus KL as the fold screen and optimizer target. On LFM2 it ranks children as well as anything measured (rho about 0.9) and slightly better among living children. It matches D 821 (top-n as metric and target) and keeps top-1 as a reported column rather than the goal.
- Stop using free-running greedy token match as a screen. It is the weakest whole-model metric on the same children (0.685 vs 0.890), and the misleading "token match" history comes from it.
- Never screen on local write cosine or local held-out relL2 for composed children.
- The unrun test that matters is X2: a fold fitted to a top-k/KL objective vs the same fold fitted by ridge, at equal ratio, read by behaviour. Section 3b gives the screen to compare them on before the reads.

## 7. Not done / could not verify

- No new behaviour reads. The 3b outcome is the existing frozen-15 reads (Claude), and no Cyril-produced count exists for any child (FOLDING_RECORD section 9).
- V* (vocab-coreset) children, K3, Qwen-scale children and trained children were not probed. Token-level agreement is undefined across changed vocabularies, and the larger children are deleted or never decoded.
- The probe results were not ingested into omni_lab (writes only go through its ingesters after a dump). They live in `objective-retention-20261029/probe_all.jsonl` for a later ingest.
- The probe used the HD48 dossier's parent responses as set A. That dossier's parent decode was GPU fp32 at a 1,024-token cap, and I used the first 384 response tokens.

## Appendix A: queries

Run as `omni_lab_ro` via `omni_lab_db.connect("ro")` (`objective-retention-20261029/dump.py`) or `python omni_lab.py sql "<query>"`.

```sql
-- dump (dump.py)
SELECT experiment_id, arm_key, title, program, wave, owner, parent_name, child_ref, ratio_body, ratio_net, ratio_text,
       objective_fit, objective_target, objective_k, objective_horizon, objective_text, training_used, training_text,
       verdict_class, verdict_text, axes FROM fold_arm;
SELECT id, experiment_id, ordinal, suite, scorer, score, score_value, score_denominator, useful, partial, miss, total,
       column_b, parent_score, text FROM fold_behaviour;
SELECT id, experiment_id, ordinal, name, value_text, value_num, value_unit, horizon, denominator, split, note FROM fold_diagnostic;
SELECT experiment_id, ordinal, kind, layers, technique_text, ratio, ratio_text, rank_or_k FROM fold_component;

-- diagnostic names and coverage
SELECT name, count(*) n, count(value_num) nn, count(distinct experiment_id) arms FROM fold_diagnostic GROUP BY 1 ORDER BY 2 DESC;
-- scorer x suite
SELECT scorer, left(suite,50) suite, count(*) n, count(score_value) nv, count(score_denominator) nd, count(total) nt
  FROM fold_behaviour GROUP BY 1,2 ORDER BY 3 DESC;
-- top-k rows anywhere
SELECT a.arm_key, a.program, d.name, d.value_text, d.note FROM fold_diagnostic d JOIN fold_arm a USING (experiment_id)
 WHERE d.value_text ~* 'top5|top-5|top_5|top10|top-10' ORDER BY a.arm_key;
-- sparse-semantic behaviour
SELECT a.arm_key, a.verdict_class, a.ratio_net, b.scorer, b.score, b.useful, b.partial, b.miss, b.total, left(b.suite,40)
  FROM fold_arm a LEFT JOIN fold_behaviour b USING (experiment_id)
 WHERE a.program='sparse-semantic' AND (b.scorer IS NULL OR b.scorer<>'none') ORDER BY a.arm_key;
-- objectives x training x human-read coverage
SELECT objective_fit, objective_target, training_used, count(*) n,
       count(*) FILTER (WHERE EXISTS (SELECT 1 FROM fold_behaviour b WHERE b.experiment_id=a.experiment_id
              AND b.scorer IN ('claude','codex','mixed','cyril') AND b.score_value IS NOT NULL)) nh
  FROM fold_arm a GROUP BY 1,2,3 HAVING count(*)>=10 ORDER BY n DESC;
-- gradient / KD / OPD arms with any behaviour
SELECT a.arm_key, a.program, a.objective_fit, a.objective_target, a.training_used, a.ratio_net, b.scorer, b.score
  FROM fold_arm a JOIN fold_behaviour b USING (experiment_id)
 WHERE (a.objective_fit IN ('gradient','kd','on_policy') OR a.training_used IN ('gradient_fit_of_fold','recovery_after_fold'))
   AND b.scorer <> 'none' AND b.score IS NOT NULL AND b.score <> '-' ORDER BY a.program, a.arm_key;
-- retention studies
SELECT a.arm_key, d.name, d.value_text FROM fold_diagnostic d JOIN fold_arm a USING (experiment_id) WHERE a.arm_key ~* 'retention-';
SELECT a.arm_key, b.scorer, b.score, left(b.suite,50) FROM fold_behaviour b JOIN fold_arm a USING (experiment_id) WHERE a.arm_key ~* 'retention-';
-- children on disk with a human read
SELECT a.program, a.child_exists_now, count(distinct a.experiment_id) FROM fold_arm a
 WHERE EXISTS (SELECT 1 FROM fold_behaviour b WHERE b.experiment_id=a.experiment_id
               AND b.scorer IN ('claude','codex','mixed','cyril') AND b.score_value IS NOT NULL) GROUP BY 1,2;
```

## Appendix B: files

`C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/objective-retention-20261029/`:
- `dump.py`: read-only DB dump (the dump itself was not kept; re-run to regenerate)
- `analyze.py`: outcome and metric-family mapping, section 2a tables → `results.json`
- `analyze2.py`: alive band, ARCH vs parent-shaped, paired differences, S4 technique table, sealed vs robust → `results2.json`
- `verdict_sens.py`: verdict-word sensitivity → `verdict_sens.json`
- `sparse_table.py`: section 3a AUCs
- `topk_probe.py`: the CPU probe (server copy and logs in `/home/cereal/ora-20261029/`) → `probe_all.jsonl`, frozen text `probe_seqs.json`
- `analyze_probe.py`: section 3b → `probe_results.json` (includes the per-child table)
- `analyze_stats.py`: rank statistics
