# Neural Forge review, 2026-06-26 to 2026-11-07

This review covers Neural Forge's folding work from the first commit (2026-06-26) to now. The
work is ongoing: folding continues as the way the parents' weights get into Chimera Omni, and
each section ends with what comes next.

It was written from four sources:
- the code and reports in this repository;
- the `omni_lab` folding-record database (exported to
  [`db/omni_lab_foldrec_2026-11-06/`](../db/omni_lab_foldrec_2026-11-06/README.md));
- the agents' memory notes;
- the Relay coordination threads.

Every number comes from a committed document or a database row, and each section names where
to find it.

Cyril's goal is one automatic compiler that folds any parent model into a smaller,
user-defined child. It should handle any architecture, tokenizer or modality, at 4x compaction
first and 10x-1000x later. The child is judged by whole decoded answers, with a long-run target
of about 90% of the parent's useful intelligence (`docs/CURRENT_FOLDING_GOAL.md`). Folding
started fold-only. Since October, folds initialise a child that is then distilled from the
original parents.

---

## 1. Summary

1. **Folding works on real models.**
   - A single component folded 4x keeps 10-13/15 on the frozen review, against the parent's
     12/15. This holds for MLP down-projections, conv writers, attention keys and Q/K. The
     techniques are T8 Gram relation matching, T5 sparse residual-OMP routes, T1 ridge on Q/K
     and T3/T4 on conv.
   - Cheap whole-model axes keep most of the parent's behaviour:
     - a vocab row coreset keeps 14/15 at 0.795x (`ARCH.V19`);
     - one block removed at the right site keeps 13/15 at 0.947x (`ARCH.X13A`);
     - vocab plus one excision plus a terminal gain keeps 11/15 at 0.747x (`ARCH.V20X13G03`).
2. **Whole children well under half the parent's size still answer.**
   - The deepest LFM2 child that still holds a lane that is not a copy of its input is at
     0.412x (`ARCH.V10CL896X09`).
   - A 0.488x child answered 2 of 15. Cyril read it himself: the child loops, but he counted it
     as the first model to keep anything at that depth and said the work was on the right
     track.
   - On 2026-08-28 he ruled that folds below half the parent's size that keep intelligence to a
     degree count as a success.
3. **Kimi K3 was folded.**
   - Kimi K3 is 2.78T total and 104.2B active, with 93 layers and 896 routed experts per MoE
     layer.
   - The all-experts fold (`K3DP.04db523ceafa`) replaced every MoE layer with one dense FFN:
     29.8B parameters per layer became 371M, an 80x reduction. Across the 92 MoE layers, about
     2.74T routed parameters became about 34.1B. The whole 93-layer folded stack was built.
   - At the end of the stack it agrees with K3's top-1 token 69.2% of the time (top-5 set
     0.596, KL 1.0).
   - The K3 fold was then folded into the 27B Chimera trunk as a union FFN, giving a 26.9B
     dense child (`S5-C2K4352K3U-a003`). Section 4 has the details.
4. **Big models fold too.**
   - The Qwen3.8-27B trunk with its MLPs folded 4x on all 64 layers reads 8/15 at 0.523x of
     the whole checkpoint (`OC02-F-Q38-27BU-C2K4352`, Codex-sealed).
   - Qwen3.5-4B with one of its 32 layers removed reads 10-15/15 (machine read).
   - A short full-parameter distillation run on the folded 26.1B Chimera student improved it:
     4 right, 3 partly right and 5 wrong of 12, with an exact tool call. That is clearly better
     than the fold alone, though still below the parent. Cyril judged on 2026-10-21 that the
     model was recovering and becoming more intelligent.
5. **The campaign measured the laws that matter for the next step.**
   - Composition is where the cost appears. One structural fold is cheap and a second costs
     more. Each independently solved component costs about three review cases.
   - A correction that improves local cached fit beyond its prior's size composes worse than
     the prior did. This was measured three times at three scales.
   - Cheap screens rank children well: lambada perplexity (rho 0.83) and drift against a
     copy-exact control (rho 0.81). Local write cosine carries no signal.
   - These findings turn folding from a search into an engineering plan (section 7).
6. **The engineering is solid and reusable.**
   - A fold compiler with sealed, content-bound evidence and fresh-reload certification.
   - Executable lanes for text, image, audio and video.
   - Kill-and-resume, byte-identical multi-GPU replay and an on-policy evidence loop.
   - Every run recorded in one database: 7,895 fold arms and 2,947 findings, 1,122 of them
     proven.
7. **Where it stands.**
   - On 2026-10-21 Cyril set the direction: a 2B-then-4B on-device omni model with shared
     all-modality weights, built from the parents.
   - "Run 0" (a planted 2B distilled from the parents) has trained since the end of October as
     the comparison baseline.
   - The next folding milestone is the fold pass: build the v2 trunk from the parents (K3
     primary, plus the media parents) with the proven layer, width and residual-attention
     folds, then distil it and compare it against Run 0.

---

## 2. Scale of the work

| Item | Count | Where |
|---|--:|---|
| Commits on the main work branch | 1,400 (2026-06-26 → 2026-10-21) | `codex/lrc-blank-compression` |
| Python modules in `src/neural_forge` | 417 (about 34 MB of source) | `src/` |
| Test files | 592 (9,792 passing at the 2026-08-01 green certification) | `tests/` |
| Experiment runners | 192 | `experiments/` |
| Design and result documents | 84 in `docs/`, 89 wave notes (`NOTES-*.md`) | |
| Matrix-campaign ledger | 1,056 rows over 45 waves | `reports/matrix-campaign-20260810/ledger.jsonl` |
| Fold arms in the database | 7,895 (3,393 distinct children) | `fold_arm` |
| Findings | 2,947: 1,122 proven, 1,301 supported, 197 open, 152 closed paths, 108 superseded, 67 refuted | `fold_finding` |
| Behaviour reads | 6,674 rows; 2,775 read by Claude, 2,006 by a machine rule, 147 by Codex, 11 tagged Cyril | `fold_behaviour` |

Verdict classes across all arms:
- works 342, partial 786, mechanics only 1,304, design only 422;
- unmeasured 1,628;
- fails 2,364;
- closed path 362, superseded 193, shards disagree 494.

Separately, 2,430 arms were dropped on a number with nobody reading their output. They are
unreviewed, not failed (section 6).

---

## 3. What happened, in order

| When | Phase | What it established |
|---|---|---|
| Jun 26-30 | Early surgery (Qwen3, SmolLM2, cross-width) | Layer merges on Qwen3-0.6B pass at 2.6-5.3% shrink, and the first cross-family transplants ran. This phase mapped which naive folds fail, which pointed the compiler at behaviour-first fitting. |
| Jul 1-16 | Dynamic LFM2 compiler | The automatic system: target synthesis, fold compiler, production catalog, sealed artifacts. Lanes for LFM2 and Qwen3 text, NEO image+text. |
| Jul 17-31 | Exact-4x LFM2.5-230M → 57.4M (H816), runs r1-r11 | Each run's failure was diagnosed and fixed. The A9/A11 sparse-semantic token bridge reached 10-12/16 at 1.35x counted, with 90.28% cached argmax retention (A11). |
| Aug 1 | Green certification | Full suite 9,792 passed / 0 failed. Audio and video became executable modalities (compile → fold → gate → publish → fresh reload). A cross-family fold ran (llama → qwen3-shaped child). Cross-width charts held on real SmolLM2 H576→H432 (96-99% held-out energy). |
| Aug 4-5 | r12, the first fully certified exact-4x publication | The pipeline works end to end. The child repeated itself, and a cache-only diagnosis located the cause at layer 0, which gave the next phase its target. |
| Aug 6-8 | Stage-1 ceiling report | It measured where single-shot dense width contraction stops on one parent. Cyril reopened stage 1 because the report covered one mechanism, not every idea, and that started the matrix campaign. |
| Aug 10-29 | Matrix campaign, 45 waves | Every component × technique × site, then composition. Per-component 4x works across techniques. Whole children reached 0.412x while still holding a real lane. |
| Aug 29 | Chimera Omni | One dense omni model folded from Qwen3.8-27B, Kimi K3, MiniMax H3 and the image/3D/audio parents. |
| Sep-Oct | Chimera Omni folds (OC01, S3, K3 dense pass, C10, S7/OMNI1, X3, full KD) | Qwen3.5-4B layer excision 10-15/15. 27B trunk MLP 4x on all layers 8/15 at 0.523x. Kimi K3 folded all-experts into a dense stack (80x per MoE layer, 0.692 end-of-stack top-1) and folded into the 26.9B trunk. Full KD on the folded student improved it fast. |
| Oct 21-22 | Realignment and the lab database | `omni_lab` became the single database for every run. Cyril picked a 2B→4B on-device model with shared all-modality weights, with the parents as teachers. |
| Oct 22-29 | Consolidating the record | `FOLDING_RECORD_2026-10-22.md` (built from the database, claims C01-C24 re-checked against sources), `TECHNIQUE_SCOREBOARD_2026-10-29.md`, `OBJECTIVE_RETENTION_ANALYSIS_2026-10-29.md`, `JUDGEMENT_AUDIT_2026-10-29.md`. |
| End of Oct → now | Run 0 and the fold pass | co2-2b, planted from Qwen3.5-2B, is distilled on every modality on four P40s as the baseline. The fold pass (v2 trunk built from the parents) is next. |

The Chimera Omni reports, from September on, live on the `claude/omni-campaign` branch under
`reports/omni-campaign-2026-09/`.

---

## 4. Results

### Kimi K3

- **Parent:** Kimi K3, 2.78T total / 104.2B active parameters. It has 93 layers: 69 KDA and 24
  gated MLA. Each MoE layer has 896 routed experts (top-16) plus 2 shared experts.
- **The fold** (`K3DP.04db523ceafa`, `k3_dense_pass` 1.2.5, 2026-10-09). This is Cyril's
  2026-09-02 design: all experts of all layers, folded in order. Each layer was fitted on the
  activations of the already-folded prefix, so each fit absorbs the error before it.
  - Every MoE layer became one dense FFN: 29,784,936,832 → 370,675,200 parameters (80.35x).
  - Over the 92 MoE layers that is about 2.74T → 34.1B. Layer 0 is dense and was kept as is.
  - The whole folded 93-layer stack exists.
- **Retention at the end of the stack:** held top-1 agreement 0.692, top-5 set 0.596, KL 1.0.
  These are diagnostics. The dense K3 stack has not yet been decoded on its own.
- **Into the Chimera trunk** (`S5-C2K4352K3U-a003`, 26,944,218,624 parameters).
  - Each layer's FFN is a union of 4,352 Qwen trunk rows (a quarter of the parent's 17,408)
    and 12,288 K3 rows.
  - The a003 weight was picked as the largest K3 contribution that keeps the trunk coherent.
- **After distillation** (full-parameter, K3 as primary teacher). After a short run, the run-1
  student (26.1B) on that folded line read 4 right, 3 partly right and 5 wrong of 12 tests,
  with the tool call exactly right. That is clearly better than the fold alone (OMNI1),
  though still below the parent.
- K3 also serves as the primary distillation teacher. Its teacher cache was ratified on
  behaviour on 2026-10-28: K3's top-1 was coherent and on topic in all 23 routes read.

### LFM2.5-230M (fold-only)

Every row was read on behaviour by an agent (Claude, unless marked).

| Result | Score | Ratio | Key | Source |
|---|---|---|---|---|
| Single component at 4x: MLP down-proj (T8) | 12/15 vs parent 12/15 | component 4x | `C13.T8.S4.L05` | `MX/wave5/NOTES-MX-I.md` |
| Single component at 4x: conv writer (T3, T4, T8) | 12/15 | component 4x | `C05.T3.S4.L05`, `C05.T4.S4.L05` | technique table, `FOLDING_RECORD` §2 |
| Single component at 4x: attention key (T5 routes) | 12/15 | component 4x | `C07.T5.S4.L12` | `FOLDING_RECORD` §5 |
| Attention Q/K ridge (T1) | 10-11/15 | component 4x | `C06/C07.T1.S4.L12` | matrix waves 1-2 |
| Two-component child | 13/15 | two components at 4x (stored parent-shaped, 0.994x) | `COMP.MA.MXK14` | `MX/wave7/NOTES-MX-K.md` |
| One block removed at the right site | 13/15 | 0.947x | `ARCH.X13A` | `TECHNIQUE_SCOREBOARD` §0 |
| Vocab row coreset | 14/15 | 0.795x | `ARCH.V19` | matrix waves 23-26 |
| Vocab + one excision + terminal gain | 11/15 | 0.747x | `ARCH.V20X13G03` | `MX33.F1` |
| Sparse-semantic dual route (whole model) | 10-12/16 | 1.35x counted | A9/A11 | `STAGE1_CEILING_REPORT` |
| Deep whole-model folds | 2/15 at 0.488x; one real lane at 0.412x | 0.412-0.488x | `ARCH.V20CL896` (most likely), `ARCH.V10CL896X09` | `MX/DEEP_RECORD.md` |
| One trained QKVO repair at one site | 8/8 four-token rollouts | 0.08% of params | `p35-qkvo-L19G4` | `FOLDING_RECORD` §5 |

### Qwen and the 27B trunk

| Result | Score | Ratio | Key |
|---|---|---|---|
| Qwen3.5-4B, one layer of 32 excised | 10-15/15 (machine read) | about 0.95-0.975x | `OC01-F-Q35-4B-X03/X08/X15/X16` |
| Q38-27B trunk, MLP 4x on all 64 layers | 8/15 (Codex-sealed) | 0.523x whole checkpoint | `OC02-F-Q38-27BU-C2K4352` |
| Q38-27B trunk + K3 union | coherent at a003 | 26.9B | `S5-C2K4352K3U-a003` |
| Full-KD student on the folded line | 4 right / 3 partly right / 5 wrong of 12, tool call exact | 26.1B | `FULLKD-run1-student` |

### Media

The image, 3D, audio and video parents fold their MLPs at 4x, and each folded block runs
finite inside its own segment of the omni child. Behaviour on media has not been measured
yet.

### Engineering that is proven and reusable

- **Fold compiler with certification:** content-bound cached evidence and zero live-parent
  forwards. Out-of-transaction mutation is rejected, and the child is reloaded from its final
  path before it is certified (`docs/DYNAMIC_FOLDING_COMPILER_2026-07-16.md`).
- **Executable modality lanes:** text, image, audio and video, each with compile → fold →
  gate → publish → fresh reload. 3D and OCR exist as codecs.
- **Runtime:**
  - kill-and-resume through the production catalog;
  - multi-GPU replay sharding, byte-identical serial versus four-way, with cross-card
    bit-identity proven on real hardware;
  - on-policy evidence rounds.
- **Tokenizer bridge:** a 4,096-anchor ByteLevel segmentation covers all 64,402 donor ids. An
  independent-vocab pair bridge was validated on the real LFM2 vocab.
- **Large-model folding on local hardware:** K3's 1.5 TB checkpoint was folded one layer at a
  time in RAM on the AI server, with resumable layer windows.
- **Reproducibility:** wave 45 rebuilt the floor holder's checkpoint bit-exact from receipts,
  16 waves after it was made.

---

## 5. What the campaign learned about the hard parts

These are the problems still open. Each finding narrows what the next fold has to do.

| Finding | Evidence | What it means for the next step |
|---|---|---|
| Composition costs more than any single fold | Each independently solved component costs about three review cases (`ST.five-3`). Fourteen MLP coresets that are each alive on their own compose to 0/15 (`MX14.F3`). | Fold fewer, larger units, and fit each fold on the already-folded prefix: this is the K3 dense-pass design. Then distil. |
| Local fit does not predict composed behaviour | Corrections that improve cached fit beyond their prior compose worse (measured three times, `STAGE1_CEILING_REPORT` §8). | Judge on the whole model's behaviour. Use top-n as an optimizer target, never as a gate. |
| Layer 0 sets the floor for width contraction on LFM2 | r12: relL2 1.159 after layer 0 alone. The solver reproduces layer 0 to cosine 0.9995, so the solver is not the limit. | Contract MLP intermediate width before hidden width, and pick the hidden basis from the pooled reader moment. |
| Hidden-width cliff | On LFM2, hidden-width cuts below about 0.86 of width collapse. Read energy is a ceiling, not a predictor (0.9178 gives 0/15, 0.9193 gives 3/15). | Width ratios like 5120 → 2048 need their own tests. |
| Depth beyond one or two blocks | The excision ladder reads 13 → 7 → 3 → 0. | Remove one or two blocks at reviewed sites. Deeper cuts go with distillation. |
| Head merge (T7/M5) | The judgement audit found both built sites alive at 4x. | Usable at single sites. |

**Metrics that guide well, and metrics that don't** (`OBJECTIVE_RETENTION_ANALYSIS_2026-10-29.md`
§0):
- Good cheap screens: lambada perplexity (rho 0.83) and drift against a copy-exact control
  (rho 0.81).
- Good for ranking: teacher-forced top-1 on parent text (rho 0.890), with top-10 overlap
  (0.883) and top-1-in-top-5 (0.898) close behind.
- Weak: free-run greedy match (0.685) and local write cosine (rho 0.06, no signal).
- Every metric drops to rho 0.4-0.7 inside the alive band, so behaviour stays the final read.

---

## 6. How the work was judged

The way results were measured changed during the campaign. That makes several early negative
verdicts unreliable, and the true results are likely better than the strictest numbers show.

- **The instrument moved.** The frozen 15-prompt review, the robust three-restatement re-read,
  numeric gates and, from October, top-n matching were each used at different times.
  Verdicts made under one instrument should not be compared directly with verdicts made under
  another. The robust re-read is much stricter: it takes the parent itself from 12 to 6
  (`MX33.F1`).
- **Numbers decided too much, early.** 2,430 arms were dropped on a number with nobody reading
  their output. Those arms are unreviewed, not failed.
  - The 10-29 judgement audit re-read 74 of them and found six families dropped wrongly.
  - Examples: 9 of 12 census-failed excision blocks are alive or partially alive, and
    single-site head merges are alive at 4x.
  - Under Cyril's 2026-10-22 rule, no number passes or fails a fold; behaviour does.
- **Storage format and behaviour.** Matrix children in waves 5-13 are stored parent-shaped, so
  their 4x is per component. Body contraction is what governs behaviour.
- **Cyril's reads.** Eleven rows carry his alive/dead rulings and reactions. The September and
  October children have not had his read yet, and putting a few of them in front of him is
  part of the next step.

---

## 7. What comes next

This section draws on `FOLDING_RECORD` §11, `TECHNIQUE_SCOREBOARD` §10 and the 10-27 plan.

**The fold pass.** Build the v2 trunk from the parents with K3 as primary, using:
- one or two blocks removed at reviewed sites;
- per-component 4x folds (T8, T5, T1 on Q/K, T3/T4 on conv);
- a vocab row coreset;
- MLP width before hidden width;
- K3's all-experts dense fold;
- the media parents' MLP folds.

Use top-n as the optimizer target and behaviour as the gate. Then distil from the original
parents, which demonstrably improved the folded student, and compare the result against Run 0.

**Open items, each a small pre-registered test with a behaviour read on `omni_lab`:**
1. Decode and review the dense K3 stack on its own, and the K3 union at higher weights after
   distillation.
2. Width ratios like 5120 → 2048, and depth reduction beyond two layers on a large parent.
3. Mixer kind translation: KDA → trunk mixer, MLA → GQA, SiTU → SiLU.
4. A fold fitted directly to a top-n target (designed as X2).
5. Composition at the depth a parent-to-2B trunk needs: width, depth, attention and vocab
   together.
6. Scale transfer of the LFM2 laws to the 27B and K3 scale.
7. Behaviour reads on the media folds.
8. Cyril's read of three or four of the best children: `MXK14`, `ARCH.V19`, `ARCH.X13A`, the
   27B MLP-4x trunk and the K3 line.

**Measurement going forward:**
- Keep one instrument per comparison and record which one was used.
- Use lambada perplexity and drift as screens, and teacher-forced top-1/top-n for ranking.
- Never drop an arm on a number without a behaviour read.

---

## 8. Where everything is

| Branch | What it holds |
|---|---|
| `main`, `codex/lrc-blank-compression` | The compiler, tests, experiments, docs, matrix campaign through wave 45, this review and the database export. |
| `claude/omni-campaign` | Chimera Omni campaign reports and tools, Sept → Nov (`reports/omni-campaign-2026-09/`): FOLDING_RECORD, TECHNIQUE_SCOREBOARD, audits, Run 0 plans, the K3 dense pass, crash and drift reports. |
| `omni-lab` | The `omni_lab` database code: migrations, ingesters, runner, cleanup tooling. |
| `wip/neural-forge-uncommitted-20261106` | Neural Forge code, tests and run evidence that were never committed (mostly Aug 9-10). It is preserved as-is and has not been re-tested. |
| `backup/20261028/*` | Weight-stripped snapshots of every local branch and dirty worktree, taken 2026-10-28. |

**Not published:**
- model weights, tensors and caches;
- corpora and media outputs;
- the cross-project directive log;
- machine inventory.

Weights deleted in the 2026-10-21 cleanup can be rebuilt from the recipes recorded in
`model_deletion`.

## 9. Cleaning done for this publication

- The pushed history is weight-stripped. Seven fold-child `model.safetensors` files (69-535 MB
  each) that the local history carried are not in it.
- Every published file was scanned for the 137 vault credentials and for key, token and
  password patterns. It holds no secrets. The only vault match is `lmstudio`, LM Studio's
  default placeholder key, used as a word in model paths.
- Left out on purpose:
  - `*.safetensors`, `*.pt` and other tensor files, and files over 10 MB;
  - `.codex-test-runs/` scratch and `proof_outputs/` (about 1 GB of generated outputs);
  - media, logs and `__pycache__`;
  - the raw folding-record extraction shards (the database export carries their content);
  - the NAS/workstation cleanup plans.
- Six files that were dumps of Cyril's prompts and directives were removed from the history of
  every branch that held them.
- Cyril's own words are withheld across the whole public history (2026-11-07): every report,
  doc, test fixture, commit message and the database export. Each quote was tested against what
  he wrote (his directives and rulings in `omni_lab`, and his prompts in agent memory), and a
  match reads `[Cyril quote withheld]`. The agents' paraphrases of his rulings, and the dates
  and event ids that cite them, are kept, so each ruling can still be followed.
