# Neural Forge review, 2026-06-26 to 2026-11-06

This review covers all of Neural Forge's folding work, from the first commit (2026-06-26) to
the end of folding as the main line (2026-10-29), plus what has happened since. It was written
from four sources: the code and reports in this repository, the `omni_lab` folding-record
database (exported to [`db/omni_lab_foldrec_2026-11-06/`](../db/omni_lab_foldrec_2026-11-06/README.md)),
the agents' memory notes, and the Relay coordination threads. Every number is quoted from a
committed document or a database row, and each section names where to find it.

The goal Cyril set for the project was one automatic compiler that folds any parent model into
a smaller, user-defined child: any architecture, tokenizer or modality, at 4x compaction now
and 10x-1000x later. It was to be fold-only (no training, pruning, quantization or
distillation), judged by whole decoded answers, with a target of about 90% of the parent's
useful intelligence kept (`docs/CURRENT_FOLDING_GOAL.md`).

---

## 1. Bottom line

1. **Fold-only did not reach 4x with retained intelligence on any whole model.** On
   LFM2.5-230M, every child at or below 1.559x *body* contraction scored 0/15 on the frozen
   review. The best whole-model fold-only result is the A9/A11 dual-route child: 1.35x counted
   compaction, but 1.000x body, with 10-12 of 16 prompts useful or partial
   (`docs/STAGE1_CEILING_REPORT_2026-08-08.md`).
2. **Folding works one component at a time.** A single component folded 4x, such as an MLP
   down-projection, a conv writer or an attention key, keeps 10-13/15 against the parent's
   12/15. The techniques are T8 Gram relation matching, T5 sparse residual-OMP routes and T1
   ridge on Q/K. The best composed child (`COMP.MA.MXK14`) reads 13/15, but it stores
   0.994x of the parent: the model on disk is parent-shaped (`MX14.F1`).
3. **Composition is the wall, not the solver.** One structural fold is survivable and a
   second is often fatal. Fourteen MLP coresets, each inside the "alive" fidelity band on its
   own, compose to 0/15. The program's central law: a closed-form correction that improves
   local cached fit beyond the size of its prior composes *worse* than the prior did. It was
   measured three times at three scales (`STAGE1_CEILING_REPORT` §8; `FOLDING_RECORD` §7).
4. **The cheap axes are real but small.** A vocab row coreset kept 14/15 at 0.795x
   (`ARCH.V19`). Vocab plus one block excision plus a terminal gain kept 11/15 at 0.747x
   (`ARCH.V20X13G03`). Hidden-width cuts below about 0.86x of width are dead.
5. **The measuring was weaker than the folding.** Review counts are upper bounds: re-read with
   three restatements per lane, the parent itself drops from 12 to 6. Cyril produced none of
   the counts on record. 2,430 of 7,895 arms were dropped on a number with nobody reading
   their output, and the 10-29 judgement audit found six families among them that had been
   dropped wrongly (`JUDGEMENT_AUDIT_2026-10-29.md`).
6. **The engineering is the durable asset.** It includes a fold compiler with sealed,
   content-bound evidence and fresh-reload certification, and executable lanes for text,
   image, audio and video. Kill-and-resume, multi-GPU replay and an on-policy evidence loop
   are proven, and every run is recorded in one database. The first fully certified exact-4x
   publication (r12) proved that the pipeline works end to end, even though its child was
   incoherent.
7. **Where it stands now.** On 2026-10-21 Cyril realigned the program toward a 2B-then-4B
   on-device omni model, with the parents used as teachers. Folding became an initializer and
   distillation became the main line ("Run 0", training since the end of October). Folding work is closed
   out in the folding record, the technique scoreboard and the judgement audit.

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

**Verdict classes across all arms:** fails 2,364; unmeasured 1,628; mechanics only 1,304;
partial 786; works 342; design only 422; closed path 362; superseded 193; shards disagree 494.
"Works" includes very small shrinks and mechanism-only passes. It never means "4x with kept
intelligence" by itself.

---

## 3. What happened, in order

| When | Phase | What it established |
|---|---|---|
| Jun 26-30 | Early surgery (Qwen3, SmolLM2, cross-width) | Layer merges on Qwen3-0.6B pass their gates only at 2.6-5.3% shrink. Blank-target width folds and early cross-family transplants fail on behaviour. |
| Jul 1-16 | Dynamic LFM2 compiler | The automatic system: target synthesis, fold compiler, production catalog, sealed artifacts. Lanes for LFM2 and Qwen3 text, NEO image+text. |
| Jul 17-31 | Exact-4x LFM2.5-230M → 57.4M (H816), runs r1-r11 | Each run died before publication, and each failure was diagnosed and fixed. The A9/A11 sparse-semantic token bridge reached 10-12/16 at 1.35x (90.28% cached argmax retention, A11). The carrier-lane end-to-end test found five stacked publication defects. |
| Aug 1 | Green certification | Full suite 9,792 passed / 0 failed. Audio and video became executable modalities (compile → fold → gate → publish → fresh reload). A cross-family fold ran (llama → qwen3-shaped child). Cross-width charts held on real SmolLM2 H576→H432 (96-99% held-out energy); conv→attention kind translation degraded on real LFM2. |
| Aug 4-5 | r12 certified, then diagnosed | The first complete certified exact-4x publication read 0/15: degenerate repetition. A cache-only diagnosis put the wall at layer 0: C_in is exact, then layer 0 alone takes held-out relL2 to 1.159, worse than emitting zero. |
| Aug 6-8 | Stage-1 ceiling (fold-only) | Cyril's staged policy: prove fold-only's ceiling before fold + distillation. Sealed report: 0/15 at ≤1.559x body. Every named remedy was exhausted against a measured cause. The solver reproduces layer 0 to cosine 0.9995, so the solver was never the limit. |
| Aug 10-29 | Matrix campaign, 45 waves | Every component × technique × site, then composition. Per-component 4x works; composition fails. Ended with two floors: column A 0.386x (echo persistence, not intelligence) and column B 0.412x (the deepest child holding one lane that is not a copy of its input). |
| Aug 29 | Pivot | Cyril: LFM2 is at its limits; move to Chimera Omni, one dense omni model folded from Qwen3.8-27B, Kimi K3, MiniMax H3 and the image/3D/audio parents. |
| Sep-Oct | Chimera Omni folds (OC01, S3, K3 dense pass, C10, S7/OMNI1, X3, v2 proxy) | Qwen3.5-4B, one layer excised: 10-15/15 (machine). MLP 4x on all layers: 5/15 on the 4B, 8/15 on the 27B trunk (Codex-sealed). The K3 all-experts dense pass is mechanically complete and was never decoded. Media units fold their MLPs only. |
| Oct 21-22 | Realignment and the lab database | `omni_lab` became the single database for every run. Architecture review: the trunk had drifted from Cyril's founding spec. Cyril picked a 2B→4B on-device model with shared all-modality weights and the parents as teachers. |
| Oct 22-29 | Closing the record | `FOLDING_RECORD_2026-10-22.md` (built from the database, claims C01-C24 re-checked against sources), `TECHNIQUE_SCOREBOARD_2026-10-29.md`, `OBJECTIVE_RETENTION_ANALYSIS_2026-10-29.md`, `JUDGEMENT_AUDIT_2026-10-29.md`. |
| End of Oct → now | Run 0 (training, not folding) | co2-2b planted from Qwen3.5-2B and distilled on every modality on four P40s. It hit and fixed a cuFFT crash at step 1230; the held-out drift was traced to the gauge's own agent-pipeline texts. |

The Chimera Omni reports, from September on, live on the `claude/omni-campaign` branch under
`reports/omni-campaign-2026-09/`.

---

## 4. What is proven to work

Every row was read on behaviour by an agent (Claude, unless marked). None is a Cyril count.

| Result | Score | Ratio | Key | Source |
|---|---|---|---|---|
| Single component at 4x: MLP down-proj (T8) | 12/15 vs parent 12/15 | component 4x | `C13.T8.S4.L05` | `MX/wave5/NOTES-MX-I.md` |
| Single component at 4x: conv writer (T3, T4, T8) | 12/15 | component 4x | `C05.T3.S4.L05`, `C05.T4.S4.L05` | technique table, `FOLDING_RECORD` §2 |
| Single component at 4x: attention key (T5 routes) | 12/15 | component 4x | `C07.T5.S4.L12` | `FOLDING_RECORD` §5 |
| Attention Q/K ridge (T1) | 10-11/15 | component 4x | `C06/C07.T1.S4.L12` | matrix waves 1-2 |
| Two-component child | 13/15 | 0.994x stored (parent-shaped) | `COMP.MA.MXK14` | `MX/wave7/NOTES-MX-K.md` |
| One block removed at the right site | 13/15 | 0.947x | `ARCH.X13A` | `TECHNIQUE_SCOREBOARD` §0 |
| Vocab row coreset | 14/15 | 0.795x | `ARCH.V19` | matrix waves 23-26 |
| Vocab + one excision + terminal gain | 11/15 (robust re-read 2/15 on its sibling) | 0.747x | `ARCH.V20X13G03` | `MX33.F1` |
| Sparse-semantic dual route (whole model) | 10-12/16 | 1.35x counted, 1.000x body | A9/A11 | `STAGE1_CEILING_REPORT` |
| One trained QKVO repair at one site | 8/8 four-token rollouts | 0.08% of params | `p35-qkvo-L19G4` | `FOLDING_RECORD` §5 |
| Qwen3.5-4B, one layer of 32 excised | 10-15/15 (machine upper bound) | — | `OC01-F-Q35-4B-X03/X08/X15/X16` | `TECHNIQUE_SCOREBOARD` §0 |
| Q38-27B trunk, MLP 4x on all 64 layers | 8/15 (Codex-sealed) | 0.523x whole checkpoint | `OC02-F-Q38-27BU-C2K4352` | `FOLDING_RECORD` §2 |

Engineering that is proven and reusable:
- **Fold compiler with certification:** content-bound cached evidence, zero live-parent
  forwards, out-of-transaction mutation rejected, and the child reloaded from its final path
  before it is certified (`docs/DYNAMIC_FOLDING_COMPILER_2026-07-16.md`).
- **Executable modality lanes:** text, image, audio and video, each with compile → fold →
  gate → publish → fresh reload. 3D and OCR exist only as codecs.
- **Runtime:** kill-and-resume through the production catalog. Multi-GPU replay sharding is
  byte-identical serial versus four-way, and cross-card bit-identity was proven on real
  hardware. On-policy evidence rounds are wired.
- **Tokenizer bridge:** a 4,096-anchor ByteLevel segmentation covers all 64,402 donor ids. An
  independent-vocab pair bridge was validated on the real LFM2 vocab.
- **Reproducibility:** wave 45 rebuilt the floor holder's checkpoint bit-exact from receipts,
  16 waves after it was made.

---

## 5. What did not work, and why

| Wall | Evidence | Cause on record |
|---|---|---|
| Whole-model exact 4x, fold-only | r12 0/15; r13 arms 0/15; three stage-1 children 0/15 | Layer 0 already exceeds the zero baseline (relL2 1.159, 93% irreducible direction error). Full-attention layers amplify magnitude about 1.49x per layer, not information. |
| Composing body folds | 14 SwiGLU coresets, each alive alone → 0/15 (`MX14.F3`); two folds on tensors that share nothing: 13/15 + 9/15 → 5/15 | Each independently solved component costs about 3 review cases (`ST.five-3`). Local fit does not predict composed behaviour, and re-solving does not repair it (`ST.five-4`). |
| Hidden width | Every W896 child with a second axis 0-3/15; dead below about 0.90 of read energy | Read energy is a ceiling, not a predictor: 0.9178 gives 0/15, 0.9193 gives 3/15. |
| Depth beyond one or two blocks | Excision ladder 13 → 7 → 3 → 0; every depth-sharing scheme collapsed; the v2 period-merge proxy read 0/9 | Pair cost depends on the distance between blocks, and that does not hold for three blocks. |
| Head merge (T7/M5) | Exact identity gate, but 8-13x more drift per parameter | One merged attention block costs more than two whole MLP lanes. The judgement audit later found both built sites alive at 4x. |
| Coupled attention Q-DEIM, CPCF, moving-gauge dense contraction | 37/37, 0 works, 0/15 | Closed paths. |
| A12 carrier replant | Cached fit improved, autoregressive coherence lost | The central law in item 3 of section 1. |
| MLP-contraction hypothesis | A 2x child given parent-width MLPs (layer-0 capacity defect cut about 5,000x) still read 0/15 | MLP width was not the limit. |
| K3 → dense, media into a shared trunk | The K3 dense pass was never decoded; media units fold MLP only; S7 outputs were worse than the noisy input | Not measured on behaviour. These are the biggest unknowns for any K3-based trunk. |

**Metrics that misled.** Teacher-forced top-1 ran *opposite* to free-run quality across four
stage-1 children. The best terminal reconstruction in the program (relL2 0.464) came with a
child that emits nothing on half the suite. Held-out local write cosine has rho 0.06 against
behaviour: it carries no information. Free-run greedy match (rho 0.685) is much weaker than
teacher-forced top-1 on the parent's own text (rho 0.890). Top-10 overlap (0.883) and
top-1-in-top-5 (0.898) rank children almost the same as top-1. Among living children top-n
leads by about 0.05 rho, inside the noise (`OBJECTIVE_RETENTION_ANALYSIS_2026-10-29.md` §0).
Lambada perplexity (0.83) and drift against a copy-exact control (0.81) are the best cheap
screens. Every metric drops to rho 0.4-0.7 inside the alive band.

---

## 6. How the work was judged, and what that means for the numbers

- **The frozen 15-prompt review is noisy and generous.** Sealed counts move by ±2 between
  reviewers. The robust re-read (a lane counts if any of three restatements is answered) took
  the parent from 12 to 6 and `ARCH.B75` from 11 to 2 (`MX33.F1`). Below about 0.4x stored, the
  15-case instrument measures echo, not language modelling (`STANDING_STATE_2026-08-29.md` §2).
- **No count on record is Cyril's.** Eleven rows are tagged `scorer='cyril'`. They carry his
  alive/dead rulings and reactions, and the two numeric values are machine counts he read. No
  child from September or October has been read by him.
- **Numbers decided too much, early.** Under Cyril's 2026-10-22 rule, no number passes or
  fails a fold; top-n matching is an optimizer target and a metric, not a gate. The judgement
  audit re-read 74 numerically dropped arms and found six families dropped wrongly: the
  excision site census (9 of 12 census-failed blocks are alive or partially alive), the June Qwen3-0.6B depth
  merges, single-site head merges, the 35/36 depth codebook, per-site MLP folds pruned on
  token match, and one embedding route.
- **Ratios were storage-format ratios.** Every matrix child in waves 5-13 is parent-shaped, so
  its "4x" is per component. Body contraction, not counted compaction, governs behaviour.

---

## 7. What the record supports next

This section is from `FOLDING_RECORD` §11 and `TECHNIQUE_SCOREBOARD` §10, restated briefly.

**Supported as fold initialisers for a small dense trunk**
- One block or layer removed at a reviewed site, not at a census-picked one.
- Per-component 4x folds (T8, T5, T1 on Q/K, T3/T4 on conv), one or two per child.
- A vocab row coreset; pick the hidden basis from the pooled reader moment's eigenvectors.
- MLP intermediate width before hidden width, at matched ratio.
- Full-parameter KD from the parents after the fold. This is training, and it did improve a
  folded child.

**Open, and each needs its own test before anyone relies on it**
1. Width ratios like 5120 → 2048. No fold has kept behaviour there, and the LFM2 cliff sits
   between 0.75 and 0.625 of width.
2. Depth reduction beyond one or two layers on any large parent.
3. Decoding the K3 dense pass and reviewing it. This is the most consequential unknown.
4. Mixer kind translation: KDA → trunk mixer, MLA → GQA, SiTU → SiLU.
5. A fold fitted to a top-n target. Designed as X2, never run.
6. Composition at the depth a parent-to-2B trunk needs: width, depth, attention and vocab
   folded together.
7. Scale transfer. Almost every reviewed result is on one 230M model.
8. Cyril's own reading of the best Sept-Oct children. None has been read.
9. Whether the folded trunk must pass the review on its own or is judged after distillation.
   Not settled.

**Recommendations**
- Keep folding as an initializer and run the open items as small, pre-registered tests on
  `omni_lab`, each with a behaviour read, rather than as ladders gated on numbers.
- Decode and review the K3 dense pass before building anything else on it.
- Put Cyril's read of three or four of the best children (`MXK14`, `ARCH.V19`, `ARCH.X13A`, the
  27B MLP-4x trunk) on record. Then the program has at least one owner-scored anchor.
- Use lambada perplexity and drift against a copy-exact control as screens; use teacher-forced
  top-1 or top-n on parent text for ranking. Never use local write cosine or free-run greedy
  match to drop an arm.

---

## 8. Where everything is (as pushed on 2026-11-06)

| Branch | What it holds |
|---|---|
| `main`, `codex/lrc-blank-compression` | The compiler, tests, experiments, docs, matrix campaign through wave 45, this review and the database export. |
| `claude/omni-campaign` | Chimera Omni campaign reports and tools, Sept → Nov (`reports/omni-campaign-2026-09/`): FOLDING_RECORD, TECHNIQUE_SCOREBOARD, audits, Run 0 plans, crash and drift reports. |
| `omni-lab` | The `omni_lab` database code: migrations, ingesters, runner, cleanup tooling. |
| `wip/neural-forge-uncommitted-20261106` | Neural Forge code, tests and run evidence that were never committed (mostly Aug 9-10). It is preserved as-is and has not been re-tested. |
| `backup/20261028/*` | Weight-stripped snapshots of every local branch and dirty worktree, taken 2026-10-28. |

**Not published:** model weights, tensors and caches; corpora and media outputs; the
cross-project directive log; machine inventory. Weights deleted in the 2026-10-21 cleanup can be
rebuilt from the recipes recorded in `model_deletion`.

## 9. Cleaning done for this publication

- The local main branch carried seven fold-child `model.safetensors` files in history (69-535
  MB each), committed in August and removed by `d71f4962`. The pushed history is the
  weight-stripped copy made on 2026-10-28. Its commits have the same messages, dates and
  authors as the local ones, but new ids. The old local ids are kept on the local-only branch
  `archive/prestrip-20261106/codex/lrc-blank-compression`.
- Every new or changed file was scanned for the 137 vault credentials and for key, token and
  password patterns. Hits: none. The only vault match anywhere is `lmstudio`, LM Studio's
  default placeholder key, used as a word in model paths.
- Everything already on GitHub (14,162 files) was scanned the same way. It holds no secrets.
- The files left out on purpose: `*.safetensors`, `*.pt` and other tensor files; files over
  10 MB; `.codex-test-runs/` scratch; `proof_outputs/` (about 1 GB of generated outputs);
  media, logs, `__pycache__`; the raw folding-record extraction shards (the database export
  carries their content); and the NAS/workstation cleanup plans.
