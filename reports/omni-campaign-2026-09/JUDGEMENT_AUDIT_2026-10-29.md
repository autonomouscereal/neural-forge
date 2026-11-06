# Judgement audit of numerically dropped fold arms, 2026-10-29

Claude, for the GOAL of 2026-10-27 ("[Cyril quote withheld]"). This applies Cyril's 2026-10-22 rule: "[Cyril quote withheld]". An arm dropped on a number counts as unreviewed until someone reads its outputs. This file is the reading.

Base record: `FOLDING_RECORD_2026-10-22.md` (omni_lab schema 7, batches 272-274). Paths: `NF` = `C:/Users/cereal/Documents/ai training/neural-forge`, `MX` = `NF/reports/matrix-campaign-20260810`, `JA` = `ai:/home/cereal/judgement-audit-20261029` (this audit's server work dir).

**How every arm here was judged.** I (Claude) read the generated text myself. Grading is artifact-tolerant (Cyril 2026-08-28): loops, missing EOS and cap rides never count against intelligence, and the question is whether correct content, retrieval, reasoning or on-topic structure survives. Each arm gets one of four classes: **alive** (retains intelligence; close to the parent on what the parent can do), **partially alive** (coherent, on-task language with some correct content, materially degraded), **dead** (no task content outside loops, echo or debris), or **cannot judge** (no outputs and no weights left, or the parent itself does not demonstrate the surface). No count decides a class. Where a machine count exists it is quoted as a diagnostic only.

## Summary

- **2,430 of 7,895 arms** were dropped (`fails` / `closed_path` / `superseded` / refused) with no human or agent behaviour read behind the verdict (section 1). **74 of them were re-judged on behaviour** for this audit, covering 52 distinct children (section 3).
- **Families that were wrongly dropped** (section 4). Each was killed on a number while the children's text shows retained intelligence:
  1. **Layer excision site census** (`MX14.F-EXCISE.*`): 9 of 12 census-failed LFM2 blocks are alive or partially alive when built.
  2. **June depth merges on Qwen3-0.6B** (19+20, 8+9+19+20; choice probe + PPL): rebuilt and decoded. One pair is alive at 2.64% shrink, two pairs partially alive at 5.28%.
  3. **Single-site attention head merge** (`MXQ.probe.*.T7.S4`, priced out on drift per parameter): both built sites (L02, L12) alive at 4x on the component.
  4. **Depth-sharing codebook at 35/36 on Qwen3-4B**: "prose collapses" is false. The July 1,000-item bank scores the unmodified parent 0/200 on two of its five prompt families.
  5. **Per-site MLP folds pruned on token match** (waves 1-3): the rescued C13.T8.S4.L05 is parent-level.
  6. **Embedding route** `CAL.C01.T5.X4-mxc` (token match 0.125 < 0.90): partially alive.
- **Correctly dead on behaviour** (section 5): hidden width below about 0.86x on LFM2; all-site MLP 4x on LFM2 and on Qwen3.5-0.8B (C2K896); every July Qwen3-4B blank target at 25-30x; extreme depth sharing; codebook at 34/36 and below; blocks 0, 2, 4 and 12; the 1.41B compiled child.
- **Cannot judge** (section 5): **the K3 dense pass**, never decoded and the most consequential gap for v2; C10/S5; every media fold (nothing was ever rendered); T7 at L04, L08 and L10 (never built); the exact-4x carrier children; the SmolLM2 merge (parent not demonstrated); every census-killed class never built.
- **The 22 held U3 blobs** (section 6): none is needed for behaviour judgement. All 22 (80,196,246,372 B) can be deleted under Cyril's 10-28 approval. Nothing was deleted here.
- **omni_lab** (section 7): one `scorer=claude` behaviour row per re-judged arm, through a foldrec shard, after a restore-tested dump.

---

## 1. The population: arms dropped on numbers or machine screens

**Query** (omni_lab, `omni_lab_ro`, 2026-10-29): `fold_arm` rows whose `verdict_class` is `fails`, `closed_path` or `superseded`, or whose `verdict_text` says "refus...", and that carry **no** `fold_behaviour` row by `claude`, `codex`, `cyril` or `mixed`. That leaves arms whose only behaviour rows are `machine`, `none` or NULL, or that have none at all.

**Result: 2,430 of 7,895 arms.**
- 1,062 have no behaviour row at all (diagnostics only, or nothing).
- 1,368 have machine/none/NULL rows only.
- Verdicts: fails 1,825, closed_path 327, superseded 187, and 91 others whose text says refused. 252 say "refus..." in the verdict text.
- 167 claim `child_exists_now = yes`. Section 2 says what is really on disk.

**By technique family.** The family is my classification from component kinds, technique names and program, because `technique.family` is "fold" on 5,802 of 5,805 rows. The `media` and `K3` families take precedence over the structural kinds.

| Family | Arms | No behaviour row | Machine-scored only | Child exists (DB) | Programs (arms) |
|---|--:|--:|--:|--:|---|
| layer (excision, merge, depth sharing) | 126 | 21 | 105 | 4 | neural-forge 42, matrix-campaign 37, early-surgery 17, cross-family 15, lfm2-dynamic 7 |
| hidden width | 249 | 43 | 206 | 12 | cross-width 176, matrix-campaign 22, lfm2-dynamic 20, early-surgery 11, neural-forge 10 |
| MLP width | 253 | 83 | 170 | 46 | neural-forge 59, matrix-campaign 57, lfm2-dynamic 37, density 33, cross-family 24 |
| residual-attention / heads | 313 | 132 | 181 | 13 | residual-attention 130, matrix-campaign 39, neural-forge 33, cpcf 27, lfm2-dynamic 22 |
| vocab / embedding / head | 114 | 36 | 78 | 2 | matrix-campaign 33, neural-forge 21, lfm2-dynamic 18, cross-family 15, sparse-semantic 14 |
| K3 (experts-to-dense, C10/C14) | 181 | 148 | 33 | 16 | k3 131, chimera-omni 21, cross-family 8, cross-width 8, media 4 |
| cross-family | 138 | 70 | 68 | 4 | cross-family 125, early-surgery 7, kernel 4 |
| whole-model contraction | 368 | 60 | 308 | 8 | early-surgery 246, chimera-omni 40, lfm2-dynamic 39, neural-forge 33 |
| media | 301 | 239 | 62 | 27 | media 274, chimera-omni 26 |
| other | 387 | 230 | 157 | 35 | matrix-campaign 103, neural-forge 83, lfm2-dynamic 63, chimera-omni 42 |
| **all** | **2,430** | **1,062** | **1,368** | **167** | |

By program: early-surgery 334, matrix-campaign 295, neural-forge 283, media 278, cross-family 209, cross-width 208, lfm2-dynamic 206, chimera-omni 166, k3 131, residual-attention 130, other 43, density 35, cpcf 27, sparse-semantic 22, stage1 21, chimera-omni-toy 18, kernel 13, stage2 11.

**What the drop rested on.** This is approximate: a first-match keyword rule over the verdict text, diagnostics and machine behaviour text, so read it as orientation, not as counts.

| Family | mechanical / refused before build | token match / free-run / TF | PPL / NLL / KL | choice / benchmark | fit, energy, relL2, cos gate | text note | n/15 machine count | none stated |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| layer | 19 | 4 | 56 | 4 | 22 | 1 | 18 | 2 |
| hidden width | 34 | 4 | 157 | 7 | 17 | 12 | 10 | 8 |
| MLP width | 40 | 17 | 64 | 8 | 59 | 21 | 23 | 21 |
| residual-attention | 76 | 35 | 23 | 6 | 55 | 63 | 14 | 41 |
| vocab | 21 | 21 | 28 | 1 | 17 | 4 | 9 | 13 |
| K3 | 62 | 44 | 1 | 0 | 37 | 1 | 1 | 35 |
| cross-family | 57 | 3 | 62 | 6 | 2 | 1 | 1 | 6 |
| whole-model | 71 | 4 | 247 | 4 | 5 | 9 | 5 | 23 |
| media | 95 | 0 | 3 | 2 | 147 | 1 | 10 | 43 |
| other | 135 | 15 | 26 | 9 | 37 | 24 | 33 | 108 |
| **all** | 610 | 147 | 667 | 47 | 398 | 137 | 124 | 300 |

The population includes duplicates. Many matrix-campaign excision and width arms appear twice: once under a summary key scored `machine` (e.g. `X07`, `LFM2.W896`), and once under their `ARCH.*` key, which Claude read in August. Those `ARCH.*` reads were sealed reviews at the time, so this audit re-read the arms whose drop rested on a census, a gate or a machine count without a matching read.

---

## 2. What still exists to judge behaviour

Every path below was checked on disk on 2026-10-29. "Exists (DB)" alone was not trusted.

| Family | Child weights that exist | Saved generations / review packets | Gaps |
|---|---|---|---|
| all LFM2 matrix-campaign families | **147 kept children** (standard Lfm2 checkpoints, 61 GB) under `/mnt/ai_data/neural-forge-matrix/wave14..wave43/children/`, the retention-manifest keep set. They include X06, X07, X09, X10, X11, X13A, P1301, P1303, T010713, W512, W704, W720, W736, W768, W896, MM768, ff320, ff640, ff1280, the CL ladder and the V-series. The other built children were deleted per `MX/wave*/children-retention-manifest-*.json`. | | the kept children can be re-decoded on CPU at any time; the dossiers already hold their full 16-case text |
| layer, LFM2 excision | kept: X06, X07, X09, X10, X11, X13A (wave16-17). Not kept: X00-X04, X08, X12 | full 16-case dossiers for every built single-block excision (`MX/wave16/human-review/READ-X*.txt`, `MX/wave17/human-review/dossier-X*.json`), pairs and triples | the census arms `MX14.F-EXCISE.L*` never built children of their own; the later `ARCH.X*` children are the same fold (pure excision, bit-identical rebuild `MX26.F9`) |
| layer, Qwen3-0.6B / SmolLM2 merges (June) | deleted; **rebuilt** for this audit with the same tool (`neural_forge.cli hf-merge-layers` @47145a30, CPU). Parameter counts match the record exactly: 580,318,976 = 2.64% shrink, 564,588,032 = 5.28%, SmolLM2 130,974,912 = 2.63% | JA dossiers (this audit) | none |
| layer, Qwen3-4B depth codebook / depth sharing (July) | 4 held U3 blobs: pn34_pa36, pn35_pa36, pn36_pa34, pn36_pa35 (compact custom format; needs `hf_qwen_codebook_model`) | `candidate_eval.json` with all 1,000 behaviour continuations + generations for p28, p32, p34/34, p34/36, p35/35, p35/36, p36/34, p36/35, p36/36 (control), tieddepth122m, depth-basis allrank1, basis-134M | none for judgement |
| hidden width (LFM2 W-ladder, MM768) | kept: W512, W704, W720, W736, W768, W896, MM768, MM896 and the W768/W896 variants. Not kept: W640 | full dossiers waves 19-22 | none |
| hidden width / cross-width (Qwen3-4B → 131M/134M/1.13B blank targets, July) | 9 children on NAS (6 x 131-134M, 3 x 1.13B), standard HF Qwen3/Llama | `candidate_eval.json` for each | none |
| MLP width (LFM2 per-site and all-site) | kept: ff320, ff640, ff1280 (wave 14), CL640-CL1792 ladder children. Per-site cells: not kept | dossiers (`MX/cells/*/frozen-review-dossier.json`, wave 14 ff640/ff1280) | many per-site cells read only in composites |
| MLP width, Qwen3.5-0.8B C2K896 | `/mnt/lexar/omni-campaign/research/fold/x3-08b/children/QF08-C2K896` + KD/OPD rungs | `x3-08b/eval/*/bank_v1_answers.md` (the same 16 prompts, verbatim) | none |
| residual-attention / head merge (LFM2) | none | `MX/wave13/human-review/COMP.MA.MXQ1.md`, `MXQ2.md` (full text) | T7 probes at L04, L08, L10 were never decoded |
| residual-attention (Qwen3-4B July: head isolation, p35 QKVO, 1.41B exact target, guided GUD) | only the 1.41B compiled child `compiled_child_cache/bdf2309d...` (held U3 blob, never measured) | run JSON with first-action / rollout numbers; no text for most | children deleted; text not saved |
| vocab / embedding route | none | `MX/cells/CAL.C01.T5.X4-mxc/mxc2-responses.json` (16 cases), C01.T4 text not found | C01.T4 svd embedding children: only the verdict sentence remains |
| exact-4x LFM2 (stage 1, moving gauge) | 4 published children under `model-surgery-lab/artifacts/neural_forge_exact4x_20260730` and `lfm2_exact4x_production_20260728` | receipts carry only `route_smoke_accepted: true`, no decoded text | needs `lfm2_carrier_review_cli` (vocab 4,096 + semantic carrier routes) |
| K3 dense pass | `/home/cereal/omni-campaign/k3-run-oct09-claude/fold` (96 GB, 127 files), `/mnt/lexar/omni-campaign/k3-continuation-oct08-claude` (77 GB), `/home/cereal/lexar-relief-local/omni-campaign/k3-fold-sep07-codex` (12 GB) | none: never decoded | no decode path; see section 5 |
| C10 / S5 (K3 union in the 27B trunk) | `/home/cereal/omni-campaign/composite/c10/S5-C2K4352K3U/child` (51 GB) | R-Q top-1 numbers only | the 27B line is retired (Cyril 10-22) |
| media (C15/C16/C17, S7 units, QIMG B30 refold) | S7 composite 35 GB, QIMG first32 native executor 123 GB (tensor captures) | no rendered image, video or audio was found for any refolded block or S7 unit | nothing to look at |

---

## 3. Re-judged arms

**74 arms re-judged on behaviour** (several are the same child under two or three keys from different shards). The tally is 14 alive, 15 partially alive, 42 dead and 3 cannot judge. They cover 52 distinct children: 13 single-block excisions, 2 head merges, 7 width rungs, 3 MLP children, 1 embedding route, 6 depth codebooks, 3 depth-sharing controls, 10 blank targets, C2K896, 4 June merges, the SmolLM2 merge and the 1.41B compiled child. Of these, 45 were judged on text that already existed, and 7 were decoded fresh on CPU for this audit: 4 rebuilt June merges, the SmolLM2 merge, the 1.41B compiled child and the 134M head-norm child. The head-norm child also has saved text. Same-frame parent controls were decoded for Qwen3-0.6B-Base and SmolLM2-135M. Twenty-nine of the 74 arms are alive or partially alive, and all 29 are recorded as `fails` in omni_lab. Of the other 45, 44 are `fails` and one (the 1.41B compiled child) is `unmeasured`.

Evidence is one of three kinds:
- the frozen 16-case dossier already on disk (the matrix campaign, the read the record did not attach to the dropped arm);
- the saved July `candidate_eval.json` continuations;
- a fresh CPU decode for this audit (`JA/dossiers/*.json`).

"What the text shows" is my read. Machine counts are not repeated here: they are in the recorded verdict column and in the record.

| # | Arm (`fold_arm.arm_key`) | Family | Program | Recorded verdict and its basis | Re-judged | What the text shows | Evidence read |
|--:|---|---|---|---|---|---|---|
| 1 | `MX14.F-EXCISE.L01` | layer: excision | matrix-campaign | fails: held-out absorbed fraction -1.2103 against pre-registered 0.50 floor - negative: re-solved upstream writer worse than parent's | **partially alive** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X01) reads: coherent on-task language; retrieval 4:15 p.m., Supplier B kept; seasons/logic wrong | `MX/wave16/human-review/READ-X01.txt` |
| 2 | `MX14.F-EXCISE.L02` | layer: excision | matrix-campaign | fails: held-out absorbed fraction -0.6331 against pre-registered 0.50 floor - negative: re-solved upstream writer worse than parent's | **dead** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X02) reads: prompt echo on nearly every case; only a copied summary | `MX/wave17/human-review/dossier-X02.json` |
| 3 | `MX14.F-EXCISE.L03` | layer: excision | matrix-campaign | fails: held-out absorbed fraction -0.0909 against pre-registered 0.50 floor - negative: re-solved upstream writer worse than parent's | **partially alive** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X03) reads: coherent; 102 kits, two-sentence summary, asks for missing sales data; retrieval wrong | `MX/wave16/human-review/READ-X03.txt` |
| 4 | `MX14.F-EXCISE.L04` | layer: excision | matrix-campaign | fails: held-out absorbed fraction +0.0345 against pre-registered 0.50 floor | **dead** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X04) reads: refusals and echo loops; one copied summary | `MX/wave17/human-review/dossier-X04.json` |
| 5 | `MX14.F-EXCISE.L06` | layer: excision | matrix-campaign | fails: held-out absorbed fraction +0.0342 against pre-registered 0.50 floor | **alive** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X06) reads: coherent answers on every case; Saturday retrieval, valid dedupe, six-step plan; arithmetic wrong | `MX/wave17/human-review/dossier-X06.json` |
| 6 | `MX14.F-EXCISE.L07` | layer: excision | matrix-campaign | fails: held-out absorbed fraction -0.0015 against pre-registered 0.50 floor - negative: re-solved upstream writer worse than parent's | **alive** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X07) reads: parent-like: 102 kits, retrieval 4:15 p.m., Supplier B + one-week risk, valid seen-set code skeleton | `MX/wave16/human-review/READ-X07.txt` |
| 7 | `MX14.F-EXCISE.L08` | layer: excision | matrix-campaign | fails: held-out absorbed fraction +0.1360 against pre-registered 0.50 floor | **partially alive** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X08) reads: coherent; Supplier B, code; retrieval and logic wrong | `MX/wave17/human-review/dossier-X08.json` |
| 8 | `MX14.F-EXCISE.L09` | layer: excision | matrix-campaign | fails: held-out absorbed fraction +0.0269 against pre-registered 0.50 floor | **alive** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X09) reads: Saturday / 4:15 p.m. retrieval, asks for the missing sales evidence, coherent scene and email | `MX/wave16/human-review/READ-X09.txt` |
| 9 | `MX14.F-EXCISE.L10` | layer: excision | matrix-campaign | fails: held-out absorbed fraction +0.0788 against pre-registered 0.50 floor | **partially alive** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X10) reads: fluent and on-topic on every case but many facts wrong | `MX/wave17/human-review/dossier-X10.json` |
| 10 | `MX14.F-EXCISE.L11` | layer: excision | matrix-campaign | fails: held-out absorbed fraction +0.0959 against pre-registered 0.50 floor | **alive** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X11) reads: 4:15 p.m., Supplier B, valid dedupe, six-step plan; seasons answer wrong | `MX/wave16/human-review/READ-X11.txt` |
| 11 | `MX14.F-EXCISE.L12` | layer: excision | matrix-campaign | fails: held-out absorbed fraction +0.3199 against pre-registered 0.50 floor | **dead** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X12) reads: script drift and tokenizer debris from the first token | `MX/wave16/human-review/READ-X12.txt` |
| 12 | `MX14.F-EXCISE.L13` | layer: excision | matrix-campaign | fails: held-out absorbed fraction +0.4669 against pre-registered 0.50 floor | **alive** | census dropped this block on absorbed fraction; the built single-block excision child (ARCH.X13A) reads: near parent: Saturday / 4:15 p.m., correct arithmetic steps, ordering | `MX/wave16/human-review/READ-X13A.txt` |
| 13 | `X00` | layer: excision | matrix-campaign | fails: Complete 14-block excision map: ... attn L06 8/15, L08/L10 7/15, L05/L03 6/15, L04 2/15, L00-conv 0/15 (most expensive block in the model), L02/L12-at | **dead** | agrees with the machine count: token cycling from the first token | `MX/wave17/human-review/dossier-X00.json` |
| 14 | `X02` | layer: excision | matrix-campaign | fails: Complete 14-block excision map: ... attn L06 8/15, L08/L10 7/15, L05/L03 6/15, L04 2/15, L00-conv 0/15 (most expensive block in the model), L02/L12-at | **dead** | agrees with the machine count: prompt echo; only a copied summary | `MX/wave17/human-review/dossier-X02.json` |
| 15 | `X04` | layer: excision | matrix-campaign | fails: Complete 14-block excision map: ... attn L06 8/15, L08/L10 7/15, L05/L03 6/15, L04 2/15, L00-conv 0/15 (most expensive block in the model), L02/L12-at | **dead** | agrees with the machine count: refusals and echo loops | `MX/wave17/human-review/dossier-X04.json` |
| 16 | `X12` | layer: excision | matrix-campaign | fails: the smallest bite (L12 attention, 4.6M) is the only 0/15 | **dead** | agrees with the machine count: script drift / debris | `MX/wave16/human-review/READ-X12.txt` |
| 17 | `docs-june-july-early:q06-depth-average-19-20` | layer: merge | early-surgery | fails: Failed | **alive** | rebuilt (2.64%): anchored debugging fix values[start:] (parent misses it), seasons + misconception, summary, email, natural greeting; arithmetic (120) and ordering wrong; June drop was a choice probe + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-avg-19+20-fs.json` |
| 18 | `docs-june-july-early:q06-depth-average-8-9_19-20` | layer: merge | early-surgery | fails: Failed | **partially alive** | rebuilt (5.28%): seasons + misconception, summary, Friday/Monday/tracking email, six-step plan naming backup/DNS/TLS; arithmetic, ordering, uncertainty (invents a cause) and snack revision lost; June drop was choice 0.25 + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-avg-8+9_19+20-fs.json` |
| 19 | `docs-june-july-early:q06-depth-fold-19-20` | layer: merge | early-surgery | fails: Failed | **alive** | rebuilt (580,318,976 params, 2.64%): seasons with the distance misconception rejected, retrieval Saturday/4:15 p.m./north gate, Supplier B + one-week slip, Friday/Monday/tracking email, asks for range/units/sample/prices; arithmetic partial (102 then 99), ordering wrong; June drop was a 4-item choice probe 0.50 + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-fold-19+20-fs.json` |
| 20 | `docs-june-july-early:q06-depth-fold-8-9_19-20` | layer: merge | early-surgery | fails: Failed | **partially alive** | rebuilt (564,588,032 params, 5.28%): coherent; summary with decision and risk, three actions, axial tilt, dedupe description; arithmetic, ordering, debugging collapse; refuses the email; June drop was choice 0.25 + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-fold-8+9_19+20-fs.json` |
| 21 | `docs-june-july-early:smollm135m-depthfold-1314` | layer: merge | early-surgery | fails: failed; repeated sentence/code patterns | **cannot judge** | rebuilt (130,974,912 params, 2.63%) and decoded, but the SmolLM2-135M base parent (smol-parent-fs.json) copies the exemplar or echoes on almost every case even in the few-shot frame, so the surface cannot separate parent and child. | `JA/dossiers/smol-fold-13+14-fs.json` |
| 22 | `pg-may-jun:nf-0626-qwen3-mergefirst-depth-average-19+20` | layer: merge | neural-forge | fails: Failed | **alive** | rebuilt (2.64%): anchored debugging fix values[start:] (parent misses it), seasons + misconception, summary, email, natural greeting; arithmetic (120) and ordering wrong; June drop was a choice probe + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-avg-19+20-fs.json` |
| 23 | `pg-may-jun:nf-0626-qwen3-mergefirst-depth-average-8+9-19+20` | layer: merge | neural-forge | fails: Failed | **partially alive** | rebuilt (5.28%): seasons + misconception, summary, Friday/Monday/tracking email, six-step plan naming backup/DNS/TLS; arithmetic, ordering, uncertainty (invents a cause) and snack revision lost; June drop was choice 0.25 + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-avg-8+9_19+20-fs.json` |
| 24 | `pg-may-jun:nf-0626-qwen3-mergefirst-depth-fold-19+20` | layer: merge | neural-forge | fails: Failed | **alive** | rebuilt (580,318,976 params, 2.64%): seasons with the distance misconception rejected, retrieval Saturday/4:15 p.m./north gate, Supplier B + one-week slip, Friday/Monday/tracking email, asks for range/units/sample/prices; arithmetic partial (102 then 99), ordering wrong; June drop was a 4-item choice probe 0.50 + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-fold-19+20-fs.json` |
| 25 | `pg-may-jun:nf-0626-qwen3-mergefirst-depth-fold-8+9-19+20` | layer: merge | neural-forge | fails: Failed | **partially alive** | rebuilt (564,588,032 params, 5.28%): coherent; summary with decision and risk, three actions, axial tilt, dedupe description; arithmetic, ordering, debugging collapse; refuses the email; June drop was choice 0.25 + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-fold-8+9_19+20-fs.json` |
| 26 | `pg-may-jun:nf-0627-smollm-fold1314-folded-13+14-raw` | layer: merge | neural-forge | fails: failed gate | **cannot judge** | rebuilt (130,974,912 params, 2.63%) and decoded, but the SmolLM2-135M base parent (smol-parent-fs.json) copies the exemplar or echoes on almost every case even in the few-shot frame, so the surface cannot separate parent and child. | `JA/dossiers/smol-fold-13+14-fs.json` |
| 27 | `server-matrix-runs:nf/archive/qwen3_layer_merge_20260626#average_19-20` | layer: merge | early-surgery | fails: failure/refusal/gate-fail recorded in status fields; average_19-20_gate.cmd.json:passed=False; average_19-20_gate.json:passed=False | **alive** | rebuilt (2.64%): anchored debugging fix values[start:] (parent misses it), seasons + misconception, summary, email, natural greeting; arithmetic (120) and ordering wrong; June drop was a choice probe + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-avg-19+20-fs.json` |
| 28 | `server-matrix-runs:nf/archive/qwen3_layer_merge_20260626#average_8-9_19-20` | layer: merge | early-surgery | fails: failure/refusal/gate-fail recorded in status fields; average_8-9_19-20_gate.cmd.json:passed=False; average_8-9_19-20_gate.json:passed=False | **partially alive** | rebuilt (5.28%): seasons + misconception, summary, Friday/Monday/tracking email, six-step plan naming backup/DNS/TLS; arithmetic, ordering, uncertainty (invents a cause) and snack revision lost; June drop was choice 0.25 + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-avg-8+9_19+20-fs.json` |
| 29 | `server-matrix-runs:nf/archive/qwen3_layer_merge_20260626#fold_19-20` | layer: merge | early-surgery | fails: failure/refusal/gate-fail recorded in status fields; fold_19-20_gate.cmd.json:passed=False; fold_19-20_gate.json:passed=False | **alive** | rebuilt (580,318,976 params, 2.64%): seasons with the distance misconception rejected, retrieval Saturday/4:15 p.m./north gate, Supplier B + one-week slip, Friday/Monday/tracking email, asks for range/units/sample/prices; arithmetic partial (102 then 99), ordering wrong; June drop was a 4-item choice probe 0.50 + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-fold-19+20-fs.json` |
| 30 | `server-matrix-runs:nf/archive/qwen3_layer_merge_20260626#fold_8-9_19-20` | layer: merge | early-surgery | fails: failure/refusal/gate-fail recorded in status fields; fold_8-9_19-20_gate.cmd.json:passed=False; fold_8-9_19-20_gate.json:passed=False | **partially alive** | rebuilt (564,588,032 params, 5.28%): coherent; summary with decision and risk, three actions, axial tilt, dedupe description; arithmetic, ordering, debugging collapse; refuses the email; June drop was choice 0.25 + PPL. Same-frame Qwen3-0.6B-Base parent: q06-parent-fs.json. | `JA/dossiers/q06-fold-8+9_19+20-fs.json` |
| 31 | `server-matrix-runs:nf/archive/smollm135m_depth_fold_reconstruct_20260627_212259#depth_fold_13_14` | layer: merge | cross-family | fails: choice 1/4 vs SmolLM2-135M baseline 2/4 and perplexity_mean 399.3371660822725 vs 347.405316388356 on the run's 4-item choice probe; gate passed=False  | **cannot judge** | rebuilt (130,974,912 params, 2.63%) and decoded, but the SmolLM2-135M base parent (smol-parent-fs.json) copies the exemplar or echoes on almost every case even in the few-shot frame, so the surface cannot separate parent and child. | `JA/dossiers/smol-fold-13+14-fs.json` |
| 32 | `docs-residual-attention:cat-codebook-uniform-34` | layer: depth sharing | neural-forge | fails: Global uniform allocation crosses a cliff. Never rerun unchanged. | **dead** | punctuation / script streams or empty answers on every probe | `NAS nf/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn34_pa34_gpu6_r1_20260710/candidate_eval.json` |
| 33 | `docs-residual-attention:cat-codebook-uniform-35` | layer: depth sharing | neural-forge | fails: Some code/math survives, but prose, model-surgery, and much math collapse. Keep as the sharp failure boundary. | **partially alive** | pn35_pa35: 'prose, model-surgery collapse' is false on the text: correct commonsense, tokenizer and neuron-merging answers (scored 0/200 by a keyword rule), correct code; arithmetic degraded (35+27=71) | `NAS nf/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn35_pa35_gpu4_r1_20260710/candidate_eval.json (re-read; child deleted, evidence kept)` |
| 34 | `qwen3_4b_codebook_pn28_pa28_gpu0_r1_20260710` | layer: depth sharing | neural-forge | fails: Deeper uniform merges remain collapsed. | **dead** | punctuation / script streams or empty answers on every probe | `NAS nf/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn28_pa28_gpu0_r1_20260710/candidate_eval.json` |
| 35 | `qwen3_4b_codebook_pn32_pa32_gpu4_r1_20260710` | layer: depth sharing | neural-forge | fails: Uniformly removing the same depth capacity from every feature compounds destructively even though each replacement is a real, activation-aligned paren | **dead** | punctuation / script streams or empty answers on every probe | `NAS nf/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn32_pa32_gpu4_r1_20260710/candidate_eval.json` |
| 36 | `qwen3_4b_codebook_pn34_pa34_gpu6_r1_20260710` | layer: depth sharing | neural-forge | fails: Removing two uniform depth variants per feature (34/34) crosses a sharp failure boundary: behavior and generation both reach zero. | **dead** | punctuation / script streams or empty answers on every probe | `NAS nf/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn34_pa34_gpu6_r1_20260710/candidate_eval.json` |
| 37 | `qwen3_4b_codebook_pn34_pa36_20260710` | layer: depth sharing | neural-forge | fails: MLP capacity crosses a catastrophic boundary between p35 and p34; the run is not an in-progress result and must not be queued again unchanged. | **dead** | punctuation / script streams or empty answers on every probe | `NAS nf/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn34_pa36_gpu6_r1_20260710/candidate_eval.json` |
| 38 | `qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710` | layer: depth sharing | neural-forge | fails: preserves some exact code and simple math but collapses too often to repeated digits or incorrect arithmetic. It is not a successful fold. | **partially alive** | coherent correct prose, definitions and code; arithmetic off by small amounts; 'not a successful fold' rested on the machine bank | `NAS nf/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710/candidate_eval.json` |
| 39 | `qwen3_4b_codebook_uniform_p32` | layer: depth sharing | neural-forge | fails: collapsed with multilingual repetition | **dead** | punctuation / script streams or empty answers on every probe | `NAS nf/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn32_pa32_gpu4_r1_20260710/candidate_eval.json` |
| 40 | `qwen3_4b_codebook_uniform_p34` | layer: depth sharing | neural-forge | fails: Sharp global capacity cliff | **dead** | punctuation / script streams or empty answers on every probe | `NAS nf/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn34_pa34_gpu6_r1_20260710/candidate_eval.json` |
| 41 | `qwen3_4b_to_tieddepth122m_a18_m18_gpu4_r1_20260710` | layer: depth sharing | neural-forge | fails: Repeated average operators cannot replace ordered residual steps. Closed. | **dead** | word salad | `NAS nf/tied_depth_controls_20260710/qwen3_4b_to_tieddepth122m_a18_m18_gpu4_r1_20260710/candidate_eval.json` |
| 42 | `server-matrix-runs:nf/depth_basis_controls_20260710/qwen3_4b_depth_basis_allrank1_fullvocab_gpu6_r1_20260710` | layer: depth sharing | early-surgery | fails: child behavior 0/1000, generation 0/1000, choice 28/250 at 8.207812884492407x | **dead** | multilingual token salad | `NAS nf/depth_basis_controls_20260710/qwen3_4b_depth_basis_allrank1_fullvocab_gpu6_r1_20260710/candidate_eval.json` |
| 43 | `server-matrix-runs:nf/qwen_basis_checkpoint_20260710/qwen3_4b_to_basis134443364_fulltopology_gpu0_r1_20260710` | layer: depth sharing | early-surgery | fails: child behavior 0/1000, generation 0/1000, choice 9/250 at 29.91942462850007x | **dead** | empty or symbol salad | `NAS nf/qwen_basis_checkpoint_20260710/qwen3_4b_to_basis134443364_fulltopology_gpu0_r1_20260710/candidate_eval.json` |
| 44 | `MXQ.probe.L02.T7.S4` | residual-attention: head merge | matrix-campaign | fails: marginal drift +0.110941; 0.046379 drift per M-param removed = 8.0x the cheapest SwiGLU lane (L13.T5.S4 0.005783) | **alive** | same component as COMP.MA.MXQ2 (C10.T7.S4.L02): 102 -> 94 -> 18 per box before a muddled finish, correct retrieval, coherent analogy; machine said dead per child | `MX/wave13/human-review/COMP.MA.MXQ2.md` |
| 45 | `MXQ.probe.L12.T7.S4` | residual-attention: head merge | matrix-campaign | fails: marginal drift +0.157983; 0.066045 drift per M-param removed = 11.4x the cheapest SwiGLU lane (L13.T5.S4 0.005783) | **alive** | same component as COMP.MA.MXQ1 (C10.T7.S4.L12): Mira/Theo/Jun/Sal correct, retrieval fully correct (Saturday, 4:15 p.m., North Gate); machine said dead per child | `MX/wave13/human-review/COMP.MA.MXQ1.md` |
| 46 | `LFM2.W768` | hidden width | matrix-campaign | fails:  | **dead** | machine 3/15 is generous: one copied summary and fragments | `MX/wave19/human-review/dossier-W768.json` |
| 47 | `LFM2.W896` | hidden width | matrix-campaign | fails: hidden width is the killer | **partially alive** | weak but on-topic: Supplier B, a domain-move plan, a dedupe description | `MX/wave19/human-review/dossier-W896.json` |
| 48 | `W512` | hidden width | matrix-campaign | fails: W896 4/15 @ 0.859x, W768 3/15 @ 0.7226x, W640/W512 0/15 — W512 = 0.4635x, first child under half the parent (dead) | **dead** | empty output / quote cycling | `MX/wave19/human-review/dossier-W512.json` |
| 49 | `W640` | hidden width | matrix-campaign | fails: W896 4/15 @ 0.859x, W768 3/15 @ 0.7226x, W640/W512 0/15 — W512 = 0.4635x, first child under half the parent (dead) | **dead** | empty output on 15 of 16 cases | `MX/wave19/human-review/dossier-W640.json` |
| 50 | `W704` | hidden width | matrix-campaign | fails: 704 dead (cliff 64 wide: 0.8953 dead / 0.9193 alive) | **dead** | empty or one-word output | `MX/wave20/human-review/dossier-W704.json` |
| 51 | `W720` | hidden width | matrix-campaign | fails: slope wall-to-wall 4-3-1-0-0 | **dead** | empty output and dash cycling | `MX/wave22/human-review/dossier-W720.json` |
| 52 | `W736` | hidden width | matrix-campaign | fails: 736 0.9075->1/15 ... W736 within 0.0008 net, dominated | **dead** | fragments and quote cycling; no task content | `MX/wave21/human-review/dossier-W736.json` |
| 53 | `ARCH.ff640` | MLP width | matrix-campaign | fails: config-standard MLP fold = all-14-sites simultaneous, dies at 4x (ff640 0.6405x 0/15 | **dead** | all 14 sites at 4x: grammatical fragments with no task content | `MX/wave14/human-review/dossier-MX14-ff640.json` |
| 54 | `LFM2.ff1280-all-sites` | MLP width | matrix-campaign | fails:  | **partially alive** | all 14 SwiGLU sites at 2x per site (sub-4x): degraded but coherent, asks for the missing sales data, partial email | `MX/wave14/human-review/dossier-MX14-ff1280.json` |
| 55 | `MX14.fourteen-swiglu-coresets` | MLP width | matrix-campaign | fails: UNBUILT_INVENTORY fate: BUILT | **dead** | the composed ff640 child (same as ARCH.ff640) | `MX/wave14/human-review/dossier-MX14-ff640.json` |
| 56 | `QF08-C2K896` | MLP width | density | fails: content 0/162 (artifact_miss 158, ungraded 4) | **dead** | echo and <think> loops on 15 of 16; the only correct content is the Mira/Theo/Jun/Sal order inside a loop | `ai:/mnt/lexar/omni-campaign/research/fold/x3-08b/eval/c2k896/bank_v1_answers.md` |
| 57 | `X3-08B.c2k896` | MLP width | density | fails: fold alone 0/162 (158 artifact misses, 4 ungraded); 0/64 five-token prefixes | **dead** | echo and <think> loops on 15 of 16; the only correct content is the Mira/Theo/Jun/Sal order inside a loop | `ai:/mnt/lexar/omni-campaign/research/fold/x3-08b/eval/c2k896/bank_v1_answers.md` |
| 58 | `X3.08b.c2k896` | MLP width | density | fails: GRADED c2k896 content_pass 0/162 vs parent 54/162; Codex: 'X3 baseline0/162 versus parent54/162 is a measured warning, not yet a winning retention rec | **dead** | echo and <think> loops on 15 of 16; the only correct content is the Mira/Theo/Jun/Sal order inside a loop | `ai:/mnt/lexar/omni-campaign/research/fold/x3-08b/eval/c2k896/bank_v1_answers.md` |
| 59 | `mem-other:mx-w12-mlp-all` | MLP width | matrix-campaign | fails: MLP -> UNSOLVED WALL (all techniques 0.31-0.56 ...). [Per FBG 41-44 these MLP folds were pruned on token-match without behaviour review = UNREVIEWED p | **alive** | the MLP cells pruned on token match 0.31-0.59 were never read; the rescued C13.T8.S4.L05 (mlp_down 4x) reads parent-level on all 16 cases | `MX/cells/C13.T8.S4.L05/frozen-review-dossier.json` |
| 60 | `x3-08b-c2k896` | MLP width | density | fails: Content /162: parent 54, c2k896 0 | **dead** | echo and <think> loops on 15 of 16; the only correct content is the Mira/Theo/Jun/Sal order inside a loop | `ai:/mnt/lexar/omni-campaign/research/fold/x3-08b/eval/c2k896/bank_v1_answers.md` |
| 61 | `x3-08b.c2k896` | MLP width | density | fails: fold-only 4x child: content 0/162 (158 artifact misses, 4 ungraded) | **dead** | echo and <think> loops on 15 of 16; the only correct content is the Mira/Theo/Jun/Sal order inside a loop | `ai:/mnt/lexar/omni-campaign/research/fold/x3-08b/eval/c2k896/bank_v1_answers.md` |
| 62 | `x3-08b:c2k896` | MLP width | density | fails: X3 baseline0/162 versus parent54/162 is a measured warning, not yet a winning retention recipe | **dead** | echo and <think> loops on 15 of 16; the only correct content is the Mira/Theo/Jun/Sal order inside a loop | `ai:/mnt/lexar/omni-campaign/research/fold/x3-08b/eval/c2k896/bank_v1_answers.md` |
| 63 | `CAL.C01.T5.X4-mxc` | vocab / embedding | matrix-campaign | fails: FAIL — campaign STILL STOPS — the route now runs to completion and publishes a gradeable dossier, but reproduces only 2/16 of the sealed A9 responses  | **partially alive** | dropped on token_match 0.125 < 0.90; text is coherent and on task (Saturday / 4:15 p.m., Supplier B + one-week risk) with bridge debris ('Saturd', 'Suppl') | `MX/cells/CAL.C01.T5.X4-mxc/mxc2-responses.json` |
| 64 | `server-matrix-runs:nf/compiled_child_cache/bdf2309d3298527f30a0c35c379ec6e2eeb95a13a39a34034b37601637539dfb` | blank-target width/cross-family | neural-forge | unmeasured: compiler_report.json:capability_status=unproven | **dead** | 1.41B compiled child of Qwen3-4B, never measured before: numeral cycling ('Five äº”ç™¾ ...') from the first token on both decoded cases (decode stopped after 2 of 16 to free CPU). | `JA/dossiers/compiled-1p41b-bdf2309d-fs.partial.json` |
| 65 | `server-matrix-runs:nf/cross_family_blank_20260710/qwen3_4b_to_blank_llama131m_delta0p1_gpu0_r1_20260710` | blank-target width/cross-family | cross-family | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.368; log tail: Loading weights: 0%/ / 0/218 [00:00<?, ?it/s] / Lo | **dead** | single-letter cycling | `NAS nf/cross_family_blank_20260710/qwen3_4b_to_blank_llama131m_delta0p1_gpu0_r1_20260710/candidate_eval.json` |
| 66 | `server-matrix-runs:nf/cross_family_blank_20260710/qwen3_4b_to_blank_llama131m_delta0p5_gpu4_r1_20260710` | blank-target width/cross-family | cross-family | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.132; log tail: Loading weights: 0%/ / 0/218 [00:00<?, ?it/s] / Lo | **dead** | 'T V Variable' cycling | `NAS nf/cross_family_blank_20260710/qwen3_4b_to_blank_llama131m_delta0p5_gpu4_r1_20260710/candidate_eval.json` |
| 67 | `server-matrix-runs:nf/cross_family_blank_20260710/qwen3_4b_to_blank_qwen134m_headnorm_delta0p1_gpu4_r2_20260710` | blank-target width/cross-family | cross-family | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.096; log tail: Loading weights: 0%/ / 0/266 [00:00<?, ?it/s] / Lo | **dead** | 'and and the' cycling; also read on the frozen 16 (template frame): no task content | `NAS nf/cross_family_blank_20260710/qwen3_4b_to_blank_qwen134m_headnorm_delta0p1_gpu4_r2_20260710/candidate_eval.json` |
| 68 | `server-matrix-runs:nf/cross_family_blank_20260710/qwen3_4b_to_blank_qwen134m_jointvocab_sourcel2_delta0p1_gpu4_r3_20260710` | blank-target width/cross-family | cross-family | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.384; log tail: Loading weights: 0%/ / 0/266 [00:00<?, ?it/s] / Lo | **dead** | 'Slee hog' cycling | `NAS nf/cross_family_blank_20260710/qwen3_4b_to_blank_qwen134m_jointvocab_sourcel2_delta0p1_gpu4_r3_20260710/candidate_eval.json` |
| 69 | `server-matrix-runs:nf/cross_token_real_medoid_20260710/qwen3_4b_to_blank_qwen134m_v50982_dual_realmedoid_gpu0_r2_20260710` | blank-target width/cross-family | cross-width | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.092; log tail: Loading weights: 42%/████▏ / 89/211 [00:00<00:00,  | **dead** | token salad | `NAS nf/cross_token_real_medoid_20260710/qwen3_4b_to_blank_qwen134m_v50982_dual_realmedoid_gpu0_r2_20260710/candidate_eval.json` |
| 70 | `server-matrix-runs:nf/progressive_closed_form_20260710/qwen3_4b_to_blank_qwen1p13b_stage1_headnorm_delta0p5_gpu6_r3_20260710` | blank-target width/cross-family | cross-width | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.116; log tail: Loading weights: 0%/ / 0/332 [00:00<?, ?it/s] / Lo | **dead** | 'The The' / digit cycling | `NAS nf/progressive_closed_form_20260710/qwen3_4b_to_blank_qwen1p13b_stage1_headnorm_delta0p5_gpu6_r3_20260710/candidate_eval.json` |
| 71 | `server-matrix-runs:nf/real_medoid_dual_20260710/qwen3_4b_to_blank_qwen1p13b_dual_realmedoid_gpu4_r2_20260710` | blank-target width/cross-family | cross-width | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.34; log tail: Loading weights: 0%/ / 0/332 [00:00<?, ?it/s] / Loa | **dead** | empty or CJK/emoji debris | `NAS nf/real_medoid_dual_20260710/qwen3_4b_to_blank_qwen1p13b_dual_realmedoid_gpu4_r2_20260710/candidate_eval.json` |
| 72 | `server-matrix-runs:nf/same_token_tokenrows_outputgate_20260710/qwen3_4b_to_blank_qwen1p13b_dual_realmedoid_tokenrows_outputgate_gpu4_r1_20260710` | blank-target width/cross-family | cross-width | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.352; log tail: Loading weights: 0%/ / 0/332 [00:00<?, ?it/s] / Lo | **dead** | 'chairs cutting' salad | `NAS nf/same_token_tokenrows_outputgate_20260710/qwen3_4b_to_blank_qwen1p13b_dual_realmedoid_tokenrows_outputgate_gpu4_r1_20260710/candidate_eval.json` |
| 73 | `server-matrix-runs:nf/trajectory_subset_targetrows_20260710/qwen3_4b_to_blank_qwen134m_targetrows_subset_dual_gpu0_r1_20260710` | blank-target width/cross-family | cross-width | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.348; log tail: Loading weights: 0%/ / 0/211 [00:00<?, ?it/s] / Lo | **dead** | single-word cycling | `NAS nf/trajectory_subset_targetrows_20260710/qwen3_4b_to_blank_qwen134m_targetrows_subset_dual_gpu0_r1_20260710/candidate_eval.json` |
| 74 | `server-matrix-runs:nf/trajectory_subset_targetrows_20260710/qwen3_4b_to_blank_qwen134m_targetrows_subset_pca_outputgate_gpu6_r1_20260710` | blank-target width/cross-family | cross-width | fails: behaviour shared_parent_pass_retention_rate 0.0; choice parent_pass_retention_rate 0.284; log tail: Loading weights: 0%/ / 0/211 [00:00<?, ?it/s] / Lo | **dead** | digit cycling | `NAS nf/trajectory_subset_targetrows_20260710/qwen3_4b_to_blank_qwen134m_targetrows_subset_pca_outputgate_gpu6_r1_20260710/candidate_eval.json` |

**Fresh decodes, case counts** (all in `JA/dossiers/`; the parents are the same-frame controls):
- Qwen3-0.6B-Base parent: 16 of 16 (`q06-parent-fs.json`).
- fold 19+20, avg 19+20, fold 8+9,19+20 and avg 8+9,19+20: 16 of 16 each. I fixed the classes at 11-12 cases for two of them, re-read all 16 after the decodes finished, and the classes did not change.
- The 13+14 merge, which passed in June: 1 case only (stopped to give its CPU to the four dropped merges).
- Snapshot of all dossiers: `JA/final_snap.tgz`, sha256 `509fa7f9488c91625012fe31978f41bd6c42319bc3cdd9d7337c439865034f9f`.
- SmolLM2 parent and fold 13+14: 16 and 16.
- 134M head-norm (template frame): 16.
- 1.41B compiled child: 2.

---

## 4. Families that were wrongly dropped, with evidence

### 4.1 Layer excision: the site census killed 9 live blocks (v2: layer folding)

`MX14.F-EXCISE.L01`-`L13` (wave 14) failed 12 of 13 LFM2 blocks on one number: the held-out "absorbed fraction" of a re-solved upstream writer, below a pre-registered 0.50 floor (for example L07 at -0.0015, L13 at +0.4669). No child was read. Waves 15-17 later built every single-block excision. MX15.F7 shows the upstream re-solve is a no-op (update scale 0.029, cos 1.0000, same review), so pure excision is the fair test of the census prediction. Reading those dossiers:

- **alive:** L06, L07, L09, L11, L13 (L07, `MX/wave16/human-review/READ-X07.txt`: 102 kits, retrieval of 4:15 p.m., Supplier B with the one-week risk, a valid seen-set dedupe skeleton; L13 is the record's 13/15 child);
- **partially alive:** L01, L03, L08, L10 (coherent, on-task language with some correct content);
- **dead:** L02, L04, L12 (and L00, which the census did not fail).

The census got 9 of 12 sites wrong, and it ranked blocks backwards (`UI.census-backwards`). **v2 consequence:** single-block removal survives at most sites of a small hybrid model. Which block is removed decides which capabilities survive (`R3.L6-blocks-choose`). Site choice for a v2 depth fold has to come from built and read children, never from a fit census.

### 4.2 Attention head merge (T7 / M5) was "priced out", but both built sites are alive (v2: residual-attention / attention folding)

`MXQ.probe.L02/L04/L08/L10/L12.T7.S4` were dropped as "8-13x more drift per M-param than the cheapest SwiGLU lane". That is efficiency gating, which Cyril ruled out on 2026-08-12 ("[Cyril quote withheld]"). Two of the five components were built as COMP children, and the machine called both "dead per child":

- `COMP.MA.MXQ2` (L02 heads at S4): 17x6 = 102, 102-8 = 94, 94/5 = 18 before a muddled finish; correct retrieval; coherent analogy. **Alive.**
- `COMP.MA.MXQ1` (L12 heads at S4): correct order Mira, Theo, Jun, Sal; retrieval fully correct (Saturday, 4:15 p.m., North Gate), better than the parent's answer on that case; Supplier B kept. **Alive.**

L04, L08 and L10 were never built. **v2 consequence:** a single-site attention head fold at true 4x on its component (753,664 of 3,145,728 parameters stored, 0.2396; whole child 0.989586x) keeps intelligence on LFM2. It belongs in the working deck. Cost per parameter only orders the attempts.

### 4.3 Depth-sharing codebook at 35/36 on Qwen3-4B: "prose collapses" is false (v2: layer folding)

`docs-residual-attention:cat-codebook-uniform-35` (pn35_pa35, 1.0257x, `fails`: "Some code/math survives, but prose, model-surgery, and much math collapse") and `qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710` (`fails`: "not a successful fold") rest on the July 1,000-item machine bank. Reading the saved continuations:

- **The bank is broken.** It holds 206 unique prompts. One tokenizer prompt and one commonsense prompt are each repeated 200 times and scored by a keyword-group rule. The **unmodified control (pn36_pa36, i.e. the parent) scores 0/200 on both**, so 400 of the 1,000 "behaviour" points cannot be earned by anyone.
- **pn35_pa35 text:** "The driver should slow down and drive with caution." (scored 0/200); a correct tokenizer-mismatch answer (0/200); a correct neuron-merging definition (scored 0/200 here, 200/200 for pn35_pa36); `def add(a, b): return a + b` (67/67); 6 + 4 = 10; arithmetic degraded (35 + 27 = 71). **Partially alive**, and on prose better than the control's off-topic commonsense answer.
- **The cliff is real.** Every child at 34 and below (p34/36, p34/34, p32, p28) is dead on its text.

**v2 consequence:** the compression is tiny (1.9-2.6%), so this is not a fold to use. It does show that per-neuron depth sharing on a real 4B model keeps intelligence up to the cliff. Every July `behaviour x/1000` number on this bank (head isolation, the blank targets, the depth-sharing controls, the codebook ladder) mis-measures coherent children. It happened not to matter for the blank-target children, which are dead on their text.

### 4.4 Per-site MLP folds pruned on token match (v2: width folding)

`mem-other:mx-w12-mlp-all` and the wave 1-3 "MLP wall" pruned C11/C12/C13 cells at token match 0.31-0.59 without a read. The rescued `C13.T8.S4.L05` (mlp_down at 4x, `MX/cells/C13.T8.S4.L05/frozen-review-dossier.json`) reads parent-level on all 16 cases: a natural follow-up question, Saturday / 4:15 p.m., Supplier B, a valid dedupe, a six-step plan. **Alive.** The record already marks the wall superseded, and this read confirms it from the text.

### 4.5 Embedding route stopped on token match (v2: vocab)

`CAL.C01.T5.X4-mxc` (the recompiled A9 sparse residual-OMP embedding route, 1.36x counted) is recorded as "FAIL — campaign STILL STOPS" on token_match 0.125 against a pre-registered 0.90. Its 16 answers (`MX/cells/CAL.C01.T5.X4-mxc/mxc2-responses.json`) are coherent and on task: Saturday and 4:15 p.m., Supplier B with the one-week slip, a seen-set dedupe. The errors are bridge debris ("Saturd", "Suppl", "Vendore"). **Partially alive**, matching the 8/15 quality-band read the same day.

### 4.6 June layer merges on Qwen3-0.6B (v2: layer folding)

The June merge-first search on Qwen3-0.6B-Base dropped four depth merges as "Failed". The evidence was a 4-item choice probe (0.50 or 0.25 against 0.75-1.00) and a PPL ratio. No generations were read (`docs-june-july-early:q06-depth-{fold,average}-{19-20,8-9_19-20}`, `pg-may-jun:nf-0626-qwen3-mergefirst-depth-*`, `server-matrix-runs:nf/archive/qwen3_layer_merge_20260626#{fold,average}_{19-20,8-9_19-20}`; `NF/docs/VALIDATION_RESULTS_2026-06-26.md:405-410`). The children were deleted.

**Rebuild and decode.** All four were rebuilt for this audit with the same tool and groups. Parameter counts match the record: 580,318,976 is 2.64% shrink for one pair, 564,588,032 is 5.28% for two. The base parent cannot follow the chat surface, so each child and the parent were decoded on the frozen 16 prompts in the same fixed two-exemplar frame (dossiers `JA/dossiers/q06-*-fs.json`). In that frame the parent is solid: 102 → 94 → 18 r 4; Mira, Theo, Jun, Sal; Saturday and 4:15 p.m.; a seen-set dedupe; a six-step backup/DNS/TLS/verify/rollback plan.

- **fold 19+20: alive.**
  - Seasons: "axial tilt of approximately 23.2 degrees ... The idea that summer is due to Earth being closer to the Sun is incorrect". Parent-level: the parent also gives tilt, sunlight and the rejected distance idea.
  - Correct: retrieval ("Saturday ... 4:15 p.m. ... the north gate"); the summary (Supplier B plus the one-week slip); the email (Friday, Monday, tracking); the uncertainty answer (asks for the time range, units, sample size and prices); three numbered actions.
  - Arithmetic is partial (102, then 102-8 = 99). Ordering is wrong. It refuses the greeting, and the debugging case loops without a fix.
- **average 19+20: alive.**
  - Debugging: "def last_three(values): start = max(0, len(values) - 3); return values[start:]", the anchored fix, which the parent misses.
  - Correct: seasons with the misconception rejected; the summary; the email; a natural greeting with one follow-up.
  - Arithmetic (120) and ordering are wrong.
- **fold 8+9,19+20: partially alive.**
  - Coherent. It keeps the summary (decision and risk), three actions, axial tilt, a dedupe description, a plan naming backup, DNS, TLS and verification, and the uncertainty answer (asks for the range, units, sample and prices).
  - Arithmetic, ordering and debugging collapse into wrong content or loops. It refuses the email and the snack revision.
- **average 8+9,19+20: partially alive.**
  - Kept: seasons with the misconception rejected; the summary; an email naming Friday, Monday and the tracking number; a six-step plan naming backup, DNS and TLS.
  - Lost: arithmetic, ordering, the uncertainty answer (it invents a cause) and the snack revision.

**v2 consequence.** On a real 28-layer Qwen3, merging one adjacent pair (fold or average) keeps parent-level intelligence on most of the surface. Merging two pairs keeps coherent language and much of the content, and loses arithmetic and ordering first. Two earlier statements were about a choice probe, not about behaviour: the record's "On plain Qwen3-0.6B, none above 2.85%" (`FOLDING_RECORD_2026-10-22.md` §3d) and June's "Layer group 13+14 survived; later and multi-group folds failed behavior gates" (`VALIDATION_RESULTS_2026-06-26.md`). Depth merging is a live v2 technique and has not been mapped past two pairs.

The same June surface on SmolLM2-135M (fold 13+14 and its ridge reconstructs) cannot be judged this way: the 135M base parent copies the exemplar or echoes on almost every case even in the few-shot frame (`JA/dossiers/smol-*-fs.json`).

---

## 5. Families correctly dead on behaviour, and arms that cannot be judged

### Dead on behaviour (the number was right; the text agrees)

- **Hidden width below about 0.86x on LFM2.** W768 (machine 3/15) reads dead to me: one copied summary and fragments. W736, W720, W704, W640 and W512 are empty output or dash/quote cycling. W896 (0.859x) is weakly partially alive. The width cliff is real on behaviour, and the record's ceiling statement stands.
- **All-site MLP 4x on LFM2** (`ARCH.ff640` = `MX14.fourteen-swiglu-coresets`): grammatical fragments with no task content, so dead. The 2x-per-site bisection child (`LFM2.ff1280-all-sites`) is partially alive.
- **Qwen3.5-0.8B C2K896, MLP 4x** (machine "0/162, 158 artifact misses"): on the same 16 prompts it echoes the prompt or loops `<think>` tags. The only correct content is the Mira/Theo/Jun/Sal order inside a loop. Dead.
- **July Qwen3-4B → 131M/134M/1.13B blank targets (25-30x), all 10 read:** token cycling or word salad on every probe. These deaths are real. Their "behaviour 0/1000" comes from a broken bank (section 4.3), but the text is dead anyway.
- **Extreme depth sharing** (tieddepth122m 32.9x, depth-basis allrank1 8.2x, basis-134M full topology 29.9x): word salad. The record's `ra-law-depth-sharing-closed` holds on behaviour.
- **Depth codebook at 34/36 and below** (p34/36, p34/34, p32/32, p28/28): punctuation streams or empty output. The cliff between 35 and 34 is real.
- **Single-block excisions at blocks 0, 2, 4 and 12** (`X00`, `X02`, `X04`, `X12`; census arms L02, L04, L12): echo, refusal loops or debris. These agree with the machine counts.
- **The 1.41B compiled child** (`compiled_child_cache/bdf2309d`, held blob 2, never measured before): numeral cycling from the first token.

### Cannot judge on behaviour

| Arm(s) | Why | What it would take |
|---|---|---|
| **K3 dense pass** (`K3DP.04db523ceafa`, `K3-DENSE-L61-L92-claude-oct09`, the "68 refused" layers, C14 union) | never decoded; the weights exist (96 GB + 77 GB + 12 GB, section 2). Decoding needs the K3 attention stack (KDA/MLA) plus all 92 dense FFNs resident (well over 100 GB in bf16), which breaks the MemAvailable >= 40 GiB rule on CPU, and no generation loop exists | a K3 dense generation harness on GPU through the gates (or a RAM plan that keeps 40 GiB free), then the frozen 16. **This is the single most consequential unreviewed arm for v2.** |
| C10 / S5 (`OC02-F-Q38-27BU-C2K4352K3U`, D4 sweep) | only R-Q top-1 numbers; the 51 GB child exists; a 27B CPU decode was not run | one GPU decode of the frozen 16. Cyril retired the 27B line on 10-22, so this is low priority |
| Media: C15 VAE ladders, C16/C17, S7 units, QIMG B30 dual-stream refold ("0/60 blocks met all targets", rejected after Cyril's ev 892366 "[Cyril quote withheld]") | every verdict is relL2 / PSNR / dB; no refolded block was ever rendered to an image, video or audio sample | render each unit's output (a QIMG image with block 30 refolded, an H3 clip with the L25 unit) and look at it |
| T7 head merge at L04, L08, L10 (`MXQ.probe.L04/L08/L10.T7.S4`) | priced out on drift per parameter; never built or decoded | rebuild with the wave-13 builder (GPU, LFM2-sized) and read the frozen 16; L02 and L12 are alive (section 4) |
| Exact-4x LFM2 children (`r13-first-composed`, `W13L.5a`, a0027, a0031 production) | weights exist, but no decoded text was saved (`route_smoke_accepted: true` only) and they need the carrier runtime; the siblings r12 and r13-armA were reviewed at 0/15 | run `lfm2_carrier_review_cli` on the four published children |
| SmolLM2-135M fold 13+14 (June; its ridge-reconstruct variants were not rebuilt) | rebuilt and decoded, but the 135M **base** parent does not demonstrate the surface even in the few-shot frame (it copies the exemplar or echoes), so parent and child cannot be told apart | an instruct parent, or a completion-style bank matched to a 135M base model |
| Census and prediction kills never built: M3 cross-layer sharing, NKP/Kronecker, HODLR-V, vocab row tying, score-bilinear q/k, XCHANGE, F-LANDMARK | no child exists; dropped on Frobenius error, energy or a census | build-don't-predict: build one child per class |
| Qwen3-4B residual-attention children (head isolation, p35 QKVO, 1.41B exact target, guided GUD) | children deleted; only first-action / rollout numbers kept, no text. The one surviving 1.41B compiled child (held blob 2) was decoded here and is dead | rebuild from the July runners if this line matters again |
| Coupled-attention Q-DEIM (37 arms), CPCF v5 (144 arms) | run directories purged under ev 577425; numbers only | rebuild if wanted |

---

## 6. The 22 held U3 blobs: keep or delete

The list is `/home/cereal/nas-cleanup-20261028/u3u4/lists/HELD_foldrec_cited.tsv` (22 files, 80,196,246,372 B, all present at their recorded sizes on 2026-10-29). Cyril approved deleting the old Neural Forge caches (2026-10-28, "[Cyril quote withheld]"). The only question here was whether any blob is still needed to judge behaviour. **None is.** Each was either judged in this audit from its saved outputs or a fresh decode, or is not a fold child at all. Every run directory keeps its JSON evidence: `candidate_eval.json`, run, certificate and compile reports. Paths are relative to `/mnt/ai_data/neural-forge/`.

| # | Blob (model.safetensors under) | Bytes | What it is | Behaviour (this audit) | Decision |
|--:|---|--:|---|---|---|
| 1 | `canonical_blank_targets/qwen3_stage1_blank_1410567168_fp16_seed0` | 2,821,172,288 | random blank 1.41B init (seed 0, certified), not a fold | n/a | delete |
| 2 | `compiled_child_cache/bdf2309d.../` | 2,821,172,288 | 1.41B "compiled child" of Qwen3-4B, capability unproven, never measured before | **dead**: numeral cycling ("Five 五百 ...") from the first token on both decoded cases (JA `dossiers/compiled-1p41b-bdf2309d-fs.partial.json`) | delete |
| 3 | `cross_family_blank_20260710/qwen3_4b_to_blank_llama131m_delta0p1_gpu0_r1_20260710/checkpoint` | 525,199,112 | Qwen3-4B → Llama 131M blank target | dead (saved outputs) | delete |
| 4 | `.../qwen3_4b_to_blank_llama131m_delta0p5_gpu4_r1_20260710/checkpoint` | 525,199,112 | same, delta 0.5 | dead | delete |
| 5 | `.../qwen3_4b_to_blank_qwen134m_headnorm_delta0p1_gpu4_r2_20260710/checkpoint` | 537,811,808 | Qwen3-4B → Qwen 134M | dead (saved outputs + frozen-16 decode) | delete |
| 6 | `.../qwen3_4b_to_blank_qwen134m_jointvocab_sourcel2_delta0p1_gpu4_r3_20260710/checkpoint` | 537,811,808 | same, joint vocab | dead | delete |
| 7 | `cross_token_real_medoid_20260710/qwen3_4b_to_blank_qwen134m_v50982_dual_realmedoid_gpu0_r2_20260710/checkpoint` | 537,805,680 | cross-token 134M (vocab 50,982) | dead | delete |
| 8 | `native_parent_cache_20260710/neo1_5_2b_materialized_1059.../` | 5,793,326,496 | a materialized copy of the public parent Paranioar/NEO1_5-2B-SFT (rev 105970970e), not a child | n/a | delete (re-downloadable) |
| 9 | `progressive_closed_form_20260710/qwen3_4b_to_blank_qwen1p13b_stage1_headnorm_delta0p5_gpu6_r3_20260710/checkpoint` | 4,520,068,088 | Qwen3-4B → 1.13B | dead | delete |
| 10-12 | `progressive_stage1_20260709/qwen4b_to_blank1p410b_{adaptive_global_native_gpu0, adaptive_observable_native_gpu4, adaptive_observable_path_gpu6_r2}_20260710/stages/01_blank_1p4b/blank_target` | 3 x 2,819,531,840 | blank 1.4B targets of three runs that failed before folding (`orchestration.status=failed`) | n/a | delete |
| 13 | `qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn34_pa36_gpu6_r1_20260710/compact_child` | 8,524,622,501 | depth codebook, MLP 34/36 | dead (saved outputs) | delete |
| 14 | `.../qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710/compact_child` | 8,674,083,493 | depth codebook, MLP 35/36, 1.019x | **partially alive** (saved outputs) | delete: the 4.4 MB `candidate_eval.json` keeps the text, and 1.9% compression makes it useless as a fold source |
| 15 | `.../qwen3_4b_codebook_pn36_pa34_gpu6_r1_20260711/compact_child` | 8,718,687,021 | attention 34/36, 1.013x | alive on the probes read (correct definitions, code, 37x19=703); machine retained 398/400 | delete (same reason) |
| 16 | `.../qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child` | 8,771,115,653 | attention 35/36, 1.0067x, already `works` | alive | delete (same reason) |
| 17 | `qwen3_4b_compensated_dynamic_v10_20260716_diagnostic_pid947219/child` | 8,044,920,384 | stores 4,022,437,376 unique params against the parent's 4,022,468,096 (1.0000076x): a diagnostic, not a fold | n/a | delete |
| 18 | `qwen_basis_checkpoint_20260710/qwen3_4b_to_basis134443364_fulltopology_gpu0_r1_20260710/checkpoint` | 268,907,584 | 134M depth basis at full topology | dead | delete |
| 19 | `real_medoid_dual_20260710/qwen3_4b_to_blank_qwen1p13b_dual_realmedoid_gpu4_r2_20260710/checkpoint` | 4,520,068,088 | 1.13B | dead | delete |
| 20 | `same_token_tokenrows_outputgate_20260710/qwen3_4b_to_blank_qwen1p13b_dual_realmedoid_tokenrows_outputgate_gpu4_r1_20260710/checkpoint` | 4,520,068,088 | 1.13B | dead | delete |
| 21 | `trajectory_subset_targetrows_20260710/qwen3_4b_to_blank_qwen134m_targetrows_subset_dual_gpu0_r1_20260710/checkpoint` | 537,805,680 | 134M | dead | delete |
| 22 | `trajectory_subset_targetrows_20260710/qwen3_4b_to_blank_qwen134m_targetrows_subset_pca_outputgate_gpu6_r1_20260710/checkpoint` | 537,805,680 | 134M | dead | delete |

**Keep: none for behaviour judgement. Delete: all 22 (80,196,246,372 B).** This audit deleted nothing. The deletion belongs to the cleanup lane: rerun `make_plan_n2.py` against `DELETE_ALL.tsv` with the hold lifted, through `nas_group_delete` with an omni_lab dump first. The foldrec citations of these paths are provenance records; the JSON evidence they cite stays in place.

---

## 7. What was written to omni_lab

**Route.** A new foldrec-v1 shard, `omni-lab/inventory/2026-10-22-foldrec/judgement-audit-2026-10-29.json` (74 records, sha256 `1376ee6284f608aceff9440e65e135ce0b136ab774238aabd1c993025726b112`), ingested with `omni_lab.py ingest foldrec`. I did not use raw SQL. A raw `fold_behaviour` insert would be deleted by the next foldrec re-run: `replace_children` drops ordinals no shard supplies.

**Record design.** Each record carries the existing arm key, one behaviour entry (`scorer: claude`, `score` = the class, `text` = the rule plus what the text shows), two sources (the evidence read, and this file) and a `check` stamp. Every scalar field is null. `merge_arm` ignores null variants, so verdict_class, ratios and the other columns keep their values, and only the behaviour list gains a row. The keys were resolved against live `fold_arm` before writing, and 0 were missing.

**Dry run** (read-only, before the write): 7,895 arms, the verdict_class distribution unchanged (fails 2,364, partial 786, works 342, ...), merged-arm conflicts unchanged at 1,441, and no validation problem for this shard.

**Backup first:** `DB_NAME=omni_lab backup_pg_db.sh run` (2.0.0) produced `/mnt/lexar/omni-campaign/backups/omni_lab/dumps/omni_lab_20261029T032652Z.dump`, sha256 `7cf80fca8889962ff8aa60a192d2544848aa884d5689b187203fcaeeba15f751`. The restore test was exact (51 tables, 300,076 rows), and the NAS copy is verified at `/mnt/ai_data/backups/omni-lab-postgres/omni_lab/`. The run row is omni_lab run 40443.

**Write:** ingest batch **276**, status ok, 2026-10-29 03:29:15 to 03:56:47Z (1,652.2 s), seen 19,607, written 444.
- `fold_behaviour` +74 inserted, all `scorer=claude`: alive 14, partially alive 15, dead 42, cannot judge 3. Total behaviour rows 6,600 → 6,674; Claude rows 2,775 → 2,849.
- `fold_source` +148 inserted. `fold_shard` +1.
- 23 behaviour and 81 source rows changed only their ordinal: the new shard sorts ahead of later shards in the union. Every non-Claude scorer count is unchanged (machine 2,006, none 1,167, mixed 390, codex 147, NULL 104, cyril 11).
- `fold_arm` 74 updated: `shards`, `record` and `member_keys` gain the new shard. No arm was inserted. Components (12,163), diagnostics (38,600), findings (2,947) and the verdict_class distribution are identical.
- **verdict_class was deliberately not changed.** The re-judgement sits beside the old verdict as its own behaviour row. Whether `fails` arms with an alive Claude read should move to `partial` or `works` is a separate decision for whoever owns the record.

**After:** `omni_lab_20261029T035817Z.dump`, sha256 `7c66f76e3841a19346ad5118bfd10f66739689a578a607836dc40f0499456ded`. Its restore test was exact (51 tables, 300,315 rows), and the NAS copy is verified.

**Inventory files:**
- `README.md` in the shard folder gained a batch-276 row.
- `SHA256SUMS` gained the new shard's line (backup of the old file: scratchpad `ja/SHA256SUMS.bak`).
- `sha256sum -c` now reports 7 older shards changed since SHA256SUMS was written (docs-june-july-early, gap1-child-deletion-cleanup-records, gap1-mx-wave-notes-p-w, gap1-run0-lateoct-chimera-repo-misc, gap2-partial-md-completions, mem-matrix-1, omni-fold-sep-oct). That drift comes from the batch 274/275 edits, not from this audit. I left those lines alone.
- Nothing was committed: the omni-lab repo has other agents' uncommitted work.

**Query to see the rows:** `select fa.arm_key, b.score, b.text from fold_behaviour b join fold_arm fa using (experiment_id) where 'judgement-audit-2026-10-29' = any(b.shards) order by 1`.

---

## 8. Deviations, limits and what was not verified

- **Who read what.** Every class in section 3 is my (Claude's) read of the text. Nobody else re-read it, and none of it is Cyril's.
- **Coverage.** 2,430 arms are in the population. This audit re-judged the ones whose technique matters for a v2 fold (layer, width, residual-attention, vocab, depth sharing) and that left text or weights. The rest are untouched, including 610 arms that failed or were refused before a child existed (nothing to read) and 301 media arms (nothing rendered).
- **Frozen-surface deviations in the CPU re-decodes** (recorded in every JA dossier under `decode`):
  - The surface is the frozen pack `research/lfm2_semantic_whole_response_prompts_v1.json` (sha256 `0ee6bf14...`, verified before every run). The anchors come from `V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md` (sha256 `e5e37768...`, verified).
  - Greedy decoding, fp32, CPU (Xeon E5-2699 v4, `nice 19`, 4-6 threads per model).
  - The cap was 512 or 256, not 1,024, with no 2,048 retry. Cases decoded before the change kept 512.
  - A loop guard stops a decode once the last 3P tokens are one P-token cycle (P <= 64). That is a termination artifact and never a miss.
  - **Base parents** (Qwen3-0.6B-Base, SmolLM2-135M, and the Qwen3-4B-Base-derived 1.41B child) echo the chat-template prompt and demonstrate nothing. They were decoded in a plain `User:` / `Assistant:` frame with a fixed two-exemplar preamble ("12 + 7 = 19"; "Saturn has rings"), identical for parent and child, stopping at the model's next `User:` turn. The instruct-tuned LFM2 dossiers from August were produced at a 1,024 cap with the chat template.
- **The 1.41B compiled child** was stopped after 2 of 16 cases to free CPU for the merges. Both cases were numeral cycling from the first token.
- **No GPU was used.** The P40 claims and `lanes.py` were checked first. Run 0 held no P40 claim at the time, and GPU0 and GPU6 were serving Strata. Everything ran on CPU, and MemAvailable read 146-184 GiB at every check.
- **Not verified:** whether the June rebuilds are byte-identical to the deleted June children (the June digests were not recorded). They come from the same tool, the same groups and the same method, with matching parameter counts. Also not verified: the machine counts quoted from the record; I did not recompute any of them.
- **Not done here:** no deletion (section 6 hands the 22 blobs to the cleanup lane); no K3 decode; no media rendering; no rebuild of the unbuilt T7 sites or of the census-killed classes.
