# STAGE-2 FIRST MEASUREMENT — fold-as-initialization vs distillation-from-scratch at matched compute

**Date:** 2026-08-08 · **Branch:** `codex/lrc-blank-compression` · **HEAD at design time:** `4e6d0ed6`
**Status:** DESIGN ONLY. Nothing here has been implemented, launched, or committed.
**Authority:** `docs/CURRENT_FOLDING_GOAL.md` (stage 2 OPEN, matched-compute controls MANDATORY) and
`docs/STAGE1_CEILING_REPORT_2026-08-08.md` section 7.4 (the two named measurement points).
**Pre-registration:** section 8 of this document is the pre-registered criteria set. Its sha256 must be
recorded in the first stage-2 training receipt before any training step runs, and section 8 may not be
edited afterward.

---

## 0. Problem

Stage 1 proved the fold-only ceiling: 10-12/16 at 1.000x body, 0/15 at and beyond 1.559x body
(`STAGE1_CEILING_REPORT_2026-08-08.md` section 2). Whether a fold is worth anything as a *training
initialization* is currently assumed, not measured. This experiment measures it: two fold-initialized
children plus minimal recovery distillation, each against a distillation-from-scratch control of
identical counted topology at matched compute, so the fold's compute value becomes a sealed number
instead of a belief.

---

## 1. Arms

Four arms minimum, per section 7.4 of the ceiling report. All training is stage-2-labeled; no stage-1
artifact is modified.

| arm | initialization | counted topology | body ratio | pairing |
|---|---|---|---|---|
| **S2-A** | A9/A11 dual-route child — saved physical child state `54b9c5f7f484fd256a985aa33346f1bf4f601bec5f1978c287341d9e27f36e85`; 166,778,624 physical + 3,867,776 route coefficients = 170,646,400 counted (1.346018x) | identical to A11 publication `860bccfe…` | **1.000x** (body uncontracted) | vs S2-C |
| **S2-B** | a0028 2x child — published artifact `f3f38cc3e87de7299196a5c8bb553ef1ca735a630ed43616029a59f694958fad`; 110,978,816 dense + 3,867,776 continuous = 114,846,592 counted (2.000000x) | `config.json` sha `5c93a191…` | **1.559x** | vs S2-D |
| **S2-C** | random init (scaled-normal per LFM2 init scheme, seed recorded) | byte-identical config to S2-A, including both k32 route tensors as trainable parameters | 1.000x-equivalent | control of S2-A |
| **S2-D** | random init, same scheme, seed recorded | byte-identical config `5c93a191…` to S2-B, including both route tensors as trainable parameters | 1.559x-equivalent | control of S2-B |

**Why these two fold points and not 4x:** ceiling report section 7.4. S2-A measures how much
distillation a *working* fold saves. S2-B is the sharper question — a child with 0.569 terminal
reconstruction, 0.860 terminal cosine, and zero behavior directly tests whether the fold geometry is
a useful basin. The 4x child's pre-install composition was already worse than emitting zeros (ceiling
report section 2.2) and is excluded from the first measurement.

**Expected asymmetry, stated up front so it cannot be spun later:** S2-A's body is the parent's own
uncontracted body; its control S2-C is a full from-scratch ~167M-body pretrain. A large S2-A win is
expected and mostly re-proves that pretrained weights beat random ones. The result stage 2 exists to
produce is the **S2-B vs S2-D** comparison, where the fold-derived body is behaviorally dead but
geometrically close to the parent.

---

## 2. Teacher and data (design question 1)

### 2.1 Teacher

`LFM2.5-230M`, local on the AI server at `/home/cereal/model-surgery-lab/models/LFM2.5-230M`
(443 MB safetensors; verified present 2026-08-08; the HF cache also holds
`models--LiquidAI--LFM2.5-230M` and `-Base`). The teacher checkpoint digest must be pinned in the
corpus manifest and in every training receipt, and it must equal the stage-1 parent digest recorded
in the sealed run receipts — distilling from a different checkpoint than the one that was folded
would void the comparison.

### 2.2 Data inventory — what actually exists (measured 2026-08-08, read-only)

| location | content | verdict |
|---|---|---|
| repo `experiments/*.json` fit/holdout prompt packs | hundreds of prompts; `lfm2_on_policy_fit_prompts_v1.json` = 120 prompts, corpus sha `4bf7263d…`, disjointness-checked against 552 digests | review/probe surfaces only — 3-5 orders of magnitude short of a training corpus |
| `research/lfm2_semantic_whole_response_prompts_v1.json` | the 16 V5 prompts, 8.6 KB | frozen review surface; must NEVER appear in training data |
| `/mnt/ai_data/omnicoder_2026/datasets/clean/full_training_ready_v7_20260530T041825Z` | 79 MB mixed-modality jsonl (text/image/video splits) | tens of M tokens of mixed text at best; insufficient volume, wrong distribution |
| `/home/cereal/omnicoder_2026_work/weights/*datasets*` | lock files only, 4 KB dirs — data purged | nothing |
| `/mnt/ai_data/training_data` | `film2`, `rvc` — 0 bytes | nothing |
| HF datasets cache | eval sets (ai2_arc, hellaswag, piqa, winogrande, SWE-bench…) | eval-only; not training text |

**Conclusion: no local corpus suffices.** The corpus is therefore **teacher self-generation**, which
is also the principled choice for pure knowledge distillation: when the optimization target is the
teacher distribution itself, teacher-sampled sequences are exactly on-distribution (sequence-level KD
in the Kim & Rush sense), and no external-download dependency or distribution-mismatch argument
enters the experiment.

### 2.3 Teacher self-generation spec

| parameter | value | rationale |
|---|---|---|
| Volume | **1.12B donor tokens** = 1.10B train + 20M held-out (held-out used only for diagnostic NLL, never gating) | covers the 1B ladder rung with zero repeats |
| Sequence length | 1,024 donor tokens | matches the runtime's bounded decode session scale |
| Prompt source | 50% unconditional from BOS; 50% seeded with 64-token prefixes drawn (seeded RNG) from the bridge folding corpus (`experiments/compile_lfm2_bridge_folding_corpus.py` output), the a7 repository-corpus and a8 natural-language-corpus prompt sets (`reports/carrier-replant-a7-repository-corpus356-20260730`, `…-a8-natural-language-corpus356-20260730`), and `experiments/lfm2_on_policy_fit_prompts_v1.json` | reuses the program's own prompt estates; nothing external |
| Sampling | ancestral, temperature 1.0, top-p 0.995 | T=1.0 gives unbiased samples of the teacher distribution (correct for forward-KL estimation); top-p 0.995 trims only the degenerate tail |
| Exclusions | the 16 V5 prompts and their 552-digest disjointness set screened out of every prefix and generated sequence by the same utf8-sha256 exact-and-stripped comparison the fit-prompt pack already uses | frozen review surface stays out of training |
| Token id domain | donor tokenizer, the 64,402 defined ids only; the 1,134 undefined physical rows asserted absent (goal doc: "must never be scored") | contract carry-over |
| Curation filter | drop sequences whose 3-gram self-repetition fraction > 0.5 (recorded per-shard in the manifest; a data-curation rule, not an acceptance metric) | a 230M teacher at T=1.0 will emit some degenerate text |
| Hardware / cost | RTX 8000 (idle, 48 GB), batch ~512, estimated ~20k tok/s → **~15.5 GPU-h, ~1 day wall** (estimate; replaced by the plan's calibration measurement) | keeps both 3090s free for training |
| Storage | uint16 donor-id shards ~2.24 GB + raw text ~5 GB + manifest, under `/mnt/ai_data/neural-forge-stage2/corpus/` | `/home` is at 93%; the CIFS mount has 1.8T free |
| Manifest | per-shard sha256, generation seed, sampler params, teacher digest, filter statistics; one top-level `corpus_manifest.json` whose own sha256 is quoted in every training receipt | sealed-evidence discipline |

**Teacher logits are computed on the fly during training**, not stored: storing top-16 logits for
1.1B tokens costs ~100 GB and buys nothing when a 230M teacher forward is ~15% of the training step.

### 2.4 Objective: hybrid — off-policy logit KD + on-policy GKD

**Chosen:** token-level forward-KL logit distillation on the teacher-generated corpus for the first
10M tokens, then a **50/50 mix of off-policy forward-KL and on-policy GKD** (sample a 256-token
continuation from the student on corpus prefixes; the teacher scores the student's own tokens;
generalized JSD loss with beta = 0.5, per Agarwal et al., GKD) for the remainder. The objective
schedule is identical for fold arms and controls.

**Why on-policy is non-negotiable here — our own sealed evidence, not just literature:** the ceiling
report section 4.3 shows teacher-forced top-1 **anti-correlated** with free-run across four children
(TF 1.86% → 33.10% while free-run fell 0.44% → 0.034-0.195%, verdict pinned at 0/15 throughout), and
W13A's causal inversion names the mechanism: a child fitted on the parent's trajectory becomes a
self-consistent function in its own wrong regime. Pure teacher-forced distillation optimizes exactly
the metric class stage 1 disqualified. Training must therefore spend a large fraction of its gradient
on prefixes the *student* visits — GKD is the gradient-era version of the stage-1 on-policy evidence
extension (`docs/ON_POLICY_EVIDENCE_EXTENSION_2026-07-31.md`).

**Rejected:** pure sequence-level KD (hard-target CE on teacher samples) — discards the teacher's
distributional information, which matters most at 230M scale; pure logit KD on off-policy data —
contradicted by the exposure-bias evidence above; RL-style reward shaping — stage 3 territory,
closed.

### 2.5 The KD interface — donor-vocab logits through the routed readout

Both fold children speak the carrier ABI: prompts donor-tokenized, one donor action = one body
advance, semantic 4,096-row table, dual k32 routes. The distillation loss is computed **in donor
vocabulary space**: student logits = routed C_out reconstruction over the 64,402 defined donor ids;
teacher logits over the same ids; undefined rows masked from every loss term and every sample. The
trainer consumes donor-token sequences directly, so training runs in the same semantic time as the
runtime — no per-spelling-token advancement (goal doc, Active mechanism 1). This one decision is
what makes fold arms and controls trainable and reviewable through a single code path.

---

## 3. Budgets and the training-curve metric (design question 2)

### 3.1 Budget ladder

**Rungs: 10M / 100M / 1B student-consumed donor tokens**, implemented as checkpoints inside a single
run per arm (not three runs): batch 16 seq x 1,024 tok = 16,384 tok/step → checkpoints at steps
**610 / 6,104 / 61,036**. "Minimal recovery" is then read off the curve: the smallest rung at which
the fold arm's frozen-review verdict clears its target (section 8).

Optimizer (identical for all four arms, pre-registered): AdamW beta=(0.9, 0.95), weight decay 0.1,
lr 3e-4 cosine to 3e-5, 100-step warmup, grad-clip 1.0, bf16 autocast with fp32 master weights.
One seed per arm in the first measurement; the only pre-registered follow-up is a seed replicate of
the decisive pair (section 8.5). An optional lr-sensitivity cell (S2-B and S2-D re-run to the 100M
rung at lr 3e-5, ~5 GPU-h total) is pre-registered as diagnostic-only, because 3e-4 may be
init-destroying for a fold-initialized body and matched hyperparameters must not quietly favor the
control.

### 3.2 The honest intermediate metric

The 16-prompt frozen review is manual and cannot ride the training curve, and the two obvious cheap
metrics are disqualified: terminal reconstruction (ceiling report section 2.3 — a 0.507 swing with
no behavioral consequence) and TF top-1 (section 4.3 — anti-correlated). Per the audit's metric
policy (`CURRENT_FOLDING_GOAL.md`, closed-path additions, clause 5: "the only rollout-level numeric
progress signals are free-run token match/lcp; above them, the frozen whole-response manual
review"):

* **Curve signal (sanctioned): free-run token match and mean longest-common-prefix vs teacher
  rollouts**, greedy, 256 tokens, on a fixed 64-prompt probe drawn (seeded) from
  `experiments/lfm2_on_policy_fit_prompts_v1.json` (already disjointness-checked against the review
  surface, corpus sha `4bf7263d…`). Logged every 1M tokens. Cheap: 64 x 256 tokens, seconds per
  probe.
* **Diagnostic-only: held-out NLL** on the 20M held-out split — logged, plotted, never in any
  acceptance path, never a headline, reported only with the section-4.3 calibration caveat attached.
* **Early stopping: none mid-rung.** Rungs are fixed checkpoints; the free-run curve informs
  *spending* decisions (whether to review a 10M dossier, section 3.3) but never a verdict.

### 3.3 Review schedule

The frozen V5 protocol (`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`) is the
only verdict. Rung checkpoints are decoded through the routed dossier machinery (identical caps:
1,024 initial / 2,048 retry) and reviewed:

* **Mandatory:** all four arms at the 100M and 1B rungs → 8 reviews.
* **Conditional (a spending rule, not a verdict rule):** an arm's 10M dossier is reviewed if its
  free-run token match at the rung is >= 1%, or if that arm's 100M review contains any
  Useful/Partial — needed to locate the *minimal* recovery budget. Worst case +4 reviews.
* **Parent baseline: 1 review** (plan step 2) — the ceiling report's `parent-not-demonstrated` gap
  (section 2.4 there) must be closed before any recovery claim is quoted against "the parent's V5
  behavior".
* **Blinding:** dossiers are handed to the reviewer under neutral run ids; the arm-to-id map is
  sealed before review and opened after all rung verdicts land.

Total: 9-13 half-day frozen reviews, **5-7 reviewer-days**. This is the schedule bottleneck and is
costed honestly rather than optimized away.

---

## 4. Compute matching (design question 3)

**Unit definitions.** (i) *Tokens*: donor tokens consumed by student forward+backward (the rung
variable; exact by construction). (ii) *GPU-hours*: measured wall-clock x device count from the
run's own receipt, per device model (all four training arms pinned to one RTX 3090 each, so
GPU-hours are commensurable). Teacher forwards are part of the KD step for **both** arm and control,
so they cancel. On-policy GKD sequences are policy-dependent, so arm and control cannot share
identical bytes in the on-policy fraction; matching is defined as **identical token counts,
identical objective mix and schedule, identical hyperparameters, and identical corpus + shard order
for the off-policy fraction** — this definition is itself part of the pre-registration.

**The fold's own compute cost — decision and justification.** Two pre-registered views, both always
reported; neither may be quoted without the other:

* **View 1 — initialization value (primary claim):** the control's budget = the arm's recovery
  budget, token-for-token. The fold cost is *stated* beside the result but not charged. Charging it
  inside the token budget would force unequal token counts and make "is the fold a better init than
  random at equal recovery compute?" unmeasurable. Claim form: "fold-init + N tokens beats
  scratch + N tokens".
* **View 2 — end-to-end savings (co-primary; required for the headline "folding saves compute"):**
  the fold arm's total = measured fold GPU-hours + recovery GPU-hours. Fold costs are read from the
  sealed receipts at implementation time (the 2x fold ran 13 h 51 m launch-to-final-receipt, 11 h
  16 m to `run_completed` — ceiling report section 6.1a; the A9/A11 publication cost comes from its
  completion artifact `55a550aa…`), never estimated. Claim form: "fold + recovery reached score band
  S in fewer total GPU-hours than scratch needed to reach S" — and if scratch never reaches S inside
  the ladder, the saving is lower-bounded by (1B − N_parity) tokens converted at the measured
  tokens/GPU-h rate, minus the fold cost.

A fold that wins View 1 but not View 2 is pre-registered as its own honest outcome: *"the fold is a
real initialization prior but not compute-positive at this scale."*

---

## 5. Trainer, hardware, checkpointing, decode equivalence (design question 4)

### 5.1 Trainer

**Smallest honest implementation: a plain PyTorch training loop, new stage-2 code in-repo.** No
trainer exists in-repo (stage 1 forbade one), and no external training framework (HF Trainer,
Lightning, accelerate) is used — they add config surface area that fights the receipt discipline,
and a ~170M-parameter single-GPU job does not need them. No database; receipts are
content-addressed JSON files exactly like stage 1. (Estate rule noted for any future stage-2 store:
raw parameterized PostgreSQL through psycopg2/asyncpg only — nothing in this measurement needs a
database.) No secrets are required anywhere in this experiment: the teacher is a local file, and any
credentialed operation would resolve from the server-manager vault; none is planned.

**Training receipts** (one JSON per run plus per-rung addenda), carrying at minimum: `stage:
"stage2"`, arm id, init provenance (fold-child digest, or init seed + scheme), teacher digest,
corpus manifest sha + shard-order seed, objective schedule, optimizer config, per-rung
`tokens_consumed`, `optimizer_steps`, `backward_calls`, wall-clock, device inventory, checkpoint
sha256s, the free-run probe series, the held-out NLL series (marked `diagnostic_only: true`), and
the sha256 of this document's section 8. Per ceiling-report section 7.4 rule 5, the stage-1 and
stage-2 provenance chains stay separate: every stage-2 artifact is labeled, and no stage-1
zero-training receipt is ever cited as covering a trained child.

### 5.2 Hardware fit (measured inventory, 2026-08-08)

| device | state | assignment |
|---|---|---|
| RTX 3090 x2 (idx 0, 4; 24 GB) | idle | one training arm each; two arms in parallel, two waves |
| Quadro RTX 8000 (idx 6; 48 GB) | idle | corpus generation, then the dossier-decode lane |
| Tesla P40 x3 (idx 1, 2, 3, 5) | idx 5 has 23 GB in use by another workload | **not used** — Pascal has no usable bf16, and idx 5 is occupied |

Memory check per 3090: student <= 167M params (bf16 compute copy 0.34 GB) + fp32 master + Adam
moments at 12 B/param ~2.0 GB + teacher bf16 0.46 GB + activations at 16 x 1,024 — comfortable in
24 GB. Throughput estimate ~12k tok/s per 3090 including the teacher forward (~1.5 GFLOP/token at
~30% MFU); **a mandatory 30-minute calibration run replaces this estimate with a measured number
before the pre-registration receipt is sealed** — the token budgets are exact regardless.

### 5.3 Checkpointing

Per arm: rung checkpoints (steps 610 / 6,104 / 61,036) with optimizer state, plus a rolling
30-minute safety checkpoint (last two kept). All under
`/mnt/ai_data/neural-forge-stage2/runs/<arm>/`, each checkpoint sha256-manifested. ~8 GB per arm,
~32 GB total.

### 5.4 Decode equivalence for the controls — the load-bearing fairness mechanism

The frozen review runs on the routed runtime, and the routed runtime admits only sealed carrier ABI
tuples (the 2x tuple was admitted via the `ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES` catalog edit
pattern, commit `e72bf169` — ceiling report section 6.1a). The controls must be reviewable on **the
same decode surface** or the comparison is void. Design:

1. **Identical counted topology including the routes.** S2-C/S2-D instantiate the exact fold-child
   configs (A11 publication topology; `config.json` `5c93a191…`) with both k32 route tensors
   (3,867,776 coefficients) as ordinary trainable parameters, randomly initialized. Counted totals
   match the fold arms exactly, so the counted-x and body-x labels stay honest.
2. **Identical parameterless integer routing.** The ByteLevel donor-to-target row codes are compiler
   tooling, not learned parameters (goal doc, Active mechanism 2); they are shared verbatim by all
   four arms. Sharing them gives the control the *plumbing* of the fold, not its *knowledge* — which
   is precisely the boundary the experiment needs to draw.
3. **A stage-2 admission path, not a weakened guard.** A new explicit admission list
   (`stage2_control: true` + config-sha + checkpoint-sha allowlist) admits trained stage-2
   checkpoints to the routed decode. The stage-1 admitted-tuple guard, its refusal messages, and its
   tests are untouched; new tests pin that stage-1 refusals still fire byte-identically. This is the
   one edit that touches guarded runtime code, and it gets adversarial review (section 10).
4. **One decode path for all dossiers.** Fold-arm checkpoints, control checkpoints, and the parent
   baseline all produce dossiers through the same routed dossier builder with the same caps, so the
   reviewer never sees a surface difference that correlates with arm identity.

---

## 6. Schema

No database is involved in this measurement; persistence is content-addressed JSON receipts and
sha-manifested binary shards (sections 2.3 and 5.1), matching the stage-1 evidence discipline. No
DDL. (If a later stage-2 tool wants a store, it uses raw parameterized PostgreSQL only.)

---

## 7. Cost summary

| item | GPU-h (est.) | wall-clock | disk |
|---|---|---|---|
| Parent V5 baseline decode | ~1 | 2-3 h + half-day review | ~10 MB |
| Calibration run | ~1 | 1 h | — |
| Corpus generation (1.12B tok, RTX 8000) | ~15.5 | ~1 day | ~8 GB |
| Training: 4 arms x 1B tokens at ~12k tok/s on 1x3090 each | ~93 (~23.2/arm) | ~2.5 days (2 arms parallel, 2 waves) | ~32 GB checkpoints |
| Rung dossier decodes: 9-13 x ~2.5 h (RTX 8000 lane) | ~32 | overlaps training | ~150 MB |
| lr-sensitivity cell (optional, diagnostic) | ~5 | overlaps | ~4 GB |
| **Total** | **~145-150 GPU-h** | **~6-8 days end-to-end** incl. reviews | **< 60 GB**, all on `/mnt/ai_data` |
| Frozen reviews (human/agent) | — | 5-7 reviewer-days (9-13 rung reviews + parent) | — |

Fold costs stated for View 2 (read from sealed receipts at implementation): 2x fold ~13.85 h wall
(GPU-hours from its own receipt's device record); A9/A11 publication cost from `55a550aa…`.

`/home` (93% full, 123 G free) is **not** used for stage-2 artifacts; everything lands under
`/mnt/ai_data/neural-forge-stage2/` (1.8 T free). No sealed-cache reclamation is required for this
experiment.

---

## 8. Pre-registered success / failure criteria (design question 5)

Sealed by digest into the first training receipt; not editable afterward. All verdicts are frozen-V5
useful-or-partial counts out of 16; no numeric metric has acceptance authority anywhere in this
section.

**8.1 Per-rung comparison vocabulary.** At a rung, the fold arm *wins* if its useful-or-partial
count exceeds its control's by >= 3 cases; the rung is a *tie* if the difference is <= 1; a
difference of exactly 2 is *marginal*. (Margin chosen above single-case review noise: the A9/A11
band itself is quoted as 10-12.)

**8.2 "The fold has initialization value at body ratio r"** — the fold arm wins >= 2 of 3 rungs, or
wins the 1B rung by >= 3, with no rung where the control wins by >= 2. For S2-B specifically, the
sharper pre-registered headline: **any nonzero useful-or-partial at the 10M or 100M rung while S2-D
sits at 0 demonstrates that the fold geometry is a useful basin** even though the untrained child is
0/15 — this is the result section 7.4 of the ceiling report names as the point of stage 2.

**8.3 "Folding saves compute"** (the headline) — requires 8.2 **and** View 2: fold GPU-h + recovery
GPU-h to reach score band S < control GPU-h to reach S, where S is the control's best achieved band;
if the control never reaches the fold arm's 10M-rung band inside the ladder, the saving is reported
as a lower bound per section 4. If 8.2 holds but View 2 fails, the sealed finding is *"a real
initialization prior, not compute-positive at this scale"* — a legitimate stage-2 outcome.

**8.4 "The fold is worthless as initialization at body ratio r"** — the control's count >=
(fold − 1) at every rung. For S2-B this is the honest null: the geometry stage 1 built at 1.559x
body carries no recoverable value, and stage-2 work at contracted bodies should route through
S2-A-style interface-only folds or await a different fold mechanism.

**8.5 Inconclusive** — any pattern not covered above (marginal differences, non-monotone rung
ordering). Pre-registered next step, the only immediate follow-up permitted before a new design: one
seed replicate of the decisive arm/control pair at the decisive rung (~25 GPU-h, 2 reviews).

**8.6 Stage-2 ceiling logic.** Per the staged policy, stage 2 also ends with a proven ceiling. Exit
form, mirroring stage 1's report: for each admitted body point, a sealed minimal-recovery curve —
best V5 band vs tokens for fold-init and for scratch, with matched-compute receipts on both sides —
plus either (a) a quantified savings statement (tokens/GPU-h saved as a function of target band, in
both Views), or (b) a proven-no-value statement at <= 1B tokens with the 8.5 replicate executed.
Extending the ladder (3B+, more seeds, the 4x point, other parents) is stage-2 continuation and does
not reopen this measurement. Every ceiling claim carries counted-x **and** body-x, per section 5.5
of the stage-1 report.

**8.7 Standing metric rules inherited verbatim** (ceiling report section 7.4 rules 1-5): frozen V5
is the verdict; compute is measured, not estimated, on both sides or the comparison is void; no
cached-frame metric in any acceptance path or headline, TF only with its calibration band; stage-1
artifacts stay sealed and read-only; the provenance chains stay separate and labeled.

---

## 9. Implementation plan (ordered; DESIGN ONLY — no step has been started)

All new code lives under `src/neural_forge/stage2/` and `tests/stage2/` (repo:
`C:\Users\cereal\Documents\ai training\neural-forge`). Server artifacts under
`/mnt/ai_data/neural-forge-stage2/`.

1. **Approval gate (Cyril):** sign off on this design, the section-8 pre-registration, the teacher
   self-generation corpus, and the stage-2 admission-path edit to the routed runtime guard
   (section 5.4 item 3).
2. **Parent V5 baseline** — decode LFM2.5-230M on the 16-prompt surface through the standard
   dossier machinery; frozen review; closes `parent-not-demonstrated`. (~1 GPU-h + half-day
   review.)
3. **Calibration run** — 30 min on one 3090: measured tok/s for the KD step and for RTX 8000
   generation; write the measured numbers into the pre-registration receipt; seal section 8's
   digest.
4. **Lane 2 first deliverable — corpus:** `corpus_generation.py` + `corpus_manifest.py`; generate
   1.12B tokens on the RTX 8000; manifest sealed.
5. **Lane 1 — trainer:** `distill_trainer.py`, `kd_objectives.py`, `training_receipts.py`,
   `freerun_probe.py`; unit tests green.
6. **Lane 2 — controls + decode:** `control_materialization.py`, `data_loader.py`,
   `routed_decode_admission.py`; tests green, including the pinned stage-1 refusal-message tests.
7. **Wave 1: S2-B + S2-D** (the sharper question runs first), one 3090 each, checkpoints at
   610/6,104/61,036 steps. Rung dossiers decode on the RTX 8000 lane as checkpoints land.
8. **Wave 2: S2-A + S2-C**, same procedure.
9. **Frozen reviews** per section 3.3, blinded labels; conditional 10M reviews per the spending
   rule.
10. **Seal `docs/STAGE2_FIRST_MEASUREMENT_REPORT_<date>.md`** with the full receipt/digest ledger,
    both Views, and the section-8 verdict mapping. No git operations unless Cyril directs them.

---

## 10. Work split for Codex (design question 6) — two independent lanes, disjoint file ownership

**Lane 1 — trainer & receipts:**
`src/neural_forge/stage2/__init__.py`, `distill_trainer.py`, `kd_objectives.py`,
`training_receipts.py`, `freerun_probe.py`;
`tests/stage2/test_distill_trainer.py`, `test_kd_objectives.py`, `test_training_receipts.py`,
`test_freerun_probe.py`; launchers `experiments/stage2/launch_s2a_foldinit.sh`,
`experiments/stage2/launch_s2b_foldinit.sh`.

**Lane 2 — data pipeline & controls:**
`src/neural_forge/stage2/corpus_generation.py`, `corpus_manifest.py`, `data_loader.py`,
`control_materialization.py`, `routed_decode_admission.py`;
`tests/stage2/test_corpus_manifest.py`, `test_data_loader.py`, `test_control_materialization.py`,
`test_routed_decode_admission.py`; launchers `experiments/stage2/generate_corpus.sh`,
`experiments/stage2/launch_s2c_control.sh`, `experiments/stage2/launch_s2d_control.sh`.

**Interfaces (fixed by this document, so the lanes never merge-conflict):** (i) the corpus manifest
JSON schema (section 2.3) — produced by lane 2, consumed read-only by lane 1's trainer; (ii) the
checkpoint directory layout `/mnt/ai_data/neural-forge-stage2/runs/<arm>/rung-<tokens>/` with a
per-checkpoint sha manifest — produced by lane 1, consumed by lane 2's decode admission; (iii) the
donor-vocab masked-logit contract (section 2.5). The one file adjacent to guarded stage-1 code,
`routed_decode_admission.py`, is owned by lane 2 with a mandatory adversarial review before merge.
Coordination goes through the Relay Desk per the standing comms channel.

---

## 11. Trade-offs ledger (chosen vs rejected)

| decision | chosen | rejected | why |
|---|---|---|---|
| Corpus | teacher self-generation, 1.12B tok, sha-manifested | FineWeb-Edu sample download (~27 GB) | nothing sufficient exists locally (section 2.2, measured); self-generation is exactly on-distribution for KD, fully local, no external dependency. The download remains a costed alternative if Cyril wants natural-text coverage: ~27 GB to `/mnt/ai_data`, no GPU cost, needs his approval. |
| Objective | hybrid forward-KL + on-policy GKD (50% after 10M tok) | pure logit KD; pure sequence-level KD; RL | our own sealed TF/free-run anti-correlation (section 2.4 here; ceiling report section 4.3) convicts pure teacher-forcing; GKD is the gradient analog of the sanctioned on-policy extension |
| Trainer | plain torch loop, new stage-2 code, JSON receipts | HF Trainer / Lightning / accelerate | receipt discipline over convenience; a single-GPU 170M job needs none of it; boring and inspectable |
| Curve metric | free-run match/lcp (sanctioned) + diagnostic-only NLL | terminal relL2; TF top-1 | both formally disqualified by stage 1 (ceiling report sections 2.3, 4.3); the audit's metric policy names free-run + manual review as the only rollout signals |
| Fold-cost accounting | two pre-registered Views, both always reported | charging fold cost inside the token budget; ignoring it | charging inside tokens destroys the init-value measurement; omitting it entirely would assume the savings — the exact failure the staged policy exists to prevent |
| Rungs | 10M/100M/1B as in-run checkpoints | separate runs per rung; adaptive early stopping | checkpoints cost nothing extra and keep one receipt per arm; adaptive stopping would put a numeric in a decision path |
| Control decode | identical topology incl. trainable routes + shared integer routing + explicit stage-2 admission list | training controls on raw target-tokenizer CE with ordinary `generate` | that would hand the reviewer two different decode surfaces and void the comparison (section 5.4) |
| Hardware | 2x RTX 3090 train, RTX 8000 generate/decode, P40s unused | using P40s for parallel arms | no bf16 on Pascal; idx 5 is occupied by another workload; commensurable GPU-hours require one device model for training |
| Seeds | 1 seed/arm + the pre-registered decisive replicate | 3 seeds/arm upfront | 3x cost (~280 GPU-h) before knowing whether any difference is near the margin; the replicate rule spends it only where it decides something |

---

## 12. Risks and gates

* **Touches guarded runtime code once:** the stage-2 admission path (section 5.4 item 3) is
  adjacent to stage-1 fresh-reload authentication. Gate: adversarial review + tests pinning stage-1
  refusal messages byte-identical. **Needs Cyril's explicit approval.**
* **Sealed artifacts:** all stage-1 run directories are read-only inputs (S2-A child state
  `54b9c5f7…`, S2-B child `f3f38cc3…`). Nothing writes into them; the trainer copies initial weights
  out and verifies digests before the first step. No backup is needed because nothing sealed is
  modified — and the initial-weight copies are themselves digest-verified in the receipts.
* **Disk:** `/home` at 93% is a standing hazard; the design writes nothing there beyond incidental
  logs. All artifacts go to `/mnt/ai_data` (1.8 T free). CIFS throughput may bottleneck shard
  reads; if the calibration run measures that it does, a bounded (< 20 GB) local staging cache on
  `/home` is permitted and logged.
* **Shared host:** P40 idx 5 carries another workload (23 GB in use); scheduling avoids all P40s
  and re-checks `nvidia-smi` before each wave. No production service is touched; no credentials are
  used anywhere in this experiment.
* **Teacher-sample quality:** a 230M teacher at T=1.0 emits some degenerate text; the 3-gram filter
  (section 2.3) bounds it and its statistics are manifested, so any corpus-quality objection is
  answerable from the manifest.
* **Review load is the critical path:** 5-7 reviewer-days. If it must shrink, the pre-registered
  cut is dropping the conditional 10M reviews — never the matched 100M/1B pairs.
* **The V5 surface must stay out of training:** enforced by the digest screen in section 2.3;
  verified by `tests/stage2/test_corpus_manifest.py`.
* **Expected-asymmetry spin risk:** S2-A vs S2-C will look spectacular and means little; the report
  template already carries the section-1 asymmetry paragraph so the S2-B/S2-D number stays the
  headline.

---

*Designed 2026-08-08 against HEAD `4e6d0ed6`. Read-only everywhere: the sealed run directories, the
server inventory, and the dataset estate were inspected without writing. This file is the only file
written. No git add, no commit.*

---

## AMENDMENT 1 — 2026-08-09: corpus arithmetic and throughput corrected

Measured on the RTX 8000 (NOTES-S2A.md): accepted throughput is 2,299
tok/s — the design's 20k tok/s estimate was 8.7x optimistic — so corpus
generation is ~65.5h wall (projected complete 2026-08-11 ~22:20Z), not
~15.5 GPU-h. The corpus target is corrected to 541,065,216 tokens (129
shards: 5 holdout generated FIRST + 124 train): by the design's own
epoch rules the three rungs draw 10M/55M/505M off-policy tokens (50%
on-policy after 10M), so 505M train + 20M holdout is the zero-repeat
floor for the full ladder — the original 1.12B figure was uncorrected
arithmetic, not a requirement. Holdout-first ordering makes any shard
boundary a valid sealed stopping point. One trial shard was discarded
for a seed-mix quota defect (archived, not deleted); the fix is
quota-enforced and injection-tested. The silent ~10-minute launcher
deaths (twice, not root-caused, hypothesis: SSH session-scope cgroup
teardown) are mitigated by a systemd transient unit with
Restart=on-failure and resume-by-missing-manifest.
