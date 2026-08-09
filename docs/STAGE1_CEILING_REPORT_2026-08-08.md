# SEALED STAGE-1 CEILING REPORT — fold-only, LFM2.5-230M, 2026-08-08

Authority: `docs/CURRENT_FOLDING_GOAL.md` §"Staged escalation policy" (commit
`6f276ade`, 2026-08-06). This is the document that policy names as the exit
condition for stage 1. It is written to be read against the sealed artifacts,
not instead of them: every number below carries its file, section, or digest.

---

## Executive summary

1. **Fold-only — zero gradients, closed-form solves on sealed cached evidence — retains zero measured behavior at and below 1.559x body contraction on LFM2.5-230M.** Three published children at three compactions, three frozen manual reviews, **0 of 15 useful-or-partial each**.
2. **The proven deliverable frontier is the A9/A11 dual-route configuration: 1.35x counted, 1.000x body, 10–12 of 16 useful-or-partial.** It is the only configuration this program has taken to the manual-review surface and gotten task behavior out of.
3. **The governing variable is BODY contraction, not counted compaction.** The 2.000x-counted child (1.559x body) scores identically to the 4.000x-counted child (3.469x body) — both zero — while the 1.35x-counted child at 1.000x body scores 10–12/16. Any ceiling quoted in counted-x without a body figure is uninterpretable (`NOTES-W13T.md` §1).
4. **The open bracket is (1.0x, 1.559x] body, and it is open on the left only.** Above 1.559x the interval carries no information: retention was already zero when it began (`reports/2x-review-20260808/TWOX_MANUAL_INTELLIGENCE_REVIEW.md` §8).
5. **Every named procedural remedy is exhausted**, each against a measured cause, each with a sealed artifact: reclosure (three policies), norm ceiling, solver ridge + prior-delta clamp, on-policy evidence, terminal floor + tripwire, EIV/margin, MLP width, depth, charts, rows.
6. **The MLP-contraction hypothesis is now experimentally closed, not just arithmetically.** The 2x child was given MLP widths `[2559, 2559, 2080, 1825, 1792, 2559, …]` against the parent's 2560 — layer-0 predicted capacity defect **0.0965 → 1.91e-05**, a ~5,000x reduction — and still scored 0/15 (verified this wave from the child's own `config.json`, sha `5c93a191…`).
7. **The solver was never the limit at layer 0 either.** The production closed-form solve reproduces the parent's layer-0 write to held-out cosine **0.9995** at r12's own width on r12's own evidence — 46x better than the shipped child (`NOTES-W13B.md` §3–§4).
8. **One law explains the collapse, and three independent production measurements state it:** closed-form corrections that improve local cached fit beyond prior magnitude compose *worse*. A12 replant (2026-07-30), the reclosure explosions (r12/W13M), and the R-arm (2026-08-08, refused at layer 0 in 32 minutes) are the same finding at three scales.
9. **Reconstruction error and behavior are decoupled below the cliff.** Best-ever terminal reconstruction and worst-ever behavior coexist: post-reclosure held-out terminal relL2 **0.4643** (arm A, 4x) is the program's best number on that field and its child emits nothing on half the suite.
10. **The terminal floor is formally disqualified as a behavioral proxy.** Same 14 installs, two reclosure policies, terminal 0.9711 vs 0.4643 — a 0.507 swing — and no behavioral difference that any review could find.
11. **Teacher-forced top-1 is anti-correlated with free-run across four children** (1.86% → 10.23% → 16.05% → 33.10% TF, while free-run went 0.44% → 0.102% → 0.034% → 0.195%), with the manual verdict pinned at 0/15 throughout.
12. **The platform is proven and is the real stage-1 asset**: exact counted-parameter compaction, fresh-reload authentication, sealed evidence, fail-closed floors, digest-pinned frozen review, and a receipt chain carrying `training_steps: 0`, `backward_calls: 0`, `optimizer_steps: 0`, `parent_model_forward_calls: 0` on every claim in this document.
13. **The any-architecture kernel estate is real and independent of the LFM2 result**: chimera-omni plans the real `k3_aligned_test_config` crossing with **zero gaps**, and a 24-action exact-4x crossing carrying a dense SiTU mixer beside gated-MLA and Stable-LatentMoE neighbours executes end to end; cross-family, cross-width and cross-kind folds are sealed against real pretrained checkpoints.
14. **Five refinement cells remain honestly open** (§6.1). None can plausibly move a zero to a nonzero; each is costed so Cyril can buy one anyway.
15. **Two instrumentation defects were never closed** (§6.2): the free-run tripwire has still never produced an observation (`evaluated: false`, 14/14 layers, both arms), and the operator branch's prior-correction reports are still computed and discarded.
16. **Nothing was weakened to reach this conclusion.** No flag relaxed, no gate bypassed, no tolerance moved, no sealed directory written. The R-arm was allowed to fail closed and stay failed.
17. **Stage 1's conditions, as stage 1 defined them, are met** — modulo §6.1's optional refinements, which are refinements and not conditions.
18. **Stage 2 is unlocked.** The first measurement it should take is fold-as-initialization at two points — the proven 1.35x/1.000x-body frontier and the 1.559x-body cliff — each plus minimal recovery distillation, each against a distillation-from-scratch control at matched compute.
19. **That control is the entire point of the staged design**: without it, the fold's initialization value is assumed rather than measured, which is the failure mode the policy was written to prevent.
20. **This is not a defeat.** It is the exit condition Cyril defined on 2026-08-06, reached with sealed evidence, on schedule, with the negative result stated as precisely as a positive one would have been.

---

## Item-mapping to `STAGE_A_ATTRIBUTION.md` §4.3

`reports/stage1-attribution-20260808/STAGE_A_ATTRIBUTION.md` §4.3 enumerates
seven required contents. Where each lives here:

| §4.3 item | section |
|---|---|
| 1. Best composed retention per compaction level, with manual-review verdict and the frozen V5 digest | §2 |
| 2. The exhaustion ledger, including the three negative results the attribution added | §3 |
| 3. The novel assets stated as results | §5 |
| 4. The zero-training provenance chain with receipt digests | §2.5 |
| 5. The field comparison (2026-08-06 taxonomy) | §4.4 |
| 6. The explicit stage-2 authorization sentence | §7 |
| 7. Open instrumentation defects, so the ceiling is not over-claimed | §6.2 |

---

## 1. The claim being sealed

**Measured fact.** Under the fold-only contract of `docs/CURRENT_FOLDING_GOAL.md`
— zero gradients, zero optimizer steps, zero parent forwards at fold time,
closed-form analytic solves against sealed cached parent evidence — at the
tested scale (parent `LFM2.5-230M`, 229,693,184 unique parameters):

> **Behavioral retention collapses to zero at or below 1.559x BODY contraction,
> with every named procedural remedy exhausted.**

Stated with its bracket, because the bracket is the honest part:

* **Proven positive:** 1.000x body (A9/A11 dual-route, 1.35x counted) retains
  **10–12 of 16** useful-or-partial observations.
* **Proven zero:** 1.559x body (2.000x counted) — **0 of 15**.
  3.469x body (4.000x counted) — **0 of 15**, twice, under two different
  reclosure policies.
* **Open interval: (1.0x, 1.559x] body.** No child has been folded and reviewed
  strictly inside it. Every claim in this report about what happens there is
  labelled as inference, not measurement.
* **Closed interval: [1.559x, ∞) body.** Carries no further information.
  Retention was already zero at its left endpoint.

**The proven deliverable frontier**, i.e. the configuration stage 2 inherits as
a working artifact rather than as a failure:

| | |
|---|---|
| Configuration | A9 same-body k32 residual-OMP route; A11 split output route |
| Counted compaction | **1.361447x** (A9), **1.346018x** (A11) |
| Body contraction | **1.000x** — the body is not contracted at all |
| Manual review | A9: 3 useful / 7 partial / 6 miss. A11: 3 useful / 9 partial / 4 miss. **10–12 of 16 useful-or-partial** |
| Source | `docs/CURRENT_FOLDING_GOAL.md` §Current evidence; `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md` §"A9/A11 corrected same-body review"; `reports/sparse-semantic-a11-k32-split-cout-20260730/manual-review.md` |

**What "fold-only" means here, checked rather than asserted.** Every child in
§2 carries, in its own receipt: `training_steps: 0`, `parameter_gradients_written: false`,
`backward_calls: 0`, `optimizer_steps: 0`, `parent_model_forward_calls: 0`.
The digests are in §2.5.

**What this claim does not say.** It does not say folding is impossible at
1.2x body. It does not say a different parent, a different family, or a
different scale behaves this way — every number here is LFM2.5-230M. It does
not say the closed-form machinery is wrong; §5 argues the opposite. It says
that at this scale, on this parent, with this evidence, every fold-only lever
this program built has been pulled and the behavioral needle has not moved off
zero anywhere the body was contracted.

---

## 2. The retention curve

### 2.1 The curve

Frozen protocol on every V5 verdict below:
`reports/r12-certified-20260804/V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md`, digest
**`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`**, sealed
by `reports/r12-certified-20260804/PROTOCOL_SEAL.json`
(`neural-forge-manual-review-protocol-seal-v1`, 2026-08-05T19:30:00Z) and
recomputed byte-identical by each reviewer. No cached-frame metric appears in
any acceptance path in any row.

| # | configuration | counted ratio | **BODY ratio** | terminal composition vs the 1.0 emit-zero floor | frozen review | receipt / review digests |
|---|---|---|---|---|---|---|
| 1 | **A9 / A11 dual-route** (same body, token interface only) | 1.361447x / 1.346018x | **1.000x** | not measured — the terminal floor postdates this lane | **3 useful + 7 partial (A9)**, **3 useful + 9 partial (A11)** of 16 — pre-V5 policy, see §2.4 | publication `complete.json` artifact `860bccfe6be46e543f67c8514f3f3ff7dc0d729eb82267f66a9ce1766e43d7bb`, completion `55a550aaf48189f9989d799b49980f7d61b59134d4f4aa8edc4cfa7e3c84bfb3`; reviews `reports/sparse-semantic-a9-…/manual-review.md`, `reports/sparse-semantic-a11-…/manual-review.md`, high-cap `reports/lfm2-sparse-semantic-highcap-20260730/manual-whole-response-review.md` |
| 2 | **r12** — first certified exact-4x child, reclosure **unguarded** | 4.000220x | **3.469006x** | **4.4728** at L13 — 4.5x *worse* than emitting zeros; no floor existed to stop it (`NOTES-W13A.md` L204) | **0 useful / 0 partial / 15 miss** | `receipt_sha256` `40e6dc722e488ae07a497bcbc75a509cb441df43c4ce114caf4393472f52dbf0`; file `33e96e6c7863a8753de0704afb5838339950f094de2305cc26706f57f6d3e469`; certificate `046272ab12e876a31d70b4ca1bacf9f8a1756441772e9983b1384460277cbc9c`; review `reports/r12-certified-20260804/MANUAL_INTELLIGENCE_REVIEW.md` |
| 3 | **arm A** — r13 refold, reclosure **defanged** by norm-ratio ceiling 2.0, first complete receipt of the composed lane | 4.000220x | **3.469006x** | **0.806258** at L13 (cos 0.6973, 13,464 rows); post-reclosure held-out terminal **0.464258** | **0 useful / 0 partial / 15 miss** | `receipt_sha256` `1cdec0a61cd94712bc3b26fffb26dce970291acd5b21817f970a030ad3f38168`; file `45bc68c2e8766e030516f6bccbe7ec2142c2af229e26454e44f3cfa71ecd4f10`; dossier `409de3ed809a75dce4905966cc659c9f6ef06562fae6f83ad0356e3bf980aa39`; review `reports/armA-review-20260808/ARMA_MANUAL_INTELLIGENCE_REVIEW.md` |
| 4 | **R-arm** — arm A + relative ridge λ=1e-5 (R1) + body prior-delta ratio 8.0 (R2) | 4.000220x | **3.469006x** | **1.00576305 at layer 0 — at or above the no-branch baseline of 1.0. REFUSED.** Run failed closed 31m52s after start | **none — no child, no publication, no dossier** | `receipt_sha256` `42bc21c308691b831ed6f294f1fa75f2a206629e09c5123d4238aca800164873`; file `5d61ae20913810f06324780e966d11522758352552d461312c549571ef40a1e7`; `NOTES-W13S.md` §6 |
| 5 | **2x ceiling arm** (a0028) — the cliff-bracketing measurement | **2.000000x** | **1.558902x** | **0.581300** best at L12, **0.608994** at L13 (cos 0.8294), post-reclosure held-out terminal **0.568747** | **0 useful / 0 partial / 15 miss** | `receipt_sha256` `a218501f4b91e98ac2e3e9567684c86c60e194e39fc0bdece76222395eff588c`; file `b1b8203e787dac1535e355b8a50fcbdb99df55bba8f730c6bda02dd102369ba6`; dossier `28981b2b54946ce01d61c87a620e02c3189b99c489f850c71946e405ce4ca6f4`; review `reports/2x-review-20260808/TWOX_MANUAL_INTELLIGENCE_REVIEW.md` |

Supporting row, not a retention point because it has no review: the **first
composed r13 run** (`…-refold.sealed-w13m-firstcomposed`, receipt
`c9a8f7468934f55a97f37cd14e7a7fb91721657dfa2f54f4218d9a962d98e4c9`) produced
the identical 14-layer install sequence as arm A and then let an unguarded
(gate-inert) reclosure carry the terminal metric to **0.971148**. It published
its child and failed closed 55 ms later on a stale-count assertion, so its
dossier never ran (`NOTES-W13M.md` §1.3, §4).

**Body-ratio arithmetic, recomputed here rather than taken on report.**
Parent body = 229,693,184 − (65,536 × 1,024 tied table) = **162,584,320**.
Child body = counted dense body − (2 × 4,096 × 816 untied tables = 6,684,672).

```
A9 / A11   body 162,584,320                      -> 1.000000x   (uncontracted)
2x  (a0028) body 110,978,816 - 6,684,672 = 104,294,144 -> 1.558902x
4x  (armA)  body  53,552,368 - 6,684,672 =  46,867,696 -> 3.469006x
```

The 2x child retains **64.15%** of the parent's body parameters; the 4x child
retains **28.83%**. Method and the A9/A11 reconstruction that validates it:
`NOTES-W13T.md` §1–§2; the 2x review recomputed 1.55890 independently
(`TWOX_MANUAL_INTELLIGENCE_REVIEW.md` §2).

### 2.2 The per-layer terminal series, arm A against the 2x arm

Both columns read the same sealed field
(`fold.closed_loop_depth_telemetry.per_layer_validation_decisions[*].metrics.terminal_relative_rmse`),
13,464 held-out rows, `terminal_no_branch_relative_rmse_baseline = 1.0` in both.

| L | arm A (3.469x body) | cos | **2x arm (1.559x body)** | cos |
|---|---|---|---|---|
| pre-install baseline | 1.0022687 | — | **0.7538707** | — |
| 0 | 0.969881 | 0.3850 | **0.760646** | 0.7340 |
| 1 | 0.932394 | 0.4572 | 0.772927 | 0.7288 |
| 2 | 0.913080 | 0.4849 | 0.733284 | 0.7427 |
| 3 | 0.889468 | 0.5191 | 0.746995 | 0.7435 |
| 4 | 0.905411 | 0.4867 | 0.730210 | 0.7578 |
| 5 | 0.885287 | 0.5164 | 0.678845 | 0.7753 |
| 6 | 0.840180 | 0.5812 | 0.643807 | 0.7883 |
| 7 | 0.847366 | 0.5702 | 0.693995 | 0.7641 |
| 8 | 0.809602 | 0.6187 | 0.635515 | 0.7931 |
| 9 | 0.791426 | 0.6321 | 0.668263 | 0.7523 |
| 10 | 0.789816 | 0.6245 | 0.625211 | 0.7916 |
| 11 | 0.721728 | 0.7045 | 0.695240 | 0.7170 |
| 12 | 0.664374 | 0.7652 | **0.581300** | 0.8166 |
| 13 | **0.806258** | 0.6973 | **0.608994** | 0.8294 |
| post-body-reclosure | **0.464258** | — | **0.568747** | 0.8603 |

Two readings that matter more than the numbers:

1. **The 2x arm's pre-install baseline is 0.754; arm A's is 1.002.** The 4x
   child's 14-layer composition was *already worse than emitting zeros before a
   single body install*. Arm A spent fourteen installs clawing back toward a
   line it started on the wrong side of; the 2x arm started below it
   (`NOTES-W13V.md` §6.2a).
2. **Neither arm improves monotonically.** Arm A's per-install regression is
   negative at every one of the first four layers; the 2x arm's is
   `+0.0068, +0.0123, −0.0396, +0.0137` — it oscillates on a lower plateau.
   A claim of monotone improvement at 2x would be false (`NOTES-W13V.md` §6.2c).

### 2.3 The number that disqualifies the terminal metric, stated once with both values side by side

The same field, `body_reclosure_validation.metrics.heldout_terminal_relative_rmse`,
same construction (`actual_full_child_recaptured: true`, 51 validation record
keys, identical `reason` string), on three runs that share a byte-identical
14-layer install sequence up to the reclosure:

| run | reclosure policy | terminal after L13 | **post-reclosure held-out terminal** | frozen review |
|---|---|---|---|---|
| r13 first composed (W13M) | unguarded; boundary gate refused 0 of 82 rewrites | 0.806258 | **0.971148** | none (dossier lost) |
| **arm A** | norm-ratio ceiling 2.0; 27 of 28 overwrites refused | 0.806258 | **0.464258** | **0/15** |
| **2x arm** | ceiling 2.0 + the ported A12 rule live; 26 of 28 refused | 0.608994 | **0.568747** | **0/15** |

**A 0.507 swing on the composed terminal metric, produced by the reclosure
policy alone on an identical fold, with no behavioral difference any review
could find.** And the program's *best* number on this field — **0.464258** —
belongs to the 4x child that emitted zero characters on eight of sixteen
prompts.

**Correction recorded, in the direction that costs us.** The 2x review states
that its 0.5687 is "by a wide margin the best terminal reconstruction this
program has produced — arm A finished at 0.8063"
(`TWOX_MANUAL_INTELLIGENCE_REVIEW.md` §7). That comparison is against arm A's
*layer-13 install-time* terminal, not against arm A's value on the same field.
On the same field arm A is **better** (0.4643 vs 0.5687). The 2x review's
conclusion — that the metric carries no behavioral information — is not merely
unharmed by this, it is strengthened: the two metrics rank the two children in
*opposite orders* and both children are 0/15. The estimator is the same in both
places: on the 2x arm, `body_reclosure_terminal_baseline_relative_rmse`
(0.6089942756285601) equals its layer-13 install terminal to the last digit, so
the two readings are commensurable. Arm A's block predates the A12-port
instrumentation and therefore carries no baseline of its own, which is why this
number was never surfaced by the arm-A review.

### 2.4 Honest caveats on the frontier row

* **A9/A11 were never scored under the frozen V5 protocol.** Their 10–12/16 was
  graded 2026-07-30 under `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`
  on the same 16-prompt surface but with a **256-donor-token cap**, against V5's
  1,024 initial / 2,048 retry. The high-cap replay
  (`reports/lfm2-sparse-semantic-highcap-20260730/`) re-ran only the censored
  rows at V5's caps and did not re-tally the suite; it resolved A11's six cap
  hits at 1,024 and A9's five of seven, and its own summary is that "neither
  lane demonstrates approximately 90% intelligence retention". Re-scoring A9/A11
  end-to-end under V5 is a review-only, zero-compute open cell (§6.1e).
* **No parent comparison exists on the V5 surface for either 4x or 2x.** Both
  reviews record `parent-not-demonstrated`
  (`ARMA_…` §6, `TWOX_…` §6). That is a gap in the evidence, not a finding
  about the parent.
* **The R-arm row is a refusal, not a retention measurement.** It has no child.
  It is in the curve because a floor refusal at layer 0 is a stronger result
  than a clean pass would have been, and because it is one of the three
  independent statements of the physics in §4.
* **Cap censoring is not failure.** 2x: 8 of 16 rows censored at 2,048; arm A:
  3 of 16. Neither review categorized any case on a censored endpoint alone;
  each censored row carries a prefix that independently proves collapse.

### 2.5 Zero-training provenance chain (§4.3 item 4)

Read from each run's own sealed receipt, not from a summary:

| run | `training_steps` | `backward_calls` | `optimizer_steps` | parent forwards at fold time | receipt digest |
|---|---|---|---|---|---|
| r12 | 0 | 0 | 0 | 0 | `40e6dc722e488ae07a497bcbc75a509cb441df43c4ce114caf4393472f52dbf0` |
| arm A | 0 | 0 | 0 | 0 | `1cdec0a61cd94712bc3b26fffb26dce970291acd5b21817f970a030ad3f38168` |
| R-arm | 0 | 0 | 0 | 0 | `42bc21c308691b831ed6f294f1fa75f2a206629e09c5123d4238aca800164873` |
| 2x arm | 0 | 0 | 0 | 0 | `a218501f4b91e98ac2e3e9567684c86c60e194e39fc0bdece76222395eff588c` |
| A9/A11 publication | 0 | 0 | 0 | 0 (`parent_trace_recaptures: 0`, `runtime_sidecar_parameters: 0`) | artifact `860bccfe6be46e543f67c8514f3f3ff7dc0d729eb82267f66a9ce1766e43d7bb` |

r12's certificate additionally carries `compilation_backward_calls: 0`,
`compilation_optimizer_steps: 0`, `fresh_reload_backward_calls: 0`,
`all_reloaded_parameters_frozen_during_behavior_callback: true`,
`forbidden_named_sidecars_present: false`
(`reports/r12-certified-20260804/certificate.json`). The layer-0
representability study, the Stage-A attribution and this report performed zero
model calls of any kind.

---

## 3. The remedy ledger — every lever, its verdict, its evidence

Format: lever → what it was aimed at → what it measured → verdict → sealed
artifact. "Closed" means: do not spend another run on it.

### 3.1 Reclosure — three policies, three outcomes, one conviction

| policy | measurement | verdict |
|---|---|---|
| **Unguarded** (r12, and r13/W13M with an inert gate) | r12: layer-0 write cosine 0.911 → **0.171**, writer Frobenius norm **×4.73**, `in_proj` byte-identical. W13M: **82** weights rewritten, **81 upward**, median ratio **33.147x**, 64 of 82 above 10x, **974.144x at layer 7**; terminal 0.806258 → **0.971148** (0.165 worse) and accepted, because the non-regression tolerance guarded only layer installs | **Destroyed the child.** Convicted. `NOTES-W13B.md` §6, `NOTES-W13M.md` §1.2–§1.3, `docs/OVERLAP_AUDIT_2026-08-07.md` Suspicion 1 |
| **Gated on the boundary's own cached fit metric** (`require_boundary_improvement`, W13C → r13) | The gate **refused 0 of 82** rewrites. A writer can grow 974x and still cut its own boundary's write error, because that is not what the metric measures | **Structurally inert.** The gate was built on the metric class A12 had already sealed as non-predictive. Cost: a 16h34m run carrying a gate that could not fire. `docs/OVERLAP_AUDIT_2026-08-07.md` Suspicion 1(2) |
| **Defanged by a norm-ratio ceiling of 2.0** (arm A, 2x arm) | arm A: **27 of 28** overwrites refused, all 14 transitions refused at least once, max accepted ratio **1.5007**; recovered the 0.165 terminal regression. 2x: **26 of 28** refused, max accepted **1.5646**, two accepted at boundary.001 and boundary.013 | **Neutral for behavior.** Removed a real defect, changed the failure's signature, produced **0/15** anyway. Necessary, not sufficient. `ARMA_…` §2, §8; `TWOX_…` §2 |
| **The A12 collateral-regression rule, ported** (commit `38cc3bc9`) | First exercised in the 2x arm: `body_reclosure_terminal_non_regression_is_the_ported_a12_rule: true`, `…has_decision_authority: true`, regression **−0.040247** under tolerance 0.25, **passed** | **Shipped and live.** The oldest open regression in the audit is closed. It did not change behavior, because it was never a behavior remedy — it is a guard |

**Ledger note.** The reclosure conviction is a genuine asset (§5.2) and a
genuine cost: the lesson was sealed at the token interface on 2026-07-30 and
re-purchased at the body twice, for 95.2 h (r12) plus 16.5 h (r13).

### 3.2 Norm ceiling

Aimed at: bounding the reclosure step. Measured: arm A max accepted write-norm
ratio 1.5007 under a 2.0 ceiling with 27/28 refused; 2x arm 1.5646 with 26/28
refused; `boundary_acceptance_sha256` `1e789a1e…` (arm A), `01297f2b…` (2x).
**Verdict: works exactly as designed as a guard, recovers metric, does not
recover behavior.** It is the crude form of W12B's trust region and it is the
correct reading of the A3 "energy without direction" lesson: it refuses rather
than claiming to fix direction.

### 3.3 Solver regularization and step bound — R1 + R2 (the last never-tried lever)

**The attributed cause** (`STAGE_A_ATTRIBUTION.md` §2.1): every one of arm A's
**42** body MLP solves (14 layers × {gate, up, down}) hit the hard-coded
`max_prior_delta_ratio = 1.0` clamp. The clamp identity
`correction_norm == 1.0 × prior_norm` holds to a worst relative error of
**1.46e-16** across all 42. The solver asked for **1.446x to 140.4x** the
prior's norm (median ≈ 37x) and was allowed 1.0x, because `ridge = 1e-6` is
*absolute* against a `XᵀX` whose σ_max is 1.4e5–7.1e5 — a relative ridge of
**1.4e-12 to 7.3e-12**, and at layer 5 the ridge supplies **76%** of the
smallest singular value of the system being inverted.

**The remedy, executed.** The R-arm (`NOTES-W13S.md`) shipped R1 (relative ridge
λ = 1e-5), R2 (body prior-delta ratio 8.0) and R3 (telemetry) on top of arm A's
*exact* composition — same parent, same corpus, same evidence set `f62110d7…`,
same inherited basis `52cbe3fd…`, same ceiling, same floor, same tolerances,
same row budgets, same devices. One variable.

**The result.** The per-boundary clause **passed** — the local solve did improve,
exactly as designed. The terminal composition floor **refused** the install:

```
layer:000000 validation rejected the folded child: held-out post-layer residual
improved/non-regressed; terminal composed held-out relative RMSE 1.00576305 is
at or above the no-branch baseline 1
```

Arm A's same layer: **0.96988110**. **+0.03588 absolute, and across the line.**

**Verdict: floor-refused. A better-conditioned, up-to-eight-times-larger body
correction improved the local boundary and made the layer-0 composition worse
than emitting nothing.** This is the last fold-only lever aimed at a measured
cause, and it failed in the direction the physics predicts (§4).

Not measured, and not claimed: λ = 1e-4, ratio 2.0, a λ-only control arm, or any
layer past 0 (`NOTES-W13S.md` §6.2). See §6.1b.

### 3.4 On-policy evidence extension

Aimed at: exposure bias — the child's own visited prefixes are off the parent's
cached trajectory. Consumed cleanly in arm A, verified structurally from its
receipt: `on_policy_extension_evidence_consumed: true`, `round_count: 1`,
`on_policy_extension_record_count: 115`, `on_policy_merged_paired_record_count: 371`,
`on_policy_parent_capture_invocations: 0`, evidence identity
`75217282d5f3d7ca397abcc14ba645d118c7833d93c12a3ebe25aecd381067e5`,
2-member evidence set `f62110d7…`.

**Verdict: consumed, contract-clean, no rescue.** Arm A is the *unconfounded*
read the audit demanded (r13's item-1 read was confounded by an exploded
reclosure sitting downstream — `docs/OVERLAP_AUDIT_2026-08-07.md` Suspicion 3).
Arm A consumed on-policy evidence with the reclosure defanged and scored 0/15.
The 2x arm deliberately dropped the round (its rollouts are the 4x child's and
are off-policy for a 2x child), returning to a 1-member base-only evidence set,
and scored 0/15 as well. On-policy evidence changes nothing on either side of
the cliff.

### 3.5 Terminal composition floor and free-run tripwire

**Floor — works, as a wreck detector.** r12 finished at 4.4728 with no floor to
stop it; the floor would have aborted it at layer 0. r13/arm A finished under
the emit-zero baseline at every layer for the first time. The R-arm was refused
by it in 12 minutes of closed-loop time. It is `fail-closed`, it has authority
even under `--dense-closed-loop-performance-metrics-diagnostic-only`
(`terminal_floor_retains_authority_under_diagnostic_only: true`), and it caught
a real regression the same week it shipped.

**Floor — formally disqualified as a behavioral proxy.** §2.3. Two children at
0.464 and 0.569 on the same field, both 0/15; a 0.507 swing from reclosure
policy with no behavioral consequence. The 2x result is the sharpest available
confirmation: cutting terminal error 24–29% relative to arm A bought **exactly
zero** anchored elements. Floor passage is never progress
(`docs/CURRENT_FOLDING_GOAL.md` §"Closed-path additions", clause 5).

**Tripwire — built, wired, and still blind.** See §6.2.

### 3.6 EIV / margin weights

Aimed at: robustness of the boundary solve to input error, and row weighting by
vocabulary margin. **Verdict: structural no-op at the failing sites.** The EIV
ridge's covariance is `None` until a layer is accepted, so it never reaches the
layer-0 operator write, and layer 0 has no input error to be robust to (measured
child-vs-parent layer-0 input drift **2.53e-08**, cosine 1.0). The margin weights
land in the reclosure manifest, not in the boundary solve. It is correctly OFF
(`--dense-closed-loop-input-error-multiplier 0`) in every arm in §2, i.e. every
arm is the EIV control arm. `NOTES-W13B.md` §5, §1; `docs/OVERLAP_AUDIT_2026-08-07.md`
Suspicion 3 (EIV/margin).

### 3.7 MLP width — arithmetically closed at 4x, and now experimentally refuted at 2x

**Arithmetic (4x).** From `STAGE_A_ATTRIBUTION.md` §4.2: the operator stack owns
**66.88%** of the 53,552,368 exact body budget (short_conv 23,910,400 + attention
11,907,840 = 35,818,240); the 14 gated MLPs own 11,025,792 (20.59%). One
intermediate unit costs 3 × 816 = 2,448 parameters. Reaching the uniform 1280
rung costs **+32,842,368 parameters = 61.3% of the entire budget**, against an
operator stack of 35.8M; 1920 costs 102% of budget, 2559 costs 143%. The absolute
arithmetic ceiling — every non-interface, non-norm parameter to the MLP and
literally nothing to the operators — is width 1,366. **There is no fold-only
remedy for the 2560 → 320 contraction at exact-4x.**

**Experiment (2x) — new in this report, and decisive.** At the 2x budget the
solver *did* buy the width, and the sealed child proves it. Verified this wave
by reading both published children's own `config.json` on the AI server,
read-only:

| | arm A (4x) | **2x arm** | parent |
|---|---|---|---|
| `config.json` sha256 | `d2cb79c34bba34344681f0b086177249a44e2f4da60aac37dde0bd21ddd936d1` | `5c93a191765f4d30564afd907eba7618b458a4356ba3ac85dff617778e1ae0c7` | — |
| per-layer MLP widths | `[321,323,320,330,320,330,320,320,320,320,320,320,320,320]` | **`[2559,2559,2080,1825,1792,2559,1280,1280,1567,1280,1567,1824,1567,1791]`** | 2560 × 14 |
| total MLP units | 4,504 (**12.6%** of parent) | **25,530 (71.2% of parent)** | 35,840 |
| MLP parameters | 11,025,792 (20.6% of body budget) | **62,497,440 (56.3%)** | — |
| short-conv internal widths | `[1024,1024,–,1024,–,1024,–,1024,–,704,–,704,–,1024]` | **all 1024 — parent width everywhere** | 1024 |
| layer-0 post-SwiGLU tail fraction | 0.6953 | **0.000108** | 0 |
| layer-0 collision fraction | 0.8746 | **0.000391** | 0 |
| layer-0 predicted capacity defect | 0.09645 | **1.91e-05** | 0 |
| frozen review | **0/15** | **0/15** | — |

The layer-0 predicted defect fell by a factor of **5,049**, the "~70%
unrepresentable post-activation tail" fell to **0.011%**, every short-conv kept
the parent's full internal width, and the manual verdict did not move off zero.
Capacity-defect tables read from each run's own
`planned_target.heterogeneous_capacity_option_bank.layer_evidence[0].mlp_capacity_risks`;
the option-bank digests are `809e4f60…` (arm A) and `8b0a3869…` (2x).

The planner agreed in advance: the 2x layout's `combined_predicted_risk` is
**0.0786** against arm A's **0.2157**, and `mean_selected_component_risk`
**0.0546** against **0.1916** — a 3.5x lower predicted risk, same behavioral
result.

**Verdict: closed, twice over.** The MLP contraction is a real capacity defect,
it is unrelievable at 4x, and relieving it almost entirely at 2x changed
nothing. It was never the cause.

### 3.8 Depth ladder

Aimed at: locating the wall by folding 3, 6, 10 layers. **Verdict: superseded
and closed.** W13A located the wall at **layer 0** — held-out post-layer relL2
1.159 against the 1.0 emit-zero baseline, 93% irreducible direction error, and a
layer-0 short-conv write at cosine 0.171. "A depth-ladder probe at 3/6/10 would
have measured a corpse three times" (`NOTES-W13A.md` L402). Independently, the
Stage-A profile shows per-layer error *improving* with depth
(ρ(depth, post-layer relL2) = **−0.767**; the scale-free cosine agrees), so depth
is not the error axis: the worst boundaries are L2/L3 and the two most singular
layers L4/L5 are on the improving side.

The cross-family lane found a *different* depth wall on a real pretrained parent
(SmolLM2-135M → blank qwen3 refuses at layer 3's attention branch, while the
random-weight control does not — `reports/cross-family-real-20260804-pretrained/README.md`
§"The depth wall"). That is a real result about pretrained parents; it is not a
lever for LFM2 layer 0.

### 3.9 Charts and rows — exonerated

* **Row subsampling: refuted decisively.** All 42 arm-A body solves used exactly
  **60,525** rows against input widths 816 and 320–330 — **74x** and **~190x**
  over-determined — with `underdetermined: false` everywhere and an *identical*
  row count at every depth. `--max-fit-rows 40656` caps the evidence partition's
  fit-path count, not the activation rows a solve consumes.
  `STAGE_A_ATTRIBUTION.md` §2.3.
* **Solver path: never fell back.** `solver_path: primal_solve` and
  `silent_gelsd_fallback_taken: false` at all 14 layers.
* **Chart identity: exonerated.** The width ladder in `NOTES-W13B.md` §4 extends
  the sealed r12 chart rung by rung (each rung strictly contains the one below;
  the bottom rung is bit-identical to the sealed chart), so width is isolated
  from chart identity. Widening 816 → 1023 — surrendering essentially the entire
  4x budget — improves the layer-0 write relL2 from 0.0245 to 0.0031, a factor
  of 7.5 on a quantity that was already **46x better than the failure**.

### 3.10 Layer-0 solve quality — representability was never the limit

`NOTES-W13B.md` §3: at r12's exact width, on r12's exact sealed evidence, through
r12's exact production compilers, the closed-form solve reproduces the parent's
layer-0 write to

| | write cosine | write relL2 | composed post-operator relL2 | held-out rows beating the emit-zero baseline |
|---|---|---|---|---|
| **this solve** | **0.99953** | **0.02452** | **0.01807** | **100.0%** |
| r12's published child | 0.1712 | 1.0986 | 0.8358 | 9.5% |
| ratio | — | **44.8x** | **46.3x** | — |

and **thirteen solver arms land on the same number**: prior-delta clamped,
unclamped, 4.0; source-derived, dense-chart, zero, and the r12 capsule's own
weights as prior; ridge 0 / 1e-8 / 1e-6 / 1e-4 / 1e-2; fp32 storage; a second
independent process. The R13-era prediction for this layout was a per-layer
defect of 0.0965; the solve lands at 0.0181, **5.3x better than predicted**.

**Verdict: the target is representable, the solver can hit it, and no
hyper-parameter is the limit.** The deficit is in *composition*, not in
capacity and not in the solve.

### 3.11 Ledger summary

| lever | verdict | closed? |
|---|---|---|
| Reclosure, unguarded | destroys the child | **closed** |
| Reclosure, gated on cached boundary fit | structurally inert | **closed** (and a standing do-NOT) |
| Reclosure, defanged by norm ceiling | neutral for behavior; keep as a guard | **closed as a remedy** |
| A12 collateral-regression rule | ported, live, passing | shipped |
| Solver relative ridge + prior-delta release (R1+R2) | floor-refused at layer 0 | **closed at λ=1e-5 / ratio 8.0**; §6.1b |
| On-policy evidence rounds | consumed cleanly, no rescue, on both sides of the cliff | **closed at round 1**; §6.1e |
| Terminal floor | wreck detector; disqualified as a proxy | keep as a guard |
| Free-run tripwire | never produced an observation | **open defect**, §6.2 |
| EIV / margin | structural no-op at the failing sites | **closed** |
| MLP width | arithmetically closed at 4x; experimentally refuted at 2x | **closed** |
| Depth ladder | wall is layer 0; error improves with depth | **closed** |
| Charts / rows / solver path | exonerated (60,525 rows, 74x over-determined, no fallback) | **closed** |
| Layer-0 representability | 0.9995 cosine achievable; 46x headroom never claimed by a production run | **closed as a suspect** |
| Layout arms (global map, historical budgets, phase-key pinning) | closed before this wave | **closed** |

**After this ledger, the list of fold-only remedies aimed at a measured cause is
empty.**

---

## 4. The physics, stated once

### 4.1 The law

> **A closed-form correction that improves local cached fit beyond the magnitude
> of its prior composes worse than the prior did.**

Three independent production measurements, at three scales, on three different
objects:

| # | measurement | local fit | composed result |
|---|---|---|---|
| 1 | **A12 carrier replant** (2026-07-30, token interface) | co-adapted rewrites improved their own cached objective; folded lanes retain 80–91% of cached parent argmax winners | the same children are **0/16 coherent**. Sealed verbatim: `coadapted_cached_fit_improvement_did_not_predict_autoregressive_coherence: true` (`reports/a12-carrier-cached-delta-audit-20260730/report.json`) |
| 2 | **Reclosure explosions** (r12 body; r13/W13M) | every one of 82 rewrites cut its own boundary's cached write error — that is what the gate measured, and it refused none | writer norms went up **median 33.1x**, to **974x**; layer-0 write cosine 0.911 → 0.171; terminal 0.806258 → **0.971148** |
| 3 | **The R-arm** (2026-08-08, body MLP solves) | the per-boundary counterfactual clause **passed**: "held-out post-layer residual improved/non-regressed" | terminal composed held-out relL2 **1.00576305 ≥ 1.0** — worse than emitting nothing. Refused. Arm A's same layer, same everything else: 0.96988110 |

W12B's toy predicted all three before any of them: `pinv(J)` requested a **40x**
step for a 1% local gain and was **6x worse held out**; the unregularized
reclosure solve blew up **+515%** and needed a trust region
(`recaptured_step_ridge_ratio = 1e-2`). The production body reclosure then ran in
r12 and r13 with no trust region, no step bound and no norm constraint at all.

**The mechanism, in one chain** (`STAGE_A_ATTRIBUTION.md` §2.1):

> the child's own post-operator activations are numerically rank-deficient in the
> 816-wide chart at depth → the normal system is near-singular → an effectively
> unregularized least-squares solution has an enormous norm → the hard 1.0×prior
> clamp truncates it to a unit-relative step → the installed MLP is the
> structural-transport prior plus 1–43% of what the evidence asked for.

Release the clamp with proper conditioning (the R-arm) and the composition
crosses the emit-zero line. Keep the clamp (arm A) and the fold is barely
engaged: mean whole-layer improvement **+0.0162 relL2** against boundaries
sitting at 0.78–1.39.

### 4.2 The addendum the 2x arm added: reconstruction and behavior are decoupled below the cliff

The 2x child produced the program's best install-time terminal composition
(0.5813 at L12, 0.6090 at L13, terminal mean cosine **0.8603** post-reclosure,
against a 1.0 emit-zero baseline) — and carries **zero task content**. Not one
anchored element appears anywhere in the suite, checked by literal substring
search across all sixteen decoded responses: no `Mira`/`Theo`/`Jun`/`Sal`, no
`102`/`94`/`18`, no `Saturday`/`4:15`/`north`, no `Supplier`, no `RAM`, no
`def`/`return`/`O(`, no `tilt`/`axis`/`orbit`, and not one question mark offered
as a follow-up (`TWOX_…` §5).

The reviewer's own summary is the right sentence and it is worth quoting rather
than paraphrasing: **"the model has recovered the register of answering without
recovering the ability to answer."** It opens four responses with first-person
answer-shaped clauses, uses topically adjacent vocabulary (`email` on the email
case, retrieval verbs on the retrieval case), and then decays into token cycling,
asterisk masks (one character in ten across the suite is `*`) and 137 non-ASCII
characters of cross-script debris.

The review explicitly refused the available inflation — grading three of those
fluent-onset rows Partial would have produced a 3/15 headline the text cannot
support — and recorded the refusal. That refusal is why the number in §1 can be
trusted.

### 4.3 The TF / free-run anti-correlation

Prior history showed teacher-forced metrics *overstating* free-run. Four
children now show them **anti-correlated**:

| child | body ratio | teacher-forced top-1 | free-run token match | frozen review |
|---|---|---|---|---|
| r12 | 3.469x | 1.86% | **0.440%** | 0/15 |
| r13 first composed | 3.469x | 10.23% | 0.102% | none |
| arm A | 3.469x | 16.05% | **0.034%** | 0/15 |
| 2x arm | 1.559x | **33.10%** | 0.195% | 0/15 |
| A11 | 1.000x | 90.28% cached argmax | — | 12/16 |

TF climbed **17.8x** across the four; free-run did not follow and the manual
verdict never moved. The explanation that fits is W13A's causal inversion:
**the child is a self-consistent function fitted to its own wrong regime**, so
pulling it toward the parent's cached trajectory can degrade its own rollouts.
This is the strongest argument in the program that no cached-frame metric of any
kind belongs in an acceptance path, and it is now a standing rule
(`docs/CURRENT_FOLDING_GOAL.md` §"Closed-path additions", clauses 1–2 and 5).

Calibration band, for anyone quoting TF later: below ~65% it has only ever
coexisted with incoherence; ~80%+ is **necessary, never sufficient**, even at
1.35x. 33.10% is nowhere near the band. (Comparability caveat: the four TF/
free-run numbers come from the cached-parent teacher-forced trajectory gate on
each run's own evidence set — arm A's is 2-member `f62110d7…`, the 2x arm's is
1-member base-only — so they are a trend, not a controlled series.)

### 4.4 The outside view (§4.3 item 5) — stated with its provenance limits

`docs/OVERLAP_AUDIT_2026-08-07.md` §Realignment records a 2026-08-06 field
taxonomy finding: **no published coherent 4x closed-form-only LLM exists, and
this program's 1.35x-works / 4x-fails boundary reproduces that cliff.**

**Honesty note: there is no sealed taxonomy artifact in this repository.** The
finding is carried in the audit's prose and is not independently reproducible
from anything in `reports/`. It is included because §4.3 item 5 requires it and
because it is the correct outside view, but it is the one claim in this report
that does not rest on an artifact digest, and it should be treated as an
argument rather than as evidence.

What *is* in-repo and points the same way: the cross-family lane's independent
depth wall on a **real pretrained** parent (SmolLM2-135M → blank qwen3 fails
closed at layer 3's attention branch, while a random-weight control at identical
geometry does not — so the wall is attributable to the parent being pretrained,
`reports/cross-family-real-20260804-pretrained/README.md`). Two different
families, two different pipelines, one shared conclusion: closed-form composition
through pretrained nonlinear dynamics accumulates error faster than local fits
suggest.

---

## 5. What was genuinely learned and built — the assets stage 2 inherits

These are results, not consolation. Each one is reusable in stage 2 unchanged.

### 5.1 The honest-measurement machinery

* **Terminal composition floor** — a fail-closed, composed, held-out measurement
  at layer 13 taken *at every install*, with an absolute arm (≥ 1.0 = worse than
  emitting zeros) and a relative non-regression arm (tolerance 0.25), retaining
  decision authority even when performance metrics are marked diagnostic-only.
  It refused the R-arm in 12 minutes of closed-loop time and cost nothing.
* **Per-boundary gates, the norm ceiling, and the ported A12 rule** — the
  reclosure is now bounded by norm ratio *and* covered by terminal
  non-regression, with the decision evidence sealed before the refusal raises.
* **Sealed evidence discipline** — content-addressed parent caches, phase-cache
  keys that rotate when the budget moves (2x: 0 hits, 6 misses, cold by
  construction; the run built its own balanced basis and inherited nothing), and
  a bounded on-policy extension contract that runs only the child and evaluates
  the parent once per round.
* **The frozen manual-review protocol, digest-pinned.**
  `e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`, sealed
  before any r13 scoring (commit `1499df58`), recomputed and confirmed by two
  independent reviewers on two different children. Sixteen anchored cases, a
  separate safety axis, cap censoring recorded separately from failure, and no
  count or ratio given acceptance authority. **This is the instrument that makes
  the negative result trustworthy**, and stage 2 must keep using it.
* **Fresh-reload authentication and counted-state verification** — the 2x child
  verifies as `standalone-counted-child-state-exact`, 110,978,816 dense +
  3,867,776 continuous = **114,846,592**, i.e. exactly 2.000000x, with 25 of 25
  checkpoint rows executed and 25 distinct runtime digests.

### 5.2 The layer-0 representability proof

§3.10. A 46x solver headroom demonstrated under the *production* compilers, on
sealed evidence, with zero forwards of any kind — and the accompanying
demonstration that thirteen solver arms and a full width ladder cannot move it.
This is the result that converts "the fold cannot express the parent" from a
hypothesis into a refuted hypothesis.

### 5.3 The clamp / ridge attribution

§3.3 and `reports/stage1-attribution-20260808/`. A complete, reproducible
forensic account of why every body boundary solve in every production run this
program has ever executed delivered exactly ‖prior‖ of correction — including
the 42/42 clamp identity verified to 1.46e-16, the relative-ridge table showing
the regularizer being inverted instead of the evidence at layer 5, and the
stage-delta decomposition showing the MLP refit was the *productive* stage
(**+0.1562 of the fold's +0.2269 total per-layer improvement, 69%**) while the
operator stage owned the damage at both layers that net-regressed (L0:
−0.1164 operator against +0.0236 MLP; L13: −0.1540 against +0.0121).

### 5.4 The any-architecture kernel estate

Independent of the LFM2 retention result, and the part of the program with the
longest reach:

* **chimera-omni plans the real `k3_aligned_test_config` crossing with zero gaps**
  — `parent kernel_gap_block_kinds ''`, `plan.ready True`, `plan.gaps []`,
  methods `native_residual_chart_interface` 14, `causal_stable_latent_moe_channel_mixer_layer` 4,
  `causal_kda_recurrent_layer` 3, `causal_latent_kv_attention_layer` 2,
  `causal_fused_gate_up_mlp_layer` 1 (the dense SiTU mixer), `copy_exact` 2 —
  every child tensor owned exactly once, with two companion tests asserting that
  a gated-activation *kind* translation and an unshared SiTU cap both correctly
  **keep** their gap (`NOTES-W13J.md` §4.1).
* **An executed exact-4x fold containing a dense SiTU mixer next to gated-MLA and
  Stable-LatentMoE neighbours** — plan `ready`, zero gaps, **24 actions**
  (1 windowed-NoPE, 1 KDA, 2 latent-KV, 1 fused gate/up SiTU, 3 Stable LatentMoE,
  14 chart interface, 2 `copy_exact`) — with 0 parent root forwards, 0 parent
  module forwards, 0 optimizer constructions/steps, 0 backward calls, and an
  asserted affine budget law (`total(H) = 12135·H + 16180`, so `H_p = 4·H_c + 4`
  is exact) (`NOTES-W13J.md` §4.2).
* **A real trained chimera-omni parent folded exactly 4x** — 1,104,128 → 276,032,
  residual from exact **0**, all 63 tensors individually exactly 4x, no
  quantization, no sparsity, no storage sharing, no parent sidecar; 73.3%
  next-token top-1 agreement against a blank child's 0.8%; **91.3% of the
  blank → parent gap closed** on the aggregate native-media objective; real
  nonempty PNG/WAV/MP4 emissions (`docs/CHIMERA_OMNI_FOLD_COVERAGE_2026-08-04.md`
  §2). Its own honest gaps — tiny synthetic parent, `local`+dense-FFN only,
  one global chart, audio worse than blank on a single sample — are recorded in
  §5 of that document and are not being restated here as successes.
* **Cross-family, cross-width and cross-kind folds against real pretrained
  checkpoints**: SmolLM2-135M → blank qwen3 (plan ready, zero gaps, 8 actions)
  and the first fold of a real pretrained *multimodal* checkpoint,
  SmolVLM2-256M-Video-Instruct (plan ready, zero gaps, 8 actions), both with
  config-field-by-field verification and weight digests sealed into the run
  identity before any compute.

### 5.5 The body-vs-counted reframing

`NOTES-W13T.md` §1. The token interface shrinks 16x on its own (65,536 donor
rows → 4,096 target rows), so counted compaction and body contraction diverge
sharply, and it is the body that governs behavior. This reframing is what makes
the 2x measurement worth taking (2.000x counted is only 1.559x body — much nearer
the surviving rung than the counted ratio suggests) and what makes the ceiling
statable at all. Every future ceiling claim must carry both numbers.

### 5.6 The closed-path ledger

`docs/CURRENT_FOLDING_GOAL.md` §"Closed paths" and §"Closed-path additions from
the 2026-08-07 overlap audit". Nine closed paths plus five do-NOT clauses, each
tied to a sealed measurement. The audit's own honesty scorecard names the
recurring institutional failure precisely — **cross-lane transfer**: lessons
sealed at the token interface or in toys were not ported to the body pipeline
before production runs, and that cost 95.2 h + 16.5 h + three waves of latency.
That diagnosis is itself an asset, and stage 2 inherits the obligation.

---

## 6. Honestly-open cells

### 6.1 Refinements that cannot plausibly change the conclusion

Listed so the claim in §1 is airtight rather than convenient. For each: why it
is expected not to move the zero, and what it costs if Cyril wants it anyway.

**(a) A bracketing arm strictly inside (1.0x, 1.559x] body.**
*Why it will not change the zero:* it cannot make the 1.559x zero nonzero; it can
only locate the cliff edge more precisely between a proven 10–12/16 and a proven
0/15. The most it can produce is a nonzero score at, say, 1.25x body, which
would refine the frontier — and would still leave stage 1's claim ("zero at and
below 1.559x") standing exactly as written.
*Cost:* one new admitted carrier ABI profile in
`ADMITTED_LFM2_CARRIER_COUNTED_ABI_PROFILES`, its catalog guard, and the test
assertions that pin the refusal message naming the admitted tuples — the same
edit shape as commit `e72bf169`, which is proved (17/17 on the carrier e2e and
fresh-process lanes) and landed; one budget derivation; ~14 h wall clock (the
2x arm ran 13 h 51 m
launch → final receipt, of which 11 h 16 m to `run_completed`) plus a ~2.5 h
dossier and a half-day frozen review. **Operational blocker: disk.** The 2x run
occupies **123 G** and `/home` is at 93% with **123 G** free; another 2x-scale arm
needs a cache reclamation decision first (three sealed caches at 94 G / 94 G /
86 G are the only large consumers — see §6.3).

**(b) R1+R2 retried at the 1.559x body.**
*Why it will not change the zero:* R1+R2 is a *solve-quality* remedy, and the 2x
arm already establishes that solve quality is not the binding variable there —
its layer-0 clamp requests were **12.37x / 8.04x / 3.58x** against arm A's
1.446x–140.4x range, i.e. a less-contracted body already asks for a far smaller
correction, and it still produced 0/15. The physics of §4.1 says a larger
released step composes worse, and the R-arm demonstrated exactly that at 4x.
*Cost:* two flags on the existing 2x launcher, one run (~14 h), one review
(~half a day), plus the same disk decision. λ=1e-4 or ratio 2.0 would be a
different point in the same space; neither has been measured.

**(c) A terminal-phase capsule protocol** (a post-structural-transport checkpoint,
`NOTES-W13B.md` §8 recommendation 5, still unimplemented).
*Why it will not change the zero:* it is pure instrumentation. It would put the
structural-transport priors themselves on disk so the prior can be scored against
the parent directly instead of only through its norm — closing the one gap that
forced §3.3's attribution to convict the prior as *budget* rather than as
*anchor*. It changes what we can see, not what the child does.
*Cost:* one capsule write in the fold's phase sequence plus its manifest schema;
~1 day of implementation and test; adds disk per run.

**(d) The W12B damped recaptured sweep, wired to the catalog.**
*Why it will not change the zero:* it is the only reclosure variant with toy
evidence of composing safely, and the reclosure is already **defanged** in both
arms that reached review — 27/28 and 26/28 overwrites refused, all transitions
refused at least once. A better reclosure can at most recover what defanging
already recovered. The reclosure has been convicted, guarded, and shown to be
necessary-but-not-sufficient; a third policy operates on a stage that is no
longer doing damage.
*Cost:* the sweep is still module-level only (no catalog flag; the adapter passes
no `mid_sweep_recapture_provider`); ~2–3 days to wire, test and prove, plus a run.

**(e) On-policy rounds ≥ 2, and an end-to-end V5 re-score of A9/A11.**
*Why they will not change the zero:* round 1 was consumed cleanly in arm A with
115 extension records merged into 371 paired records, `parent_capture_invocations: 0`,
with the reclosure defanged — the unconfounded read the audit demanded — and
produced 0/15. A second round extends the same evidence class against the same
wrong-regime child. The A9/A11 re-score is review-only: it cannot change a 4x or
2x zero; it can only sharpen the frontier row's provenance (§2.4) and would most
likely shift one row from Useful to Partial, leaving the 10–12 band intact.
*Cost:* round 2 is ~4 h of rollouts plus a ~14 h refold plus review. The A9/A11
V5 re-score is **zero compute** — the children, routes and high-cap dossiers are
saved — roughly one reviewer-day.

**None of (a)–(e) is a condition stage 1 set for itself.** They are refinements.
Stage 1's condition was "the best composed retention achievable at each
compaction level with every procedural remedy exhausted", and that is §2 + §3.

### 6.2 Open instrumentation defects (§4.3 item 7) — so the ceiling is not over-claimed

* **The free-run tripwire has still never produced an observation.**
  `terminal_free_run_tripwire_enabled: true`, `terminal_free_run_tripwire_evaluated:
  **false** at all 14 layers` in arm A *and* in the 2x arm. The 2x run at least
  says why, which is new: `terminal_free_run_tripwire_unevaluated_reason = "no
  free-run observation source: no probe is assigned and the macro-cell attractor
  free rollout is not running, so this run carries no counterfactual free-run
  measurement"`. Commit `38cc3bc9` found and fixed the root cause (the observation
  probe was assigned only in test files) and hoists the macro-cell snapshot when
  the tripwire is on — but **under bridged flags the tripwire still cannot
  observe**, and a target-token-aware rollout is named as future work. The one
  clause built to shout at chance-level free-run has been silent through every
  run whose free-run is chance-level. It is advisory and gates nothing, so it did
  not affect any acceptance — but the ceiling is stated *without* the one
  measurement designed to contradict it.
* **The operator branch's prior-correction reports are still computed and
  discarded.** `prior_correction_reports` exist for the short-conv producer and
  the attention equivalents, but the adapter serializes only the MLP triple. So
  there is **no** `update_scale`, `prior_norm`, `correction_norm`, `solver_path`
  or condition number for any operator solve at any layer, in any run — and the
  operator stage is where the damage sits at L0 and L13 (§5.3). R3(a) closes it
  with one serialization change; R3(b) (`residual_endpoint_relative_rmse`, computed
  at `hf_lfm2_structural_residual_attention_fold.py:4970` and thrown away) and
  R3(c) (`mlp_down` has no conditioning telemetry at all, the least-solved role in
  the fold) are the same size. **Consequence for this report: the MLP branch is
  attributed; the operator branch is not, and is not claimed.**
* **Refused layers publish no telemetry.** The R-arm's R1/R2/R3 per-solve numbers
  were computed at layer 0 and discarded when the layer was rejected — the reports
  reach disk only through an *accepted* layer's completion receipt. The arm that
  would most have benefited from the telemetry is exactly the arm that could not
  see it. Commit `e72bf169` deploys a fix (`_refused_layer_prior_correction_diagnostics`,
  `refused_layer_solver_diagnostics`) and it is **still unexercised**: no layer has
  been refused since it shipped.
* **`d296b602` remains unfixed and undiagnosed.** Every deployment since W13O has
  routed around it rather than through it. A 4-minute local reproducer exists.

### 6.3 Operational state, recorded honestly

The disk risk `NOTES-W13V.md` §7 flagged **did not** materialize — the 2x run
completed, published, produced its dossier, and sealed its receipt. But it left
the host tighter: `/dev/sda2` is at **93%**, **123 G** available, and the a0028
run directory alone is **123 G**. The three large reclaimable caches
(`…sealed-w13m-firstcomposed/cache` 94 G, `…sealed-armA-complete/cache` 94 G,
`a0023-…-r12/cache` 86 G) are all inside sealed run directories, and a CIFS
mount with 2.0 T free exists at `/mnt/ai_data`. **Nothing was deleted, moved or
archived** — deleting sealed evidence is a decision for Cyril, not for an agent.
Any further 2x-scale arm needs that decision first.

---

## 7. The stage-2 unlock statement

### 7.1 The conditions stage 1 set for itself

From `docs/CURRENT_FOLDING_GOAL.md` §"Staged escalation policy" (commit
`6f276ade`), stage 1 ends when, with sealed evidence, we can state:

> "the folding ceiling: the best composed retention achievable at each compaction
> level (1.35x, ~2x, 4x, ...) with every procedural remedy exhausted
> (per-boundary gates, terminal floor, on-policy evidence rounds, EIV/margin arms,
> reclosure modes, layout arms) — i.e. 'there is literally no fold-only path
> forward' is itself a proven, documented claim, not a feeling."

### 7.2 Whether they are met

**They are met.**

| condition | status | where |
|---|---|---|
| Best composed retention at 1.35x | 10–12 of 16 useful-or-partial, 1.000x body | §2.1 row 1 |
| Best composed retention at ~2x | **0 of 15**, 1.559x body, terminal 0.569–0.609 | §2.1 row 5 |
| Best composed retention at 4x | **0 of 15**, 3.469x body, terminal 0.464–0.806 | §2.1 rows 2–3 |
| Per-boundary gates | built (W13C), measured inert on the metric class A12 sealed | §3.1 |
| Terminal floor | built, exercised, refused a real arm, disqualified as a proxy | §3.5, §2.3 |
| On-policy evidence rounds | consumed cleanly, unconfounded, no rescue, both sides of the cliff | §3.4 |
| EIV / margin arms | proven structural no-op at the failing sites; every arm is the control | §3.6 |
| Reclosure modes | three policies measured; convicted, guarded, neutral | §3.1 |
| Layout arms | width closed arithmetically at 4x **and refuted experimentally at 2x**; depth closed; charts/rows exonerated | §3.7–§3.9 |
| The last never-tried lever (solver ridge + step bound) | executed as the R-arm; floor-refused at layer 0 | §3.3 |

**Modulo §6.1's optional refinements — and to say it plainly: they are
refinements, not conditions.** None of (a)–(e) is named in the policy text
above, none can convert a measured zero into a nonzero, and each is costed in
§6.1 so Cyril can buy one anyway without re-opening the stage.

The two instrumentation defects in §6.2 are stated as limits on the *claim's
scope*, not as unmet conditions: the operator branch is unattributed and is not
claimed, and the ceiling is stated without a working free-run tripwire.

### 7.3 Authorization

> **Stage 1 is complete and sealed. Stage 2 — fold + classical distillation — is
> authorized to open, under the stage-2 clause of `docs/CURRENT_FOLDING_GOAL.md`
> (commit `6f276ade`): fold-as-initialization plus the minimum distillation
> (teacher logits / short recovery finetune) needed to hit the target, measured
> against distillation-from-scratch so the fold's compute savings are quantified,
> not assumed. Stage 3 remains closed.**

### 7.4 What stage 2's first measurement must be

Two fold points, each with minimal recovery distillation, **each against a
distillation-from-scratch control at matched compute**. The control is not
optional: without it the fold's initialization value is assumed, and quantifying
it is the entire purpose of the staged design.

| arm | initialization | target | control |
|---|---|---|---|
| **S2-A** | the proven frontier: the A11 dual-route child — saved physical child state `54b9c5f7f484fd256a985aa33346f1bf4f601bec5f1978c287341d9e27f36e85`, **166,778,624** physical parameters + 3,867,776 route coefficients = 170,646,400 counted = 1.346018x, **1.000x body**; base dual-route publication artifact `860bccfe…` | recover toward the parent's V5 behavior from a child that already scores 10–12/16 | a blank child of identical counted topology, distilled from scratch at matched compute |
| **S2-B** | the cliff: the a0028 2x child, 2.000000x counted / **1.558902x body** (published child `f3f38cc3e87de7299196a5c8bb553ef1ca735a630ed43616029a59f694958fad`) | recover *any* nonzero V5 score from a child that is 0/15 but whose terminal reconstruction is 0.569 and whose terminal cosine is 0.860 | a blank child of identical counted topology (`config.json` `5c93a191…`), distilled from scratch at matched compute |

**Why these two and not 4x.** S2-A measures how much distillation a *working*
fold saves. S2-B is the sharper scientific question: it holds a child whose
reconstruction is good and whose behavior is zero, so it directly tests whether
the fold's geometry is a useful basin or an irrelevant one. If S2-B's fold-
initialized arm beats its from-scratch control at matched compute, the fold has
demonstrable initialization value even where it has no standalone behavior — and
that, not a coherent 4x child, is the result stage 2 exists to produce. The 4x
point can be added later; it is the least informative of the three because its
pre-install composition was already worse than emitting zeros (§2.2).

**Measurement rules stage 2 inherits, non-negotiably:**

1. The frozen V5 protocol (`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`)
   is the verdict. No count, ratio or token statistic is a gate.
2. Compute is measured, not estimated: wall clock, GPU-hours, tokens seen,
   optimizer steps, and the digest of the training corpus, on **both** the fold
   arm and the from-scratch control, or the comparison is void.
3. No cached-frame metric — teacher-forced top-1, boundary write error, terminal
   relL2 — appears in any acceptance path or headline. TF may be reported only
   with its calibration band attached (§4.3).
4. Everything stage 1 sealed stays sealed. Stage 2 does not refold, re-derive or
   overwrite any artifact in §2; it reads them.
5. The moment stage 2 starts writing gradients, the fold-only receipts stop
   being a claim about stage 2. Keep the two provenance chains separate and
   label every artifact with which stage produced it.

---

## 8. Caveats on this report

* **One parent, one scale.** Every retention number is LFM2.5-230M. Nothing here
  generalizes to another family without measurement.
* **Three reviewed children, not a curve.** §2's "curve" is four measured points
  (one of them a refusal). The interval (1.0x, 1.559x] contains no measurement.
* **The A9/A11 frontier row is pre-V5.** §2.4.
* **The operator branch is unattributed.** §6.2. The mechanism is shared by
  construction (same clamp, same ridge, same call-site pattern), but the operator
  numbers do not exist in any run and are not claimed.
* **The field comparison in §4.4 has no in-repo artifact.** It is the audit's
  argument, flagged as such.
* **The 2x arm moved two variables, unavoidably.** Body ratio (1.559x vs 3.469x)
  is the intended single variable and the flag sets are otherwise identical, but
  the evidence set also changed from arm A's 2-member `f62110d7…` to a 1-member
  base-only set, because dropping the on-policy round is what makes a 2x child's
  evidence on-policy. A 4x arm on base evidence only would separate them; it has
  not been run (`NOTES-W13V.md` §9).
* **§2.3's arm-A post-reclosure number (0.464258) is surfaced here for the first
  time.** It is read from arm A's own sealed receipt on the same field as the
  other two runs; arm A's review reported only the layer-13 install terminal.
  It is presented as a correction to a comparison in the 2x review, in the
  direction that costs the 2x arm its "best ever" claim.
* **Nothing in this report is an intelligence verdict of its own.** Per
  `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md` the frozen
  whole-response manual reviews are the authority; this document collates them.
* **Read-only.** The sealed run directories were accessed read-only for the two
  published-child `config.json` reads in §3.7 and for the timing extraction in
  §6.1; `…sealed-armA-complete/receipt.json` re-verifies as
  `45bc68c2e8766e030516f6bccbe7ec2142c2af229e26454e44f3cfa71ecd4f10` after all
  reads. Nothing was written, deleted, moved or archived anywhere.
* **Not committed.** No `git add`, no commit, no push. The only file written is
  this one.

---

## 9. Signature block

**Date:** 2026-08-08
**Branch:** `codex/lrc-blank-compression`
**HEAD:** `51b93f65ee59bb9cc7e5a80a93cd1045d79650f1` ("2x child frozen review: 0/15 -- the cliff is inside (1.0x, 1.559x] body")
**Policy authority:** `docs/CURRENT_FOLDING_GOAL.md` §"Staged escalation policy", commit `6f276ade`
**Frozen review protocol:** `e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b` (sealed by `reports/r12-certified-20260804/PROTOCOL_SEAL.json`, 2026-08-05T19:30:00Z)
**Prompt pack:** `research/lfm2_semantic_whole_response_prompts_v1.json`, `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b` — confirmed against each run's own `manual-prompts.sha256`

### Receipt / review sha ledger

| artifact | digest |
|---|---|
| **r12** `receipt_sha256` | `40e6dc722e488ae07a497bcbc75a509cb441df43c4ce114caf4393472f52dbf0` |
| r12 `receipt.json` file | `33e96e6c7863a8753de0704afb5838339950f094de2305cc26706f57f6d3e469` |
| r12 `configuration_sha256` | `ce40a41ffd807a7158ef69488ff66d138b2c75cac6c1a59e388a8027e2ea19d3` |
| r12 certificate | `046272ab12e876a31d70b4ca1bacf9f8a1756441772e9983b1384460277cbc9c` |
| **r13 first composed** (W13M) `receipt_sha256` | `c9a8f7468934f55a97f37cd14e7a7fb91721657dfa2f54f4218d9a962d98e4c9` |
| **arm A** `receipt_sha256` | `1cdec0a61cd94712bc3b26fffb26dce970291acd5b21817f970a030ad3f38168` |
| arm A `receipt.json` file (re-verified 2026-08-08) | `45bc68c2e8766e030516f6bccbe7ec2142c2af229e26454e44f3cfa71ecd4f10` |
| arm A `configuration_sha256` | `38d1766b8ad0228c9511d1399331b7c9a5343addc68de02a40bdf17248ebb102` |
| arm A published child artifact | `5f2977cf68a3bac5e91f4253f5949e459bd10b683a0e0212edefba99f605214d` |
| arm A routed manual dossier | `409de3ed809a75dce4905966cc659c9f6ef06562fae6f83ad0356e3bf980aa39` |
| arm A boundary acceptance | `1e789a1ecbc27db083ff36eccc21ae184fb3af587afedad00f82a3949b0f6405` |
| arm A child `config.json` | `d2cb79c34bba34344681f0b086177249a44e2f4da60aac37dde0bd21ddd936d1` |
| **R-arm** `receipt_sha256` | `42bc21c308691b831ed6f294f1fa75f2a206629e09c5123d4238aca800164873` |
| R-arm `receipt.json` file | `5d61ae20913810f06324780e966d11522758352552d461312c549571ef40a1e7` |
| R-arm `configuration_sha256` | `637bfe3f29c70250929d1310c5409390e72f923261fd3487c935a2d60e7b9f55` |
| **2x arm** `receipt_sha256` | `a218501f4b91e98ac2e3e9567684c86c60e194e39fc0bdece76222395eff588c` |
| 2x arm `receipt.json` file | `b1b8203e787dac1535e355b8a50fcbdb99df55bba8f730c6bda02dd102369ba6` |
| 2x arm `configuration_sha256` | `66eab8dc34624236fb7370b07232aaf889f045b7105b4e06c5cd814cca15da6e` |
| 2x arm SOURCE_COMMIT | `c17b28d07e78de9d6f6d58b22493939adf8d4af0` |
| 2x arm published child artifact | `f3f38cc3e87de7299196a5c8bb553ef1ca735a630ed43616029a59f694958fad` |
| 2x arm routed manual dossier | `28981b2b54946ce01d61c87a620e02c3189b99c489f850c71946e405ce4ca6f4` |
| 2x arm boundary acceptance | `01297f2b8b66d4a57e3fb9731e356a0fa29b3498b3c1a1960db83e7b82176c1f` |
| 2x arm counted child state verification | `6356be4d966e3191261416890c9947c754437d5119ca7459518ee8e0286573a1` |
| 2x arm child `config.json` | `5c93a191765f4d30564afd907eba7618b458a4356ba3ac85dff617778e1ae0c7` |
| **A9/A11** publication artifact | `860bccfe6be46e543f67c8514f3f3ff7dc0d729eb82267f66a9ce1766e43d7bb` |
| A9/A11 completion | `55a550aaf48189f9989d799b49980f7d61b59134d4f4aa8edc4cfa7e3c84bfb3` |
| A9/A11 `C_in` route tensor | `872bb10ada028d4f57eefdb29a65b0b5f4c46c6f5defa65ac0f4766afe6cf579` |
| A9/A11 `C_out` route tensor | `a3b74a35f96f4fdd32f65fdfc7eb7f7720cb0620f13ceb53f6c12a636280d042` |
| A9 high-cap dossier | `e6849cefaf31c66494a5ab033c9411e669ee6a8a16a3e637c76b08df3481f88a` |
| A11 high-cap dossier | `4d4de7c3c1cb96d9a15ec945053f7b17ed8e0b629971097ab30759d0d9a0633f` |
| **Stage-A attribution** extracted closed-loop record | `9a6e9313e2a8290a29c16d2f34837a6d0dc277d9d9f50ec812f36760c6ec0375` |
| Stage-A `diagnostics.json` | `cc8d1e64acab127aae6a729d3060dcd3c922424adbdad6c256673397b67b7774` |
| Stage-A `ledger.jsonl` | `b7d916ae989e460c7c0325cac7ffadd63508592c1c6df15df21e49613d4150bf` |
| Stage-A `child_graph.json` | `f14b244d002a087b2d972f342dc853c9098d4079c35bdfd3bf27677c1cae5958` |

### Governing documents

| document | role |
|---|---|
| `docs/CURRENT_FOLDING_GOAL.md` | objective, fold-only contract, staged policy, closed paths |
| `docs/OVERLAP_AUDIT_2026-08-07.md` | the regression audit that selected branch (b) and named this report |
| `reports/stage1-attribution-20260808/STAGE_A_ATTRIBUTION.md` | the clamp/ridge attribution and this report's §4.3 contents list |
| `reports/r12-certified-20260804/V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md` + `PROTOCOL_SEAL.json` | the review instrument |
| `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md` | manual review is authoritative; numerics are diagnostic |
| `NOTES-W13A/B/M/S/T/U/V.md` | the wave record behind §2–§4 |
| `reports/armA-review-20260808/`, `reports/2x-review-20260808/`, `reports/r12-certified-20260804/` | the three frozen verdicts |

*Compiled 2026-08-08 at HEAD `51b93f65`. Zero training operations, zero gradients,
zero model calls were performed in the production of this document. The sealed
run directories were read and re-verified; nothing was written to them.*

---

## AMENDMENT 1 — 2026-08-09: the frontier row is revised by cell (e)

The original text above stands as written; this amendment supersedes the
statements §9.1 enumerates. Open cell §6.1(e) was executed
(reports/frontier-rescore-20260808/, receipt 82561986…): the A9/A11
dual-route children were re-decoded on the sealed reuse runner (decode
equivalence proven — 15 of 15 overlapping responses byte-identical to the
sealed 2026-07-30 high-cap run) and re-scored under the frozen V5 protocol
(digest e5e37768… verified on both hosts before any model load).

**The frontier row becomes: 1.000x body = 8–9 of 15 useful-or-partial**
(A9: 3 useful / 5 partial; A11: 2 useful / 7 partial), revised from the
legacy 10–12 of 16. Four grades of thirty moved, none upward; one point per
lane comes from the protocol excluding the safety case from intelligence
accounting. Sensitivity: all close calls upward gives 9/11, downward 6/9 —
the A9 revision is robust, the A11 revision rests mainly on two calls.
Attribution caveat: three variables moved together (protocol, bf16-on-P40
vs fp32-on-SM86, device); 8 of 19 legacy-EOS rows decode differently, so
the delta cannot be attributed to grading alone. §6.1(e)'s prediction
("one row shifts, band intact") is recorded as wrong.

Every structural claim of this report survives and sharpens: 8–9/15 at
1.000x body against 0/15 at 1.559x and 3.469x; the cliff bracket, the
body-vs-counted reframing, the remedy ledger, and the stage-2 unlock are
unchanged. parent-not-demonstrated now covers every row on the curve; the
stage-2 design's parent baseline closes it.
