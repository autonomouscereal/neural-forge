# NOTES-MX-D — matrix campaign, the final unblock

Wave MX-D. Three things: the `inf`-sentinel seal defect patched at the three
registry sites `NOTES-MX-C` §5 named, the calibration question answered with the
strongest achievable honest criterion, and — because it answered PASS — the
first three real body cells of wave 1.

Repo baseline HEAD `86bd44eb`, branch `codex/lrc-blank-compression`. The frozen
collector and the compatibility policy were **not touched**. No `git add`, no
commit. Codex's live edits (`dynamic_fold_diagnostics`, the closed-loop adapter,
the `stage2_*` files and their tests) were not touched either.

**Verdicts up front:**

| deliverable | verdict | one line |
|---|---|---|
| the three registry `inf` sites | **PATCHED, 8 new tests** | each defect reproduced through the *real* sealer first, then fixed so the flag that depended on the sentinel keeps its meaning |
| cell 0b, quality-band calibration | **PASS** | **8 of 15** useful-or-partial against the sealed A9 child's **8 of 15** at the same cap — same composition, same category on every case, band delta **0** |
| wave-1 cells 1–3 (`C05.T5.M16.{L00,L05,L13}`) | **RUN, all three FAIL at `composition`** | boundary is excellent (write cos ≥ 0.9958, macrocell relL2 ≤ 0.037, all in band) and the composed terminal still drifts 1.5–2.2x past the 0.02 gate; free-run 0.662 / 0.791 / 0.688 |

The campaign's first real finding is in §4.1: on these three cells the S8
macrocell band **predicts behaviour 0 times out of 3**.

---

## 1. The `inf`-in-the-receipt defect, patched

`seal_receipt` content-addresses with `allow_nan=False`. A non-finite number in
a `solve` slot therefore does not fail the *cell* — it destroys the *receipt*,
after the solve has been paid for. That is how the first completed A9 route lost
a 32-minute compile (`NOTES-MX-C` §4.2). Three copies of the same defect were
still live.

| site | what it was | what it is now |
|---|---|---|
| `src/neural_forge/matrix_technique_registry.py:945` (was :801) | `allowed_clamp_ratio = float("inf")` whenever the T1 report carried no allowed ratio — wave-1 cells 12–21 and 29 | `None`, paired with a `None` requested ratio, because the report emits **both or neither** |
| `src/neural_forge/matrix_technique_registry.py:1031` (was :879) | C14's norm kernel set `allowed_clamp_ratio: float("inf")` *unconditionally* — wave-1 cell 30 could not have sealed | `None`; the refit runs no clamp lane, and the requested ratio it *does* measure stays a real number |
| `src/neural_forge/matrix_technique_registry.py:895` (was :785) | `condition_number = float("inf")` when the solver reported none, with `condition_cap_saturated` **derived from that sentinel** | flag derived from the **raw** reading first, then the number encoded as `None`, plus a new `condition_number_unbounded_reason` naming which unbounded case it was |

### 1.1 Why `None` and not `0.0`

The runner's own fix used `0.0` with `clamp_engaged: False`, citing `_no_solve`.
That convention is right for a technique that solved *nothing* — every number in
that block is a declared no-op. It is wrong for a kernel whose other numbers are
real measurements, because a reader cannot then tell a measured zero from an
absent one. The receipt schema already declares the alternative: every numeric
`solve.*` slot is `float | int | null`, `blank_receipt` starts each of them at
`None` "never `0.0`, so a forgotten measurement fails loudly instead of quietly
claiming a perfect zero", and `first_divergent_metric` carries an explicit branch
for a null clamp ratio. `None` is the schema's own sentinel encoding; it is what
the patch uses.

### 1.2 The load-bearing one

`condition_cap_saturated` **is** the statement the `inf` was making, and there
are two ways to reach it that mean the same thing to chain link 2:

* `_solved_system_conditioning` returns a genuine `math.inf` for an exactly
  singular solved system (`hf_lfm2_structural_residual_attention_fold.py:1113`),
  and
* a solve that reported no conditioning at all is unbounded-until-proven-
  otherwise.

Sanitizing before deriving the flag would have turned both into "condition
number 0.0, cap not saturated" — the one reading that is definitely false. The
patch derives `saturated` from the raw value, *then* encodes, and records which
case it was so a receipt reader can tell a singular system from an unmeasured
one. The relative-ridge arm (`>= 0.99 / lambda`) is unchanged and now only has
to decide the finite case.

### 1.3 Two things the audit turned up on the way

* **The nested report carries the same sentinel.** The T1 block embeds
  `Lfm2PriorCorrectionReport.as_dict()` verbatim, and that dict has its own copy
  of `system_condition_number`. Sanitizing only the flat field would have left an
  `inf` one level down and the seal would still have died — for a reason nobody
  reading the flat block could see. `sealable_report()` walks the whole
  structure, nulls non-finite leaves, and returns the dotted path of every one
  that moved into a new `prior_correction_report_non_finite` field.
* **The defect class is now refused, not just fixed.** `validate_result` runs
  before anything is installed; it now scans the entire `solve` mapping
  recursively and raises `CellTechniqueError` on any non-finite number, naming
  the path. A technique-lane kernel with this bug costs a cell and gets an
  `infra` receipt *with a reason in it*, instead of a GPU-hour and a traceback.
  A null in the same slot is explicitly allowed.

### 1.4 An engaged clamp with no ratios

`first_divergent_metric` refuses to walk link 2 when `clamp_engaged` is true and
either ratio is null — and it refuses *at seal time*. The report's own invariant
says that cannot happen (the clamp block is emitted only when the clamp was
evaluated, and then both ratios are floats), so the T1 kernel now checks it
directly and raises before the solve is believed.

### 1.5 Regression tests, one per site

`tests/test_matrix_technique_registry.py`, 8 new tests (21 → 29), every one
walking a **real** kernel result through the **real** `seal_receipt`:

```
test_a_singular_solved_system_really_reports_an_infinite_condition_number
test_t1_seals_an_unbounded_condition_number_and_keeps_the_cap_flag
test_t1_treats_an_unreported_condition_number_as_unbounded
test_t1_refuses_an_engaged_clamp_with_no_ratios
test_t1_unclamped_solve_seals_null_clamp_ratios          <- site :801
test_c14_norm_refit_solve_block_is_sealable              <- site :879
test_result_guard_rejects_non_finite_solve_telemetry
test_sealable_solve_number_encodes_absence_as_null_not_zero
```

The first test is there so the second is not a fiction: it pins that the
production conditioning helper really does return `math.inf` on an exactly
singular system, which is the sentinel the fix has to survive. The two site
tests assert the chain still stops where it should — `solve-conditioning` for an
unbounded condition number, and **not** at link 2 for an unevaluated clamp.

Verified against the pre-patch values directly:

```
$ python -c "... seal_receipt(<C14 block with allowed_clamp_ratio=inf>) ..."
PRE-PATCH BLOCK REFUSED BY SEALER: DynamicCircuitIRError IR payload is not canonical JSON
```

### 1.6 One more, found by digesting the mirror

`append_ledger` opened the ledger in **text** mode, so an append made from the
Windows workstation wrote `\r\n` while the Linux run host wrote `\n`. Two copies
of an append-only file, identical rows, different bytes — and `read_ledger`
splits on either, so it stayed invisible until the repo mirror was digested
against the authoritative file. Fixed to binary mode with an explicit `\n`
(`src/neural_forge/matrix_cell_receipt.py:863`) plus
`test_ledger_bytes_are_platform_independent`. The three rows already appended
that way were normalized; mirror and authority now agree byte for byte
(`d1574143bef80811…`).

---

## 2. Calibration: the quality-band criterion

`reports/matrix-campaign-20260810/cells/CAL.C01.T5.X4-mxc/QUALITY_BAND_REVIEW.md`
(80 KB, every response quoted verbatim from a digest-asserted export).

### 2.1 Protocol authority

`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b` recomputed on
the workstation **before any response was read**, matching
`reports/r12-certified-20260804/PROTOCOL_SEAL.json`. Prompt pack `0ee6bf14…`
confirmed against the seal.

The prompt-surface fork `NOTES-MX-A` §8 reported is **not** a deviation: the two
raw byte-forms (`ff6fa27e…`, which cell 0b consumed, and `0ee6bf14…`, which the
seal names) both load through `load_review_prompts` to the protocol's own
normalized `5615b9e4…`, field for field. Both dossiers under review record that
digest themselves.

### 2.2 One deviation, absorbed rather than waived

The dossier was decoded at `maximum_new_tokens: 256` — the sealed A9 launch
parameter — while the published A9 band of 8/15 was measured at 1,024 with a
2,048 retry. Grading a 256-token child against a 1,024-token band measures the
cap. So the **sealed A9 dossier was re-graded at its own 256 cap** as an
evidence-matched control, and both sides of the comparison rest on prefixes of
the same length. Censored rows: 8 of 15 recompiled, 7 of 15 sealed. No case in
either lane was categorized on its endpoint; every censored row carries an
explicit statement of what its prefix proves.

### 2.3 The tally

| category | recompiled (mxc2) | sealed A9 @256 | frontier re-score, A9 @1024/2048 |
|---|---:|---:|---:|
| Useful | **3** | 3 | 3 |
| Partial | **5** | 5 | 5 |
| Observed miss | **7** | 7 | 7 |
| Inconclusive/censored | **0** | 0 | 0 |
| **Useful-or-partial** | **8 of 15** | **8 of 15** | **8 of 15** |

Safety, separate axis: `concerning` in both lanes.

The categories are **identical on all fifteen cases**. Re-grading the sealed lane
at 256 also reproduced **15 of 15** of the frontier re-score's published grades,
which is the reviewer-consistency check: same reviewer, same protocol, shorter
cap, same categories.

Useful, both lanes: `instruction-three-actions`, `summarization-priorities`,
`uncertainty-insufficient-data`. Partial, both lanes:
`conversation-natural-greeting`, `context-retrieval`, `writing-concise-email`,
`planning-dependencies`, `creative-continuity`.

Two responses are **byte-identical** between the two children
(`context-retrieval`, `summarization-priorities`) — the same two the runner's
`token_match_vs_control` counts as 2/16 = 0.125. The exact-match metric and this
review are reading the same two rows and drawing opposite conclusions from them.

Where they differ within a shared category: the recompiled child is **worse** on
`explanation-seasons` (repeats one false sentence four times where the sealed
child says it once and terminates), `arithmetic-multi-step` (`60 - 8 = 60`
against the sealed child's `40 - 8 = 32`; neither reaches an anchored number),
`creative-continuity` (nothing occupies the foghorn's slot, no final beat) and
`writing-concise-email` (the tracking request is reversed into an offer). It is
**better** on `planning-dependencies` (a real roll-back bullet, concrete backup
commands, and the only `tls` token in either lane). None of those crosses a
category boundary.

### 2.4 Verdict: PASS

Tolerance applied, since the proposed ADDENDUM 2 names none: **±2 cases**, and
the justification is measured rather than chosen. The frontier re-score's §7.4
states its own A9 sensitivity interval — resolving every close call downward
gives 6, upward gives 9, around a graded 8. ±2 *is* the reviewer-judgement noise
floor on this exact 15-case surface. Tighter would measure the reviewer; looser
would admit a genuinely worse child. **The observed delta is 0**, so the verdict
does not depend on where inside that range the line is drawn.

What the verdict does **not** say: the route is not reproducible (the portability
wall in `NOTES-MX-C` §4.4 is untouched), the child is not good (7 of 15 observed
misses, safety `concerning`), and the sealed receipt `4018355521b93a30…` still
reads `status: fail`, `first_divergent_metric: free-run`, because the criterion
registered in the repository is still the 0.90 token match. Superseding it is the
coordinator's commit.

### 2.5 The sealed review instance

`experiments/seal_matrix_cell_review.py`, verdict **`incoherent`**, receipt
`28564b7156f8c9a1…`, instance `5b8e6ee3692a4dcc…-review`.

The verdict is `incoherent` and that is not in tension with the PASS. For a route
cell the runner does not decode the campaign's 3-prompt smoke set at all —
`_route_behavior` takes the first three shared dossier keys
(`run_lfm2_matrix_cell.py:1965`), which on this cell are `arithmetic-multi-step`,
`coding-deduplicate` and `comparison-ram-rom`: three of the seven cases the
**sealed A9 child also fails**, identically graded. The receipt's status is
`fail` at chain link 6 with or without the verdict, so nothing is laundered
either way.

**Harness defect found here, reported not patched:** `_route_behavior` labels
every smoke row `"terminated": "eos"` unless the text reaches 4,000 characters.
Two of those three rows were in fact censored at the 256-token cap, and the
dossier says so in a `termination_evidence` block the runner never reads. A
reviewer working only from the receipt would apply "cap hits are censored, not
failed" to zero rows that deserve it. The runner belongs to a lane with live
edits in it.

---

## 3. Proposed ADDENDUM 2 — final text for the coordinator to commit

> ## ADDENDUM 2 — 2026-08-10: cell 0b's calibration criterion
>
> §4.1 enters cell 0b to prove the harness can drive the one known behavioral
> success. Two criteria have been registered for it and both are now known to be
> unsatisfiable *in principle* at HEAD, through no fault of the harness. This
> addendum records why each fell, registers the third, and reports its result.
> §5 forbids editing a rule, so the earlier registrations are superseded here
> rather than amended in place.
>
> **Criterion 1 — artifact bit-identity (`NOTES-MX-A` §5.2, tolerance 0).**
> A9's *numeric* configuration is fully reachable again on an SM86 card: every
> one of the eleven dtype roles in
> `neural-forge-lfm2-sparse-semantic-numeric-dtype-contract-v1`, including
> `forward_dtype: bfloat16`, reproduces the sealed contract exactly, and two of
> the six sealed artifact identities — `candidate_support_math_abi_sha256` and
> `coefficient_solve_math_abi_sha256` — match A9 to the digest. What does not
> reproduce is the contract's *digest*, because the runtime dtype resolver's own
> receipt is hashed into it and that receipt's schema moved to
> `…-runtime-dtype-resolution-v2` after A9 was sealed. Every downstream phase key
> moves with it, nothing hits the sealed codebook phase, and the 4,096-row
> residual-OMP support search recomputes from scratch. That search runs in fp32
> on the GPU with near-ties broken by `|residual cosine|`; Pascal and Ampere fp32
> kernels do not agree to the last ulp, one tie flips one atom, and the support
> diverges from there. **A9's compiled artifact is reproducible only on the
> hardware that compiled it — a P40 under torch 2.6 — and that combination
> cannot open any parent trace cache at HEAD.** There is no configuration in
> which both constraints hold. Bit-identity is unreachable in principle, not in
> practice.
>
> **Criterion 2 — `behavior.free_run.token_match_vs_control` ≥ 0.90
> (`NOTES-MX-C` §3).** Measured 0.125. The threshold was not missed by a margin
> that invites tuning; the instrument is wrong for the question. Exact
> whole-response equality between two greedy decoders that diverge at one OMP tie
> measures *decode divergence*, not harness fidelity. The two responses it counts
> are the two shortest in the set (102 and 155 characters); the other thirteen
> agree in content while differing in wording, and the two children are four to
> five times more similar to each other than either is to the parent. Lowering
> 0.90 to fit 0.125 would be the goalpost move pre-registration exists to
> prevent, so the criterion is retired rather than relaxed.
>
> **Criterion 3, registered here — quality-band agreement.** The gate at chain
> link 1 remains the infrastructure question only: did the harness drive the
> mechanism to a complete, sealed route artifact with a gradeable whole-response
> dossier? The acceptance question is answered by the campaign's own
> authoritative behavioral judge, the frozen V5 manual review
> (`e5e37768…`): **the recompiled child's 16 responses are graded under the
> frozen protocol and must land within ±2 cases of the sealed A9 child's
> useful-or-partial band, graded under the same protocol at the same generation
> cap.**
>
> Three properties make this the strongest criterion still available rather than
> a weaker one chosen for convenience. It is *pre-registered against a published
> number* (the frontier re-score's 8 of 15, sealed 2026-08-08, before this cell
> ran). Its tolerance is *measured, not chosen*: ±2 is the reviewer-judgement
> sensitivity interval that re-score states for its own A9 lane (6–9 around a
> graded 8), so a tighter band would measure the reviewer and a looser one would
> admit a materially worse child. And it is *falsifiable in the direction that
> matters*: a T5 generalization or a harness that corrupted the mechanism would
> lose cases, and losing three is a fail.
>
> The cap deviation is absorbed, not waived. The dossier under review was decoded
> at the sealed A9 launch's 256-token allowance rather than the protocol's
> 1,024/2,048, so the sealed A9 child is re-graded at *its own* 256 cap as the
> control and both sides rest on prefixes of the same length. No case in either
> lane may be categorized on a censored endpoint.
>
> **Result: PASS.** 8 of 15 useful-or-partial (3 useful / 5 partial / 7 observed
> miss) against the control's 8 of 15 (3 / 5 / 7), with identical categories on
> all fifteen cases and two responses byte-identical. Band delta 0.
> `reports/matrix-campaign-20260810/cells/CAL.C01.T5.X4-mxc/QUALITY_BAND_REVIEW.md`,
> sealed as instance `5b8e6ee3692a4dcc…-review`, receipt `28564b7156f8c9a1…`.
>
> This is a *weaker* calibration than bit-identity and is recorded as such. It is
> not a waiver: nothing is pinned, no compatibility list is extended, the
> artifact-identity comparison is still computed and sealed on every 0b instance
> as a diagnostic, and a 0b that fails this criterion stops the campaign exactly
> as §4.1 requires. It also does not license any claim that the route is
> reproducible — it is not, and the reason is recorded in criterion 1.

---

## 4. Wave-1 cells 1–3

`C05.T5.M16.{L00,L05,L13}` — `conv_out_proj`, the T5 body-route generalization
at M16, against the fresh capture `2dec87ac…` with the pre-registered 5e-3
calibration tolerance, `CUDA_VISIBLE_DEVICES=0` (RTX 3090, SM86), float32,
knobs `/mnt/ai_data/neural-forge-matrix/c05-t5-m16-knobs.json` — no T5 knob set,
so `active_k=8` and the rest are `NOTES-MX-B`'s registered defaults.

| | L00 | L05 | L13 |
|---|---:|---:|---:|
| receipt | `853ac00d38a0…` | `5b250adb83f0…` | `12c4561c0f9b…` |
| wall | 52.0 s | 52.7 s | 53.2 s |
| **calibration relL2** (tol 5e-3) | **0.001657** | **0.001681** | **0.001895** |
| kept / dropped channels | 960 / 64 | 960 / 64 | 960 / 64 |
| stored parameters | 983,040 | 983,040 | 983,040 |
| condition number | 834.0 | 656.1 | 3359.7 |
| channel reconstruction relL2 (heldout) | 0.9754 | 0.9833 | 0.9437 |
| **write_heldout_cos** | **0.99908** | **0.99693** | **0.99585** |
| write_heldout_relL2 | 0.03644 | 0.07062 | 0.08426 |
| **macrocell relL2** (band ≤ 0.4) | **0.03675** | **0.02260** | **0.01665** |
| macrocell cos | 0.99904 | 0.99969 | 0.99982 |
| **composed vs control delta** (gate 0.02) | **0.04355** | **0.04048** | **0.02948** |
| norm ratio min/max | 0.944 / 1.035 | 0.987 / 1.011 | 0.996 / 1.002 |
| emit-zero floor | passed | passed | passed |
| **free-run token match** | **0.662** | **0.791** | **0.688** |
| mean lcp | 0.627 | 0.771 | 0.686 |
| decode smoke | 3/3 on task, all EOS | 3/3 on task, 1 cap | 3/3 on task, 1 cap |
| review | `coherent`, `5d6653040be3…` | `coherent`, `1c4a09cd549a…` | `coherent`, `be533d1bd1bd…` |
| **status / first_divergent_metric** | **fail / `composition`** | **fail / `composition`** | **fail / `composition`** |

The calibration arm reproduced the pre-registered floor to three digits on real
cells (0.001657 / 0.001681 / 0.001895 against the pilot's 0.001661 / 0.001650 /
0.001894), which is the first end-to-end confirmation that the 5e-3 tolerance was
a measurement and not a guess. The 1e-3 default would have refused all three.

### 4.1 What these three cells actually say

**The solve is good and the composition is not.** Every boundary number is
excellent — the component write keeps direction at cosine ≥ 0.9958 and the
composed layer output lands at 1.7–3.7% relative error, an order of magnitude
inside the S8 band. The chain then crosses at link 5: the terminal state drifts
2.9–4.4% from the copy_exact control against a 2% gate, on **one** folded
component out of the whole model.

**The macrocell band predicted behaviour 0 times out of 3.** All three cells are
comfortably in band and all three lost behaviour. That is §5 rule 4's 2×2 with
its first three entries filled, and it is the campaign's first substantive
result: at least for `conv_out_proj` at M16, a good local boundary score does not
survive composition. The rendered ledger reports it as `Agreement 0/3 (0%)`.

**Free-run degrades with distance from the terminal state, not with layer
depth.** L05 keeps 0.791 of the control's tokens, L13 0.688, L00 0.662 — the
quiet mid conv layer is the most forgiving and layer 0 the least, which is the
S3 operator-damage ordering showing up behaviourally rather than in the band.

**The decode smoke stays coherent everywhere.** All nine outputs are fluent, on
task and mostly correct: the circuit-breaker explanation is right at all three
sites, and the failures are instruction misses (L00's dinner plan answers with a
ten-dish menu and no cooking order) rather than collapse. A cell can lose 34% of
its control tokens and still write a competent paragraph — which is exactly why
§5 rule 1 makes the human review necessary but not sufficient.

### 4.2 Stopped here, as instructed

Three cells, then stop. Cells 4–30 are not run and the coordinator reviews first.
The obvious next question — whether `composition` is a property of `conv_out_proj`
at M16 or of the campaign's 0.02 gate against a single-component fold — is a
knob-and-level question (M8, M4, and the T5 `active_k` default of 8) that should
be asked deliberately, not absorbed into a batch.

---

## 5. Deployment correspondence — a real problem, fixed before the cells ran

The deployment's technique lane was **stale**, and it would have made every
wave-1 receipt claim provenance it did not have:

| file | deployed before | repo at HEAD | lines |
|---|---|---|---|
| `matrix_technique_body_routes.py` | `676532521229aff6…` | `055082c8a081317c…` | 1521 → 1712 |
| `matrix_technique_lowrank_writers.py` | `28047e6318fec9c1…` | `ab53ef253b35644b…` | 871 → 1023 |
| `matrix_technique_relation_matching.py` | `2c4b5554503624aa…` | `8281e44123b39970…` | 1680 → 2089 |

The deployment was built from the `dee1c952` worktree; the technique lane's work
landed in `a31747d8`. `code_identity.src_tree_sha` digests the *deployment's* own
tree, so the receipts would have been self-consistent and still wrong about which
kernel ran — a cell sealed with `--repo-head 86bd44eb` would have executed a T5
kernel that is not in that commit.

All four technique files (the three above plus the patched registry) and the
patched `matrix_cell_receipt.py` were deployed after a full backup, and the
deployed copies are now **byte-identical to the repo's**:

```
4f72ca3b0a817cb3  src/neural_forge/matrix_cell_receipt.py
c2b533774ac6d990  src/neural_forge/matrix_technique_registry.py
055082c8a081317c  src/neural_forge/matrix_technique_body_routes.py
ab53ef253b35644b  src/neural_forge/matrix_technique_lowrank_writers.py
8281e44123b39970  src/neural_forge/matrix_technique_relation_matching.py
f72f5511fb326fff  experiments/run_lfm2_matrix_cell.py
13715e6883322cc3  experiments/render_matrix_ledger.py
08fe52edc834ee0b  experiments/seal_matrix_cell_review.py
```

`registered_pairs()` on the deployment now reproduces `NOTES-MX-B` §6 exactly
(26 entries), and the three wave-1 receipts record
`t5_body_route_technique` from `neural_forge.matrix_technique_body_routes`.

---

## 6. Backups and the destructive-change check

| what | where |
|---|---|
| ledger before the review append | `/mnt/ai_data/neural-forge-matrix/backup/ledger.jsonl.before-mxd-20260810T172754Z` (`8ccf5105a4a7d277…`, 7 rows) |
| ledger before the wave-1 review appends | `…/backup/ledger.jsonl.before-mxd-reviews-20260810T172754Z` (11 rows) |
| the eight deployed matrix-lane files, pre-sync | `…/backup/deployment-before-mxd-20260810T172754Z/` |
| repo mirror ledger, pre-sync | scratch copy (`repo-ledger.before-mxd.jsonl`, 6 rows) |

| check | result |
|---|---|
| `find -newermt` over the fresh evidence base `2dec87ac…` | **empty** |
| `find -newermt` over the sealed A9 run dir `neural_forge_sparse_semantic_bridge_20260730` | **empty** |
| `find -newermt` over `lfm2_tokenizer_bridge_evidence_20260717` | **empty** |
| ledger rows | 7 → 8 (review) → 11 (three cells) → 14 (three reviews); **no row rewritten**, verified by comparing the prefix before and after every append |
| mirror vs authority | byte-identical, `d1574143bef80811…` |
| both dossiers | opened read-only, downloaded, never written |
| `/` free | 15 GB → 14 GB (three per-site evidence caches under `/home/cereal/nf-mxd/`) |

---

## 7. Verification

```
# local, .venv (python 3.11.9)
python -m pytest tests/test_matrix_cell_receipt.py tests/test_matrix_technique_registry.py -q
# -> 54 passed        (25 + 29; 8 of the registry tests are this wave's)

python -m pytest tests/ -q -k matrix --ignore=tests/test_hf_lfm2_matrix_local_isolation_backend.py
# -> 267 passed

python -m pytest tests/test_run_lfm2_matrix_cell_fresh_evidence.py \
                 tests/test_run_lfm2_layer0_representability_study.py -q
# -> 17 passed

python -m ruff check src/neural_forge/matrix_technique_registry.py \
                     src/neural_forge/matrix_cell_receipt.py \
                     tests/test_matrix_technique_registry.py \
                     tests/test_matrix_cell_receipt.py
# -> All checks passed!
```

Server: the deployment sync and its import/registry check, the three wave-1 cell
runs, the four ledger appends, and the destructive-change scan. The review's two
extractors and its generator re-run from the sealed dossiers and reproduce the
document byte for byte.

**Not verified:** no cell at M8 or M4, no component other than `C05`, and no T5
knob other than the registered defaults. The `composition` result is three cells
of one component at one level — enough to report, not enough to generalize.

## 8. Paths

* review: `reports/matrix-campaign-20260810/cells/CAL.C01.T5.X4-mxc/QUALITY_BAND_REVIEW.md` (+ `_extract_responses.py`, `_extract_a9_256_control.py`, `_build_review.py`, `mxc2-responses.json`, `a9-256-control-responses.json`, `cell-receipt.json`, `cell-receipt-reviewed.json`)
* wave-1 cells: `reports/matrix-campaign-20260810/cells/C05.T5.M16.{L00,L05,L13}/` (receipt, reviewed receipt, diagnostics, run log)
* ledger + report: `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`, `reports/matrix-campaign-20260810/{ledger.jsonl,LEDGER.md}`
* campaign store: `/mnt/ai_data/neural-forge-matrix/cells/{CAL.C01.T5.X4-mxc,C05.T5.M16.L00,C05.T5.M16.L05,C05.T5.M16.L13}/`
* run logs: `/home/cereal/nf-mxd/cell-C05.T5.M16.L{00,05,13}.log`
* backups: `/mnt/ai_data/neural-forge-matrix/backup/` (see §6)
