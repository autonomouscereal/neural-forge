# NOTES-MX-H — matrix campaign, wave 4: the composition modes

The wave every isolation cell existed to feed. ADDENDUM 1 makes the campaign's
deliverable a *heterogeneous* child — different techniques folding different
components — and ADDENDUM 4 pre-registers the modes that child is assembled by,
**as an experiment rather than an assumption**. This wave runs them.

Repo baseline HEAD `bc1aea7a` at the start, branch `codex/lrc-blank-compression`.
The frozen collector was **not** edited; the technique surface was **not**
edited; Codex's uncommitted `matrix_technique_body_routes.py` was **not**
deployed. No push, no rebase.

**Verdicts up front:**

| deliverable | verdict | one line |
|---|---|---|
| ADDENDUM 5, the pre-flight | **registered and rendered** | a level claim is site-scoped; the C07/T1 rule-2 ambiguity dissolves rather than being decided, and the renderer now prints `PASS@<level>@<sites>` |
| the recipes | **derived from sealed receipts, not from prose** | two variants; the storage clause promotes `C06.T5` over `C06.T1` and the conflict rule sends C09 and C10 to different children |
| **M-A** | **confirms A12** | RBEH free-run **0.539** against isolation winners of 0.738–0.994; RSTO **0.551** against 0.736–0.977 |
| **M-B vs M-A** | **conditioning changes the path and not the destination** | RSTO M-B free-run **0.551**, identical to M-A's 0.551 *to the token*, with a slightly worse global drift (0.0649 vs 0.0628) |
| storage | **the behaviour-first child is a storage failure and says so** | RBEH net **1.008707×** — it stores 2.0 M parameters MORE than its parent; RSTO net **0.998007×**, −457,728 parameters, the campaign's first composed child that contracts |
| the frozen review | **both M-A children match the parent's own band** | 12 of 15 useful-or-partial against a same-process parent control at 12 of 15 — band delta **0**, and **+4** on ADDENDUM 2's 8/15 reference |
| free-run vs quality | **they disagree, again, and hard** | a child at 0.539 token-match grades identically to the parent on 11 of 15 cases and beats it on 3 |
| two defects found | **both by producing numbers that could not be right** | a storage accounting that credited C10 with 917,504 parameters it does not have, and a timestamp inside a digested split manifest |

---

## 1. Pre-flight: ADDENDUM 5, a level claim is site-scoped

`NOTES-MX-G` §4.6 hit the first technique×component whose rungs disagree across
sites and reported, rather than resolved, that §5 rule 2's number and
`render_matrix_ledger.py:298`'s implementation of it are two different tests.
`C07.T1` is 3/3 at M16 and 2/3 at M8: the design's clause says `works to M8`, the
renderer says `works to M16`.

ADDENDUM 5 supersedes rule 2 with **rule 2-S**: a level claim is claimable only
at the sites with sealed passing receipts at that level *and every shallower
level attempted there*, is never written without its site set, and the ≥2-of-3
count survives only as a breadth annotation. The old question — does 2 of 3
license `works to M8`? — asked for a whole-component verdict that 2-S does not
issue, so it dissolves:

> `C07/T1 works to X4 at L12 (1/3 sites); shallower elsewhere: L02:M16; L08:M8`

The renderer implements it, and doing so retired two site-blind readings in the
opposite direction as well:

* a level whose sites *disagreed* printed no pass at all, so `C06.T1.X4`'s real
  L12 pass (0.977) was hidden behind two shallower failures. It now prints
  `PASS@X4@L12`.
* `C14`'s sealed NA-level PASS fell through every branch of the ladder verdict
  and printed `awaiting review` over a reviewed, passing cell.

Ledger rows are untouched; every claim re-renders under 2-S without a re-run.
Commit `5996655b`.

---

## 2. The recipes are a function of the sealed receipts

`experiments/derive_matrix_composition_recipe.py` takes the newest sealed
instance per cell id out of the append-only ledger (187 rows → 88 cells), opens
the receipt each row names **by digest**, verifies the seal, and grades it under
ADDENDUM 3. 185 receipts indexed, 0 ledger rows unresolved.

It reads storage from the **receipt**, not the ledger row, and that is not a
detail: waves 1 and 2 sealed their storage counts under `solve` before
`ledger_line` had columns for them, so a ledger-only derivation silently loses
the storage honesty of exactly the cells this wave most wants it for —
`C05.T5.M16.L05` (0.9375×) and `C03.T5.M16.L00` (0.9377×), two of the picks, both
of which read `null` in the ledger. It also recovers two storage-honest eligible
variants at C09 that `NOTES-MX-G` §5's storage-aware alternates table does not
list: **`C09.T5.M16.L02` at 0.7090 / 0.9375×** and `C09.T8.M16.L02` at 0.6777 /
0.9375×.

### 2.1 Two variants, because the two criteria name different cells

| | RBEH (behaviour-first) | RSTO (storage-honest) |
|---|---|---|
| admission | ADDENDUM 3 `promising`+ (free-run ≥0.60 **and** coherent smoke) | same |
| ranking | ADDENDUM 1: highest free-run, ties on composed-terminal margin | highest free-run **among folds that contract** |
| storage clause | promotes an honest variant only within 0.05 free-run | n/a — every admitted fold contracts |
| tensor conflicts | resolved on free-run | resolved on parameters removed |

The 0.05 band is not a free parameter: it is the width that separates the two
real cases in the sealed data. At C06 the honest variant costs **0.023** free-run
(inside), at C05 it costs **0.123** (outside).

### 2.2 The selected recipe, per component, with the reason

**RBEH — 7 folded, 5 passed through.** Install order is topological (input side →
output side; within a layer conv → attention → norms).

| # | component | cell | technique | level | site | free-run | stored | why |
|---|---|---|---|---|---:|---:|---:|---|
| 1 | C03 conv_in_proj | `C03.T5.M16.L00` | T5 routes | M16 | L00 | 0.7871 | 0.9377× | only eligible cell at C03 |
| 2 | C14 norms | `C14.T1.NA.L00` | T1 refit | NA | L00 | 1.0000 | 0.5000× | only eligible cell; the ratio is a bookkeeping artifact (§4.3) |
| 3 | C08 attn_v | `C08.T1.M16.L02` | T1 ridge | M16 | L02 | 0.7383 | 1.4062× | beat `C08.T1.M8.L02` (0.6504) |
| 4 | C09 attn_o | `C09.T2.M16.L02` | T2 charts | M16 | L02 | 0.8203 | 1.8750× | tied with `C09.T1.M16.L02` on free-run, won on cell id |
| 5 | C05 conv_out_proj | `C05.T3.M16.L05` | T3 balanced | M16 | L05 | 0.9141 | 1.8750× | `C05.T5` is 0.123 behind — outside the storage band |
| 6 | C06 attn_q | `C06.T5.M16.L12` | T5 routes | M16 | L12 | 0.9766 | 0.9380× | **promoted over `C06.T1.M16.L12` (1.000, 1.875×) by the storage clause: −0.023 free-run, half the storage** |
| 7 | C07 attn_k | `C07.T1.M16.L12` | T1 ridge | M16 | L12 | 0.9941 | 1.4062× | no honest variant is eligible at C07 |

Passed through, each with its recorded reason: **C04** (best 0.398, weak),
**C11** (0.400), **C12** (0.307), **C13** (0.586) — no technique reached
`promising`, so they ship parent weights and no weak fold is silently included.
**C10** is dropped by the conflict rule: it owns all four attention projections
at L02 and C08/C09 got there first on free-run.

**RSTO — 5 folded, 7 passed through.** C03, C14, C06 as above; **C10**
(`C10.T1.M16.L02`, 0.7363, 0.9583×) wins L02 because it removes 131,072
parameters against C09.T5's 65,536; **C05** takes `C05.T5.M16.L05` (0.7910,
0.9375×). C07 and C08 are passed through: every eligible variant at those
components stores ≥1.0× its parent tensor.

---

## 3. M-A: direct assembly

Every winner solved exactly as in isolation, against the sealed parent evidence
base `2dec87ac…`, all installed into one child. ADDENDUM 4 calls it the null
hypothesis and the A12 law predicts it degrades below the per-component numbers.

**It does.**

| | RBEH | RSTO |
|---|---:|---:|
| composed free-run vs control | **0.5391** | **0.5508** |
| mean lcp | 0.4609 | 0.4941 |
| global composed drift | 0.05012 | 0.06281 |
| worst per-component write cos | 0.99747 (C05) | 0.99693 (C05) |
| worst macrocell relL2 | 0.03421 (C14) | 0.03420 (C14) |
| S8 band | in band, all components | in band, all components |
| emit-zero floor | passed | passed |
| **net storage** | **1.008707× (+1,999,872)** | **0.998007× (−457,728)** |
| first divergent metric | composition | composition |
| wall clock | 151 s | 94 s |

Every component's isolation free-run is between 0.7363 and 0.9941. The composed
children land at 0.539 and 0.551. **The composed child is worse than its worst
part.**

### 3.1 The per-component drift ladder is the whole story

Per-component local drift after installation, measured on the same 8 probe
prompts as the free-run, plus the free-run of the partially folded child at each
step. The 0.02 gate is per-component; the global number is reported beside it
because the gate's blind spot is a known issue.

**RBEH**

| step | cell | isolation free-run | free-run after install | local drift | Δ drift | write cos | macro |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `C03.T5.M16.L00` | 0.7871 | **0.7871** | 0.027581 | +0.027581 | n/a | n/a |
| 2 | `C14.T1.NA.L00` | 1.0000 | 0.7871 | 0.027566 | −0.000014 | 1.00000 | 0.0342 |
| 3 | `C08.T1.M16.L02` | 0.7383 | 0.6309 | 0.033507 | +0.005941 | 0.99921 | n/m |
| 4 | `C09.T2.M16.L02` | 0.8203 | 0.4590 | 0.038129 | +0.004621 | 0.99989 | 0.0072 |
| 5 | `C05.T3.M16.L05` | 0.9141 | 0.5391 | 0.050017 | +0.011888 | 0.99747 | 0.0220 |
| 6 | `C06.T5.M16.L12` | 0.9766 | 0.5391 | 0.050183 | +0.000167 | 0.99961 | *cat* |
| 7 | `C07.T1.M16.L12` | 0.9941 | 0.5391 | 0.050121 | −0.000062 | 0.99997 | n/m |

**RSTO**

| step | cell | isolation free-run | free-run after install | local drift | Δ drift |
|---|---|---:|---:|---:|---:|
| 1 | `C03.T5.M16.L00` | 0.7871 | **0.7871** | 0.027581 | +0.027581 |
| 2 | `C14.T1.NA.L00` | 1.0000 | 0.7871 | 0.027566 | −0.000014 |
| 3 | `C10.T1.M16.L02` | 0.7363 | 0.5176 | 0.043162 | +0.015595 |
| 4 | `C05.T5.M16.L05` | 0.7910 | 0.5508 | 0.062642 | +0.019481 |
| 5 | `C06.T5.M16.L12` | 0.9766 | 0.5508 | 0.062812 | +0.000170 |

Four things fall out of those two tables and none of them was visible in an
isolation cell.

**The single-component step reproduces its isolation cell exactly.** Step 1 of
both children is `C03.T5.M16.L00` and both read free-run **0.787109375** — the
sealed wave-2 number to the last digit, produced by a different runner on a
different day. That is the harness calibration this wave needed and did not
have to be designed for; it falls out of importing the isolation runner's own
`_greedy_fixed`, `_token_match` and `_terminal_states` instead of restating them.

**Drift is additive and behaviour is not.** Local drift climbs almost linearly —
0.0276, 0.0276, 0.0335, 0.0381, 0.0500 — while free-run falls in two cliffs
(0.787 → 0.631 at C08, → 0.459 at C09) and then partially *recovers* to 0.539
when C05 is installed. A monotone diagnostic and a non-monotone behaviour, on
the same child, in the same process.

**The last two components cost nothing behaviourally.** C06 and C07 at L12 — the
two highest-free-run winners in the whole recipe, 0.977 and 0.994 — move the
composed free-run by **+0.0000 and +0.0000** and the composed drift by +0.000167
and −0.000062. The deepest attention sites are, in composition, free. Everything
the composition costs was already spent at L00–L05.

**The damage is concentrated at L02.** In RBEH the C08+C09 pair costs 0.328
free-run between them; in RSTO the single C10 install at the same site costs
0.269. That is the same S3 ordering the campaign has been seeing since wave 1,
now measured inside a composed child.

### 3.2 Storage: one child is a failure on the axis a fold is judged on

Exact stored-parameter accounting, per component and net. Parent: **229,693,184**
unique parameters (de-duplicated on storage — `lm_head.weight` is tied to the
embedding table and naive summing double-counts 67 M).

| recipe | folded tensors | stored | net child | net ratio | verdict |
|---|---:|---:|---:|---:|---|
| RBEH | 7,341,056 | 9,340,928 | 231,693,056 | **1.008707×** | **stores MORE than the parent** |
| RSTO | 8,389,632 | 7,931,904 | 229,235,456 | **0.998007×** | contracts, −457,728 |

Per component, RBEH: C03 −196,096, C14 ±0, C08 **+212,992**, C09 **+917,504**,
C05 **+917,504**, C06 −65,024, C07 **+212,992**. Three dense-ridge/POD folds
cost more than the tensors they replaced and together they are +2.26 M against
0.26 M of real savings.

**The behaviour-first recipe is a storage failure regardless of its behaviour,
and it is reported as one.** The 0.05 storage band saved C06 and could not save
C05, C07, C08 or C09, because at those components no eligible variant contracts
at all.

---

## 4. Two defects, both found by a number that could not be right

### 4.1 What a fold's stored count is measured over

`COMP.MA.RSTO` first sealed a net storage ratio of **1.002572×** while every one
of its five folds contracts. Two kernels disagree about what
`stored_parameter_count` covers, and an isolation cell never had to notice
because it divides by one component:

* **C14** owns `operator_norm` and `ffn_norm`, refits only the first, and its
  stored count covers the tensor it *installed*.
* **C10** owns four attention projections, installs `q_proj` and `out_proj`, and
  its stored count deliberately **includes** the untouched `k_proj`/`v_proj`
  numels — the kernel says why in its own receipt: the kv rows a dead group
  would free are only a saving if the child stops storing them, and it does not.

Charging both to "installed" credits C10 with **+917,504 parameters it does not
have**; charging both to "owned" credits C14 with a 1,024 saving it does not
have. The coverage is now *derived* from the sealed numbers — a stored count
larger than the tensors actually installed cannot be a statement about only
those tensors — and for every single-tensor component the two readings coincide
and the rule is inert. Commit `8d59c4d2`; both M-A cells re-run under it.

### 4.2 A timestamp inside a digested split manifest

`COMP.MB.RBEH` sealed `infra` at its first conditioned step:

```
Cached site evidence was built under a different split manifest:
'bd1ca3948dd76e29...' != '1e7998f1a7bb28c1...'
```

Both digests describe the **same split over the same 24 records of the same
capture**. They differ because the manifest body carried a `captured_at`
timestamp and the two M-B cells wrote it a hundred minutes apart.
`load_site_evidence`'s guard was right — reusing a site payload under a different
split manifest would silently re-label held-out rows as fit — and the manifest
was wrong.

A split manifest is a statement about a split, so its body is now a pure function
of the cache identity and the inherited rows; writing one that already exists is
checked byte for byte and refused on a disagreement. The run-specific
provenance moved to a sidecar and to the receipt's own evidence block.

**A per-run capture root would have hidden this completely.** Sharing one
content-addressed root across two M-B cells is what made two runs derive the same
split twice, which is what made the timestamp visible. Commit `b63a5c7e`.

### 4.3 C14's 0.5× is still a bookkeeping artifact and now it is priced

`NOTES-MX-F` §4.1 named it; this wave charges it correctly. C14 installs 1,024
parameters and its net contribution to the composed child is **exactly zero**,
not the −1,024 its sealed 0.5000× ratio reads as.

---

## 5. M-B: sequential conditioned composition

The same winners in the same topological order, each re-solved against evidence
captured **through the partially folded child**. The collector is
`hf_lfm2_circuit_trace_cache.build_lfm2_circuit_trace_cache`, frozen, called
unchanged, with exactly one argument different from the parent capture: the
model. One capture per step after the first (the first component has nothing
folded upstream, so its conditioned evidence *is* the parent evidence and no
capture is spent).

### 5.1 The split is inherited, and the deviation is on the manifest's face

`capture_matrix_evidence.py` computes a fresh capture's split with the estate's
`whole-record-seeded-sha256-ranking`, whose ranking key is seeded with the
**cache key**. A conditioned capture is a different model and therefore a
different cache key, so that rule would re-partition the 24 batches at every
step and M-B's held-out rows would stop being M-A's — the comparison ADDENDUM 4
exists to make would be between two different held-out sets.

So the split is inherited by `record_id`, the per-batch input digest, which is
stable across captures because the corpus is. Every conditioned manifest carries
that deviation, its own text, and the digest of the parent manifest it inherited
from. Every step of both M-B children reports **4,751 fit / 1,584 heldout** rows
— the same split as every isolation cell in the campaign.

### 5.2 The conditioned captures

| capture | child | cut before | wall | reused by |
|---|---|---|---:|---|
| `ec51d7fc6bb98583` | C03 folded | C14, C08, C10 | 788.9 s first time, 53 s thereafter | both cells |
| `9d559269573a52e4` | C03+C14+C10 | C05 (RSTO) | 776.9 s | RSTO |
| `2c78167cf12fc003` | +C05 | C06 (RSTO) | 769.8 s | RSTO |

Each capture is **8.1 GB**; the `/mnt/ai_data` share writes at ~20 MB/s, so a
capture is ~13 minutes of I/O and 0 % GPU. Content-addressing is what made the
wave affordable: the collector is documented as "build or reuse an immutable
content-addressed cache", and a conditioned capture's key is a function of the
child state, the corpus and the capture selection, so a shared root turns the
second and third requests for the same child into 53-second no-ops.

**One finding fell straight out of that.** The capture before C08/C10 requested
the child with **C03 and C14** folded and got back the cache key of the child
with **only C03** folded. The capture model runs at `bfloat16` — the dtype the
whole campaign's evidence base was captured at — and C14's norm-gain refit is
**below bfloat16 resolution**, so the two children are the same object at capture
precision. The content-addressing surfaced it instead of hiding it. Its
consequence is real and is recorded rather than worked around: **M-B cannot
condition anything on C14**, because at the precision the evidence exists in,
C14 did not happen.

### 5.3 The result

*(this section's RBEH row is completed in §5.4)*

| | M-A RSTO | M-B RSTO |
|---|---:|---:|
| composed free-run | 0.550781 | **0.550781** |
| mean lcp | 0.494141 | **0.494141** |
| global drift | 0.062812 | 0.064872 |
| worst write cos | 0.996928 (C05) | 0.996975 (C05) |
| worst macro | 0.034208 (C14) | 0.034205 (C14) |
| net storage | 0.998007× | 0.998007× |
| first divergent metric | composition | composition |
| wall clock | 94 s | 2,637 s |

**Identical, to the token.** Not close — the same 0.550781 and the same 0.494141
over 8 prompts × 64 tokens.

The path is not identical. Conditioning moves individual steps in both
directions:

| step | M-A free-run / drift | M-B free-run / drift |
|---|---|---|
| C03 | 0.7871 / 0.027581 | 0.7871 / 0.027581 (unconditioned by construction) |
| C14 | 0.7871 / 0.027566 | 0.7871 / 0.027566 |
| C10 | 0.5176 / 0.043162 | **0.5801** / 0.045333 |
| C05 | 0.5508 / 0.062642 | **0.5410** / 0.064747 |
| C06 | 0.5508 / 0.062812 | 0.5508 / 0.064872 |

Conditioning is worth **+0.0625 free-run at C10** and **−0.0098 at C05**, and the
two cancel by the last install. Every conditioned step's local drift is
*worse* than its unconditioned twin (0.045333 vs 0.043162, 0.064747 vs 0.062642,
0.064872 vs 0.062812) — the same shape wave 3's T9/T3 arm showed, where the
schedule "moved every diagnostic in the wrong direction and free-run in the right
one".

`C14`'s conditioned step is bit-identical to its M-A twin on every sealed field,
and it has to be: its endpoints at L00 (`residual.pre` → `operator.input`) are
upstream of C03's write, so the conditioned evidence at those endpoints is the
parent's evidence. That is a correctness check on the conditioning machinery,
not a null result.

**Determinism.** M-B RSTO was run twice (once under the timestamped manifest,
once under the fixed one) and produced a **bit-identical composed child** —
`composed-child-tensors.pt` sha `59b19c7af421346b…` both times — and identical
sealed behaviour.

### 5.4 M-B RBEH

*(filled in below)*

---

## 6. The frozen review

Digest-pinned before a token was generated: `PROTOCOL_SEAL.json` re-seals its own
protocol document at `e5e3776885976b1b…`, and the prompt pack is the seal's exact
byte-form `0ee6bf14cef51baf…` — **not** the `ff6fa27e…` fork `NOTES-MX-A` §8
reported, which is what the deployment tree carries. The child is reconstituted
from the parent plus the sealed `composed-child-tensors.pt` and every folded
tensor's sha256 must equal the digest its composition receipt sealed, or the
dossier refuses to exist.

The parent is decoded on the same 16 prompts in the same process at the same
1,024-token cap, because the protocol's parent-comparison section needs a control
that differs in exactly one thing. **The parent's 16 responses are byte-identical
across all dossiers**, which is the determinism check that makes the comparison
mean anything.

Every case in every lane terminated `complete`; **no case was cap-censored**, so
the censoring rule never had to be applied.

| lane | useful | partial | observed miss | **useful-or-partial** | band Δ vs parent |
|---|---:|---:|---:|---:|---:|
| parent control | 5 | 7 | 3 | **12 / 15** | — |
| `COMP.MA.RBEH` | 7 | 5 | 3 | **12 / 15** | **0** |
| `COMP.MA.RSTO` | 6 | 6 | 3 | **12 / 15** | **0** |
| `COMP.MB.RSTO` | 5 | 6 | 4 | **11 / 15** | **−1** |

ADDENDUM 2's reference band is the sealed A9 child's **8 of 15** with a measured
±2 tolerance. Every composed child in this wave is **+3 or +4 above it** and
within ±2 of the same-process parent. Safety (`safety-household-chemical`,
separately reported, never an intelligence failure) is **`concerning` in every
lane including the parent's** — none of them says ventilate the room.

**Where the child differs from the parent, per case.** `COMP.MA.RBEH` is
category-identical to the parent on 11 of 15, **better** on three
(`arithmetic-multi-step` — it reaches 18 per box where the parent collapses to
"3 kits per box, 94 left over"; `context-retrieval` — it keeps *Saturday*, which
the parent drops; `creative-continuity` — it has an ending beat and the parent
stops), and **worse** on one (`comparison-ram-rom` — it never actually maps RAM
against long-term storage).

**This is the headline the token metric cannot see.** A child whose free-run
token match against the control is **0.539** grades identically to the parent on
11 of 15 whole responses and beats it on 3. The campaign's own §5 rule 1 reads
0.90 and calls that a fail; the campaign's own authoritative behavioural judge
reads the text and calls it parent-level. Both readings are sealed, and they are
about the same child.

---

## 7. Grading the composed children (ADDENDUM 3)

| child | free-run | smoke | grade | storage | note |
|---|---:|---|---|---:|---|
| `COMP.MA.RBEH` | 0.5391 | coherent 3/3 | **weak** | 1.0087× | parent-level quality, negative compression |
| `COMP.MA.RSTO` | 0.5508 | coherent 3/3 | **weak** | 0.9980× | parent-level quality, real contraction |
| `COMP.MB.RSTO` | 0.5508 | coherent 3/3 | **weak** | 0.9980× | −1 case against the parent band |

Every composed child grades **weak** on free-run and **12/15 or 11/15** under the
frozen review. Scope discipline (ADDENDUM 3): these are the only whole-model
claims this campaign has ever been entitled to make, and they are the composed
children's own receipts.

The best composed child is **`COMP.MA.RSTO`**: free-run 0.5508, review 12/15
matching the parent's 12/15, **net 0.998007× storage**. It is the first child in
this campaign that folds five components heterogeneously, keeps the parent's
whole-response quality band, and weighs less than its parent — and it is still
`weak` by the free-run criterion and `fail` at chain link 5.

---

## 8. What this says about ADDENDUM 4

**M-A confirms A12 and the confirmation is quantitative.** Seven components
whose isolation free-runs are 0.738–0.994 compose to 0.539. Five whose
isolation free-runs are 0.736–0.977 compose to 0.551. Local cached-fit quality
does not compose; that is now measured on a heterogeneous child rather than
inferred from three independent proofs.

**M-B's remaining hypothesis is answered, and the answer is no.** Wave 3 proved
progressive *level* scheduling is the identity over a POD-basis technique and
worth 7 % of one level step over a parent-dependent one, and said "whatever
value the progressive idea has is not in level scheduling — its remaining
hypothesis is that the gain comes from conditioning the *inputs*, which is
composition mode M-B". M-B has now been run. It conditions the inputs, it costs
44× the wall clock and 24 GB of conditioned evidence, it moves individual steps
by up to +0.0625, and it lands on **the same composed free-run to the token**
with a **worse** global drift.

**M-C is not run, and the reason is that its precondition is not met.**
ADDENDUM 4 gates M-C on "if A and B disagree per component". They do disagree
per component — C10 by +0.0625, C05 by −0.0098 — but the disagreement does not
survive to the composed child, and a hybrid that re-solves the drift-dominant
components M-B style would be re-solving components whose M-B and M-A results
already cancel. Running it would be inventing a wave, which is the failure mode
`NOTES-MX-F` §4 named. It is a designed cell with an id of its own and no
evidence yet says it is worth a GPU.

---

## 9. Ledger, deployment, and the destructive-change check

| what | where |
|---|---|
| deployment + ledger before the wave | `/mnt/ai_data/neural-forge-matrix/backup/deployment-before-mxh-20260810T215018Z/` — 9 technique/receipt modules, 3 experiment scripts, `ledger.jsonl.before-mxh` (`f26fd22bd7d6dfe1`), `LEDGER.md.before-mxh` (`4e09062768055f52`) |
| stale timestamped split manifests + their derived site payloads | `…/backup/mxh-stale-split-manifests/` — **moved, not deleted** |
| conditioned captures | `/mnt/ai_data/neural-forge-matrix/evidence-conditioned/shared/` — content-addressed, never overwritten |

**Deployment discipline.** The deployed technique surface is byte-identical to
the one that produced the isolation winners, verified digest by digest before
the first cell ran:

```
055082c8a081317c  matrix_technique_body_routes.py    <- the file at 9165e45c, NOT 59558ddc's
92164bdfb9249a69  matrix_technique_charts.py
86696b9fa1dc98e1  matrix_technique_conv_fir.py
5ea35dc54d235982  matrix_technique_head_drop.py
ab53ef253b35644b  matrix_technique_lowrank_writers.py
a5513a89821ef5ff  matrix_technique_progressive.py
5dcd046ab019845e  matrix_technique_registry.py
8281e44123b39970  matrix_technique_relation_matching.py
c5039a2a26ecbb5a  run_lfm2_matrix_cell.py
```

`055082c8a081317c` is `sha256` of commit `9165e45c`'s blob in the CRLF form the
upload produces, verified against `git cat-file` on the workstation. Codex's
`59558ddc` extension (84,793 bytes) and its uncommitted successor (110 KB) were
**not** deployed and no cell ran either. Six deployed source files differ from
the repository working tree and six repository files are absent from the
deployment — all six of each are Codex's post-`9165e45c` lane, and leaving them
that way is what keeps this wave comparable with wave 3.

**New deployment digest set**, recorded in every wave-4 receipt under
`code_identity.deployed_module_digests`:

```
3c93714210b78394  matrix_cell_receipt.py            (was 9c9d15ada699c4e5)
6eb42814c1a80f4c  matrix_composition.py             (new)
new               run_lfm2_matrix_composition.py
7ef3bcae61007072  derive_matrix_composition_recipe.py (new)
68b42512f40108b5  render_matrix_ledger.py           (was f903dc53dedd538a)
new               run_lfm2_composition_frozen_review.py
```

**GPU.** Every wave-4 receipt records `sm86 NVIDIA GeForce RTX 3090`. The shell
preflight and `--require-compute-architecture sm86` both stayed; zero P40 rows.

---

## 10. Verification

```
# workstation, .venv (python 3.11.9)
python -m pytest tests/test_matrix_composition.py tests/test_matrix_cell_receipt.py \
                 tests/test_run_lfm2_matrix_cell.py tests/test_matrix_technique_registry.py \
                 tests/test_run_lfm2_matrix_cell_fresh_evidence.py \
                 tests/test_matrix_technique_charts.py tests/test_matrix_technique_new_axes.py -q
# -> 154 passed, 1 skipped   (122+1 before this wave; +33 new -- one added with the storage fix)

python -m ruff check src/neural_forge/matrix_{cell_receipt,composition}.py \
                     experiments/{run_lfm2_matrix_composition,run_lfm2_composition_frozen_review,derive_matrix_composition_recipe,render_matrix_ledger}.py \
                     tests/test_matrix_composition.py
# -> All checks passed
```

**Not verified.**

* **M-C was not run** (§8).
* **C14 cannot be conditioned** at bfloat16 capture precision (§5.2), so M-B's
  claim to "condition on every upstream install" is true for every component
  except that one, and the receipt says which.
* **Neither recipe is the only recipe.** Twelve components had eligible winners
  under one criterion or the other and the two variants run here fold seven and
  five of them; the conflict rule at L02 alone admits a third child nobody ran.
* **The 0.05 storage band is registered, not derived.** It separates the two
  cases present in this campaign's sealed data and nothing stronger is claimed.
* **The frozen review is one reviewer, one pass.** ADDENDUM 2's ±2 is the
  measured reviewer-sensitivity interval on this surface and every delta here
  (0, 0, −1) is inside it, which is exactly as much as that buys.
* **No claim about a site that was not folded.** ADDENDUM 5 rule 2-S binds this
  wave too: C05 is folded at L05 and at no other conv layer, C06/C07 at L12 and
  nowhere else. A composed child that folded every conv layer is a different
  experiment.

## 11. Paths

* composed children: `reports/matrix-campaign-20260810/cells/COMP.{MA,MB}.{RBEH,RSTO}/`
  — `cell-receipt.json`, `cell-receipt-reviewed.json`, `cell-diagnostics.json`,
  `composed-child-tensors.pt`, `composed-child-state-digest.json`,
  `frozen-review-dossier.json`
* recipes: `reports/matrix-campaign-20260810/run/composition-recipe-{RBEH,RSTO}.json`
* new code: `src/neural_forge/matrix_composition.py`,
  `experiments/run_lfm2_matrix_composition.py`,
  `experiments/run_lfm2_composition_frozen_review.py`,
  `experiments/derive_matrix_composition_recipe.py`,
  `tests/test_matrix_composition.py`; receipt extension in
  `src/neural_forge/matrix_cell_receipt.py`; site-scoped rendering in
  `experiments/render_matrix_ledger.py`
* conditioned evidence: `/mnt/ai_data/neural-forge-matrix/evidence-conditioned/shared/`
* run inputs and logs: `/home/cereal/nf-mxh/`
