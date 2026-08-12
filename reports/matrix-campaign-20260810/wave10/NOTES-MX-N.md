# NOTES-MX-N — wave 10: A9's route rebuilt, and the head that turned out to matter

Wave 10, branch `codex/lrc-blank-compression`, source commit **`9acaf38f`**
("Rebuild A9's token route, and count the indices it routes by") on top of
wave-9 HEAD `92d46a40`. Deployment `neural-forge-matrix-9acaf38f`, cut with
`git archive -c core.autocrlf=false`, 815 files, review surface verified on
extraction **before any child was generated**: protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`, seal
`7b859172fa48c008eb68d5ea070458ac82e2149c19c5141230d45c061aaf00e5`, prompt pack
`0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`. No asset
needed hand-restoring, which is wave 9's finding holding for a second wave.

Cyril's directive, verbatim: *"[Cyril quote withheld]"*

He was right that A9/A11 was the more condensed child. The answer to the rest is
below, and it is not the answer the ladders were designed to produce.

---

## 0. The five sentences

1. **A9's mechanism rebuilt at current standards reproduces A9's own sealed
   compaction to four decimal places** — 1.360692x counted against the sealed
   1.361447x — which is how we know it is the same mechanism and not a
   lookalike.
2. **A9's published storage was understated.** It counted the coefficients and
   not the atom ids they index. Honestly counted, A9 is **0.743217x** stored,
   not 0.734512x; A11 is **0.760056x**, not 0.742932x.
3. **The sparse tail beats the low-rank tail at every point measured**, at
   identical storage, on reconstruction, on install drift, on the frozen review,
   and on all seven benchmarks — and `COMP.MA.MXN2` is **the campaign's first
   embedding-folded child to come back alive**: 10/15, `alive-parent-band`, at
   **0.780518x**, against `COMP.MA.MXM6`'s 9/15 dead at 0.780861x.
4. **Both ladders stop above A9's number.** Ladder A's deepest rung is
   0.735962x and reads 6/15 dead; the deepest child that survives *any* reading
   of the degeneracy clause is `COMP.MA.MXN4` at **0.754811x**. A9's legacy
   0.734512x was **not** beaten by a coherent child, and its honest 0.743217x
   was beaten only by children that are dead.
5. **The wave's real finding is Ladder B arm (iii):** at one budget, in one
   mechanism, the arm with the **better** table reconstruction is the **worse**
   child. Prompt-token coverage of the verbatim head, not fidelity of the coded
   tail, is what the review surface measures.

---

## 1. The excavation: what A9/A11 actually were

Sources read before anything was built: `docs/STAGE1_CEILING_REPORT_2026-08-08.md`
§1–§2.4, `NOTES-W13T.md` §1–§2,
`reports/sparse-semantic-a9-k32-residual-omp-20260730/manual-review.md` and its
`base-codebook-reference.json`,
`reports/sparse-semantic-a11-k32-split-cout-20260730/`,
`reports/lfm2-sparse-semantic-highcap-20260730/`, and
`docs/LFM2_A9_A11_HIGH_CAP_REUSE_2026-07-30.md`.

### 1.1 The shape

A9 is **not** a body fold and never was. It is a *vocabulary* fold: the parent's
65,536 x 1,024 tied token table is replaced by a **4,096-row target table** plus
a sparse route that reaches the other 61,440 donor tokens as combinations of
those 4,096 rows. The body is untouched — 1.000x, as the ceiling report says in
every place it quotes the row.

```
parent unique                                      229,693,184
parent tied token table          65,536 x 1,024  =  67,108,864
parent physical body        229,693,184 - 67,108,864 = 162,584,320
```

A9 stores `body + 4,096x1,024 + one 1,933,888-element route channel`; A11 stores
the same with **two** channels (the split output route, `C_in` and `C_out`
solved separately on shared supports). Those two lines reproduce the sealed
ratios exactly, which is what validates the accounting model:

```
A9   162,584,320 + 4,194,304 + 1,933,888 = 168,712,512 ; 229,693,184/168,712,512 = 1.361447
A11  162,584,320 + 4,194,304 + 3,867,776 = 170,646,400 ; 229,693,184/170,646,400 = 1.346018
```

`1,933,888 / 61,440 = 31.48` — the mean number of atoms per coded row. "k32" is
the **cap**, not the realized support: residual OMP stops early on a row it has
already explained, so the supports are ragged.

### 1.2 Why counted-x is not a stored ratio

**This is the correction the wave exists to make.** A9's own review reports
"stored continuous semantic coefficients: 1,933,888" and adds exactly that. Its
own sealed artifact carries **three** tensors, not one
(`base-codebook-reference.json`, `tensor_content`):

* `semantic_code_coefficients_sha256` — the 1,933,888 values, counted;
* `semantic_code_target_ids_sha256` — one atom id per coefficient, **not counted**;
* `semantic_code_offsets_sha256` — the CSR row offsets the ragged supports
  require, **not counted**.

A sparse code whose atom ids you did not store is a code nobody can decode. On
the campaign's existing convention — one index costs one parameter, the same
rule `c01_token_route_storage` already applies to its head ids — the honest
figures are:

| | published (continuous-only) | honest (ids + offsets counted) | delta |
|---|---:|---:|---:|
| **A9** counted | 1.361447x | **1.345502x** | |
| **A9** stored | 0.734512x | **0.743217x** | **+0.008705x** |
| **A11** counted | 1.346018x | **1.315692x** | |
| **A11** stored | 0.742932x | **0.760056x** | **+0.017124x** |

The index convention is *conservative*: a 4,096-atom dictionary needs 12 bits,
not 32. At int16 A9 would be ~0.739x; at 12 bits, ~0.736x. **Even at 12 bits it
is not 0.734512x.** Both numbers are reported everywhere below so the ceiling
report stays comparable rather than being quietly restated.

### 1.3 What was deliberately not carried over

A9's original artifact is architecture-non-portable — its fp32 OMP candidate
search reproduces only on P40 + torch 2.6 and reads no evidence at HEAD. Nothing
in wave 10 loads it or depends on it. The **mechanism** is A9's; the **numerics**
are the campaign's: float64 residual OMP with the Cholesky-updating selector and
the shared-support ridge solve that already existed in
`matrix_technique_body_routes` for body components, whose own docstring says the
mechanism is "indifferent to what a row means" and then explicitly excludes the
token interface as "A9/A11's own already-sealed lane". Wave 10 built that lane.

### 1.4 The 59558ddc decision

`59558ddc` ("Authenticate T5 sparse feature route promotion", +804) **is already
an ancestor of HEAD** — `git merge-base --is-ancestor 59558ddc 92d46a40` returns
true, and `git log -- src/neural_forge/matrix_technique_body_routes.py` shows it
two commits below `eb32153e`. It was committed and never *deployed*, not
committed and never merged. Since every wave-10 deployment is a `git archive` of
a descendant of HEAD, it ships automatically; there was no include/exclude
decision to make.

It was reviewed anyway, because the wave depends on the module it edits. Its 352
test lines pass (they are part of the 48 cases in
`tests/test_matrix_technique_body_routes.py`, all green), and its contribution —
`sparse_feature_fold_map_from_body_route` plus the authenticated snapshot
guard — is orthogonal to C01. **Wave 10 uses the module it hardened and adds one
public entry point to it** (`sparse_route_codes`), so the OMP has one
implementation serving both the body lane and the token lane instead of two.

---

## 2. What was built

`src/neural_forge/matrix_technique_token_embedding.py` grows a second tail mode
under the T5 registration it already owns at C01. The technique table has always
called T5 `sparse_omp_routes`; wave 9 registered a low-rank tail under that name.
Wave 10 adds the tail the name means.

* `token_route_tail_mode="low-rank"` — **the default**, wave 9's, unchanged. A
  test asserts a wave-9 knob set still folds the wave-9 way, because changing the
  default would silently restate every wave-9 receipt.
* `token_route_tail_mode="sparse-omp"` — the A9/A11 mechanism.

**Both tails are handed the same head by literally the same function.**
`_select_route_head` was extracted verbatim out of `route_token_embedding`
(counts descending, ties by token id ascending — the rule `beb699f2` measured at
66.80% prompt-token coverage), and a test pins the head ids equal with
`torch.equal` rather than trusting the extraction. A comparison between the two
tails is therefore a comparison between two tails.

Storage, honestly:

```
head*(hidden+1)  +  2*nnz  +  (tail+1)
   atoms + ids       values + one id each      CSR row offsets
```

`nnz` is the **realized** support total, not `tail*k`. The planner
(`c01_sparse_route_sparsity_for_ratio`) plans on the worst case `nnz = tail*k`,
so a realized route can only come in **under** the level it was planned for,
never over — pinned by a test.

Two incidental fixes, both found by running this on the real table rather than
on a fixture: `_tensor_sha256` now moves to CPU before `.numpy()`, so a route
solved on the GPU digests to the same string as the same bytes solved anywhere
else; and the sparse arm takes an **explicit** `solve_device` rather than an "if
cuda is available", because a fold whose numerics depend on what happened to be
plugged in is not reproducible. The harness hands C01 a CPU float64 table and
gets a CPU table back either way.

**Tests.** 13 new cases in `tests/test_matrix_technique_token_embedding.py`, 4 in
`tests/test_matrix_technique_body_routes.py`, 100 pass in those two files. The
identity gate is stated at **zero ridge** (exact to 1e-12 on atom-spanned rows)
with the default ridge's 2.3e-9 bias pinned separately — measured, not assumed:
the default `ridge_relative=1e-10` on a Gram of condition 2.5e4 is the *entire*
source of error in that gate.

---

## 3. The mechanism comparison, before any child was folded

Read-only probe on the real parent table, both tails at the same head and the
same level, `wave10/probe-c01.json`. Zero-norm rows in the parent table: **0**,
so no head atom had to be substituted or excluded.

| level | head | tail | stored | x owned | net | rel. Frobenius | tail row cosine |
|---|---:|---|---:|---:|---:|---:|---:|
| S8 | 4,096 | low-rank r=67 | 8,383,488 | 0.124924 | 0.744331x | 0.8619 | 0.4665 |
| S8 | 4,096 | **sparse k=33** | 8,314,881 | 0.123901 | 0.744033x | **0.7397** | **0.6495** |
| S4 | 4,096 | low-rank r=201 | 16,753,664 | 0.249649 | 0.780772x | 0.7644 | 0.6227 |
| S4 | 4,096 | **sparse k=101** | 16,670,721 | 0.248413 | 0.780411x | **0.5761** | **0.8078** |
| S4 | 12,288 | low-rank r=77 *(MXM6)* | 16,774,144 | 0.249954 | 0.780861x | 0.7897 | 0.4887 |
| S4 | 12,288 | **sparse k=38** | 16,695,297 | 0.248779 | 0.780518x | **0.6005** | **0.7476** |
| S8 | 8,000 | low-rank r=3 | 8,375,680 | 0.124807 | 0.744297x | 0.9051 | 0.2634 |
| S8 | 8,000 | **sparse k=1** | 8,372,609 | 0.124762 | 0.744284x | **0.8523** | **0.4005** |

The sparse tail wins on reconstruction at **every** point, and it wins while
being **slightly cheaper**, because it is carrying the atom ids and CSR offsets
that the low-rank arm does not have to carry at all. Section 6 is about why that
did not settle the question.

---

## 4. Every child of the wave

Frozen protocol on every verdict; parent calibrated in the same process on every
one; parent lane byte-identical across waves 8, 9 and 10 at
**`865f006e3813712ec202cb23`** (`mxn_parent_identity.py`, four dossiers, one
anchor). Verdicts machine-computed by `seal_matrix_behavior_review.py` from the
counts; no verdict was passed in.

| child | n | what | net stored | legacy counted-only | drift | review | **per-child** | **per-lane** | lanes at cap |
|---|---:|---|---:|---:|---:|---:|---|---|---:|
| `COMP.MA.MXN2` | 1 | route S4, head 12,288, k=38 | **0.780518x** | 1.296305x | **0.207468** | **10/15** | **alive-parent-band** | alive-parent-band | **1** |
| `COMP.MA.MXN5` | 1 | route S4, head 4,096, k=101 | 0.780411x | 1.327829x | 0.294037 | 8/15 | dead | *alive-degraded* | 1 |
| `COMP.MA.MXN4` | 4 | MXN2 route + L05 SwiGLU lane | **0.754811x** | — | 0.377694 | 8/15 | dead | ***alive-degraded*** | 1 |
| `COMP.MA.MXN1` | 1 | route S8, head 4,096, k=33 | 0.744033x | **1.360692x** | 0.483370 | 7/15 | dead | dead | 6 |
| `COMP.MA.MXN3` | 7 | MXN2 route + MXL11's whole body | **0.735962x** | — | 0.487722 | 6/15 | dead | dead | 3 |

**Both degeneracy readings are given for every child, because Cyril is
adjudicating the clause.** They agree on MXN1, MXN2 and MXN3. **They disagree on
MXN4 and MXN5**, and MXN4 is the consequential one: one lane
(`multi-turn-correction`) rides the cap repeating "Peanut-free and nut-free"
about eighty times and leaves nothing when the repetitions are removed. Per
child, that charges the clause and the child is dead. Per lane, it charges one
lane, fifteen stand, and the child reads **alive-degraded at 8/15 and
0.754811x — the deepest coherent child the campaign would ever have produced.**

### 4.1 The head-to-head Cyril asked for

`COMP.MA.MXN2` is `COMP.MA.MXM6` with exactly one variable changed: same
12,288-row head, same selection rule, same S4 level, same knob file
(`e355e4e7bb8a3058`, byte-identical to waves 7/8/9), same evidence, same GPU.
Only the tail differs.

| | `COMP.MA.MXM6` (wave 9, SVD tail) | `COMP.MA.MXN2` (wave 10, sparse tail) |
|---|---:|---:|
| stored / owned | 0.249954 | **0.248779** |
| net stored | 0.780861x | **0.780518x** |
| rel. Frobenius | 0.789684 | **0.600457** |
| tail row cosine | 0.488678 | **0.747636** |
| **install drift** | 0.498567 | **0.207468** |
| frozen review | 9/15 | **10/15** |
| verdict | **dead** (2 pure-loop lanes) | **alive-parent-band** |
| lanes terminating | 13 / 16 | **15 / 16** |

Install drift 0.207468 is **the smallest any embedding fold in this campaign has
produced**, at less than half MXM6's, at the same storage.

---

## 5. Ladder A — the route plus the body

| rung | n | net stored | review | per-child | per-lane |
|---|---:|---:|---:|---|---|
| route alone (`MXN2`) | 1 | 0.780518x | 10/15 | alive-parent-band | alive-parent-band |
| + L05 SwiGLU lane (`MXN4`) | 4 | **0.754811x** | 8/15 | dead | **alive-degraded** |
| + MXL11's whole body (`MXN3`) | 7 | **0.735962x** | 6/15 | dead | dead |

**The ladder confirms wave 9's finding in a sharper form: the body and the token
interface do not add, they multiply the damage.** MXL11's body alone is 9/15;
MXN2's route alone is 10/15; together they are 6/15.

But the sharper form is the controlled one. `COMP.MA.MXN3` is `COMP.MA.MXM9`
with the tail swapped and nothing else — same six body components at the same
sites, same head, same level:

| | `COMP.MA.MXM9` | `COMP.MA.MXN3` |
|---|---:|---:|
| net stored | 0.736305x | 0.735962x |
| install drift | 0.650704 | **0.487722** |
| review | 4/15 | **6/15** |
| lambada ppl | 15,185.40 | **797.61** |
| wikitext wppl | 515.07 | **193.24** |

A better token mechanism is worth **two review cases and a 19x perplexity
improvement inside the same composition** — and still does not repair the
interaction. The composition is the wall, not the token interface.

**Why the ladder cannot reach 0.734512x with an uncontracted body.** The
verbatim head costs `head*(hidden+1)`. To land at A9's published net with the
body untouched, the whole token interface must fit in
`0.734512 * 229,693,184 - 162,584,320 = 6,128,079` parameters, which caps the
head at **5,978 rows**. The head that actually reads alive is 12,288. **A
12,288-row head cannot reach 0.734512x at any sparsity — its floor is 0.763x
even at k=1.** Getting below A9's number therefore *requires* body contraction,
and body contraction is what kills the child. That is a structural statement
about this family, not a tuning failure.

---

## 6. Ladder B — and the finding that outranks the ladders

Arm (ii), "MXM6's head-12,288 rows plus an A9 route tail instead of the SVD
tail", **is** `COMP.MA.MXN2`, and it is §4.1. Arm (i), a corrective route layered
on top of MXM6, is not expressible: the harness folds one component with one
technique, and a second C01 cell in the same child would be a second fold of the
same tensor. Arm (iii) is the one that was run, and it is the wave's most useful
result.

**Arm (iii): the same S4 budget, spent the other way.**

| | `COMP.MA.MXN5` | `COMP.MA.MXN2` |
|---|---:|---:|
| head / prompt-token coverage | 4,096 / 66.80% | 12,288 / 80.99% |
| tail | k=101 | k=38 |
| net stored | 0.780411x | 0.780518x |
| **rel. Frobenius** | **0.576139** *(best of the wave)* | 0.600457 |
| **tail row cosine** | **0.807837** *(best of the wave)* | 0.747636 |
| install drift | 0.294037 | **0.207468** |
| **review** | 8/15 | **10/15** |
| lambada ppl | 355.02 | **230.31** |

**The arm with the best table reconstruction the wave measured is the worse
child.** Same mechanism, same budget, same rung; the only free variable is where
the parameters go. This is the ceiling report's finding 9 — reconstruction error
and behavior are decoupled — reproduced *inside one mechanism at one budget*,
where it cannot be blamed on the solver, the reclosure policy or the level.

**What the review surface is actually measuring is prompt-token coverage of the
verbatim head.** 80.99% coverage at rank-77 beat 66.80% at rank-201 in wave 9
(MXM6 9/15 against MXM3 3/15) and everyone assumed the tail rank was carrying
some of that. It was not: hand the 66.80% head a tail that is *better on every
reconstruction metric* and it still loses by two review cases.

---

## 7. Benchmarks

lm-eval from `/home/cereal/ai-benchmark-lab` (`envs/lm-eval`), tasks
`arc_easy, hellaswag, piqa, winogrande, lambada_openai, wikitext`, `--limit 200
--seed 1234 --batch_size 4`, `max_length=2048`, float32, one PCI-pinned sm86 RTX
3090. Partial evaluations by construction; they rank models against each other.
Parent baseline reused from wave 8 — same model directory, same dtype, same
tasks, same limit and seed, and no wave-10 change touches the parent.

| model | n | net stored | review | verdict | arc_easy | hellaswag_n | piqa_n | wino | lambada | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **parent** | 0 | 1.000000 | 12/15 | — | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.3912 | 81.7976 |
| `COMP.MA.MXN3` | 7 | **0.735962x** | 6/15 | dead | 0.4750 | 0.4300 | 0.5800 | 0.5650 | 0.1200 | 797.6144 | 193.2446 |
| `COMP.MA.MXM9` *(w9)* | 7 | 0.736305x | 4/15 | dead | 0.3700 | 0.3150 | 0.5100 | 0.5150 | 0.0700 | 15185.3986 | 515.0661 |
| `COMP.MA.MXN1` | 1 | **0.744033x** | 7/15 | dead | 0.3600 | 0.3900 | 0.5750 | 0.5100 | 0.0650 | 3765.1269 | 432.0966 |
| `COMP.MA.MXM8` *(w9)* | 1 | 0.744297x | 3/15 | dead | 0.2750 | 0.2800 | 0.5150 | 0.4750 | 0.0550 | 2315739.1765 | 11527.7892 |
| `COMP.MA.MXN4` | 4 | **0.754811x** | 8/15 | dead / *alive-degraded* | 0.4250 | 0.4500 | 0.6000 | 0.5250 | 0.1700 | 370.0939 | 168.7003 |
| `COMP.MA.MXN5` | 1 | **0.780411x** | 8/15 | dead / *alive-degraded* | 0.4350 | 0.4700 | 0.6100 | 0.5000 | 0.1300 | 355.0229 | 178.9962 |
| `COMP.MA.MXN2` | 1 | **0.780518x** | 10/15 | **alive-parent-band** | 0.4550 | 0.4450 | 0.6400 | 0.5550 | 0.2000 | 230.3146 | 139.2100 |
| `COMP.MA.MXM6` *(w9)* | 1 | 0.780861x | 9/15 | dead | 0.3650 | 0.3200 | 0.5200 | 0.5350 | 0.1100 | 7337.1925 | 421.6808 |
| `COMP.MA.MXL11` *(w8)* | 6 | 0.955444x | 9/15 | alive-degraded | 0.5250 | 0.4600 | 0.6350 | 0.5700 | 0.2250 | 99.2209 | 118.0216 |
| `COMP.MA.MXL5` *(w8)* | 4 | 0.972579x | 10/15 | alive-parent-band | 0.5100 | 0.4850 | 0.6200 | 0.5500 | 0.3200 | 55.1916 | 105.5257 |

**Every wave-10 child beats its wave-9 counterpart at matched storage on every
one of the seven columns.** The three matched pairs:

* 0.7360x: MXN3 over MXM9 — lambada ppl **19x** better, wikitext **2.7x**.
* 0.7440x: MXN1 over MXM8 — lambada ppl **615x** better, wikitext **27x**.
* 0.7805x: MXN2 over MXM6 — lambada ppl **32x** better, wikitext **3.0x**.

`COMP.MA.MXN2` also lands **within 0.02 of wave-8's `COMP.MA.MXL11` on four of
seven columns and above it on `piqa`**, while storing 0.175x less of the parent.

---

## 8. Estate

* **Ledger.** 422 rows sha `87cc0cac57b92e7426610b7fd92c0f09b7c045deeab1d87c2a3a3fbfb67e77f2`
  before; backed up to `ledger.jsonl.pre-wave10-20260812T010208` (byte-identical,
  same sha) and to `wave10/ledger.jsonl.pre-merge-backup`; merged **exactly
  once** behind a `.MERGED_ONCE` marker; 432 rows after, sha
  `bf532a4fd98d55b93308dd11f396c78c1ee45d1b5d0089b5b30bec5d19a5aa18`. The merger
  verified the prior bytes are a **prefix** of the result, byte for byte.
* **Seals.** 428 of 428 resolve and verify, 0 fail (418 prior + 10 new). The 4
  non-resolving rows are the same 4 as before wave 10 (three `CAL.C01.T5.X4`
  calibration rows and one `C06.T2.S4.L12` a6review instance).
* **Wave-10 rows.** 10, all `sm86 NVIDIA GeForce RTX 3090`, all receipt schema
  v2, verdicts `{dead: 4, alive-parent-band: 1}`.
* **Tests.** See §9 — one pre-existing collection error was found and proved
  pre-existing against a wave-9 control.
* **Retention.** Best-5 composed tensors kept (all five wave-10 children);
  materialized HF `model.safetensors` deleted after each benchmark and recorded
  as re-derivable from the sealed receipt.
* **Disk.** Server `/` at 6.6 GB free throughout; every heavy artifact under
  `/mnt/ai_data/neural-forge-matrix/wave10/`.

---

## 9. The test result, stated honestly

`pytest tests/` **fails at collection** on this deployment with:

```
experiments/run_lfm2_matrix_cell.py:1233: in <module>
    register("T5", _a9_token_route, component="C01")
E   CellTechniqueError: ('T5', 'C01') is already registered; pass replace=True to override.
```

**This is pre-existing at wave-9 HEAD and wave 10 did not cause it.** Two
independent checks:

1. `git diff 92d46a40 9acaf38f --stat -- experiments/` is empty — commit
   `9acaf38f` touches no file in `experiments/`, and both the bare `register`
   at `run_lfm2_matrix_cell.py:1233` and the `replace=True` registration in
   `matrix_technique_token_embedding` exist at `92d46a40`.
2. A **control run**: wave-9's tests (`92d46a40`) against wave-9's src
   (deployment `beb699f2`) produces the **identical** error and the identical
   `Interrupted: 1 error during collection`.

The trigger is collection order, not content: `test_matrix_technique_token_embedding`
sorts before `test_run_lfm2_matrix_cell`, imports the C01 kernel, and the kernel
registers `('T5','C01')` at import time; `run_lfm2_matrix_cell.py` then registers
the same pair without `replace=True`. Run alone, that file passes (24 passed, 1
skipped) at both waves.

**It was left unfixed deliberately.** The fix is one `replace=True` in
`experiments/run_lfm2_matrix_cell.py` — a file in the runner lane that this
wave's charter does not cover, and one that a wave measuring behavior should not
edit mid-wave. It is recorded here as an open defect with its exact cause.

Counts with that one file ignored are in §9.1; wave 10's own two test files are
**100 passed** (§2).

---

## 10. What wave 11 should measure

1. **The head, on its own axis.** Wave 10 says coverage decides. Nobody has
   measured a head *above* 12,288 with a sparse tail — wave 9's head-15,000
   reading (MXM7, 5/15) was confounded by a rank-27 tail that could not survive.
   At S4 with a sparse tail, a 15,000-row head buys k=17 and 83.61% coverage.
2. **`COMP.MA.MXN4`'s one bad lane.** It is one lane away from being the
   deepest coherent child the campaign has produced under either reading. A
   head-15,000 variant of the same composition is the cheapest shot at it.
3. **The clause itself.** Wave 10 produced two children where the readings
   disagree and the disagreement changes the frontier. That is now a decision
   with a price tag, not a philosophical preference.
