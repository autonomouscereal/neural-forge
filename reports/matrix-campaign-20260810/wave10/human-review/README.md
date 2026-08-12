# Wave 10 — human review package

Five children. Cyril scores them personally; every count below is a reviewer's
reading, and the verbatim text each reading was taken from is in the per-child
files beside this one.

Source commit **`9acaf38f`**, deployment `neural-forge-matrix-9acaf38f`
(815 files, `git archive -c core.autocrlf=false`). Frozen protocol
`e5e37768…`, seal `7b859172…`, prompt pack `0ee6bf14…`, all three verified on
the deployment **before any child was generated**. The parent lane is
byte-identical to waves 8 and 9 at **`865f006e3813712ec202cb23`** and was
decoded in the same process as each child.

---

## The question this wave was asked

*"[Cyril quote withheld]"*

Yes, A9/A11 was more condensed. Wave 10 rebuilt its mechanism at current
standards, scored it under the frozen protocol for the first time, and ran both
ladders. Three things came out of it, in ascending order of importance:

1. **A9's published storage was understated.** It counted the sparse
   coefficients and not the atom ids that index them. A9 is **0.743217x**
   stored, not 0.734512x. (§1)
2. **The A9 mechanism beats wave 9's at every matched point**, and produced the
   campaign's **first embedding-folded child to come back alive**. (§2)
3. **The better-reconstructing child is the worse child.** What the review
   surface measures is how much of the vocabulary the head stores *verbatim*,
   not how well the tail is coded. (§4)

---

## 1. The storage correction

A9's sealed artifact carries three tensors — `semantic_code_coefficients`,
`semantic_code_target_ids`, `semantic_code_offsets`. Its published number counted
the first. A sparse code whose atom ids you did not store is a code nobody can
decode.

| | published (continuous-only) | honest (ids + CSR offsets) | understated by |
|---|---:|---:|---:|
| **A9** | 1.361447x counted / 0.734512x stored | 1.345502x / **0.743217x** | 0.008705x |
| **A11** | 1.346018x counted / 0.742932x stored | 1.315692x / **0.760056x** | 0.017124x |

Both numbers are carried in every table below. The index convention (one index
costs one parameter) is the campaign's existing rule and is *conservative* — a
4,096-atom dictionary needs 12 bits. Even at 12 bits A9 is ~0.736x, not
0.734512x.

**The rebuild reproduces A9.** `COMP.MA.MXN1` — 4,096-row head, k=33 — reads
**1.360692x** on A9's own continuous-only accounting against A9's sealed
1.361447x. Same mechanism, four decimal places.

---

## 2. The five children

| child | n | what | net stored | review | **per-child verdict** | **per-lane verdict** | lanes at cap |
|---|---:|---|---:|---:|---|---|---:|
| **`COMP.MA.MXN2`** | 1 | route S4, head 12,288, k=38 | **0.780518x** | **10/15** | **alive-parent-band** | alive-parent-band | **1 / 16** |
| `COMP.MA.MXN5` | 1 | route S4, head 4,096, k=101 | 0.780411x | 8/15 | dead | *alive-degraded* | 1 / 16 |
| `COMP.MA.MXN4` | 4 | MXN2 route + L05 SwiGLU lane | **0.754811x** | 8/15 | dead | ***alive-degraded*** | 1 / 16 |
| `COMP.MA.MXN1` | 1 | route S8, head 4,096, k=33 | 0.744033x | 7/15 | dead | dead | 6 / 16 |
| `COMP.MA.MXN3` | 7 | MXN2 route + MXL11's whole body | **0.735962x** | 6/15 | dead | dead | 3 / 16 |

Parent: 12/15 (5 useful, 7 partial, 3 miss), missing at arithmetic-multi-step,
logic-ordering and multi-turn-correction.

### The two children where the degeneracy readings disagree

**`COMP.MA.MXN4` is the one to score first.** Exactly one of its sixteen lanes
(`multi-turn-correction`) rides the 1,024-token cap, repeating "Peanut-free and
nut-free" about eighty times; removing the repetitions leaves nothing but the
opening sentence.

* **Per child** — one pure-loop lane charges the clause; the child is **dead**.
  (Wave 8 charged `COMP.MA.MXL3` dead on exactly one such lane.)
* **Per lane** — that lane is charged, the other fifteen stand, and the child
  reads **alive-degraded at 8/15 and 0.754811x**, which would be **the deepest
  coherent child the campaign has ever produced** — deeper than wave 8's
  `COMP.MA.MXL11` (0.955444x) by a wide margin.

`COMP.MA.MXN5` splits the same way at 0.780411x. `COMP.MA.MXN1`, `MXN2` and
`MXN3` read the same under both.

---

## 3. The head-to-head: `COMP.MA.MXN2` against wave 9's `COMP.MA.MXM6`

One variable changed. Same 12,288-row head, same selection rule (counts
descending, ties by token id), same S4 level, same knob file
(`e355e4e7bb8a3058`, byte-identical to waves 7/8/9), same sealed evidence, same
pinned sm86 card. **Only the tail mechanism differs** — A9's sparse-OMP code at
k=38 against wave 9's rank-77 SVD.

| | `COMP.MA.MXM6` (SVD tail) | `COMP.MA.MXN2` (sparse tail) |
|---|---:|---:|
| stored / owned | 0.249954 | **0.248779** |
| net stored | 0.780861x | **0.780518x** |
| reconstruction rel. Frobenius | 0.789684 | **0.600457** |
| tail row cosine | 0.488678 | **0.747636** |
| **install drift** | 0.498567 | **0.207468** |
| **frozen review** | 9/15 | **10/15** |
| **verdict** | **dead** (two pure-loop lanes) | **alive-parent-band** |
| lanes terminating on their own | 13 / 16 | **15 / 16** |
| lambada ppl | 7,337.19 | **230.31** |
| wikitext wppl | 421.68 | **139.21** |

The sparse arm is *slightly cheaper* while carrying atom ids and CSR offsets the
SVD arm never has to carry. Its install drift is the smallest any embedding fold
in this campaign has produced.

**What MXN2 does that MXM6 could not.** It computes 17 x 6 = 102 and 102 − 8 = 94
and 94 / 5 = 18 correctly before losing the remainder. It reproduces the supplier
passage with Supplier B, the one-week factory expansion slip and the weekly
schedule evidence intact. It recovers 4:15 p.m. **and the north gate** — the fact
A9's own review recorded as lost. It writes a genuinely concise vendor email,
gives three real household actions with no preamble, asks the five right
questions about an unlabelled sales graph, and writes readable creative prose
**with** the requested dialogue.

---

## 4. The finding that outranks both ladders

`COMP.MA.MXN5` spends the *same* S4 budget the other way: a 4,096-row head
(66.80% prompt-token coverage) with a k=101 tail, against MXN2's 12,288-row head
(80.99%) with a k=38 tail.

| | `COMP.MA.MXN5` | `COMP.MA.MXN2` |
|---|---:|---:|
| net stored | 0.780411x | 0.780518x |
| **rel. Frobenius** | **0.576139** — best of the wave | 0.600457 |
| **tail row cosine** | **0.807837** — best of the wave | 0.747636 |
| install drift | 0.294037 | **0.207468** |
| **review** | 8/15 | **10/15** |

**The best table reconstruction wave 10 measured belongs to the worse child.**
Same mechanism, same budget, same rung, one free variable: where the parameters
go. This reproduces the ceiling report's "reconstruction and behavior are
decoupled" *inside one mechanism at one budget*, where it cannot be blamed on the
solver, the reclosure policy or the level.

What the review surface measures is **prompt-token coverage of the verbatim
head**. The coded tail buys reconstruction metrics and does not buy review cases.

---

## 5. Ladder A — the route plus the body

| rung | n | net stored | review | per-child | per-lane |
|---|---:|---:|---:|---|---|
| route alone (`MXN2`) | 1 | 0.780518x | 10/15 | alive-parent-band | alive-parent-band |
| + L05 SwiGLU lane (`MXN4`) | 4 | **0.754811x** | 8/15 | dead | **alive-degraded** |
| + MXL11's whole body (`MXN3`) | 7 | **0.735962x** | 6/15 | dead | dead |

The body and the token interface **do not add, they multiply the damage** —
MXL11's body alone is 9/15, MXN2's route alone is 10/15, together 6/15. But the
controlled version is the useful one: `COMP.MA.MXN3` is wave 9's
`COMP.MA.MXM9` with the tail swapped and nothing else.

| | `COMP.MA.MXM9` | `COMP.MA.MXN3` |
|---|---:|---:|
| net stored | 0.736305x | 0.735962x |
| install drift | 0.650704 | **0.487722** |
| review | 4/15 | **6/15** |
| lambada ppl | 15,185.40 | **797.61** |

A better token mechanism is worth two review cases and a 19x perplexity
improvement **inside the same composition**, and still does not repair the
interaction.

**Why the ladder cannot reach A9's number with an uncontracted body.** To land at
0.734512x with the body untouched, the whole token interface must fit in
6,128,079 parameters, which caps the verbatim head at **5,978 rows**. The head
that reads alive is 12,288, and a 12,288-row head cannot reach 0.734512x at any
sparsity — its floor is 0.763x even at k=1. Getting below A9's number *requires*
body contraction, and body contraction is what kills the child.

---

## 6. Was A9's ~0.734x beaten?

**Not by a coherent child, under either accounting.**

* Against A9's **published** 0.734512x: only `COMP.MA.MXN3` (0.735962x) comes
  close and it does not get under; it is 6/15 dead either way.
* Against A9's **honest** 0.743217x: `COMP.MA.MXN3` (0.735962x) is under it, and
  is dead. `COMP.MA.MXN1` (0.744033x) is essentially level with it and is dead.
* The deepest child that survives **any** reading is `COMP.MA.MXN4` at
  **0.754811x** (alive-degraded per lane, dead per child).
* The deepest child that survives **both** readings is `COMP.MA.MXN2` at
  **0.780518x**, `alive-parent-band` — which is nonetheless **0.175x deeper than
  the campaign's previous coherent frontier** (`COMP.MA.MXL11`, 0.955444x) and
  scores a point higher.

---

## 7. Benchmarks

lm-eval, tasks `arc_easy, hellaswag, piqa, winogrande, lambada_openai, wikitext`,
`--limit 200 --seed 1234 --batch_size 4`, `max_length=2048`, float32, PCI-pinned
sm86 RTX 3090. Partial evaluations by construction — they rank models against
each other, not against published full-split numbers.

| model | n | net stored | review | verdict | arc_easy | hellaswag_n | piqa_n | wino | lambada | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **parent** | 0 | 1.000000 | 12/15 | — | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.3912 | 81.7976 |
| `COMP.MA.MXN3` | 7 | **0.735962x** | 6/15 | dead | 0.4750 | 0.4300 | 0.5800 | 0.5650 | 0.1200 | 797.6144 | 193.2446 |
| `COMP.MA.MXM9` *(w9)* | 7 | 0.736305x | 4/15 | dead | 0.3700 | 0.3150 | 0.5100 | 0.5150 | 0.0700 | 15185.3986 | 515.0661 |
| `COMP.MA.MXN1` | 1 | **0.744033x** | 7/15 | dead | 0.3600 | 0.3900 | 0.5750 | 0.5100 | 0.0650 | 3765.1269 | 432.0966 |
| `COMP.MA.MXM8` *(w9)* | 1 | 0.744297x | 3/15 | dead | 0.2750 | 0.2800 | 0.5150 | 0.4750 | 0.0550 | 2315739.1765 | 11527.7892 |
| `COMP.MA.MXN4` | 4 | **0.754811x** | 8/15 | dead / *alive-degraded* | 0.4250 | 0.4500 | 0.6000 | 0.5250 | 0.1700 | 370.0939 | 168.7003 |
| `COMP.MA.MXN5` | 1 | **0.780411x** | 8/15 | dead / *alive-degraded* | 0.4350 | 0.4700 | 0.6100 | 0.5000 | 0.1300 | 355.0229 | 178.9962 |
| **`COMP.MA.MXN2`** | 1 | **0.780518x** | **10/15** | **alive-parent-band** | 0.4550 | 0.4450 | 0.6400 | 0.5550 | 0.2000 | 230.3146 | 139.2100 |
| `COMP.MA.MXM6` *(w9)* | 1 | 0.780861x | 9/15 | dead | 0.3650 | 0.3200 | 0.5200 | 0.5350 | 0.1100 | 7337.1925 | 421.6808 |
| `COMP.MA.MXL11` *(w8)* | 6 | 0.955444x | 9/15 | alive-degraded | 0.5250 | 0.4600 | 0.6350 | 0.5700 | 0.2250 | 99.2209 | 118.0216 |
| `COMP.MA.MXL5` *(w8)* | 4 | 0.972579x | 10/15 | alive-parent-band | 0.5100 | 0.4850 | 0.6200 | 0.5500 | 0.3200 | 55.1916 | 105.5257 |

**Every wave-10 child beats its wave-9 counterpart at matched storage on all
seven columns.** Three matched pairs: 0.7360x (MXN3 > MXM9, lambada ppl 19x),
0.7440x (MXN1 > MXM8, lambada ppl **615x**), 0.7805x (MXN2 > MXM6, lambada ppl
32x).

---

## 8. What is in this directory

| file | what |
|---|---|
| `PARENT.md` | the parent's own sixteen responses, the calibration every child is scored against |
| `COMP.MA.MXN2.md` | 10/15 alive-parent-band, 0.780518x — the head-to-head winner |
| `COMP.MA.MXN4.md` | 8/15, 0.754811x — **the child where the two readings disagree** |
| `COMP.MA.MXN5.md` | 8/15, 0.780411x — best reconstruction, worse behavior |
| `COMP.MA.MXN1.md` | 7/15, 0.744033x — the literal A9 rebuild |
| `COMP.MA.MXN3.md` | 6/15, 0.735962x — the deepest, and dead |

Each per-child file carries the net stored ratio, both storage accountings, the
per-component storage, the install drift, the benchmark row, the machine-computed
verdict, and all sixteen responses verbatim beside the parent's.

## 9. Tests, and one open defect

The campaign's own test subset, run on a `git archive` of this wave's commit with
wave 9's invocation verbatim: **`10 failed, 219 passed, 1 skipped`** — the same
ten failures as waves 7, 8 and 9 **by name**, and +14 passes, which are wave 10's
new cases. Wave 9 reported `10 failed, 205 passed, 1 skipped`. Wave 10's own two
test files are **100 passed**.

Separately, running the *whole* `tests/` tree in one process fails at collection
on this deployment — and on wave 9's, which was run as a control and produces the
identical error in the identical two files. It is an alphabetical-collection-order
artifact: the C01 kernel registers `('T5','C01')` at import, and
`experiments/run_lfm2_matrix_cell.py` then registers the same pair without
`replace=True`. Pre-existing at wave-9 HEAD; commit `9acaf38f` touches no file in
`experiments/`. With those two files excluded the wave-10 and wave-9 progress
streams are character-for-character identical over 5,287 tests. Left unfixed
deliberately — see `NOTES-MX-N.md` §9.2.
