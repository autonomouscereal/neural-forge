# STANDING STATE — Neural Forge matrix campaign, 2026-08-29 (end of wave 45)

One page. Where everything stands, what each number means, and what is blocked
on whom. Every claim here is checkable from a committed artifact named beside it.

---

## 1. THE TWO FLOORS, AND WHAT EACH ONE MEASURES

The campaign publishes **two** floors because Cyril's ruling on the copy-back
clause is still pending. They are not two estimates of one quantity. They measure
**different things**, and after wave 44 that is a measured claim rather than a
caveat.

### Column A — **0.3859919152x**, `ARCH.V3CL768` (K=3,072, 88,659,712 params, wave 44)

**Column A measures ECHO PERSISTENCE. It stopped measuring intelligence
somewhere above this rung, and the machine says so without being asked.**

- The child holds **1/15**. Its held lane is `summarization-priorities`, and that
  lane is **the whole prompt returned, instruction line included**, with one
  visible corruption in 484 characters.
- **Eleven of its sixteen lanes return the prompt itself.** Four are byte-identical
  to the prompt with no edit at all.
- The pre-registered union copy measure, applied **prospectively** and with no
  human in the loop, called that held lane `COPY_BACK` at union coverage
  **1.000000**.
- Wave 45 bracketed the mechanism on one body:

  | rung | K | ratio | sealed | the summarisation lane |
  |---|---:|---:|---:|---|
  | `ARCH.V3CL768` | 3,072 | 0.3859919152x | **1/15** | echo intact, 1 corruption in 484 chars, **holds** |
  | `ARCH.V2CL768` | 2,048 | 0.3814267993x | 0/15 | echo intact and **complete at 168 tokens**; decision preserved verbatim, **risk destroyed** |
  | `ARCH.V1CL768` | 1,024 | 0.3768616835x | 0/15 | **disintegrated below the word** — `su ppulpulpul` to the cap |

  The echo does not shorten, it **degrades, selectively**: the high-frequency
  sentence frame survives while low-frequency content words die. `battery` →
  `bitties`, `launch` → `lacuan`, and *"The remaining risk is that its factory
  expansion may slip by one week"* → *"The reminance risk is that its facrept by
  one week"*.

**Column A has not moved for the first time in six waves.** Waves 40–44 moved it
five consecutive times; wave 45 registered both remaining rungs at zero and both
read zero.

### Column B — **0.4116484014x**, `ARCH.V10CL896X09` (K=10,240, 94,552,832 params, wave 38)

**Column B measures a MACHINE-VERIFIED intelligence floor: the deepest child
holding a lane that is not a copy of its own input.**

- Its held lane is `uncertainty-insufficient-data` — a `SELECT` ask, which
  wave 40's precedents say a copy cannot satisfy.
- **It is reproduced by machine with zero human input.** Wave 44 pre-registered
  `COPY_BACK_UNION_COVERAGE = 0.60` at its tree cut, from a two-wave-old
  artifact's empty 0.343-wide separating interval, and the machine-only pass
  returned 0.4116484014x, `ARCH.V10CL896X09` — the number a human reading had
  been the only way to produce.
- **Unmoved by wave 45's anchor change and by wave 45's two children.**

### The gap

**0.0256564862 of stored ratio.** Everything inside it is a child that holds a
lane by copying its input.

---

## 2. THE 15-CASE INSTRUMENT IS NOT MEASURING LANGUAGE MODELLING AT THIS DEPTH

Wave 45 closed this with six rungs on one body. Wikitext word perplexity is
**strictly monotone in K**; the sealed count is not monotone at all.

| K | cell | wiki wppl | byte ppl | sealed |
|---:|---|---:|---:|---:|
| 6,144 | `ARCH.V6CL768` | 5,035 | 4.92 | 1/15 |
| 5,120 | `ARCH.V5CL768` | 6,657 | 5.19 | 1/15 |
| 4,096 | `ARCH.V4CL768` | 8,749 | 5.46 | **0/15** |
| 3,072 | `ARCH.V3CL768` | 14,019 | 5.96 | **1/15** |
| 2,048 | `ARCH.V2CL768` | 31,968 | 6.96 | 0/15 |
| 1,024 | `ARCH.V1CL768` | 961,032 | 13.15 | 0/15 |

Parent: 81.80 wppl / 2.28 byte ppl.

**The 30-prompt extension surface, run in wave 45, disagrees with the sealed
fifteen at the top of the frontier** (`wave45/rider-concordance-wave45.json`):

| cell | ratio | sealed 15 | ext held /30 | ext charged /30 |
|---|---:|---:|---:|---:|
| `ARCH.V12CL896G03` | 0.4510448164x | **4/15** | 0 | **14** |
| `ARCH.V10CL896X09` | 0.4116484014x | 1/15 | 0 | **2** |
| `ARCH.V6CL768` | 0.3996872628x | 1/15 | 0 | 19 |
| `ARCH.V5CL768` | 0.3951221469x | 1/15 | **1** | 19 |
| `ARCH.V3CL768` | 0.3859919152x | 1/15 | 0 | 16 |
| *parent* | 1.0x | 12/15 | 5 | 2 |

The sealed column's best child of the five is charged on **14 of 30** extension
prompts; the column-B floor holder, which is deeper and reads 1/15, is charged on
**2**. A **sevenfold** difference in degeneracy running the opposite way to the
sealed column.

**The surface is legitimate to compare across waves and that is measured, not
assumed.** Wave 34 ran the same bank on the parent and on `ARCH.V12CL896G03`;
wave 45 re-ran both eleven waves and three deployments later and reproduced
**every scored field and all 60 per-response digests exactly**.

---

## 3. THE TWO PENDING RULINGS, AND EXACTLY WHAT EACH ONE UNBLOCKS

### Ruling 1 — the PROBE storage claim (`NOTES-MX-AS` §12)

**Blocks three classes outright: K4, K3(c), and ALIAS.** No work on any of them
is authored, scheduled or possible until it lands. Nothing else in the campaign
touches them, so the cost of the delay is exactly three unexplored classes and no
drift anywhere else.

### Ruling 2 — the copy-back clause

**Does not block any build. It decides which of §1's two numbers the campaign
calls "the floor".** Both readings are complete over 57 children, per child and
per lane, with the response digest that makes each call checkable
(`wave45/prereads/deep-band-dual-assessment-handmerged-wave45.json`).

- **Ruling YES** (a copy-back counts): the floor is 0.3859919152x and the
  campaign's deepest living child is a model that returns its input on eleven of
  sixteen lanes.
- **Ruling NO** (a copy-back is the absence of a decode): the floor is
  0.4116484014x and column A becomes a published diagnostic of echo persistence.

Applying it costs nothing either way; **the dual column is maintained in every
document until it lands.**

---

## 4. GATED AND BLOCKED CLASSES

| class | state | what it waits on |
|---|---|---|
| K4, K3(c), ALIAS | **BLOCKED** | Cyril, PROBE storage-claim ruling |
| ROWSET Arms A and B | **BLOCKED** | a corpus that has not appeared in five waves |
| NORM-ESTATE Arms M and S | **NOT AUTHORED** | a solver nobody has written |
| PLANE (frequency band) | **RE-SEALED** by the child that opened it, wave 43, under the slate's own kill gate |
| XCHANGE | **CLOSED** by its own pre-registered gate, wave 43. A result, not a deferral |
| CL768 vocabulary ladder | **CLOSED AT THE BOTTOM**, wave 45. K=768 is the last rung the 765-row mandatory set permits; K=1,024 is built and dead |
| block-13 cliff (CL896X13) | five consecutive dead rungs, K=8,192 down to K=5,120; wave 44's matched-K pair proved the death is the **body's** |

---

## 5. THE LAW LIST

The complete standing rule set is not restated here. It is carried in the
`NOTES-MX-*` chain, one wave per file, in
`reports/matrix-campaign-20260810/`'s parent directory:

- **Pointer: `NOTES-MX-AW.md` §12** — "what the next wave should know", which is
  the live head of the list.
- Each wave's §8 "Method and controls" and §10 "What this wave did NOT build"
  carry the rules that wave added or exercised.
- The rules that are *code* rather than prose live in `run/mx45/cut_wave45.py`'s
  `audit()`, which is the only place a rule is enforced rather than described.
  As of wave 45 it pins **four external surfaces** the cut tree does not contain:
  the builder's `BASIS_SCHEMAS_ACCEPTED`, the sealer's `BUILD_RECEIPT_SCHEMAS`,
  the behavioural sealer's argparse defaults, and — new in wave 45 —
  `PARENT_REVIEW_BAND = 12`, `A9_REVIEW_BAND = 8` and
  `PARENT_BAND_TOLERANCE = 2` in `src/neural_forge/matrix_cell_receipt.py`, the
  three integers **every behavioural verdict in the ledger is computed from**,
  which nothing had ever asserted in forty-four waves.

---

## 6. ESTATE INTEGRITY

| quantity | value |
|---|---|
| branch | `codex/lrc-blank-compression`, **never pushed, never rebased** |
| ledger rows | **1,056** |
| ledger sha256 | `d2027831e229d763a251ca602655c729a4432645bed43a9ba2e4a8824ef211ea` |
| seal verification | **1,052 resolve, 1,052 verify, 0 fail**, 4 do not resolve (3 × `CAL.C01.T5.X4`, 1 × `COMP.MC.MXO0` — pre-existing since before wave 40) |
| declared seal roots | **41**, audited against disk, **0 undeclared** |
| reviewed whole-model children | 358, of which 286 score ≥ 1/15 |
| benchmark rows | 268, plus **32 parent rows all matching the reference** |
| deployments this wave | 3, each verified file-by-file against its commit's blob digests: **0 mismatches, 0 missing, 0 undeclared, 0 hardlinks** |
| Codex lane | **untouched**, 324 uncommitted paths, every commit staged by explicit path |
| `/mnt/ai_data` | 4.5T, 1.7T available |

### The retention promise, cashed for the first time

Sixteen waves of retention manifests justify every DROP with *"one CPU pass off a
retained source"*. **No wave had ever tested one.** Wave 45 needed the column-B
floor holder for the ordered extension surface and its checkpoint had been
dropped by wave 38, so it rebuilt it — and the rebuild reproduced wave 38's
committed digests **exactly**:

```
child_tensors_sha256   32ed4000ba7107809ff6a2860de23ede3793bfdbb7b05957cb16d364a52f4c26
child_directory_sha256 24b8a5b106924e9fe1b71282060784a7b9d46027833c25a22898558c4673fddd
child_parameters       94,552,832       net_stored_ratio 0.4116484013735471
```

The sealed wave-38 cell receipt bound the rebuilt bytes with no modification, and
the extension surface then decoded through it. **The promise is now a
measurement.**

---

## 7. WHAT IS NOT KNOWN, STATED AS SUCH

1. **Whether column A's echo is worth anything at all** is a ruling, not a
   measurement, and it is Cyril's.
2. **The residual defect in the copy measure is a DENOMINATOR, not an anchor.**
   Wave 45 paid the ordered `writing-concise-email` anchor debt at its tree cut
   and registered, before the pass, that it would not close the seven rows it was
   ordered to close. It did not. All eight void-changing machine–hand
   disagreements survive. The response-side column — same union, same threshold,
   response as denominator — calls both distinct responses copies at 0.683824 and
   0.635714 with no anchor at all. It is **published and decides nothing**, which
   is the first of the two waves of evidence the campaign's own promotion
   protocol asks for.
3. **The eighth void-changing disagreement is on `uncertainty-insufficient-data`,
   which is the same lane the column-B floor holder is held on.** An anchor there
   could move column B. It is priced and deliberately not added.
4. **No wave has measured what any of this transfers to.** The token interface is
   29.2% of this parent and about 1% of a 70B model by the same arithmetic
   (`docs/PHASE1` §4.1). These are this model's phase-1 numbers.
