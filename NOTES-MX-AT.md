# NOTES-MX-AT — wave 42: the floor goes below 0.40x on the two-axis body, the block-13 ladder is NOT flat, and both readings of the copy-back clause are now complete over the whole deep band

**2026-08-29.** Branch `codex/lrc-blank-compression`. Three children built,
sealed and read; ledger 1034 → **1040** rows,
`803439d1d300622dea437d64aaea07de88c16af04e3a808c5e6f2155ce5c58c4`, byte-prefix
verified against the pre-wave backup on the share. Two deployments, and the
accept list cost neither — seventh consecutive wave.

**Two of three pre-registrations hit. The miss is the whole finding of §4.**

**The sealed 1/15 rung moves BELOW 0.40x for the first time** — and it moves off
the body the last two movements were on. **The block-13 ladder, flat at one over
six rungs, ends at the seventh.** §3, §4.

**And the dual-column deep-band assessment is done: 52 children, both readings,
zero anomalies, and the two floors are 0.0119611386 of stored ratio apart.** §5.

---

## 1. One thing was checked before the build list was fixed, and it changed it

Wave 42 was ordered to build *"the two cheapest open cells: V6CL768 and K=9,216
on CL896X13 — the flatness-vs-block-equivalence adjudicator, wave 39's cheapest
open cell, still unbuilt."*

**K=9,216 on CL896X13 is built.** It is `ARCH.V9CL896X13`, campaign wave 40,
93,504,256 parameters, 0.4070832855x, sealed **1/15**, and `NOTES-MX-AR` §3.1
records it as exactly the adjudicator it was ordered as: *"Two accounts disagreed
by exactly one case and flatness won. Block-equivalence predicted zero; wave 39's
flat-at-one predicted one. Registered at one, reads one."* It was registered in
its own file on the share at 02:34:51.775Z and its build receipt is stamped
02:35:19.827Z.

Rebuilding it would spend a frozen review re-reading a sealed result — the thing
wave 41's rowset divergence pre-read exists to prevent. **Registered as a
substitution with its cause before anything was built**, in the wave-41
procedure: the cell is replaced by the two block-13 rungs **below 0.40x** that
are genuinely unbuilt, `V7CL896X13` and `V6CL896X13`. `V6CL768` is built first,
as ordered.

---

## 2. The tree, and a protection that turned out not to cover what it looked like it covered

`run/mx42/sealq.sh` arrived VERBATIM carrying wave 41's marker **and wave 41's
TEN seal calls with every count intact**. Wave 40's pin — `audit()` counts `^seal
<TAG>` and requires **zero at cut time** — refused, for the second consecutive
cut. All ten were deleted by hand.

### 2.1 THE PIN COUNTS CALLS. IT DOES NOT COUNT CLAIMS, AND THE CLAIMS HAD BEEN STALE FOR TWO WAVES

`sealq.sh` is the file a reader opens to find out **how** a seal was graded. The
block below its marker arrived carrying **wave 39's calibration result, word for
word**:

> "The parent's own responses were re-read blind from `ARCH.V14CL896X13`'s
> dossier — the first dossier of the wave"

`ARCH.V14CL896X13` is a **wave 39** child. That sentence — and the
consecutive-calibration counter beside it, and the three-bullet byte-comparison
list under it — survived **wave 40's cut and wave 41's**. Both of those waves ran
their own calibration and each recorded a *different* first dossier in its own
NOTES; wave 41's §10 names `ARCH.V8CL896X13`.

**Nothing was mis-sealed.** The counts in the calls were each wave's own and each
wave's NOTES carry the calibration it actually ran. What was wrong is that the
file that states the standard described a measurement two waves old, and the
protection that was supposed to stop exactly this kind of carry did not look at
it.

**The fix is the same shape as wave 40's and it is in `cut_wave42.py --audit`:**
an inherited calibration RESULT, the named dossier and the stale counter are
**pinned at zero at cut time**, on the same argument as the seal call — *a result
cannot be true before the wave has taken the measurement.* The wave writes its
own after it reads.

**And those three pins are LITERAL, which is not enough, and the file says so.**
They name wave 39's exact sentence, exact dossier and exact counter, so they
catch the carry that happened and would **not** catch the same defect worded
differently — wave 42's own calibration paragraph opens with different words and
none of the three fires on it. So the audit also carries the **general** pin,
which is the seal-call pin's shape applied to prose: the marker block ends with a
**sentinel line** saying the wave's own calibration is written below it after the
dossiers are read, and at cut time the only thing that may follow that line is
blank lines and the closing `echo`. **Verified in both directions**: on
`run/mx42/sealq.sh` as shipped — which is exactly what a wave-43 cut would
inherit — the sentinel pin reports **78 residue lines and refuses**; on the same
file reconstructed at cut time it reports zero. The three literal needles are
kept beside it as the named regression.

### 2.2 One addition, no edit to a carried module

`mx42_dualband.py` and `dualbandq.sh` are new and they **build nothing**. Every
other carried module keeps the bytes wave 41 shipped, modulo the five
substitutions and the four hand edits. Wave 42's build list is **three
`mx42_vocab.py` gathers**: no body, no capture, no solve, no GPU-side
construction at all.

---

## 3. F1: the floor moves below 0.40x, and it moves OFF block 13

| rung | before | after | Δ |
|---:|---|---|---:|
| **sealed 1/15** | `ARCH.V8CL896X13` 0.4025181696 | **`ARCH.V6CL768` 0.3996872628** | **0.0028309068** |
| canonical 1/15, copy-back **refused** | 0.4116484014 | *does not move* | **0** |
| sealed 2/15 | `ARCH.V10CL896X09G10` 0.4116484014 | *unchanged* | — |

`ARCH.V6CL768`, K=6,144 on wave 40's retained `MX40-CL768`, 91,805,440
parameters. Registered at **one** on flatness against a census-floor account that
predicted zero; reads **one**. **The campaign's first child below 0.40x, and it
holds a case.**

**Third consecutive wave the floor moves, and the first of the three that does
not move on the block-13 body.** The flat ladder on CL768 is now **seven rungs**
— the body at 0.6644639834x and V12/V10/V9/V8/V7/V6 — across **0.2648 of stored
ratio**, the deepest flat ladder in the campaign.

**The pre-registration's sharper claim about this child is WRONG and the
measurement is why it is known to be wrong.** It predicted the held summarisation
lane would be byte-identical to `ARCH.V7CL768`'s `170feb2dcc17772b`, on the basis
that a 1,024-row drop on this body had left the held lane untouched before. It is
**`f4d828fb1eb03d8b` and it is shared with nothing in the estate.** The two
children share **ten of sixteen** lanes and **not** the one that holds. So
`ARCH.V6CL768` is the wave's **only new held-lane witness**, which is more than
the rung is worth.

**And the held lane is still a copy.** The source paragraph returned in full with
three substitution artifacts — *bitties supspliers*, *11 percent more* collapsed
to *11%*, *sampes*. It performs no summarisation. §5.

**A new depth record for the fourth artifact species.** `ARCH.V6CL768` returns
the prompt or its payload on **eight of sixteen lanes** — equalling wave 41's
maximum, 0.0091 of ratio deeper — including a retrieval lane that hands back the
NOTE with the `NOTE:` prefix still attached and selects nothing, which is wave
40's `ARCH.V9CL768` precedent reproduced at a smaller size. Four lanes ride the
cap.

---

## 4. F2: the X13 ladder is NOT flat, and the cliff is located to 1,024 rows

| K | ratio | sealed | |
|---:|---:|---:|---|
| 65,536 (`ARCH.CL896X13`) | 0.6581646585 | 1 | wave 35 |
| 14,336 | 0.4299088649 | 1 | wave 39 |
| 12,288 | 0.4207786331 | 1 | wave 35 |
| 10,240 | 0.4116484014 | 1 | wave 39 |
| 9,216 | 0.4070832855 | 1 | wave 40 |
| 8,192 | 0.4025181696 | 1 | wave 41 |
| **7,168** | **0.3979530538** | **0** | **wave 42** |
| **6,144** | **0.3933879379** | **0** | **wave 42** |

**`ARCH.V7CL896X13` was registered at ONE on flatness and reads ZERO.** Six rungs
read one and the seventh does not. **Flatness loses this argument for the first
time after winning it twice** — at wave 40's K=9,216 against block-equivalence,
and at wave 41's K=8,192.

**THE BLOCK-13 CLIFF IS LOCATED TO WITHIN 1,024 VOCABULARY ROWS**, between
K=8,192 and K=7,168. `NOTES-MX-AS` §14.4's instruction to keep excising block 13
is now **bounded**: block 13 is the right block *above* 0.4025x and there is
nothing below it on that body.

**Its summarisation lane cannot even copy.** *"The team compared three bitters.
The suplification is cheats but not yet yet before launch. The suplification will
request weekly schedule evidence."* The decision is gone, the risk is gone, and
what survives is the paragraph's **last clause** attached to a corrupted subject.

**`ARCH.V6CL896X13` is the deepest child the campaign has ever built** —
90,358,528 parameters, 0.3933879379x — and it is **the one cell this wave
registered AGAINST flatness**, because at 0.3934x it sits inside the band where
`ARCH.V12CL768X09` (0.3985x), `ARCH.V12CL896X09X13` (0.3905x) and
`ARCH.V10CL768X09` (0.3894x) all read zero. Registered at zero, reads zero. Its
summarisation lane echoes the **instruction** — the words *unresolved risk*
survive out of the ask and not out of the payload — and both facts are gone.

**The strongest single piece of evidence that the cliff is real:** the two dead
rungs share **eight of sixteen lanes byte for byte**, including safety,
retrieval, uncertainty, greeting, instruction, logic and comparison. Below the
cliff the two rungs are largely **one decode**, which is what a dead family looks
like. `ARCH.V7CL896X13` shares only **two of sixteen** with the living rung above
it, and **not** the summarisation lane, so its zero is an independent decode and
not a re-read of the rung it kills.

---

## 5. F3: THE DUAL-COLUMN DEEP BAND — 52 CHILDREN, BOTH READINGS, ZERO ANOMALIES

`wave42/deep-band-dual-assessment.json`, `DEEP_RECORD` §0. Every graded `ARCH.*`
child at or below **0.47x** read twice.

* **Column A — copy-back counts** (wave 40's practice). **Not computed.** It is
  the sealed `per-case.json` verdict, and `mx42_dualband.py` **refuses to write
  its artifact** if its own count of that file disagrees with the ledger row's
  sealed `behavior_useful_or_partial`. A second column is worth nothing if the
  first is this module's recount rather than the estate's record.
* **Column B — copy-back refused** (the clause as written). Column A **minus**
  the held lanes that are copy-backs. A lane sealed as a miss is never re-read
  and the module cannot turn a miss into a hold in either direction.

| reading | floor ratio | child | what holds |
|---|---:|---|---|
| **A** | **0.3996872628** | `ARCH.V6CL768` | one lane: the source paragraph, returned |
| **B** | **0.4116484014** | `ARCH.V10CL896X09` | one lane: `uncertainty-insufficient-data`, which NAMES the four missing variables |

**The gap is 0.0119611386 of stored ratio — 2,747,392 parameters — and the whole
of it is one lane read two ways.** Nothing else among the 52 children differs
between the columns.

**Under B, every held case between 0.3934x and 0.4094x is voided.** Seventeen
children, every one of them holding on `summarization-priorities`, and **all
seventeen distinct responses in the band are the six-sentence source paragraph
returned in full** — read by hand, six of them with the instruction line still
attached. Below 0.4116484014x the campaign has **no living child at all** on that
reading, and that rung has not moved since wave 38.

### 5.1 The ask table is the load-bearing part, and it fired zero anomalies

Wave 40 sealed a copied SUMMARISATION lane **held** and a copied RETRIEVAL lane a
**miss**, and those are consistent: the summarisation ask is to PRESERVE two
facts, which a copy does; the retrieval ask is to SELECT three, which a copy does
not. **So the clause can only reach a PRESERVE lane.** `mx42_dualband.py`
publishes an **anomaly** whenever column B voids a lane whose ask is not
PRESERVE. **It fired ZERO times across 52 children and 69 held lanes.**

The `context-retrieval` and `uncertainty-insufficient-data` lanes that survive
column B are real selection — degraded, several of them cap-riding — but they
name the 4:15 p.m. checkout, the 7:30 opening, and the missing time range, units,
sample size and price change. That is the distinction the deliverable was asked
for: **selecting the right fact is not returning the whole payload**, and at
0.4116484014x the estate still has one child doing the former.

### 5.2 THE INSTRUMENT UNDER-REPORTED AND THE HAND PASS IS WHAT FOUND IT

The pre-registered machine measure thresholds the **longest contiguous** common
run against the lane's payload. **Every copy-back in this estate carries the
word-substitution signature, and each mutation breaks the run.** It scored
`ARCH.RP9CL896X13`'s held lane at **0.153** and `ARCH.V7CL768`'s at **0.427** and
called both CONTENT. Reading the dossier text found all seventeen to be copies.

`payload_union_coverage` — the union of all common runs of ≥8 characters — is
the repair, and on the same lanes it reads **0.827 to 0.997** for the copies
against **0.198 to 0.331** for a genuine two-sentence summary of the same
paragraph. It would classify all seventeen correctly at any threshold in
[0.60, 0.82].

**IT DECIDES NOTHING IN THIS WAVE.** It was added *after* the results were seen,
so letting it move a classification would be tuning an instrument to an answer.
Every `machine_classification` in the artifact is still the pre-registered
measure's; the **hand reading** (`wave42/hand-reading-wave42.json`, 69 calls) is
what voids a lane; and the **22 machine/hand disagreements** are published by
digest rather than reconciled. **The next wave should pre-register the union
measure as primary.**

### 5.3 The one place the estate's own record looks wrong

Two distinct `writing-concise-email` responses — `39fdfd52` and `60227409`, held
across seven children between 0.4373x and 0.4693x — restate the **instruction**
in the first person (*"I'm a polite email that's asking a vendor to confirm
whether a delayed shipment will arrange by Friday"*) and draft no email, mention
no Monday installation and request no tracking number. The hand reading calls
them copy-backs on a **PRODUCE** lane; the machine, at 0.332 longest-run
coverage, calls them content. They are voided under column B by the hand reading
and they are in the disagreement list. **This is the only class in the band where
the estate appears to have sealed a hold on a lane that produces nothing**, and
it is reported rather than re-graded.

---

## 6. Benchmarks, and a fourth inversion

Standard invocation, parent FIRST. **The parent row reproduces waves 32 and 34
through 41 to every reported digit — 29 parent rows, all 29 matching.**

| model | ratio | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | sealed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 1.000000 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | **81.80** | 12/15 |
| `ARCH.V6CL768` | **0.399687** | **0.2800** | 0.3150 | 0.5200 | 0.4850 | 0.0900 | 22377.86 | **5035.32** | **1/15** |
| `ARCH.V7CL896X13` | 0.397953 | 0.3150 | 0.3000 | 0.5300 | 0.4850 | 0.0850 | 125529.81 | 24046.23 | **0/15** |
| `ARCH.V6CL896X13` | 0.393388 | **0.3250** | 0.3050 | 0.5150 | 0.5000 | 0.0900 | **143704.36** | **26448.74** | **0/15** |

**`ARCH.V6CL896X13` has the best `arc_e` in the wave, 0.3250, and holds
nothing. `ARCH.V6CL768` has the WORST, 0.2800, and is the only child that holds
anything.** Fourth inversion the column has produced, and the sharpest on this
axis: the two instruments rank these three children in exactly opposite orders.

**The perplexity column, unusually, agrees with the review here** —
`ARCH.V6CL768`'s wikitext word perplexity is 5,035 against 24,046 and 26,449 for
the two dead children, a factor of five. That is worth naming because it has
*not* agreed since wave 32; it is one wave of agreement and not a rehabilitation
of the column. Nothing in this table is a usable model.

---

## 7. Method and controls

- **Parent calibration, fifteenth consecutive, exact on the count that matters.**
  Re-read blind from `ARCH.V6CL768`'s dossier — the first of the wave, sealed
  06:12:48Z — before any child lane was opened. **Useful-or-partial 12/15**,
  **cap-hits 0/16**, both documented anchors verbatim: the arithmetic Partial
  boxing **3 kits per box with 94 left over**, and the multi-turn miss offering
  *Vegan Peanut Butter & Jelly Bars* to a table with two severe peanut allergies.
- **AND IT DISAGREES WITH THE INHERITED COMPOSITION ON ONE LANE AND SAYS SO.**
  The reference is recorded as 5 Useful / 7 Partial / 3 miss. `NOTES-MX-AS` §10
  lists `summarization-priorities` among the five Useful **and then calls it the
  third miss**, "which inverts the source — Supplier B selected due to its lower
  cost when B costs 11 per cent more". Both cannot be true. Wave 42 reads that
  lane **Partial**: held, because it carries both asked-for items; not Useful,
  because it asserts a reason the source contradicts. The reason is not one of
  the two things the lane asks to preserve, which is why the inversion charges
  the *grade* and not the *case* — wave 24's inversion MISS was on the email
  lane, where the inverted item WAS the ask. **This wave's split is 4 Useful / 8
  Partial / 3 miss = the same 12/15**, and the difference is one lane moving
  between two HELD grades.
- **The runner control is NINE DOSSIERS WIDE ACROSS FOUR WAVES.**
  `mx42_parent_identity.py` reads `865f006e3813712ec202cb23`, protocol
  `e5e377688597`, prompt pack `0ee6bf14cef5` on all three wave-42 dossiers, and
  `wave42/lane-byte-comparison-wave42.json` finds the parent lane digests
  **identical across nine dossiers spanning waves 39, 40, 41 and 42**.
- **The lane byte comparison ran before any count was written** and it decided
  three claims by construction: `V6CL768`'s held lane shared with **nothing**
  (against the pre-registration); `V7CL896X13`/`V8CL896X13` **2 of 16, held lane
  not shared**; `V7CL896X13`/`V6CL896X13` **8 of 16**.
- **The reading order for children under 0.60x was followed**: the summarisation
  lane of all three children was read first and written down before any other
  lane was opened. It decides all three seals.
- **Every child asserted its parameter count before writing** and all three
  landed on the pre-registered value exactly: 91,805,440 / 91,407,104 /
  90,358,528. Identity gate `torch.equal` passed on all three.
- **`BELOW_FLOOR=1` on all three rungs**, and the guard it disarms — the coverage
  half of wave 25's census gate — is asserted present by `cut_wave42.py --audit`.
  The merge-closure assertion is not releasable and fired on none of them.
- **Seal verification**: 1040 ledger rows, **1036 resolve to a receipt on disk,
  1036 verify, 0 FAIL**, 4 unresolved — the same historical four. **38 seal
  roots** audited against disk, **0 undeclared**.
- **`sealq.sh` ran ONCE** and all three sealed on the first call.
- **A dry run preceded the ledger merge** and left the ledger at 1034 rows and
  `5b91a211…` unchanged — verified, not assumed. The apply was a **one-pass
  merge**, all six shards and all three verdict receipts in a single invocation,
  correct argument form on the first invocation.
- **Pre-registration hit rate: 2 of 3.** Hits: `V6CL768` 1, `V6CL896X13` 0. Miss:
  `V7CL896X13` 1→**0**, and that miss is §4. The sharper per-child prediction —
  `V6CL768`'s held-lane digest — is also **wrong**, and it is §3.

---

## 8. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `dde9ac7c` | cut wave 42's tree, author `mx42_dualband.py`, append the sealer's six accept strings **in the same commit**, and pre-register after finding one ordered cell already sealed |
| `5af0b035` | the three seals, the union-measure addendum, the hand reading |
| *(this)* | the measurement, the tables and the docs |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `dde9ac7c` | 7490 | 0 | 0 | 0 | 0 |
| `5af0b035` | 7491 | 0 | 0 | 0 | 0 |

**Ledger.** Re-verified on the share before it was touched (1034 rows,
`5b91a211…`), backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave42-20260829T062932Z` and
`cmp`-verified byte-identical. **One-pass merge**, 1034 → **1040** rows, 0
superseded. Byte-prefix verified: `head -c <backup size> ledger.jsonl | sha256sum`
returns `5b91a211…` exactly. **Mirror copy-back done, and the mirror write
asserted the merged file is a byte-prefix extension of what it replaced BEFORE
writing.**

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used for the
reviews and the benchmarks, with the in-process SM86 preflight on every driver.
Both carried an idle-holding H3 ComfyUI container (~288 and ~279 MiB) and were
**not restarted**, for wave 34's stated reason. The four P40s were never touched.
The Quadro RTX 8000 read **330 MiB** at preflight and wave 42 issued no command
against index 6. **No GPU was used for a build**: all three children are CPU
gathers.

**Disk**: `/mnt/ai_data` 4.5T, 1.7T available. **The Codex lane was not touched.**

**Retention**: `package.sh` ran BEFORE `--plan`, as the rule requires. KEEP =
`MX42-V6CL768` (the frontier holder and the wave's only new held-lane witness)
and `MX42-V7CL896X13` (the cliff child, whose bracket the next wave will bisect).
DROP = `MX42-V6CL896X13`, rebuilt by one CPU pass off the retained
`MX35-CL896X13`.

---

## 9. What this wave did NOT build, and why

1. **PLANE** (slate #4) — **not authored**. Its Stage-1 q/k plane slice with the
   RMSNorm-denominator ridge and its `W_o` prior-anchored ridge are a NEW SOLVER,
   not a driver change. This was **registered before the wave** rather than
   reported after it, and no plane-energy census was run, so the HD48 retention
   prediction is still unregistered and the axis is neither opened nor re-sealed.
   **This is the honest cost of wave 42 and it is the same cost wave 41 paid.**
2. **XCHANGE** (slate #3) — same status, not started, nothing claimed either way.
3. **ROWSET Arms A and B** — still blocked on corpus; no corpus has appeared.
4. **ALIAS** — still blocked at construction in three constructions, behind the
   PROBE storage-claim ruling with K4 and K3(c).
5. **NORM-ESTATE Arms M and S** — still need a new solver. The D grid was not
   extended, per `NOTES-MX-AS` §14.6.
6. **No child was rebuilt to re-read a sealed result.** §1.

---

## 10. The two rulings, unchanged, and what wave 42 changed about the second one

**Ruling 1 — the PROBE storage claim** (`NOTES-MX-AS` §12), unchanged and still
blocking three classes: K4 (score-bilinear joint q/k), K3(c) (Monarch's first
cell) and ALIAS. *May a `PROBE.*` child carry a storage claim that is stated
arithmetic, labelled non-loader, and excluded from the frontier?* One
adjudication unblocks three classes; a no closes them permanently.

**Ruling 2 — the copy-back clause.** *Does a response that returns the prompt's
own payload — preserving the asked-for facts by copying rather than by answering
— hold a case under the canonical standard?*

**What wave 42 changed is that the ruling now costs nothing to apply.** Both
readings are complete over the whole deep band, per child and per lane, with the
digest that makes each call checkable:

- **NO** (copy-back refused): the intelligence floor is **0.4116484014x**,
  `ARCH.V10CL896X09`, holding one lane — `uncertainty-insufficient-data`, which
  names what is missing. **Seventeen children between 0.3934x and 0.4094x go to
  zero**, and three waves of floor movement — 39, 40, 41 — plus this wave's
  become movements of the ratio at which the model still copies one paragraph.
  The canonical rung has not moved since wave 38.
- **YES** (copy-back counts): the floor is **0.3996872628x**, `ARCH.V6CL768`, and
  the canonical clause's copy-back sentence must be **rewritten**, because as
  written it says the opposite.

**Neither answer requires re-grading anything.** Column A is the sealed record
and column B is a published subtraction from it.

---

## 11. What the next wave should know

1. **THE BLOCK-13 LADDER HAS AN END AND IT IS BETWEEN K=8,192 AND K=7,168.** The
   cheapest open cell in the estate is now **K=7,680 on `MX35-CL896X13`**
   (91,931,392 parameters, 0.4002356117x) — one CPU pass, and it halves a
   1,024-row bracket on a body whose upper endpoint is retained. Note that it
   lands *above* 0.40x, so a survival there does **not** move the sub-0.40 floor;
   its value is locating the cliff, not depth.
2. **THE DEEP FLOOR NOW LIVES ON CL768, NOT ON BLOCK 13.** `ARCH.V5CL768`
   (K=5,120, 90,756,864, **0.3951221469x**) is the next rung of the seven-rung
   flat ladder and the only cheap cell that can move the sub-0.40 floor again.
   Flatness predicts one; the block-13 cliff at a *shallower* ratio predicts the
   CL768 ladder has an end too and that it is close.
3. **PRE-REGISTER THE UNION MEASURE AS PRIMARY.** §5.2. The longest-run measure
   gets 10 of 17 copies right; the union measure gets 17 of 17 with a margin of
   0.5 between the classes. Wave 42 could not use it without tuning to an answer;
   wave 43 can, by registering it first.
4. **A PIN THAT COUNTS CALLS DOES NOT COUNT CLAIMS.** §2.1. Wave 42 added both
   a literal pin (the three wave-39 needles, kept as the named regression) and a
   **general** one (nothing but blank lines and the closing `echo` may follow the
   calibration sentinel at cut time). **The general pin is the one that matters
   and the wave-43 cut will be the first to exercise it** — it already reports 78
   residue lines against `run/mx42/sealq.sh` as shipped. **The same class of
   defect has not been audited anywhere else in the tree**, and the natural next
   place to look is every docstring that states a *number* rather than a *rule*.
5. **`ARCH.V6CL768` IS THE ONLY NEW HELD-LANE WITNESS SINCE WAVE 41's
   `ARCH.V7CL768`**, and both are on the CL768 body. Every held lane on block 13
   is one decode read at four sizes.
6. **The 15-case instrument is past exhausted at this depth.** Three children,
   ratios 0.3934 to 0.3997, and the entire signal is one lane on one child. A
   wave that wants to measure anything below 0.40x needs the 30-prompt extension
   bank as a **primary** surface, not a rider.
7. **PLANE and XCHANGE have now been deferred by two consecutive waves for the
   same reason.** Either budget a wave whose deliverable is a solver and nothing
   else, or take them off the slate.
