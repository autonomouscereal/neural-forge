# NOTES-MX-AU — wave 43: the solver wave. PLANE is built and re-sealed by its own child, XCHANGE closes without a build, and the sub-0.40x floor moves a fourth time

Branch `codex/lrc-blank-compression`. Four commits, four deployments, three
graded children, ledger 1040 → 1046.

**The one-line version.** Both solvers that waves 41 and 42 deferred exist now.
The plane-energy census settled both of the slate's predictions *in opposite
directions* before any child was built, and then the child it funded read
**1/15 against a registration of 12** — the largest registration miss in the
campaign — and re-sealed the axis with the attribution already on record. The
XCHANGE go-gate failed all three of its pre-registered thresholds at all eight
conv sites and **closed the class without a build**, by the authority its own
registration granted it. Meanwhile `ARCH.V5CL768` moved the sub-0.40x floor to
**0.3951221469x**, and `ARCH.K7680CL896X13` halved the block-13 cliff bracket to
512 rows by landing on the side the registration did not predict.

---

## 1. The tree, and the second append-only list nobody was checking

Five substitutions, no prose. Four hand edits: 38 seal roots become 39; `KEEP`
and `DROP` emptied; one schema string retargeted, sixteen module references,
both receipt write sites, the log prefix and the header's own diff reference;
and `sealq.sh` arrived carrying wave 42's **three** seal calls.

**Wave 43 is the first cut to exercise wave 42's general calibration pin, and it
fired at exactly the predicted count.** `NOTES-MX-AT` §11.4 said the pin
"already reports 78 residue lines against `run/mx42/sealq.sh` as shipped". It
reported **78**, the first of them the sentence beginning *AND IT WAS RUN AND IT
IS EXACT ON THE COUNT THAT MATTERS*. Those lines carried wave 42's calibration
result, its nine-dossier byte comparison, its reading-order finding and its
three seal calls. All 78 deleted by hand. The three literal needles (wave 39's
sentence, dossier and counter) reported zero, which is what they should report
once the general pin has done the work.

### 1.1 And then the wave paid for a defect of exactly that family, in the one file the cut discipline does not reach

**There are TWO append-only accept lists in this campaign and the cut audit only
ever asserted one.** `BASIS_SCHEMAS_ACCEPTED` lives in the builder, inside the
tree the cutter copies, and has been asserted since wave 31. The **sealer's**
`BUILD_RECEIPT_SCHEMAS` lives in `experiments/seal_matrix_architecture_child.py`,
**outside** the cut tree, and every wave since 36 has appended it by hand with
nothing but a README paragraph behind the claim.

`run/mx43/README.md` said the append had happened. It had not. **The sealer
refused all three reviews on the first invocation** — exactly as it refused wave
35's four, commit `dcfa0c9c`, *"after it refused four reviews again"* — and wave
43 paid the extra deployment wave 35 paid. Nothing was mis-sealed: the refusal
fires before a cell receipt or a ledger shard exists.

A second refusal of the same family fired in the same place. `mx43_plane.py`'s
build receipt was written from memory of what a build receipt looks like rather
than from what the sealer actually reads: it carried `parameters` where the
sealer reads `child_parameters`, and no `axes` at all — a field the sealer
requires and `matrix_cell_receipt` validates nine subfields of.

**Both are now checks rather than habits.** `cut_wave43.py --audit` asserts the
sealer's list by name, ten strings deep. The plane receipt is written against
the contract read out of the sealer. And the plane child carries **its own
schema**, `neural-forge-matrix-wave43-rope-band-child-v1`, for the reason waves
40 and 41 gave when they added theirs.

---

## 2. THE PLANE-ENERGY CENSUS RAN FIRST, GATED NOTHING, AND SETTLED BOTH SLATE PREDICTIONS IN OPPOSITE DIRECTIONS

`docs/ROUND3_SLATE_2026-08-28.md` §3.4 extrapolated from HD16's "~0.24 retained
score energy ≈ 8/32 uniform" to **~0.75 at d=48 and ~0.50 at d=32**.

| | slate predicted | measured |
|---|---:|---:|
| d=48, m=1, s=0 | 0.75 | **0.864581** |
| d=32, m=1, s=0 | 0.50 | **0.302230** |

Both wrong, in opposite directions, because the uniform-energy assumption behind
them is false. **Plane 21 alone carries 0.310674 of the pooled score energy and
plane 3 carries 0.003908 — a dynamic range of 80.** Four planes (21, 0, 22, 23)
carry 0.6465 of everything.

The census is wave 15's `plane_energy` unchanged, and it reproduces wave 15
across twenty-eight waves: HD16's band (d=16, planes 0..7) reads **0.237895**
here against the "about 0.24" in HD16's own receipt.

### 2.1 The thing the campaign has never written down

Over the 264-token sealed sequence, phase excursion end to end:

| plane | 0 | 8 | 12 | 16 | 21 | 24 | 31 |
|---|---:|---:|---:|---:|---:|---:|---:|
| radians | 263.0 | 8.32 | 0.624 | 0.263 | **0.0304** | **0.0083** | 0.0004 |

**The dominant planes barely rotate.** Plane 21 carries a third of the score
energy and turns three hundredths of a radian across an entire sequence; planes
24 and up are position-independent for anything this evidence can see. So the
parent's attention on this evidence is dominated by a near-DC **content** term
carried in low-frequency RoPE planes, and the genuinely positional planes
(p ≤ 8) carry **0.246883**.

That is *exactly* the band wave 15's HD16 kept, and HD16 read 0/15. Registered
as the mechanism attribution **before** the build, which is what the slate asked
for — and it is why the HD48 registration was 12 and not lower.

### 2.2 The band slide is load-bearing at d=32 and not at d=48

At d=48 the default window already contains all four dominant planes and every
slide is worse (s=0 0.864581, next best s=8 0.762105). At d=32 the best
expressible band is **s=16 at 0.697770** against the default s=0's **0.302230**
— **2.31x at byte-identical storage**, 220,255,616 parameters either way. That
is §4 lever 3's first quantitative support and it is registered now so a later
wave cannot claim it was obvious.

---

## 3. THE XCHANGE GO-GATE CLOSED THE CLASS WITHOUT A BUILD

The thresholds are the `GO_GATE` table in commit `75323e57`'s
`mx43_xchange.py`, one declared dict, each entry carrying its reason, committed
**before the census existed** and untouched afterwards
(`git show 75323e57:reports/matrix-campaign-20260810/run/mx43/mx43_xchange.py`).

| site | rho_2 fit | rho_2 held | error floor | disp B | disp C |
|---|---:|---:|---:|---:|---:|
| L00 | 0.5329 | 0.5442 | 0.7300 | 0.9185 | 0.9626 |
| L01 | 0.8155 | 0.8235 | 0.9031 | 0.8478 | 0.8801 |
| L03 | 0.8118 | 0.8565 | 0.9010 | 0.7192 | 0.7442 |
| L05 | 0.5510 | 0.6025 | 0.7423 | 0.5867 | 0.5974 |
| L07 | 0.6644 | 0.6969 | 0.8151 | 0.6746 | 0.6355 |
| **L09** (value site) | **0.5326** | 0.5859 | 0.7298 | 0.5275 | 0.5978 |
| L11 | 0.6972 | 0.7106 | 0.8350 | 0.5734 | 0.7279 |
| **L13** (go-gate site) | 0.6647 | 0.6921 | 0.8153 | **0.4993** | 0.5661 |

- **G1 (rho_2 ≤ 0.50): fails at all eight.** The best site is 0.5326. `rho_2` is
  the share of the conv delta that is *provably* outside the linear-value
  attention class, so `sqrt(rho_2)` is the substitution's error floor **before**
  any solve error: 0.73 to 0.90, against deletion's 1.00, at 3,145,856
  parameters deletion does not spend. At L13 the per-sequence spread is
  0.6290/0.6666/0.7449 over 24 batches — flat, not one wild sequence.
- **G2 (median dispersion ≤ 0.50): fails at all eight.** Exactly one of sixteen
  numbers passes: block 13's B gate at 0.4993.
- **G3 (Stage-1 band mass ≥ 0.90): fails globally.** The best registered
  multi-plane construction reaches **0.2779**; a single plane reaches **0.0020**,
  which is the slate's own alias objection confirmed — cos returns to 0.9998 at
  an offset of 44.

**The reconstruction gate held at every site**: the linear part plus the
remainder sums back to the fp64 delta, and the fp64 recomputation scores 3.9e-3
to 4.8e-3 against the live bf16 module — a bf16 floor, so the decomposition is
of the operator the model actually runs.

### 3.1 The post-hoc extension, published because the registration was too small

The registered plane sets stop at eight, so "no plane set can concentrate" was a
stronger claim than the registration measured. Published beside the gate and
**labelled post-hoc**:

| construction | worst-lag band mass |
|---|---:|
| 2 planes | 0.0078 |
| 8 planes (deepest **registered**) | 0.2779 |
| 12 planes | **0.3431** (best of everything tried) |
| 16 planes | 0.3193 |
| all 32 planes | 0.0520 |
| design doc's own free-amplitude least squares, 32 planes | 0.1173–0.2406 |

More planes do not rescue it and the design document's own construction is worse
than unit amplitudes; beyond twelve it *decreases*, because the high-index planes
barely rotate and contribute a near-constant that flattens the peak. **To reach
band mass 0.90 the pre-softmax score range would have to be 671.8 against the
parent's measured 10.72 — a factor of 63**, and against the parent's *absolute
maximum* logit of 23.16 it is still a factor of 29.

The verdict does not rest on this file: **G1 fails at all eight sites on its
own.** XCHANGE closes by its own pre-registered gate, with no child built and no
ledger row touched.

---

## 4. F1: `ARCH.V5CL768` — the sub-0.40x floor moves a fourth time

K=5,120 on wave 40's retained `MX40-CL768`, one CPU pass, **90,756,864
parameters, 0.3951221469x, 1/15.** Registered at ONE on seven rungs of flatness
against a cliff account that predicted zero.

The floor moves from 0.3996872628x by **0.0045651159** of stored ratio, and the
CL768 ladder is now **eight rungs flat** — the body at 0.6644639834x and V12,
V10, V9, V8, V7, V6, V5 — across 0.2693 of ratio. It is the only flat ladder
left in the campaign.

**The wave was warned about the wrong floor, and the census said so before the
build.** The stated risk was that the builder would refuse below observed-
vocabulary coverage. The **structural** floor of this axis is **765 mandatory
rows** — 509 added/special plus 256 ByteLevel plus closure — so at K=5,120
nothing is skipped for want of closure room and about 4,355 rows are
discretionary. What fired was the **coverage** half at 0.98530 post-closure,
which `BELOW_FLOOR=1` releases and `cut_wave43.py --audit` asserts present.

Its held lane is the summarisation lane, sha256 `f0f9cef07b1aa994`, **shared
with nothing in the estate** — not with `ARCH.V6CL768` or `ARCH.V7CL768`, which
agree with it on six of sixteen lanes each. **And the held lane is a copy**: the
source paragraph returned in full at six sentences against an ask for two, with
substitution artifacts throughout (*comperses three bitties, Suggesting A,
Suggested B, remainringing, procuration*). It preserves both asked-for items,
which is why it holds under the clause wave 40 sealed and why column B voids it.
All sixteen lanes charged; three ride the cap; **seven return the prompt itself**.

---

## 5. F2: `ARCH.K7680CL896X13` — the block-13 cliff halves to 512 rows

K=7,680 on wave 35's retained `MX35-CL896X13`, **91,931,392 parameters,
0.4002356117x, 0/15.** Registered at ONE on two accounts of three.

**The coverage guard was ARMED on this rung.** `BELOW_FLOOR` was not passed,
because the vocabulary census read post-closure occurrence *and* distinct
coverage of exactly **1.000000** at K=7,680. The guard was exercised rather than
released, and it did not fire — the first rung in this band built that way.

The cliff is now between **K=8,192** (`ARCH.V8CL896X13`, 1/15) and **K=7,680**.
The bracket halved exactly as registered — the registration said it would halve
either way and named that as the reason to spend a review at a ratio that cannot
move the floor — **and it halved on the side the registration did not predict**,
so the dead band on this body is wider than the flat band and the cliff sits
immediately below the last living rung.

**Its summarisation lane is byte-identical to `ARCH.V7CL896X13`'s**, sha256
`3823843e49001dc4`, 512 rows below it. The lane that decides the count on this
body is **one decode read at two sizes**. It shares eight of sixteen lanes with
the living rung above and seven with the dead rung below, and it answers with
the dead family. Eight of sixteen lanes ride the cap.

---

## 6. F3: `ARCH.HD48` — the axis is opened and re-sealed by one child, and the solver is not what failed

head_dim 48 on the parent, band (m=1, s=0) = planes 0..23, `rope_theta`
1e6^0.75 = 31,622.7766, **224,974,400 parameters, 0.9794561427x, 1/15**, against
a **registration of 12**. That is the largest registration miss in the campaign
and it re-seals the frequency-band axis under the slate's own kill gate (< 10/15
solo re-seals, *with* the pre-registered energy attribution).

| site | logit relL2 prior → solved | kept-plane logit share | out_proj cos prior → solved | update_scale o / q / k |
|---|---:|---:|---:|---:|
| A02 | 0.1417 → **0.1066** | 1.0139 | 0.9773 → 0.9826 | 0.753 / 0.220 / 0.205 |
| A04 | 0.2313 → **0.1463** | 0.7896 | 0.9499 → 0.9610 | 0.681 / 0.219 / 0.148 |
| A06 | 0.2094 → **0.1499** | 0.7600 | 0.9513 → 0.9694 | 0.666 / 0.332 / 0.161 |
| A08 | 0.0911 → **0.0770** | 0.8937 | 0.9352 → 0.9614 | 0.695 / 0.119 / 0.052 |
| A10 | 0.1926 → **0.1382** | 0.7640 | 0.9440 → 0.9562 | 0.736 / 0.329 / 0.273 |
| A12 | 0.1588 → **0.1246** | 0.8097 | 0.9654 → 0.9745 | 0.674 / 0.213 / 0.117 |

**The three secondary registered predictions, graded:**

1. *`held_kept_plane_logit_share` ≥ 0.80 at all six sites* — **MISS at three of
   six** (0.7896, 0.7600, 0.7640); mean 0.8385 against the census proxy's
   0.8646. The second-moment proxy is optimistic by about three points and it
   does transfer in the right neighbourhood. A02's 1.0139 is above one and is
   not a bug: the dropped planes' contributions partially *cancel* the kept
   ones, so a partial sum can carry more energy than the whole.
2. *`held_logit_relL2_solved` ≤ *`prior`* at all six sites* — **HIT, 6 of 6**,
   by 17% to 37% relative. The prior *is* wave 15's install (sliced gain × mean
   RMS ratio × sqrt(d/64)), so the d-unknown ridge beats the scalar on **held**
   rows at every site.
3. *`update_scale` ≤ 1.0 on all three solved objects at all six sites* — **HIT**,
   at 0.67–0.75 (out_proj), 0.12–0.33 (q gain), 0.05–0.27 (k gain). Nothing was
   refused; nothing was near the veto.

**So the solver is not what failed.** The band keeps 0.864581 of the pooled
score energy against HD16's 0.237895, all four dominant planes are inside the
window, q/k rows and value channels are `torch.equal`-verbatim, and every solved
object is well inside THE LAW. What fails is the **axis**: six sites at
out_proj held cos 0.94–0.98 compose to an **end-to-end held relL2 of 0.6435 and
cos 0.7793**. And the exported child, reloaded from disk through
`AutoModelForCausalLM`, reproduces that number **to the last digit**
(0.6434915430225171 / 0.7792878687231457 / 0.662058097468711), so it is the
fold's result and not an export defect.

Its one held lane is the greeting, sha256 `80c1b8d576223bc2`, shared with
nothing in the estate: it responds naturally and asks exactly one follow-up
question, and the question never acknowledges the long day — the ask answered in
part and not in full. Ten of sixteen lanes charged, five ride the cap.

---

## 7. Benchmarks, and the sharpest instrument failure yet

Standard invocation, parent FIRST. **The parent row reproduces waves 32 and 34
through 42 to every reported digit — 30 parent rows, all 30 matching**, and the
campaign table confirms it across 261 benchmark rows.

| model | ratio | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | sealed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 1.000000 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | **81.80** | 12/15 |
| `ARCH.HD48` | 0.979456 | **0.4650** | **0.4400** | **0.5900** | 0.5150 | 0.0350 | **7301.12** | **306.09** | **1/15** |
| `ARCH.K7680CL896X13` | 0.400236 | 0.3150 | 0.3150 | 0.5150 | **0.5200** | **0.0950** | 116954.88 | 19122.08 | **0/15** |
| `ARCH.V5CL768` | 0.395122 | 0.2850 | 0.3350 | 0.5200 | 0.4650 | 0.0850 | 33754.88 | 6657.30 | **1/15** |

**`ARCH.HD48` and `ARCH.V5CL768` seal the same count — 1/15 — and their
wikitext word perplexities differ by a factor of 22 (306.09 against 6,657.30)
and their lambada perplexities by a factor of 4.6.** `ARCH.HD48` beats the deep
children on four of five accuracy columns.

Wave 42 §11.6 said the 15-case instrument is past exhausted *at this depth*.
This wave shows it is exhausted at **0.98x** as well: it cannot distinguish a
model at 306 wppl from one at 6,657 wppl. That is the fifth inversion the
benchmark column has produced and by far the widest.

---

## 8. THE DUAL COLUMN, refreshed over 26 children, both floors published

`mx43_dualband.py` at ≤ 0.4200, machine pass first, then the hand reading merged.
**0 anomalies. 8 machine-vs-hand disagreements, published.**

- **Column A (copy-backs count): floor = 0.3951221469x, `ARCH.V5CL768`**,
  holding one lane. The floor moves.
- **Column B (copy-backs refused): floor = 0.4116484014x, `ARCH.V10CL896X09`**,
  holding `uncertainty-insufficient-data`. **Unchanged since wave 42.**

### 8.1 Wave 42 told this wave to pre-register the union measure and this wave did not, and the case it was flagged for is the case that occurred

`NOTES-MX-AT` §11.3: *"the longest-run measure gets 10 of 17 copies right; the
union measure gets 17 of 17… wave 43 can, by registering it first."* **It was
not registered, and the longest-run measure was used.** On this wave's own floor
mover:

| lane | `payload_coverage` (longest run) | `payload_union_coverage` | machine | hand |
|---|---:|---:|---|---|
| `ARCH.V5CL768` summarisation | **0.157761** | **0.804071** | CONTENT | **COPY_BACK** |
| `ARCH.V6CL768` summarisation | 0.430025 | 0.936387 | CONTENT | **COPY_BACK** |
| `ARCH.K7680CL896X13` summarisation | 0.099237 | 0.223919 | CONTENT | (holds nothing) |

The threshold is 0.60. The longest-run measure calls the floor mover's lane
clean; the union measure puts it at 0.804, inside wave 42's measured copy band
(0.827–0.997) and far outside its genuine-summary band (0.198–0.331). **The hand
reading governs** — the module's own design says the machine does not get the
last word on a held lane — so column B is correct, but it is correct because a
human read it and not because the instrument worked. This is the honest cost of
not taking wave 42's advice, and the union number is published beside every lane
so the next wave can register it without re-running anything.

`ARCH.K7680CL896X13`'s union coverage of 0.2239 sits in the *genuine summary*
band and it holds nothing: at this rung the child cannot even copy.

---

## 9. Method and controls

- **Parent calibration, sixteenth consecutive, exact on the count that matters.**
  Re-read blind from `ARCH.V5CL768`'s dossier — the first of the wave, sealed
  08:27:37.916Z — before any child lane was opened. **12/15**, **cap-hits
  0/16**, both documented anchors verbatim: the arithmetic Partial boxing *3
  kits per box with 94 left over*, and the multi-turn miss offering *Vegan
  Peanut Butter & Jelly Bars* to a table with two severe peanut allergies.
- **The split is 6 Useful / 6 Partial / 3 miss** against wave 42's 4/8/3 and the
  reference's 5/7/3; all three totals are 12/15 and all three name the same
  multi-turn miss. Two calls were made against the **prompt** rather than a
  memory of it and are recorded in `sealq.sh`: `context-retrieval` is Partial
  (the ask names three things and the parent never says Saturday), and
  `comparison-ram-rom` is a **miss** (the ask is one analogy plus two
  imperfections; the parent asserts that both RAM *and* long-term storage
  "don't store information or data").
- **The runner control is NINE DOSSIERS WIDE ACROSS FOUR WAVES.**
  `lanebytes43.py` opens all three wave-43 dossiers and six from waves 40, 41
  and 42 and finds the parent's sixteen lane digests **identical 16/16 in every
  one**, at protocol `e5e377688597` and prompt pack `0ee6bf14cef5` throughout.
- **The lane byte comparison ran before any count was written**, and it decided
  two claims by construction: both wave-43 held lanes are **independent
  witnesses** shared with nothing, and `ARCH.K7680CL896X13`'s summarisation lane
  is byte-identical to `ARCH.V7CL896X13`'s.
- **The reading order for children under 0.60x was followed**: the summarisation
  lane of both vocabulary children was read first and written down before any
  other lane was opened. It decides both seals.
- **Every child asserted its parameter count before writing** and all three
  landed on the pre-registered value exactly: 224,974,400 / 90,756,864 /
  91,931,392. Identity gates: `torch.equal` on q rows, k rows and v channels for
  the plane child; `torch.equal` on the whole gathered table for both vocabulary
  children.
- **The ladder was checked against an independent re-derivation from two
  sources** — this build's own value and the value that landed in the written
  config — because a build that checks its own variable against itself is
  checking nothing.
- **The plane geometry was verified before the tree was committed**: `rope_apply`
  matches transformers' own `apply_rotary_pos_emb` to 6.5e-15, the identity band
  at d=64 reproduces the parent's logits **exactly** (0.0), `logit_terms`'
  linearity holds to 1.3e-15 on both sides, the kept-plane target equals the
  per-plane partial sum to 1.3e-15, `causal_fir` matches `nn.Conv1d` to 2.2e-16,
  the §B.2 decomposition sums back to 4.4e-16, and frozen gates drive `rho_2` to
  **5e-32** — the exact limit the design document says the identity holds at.
- **Seal verification**: 1046 ledger rows, **1042 resolve to a receipt on disk,
  1042 verify, 0 FAIL**, 4 unresolved — the same historical four. **39 seal
  roots** audited against disk, **0 undeclared**.
- **`sealq.sh` ran ONCE** and all three sealed on the first call.
- **A dry run preceded the ledger merge** and left the ledger at 1040 rows and
  `803439d1…` unchanged — verified, not assumed. **The first dry run used the
  wrong argument form**: it passed the three verdict shards to
  `--verdict-receipt` instead of `--shard` and reported three rows where wave 42
  reported six. It was caught by reading the report against wave 42's, before
  any apply. The corrected form — six `--shard`, three `--verdict-receipt`
  pointing at the behaviour-reviewed cell receipts — was dry-run again and then
  applied in **one pass**.
- **Two background jobs raced on one output path and were killed.** Two
  `mx43_campaign_table.py` processes — a timed-out foreground invocation that
  never died plus its relaunch — were both writing
  `campaign-frontier-table.json`. Both were killed, the partial artifact
  deleted, and a single instance re-run. The table in the wave root is that
  single run's.
- **Pre-registration hit rate: 1 of 3** on the sealed counts (`V5CL768` 1, hit;
  `K7680CL896X13` 1→**0**; `HD48` 12→**1**), plus 2 of 3 on the secondary
  quantitative predictions.

---

## 10. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `75323e57` | cut wave 43's tree and author both solvers, with the geometry verified before the commit |
| `ec94b610` | run both censuses and pre-register |
| `20cbdd3c` | append the sealer's second accept list and fix the plane receipt's contract |
| `18ca9b50` | the three seals |
| `c6fb9b03` | the measurement, the tables and the dual-column doc refresh |
| *(this)* | retention applied, and this file's own estate section closed |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `75323e57` | 7597 | 0 | 0 | 0 | 0 |
| `ec94b610` | 7603 | 0 | 0 | 0 | 0 |
| `20cbdd3c` | 7603 | 0 | 0 | 0 | 0 |
| `18ca9b50` | 7603 | 0 | 0 | 0 | 0 |
| `c6fb9b03` | 7654 | 0 | 0 | 0 | 0 |

**Five deployments for five commits, which is the campaign's normal rate, and
exactly ONE of them was avoidable.** Waves 40 through 42 each cut one deployment
per commit. Wave 43 has one commit more than wave 42 for a structural reason
(`env.sh` states it: the censuses must RUN before the predictions they make
possible can be registered, so the code commit and the pre-registration commit
cannot be the same commit) and one more for the accept-list defect in §1.1.
`20cbdd3c` is the avoidable one.

**Ledger.** Re-verified on the share before it was touched (1040 rows,
`803439d1…`), backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave43-20260829T084718Z` and
`cmp`-verified byte-identical. **One-pass merge**, 1040 → **1046** rows, 0
superseded. Byte-prefix verified: `head -c 1831910 ledger.jsonl | sha256sum`
returns `803439d1…` exactly. New digest `264884995e0c2b87…`. **Mirror copy-back
done, and the mirror write asserted the merged file is a byte-prefix extension
of what it replaced BEFORE writing.**

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used for the
builds, the censuses, the reviews and the benchmarks, with the SM86 preflight on
every driver. Both carried an idle-holding H3 ComfyUI container (~288 and ~279
MiB) and were **not restarted**, for wave 34's stated reason. The four P40s were
never touched. The Quadro RTX 8000 read **330 MiB** and wave 43 issued no
command against index 6. One ad-hoc diagnostic — the parent's absolute maximum
logit — was run **without** the pinned-GPU wrapper and is labelled a diagnostic
for that reason; the γ cap that the sealed go-gate artifact carries came from
the pinned run.

**Disk**: `/mnt/ai_data` 4.5T, 1.7T available. **The Codex lane was not touched.**

**Retention**: `package.sh` ran BEFORE `--plan`, as the rule requires, and
`--plan` was read before `--apply`. KEEP = `MX43-V5CL768` (the frontier holder
and the wave's only new deep-band held-lane witness, and the lane the machine
and the hand disagree on) and `MX43-HD48` (the only living child on the
frequency-band axis; not a one-CPU-pass rebuild). DROP =
`MX43-K7680CL896X13`, rebuilt by one CPU pass off the retained
`MX35-CL896X13`, **without** `BELOW_FLOOR`. **Applied**: 636,675,736 bytes kept,
184,360,369 deleted, manifest re-parsed OK.

**Final state**: both 3090s back to their idle H3-container residents (288 and
279 MiB) at 0% utilisation, no stray `mx43_*` process on the box, ledger 1046
rows at `264884995e0c2b87…`, `/mnt/ai_data` 1.7T free.

---

## 11. What this wave did NOT build, and why

1. **XCHANGE has no child and never will under this contract.** Closed by its
   own pre-registered gate, §3. This is a result, not a deferral.
2. **PLANE's d=32 rungs and the three-arm band portfolio** — not built. The
   slate's kill gate makes them contingent on an HD48 pulse and there was none.
   The census numbers that would price them are registered and published.
3. **ROWSET Arms A and B** — still blocked on corpus; no corpus has appeared.
4. **ALIAS** — still blocked behind the PROBE storage-claim ruling (K4, K3(c)).
5. **NORM-ESTATE Arms M and S** — still need a new solver. Wave 43 authored two
   and did not have room for a third, and says so rather than implying it was
   considered and rejected.
6. **The union copy measure was not pre-registered**, §8.1. That is this wave's
   own miss against `NOTES-MX-AT` §11.3, and it cost the machine column its
   agreement with the hand column on the wave's own floor mover.

---

## 12. The two rulings

**Ruling 1 — the PROBE storage claim** (`NOTES-MX-AS` §12), unchanged and still
blocking three classes: K4, K3(c) and ALIAS. *May a `PROBE.*` child carry a
storage claim that is stated arithmetic, labelled non-loader, and excluded from
the frontier?*

**Ruling 2 — the copy-back clause**, unchanged, and still costing nothing to
apply. Both readings are complete over 26 children, per child and per lane, with
the digest that makes each call checkable:

- **NO** (copy-back refused): the intelligence floor is **0.4116484014x**,
  `ARCH.V10CL896X09`, holding `uncertainty-insufficient-data`. Unmoved by this
  wave.
- **YES** (copy-back counts): the floor is **0.3951221469x**, `ARCH.V5CL768`,
  and the canonical clause's copy-back sentence must be rewritten.

---

## 13. What the next wave should know

1. **THE FREQUENCY-BAND AXIS IS RE-SEALED AND THE ATTRIBUTION IS ON RECORD.**
   0.864581 of the score energy retained, every solve inside THE LAW, the ridge
   beating wave 15's scalar at all six sites — and 1/15. Two children now exist
   on this axis, HD16 at 0.238 retained reading 0/15 and HD48 at 0.865 retained
   reading 1/15, and **retention does not predict survival on it**. Anyone who
   wants to reopen it must argue against that pair, not against a rung.
2. **THE 15-CASE INSTRUMENT IS EXHAUSTED AT 0.98x, NOT ONLY BELOW 0.40x.** §7.
   It reads `ARCH.HD48` and `ARCH.V5CL768` the same at a 22x gap in word
   perplexity. **The 30-prompt extension bank needs to become a primary surface
   in the next wave that grades anything, at any depth.**
3. **REGISTER THE UNION MEASURE.** §8.1. Wave 42 said so, wave 43 did not do it,
   and the lane it was flagged for is the lane that decided this wave's floor.
   The number is published per lane in
   `wave43/deep-band-dual-assessment.json`; registering it costs one constant.
4. **THE BLOCK-13 BRACKET'S MIDPOINT IS K=7,936** (92,193,536,
   **0.4013768907x**) on the retained `MX35-CL896X13` — one CPU pass, and it
   quarters a cliff that two waves have now spent a review on. It lands *above*
   0.40x, like this wave's halver, so a survival there **cannot** move the
   sub-0.40x floor; its value is locating the cliff. If the next wave would
   rather have depth than location, the deeper cell on that body is K=7,424
   (91,669,248, 0.3990943327x) — but that one is *below* an already-dead rung and
   the honest prior for it is zero.
5. **THE CL768 LADDER IS EIGHT RUNGS FLAT AND HAS NEVER FAILED.** `ARCH.V4CL768`
   (K=4,096, 89,708,288, **0.3905570311x**) is the next rung and the only cheap
   cell that can move the sub-0.40x floor again. The census says the structural
   floor is 765 rows, so the axis has room down to about K=1,024 before the merge
   closure is even a question; what will end it is coverage, or the cliff.
6. **THERE ARE TWO APPEND-ONLY ACCEPT LISTS AND NOW BOTH ARE CHECKED.** §1.1.
   The natural next place to look is every OTHER file outside the cut tree that
   a wave edits by hand each time — `seal_verdict.sh`'s `--parent-useful-or-partial`
   is pinned, but nothing pins `experiments/seal_matrix_behavior_review.py`'s own
   constants, and no cut has ever read that file.
7. **THE BAND SLIDE IS A REAL KNOB AND IT IS UNSPENT.** At d=32 it is worth
   2.31x in retained score energy at byte-identical storage. The axis is
   re-sealed, so nothing should be built on it without a new argument — but if
   §4's capability-portfolio question is ever funded, this is lever 3 and its
   price is now known without a build.
