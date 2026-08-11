# Wave 9 (MX-M) — the phase-1 package Cyril scores

**The ask, verbatim:** *"[Cyril quote withheld]"*

**The answer, in one line: 0.25× was not reached coherently. The deepest child
built is 0.736305×, the most coherent is 0.780861×, and every one of the ten is
machine-sealed `dead` on the degeneracy clause — but one of them is 9/15 against
a 12/15 parent with thirteen of sixteen lanes terminating on their own, and
whether that is a "coherent answer" is the judgement this package exists to put
in front of you.**

Read `COMP.MA.MXM6.md` first. It is the whole argument.

---

## 1. Why 0.25× is arithmetic and not effort

The parent is 229,693,184 unique parameters. **67,108,864 of them — 29.2% — are
one tensor**, `model.embed_tokens.weight`, 65,536 × 1,024, tied to the readout.

| recipe | net |
|---|---:|
| the whole **body** at a true quarter, embedding untouched | **0.4691×** |
| the **embedding** at a true quarter, body untouched | **0.7808×** |
| both, everywhere, at a true quarter | **0.2500×** |
| wave 8's deepest **coherent** child (6 body components) | 0.9554× |
| wave 9's deepest child of any kind (6 body + embedding) | **0.7363×** |

0.25× is not "fold harder". It requires the embedding at 4× **and** the body at
4× at essentially every one of about a hundred sites. Wave 8 measured the body's
coherent limit at **six** components; the seventh is dead. Wave 9 measures the
embedding at 4× as coherent-adjacent but degenerate on its own. **The two
halves of the target fail independently, and they fail worse together.**

---

## 2. The frontier curve

Net stored ratio against the review band, every wave-9 child plus the two wave-8
children that bracket the old frontier. Deepest first.

| child | folded | net stored | Δ params vs parent | what is folded | review | machine verdict | lanes at cap |
|---|---:|---:|---:|---|---:|---|---:|
| `COMP.MA.MXM9` | 7 | **0.736305×** | −60,568,842 | MXL11's body + embedding route S4 | 4/15 | dead | 3 |
| `COMP.MA.MXM8` | 1 | 0.744297× | −58,733,184 | embedding route **S8**, head 8,000 | 3/15 | dead | 2 |
| `COMP.MA.MXM4` | 1 | 0.744331× | −58,725,376 | embedding route S8, head 4,096 *(old head rule)* | 0/15 | dead | 16 |
| `COMP.MA.MXM10` | 5 | 0.753440× | −56,633,225 | MXL5's body + embedding route S4 | 6/15 | dead | 4 |
| `COMP.MA.MXM5` | 1 | 0.780727× | −50,365,440 | embedding route S4, head 8,192 | 6/15 | dead | 4 |
| `COMP.MA.MXM3` | 1 | 0.780772× | −50,355,200 | embedding route S4, head 4,096 *(old head rule)* | 3/15 | dead | 16 |
| `COMP.MA.MXM7` | 1 | 0.780830× | −50,341,744 | embedding route S4, head 15,000 | 5/15 | dead | 4 |
| `COMP.MA.MXM1` | 1 | 0.780857× | −50,335,744 | embedding **plain SVD** rank 252 | 0/15 | dead | 10 |
| `COMP.MA.MXM2` | 1 | 0.780857× | −50,335,744 | embedding **frequency-weighted SVD** rank 252 | 0/15 | dead | 16 |
| **`COMP.MA.MXM6`** | **1** | **0.780861×** | −50,334,720 | **embedding route S4, head 12,288, tail rank 77** | **9/15** | dead | **3** |
| `COMP.MA.MXL11` *(wave 8)* | 6 | 0.955444× | −10,234,122 | six body components, no embedding | 9/15 | **alive-degraded** | 0 |
| `COMP.MA.MXL5` *(wave 8)* | 4 | 0.972579× | −6,298,505 | four body components, no embedding | 10/15 | **alive-parent-band** | 0 |

**Every folded component in every row stores at or below a true quarter of its
own parent tensor**, re-derived from the sealed counts by
`run/mxm/mxm_addendum8.py`, which refuses a hand-supplied success value. Five of
the ten wave-9 children store at or below a true **eighth**.

---

## 3. The full eval matrix

lm-eval, pinned identically for every row: `--limit 200 --seed 1234
--batch_size 4`, `max_length=2048`, float32, one PCI-pinned sm86 RTX 3090.
**These are partial evaluations by construction** (200 documents per task); they
rank these models against each other and are not comparable to published
full-split numbers. Random-chance floors: arc_easy 0.25, hellaswag 0.25,
piqa 0.50, winogrande 0.50, lambada 0.

| model | n | net stored | review | arc_easy acc | hellaswag acc_n | piqa acc_n | wino acc | lambada acc | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **parent (unfolded)** | 0 | 1.000000 | 12/15 | **0.5450** | **0.5050** | **0.6800** | 0.5500 | **0.4600** | **24.39** | **81.80** |
| `COMP.MA.MXM9` | 7 | 0.736305× | 4/15 | 0.3700 | 0.3150 | 0.5100 | 0.5150 | 0.0700 | 15,185.40 | 515.07 |
| `COMP.MA.MXM8` | 1 | 0.744297× | 3/15 | 0.2750 | 0.2800 | 0.5150 | 0.4750 | 0.0550 | 2,315,739.18 | 11,527.79 |
| `COMP.MA.MXM10` | 5 | 0.753440× | 6/15 | 0.3950 | 0.3200 | 0.5400 | 0.4650 | 0.0800 | 11,613.26 | 501.34 |
| `COMP.MA.MXM5` | 1 | 0.780727× | 6/15 | 0.3800 | 0.3600 | 0.5400 | 0.5500 | 0.0900 | 2,628.37 | **313.97** |
| `COMP.MA.MXM3` | 1 | 0.780772× | 3/15 | 0.3850 | **0.4400** | **0.5600** | 0.5100 | 0.0600 | 5,496.31 | 471.21 |
| `COMP.MA.MXM6` | 1 | 0.780861× | **9/15** | 0.3650 | 0.3200 | 0.5200 | 0.5350 | **0.1100** | 7,337.19 | 421.68 |
| `COMP.MA.MXL11` *(w8)* | 6 | 0.955444× | 9/15 | 0.5250 | 0.4600 | 0.6350 | **0.5700** | 0.2250 | 99.22 | 118.02 |
| `COMP.MA.MXL5` *(w8)* | 4 | 0.972579× | 10/15 | 0.5100 | 0.4850 | 0.6200 | 0.5500 | 0.3200 | 55.19 | 105.53 |

**The benchmark says something the review cannot, and it is the most important
number in this package.** Wave 8's body folds kept the parent's *discrimination*
almost intact — `MXL11` at 0.955444× is within 0.02 of the parent on arc_easy
and piqa — and lost only the precision to emit the exact next token. **The
embedding fold loses both.** Every wave-9 child sits at or barely above chance
on all four multiple-choice tasks: arc_easy 0.275–0.395 against a 0.25 floor,
piqa 0.51–0.56 against a 0.50 floor, winogrande 0.465–0.55 against a 0.50 floor.

That is a qualitatively different failure. A body fold degrades a model. **A
4× embedding fold replaces the coordinate system every one of the fourteen
blocks was compiled against**, and the blocks do not survive it — even though,
as `COMP.MA.MXM6` shows, enough survives to write a usable email.

---

## 4. The judgement this package is asking for

`COMP.MA.MXM6`, at **0.780861×** of the parent, with the tied embedding stored at
**0.249954** of itself:

* **thirteen of sixteen lanes terminate on their own**;
* five responses are graded useful, four partial, six missed — **9/15 against
  the parent's 12/15**, the same score as wave 8's frontier child;
* it writes a usable email, gives three genuinely actionable items with no
  preamble, plans a domain migration, asks exactly the right four follow-up
  questions about an unlabelled sales graph, writes a good creative
  continuation, and computes 17 × 6 = 102 correctly before losing the thread;
* **three lanes ride the 1,024-token cap**, and two of those three are pure
  loops that leave nothing behind — `context-retrieval` repeats *the latest
  available information on* and `debugging-off-by-one` repeats `max(0,`.

It is sealed **`dead`** because the campaign's degeneracy clause is charged
whenever a lane's loop leaves no response behind, and that is the same test that
charged wave 8's `COMP.MA.MXL3` on one lane and spared `COMP.MA.MXL6` on one.

**A reviewer who charges the clause per LANE rather than per CHILD reads this
child `alive-degraded` at 9/15 and 0.780861×.** Both readings are defensible
from the same text, the text is reproduced in full in `COMP.MA.MXM6.md`, and the
call is yours.

---

## 5. What is in this directory

| file | what |
|---|---|
| `PARENT.md` | the unfolded parent's sixteen responses in full, plus its benchmark baseline and its 12/15 calibration |
| `COMP.MA.MXM6.md` | **the one to read** — embedding route at a true quarter, head 12,288 |
| `COMP.MA.MXM9.md` | the deepest child ever built: 0.736305×, body + embedding |
| `COMP.MA.MXM10.md` | 0.753440×, four body folds + embedding |
| `COMP.MA.MXM5.md`, `COMP.MA.MXM7.md` | the head sweep either side of 12,288 |
| `COMP.MA.MXM3.md`, `COMP.MA.MXM4.md` | the route under the *old* head-selection rule, which is why the rule changed |
| `COMP.MA.MXM1.md`, `COMP.MA.MXM2.md` | plain and frequency-weighted SVD of the table — the two arms that proved the table is not low rank |
| `COMP.MA.MXM8.md` | the embedding at a true **eighth** |

Every file carries: the full recipe with per-component stored/owned counts, the
net ratio, the benchmark table against the parent, the machine verdict as an
**annotation**, the per-case grades, the reviewer's sealed notes, and **all
sixteen responses verbatim and unedited, paired with the parent's own response
to the same prompt in the same process.** Nothing is truncated or summarized.

The narrative, the mechanism and the estate record are in
`../NOTES-MX-M.md`.
