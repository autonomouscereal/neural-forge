# Campaign scorecard — one page, for scoring

**Campaign:** stage-1 matrix, waves 1–26, `reports/matrix-campaign-20260810/`.
**Parent:** `LFM2.5-230M`, **229,693,184 parameters**, sealed **12/15** on the
frozen 15-prompt review (parent lane `865f006e3813712e…`, identical on every
dossier in every wave).
**Ledger:** 724 rows, sha `de12c860276e686eb519dad3161cf115c9d3b07ba149b349599513b2028db49b`.
**Reviewed whole-model children:** 193. **Scoring at least 1 of 15:** 145.

## Wave 26 stacked off wave 25's thirteen, and the thirteen held

Wave 25 sealed a 13/15 at 0.799135x and then wrote down four reasons nobody
should believe it. Wave 26 was sent to test the load-bearing one — that a single
reading of a single surface at a single K might be a measurement artefact — by
the only method that settles it: **build on top of it.** An artefact does not
carry a second fold.

| child | net | review | shape | what it is |
|---|---:|---:|---|---|
| `ARCH.V20X13G03` | **0.746614** | **11/15** | standard | the **deepest eleven in the campaign**, 1.6 points below the previous one. The 20,480-row coreset on the block-13 excision with one 1,024-entry diagonal re-solved at λ = 0.3 — zero parameters added, zero shapes moved. |
| `ARCH.V20X13` | **0.746614** | **10/15** | standard | the **deepest ten in the campaign by 5.5 points**, and the zero-solve control the row above is differenced against at an identical parameter count. Third child ever to hold `coding-deduplicate`. |
| `ARCH.V18` | **0.790005** | **13/15** | standard | K = 18,432. Ties wave 25's best review **nine thousandths of ratio deeper**. |
| `ARCH.V22` | **0.808265** | **13/15** | standard | K = 22,528. With `V18` it makes the campaign's highest review a **4,096-row plateau** rather than one lucky table. |

**The thirteen is a shelf.** 18,432 / 20,480 / 22,528 read 13 / 13 / 13. That
retires exactly one of wave 25's four doubts — the number is reproducible in K —
and none of the other three. It is still one reading of one sixteen-prompt
surface, the child still beats the parent on lanes where the parent is itself
weak, and two cases on fifteen is inside anybody's noise. **The claim remains:
between K = 18,432 and K = 22,528 this axis costs nothing this surface measures.**

**And the base carries a second fold.** `ARCH.V20X13` is the 13/15 body with the
campaign's mildest block excision composed onto it, and it reads ten at 0.7466x.
The excision costs three cases, not the one it cost the 32,768 rung, and it takes
them out of completeness rather than out of coverage: five lanes answered in full
become three. A third axis kills it — `ARCH.V20X13X07` reads 5/15 with nothing in
full — exactly as L3 has said since wave 23.

**Where the vocabulary axis dies:** inside 1,024 rows. `ARCH.V9` reads **9/15** at
0.748919x and 8,192 reads 4/15 at 0.744354x. Four thousandths of ratio separate a
nine from a four.

## Wave 25 changed the OTHER axis, and the change is larger than any row below

The table on this page ranks by **depth**, which is Cyril's criterion, and wave
25 moved that table by one row. What it moved much further is the **score** axis,
and those children are too shallow to appear below, so they are stated here.

| child | net | review | shape | what it is |
|---|---:|---:|---|---|
| `ARCH.V20` | **0.799135** | **13/15** | standard | vocabulary coreset at K = 20,480, **zero solve**, zero cap hits. The highest review in 25 waves, and **one more than the parent** on the same frozen surface in the same process. |
| `ARCH.V12` | **0.762614** | **11/15** | standard | the same axis at K = 12,288. Ties wave 24's best non-excision child **nine points of ratio deeper**, and strictly dominates `ARCH.V16` (9/15 at 0.780874x) on both axes at once. |
| `ARCH.V32X13` | **0.801395** | **10/15** | standard | K = 32,768 on the block-13 excision — the deepest **ten** in the campaign, and the first child ever to hold the deduplication lane at all. |

**Read the 13 with the scepticism it deserves.** It is one reading of one
sixteen-prompt surface; the parent's own 12 is the same kind of number; the child
beats the parent on lanes (planning, uncertainty, arithmetic) where the parent's
sealed lanes are themselves weak or wrong; and two cases on fifteen items is
inside anybody's noise. The claim it supports is narrow and it is enough:
**at 0.799x this axis costs nothing measurable on this surface.**

**And wave 25 falsified the law wave 24 proposed for that axis.** Wave 24 had
three points (K = 32,768 / 16,384 / 8,192 reading 11 / 9 / 4) and reported that
the review tracks the measured position agreement with the parent's segmentation
(0.3264 / 0.1460 / 0.0565) rather than the parameter ratio. Wave 25 built four
more rungs off the identical census. The curve is **non-monotone in both**:

| K | net stored | position agreement | review |
|---:|---:|---:|---:|
| 32,768 | 0.853916 | 0.3264 | 11/15 |
| 22,528 | 0.808265 | — | **13/15** |
| 20,480 | 0.799135 | 0.1736 | **13/15** |
| 18,432 | 0.790005 | — | **13/15** |
| 16,384 | 0.780874 | 0.1460 | 9/15 |
| 14,336 | 0.771744 | 0.1322 | 9/15 |
| 12,288 | 0.762614 | 0.0992 | **11/15** |
| 10,240 | 0.753484 | 0.0992 | 10/15 |
| 9,216 | 0.748919 | — | 9/15 |
| 8,192 | 0.744354 | 0.0565 | 4/15 |

K = 12,288 agrees with the parent's segmentation **less often** than K = 16,384
and reads two cases **higher**; K = 10,240 and K = 12,288 agree with it *equally
often, to four places*, and read 10 and 11. There is no law here yet — there is a
**cliff**, and wave 26 located it to within **1,024 rows**: 9,216 reads 9 and
8,192 reads 4, four thousandths of ratio apart. Wave 26 also added the top of the
curve: 18,432 / 20,480 / 22,528 all read 13, so the maximum is a shelf and the
curve has exactly one collapse in it.

**Wave 24 put a child seven cases deep BELOW the re-review line.** `ARCH.V8X13GL01`
is the 8,192-row vocabulary coreset on the excised body with **one 1,024-entry
diagonal re-solved** -- zero parameters added, zero shapes moved, stock
`Lfm2ForCausalLM` -- at **0.691832x and 7/15**, which is 8.4% below 0.7548x and
the deepest seven-case child in 24 waves on the standard-shape column. Two things
under it are worth as much as the row. First, the vocabulary axis now has a
**dose-response** and it does not track the parameter ratio: K = 32,768 / 16,384 /
8,192 read **11 / 9 / 4** while the ratio moves only 0.8539 -> 0.7444, and what
those three reviews DO track is the measured position agreement with the parent's
segmentation (0.3264 / 0.1460 / 0.0565). Second, the lambda that produced the
seven is **not** the lambda the solver's own out-of-sample criterion chose: four
rungs of one ladder at one parameter count read 6, 6, 7 and 4, and five
ten-thousandths of held-out terminal logit SSE separate the four from the seven.

The full table with both readings and every benchmark column is
`docs/PHASE1_FOLD_ONLY_FRONTIER_2026-08-12.md` §1. This page is the top of it,
ranked by **the criterion Cyril scores on: any intelligence per compression** —
the deepest children that still answer something. Rank is net stored ratio
ascending among children that hold at least one review case, because a child
that answers nothing has no intelligence to price and a child that answers at
0.99x has bought nothing with it.

Every child below is `dead` on the per-child ADDENDUM 6 calculator. That word is
a band, not a reading of the text; the per-lane column in the full table and the
character column here are what the text actually does.

## The fifteen

| rank | child | net | review | shape | the character of its failure | READ- dossier |
|---:|---|---:|---:|---|---|---|
| 1 | `ARCH.V16CL896` | **0.469305** | **1/15** | standard | under half the parent and it holds one lane: summarisation echoes the prompt and then names the decision, the risk and the mitigation correctly. Six of sixteen lanes fill the cap. A marginal reading at a depth nothing had read non-zero at. | `reports/matrix-campaign-20260810/wave23/human-review/READ-V16CL896.txt` |
| 2 | `ARCH.V16W896X13` | **0.623329** | **1/15** | standard | three axes; the third step is fatal exactly as L3 says. The one lane it holds is decided by the wave-20 planning calibration rather than answered. | `reports/matrix-campaign-20260810/wave23/human-review/READ-V16W896X13.txt` |
| 3 | `ARCH.V20X13W896` | **0.639307** | **2/15** | standard | **wave 26.** Three axes — 20,480 rows, width 896, block 13 gone — and it is the **deepest child ever to answer a lane in FULL**: summarisation names the decision, the one-week risk and the weekly-schedule mitigation, complete and correct, at 64% of the parent. Five of sixteen lanes fill the cap, two return the empty string. | `reports/matrix-campaign-20260810/wave26/human-review/READ-V20X13W896.txt` |
| 4 | `ARCH.V16W896` | **0.667288** | **2/15** | standard | the multiplying pair. Summarisation survives intact, the uncertainty lane names all four data points and then degenerates, two lanes return the empty string and retrieval returns a bare dash. | `reports/matrix-campaign-20260810/wave23/human-review/READ-V16W896.txt` |
| 5 | `ARCH.V20W896` | **0.683266** | **2/15** | standard | **wave 26.** The 20,480-row coreset on the width-896 body. Thirteen of sixteen lanes are a loop, an empty string, a fragment or **the prompt returned** — the third prompt-echo sighting in the campaign and the third on a width child. Only summarisation carries content. | `reports/matrix-campaign-20260810/wave26/human-review/READ-V20W896.txt` |
| 6 | `ARCH.CL896` | **0.688431** | **4/15** | standard | reproduces the supplier summary with the 11% figure, writes a real if apologetic email, names the four missing quantities -- and restates the prompt verbatim wherever ordering or arithmetic is asked for; 3 of 16 lanes run to the cap | `reports/matrix-campaign-20260810/wave15/human-review/READ-CL896.txt` |
| 7 | `ARCH.W736` | **0.689220** | **1/15** | standard | one case: it acknowledges the multi-turn correction and restates the corrected quantity. Everything else is a loop or empty, and the summarisation lane every other living width child holds is gone | `reports/matrix-campaign-20260810/wave21/human-review/READ-W736.txt` |
| 8 | `ARCH.V8X13GL01` | **0.691832** | **7/15** | standard | seven of fifteen at 0.691832x, the deepest seven in the campaign. It greets a person properly, writes the email with all three facts in thirty words, orders four runners correctly where the PARENT does not, and names the closed gate, the decision, the risk and the four missing data points. It cannot do arithmetic, cannot write the deduplication function and cannot finish the lighthouse. | `reports/matrix-campaign-20260810/wave24/human-review/READ-V8X13GL01.txt` |
| 8= | `ARCH.V8X13GL03` | **0.691832** | **7/15** | standard | **wave 25.** The same child at a lambda the solver's fixed ladder does not contain (0.3, between the two rungs that read 7 and 6). Same seven at the same 158,909,184 parameters, with **three lanes answered in full instead of two**: the greeting answers the person and asks exactly one question, the email carries all three named items inside a salutation and a sign-off and *requests* the tracking number, and three distinct detailed actions land. | `reports/matrix-campaign-20260810/wave25/human-review/READ-V8X13GL03.txt` |
| 8= | `ARCH.V8X13` | **0.691832** | **6/15** | standard | the 8,192-row coreset on the 13/15 body. Everything structural survives -- Saturday, 4:15 p.m., decision/risk/mitigation, the four missing data points, and an email that carries all three named items -- and everything requiring a computation is gone. Composition ran the OTHER way here: the same excision costs the 16,384 rung four cases and buys this one two. | `reports/matrix-campaign-20260810/wave24/human-review/READ-V8X13.txt` |
| 8= | `ARCH.V8X13GL1` | **0.691832** | **6/15** | standard | the mildest rung on the deep base: six lanes held, none in full. The word-substitution signature is thinner here than anywhere else in the wave (one cap hit of sixteen) and the content is correspondingly flatter. | `reports/matrix-campaign-20260810/wave24/human-review/READ-V8X13GL1.txt` |
| 8= | `ARCH.V8X13G` | **0.691832** | **4/15** | standard | the terminal-read solve at the lambda its own out-of-sample criterion picked, and the picking is the failure: it trades three held lanes for two Useful ones, including the only greeting in the wave that answers the person AND asks its question, and the only finish order in the wave that is correct. | `reports/matrix-campaign-20260810/wave24/human-review/READ-V8X13G.txt` |
| 9 | `ARCH.V20X13X07` | **0.694093** | **5/15** | standard | **wave 26.** The 20,480-row coreset on the block 13 AND block 7 double excision: five lanes held, **none in full**, four of sixteen at the cap. V20 alone reads 13 and the double excision alone read 6; composed they read 5. The third axis is fatal on the best base the campaign has. | `reports/matrix-campaign-20260810/wave26/human-review/READ-V20X13X07.txt` |
| 10 | `COMP.MA.MXO4` | **0.703396** | **5/15** | parent-shaped | five partials and no useful; the degeneracy is no longer confined to one lane, which is where wave 11's ladder stopped | `not packaged (pre-wave-15)` |
| 11 | `COMP.MA.MXP2` | **0.703409** | **6/15** | parent-shaped | the same depth as MXO4 bought by shopping the lane instead of taking it in depth order: one more case, half the lambada perplexity, still no useful | `not packaged (pre-wave-15)` |
| 12 | `ARCH.V12X13` | **0.710093** | **7/15** | standard | **wave 25.** K = 12,288 on the excised body, zero solve, zero cap hits. The retrieval lane lands in full — Saturday, 4:15 p.m., the north gate closed, avoid it — and six more are held in part. Built as a diagnostic and it earned a row: it is the deepest STANDARD-shape seven that is not the 0.691832x cluster. | `reports/matrix-campaign-20260810/wave25/human-review/READ-V12X13.txt` |
| 13 | `COMP.MA.MXQ5` | **0.710270** | **7/15** | parent-shaped | the healthiest deep child in the campaign -- two Useful gradings and one lane at the cap -- and it costs 1.58M added repair parameters charged in full | `not packaged (pre-wave-15)` |
| 14 | `COMP.MA.MXP1` | **0.712544** | **7/15** | parent-shaped | repair on the autopilot child: two cases bought for 2.1M added parameters, and no benchmark row exists because forward-hook routes cannot be written into a state_dict | `not packaged (pre-wave-15)` |
| 15 | `ARCH.W768` | **0.722610** | **3/15** | standard | the summarisation lane names the decision, the risk and the mitigation with the supplier's name corrupted to "Supper B"; eleven of sixteen lanes are loops or empty | `reports/matrix-campaign-20260810/wave19/human-review/READ-W768.txt` |
| 15= | `ARCH.W768LS` | **0.722610** | **3/15** | standard | the same three cases as its control on a DIFFERENT set: it recovers context-retrieval, the width axis's signature zero, and gives up uncertainty | `reports/matrix-campaign-20260810/wave20/human-review/READ-W768LS.txt` |


## The three numbers that qualify every row above

1. **`standard` and `parent-shaped` are different claims.** A `standard` child
   loads as a stock `Lfm2ForCausalLM` with no `trust_remote_code` and the loader
   counts the ratio shown. A `parent-shaped` child's ratio is a **storage-format**
   ratio: the model on disk is 229,693,184 parameters and reconstructs the
   parent's shapes to run. Both counts are honest under their own storage model;
   they are not the same quantity. Ranks 6–9 are parent-shaped; every other row on this page is standard.
2. **Depth and score do not co-rank, and ranks 1–3 are the sharpest case yet.**
   Rank 1 is 22 points deeper than rank 4 and three cases worse; rank 15 is four
   points shallower than rank 4 and one case better. A single number cannot
   order this page and it is not meant to.
3. **The deepest child ever built is `ARCH.W512` at 0.463481x** — under half the
   parent, standard shape, loads and counts — and it is **0/15**, eight of
   sixteen lanes empty. It is not on this page because it answers nothing. Rank
   1 is 0.469305x, five thousandths shallower, and holds one case: that gap is
   the whole of what "coherent at half the parent" is worth so far.

## What the whole campaign bought, in one line each

* **The deepest child that reads at the parent's own band:** `COMP.MA.MXN2`,
  **0.780518x at 10/15** — the token interface alone, nothing in the body folded.
* **The deepest child in the ADDENDUM 8 §8.2 success band (6–7/15):**
  `COMP.MA.MXP2` at **0.703409x, 6/15**.
* **The deepest scoring child of any kind:** `ARCH.V16CL896`, **0.469305x,
  1/15**, standard shape (wave 23). The deepest at **four or more** cases is
  still `ARCH.CL896`, **0.688431x, 4/15**.
* **The deepest STANDARD-SHAPE child at 8/15 or better:** `ARCH.V20X13G03`,
  **0.746614x, 11/15** (wave 26) — sixteen thousandths deeper than the previous
  holder `ARCH.V12` at the same review, stock loader, one 1,024-entry diagonal
  re-solved and nothing else moved.
* **The highest-scoring child of any kind:** a three-way tie at **13/15**, one
  case above the parent's own sealed 12/15 on the same surface — `ARCH.V18`
  (**0.790005x**, wave 26), `ARCH.V20` (0.799135x, wave 25) and `ARCH.V22`
  (0.808265x, wave 26). The deepest of the three is `ARCH.V18`.
* **The deepest child at 10/15 or better:** `ARCH.V20X13G03`, **0.746614x,
  11/15** (wave 26); the deepest **ten** is its own zero-solve control
  `ARCH.V20X13` at the same 0.746614x. Both are 5.5 points deeper than the
  previous holder at ten.
* **The deepest child at 7/15 or better: unchanged by wave 26.** `ARCH.V8X13GL01`
  and `ARCH.V8X13GL03` still hold it at **0.691832x, 7/15**. Wave 26's deepest
  child below that line reads 5.
* **The deepest child of any score:** `ARCH.W512`, **0.463481x, 0/15**.
* **0.5x with a coherent child: not reached.** Two independent 0.5x children in
  two lane orders, with and without repair, plus a third by a different axis, all
  0/15.
