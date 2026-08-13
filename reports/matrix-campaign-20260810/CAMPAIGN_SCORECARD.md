# Campaign scorecard — one page, for scoring

**Campaign:** stage-1 matrix, waves 1–24, `reports/matrix-campaign-20260810/`.
**Parent:** `LFM2.5-230M`, **229,693,184 parameters**, sealed **12/15** on the
frozen 15-prompt review (parent lane `865f006e3813712e…`, identical on every
dossier in every wave).
**Ledger:** 682 rows, sha `5d378f85f94cab9eb55c9f46aab7e01a2c52a36a3504690dd4d911612603f1ec`.
**Reviewed whole-model children:** 172. **Scoring at least 1 of 15:** 124.

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
| 3 | `ARCH.V16W896` | **0.667288** | **2/15** | standard | the multiplying pair. Summarisation survives intact, the uncertainty lane names all four data points and then degenerates, two lanes return the empty string and retrieval returns a bare dash. | `reports/matrix-campaign-20260810/wave23/human-review/READ-V16W896.txt` |
| 4 | `ARCH.CL896` | **0.688431** | **4/15** | standard | reproduces the supplier summary with the 11% figure, writes a real if apologetic email, names the four missing quantities -- and restates the prompt verbatim wherever ordering or arithmetic is asked for; 3 of 16 lanes run to the cap | `reports/matrix-campaign-20260810/wave15/human-review/READ-CL896.txt` |
| 5 | `ARCH.W736` | **0.689220** | **1/15** | standard | one case: it acknowledges the multi-turn correction and restates the corrected quantity. Everything else is a loop or empty, and the summarisation lane every other living width child holds is gone | `reports/matrix-campaign-20260810/wave21/human-review/READ-W736.txt` |
| 6 | `ARCH.V8X13GL01` | **0.691832** | **7/15** | standard | seven of fifteen at 0.691832x, the deepest seven in the campaign. It greets a person properly, writes the email with all three facts in thirty words, orders four runners correctly where the PARENT does not, and names the closed gate, the decision, the risk and the four missing data points. It cannot do arithmetic, cannot write the deduplication function and cannot finish the lighthouse. | `reports/matrix-campaign-20260810/wave24/human-review/READ-V8X13GL01.txt` |
| 7 | `ARCH.V8X13` | **0.691832** | **6/15** | standard | the 8,192-row coreset on the 13/15 body. Everything structural survives -- Saturday, 4:15 p.m., decision/risk/mitigation, the four missing data points, and an email that carries all three named items -- and everything requiring a computation is gone. Composition ran the OTHER way here: the same excision costs the 16,384 rung four cases and buys this one two. | `reports/matrix-campaign-20260810/wave24/human-review/READ-V8X13.txt` |
| 8 | `ARCH.V8X13GL1` | **0.691832** | **6/15** | standard | the mildest rung on the deep base: six lanes held, none in full. The word-substitution signature is thinner here than anywhere else in the wave (one cap hit of sixteen) and the content is correspondingly flatter. | `reports/matrix-campaign-20260810/wave24/human-review/READ-V8X13GL1.txt` |
| 9 | `ARCH.V8X13G` | **0.691832** | **4/15** | standard | the terminal-read solve at the lambda its own out-of-sample criterion picked, and the picking is the failure: it trades three held lanes for two Useful ones, including the only greeting in the wave that answers the person AND asks its question, and the only finish order in the wave that is correct. | `reports/matrix-campaign-20260810/wave24/human-review/READ-V8X13G.txt` |
| 10 | `COMP.MA.MXO4` | **0.703396** | **5/15** | parent-shaped | five partials and no useful; the degeneracy is no longer confined to one lane, which is where wave 11's ladder stopped | `not packaged (pre-wave-15)` |
| 11 | `COMP.MA.MXP2` | **0.703409** | **6/15** | parent-shaped | the same depth as MXO4 bought by shopping the lane instead of taking it in depth order: one more case, half the lambada perplexity, still no useful | `not packaged (pre-wave-15)` |
| 12 | `COMP.MA.MXQ5` | **0.710270** | **7/15** | parent-shaped | the healthiest deep child in the campaign -- two Useful gradings and one lane at the cap -- and it costs 1.58M added repair parameters charged in full | `not packaged (pre-wave-15)` |
| 13 | `COMP.MA.MXP1` | **0.712544** | **7/15** | parent-shaped | repair on the autopilot child: two cases bought for 2.1M added parameters, and no benchmark row exists because forward-hook routes cannot be written into a state_dict | `not packaged (pre-wave-15)` |
| 14 | `ARCH.W768` | **0.722610** | **3/15** | standard | the summarisation lane names the decision, the risk and the mitigation with the supplier's name corrupted to "Supper B"; eleven of sixteen lanes are loops or empty | `reports/matrix-campaign-20260810/wave19/human-review/READ-W768.txt` |
| 15 | `ARCH.W768LS` | **0.722610** | **3/15** | standard | the same three cases as its control on a DIFFERENT set: it recovers context-retrieval, the width axis's signature zero, and gives up uncertainty | `reports/matrix-campaign-20260810/wave20/human-review/READ-W768LS.txt` |

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
* **The deepest STANDARD-SHAPE child at 8/15 or better:** `ARCH.V16`,
  **0.780874x, 9/15** — seventeen points deeper than the previous holder
  (`ARCH.X13A`, 0.947479x), and at the same ratio as `MXN2` with one case less
  and a stock loader instead of a reconstituted `lm_head`.
* **The deepest child of any score:** `ARCH.W512`, **0.463481x, 0/15**.
* **0.5x with a coherent child: not reached.** Two independent 0.5x children in
  two lane orders, with and without repair, plus a third by a different axis, all
  0/15.
