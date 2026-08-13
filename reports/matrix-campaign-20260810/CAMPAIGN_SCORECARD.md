# Campaign scorecard — one page, for scoring

**Campaign:** stage-1 matrix, waves 1–22, `reports/matrix-campaign-20260810/`.
**Parent:** `LFM2.5-230M`, **229,693,184 parameters**, sealed **12/15** on the
frozen 15-prompt review (parent lane `865f006e3813712e…`, identical on every
dossier in every wave).
**Ledger:** 652 rows, sha `33a8326ae956e15aed0f1b2443a25377351d167b0bbb78667f5a0d8a0f74fe29`.
**Reviewed whole-model children:** 157. **Scoring at least 1 of 15:** 110.

**Wave 22 changed nothing on this page and that is a result.** Its three
children score 2/15 at 0.859023x, 1/15 at 0.841047x and 0/15 at 0.672632x; the
first two are too shallow to rank and the third answers nothing, so the fifteen
rows below are byte-identical to wave 21's.

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
| 1 | `ARCH.CL896` | **0.688431** | **4/15** | standard | reproduces the supplier summary with the 11% figure, writes a real if apologetic email, names the four missing quantities -- and restates the prompt verbatim wherever ordering or arithmetic is asked for; 3 of 16 lanes run to the cap | `reports/matrix-campaign-20260810/wave15/human-review/READ-CL896.txt` |
| 2 | `ARCH.W736` | **0.689220** | **1/15** | standard | one case: it acknowledges the multi-turn correction and restates the corrected quantity. Everything else is a loop or empty, and the summarisation lane every other living width child holds is gone | `reports/matrix-campaign-20260810/wave21/human-review/READ-W736.txt` |
| 3 | `COMP.MA.MXO4` | **0.703396** | **5/15** | parent-shaped | five partials and no useful; the degeneracy is no longer confined to one lane, which is where wave 11's ladder stopped | `not packaged (pre-wave-15)` |
| 4 | `COMP.MA.MXP2` | **0.703409** | **6/15** | parent-shaped | the same depth as MXO4 bought by shopping the lane instead of taking it in depth order: one more case, half the lambada perplexity, still no useful | `not packaged (pre-wave-15)` |
| 5 | `COMP.MA.MXQ5` | **0.710270** | **7/15** | parent-shaped | the healthiest deep child in the campaign -- two Useful gradings and one lane at the cap -- and it costs 1.58M added repair parameters charged in full | `not packaged (pre-wave-15)` |
| 6 | `COMP.MA.MXP1` | **0.712544** | **7/15** | parent-shaped | repair on the autopilot child: two cases bought for 2.1M added parameters, and no benchmark row exists because forward-hook routes cannot be written into a state_dict | `not packaged (pre-wave-15)` |
| 7 | `ARCH.W768` | **0.722610** | **3/15** | standard | the summarisation lane names the decision, the risk and the mitigation with the supplier's name corrupted to "Supper B"; eleven of sixteen lanes are loops or empty | `reports/matrix-campaign-20260810/wave19/human-review/READ-W768.txt` |
| 8 | `ARCH.W768LS` | **0.722610** | **3/15** | standard | the same three cases as its control on a DIFFERENT set: it recovers context-retrieval, the width axis's signature zero, and gives up uncertainty | `reports/matrix-campaign-20260810/wave20/human-review/READ-W768LS.txt` |
| 9 | `ARCH.W768ST` | **0.722610** | **2/15** | standard | the closed-form fold with nothing re-solved -- two cases against the conditioned child's three, which is what the conditioned solve is worth on this axis | `reports/matrix-campaign-20260810/wave20/human-review/READ-W768ST.txt` |
| 10 | `ARCH.W768EV` | **0.722610** | **1/15** | standard | the evidence-PCA basis: all sixteen lanes fill the cap, and the one case it holds is the planning lane naming backups, DNS, TLS and rollback before it degenerates | `reports/matrix-campaign-20260810/wave20/human-review/READ-W768EV.txt` |
| 11 | `ARCH.BX13` | **0.724929** | **1/15** | standard | intermediate_size 1280 plus the GOOD block excised: 13/15 and 5/15 compose to 1/15, and only two lanes reach the cap, so it is interference and not collapse | `reports/matrix-campaign-20260810/wave16/human-review/READ-BX13.txt` |
| 12 | `COMP.MA.MXO3` | **0.729103** | **8/15** | parent-shaped | eight partials, no useful, and the deepest child in the campaign that still reads as connected prose in most lanes | `not packaged (pre-wave-15)` |
| 13 | `COMP.MA.MXN3` | **0.735962** | **6/15** | parent-shaped | the sparse token tail at a true quarter on wave 8's body: six cases, and the tail mechanism is the only variable against MXM9 | `not packaged (pre-wave-15)` |
| 14 | `COMP.MA.MXM9` | **0.736305** | **4/15** | parent-shaped | retrieves the garden hours and the dog rule verbatim from the context and gives three real household actions; three lanes ride the cap and the summary is gone | `not packaged (pre-wave-15)` |
| 15 | `COMP.MA.MXN1` | **0.744033** | **7/15** | parent-shaped | the rebuilt A9 token interface at a true eighth: seven cases at lower install drift than a child twice its depth | `not packaged (pre-wave-15)` |

## The three numbers that qualify every row above

1. **`standard` and `parent-shaped` are different claims.** A `standard` child
   loads as a stock `Lfm2ForCausalLM` with no `trust_remote_code` and the loader
   counts the ratio shown. A `parent-shaped` child's ratio is a **storage-format**
   ratio: the model on disk is 229,693,184 parameters and reconstructs the
   parent's shapes to run. Both counts are honest under their own storage model;
   they are not the same quantity. Ranks 3–6 and 12–15 are parent-shaped.
2. **Depth and score do not co-rank.** Rank 1 dominates rank 2 outright — deeper
   *and* three cases better. The best-reading child in the whole campaign at any
   depth below 0.78x is rank 12 at 8/15, which is 4 points of net ratio
   shallower than rank 1.
3. **The deepest child ever built is `ARCH.W512` at 0.463481x** — under half the
   parent, standard shape, loads and counts — and it is **0/15**, eight of
   sixteen lanes empty. It is not on this page because it answers nothing.

## What the whole campaign bought, in one line each

* **The deepest child that reads at the parent's own band:** `COMP.MA.MXN2`,
  **0.780518x at 10/15** — the token interface alone, nothing in the body folded.
* **The deepest child in the ADDENDUM 8 §8.2 success band (6–7/15):**
  `COMP.MA.MXP2` at **0.703409x, 6/15**.
* **The deepest scoring child of any kind:** `ARCH.CL896`, **0.688431x, 4/15**.
* **The deepest child of any score:** `ARCH.W512`, **0.463481x, 0/15**.
* **0.5x with a coherent child: not reached.** Two independent 0.5x children in
  two lane orders, with and without repair, plus a third by a different axis, all
  0/15.
