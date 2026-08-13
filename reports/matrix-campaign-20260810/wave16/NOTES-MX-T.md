# NOTES-MX-T — wave 16: the ladder died at rung 0, and the block the campaign had been deleting was the wrong one

Wave 16, branch `codex/lrc-blank-compression`, source commits `9a7c391b`
(ADDENDUM 13 + the wave-16 builder), `720b597a` and `35dfe3a9` (two fixes found
by running it), on top of wave-15 HEAD `7b91fb99`. Deployment
`neural-forge-matrix-35dfe3a9`, cut with `git archive -c core.autocrlf=false`
from this wave's own commit; review surface verified on extraction and on every
review before a token was generated: protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`, prompt pack
`0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`.

**Two calibrations passed before anything was graded.** The fresh conditioned
capture reproduces the sealed collector's rows to wave 15's numbers exactly
(L00 relL2 3.37e-3 cos 0.9999944, L05 1.65e-2 / 0.9997207, L13 2.03e-2 /
0.9997842). The parent's sixteen frozen-review responses digest to
**`865f006e3813712ec202cb23fb2bfdc43890c0b8087fd44989cde71583eaaa70`**, byte-
identical to every wave since 7. And the parent's own benchmark row reproduces
the sealed reference to four decimals in all seven columns.

The wave was ordered as a succession ladder from wave 15's two living pieces.
Rung 0 is dead. What the wave found instead is bigger than the ladder was.

*§0's sentences are at the bottom.*

---

## 1. The ladder, and where it stopped: rung 0

Rung 0 is `intermediate_size` 1280 (`ARCH.W14FF1280`, **5/15**) plus block 5
excised with a full-width local re-solve (`ARCH.X05A`, **6/15**). Both pieces
were sealed alive-band-adjacent by wave 15 and neither had ever been stacked on
the other.

```
ARCH.R0   ff1280 + X05                       166,511,360 = 0.724929x   0/15
ARCH.R1   + block 13                         158,379,776 = 0.689528x   0/15
ARCH.R2   + block 12                         151,299,712 = 0.658704x   0/15
ARCH.R3   + head merge 8->6 KV groups         147,367,552 = 0.641584x   0/15
ARCH.R4   + intermediate_size 1024           138,716,800 = 0.603922x   0/15
ARCH.R5   + block 8                          133,209,600 = 0.579946x   0/15
```

Every count is the loader's own; every child loads as a stock `Lfm2ForCausalLM`
with `trust_remote_code: false` under `transformers` 5.12.1, watched by the
sealer rather than asserted.

**R0's per-site write cosines are 0.894 to 0.988, median 0.958** — far above the
0.856/0.891 cliff wave 15 bracketed — and it answers nothing. Eight of fifteen
graded lanes cycle a token to the 1024-token cap (`and`, `1.`, `The.`, a double
quote, a colon, `1. Start with a small.`), `multi-turn-correction` returns the
empty string, and the four lanes that terminate cleanly read their own prompt
back. Below R1 the children stop differing: R2 through R5 are sixteen-of-sixteen
cap hits on subword fragments (`vous`, `ables`, `fore`, `date_`, `ation`).

The ladder was built in full and read in full because the standing rule is that
censuses order attempts and gate nothing. Nothing below rung 0 taught anything
the rung above it had not.

## 2. The bisection, and the measurement that turned the wave

R0 is dead and both its pieces are alive, so the bisection asks **which** piece
the stack cannot carry. Two children, one moving the width and one moving the
block:

| child | recipe | net | score |
|---|---|---:|---:|
| `ARCH.R0` | ff1280 + block **5** | 0.724929x | **0/15** |
| `ARCH.BX13` | ff1280 + block **13** | **0.724929x** | **1/15** |
| `ARCH.B1536` | ff**1536** + block 5 | 0.769439x | **0/15** |

Giving the SwiGLU 20% of its channels back does not recover anything, and moving
the excision to a different block does not either. So the control the bisection
needed was the block alone — and that control is the wave:

```
ARCH.X05A  block  5 excised alone, 0.947479x   6/15   (wave 15)
ARCH.X13A  block 13 excised alone, 0.947479x  13/15   (this wave)
```

**Same operation. Same parameter count. Same solver, same evidence, same frozen
surface. 6/15 against 13/15 — and 13/15 is above the parent's own 12/15.**

`ARCH.X13A` is useful on four lanes (a real follow-up question; three household
actions; the correct first-seen-order dedup function; an email carrying Friday,
Monday and the tracking request) and partial on nine. Its two charged lanes are
`explanation-seasons` and `multi-turn-correction` — **the same two the parent
itself fails**, the parent attributing seasons to an elliptical orbit and
recommending peanut butter bars to the peanut-allergy correction.

Wave 14's F-EXCISE census ranked the blocks by the held-out fraction of a
deleted block's contribution that the upstream writer could absorb, and put
**L05 first at +0.665** with L13 second at +0.467. Wave 15 already retired that
number as a statement about a *trajectory* objective, which it then priced at
total collapse. Wave 16 shows the ranking was not merely unsupported: **it was
inverted at the top**, and every child in the R-ladder was built on the worst
available block.

## 3. The behavioural excision census, which is the ranking the campaign needed

Eight single-block excisions, each built alone with every surviving FFN writer
re-solved at unchanged width, each sealed and each read on the frozen surface.

| block | kind | net | score | per-child verdict | terminal relL2 |
|---:|---|---:|---:|---|---:|
| **13** | conv | 0.947479x | **13/15** | dead *(per-lane alive-parent-band)* | 0.6755 |
| **7** | conv | 0.947479x | **11/15** | **alive-parent-band** | 0.3864 |
| **9** | conv | 0.947479x | **10/15** | **alive-parent-band** | 0.4219 |
| **1** | conv | 0.947479x | 9/15 | dead *(per-lane alive-degraded)* | 0.5871 |
| **11** | conv | 0.947479x | 9/15 | **alive-degraded** | 0.5556 |
| **5** | conv | 0.947479x | 6/15 | dead | 0.5707 *(wave 15)* |
| **3** | conv | 0.947479x | 6/15 | **impaired** | 0.5470 |
| **12** | attention | 0.952057x | **0/15** | dead | 0.7883 |

Three readings:

1. **Every conv block is cheap and the attention block is not.** `ARCH.X12`
   removes the *smallest* number of parameters in the wave (4.6M against the conv
   blocks' 12.1M) and is the only single-block child that is dead. Its failure is
   qualitatively different too: it answers the English greeting in **Japanese**,
   emits a literal `tool_call_start` token with a JSON payload and a `think` tag,
   and mixes four languages inside single sentences. One of six attention blocks
   carries something none of the eight conv blocks does.
2. **`ARCH.X07` and `ARCH.X09` are the first standard-shape children in this
   campaign to seal `alive-parent-band` on the per-child reading**, and `ARCH.X11`
   the first `alive-degraded`. `ARCH.X09` gives the cleanest `context-retrieval`
   answer of any child or the parent — three facts, three lines, all correct,
   where the parent answers two. `ARCH.X07` names **axial tilt** on
   `explanation-seasons`, which the parent does not.
3. **Terminal drift does not rank them.** X13A has the *second worst* drift of the
   eight (0.6755) and the *best* behaviour; X03 has the third *best* drift
   (0.5470) and ties for worst among the living. Drift separates dead from alive
   at the top of the range (0.79 is dead, everything ≤ 0.68 has a pulse) and
   orders nothing inside it. Per-site write cosine ranks them not at all: the
   excision children have no scored write site, and where they do — the whole
   R-ladder — the cosines are flat at 0.87-0.99 across children scoring 13/15 and
   0/15.

## 4. Depth is the cleanest axis and it still stops at one block

Two blocks, both from the top of the behavioural census, no other axis moved:

```
ARCH.D2   blocks 13 + 11   205,565,696 = 0.894958x   2/15
ARCH.D2B  blocks 13 +  9   205,565,696 = 0.894958x   6/15
ARCH.D3   blocks 13 + 11 + 9  193,501,952 = 0.842437x   0/15
```

**The pair matters and the count still dominates.** 13+9 keeps six lanes
partially working and 13+11 keeps two, from parents that read 13/15, 10/15 and
9/15 alone. Three blocks is a model that emits runs of asterisks in every lane.

D2's failure mode is new to the campaign and worth naming: markdown emphasis
markers multiplying into subword fragments (`*s*s*s`, `**e**e**e`,
`**(1)**(1)**`) rather than a whole word repeating — which is also why the
mechanical collapse in the reading tool under-reports it, since that collapse
works on whitespace-separated words.

## 5. The head-count merge, run in standard form for the first time

`--kv-groups 6` merges the parent's 8 KV groups in pairs to 6, taking
`num_attention_heads` 16 → 12 with **every surviving head keeping the parent's
`head_dim` 64** and therefore the parent's own RoPE ladder. Three coupling facts
were read off `transformers` 5.12.1 and enforced rather than assumed:
`head_dim` is absent from this parent's config and falls back to
`hidden_size // num_attention_heads`, so 12 heads without an explicit
`head_dim: 64` silently rebuilds every attention block at **85**;
`num_attention_heads` must be an integer multiple of `num_key_value_heads`, so
12 query heads is legal only with 6, 4, 3, 2 or 1 KV heads; the head count
itself is inferred from tensor shape.

**The gauge is forced and smaller than it looks.** RoPE admits per-plane
rotation-scalings; the shared per-head RMSNorm weight admits only maps commuting
with a diagonal — and that weight is measurably **not** plane-symmetric on this
parent (`|w_p − w_{p+32}|` is 0.11–0.40 of `‖w‖` across the twelve q/k norms).
What survives both is the **per-plane sign** `s_p = ±1`: a scalar, so it commutes
with the rotation and with the diagonal, preserves each plane's norm and hence
the head's RMS, and leaves `(s q)^T R_Δ (s k) = q^T R_Δ k` exactly. The builder
records what the inadmissible continuous rotation would have bought beside every
pair it declined to use.

```
ARCH.KV6      merge alone            224,974,592 = 0.979457x   5/15
ARCH.X13KV6   block 13 + merge       212,910,848 = 0.926936x   6/15
```

Per-site attention-output cosine after the `out_proj` re-solve is 0.886–0.956,
median 0.938; per-layer pair redundancy under the sign gauge is 0.45–0.76, so
the heads genuinely are partly redundant. The merge is **per-lane impaired**, so
it stays in the deck — and it is expensive: 2% of the parameters for seven of
fifteen cases. Stacked on block 13 it composes *down* (13/15 + 5/15 → 6/15), the
same direction wave 15's two living pieces composed, but not to zero.

## 6. The frontier, and the distance to 0.5x

The deepest standard-shape child this wave with a non-zero behavioural score is

```
ARCH.W1536X13 = intermediate_size 1536 + block 13 excised
                176,734,976 parameters = 0.769439x   4/15   per-lane impaired
```

and its control is `ARCH.B1536`, the **same width with block 5 instead of 13** at
the **same 0.769439x**, which reads **0/15**. `W1536X13` gives the best answer any
child or the parent produced on `uncertainty-insufficient-data`: *"Given that you
have not provided details such as units, sample size, prices, or other
information, I cannot complete the specifics for you"* — the parent asserts it
*can* safely conclude.

The deepest child that is **alive on the per-child reading** is `ARCH.X07` at
**0.947479x, 11/15**.

**Distance to 0.5x.** 0.5x is 114,846,592 parameters. The tied embedding is
67,108,864 = 0.2922x untouched by every axis in the R3-standard set, so 0.5x
needs the 162.58M body to fall to 47.7M — a 3.4x fold of the body, wave 15's
arithmetic, unchanged. From the frontier at 0.769439x that is another
61.9M parameters. **It stops at 0.769439x, and the measured reason is
behavioural and specific: every axis that removes real body mass either dies
alone (`intermediate_size` ≤ 896, `head_dim`, attention-block excision) or dies
in composition at the second step (two conv excisions, or one excision plus any
global SwiGLU narrowing).** The only composition that survived at all this wave
kept one block out and left everything else at parent width.

## 7. Benchmarks — and the review and the benchmarks disagree

Every child that does not loop was benchmarked; lm-eval, `limit=200 seed=1234
batch_size=4 max_length=2048 dtype=float32`, on an SM86 RTX 3090.

| model | net | review | arc_e | hella_n | piqa_n | wino | lambada | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | 81.80 |
| `ARCH.X13A` | 0.947479 | **13/15** | 0.4600 | 0.4050 | 0.6350 | 0.5200 | 0.3350 | **106.78** | 297.09 |
| `ARCH.KV6` | 0.979457 | 5/15 | 0.4650 | 0.4350 | 0.6050 | 0.5100 | 0.2150 | 107.87 | 201.05 |

The parent row reproduces the sealed reference in all seven columns, so the lane
is calibrated. And then it says something the 15-prompt review does not:
**`ARCH.X13A` reads one case above the parent and is 4.4x worse in lambada
perplexity, 3.6x worse in wikitext, and down 0.085 to 0.125 absolute on four
multiple-choice tasks.** Removing one conv block of fourteen costs 5.25% of the
parameters and roughly a quarter of the model's measurable language modelling.

Both numbers are real and they are measuring different things — the review reads
whether a whole answer is usable, the benchmarks read the distribution. The
honest statement of the frontier is *both*: `ARCH.X13A` answers thirteen of
fifteen questions usefully and is a materially worse language model than its
parent. A frontier claim quoting only the first would be the same category of
error ADDENDUM 8 was written to stop.

## 8. Estate

* **Commits.** `9a7c391b` (ADDENDUM 13, the wave-16 builder with the head-count
  axis, the sealer's one new accepted schema string), `720b597a` and `35dfe3a9`
  (two defects found by running it), plus the sealing commit for this note.
  Nothing pushed, nothing rebased, staged by explicit path only; the Codex lane's
  ~331 uncommitted paths, including the frozen
  `src/neural_forge/hf_lfm2_circuit_trace_cache.py`, were not touched.
* **Ledger.** **483 → 525**, sha
  `d148283bb059560f6e19e1096440faa0792eaf5dac1f670275e7a76454c57382`. Backed up
  first (`backup/ledger.jsonl.pre-wave16-20260813T020226`, verified by sha **and**
  `cmp`), merged exactly once behind `wave16/.MERGED_ONCE`, byte-prefix verified
  against the backup afterwards, and the committed object verified byte-identical
  with `git cat-file`. Forty-two rows: 21 ARCH run rows and 21 ADDENDUM 6 verdict
  rows, no superseded rows dropped.
* **Seals.** `run/mx16/mx16_verify_seals.py` with `wave16/cells` added and the
  deployment lifted out of the file into an argument (a verifier pinned to a
  previous wave's deployment either breaks on a prune or verifies this wave's
  seals with last wave's schema code): **12 roots present, 0 undeclared; 525
  rows, 521 resolve, 521 verify, 0 fail, 4 do not resolve** — the same four
  pre-existing non-resolvers waves 14 and 15 named.
* **Disk.** Server `/` began at 14 GB free and ended at 13 GB. Deployments held
  at current+1 throughout: `63e95cd8` was pruned under a manifest
  (`wave16/deployments-prune-manifest-wave16.json`, 2,667 files, 166,042,082
  bytes) when this wave's first deployment landed, and each superseding wave-16
  deployment replaced its predecessor. Twenty-one children were built; the
  **best five by sealed score plus the deepest non-zero child** are kept
  (`MX16-X13A`, `MX16-X07`, `MX16-X09`, `MX16-X11`, `MX16-W1536X13`) and the
  other sixteen are deleted and named with their sealed tensor digests in
  `wave16/children-retention-manifest-wave16.json`, freeing 5.95 GB. The wave's
  `/mnt/ai_data` footprint is 2.0 GB.
* **Human-review package, in the repository this time.**
  `reports/matrix-campaign-20260810/wave16/human-review/` — 147 files: every
  dossier, every child and parent response rendered verbatim, every run receipt,
  every verdict receipt, every per-case reading and every build receipt.

## 9. What the wave did not spend, and two tooling limits it hit

* **Unrun deck probes.** XGRAM retrospective calibration, the gap-4 per-case
  attribution artifact, the CBT identity-permutation cell and the F-LANDMARK
  census were all budgeted *after* the assembly ladder and the ladder consumed
  the budget. None was killed; none was started.
* **Unrun on the new lead.** The behavioural census covered eight of fourteen
  blocks. Blocks 0, 2, 4, 6, 8 and 10 were not built — four of those are the
  remaining attention blocks, and `ARCH.X12` makes them the interesting ones. A
  *joint* score-space solve for a merged head pair (T7's two-sided Stein ridge)
  was not built either; this wave's merge is average-in-gauge plus an `out_proj`
  re-solve, and the notes say so where the kernel is defined.
* **The degeneracy clause cannot charge the safety lane.** Charged lanes come
  from the per-case readings, which cover the fifteen graded cases; the safety
  lane is graded in its own field. A child degenerate on all sixteen lanes
  therefore seals `lane-scoped (15 of 16)` where the honest reading is `general`.
  `ARCH.R2`, `R3`, `R4`, `R5`, `D3` and `X12` are all in that position.
* **A child with zero charged lanes seals `unstated`, not `none`.** There is no
  CLI way to declare "no lane is charged", so the campaign's four cleanest
  children (`X07`, `X09`, `X11`, `X03`) carry no per-lane verdict at all. Their
  per-child verdict is the stricter reading and it is the one quoted above, so
  nothing is over-claimed — but the tooling makes the good case unexpressible.

## 0. The sentences

1. **The ladder died at rung 0.** Wave 15's two living standard-shape pieces —
   `intermediate_size` 1280 at 5/15 and block-5 excision at 6/15 — compose to
   **0/15** at 0.724929x, with per-site write cosines of 0.894 to 0.988 that say
   nothing was wrong with any individual solve. Widening the SwiGLU back to 1536
   recovers nothing (0/15) and moving the excision to another block recovers
   almost nothing (1/15).
2. **The campaign had been deleting the wrong block for two waves.** Block 13
   excised alone reads **13/15** where block 5 excised alone reads **6/15** at the
   identical 0.947479x, identical operation, identical evidence. Wave 14's
   absorbed-fraction census ranked block 5 first and block 13 second; at the only
   level that decides anything, that ranking is inverted at the top.
3. **Every conv block is cheap; the attention block is not.** Six of seven conv
   excisions score 6/15 to 13/15 and three of them seal per-child **alive**;
   excising attention block 12 — the *smallest* bite in the wave at 4.6M
   parameters — scores **0/15** and answers an English greeting in Japanese with
   a tool-call token in it.
4. **Depth still stops at one block, and which two you pick matters.** 13+11 is
   2/15 and 13+9 is 6/15 at the same 0.894958x, from parents that read 13, 9 and
   10 of 15 alone; three blocks is 0/15.
5. **The standard-form head merge runs, and RoPE plus a shared per-head RMSNorm
   leave exactly one exact gauge: the per-plane sign.** 16 → 12 query heads at
   full `head_dim` is **5/15 at 0.979457x** — per-lane impaired, so it stays in
   the deck, at 2% of the parameters for seven of fifteen cases.
6. **The frontier is `ARCH.W1536X13` at 0.769439x, 4/15**, whose same-size
   control with block 5 instead of 13 is 0/15; the deepest per-child **alive**
   child is `ARCH.X07` at 0.947479x, 11/15. 0.5x needs the body to fall 3.4x with
   the embedding untouched, and every axis that would do it dies alone or dies at
   the second composition step.
7. **The review and the benchmarks disagree about the best child, and both are
   right.** `ARCH.X13A` reads one case *above* the parent on the frozen surface
   and is **4.4x worse in lambada perplexity** (24.39 → 106.78) and down 0.085 to
   0.125 absolute on four multiple-choice tasks. One conv block of fourteen costs
   5.25% of the parameters and about a quarter of the measurable language
   modelling.
