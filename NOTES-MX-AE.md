# NOTES-MX-AE — wave 29: the plan is funded, and three of its four cheapest bets are void

Wave 29, branch `codex/lrc-blank-compression`, source commits `badef10b` (the
wave-29 tree, the extension bank, ADDENDUM 16) through `4eda4c16`, plus the
sealing commit for this note. `NOTES-MX-AD` is wave 28; the series is contiguous.

Fifteen days separate wave 28's sealing commit from this one and **nothing moved
in them**: `HEAD` was still `9e600c6f`, the ledger still 804 rows at
`445ff68e6c08c99d…` on both the workstation and the share, the Codex lane's 324
uncommitted paths still uncommitted. The gap was a gap in attention, not in state.

This wave put `docs/INTELLIGENCE_RETENTION_PLAN_2026-08-14.md` into action.
Its first result is that the plan's own cheapest chasm entry does not exist.

## 1. The instrument came first, and it validated — above the chasm

ADDENDUM 15 accepted the 30-prompt extension bank in principle and refused to
fund it. ADDENDUM 16.1 funds it. Thirty prompts, two per sealed capability lane
at two declared difficulty steps, ten of them deliberately off-ladder, authored
from the L12 taxonomy and the fifteen case ids alone. Three mechanical lints
replaced the promise of independence: **the lint refused the first draft** over a
7-gram it shared with the sealed pack (`write a short scene in which a`), and the
digest pin refused the first *run* because the bank had been sealed as CRLF while
the deployment ships the commit's LF. Both refusals did their job before a single
child was graded.

**The retro-score, on ten sealed checkpoints spanning 14/15 down to 0/15:**

| child | sealed 15 | extension 30 | charged of 30 |
|---|---:|---:|---:|
| `V19` | 14 | 12 | **0** |
| `V18` | 13 | 14 | 2 |
| `V19X13G03` | 11 | 13 | 1 |
| `V19X13` | 10 | 11 | 2 |
| `V19X09` | 10 | **5** | 2 |
| `V19X13X09` | 5 | 1 | 13 |
| `V18X13X01G03` | 4 | 5 | 13 |
| `V18CL896` | 2 | 5 | 17 |
| `V16CL896` | 1 | 5 | 14 |
| `V19X13CL896` | 0 | 1 | 25 |

Spearman **0.8023**. The sealed ordering reproduces on a second instrument
authored without access to a single one of the 232 sealed readings. The map this
campaign navigates by is not an artifact of its own fifteen prompts.

**Two things the second surface says that the first cannot.**

* **It splits the tie wave 28 could not.** `V19X13` and `V19X09` both read 10/15
  and wave 28 sealed that as "same landing, different route". On thirty prompts
  they read **11 and 5**. The sealed instrument saw a tie; the extension surface
  sees a six-point gap and says `V19X13` is the better child by a wide margin.
  This is exactly the resolution the bank was funded to buy.
* **The deep zone dissolves.** `V18X13X01G03` (4 sealed), `V18CL896` (2) and
  `V16CL896` (1) all read **5/30** — identical. The extension surface does not
  reproduce the 4 > 2 > 1 ordering at all. So the go/no-go outcome is neither of
  the two the plan pre-registered: the rank order reproduces *and* the
  separations tighten above ~5/15, while the deep-zone fine structure is at or
  below the resolution of **both** instruments.

**The consequence, stated plainly, because it re-prices a sealed claim.** Wave
28's "+1 at 0.4876x, the deepest a solve has ever bought a case" is a one-case
move in the band where neither instrument separates children. It is not
withdrawn — the sealed reading stands — but it may not be leaned on, and any
future deep-zone +1 needs a mechanism, not a count.

The charged-lane count is the sharper depth signal: 0, 2, 1, 2, 2, 13, 13, 17,
14, 25 down the ladder. `V19` answers thirty unfamiliar prompts with **zero**
charged lanes.

## 2. CHASM BRIDGE is void, and the reason is in the base's own receipt

The plan's #3 was its cheapest entry and its only zero-solve one: install the
sealed V19 vocabulary gather on the abandoned parent-shaped body frontier for
`V19×MXO3 = 0.523673x`, `V19×MXP2 = 0.497979x`, `V19×MXQ5 = 0.504840x`.

**The arithmetic is void.** Every one of those bases — MXO2, MXO3, MXP2 and
MXQ5, all four — already carries `C01.T5.S4`, an embedding fold that **owns the
whole 67,108,864-parameter `model.embed_tokens.weight`** and stores it in
16,695,297. The plan subtracted the V19 gather's 47,185,920-parameter saving
*on top of* a fold that had already banked 50,413,567 on the same tensor. The
two mechanisms are not two axes. They are one tensor, and on storage the coreset
is the **worse** of the two: 19,922,944 rows against 16,695,297, an **increase**
of 3,227,647 parameters.

| child | plan's budget | counted-stored, parent-shaped | as the directory on disk |
|---|---:|---:|---:|
| `V19MXO3` | 0.523673x | **0.743155x** | 0.794570x |
| `V19MXP2` | 0.497979x | **0.717461x** | 0.794570x |

Both bridges land *above* their own bases, not below them. The reproduction is
otherwise perfect — 131 of 132 tensors bit-identical to MXO3, only the embedding
changed — and the materialization of MXO3 came out **byte-identical to the one
wave 11 sealed**, fifteen waves later, which is the strongest cross-wave
reproduction this campaign has recorded.

**And the children are dead anyway.** The wave-27 summit `ARCH.V19` reads 14/15
alone. On the MXO3 body — itself 8/15 at 0.729103x — the same vocabulary reads
**3/15**. On MXP2 it reads **2/15**, with six lanes returning a bare token ("The
answer is: 8", "The answer is: Mira", "The answer is: 7") and the debugging lane
returning a function that calls itself.

So #3 fails twice over, on storage and on behaviour, and the second failure is
the L3b×L7 multiplication at its most brutal: fourteen of fifteen composed onto
eight of fifteen gives three.

**A third statement that costs nothing and is worth having:** materialising a
parent-shaped child into a directory stock `transformers` loads **bakes its folds
back to dense shapes**. Its storage saving lives in its factors. That is why the
axis in the transcribed base receipt records `parameters_removed: 0`, and it is
why the two-column frontier is not a presentational choice.

## 3. COMPLEMENT-MERGE: the swap is zero-dimensional, measured five ways

`V19X13` (10/15, holds the dedup LIST) and `V19X13G01` (11/15, gains four lanes,
always loses dedup) differ in exactly 1,024 numbers. Four blends at byte-identical
storage and an identical 170,443,520 parameters:

| child | arm | read |
|---|---|---:|
| `V19X13` | control, α = 0 | 10/15 |
| `ARCH.B25` | α = 0.25 | 10/15 |
| `ARCH.B50` | α = 0.50 | 10/15 |
| `ARCH.B75` | α = 0.75 | **11/15** |
| `V19X13G01` | solved twin, α = 1 | 11/15 |
| `ARCH.BP64` | top-64 \|Δ\| coordinates | 10/15 |
| `ARCH.MASK03` | masked solve, 100 coords held at prior | **9/15** |

**Not one child on that segment holds `coding-deduplicate`.** Every single one —
including α = 0.25, a quarter of the way from a control that accumulates into a
LIST — writes `unique_items = set()`. The union the arm was built to find does
not exist at any interior point. The segment is monotone 10/10/10/11/11 with no
interior point above either endpoint, which is the second half of the
pre-registered falsifier firing as well.

**`BP64` is the sharp one.** Sixty-four coordinates out of 1,024 reproduce the
twin's behaviour — `uncertainty-insufficient-data` goes Useful, dedup goes SET.
Read against `MASK03`, which held the top **hundred** coordinates at the prior and
**still** returned SET, the mechanism is over-determined: a small subset is
*sufficient* to cause the flip and removing that subset is *not sufficient* to
prevent it.

## 4. The masked terminal gain loses twice and recovers nothing

`MASK03` is the plan's #6 built exactly as specified: a mask measured by
`mx29_listset.py` and written down **before** the build that cites it. One
hundred coordinates carrying 51.6% of the solve's update energy held at the prior;
update scale 0.2136 → 0.1485.

It reads **9/15** — below the unmasked twin at eleven and below the zero-solve
control at ten. The dedup lane is still a SET. It loses `instruction-three-actions`
(the same suggestion emitted twice) and `creative-continuity` (one sentence, then
that sentence again verbatim).

The forensic that produced the mask predicted this in one word.

## 5. GAP 3: the LIST→SET swap has been read instead of inferred

Six waves routed around this mechanism and none autopsied it, though the twins
differ in 1,024 numbers sitting on disk. `mx29_listset.py` reads it for zero
builds:

* **1,012 of 1,024 coordinates move.** 95 carry half the change; **402 are needed
  for 90%**. The verdict the script computed for itself is `DISTRIBUTED`, and
  that pre-doomed both the partition arm and the mask before either ran.
* **The flip is at generated token 8**, in the preamble, long before the code:
  the control writes "ensures **first**-seen order", the solved twin writes
  "ensures **order** (first-seen)". Top logits 17.36 against 16.07.
* Attribution is not local either: the top 24 coordinates by contribution carry
  **15.8%** of it.

So the first mechanism-level law for a case swap in this campaign is that there
isn't a local one. The terminal-gain solve does not trade case X for case Y at
some coordinate; it changes the generation trajectory eight tokens in, and the
container the model reaches for downstream follows from that.

## 6. The emission pre-read: margin collapse is NOT what kills the deepest children

The plan's #5 asked whether loops/echo/empty are logit-margin collapse over
surviving discrimination, and pre-registered that a null kills MARGIN-RESTORE for
free. ADDENDUM 16.6 added the mechanism the 2026 literature says to look for.
Six sealed children, pure inference, against the parent:

| child | copy margin | generative margin | terminal entropy | 5-gram repetition Δ | sink mass | massive dims kept |
|---|---:|---:|---:|---:|---:|---:|
| `W720` | 0.757x | **0.320x** | **3.99x** | +0.229 | 0.937 | **0 of 1** |
| `MM768` | 0.877x | **0.330x** | **2.90x** | +0.548 | 0.865 | **0 of 1** |
| `V16CL896` | 0.954x | 0.697x | 1.65x | +0.083 | 0.919 | 1 of 1 |
| `V18CL896` | 0.952x | 0.745x | 1.67x | +0.015 | 0.922 | 1 of 1 |
| `V19X13CL896` | 1.421x | **0.956x** | 1.19x | +0.075 | 0.926 | 1 of 1 |
| `V19X13` (10/15) | 1.289x | 0.965x | 0.82x | 0.000 | 0.983 | 1 of 1 |

**The decisive row is `V19X13CL896`.** It is the deepest and deadest child this
campaign has built — 0/15 at 0.452734x — and its generative margin is **96% of
the parent's**. Margins are intact and it still cannot answer. A moment-matching
terminal recalibration aimed at margin collapse would have had nothing to
correct on the one child it was most wanted for, and MARGIN-RESTORE is therefore
**killed at depth for the price of one forward pass**, exactly as #5 designed.

**What the same pass found instead.** This parent carries exactly **one**
massive-activation dimension. `W720` and `MM768` are the only two children that
**lose it** — and they are precisely the two whose generative margin collapses to
a third of the parent's, whose terminal entropy inflates three- to four-fold, and
whose repetition rate rises most. Every child that keeps the channel keeps its
margins, including the 0/15 one. Sink-head rank correlation degrades in the same
order (0.997 on the living child, 0.71–0.83 on the dying ones).

That is ADDENDUM 16.6's first adopted finding firing on its first measurement,
and it re-aims the repair: the target is the sink and massive-activation
channels on the empty-string class, not a generic diagonal on the deep class.

**One correction to this wave's own instrument, stated rather than hidden.**
`mx29_emission.py` prints `MARGIN INVERSION CONFIRMED` from a difference of two
retained-fraction ratios, and the parent's copy margin is **negative** (−4.92:
teacher-forced, the parent prefers its own continuation to the prompt's next
token). A ratio of negative numbers does not carry the sign the boolean assumes,
so that flag fired on all six children and means nothing. **The table above is
the result; the flag is not.** The underlying measurements are unaffected.

## 7. GLOBAL-KKT: the null is not "no cases", it is "the linearisation is not the model"

A1's scope was confirmed for the all-writer form (ADDENDUM 16.2). The honest
tractable statement of "solve jointly over all folded writer tensors" is a
per-writer output gain solved jointly in the reader metric, with genuine
cross-writer blocks — those off-diagonal blocks being the downstream cross term
M1's autopsy named as the object no per-component objective can see.

Pilot: `V19CL896` at **0.483001x**, whose CL896 fold gives it **20 installed
writers linear under frozen gates** (14 `w2`, 6 `o_proj`) — a genuinely non-empty
free set at the 0.5x band.

| arm | unknowns | λ chosen | held-out relative SSE | rung-0: error ÷ repair |
|---|---:|---:|---:|---:|
| diagonal | 20,480 | 0.1 | 0.9111 | **2.2517** |
| scalar | 20 | 0.1 | 0.9848 | **1.5309** |

The λ→∞ rung was **built, not asserted**: `ARCH`-less `MX29-KKTINF` digests
`0fa5175dda6d58ec…`, byte-identical to its source.

**The objective improves by ten percent and the frozen-gate prediction of what
that improvement does is wrong by more than the improvement itself.** On both
arms. So the wave-29 KKT null is a specific, load-bearing one:

> A joint convex terminal re-solve over every installed writer of a composed
> child, in the reader metric, cannot be evaluated by the frozen-gate linear
> response at 0.48x, because that response's error exceeds the repair it
> predicts. The terminal-metric branch is not closed by a case count here — it
> is closed by its own precondition failing, measured.

`PLAN_GAPS_2026-08-14.md` GAP 1 was right to narrow the sealing language, and it
is narrowed further by this: the residual-corrected sequential family stays open
by name, and so does any global form that does not depend on frozen gates.

## 8. Estate

Eleven commits, **nine deployments, every one verified file-by-file against the
commit's own blob digests**: 5,376 files at the first two and 5,379 at the rest,
**0 content mismatches, 0 missing, 0 undeclared, 0 hardlinks** every time. Nine
deployments in one wave is far more than this campaign's habit, and the reason is
worth stating rather than hiding: six of them exist because a guard refused, and
the estate's rule is that the fix ships as a commit and a verified deployment
rather than as an edit to a deployed file. Deployments were pruned as they were
superseded; current+1 holds. Parent lane calibration
`865f006e3813712ec202cb23` on protocol `e5e377688597` / pack `0ee6bf14cef5` —
the anchor, unmoved, on all seven wave-29 dossiers.

Both sm86 3090s were taken for the wave. Two ComfyUI H3 containers held ~17 GiB
each on indices 0 and 4 with **empty queues and zero utilisation**; nothing was
running and nothing was queued, so nothing was killed. The `/free` endpoint did
not release the allocator, so both containers were **restarted** — the least
destructive move that returns the cards — and both came back healthy with the
service up. 288 MiB and 279 MiB free-standing afterwards.

Seven refusals fired during the wave and every one of them was correct:

* the bank lint, on a 7-gram shared with the sealed pack;
* the bank digest pin, on CRLF-versus-LF — the pin named the working copy's bytes
  and the deployment ships the commit's;
* the sealer's build-receipt allowlist, on three wave-29 builder schemas;
* the receipt validator, twice, on axes missing `evidence` / `parameters_removed`
  / `heldout` — and filling those in is where `parameters_removed: 0` for a
  materialized parent-shaped body came from;
* `Streams.masks`, on a joint capture that kept its row bookkeeping a second way;
* and the sealer's own "1 of 16 lanes charged" line, which caught seven seals
  that named two or three charged lanes and sealed one, **before the ledger
  merge**. Those seven receipts and shards were deleted and the pass re-run, so
  no superseded row reaches the ledger. `seal ()` now refuses a call that does
  not pass exactly eight arguments.
