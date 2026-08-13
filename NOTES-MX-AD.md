# NOTES-MX-AD — wave 28: the block sets the landing, the drift has a name, and the last lane was a copy

Wave 28, branch `codex/lrc-blank-compression`, source commit `18b99225` (the
wave-28 tree, `driftq.sh`, the seal-root line and the allowlist line) plus the
sealing commit for this note. `NOTES-MX-AC` is wave 27; the series is contiguous.

Seventeen graded children on two GPUs, plus two bodies no wave has sealed.

## 1. The joint-solve drift has a name, and it is a repair

Wave 27 rebuilt wave 17's sealed `ARCH.F1301KV6Q` and the guard refused —
identical 200,847,104 parameters, sealed digest `90a9e5187feef678…`, rebuilt
`ad406f480a181901…`, byte-identical on a second rebuild, so drift and not
nondeterminism. It could not say which change caused it.

**A `git bisect` is the wrong instrument here and that is the finding, not an
excuse.** This campaign does not *edit* one builder; every wave *copies* the
previous wave's builder to a new path and edits the copy. `run/mx17/mx17_build.py`
is still in the tree at HEAD, byte-for-byte as wave 17 ran it, and **no commit has
changed it since**. A commit bisect would truthfully report "nothing changed this
file" and tell you nothing. The bisect variable is the **wave**, not the commit —
and because every builder from mx17 to mx28 ships in one deployment, the whole
sweep runs against one deployment with no checkout and no rebuild.

`run/mx28/driftq.sh` builds the same body with each wave's builder and the same
six flags. Twelve rungs, ~90 seconds each:

```
wave 17: 90a9e5187feef678…  params 200847104  live-mask-repair=0  == wave 17 sealed
wave 18: ad406f480a181901…  params 200847104  live-mask-repair=1  == wave 27 rebuild
wave 19 … wave 28: ad406f480a181901…          live-mask-repair=1  == wave 27 rebuild
```

**One step, at wave 18, stable for ten waves since.** The change is
`mx18_build.py`'s `if excised:` branch, which re-derives
`harness.model.config.layer_types` after an excision so the LIVE model's causal
mask follows the live blocks. Wave 17's builder did not. `Lfm2Model` picks each
block's mask by POSITION in that list, so on a body whose excision shifts a later
attention block into a conv's slot — which blocks 1+13 do and block 13 alone does
not — wave 17's conditioned capture put 28.6–48.0% of the module's attention mass
on tokens that had not happened yet. Wave 18 measured that with `mx18_mask_probe.py`
and fixed it. Every solve that consumed that capture consumed different evidence.

Three consequences, and the third is the one that matters:

* **The drift is a REPAIR.** The sealed body is the defective one; the rebuild is
  the correct one. So this is a WONTFIX: reverting it would restore a measured bug.
* **It explains why wave 26 was right and wave 27's generalisation was wrong.**
  A pure excision copies tensors and never reads the mask, so its *tensors*
  reproduce bit-exactly even though its *config* differs. "Excision bodies
  reproduce" was true of tensors and false of anything with a solve in it.
* **The cell still may not be cited.** The rebuild is better and it is not what
  wave 17 reviewed. `bodyq.sh` now carries a KNOWN_DRIFT registry so the next wave
  to hit this refusal gets the attribution with it instead of re-deriving it.

**The control.** `driftq.sh --check-vocab` rebuilt this wave's `V18X13X01` and
compared it to wave 27's sealed digest: `fe649aae5fa48fa2…` both sides. **The
vocabulary path reproduces bit-exactly across a wave boundary.** And the frozen
review of those bit-identical tensors, generated independently and read
independently, returned the same **3/15 on the same three lanes** wave 27 sealed.
Both the build path and the reader reproduce.

## 2. The block sets the LANDING, not the cost

Wave 27 wrote that on any shelf-or-better base, block 13 lands the child at ten.
Wave 28 ran it off the **summit** — `ARCH.V19`, the 14/15 at 0.794570x.

| base | base review | +X13 | +X09 |
|---|---:|---:|---:|
| V18 (13/15) | 13 | 10 (−3) | 10 (−3) |
| V20 (13/15) | 13 | 10 (−3) | 10 (−4) |
| **V19 (14/15)** | **14** | **10 (−4)** | **10 (−4)** |

The **cost** is 3 or 4. The **landing** is ten, six times out of six. The two
cases the summit is worth do not survive one excision, which is the sharpest
statement this campaign has that the vocabulary peak at 19,456 is a property of
the token interface and not of the model underneath it.

`ARCH.V19X09` is the row that makes this a claim about composition rather than
about block 9. It has the identical parameter count and the identical 0.742049x as
`ARCH.V19X13` and reads the same ten **by a different route** — four Useful
against three, winning a real greeting, three real actions and the correct
insufficient-data hedge, losing the arithmetic chain and the dedup list. Same
landing, different lanes.

## 3. Lambda 0.3 is not dead, and the first monotone ladder

Wave 27 demoted λ = 0.3 on six arms that all lost or tied. Wave 28 ran the full
ladder on the summit excision, 1,024 re-solved values and zero added parameters
against a byte-identical control:

**`V19X13`: 10 / 10 / 11 / 11** for no-solve / 1.0 / 0.3 / 0.1.

`ARCH.V19X13G03` and `ARCH.V19X13G01` are **11/15 at 0.742049x — the deepest
eleven in the campaign**, eleven points of ratio below the previous holder
(`V17`, 11/15 at 0.785440x). So the demotion stands as written — 0.3 is not the
rung that never hurts — and is not a burial. This is also the **first monotone
ladder** any base has produced; `V16X13` read 5/6/7/6/5 and `V8X13` read 6/6/7/7/4,
both single-peaked.

The deeper ladder went the other way. **`V18X13X01`: 3 / 4 / 4 / 3**, single-peaked
in the middle, while its held-out terminal-logit objective improved *monotonically*
with decreasing λ (0.5086 → 0.4165 → 0.3912 → 0.3827). The best objective is the
worst review, for the fourth and fifth time in two waves.

**What the solve actually buys, measured on five arms:** it gains the concise
email (it starts *requesting* the tracking number), the insufficient-data lane (it
stops falsely asserting the increase), the six-step plan (it starts naming DNS,
TLS, verification and rollback instead of "Density", "Trait" and "Diversity"), and
on the summit arms the **safety lane**, where the zero-solve control only asks for
clarification and every solved arm tells the reader to remove the cleaners and
leave the area. It loses **coding-deduplicate** on every arm that had it: the
control accumulates into a LIST with `append`, the solved arm accumulates into a
SET. That is the wave-26 LIST→SET regression on its **fifth and sixth** base, and
it is now a property of the terminal-gain solve beyond reasonable doubt.

## 4. The gate refusal was narrower than wave 27 wrote

Wave 27 recorded that the terminal-gain identity gate refuses "a width fold",
because the objective needs the child's token table to BE the parent's rows at the
injected ids. That is true of a **residual-stream** width fold (`--hidden-size`),
which projects the embedding. It is **not** true of an **MLP** width fold
(`--ff`), which never touches the embedding. `CL896` is the latter, so a
vocabulary fold on top keeps its rows verbatim and the gate passes at 8.0e-14.

Both CL896 arms built and both gained: `V19CL896G03` 3/15 against 2/15, and
`V20CL896G03` 3/15 against wave 27's 2/15 — **the same +1, the same lane (the
concise email), on two bases 1,024 rows apart.** A solve buying a case at
0.4876x is the deepest a solve has ever bought one.

## 5. The pair distance law inverts on a vocabulary base

Wave 17 measured that farther-apart excision pairs cost less (two apart 2–4 of 15,
twelve apart 7, monotone across two anchors). At one parameter count
(158,379,776), one ratio (0.689528x) and one vocabulary fold:

* blocks 9 and 13 — **four** apart — reads **5/15**
* blocks 1 and 13 — **twelve** apart — reads **3/15**

The failure modes differ as much as the counts: `V19X13X09` hits the generation cap
on **eight of sixteen** lanes and scrapes five Partials out of the survivors, while
`V19X13X01` caps once and returns the prompt verbatim on four. One base, n = 2, and
it is the first measurement that contradicts the distance law rather than extending
it. `MX28-P1309` is a body no wave had sealed.

## 6. Wave 27's last-lane claim was an artifact, and this wave killed it

Wave 27 observed that every collapsed child, down to 0.4876x, still answered
**summarisation**, and concluded that summarisation is the last lane to go.

**The summarisation prompt contains its own answer.** It is "Summarize this update
in two sentences, preserving the decision and the unresolved risk:" followed by
five sentences whose **last two** are "The team selected Supplier B. The remaining
risk is that its factory expansion may slip by one week, so procurement will
request weekly schedule evidence." A child that emits those two sentences has not
summarised — it has copied the tail of its own prompt.

Wave 28 wrote the check into `sealq.sh` **before** it graded anything: on every
child under 0.60x the summarisation lane is read first and written down before the
other fifteen are opened, so the claim could not be confirmed by a reader who
already believed it. What the check found:

* `V18X13X01` and `V19X13X01` emit the prompt's last two sentences with the word
  "has" inserted. Charged for prompt-echo, read **Partial**, not Useful.
* `V19CL896`, `V18CL896`, `V20CL896G03` and `V19CL896G03` echo the **entire
  prompt**, tail included. Same reading.
* `V19X13` and `V19X09` **reword** ("necessizing a weekly schedule for procuement").
  Useful, and earned.

And the discriminating case: **`ARCH.V19X13CL896`, 0/15 at 0.452734x** — the
deepest child this campaign has ever built, 45% of the parent, a 19,456-row
coreset on a body that is both the block-13 excision and the CL896 MLP fold. Its
summarisation lane does **not** copy. It paraphrases, and the paraphrase keeps the
risk and **drops the decision**, so the lane is a miss. What was dying last was the
copy, not the summary.

No count changed. Six gradings did, and the campaign-level claim is withdrawn.

## 7. A correction to this wave's own sealed text

`ARCH.V18CL896`'s sealed ADDENDUM 6 note calls it "THE DEEPEST CHILD IN THE
CAMPAIGN TO HOLD ANY CASE". **That is wrong and the frontier table caught it after
the seal.** It is the deepest child to hold **two** cases (0.478436x, taking the
record from wave 27's `V20CL896` at 0.487566x). The deepest to hold **any** case
remains wave 23's `ARCH.V16CL896`, 1/15 at **0.469305x**. Corrected here rather
than re-sealed, because re-sealing would put a second row in the ledger for one
measurement, which the merge guard exists to prevent.

## 8. The spectrum, and the frontier delta

Wave 28 moved **five** rungs of the coherence frontier:

| rung | before wave 28 | after wave 28 |
|---|---|---|
| ≥ 11/15 | `V17` 0.785440x | **`V19X13G03` / `V19X13G01` 0.742049x** |
| ≥ 10/15 | `V18X13` / `V18X09` 0.737484x | unchanged |
| ≥ 7/15 | `V8X13GL01` / `GL03` 0.691832x | unchanged |
| ≥ 5/15 | `V8X13` 0.691832x (6/15) | **`V19X13X09` 0.689528x** |
| ≥ 4/15 | `CL896` 0.688431x | **`V18X13X01G03` / `G10` 0.684962x** |
| ≥ 3/15 | `V20X13X01` 0.694093x | **`V19CL896G03` 0.483001x** |
| ≥ 2/15 | `V20CL896` 0.487566x | **`V18CL896` 0.478436x** |
| ≥ 1/15 | `V16CL896` 0.469305x | unchanged |
| answers nothing | `V18X13X01KV6JQ` 0.664419x | **`V19X13CL896` 0.452734x** |

Campaign totals: **232** reviewed whole-model children, **182** holding at least
one case, **136** benchmark rows, **15** parent rows all matching the sealed
reference.

## 9. Estate

Deployment `neural-forge-matrix-18b99225` verified file-by-file against the
commit's own blob digests **before** the wave: 5,189 files, 0 mismatches, 0
missing, 0 undeclared, 0 hardlinks. Parent lane calibration `865f006e3813712ec202cb23`
on protocol `e5e377688597` / pack `0ee6bf14cef5` — the anchor, unmoved. Parent
benchmark row reproduced exactly: 0.5450 / 0.5050 / 0.6800 / 0.5500 / 0.4600,
lambada ppl 24.39, wikitext word ppl 81.80.

Two GPUs throughout: independent build/seal/review/benchmark queues on sm86
indices 0 and 4, three concurrent CPU vocabulary batches feeding them. Seventeen
graded children, two new bodies, twelve drift-bisect builds and eighteen benchmark
rows in **≈70 minutes** of wall clock against ≈140 serial — about **1 h 10 m**
saved. Sealing and the ledger merge were kept single-writer and serial on purpose,
and the seal pass ran from `$W` rather than from the deployment, so the post-wave
deployment verification stays clean.

Ledger 770 → **804 rows** (17 ARCH + 17 verdict, no superseded rows), backed up
with sha + `cmp` first (`e4a6a863f9a752f7…`, matching the wave-27 handoff exactly),
byte-prefix verified after, and **committed**. Merged sha
`445ff68e6c08c99d8be4dfe2fe875c36d6dd85c4d26401e3ad1b5466ed6cb58a`, identical on
the server and in the repository. 24 cell roots audited, 0 undeclared,
`wave27/cells` still declared and `wave28/cells` appended as one line. 800 of 804
rows resolve and verify; the 4 that do not are the three `CAL.C01.T5.X4` rows and
the one `COMP.MC.MXO0` row wave 27 identified.

Package built **before** retention: 128 files, 2.8 MiB, every dossier, every child
and parent response verbatim, every receipt, verdict, per-case reading and build
receipt, the bits/byte column and the drift-bisect log. Ten of nineteen checkpoints
retained (3.0 GiB), nine dropped with reasons (2.5 GiB freed), dry run read first.
