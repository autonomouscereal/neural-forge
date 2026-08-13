# NOTES-MX-Z — wave 22: the objective loses where it had room, the slope has no step left to hide, and two recovery mechanisms do not add

Wave 22, branch `codex/lrc-blank-compression`, source commits `5e17bb58` (the
wave-22 drivers, whose builder differs from wave 21's by nine lines of body),
`70a2498d` (the one-line basis-schema defect the census refused to run with),
plus the sealing commit for this note. **Two deployments, and every receipt says
which:** `neural-forge-matrix-5e17bb58` built and sealed all three children and
is named in `repo_head` on all three ARCH receipts;
`neural-forge-matrix-70a2498d` ran the endpoint census and sealed nothing. Both
were cut with `git archive -c core.autocrlf=false` and verified file by file
against their own commit's blob digests: **3,736 files, 0 content mismatches, 0
missing, 0 undeclared, 0 hardlinks**, each.

Protocol `e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b` and
prompt pack `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`
verified digest-true on every dossier.

**This wave added no mechanism.** `git diff` between the two builders is four
schema strings, one log tag, two provenance strings and two receipt filenames —
the solver, the census, the three basis objectives, both lambda splits, the conv
coreset, the head-gain solve, the excision path, the merge and the export are
byte-identical. Everything below is therefore a measurement of cells that were
already buildable, which is what a wave should look like when the mechanism
search has stopped paying.

**Three calibrations passed before anything was graded.** The conditioned
capture reproduces the sealed collector to waves 15, 19, 20 and 21's numbers
exactly (L00 relL2 3.367e-3 cos 0.9999944, L05 1.647e-2 / 0.9997207, L13
2.034e-2 / 0.9997842). The parent's sixteen frozen-review responses digest to
**`865f006e3813712ec202cb23`** on all three wave-22 dossiers. The parent's
benchmark row reproduces the sealed reference in all seven columns (0.5450 /
0.5050 / 0.6800 / 0.5500 / 0.4600, lambada ppl 24.39, wikitext 81.80) — the
**ninth** wave to re-run it and the ninth to match. `mx22_selftest.py` is
**24/24** with no model loaded, and it adds no check of its own on purpose: this
wave adds no mechanism, so a new check would be a check of nothing, and what the
twenty-three are for here is to prove the copy of the builder is the one waves
19–21 measured with.

*§0's sentences are at the bottom.*

---

## 1. The max-min objective at the width where it had something to lose

Wave 21 built `--basis reader-maxmin` at 768 and it read 0/15 with lambada
0.0000 and perplexity 23,120. The one defensible objection to that result was
that 768 is a child with three cases and 39 of 768 directions moved: a dead
child there could be the objective, or it could be that there was no room. At
896 the control holds **four** cases, the worst endpoint is already 0.9440, and
the free measurement priced the exchange at 24.7 of 896 directions.

`ARCH.MM896`, 0.859023x, **197,311,616 parameters — the same count as wave 19's
sealed 4/15 child and wave 21's 3/15 lambda-split child**, differing only in
which 896-dimensional subspace of the residual it keeps.

**The objective is met almost exactly:**

| at width 896 | plain reader-PCA | reader-maxmin |
|---|---:|---:|
| worst endpoint, read, held-out | 0.9440 | **0.9530** |
| worst endpoint, read, **fit** | 0.9580 | **0.9706** |
| spread across 29 readers, read, **fit** | 0.0231 | **0.00055** |
| spread across 29 readers, read, held-out | 0.0307 | **0.0132** |
| pooled read, held-out | **0.9627** | 0.9603 |
| pooled raw, held-out | **0.9777** | 0.9746 |
| directions exchanged | 0 | **24.69 of 896** |
| final-norm relative logit SSE | 0.1752 | **0.1513** |
| review | **4/15** | **2/15** |

Twenty-nine readers retained to within **0.00055 of each other** on the fit
rows, against a spread of 0.0231 for the basis it replaces. The weight the
iteration moved is where the measurement said to move it — `B02.ffn` 0.120,
`B01.ffn` 0.107, `B00.ffn` 0.076, `OUT.final` 0.068, `B03.ffn` 0.063 — and the
child reconstructs the parent's own logits **better** than either control.

**And it holds two cases where the controls hold four and three.** What it loses
is not random. The summarisation lane — Useful in `W896`, Useful in `W896LS`,
Useful in `W896LSX13`, Useful in every 896 child ever built — **returns the empty
string**, which is what it returned at 768 as well. What it keeps is the two
lanes with the most task nouns to echo: the email lane puts both asks the prompt
names in a subject line under a salutation ("Subject: Can you confirm the
shipment status and the tracking number? / Dear [vendor]") before it loops on the
prompt's own 90-word constraint, and the planning lane names backups, DNS, TLS,
verification and rollback inside a six-step frame before it degenerates into
counting steps. Both are read against sealed calibrations, not against this
wave's expectations: `W896`'s Partial on the email lane is the ask with no form,
`W896LS`'s miss is the form with no ask, and `W768EV`'s sealed Partial on the
planning lane is four numbered items naming backups, DNS, TLS and rollback
before it fills the cap with "14".

Every one of the sixteen lanes carries the same fingerprint the 768 child had:
**"I am a" followed by a number lifted from the prompt** — "I am a 2- to 3- s",
"I am a 3-step plan", "I am a 12-year-old", "I am a 7:30 a.m.", "I am a 1- to 2-
sided sizing", "I am a 4- to 5- s". The basis that equalises the readers produces
a model that reads the prompt's own numerals and cannot leave them.

**So the objection is answered: it loses where it had room.** Six objective
tests have now failed against the pooled reader moment's own top eigenvectors —
case-weighted at two betas, evidence-PCA, reader-generalised, and max-min at two
widths — at exchanges from 17 to 231 directions, in every direction the
objective can be moved.

## 2. Rung 720, and the slope has 16 dimensions left

`ARCH.W720`, 0.672632x, 154,499,088 parameters, retained read energy **0.9015**
pooled held-out, **0/15**.

| width | net | retained read (hld) | worst endpoint | readers under 0.90 | review |
|---:|---:|---:|---:|---:|---:|
| 896 | 0.859023x | 0.9627 | 0.9440 | 0 of 29 | **4/15** |
| 768 | 0.722610x | 0.9193 | 0.8801 | 4 of 29 | **3/15** |
| 736 | 0.689220x | 0.9075 | 0.8632 | 9 of 29 | **1/15** |
| **720** | **0.672632x** | **0.9015** | **0.8545** | **11 of 29** | **0/15** |
| 704 | 0.656116x | 0.8953 | — | — | 0/15 |
| 640 | 0.590763x | 0.8697 | 0.8084 | 26 of 29 | 0/15 |
| 512 | 0.463481x | 0.8124 | 0.7268 | 29 of 29 | 0/15 |

**The last interval that can still hide a step is 736 → 720: sixteen dimensions
and 0.0060 of pooled retained read energy.** Wave 19 bracketed the collapse at
128 and called it a step; waves 20, 21 and 22 have halved it three times, and
every rung inside it has been monotone.

What 720 loses against 736 is the one case 736 held. `W736`'s multi-turn lane is
"Good catch, I am planning for 8 people. I am planning for 4." — the correction
acknowledged and the corrected quantity restated. `W720`'s is **"Good, I am happy
to help plan snacks for 8 people. I was thinking everything will contain
peannuts."** — the user's own PRE-correction premise echoed back, which is the
thing the correction corrects. Seven of its sixteen lanes return the empty
string and five fill the cap with a bare dash. Nothing in it names a quantity, a
step or a fact from any prompt.

The per-site install is, as at every dead width, fine: median write cosine
**0.99986**, worst site the layer-1 conv out at 0.9595, max input drift 0.4083.

**The max-min basis at 720 was priced and not built.** The same census says it
would lift the worst endpoint from 0.8545 to 0.8783 and drop the pooled number
from 0.9015 to 0.8944, exchanging 43.6 of 720 directions. There is nothing to
buy: the plain arm at that width is already 0/15.

## 3. The two case-recovery mechanisms, composed, do not add

The writers' internal fit split is the campaign's only measured case-**recovery**
mechanism: it returns `context-retrieval` at 768, `planning-dependencies` at 896,
and `planning-dependencies` again inside the 896+excision composition. The head
merge is the campaign's best two-axis partner. They had never been put in one
child.

`ARCH.W896LSKV6JQ` — `--hidden-size 896 --width-lambda-split --kv-groups 6
--kv-solve joint --kv-als 0` — 0.841047x, **193,182,848 parameters, which is wave
19's `W896KV6JS` parameter for parameter**. The split ran on 3,168 inner rows
against 1,583 check rows (every third fit batch from the third). The merge
removed 4,128,768 parameters at median head cosine 0.9459, worst 0.9210.

| child | recipe | review | per-lane reading |
|---|---|---:|---|
| `ARCH.W896KV6JS` (wave 19) | 896 + merge with the **merge's** λ split, 1 ALS sweep | **2/15** | — |
| `ARCH.W896LSKV6JQ` (wave 22) | 896 + **width** λ split + merge, 0 ALS sweeps | **1/15** | impaired (15/16) |

**It holds the summarisation lane and nothing else.** "The team selected Supplier
B. The team selected Supplier B. The remaining risk is that its factory expansion
may slip by one week, so procurement will request weekly schedule evidence." —
the decision, the unresolved risk and the mitigation, with one sentence repeated,
in 184 characters against the parent's own 160.

**And the capability the split exists to recover is gone.** `planning-
dependencies` — returned at 896 alone as backup/DNS/TLS/verify/rollback in the
parent's order, and returned again inside the excision composition — fills the
cap here with "- 1. Sending a d d d d". The split's recovery survives an
excision and does not survive the merge.

**The gate fired and the triple was not built.** Wave 22 declared before building
that `X07 + KV6JQ + 896` would be built only if this child held at least three
cases. It held one. A triple on a two-axis child that has already collapsed
measures the collapse, and the GPU-hour was not spent.

## 4. Benchmarks, and the first wave in four where the instruments agree

ADDENDUM 8 invocation unchanged (limit 200, seed 1234, batch 4, max_length 2048,
float32, sm86):

| model | net | review | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `W896` (wave 19) | 0.859023 | **4/15** | 0.3550 | 0.4000 | 0.5650 | 0.4900 | **0.1450** | **638.72** | 500.08 |
| `W896LS` (wave 21) | 0.859023 | **3/15** | 0.3650 | 0.4050 | 0.5650 | 0.5100 | 0.1400 | 622.20 | 487.20 |
| `MM896` | 0.859023 | **2/15** | 0.3200 | **0.4350** | 0.5550 | 0.5050 | **0.0550** | **1880.53** | 554.62 |
| `W896KV6JS` (wave 19) | 0.841047 | **2/15** | 0.3150 | 0.3850 | 0.5200 | 0.5150 | 0.0650 | 2821.11 | 1161.46 |
| `W896LSKV6JQ` | 0.841047 | **1/15** | 0.3300 | 0.3950 | 0.5000 | 0.5550 | 0.0600 | 2829.20 | 1128.35 |
| `W736` (wave 21) | 0.689220 | **1/15** | 0.2850 | 0.4050 | 0.5150 | 0.4800 | 0.0150 | 14146.56 | 1787.96 |
| `W720` | 0.672632 | **0/15** | 0.2600 | 0.4050 | 0.5050 | 0.4850 | 0.0150 | 17773.80 | 2046.32 |

**At 896 the two instruments agree for the first time in four waves.** The three
arms at the identical parameter count order 4 > 3 > 2 on the review and 0.1450 >
0.1400 > 0.0550 on lambada, 638.72 < 622.20 ≈ < 1880.53 on perplexity. Waves 20
and 21 both had to report a negative whose accuracy columns preferred the dead
child; this one does not.

**And at 0.841047x they cannot see the difference at all.** `W896KV6JS` and
`W896LSKV6JQ` are the same parameter count and differ by one review case, and
their benchmark rows are indistinguishable: lambada 0.0650 vs 0.0600, perplexity
2821.11 vs 2829.20, wikitext 1161.46 vs 1128.35, four accuracy columns within
0.0150 of each other. Seven columns and 200 samples each cannot resolve what one
reading of sixteen lanes can.

**One column worth naming: `MM896` has the best `hellaswag` of any 896 child ever
built** (0.4350 against 0.4000 and 0.4050) and the worst lambada. A wave
reporting the accuracy columns alone would have called the max-min basis a win
at this width too.

## 5. The defect this wave introduced, and the measurement that refused to run

`BASIS_SCHEMAS_ACCEPTED` exists so a wave can READ a basis an earlier wave
sealed rather than rebuild it, and the comment above it says why: a schema string
that refuses last wave's artifact quietly turns a comparison into a rebuild, and
a rebuilt basis is not the sealed one.

Cutting the wave-22 tree by a token rename rewrote the **first** entry of that
tuple — `BASIS_SCHEMA` itself — and left the other two, so wave 21's own sealed
bases became unreadable by the wave whose job is to compare against them. The
endpoint census refused with `E-reader-pca-736.safetensors declares schema
'neural-forge-matrix-wave21-residual-basis-v1'` and exited 1 **without writing
its report**, which is the only reason this is a one-line fix (`70a2498d`)
instead of a silent rebuild of a sealed artifact.

It is the same class of defect the seal verifier has warned about five times,
in a different file: a rename that deletes last wave from a list this wave has to
read. The verifier itself was excluded from the rename **by name** and extended
by one line, which is the third time that warning has been obeyed as written.

**No child is affected.** All three were built against wave 22's own two bases
and wave 19's sealed 896, all of which were already accepted, and the tuple is
read only when a basis file is opened; it changes no solve, no selection and no
export. `git diff 5e17bb58 70a2498d -- src experiments` is **empty**.

## 6. Estate

* **Commits.** `5e17bb58` (the wave-22 drivers: the builder with nine lines of
  body changed, the queues, the verifier with one added root, the summarize glob
  fixed, a `sealq.sh` with no verdicts and a `mx22_retain.py` that refuses on an
  empty KEEP set), `70a2498d` (the basis-schema fix), plus the sealing commit for
  this note. Nothing pushed, nothing rebased, staged by explicit path only; the
  Codex lane's uncommitted paths, including the frozen
  `src/neural_forge/hf_lfm2_circuit_trace_cache.py`, were not touched.
* **One line in the shared sealer.** `experiments/seal_matrix_architecture_child.py`
  gains `neural-forge-matrix-wave22-standard-shape-child-v1` and nothing else.
* **Deployments, both verified byte-true.** `neural-forge-matrix-5e17bb58` and
  `neural-forge-matrix-70a2498d`, each **3,736 files, 0 content mismatches, 0
  missing, 0 undeclared on disk, 0 hardlinks**, against a manifest of their own
  commit's blob digests; seven files over 20 MB excluded by name, as every
  deployment in this campaign has excluded them. Waves 20 and 21's deployments
  were pruned under current+1 and named in
  `wave22/deployments-prune-manifest-wave22.json` (0.41 GiB freed).
* **Ledger.** **646 → 652**, sha
  `33a8326ae956e15aed0f1b2443a25377351d167b0bbb78667f5a0d8a0f74fe29`. Backed up
  first (`backup/ledger.jsonl.pre-wave22-20260813T122626`, verified by sha **and**
  `cmp`), merged exactly once behind `wave22/.MERGED_ONCE`, byte-prefix verified
  against the backup afterwards and again in the repository copy. Six rows: 3
  ARCH run rows and 3 ADDENDUM 6 verdict rows, no superseded rows.
* **Seals.** `mx22_verify_seals.py --deployment <deployment>`: **18 roots
  present, 0 undeclared; 652 rows, 648 resolve, 648 verify, 0 fail, 4 do not
  resolve** — the same four pre-existing non-resolvers waves 14–21 named.
* **Retention.** Three children built, **all three kept**: each is one arm of a
  controlled comparison whose other arm is sealed in an earlier wave, two of them
  at the identical parameter count. Nothing deleted, 0 GiB freed, and the
  manifest says so per child.
* **Disk.** Server `/` began and ended at **12 GB free**.
* **Human-review package, in the repository.**
  `reports/matrix-campaign-20260810/wave22/human-review/` — 31 files.
* **The scorecard did not move, and that is reported rather than hidden.** All
  three children are too shallow to rank (0.859023x, 0.841047x) or answer nothing
  (0.672632x at 0/15), so the fifteen rows of `CAMPAIGN_SCORECARD.md` are
  byte-identical to wave 21's. The consolidated report's counts moved — 154 → 157
  reviewed whole-model children, 108 → 110 scoring at least one case — and its
  §1 table, §2.4 width law, §2.4 objective table and §6 unbuilt list are
  regenerated or rewritten.

## 7. What this wave did not spend

* **The triple.** Gated out by its own pre-registered rule at 1/15, not
  forgotten. If a later wave wants it, the gate is the thing to argue with.
* **The 736 → 720 rung.** Sixteen dimensions. One build, one review, and the
  width slope has no interval left that could hide a step.
* **The max-min basis at 720**, priced free and refused for cause.
* **The λ split at 720 or below.** The plain arm is 0/15 and the split has never
  bought a case in a child that had none.
* **Nothing was tried between the pooled reader moment and the six objectives
  that have failed.** Wave 21's honest reading — that the search space around the
  pooled eigenvectors is not merely narrow, it is a point — survives a second
  width.

## 0. The sentences

1. **The max-min objective loses at the width where it had room, so the width
   was never the explanation.** `ARCH.MM896` equalises twenty-nine readers to
   within **0.00055** on the fit rows, lifts the worst held-out reader from
   0.9440 to 0.9530, reconstructs the parent's logits better than either control
   (0.1513 against 0.1752), and reads **2/15** against 4/15 and 3/15 at the
   identical 197,311,616 parameters.
2. **Six objective tests have now failed against the pooled reader moment's own
   top eigenvectors**, at exchanges from 17 to 231 directions and at two widths.
   The only constructions that keep the pooled subspace exactly are the only ones
   that read at the top of their width.
3. **The summarisation lane is the max-min basis's signature casualty at both
   widths.** It is Useful in every other 896 child ever built and returns the
   empty string here, exactly as at 768.
4. **The width slope has sixteen dimensions left.** `ARCH.W720` reads **0/15** at
   0.9015 retained read energy, so the ladder is 4 → 3 → 1 → 0 → 0 and the last
   interval that could hide a step is **736 → 720**, 0.0060 of energy wide.
5. **720 does not fail quietly, it fails backwards.** Where 736 acknowledges the
   correction and restates the corrected quantity, 720 echoes the user's
   pre-correction premise — "everything will contain peannuts" — which is the
   thing being corrected.
6. **The campaign's two case-recovery mechanisms do not add.** The width λ split
   composed with the head merge reads **1/15** against the merge's own split at
   **2/15** and the same 193,182,848 parameters, and the capability the split
   exists to recover does not survive the merge though it survives an excision.
7. **At 896 the review and the distribution agree for the first time in four
   waves**: 4 > 3 > 2 on cases and 0.1450 > 0.1400 > 0.0550 on lambada, at one
   parameter count.
8. **At 0.841047x the distribution cannot see a difference the review can.** Two
   children, same parameter count, one review case apart, and seven benchmark
   columns that put them within 0.0150 of each other on everything.
9. **A wave reading the accuracy columns would have called the max-min basis a
   win at 896 too.** It has the best `hellaswag` of any 896 child in the campaign
   and holds two cases where the baseline holds four.
10. **The rename that cut this wave's tree deleted last wave's bases from a list
    this wave had to read, and the free measurement refused rather than
    rebuilt.** One line, one commit, no child affected, and the fifth instance of
    exactly that defect class in this campaign.
11. **The deployment was verified file by file both times**, 3,736 files and 0
    content mismatches each, and the wave ran from two of them because a guard
    fired mid-wave; every receipt names the one that sealed it.
12. **The scorecard did not move.** Three children, none of them deep enough to
    rank or alive enough to qualify, and the honest report of a wave is the page
    it failed to change.
