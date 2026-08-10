# NOTES-MX-F — matrix campaign, wave 2

Wave MX-F. The adaptive deepening wave 1's sealed results dictated: 32 cells
against the same fresh evidence base `2dec87ac…` at the same pre-registered 5e-3
calibration tolerance, on both RTX 3090s, plus the three harness fixes
`NOTES-MX-E` left as obligations.

Repo baseline HEAD `347e8d93`, branch `codex/lrc-blank-compression`. The frozen
collector, the compatibility policy and Codex's live-edit files
(`dynamic_fold_diagnostics`, `dynamic_model_inspector`, the closed-loop adapter,
the `stage2_*` files and their tests) were **not** touched. No `git add`, no
commit. Wave-1 receipts were never opened for writing.

**Verdicts up front:**

| deliverable | verdict | one line |
|---|---|---|
| 32 cells, wave 2 | **ALL SEALED** | 7 PASS, 23 FAIL, 2 INFRA; 19:28→19:42 UTC on two GPUs |
| §5 rule 2 for the attention readers | **BOTH HOLD, at all three sites** | C06/T1 3/3 (0.977 / 0.977 / 1.000) and C07/T1 3/3 (0.941 / 0.936 / 0.994) at M16 |
| the M-ladder | **behaviour is flat, then it cliffs** | C06/T1 loses 0.006 from M16→M8 and 0.36 from M8→M2, while composed drift roughly doubles per rung |
| the MLP wall | **still standing, and the rescue made it worse** | doubling the OMP route budget at C13/L05 moved free-run 0.502 → **0.430** |
| T3, never run before | **the best conv-writer cell in the campaign** | `C05.T3.M16.L05` free-run **0.914** against T1/T2's 0.744 and T5's 0.791 at the same site |
| T2, never run before | **not a distinct technique at a body component** | its chart decoder measures out to the encoder transpose at 9e-9, and its boundary numbers equal T1's to ten digits |
| the four void cells | **two are now measurements, two are localized to a missing kernel** | `C14.T1.NA.L00` **PASSES** (free-run 1.000); `C03.T5.M16.L00` is 0.787; C04/C10 have no registered kernel for their contraction axis |
| the storage trap | **8 of 30 sealed cells with a storage reading actually contract** | and M2 lands on exactly 1.0000 at both square components, as the arithmetic said it would |
| the S8 band | **first genuine agreement in 31 measured cells** | `C05.T3` is in band *and* behaved; the other 30 are in band and did not |

---

## 1. The three harness fixes, before any cell ran

### 1.1 `compute_architecture` is now a mandatory receipt field

`NOTES-MX-E` §2 registered it as a campaign rule and left it unimplemented: the
receipt schema had no field for the compute architecture a cell ran on, and it
is the one input to a matrix cell the sealed identities did not cover. Wave 1
came within an evidence-loading crash of merging five sm_61 rows into a ledger
of sm_86 rows, with nothing in either row saying which was which — and A9's fp32
OMP candidate search is architecture-non-portable (`NOTES-MX-C` §4.4), so those
rows would not have been comparable.

`code_identity.compute_architecture` is now in `_MANDATORY`
(`src/neural_forge/matrix_cell_receipt.py:345`), validated non-empty
(`:483`), and carried into every ledger row (`:942`). The runner resolves it
from the *device*, not from the environment — `compute_architecture()` at
`experiments/run_lfm2_matrix_cell.py:203` reads
`torch.cuda.get_device_capability(index)` and `get_device_name(index)` off the
device torch actually opened, which is precisely the reading the
`FASTEST_FIRST`/`PCI_BUS_ID` ordering surprise defeated.

The guard moved inside the process too. Wave 1's preflight was a shell `if` in
the batch driver; it worked, and it was one careless invocation from not
existing. `--require-compute-architecture sm86` now refuses **before
`blank_receipt` is called**, so a wrong-GPU cell produces no receipt at all
rather than a sealed row somebody has to quarantine afterwards. Every wave-2
cell passed the flag; every wave-2 receipt seals
`sm86 NVIDIA GeForce RTX 3090`.

**Consequence, recorded rather than absorbed:** receipts sealed before this
field exists — all of wave 1 — do not validate under the current schema. They
are preserved byte-for-byte as sealed and re-verify their *seals* (124/124 in
the repo, §6); what they cannot do is round-trip through `validate_receipt`
again. That is the honest state: they genuinely lack the input. The ledger
column reads `--` for them and the renderer prints the dash rather than
assuming a GPU.

### 1.2 Declared inapplicability, for the links that need a per-component predictor

`NOTES-MX-E` §4.5 measured one harness gap with four faces: the generic
`published_child` calibration gate assumes the component is a single 2-D linear
map, `held_input @ parent_weight.T`, and C03 (three stacked projection blocks),
C04 (FIR taps), C10 (four tensors behind a softmax) and C14 (a 1-D RMSNorm gain)
each violate that differently. All four cells died at chain link 1 with no
evidence at all about the technique.

The coordinator's decision, implemented: those cells declare the **local** links
inapplicable — with a reason — and rest their verdict on the links that need no
per-component predictor.

Two new mandatory flags, both additive, both fail-closed:

* `calibration.applicable` — when false, `first_divergent_metric` skips link 1
  (`matrix_cell_receipt.py:557`). Declaring it costs a nonempty
  `calibration.inapplicable_reason`; a bare `false` raises.
* `boundary.local_solve_applicable` — when false, link 3 (write cosine) is
  skipped while link 4 (macrocell band) is still walked (`:585`). Same reason
  requirement. The two boundary links nest: `local_solve_applicable` cannot be
  true inside a boundary that did not run, because that would claim a
  measurement inside a stage that never happened.

Neither flag is reachable from `campaign_pass`, whose clause is still behavior
plus the wreck detector and nothing else.

The exemption is a **closed pre-registered list of components**
(`run_lfm2_matrix_cell.py:125`, `NO_PARENT_MATMUL_PREDICTOR`), not a fallback
triggered by catching an exception. A shape mismatch for some other reason still
voids its cell loudly, and each of the four entries carries the wave-1
measurement that established it. Two further consequences are wired:

* when the technique's prediction and the fallback write endpoint disagree in
  width (C03: 3072 against 1024), the **whole boundary stage** is declared
  inapplicable with the two widths named, instead of crashing;
* the exception path still seals `applicable: true` — a technique that refuses,
  or a stage that raises, is a real `infra` outcome and must not inherit an
  exemption that was granted to one specific gate.

### 1.3 The rest: M2, T2/T3 adapters, and the storage columns

* **`M2` joined the level registry** (`matrix_cell_receipt.py:156`). The
  marginal family is `M_k = 1 − 1/k`, so M2 — keep one half — is its floor, and
  by `NOTES-MX-E` §4.1's arithmetic it is the exact break-even for any technique
  whose stored form is a `w × r` basis plus an `r × w` factor: `r·(in+out)`
  against a dense `in·out` is a contraction only when `r < in·out/(in+out)`. The
  renderer's ladder (`render_matrix_ledger.py:64`) walks M16 → M8 → M4 → M2 → X4.
* **T2 and T3 got registry adapters**, `src/neural_forge/matrix_technique_charts.py`
  — adapter-only, no new linear algebra. §5 has the two recorded deviations and
  §5.1 has what they measured.
* **Storage rides in the ledger.** `ledger_line` gained
  `stored_parameter_count`, `parent_parameter_count` and
  `stored_parameter_ratio` (`matrix_cell_receipt.py:956`), the runner writes a
  storage sentence into `outcome.notes` (`run_lfm2_matrix_cell.py:2099`), and
  the rendered report gained a `## 4b. Storage honesty` section and a
  `stored/dense` column in the drill-down. Wave 1's ledger had no column that
  would have shown a "contraction" storing 1.88× its parent.

**Tests.** `tests/test_matrix_cell_receipt.py` 25 → 31 (five new, one rewritten),
`tests/test_run_lfm2_matrix_cell.py` 20 → 24 + 1 skip (CUDA-only),
`tests/test_matrix_technique_charts.py` is new at 9. Full matrix suite:
**106 passed, 1 skipped**. `ruff` reports the same two pre-existing unused
imports it reports at HEAD and nothing new.

One pre-existing test changed meaning and was updated rather than deleted:
`test_most_specific_registration_wins` borrowed `T3` as a technique id nothing
registered, which stopped being true when T3's per-component adapters landed. It
now borrows `T6`, with the reason in the test.

---

## 2. The results

32 cells. Boundary/composed numbers are diagnostic; `free-run` and the review
are the only fields §5 rule 1 may read. **Grade** is ADDENDUM 3's interpretation
layer, which reads free-run and the review and nothing else. **stored** is
stored parameters over the dense parent — above 1.0 the fold costs more storage
than the tensor it replaced.

| # | cell | comp | tech | site | write cos | macro | vs control | free-run | **stored** | link | status | **grade** |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | `C06.T1.M16.L08` | attn_q | T1 | L08 | **1.00000** | *cat* | **0.00111** | **0.977** | 1.875 | — | **PASS** | strong |
| 2 | `C06.T1.M16.L12` | attn_q | T1 | L12 | **1.00000** | *cat* | 0.00148 | **1.000** | 1.875 | — | **PASS** | strong |
| 3 | `C07.T1.M16.L08` | attn_k | T1 | L08 | 0.99995 | *n/m* | 0.00950 | **0.936** | 1.406 | — | **PASS** | strong |
| 4 | `C07.T1.M16.L12` | attn_k | T1 | L12 | 0.99997 | *n/m* | 0.00541 | **0.994** | 1.406 | — | **PASS** | strong |
| 5 | `C06.T7.M16.L12` | attn_q (QK) | T7 | L12 | 0.99909 | *cat* | 0.01688 | 0.680 | 0.996 | free-run | fail | promising |
| 6 | `C06.T1.M8.L02` | attn_q | T1 | L02 | 0.99998 | *cat* | 0.00286 | **0.971** | 1.750 | — | **PASS** | strong |
| 7 | `C06.T1.M4.L02` | attn_q | T1 | L02 | 0.99983 | *cat* | 0.00820 | 0.871 | 1.500 | free-run | fail | promising |
| 8 | `C06.T1.M2.L02` | attn_q | T1 | L02 | 0.99815 | *cat* | 0.02573 | 0.609 | **1.000** | composition | fail | promising |
| 9 | `C07.T1.M8.L02` | attn_k | T1 | L02 | 0.99973 | *n/m* | 0.01272 | 0.793 | 1.312 | free-run | fail | promising |
| 10 | `C07.T1.M4.L02` | attn_k | T1 | L02 | 0.99894 | *n/m* | 0.02363 | 0.631 | 1.125 | composition | fail | promising |
| 11 | `C07.T1.M2.L02` | attn_k | T1 | L02 | 0.99439 | *n/m* | 0.04853 | 0.529 | **0.750** | composition | fail | weak |
| 12 | `C06.T7.M8.L02` | attn_q (QK) | T7 | L02 | 0.99677 | *cat* | 0.01904 | **0.916** | 0.930 | — | **PASS** | strong |
| 13 | `C06.T7.M4.L02` | attn_q (QK) | T7 | L02 | 0.99034 | *cat* | 0.04274 | 0.547 | 0.797 | composition | fail | weak |
| 14 | `C09.T1.M8.L02` | attn_o | T1 | L02 | 0.99958 | 0.0155 | 0.02516 | 0.736 | 1.750 | composition | fail | promising |
| 15 | `C09.T1.M4.L02` | attn_o | T1 | L02 | 0.99767 | 0.0374 | 0.05726 | 0.445 | 1.500 | composition | fail | weak |
| 16 | `C09.T1.M16.L08` | attn_o | T1 | L08 | 0.99748 | 0.0305 | 0.04936 | 0.584 | 1.875 | composition | fail | weak |
| 17 | `C08.T1.M8.L02` | attn_v | T1 | L02 | 0.99788 | *n/m* | 0.02441 | 0.650 | 1.312 | composition | fail | promising |
| 18 | `C08.T1.M4.L02` | attn_v | T1 | L02 | 0.99333 | *n/m* | 0.03885 | 0.553 | 1.125 | composition | fail | weak |
| 19 | `C05.T1.M8.L05` | conv_out | T1 | L05 | 0.99236 | 0.0376 | 0.05392 | 0.746 | 1.750 | composition | fail | promising |
| 20 | `C05.T1.M2.L05` | conv_out | T1 | L05 | 0.94817 | 0.0991 | 0.11851 | 0.246 | **1.000** | local-solve | fail | weak |
| 21 | `C05.T5.M8.L05` | conv_out | T5 | L05 | 0.99169 | 0.0382 | 0.05473 | 0.664 | **0.875** | composition | fail | promising |
| 22 | `C05.T2.M16.L05` | conv_out | T2 | L05 | 0.99684 | 0.0239 | 0.03866 | 0.744 | 1.875 | composition | fail | promising |
| 23 | `C05.T3.M16.L05` | conv_out | **T3** | L05 | 0.99747 | 0.0220 | 0.02677 | **0.914** | 1.875 | composition | fail | **strong** |
| 24 | `C09.T2.M16.L02` | attn_o | T2 | L02 | 0.99989 | 0.0072 | 0.01308 | 0.820 | 1.875 | free-run | fail | promising |
| 25 | `C13.T5.M16.L05` (k=16) | mlp_down | T5 | L05 | 0.99347 | 0.0403 | 0.05746 | 0.430 | **0.938** | composition | fail | weak |
| 26 | `C13.T1.M16.L13` | mlp_down | T1 | L13 | 0.99751 | 0.0266 | 0.04723 | 0.428 | 3.281 | composition | fail | weak |
| 27 | `C13.T2.M16.L05` | mlp_down | T2 | L05 | 0.99485 | 0.0350 | 0.06289 | 0.324 | 3.281 | composition | fail | weak |
| 28 | `C13.T3.M16.L05` | mlp_down | **T3** | L05 | 0.99222 | 0.0437 | 0.06250 | 0.586 | 3.281 | composition | fail | weak |
| 29 | `C14.T1.NA.L00` | norms | T1 | L00 | **1.00000** | *cat* | **0.00010** | **1.000** | 0.500† | — | **PASS** | strong |
| 30 | `C03.T5.M16.L00` | conv_in_proj | T5 | L00 | *n/a* | *n/a* | 0.02758 | 0.787 | **0.938** | composition | fail | promising |
| 31 | `C04.T1.M16.L00` | conv_kernel | T1 | L00 | — | — | — | — | — | infra | **infra** | — |
| 32 | `C10.T1.M16.L02` | attn_per_head | T1 | L02 | — | — | — | — | — | infra | **infra** | — |

*n/m* = the write is narrower than the residual and the runner records the
`#not-composable-from-write` sentinel. *cat* = dimensionally composable and
mechanically a category error (§3.4). *n/a* = the boundary stage was declared
inapplicable (§4). † = a bookkeeping artifact, not a compression (§4.1).

**Grades:** 8 strong, 12 promising, 10 weak, 0 dead, 2 infra. Every one of the
30 cells with decode-smoke text was reviewed **coherent** from the verbatim
receipt; two carried a recorded semantic drift (§3.5). Every calibration relL2
that was measured landed in `0.000638 – 0.002282` against the 5e-3 tolerance.

---

## 3. What the cells said

### 3.1 Rule 2: do the attention passes hold across sites? Yes, both.

§5 rule 2 needs a pass at the level *and every shallower level* at ≥2 of 3
sites. Wave 1 had one site each for C06/T1 and C07/T1 and the rendered ledger
was already calling them "works to M16" off it. Wave 2 ran the other two.

| technique×component @ M16 | L02 | L08 | L12 | rule 2 |
|---|---:|---:|---:|---|
| C06 attn_q / T1 | 0.977 | **0.977** | **1.000** | **3 of 3 — holds** |
| C07 attn_k / T1 | 0.941 | **0.936** | **0.994** | **3 of 3 — holds** |
| C06 attn_q (QK) / T7 | 0.920 | 0.846 | **0.680** | **1 of 3 — does not hold** |

Both dense-ridge attention readers now clear rule 2 at M16 outright, and the
deepest site is the *best* one: `C06.T1.M16.L12` reproduces the copy_exact
control's probe token stream **exactly** — free-run 1.000, 512 of 512 tokens —
and `C07.T1.M16.L12` misses three tokens in 512. That is the campaign's first
technique×component that is not a single-site anecdote.

T7 goes the other way and now has all three sites: it passes only where it was
first tried, and it degrades monotonically with depth — 0.920 at L2, 0.846 at
L8, 0.680 at L12 — while its composed drift stays small (0.0112 / 0.0094 /
0.0169). The QK score-kernel relation is a shallow-layer result.

The rendered ledger's `works to M8` for C06/T1 should be read with its
arithmetic in view: M8 was run at **one** site (L02). One site is not two, and
the renderer's ladder credits `1/1`. The rule-2 obligation for C06/T1 at M8 is
L08 and L12, and it is a wave-3 item, not a result.

### 3.2 The M-ladder: behaviour is flat, then it cliffs

Every rung at one site, so nothing here is a cross-site comparison.

| component / technique @ site | M16 | M8 | M4 | M2 |
|---|---:|---:|---:|---:|
| C06 attn_q / T1 @ L02 — free-run | 0.977 | 0.971 | 0.871 | 0.609 |
| — composed drift | 0.00124 | 0.00286 | 0.00820 | 0.02573 |
| C07 attn_k / T1 @ L02 — free-run | 0.941 | 0.793 | 0.631 | 0.529 |
| — composed drift | 0.00743 | 0.01272 | 0.02363 | 0.04853 |
| C06 attn_q (QK) / T7 @ L02 — free-run | 0.920 | 0.916 | 0.547 | — |
| C08 attn_v / T1 @ L02 — free-run | 0.738 | 0.650 | 0.553 | — |
| C09 attn_o / T1 @ L02 — free-run | 0.820 | 0.736 | 0.445 | — |
| C05 conv_out / T1 @ L05 — free-run | 0.744 | 0.746 | — | 0.246 |
| C05 conv_out / T5 @ L05 — free-run | 0.791 | 0.664 | — | — |

Three things fall out of this and none of them were visible at one level.

**Composed drift is close to geometric in the level; behaviour is not.** C06/T1
multiplies its drift by 2.3, 2.9, 3.1 across the four rungs and C07/T1 by 1.7,
1.9, 2.1 — a smooth, near-exponential response to width. Free-run does something
else entirely: C06/T1 gives up **0.006** going M16 → M8 and then **0.362** going
M8 → M2. The first rung is nearly free and the last is a cliff.

**Where the cliff sits is component-specific, and it tracks how good the M16 cell
was.** C06/T1 starts at 0.977 and survives to M8. C07/T1 starts at 0.941 and
loses 0.148 on the same rung. C09/T1 starts at 0.820 and is at 0.445 by M4. A
component that is merely *promising* at M16 has no ladder left.

**T5 and T1 swap places down the ladder at the conv writer.** At M16, T5 leads
(0.791 vs 0.744); at M8, T1 leads (0.746 vs 0.664) and T1 has not moved at all
from its own M16. The wave-1 verdict that the two are a behavioural tie at this
component survives a second level, and the tie is noisy in both directions.

### 3.3 The MLP wall: the rescue made it worse

Four probes at `mlp_down`, the component `NOTES-MX-E` §4 named the campaign's
wall. Every one fails at composition, with composed drift 0.047–0.063 against a
0.02 gate.

| probe | site | free-run | vs control | stored | versus |
|---|---|---:|---:|---:|---|
| T5, OMP k **doubled** to 16 | L05 | **0.430** | 0.05746 | 0.938 | wave 1's k=8: **0.502**, drift 0.05593 |
| T1 at the unrun site | L13 | 0.428 | 0.04723 | 3.281 | T5 at L13: 0.561 |
| T2 | L05 | 0.324 | 0.06289 | 3.281 | T1 at L05: 0.383 |
| T3 | L05 | **0.586** | 0.06250 | 3.281 | best C13 cell in the campaign |

**Doubling the route budget is a regression.** Same component, same site, same
evidence, same GPU, one knob: `active_k` 8 → 16. The write cosine improves in
the fourth decimal (0.99336 → 0.99347), the composed drift gets *worse*
(0.05593 → 0.05746), and free-run drops 0.072. More support per row buys a
marginally better local fit and a marginally worse child. The MLP wall is not a
route-budget problem, and this is the cheapest possible demonstration of it.

**T3 is the new C13 record and it is still nowhere near.** 0.586 beats every
previous mlp_down cell (T5's 0.561 at L13, 0.502 at L05; T1's 0.383; T8's
0.199) and is still 0.31 short of the goal and 0.014 short of even the
*promising* band. Four techniques with four different objectives all land in
0.32–0.59 at drift 0.047–0.063. The wall is a property of the component.

### 3.4 The S8 band gets its first genuine agreement, in 31 measured cells

`NOTES-MX-E` §4.4 established that the rendered agreement number must not be
read, because most of its rows are sentinels or category errors. The same
classification applied to wave 2:

| class | wave-2 cells | what the reading is |
|---|---:|---|
| **measured** | 13 | C05 ×5, C09 ×4, C13 ×4 — `residual.post_layer` composed from a write that really is added into the residual |
| category error | 9 | C06 ×8 (`q_linear` happens to match the residual width and enters through a softmax, not additively) and **C14** ×1 (`operator.input` is the operator's normalized input, not a residual write) |
| sentinel | 6 | C07 ×4, C08 ×2 — write narrower than the residual |
| none | 4 | C03 (boundary declared inapplicable) + C04/C10 (never reached it) |

On the 13 genuinely measured cells the band is in-band 13 times and **behaved
once** — `C05.T3.M16.L05`, macrocell 0.0220, free-run 0.914. Folded into wave
1's 18: the campaign's §5 rule-4 tally is now **in band and behaved 1, in band
and did not behave 30, out of band 0**. The band has still never been observed
to fail. It has now predicted correctly once in 31 tries, which is a positive
predictive value of 3% and is not a criterion.

C14 is worth calling out because it is the *third* way to manufacture a
plausible-looking band reading: the receipt shows 0.0342 and the endpoint
composes dimensionally, but `operator.input` is not a residual write, so the
number is arithmetic on the wrong pair. It joins C06's q_linear in the
category-error column rather than the measured one.

### 3.5 Smaller things the receipts said

* **The composed gate is not comparable across components.**
  `C05.T3.M16.L05` drifts 0.02677 and behaves at 0.914; `C09.T1.M8.L02` drifts
  0.02516 — *less* — and behaves at 0.736. Nearly identical composed-terminal
  numbers, 0.18 apart on the only metric §5 rule 1 reads. `NOTES-MX-E` §4.6
  found the composed ordering does not survive across sites; it does not survive
  across components either, and the 0.02 gate is a per-component threshold
  wearing a global number.
* **`C05.T1.M2.L05` is the wave's only local-solve failure**, write cosine
  0.94817 against the 0.95 gate — 0.0018 short. Halving the conv writer's input
  width is the first contraction in the campaign that the *local* fit could not
  absorb.
* **Two smoke outputs drifted semantically without degenerating**, and both were
  reviewed coherent with the drift recorded: `C09.T1.M4.L02` inverts the trailer
  note (the writer moved the trailer and wants to relocate it) and
  `C06.T7.M4.L02` sends it "to a nearby park". Neither loops, both terminate on
  EOS, both keep the four-line format. Wave 1's only `incoherent` verdict was a
  forty-fold repetition; the rubric is applied the same way here, and these two
  cells are `weak` on free-run alone regardless of the verdict.
* **`C06.T1.M16.L12` is the strongest cell in the campaign.** Write cosine
  1.00000, composed drift 0.00148, free-run 1.000, and all three decode-smoke
  outputs read as the control's. It is also, at 1.875× storage, one of the most
  expensive.
* **Wall clock**: 32 cells in 14 minutes on two GPUs, longest cell 213s
  (`C13.T3.M16.L05`, whose 2560-wide balanced canonicalization is an O(r²)
  Gram-Schmidt). The 20-minute hard cap was never approached.

---

## 4. The four void cells, resolved

Under §1.2's treatment, run again on the same evidence:

| cell | wave 1 | wave 2 | what changed |
|---|---|---|---|
| `C14.T1.NA.L00` norms | `infra` — `size of tensor a (1584) must match tensor b (1024)` | **PASS**, free-run **1.000**, composed drift **0.00010** | the gate was declared inapplicable; the C14 kernel's own predictor scored the boundary and the child reproduced the control exactly |
| `C03.T5.M16.L00` conv_in_proj | `infra` — `size of tensor a (3072) must match tensor b (1024)` | **fail at composition**, free-run 0.787, drift 0.02758, **stores 0.938×** | gate inapplicable *and* boundary inapplicable (no nameable stacked write endpoint); the verdict came from the composed forward and the probe |
| `C04.T1.M16.L00` conv_kernel | `infra` — `Expected size … [3, 1024] but got [3, 1]` | `infra` — **`T1's reference kernel contracts linear maps; C04 has axis 'kernel'`** | the harness now reaches the technique, and the technique refuses |
| `C10.T1.M16.L02` attn_per_head | `infra` — calibration relL2 **4.4955**, refused | `infra` — **`T1's reference kernel contracts linear maps; C10 has axis 'heads'`** | same |

**Two of four are now evidence.** N10 — the standalone norm refit nobody had
ever measured — is a **pass**, and it is the campaign's first non-attention
pass. `conv_in_proj` has its first number ever, and it is the best conv-family
free-run in the wave outside T3.

**Two of four moved from a harness reason to a technique reason.** That is the
whole point of the treatment and it is not a smaller result than a number: wave
1 could not say whether C04 and C10 were hard or unsupported. They are
unsupported — no kernel is registered for the `kernel` (FIR taps) or `heads`
axes, T1 refuses both by name and T5 refuses them in the same words. Building
those two kernels is a technique-lane item; inventing them mid-wave would have
produced a confident cell nobody could defend. Per the campaign's own rule, a
technique refusal is a sealed result, and these two are sealed with the refusal
text in `outcome.notes`.

### 4.1 The C14 storage number is a bookkeeping artifact, not a compression

`C14.T1.NA.L00` reports `stored 1024 / parent 2048`, ratio 0.5000, and the
rendered storage section counts it among the eight cells that contract. It
should not be read that way. `component_spec` gives C14 **both** layer norms
(`operator_norm` and `ffn_norm`, 1024 each) and the standalone refit kernel
installs only `weight_keys[0]`. The 0.5 is "one of two tensors", not a
contraction — the level is `NA` precisely because norms are diagonal and have no
width to contract. The pass is a statement about *refit quality*, which is what
N10 was always asking, and nothing about storage.

---

## 5. T2 and T3: first data, and one of them is not a technique here

Both were wired as adapters over kernels this estate already owned
(`src/neural_forge/matrix_technique_charts.py`). Two deviations, recorded in the
module's own docstring:

1. **T2's encoder is the campaign's POD, not the estate's balanced POD.**
   `compile_residual_boundary_chart_set` selects its encoder with
   `build_balanced_circuit_basis`, which needs the parent's residual-facing
   operator inventory, a child `DecoderLayout` and one snapshot block per
   evidence split — objects that exist at whole-crossing scope, not at
   one-component-one-site scope. Rebuilding them inside a cell would be a second,
   unsealed implementation of the crossing compiler. Using
   `matrix_technique_registry._pod_basis` instead makes T2's basis *identical to
   T1's* at the same component/level/site, which turns T2-vs-T1 into a
   controlled comparison of exactly one thing: how the second factor is made.
   Everything downstream — `_fit_decoder`, `_boundary_defects`, `_seal_chart`,
   `ResidualBoundaryChart.verify` — is the estate's, unmodified, and the chart's
   own defect bounds still fail the cell closed.
2. **T3 covers the input axis only.** Balanced truncation needs a
   controllability lane and an observability lane over one shared state. For a
   writer both exist inside the cell: the state is the component's input, the
   sealed rows are its controllability evidence, and the parent weight *is*
   `∂output/∂state`, one row per output channel. For a reader the observability
   operator is the downstream consumer — the attention score kernel at C06/C07 —
   which a single-component cell does not own. The reader arm refuses with that
   reason rather than substituting an identity observability, which would reduce
   balanced truncation to plain POD and produce a confident duplicate of T1.

### 5.1 T2 is T1's basis with a decoder that measures out to the transpose

The adapter seals a diagnostic for exactly this:
`chart_decoder_vs_encoder_transpose_relative_deviation`. On real evidence it
reads **8.6e-9** at C05, **9.2e-9** at C09 and **1.0e-8** at C13. The regularized
reciprocal of an orthonormal encoder *is* its transpose, and the cycle-anchored
ridge with weight 1e-2 does not move it, so the chart round-trip is an orthogonal
projection and T2's fold is a POD truncation with no refit.

The cells say the same thing without the diagnostic:

| site | write cos | macrocell relL2 | vs control | free-run |
|---|---:|---:|---:|---:|
| C05 @ L05 — T1 (wave 1) | 0.9968404968 | 0.0239094543 | 0.03870 | 0.744 |
| C05 @ L05 — **T2** | 0.9968404968 | 0.0239094543 | 0.03866 | 0.744 |
| C09 @ L02 — T1 (wave 1) | 0.9998917654 | 0.0072151982 | 0.01322 | 0.820 |
| C09 @ L02 — **T2** | 0.9998917654 | 0.0072151982 | 0.01308 | 0.820 |
| C13 @ L05 — T1 (wave 1) | 0.9948463402 | 0.0349558729 | 0.06287 | 0.383 |
| C13 @ L05 — **T2** | 0.9948463402 | 0.0349558729 | 0.06289 | 0.324 |

Boundary numbers identical to **ten significant digits** at all three
components; composed drift differing in the fourth; free-run identical at two of
three and 0.059 apart at the third, which is the decode divergence
`NOTES-MX-C` documented amplifying a fourth-decimal difference.

**Verdict: at a body component with parent-width neighbours, T2 is not a distinct
technique.** The design's S1 result is about a *residual crossing*, where the
whole boundary is charted and the encoder is a balanced POD selected against
downstream observability. A single-component cell charts one endpoint and hands
the result back to parent-width consumers, and in that setting the chart has
nothing to do that the projection does not already do. This is a real campaign
output — "T2 untested" becomes "T2 measured, and it collapses onto T1 at three
components" — and it retires nine matrix positions cheaply.

Two further honest notes: T2's stored form is the encoder plus the child factor,
`r·(in+out)`, exactly T1's; and its decoder is spent at fit time and never
installed, so the adapter reports it separately as
`chart_fitting_only_parameters` rather than counting it as storage or pretending
it was free.

### 5.2 T3 is the wave's best technique finding

`C05.T3.M16.L05` is the only non-attention cell in the campaign to reach the
0.90 band: **free-run 0.914**, against T1's and T2's 0.744 and T5's 0.791 at the
same component, level and site. Same rank, same rows, same storage
(`r·(in+out)`, 1.875×). The only difference is which subspace is kept.

The receipt says what the trade was:

| cell | fit controllability energy retained | observability energy retained | free-run |
|---|---:|---:|---:|
| `C05.T3.M16.L05` | 0.99628 | 0.96370 | **0.914** |
| `C13.T3.M16.L05` | 0.99955 | 0.93500 | 0.586 |

T3 gives up a few tenths of a percent of input energy — which is all a POD
optimizes — to keep the directions the output is actually sensitive to, and at
`conv_out_proj` that is worth **0.17 free-run** over the POD techniques and at
`mlp_down` **0.20** over T1. It is the first evidence in this campaign that the
basis *criterion*, not the basis rank, is where the behavioural headroom is.

It still fails: composed drift 0.02677 against a 0.02 gate, so the cell stops at
chain link 5. Under ADDENDUM 3 it grades **strong**, and by ADDENDUM 1's
selection rule it is `conv_out_proj`'s winner for the composition endgame.

Caveats, stated because the number is attractive: one site, one level, one
component family, and the reader arm of T3 does not exist. Its rule-2 obligation
(L00 and L13) is unrun.

---

## 6. Ledger, backups and the destructive-change check

Two runners, two shards, one serialized appender — the same discipline wave 1
established, for the same reason (`append_ledger` issues two `write()` calls per
row and `O_APPEND` makes each atomic, not the pair).

```
rows 64 -> 96 (+32)      cells, merged from ledger-a.jsonl + ledger-b.jsonl
prefix verified byte-for-byte
rows 96 -> 126 (+30)     reviews, sealed from one process
```

**Five wave-2 cells re-run a cell id wave 1 had already sealed** — the four void
cells and the C13.T5 rescue probe. The ledger keeps both instances by design;
the *files* would have collided, so the batch driver detects an existing
`cell-receipt.json` and writes the re-run into a `wave2/` subdirectory beside
wave 1's artifacts. No wave-1 receipt was opened for writing, and `git status`
shows exactly two modified files in the report tree: `ledger.jsonl` and
`LEDGER.md`.

**One concurrent edit, noted and not touched.** Codex added 431 lines to
`src/neural_forge/matrix_technique_body_routes.py` and 307 to its test file in
the working copy while this wave ran — a sparse-feature-fold compiler boundary
on top of the sealed T5 record, purely additive (0 deletions, `t5_body_route_technique`
unchanged). It cannot have touched a single wave-2 number: the runner executes
from the separate deployment tree, whose copy of that file still digests
`055082c8a081317c`, exactly the value `NOTES-MX-E` §1 recorded. Its suite and
the two neighbouring technique suites were run as a courtesy check against the
registry changes in §1.3 — `tests/test_matrix_technique_{body_routes,lowrank_writers,relation_matching}.py`,
**75 passed** — and nothing in that lane was edited.

| what | where |
|---|---|
| deployment + ledger before the wave | `/mnt/ai_data/neural-forge-matrix/backup/deployment-before-mxf-20260810T191828Z/` — `matrix_cell_receipt.py` (`4f72ca3b0a817cb3`), `run_lfm2_matrix_cell.py` (`e34a8bdfea23fb34`), `render_matrix_ledger.py` (`13715e6883322cc3`), `ledger.jsonl.before-mxf` (`ef1b271a8c7a5c1f`) |
| ledger before the shard merge | `…/backup/ledger.jsonl.before-mxf-merge-20260810T194223Z` |
| ledger before the 30 review appends | `…/backup/ledger.jsonl.before-mxf-reviews-20260810T194441Z` |
| repo mirror ledger + LEDGER.md, pre-wave | scratch (`ledger.jsonl.before-mxf` `ef1b271a8c7a5c1f`, `LEDGER.md.before-mxf` `2513c9090bae7493`) |

| check | result |
|---|---|
| ledger rows | 64 → 96 (cells) → 126 (reviews); **no row rewritten**, prefix verified after every append |
| mirror vs authority | byte-identical, `147656cb399013b3`; the pre-wave mirror is a byte prefix of it |
| receipt seals re-verified | **123 / 123** on the server, **124 / 124** in the repo, no failures |
| deployed file digests vs local | 4/4 identical after upload (`9c9d15ada699c4e5`, `92164bdfb9249a69`, `c5039a2a26ecbb5a`, `f903dc53dedd538a`) |
| sealed evidence base `2dec87ac…` | opened read-only; `find … -newermt` over the sealed tree returns nothing written during the wave. Four new per-(site, component) caches were built through the runner's authenticated selective session (L12-C06, L08-C07, L12-C07, L08-C09); the other 28 cells reused an existing cache only after re-deriving its payload digest against the recorded binding |
| compute architecture in every wave-2 receipt | **32 / 32** `sm86 NVIDIA GeForce RTX 3090`; zero P40 rows in any ledger |
| pilot runs | 5 cells were run into `/home/cereal/nf-mxf/pilot/` with their own ledger before the batches, to prove the new treatment and the new kernels; that ledger was **not** merged |

---

## 7. Verification

```
# local, .venv (python 3.11.9)
python -m pytest tests/test_matrix_cell_receipt.py tests/test_run_lfm2_matrix_cell.py \
                 tests/test_matrix_technique_registry.py \
                 tests/test_run_lfm2_matrix_cell_fresh_evidence.py \
                 tests/test_matrix_technique_charts.py -q
# -> 106 passed, 1 skipped        (87 before this wave; +5 receipt, +4 runner, +9 charts, +1 skip)

python -m ruff check experiments/run_lfm2_matrix_cell.py experiments/render_matrix_ledger.py \
                     src/neural_forge/matrix_cell_receipt.py \
                     src/neural_forge/matrix_technique_charts.py tests/test_matrix_*.py
# -> 2 errors, both pre-existing at HEAD

python experiments/render_matrix_ledger.py --ledger reports/matrix-campaign-20260810/ledger.jsonl
# -> 126 rows render; the repo's LEDGER.md is the server render, byte-identical to the authority
```

Server: the four-file deployment digest check, a 5-cell pilot of the new
treatment and the new kernels, 32 cell runs across two SM86 GPUs, the shard
merge, 30 review seals, the ledger render, 123/123 seal re-verification, and the
free-space and read-only-evidence checks.

**Not verified.** No X4 cell anywhere. C06/T1 at M8 has one site, not two, so its
`works to M8` is not rule-2 complete. T3 has one site at each of two components
and no reader arm. T2's collapse onto T1 is measured at three components and is
not proven for the token interface, which is where S1's result actually lives.
C04 and C10 still have no technique kernel and therefore still no evidence. No
composition-mode cell of any kind — ADDENDUM 4 is explicitly a later wave and
nothing here licenses a multi-component claim (§5 rule 6).

## 8. Paths

* wave-2 cells: `reports/matrix-campaign-20260810/cells/<cell-id>/` (27 new
  directories) and `…/cells/<cell-id>/wave2/` (5 re-runs: `C03.T5.M16.L00`,
  `C04.T1.M16.L00`, `C10.T1.M16.L02`, `C13.T5.M16.L05`, `C14.T1.NA.L00`)
* ledger + rendered report:
  `reports/matrix-campaign-20260810/{ledger.jsonl,LEDGER.md}` (mirrors
  `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`)
* run inputs: `reports/matrix-campaign-20260810/run/{mxf-knobs.json,mxf-knobs-t5-k16.json,run-mxf-batch.sh,mxf-reviews.json,mxf_summarize.py,mxf_dump_smoke.py,mxf_seal_reviews.py}`
* harness changes: `src/neural_forge/matrix_cell_receipt.py`,
  `experiments/run_lfm2_matrix_cell.py`, `experiments/render_matrix_ledger.py`
* new technique adapters: `src/neural_forge/matrix_technique_charts.py` with
  `tests/test_matrix_technique_charts.py`
* campaign store: `/mnt/ai_data/neural-forge-matrix/cells/`, run logs
  `/home/cereal/nf-mxf/logs/cell-*.log`, batch logs
  `/home/cereal/nf-mxf/batch-{a,b}.log`, backups
  `/mnt/ai_data/neural-forge-matrix/backup/` (see §6)
