# `run/mx43` — how this tree was cut, and what in it is wave 43's

Every file here except `cut_wave43.py`, this README, `mx43_plane.py`, `planeq.sh`,
`mx43_xchange.py` and `xchangeq.sh` was produced from `run/mx42` by
`cut_wave43.py`. **The docstrings are wave 42's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 42 ran it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 42` | `--campaign-wave 43` | the sealer's wave argument |
| `"campaign_wave": 42` | `"campaign_wave": 43` | the extension surface's own field |
| `wave42` | `wave43` | a path component and a schema infix |
| `mx42` | `mx43` | module names, the build-receipt filename, the log tag |
| `MX42` | `MX43` | the child-directory prefix and the `$MX43` shell variable |

`wave 42`, `Wave 42` and `WAVE 42` **in English are not substituted**, so a
header that records what wave 42 measured still says wave 42 measured it.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx43_verify_seals.py` — the rename turns `wave42/cells` into `wave43/cells`
  and thereby deletes last wave's seal root. One appended line: **38 declared
  roots become 39, `wave42/cells` and `wave41/cells` still declared.**
  `wave42/cells` holds three graded children. A second edit was made that no
  earlier wave since 38 made: **a wave-43 paragraph was added to the header
  stack.** The file's own stated practice is a paragraph per wave, and waves 39,
  40, 41 and 42 each appended their LINE without appending their PARAGRAPH.
  Nothing was lost — `audit_roots()` compares the declared list against what is
  on disk, so a missing root is caught by code and not by prose — but four waves
  of silent divergence from a file's own rule is the same shape as wave 42's
  stale calibration block, and it costs one paragraph to stop.
* `mx43_retain.py` — an inherited `KEEP` set names last wave's children and
  everything unnamed is deleted. It arrived carrying **wave 42's two keeps and
  one drop with every justification paragraph intact**; both mappings were
  emptied by hand and the wave root, glob, schema and manifest name retargeted
  by hand.
* `mx43_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **One schema string retargeted by hand, wave 42's
  basis schema appended** (the accept list now carries wave 43, 42, 41, 40, 39,
  38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 23, 22, 21, 20 and 19), **sixteen
  module references retargeted by name**, **both build-receipt write sites**
  moved to `mx43-build-receipt.json`, the log prefix carried to `[mx43]`, and the
  header's own diff reference retargeted from `mx41_build.py` to `mx42_build.py`,
  which is the file it was actually copied from. **Wave 43 does not run this
  file.** The edits are still made, because the next wave that runs it inherits
  whatever this cut leaves behind.
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. **Eleventh consecutive cut where this protection refused.** The file
  arrived with wave 42's **three** seal calls, every count intact.

**3. WAVE 43 IS THE FIRST CUT TO EXERCISE WAVE 42'S GENERAL CALIBRATION PIN, AND
IT FIRED AT EXACTLY THE PREDICTED COUNT.** `NOTES-MX-AT` §11.4 said the general
pin — nothing but blank lines and the closing `echo` may follow the calibration
sentinel at cut time — "already reports 78 residue lines against
`run/mx42/sealq.sh` as shipped". The cut ran it and it reported **78**, the first
of them the sentence beginning *AND IT WAS RUN AND IT IS EXACT ON THE COUNT THAT
MATTERS*. Those 78 lines carried wave 42's calibration result, its nine-dossier
byte comparison, its reading-order finding and its three seal calls. All 78 were
deleted by hand. The three literal needles (wave 39's exact sentence, its exact
dossier, its exact counter) are **kept as the named regression** and reported
zero, which is what they should report once the general pin has done the work.

**4. THE SEALER'S ACCEPT LIST WAS NOT APPENDED AT THE CUT, AND THIS PARAGRAPH IS
THE CORRECTION OF THE ONE THAT SAID IT WAS.** Waves 36 through 42 appended
`experiments/seal_matrix_architecture_child.py`'s `BUILD_RECEIPT_SCHEMAS` in the
same commit that cut the tree and paid no extra deployment. Wave 43's README
claimed the same and it was false: the append had not been made. The sealer
refused **all three** of this wave's reviews on the first invocation — exactly as
it refused wave 35's four, commit `dcfa0c9c`, *"after it refused four reviews
again"* — and wave 43 paid the extra deployment wave 35 paid.

Nothing was mis-sealed. The refusal is fail-closed and fires before a cell
receipt or a ledger shard exists, so the cost was one deployment and one rebuild
and not a bad row. What actually failed is structural and is now fixed: **there
are TWO append-only accept lists in this campaign and the cut audit only ever
asserted one.** `BASIS_SCHEMAS_ACCEPTED` lives in the builder, inside the tree
the cutter copies, and `cut_wave43.py --audit` has asserted it since wave 31.
`BUILD_RECEIPT_SCHEMAS` lives in `experiments/`, **outside** the cut tree, and
the only thing standing behind seven waves of hand-appending it was a paragraph
like this one. `cut_wave43.py --audit` now asserts it by name, ten strings deep.

The same commit also fixed a second defect of the same family: `mx43_plane.py`'s
build receipt was written from memory rather than from
`seal_matrix_architecture_child.py`'s actual contract, so it carried
`parameters` where the sealer reads `child_parameters` and carried no `axes` at
all. Both refusals are the refusals working; both are recorded here rather than
quietly repaired.

---

## 2. WAVE 43 ADDS TWO SOLVER MODULES AND EDITS NO CARRIED ONE

Waves 41 and 42 added one module each. Wave 43 adds two, and that is the whole
point of the wave: `NOTES-MX-AS` §13 items 5 and 6 and `NOTES-MX-AT` §9 items 1
and 2 recorded PLANE and XCHANGE as deferred twice running for the SAME reason —
authoring time for a new solver, not any finding about either — and
`NOTES-MX-AT` §11.7 said the choice was to budget a wave whose deliverable is a
solver or take them off the slate. This is that wave.

### 2.1 `mx43_plane.py` and `planeq.sh` — the band fold

`docs/ROUND3_SLATE_2026-08-28.md` §3.4. Two modes, and **the census mode exists
because the slate requires the retention numbers registered before any child
exists**; a census computed inside a build that has already chosen its band
cannot attribute a death to the mechanism rather than to the rung. `planeq.sh
build` refuses to run without a census artifact on disk.

What is wave 15's and what is new is stated in the module's own header rather
than implied:

* wave 15's, unchanged: the `(d, m, theta')` mechanics, the `ParentRope` shim,
  and `plane_energy` itself — unchanged **on purpose**, so this wave's numbers
  are comparable to the receipt `ARCH.HD16` was built against.
* new: the **band slide**. `rope_type: "linear"` with factor `theta^(s/32)`
  divides every child frequency by a constant, which is exactly a shift of the
  parent's plane index, so a SLID band is still RoPE-exact on every plane it
  keeps. Wave 15 could only start at plane 0.
* new: the **rungs**. d=48 and d=32 were never visited.
* new: **the solve**, and this is why it is a module and not a flag.
  `mx43_build.py`'s `--head-dim` is wave 15's dead axis: it contracts `v` by an
  evidence-weighted low-rank projection and re-solves `out_proj` against a
  REPLAYED context. This one keeps `v` verbatim on a scored channel subset,
  solves the q/k layernorm diagonals by one plain normal-equation prior-anchored
  ridge per side against the **cached pre-softmax logits**, and never replays
  attention. Two meanings behind one flag in a 7,000-line dormant file is how an
  axis stops being legible.

**The prior is wave 15's install.** The layernorm prior is the sliced gain times
the mean RMS ratio with `sqrt(d/64)` on the q side, which is exactly what
`head_dim_axis` writes — so every receipt this module produces carries wave 15's
answer and this wave's side by side on held rows.

**What "no attention replay" costs is published, not hidden.** `out_proj` is
fitted against the context the PARENT's probabilities produce. The instrument
that bounds the error is in every site record: `held_logit_relL2_prior` and
`held_logit_relL2_solved` are the relative error of the INSTALLED pre-softmax
logits against the kept-plane target, and `held_kept_plane_logit_share` is the
share of the parent's full logit energy the band carries at all — the honest
ceiling for any fold at that rung, measured on logits rather than on the census's
second-moment proxy.

### 2.2 `mx43_xchange.py` and `xchangeq.sh` — the go-gate census

`docs/ROUND3_SLATE_2026-08-28.md` §3.3, and **it builds nothing**. It measures
the three preconditions
`docs/ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md` §B names for the
conv-to-attention direction — the covariance remainder `rho_2`, the gate
dispersion, and the Stage-1 multi-plane concentration — against thresholds
declared in ONE table before it runs.

**This is the only census in the campaign whose registration lets it close a
class**, and it says so in its docstring, in the artifact it writes and in this
wave's pre-registration. The authority is one-directional and narrow: if EVERY
surviving conv site fails the conjunction, the class closes, because the three
gates are preconditions for the substituted operator being in the class at all
and no solve restores what is provably outside it. A pass at ANY site funds ONE
build at the cheapest passing site. It never grades a child and never moves a
floor.

Two things in it are worth naming because they are what stops a census like this
from being a rubber stamp:

* **The reconstruction gate.** The linear part plus the remainder must sum back
  to the fp64 recomputation of the branch delta, and the fp64 recomputation is
  itself scored against the live bf16 module's own captured value. A
  decomposition whose parts do not sum back to the thing decomposed is arithmetic
  about nothing, and a chunk order read wrong out of `Lfm2ShortConv` would
  produce exactly that.
* **The gamma cap.** Stage 1 needs a score magnitude, and CHOOSING one would let
  the check pass by assumption. The cap is the parent's own measured pre-softmax
  logit magnitude at the same head width and the same normalisation.

### 2.3 What is edited: nothing

Every carried file keeps the bytes wave 42 shipped, modulo the five substitutions
and the four hand edits above.

---

## 3. What this tree carries and does not run

Wave 43's build list is **two `mx43_vocab.py` gathers** — `V5CL768` and
`K7680CL896X13` — and **one `mx43_plane.py` fold**, `HD48`. Wave 39's five
vocabulary-gather lines are live code this wave and are pinned at their indented
form.

Carried and **run**: `mx43_dualband.py`, on this wave's own children. The
dual-column discipline is standing until Cyril rules on the copy-back clause, so
a wave that seals a held lane and does not run the reader has published one
column, not two.

Carried and **not run**: `mx43_build.py` (wave 36's composed width × MLP path,
wave 37's live-mask repair, wave 38's descending-order excision), and
`mx43_vocabtie.py`, `mx43_vocabrowset.py` and `mx43_termgain.py` with their
drivers. Their load-bearing lines are still pinned. Wave 39 was criticised for
asserting lines in a file it never opened; the answer is not to drop the pins
when the file is dormant — it is to say which is which, which is what this
section does.

**NORM-ESTATE is still not authored, ALIAS is still blocked on the PROBE storage
ruling, and ROWSET Arms A and B are still blocked on a corpus.** None of those
three is an authoring problem, which is why the wave that was budgeted for
authoring did not spend itself on them.
