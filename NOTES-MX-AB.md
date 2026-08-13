# NOTES-MX-AB — wave 26: stacking off a number you told everyone to doubt

Wave 26, branch `codex/lrc-blank-compression`, source commits `c817308b` (wave
25's uncommitted ledger, committed), `b6f79b56` (the wave-26 tree and the
sealer's allowlist line), plus the sealing commit for this note.

**The series skipped two.** `NOTES-MX-AA` is wave 23. Waves 24 and 25 wrote no
note; their record is
`reports/matrix-campaign-20260810/wave24/human-review/SCORECARD.md` and
`.../wave25/human-review/SCORECARD.md`, which are complete, and the gap is
recorded here rather than papered over.

## 1. What the wave was for

Wave 25 sealed `ARCH.V20` at **13/15, 0.799135x**, one case above the parent, and
then wrote four reasons nobody should believe it: one reading, one sixteen-prompt
surface, one K, and wins on lanes the parent itself flubs. Two of those are
testable without a new review protocol, and wave 26 tested both.

* **Is it one K?** Build the neighbours. K = 18,432 and K = 22,528 both read
  **13/15**. It is a 4,096-row shelf.
* **Is the base real?** Stack on it. A base that is a measurement artefact does
  not carry a second fold. `ARCH.V20X13` — the same coreset on the campaign's
  mildest block excision — reads **10/15 at 0.746614x**, the deepest ten in
  twenty-six waves by five and a half points of ratio.

Neither answers the two doubts that remain (one reading, one surface). Both were
worth the compute because they are the doubts a build can address.

## 2. The three numbers this wave moved

| what | was | now |
|---|---|---|
| deepest **11/15** | `ARCH.V12`, 0.762614x (w25) | **`ARCH.V20X13G03`, 0.746614x** |
| deepest **10/15** | `ARCH.V32X13`, 0.801395x (w25) | **`ARCH.V20X13`, 0.746614x** |
| highest review | `ARCH.V20`, 13/15 (w25) | **three-way tie at 13/15**; deepest is `ARCH.V18` at 0.790005x |

Deepest **7/15 or better** did **not** move: `ARCH.V8X13GL01` / `ARCH.V8X13GL03`
still hold it at 0.691832x. Wave 26's deepest child below that line reads five.

## 3. Four things that are true about the algebra now

**The lambda atlas has no fixed peak and no fixed shape.** Four ladders:
`V16X13` 5/6/**7**/6/5, `V8X13` 6/6/**7**/7/4, `V32X13` 10/10/**10**/9/10,
`V20X13` 10/—/**11**/—/—. λ = 0.3 is the peak or joint peak on all four and buys
literally nothing on one of them. Wave 25 proposed it as *the* peak; it is better
described as the rung that never hurts. And on `V32X13` the *count* is flat across
five rungs while the *held set* moves hard — the λ = 0.3 rung holds **four** lanes
in full against the zero-solve control's one. A ladder read only by its count is
a ladder read at a quarter resolution.

**The vocabulary axis has exactly one collapse and it is 1,024 rows wide.**
9,216 reads 9, 8,192 reads 4, and four thousandths of stored ratio separate them.
Everything from 9,216 to 32,768 reads between 9 and 13.

**Prompt-echo is a width signature.** Three sightings — `V8CL896` (w24),
`V32W896` (w25), `V20W896` (w26) — all width children, across three different
second folds and two tokenizers. It is not a property of a body and not an
alternative to looping; `V20X13W896` does both.

**The third axis is still fatal, now measured on the best base there is.**
V20 alone 13, the 13+7 double excision alone 6, composed **5**, nothing in full.
L3 has said this since wave 23 and this is its cleanest instance.

## 4. Four things that went wrong, and what each one costs

* **Wave 25 never committed its merged ledger.** Its scorecard quotes 706 rows at
  sha `cd22df33` and blob `94ef44d3` — all three correct — while the repository
  stood at 682 since wave 24. The merge ran, the report was written, the copy-back
  did not happen. Fixed first thing (`c817308b`), byte-prefix checked rather than
  asserted. **The lesson is that "merge the ledger" is two steps and only one of
  them is on the server.**
* **Retention ran before the package.** Three children were deleted and their
  build receipts went with them, because a build receipt lives inside the child
  directory. All three were rebuilt and every one reproduced its **model tensors
  byte for byte** against the sealed cell receipt. The receipt *file* digest does
  not reproduce, and `wave26-drop-reproduction.json` records exactly which fields
  are non-deterministic (`built_at`, `elapsed_seconds`, `child_model`) — measured
  by rebuilding one child twice and diffing, not assumed. **Package before you
  prune.**
* **`mx*_compare.py` has been silently broken since wave 25.** It raises
  `KeyError: net_ratio` on any width child built by the vocabulary builder,
  because the two builders write the stored ratio under different keys. Wave 25
  built one such child and shipped a package with no comparison file at all, and
  nobody noticed because the packaging step is `[ -f "$t" ] && cp`. **A packaging
  step that silently skips a missing artifact will hide a crashed generator
  forever.**
* **`--campaign-wave 25` is a bare integer** and a bare integer is invisible to
  the token rename that cuts each wave's tree from the last one's. Caught by name
  before the first seal and asserted on the first sealed receipt. It would have
  put nine wave-26 rows into the ledger labelled wave 25.

## 5. The rename, again

`mx26_verify_seals.py` and `mx26_retain.py` were held out of the blanket rename
**by name**, in an explicit hand-edit set in the copier, and the warning that has
been in the verifier's docstring for six waves was not violated for a second
time: 20 declared cell roots became 21, `wave25/cells` still declared. Two more
files joined the hold-out set for a new reason — `mx26_rho.py` and `rhoq.sh` name
three *sealed bodies* per triple, and sealed bodies do not move when the wave
does. Two legs of the `V8X13` triple pointed at `$W/children`, which under
`$W` = wave26 resolves to nothing; repinned to the wave-24 share paths they
actually live at. A triple that resolves to nothing prices nothing and says so
in no log.

## 6. Estate

Root filesystem went from **11 GiB** free to **16 GiB** — 5.34 GB reclaimed from
one wave-24 deployment tree, eight superseded deployment tarballs, and docker's
dangling images and build cache. What was **not** deleted is in
`deployments-prune-manifest-wave26.json` with reasons: the waves 5–9 capture
caches (5.3 G + 743 M) are named by those waves' own run scripts and regenerating
them costs GPU hours, so deleting them would have traded the reproducibility of
sealed cells for space nothing needed; and the 806 G of folding-project artifacts
are not this campaign's to delete.

Seven of nine checkpoints retained (2.3 GiB on the share), three dropped with
reasons and with a reproduction check. Package is 73 files and committed.
