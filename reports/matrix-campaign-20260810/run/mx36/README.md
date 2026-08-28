# `run/mx36` — how this tree was cut, and what in it is wave 36's

Every file here except `cut_wave36.py`, this README and
`wave36-tree-cut-manifest.json` was produced from `run/mx35` by `cut_wave36.py`.
**The docstrings are wave 35's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 36 run it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 35` | `--campaign-wave 36` | the sealer's wave argument |
| `"campaign_wave": 35` | `"campaign_wave": 36` | the extension surface's own field |
| `wave35` | `wave36` | a path component and a schema infix |
| `mx35` | `mx36` | module names, the build-receipt filename, the log tag |
| `MX35` | `MX36` | the child-directory prefix and the `$MX36` shell variable |

`wave 35`, `Wave 35` and `WAVE 35` **in English are not substituted**, so a
header that records what wave 35 measured still says wave 35 measured it. The
provenance is here, once, rather than rewritten thirty times.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx36_verify_seals.py` — the rename turns `wave35/cells` into `wave36/cells`
  and thereby deletes last wave's seal root. One appended line: **31 declared
  roots become 32, `wave35/cells` and `wave34/cells` still declared.** Wave 35
  appended its own root and left the file's header saying WAVE 34; the root was
  right and the paragraph was stale, and the paragraph is corrected here rather
  than carried a third time.
* `mx36_retain.py` — an inherited `KEEP` set names last wave's children, and
  everything unnamed is deleted. Wave 35's **four keeps and seven drops** arrived
  here verbatim. `KEEP` and `DROP` are **emptied by decision**, the wave root,
  the glob, the schema and the manifest name retargeted by hand. The inherited
  justification paragraphs were deleted rather than renamed.
* `mx36_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **Four schema strings retargeted by hand, wave 35's
  basis schema appended** (the accept list now carries wave 36, 35, 34, 33, 32,
  31, 30, 29, 23, 22, 21, 20 and 19), **seven module references retargeted by
  name**, and **both build-receipt write sites** moved to
  `mx36-build-receipt.json`. **And then this wave's one behavioural edit, which
  is §4 below.**
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. It arrived with **wave 35's marker and wave 35's eight seal calls**,
  `cut_wave36.py --audit` refused on the marker before anything ran, and the
  eight were deleted by hand. It also arrived sourcing last wave's `env.sh` and
  calling last wave's per-case and seal-verdict paths, and those were retargeted
  by hand too. **Fourth consecutive cut where this protection refused.**

**3. The sealer's accept list was appended IN THE SAME COMMIT AS THIS CUT.**
`experiments/seal_matrix_architecture_child.py` carries a second append-only
accept list, and it has refused the first four reviews of waves 34 AND 35 — each
time after the GPU work was already done, each time costing a whole extra
deployment. That file is in the Codex lane's `experiments/` tree, so
`cut_wave36.py` has no standing to assert anything about it and does not try.
What it can do, and what wave 35's note told the next wave to do, is append the
four strings at the cut instead of at the first seal. Wave 36 did that. The four
strings are `neural-forge-matrix-wave36-{standard-shape-child,blend-build-receipt,logit-metric-writer,body-blend-build-receipt}-v1`.

`cut_wave36.py --audit` reads every file back and refuses a surviving `mx35_`,
`MX35-`, `/wave35/` or `--campaign-wave 35`, with one declared exception at a
pinned count: `mx36_build.py`'s wave-36 header names `mx35_build.py` as the file
it states its diff against. (A second exemption is pre-declared for
`mx36_retain.py`, which cannot obey the retention rule — *name what rebuilds each
deleted child, and from which retained source* — without naming wave 35's
retained checkpoints in prose. It is inert until retention is written.) It also
asserts five seal roots are declared, that the accept list carries eight basis
schemas, that `seal_verdict.sh` still passes `--parent-useful-or-partial 12` and
`--lanes-total 16`, that `sealq.sh` carries this wave's marker, that `gainq.sh`
exists, that both extension surfaces have a scorer with the FROZEN v1 rubric
still naming two of its own entries by id, that the penalty diagnostic kept its
`NOT_A_FOLD` banner and its identity refusal — **and five new assertions that
guard the one file this wave edits behaviourally.**

---

## 4. Wave 36's only behavioural edit: the composed width × MLP build

`NOTES-MX-AM` §3.5 records a refusal. `mx35_build.py`'s `main()` would not
compose `--hidden-size` with `--ff` in one build, on the grounds that

> composing `--ff` or `--head-dim` on top of it in one build would run a second
> selection inside tensors the first one has already replaced, with neither
> axis's census having seen the other.

Wave 35 was ordered to build a `W896` + `CL1792` body, met that refusal, recorded
it and left the pairing unbuilt. **Wave 36 was ordered to lift the guard
properly, and the honest finding is that the guard was half right, so exactly
half of it is lifted.** The full algebra is in `mx36_build.py`'s module
docstring; the short version:

**The weights commute.** The residual-basis fold acts on the *hidden* index and
only on it — readers `w1`/`w3` become `sqrt(1024/H)·W·diag(g)·E·diag(1/g_c)`
with the `F` row index untouched, the writer `w2` becomes `Eᵀw2` with the `F`
column index untouched. The MLP coreset selects a subset `S` of that same `F`
index, acting on the *other* side of every one of those tensors. So
`Eᵀ(w2·S) = (Eᵀw2)·S`, and the composed prior is the same object in either build
order. `--head-dim` has no such property — it rewrites the attention block's row
structure, re-picks a RoPE ladder and moves the `head_dim` config field this axis
*pins* — so **that half of the refusal stands, in the same words.**

**The census does not commute, and moving it is the build.** The selection is a
pivoted-Cholesky coreset of the *gated intermediate*, which is data. The sealed
`--ff` path ranks it on the parent's stream; on a narrowed body that stream does
not exist. `narrow_mlp_site` moves the census into the narrowed space — the MLP
counterpart of what `parent_gated_rows` already does for the conv coreset on this
same axis — and the sweep runs **in forward order**, so block *i*'s census is
captured through a child that already carries every narrowing above it.

**Three things are measured rather than assumed**, and the cutter's audit asserts
all three by name:

1. `COMPOSED_READER_EXACTNESS` — the child's gated intermediate is compared with
   the parent's at the lifted row at every site. They are equal in exact
   arithmetic (that is `fold_reader`'s whole docstring), and the equality is the
   licence for selecting on the child's own rows. The gap is recorded whether or
   not it refuses.
2. Wave 14 §3's identity precondition, **on the child**: the child's own folded
   `w1`/`w3` must reproduce `w2`'s captured input, because the selection indexes
   `phi`'s columns and `install_narrow_mlp` slices `w1`/`w3`'s rows by the same
   index.
3. The identity gate, at `--hidden-size 1024 --basis identity --ff 2560
   --no-fold-solve --no-head-gain --identity-gate strict`. At those settings the
   width fold is provably the identity and the coreset is `arange`, so the whole
   composed path must reproduce the parent **bit for bit**.

**Three new refusals** come with the lift, all of them about the census rather
than the weights: `--ff` composed with `--hidden-size` requires a conditioned arm
(the stale arm never captures through the child, so it would rank channels on the
parent's stream — the exact thing the composition replaces); `--select-metric
logit` is refused on the composed path (that metric is the parent's 1024-wide
softmax-Hessian reader and a width child's writer lands somewhere it has no rows
for); and `--ff` above the parent's own `intermediate_size` is not a narrowing.

**Nothing else moved.** `narrow_mlp_site` is reachable only when both flags are
set. The `--ff`-only path in `build()` — the one that produced every sealed
`CL*` cell in this campaign — is not touched by one byte, and `predicted_parameters`
gains one keyword whose default reproduces its old answer exactly.
