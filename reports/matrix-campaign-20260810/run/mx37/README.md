# `run/mx37` — how this tree was cut, and what in it is wave 37's

Every file here except `cut_wave37.py`, this README and
`wave37-tree-cut-manifest.json` was produced from `run/mx36` by `cut_wave37.py`.
**The docstrings are wave 36's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 37 run it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 36` | `--campaign-wave 37` | the sealer's wave argument |
| `"campaign_wave": 36` | `"campaign_wave": 37` | the extension surface's own field |
| `wave36` | `wave37` | a path component and a schema infix |
| `mx36` | `mx37` | module names, the build-receipt filename, the log tag |
| `MX36` | `MX37` | the child-directory prefix and the `$MX37` shell variable |

`wave 36`, `Wave 36` and `WAVE 36` **in English are not substituted**, so a
header that records what wave 36 measured still says wave 36 measured it. The
provenance is here, once, rather than rewritten thirty times.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx37_verify_seals.py` — the rename turns `wave36/cells` into `wave37/cells`
  and thereby deletes last wave's seal root. One appended line: **32 declared
  roots become 33, `wave36/cells` and `wave35/cells` still declared.** Wave 36
  corrected the stale-header failure mode that waves 34 and 35 carried; this
  wave keeps the correction by writing its own paragraph instead of editing the
  count inside wave 36's.
* `mx37_retain.py` — an inherited `KEEP` set names last wave's children, and
  everything unnamed is deleted. Wave 36's **three keeps and five drops** arrived
  here verbatim. `KEEP` and `DROP` are **emptied by decision**, the wave root,
  the glob, the schema and the manifest name retargeted by hand. The inherited
  justification paragraphs were deleted rather than renamed.
* `mx37_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **Four schema strings retargeted by hand, wave 36's
  basis schema appended** (the accept list now carries wave 37, 36, 35, 34, 33,
  32, 31, 30, 29, 23, 22, 21, 20 and 19), **fourteen module references retargeted
  by name**, and **both build-receipt write sites** moved to
  `mx37-build-receipt.json`. **And then nothing, which is §4 below.**
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. It arrived with **wave 36's marker and wave 36's six seal calls**,
  `cut_wave37.py --audit` refused on the marker before anything ran, and the six
  were deleted by hand. It also arrived sourcing last wave's `env.sh` and calling
  last wave's per-case and seal-verdict paths, and those were retargeted by hand
  too. **Fifth consecutive cut where this protection refused.**

**3. The sealer's accept list was appended IN THE SAME COMMIT AS THIS CUT.**
`experiments/seal_matrix_architecture_child.py` carries a second append-only
accept list, and it refused the first four reviews of waves 34 AND 35 — each
time after the GPU work was already done, each time costing a whole extra
deployment. That file is in the Codex lane's `experiments/` tree, so
`cut_wave37.py` has no standing to assert anything about it and does not try.
Wave 36 appended the four strings at the cut and paid no deployment for them;
wave 37 does the same. The four strings are
`neural-forge-matrix-wave37-{standard-shape-child,blend-build-receipt,logit-metric-writer,body-blend-build-receipt}-v1`.

`cut_wave37.py --audit` reads every file back and refuses a surviving `mx36_`,
`MX36-`, `/wave36/` or `--campaign-wave 36`, with three declared exceptions at
pinned counts: `mx37_build.py`'s wave-37 header names `mx36_build.py` as the file
it states its diff against, and two exemptions are pre-declared for
`mx37_retain.py` and `sealq.sh`, which cannot obey the retention rule and the
provenance rule — *name what rebuilds each deleted child, and from which retained
source*; *name the retained source an arm was solved on* — without naming wave
36's retained checkpoints in prose. Both are inert until retention and the seals
are written. It also asserts six seal roots are declared, that the accept list
carries nine basis schemas, that `seal_verdict.sh` still passes
`--parent-useful-or-partial 12` and `--lanes-total 16`, that `sealq.sh` carries
this wave's marker, that `gainq.sh` exists, that both extension surfaces have a
scorer with the FROZEN v1 rubric still naming two of its own entries by id, that
the penalty diagnostic kept its `NOT_A_FOLD` banner and its identity refusal,
that **wave 36's composed width × MLP path still carries all five of its own
guards** — and two new assertions of its own, which are §4.

---

## 4. Wave 37 makes no behavioural edit, and asserts the two lines it depends on

Wave 36's cut carried one behavioural edit and five assertions to guard it. Wave
37 carries the edit and **runs the five assertions on this wave's copy** rather
than retiring them as last wave's business. A guard rots in exactly the wave that
does not use the path it guards, and this is that wave: nothing in wave 37's
build list touches `--hidden-size`.

Wave 37's own build list needs no new code. `--ff` has been in the builder since
wave 15, `--excise` since wave 16, their composition since wave 35 sealed
`ARCH.CL896X13`, the vocabulary gather since wave 25 and the terminal-gain solve
since wave 24. Adding an edit to look busy would put an unreviewed change under
six seals.

**What it does depend on is one two-line branch nobody has asserted.** Every
excision this campaign has sealed since wave 26 has been **block 13** — the final
conv — and removing the final block shifts nothing. Wave 37 excises **block 9**,
so blocks 10 through 13 shift down one slot and an attention block lands in a
conv's index. `Lfm2Model` picks each block's mask by *position* in
`config.layer_types`, so a builder that deletes a module without re-deriving that
list hands the shifted attention block the padding mask — `None` on this corpus —
instead of the causal one. `mx18_mask_probe.py` measured the cost: 0.0000 future
attention mass with block 13 alone excised, **28.6% to 48.0%** once anything
shifts. Wave 18 wrote the repair; wave 29 bisected the wave-17 body that lacked
it; nothing since has checked that it is still there.

So `--audit` asserts, by name and at a pinned count of one each:

* `harness.model.config.layer_types = [` — the repair itself;
* `so the live mask follows the live block` — the log line the build log carries,
  because a repair that stops announcing itself is a repair the next wave cannot
  confirm ran from the artifact it keeps.

Both are asserted **before** the build, not after a dossier comes back strange.
