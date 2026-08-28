# `run/mx38` — how this tree was cut, and what in it is wave 38's

Every file here except `cut_wave38.py`, this README and
`wave38-tree-cut-manifest.json` was produced from `run/mx37` by `cut_wave38.py`.
**The docstrings are wave 37's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 38 run it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 37` | `--campaign-wave 38` | the sealer's wave argument |
| `"campaign_wave": 37` | `"campaign_wave": 38` | the extension surface's own field |
| `wave37` | `wave38` | a path component and a schema infix |
| `mx37` | `mx38` | module names, the build-receipt filename, the log tag |
| `MX37` | `MX38` | the child-directory prefix and the `$MX38` shell variable |

`wave 37`, `Wave 37` and `WAVE 37` **in English are not substituted**, so a
header that records what wave 37 measured still says wave 37 measured it. The
provenance is here, once, rather than rewritten thirty-one times.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx38_verify_seals.py` — the rename turns `wave37/cells` into `wave38/cells`
  and thereby deletes last wave's seal root. One appended line: **33 declared
  roots become 34, `wave37/cells` and `wave36/cells` still declared.**
* `mx38_retain.py` — an inherited `KEEP` set names last wave's children, and
  everything unnamed is deleted. Wave 37's **two keeps and seven drops** arrived
  here verbatim. `KEEP` and `DROP` are **emptied by decision**, the wave root,
  the glob, the schema and the manifest name retargeted by hand. The inherited
  justification paragraphs were deleted rather than renamed.
* `mx38_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **Four schema strings retargeted by hand, wave 37's
  basis schema appended** (the accept list now carries wave 38, 37, 36, 35, 34,
  33, 32, 31, 30, 29, 23, 22, 21, 20 and 19), **the module references retargeted
  by name**, and **both build-receipt write sites** moved to
  `mx38-build-receipt.json`. **And then nothing, which is §4 below.**
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. It arrived with **wave 37's marker and wave 37's six seal calls**,
  `cut_wave38.py --audit` refused on the marker before anything in the tree ran,
  and the six were deleted by hand. It also arrived sourcing last wave's `env.sh`
  and calling last wave's per-case and seal-verdict paths, and those were
  retargeted by hand too. **Sixth consecutive cut where this protection
  refused.**

**3. The sealer's accept list was appended IN THE SAME COMMIT AS THIS CUT.**
`experiments/seal_matrix_architecture_child.py` carries a second append-only
accept list, and it refused the first four reviews of waves 34 AND 35 — each
time after the GPU work was already done, each time costing a whole extra
deployment. That file is in the Codex lane's `experiments/` tree, so
`cut_wave38.py` has no standing to assert anything about it and does not try.
Waves 36 and 37 appended the four strings at the cut and paid no deployment for
them; wave 38 is the third, which is where a practice stops being a suggestion.
The four strings are
`neural-forge-matrix-wave38-{standard-shape-child,blend-build-receipt,logit-metric-writer,body-blend-build-receipt}-v1`.

`cut_wave38.py --audit` reads every file back and refuses a surviving `mx37_`,
`MX37-`, `/wave37/` or `--campaign-wave 37`, with three declared exceptions at
pinned counts. **Two of the three are expected to FIRE this wave, which is the
difference from wave 37**: wave 37 pre-declared the `mx38_retain.py` and
`sealq.sh` exemptions and used neither, because all of its sources were its own.
Five of wave 38's six children are one CPU pass off wave 37's **retained**
`MX37-CL896X09`, so the retention rule (*name what rebuilds each deleted child,
and from which retained source*) and the provenance rule (*name the retained
source a child was built on*) both require a wave-37 directory name in prose.
The counts are pinned at 6 and 5 rather than waived.

It also asserts six seal roots are declared, that the accept list carries ten
basis schemas, that `seal_verdict.sh` still passes `--parent-useful-or-partial
12` and `--lanes-total 16`, that `sealq.sh` carries this wave's marker, that
`gainq.sh` exists, that both extension surfaces have a scorer with the FROZEN v1
rubric still naming two of its own entries by id, that the penalty diagnostic
kept its `NOT_A_FOLD` banner and its identity refusal, that **wave 36's composed
width × MLP path still carries all five of its own guards**, that **wave 37's
live-mask repair and its log string are still there** — and two new assertions of
its own, which are §4.

---

## 4. Wave 38 makes no behavioural edit either, and asserts the three lines its double excision depends on

Wave 37 carried wave 36's one behavioural edit and ran its five assertions
without touching the path. Wave 38 is the **second** consecutive wave in that
position, which is when a carried guard is most likely to have been quietly
dropped, so all five are asserted again — and wave 37's two with them.

Wave 38's own build list needs no new code. `--ff` has been in the builder since
wave 15, `--excise` since wave 16, **multi-index excision since wave 17**, their
composition since wave 35 sealed `ARCH.CL896X13`, the vocabulary gather since
wave 25 and the terminal-gain solve since wave 24. Adding an edit to look busy
would put an unreviewed change under six seals.

**What it does depend on is the part of `--excise` that only a MULTI-INDEX
excision exercises.** Wave 37 removed block 9 — one block, not the last — and
asserted wave 18's live-mask repair because that is what a single shifting
excision needs. Wave 38's G2 body is `--ff 896 --excise 9 --excise 13`, the first
two-block excision since wave 28 and **the first ever composed with `--ff`**. Two
further lines carry it, and neither has ever been asserted:

* **`for index in sorted(excised, reverse=True):`** — the deletion loop. In
  ascending order the second `del` addresses a list the first one already
  shortened. This wave's own pair would raise `IndexError` rather than lie, but
  `--excise 9 --excise 11` in ascending order would delete block 9 and then
  whatever block 12 used to be, **silently**, with a correct-looking parameter
  count and a correct-looking depth. Descending order is the whole of the fix.
* **`layer.self_attn.layer_idx = position` and `layer.conv.layer_idx =
  position`** — the module re-index. `Lfm2Attention` and the conv block both
  address the KV cache and the conv state by that integer. One excision
  renumbers a tail; two renumber a longer one, and some of it twice.

Pinned at one occurrence each, asserted **before** the build, not after a dossier
comes back strange.

**And the manifest is parsed in the command that writes it.** `NOTES-MX-AO` §9
told the next wave to make that the default after wave 37 reproduced wave 35's
stdout-into-its-own-output corruption. `cut_wave38.py` re-reads
`wave38-tree-cut-manifest.json` with `json.loads` and asserts its schema and file
count before it returns, so the first artifact this wave writes obeys the rule.
