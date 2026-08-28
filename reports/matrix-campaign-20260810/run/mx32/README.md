# `run/mx32` — how this tree was cut, and what in it is wave 32's

Every file here except `cut_wave32.py`, `mx32_blend_body.py`, `bodyblendq.sh`,
this README and `wave32-tree-cut-manifest.json` was produced from `run/mx31` by
`cut_wave32.py`. **The docstrings are wave 31's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; wave 32 runs it again with a two-line diff and no argument
re-derived.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 31` | `--campaign-wave 32` | the sealer's wave argument |
| `"campaign_wave": 31` | `"campaign_wave": 32` | the extension surface's own field |
| `wave31` | `wave32` | a path component and a schema infix |
| `mx31` | `mx32` | module names, the build-receipt filename, the log tag |
| `MX31` | `MX32` | the child-directory prefix and the `$MX32` shell variable |

`wave 31`, `Wave 31` and `WAVE 31` **in English are not substituted**, so a
header that records what wave 31 measured still says wave 31 measured it. The
price is that most headers below open by describing their ancestor's wave. That
is why this file exists: the provenance is here, once, rather than rewritten
thirty times.

**2. Three files are copied with no substitution at all**, and every edit to
them was made by hand and read back:

* `mx32_verify_seals.py` — the rename turns `wave31/cells` into `wave32/cells`
  and thereby deletes last wave's seal root. One appended line: **27 declared
  roots become 28, `wave31/cells` and `wave30/cells` still declared.**
* `mx32_retain.py` — an inherited `KEEP` set names last wave's children, and
  everything unnamed is deleted. `KEEP` and `DROP` are emptied; the wave root,
  the glob, the schema and the manifest name are retargeted by hand; the script
  refuses to run until this wave's verdicts are sealed.
* `mx32_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. Four schema strings retargeted by hand, **wave 31's
  basis schema appended** (the accept list now carries wave 32, 31, 30, 29, 23,
  22, 21, 20 and 19), nine module references and the log tag retargeted by name.

`cut_wave32.py --audit` reads every file back and refuses a surviving `mx31_`,
`MX31-`, `/wave31/` or `--campaign-wave 31`, with one declared exception at a
pinned count: `mx32_build.py`'s wave-32 header names `mx31_build.py` as the file
it states its diff against. It also asserts all three recent seal roots are
declared, that the accept list still carries the four schemas a rename would have
eaten, and that `seal_verdict.sh` still passes `--parent-useful-or-partial 12`
and `--lanes-total 16`.

**The diff against `cut_wave31.py` is two things.** `NOT_CARRIED` is new — wave
31's tree-cut manifest is wave 31's receipt and stays in wave 31's tree, rather
than being renamed forward and then overwritten by this wave's, which would have
left a manifest whose own file list said it renamed itself. And
`ALLOWED_HISTORICAL` starts with only the build header: wave 31's seven-count
waiver on the retention script was pinned to retention prose this wave has not
written yet, so carrying it forward would waive a file that is about to be
emptied by hand. When this wave's retention policy cites a retained ancestor,
the audit fires and the count is pinned against the prose that exists.

## What wave 32 authored, beyond the cut

* **`mx32_blend_body.py`** — the wave's centre. Wave 29's `mx32_blend.py` blends
  two children that differ in exactly the 1,024-number terminal diagonal and
  refuses anything else; the three `ARCH.CL1792` siblings differ in the whole
  feed-forward interior, so that refusal fires and it is right to. The new
  builder exists because the `intermediate_size` fold is a **channel
  selection**: `w1`/`w3` rows are verbatim parent rows in the pivot's own order,
  so two siblings that disagree about which channels to keep store the same
  parent rows *in a different order*. Blending row `i` with row `i` blends two
  unrelated channels. `--align channel` recovers each sibling's selection by
  exact row match against the parent, keeps the source's `w1`/`w3` untouched,
  and blends `w2` **column-wise on the recovered channel intersection** with the
  convex weight renormalised over the siblings that actually kept the channel.
  `--align none` is the naive arm, built once so that "meaningless" is a number
  rather than an assertion. `--census` answers "are these two even alignable"
  for one CPU pass and no checkpoint.
* **`bodyblendq.sh`** — the driver. Its spec is verbose rather than
  colon-packed, because wave 29's `blendq.sh` field order has no room for a
  three-way simplex point that anyone would read correctly.
* **`experiments/seal_matrix_architecture_child.py`** gains four strings:
  wave 32's three renamed builders and one genuinely new class,
  `neural-forge-matrix-wave32-body-blend-build-receipt-v1`. They were appended
  in a commit and shipped as a verified deployment **before** the first build,
  because wave 31 discovered the hard way that this list refuses six reviews at
  the top of the sealer rather than in the middle of it.
