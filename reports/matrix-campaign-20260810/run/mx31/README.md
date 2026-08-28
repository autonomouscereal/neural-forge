# `run/mx31` — how this tree was cut, and what in it is wave 31's

Every file here except `cut_wave31.py`, this README and
`wave31-tree-cut-manifest.json` was produced from `run/mx30` by
`cut_wave31.py`. **The docstrings are wave 30's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. The
fix it named and did not apply is applied here, in two parts.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 30` | `--campaign-wave 31` | the sealer's wave argument |
| `"campaign_wave": 29` | `"campaign_wave": 31` | the extension surface's own stale field |
| `wave30` | `wave31` | a path component and a schema infix |
| `mx30` | `mx31` | module names, the build-receipt filename, the log tag |
| `MX30` | `MX31` | the child-directory prefix and the `$MX31` shell variable |

`wave 30`, `Wave 30` and `WAVE 30` **in English are not substituted**, so a
header that records what wave 30 measured still says wave 30 measured it. The
price is that most headers below open by describing their ancestor's wave. That
is why this file exists: the provenance is here, once, rather than rewritten
thirty times.

**2. Three files are copied with no substitution at all**, and every edit to
them was made by hand and read back:

* `mx31_verify_seals.py` — the rename turns `wave30/cells` into `wave31/cells`
  and thereby deletes last wave's seal root. One appended line: **26 declared
  roots become 27, `wave30/cells` still declared.**
* `mx31_retain.py` — an inherited `KEEP` set names last wave's children, and
  everything unnamed is deleted. `KEEP` and `DROP` are emptied; the wave root,
  the glob, the schema and the manifest name are retargeted by hand; the script
  refuses to run until this wave's verdicts are sealed.
* `mx31_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. Four schema strings retargeted by hand, **wave 30's
  basis schema appended** (the accept list now carries wave 31, 30, 29, 23, 22,
  21 and 19), seven module references and the log tag retargeted by name.

`cut_wave31.py --audit` reads every file back and refuses a surviving `mx30_`,
`MX30-`, `/wave30/` or `--campaign-wave 30`, with one declared exception at a
pinned count: `mx31_build.py`'s wave-31 header names `mx30_build.py` as the file
it states its diff against. It also asserts both seal roots are declared, that
the accept list still carries the three schemas a rename would have eaten, and
that `seal_verdict.sh` still passes `--parent-useful-or-partial 12` and
`--lanes-total 16`.

`wave31-tree-cut-manifest.json` is the explicit diff: per file, how many of each
substitution fired.

## What wave 31 changed on purpose, beyond the cut

* `mx31_allocator.py` — the recommendation dict is keyed on **six** decimals of
  the target ratio, not two. Wave 31 pre-registered ten named splits in one
  invocation and got nine back: `intermediate_size` 1536 at 0.808265x and
  1792-with-block-13-excised at 0.813949x both round to `"0.81"`. The two are
  0.0057 apart and the *default* tolerance is 0.006, so the collision is
  reachable by anyone pricing two nearby splits at once, and the failure mode is
  a silently missing pre-registration.
* `mx31_emitcoverage.py` — the ladder takes a `None` review, so a rung can be
  measured on the coverage law **before** it is graded. Wave 31's two new
  vocabulary rungs are read that way and print `-` rather than a borrowed count.
* `mx31_build.py` — the wave-31 header, which states the diff and nothing else.
