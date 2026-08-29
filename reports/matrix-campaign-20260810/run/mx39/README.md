# `run/mx39` — how this tree was cut, and what in it is wave 39's

Every file here except `cut_wave39.py`, this README and
`wave39-tree-cut-manifest.json` was produced from `run/mx38` by `cut_wave39.py`.
**The docstrings are wave 38's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 39 run it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 38` | `--campaign-wave 39` | the sealer's wave argument |
| `"campaign_wave": 38` | `"campaign_wave": 39` | the extension surface's own field |
| `wave38` | `wave39` | a path component and a schema infix |
| `mx38` | `mx39` | module names, the build-receipt filename, the log tag |
| `MX38` | `MX39` | the child-directory prefix and the `$MX39` shell variable |

`wave 38`, `Wave 38` and `WAVE 38` **in English are not substituted**, so a
header that records what wave 38 measured still says wave 38 measured it. The
provenance is here, once, rather than rewritten thirty-two times.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx39_verify_seals.py` — the rename turns `wave38/cells` into `wave39/cells`
  and thereby deletes last wave's seal root. One appended line: **34 declared
  roots become 35, `wave38/cells` and `wave37/cells` still declared.**
* `mx39_retain.py` — an inherited `KEEP` set names last wave's children, and
  everything unnamed is deleted. Wave 38's **two keeps and four drops** arrived
  here verbatim. `KEEP` and `DROP` are **emptied by decision**, the wave root,
  the glob, the schema and the manifest name retargeted by hand. The inherited
  justification paragraphs were deleted rather than renamed.
* `mx39_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **Four schema strings retargeted by hand, wave 38's
  basis schema appended** (the accept list now carries wave 39, 38, 37, 36, 35,
  34, 33, 32, 31, 30, 29, 23, 22, 21, 20 and 19), **thirteen module references
  retargeted by name in eleven edits**, **both build-receipt write sites** moved
  to `mx39-build-receipt.json` — **and one line that is not mechanical, which is
  §4 below.**
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. It arrived with **wave 38's marker and wave 38's six seal calls**,
  `cut_wave39.py --audit` refused on the marker before anything in the tree ran,
  and the six were deleted by hand. It also arrived sourcing last wave's `env.sh`
  and calling last wave's per-case and seal-verdict paths, and those were
  retargeted by hand too. **Seventh consecutive cut where this protection
  refused.**

**3. The sealer's accept list was appended IN THE SAME COMMIT AS THIS CUT.**
`experiments/seal_matrix_architecture_child.py` carries a second append-only
accept list, and it refused the first four reviews of waves 34 AND 35 — each
time after the GPU work was already done, each time costing a whole extra
deployment. That file is in the Codex lane's `experiments/` tree, so
`cut_wave39.py` has no standing to assert anything about it and does not try.
Waves 36, 37 and 38 appended the four strings at the cut and paid no deployment
for them; wave 39 is the fourth. The four strings are
`neural-forge-matrix-wave39-{standard-shape-child,blend-build-receipt,logit-metric-writer,body-blend-build-receipt}-v1`.

`cut_wave39.py --audit` reads every file back and refuses a surviving `mx38_`,
`MX38-`, `/wave38/` or `--campaign-wave 38`, with three declared exceptions at
pinned counts. **Two of the three are expected NOT to fire this wave, which is
the difference from wave 38**: wave 38 used both, because five of its six
children were one CPU pass off wave 37's retained `MX37-CL896X09` and both the
retention rule and the provenance rule required that directory name in prose.
**Wave 39's source is wave 35's retained `MX35-CL896X13`**, which no needle
forbids, so the bounded exemptions are declared at two rather than waived and
are expected to read zero.

It also asserts six seal roots are declared, that the accept list carries eleven
basis schemas, that `seal_verdict.sh` still passes `--parent-useful-or-partial
12` and `--lanes-total 16`, that `sealq.sh` carries this wave's marker, that
`gainq.sh` exists, that both extension surfaces have a scorer with the FROZEN v1
rubric still naming two of its own entries by id, that the penalty diagnostic
kept its `NOT_A_FOLD` banner and its identity refusal, that **wave 36's composed
width × MLP path still carries all five of its own guards**, that **wave 37's
live-mask repair and its log string are still there**, that **wave 38's three
multi-index excision lines are still there** — and five new assertions of its
own, which are §5.

---

## 4. Wave 39's one edit is a log string in a file it never runs

`mx39_build.py`'s `log()` has printed `[mx32]` since wave 32's cut, because the
literal sits inside an f-string and the substitution list carries identifiers,
paths and schema strings only. Wave 17 shipped the same defect as `[mx16]` and
wave 18's docstring records it. `NOTES-MX-AP` §1 calls it the seventh issue,
cosmetic, and waves 37 and 38 both declined to fix it on the stated grounds that
an edit in a no-behavioural-edit wave is itself a behavioural edit.

**Wave 39 can afford it because wave 39 does not run that file.** The build list
is two `mx39_vocab.py` vocabulary passes on `MX35-CL896X13` — a body wave 35
built, sealed and retains — with no body build, no conditioned capture and no
solve. `mx39_build.py` is carried, retargeted, asserted and never executed, so
the repair goes under no seal, touches no artifact and can move no number.

It is pinned in **both** directions and **verified by calling the function**,
not by reading it: the wave imports the module on the server and calls
`log("probe")`.

**And the first version of that pin refused on this cut's own header.** It
pinned the bare strings `[mx32]` at zero and `[mx16]` at one; the wave-39 header
explaining the repair names both literals four times between them, so the audit
failed on the prose. That is `NOTES-MX-AP` lesson 6 — *assert code lines at
their INDENTED form* — arriving a second time in a different file, one wave
after it was written down. The pins are now
`    print(f"[mx39] {message}", flush=True)` at one and the `[mx32]` and
`[mx16]` forms of the same line at zero. Wave 18's docstring recording `[mx16]`
and this wave's header explaining the repair are untouched, because rule 1 does
not rewrite prose that records history.

---

## 5. Wave 39 asserts the vocabulary gather, which is the module it actually runs

Every cut since wave 31 has asserted the lines that wave's own build list
depends on: wave 36 the composed width × MLP site, wave 37 the live-mask repair,
wave 38 the descending deletion order and the two module re-index assignments.
**Wave 39's build list is two `mx39_vocab.py` passes and nothing else** — and in
eleven waves of folding on that axis, **no cutter has ever asserted a single
line of it.** It is the least-guarded module this campaign still runs and the
only module wave 39 runs at all.

Five lines carry it, all pinned at one:

* `if not torch.equal(gathered, table[index]):` — the identity gate for a
  **zero-solve** arm. Every kept row is the source's own bytes, so this is
  `torch.equal` and not a tolerance. Delete it and a gather that silently
  permuted or resampled rows would still export, still load, still decode, and
  still be sealed as a zero-solve fold of the parent's own weights.
* `if not torch.equal(gathered[probe], table[order[probe]]):` — the four-row
  spot check that runs first, so a failure names a row rather than a tensor.
* `closure_state = merge_closure_intact(vocab, keptset)` — the **structural**
  half of the census gate, the half with no flag, because a kept token that
  cannot be produced from the restricted merge list is a broken tokenizer and
  never a coverage choice.
* `if lost and not args.below_floor:` — the **coverage** half, and the guard
  that decides whether this wave needs `--below-floor` at all.

  **The first version of this README said both of wave 39's children are built
  below the census floor deliberately and that the wave therefore passes that
  flag. That was wrong and it was checked rather than shipped.** Wave 38 built
  K = 14,336, 10,240 and 9,216 on the sibling body and every one of them logged
  *sealed-corpus coverage: donor-mapped 1.000000 (0 lost), canonical 1.000000
  (0 lost)* — the refusal never fired and the flag was never passed. Wave 39
  passes it on neither child. The line is pinned anyway, because a wave that
  does not need a guard is exactly the wave that will not notice the guard is
  gone: if it were missing, a rung that DID lose coverage would build silently
  and the receipt's `below_census_floor` field would say nothing.
* `"the child tokenizer is not lossless on text; the byte alphabet or "` — the
  refusal behind `decode(child_encode(t)) == t`. This parent has
  `byte_fallback: false` and `unk_token: null`, so the 256 byte-alphabet rows
  are the only reason an arbitrary byte can still be encoded at all.

**Every needle carries its indentation.** The bare strings `torch.equal`,
`merge_closure_intact` and `lossless on text` occur **6, 3 and 4 times** in that
file — the module's own docstring and its receipt fields quote every one of them
— so a bare-string pin would pass with the code deleted and only the prose left.
Each indented form occurs exactly once.

**And the manifest is parsed in the command that writes it**, as `NOTES-MX-AO`
§9 told the next wave to make the default. `cut_wave39.py` re-reads
`wave39-tree-cut-manifest.json` with `json.loads` and asserts its schema and file
count before it returns.
