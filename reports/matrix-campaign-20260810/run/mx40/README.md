# `run/mx40` — how this tree was cut, and what in it is wave 40's

Every file here except `cut_wave40.py`, this README, `mx40_vocabtie.py`,
`tieq.sh` and `wave40-tree-cut-manifest.json` was produced from `run/mx39` by
`cut_wave40.py`. **The docstrings are wave 39's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 39 ran it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 39` | `--campaign-wave 40` | the sealer's wave argument |
| `"campaign_wave": 39` | `"campaign_wave": 40` | the extension surface's own field |
| `wave39` | `wave40` | a path component and a schema infix |
| `mx39` | `mx40` | module names, the build-receipt filename, the log tag |
| `MX39` | `MX40` | the child-directory prefix and the `$MX40` shell variable |

`wave 39`, `Wave 39` and `WAVE 39` **in English are not substituted**, so a
header that records what wave 39 measured still says wave 39 measured it.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx40_verify_seals.py` — the rename turns `wave39/cells` into `wave40/cells`
  and thereby deletes last wave's seal root. One appended line: **35 declared
  roots become 36, `wave39/cells` and `wave38/cells` still declared.** Note that
  `wave39/cells` is not an empty formality this wave: wave 39 sealed two cell
  receipts and then never graded them, so that root holds two real children.
* `mx40_retain.py` — an inherited `KEEP` set names last wave's children and
  everything unnamed is deleted. It arrived with **`KEEP` and `DROP` already
  empty**, because wave 39 never reached its retention step; the wave root, the
  glob, the schema, the manifest name and two usage lines were retargeted by
  hand.
* `mx40_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **Four schema strings retargeted by hand, wave 39's
  basis schema appended** (the accept list now carries wave 40, 39, 38, 37, 36,
  35, 34, 33, 32, 31, 30, 29, 23, 22, 21, 20 and 19), **the module references
  retargeted by name**, **both build-receipt write sites** moved to
  `mx40-build-receipt.json`, and **wave 39's log-prefix repair carried to
  `[mx40]`**.
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. **Eighth consecutive cut where this protection refused** — and the
  first where the file arrived carrying a marker and *no* seal calls, because
  wave 39 never graded either of its children. The marker was rewritten by hand
  and the three path resolutions a verbatim copy cannot fix for itself (last
  wave's `env.sh`, `mx*_percase.py` and `seal_verdict.sh`) were retargeted by
  hand.

**3. The sealer's accept list was appended IN THE SAME COMMIT AS THIS CUT.**
Waves 34 and 35 each paid an entire extra deployment to meet
`experiments/seal_matrix_architecture_child.py`'s second append-only accept list
at their first seal, after the GPU work was done. Waves 36 through 39 appended
at the cut and paid none; wave 40 is the fifth. It appends **five** strings
rather than four — the four standard ones plus
`neural-forge-matrix-wave40-tied-vocabulary-child-v1`, which is
`mx40_vocabtie.py`'s schema and **the first genuinely new build-receipt schema
this campaign has added since wave 25's vocabulary gather.**

---

## 4. Wave 40 ADDS a build module, edits none, and RUNS every path wave 39 only asserted

Waves 38 and 39 made no behavioural edit. Wave 40 makes no *edit* either — every
carried module keeps the bytes wave 39 shipped — but it **adds** two files, and
it is the first wave since 38 to execute the builder at all.

### 4.1 What is new: `mx40_vocabtie.py` and `tieq.sh`

`docs/UNBUILT_INVENTORY_2026-08-28.md` §2 ranks **C1, vocabulary-axis row
TYING** first of nine classes closed without a child. It was killed at wave 14
on a nearest-neighbour cosine of **0.2977** over the post-12,288 tail — and that
statistic compared tying against **verbatim**, when the mechanism that actually
shipped on this axis and set the campaign's floor is **deletion**. Tie-vs-delete
is the comparison that decides the class and it was never run.

**The form the arm is built in was decided by a measurement taken before the
builder was written.** C1 as literally specified is a *many-to-one* vocabulary:
several tail tokens addressing one shared row. On this stack that is **not a
constructible child.** A `tokenizer.json` whose `model.vocab` maps seven tokens
onto one id is refused at load by `transformers` 5.12.1 / `tokenizers` 0.22.2:

```
Exception: Error while attempting to unpickle Tokenizer:
Token `Ã¹` out of vocabulary at line 1 column 2145524
```

**and it names a different token on different runs** — five loads of the
identical file named three different tokens at four different column offsets.
That is the Rust `HashMap` id→token inverse being built in a nondeterministic
order, surfacing as a nondeterministic refusal. Had it loaded, `decode` of a
tied id would have been a non-function across processes, and a frozen review of
a child whose decode depends on the process is not a frozen review.

So wave 40 builds the **injective** realisation, which is the better controlled
experiment anyway:

| | deletion twin | tie arm |
|---|---|---|
| kept ids | top R by census+closure | **the same R, by the same function** |
| tokenizer | closure-restricted, dense remap | **the same bytes** |
| parameter count | R × 1024 + body | **the same, asserted on the command line** |
| rows 0 … R−c−1 | verbatim | **verbatim, `torch.equal`** |
| rows R−c … R−1 | verbatim | **closed-form weighted centroid of themselves and every dropped token nearest to them** |

The only difference between the two children is the values in `c` rows of one
tensor, and `moved_rows` is asserted to equal `c` exactly. The many-to-one form
would have moved the tokenizer and the weights at once, and a difference could
then have been either.

`tieq.sh` adds a **cross-process** decode check the builder cannot make of
itself: the same child's full id→token map is hashed in two independently
launched interpreters and the digests compared. Finding a non-function decode at
review time costs a GPU hour; finding it here costs two seconds.

### 4.2 What is asserted, and why the count went up

`cut_wave40.py --audit` runs everything wave 39's audit ran — wave 36's five
composed width × MLP lines, wave 37's two live-mask lines, wave 38's three
multi-index excision lines, wave 39's five vocabulary-gather lines, the seal
roots, the accept list, the receipt filenames, the log prefix, the frozen v1
rubric, the penalty diagnostic's banner — **and seven new ones on the tie
builder**, which is the module with no history of surviving a cut at all.

**And this wave's assertions guard live code.** Wave 39 asserted the builder's
excision and width paths on a file it never opened. Wave 40's build list is
`--ff 768`, `--ff 768 --excise 9`, `--ff 1728` and `--ff 736`: the descending
deletion order, both module re-index assignments, the live-mask repair and the
composed width × MLP site are all executed this wave. Wave 39's log-prefix
repair, which that wave could afford precisely *because* it never ran the file,
is now inside four build receipts.

**And the manifest is parsed in the command that writes it**, `NOTES-MX-AO` §9's
rule, as it has been since wave 38.
