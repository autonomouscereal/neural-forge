# `run/mx42` — how this tree was cut, and what in it is wave 42's

Every file here except `cut_wave42.py`, this README, `mx42_dualband.py` and
`dualbandq.sh` was produced from `run/mx41` by `cut_wave42.py`. **The docstrings
are wave 41's and earlier, on purpose.**

`NOTES-MX-AF` §9 records that wave 30's blanket rename broke three things: it
deleted `wave29/cells` from the seal verifier's declared roots, it dropped wave
29's string from the append-only `BASIS_SCHEMAS_ACCEPTED`, and it rewrote four
docstrings that recorded what wave 29 did into claims that wave 30 did it. Wave
31 wrote the fix; waves 32 through 41 ran it with a small stated diff each time.

**1. The substitution list carries no prose.** It is exactly five entries:

| from | to | what it is |
|---|---|---|
| `--campaign-wave 41` | `--campaign-wave 42` | the sealer's wave argument |
| `"campaign_wave": 41` | `"campaign_wave": 42` | the extension surface's own field |
| `wave41` | `wave42` | a path component and a schema infix |
| `mx41` | `mx42` | module names, the build-receipt filename, the log tag |
| `MX41` | `MX42` | the child-directory prefix and the `$MX42` shell variable |

`wave 41`, `Wave 41` and `WAVE 41` **in English are not substituted**, so a
header that records what wave 41 measured still says wave 41 measured it.

**2. Four files are copied with no substitution at all**, and every edit to them
was made by hand and read back.

* `mx42_verify_seals.py` — the rename turns `wave41/cells` into `wave42/cells`
  and thereby deletes last wave's seal root. One appended line: **37 declared
  roots become 38, `wave41/cells` and `wave40/cells` still declared.**
  `wave41/cells` holds ten graded children.
* `mx42_retain.py` — an inherited `KEEP` set names last wave's children and
  everything unnamed is deleted. It arrived carrying **wave 41's two keeps and
  eight drops with every justification paragraph intact**; both mappings were
  emptied by hand and the wave root, glob, schema and manifest name retargeted
  by hand.
* `mx42_build.py` — `BASIS_SCHEMAS_ACCEPTED` is append-only and a substitution
  rewrites its first entry. **Four schema strings retargeted by hand, wave 41's
  basis schema appended** (the accept list now carries wave 42, 41, 40, 39, 38,
  37, 36, 35, 34, 33, 32, 31, 30, 29, 23, 22, 21, 20 and 19), **the module
  references retargeted by name**, **both build-receipt write sites** moved to
  `mx42-build-receipt.json`, and the log prefix carried to `[mx42]`. **Wave 42
  does not run this file.** The edits are still made, because the next wave that
  runs it inherits whatever this cut leaves behind.
* `sealq.sh` — arrives VERBATIM carrying last wave's marker and last wave's seal
  calls. **Tenth consecutive cut where this protection refused, and the second
  where wave 40's pin is what caught it rather than the marker.** The file
  arrived with wave 41's **ten** seal calls, every count intact. They were
  deleted, the marker rewritten, and the three path resolutions a verbatim copy
  cannot fix for itself (last wave's `env.sh`, `mx*_percase.py` and
  `seal_verdict.sh`) retargeted by hand at pinned counts of one each.

**3. The sealer's accept list was appended IN THE SAME COMMIT AS THIS CUT.**
Waves 36 through 41 appended at the cut and paid no extra deployment; wave 42 is
the seventh.

---

## 2. WAVE 42 ADDS A MEASUREMENT MODULE AND EDITS NO CARRIED ONE

### 2.1 What is new: `mx42_dualband.py` and `dualbandq.sh`

`NOTES-MX-AS` §4.1 flagged a contradiction and deliberately left it open. The
canonical clause — *a copy-back is not a decode failure, it is the absence of a
decode* — reads all ten wave-41 children **0/15**. The same clause applied to
wave 40's sealed floor child `ARCH.V9CL896X13` gives canonical **zero**, and
wave 40 published canonical **one**. The two children's held summarisation lanes
are byte-identical, `111e342d9cb5438c`. Either the clause does not mean what it
says, or the canonical 1/15 rung has been zero since wave 39.

**No wave can settle that by building a child, and wave 41 was right not to
re-grade a sealed count to settle it.** What a wave can do is make both readings
**complete and checkable over the whole deep band**, so that the ruling costs no
re-grading either way. That is the entire job of this module.

* **Column A, `copy_back_counts`, is not computed.** It is the sealed
  `per-case.json` verdict read off the share, and the module **refuses to write
  its artifact** if its own count of that file disagrees with the ledger row's
  sealed `behavior_useful_or_partial`. A second column is worth nothing if the
  first is this module's recount rather than the estate's record.
* **Column B, `copy_back_refused`, is column A minus a subtraction.** A held
  lane whose response is the prompt's own payload returned back is voided.
  Nothing else moves; a lane sealed as a miss is never re-read, and the module
  cannot turn a miss into a hold in either direction.
* **The lane-ask table is the load-bearing part, not the threshold.** Wave 40
  sealed a copied SUMMARISATION lane held and a copied RETRIEVAL lane a miss,
  and wave 41 showed why those are consistent: the summarisation ask is to
  PRESERVE two facts, which a copy does; the retrieval ask is to SELECT three
  facts, which a copy does not. So the clause can only reach a PRESERVE lane. If
  it ever reaches another, the module publishes it as an **anomaly** against
  wave 40's own precedents rather than absorbing it.
* **The measure is deterministic and published.** Longest common substring
  between the whitespace-normalised, case-folded response and the same
  normalisation of the lane's payload, as a fraction of each. The threshold is
  stated once (`COPY_BACK_PAYLOAD_COVERAGE = 0.60`) and **both raw fractions are
  published per lane**, so a reader who disagrees with the threshold can
  re-decide any lane from the artifact without re-running anything.
* **The machine does not get the last word on a held lane.** `--hand-reading`
  merges the reviewer's per-lane adjudications and publishes an explicit
  machine-vs-hand disagreement list. A threshold is an instrument, not a
  verdict.

The driver runs the module **twice on purpose**: machine-only first, so the
artifact the reviewer reads from is not conditioned on the reviewer's
expectation, then with the hand reading merged.

### 2.2 What is edited: nothing

Wave 41 added a module and edited one (`mx41_child_index.py`, wave 40's §11.6).
Wave 42 adds a module and edits none. Every carried file keeps the bytes wave 41
shipped, modulo the five substitutions and the four hand edits above.

---

## 3. What this tree carries and does not run

Wave 42's build list is **three `mx42_vocab.py` gathers** — `V6CL768`,
`V7CL896X13` and `V6CL896X13` — and **no body, no capture, no solve and no
GPU-side construction at all**. Wave 39's five vocabulary-gather lines are
therefore live code this wave and are pinned at their indented form.

Carried and **not run**: `mx42_build.py` (wave 36's composed width × MLP path,
wave 37's live-mask repair, wave 38's descending-order excision), and
`mx42_vocabtie.py`, `mx42_vocabrowset.py` and `mx42_termgain.py` with their
drivers. Their load-bearing lines are still pinned. Wave 39 was criticised for
asserting lines in a file it never opened; the answer is not to drop the pins
when the file is dormant — it is to say which is which, which is what this
section does.

**PLANE and XCHANGE are still not authored.** `NOTES-MX-AS` §13 items 5 and 6
named them as the honest cost of wave 41 and gave the reason as authoring time
for a new solver rather than any finding about them. That reason has not
changed, and this tree carries no driver for a module that does not exist.
