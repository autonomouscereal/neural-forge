# NOTES-MX-AO — wave 37: the campaign floor holds two, on a margin two tokens wide

**2026-08-28.** Branch `codex/lrc-blank-compression`. Six children built, sealed
and read twice, plus three reproduction controls and one robustness read that are
not cells. Ledger 964 → **976** rows,
`a94caa551d9a7a2daeac2021b559bb09a32b3defd41ede778e77988f48797ede`, byte-prefix
verified against the pre-wave backup on the share and again in the repository
before the copy-back was written. **Two deployments, and the accept list cost
neither of them — second consecutive wave.**

**A lean wave with no behavioural edit anywhere in the tree, and that is a
decision rather than an omission.** §1.

---

## 1. Wave 37 changed no code, and asserted the two lines it depends on

Wave 36 lifted a build guard. Wave 37 was ordered to finish the deep family's two
open arms and to try one alternative excision. `--ff` has been in the builder
since wave 15, `--excise` since wave 16, their composition since wave 35 sealed
`ARCH.CL896X13`, the vocabulary gather since wave 25 and the terminal-gain solve
since wave 24. **Nothing in the build list needed a code path that did not
exist**, and inventing one to look busy would have put an unreviewed edit
underneath six seals.

What the tree cut did instead was two things.

**It ran wave 36's five guarding assertions on this wave's copy of the builder.**
Wave 36's composed width × MLP path carries a surviving refusal, an exactness
assertion and two identity preconditions. Wave 37 does not touch that path and
does not build on it, **which is exactly the condition under which a guard
rots**, so `cut_wave37.py --audit` asserts all five by name on the carried copy.

**It added two assertions of its own, about a branch nobody had checked.** Every
excision this campaign has sealed since wave 26 has been **block 13** — the final
conv — and removing the final block shifts nothing. Wave 37 excises **block 9**,
so blocks 10 through 13 shift down one slot and a `full_attention` block lands in
a conv's index. `Lfm2Model` picks each block's mask by *position* in
`config.layer_types`, so a builder that deletes a module without re-deriving that
list hands the shifted attention block the padding mask — `None` on this
corpus — instead of the causal one. `mx18_mask_probe.py` measured the cost:
0.0000 future attention mass with block 13 alone excised, **28.6% to 48.0%** once
anything shifts. Wave 18 wrote the repair; wave 29 bisected the wave-17 body that
lacked it; **nothing since had asserted it was still there.**

So the audit pins `harness.model.config.layer_types = [` and its log string at
one occurrence each, **before the build**. All three acceptance criteria were
registered in advance and met exactly:

> `[mx32] excised blocks [9]; child depth 13; config.layer_types re-derived (13 entries) so the live mask follows the live block`

> `layer_types: conv conv full_attention conv full_attention conv full_attention conv full_attention full_attention conv full_attention conv`

**Note the double `full_attention` at positions 8 and 9**, which the parent never
has. That is the shift, and it is what makes `ARCH.CL896X09` a structurally
different body from every `X13` body in the estate at an identical parameter
count.

---

## 2. A documented law was wrong, and the correction is in the pre-registration

`NOTES-MX-AM` and `NOTES-MX-AN` both state that the CL896 terminal-gain arm is
worth **exactly +1** at K = 10,240, 14,336, 16,384, 19,456 and 20,480, and +2
only at K = 12,288.

**At K = 16,384 it is +2.** `ARCH.V16CL896` seals ONE (wave 23,
`mx23-v16cl896-0001-a6review`); `ARCH.V16CL896G03` seals THREE (wave 34,
`mx34-v16cl896g03-0001-a6review`); there is no superseding row for either. And
**wave 34's own seal note says so in words**, verbatim from the ledger:

> *reads THREE at 0.469305x against its control ONE, so the gain buys TWO here as
> it bought two on the K=12,288 base.*

The law was written a wave later and contradicted the seal it was summarising.
K = 16,384 is one of the two bases wave 37 built a λ1.0 arm on, so a prediction
made from the stated law would have started from the wrong control delta. The
corrected table is in `wave37/prereads/preregistration-wave37.json`, written
before any GPU ran, and now lives permanently in `DEEP_RECORD.md` §8.1.

**Restated: the λ0.3 arm lands the child at THREE at five of six K values,
regardless of where its control was. The invariant is the arm's count, not its
delta.**

The order's own parameter estimate was corrected in the same file. Wave 37 was
dispatched with `V12CL896X09 (~0.427x)`. Blocks 9 and 13 are both conv, so the
closed form gives **96,649,984 = 0.420779x exactly** — the same count as
`ARCH.V12CL896X13` to the byte. That is not a rounding difference: it turns the
pair into a controlled same-ratio, same-K, one-block-apart comparison at the
campaign floor rather than a new rung. The loader confirmed both counts.

---

## 3. F1: the λ1.0 ladder is complete, and it is not monotone in K

| K | control | λ0.3 | λ1.0 | λ1.0 Δ |
|---:|---:|---:|---:|---:|
| 16,384 `V16CL896` | 1 | 3 | **2** | **+1** |
| 14,336 `V14CL896` | 2 | 3 | **2** | **+0** |
| 12,288 `V12CL896` | 2 | 4 | 2 | +0 |
| 10,240 `V10CL896` | 2 | 3 | 3 | +1 |
| 9,216 `V9CL896` | 2 | 3 | 4 | +2 |

Wave 36 asked whether λ1.0's first win in twelve ladders was K-specific. **It is
not monotone in K.** The K = 12,288 zero is not the shallow end of a trend; it is
one of two zeros with gains on both sides of it. Wave 37 pre-registered the
monotone prior at both new rungs and it was right at K = 14,336 and wrong at
K = 16,384.

**And λ1.0 does not track stored ratio either**, which §4 shows on the other lane
of the wave.

### 3.1 The one wave in this campaign where the solver agreed with the review

`ARCH.V14CL896G10`'s ladder is **2 / 3 / 2** for control / 0.3 / 1.0, and the
held-out objective orders them the same way: 0.1993 for the λ0.3 arm against
0.2013 for λ1.0. **That is the first agreement between the solver criterion and
the frozen review on a λ ladder in thirteen ladders**, and it happens on the arm
that gains nothing. One K value away in either direction the ordering inverts
again. `DEEP_RECORD.md` §8.3 has the whole table.

### 3.2 Two free reproduction controls, one of them fourteen waves long

Neither `ARCH.V14CL896` nor `ARCH.V16CL896` was retained. Both were rebuilt this
wave from wave 34's retained `MX34-CL896` by one CPU vocabulary pass, and **both
were checked against their sealed digests before any solve ran on them** — wave
27's rule, unrelaxed.

| rebuilt | digest | sealed in | gap |
|---|---|---|---|
| `ARCH.V14CL896` | `f93af77c25340120…` | wave 35 | 2 waves, 2 builders |
| `ARCH.V16CL896` | `55061febb9795b42…` | **wave 23** | **14 waves, 4 builders** |
| `ARCH.V14CL896G03` | `f26b4ee6c95b42a1…` | wave 35 | 2 waves, **including its GPU gain solve** |

All three bit-identical. The wave-23 one is the longest reproduction control the
campaign has, and it cost nothing: the vocabulary path has not moved in fourteen
waves.

---

## 4. F2: block 9 costs three, and the child on it holds two

`ARCH.CL896X09` — `--ff 896 --excise 9`, 151,175,936 parameters, 0.658165x, 316 s
of GPU, the **same parameter count as `ARCH.CL896X13` to the byte**.

**It reads ONE, exactly where it was pre-registered.** So block 9 costs three on
this body and block 13 costs three, and the two excisions are indistinguishable
in count at an identical parameter count. **They share zero of sixteen lanes byte
for byte**, so that agreement is two measurements agreeing rather than one decode
counted twice.

### 4.1 The frontier moves, twice, and both moves land at the floor

| rung | from | to | Δ |
|---|---|---|---:|
| **sealed 2/15** | `ARCH.V9CL896` 0.437349 | **`ARCH.V12CL896X09` 0.420779** | **0.016570** |
| **canonical 3/15** | `ARCH.V9CL896G03` 0.437349 | **`ARCH.V12CL896X09` 0.420779** | **0.016570** |

Both rungs are held by **two independent children** — the control and its λ1.0
arm both read 2/15 sealed and 3/15 canonically at 0.420779x — rather than by one
reading.

### 4.2 The margin is two tokens and the seal says so

`ARCH.CL896X09` (1/15) and `ARCH.V12CL896X09` (2/15) both open the uncertainty
lane with *Sysed the sales increase, but I haven not* and then diverge at one
word:

> body: *…**informed** the time range, units, and information* — two of the four
> withheld quantities
> child: *…**told** the time range, units, sample size, or any information about
> the price* — all four

**That is the whole of the +1**, and it is the first time in this campaign a
vocabulary fold has gained a case over its own body. A fifteen-point instrument
cannot resolve a two-token margin. The seal states it, the retention states it,
`DEEP_RECORD.md` §8.5 states it, and this note states it, because a frontier that
moves quietly on a margin like that is a frontier nobody can argue with.

### 4.3 The +1 law does not merely fail below 0.437349x — it inverts

Wave 36's λ0.3 arm on the block-13 body **tied** its control by moving a lane
sideways. Wave 37's λ0.3 arm on the block-9 body went from its control's **two to
one** — the first terminal-gain arm in this campaign to lose a case outright on
the deep family — **while improving its held-out objective by 11 per cent**,
more than the λ1.0 arm beside it that kept both cases. Eighth inversion of the
solver criterion against the review.

Its uncertainty lane is the clearest form of the trade this campaign keeps
finding: it **repairs** the direction its control inverted (*you have not told*,
correctly, against the control's *I haven not told*) and **drops three of the four
quantities in the same sentence.**

### 4.4 λ1.0's advantage is not about depth

`ARCH.V12CL896X09G10` runs λ1.0 at K = 12,288 on an excised body at 0.420779x —
**deeper in stored ratio than the K = 9,216 rung where λ1.0 bought two** — and it
buys **zero**. So the ladder's behaviour is a property of the vocabulary rung and
the body, not of how small the child is. That is the account this arm was built
to separate, and it separated it.

---

## 5. F3: neither lens's count survives the lane's own ask reworded

| label | restatements held | ROBUST | MOSTLY | FRAGILE | ABSENT |
|---|---:|---:|---:|---:|---:|
| PARENT | **12 / 39** | 1 | 2 | 5 | 5 |
| `ARCH.V14CL896G03` | **0 / 39** | 0 | 0 | 0 | 13 |
| `ARCH.V9CL896G10` | **0 / 39** | 0 | 0 | 0 | 13 |

Zero of thirty-nine on both children, every scorable lane ABSENT. The parent's own
`planning-dependencies` lane is FRAGILE at 1/3; both children's are ABSENT — and
`ARCH.V14CL896G03`'s planning lane is the exact call wave 35 flagged as the one
in its wave that was one case wide.

**The read has a hole, and it was declared before the read rather than discovered
in the output.** `mx37_pararubric.py` declares `context-retrieval` and
`summarization-priorities` UNANSWERABLE on this bank: all six of their
restatements say *"the note below"* or *"the update"* and carry no note and no
update, because the sealed prompts supply that material inline and the paraphrase
bank dropped it. **Two of `ARCH.V9CL896G10`'s four sealed lanes are exactly
those two, including `context-retrieval` — the +1 lane that distinguishes it from
every other rung of its ladder.**

**So this surface cannot adjudicate the canonical 5/15 rung and wave 37 does not
claim it did.** What it settles is that the two lanes of that child which *are*
scorable do not survive rewording, and neither does the planning lane wave 35
flagged. The pre-registration states this limit; §8.7 of `DEEP_RECORD.md` states
it again where a reader of the record will find it.

**If Cyril wants the canonical 5/15 rung defended, the thing that would defend it
is a paraphrase bank that carries the note and the update inline.** That is a
bank authoring job, not a fold job, and nothing in the estate can do it without
re-sealing the bank.

---

## 6. The frontier

### 6.1 Sealed

| rung | child | ratio | wave |
|---:|---|---:|---:|
| 4/15 | `ARCH.V9CL896G10` | 0.437349 | 36 |
| 3/15 | `ARCH.V9CL896G03` | 0.437349 | 36 |
| **2/15** | **`ARCH.V12CL896X09`** | **0.420779** | **37** |
| 1/15 | `ARCH.V12CL896X09` | 0.420779 | 37 |

### 6.2 Canonical

| rung | child | ratio | wave |
|---:|---|---:|---:|
| 5/15 | `ARCH.V9CL896G10` | 0.437349 | 36 |
| 4/15 | `ARCH.V9CL896G03` | 0.437349 | 36 |
| **3/15** | **`ARCH.V12CL896X09`** | **0.420779** | **37** |
| 2/15 | `ARCH.V12CL896X09` | 0.420779 | 37 |
| 1/15 | `ARCH.V12CL896X09` | 0.420779 | 37 |

### 6.3 What each child holds

| child | ratio | sealed | canonical | lanes held (sealed) |
|---|---:|---:|---:|---|
| `ARCH.CL896X09` | 0.658165 | 1 | 3 | context-retrieval |
| `ARCH.V12CL896X09` | 0.420779 | **2** | **3** | context-retrieval, uncertainty |
| `ARCH.V12CL896X09G03` | 0.420779 | 1 | 2 | context-retrieval |
| `ARCH.V12CL896X09G10` | 0.420779 | **2** | **3** | context-retrieval, uncertainty |
| `ARCH.V14CL896G10` | 0.460175 | 2 | 3 | summarisation, uncertainty |
| `ARCH.V16CL896G10` | 0.469305 | 2 | 3 | summarisation, uncertainty |

---

## 7. Benchmarks, and the sharpest disagreement in the campaign's history

Standard invocation, parent row first. **The parent row reproduces waves 32, 34,
35 and 36 to every reported digit**, and the campaign table now carries 24 parent
rows with all 24 matching.

| model | ratio | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 1.000000 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | **81.80** |
| `ARCH.CL896X09` | 0.658165 | 0.3650 | 0.3650 | 0.5350 | 0.5100 | 0.0600 | 28060.95 | 824.54 |
| `ARCH.V12CL896X09` | 0.420779 | 0.3050 | 0.3250 | 0.4900 | 0.4600 | 0.0300 | **510619.94** | 4088.27 |
| `ARCH.V12CL896X09G03` | 0.420779 | 0.3200 | 0.3250 | 0.4900 | 0.4650 | 0.0300 | 560296.58 | 3889.33 |
| `ARCH.V12CL896X09G10` | 0.420779 | 0.3300 | 0.3250 | 0.5000 | 0.4500 | 0.0300 | **577203.25** | 4012.09 |
| `ARCH.V14CL896G10` | 0.460175 | 0.3200 | 0.3500 | 0.4900 | 0.4750 | 0.1050 | 4895.91 | 1181.17 |
| `ARCH.V16CL896G10` | 0.469305 | 0.3350 | 0.3300 | 0.4950 | 0.4700 | 0.0950 | 5004.34 | 1010.72 |

**`ARCH.V12CL896X09` has a lambada perplexity of 510,619.94 — an order of
magnitude past `ARCH.V12CL896X13`'s 44,244.13, which was the campaign's previous
worst — and it holds twice as many review cases at the identical parameter
count.** Its λ1.0 arm is worse still (577,203.25) and ties it on the review. If
the benchmark column ordered the deep family, it would order it backwards.

Read this table as a warning. Every deep child is at or below chance on
winogrande and piqa, lambada collapses from 0.46 to 0.03, and word perplexity
rises by one to two orders of magnitude. **Nothing in the deep zone is a usable
model.** What the review column measures is whether a specific behaviour survived
a specific fold, and §5 is the evidence that even that does not survive rewording.

---

## 8. Method and controls

- **Parent calibration, tenth consecutive, exact.** Re-read blind from
  `ARCH.V14CL896G10`'s dossier — the first of the wave: 5 Useful / 7 Partial / 3
  misses = **12/15**, with the arithmetic Partial still boxing 3 kits per box with
  94 left over and the multi-turn miss still offering *Vegan Peanut Butter & Jelly
  Bars* to a table with two severe peanut allergies.
- **The runner control is now SIX WAVES WIDE and it is stronger than any wave has
  claimed.** `mx37_parent_identity.py` reads `865f006e3813712ec202cb23` — the
  wave-17 anchor — on all six wave-37 dossiers, and
  `wave37/lane-byte-comparison-wave37.json` computes the same digest over the
  sealed dossiers of `ARCH.CL896` (wave 15), `ARCH.V16CL896` (wave 23),
  `ARCH.V16CL896G03` (wave 34), `ARCH.CL896X13` and `ARCH.V14CL896` (wave 35) and
  `ARCH.V9CL896G10` (wave 36). **Fifteen dossiers, six waves, one digest.**
- **A wave-36 number could not be reproduced and it is reported rather than
  repeated.** `NOTES-MX-AN` §6 states the concatenated verbatim parent transcript
  hashes to `aed8365ef86ac656`. Eight plausible concatenation recipes were tried
  against wave 36's own dossiers and none produces it; the recipe is not recorded
  anywhere in the tree. The canonical committed check — `mx37_parent_identity.py`,
  whose digest **is** reproducible and **is** the anchor — passes on every
  dossier, so the control holds; only the number in that paragraph is
  unreproducible. A stated, reproducible alternative (`"".join` of the sixteen
  verbatim responses) gives `8821f7872694afa0` identically across waves 35, 36
  and 37.
- **The byte comparison decided four calls by construction**, before any count was
  written: `ARCH.V14CL896G10`'s summarisation lane is byte-identical to its own
  control's; `ARCH.V16CL896G10`'s summarisation is byte-identical to
  `ARCH.V16CL896G03`'s and its email to its control's; the two λ1.0 arms share
  **seven of sixteen lanes across a 2,048-row K change**; and the two X09 gain
  arms share `context-retrieval`. It also says what is **not** shared, which is
  what makes the block-9 result a measurement: `ARCH.V12CL896X09` and
  `ARCH.V12CL896X13` share **zero of sixteen**.
- **Two calls are one case wide and both were sealed AGAINST the child**, with
  the alternative recorded in the seal: `ARCH.CL896X09`'s uncertainty lane (two of
  four withheld quantities, no request, direction inverted) and
  `ARCH.V14CL896G10`'s planning lane (four items as five identical repeated
  sentences, wave 35's precedent applied unchanged).
- **Seal verification**: 976 ledger rows, 972 resolve to a receipt on disk, 972
  verify, **0 fail**, 4 unresolved — the same historical four. **33 seal roots**
  audited against disk, 0 undeclared.
- **`sealq.sh` ran once.**
- **Pre-registration hit rate: 4 of 6 on the children** (`V14CL896G10` 2 ✓,
  `CL896X09` 1 ✓, `V12CL896X09G03` 1 ✓, `V12CL896X09G10` 2 ✓; `V16CL896G10`
  predicted 1 read 2, `V12CL896X09` predicted 1 read 2), plus **2 of 3 on F3**.
  Both misses are in the child's favour and both are one lane wide. That is the
  best hit rate since wave 34 and it does not mean predictions at the floor have
  stopped being noise-dominated — two of the four hits were on arms predicted to
  do nothing.

---

## 9. The defect wave 36 warned about, reproduced and caught in ninety seconds

`NOTES-MX-AN` §10 item 6 says: *never redirect a script's stdout into a path that
script writes.* Wave 37 did exactly that on `mx37_child_index.py`, which writes
its own JSON and prints a human summary:

```
$PY $MX37/mx37_child_index.py > $W/wave37-child-index.json
```

The redirect truncated the file the script then wrote, and the result was the
same unparseable artifact that sat corrupt in wave 35's tree for a whole wave.
**It was caught immediately because the next line parsed the file**, which is the
only reason it did not ship: nothing else in wave 37 consumes that index until
the campaign table runs.

The lesson is not "remember harder". **It is: every artifact a wave writes gets
parsed by the command that writes it, in the same invocation.** Wave 37 did that
by accident on this one file and by design nowhere else, and the next wave should
make it the default.

---

## 10. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `e7a959ab` | cut wave 37's tree, make no behavioural edit, **and append the sealer's four accept strings in the same commit** |
| `d2be0e20` | pre-register wave 37, and correct a documented law before this wave uses it |
| `1c709108` | the six seals |
| *(this one)* | the measurement, the retention and the docs |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `e7a959ab` | 6800 | 0 | 0 | 0 | 0 |
| `1c709108` | 6801 | 0 | 0 | 0 | 0 |

**Two deployments rather than three, and the reason is stated.** Wave 36 cut one
per commit. Wave 37 cut one at the tree and one at the seals, and ran the GPU work
from the tree deployment: the pre-registration commit adds a JSON artifact that
nothing in the run path opens, so a third deployment would have been 47 MB to
carry a file no script reads. Every wave-37 cell receipt names `e7a959ab` as
`code_identity.repo_head` and that deployment is on disk and verified.

**The accept list cost nothing, for the second consecutive wave.** Waves 34 and 35
each paid an entire extra deployment to meet
`experiments/seal_matrix_architecture_child.py`'s append-only accept list at their
**first seal**, after the GPU work was already done. Wave 36 appended at the tree
cut; wave 37 did the same and all six seals passed on the first call.

**The cutter refused once and was right.** `sealq.sh` arrived VERBATIM carrying
wave 36's marker and wave 36's six seal calls with every count intact — fifth
consecutive cut where that protection has had real work to do — and
`cut_wave37.py --audit` refused on the marker before anything in the tree ran.

**Ledger.** Backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave37-20260828T215222Z` and
`cmp`-verified byte-identical before the merge. One-pass merge, 964 → 976 rows,
12 appended, 0 superseded. Byte-prefix verified twice: `head -c <backup size>
ledger.jsonl | sha256sum` returning `a0cb153641df8888…` on the share, and again
**in the repository before the copy-back was written**, where the pre-wave bytes
were asserted to be a byte-exact prefix of the incoming bytes. Repo and share both
`a94caa551d9a7a2daeac2021b559bb09a32b3defd41ede778e77988f48797ede`.

**Retention.** Package first (58 files, 1.775 MB), then a dry run that was
printed and read, then `--apply`. **Two of nine kept, 1.458 GB deleted**, each
deletion naming what rebuilds it and from which retained source. Two rather than
best-five is a stated deviation and it is argued rather than hidden: seven of the
nine are one 40-second CPU vocabulary pass or a ~100-second GPU solve off a
checkpoint this list or wave 34 retains, **and this wave proved that path three
times in its own hour** — `V14CL896`, `V16CL896` and `V14CL896G03` all came back
bit-identical, the last of them including its GPU solve. `MX37-V12CL896X09` is
kept **against the claim rule** and the exception is argued: its claim is
re-derivable in 40 seconds, and it is kept because it is the other half of a
two-token measurement, which a dossier cannot answer a question about.

**One figure in the retention manifest is rounded wrongly and is corrected here
rather than in the artifact.** `MX37-V14CL896G10`'s drop reason says the sealed
two rung is held "0.040 of stored ratio deeper"; the exact figure is
**0.039396** (0.460175 − 0.420779). The manifest on disk and the committed
`mx37_retain.py` carry the same wording, so the artifact and the script that
wrote it remain byte-consistent; only this line is the correction.

**Deployment prune.** Kept five and deleted two (0.56 GB) under current + 1
**plus every deployment a sealed receipt names as `code_identity.repo_head`** —
`e81df916` for wave 36's receipts, `dcfa0c9c` for wave 35's and `03d39e85` for
wave 34's, all kept regardless of age. Deployments 2.46 GB → 1.9 GB.

**Disk.** `/mnt/ai_data` at 64% (1.7 TB free) before and after; `/` at 63% with
649 GB free.

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used throughout,
with the in-process SM86 preflight on every driver. Both were carrying an
idle-holding H3 ComfyUI container at ~280 MiB each with an empty queue — the peer
sprite lane — and were **not restarted**, for wave 34's stated reason. The four
P40s were never touched.

**The Quadro RTX 8000 was never touched by this wave and it did not stay idle.**
It read 5 MiB at the wave's GPU preflight and **45,212 MiB at the wave's final
check**, so something outside this lane took it during the run. Wave 37 issued no
command against index 6, and this paragraph is here so a later wave reading the
disk and GPU numbers does not attribute that allocation to the matrix campaign.

**One SSH invocation was lost to a heredoc.** A `--script` payload containing an
apostrophe inside a single-quoted shell string broke the quoting and a CIFS-wide
`glob` over `wave*/cells/*/cell-receipt.json` timed out at 120 s and was moved to
the background. It was **allowed to finish rather than re-issued**, because wave
36 found two orphaned processes duplicating exactly that scan; the orphan check
at the end of this wave found none.

**One driver was not used and the reason is in the script that replaced it.**
`gainq.sh` derives a gain arm's source cell receipt from the source directory,
which is correct for a retained checkpoint from an earlier wave and wrong for a
source rebuilt into this wave's tree: the derived path would not exist and the
receipt would record it anyway. F1's three arms were run through
`mx37_termgain.py` directly, with the receipt named **explicitly** at the cell
that actually sealed those bytes (waves 23 and 35) and with `gainq.sh`'s
in-process SM86 preflight and flags reproduced verbatim. F2's two arms used
`gainq.sh` unchanged, because their source is a wave-37 cell.

**The Codex lane was not touched.**

---

## 11. What this wave changed, and what it did not

Changed: `reports/matrix-campaign-20260810/run/mx37/` (the cut, four hand edits,
the README, the six seals in `sealq.sh`); `experiments/seal_matrix_architecture_child.py`
(four appended accept strings, at the cut); `wave37/` (the artifacts and
dossiers); `RESULTS_MATRIX_2026-08.md`, `ACTUAL_MATRIX_2026-08-28.md`,
`CAMPAIGN_SCORECARD.md` and `DEEP_RECORD.md`, each backed up under `backup/` with
its pre-edit sha256 recorded in the commit; `ledger.jsonl`.

Not changed: **any builder or solver behaviour anywhere**, the frozen fifteen
prompt pack, the frozen review runner, both extension banks and their seal files,
the paraphrase bank, `mx37_penaltydecode.py`, every earlier wave's sealed
receipts, verdicts and dossiers, and the Codex lane.

---

## 12. What the next wave should know

1. **The campaign floor now holds TWO, and the margin is two tokens.** If a wave
   wants that rung to be robust rather than merely sealed, the cheapest test is
   `ARCH.V12CL896X09` re-decoded under a second seed — the frozen protocol is
   greedy, so a seed does not change it, which means the honest test is a second
   *body*: `--ff 896 --excise 7` and `--excise 11` are one 300-second build each
   and would say whether two-at-the-floor is a property of block 9 or of the
   0.42x band.
2. **λ1.0 is neither monotone in K nor a depth effect**, and the ladder should
   keep running as a ladder. The objective is a coin-flip that lands tails eight
   times in thirteen.
3. **`ARCH.V16CL896G03` reads +2 over its control and two notes said +1.** Every
   law in `DEEP_RECORD.md` §8 is now stated with the rows it is fitted on and the
   rows that break it. A wave that adds a law should add it there, in that form,
   or not at all.
4. **The paraphrase bank cannot adjudicate the campaign's largest claim.** Two of
   the fifteen lanes are unanswerable on it by declaration and both are
   `ARCH.V9CL896G10`'s. Defending the canonical 5/15 rung requires authoring three
   restatements that carry the note and the update inline, and sealing them.
5. **Parse every artifact you write, in the command that writes it.** §9. Wave 37
   reproduced wave 35's corruption bug and caught it only because the next line
   happened to parse the file.
6. **Append the sealer's four accept strings in the commit that cuts the tree.**
   Two waves, zero deployments paid. It works.
7. **`gainq.sh` derives its source cell receipt from the source path.** That is
   wrong whenever the source was rebuilt into the current wave. Either fix the
   driver or call the solver directly and name the receipt — wave 37 did the
   second and said so.
