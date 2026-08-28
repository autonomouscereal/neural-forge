# NOTES-MX-AM — wave 35: the deep zone gets a floor, and the artifact clause gets a grid

**2026-08-28.** Branch `codex/lrc-blank-compression`. Eight children built,
sealed and read twice. Ledger 936 → **952** rows,
`5517dd9bc388b2dbd42a71b5b8da28419cb5db7af4871c28f5817d27c8a0c9d7`, byte-prefix
verified against the pre-wave backup two ways and again in the repository after
the copy-back. **Four deployments, one of them forced by a guard refusing
correctly for the second consecutive wave.**

---

## 1. Every parameter count landed exactly, including the one nobody had built

| child | ratio | stored | how it was made | pre-registered |
|---|---:|---:|---|:--:|
| `ARCH.V12CL896X13` | **0.420779** | 96,649,984 | K=12,288 on the new body | exact |
| `ARCH.V9CL896` | **0.437349** | 100,456,192 | K=9,216 on wave 34's `MX34-CL896` | exact |
| `ARCH.V10CL896G01/G03/G10` | **0.441915** | 101,504,768 | λ 0.1/0.3/1.0 on `MX34-V10CL896` | exact |
| `ARCH.V14CL896` | 0.460175 | 105,699,072 | K=14,336 on `MX34-CL896` | exact |
| `ARCH.V14CL896G03` | 0.460175 | 105,699,072 | λ0.3 on the above | exact |
| `ARCH.CL896X13` | 0.658165 | 151,175,936 | **NEW BODY**, `--ff 896 --excise 13` | exact |

The arithmetic was derived closed-form before any build and cross-checked
against the allocator's own constants: the block-13 excision cost on an
`--ff` body decomposes to 4,199,424 plus that site's MLP, which at the parent's
channel count is **12,063,744 — exactly the allocator's conv-block constant**.
The model also reproduced `ARCH.V9CL896` at 100,456,192 / 0.437349x, the number
`NOTES-MX-AL` §8 quotes independently.

---

## 2. The frontier: seven rungs move, and one of them is the second-largest move on record

### 2.1 Sealed

| rung | before | after | Δ |
|---:|---|---|---:|
| 1/15 | `ARCH.V10CL896` 0.441915 | **`ARCH.V12CL896X13` 0.420779** | **0.021136** |
| 2/15 | `ARCH.V10CL896` 0.441915 | **`ARCH.V9CL896` 0.437349** | 0.004566 |
| 3/15 | `ARCH.V12CL896G03` 0.451045 | **`ARCH.V10CL896G03` 0.441915** | 0.009130 |
| 4/15 | `ARCH.V12CL896G03` 0.451045 | *does not move* | — |

### 2.2 Canonical

| rung | before | after | Δ |
|---:|---|---|---:|
| 1/15 | `ARCH.V10CL896` 0.441915 | **`ARCH.V12CL896X13` 0.420779** | **0.021136** |
| 2/15 | `ARCH.V10CL896` 0.441915 | **`ARCH.V12CL896X13` 0.420779** | **0.021136** |
| 3/15 | `ARCH.V12CL896G03` 0.451045 | **`ARCH.V10CL896G01` 0.441915** | 0.009130 |
| 4/15 | `ARCH.V12CL896G03` 0.451045 | **`ARCH.V10CL896G03` 0.441915** | 0.009130 |

**The ladders still agree on ratio at rungs one to four and for the first time
they DISAGREE ON IDENTITY at one of them**: at three, the canonical lens names
`ARCH.V10CL896G01` and the sealed lens names `ARCH.V10CL896G03`, both at
0.441915x. The disagreement is one lane and it is named in §5.

### 2.3 What each child holds

| child | ratio | sealed | canonical | art | con | pre-reg sealed | pre-reg canon |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ARCH.V12CL896X13` | 0.420779 | **1** | **2** | 11 | 2 | 0 | 1 |
| `ARCH.V9CL896` | 0.437349 | **2** | **2** | 12 | 1 | 1 | 1 |
| `ARCH.V10CL896G01` | 0.441915 | **2** | **3** | 11 | 1 | 2 | 2 |
| `ARCH.V10CL896G03` | 0.441915 | **3** | **4** | 9 | 2 | 3 | 3 |
| `ARCH.V10CL896G10` | 0.441915 | **3** | **4** | 10 | 1 | 2 | 2 |
| `ARCH.V14CL896` | 0.460175 | **2** | **3** | 11 | 1 | 2 | 2 |
| `ARCH.V14CL896G03` | 0.460175 | **3** | **4** | 9 | 2 | 3 | 3 |
| `ARCH.CL896X13` | 0.658165 | **1** | **1** | 12 | 2 | 2 | 3 |

Five of eight sealed counts landed on the pre-registered number; two beat it
(`V12CL896X13` 0→1, `V9CL896` 1→2) and one missed low (`CL896X13` 2→1).

**Eighty-five of the 120 lanes — 70.8 per cent — are compression artifacts**,
against a campaign-wide 29.8 per cent. That is what the sub-0.47x band is: the
decode has collapsed nearly everywhere and only thirteen lanes in the entire wave
are readable-and-wrong.

---

## 3. The results, in the order they matter

### 3.1 The deepest child ever built is below the content death, and it holds a case

`ARCH.V12CL896X13` at **0.420779x — 42.1 per cent of the parent** — is 0.012
*below* `ARCH.V8CL896`, one of the estate's three honest content deaths. It was
pre-registered at **zero** and both available arguments pointed there: the
allocator said 3.1 with all three pair terms firing, and the estate's worst
allocator miss — `ARCH.V19X13CL896`, the only other sealed depth+mlp+vocab child
on this body, predicted 7.4 and sealed **0** — said less.

It seals **one**, on `writing-concise-email`, and it is held on the ASK rather
than the form, which is wave 21's criterion applied unchanged:

> *Dividing the delayed shipment, we are looking at the delayers and the tracking
> number. If you're concerned, we'll be aware that the delayers will be arriving
> by the Friday.*

**Its summarisation lane is sealed a MISS and the precedent is applied in the
direction that costs the child a case.** It keeps the unresolved risk verbatim
and drops the decision — *the team is not a cheapest but cannot guarantee
delivery before launch*, which is Supplier A's property attached to no supplier.
Wave 28 sealed `ARCH.V19X13CL896` at ZERO for exactly that, on exactly this
composition.

### 3.2 The 0/15 bracket is halved, from the surviving side

`ARCH.V9CL896` at 0.437349x holds **two**, so the bracket around the campaign's
floor is now **0.004565 of stored ratio — 1,024 vocabulary rows** — between
`ARCH.V8CL896` (K = 8,192, dead on both lenses) and `ARCH.V9CL896` (K = 9,216,
two on both). That is the sharpest transition anywhere in this campaign.

It also stops being *the* floor: `ARCH.V12CL896X13` reaches 0.4208 by adding an
excision instead of removing rows, and holds a case there. **So the death at
K = 8,192 is a property of that vocabulary rung on that body, not of the ratio.**

### 3.3 The λ ladder settles what wave 34 could not: +1 is the body, +2 was the base

`NOTES-MX-AL` §8 item 2 asked why `ARCH.V12CL896G03` gained **two** when every
other CL896 gain arm gains one. Wave 35 built the arm on two more bases:

| base | K | control | λ0.1 | λ0.3 | λ1.0 |
|---|---:|---:|---:|---:|---:|
| `ARCH.V10CL896` | 10,240 | 2 | 2 | **3** | **3** |
| `ARCH.V14CL896` | 14,336 | 2 | — | **3** | — |
| `ARCH.V12CL896` (wave 34) | 12,288 | 2 | 2 | **4** | 2 |

**The CL896 gain arm is worth exactly +1 at K = 10,240, 14,336, 16,384, 19,456
and 20,480, and the +2 at K = 12,288 is a property of that one base.** The lane
it buys is the same every time: the first-person email restatement.

**λ1.0 moved a sealed count up for the first time in ten ladders** — and the
byte comparison says why it is not a coincidence. `ARCH.V10CL896G10` shares
**ten of sixteen lanes byte-identically with `ARCH.V10CL896G03`, including
`writing-concise-email`**. On the lane that carries the +1 the two arms are not
similar; they are the same bytes. Their objectives, meanwhile, order the arms in
the *opposite* direction to the review: λ0.1 has the largest update (0.1119) and
the best held-out relSSE (0.1906 → 0.1817) and buys nothing. **Sixth wave to
record an objective improving while the review does not move.**

### 3.4 The excision costs three cases, on both bodies, and that replicates

`ARCH.CL896X13` is the first `--ff 896 --excise 13` body any wave has sealed. It
reads **one** against a prediction of two and against the allocator's 4.1.

| body | without X13 | with X13 | cost |
|---|---:|---:|---:|
| `ARCH.CL1792` | 7 | 4 | **3** |
| `ARCH.CL896` | 4 | 1 | **3** |

**The price of block 13 on this parent is three sealed cases and it is
independent of the MLP channel count it is composed onto.** That is a
replication, and it is why the body was worth its 296 s of GPU even though it
takes no rung: it turns wave 31's single observation into a law with two points
on two very different bodies.

### 3.5 D3 was refused by a guard, and the refusal is recorded rather than routed around

The wave was ordered to build a `W896` + `CL1792` composed body. `mx35_build.py`
`main()` **refuses** it:

> `--hidden-size` already re-solves every reader and every writer in the model
> against a new residual basis; composing `--ff` or `--head-dim` on top of it in
> one build would run a second selection inside tensors the first one has already
> replaced, with neither axis's census having seen the other. Compose it with
> `--excise` or `--kv-groups`, which are the two axes whose machinery is
> width-agnostic.

The guard was read in the source **before** the pre-registration was written, the
refusal was registered there as the expected outcome, and then it was
**exercised on the server so the exact text is on the record**. It was not routed
around: the machinery that would compose two bodies *sequentially* does not exist
in this estate, and building it to satisfy a build order is how a guard becomes
decorative. The arithmetic is registered anyway — 168,410,240 for the body,
120,700,032 for K=12,288 on it — so a later wave that builds the composition
machinery has a number to check itself against, and the pre-registration says in
terms that no frontier row may cite it.

The width family could not have reached the deep zone anyway. `ARCH.W896` is a
0.859x body against `ARCH.CL896`'s 0.688x, and the deepest thing composable on it
that the guard permits — `ARCH.V16W896X13` — already exists at 0.623329x.

### 3.6 D4: the artifact clause, priced across a grid, with the screen's own null

The canonical standard strikes 846 lanes — 43.6 per cent of every behavioural
failure in this campaign — as "fixable with token penalties and retraining".
Wave 34 priced **one value of one knob** (repetition penalty 1.3) and found it
converted every loop into fluent nothing. 1.3 is the aggressive end, so wave 35
priced the mild end and the other mechanism the phrase actually covers, **on wave
34's own five children** so the rows compare rather than merely resemble.

| setting | artifact readable | flagged | **recovered** | signal held |
|---|---:|---:|---:|---:|
| p1.0 n0 — **identity / screen null** | 31 / 37 | 9 | **0** | 13 / 13 |
| p1.1 n0 | 37 / 37 | 8 | **0** | 10 / 13 |
| p1.15 n0 | 37 / 37 | 7 | **0** | 6 / 13 |
| p1.0 n3 — the hard ban alone | 31 / 37 | 6 | **0** | 10 / 13 |
| p1.1 n3 | 37 / 37 | 5 | **0** | 9 / 13 |
| p1.3 n0 — **wave 34's row** | 37 / 37 | 7 | **0** | 0 / 13 |

**Twenty child-settings, 300 lanes, zero hand-confirmed recoveries, and every
setting net-negative.** The damage is monotone and complete — 1.1 costs three of
thirteen, 1.15 costs seven, 1.3 costs all thirteen — and the recovery side is
flat at zero the whole way down.

**The screen now has a null, and that is wave 35's methodological addition.** It
was run on the sealed decode itself and flags **nine** artifact lanes as
readable-and-keyed; all nine are false positives. Every real setting flags
*fewer* than that. Wave 34 hand-checked its seven flags and found seven false
positives; this wave adds the number a setting that does nothing at all
produces. All 35 flags across all five settings were read at full length and
**35 of 35 are false positives.**

**The mechanistic finding is the part worth keeping.** Every arm with a penalty
in it makes 37 of 37 artifact lanes readable. **The hard n-gram ban alone leaves
six unreadable — the same six the sealed decode fails on** — because a loop with
a period longer than three tokens is out of its reach. So the readability the
penalty buys is an artifact of forcing novel tokens into the argmax, not of
removing something that was suppressing an answer. The n-gram arm is also the
one with the strongest prior reason to work — it leaves every non-repeating
token at *exactly* its sealed logit, so it cannot flatten a lane that was
answering — and it recovers nothing either.

The nearest miss in the entire grid, recorded verbatim because it is the closest
anything came (`p1.0 n3`, `V10CL896`, `context-retrieval`):

> *The North gate has been provided this month. What is the tool checkout for you?*

It surfaces `North`, which this child's sealed decode does not. It also says the
gate was *provided* rather than closed, gives no day and no time, and inverts
into a question back at the asker. **A correct token is not an answer.**

This does not refute the canonical standard — a lane with no readable text is
still no evidence in either direction, which is the standard's actual claim. It
refutes the **remedy**, now across six settings and two mechanisms.

---

## 4. Method and controls

- **Parent calibration, eighth consecutive, in a stronger form.** Re-read blind
  from `ARCH.V14CL896`'s dossier: 5 Useful / 7 Partial / 3 misses = **12/15**,
  reproducing the sealed reference exactly, arithmetic still boxing 3 kits per
  box with 94 left over and multi-turn still offering *Vegan Peanut Butter &
  Jelly Bars* to a table with two severe peanut allergies. **All EIGHT dossiers
  carry ONE parent transcript**, hashed and compared lane by lane, so the runner
  is provably fixed across the whole wave rather than sampled at its start.
- **The byte-comparison control decided more lanes than in any previous wave.**
  All three `V10CL896` gain arms decode `summarization-priorities`,
  `logic-ordering` and `coding-deduplicate` byte-identically to their control;
  `G03` and `G10` share **ten of sixteen**; `V14CL896G03` shares **zero** with
  its control, so its +1 is not an artifact of an unchanged decode.
- **Two reproduction results, both checked rather than assumed.** The D4 grid
  needed two children wave 34's retention had deleted. Both were re-made from
  retained sources and both came back **bit-identical to wave 34's sealed
  digests** — `ARCH.V16CL896G03` **including its GPU terminal-gain solve** — and
  both then reproduced their sealed dossiers **byte for byte on 16 of 16 lanes**
  under the identity control. That is a stronger statement than a digest match:
  the rebuilt bytes *decode* the same.
- **Every lane of every child is charged.** At and below 0.46x nothing in this
  wave terminates without a cap-ride, a prompt echo or a substitution signature,
  so every case held is held *inside* a charged lane. Wave 23's rule that the two
  readings are independent is what makes those cases sealable at all.
- **Two calls are one case wide and both were sealed AGAINST the child**, with
  the alternative recorded in the seal: `V14CL896G03`'s planning lane (a reader
  who credits its five named items seals four) and `V12CL896X13`'s summarisation
  lane (wave 28's paraphrase precedent).
- **Benchmarks**: nine rows, standard invocation. The parent row is identical to
  waves 32 and 34's to every reported digit — arc 0.5450, hellaswag_norm 0.5050,
  piqa_norm 0.6800, winogrande 0.5500, lambada 0.4600, lambada ppl 24.39,
  wikitext word ppl 81.80. **`bench/benchmark-table.json` is 2,258 bytes**; wave
  33's was zero, and that loss is not recurring.
- **Seal verification**: 952 ledger rows, 948 resolve to a receipt on disk, 948
  verify, **0 fail**, 4 unresolved — the same historical four. **31 seal roots**
  audited against disk, 0 undeclared.
- **`sealq.sh` ran once.**

---

## 5. The one lane that separates the two standards, and how to subtract it

Six of eight children read one higher canonically than sealed, and **in five of
those six the single flipped lane is `planning-dependencies`.** The shape is the
same every time: the child names backups, DNS, TLS, verification and rollback —
all five work items, correctly — and then repeats one sentence as five identical
numbered steps instead of sequencing them. The frozen protocol seals that a miss
under wave 23's precedent that items which are one item repeated do not multiply
into steps. The canonical standard reads it signal, because five correct,
specific, usable elements are present in the child's own words.

**This is the first wave whose sealed-canonical gap is dominated by one lane and
one precedent rather than distributed across the artifact class**, and it is
stated here so a reader can subtract it: strike that rule and wave 35's canonical
column becomes 17 rather than 22, and the canonical 4/15 rung does not move.

---

## 6. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `de749f68` | cut wave 35's tree; the third consecutive cut where `sealq.sh` refused |
| `33070478` | pre-register wave 35, including the child that cannot be built |
| `dcfa0c9c` | append wave 35 to the sealer's accept list, after it refused four reviews |
| `2ae30d47` | the eight seals, and the controls that decided their load-bearing lanes |
| `53928360` | the decode-repair grid, the canonical re-read, the retention policy |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `de749f68` | 6325 → 6486 | 0 | 0 | 0 | 0 |
| `dcfa0c9c` | 6487 | 0 | 0 | 0 | 0 |
| `2ae30d47` | 6487 | 0 | 0 | 0 | 0 |
| `53928360` | 6489 | 0 | 0 | 0 | 0 |

**Four deployments, and the one extra was bought by a guard refusing correctly
for the SECOND CONSECUTIVE WAVE.** `experiments/seal_matrix_architecture_child.py`'s
append-only accept list refused all four of this wave's round-A reviews, on
exactly the four tags wave 34 was refused on. Wave 34 wrote down why that cost is
asymmetric — the builder's list refuses at the *tree cut* for one line of hand
edit, this one refuses at the *first seal* after the GPU work is done — and wave
35 read that paragraph and still did not pre-empt it. **The fix is in the file
rather than in a note**, beside the four new strings, because a note is not where
the next wave will be looking. The cutter's audit cannot be extended to cover it:
that file lives in the Codex lane's `experiments/` tree, so a per-wave cutter has
no standing to assert anything about it. What a later wave *can* do is append its
four strings in the same commit that cuts its tree.

**The cutter refused once more, and correctly.** `mx35_retain.py`'s DROP reasons
name the wave-34 checkpoints that rebuild the deleted children — which the
retention rule *requires* them to do — and `MX34-` is a forbidden needle. The fix
was a **declaration at a pinned count of six**, not a rewording, because
suppressing the reference would have suppressed the thing the rule asks for.

**Package, then retention, then prune.** The bundle is 89 files and 1.83 MB;
128 files are committed under `wave35/`. Retention kept **four** and deleted
seven (1.52 GB), each deletion naming what rebuilds it and from which retained
source, after a dry run that was printed and read. Four rather than best-five is
a stated deviation: three of the four hold a frontier rung and the fourth is the
wave's only conditioned capture. **`MX35-V10CL896G10` ties the 3/15 arm and is
dropped anyway**, on the strength of a digest comparison showing ten of its
sixteen lanes are byte-identical to the arm that is kept. The deployment prune
kept four and deleted three (0.81 GB) under current+1 **plus every deployment a
sealed receipt names as `code_identity.repo_head`** — `dcfa0c9c` for all sixteen
wave-35 receipts and `03d39e85` for wave 34's, so both are kept regardless of
age.

**Ledger.** Backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave35-…` and `cmp`-verified
byte-identical before the merge. One-pass merge, 936 → 952 rows, 16 appended, 0
superseded. Byte-prefix verified **three** ways: the merge tool's own post-write
check, `head -c <backup size> ledger.jsonl | sha256sum` returning `cff07ea4…`,
and **again in the repository after the copy-back**, where the pre-wave bytes
were asserted to be a prefix of the incoming bytes *before* they were written.
Repo and share both `5517dd9bc388b2dbd42a71b5b8da28419cb5db7af4871c28f5817d27c8a0c9d7`.

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used throughout,
with the in-process SM86 preflight on every driver. Both were carrying an
idle-holding H3 ComfyUI container at ~256 MiB each with an empty queue — the peer
sprite lane — and were **not restarted**, for wave 34's stated reason: 256 MiB on
a 24 GB card is not worth taking a peer's container down for, and this wave's
peak was 1.9 GB. The Quadro RTX 8000 and the four P40s were never touched.

**Disk.** `/mnt/ai_data` at 64% (1.7 TB free) before and after; `/` at 63% with
650 GB free, deployments down from 2.3 GB to 1.5 GB.

**The Codex lane was not touched.** Nothing outside the explicit paths in §7 was
staged.

---

## 7. What this wave changed, and what it did not

Changed: `reports/matrix-campaign-20260810/run/mx35/` (the cut, four hand edits,
the README, the `--no-repeat-ngram` arm in `mx35_penaltydecode.py`, the eight
seals in `sealq.sh`, the retention policy, a third declared historical reference
in `cut_wave35.py`); `experiments/seal_matrix_architecture_child.py` (four
appended accept strings); `wave35/` (128 artifacts and dossiers);
`DEEP_RECORD.md` (new); `ACTUAL_MATRIX_2026-08-28.md` and
`CAMPAIGN_SCORECARD.md`, each backed up under `backup/` with its pre-edit sha256
recorded in the commit; `ledger.jsonl`.

Not changed: the frozen fifteen prompt pack, the frozen review runner, both
extension banks and their seal files, `RESULTS_MATRIX_2026-08.md`, every earlier
wave's sealed receipts, verdicts and dossiers, and the Codex lane.

## 8. What the next wave should know

1. **The 0/15 bracket is 0.004565 wide and still empty.** `ARCH.V8CL896` (K =
   8,192) is dead on both lenses; `ARCH.V9CL896` (K = 9,216) holds two on both.
   There is no rung between them on this axis — 1,024 rows is one step — so
   closing it further means changing what a step is, not building another child.
2. **The deep frontier's floor is now an EXCISION, not a vocabulary rung.**
   `ARCH.V12CL896X13` at 0.420779x holds the 1/15 rung, and the obvious
   unbuilt corners are `V10CL896X13` (0.411648x) and `V9CL896X13` (0.407083x),
   each one CPU pass on the RETAINED `MX35-CL896X13`. Nobody knows whether the
   email lane survives another 2,048 rows down there.
3. **The λ ladder is closed on this body.** +1 at every K except 12,288, where
   it is +2. Do not re-run the ladder on a CL896 base expecting anything else;
   the open question is *why* 12,288, and that is a mechanism question, not a
   build.
4. **The artifact clause's remedy is now refuted across six settings and two
   mechanisms**, with the screen calibrated against its own null. Any claim built
   on "846 lanes are recoverable" needs a mechanism that is not a decode-time
   knob. Retraining is the untested half and it is a stage-2 activity under the
   staged escalation policy, so it may not be prototyped while stage 1 is open.
5. **`--hidden-size` cannot be composed with `--ff` in one build and should not
   be made to.** The width×MLP body is unbuildable by design. If a wave wants it,
   the honest route is *sequential* composition machinery — build one body, then
   fold the second axis against a fresh census of the first's output — and that
   is a wave of tool-building, pre-registered as such, not a flag change.
6. **Append the sealer's four accept strings in the same commit that cuts the
   tree.** Two consecutive waves have now paid a whole deployment for that line.
