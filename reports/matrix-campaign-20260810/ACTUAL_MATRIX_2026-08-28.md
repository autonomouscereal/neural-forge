# THE ACTUAL MATRIX — Neural Forge fold campaign, re-read 2026-08-28

**What this file is.** A hand re-read of every reviewed whole-model child's verbatim
dossier against Cyril's canonical standard, which supersedes prior grading
*interpretations for assessment only*. No sealed artifact was modified, no ledger row
was written, and no measurement was re-run. The sealed columns are carried alongside
every re-read column so the two can always be compared.

> [Cyril quote withheld]

## The grading rule, stated so it can be argued with

Every one of the frozen fifteen lanes is put in exactly one of three boxes.

| box | definition | counts against intelligence? |
|---|---|---|
| **intelligence signal** | sealed Useful or Partial, **or** a sealed miss whose text contains at least one correct, specific, usable element of what the lane asked for, in the child's own words | no — this is the evidence |
| **artifact-miss** | empty, pure loop, cap-ride with no unique content, indiscriminate prompt copy, or multilingual token salad | **no** — a compression artifact, and no evidence in either direction |
| **content-miss** | readable, on-register output that is wrong, vacuous, deflecting, or a restatement with no answer in it | **yes** — this is the only kind of miss that is about intelligence |

**INTELLIGENCE SCORE = signal lanes / 15.** Degeneracy vetoes are lifted for assessment;
the robustness/restatement column is diagnostic only. Sub-0.5x children with any signal
are successes.

Two rules earn their keep and are worth naming, because they cut in opposite directions:

- **Purposive reuse of the prompt is signal; indiscriminate reproduction is artifact.** A
  child that returns the whole supplier paragraph including the instruction line has
  copied, not answered — `ARCH.V8CL896` does exactly this and passes a naive answer key
  on it. A child that selects *the risk sentence* and frames it as the risk has answered.
- **Looping after content is free.** `ARCH.GV12L01` gives three practical actions and then
  repeats one of them eighteen times into the cap. The three actions are the finding.

## Coverage

- **222 whole-model children hand re-read** (188 in the original pass, plus wave 34's eight, wave 35's eight, wave 36's six, wave 37's six and wave 38's six, each re-read in the same pass as its own seal)
- **188 whole-model children hand re-read** from their verbatim `READ-*.txt` dossiers
  (2,835 scored lanes). Every child that ever appeared on a frontier ladder, the campaign
  scorecard or the wave-33 robustness table is in that set.
- 290 whole-model children carry a sealed behaviour row. The 102 not re-read are listed in
  §5 with their sealed counts as a **lower bound**: 35 have a `READ-` file that retention
  emptied (dossier-lost, per-case JSON survives), 67 are `COMP.MA.*`/`COMP.MB.*` composed
  children from waves 6–13 that never got a per-child dossier in the repo.

## 1. The intelligence-per-compression frontier under the canonical standard

Deepest child holding at least N of 15 lanes of intelligence signal, with the sealed
frontier from `RESULTS_MATRIX_2026-08.md` beside it.

| at least | child (canonical) | net ratio | intel | sealed | sealed-frontier child | sealed net | deepened by |
|---:|---|---:|---:|---:|---|---:|---:|
| 14/15 | `ARCH.V12` | **0.762614** | 14 | 11 | `ARCH.V19` | 0.794570 | **0.031956** |
| 13/15 | `ARCH.BP64` | **0.742049** | 13 | 10 | `ARCH.V18` | 0.790005 | **0.047956** |
| 12/15 | `ARCH.V18X09` | **0.737484** | 12 | 10 | `ARCH.V18` | 0.790005 | **0.052521** |
| 11/15 | `ARCH.V18X09` | **0.737484** | 12 | 10 | `ARCH.B75` | 0.742049 | **0.004565** |
| 10/15 | `ARCH.V12X13` | **0.710093** | 10 | 7 | `ARCH.V18X09` | 0.737484 | **0.027391** |
| 9/15 | `ARCH.V8X13GL01` | **0.691832** | 9 | 7 | `ARCH.V18X07` | 0.737484 | **0.045652** |
| 8/15 | `ARCH.GV16L01` | **0.637073** | 8 | 8 | `ARCH.GV16L01` | 0.637073 | — |
| 7/15 | `ARCH.V8CL1792` | **0.600552** | 7 | 5 | `ARCH.GV12L03` | 0.618813 | **0.018261** |
| 6/15 | `ARCH.V8CL1792` | **0.600552** | 7 | 5 | `ARCH.V9CL1792` | 0.605117 | **0.004565** |
| 5/15 | `ARCH.V9CL896G10` | **0.437349** | 5 | 4 | `ARCH.V8CL1792` | 0.600552 | **0.163203** |
| 4/15 | `ARCH.V9CL896G03` | **0.437349** | 4 | 3 | `ARCH.V9CL896G10` | 0.437349 | — |
| 3/15 | `ARCH.V12CL896X09` | **0.420779** | 3 | 2 | `ARCH.V9CL896G03` | 0.437349 | **0.016570** |
| 2/15 | `ARCH.V10CL896X09G10` | **0.411648** | 2 | 2 | `ARCH.V10CL896X09G10` | 0.411648 | — |
| 1/15 | `ARCH.V10CL896X09` | **0.411648** | 1 | 1 | `ARCH.V10CL896X09` | 0.411648 | — |
| 0/15 | `ARCH.V12CL896X09X13` | **0.390512** | 0 | 0 | `ARCH.V12CL896X09X13` | 0.390512 | — |

**REFRESHED FOR WAVES 34 THROUGH 38.** The ladder above is now computed rather than
typed: the sealed column is read off `ledger.jsonl`'s 988 rows and the canonical column
off the re-read tables in this file plus the sealed canonical re-read artifacts of waves
34, 35, 36, 37 and 38. **Ties are broken by cell id**, so where two children sit at the same
ratio the table names one deterministically and the other is not a different result —
`ARCH.GV16L01` and `ARCH.V16CL1792` are both 8/15 at 0.637073x, `ARCH.GV12L03` and
`ARCH.V12CL1792` are both 7/15 at 0.618813x, and `ARCH.BP64`, `ARCH.B75` and
`ARCH.V19X09` all sit at 0.742049x. **A gap of "—" is a tie to within the six decimal
places this file prints**, which is the precision the canonical column was re-read at;
the sealed column carries the ledger's full precision and the two agree to 1e-6.

**WAVE 38 MOVED THE CANONICAL 1/15 AND 2/15 RUNGS TO 0.4116484014x AND THE TWO STANDARDS
AGREE ON ALL SIX OF ITS CHILDREN.** `ARCH.V10CL896X09` reads **1/15 on both standards**
and its λ1.0 arm `ARCH.V10CL896X09G10` reads **2/15 on both**, 0.0091302317 of stored
ratio below wave 37's rungs. **This is the first wave since 33 where the canonical lens
moved nothing on any child**, and the reason is in the material rather than in the rule:
**61% of wave 38's ninety scored lanes are artifacts** — loops, cap-rides, markdown-marker
salad and one empty string — and the canonical lens exists to forgive exactly those. It
did that 55 times and had nothing left to forgive into. Below roughly 0.43x the canonical
column stops being an independent reading.

**WAVE 37 MOVED THE CANONICAL 3/15 RUNG TO ITS FLOOR AND THE TWO STANDARDS AGREED AT THE
BOTTOM THREE RUNGS ON THE SAME CHILD.** `ARCH.V12CL896X09` reads **3/15
canonically and 2/15 sealed at 0.420779x**, and its λ1.0 arm `ARCH.V12CL896X09G10` reads
the same counts at the same ratio, so both of wave 37's frontier moves are held by two
independent children rather than one reading. The canonical move is 0.016570 of stored
ratio off `ARCH.V9CL896G03`. **That rung is unchanged by wave 38** — nothing this wave
built reads three on either lens.

**The margin under that move is two tokens and this file will not hide it.** The body
`ARCH.CL896X09` and its K = 12,288 fold both open the uncertainty lane *Sysed the sales
increase, but I haven not* and diverge at one word — the body names two of the four
withheld quantities, the fold names all four. The canonical standard reads the body's
two as signal and the sealed standard does not, which is why the two lenses separate by
one case on that child and by one on the fold as well.

**The two standards also AGREE at three consecutive rungs — 1, 2 and 3 of fifteen, on
the same child — and the largest single disagreement in the campaign is still at five.** `ARCH.V9CL896G10`
reads **5/15 canonically at 0.437349x** where the sealed lens holds five only at
0.600552x, a gap of **0.163203 of stored ratio**. That gap rests entirely on one rule
about one lane, and `wave36/canonical-reread-wave36.json` states the subtraction: strike
the `planning-dependencies` rule wave 35 wrote down in `NOTES-MX-AM` §5 and that child
reads four canonically, the canonical 5/15 rung stays at 0.600552x, and every other rung
above is unchanged.

Everywhere else the picture is the one wave 33 recorded: the canonical standard finds the
same depth of intelligence in a *smaller* child, because the sealed reading charged lanes
that were loops, echoes and cap-rides.

## 2. The sub-0.5x successes

Under the canonical standard a sub-0.5x child with any signal is a success. Before wave
34, **eight of the eleven children ever reviewed below half the parent were successes**,
and two of those eight — `ARCH.V19X13CL896` and `ARCH.KKTS` — were sealed at **zero of
fifteen**. Wave 34 built **six more sub-0.5x children and every one of them holds**, so
the count is now **fourteen of seventeen**, and the three that do not hold are still the
same three: `ARCH.V8CL896`, `ARCH.W512` and `ARCH.KKTD`. Every child in this table carries
the sealed verdict `dead`.

Wave 34's six, wave 35's seven and wave 36's three are marked. Every one of those waves
read the canonical count in the same pass as the seal. On wave 34's six the two readings
agreed exactly; on wave 35's seven, six read one HIGHER canonically, and in five of those
six the single flipped lane is `planning-dependencies` — the child names backups, DNS,
TLS, verification and rollback and then loops instead of sequencing them. Strike that one
rule and wave 35's canonical column drops from 22 to 17 and the canonical 4/15 rung does
not move.

**Wave 36 flipped that lane twice more and did not have to judge it either time.** Both
of its flipped lanes are BYTE-IDENTICAL to lanes wave 35 already read — `ARCH.V9CL896G03`
shares its planning lane with `ARCH.V10CL896G03` (sha `8dd9e53045f9a211`, 3,490 bytes) and
`ARCH.V9CL896G10` shares its with `ARCH.V10CL896G10` (sha `307c301603c0ba3f`, 1,020
bytes). Wave 33's byte-comparison control had only ever been applied to the sealed
reading; wave 36 applied it to the assessment layer, which takes the two flips out of the
reviewer's hands entirely. The subtraction is unchanged and still available: strike the
rule and wave 36's canonical column is 1/0/0/3/4/1 instead of 1/0/0/4/5/1, and the
canonical 5/15 rung does not move.

**The whole sub-0.5x band, all twenty-four children, is now in `DEEP_RECORD.md`.**

| child | net ratio | stored params | sealed | **intel** | artifact-miss | content-miss | what it can still do |
|---|---:|---:|---:|---:|---:|---:|---|
| `ARCH.V12CL896X13` | **0.420779** | 96,649,984 | 1 | **2/15** | 11 | 2 | **wave 35. THE DEEPEST CHILD EVER BUILT**, below the V8CL896 death, and it writes a vendor email naming Friday and the tracking number. |
| `ARCH.V12CL896X13G03` | **0.420779** | 96,649,984 | 1 | **1/15** | 11 | 3 | **wave 36.** The terminal-gain arm at the campaign floor. It TIES its control's one and LOSES a canonical case: it repairs the summarisation lane and hands the email prompt back verbatim. |
| `ARCH.V8CL896` | **0.432784** | 99,407,616 | 0 | **0/15** | 7 | 8 | the floor. Copies its prompts back verbatim and answers none of them: seven artifact lanes, eight readable-and-empty. The copy path outlived the answer path. |
| `ARCH.V9CL896` | **0.437349** | 100,456,192 | 2 | **2/15** | 12 | 1 | **wave 35.** Halves the 0/15 bracket to 0.0046 from the surviving side; names all four withheld quantities. |
| `ARCH.V9CL896G03` | **0.437349** | 100,456,192 | 3 | **4/15** | 9 | 2 | **wave 36.** +1 over its control on the first-person vendor email, on the same BYTES `ARCH.V10CL896G03` bought it with. Extends wave 35's law down to K = 9,216. |
| `ARCH.V9CL896G10` | **0.437349** | 100,456,192 | 4 | **5/15** | 8 | 2 | **wave 36. THE DEEPEST FOUR SEALED AND THE DEEPEST FIVE READ.** lambda 1.0 beats lambda 0.3 for the first time in twelve ladders, on a retrieval lane it shares with neither: *The latest checkout time is 4:15 p.m.* |
| `ARCH.V10CL896` | **0.441915** | 101,504,768 | 2 | **2/15** | 11 | 2 | **wave 34.** The deepest child ever built. Eleven artifact lanes and only two readable-and-wrong: the decode has collapsed and the content has not. |
| `ARCH.V10CL896G01` | **0.441915** | 101,504,768 | 2 | **3/15** | 11 | 1 | **wave 35.** lambda 0.1: largest update, best objective, buys nothing sealed. |
| `ARCH.V10CL896G03` | **0.441915** | 101,504,768 | 3 | **4/15** | 9 | 2 | **wave 35. THE DEEPEST THREE SEALED AND THE DEEPEST FOUR READ.** |
| `ARCH.V10CL896G10` | **0.441915** | 101,504,768 | 3 | **4/15** | 10 | 1 | **wave 35.** lambda 1.0 ties lambda 0.3, on ten byte-identical lanes. |
| `ARCH.V12CL896` | **0.451045** | 103,601,920 | 2 | **2/15** | 6 | 7 | **wave 34.** The zero-solve control of the lambda ladder. |
| `ARCH.V12CL896G01` | **0.451045** | 103,601,920 | 2 | **2/15** | 9 | 4 | **wave 34.** lambda 0.1; "the latest checkout time is 4:15 MINUTES". |
| `ARCH.V12CL896G03` | **0.451045** | 103,601,920 | 4 | **4/15** | 8 | 3 | **wave 34. FOUR at 0.4510x** -- summary, email, the four withheld quantities, and 4:15 p.m. |
| `ARCH.V12CL896G10` | **0.451045** | 103,601,920 | 2 | **2/15** | 8 | 5 | **wave 34.** lambda 1.0; four lanes byte-identical to its source. |
| `ARCH.V19X13CL896` | **0.452734** | 103,990,016 | 0 | **1/15** | 10 | 4 | one lane of intelligence at 0.4527x: it compresses the supplier update in its own words and lands the unresolved risk -- a one-week factory-expansion slip. |
| `ARCH.V14CL896` | **0.460175** | 105,699,072 | 2 | **3/15** | 11 | 1 | **wave 35.** The interior rung; the body prior beat the allocator by 1.3 cases. |
| `ARCH.V14CL896G03` | **0.460175** | 105,699,072 | 3 | **4/15** | 9 | 2 | **wave 35.** Settles that the CL896 gain arm is worth +1 and V12's +2 is K-specific. |
| `ARCH.W512` | **0.463481** | 106,458,368 | 0 | **0/15** | 15 | 0 | fifteen for fifteen artifact. Nothing readable came out at all, so nothing here is evidence about intelligence in either direction. |
| `ARCH.V16CL896` | **0.469305** | 107,796,224 | 1 | **2/15** | 5 | 8 | summarises and asks for the four things it was not told. Under half the parent and it still knows what it does not know. |
| `ARCH.V16CL896G03` | **0.469305** | 107,796,224 | 3 | **3/15** | 6 | 6 | **wave 34.** The family's missing gain arm at K=16,384; three, on the lanes its two larger siblings hold. |
| `ARCH.V18CL896` | **0.478436** | 109,893,376 | 2 | **2/15** | 6 | 7 | two summarisation-family lanes and nothing else; the other seven readable lanes are on-register and empty. |
| `ARCH.V19CL896` | **0.483001** | 110,941,952 | 2 | **2/15** | 5 | 8 | the same two lanes as V18CL896 at the same body -- the vocabulary axis buys ratio without buying a new failure. |
| `ARCH.V19CL896G03` | **0.483001** | 110,941,952 | 3 | **4/15** | 7 | 4 | the deepest child in the campaign to hold FOUR: summary, email, uncertainty, and it retrieves 4:15 from the note. 0.483x. |
| `ARCH.KKTD` | **0.483001** | 110,941,952 | 0 | **0/15** | 3 | 12 | the honest content death. Three artifact lanes and TWELVE readable, fluent, on-register lanes that say nothing -- 'the team has to contact the following email address: https://www.com'. |
| `ARCH.KKTS` | **0.483001** | 110,941,952 | 0 | **2/15** | 9 | 4 | sealed dead by degeneracy, and it still orders a migration (backups, then TLS, then verify) and names the four withheld quantities. |
| `ARCH.V20CL896` | **0.487566** | 111,990,528 | 2 | **2/15** | 5 | 8 | two lanes at 0.4876x; the context-retrieval lane is the question quoted back at itself. |
| `ARCH.V20CL896G03` | **0.487566** | 111,990,528 | 3 | **4/15** | 7 | 4 | four lanes at 0.4876x, the terminal-gain twin of V19CL896G03, and it retrieves the same 4:15. |

Extending to the whole sub-0.55x band, which is where the campaign's compression claims live:

| child | net ratio | sealed | **intel** | artifact | content | character |
|---|---:|---:|---:|---:|---:|---|
| `ARCH.V32CL896` | **0.542347** | 2 | **2/15** | 7 | 6 | the largest table on the smallest MLP: the supplier summary and the missing-quantities list survive; it AFFIRMS the seasons misconception. |
| `ARCH.V8CL1792X13` | **0.558303** | 1 | **2/15** | 6 | 7 | **wave 34.** 4:15 p.m. AND Saturdays off the note, and an email naming the Monday installation in its right slot. |

## 3. Artifact versus content: how much of the campaign's death was an artifact

Across the 188 hand re-read children, 2,835 scored lanes. **Wave 34's eight new children
are reported separately below rather than pooled in**, because they were read in the same
pass as their own seal and pooling them would let a wave grade its own contribution to a
campaign statistic:

| | lanes | share of all lanes | share of the sealed misses |
|---|---:|---:|---:|
| **intelligence signal** | **1076** | **38.0%** | — |
| of which sealed Useful/Partial | 894 | 31.5% | — |
| of which recovered by this re-read | 182 | 6.4% | 9.4% |
| **artifact-miss** (loop / echo / empty / cap / salad) | **846** | **29.8%** | **43.6%** |
| **content-miss** (readable and wrong) | **913** | **32.2%** | **47.0%** |

**43.6% of everything this campaign scored as a behavioural failure is a compression
artifact carrying no evidence about intelligence at all.** A further 9.4% is content
the sealed review charged as a miss and this re-read reads as an answer. Only 47.0% of
the sealed misses are the thing Cyril means by a miss: a model that produced readable
output and got it wrong.

### Wave 34's own eight, counted separately

| | lanes | share of the 120 |
|---|---:|---:|
| intelligence signal | **19** | 15.8% |
| artifact-miss | **61** | 50.8% |
| content-miss | **40** | 33.3% |

**Half of everything wave 34's children produced is an artifact**, against 29.8% over the
188. That is what a deep population looks like: five of these eight sit below 0.47x, and
the artifact share rises as the ratio falls. `ARCH.V10CL896` at 0.4419x is 11 of 15
artifact, `ARCH.V12CL1792X13` at 0.5766x is 7 of 15, and the two are 0.13 of stored ratio
apart.

**And this is exactly the population wave 34's C3 diagnostic tested the artifact clause
on** -- see `wave34/penalty-decode-diagnostic.json`. The answer is in §3a.

### Wave 36's own six, counted separately

| | lanes | share of the 90 |
|---|---:|---:|
| intelligence signal | **11** | 12.2% |
| artifact-miss | **61** | 67.8% |
| content-miss | **18** | 20.0% |

**Sixty-eight per cent of everything wave 36's children produced is an artifact**, against
29.8% over the original 188 and 50.8% over wave 34's eight. That is the deepest-artifact
population any wave has produced, and it is not only a depth effect: `ARCH.W896CL1792` at
**0.733197x** is 11 of 15 artifact against `ARCH.V32W896`'s 7 of 15 at 0.731199x — the
same ratio, a different composition, four more artifact lanes. **Two of its lanes return
zero characters.** Nine children above 0.7x are more artifact-heavy still (13 to 15 of
15), from the width, blend and head-dim families, but none of them has more than TWO
content misses, so almost nothing is known about them in either direction; `ARCH.W896CL1792` has three
readable-and-wrong lanes and one signal lane, so something is.

The composed width × MLP body is the reason the canonical standard earns its keep here in
the *other* direction. Its three composed children read 1, 0 and 0 canonically against 0,
0 and 0 sealed, so the canonical lens rescues almost nothing — and the four readable
lanes it does leave on `ARCH.V16W896CL1792` include a seasons lane asserting the exact
misconception the prompt asks it to correct. A zero here is a zero of intelligence and not
of legibility, and that distinction is only available because the artifact column is
counted separately.

### Wave 37's own six, counted separately

| | lanes | share of the 90 |
|---|---:|---:|
| intelligence signal | **17** | 18.9% |
| artifact-miss | **43** | 47.8% |
| content-miss | **30** | 33.3% |

**Wave 37's population is the LEAST artifact-heavy deep population any wave has
produced** — 47.8% against wave 36's 67.8% and wave 34's 50.8% — and that is a fact about
the excision, not about the reading. Five of its six children are on the `--excise 9`
body, which is the first non-final excision this campaign has sealed since wave 28, and
its children fail by *saying wrong things* rather than by looping: thirty content misses
across six children, the highest content share of any wave in the deep band.

That cuts against the child on both lenses and it is worth stating plainly. A child that
loops is a child nothing is known about; a child that writes *The workshop is a complete
set of 17 kits, and the remaining kits are placed into the box* is a child that has been
measured and found wrong. Wave 37's children are more measured and no better.

**The one place the canonical lens moves a frontier here it moves it on a two-token
margin, and the margin is on the artifact/content boundary itself.** `ARCH.CL896X09`'s
uncertainty lane names two of the four withheld quantities and stops; the canonical
standard reads two correct specific elements as signal and the sealed standard reads an
incomplete list with no request as a miss. That single lane is the difference between
block 9 costing three and block 9 costing two, and both readings are in the child's own
seal.

**One rule had to be sharpened and it is in `wave37/canonical-reread-wave37.json`.**
Wave 35 established that a child naming backups, DNS, TLS, verification and rollback and
then looping is canonical signal. `ARCH.V16CL896G10` names all five **only inside the
echoed instruction** — *Give me a dependency on a plan that covers backups, DNS, TRLS,
verification, and rollback* — which is the prompt's words, and the standard's own artifact
clause calls that an indiscriminate prompt copy. It is read as an artifact, not signal.
`ARCH.V14CL896G10` names four as its own numbered headings and is read as signal. The
distinction is whether the items are the child's structure or a copy of the question, and
counting a prompt copy as planning would make every child that can echo look like a child
that can plan.

### 3a. The artifact clause, priced

Cyril's standard strikes 846 lanes on the grounds that loops and cap-rides are "fixable
with token penalties". Wave 34 re-decoded the five deepest canonical-signal children
through the SAME WEIGHTS with a repetition penalty of 1.3 and nothing else changed. The
diagnostic's own identity control at penalty 1.0 reproduced the sealed dossier on 16/16
lanes byte for byte, so the loop is the sealed loop plus one line.

| | lanes | |
|---|---:|---|
| artifact lanes under the sealed decode | 37 | |
| → recovered a correct answer under penalty | **0** | a keyed screen flagged 7; all 7 were read at full length and all 7 are false positives |
| → became readable and empty | **37** | |
| → remained artifact | 0 | the penalty eliminates the artifact class completely |
| intelligence-signal lanes under the sealed decode | 13 | |
| → still holding under penalty | **0** | |
| → **lost** under penalty | **13** | not one of the twelve sealed Partials survives |

Every key element is gone: Supplier B, 4:15 p.m., the four withheld quantities, Friday,
the Monday installation, the tracking request. **The penalty buys readability and pays for
it in content.** It converts loops into fluent nothing.

This does not refute the canonical standard — a lane that produces no readable text is
still no evidence about intelligence, which is the standard's actual claim. What it
refutes is the *remedy* attached to it. The 846 artifact lanes are not answers waiting
behind a decoding bug; on this population, at this penalty, there is nothing behind the
loop. Whatever fixes them, a repetition penalty at inference is not it.

### The two failure modes are different populations

**Artifact-dead** — zero signal and at least twelve artifact lanes. These children
produced almost no readable text. Nothing here is evidence that folding destroyed
intelligence; it is evidence that the decode collapsed.

| child | net ratio | artifact | content |
|---|---:|---:|---:|
| `ARCH.W512` | 0.463481 | 15 | 0 |
| `ARCH.A552` | 0.552023 | 15 | 0 |
| `ARCH.R5` | 0.579946 | 15 | 0 |
| `ARCH.W14FF320` | 0.580580 | 15 | 0 |
| `ARCH.W640` | 0.590763 | 15 | 0 |
| `ARCH.R4` | 0.603922 | 15 | 0 |
| `ARCH.CT640` | 0.640497 | 15 | 0 |
| `ARCH.R3` | 0.641584 | 15 | 0 |
| `ARCH.W704` | 0.656116 | 15 | 0 |
| `ARCH.W704AW25` | 0.656116 | 15 | 0 |
| `ARCH.R2` | 0.658704 | 15 | 0 |
| `ARCH.W720` | 0.672632 | 14 | 1 |
| `ARCH.W736LS` | 0.689220 | 14 | 1 |
| `ARCH.R1` | 0.689528 | 12 | 3 |
| `ARCH.W768AW100` | 0.722610 | 15 | 0 |
| `ARCH.W768AW25` | 0.722610 | 14 | 1 |
| `ARCH.MM768` | 0.722610 | 14 | 1 |
| `ARCH.R0` | 0.724929 | 12 | 3 |
| `ARCH.B1536` | 0.769439 | 13 | 2 |
| `ARCH.Q01050913` | 0.789916 | 12 | 3 |
| `ARCH.BM50N` | 0.856199 | 15 | 0 |
| `ARCH.HD16` | 0.938368 | 13 | 2 |

**Content-dead** — zero signal and mostly readable output. These are the only children in
the campaign that are honest evidence of destroyed intelligence, and there are three.

| child | net ratio | artifact | content | character |
|---|---:|---:|---:|---|
| `ARCH.V8CL896` | 0.432784 | 7 | 8 | the floor. Copies its prompts back verbatim and answers none of them: seven artifact lanes, eight readable-and-empty. The copy path outlived the answer path. |
| `ARCH.KKTD` | 0.483001 | 3 | 12 | the honest content death. Three artifact lanes and TWELVE readable, fluent, on-register lanes that say nothing -- 'the team has to contact the following email address: https://www.com'. |
| `ARCH.X12` | 0.952057 | 6 | 9 | 0.9521x and sealed ZERO -- and it is a CONTENT death, not an artifact one: nine readable lanes of confident nonsense plus six lanes of multilingual token salad. |

### The artifact population, broken out

| artifact kind | lanes |
|---|---:|
| loop or cap-ride with under ~80 characters of unique text | 742 |
| empty string after content, or empty outright | 85 |
| prompt echo covering the whole response | 16 |
| multilingual / sub-word token salad | 3 |
| **total** | **846** |

Cyril's instruction names exactly these: looping, repetition, prompt-echo, cap-rides,
missing-EOS and empty-after-content. All 846 are struck from the intelligence question.

## 4. The full matrix, deepest first

`intel` is the column. `sealed` is the frozen fifteen as recorded. `Δ` is the disagreement.

| child | wave | net ratio | stored | sealed | **intel** | artifact | content | Δ | character |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ARCH.V12CL896X13` | 35 | **0.420779** | 96,649,984 | 1 | **2/15** | 11 | 2 | +1 | wave 35's floor child: a vendor email naming Friday and the tracking number, below the V8CL896 death. |
| `ARCH.V12CL896X13G03` | 36 | **0.420779** | 96,649,984 | 1 | **1/15** | 11 | 3 | 0 | the gain arm at the floor. It moves a lane SIDEWAYS -- repairs summarisation, loses email to a prompt echo -- for a net of zero sealed and minus one canonical. |
| `ARCH.V8CL896` | 24 | **0.432784** | 99,407,616 | 0 | **0/15** | 7 | 8 | 0 | the floor. Copies its prompts back verbatim and answers none of them: seven artifact lanes, eight readable-and-empty. The copy path outlived the answer path. |
| `ARCH.V9CL896` | 35 | **0.437349** | 100,456,192 | 2 | **2/15** | 12 | 1 | 0 | wave 35's halving of the 0/15 bracket from the surviving side; all four withheld quantities named in garbled syntax. |
| `ARCH.V9CL896G03` | 36 | **0.437349** | 100,456,192 | 3 | **4/15** | 9 | 2 | +1 | lambda 0.3 at K = 9,216: +1 on the vendor email, on the same bytes the K = 10,240 arm bought it with. |
| `ARCH.V9CL896G10` | 36 | **0.437349** | 100,456,192 | 4 | **5/15** | 8 | 2 | +1 | THE DEEPEST FOUR SEALED AND FIVE READ. lambda 1.0 beats lambda 0.3 for the first time, on retrieval: 4:15 p.m. in the right unit. |
| `ARCH.V10CL896` | 34 | **0.441915** | 101,504,768 | 2 | **2/15** | 11 | 2 | 0 | the deepest child ever built and it holds two. ELEVEN artifact lanes against TWO content ones: at 0.4419x the decode has collapsed and the content has not. |
| `ARCH.V12CL896` | 34 | **0.451045** | 103,601,920 | 2 | **2/15** | 6 | 7 | 0 | the lambda ladder's zero-solve control; the one deep child whose content misses outnumber its artifacts. |
| `ARCH.V12CL896G01` | 34 | **0.451045** | 103,601,920 | 2 | **2/15** | 9 | 4 | 0 | lambda 0.1: the largest update, the joint-worst review, and '4:15 MINUTES' -- the right number in the wrong unit. |
| `ARCH.V12CL896G03` | 34 | **0.451045** | 103,601,920 | 4 | **4/15** | 8 | 3 | 0 | THE DEEPEST FOUR IN THE CAMPAIGN under both standards: summary, email, the four withheld quantities, and 4:15 p.m. off a note its own control loses to a loop. |
| `ARCH.V12CL896G10` | 34 | **0.451045** | 103,601,920 | 2 | **2/15** | 8 | 5 | 0 | lambda 1.0; four of sixteen lanes byte-identical to its source. |
| `ARCH.V19X13CL896` | 28 | **0.452734** | 103,990,016 | 0 | **1/15** | 10 | 4 | +1 | one lane of intelligence at 0.4527x: it compresses the supplier update in its own words and lands the unresolved risk -- a one-week factory-expansion slip. |
| `ARCH.W512` | 19 | **0.463481** | 106,458,368 | 0 | **0/15** | 15 | 0 | 0 | fifteen for fifteen artifact. Nothing readable came out at all, so nothing here is evidence about intelligence in either direction. |
| `ARCH.V12W896CL1792` | 36 | **0.525484** | 120,700,032 | 0 | **0/15** | 11 | 4 | 0 | K = 12,288 on the composed width x MLP body. Zero on both standards; its summarisation lane invents a seven-day deadline that is nowhere in the update. |
| `ARCH.V16W896CL1792` | 36 | **0.541462** | 124,370,048 | 0 | **0/15** | 11 | 4 | 0 | K = 16,384 on the same body, built as the control for the rung above it. Its seasons lane asserts the exact misconception the prompt asks it to correct. |
| `ARCH.W896CL1792` | 36 | **0.733197** | 168,410,240 | 0 | **1/15** | 11 | 3 | +1 | THE LAST UNBUILT TWO-AXIS BODY, AND AN HONEST CONTENT DEATH 0.30 OF RATIO ABOVE EVERY OTHER ONE. width 896 x mlp 1792; ARCH.W896 alone reads 4, ARCH.CL1792 alone reads 7, together they read 0. |
| `ARCH.V16CL896` | 23 | **0.469305** | 107,796,224 | 1 | **2/15** | 5 | 8 | +1 | summarises and asks for the four things it was not told. Under half the parent and it still knows what it does not know. Rebuilt at wave 34, bit-identical. |
| `ARCH.V16CL896G03` | 34 | **0.469305** | 107,796,224 | 3 | **3/15** | 6 | 6 | 0 | the CL896 family's missing gain arm; three, on the same lanes its K=19,456 and K=20,480 siblings hold. |
| `ARCH.V18CL896` | 28 | **0.478436** | 109,893,376 | 2 | **2/15** | 6 | 7 | 0 | two summarisation-family lanes and nothing else; the other seven readable lanes are on-register and empty. |
| `ARCH.V19CL896` | 28 | **0.483001** | 110,941,952 | 2 | **2/15** | 5 | 8 | 0 | the same two lanes as V18CL896 at the same body -- the vocabulary axis buys ratio without buying a new failure. |
| `ARCH.V19CL896G03` | 28 | **0.483001** | 110,941,952 | 3 | **4/15** | 7 | 4 | +1 | the deepest child in the campaign to hold FOUR: summary, email, uncertainty, and it retrieves 4:15 from the note. 0.483x. |
| `ARCH.KKTD` | 29 | **0.483001** | 110,941,952 | 0 | **0/15** | 3 | 12 | 0 | the honest content death. Three artifact lanes and TWELVE readable, fluent, on-register lanes that say nothing -- 'the team has to contact the following email address: https://www.com'. |
| `ARCH.KKTS` | 29 | **0.483001** | 110,941,952 | 0 | **2/15** | 9 | 4 | +2 | sealed dead by degeneracy, and it still orders a migration (backups, then TLS, then verify) and names the four withheld quantities. |
| `ARCH.V20CL896` | 27 | **0.487566** | 111,990,528 | 2 | **2/15** | 5 | 8 | 0 | two lanes at 0.4876x; the context-retrieval lane is the question quoted back at itself. |
| `ARCH.V20CL896G03` | 28 | **0.487566** | 111,990,528 | 3 | **4/15** | 7 | 4 | +1 | four lanes at 0.4876x, the terminal-gain twin of V19CL896G03, and it retrieves the same 4:15. |
| `ARCH.V32CL896` | 27 | **0.542347** | 124,573,440 | 2 | **2/15** | 7 | 6 | 0 | the largest table on the smallest MLP: the supplier summary and the missing-quantities list survive; it AFFIRMS the seasons misconception. |
| `ARCH.V8CL1792X13` | 34 | **0.558303** | 128,238,336 | 1 | **2/15** | 6 | 7 | +1 | sealed one, reads two: 4:15 p.m. AND Saturdays, and the Monday installation in an email's right slot with a tracking request. |
| `ARCH.V12CL1792X13` | 34 | **0.576563** | 132,432,640 | 2 | **2/15** | 7 | 6 | 0 | THE C2 THESIS FALSIFIED: predicted FIVE canonical off its K=19,456 sibling's six, reads TWO. The canonical lens does not carry the triple down the vocabulary axis. |
| `ARCH.A552` | 15 | **0.552023** | 126,795,968 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V19CL1280` | 30 | **0.554901** | 127,457,024 | 1 | **1/15** | 7 | 7 | 0 | one lane. The uncertainty lane is the campaign's cleanest anti-pattern -- it invents a cause ('suppression of the price') for a graph it was told nothing about. |
| `ARCH.V19CL1408` | 31 | **0.578868** | 132,962,048 | 1 | **1/15** | 6 | 8 | 0 | one lane; five key hits that are all fake -- it names the missing quantities only after inventing a reason. |
| `ARCH.R5` | 16 | **0.579946** | 133,209,600 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.W14FF320` | 15 | **0.580580** | 133,355,264 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V4CL1792` | 32 | **0.582292** | 133,748,480 | 1 | **2/15** | 2 | 11 | +1 | the vendor email, plus a summarisation lane the seal missed: 'The team selects Supplier B ... may be one week, so they request scheduled delivery'. |
| `ARCH.W640` | 19 | **0.590763** | 135,694,208 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V6CL1792` | 32 | **0.591422** | 135,845,632 | 3 | **3/15** | 1 | 11 | 0 | three household actions, the checkout time, the supplier summary. The safety lane tells you to mish your way through the spill. |
| `ARCH.V8CL1792` | 31 | **0.600552** | 137,942,784 | 5 | **7/15** | 2 | 6 | +2 | seven at 0.6006x. Checkout time, summary, email, missing quantities, a natural greeting, a true statement about RAM bandwidth, and a usable triage action. |
| `ARCH.V8CL1792M` | 32 | **0.600552** | 137,942,784 | 3 | **4/15** | 2 | 9 | +1 | the reader-metric twin: three sealed partials plus the 4:15 retrieval the seal called a miss. It offers the tracking number instead of asking for it. |
| `ARCH.V19CL1536` | 31 | **0.602835** | 138,467,072 | 2 | **2/15** | 7 | 6 | 0 | two lanes; the multi-turn lane returns the user's own last turn verbatim. |
| `ARCH.R4` | 16 | **0.603922** | 138,716,800 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V9CL1792` | 31 | **0.605117** | 138,991,360 | 6 | **7/15** | 1 | 7 | +1 | seven at 0.6051x -- the second-best email in its wave, the garden day, and a usable list-making action. |
| `ARCH.V9CL1792M` | 32 | **0.605117** | 138,991,360 | 2 | **4/15** | 1 | 10 | +2 | four: summary, email, and TWO lanes the seal missed -- 7:30 a.m. and 4:15 p.m. off the note, and a missing-data request that invents nothing. |
| `ARCH.V19CL1792X13` | 31 | **0.608519** | 139,772,672 | 2 | **6/15** | 5 | 4 | +4 | sealed two, reads six. 4:15 off the note, O(n) stated correctly, a real vendor email naming the Monday install, and three usable prioritisation actions. |
| `ARCH.V12CL1792` | 31 | **0.618813** | 142,137,088 | 7 | **7/15** | 0 | 8 | 0 | seven with ZERO artifact lanes -- everything it produced was readable. The honest deep frontier of the sealed campaign. |
| `ARCH.V12CL1792M` | 32 | **0.618813** | 142,137,088 | 4 | **4/15** | 1 | 10 | 0 | five: four sealed plus a concrete household action. Its arithmetic lane says 85 and boxes it. |
| `ARCH.GV12L01` | 33 | **0.618813** | 142,137,088 | 6 | **6/15** | 1 | 8 | 0 | six. Terminal-gain lambda 0.1 loses a lane to a CAP HIT after eighteen repetitions of 'use a list of tasks for today' -- an artifact, not a content loss. |
| `ARCH.GV12L03` | 33 | **0.618813** | 142,137,088 | 7 | **7/15** | 1 | 7 | 0 | seven, flat at its source. The lambda ladder buys nothing and costs nothing. |
| `ARCH.GV12L10` | 33 | **0.618813** | 142,137,088 | 7 | **7/15** | 1 | 7 | 0 | seven, flat at its source at the most aggressive rung. |
| `ARCH.V16W896X13` | 23 | **0.623329** | 143,174,400 | 1 | **2/15** | 10 | 3 | +1 | holds summary, planning. |
| `ARCH.V19CL1664` | 31 | **0.626802** | 143,972,096 | 3 | **3/15** | 3 | 9 | 0 | holds summary, email, uncertainty. |
| `ARCH.V16CL1792` | 31 | **0.637073** | 146,331,392 | 8 | **8/15** | 1 | 6 | 0 | eight at 0.6371x with one artifact lane. The deepest EIGHT in the campaign under either standard. |
| `ARCH.V16BMV3` | 32 | **0.637073** | 146,331,392 | 3 | **6/15** | 1 | 8 | +3 | sealed three, reads six: 4:15 off the note, a real vendor email, and a clean missing-data request. |
| `ARCH.V16CL1792M` | 32 | **0.637073** | 146,331,392 | 3 | **6/15** | 0 | 9 | +3 | sealed three, reads six: it gets 17 x 6 = 102 right before losing the chain, and asks for all four withheld quantities. |
| `ARCH.GV16L01` | 33 | **0.637073** | 146,331,392 | 8 | **8/15** | 0 | 7 | 0 | eight, tying V16CL1792 lane for lane, at the largest update scale of its six rungs. |
| `ARCH.GV16L03` | 33 | **0.637073** | 146,331,392 | 7 | **7/15** | 0 | 8 | 0 | seven; the greeting lane is a menu of questions rather than a reply. |
| `ARCH.GV16L10` | 33 | **0.637073** | 146,331,392 | 7 | **7/15** | 1 | 7 | 0 | seven, one below the lambda-0.1 rung, and the lane it loses is the greeting. |
| `ARCH.V20X13W896` | 26 | **0.639307** | 146,844,416 | 2 | **3/15** | 11 | 1 | +1 | holds retrieval, summary, planning. |
| `ARCH.CL640` | 15 | **0.640497** | 147,117,824 | 0 | **1/15** | 7 | 7 | +1 | sealed ZERO and it is not zero: it pulls 4:15 p.m. and 7:30 a.m. off the note. Seven of its other lanes are pure loop. |
| `ARCH.CT640` | 15 | **0.640497** | 147,117,824 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.S640` | 15 | **0.640497** | 147,117,824 | 0 | **0/15** | 10 | 5 | 0 | content-dead: 5 readable lanes, all wrong or vacuous; 10 artifact. |
| `ARCH.W14FF640` | 15 | **0.640497** | 147,117,824 | 0 | **0/15** | 11 | 4 | 0 | content-dead: 4 readable lanes, all wrong or vacuous; 11 artifact. |
| `ARCH.R3` | 16 | **0.641584** | 147,367,552 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V18CL1792` | 31 | **0.646204** | 148,428,544 | 7 | **7/15** | 1 | 7 | 0 | holds greeting, 3-actions, retrieval, summary, debug, email, uncertainty. |
| `ARCH.V19CL1792` | 30 | **0.650769** | 149,477,120 | 7 | **7/15** | 0 | 8 | 0 | holds greeting, 3-actions, retrieval, summary, debug, email, uncertainty. |
| `ARCH.V19CL1792M` | 30 | **0.650769** | 149,477,120 | 6 | **6/15** | 0 | 9 | 0 | holds retrieval, summary, debug, email, planning, uncertainty. |
| `ARCH.V19CL1792ME` | 31 | **0.650769** | 149,477,120 | 3 | **5/15** | 1 | 9 | +2 | holds summary, debug, email, planning, uncertainty. |
| `ARCH.V19CL1792MV` | 31 | **0.650769** | 149,477,120 | 5 | **5/15** | 2 | 8 | 0 | holds greeting, retrieval, summary, email, uncertainty. |
| `ARCH.W704` | 20 | **0.656116** | 150,705,344 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.W704AW25` | 20 | **0.656116** | 150,705,344 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.R2` | 16 | **0.658704** | 151,299,712 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V18X13X01KV6JQ` | 27 | **0.664419** | 152,612,608 | 0 | **0/15** | 9 | 6 | 0 | content-dead: 6 readable lanes, all wrong or vacuous; 9 artifact. |
| `ARCH.V16W896` | 23 | **0.667288** | 153,271,424 | 2 | **3/15** | 10 | 2 | +1 | holds 3-actions, summary, uncertainty. |
| `ARCH.W720` | 22 | **0.672632** | 154,499,088 | 0 | **0/15** | 14 | 1 | 0 | artifact-dead: 14 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V24CL1792` | 31 | **0.673594** | 154,720,000 | 7 | **7/15** | 0 | 8 | 0 | holds greeting, 3-actions, retrieval, summary, debug, email, uncertainty. |
| `ARCH.V18W896` | 27 | **0.675277** | 155,106,432 | 2 | **3/15** | 8 | 4 | +1 | holds 3-actions, summary, uncertainty. |
| `ARCH.V26CL1792` | 31 | **0.682724** | 156,817,152 | 7 | **7/15** | 0 | 8 | 0 | holds greeting, 3-actions, retrieval, summary, debug, email, uncertainty. |
| `ARCH.V20W896` | 26 | **0.683266** | 156,941,440 | 2 | **3/15** | 8 | 4 | +1 | holds 3-actions, summary, uncertainty. |
| `ARCH.V18X13X01` | 27 | **0.684962** | 157,331,200 | 3 | **4/15** | 3 | 8 | +1 | holds 3-actions, summary, debug, uncertainty. |
| `ARCH.V18X13X01` | 28 | **0.684962** | 157,331,200 | 3 | **4/15** | 3 | 8 | +1 | holds 3-actions, summary, debug, uncertainty. |
| `ARCH.V18X13X01G01` | 28 | **0.684962** | 157,331,200 | 3 | **4/15** | 3 | 8 | +1 | holds 3-actions, summary, debug, planning. |
| `ARCH.V18X13X01G03` | 28 | **0.684962** | 157,331,200 | 4 | **5/15** | 3 | 7 | +1 | holds 3-actions, arithmetic, summary, debug, planning. |
| `ARCH.V18X13X01G10` | 28 | **0.684962** | 157,331,200 | 4 | **4/15** | 2 | 9 | 0 | holds 3-actions, summary, debug, planning. |
| `ARCH.CL896` | 15 | **0.688431** | 158,127,872 | 4 | **6/15** | 5 | 4 | +2 | holds retrieval, summary, code, email, planning, uncertainty. |
| `ARCH.W736` | 21 | **0.689220** | 158,309,216 | 1 | **1/15** | 12 | 2 | 0 | holds multi-turn. |
| `ARCH.W736LS` | 21 | **0.689220** | 158,309,216 | 0 | **0/15** | 14 | 1 | 0 | artifact-dead: 14 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.R1` | 16 | **0.689528** | 158,379,776 | 0 | **0/15** | 12 | 3 | 0 | artifact-dead: 12 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V19X13X01` | 28 | **0.689528** | 158,379,776 | 3 | **4/15** | 3 | 8 | +1 | holds 3-actions, summary, debug, uncertainty. |
| `ARCH.V19X13X01G03` | 28 | **0.689528** | 158,379,776 | 4 | **4/15** | 3 | 8 | 0 | holds 3-actions, summary, debug, planning. |
| `ARCH.V19X13X09` | 28 | **0.689528** | 158,379,776 | 5 | **8/15** | 1 | 6 | +3 | eight at 0.6895x, and the seasons lane REJECTS the closer-to-the-Sun account outright -- the correction the lane is built to test. |
| `ARCH.V8X13` | 24 | **0.691832** | 158,909,184 | 6 | **7/15** | 1 | 7 | +1 | seven; the coding lane sketches the seen-set plus order-list algorithm correctly. |
| `ARCH.V8X13G` | 24 | **0.691832** | 158,909,184 | 4 | **7/15** | 1 | 7 | +3 | sealed four, reads seven: Saturday and 4:15 together, Supplier B with its one-week risk, and a complete correct dedup loop. |
| `ARCH.V8X13GL01` | 24 | **0.691832** | 158,909,184 | 7 | **9/15** | 0 | 6 | +2 | nine at 0.6918x: 17 x 6 = 102 then 102 - 8, and the dedup algorithm. Its multi-turn lane names a peanut-free option and then names no food, so it does not count. |
| `ARCH.V8X13GL1` | 24 | **0.691832** | 158,909,184 | 6 | **7/15** | 1 | 7 | +1 | eight; it derives Mira -> Theo -> Jun explicitly. |
| `ARCH.V8X13GL03` | 25 | **0.691832** | 158,909,184 | 7 | **9/15** | 0 | 6 | +2 | nine; Saturday plus the closed north gate, and the dedup loop. |
| `ARCH.V20X13X07` | 26 | **0.694093** | 159,428,352 | 5 | **6/15** | 2 | 7 | +1 | holds greeting, 3-actions, summary, code, email, planning. |
| `ARCH.V20X13X01` | 27 | **0.694093** | 159,428,352 | 3 | **4/15** | 3 | 8 | +1 | holds 3-actions, summary, debug, uncertainty. |
| `ARCH.V20X13X01G03` | 27 | **0.694093** | 159,428,352 | 1 | **3/15** | 3 | 9 | +2 | holds 3-actions, summary, planning. |
| `ARCH.V9X13` | 27 | **0.696398** | 159,957,760 | 6 | **8/15** | 0 | 7 | +2 | holds greeting, 3-actions, retrieval, summary, code, debug, email, uncertainty. |
| `ARCH.V12X13` | 25 | **0.710093** | 163,103,488 | 7 | **10/15** | 0 | 5 | +3 | TEN at 0.7101x, the deepest ten in the campaign: the RAM/storage contrast, the dedup loop, and a vendor email naming the Monday install. |
| `ARCH.V18X13KV6JQ` | 27 | **0.716941** | 164,676,352 | 2 | **2/15** | 7 | 6 | 0 | holds greeting, summary. |
| `ARCH.W768` | 19 | **0.722610** | 165,978,624 | 3 | **3/15** | 11 | 1 | 0 | holds summary, uncertainty, multi-turn. |
| `ARCH.W768AW100` | 20 | **0.722610** | 165,978,624 | 0 | **0/15** | 15 | 0 | 0 | artifact-dead: 15 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.W768AW25` | 20 | **0.722610** | 165,978,624 | 0 | **0/15** | 14 | 1 | 0 | artifact-dead: 14 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.W768EV` | 20 | **0.722610** | 165,978,624 | 1 | **1/15** | 14 | 0 | 0 | holds planning. |
| `ARCH.W768LS` | 20 | **0.722610** | 165,978,624 | 3 | **3/15** | 10 | 2 | 0 | holds retrieval, summary, multi-turn. |
| `ARCH.W768RG` | 20 | **0.722610** | 165,978,624 | 0 | **0/15** | 10 | 5 | 0 | content-dead: 5 readable lanes, all wrong or vacuous; 10 artifact. |
| `ARCH.W768ST` | 20 | **0.722610** | 165,978,624 | 2 | **2/15** | 12 | 1 | 0 | holds uncertainty, multi-turn. |
| `ARCH.MM768` | 21 | **0.722610** | 165,978,624 | 0 | **0/15** | 14 | 1 | 0 | artifact-dead: 14 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.BX13` | 16 | **0.724929** | 166,511,360 | 1 | **4/15** | 1 | 10 | +3 | holds summary, email, planning, uncertainty. |
| `ARCH.R0` | 16 | **0.724929** | 166,511,360 | 0 | **0/15** | 12 | 3 | 0 | artifact-dead: 12 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V20X13KV6JQ` | 27 | **0.726071** | 166,773,504 | 2 | **3/15** | 5 | 7 | +1 | holds greeting, retrieval, summary. |
| `ARCH.V20X13KV6JQG03` | 27 | **0.726071** | 166,773,504 | 2 | **3/15** | 7 | 5 | +1 | holds greeting, retrieval, summary. |
| `ARCH.V16X13` | 23 | **0.728353** | 167,297,792 | 5 | **7/15** | 0 | 8 | +2 | holds greeting, 3-actions, retrieval, summary, RAM/ROM, code, uncertainty. |
| `ARCH.V16X13G` | 24 | **0.728353** | 167,297,792 | 5 | **10/15** | 0 | 5 | +5 | the largest reversal in the campaign: sealed five, reads TEN. 102 AND 94 both correct, the dedup loop, a real email, and a lighthouse scene with the line 'The foghorn is not just a fireshore. It is a memory.' |
| `ARCH.V16X13GL01` | 24 | **0.728353** | 167,297,792 | 6 | **8/15** | 2 | 5 | +2 | holds greeting, 3-actions, retrieval, summary, RAM/ROM, code, email, uncertainty. |
| `ARCH.V16X13GL1` | 24 | **0.728353** | 167,297,792 | 6 | **9/15** | 2 | 4 | +3 | holds greeting, 3-actions, ordering, retrieval, summary, RAM/ROM, code, planning, uncertainty. |
| `ARCH.V16X13GL03` | 25 | **0.728353** | 167,297,792 | 7 | **10/15** | 2 | 3 | +3 | ten at 0.7284x; 'Our installation is scheduled for Monday' lands in its right slot. |
| `ARCH.V32W896` | 25 | **0.731199** | 167,951,488 | 3 | **3/15** | 7 | 5 | 0 | holds summary, email, uncertainty. |
| `ARCH.V22X13KV6JQ` | 27 | **0.735201** | 168,870,656 | 2 | **2/15** | 5 | 8 | 0 | holds greeting, summary. |
| `ARCH.V18X07` | 27 | **0.737484** | 169,394,944 | 9 | **10/15** | 0 | 5 | +1 | holds greeting, 3-actions, seasons, retrieval, summary, RAM/ROM, email, planning, uncertainty, creative. |
| `ARCH.V18X07G03` | 27 | **0.737484** | 169,394,944 | 8 | **9/15** | 0 | 6 | +1 | holds greeting, 3-actions, retrieval, summary, RAM/ROM, email, planning, uncertainty, creative. |
| `ARCH.V18X09` | 27 | **0.737484** | 169,394,944 | 10 | **12/15** | 0 | 3 | +2 | TWELVE at 0.7375x: 102 correct and set-based dedup with O(n) named. Its four 'nut-free' options are nut-free nuts, which is not an option. |
| `ARCH.V18X09G03` | 27 | **0.737484** | 169,394,944 | 9 | **12/15** | 0 | 3 | +3 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, email, planning, uncertainty, creative, multi-turn. |
| `ARCH.V18X13` | 27 | **0.737484** | 169,394,944 | 10 | **12/15** | 0 | 3 | +2 | holds greeting, 3-actions, seasons, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty. |
| `ARCH.V18X13G03` | 27 | **0.737484** | 169,394,944 | 10 | **11/15** | 1 | 3 | +1 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty. |
| `ARCH.V19X09` | 28 | **0.742049** | 170,443,520 | 10 | **13/15** | 0 | 2 | +3 | thirteen at 0.7421x with ZERO artifact lanes -- everything it produced was readable. 102, set-based dedup with O(n), a backup-first migration with a test step. |
| `ARCH.V19X13` | 28 | **0.742049** | 170,443,520 | 10 | **13/15** | 0 | 2 | +3 | holds greeting, 3-actions, seasons, arithmetic, ordering, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty. |
| `ARCH.V19X13G01` | 28 | **0.742049** | 170,443,520 | 11 | **12/15** | 1 | 2 | +1 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.V19X13G03` | 28 | **0.742049** | 170,443,520 | 11 | **12/15** | 1 | 2 | +1 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.V19X13G10` | 28 | **0.742049** | 170,443,520 | 10 | **12/15** | 2 | 1 | +2 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.B25` | 29 | **0.742049** | 170,443,520 | 10 | **12/15** | 0 | 3 | +2 | holds greeting, 3-actions, arithmetic, ordering, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty. |
| `ARCH.B50` | 29 | **0.742049** | 170,443,520 | 10 | **12/15** | 2 | 1 | +2 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.B75` | 29 | **0.742049** | 170,443,520 | 11 | **12/15** | 1 | 2 | +1 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.BP64` | 29 | **0.742049** | 170,443,520 | 10 | **13/15** | 0 | 2 | +3 | thirteen at 0.7421x: three valid derived orderings, the dedup loop, and a scene with the line 'We're not here for the light.' |
| `ARCH.MASK03` | 29 | **0.742049** | 170,443,520 | 9 | **11/15** | 0 | 4 | +2 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty. |
| `ARCH.SR24` | 30 | **0.742049** | 170,443,520 | 10 | **12/15** | 1 | 2 | +2 | holds greeting, 3-actions, arithmetic, ordering, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty. |
| `ARCH.SR64` | 30 | **0.742049** | 170,443,520 | 10 | **11/15** | 1 | 3 | +1 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty. |
| `ARCH.V8` | 24 | **0.744354** | 170,972,928 | 4 | **7/15** | 0 | 8 | +3 | holds greeting, ordering, retrieval, summary, email, uncertainty, multi-turn. |
| `ARCH.V20X13` | 26 | **0.746614** | 171,492,096 | 10 | **13/15** | 1 | 1 | +3 | holds greeting, 3-actions, seasons, arithmetic, ordering, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty. |
| `ARCH.V20X13G03` | 26 | **0.746614** | 171,492,096 | 11 | **12/15** | 1 | 2 | +1 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.V20X07` | 27 | **0.746614** | 171,492,096 | 9 | **10/15** | 0 | 5 | +1 | holds greeting, 3-actions, seasons, retrieval, summary, RAM/ROM, email, planning, uncertainty, creative. |
| `ARCH.V20X09` | 27 | **0.746614** | 171,492,096 | 9 | **12/15** | 0 | 3 | +3 | holds greeting, 3-actions, seasons, arithmetic, retrieval, summary, RAM/ROM, code, email, planning, uncertainty, creative. |
| `ARCH.V20X09G03` | 27 | **0.746614** | 171,492,096 | 8 | **11/15** | 0 | 4 | +3 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, email, planning, uncertainty, creative. |
| `ARCH.V9` | 26 | **0.748919** | 172,021,504 | 9 | **11/15** | 0 | 4 | +2 | holds greeting, 3-actions, ordering, retrieval, summary, RAM/ROM, code, email, planning, uncertainty, multi-turn. |
| `ARCH.V10` | 25 | **0.753484** | 173,070,080 | 10 | **11/15** | 0 | 4 | +1 | holds greeting, 3-actions, seasons, ordering, retrieval, summary, code, email, planning, uncertainty, multi-turn. |
| `ARCH.W14FF1280` | 15 | **0.760331** | 174,642,944 | 5 | **5/15** | 4 | 6 | 0 | holds greeting, retrieval, summary, email, uncertainty. |
| `ARCH.CL1280` | 30 | **0.760331** | 174,642,944 | 4 | **5/15** | 4 | 6 | +1 | holds summary, debug, email, planning, uncertainty. |
| `ARCH.V12` | 25 | **0.762614** | 175,167,232 | 11 | **14/15** | 0 | 1 | +3 | FOURTEEN at 0.762614x -- the deepest fourteen in the campaign under the canonical standard, and the sealed column called it eleven. 17 x 6 = 102, the dedup loop, and a scene whose growl makes the keeper's breath hitch. |
| `ARCH.B1536` | 16 | **0.769439** | 176,734,976 | 0 | **0/15** | 13 | 2 | 0 | artifact-dead: 13 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.W1536X13` | 16 | **0.769439** | 176,734,976 | 4 | **4/15** | 6 | 5 | 0 | holds greeting, RAM/ROM, email, uncertainty. |
| `ARCH.V14` | 25 | **0.771744** | 177,264,384 | 9 | **13/15** | 0 | 2 | +4 | holds greeting, 3-actions, seasons, arithmetic, retrieval, summary, RAM/ROM, code, email, planning, uncertainty, creative, multi-turn. |
| `ARCH.V16` | 23 | **0.780874** | 179,361,536 | 9 | **13/15** | 0 | 2 | +4 | holds greeting, 3-actions, seasons, arithmetic, retrieval, summary, RAM/ROM, code, email, planning, uncertainty, creative, multi-turn. |
| `ARCH.CL1408` | 31 | **0.784298** | 180,147,968 | 3 | **5/15** | 6 | 4 | +2 | holds retrieval, summary, email, planning, uncertainty. |
| `ARCH.V17` | 27 | **0.785440** | 180,410,112 | 11 | **14/15** | 0 | 1 | +3 | holds greeting, 3-actions, seasons, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative, multi-turn. |
| `ARCH.Q01050913` | 19 | **0.789916** | 181,438,208 | 0 | **0/15** | 12 | 3 | 0 | artifact-dead: 12 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.V18` | 26 | **0.790005** | 181,458,688 | 13 | **14/15** | 0 | 1 | +1 | fourteen at 0.7900x; the creative lane is a complete scene with dialogue -- Keeper Mara and the cracked spectroscope. |
| `ARCH.V19` | 27 | **0.794570** | 182,507,264 | 14 | **14/15** | 0 | 1 | 0 | fourteen, the sealed top of the campaign; its one miss is the ordering lane restated rather than solved. |
| `ARCH.V19MXO3` | 29 | **0.794570** | 182,507,264 | 3 | **5/15** | 2 | 8 | +2 | holds 3-actions, arithmetic, email, planning, uncertainty. |
| `ARCH.V19MXP2` | 29 | **0.794570** | 182,507,264 | 2 | **3/15** | 4 | 8 | +1 | holds 3-actions, RAM/ROM, email. |
| `ARCH.V20` | 25 | **0.799135** | 183,555,840 | 13 | **14/15** | 0 | 1 | +1 | holds greeting, 3-actions, seasons, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative, multi-turn. |
| `ARCH.V32X13` | 25 | **0.801395** | 184,075,008 | 10 | **13/15** | 1 | 1 | +3 | holds greeting, 3-actions, arithmetic, ordering, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.V32X13G` | 25 | **0.801395** | 184,075,008 | 10 | **12/15** | 0 | 3 | +2 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.V32X13GL01` | 25 | **0.801395** | 184,075,008 | 9 | **11/15** | 1 | 3 | +2 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty. |
| `ARCH.V32X13GL1` | 25 | **0.801395** | 184,075,008 | 10 | **12/15** | 1 | 2 | +2 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.V32X13G03` | 26 | **0.801395** | 184,075,008 | 10 | **12/15** | 0 | 3 | +2 | holds greeting, 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.V22` | 26 | **0.808265** | 185,652,992 | 13 | **13/15** | 0 | 2 | 0 | holds greeting, 3-actions, seasons, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.CL1536` | 31 | **0.808265** | 185,652,992 | 4 | **4/15** | 9 | 2 | 0 | holds retrieval, summary, email, uncertainty. |
| `ARCH.CL1792X13` | 31 | **0.813949** | 186,958,592 | 4 | **4/15** | 5 | 6 | 0 | holds retrieval, summary, email, uncertainty. |
| `ARCH.W896X13` | 19 | **0.815064** | 187,214,592 | 1 | **1/15** | 12 | 2 | 0 | holds summary. |
| `ARCH.W896LSX13` | 21 | **0.815064** | 187,214,592 | 2 | **2/15** | 11 | 2 | 0 | holds summary, planning. |
| `ARCH.CL1664` | 31 | **0.832232** | 191,158,016 | 5 | **5/15** | 1 | 9 | 0 | holds 3-actions, retrieval, summary, email, uncertainty. |
| `ARCH.W896KV6JS` | 19 | **0.841047** | 193,182,848 | 2 | **2/15** | 12 | 1 | 0 | holds 3-actions, summary. |
| `ARCH.W896LSKV6JQ` | 22 | **0.841047** | 193,182,848 | 1 | **1/15** | 13 | 1 | 0 | holds summary. |
| `ARCH.D3` | 16 | **0.842437** | 193,501,952 | 0 | **1/15** | 14 | 0 | +1 | sealed ZERO at 0.8424x and it is not zero: 'The team's decision to B, a 11% cost ... the risk is a potential 1-week expansion' carries the decision, the price delta and the risk. |
| `ARCH.V32` | 24 | **0.853916** | 196,138,752 | 11 | **14/15** | 0 | 1 | +3 | holds greeting, 3-actions, seasons, arithmetic, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative, multi-turn. |
| `ARCH.CL1792` | 30 | **0.856199** | 196,663,040 | 7 | **7/15** | 0 | 8 | 0 | holds 3-actions, retrieval, summary, debug, email, uncertainty, creative. |
| `ARCH.CL1792LG` | 30 | **0.856199** | 196,663,040 | 7 | **7/15** | 0 | 8 | 0 | holds 3-actions, retrieval, summary, debug, email, uncertainty, creative. |
| `ARCH.CL1792M` | 30 | **0.856199** | 196,663,040 | 8 | **9/15** | 0 | 6 | +1 | holds greeting, 3-actions, retrieval, summary, debug, email, planning, uncertainty, creative. |
| `ARCH.CL1792ME` | 31 | **0.856199** | 196,663,040 | 2 | **3/15** | 2 | 10 | +1 | holds seasons, retrieval, summary. |
| `ARCH.CL1792MV` | 31 | **0.856199** | 196,663,040 | 7 | **7/15** | 1 | 7 | 0 | holds greeting, 3-actions, arithmetic, summary, email, planning, uncertainty. |
| `ARCH.BM25` | 32 | **0.856199** | 196,663,040 | 6 | **7/15** | 0 | 8 | +1 | holds 3-actions, retrieval, summary, debug, email, uncertainty, creative. |
| `ARCH.BM50` | 32 | **0.856199** | 196,663,040 | 4 | **4/15** | 0 | 11 | 0 | holds 3-actions, retrieval, summary, uncertainty. |
| `ARCH.BM50N` | 32 | **0.856199** | 196,663,040 | 0 | **0/15** | 15 | 0 | 0 | fifteen for fifteen artifact at 0.8562x -- a body that decodes nothing, which is a statement about the fold and not about intelligence. |
| `ARCH.BM75` | 32 | **0.856199** | 196,663,040 | 2 | **4/15** | 1 | 10 | +2 | holds arithmetic, summary, email, uncertainty. |
| `ARCH.BMS75` | 32 | **0.856199** | 196,663,040 | 5 | **6/15** | 1 | 8 | +1 | holds greeting, arithmetic, retrieval, summary, email, uncertainty. |
| `ARCH.BMV3` | 32 | **0.856199** | 196,663,040 | 7 | **7/15** | 0 | 8 | 0 | holds 3-actions, arithmetic, retrieval, summary, email, planning, uncertainty. |
| `ARCH.BV50` | 32 | **0.856199** | 196,663,040 | 4 | **5/15** | 0 | 10 | +1 | holds 3-actions, retrieval, summary, email, uncertainty. |
| `ARCH.CL1792S` | 32 | **0.856199** | 196,663,040 | 4 | **4/15** | 1 | 10 | 0 | holds 3-actions, retrieval, summary, uncertainty. |
| `ARCH.W896` | 19 | **0.859023** | 197,311,616 | 4 | **4/15** | 5 | 6 | 0 | holds 3-actions, summary, email, uncertainty. |
| `ARCH.W896LS` | 21 | **0.859023** | 197,311,616 | 3 | **3/15** | 8 | 4 | 0 | holds summary, planning, uncertainty. |
| `ARCH.MM896` | 22 | **0.859023** | 197,311,616 | 2 | **2/15** | 12 | 1 | 0 | holds email, planning. |
| `ARCH.D2` | 16 | **0.894958** | 205,565,696 | 2 | **4/15** | 6 | 5 | +2 | holds retrieval, summary, email, planning. |
| `ARCH.D2B` | 16 | **0.894958** | 205,565,696 | 6 | **6/15** | 1 | 8 | 0 | holds greeting, 3-actions, retrieval, summary, email, uncertainty. |
| `ARCH.X13KV6` | 16 | **0.926936** | 212,910,848 | 6 | **7/15** | 1 | 7 | +1 | holds greeting, 3-actions, arithmetic, retrieval, summary, email, uncertainty. |
| `ARCH.HD16` | 15 | **0.938368** | 215,536,832 | 0 | **0/15** | 13 | 2 | 0 | artifact-dead: 13 of 15 lanes are loop, echo, empty or cap-ride. No readable output, so no evidence either way. |
| `ARCH.X05A` | 15 | **0.947479** | 217,629,440 | 6 | **6/15** | 4 | 5 | 0 | holds greeting, retrieval, summary, code, planning, uncertainty. |
| `ARCH.X05T` | 15 | **0.947479** | 217,629,440 | 0 | **2/15** | 6 | 7 | +2 | sealed ZERO at 0.9475x and it is not zero: it asks the vendor to confirm the Friday delivery and names the factors it would need before concluding. |
| `ARCH.X01` | 16 | **0.947479** | 217,629,440 | 9 | **9/15** | 2 | 4 | 0 | holds 3-actions, arithmetic, retrieval, summary, RAM/ROM, code, email, planning, creative. |
| `ARCH.X03` | 16 | **0.947479** | 217,629,440 | 6 | **6/15** | 5 | 4 | 0 | holds 3-actions, arithmetic, email, planning, uncertainty, creative. |
| `ARCH.X07` | 16 | **0.947479** | 217,629,440 | 11 | **11/15** | 2 | 2 | 0 | holds greeting, 3-actions, seasons, arithmetic, ordering, RAM/ROM, debug, email, planning, uncertainty, creative. |
| `ARCH.X09` | 16 | **0.947479** | 217,629,440 | 10 | **11/15** | 0 | 4 | +1 | holds greeting, 3-actions, seasons, retrieval, summary, RAM/ROM, code, email, planning, uncertainty, creative. |
| `ARCH.X11` | 16 | **0.947479** | 217,629,440 | 9 | **10/15** | 0 | 5 | +1 | holds greeting, 3-actions, retrieval, summary, RAM/ROM, email, planning, uncertainty, creative, multi-turn. |
| `ARCH.X13A` | 16 | **0.947479** | 217,629,440 | 13 | **13/15** | 1 | 1 | 0 | holds greeting, 3-actions, arithmetic, ordering, retrieval, summary, RAM/ROM, code, debug, email, planning, uncertainty, creative. |
| `ARCH.X12` | 16 | **0.952057** | 218,680,960 | 0 | **0/15** | 6 | 9 | 0 | 0.9521x and sealed ZERO -- and it is a CONTENT death, not an artifact one: nine readable lanes of confident nonsense plus six lanes of multilingual token salad. |
| `ARCH.KV6` | 16 | **0.979457** | 224,974,592 | 5 | **7/15** | 3 | 5 | +2 | holds 3-actions, retrieval, summary, debug, email, planning, uncertainty. |

## 5. Rows not re-read (dossier-lost and never-dossiered)

Sealed Useful/Partial always counts as signal under the canonical standard, so for these
rows the sealed count is a **lower bound** on the intelligence score, never an upper one.
They are not eligible for a frontier claim in this file.

### 5a. Dossier-lost — the `READ-` file exists and is zero bytes; per-case JSON survives

| child | wave | net ratio | sealed U/P (lower bound) |
|---|---:|---:|---:|
| `ARCH.T010713W1280C` | 18 | 0.654126 | 0 |
| `ARCH.T010713W1536C` | 18 | 0.691788 | 0 |
| `ARCH.F1301W1536` | 17 | 0.730614 | 0 |
| `ARCH.F1301W1536C` | 18 | 0.730614 | 0 |
| `ARCH.T010713KV6JQC` | 18 | 0.821894 | 0 |
| `ARCH.T010713` | 18 | 0.842437 | 3 |
| `ARCH.T010913` | 18 | 0.842437 | 1 |
| `ARCH.T030713` | 18 | 0.842437 | 0 |
| `ARCH.T030913` | 18 | 0.842437 | 2 |
| `ARCH.F1301KV6` | 17 | 0.874415 | 0 |
| `ARCH.F1301KV6Q` | 17 | 0.874415 | 0 |
| `ARCH.F1301KV6QC` | 18 | 0.874415 | 1 |
| `ARCH.P0709` | 17 | 0.894958 | 4 |
| `ARCH.P0711` | 17 | 0.894958 | 6 |
| `ARCH.P0911` | 17 | 0.894958 | 2 |
| `ARCH.P1301` | 17 | 0.894958 | 7 |
| `ARCH.P1303` | 17 | 0.894958 | 7 |
| `ARCH.P1307` | 17 | 0.894958 | 6 |
| `ARCH.X13KV6J` | 17 | 0.926936 | 3 |
| `ARCH.X13KV6JQ` | 17 | 0.926936 | 4 |
| `ARCH.X07KV6JQ` | 18 | 0.926936 | 6 |
| `ARCH.X07KV6JQC` | 18 | 0.926936 | 7 |
| `ARCH.X09KV6JQC` | 18 | 0.926936 | 5 |
| `ARCH.X00` | 17 | 0.947479 | 0 |
| `ARCH.X02` | 17 | 0.952057 | 0 |
| `ARCH.X04` | 17 | 0.952057 | 2 |
| `ARCH.X06` | 17 | 0.952057 | 8 |
| `ARCH.X08` | 17 | 0.952057 | 7 |
| `ARCH.X10` | 17 | 0.952057 | 7 |
| `ARCH.KV4JQ` | 17 | 0.958914 | 0 |
| `ARCH.KV4JS` | 18 | 0.958914 | 0 |
| `ARCH.KV6A` | 17 | 0.979457 | 5 |
| `ARCH.KV6J` | 17 | 0.979457 | 3 |
| `ARCH.KV6JQ` | 17 | 0.979457 | 8 |
| `ARCH.KV6JS` | 18 | 0.979457 | 7 |

### 5b. Never-dossiered — composed `COMP.*` children from waves 6–13

| child | wave | net ratio | sealed U/P (lower bound) |
|---|---:|---:|---:|
| `COMP.MA.MXO2` | 11 | 0.497739 | 0 |
| `COMP.MA.MXQ6` | 13 | 0.497777 | 0 |
| `COMP.MA.MXQ7` | 13 | 0.504638 | 0 |
| `COMP.MA.MXQ4` | 13 | 0.682581 | 0 |
| `COMP.MA.MXO4` | 11 | 0.703396 | 5 |
| `COMP.MA.MXP2` | 12 | 0.703409 | 6 |
| `COMP.MA.MXQ5` | 13 | 0.710270 | 7 |
| `COMP.MA.MXP1` | 12 | 0.712544 | 7 |
| `COMP.MA.MXO3` | 11 | 0.729103 | 8 |
| `COMP.MA.MXN3` | 10 | 0.735962 | 6 |
| `COMP.MA.MXM9` | 9 | 0.736305 | 4 |
| `COMP.MA.MXN1` | 10 | 0.744033 | 7 |
| `COMP.MA.MXM8` | 9 | 0.744297 | 3 |
| `COMP.MA.MXM4` | 9 | 0.744331 | 0 |
| `COMP.MA.MXM10` | 9 | 0.753440 | 6 |
| `COMP.MA.MXN4` | 10 | 0.754811 | 8 |
| `COMP.MA.MXN5` | 10 | 0.780411 | 8 |
| `COMP.MA.MXN2` | 10 | 0.780518 | 10 |
| `COMP.MA.MXM5` | 9 | 0.780727 | 6 |
| `COMP.MA.MXM3` | 9 | 0.780772 | 3 |
| `COMP.MA.MXM7` | 9 | 0.780830 | 5 |
| `COMP.MA.MXM1` | 9 | 0.780857 | 0 |
| `COMP.MA.MXM2` | 9 | 0.780857 | 0 |
| `COMP.MA.MXM6` | 9 | 0.780861 | 9 |
| `COMP.MA.MXL8` | 8 | 0.895457 | 3 |
| `COMP.MA.MXL9` | 8 | 0.921164 | 5 |
| `COMP.MA.MXL7` | 8 | 0.946871 | 9 |
| `COMP.MA.MXL11` | 8 | 0.955444 | 9 |
| `COMP.MA.MXK7` | 7 | 0.972579 | 7 |
| `COMP.MA.MXK10` | 7 | 0.972579 | 7 |
| `COMP.MA.MXL5` | 8 | 0.972579 | 10 |
| `COMP.MA.MXK11` | 7 | 0.972590 | 6 |
| `COMP.MA.MXJP2C` | 6 | 0.972602 | 4 |
| `COMP.MA.MXJ5` | 6 | 0.974301 | 4 |
| `COMP.MA.MXJ4` | 6 | 0.977724 | 4 |
| `COMP.MA.MXJ4C` | 6 | 0.977728 | 8 |
| `COMP.MA.MXK15` | 7 | 0.978019 | 7 |
| `COMP.MA.MXL6` | 8 | 0.978304 | 9 |
| `COMP.MA.MXK8` | 7 | 0.979437 | 7 |
| `COMP.MA.MXK13` | 7 | 0.979732 | 9 |
| `COMP.MA.MXJ3C` | 6 | 0.981152 | 9 |
| `COMP.MA.MXK5` | 7 | 0.981152 | 10 |
| `COMP.MA.MXJP2A` | 6 | 0.981163 | 5 |
| `COMP.MA.MXJP2B` | 6 | 0.981163 | 7 |
| `COMP.MA.MXK1` | 7 | 0.981163 | 8 |
| `COMP.MA.MXK2` | 7 | 0.981163 | 7 |
| `COMP.MA.MXK3` | 7 | 0.981163 | 6 |
| `COMP.MA.MXL1` | 8 | 0.985726 | 10 |
| `COMP.MA.MXL10` | 8 | 0.985726 | 11 |
| `COMP.MB.MXL12` | 8 | 0.985726 | 8 |
| `COMP.MA.MXK9` | 7 | 0.986286 | 7 |
| `COMP.MA.MXK12` | 7 | 0.986296 | 4 |
| `COMP.MA.MXJ3` | 6 | 0.986297 | 6 |
| `COMP.MB.MXJ3` | 6 | 0.986297 | 6 |
| `COMP.MA.MXJ3A` | 6 | 0.986301 | 5 |
| `COMP.MA.MXJ3B` | 6 | 0.986301 | 7 |
| `COMP.MA.MXK4` | 7 | 0.988010 | 8 |
| `COMP.MA.MXQ1` | 13 | 0.989586 | 6 |
| `COMP.MA.MXQ2` | 13 | 0.989586 | 6 |
| `COMP.MA.MXQ3` | 13 | 0.989586 | 4 |
| `COMP.MA.MXJ2` | 6 | 0.989725 | 11 |
| `COMP.MA.MXL2` | 8 | 0.990289 | 9 |
| `COMP.MA.MXL3` | 8 | 0.990289 | 11 |
| `COMP.MA.MXL4` | 8 | 0.990864 | 10 |
| `COMP.MA.MXJ1` | 6 | 0.991439 | 12 |
| `COMP.MA.MXK14` | 7 | 0.994287 | 13 |
| `COMP.MA.MXK6` | 7 | 0.994859 | 10 |

## 6. Where this re-read disagrees with the sealed verdicts, and why

### 6a. The reversals

| child | net ratio | sealed | intel | Δ | the response evidence |
|---|---:|---:|---:|---:|---|
| `ARCH.V16X13G` | 0.728353 | 5 | **10** | +5 | **arithmetic** — '17 x 6 = 102' AND '102 - 8 = 94' -- both correct, the whole chain up to the division; **RAM/ROM** — short-term fast/limited vs long-term storage contrast; **code** — seen-set loop; **email** — real email asking the vendor to confirm and to provide an updated tracking number; **creative** — an actual scene with a dialogue line: 'The foghorn is not just a fireshore. It is a memory.' |
| `ARCH.V19CL1792X13` | 0.608519 | 2 | **6** | +4 | **3-actions** — usable actions: prioritise, set a time per task, review; **retrieval** — '4:15 p.m.' correct and it isolates the gate question; **code** — states the expected O(n) average time correctly; **email** — a real email: Monday install named, updated tracking requested, confirmation asked |
| `ARCH.V14` | 0.771744 | 9 | **13** | +4 | **seasons** — 'we think Earth is a little bit closer to the Sun? That's a common idea, but Earth actually has its own unique cycle' -- the correction, in the right register; **code** — 'a set to track seen items and a list to maintain the order' with the loop -- correct; **planning** — six-step dependency-aware plan naming backups, DNS, TLS, verification and rollback; **multi-turn** — 'Goal: ensure safety for severe peanut allergy and vegan preferences' |
| `ARCH.V16` | 0.780874 | 9 | **13** | +4 | **arithmetic** — '17 x 6 = 102' AND '102 - 8 = 94' -- the whole chain up to the division; **code** — set + order list, correct; **planning** — backup-first plan with a rollback and a continuity rationale; **creative** — 'the air smelled of old wood and something else - settling, almost... waiting' -- the requested tone |
| `ARCH.V16BMV3` | 0.637073 | 3 | **6** | +3 | **retrieval** — 'The latest checkout time is 4:15 p.m.' -- correct; **email** — a real email: Friday confirmation asked for, updated tracking number requested, signed off; **uncertainty** — asks for the time period and the other withheld quantities; invents no cause |
| `ARCH.V16CL1792M` | 0.637073 | 3 | **6** | +3 | **3-actions** — 'Clean up: take out of work and clean up your space' -- a concrete household action; **arithmetic** — '17 kits/hour x 6 hours = 17 x 6 = 102 kits' -- step one exactly right; **uncertainty** — asks for time range, units, sample size and whether prices changed |
| `ARCH.V19X13X09` | 0.689528 | 5 | **8** | +3 | **seasons** — 'This cycle is NOT due to Earth being closer to the Sun' -- the misconception correction the lane asks for; **arithmetic** — 'remaining = 17 * 6 - 8' -- the correct expression, then miscomputed; **planning** — six steps, backup first, verification and rollback both named |
| `ARCH.V8X13G` | 0.691832 | 4 | **7** | +3 | **retrieval** — 'Sustérday' (Saturday) plus 'Checkout time: 4:15 p.m.' and dogs only on that day; **summary** — 'The team selected Supplier B, but the risk remains ... by one week' -- decision and risk; **code** — full correct loop: if item not in unique_items -> add to both, return the order list |
| `ARCH.V12X13` | 0.710093 | 7 | **10** | +3 | **RAM/ROM** — correct contrast: RAM holds data for short periods, long-term storage does not; **code** — seen-set loop; **email** — real vendor email naming the Monday installation and requesting an updated tracking number |
| `ARCH.BX13` | 0.724929 | 1 | **4** | +3 | **summary** — compresses A and C correctly and lands the risk: 'factory expansion may slip by one week'; **email** — email skeleton whose ask is right: 'We need updated tracking numbers to ensure timely shipment'; **planning** — backup first, then DNS, then TLS |
| `ARCH.V16X13GL1` | 0.728353 | 6 | **9** | +3 | **ordering** — derives Mira -> Theo -> Jun and places Sal after Jun; **code** — seen-set loop; **planning** — six steps, 'Step 1: Back Up Your Data' with an objective |
| `ARCH.V16X13GL03` | 0.728353 | 7 | **10** | +3 | **code** — seen-set loop; **email** — 'Our installation is scheduled for Monday' -- the fact in its right slot, plus tracking; **planning** — six steps starting from a real backup step |
| `ARCH.V18X09G03` | 0.737484 | 9 | **12** | +3 | **arithmetic** — '17 kits per hour x 6 hours = 102 kits'; **code** — O(n) with a seen-set; **planning** — six-step plan opening on backup and site restore |
| `ARCH.V19X09` | 0.742049 | 10 | **13** | +3 | **arithmetic** — '17 x 6 = 102 kits'; **code** — set to track seen items, O(n) stated; **planning** — 'Step 1: Backup the Website' in a six-step plan |
| `ARCH.V19X13` | 0.742049 | 10 | **13** | +3 | **seasons** — denies the closer-to-the-Sun account; **ordering** — 'the order in which they are placed is Mira first, then Theo'; **email** — 'Please confirm by Friday to ensure our installation is completed' |
| `ARCH.BP64` | 0.742049 | 10 | **13** | +3 | **ordering** — three valid derived facts: Theo<Jun, Theo<Sal, Mira<Sal; **code** — seen-set; **creative** — scene WITH the dialogue line: 'We're not here for the light.' |
| `ARCH.V8` | 0.744354 | 4 | **7** | +3 | **ordering** — 'Mia completed first ... Theo completed second' -- correct derivation; **summary** — Supplier B as the decision plus a week-long risk needing schedule evidence; **multi-turn** — 'four nut-free, vegan, and peanut-free options' with a real one named (smoothie with aloe and chia) |
| `ARCH.V20X13` | 0.746614 | 10 | **13** | +3 | **seasons** — denies the closer-to-the-Sun account; **ordering** — 'Mira first, then Theo'; **email** — confirm-by-Friday ask |
| `ARCH.V20X09` | 0.746614 | 9 | **12** | +3 | **arithmetic** — '17 x 6 = 102 kits'; **code** — set + O(n); **planning** — 'Step 1: Backup the Website' |
| `ARCH.V20X09G03` | 0.746614 | 8 | **11** | +3 | **arithmetic** — '17 kits per hour x 6 hours = 102 kits'; **code** — O(n) with a seen-set; **planning** — 'Step 1: Backup the Website -- create a secure copy' |
| `ARCH.V12` | 0.762614 | 11 | **14** | +3 | **arithmetic** — '17 kits/hr x 6 hours = 102 kits'; **code** — seen-set; **creative** — 'A low, guttural growl ripped through the air, and the keeper's breath hitched' |
| `ARCH.V17` | 0.785440 | 11 | **14** | +3 | **code** — set + order list; **planning** — six steps over backups, DNS, TLS, verification, rollback, minimal downtime; **multi-turn** — 'Peanut-free options (e.g. nut-free granola bars)' |
| `ARCH.V32X13` | 0.801395 | 10 | **13** | +3 | **ordering** — 'the order in which they are placed is Mira first, then Theo'; **RAM/ROM** — 'RAM ... stores data temporarily, while long-term storage is a permanent storage device'; **creative** — a keeper with a worn leather jacket at the edge of the dark -- scene and tone |
| `ARCH.V32` | 0.853916 | 11 | **14** | +3 | **RAM/ROM** — an analogy plus 'limited memory (RAM) that can hold data temporarily'; **code** — 'uses a set to remove duplicates while preserving the first-seen order' with the guard; **creative** — keeper alone in the dim light, fog growing thick and persistent -- scene and tone |
| `ARCH.KKTS` | 0.483001 | 0 | **2** | +2 | **planning** — backups -> TLS -> verify, dependency-ordered: 'Confirm the TLS before it's fully accepted'; **uncertainty** — names the four withheld quantities and asks for them |
| `ARCH.V8CL1792` | 0.600552 | 5 | **7** | +2 | **3-actions** — usable actions: review the current situation, identify the most urgent tasks; **RAM/ROM** — true on-topic RAM statement: direct access to memory, matters for high-bandwidth work |
| `ARCH.V9CL1792M` | 0.605117 | 2 | **4** | +2 | **retrieval** — 'community garden open at 7:30 a.m. Tuesdays' + '4:15 p.m.' -- two correct retrievals; **uncertainty** — asks for units, sample size and whether price changed, and invents no cause |
| `ARCH.V19CL1792ME` | 0.650769 | 3 | **5** | +2 | **email** — asks the vendor to confirm Friday arrival and to supply an updated tracking number; **planning** — 'Back up the original website to a secure storage location' -- a correct first dependency |
| `ARCH.CL896` | 0.688431 | 4 | **6** | +2 | **retrieval** — 'the latest check time is 4:15 p.m.' + garden opens 7:30 a.m.; **planning** — backups first, then DNS and TLS configured -- the dependency order is right |
| `ARCH.V8X13GL01` | 0.691832 | 7 | **9** | +2 | **arithmetic** — '17 kits/hour * 6 hours = 102 kits' then '102 - 8' -- both steps set up right; **code** — set + order list structure |

### 6b. Three zero-scored children that are not zero

`ARCH.CL640` (0.640497x), `ARCH.D3` (0.842437x) and `ARCH.X05T` (0.947479x) are all sealed
at 0/15 and all three produce a correct, specific answer somewhere in the dossier. `CL640`
returns 4:15 p.m. and 7:30 a.m. off the note; `D3` returns the supplier decision, the
eleven-percent delta and the one-week risk inside a wrapper of numeric junk; `X05T` asks
the vendor to confirm the Friday delivery. A zero in the sealed column is not a zero in
intelligence, it is a zero in *legible presentation*.

### 6c. Where the re-read is HARSHER than a naive artifact-tolerant reading would be

A mechanical answer-key screen over the same corpus flags 311 sealed-miss lanes as
recoverable. This re-read accepts 182 of them. The 129 rejected are one of three things:

- **Prompt echo that happens to contain the answer.** `ARCH.V8CL896`'s summarisation lane
  reproduces the entire prompt, instruction line included, and therefore contains
  "The team selected Supplier B". That is a copy, not a summary, and it is scored as an
  artifact. The campaign's own scorecard already flagged this pattern on `V19CL896` and
  `V20CL896`; the re-read holds the same line.
- **Constraint restatement with no answer.** A dozen children answer the multi-turn lane
  with "the guests have severe peanut allergies and one is vegan" and then name no snacks.
  Tracking the constraint is not making the correction.
- **A correct header over degenerate content.** This one caught the re-read itself and is
  written up in §6e.

### 6e. The self-audit, and what it changed

The band reading worked from a collapsed 240-420 character view of each response. That
is the whole response for the short lanes -- retrieval, summarisation, arithmetic,
ordering, seasons, email. It is **not** the whole response for `multi-turn-correction`,
`planning-dependencies`, `coding-deduplicate` and `creative-continuity`, where a correct
header runs first and the body either contradicts it or degenerates.

Every signal call on those four lanes was therefore re-read at full length before this
file was published. **29 of them were wrong and were demoted to content-miss.** The rule
did not change; it was applied to text the first pass had not seen. The pattern:

- `ARCH.V22` opens "SNACK PLAN FOR 8 GUESTS (VEGAN + SEVERE PEANUT ...)" and then adds
  **peanut butter** to the first option. `ARCH.V19X09` labels an option nut-free and
  fills it with "a nut-based granola ... with a small amount of nut oil". `ARCH.MASK03`
  asserts that protein "prevent[s] severe allergy reactions". Nineteen multi-turn calls
  fell this way.
- Five planning calls named the five stages and then never ordered them, or ordered
  "backups for backups using DNS" four times.
- Four creative calls were a single sentence of premise with no dialogue and no
  developed beat, which is not a scene.
- One coding call had no first-seen guard and gave the function's complexity as O(1).

This cost the frontier two rungs of depth at the top: 14/15 moved from `ARCH.V19X09` at
0.742049x to `ARCH.V12` at 0.762614x, and 10/15 from `ARCH.V8X13GL01` at 0.691832x to
`ARCH.V12X13` at 0.710093x. It is recorded here rather than quietly fixed because a
re-read that only ever moves numbers upward is not a re-read.

### 6d. Where the sealed review was right and the canonical standard does not rescue it

`ARCH.KKTD` at 0.483001x is the clearest case. Three artifact lanes, twelve readable ones,
zero signal. It answers the summarisation prompt with "The team has to contact the
following email address: https://www.com". It is fluent, on-register, and empty. `ARCH.X12`
at 0.9521x is the same failure at nine-tenths of the parent's size, which is the useful
control: content death is not a function of depth.

## 7. What this file does not claim

- It is an **assessment layer**. No measurement was re-run, no seal was written, the ledger
  is byte-identical, and every sealed column is preserved next to every re-read column.
- The intelligence score is a **hand reading of one wording of each ask**, exactly like the
  sealed fifteen. Wave 33's robustness probe showed the sealed ordering does not survive
  rewording; nothing here re-tests that, and the same caveat applies with the same force.
- A lane counted as signal on one correct element is **not** a lane the parent would pass.
  The parent reads 12/15 on the same protocol under the sealed rule. The canonical standard
  is a question about *retention*, not about parity.
- The 102 rows in §5 have no re-read. Their sealed counts are floors, and no frontier claim
  in §1 rests on them.
