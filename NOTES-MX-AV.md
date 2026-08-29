# NOTES-MX-AV — wave 44: the instrument wave. The union copy measure is pre-registered and reproduces the hand column with nobody in the loop, and the CL768 ladder ends at K=4,096 and then comes back at K=3,072

Branch `codex/lrc-blank-compression`. Five commits, five deployments, three
graded children, ledger 1046 → 1052.

**The one-line version.** The measure wave 42 asked for and wave 43 did not
register is registered now, in the first commit, before this wave decoded a
token — and run over 57 children **with no hand input at all** it reproduces the
floor that until now only a human reading could produce. Meanwhile the campaign's
only flat ladder ended, and then un-ended: `ARCH.V4CL768` was registered at ONE
on eight rungs of precedent and read **0/15**, and `ARCH.V3CL768` one rung deeper
was registered at ZERO and read **1/15**, moving the sub-0.40x column-A floor a
fifth consecutive wave to **0.3859919152x**. Both registrations are wrong in
opposite directions on adjacent rungs of one ladder. And what holds that floor is
a model that has collapsed into copying its input.

---

## 1. The tree, and the two pins nobody had

Five substitutions, no prose. Four hand edits: 39 seal roots become 40; `KEEP`
and `DROP` emptied; one schema string retargeted, sixteen module references, both
receipt write sites, the log prefix and the header's own diff reference; and
`sealq.sh` arrived carrying wave 43's **three** seal calls.

Wave 42's general calibration pin fired for the **second** time and reported
**62** residue lines below the sentinel in `run/mx43/sealq.sh` as shipped — wave
43's calibration result, its nine-dossier byte comparison, its reading-order
finding and its three seal calls. All 62 deleted by hand. Wave 43 was the first
cut to run it and it reported exactly the 78 lines `NOTES-MX-AT` §11.4
predicted; two firings with a number each make it a measurement rather than a
theory.

### 1.1 TWO OF `sealq.sh`'s THREE HAND-RETARGETED PATHS HAD NO PIN, AND THAT IS MEASURED RATHER THAN ASSERTED

`NOTES-MX-AU` §13 item 6 said the next place to look is every file a wave edits
by hand each time with nothing but prose behind it. The nearest one is not
another file — it is three lines inside a file that is already in `HAND_EDIT`.

Because `sealq.sh` is copied VERBATIM, every cut retargets by hand the wave's
`env.sh`, the wave's per-case writer and the wave's `seal_verdict.sh`. Wave 43's
copy says all three "were retargeted by hand at pinned counts of one each".
`FORBIDDEN_AFTER`'s needles are `mx43_` and `MX43-`; `nf-mx43/env.sh` carries
`mx43/` and `$MX43/seal_verdict.sh` carries `MX43/`, and **neither needle reaches
either one**. Checked, not argued. Only the per-case writer was covered, and only
because its filename happens to carry an underscore.

The cost of that hole is not cosmetic. `env.sh` exports `$W`, the wave root every
cell, shard, log and dossier is written under. A `sealq.sh` that kept last wave's
source line would have written this wave's seals into last wave's wave root under
this wave's cell ids, and nothing in the campaign would have said so.
`cut_wave44.py --audit` now pins all three in both directions, and this wave's own
copy of the explanatory paragraph deliberately does **not** spell the needles out,
because a file whose job is to carry zero references to the previous wave should
not contain three of them in a paragraph explaining why.

### 1.2 The builder edit broke the file once, and the refusal was free

The header replacement spliced at the docstring's OPENING quotes instead of its
closing ones and left wave 43's docstring body loose at module level.
`py_compile` refused it immediately, the file was re-copied verbatim and the edit
redone against both quote positions. It is recorded in `mx44_build.py`'s own
header rather than quietly repaired. The refusal cost nothing because it fired
before anything was committed, which is the only kind of refusal that is free.

### 1.3 The sealer's second accept list was appended AT THE CUT

Seven strings, in the same commit that cut the tree, and `cut_wave44.py --audit`
asserts them ten strings deep. Wave 43 claimed this and had not done it, and paid
a whole deployment when the sealer refused all three of its reviews after the GPU
work was already finished. **Wave 44's `sealq.sh` ran ONCE and all three sealed on
the first call.**

---

## 2. THE INSTRUMENT: REGISTERED FIRST, THEN TESTED, AND ONE OF THREE PREDICTIONS IS MINE TO OWN

`mx44_dualband.py`'s `payload_union_coverage` is the **primary** copy-back
measure at `COPY_BACK_UNION_COVERAGE = 0.60`. The longest-contiguous-run measure
is computed on every lane, published as `legacy_longest_run_classification`, and
decides nothing. The constant and the three predictions below were committed at
`2fed6b56`, the tree cut, **before the pass that tests them ran and before any
wave-44 child existed**.

**Where 0.60 comes from, and it is not from anything this wave measured.**
`wave42/deep-band-dual-assessment.json` — sha256 `fe20e916…`, committed at
`5af0b035` two waves before this file existed — carries **34 distinct
summarisation responses**. Seventeen were hand-read as copies and their union
coverage runs **0.826972 to 0.997455**; every other distinct response sits at
**0.483461 or below**. The separating interval is **(0.483461, 0.826972)**, it is
0.3435 wide, and nothing lies inside it. 0.60 clears the highest non-copy by
0.1165 and undercuts the lowest copy by 0.2270, and it is the numeral the
campaign already declared for the demoted measure, so this wave changes the
MEASURE without also changing the NUMBER. **Any threshold inside that interval
gives identical calls**, which is what makes the specific value not
load-bearing.

| # | quantity | registered | measured | |
|---|---|---|---|---|
| **P1** | `column_b_machine_union_only` floor, no hand input | 0.4116484014x, `ARCH.V10CL896X09` | **0.4116484014x, `ARCH.V10CL896X09`** | **HIT** |
| **P2** | `column_b_machine_legacy_only` floor, same pass | 0.3951221469x, `ARCH.V5CL768` | **0.3951221469x, `ARCH.V5CL768`** | **HIT** |
| **P3** | `machine_hand_disagreements`, hand-merged pass | ≤ 3, down from 23 | **15** | **MISS** |

**P1 is the wave's result.** The union measure alone, with nobody in the loop,
reproduces the column-B floor that until this wave only a human reading could
produce. Wave 43's machine-only pass put that floor at 0.3951221469x and was
wrong; wave 44's puts it at 0.4116484014x and is right.

**P2 is the regression check** that promoting one measure did not silently edit
the one it replaced. It reproduces `wave43/deep-band-dual-assessment-machine.json`
exactly.

### 2.1 P3 IS A MISS AND THE REGISTRATION IS WHAT WAS WRONG

Seven of the fifteen disagreements are `uncertainty-insufficient-data` lanes
where the machine says `CAP_RIDE` and the hand says `CONTENT`. **Neither label is
`COPY_BACK`, so neither voids anything**, and the same seven rows are inside wave
43's 23. The machine has five labels and the reviewer was only ever asked "copy
or not", so that metric has a seven-row floor no measure change can reach.
Registering 3 against it was my defect and it is recorded as one rather than
explained away.

**The number that should have been registered is the VOID-CHANGING count** — rows
where exactly one side says `COPY_BACK`, the only rows that can move column B:

| measure | total disagreements | **void-changing** | decomposition |
|---|---:|---:|---|
| legacy longest run (wave 43) | 23 | **16** | 9 `summ` + 7 `email` |
| union primary (wave 44) | 15 | **8** | 7 `email` + 1 `unc` |

**The union measure halves the void-changing error rate and clears the
summarisation lane completely, 9 → 0.** Every summarisation copy in the band is
now found by machine.

### 2.2 The failure mode named in the registration happened, once, above the floor

`run/mx44/README.md` §2.1 warned before the measurement that
`uncertainty-insufficient-data` has no payload anchor, so its payload is the
whole prompt, and that if it cleared 0.60 the machine would void a lane the hand
read as CONTENT. **`ARCH.V9CL896G03`'s does, at union coverage 0.693333**, and the
machine-only pass publishes it as an **anomaly** — a copy-back on a SELECT lane,
which wave 40's precedents say cannot happen. Wave 43's artifact fired zero
anomalies. It sits at 0.4373492x, 0.0257 of ratio **above** the column-B floor,
so P1 survives it, and P1 is hit with one known false positive rather than none.

### 2.3 ALL EIGHT RESIDUAL ERRORS HAVE ONE MECHANISM, AND IT IS MEASURED

Every void-changing error is on a lane with **no `PAYLOAD_ANCHORS` entry**.
`PAYLOAD_ANCHORS` covers `summarization-priorities`, `context-retrieval` and
`debugging-off-by-one`; a lane with no entry is measured against the
concatenation of its user turns, which is the whole prompt **including the
instruction text**. That produces both error directions from one cause:

* `writing-concise-email` **under-reads**: seven held lanes the hand called
  copies land at union 0.461140–0.481865, because the copied material is the ask
  and the ask is about half the prompt.
* `uncertainty-insufficient-data` **over-reads**: a response that echoes the
  instruction text scores against instruction text sitting in the denominator,
  which is how 0.693333 happens on a lane that is not a copy of anything.

**It is not fixed this wave.** Adding an anchor would move seven classifications
*after* the measurement that revealed them, which is exactly what wave 42 refused
to do and exactly what this wave's whole argument depends on not doing. The
mechanism is measured, published, and left for a wave that can register the
anchors first.

### 2.4 The retrospective column, and what it is not

All 54 pre-wave-44 children carry `instrument_reading_is_retrospective: true`.
**Nothing about them is re-graded**: sealed counts stand, column A is still the
sealed per-case verdict the module refuses to disagree with, and the hand calls
waves 42 and 43 recorded remain authoritative. The estate's column B is unchanged
at 0.4116484014x. Ten held lanes across the band read differently under the two
measures, and the eight distinct responses that move are listed by digest in the
artifact.

**And the measure was used PROSPECTIVELY on this wave's own children.** The
pre-registration predicted that if any of the three held a case, its held lane
would be `summarization-priorities` and the union measure would call it
`COPY_BACK` at ≥ 0.60 without a hand reading. `ARCH.V3CL768`'s held lane reads
**union coverage 1.000000** and the machine called it, before anybody read the
dossier. The other two children's summarisation lanes read 0.086514 and 0.053435
and were correctly left alone.

---

## 3. THE VOCABULARY CENSUS: THE STRUCTURAL FLOOR DOES NOT BIND ANYWHERE ON THIS LADDER

| K | mandatory | admitted by ranking | skipped for closure | merges | occ coverage | distinct coverage | captured-batch occ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,120 | 765 | 3,027 | **0** | 4,355 | 0.985301 | 0.700428 | 0.998264 |
| 4,096 | 765 | 2,273 | **0** | 3,331 | 0.971463 | 0.553342 | 0.994475 |
| 3,072 | 765 | 1,533 | **0** | 2,307 | 0.947327 | 0.401043 | 0.986109 |

K=5,120 was re-censused as a control although wave 43 published it, and it
reproduces wave 43 to every published digit (0.985301 occurrence coverage, 3,027
admitted). A census whose only new rows are the ones it is about to justify has
no control in it.

**The merge-closure gate — the half with no flag, the half `--below-floor` cannot
release — does not fire at K=3,072**, four times the 765-row structural floor. So
whatever ends this ladder, it is not the closure. What moves is coverage, and it
moves on the DECODE surface far faster than on the evidence the body was solved
against: distinct coverage falls 0.700428 → 0.553342 → 0.401043 while
captured-batch occurrence coverage only falls 0.998264 → 0.994475 → 0.986109.

---

## 4. F1 and F2: THE CL768 LADDER IS NOT MONOTONE IN K

| cell | K | parameters | ratio | registered | **sealed** |
|---|---:|---:|---:|---:|---:|
| `ARCH.V5CL768` (wave 43) | 5,120 | 90,756,864 | 0.3951221469 | 1 | 1/15 |
| **`ARCH.V4CL768`** | **4,096** | **89,708,288** | **0.3905570311** | **1** | **0/15** |
| **`ARCH.V3CL768`** | **3,072** | **88,659,712** | **0.3859919152** | **0** | **1/15** |

Eight consecutive rungs — the body at 0.6644639834x and V12, V10, V9, V8, V7,
V6, V5 — held 1/15 across 0.2693 of stored ratio through eight identical
1,024-row steps. **The ninth step kills it and the tenth brings it back.** Both
of this wave's vocabulary registrations are wrong and they are wrong in
**opposite directions on adjacent rungs of one ladder**, which is a thing no
single-cell wave could have found and the reason both rungs were built.

**`ARCH.V4CL768` dies with the copy path HALF FORMED.** Its summarisation lane
opens with the same substitution signature the living rung above it carries —
*The team comperses three bitties, including the A and B cost* — and then
collapses into *Suggesting the most of the most of the most* for a thousand
tokens and rides the cap. The decision is gone, the risk is gone, and there is
nothing left for the copy-back clause to reach. All sixteen lanes charged, six
ride the cap, five return the prompt itself.

**`ARCH.V3CL768` moves the sub-0.40x column-A floor a FIFTH consecutive wave**,
from 0.3951221469x by 0.0091302317 of stored ratio, and it is the deepest child
the campaign has ever built that holds anything. Its held lane is the
summarisation lane, sha256 `d10952d96d99eeea`, an **independent witness shared
with nothing in the estate**.

### 4.1 WHAT HOLDS THE FLOOR IS AN ECHO, AND THAT IS THE FINDING

At K=3,072 the child has stopped answering and started copying. **Eleven of
sixteen lanes return the prompt itself**, and four of those — `logic-ordering`,
`coding-deduplicate`, `creative-continuity`, `multi-turn-correction` — are the
prompt with **no edit at all**. The lane-byte comparison decided that by
construction: `ARCH.V3CL768` and `ARCH.V4CL768` are byte-identical to *each
other* and to `ARCH.V5CL768`, `ARCH.V6CL768` and `ARCH.V7CL768` on
`creative-continuity`, and to `ARCH.CL1408` and `ARCH.V19CL1536` on
`multi-turn-correction`, because on those lanes all of them return the prompt and
the prompt is the same bytes for everybody.

The held lane is the whole summarisation prompt, **instruction line included**,
with one visible corruption in 484 characters (*l'aunch*), and it preserves the
decision (*The team selected Supplier B*) and the unresolved risk (*the factory
expansion may slip by one week, so procurement will request weekly schedule
evidence*) verbatim.

### 4.2 THE ADJUDICATION, PUBLISHED RATHER THAN ABSORBED

Holding a response that echoes the instruction line is a step past anything the
campaign has sealed, and the alternative reading was available: **"returns the
PROMPT ITSELF" is a named degeneracy** in `sealq.sh`'s own reading standard, and
every other lane in the campaign that has done it has been read a miss.

It is held under column A anyway, for one reason. Wave 43's seal of
`ARCH.V5CL768` — the very child whose floor this one takes — states the criterion
it used in one clause: *"it preserves both asked-for items, the decision and the
unresolved risk"*. `ARCH.V5CL768`'s held lane, sha256 `f0f9cef07b1aa994`, is the
payload paragraph with heavy substitution (*comperses three bitties*, *Suggesting
A*, *Suggested B*, *remainringing*, *procuration*). `ARCH.V3CL768`'s preserves the
same two items **more cleanly**. Refusing to hold it because it *also* echoed the
instruction would apply a stricter standard to a better-preserving response than
the campaign applied one wave earlier. That is not a standard.

The standard's own escape hatch is explicit and was applied: charging and holding
are independent — *"a lane can carry a substitution and still answer"* — so the
lane is **charged and held**, exactly as wave 43's was. Column B voids it either
way. A reader who prefers the other reading can re-decide it from
`d10952d96d99eeea` against `f0f9cef07b1aa994` without re-running anything, and
the whole of the disagreement is worth 0.0091302317 of column-A ratio and nothing
at all to column B.

---

## 5. F3: `ARCH.V5CL896X13` — the campaign's first matched-K pair across two bodies

K=5,120 on wave 35's retained `MX35-CL896X13`, **89,309,952 parameters,
0.3888228220x, 0/15**, registered at ZERO on the cliff account. It is the wave's
only correct registration.

`ARCH.V5CL768` and `ARCH.V5CL896X13` are both K=5,120 gathers taken with the same
census, the same `(-census_count, parent_token_id)` ranking and the same closure,
so **they keep the same rows** and differ only in the body underneath: the
two-axis CL768 body against the block-13 excision body. One holds a case at
0.3951221469x and the other holds none at 0.3888228220x. **The death below the
block-13 cliff is therefore a property of the BODY and not of the vocabulary
fold** — a claim waves 42, 43 and 44 have all asserted and none had controlled.

It also confirms the cliff at a fourth consecutive dead rung: 1/15 at K=8,192 and
zero at K=7,680, K=7,168, K=6,144 and now K=5,120.

**Its uncertainty lane had to be read twice and it is a miss.** It answers *The
graph shows that you have not yet provided the time range, units, and sample
size* — three of the four variables **the prompt itself enumerates**, handed
straight back, with no statement of what can safely be concluded and no request
for anything it was not given. Wave 40's precedent is that a copy satisfies a
PRESERVE ask and neither a SELECT ask nor a PRODUCE ask; this lane's ask is to
SELECT, and returning the prompt's own list selects nothing. All sixteen lanes
charged, seven ride the cap, seven return the prompt itself.

---

## 6. Benchmarks, and the sharpest inversion in the deep band

Standard invocation, parent FIRST. **The parent row reproduces waves 32 and 34
through 43 to every reported digit — 31 parent rows, all 31 matching** across 265
benchmark rows and 356 reviewed children.

| model | ratio | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | sealed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 1.000000 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | **81.80** | 12/15 |
| `ARCH.V4CL768` | 0.390557 | 0.2900 | **0.3400** | 0.5400 | 0.4650 | **0.1000** | **36,052.81** | **8,749.05** | **0/15** |
| `ARCH.V5CL896X13` | 0.388823 | 0.3100 | 0.3350 | 0.5200 | **0.4900** | 0.0700 | 163,514.75 | 35,554.55 | **0/15** |
| `ARCH.V3CL768` | 0.385992 | **0.3150** | 0.3100 | **0.5500** | 0.4750 | 0.0700 | 61,371.70 | 14,018.55 | **1/15** |

**`ARCH.V4CL768` reads 0/15 and is the better language model.** It beats the
1/15 child by 1.6x on wikitext word perplexity and 1.7x on lambada perplexity,
and by three points on hellaswag. The child that holds a case holds it by
echoing, and every aggregate metric says so. That is the sixth inversion the
benchmark column has produced and the first where the inversion has a stated
mechanism instead of a shrug.

---

## 7. THE DUAL COLUMN, refreshed over 57 children, both floors published

`mx44_dualband.py` at ≤ 0.4700, machine pass first, then the hand reading merged.
**0 anomalies on the hand-merged pass, 1 on the machine pass, 15
machine-vs-hand disagreements, all published.**

- **Column A (copy-backs count): floor = 0.3859919152x, `ARCH.V3CL768`**,
  holding one lane. **The floor moves for the fifth consecutive wave.**
- **Column B (copy-backs refused): floor = 0.4116484014x, `ARCH.V10CL896X09`**,
  holding `uncertainty-insufficient-data`. **Unchanged since wave 38 — six
  waves.**

The gap is **0.0256564862 of stored ratio, 5,893,120 parameters**, and the whole
of it is one lane read two ways. Twenty-four children sit below the column-B
floor and **twelve** of them hold at least one case under A; B voids every one,
all on `summarization-priorities`.

**And the machine now agrees with the hand about where B's floor is**, which is
the first time in the campaign that has been true. `column_b_machine_union_only`
= 0.4116484014x against `column_b_machine_legacy_only` = 0.3951221469x.

---

## 8. Method and controls

- **Parent calibration, seventeenth consecutive, exact on the count AND on the
  split.** Re-read blind from `ARCH.V5CL896X13`'s dossier — the first of the
  wave, sealed 10:33:59.979Z, four seconds before `ARCH.V4CL768`'s — before any
  child lane was opened. **12/15**, **cap-hits 0/16**, both documented anchors
  verbatim. The split is **6 Useful / 6 Partial / 3 miss**, which is wave 43's
  split lane for lane, against the reference's 5/7/3 and wave 42's 4/8/3; all
  four totals are 12/15 and all four name the same multi-turn miss. Two calls
  were made against this file's own stated calibrations rather than a first
  impression, and `sealq.sh` records both: `planning-dependencies` is Useful on
  wave 20's five-items rule (backups, DNS, TLS, verification and rollback are
  Steps 1–5 of six), and `creative-continuity` is Partial because wave 29's
  calibration caps it there even when all three motifs carry.
- **The runner control is NINE DOSSIERS WIDE ACROSS FOUR WAVES.**
  `lanebytes44.py` opens all three wave-44 dossiers and six from waves 41, 42 and
  43 and finds the parent's sixteen lane digests **identical 16/16 in every one**,
  at protocol `e5e377688597` and prompt pack `0ee6bf14cef5` throughout — the same
  two digests wave 43 reported.
- **The lane byte comparison ran before any count was written**, and it decided
  three claims by construction: `ARCH.V3CL768`'s held lane is an independent
  witness shared with nothing; `ARCH.V4CL768` shares five of sixteen lanes and
  the other two children two each; and the shared lanes are exactly the lanes on
  which the children return the prompt.
- **The reading order for children under 0.60x was followed on all three.** The
  summarisation lane of each child was read first and written down before any
  other lane of that child was opened. It decides all three seals.
- **Every child asserted its parameter count before writing** and all three
  landed on the pre-registered value exactly: 89,708,288 / 88,659,712 /
  89,309,952. Identity gate: `torch.equal` on the whole gathered table for all
  three.
- **`BELOW_FLOOR=1` on all three rungs, and the guard it disarms is asserted
  present** by `cut_wave44.py --audit`. The census said before the build that the
  structural half would not fire and the coverage half would.
- **Seal verification**: 1052 ledger rows, **1048 resolve to a receipt on disk,
  1048 verify, 0 FAIL**, 4 unresolved — the same historical four. **40 seal
  roots** audited against disk, **0 undeclared**.
- **`sealq.sh` ran ONCE** and all three sealed on the first call. The sealer's
  accept list was appended at the tree cut, so nothing was refused.
- **A dry run preceded the ledger merge** and left the ledger at 1046 rows and
  `264884995e0c2b87…` unchanged — verified, not assumed. The correct argument
  form was used first time: six `--shard`, three `--verdict-receipt`. Applied in
  **one pass**, 1046 → **1052**, 0 superseded.
- **Two background jobs raced on nothing.** Both long passes wrote unique output
  paths and every launch ran a `pgrep` pre-kill check first. The 300-second SSH
  command ceiling orphaned two waiting loops and one `mx44_verify_seals` run; the
  underlying `nohup`'d work was unaffected, the loops exited with their channels,
  and `pgrep -af mx44` reads clean.
- **Pre-registration hit rate: 1 of 3** on the sealed counts (`V4CL768` 1→**0**;
  `V3CL768` 0→**1**; `V5CL896X13` 0→**0**, hit), plus 2 of 3 on the instrument
  predictions and 4 of 4 on the secondary quantitative ones.

---

## 9. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `2fed6b56` | cut wave 44's tree and PRE-REGISTER the union copy measure as primary |
| `165a1555` | run the census and the retrospective pass, and pre-register the three cells |
| `a020ccd1` | the three seals |
| *(this)* | the measurement, the tables and the dual-column doc refresh |
| *(next)* | retention applied, and this file's own estate section closed |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `2fed6b56` | 7721 | 0 | 0 | 0 | 0 |
| `165a1555` | 7726 | 0 | 0 | 0 | 0 |
| `a020ccd1` | 7726 | 0 | 0 | 0 | 0 |

**Five deployments for five commits, which is the campaign's normal rate, and
NONE of them is avoidable.** The extra commit against waves 40–42's four is the
same structural one wave 43 had and `env.sh` states it: the census and the
retrospective pass must RUN before the predictions they make possible can be
registered, so the code commit and the pre-registration commit cannot be the same
commit. Wave 43's avoidable extra — the accept-list defect — did not recur.

**Ledger.** Re-verified on the share before it was touched (1046 rows,
`264884995e0c2b87…`), backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave44-20260829T105851Z` and
`cmp`-verified byte-identical. **One-pass merge**, 1046 → **1052** rows, 0
superseded. Byte-prefix verified: `head -c 1846137 ledger.jsonl | sha256sum`
returns `264884995e0c2b87…` exactly. New digest `9e4ff05ea3a5b3d0…`. **Mirror
copy-back done, and the mirror write asserted the merged file is a byte-prefix
extension of what it replaced BEFORE writing.**

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used for the
reviews and the benchmarks, with the SM86 preflight on every driver. Both carried
an idle-holding H3 ComfyUI container (288 and 279 MiB) and were **not restarted**,
for wave 34's stated reason. The four P40s and the Quadro RTX 8000 were never
touched; index 6 read 330 MiB throughout and this wave issued no command against
it. All three builds were **CPU-only** vocabulary gathers.

**Disk**: `/mnt/ai_data` 4.5T. **The Codex lane was not touched** — 324
uncommitted paths before the wave and 324 after, and every commit staged by
explicit path.

---

## 10. What this wave did NOT build, and why

1. **No PLANE child.** Wave 43 re-sealed the frequency-band axis with the child
   that opened it, under the slate's own kill gate. `mx44_plane.py` is carried,
   fully pinned and dormant.
2. **No XCHANGE child.** Closed by its own pre-registered gate at wave 43. This
   is a result, not a deferral, and it stays one.
3. **ROWSET Arms A and B** — still blocked on corpus; no corpus has appeared.
4. **ALIAS** — still blocked behind the PROBE storage-claim ruling (K4, K3(c)).
5. **NORM-ESTATE Arms M and S** — still need a new solver. Wave 43 authored two
   and had no room for a third; wave 44's deliverable was an instrument and it
   authored no module at all.
6. **The 30-prompt extension surface was NOT run**, and that is a deliberate
   decline of `NOTES-MX-AU` §13 item 2, which asked the next wave that grades
   anything to make it a primary surface. The ordered slate for this wave was the
   union measure, the two ladder rungs and the cross-check; adding an unordered
   GPU column would have been the same class of drift the wave exists to correct.
   It costs about seven minutes of 3090 time and it is the cheapest unspent item
   on the board.
7. **The `writing-concise-email` payload anchor was NOT added**, §2.3. The defect
   is measured and published; the repair belongs to a wave that can register it
   before the pass that uses it.

---

## 11. The two rulings

**Ruling 1 — the PROBE storage claim** (`NOTES-MX-AS` §12), unchanged and still
blocking three classes: K4, K3(c) and ALIAS.

**Ruling 2 — the copy-back clause**, unchanged, still costing nothing to apply,
and **sharper this wave than it has ever been**. Both readings are complete over
57 children, per child and per lane, with the digest that makes each call
checkable:

- **NO** (copy-back refused): the intelligence floor is **0.4116484014x**,
  `ARCH.V10CL896X09`, holding `uncertainty-insufficient-data`. Unmoved by this
  wave, and now reproduced by machine with no human in the loop.
- **YES** (copy-back counts): the floor is **0.3859919152x**, `ARCH.V3CL768`
  — a child that returns the prompt itself on eleven of sixteen lanes, whose
  held lane is the whole summarisation prompt including the instruction, and
  which is measurably a *worse* language model than the 0/15 child one rung
  above it.

The question is no longer abstract. Under reading A the campaign's deepest living
child is an echo, and the benchmark column agrees.

---

## 12. What the next wave should know

1. **THE UNION MEASURE WORKS AND ITS RESIDUAL ERROR IS ONE DICT ENTRY WIDE.**
   §2.3. Eight void-changing errors, all on lanes with no `PAYLOAD_ANCHORS`
   entry. Register anchors for `writing-concise-email` and
   `uncertainty-insufficient-data` BEFORE the pass that uses them, predict
   void-changing disagreements falling from 8 to 0 or 1, and it costs no GPU at
   all.
2. **THE CL768 LADDER IS NOT MONOTONE AND THE NEXT RUNG IS CHEAP.**
   `ARCH.V2CL768` (K=2,048, 87,611,136, **0.3814268x**) is one CPU pass off wave
   40's retained `MX40-CL768`. The structural floor is 765 rows, so the axis has
   room; the census will price it before the build. After the V5/V4/V3 pattern,
   an honest registration has to state which of "the echo survives" and "the echo
   breaks" it is betting on and why, because the flat-ladder prior is now dead.
3. **A MATCHED-K PAIR IS THE CHEAPEST CONTROL IN THIS CAMPAIGN.** §5. Two gathers
   at one K on two bodies isolate the body for one review, because
   `choose_kept` depends on the parent vocabulary, the census and K and on
   nothing about the source model. The estate has thirteen waves of body
   comparisons that were never controlled this way and could be.
4. **THE 15-CASE INSTRUMENT IS NOT MEASURING LANGUAGE MODELLING AT THIS DEPTH.**
   §6. A 0/15 child with 8,749 wikitext wppl and a 1/15 child with 14,019.
   Wave 43 said the same at 0.98x with a 22x gap. **The 30-prompt extension bank
   is still unspent and is now the cheapest unclaimed item on the board**, §10.6.
5. **RETURNING THE PROMPT ITSELF IS THE DEEP BAND'S TERMINAL BEHAVIOUR AND THE
   STANDARD DOES NOT NAME IT SEPARATELY.** §4.2. `sealq.sh` charges it as a
   degeneracy but explicitly leaves holding independent of charging, so a pure
   echo can hold a PRESERVE lane. That is defensible and it is what wave 44
   applied — but a wave that wants the other reading should propose it as an
   amendment to the standard, before it grades anything, rather than discovering
   it at a seal.
6. **BOTH APPEND-ONLY ACCEPT LISTS AND ALL THREE `sealq.sh` PATH RESOLUTIONS ARE
   NOW CHECKED.** §1.1 and §1.3. The next place to look is
   `experiments/seal_matrix_behavior_review.py`'s own constants, which
   `seal_verdict.sh` passes flags to and which no cut has ever read — wave 43
   named it and it is still true.
