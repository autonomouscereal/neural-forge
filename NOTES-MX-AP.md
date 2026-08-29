# NOTES-MX-AP — wave 38: the block-9 ladder decays with K, and the excision axis is exhausted

**2026-08-28.** Branch `codex/lrc-blank-compression`. Six children built, sealed
and read twice; no reproduction control was needed because every source was
either retained or built this wave. Ledger 976 → **988** rows,
`2beb2c7e966a8f7fe312b8a6d000b266a6bb3dec662cc1408d15da5a22139c7f`, byte-prefix
verified against the pre-wave backup on the share. **Two deployments, and the
accept list cost neither of them — third consecutive wave.**

**Four of six pre-registrations hit, and the two misses are the wave's result.**
§2.

---

## 1. Wave 38 changed no code either, and asserted the three lines it depends on

Wave 37 made no behavioural edit. Wave 38 is the second consecutive wave in that
position, **which is when a carried guard is most likely to have been quietly
dropped**, so `cut_wave38.py --audit` runs wave 36's five width × MLP assertions
and wave 37's two live-mask assertions on this wave's copy of the builder before
anything builds.

Wave 38's own build list needs no new code either: `--ff` since wave 15,
`--excise` since wave 16, multi-index excision since wave 17, their composition
since wave 35, the vocabulary gather since wave 25.

**What it does exercise that no wave has since 28 is `--excise` with TWO
indices**, and **no wave has ever composed one with `--ff`.** Three lines carry
it and none had ever been asserted:

* `        for index in sorted(excised, reverse=True):` — the deletion loop.
  Ascending, the second `del` addresses a list the first already shortened. This
  wave's own pair (9, 13) would raise; **`--excise 9 --excise 11` in ascending
  order would delete block 9 and then whatever block 12 used to be, silently, at
  a correct-looking parameter count and a correct-looking depth.**
* `                layer.self_attn.layer_idx = position` and
  `                layer.conv.layer_idx = position` — the module re-index. Both
  address the KV cache and the conv state by that integer, and two excisions
  renumber a longer tail than one does.

**The needles carry their indentation and that is not decoration.** The wave-38
header of `mx38_build.py` quotes all three lines by name, so a bare-string count
reads 2 for each — and a pin of 2 would still pass with the code deleted and only
the prose left. The indented form exists exactly once, inside `build()`'s
`if excised:` branch. That trap was found by the audit refusing, not by
reasoning.

All three registered acceptance criteria for the G2 body were met exactly:

> `[mx32] excised blocks [9, 13]; child depth 12; config.layer_types re-derived (12 entries) so the live mask follows the live block`

> `layer_types: conv conv full_attention conv full_attention conv full_attention conv full_attention full_attention conv full_attention`

> loader count `144,224,000`

**Note the double `full_attention` at positions 8 and 9 the parent never has, and
that the body ENDS ON `full_attention`** — which no `CL896` body in this estate
does. `ARCH.CL896X09X13` is structurally unlike anything the campaign has sealed.

**The `[mx32]` in that log line is a carried cosmetic defect and not a wrong
deployment.** The builder's log prefix has said `[mx32]` since wave 32's cut;
wave 37's log said it too and `NOTES-MX-AO` quotes it. Every receipt, schema
string and filename in the wave is wave 38's — `mx38-build-receipt.json`,
`neural-forge-matrix-wave38-standard-shape-child-v1`, `code_identity.repo_head =
4eaf5a18` — and the audit pins both receipt write sites at the wave-38 name. It
is wave 17's `[mx16]` defect, seventh issue, still cosmetic, still not fixed
because fixing it in a wave that makes no behavioural edit would be a behavioural
edit.

---

## 2. F1: the X09 ladder is not flat, it decays, and the decay is monotone

| K | child | ratio | registered | **sealed** | canonical |
|---:|---|---:|---:|---:|---:|
| 14,336 | `ARCH.V14CL896X09` | 0.4299088649 | 2 | **2** | 2 |
| 12,288 | `ARCH.V12CL896X09` *(wave 37)* | 0.4207786331 | — | **2** | 3 |
| 10,240 | `ARCH.V10CL896X09` | 0.4116484014 | 2 | **1** | 1 |
| 9,216 | `ARCH.V9CL896X09` | 0.4070832855 | 2 | **0** | 0 |

`DEEP_RECORD.md` §8.5 says the vocabulary fold does not change a body's count
across K — flat at 3 on `ARCH.W896`, 7 on `ARCH.CL1792`, 2 on `ARCH.CL896`, 0 on
`ARCH.W896CL1792`. **On `ARCH.CL896X09` it decays, one case per two rungs, with
no inversion.** Two of the four rows are wave 38's misses and they are the
result rather than noise around it.

### 2.1 The ladder loses retrieval first and uncertainty second

The two lanes the X09 family ever holds are `context-retrieval` and
`uncertainty-insufficient-data`, and they do not go together:

* at K = 14,336 and 12,288 both hold;
* at K = 10,240 **retrieval fails** — *I don't get a sholner the question* — and
  uncertainty still names all four withheld quantities;
* at K = 9,216 uncertainty fails too — *Sysed the sales, including the first, and
  the most of the time* — and names none of them.

### 2.2 The second vocabulary death in the campaign, and the first on an excised body

`ARCH.V9CL896X09` reads **0/15 sealed and 0/15 canonically at 0.4070832855x**.

The first is `ARCH.V8CL896` at K = 8,192 on the unexcised body. This one is at
**K = 9,216 — one full rung shallower** — and `ARCH.V9CL896`, *the same K on the
unexcised body*, holds **two**. So excising block 9 **moves the vocabulary death
up the K axis by at least 1,024 rows**, at a saving of 0.0302516621 of stored
ratio.

**And it is a content death, not a decode death.** Eight of its fifteen misses
are readable-and-wrong and only seven are artifacts — **the most content-weighted
zero in the deep record**, tying `ARCH.V8CL896`'s exact 7/8 profile and nothing
like `ARCH.W512`'s 15/0. It has **fewer cap-hits (4 of 16) than the child above
it that holds a case.** It is still forming sentences on register and has stopped
having anything to put in them.

### 2.3 The X09-over-X13 question is answered only where it was asked, and this note says so

The order asked whether the X09-over-X13 advantage holds across K. **It cannot be
answered from this wave's build list and wave 38 does not claim it can.** The
advantage was measured at exactly one K — `ARCH.V12CL896X09` 2 against
`ARCH.V12CL896X13` 1, at K = 12,288 — and wave 38 built **no new X13 rung**. What
wave 38 establishes is that the X09 body's own ladder decays with K, so *any*
advantage it has is rung-specific by construction. Settling the comparison needs
`V14CL896X13`, `V10CL896X13` and `V9CL896X13`: three 40-second CPU passes off
wave 35's retained `MX35-CL896X13`, and three reviews.

---

## 3. F2: the excision pair at depth, and there is no headroom

| child | ratio | params | registered | **sealed** | canonical | cap-hits |
|---|---:|---:|---:|---:|---:|---:|
| `ARCH.CL896X09X13` | 0.6278984752 | 144,224,000 | 0 | **0** | 0 | **11/16** |
| `ARCH.V12CL896X09X13` | **0.3905124499** | 89,698,048 | 0 | **0** | 0 | **14/16** |

Both hit. Two independent accounts clipped to the same number: additive
(`ARCH.CL896` 4, minus three per excision) and wave 28's V19 anchor (the 9-and-13
pair costing one case *more* than additive).

**`ARCH.V12CL896X09X13` is the deepest child this campaign has ever built** —
0.0302661832 of stored ratio below the previous floor, 39.1% of the parent — and
**fourteen of its sixteen lanes ride the cap**, the most of any child in the
estate. Its `arithmetic-multi-step` lane emits the fragment *sh* for two thousand
characters.

**The bodies share ZERO of sixteen lanes with either single-excision body**, and
the child shares zero with `ARCH.V12CL896X09` and zero with `ARCH.V12CL896X13`.
Nothing here is one decode counted twice.

### 3.1 The finding is in the body's summarisation lane, not in its count

`ARCH.CL896X09X13`'s cleanest English is:

> *The team compared three suppliers. Supplier A is cheap but not guarantee
> delivery before launch. Supplier B costs 11 percent more and passed every
> test.*

Correctly spelled, correctly numbered, every fact true — **and it is the prompt's
own premises handed back.** The lane asked for the decision and the unresolved
risk and produced neither. That is `ARCH.V8CL896`'s copy-back failure mode
appearing **0.195 of stored ratio above it**, on a body whose next-cleanest lane
is markdown bold markers and whose `creative-continuity` lane returns **the empty
string**.

---

## 4. F3: the condition failed, the arm was substituted, and the K account wins

**The order's G3 was conditional and the condition failed.** It reads: *the G10
arm on the floor, `V9CL896X09G10`, if G1's V9X09 shows any signal.*
`ARCH.V9CL896X09` reads zero. A gain arm differenced against a dead control
measures nothing, so **that arm was not built.**

It was **substituted** by `ARCH.V10CL896X09G10` — λ1.0 one rung up, on the
deepest live control this body has — and **the substitution, its reasoning and
its predicted count were written to the share at 16:52:22 and the solve's build
receipt is stamped 16:53:08.** The prediction precedes the measurement by
forty-six seconds and the share carries the proof;
`wave38/prereads/preregistration-wave38-G3-substitution.json` is the file.

**Registered at two, reads two.**

| K | body | control | λ1.0 | Δ |
|---:|---|---:|---:|---:|
| 12,288 | `CL896` | 2 | 2 | **+0** |
| 12,288 | `CL896X09` | 2 | 2 | **+0** |
| 10,240 | `CL896` | 2 | 3 | **+1** |
| 10,240 | `CL896X09` | **1** | **2** | **+1** |

**Two rungs, two bodies, the same delta at each rung.** λ1.0's gain is a property
of the **vocabulary rung** and not of the body and not of stored ratio. That is
the account wave 37 could not separate, because its only deep arm sat at a rung
where both accounts predict zero.

**And the two bodies share zero of sixteen lanes at this rung**, so the agreement
is two independent decodes agreeing rather than one recipe re-read.

### 4.1 What this arm does NOT add, and it is a stated deviation

`DEEP_RECORD.md` §8.3 exists because the solver's objective anti-correlates with
the review, and its stated purpose is *the reason the ladder runs as a ladder*.
**Wave 38 ran a single arm, not a ladder** — the order named G10 only — so §8.3's
n stays at **thirteen** and this wave contributes no row to it. The arm's
objective improved (held-out logit relSSE 0.2655 → 0.2397, 9.7%) and its review
improved, but arm-against-control is not the comparison that law is about.

---

## 5. The frontier: four rungs move, all to one ratio

### 5.1 Sealed

| rung | before | after | Δ |
|---:|---|---|---:|
| **1/15** | `ARCH.V12CL896X13` / `ARCH.V12CL896X09` 0.4207786331 | **`ARCH.V10CL896X09` + `ARCH.V10CL896X09G10` 0.4116484014** | **0.0091302317** |
| **2/15** | `ARCH.V12CL896X09` 0.4207786331 | **`ARCH.V10CL896X09G10` 0.4116484014** | **0.0091302317** |
| 3/15 | `ARCH.V9CL896G03` 0.437349 | *unchanged* | — |
| 4/15 | `ARCH.V9CL896G10` 0.437349 | *unchanged* | — |

### 5.2 Canonical

| rung | before | after | Δ |
|---:|---|---|---:|
| **1/15** | 0.4207786331 | **0.4116484014** | **0.0091302317** |
| **2/15** | 0.4207786331 | **0.4116484014** | **0.0091302317** |
| 3/15 | `ARCH.V12CL896X09` 0.4207786331 | *unchanged* | — |
| 4/15 | `ARCH.V9CL896G03` 0.437349 | *unchanged* | — |
| 5/15 | `ARCH.V9CL896G10` 0.437349 | *unchanged* | — |

**The sealed 1/15 rung is held by two independent children** at the same ratio.

### 5.3 The deep floor is now bracketed, and the bracket is exactly one vocabulary rung

**Signal exists at 0.4116484014x and does not at 0.4070832855x.** The two
children are 1,048,576 parameters apart — **1,024 vocabulary rows** — which is
**0.0045651159 of stored ratio.**

That is the *same width* as `DEEP_RECORD.md` §5's bracket between `ARCH.V8CL896`
(dead) and `ARCH.V9CL896` (two), because both are one 1,024-row step on the same
axis. **The campaign now has two brackets of identical width on two bodies**, and
the block-9 body's sits 0.0257 of stored ratio deeper. Nothing has been built
inside either.

### 5.4 The canonical lens moved nothing, for the first time since wave 33

All six children read canonically exactly what they read sealed. Wave 37 had four
of six read one above their seal; wave 36 had three of six; wave 38 has zero.

**That is not a stricter standard** — the clauses are quoted unchanged in
`wave38/canonical-reread-wave38.json`. **It is what the deep zone looks like
below 0.43x: 55 of this wave's 90 scored lanes (61%) are artifacts**, and the
canonical lens exists to forgive exactly those. It did that 55 times. What it
cannot do is manufacture a correct element out of a lane that emitted markdown
bold markers.

---

## 6. Benchmarks, and the one place the column finally agrees

Standard invocation, parent row first. **The parent row reproduces waves 32, 34,
35, 36 and 37 to every reported digit** — the campaign table now carries 25
parent rows with all 25 matching.

| model | ratio | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 1.000000 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | **81.80** |
| `ARCH.CL896X09X13` | 0.627898 | 0.3550 | 0.3500 | 0.5200 | 0.5000 | 0.0350 | 306921.64 | 2090.87 |
| `ARCH.V14CL896X09` | 0.429909 | 0.3050 | 0.3150 | 0.4950 | 0.4700 | 0.0300 | 398439.22 | **3205.91** |
| `ARCH.V10CL896X09` | 0.411648 | 0.2650 | 0.3050 | 0.5050 | 0.4750 | 0.0200 | 779799.42 | **5791.65** |
| `ARCH.V10CL896X09G10` | 0.411648 | 0.2800 | 0.3000 | 0.5100 | 0.4700 | 0.0150 | 869133.88 | 5678.69 |
| `ARCH.V9CL896X09` | 0.407083 | 0.2750 | 0.2850 | 0.4950 | 0.4800 | 0.0200 | 957858.30 | **7646.69** |
| `ARCH.V12CL896X09X13` | 0.390512 | 0.3100 | 0.3100 | 0.4600 | 0.4700 | 0.0250 | **4112020.14** | **11676.14** |

**`ARCH.V12CL896X09X13`'s lambada perplexity is 4,112,020.14 — the campaign's
first seven-digit perplexity, eight times wave 37's record — and it holds
nothing**, which is the first time in three waves the worst perplexity and the
worst review have been the same child.

**And on the vocabulary ladder the benchmark column agrees with the review, which
it has never done before on a ladder in this campaign.** Wikitext word perplexity
runs **3205.91 / 5791.65 / 7646.69** for the children that hold **2 / 1 / 0** —
monotone, no inversion, across three rungs.

**It still inverts everywhere else in the same table.** `ARCH.V10CL896X09G10`
holds **twice** its control's cases at a **worse** lambada perplexity
(869,133.88 against 779,799.42), and `ARCH.CL896X09X13` has the **best**
perplexity of all six children (306,921.64 lambada, 2,090.87 wikitext) and holds
**zero**. So the column tracks the vocabulary axis and does not track the
excision axis or the gain axis.

Read this table as a warning. Every child is at or below chance on winogrande and
piqa, lambada collapses from 0.46 to 0.015–0.035, and word perplexity rises by
one to two orders of magnitude. **Nothing in the deep zone is a usable model.**

---

## 7. Method and controls

- **Parent calibration, eleventh consecutive, exact.** Re-read blind from
  `ARCH.V9CL896X09`'s dossier — the first of the wave: 5 Useful / 7 Partial / 3
  misses = **12/15**, cap-hits 0/16, with the arithmetic Partial still boxing 3
  kits per box with 94 left over and the multi-turn miss still offering *Vegan
  Peanut Butter & Jelly Bars* to a table with two severe peanut allergies.
- **The runner control is now SEVEN WAVES AND TWENTY DOSSIERS WIDE**, the
  strongest it has ever been. `mx38_parent_identity.py` reads
  `865f006e3813712ec202cb23` on all six wave-38 dossiers, and
  `wave38/lane-byte-comparison-wave38.json` computes the same digest over the
  sealed dossiers of waves 15, 23, 34, 35, 36 and 37 as well — twenty dossiers,
  seven waves, one digest, protocol `e5e377688597`, prompt pack `0ee6bf14cef5`.
- **The byte comparison decided one call by construction and framed three
  more**, and it ran before any count was written. `ARCH.V14CL896X09`'s
  uncertainty lane is byte-identical to `ARCH.V12CL896X09`'s, so that Partial is
  wave 37's verdict on both standards. `ARCH.V9CL896X09` and `ARCH.V10CL896X09`
  share **six of sixteen** lanes across a 1,024-row K change, so the whole
  difference between one and zero lives in the ten they do not share.
  `ARCH.V10CL896X09G10` shares **zero of sixteen** with `ARCH.V10CL896G10`.
  `ARCH.CL896X09X13` shares **zero** with either single-excision body, and
  `ARCH.V12CL896X09X13` shares **zero** with either 0.4207786331x child.
- **Two calls are one lane wide and both were sealed AGAINST the child**, with
  the alternative recorded: `ARCH.CL896X09X13`'s summarisation lane (accurate
  premises, neither the decision nor the risk) and `ARCH.V12CL896X09X13`'s
  retrieval lane (7:30 a.m. carried with the correct meridiem, which no other
  X09-family child manages, and none of the three asks answered). A reader who
  credits either reads that child at one.
- **Seal verification**: 988 ledger rows, 984 resolve to a receipt on disk, 984
  verify, **0 fail**, 4 unresolved — the same historical four. **34 seal roots**
  audited against disk, 0 undeclared.
- **`sealq.sh` ran once**, and all six sealed on the first call.
- **Pre-registration hit rate: 4 of 6** (`CL896X09X13` 0 ✓, `V14CL896X09` 2 ✓,
  `V12CL896X09X13` 0 ✓, `V10CL896X09G10` 2 ✓; `V10CL896X09` predicted 2 read 1,
  `V9CL896X09` predicted 2 read 0). **Both misses are against the child and both
  are on the flatness law** — which is the law this wave was testing, so the
  misses are the measurement and not noise around it. Note that two of the four
  hits were zeros predicted on a body already at 1/15, which is a cheap kind of
  hit and is counted as one anyway.

---

## 8. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `f822cd03` | cut wave 38's tree, make no behavioural edit for the second wave running, **and append the sealer's four accept strings in the same commit** |
| `4eaf5a18` | pre-register wave 38 |
| `3cc92960` | the six seals and the G3 substitution's own pre-registration |
| `4036759b` | the measurement, the retention and the docs |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `4eaf5a18` | 6937 | 0 | 0 | 0 | 0 |
| `3cc92960` | 6938 | 0 | 0 | 0 | 0 |

**Two deployments, and the first one carries the pre-registration.** Wave 37 cut
its GPU deployment at the tree and skipped the pre-registration commit because
nothing in the run path opens that JSON. Wave 38 cut ONE deployment at the
pre-registration commit instead, for the same price, so **every wave-38 cell
receipt names a commit that contains both the code that ran and the predictions
it was run against.** The second is the seals.

**The accept list cost nothing, for the third consecutive wave.** Waves 34 and 35
each paid an entire extra deployment to meet
`experiments/seal_matrix_architecture_child.py`'s append-only accept list at
their first seal, after the GPU work was done. Waves 36, 37 and 38 appended at
the tree cut. Three waves is where that stops being a coincidence.

**The cutter refused once and was right.** `sealq.sh` arrived VERBATIM carrying
wave 37's marker and wave 37's six seal calls with every count intact — sixth
consecutive cut where that protection has had real work to do — and
`cut_wave38.py --audit` refused on the marker before anything in the tree ran.

**Two pre-declared exemptions, and unlike wave 37's, they were used.** Wave 37
pre-declared the `retain`/`sealq` wave-reference exemptions and used neither.
Five of wave 38's six children are one CPU pass off wave 37's **retained**
`MX37-CL896X09`, so the provenance rule requires that name in prose; the seal
file carries it three times against a declared bound of six. **The bound is an
upper bound and the cutter now supports both comparisons**, because an exact pin
that must be guessed before the verdicts exist is a pin that gets waived the
moment it is wrong. The header exemption, whose count *is* knowable, stays exact.

**Ledger.** Backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave38-20260828T231539Z` and
`cmp`-verified byte-identical before the merge. One-pass merge, 976 → 988 rows,
12 appended, 0 superseded. Byte-prefix verified: `head -c <backup size>
ledger.jsonl | sha256sum` returns `a94caa551d9a7a2da…`, the pre-wave digest,
exactly.

**Every artifact this wave wrote is parsed in the command that wrote it.**
`NOTES-MX-AO` §9 told the next wave to make that the default. `cut_wave38.py`
re-reads its own manifest; `mx38_retain.py` was given a `json.loads` read-back
that refuses if the manifest does not read as written; the lane byte comparison
re-parses its own output; and the child index — the file wave 37 corrupted by
redirection — was written and parsed in one invocation and came back clean.

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used throughout,
with the in-process SM86 preflight on every driver. Both were carrying an
idle-holding H3 ComfyUI container at ~280 MiB each and were **not restarted**,
for wave 34's stated reason. The four P40s were never touched.

**The Quadro RTX 8000 was never touched by this wave and was not idle**, exactly
as at the end of wave 37: it read **45,212 MiB at this wave's GPU preflight and
45,212 MiB throughout**. Wave 38 issued no command against index 6. This
paragraph is here so a later wave reading the GPU numbers does not attribute that
allocation to the matrix campaign — and note that it is now a *standing*
allocation across two waves rather than something that appeared during one.

**The Codex lane was not touched.**

---

## 9. What this wave changed, and what it did not

Changed: `reports/matrix-campaign-20260810/run/mx38/` (the cut, four hand edits,
the README, the six seals in `sealq.sh`);
`experiments/seal_matrix_architecture_child.py` (four appended accept strings, at
the cut); `wave38/` (the artifacts and dossiers); `RESULTS_MATRIX_2026-08.md`,
`ACTUAL_MATRIX_2026-08-28.md`, `CAMPAIGN_SCORECARD.md` and `DEEP_RECORD.md`, each
backed up under `backup/` with its pre-edit sha256 recorded in the commit;
`ledger.jsonl`.

Not changed: **any builder or solver behaviour anywhere**, the frozen fifteen
prompt pack, the frozen review runner, both extension banks and their seal files,
the paraphrase bank, `mx38_penaltydecode.py`, every earlier wave's sealed
receipts, verdicts and dossiers, and the Codex lane.

---

## 10. What the next wave should know

1. **The deep floor is bracketed to one vocabulary rung on two bodies now.**
   `ARCH.V10CL896X09` holds at 0.4116484014x and `ARCH.V9CL896X09` is dead at
   0.4070832855x — 1,024 rows and 0.0045651159 of stored ratio apart, the same
   width as the `V8CL896`/`V9CL896` bracket. **Nothing has been built inside
   either bracket** and both are one 40-second CPU pass to halve.
2. **The X09-over-X13 comparison is still one data point.** Wave 38 built no X13
   rung. `V14CL896X13`, `V10CL896X13` and `V9CL896X13` are three CPU passes off
   wave 35's retained `MX35-CL896X13` and would turn a single-K observation into
   a ladder-against-ladder result. That is the cheapest open question in the
   estate.
3. **The excision axis is exhausted on this body.** Both halves of the 9+13 pair
   read zero, and the deeper one rides the cap on fourteen of sixteen lanes.
   A third excision is not worth a wave.
4. **λ1.0 is a function of the vocabulary rung.** +0 at K = 12,288 on both
   bodies, +1 at K = 10,240 on both. The untested rungs on the excised body are
   K = 14,336 (where λ1.0 buys +0 on `CL896`) and K = 16,384 (+1), and the K
   account now makes a falsifiable prediction at each.
5. **§8.3's n is still thirteen and wave 38 did not raise it.** The order named a
   single λ arm, so no ladder ran. If the objective's anti-correlation matters,
   a wave has to spend two solves, not one.
6. **Assert code lines at their INDENTED form.** Wave 38's audit failed on its
   own docstring quoting the three lines it pins, and a pin of 2 would have
   passed with the code deleted. The indentation is the difference between an
   assertion about the builder and an assertion about the builder's comments.
7. **Append the sealer's four accept strings in the commit that cuts the tree.**
   Three waves, zero deployments paid.
8. **When a conditional cell's condition fails, substitute and pre-register — do
   not skip and do not proceed.** Wave 38's G3 control read zero; the arm was
   moved one rung up, the substitution was written to the share forty-six seconds
   before the solve's build receipt, and the wave's sharpest result came out of
   the substitute rather than out of the cell that was ordered.
