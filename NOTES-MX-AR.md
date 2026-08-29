# NOTES-MX-AR — wave 40: the floor moves on a block choice, the mlp cliff is 128 channels wide, and C1 is closed by a child

**2026-08-29.** Branch `codex/lrc-blank-compression`. Eleven children built,
sealed and read; ledger 992 → **1014** rows,
`c0aa462fb9c2f15c1029a78d3cfcc5eb329ee66769f1b9086fc9f483d5671c1a`, byte-prefix
verified against the pre-wave backup on the share. Two deployments, and the
accept list cost neither — fifth consecutive wave.

**Eight of eleven pre-registrations hit. All three misses are against the child
and all three are on the mlp axis, which is the axis this wave was measuring.**

**The sealed and canonical 1/15 rungs moved to 0.4070832855x.** §3.

---

## 1. Wave 40 added a build module, edited none, and RAN every path wave 39 only asserted

Waves 38 and 39 made no behavioural edit. Wave 40 makes no *edit* either — every
carried module keeps the bytes wave 39 shipped — but it **adds** `mx40_vocabtie.py`
and `tieq.sh`, the first new build module this campaign has authored since wave
25's vocabulary gather, and it is the first wave since 38 to execute
`mx*_build.py` at all.

Wave 39 asserted the builder's excision and width paths on a file it never
opened. **Wave 40's build list is `--ff 768`, `--ff 768 --excise 9` and
`--ff 1728`**, so wave 36's composed width × MLP site, wave 37's live-mask repair
and wave 38's descending deletion order and both module re-index assignments are
all live code. The live-mask log line came back exactly as registered:

> `[mx40] excised blocks [9]; child depth 13; config.layer_types re-derived (13 entries) so the live mask follows the live block`

And **wave 39's log-prefix repair is now under four seals.** That wave fixed
`log()`'s `[mx32]` prefix on the explicit ground that it never ran the file, so
the repair could not move a number. It reads `[mx40]` in every wave-40 build log.

### 1.1 The cut carried two of wave 39's seal calls and the audit did not notice

`run/mx40/sealq.sh` arrived VERBATIM with wave 39's marker **and wave 39's two
seal calls, counts intact**. The marker was rewritten by hand; the calls were
not. `cut_wave40.py --audit` passed, **because it asserted the marker and not the
call count** — the marker is one line at the top and the calls are anywhere below
it.

They would have errored rather than mis-sealed, since no wave-40 receipt carries
those tags. **"It errors" is not the protection.** A peer wave found them, not
this file. Both are deleted and `audit()` now pins `^seal <TAG>` at **zero at cut
time**, which is wave 39's own recommendation and the assertion that would have
caught this without a peer. Committed as `889f1ec5`.

---

## 2. THE MEASUREMENT THAT DECIDED THE TIE BUILDER'S SHAPE, TAKEN BEFORE IT WAS WRITTEN

C1 as specified in `FOLD_CLASS_CRITIQUE` is a **many-to-one** vocabulary: tail
tokens keep their merges and their positions and address one shared row. That
form was tried first, on this parent, on this stack. **It is not a constructible
stock child.**

```
Exception: Error while attempting to unpickle Tokenizer:
Token `Ã¹` out of vocabulary at line 1 column 2145524
```

**and it names a different token on different runs.** Five loads of the identical
file named three different tokens at four different column offsets. That is the
Rust `HashMap` id→token inverse being built in a nondeterministic order,
surfacing as a nondeterministic refusal. Had it loaded, `decode` of a tied id
would not have been a function across processes, and a frozen review of a child
whose decode depends on the process is not a frozen review.

**So the injective realisation is what was built, and it is the better controlled
experiment anyway** — the many-to-one form moves the tokenizer and the weights at
once and a difference could then be either.

---

## 3. F1: the floor moves 0.0045651159, and it moves on WHICH BLOCK

| rung | before | after | Δ |
|---:|---|---|---:|
| **sealed 1/15** | `ARCH.V10CL896X09` / `…G10` / `ARCH.V10CL896X13` 0.4116484014 | **`ARCH.V9CL896X13` 0.4070832855** | **0.0045651159** |
| **canonical 1/15** | 0.4116484014 | **`ARCH.V9CL896X13` 0.4070832855** | **0.0045651159** |
| sealed 2/15 | `ARCH.V10CL896X09G10` 0.4116484014 | *unchanged* | — |

**This is the first floor movement in the campaign that comes from a block choice
rather than a deeper rung.** `ARCH.V9CL896X13` and `ARCH.V9CL896X09` are the same
K on the same-sized body at **93,504,256 parameters each, identical to the byte**,
because blocks 9 and 13 are both conv. The block-13 child holds one; the block-9
child, sealed at wave 38, holds **zero**.

**`DEEP_RECORD` §8.4 — "X13 costs three, and so does X09" — does not survive at
0.407x.** Both excisions cost three at `CL896`; that is a statement about the
shallow end. The X09 ladder decays one case per two rungs and dies at 9,216; the
X13 ladder is flat at one through **five consecutive rungs** and is still alive
there. **A wave that wants to go deeper should excise block 13, not block 9**, and
that is a reversal of the family this campaign has spent four waves on.

### 3.1 The cell was not in the main pre-registration and it says why

Wave 40 registered at commit `58ac7984`, when wave 39's two X13 rungs were built
and ungraded on the share and wave 40's declared intent was to *read* them. Wave
39 then sealed, graded and merged them itself while wave 40 was building. That
made K = 9,216 on the same body the cheapest open cell in the estate, and it did
not exist as an option when wave 40 registered.

It was registered **in its own file on the share**,
`wave40/prereads/preregistration-wave40-V9CL896X13-addition.json`, at
**02:34:51.775Z**; its build receipt is stamped **02:35:19.827Z**. The prediction
precedes the measurement by twenty-eight seconds and the share carries the proof.
Wave 38's G3-substitution procedure, applied to an addition.

**Two accounts disagreed by exactly one case and flatness won.** Block-equivalence
predicted zero; wave 39's flat-at-one predicted one. Registered at one, reads one.

### 3.2 The honest qualifier, and it is in the seal

`ARCH.V9CL896X13`'s held summarisation lane is **byte-identical to
`ARCH.V10CL896X13`'s**. Dropping a further 1,024 vocabulary rows did not change
that lane by one byte, so **this is wave 39's decode read at a smaller size, not a
second independent witness.** What is new is the size.

---

## 4. F2: the mlp cliff sits ABOVE 768, and the big transition is 64 channels wide

| `--ff` | ratio | sealed | |
|---:|---:|---:|---|
| 640 | 0.6404971251 | **0** | wave 15 |
| **768** | **0.6644639834** | **1** | **wave 40, registered 2** |
| 896 | 0.6884308417 | **4** | wave 15 |
| 1280–1664 | 0.76–0.83 | 4 / 3 / 4 / 5 | waves 30–31 |
| **1728** | **0.8442154209** | **4** | **wave 40, registered 6** |
| 1792 | 0.8561988500 | **7** | wave 30 |

**`ARCH.CL768` reads ONE, not the two a linear reading of the 640→896 bracket
predicts.** The four-case drop is therefore **768→896: 128 channels, three
cases** — the cliff is in the upper half of the bracket, not at its midpoint.
`CG1-b`'s hope that a `CL768` body would carry the floor's ladder cheaply is
falsified at the body.

**Five of `ARCH.CL768`'s sixteen lanes return the prompt itself**, at
**0.664464x** — higher than either child previously known to do that.

**And from 896 to 1728 the axis is flat within ±1**, so the campaign's largest
single mlp transition is **the last 64 channels, 1728→1792, worth three cases.**
`ARCH.CL1664` at 5 and `ARCH.CL1728` at 4 is a one-case inversion inside that
noise band, recorded rather than smoothed. **The same edge appears on the V19
base**: `V19CL1664` 3, `V19CL1728` **4**, `V19CL1792` 7.

---

## 5. F3: the cheap axis pair reaches the floor's depth band and the vocabulary costs ZERO

| child | K | ratio | sealed |
|---|---:|---:|---:|
| `ARCH.CL768` | — | 0.6644639834 | **1** |
| `ARCH.V12CL768` | 12,288 | 0.4270779580 | **1** |
| `ARCH.V10CL768` | 10,240 | 0.4179477263 | **1** |
| `ARCH.V9CL768` | 9,216 | **0.4133826104** | **1** |

**Flat at one across 0.2511 of stored ratio.** `DEEP_RECORD` §8.5's flatness law
holds on a **fifth** body, and this is the deepest flat ladder in the campaign.
The vocabulary axis costs this body **zero cases** — wave 30's M2 result
reproduced near the instrument's floor. `ARCH.CL1728`/`ARCH.V19CL1728` reproduce
it again at 4 and 4 across 0.2054.

**`ARCH.V9CL768` reaches 0.4133826104x — the floor's depth band — with NO
EXCISION AT ALL**, on mlp × vocabulary, the pair wave 30's allocator prices at
**1.17 cases** against the **4.44** every child at that depth has spent. **It does
not take the rung**: 0.4133826104x is 0.0017342090 *above* the sealed 1/15 ratio.
A tie is reported as a tie.

**The qualifier is in the seals.** `V9CL768`'s held lane is byte-identical to
`V10CL768`'s and *not* to `V12CL768`'s, so the flat ladder is **two independent
decodes, not three.**

### 5.1 The deepest child ever built, and the inventory's arithmetic

`ARCH.V10CL768X09` at **0.3893934615x** — 89,441,024 parameters, 38.9% of the
parent — is the deepest child this campaign has built, and it holds nothing, as
registered. Two of its sixteen lanes return the **empty string**, which no child
in the estate had done twice before.

**The published inventory's figure for this cell is wrong and it was corrected
before the build.** `UNBUILT_INVENTORY_2026-08-28.md` §1.5 states 0.387682x; it
priced the block-9 excision at 6,951,936, which is that block's cost on an
`ff=896` body. Block 9 is a conv block and its MLP scales with `--ff`, so at
`ff=768` it costs 6,558,720. The corrected 89,441,024 was pre-registered and the
build returned 89,441,024. **The claim survives** — 0.3893934615x is still below
`ARCH.V12CL896X09X13`'s 0.3905124499x, by 0.0011189884 rather than 0.0028.

The decomposition was validated against two sealed counts to the byte first:
43,008 per channel over 14 sites reproduces `CL1792` from `CL896`, and
in_proj + out_proj + conv + two norms + 3·1024·ff reproduces `CL896` − `CL896X09`.

---

## 6. F4: C1 is closed by a built child, and row tying moves ONE LANE OF SIXTEEN

`ARCH.T9CL896X09` is `ARCH.V9CL896X09` with **512 of its 9,216 rows replaced by
closed-form weighted centroids** carrying all **55,186** dropped tokens. Same
body, same kept set, same tokenizer bytes, **same parameter count to the byte**
(asserted by the builder before it wrote anything), exactly 512 rows different.

**The two children share FIFTEEN OF SIXTEEN LANES BYTE FOR BYTE**, and the lane
that differs holds nothing. Both read 0/15. Its uncertainty lane is the identical
string wave 38 quoted as its twin's failure — *Sysed the sales, including the
first, and the most of the time* — and its retrieval lane is the identical *I
don't get a shogike in the question*.

**The mechanism is visible in which rows were spent.** The 512 tie rows are the
lowest-ranked non-mandatory rows in the budget, so the tail's folded content sits
in rows the decode does not visit. A tie spending *high*-ranked rows would move
more and would also destroy the rows that matter — the trade the class never had
a child to price.

**Wave 14's census is confirmed, for the first time in a currency that ranks.**
The census compared tying against **verbatim** at median cosine 0.2977; this child
compares it against **deletion** at median cosine 0.1816 (pre-registered to read
lower, because it measures against 512 tie rows rather than the whole kept set).
The comparison the class was closed on had never been run; now it has, and the
kill holds.

---

## 7. Benchmarks, and the sharpest inversion the column has produced

Standard invocation, parent FIRST. **The parent row reproduces waves 32, 34, 35,
36, 37, 38 and 39 to every reported digit** — the campaign table now carries **27
parent rows with all 27 matching.**

| model | ratio | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 1.000000 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | **81.80** |
| `ARCH.CL1728` | 0.844215 | 0.3900 | 0.4050 | 0.5550 | 0.5450 | 0.2000 | 208.34 | **194.12** |
| `ARCH.V19CL1728` | 0.638785 | 0.3500 | 0.3850 | 0.5750 | **0.5700** | 0.1850 | 374.56 | 446.05 |
| `ARCH.CL768` | 0.664464 | 0.3650 | 0.3550 | 0.5750 | 0.5000 | 0.0600 | 5115.34 | 425.43 |
| `ARCH.CL768X09` | 0.635910 | 0.3600 | 0.3800 | 0.5450 | 0.5250 | 0.0400 | 51776.62 | 885.13 |
| `ARCH.V9CL768` | 0.413383 | 0.3000 | 0.3300 | 0.5400 | 0.4850 | 0.0750 | 10945.00 | 2489.05 |
| `ARCH.T9CL896X09` | 0.407083 | 0.2750 | 0.2900 | 0.5000 | 0.4750 | 0.0200 | **1488667.53** | 7456.04 |
| `ARCH.V9CL896X13` | 0.407083 | 0.2700 | 0.3500 | 0.4800 | 0.4850 | 0.0950 | 68004.41 | **11321.01** |
| `ARCH.V10CL768X09` | 0.389393 | 0.3000 | 0.2950 | 0.5050 | 0.5200 | 0.0300 | 1280073.39 | 6615.42 |

**The two children at 0.4070832855x invert the column completely.**
`ARCH.V9CL896X13` has the **worst wikitext perplexity in the wave** — 11,321.01,
worse than the deepest child ever built — and it is the only child at that ratio
that **holds a case.** `ARCH.T9CL896X09` has a wikitext perplexity 34% *better*
and a lambada perplexity 22× *worse*, and holds nothing. At an identical
parameter count, on the same axis, the column disagrees with the review in both
directions at once.

**`ARCH.CL768` versus `ARCH.CL1728` is the other reading worth having.** CL768 is
0.18 of stored ratio cheaper and its wikitext perplexity is 425 against 194 —
about double — while its review is 1 against 4. **On the mlp axis the column
tracks; on the vocabulary and block axes it does not.**

Read the table as a warning, as every wave since 32 has. Nothing here is a usable
model.

---

## 8. Method and controls

- **Parent calibration, thirteenth consecutive, exact.** Re-read blind from
  `ARCH.CL768`'s dossier — the first of the wave — before any child was opened.
  Both documented anchors reproduce: the arithmetic Partial boxing **3 kits per
  box with 94 left over**, and the multi-turn miss offering *Vegan Peanut Butter
  & Jelly Bars* to a table with two severe peanut allergies. **cap-hits 0/16.**
- **The runner control is EIGHT WAVES WIDE.** `mx40_parent_identity.py` reads
  `865f006e3813712ec202cb23`, protocol `e5e377688597`, prompt pack `0ee6bf14cef5`
  on **all eleven** wave-40 dossiers.
- **The lane byte comparison ran before any count was written** and it decided
  four claims by construction rather than by assertion:
  `T9CL896X09`/`V9CL896X09` **15 of 16**; `V9CL896X13`/`V10CL896X13` **6 of 16
  including the held lane**; `V9CL768`/`V10CL768` **8 of 16 including the held
  lane**; `V12CL768`/`V9CL768` **4 of 16 excluding it**. **80 of the 91 pairs
  share nothing.** `wave40/lane-byte-comparison-wave40.json`.
- **Four calls are one lane wide and all four are sealed AGAINST the child**,
  with the alternative recorded: `CL768`'s debugger (`values[start - 1:]`, the
  correct slice shifted by one), `V9CL768`'s retrieval (the source NOTE returned
  verbatim, containing all three answers and answering none), `V12CL768`'s email
  (two of three named items, correctly requested) and `CL1728`'s retrieval
  (all three asks answered under a contradictory opening day — sealed Partial,
  not Useful).
- **Seal verification**: 1014 ledger rows, 1010 resolve to a receipt on disk,
  1010 verify, **0 fail**, 4 unresolved — the same historical four. **36 seal
  roots** audited against disk, 0 undeclared.
- **`sealq.sh` ran once** and all eleven sealed on the first call.
- **Pre-registration hit rate: 8 of 11.** Hits: `CL768X09` 0, `V12CL768` 1,
  `V10CL768` 1, `V9CL768` 1, `V12CL768X09` 0, `V10CL768X09` 0, `T9CL896X09` 0,
  `V9CL896X13` 1. Misses: `CL768` 2→**1**, `CL1728` 6→**4**, `V19CL1728` 5→**4**.
  **All three misses are on the mlp axis and all three are against the child** —
  which is the axis this wave existed to bracket, so the misses are the
  measurement.
- **Both conditional cells were correctly NOT built.** `ARCH.CL736`'s registered
  gate was "only if `CL768` reads four or more". `CL768` reads one. The gate was
  not met, the cell was not built, and no cell was substituted for it — that is
  the pre-registration working rather than being waived.

---

## 9. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `a8df1922` | cut wave 40's tree, author `mx40_vocabtie.py`, **append the sealer's five accept strings in the same commit** |
| `58ac7984` | pre-register wave 40 and correct the inventory's CG1-b arithmetic |
| `889f1ec5` | delete two wave-39 seal calls the cut carried, pin the seal-call count at zero at cut time |
| `bc732a98` | the eleven seals and the V9CL896X13 addition's own pre-registration |
| *(this)* | the measurement, the tables and the docs |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `58ac7984` | 7146 | 0 | 0 | 0 | 0 |
| `bc732a98` | 7193 | 0 | 0 | 0 | 0 |

**The accept list cost nothing, fifth consecutive wave** — and it took **five**
strings rather than four, the fifth being
`neural-forge-matrix-wave40-tied-vocabulary-child-v1`, the first genuinely new
build-receipt schema since wave 25.

**Ledger.** Backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave40-20260829T025401Z` and
`cmp`-verified byte-identical before the merge. **One-pass merge**, all 22 shards
in a single invocation, 992 → **1014** rows. Byte-prefix verified:
`head -c <backup size> ledger.jsonl | sha256sum` returns `dfe27e468f09fc19…`, the
pre-wave digest, exactly. **Mirror copy-back done**, and the mirror write asserted
the merged file is a byte-prefix extension of what it replaced *before* writing.

**Two waves ran on this branch at once and the shared-path list was respected.**
Wave 39 landed its seals, its four ledger rows and its refresh of all four report
documents while wave 40 was building. Wave 40's ledger work started from **992
rows / `dfe27e46…`**, re-verified on the share rather than assumed, and its
document edits are built on **wave 39's committed versions**, each backed up
under `backup/` with its pre-edit sha256 recorded in the commit.

**I reproduced wave 37's stdout-redirect corruption while generating the child
index** — `mx40_child_index.py > $W/wave40-child-index.json` wrote a file that
would not parse — and caught it because `NOTES-MX-AO` §9's rule made me parse it
in the same invocation. The generator prints to stdout and takes no `--out`;
that is a defect worth fixing in the next cut, and the rule caught it in seconds.

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used throughout,
with the in-process SM86 preflight on every driver. Both carried an idle-holding
H3 ComfyUI container at ~280 MiB each and were **not restarted**, for wave 34's
stated reason. The four P40s were never touched.

**The Quadro RTX 8000's standing allocation is gone and this wave did not release
it.** Waves 37, 38 and 39 recorded 45,212 MiB on index 6 as belonging to
something outside this campaign. At wave 40's preflight it read **330 MiB** and it
read 330 MiB throughout. Wave 40 issued no command against index 6.

**Disk**: `/mnt/ai_data` 4.5T, 1.7T available. **The Codex lane was not touched.**

---

## 10. What this wave did NOT build, and why

Three ordered items were declared unbuilt **in the pre-registration, before
anything ran**, and the reasons are cost corrections to the inventory rather than
verdicts on the classes:

1. **K4, score-bilinear joint q/k.** Its exact-4x storage claim cannot be
   realised in a stock checkpoint on this parent: contracting `head_dim` for q
   and k alone is not a stock config axis when v shares it, and zero-padding is
   on the kill list. It is therefore a research probe with a materialised dense
   export whose storage claim is **arithmetic rather than a loader count** — and
   `PHASE1` §6 exists because arithmetic storage claims are what "never priced
   behaviourally" means. **It remains the strongest unspent lead and it is now
   twenty-six waves old.**
2. **K3(c), Monarch's first cell.** Same shape of problem. A Monarch
   factorisation is not expressible in a stock `Lfm2ForCausalLM` config, so the
   child is either a custom-module probe with no `ARCH` id and no loader ratio —
   which is where A8 F-LANDMARK has sat for eighteen waves for want of a loader —
   or a materialised dense export with an arithmetic storage claim.
3. **K3(b), the ORTHO-PAIR soft arm.** Its base `ARCH.X07KV6JQC` is a
   wave-17-to-22 child and **is not on the share**. The inventory prices this
   class at "1 build + 1 review, zero storage delta"; that prices the *solve* and
   not the reconstruction of the base, which needs a body build, a conditioned
   capture and a solve first. **Its registration still says build regardless of
   census and it still stands.**

**The honest summary of the omissions: two of the three need a storage-claim
ruling this wave had no standing to make, and the third needs a body the estate
deleted.** None of them is closed.

---

## 11. What the next wave should know

1. **EXCISE BLOCK 13, NOT BLOCK 9.** At an identical 93,504,256 parameters the
   block-13 body holds a case and the block-9 body is dead. §8.4's
   block-equivalence is a shallow-end law. Four waves of deep work went into the
   X09 family; the X13 family is the one that survives to the floor.
2. **The next floor cell is `V8CL896X13`, K = 8,192 on the block-13 body.** One
   CPU pass, ~0.402x. The X13 ladder is now flat at one over **five** rungs and
   the X09 ladder died two rungs above this one. Flatness predicts one and the
   census floor predicts zero, which is a real disagreement and the cheapest one
   in the estate.
3. **The mlp axis is bracketed to 64 channels at the top and 128 at the bottom.**
   `CL1760` splits a three-case transition; `CL832` splits the other. Both are
   ~450 s of GPU. The 1728 rung is the better buy: it is the campaign's largest
   single mlp transition and it sits on a body whose vocabulary axis is free.
4. **The canonical lens is bracketed to 0.0092 of stored ratio.** It moves at
   0.4271x and does not at 0.4179x. One rung inside that bracket settles §8.8.
5. **The prompt-returned-verbatim artifact is a fourth species and it needs a
   name in the taxonomy.** It is not a loop, a cap-ride or an empty string; it
   terminates normally in clean English. It is charged under both standards,
   which is *why* the canonical column goes flat, and `ARCH.CL768` does it on
   five lanes at 0.664464x — higher than any child previously known to.
6. **Fix `mx40_child_index.py` to take `--out`.** It prints to stdout, so the
   obvious invocation reproduces wave 35's corruption bug. `NOTES-MX-AO` §9's
   parse-what-you-write rule caught it here in seconds; the next wave should not
   need the rule for this.
7. **A many-to-one token map cannot be a stock child on this parent.** Measured,
   not predicted. Any future vocabulary class needing one — and #12 V-ROUTE and
   #20 clustered LM head both might — must budget its own runtime and will land
   in `PROBE.*` with no loader ratio.
8. **Two waves on one branch worked, and it worked because both of them
   checked.** Wave 39 named the five contended paths; wave 40 re-verified the
   ledger on the share before backing it up, based its document edits on wave
   39's committed versions, and had its own inherited-seal-call defect found by
   the peer. **Neither of those was luck, and neither is a substitute for the
   ownership agreement `NOTES-MX-AQ` §11.9 asked for.**
