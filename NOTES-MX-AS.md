# NOTES-MX-AS — wave 41: the floor moves again on block 13, row identity is inert at fixed K, and the campaign's alive-at-the-floor lane is a copy of the prompt

**2026-08-29.** Branch `codex/lrc-blank-compression`. Ten children built, sealed
and read; ledger 1014 → **1034** rows,
`5b91a21110a7579a56d62aa569b1b37b7cdf478448165a2f579a15288724ac26`, byte-prefix
verified against the pre-wave backup on the share. Two deployments, and the
accept list cost neither — sixth consecutive wave.

**Seven of ten pre-registrations hit.** All three misses are one case wide and
all three are on the zero-storage gain arms and the leverage arm.

**The SEALED 1/15 rung moved to 0.4025181696x. The CANONICAL rung did not
move at all** — under its own clause it reads **0/15 on all ten children**,
and the two lenses have separated completely for the first time. §4, §4.1.

**And the thing this wave actually found is in §3, not §4.**

---

## 1. Two measurements were taken BEFORE the build list was fixed, and both changed it

Wave 40 decided its tie builder's shape by a measurement taken before the
builder was written. Wave 41 did the same thing twice, and in both cases the
answer was that a ranked slate item could not be built as specified.

### 1.1 ALIAS is NOT CONSTRUCTIBLE as a stock child on this parent

ALIAS is slate item #1 and its entire distinguishing variable is a **many-to-one
encoder**: dropped types addressing a surviving row's id, so that token count and
segmentation are preserved and only type identity degrades. Wave 40 measured
that the many-to-one form of C1 does not load. Wave 41 re-took that measurement
on **ALIAS's own construction**, on the real floor child
`MX40-V9CL896X13`, four constructions, five loads each:

| construction | result |
|---|---|
| **A** control, the sealed injective child | 5 loads, **1 digest, 0 errors** — the probe is sound |
| **B** ONE alias in `model.vocab` | 5 loads, **4 errors and 1 SUCCESS** |
| **C** 256 aliases in `model.vocab` | 5 loads, **5 errors**, two tokens named at four column offsets |
| **D** alias in `added_tokens` with a duplicate id | 5 loads, **1 digest, 0 errors — and refuted anyway** |

**B's single success is the worse half of B's result.** Its id→token digest is
*byte-identical to the control's* — the alias was silently discarded. Wave 40
recorded "it does not load"; the truth is that it loads about a fifth of the
time and **lies when it does**. A wave that built this and got lucky would have
shipped a child indistinguishable from its deletion twin while claiming an alias.

**D fails differently and that is why it was worth running.** `model.vocab`
stays injective, so the merge-closure refusal cannot fire — and it does not.
But `tokenizers` **ignores the requested duplicate id and re-ids the added
entries at the end of the table**: `vocab_size` 9,214 with `len(tokenizer)`
9,217 and 9,472. The alias is silently converted into a vocabulary *extension*,
which is not a fold, and the probe text ` understanding the document` encodes to
id **9,420 — outside the child's own embedding table**.

So wave 40's §11.7 hardens **from one construction to three**, and it gains a
named silent-failure mode. ALIAS is **blocked at construction, not closed**, and
joins K4 and K3(c) in the PROBE queue. **The slate's ALIAS export ruling is moot
as written** — there is no id-remapped-tokenizer child for Cyril to rule on. §11.

### 1.2 ROWSET's Arms A and B are blocked on corpus, and every candidate fails in its own words

The slate specifies Arms A and B as re-rankings by a second census over held-out
text "sealed by hash, disjoint from the frozen 15 AND the 30-prompt extension
bank". Every corpus this estate holds was checked against that, and every one
fails:

| corpus | why it cannot be the second census |
|---|---|
| `lfm2_paraphrase_triplets_v1.json` | its own `purpose`: "three restatements of each sealed case's ask". A paraphrase **of** the frozen 15 |
| `lfm2_causal_support_future_manual_review_v1.json` | its own `purpose`: "never support selection, fitting, training, or an automatic numeric gate" — and 2,069 characters |
| `lfm2_extension_bank_prompts_v1/v2.json` | the rider bank; the slate voids any rider whose scoring text overlaps it |
| the 256-row sealed corpus / 24 captured batches | this **is** the evidence census that supplies the primary key |
| wikitext / lambada | the benchmark column |

Registered as a **substitution with its cause before anything was built**: wave
41 builds the three arms that need no corpus — **P** (the sealed tie-break run
backwards), **R** (a seeded permutation, seed 41), **C** (the slate's leverage
arm verbatim). They *bracket* the ordering space instead of sampling two points
of it. Arms A and B stay open.

---

## 2. The tree, and the ninth exercise of a protection that finally had work to do from the start

`run/mx41/sealq.sh` arrived VERBATIM carrying wave 40's marker **and wave 40's
ELEVEN seal calls with every count intact**. Wave 40's own cut had carried two of
wave 39's and the audit said nothing, because the marker is one line at the top
and the calls are anywhere below it; a peer wave found them. Wave 40 then added
the pin that fixes it — `audit()` counts `^seal <TAG>` and requires **zero at cut
time**. **Wave 41 is the first cut where that pin had work to do from the
start**, and it refused until all eleven were deleted by hand.

**Wave 41 adds one build module and edits exactly one carried module.**
`mx41_vocabrowset.py` and `rowsetq.sh` are new. The one edit is
`mx41_child_index.py`, which is wave 40's §11.6 recommendation: it gains `--out`
and parses what it writes **inside the generator**, so the parse-what-you-write
rule is no longer the only thing between the estate and a corrupt index. The
stdout path every earlier wave used is kept and pinned. It ran clean this wave.

---

## 3. THE READING THIS WAVE HAD TO MAKE, AND IT IS ABOUT THE CONTROL

**`ARCH.V9CL896X13` is the campaign's sealed 1/15 floor child. Its one held case
is summarisation. Its summarisation lane is the source paragraph returned
verbatim** — six sentences, no compression, one word mutated (`procurement` →
`proceeding`). Wave 40 sealed that lane Partial and did **not** charge it.

This was found by checking the control before grading, not by grading first.
The first pass over this wave's children read six of them as 0/15 on the
grounds that a copied source is not a summary; the control check inverted six
counts, and the inversion is recorded here rather than smoothed.

Wave 41 keeps wave 40's **case** reading and **charges** the lane under ADDENDUM
9's second, independent reading. The criterion is stated once in `sealq.sh` and
applied to all ten children:

> `summarization-priorities` is **held (Partial)** if the response carries BOTH
> the decision (Supplier B was selected) and the unresolved risk (factory
> expansion may slip by one week) intact. It is a **miss** if either is absent
> or corrupted. Whether the response summarises or **copies** is charged under
> the degeneracy reading and does not decide the case.

**That criterion is not invented here.** It is the only reading consistent with
both of wave 40's own sealed precedents on this artifact: `V9CL896X13`'s copied
summary sealed **held**, and `V9CL768`'s copied retrieval NOTE sealed a **miss**
— because the retrieval lane's ask is to *select* three facts and a copy selects
none, while the summarisation lane's ask is to *preserve* two facts and a copy
preserves both.

**The consequence is the finding of the wave.** Every child this campaign calls
alive at the floor is alive on ONE lane, and that lane is a copy of the prompt's
own payload. **Wave 41 charges all sixteen lanes on all ten children, and the one
held case sits on a charged lane.** Waves 39, 40 and 41's floor movements are all
movements of the ratio at which the model still copies one paragraph.

This wave does not re-grade wave 40's sealed count. It does not leave the
artifact uncharged either.

---

## 4. F1: the floor moves 0.0045651159 again, and again on block 13

| rung | before | after | Δ |
|---:|---|---|---:|
| **sealed 1/15** | `ARCH.V9CL896X13` 0.4070832855 | **`ARCH.V8CL896X13` 0.4025181696** | **0.0045651159** |
| **canonical 1/15** | 0.4070832855 | **DOES NOT MOVE — see §4.1** | **0** |
| sealed 2/15 | `ARCH.V10CL896X09G10` 0.4116484014 | *unchanged* | — |

Registered at one on flatness, reads one. **Second consecutive wave the floor
moves, and the second consecutive time it moves on the block-13 body.** The X13
ladder is now flat at one over **six** rungs, and `NOTES-MX-AR` §11.1's
instruction to excise block 13 rather than block 9 is confirmed by a second
child.

**The qualifier is worse here than it was at wave 40, not better.** Its held
summarisation lane is **byte-identical** to `ARCH.V9CL896X13`'s
(`111e342d9cb5438c`), and **seven of sixteen lanes** are byte-identical to that
control's. Dropping 1,024 further vocabulary rows moved nine lanes and not the
one that holds. **This is wave 40's decode read at a smaller size for the second
wave running.** Three waves of "floor movement" now rest on one decode.

**The sealed 2/15 rung did not move and could not**: §8.

### 4.1 The two lenses have separated completely, and the canonical lens reads ZERO

`wave41/canonical-reread-wave41.json`. The canonical standard forgives an
artifact — a cap-ride, a loop, a word substitution — where the substantive ask is
nevertheless answered. **It does not forgive a lane that returns the prompt
itself**, and wave 40's own canonical artifact says why in those words: *a
copy-back is not a decode failure, it is the absence of a decode.*

**Every held case in this wave is a copy-back. So all ten children read 0/15
canonical.** The sealed lens moves the 1/15 rung to 0.4025181696x; the canonical
lens does not move, because under its own clause there is nothing at this depth
to move. **This is the first wave in the campaign where the two lenses separate
completely**, and one artifact on one lane is the whole of the separation.

**And it exposes a contradiction this wave does not resolve.** The same clause,
applied to wave 40's `ARCH.V9CL896X13`, gives canonical **zero**; wave 40
published canonical **one**. That child's held lane is the source paragraph
returned verbatim — read byte for byte this wave, `111e342d9cb5438c`, and
`V8CL896X13`'s held lane is byte-identical to it. So either the canonical clause
does not mean what wave 40's artifact says it means, **or the canonical 1/15 rung
has been zero since wave 39 and three waves of canonical floor movement are
movements of a rung that is not there.**

**Wave 41 does not re-grade a sealed child.** It records the contradiction, names
the digest that makes it checkable, and leaves the adjudication to the caller.
§12.

---

## 5. F2: ROWSET is closed by three built children — row identity is INERT at fixed K

Sixteen waves have moved **how many** vocabulary rows are kept. None had moved
**which**, although `mx41_vocab.py`'s own docstring says the tie-break decides
most of the table and every receipt admits only 4,633 observed tokens.

All three arms sit on `MX35-CL896X13` at K=9,216, **93,504,256 parameters —
`ARCH.V9CL896X13`'s count to the byte, asserted by the builder before it wrote
anything** — with the mandatory set, the merge closure and the tokenizer writer
held exactly as sealed.

| arm | key | rows moved | jaccard | footrule | lanes matching control | read |
|---|---|---:|---:|---:|---:|---:|
| control | `(-census, id)` | — | 1.000000 | 0.000000 | 16/16 | **1/15** |
| **P** | `(-census, -id)` | **3,276** | 0.698176 | 0.841210 | 1/16 | **1/15** |
| **R** | `(-census, blake2b(41,id))` | **3,008** | 0.719403 | 0.560167 | 2/16 | **1/15** |
| **C** | `(-census, -leverage, id)` | **3,452** | 0.684518 | 0.676124 | 1/16 | **1/15** |

**Each arm swaps about 1,638 rows out and 1,638 in — more rows than the entire
1,024-row bracket that separates the living `V10CL896X09` from the dead
`V9CL896X09`. Each changes fourteen or fifteen of sixteen lanes by bytes. None
changes a single case.** Each arm's held lane is a *different string* from the
control's and from the other arms'.

The forward tie-break, its exact reverse, a seeded permutation and a
reader-metric leverage ranking all read one. **Row-set identity is inert at fixed
K**, and future vocabulary waves argue only about how many rows, never which.

**Arm C was registered to FAIL and did not.** `NOTES-MX-M` §2.3 recorded wave 9's
row-norm ordering at 2.34% coverage and the slate registered leverage to fail
like it. It ties instead. **The predict-and-skip was refused and the build is why
the prediction is known to be wrong** — which is the entire argument for building
predicted failures.

**One thing the gate as written could not do.** The slate's kill gate reads "all
three arms 0/15 … → inert", because it was registered against the *dead* X09
twin at 0/15. The floor moved to X13 between the slate and the wave, so the
control is 1/15 and the gate's number moved with it. The gate's **logic** fires;
its **number** was stale on arrival, and that is stated rather than quietly
re-based.

---

## 6. F3: the deepest TWO-AXIS child in the campaign, and the bracket is not a ratio wall

| child | K | ratio | sealed |
|---|---:|---:|---:|
| `ARCH.CL768` | — | 0.6644639834 | 1 |
| `ARCH.V12/V10/V9CL768` | 12,288–9,216 | 0.4271–0.4134 | 1 / 1 / 1 |
| **`ARCH.V8CL768`** | 8,192 | **0.4088174946** | **1** |
| **`ARCH.V7CL768`** | 7,168 | **0.4042523787** | **1** |

**Flat at one across 0.2602 of stored ratio, on six rungs.** The deepest flat
ladder in the campaign, extended twice.

**`ARCH.V7CL768` is MLP-QUANTUM's bracket entry, corrected before the build.**
The slate registers `V14CL768X09` as the second-axis entry; wave 40 then sealed
`ARCH.CL768X09` at 0/15 **as a body**, so by the flatness law every rung on it
must read zero and the cell could not have distinguished a ratio wall from a dead
body. The entry was moved to the body that is alive — and it needed no 320-second
body rebuild.

It is **the deepest two-axis child this campaign has built**: mlp × vocabulary,
**no excision at all**, the pair wave 30's allocator prices at **1.17 cases**
against the **4.44** every child at this depth has spent. At 0.4042523787x it
sits **0.0028309068 below the floor wave 40 sealed**.

**It does not take the rung** — 0.0017342091 above this wave's own
`V8CL896X13`. A near-miss is reported as a near-miss.

**So the death bracket is NOT a ratio wall in the 0.404–0.407 band**: a two-axis
child is alive inside it while three-axis children at the same depth are not.
`V7CL768`'s held lane (`b5c9c35edfad354b`) is **the only new witness in this
wave** — `V8CL896X13`'s and `V8CL768`'s are byte-identical to their own controls'.

**`V8CL768` shares ELEVEN of sixteen lanes with `V9CL768`**, the
highest-agreeing pair in the wave: dropping 1,024 rows moved five lanes. **The
flat ladder is one decode read at five sizes.**

### 6.1 The fourth artifact species has a new maximum and it is not monotone in ratio

`ARCH.V8CL768` returns the prompt verbatim on **eight of sixteen lanes** —
against wave 40's previous maximum of five — and it does it at **0.4088174946x**,
not at wave 40's 0.664464x. `V7CL768` does it on six, including **the buggy
function from the debugging prompt handed back unchanged**. `V8CL768` rides the
cap on only **two** lanes and still holds nothing but a copy.

**The artifact is not monotone in ratio, and cap-hits do not predict it.**

---

## 7. F4: the X09 cliff is not at the ladder step and not at the midpoint

`ARCH.V95CL896X09`, K=9,728 on `MX37-CL896X09`, 94,028,544 parameters,
0.4093658434x. **Registered at zero, reads zero.**

Nothing had ever been built inside either 1,024-row death bracket, because 1,024
is the *ladder's* step and was never a measured quantum. It is measured now: the
X09 family reads **1/15 at K=10,240 and zero at K=9,728**, so the cliff is in the
**upper half** of the bracket and is **located to within 512 rows**. "Quantized"
was the ladder, only in the sense that the ladder was too coarse to see the edge.

Its retrieval lane is the identical *I do not get a shogike in the question* and
its uncertainty lane the identical *Sysed the sales, including the first, and the
most of the time* that wave 40 quoted for `ARCH.V9CL896X09` and
`ARCH.T9CL896X09`. **This interior rung decodes as a member of the dead X09
family one rung above where that family was known to die.** It is also the only
child in the wave whose summarisation lane loses both facts — **the only one that
cannot even copy**.

---

## 8. F5: NORM-ESTATE — the only repeatable +1 does not repeat, and more free parameters made it worse

All three arms on `MX40-V9CL896X13`, **zero storage delta**, 93,504,256
parameters before and after, byte-identical shapes and config. **Never quoted as
a ratio result.**

| arm | unknowns | update_scale | held-out relSSE | improvement | read |
|---|---:|---:|---|---:|---:|
| **G10** terminal λ1.0 | 1,024 | 0.1540 | 0.3781 → 0.3070 | 18.8% | **1/15** (reg 2) |
| **G03** terminal λ0.3 | 1,024 | 0.2516 | 0.3781 → 0.2909 | **23.1%** | **1/15** (reg 1) |
| **G29** 29-diagonal λ1.0 | 30,464 | 0.1578 | 0.3717 → 0.2972 | 20.0% | **0/15** (reg 1) |

**The estate's only repeatable +1 does not repeat at the floor.**
`ARCH.V10CL896X09` 1/15 with its G10 twin at 2/15 was the mechanism; on this body
G10 reads one. Its held lane is **byte-identical to the source child's**, so the
terminal re-solve did not touch the one lane that holds.

**G29 is the cleanest demonstration of the wave-29 free-parameter hazard in the
estate.** 30,464 unknowns is above the filler count wave 29 measured; the solve
improved its held-out objective **more** than the 1,024-unknown arm did; and it
**destroyed the one case the source held** — its summarisation lane is the source
paragraph's first sentence only, decision and risk both gone. **More free
parameters, better objective, fewer cases, at zero storage, on the floor child.**

**Across the ladder the objective improved 18.8 / 23.1 / 20.0 per cent and the
case count went 1 / 1 / 0.** Wave 27's finding reproduced at the floor on a full
ladder: the objective does not get to choose, and on this body it does not even
correlate.

**The estate does not close on this.** L8 is supported as a terminal-site law
**from the hazard side only**. The slate's Arm M (moment-match, zero fitted
parameters) and Arm S (27 scalars per endpoint) need a new solver and were **not
built**; that limitation was named in the pre-registration, not discovered in the
reading.

---

## 9. Benchmarks, and the column disagrees with the review in both directions again

Standard invocation, parent FIRST. **The parent row reproduces waves 32 and 34
through 40 to every reported digit — the campaign table now carries 28 parent
rows with all 28 matching.**

| model | ratio | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl | sealed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 1.000000 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | **81.80** | 12/15 |
| `ARCH.V8CL768` | 0.408817 | 0.2750 | 0.3150 | 0.5350 | 0.5000 | 0.0800 | **12936.78** | **3255.58** | 1/15 |
| `ARCH.V7CL768` | 0.404252 | 0.3000 | 0.3100 | 0.5300 | 0.4750 | 0.1000 | 19405.37 | 4643.37 | 1/15 |
| `ARCH.V9CL896X13G29` | 0.407083 | 0.2950 | **0.3550** | 0.5100 | 0.4600 | 0.1000 | 21836.20 | 7309.56 | **0/15** |
| `ARCH.V8CL896X13` | **0.402518** | 0.3000 | 0.3100 | 0.5200 | 0.4950 | 0.1000 | 83446.31 | 15521.16 | **1/15** |
| `ARCH.RP9CL896X13` | 0.407083 | 0.2850 | 0.2900 | 0.4700 | 0.5250 | 0.1000 | **102366.37** | **20583.58** | 1/15 |

**Three inversions, and the third is new.**

**`ARCH.V9CL896X13G29` has the best `hella_n` in the wave — better than every
child that holds a case — and it is the only benchmarked child that holds
nothing.** Zero storage separates it from its source; a diagonal re-solve bought
`hella_n` and cost the case.

**`ARCH.RP9CL896X13` is the sharpest reading the column has ever given, because
its control is exact.** It has the same body, the same K, the same parameter
count to the byte and the same sealed count as `ARCH.V9CL896X13` — the ONLY
difference is which 1,638 rows were kept — and it carries **the worst wikitext
perplexity in the wave, 20,583, worse than the deeper `V8CL896X13`'s 15,521**.
**So re-ranking the vocabulary at fixed K moves the perplexity column by a large
factor and the review by nothing at all.** The two instruments do not merely
disagree about which child is better; here they disagree about whether anything
changed.

**And `V8CL768` versus `V8CL896X13` is the same disagreement on the ratio axis**:
4.8x the wikitext perplexity separates them and both read 1/15.

Read the table as a warning, as every wave since 32 has. Nothing here is a
usable model.

---

## 10. Method and controls

- **Parent calibration, fourteenth consecutive, exact.** Re-read blind from
  `ARCH.V8CL896X13`'s dossier — the first of the wave — before any child was
  opened. **5 Useful / 7 Partial / 3 miss = 12/15**, **cap-hits 0/16**, and both
  documented anchors reproduce: the arithmetic Partial boxing **3 kits per box
  with 94 left over**, and the multi-turn miss offering *Vegan Peanut Butter &
  Jelly Bars* to a table with two severe peanut allergies. The third miss is
  summarisation, which **inverts the source** — Supplier B selected "due to its
  lower cost" when B costs 11 per cent more — charged under wave 24's carried
  inversion rule.
- **The runner control is NINE WAVES WIDE.** `mx41_parent_identity.py` reads
  `865f006e3813712ec202cb23`, protocol `e5e377688597`, prompt pack
  `0ee6bf14cef5` on **all ten** wave-41 dossiers.
- **The rowset divergence pre-read ran before any arm was built**, published
  either way, and gated nothing. Its `SEALED` arm moved **0 rows, jaccard
  1.000000, footrule 0.000000** — the ranker reproduces `choose_kept` exactly, so
  a null on an arm is a null about selection and not about the builder. **No
  top-up fired on any arm**; every ordering closed the budget exactly.
- **Every child asserted its twin's parameter count before writing.** The three
  rowset arms each refused-or-passed on 93,504,256 exactly.
- **The lane byte comparison ran before any count was written**
  (`wave41/lane-byte-comparison-wave41.json`) and it decided five claims by
  construction: `V8CL896X13`/control **7 of 16 including the held lane**;
  `V8CL768`/`V9CL768` **11 of 16 including the held lane**; the three rowset arms
  **1, 2 and 1 of 16** against the control with four distinct held-lane digests;
  `V7CL768`'s held lane shared with **nothing in the estate**.
- **Seal verification**: 1034 ledger rows, **1030 resolve to a receipt on
  disk, 1030 verify, 0 FAIL**, 4 unresolved -- the same historical four.
  **37 seal roots** audited against disk, **0 undeclared**.
- **`sealq.sh` ran ONCE** and all ten sealed on the first call.
- **Pre-registration hit rate: 7 of 10.** Hits: `V8CL896X13` 1, `V7CL768` 1,
  `V8CL768` 1, `V95CL896X09` 0, `RP9CL896X13` 1, `RR9CL896X13` 1, `G03` 1.
  Misses: `RC9CL896X13` 0→**1** (better than registered, and the prediction was
  the slate's), `G10` 2→**1**, `G29` 1→**0**. **All three misses are one case
  wide and all three are on arms whose storage delta is zero.**
- **The one conditional cell whose gate failed was NOT built.** `ARCH.V6CL768`'s
  registered gate was "only if `V7CL768` reads 1 or more" — it reads 1, so the
  gate **was** met and the cell was still not built, for budget; that is recorded
  as a deviation and not as a gate. `V9CL896X13G03`'s gate ("only if G10 and G29
  both complete") was met and it was built.

---

## 11. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `724c82c5` | cut wave 41's tree, author `mx41_vocabrowset.py`, commit the round-3 slate into `docs/`, append the sealer's six accept strings **in the same commit**, and pre-register |
| `ab73e175` | the ten seals |
| *(this)* | the measurement, the tables and the docs |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `724c82c5` | 7350 | 0 | 0 | 0 | 0 |
| `ab73e175` | 7350 | 0 | 0 | 0 | 0 |

**Ledger.** Re-verified on the share before it was touched (1014 rows,
`c0aa462f…`), backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave41-20260829T043331Z` and
`cmp`-verified byte-identical. **One-pass merge**, all 20 shards and all 10
verdict receipts in a single invocation, 1014 → **1034** rows, 0 superseded.
Byte-prefix verified: `head -c <backup size> ledger.jsonl | sha256sum` returns
`c0aa462f…` exactly. **Mirror copy-back done, and the mirror write asserted the
merged file is a byte-prefix extension of what it replaced BEFORE writing.**

**A dry run caught a bad invocation before it could write.** The first merge
attempt used the wrong argument form, exited non-zero, and left the ledger at
1014 rows and `c0aa462f…` unchanged — verified, not assumed, before the correct
invocation was issued.

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used throughout,
with the in-process SM86 preflight on every driver. Both carried an idle-holding
H3 ComfyUI container (~288 and ~279 MiB) and were **not restarted**, for wave
34's stated reason. The four P40s were never touched. The Quadro RTX 8000 read
**330 MiB** at preflight and wave 41 issued no command against index 6.

**Disk**: `/mnt/ai_data` 4.5T, 1.7T available. **The Codex lane was not touched.**

---

## 12. The export-ruling item for Cyril

**The slate's ALIAS ruling is moot as written and is replaced by a narrower,
older ask.**

The slate asked: *does an id-remapped tokenizer count as a standard checkpoint?*
That question presupposes an ALIAS child exists to rule on. §1.1 measured that
one cannot be built: a many-to-one encoder is not expressible in a stock
`tokenizer.json` on this parent, in three distinct constructions with three
distinct failure modes.

**What is actually blocked, and has been for twenty-seven waves, is the PROBE
storage-claim ruling.** Three classes now sit behind it:

1. **K4**, score-bilinear joint q/k — the strongest unspent lead in the estate.
2. **K3(c)**, Monarch's first cell.
3. **ALIAS**, as of this wave.

Each is realisable only as a materialised dense export whose storage claim is
**arithmetic rather than a loader count**, and `PHASE1` §6 exists because
arithmetic storage claims are what "never priced behaviourally" means.

**The ask is one ruling, not three:** may a `PROBE.*` child carry a storage
claim that is stated arithmetic, labelled non-loader, and excluded from the
frontier — so that these classes can be *priced behaviourally* without ever
being quoted as a ratio? A yes unblocks three classes for the cost of one
adjudication. A no closes them permanently and the estate stops budgeting them.

**And there is a second ruling, which is larger and which this wave did not have
the standing to make.** §4.1: does a response that returns the prompt's own
payload — preserving the asked-for facts by copying rather than by answering —
hold a case under the canonical standard?

- **NO** makes the canonical frontier **0/15 below roughly 0.41x**, retires
  three waves of canonical floor claims, and means the campaign's deep band has
  no living child on that lens at all.
- **YES** keeps them, and requires the canonical clause's copy-back sentence to
  be **rewritten**, because as written it says the opposite.

This is not a measurement and no wave can settle it by building another child.
It is worth more than any cell on the round-3 slate: it decides whether the last
three waves of floor movement are real on the second lens.

---

## 13. What this wave did NOT build, and why

1. **ALIAS**, both arms — blocked at construction (§1.1). Not closed.
2. **ROWSET Arms A and B** — blocked on corpus (§1.2). Not closed;
   re-proposable the moment a corpus exists that is real text, disjoint from the
   frozen 15 and the rider bank, and not contractually barred from selection.
3. **`V14CL768X09`** as the slate wrote it — its body reads 0/15, so the cell
   cannot distinguish a ratio wall from a dead body. Substituted by `V7CL768`
   with the cause registered before the build (§6).
4. **NORM-ESTATE Arms M and S** — need a new solver. The grid therefore straddles
   the wave-29 hazard **on its hazard side only** (§8).
5. **XCHANGE** — its Stage-1 multi-plane attention construction and Stage-2
   alternating solve are a new solver, not a driver change. **NOT STARTED; no
   census was run, so nothing about it is claimed either way.**
6. **PLANE** — same. The plane-energy census that would have pre-registered the
   HD48 rung was **not run**, so the HD48 prediction remains unregistered and the
   axis is neither opened nor re-sealed.
7. **ROWSET-LANE** — its steering corpus is the corpus §1.2 found does not exist.
8. **SC-CORESET** — research probe, adjudication-gated, and a new solver.
9. **`ARCH.V6CL768`** — its gate was met and it was not built, for budget. A
   deviation, recorded as one.

**Items 5 and 6 are the honest cost of this wave.** Two of the slate's four
highest-floor items were not touched at all, and the reason is authoring time for
new solvers rather than any finding about them.

---

## 14. What the next wave should know

1. **THE FLOOR'S ONE CASE IS A COPY OF THE PROMPT.** Every child at the floor is
   alive on the summarisation lane and that lane returns the source paragraph.
   Waves 39, 40 and 41 have each moved "the floor" while the held lane stayed
   byte-identical to the wave before. **Before the next floor cell is built,
   decide whether a copied source paragraph is a held case at all** — because if
   it is not, the campaign's sealed 1/15 rung is 0/15 and has been for three
   waves. This wave charged the lane and did not re-grade the count; that
   decision is the caller's, not a wave's.
2. **The 15-case instrument is exhausted at this depth.** Ten children, four
   distinct mechanisms, ratios from 0.4025 to 0.4094, and the entire signal is
   one lane. A wave that wants to measure anything below 0.41x needs the
   30-prompt extension bank as a primary surface, not a rider.
3. **`ARCH.V6CL768`, K=6,144 on the retained `MX40-CL768`, is the cheapest open
   floor cell in the estate** — 0.3996872628x, one CPU pass, **below 0.40x**, on
   a ladder that is flat at one over six rungs. Flatness predicts one; the census
   floor predicts zero. It was gated, the gate was met, and budget stopped it.
4. **Excise block 13, and stop building X09 rungs.** Confirmed by a second child.
   The X09 cliff is now located to within 512 rows and there is nothing left to
   learn from that family cheaply.
5. **Stop budgeting the vocabulary tie-break.** Three built children close it.
6. **The zero-storage gain estate should be closed or opened by Arms M and S,
   not by more D arms.** Three D-arm rungs on one body gave 1 / 1 / 0 with the
   objective improving monotonically; a fourth will not say more.
7. **`mx41_child_index.py --out` works and parses itself.** Wave 40's §11.6 is
   discharged.
8. **The prompt-returned-verbatim artifact is NOT monotone in ratio and cap-hits
   do not predict it.** Eight of sixteen lanes at 0.4088x with two cap-hits;
   five of sixteen at 0.4025x with nine. It needs a name and a counter in the
   dossier reader, not a note in a wave record.
