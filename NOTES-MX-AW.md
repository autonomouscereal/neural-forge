# NOTES-MX-AW — wave 45: the echo-depth close-out and the two named debts. The anchor was paid at the cut and did not do what it was ordered to do, the echo dies by corruption rather than truncation, and the CL768 ladder is closed at the bottom

Wave 44 left three things named and unpaid: a `PAYLOAD_ANCHORS` entry it had
diagnosed and correctly refused to add mid-measurement, a 30-prompt extension
surface declined twice as unordered, and two rungs of vocabulary ladder nobody
had priced. Wave 45 pays all three and writes the standing-state document. It
authors no module, builds two children, and its most useful result is a
registered prediction that hit and a registered *mechanism* that missed.

---

## 1. The tree, and the audit surface nobody had ever read

`cut_wave45.py` cut `run/mx45` from `run/mx44` on the same five-entry
substitution list waves 31–44 used. The four hand-edited files behaved as they
always do:

* `mx45_verify_seals.py` — **40 declared roots become 41**, `wave44/cells` still
  declared. Twelve roots asserted by name.
* `mx45_retain.py` — arrived carrying wave 44's one keep and two drops with every
  justification paragraph intact. Both mappings emptied by hand; six retargets at
  pinned counts of one each. Sixteenth consecutive wave.
* `mx45_build.py` — one schema string retargeted, **wave 44's basis schema
  appended** (the accept list now carries wave 45 back to 19, seventeen strings),
  eighteen module references, both receipt-write sites, the log prefix. **Wave 45
  does not run this file.**
* `sealq.sh` — arrived with wave 44's **three** seal calls intact, and wave 42's
  general calibration pin reported **83 residue lines** below the sentinel, the
  **largest count that pin has ever reported** (wave 43: 78, wave 44: 62). All 83
  deleted by hand. Twelfth consecutive refusal.

### 1.1 THE ITEM `NOTES-MX-AV` §12.6 NAMED, AND IT LEADS SOMEWHERE WORSE

§12.6 said the next place to look is `experiments/seal_matrix_behavior_review.py`'s
own constants, which `seal_verdict.sh` passes flags to and which no cut has ever
read. Reading it turns up something narrower and worse than a constant in that
file: **the numbers that turn a reviewer's count into a VERDICT are not in that
file at all.**

`behavior_verdict_for` reads `PARENT_REVIEW_BAND = 12`, `A9_REVIEW_BAND = 8` and
`PARENT_BAND_TOLERANCE = 2` out of `src/neural_forge/matrix_cell_receipt.py` —
outside the cut tree, outside the sealer, and **never asserted by anything in
forty-four waves**. Every `alive-parent-band` / `alive-degraded` / `impaired`
label in the ledger rests on those three integers, and the `dead` line that every
deep-band child in this campaign is sealed through is one `if` beside them.

`cut_wave45.py --audit` now pins all three **by value**, plus the band arithmetic
and the death line at their indented forms, plus both verdict vocabularies, plus
the sealer's three argparse defaults — and the *absence* of a default on
`--parent-useful-or-partial`, which must stay absent so a wave that forgets the
calibration flag fails instead of inheriting a number.

**Nothing was found wrong, and that is the result.** This is the fourth external
surface the cutter checks and the first one found *before* it cost anything. The
builder's accept list cost wave 22 a silent rebuild; the sealer's accept list
cost wave 43 a whole deployment; two of `sealq.sh`'s three path pins did not
exist until wave 44 measured that they did not.

---

## 2. THE ANCHOR DEBT: PAID AT THE CUT, AND ITS OUTCOME REGISTERED BEFORE THE PASS

`NOTES-MX-AV` §2.3 measured that **all eight void-changing machine–hand
disagreements sit on lanes with no `PAYLOAD_ANCHORS` entry**, seven of them on
`writing-concise-email`, and §12.1 ordered the entry registered before the pass
that uses it. It went in at commit `16f443b0`, the tree cut, before any wave-45
pass existed:

```
"writing-concise-email": ("under 90 words", None),
```

The cut point is the rule the other three anchors already follow — strip the
leading task directive, keep the material the lane supplies — applied to a prompt
whose directive is *Draft a polite email under 90 words*.

### 2.1 THE REGISTERED PREDICTION WAS A NULL RESULT, AND ALL FOUR PARTS HIT

The seven rows carry **two distinct responses**, both committed in wave 44's
artifact and both readable in the frozen dossiers. Both are the same failure —
*"I'm here here. Please. I'm a polite email that's asking a vendor to confirm
whether a delayed shipment will arrange by Friday. I'm here."* — the model hands
back the prompt's **first sentence** and drops the second entirely.

| # | registered at `16f443b0` | measured | verdict |
|---|---|---|---|
| P1 | `writing-concise-email` void-changing disagreements **stay at 7** | 7 | **HIT** |
| P2 | total void-changing disagreements **stay at 8** | 8 | **HIT** |
| P3 | both floors unmoved (A 0.3859919152x, B 0.4116484014x) | unmoved, and both machine-only columns too | **HIT** |
| P4 | `response_union_coverage` reads **0.683824** and **0.635714** | exactly | **HIT** |

**The anchor IS live and that is measured separately.** All 45 lanes in the band
that carry a `writing-concise-email` response move from payload 193 characters to
157, so this is a resolving lookup and not a silent failure. A registration that
predicts no change has to prove the change was applied.

**Exactly one machine classification moves in 912 lanes**: `ARCH.V7CL896X13`'s
writing lane, `COPY_BACK → CONTENT` at 0.616580 → 0.528662. That lane is sealed
*Observed miss*, was never held, and cannot be voided; the child reads 0 on all
four columns before and after. Its response-side number, 0.410345, agrees with
the anchored call, so the single move the anchor makes is **toward** the other
measure and not away from it.

### 2.2 WHY REGISTER A PREDICTION YOU EXPECT TO HIT

Because the alternative is worse, and the sweep is published so the refusal is
checkable rather than merely stated. Payload = the normalised user turn under
each candidate cut, same union, same `UNION_MIN_BLOCK = 8`, same 0.60:

| candidate | payload chars | `39fdfd…` | `6022…` | verdict |
|---|---:|---:|---:|---|
| none (wave 44's fallback: whole prompt) | 193 | 0.481865 | 0.461140 | CONTENT |
| **`("under 90 words", None)` — REGISTERED** | 157 | 0.477707 | 0.452229 | CONTENT |
| `("Draft a polite email", None)` | 172 | 0.447674 | 0.424419 | CONTENT |
| `("asking", None)` | 150 | 0.453333 | 0.426667 | CONTENT |
| first sentence only — **REFUSED** | 112 | 0.821429 | 0.785714 | COPY_BACK |
| second sentence only | 80 | 0.000000 | 0.000000 | CONTENT |

**Four of the five defensible cuts agree with each other and disagree with the
hand.** The fifth works by deleting the exact sentence the model failed to copy.
Registering that one would be fitting the instrument to the answer — the thing
wave 42 refused, wave 44 refused, and this wave refuses **in writing with its
number published** rather than by quietly not trying it.

### 2.3 THE RESIDUAL IS A DENOMINATOR, NOT AN ANCHOR — AND IT IS PUBLISHED, NOT PROMOTED

Wave 44's mechanism — *"the copied material is the ask and the ask is only about
half of the prompt"* — is right about the arithmetic and wrong about the repair.
**No cut of this prompt turns half of it into 0.60 of itself.**

The payload-side question is *how much of the supplied material came back*, which
is right for a lane that supplies material. `writing-concise-email` is a `PRODUCE`
lane: it supplies nothing, the whole prompt is instruction, and a model that hands
back half the instruction cannot score 0.60 against the whole of it no matter
where the anchor is cut. The question that lane needs is *how much of what this
model said is lifted from the prompt* — a **denominator** change.

`response_union_coverage` is that column: same union, same minimum block, same
0.60 numeral, response as denominator, **whole prompt as source**.
`RESPONSE_SIDE_DECIDES_NOTHING = True` is declared in one constant and
`cut_wave45.py --audit` pins its decision line at **zero occurrences**. This is
the campaign's own promotion protocol — waves 42 and 43 published the union
measure beside the deciding one for two waves before wave 44 promoted it on that
evidence. Wave 45 publishes the first of the two.

### 2.4 A DEFECT THIS WAVE MADE AND CAUGHT INSIDE THE CUT

The first version read `union_coverage(verbatim, payload)` — the response against
the **anchored** slice. It compiled, it ran, and it returned 0.551471 and
0.507143, both under 0.60: a column that appeared to say the denominator change
buys nothing. Anchoring the source removes the instruction text, which is exactly
what a PRODUCE-lane copy lifts.

It was caught by checking the README's registered numbers against the module that
would produce them **before the cut was committed**. The fix is
`lane_whole_prompt()`, a named accessor both paths share, and the audit pins the
call site at its indented form *including its source argument*. Recorded because
a wave that only reports the defects it finds in other waves is not measuring
itself.

---

## 3. THE CENSUS: THE MERGE CLOSURE FIRES FOR THE FIRST TIME ON THIS AXIS, AND IT IS NON-MONOTONE IN K

| K | admitted by ranking | **skipped for closure** | merges retained | post-closure occ. | post-closure distinct | captured-batch occ. |
|---:|---:|---:|---:|---:|---:|---:|
| 3,072 | 1,533 | **0** | 2,307 | 0.947327 | 0.401043 | 0.986109 |
| 2,048 | 827 | **1** | 1,283 | 0.895980 | 0.242227 | 0.958011 |
| 1,024 | 138 | **0** | 259 | 0.677894 | 0.067213 | 0.760537 |

Wave 44 registered zero at all three of its rungs and wrote that *"whatever ends
this ladder, it is not the closure"*. At K=2,048 it is, for one row — and **it is
non-monotone in K**, which no account in this campaign predicts. The gate measures
**ranking depth against free rows**, not K: at K=1,024 the ranking admits only 138
tokens against 259 free rows so every closure fits, while at K=2,048 it reaches
deeper into the merge tree and one candidate's closure does not. Both build
receipts reproduce the census exactly.

**The structural refusal was reachable and did not fire.** 765 mandatory rows
against K=1,024 leaves 259 discretionary rows. `if mandatory_size > k:` is pinned
at its indented form by this wave's cutter precisely because this is the first
wave that could reach it. **K=768 is the last rung on this axis the mandatory set
permits at all.**

**K=3,072 is re-censused as the cross-wave control and reproduces wave 44 to every
published digit, including `merges_retained`, which wave 44 did not compare
across waves.**

---

## 4. F1 AND F2: THE ECHO DOES NOT SHORTEN, IT DEGRADES

**Both registered counts are HITS and the registered MECHANISM is a MISS**, and
both are recorded, because registering only the number is how a wave learns
nothing from being right. The pre-registration named **truncation**: the echo
would start and ride the cap without reaching the decision and the risk.

### 4.1 `ARCH.V2CL768` — K=2,048, 87,611,136, **0.3814267993x**, **0/15**

Its summarisation lane, `959f866ca8f0e732`, terminates **complete at 168 tokens**
against a 1,024-token cap. **It is not truncated at all.** It is still the whole
prompt returned including the instruction line, and it preserves the decision
**verbatim**: *The team selected Supplier B*.

What it destroys is the **unresolved risk**, and the corruption is **selective** —
the high-frequency sentence frame survives while low-frequency content words die:

| parent text | what comes back |
|---|---|
| three **battery** suppliers | three **bitties** suppliers |
| before **launch** | before **lacuan** |
| The **remaining** risk | The **reminance** risk |
| its **factory expansion may slip** by one week | its **facrept** by one week |

Wave 44's held rung one step above carries **one** visible corruption in 484
characters; this carries **four**, and one lands exactly on the risk. A reader of
the response alone cannot recover *what* may slip, so wave 40's criterion — the
decision **and** the unresolved risk — has one of its two halves and the lane is
not held.

That is what a **vocabulary-coverage** loss predicts and not what a **token-budget**
loss predicts. The coverage account owns this result and the echo-budget account
this wave bet on does not.

All sixteen lanes charged, four ride the cap, seven return the prompt itself.

### 4.2 `ARCH.V1CL768` — K=1,024, 86,562,560, **0.3768616835x**, **0/15**

**The deepest child this campaign has ever built.** The echo is gone and what is
left is sub-word debris. Its summarisation lane, `c94f8e7b0a037cf7`, is
`su ppulpulpul…` to the cap — not a corrupted copy of the prompt but a
three-character fragment repeated a thousand times, so the lane wave 27 observed
is the last to go **has gone**.

**Ten of sixteen lanes ride the cap, and the loop unit is below the word on nine
of them**: `tzhe`, `kjp`, `vvvv`, `it_it`, `pnlp`, `ttketu`, `ttlhouset`,
`wowowo`, `ppul`. **One lane returns the empty string** — `context-retrieval`,
zero generated tokens.

**And its last surviving sentence of English is a copy of the prompt's own
assistant turn.** `multi-turn-correction` answers *"Before planning, are there any
alelergies or dietary restrictions?"*, which is the conversation's own second
message returned with one corruption, and it is the **only grammatical clause this
child produces on any of sixteen lanes**. Wave 44 found that the deep band's
terminal behaviour is returning the prompt itself; this child shows that the last
thing to survive is that echo, on the shortest copyable span in the surface.

### 4.3 THE BRACKET

| rung | K | ratio | sealed | the summarisation lane |
|---|---:|---:|---:|---|
| `ARCH.V3CL768` | 3,072 | 0.3859919152 | **1/15** | echo intact, one corruption in 484 chars, **holds** |
| `ARCH.V2CL768` | 2,048 | 0.3814267993 | 0/15 | echo intact and complete; decision verbatim, **risk destroyed** |
| `ARCH.V1CL768` | 1,024 | 0.3768616835 | 0/15 | **sub-word debris** |

**Column A does not move for the first time in six waves.** Waves 40–44 moved it
five consecutive times.

Zero of this wave's thirty-two lane responses are shared with each other or with
`ARCH.V3CL768`; every one is an independent witness.

---

## 5. BENCHMARKS: THE COLUMN THE SEALED FIFTEEN CANNOT ORDER

| K | cell | wiki wppl | byte ppl | sealed |
|---:|---|---:|---:|---:|
| 6,144 | `ARCH.V6CL768` | 5,035 | 4.92 | 1/15 |
| 5,120 | `ARCH.V5CL768` | 6,657 | 5.19 | 1/15 |
| 4,096 | `ARCH.V4CL768` | 8,749 | 5.46 | **0/15** |
| 3,072 | `ARCH.V3CL768` | 14,019 | 5.96 | **1/15** |
| 2,048 | `ARCH.V2CL768` | 31,968 | 6.96 | 0/15 |
| 1,024 | `ARCH.V1CL768` | **961,032** | **13.15** | 0/15 |

Parent 81.80 / 2.28, re-run this wave and matching the sealed reference.

**Word perplexity is strictly monotone in K over six rungs. The sealed count is
not monotone at all.** Wave 44 said this with two rungs and a 1.6x gap; six rungs
and a 191x spread say it without needing to be argued.

The K=2,048 → K=1,024 step is a **30x** cliff in word perplexity, which is the
same step where the echo disintegrates below the word.

---

## 6. THE EXTENSION DEBT: PAID, WITH ITS OWN REPRODUCTION CONTROL

`NOTES-MX-AV` §10.6 declined this twice as unordered and §12.4 called it the
cheapest unclaimed item on the board. It cost about seven minutes of 3090 time.

| cell | ratio | sealed 15 | ext held /30 | ext charged /30 |
|---|---:|---:|---:|---:|
| `ARCH.V12CL896G03` | 0.4510448164 | **4/15** | 0 | **14** |
| `ARCH.V10CL896X09` | 0.4116484014 | 1/15 | 0 | **2** |
| `ARCH.V6CL768` | 0.3996872628 | 1/15 | 0 | 19 |
| `ARCH.V5CL768` | 0.3951221469 | 1/15 | **1** | 19 |
| `ARCH.V3CL768` | 0.3859919152 | 1/15 | 0 | 16 |
| *parent* | 1.0 | 12/15 | 5 | 2 |

**The registered prediction is a MISS on its one agreement point and a PARTIAL
HIT on its tie-break.** It was registered that `ARCH.V12CL896G03` at 4/15 would
also score highest of the five. It does not: it holds **0** while
`ARCH.V5CL768`, at 1/15, holds **1**. The three-way 1/15 tie breaks on the held
column (1, 0, 0) and fully on the degeneracy column (19, 16, 19).

**THE TWO INSTRUMENTS INVERT AT THE TOP OF THE FRONTIER.** The sealed column's
best child of the five is charged on **14 of 30** extension prompts; the column-B
floor holder, which is deeper and reads 1/15, is charged on **2**. A **sevenfold**
difference in degeneracy running the opposite way to the sealed column, on the two
children the standing state names as the top of column B and the top of the graded
band.

**And the comparison is legitimate across waves, which is measured rather than
assumed.** Wave 34 ran this bank on the parent and on `ARCH.V12CL896G03`. Wave 45
re-ran both eleven waves and three deployments later and reproduced **every
scored field and all 60 per-response digests exactly**. That is a decode-path
reproduction test across eleven waves that nothing else in this campaign performs.

The parent itself holds only 5 of 30 on this rubric. It is a deliberately
low-resolution machine instrument, acceptable for *ordering* children and
unacceptable for *grading* one — exactly the split ADDENDUM 16.1 draws. **Nothing
here is a verdict, no ledger row is generated from it, and no frontier claim
fires on it.**

---

## 7. THE RETENTION PROMISE, CASHED FOR THE FIRST TIME IN SIXTEEN WAVES

The ordered surface named `ARCH.V10CL896X09` — and **the column-B floor holder's
checkpoint was not on disk.** Wave 38 dropped it on the ground that it is *"one
40-second CPU vocabulary pass off wave 37's RETAINED `MX37-CL896X09`"*. **Sixteen
waves of DROP reasons make that promise and no wave has ever tested one.**

Wave 45 rebuilt it, without `--below-floor` because wave 38's receipt records
`below_census_floor: false`, so the coverage guard was exercised **armed**:

```
child_tensors_sha256   32ed4000ba7107809ff6a2860de23ede3793bfdbb7b05957cb16d364a52f4c26  MATCH
child_directory_sha256 24b8a5b106924e9fe1b71282060784a7b9d46027833c25a22898558c4673fddd  MATCH
child_parameters       94,552,832    net_stored_ratio 0.4116484013735471   tokens/byte 0.2655
```

The frozen runner binds a child **by digest and not by path**, so wave 38's
sealed cell receipt loaded the rebuilt bytes with no modification and the
extension surface decoded through it. **The promise is now a measurement.**

---

## 8. Method and controls

- **Pre-registration.** The anchor and its four predictions at `16f443b0`, before
  any wave-45 pass; the two cells and six secondary predictions at `3ca5b713`,
  before either child existed.
- **Parent calibration.** Re-read blind from `ARCH.V2CL768`'s dossier — the FIRST
  of this wave, sealed 94 seconds before the other — against the standard at the
  top of `sealq.sh`. **Eighteenth consecutive.** 12/15, cap hits 0/16, both
  documented anchors verbatim. **The split is 6/6/3, which is wave 44's split lane
  for lane and wave 43's before it**; the inherited reference is 5/7/3 and wave 42
  read 4/8/3, and all five total 12/15 and name the same multi-turn miss.
- **Reading order.** Both children are under 0.60x, so the summarisation lane was
  opened and written down FIRST on each, before any other lane of either child.
- **Wave 44's adjudication is binding and unamended**: a response that returns the
  prompt itself is charged as a degeneracy and can still HOLD a PRESERVE lane.
  Wave 45 proposes no amendment, so the same rule decided its rungs.
- **SSH discipline.** Every run over 300 s launched under `nohup` to a unique
  path, with a `pgrep` pre-kill check first. The 300-second command ceiling
  orphaned three waiting loops; the underlying work was unaffected and
  `pgrep -af mx45` reads clean.
- **Pre-registration hit rate**, stated exactly: **2 of 2** on the sealed counts;
  **4 of 4** on the anchor predictions; **4 hit, 1 missed and 1 vacuous of 6** on
  the secondary ones — parameter counts, closure counts in both build receipts,
  the retention digest and the extension reproduction control all HIT, the
  concordance prediction MISSED on its agreement point, and the prospective union
  verdict is VACUOUS because neither child held a lane for it to decide. **And
  the registered MECHANISM is a MISS**, which is the one grade this wave learned
  something from.

---

## 9. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `16f443b0` | cut wave 45's tree and PAY THE ANCHOR DEBT AT THE CUT |
| `3ca5b713` | run the census and the anchor-delta pass, and pre-register the two cells |
| `b93a2c26` | the two seals |
| `1ade6ade` | the measurement, the standing-state document, the tables and the dual-column refresh |
| *(this)* | retention applied, and this file's own estate section closed |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `16f443b0` | 7840 | 0 | 0 | 0 | 0 |
| `3ca5b713` | 7846 | 0 | 0 | 0 | 0 |
| `b93a2c26` | 7846 | 0 | 0 | 0 | 0 |

**THREE deployments for five commits**, which is one fewer than waves 43 and 44
and is not an improvement in discipline — it is the shape of a wave whose seals
went in as a code commit rather than as a hand-run script. The measurement commit
and the retention commit get no deployment, for the reason wave 43 and 44 stated:
each is written *after* the thing it describes, so neither can be inside the
deployment the wave was measured from.

**Ledger.** Re-verified on the share before it was touched (1052 rows,
`9e4ff05ea3a5b3d0…`), backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave45-20260829T122712Z` and
`cmp`-verified byte-identical. **One-pass merge**, 1052 → **1056** rows, 0
superseded. Byte-prefix verified: `head -c 1859949 ledger.jsonl | sha256sum`
returns `9e4ff05ea3a5b3d0…` exactly, and `cmp -n 1859949` against the backup
agrees. New digest `d2027831e229d763a251ca602655c729a4432645bed43a9ba2e4a8824ef211ea`.

**Seals.** 41 declared roots, audited against disk, 0 undeclared. 1,056 rows,
**1,052 resolve, 1,052 verify, 0 fail**, 4 do not resolve — the same three
`CAL.C01.T5.X4` rows and one `COMP.MC.MXO0` that have not resolved since before
wave 40.

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used for the two
reviews, the extension surface and the benchmarks, with the SM86 preflight on
every driver. Both carried an idle-holding H3 ComfyUI container and were **not
restarted**, for wave 34's stated reason. The four P40s and the Quadro RTX 8000
were never touched. All three builds — two cells plus the `V10CL896X09` rebuild —
were **CPU-only** vocabulary gathers.

**Retention**: `package.sh` ran BEFORE `--plan`, as the rule requires, and
`--plan` was read before `--apply`.

**KEEP = `MX45-V10CL896X09`, and it is a DEVIATION from the drop rule, argued
rather than hidden.** By the letter of the rule it should go: it is 25 seconds of
CPU off a retained source, and *this wave proved that*. It is kept on **wave 40's
frontier-holder rule, applied to column B's holder for the first time**. That
rule — *a floor whose child has been deleted is a floor that can only be cited
and not checked* — has retained column A's holder for five consecutive waves.
Column B's holder has had **no checkpoint since wave 38**, and wave 38 was right
to drop it at the time: the dual column did not exist until wave 42, and the
child was then only a joint holder of a rung another retained checkpoint also
held. It is now the sole cell behind 0.4116484014x — the number the pending
copy-back ruling may make *the* floor of the whole campaign — and wave 45
measured the concrete cost of its absence: the ordered surface could not touch it
until it was rebuilt. **A ruling that lands on column B should not have to wait
on a build.**

**DROP = `MX45-V2CL768` and `MX45-V1CL768`**, both one CPU pass off wave 40's
retained `MX40-CL768`, and in both cases the finding is a reading of committed
dossier text rather than anything a reader can do with the bytes. **Applied**:
189,733,004 bytes kept, 348,709,456 deleted, manifest re-parsed OK.

**Disk**: `/mnt/ai_data` 4.5T, **1.7T available**. **The Codex lane was not
touched** — every commit staged by explicit path, and the 324 uncommitted paths it
carries are untouched.

**Final state**: both 3090s back to their idle H3-container residents (288 and
279 MiB) at 0% utilisation, the four P40s at 5 MiB and the Quadro RTX 8000 at
330 MiB, **no stray `mx45_*` process on the box**, ledger 1,056 rows at
`d2027831e229d763…` on the share **and in the repository mirror**, one retained
child on the wave root.

---

## 10. What this wave did NOT build, and why

1. **No PLANE child.** Re-sealed at wave 43 under the slate's own kill gate.
2. **No XCHANGE child.** Closed by its own pre-registered gate at wave 43.
3. **ROWSET Arms A and B** — still blocked on a corpus; none has appeared.
4. **ALIAS, K4, K3(c)** — still blocked behind the PROBE storage-claim ruling.
5. **NORM-ESTATE Arms M and S** — still need a solver nobody has authored.
6. **No promotion of the response-side copy measure.** §2.3. It is published and
   decides nothing, which is the first of the two waves of evidence the
   campaign's own promotion protocol asks for.
7. **No anchor for `uncertainty-insufficient-data`.** The eighth void-changing
   disagreement is a SELECT lane the machine over-reads at 0.693333, and it is
   **the same lane the column-B floor holder is held on**. An anchor there could
   move column B, and a wave that has already registered one anchor's outcome
   should not add a second whose effect it has not priced. Left with its price on
   the record.
8. **No K=768 rung.** It is the last the mandatory set permits, and the rung above
   it produces sub-word debris on every lane. Building it would buy a number and
   no information.

---

## 11. The two rulings

**Ruling 1 — the PROBE storage claim** (`NOTES-MX-AS` §12), unchanged, still
blocking K4, K3(c) and ALIAS.

**Ruling 2 — the copy-back clause**, unchanged and still costing nothing to
apply. Both readings complete over 57 children, per child and per lane, with the
digest that makes each call checkable:

- **NO** (copy-back refused): **0.4116484014x**, `ARCH.V10CL896X09`, holding
  `uncertainty-insufficient-data`. Unmoved.
- **YES** (copy-back counts): **0.3859919152x**, `ARCH.V3CL768`. Unmoved for the
  first time in six waves.

**`reports/matrix-campaign-20260810/STANDING_STATE_2026-08-29.md` is the one page
that says where both stand and exactly what each ruling unblocks.**

---

## 12. What the next wave should know

1. **THE COPY MEASURE'S REAL DEFECT IS A DENOMINATOR AND IT IS NOW PUBLISHED
   ONCE.** §2.3. Register `response_union_coverage` as PRIMARY **for PRODUCE
   lanes only**, keeping the payload-side column primary for PRESERVE and SELECT,
   before the pass that uses it, with the prediction that void-changing
   disagreements fall from 8 to 1. It costs no GPU. One more wave of publication
   and it will have the same evidence behind it that wave 44's promotion had.
2. **THE CL768 AXIS IS FINISHED AND SO IS COLUMN A'S MOVEMENT ON IT.** §3, §4.
   K=768 is the last rung the mandatory set permits and the rung above it is
   sub-word debris. Any further column-A movement has to come from a different
   body, and the only bodies with room are the ones behind the block-13 cliff,
   which wave 44 controlled and found dead for the body's own reasons.
3. **THE SEALED FIFTEEN AND THE EXTENSION SURFACE NOW DISAGREE ABOUT WHICH CHILD
   IS BEST, AND SO DOES PERPLEXITY.** §5, §6. Three instruments, three orderings,
   one frontier. A wave that wants to rank children at this depth has to say which
   instrument it is ranking by and why, before it ranks. The extension surface's
   cross-wave reproduction is exact, so it is now a usable coordinate and not a
   one-off.
4. **A REBUILD-BY-DIGEST IS THE CHEAPEST INTEGRITY TEST IN THIS CAMPAIGN.** §7.
   One CPU pass proved a promise sixteen waves of retention manifests had been
   making on credit. There are dozens of dropped children whose DROP reasons make
   the same promise; testing two more per wave costs minutes and would make the
   whole retention policy a measured claim rather than a stated one.
5. **THE NEXT AUDIT SURFACE IS THE FROZEN RUNNER ITSELF.** §1.1 closed the
   verdict-band constants. `run_lfm2_composition_frozen_review.py`'s own
   generation constants — the greedy loop, the cap accounting, the termination
   taxonomy — are what every dossier in 1,056 rows was produced by, they live
   outside the cut tree, and no cut has ever read them either. The
   `--maximum-new-tokens 1024` in `seal_and_review.sh` is pinned by nothing.
6. **REGISTERING A MECHANISM ALONGSIDE A COUNT IS WORTH MORE THAN THE COUNT.**
   §4. This wave's two counts were right and its mechanism was wrong, and the
   wrong mechanism is what produced §4.1's table. A registration that is only a
   number can only be graded; a registration that names *how* can be corrected.
