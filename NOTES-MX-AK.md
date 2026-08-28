# NOTES-MX-AK — wave 34R: crash cleanup, and the estate re-read under the canonical standard

**2026-08-28.** Branch `codex/lrc-blank-compression`. No GPU work, no deployment, no
seal call, no `sealq.sh` invocation. The ledger is byte-identical at 920 rows,
`c9f6cbe5971dd122265eb3f940d6430d1a7cef41f6e76fa8b9932c4cc393528b`, before and after
this wave.

---

## 1. Crash cleanup — what wave 34 left on the floor

Wave 34 was killed by an auth error mid-flight. Its last recorded act was a hand edit
of the retention script. Everything below was found on disk, uncommitted, and every
claim here was checked rather than assumed.

### 1.1 The thing that mattered most: no retention deletion ran

The dead agent's last words were "retention script hand edit", which is the most
dangerous place in this estate to die. It did not delete anything.

- `/mnt/ai_data/neural-forge-matrix/wave34/` **does not exist** on the AI server. No
  wave-34 build, no wave-34 deployment, no wave-34 children directory.
- `/mnt/ai_data/neural-forge-matrix/wave33/children/` contains exactly
  `MX33-GV12L01` and `MX33-GV16L01` — **exactly** wave 33's own `KEEP` set, and nothing
  else. `children-retention-manifest-wave33.json` is present, stamped
  `2026-08-28 06:07:25`, twenty minutes **before** the wave-34 tree was cut at 06:27.
- So the two retained children on disk are wave 33's retention, applied by wave 33.
  Wave 34's retention never ran and could not have: see §1.2.

### 1.2 The retention hand edit was finished, and finished correctly

`run/mx34/mx34_retain.py` is a complete, coherent hand edit and the only one of the
cutter's four `HAND_EDIT` files that got done:

- `KEEP` and `DROP` are both **emptied** — a decision, not a substitution artifact. The
  cutter copies this file verbatim, so wave 33's two keeps and four drops arrived here
  still spelled for wave 33; they were deliberately removed rather than renamed.
- The refusal paragraph is preserved and updated: "the fifth wave to say so in this file".
- Wave-33 policy prose was deleted rather than renamed, per the standing rule.
- Paths and schema retargeted to `wave34`. Zero `mx33`/`wave33`/`MX33` needles remain.
- `--plan` is still the default and `--apply` still requires the plan to have been read.

With an empty `KEEP`/`DROP` and no `wave34/children/` directory on the server, this
script could not have deleted anything even if it had been invoked.

### 1.3 What else was half-done, and the state it is in

`run/mx34/` is a **complete tree cut with three of four hand edits outstanding.** The
cutter's own audit refuses, which is the design working:

```
$ python cut_wave34.py --audit
AUDIT FAILED: mx34_build.py: a wave-33 module reference survived the copy ('mx33_' x9)
AUDIT FAILED: sealq.sh: a wave-33 module reference survived the copy ('mx33_' x1)
AUDIT FAILED: mx34_build.py: BASIS_SCHEMAS_ACCEPTED is missing neural-forge-matrix-wave34-residual-basis-v1
AUDIT FAILED: sealq.sh does not carry this wave's seal marker, so its seal calls are
              last wave's counts on this wave's cell ids
exit 1
```

| file | in `HAND_EDIT` | state | wave-33 needles left |
|---|---|---|---:|
| `mx34_retain.py` | yes | **done** | 0 |
| `mx34_verify_seals.py` | yes | **outstanding** | 3 |
| `mx34_build.py` | yes | **outstanding** | 14 |
| `sealq.sh` | yes | **outstanding** | 3 |
| `run/mx34/README.md` | in `AUTHORED_THIS_WAVE` | **never written** | — |

The tree therefore cannot produce a wave-34 measurement, and the failure mode it is
protected against — last wave's seal counts fired at this wave's cell ids — is exactly
the one the guard names.

### 1.4 The extension-bank-v2 authoring is finished work, and it verifies

This is the part that would have been easy to throw away and should not be.
`research/lfm2_extension_bank_prompts_v2.json` plus `run/mx34/mx34_extrubric2.py` and
`run/mx34/mx34_extbank2.py` are complete and self-verifying. Run this wave, read-only:

```
$ python mx34_extrubric2.py --bank <research/lfm2_extension_bank_prompts_v2.json> --self-test
self-test: 30 prompt rules over 15 lanes, two per lane, A/B staggered; no prompt
satisfies its own rule; no rule grades form alone; the degeneracy veto is closed
exit 0

$ python mx34_extbank2.py --bank <v2> --sealed-pack <lfm2_semantic_whole_response_prompts_v1.json>
                          --v1-bank <lfm2_extension_bank_prompts_v1.json>
                          --paraphrase <lfm2_paraphrase_triplets_v1.json> --out <scratch>
self-test: ... (as above)
reachability: 30/30 reference answers hold
ngram_disjointness_n 7 -- collisions: sealed_pack 0, extension_v1 0, paraphrase_triplets 0
lanes_covered 15, prompts_per_lane [2], difficulty_staggered_A_B true
prompts_carrying_an_external_reference: []   prompts_with_an_unexplained_assistant_turn: []
exit 0
bank sha256   340694a64f7e886ecf1576dd9245a1445d8e763036d842f3730c4c8ab5c9c592
rubric sha256 fc7564d7f0ebc9344ee02eb359d39f6d13edc5127df2915c70328a3ce540cfe9
```

Thirty prompts, two per frozen lane, A/B difficulty staggered, zero seven-gram overlap
with any of the three existing prompt surfaces, and every reference answer reachable.
Wave 33 retired the extension-v1 verdict string because the instrument could not price a
one-case margin; this v2 bank is the honest attempt at a replacement and it is intact.

### 1.5 What was done with the debris

**Nothing was discarded.** Every uncommitted wave-34 path was checked for coherence
first and all of it passed:

- `run/mx34/` — 55 files, and every Python module in it compiles clean under
  `py_compile`. Committed **with the audit still failing**, and this note records exactly
  which three hand edits are outstanding so a future wave-34 picks it up knowing the
  state rather than mistaking it for a finished tree.
- `research/lfm2_extension_bank_prompts_v2.json` — committed; verified as above.
- `reports/matrix-campaign-20260810/ledger.jsonl` — untouched, hash re-verified.
- **The Codex lane was not touched.** `.codex-test-runs/`, the modified
  `src/neural_forge/hf_lfm2_circuit_trace_cache.py` and every other peer-owned path in
  the working tree were left exactly as found. Nothing outside the explicit paths listed
  in §6 was staged.

---

## 2. The re-read: methodology

### 2.1 The standard being applied

Cyril's canonical statement, applied to **assessment only**. No sealed artifact was
modified; no measurement was re-run; the frozen review protocol is untouched.

> [Cyril quote withheld]

Every one of the frozen fifteen lanes goes in exactly one box:

- **intelligence signal** — sealed Useful or Partial, **or** a sealed miss whose text
  contains at least one correct, specific, usable element of what the lane asked for, in
  the child's own words.
- **artifact-miss** — empty, pure loop, cap-ride with no unique content, indiscriminate
  prompt copy, multilingual token salad. Never counts against intelligence.
- **content-miss** — readable, on-register output that is wrong, vacuous, deflecting, or
  a restatement with no answer in it. The only miss that is about intelligence.

Degeneracy vetoes are lifted for assessment and stay recorded as diagnostics. The
robustness/restatement column is a diagnostic, not a gate.

### 2.2 How the reading was actually done

1. **Parse.** Every `READ-*.txt` child dossier in
   `reports/matrix-campaign-20260810/wave*/human-review/` was parsed into per-case
   records: lane, status, char count, collapsed count, verbatim text. 191 dossiers parsed
   at 16 cases each (3,056 case blocks); 35 more exist and are zero bytes; the three
   wave-14 files are earlier duplicates of `ARCH.W14FF320/640/1280`, which are re-read
   from their wave-15 dossiers instead.
2. **Join.** To the 224 per-case verdict JSONs and to the newest sealed ledger row per
   `cell_id` (net ratio, stored/parent parameter counts, sealed U/P, verdict, notes).
3. **Bucket the artifacts mechanically**, because that part is mechanical: `chars == 0`;
   `collapsed <= 80`; novel-content under 60 characters after dropping repeated 6-grams
   and prompt 6-grams; unique-word count under 25 with repetition over 0.7; prompt
   coverage over 0.70 with under 25 novel tokens; non-ASCII character fraction over 0.04.
4. **Screen the rest with an answer key** — 16 lane-specific checks written against the
   frozen prompts' own `context` fields (18 per box and 4 left over; Mira → Theo → Jun →
   Sal; Saturday / 4:15 p.m. / north gate; Supplier B plus the expansion slip;
   `values[start:]`; seen-set dedup; peanut-free plus vegan; Friday / Monday / tracking;
   axial tilt plus the distance correction; and so on). The screen's recall was measured
   against the sealed Useful lanes as a control: 21/21 on retrieval, 71/71 on
   summarisation, 60/61 on instruction-following, 41/41 on uncertainty, 24/24 on email.
5. **Hand-read.** Every sealed-miss lane still carrying readable content was read, with
   repetition collapsed so the content behind the artifact is visible — 1,098 lanes across
   188 children, worked deepest-first in seven ratio bands. The screen aimed the reading;
   it never made a decision.

### 2.3 Two rules that decided the hard cases

- **Purposive reuse of the prompt is signal; indiscriminate reproduction is artifact.**
  `ARCH.V8CL896`'s summarisation lane reproduces the whole prompt, instruction line
  included, and therefore contains "The team selected Supplier B". The answer key fires;
  the re-read scores it an artifact. A child that selects the risk sentence and frames it
  as the risk has answered.  Together with the self-audit in §2.4 this rejected 129 of
  the screen's 311 hits.
- **Looping after content is free.** `ARCH.GV12L01` gives three practical actions and then
  repeats one of them eighteen times into the cap. The three actions are the finding.

### 2.4 The self-audit, and the 29 calls it reversed

The band reading in step 5 worked from a collapsed 240–420 character view. That is the
whole response for the short lanes — retrieval, summarisation, arithmetic, ordering,
seasons, email. It is **not** the whole response for `multi-turn-correction`,
`planning-dependencies`, `coding-deduplicate` and `creative-continuity`, where a correct
header runs first and the body either contradicts it or degenerates to nothing.

Every signal call on those four lanes was re-read at full length before publication.
**29 were wrong and were demoted to content-miss.** The rule did not change; it was
applied to text the first pass had not seen.

- **Nineteen multi-turn calls.** `ARCH.V22` opens "SNACK PLAN FOR 8 GUESTS (VEGAN +
  SEVERE PEANUT ...)" and then puts **peanut butter** in the first option. `ARCH.V19X09`
  labels an option nut-free and fills it with "a nut-based granola ... with a small amount
  of nut oil". `ARCH.MASK03` asserts that protein "prevent[s] severe allergy reactions".
  `ARCH.B75`, `ARCH.BP64`, `ARCH.V19X13G01/G03`, `ARCH.V32X13G/G03/GL01/GL1` and
  `ARCH.V20X13G03` all name the constraint in the header and then emit
  "protein-based protein-based" with no food in it.
- **Five planning calls** named the five stages and then never ordered them, or ordered
  "create backups for backups using DNS" four times (`ARCH.V19X13X01`, `ARCH.V20X13X01`,
  `ARCH.BM25`, `ARCH.CL1792`, `ARCH.CL1792LG`).
- **Four creative calls** were a single sentence of premise with no dialogue and no
  developed beat, which is not a scene (`ARCH.B25`, `ARCH.SR24`, `ARCH.SR64`, `ARCH.V10`).
- **One coding call** had no first-seen guard and gave the function's complexity as O(1)
  (`ARCH.V8X13GL1`).

This cost the frontier two rungs of depth at the top: 14/15 moved from `ARCH.V19X09` at
0.742049x to `ARCH.V12` at 0.762614x, and 10/15 from `ARCH.V8X13GL01` at 0.691832x to
`ARCH.V12X13` at 0.710093x. It is recorded rather than quietly fixed, because a re-read
that only ever moves numbers in one direction is not a re-read.

### 2.5 What the re-read is not

It is a hand reading of **one wording** of each ask, exactly like the sealed fifteen.
Wave 33's robustness probe showed the sealed ordering does not survive rewording. Nothing
here re-tests that and the caveat applies with the same force. A lane counted as signal on
one correct element is not a lane the parent would pass; the parent reads 12/15 under the
sealed rule. This is a question about retention, not about parity.

---

## 3. Results

Full tables in `reports/matrix-campaign-20260810/ACTUAL_MATRIX_2026-08-28.md`.

**The artifact-versus-content split**, across 188 children and 2,835 scored lanes:

| | lanes | share of all lanes | share of the sealed misses |
|---|---:|---:|---:|
| intelligence signal | **1,076** | 38.0% | — |
| — sealed Useful/Partial | 894 | 31.5% | — |
| — recovered by this re-read | 182 | 6.4% | **9.4%** |
| artifact-miss | **846** | 29.8% | **43.6%** |
| content-miss | **913** | 32.2% | **47.0%** |

Broken out: 742 loop-or-cap-ride lanes with under ~80 characters of unique text, 85 empty,
16 whole-response prompt echoes, 3 token salad.

**Forty-four percent of everything this campaign scored as a behavioural failure is a
compression artifact carrying no evidence about intelligence in either direction.**

**The frontier moves at eleven of the fifteen rungs.** It does not move at 0, 3, 5 and 8,
and the eight rung is the notable one: `ARCH.V16CL1792` at 0.637073x reads eight under both
standards, so the campaign's most-argued-about row is the one the two readings agree on.
The largest move is at four of fifteen — 0.600552x becomes **0.483001x**, a further 0.118
of stored ratio off the parent at the same depth.

**Eight of the eleven children ever reviewed below half the parent hold intelligence**, and
two of those eight were sealed at zero.

**Three children in the whole campaign are honest content deaths** — readable, fluent,
on-register and empty: `ARCH.KKTD` (0.483x), `ARCH.V8CL896` (0.433x) and `ARCH.X12`
(0.952x). That last one is the control that matters: content death is not a function of
depth. Twenty-two more children score zero because nothing legible came out of them at
all, which is a statement about the decode and not about intelligence.

### 3.1 The reversals worth naming

| child | net | sealed | intel | the response |
|---|---:|---:|---:|---|
| `ARCH.V16X13G` | 0.728353 | 5 | **10** | `17 x 6 = 102` and `102 - 8 = 94` both correct; the seen-set dedup loop; a real vendor email; and a lighthouse scene with the line *"The foghorn is not just a fireshore. It is a memory."* |
| `ARCH.V12` | 0.762614 | 11 | **14** | `17 x 6 = 102`, the dedup loop, and a scene whose growl makes the keeper's breath hitch — the deepest fourteen in the campaign |
| `ARCH.V19X09` | 0.742049 | 10 | **13** | 102 correct, set-based dedup with O(n) named, a backup-first migration with a test step, and zero artifact lanes |
| `ARCH.BP64` | 0.742049 | 10 | **13** | three valid derived orderings, the dedup loop, and a scene with *"We're not here for the light."* |
| `ARCH.V19CL1792X13` | 0.608519 | 2 | **6** | 4:15 p.m. off the note, O(n) stated correctly, a real email naming the Monday install, three usable prioritisation actions |
| `ARCH.V19X13X09` | 0.689528 | 5 | **8** | *"This cycle is **not** due to Earth being closer to the Sun"* — the exact correction the seasons lane is built to test |
| `ARCH.CL640` | 0.640497 | **0** | **1** | pulls 4:15 p.m. and 7:30 a.m. off the note. Seven of its other lanes are pure loop |
| `ARCH.D3` | 0.842437 | **0** | **1** | *"The team's decision to B, a 11% cost ... the risk is a potential 1-week expansion"* — decision, price delta and risk inside a wrapper of numeric junk |
| `ARCH.X05T` | 0.947479 | **0** | **2** | asks the vendor to confirm the Friday delivery; names the factors it would need before concluding |
| `ARCH.KKTS` | 0.483001 | **0** | **2** | sealed dead by degeneracy; orders a migration backups → TLS → verify and names the four withheld quantities |
| `ARCH.V19X13CL896` | 0.452734 | **0** | **1** | at 0.4527x it compresses the supplier update in its own words and lands the one-week expansion slip |

### 3.2 Where the sealed review was right

`ARCH.KKTD` at 0.483001x is the clearest content death in the estate: three artifact lanes,
twelve readable ones, zero signal. It answers the summarisation prompt with "The team has
to contact the following email address: https://www.com". Fluent, on-register, empty. The
canonical standard does not rescue it and should not.

---

## 4. Per-child re-read log

188 children, deepest first. `intel` is signal/15. `art` and `con` are the artifact-miss and
content-miss counts. The last column names only the lanes this re-read flipped from a sealed
miss to signal; every other signal lane in the count is a sealed Useful or Partial.

| child | wave | net | sealed | intel | art | con | lanes flipped by the re-read |
|---|---:|---:|---:|---:|---:|---:|---|
| `ARCH.V8CL896` | 24 | 0.432784 | 0 | **0** | 7 | 8 | — |
| `ARCH.V19X13CL896` | 28 | 0.452734 | 0 | **1** | 10 | 4 | summ |
| `ARCH.W512` | 19 | 0.463481 | 0 | **0** | 15 | 0 | — |
| `ARCH.V16CL896` | 23 | 0.469305 | 1 | **2** | 5 | 8 | unc |
| `ARCH.V18CL896` | 28 | 0.478436 | 2 | **2** | 6 | 7 | — |
| `ARCH.V19CL896` | 28 | 0.483001 | 2 | **2** | 5 | 8 | — |
| `ARCH.V19CL896G03` | 28 | 0.483001 | 3 | **4** | 7 | 4 | retr |
| `ARCH.KKTD` | 29 | 0.483001 | 0 | **0** | 3 | 12 | — |
| `ARCH.KKTS` | 29 | 0.483001 | 0 | **2** | 9 | 4 | plan, unc |
| `ARCH.V20CL896` | 27 | 0.487566 | 2 | **2** | 5 | 8 | — |
| `ARCH.V20CL896G03` | 28 | 0.487566 | 3 | **4** | 7 | 4 | retr |
| `ARCH.V32CL896` | 27 | 0.542347 | 2 | **2** | 7 | 6 | — |
| `ARCH.A552` | 15 | 0.552023 | 0 | **0** | 15 | 0 | — |
| `ARCH.V19CL1280` | 30 | 0.554901 | 1 | **1** | 7 | 7 | — |
| `ARCH.V19CL1408` | 31 | 0.578868 | 1 | **1** | 6 | 8 | — |
| `ARCH.R5` | 16 | 0.579946 | 0 | **0** | 15 | 0 | — |
| `ARCH.W14FF320` | 15 | 0.580580 | 0 | **0** | 15 | 0 | — |
| `ARCH.V4CL1792` | 32 | 0.582292 | 1 | **2** | 2 | 11 | summ |
| `ARCH.W640` | 19 | 0.590763 | 0 | **0** | 15 | 0 | — |
| `ARCH.V6CL1792` | 32 | 0.591422 | 3 | **3** | 1 | 11 | — |
| `ARCH.V8CL1792` | 31 | 0.600552 | 5 | **7** | 2 | 6 | 3act, ram |
| `ARCH.V8CL1792M` | 32 | 0.600552 | 3 | **4** | 2 | 9 | retr |
| `ARCH.V19CL1536` | 31 | 0.602835 | 2 | **2** | 7 | 6 | — |
| `ARCH.R4` | 16 | 0.603922 | 0 | **0** | 15 | 0 | — |
| `ARCH.V9CL1792` | 31 | 0.605117 | 6 | **7** | 1 | 7 | 3act |
| `ARCH.V9CL1792M` | 32 | 0.605117 | 2 | **4** | 1 | 10 | retr, unc |
| `ARCH.V19CL1792X13` | 31 | 0.608519 | 2 | **6** | 5 | 4 | 3act, retr, code, email |
| `ARCH.V12CL1792` | 31 | 0.618813 | 7 | **7** | 0 | 8 | — |
| `ARCH.V12CL1792M` | 32 | 0.618813 | 4 | **4** | 1 | 10 | — |
| `ARCH.GV12L01` | 33 | 0.618813 | 6 | **6** | 1 | 8 | — |
| `ARCH.GV12L03` | 33 | 0.618813 | 7 | **7** | 1 | 7 | — |
| `ARCH.GV12L10` | 33 | 0.618813 | 7 | **7** | 1 | 7 | — |
| `ARCH.V16W896X13` | 23 | 0.623329 | 1 | **2** | 10 | 3 | summ |
| `ARCH.V19CL1664` | 31 | 0.626802 | 3 | **3** | 3 | 9 | — |
| `ARCH.V16CL1792` | 31 | 0.637073 | 8 | **8** | 1 | 6 | — |
| `ARCH.V16BMV3` | 32 | 0.637073 | 3 | **6** | 1 | 8 | retr, email, unc |
| `ARCH.V16CL1792M` | 32 | 0.637073 | 3 | **6** | 0 | 9 | 3act, arith, unc |
| `ARCH.GV16L01` | 33 | 0.637073 | 8 | **8** | 0 | 7 | — |
| `ARCH.GV16L03` | 33 | 0.637073 | 7 | **7** | 0 | 8 | — |
| `ARCH.GV16L10` | 33 | 0.637073 | 7 | **7** | 1 | 7 | — |
| `ARCH.V20X13W896` | 26 | 0.639307 | 2 | **3** | 11 | 1 | retr |
| `ARCH.CL640` | 15 | 0.640497 | 0 | **1** | 7 | 7 | retr |
| `ARCH.CT640` | 15 | 0.640497 | 0 | **0** | 15 | 0 | — |
| `ARCH.S640` | 15 | 0.640497 | 0 | **0** | 10 | 5 | — |
| `ARCH.W14FF640` | 15 | 0.640497 | 0 | **0** | 11 | 4 | — |
| `ARCH.R3` | 16 | 0.641584 | 0 | **0** | 15 | 0 | — |
| `ARCH.V18CL1792` | 31 | 0.646204 | 7 | **7** | 1 | 7 | — |
| `ARCH.V19CL1792` | 30 | 0.650769 | 7 | **7** | 0 | 8 | — |
| `ARCH.V19CL1792M` | 30 | 0.650769 | 6 | **6** | 0 | 9 | — |
| `ARCH.V19CL1792ME` | 31 | 0.650769 | 3 | **5** | 1 | 9 | email, plan |
| `ARCH.V19CL1792MV` | 31 | 0.650769 | 5 | **5** | 2 | 8 | — |
| `ARCH.W704` | 20 | 0.656116 | 0 | **0** | 15 | 0 | — |
| `ARCH.W704AW25` | 20 | 0.656116 | 0 | **0** | 15 | 0 | — |
| `ARCH.R2` | 16 | 0.658704 | 0 | **0** | 15 | 0 | — |
| `ARCH.V18X13X01KV6JQ` | 27 | 0.664419 | 0 | **0** | 9 | 6 | — |
| `ARCH.V16W896` | 23 | 0.667288 | 2 | **3** | 10 | 2 | 3act |
| `ARCH.W720` | 22 | 0.672632 | 0 | **0** | 14 | 1 | — |
| `ARCH.V24CL1792` | 31 | 0.673594 | 7 | **7** | 0 | 8 | — |
| `ARCH.V18W896` | 27 | 0.675277 | 2 | **3** | 8 | 4 | 3act |
| `ARCH.V26CL1792` | 31 | 0.682724 | 7 | **7** | 0 | 8 | — |
| `ARCH.V20W896` | 26 | 0.683266 | 2 | **3** | 8 | 4 | 3act |
| `ARCH.V18X13X01` | 27 | 0.684962 | 3 | **4** | 3 | 8 | 3act |
| `ARCH.V18X13X01` | 28 | 0.684962 | 3 | **4** | 3 | 8 | 3act |
| `ARCH.V18X13X01G01` | 28 | 0.684962 | 3 | **4** | 3 | 8 | 3act |
| `ARCH.V18X13X01G03` | 28 | 0.684962 | 4 | **5** | 3 | 7 | arith |
| `ARCH.V18X13X01G10` | 28 | 0.684962 | 4 | **4** | 2 | 9 | — |
| `ARCH.CL896` | 15 | 0.688431 | 4 | **6** | 5 | 4 | retr, plan |
| `ARCH.W736` | 21 | 0.689220 | 1 | **1** | 12 | 2 | — |
| `ARCH.W736LS` | 21 | 0.689220 | 0 | **0** | 14 | 1 | — |
| `ARCH.R1` | 16 | 0.689528 | 0 | **0** | 12 | 3 | — |
| `ARCH.V19X13X01` | 28 | 0.689528 | 3 | **4** | 3 | 8 | 3act |
| `ARCH.V19X13X01G03` | 28 | 0.689528 | 4 | **4** | 3 | 8 | — |
| `ARCH.V19X13X09` | 28 | 0.689528 | 5 | **8** | 1 | 6 | seas, arith, plan |
| `ARCH.V8X13` | 24 | 0.691832 | 6 | **7** | 1 | 7 | code |
| `ARCH.V8X13G` | 24 | 0.691832 | 4 | **7** | 1 | 7 | retr, summ, code |
| `ARCH.V8X13GL01` | 24 | 0.691832 | 7 | **9** | 0 | 6 | arith, code |
| `ARCH.V8X13GL1` | 24 | 0.691832 | 6 | **7** | 1 | 7 | order |
| `ARCH.V8X13GL03` | 25 | 0.691832 | 7 | **9** | 0 | 6 | retr, code |
| `ARCH.V20X13X07` | 26 | 0.694093 | 5 | **6** | 2 | 7 | code |
| `ARCH.V20X13X01` | 27 | 0.694093 | 3 | **4** | 3 | 8 | 3act |
| `ARCH.V20X13X01G03` | 27 | 0.694093 | 1 | **3** | 3 | 9 | 3act, plan |
| `ARCH.V9X13` | 27 | 0.696398 | 6 | **8** | 0 | 7 | 3act, code |
| `ARCH.V12X13` | 25 | 0.710093 | 7 | **10** | 0 | 5 | ram, code, email |
| `ARCH.V18X13KV6JQ` | 27 | 0.716941 | 2 | **2** | 7 | 6 | — |
| `ARCH.W768` | 19 | 0.722610 | 3 | **3** | 11 | 1 | — |
| `ARCH.W768AW100` | 20 | 0.722610 | 0 | **0** | 15 | 0 | — |
| `ARCH.W768AW25` | 20 | 0.722610 | 0 | **0** | 14 | 1 | — |
| `ARCH.W768EV` | 20 | 0.722610 | 1 | **1** | 14 | 0 | — |
| `ARCH.W768LS` | 20 | 0.722610 | 3 | **3** | 10 | 2 | — |
| `ARCH.W768RG` | 20 | 0.722610 | 0 | **0** | 10 | 5 | — |
| `ARCH.W768ST` | 20 | 0.722610 | 2 | **2** | 12 | 1 | — |
| `ARCH.MM768` | 21 | 0.722610 | 0 | **0** | 14 | 1 | — |
| `ARCH.BX13` | 16 | 0.724929 | 1 | **4** | 1 | 10 | summ, email, plan |
| `ARCH.R0` | 16 | 0.724929 | 0 | **0** | 12 | 3 | — |
| `ARCH.V20X13KV6JQ` | 27 | 0.726071 | 2 | **3** | 5 | 7 | retr |
| `ARCH.V20X13KV6JQG03` | 27 | 0.726071 | 2 | **3** | 7 | 5 | retr |
| `ARCH.V16X13` | 23 | 0.728353 | 5 | **7** | 0 | 8 | ram, code |
| `ARCH.V16X13G` | 24 | 0.728353 | 5 | **10** | 0 | 5 | arith, ram, code, email, creat |
| `ARCH.V16X13GL01` | 24 | 0.728353 | 6 | **8** | 2 | 5 | code, email |
| `ARCH.V16X13GL1` | 24 | 0.728353 | 6 | **9** | 2 | 4 | order, code, plan |
| `ARCH.V16X13GL03` | 25 | 0.728353 | 7 | **10** | 2 | 3 | code, email, plan |
| `ARCH.V32W896` | 25 | 0.731199 | 3 | **3** | 7 | 5 | — |
| `ARCH.V22X13KV6JQ` | 27 | 0.735201 | 2 | **2** | 5 | 8 | — |
| `ARCH.V18X07` | 27 | 0.737484 | 9 | **10** | 0 | 5 | seas |
| `ARCH.V18X07G03` | 27 | 0.737484 | 8 | **9** | 0 | 6 | plan |
| `ARCH.V18X09` | 27 | 0.737484 | 10 | **12** | 0 | 3 | arith, code |
| `ARCH.V18X09G03` | 27 | 0.737484 | 9 | **12** | 0 | 3 | arith, code, plan |
| `ARCH.V18X13` | 27 | 0.737484 | 10 | **12** | 0 | 3 | seas, email |
| `ARCH.V18X13G03` | 27 | 0.737484 | 10 | **11** | 1 | 3 | code |
| `ARCH.V19X09` | 28 | 0.742049 | 10 | **13** | 0 | 2 | arith, code, plan |
| `ARCH.V19X13` | 28 | 0.742049 | 10 | **13** | 0 | 2 | seas, order, email |
| `ARCH.V19X13G01` | 28 | 0.742049 | 11 | **12** | 1 | 2 | code |
| `ARCH.V19X13G03` | 28 | 0.742049 | 11 | **12** | 1 | 2 | code |
| `ARCH.V19X13G10` | 28 | 0.742049 | 10 | **12** | 2 | 1 | code, creat |
| `ARCH.B25` | 29 | 0.742049 | 10 | **12** | 0 | 3 | order, code |
| `ARCH.B50` | 29 | 0.742049 | 10 | **12** | 2 | 1 | code, creat |
| `ARCH.B75` | 29 | 0.742049 | 11 | **12** | 1 | 2 | code |
| `ARCH.BP64` | 29 | 0.742049 | 10 | **13** | 0 | 2 | order, code, creat |
| `ARCH.MASK03` | 29 | 0.742049 | 9 | **11** | 0 | 4 | 3act, code |
| `ARCH.SR24` | 30 | 0.742049 | 10 | **12** | 1 | 2 | order, code |
| `ARCH.SR64` | 30 | 0.742049 | 10 | **11** | 1 | 3 | code |
| `ARCH.V8` | 24 | 0.744354 | 4 | **7** | 0 | 8 | order, summ, multi |
| `ARCH.V20X13` | 26 | 0.746614 | 10 | **13** | 1 | 1 | seas, order, email |
| `ARCH.V20X13G03` | 26 | 0.746614 | 11 | **12** | 1 | 2 | code |
| `ARCH.V20X07` | 27 | 0.746614 | 9 | **10** | 0 | 5 | seas |
| `ARCH.V20X09` | 27 | 0.746614 | 9 | **12** | 0 | 3 | arith, code, plan |
| `ARCH.V20X09G03` | 27 | 0.746614 | 8 | **11** | 0 | 4 | arith, code, plan |
| `ARCH.V9` | 26 | 0.748919 | 9 | **11** | 0 | 4 | code, multi |
| `ARCH.V10` | 25 | 0.753484 | 10 | **11** | 0 | 4 | code |
| `ARCH.W14FF1280` | 15 | 0.760331 | 5 | **5** | 4 | 6 | — |
| `ARCH.CL1280` | 30 | 0.760331 | 4 | **5** | 4 | 6 | email |
| `ARCH.V12` | 25 | 0.762614 | 11 | **14** | 0 | 1 | arith, code, creat |
| `ARCH.B1536` | 16 | 0.769439 | 0 | **0** | 13 | 2 | — |
| `ARCH.W1536X13` | 16 | 0.769439 | 4 | **4** | 6 | 5 | — |
| `ARCH.V14` | 25 | 0.771744 | 9 | **13** | 0 | 2 | seas, code, plan, multi |
| `ARCH.V16` | 23 | 0.780874 | 9 | **13** | 0 | 2 | arith, code, plan, creat |
| `ARCH.CL1408` | 31 | 0.784298 | 3 | **5** | 6 | 4 | retr, email |
| `ARCH.V17` | 27 | 0.785440 | 11 | **14** | 0 | 1 | code, plan, multi |
| `ARCH.Q01050913` | 19 | 0.789916 | 0 | **0** | 12 | 3 | — |
| `ARCH.V18` | 26 | 0.790005 | 13 | **14** | 0 | 1 | creat |
| `ARCH.V19` | 27 | 0.794570 | 14 | **14** | 0 | 1 | — |
| `ARCH.V19MXO3` | 29 | 0.794570 | 3 | **5** | 2 | 8 | 3act, arith |
| `ARCH.V19MXP2` | 29 | 0.794570 | 2 | **3** | 4 | 8 | ram |
| `ARCH.V20` | 25 | 0.799135 | 13 | **14** | 0 | 1 | multi |
| `ARCH.V32X13` | 25 | 0.801395 | 10 | **13** | 1 | 1 | order, ram, creat |
| `ARCH.V32X13G` | 25 | 0.801395 | 10 | **12** | 0 | 3 | code, creat |
| `ARCH.V32X13GL01` | 25 | 0.801395 | 9 | **11** | 1 | 3 | code, plan |
| `ARCH.V32X13GL1` | 25 | 0.801395 | 10 | **12** | 1 | 2 | code, creat |
| `ARCH.V32X13G03` | 26 | 0.801395 | 10 | **12** | 0 | 3 | code, plan |
| `ARCH.V22` | 26 | 0.808265 | 13 | **13** | 0 | 2 | — |
| `ARCH.CL1536` | 31 | 0.808265 | 4 | **4** | 9 | 2 | — |
| `ARCH.CL1792X13` | 31 | 0.813949 | 4 | **4** | 5 | 6 | — |
| `ARCH.W896X13` | 19 | 0.815064 | 1 | **1** | 12 | 2 | — |
| `ARCH.W896LSX13` | 21 | 0.815064 | 2 | **2** | 11 | 2 | — |
| `ARCH.CL1664` | 31 | 0.832232 | 5 | **5** | 1 | 9 | — |
| `ARCH.W896KV6JS` | 19 | 0.841047 | 2 | **2** | 12 | 1 | — |
| `ARCH.W896LSKV6JQ` | 22 | 0.841047 | 1 | **1** | 13 | 1 | — |
| `ARCH.D3` | 16 | 0.842437 | 0 | **1** | 14 | 0 | summ |
| `ARCH.V32` | 24 | 0.853916 | 11 | **14** | 0 | 1 | ram, code, creat |
| `ARCH.CL1792` | 30 | 0.856199 | 7 | **7** | 0 | 8 | — |
| `ARCH.CL1792LG` | 30 | 0.856199 | 7 | **7** | 0 | 8 | — |
| `ARCH.CL1792M` | 30 | 0.856199 | 8 | **9** | 0 | 6 | 3act |
| `ARCH.CL1792ME` | 31 | 0.856199 | 2 | **3** | 2 | 10 | seas |
| `ARCH.CL1792MV` | 31 | 0.856199 | 7 | **7** | 1 | 7 | — |
| `ARCH.BM25` | 32 | 0.856199 | 6 | **7** | 0 | 8 | debug |
| `ARCH.BM50` | 32 | 0.856199 | 4 | **4** | 0 | 11 | — |
| `ARCH.BM50N` | 32 | 0.856199 | 0 | **0** | 15 | 0 | — |
| `ARCH.BM75` | 32 | 0.856199 | 2 | **4** | 1 | 10 | arith, email |
| `ARCH.BMS75` | 32 | 0.856199 | 5 | **6** | 1 | 8 | arith |
| `ARCH.BMV3` | 32 | 0.856199 | 7 | **7** | 0 | 8 | — |
| `ARCH.BV50` | 32 | 0.856199 | 4 | **5** | 0 | 10 | email |
| `ARCH.CL1792S` | 32 | 0.856199 | 4 | **4** | 1 | 10 | — |
| `ARCH.W896` | 19 | 0.859023 | 4 | **4** | 5 | 6 | — |
| `ARCH.W896LS` | 21 | 0.859023 | 3 | **3** | 8 | 4 | — |
| `ARCH.MM896` | 22 | 0.859023 | 2 | **2** | 12 | 1 | — |
| `ARCH.D2` | 16 | 0.894958 | 2 | **4** | 6 | 5 | retr, plan |
| `ARCH.D2B` | 16 | 0.894958 | 6 | **6** | 1 | 8 | — |
| `ARCH.X13KV6` | 16 | 0.926936 | 6 | **7** | 1 | 7 | retr |
| `ARCH.HD16` | 15 | 0.938368 | 0 | **0** | 13 | 2 | — |
| `ARCH.X05A` | 15 | 0.947479 | 6 | **6** | 4 | 5 | — |
| `ARCH.X05T` | 15 | 0.947479 | 0 | **2** | 6 | 7 | email, unc |
| `ARCH.X01` | 16 | 0.947479 | 9 | **9** | 2 | 4 | — |
| `ARCH.X03` | 16 | 0.947479 | 6 | **6** | 5 | 4 | — |
| `ARCH.X07` | 16 | 0.947479 | 11 | **11** | 2 | 2 | — |
| `ARCH.X09` | 16 | 0.947479 | 10 | **11** | 0 | 4 | seas |
| `ARCH.X11` | 16 | 0.947479 | 9 | **10** | 0 | 5 | multi |
| `ARCH.X13A` | 16 | 0.947479 | 13 | **13** | 1 | 1 | — |
| `ARCH.X12` | 16 | 0.952057 | 0 | **0** | 6 | 9 | — |
| `ARCH.KV6` | 16 | 0.979457 | 5 | **7** | 3 | 5 | plan, unc |

`ARCH.V18X13X01` appears twice: it was reviewed in wave 27 and again in wave 28 from two
separately-sealed dossiers, and both rows are kept rather than merged.

## 5. Rows with no re-read

102 whole-model children carry a sealed behaviour row and were not re-read. Under the
canonical standard a sealed Useful or Partial always counts as signal, so for these rows
the sealed count is a **lower bound** on the intelligence score, never an upper one. No
frontier claim in this wave rests on any of them. They are enumerated in
`ACTUAL_MATRIX_2026-08-28.md` §5, split into the 35 whose `READ-` file exists at zero bytes
(dossier-lost to retention in waves 17–18; the per-case JSON survives for every one) and
the 67 `COMP.MA.*`/`COMP.MB.*` composed children from waves 6–13 that never got a per-child
dossier in the repo.

## 6. What this wave changed and did not change

Changed:

- `reports/matrix-campaign-20260810/ACTUAL_MATRIX_2026-08-28.md` — new.
- `reports/matrix-campaign-20260810/CAMPAIGN_SCORECARD.md` — re-headlined under the
  canonical standard, with an `intel` column added beside the sealed `review` column and
  **every sealed number preserved unedited**. Backed up first to
  `reports/matrix-campaign-20260810/backup/CAMPAIGN_SCORECARD.md.pre-wave34R-20260828T072854`
  (sha256 `cd96ffedffa940fb76324060ae4929e679b51cf6fb35a8949368f9456a4d48f6`).
- `NOTES-MX-AK.md` — this file.
- `reports/matrix-campaign-20260810/run/mx34/` and
  `research/lfm2_extension_bank_prompts_v2.json` — wave 34's uncommitted work, verified
  and committed rather than discarded.

Not changed:

- `reports/matrix-campaign-20260810/ledger.jsonl` — 920 rows,
  `c9f6cbe5971dd122265eb3f940d6430d1a7cef41f6e76fa8b9932c4cc393528b`. No seal ran;
  `sealq.sh` was never invoked; this is an assessment layer and not a measurement.
- `RESULTS_MATRIX_2026-08.md` — left as sealed. It is the frozen-fifteen join, and the
  canonical reading lives in its own file beside it rather than overwriting it.
- Every per-wave sealed artifact, receipt, verdict and dossier — read-only throughout.
- The Codex lane.
