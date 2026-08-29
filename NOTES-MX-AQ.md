# NOTES-MX-AQ — wave 39: the X13 ladder holds one at every rung and the lane rotates

**2026-08-29.** Branch `codex/lrc-blank-compression`. Two children built, sealed
and read twice; no reproduction control was needed because both sources are one
CPU pass off a body wave 35 built, sealed and retains. Ledger 988 → **992** rows,
`dfe27e468f09fc1907b32128090aac6c2cb05ec5946cf24a08a3a999f55d29cc`, byte-prefix
verified against the pre-wave backup on the share **and again in the repository
before the copy-back**. Two deployments, and the accept list cost neither of them
— fourth consecutive wave.

**Both pre-registrations hit. Both parameter counts landed to the byte. No
frontier rung moved, and that was predicted in writing before the build.** §2.

---

## 1. Wave 39 made ONE edit, and it is a log string in a file the wave never runs

Waves 37 and 38 made no behavioural edit at all. Wave 39 made one, and the whole
of it is this:

```
-    print(f"[mx32] {message}", flush=True)
+    print(f"[mx39] {message}", flush=True)
```

`mx39_build.py`'s logger has printed `[mx32]` since wave 32's cut, because the
literal sits inside an f-string and the substitution list carries identifiers,
paths and schema strings only. Wave 17 shipped the same defect as `[mx16]`.
`NOTES-MX-AP` §1 calls it the seventh issue, cosmetic, and waves 37 and 38 both
declined it on the stated ground that an edit in a no-edit wave is itself an
edit.

**Wave 39 could afford it because wave 39 does not run that file.** The build
list is two `mx39_vocab.py` vocabulary passes on `MX35-CL896X13` — no body build,
no conditioned capture, no solve — so `mx39_build.py` is carried, retargeted,
asserted and never executed. The repair goes under no seal and can move no
number. It is pinned in both directions and **verified by calling the function**:

> `log() emitted: '[mx39] log-prefix probe\n'`

Eight waves, closed for the cost of one line and one import.

### 1.1 The first version of that pin refused on this cut's own header, one wave after the lesson was written down

`NOTES-MX-AP` §10 item 6 says *assert code lines at their INDENTED form*, because
wave 38's audit failed on a docstring quoting the three lines it pinned. Wave 39
pinned the **bare** strings `[mx32]` at zero and `[mx16]` at one — and the
wave-39 header explaining the repair names both literals four times between them,
so the audit refused on this cutter's own prose.

**Same trap, different file, one wave later, and it was the audit that caught it
rather than a reader.** The pins are now the code's own line in each of its three
forms. Wave 18's docstring recording `[mx16]` is untouched, because rule 1 does
not rewrite prose that records history.

### 1.2 The wave's own audit addition is the vocabulary gather, which is overdue

Every cut since wave 31 asserts the lines that wave's build list depends on: wave
36 the composed width × MLP site, wave 37 the live-mask repair, wave 38 the
descending deletion order and the two module re-index assignments. **Wave 39's
build list is two `mx39_vocab.py` passes and nothing else — and in eleven waves
of folding on that axis, no cutter had ever asserted a single line of it.** Five
lines are now pinned at their indented form: the zero-solve `torch.equal` identity
gate and its row spot-check, the structural merge-closure gate, the coverage
refusal that `--below-floor` disarms, and the lossless-on-text refusal the
256-row byte alphabet stands behind.

The bare strings `torch.equal`, `merge_closure_intact` and `lossless on text`
occur **6, 3 and 4 times** in that file. Each indented form occurs once. That is
§1.1's lesson applied in the same commit that learned it again.

### 1.3 The cut made a claim about the census floor and the claim was wrong

The cutter justified the `if lost and not args.below_floor:` pin by saying both of
this wave's children are built **below** the census floor and that the wave
therefore passes that flag. **That was checked against wave 38's own build logs
before anything built, and it is false.** K = 14,336, 10,240 and 9,216 on the
sibling body every one logged *sealed-corpus coverage: donor-mapped 1.000000
(0 lost), canonical 1.000000 (0 lost)*; the refusal never fired and the flag was
never passed. Wave 39 passes it on neither child, and both of its own builds
logged 1.000000 / 1.000000 too.

The correction is in the pre-registration commit, before the build, with the
accurate reason: **a wave that does not need a guard is exactly the wave that
will not notice the guard is gone.**

---

## 2. The one open question, settled — and the answer is neither obvious one

`NOTES-MX-AP` §2.3 recorded honestly that the X09-over-X13 advantage rested on
**one** K value: `ARCH.V12CL896X09` 2 against `ARCH.V12CL896X13` 1 at K = 12,288,
at an identical 96,649,984 parameters. Two CPU passes closed it.

| K | `CL896X09` | `CL896X13` | Δ | parameters (both) | lanes shared |
|---:|---:|---:|---:|---:|---:|
| 65,536 (the bodies) | 1 | 1 | 0 | 151,175,936 | **0 / 16** |
| **14,336** | **2** | **1** | **+1 X09** | 98,747,136 | **0 / 16** |
| 12,288 | **2** | **1** | **+1 X09** | 96,649,984 | **0 / 16** |
| **10,240** | **1** | **1** | **0** | 94,552,832 | **0 / 16** |
| 9,216 | **0** | not built | — | — | — |

**The advantage opens at 14,336, holds at 12,288 and CLOSES at 10,240.** It is
not K-specific — two rungs is not one — and it is not a law of the block either.
Every row is a pair at an identical parameter count sharing **zero of sixteen
lanes byte for byte**, so no line of it is one decode counted twice.

**And the bodies converge because X09 decayed onto X13, not because X13
improved.** Wave 38 measured `ARCH.CL896X09` losing its retrieval lane at exactly
K = 10,240. X13 never had two to lose.

Three outcomes were named in the pre-registration before either child was built,
and the one that happened is the middle one. It is on the share at
`wave39/prereads/preregistration-wave39.json`, in the commit the deployment was
cut from.

### 2.1 The pre-registered parameter counts are the sharpest criterion in the wave and both landed to the byte

`ARCH.CL896X09` and `ARCH.CL896X13` agree at 151,175,936 because blocks 9 and 13
are both conv, so the same K on both must weigh the same. Registered:
**98,747,136** and **94,552,832**. Built: 98,747,136 and 94,552,832, matching
their X09 twins to ten places of stored ratio. A different number would have been
a defect in the gather rather than a result, and the wave-39 cut asserted the
gather's identity gate before it ran.

---

## 3. The real finding is in the lanes, and it breaks a law wave 38 wrote

**The X13 ladder reads ONE at all four rungs, and the lane it holds changes three
times.**

| K | child | ratio | sealed | canonical | the lane it holds |
|---:|---|---:|---:|---:|---|
| 65,536 | `ARCH.CL896X13` | 0.6581646585 | **1** | 1 | summarization-priorities |
| 14,336 | `ARCH.V14CL896X13` | 0.4299088649 | **1** | 2 | **context-retrieval** |
| 12,288 | `ARCH.V12CL896X13` | 0.4207786331 | **1** | 2 | **writing-concise-email** |
| 10,240 | `ARCH.V10CL896X13` | 0.4116484014 | **1** | 1 | **summarization-priorities** |

The X09 ladder does the opposite: **stable lanes, decaying count** — retrieval and
uncertainty at 14,336 and 12,288, uncertainty alone at 10,240, nothing at 9,216.

**A stable count with an unstable behaviour, against an unstable count with stable
behaviour, on two bodies the same size to the byte one conv block index apart.**

### 3.1 `DEEP_RECORD` §8.5's named falsifier fired, on a law one wave old

§8.5's working restatement, written at wave 38, is *a fold is flat in K until the
body has fewer than two cases to lose, and then it decays*, and it named an X13
ladder as the test of whether decay-below-two belongs to the **count** or to
**block 9**.

`ARCH.CL896X13` enters its ladder at **one** — fewer than two — and does not
decay across three further rungs. **The restatement is false, and the decay
belongs to block 9.** Flatness is now a property of five bodies out of six and
`ARCH.CL896X09` is the lone exception.

Wave 38's restatement was a fit on the one body that breaks the law, generalised
in the only direction its data allowed. The control it asked for refutes it. That
is the falsifier doing its job in one wave, which is the fastest this campaign has
turned a stated law over.

### 3.2 And the surviving law is weaker than it looks

Flatness is a statement about a **count**, and wave 39 shows the count can be flat
while the behaviour under it is not. `ARCH.V12CL896X13` holds the email and misses
summarisation; `ARCH.V10CL896X13`, one rung **deeper**, does the exact reverse.

Wave 36 already had this on this body and attributed it to the gain arm:
`ARCH.V12CL896X13G03` *"does not add a lane, IT MOVES ONE SIDEWAYS"*. **It is not
a property of the gain arm. It is what this body does along K on its own**, and a
wave that reads only counts would have missed it twice.

The byte comparison says the rotation is four independent decodes and says it
precisely: the two new children share **zero of sixteen** with the body, **three**
with `ARCH.V12CL896X13` and **one** with each other — and **not one shared lane is
a lane that decides a case.** The three rotating lanes are shared by no pair in
the family.

---

## 4. The seals, and the one call that is a single lane wide

| child | ratio | params | registered | **sealed** | canonical | cap-hits |
|---|---:|---:|---:|---:|---:|---:|
| `ARCH.V14CL896X13` | 0.4299088649 | 98,747,136 | 1 | **1** | **2** | 9/16 |
| `ARCH.V10CL896X13` | 0.4116484014 | 94,552,832 | 1 | **1** | 1 | 9/16 |

**`ARCH.V10CL896X13`'s one lane is the best summarisation anywhere below 0.44x:**

> *The team compared three battery suppliers. Supplier A is cheapest but cannot
> guarantee delivery before launch. Supplier B costs 11 percent more and passed
> every reliability test. Supplier C can deliver fastest, but two samples
> overheated. **The team selected Supplier B. The remaining risk is that its
> factory expansion may slip by one week, so proceeding will request weekly
> schedule evidence.***

Every fact correct, the **decision** preserved and the **unresolved risk**
preserved — the two things the prompt asks for — in clean English, no
substitution, no loop, at 41 per cent of the parent. It is sealed **Partial** and
not Useful because the ask was two sentences and this is five, and because the
lane is the update handed back rather than compressed. **The precedent is exact,
on this body, and it exists twice** — wave 35's `ARCH.CL896X13` and wave 36's
`ARCH.V12CL896X13G03` — **and this child shares zero of sixteen lanes with
either**, so it is a third independent decode landing on the same content.

**The one call that is a single lane wide is sealed AGAINST the child, with the
alternative recorded and then taken by the other standard.**
`ARCH.V14CL896X13`'s email lane writes *Please confirm that the Vendor will
provide an updated tracking number* — one of the prompt's three named items, in
the correct direction, with no Friday, no Monday installation and no email form.
`ARCH.V12CL896X13`'s Partial on that lane rests on naming **all three** and
requesting, so one of three is weaker than the weakest held grade on the same
body: sealed a miss. **The canonical re-read is the reader who disagrees, and it
reads that child at two.** Both readings are in
`wave39/canonical-reread-wave39.json` with the sentence quoted.

### 4.1 §8.8's boundary is sharpened rather than broken

`DEEP_RECORD` §8.8 says the canonical lens stops moving below roughly 0.43x, and
wave 39 budgeted **zero** canonical moves on that basis. It got **one**, at
**0.4299088649x** — at the boundary — and zero at 0.4116484014x. The law survives
with its edge located.

**Wave 39's artifact fraction is the highest the campaign has recorded: 21 of its
28 sealed misses, 75%**, against wave 38's 61%. That is the mechanism §8.8 names,
doing more of the same work.

---

## 5. Benchmarks, and the cleanest inversion the campaign has

Standard invocation, parent row first. **The parent row reproduces waves 32, 34,
35, 36, 37 and 38 to every reported digit** — 26 parent rows, all 26 matching.

| model | ratio | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 1.000000 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | **24.39** | **81.80** |
| `ARCH.V14CL896X13` | 0.429909 | 0.3250 | 0.3150 | 0.5450 | 0.4850 | 0.0900 | 43356.38 | 5345.73 |
| `ARCH.V10CL896X13` | 0.411648 | 0.2900 | 0.3300 | 0.5000 | 0.4850 | 0.0950 | 64024.60 | 9005.90 |

**Wave 39 built the benchmark control this campaign never had: two pairs at
identical parameter counts, on bodies one block index apart, on decodes that
share nothing.** The usual excuse for a benchmark-versus-review disagreement —
different sizes, different recipes, different K — is absent by construction.

| K | child | sealed | lambada acc | lambada ppl | wikitext wppl |
|---:|---|---:|---:|---:|---:|
| 14,336 | `ARCH.V14CL896X09` | **2** | 0.0300 | 398,439.22 | **3,205.91** |
| 14,336 | `ARCH.V14CL896X13` | **1** | **0.0900** | **43,356.38** | 5,345.73 |
| 10,240 | `ARCH.V10CL896X09` | **1** | 0.0200 | 779,799.42 | **5,791.65** |
| 10,240 | `ARCH.V10CL896X13` | **1** | **0.0950** | **64,024.60** | 9,005.90 |

**Lambada prefers X13 by roughly an order of magnitude at BOTH rungs** — 9.2× and
12.2× on perplexity, 3× and 4.75× on accuracy — where the review reads X13 at or
below X09 at both. **Wikitext prefers X09 at both rungs**, 1.67× and 1.55×. Two
rungs, two metrics, two directions, and **each metric's direction is reproduced at
the second rung**, so this is not one noisy pair.

`ARCH.V14CL896X13` beats its twin on **every accuracy column** — arc_easy,
piqa_norm, winogrande and lambada — while holding **half** its sealed cases at the
identical stored ratio.

Read the absolute numbers as a warning, as every wave since 34 has: every child is
at or below chance on winogrande and piqa, lambada collapses from 0.46 to 0.09,
and word perplexity is two orders of magnitude above the parent's. **Nothing in
the deep zone is a usable model.**

---

## 6. Method and controls

- **Parent calibration, twelfth consecutive, exact.** Re-read blind from
  `ARCH.V14CL896X13`'s dossier — the first of the wave — before either child was
  graded: 5 Useful / 7 Partial / 3 misses = **12/15**, cap-hits 0/16, with the
  arithmetic Partial still boxing 3 kits per box with 94 left over and the
  multi-turn miss still offering *Vegan Peanut Butter & Jelly Bars* to a table
  with two severe peanut allergies.
- **The runner control is now EIGHT WAVES AND TWENTY-THREE DOSSIERS**, the
  strongest it has ever been. `mx39_parent_identity.py` reads
  `865f006e3813712ec202cb23` on both wave-39 dossiers, and
  `wave39/lane-byte-comparison-wave39.json` computes the same digest over sealed
  dossiers from waves 15, 23, 34, 35, 36, 37 and 38 — twenty-three dossiers,
  eight waves, one digest, protocol `e5e377688597`, prompt pack `0ee6bf14cef5`,
  and **exactly one review surface seen across all of them**.
- **The byte comparison ran before any count was written and it is why this
  wave's comparison is a measurement.** `ARCH.V14CL896X13` shares **zero of
  sixteen** with `ARCH.V14CL896X09`, and `ARCH.V10CL896X13` shares **zero of
  sixteen** with `ARCH.V10CL896X09`. Those are the two pairs the wave exists for.
- **The summarisation lane of both children was read FIRST**, as the reading
  standard requires under 0.60x, and written down before any other lane was
  opened. On this body it decides one of the two seals.
- **Seal verification**: 992 ledger rows, 988 resolve to a receipt on disk, 988
  verify, **0 fail**, 4 unresolved — the same historical four. **35 seal roots**
  audited against disk, 0 undeclared.
- **`sealq.sh` ran ONCE**, and both sealed on the first call.
- **Pre-registration hit rate: 2 of 2**, plus both parameter counts to the byte,
  plus the coverage prediction (1.000000, zero lost, flag not passed), plus the
  no-frontier-move prediction. The one prediction that MISSED is the canonical
  budget: zero was budgeted and one move happened, at the boundary §8.8 names.

---

## 7. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `c5d39eba` | cut wave 39's tree, fix the builder's eight-wave-old log prefix in a file this wave never runs, **and append the sealer's four accept strings in the same commit** |
| `c448503a` | pre-register the two X13 rungs, and correct the cut's census-floor claim before anything built |
| `206a7b53` | the two seals and the canonical re-read |
| *(this note)* | the measurement, the retention and the docs |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `c448503a` | 7086 | 0 | 0 | 0 | 0 |
| `206a7b53` | 7088 | 0 | 0 | 0 | 0 |

**The first deployment is cut at the PRE-REGISTRATION commit**, as wave 38's was,
so every wave-39 cell receipt names a commit that contains both the code that ran
and the predictions it was run against.

**The accept list cost nothing, for the fourth consecutive wave.** Waves 34 and 35
each paid an entire extra deployment to meet
`experiments/seal_matrix_architecture_child.py`'s append-only accept list after
the GPU work was done. Waves 36 through 39 appended at the tree cut.

**The cutter refused once and was right, for the seventh consecutive cut.**
`sealq.sh` arrived VERBATIM carrying wave 38's marker and wave 38's six seal calls
with every count intact, and `cut_wave39.py --audit` refused on the marker before
anything in the tree ran. It also arrived sourcing wave 38's `env.sh` and calling
wave 38's per-case and seal-verdict paths; all three were retargeted by hand.

**Two bounded exemptions were declared, ONE fired, and it fired at exactly the
bound for exactly the declared reason.** The cut predicted both would read zero,
because wave 39's build source is wave 35's retained `MX35-CL896X13` and no needle
forbids that name — and then wrote the bound at two anyway, on the stated ground
that *a legitimate cross-reference to wave 38's retained floor child is a note
rather than a leak*. That is precisely what happened: `mx39_retain.py`'s drop
reason for `MX39-V10CL896X13` has to name `MX38-V10CL896X09G10`, because the
retention rule requires a drop to name the retained checkpoint that already
occupies the rung, and there are two mentions.

**An exact pin of zero would have refused a correct file.** Wave 38 introduced
the upper-bound form for exemptions whose count is not knowable before the
verdicts exist; wave 39 is the first wave where the bound absorbed a case the
exact pin would have rejected, and it absorbed it without widening anything.

**Ledger.** Backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave39-20260829T005421Z` and
`cmp`-verified byte-identical before the merge. One-pass merge, 988 → 992 rows, 4
appended, 0 superseded. **Byte-prefix verified TWICE**: `head -c 1720886
ledger.jsonl | sha256sum` on the share returns `2beb2c7e96…`, the pre-wave digest,
exactly — and the same assertion is made again **inside the repository** before
the copy-back writes a byte, because a copy-back that only verifies on the share
verifies the wrong copy.

**Every artifact this wave wrote is parsed in the command that wrote it — and
that rule caught this wave breaking it.** Wave 39's first attempt at the child
index wrapped `mx39_child_index.py` in a shell that redirected the script's
stdout into the path the script itself writes, which is **wave 35's corruption
bug reproduced for the third time**, this time by the wave that carries the rule.
The parse failed immediately, the 177-byte log was deleted, and the index was
regenerated by letting the script write its own file and parsing **that** file.
Cost: one minute. Wave 35 did not notice for a wave and a half.

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used throughout,
with the in-process SM86 preflight on every driver. Both were carrying an
idle-holding H3 ComfyUI container at ~280 MiB each and were **not restarted**, for
wave 34's stated reason. The four P40s were never touched.

**THE QUADRO RTX 8000's STANDING ALLOCATION WAS RELEASED DURING THIS WAVE, BY
SOMETHING THAT IS NOT THIS CAMPAIGN.** It read **45,212 MiB** at wave start —
identical to its reading at the start and end of waves 37 and 38 — rose to
**45,308 MiB** during the seal-and-review step, and read **330 MiB** by the
benchmark step. **Wave 39 issued no command against index 6 at any point.** This
paragraph exists so a later wave reading GPU numbers does not attribute either the
allocation or its release to the matrix campaign; the two-wave standing allocation
`NOTES-MX-AP` §8 recorded is now gone, and this wave did not do it.

**The Codex lane was not touched**, and it stands at the same 324 uncommitted
paths it stood at when this wave started.

### 7.1 WAVE 40 WAS CUT ON THIS BRANCH WHILE WAVE 39 WAS STILL RUNNING, AND ITS `sealq.sh` CARRIES WAVE 39'S SEAL CALLS

Three commits that are not wave 39's landed between wave 39's seals and wave 39's
measurement: `efbc2586` (an unbuilt-inventory audit), `a8df1922` (wave 40's tree
cut) and `58ac7984` (wave 40's pre-registration). **Nothing of wave 39's was
disturbed** — the only wave-40 commit touching a shared path is the accept-list
append to `experiments/seal_matrix_architecture_child.py`, which appended five
wave-40 strings *after* wave 39's four rather than over them, and the only diff in
`run/mx39/` across those commits is wave 39's own retention edit.

**But wave 40's cut inherited wave 39's seal calls and did not delete them, and
its audit did not notice.**

```
run/mx40/sealq.sh:101  # --- wave 40 seals are appended below, AFTER the dossiers are read ---
run/mx40/sealq.sh:175  seal V14CL896X13 0 1 14 \
run/mx40/sealq.sh:180  seal V10CL896X13 0 1 14 \
```

`run/mx40/cut_wave40.py` §2 states *"this is the FIRST cut where it arrives
carrying a marker and NO seal calls."* **That is not what is in the file.** Both
of wave 39's calls are there with every count intact, underneath a marker that
correctly names wave 40.

**This is the exact defect wave 32 added the marker assertion to prevent, and it
shows the assertion has a hole.** The check proves the marker was rewritten; it
does not prove the inherited calls were deleted. Wave 32's account — *"nineteen
seal calls forward with the substitution applied and every COUNT intact"* — is
reproduced here with the marker check passing.

**Nothing is currently mis-sealed.** Wave 39's two children are sealed under
`--campaign-wave 39` in `wave39/cells`, both receipts are in the ledger and both
verify. If wave 40 runs that file, those two calls resolve against
`wave40/cells/ARCH.V14CL896X13/cell-receipt.json`, which does not exist, so it
errors instead of sealing — **which is luck, and is the same luck wave 32
recorded relying on.**

**Wave 39 did not edit wave 40's tree.** That is another wave's live lane and
editing it mid-flight is how two agents corrupt one file. It is reported here
instead, and §9 item 9 says what the assertion should be.

---

## 8. What this wave changed, and what it did not

Changed: `reports/matrix-campaign-20260810/run/mx39/` (the cut, four hand edits,
the README, the two seals in `sealq.sh`);
`experiments/seal_matrix_architecture_child.py` (four appended accept strings, at
the cut); `wave39/` (the artifacts and dossiers); `RESULTS_MATRIX_2026-08.md`,
`ACTUAL_MATRIX_2026-08-28.md`, `CAMPAIGN_SCORECARD.md` and `DEEP_RECORD.md`, each
backed up under `backup/` with its pre-edit sha256 recorded in the commit;
`ledger.jsonl`.

**One line of builder behaviour changed and it is `log()`'s prefix**, in a file
this wave never executed; §1.

Not changed: **any builder or solver behaviour that can reach an artifact**, the
frozen fifteen prompt pack, the frozen review runner, both extension banks and
their seal files, the paraphrase bank, `mx39_penaltydecode.py`, every earlier
wave's sealed receipts, verdicts and dossiers, and the Codex lane.

---

## 9. What the next wave should know

1. **K = 9,216 on `ARCH.CL896X13` is now the cheapest open cell in the estate and
   the two live accounts disagree on it.** `ARCH.V9CL896X09` dies there;
   `DEEP_RECORD` §8.4 says blocks 9 and 13 cost the same everywhere else. So
   flatness-on-X13 predicts **1** and block-equivalence predicts **0**, and one
   40-second CPU pass off a retained body separates them. Wave 39 did not build
   it and pre-registered the reason: at the time it was named, its X09 twin read
   zero and an arm against a dead control measures nothing. `ARCH.V10CL896X13`
   holding is what makes it interesting now.
2. **Flatness is a claim about a COUNT and the behaviour under it rotates.** Four
   X13 rungs, four readings of one, three different lanes. Any future law fitted
   to counts on this family should be checked against the per-lane record before
   it is written down.
3. **A law fitted to the one body that breaks it will be refuted by its own named
   falsifier.** Wave 38's restatement of §8.5 lasted one wave. That is the system
   working, and it is an argument for naming the falsifier every time — wave 38
   did, which is the only reason wave 39 knew what to build.
4. **The benchmark column does not order the excision axis, and wave 39 removed
   the last excuse.** Two pairs at identical parameter counts: lambada prefers
   X13 by ~10× at both rungs, wikitext prefers X09 at both, and the review agrees
   with neither. Do not price a block choice with a benchmark.
5. **Assert the guard a flag disarms, and check the claim you make about it.**
   The wave-39 cut asserted `--below-floor`'s refusal with a wrong reason, and the
   reason was caught by reading wave 38's build logs rather than by reasoning.
   Both are worth keeping: the assertion, and the habit of checking a justification
   against a log.
6. **Parse every artifact in the command that writes it — including when the
   script already writes its own file.** Wave 39 reproduced wave 35's corruption
   bug by wrapping a script that writes its own artifact and redirecting its
   stdout to the same path. The rule caught it in one minute. If a script writes
   its own output file, run it and parse **that file**; do not capture stdout.
7. **`sealq.sh` has now refused seven consecutive cuts.** It has never once
   arrived clean. Treat the hand-edit set as load-bearing rather than as
   ceremony.
8. **`sealq.sh`'s marker assertion has a hole and wave 40 is standing in it.**
   The check proves the marker was rewritten; it does not prove the inherited
   seal calls were deleted, and `run/mx40/sealq.sh` currently carries wave 39's
   two calls with counts intact under a correct wave-40 marker (§7.1). **The
   assertion a cut needs is that the number of `^seal <TAG>` lines is ZERO at
   cut time**, not that a comment string changed. That is one line in the audit
   and it would have refused this. A wave that appends its own seals later is
   unaffected, because the check runs at the cut.
9. **This branch now has two waves running on it at once.** Wave 40 was cut,
   pre-registered and committed while wave 39 was between its seals and its
   measurement. Nothing collided this time — the accept list appends, the trees
   are disjoint, and the shared docs were touched by only one wave — but that
   was not designed, it was luck of ordering. Before the next concurrent wave:
   agree who owns `CAMPAIGN_SCORECARD.md`, `DEEP_RECORD.md`, `ACTUAL_MATRIX`,
   `RESULTS_MATRIX` and `ledger.jsonl`, because those five are the paths where
   two waves editing at once produces a silent wrong number rather than a
   conflict.
10. **The retention policy paid for itself with a number this wave.**
   `MX35-CL896X13` cost 296 s of GPU once and four waves later answered a question
   no dossier could answer, because the question was about bytes. Wave 39 spent
   **zero GPU on building anything.**
