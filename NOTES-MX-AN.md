# NOTES-MX-AN — wave 36: a guard lifted in half, and the last body region opened onto a death

**2026-08-28.** Branch `codex/lrc-blank-compression`. Six children built, sealed
and read twice, plus one identity gate and one reproduction control that are not
cells. Ledger 952 → **964** rows,
`a0cb153641df88883becc86c9dcdafa1cbea75d61b0efb18af15d91e30b50632`, byte-prefix
verified against the pre-wave backup on the share and again in the repository
before the copy-back was written. **Three deployments, and none of them was
bought by the sealer's accept list, which refused waves 34 and 35 for one line
each.**

---

## 1. The guard, and the half of it that was wrong

`NOTES-MX-AM` §3.5 records a refusal. `mx35_build.py`'s `main()` would not
compose `--hidden-size` with `--ff` in one build:

> `--hidden-size` already re-solves every reader and every writer in the model
> against a new residual basis; composing `--ff` or `--head-dim` on top of it in
> one build would run a second selection inside tensors the first one has already
> replaced, with neither axis's census having seen the other.

Wave 36 was ordered to lift it properly. **It was half right, and exactly half of
it is lifted.**

### 1.1 The weights commute, and that is provable rather than argued

The residual-basis fold acts on the **hidden** index and only on it:

* readers `w1`, `w3`: `[F, 1024]` → `sqrt(1024/H)·W·diag(g)·E·diag(1/g_c)`, shape
  `[F, H]`. The row index `F` is untouched — every intermediate channel keeps its
  own row, re-expressed as a functional of the narrowed residual.
* writer `w2`: `[1024, F]` → `Eᵀw2`, shape `[H, F]`. The column index `F` is
  untouched.

The MLP coreset selects a subset `S` of that same `F` index, and it acts on the
**other side** of every one of those tensors. So `Eᵀ(w2·S) = (Eᵀw2)·S`, and
`(w1·S)` folded is `(w1 folded)·S` row for row. **The two folds commute exactly,
and the composed prior is the same object in either build order.**

`--head-dim` has no such property. It rewrites the attention block's row
structure, re-picks a RoPE frequency ladder from pooled plane energy, and moves
`head_dim` — the one config field this axis *pins*, because `Lfm2Attention`
otherwise derives it from `hidden_size // num_attention_heads`. Those are two
rewrites of one tensor, not two coordinates of one fold. **`main()` still refuses
that pairing, in the same words**, and the cutter's audit asserts the refusal by
name.

### 1.2 What does not commute is the census, and moving it is the build

The selection is a pivoted-Cholesky coreset of the **gated intermediate**, which
is data. The sealed `--ff` path ranks it on `phi = swiglu(X, w1, w3)` where `X`
is the block's post-norm row on a parent-shaped conditioned stream. On a narrowed
body that stream does not exist.

`narrow_mlp_site` moves the census into the narrowed space. It is the MLP
counterpart of what this same file already does for the **conv** coreset on the
width axis, and `parent_gated_rows`'s own docstring is the argument for it:

> which channels the child should keep is a question about the stream the child
> actually has, and the answer moves once the blocks above it are folded

The sweep runs in **forward order**, one block at a time, and every capture goes
through the live child — so block *i*'s census sees every narrowing above it, the
operator branch is solved before the MLP at the same block, and the final norm's
gain solve sees all fourteen. Nothing is solved against a stream a later decision
invalidates.

**The one thing that does not see the other axis is the basis**, which is
computed once from the parent's 29 residual endpoints before any fold. That is
stated rather than hidden, and it is equally true of every sealed width build
that composes `--excise` or `--kv-groups`.

### 1.3 The identity gate, and it passed harder than it had to

Registered in advance: `--hidden-size 1024 --basis identity --ff 2560
--no-fold-solve --no-head-gain --identity-gate strict` must export a child whose
every tensor is bit-identical to the parent's.

> `identity gate: 132/132 tensors bit-identical to the parent, passed=True`
> `child 229,693,184 parameters = 1.000000x  (196s)`

And it is stronger than the criterion asked for: the exported
`model.safetensors` **digests
`f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd`, which is the
parent's own file digest.** Not tensor-by-tensor equality — the same file.

**No wave had ever run this gate.** It has been in the builder since the width
axis was written and nothing in seventeen waves exercised it. Wave 36 registered
that fact in advance, together with the statement that a failure predating this
wave would be reported as a finding about the width axis rather than patched.

### 1.4 The second proof: `ARCH.W896`, rebuilt bit-identically across seventeen waves

The gate proves the composed path is a generalisation of the *parent*. It does
not prove the width path is unchanged under the lift. So this wave rebuilt
`ARCH.W896` — wave 19's sealed width body — with wave 36's builder, against wave
19's own sealed basis artifact (`E-reader-pca-896.safetensors`, sha
`d054349102f2621e…`):

> rebuilt tensors sha256 `bfa58964549778222141e01f071e8e7fd63935499ad3f26daf203da1adbb5b82`
> sealed&nbsp;&nbsp;tensors sha256 `bfa58964549778222141e01f071e8e7fd63935499ad3f26daf203da1adbb5b82`
> **BIT-IDENTICAL to the sealed body**

That is what makes §2's subtraction legitimate: `ARCH.W896` reads 4/15,
`ARCH.W896CL1792` reads 0/15, and the width path between them is proven not to
have moved.

### 1.5 The exactness claim, measured at every site

The composed selection rests on `phi_child` being **equal** to the parent's gated
intermediate at the lifted row — `E` has orthonormal columns, so the lift
preserves the norm, so the two RMSNorms differ by exactly the `sqrt(1024/H)` the
reader fold already carries. In exact arithmetic they are the same vector.
`COMPOSED_READER_EXACTNESS` measures the gap at every composed site and refuses
above 5e-2.

| build | max | median | registered |
|---|---:|---:|---|
| identity gate (H=1024, E=I) | 3.55e-3 | 3.28e-3 | 1e-3 to 5e-3 |
| `ARCH.W896CL1792` (H=896) | **8.38e-3** | 7.66e-3 | 1e-3 to 5e-3 |

**The real build's floor is 2.3× the identity build's and it is outside the band
this wave registered.** The registration was written for the identity case and
the real fold's `diag(1/g_c)` division amplifies bf16 error; the number is six
times under the refusal floor and the same order, but the pre-registration was
wrong about the band and this table is where that is recorded.

Three refusals came with the lift, all about the census rather than the weights:
`--ff` with `--hidden-size` requires a conditioned arm; `--select-metric logit`
is refused on the composed path; and `--ff` above the parent's own
`intermediate_size` is not a narrowing.

---

## 2. `ARCH.W896CL1792`: the last unbuilt two-axis body, and it is dead

| | |
|---|---|
| build | `--hidden-size 896 --ff 1792 --arm conditioned-local`, 793 s of GPU |
| stored | **168,410,240 parameters, 0.733197x** |
| sealed | **0/15** |
| canonical | **1/15** |

**`NOTES-MX-AM` §3.5 registered 168,410,240 for a child that could not be built
and forbade any row from citing it. The loader counted 168,410,240.** The
registration was correct and is retired as a prediction. So is the K=12,288
number: registered 120,700,032, counted 120,700,032.

**Two priors were pre-registered against each other and the allocator won for the
first time in three waves.** The additive model says `12 − 8 (width@896) − 5
(mlp@1792) = −1`. The same-body precedent — `ARCH.W896` 4/15 →
`ARCH.W896X13` 1/15, a second structural axis costing three — says 1. **This wave
registered 1, on the body prior, and observed 0.**

`mx36_allocator.py` could not have done better than additive here and it is worth
saying why: its interference fit **skips every cell with `hidden != 1024`**, so
there has never been a `widthxmlp` constant and there is not one now. Wave 36's
children are the first observations that could fit one, and the model's own gate
would still exclude them.

### 2.1 The fold is healthy, which is what makes the death informative

| | |
|---|---:|
| composed MLP sites, held-out cos | 0.9456 – 0.9895 |
| **width fold alone, before any channel is dropped** | **cos 0.999963 at every site** |
| final-norm gain solve, logit relative SSE | 0.5572 → **0.1701** |
| worst site | `mlp.down` at L02, relL2 0.3418 |

`full_width_prior_cos` is the control this wave added and it decides the
attribution: **all of the MLP branch's loss is the coreset's**, because the width
fold on its own reproduces the parent's MLP write at the lifted row to six nines
at all fourteen blocks. The solve beats the truncated prior at thirteen of
fourteen sites; the exception is L01, where the solve rides `update_scale` 0.886
and held-out cos falls from 0.9581 to 0.9536, and `THE LAW`'s clamp engaged at a
requested ratio of 1.2221.

### 2.2 What it produces

Eleven of fifteen lanes are compression artifacts, **including two lanes that
return zero characters**. Its retrieval lane reproduces the whole garden context
verbatim — 4:15 p.m. and the closed north gate included — and then echoes the
*question* instead of answering it. Its summarisation lane, the only one the
canonical standard rescues:

> *The team is preparing to schedule the factory expansion and requests weekly
> schedule evidence to ensure compliance and adherence to standards.*

The mitigation survives, attached to an inversion of the risk it mitigates.

### 2.3 The two vocabulary folds on it are zero, and one of them says why

`ARCH.V12W896CL1792` (0.525484x) and `ARCH.V16W896CL1792` (0.541462x) both read
**0/15 on both standards**. The vocabulary fold is flat on this body exactly as
it is flat on `ARCH.W896` (four children at 3), on `ARCH.CL1792` (eight at 7) and
on `ARCH.CL896` (eleven thousand rows at 2) — and here it is flat at nothing,
which is the first time the flatness law has been tested on a body with nothing
to be flat about. The K=16,384 rung was built as the *control* for K=12,288
because those two rungs are ordered differently on the two constituent bodies
(V16 one above V12 on CL1792, tied on CL896); a dead body separates nothing and
that question is unanswerable here.

`ARCH.V16W896CL1792`'s seasons lane is the wave's cleanest content miss:

> *The reason for the summer season is indeed quite close to the Sun.*

That is the exact misconception the prompt asks it to correct. **A zero here is a
zero of intelligence and not of legibility**, and only the artifact/content split
makes that statement available.

### 2.4 The three-axis body was registered, conditioned, and not built

The wave order made `X13W896CL1792` conditional on `ARCH.V12W896CL1792` reading
≥ 3/15. It read 0. The body is 160,377,600 parameters at 0.698225x and its
K=12,288 fold would be 112,667,392 at 0.490513x — **numbers for a child that does
not exist**, registered in advance and carrying the same prohibition
`NOTES-MX-AM` §3.5's did: no frontier row may cite them.

---

## 3. The deep family: the ladder inverts, and the +1 law finds its floor

| child | ratio | sealed | canonical | pre-reg sealed | pre-reg canon |
|---|---:|---:|---:|---:|---:|
| `ARCH.V9CL896G03` | 0.437349 | **3** | 4 | 3 | 3 |
| `ARCH.V9CL896G10` | 0.437349 | **4** | **5** | 2 | 3 |
| `ARCH.V12CL896X13G03` | 0.420779 | **1** | 1 | 2 | 3 |

### 3.1 λ1.0 beats λ0.3, for the first time in twelve ladders

On K = 9,216 the ladder is **2 / 3 / 4** for control / λ0.3 / λ1.0. λ1.0 has the
*smallest* update (0.0425 against 0.0793) and the *worst* held-out objective
(relSSE 0.1844 → 0.1785 against → 0.1769). **Seventh wave to record an objective
ordering arms in the opposite direction to the review.**

The lane that separates them is `context-retrieval`, and it is the one lane λ1.0
shares byte-identically with neither its control nor the λ0.3 arm:

> *The latest checkout time is 4:15 p.m. The gate should be considered to be
> closed for repairs.*

**Wave 34's precedent decided this call and it is named so it can be argued
with.** Wave 34 sealed `ARCH.V12CL896G03`'s retrieval lane **Partial** on
"The latest checkout time is 4:15 p.m." with a *nonsense* gate clause, and in the
same wave sealed `ARCH.V12CL896G01`'s a **miss** on "4:15 MINUTES" — the right
number in the wrong unit. Wave 35 applied the miss half to `ARCH.V10CL896G03`.
Wave 36 applies the Partial half here, to text with the right unit and a gate
clause that is *true* of the north gate. Four sealed CL1792 children at 7/15 and
8/15 hold this lane on text that gets the gate flatly wrong ("the gate should be
Saturday"), so refusing it here would apply a stricter standard to a deep child
than to a shallow one. A reader who requires the lane to name the north gate
seals this child at three.

### 3.2 The +1-at-every-K law extends down, in the strongest form available

`ARCH.V9CL896G03` was registered at exactly 3 and landed there, on exactly the
lane the law names. But the byte comparison says more than "the same shape":

| comparison | byte-identical lanes |
|---|---:|
| `ARCH.V9CL896` vs `ARCH.V10CL896` (the two controls) | 6 / 16 |
| `ARCH.V9CL896G03` vs `ARCH.V10CL896G03` | 6 / 16 |
| `ARCH.V9CL896G10` vs `ARCH.V10CL896G10` | 6 / 16 |

**Five lanes are byte-identical across a 1,024-row change in K at all three rungs
of the ladder** — `instruction-three-actions`, `logic-ordering`,
`writing-concise-email`, `planning-dependencies`, `safety-household-chemical`.
Wave 33's byte-comparison control had only ever compared an arm with its own
source; this is the first cross-child, cross-wave application of it.

The consequence is the strongest form of wave 35's law available: **the +1 lane
at K = 9,216 is not similar to the +1 lane at K = 10,240, it is the same bytes,
and both children's controls miss that lane with byte-identical text too.**

### 3.3 And it breaks at the floor, by moving a lane sideways

`ARCH.V12CL896X13G03` was registered at 2 on the law and reads **1**. The failure
is not a flat arm — it is a kind of arm no previous ladder has produced:

* it **gains** `summarization-priorities`, repairing exactly what wave 35 sealed
  its control a miss for. The control paraphrased and dropped the decision; this
  arm reproduces the update with *"The team selected Supplier B. The remaining
  risk is that its expansion may slip by one week, so proceeding will request
  weekly schedule evidence"* in its tail.
* it **loses** `writing-concise-email`, which its control held on the ASK, to a
  verbatim prompt echo: *"Draft a polite email under 90 words asking a Vendor to
  confirm whether a delayed shipment will arriving by the Friday."* That is the
  instruction handed back with two corruptions and no ask of its own.

**Net zero sealed, minus one canonical.** The call on the email lane is one case
wide and it is sealed AGAINST the child, with the alternative recorded: a reader
who credits the restatement for naming Friday, the Monday installation and the
tracking number seals this child at two, which would take the sealed 2/15 rung
from 0.437349x to 0.420779x.

So the law's scope is now bounded on both sides: **it holds on the vocabulary
axis down to the deepest rung that axis has, and it fails when an excision is
composed under it.**

---

## 4. The frontier

### 4.1 Sealed

| rung | before | after | Δ |
|---:|---|---|---:|
| 4/15 | `ARCH.V12CL896G03` 0.451045 | **`ARCH.V9CL896G10` 0.437349** | **0.013696** |
| 3/15 | `ARCH.V10CL896G03` 0.441915 | **`ARCH.V9CL896G03` 0.437349** | 0.004566 |
| 2/15 | `ARCH.V9CL896` 0.437349 | *does not move* | — |
| 1/15 | `ARCH.V12CL896X13` 0.420779 | *does not move* | — |

### 4.2 Canonical

| rung | before | after | Δ |
|---:|---|---|---:|
| 5/15 | `ARCH.V8CL1792` 0.600552 | **`ARCH.V9CL896G10` 0.437349** | **0.163203** |
| 4/15 | `ARCH.V10CL896G03` 0.441915 | **`ARCH.V9CL896G03` 0.437349** | 0.004566 |
| 3/15 | `ARCH.V10CL896G01` 0.441915 | **`ARCH.V9CL896G03` 0.437349** | 0.004566 |
| 2/15 | `ARCH.V12CL896X13` 0.420779 | *does not move* | — |
| 1/15 | `ARCH.V12CL896X13` 0.420779 | *does not move* | — |

**The canonical 5/15 move is the largest single frontier move in the campaign's
record on either ladder** — 0.163203, ahead of wave 34's 0.1495 — and it carries
the campaign's largest caveat. It rests entirely on the `planning-dependencies`
rule wave 35 stated in `NOTES-MX-AM` §5 so a reader could decline it. Strike that
rule and wave 36's canonical column is 1/0/0/**3**/**4**/1 instead of
1/0/0/**4**/**5**/1, the canonical 5/15 rung does not move at all, and the 3/15
and 4/15 rungs still move to 0.437349x.

**Wave 36 did not have to judge either flip.** Both flipped lanes are
byte-identical to lanes wave 35 already read — `8dd9e53045f9a211` (3,490 bytes)
and `307c301603c0ba3f` (1,020 bytes). Wave 33's byte-comparison control had only
ever been applied to the *sealed* reading; applying it to the assessment layer
takes the two flips out of the reviewer's hands entirely, and that is wave 36's
methodological addition.

### 4.3 What each child holds

| child | ratio | sealed | canonical | art | con | pre-reg sealed | pre-reg canon |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ARCH.V12CL896X13G03` | 0.420779 | 1 | 1 | 11 | 3 | 2 | 3 |
| `ARCH.V9CL896G03` | 0.437349 | 3 | 4 | 9 | 2 | 3 | 3 |
| `ARCH.V9CL896G10` | 0.437349 | **4** | **5** | 8 | 2 | 2 | 3 |
| `ARCH.V12W896CL1792` | 0.525484 | 0 | 0 | 11 | 4 | 1 | 1 |
| `ARCH.V16W896CL1792` | 0.541462 | 0 | 0 | 11 | 4 | 1 | 1 |
| `ARCH.W896CL1792` | 0.733197 | 0 | 1 | 11 | 3 | 1 | 2 |

One of six sealed counts landed on the pre-registered number, three missed high
(the three composed children, all registered at 1 and reading 0), one missed low
by two in the child's favour (`V9CL896G10`, 2 → 4) and one missed low by one
(`V12CL896X13G03`, 2 → 1). Wave 35 landed five of eight exactly; **this wave
landed one of six, and it is stated first rather than buried.** The wave's
headline prior — that a second structural axis costs 2–3 on `ARCH.W896` — was
wrong in the same direction three times, and the additive model it was registered
against was right all three times.

**Sixty-one of the ninety lanes — 67.8 per cent — are compression artifacts**,
against a campaign-wide 29.8 per cent and wave 34's 50.8. `ARCH.W896CL1792` is
11 of 15 artifact **at 0.733197x**, against `ARCH.V32W896`'s 7 of 15 at
0.731199x — the same ratio, a different composition, four more artifact lanes.
That is not a record for the band: nine children above 0.7x are 13 to 15 of 15
artifact, all of them from the width, blend and head-dim families. It is a record for a
child at this ratio that also fails on CONTENT: `ARCH.W768AW100` at 0.722610x is
15 of 15 artifact and 0 of 15 content, so nothing is known about it either way;
`ARCH.W896CL1792` has three readable-and-wrong lanes and one signal lane, so
something is.

---

## 5. Benchmarks, and the sharpest disagreement the campaign has recorded

Seven rows, standard invocation. The parent row is identical to waves 32, 34 and
35's to every reported digit — arc 0.5450, hellaswag_norm 0.5050, piqa_norm
0.6800, winogrande 0.5500, lambada 0.4600, lambada ppl 24.39, wikitext word ppl
81.80. **Third consecutive calibration, exact.**

| model | ratio | sealed | arc_e | hella_n | wino | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|
| PARENT | 1.000000 | 12 | 0.5450 | 0.5050 | 0.5500 | **24.39** | **81.80** |
| `ARCH.W896CL1792` | 0.733197 | **0** | 0.3150 | **0.3900** | **0.5250** | **2838.85** | **781.25** |
| `ARCH.V16W896CL1792` | 0.541462 | 0 | 0.3150 | 0.3600 | 0.4550 | 4648.23 | 1883.31 |
| `ARCH.V12W896CL1792` | 0.525484 | 0 | 0.2850 | 0.3400 | 0.4450 | 5506.92 | 2700.43 |
| `ARCH.V9CL896G03` | 0.437349 | 3 | 0.2700 | 0.3450 | 0.4850 | 8072.45 | 2332.36 |
| `ARCH.V9CL896G10` | 0.437349 | **4** | 0.2700 | 0.3500 | 0.4750 | 7961.05 | 2370.59 |
| `ARCH.V12CL896X13G03` | 0.420779 | 1 | 0.3500 | 0.3250 | 0.4700 | 24675.04 | 4323.38 |

**The best benchmark row in the wave belongs to the only child that holds
nothing.** `ARCH.W896CL1792` has the best word perplexity, the best hellaswag and
the best winogrande of the six children, and reads 0/15 on both standards.
`ARCH.V9CL896G10` has three times its perplexity, sits 0.296 of stored ratio
deeper, and reads 4/15. Same table, same hour, same invocation, opposite
orderings.

---

## 6. Method and controls

- **Parent calibration, ninth consecutive, exact.** Re-read blind from
  `ARCH.V9CL896G03`'s dossier: 5 Useful / 7 Partial / 3 misses = **12/15**,
  arithmetic still boxing 3 kits per box with 94 left over, multi-turn still
  offering *Vegan Peanut Butter & Jelly Bars* to a table with two severe peanut
  allergies.
- **The runner control now spans a wave boundary.** The concatenated verbatim
  parent transcript of all sixteen lanes hashes to `aed8365ef86ac656` in **every
  wave-36 dossier and every wave-35 dossier**. The runner is provably fixed
  across two waves, not merely within one.
- **Two byte-comparison controls, one of them new.** The within-wave one:
  `ARCH.V9CL896G10` shares nine of sixteen lanes with `ARCH.V9CL896G03`
  including the +1 lane, and shares `summarization-priorities` with their common
  control, so that lane's verdict is the control's by construction. The new one
  is cross-wave and cross-K, and §3.2 and §4.2 are what it bought.
- **`full_width_prior_cos`**, added with the composed site, is what makes §2.1's
  attribution possible: it prices the width fold's own writer at every block
  before a single channel is dropped.
- **Two calls are one case wide and both were sealed AGAINST the child**, with
  the alternative recorded in the seal: `V12CL896X13G03`'s email lane and
  `V9CL896G10`'s retrieval lane (which was sealed FOR the child on wave 34's
  precedent, with the stricter reading named).
- **Seal verification**: 964 ledger rows, 960 resolve to a receipt on disk, 960
  verify, **0 fail**, 4 unresolved — the same historical four. **32 seal roots**
  audited against disk, 0 undeclared.
- **`sealq.sh` ran once.**

---

## 7. A wave-35 artifact was found corrupt, and repaired from its own sources

`wave35/wave35-child-index.json` was **not JSON**. It was the shell log of wave
35's own index run written over the JSON that same run wrote — redirecting the
script's stdout into the path the script itself writes truncates the file the
script then writes. It had been unparseable since wave 35 in both the repository
and the share, and **nothing noticed because nothing consumed it**: wave 35
deliberately did not refresh `RESULTS_MATRIX`, so `mx35_campaign_table.py` never
ran. Wave 36's did, and crashed on it.

The repair is a repair and not a rewrite: the file was regenerated by wave 36's
own `mx36_child_index.py` with its root pointed at wave 35, **from wave 35's own
sealed cell receipts and verdict receipts**, which were never touched. It
reproduces the eight children with the counts wave 35 sealed. The corrupt bytes
are kept at `backup/wave35-child-index.json.corrupt-found-by-wave36`
(`0e3dd0f9609f27ff…`).

**The lesson is one line and it belongs in the next cutter:** never redirect a
script's stdout into a path that script writes.

---

## 8. Estate

**Commits** (branch `codex/lrc-blank-compression`, never pushed, never rebased):

| commit | what |
|---|---|
| `6dc334d5` | cut wave 36's tree, lift the guard in half, **and append the sealer's four accept strings in the same commit** |
| `e81df916` | pre-register wave 36, including the guard-lift's own acceptance criterion |
| `8a81a924` | the six seals |
| `50bff133` | the retention, and the second prose reference the cutter refused |

**Deployments**, each verified file-by-file against its commit's blob digests:

| commit | files | mismatches | missing | undeclared | hardlinks |
|---|---:|---:|---:|---:|---:|
| `e81df916` | 6675 | 0 | 0 | 0 | 0 |
| `8a81a924` | 6675 | 0 | 0 | 0 | 0 |
| `50bff133` | 6675 | 0 | 0 | 0 | 0 |

**Three deployments, and the accept list cost none of them.** Waves 34 and 35
each paid an entire extra deployment to meet
`experiments/seal_matrix_architecture_child.py`'s append-only accept list at
their **first seal**, after the GPU work was already done, on exactly the same
four tags. Wave 35's note told the next wave to append the strings at the tree
cut instead of remembering harder. Wave 36 did, in commit `6dc334d5`, and all six
seals passed on the first call. **That is the third consecutive wave to write
that paragraph and the first to act on it.**

**The cutter refused twice and was right twice.** `sealq.sh` arrived VERBATIM
carrying wave 35's marker and wave 35's eight seal calls with every count intact
— fourth consecutive cut where that protection has had real work to do — and
later refused two seal notes for naming `MX35-V9CL896` as the retained source
each terminal-gain arm was solved on. That is provenance the seal is *required*
to carry, so it was declared at a pinned count of two rather than reworded away.

**Package, then retention, then prune.** The bundle is 52 files and 1.81 MB.
Retention kept **three** and deleted five (1.35 GB), each deletion naming what
rebuilds it and from which retained source, after a dry run that was printed and
read. Three rather than best-five is a stated deviation: one deleted checkpoint
is **byte-identical to the parent** (the identity gate — its whole content is a
sha256 that is already in a receipt), two are single CPU passes off a kept body,
and two are ~100-second gain arms off wave-35 sources wave 35 retained.
**`MX36-W896R` is kept AGAINST the claim rule and the exception is argued rather
than hidden**: it is the only existing checkpoint of `ARCH.W896`, provably wave
19's bytes, and the control the whole of §2 is differenced against.

The deployment prune kept four and deleted two (0.58 GB) under current + 1
**plus every deployment a sealed receipt names as `code_identity.repo_head`** —
`e81df916` for all six wave-36 receipts, `dcfa0c9c` for wave 35's and
`03d39e85` for wave 34's, all kept regardless of age.

**Ledger.** Backed up to
`/mnt/ai_data/neural-forge-matrix/ledger.jsonl.pre-wave36-20260828T200149Z` and
`cmp`-verified byte-identical before the merge. One-pass merge, 952 → 964 rows,
12 appended, 0 superseded. Byte-prefix verified twice: `head -c <backup size>
ledger.jsonl | sha256sum` returning `5517dd9b…` on the share, and again **in the
repository before the copy-back was written**, where the pre-wave bytes were
asserted to be a prefix of the incoming bytes. Repo and share both
`a0cb153641df88883becc86c9dcdafa1cbea75d61b0efb18af15d91e30b50632`.

**GPUs.** Both RTX 3090s (indices 0 and 4 under PCI bus order) used throughout,
with the in-process SM86 preflight on every driver. Both were carrying an
idle-holding H3 ComfyUI container at ~280 MiB each with an empty queue — the peer
sprite lane — and were **not restarted**, for wave 34's stated reason. The Quadro
RTX 8000 and the four P40s were never touched. Two orphaned processes from
timed-out SSH invocations were found duplicating a CIFS scan and killed before
either could write the same output file twice.

**Disk.** `/mnt/ai_data` at 64% (1.7 TB free) before and after; `/` at 63% with
649 GB free, deployments down from 2.08 GB to 1.50 GB.

**The Codex lane was not touched.**

---

## 9. What this wave changed, and what it did not

Changed: `reports/matrix-campaign-20260810/run/mx36/` (the cut, four hand edits,
the README, **`narrow_mlp_site` and the lifted guard in `mx36_build.py`**, the
six seals in `sealq.sh`, the retention policy, two declared historical
references in `cut_wave36.py`); `experiments/seal_matrix_architecture_child.py`
(four appended accept strings, at the cut); `wave36/` (the artifacts and
dossiers); `RESULTS_MATRIX_2026-08.md`, `ACTUAL_MATRIX_2026-08-28.md`,
`CAMPAIGN_SCORECARD.md` and `DEEP_RECORD.md`, each backed up under `backup/` with
its pre-edit sha256 recorded in the commit; `wave35/wave35-child-index.json`
(repaired, with the corrupt bytes backed up); `ledger.jsonl`.

Not changed: the frozen fifteen prompt pack, the frozen review runner, both
extension banks and their seal files, `mx36_penaltydecode.py` (the decode-repair
clause is settled and this wave did not re-run it), every earlier wave's sealed
receipts, verdicts and dossiers, and the Codex lane.

---

## 10. What the next wave should know

1. **`--hidden-size` composes with `--ff` now, and the composition is a content
   death.** The region is open and there is no reason to go back into it: two
   axes each past their own transition compose to zero, and the additive model
   said so. If a wave wants a width composition that survives, the axis to
   compose with is one whose solo deficit leaves room — `--excise` at block 13
   costs three on every body it has been measured on, and `ARCH.V16W896X13`
   already reads 2/15 at 0.623329x.
2. **`--head-dim` still cannot be composed with `--hidden-size` and the reason is
   in the builder's docstring, not in a note.** It is not an orthogonal
   coordinate of the fold; it is a second rewrite of the same tensor plus a
   config field this axis pins.
3. **λ1.0 is not a dead rung.** It has now moved a sealed count up on two
   consecutive bases and on the second one it BEAT λ0.3. The ladder is
   1.0 / 0.3 / 0.1 and it should keep running as a ladder; the objective still
   orders the arms backwards.
4. **The +1-at-every-K law is bounded on both sides.** It holds on the vocabulary
   axis down to K = 9,216, in bytes, and it fails at 0.420779x on an excised body
   by moving a lane sideways. The open corners are `V10CL896X13` (0.411648x) and
   `V9CL896X13` (0.407083x), each one CPU pass on wave 35's retained
   `MX35-CL896X13`, and nobody knows whether the summarisation repair this wave
   found survives another 2,048 rows down.
5. **The canonical 5/15 rung is the campaign's largest claim and it rests on one
   lane rule.** If Cyril wants that rung defended, the thing to settle is whether
   `planning-dependencies` — five correct work items, named, then looped instead
   of sequenced — is signal. Everything else in the canonical column is robust to
   that decision.
6. **Never redirect a script's stdout into a path the script writes.** §7.
7. **Append the sealer's four accept strings in the commit that cuts the tree.**
   It worked. Wave 36 is the first wave in three not to pay a deployment for that
   line, and the paragraph in
   `experiments/seal_matrix_architecture_child.py` now says so.
