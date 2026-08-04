# Cross-family fold from a **real pretrained** parent — SmolLM2-135M into a blank qwen3 (2026-08-04)

Runner: `experiments/run_cross_family_real_scale_fold.py`
(`sha256 f8776cb8d3a955b0172e624082bf981d87ae2be31bf43562ffd7d0c3bcf7d94c`,
unchanged across every run below).  Workstation, CPU only,
`torch 2.13.0+cpu` / `transformers 5.13.0`, six threads, no network, no server,
`HF_HUB_OFFLINE=1`.

This directory is the sequel to `reports/cross-family-real-20260803`, whose
single largest caveat was that **no pretrained parent existed on this
workstation**, so every number there was measured against a seeded random
initialization at SmolLM2-135M's geometry.  The checkpoint is here now:

```
C:/Users/cereal/Documents/ai training/models/SmolLM2-135M
  config.json          sha256 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
  model.safetensors    sha256 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
                       269,060,552 bytes, bfloat16 on disk, loaded as float32
  tokenizer.json / vocab.json / merges.txt   49,152-row ByteLevel BPE
```

`--parent-weights local-pretrained --parent-model-directory <dir>` loads it
with `local_files_only=True`, verifies eight config fields against the declared
`smollm2-135m` geometry before any compute happens, and seals the weight
digest, the config digest and the verified fields into the configuration
identity every later stage chains off.  Nothing under `src/` was modified; the
committed 20260803 reports were not touched and their configuration identity
still reproduces byte for byte under this runner
(`53abd116e9f28947fc5368cc0a9cdb7c9dd357bb6430d6b393cbfa166fdc4282`).

**The headline is a depth wall and a behavioral signal.**  Six layers of the
real parent does **not** fold — the promoted operator's fail-closed guard
refuses layer 3's attention branch and the child is rolled back to blank.
Three layers does, and the child folded from the real parent reproduces the
parent far better than the child folded from a random one at the same depth:
25.70% top-1 agreement against 3.13%, and greedy continuations made of English
function words and punctuation instead of arbitrary rare tokens.  That signal
is real and it is small; the honest size of it is quantified below.

262 MB in this directory, of which 257 MB is three folded children (101 + 78 +
78) and four compiled child tokenizers.  The `.log` file beside each run root
is the raw stdout of the command that produced it.

---

## The four runs

| run | parent | requested depth | outcome |
|---|---|---|---|
| `L6-prefix-pretrained` | **real SmolLM2-135M**, full rank | 6 of 30 | **failed closed at layer 3 attention**, rolled back to blank |
| `L3-prefix-pretrained` | **real SmolLM2-135M**, full rank | 3 of 30 | complete, decoded |
| `L3-prefix-random-control` | seeded random, full rank | 3 of 30 | complete, decoded |
| `L6-prefix-random-control` | seeded random, full rank | 6 of 30 | complete, decoded — **the random parent survives the depth the real one refuses** |

`L3-prefix-random-control` is a **single-variable** A/B against
`L3-prefix-pretrained`: same source tokenizer, same compiled child vocabulary,
same corpus, same windows, same child geometry, same seeds, same runner.  Only
the parent's weights differ.  The committed 20260803 runs cannot play that role
because they used a different tokenizer (50,257-row GPT-2) and, for the
6-layer arm, an exact-subspace parent.

Every completed run verifies:

```
python experiments/run_cross_family_real_scale_fold.py \
    --report-root reports/cross-family-real-20260804-pretrained/<run> --verify-only
# {"verified": true, "files": 13, "problems": []}
```

`L6-prefix-pretrained` has no manifest, because a manifest is only written
after the behavior stage.  `--verify-only` refuses it with `No sealed
manifest`, which is the intended shape of a refused run: a sealed
`fold-failure.json`, no `fold.json`, no child on disk.

---

## Read this before the numbers

**1. A layer prefix of a pretrained model is not a pretrained language model.**
`--layers 3` keeps the first three of SmolLM2-135M's thirty decoder layers and
then applies the real `model.norm` and the real `lm_head` to whatever those
three produce.  Every weight is pretrained; the object as a whole was never
trained to emit a distribution from layer three.  So these runs answer *does
the child reproduce this parent, on real structured anisotropic weights* —
which is the question the fold mechanism is actually about — and they do **not**
answer *can a folded child write English*.  Only a full-depth run answers the
second question, and a full-depth run does not exist here because the fold
stops at layer 3.  The depth actually folded is sealed in every receipt as
`folded_layer_prefix` / `full_depth` / `bounded_layer_prefix`.

**2. The pretrained runs are full rank.**  `--parent-residual-rank` rewrites
every residual-facing weight into one random subspace; applied to a checkpoint
it produces a model that is not the checkpoint while the receipt still says
`pretrained: true`, so the runner refuses the combination.  The committed
`L6-prefix-fold` was an exact-subspace control and is therefore **not** a
single-variable baseline for anything here.  Use `L3-prefix-random-control`.

**3. No sealed tokenizer pair bridge can exist for this checkpoint.**  The
design was: parent's own 49,152-row tokenizer as the pair-bridge source, a
pruned 16,384-row subset as the child's vocabulary, installed through the
sealed pair bridge exactly as the 20260803 runs did.  Both halves refuse.
`prune_hf_tokenizer` requires the donor to carry all 256 ByteLevel byte atoms
and SmolLM2-135M's real tokenizer carries **235**; `compile_hf_tokenizer_pair_bridge`
requires the same completeness of *both* vocabularies, in both code-table
directions, because a single-byte row is its only exact fallback for a token
that does not anchor.  Verbatim refusals, both sealed in `tokenizers.json`:

```
HFTokenizerPruneError: donor BPE vocabulary lacks required ByteLevel byte
atoms: 'À', 'Á', 'ñ', 'ò', 'õ', 'ö', '÷', 'ø'

TokenizerPairBridgeError: target tokenizer does not cover the complete 256-row
ByteLevel byte alphabet; single-byte rows are missing for 0x04, 0x06, 0x13,
0x14, 0x16, 0x1d, 0xc0, 0xc1. A pair bridge has no exact fallback without them.
```

The runner falls back to two local compositions that reuse `src`'s own steps
and change no math — a subset compiler seeded with the atoms the donor actually
has, and a **strict one-row-to-one-row donor transport** as the interface
operator's row source instead of pair codes.  Against a subset target the two
are close by construction — every target row is byte-identical to exactly one
donor row, so the pair lane's anchor-preferred blend takes the anchor branch
wherever it anchors, and the anchor branch *is* that donor row.  I did not
measure how many of the 20260803 run's 16,384 rows anchored rather than
averaging, so "close" is as strong as this should be put; what the receipts do
show is that both lanes covered every target row (`uncovered_output_rows: 0`).
Either way it **is** a weaker claim about the crossing — an
unrelated-vocabulary crossing is no longer being exercised, only a subset one.
Full detail, including what a proper `src` fix would touch, is in
`NOTES-W7A.md` section 7.

**4. The random control's final-norm number is trivially easy.**  A freshly
initialized `LlamaRMSNorm` gain is exactly ones, so fitting the child's final
norm against it is fitting a constant.  The real checkpoint's `model.norm` is a
learned non-uniform gain.  That is why `final_norm_relative_rmse` reads 0.0019
for the random arm and 0.5393 for the pretrained one, and it is the one place
in the A/B where the random arm is easy for a reason that has nothing to do
with the fold.

---

## Configuration

| | parent | child |
|---|---|---|
| family | llama | qwen3 |
| weights | SmolLM2-135M checkpoint / seeded random | blank, then folded |
| residual width | 576 | 432 (0.75x) |
| layers folded | 3 (prefix of 30) | 3 |
| query / key-value heads | 9 / 3 | 6 / 3 |
| head_dim | 64 | 64 (must match: the planner compares `rotary_dim`) |
| MLP intermediate | 1536 | 1152 |
| per-head Q/K norm | none (identity) | RMSNorm, fitted by the coupled kernel |
| vocabulary | 49,152 (tied embed/head in the checkpoint) | 16,384, untied |
| parameters | 38,932,416 (pretrained arm, 3 layers) | 20,131,152 |

Child vocabulary: a real ByteLevel-BPE **subset of the parent's own tokenizer**,
16,384 rows, 16,132 retained merges, complete merge dependency chains, every
special token, and the donor's byte coverage exactly (235 of 256 atoms — the
gap is the donor's, sealed under `tokenizers.byte_level_coverage`).

Evidence: 48 calibration prompts from
`reports/kernel-stage2-20260801/calibration_prompts_64.json` (indices 0..47),
tokenized by the **child** tokenizer into 128-token right-padded windows; the
parent reads the donor preimage of exactly those rows, so both models see the
same text at the same positions.  6144 positions, **4746 valid**, 2373 fit /
2373 validation, 1398 padded.  **One parent forward** for the whole run.
Held-out: prompts 48..63, 80-token unpadded windows, never seen by any fit.

---

## Mechanism: what the executor did

`L3-prefix-pretrained`, `fold.json`:

* Plan **ready, zero gaps**, 8 actions: `token_interface_bridge` x1,
  `copy_exact` x1 (the parameter-free rotary global),
  `causal_gqa_rope_residual_attention_layer` x3,
  `causal_swiglu_residual_mlp_layer` x3.
  `partitions_every_learned_child_tensor: true`.
* **8 attempts, 8 accepted**, status `complete`; 36 changed state paths, all
  unique; `completed_every_learned_child_tensor: true`.
* `observed_parent_root_forward_calls: 0`,
  `observed_parent_module_forward_calls: 0`, and an independent
  `register_forward_hook` counter on the parent also read **0** during
  execution.  The only parent forward in the run is the single evidence capture.
* `optimizer_constructions: 0`, `optimizer_steps: 0`, `backward_calls: 0`,
  `autograd_grad_calls: 0`, `pruning_operations: 0`,
  `quantization_operations: 0`,
  `executed_fold_static_zero_training_audit_passed: true`,
  `operator_static_zero_training_audit_passed: true` — the static audit covers
  the locally composed interface operator too.
* `child_recapture_calls: 6` — one live sequential recapture of the child per
  branch action.
* The token interface installed **16,384 of 16,384 target rows** (0 uncovered)
  from the strict donor-row transport, plus the fitted final-norm gain.

Timings, uncontended (`seconds` in `fold.json`): evidence capture 29 s, chart
compile 84 s, execute 445 s, behavior 41 s.  The 6-layer pretrained attempt
spent 571 s in execution before the guard refused it, and 490 s compiling its
13 boundary charts.  `L6-prefix-random-control`, the only completed 6-layer run
here, took 37 / 170 / 973 s.  The committed 20260803 6-layer run reports
134 / 1195 / 2059 s because a second agent process shared the six threads for
part of it; nothing here was contended by another fold.

---

## Numbers

### The depth wall

`L6-prefix-pretrained/fold-failure.json`, verbatim:

```
Cross-width causal attention branch operator endpoint relative RMSE
1.0923883067551874 is at or above the no-branch baseline bound 1.0
at boundary 'layer003.mlp_in'.
```

`child_recapture_calls: 7` — six branches accepted (layers 0, 1 and 2, both
kernels each), then layer 3's attention branch refused.
`child_state_rolled_back_to_blank: true`: the branch that failed and every
branch accepted before it are undone, so a partially folded child never
survives.  That run has no `fold.json`, no `child/` and no decoded output.

The no-branch baseline is 1.0 by construction — a zero write predicts nothing —
so an endpoint relative RMSE of 1.09 means **the fitted branch is worse than
writing nothing at all**, and the operator says so instead of installing a
regression.  Across both this directory and `reports/cross-family-real-20260803`
this is the only refusal at real geometry: the other sealed refusal,
`tiny-fail-closed-control`, was at H128 -> H96 with deliberately starved
evidence (192 fit rows against 192 target MLP neurons).  This one had 2373 fit
rows against 1152 target MLP neurons, the same ratio every accepted run here
used.

### Per-branch held-out endpoint error — the single-variable A/B

Same depth, same tokenizer, same corpus, same child.  Only the parent's weights
differ.

| layer | attention, **pretrained** | attention, random | gated-MLP, **pretrained** | gated-MLP, random |
|---|---|---|---|---|
| 0 | **0.13119** | 0.25289 | 0.61223 | **0.50308** |
| 1 | 0.75952 | **0.31470** | 0.98900 | **0.70482** |
| 2 | 0.98031 | **0.38447** | **0.60727** | 0.79135 |

| layer | upstream error energy ratio, attn / mlp — **pretrained** | random |
|---|---|---|
| 0 | 7.04e-16 / 3.04e-05 | 7.36e-17 / 8.34e-03 |
| 1 | 3.82e-03 / 7.76e-02 | 1.88e-02 / 7.24e-02 |
| 2 | **1.54e-01** / 6.36e-03 | 2.08e-02 / 1.08e-01 |

Two things in that table matter more than the values.

1. **The real parent's first attention boundary folds nearly twice as well**
   (0.131 against 0.253).  Real structure at the first residual boundary is
   *more* compressible into 432 coordinates than random structure, not less.
2. **And then it runs away.**  0.131 -> 0.760 -> 0.980 against
   0.253 -> 0.315 -> 0.384, with the accumulated upstream error energy ratio
   seven times higher by layer 2 (1.54e-01 against 2.08e-02).  Addendum 4 of
   `NOTES-W7A.md` recorded that a random parent's error accumulation *stopped
   growing* by layer 2 and read that as evidence depth was not the wall.  On a
   real parent it does not stop, and the wall arrives at layer 3.

Per-head Q/K norm gains, which the coupled kernel fits rather than copies, are
unaffected by any of this: query 0.204/0.186/0.170 and key 0.273/0.217/0.183
on the pretrained arm against query 0.255/0.303/0.246 and key 0.190/0.256/0.178
on the random arm.

### Boundary charts

| | **L3 pretrained** | L3 random | L6 random | committed L6 exact-subspace | committed L2 random full-rank |
|---|---|---|---|---|---|
| boundaries | 7 | 7 | 13 | 13 | 5 |
| cycle defect, median | 0.1354 | 0.1242 | 0.1097 | 1.265e-07 | 0.1258 |
| norm-commutation row fluctuation, median | 7.64e-03 | 3.39e-03 | 1.84e-03 | 1.30e-15 | 5.51e-03 |
| held-out retained energy, min / median | 0.9571 / 0.9716 | 0.9003 / 0.9770 | 0.9008 / 0.9823 | ~1.0 / ~1.0 | 0.8955 / 0.9751 |
| absorption-demand unabsorbed fraction, median | 0 | 0 | 0 | 9.1e-05 | 0 |

The chart is **not** what separates the two arms.  A real SmolLM2-135M residual
keeps 97.2% of its held-out energy in 432 of 576 dimensions and a random one
keeps 97.7%; the real parent's worst boundary is *better* than the random
parent's worst (0.957 against 0.900).  Stage 2 measured 0.9926 median held-out
retention on this same checkpoint at two adjacent boundaries, and the small
shortfall here is the deeper boundaries, not a different finding.  Whatever
breaks at layer 3 breaks in the **branch kernels**, not in the residual
crossing.

### Whole-model teacher-forced agreement, held-out prompts

1280 held-out rows each.  The child's logits are compared against the parent's
restricted to the child's 16,384 rows, on the same text at the same positions.

| run | logit relative RMSE | centered logit relative RMSE | top-1 agreement |
|---|---|---|---|
| **`L3-prefix-pretrained`** | **0.6870** | 0.7066 | **25.70%** |
| `L3-prefix-random-control` | 0.9551 | 0.9551 | 3.13% |
| `L6-prefix-random-control` | 1.0772 | 1.0772 | 1.72% |
| committed `L6-prefix-fold` (exact subspace, random) | 0.6773 | 0.6773 | 16.25% |
| committed `L2-full-rank-parent-control` (random) | 0.8771 | 0.8771 | 4.84% |

A relative RMSE above 1.0, as `L6-prefix-random-control` reaches, means the
folded child predicts the parent's logits worse than predicting zero would.
Every one of its fourteen branch transactions passed the operator's own
fail-closed guard.  The guard is per-branch; it is not a guarantee about the
assembled model.

### How much of that top-1 agreement is real

A three-layer prefix of SmolLM2-135M has an extremely concentrated argmax: over
1280 held-out rows it emits only **32 distinct top-1 tokens**, and 20.47% of
all rows are the single token `','`.  A child that emitted `','` everywhere and
nothing else would therefore score 20.47%.  The measured numbers against that
floor:

| | parent distinct top-1 tokens | modal-token floor | measured agreement | margin above floor |
|---|---|---|---|---|
| **`L3-prefix-pretrained`** | 32 | 20.47% (`','`) | 25.70% | **+5.24 points** |
| `L3-prefix-random-control` | 366 | 2.97% (`'del'`) | 3.13% | +0.16 points |
| `L6-prefix-random-control` | 230 | 4.69% (`'oons'`) | 1.72% | **-2.97 points** |

So most of the eightfold headline gap is the *target getting easier*: 20.47 of
the pretrained arm's 25.70 points are available to a child that has learned
nothing except to emit a comma.  What the fold actually bought is the remaining
**5.24 points**, against 0.16 for the random arm.

What does transfer visibly is the **shape** of the distribution: the parent's
five most frequent argmax tokens are `','` 20.5%, `'Ġ'` 19.8%, `'-'` 14.2%,
`'.'` 14.1%, `'Ġ('` 9.8%, and the child's are `','` 23.0%, `'Ġ'` 17.7%, `'-'`
6.7%, `'Ġ('` 6.2%, `'Ġother'` 4.8% — the same four tokens in the same order.
The random arm shares nothing: parent `'del' '.' 'Fi' 'Ġdeli' 'direct'`
against child `'Ġsimilarly' 'Ġobject' 'Ġreported' 'Ġlack' 'return'`.  The child
also spreads much wider than its parent (205 distinct top-1 tokens against 32),
which is the fold failing to reproduce the parent's *confidence*, not its
ranking.

Recompute the floor from the sealed artifacts with:

```python
# parent argmax over the same held-out rows the behavior stage used
counts = collections.Counter(parent_logits.index_select(2, donor).argmax(-1).flatten().tolist())
floor = counts.most_common(1)[0][1] / sum(counts.values())
```

---

## Decoded continuations — complete, verbatim

Greedy argmax, no sampling, no temperature, no beams; 32 new tokens per prompt;
16 held-out prompts (corpus indices 48..63) that no fit ever saw.  Three decodes
per prompt: the folded **child** over its own 16,384-row vocabulary; the
**parent (rows)** restricted to the child's row set, which is the only
token-for-token comparable decode; and the **parent (free)** over its full
49,152 rows.  Complete objects, including token ids and per-token surface forms,
are in `L3-prefix-pretrained/behavior.json` under `continuations`.

On this run `parent (rows)` and `parent (free)` are **identical for all sixteen
prompts** — the 16,384-row subset already contains every token the full parent
would have chosen, so the restriction costs nothing.

Greedy prefix agreement between child and parent-restricted decodes: median
**0** tokens of 32, max **2**, min 0.

### held-out prompt 0
```
prompt tail  : ...whose genuine recollections are equally inconsistent. Good practice is to treat inconsistency as a
child        :,                               
parent (rows):,  and,  and,  and,  and,  and,  and,  and,  and,  and,  and, 
parent (free):,  and,  and,  and,  and,  and,  and,  and,  and,  and,  and, 
greedy prefix agreement: 2 of 32 tokens
```

### held-out prompt 1
```
prompt tail  : ... component used by everything else removes an entire class of concern from the components above it
child        :ediat pytest uter- and otheritutes          I strugg, a,           
parent (rows): all other,   and,  and,  and,  and,  and,  and,  and,  and,  and, 
parent (free): all other,   and,  and,  and,  and,  and,  and,  and,  and,  and, 
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 2
```
prompt tail  : ... new channel crossed land that had been developed in the intervening decades. The subsequent works
child        :// lapt,                            
parent (rows): (  and,  and,  and,  and,  and,  and,  and,  and,  and,  and, 
parent (free): (  and,  and,  and,  and,  and,  and,  and,  and,  and,  and, 
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 3
```
prompt tail  : ...ucing overhead but says nothing about whether the computation was useful. Facilities can therefore
child        :, (ands and ( ( feas- and other,                     
parent (rows):,   and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
parent (free):,   and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
greedy prefix agreement: 1 of 32 tokens
```

### held-out prompt 4
```
prompt tail  : ...top. Apprenticeship transmits that kind of knowledge efficiently and everything else inefficiently
child        : isediat,aints strugg,                          
parent (rows):. , and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
parent (free):. , and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 5
```
prompt tail  : ...tated hostility but did not change policy preferences. That combination suggests the misperception
child        :,                               
parent (rows): and,   and,  and,  and,  and,  and,  and,  and,  and,  and,  and
parent (free): and,   and,  and,  and,  and,  and,  and,  and,  and,  and,  and
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 6
```
prompt tail  : ...utinely destroyed by bottom trawling before anyone knew they existed. Mapping them is now a precon
child        :ets the D I<filename>, a,                        
parent (rows): and   and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
parent (free): and   and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 7
```
prompt tail  : ...ace and encourage sloppiness about what the schema actually is. The decision is usually made early
child        :,                               
parent (rows):- and (  and,  and,  and,  and,  and,  and,  and,  and,  and,  and
parent (free):- and (  and,  and,  and,  and,  and,  and,  and,  and,  and,  and
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 8
```
prompt tail  : ...felt slow to the singers and was visible to nobody until it was suddenly complete. The passage was
child        : also,                              
parent (rows):. , and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
parent (free):. , and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 9
```
prompt tail  : ...Breaking that loop requires changing several things at once, which is difficult under budgets that
child        :                                
parent (rows):   and,  and,  and,  and,  and,  and,  and,  and,  and,  and, 
parent (free):   and,  and,  and,  and,  and,  and,  and,  and,  and,  and, 
greedy prefix agreement: 2 of 32 tokens
```

### held-out prompt 10
```
prompt tail  : ...ingly implement charge limits and thermal throttling to manage this, sometimes without telling the
child        :                                
parent (rows):  and,  and,  and,  and,  and,  and,  and,  and,  and,  and,  and
parent (free):  and,  and,  and,  and,  and,  and,  and,  and,  and,  and,  and
greedy prefix agreement: 1 of 32 tokens
```

### held-out prompt 11
```
prompt tail  : ...a distinctive abbreviation for a common word, which points to shared training. None of this is con
child        :ands/ity- and other,                         
parent (rows):- and (  and,  and,  and,  and,  and,  and,  and,  and,  and,  and
parent (free):- and (  and,  and,  and,  and,  and,  and,  and,  and,  and,  and
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 12
```
prompt tail  : ...he abstraction actually guarantees rather than which ones it appears to. Most production incidents
child        :/uesday strugg,                 I enthus (  I, aesidesuel psychiat affiliands
parent (rows): such   and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
parent (free): such   and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 13
```
prompt tail  : ...ing the dispersion accurately requires knowing the eruption column height in near real time, which
child        :akistansadays ( T─,                         
parent (rows): we,   and,  and,  and,  and,  and,  and,  and,  and,  and,  and
parent (free): we,   and,  and,  and,  and,  and,  and,  and,  and,  and,  and
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 14
```
prompt tail  : ...ic fussiness. The result is indistinguishable from the surrounding fabric, which was the point and
child        : other, eesides strugg,                          
parent (rows): ,  and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
parent (free): ,  and,  and,  and,  and,  and,  and,  and,  and,  and,  and,
greedy prefix agreement: 0 of 32 tokens
```

### held-out prompt 15
```
prompt tail  : ...vation operator and a great deal of work on bias correction. It is one of the clearest examples in
child        : a,                              
parent (rows):   and,  and,  and,  and,  and,  and,  and,  and,  and,  and, 
parent (free):   and,  and,  and,  and,  and,  and,  and,  and,  and,  and, 
greedy prefix agreement: 0 of 32 tokens
```

### Character of the outputs, and the same four prompts from the random control

The three-layer pretrained parent is a **function-word machine**.  It has
learned that after almost any English clause the likeliest next token is a
comma, a space, a hyphen, a period or `and`, and greedy decoding locks onto
`", and, and, and"` on every one of the sixteen prompts.  That is a degenerate
loop, but it is a degenerate loop made of the right tokens: it is what a real
language model looks like when you take away twenty-seven of its thirty layers.

The child folded from it is **recognizably the same kind of object**.  It opens
with `,` or a space on ten of the sixteen prompts, it emits ` and other`,
` I strugg`, ` all other`-shaped fragments, and it fills the rest with spaces.
It never produces a sentence.  What it reproduces is the parent's *register* —
punctuation and high-frequency connectives — and not its choices.

The random control at the same depth, same prompts, for contrast:

```
prompt 0  child :  agencyank massivehesdbhesfront ster ster ster ster ster ster steracking Southern ...
          parent:  chang chang chang chang chang chang transmitted chang chang civ cars bottles lack ...
prompt 1  child : erb Kat goddessgu goddessopathyarticleJohn every goddessopathyJohn every goddess Jupiter ...
          parent:  methodologyuscript methodologyuscript methodologyuscript Univers accom transmission ...
prompt 2  child :  Medicineney illnesses Web equ pestney Medicineett pestney Medicineett pestney paint ...
          parent:  locality mothers wildlife locality mothers wildlife locality mothers wildlife locality ...
prompt 3  child : working object E intended ev object E intended ev object E intended ev objectancies ...
          parent: Are� ethicspert ethicsats box box box box box box box box box box ethicsats ethics ...
```

Both are repetition loops.  The random parent's loop is built from arbitrary
mid-frequency content words (`chang`, `locality mothers wildlife`, `box`); the
pretrained parent's is built from the punctuation and connectives that actually
dominate English text.  The child inherits its own parent's kind of loop in
both arms, and only in the pretrained arm is that kind recognizable as language.
Greedy prefix agreement in the random arm is **0 of 32 on all sixteen prompts**
(max 0), against a max of 2 in the pretrained arm.

---

## Depth attribution

`L6-prefix-random-control` asks whether the layer-3 wall is caused by
*pretraining* or merely by the parent being *full rank*.  It is the same
6-layer request as the failed pretrained run, with the same tokenizer and the
same corpus, against a seeded random full-rank parent.

**It completes.  All six layers, fourteen attempts, fourteen accepted.**

| layer | attention endpoint RMSE | gated-MLP endpoint RMSE | upstream error energy ratio, attn / mlp |
|---|---|---|---|
| 0 | 0.25387 | 0.50371 | 7.41e-17 / 8.93e-03 |
| 1 | 0.30688 | 0.69185 | 1.80e-02 / 6.73e-02 |
| 2 | 0.37225 | 0.76262 | 1.97e-02 / 9.20e-02 |
| 3 | 0.43425 | 0.81471 | 2.04e-02 / 1.06e-01 |
| 4 | 0.47956 | 0.84584 | 1.99e-02 / 1.20e-01 |
| 5 | 0.52360 | 0.88834 | 2.12e-02 / 1.44e-01 |

The attention upstream ratio is **flat** at roughly 2.0e-02 from layer 1 to
layer 5.  The pretrained parent reached 1.54e-01 by layer 2 and refused at
layer 3.  Chart quality is not the difference — the random parent's held-out
retention is 0.9823 median / 0.9008 min against the pretrained parent's
0.9716 / 0.9571, so the real parent's charts are in fact *more uniformly* good.

So the wall is attributable to **the parent being pretrained**, not to the
parent being full rank.  Real learned attention is what the cross-width branch
kernel cannot carry across three residual boundaries; unlearned attention it
carries across six with the per-branch endpoint error rising slowly
(0.254 -> 0.524) and the accumulated upstream error not rising at all.  That is
the opposite of the reading in `NOTES-W7A.md` addendum 4, which observed the
flat accumulation on a random parent and inferred that depth might not be the
wall.  Depth is not the wall for random weights.  It is the wall for real ones.

The price the random arm pays for surviving is that it has nothing to transfer:
at six layers its child reaches logit relative RMSE **1.0772** — *above* the
no-op baseline of 1.0, meaning the folded child predicts the parent's logits
worse than a zero prediction would — with **1.72%** top-1 agreement against a
modal-token floor of **4.69%** for that parent.  It is not merely weak, it is
**below the floor**: emitting `'oons'` on every row would have scored better
than the folded child does.  Every one of its fourteen branches passed its own
guard and the assembled model is still worse than nothing.  The guard bounds
each branch against its own local baseline; it does not bound the composition,
and this run is the clearest evidence in this directory that those are
different claims.

---

## What limits these numbers

* **Three layers of thirty.**  The fold refuses at layer 3 on this parent, so
  three is not a budget choice, it is the measured maximum for this crossing.
  Nothing here says anything about a full-depth fold except that this
  configuration does not reach one.
* **The reverse reclosure sweep never ran.**  Every branch receipt carries
  `reverse_reclosure_sweep_executed: false`.  The design prescribes exactly one
  reverse sweep — charts frozen, writers re-solved back to front against
  terminal-anchored targets — for exactly the residual error that accumulates
  here, and nothing in `src` executes it.  Given that the wall is error
  accumulation and not chart quality, this is the single most likely thing to
  move the wall.
* **No unrelated-vocabulary crossing.**  The strict donor-row transport
  installs a subset relation.  The masked-row policy is enforced and sealed and
  again had nothing to leave alone (0 uncovered rows of 16,384).
* **The child's vocabulary is not byte complete.**  235 of 256 ByteLevel atoms,
  inherited exactly from the donor.  It cannot render bytes its parent cannot
  render either.
* **2373 fit rows against 1152 target MLP neurons**, a ratio of 2.06, the same
  neighbourhood stage 2 worked in.  Enough to be determined, not comfortable.
* **The pretrained parent is tied and the random control is not.**
  SmolLM2-135M ties `lm_head.weight` to `model.embed_tokens.weight`; a
  synthesized parent is built untied.  Both arms install the child's untied
  embedding and head from their parent's corresponding rows through two
  different interface charts, so the lane is the same, but the pretrained arm's
  two source matrices are one matrix.  This is a property of the checkpoint,
  not a choice.
* **One head geometry.**  `head_dim` must match the parent exactly or the
  planner gaps the crossing on the rotary node, and dense qwen3 refuses
  `num_key_value_heads == 1` or `== num_attention_heads`.  6/3 was chosen, so
  the key/value width is held and only the query heads contract.
* **The interface operator is not in `src`,** and neither is the subset
  compiler.  See `NOTES-W7A.md` sections 2 and 7.

## What this does and does not establish

Established, at real geometry, from a **real pretrained checkpoint**, by
artifacts that verify:

* A real SmolLM2-135M parent and a blank qwen3 child differing in **family**,
  **residual width** (576 -> 432), **head count** (9 -> 6 query heads), **MLP
  width** (1536 -> 1152), **Q/K normalization** (identity -> per-head RMS,
  fitted) and **vocabulary** (49,152 -> 16,384) can be planned gap-free and
  installed completely by eight closed-form transactions in one executor run,
  to a depth of three layers.
* From **one** parent forward, with **zero** parent forwards during execution
  and zero optimizer steps, backward calls, gradient calls, pruning or
  quantization operations, under a static source audit that also covers the
  locally composed operator.
* The residual charts are **not** the limit: a real pretrained residual keeps
  97.2% of its held-out energy in 432 of 576 dimensions, comparable to a random
  one and consistent with stage 2's 0.9926 on the same checkpoint.
* The branch kernels **are** the limit, and on a real parent their error
  accumulates instead of settling.  The fail-closed guard refuses layer 3's
  attention branch at endpoint relative RMSE 1.09 against a no-branch baseline
  of 1.0, and the child is rolled back to blank rather than left partially
  folded.
* **The wall is caused by the parent being pretrained, not by it being full
  rank.**  A seeded random full-rank parent, identical in every other respect
  including the tokenizer, folds all six layers with a flat error accumulation
  (2.0e-02 from layer 1 to layer 5) where the real parent reached 1.5e-01 by
  layer 2.
* Pretraining changes the behavioral outcome in the direction it should:
  0.687 logit relative RMSE against 0.955, 25.70% top-1 agreement against
  3.13%, and decoded output whose register matches the parent's.
* **A per-branch guard is not a guarantee about the assembled model.**  The
  six-layer random control passed all fourteen of its branch guards and
  assembled a child whose logit relative RMSE is 1.077, worse than predicting
  zero.

Not established, and not claimed:

* **Behavior transfer.**  A 0.69 relative RMSE against the teacher's own logits
  is a failure to reproduce the parent, and a +5.24 point margin over a
  modal-token floor is a small signal, not a working model.
* **That the child writes English.**  It does not.  Neither does its parent at
  this depth.
* **That depth beyond three is unreachable in principle.**  It is unreachable
  for *this* crossing, as configured, with no reverse reclosure sweep, a 0.75x
  residual contraction and a 9 -> 6 query-head contraction.  Which of those
  moves the wall is untested.
* **That a folded child is useful.**  The six-layer random control scores below
  its own modal-token floor.
* **That the vocabulary crossing is hard.**  A subset target makes the row map
  exact.

---

## Reproduction

```bash
# the run that failed closed, kept because the refusal is the measurement
HF_HUB_OFFLINE=1 python experiments/run_cross_family_real_scale_fold.py \
    --report-root reports/cross-family-real-20260804-pretrained/L6-prefix-pretrained \
    --parent-weights local-pretrained \
    --parent-model-directory "C:/Users/cereal/Documents/ai training/models/SmolLM2-135M" \
    --layers 6 --calibration-prompts-count 48 --holdout-prompts-count 16 \
    --evidence-window 128 --holdout-window 80 --decode-new-tokens 32 --threads 6

# the headline run
HF_HUB_OFFLINE=1 python experiments/run_cross_family_real_scale_fold.py \
    --report-root reports/cross-family-real-20260804-pretrained/L3-prefix-pretrained \
    --parent-weights local-pretrained \
    --parent-model-directory "C:/Users/cereal/Documents/ai training/models/SmolLM2-135M" \
    --layers 3 --calibration-prompts-count 48 --holdout-prompts-count 16 \
    --evidence-window 128 --holdout-window 80 --decode-new-tokens 32 --threads 6

# the single-variable controls: identical but for the parent's weights
HF_HUB_OFFLINE=1 python experiments/run_cross_family_real_scale_fold.py \
    --report-root reports/cross-family-real-20260804-pretrained/L3-prefix-random-control \
    --parent-weights seeded-random-init \
    --source-tokenizer "C:/Users/cereal/Documents/ai training/models/SmolLM2-135M" \
    --layers 3 --calibration-prompts-count 48 --holdout-prompts-count 16 \
    --evidence-window 128 --holdout-window 80 --decode-new-tokens 32 --threads 6

HF_HUB_OFFLINE=1 python experiments/run_cross_family_real_scale_fold.py \
    --report-root reports/cross-family-real-20260804-pretrained/L6-prefix-random-control \
    --parent-weights seeded-random-init \
    --source-tokenizer "C:/Users/cereal/Documents/ai training/models/SmolLM2-135M" \
    --layers 6 --calibration-prompts-count 48 --holdout-prompts-count 16 \
    --evidence-window 128 --holdout-window 80 --decode-new-tokens 32 --threads 6
```

The runner is resumable: each stage seals a receipt whose `input_sha256` chains
the stages above it, and a stage is reused only when nothing it depends on has
moved.  The folded child is saved under `<run>/child` and reloaded against its
sealed state fingerprint, so the behavior stage can be re-run without refolding.

Input corpus:
`reports/kernel-stage2-20260801/calibration_prompts_64.json`
(`sha256 45852453b3bbdcf91ecb9ca785d4e3aa340138b7766aad6e0d2855ce28bf3791`).

Schema test: `tests/test_cross_family_real_scale_fold_seal.py` — 34 tests,
including the pretrained-checkpoint geometry verification, the refused argument
combinations, the strict donor-row source, and the invariant that none of this
moved a synthesized run's identity.
