# NOTES-W7A — real-scale cross-family fold: what `src` cannot do yet

Wave W7A elevated the cross-family fold proof from the H32 toy in
`tests/test_cross_family_causal_fold_execution.py` to a thirty-layer H576 -> H432
crossing driven by `experiments/run_cross_family_real_scale_fold.py`.  Three
things stopped inside `src` rather than being worked around silently.  Each one
is written here with the exact symbol, the exact message, and what a fix would
have to touch, because the task owns new files only.

---

## 1. No pretrained parent is reachable from this workstation

**Status: sub-item stopped. Not a source defect — an environment fact.**

The task assumed SmolLM2-135M was in the local Hugging Face cache, as it is on
the AI server where `reports/kernel-stage2-20260801` was produced.  It is not
here.  The complete local model cache is:

```
C:/Users/cereal/.cache/huggingface/hub/
  models--Systran--faster-whisper-large-v3        (CTranslate2 ASR, not a causal LM)
  models--Systran--faster-whisper-tiny.en         (CTranslate2 ASR, not a causal LM)
  models--hf-internal-testing--tiny-random-GPT2LMHeadModel
  models--hf-internal-testing--tiny-random-LlamaForCausalLM
  models--hf-internal-testing--tiny-random-MistralForCausalLM
  models--sshleifer--tiny-gpt2
```

A filesystem-wide search of `C:\Users\cereal`, `D:\` and `E:\` for
`*.safetensors`, `pytorch_model*.bin`, `*.gguf` and `model.safetensors.index.json`
returned no pretrained decoder weights of any size — only the tiny random test
fixtures above.  `HF_HUB_OFFLINE=1` forbids downloading, and the task forbids
SSH to the server that does hold the checkpoint.

Consequences, all of them recorded in the run's `configuration.json` seal so no
receipt can be mistaken for a pretrained one:

* `parent_weights = "seeded-random-init"` is the only implemented mode.  The
  parent is built at the exact SmolLM2-135M geometry (H576, 30 layers, 9 query
  heads over 3 key/value heads, head_dim 64, intermediate 1536, RoPE base
  100000, RMSNorm epsilon 1e-5) and initialized from a sealed seed.
* The tokenizer *is* real: `sshleifer/tiny-gpt2` ships the complete 50257-row
  GPT-2 ByteLevel-BPE tokenizer (`vocab.json` 898 KB, `merges.txt` 456 KB), and
  the child's independent vocabulary is a real compiled ByteLevel-BPE subset of
  it, so the vocabulary crossing is not simulated.
* A randomly initialized parent has a nearly isotropic residual spectrum.  The
  balanced controllability/observability POD that selects each boundary chart
  therefore has almost nothing to keep, and the H576 -> H432 crossing is not a
  compression of structure but a straight loss of a quarter of the coordinates.
  Measured directly: with a **full-rank** random parent at the tiny geometry
  (H128 -> H96, two layers) the very first MLP branch fails the promoted
  operator's own fail-closed guard —

  ```
  DynamicCausalOperatorPromotionError: Cross-width causal gated_mlp branch
  operator endpoint relative RMSE 1.0297561747172 is at or above the no-branch
  baseline bound 1.0 at boundary 'layer001.attention_in'.
  ```

  That is the guard doing its job, and it is the honest answer to "what does a
  cross-width fold of an unstructured parent buy": nothing, and the operator
  says so instead of installing a regression.
* The published run therefore uses `--parent-residual-rank 432`, the
  exact-subspace control `tests/test_cross_width_causal_fold_execution.py`
  already relies on, lifted to real geometry.  Every residual-facing parent
  weight is confined to one 432-dimensional subspace and every RMSNorm gain is
  set to one, so the parent's residual states lie exactly in the child's width.
  A chart that finds that subspace discards nothing, which makes every endpoint
  number a statement about the branch kernels rather than about how much a
  random matrix can be compressed.

**What would lift this:** the checkpoint, nothing else.  The driver needs no
change to consume a pretrained parent beyond replacing `_build_parent` with a
`from_pretrained(..., local_files_only=True)` load; the `--parent-weights`
choice list exists for exactly that.

---

## 2. No `src` operator can install the interface action of a simultaneous
   family + width crossing

**Status: worked around in the new runner file. A source change is required to
fix it properly.**

`DynamicFoldPlanner` emits exactly one `token_interface_bridge` action per plan.
Its target bundle depends on the crossing:

| crossing | bundle source | target tensors |
|---|---|---|
| vocabulary only | `_global_token_interface_bundle` | `model.embed_tokens.weight`, `lm_head.weight` |
| width only | `_cross_width_terminal_interface_bundle` | the two above **plus** `model.norm.weight` |
| vocabulary **and** width | `_cross_width_terminal_interface_bundle` | the three above |

`src/neural_forge/dynamic_fold_planner.py:2857-2872` applies the bridge bundle
first and then lets the chart bundle overwrite it, so under a simultaneous
crossing the action owns three tensors and is marked
`boundary_chart_required=True` with
`fit_objective="byte_span_and_semantic_token_interface"`.

There are three registered operators for that method id, and none of them can
execute that action:

* `CrossWidthTokenHeadInterfaceOperator`
  (`dynamic_causal_operator_promotion.py:1605-1612`) owns all three tensors but
  refuses anything except one shared tokenizer identity:
  *"requires one shared tokenizer identity; an independent vocabulary needs a
  sealed tokenizer bridge as well."*
* `StrictTokenInterfaceBridgeOperator` (`:1865`) and
  `PairCodeTokenInterfaceBridgeOperator` (`:2135`) both call `_installed_paths`
  with only the embedding and the head, which fails the exact-match contract at
  `:406`: *"layout paths differ from the planned target parameters."*

The runner therefore defines
`ChartProjectedPairCodeTokenHeadInterfaceOperator` locally.  It composes the two
halves without inventing any math: the vocabulary rows come from the sealed pair
bridge under the same anchor-preferred blend
(`compile_pair_bridge_anchor_transport` over
`compile_pair_bridge_length_normalized_transport`) and the same
`leave-uncovered-target-rows-at-incumbent-blank` masked-row policy the promoted
pair operator uses; the coordinates come from the two interface-locked charts,
which `ResidualBoundaryChart` already advertises as dense vocabulary projectors
(`hf_causal_cross_width_charts.py:534-543`); and the final-norm gain is the same
`fit_rmsnorm_diagonal` solve on chart-encoded parent states that
`CrossWidthTokenHeadInterfaceOperator` performs.

**What a proper fix would touch.**  The cleanest shape is to give
`PairCodeTokenInterfaceBridgeOperator` (and the strict one) the optional
final-norm ownership the chart lane already has: accept a `chart_set`, and when
the action carries the third tensor and is marked `boundary_chart_required`,
install the chart-projected rows plus the fitted final-norm gain.  That is one
new constructor argument and one branch in `prepare`, plus a planner-contract
test that a three-tensor interface action is only ever emitted with
`boundary_chart_required=True`.  It was not attempted here because W7A owns new
files only.

---

## 3. Qwen3 blank targets cannot use a single key/value head

**Status: not a defect. Recorded because it silently constrains the child.**

`build_decoder_layout` rejects a Qwen3 child whose `num_key_value_heads` is one,
or equal to `num_attention_heads`:

```
UnsupportedDecoderLayoutError: Unsupported qwen3 decoder layout: dense Qwen3
requires grouped-query attention with more than one num_key_value_heads
strictly dividing num_attention_heads.
```

surfacing as `DynamicModelInspectionError: The Hugging Face decoder failed its
complete topology contract.`  Combined with the planner's contraction-only
contract in `_family_neutral_causal_layer_contract` (every one of
`query_heads`, `key_value_heads`, `head_dim`, `intermediate_size` must not
exceed the parent's), a SmolLM2-shaped parent with 9 query heads over 3
key/value heads admits child head counts of 6/3, 6/2, 9/3 and little else.  The
published run uses 6 query heads over 3 key/value heads, which is a real query
contraction with the key/value width held.

---

## Concurrency note

While this wave ran, a second agent process was executing the same driver on the
same workstation (`--report-root .../scratchpad/real2 --layers 2`).  The two
runs share six CPU cores, so the wall-clock timings in
`reports/cross-family-real-20260804/fold.json` are contended and should be read
as upper bounds.  Nothing was shared between the runs except the CPU: the report
roots are disjoint and every artifact is content addressed.

---

## Addendum (second agent, `reports/cross-family-real-20260803`)

Written by the Claude harness working the same wave concurrently.  Everything
above is the other process's and has not been edited.  Sections 1-3 above match
what I measured independently; this addendum only adds what they do not cover.
My artifacts are in `reports/cross-family-real-20260803/` (README there);
theirs are in `reports/cross-family-real-20260804/`.  Shared files
(`experiments/run_cross_family_real_scale_fold.py`,
`tests/test_cross_family_real_scale_fold_seal.py`) have two authors; I created
them and left every edit made after 08:35 in place, including the
`rope_theta_kilo` correction, which is right — SmolLM2-135M does use base
100000.

### 4. Depth accumulation was the predicted wall and was not the actual wall

A tiny-geometry rehearsal (H128 -> H96, two layers) showed
`fit_upstream_error_energy_ratio` multiplying by roughly ten per residual-write
boundary, which predicts the fail-closed endpoint bound tripping within a
handful of layers.  At real geometry (H576 -> H432, exact-subspace parent,
2399 fit rows) it does not:

| boundary | attention upstream ratio | gated-MLP upstream ratio |
|---|---|---|
| layer 0 | 7.63e-17 | 2.72e-03 |
| layer 1 | 1.04e-02 | 2.86e-02 |
| layer 2 | 1.01e-02 | 3.68e-02 |
| layer 3 | 9.05e-03 | 3.35e-02 |
| layer 4 | 7.46e-03 | 2.95e-02 |
| layer 5 | 6.13e-03 | 2.69e-02 |

It rises for two layers and then **declines**.  Six boundaries composed with
every branch accepted (`reports/cross-family-real-20260803/L6-prefix-fold`,
14 attempts, 14 accepted, `status: complete`).  The live sequential recapture
plus the per-boundary chart bounds the accumulated discrepancy instead of
compounding it.  Depth beyond six is still unproven, but the extrapolation that
said it was hopeless came from a geometry too small to trust.

The binding constraint is the **gated-MLP branch**, not depth and not the family
change: its held-out endpoint relative RMSE runs 0.359 -> 0.686 across those six
layers while attention runs 0.122 -> 0.360.  Stage 2 saw the same ordering on
the real pretrained parent (0.0759 attention against 0.6657 MLP).  The
1536 -> 1152 SwiGLU contraction is where this fold loses most of what it loses.

### 5. The full-rank random parent is not as hopeless as section 1 concludes

Section 1 above reasons that a random parent has a nearly isotropic residual
spectrum and that the crossing is therefore "a straight loss of a quarter of the
coordinates", citing a tiny-geometry fail-closed.  Measured at real geometry
(`reports/cross-family-real-20260803/L2-full-rank-parent-control`), a full-rank
random SmolLM2-shaped parent keeps **97.5% of its held-out residual energy in
432 of 576 dimensions** (min 89.6%), and the two-layer fold completes with every
branch accepted.  RMSNorm plus residual accumulation induce real anisotropy even
without training.  It is worse than the exact-subspace control — cycle defect
0.126 against 1.3e-07, teacher-forced logit relative RMSE 0.877 against 0.566,
top-1 agreement 4.8% against 23.0% — but it is not a wall.  The tiny-geometry
failure that motivated the confinement was confounded by starved fit rows
(192 fit rows against 192 target MLP neurons); the same configuration with
480 fit rows completes.

### 6. The behavioral answer is negative and should be recorded as such

At six layers, exact-subspace parent, the folded child reaches teacher-forced
logit relative RMSE **0.677** against the row-aligned parent with **16.25%**
top-1 agreement on held-out text, and its greedy continuations agree with the
parent's for a median of **0 tokens of 32** (15 of 16 prompts diverge on the
first token).  The mechanism installs a complete runnable child; the child does
not reproduce the parent.  Both collapse into repetition loops, because the
parent is not pretrained, so no statement about text quality is available from
this wave in either direction.

---

## Addendum 2 — the pretrained parent arrived (`reports/cross-family-real-20260804-pretrained`)

Section 1's blocker is gone.  Real SmolLM2-135M weights are on this workstation
at `C:/Users/cereal/Documents/ai training/models/SmolLM2-135M`
(`config.json`, `model.safetensors` 269,060,552 bytes
`sha256 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`,
complete tokenizer files), so `experiments/run_cross_family_real_scale_fold.py`
grew `--parent-weights local-pretrained` and `--parent-model-directory`.  The
mode is additive: a `seeded-random-init` run's configuration identity is
byte-identical to what it was before, which I checked by reproducing the
committed `L6-prefix-fold` configuration `input_sha256`
(`53abd116e9f28947fc5368cc0a9cdb7c9dd357bb6430d6b393cbfa166fdc4282`) with the
new driver.  Every field the new mode seals -- the weight digest, the config
digest, the eight verified geometry fields -- is conditional on the mode being
active, and `tests/test_cross_family_real_scale_fold_seal.py` pins that
invariant as a key-set assertion rather than a hash a later reader cannot
recompute.

Section 1 predicted the driver "needs no change to consume a pretrained parent
beyond replacing `_build_parent` with a `from_pretrained(...)` load".  That
part was true and took twenty lines.  What section 1 could not have known is
that the *tokenizer* half is where `src` stops.

### 7. No sealed tokenizer pair bridge can exist for this checkpoint

**Status: blocked in `src`, worked around locally in the runner.  A source
change is required to fix it properly.**

The task's design was: use the parent's own real 49,152-row ByteLevel-BPE
tokenizer as the pair-bridge source, prune a 16,384-row subset of it as the
child's vocabulary, install through the sealed pair bridge exactly as the
20260803 runs did.  Both halves of that refuse, for the same underlying reason.

**7a. `prune_hf_tokenizer` refuses the donor.**
`hf_tokenizer_prune.py:389-405` requires the donor's BPE vocabulary to contain
all 256 ByteLevel atoms as their own rows before it will select anything:

```
HFTokenizerPruneError: donor BPE vocabulary lacks required ByteLevel byte
atoms: 'À', 'Á', 'ñ', 'ò', 'õ', 'ö', '÷', 'ø'
```

SmolLM2-135M's real tokenizer carries **235 of the 256**.  Twenty-one atoms --
`À Á ñ ò õ ö ÷ ø ù ú û ü ý þ ÿ Ą Ć ē Ĕ Ė ĝ` -- exist only inside merged rows,
so the donor itself cannot render those bytes.  The guard is asking a subset to
be *more* byte-complete than the vocabulary it is a subset of, which no subset
can be.

**7b. `compile_hf_tokenizer_pair_bridge` refuses the pair, in both
directions.**  Even with a subset compiled some other way, the bridge itself
demands complete byte coverage of whichever vocabulary it is segmenting into
(`hf_tokenizer_pair_bridge.py:882-897`), and it builds *two* code tables --
forward source->target at `:1083` and reverse target->source at `:1089` -- so
both vocabularies must be complete:

```
TokenizerPairBridgeError: target tokenizer does not cover the complete 256-row
ByteLevel byte alphabet; single-byte rows are missing for 0x04, 0x06, 0x13,
0x14, 0x16, 0x1d, 0xc0, 0xc1. A pair bridge has no exact fallback without them.
```

That is a correctness guard and it is right: without a single-byte row there is
no exact fallback for a token that does not anchor.  But it means **no pair
bridge involving this checkpoint's tokenizer can exist at all**, in either
role, and therefore no `PairCodeTokenInterfaceBridgeOperator` lane and no
chart-projected pair-code lane either.  The 20260803 runs never hit this
because `sshleifer/tiny-gpt2` ships the complete GPT-2 vocabulary, which does
carry all 256.

**What the runner does instead.**  Two local compositions, both of which reuse
`src`'s own steps and change no math:

* `compile_donor_byte_coverage_subset` is `prune_hf_tokenizer`'s pipeline --
  `_parse_merges`, `_merge_dependency_closure`, `_select_token_ids`,
  `_pruned_tokenizer_data`, `_build_hf_tokenizer`, `_verify_target`, and the
  same `build_strict_token_transport_profile` proof -- with the required-row
  seed changed from "all 256 atoms" to "every atom the donor has".  The subset
  inherits the donor's byte coverage exactly: 235 of 256, sealed in the
  tokenizers receipt under `byte_level_coverage` so nobody can read the child
  as byte complete.
* `SealedStrictDonorRowInterfaceRows` replaces the pair codes as the interface
  operator's row source.  The subset relation gives an exact one-row-to-one-row
  map, which `build_strict_token_transport_profile` proves from the donor's own
  token surfaces, so each target row's evidence is the donor row it came from.
  Everything else in `ChartProjectedPairCodeTokenHeadInterfaceOperator` is
  untouched: the same two interface-locked charts project the rows, the same
  masked-row policy is enforced and sealed, the same `fit_rmsnorm_diagonal`
  solve fits the final-norm gain.

Against a subset target the two lanes are close by construction: every target
row is byte-identical to exactly one donor row, so the pair lane's
anchor-preferred blend takes the anchor branch wherever it anchors and the
anchor branch is that donor row.  I did not instrument how many of the
20260803 run's 16,384 rows anchored rather than averaging, so "close" is as
strong as this should be put; both lanes do report `uncovered_output_rows: 0`.
It **is** a weaker claim about the crossing: an unrelated-vocabulary crossing
is no longer being exercised, only a subset one, and the masked-row policy
again had nothing to leave alone.

**What a proper fix would touch.**  Two independent changes, neither attempted
here because this wave owns the driver and its test only:

1. `hf_tokenizer_prune._required_token_ids` should seed with
   `alphabet & vocabulary.keys()` and record the donor's coverage on the
   result, instead of refusing the donor.  A subset cannot be more complete
   than its donor and demanding it is unsatisfiable; the honest contract is
   "the subset preserves the donor's byte coverage exactly".  That is one line
   plus a field on `HFTokenizerPruneResult` and a test that a byte-incomplete
   donor prunes and reports its gap.
2. `hf_tokenizer_pair_bridge._segmentation_dictionary` should be allowed to
   return a partial segmenter and let `_build_code_table` mark the rows it
   cannot segment as uncovered, which is exactly what the masked-row policy
   downstream already exists to handle.  Today an uncoverable byte is a fatal
   error for the whole pair; it should be an uncovered *row*.  That is a real
   design change to the code-table contract, not a one-liner, and it should
   carry a test that a deliberately byte-incomplete pair produces uncovered
   target rows rather than an exception.

### 8. A layer prefix of a pretrained parent is not a pretrained language model

Worth stating plainly because it bounds what the decoded text in
`reports/cross-family-real-20260804-pretrained` can mean.  `--layers 6` keeps
the first six of SmolLM2-135M's thirty decoder layers and then applies the real
`model.norm` and the real `lm_head` to whatever those six produce.  Every weight
in that object is pretrained, and the object as a whole was never trained to
emit a distribution from layer six.  So a six-layer prefix run answers *does
the child reproduce this parent* on real, structured, anisotropic weights --
which is the question the fold is actually about -- and it does **not** answer
*can a folded child write English*.  Only a full-depth run answers the second
question, and the depth actually run is sealed in every receipt as
`folded_layer_prefix` / `full_depth` / `bounded_layer_prefix`.

### 9. The residual confinement is refused for a pretrained parent

`confine_parent_residual` rewrites every residual-facing weight into one random
rank-432 subspace.  Applied to a checkpoint it produces a model that is not the
checkpoint while the receipt still says `pretrained: true`, so
`--parent-residual-rank` and `--parent-weights local-pretrained` are refused
together.  The consequence for reading the numbers is that the pretrained runs
are **full rank**, so the committed `L2-full-rank-parent-control` is their
chart-side baseline and the committed `L6-prefix-fold` (exact subspace) is
their depth-side baseline.  Neither is a single-variable A/B, which is why
`reports/cross-family-real-20260804-pretrained` runs its own random-parent
controls at the same depth and through the same tokenizer.

### 10. Addendum 4's reading of depth was right about random weights and wrong about real ones

Addendum 4 above observed that the upstream error energy ratio rose for two
layers and then declined on an exact-subspace random parent, and concluded that
"the extrapolation that said it was hopeless came from a geometry too small to
trust".  Measured on the real checkpoint, the extrapolation was right and the
geometry was not the confound -- the *parent* was:

| | real SmolLM2-135M | seeded random, same tokenizer and depth |
|---|---|---|
| attention endpoint RMSE, layers 0/1/2 | 0.131 / 0.760 / 0.980 | 0.253 / 0.315 / 0.384 |
| attention upstream ratio, layer 2 | 1.54e-01 | 2.08e-02 |
| deepest layer folded | **refused at layer 3** | 6 of 6 accepted |
| held-out chart retention, min / median | 0.9571 / 0.9716 | 0.9008 / 0.9823 |

The real parent's *first* attention boundary folds nearly twice as well as the
random one, and then its error accumulates seven times faster and trips the
promoted operator's fail-closed bound at layer 3:

```
DynamicCausalOperatorPromotionError: Cross-width causal attention branch
operator endpoint relative RMSE 1.0923883067551874 is at or above the no-branch
baseline bound 1.0 at boundary 'layer003.mlp_in'.
```

The chart is not the difference -- the real parent's charts are the more
uniformly good of the two.  What the cross-width branch kernel cannot carry is
real learned attention across three residual boundaries.

One more thing that only shows up with a control at depth: the six-layer random
run accepted all fourteen branches and produced a child whose teacher-forced
logit relative RMSE is **1.077**, above the 1.0 no-op baseline.  Every branch
cleared its own local guard and the assembled model is still worse than
predicting zero.  Nothing in `src` bounds the composition, only the branches,
and the reverse reclosure sweep the design prescribes for exactly this residual
still does not exist.
