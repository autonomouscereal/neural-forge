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
