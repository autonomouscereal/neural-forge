# LFM2 a0004 4x Fold: Manual Intelligence Result

Date: 2026-07-22

This is the behavioral result for the three-lane `a0004` fold. It records a
failed experiment, not a rejection produced by a numeric threshold.

## What was physically proved

Each lane folded the same local LFM2.5 230M parent into a blank, smaller LFM2
target with `57,405,368` dense BF16 parameters. The parent has `229,693,184`
parameters, so the physical parameter-count ratio is approximately `4.001249x`.
The child artifacts are standalone checkpoints; the result does not use
quantization, pruning, sparsity, parameter sharing, a parent sidecar, training,
optimizer steps, backward passes, or regenerated parent traces.

The three complete body recipes were:

- `fit_guided`
- `full_strength_structural`
- `residual_writer`

All three structural publications passed the independent postfold validator.
Each terminal solve retained and materialized 36 dense candidate children, for
108 standalone candidates in total.

## How intelligence was judged

Every candidate generated a complete response for the same 16-prompt semantic
pack, with up to 256 new tokens per response. This produced `108 * 16 = 1,728`
decoded responses. Three independent lane reviews then read all responses in
full against task truth. Diagnostic metrics were not used to accept, reject,
rank, or estimate intelligence.

The manual review artifacts and their SHA-256 hashes are:

| Lane | Responses read | Manual review SHA-256 |
|---|---:|---|
| `fit_guided` | 576 | `ae5629d20454157458eca74275a2489ff01cfaadd410c22396400b5c61503bef` |
| `full_strength_structural` | 576 | `aadff5df7a863627c99737997eb4b8561d37f94e4617f59f63b9b676e137fdab` |
| `residual_writer` | 576 | `c7e4b1597a2363956d5073557345a79888b7f365f627d1cc0b5bd8c6f2f7c6ba` |

## Authoritative result

All 1,728 responses failed semantically. There was no coherent candidate, no
partial island worth retaining as a behavioral success, and no defensible best
candidate in any lane. Failure begins at the first generated tokens: malformed
fragments, tokenizer-like word salad, prompt-insensitive repetition, and broken
sentence formation. It is therefore not merely a long-horizon decay problem,
an alternate-but-correct wording issue, or a failure that appears only after a
64-token boundary.

The `a0004` experiment proves real 4x physical dense compaction and a complete
automatic fold/materialization/evaluation path. It does **not** prove retained
intelligence or a coherent child. The roughly-90%-retention objective remains a
holistic target, not a hard score cutoff.

## Next bounded experiment

The next lane is `source_producer_residual_writer`, replayed from the same sealed
blank-target foundation rather than spliced from an `a0004` checkpoint. It
combines:

- full-alpha source-derived attention Q/K;
- fixed-source V plus an analytic residual O writer;
- the full-source short-convolution producer; and
- full-source MLP W1/W3 producers plus the analytic W2 residual writer.

This tests whether producer coverage and child-side readout repair must be
present together throughout one causal body. It remains a closed-form,
no-training fold. Every structurally valid terminal candidate will be retained
for decoded whole-response review; numeric diagnostics may guide inspection but
cannot decide the result.
