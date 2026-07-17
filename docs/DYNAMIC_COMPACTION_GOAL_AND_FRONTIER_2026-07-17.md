# Dynamic Compaction Goal and Frontier

Date: 2026-07-17

## Operational goal

Build one automatic Neural Forge system that can inspect one or more local
parent models, construct an explicitly requested genuinely smaller dense blank
target, and fold the parent circuits into that target without pruning,
quantization, gradient training, or caller-written per-layer surgery.

The long-term capability envelope is:

- any registered parent family, architecture, tokenizer, and modality;
- one or more parents into one requested child architecture and parameter
  budget;
- cross-family and cross-tokenizer bridges;
- modality preservation, removal, or addition when the registered target and
  evidence provide the required interfaces; and
- increasingly aggressive dense parameter ratios, including 2x, 4x, 10x, and
  larger ratios when the source capability is sufficient.

Registration is allowed for a new architecture or modality. Per-run manual
neuron surgery is not.

## Current severe-compaction execution objective

Marginal parameter removal and small-ratio controls do not satisfy the current
project objective.  The active real-model proof sequence is:

1. fold the local 229,693,184-parameter LFM2.5 parent into a separately
   instantiated 2,291,840-parameter dense blank child (100.222x);
2. save and reload that child with no parent, sidecar, sparse state, or runtime
   reconstruction dependency;
3. require coherent free-running responses and score retained capability over
   complete responses on disjoint evidence;
4. after the 100x compiler path is structurally and behaviorally understood,
   fold the same parent into the 916,928-parameter dense blank child (250.503x);
5. generalize the proven compiler components through registered architecture,
   tokenizer, and modality adapters rather than caller-written layer surgery.

The 100x target preserves all 14 ordered LFM2 operators with hidden width 128,
FFN width 175, two query heads, one key/value head, and head dimension 64.  The
250x target preserves all 14 ordered operators with hidden width 64, FFN width
166, two query heads, one key/value head, and head dimension 32.  Both use the
sealed 4,096-row child tokenizer bridge.  Their sizes count dense learned
parameters, not bytes.

The 2x, 4x, and 10x plans remain useful controls and diagnostics.  They are not
allowed to be reported as completion of the severe-compaction goal.

## Non-negotiable acceptance

A promoted result must satisfy all of the following:

1. The target begins as a compiler-authenticated dense blank.
2. The learned parameter count is the requested genuinely smaller count.
3. Parameter reduction is not credited to dtype, packing, sparsity,
   quantization, pruning, external adapters, or shared parent storage.
4. Folding performs zero optimizer steps, backward calls, or live parent
   forwards when sealed parent evidence is available.
5. The saved child is standalone and survives a fresh local reload.
6. Free-running whole responses are coherent and retain at least 90% of the
   parent-demonstrated task capability on a sealed, disjoint evaluation.
7. Exact token/text equality, local RMSE, first-token agreement, and parameter
   arithmetic are diagnostics only.

Until item 6 passes, a result is an experiment or a structural proof, not a
successful intelligent compaction.

## Current exact dense topology frontier

For the real 229,693,184-parameter LFM2.5-230M parent, the capability-aware
planner currently selects these full-depth, compiler-supported targets:

| Requested ratio | Dense target parameters | Realized ratio | Topology |
|---|---:|---:|---|
| 2x | 114,838,464 | 2.00014x | hidden 576, depth 14, Q8/KV4, head 64, FFN 2527 |
| 4x | 56,984,048 | 4.03083x | hidden 304, depth 14, Q4/KV2, head 64, FFN 2560 |
| 10x | 22,967,168 | 10.00094x | hidden 128, depth 14, Q2/KV1, head 64, FFN 2558 |

These selections preserve full layer/operator order and have zero missing
declared compiler transforms. They prove target feasibility only; they do not
imply behavioral or intelligence retention.

The automatic production stack now has exact end-to-end blank-target
structural proofs at three ratios:

| Requested ratio | Parent parameters | Dense blank child parameters | Realized ratio | Child topology |
|---|---:|---:|---:|---|
| 2x | 80,992 | 40,312 | 2.00913x | hidden 40, FFN 117, depth 2 |
| 4x | 80,992 | 20,096 | 4.03025x | hidden 32, FFN 59, depth 2 |
| 10x | 2,535,008 | 252,224 | 10.05062x | hidden 32, FFN 148, depth 14 |
| 100x | 1,285,568 | 12,848 | 100.05978x | hidden 32, FFN 32, depth 2 |
| 250x | 3,212,864 | 12,848 | 250.06725x | hidden 32, FFN 32, depth 2 |
| 1000x | 12,848,000 | 12,848 | 1000.00000x | hidden 32, FFN 32, depth 2 |

For every row, the same production path performs automatic budget selection,
compiler-authenticated zero initialization, dense folding, child-root
execution, publication, fresh reload, and state fingerprint verification.
All six preserve the requested depth/operator sequence, use no training,
pruning, or quantization, and leave no missing registered compiler transform.
The 250x case represents the ratio in the user's 1-trillion-to-4-billion
example and removes 99.6001076% of learned parameters. The 1000x case removes
99.9%. Both retain singleton source spans for every target layer. These runs
prove that the automatic size compiler and artifact transaction work from 2x
through three orders of magnitude. Synthetic executability does not establish
semantic coherence or intelligence retention.

## Current real-model behavioral frontier

The completed dense LFM2 child is:

```text
229,693,184 -> 114,067,328 parameters
50.339263% removed
2.014x smaller
BF16 dense learned state
```

It is not coherent and is therefore rejected:

| Basis/candidate | Parent-nonempty retention | New empties | New generation-limit saturations | Task retention |
|---|---:|---:|---:|---:|
| Weight-complete base | 45.83% | 13 | 23 | 0% |
| Circuit-balanced base | 54.17% | 11 | 16 | 0% |
| Circuit-balanced plus cached vocabulary-margin geometry | 79.17% | 5 | 23 | 0% |
| Layer 2 plus fitted tied-head transform | 0% | 24 | 23 | 0% |
| Layer 2 with tied head byte-identical | 0% | 24 | 23 | 0% |
| Layer 2 coupled V/O with tied head byte-identical | 0% | 24 | 23 | 0% |
| Coupled layers 0 and 2 plus fitted tied-head transform | 37.5% | 15 | 23 | 0% |

Vocabulary-margin observability is the best current substrate and the first
large movement in the free-running gate: it retained 19 of the 24 parent
nonempty responses instead of 13 for the prior circuit-balanced basis.
However, all 28 cases still regressed on repetition, all 28 saturated the
64-token evaluation stopping limit, and task retention remained zero. It is
therefore progress toward a compaction, not a successful intelligent
compaction.

## Diagnosed collapse and next compiler lane

The real child repeatedly selects a very small whitespace/punctuation token
set, especially token IDs 730, 523, and 509. This is evidence of a
vocabulary-margin geometry collapse. Preserving residual energy alone is not
enough. The layer-2 identity-head ablation produced token 730 first on all 28
cases even though the tied input embedding/output-head bytes were unchanged.
The body fold, rather than the terminal head rewrite, therefore crossed the
vocabulary decision boundaries.

The active zero-training lane combines:

1. a circuit-balanced residual basis;
2. cached parent top-1/top-k vocabulary-margin observability so the basis
   preserves token decision directions instead of only raw embedding energy;
3. a candidate-level vocabulary-margin replay gate that rolls back any fold
   crossing too many held-out top-token boundaries;
4. score-aware Q/K folding;
5. coupled closed-form V/O fitting with fixed Q/K and exact rollback;
6. an explicitly identity-preserved tied embedding/output-head control;
7. macro-cell whole-output acceptance relative to the current incumbent; and
8. promotion only after the independent free-running behavior gate.

The active compiler executes the 100x target directly.  Smaller ratios may be
run only as diagnostic controls when a rejected 100x component needs an
isolated capacity comparison; they do not replace the severe target.  Rejected
whole-output evidence updates operator fitting, boundary coordinates, and
topology ranking.  Arithmetic feasibility alone never advances or completes a
ratio.
