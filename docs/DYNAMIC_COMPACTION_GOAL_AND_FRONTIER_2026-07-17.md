# Dynamic Compaction Goal and Frontier

Updated: 2026-07-23

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

## Current compaction execution objective

Marginal parameter removal and synthetic size controls do not satisfy the
project objective. The active real-model proof sequence is:

1. first establish a coherent, standalone exact-ratio control by folding the
   local 229,693,184-parameter LFM2.5 parent into a separately instantiated
   57,405,368-parameter dense blank child (4.001249x);
2. save and reload that child with no parent, sidecar, sparse state, or runtime
   reconstruction dependency;
3. inspect complete free-running responses and parent-relative task behavior
   on disjoint evidence, using numeric measurements as diagnostics rather than
   automatic accept/reject authority;
4. once the exact 4x child is coherent, increase width/depth compaction through
   10x, 100x, 250x, and larger targets while diagnosing where behavior is lost;
5. generalize the mechanisms proven by those real folds through registered
   architecture, tokenizer, and modality adapters rather than caller-written
   layer surgery.

The 100x target preserves all 14 ordered LFM2 operators with hidden width 128,
FFN width 175, two query heads, one key/value head, and head dimension 64.  The
250x target preserves all 14 ordered operators with hidden width 64, FFN width
166, two query heads, one key/value head, and head dimension 32.  Both use the
sealed 4,096-row child tokenizer bridge.  Their sizes count dense learned
parameters, not bytes.

The 4x proof is an intermediate behavioral control, not completion of the
arbitrary-ratio, cross-architecture, cross-modality objective.

## Non-negotiable structural validity

A promoted result must satisfy all of the following:

1. The target begins as a compiler-authenticated dense blank.
2. The learned parameter count is the requested genuinely smaller count.
3. Parameter reduction is not credited to dtype, packing, sparsity,
   quantization, pruning, external adapters, or shared parent storage.
4. Folding performs zero optimizer steps, backward calls, or live parent
   forwards when sealed parent evidence is available.
5. The saved child is standalone and survives a fresh local reload.
6. Exact token/text equality, local RMSE, first-token agreement, and parameter
   arithmetic are diagnostics only.

## Behavioral objective and review policy

Roughly 90% retention of useful parent-demonstrated intelligence remains the
quality objective, not a hard numerical cutoff. No scalar score, threshold, or
single binary manual verdict has authority to stop or promote a folding lane.
Complete decoded responses, coherence across turns, reasoning, factual and task
behavior, breadth, failure modes, and parent-relative regressions are reviewed
together. Metrics prioritize diagnosis and candidate selection; they do not
replace judgment about whether the standalone child is intelligent.

A child with valid structure but incoherent behavior is a structural proof,
not a successful intelligent compaction. A coherent child is described with
its observed strengths, regressions, and compression ratio rather than reduced
to a pass/fail score.

The executable policy and active-path boundary are recorded in
[`BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`](BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md).

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

It is not coherent, so it is behavioral failure evidence rather than a
successful intelligent compaction. The checkpoint and complete outputs remain
retained for diagnosis; this observation does not close the folding lane:

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
   represents token decision directions instead of only raw embedding energy;
3. candidate-level vocabulary-margin replay diagnostics that identify where a
   fold crosses held-out token boundaries without automatically rolling back or
   deleting the candidate;
4. score-aware Q/K folding;
5. coupled closed-form V/O fitting with fixed Q/K and exact rollback;
6. an explicitly identity-preserved tied embedding/output-head control;
7. complete macro-cell whole-output evidence relative to the current incumbent;
   and
8. retained standalone candidates reviewed across coherence, competence,
   reasoning, task behavior, instruction following, safety, breadth, and
   concrete failure modes, without a scalar or binary promotion gate.

The active compiler first executes the exact 4x real-model control, then pushes
larger ratios. Smaller or larger variants remain useful diagnostic controls;
none is deleted or stopped merely because one score, threshold, or manual label
looks poor. Observed whole-output regressions update operator fitting, boundary
coordinates, and topology ranking. Arithmetic feasibility alone never
establishes intelligence retention.
