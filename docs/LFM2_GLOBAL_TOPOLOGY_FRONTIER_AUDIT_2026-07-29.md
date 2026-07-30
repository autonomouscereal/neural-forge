# LFM2 global topology frontier audit

Date: 2026-07-29

## Why this is not another H512/H640 retry

The historical full-depth structural lane declared four uniform target
architectures:

| Declared target | Dense parameters | Historical status |
|---|---:|---|
| H640 / Q10 / KV5 / D64 / FF1920 | 114,067,328 | Actually folded, saved, reloaded, and whole-output reviewed. |
| H640 / Q10 / KV5 / D64 / FF1088 | 91,703,168 | Declared feasibility candidate; no authenticated whole-child result was found. |
| H512 / Q8 / KV4 / D64 / FF1568 | 80,407,808 | Declared feasibility candidate; no authenticated whole-child result was found. |
| H512 / Q8 / KV4 / D64 / FF1024 | 68,709,632 | Declared feasibility candidate; no authenticated whole-child result was found. |

The executed H640/FF1920 child was not coherent. The later circuit-balanced
basis and cached vocabulary-margin geometry improved parent-nonempty response
retention, but task retention remained zero and generation remained repetitive
and saturated. That result rejects blind repetition of the fixed uniform H640
lane. It does not establish H512 as a behavioral result, because neither H512
candidate was actually published and reviewed.

The new path therefore contains no H512, H640, H784, or other hand-curated
architecture shortlist. It enumerates the registered LFM2 catalog's legal
hidden-width multiples from the inspected source width, proves an exact
full-depth heterogeneous MLP/operator lattice separately for every width, and
selects one complete child at the same requested parameter count.

## Minimal production seam

Only the upstream cached heterogeneous selection-input builder changes when
the new mode is explicitly enabled. It returns the same existing objects:

- one authenticated `Lfm2HeterogeneousLayoutSelectionInput`; and
- its sealed cached capacity option bank.

The selected object then follows the unchanged registered dense topology
catalog, exact architecture-discovered solver, compiler-owned blank target,
physical ragged LFM2 materializer, complete fold executor, artifact publisher,
and fresh-reload validation path.

The fixed-H mode and its defaults remain unchanged. The live a0007 run is not
modified.

The production runner exposes this seam as
`--dense-heterogeneous-global-hidden-frontier`. Its default is off. When it is
on, `dense_minimum_hidden_size` is the lower bound of a dynamic multiple-of-8
enumeration, not a manually selected test architecture. The runner records the
complete frontier, rejected exact-lattice widths, selected candidate, cached
functional spectrum, option-bank seal, exact descriptor, and selection receipt
in planned-target evidence.

## Cross-width evidence that is not parameter arithmetic

Exact parameter count is an eligibility constraint, not the width-quality
score. Every candidate has the same requested count.

Residual widths are compared with one authenticated spectrum built from the
sealed parent cache. It includes four causal roles: residual operator context,
operator residual writes, nonlinear SwiGLU residual writes, and the terminal
whole-stack residual state. Rows are L2-normalized, each role Gram is
normalized to unit trace, and roles receive equal weight. The retained
best-rank-H fraction of that role-balanced functional spectrum supplies the
hidden-width loss.

That loss is combined with the selected exact layout's mean cached nonlinear
MLP/operator defect. The latter comes from the broad per-layer option bank and
not from the old binary 566/567 filler. Thus a wider residual stream must pay
for the nonlinear/attention/ShortConv capacity it forces out of the same exact
budget, while a narrower stream pays its measured cached functional-direction
loss. Parameter count itself cannot distinguish candidates because it is
identical across the frontier.

All of this is fold selection. It performs no model training, backward pass,
optimizer step, pruning, quantization, sparse storage, parameter sharing, or
parent-sidecar construction.

## Implemented orchestration and checks

The global allocator now:

1. discovers source layer types and SwiGLU widths from the inspected graph;
2. enumerates every legal hidden width dynamically;
3. derives an automatic exact-budget MLP/operator lattice for each width;
4. unions those width sets and performs one authenticated nonlinear-cache scan;
5. builds one bounded role-balanced functional spectrum for all widths;
6. filters and reseals a width-specific option bank without rescanning;
7. invokes the existing whole-model exact selector for each feasible width; and
8. returns the minimum combined cached-functional/component-risk candidate to
   the unchanged production catalog.

Focused executable coverage proves that three distinct residual widths can
produce complete four-layer children at the identical exact parameter count,
that the nonlinear cache builder is called once for their union, that no parent
or child forward occurs during selection, and that the selected descriptor is
standalone and exact. Separate selection tests prove both a narrow and a wide
candidate can win when their cached functional/component evidence changes;
there is no hidden-width preference in the tie-break policy.

The production-shaped exact-4x fixture (22 layers, source H1024/I2560,
target vocabulary 4096, exact 57,420,144 parameters) enumerates 125 legal
multiple-of-8 residual widths from H32 through H1024. Pure integer-lattice
proof rejects 113 before layout solving; 12 remain feasible. The broad H784
bank contains 780 authenticated layer options and solved the exact descriptor
in 18.0 seconds in the local verification run, below its 20,000-state ceiling.
This check constructs no model and launches no fold.
