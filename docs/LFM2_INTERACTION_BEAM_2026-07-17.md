# LFM2 Top-K Cumulative Interaction Beam

Status: implemented in a new post-frontier module; remote regression verified.
It is intentionally not imported by the in-flight cached-causal producer.

## Why this exists

The original global causal allocator is an authenticated additive control. It
measures every physical layer/width proposal in isolation and selects the
minimum summed damage at one exact parameter budget. That control is still
valuable, but isolated damage is not guaranteed to be additive after many
layers are folded together.

`src/neural_forge/hf_lfm2_interaction_beam_allocator.py` retains multiple exact
allocations and measures each complete cumulative child. It therefore tests the
residual/attention interactions that the additive control can only estimate.

## Structural search contract

- Preserve the single minimum-additive allocation as rank zero.
- Retain deterministic top-K alternatives at the identical physical saving.
- Keep alternatives layer-diverse while preserving exact parameter accounting.
- Bind every selected option to its measured proposal SHA-256.
- Ignore diagnostic attention/conv savings until a real physical installer and
  payload exist.
- Fail closed above the FFN-only ceiling instead of claiming imaginary
  compression.

The enumerator accepts an explicit target fraction, saved-parameter count, or
target parameter count. The current FFN frontier can prove 35% but cannot reach
50% or larger by itself. Exact 50/55/60/65% work must add authenticated
attention, topology, depth, vocabulary, and/or codec payloads.

## Canonical multi-pack materialization

Cross-device clustering is not assumed deterministic. Each isolated proposal is
loaded from its canonical changed-layer pack. The multi-pack materializer:

1. authenticates the proposal, medoids, assignment, consumer coefficients, and
   changed tensors for every selected layer;
2. combines those exact changed-layer payloads into one complete fourteen-layer
   bundle;
3. clones only unchanged layers from the authenticated parent;
4. performs zero clustering, medoid selection, coefficient fitting, or consumer
   aggregation; and
5. seals the combined plan, per-layer bindings, and bundle.

A regression test hard-disables the clustering function and proves the combined
plan, bundle, and every layer binding are identical to the corresponding full
physical plan.

The sharded merge now enters through
`experiments/finalize_lfm2_cached_causal_merge_canonical.py`. The byte-frozen
producer still parses and authenticates shard records, but the wrapper replaces
only its final callback. Evidence sealing and every cumulative stage consume
the exact canonical packs; any clustering call, parent forward, or training
operation fails closed. A separate sidecar binds the run contract, frontier,
evidence, plan, pack manifests/tensors, implementation hashes, and zero-
operation audit.

Post-merge authentication uses
`src/neural_forge/hf_lfm2_canonical_frontier_replay.py`. It rebuilds only the
cache-derived source evidence and rehydrates every proposal and stage from the
same packs. This closes the older replay path that would otherwise recluster a
correct frontier after merge.

## Cumulative evaluation contract

Every top-K alternative is installed into a real blank child and evaluated only
from sealed parent caches. Search children are never committed.

Ranking uses:

- parent-demonstrated intelligence retention;
- coherent whole-response quality;
- full-prefix trajectory retention;
- residual-path retention;
- attention-context-path retention;
- termination quality; and
- repetition quality.

Exact token agreement is diagnostic only and has zero ranking weight.

The default acceptance boundary follows the project objective:

- intelligence retention at least 0.90; and
- coherent whole-response score at least 0.90.

Residual and attention metrics rank alternatives even when several pass the
same behavioral boundary.

## Safety invariants

- zero live-parent forwards;
- zero parent cache rebuilds or response regeneration;
- zero optimizer construction or steps;
- zero backward or autograd-gradient calls;
- exact structural target installation;
- byte-exact state and runtime rollback for every search child; and
- no checkpoint publication during search.

Tests actively attempt both a live-parent forward and optimizer construction.
Both fail closed and restore the child.

## Verification

Remote focused interaction suite:

```text
12 passed
```

Expanded variable-width, canonical sharding, global causal, and interaction
suite:

```text
43 passed
```

Verified implementation SHA-256 values:

```text
interaction/canonical allocator:
f543c9180ff3da8d6e5c378cb0e203df4a1ba4570469c13538d7196c1afa7364

canonical merge wrapper:
87222c699b77f601e53e7085c7a792635846787b06a1df5147a82ddbcf440e3c

canonical replay authenticator:
920bdca68bef8153d6f4dec7c042a2f0c854268b81f8644853b76d48fa63e2af

future merge launcher:
1dc27f2fd0744ef87e8781c2c85a3b9dba91b6391f0545eed8213bceeb3501e2
```

## Larger mixed-budget continuation

After the sealed causal frontier completes, its site and coupling evidence
should guide a zero-backprop mixed stack:

- OSAQ-style Hessian-null additive outlier suppression;
- dynamic row-aligned grouping scored by error per byte;
- AAAC dual codebooks;
- TurboBoA-style joint attention-channel compensation and preceding-layer
  correction;
- CoreQ-style adaptive mismatch and successive rounding; and
- sparse SERQ-style low-rank residual writers.

For LFM2.5-230M, hidden width is 1,024 and FFN width is 2,560. Group size must
remain a measured variable. The row-aligned g256 control must not be mislabeled
as cross-row contamination; exact-outlier peeling and 4/5-bit clipping are
retained because equal-coverage measurements support them.
