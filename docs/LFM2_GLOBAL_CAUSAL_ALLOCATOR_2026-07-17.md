# LFM2 Global Causal Large-Target Allocator

Status: implemented and locally verified; no GPU run launched.

## What this lane is

This is the first large-target allocator that makes a global decision from
downstream behavioral damage rather than selecting widths from local activation
importance alone.

Each candidate is a real physical proposal. For the currently authorized FFN
slice, that means:

- one exact layer and target width;
- the real parent medoid producer rows;
- the real source-to-target neuron assignment;
- the real down-consumer coefficients and weights;
- a content hash over that complete layer payload; and
- an install path through the existing variable-width transaction.

The causal measurement bound to that proposal contains:

- final-logit margin damage;
- EOS/termination damage;
- repetition damage;
- task-anchor damage;
- final residual-path damage; and
- attention-context-path damage.

Exact token equality is not an optimization objective.

## Explicitly excluded closed lanes

The allocator does not fall back to:

- uniform width contraction;
- local activation-importance allocation;
- independent per-layer W2 factorization;
- shared W2 bases;
- whole-layer parameter sharing;
- exact-token optimization; or
- unbound head/conv parameter estimates.

Those options cannot enter the exact physical budget.

## Parent and child execution contract

Parent evidence is referenced through the existing immutable circuit-cache key,
corpus hash, parent-output envelope hash, and task-anchor manifest hash. Parent
response generation is zero during candidate measurement and child promotion.

Optional JVP evidence is allowed only for residual activation states. Parameter
gradients, autograd training, optimizer construction, optimizer steps, and
backward calls are forbidden.

Every child proposal uses `full_prefix_canonical` traces. Promotion is staged
and cumulative. Each stage:

1. creates a real child through a caller-supplied local factory;
2. installs the real physical variable-width plan;
3. evaluates trajectory, parent-relative intelligence, and coherent-response
   gates without exposing the live parent to the evaluator;
4. observes and rejects any live-parent forward;
5. commits only the accepted final stage; and
6. otherwise restores the blank child exactly.

The executor uses the repository fold-only API guard and verifies state/runtime
identity after rollback.

## Exact large-target accounting

For the 229,693,184-parameter LFM2.5-230M parent and 1,024 hidden width, one
physical FFN neuron saves exactly 3,072 parameters.

| Request | Removed FFN neurons | Exact physical saving | Target parameters | Realized fraction |
|---|---:|---:|---:|---:|
| 35% | 26,169 | 80,391,168 | 149,302,016 | 34.9993703% |
| 50% | 37,385 | 114,846,720 | 114,846,464 | 50.0000557% |

The maximum FFN-only contraction that preserves one neuron in each of fourteen
layers saves 110,057,472 parameters, or 47.9149925% of the parent.

Therefore:

- 35% is physically reachable in principle through the current FFN installer,
  once the sealed causal width frontier contains an exact allocation.
- 50% is not physically reachable through FFN width alone.
- Attention-head and short-convolution candidates can be scored and retained,
  but they are diagnostic-only until a real topology-correct installer exists.
- A 50% plan fails closed today instead of counting imaginary head/conv
  savings.

## Files

- `src/neural_forge/hf_lfm2_global_causal_allocator.py`
  - physical proposal identity;
  - causal evidence schema;
  - exact global dynamic program;
  - monotonic cumulative stage planner;
  - real child install/evaluate/commit/rollback executor.
- `src/neural_forge/hf_lfm2_variable_width_fold.py`
  - fixed-width evidence/plan bundle;
  - layer-payload binding fingerprint;
  - fixed-bundle verification.
- `src/neural_forge/hf_lfm2_cached_causal_evidence.py`
  - row-disjoint selection/fit/validation seal;
  - broad exact-35 width frontier;
  - uncached batched full-prefix child rollouts;
  - parent-action margin, EOS, repetition, task, residual, and downstream
    attention damage;
  - resumable candidate records;
  - portable frontier sealing and tensor rehydration from the parent cache.
- `experiments/build_lfm2_cached_causal_evidence.py`
  - opens the existing 16-case parent cache directly;
  - prepares, installs, measures, and exactly rolls back every physical
    candidate;
  - rejects any live-parent forward;
  - prunes temporary child caches while retaining sealed resumable records;
  - emits an exact 35% frontier document and a bound 50% infeasibility result.
- `experiments/run_lfm2_global_causal_allocator.py`
  - rehydrates and authenticates the producer document before prepared
    35%/50% frontier execution.
- `tests/test_hf_lfm2_global_causal_allocator.py`
  - exact real-model budget math;
  - real physical shape change and exact rollback;
  - global staged allocation;
  - unsupported head/conv exclusion;
  - parent-forward rejection;
  - proposal-binding tamper rejection;
  - split-leakage and frontier-tamper rejection;
  - exact prompt, `use_cache=False` child rollout;
  - portable frontier replay with exact 35% and infeasible 50%;
  - prepared frontier execution contract.

## Verification

Focused result:

```text
71 passed
```

That set covers the new allocator, the reused variable-width installer, the
LFM2 circuit cache, and the general dynamic transaction guard.

No checkpoint and no experimental run artifact were published.

## Real 35% launch boundary

The evidence producer is implemented and launch-ready. Its runtime contract is:

1. reuse the sealed 16-case parent trace/output caches without rebuilding;
2. select medoids only from authenticated selection rows;
3. measure the real widths `1792,1280,896,691,690,512` in every layer;
4. use validation-only parent-action margin, EOS, repetition, task-anchor,
   residual, and downstream-attention damage;
5. run all free child behavior with exact cached prompt IDs and
   `use_cache=False`;
6. reserve the independent 64-case holdout for final cumulative promotion;
7. seal every candidate and emit a document the allocator runner rebuilds from
   the immutable parent cache; and
8. prove an exact 35% plan while retaining 50% as an authenticated
   FFN-only infeasibility.

The producer has not been launched. The first run should produce evidence only;
it does not train, publish a checkpoint, or execute the independent promotion
holdout.
