# Behavior-first residual-attention functional scorer

`src/neural_forge/hf_residual_attention_functional.py` scores already-collected
parent and candidate cached traces using output function only. It deliberately
does not read or rank Q/K/V, weights, attention distributions, activations,
residuals, or cache similarity.

## API

```python
from neural_forge.hf_residual_attention_functional import (
    build_residual_attention_functional_comparison,
    rank_functional_reports,
    score_residual_attention_function,
)

calibration = score_residual_attention_function(
    parent_calibration_traces,
    candidate_calibration_traces,
    candidate_id="redirect-L19G4-to-L21G6",
    split="calibration-selection",
    top_k=5,
)

heldout = score_residual_attention_function(
    parent_heldout_traces,
    candidate_heldout_traces,
    candidate_id="redirect-L19G4-to-L21G6",
    split="heldout-evaluation",
    top_k=5,
)

comparison = build_residual_attention_functional_comparison(
    parent_calibration_traces,
    parent_heldout_traces,
    candidate_calibration_traces_by_id,
    candidate_heldout_traces_by_id,
    top_k=5,
)
```

`calibration-selection` reports may select or rank candidates, but are never
promotion evidence. `heldout-evaluation` reports must be excluded from fitting
and candidate selection. `rank_functional_reports(..., purpose="selection")`
accepts only calibration reports; `purpose="evaluation"` accepts only held-out
reports. Mixed splits fail closed.

Selection also fails closed unless every forced calibration driver is the
actual parent top-1 action (`parent_driver_top1_rate == 1.0`). A held-out
report records this condition but can never select. The comparison bundle
names exactly one calibration winner and emits a separate held-out ordering
with `winner_named=false`.

The legacy base O probe has only one prompt pool, and that pool also feeds its
O fit. It therefore uses `calibration-exploratory`, which is non-rankable,
non-selectable, and non-promotable. No held-out split is inferred or fabricated.

## Primary parent anchors

Forced parent and candidate traces must use identical parent-driven prefixes
and driver actions. At every forced action the scorer records:

- candidate top-1 agreement with the parent top-1 action;
- candidate log probability of that parent action;
- parent-action top-k coverage and deterministic 1-based rank;
- whether the supplied parent driver itself was parent top-1.

These parent-anchor fields come first in the deterministic ranking key. This
retains the lesson from the July 9 child-action-only failure: the child's own
trajectory cannot be its only teacher.

## Independent greedy gate

Parent and candidate greedy traces are validated independently. Their consumed
prefixes may diverge; the scorer pairs them by sequence ID and prompt and does
not compare internal tensors after divergence. It reports:

- positional exact-token retention;
- longest common prefix and zero-based first divergence step;
- exact full-rollout retention;
- unique-token repetition fraction;
- adjacent-token repetition fraction;
- longest identical-token run;
- excess candidate repetition and run length relative to the parent.

Thus the candidate greedy trajectory remains a closed-loop survival and
degeneration gate without becoming the sole teacher.

## Validation and ranking boundary

The scorer requires both forced and greedy lanes, finite floating-point logits,
one constant and identical parent/candidate output vocabulary, in-range token
IDs, contiguous step indices, true argmax actions in greedy traces, and exact
forced-prefix alignment. Multiple layer/group traces of the same functional
trajectory are deduplicated so one prompt is not overweighted; conflicting
duplicates are rejected.

The ascending ranking key uses only:

1. forced parent top-1 agreement;
2. forced parent-action mean log probability;
3. forced top-k coverage, mean rank, and worst rank;
4. greedy positional retention, longest-common-prefix retention, and exact
   rollout retention;
5. candidate repetition and longest-run excess;
6. candidate ID as the deterministic final tie breaker.

The module performs arithmetic over detached trace outputs only. It constructs
no optimizer, takes no gradient, mutates no model, and writes no checkpoint.
It is integrated at the report boundary of the base O, raw-V, residual-value
basis, and Q/K-anchor runners. Existing fits and internal attention,
residual, cache, and activation diagnostics remain unchanged. Raw-V and basis
four-control winners, plus Q/K winners per control, come only from calibration
functional outputs; held-out and internal diagnostics cannot alter them.
