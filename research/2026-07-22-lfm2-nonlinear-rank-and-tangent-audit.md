# LFM2 nonlinear rank and analytic-tangent audit

Date: 2026-07-22

Status: completed read-only construction diagnostic; no child was accepted or
rejected from these numbers.

## Question

The 4x dense blank child has an MLP intermediate width of `797`. The existing
semantic fold uses cached parent activations to choose or combine the parent's
`2560` SwiGLU neurons. This audit asks whether those cached activation rows can
span all `797` child nonlinear features, and whether exact analytic SwiGLU
tangents can supply the missing structural directions without another parent
capture or any training.

Inputs were the sealed local cache
`dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b`
and the local `LFM2.5-230M` safetensors. No Hugging Face download, model
generation, forward recapture, gradient, optimizer, pruning, quantization, or
weight update was performed.

## Result

Across all 14 MLP layers:

| Evidence bank | Stable feature-rank range | Can directly span 797? |
|---|---:|---|
| FIT activation rows, 224 rows | 97-182 | no |
| all cached activation rows, 648 rows | 309-544 | no |
| FIT plus one full-parent-coordinate analytic tangent bank | 1121-1206 | yes, in all 14 layers |

The rank diagnostic used singular values above `1e-6` of the largest singular
value. That tolerance is a numerical description, not an intelligence gate.

This establishes a concrete limitation of the trace-only construction: even
using every cached row, the observed post-SwiGLU feature bank is narrower than
the blank child's 797-neuron nonlinear interface. A compiler that obtains its
entire nonlinear basis from those traces cannot identify a full-width child
nonlinear chart.

For `phi(x) = silu(x W1^T) * (x W3^T)`, the exact directional derivative used
in the feasibility audit was:

```text
dphi = silu'(gate) * (dx W1^T) * up
       + silu(gate) * (dx W3^T)
```

These Jacobian-vector rows are derived directly from authenticated parent
weights and cached gate/up states. They require no additional behavioral
trace and no iterative optimization.

## Production implication

The full-parent-coordinate tangent bank is a feasibility ceiling, not the
production fold. The production compiler must use only directions reachable
from the child: rows of the authenticated child-to-parent normalized-input
decoder. If one anchor is insufficient, it can add a small deterministic set
of cached anchors while retaining the same `797`-wide dense target.

The intended nonlinear fold is therefore:

1. combine cached post-SwiGLU rows with authenticated child-lifted analytic
   tangent rows;
2. select intact parent gate/up neuron pairs with deterministic Q-DEIM;
3. never average the selected nonlinear producers;
4. reconstruct a dense child W2 from every parent W2 column through the
   empirical-interpolation operator and residual chart; and
5. materialize the resulting standalone child for decoded whole-response
   review.

Raw tangent magnitudes also exposed an important scaling issue: at layer 5,
unbalanced tangents concentrated 99% of snapshot energy into only five modes
(one after naive writer-norm column weighting), despite a stable rank of 1206.
Production therefore needs authenticated structural row balancing, such as
child-coordinate RMS direction scales and equalized empirical/tangent block
energy. This is a construction choice to expose as candidate provenance, not a
semantic threshold.

## Decision authority

No rank, energy, reconstruction, or fit number in this audit accepts, rejects,
or promotes a child. The numbers identify why the previous trace-only fold was
underdetermined and how to construct a broader fold. Intelligence retention is
judged from complete decoded behavior after materialization.
