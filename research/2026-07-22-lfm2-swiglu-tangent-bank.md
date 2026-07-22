# LFM2 analytic SwiGLU tangent bank

## Purpose

`hf_lfm2_swiglu_tangent_bank.py` supplies the missing child-lifted nonlinear
directions for the standalone MLP Q-DEIM compiler. It is a folding primitive,
not a learner. It consumes authenticated parent W1/W3 tensors, authenticated
cached gate/up/post-SwiGLU rows, and an authenticated normalized-input decoder
with shape `[child_width, parent_hidden]`.

For every selected cached anchor and every decoder row `d`, it computes the
analytic JVP in CPU FP64:

```text
g = x W1^T
u = x W3^T
phi(x) = silu(g) * u
dphi = silu'(g) * (d W1^T) * u + silu(g) * (d W3^T)
```

The output orientation is
`[anchor_count * child_width, parent_intermediate]`. No parent-coordinate
identity is synthesized. The authenticated decoder rows are the only allowed
production directions.

## Structural policy and authentication

- One to four anchors are chosen at fixed evenly spaced positions in canonical
  cache-row order. Activation values, ranks, spectra, and sample weights do not
  affect this choice.
- W1, W3, gate, up, post, decoder, optional positive sample weights, and
  optional positive child-coordinate RMS values require dtype/shape/raw-byte
  SHA-256 authentication.
- Inputs cannot be parameters or require gradients. All source hashes are
  checked again after construction.
- The receipt binds raw tangent rows, repeated source sample weights,
  coordinate RMS row scales, coordinate-scaled Q-DEIM rows, the global balance
  recommendation, exact anchor/direction provenance, and a stable bank digest.
- The operation audit records zero model/parent forwards, generation,
  backward, autograd-gradient, optimizer, and training calls.

## Amplitude balancing without a metric gate

The audit of all 14 LFM2 layers found that tangent rows lift algebraic rank but
can be severely amplitude-imbalanced. The bank therefore preserves three
separate quantities:

1. raw analytic JVP rows;
2. authenticated child-coordinate RMS amplitude scales; and
3. repeated cached source-row sample weights.

It also recommends a single deterministic global amplitude scale

```text
s = ||sqrt(w_fit) * parent_post||_F
    / ||sqrt(w_tangent) * coordinate_RMS * raw_tangent||_F
```

when both exact block norms are nonzero. This merely equalizes the total
weighted Frobenius norm of the empirical post-SwiGLU and tangent snapshot
blocks. It does not select anchors, accept a candidate, reject a candidate, or
claim semantic quality.

`qdeim_functional_snapshot_kwargs()` converts the sealed bank losslessly to
the Q-DEIM snapshot-block contract without importing or coupling to that
module. It passes coordinate-RMS-scaled rows, unchanged repeated cache sample
weights, and `block_weight = s^2`. The square is required because Q-DEIM
multiplies each row by `sqrt(block_weight * row_sample_weight)` before its FP64
SVD. The Q-DEIM evidence digest independently re-authenticates the scaled rows
and both weight levels.

Per-anchor norms, per-decoder-direction norms, and a bounded deterministic
structural-submatrix spectrum/rank are emitted only as advisory diagnostics.
They have no semantic authority and cannot auto-veto a structurally valid
bank.

## Local verification receipt

Focused tests cover:

- analytic JVP agreement with a centered finite-difference diagnostic;
- decoder orientation and the absence of an identity fallback;
- strict required/optional tensor authentication;
- deterministic hashes and source immutability;
- exact Q-DEIM block conversion and weighted-energy equalization;
- an example where child-lifted tangents raise snapshot rank; and
- rejection of gradient-bearing sources and nonpositive weights.

This component is standalone and is not yet wired into an active semantic body
recipe or remote run. Whole-response manual intelligence review remains the
authority for any future materialized child.
