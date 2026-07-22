# LFM2 nonlinear MLP Q-DEIM compiler prototype

## Scope

`hf_lfm2_mlp_qdeim.py` is a standalone width-compilation experiment for one
dense LFM2 SwiGLU MLP. It does not register a body recipe, mutate a checkpoint,
run a model forward pass, optimize weights, or publish a winner. It materializes
dense BF16 candidates that can be installed elsewhere for decoded whole-response
manual review.

The source MLP convention is:

```text
W1 (gate)  [parent_intermediate, parent_hidden]
W3 (up)    [parent_intermediate, parent_hidden]
W2 (down)  [parent_hidden, parent_intermediate]
phi(x) = silu(x W1.T) * (x W3.T)
```

The normalized-input decoder has shape `[child_hidden, parent_hidden]`. The
residual output chart has shape `[parent_hidden, child_hidden]`. Both are
authenticated, finite, FP64 during compilation, and structurally full rank.
The blank target contributes only its dense BF16 shapes and dtypes; its values
are never read.

## Functional POD and real-neuron Q-DEIM

Let `F` contain authenticated cached parent post-SwiGLU FIT rows. Each FIT case
has a positive sample weight. Optional separately authenticated structural
blocks may append rows already expressed in the parent post-SwiGLU neuron
coordinates. This is the integration boundary for child-lifted analytic
tangents; the Q-DEIM compiler does not generate them.

For positive energy weights `w`, the compiler forms:

```text
Fw = diag(sqrt(w)) F
Fw = L Sigma V.T
U = V[:, :k]                         # parent_intermediate x k
```

All SVD and selection arithmetic is deterministic CPU FP64. POD signs are
canonicalized. A stable lowest-source-index tie rule is used by column-pivoted
modified QR of `U.T`, selecting exactly `k = child_intermediate` real source
neurons. If the authenticated joint snapshot rank is below `k`, compilation
raises `Lfm2MlpQdeimInsufficientRankError` with a sealed typed report. It does
not use random modes, a pseudoinverse fill, heldout rows, or an unreported
fallback.

For selected row indices `p`, define:

```text
S = U[p, :]                           # k x k
R = solve(S.T, U.T)                   # k x parent_intermediate
F ~= F[:, p] R
```

`R` is dense. It therefore represents omitted source features through the
selected real nonlinear features rather than simply deleting their writers.

## Coupled producer and all-source writer transport

Selected producer pairs remain coupled and intact:

```text
W1_child = W1_parent[p, :] decoder.T
W3_child = W3_parent[p, :] decoder.T
```

No selected gate/up row is averaged with another neuron. The structural writer
prior consumes every source W2 column:

```text
W2_child_prior = chart.T W2_parent R.T
```

Thus an omitted neuron with a nonzero reconstruction column can still change
the child writer. The receipt hashes the reconstruction and the full writer
formula inputs and result.

After W1/W3 are materialized to BF16, the compiler recomputes their actual
target post-SwiGLU features on FIT inputs. It can optionally solve one separate
closed-form FP64 SVD/ridge residual correction to W2. This creates two explicit
candidates:

```text
qdeim_structural_prior
qdeim_fit_residual_corrected
```

The correction uses FIT endpoint residuals only. Structural/tangent blocks
never enter the endpoint correction. No numeric diagnostic chooses between the
candidates, accepts one, or publishes one.

## Authenticated tangent-completion contract

`Lfm2MlpQdeimFunctionalSnapshotBlock` accepts:

```text
block_id
snapshot_kind = analytic_child_lifted_parent_swiglu_tangent
functional_rows [row_count, parent_intermediate]
generator_receipt_sha256
block_weight > 0
row_sample_weights[row_count] > 0
```

Both weights are energy weights. The effective POD row is:

```text
sqrt(block_weight * row_sample_weight) * functional_row
```

Consequently, a desired tangent amplitude multiplier `s` must be represented
by weight `s**2`. Keeping per-direction RMS scaling in `functional_rows`, cached
sample weights in `row_sample_weights`, and empirical-versus-tangent energy
balancing in `block_weight` makes the conversion lossless and separately
auditable.

The separately implemented `hf_lfm2_swiglu_tangent_bank.py` generator evaluates
the following derivative at authenticated anchors for parent neuron `j`:

```text
g_j = W1_j x
u_j = W3_j x
D phi_j(x)[dx]
  = silu'(g_j) * (W1_j dx) * u_j
  + silu(g_j) * (W3_j dx)
```

Child-coordinate directions are lifted through the authenticated normalized
input decoder before this derivative is evaluated. Its lossless kwargs adapter
emits the exact block orientation and energy-weight convention required above.
The generator and compiler remain deliberately decoupled and are not yet wired
into a body recipe. Generation, anchor choice, and block-balance recommendations
remain separately sealed responsibilities.

The current real-cache audit explains why the typed completion path matters:
target intermediate width is 797, while FIT-only stable ranks are below 200 and
the union of all cached trace rows remains below 797 on audited layers. Raw
analytic tangent augmentation can exceed 797, but tangent magnitude can
dominate the POD. Explicit authenticated row and block energy weights are
therefore part of the API, not an implicit numeric acceptance rule.

## Difference from the existing medoid and Qwen paths

The existing normalized LFM2 real-medoid compiler clusters a weight-complete
gate/up/down structural signature, chooses local medoid supports, and uses a
sparse multi-anchor reconstruction. This prototype instead selects coordinates
from a cached nonlinear functional POD, preserves one intact coupled producer
per selected coordinate, and bakes a dense all-source interpolation directly
into W2.

The Qwen path-decomposition work selects donor paths from response signatures
and supports depth sharing, quotas, interpolation fitting, and diagnostic
sketches. It is not this strict POD/Q-DEIM lane. This prototype has no seeded
sketch selection, validation split, quota, auto-threshold, or cross-depth
codebook. Its only hard numeric failures are malformed/authentication failures
and structural inability to form the requested full-rank interpolation.

These methods remain complementary: weight-complete medoid signatures can
provide structural diagnostics, and response-aware Qwen machinery can inspire
later decoded evaluation, but neither is silently substituted for Q-DEIM in
this compiler.
