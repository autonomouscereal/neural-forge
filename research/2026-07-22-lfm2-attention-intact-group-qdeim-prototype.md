# LFM2 intact-GQA-group Q-DEIM prototype

## Result

`hf_lfm2_attention_group_qdeim.py` is a standalone, no-training attention
folding prototype. It implements the missing attention-side nonlinear
empirical-interpolation experiment without changing the active semantic fold
adapters or recipe catalog.

It is structural evidence, not a behavioral success claim. Numeric
reconstruction diagnostics are advisory. Decoded whole-response intelligence
is the primary semantic evidence, but neither a metric threshold nor a binary
manual gate decides whether experimentation continues. Retention targets guide
iteration rather than acting as brittle cutoffs.

## Construction

For parent post-softmax context
`C[parent_row, parent_q_head, head_coordinate]`, the compiler preserves the
parent GQA grouping. With `r = q_heads / kv_heads`, it reshapes cached context
into a functional matrix with one column per complete GQA group and
`rows * r * head_dim` observations. Explicit positive row weights are applied
before an uncentered FP64 SVD.

The first `target_kv_heads` right singular modes form the group basis `U`.
Deterministic block-constrained pivoted QR selects exactly that many complete
parent groups. If `U_P` is the square basis at selected groups, the
interpolation is

```text
A = solve(U_P.T, U.T)                         # [target_groups, source_groups]
R = kron(A, I_(r * head_dim))                # selected context -> all context
```

`A[:, selected_groups]` is the selected-group identity. Each selected group
carries all `r` of its Q heads and its one K/V head. The target producer
weights are direct real-parent rows mapped through the normalized-input
decoder `D[target_hidden, source_hidden]`:

```text
Q_child = Q_parent[selected_q_rows, :] @ D.T
K_child = K_parent[selected_kv_rows, :] @ D.T
V_child = V_parent[selected_kv_rows, :] @ D.T
```

There is no pre-softmax Q/K/V average. With residual chart
`B[source_hidden, target_hidden]`, every parent O column contributes to the
dense target writer:

```text
O_child = B.T @ O_parent @ R.T
```

Thus `C_selected @ O_child.T` reconstructs the charted parent attention
residual whenever the group-context interpolation is exact.

## Authentication and failure behavior

The compiler authenticates the coupled source Q/K/V/O state, blank-target
BF16 shape/dtype template, decoder, residual chart, cached context plus row
weights, and any separately receipted structural/tangent snapshot blocks. It
rehashes source state after compilation and returns ordinary standalone dense
BF16 tensors.

If authenticated functional context rank is below `target_kv_heads`, it raises
`Lfm2AttentionGroupQdeimInsufficientRankError` with a hashed deficit receipt.
It does not use random completion, held-out rows, silent pseudoinverses,
training, gradients, pruning, quantization, sparsity, or parameter sharing.

## Focused proof

`tests/test_hf_lfm2_attention_group_qdeim.py` proves:

- exact recovery for a known block-low-rank post-softmax context;
- selected-group self-identity and the O reconstruction orientation;
- Q/K/V are intact selected parent rows through the decoder, not averages;
- deterministic dense standalone BF16 output and source immutability;
- failed authentication and authenticated insufficient-rank behavior;
- separate structural/tangent snapshot authentication; and
- dynamic derivation of the intended LFM2 `16 Q / 8 KV / 64` to
  `10 Q / 5 KV / 64` topology.

The focused suite passes 6/6. The next meaningful step is to compose this
attention operator with the nonlinear MLP fold and materialize the combined
candidate against the sealed LFM2 evidence. Its complete decoded behavior—not
its reconstruction score—will steer the next iteration.
