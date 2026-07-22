# Semantic intact-group Q-DEIM cache adapter

## Cache evidence finding

The sealed LFM2 circuit cache already contains the required true nonlinear
attention producer evidence. For every captured full-attention phase it stores:

- `attention.context`: the actual pre-O tensor captured after the runtime
  softmax/value aggregation, shaped
  `[batch, source_q_heads, sampled_query_rows, head_dim]`;
- `attention.context_from_probabilities`: an enhanced-cache reconstruction
  authenticated against the actual context on the execution device; and
- `attention.probabilities` plus `attention.value_history`, from which the same
  structural context can be computed as
  `probabilities @ repeat_interleave(value_history, q_per_kv, head_axis)`.

The first tensor is the true functional FIT source. The other two routes are
valid cache-derived structural reconstructions, but the adapter does not
silently pretend they are captured context.

## Adapter behavior

`hf_lfm2_semantic_attention_group_qdeim_adapter.py` reads only explicitly
declared FIT batches through the selective authenticated cache session. It
discovers every phase from the sealed attention-Q inventory and requires an
`attention.context` tensor for every one. Held-out payloads are neither loaded
nor fitted.

Each supplied batch weight is divided equally across its causal phases, then
each phase budget is divided equally across that phase's sampled rows. This
prevents a many-row prefill from overwhelming one-row decode phases. The
receipt records and authenticates every per-phase weight budget and realized
row-weight sum.

The adapter accepts the residual chart only with the existing sealed
`neural-forge-lfm2-layer-stack-residual-chart-v1` receipt. It accepts the
normalized-input decoder only with the existing sealed
`neural-forge-lfm2-normalized-operator-input-decoder-v1` receipt bound to that
exact chart. It does not recompute either map.

The authenticated true context bank is passed to the standalone intact-group
Q-DEIM compiler. The output is a detached dense BF16 Q/K/V/O state plus a
receipt binding:

- immutable cache/session and selected-payload lineage;
- direct local source Q/K/V/O inventory;
- existing chart and normalized-decoder receipts;
- batch/phase-balanced true-context evidence;
- standalone group selection/interpolation result; and
- hashes of every materialized candidate tensor.

If true context is missing, the adapter returns a typed no-candidate evidence
deficit. When possible it materializes the nearest cache-derived structural
context and labels its exact derivation, but never feeds that proxy into the
compiler automatically. Insufficient true-context functional rank is also a
typed no-candidate deficit with no random, held-out, or pseudoinverse
completion.

No metric or binary manual threshold accepts or vetoes iteration. Whole
decoded behavior remains the primary intelligence evidence, and retention is
an objective rather than a numeric cutoff.

## Focused validation

`tests/test_hf_lfm2_semantic_attention_group_qdeim_adapter.py` covers:

- dense compilation from true cached context while a held-out batch remains
  unread;
- causal phase-balanced weights for unequal prefill/decode row counts;
- typed missing-true-context deficit with separately labelled cached
  `context_from_probabilities`;
- exact structural reconstruction from cached probabilities/value history;
- typed functional-rank deficit without silent completion; and
- rejection of a normalized-decoder receipt bound to another chart.

The focused suite passes 5/5. This adapter is intentionally separate from the
active semantic body recipe and does not install or select a production
candidate by itself.
