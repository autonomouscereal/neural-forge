# Architecture

Neural Forge is organized around shape surgery primitives, not a single model
family.

## Current Engine

The current dense engine expects a feed-forward checkpoint where each hidden
unit has:

- an incoming row in `W[layer]`;
- a bias entry in `b[layer]`;
- an outgoing column in `W[layer + 1]`.

This makes three operations straightforward:

- shrink: cluster hidden units and fold each cluster into one survivor;
- expand: copy hidden units and split outgoing columns;
- merge: resize source models to a common hidden shape, align hidden units, then
  average or SLERP.

## Extension Contract

Future model-family adapters should expose equivalent axes:

- Transformer FFN intermediate channels map to dense hidden neurons.
- Attention heads map to groups of Q/K/V/O tensor slices.
- MoE experts map to duplicated or merged expert modules plus router repair.
- Embedding/vocabulary expansion maps to extra embedding rows plus output-head
  rows and tokenizer/config metadata.
- Native multimodal expansion maps to added token streams or input projections
  plus modality and positional metadata.

Adapters should provide:

1. `inspect`: shapes, tied weights, config metadata, and parameter counts.
2. `dry_run`: proposed shape edits without writing a checkpoint.
3. `apply`: streamed tensor rewrite.
4. `verify`: shape validation plus optional logit/activation drift checks.
5. `report`: JSON metadata for every changed tensor.

## Design Rules

- Prefer folding/merging over discard-only pruning.
- Keep merge/prune/grow recipes declarative.
- Use calibration data only when the user supplies it.
- Keep database-free local operation; no ORM, Pydantic, or SQLAlchemy.
- Avoid vendoring GPL or no-license research code. Mirror algorithms cleanly.

