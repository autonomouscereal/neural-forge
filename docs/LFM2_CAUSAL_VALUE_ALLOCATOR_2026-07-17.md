# LFM2 Cached-Causal Value Allocator

Date: 2026-07-17

Status: implemented and locally verified; no GPU run or checkpoint launched by
this implementation task.

## Why this lane exists

The completed uniform packed-value control proved real full-directory
compression at the requested large targets:

- `50.0000034473%` serialized-byte reduction remained coherent, but retained
  only `9/11 = 81.8%` parent-correct tasks and `6/8 = 75%`
  parent-demonstrated capability contracts.
- `75.0000062483%` collapsed, with `56/64` empty responses.

The 50% artifact therefore provides a useful control and a real representation
to improve. Its allocator chose seven-bit groups from normalized weight
reconstruction error alone. Inspection showed that this downgraded every
`embedding_norm`, `operator_norm`, `ffn_norm`, attention Q/K norm, and tiny
depthwise-convolution kernel because those small tensors looked cheap by
weight SSE. It also made no causal distinction among FFN producers, residual
writers, attention values, Q/K paths, or tied token rows.

This lane keeps the already-proven standalone packed runtime and changes the
allocation decision. The intended comparison is the exact same 50% physical
checkpoint ceiling, with a promotion target of at least 90% of intelligence
actually demonstrated by the parent.

## Allocation policy

The allocator authenticates the portable cached-causal frontier produced by
`build_lfm2_cached_causal_evidence.py`. Each physical FFN proposal in that
frontier was selected from cached activation rows and measured on disjoint
validation evidence for:

- final parent-action logit margin;
- EOS and termination;
- repetition;
- task anchors;
- downstream residual paths; and
- downstream Q/K/V/context attention paths.

For each layer, the allocator conservatively takes the maximum weighted damage
per fraction of FFN width removed. It normalizes those layer sensitivities and
ranks the incremental 8-to-7-bit group distortion with:

```text
incremental normalized reconstruction SSE
* (1 + 15 * normalized layer sensitivity)
* role risk multiplier
```

Role multipliers are:

| Role | Multiplier | Reason |
|---|---:|---|
| FFN `w1` / `w3` producer | `1` | preferred low-precision spend |
| FFN `w2` residual writer | `2` | directly writes the residual stream |
| attention `v_proj` | `4` | affects cached/contextual values |
| attention `out_proj` residual writer | `6` | writes attention context into the residual stream |

Only those four roles may receive seven-bit groups. Tied embeddings/output,
all normalization, all Q/K projections, convolution operators, biases, and
unclassified tensors are excluded from the lower-precision candidate set.

## Hard exact anchors and their byte exchange

The first correction is deliberately simpler than the causal ranking:

- every normalization tensor, including Q/K norms, reloads byte-exactly at
  source `bfloat16`/`float16`;
- every depthwise-convolution kernel reloads byte-exactly;
- all local BOS, EOS, PAD, other special, and chat-control token rows reload
  byte-exactly;
- optional frequency/margin-selected rows are accepted only from a sealed
  calibration/selection-cache document bound to the causal parent cache, its
  exact `selection_rows_sha256` seal, and its declared frequency/margin
  selection method; and
- no independent 64-case holdout prompt, output, or token ID can enter
  allocation.

Because the embedding and LM head are tied to the same physical tensor store,
an exact protected token row preserves both input-token identity and that
token's competing output logit.

Exact source bytes are base64 encoded inside the already counted, hashed
checkpoint manifest. This avoids an unclaimed sidecar and preserves the strict
existing loader. The artifact records:

- exact source-value bytes;
- encoded manifest characters and canonical overlay-record bytes;
- the lower-group count required without exact overlays;
- additional low-risk groups used to pay for exact overlays; and
- the exact compensating packed-payload bytes saved.

The full manifest remains part of the exact 50% directory ceiling. Two fresh
reloads must be logit-bitwise identical. The runtime has no persistent dense
weight Parameters; exact anchors are immutable buffers reconstructed from the
authenticated manifest.

## Parent-relative intelligence contract

Absolute manual scoring remains diagnostic, not the promotion criterion. It
previously penalized inherited parent failures:

- a parent and child that were both blank;
- an exact copy of a parent misconception; and
- other abilities the parent never demonstrated.

The new paired review packet requires the reviewer to score parent and
candidate separately on the same `0..4` rubric for correctness, coherence,
instruction following, and capability retention. A dimension is eligible only
when the parent scores at least `3`. Parent scores below `3` are neutralized.
The child retains a demonstrated dimension when it also scores at least `3`.

Promotion requires:

- every automatic parent-relative component to meet its existing `0.90`
  threshold;
- at least `0.90` paired manual demonstrated-dimension retention; and
- zero clear manual regression dimensions.

Thus a large regression such as the prior `manual_05` loss of prompt binding,
diagnostic questions, next-step planning, and termination rejects the child
even if an aggregate average could hide it. Exact wording, token identity, and
sentence-structure matching remain forbidden.

## Representation and execution guarantees

This is serialized-value compression, not structural parameter-count
compaction. Every logical layer, head, neuron, tokenizer entry, and tied
embedding/head role remains present.

The implementation:

- reuses `hf_lfm2_value_fold_control.py` for group analysis, raw bit packing,
  payload hashing, exact directory accounting, meta-device construction,
  packed operators, and fresh reload;
- stores no dense checkpoint or external weight sidecar;
- uses local model/tokenizer directories only;
- performs zero parent forwards or generations during allocation;
- performs zero backward calls, optimizer construction, optimizer steps, or
  training;
- generates only the candidate on the sealed independent 64-case suite; and
- supports a model-free, cached-output-only review finalizer so a preliminary
  checkpoint is not duplicated after human ratings.

## Files

- `src/neural_forge/hf_lfm2_causal_value_allocator.py`
  - frontier authentication and causal sensitivity reduction;
  - role-aware exact-50% allocation;
  - exact norm/conv/special-token overlays;
  - physical byte-exchange accounting;
  - strict wrapper loader and inspector.
- `src/neural_forge/hf_lfm2_parent_relative_manual_gate.py`
  - paired packet, rating, and report seals;
  - parent-demonstrated dimension selection;
  - absent-parent capability neutralization;
  - 90% retention and clear-regression veto.
- `experiments/run_lfm2_causal_value_allocator.py`
  - local parent identity checks;
  - exact checkpoint export;
  - two-reload proof;
  - independent 64-case candidate generation and cached-parent scoring.
- `experiments/finalize_lfm2_causal_value_allocator.py`
  - model-free ratings postflight;
  - separately sealed final report without replacing preliminary artifacts.

## Local verification

Focused implementation tests:

```text
13 passed
```

They cover causal evidence reduction, exact physical accounting, protected
role enforcement, exact source-dtype reloads, tied protected token rows,
bitwise-identical fresh reloads, ordinary-control rejection, paired
parent-relative manual scoring, inherited-failure neutralization, clear
regression rejection, runner configuration, and the no-training/no-hub static
contract.

## Launch boundary

The first real run should start only after the active cached-causal frontier
producer emits and authenticates its completed frontier document. It should:

1. reuse that frontier without rebuilding parent traces;
2. use local tokenizer special/chat-control rows plus an optional
   selection-cache-only token-row document bound to the exact cached selection
   rows;
3. export exactly one 50% checkpoint;
4. compare it against the existing sealed independent 64-case parent envelope;
5. review the paired parent/candidate packet;
6. finalize from cached outputs only; and
7. promote only if the final parent-relative lower bound is at least `0.90`
   with no clear regression.

The completed uniform 50% control remains the comparison baseline. This
implementation does not claim improvement until the real cached-causal
checkpoint finishes the same holdout and paired review.
