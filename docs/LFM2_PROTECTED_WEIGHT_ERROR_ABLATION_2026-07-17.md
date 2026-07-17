# LFM2 Protected-Weight-Error Exact-50% Ablation

Status: implemented and locally verified; this document does not claim a real
parent-relative intelligence result.

## Purpose

This is a deliberately noncausal large-target ablation that can run while the
cached-causal frontier is still being produced. It tests whether the strongest
currently justified hard-anchor policy can improve on the completed uniform
packed-value control without waiting for causal layer sensitivities.

It is not a substitute for cached-causal residual/attention-aware allocation.
Its causal multiplier is fixed to exactly `1.0`.

## Allocation rule

The checkpoint remains the existing standalone packed-value format with the
same exact `0.50` full-directory byte ceiling.

Only these groups may move from eight to seven bits:

- FFN W1/W3 producer groups, role multiplier `1`;
- FFN W2 residual-writer groups, role multiplier `2`;
- attention V groups, role multiplier `4`; and
- attention output residual-writer groups, role multiplier `6`.

Their global spend order is:

```text
incremental normalized reconstruction SSE
* role risk multiplier
* fixed causal multiplier 1
```

Embedding/output, every normalization tensor, attention Q/K, convolution
state, biases, and unclassified tensors are never eligible for seven bits.

## Exact anchors

The following source-dtype values are encoded inside the counted manifest and
restored after a strict fresh reload:

- every normalization tensor, including Q/K norms;
- every depthwise-convolution kernel; and
- all locally declared BOS, EOS, PAD, other special, and chat-control token
  rows in the tied input/output embedding.

The manifest records the exact raw bytes, base64 characters, canonical overlay
record size, added lower groups that pay for the overlays, and compensating
payload bytes. No sidecar or uncounted dense parameter exists.

## Distinct noncausal seal

The artifact uses:

- manifest key `protected_weight_error_allocation`;
- schema `neural-forge-lfm2-protected-weight-error-allocation-v1`;
- policy `lfm2_protected_weight_error_role_aware_8b7b_v1`; and
- source binding
  `neural-forge-lfm2-protected-weight-error-source-binding-v1`.

It explicitly records:

- no cached-causal frontier;
- no causal evidence;
- no parent trace cache;
- no independent holdout evidence during allocation;
- no parent forward or generation;
- no gradients, optimizer, backward call, or training.

The cached-causal loader rejects this artifact, and the ablation loader rejects
ordinary or relabeled packed artifacts.

## Behavioral gate

After export and two bitwise-identical fresh reload probes, only the candidate
generates responses. Its 64 outputs are evaluated against the existing sealed
independent64 parent envelope.

Open-ended cases use the same paired parent-relative gate as the causal lane:

- abilities the parent does not demonstrate are neutralized;
- exact wording is not required;
- candidate responses must remain correct, coherent, instruction-following,
  and capable relative to the parent;
- promotion requires at least `0.90` demonstrated-capability retention; and
- any clear regression rejects promotion.

The first pass emits `parent_relative_manual_packet.json`. The separate
finalizer consumes paired ratings from cached outputs only and loads no model.

## Local verification

Focused result:

```text
9 passed
```

That verifies exact byte accounting, protected-role exclusion, causal factor
`1`, the distinct seal, parent immutability, exact anchors, tied identity, two
fresh reloads, the no-frontier CLI, and the no-training/no-hub static contract.

The existing cached-causal and uniform-control tests remain the compatibility
boundary.
