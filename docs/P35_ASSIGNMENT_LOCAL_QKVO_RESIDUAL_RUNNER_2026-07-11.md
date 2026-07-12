# Decisive p35 assignment-local Q/K/V/O residual run

Date: 2026-07-11

## Why this run follows the O-only result

The completed rank-8 O-only residual experiment selected step `0`. Calibration
and heldout discrete behavior were unchanged:

```text
calibration top-1           38/48 (0.7916667)
calibration mean rank       1.3541667
heldout top-1 baseline      56/64 (0.875)
heldout top-1 locked        56/64 (0.875)
heldout mean rank           1.140625 baseline and locked
heldout exact rollouts      5/8 baseline and locked
heldout token matches       20/32 baseline and locked
```

The verified O-only report file SHA-256 is
`531942412df7a1b9900014ad5481a03aba0e331609e6d5bb27ae447dbbeff243`.
Its report hash is
`d84670b556b2df03e289085d0e16c3670df60bfaa010f4317d20b0aa0d17cac4`.
Those values are negative provenance only; they are not fit or selection data.

O-only repair could change the final residual write, but it could not change
which content the assignment addressed or what K/V state entered the cache.
The decisive fallback therefore trains all four assignment-local residual
branches at the exact `L19/G4 -> L21/G6` redirect:

```text
Q correction: before Q norm and RoPE
K correction: before K norm, RoPE, and cache append
V correction: before cache append
O correction: post-softmax group context before residual add
```

Projected factors are free to become numerically unrelated to parent weights.
Only causal output actions and parent output logits supervise them.

## Fixed run

```text
prototype transition       280 -> 279
rank                       8
Q parameters               24,576
K parameters               21,504
V parameters               21,504
O parameters               24,576
adapter total              92,160
net parameters saved       3,183,488
cached rollout horizon     4
fit prompts                8
calibration prompts        6
heldout prompts            8
fit updates                32 batch-2 updates
Q/K learning rate          0.001
V/O learning rate          0.002
gradient norm clip         1.0
calibration gates          0, 8, 16, 32
initialization             U=0, deterministic random R, seed 0
```

Thirty-two updates consume the complete deterministic fit schedule exactly
once. Parent-greedy and baseline-child-greedy prefix batches are interleaved.
Fit alone receives gradients. Calibration alone may lock a state, and only a
higher parent-action top-1 rate or lower parent-action rank can promote it.
Continuous loss and log-probability changes are diagnostic only.

Heldout model forwards begin after `winner-locked`. Heldout compares the
zero-adapter tentative child and the locked QKVO state, names no winner, and
cannot change calibration selection. When their student rollouts diverge, the
combined action comparison is labeled as an own-rollout-domain comparison,
not a common-prefix causal delta.

## Learning and movement diagnostics

The O-only report restored step 0 before serializing its final metrics, so it
could not show whether rejected steps moved or had inactive gradients. The
QKVO report retains the evidence needed to answer that question:

- all 32 fit-step total, hard-action, and output-distribution losses;
- total raw gradient norm and gradient clipping scale at every step;
- raw gradient norm for Q, K, V, and O at every step;
- cumulative parameter-delta norm for each branch after every update;
- nonzero-gradient activation for all eight `left/right` factor tensors;
- aggregate loss, gradient, and parameter-movement summaries;
- full calibration output metrics and adapter-state hashes at steps
  `0/8/16/32`;
- the pre-lock final state hash and the separately restored winner hash.

These diagnostics cannot select or promote. If step 0 wins again, the history
will distinguish three very different outcomes: no gradient reached a branch,
parameters moved but output loss worsened, or continuous output fit improved
without a discrete action change.

## Transaction and non-claims

The runner reinstalls the selected in-memory QKVO state exactly for heldout,
then removes all Q/K/V/O wrappers and restores incumbent p35 exactly. The same
cleanup executes after training failure. It writes one JSON report only and
has no save or commit flag.

Implementation:

```text
src/neural_forge/hf_assignment_local_qkvo_residual_adapter.py
experiments/run_qwen_p35_assignment_local_qkvo_residual.py
```

Prepared GPU3 launch script:

```text
experiments/remote_run_p35_qkvo_residual_rank8_h4_s32_fp16_p40_gpu3_20260711.sh
```

The authoring task did not upload or launch it. A favorable report remains an
experiment result, not a committed checkpoint or certified full p35-to-p34
compaction.
