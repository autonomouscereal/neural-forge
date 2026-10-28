# A0011 global-coordinate exact-4x control outcome

## Verdict

A0011 is an infrastructure failure, not a folding or intelligence result.
The run failed closed before compilation when the single-pass FP32 replay of
`prefill.output.lm_head_input` differed from the canonical prefix replay by a
maximum absolute value of `1.60933e-05`.

The planned blank child was a complete dense 14-layer LFM2 model with
`57,420,144` unique parameters versus the `229,693,184`-parameter parent, an
exact planned parameter compaction ratio of `4.0002195745102975x`.  It used a
4096-row untied token interface and no pruning, quantization, sparsity,
parameter sharing, training, optimizer, gradient, or parent runtime sidecar.

## What this run establishes

- The exact-4x blank topology remained executable and exactly on budget.
- The sealed parent cache and orchestration reached replay certification.
- The current replay verifier still confuses legal fused-kernel reduction
  drift with semantic trace corruption.
- No child was published, no decoded child response was produced, and this run
  provides no evidence for or against the single global residual-map folding
  hypothesis.
- The ledger remained empty for the entire roughly 90-minute job and only the
  terminal receipt/diagnostics appeared, confirming the need for phase
  checkpoints, heartbeats, and incremental structured logging.

## Decision

Do not rerun the global residual-map hypothesis.  Replace the replay tolerance
table with the centralized numeric execution policy, and continue through the
cached pre-causal/token-span path and moving boundary-local reclosure.

## Source artifacts

- `receipt.json`: authenticated failed-closed run receipt
- `diagnostics.json`: terminal diagnostics
- `run.log`: terminal JSON copy emitted by the launcher

