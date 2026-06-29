# Research Paths For Model Folding And Recovery

Current as of 2026-06-29. This file tracks implementation directions for the
blank-target, cross-family compression goal. It deliberately focuses on
2025-2026 work and actionable hooks for Neural Forge.

## Highest-Priority Paths

No 2025-2026 primary source found so far proves arbitrary "large parent into a
totally random blank child at 25x+ with tiny recovery" as a solved recipe. The
practical path remains staged: tokenizer transplantation, activation transport,
merge/fold width and depth, local reconstruction compensation, then short
behavior-gated recovery. Each stage needs its own ablation because local PPL or
activation reconstruction gains have repeatedly failed to imply math/code
behavior recovery in the strict blank-target runs.

| Path | Current Source Trail | Neural Forge Hook |
|---|---|---|
| On-policy representation distillation | OPRD, arXiv `2606.06021`; Improving Language Model Distillation through Hidden State Matching, ICLR 2025 | Extend `hf-xfamily-block-recover` so child rollouts and donor-generated completions are decoded, fed through both models, and matched through donor/child hidden projections at selected layers. Existing `--on-policy-hidden-kd-weight` and `--teacher-generate-hidden-kd-weight` are the seeds; the next version should use stronger rollout scheduling and layer/interface projectors. |
| Cross-tokenizer behavior transfer | Breaking the Tokenizer Barrier, arXiv `2606.09456`; X-Token, arXiv `2605.21699` | Use chunk/span alignment and sparse tokenizer projections for behavior-level recovery. `--teacher-generate-lm-weight` is the current lightweight primitive: donor completions become child-token training text while the folded block is active. |
| Byte-level cross-tokenizer interface | Cross-Tokenizer LLM Distillation through a Byte-Level Interface, arXiv `2604.07466`; Cross-Tokenizer Likelihood Scoring, arXiv `2512.14954` | `--byte-kd-weight` is the logit-space probe. `--byte-hidden-kd-weight` now routes donor first-byte targets directly into active block hidden states through a fixed child byte probe, avoiding tied-head mutation. The stronger version should fit/task-select byte probes instead of only averaging output rows. |
| Low-rank clone initialization | Low-Rank Clone, arXiv `2505.12781` | Add a teacher-to-student low-rank projection initializer for attention/MLP linear weights, train only projection factors/RMSNorm on activation clone + logit KL + CE, then materialize a normal child checkpoint. This is a larger follow-up to `pre-linear`: it constrains activation maps so they are not full-rank behavior-damaging edits. |
| Activation-space folding | FLAT-LLM, arXiv `2505.23966` | Collect calibration activations, PCA-fold attention value/output heads, use activation-correlation channel selection for MLPs, and allocate ranks by decoder input/output dissimilarity. This is the next training-free folding path if full-rank residual maps keep hurting generation. |
| Activation-space optimal transport folding | DOTResize, arXiv `2507.04517`; Transport and Merge, arXiv `2602.05495` | Replace simple rank/depth grouping with entropic transport over activation signatures for MLP channels, attention heads, and residual dimensions before any pruning. `--activation-feature-mode covariance` is now implemented as an opt-in co-activation signature for this path; it still needs broad-gate validation before promotion. |
| Sequential activation compensation | GRAIL, arXiv `2602.23795`; Chain of Merges, arXiv `2508.21421`; RegMean++, arXiv `2508.03121` | After folding each donor layer group into the target block, solve a local ridge/Gram compensation map, absorb it into downstream projections, then recompute activations before moving to the next block. Gate promotion by behavior, not just local reconstruction error. |
| Cross-tokenizer likelihood matching | ALM, arXiv `2503.20083`; Cross-Tokenizer Likelihood Scoring, OpenReview 2026 | Add byte-offset chunk likelihood regression and tokenizer-bias debiasing as a regression test for cross-tokenizer recovery. Useful when projected-vocab and byte losses disagree with greedy behavior. |
| Geometry-preserving hidden recovery | CKA-QAD, arXiv `2606.05682`; Flex-KD / task-tangent geometry, arXiv `2507.10155` | Keep `projected-rsd` and CKA losses, but add task-focused feature selection for math/code spans so the tiny child preserves behavior-relevant directions instead of average hidden geometry. |
| Submodule-level replacement | SubFit, arXiv `2606.02559`; Merging Feed-Forward Sublayers, arXiv `2501.06126` | Fold and fit attention and MLP residual outputs independently, instead of treating each transformer block as one indivisible unit. |

## Current Evidence

The 2026-06-28 and 2026-06-29 runs show that output-interface losses alone are not enough for
the strict Qwen3-4B to blank SmolLM2-135M `29.90x` case:

- static projected-vocab KD, byte KD, and optional head training are active and
  behavior-gated;
- behavior-best per-layer selection prevents lower-loss states from replacing
  better-generation states;
- donor-generated completions are now active as block-coupled LM/hidden
  recovery streams, but the first strict 29.90x runs preserved baseline behavior
  rather than recovering arithmetic/code;
- arithmetic and Python outputs still collapse, so the next useful work is
  structural activation compensation and stronger on-policy representation
  recovery.
- per-layer input activation projections in `hf-xfamily-block-recover` prevent
  the worst global-projection collapses, but the best 2026-06-29 run still held
  math at `1/40`;
- response-token LM losses were added to blockwise folding. Tied output-head
  response training lowered PPL while collapsing behavior, and no-head response
  training preserved the conservative profile without improving arithmetic. The
  next attempt should avoid tied-head mutation and use staged hidden/interface
  transfer before response CE.
