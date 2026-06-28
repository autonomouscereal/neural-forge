# Research Paths For Model Folding And Recovery

Current as of 2026-06-28. This file tracks implementation directions for the
blank-target, cross-family compression goal. It deliberately focuses on
2025-2026 work and actionable hooks for Neural Forge.

## Highest-Priority Paths

| Path | Current Source Trail | Neural Forge Hook |
|---|---|---|
| On-policy representation distillation | OPRD, arXiv `2606.06021` | Extend `hf-xfamily-block-recover` so child rollouts are decoded, fed to the donor, and matched through donor/child hidden projections at selected layers. Existing `--on-policy-hidden-kd-weight` is the seed; the next version should use stronger rollout scheduling and layer/interface projectors. |
| Byte-level cross-tokenizer interface | Cross-Tokenizer LLM Distillation through a Byte-Level Interface, arXiv `2604.07466`; Cross-Tokenizer Likelihood Scoring, arXiv `2512.14954` | The first-byte KD added here is a cheap probe. The stronger version should attach a temporary byte decoder head to hidden states, train it during recovery, and backprop byte-level donor targets into the child block without relying on the tied LM head. |
| Activation-space optimal transport folding | DOTResize, arXiv `2507.04517`; Transport and Merge, arXiv `2602.05495` | Replace simple rank/depth grouping with entropic transport over activation signatures for MLP channels, attention heads, and residual dimensions before any pruning. |
| Sequential activation compensation | GRAIL, arXiv `2602.23795`; Chain of Merges, arXiv `2508.21421`; RegMean++, arXiv `2508.03121` | After folding each donor layer group into the target block, solve a local ridge/Gram compensation map, absorb it into downstream projections, then recompute activations before moving to the next block. Gate promotion by behavior, not just local reconstruction error. |
| Geometry-preserving hidden recovery | CKA-QAD, arXiv `2606.05682`; Flex-KD / task-tangent geometry, arXiv `2507.10155` | Keep `projected-rsd` and CKA losses, but add task-focused feature selection for math/code spans so the tiny child preserves behavior-relevant directions instead of average hidden geometry. |
| Submodule-level replacement | SubFit, arXiv `2606.02559`; Merging Feed-Forward Sublayers, arXiv `2501.06126` | Fold and fit attention and MLP residual outputs independently, instead of treating each transformer block as one indivisible unit. |

## Current Evidence

The 2026-06-28 runs show that output-interface losses alone are not enough for
the strict Qwen3-4B to blank SmolLM2-135M `29.90x` case:

- static projected-vocab KD, byte KD, and optional head training are active and
  behavior-gated;
- behavior-best per-layer selection prevents lower-loss states from replacing
  better-generation states;
- arithmetic and Python outputs still collapse, so the next useful work is
  structural activation compensation and stronger on-policy representation
  recovery.
