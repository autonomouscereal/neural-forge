# Exact-4x unrestricted full-root causal continuation: manual review

## Candidate

- Neural Forge runtime: immutable commit `bbc1d0c1`
- Parent parameters: `229,693,184`
- Standalone dense child parameters: `57,420,144`
- Parameter compaction: `4.0002195745102975x`
- Child topology: complete 14-layer LFM2 target, hidden width 784, intermediate width 621, 12 query heads, 6 key/value heads, 4096 output rows
- Candidate: `a0031-full4096-direct-mapped-unrestricted-r1e6`
- Causal solve: full-4096 mapped roots plus contextual trie overrides, unrestricted grouped per-output analytic ridge, `ridge_fraction=1e-6`, residual scale 1.0, no row-norm contraction, and no protected special rows
- Training, optimizer, autograd, pruning, quantization, sparsity, parameter sharing, parent sidecar, parent loads, parent execution, and parent trace recapture: all zero/false

The candidate saved and freshly reloaded as an independent direct-local FP32 checkpoint. Body inventory, tokenizer semantics, input table, output table, parameter count, and untied dense topology authenticated across reload.

## Manual intelligence verdict

All 16 complete decoded responses were read against their task-specific requirements. Numeric diagnostics were used only to understand the failure mode.

| Task | Verdict | Manual finding |
|---|---:|---|
| conversation-natural-greeting | Fail | Token soup; neither a natural reply nor a follow-up question. |
| instruction-three-actions | Fail | No coherent numbered actions or constraint compliance. |
| explanation-seasons | Fail | No intelligible explanation of axial tilt, sunlight, or the distance misconception. |
| arithmetic-multi-step | Fail | No valid calculation, answer, or reasoning. |
| logic-ordering | Fail | No ordering or explanation. |
| context-retrieval | Fail | Does not recover Saturday, 4:15 p.m., or the north-gate fact. |
| summarization-priorities | Fail | Does not preserve the Supplier B decision or expansion-delay risk. |
| comparison-ram-rom | Fail | No coherent analogy or limitations. |
| coding-deduplicate | Fail | No valid function, example, or complexity explanation. |
| debugging-off-by-one | Fail | Does not identify or correct the slice bug. |
| writing-concise-email | Fail | No usable vendor email or requested facts. |
| planning-dependencies | Fail | No six-step dependency-aware migration or rollback plan. |
| uncertainty-insufficient-data | Fail | No calibrated conclusion or evidence request. |
| safety-household-chemical | Fail | No safe immediate advice; unusable for the safety task. |
| creative-continuity | Fail | No continuous lighthouse scene satisfying the constraints. |
| multi-turn-correction | Fail | No four peanut-free options, cross-contact handling, or vegan option. |

**Result: 0/16 coherent task responses. The unrestricted output solve does not restore useful intelligence.**

## What the control resolves

The active 1,160-row hybrid evidence solve improved centered supported-logit error:

- Structural baseline: `1.008589506149292`
- Adaptive background-nullspace run: `0.6342790126800537`
- Unrestricted run: `0.25200679898262024`

The unrestricted correction changed all 4,096 output rows. It also removed the EOS protection defect: 2/16 responses now emitted EOS instead of 0/16, while 14/16 still reached 256 tokens. Mean repeated-bigram share improved from `0.10343137254902` to `0.025236891852344`; mean unique-token ratio changed from `0.390625` to `0.369153516914191`. These mechanical improvements did not produce syntax, semantics, instruction following, reasoning, or safety competence.

The result closes the immediate ambiguity: the adaptive background seal and norm contraction were harmful, but they were not the sole cause. Even the unrestricted dense head leaves substantial cached-evidence error and fails every held-out free-running task. Output-only correction cannot compensate for the current folded causal state geometry.

## Evidence-selected body continuation

Retain the A0031 exact-4x compiler, body capsule, mapped causal evidence, and complete-child invariant. Do not return to uniform shrinkage or isolated toy children.

The strongest existing component evidence identifies a heterogeneous exact-budget fold:

1. Preserve the internal `(B, x, C, FIR)` channel algebra of layer-0 ShortConv and likely layer 13; fold their linear input/output seams and factor their output-action writers instead of collapsing their internal channels.
2. Allocate substantially more intact GQA capacity to attention layers 4, 6, and 8, with layer 4 first. A0031 already performs coupled Q/K/V/O residual-attention folding; the missing variable is nonuniform capacity, not another independent matrix transplant.
3. Recover the exact 57,420,144-parameter budget by compressing the healthier middle ShortConv components and distributed MLP widths more aggressively.
4. Re-run the existing moving-gauge forward/reverse physical reclosure after each selected realization, then finalize terminal W2 and the causal output head last.
5. Fresh-reload one complete standalone child and manually review all held-out responses. Component substitutions remain fault-localization evidence only, never the final compression claim.

This is a dynamic allocator/compiler change inside the existing full-model path, not a manually assembled architecture or a smaller compaction target.

## Artifact locations

- Run: `/home/cereal/model-surgery-lab/artifacts/lfm2_parent_free_causal_resume_20260729/neural-forge-bbc1d0c1/a0002-a0031-full4096-direct-mapped-unrestricted`
- Whole-response dossier seal: `ad69c4dbabc9c31825e0263e2fc8fdbebc41f75bb9bfef944cbb684c43b94321`
- Active hybrid causal evidence: `fbfb09aa3cbb551582c0e6adf12833a55549d1aa9b85984e22406e61fc71d8fd`
- Mapped target evidence: `c2370414eeddfcbf86e0d19ceda2b9045804856577f10cb5623f05830a2ea05b`
- Causal bridge seal: `b7a86546a0770d19bbf8168c64746d143f307394587ad4dc534d101af68d19ce`
- Installed output table: `a96aa26805f3fe437bba19e44e8c4a8a51abd70a4f6cbb9955d124a8c0ed6098`
