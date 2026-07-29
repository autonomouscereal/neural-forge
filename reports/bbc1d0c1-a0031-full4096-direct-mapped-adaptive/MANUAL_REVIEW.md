# Exact-4x full-mapped causal continuation: manual review

## Candidate

- Neural Forge commit: `bbc1d0c1`
- Parent parameters: `229,693,184`
- Standalone dense child parameters: `57,420,144`
- Parameter compaction: `4.0002195745102975x`
- Child topology: full 14-layer LFM2 target, hidden width 784, intermediate width 621, 12 query heads, 6 key/value heads, 4096 target-token rows
- Candidate: `a0001-a0031-full4096-direct-mapped-adaptive`
- Fold policy: exact strict target-to-donor root-row projection, observed contextual trie cells overriding root anchors, analytic grouped masked residual solve, and adaptive strongest-surviving background seal
- Training, optimizer, autograd, pruning, quantization, sparsity, parameter sharing, parent sidecar, and parent execution during resume: all zero/false

The candidate saved and freshly reloaded as a standalone direct-local child. Its body inventory, input table, parameter count, and independent output-table topology authenticated across reload.

## Manual intelligence verdict

The decoded whole responses are the acceptance evidence. All 16 were read in full against their task-specific rubrics.

| Task | Verdict | Manual finding |
|---|---:|---|
| conversation-natural-greeting | Fail | Broken token fragments; neither a natural reply nor a useful question. |
| instruction-three-actions | Fail | No coherent numbered household actions and no constraint compliance. |
| explanation-seasons | Fail | No explanation of axial tilt, sunlight, day length, or the distance misconception. |
| arithmetic-multi-step | Fail | No valid calculation, answer, or interpretable reasoning. |
| logic-ordering | Fail | No ordering of Mira, Theo, Jun, and Sal. |
| context-retrieval | Fail | Does not recover Saturday, 4:15 p.m., or the north-gate fact. |
| summarization-priorities | Fail | Does not state the Supplier B decision or expansion-delay risk. |
| comparison-ram-rom | Fail | No coherent analogy or limitations. |
| coding-deduplicate | Fail | No valid Python implementation, example, or complexity explanation. |
| debugging-off-by-one | Fail | Does not identify or correct the slice bug. |
| writing-concise-email | Fail | No usable email or requested shipment facts. |
| planning-dependencies | Fail | No six-step migration plan, dependencies, or rollback path. |
| uncertainty-insufficient-data | Fail | No calibrated conclusion or relevant evidence requests. |
| safety-household-chemical | Fail | No safe immediate response; unusable for the safety task. |
| creative-continuity | Fail | No continuous lighthouse scene, dialogue, or requested ending. |
| multi-turn-correction | Fail | No four peanut-free options, cross-contact handling, or vegan option. |

**Result: 0/16 coherent task responses. This candidate does not retain useful intelligence.**

## What improved, and what did not

The candidate is not intelligent, but the full mapped roots changed the failure mode rather than reproducing the old collapsed loop:

- Exact response matches against the fresh-reload A0031 baseline: `0/16`
- Mean unique-token ratio: `0.07763671875 -> 0.390625`
- Mean maximum repeated-bigram share: `0.71593137254902 -> 0.10343137254902`

Those diagnostics are advisory only. They show that complete root coverage substantially reduced repetitive mode collapse, but manual review shows that it did not restore syntax, semantics, instruction following, or reasoning.

## Located gap

The analytic bridge used 1,160 child prefix feature rows over a 784-dimensional hidden state. It exposed 2,653,535 effective supported target-logit cells and attempted contextual correction for 4,093 non-protected vocabulary rows. The centered fit error changed from `1.0085856914520264` to `0.6342790126800537`, but:

- strict contextual-cell pass rows: `0`
- strict contextual-cell failure rows: `4,093`
- adaptive fallback rows: `4,093`
- fully unsupported/protected baseline rows: `3`

Therefore, missing vocabulary coverage was not the sole blocker. Exact context-free root anchors suppress the worst repetitive collapse, but the aggressively folded child body is not producing a causal/contextual state geometry from which one static output matrix can recover the parent behavior. Adding more targets to the same output-only solve would add constraints to an already incompatible feature geometry.

## Evidence-selected next stage

Keep this complete exact-4x whole child and its authenticated cached evidence. Locate the first parent/child causal divergence at dynamic component boundaries, then fold the residual into the responsible body circuit—attention Q/K/V/O, short convolution/state path, MLP, or their residual handoff—rather than asking `lm_head` to compensate for every upstream error. Component isolation is diagnostic and stage-selective; the compiled artifact remains a fully compacted standalone whole model. No training or old scalar/grid reruns are authorized by this result.

## Artifact locations

- Run: `/home/cereal/model-surgery-lab/artifacts/lfm2_parent_free_causal_resume_20260729/neural-forge-bbc1d0c1/a0001-a0031-full4096-direct-mapped-adaptive`
- Whole-response dossier: `whole-response-dossier.json`, SHA-256 seal `50b0019a6f3563151432d1e122e73ba6a20fc9474a879c8203e64d9dcf339848`
- Active hybrid causal evidence: `fbfb09aa3cbb551582c0e6adf12833a55549d1aa9b85984e22406e61fc71d8fd`
- Mapped target evidence: `c2370414eeddfcbf86e0d19ceda2b9045804856577f10cb5623f05830a2ea05b`
- Installed output table: `dca0bccce8e56070911ecd74e22a896931e6fbd35ff7beed01ae9df11d50d646`
