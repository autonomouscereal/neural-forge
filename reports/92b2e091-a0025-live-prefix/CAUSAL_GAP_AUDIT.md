# A0024/A0025 causal-gap audit

This audit compares the complete exact-4x A0024 baseline with the complete
exact-4x A0025 live-child-full-prefix fold. It records structural and numeric
evidence only. The decoded whole-response review remains the behavioral
authority; both children were manually judged incoherent on all 16 prompts.

## What A0025 proved

- A0025 saved and freshly reloaded a standalone 57,422,440-parameter child.
- The child state was
  `3f47d738b1c50a21d91d8c80fd9df310582bd7bac5c713455bbe6b804b8a85fc`;
  the checkpoint model SHA-256 was
  `a6d44b4d28296e72aee5b7c6d1c1ad89712a1bb4af90e5821be0f259f22bfab4`.
- All audited training, backward, optimizer, pruning, quantization,
  parent-forward, and parent-trace-recapture counts were zero.
- Live-prefix replay was real: 42 full-depth recaptures (three per layer),
  9,408 analytic child macrocell replays, 42 unique child-state hashes, and
  replay from embeddings through all 14 layers on every recapture.

## Measured causal gap

The component fits improved locally but did not compose into the complete
child trajectory.

- A0025 ShortConv corrected local-action relative RMSE was usually
  0.00457-0.0484, while complete endpoint relative RMSE was 4.99-14.37.
- Layer 1 improved locally from 0.06334 to 0.01124 while its endpoint worsened
  from 4.653 to 4.991.
- Terminal ShortConv layer 13 worsened from 8.684 to 14.375 endpoint relative
  RMSE and from 63.698 to 102.290 maximum absolute endpoint error.
- Attention local-output corrections were often 0.000579-0.101674 relative
  RMSE, while complete residual-chart endpoints remained 0.866-1.022 with
  mean cosine near zero (-0.00893 to 0.03812).

The global residual closure was physically a no-op in both complete runs.

| Evidence | A0024 | A0025 |
|---|---:|---:|
| Continuous dense delta L2 | 3.5367e-17 | 1.0106e-17 |
| Physically changed coordinates | 0 | 0 |
| Physically changed parameter IDs | 0 | 0 |
| Installed delta L2 | 0 | 0 |
| First effectful ray multiplier | 16,858.5 | 74,583.4 |
| Residual before | 0.807871455539 | 0.812595952917 |
| Residual after installed correction | unchanged | unchanged |

Multiplying the same direction by the first-effectful factor is not a
principled repair: the factors are extremely large and do not establish that
the direction closes the causal circuit. The replacement must synthesize
target-dtype-representable physical weights directly.

## Output-boundary evidence

- Donor vocabulary: 65,536; target vocabulary: 4,096.
- Supported root logit cells: 32,414, about 1.22% of the 648 x 4,096 root
  matrix.
- Retained internal-prefix rows: 512 of 21,601 (about 2.37%); 21,089 were
  omitted.
- Suffix rewrites quarantined: 459; next-donor conditionals were unavailable.
- The causal-trie bytes were identical between A0024 and A0025.

The complete structural tokenizer bridge must remain the foundation for every
target embedding/head row. Bounded trie evidence may contribute only a
residual correction; it cannot define the unsupported output space.

## Causal interpretation

The primary measured failure is not merely insufficient planting strength or
insufficient width. Source-local component actions can fit while their outputs
land in the wrong evolving child coordinates. The current exact-4x semantic
stack forces a single self-dual residual gauge and suppresses non-identity
per-layer chart candidates. Live replay updates the next local solve, but it
does not reconcile earlier layers after downstream coordinates change, and
the final global correction changes no physical weights.

The next complete child must therefore combine, rather than separately retry:

1. A deterministic moving residual chart at every internal boundary, with
   distinct encoder and decoder maps, cycle consistency, and boundary-0 and
   terminal compatibility with the full token/head interface.
2. Whole-macrocell transport in those charts. The transition defect
   `E_(l+1) D_l - I` must be absorbed into ordinary physical writers so the
   saved child remains a native standalone model.
3. One forward analytic sweep and one reverse reconciliation sweep from cached
   parent/live-child evidence, with direct FP32 replacement synthesis. A
   nonzero requested closure must cause at least one representable physical
   coordinate change; that is an execution-integrity invariant, not an
   intelligence score.
4. State-aware ShortConv realization using cached full-prefix histories and a
   balanced controllability/observability construction if moving-gauge closure
   alone remains insufficient.
5. A vocabulary-complete structural token/head bridge, with contextual trie or
   vocabulary-margin evidence used only as a bounded residual correction.

Topology reallocation can provide capacity, but it cannot by itself repair the
measured coordinate closure, recurrent-state accumulation, zero
materialization, or output-coverage gaps.

## Artifact roots and evidence hashes

- A0024:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_composite_20260724/neural-forge-cf6adf89/fp32-source-local-residual-rho025-whole4x-a0024`
- A0025:
  `/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_composite_20260724/neural-forge-92b2e091/fp32-source-local-residual-rho025-live-full-prefix-whole4x-a0025`

| Artifact | A0024 SHA-256 | A0025 SHA-256 |
|---|---|---|
| `semantic-layer-stack.json` | `d7ca6a272a028acaf397b5dca66df3d85d0a17bbe0df88e616484562f6ad627f` | `c361791bd3a490e4a47778cbdf65444a7fd449f9a64ed7fc9f7c1aad04603950` |
| Internal stack result | `36ec53b4550aced83e7acd9417a25d47e25a8b5d7afe3659b9e000c6d16b5267` | `c05264babbf5b88fb9c173a38d658fbe3cecb48f48fd04134068407195f5def4` |
| `body-fold-realization.json` | `a7ad473e7fc682f4b38949ef93876f56e0efa235dc379a4ebf94b13eba690c5f` | `a724f9093a4ee6aa31805f21c12780387fa1cf2d2f2b85338c9ce8df1cbf6e1d` |
| `causal-token-trie-evidence.json` | `6558c4e92d0da7968ef7818ec47df7770d06fb6af91c4a6f93b0109f410be436` | same |
| `terminal-cache-evidence.json` | `8321e566f97d4cb958138ab0272c852f26d099100af35a909dfe156c7ff86f9a` | same |
| `structural-interface-install.json` | `5dde1cd00d0027862d888b951ad77ae9e46dd4132731e498d3a15ebb27f5eb43` | `ecb45689fbb4ad90071ab0c0b76d89e6c57c9e0ce65b3b2602bbf5e1f799fca1` |
| `causal-output-install.json` | `0fe50bafa800d281c326754f22311aab013e6bf4ef1de34a4e85f66a0fd600fb` | `e553343efdd26144fff49d1b59654c9ff0fafb8f11bb1d81792cebc2b01c3b85` |
| `whole-response-dossier.json` | `40cf6bbdc14092b1df65d396f8e26af25688f3187a81f727e13018943679e841` | `b399ed8acaeabbeb34c8f09f77fa1a02513bfe32767397c6acecde014899cae7` |
