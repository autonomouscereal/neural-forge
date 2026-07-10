# Fail-Closed Transport Pivot (2026-07-09)

This note records the implementation consequences of current 2025-2026 primary
sources. It does not treat tensor-shape compatibility as evidence of capability
transfer.

## Evidence

### DOTResize (2025, revised 2026)

Source: [DOTResize: Reducing LLM Width via Discrete Optimal Transport-based
Neuron Merging](https://arxiv.org/abs/2507.04517)

- Width compression is formulated as transport of the full activation signal
  onto a smaller neuron support, rather than deletion followed by compensation.
- Transport maps are estimated from activation signatures on exemplar tokens.
- A map applied to one operator must be inverted or pseudoinverted at the
  neighboring consumer. Independent per-tensor projections do not preserve the
  composed path.
- General residual maps are QR-factorized so the orthogonal factor crosses
  RMSNorm and the remaining factor is absorbed into neighboring linear maps.
  The paper reports that omitting this normalization-preserving treatment is
  unstable.
- Attention and MLP junctions need separate, consistently absorbed maps. GLU
  branches and grouped-query attention are explicit topology, not interchangeable
  matrices with similar shapes.
- Their zero-shot evidence is for 20-30% width reduction. It does not establish
  arbitrary or thousand-fold zero-recovery compression.

### Transport and Merge (2026)

Source: [Transport and Merge: Cross-Architecture Merging for Large Language
Models](https://arxiv.org/abs/2602.05495)

- Heterogeneous models are aligned from activations on shared stimuli, then the
  activation correspondences are lifted into weight-space transport operators.
- Pre-projection and post-projection features are different representation
  spaces and receive separate transport plans.
- Layer correspondence is global rather than a proportional index lookup.
- A transported linear operator has the form of a source operator bracketed by
  target-to-source and source-to-target coordinate maps. This supplies the
  mathematical contract for path-consistent weight transport.
- The published no-adaptation experiments improve established target models;
  they do not prove transfer into a fully random target. A blank target therefore
  needs anchors induced from the parent, not activation correspondences to
  meaningless random target features.

### Byte-Pair Visual Encoding (2025)

Source: [Unified Multimodal Understanding via Byte-Pair Visual
Encoding](https://arxiv.org/abs/2506.23639)

- Visual structure can be represented directly in a unified token stream without
  a separate frozen vision encoder.
- Serialization alone does not confer vision competence. Structural tokenization,
  curriculum, and multimodal examples are still required to make those rows and
  shared transformer paths useful.

### Merging Feed-Forward Sublayers (2025)

Source: [Merging Feed-Forward Sublayers for Compressed
Transformers](https://arxiv.org/abs/2501.06126)

- Adjacent feed-forward sublayers are activation-aligned by neuron permutation,
  merged, and then tied. The execution depth remains intact while the number of
  unique stored parameters falls.
- The method reports that merging one-third of feed-forward sublayers is close to
  the original pre-recovery behavior and is consistently stronger than unaligned
  averaging. It does not establish extreme compression or full-block tying.
- This separates two meanings of depth compression: reducing executed blocks and
  reducing unique parameter sets. Neural Forge must report both rather than
  treating them as interchangeable.

## Decisions

1. Replace permissive suffix matching with explicit architecture layouts. Every
   target role must be classified and handled, or the operation aborts.
2. Align residual spaces once per boundary and reuse the same maps for all
   producers and consumers at that boundary.
3. Treat normalization, attention topology, MLP activation/gating, positional
   encoding, embeddings, and the language head as operators in the transaction.
4. Preserve tokenizer bytes, including leading spaces. Ambiguous decoded-token
   collisions remain explicit and cannot silently select the first token.
5. For blank targets, derive target coordinates from parent activation subspaces
   and materialize a complete target computation. Do not estimate alignment from
   random child activations.
6. Gate on complete operator coverage, held-out path/logit reconstruction, cached
   parent behavior benchmarks, and manual generations. Perplexity remains one
   signal, not the objective.
7. Keep raw modality framing shared and architecture-native. Claim capability
   only after a parent modality path has been transported and validated on held-out
   image/audio/video/OCR/3D tasks.
8. Add activation-aligned parameter sharing as a depth-compaction axis. Tied
   operators preserve the number of computation steps and offer a direct expansion
   inverse by materializing independent copies. Report unique parameters, dense
   equivalent parameters, storage, and inference compute separately.

## First Proof Slice

The first bounded proof is GPT-2-family to Llama-family and the reverse, using
tiny independently initialized targets. It must provide:

- 100% required-role coverage with no random required operator;
- contiguous, non-overlapping depth composition;
- deterministic save/reload;
- held-out block and next-token-logit error below the blank target in both
  directions;
- generated text that is manually coherent before any recovery training.

This slice intentionally excludes LFM convolution/state-space operators, Gemma
extra normalization schedules, multi-parent arbitration, and modality capability
transport until their operator adapters have equally explicit contracts.
