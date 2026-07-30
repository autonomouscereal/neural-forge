# A12 cached teacher-forced trajectory factorial

This is the authenticated four-lane causal audit of the A11 independent input
and output token routes. It reuses the immutable 16-prompt parent trace cache;
it performs no parent forward, trace recapture, training, backward pass,
optimizer step, pruning, quantization, or global residual-map fit.

The body consumes the full donor-token trajectory. Generation caps and safety
judgments are not part of this diagnostic. Whole-response manual review remains
authoritative for intelligence.

## Result

| Prompt/continuation input | Readout | Parent-winner agreement | Same-run hidden cosine | Same-run hidden relative L2 |
|---|---|---:|---:|---:|
| exact / exact | exact parent table | 99.1781% | 1.0000 | 0.0000 |
| exact / folded | exact parent table | 86.5753% | 0.9547 | 0.2409 |
| folded / exact | exact parent table | 88.4932% | 0.9326 | 0.3120 |
| folded / folded | exact parent table | 80.5479% | 0.8900 | 0.4248 |
| exact / exact | folded C_out | 90.9589% | 1.0000 | 0.0000 |
| exact / folded | folded C_out | 81.9178% | 0.9547 | 0.2409 |
| folded / exact | folded C_out | 84.3836% | 0.9326 | 0.3120 |
| folded / folded | folded C_out | 79.1781% | 0.8900 | 0.4248 |

The exact/exact lane is the same-run numerical control. It shows that the
runtime/cached-parent mismatch is small and cannot explain the folded-lane gap.
Folded prompt inputs create the larger hidden-state distortion; folded
continuation inputs create a second causal loss; the independent folded output
route adds a smaller, separable readout loss. Their combination compounds.

This supports the next implementation step: select `C_in` and `C_out`
independently from position- and context-specific causal damage, then locally
reclose affected routes. It rejects treating one shared support/global map as a
sufficient repair.

## Identity

- Audit file SHA-256: `95972ac8e1155e2baf8d6a70fb0b6c4d22657f6fcec162a78b3f525224c3ede2`
- Semantic audit SHA-256: `36b83e86bfc03e81e36949c3ee71e720ffbef32cc4ba8f860f8b978416020c3e`
- Request cache key: `9d803837179575e74c3f381deaeff19b210317594ddac71d732a26c9987acd82`
- Runtime: Quadro RTX 8000, explicit FP32 fallback from requested BF16
- Prompt count: 16; aligned trajectory states: 365 per lane
- Remote wall interval: 2026-07-30 16:04:54Z to 16:05:50Z

`audit.json` is the authoritative result. `complete.json` authenticates its file
and semantic hashes. `job.log` preserves the append-only execution record.
