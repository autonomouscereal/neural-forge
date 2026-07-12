# Qwen3 factorial and multi-anchor evidence - 2026-07-10

All results below are zero-recovery. No optimizer, backward pass, distillation,
or cleanup training was used. Parent results were cached once and compared by
shared probe ID.

## Parent

- Model: `Qwen/Qwen3-4B-Base`
- Unique parameters: `4,022,468,096`
- Perplexity: `877.579654`
- Behavior: `400/1000`
- Choice: `250/250`
- Generation health: `388/1000`

## Completed factorial controls

| Fold | Child parameters | Source/child | PPL | Parent behavior retained | Choice | Parent generation retained | Promotion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MLP `p35`, attention `p36` | 3,947,073,536 | 1.019101x | 1,581.629490 | 245/400 (61.25%) | 203/250 | 175/388 (45.10%) | Fail |
| MLP `p36`, attention `p35` | 3,995,570,176 | 1.006732x | 1,012.671013 | 400/400 (100%) | 250/250 | 372/388 (95.88%) | Strong boundary; not uniformly safe |

The MLP-only result preserves some exact code and simple math but collapses too
often to repeated digits or incorrect arithmetic. It is not a successful fold.

The attention-only result removes `26,897,920` unique parameters and preserves
all parent-passing behavior and choice probes. The historical run report counted
633 child generation passes under the earlier permissive scorer. Offline
rescoring with the finalized strict scorer counts `546/1000` child passes versus
the parent's `388/1000`; parent-pass retention remains `372/388`. This must not
be read as a blanket gain: some apparent gains are prompt/relevance artifacts
and the child loses 16 parent-passing generations. The same strict rescore gives
the MLP-only child `175/1000` passes and `175/388` parent-pass retention.

## Manual paired inspection

The old factorial runner's manual packet retained prompts and pass flags but
omitted output text. Parent and child outputs were therefore inspected directly
from the exact cached-parent and candidate reports for the attention-only fold.

- `generation_python_add_0044`: both write a correct addition function. The
  child emits clean fenced Python; the parent emits a shorter valid function.
- `generation_math_answer_0054`: both answer `15 * 18` with `270`.
- `generation_model_merge_risk_0141`: the parent emits no continuation; the
  child gives a coherent risk involving conflicting information or bias.
- `generation_pruning_0511`: the parent gives a coherent two-sentence pruning
  explanation; the child collapses to repeated zeros.
- `generation_python_add_0905`: the parent writes valid addition code; the child
  collapses to repeated zeros.
- `generation_rain_safety_0046`: both parent and child collapse to repeated
  numeric text. This is a shared parent failure, not knowledge the fold lost.
- `generation_tokenizer_mismatch_0052`: the parent is blank; the child repeats
  variants of the prompt rather than answering it. The automated gain is not a
  useful qualitative gain.

Conclusion: the attention pathway has a safe-looking first compression boundary,
but exact output retention is not uniform. It is a component to combine with
selective methods, not evidence that arbitrary compression is solved.

## Real-layer sparse multi-anchor probe

Model layer 18 was measured on 256 calibration and 256 disjoint validation
activation rows. Five percent of high-importance neurons were protected. Both
methods physically removed 64 of 9,728 SwiGLU intermediate neurons and absorbed
their contribution into surviving down-projection columns.

| Method | Calibration layer relative L2 | Validation layer relative L2 | Fit time |
| --- | ---: | ---: | ---: |
| Pair compensation | 0.0150951 | 0.0106433 | 8.248 s |
| Sparse multi-anchor compensation | 0.0086342 | 0.0050901 | 8.512 s |

Multi-anchor validation error is `0.47825x` pair error, a `52.18%` reduction,
for essentially the same measured fit time. Physical parameter savings for this
single layer are `491,520` parameters.

This probe is diagnostic only. It proves a better held-out activation fold, not
retained language-model behavior. Promotion requires installing the fold into
the complete model and passing the parent-relative selection and untouched
certification suites plus manual output review.

### 256-neuron stress probe

The same layer and data split were then tested at 256 removed neurons with up
to eight anchors per dropped path. The bounded run performed 233,472 projection
fits and 8,415,232 similarity products.

| Method | Calibration layer relative L2 | Validation layer relative L2 | Fit time |
| --- | ---: | ---: | ---: |
| Pair compensation | 0.0351505 | 0.0248835 | 5.444 s |
| Sparse multi-anchor compensation | 0.0167462 | 0.0100573 | 54.695 s |

Multi-anchor validation error is `0.40418x` pair error, a `59.58%` reduction.
The single-layer physical saving is `1,966,080` parameters. The stronger fit is
about ten times slower at this stress size, so candidate-pool and anchor depth
must remain explicitly bounded.

## Artifact locations

- Attention-only run:
  `/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710`
- MLP-only run:
  `/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710`
- Multi-anchor probe:
  `/mnt/ai_data/neural-forge/qwen3_4b_multi_anchor_probe_20260710/layer18_drop64_anchor4_gpu4_r1/report.json`
- Multi-anchor stress probe:
  `/mnt/ai_data/neural-forge/qwen3_4b_multi_anchor_probe_20260710/layer18_drop256_anchor8_gpu0_r3/report.json`
- Extracted cached-parent manual rows:
  `/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/parent_manual_output_extract_20260710.json`

## Completed p34 MLP boundary and compensated full-model controls

The MLP `p34` / attention `p36` control completed after the earlier report was
written. It has `3,872,362,496` parameters (`1.038763x` compression), PPL
`9,132.495930`, behavior `0/1000`, exact generation `0/1000`, and choice
`82/250`. This confirms that MLP capacity crosses a catastrophic boundary
between `p35` and `p34`; the run is not an in-progress result and must not be
queued again unchanged.

The compensated pair and sparse multi-anchor methods were also installed into
the full Qwen model. All runs used zero recovery and retained `400/400`
parent-passing behavior plus `250/250` choice, but the strict untouched
generation certification rejected every candidate:

| Fold | Parameters saved | Child generation | Parent-pass losses | Result |
| --- | ---: | ---: | ---: | --- |
| Pair, 64 neurons | `491,520` | `385/1000` | `4` | rejected |
| Multi-anchor, 64 neurons | `491,520` | `382/1000` | `7` | rejected |
| Pair, 1 neuron | `7,680` | `386/1000` | `2` | rejected |
| Multi-anchor, 1 neuron | `7,680` | `385/1000` | `2` | rejected |

A later layer-specific pair run found a strictly better one-neuron frontier at
layer 35. It saved `7,680` parameters, retained behavior `400/400` and choice
`250/250`, and scored generation `384/1000` against a bound parent that also
scored `384/1000`. There was one parent-pass loss and one gain, so retained
parent generation was `383/384` (`99.7396%`) with no net pass-count loss. PPL
was `882.565651`. The artifact was still refused publication solely because the
old certification contract allowed zero lost IDs. The matched layer-18 search
rejected all eight attempted proposals before publication.

The drop-1 runs used a separately bound parent report that scored `384/1000`
generation rather than the earlier cached `388/1000`. That parent variance,
despite exact identity binding, means a zero-lost-ID rule is too brittle to act
as the only frontier criterion. Multi-anchor remains a useful reusable local
fit primitive, but its 52-60% layer-error improvement did not beat pair folding
on full-model free-running generation. Future selection must retain the full
parent-relative Pareto record and manual evidence rather than promote by local
error or reject all net-positive candidates on one unstable exact-ID threshold.

The layer-35 result is direct evidence for that change: it is not a target-size
success, but it is Pareto-admissible under the project's approximately 90%
fold-retention objective and should remain available as an uneven-allocation
comparison point.

### Late layer-specific controls

The queued layer-34 and layer-33 pair controls finished after the first audit.
Both passed their adaptive selection gates, removed one MLP path (`7,680`
parameters), retained behavior `400/400` and choice `250/250`, and lost only one
of the bound parent's `384` strict-generation passes (`383/384`, `99.7396%`).
They also gained more strict passes than they lost:

| Layer | Candidate generation | Losses | Gains | Net pass-count change | Publication under old zero-loss rule |
| ---: | ---: | ---: | ---: | ---: | --- |
| 35 | `384/1000` | 1 | 1 | 0 | refused |
| 34 | `385/1000` | 1 | 2 | +1 | refused |
| 33 | `386/1000` | 1 | 3 | +2 | refused |

The layer-34 and layer-33 certification parents both scored `384/1000` on the
bound run. Their candidates remained at behavior `400/1000` and choice
`250/250`. These are three independent adjacent-layer examples showing that a
zero-lost-ID publication rule discards high-retention, net-positive Pareto
steps. They do not prove that the steps compose safely; composition still
requires a refreshed parent-relative gate after each accepted transaction.

The matched sparse multi-anchor layer-35 control rejected all eight candidates
at the small gate. Each retained the small-gate behavior and choice items but
lost the same generation item. That result closes the unchanged layer-35
multi-anchor retry lane; the multi-anchor primitive remains available for
different evidence/candidates, but its better local layer fit is not itself a
reason to rerun those eight proposals.

Artifact directories:

- `/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710/qwen3_4b_pair_drop1_layer34_dev_reused_eval_20260710`
- `/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710/qwen3_4b_pair_drop1_layer33_dev_reused_eval_20260710`
- `/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710/qwen3_4b_multi_anchor_drop1_layer35_dev_reused_eval_20260710`
