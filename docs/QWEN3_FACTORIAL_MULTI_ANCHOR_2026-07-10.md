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

## Artifact locations

- Attention-only run:
  `/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710`
- MLP-only run:
  `/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710`
- Multi-anchor probe:
  `/mnt/ai_data/neural-forge/qwen3_4b_multi_anchor_probe_20260710/layer18_drop64_anchor4_gpu4_r1/report.json`
- Extracted cached-parent manual rows:
  `/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/parent_manual_output_extract_20260710.json`

The MLP `p34` / attention `p36` control remains in progress and is intentionally
not summarized before its full evaluation completes.
