# Validation Results - 2026-06-29

## Qwen 4B Donor Into Blank 135M Target

Run root on the AI server:

`/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627`

Compression target:

- Donor: Qwen 4B family checkpoint.
- Target: blank SmolLM-shaped causal LM.
- Final target size: 134,515,008 parameters.
- Approximate compression ratio: 29.7x from 4B to 135M.

Best text-only checkpoint:

`qwen4b_mathboost_answer_cleanopen_800_gpu4`

Standard broad eval slice:

- Generation health: 200/200.
- Behavior probes: 161/200.
- Choice probes: 40/64.
- Mean PPL: 37,663.40.
- Behavior breakdown: math 1/40, tokenizer 40/40, commonsense 40/40, model surgery 40/40, code 40/40.

Manual inspection showed clean useful text for pruning, merge risk, tokenizer mismatch, rainy-road advice, neuron merging definition, and small Python functions. Arithmetic remained unreliable.

## Recovery Progression

The earlier recovered model had clean generation but failed behavior probes. Multi-GPU recovery improved actual behavior:

- `qwen4b_promptformat_stopcleanup_600s_gpu4_20260629_020535`: generation 200/200, behavior 0/200, choice 22/64.
- `qwen4b_contract_heavy_full_1100b_gpu4`: generation 200/200, behavior 161/200, choice 36/64.
- `qwen4b_mathboost_answer_2400_gpu0`: generation 200/200, behavior 160/200, choice 43/64.
- `qwen4b_mathboost_answer_cleanopen_800_gpu4`: generation 200/200, behavior 161/200, choice 40/64.

The math-heavy passes improved choice slightly but did not recover robust generated arithmetic. This is the main unresolved capability gap in the compressed blank target.

## Arithmetic Recovery Follow-Up

Additional multi-GPU sweeps tested answer-contrastive recovery, Qwen-donor ranked answer transfer, stepwise donor correction, and sharper numeric drill recovery.

Teacherless answer-contrastive results:

- `qwen4b_contrast_math_headnorm_2200_gpu0`: generation 138/200, behavior 160/200, choice 45/64, math 0/40.
- `qwen4b_contrast_math_last4_2000_gpu4`: generation 183/200, behavior 161/200, choice 40/64, math 1/40.
- `qwen4b_contrast_math_all_best_1800_gpu6`: generation 200/200, behavior 161/200, choice 43/64, math 1/40.

Donor-aware answer ranking:

- `qwen4b_donor_rank_numeric_last4_700_gpu4`: generation 179/200, behavior 161/200, choice 38/64, math 1/40.

Best narrow positive result:

- `qwen4b_mathdrill_last8_lr1e5_contrast_150s_gpu6`: generation 200/200, behavior 162/200, choice 41/64, math 2/40, PPL 37,366.27.

Manual inspection of the best narrow result still showed obvious arithmetic failures: `6 + 4 -> 30`, `41 * 30 -> 700`, and `35 + 27 -> 75`. The same checkpoint retained clean neuron-merging, tokenizer-mismatch, rainy-road, and Python `add` outputs. This means the current approach can nudge held-out numeric probes without destroying recovered text behavior, but it has not recovered general arithmetic reasoning.

Implementation changes from this sweep:

- `hf-lm-recover` now exposes answer-contrastive recovery with `--answer-contrastive-weight` and `--answer-contrastive-max-negatives`.
- Generation-heavy cross-tokenizer recovery helpers now temporarily use left padding during decoder-only `.generate()` calls and restore tokenizer state afterward.

## Blockwise Folding Follow-Up

Added two blockwise recovery controls:

- `--hidden-projection-scope layer-input`: refits donor-hidden to child-hidden
  projections from each active block's paired input activations instead of using
  only the global shared-token embedding map.
- `--layer-checkpoint-interval`: saves resumable/evaluable checkpoints after
  every N completed folded target layers.

Layer-input projection avoided the catastrophic global-projection failures but
did not recover arithmetic:

| Run | Start | Generation | Behavior | Choice | Mean PPL | Behavior breakdown |
|---|---|---:|---:|---:|---:|---|
| `qwen4b_block_layerproj_delta_passgate_s6_gpu0_20260629` | best 135M child | 195/200 | 161/200 | 38/64 | 38,350.52 | math 1/40; all other categories 40/40 |
| `qwen4b_block_layerproj_hidden_passgate_s6_gpu4_20260629` | best 135M child | 200/200 | 161/200 | 39/64 | 40,204.89 | math 1/40; all other categories 40/40 |
| `qwen4b_block_layerproj_conservative_passgate_s4_gpu6_20260629` | best 135M child | 200/200 | 161/200 | 42/64 | 29,254.60 | math 1/40; all other categories 40/40 |

Manual inspection of the conservative layer-input run still failed arithmetic:
`6 + 4 -> 40`, `41 * 30 -> 1258`, `35 + 27 -> 85`. It retained the Python
`add` function but degraded some explanatory text compared with the donor.

Added response-focused blockwise folding controls:

- `--response-lm-weight`
- `--first-token-lm-weight`
- `--response-end-lm-weight`
- `--response-loss-mode`

Validation showed that direct response LM is dangerous when it mutates tied
output embeddings:

| Run | Start | Head/embedding trained | Generation | Behavior | Choice | Mean PPL | Read |
|---|---|---:|---:|---:|---:|---:|---|
| `qwen4b_block_resplm_from_best_s8_gpu0_20260629` | best 135M child | yes | 13/200 | 54/200 | 39/64 | 26,816.79 | PPL improved while behavior collapsed. |
| `qwen4b_block_resplm_from_random_s8_gpu4_20260629` | pure random SmolLM blank | yes | 0/200 | 0/200 | 22/64 | 79,190.30 | Direct response CE did not plant capability into a pure random target. |
| `qwen4b_block_resplm_from_donorproj_s8_gpu6_20260629` | donor-projected blank init | yes | 0/200 | 0/200 | 21/64 | 32,776.90 | Donor tensor init plus tied-head response CE still collapsed. |
| `qwen4b_block_resplm_nohead_from_best_s8_gpu0_20260629` | best 135M child | no | 200/200 | 161/200 | 42/64 | 29,817.96 | Preserved the conservative profile but did not improve math. |

Manual inspection of `qwen4b_block_resplm_nohead_from_best_s8_gpu0_20260629`
showed `6 + 4 -> 40`, `41 * 30 -> 1258`, `35 + 27 -> 85`; Python `add` was
correct, but tokenizer/neuron-merge explanations were worse than the baseline.

Conclusion: response-token CE is useful instrumentation, but the tested
schedules are not the compression solution. Tied-head response training lowers
PPL while destroying behavior. No-head response training can preserve the
current model but does not transfer donor arithmetic. The next folding work
should focus on staged representation/interface geometry rather than stronger
answer CE on top of a weak fold.

## Raw-Patch Vision Graft

Implementation change:

- Added optional text-retention loss to `hf-vision-graft`.
- New CLI flags:
  - `--text-retention-file`
  - `--text-retention-weight`
  - `--text-retention-batch-size`
  - `--text-retention-max-length`

The vision experiment uses raw 16x16 image patches projected directly into the causal LM embedding stream before text tokens. No external frozen vision encoder is used.

Vision results on the compressed text checkpoint:

- `qwen4b_cleanopen_rawpatch_graft_frozen_900_gpu0`: frozen LM, vision 108/192, text model unchanged.
- `qwen4b_cleanopen_rawpatch_graft_last2_700_gpu4`: vision 192/192, but text degraded sharply.
- `qwen4b_cleanopen_rawpatch_graft_last4_600_gpu6`: vision 192/192, but text degraded sharply.
- `qwen4b_cleanopen_rawpatch_graft_last2_ret05_700_gpu0`: vision 192/192, retained text behavior 133/200.
- `qwen4b_cleanopen_rawpatch_graft_last2_ret10_700_gpu4`: vision 192/192, retained text behavior 120/200.
- `qwen4b_cleanopen_rawpatch_graft_last4_ret10_600_gpu6`: vision 192/192, retained text behavior 147/200.

Best integrated vision/text compromise:

`qwen4b_cleanopen_rawpatch_graft_last4_ret10_600_gpu6`

- Vision: 192/192.
- Text generation health: 117/200.
- Text behavior probes: 147/200.
- Text choice probes: 35/64.

The retained vision run proves raw visual patch prefixes can be grafted onto the compressed blank-target model, but the current retention method still trades away too much general generation quality. The safest production split is currently the clean text-only checkpoint plus the frozen raw-patch graft; the strongest integrated checkpoint is the retained last-4 variant.

## Hardware Use

The run used the available CUDA-capable GPUs in parallel:

- RTX 3090 `GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561`
- RTX 3090 `GPU-2ce6a95c-7b1f-644a-3808-d781732a0582`
- Quadro RTX 8000 `GPU-f5a09535-fc85-a027-50f7-734b344747dd`

The Tesla P40 cards were not used for these PyTorch jobs because the installed PyTorch build does not support their compute capability.

## Current Read

This is not the full end goal yet. The compression path now works well enough to produce useful text and code behavior from a 4B-to-135M blank target, and the raw vision graft can reach perfect accuracy on the synthetic image task. Remaining work is to recover arithmetic/reasoning more robustly and reduce text degradation during integrated vision training.
