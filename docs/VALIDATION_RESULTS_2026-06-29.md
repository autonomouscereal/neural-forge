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
