# Speaking recovery preparation

`speech_flow.py` derives a separate speaking task from the unchanged globally
admitted speech transcript/voice source. `train_speech_flow.py` provides the
full-depth native engine, executable preflight/train, target-free32-step Euler
sampler, exact Adam/RNG/sampler resume and cumulative state export. These are
waveform-supervised rectified flow operations, not H3 guided-teacher KD or
audio-understanding CE. Current speech source completion, actual H3 encoder
proofs, source admission and production parity remain independent requirements.

First finish the combined train/held source producer and its global admission.
Use `audio_cache.py encode-admitted-source` on that unchanged snapshot in an
isolated genuine Comfy CPU runtime. This visits every retained row, including
held diagnostics, and creates real model-space H3 targets. Official Qwen/Fish
codec indices never enter the current model. Preserve all original PCM; only
the source producer's explicitly proven minimal tail is permitted.

Append the selected dated candidate wording from
`PREREGISTRATION_SPEECH_FLOW_CANDIDATE_2026-10-13.md` to the actual stage-D
preregistration without changing its frozen body. The adapter needs the newly
sealed amendment SHA, authentic parent tokenizer and charted attention source.
The candidate itself grants no production admission.

```bash
python experiments/chimera_recovery/speech_flow.py \
  --source /work/complete-speech-v1/source_ready.jsonl \
  --clean-manifest /work/clean-speech-cache-v1/manifest.json \
  --clean-manifest-sha256 ACTUAL_FULL_CLEAN_MANIFEST_SHA \
  --admission /work/complete-speech-v1/global_admission/manifest.json \
  --admission-sha256 ACTUAL_FULL_GLOBAL_ADMISSION_SHA \
  --tokenizer-dir /model \
  --charted-attention /work/chimera-omni/src/chimera_omni/modeling/charted.py \
  --amendment-sha256 ACTUAL_NEW_SELECTED_AMENDMENT_SHA \
  --out /work/native-speaking-batches-v1
python experiments/chimera_recovery/train_speech_flow.py preflight \
  --config /work/speaking-stage-v1.json --out /work/speaking-preflight-v1.json
```

Each accepted waveform supplies four source-pinned positive-sigma pairs. Only
the seeded noisy latent occupies feature480:512; clean model-space targets
occupy emission224:256. The text condition contains the exact transcript and
official synthetic voice as intended TTS input. The native assistant generation
prefix is preserved; noisy audio rows follow it. There are no reference answer
tokens or text CE targets in this objective. Physical40Hz/stereo observations
support the explicit new generation chart; no native parent XYZ parity is claimed.

Production config schema is `chimera_recovery.speech_flow_stage_config.v1`,
stage`D3d_speech_flow`,objective`supervised_speech_rectified_flow`,read policy
`h3_generation_prompt_and_noisy_audio_only`,after`D3d_audio_ce`,before`D4`,
sampling`one_pass_row_sigma`,seed20261013. Pin preregistration,immutable base,
hybrid module, actual current-base speaking full-depth/read-policy parity,
exclusive allocator assignment,complete batches,shared-r16 and cumulative native
state. LoRA is existing r16/alpha16/noK3/no gain. Steps must equal accepted
row/sigma inventory. Optimizer settings are exposed by `OPTIMIZER`; safety is
the existing8consecutive/20%of100 skip halt and30-minute rolling checkpoints.

`train --config ... --out NEW --resume OPTIONAL_EXACT_CHECKPOINT` never creates
a teacher or allocation. It inherits the completed preceding bank/native state
before creating its fresh optimizer. Each batch is independently reconstructed
from unchanged source, exact encoder tensors, deterministic noise, native
prefix and chart. At25% of budget it stops until the stable configured
`speech_decode_review` path has actual held target-free sampler/VAE/ASR review
under the exact run identity. Accepted evidence is pinned with every checkpoint.

`sample_latents(engine,segment,prompt_length=...,seed=...)` has no target
argument. It discards every supplied feature value, initializes Gaussian noise,
then follows the selected32-step endpoint-to-velocity Euler path. It returns
model-space native latents plus trajectory hashes. Actual H3 VAE waveform decode
and independent ASR/listening remain separate required review operations. A
finite latent or low teacher-forced loss does not show intelligible speech.

Tests include generation role/source identity, voice/ASR refusals, positive-sigma
input-target separation, sigma1 target independence, actual tiny full-depth
gradient parity, all four optimizer groups, exact interrupted resume, and
target-free sampler feature invariance. Tiny fixtures make no production-model
quality claim. The actual full source cache cannot be prepared from the earlier
four-row diagnostic probe; its arithmetic row003 overlaps frozen S8.
