# Actual native audio recovery preparation (2026-10-13)

`train_audio.py` owns `D3d_audio_ce` after vision D3d and before D4. It teaches
audio observation to assistant text with real clean H3 audio latents and authored
answers. It does not teach speech output. The new chart is an explicit standalone
observation convention, not measured parity with the parent's video grid.

Actual ingress evidence: all24 procedural source rows pass the real Qwen parent
tokenizer insertion and assistant-EOS ownership inspection. Actual H3 CPU proofs
pass full1s/40frame and full4.175s/167frame input, raw/model space and bitwise
repeat encoding. The latter real decoded waveform also retains the known boat
sentence under independent ASR. These are ingress checks, not Chimera responses.

The source-admission producer `audio_admission.py` separately traverses frozen
eval, normalized original registry, prior media admission/exclusions and every
fixture. The actual24-row snapshot admits eight training rows and retains eight
validation/eight test rows outside optimizer data. The native preparer checks
all its source-file pins and exact waveform/answer/row/group joins.

Use an isolated genuine Comfy CPU container with `CUDA_VISIBLE_DEVICES=''` and
no GPU exposure. Mount source assets and parent weights read-only; write only to
the private preparation output. The source and preflight copies must retain
their exact original bytes. The original pinned StageC tool and Comfy tree must
match `h3_audio_probe.py`'s recorded hashes.

```bash
python experiments/chimera_recovery/audio_cache.py encode-source \
  --source /work/audio-cache-inputs-v1/source.jsonl \
  --source-preflight /work/audio-cache-inputs-v1/source-preflight.json \
  --source-preflight-sha256 26a0e830d863afe90fcd426242a5bdec65e4b481944c6b78af4573d609251f95 \
  --asset-root /work/audio-cache-inputs-v1/assets \
  --stage-c /work/audio-cache-inputs-v1/stage_c_media_a.py \
  --comfy-root /opt/ComfyUI \
  --audio-vae /opt/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --out /work/clean-audio-cache-v1
```

This initial command handles the authenticated procedural preflight. It visits
all24 rows and uses the exact full CPU encode/decode proof for each. A failure
preserves diagnostic partial files without a seal. A successful output still
grants no optimizer admission.

For the next command use the actual native parent tokenizer, the actual pinned
native `charted.py`, the newly appended preregistration amendment digest and the
globally screened source-admission snapshot. If transferring that snapshot to a
different machine, preserve original receipts and emit an explicit derived path
binding rather than editing original manifests. Every referenced source file and
waveform must resolve with the same original SHA.

```bash
python experiments/chimera_recovery/audio_cache.py prepare \
  --source /work/audio-cache-inputs-v1/source.jsonl \
  --clean-manifest /work/clean-audio-cache-v1/manifest.json \
  --clean-manifest-sha256 ACTUAL_SEALED_CLEAN_MANIFEST_SHA \
  --admission /work/audio-admission-derived-v1/manifest.json \
  --admission-sha256 ACTUAL_SOURCE_ADMISSION_SHA \
  --tokenizer-dir /model \
  --charted-attention /work/chimera-omni/src/chimera_omni/nn/charted.py \
  --amendment-sha256 ACTUAL_APPENDED_AMENDMENT_SHA \
  --out /work/native-audio-batches-v1
python experiments/chimera_recovery/train_audio.py preflight \
  --config /work/audio-stage-config-v1.json --out /work/audio-preflight-v1.json
```

Preflight loads no model weights. It requires exact immutable native base/parity,
completed preceding D3d shared-r16 and cumulative native state, explicit exclusive
runtime allocation and the dated objective/read/chart/insertion selection. The
production entry point `train --config ... --out NEW_RUN --resume OPTIONAL_EXACT`
consumes one accepted row per successful update. Same-stage checkpoint resume
restores Adam/RNG/sampler and the exact accepted held free-answer review. A stage
transition copies parameter state before optimizer creation and starts new Adam.
Every batch is independently reconstructed from its source encoder tensors and
actual dialogue before update.

Genuine speech sources need additional authenticated utterance/speaker grouping,
independent ASR checks, global exclusions and original/derived PCM provenance.
Do not admit diagnostic probe row003: its17-times6 sentence overlaps frozen S8.
If full source duration does not divide the actual25ms lattice, preserve every
original PCM sample and append only the explicitly recorded minimal zero tail
to that lattice in a separately sealed derived WAV. The encoder still receives
the entire derived source without internal crop or corrective padding. The real
167-frame proof establishes this measured scope; no40-frame assumption applies.

Speech output requires its separate denoising/flow objective and source-sealed
generation prompt contract. Native clean observations and assistant CE do not
authorize a claim that the checkpoint can speak. Qwen/Fish speech codec IDs never
enter Chimera's music band or H3's32-channel continuous bridge.
