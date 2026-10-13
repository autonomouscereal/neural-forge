# Combined native audio and speech preparation

`audio_sources.py` preserves the original procedural and speech source records,
their complete inventories and prior exclusions. `prepare-source` runs the pinned
global admission helper over their combined inventory. It verifies the original
procedural ground truth and every source asset, and independently rechecks the
speech waveform's exact original PCM prefix and explicitly recorded minimal zero
tail. Source admission does not grant native training readiness.

```powershell
python experiments/chimera_recovery/audio_sources.py prepare-source `
  --procedural-admission PROCEDURAL_MANIFEST --procedural-sha256 PROCEDURAL_SHA `
  --speech-admission SPEECH_MANIFEST --speech-sha256 SPEECH_SHA `
  --admission-helper AUDIO_ADMISSION_SOURCE --helper-sha256 HELPER_SHA `
  --procedural-asset-root ORIGINAL_ASSET_ROOT --out FRESH_JOINT_SOURCE
```

The output includes unchanged `source_ready.jsonl`, all original rows in
`source_inventory.jsonl`, a standard `recovery_audio_admission.v1` global
admission, and `chimera_recovery.joint_audio_source_preparation.v1` preparation
receipt. Every source row's canonical digest stays unchanged. Source groups are
recomputed across both campaigns and the original registry, evaluation and prior
exclusions. A prior blocked row cannot become eligible through composition.

`compose-cache` accepts a JSON list of exact `{path, sha256}` pins to complete
`chimera_recovery.clean_audio_cache.v1` manifests. It requires every retained
joint source row exactly once, compares unchanged source records, then verifies
each original genuine H3 encoder receipt and its actual finite waveform, latent,
model-space and decoded tensor bytes. It preserves each original proof pin and
runtime adapter identity. It produces a new clean manifest without rewriting any
proof or fabricating a new encoder execution.

```powershell
python experiments/chimera_recovery/audio_sources.py compose-cache `
  --source FRESH_JOINT_SOURCE/source_ready.jsonl `
  --admission FRESH_JOINT_SOURCE/global_admission/manifest.json `
  --admission-sha256 JOINT_ADMISSION_SHA --clean-manifests CLEAN_PINS_JSON `
  --out FRESH_JOINT_CLEAN_CACHE
```

Run `audio_cache.py prepare` against that new joint source, admission and clean
manifest, with the selected dated amendment. It prepares a single
`D3d_audio_ce` dataset: native clean H3 audio observations, authentic parent chat
template, assistant body and EOS targets, answer-independent observation chart,
and no teacher logits. Holdout rows retain genuine encoder proofs but cannot enter
optimizer batches. `train_audio.py` remains responsible for actual model, stage
inheritance, amendment, ownership and checkpoint preflight.

For the held-answer review, `audio_sampling.sample_answer(engine, segment,
prompt_length=..., completion_eos_id=..., max_new_tokens=256)` runs the actual
full-depth model from clean audio and the native generation prefix. It discards
all reference-answer IDs and coordinates before model access, extends the
explicitly amended text ordinal chart, and keeps H3 joint reads restricted to the
original prompt and observations. It returns generated native token IDs and EOS
status with input, engine and sampler identities. Decode with the pinned parent
tokenizer and grade against the independently retained held answer afterward.
The sampler does not grant the production driver's review pass. Its actual tiny
native CPU tests prove reference-tail invariance, no parameter updates, EOS and
budget behavior, nonfinite refusal, and restoration of the bounded read policy.

Keep speaking separate. `speech_flow.py prepare` consumes the original admitted
speech-only source and its complete genuine encoder manifest. Its
`D3d_speech_flow` targets are supervised rectified flow from speech waveforms in
H3's native 32-channel space, with four sealed sigmas and target-free text
conditioning. This is distinct from understanding CE and from guided H3 teacher
velocity distillation. Foreign speech codec indices never enter native training.

The actual October 13 joint source audit retained 225 candidates from 352 full
inventory rows: 8 procedural and 186 speech rows admitted for training, with 31
holdout rows excluded. The 201-wave genuine speech encoder must finish and seal
before the joint native cache can be produced. These counts describe preparation,
not current Chimera model quality or a completed training run.
