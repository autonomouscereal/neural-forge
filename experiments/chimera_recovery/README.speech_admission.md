# Recorded speech preparation, 2026-10-13

`speech_admission.py` prepares the actual pinned Qwen speech campaign after
both TTS and independent ASR finish. It verifies the complete planned-ID,
text, official-voice, group and asset joins, original and minimally padded
PCM bytes, recorded amplitude, generation cap and declared geometry. It
recomputes the campaign's exact ASCII case/punctuation normalization. It
never repairs a mismatching transcript into an accepted target. Those rows
remain in the inventory with no supervised messages and in a separate
quarantine file. Same-utterance voice variants and >50% shared four-gram
Jaccard utterances share connected identities. Original waveform hashes
also link globally across any explicit minimal-tail derivation.

The download receipt is checked against the source-audited exact official
ASR and TTS revisions and main weight/config/12Hz codec hashes. The actual
download receipt passed this check. The actual completed four-voice canary
was deliberately refused as admission evidence for the 192-row plan.
The full campaign remained incomplete at implementation time; no speech
training admission or native tensor readiness is claimed by that check.

Executable preparation requires the existing global audio pins file plus
a separate `{name:{path,sha256}}` JSON for `speech_plan`, `tts_receipt`,
`asr_receipt`, `teacher_download` and reviewed `teacher_writer`. These are
the actual BF16 campaign `tts-192.json`/`asr-192.json`, its original plan,
download completion and captured writer. Point `speech-asset-root` at the
copied or host-mounted **assets directory**; original receipt paths under
`/work/speech-campaign/assets/ID/` must map exactly to that directory's
`ID/original.wav` and `ID/h3-aligned.wav`. No arbitrary path is followed.

```powershell
python -m experiments.chimera_recovery.speech_admission `
  --global-pins 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_audio_admission_pins_oct13_v1.json' `
  --speech-pins 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_speech_admission_pins_oct13_v1.json' `
  --procedural-asset-root 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_media_seed_oct13_v2/assets' `
  --speech-asset-root 'C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/recovery-preparation/speech-gpu-campaign-bf16-v2/assets' `
  --output 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_speech_source_admission_oct13_v1'
```

Preparation emits unchanged-admission-input `source_ready.jsonl`, full
`source_inventory.jsonl`, unsupervised `quarantine.jsonl`, a pinned
`preparation.json` and `global_admission/manifest.json` using the same
`recovery_audio_admission.v1` reader contract as procedural audio. The
global seal also traverses all prior procedural source families and
existing registry/frozen/prior media exclusions. Source route remains
`speech_understand`; an explicit native dialogue adapter can map it to
`audio_understand` while retaining original source record hash and route.
Native clean H3 encodings, answer-safe attention masks, chart amendment,
native batches and driver/decoder validation remain required.

`h3_audio_fixture.py` packages a real source/probe/safetensors join without
Torch or teacher execution. It checks actual source PCM and file hashes,
the reviewed Stage-C/Comfy/VAE pins, measured full encoder coverage, tensor
file hash and all six F32 tensor shapes/byte spans. Optional global audio
admission joins by the exact input asset hash. Optional tail derivation
joins the preserved original source. It emits diagnostic evidence only;
the native packer independently validates actual tensor values and the
complete source inventory. One admitted tone diagnostic is not a sealed
24-row clean cache, and the old CPU speech diagnostic stays unadmitted.
