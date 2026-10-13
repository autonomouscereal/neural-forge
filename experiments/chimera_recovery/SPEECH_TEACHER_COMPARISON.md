# Separate selected 1.7B speech comparison

The user's original Dot reference selected Qwen3-TTS-12Hz-1.7B-CustomVoice; its
paired speech selection report also names Qwen3-ASR-1.7B. The completed current
synthetic teacher campaign used the separately pinned 0.6B variants. Keep both
variant histories. The 1.7B candidate is a comparison and does not replace any
sealed 0.6B waveform, admission, native cache or result.

`speech_teacher_comparison.py prepare` validates the complete source inventory,
exact ready-source and global-admission joins, original splits, official speaker
identities, authored transcripts, full PCM bytes and minimal silence tails.
Every comparison row has `optimizer_admitted:false`. The current actual plan
preserves 201 sources: 186 train references, 8 validation and 7 test references;
seven prior ASR disagreements remain outside the ready-source set. It plans one
ASR comparison per identical baseline waveform and a deterministic eight-row
serial TTS canary before any larger comparison. These commands do not run
inference:

```text
python -m experiments.chimera_recovery.speech_teacher_comparison prepare --spec SPEC.json --output NEW_COMPARISON_DIR
python -m experiments.chimera_recovery.speech_teacher_comparison download-candidates --plan NEW_COMPARISON_DIR/manifest.json --output NEW_PRIVATE_MODEL_DIR
python -m experiments.chimera_recovery.speech_teacher_comparison verify-download --plan NEW_COMPARISON_DIR/manifest.json --kind asr --model-dir NEW_PRIVATE_MODEL_DIR/Qwen3-ASR-1.7B
python -m experiments.chimera_recovery.speech_teacher_comparison verify-download --plan NEW_COMPARISON_DIR/manifest.json --kind tts --model-dir NEW_PRIVATE_MODEL_DIR/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

The spec contains `{path,sha256}` pins for `speech_preparation`,
`global_admission`, `baseline_completion`, and `parents.asr`/`parents.tts`
(revision-specific Hugging Face API metadata including upstream file blobs).
Optional availability, runtime-package and native-replay receipts are verified
when supplied. The selected parent revisions are fixed in source. Downloading
requires a new namespace plus 32 GiB of disk headroom after all model files.
Every LFS file is checked by upstream SHA256/size, and every ordinary repository
file by git-blob SHA1/size; the final offline pass also records raw SHA256 for
each actual file. Interrupted or mismatched transfers preserve a failure receipt
and partial file without a complete seal. No model is loaded by this helper.

Use the already validated private `qwen-asr==0.0.6` and `qwen-tts==0.1.1`
runtimes separately; their Transformers requirements differ. Loading the 1.7B
parents and checking actual memory/canary behavior remain distinct prerequisites.
Any future GPU invocation needs its own exclusive runtime allocation. No
comparison result, new waveform or tokenizer code is automatically training
data. New source hashes need fresh global admission and per-wave H3 proofs.

ASR compares exactly the same source and aligned waveforms and durations. Native
CustomVoice generation has no explicit duration argument, so the plan pins the
baseline reference duration and requested native T for each row while requiring
measured duration deltas from complete 1.7B output. A matched-duration TTS lane
remains closed; do not silently trim, stretch or pad output to the baseline
duration. Any minimal native-grid silence tail must retain the complete new PCM
prefix and have its own derivation receipt.

The ASR audio encoder uses bidirectional attention inside its input blocks; its
text decoder is causal. The TTS talker/code predictor is causal. These selected
models provide understanding and speech generation through separate paths; the
source does not establish a single fully bidirectional speech/text decoder or a
compatible Chimera codec-index mapping.
