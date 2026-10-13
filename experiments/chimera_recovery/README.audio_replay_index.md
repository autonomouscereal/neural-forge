# Actual audio and speaking replay index

`audio_replay_index.py` audits sealed native caches using their exact original
producer source directories. It reconstructs every clean encoder proof and every
CE or speaking tensor batch. It verifies the full joint source admission again,
including held diagnostics, and refuses speaking rows newly excluded by that
joint proof. An earlier speech group ID is preserved beside the current union
group ID; it never substitutes group names for the exact source-row join.

```bash
CUDA_VISIBLE_DEVICES='' python audio_replay_index.py \
  --ce-manifest /joint/native-audio-ce-v1/manifest.json \
  --ce-sha256 ACTUAL_CE_MANIFEST_SHA \
  --ce-source-dir /joint/source/experiments/chimera_recovery \
  --flow-manifest /work/native-speech-flow-v1/manifest.json \
  --flow-sha256 ACTUAL_FLOW_MANIFEST_SHA \
  --flow-source-dir /work/source/experiments/chimera_recovery \
  --amendment-sha256 ACTUAL_SELECTED_SPEECH_AMENDMENT_SHA \
  --out /audit/replay-index.json
```

Output schema is `chimera_recovery.audio_speech_replay_index.v1`, with
`sealed`, `passed`, `verified`, and `complete_source_traversal` all true only
after all reconstructions pass. Source manifest pins are absolute paths in the
original preparation mount namespace; batch pins retain their original bytes.
Any relative batch path resolves against its native source manifest parent.
Each entry
preserves the original batch ID, receipt/tensor SHA, storage contract, objective,
source manifest/admission, source row SHA, producer and union group IDs, prompt
length, native row count, latent frames per channel, and amendment. The CE
denominator is the exact shifted assistant target count including its completion
EOS. The speaking denominator is the actual weighted native audio row count
times32 emission channels. Both are measured from independently verified tensors.

CE covers every admitted source once; speaking covers every admitted speech
source/sigma once. Duplicate, dropped, held and newly excluded identities fail
the audit. Per-objective totals come from those measurements. There is no
model-weight load, forward pass, teacher call, optimizer or GPU operation. The
index remains `production_training_ready:false`: accepted preceding stage state,
current-model full-depth parity, exclusive runtime and held free-answer/decode
reviews remain separate requirements for training or consolidation.
