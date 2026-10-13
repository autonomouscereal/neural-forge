### Candidate amendment 2026-10-13: waveform-supervised speaking recovery

This is a reviewable candidate selection. It is not an appended or sealed
production preregistration. The unchanged stage-D frozen body remains
15339 bytes with SHA c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370
as verified by the existing stage-D receipt tool; append the selected wording
under its Amendments heading and bind the resulting actual amendment digest
before a weight-changing step. The existing tool verifies the body directly;
do not replace its source pin with this candidate document.

The added stage is `D3d_speech_flow`, after `D3d_audio_ce` and before `D4`.
Understanding uses authored assistant CE separately. Speaking uses
`supervised_speech_rectified_flow` from genuine waveform latents, rather than
guided H3 teacher-trajectory KD or foreign discrete speech codec IDs.

Source rows remain the unchanged `speech_understand` records from a complete
sealed speech source/admission snapshot. Each accepted target must preserve
the exact official synthetic voice (Ryan, Aiden, Vivian or Serena), full authored
transcript, actual independent exact ASR match, original TTS waveform SHA,
minimal explicitly recorded PCM zero tail, full derived waveform SHA and all
source/plan/teacher/ASR provenance. No user reference voice is used. Exact ASR
does not stand in for human listening; that limitation remains in receipts.
Diagnostic probe row003 (17 times6) is excluded for frozen S8 overlap. The
complete combined training/held source, frozen registry/eval/prior exclusions
and connected speaker/utterance groups must be traversed by the admission
producer. Held rows are encoded only for diagnostics and never admitted.

The actual H3 encoder, model-space transform and source pins from
`h3_audio_probe.py` apply. Every source requires its own full-coverage measured
proof:32kHz stereo,800 samples per nominal frame, raw/model latent
[1,32,2,T], channel-major packed[2T,32], no internal trim/pad, bitwise repeat
encode and verified raw/model/inverse transform. A non-default T requires its
own actual proof; no40-frame assumption is used for speech.

The condition is exactly
`Generate speech in the synthetic {speaker} voice, saying exactly:\n{transcript}`
as one ordinary user message in the pinned parent chat template with
enable_thinking=false and an authentic assistant generation prefix. Its full
rendered/native token IDs must agree. Append noisy type3 placeholder rows after
that complete prefix, using
`native_noisy_audio_after_assistant_generation_prefix_v1`. There is no reference
assistant answer, no codec ID and no clean-latent copy in the model inputs.
The transcript is a legitimate TTS condition and is pinned as such.

Select `audio40hz_generation_prompt_offset_categorical_stereo_v1` and
`h3_generation_prompt_and_noisy_audio_only`. The measured nominal physical
bridge chart is[(i+.5)*.025,0,.025,c]; the separately selected rotary chart is
[prompt_text_count+i,0,c], where c is categorical stereo0/1. Text positions are
[text_ordinal,0,0]. H3 spec types remain[2,3], actual rows are only type3, and
the bounded H3 read policy includes just the generation prefix and noisy audio.
This is an explicit new generation chart derived from proved source-lattice
mathematics; native parent XYZ parity is false and must not be claimed.

Select `speech_sigma_grid=[0.05,0.3,0.7,1.0]`, seed20261013 and the deterministic
`source_row_sigma_sha256_first63bits_v1` noise policy. For each admitted source
and each sigma, sample FP32 epsilon from the sealed seed and form
x_t=(1-sigma)*x0+sigma*epsilon. Only x_t enters the native feature band480:512;
only x0 supervises emission band224:256. Target/noise tensors are detached,
finite and source-reconstructed bitwise. The student type3 emission is explicitly
interpreted as an x0 endpoint for this new supervised objective. Canonical
velocity=(x_t-x0_prediction)/sigma is matched by per-channel MSE to
(x_t-x0_target)/sigma=epsilon-x0_target. Weight1, per-row/channel mean, full-depth
gradient propagation through the shared text trunk. Positive sigma is mandatory.
The sigma weighting is intentional; this is not a declaration that the initial
S8 head already has these endpoint semantics or has learned to speak.

Train the H3 unit, type3 bridge/input embedding slices, type3 emission slices
and shared all-layer r16/alpha16 bank. Freeze text head, other native units,
other bridge/emission bands, music tables, K3 sites and gain. FP32 trainable
masters; unit/bridge/emission learning rates2e-5, shared-r16 learning rate1e-4;
AdamW beta(.9,.95),eps1e-8,weight decay0,clip1,warmup20 successful steps,
cosine minimum ratio.1,seed20261013. Inherit the preceding complete shared bank
as a replacement and complete cumulative native state before optimizer census;
use fresh Adam at stage transition. Exact same-stage resume restores all
trainable masters, Adam,RNG,sampler and accepted review evidence.

The initial budget is one successful update per admitted source/sigma pair,
with the exact manifest/count pinned. Nonfinite skips do not advance the
sampler. Keep the existing eight consecutive /20% of100 skip halt and30-minute
rolling checkpoints,keep2. At25% budget, stop spending until held target-free
sampling has been reviewed with the actual H3 VAE decoder, independent ASR
and explicit intelligibility/content observations. Select
`speech_euler_steps=32`: Gaussian initialization at sigma1,32 equal descending
Euler steps to0,velocity=(x_t-predicted_endpoint)/sigma. No clean target is
supplied to the sampler; physical duration/chart and transcript/voice remain
fixed. Noise/ASR/voice quality are reported, not replaced with teacher-forced
loss. No standalone waveform quality threshold is invented.

D4 keeps its original3M Q38/1M K3/300pairs per original media lane. Add one
replay per accepted source row from the source-admitted audio/speech CE cache,
and one replay per accepted source/sigma speaking pair from the speaking cache,
with exact cache manifests, counts and source-group exclusions pinned. These
are separate explicit replay routes with the same objective definitions and
existing lr multiplier0.3; held diagnostics contribute zero updates. This
addition changes replay scope only through this dated amendment. The D4 runtime
must support these routes before it can claim coverage.

Production additionally requires actual current-base full-depth/read-policy
gradient parity, immutable source identity, complete preceding cumulative
states, sealed full source caches and an exclusive allocator runtime. This
candidate, tiny fixtures and successful parent ingress do not grant those proofs.
