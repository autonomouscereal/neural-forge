# Candidate amendment: D3d_audio_ce (2026-10-13)

This is a reviewable additive candidate, not the frozen preregistration document.
Preserve the D1–D4 body digest
`c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370`
and its newline-inclusive15,339 bytes.
The root integration must append the accepted exact wording and pin its new
document and amendment digests before any production training.

After vision/video D3d and before D4, introduce **D3d_audio_ce**, objective
`authored_audio_assistant_ce`, weight1.0, text vocabulary[0,248320). This distinct
supervised lane uses actual audio observations and verified authored assistant
answers. It uses no invented audio-teacher logits, Qwen VL audio path, ViT
feature target or foreign speech-codec mapping. It makes no ASR or general sound
capability claim. The initial24 controlled non-speech fixtures have8 train,
8 validation and8 test rows; only the eight admitted training rows enter the
initial one-pass budget. Additional verified speech sources require separately
pinned admission and a new immutable source/budget plan.

Preserve the actual parent chat template and every native text ID. Insert the
native media placeholders inside the user's turn immediately before the native
user-closing EOS. The exact native assistant generation-prefix anchor owns only
the assistant body and completion EOS; prompt, empty-thinking prefix, media and
right padding carry no loss. Normalize CE by the actual supervised positions.
This insertion policy is `native_audio_before_user_close_v1` and must be tested
with the actual parent tokenizer before batch sealing.

Clean observations use the actual source-pinned H3 FP32 audio VAE, full source
WAV coverage, genuine libav stereo32k resampling,800-sample hops, raw
[1,32,2,T] latents and channel-major[2T,32] packing. No waveform or latent trim,
padding or silence fallback is allowed. The actual H3 unit-scale raw/model
transform is source-pinned and checked bitwise. Only native input feature
columns480:512 receive those clean model-space values; sigma is exactly0.
Keep the existing H3 unit's supported type set{2,3}, while this lane supplies
only type3 observations and no fabricated video rows.

Select a **new standalone observation chart**, not parent standalone-XYZ parity.
For stereo branch c∈{0,1}, frame i and measured hop h=800/32000=.025s, native
bridge metadata is `((i+.5)*h,0,h,c)`; the final component is categorical stereo
identity, not a measured spatial coordinate. H3 rotary coordinates are
`(prompt_text_count+i,0,c)`. Actual prompt text receives its ordinal
`(text_ordinal,0,0)` coordinate; reference-answer coordinates are hidden from
H3 by the read policy. All arrays must bind the exact encoded source and actual
native text/media scatter. This chart removes the old dependency on a fabricated
video grid; it is an explicitly amended observation policy requiring source
ownership, angle/scatter and prefix/answer-invariance tests. No equivalence to
the parent's video-relative stereo coordinates is claimed.

The H3 attention policy is `h3_prompt_and_clean_observation_only`: only prompt
text and clean observed audio enter that unit's joint attention. Every reference
assistant-answer row is hidden from H3, preventing future-answer → audio →
earlier-answer leakage. The shared trunk retains its real causal/full-depth
forward and backward. Train only the H3 unit, type3 input bridge columns480:512
and type embedding row3, plus all shared trunk r16/alpha16 adapters. Keep the
other native units, text embedding/head and generative emission frozen.

Initial explicit optimizer: AdamW, H3 unit_lr2e-5, type3 bridge_lr2e-5,
shared LoRA_lr1e-4, betas(.9,.95), eps1e-8, weight_decay0, global grad clip1.0,
20 warmup steps, cosine to minimum ratio.1; FP32 trainable masters, explicit
BF16 or FP32 frozen forwards, seed20261013. These are the initial recovery
media optimizer values and D2's shared-adapter rate, not an optimum claim.
The initial budget is exactly one successful update per accepted training row,
with deterministic source order, no repeated holdouts and exact Adam/RNG/sampler
resume. A source addition changes the pinned one-pass budget and identity.

Require actual native/hybrid parity on the current immutable checkpoint, the
source-pinned full clean-encoder proof, actual parent-tokenizer insertion proof,
globally held-excluded source admission, finite full-depth gradients and
nonempty optimizer ownership before the production entry point opens. Transfer
the cumulative native state and complete shared-r16 bank from the preceding
vision D3d stage before creating the optimizer. Resume the same stage only with
its exact checkpoint identity, optimizer and sampler; stage transition carries
parameters but starts its own optimizer. Export complete replacement shared-r16
state and cumulative native overlays to D4; never sum sequential adapter banks.

Retain30-minute rolling checkpoints with optimizer/RNG/sampler, keep2; halt on
8 consecutive nonfinite skips or more than20 skips in the last100 attempts.
At25% of the initial budget, stop for a pinned actual free-answer review on
held audio fixtures. Record parsed events/count/onsets and transcript metrics
only where genuine speech labels exist. Teacher-forced CE and waveform SNR are
diagnostics and do not establish understanding. D4 replay and learning-rate
multiplier.3 remain as frozen, with this lane's replay inclusion explicitly
pinned in the consolidation source plan.
