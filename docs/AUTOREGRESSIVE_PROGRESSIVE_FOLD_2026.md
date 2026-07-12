# Autoregressive Progressive Fold

## Completion Contract

Neural Forge is not complete when tensors merely fit a requested shape. A
successful proof must begin with a randomly initialized target architecture,
run no optimizer or recovery step during folding, and produce coherent useful
output after folding. Every report compares the parent and child on the same
held-out behavior, exact generation, choice, math, code, and manual prompts.

The final tool accepts a user parameter budget and reports when the requested
shape cannot contain the measured parent function. It may always materialize a
valid architecture, but it must never claim arbitrary capability retention
when the target capacity or measurements do not support that claim.

## Same-Tokenizer Path

1. Plan one or more blank intermediate shapes between the parent and final
   budget. Limit the parameter compression of each stage instead of performing
   one destructive jump.
2. Trace donor residual inputs/outputs on domain-balanced calibration prompts.
   Score singleton and adjacent-pair depth spans by residual magnitude,
   composed-delta rank loss, and delta incompatibility.
3. Partition every donor layer exactly once with deterministic dynamic
   programming. Preserve critical layers as singleton spans and merge low-cost
   adjacent spans.
4. Fold whole nonlinear circuits. Q/K/V/O and gate/up/down are one transaction;
   selected post-softmax and post-SwiGLU paths retain their native consumer
   directions unless an all-path aggregate improves held-out response RMSE and
   cosine similarity.
5. Generate prefixes with the current child. Run the parent on those exact
   token IDs and fit the child on causal decision states only. Candidate block
   blends are committed only when parent action agreement or parent action
   log-probability improves without repetition, longest-run, or premature-EOS
   regression.
6. Rebuild rollout traces after every globally accepted block and repeat for a
   bounded number of zero-gradient rounds.
7. Run the immutable 1000 behavior, 1000 generation, and 250 choice suite,
   parent-relative retention, and manual output inspection. PPL is diagnostic,
   never the promotion objective.

The first progressive schedule to test is approximately:

```text
Qwen3 4B parent
  -> blank Qwen3 ~1.4B, 30 layers
  -> blank Qwen3 ~600M, 24 layers
  -> blank Qwen3 158M, 18 layers
```

Every stage remains a fold into blank weights. An intermediate is the output of
the preceding fold, not a pretrained child shortcut.

## Cross-Tokenizer And Cross-Architecture Path

Exact token-ID traces are used only when tokenizer maps truly match. For
different tokenizers, text is represented as canonical UTF-8 byte offsets:

1. Record each parent and child token's covered byte interval.
2. Align causal states only at common byte boundaries, without assuming equal
   token counts or decoding one model's IDs inside the other model.
3. Fit generic block-interface residual targets in a shared observable
   coordinate system. Architecture-specific QKV/SwiGLU transplantation is an
   optional prior, not a requirement.
4. For incompatible blocks, retain the blank child's nonlinear features and
   solve consumer maps against parent residual deltas. Alternate producer and
   consumer closed-form solves only when held-out autoregressive gates improve.
5. Fit the child output head against parent likelihood over complete child
   token byte spans, then apply the same free-running promotion gates.

This path is required for Qwen, Gemma, LFM, GPT-style, recurrent/convolutional
hybrid, and other decoder families to enter one target architecture.

## Unified Raw Modalities

The target model uses one shared token stream. No learned vision tower, audio
encoder, modality projector, adapter, bridge, or neural modality decoder is
permitted.

Image, PCM audio, video frames, OCR payloads, and quantized 3D arrays are
mechanically framed and serialized through the same 256 raw-byte token rows.
Shape, dtype, length, and checksums are ordinary header tokens. Output is
mechanically deserialized; there is no modality-specific learned decoding
network.

The raw-token rows become normal embedding and output-head parameters in the
child. Multimodal parent traces are aligned to raw payload byte boundaries and
enter the same progressive folding and child-rollout gates as text. A modality
proof must include exact codec round trips, parent-before/child-after task
metrics, and manually inspected generated text or media.
