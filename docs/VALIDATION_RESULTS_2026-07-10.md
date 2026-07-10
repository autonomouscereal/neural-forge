# Neural Forge Pivot Validation - 2026-07-10

## Promotion Contract

- The child starts from seeded random weights.
- Folding may use deterministic merging, projection, SVD/eigendecomposition,
  clustering, and closed-form solves.
- Recovery training, backward calls, and optimizer steps are zero.
- A fold is not successful unless free-running text is coherent and retains
  parent behavior. Hidden-state cosine, PPL, or tensor shape alone cannot
  promote a checkpoint.
- Parent outputs are cached once and reused for every comparison.

The cached Qwen3-4B parent has `4,022,468,096` parameters and scored PPL
`877.579654`, behavior `400/1000`, exact generation `388/1000`, and forced
choice `250/250`.

## Completed 29.9x Control

Run:

`qwen3_4b_to_blank_qwen134m_jointvocab_sourcel2_delta0p1_gpu4_r3_20260710`

- Child: blank Qwen3, `134,445,568` parameters.
- Compression: `29.918934x`.
- Residual map: joint activation plus tied-vocabulary sparse fold.
- Vocabulary row repair: donor L2 norm.
- Recovery/optimizer/backward: zero.
- Full evaluation completed: behavior `0/1000`, exact generation `0/1000`,
  forced choice `96/250`, PPL `153,086,609.22`.
- Parent-pass behavior and generation retention were both zero.

The local held-out gate accepted the fold because final-state relative RMSE
improved by `3.13%` and logit relative RMSE improved by `36.90%`. Candidate
final-state cosine was `0.4511`, logit cosine was `0.7900`, but teacher top-1
agreement remained exactly zero. This is another direct demonstration that
local cosine/RMSE improvement is not a behavioral promotion signal.

Manual inspection across math, code, tokenizer, commonsense, and model-surgery
prompts found repeated `Slee`, `hog`, and mixed-language fragments. It did not
produce a correct number, usable Python function, or coherent explanation.

Evidence bundle SHA-256:

`25eb51d849201d6266c3cdf05209c4d0f7f6a997b9f5beae3b860b808579e2f4`

## Native Raw-RGB Parent Baseline

Parent: `Paranioar/NEO1_5-2B-SFT`, `2,896,618,496` parameters.

The model consumes raw RGB patches through its integral vision path. The
evaluation uses 256 deterministic images covering color, count, spatial, and
OCR tasks. It does not use a frozen external vision tower or a post-hoc
adapter.

| Slice | Result |
|---|---:|
| Overall | `239/256` |
| Calibration | `180/192` |
| Held out | `59/64` |
| Color | `64/64` |
| Count | `63/64` |
| Spatial | `48/64` |
| OCR | `64/64` |

Manual output is direct and coherent. The sole count miss answered `6` for an
expected `5`; spatial failures consistently confused `below` with `right`.
The held-out baseline therefore records what the modality parent actually
knows before any fold.

- Output-cache SHA-256:
  `d02b28bbc91b5206dc7f86b48ba7743aa66a334a037ea390fc702259fd2d9e7f`
- Inference fingerprint:
  `d8dd8e98f5d271f7970a7841b1a22d3b2acd589a765420be3474a671f0ca47b8`

## Capacity Reallocation

A strict Qwen ByteLevel-BPE reducer now creates ordinary Hugging Face
tokenizers whose target rows map one-to-one to exact donor rows. The first
production tokenizer has `50,982` rows, retains `50,712` BPE merges, and was
selected using all `2,906` calibration records.

- Transport fingerprint:
  `d305ef401e219ce0c420cc6625d6290424a3def1d98559404e240a6a9d2c5973`
- Selected-row fingerprint:
  `dbf0e9acac9aefca4ad7e2ed4935114c574fe5c70562af5769ccaccf947fea04`

This allows an exact `134,445,568` parameter Qwen3 child with hidden size
`768`, 19 layers, intermediate size `1,408`, 6 query heads, 3 KV heads, and
128-dimensional heads. Compared with the old 151,936-row child, roughly 52
million parameters move from tied vocabulary rows into the transformer body.

## Mathematical Pivot

The completed controls show four noncommuting errors that must not be hidden by
local regression metrics:

1. Independent Q/K rotations change attention dot products.
2. Averaging Q/K paths before softmax introduces cross terms absent from the
   parent.
3. Averaging gate/up paths before SwiGLU does not equal merging their
   post-nonlinearity contributions.
4. Averaging two residual blocks does not equal their ordered composition.

The current implementation therefore uses one orthonormal residual basis but
keeps GQA/RoPE paths and gated FFN paths as coupled sparse circuits. The next
operator selects real activation medoids for nonlinear producers and combines
discarded paths only in fitted residual consumers.

## Active Full Evaluations

- GPU 6: Qwen3-4B to blank Qwen3-1.13B sparse stage, `3.5597x`, full
  `1000/1000/250` evaluation.
- GPU 4: the same 1.13B stage with global PCA residual coordinates and coupled
  sparse nonlinear paths.
- GPU 0: Qwen3-4B to exact 134M reduced-tokenizer child with PCA residual
  coordinates and coupled sparse nonlinear paths.

None of these active runs is a claimed success until the full reports and
manual parent/child outputs are complete.
