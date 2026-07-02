# Validation Results - 2026-07-01

## Remote Sync Correction

The GitHub remote exists at `https://github.com/autonomouscereal/neural-forge.git`.
Current active work is on branch `codex/lrc-blank-compression`, not `main`.

Latest pushed commits on the active branch:

- `1374a9e` - Add donor fold projection diagnostics.
- `284c252` - Expose chain pre-reconstruction controls.
- `1ff3568` - Initialize special token rows from donor.

The AI-server experiment source at
`/home/cereal/model-surgery-lab/neural-forge-src` is not itself a git checkout,
so code changes must be committed/pushed locally and then uploaded into that
runtime copy before remote experiments use them.

## Fold Projection Diagnostics

`hf-donor-project-init` now writes per-layer donor-fold diagnostics into
`layer_reports[*].projection_diagnostic` and a top-level
`projection_diagnostic_summary`.

The diagnostic fields are intentionally not a promotion gate. They expose
whether a fold is numerically sane before behavior recovery:

- source, projected, and target-before finite fraction/RMS/mean-abs/max-abs.
- projected/source and projected/target RMS ratios.
- target update relative RMS.
- projected-vs-target cosine for same-shaped tensors.
- bounded deterministic sampling for large tensors, with original `numel` still
  reported.

Validation:

- Local focused: `1 passed, 3 skipped, 193 deselected`.
- Local full: `129 passed, 119 skipped`.
- Remote focused: `4 passed, 193 deselected`.
- Remote full: `248 passed`.

## Cross-Family Diagnostic Smoke

Smoke artifact:

`/home/cereal/model-surgery-lab/neural-forge-src/runs/diagnostic_smokes/gpt2_to_blank_smollm135m_projection_diag_20260701_220004`

Command shape:

- target config/tokenizer: `HuggingFaceTB/SmolLM2-135M`.
- donor: `openai-community/gpt2`.
- blank target, no target base weights.
- `fold_mode=mean`.
- `donor_layer_fold_mode=fold`.
- `projection_scale=target-rms`.
- `token_init_mode=donor-tokenized-weighted`.
- `dtype=float32`.

Result:

- Parameter count: `134,515,008`.
- Layer tensors initialized: `270`.
- Synthetic cross-architecture layer tensors initialized: `270`.
- Nonfinite diagnostic count: `0`.
- Token rows initialized: `48,993` embeddings, with `17` special rows and
  `18,233` donor-tokenized rows.

Projection diagnostic summary:

| Metric | Mean | Min | Max | P95 |
| --- | ---: | ---: | ---: | ---: |
| projected/source RMS ratio | 0.9892 | 0.1694 | 5.4047 | 3.8418 |
| projected/target RMS ratio | 0.9997 | 0.9744 | 1.0179 | 1.0025 |
| target update relative RMS | 1.1386 | 0.1082 | 1.4279 | 1.4173 |
| target projection cosine | 0.2183 | -0.0069 | 0.9942 | 0.9933 |

Interpretation:

The fold is numerically stable and the target-RMS scaling does what it claims:
projected tensors land at the blank target's scale. Dense weight cosine remains
near random for many tensors because the blank target geometry is unrelated to
the donor. That makes this diagnostic useful for deciding where activation
geometry, transport, block-interface pre-reconstruction, and residual
compensation are actually needed.

## Running Long Jobs

As of the remote poll after this patch:

- `qwen4b_cleanopen_anchor_newloss_full_gpu4_20260701` is still running and has
  reached layer 21 with behavior around `802/1000` and generation health
  `400/400`.
- `qwen4b_mathdrill_anchor_newloss_full_gpu6_20260701` is still running and has
  reached layer 19 with behavior around `807/1000` and generation health
  `400/400`.
- `qwen4b_mathdrill_anchor_eora_geom_full_gpu0_fix_20260701` is still running
  and has reached layer 9 with behavior around `807/1000` and generation health
  `400/400`.
- `qwen4b_blank_smollm_eora_grail_cosine_full_gpu6_20260702` is queued behind
  GPU6 and will use the newly uploaded diagnostics code when it starts.
