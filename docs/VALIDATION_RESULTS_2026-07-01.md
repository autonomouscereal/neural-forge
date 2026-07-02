# Validation Results - 2026-07-01

## Remote Sync Correction

The GitHub remote exists at `https://github.com/autonomouscereal/neural-forge.git`.
Current active work is on branch `codex/lrc-blank-compression`, not `main`.

Latest pushed commits on the active branch:

- `2705f30` - Add geomean RMS fold scaling.
- `4ac6b55` - Add source RMS fold scaling.
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

`hf-donor-project-init` and `hf-xfamily-chain-anchor-compress` now also accept
`source-rms` for `--projection-scale` / `--init-projection-scale`. This keeps
the folded donor tensor at donor/source RMS after resizing instead of matching
the blank target initializer RMS. It is opt-in; existing `target-rms` and `none`
runs are unchanged.

They also accept `geomean-rms`, which scales the folded tensor to the geometric
mean of source and target RMS. This is the middle lane between target-RMS
damping and full donor/source-RMS preservation.

Validation:

- Local focused: `1 passed, 3 skipped, 193 deselected`.
- Local full: `129 passed, 119 skipped`.
- Remote focused: `4 passed, 193 deselected`.
- Remote full: `248 passed`.
- Source-RMS patch validation: local full `129 passed, 120 skipped`; remote
  focused `4 passed, 194 deselected`; remote full `249 passed`.
- Geomean-RMS patch validation: local focused `2 passed, 3 skipped`; local full
  `129 passed, 121 skipped`; remote focused `5 passed, 194 deselected`; remote
  full `250 passed`.

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

## Source-RMS Fold Smoke

Smoke artifact:

`/home/cereal/model-surgery-lab/neural-forge-src/runs/diagnostic_smokes/gpt2_to_blank_smollm135m_source_rms_projection_diag_20260701_220904`

This repeats the GPT-2 -> blank SmolLM2-135M smoke with
`projection_scale=source-rms`.

Result:

- Parameter count: `134,515,008`.
- Layer tensors initialized: `270`.
- Nonfinite diagnostic count: `0`.

Projection diagnostic summary:

| Metric | Mean | Min | Max | P95 |
| --- | ---: | ---: | ---: | ---: |
| projected/source RMS ratio | 0.9996 | 0.9729 | 1.0368 | 1.0036 |
| projected/target RMS ratio | 2.5145 | 0.1850 | 5.8962 | 4.2958 |
| target update relative RMS | 2.7171 | 0.4729 | 5.9856 | 4.4081 |
| target projection cosine | 0.2183 | -0.0069 | 0.9942 | 0.9933 |

Interpretation:

`source-rms` preserves donor tensor scale after folding. It is deliberately more
aggressive than `target-rms`; the next question is behavioral, not numeric:
whether block-interface reconstruction plus EoRA compensation can use the
larger donor signal or whether it destabilizes blank-target recovery.

## Geomean-RMS Fold Smoke

Smoke artifact:

`/home/cereal/model-surgery-lab/neural-forge-src/runs/diagnostic_smokes/gpt2_to_blank_smollm135m_geomean_rms_projection_diag_20260701_221515`

This repeats the GPT-2 -> blank SmolLM2-135M smoke with
`projection_scale=geomean-rms`.

Result:

- Parameter count: `134,515,008`.
- Layer tensors initialized: `270`.
- Nonfinite diagnostic count: `0`.

Projection diagnostic summary:

| Metric | Mean | Min | Max | P95 |
| --- | ---: | ---: | ---: | ---: |
| projected/source RMS ratio | 0.8457 | 0.4113 | 2.3248 | 1.9600 |
| projected/target RMS ratio | 1.4953 | 0.4301 | 2.4282 | 2.0730 |
| target update relative RMS | 1.6780 | 0.3153 | 2.6309 | 2.2996 |
| target projection cosine | 0.2183 | -0.0069 | 0.9942 | 0.9933 |

Interpretation:

`geomean-rms` lands between `target-rms` and `source-rms` as intended. It may be
the better blank-target compromise if full source-RMS is too destabilizing but
target-RMS damps donor signal before recovery can use it.

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
- `qwen4b_blank_smollm_eora_grail_source_rms_full_gpu4_20260702` is queued
  behind GPU4. It matches the current EoRA/GRAIL strict blank-target lane but
  switches init projection scaling to `source-rms`.
- `qwen4b_blank_smollm_eora_grail_geomean_rms_full_gpu0_20260702` is queued
  behind GPU0. It matches the same EoRA/GRAIL strict blank-target lane but
  switches init projection scaling to `geomean-rms`.

## Multi-Donor Blank-Target Fold Smoke

Code revision: `1d4c53e` (`Add multi donor projected init`).

Remote validation before smoke:

- Focused multi-donor tests: `3 passed`.
- Full remote test suite: `252 passed`.

Smoke artifact:

`/home/cereal/model-surgery-lab/runs/diagnostic_smokes/multidonor_gpt2_qwen06_to_blank_smollm135_geomean_20260701_222743`

This verifies the first explicit A+B -> C fold path:

- Target shape: blank `HuggingFaceTB/SmolLM2-135M`.
- Donor A: `openai-community/gpt2`.
- Donor B: `Qwen/Qwen3-0.6B-Base`.
- Donor weights: `0.35,0.65`.
- Projection scale: `geomean-rms`.
- Token initialization: `donor-tokenized-weighted`.

Result:

- Final parameter count: `134,515,008`.
- Stage 0 effective blend alpha: `1.0`.
- Stage 1 effective blend alpha: `0.65`.
- Stage 0 layer tensors initialized: `270`.
- Stage 1 layer tensors initialized: `270`.
- Final nonfinite diagnostic count: `0`.
- Final initialized embedding rows: `48,993`.
- Final weighted tokenized rows initialized: `9,550`.

Final projection diagnostic summary:

| Metric | Mean | Min | Max | P95 |
| --- | ---: | ---: | ---: | ---: |
| projected/source RMS ratio | 1.2175 | 0.1322 | 1.8120 | 1.5863 |
| projected/target RMS ratio | 0.7703 | 0.4962 | 3.7498 | 1.7365 |
| target update relative RMS | 0.7631 | 0.2243 | 2.9622 | 0.8179 |
| target projection cosine | 0.7223 | 0.5762 | 0.9874 | 0.9802 |

Interpretation:

The sequential weighted projection path is operational for different donor
families into a blank target architecture. This is not yet an intelligence
retention claim; it is a checkpoint-construction and diagnostics milestone that
enables the next real behavior lane: run the same multi-donor init through
block-interface recovery and compare parent/donor/candidate behavior reports.

## Multi-Donor Chain Recovery Wiring

Code revisions:

- `f7f85e0` (`Wire multi donor init into chain recovery`)
- `f1b4cbe` (`Parameterize blank SmolLM chain launcher`)
- `259a459` (`Expose full eval controls in blank SmolLM launcher`)

Validation:

- Local full test suite after chain wiring: `134 passed`, `121 skipped`.
- Remote focused chain tests: `5 passed`.
- Remote full test suite: `255 passed`.
- Remote launcher syntax: `bash -n` passed.

Implementation:

- `hf-xfamily-chain-anchor-compress` now accepts comma-separated sources and
  `--donor-weights`/`--source-weights`.
- The chain `init` stage dispatches to `multi_donor_projected_init` when
  multiple donors or explicit weights are provided.
- Existing single-donor chain behavior is preserved, including positional
  `stages` compatibility.
- `block` and `alm` recovery stages still use a single teacher API; they now
  use the first source model as `primary_source_model` and record the full
  source list and weights in `chain_anchor_summary.json`.

Queued full behavior lane:

- Tag: `qwen4b_gpt2_multidonor_blank_smollm_eora_full_gpu6_20260702`.
- Queue PID: `2203637`.
- Queue script:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_gpt2_multidonor_blank_smollm_eora_full_gpu6_20260702.queue.sh`
- Queue log:
  `/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/logs/qwen4b_gpt2_multidonor_blank_smollm_eora_full_gpu6_20260702.queue.log`
- Sources: `Qwen/Qwen3-4B-Base,openai-community/gpt2`.
- Weights: `0.85,0.15`.
- Primary recovery teacher: `Qwen/Qwen3-4B-Base`.
- Target: blank `SmolLM2-135M`.
- Init scale: `geomean-rms`.
- Final eval caps: `1000` behavior probes, `1000` generation prompts, `250`
  choice probes, `128` text/PPL items.
- Block layer gate caps: `1000` behavior probes and `400` generation prompts.

The queue is intentionally behind the existing GPU6 queue
`qwen4b_blank_smollm_eora_grail_cosine_full_gpu6_20260702`, then waits for
GPU6 Python jobs to clear before launching.

## Projected Vocab Sparse Surface Recovery

Code revision: pending commit after `6b377e4`.

Implementation:

- `hf_recovery._candidate_to_source_token_groups` now falls back to the existing
  sparse surface-piece selector when direct source-tokenizer encoding is empty
  or truncates fragile digit/code/mixed-digit tokens.
- Cross-tokenizer ALM recovery, blockwise recovery, and LM-head calibration now
  record `projected_vocab_group` coverage summaries, including valid rows,
  multi-source rows, digit rows, and code rows.

Validation:

- Local syntax/collection: `134 passed`, `123 skipped` in the Windows venv.
- Remote focused Torch tests:
  `2 passed, 204 deselected`.
- Remote full suite:
  `257 passed`.

Interpretation:

This is not a standalone intelligence-retention claim. It closes one specific
cross-tokenizer failure mode from the X-Token/byte-interface research path: the
recovery losses can now see and report sparse donor token groups for compact
child tokens such as `42` instead of silently losing or truncating those rows.
