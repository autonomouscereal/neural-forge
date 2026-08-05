# chimera-omni fold readiness — exact 4x, 2026-08-04

Driver: `experiments/run_chimera_omni_toy_fold.py`
Coverage matrix and kernel-interface expectations:
`docs/CHIMERA_OMNI_FOLD_COVERAGE_2026-08-04.md`
Target platform: `C:/Users/cereal/Documents/ai training/chimera-omni` (read-only;
nothing was vendored, and the eight source files this run read are digested in
`configuration.json`).

Reproduce:

```
.venv/Scripts/python.exe experiments/run_chimera_omni_toy_fold.py \
  --out reports/chimera-omni-fold-readiness-20260804
```

Re-verify from disk: `pytest tests/test_run_chimera_omni_toy_fold.py`.

## Headline

| | |
|---|---|
| parent unique parameters | **1,104,128** |
| child unique parameters | **276,032** |
| compaction | **exactly 4x** (`residual_from_exact = 0`, all 63 tensors individually 4x) |
| quantization / sparsity / storage sharing / sidecar | none |
| gradient steps during the fold | **0** (fold-only guard `clean = True`) |
| parent forwards / child recaptures | 1 / 37 |
| emitted media interfaces | 6 (`image`, `video`, `audio`, `music`, `tts`, `ocr`) |

## What was folded

A real `chimera_omni.modeling.ChimeraOmniModel`, built through chimera-omni's
own `ChimeraConfig`, trained on chimera-omni's own tiny-proof batches. It passes
chimera-omni's own criterion — `chimera_tiny_proof_status = "passed"`, loss drop
ratio 0.158 over 180 steps.

The compaction axis is the **residual width only**: `hidden_size` 128 → 32 with
`num_heads * attention_head_dim` and `intermediate_size` pinned. Every chimera
parameter block is exactly linear in `hidden_size` under those pins, so 4x is an
identity rather than a rounding.

The fold is a closed-form, zero-gradient install of all 63 child tensors using
`neural_forge.hf_closed_form_fit.fit_ridge_linear` and `fit_diagonal_norm`, plus
exact chart transport for the two one-hot-fed tables (token embedding, media type
embedding), run inside `run_fold_only_callback` with the parent protected.

## Behavior, on a held-out batch the fold never saw

| | trained parent | folded child (4x smaller) | blank child (same size) |
|---|---|---|---|
| native-media masked MSE | 0.0524 | **0.1221** | 0.8523 |
| next-token top-1 agreement with parent | — | **73.3%** | 0.8% |
| emitted-media relative RMSE vs parent | — | **0.356** | 1.185 |
| text cross-entropy | 7.253 | 7.204 | 7.266 |

The folded child closes **91.3%** of the blank → parent gap on the native-media
objective. Text cross-entropy is not a discriminating axis on this corpus (it is
almost all placeholder tokens) and should not be read as one.

## Emitted output

`emissions/{parent,folded-child,blank-child}/` hold real artifacts rendered by
chimera-omni's own renderers: `emitted_image.png`, `emitted_audio.wav`,
`emitted_music.wav`, and `emitted_video_frames/` (four PNGs plus an ffmpeg MP4).
All nonempty; every file digested in `manifest.json`. All three models answered
the **same** stimulus — `stimulus_sha256` is identical across them, which is
what makes their errors comparable.

Per-modality reconstruction error against the tiny-proof target, single sample
each:

| modality | parent | folded child | blank child |
|---|---|---|---|
| video | 0.173 | **0.175** | 1.060 |
| music | 0.415 | **0.772** | 1.407 |
| image | 0.333 | **1.302** | 1.533 |
| audio | 0.508 | **1.504** | 1.183 ← folded child is **worse** than blank here |

One sample per modality is not a measurement; the aggregate masked MSE above is.
The audio row is kept because dropping it would be dishonest.

## What this does not show

* No claim of retained intelligence. 1.1M parameters, 4 layers, a synthetic
  objective.
* Only the `local` sequence mixer and the dense channel mixer were folded.
  `kda`, `csa`, `hca`, `gated_mla` and `stable_latent_moe` were **avoided by
  configuration** (see `configuration.json → avoided_block_families`), not
  covered. The inspector compiles all of them.
* One global orthonormal residual chart, not a sealed per-boundary
  `ResidualBoundaryChartSet` — that artifact requires an
  `HFCausalCircuitEvidence` bundle a chimera parent cannot produce.
* `plan.json` records `planner_ready = false`. That is correct: the promoted
  causal branch kernels require `inspector = "hf-causal-decoder"` and
  `position_policy = "rope"`, and chimera declares neither. The numeric fold
  here is a driver-local transaction, not a planned action, and it has no
  ledger record.

## Files

| file | contents |
|---|---|
| `configuration.json` | runner digest, torch/python versions, chimera-omni source digests, parent/child manifests, avoided families, seeds |
| `parent.json` | training curve and chimera-omni's own pass criterion, blank/trained state digests |
| `inspection.json` | both IR graph fingerprints, the coverage matrix, every interface, the depth-residual nodes |
| `budget.json` | exact-4x accounting and the per-tensor map |
| `plan.json` | the planner's verbatim refusal and why it is correct |
| `fold.json` | the chart report, the fold-only guard report, and all 63 install actions with their fit metrics |
| `behavior.json` | holdout scores, parent-relative comparisons, emitted-output evidence |
| `complete.json` | the summary the driver prints |
| `manifest.json` | byte counts and SHA-256 for every file above plus every emitted artifact |
