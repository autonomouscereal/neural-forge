# Neural Forge kernel stage-2 proofs (real models, 2026-08-01)

Repository commit `ab3ec99a8359112ca66192c116df487106420c6f`, cloned to a fresh deployment on the
AI server.  Every measurement below comes from CPU-only, `nice -n 15`,
8-thread runs in
`/home/cereal/model-surgery-lab/artifacts/kernel_stage2_20260801/`.
Nothing was published and no live run directory was touched.

## Evidence

* Kind translation: real LFM2.5-230M, 64 prompts, 64x121 tokens, 6464 valid rows (4041 fit / 805 selection / 1618 validation), 1 parent forward, float32 parent, float64 fold.
* Cross width: real SmolLM2-135M, 24 prompts, 24x224 tokens, 4713 valid rows (2357 fit / 2356 validation), 1 parent forward, 61 residual-write boundaries.

## Why the cross-width stage 2 does not run on LFM2

```
build_decoder_layout: UnsupportedDecoderLayoutError: Unsupported causal decoder architecture: model_type 'lfm2'; supported model types are 'gpt2', 'llama', 'qwen3', 'gemma2', and 'gemma3_text'.
capture_hf_causal_circuit_evidence: UnsupportedDecoderLayoutError: Unsupported causal decoder architecture: model_type 'lfm2'; supported model types are 'gpt2', 'llama', 'qwen3', 'gemma2', and 'gemma3_text'.
```

## Measurements

### Direction B: real ShortConv layer -> compiled softmax GQA/RoPE attention

`linearize_short_conv_branch` then `compile_short_conv_to_attention_translation`,
which feeds synthesized band evidence to the unchanged
`compile_hf_causal_gqa_attention_fold`.  `identity_qk` targets a child whose
attention has no head RMSNorm; `lfm2_rms_qk` targets LFM2's own attention
geometry, where the kernel jointly fits the Q/K norm gains.

| layer / target | gate B disp | gate C disp | rho_2 | const energy | frozen-gate val RMSE | val endpoint RMSE | val endpoint RMSE bf16 | min assigned lag mass | routed remainder energy | compiled |
|---|---|---|---|---|---|---|---|---|---|---|
| L1 identity_qk | 0.664289 | 0.79774 | 0.770495 | 0.375284 | 0.876542 | 0.825465 | 0.825484 | 0.65438 | 0.681393 | yes |
| L1 lfm2_rms_qk | 0.664289 | 0.79774 | 0.770495 | 0.375284 | 0.876542 | 0.820067 | 0.820143 | 0.675009 | 0.672509 | yes |
| L3 identity_qk | 0.601491 | 0.69675 | 1.045359 | 0.837843 | 1.018613 | 0.773536 | 0.773454 | 0.650415 | 0.598358 | yes |
| L3 lfm2_rms_qk | 0.601491 | 0.69675 | 1.045359 | 0.837843 | 1.018613 | 0.769345 | 0.769336 | 0.675009 | 0.591892 | yes |

### Direction A: real attention layer -> ShortConv branch

`attention_band_decomposition` then
`compile_attention_to_short_conv_translation`.  `rho_band` / `rho_far` are
energy fractions of the exact lag split; the separable floor is the best
*unconstrained* lag-separable linear operator on the same rows, which bounds
everything the rank-one-per-channel conv could reach.

| layer | rho_band (fit) | rho_far (fit) | rho_far (val) | band mass mean | band mass p10 | conv fit band RMSE | conv val band RMSE | floor fit RMSE | floor val RMSE | routed remainder energy | compiled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L2 | 0.224888 | 0.69443 | 0.698033 | 0.206269 | 0.115242 | 0.405579 | 0.851652 | 0.221683 | 0.772243 | 0.855246 | yes |
| L4 | 0.455288 | 0.524224 | 0.526013 | 0.257689 | 0.156935 | 0.440803 | 0.655282 | 0.205841 | 0.587078 | 0.709568 | yes |

### Cross-width residual boundary charts

Parent `/home/cereal/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (`llama`, H=576, 30 layers) -> blank child H=432.

#### cross_width_576_to_432

plan ready: True; methods: causal_gqa_rope_residual_attention_layer, causal_swiglu_residual_mlp_layer, copy_exact, token_interface_bridge; child recaptures: 2

| defect over all boundaries | min | median | max |
|---|---|---|---|
| cycle defect c_b | 0.035848 | 0.076219 | 0.136363 |
| norm-commutation row fluctuation nu_b | 0.003077 | 0.01126 | 0.023772 |
| fit retained energy fraction | 0.980769 | 0.994104 | 0.998705 |
| held-out retained energy fraction | 0.961266 | 0.992562 | 0.998463 |
| absorption-demand unabsorbed fraction | 0 | 0 | 0 |

| branch | committed | input boundary | output boundary | fit endpoint RMSE | validation endpoint RMSE | upstream error energy ratio | expansion max abs error | fail-closed reason |
|---|---|---|---|---|---|---|---|---|
| token_interface_bridge LNone | yes | None | None | n/a | n/a | n/a | n/a |  |
| causal_gqa_rope_residual_attention_layer L0 | yes | layer000.attention_in | layer000.mlp_in | n/a | 0.075903 | 6.985e-16 | 1.019e-07 |  |
| causal_swiglu_residual_mlp_layer L0 | yes | layer000.mlp_in | layer001.attention_in | n/a | 0.665663 | 5.698e-06 | 8.690e-07 |  |

#### same_width_baseline_576

plan ready: True; methods: causal_gqa_rope_residual_attention_layer, causal_swiglu_residual_mlp_layer, copy_exact; child recaptures: 2

| defect over all boundaries | min | median | max |
|---|---|---|---|
| cycle defect c_b | 0 | 0 | 0 |
| norm-commutation row fluctuation nu_b | 0 | 0 | 0 |
| fit retained energy fraction | n/a | n/a | n/a |
| held-out retained energy fraction | n/a | n/a | n/a |
| absorption-demand unabsorbed fraction | 0 | 0 | 0 |

| branch | committed | input boundary | output boundary | fit endpoint RMSE | validation endpoint RMSE | upstream error energy ratio | expansion max abs error | fail-closed reason |
|---|---|---|---|---|---|---|---|---|
| copy_exact LNone | yes | None | None | n/a | n/a | n/a | n/a |  |
| copy_exact LNone | yes | None | None | n/a | n/a | n/a | n/a |  |
| causal_gqa_rope_residual_attention_layer L0 | yes | layer000.attention_in | layer000.mlp_in | n/a | 1.195e-05 | 0 | 1.192e-07 |  |
| causal_swiglu_residual_mlp_layer L0 | yes | layer000.mlp_in | layer001.attention_in | n/a | 0.379676 | 1.185e-13 | 1.580e-06 |  |
| copy_exact LNone | yes | None | None | n/a | n/a | n/a | n/a |  |
| copy_exact LNone | yes | None | None | n/a | n/a | n/a | n/a |  |

#### Branch endpoint: crossing versus same-width baseline

| branch method | same-width H576 validation endpoint RMSE | cross-width H576->H432 validation endpoint RMSE | delta attributable to the crossing |
|---|---|---|---|
| causal_gqa_rope_residual_attention_layer | 1.195e-05 | 0.075903 | 0.075891 |
| causal_swiglu_residual_mlp_layer | 0.379676 | 0.665663 | 0.285987 |
| token_interface_bridge | n/a | n/a | n/a |


## What limits these numbers

* **Direction A ran a reduced policy.**  `channel_width = 128` against a
  1024-wide residual, `bias = False` (LFM2's own conv is bias-free), and a
  two-candidate ladder (`value_transport` / `lag_blocked`, 3 rounds,
  `alpha = 1e-4`, condition cap `1e4`).  The full default ladder is 24
  candidates at `channel_width = 1024`; the per-channel gate ridge is
  `O(channels * rows * H^2)`, which is hours per layer on the 8-thread CPU
  budget this run was held to.  `lag_blocked` won the selection partition
  in both layers.  The condition cap was saturated (`1e4`) on every
  sub-solve, so the ridge is set by conditioning, not by `alpha`.
* **Fit rows are scarce relative to the free columns.**  4041 fit rows
  against a 3072-column lag-separable design means the separable floor's
  *fit* number is close to interpolation; its *validation* number is the
  trustworthy one, and the conv-versus-floor comparison should be read on
  validation only.  The same caveat explains the large fit/validation gap
  in the conv band fits.
* **Parent float32, fold float64.**  Direction A's fit target is the
  float64 reconstruction `contract(P, V) @ W_o^T`, which reproduces the
  live float32 branch to ~7e-7 relative RMSE (reported per layer).  The
  band decomposition then reconstructs that target to ~5e-16, so the
  kernel's evidence-consistency gate measures the kernel rather than the
  parent's rounding.  Direction B is fitted directly against the parent's
  captured float32 delta.
* **The same-width MLP baseline narrows the FFN.**  Its child is the
  cross-width child with the residual width put back to 576, so its
  intermediate size is 1152 in both arms and the difference isolates the
  crossing.  It is not a no-change control: a child identical to the
  parent is planned as `copy_exact` and never fits anything.
* **Two adjacent boundaries only.**  Layers 1..29 of the cross-width child
  were not folded, so nothing here measures error accumulation across
  depth, which the design names as its dominant risk.  The reverse
  reclosure sweep is sealed as `executed: false` in every branch receipt.
* Every branch receipt records `parent_forward_calls: 0`,
  `optimizer_steps: 0`, `backward_calls: 0`, `pruning_operations: 0` and
  `quantization_operations: 0`.

Merged JSON self-seal: `6fceb1005c65dce31ac9349a1bb6fa7df901712067c2d2fa368ae36e54722ba5`.
