# NOTES-1F — family-neutral causal operator promotion

Working notes for other agents from the change that promoted the family-neutral
causal kernels into the dynamic fold operator registry.

## What is now registered and executable

| method id | status | operator |
|---|---|---|
| `causal_gqa_rope_residual_attention_layer` | experimental | `CausalGQAAttentionBranchOperator` |
| `causal_swiglu_residual_mlp_layer` | experimental | `CausalGatedMLPBranchOperator` |
| `token_interface_bridge` (pre-existing spec) | active | `StrictTokenInterfaceBridgeOperator` |

Operators live in `src/neural_forge/dynamic_causal_operator_promotion.py` and are
injected per system, exactly like `Lfm2DenseBudgetModelOperator`. Nothing is
registered globally.

## Planner opt-in

`DynamicFoldPlanner(enabled_experimental_method_ids=...)` (and the same keyword on
`plan_dynamic_fold`) is the only way the two experimental branch methods are ever
selected. With the default empty tuple the planner still emits the
`cross_family_closed_loop_layer_operator` gap exactly as before — no existing
caller changes behavior. Both branch ids must be enabled together; a layer whose
attention half has no owning transaction cannot be closed by folding its MLP half
alone.

One target layer becomes **two** actions (attention branch, then gated-MLP
branch), each owning its input norm, its coupled operator, and its ordered
residual write. If you consume plans positionally, note that layer actions are no
longer 1:1 with layers when the opt-in is on.

## Family gate on `hf_causal_circuit_evidence` was NOT widened

`HFCausalArchitectureEvidence` still accepts only `llama` and `qwen3`. gpt2 and
gemma2/gemma3_text are inspectable by `build_decoder_layout`, but the capture is
**not** layout-safe for them. Verified against real tiny models:

* **gpt2** — `position.kind == "learned_absolute"` (no RoPE), `layer_norm` rather
  than RMSNorm, `mlp.kind == "dense"` with no gate projection, fused Conv1D QKV.
  The record has no slot for any of that.
* **gemma2 / gemma3_text** — `embeddings.token_scale == 4.0`, RMSNorm
  `weight_kind == "unit_offset"` (gains are `1 + w`, and both `fit_rmsnorm_diagonal`
  and the kernel's `_head_rms_norm` fit direct-scale gains), `residual.kind ==
  "serial_pre_post_norm"` with post-attention and post-MLP norms that
  `HFCausalLayerEvidence` cannot hold, `attn_logit_softcapping = 50.0` and
  `final_logit_softcapping = 30.0` on gemma2 (the kernel's
  `causal_attention_probabilities` has no tanh softcap), alternating
  `sliding_attention` layers with `sliding_window = 8`, and one RoPE variant per
  layer type against the record's single `rope_theta`.

Because the old uniformity check compared only head counts, head_dim, biases and
Q/K-norm presence, the family allowlist was the *only* thing stopping a silently
lossy gemma record. `_assert_recordable_layout` in
`src/neural_forge/hf_causal_circuit_evidence.py` now rejects scaled token
embeddings, final/attention logit softcapping, per-layer-type RoPE variants,
post-operator block norms, non-direct-scale or non-RMS norms, sliding windows,
and heterogeneous layer types. llama and qwen3 pass all of them unchanged.
Anyone widening the family set should delete the corresponding guard *and* add
the matching evidence field in the same change.

## Deferred / known holes

1. **Target per-head Q/K normalization.** `compile_hf_causal_gqa_attention_fold`
   fits pre-RoPE target coordinates with identity Q/K semantics and refuses a
   non-identity target norm. A qwen3-shaped child therefore *plans* through the
   attention branch method and then fails closed in the operator with
   "cannot yet install a target with per-head Q/K normalization". Filling this
   needs a kernel that either fits through the target `q_norm`/`k_norm` gains or
   installs them jointly.
2. **Cross-width residual crossing.** Both branch operators require
   `target hidden_size == source hidden_size`; the planner contract enforces the
   same thing before emitting actions. Cross-width folding still needs its own
   boundary-map operator (the semantic compiler's `_boundary_map` is the model).
3. **Parent-coordinate endpoints, not sequential child replay.** Each branch is
   fitted against cached parent residual endpoints and is independent of the
   child's actual residual stream (`sequential_child_replay: False` in the
   receipt). Per-layer error compounds at runtime; the behavior gate remains the
   authority. The whole-model `HFCausalSameDepthSemanticCompiler` is still the
   path that does sequential child replay.
4. **Source activation is not recorded.** `fit_functional_swiglu` fits SiLU child
   features regardless of the source gate activation, and
   `HFCausalArchitectureEvidence` has no activation field. Adding one changes the
   sealed dataclass shape and would invalidate persisted bundles, so it was left
   alone.

## Pre-existing failure seen while regressing (not caused by this change)

`tests/test_dynamic_lfm2_cpcf_catalog.py::test_supports_uses_exact_build_validation_and_never_hijacks_fallback`
fails on this branch with
`DynamicTargetSynthesisError: Authenticated component-allocation receipt differs
from the request seals...`. It does `replace(target.request,
exact_unique_parameters=+1)` and expects an adapter rejection, but component
allocation receipt authentication in `DynamicTargetRequest.__post_init__` now
rejects the forged request at construction. Confirmed pre-existing by stashing
this change and re-running. It is the same class of problem as the xfail that was
just removed from `tests/test_hf_causal_semantic_compiler.py`: the forged request
needs to be legitimately re-sealed (see `_resealed_depth_span_tamper` there for
the pattern) so the adapter's own guard is what fires. Owner of the LFM2 CPCF
catalog tests should apply the same fix.
