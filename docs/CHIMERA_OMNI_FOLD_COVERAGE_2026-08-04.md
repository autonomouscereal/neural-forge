# chimera-omni fold coverage — 2026-08-04

Target platform: `C:/Users/cereal/Documents/ai training/chimera-omni`
(`dense_decoder_native_omnimodal_encoder_free`, read-only for this work).

Inspector: `ChimeraOmniCircuitInspector`
(`src/neural_forge/dynamic_model_inspector.py`, `name = "chimera-omni-native-decoder"`,
priority 245, registered in `default_inspector_registry`).

Sealed proof: `reports/chimera-omni-fold-readiness-20260804/`
Driver: `experiments/run_chimera_omni_toy_fold.py`

---

## 1. The architecture, as the IR sees it

### 1.1 Sequence mixers (`ChimeraBlock.mixer`)

| chimera family | module | `CircuitKind` | state/cache edge | why that kind |
|---|---|---|---|---|
| `local` | `LocalCausalAttention` | `ATTENTION` | `CACHE`, recurrent self-edge | split Q/K/V/O, `num_heads`/`num_kv_heads` grouped-query, causal sliding window (`local_window`), bias-free |
| `csa` | `SparseLatentAttention(stride=csa_compress_stride)` | **two** `ATTENTION` nodes + one `RESIDUAL_ADD` | `CACHE` on both | it is genuinely two circuits: a nested `LocalCausalAttention` **plus** a strided mean-summary attention over compressed blocks, summed |
| `hca` | `SparseLatentAttention(stride=hca_compress_stride)` | same as `csa` | `CACHE` on both | identical topology, different compression stride |
| `gated_mla` | `GatedMultiHeadLatentAttention` | `ATTENTION` | `CACHE`, recurrent self-edge | compressed-latent Q and KV paths (`q_down/q_norm/q_up`, `kv_down/kv_norm/kv_up`) plus a full-rank sigmoid output gate; **not** split-QKV, so the split-QKV role contract fails closed on it |
| `kda` | `KimiDeltaAttention` | `RECURRENT_MIXER` | `STATE`, recurrent self-edge | a bounded gated delta-rule recurrence carrying an explicit `[batch, heads, key_dim, value_dim]` state through `forward_with_state`; softmax-free; the semantics of `RECURRENT_MIXER` fit exactly |

`kda`'s three internal depthwise causal short convolutions (`q_conv`, `k_conv`,
`v_conv`) are **not** emitted as separate `SHORT_CONV` nodes. They do not touch
the residual stream: they filter the already-projected q/k/v streams inside the
recurrence, and the delta rule consumes them directly. Splitting them out would
suggest a fold boundary that does not exist. They are named on the node instead
(`internal_short_conv_operators`, `internal_short_conv_kernel_size`,
`internal_short_conv_paths`).

### 1.2 Channel mixers (`ChimeraBlock.ffn`)

| chimera kind | module | `CircuitKind` decomposition |
|---|---|---|
| `dense` | `FeedForward` | one `GATED_MLP`. Note the layout: gate and up are **one fused `up` tensor of shape `[2 * intermediate, hidden]`**, split by `chunk(2, dim=-1)`, then `down`. This is *not* the `gate_proj`/`up_proj`/`down_proj` layout the promoted SwiGLU branch kernel matches on. |
| `stable_latent_moe` | `StableLatentMoEFeedForward` | `INTERFACE_PROJECTION` router → `INTERFACE_PROJECTION` `latent_down` → one `GATED_MLP` per routed expert → `RESIDUAL_ADD` weighted combine (fed a `CONTROL` edge from the router) → `NORM` `routed_norm` → `INTERFACE_PROJECTION` `latent_up` → one `GATED_MLP` per full-width shared expert → `RESIDUAL_ADD` combine. `router_bias` is a non-gradient buffer and is correctly absent from parameter ownership. |

Activation is `silu` or `situ` (`situ` is a tanh-capped gate/up variant); it is
carried as the `activation` attribute and changes no topology.

### 1.3 Boundary projection — `MODALITY_INPUT` side

`NativeContinuousMediaBridge` is the whole media front end. There is **one**
learned projection, no tower, no frozen encoder:

| node | `CircuitKind` | owns |
|---|---|---|
| `input.native_media.boundary_projection` | `MODALITY_INPUT` | `native_media_bridge.feature_proj.weight` — `[hidden, native_feature_dim]` |
| `input.native_media.type_embedding` | `MODALITY_INPUT` | `native_media_bridge.type_embed.weight` — `[native_type_count, hidden]` |
| `input.native_media.position_projection` | `POSITION_ENCODING` | `native_media_bridge.position_proj.weight` — `[hidden, native_position_dim]` |
| `input.native_media.bridge_norm` | `NORM` | `native_media_bridge.norm.weight` |
| `input.native_media_text_fusion` | `INTERFACE_PROJECTION` | nothing — `scatter_add` of media rows into the trunk residual at `native_media_indices` |

The boundary node carries **one typed input port per chimera media lane**
(`image_features`, `video_features`, `audio_features`, `music_features`,
`tts_features`, `ocr_features`) and exactly one learned tensor. That pairing is
the claim: six typed entries, one shared projection.
`modality_tower_count = 0` and `external_frozen_encoder = False` are asserted
on the node.

**Lane canonicalization.** `CANONICAL_MODALITIES` has no `music` and no `tts`.
Both are raw-waveform segment lanes that `waveform_to_native_segments` builds
exactly the way it builds `audio`, differing only in `type_id` and default
segment length, so they enter the IR as canonical `audio`. The chimera lane
name survives on the interface id (`native_media.music.input`) and the
representation (`chimera_native_continuous_segments_music`), and the mapping is
stated in graph metadata as
`chimera_media_lane_canonicalization = "image->image, video->video, audio->audio, music->audio, tts->audio, ocr->ocr"`.
Nothing is renamed silently.

### 1.4 Emission head — the second real `MODALITY_OUTPUT`

`UnifiedNativeMediaEmissionHead` is the repository's second real emitter after
the points3d lane's quantized centroid head
(`src/neural_forge/dynamic_points3d_output.py` holds that contract).

| node | `CircuitKind` | owns |
|---|---|---|
| `output.native_media.gather` | `INTERFACE_PROJECTION` | nothing — gathers media rows out of the trunk's final-normed hidden |
| `output.native_media.norm` | `NORM` | `native_media_head.norm.weight` |
| `output.native_media.emission_head` | `MODALITY_OUTPUT` | `native_media_head.out.weight` — `[native_feature_dim, hidden]` |

It reads the **same** `output.final_norm` output the text head reads, so the
two are siblings rather than a chain, matching the points3d contract's
"an emitted-media head is an additional output of the decoder, not a
post-processor of text". Six `direction="output"` interfaces are emitted, one
per lane; `emitted_media_interfaces(graph)` returns all six.

Difference from points3d worth stating: points3d emits a **quantized** payload,
so its output gate can compare by content digest. chimera emits **continuous**
features (`continuous_shared_reconstruction`). Content-digest retention is not
decidable there, so output-side review for chimera is relative error plus
decoded artifacts, not exact-match retention.

### 1.5 Depth axis — `block_attnres` in the residual-edge model

`residual_mode = "block_attnres"` is a *block* scheme, and the IR carries it
literally:

* `DepthAttentionResidual` becomes a `RESIDUAL_ADD` node owning its `query` and
  `norm.weight`, with **one explicit `EdgeKind.RESIDUAL` input port per depth
  source** (`depth_source_000`, `depth_source_001`, …) plus the running prefix
  sum. Attributes: `depth_source_count`, `mixed_source_count`,
  `selection = "learned_softmax_over_rms_normalized_depth_sources"`,
  `learned_pseudo_query`, `invoked_at_this_depth`, `identity_passthrough`.
* Depth sources are the checkpointed prefix sums, taken at layers where
  `layer_index % attnres_block_size == 0`. The producing node is the trunk
  input (for checkpoint 0) or the previous layer's `feed_forward_residual`.
* **At a checkpoint depth the accumulator restarts.** `_forward_trunk` sets
  `prefix_sum = None` and then `prefix_sum = attention_output`; there is no base
  term. `layer.NNN.operator_residual` at those depths therefore has a single
  `delta` input and carries `residual_accumulator_reset = True`. Emitting a
  `base` edge there would put an edge in the graph the model never executes.
* The first block's *attention* selector is never invoked (no block has
  completed at depth 0). It is emitted with `depth_source_count = 0`,
  `invoked_at_this_depth = False`, `identity_passthrough = True` — dead weight
  at that depth, recorded as such.
* `residual_mode = "none"` is also supported and compiles to the ordinary
  pre-norm two-write residual layer with no selectors.

### 1.6 Everything else

| chimera part | `CircuitKind` |
|---|---|
| `embed` | `TOKEN_EMBEDDING` (typed ledger ranges named on the node) |
| every `RMSNorm` | `NORM` |
| `lm_head` | `OUTPUT_HEAD` (`head_role = "next_token"`) |
| `mtp_heads[i]` | `OUTPUT_HEAD` (`head_role = "multi_token_prediction"`) |
| `flow_head` / `grounding_head` / `sync_head` | `OUTPUT_HEAD` (`head_role = "flow"`/`"grounding"`/`"sync"`) |

`position_policy = "nope"`. chimera has no RoPE anywhere: KDA carries order
through its recurrence, and `local`/`csa`/`hca`/`gated_mla` carry it through the
causal mask and the window alone. This single fact is what makes the promoted
`causal_gqa_rope_residual_attention_layer` kernel inapplicable — see §3.

### 1.7 Observed matrix (the proof's parent, `local`-only)

From `reports/chimera-omni-fold-readiness-20260804/inspection.json`:

| `CircuitKind` | nodes | logical params | operator types | required kernel family |
|---|---|---|---|---|
| `attention` | 4 | 196,608 | `local_causal_attention` | `causal_sliding_window_split_qkv_attention` |
| `gated_mlp` | 4 | 491,520 | `dense` | `fused_gate_up_swiglu` |
| `output_head` | 5 | 263,296 | — | — |
| `token_embedding` | 1 | 131,072 | — | — |
| `modality_input` | 2 | 9,216 | — | — |
| `modality_output` | 1 | 8,192 | — | — |
| `residual_add` | 17 | 2,304 | — | — |
| `norm` | 11 | 1,408 | — | — |
| `position_encoding` | 1 | 512 | — | — |
| `interface_projection` | 2 | 0 | — | — |

48 nodes, 69 edges, 14 interfaces (7 input, 7 output), 1,104,128 parameters
across 63 tensors, every one owned by exactly one node. The edge count is 69
rather than 71 because the two checkpoint depths (layers 0 and 2) carry no
residual base edge — see §1.5.

---

## 2. What folds today

**Proven, in `reports/chimera-omni-fold-readiness-20260804/`:**

* A real trained chimera-omni parent (chimera's own tiny-proof objective,
  chimera's own pass criterion: `chimera_tiny_proof_status = "passed"`,
  loss drop ratio 0.158).
* **Exact 4x** unique-parameter compaction: 1,104,128 → 276,032, residual from
  exact = 0, all 63 tensors individually exactly 4x, no quantization, no
  sparsity, no storage sharing, no parent sidecar.
* A zero-gradient closed-form fold of the whole coverable subgraph — token
  embedding, media boundary projection, position projection, every norm, every
  depth selector, `local` attention Q/K/V/O, the fused-gate dense FFN, the text
  head, the MTP head, all three auxiliary heads, and the shared media emission
  head — inside the repository's fold-only guard (`clean = True`, 0 optimizer
  steps, 0 backward calls, 0 autograd-grad calls, parent unchanged).
* Decoded and emitted output evidence: 73.3% next-token top-1 agreement with
  the parent (blank child of the same size: 0.8%), 0.356 emitted-media relative
  RMSE against the parent (blank: 1.185), and real nonempty PNG/WAV/MP4
  artifacts under `emissions/`.

**How the compaction is exact.** With `attention_head_dim` pinned (so
`num_heads * attention_head_dim` is independent of `hidden_size`) and
`intermediate_size` pinned, *every* chimera parameter block is exactly linear in
`hidden_size`. Halving `hidden_size` twice divides the unique parameter count by
exactly four with nothing left over. That is a property of the architecture, not
an accounting trick, and it makes the residual width the single compaction axis.

**Which kernels ran.** `neural_forge.hf_closed_form_fit.fit_ridge_linear` and
`fit_diagonal_norm` (plus `normalize_rows`), with exact chart transport for the
two one-hot-fed tables. All closed form, no autograd. The child is recaptured
between install groups (37 child forwards, 1 parent forward) so every fit sees
the rows the child will really produce.

---

## 3. What does not fold, and why

### 3.1 The planner refuses this crossing, correctly

`plan.json` records `planner_ready = false` with gaps
`cross_family_closed_loop_layer_operator`, `global_interface_projection`,
`global_structural_semantic_alignment`, `tokenizer_fingerprint`.

`_family_neutral_causal_layer_contract` requires **both** graphs to declare
`inspector == "hf-causal-decoder"` and `position_policy == "rope"`. A chimera
graph declares `chimera-omni-native-decoder` and `"nope"`. The refusal is not a
bug: `CausalGQAAttentionBranchOperator` calls `build_decoder_layout()` and reads
`HFCausalCircuitEvidence`, neither of which a chimera model can satisfy, so a
planned action would fail at execution instead of at planning.

### 3.2 Kernel gaps, by family

| family | gap | what a kernel has to provide |
|---|---|---|
| `kda` | **no fold kernel** | see §4 |
| `csa` / `hca` | **no fold kernel** | see §4 |
| `gated_mla` | **no fold kernel** | see §4 |
| `stable_latent_moe` | **no fold kernel** | expert-count and latent-width reduction with router-bias re-balancing; the quantile-balanced bias is a buffer, so a fold must recompute it rather than transport it |
| `local` attention | folds today, driver-locally | a *registered* operator + method id, so it becomes a planned transaction rather than driver code |
| dense FFN | folds today, driver-locally | the fused `[2*intermediate, hidden]` gate/up layout, which `CausalGatedMLPBranchOperator`'s `gate_proj`/`up_proj`/`down_proj` role contract does not match |

### 3.3 Cross-cutting gaps

1. **Per-boundary charts.** The fold used **one global orthonormal residual
   chart** (rank 32 of 128, 96.2% retained energy, projection relative RMSE
   0.195, orthonormality defect 2e-15). The production artifact —
   `ResidualBoundaryChartSet` with one chart per residual-write boundary,
   cycle/absorption diagnostics and a lineage seal — authenticates against a
   sealed `HFCausalCircuitEvidence` bundle and cannot accept a chimera parent.
   Until an evidence bundle exists for non-HF trunks, chimera cannot use the
   per-boundary path.
2. **No sealed evidence bundle.** `capture_hf_causal_circuit_evidence` is
   HF-shaped. The chimera fold captures its own activations through forward
   hooks and seals batch digests instead. That is weaker: there is no
   fit/validation token split and no cached-evidence identity.
3. **`block_attnres` depth transport.** The depth selectors were folded, but
   only pointwise (a diagonal norm refit plus a ridge-fit pseudo-query). A real
   depth kernel has to reason about the checkpoint resets and the softmax mix
   jointly, not one selector at a time.
4. **Depth reduction is untouched.** Only width moved. Reducing `num_layers`
   needs `ordered_residual_span_compose`, which is `standalone=False`.
5. **No registered chimera fold method.** The fold is a driver-local
   transaction. It is not in `default_method_registry`, so it has no ledger
   record, no experiment signature, and no `DynamicFoldExecutor` run receipt.

---

## 4. Interface expectations for the KDA / CSA / HCA kernels

A parallel architect owns these. This section is the contract the platform
already exposes, not an implementation.

### 4.1 What the IR hands a kernel

Every mixer node the kernel will see already carries its geometry. A kernel
should read attributes, never module names:

* **KDA** (`CircuitKind.RECURRENT_MIXER`, `operator_type = "kimi_delta_attention"`):
  `heads`, `key_head_dim`, `value_head_dim`, `decay_rank`, `gate_lower_bound`,
  `state_elements`, `state_layout = "batch_heads_key_value"`,
  `internal_short_conv_operators = 3`, `internal_short_conv_kernel_size`,
  `internal_short_conv_paths`, `internal_headwise_output_norm = True`,
  `softmax_free = True`, `not_attention_qkv = True`,
  `fold_kernel_family_required = "bounded_gated_delta_recurrence"`.
  Ports: one `NORMALIZED_RESIDUAL` in, one `BRANCH_DELTA` out, and a
  `PortKind.RECURRENT_STATE` in/out pair joined by a recurrent
  `EdgeKind.STATE` self-edge.
* **CSA/HCA**: two `ATTENTION` nodes and one `RESIDUAL_ADD`, linked by
  `compound_parent` (`"csa"`/`"hca"`) and `compound_branch`
  (`"local_window"`/`"compressed_summary"`). The summary node carries
  `compress_stride` and
  `summary_reduction = "mean_over_stride_with_leading_zero_summary"`. A kernel
  may fold the two branches independently **only** if it also owns the combine;
  the branches share a residual write.
* **gated MLA**: `projection_style = "compressed_latent_kv"`, `key_head_dim`,
  `value_head_dim`, `extra_head_dim`, `kv_latent_size`, `query_latent_size`,
  `full_rank_output_gate = True`.

### 4.2 What a kernel must declare to become plannable

1. A `FoldMethodSpec` in `default_method_registry` with `source_kinds` /
   `target_kinds` naming a new bundle requirement (for example
   `chimera_kda_layer`), the evidence channels it consumes, and its
   `coupled_roles`. `_bundle_kind_contract` in `dynamic_fold_planner.py` needs a
   matching branch. Follow `native_points3d_model` — it is the closest existing
   shape (metadata-gated, modality-set-gated, emission-declaration-gated).
2. A gate on `graph.metadata["inspector"] == "chimera-omni-native-decoder"` and
   `position_policy == "nope"`. Do **not** relax the existing
   `hf-causal-decoder`/`rope` gate to let chimera through: that gate protects
   kernels that really do require RoPE and a `DecoderLayout`.
3. A `FoldOperator` whose `supports` reads the IR node attributes above and
   whose transaction refuses any geometry it has not proven. The registry
   authenticates operator implementations (`_bind_transaction_implementations`),
   so the transaction must be a clean callable with no training calls.

### 4.3 Numerical expectations

* **KDA.** The state is `[batch, heads, key_head_dim, value_head_dim]`. A width
  fold that changes `heads` or either head dim changes the state rank, which is
  a serving-contract change (`runtime_kernel_plan.paged_state`), not only a
  weight change. Expect a kernel to fold `hidden_size` while pinning the head
  geometry first — the same restriction this proof accepted for attention — and
  to treat head-count reduction as a separate, later capability. `q/k/v` are
  L2-normalized before the recurrence, so a fit against unnormalized q/k
  targets will not transfer; fit against the post-normalization rows. The
  decay path (`decay_down → decay_up → +decay_bias → sigmoid(scale··) →
  gate_lower_bound·`) is bounded in `[gate_lower_bound, 0]` after the log; a
  kernel must preserve that bound or the recurrence stops being contractive.
* **CSA/HCA.** The summary path pads to the stride and prepends a zero summary
  row, so the summary key/value sequence is `1 + ceil(seq/stride)` long and
  block-causally masked. Row counts differ between the two branches; a kernel
  that pools rows across the branches to fit one map will silently mix two
  different sequence axes.
* **gated MLA.** `kv_down` emits `kv_latent_size + extra_head_dim` columns and
  the extra columns are broadcast to every head as positional-free key material.
  Folding `kv_latent_size` without folding that tail changes the split point.
* **Everyone.** The residual coordinate system is shared. Until per-boundary
  charts exist for non-HF trunks, a chimera kernel should accept an
  externally-supplied residual chart object exposing `encode`,
  `fold_consumer_weight`, `source_width`, `target_width` and `map_sha256` —
  the surface `ResidualBoundaryChart` already exposes — rather than fitting its
  own.

---

## 5. Honest gaps in this work

* The proof is **tiny and synthetic**: 1.1M parameters, 4 layers, chimera's own
  tiny-proof objective. It shows the mechanism, not capability retention.
* The fold covers `local` + dense FFN only. `kda`, `csa`, `hca`, `gated_mla`
  and `stable_latent_moe` were **avoided by configuration**, and that is
  recorded in `configuration.json` under `avoided_block_families` and
  `avoided_channel_mixers`. The inspector compiles all of them; the fold does
  not attempt any of them.
* One global chart, not a sealed per-boundary chart set (§3.3.1).
* Driver-local transaction, not a registered fold method (§3.3.5).
* On the aggregate holdout the folded child closes **91.3%** of the blank →
  parent gap on the native-media objective (0.852 → 0.122 against the parent's
  0.052). On the four single-sample renders it is mixed: video 0.175 vs the
  parent's 0.173 and the blank's 1.060, music 0.772 vs 1.407, image 1.302 vs
  1.533 — but **audio 1.504, worse than the blank child's 1.183**. One sample
  per modality is not a measurement; the aggregate is. Both are in
  `behavior.json` and neither was dropped.
* Text cross-entropy moves very little (parent 7.253, folded 7.204, blank
  7.266) because the tiny-proof corpus is almost all placeholder tokens — text
  is not a discriminating axis on this objective and should not be read as one.
* Nothing here was folded on a GPU, at scale, or from a trained checkpoint of
  the real target size. The target size is still explicitly caller-selected in
  chimera-omni and no size was baked in here either.
