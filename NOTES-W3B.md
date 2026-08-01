# NOTES-W3B — the native audio lane is executable end to end

Written 2026-07-31 by the audio-executability wave, on top of the modality
foundation (`ab97948f`) and its integration list in `NOTES-1H.md`.

This wave closed items 1–3 of `NOTES-1H.md` and made item 4 concrete *at toy
scale only*. What follows is what exists now, what was proved, and what a
later wave still has to do.

---

## What landed

### 1. Inspector — `src/neural_forge/dynamic_model_inspector.py`

`NativeAudioCircuitInspector` (`name="native-audio"`, `priority=240`,
registered in `default_inspector_registry()` above the catch-all
`HFCausalDecoderInspector` at 100 and below `NeoNativeVisionCircuitInspector`
at 250).

It emits one `CircuitKind.MODALITY_INPUT` node,
`input.audio.pcm16_frame_encoder`, whose **input** port is
`PortKind.MODALITY` with `modality=canonical_modality("audio")`, bound to a
`ModalityInterface(interface_id="audio.pcm16_frames.input", direction="input",
representation="pcm16_frames")`. Its output feeds
`input.audio.dense_projection` (`INTERFACE_PROJECTION`) and then
`input.audio_text_embedding_fusion`, so `{i.modality for i in
graph.interfaces} == {"audio", "text"}` and
`graph.nodes_of_kind(CircuitKind.MODALITY_INPUT)` is non-empty. Every learned
tensor has exactly one semantic owner; `_GraphBuilder.finish()` still enforces
exact parameter coverage, and the inspector additionally rejects any learned
tensor inside the audio front end that is not owned by `frame_conv` or
`projection`.

`supports()` is keyed on an **explicit opt-in marker**, not on a model type
alone:

```
config.neural_forge_native_audio_lane == NATIVE_AUDIO_LANE_MARKER
    and config.model_type == "native_audio_chat"
    and hasattr(model, "audio_encoder") and hasattr(model, "language_model")
```

so it cannot hijack a real audio-capable model that has not declared this
exact topology. `NATIVE_AUDIO_LANE_MARKER`, `NATIVE_AUDIO_LANE_CONFIG_KEY`,
`NATIVE_AUDIO_MODEL_TYPE`, `NATIVE_AUDIO_FAMILY`, and
`NATIVE_AUDIO_REPRESENTATION` are exported so a real model can opt in without
copying string literals.

The inspector also refuses a front end that consumes precomputed features:
`audio_config.sample_encoding` must be `"int16_normalized_pcm"`. That is the
mechanical form of the project's native-modality rule — no frozen external
encoder in the graph.

### 2. Target adapter — `src/neural_forge/dynamic_audio_adapter.py` (new)

`NativeAudioTargetSynthesisAdapter` (`adapter_id
"neural_forge.audio.native_audio"`, version 1, priority 240) is the audio
mirror of `NeoNativeVisionTargetSynthesisAdapter`: it clones the loaded
parent's own class and config, changes exactly
`decoder_config.intermediate_size`, constructs the blank directly on the
parent device/dtype under a forked RNG, re-inspects independently, and proves
that only the coupled gate/up/down triplets changed shape and that the
parameter delta equals `layers * 3 * hidden * (source - target)`.

No entry was added to the compiler-reviewed structural-buffer registry in
`dynamic_target_synthesis.py`, because the architecture registers **zero
nonzero runtime buffers** (learned absolute position embeddings instead of
rotary `inv_freq` buffers). The synthesis path therefore resolves the empty
compiler policy and the audit reports `nonzero_buffer_count == 0`. A real
audio parent with rotary buffers *will* need a registry entry; see "Deferred".

### 3. Planner contract — `src/neural_forge/dynamic_fold_planner.py` (additive)

* `_single_neuron_swiglu_layer_contract(..., family=...)` — the
  family-parameterized form of the validated NEO layer contract.
* `_native_audio_model_contract(source, target)` — family
  `native_audio_chat`, identical layer indices, identical global-node
  identities/ports/attributes, `{audio, text}` interfaces on both sides via
  `MODALITY_AUDIO`/`MODALITY_TEXT` rather than literals, and the
  `source.tokenizer_fingerprint != target.tokenizer_fingerprint` rejection
  **kept unchanged** — folding must not touch vocabulary.
* `_bundle_kind_contract`: new `"native_audio_model"` branch requiring the
  complete model bundle, `{audio, text}` interfaces, at least one
  `MODALITY_INPUT` node, metadata `native_pcm16_frame_encoder is True` and
  `external_frozen_audio_encoder is False`, and an ordered residual contract
  per layer.
* Dispatch branch next to the NEO one emitting `method_id
  "native_audio_swiglu_model"` with
  `fit_objective="audio_conditioned_single_swiglu_neuron_merge_per_layer_with_exact_native_pcm16_front_end_copy"`,
  `selection_policy="sealed_calibration_trace_low_importance_anchor_regression"`,
  `promotion_gate="cached_parent_audio_behavior_then_fresh_reload"`.

Nothing existing in the planner was modified; the NEO contract still runs
through its own untouched code path.

### 4. Method registry — `src/neural_forge/dynamic_fold_registry.py` (additive)

```
method_id          native_audio_swiglu_model
family_id          native_audio_coupled_swiglu_model
status             EXPERIMENTAL
source/target      ("native_audio_model",)
evidence_channels  audio_conditioned_post_swiglu, cached_parent_outputs,
                   child_state, residual_endpoints
coupled_roles      q_k_v_o_cache, gate_up_post_swiglu_down,
                   all_layers_ordered_residual
```

`EXPERIMENTAL` is selectable, so the planner can emit it, but the status
records honestly that this lane has one toy proof and no real-model evidence.

### 5. Fold system — `src/neural_forge/dynamic_audio_fold.py`

New in this module, mirroring `dynamic_neo_fold.py`:

* `NativeAudioLayerFoldSelection` — one removed neuron plus its sparse
  survivor anchors and compensation coefficients.
* `_sparse_anchor_regression` / `_selection_from_statistics` — heldout-blind
  low-importance candidate ranking with sparse anchor regression.
* `NativeAudioFoldCache` + `derive_native_audio_fold_cache(...)` — sealed
  selections derived from an `AudioTraceStatisticsCache` with **no new parent
  forward**. `verify()` rebuilds every selection from the sealed statistics
  and rejects a mismatch.
* `NativeAudioEvidenceProvider`, `NativeAudioFoldOnlyInitializer`,
  `NativeAudioFoldTransaction`, `NativeAudioSwiGLUFoldOperator`,
  `NativeAudioFoldAdmissionEvaluator`.
* `NativeAudioStackResources` / `NativeAudioFoldStackAdapter`
  (`target_families=("native_audio_chat",)`,
  `target_modalities=("audio", "text")`), wiring the parent-relative
  `NativeAudioBehaviorGate` as **both** `final_evaluator` and
  `fresh_reload_evaluator`.
* `write_native_audio_dynamic_fold_receipt(...)`.
* `audio_lane_status()` updated to state what is now provided.

**Deliberate deviation from `NOTES-1H.md` item 3.** That note suggested adding
a `selections` field to `AudioTraceStatisticsCache._body()` and bumping
`TRACE_CACHE_SCHEMA` to `-v2`. This wave did **not** do that. Instead the
selection lives in a separate sealed artifact (`NativeAudioFoldCache`,
schema `neural-forge-native-audio-fold-cache-v1`) that cites the statistics
cache through `source_cache_key`. Reasons:

1. `AudioTraceStatisticsCache` stays exactly what its docstring says it is —
   evidence with no policy — and its v1 seal, manifest, and
   `fold_selection_included: False` flag are unchanged, so nothing already
   sealed has to be re-sealed.
2. A later change to the *selection policy* reseals only the fold cache, not
   the audio-conditioned evidence it was derived from. NEO conflates the two
   in one artifact and pays for it with `derive_neo_native_vision_fold_cache`
   copying all statistics into every portfolio arm.

The audio fold cache does copy the statistics blob so it is self-contained on
reload; the cost at real scale is one extra copy of the sealed statistics per
selection policy arm.

---

## What was proved

`tests/test_native_audio_executable_lane.py` (12 tests, all passing).

The toy architecture (`_TinyNativeAudioChatModel`) is a learned strided
`Conv1d(1 -> 4, kernel=stride=500)` over int16-normalized PCM samples, a
bias-free `Linear(4 -> 8)` projection, and a 2-layer causal decoder
(GQA 4 query / 2 key-value heads, head_dim 2, SwiGLU intermediate 6, vocab 12,
learned absolute positions). Audio frames are concatenated ahead of the text
tokens in the residual stream, exactly as the emitted graph declares. No
external encoder, no frozen weights, nothing loaded from a hub.

The end-to-end test
`test_native_audio_fold_cache_stack_behavior_artifact_and_receipt`:

1. Generates a sealed probe suite with `build_native_audio_probe(...,
   examples_per_category=4, seed=17)` → 12 probes, 9 calibration / 3 heldout,
   every waveform re-decoded and re-hashed under `AUDIO_HASH_BASIS`.
2. Caches parent behavior through `write_audio_parent_output_cache` and
   authenticates it with `load_audio_parent_behavior_cache`.
3. Captures audio-conditioned traces with
   `build_native_audio_fold_trace_cache` — the trace runner feeds the *real
   decoded PCM16* of each calibration probe through the whole model. Asserted:
   trace calls equal the calibration probe ids exactly and are disjoint from
   the heldout ids; `trace_rows == (9 * (16 audio frames + 6 prompt tokens),)
   * 2 layers`.
4. Derives the fold cache (`candidate_count=3`, `anchor_count=2`) with **no
   further trace calls**, and shows a one-byte manifest edit fails
   `NativeAudioFoldCache.open` on the seal.
5. Runs `AutomaticDynamicFoldingSystem.run(...)` — i.e. the full
   `DynamicFoldingCompiler` synthesize → plan → fold → admit → behavior-gate →
   publish → fresh-reload path.

Results asserted by the test:

* `result.complete` is true; the plan has exactly one action whose
  `method_id == "native_audio_swiglu_model"`.
* `published.artifact.fresh_reload_decision.accepted` is true and
  `published.artifact.fresh_reload_parameters_frozen is True` — the loader does
  `model.requires_grad_(False).eval()`, which is the lesson of `18ccb1b4`.
* `behavior_gate.last_report["parent_forward_count"] == 0`,
  `failures == []`, and heldout `parent_correct_retained ==
  parent_correct_total`. Every parent-correct answer, calibration and heldout,
  is retained. `behavior_gate.evaluation_count == 4`.
* Exact parameter accounting: `expected_delta = 2 layers * 3 tensors * 8 hidden
  = 48`; the child graph, the request budget, and
  `artifact.exact_unique_parameters` all equal
  `parent_unique - 48`, and the summed per-tensor element delta between the
  reloaded child and the parent equals 48.
* Every non-MLP tensor in the reloaded child — the whole audio front end,
  every attention projection, both norms per layer, the embeddings, the final
  norm and the head — is `torch.equal` to the parent's.
* For each layer, `gate_proj`/`up_proj` equal the parent's rows index-selected
  by the survivors, and `down_proj` equals the parent's survivor columns plus
  the compensated removed column, recomputed independently in the test.
* The reloaded child still runs the audio path: logits are
  `(1, 22, 12)` and finite.
* The receipt self-hash verifies and records
  `parent_forward_count_during_fold_and_gates == 0`,
  `heldout_samples_used_for_selection == 0`, `optimizer_steps == 0`,
  `backward_calls == 0`, `fresh_reload_parameters_frozen is True`, and
  `source_cache_key == statistics.cache_key`.

Command:

```
& "C:/Users/cereal/Documents/ai training/neural-forge/.venv/Scripts/python.exe" \
  -m pytest tests/test_dynamic_model_inspector.py tests/test_dynamic_fold_planner.py \
  tests/test_dynamic_fold_registry.py tests/test_native_audio_lane.py \
  tests/test_dynamic_neo_native_vision.py tests/test_native_audio_executable_lane.py -q
```

→ `77 passed`.

### Regression scope

Adding a method to `default_method_registry()` changes
`FoldMethodRegistry.fingerprint`, and therefore `DynamicFoldPlan.fingerprint`
for **every** lane. Adding an inspector changes `default_inspector_registry()`
for every model. Both were checked against a same-commit baseline rather than
assumed harmless:

* The whole suite was run: `45 failed, 9580 passed, 30 skipped` in 50 minutes.
  None of the failures are in this wave's files.
* Every failing file was then re-run twice from this working tree — once
  normally, once with `PYTHONPATH` pointed at a detached `git worktree` of the
  same commit's `src/`, so only the library code differed while tests and the
  untracked `reports/` data stayed identical. Both runs produced
  `34 failed, 256 passed, 1 skipped`, with **byte-identical failure lists**.
  Same procedure, same result, for every test file that imports
  `dynamic_fold_planner`, `dynamic_fold_registry`, `dynamic_model_inspector`,
  `dynamic_system`, `default_method_registry`, or
  `default_inspector_registry`.
* Pre-existing failures at that commit, unrelated to this wave, live in
  `tests/test_dynamic_search.py`, `tests/test_hf_blank_checkpoint.py`,
  `tests/test_hf_lfm2_cpcf_v5_terminal_action_integration.py`,
  `tests/test_run_dynamic_qwen3_codebook_fold.py`,
  `tests/test_run_lfm2_cpcf_v5_selective_blank_fold.py`,
  `tests/test_run_lfm2_dynamic_catalog.py`, the three
  `tests/test_run_qwen_p35_*` runners, and
  `tests/test_surgery_meta_compiler.py`. Notably
  `test_existing_attention_profile_hashes_remain_byte_stable` fails
  identically with and without this wave's registry addition, so the new
  method id did not move any recorded profile hash.
* `HEAD` advanced from `29939aea` to `988cb7fc` while this wave was in
  progress (three commits from the cross-family/on-policy wave). None of them
  touch any file this wave changed; the baseline comparison above was redone
  against `988cb7fc`.

### Honest limits of the behavior proof

The toy decoder cannot transcribe audio. As in the validated NEO vision proof,
the closed-set answer returned by the gate's child runner is gated on a
parent-derived marker weight in the audio front end, not on emergent
competence. What the gate actually proves is: the child executed its declared
inference entrypoint on the real decoded waveform for all 12 probes, the
parent was never forwarded, and the child's answers matched the cached parent
answers on every parent-correct probe in every split × category cell. What it
does **not** prove is that folding preserves real audio understanding — that
requires item "a production audio parent" below.

`tests/test_native_audio_lane.py::test_audio_lane_status_declares_its_open_seams`
was updated in place, because `audio_lane_status()` no longer lists the
inspector, selection rule, and method as missing. Its replacement assertions
pin what is still genuinely open.

---

## Deferred — what a later wave still owes

1. **A production audio-capable parent.** `NOTES-1H.md` item 4 is still open in
   substance. The toy architecture proves the machinery; it does not choose a
   real model. The selection criteria in `NOTES-1H.md` item 4 remain correct,
   with two additions learned here:
   * the parent must expose `language_model.model.layers[i].mlp.down_proj` or
     supply a `projection_resolver`, and its layer forward must return the
     residual endpoint (tensor or leading tuple element);
   * if the parent registers **nonzero runtime buffers** (rotary `inv_freq`,
     cached cos/sin tables, mel filterbanks), `dynamic_target_synthesis.py`'s
     `_builtin_structural_buffer_policy` needs a compiler-reviewed entry keyed
     on `("neural_forge.audio.native_audio", 1)`. That file was outside this
     wave's ownership, so no entry was added. Synthesis will fail closed with
     "Nonzero runtime buffers lack an explicit structural policy" until it is.
2. **Promote the method past `EXPERIMENTAL`.** One toy proof is not evidence.
   Promotion should follow a real parent clearing the same gate.
3. **A real answer decoder for the gate.** The marker-gated child runner is a
   scaffold. A production run needs a runner that reads the model's actual
   generated text and normalizes it through `normalize_audio_answer`, which is
   already closed-set and side-effect free.
4. **Multi-neuron and multi-layer scopes.** The registered method removes
   exactly one neuron per layer (`source_neurons == target_neurons + 1`), the
   smallest honest fold, matching NEO's 6144→6143. Wider scopes need their own
   contract, method id, and evidence.
5. **`NOTES-1H.md` items 5, 6, 7 are untouched** (migrating the NEO gate onto
   the shared case schema, the README raw-protocol section, and the
   `canonical_modality()` mechanical pass at existing emission sites). The new
   audio inspector already routes both its modality strings through
   `canonical_modality()`.
6. **`MODALITY_OUTPUT` remains unimplemented output-side**, unchanged from the
   limitation carried forward in `NOTES-1H.md`. `audio_lane_status()` now says
   so explicitly.
7. **A shared selector module.** `_sparse_anchor_regression` and
   `_selection_from_statistics` in `dynamic_audio_fold.py` are a deliberate
   copy of the NEO numerics so the two lanes can diverge independently. If a
   third lane needs them, extract them into a modality-neutral module and have
   both lanes delegate — as a single reviewable no-op diff, not piecemeal.
