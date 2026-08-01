# NOTES-W5C — the native video lane, built on the audio template

Written 2026-07-31 by the video-foundation wave, on top of the modality
foundation (`docs/MODALITY_FOUNDATION_2026-07-31.md`, `NOTES-1H.md`) and the
executable audio lane (`c69d5a64`, `NOTES-W3B.md`). Branch
`codex/lrc-blank-compression`, base `ab3ec99a` (`HEAD` advanced to `09825ab2`
mid-wave; see "Regression scope"). Fold-only, nothing committed.

The audio lane was the template and was treated as read-only. Everything below
mirrors it deliberately; where the video lane diverges, the divergence is
stated and justified rather than left for a reader to discover by diffing.

---

## What landed

### 1. Canonical video content hash — `src/neural_forge/dynamic_behavior_gate.py` (additive)

`MediaReference` required an explicit `hash_basis` for `video` because no
canonical one existed. It now exists:

```
VIDEO_HASH_BASIS = "sha256('VIDEO:<width>x<height>:<fps>:<frames>\n' + concatenated per-frame RGB bytes)"
```

* `CANONICAL_MEDIA_HASH_BASES["video"]` resolves it, so a video
  `MediaReference` no longer has to name the basis at the call site — which was
  exactly the extension point Section 2 of the design doc described.
* `video_content_sha256(width, height, fps, frames, rgb_bytes)` is the helper.
  It fails closed on non-positive geometry, non-positive fps, a non-positive
  frame count, a non-bytes payload, and — the important one — a payload length
  that disagrees with `frames * height * width * 3`. A clip that lost or gained
  a frame, or that was decoded at a different geometry, cannot be sealed into a
  reviewed case.
* The basis binds the *decoded* frames plus geometry, playback rate, and frame
  count. It is deliberately **not** the image basis repeated per frame: a clip
  is one reviewed payload, not N independent images, and a per-frame basis
  would let a frame reordering or an fps relabel pass unnoticed.

Nothing existing in that module changed. `IMAGE_HASH_BASIS`,
`AUDIO_HASH_BASIS`, `TEXT_HASH_BASIS`, and every gate class are byte-for-byte
untouched, and `tests/test_multimodal_behavior_cases.py` still passes
unmodified (its "modality with no canonical basis" case uses `points3d`, which
still has none).

### 2. Probe generator — `src/neural_forge/native_video_probe.py` (new)

Schema `neural-forge-native-video-probe-v1`. 32x32 RGB24, 6 frames, 8 fps,
stdlib + NumPy only.

| category | answers | content |
|---|---|---|
| `motion_direction` | `left`/`right`/`up`/`down`/`none` | one square translating at a constant integer pixel step, or static |
| `shape_count` | `1`–`3` | N non-overlapping squares drifting together |
| `shape_color` | six palette names | one moving square; half the probes switch palette entry at the midpoint frame, and the prompt asks about the **final** frame |
| `shape_event` | `appears`/`disappears`/`persists` | presence thresholded on the midpoint frame |

**Determinism is structural and stricter than the audio lane's.** Every pixel
is an integer palette entry written into an integer array by integer slice
arithmetic: *no floating-point value is produced anywhere during generation*,
so there is no libm-rounding exposure at all (the audio lane had exactly one,
its `math.sin` wavetable). Jitter comes from a string-seeded
`random.Random(f"neural-forge-video-probe:{seed}:{category}:{index}")`, so no
probe depends on generation order. The palette digest is sealed into the
manifest as `palette_sha256` and re-checked on authentication, which is the
video analog of `wavetable_sha256` — a suite generated against a different
palette fails loudly instead of silently producing different frames under the
same seed.

**Container decision.** Clips are stored in `NFVID1`, a raw container defined
in this module: eight magic bytes plus six little-endian `uint32` header fields
(width, height, frames, fps, channels, payload bytes) followed by row-major
RGB24 frames in playback order. Rationale:

* it keeps the probe importable with **no new dependency** — Pillow is only in
  the `hf` extra, and the audio lane's stdlib+NumPy rule is the one being
  mirrored;
* it is self-describing, so `read_rgb24_video` recovers geometry and fps *from
  the file* and `authenticate_native_video_probe` compares them against the
  manifest rather than trusting it. This is the exact property
  `read_pcm16_wav` gives the audio lane through the WAV header, and it is why a
  bare `.npy` (no fps field) was rejected;
* no compression means byte-level determinism with no encoder version in the
  trust boundary.

`authenticate_native_video_probe` checks the manifest self-seal, the schema,
the hash basis, the palette digest, closed-set membership, path escapes, the
per-record container digest and byte count, and then **re-decodes and re-hashes
every clip** and cross-checks decoded frames/width/height/fps against the
record before returning.

### 3. Inspector — `src/neural_forge/dynamic_model_inspector.py` (additive)

`NativeVideoCircuitInspector` (`name="native-video"`, `priority=240`),
registered in `default_inspector_registry()` after `NativeAudioCircuitInspector`
and before the catch-all `HFCausalDecoderInspector`.

It emits one `CircuitKind.MODALITY_INPUT` node,
`input.video.rgb24_tubelet_encoder`, whose **input** port is `PortKind.MODALITY`
with `modality=canonical_modality("video")`, bound to a
`ModalityInterface(interface_id="video.rgb24_frames.input", direction="input",
representation="rgb24_frames")` — the representation the task specified. Its
output feeds `input.video.dense_projection` (`INTERFACE_PROJECTION`) and then
`input.video_text_embedding_fusion`, so
`{i.modality for i in graph.interfaces} == {"video", "text"}`.

`supports()` is marker-gated exactly like the audio inspector:

```
config.neural_forge_native_video_lane == NATIVE_VIDEO_LANE_MARKER
    and config.model_type == "native_video_chat"
    and hasattr(model, "video_encoder") and hasattr(model, "language_model")
```

Video-specific validation beyond the audio mirror:

* the front end must be a `Conv3d` whose kernel **and** stride equal
  `(temporal_patch_size, patch_size, patch_size)` — the temporal stride is what
  makes this a video front end rather than a per-frame image front end, so it
  is checked rather than assumed;
* `video_config.frame_encoding` must be `"uint8_normalized_rgb"`, the video
  form of the project's native-modality rule (no precomputed CLIP/feature front
  end);
* clip geometry must divide exactly into tubelets
  (`num_frames % temporal_patch_size`, `frame_height % patch_size`,
  `frame_width % patch_size` all zero), and the derived
  `video_token_count` must leave at least one learned position for text.

`NATIVE_VIDEO_LANE_CONFIG_KEY`, `NATIVE_VIDEO_LANE_MARKER`,
`NATIVE_VIDEO_MODEL_TYPE`, `NATIVE_VIDEO_FAMILY`, and
`NATIVE_VIDEO_REPRESENTATION` are exported so a real model can opt in without
copying string literals. No existing inspector was modified.

### 4. Target adapter — `src/neural_forge/dynamic_video_adapter.py` (new)

`NativeVideoTargetSynthesisAdapter` (`adapter_id
"neural_forge.video.native_video"`, version 1, priority 240). Clones the loaded
parent's own class and config, changes exactly
`decoder_config.intermediate_size`, constructs the blank directly on the parent
device/dtype under a forked RNG, re-inspects independently, and proves only the
coupled gate/up/down triplets changed shape with delta
`layers * 3 * hidden * (source - target)`.

`TARGET_MODALITIES = ("text", "video")` — canonical **sorted** order, because
`DynamicTargetRequest.__post_init__` sorts and lowercases the requested
modalities. (The audio lane's `("audio", "text")` is sorted too; for video the
sort puts text first, which is the one place a blind rename of the audio module
would have produced a silently non-matching adapter.)

As with audio, no entry was added to the compiler-reviewed structural-buffer
registry in `dynamic_target_synthesis.py`: this architecture registers **zero
nonzero runtime buffers** (learned absolute positions, no rotary `inv_freq`),
and the audit reports `nonzero_buffer_count == 0`. A real video parent with
rotary or normalization buffers will need one — see "Deferred".

### 5. Planner contract — `src/neural_forge/dynamic_fold_planner.py` (additive)

* `_native_video_model_contract(source, target)` — family `native_video_chat`,
  identical layer indices, identical global-node identities/ports/attributes,
  `{video, text}` interfaces on both sides via `MODALITY_VIDEO`/`MODALITY_TEXT`
  rather than literals, and the
  `source.tokenizer_fingerprint != target.tokenizer_fingerprint` rejection
  **kept unchanged**. It reuses the existing family-parameterized
  `_single_neuron_swiglu_layer_contract(..., family=...)` the audio wave added;
  no shared helper was modified.
* `_bundle_kind_contract`: new `"native_video_model"` branch requiring the
  complete model bundle, `{video, text}` interfaces, at least one
  `MODALITY_INPUT` node, metadata `native_rgb24_tubelet_encoder is True` and
  `external_frozen_video_encoder is False`, and an ordered residual contract
  per layer.
* Dispatch branch after the audio one emitting `method_id
  "native_video_swiglu_model"` with
  `fit_objective="video_conditioned_single_swiglu_neuron_merge_per_layer_with_exact_native_rgb24_front_end_copy"`,
  `selection_policy="sealed_calibration_trace_low_importance_anchor_regression"`,
  `promotion_gate="cached_parent_video_behavior_then_fresh_reload"`.

Nothing existing in the planner was modified; the NEO and audio contracts still
run through their own untouched code paths.

### 6. Method registry — `src/neural_forge/dynamic_fold_registry.py` (additive)

```
method_id          native_video_swiglu_model
family_id          native_video_coupled_swiglu_model
status             EXPERIMENTAL
source/target      ("native_video_model",)
evidence_channels  cached_parent_outputs, child_state, residual_endpoints,
                   video_conditioned_post_swiglu        (registry sorts these)
coupled_roles      q_k_v_o_cache, gate_up_post_swiglu_down,
                   all_layers_ordered_residual
```

`EXPERIMENTAL` is selectable, so the planner can emit it, while recording
honestly that this lane has one toy proof and no real-model evidence.

### 7. Fold system — `src/neural_forge/dynamic_video_fold.py` (new)

The full audio lane mirrored one-to-one: `VideoBehaviorRecord` /
`VideoParentBehaviorCache` / `load_video_parent_behavior_cache`,
`write_video_parent_output_cache`, `normalize_video_answer`,
`VideoBehaviorGateConfig`, `NativeVideoBehaviorGate`,
`build_native_video_fold_trace_cache` + `VideoTraceStatisticsCache`,
`NativeVideoLayerFoldSelection` + `_sparse_anchor_regression` +
`_selection_from_statistics`, `NativeVideoFoldCache` +
`derive_native_video_fold_cache`, `NativeVideoEvidenceProvider`,
`NativeVideoFoldOnlyInitializer`, `NativeVideoFoldTransaction`,
`NativeVideoSwiGLUFoldOperator`, `NativeVideoFoldAdmissionEvaluator`,
`NativeVideoStackResources` / `NativeVideoFoldStackAdapter`,
`write_native_video_dynamic_fold_receipt`, `video_lane_status()`.

The behavior gate consumes the shared multimodal case schema through
`record.as_case()` and `build_multimodal_behavior_gate_binding` — no cloned case
type — and reports `parent_forward_count == 0` in every report. The
trace-statistics cache reuses the modality-agnostic hook pattern verbatim: a
forward pre-hook on each layer's MLP down projection plus a forward hook on the
layer for residual endpoint energy. As in the audio lane, the statistics cache
carries **no fold selection**; selection lives in the separate
`neural-forge-native-video-fold-cache-v1` artifact that cites it through
`source_cache_key`.

Substantive differences from a mechanical rename of `dynamic_audio_fold.py` —
the complete list, so a reviewer can diff the rest as a no-op:

1. `VideoBehaviorRecord` carries `frames`/`width`/`height`/`fps` instead of
   `frames`/`sample_rate`/`channels`, and its `MediaReference.attributes` are
   `{fps, frames, height, width}`.
2. `normalize_video_answer` routes `shape_count` (not `tone_count`) through the
   digit matcher; the other three categories use the word matcher.
3. `NativeVideoFoldStackAdapter.target_modalities` and its `_validate` compare
   `("text", "video")` — sorted order, per item 4 above.
4. Docstrings.

---

## What was proved

`tests/test_native_video_executable_lane.py` — 13 tests — and
`tests/test_native_video_lane.py` — 22 tests. Both new.

The toy architecture (`_TinyNativeVideoChatModel`) is a learned strided
`Conv3d(3 -> 4, kernel=stride=(3, 16, 16))` over uint8-normalized RGB frames, a
bias-free `Linear(4 -> 8)` projection, and a 2-layer causal decoder (GQA 4
query / 2 key-value heads, head_dim 2, SwiGLU intermediate 6, vocab 12, learned
absolute positions). A 6-frame 32x32 clip becomes 8 video tokens
(`2 temporal x 2 x 2 spatial`) concatenated ahead of 6 prompt tokens, exactly as
the emitted graph declares. 10,412 parent parameters, of which 9,252 are the
video front end. No external encoder, no frozen weights, nothing from a hub.

`test_native_video_fold_cache_stack_behavior_artifact_and_receipt`:

1. Builds a sealed probe suite (`examples_per_category=4`, `seed=17`) → 16
   probes, 12 calibration / 4 heldout, every clip re-decoded and re-hashed
   under `VIDEO_HASH_BASIS`.
2. Caches parent behavior and authenticates it.
3. Captures video-conditioned traces — the trace runner feeds the **real
   decoded RGB24** of each calibration probe through the whole model. Asserted:
   trace calls equal the calibration probe ids exactly and are disjoint from
   the heldout ids; `trace_rows == (12 * (8 video tokens + 6 prompt tokens),) *
   2 layers`, i.e. `(168, 168)`.
4. Derives the fold cache (`candidate_count=3`, `anchor_count=2`) with **no
   further trace calls**, and shows a one-field manifest edit fails
   `NativeVideoFoldCache.open` on the seal.
5. Runs `AutomaticDynamicFoldingSystem.run(...)` — the full synthesize → plan →
   fold → admit → behavior-gate → publish → fresh-reload path.

Results asserted:

* `result.complete`; exactly one action, `method_id ==
  "native_video_swiglu_model"`.
* `published.artifact.fresh_reload_decision.accepted` is true and
  `fresh_reload_parameters_frozen is True` (the standalone loader returns
  `model.requires_grad_(False).eval()`).
* `behavior_gate.last_report["parent_forward_count"] == 0`, `failures == []`,
  and heldout `parent_correct_retained == parent_correct_total`. Every
  parent-correct answer, calibration and heldout, retained.
  `behavior_gate.evaluation_count == 4`.
* Exact parameter accounting: `expected_delta = 2 layers * 3 tensors * 8 hidden
  = 48`; child graph, request budget, and `artifact.exact_unique_parameters`
  all equal `10412 - 48 = 10364`, and the summed per-tensor element delta
  between the reloaded child and the parent equals 48.
* Every non-MLP tensor in the reloaded child — the whole video front end, every
  attention projection, both norms per layer, the embeddings, the final norm
  and the head — is `torch.equal` to the parent's.
* Per layer, `gate_proj`/`up_proj` equal the parent's rows index-selected by
  the survivors, and `down_proj` equals the parent's survivor columns plus the
  compensated removed column, recomputed independently in the test.
* The reloaded child still runs the real video path: logits are `(1, 14, 12)`
  and finite.
* The receipt self-hash verifies and records
  `parent_forward_count_during_fold_and_gates == 0`,
  `heldout_samples_used_for_selection == 0`, `optimizer_steps == 0`,
  `backward_calls == 0`, `fresh_reload_parameters_frozen is True`, and
  `source_cache_key == statistics.cache_key`.

`tests/test_native_video_lane.py` additionally pins: same-seed digest
reproducibility and cross-seed divergence; **content semantics** (the measured
centroid shift of the foreground actually matches the declared
`motion_direction` for all five answers, the presence pattern matches
`shape_event`, and the decoded frames decompose into exactly `shape_count`
8x8 squares); three independent manifest tamper paths; the foreign-palette
rejection; geometry/fps/frame-count binding in the hash; container
self-description plus truncation and magic-byte rejection; parent-output cache
tamper paths; the gate's retention, direct-answer, parent-relative, coverage,
and execution-contract behaviors; and trace-cache sealing, reload, tamper, and
zero-gradient/depth guards.

### Honest limit of the behavior proof

Stated in the e2e module docstring as well. The toy decoder cannot watch a
clip. As in the validated NEO vision and native audio proofs, the closed-set
answer returned by the gate's child runner is gated on a parent-derived marker
weight in the video front end, not on emergent competence. What is proved is
the **compiler contract**: the child executed its declared inference entrypoint
on the real decoded RGB24 payload for all 16 probes, the parent was never
forwarded, the child's answers matched the cached parent answers in every
parent-correct split x category cell, and the published artifact survives a
standalone fresh reload with frozen parameters and an exact parameter delta.
What is **not** proved is that folding preserves real video understanding.

---

## Commands run

Required-green set plus the two new suites and the modality/gate/synthesis
suites, at the final state of the tree:

```
& "C:/Users/cereal/Documents/ai training/neural-forge/.venv/Scripts/python.exe" -m pytest `
  tests/test_native_audio_executable_lane.py tests/test_native_audio_lane.py `
  tests/test_dynamic_model_inspector.py tests/test_dynamic_fold_planner.py `
  tests/test_dynamic_fold_registry.py tests/test_multimodal_behavior_cases.py `
  tests/test_dynamic_neo_native_vision.py tests/test_native_video_lane.py `
  tests/test_native_video_executable_lane.py tests/test_dynamic_modality_registry.py `
  tests/test_dynamic_behavior_gate.py tests/test_dynamic_target_synthesis.py `
  tests/test_dynamic_system.py -q
```
→ `187 passed in 24.81s`.

The seven required-green suites alone: `91 passed`. The two new suites alone:
`13 passed` (executable lane) and `22 passed` (lane), and `35 passed` in both
file orders, so neither depends on the other's module-scoped fixtures.

### Regression scope

Adding a method to `default_method_registry()` changes
`FoldMethodRegistry.fingerprint` and therefore `DynamicFoldPlan.fingerprint`
for **every** lane; adding an inspector changes `default_inspector_registry()`
for every model; adding a key to `CANONICAL_MEDIA_HASH_BASES` changes the
default basis resolution for `video` only. All three were checked
differentially rather than assumed harmless.

**Whole suite.** `pytest tests -q` → `16 failed, 9737 passed, 30 skipped in
3129.88s (0:52:09)`. None of the failures are in this wave's files. (13 of the
16 names survive in the captured log; pytest's progress output overwrote the
first summary lines. The differential run below has complete lists.)

**Differential.** Every test file importing `dynamic_fold_planner`,
`dynamic_fold_registry`, `dynamic_model_inspector`, `dynamic_system`,
`dynamic_behavior_gate`, `default_method_registry`, or
`default_inspector_registry` — 61 files, excluding this wave's two new ones —
was run twice from this working tree, once normally and once with `PYTHONPATH`
pointed at a copy of `src/` in which this wave's four edited modules were
restored from `HEAD` and its three new modules deleted. Only the library delta
differed; tests and the untracked `reports/` data were identical.

| run | result |
|---|---|
| current tree | `3 failed, 2462 passed, 3 skipped in 763s` |
| this wave's library delta reverted | `12 failed, 2453 passed, 3 skipped in 734s` |

The three current-tree failures appear **byte-identically** in the reverted
run, so they are unrelated to this wave:

```
tests/test_dynamic_qwen_compensated_adapter.py::test_tiny_real_qwen_replay_runs_through_automatic_system
tests/test_dynamic_qwen_global_codebook.py::test_automatic_qwen_global_codebook_fold_reuses_checkpoint_and_erases_seed
tests/test_run_dynamic_qwen3_codebook_fold.py::test_tiny_production_runner_publishes_and_freshly_gates_without_training
```

(Verified directly: the qwen global-codebook one fails with
`DynamicFoldArtifactError: Forbidden resume/cache/coordinator/checkpoint/...
reference in folded child artifact` with this wave's code not even imported.)

The nine **extra** failures in the reverted run are all
`tests/test_dynamic_lfm2_cpcf_v5_production.py::test_v5_parity_file_*` and are
an artifact of the method, not a regression in either direction: that suite
authenticates canonical source files by path, so loading the package from a
copied tree fails with `Lfm2CpcfCatalogError: V5 GPU parity source files are
missing or unsafe`. All nine pass in the current tree.

**Note on the working tree.** It already carried another wave's uncommitted
edits to `experiments/run_lfm2_dynamic_catalog.py`,
`src/neural_forge/hf_lfm2_on_policy_evidence_round.py`,
`hf_lfm2_dense_budget_model_fold.py`, `hf_lfm2_dynamic_closed_loop_adapter.py`,
`hf_lfm2_sealed_evidence_set.py`, and four `tests/test_hf_lfm2_*` /
`tests/test_run_lfm2_*` files. This wave touched none of them. A first
comparison against a clean `git worktree` of `ab3ec99a` was discarded because
those uncommitted edits made two test modules fail to import there
(`BRIDGE_MAPPED_CAPTURE_INPUT_MODE`); the isolated-baseline method above
replaces it and isolates exactly this wave's delta.

**`HEAD` advanced during this wave**, `ab3ec99a` → `09825ab2` ("Seal target
preimages into bridged rounds"), which committed most of those edits. It
touches **none** of the files this wave owns or reads as a template, so no
rebase was needed. The required-green set plus both new suites were re-run
afterwards: `126 passed in 17.22s`.

---

## Deferred — what a later wave still owes

1. **A production video-capable parent.** Not chosen, on purpose, exactly as
   `NOTES-1H.md` item 4 left the audio choice open. The lane stays
   model-agnostic through `trace_runner`, `child_runner`, `parent_runner`,
   `layer_resolver`, `projection_resolver`, and
   `inference_execution_contract`. The audio criteria carry over verbatim, plus
   one video-specific addition: a real parent that registers **nonzero runtime
   buffers** needs a compiler-reviewed entry in
   `dynamic_target_synthesis._builtin_structural_buffer_policy` keyed on
   `("neural_forge.video.native_video", 1)`. That file was outside this wave's
   ownership; synthesis will fail closed with "Nonzero runtime buffers lack an
   explicit structural policy" until it is added.
2. **The shared selector extraction that `NOTES-W3B.md` item 7 asked for.**
   That note said: if a third lane needs `_sparse_anchor_regression` /
   `_selection_from_statistics`, extract them into a modality-neutral module
   and have both lanes delegate, as one reviewable no-op diff. This *is* the
   third lane, and the extraction was **not** done, because it necessarily
   rewrites `dynamic_audio_fold.py` and `dynamic_neo_fold.py`, both read-only
   for this wave. The debt is now three copies of identical numerics. The
   extraction is still the right move and is now genuinely owed; it needs a
   wave that owns all three modules and can prove byte-identical cache keys
   before and after.
3. **Promote the method past `EXPERIMENTAL`.** One toy proof is not evidence.
   Promotion should follow a real parent clearing the same gate.
4. **A real answer decoder for the gate.** The marker-gated child runner is a
   scaffold. A production run needs a runner that reads the model's actual
   generated text and normalizes it through `normalize_video_answer`, which is
   already closed-set and side-effect free.
5. **Wider fold scopes.** The registered method removes exactly one neuron per
   layer (`source_neurons == target_neurons + 1`), matching NEO and audio.
   Multi-neuron, multi-layer, or front-end-touching scopes need their own
   contract, method id, and evidence.
6. **Temporal-specific evidence.** The trace cache conditions on video but its
   statistics are the modality-agnostic ones (`mean_abs`, `mean_square`,
   `sketch`, `down_norm`, `residual_endpoint_mean_square_l2`). Nothing yet
   distinguishes a neuron that carries *motion* from one that carries a static
   frame feature. A per-tubelet-position or inter-frame-difference statistic
   would be the first genuinely video-specific evidence channel, and would need
   a `TRACE_CACHE_SCHEMA` bump to `-v2`.
7. **Stale prose in files this wave does not own.**
   * `src/neural_forge/dynamic_circuit_ir.py` `_MODALITY_NOTES[MODALITY_VIDEO]`
     still reads "No lane emits this yet"; a lane now does. No test asserts the
     text, so this is documentation drift, not a failure.
   * `docs/MODALITY_FOUNDATION_2026-07-31.md` Section 2 still lists `video`
     among the modalities that "must supply `hash_basis` explicitly", and its
     hash-basis block predates `VIDEO_HASH_BASIS`. Both should be updated when
     that doc is next opened; `points3d` and `ocr` remain correct entries.
8. **`NOTES-1H.md` items 5, 6, 7 are still untouched** (migrating the NEO gate
   onto the shared case schema, the README raw-protocol section, and the
   `canonical_modality()` mechanical pass at existing emission sites). The new
   video inspector already routes both its modality strings through
   `canonical_modality()`.
9. **`MODALITY_OUTPUT` remains unimplemented output-side.** Unchanged. A video
   lane makes this more tempting to hand-wave, so to be explicit:
   `video_lane_status()["not_provided"]` says so, and this lane reviews video
   on the **input** side and text on the output side only.
