# NOTES-W7B — the native points3d lane and the output half of modality support

Written 2026-08-04 by the points3d wave, on top of the modality foundation
(`docs/MODALITY_FOUNDATION_2026-07-31.md`), the executable audio lane
(`c69d5a64`, `NOTES-W3B.md`), the executable video lane (`8c118efb`,
`NOTES-W5C.md`), and the shared selector extraction (`c8a1cac5`,
`NOTES-W6B.md`). Branch `codex/lrc-blank-compression`, base `8369f42e`.
Fold-only, nothing committed, nothing staged.

The **video lane was the template and was treated as read-only.** Everything in
Task 1 mirrors it deliberately; where the points3d lane diverges, the divergence
is stated and justified rather than left for a reader to discover by diffing.
Task 2 has no template — it is the first output-side review in the repository.

---

## 0. How this wave was executed — read this first

**Two agents implemented this wave in the same working tree at the same time.**
That is unusual enough to write down, because it is visible in the artifacts.

The overlap was detected mid-wave: files appeared on disk (first
`native_points3d_probe.py` and the `dynamic_behavior_gate` / `dynamic_circuit_ir`
arms, later `dynamic_points3d_fold.py`, the planner and registry arms, the
output module, and the three test files) at the moment they were about to be
written by the other agent. The working tree was **backed up before anything was
overwritten** — the pre-existing uncommitted diff, `HEAD`, and every affected
file are in the session scratchpad under `pre-w7b-backup/`.

The two implementations converged on the same architecture independently: the
same module split, the same schemas, the same `dynamic_points3d_output` boundary,
the same duck-typed stack resource. Where both produced a file, the on-disk
version was **read, evaluated, and verified by execution** rather than
overwritten. Nothing was clobbered.

Two consequences a reviewer will see:

1. **Two contract documents existed.** `docs/MODALITY_OUTPUT_DESIGN_2026-08-03.md`
   and `docs/MODALITY_OUTPUT_DESIGN_2026-08-04.md` described the same
   implementation from different angles and did not contradict each other. They
   have been **merged into `docs/MODALITY_OUTPUT_DESIGN_2026-08-04.md`** — the
   filename this wave's task specifies — taking the threat model (§7) and
   second-modality checklist (§8) from one draft and the prerequisite-closure
   table (§0) and executable-proof section (§6) from the other. The 08-03 file
   was removed and the merge is recorded in the surviving document's provenance
   note. `tests/test_native_points3d_output_lane.py` was repointed at the
   surviving path.
2. **Naming is not perfectly uniform.** The fold lane uses `Points3d`/`points3d`
   consistently, but the output module's helpers are `quantize_emitted_points3d`
   / `emitted_points3d_media` / `assert_graph_emits_points3d` rather than a
   `centroid`-centric spelling. That is cosmetic, it is internally consistent
   within each module, and it was left alone rather than churned late in the
   wave.

Everything below describes the tree as it now stands, and every claim in the
"What was proved" section was re-run against that tree.

---

## Task 1 — points3d, the fifth executable input modality

### 1. Canonical points3d content hash — `dynamic_behavior_gate.py` (additive)

`MediaReference` required an explicit `hash_basis` for `points3d` because no
canonical one existed. It now exists:

```
POINTS3D_HASH_BASIS =
  "sha256('POINTS3D:<dtype>:<points>:<scale_x>,<scale_y>,<scale_z>:<zero_x>,<zero_y>,<zero_z>\n'
   + little-endian quantized XYZ triples)"
```

* The basis binds the **dequantization**, not just the integers. Two clouds
  whose stored int16 values agree but whose scale or zero point differs are
  different geometry once dequantized, so a relabel fails closed instead of
  colliding. This is the points3d form of the video basis binding fps and frame
  count rather than pixels alone.
* `points3d_content_sha256(dtype, points, scale, zero_point, payload)` is the
  helper. It fails closed on an unsupported dtype, a non-positive point count, a
  non-three-axis quantization, a non-positive scale, a zero point outside int32,
  a non-bytes payload, and a payload length that disagrees with
  `points * 3 * dtype_width`.

**The fail-closed count check.** Registering the fifth basis made the table
`CANONICAL_MEDIA_HASH_BASES` cover five of six canonical modalities.
`_assert_canonical_media_hash_basis_coverage()` runs **at import** and requires
that every canonical modality is either based or listed in the new
`MODALITIES_WITHOUT_CANONICAL_HASH_BASIS` (today: `{"ocr"}` only), that the two
sets are disjoint, and that no basis string is duplicated. A future canonical
modality can no longer be added with its content binding left undecided — that
becomes an import-time failure instead of a surprise the first time some lane
builds a `MediaReference`.

`ocr` is deliberately left unbased and the reason is written down in the module:
its decoded payload is already covered by the image or text basis depending on
which side of the render a case binds, so a third basis would let one payload
seal two ways.

Nothing existing in that module changed. `IMAGE_HASH_BASIS`,
`AUDIO_HASH_BASIS`, `VIDEO_HASH_BASIS`, `TEXT_HASH_BASIS`, and every gate class
are byte-for-byte untouched.

### 2. Probe generator — `native_points3d_probe.py` (new)

Schema `neural-forge-native-points3d-probe-v1`. 60 int16 XYZ points per probe on
a lattice bounded by ±1024, stdlib + NumPy only.

| category | answers | content |
|---|---|---|
| `cluster_count` | `1`–`5` | N tight clusters separated along x by a constant integer gap |
| `dominant_axis` | `x`/`y`/`z` | one axis spread 400–480 lattice units, the others 40–80 |
| `shape_kind` | `line`/`plane`/`cluster`/`scatter` | 1, 2, 0, or 3 axes spread wide |
| `origin_side` | `negative`/`positive`/`straddle` | sign of the stored x extrema |

Four categories rather than the three the task named. `origin_side` was kept
because the gate's split×category cross-product coverage is stronger with four
cells per split, matching the video lane's shape exactly.

**Determinism is structural and held to the video lane's standard.** Every
coordinate is a signed integer produced by integer arithmetic on integer inputs;
no floating-point value participates in generating a point. Jitter comes from a
string-seeded `random.Random(f"neural-forge-points3d-probe:{seed}:{category}:{index}")`,
so no probe depends on generation order.

**The one float in the lane** is the declared per-axis quantization scale
`(0.25, 0.5, 0.125)` with zero point `(-8, 4, 0)`. It is a fixed module
constant, never computed, chosen as exactly-representable binary fractions so
its `repr` in the hash header is stable everywhere, and sealed into both the
manifest and every stored container. `authenticate_native_points3d_probe`
rejects a manifest generated against a different quantization — the points3d
analog of the video palette digest and the audio wavetable digest.

**Quantization contract reuse.** The contract is not invented here. It is the
one `raw_modality_tokens.PointQuantization` already defines for the raw-byte
codec — a `(point_count, 3)` integer array plus positive per-axis `scale` and
signed per-axis `zero_point`, dequantized as `(points - zero_point) * scale`.
The probe module imports that class and stores probes with that exact triple, so
one array can be handed to `RawModalityCodec.encode_points3d` and to this
container without either re-describing what the integers mean.
`test_container_round_trips_the_codec_quantization_contract` pins the
dequantized values.

**Container decision.** `NFPC1`: eight magic bytes, four little-endian `uint32`
header fields (point count, components, dtype code, payload bytes), three
little-endian `float64` axis scales, three little-endian `int32` axis zero
points, then row-major little-endian XYZ triples. 60 bytes of header. Same
rationale as `NFVID1`: no new dependency (no `open3d`, `trimesh`, or `plyfile`),
self-describing so `read_points3d_cloud` recovers the geometry *and the
quantization* from the file and authentication compares them against the
manifest rather than trusting it, and no compression so there is no encoder
version in the trust boundary.

**Content semantics are re-derived, not trusted.**
`test_synthesized_geometry_matches_its_declared_class` decodes every stored
cloud and recomputes the answer from the integers: per-axis ranges for
`dominant_axis` (with an unambiguity check that the widest axis is more than
twice the runner-up) and `shape_kind`, the sign of the x extrema for
`origin_side`, and single-link grouping over the separating axis for
`cluster_count`. All four closed sets are covered.

### 3. Inspector — `dynamic_model_inspector.py` (additive)

`NativePoints3dCircuitInspector` (`name="native-points3d"`, `priority=240`),
registered in `default_inspector_registry()` after `NativeVideoCircuitInspector`
and before the catch-all `HFCausalDecoderInspector`.

It emits one `CircuitKind.MODALITY_INPUT` node,
`input.points3d.quantized_point_encoder`, whose **input** port is
`PortKind.MODALITY` with `modality=canonical_modality("points3d")`, bound to a
`ModalityInterface(interface_id="points3d.quantized_xyz.input",
direction="input", representation="quantized_xyz")`. Its output feeds
`input.points3d.dense_projection` and then
`input.points3d_text_embedding_fusion`.

`supports()` is marker-gated exactly like the audio and video inspectors:

```
config.neural_forge_native_points3d_lane == NATIVE_POINTS3D_LANE_MARKER
    and config.model_type == "native_points3d_chat"
    and hasattr(model, "point_encoder") and hasattr(model, "language_model")
```

**The front end is the one genuinely new shape.** A point set has no grid, so
the front end cannot be a convolution. It is the standard point-set shape: one
shared `Linear(3 -> K*D)` applied to every point independently, then a
**symmetric** reduction over the point axis into K context tokens. Permutation
invariance is what makes this a point-*set* encoder rather than a sequence
encoder over an accidental storage order, so the declared reduction is validated
against `NATIVE_POINTS3D_POOLINGS` rather than assumed — the direct analog of
the video inspector validating that its `Conv3d` stride is genuinely temporal.
`test_native_points3d_front_end_is_permutation_invariant` shuffles a cloud and
asserts bit-identical front-end output, so the declaration is checked against
behavior and not only against a config string.

Points3d-specific validation beyond the video mirror:

* `num_coordinates` must be exactly 3;
* `points_config.point_encoding` must be `"int16_quantized_xyz"`, the geometry
  form of the project's native-modality rule (no precomputed PointNet++/CLIP
  feature front end);
* `point_linear.out_features` must equal `num_tokens * hidden_size` exactly —
  the points3d analog of "clip geometry must divide exactly into tubelets";
* `num_tokens + 1` must leave a learned position for text.

`NATIVE_POINTS3D_LANE_CONFIG_KEY`, `NATIVE_POINTS3D_LANE_MARKER`,
`NATIVE_POINTS3D_MODEL_TYPE`, `NATIVE_POINTS3D_FAMILY`,
`NATIVE_POINTS3D_REPRESENTATION`, `NATIVE_POINTS3D_EMITTED_REPRESENTATION`, and
`NATIVE_POINTS3D_POOLINGS` are exported. No existing inspector was modified.

### 4. Target adapter — `dynamic_points3d_adapter.py` (new)

`NativePoints3dTargetSynthesisAdapter` (`adapter_id
"neural_forge.points3d.native_points3d"`, version 1, priority 240). Clones the
loaded parent's own class and config, changes exactly
`decoder_config.intermediate_size`, constructs the blank directly on the parent
device/dtype under a forked RNG, re-inspects independently, and proves only the
coupled gate/up/down triplets changed shape with delta
`layers * 3 * hidden * (source - target)`.

`TARGET_MODALITIES = ("points3d", "text")` — canonical **sorted** order, because
`DynamicTargetRequest.__post_init__` sorts the requested modalities. As with
audio and video, no entry was added to the compiler-reviewed structural-buffer
registry: this architecture registers zero nonzero runtime buffers and the audit
reports `nonzero_buffer_count == 0`. A real geometry parent with rotary or
normalization buffers will need one — see "Deferred".

### 5. Planner contract — `dynamic_fold_planner.py` (additive)

* `_native_points3d_model_contract(source, target)` — family
  `native_points3d_chat`, identical layer indices, identical global-node
  identities/ports/attributes, `{points3d, text}` interfaces on both sides via
  `MODALITY_POINTS3D`/`MODALITY_TEXT` rather than literals, and the
  tokenizer-fingerprint rejection kept unchanged. It reuses the existing
  family-parameterized `_single_neuron_swiglu_layer_contract(..., family=...)`;
  no shared helper was modified.
* **One check the earlier lanes could not need:** it also requires
  `source.interfaces == target.interfaces`, i.e. full
  `(id, modality, direction, node, port, representation)` identity rather than
  the modality *set*. A points3d model may own an emitted-centroid
  `MODALITY_OUTPUT` interface, and folding must never change whether — or what
  — a child emits. Folding an emitting parent into a silent child is therefore
  unplannable, which
  `test_planner_refuses_to_fold_an_emitting_parent_into_a_silent_child` pins.
* `_bundle_kind_contract`: new `"native_points3d_model"` branch requiring the
  complete model bundle, `{points3d, text}` interfaces, at least one
  `MODALITY_INPUT` node, metadata `native_quantized_point_set_encoder is True`
  and `external_frozen_points3d_encoder is False`, an ordered residual contract
  per layer, and — new — that `emits_quantized_centroid` is a real boolean that
  *agrees with the graph*: a model claiming to emit without a
  `MODALITY_OUTPUT` node, or carrying one without declaring it, is not
  plannable.
* Dispatch branch after the video one emitting `method_id
  "native_points3d_swiglu_model"`, whose configuration carries
  `emits_modality_output` and selects one of two promotion-gate strings
  accordingly.

Nothing existing in the planner was modified; the NEO, audio, and video
contracts still run through their own untouched code paths.

### 6. Method registry — `dynamic_fold_registry.py` (additive)

```
method_id          native_points3d_swiglu_model
family_id          native_points3d_coupled_swiglu_model
status             EXPERIMENTAL
source/target      ("native_points3d_model",)
evidence_channels  cached_parent_emitted_media, cached_parent_outputs,
                   child_state, points3d_conditioned_post_swiglu,
                   residual_endpoints            (registry sorts these)
coupled_roles      q_k_v_o_cache, gate_up_post_swiglu_down,
                   all_layers_ordered_residual
```

`cached_parent_emitted_media` is the first output-side evidence channel in the
registry. It names what the *gate* replays, not a new trace capture — see the
Task 2 evidence decision below.

### 7. Fold system — `dynamic_points3d_fold.py` (new)

The video lane mirrored one-to-one: `Points3dBehaviorRecord` /
`Points3dParentBehaviorCache` / `load_points3d_parent_behavior_cache`,
`write_points3d_parent_output_cache`, `normalize_points3d_answer`,
`Points3dBehaviorGateConfig`, `NativePoints3dBehaviorGate`,
`build_native_points3d_fold_trace_cache` + `Points3dTraceStatisticsCache`,
`NativePoints3dLayerFoldSelection` + `_selection_from_statistics`,
`NativePoints3dFoldCache` + `derive_native_points3d_fold_cache`,
`NativePoints3dEvidenceProvider`, `NativePoints3dFoldOnlyInitializer`,
`NativePoints3dFoldTransaction`, `NativePoints3dSwiGLUFoldOperator`,
`NativePoints3dFoldAdmissionEvaluator`, `NativePoints3dStackResources` /
`NativePoints3dFoldStackAdapter`,
`write_native_points3d_dynamic_fold_receipt`, `points3d_lane_status()`.

**The shared selector is used, not cloned.** `NOTES-W6B.md` extracted the
heldout-blind low-importance selector and its sparse anchor regression into
`dynamic_modality_fold_selection.py` and had the NEO, audio, and video lanes
delegate. This lane delegates from the start: it declares one
`ModalityFoldSelectionVocabulary(error_type=DynamicPoints3dFoldError,
subject_label="Native points3d", selection_label="native points3d", ...)` and
calls `normalize_sparse_anchor_selection`, `sparse_anchor_regression`,
`selection_from_statistics`, `sparse_compensation_terms`, and `survivors_for`.
No numerics were copied, so the fourth lane added zero lines of duplicated
selection mathematics. `tests/test_dynamic_modality_fold_selection.py` is
unmodified and still green.

Substantive differences from a mechanical rename of `dynamic_video_fold.py` —
the complete list, so a reviewer can diff the rest as a no-op:

1. `Points3dBehaviorRecord` carries
   `point_count`/`components`/`dtype`/`scale`/`zero_point` instead of
   `frames`/`width`/`height`/`fps`, and its `MediaReference.attributes` are
   `{components, dtype, point_count, scale_x, scale_y, scale_z, zero_x, zero_y,
   zero_z}` — the quantization is part of what the reference describes because
   it is part of what the digest binds.
2. `normalize_points3d_answer` routes `cluster_count` (not `shape_count`)
   through the digit matcher; the other three categories use the word matcher.
3. `NativePoints3dStackResources.behavior_gate` is **duck-typed** rather than
   pinned to `NativePoints3dBehaviorGate`, so the Task 2 emitted-output gate can
   drive the same published stack. It is still validated: the gate must expose a
   `Points3dParentBehaviorCache` and an `evaluate()`, and its parent-output
   provenance must match the fold cache's.
4. `write_native_points3d_dynamic_fold_receipt` duck-types the gate on
   `last_report` + `behavior_gate_manifest_fingerprint` for the same reason.
5. `NativePoints3dFoldStackAdapter.target_modalities` compares
   `("points3d", "text")` — sorted order.
6. Docstrings.

---

## Task 2 — MODALITY_OUTPUT

`docs/MODALITY_OUTPUT_DESIGN_2026-08-04.md` is the contract, with every claim
grounded at `file:line`. It supersedes Section 5 of the modality foundation doc,
which listed five prerequisites for output-side support; four are now
implemented and the fifth is explicitly still deferred.

### Design decisions, and why

**Closed sets first, and only.** The reviewed emission is a quantized XYZ
centroid — one three-integer point on the same lattice, under the same
`POINTS3D_HASH_BASIS`, as the cloud the model consumed. On a closed lattice
"did the child emit the same geometry?" is a digest comparison. On open
generation there is no closed reference, so exact match is meaningless and any
threshold is a perceptual judgement in disguise. Open generation is routed to
**manual review** and that policy is returned by
`emitted_output_contract_status()` rather than left to a reader's assumption.

**A decode contract is mandatory, and is treated exactly like the inference
contract.** `MediaReference` binds decoded content, and a raw head output is not
decoded content — between the tensor and the digest sits a rounding/clipping/
codebook decision that changes the answer. `quantize_emitted_points3d()` is that
decision; its `BehaviorCallableIdentity` is sealed into the gate manifest as its
own component (`neural-forge-emitted-media-decode-contract-v1`) recording
`comparison: decoded_content_digest_exact_match`, `tolerance: none`, and
`open_generation_reviewed_here: False`. Without it, "the child emitted the same
media" is unverifiable: two runs could decode one tensor differently and both
report retention.

**Decode failure is a reviewable regression, not a crash.**
`Points3dEmissionDecodeError` is caught per row, recorded as `decode_failed`,
and thresholded by `maximum_decode_failures`. A child whose head emits NaN or
leaves the lattice has regressed in a way the gate must report.

**Output-side fold evidence: the head is evidence-bearing but not fold-scoped.**
The fold removes one coupled SwiGLU neuron per decoder layer and copies the
emitting head byte-exact, so the emission changes only through the decoder
residual it reads — which is exactly what `residual_endpoint_mean_square_l2`,
already captured by the input-side trace cache, measures. Sealing a second trace
capture keyed on the head would freeze a policy this lane has not validated,
which is the same reason `Points3dTraceStatisticsCache` carries no fold
selection. The claim is made checkable rather than assumed by the planner's full
interface-identity comparison and by the test that asserts
`centroid_head.weight` is `torch.equal` to the parent's after a fresh reload.

**Non-text regression notions.** `parent_empty`, `fragment_like`, and
`eos_margin` are text ideas and are not reused.
`Points3dEmittedOutputGateConfig` declares `minimum_emitted_retention` (default
`1.0`), `maximum_decode_failures` (`0`), `maximum_empty_emissions` (`0`), plus
per-split and per-split×category enforcement and parent-emission coverage.

**The sealed reference-output store.**
`write_points3d_parent_emission_cache()` stores every emitted payload as its own
re-decodable `NFPC1` container and indexes **both** digests `MediaReference` has
always carried: the container digest (proves the file is unchanged) and the
decoded-content digest (proves it still means the same thing). Both are
re-verified on every load. This is the cache writer the schema supported and
nobody had written. Child emissions are deliberately **not** stored: they are
content-bound but not container-bound, so their references carry
`container_sha256=None` and the explicit `memory://` scheme, because inventing a
filename for a tensor that only ever existed in memory would let a review claim
a container verification it never performed.

**Publication and fresh reload need no new machinery, which is the point.**
Because `NativePoints3dStackResources` duck-types the gate, the emitted-output
gate drives the same `DynamicFoldSystemStack` as both `final_evaluator` and
`fresh_reload_evaluator` with **no second stack adapter**. The additional
output-side verification is that the reloaded artifact is still an *emitting*
artifact: `assert_graph_emits_points3d()` re-inspects the reloaded published
child and fails closed unless it really carries a points3d `MODALITY_OUTPUT`
node and a `direction="output"` points3d interface — checked against the IR, not
against a config flag.

**`emitted_media_interfaces()` in the IR.** Every lane's text head already
registers `direction="output"`, so "does this model have an output interface?"
answers yes for every model in the repository and means nothing. The question
that matters — output-direction interfaces whose modality is not text — is now
spelled once in `dynamic_circuit_ir.py` instead of being filtered by hand at
each call site.

### The smallest executable proof

The toy model is the *same* `_TinyNativePoints3dChatModel` the input-side proof
folds, with one config flag flipped. `emit_centroid=True` adds a bias-free
`Linear(8 -> 3)` centroid head reading the same normalized residual the text
head reads: 1320 parameters become 1344, and the graph gains one node and one
interface (19/3 → 20/4).

`test_emitting_child_folds_publishes_and_reloads_under_output_review` runs the
full synthesize → plan → fold → admit → **emitted-output gate** → publish →
fresh-reload path and asserts:

* the plan carries `emits_modality_output is True` and the emission promotion
  gate string, and `centroid_head.weight` is owned by the atomic action;
* a child that lost the marker **fails** emitted review, so the accepted run is
  evidence rather than a tautology;
* `result.complete`, exact parameter delta 48 (`2 layers * 3 tensors * 8
  hidden`), `fresh_reload_decision.accepted`, `fresh_reload_parameters_frozen`;
* the final and fresh-reload reports are both `review_side == "output"` with
  `failures == []`, `parent_forward_count == 0`, and heldout
  `emitted_retention == 1.0`;
* the reloaded child still passes `assert_graph_emits_points3d`, still declares
  `emits_quantized_centroid`, has a byte-identical `centroid_head.weight`, and
  really emits a finite 3-vector through the real head on a real decoded cloud;
* the receipt self-hash verifies and records
  `behavior_report.review_side == "output"`,
  `generative_quality_reviewed is False`,
  `parent_forward_count_during_fold_and_gates == 0`, `optimizer_steps == 0`,
  `backward_calls == 0`.

### Honest limit of the output proof

Stated in the suite docstring, in the gate's own accept reason, in
`report["generative_quality_reviewed"] is False`, and in
`points3d_lane_status()["not_provided"]`.

The toy centroid head is randomly initialized and cannot compute a centroid.
Exactly as the NEO vision, native audio, and native video proofs gate their
*text* answers, the reviewed emission is gated on a parent-derived marker
surviving the fold. What is proved is the output-side **contract**: a real
`MODALITY_OUTPUT` node in a real compiled graph, a sealed reference-output store
whose containers and decoded payloads both re-verify, a fingerprinted decode
contract sealed as a manifest component, parent-relative retention by exact
decoded-content digest with split and split×category coverage, decode failure
and emitted drift each rejected, and the replay of all of it through publication
and a standalone fresh reload. What is **not** proved is that folding preserves
— or that the model has — any generative quality whatsoever.

---

## What was proved, and the commands

```
& "C:/Users/cereal/Documents/ai training/neural-forge/.venv/Scripts/python.exe" -m pytest `
  tests/test_native_points3d_lane.py `
  tests/test_native_points3d_executable_lane.py `
  tests/test_native_points3d_output_lane.py -q -p no:randomly
```
→ `59 passed` (24 lane, 17 executable lane, 18 output lane).

Required-green set plus the modality/gate suites:

```
& "C:/Users/cereal/Documents/ai training/neural-forge/.venv/Scripts/python.exe" -m pytest `
  tests/test_native_audio_executable_lane.py tests/test_native_audio_lane.py `
  tests/test_native_video_lane.py tests/test_native_video_executable_lane.py `
  tests/test_dynamic_neo_native_vision.py tests/test_dynamic_model_inspector.py `
  tests/test_dynamic_fold_planner.py tests/test_dynamic_fold_registry.py `
  tests/test_dynamic_modality_fold_selection.py tests/test_dynamic_modality_registry.py `
  tests/test_multimodal_behavior_cases.py -q -p no:randomly
```

Lint: `ruff check` clean over the wave's files.

### The one required-green test that had to change

`tests/test_multimodal_behavior_cases.py::test_media_reference_requires_an_explicit_basis_without_a_canonical_one`
used `points3d` as its "modality with no canonical basis" example — which
`NOTES-W5C.md` §1 explicitly noted was still true after the video wave. Adding
`POINTS3D_HASH_BASIS` makes it false. The example moved to `ocr`, the one
canonical modality that still deliberately owns none, and the test now asserts
`MODALITIES_WITHOUT_CANONICAL_HASH_BASIS == {"ocr"}` so the set is declared
rather than counted by hand. A second test,
`test_points3d_now_resolves_a_canonical_basis`, pins the new default and that
every canonical modality is now either based or explicitly unbased.

This is the only behavioral change to an existing test in the wave, and it is a
correction rather than a relaxation.

### Regression scope

Three registry-level changes have repository-wide blast radius and were treated
as such:

* adding a method to `default_method_registry()` changes
  `FoldMethodRegistry.fingerprint` and therefore `DynamicFoldPlan.fingerprint`
  for **every** lane;
* adding an inspector changes `default_inspector_registry()` for every model;
* adding a key to `CANONICAL_MEDIA_HASH_BASES` changes default basis resolution
  for `points3d` only — and the new import-time coverage check now runs for
  every importer of `dynamic_behavior_gate`.

`NOTES-W5C.md` §"Regression scope" records three pre-existing failures unrelated
to the modality lanes
(`test_dynamic_qwen_compensated_adapter.py::test_tiny_real_qwen_replay_runs_through_automatic_system`,
`test_dynamic_qwen_global_codebook.py::test_automatic_qwen_global_codebook_fold_reuses_checkpoint_and_erases_seed`,
`test_run_dynamic_qwen3_codebook_fold.py::test_tiny_production_runner_publishes_and_freshly_gates_without_training`).
Those were **not** re-verified by this wave; see "Deferred" item 7.

---

## Deferred — what a later wave still owes

1. **A production geometry-capable parent.** Not chosen, on purpose, exactly as
   the audio and video waves left theirs open. The lane stays model-agnostic
   through `trace_runner`, `child_runner`, `parent_runner`, `layer_resolver`,
   `projection_resolver`, and `inference_execution_contract`, plus
   `parent_emitter` and `child_emitter` on the output side. A real parent that
   registers **nonzero runtime buffers** needs a compiler-reviewed entry in
   `dynamic_target_synthesis._builtin_structural_buffer_policy` keyed on
   `("neural_forge.points3d.native_points3d", 1)`; synthesis fails closed with
   "Nonzero runtime buffers lack an explicit structural policy" until it exists.
2. **Output-side fold operators.** Item 5 of the foundation doc's list, still
   open. Every operator folds decoder-internal circuits; none rewrites an
   emitting head or a media codebook. That needs its own contract, method id,
   and — unlike this wave — a trace channel measuring the head's own
   sensitivity.
3. **A second emitting modality.** The output contract is modality-neutral but
   only points3d has been executed. Audio is the natural second: it already has
   a canonical basis, and a codec head is the media-codebook shape §1 of the
   design doc describes.
4. **Open-ended generation review.** Manual today. Automating it needs either a
   fingerprinted learned judge — with the same identity and manifest treatment
   the decoder now gets — or a reference-free metric whose thresholds are
   themselves reviewed evidence. Perceptual/distributional metrics (Chamfer, EMD
   for geometry) are tolerance-based and each needs a defensible threshold.
5. **Promote the method past `EXPERIMENTAL`.** One toy proof is not evidence.
   Promotion should follow a real parent clearing the same gates.
6. **A real answer decoder and a real emitter for the gates.** The marker-gated
   child runner and child emitter are scaffolds. A production run needs a runner
   that reads the model's actual generated text through
   `normalize_points3d_answer` (already closed-set and side-effect free) and an
   emitter that returns the head's own output through
   `quantize_emitted_points3d` (already the declared decode contract).
7. **A whole-suite regression sweep.** This wave ran the required-green set, the
   modality/gate suites, and its own three new suites. It did **not** run
   `pytest tests -q` (≈52 minutes at last measurement) nor repeat the
   differential reverted-tree comparison `NOTES-W5C.md` performed. The three
   pre-existing qwen failures recorded there were not re-checked and should not
   be assumed either fixed or still failing.
8. **Points-specific evidence.** The trace cache conditions on geometry but its
   statistics are the modality-agnostic five. Nothing yet distinguishes a neuron
   carrying *spatial extent* from one carrying cluster identity. A per-token or
   inter-cluster statistic would be the first genuinely geometric evidence
   channel and would need a `TRACE_CACHE_SCHEMA` bump to `-v2`.
9. **Stale prose in files this wave did not fully own.**
   `docs/MODALITY_FOUNDATION_2026-07-31.md` §2 still lists `points3d` among the
   modalities that "must supply `hash_basis` explicitly" and its §5 still says
   `MODALITY_OUTPUT` is dead code. Both are now false. The new design doc states
   that it supersedes §5, and `dynamic_circuit_ir._MODALITY_NOTES[MODALITY_POINTS3D]`
   was updated in this wave, but the foundation doc itself was left for whoever
   next opens it. `NOTES-W5C.md` item 7 recorded the same debt for `video` and
   it is still outstanding.
