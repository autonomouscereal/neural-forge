# NOTES-1H — integration instructions for the modality expansion pack

Written 2026-07-31 by the modality-foundation wave. Design reference:
`docs/MODALITY_FOUNDATION_2026-07-31.md`.

Everything below touches files this wave deliberately did **not** modify. Each
item is self-contained: what to change, where, what it must satisfy, and how to
tell it worked. Items 1–4 are the audio lane; items 5–8 are cleanups that make
the foundation load-bearing.

Landed and already green (do not redo):

* `src/neural_forge/dynamic_circuit_ir.py` — canonical modality registry.
* `src/neural_forge/dynamic_behavior_gate.py` — `MediaReference`,
  `MultimodalWholeOutputCase`, `MultimodalCandidateOutput`, hash bases,
  `build_multimodal_behavior_gate_binding`.
* `src/neural_forge/raw_modality_tokens.py`,
  `src/neural_forge/hf_native_raw_tokens.py`,
  `src/neural_forge/hf_native_raw_eval.py` — protocol decision + shared
  `raw_token_lane_compatibility()`.
* `src/neural_forge/native_audio_probe.py`,
  `src/neural_forge/dynamic_audio_fold.py` — audio lane foundation.
* `tests/test_dynamic_modality_registry.py`,
  `tests/test_multimodal_behavior_cases.py`,
  `tests/test_native_audio_lane.py`.

---

## 1. Inspector: emit `MODALITY_INPUT` nodes with modality `"audio"`

**File:** `src/neural_forge/dynamic_model_inspector.py` (not owned by this wave)

**Why:** `CircuitKind.MODALITY_INPUT` is currently emitted by exactly one
inspector — `NeoNativeVisionCircuitInspector`, around lines 1413–1560 — and only
with modality `"image"`. Until an audio inspector exists, no real model can be
planned through the audio lane;
`tests/test_native_audio_lane.py::_audio_graph` hand-builds the graph shape the
inspector must produce.

**Do:**

1. Add `NativeAudioCircuitInspector` following the structure of
   `NeoNativeVisionCircuitInspector`. Emit, before the decoder nodes:
   * one `CircuitKind.MODALITY_INPUT` node whose **input** port is
     `PortKind.MODALITY` with `modality="audio"` (the waveform or feature
     entry), and whose **output** port carries the encoder result into the
     decoder residual stream;
   * one `ModalityInterface(direction="input", modality="audio",
     representation=...)` bound to that input port. Use
     `representation="pcm16"` for a raw-waveform entry, or the concrete feature
     name (e.g. `"log_mel"`) if the model front end consumes features.
2. Validate the modality string through
   `neural_forge.dynamic_circuit_ir.canonical_modality("audio")` rather than
   writing the literal. This is the opt-in registry path (see item 7).
3. Register the inspector in `default_inspector_registry()` (~line 3797),
   **before** `HFCausalDecoderInspector`, which is the catch-all.
4. Keep `graph.logical_parameter_count` exact: audio front-end parameters must
   be captured by the new node(s) or the graph will refuse to construct.

**Done when:** `inspect_model_circuits(model, ...)` on the chosen parent
returns a graph where
`{i.modality for i in graph.interfaces} == {"audio", "text"}` and
`graph.nodes_of_kind(CircuitKind.MODALITY_INPUT)` is non-empty.

## 2. Planner: audio fold contract

**File:** `src/neural_forge/dynamic_fold_planner.py`

**Why:** planning is gated on family/modality predicates. The vision analog is
`_neo_single_neuron_model_contract` (~line 923) plus its dispatch site (~line
2226).

**Do:**

1. Add `_native_audio_model_contract(source, target)` mirroring
   `_neo_single_neuron_model_contract`, with these substitutions:
   * `source.family == target.family == "<audio parent family id>"` (from item 4);
   * `{i.modality for i in source.interfaces} == {"audio", "text"}` and the same
     for the target — reuse `CANONICAL_MODALITIES` membership rather than
     literals where practical;
   * keep the `source.tokenizer_fingerprint != target.tokenizer_fingerprint`
     rejection **unchanged**. It is exactly the constraint that makes
     `NFHFRAW1` the canonical raw transport (item 6 / Section 3 of the design
     doc). Do not relax it for an audio lane.
2. Add the dispatch branch next to the NEO one with
   `method_id="<audio method id>"` and a `configuration` mapping that names the
   audio conditioning explicitly, e.g.
   `"fit_objective": "audio_conditioned_single_swiglu_neuron_merge_per_layer"`,
   `"selection_policy": "sealed_calibration_trace_low_importance_anchor_regression"`,
   `"promotion_gate": "cached_parent_audio_behavior_then_fresh_reload"`.
3. `evidence_requirements` on the action must equal the method's
   `evidence_channels` exactly (the planner asserts this at ~line 1900).

**Done when:** a parent/child graph pair from item 1 yields a one-action plan
with the audio method id and `validate_fold_action_evidence_channels` passes.

## 3. Registry: register the audio fold method

**File:** `src/neural_forge/dynamic_fold_registry.py` (~line 1155, next to
`neo_native_vision_swiglu_model`)

**Do:** add a `_method(...)` entry:

```
method_id:         "native_audio_swiglu_model"          (or the chosen name)
family_id:         "native_audio_coupled_swiglu_model"
status:            MethodStatus.EXPERIMENTAL            # promote after evidence
source/target:     ("native_audio_model",)
evidence_channels: ("audio_conditioned_post_swiglu",
                    "cached_parent_outputs",
                    "child_state",
                    "residual_endpoints")
coupled_roles:     ("q_k_v_o_cache",
                    "gate_up_post_swiglu_down",
                    "all_layers_ordered_residual")
reason:            one sentence naming audio conditioning and the exact
                   interface-copy guarantee
```

The channel names must match what the evidence provider publishes.
`AudioTraceStatisticsCache` currently seals `mean_abs`, `mean_square`,
`sketch`, `down_norm`, `residual_endpoint_mean_square_l2` per layer — i.e. it
already satisfies `audio_conditioned_post_swiglu` and `residual_endpoints`;
`cached_parent_outputs` is `AudioParentBehaviorCache`.

**Also needed at method-registration time** (all *new* code, no ownership
conflict): an evidence provider and operator pair modeled on
`NeoNativeVisionEvidenceProvider` / `NeoNativeVisionSwiGLUFoldOperator` in
`dynamic_neo_fold.py`, plus a stack adapter modeled on
`NeoNativeVisionFoldStackAdapter` with
`target_modalities = ("audio", "text")`. Put them in
`src/neural_forge/dynamic_audio_fold.py` (owned by this wave's design, safe to
extend) or a sibling `dynamic_audio_operator.py`.

**Note:** `dynamic_audio_fold.AudioTraceStatisticsCache` deliberately carries
**no fold selection**. Whoever adds the operator adds the selection rule and
should extend the cache body with a `selections` field and a
reproducible-from-statistics check, exactly as
`NeoNativeVisionFoldCache.verify()` does (it rebuilds every selection from the
sealed statistics and rejects a mismatch). Adding `selections` changes
`_body()` and therefore `cache_key`; bump `TRACE_CACHE_SCHEMA` to `-v2` when
you do.

## 4. Choose a real audio-capable parent model

**Not chosen on purpose.** The lane is model-agnostic through `trace_runner`,
`child_runner`, `parent_runner`, `layer_resolver`, `projection_resolver`, and
`inference_execution_contract`, so this decision is separable — but it must be
made before any real evidence run.

Selection criteria, in priority order:

1. **A decoder stack with observable per-layer MLP down projections.** The
   trace hooks require `layer.mlp.down_proj` (or an equivalent supplied through
   `projection_resolver`) and a layer module whose forward output is the
   residual endpoint.
2. **Audio in, text out.** The lane reviews text answers to closed-set audio
   questions. An audio-out model is not reviewable yet (Section 5 of the design
   doc).
3. **A tokenizer whose fingerprint the child preserves.** Folding must not
   touch vocabulary. This rules out any approach that adds audio tokens to the
   text vocabulary — see item 6.
4. **A wrapper entrypoint that can be sealed.** If the model is called through
   something other than `__call__` (NEO uses `chat`), pass a matching
   `BehaviorInferenceExecutionContract(entrypoint=..., required_module_path=...)`
   to `NativeAudioBehaviorGate`; the default is top-level forward.
5. **Baseline competence on the probes.** Run
   `write_audio_parent_output_cache()` first and inspect
   `AudioParentBehaviorCache`: the gate needs at least one parent-correct record
   in every split × category cell, or it rejects on coverage before any child
   runs. If a real parent cannot answer the synthetic probes, either the probe
   prompts need adapting to the model's answer style (keep the closed sets) or a
   different parent is needed. Do **not** relax the coverage policy to hide
   this — that is what `test_gate_rejects_incomplete_parent_correct_coverage`
   pins.

Two viable shapes: (a) an audio encoder + text decoder wrapper where only the
decoder folds; (b) a natively multimodal decoder whose audio front end is a
`MODALITY_INPUT` node copied exactly, as NEO copies its vision tower. Shape (b)
is the closer analog to the validated NEO lane and is the recommended target.

## 5. Migrate the NEO vision gate onto the shared case schema

**File:** `src/neural_forge/dynamic_neo_fold.py`

**Why:** NEO built a bespoke case type (`NeoVisionBehaviorRecord`, ~196–253) and
gate (`NeoNativeVisionBehaviorGate`, ~1938–2244) because no media-capable case
schema existed. One now does, and the audio lane consumes it.

**Do (mechanical, behavior-preserving):**

1. Replace the literal at line 52 with an import:
   `from .dynamic_behavior_gate import IMAGE_HASH_BASIS`. The constant is
   **byte-identical** to NEO's (pinned by
   `tests/test_multimodal_behavior_cases.py::test_hash_bases_match_the_documented_decoded_content_formulas`),
   so **no existing probe manifest or cache fingerprint changes.**
2. Add `NeoVisionBehaviorRecord.media` and `.as_case()` returning a
   `MediaReference(modality="image", ..., hash_basis=IMAGE_HASH_BASIS)` and a
   `MultimodalWholeOutputCase`, mirroring `AudioBehaviorRecord` in
   `dynamic_audio_fold.py`. Note NEO records do not currently carry a container
   digest or byte count; either add them to the record or leave
   `container_sha256=None` (the decoded-content digest is the binding one).
3. **Gate-version decision.** Switching
   `NeoNativeVisionBehaviorGate.__init__` from
   `cases=[record.identity() ...]` to
   `build_multimodal_behavior_gate_binding(cases=behavior_cache.cases(), ...)`
   changes `manifest_fingerprint` for every NEO gate. That is a real receipt
   change: published NEO receipts embed the old fingerprint. Either
   * bump `gate_version` 2 → 3 and re-publish, or
   * keep the current binding and only add `.as_case()` as an additive
     accessor.

   Do not silently change the manifest at the same `gate_version`.
4. `_normalize_direct_answer` and the NEO category set are unaffected. The
   `"ocr"` NEO category stays a **task category**; do not confuse it with the
   canonical `ocr` modality (design doc Section 1).

## 6. README: the raw-protocol decision is not documented

**File:** `README.md` (not owned by this wave)

The README documents `raw_modality_tokens` (`NFRT`, 266 added tokens) and does
not mention `hf_native_raw_tokens` (`NFHFRAW1`, zero added rows), which is the
protocol every folding lane can actually use. Add a short section stating the
decision and linking `docs/MODALITY_FOUNDATION_2026-07-31.md` Section 3. The
machine-readable source of truth is
`neural_forge.hf_native_raw_tokens.raw_token_lane_compatibility()`; quote it
rather than paraphrasing, so the README cannot drift from the code.

## 7. Optional: adopt `canonical_modality()` at existing emission sites

**Files:** `src/neural_forge/dynamic_model_inspector.py`,
`src/neural_forge/dynamic_neo_adapter.py`

Existing sites write `modality="text"` / `modality="image"` as literals.
Replacing them with `canonical_modality("text")` (or the `MODALITY_TEXT`
constant) is behavior-preserving — the strings are identical — and makes a typo
a construction error instead of a silently divergent graph. This is optional
and should be done as a single mechanical pass, not piecemeal, so it can be
reviewed as a no-op diff.

## 8. Nothing to do in these files

`dynamic_fold_executor.py`, `dynamic_closed_loop_fold.py`, `hf_lfm2_*.py`,
`experiments/`, `docs/CURRENT_FOLDING_GOAL.md`. The foundation adds no new
requirement to them. The audio lane reuses `FoldDecision`,
`FoldEvaluationContext`, `fingerprint_model_state`, and
`fingerprint_model_runtime_state` from the executor exactly as NEO does, with
no executor change.

---

## Known limitation carried forward

`CircuitKind.MODALITY_OUTPUT` remains **unimplemented output-side**: no
inspector emits it, and no gate can review a media output even though
`MultimodalWholeOutputCase` and `MultimodalCandidateOutput` now accept one.
Design doc Section 5 lists the five steps that would be required (inspector
emission, a sealed decoder contract in the gate, non-text regression notions,
a cached parent *media* output store, and operators that touch a modality
decoder). Do not describe the compiler as supporting output-side modality until
at least steps 1–3 exist.
