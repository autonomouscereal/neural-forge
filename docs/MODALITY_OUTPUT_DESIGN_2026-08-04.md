# MODALITY_OUTPUT — the contract for children that emit non-text media

Written 2026-08-04 by the points3d wave. Branch `codex/lrc-blank-compression`,
base `8369f42e`. Executable half: `tests/test_native_points3d_output_lane.py`.

This closes the limitation `docs/MODALITY_FOUNDATION_2026-07-31.md:311-345`
recorded: *"`CircuitKind.MODALITY_OUTPUT` exists in the IR and no inspector
emits it… this repo can review media on the input side and text on the output
side."* Both halves of that sentence are now false. This document is the
contract; every claim below cites the code that implements it.

It is deliberately written against **one** modality (`points3d`) because that is
the only one with an executable proof today. Sections 2–7 are modality-neutral;
section 8 says exactly what a second emitting modality would have to supply.

> **Provenance.** This wave was implemented by two agents in the same working
> tree at the same time (see `NOTES-W7B.md` §0) and each drafted a contract
> document. They described the same implementation and did not contradict each
> other. This file is the merge: the threat model (§7) and second-modality
> checklist (§8) come from one draft, the prerequisite-by-prerequisite closure
> table (§0) and the executable-proof section (§6) from the other. The
> superseded `docs/MODALITY_OUTPUT_DESIGN_2026-08-03.md` was removed.

---

## 0. What "output-side" means, precisely

`ModalityInterface.direction` has always allowed `"output"`
(`src/neural_forge/dynamic_circuit_ir.py:483-488`), and every lane already has
one: the text head's `text.logits.output`
(`src/neural_forge/dynamic_model_inspector.py:4224-4232`). So *"does this model
have an output interface?"* is the wrong question — every model does.

The question is *"does this model emit **media**?"*, and the single spelling of
it is `emitted_media_interfaces()`
(`src/neural_forge/dynamic_circuit_ir.py:776-793`): every `direction="output"`
interface whose modality is not `text`. It is a function rather than a
convention so callers cannot each invent their own filter.

Five design decisions follow from this, and they are the rest of the document:

| # | decision | where |
|---|---|---|
| 1 | An emitting head is a **sibling** of the text head, not a post-processor of it. | §1 |
| 2 | Emitted media is bound by **decoded content**, never by container or by tensor. | §2 |
| 3 | Evidence for output-side review is a **sealed parent emission store**, one container per emitted payload. | §3 |
| 4 | Review is **parent-relative retention on a closed answer set**; open generation is routed to manual review. | §4 |
| 5 | Publication and fresh reload must **replay the emitted review**, not just the text review. | §5 |

### Closing the foundation doc's five prerequisites

`MODALITY_FOUNDATION_2026-07-31.md` §5 listed five things output-side support
would require, "in order". Four are now implemented and the fifth is explicitly
still deferred:

| Foundation doc's step | Status | Where |
|---|---|---|
| 1. Inspector emission | **implemented** | `src/neural_forge/dynamic_model_inspector.py:4236-4290` |
| 2. A decode contract in the gate | **implemented** | `src/neural_forge/dynamic_points3d_output.py:175-226`, sealed at `:904-913` |
| 3. Parent-relative metrics that are not text metrics | **implemented** | `src/neural_forge/dynamic_points3d_output.py:717-802` |
| 4. A sealed reference-output store | **implemented** | `src/neural_forge/dynamic_points3d_output.py:533-611` |
| 5. Fold operators that touch the output head | **still deferred** | §3.2 and §8 |

The two schema halves that existed with no producer now have one:
`MultimodalWholeOutputCase.parent_output` may be a `MediaReference`
(`src/neural_forge/dynamic_behavior_gate.py:1199`), produced at
`src/neural_forge/dynamic_points3d_output.py:381-401`; and
`MultimodalCandidateOutput.output` may be a `MediaReference`
(`src/neural_forge/dynamic_behavior_gate.py:1391`), produced per reviewed
rollout at
`src/neural_forge/dynamic_points3d_output.py:1043-1049`. Consequently
`multimodal_case_coverage()` (`src/neural_forge/dynamic_behavior_gate.py:1484`)
reports a **nonzero `media_output_count`** for the first time.

---

## 1. Inspector emission

### 1.1 What the node looks like

`NativePoints3dCircuitInspector`
(`src/neural_forge/dynamic_model_inspector.py:3217`) emits the repository's
first `CircuitKind.MODALITY_OUTPUT` node at
`src/neural_forge/dynamic_model_inspector.py:4242-4243`, with:

* one input port on `PortKind.NORMALIZED_RESIDUAL` — the *same* normalized
  residual the text head reads (`dynamic_model_inspector.py:4180-4192` vs
  `:4246-4254`), wired by its own `EdgeKind.DATA` edge from `final_norm`
  (`:4272-4278`);
* one output port on `PortKind.MODALITY` carrying `modality="points3d"`
  (`:4255-4264`);
* a `direction="output"` `ModalityInterface` whose representation is
  `NATIVE_POINTS3D_EMITTED_REPRESENTATION` (`:4279-4288`,
  `dynamic_model_inspector.py:3213`).

**Sibling, not chain.** The head reads the final norm directly rather than the
logits. An emitted-media head is an additional *output of the decoder*; making
it consume text would have made emitted media a function of a sampled string
and put a decoding policy inside the graph.

### 1.2 The declaration must match the graph, in both directions

`emit_centroid` is read from the config at
`src/neural_forge/dynamic_model_inspector.py:3470`, and both mismatches fail
closed:

* declared but absent/misshaped/biased head → error (`:3472-3481`);
* **head present but not declared** → error (`:3482-3487`), with the reason
  spelled out: *"an unreviewed emission path is not an inspectable graph."*

That second direction is the one that matters. A model that can emit media
through a tensor no interface names is a model whose emissions no gate will
ever see.

The graph then carries `emits_quantized_centroid` in metadata
(`dynamic_model_inspector.py:4310`) so downstream contracts can ask the
question without re-walking nodes.

### 1.3 Additivity

A model with `emit_centroid=False` compiles to exactly the graph the input-side
lane shipped: no `MODALITY_OUTPUT` node, no extra interface, unchanged metadata
key value. Proved by `test_a_silent_points3d_model_emits_no_modality_output_node`
and by the silent-model assertions in
`tests/test_native_points3d_executable_lane.py:515`.

---

## 2. Content binding for emitted media

### 2.1 The hash basis, and why the registry now fails closed

`MediaReference` (`src/neural_forge/dynamic_behavior_gate.py:980`) binds
decoded content, and when a caller omits `hash_basis` it falls back to
`CANONICAL_MEDIA_HASH_BASES` (`dynamic_behavior_gate.py:1013-1019`). Before
this wave a modality missing from that table produced a runtime error the first
time some lane built a reference — arbitrarily far from the omission.

`points3d` now has a basis (`dynamic_behavior_gate.py:89-92`) and the table has
an import-time coverage check
(`_assert_canonical_media_hash_basis_coverage`, `dynamic_behavior_gate.py:114`,
invoked at `:147`): every canonical modality either owns a basis or is listed in
`MODALITIES_WITHOUT_CANONICAL_HASH_BASIS` (`:103`), the counts must agree, the
two sets must be disjoint, and no two modalities may share a basis string.
Adding a canonical modality without deciding its binding now breaks the import,
not a downstream lane.

`ocr` is the one deliberately unbased modality, and the reason is written next
to it: it is a transport spelling of rendered-text imagery whose payload is
already covered by the image or text basis depending on which side of the
render a case binds. Giving it a third basis would let one payload seal two
ways.

### 2.2 The points3d basis binds the dequantization

```
sha256('POINTS3D:<dtype>:<points>:<sx>,<sy>,<sz>:<zx>,<zy>,<zz>\n'
       + little-endian quantized XYZ triples)
```

(`dynamic_behavior_gate.py:89-92`, implemented at
`points3d_content_sha256`, `dynamic_behavior_gate.py:874`.)

The quantization is *inside* the header, not beside it, because two clouds whose
stored integers agree but whose scale or zero point differs are different
geometry once dequantized. The contract reused is the codec's own —
`raw_modality_tokens.PointQuantization` (`src/neural_forge/raw_modality_tokens.py:212-227`),
whose `dequantize` is `(points - zero_point) * scale`. Per-axis scale is
serialized with Python's `repr`, which is the same shortest round-trip form the
codec already canonicalizes its JSON `scale` metadata through
(`raw_modality_tokens.py:858-861`).

Payload length is checked against `points * 3 * dtype_width`
(`dynamic_behavior_gate.py:920-925`), so a truncated or padded emission cannot
seal at all.

### 2.3 One decode step, declared and fingerprinted

A raw head output is a tensor, and a tensor is not reviewable: two runs could
round it differently and both call the result "the same emission". The decode
step is therefore a named function with a sealed identity:

* `quantize_emitted_points3d` (`src/neural_forge/dynamic_points3d_output.py:175`)
  — the exact inverse of `PointQuantization.dequantize`, using NumPy
  half-to-even rounding, rejecting non-finite values, wrong rank, and
  coordinates off the int16 lattice;
* its `BehaviorCallableIdentity` is sealed into the gate manifest as its own
  component under `EMISSION_DECODER_SCHEMA`
  (`dynamic_points3d_output.py:101`, built at `:904-914`), alongside
  `"comparison": "decoded_content_digest_exact_match"` and
  `"tolerance": "none"`.

This is the output-side analogue of what
`BehaviorInferenceExecutionContract` (`src/neural_forge/dynamic_behavior_execution.py:57`)
does for the inference entrypoint: the manifest states *how* the thing was
observed, not merely *what* was observed.

A decode failure is a **reviewable regression**, not a crash:
`Points3dEmissionDecodeError` (`dynamic_points3d_output.py:127`) is caught per
case (`:1047-1054`) and counted, because "the child's head emitted NaN" is
exactly the kind of failure a gate exists to report.

---

## 3. Evidence capture for output-side folding

### 3.1 A sealed reference-output store

`docs/MODALITY_FOUNDATION_2026-07-31.md:336-338` named this as step 4 —
*"cached parent media outputs must be stored as files with container and
decoded-content digests, which the `MediaReference` schema already supports but
no cache writer produces."* `write_points3d_parent_emission_cache`
(`src/neural_forge/dynamic_points3d_output.py:533`) is that writer.

It is the output-side twin of `write_points3d_parent_output_cache`
(`src/neural_forge/dynamic_points3d_fold.py:472`) and obeys the same rule: it is
the **only** place in the lane that runs a parent for emission, and everything
downstream replays the file with zero parent forwards. It writes:

* `emissions/<probe_id>.nfp` — one self-describing `NFPC1` container per
  emitted payload, so the store is re-decodable without the index;
* `emissions.jsonl` — one canonical-JSON line per probe binding the probe
  identity (`record_sha256`), the *consumed* cloud (`input_points_sha256`), the
  emitted container digest, its byte count, and the decoded-content digest.

`load_points3d_parent_emission_cache` (`:614`) re-decodes and re-hashes every
stored emission (`Points3dEmissionRecord.verify`, `:420`) before any of it is
trusted, then seals the whole store under one fingerprint (`:698-706`).

### 3.2 Evidence for *folding* is still input-side, and that is deliberate

The trace-capture evidence this lane folds on
(`build_native_points3d_fold_trace_cache`,
`src/neural_forge/dynamic_points3d_fold.py:1517`) hooks decoder-MLP
down-projection inputs and decoder-layer residual endpoints. It is unchanged by
the emitting head, because **no operator in this repository rewrites a modality
decoder**. Every fold operator touches decoder-internal circuits only.

So the honest statement is: the emitting head is *copied exactly and reviewed*,
not folded. `emitted_output_contract_status()`
(`dynamic_points3d_output.py:1187`) says so in its `not_provided` list, and the
e2e asserts `centroid_head.weight` is byte-identical across the fold
(`tests/test_native_points3d_output_lane.py`, in
`test_emitting_child_folds_publishes_and_regates_on_emitted_retention`).

What output-side evidence *would* look like if a later wave folds the head:
the same accumulator shape, hooked on the head's input residual and its emitted
payload rather than on an MLP down projection. Nothing in
`Points3dTraceStatisticsCache` (`dynamic_points3d_fold.py:1214`) is
modality-specific; only the conditioning stimulus and the hook sites are.

---

## 4. The behavior-review contract for emitted media

### 4.1 The case schema was already right; nothing needed to change

`MultimodalWholeOutputCase` has accepted a `MediaReference` as `parent_output`
and `expected` since the audio wave (`dynamic_behavior_gate.py:1199`, `:1232-1236`), and
`MultimodalCandidateOutput` has accepted one as `output`
(`dynamic_behavior_gate.py:1391`). `multimodal_case_coverage`
(`dynamic_behavior_gate.py:1484`) has always computed `output_modalities` and
`media_output_count` from the parent output.

Until now those two fields were structurally dead: every lane passed a string.
`Points3dEmissionRecord.as_case` (`dynamic_points3d_output.py:381`) is the
first caller to populate them — media on the input side *and* the output side
of one reviewed case. The e2e pins
`coverage["media_output_count"] == len(records)` and
`coverage["output_modalities"] == ["points3d"]`.

**Finding: no schema change was required to support output-side review.** The
media case types were designed for it and are used as designed.

### 4.2 Parent-relative retention, on a closed answer set

`NativePoints3dEmittedOutputGate` (`dynamic_points3d_output.py:830`) is the
review. Its shape is deliberately identical to the input-side gate
(`NativePoints3dBehaviorGate`, `dynamic_points3d_fold.py:701`):

| property | both gates |
|---|---|
| canonical baseline | a sealed parent cache, never a live parent |
| parent forwards during gating | `0` |
| entrypoint proof | `observe_model_execution` + `execution_contract_satisfied` (`dynamic_behavior_execution.py:181`, `:307`) |
| coverage | required splits, required categories, and their cross product |
| manifest | `build_multimodal_behavior_gate_binding` (`dynamic_behavior_gate.py:1510`) |
| evaluator surface | `evaluate(context)`, `evaluate_model(model)`, `__call__(model)` |

What differs is **what a regression is**. `parent_empty`, `fragment_like`, and
`eos_margin` are text notions and do not appear. The output-side notions, in
`Points3dEmittedOutputGateConfig` (`dynamic_points3d_output.py:718`), are:

* `minimum_emitted_retention` — share of parent emissions the child reproduces
  by **exact decoded-content digest**, default `1.0`;
* `maximum_decode_failures` — payloads that could not be decoded at all,
  default `0`;
* `maximum_empty_emissions` — zero-byte emissions, default `0`;
* per-split and per-split/category retention, so a lane cannot trade a whole
  category away against an average.

**Closed answer sets first.** Exact-match retention is only meaningful because
the emitted target is quantized: on a closed lattice, "did the child emit the
same geometry?" is decidable without a tolerance parameter, and a tolerance
parameter is precisely where an output-side gate would otherwise become
unfalsifiable. This is the same reason every input-side lane probe has a closed
answer set.

### 4.3 Open generation is out of scope, explicitly

An open emitted payload — a rendered image, a synthesized waveform, an
unconstrained mesh — has no closed reference to compare against. This contract
does not gate it, and `emitted_output_contract_status()["open_generation_policy"]`
(`dynamic_points3d_output.py:1216-1219`) states the rule: **manual review; it
must not be auto-promoted.**

A future wave that wants automatic open-generation review needs a *different*
instrument — a declared perceptual or distributional metric with its own
identity, its own thresholds, and its own honesty about what it cannot see.
Bolting a tolerance onto this gate would silently convert an exact contract into
an unfalsifiable one.

### 4.4 Child emissions are bound by content, not by path

Parent emissions are stored; child rollouts are not. A child emission becomes a
`MediaReference` with `container_sha256=None` and a
`memory://neural-forge/points3d-emission/<probe_id>` URI
(`dynamic_points3d_output.py:112`, used at `:1045`).

This is deliberate. `NOTES-W6B.md` §4.3/§6 item 1 recorded that
`artifact_certificate_sha256`, `receipt_sha256`, and
`behavior_gate_manifest_fingerprint` are **not reproducible** today, because
absolute `tmp_path` strings reach `_stable_resource_value` through closure cells
and through `MediaReference.uri`. Writing every child rollout to a temp file
would have added one more nondeterministic path per case per evaluation. The
transient side of the review binds content only.

---

## 5. What publication and fresh reload must additionally verify

`DynamicFoldSystemStack` already requires that the final evaluator and the
fresh-reload evaluator resolve to the **same sealed manifest**
(`src/neural_forge/dynamic_fold_compiler.py:454-461`). That is the mechanism
that makes output-side review survive publication, and it needs no change: the
emitted-output gate exposes the same binding fields as the text gate through
`expose_behavior_gate_binding` (`dynamic_behavior_gate.py:570`), so it is a
drop-in for both slots.

`NativePoints3dStackResources` (`dynamic_points3d_fold.py:2621`) therefore
duck-types its gate: it requires a `Points3dParentBehaviorCache` binding and an
`evaluate()` entry point rather than a concrete class, and cross-checks that the
gate and the fold cache were built from the same parent outputs
(`:2637-2650`). The emitted-output gate satisfies both because
`Points3dParentEmissionCache` *extends* a behavior cache rather than replacing
it (`dynamic_points3d_output.py:457`).

Concretely, an emitting child must additionally verify at publication time:

1. **The emission store is intact.** `evaluate_model` calls
   `emission_cache.verify()` first (`dynamic_points3d_output.py:976`), which
   re-hashes the index and re-decodes every stored container. A reviewed
   baseline that changed between fold and publish fails closed.
2. **The reloaded artifact still owns the emitting head.** The planner contract
   `_native_points3d_model_contract`
   (`src/neural_forge/dynamic_fold_planner.py:1349`) compares full interface
   identity — `source.interfaces != target.interfaces` rejects — so a fold that
   would silence an emitting parent is not plannable at all. The requirement
   check (`dynamic_fold_planner.py:1693-1712`) additionally asserts that the
   declared `emits_quantized_centroid` and the presence of a `MODALITY_OUTPUT`
   node **agree**.
3. **The emitted review actually ran against the reloaded model.** Because the
   gate is the `fresh_reload_evaluator`, the standalone reload path replays
   emitted retention against the model loaded from disk, not against the
   in-memory child.
4. **The plan records that it was an emitting fold.** The action configuration
   carries `emits_modality_output` and a distinct `promotion_gate` string
   (`dynamic_fold_planner.py:2891`), so a receipt cannot later be read as
   if only text was reviewed.
5. **The receipt carries the emitted report.**
   `write_native_points3d_dynamic_fold_receipt`
   (`dynamic_points3d_fold.py:2811`) embeds `behavior_report` and its
   `canonical_sha256`; for an emitting fold that report has
   `review_side == "output"` and `generative_quality_reviewed is False`.

**Not verified, and worth saying:** publication does not prove the *head* is
still functional beyond the reviewed probes, and the artifact certificate does
not separately fingerprint the emitted interface. The graph fingerprint covers
it transitively (interfaces are in `ModelCircuitGraph.as_dict`,
`dynamic_circuit_ir.py:738`), but there is no dedicated
`emitted_interface_fingerprint` field on the certificate.

---

## 6. The executable proof, and exactly what it proves

`tests/test_native_points3d_output_lane.py` (18 tests) folds an emitting toy
model end to end with the emitted-output gate as **both** the final evaluator
and the fresh-reload evaluator.

The model is the same `_TinyNativePoints3dChatModel` the input-side proof folds,
with `points_config.emit_centroid` switched on: a bias-free `Linear(8 -> 3)`
centroid head reading the decoder's final normalized residual. 1320 parameters
become 1344; the compiled graph gains one node and one interface (19/3 → 20/4).

**Proved.** A real `MODALITY_OUTPUT` node in a real compiled graph; a sealed
reference-output store whose containers *and* decoded payloads both re-verify;
a fingerprinted decode contract sealed as its own manifest component;
parent-relative retention by exact decoded-content digest with split and
split×category coverage; emitted drift and decode failure each rejected;
publication and a standalone fresh reload replaying emitted-output review; the
reloaded child still passing `assert_graph_emits_points3d` against its
**recompiled** IR with a `torch.equal` `centroid_head.weight`; exact parameter
delta 48 (`2 layers × 3 tensors × 8 hidden`); and zero parent forwards, zero
optimizer steps, zero backward calls throughout.

**Not proved: generative quality, at all.** The toy centroid head is randomly
initialized and cannot compute a centroid. As in the NEO vision, native audio,
and native video proofs — which gate their *text* answers the same way — the
retained emission is gated on a parent-derived marker surviving the fold
exactly; when the marker is gone the runner falls through to the model's own
head output, which is genuinely different geometry and is what the negative
cases exercise. A pass means the emitted-media review machinery is sound. It
does not mean either model emits good geometry, and the gate says so itself:
its accept reason ends `"...generative quality is not reviewed."`
(`src/neural_forge/dynamic_points3d_output.py:1174-1176`), and its report
carries `generative_quality_reviewed: False` (`:1143`).

---

## 7. The threat model, stated plainly

An output-side gate can be defeated in ways an input-side gate cannot. What
this design does about each:

| attack | defense |
|---|---|
| Child emits through a tensor no interface names. | Inspector rejects an undeclared head (`dynamic_model_inspector.py:3482-3487`). |
| Child re-encodes the same payload in a different container to look different — or in a different quantization to look the same. | Content digest binds the decoded payload *and* the dequantization (§2.2). |
| Runner returns a cached payload without running the model. | `observe_model_execution` requires root and module forwards (`dynamic_points3d_output.py:1026`, `:1075`). |
| Reference store is edited after sealing. | Container digest, byte count, and re-decoded content digest are all re-checked on every evaluation (`:420-446`). |
| Decoder silently clips a bad emission into a valid one. | The decoder raises rather than clipping (`:220-227`), and the raise is counted. |
| Retention averaged across categories hides a dead category. | Per-split/category retention (`:1106-1113`). |

---

## 8. What a second emitting modality must supply

Nothing in §2–§7 is points3d-specific except the payload codec. A second
emitting lane needs exactly five things:

1. **A hash basis** in `CANONICAL_MEDIA_HASH_BASES` — or an entry in
   `MODALITIES_WITHOUT_CANONICAL_HASH_BASIS` with a written reason. The
   import-time check (§2.1) will not let it be skipped.
2. **An inspector arm** emitting a `MODALITY_OUTPUT` node whose output port
   carries the modality, plus a `direction="output"` interface, plus the
   two-way declaration check.
3. **A decode contract** — one named function mapping the head's raw output to
   a canonical payload, sealed by identity into the manifest.
4. **A closed emitted answer set** for the probe suite, or an explicit decision
   to route the lane to manual review.
5. **Its own regression notions.** Retention, decode failure, and emptiness are
   the three that generalize; a waveform lane also wants duration collapse and
   silence share, an image lane wants an all-constant-frame check. These belong
   in that lane's config, not in a shared one.

The gate class itself is close to reusable but is not yet generic: it names
`quantize_emitted_points3d` and the points3d record types directly. Extracting
a shared emitted-output gate is the right move **when a second lane exists** —
on the same acceptance bar `NOTES-W6B.md` set for the selector extraction, not
before.

---

## 9. Files

New:

* `src/neural_forge/dynamic_points3d_output.py` — sealed emission store, decode
  contract, emitted-output gate, contract status.
* `docs/MODALITY_OUTPUT_DESIGN_2026-08-04.md` (this file).
* `tests/test_native_points3d_output_lane.py` — the executable proof.

Additive edits:

* `src/neural_forge/dynamic_circuit_ir.py` — `emitted_media_interfaces()`;
  points3d registry note.
* `src/neural_forge/dynamic_behavior_gate.py` — `POINTS3D_HASH_BASIS`,
  `points3d_content_sha256`, the fail-closed basis-coverage check.
* `src/neural_forge/dynamic_model_inspector.py` — the optional
  `MODALITY_OUTPUT` arm of `NativePoints3dCircuitInspector`.
* `src/neural_forge/dynamic_fold_planner.py` — interface-identity comparison and
  the declaration/graph agreement check in the points3d contract;
  `emits_modality_output` in the emitted action configuration.
* `src/neural_forge/dynamic_points3d_adapter.py` — the emitting head is proven
  copied, not re-dimensioned.
