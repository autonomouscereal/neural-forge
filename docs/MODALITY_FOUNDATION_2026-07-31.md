# Modality Foundation — 2026-07-31

Status: foundation landed, integration pending (see `NOTES-1H.md` at the repo
root for the exact wiring instructions a later wave must apply).

This document is the single reference for four things the compiler previously
left implicit: what a modality is *called*, how a reviewed case carries
*media*, which raw-byte protocol is *canonical*, and what the audio lane
actually provides today.

---

## 1. Canonical modality registry

`src/neural_forge/dynamic_circuit_ir.py`

```python
CANONICAL_MODALITY_ORDER = ("text", "image", "audio", "video", "ocr", "points3d")
CANONICAL_MODALITIES = frozenset(CANONICAL_MODALITY_ORDER)

canonical_modality(value, label="modality") -> str   # validates, raises DynamicCircuitIRError
is_canonical_modality(value) -> bool                 # non-raising predicate
modality_registry() -> dict                          # JSON-safe self-describing statement
```

### Compatibility rule (deliberate)

`CircuitPort.modality` and `ModalityInterface.modality` **remain free-form
strings**. They are validated only as stable identifiers, exactly as before.
Nothing in this change tightens them, because:

* every already-published `ModelCircuitGraph` fingerprint, fold receipt, and
  behavior manifest embeds the modality string it was built with, and
* a graph that fails to construct is a graph that cannot be re-verified.

The registry is therefore **opt-in for new lanes**: a new lane calls
`canonical_modality(...)` and gets a hard failure on a private spelling; an old
graph keeps validating untouched. `tests/test_dynamic_modality_registry.py`
pins both halves of that contract, including a `CircuitPort(modality="depth")`
that still constructs.

### The `ocr` collision, written down

`ocr` means two different things in this repo and the registry now says so:

| Where | Meaning |
|---|---|
| `RawModality.OCR`, IR registry | a **transport modality** — rendered-text imagery framed by the raw-byte codecs |
| NEO probe suite (`_CATEGORIES` in `dynamic_neo_fold.py`) | a **task category** — "read the three-digit code", answered in text |

`modality_registry()["task_categories_are_not_modalities"] is True`. A category
name must never be used as a modality name and vice versa. The NEO `ocr`
category is a text task over an `image` modality input.

### Enum reconciliation

`RawModality` members are now *defined from* the IR constants:

```python
class RawModality(str, Enum):
    IMAGE = MODALITY_IMAGE
    AUDIO = MODALITY_AUDIO
    ...
```

so `{m.value for m in RawModality} <= CANONICAL_MODALITIES` holds by
construction and cannot drift. `hf_native_raw_tokens.canonical_raw_modality()`
is the same opt-in validator for the native codec, which otherwise keeps
accepting free-form labels (`"tensor"`, `"raw"`, `"3d.point-cloud"` are live in
existing corpora and tests).

---

## 2. Media-capable review cases

`src/neural_forge/dynamic_behavior_gate.py` — **strictly additive**.
`WholeOutputCase`, `CandidateWholeOutput`, `WholeOutputGateConfig`,
`TaskScoreCoveragePolicy`, and `CachedWholeOutputBehaviorGate` are byte-for-byte
unchanged.

### Types

| Type | Purpose |
|---|---|
| `MediaReference` | one decoded payload: `modality`, `content_sha256`, `uri`, `byte_count`, `codec`, `hash_basis`, optional `container_sha256`, sorted scalar `attributes` |
| `MultimodalWholeOutputCase` | `case_id`, `prompt`, `parent_output` (text **or** media), ordered `inputs: tuple[MediaReference, ...]`, optional `expected`, plus the text gate's `category` / `split` / `parent_eos_margin` / `critical_nonempty` / `parent_metadata` fields |
| `MultimodalCandidateOutput` | child rollout whose `output` may be text or media |

### Content binding

Binding is on **decoded content**, never on the container. Two PNG or WAV
encodings of the same decoded payload share one `content_sha256`; a re-encode
that changes one decoded sample fails closed.

```
IMAGE_HASH_BASIS = "sha256('RGB:<width>x<height>\n' + decoded RGB bytes)"
AUDIO_HASH_BASIS = "sha256('PCM16:<rate>:<channels>:<frames>\n' + little-endian int16 samples)"
TEXT_HASH_BASIS  = "sha256('TEXT:utf-8\n' + UTF-8 bytes)"
```

`IMAGE_HASH_BASIS` is **byte-identical** to the string already sealed into every
NEO native-vision probe manifest (`dynamic_neo_fold.IMAGE_HASH_BASIS`), verified
by `tests/test_multimodal_behavior_cases.py`. NEO can migrate to the shared
constant with an import and no manifest change.

Helpers: `media_content_sha256(header, payload)` (generic
`sha256(header + "\n" + payload)`), `image_content_sha256`,
`audio_content_sha256`, `text_content_sha256`. The audio helper rejects a frame
count that disagrees with the payload length, so an unsound case cannot be
sealed.

A modality with no canonical basis (`points3d`, `ocr`) **must** supply
`hash_basis` explicitly. That is the extension point: define the basis string
with the lane, do not invent one at the call site. Video gained its canonical
basis (`VIDEO_HASH_BASIS`, whole-clip RGB24 with geometry and fps bound into
the digest) when the native video lane landed.

### Verification surface

* `MediaReference.verify_container(root=None)` — re-hashes the stored file and
  compares size; requires a declared `container_sha256`.
* `MediaReference.verify_content(observed_sha256)` — compares a freshly decoded
  digest.
* `MediaReference.local_path(root)` — refuses `..` escapes and non-local URIs.
* `MultimodalWholeOutputCase.verify_inputs(root)` — container sweep.

### Serialization

`as_dict()` / `from_dict()` on all three types, canonical-JSON round-trip
exact, with strict unknown/missing-key rejection and a schema check.
`MultimodalWholeOutputCase.as_dict()` carries a derived `modalities` summary
that `from_dict()` recomputes and cross-checks, so a modality relabel in a
sealed manifest fails closed.

Manifest integration mirrors `build_behavior_gate_binding` exactly:

```python
build_multimodal_behavior_gate_binding(...)   -> BehaviorGateBinding  (same seal, same reload contract)
multimodal_case_baseline_fingerprint(cases)   -> str
multimodal_case_coverage(cases)               -> dict   (added as one manifest component)
multimodal_cases_from_manifest(manifest)      -> tuple[MultimodalWholeOutputCase, ...]
```

The binding is a real `BehaviorGateBinding`: identical body layout, identical
`manifest_fingerprint` computation, identical `inference_execution` contract
handling. A future gate reuses `expose_behavior_gate_binding` and
`resolve_behavior_gate_binding` with no changes — `NativeAudioBehaviorGate`
already does, which is the proof that a new modality no longer needs a gate
clone to describe its cases.

---

## 3. Raw-byte protocol decision

### Decision

**`neural_forge.hf_native_raw_tokens` (`NFHFRAW1`) is the canonical raw-byte
transport for folding lanes.** `neural_forge.raw_modality_tokens` (`NFRT`) is
kept and supported as an explicit **vocabulary-expansion research utility**.

### Rationale

The two protocols differ in exactly one consequential way:

| | `NFHFRAW1` | `NFRT` |
|---|---|---|
| module | `hf_native_raw_tokens` | `raw_modality_tokens` |
| added vocabulary rows | **0** | **266** (10 control + 256 byte) |
| mutates tokenizer | no | yes |
| changes tokenizer fingerprint | **no** | **yes** |
| requires untied output head | no | yes |
| requires embedding resize | no | yes |

Every folding lane in this repo compares parent and child
`ModelCircuitGraph.tokenizer_fingerprint`, replays a sealed parent behavior
cache produced *before* any child edit, and requires exact state restore on
rejection. All three properties are destroyed by adding vocabulary rows, and
none are affected by reusing byte tokens that already exist. The decision is
therefore forced by the lane contracts, not by a preference between codecs.

`NFRT` is not deprecated: it is the only path that studies what added modality
rows do, and it stays fully tested. What changed is that its lane
incompatibility is now **stated in code** instead of being an unwritten
assumption. (Note that the README documents `NFRT` and not `NFHFRAW1`; the
README is owned by another wave — see `NOTES-1H.md` item 6.)

### Compatibility matrix

`raw_token_lane_compatibility()` — defined in `hf_native_raw_tokens`,
re-exported from `raw_modality_tokens` so both modules quote *one* statement
(`reexported is original` is asserted in tests).

| Lane | `NFHFRAW1` | `NFRT` | Why |
|---|---|---|---|
| `parent_child_tokenizer_fingerprint_equality` | allowed | **blocked** | 266 added tokens change the fingerprint; equality can never hold |
| `cached_parent_behavior_gate` | allowed | **blocked** | cached parent outputs predate expansion, so control-token IDs do not exist in the parent |
| `dynamic_fold_plan_execution` | allowed | **blocked** | `resize_token_embeddings` changes parameter counts and invalidates exact state restore |
| `vocabulary_expansion_research` | allowed | allowed | this is the lane the expansion utility exists for |

### Bridge (absent)

`raw_token_lane_compatibility()["bridge"]["status"] == "absent"`. Opening the
three blocked lanes to `NFRT` would require, at minimum:

1. a **tokenizer-fingerprint lineage record** — parent fingerprint, added token
   IDs, expansion receipt digest — that lanes may accept in place of strict
   equality;
2. proof that **no pre-existing token ID or embedding row changed** during
   expansion (`expand_hf_raw_token_vocabulary` already reports enough to build
   this; nothing consumes it);
3. a deterministic, replayable **initializer receipt** for every added input,
   output, and bias row;
4. parent behavior caches **regenerated after expansion**, or a proof that
   added rows are unreachable for cached prompts.

Until all four exist, the matrix above is the contract.

---

## 4. Audio lane

Two new modules mirroring the NEO native-vision file pair, with one deliberate
difference: **nothing names a parent architecture.**

### `src/neural_forge/native_audio_probe.py` — probe generator

* Schema `neural-forge-native-audio-probe-v1`, PCM16 mono 16 kHz, 8000 frames
  (0.5 s) per clip, stored as RIFF/WAVE through the stdlib `wave` module.
  Dependencies: stdlib + NumPy only. No `soundfile`, no `librosa`.
* Categories and closed answer sets:

  | category | answers | content |
  |---|---|---|
  | `tone_count` | `1`–`5` | N rhythmic tone bursts separated by silence |
  | `pitch_direction` | `rising` / `falling` / `constant` | three consecutive tone segments on a frequency ladder |
  | `sound_type` | `silence` / `tone` / `noise` | exact zeros, a pure tone or a three-note chord, or white noise |

* **Determinism is structural.** Tones come from an integer phase accumulator
  indexing a fixed int16 wavetable; chords are integer sums of phase-offset
  tones; noise is a seeded PCG64 *integer* stream; jitter comes from a
  string-seeded `random.Random(f"...:{seed}:{category}:{index}")`, so no probe
  depends on generation order. The only `math.sin` call builds the wavetable,
  whose digest is sealed into the manifest as `wavetable_sha256` — a platform
  whose libm rounds a table entry differently fails authentication loudly
  instead of silently producing different audio under the same seed.
* Manifest is **self-sealing**: `manifest_seal = canonical_sha256(body)`.
  `authenticate_native_audio_probe()` checks the seal, the schema, the hash
  basis, the wavetable digest, per-record container digest and byte count,
  path escapes, closed-set membership, and then **re-decodes and re-hashes
  every waveform** before returning.

### `src/neural_forge/dynamic_audio_fold.py` — records, gate, evidence

Mirrors `NeoVisionBehaviorRecord` / `load_neo_parent_behavior_cache` /
`NeoNativeVisionBehaviorGate` / `build_neo_native_vision_fold_cache`:

* `AudioBehaviorRecord` + `AudioParentBehaviorCache` +
  `load_audio_parent_behavior_cache(manifest, outputs, expected_*_sha256=None)`
  — authenticated manifest, re-decode + re-hash of every file, per-record
  manifest digest, parent raw-output digest, closed-set answer normalization,
  and a cache fingerprint re-checked by `verify()` on every use.
* `AudioBehaviorRecord.media` returns a `MediaReference`; `.as_case()` returns
  a `MultimodalWholeOutputCase`. The gate seals those cases through
  `build_multimodal_behavior_gate_binding`, i.e. the audio lane consumes the
  Section 2 schema rather than cloning a case type.
* `NativeAudioBehaviorGate` — parent-relative: every parent-correct answer must
  be retained, no split or split×category accuracy regression, no direct-answer
  regression, parent-correct coverage required across the split×category cross
  product, and the declared
  `BehaviorInferenceExecutionContract` must be observed to have executed
  (default: top-level forward; caller-supplied for wrapper entrypoints such as
  NEO's `chat`). `parent_forward_count` is **0** in every report.
* `write_audio_parent_output_cache(manifest, destination, parent_runner=...)`
  is the *only* function in the lane that touches a parent for behavior; the
  gate replays its JSONL output.
* `build_native_audio_fold_trace_cache(...)` — the audio analog of NEO's
  evidence builder. The hook pattern is the modality-agnostic half: a forward
  **pre-hook on each layer's MLP down projection** captures the activation rows
  a fold must reproduce, and a **forward hook on the layer** captures residual
  endpoint energy. Held-out probes are never traced; the parent is asserted
  zero-gradient before capture and its parameter *and* runtime state
  fingerprints must be unchanged after; hooks and training modes are restored
  in a `finally`.
* `AudioTraceStatisticsCache` seals statistics as canonical little-endian
  float32 in one blob plus a sealed manifest (per-tensor offset, byte count,
  digest; blob digest; logical cache key). No pickle, no `safetensors`
  dependency. It contains **statistics only** — no fold selection, because a
  selection rule belongs to the method that consumes the evidence and sealing
  one here would freeze a policy this lane has not validated.

### Model-agnostic seams

`trace_runner`, `child_runner`, `parent_runner`, `layer_resolver`,
`projection_resolver`, `inference_execution_contract`. `audio_lane_status()`
returns this list plus what the lane does and does not provide.

### What remains for the audio lane

1. **Inspector emission.** No inspector emits `CircuitKind.MODALITY_INPUT` with
   modality `"audio"`; `tests/test_native_audio_lane.py` builds such a graph by
   hand to prove the IR accepts it. See `NOTES-1H.md` item 1.
2. **Planner contract.** No `FoldAction` evidence channel names audio trace
   evidence. `NOTES-1H.md` item 2.
3. **Method registration.** No `METHOD_ID`, operator, or registry entry — this
   lane deliberately stops before folding mathematics. `NOTES-1H.md` item 3.
4. **A real audio-capable parent model.** Not chosen. Selection criteria and
   the two candidate shapes are in `NOTES-1H.md` item 4.

---

## 5. Acknowledged limitation: `MODALITY_OUTPUT` is still dead code

`CircuitKind.MODALITY_OUTPUT` exists in the IR and **no inspector emits it**.
Nothing in this change alters that. The media-capable case schema now *allows*
a media parent output and a media candidate output
(`MultimodalWholeOutputCase(parent_output=MediaReference(...))`,
`MultimodalCandidateOutput(output=MediaReference(...))`), which is the case
schema half of output-side support, but the compiler half does not exist.

Implementing output-side modality would require, in order:

1. **Inspector emission** — a `MODALITY_OUTPUT` node with a `PortKind.MODALITY`
   output port and a `direction="output"` `ModalityInterface`, for a model that
   actually has a decoder head (vocoder, image detokenizer, codec head). Today
   every inspected model ends at `OUTPUT_HEAD` producing `PortKind.LOGITS`.
2. **A decode contract in the gate.** A media output must be decoded and
   content-hashed before it can be compared, so a gate needs a declared,
   fingerprinted decoder callable — the same treatment
   `BehaviorInferenceExecutionContract` gives the inference entrypoint. Without
   it, "the child produced audio" is unverifiable.
3. **Parent-relative metrics that are not text metrics.** `parent_empty`,
   `fragment_like`, and `eos_margin` are text notions. An audio or image output
   needs its own regression notions (duration collapse, silence share,
   all-black frames, decoder failure), defined per modality and thresholded in
   a config like `WholeOutputGateConfig`.
4. **A sealed reference-output store.** Cached parent *media* outputs must be
   stored as files with container and decoded-content digests, which the
   `MediaReference` schema already supports but no cache writer produces.
5. **Fold operators that touch the output head.** Every current operator folds
   decoder-internal circuits; none rewrites a modality decoder, so acceptance
   evidence for an output-side fold does not exist yet.

Until step 1 lands, `MODALITY_OUTPUT` remains a reserved kind, and the honest
statement is: **this repo can review media on the input side and text on the
output side.**

---

## 6. Files and tests

New:

* `src/neural_forge/native_audio_probe.py`
* `src/neural_forge/dynamic_audio_fold.py`
* `docs/MODALITY_FOUNDATION_2026-07-31.md` (this file)
* `NOTES-1H.md` (integration instructions for the next wave)
* `tests/test_dynamic_modality_registry.py`
* `tests/test_multimodal_behavior_cases.py`
* `tests/test_native_audio_lane.py`

Additive edits:

* `src/neural_forge/dynamic_circuit_ir.py` — registry constants, validation
  helper, `modality_registry()`; no existing class behavior changed.
* `src/neural_forge/dynamic_behavior_gate.py` — hash bases, digest helpers,
  media case types, multimodal binding helpers; no existing class touched.
* `src/neural_forge/raw_modality_tokens.py` — docstring decision record, enum
  reconciliation, re-exported compatibility statement.
* `src/neural_forge/hf_native_raw_tokens.py` — docstring decision record,
  `canonical_raw_modality()`, `raw_token_lane_compatibility()`.
* `src/neural_forge/hf_native_raw_eval.py` — docstring cross-reference.
