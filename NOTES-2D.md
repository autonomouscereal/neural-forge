# NOTES-2D — independent-vocabulary tokenizer pair bridge

Branch `codex/lrc-blank-compression`. Wave 2D. Everything here is **additive and
unwired**: two new files, no existing module touched, no runner or planner
changed. Nothing in the repo imports the new module yet — that is deliberate,
because the operator lane, the planner, and the LFM2 adapters are being edited
concurrently. This file is the exact wiring recipe for whoever merges.

Files added:

- `src/neural_forge/hf_tokenizer_pair_bridge.py` (new)
- `tests/test_hf_tokenizer_pair_bridge.py` (new, 17 tests, ~11 s CPU)

Files deliberately **not** modified: `src/neural_forge/hf_tokenizer_bridge.py`
(the one helper needed, `fingerprint_hf_tokenizer_semantics`, is imported as-is —
no shared-module edit was genuinely required), `hf_token_align.py`,
`dynamic_causal_operator_promotion.py`, `dynamic_fold_planner.py`,
`dynamic_fold_registry.py`, `dynamic_fold_executor.py`, every `hf_lfm2_*`
module, `experiments/*`.

---

## 1. Why this exists

`compile_hf_target_tokenizer_bridge` **manufactures** its child vocabulary as a
strict subset of the donor's rows via `prune_hf_tokenizer`, so every target row
*is* a donor row and `StrictTokenTransportProfile` is a total function
target-row → donor-row. That contract is structurally unavailable for
cross-FAMILY folding: a Qwen3 parent and an LFM2-shaped child carry two
pre-existing, unrelated vocabularies, and in general **no row of one is a row of
the other**.

The pair bridge compiles the alignment that is still exact for that case. Every
defined row of one tokenizer is segmented into a nonempty ordered sequence of
rows of the other — its **code** — over raw ByteLevel token strings, in both
directions. Singleton Unicode transport is not attempted anywhere (the LFM2 lane
already proved it lossy); the multi-code readout is a downstream concern.

The module reads tokenizer vocabularies only. It never opens a checkpoint,
reads a weight, or allocates a model.

---

## 2. API surface

### Compiler

```python
compile_hf_tokenizer_pair_bridge(
    source_tokenizer, target_tokenizer, output_dir, *,
    source_row_capacity: int | None = None,
    target_row_capacity: int | None = None,
    map_special_roles: bool = True,
) -> TokenizerPairBridgeArtifact
```

`output_dir` is created when absent and must be **dedicated to the bridge** —
only the sealed manifest and its content-addressed payload may live there
(a shared tokenizer directory fails closed). Re-compiling the identical bridge
is idempotent; a bridge that differs in any way fails closed rather than
overwriting the sealed one.

Row capacities default to the tokenizer lengths and may be raised to the
physical embedding-matrix row count (padded vocab). Codes are only ever defined
for `0 .. vocab_size-1`; the capacity is carried so the operator layer knows the
matrix shape and which trailing rows carry no evidence.

Companion entry points:

```python
build_hf_tokenizer_pair_code_profile(source, target, *, ...) -> TokenizerPairCodeProfile
load_hf_tokenizer_pair_bridge_artifact(bridge_dir, *, source_tokenizer=None,
    target_tokenizer=None, expected_artifact_sha256=None, packaged=False)
tokenizer_pair_bridge_configuration(bridge) -> dict[str, str]
```

`build_..._code_profile` compiles the full contract without writing a file —
use it for stats, dry runs, and the re-verification path.

### Segmentation contract

Both tokenizers must be Hugging Face **fast** tokenizers whose backend proves,
independently:

- `model.type == "BPE"`,
- a `ByteLevel` component in `pre_tokenizer` **and** in `decoder`,
- `model.byte_fallback` falsy,
- `continuing_subword_prefix` and `end_of_word_suffix` empty.

Anything else raises `TokenizerPairBridgeError` naming the missing extension
(see §6). The two tokenizers' pre-tokenizers, decoders, and normalizers are
recorded but are **not required to match** — that is exactly the constraint the
strict bridge imposes and the pair bridge must not.

Each row is reduced to an exact byte string: an added-token row uses
`content.encode("utf-8")`; every other row uses the ByteLevel character→byte
map. The map is derived locally and cross-checked against
`tokenizers.pre_tokenizers.ByteLevel.alphabet()` on every compile.

The segmentation dictionary is built from the output tokenizer's **model rows
only** (added and special rows excluded), so a special token can never appear
inside a multi-code. The output tokenizer is then required to cover all 256
single-byte rows; if it does not, the compile fails closed, because without them
greedy longest match has no guaranteed terminating fallback.

Resolution order per row, highest priority first
(`CODE_RESOLUTION_ORDER`):

| class | when | code length |
|---|---|---|
| `special-role` | both tokenizers declare the same role (`bos`/`eos`/`pad`/`unk`) | 1 |
| `anchor-exact` | the row's byte string is exactly some output row's byte string | 1 |
| `multi-code` | greedy longest match over the output model-row byte strings | ≥ 2 |

Invariant asserted at compile time and in tests: **code length 1 ⟺ class is
`special-role` or `anchor-exact`**. A one-piece greedy segmentation would mean
the whole byte string is an output row, which `anchor-exact` already caught.

Anchor ties (a byte string present as both an added row and a model row) resolve
to the same row kind as the input row first, then the other kind — deterministic
and recorded.

`map_special_roles=False` removes the role path entirely, so **every** row is
byte-exact. Use it when you want pure byte semantics with no role remapping;
it produces a different `profile_fingerprint`, so the two are never confusable.

Byte-exactness is verified per row *inside the compiler*: the concatenation of
the code's output byte strings must equal the input byte string. Role-mapped
rows are role-exact, not byte-exact, and are counted separately.

### Segmentation stats format

Each direction carries a `statistics` block with exactly these keys (all
integers — no floats reach the seal, so canonical JSON stays exact):

```
direction                 "source-row-to-target-code" | "target-row-to-source-code"
input_vocab_size          input_row_capacity
output_vocab_size         output_row_capacity
defined_rows              unresolved_rows          (always 0; nonzero is impossible by construction)
anchor_exact_rows         special_role_rows        multi_code_rows
byte_exact_rows           byte_mismatch_rows       (always 0)
single_byte_anchor_rows   rows that anchor directly onto a raw byte row
pure_byte_fallback_rows   multi-code rows whose entire code is raw byte rows
total_code_entries        sum of all code lengths (mean = total / defined)
min_code_length           max_code_length
code_length_histogram     [[length, count], ...] ascending
covered_output_rows       uncovered_output_rows
```

`pure_byte_fallback_rows` is the number the LFM2 audit cares about — rows that
survive only as raw byte fragments, the direct analogue of the 665 lossy
byte-fragment rows.

Measured on the real 64,402-row LFM2 parent tokenizer against a real 1,024-row
ByteLevel-BPE target (`.codex-test-runs/lfm2-parent-tokenizer` ×
`runs/local_tiny_smoke/llama_to_blank_gpt2_lrc`):

```
forward   defined 64402  anchors 986   multi 63416  unresolved 0  byte_mismatch 0
          pure_byte_fallback 17775     total_code_entries 242825   max_code_length 17
          hist [[1,986],[2,16442],[3,21058],[4,11327],[5,4684],[6,4715],[7,1462], ...]
reverse   defined 1024   anchors 986   multi 38     unresolved 0  byte_mismatch 0
          total_code_entries 1065      max_code_length 3
```

Both directions compile in **0.93 s**; compile+seal+reload is 4.81 s;
`verify_live_tokenizers` is 2.24 s; a 64402×128 transport is 0.35 s.

---

## 3. Artifact schema

Schema `neural-forge-tokenizer-pair-bridge-v1`, kind
`sealed-independent-tokenizer-pair-bridge`. Two files in `output_dir`:

| file | content |
|---|---|
| `neural_forge_tokenizer_pair_bridge.json` | canonical-JSON sealed manifest |
| `neural_forge_tokenizer_pair_bridge_codes.safetensors` | content-addressed code payload |

The code tables are **not** inline. At LFM2 scale the manifest stays 4.4 KB
(human-inspectable) while the payload is 1.3 MB of `int32`/`int8` arrays —
following the repo's existing safetensors-payload + canonical-JSON-manifest
convention (`hf_lfm2_canonical_proposal_pack`, `dynamic_evidence_catalog_artifact`).

Payload arrays, per direction (`forward_` / `reverse_` prefix):

```
{prefix}_code_flat      int32  [total_code_entries]     concatenated codes
{prefix}_code_offsets   int32  [defined_rows + 1]       CSR-style row offsets
{prefix}_code_classes   int8   [defined_rows]           index into CODE_RESOLUTION_ORDER
```

Manifest body (the exact key set the loader requires — extra or missing keys
fail closed):

```
schema, kind, profile_fingerprint, pair_profile,
source_tokenizer_semantics_sha256, target_tokenizer_semantics_sha256,
payload_files[{relative_path,size_bytes,sha256}],
model_parameter_operations, artifact_sha256
```

`pair_profile` carries both tokenizer classes, both vocab sizes and capacities,
the `segmentation` descriptor (algorithm, resolution order, `map_special_roles`,
`byte_alphabet_size: 256`, `singleton_unicode_transport_used: false`), the
`special_role_map`, both directions' `statistics` and `array_sha256`, both
backends' semantics (including a `backend_sha256` over the full serialized
tokenizer JSON), and the zero-model-parameter audit.

Zero-operation audit, enforced on load:

```
model_parameter_operations = {training_steps: 0, backward_calls: 0,
    pruned_parameters: 0, quantized_parameters: 0, random_target_rows: 0}
pair_profile.model_weight_reads = 0
pair_profile.model_weight_writes = 0
pair_profile.model_weight_payloads_resident = false
```

The `model_parameter_operations` block is byte-identical to the strict bridge's,
so a consumer can audit either kind with one check.

**Fail-closed layering** (all four proven in the tests):

1. `array_sha256` per code array, bound into `profile_fingerprint` →
   `artifact_sha256`. A cold load hashes the arrays *before* rebuilding the
   profile, so a swapped payload is rejected as `"... was tampered with"` even
   if the attacker also patched the file record.
2. `payload_files` byte records → `verify_files()` rejects any content change,
   addition, or removal; `verify_packaged_files()` authenticates the sealed
   records only and tolerates unrelated package files.
3. `artifact_sha256` over the canonical body → any manifest edit is rejected as
   `"seal is invalid"`.
4. `verify_live_tokenizers()` re-fingerprints both tokenizers **and recompiles
   both code tables**, comparing the fingerprint and the raw codes.

Config-seal bundle (mirrors `strict_tokenizer_bridge_configuration`):
`_neural_forge_tokenizer_pair_bridge_artifact_sha256`,
`_neural_forge_tokenizer_pair_profile_sha256`,
`_neural_forge_pair_source_tokenizer_semantics_sha256`,
`_neural_forge_pair_target_tokenizer_semantics_sha256`
(`TOKENIZER_PAIR_BRIDGE_CONFIG_KEYS`).

---

## 4. Shared bridge protocol (spec — NOT yet implemented anywhere)

`StrictTokenizerBridgeArtifact` and `TokenizerPairBridgeArtifact` already agree
on everything the operator layer needs for *identity and authentication*. They
deliberately do **not** agree on transport, because the transports are genuinely
different functions (one row-select, one code-average). The right protocol is
therefore narrow:

```python
class SealedTokenizerBridge(Protocol):
    """Identity and authentication surface shared by both bridge kinds."""

    root_dir: Path
    artifact_sha256: str

    @property
    def bridge_kind(self) -> str: ...
    @property
    def profile_fingerprint(self) -> str: ...
    @property
    def parent_tokenizer_semantics_sha256(self) -> str: ...
    @property
    def child_tokenizer_semantics_sha256(self) -> str: ...
    @property
    def target_vocabulary_size(self) -> int: ...

    def verify_files(self) -> None: ...
    def verify_packaged_files(self) -> None: ...
    def verify_live_tokenizers(self, parent, child, *, packaged: bool = False) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
```

Current conformance:

| member | strict artifact | pair artifact |
|---|---|---|
| `artifact_sha256`, `profile_fingerprint`, `target_vocabulary_size` | yes | yes |
| `verify_files` / `verify_packaged_files` / `verify_live_tokenizers` / `to_dict` | yes | yes |
| `bridge_kind` | **missing** — value is only in `to_dict()["kind"]` (`"sealed-strict-target-tokenizer"`) | yes (`"sealed-independent-tokenizer-pair-bridge"`) |
| `parent_tokenizer_semantics_sha256` | **missing** — spelled `donor_tokenizer_semantics_sha256` | yes (alias added, alongside `source_…`) |
| `child_tokenizer_semantics_sha256` | **missing** — spelled `target_tokenizer_semantics_sha256` | yes (alias added, alongside `target_…`) |

**Wiring item 1 (owner of `hf_tokenizer_bridge.py`):** add three additive
properties to `StrictTokenizerBridgeArtifact` — `bridge_kind` returning
`_STRICT_TOKENIZER_BRIDGE_KIND`, `parent_tokenizer_semantics_sha256` returning
`donor_tokenizer_semantics_sha256`, and `child_tokenizer_semantics_sha256`
returning `target_tokenizer_semantics_sha256`. Pure aliases; they change no
sealed bytes and no fingerprint. I did not make this edit because the file is
shared with three concurrent waves and the change was not required for anything
in wave 2D.

**Wiring item 2:** put the `Protocol` itself in a new neutral module (suggested:
`src/neural_forge/tokenizer_bridge_protocol.py`) rather than in either bridge
module, so neither imports the other. Both bridges are structurally conformant
today; nothing needs to subclass anything.

---

## 5. Handoff to `token_interface_bridge` (`dynamic_causal_operator_promotion.py`)

`StrictTokenInterfaceBridgeOperator` today does, in `prepare()`:

1. `artifact.verify_files()`
2. `artifact.donor_tokenizer_semantics_sha256 == parent_graph.tokenizer_fingerprint`
3. `artifact.target_tokenizer_semantics_sha256 == child_graph.tokenizer_fingerprint`
4. target rows == `artifact.target_vocabulary_size`
5. `compile_strict_vocabulary_operators(...)` → embedding/head rows
6. install atomically under a snapshot, report `_fold_only_operation_audit()`

Steps 1–4 and 6 are **identical** for a pair bridge once the §4 aliases land.
Only step 5 differs. Proposed shape for the pair lane, as a *sibling* operator
class in the same module (do not overload the strict one):

```python
class PairCodeTokenInterfaceBridgeOperator:
    method_id = TOKEN_INTERFACE_BRIDGE_METHOD   # same planner method

    def __init__(self, *, artifact=None, bridge_directory=None,
                 embedding_transport="length-normalized",   # or "anchor-exact"
                 lm_head_transport="length-normalized",
                 token_frequencies=None,
                 embedding_coordinates=None, lm_head_coordinates=None,
                 snapshot_config=None): ...
```

`prepare()` differences from the strict operator:

- **Evidence, then residual coordinates.** Call
  `compile_pair_bridge_length_normalized_transport(source_embedding, artifact)`
  (or the anchor / occurrence-weighted variant) to get
  `[target_row_capacity, source_width]` evidence in float64, **then** push it
  through `embedding_coordinates` exactly as the strict operator does. The pair
  module deliberately does not take a `CoordinateMap`: it builds evidence in the
  source width and stops. Width folding stays where it already lives.
- **`evidence_mask` is load-bearing.** Unlike the strict bridge, a pair bridge
  does not guarantee every target row is reached — at LFM2 scale 33/1024 target
  rows had no forward incidence. The operator MUST decide a policy for masked
  rows and record it in the report. It must **not** silently install the zeros
  the evidence builder returns. Options, in the order I would try them:
  1. blend: `anchor-exact` where its mask is true, `length-normalized`
     elsewhere, leave the remaining masked rows at the child's incumbent value
     (install only `evidence_mask` rows) — this is the only option that
     installs nothing unproven;
  2. fail closed when `uncovered_output_rows > 0` (defensible for a first
     cross-family run, and the easiest to audit);
  3. reverse-code fallback: for an uncovered target row `t`, use
     `reverse_code(t)` to average the source rows that spell it. Cheap, and it
     covers rows that the forward direction structurally cannot reach.
- **Tied storage.** The strict operator asserts exact equality of embedding and
  head when the child ties them. For the pair bridge, run the transport **once**
  and use the same tensor for both, exactly as `require_tied_target` does — do
  not run the builder twice and compare, because the two calls are only
  bit-identical when the same weighting and inputs are used.
- **Report.** Carry `bridge_artifact_sha256`, `pair_profile_fingerprint`, both
  transport `report_sha256` values, `weighting`, `covered_output_rows`,
  `uncovered_output_rows`, and the masked-row policy. Every transport report
  already carries the zero-operation audit and a `report_sha256`.

Planner/registry note: the pair operator needs the planner to expose the *parent*
tokenizer fingerprint as `parent_graph.tokenizer_fingerprint` (already true) and
to allow a `token_interface_bridge` action whose parent and child tokenizer
fingerprints are unrelated. Whatever check currently assumes the child tokenizer
is a pruned donor must be relaxed for `bridge_kind ==
"sealed-independent-tokenizer-pair-bridge"`. I did not audit
`dynamic_fold_planner.py` / `dynamic_fold_registry.py` for that — out of my
ownership.

---

## 6. SentencePiece byte-fallback extension (v2 — deliberately not implemented)

Every non-ByteLevel input fails closed with a message naming this work.
Verified against the real 32,000-row Metaspace/SentencePiece Llama tokenizer in
`runs/local_tiny_smoke/gpt2_to_blank_llama_synth_fixed`:

```
target tokenizer does not use ByteLevel pre-tokenization. The v1 pair bridge
aligns two ByteLevel-BPE vocabularies only. A SentencePiece/Unigram vocabulary
requires the SentencePiece byte-fallback extension -- decompose every piece
through its '<0xNN>' byte-fallback rows into raw bytes before segmentation --
which is deliberately not implemented in v1 and therefore fails closed here.
```

What v2 needs:

1. A `_row_byte_strings` variant for SentencePiece: strip the `▁` (U+2581)
   metaspace marker to `0x20`, decode `<0xNN>` rows to the literal byte, and
   UTF-8 encode the rest. The `Metaspace` `prepend_scheme` and `split` settings
   change what a leading `▁` means and must be read, not assumed.
2. A byte-coverage proof over the `<0x00>`..`<0xFF>` rows instead of the
   ByteLevel alphabet — `model.byte_fallback` must be true and all 256 rows
   present, otherwise there is still no exact fallback and it must fail closed.
3. Nothing else changes. `_ByteSegmenter`, the resolution order, the statistics,
   the payload, the seal, and every transport helper already operate on raw
   `bytes` and are vocabulary-format agnostic.

A Unigram model with byte fallback is the same job; a Unigram model *without*
byte fallback has no exact decomposition and must keep failing closed.

---

## 7. Verification

```
.venv/Scripts/python.exe -m pytest tests/test_hf_tokenizer_pair_bridge.py -q
17 passed in 10.91s

.venv/Scripts/python.exe -m pytest tests/test_hf_tokenizer_bridge.py -q
10 passed          # shared module untouched; kept green

.venv/Scripts/python.exe -m pytest tests/test_hf_token_align.py tests/test_hf_tokenizer_prune.py -q
34 passed

.venv/Scripts/python.exe -m ruff check   src/neural_forge/hf_tokenizer_pair_bridge.py tests/test_hf_tokenizer_pair_bridge.py
.venv/Scripts/python.exe -m ruff format --check  (same two files)
All checks passed / 2 files already formatted
```

The tests build **two real ByteLevel-BPE tokenizers trained with the
`tokenizers` library on two different tiny corpora** (English technical prose vs
a French/Chinese/physics mix), giving 406 and 400 rows with genuinely divergent
merges: 275 anchors / 127 multi-code / 4 role-mapped forward, max code length 10,
61 pure byte-fallback rows, zero unresolved and zero byte mismatches in both
directions. The assertions use floors (`multi >= 32`, `max_code_length >= 3`) so
a `tokenizers` version bump cannot make the suite brittle. The minimal training
helper is duplicated from the pattern in
`tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py::_build_donor_tokenizer`;
nothing is imported across test modules.

The byte↔character map is reimplemented independently in the test and
cross-checked against `tokenizers.pre_tokenizers.ByteLevel.alphabet()`, so the
byte-fidelity assertions are not circular. Byte fidelity is additionally checked
a second, map-free way: for rows where both sides are plain model rows,
concatenating the code's raw *token strings* must reproduce the input token
string exactly.

Real-scale behavior (§2) was verified manually against the checked-in LFM2 and
GPT-2-style tokenizers; that check is **not** in the test suite because it
depends on files under `.codex-test-runs/` and `runs/`, which are not test
fixtures. Re-run it by hand if the segmenter changes.

---

## 8. Deliberately left out

- **No planner, registry, executor, or operator change.** The artifact is inert
  until someone lands §5.
- **No edit to `hf_tokenizer_bridge.py`.** The alias properties in §4 are the
  only thing I would have added, and they are not needed by anything in 2D.
- **No `authenticated_live_session` analogue.** The strict bridge's session
  object exists to make repeated *inner* verification cheap during a long
  selective-runtime capture. Nothing consumes a pair bridge in that pattern yet;
  adding it now would be an unexercised 200 lines. `verify_live_tokenizers`
  costs 2.24 s at LFM2 scale, so a session cache is only worth building once
  there is a caller that calls it in a loop.
- **No lm-head-specific helper.** The transport builders are width-agnostic and
  work identically on an output head; a separate function would be duplication.
- **No frequency estimation.** `compile_pair_bridge_occurrence_weighted_transport`
  takes an exact integer occurrence table and hashes it into the report. Where
  that table comes from (corpus counts, parent logit mass) is a caller decision
  and must stay auditable.
- **No cross-tokenizer *sequence* alignment.** This bridge aligns vocabulary
  rows, not encoded token streams. Aligning two tokenizations of the same text
  is `hf_byte_span_trace.align_byte_span_traces`, which already exists and is a
  different job.
