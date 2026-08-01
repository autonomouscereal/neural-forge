# NOTES-W6B — one selector, three lanes

Written 2026-07-31 by the shared-selector wave. Branch
`codex/lrc-blank-compression`, base `35d6e86f`. Fold-only, nothing committed.

This wave pays the debt that `NOTES-W3B.md` item 7 opened and `NOTES-W5C.md`
item 2 renewed:

> if a third lane needs `_sparse_anchor_regression` /
> `_selection_from_statistics`, extract them into a modality-neutral module and
> have both lanes delegate — as a single reviewable no-op diff, not piecemeal.

`NOTES-W5C.md` then added the acceptance bar, because the video lane made the
copies three: *"it needs a wave that owns all three modules and can prove
byte-identical cache keys before and after."* That proof is Section 4 below and
it is the reason to trust the rest of this note.

---

## 1. What the three copies actually were

Before touching anything, the three lanes were diffed structurally — parsed to
ASTs, lane vocabulary (`DynamicNeoFoldError` / `NeoLayerFoldSelection` and
their audio and video counterparts, plus the label strings `NEO`, `Native
audio`, `Native video`) substituted out, docstrings dropped — and the results
compared. The finding, function by function:

| function | verdict |
|---|---|
| `_sparse_anchor_regression` | **identical in all three**, including every string literal. Only the docstrings differ. |
| `_selection_from_statistics` | **identical in all three** apart from three message strings and the constructed selection dataclass. Every operation, argument order, dtype promotion, and iteration order matches. |
| `*LayerFoldSelection.__post_init__` | **identical in all three** after label substitution. |
| `*LayerFoldSelection.survivors` / `.anchor_count` / `.compensation_terms` | **identical in all three.** |
| `_positive_int` / `_nonnegative_int` | **identical in all three**, messages included. |
| `*LayerFoldSelection.as_dict` | **divergent, by design.** See §2.1. |
| `_finite` | **divergent, one message.** See §2.2. |
| `_tensor_descriptor` | **divergent, semantically.** See §2.3 — deliberately **not** unified. |

So the extraction target was real: roughly 300 lines per lane of genuinely
byte-identical numerics, carried three times.

---

## 2. Divergences found, and what was done about each

### 2.1 `as_dict` carries a lane-specific front-end key — left alone

`NeoLayerFoldSelection.as_dict()` emits no front-end key. The audio copy emits
`"audio_front_end_copied_exactly": True`; the video copy emits
`"video_front_end_copied_exactly": True`. That key is part of the sealed body
(`selections` inside each fold cache manifest), so it is a schema fact, not
duplication. `as_dict` stays in each lane, unchanged, and the shared module
never sees it.

Key *order* is irrelevant to the seal — `canonical_sha256` sorts keys — but
key *presence* is not, which is exactly why this one is not shared.

### 2.2 `_finite` disagrees on one message — preserved, not unified

For a value that is not a real number at all:

* NEO raises `"<name> must be finite."`
* audio and video raise `"<name> must be a real number."`

Both then raise `"<name> must be finite."` for a non-finite float. This is a
pre-existing divergence on a fail-closed path, and unifying it would change
observable behavior in at least one lane for no benefit. So the shared
normalizer does **not** define its own `_finite`; each lane injects its own
through `ModalityFoldSelectionVocabulary.finite`, and
`tests/test_dynamic_modality_fold_selection.py` pins both texts so the
divergence cannot be "tidied up" by accident.

If a later wave wants them unified, that is a deliberate, separately reviewed
change — not a side effect of a de-duplication.

### 2.3 `_tensor_descriptor` is semantically divergent — STOPPED, not unified

This one is reported rather than merged, per the wave's own rule.

```python
# dynamic_neo_fold.py
tensor = value.detach().to(device="cpu").contiguous()
return {"shape": ..., "dtype": str(tensor.dtype), "sha256": _tensor_sha256(tensor)}

# dynamic_audio_fold.py and dynamic_video_fold.py
payload = _canonical_float32_bytes(value)
return {"shape": ..., "dtype": "float32", "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest()}
```

The audio and video lanes canonicalize to float32 bytes and add a
`byte_count` field; the NEO lane hashes the tensor at its native dtype and
emits no `byte_count`. These descriptors go straight into each lane's fold
cache `_body()`, so they *are* the cache key. Unifying them would reseal every
existing artifact in at least one lane, which is precisely what this wave was
chartered to avoid. `_tensor_descriptor` therefore stays triplicated and
divergent, in all three lanes, untouched.

**This is now open debt.** Whether the NEO lane should adopt the
float32-canonical descriptor (or the reverse) is a real question with a real
answer, but it is a schema decision — it needs a `FOLD_CACHE_SCHEMA` bump and
a migration story, not a refactor.

---

## 3. What landed

### `src/neural_forge/dynamic_modality_fold_selection.py` (new, 434 lines)

The single copy. It names no modality and imports nothing from any lane.

* `ModalityFoldSelectionVocabulary` — the whole lane-specific surface:
  `error_type`, `subject_label` (`"NEO"` / `"Native audio"` / `"Native
  video"`), `selection_label` (`"NEO"` / `"native audio"` / `"native video"`),
  and the lane's own `nonnegative_int`, `positive_int`, `finite`. Two labels
  rather than one because NEO capitalizes in both sentence positions and the
  other two lanes do not; that is how all nine message strings are reproduced
  character for character.
* `sparse_anchor_regression(...)` — the numerics verbatim: the zero-norm early
  return, the single-anchor closed form with its `[-4, 4]` clamp, and the
  greedy orthogonal-matching loop with its `trace(gram)/k * 1e-6` ridge and
  per-step re-solve. Nothing reordered, nothing "simplified".
* `selection_from_statistics(...)` — the heldout-blind low-importance ranking:
  float64 promotion, the malformed-statistics guard, `argsort(..., stable=True)`
  over `mean_abs * down_norm`, the endpoint-RMSE weighting, and `min(choices)`
  on the score tuple. It takes a `selection_factory` and calls it with exactly
  the keyword arguments the three inline copies used.
* `normalize_sparse_anchor_selection(selection, *, vocabulary)` — the shared
  body of all three `__post_init__` methods, mutating the frozen dataclass
  through `object.__setattr__` in the same order as before.
* `survivors_for(...)` and `sparse_compensation_terms(...)` — the compensation
  accessors.

`torch` is still imported lazily inside the two numeric functions, as it was
in each lane.

### The three lanes — delegation only

`dynamic_neo_fold.py` (−303/+41), `dynamic_audio_fold.py` (−318/+43),
`dynamic_video_fold.py` (−318/+43). Each gained one import block and one
module-level `_SELECTION_VOCABULARY`; `__post_init__`, `survivors`,
`compensation_terms`, `_sparse_anchor_regression` and
`_selection_from_statistics` became one-call bindings.

Deliberately **not** changed, so this reads as a no-op diff:

* The private names `_sparse_anchor_regression` and
  `_selection_from_statistics` still exist in each lane with their original
  signatures. Every internal call site — `derive_*_fold_cache`,
  `build_*_fold_cache`, and each cache's `verify()` — is untouched.
* No mixin, no new base class. The three selection dataclasses keep their
  exact MRO, field order, `__init__` signature, equality, and `as_dict`.
  Adding a shared base would have been tidier and was rejected for that reason.
* `_positive_int`, `_nonnegative_int`, `_finite`, `_digest`, `_canonical_json`,
  `_tensor_descriptor`, `__all__`, every schema constant, and every seal format
  are unchanged.
* `anchor_count` stays an inline `len(self.anchor_neurons)` in each lane. It is
  one expression; routing it through a shared helper would have added a call
  and removed nothing.

### `tests/test_dynamic_modality_fold_selection.py` (new, 653 lines, 43 tests)

Additive; no existing test file was edited. It pins:

* `REGRESSION_DIGEST` — `sparse_anchor_regression` over **every** removable
  column of 11 fixtures.
* `SELECTION_DIGEST` — `selection_from_statistics` over all 11 fixtures in all
  three lanes.
* `SELECTION_STRUCTURE` — the chosen neuron, anchor order, and target indices
  per fixture, compared with **no** float tolerance at all.
* Cross-lane equality: identical statistics must give all three lanes the same
  selection once the front-end key of §2.1 is dropped.
* Every fail-closed message, per lane, verbatim — including the §2.2
  divergence.

The fixtures are RNG-free. Every tensor comes from a fixed splitmix64 stream
that yields exactly-representable float64 values, so the pins do not depend on
a torch version, a BLAS backend, or a platform. Floats are compared through a
12-significant-digit canonical form: loose enough to survive last-bit
reduction-order noise on other hardware, tight enough that any change of
clamp, ridge, tie-break, or scoring trips it.

The fixtures cover `anchor_count` 1 (closed form), 2/3/4 (matching-pursuit
loop), the clamp path where `anchor_count` exceeds both the survivor count and
the sketch row count, float32 input promoted to float64, zeroed sketch columns
(the `removed_norm <= 0` early return), duplicated/collinear columns (the
ridge), and the minimum legal width of two neurons.

---

## 4. The byte-identity proof

Three independent comparisons, all against `35d6e86f` checked out into a
separate worktree.

### 4.1 Direct numerics — byte-identical

A probe called `_sparse_anchor_regression` and `_selection_from_statistics`
directly, in all three lanes, over the 11 fixtures — plus every fail-closed
message and 14 sparse-anchor validation failures per lane — and dumped full
`repr()` precision to JSON.

```
baseline (HEAD worktree)  sha256 e51cefa3fc73832b224e32ceae6088046c37e370ed8b952bc7b0ede9ee7d9ad2
after    (refactored)     sha256 e51cefa3fc73832b224e32ceae6088046c37e370ed8b952bc7b0ede9ee7d9ad2
```

Byte-identical, including every message string.

### 4.2 The pinned digests encode HEAD's numerics, not new ones

The two digests committed in the new test file were recomputed from HEAD's
three *private* copies and from the refactored shared module:

```
HEAD        REGRESSION 7647bd3f...7470   SELECTION df2e698d...b585a
refactored  REGRESSION 7647bd3f...7470   SELECTION df2e698d...b585a
```

The pins are HEAD's behavior. They are not a snapshot of whatever the refactor
happened to produce.

### 4.3 Sealed artifacts through the real lane fixtures

A pytest plugin (outside the repository) wrapped every fold-cache constructor,
every trace-statistics-cache constructor, and all three receipt writers, then
recorded — for every artifact the five lane suites build — the cache key,
fingerprint, source cache key, anchor/candidate counts, per-selection
`canonical_sha256`, the full selection payloads, the compensation terms, the
survivor lists, a digest over the sealed statistics tensors, and the complete
receipt body. 31 records: 11 fold caches, 16 statistics caches, 4 receipts.

Result at field level:

```
HEAD(worktree)  vs REFACTORED : IDENTICAL   (masked, see below)
HEAD(main cwd)  vs REFACTORED : IDENTICAL   (masked, see below)
```

**Every cache key, every fingerprint, every selection digest, every selection
payload, every compensation term, and every statistics digest is byte-identical
— unmasked, no tolerance.** Not one moved.

Five *receipt-level* fields did move, and the honest finding is that they move
at HEAD too:

| field | behavior |
|---|---|
| `artifact_certificate_sha256` | differs between two runs of unmodified HEAD |
| `receipt_sha256` | differs (downstream of the above) |
| `behavior_gate_manifest_fingerprint` (audio, video) | differs between two runs of unmodified HEAD |
| `behavior_report.behavior_gate_manifest_fingerprint` | differs (same value) |
| `behavior_report_sha256` | differs (downstream) |

Three control runs of **unmodified HEAD code** — worktree cwd, main cwd, and
two further repeats in the same cwd — produce a different value for each of
these five fields on every run, and produce *identical* values for everything
else. So the mask is exactly the set of fields that are already
nondeterministic, and it was validated before it was applied:

```
CONTROL A  HEAD run1 vs HEAD run2 (same code, same cwd), masked : IDENTICAL
CONTROL B  HEAD run2 vs HEAD run3 (same code, same cwd), masked : IDENTICAL
CONTROL C  HEAD worktree vs HEAD main cwd,               masked : IDENTICAL
PROOF 1    HEAD worktree vs REFACTORED,                  masked : IDENTICAL
PROOF 2    HEAD main cwd vs REFACTORED,                  masked : IDENTICAL
```

The cause is pre-existing and is a property of the *test fixtures*, not of the
lane code: `dynamic_behavior_gate._stable_resource_value` renders a `Path` as
its literal string, and `_callable_resource_fingerprint` folds closure cell
contents into the callable identity. The lane tests build evidence under
`tmp_path` / `tmp_path_factory`, whose directory index advances every run, so
the saver/loader closures and the audio and video `MediaReference.uri` values
differ run to run. NEO's gate manifest does not move because its cases carry no
`MediaReference`; its artifact certificate does, because the saver closure
captures a temp path.

**This is worth recording as its own debt** (§6 item 1): these are advertised
as seals, and today a bit-identical rebuild of the same child from the same
evidence does not reproduce them. Nothing in this wave caused it and nothing in
this wave fixed it.

---

## 5. Test results

```
$ .venv/Scripts/python.exe -m pytest tests/test_dynamic_neo_native_vision.py \
    tests/test_native_audio_lane.py tests/test_native_audio_executable_lane.py \
    tests/test_native_video_lane.py tests/test_native_video_executable_lane.py \
    tests/test_dynamic_modality_fold_selection.py -q
119 passed
```

The five suites that had to stay green are the 76 of those; they passed
unmodified, and they passed identically at HEAD (76 passed) in every control
run above. The new file contributes 43.

---

## 6. Deferred — what this wave did not do

1. **The receipt fields that are not reproducible seals.**
   `artifact_certificate_sha256`, `receipt_sha256`,
   `behavior_gate_manifest_fingerprint`, and `behavior_report_sha256` change
   run to run at HEAD because absolute temp paths reach
   `_stable_resource_value` through closure cells and `MediaReference.uri`.
   Either those fingerprints should exclude filesystem location, or the
   fixtures should bind content rather than path. Both files involved
   (`dynamic_behavior_gate.py`, `dynamic_fold_artifact.py`) were outside this
   wave's ownership.
2. **`_tensor_descriptor` stays divergent** — §2.3. Semantic, seal-affecting,
   and correctly out of scope for a de-duplication.
3. **`_finite` stays divergent** — §2.2. One message, deliberately preserved
   and now pinned.
4. **The other cross-lane duplicates were left alone.** The survey also found
   `_digest`, `_canonical_json`, `_file_sha256`, `_write_exclusive_json`,
   `_live_state`, `_canonical_float32_bytes`, `default_layer_resolver`, and
   `default_projection_resolver` duplicated across two or three lanes. They are
   not selection numerics, they do not feed the selector, and folding them in
   would have widened a diff whose whole value is that it is provably a no-op.
   They are a fine next wave, on the same acceptance bar.
5. **No lane's public API, dataclass shape, schema constant, or seal format
   changed**, which also means none of the deferred items from `NOTES-W3B.md`
   or `NOTES-W5C.md` — the `EXPERIMENTAL` status, the real answer decoder,
   wider fold scopes, a production video parent, temporal-specific evidence —
   moved at all. Only item 7 / item 2 is closed.
