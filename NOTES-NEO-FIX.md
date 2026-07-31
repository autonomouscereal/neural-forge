# NEO frozen-reload fix — deferred out-of-scope change

## Status

The failing test `tests/test_dynamic_neo_native_vision.py::test_neo_dynamic_fold_cache_stack_behavior_artifact_and_receipt`
is **fixed** (see `tests/test_dynamic_neo_native_vision.py::_tiny_model_loader`).

The **production** NEO standalone loader has the identical defect and is still
broken. It lives in `experiments/run_neo_native_vision_dynamic_fold.py`, which
is outside the edit scope of this task, so the exact change is written here
instead of applied.

## Why this blocks the vision-modality certification

`publish_dynamic_fold_artifact` (`src/neural_forge/dynamic_fold_artifact.py:801`,
`_assert_frozen_inference_parameters`) requires the declared `model_loader` to
return a model whose parameters are **all** `requires_grad=False`. The certificate
field `fresh_reload_parameters_frozen` is the published proof of that property, and
`load_dynamic_fold_artifact_certificate` rejects any v4 certificate where it is not
`True` (`dynamic_fold_artifact.py:2258`).

`DirectNeoCheckpointIO.load` → `load_local_neo_model` only calls `model.eval()`,
which sets module training mode and does **not** touch `requires_grad`. The freshly
constructed `runtime.model_class(config)` therefore hands back fully trainable
parameters.

Measured against the tiny concrete snapshot built by
`tests/test_run_neo_native_vision_dynamic_fold.py::_write_tiny_concrete_snapshot`:

```
load_local_neo_model        -> trainable 3/3, training_mode=False
DirectNeoCheckpointIO.load  -> trainable 3/3, training_mode=False
_assert_frozen_inference_parameters(...) ->
    DynamicFoldArtifactError: Stage-path fresh reload inference parameters are not all frozen.
```

So a real NEO fold run through `run_neo_native_vision_dynamic_fold.py` will abort at
the stage-path fresh reload before it can ever publish a certified artifact.

## Exact proposed change

File: `experiments/run_neo_native_vision_dynamic_fold.py`
Function: `load_local_neo_model` (ends at line 605)

```diff
     native_eval._force_neo_eager_prefill_attention(model)
+    # Fold-only runtime: the local NEO loader is the published artifact's
+    # inference contract, so it must return frozen parameters. eval() only
+    # sets module training mode and leaves requires_grad untouched, which
+    # fails _assert_frozen_inference_parameters during artifact publication.
+    model.requires_grad_(False)
     model.eval()
     return model
```

`load_local_neo_model` is the right seam rather than `DirectNeoCheckpointIO.load`:
it is the single local NEO runtime loader, so one change covers both the checkpoint
I/O reload path and the direct eval path in `experiments/eval_neo_native_vision.py`.
Neither file contains any `requires_grad` reference today.

This is safe for the fold-only repo: `DynamicFoldExecutor` already freezes and then
restores parameter trainability around evaluation
(`src/neural_forge/dynamic_fold_executor.py:2093-2110`, `2253-2270`), and
`fingerprint_model_state` excludes trainability, so a frozen reload still matches the
accepted in-memory child byte for byte. Only `fingerprint_model_runtime_state`
observes `requires_grad`, and it is only ever compared reload-against-itself.

## Suggested regression guard (also out of scope)

In `tests/test_run_neo_native_vision_dynamic_fold.py`, inside
`test_direct_local_runtime_checkpoint_and_tokenizer_roundtrip`, after
`reloaded = checkpoint_io.load(destination)`:

```python
assert all(not parameter.requires_grad for parameter in reloaded.parameters())
```

That pins the loader contract at the production seam so the certification gate is
never the first thing to discover a regression.

## Considered and rejected

Auto-freezing the caller-supplied loader inside
`src/neural_forge/dynamic_neo_fold.py` (`NeoNativeVisionFoldStackAdapter.build_stack`)
would make the failure disappear for both the test and production in one in-scope
edit. It was rejected: the wrapper does not travel with the published standalone
artifact, so a downstream consumer loading the artifact with the declared runtime
would still get trainable parameters while the certificate claimed
`fresh_reload_parameters_frozen=True`. That converts a real proof into a vacuous
one for the NEO lane only, and it would defeat the intent of
`tests/test_dynamic_fold_artifact.py::test_publication_rejects_trainable_fresh_reload_parameters`.
The loader must satisfy the contract, not be silently corrected into it.

---

# LFM2 carrier lane — `block_ff_dim` reload alias (second failure, same check family)

Applies to `tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py::test_toy_carrier_lane_folds_through_every_reachable_stage`,
which failed after stage `dense_closed_loop_completed` with:

```
Stage-path fresh reload structural topology differs from the compiled child;
ParameterSpec.trainable is the only excluded graph leaf.
```

## Differing leaves

Exactly **one** leaf differs, and it is not a parameter, storage-alias group, or
tokenizer property:

```
config_fingerprint: '06d02fe4…' != '23957b4f…'
```

Diffing the two `_config_payload` mappings that produce those hashes gives a
single differing configuration key:

```
ONLY-IN-RELOAD  block_ff_dim = 11
```

The compiled in-memory child config has no `block_ff_dim`; the stage-path
reloaded child does.

## Root cause — loader-side

`block_ff_dim` is a **lossy serialized compatibility alias** for
`intermediate_size`. `transformers/models/lfm2/configuration_lfm2.py:91` pops it
during `Lfm2Config.__post_init__`:

```python
self.intermediate_size = kwargs.pop("block_ff_dim", self.intermediate_size)
```

so a config constructed from a checkpoint that declared the alias comes back
**without** it. Verified on the installed transformers:

```
fresh config has block_ff_dim in to_dict(): False
after setattr, to_dict() has block_ff_dim: True 11
kwarg alias -> intermediate_size: 11 | block_ff_dim in to_dict(): False
```

`experiments/run_lfm2_dynamic_catalog.py:5935-5936` compensates for that loss —
but **unconditionally**:

```python
        if getattr(model.config, "block_ff_dim", None) is None:
            model.config.block_ff_dim = int(model.config.intermediate_size)
```

`atomic_save_local_checkpoint` (`experiments/run_lfm2_residual_attention_fold.py:3646`)
writes `model.config.to_dict()` verbatim, so `config.json` is a faithful image of
the compiled child. In this lane that child config never carried the alias, so
the loader **fabricates** a config key that was never published. Instrumented at
the exact reload that fails:

```
config.json declares block_ff_dim: False (value=None)
config.json intermediate_size: 11
loader returned config.block_ff_dim: 11
in to_dict(): True
```

`setattr` puts the key in `config.__dict__`, `to_dict()` emits it,
`config_fingerprint` moves, and the artifact gate correctly fails closed.

This is a **pre-existing loader defect newly reached**, not a regression: the
injection dates to `275cd218` (2026-07-17) and has never changed. The lane simply
never got this deep before. The check itself is right — the reloaded graph
identity must be a pure function of the published bytes, and today it is a
function of a loader-side assumption.

The repo already documents the intended narrower rule
(`experiments/run_lfm2_residual_attention_fold.py:643-647`):

> Transformers 5.12.1 consumes `block_ff_dim` while constructing Lfm2Config and
> may not retain the legacy attribute in memory **even when config.json contains
> it**. […] callers that load a checkpoint restore the alias before validation so
> a subsequent save remains shape-stable.

"even when config.json contains it" is the operative condition. The catalog
loader dropped it.

## Prescribed change — `experiments/run_lfm2_dynamic_catalog.py` (do not edit; apply yourself)

In `DirectLocalArtifactIO.load`, replace lines 5935-5936:

```diff
         for parameter in model.parameters():
             parameter.grad = None
             parameter.requires_grad_(False)
-        if getattr(model.config, "block_ff_dim", None) is None:
-            model.config.block_ff_dim = int(model.config.intermediate_size)
+        # ``block_ff_dim`` is a lossy serialized alias: Lfm2Config.__post_init__
+        # pops it into intermediate_size, so a checkpoint that declared it
+        # reloads without it.  Restore it only when the published config.json
+        # actually declared it -- restoring it unconditionally fabricates a
+        # configuration leaf the compiled child never had, which moves
+        # config_fingerprint and makes the artifact structural-topology gate
+        # fail closed on a difference the artifact does not contain.
+        declared_config = json.loads(
+            (source / "config.json").read_text(encoding="utf-8")
+        )
+        if (
+            "block_ff_dim" in declared_config
+            and getattr(model.config, "block_ff_dim", None) is None
+        ):
+            model.config.block_ff_dim = int(declared_config["block_ff_dim"])
         return model
```

`json` and `Path` are already imported in that module (`json.loads` is used a few
lines above at ~5902).

**Verified.** Simulating exactly these semantics via a monkeypatch that strips the
fabricated alias when `config.json` does not declare it:

```
tests/test_run_lfm2_dynamic_catalog_carrier_e2e.py::test_toy_carrier_lane_folds_through_every_reachable_stage
1 passed in 62.99s
```

versus `1 failed` unpatched. Single factor, failure appears and disappears.

### Why loader-side and not save-side or artifact-side

* **Not artifact-side.** Excluding `config_fingerprint`, or special-casing
  `block_ff_dim`, from `as_structural_topology_dict` would weaken a real gate to
  hide a real reload difference. Forbidden, and wrong: the difference is genuine.
* **Not save-side.** "Make every child config carry the alias" also works for this
  lane (that is what the residual lane does via
  `dynamic_lfm2_adapter.py:4391`), but it has to be remembered by every
  child-config synthesis path and silently regresses the moment one forgets. The
  loader fix is general: it round-trips correctly whether or not a given lane's
  child config carries the alias.
* **Loader-side is the invariant.** The reloaded graph identity must be a faithful
  function of the serialized artifact. A loader that adds configuration keys that
  are not in the artifact is mutating published identity.

### Blast radius

`block_ff_dim` appears in `experiments/run_lfm2_dynamic_catalog.py` **only** at the
injection site; nothing in the catalog lane reads it afterwards.
`validate_target_aliases` lives in the residual lane and is not applied to the
catalog reload. The residual lane's own restore
(`run_lfm2_residual_attention_fold.py:3716-3717`) is unaffected — its `config.json`
does declare the alias (asserted at line 3700), so the conditional still fires
there. Consider applying the same conditional there for consistency; it is not
required to fix this failure.

## Applied on my side

`src/neural_forge/dynamic_fold_artifact.py` — `_assert_structural_topology_matches`
now reports the differing leaves instead of only stating that the topology moved.
This is a strictly additive diagnostic: the same conditions still raise the same
fail-closed error, with `Differing leaves: <path>: <expected> != <observed>`
appended (capped at 12 leaves). Without it this defect is a bare fingerprint
mismatch with no way to tell a config alias from a parameter-shape change.
Covered by `tests/test_dynamic_fold_artifact.py::test_structural_topology_drift_error_names_the_differing_leaves`
and `::test_structural_topology_differences_reports_every_leaf_shape`.
