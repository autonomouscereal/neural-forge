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
