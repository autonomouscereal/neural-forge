# NOTES-1A — wiring requests outside my file ownership

Everything below is a change I deliberately did **not** make because the file is
outside the ownership boundary I was given. Each item is self-contained.

## 1. `src/neural_forge/hf_lfm2_dense_budget_model_fold.py` — thread the writer

`ChildClosedLoopCompiler` now takes an optional
`restart_capsule_config=<AcceptedLayerRestartCapsuleWriter>`. Capsule writing is
OFF when it is `None`, which is the current behaviour of this call site
(`hf_lfm2_dense_budget_model_fold.py:3646`). To enable accepted-layer restart
capsules for the dense body fold, build the writer next to the adapter and pass
it in:

```python
from neural_forge.hf_lfm2_closed_loop_restart_capsule import (
    Lfm2ClosedLoopRestartCapsuleWriter,
    latest_lfm2_closed_loop_restart_capsule,
    lfm2_restart_capsule_split_sha256,
    load_lfm2_closed_loop_restart_capsule,
)

compiler_config = replace(
    self.config.closed_loop_compiler_config,
    fold_body_reclosure=(
        self.config.body_reclosure_mode == LFM2_BODY_RECLOSURE_MOVING_GAUGE_V3
    ),
)
restart_writer = None
if restart_capsule_directory is not None:
    restart_writer = Lfm2ClosedLoopRestartCapsuleWriter(
        adapter=adapter,
        directory=restart_capsule_directory,
        run_id=<catalog/run id>,
        child_name=<child name>,
        trajectory_split_sha256=lfm2_restart_capsule_split_sha256(
            adapter, compiler_config
        ),
        run_schema=RUN_SCHEMA,
        rotation_keep=2,
        fold_progress_sink=fold_progress,   # the live _Lfm2DenseFoldProgress
    )
closed_loop_report = ChildClosedLoopCompiler(
    adapter,
    compiler_config,
    progress_sink=fold_progress,
    restart_capsule_config=restart_writer,
).run()
```

Resuming is caller-side and must happen **before** the adapter is constructed,
because `Lfm2DynamicClosedLoopAdapter.__init__` snapshots the child's input
vocabulary digest into the identity bindings the capsule pins:

```python
reference = latest_lfm2_closed_loop_restart_capsule(restart_capsule_directory)
resume = None
if reference is not None:
    resume = load_lfm2_closed_loop_restart_capsule(
        reference, child_model=<fresh child>
    )
# ... construct the adapter from that already-restored child, then pass
# resume=resume to Lfm2ClosedLoopRestartCapsuleWriter(...).
```

Important: a mid-body resume is refused when `fold_body_reclosure` is enabled and
the adapter runs `moving_gauge_v3`. See the module docstring of
`hf_lfm2_closed_loop_restart_capsule.py` for why (the reclosure stage asserts that
*this* process installed every layer). If the dense fold wants both, it must
resume only at layer 0 or defer reclosure to a separate stage.

## 2. `src/neural_forge/hf_lfm2_dense_budget_model_fold.py` — public chain head

`_Lfm2DenseFoldProgress` exposes `binding_sha256` publicly but keeps
`_sequence` and `_previous_sha256` private.
`Lfm2DenseFoldProgressChain` in the capsule module has to read both to record the
fold-progress chain position a capsule binds to, so it currently duck-types the
private names and raises a precise error if they are absent.

Requested (purely additive) on `_Lfm2DenseFoldProgress`:

```python
    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def head_sha256(self) -> str:
        return self._previous_sha256
```

I will switch `Lfm2DenseFoldProgressChain.as_dict` to prefer the public names and
keep the private fallback for one release.

## 3. `experiments/run_lfm2_dynamic_catalog.py` — CLI surface

To make r10+ restartable, the catalog runner needs two options threaded to the
dense fold: a restart-capsule directory and a rotation count (default 2). No
other change is required; discovery, authentication, rotation, and the resume
admission all live in `hf_lfm2_closed_loop_restart_capsule.py`.

## 4. Optional: default the capture merge on

`ClosedLoopCompilerConfig.merge_adjacent_layer_captures` defaults to `False`. It
is proven equivalent (see the report accompanying this change) but it does change
sealed run-report content and the `child_replay_capture_calls` audit counter for
every existing caller, so flipping the default is a deliberate call for whoever
owns the callers, not for this change.
