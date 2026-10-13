# Sequential native state handoff

`native_state_inheritance.py` supplements the full shared-r16 handoff. Both
exports must bind the same finalized engine checkpoint, immutable base, parent
configuration, native module, producing run and step. Native inventory pins the
actual source/compose plan, schedule, interface bands and table dimensions.

The first D3a unit explicitly initializes both policies:

```json
{"shared_r16":{"mode":"initialize","first_refit":true},
 "native_state":{"mode":"initialize","first_refit":true}}
```

Every later unit/stage supplies separate pinned `manifest.json` files from the
same predecessor's `shared-r16` and `native-state` exports:

```json
{"shared_r16":{"mode":"inherit","source":{"path":"PREVIOUS/shared-r16/manifest.json","sha256":"ACTUAL_FILE_SHA256"}},
 "native_state":{"mode":"inherit","source":{"path":"PREVIOUS/native-state/manifest.json","sha256":"ACTUAL_FILE_SHA256"}}}
```

`initialize_stage` runs after shared-bank initialization and before engine census
or optimizers. It copies touched unit parameters into their existing objects and
keeps source-frozen coefficients unchanged. Prior bridge/emission masters stay
frozen outside the fresh active delta lists. Source-pinned instance forward
adapters add inherited and current stage deltas in fp32, then cast the combined
delta once exactly as native `SliceDelta.full` does. This also avoids overwriting
disjoint types through a repeated functional-call parameter key.

Music embedding/head masters retain written rows and protect immutable ledger
and reserved rows. The music engine copies inherited fp32 masters before making
fresh LazyRowAdam state. Other stages retain those tables without owning them.
Each stage exports all earlier touched families plus its current learned state;
no previous Adam state crosses a stage boundary. Exact resume within one stage
still uses its own normal checkpoint and unchanged inheritance identity.

The metadata preflight checks actual file hashes, safetensors extents, matching
producer checkpoint and shared-bank seals. Runtime admission checks shapes,
finite fp32 masters, exact native ownership bands and frozen states. Export
verifies the actual current checkpoint's trainable tensors and forbids changed
frozen native parameters or inherited masters. Existing artifacts are preserved.

This implements stage-to-stage runtime continuity. It does not compose a native
registered production checkpoint, implement a D4 union optimizer, establish
cross-modality quality, or replace independent content/retention evaluation.
The optional genuine tiny-Hybrid CPU test runs actual media, music and vision
optimizers, checks handoff forward equality, and confirms fresh optimizer state,
protected reserves and unique live trainable ownership. Synthetic test teacher
targets measure transport and differentiation only.
