# Shared-capture full60 development folding

`c16_qimg_stack_fold.py` reads the sealed multi-block capture V4 explicitly. It
does not pretend the new journals are legacy single-block receipts. One bounded
metadata session binds original source/input/runtime identity, completed shared
forwards and resume ancestry. Each block then loads only its own verified FP8
coefficients and complete original BF16 MLP rows. Image-disjoint fit/held images
remain mandatory; held rows never select channels or fit coefficients.

Each block runs the unchanged corrected development recipe once, writes a new
36-key source block/time file plus exact integer channel selections, and reloads
both through the unchanged actual joint reader. Persisted tensors are scored
without refitting. Finite fidelity, cosine, update, rank, conditioning and
compression observations are diagnostics. Shape, storage, source-byte identity,
file/hash completeness and actual nonfinite tensor/arithmetic failures refuse
publication. Historical receipts and recipes remain untouched.

The canonical exclusive `fold_receipt.json` is a component publication marker.
Immutable snapshots and the final `stack_fold_manifest.json` state exactly which
blocks completed. A failed fit preserves prior components and partial artifacts.
Resume uses a new nonoverlapping directory and exact previous manifest SHA;
completed component identities are reloaded, never fitted again. A subset stays
partial and explicitly lists the remaining source blocks.

After all60 blocks complete, assembly streams the physical safetensors data
ranges into `plain_qimg_full60.safetensors`, with full tensor hash readback before
exclusive publication. It contains exactly1920 block tensors, shared4 timestep
tensors once, and9 original source globals. Sixty shared-time copies in component
files provide no parameter/compression credit. Source names/shapes were recovered
from the actual1933-key parent header SHA
`bd90f4d6af8d235cb3b46ccf5f426abb351995e28e852b04fc678a541d87158e`.

```console
python -B c16_qimg_stack_fold.py \
  --capture-manifest /new-capture/stack_capture_manifest.json \
  --capture-sha256 EXACT_CAPTURE_SHA \
  --policy-path /pinned-policy/policy.json --policy-sha256 EXACT_POLICY_SHA \
  --out /new-fold/full60 --blocks 0-59 --device cuda:0
```

The policy is an explicitly pinned existing diagnostic reference, not an invented
fresh preregistration. The tool allocates no GPU, starts no service and overwrites
no model. Runtime scheduling remains the owner's obligation.

The companion `c16_qimg_native_stack.py` loads the exact complete block receipts,
source/global bytes and physical1933-key state into actual frozen native modules.
Its metadata-only activation mode is explicit: it rechecks source/candidate bytes
and bound capture journals, while the fitter owns activation payload proofs.
The complete path is plain QIMG; ImageEdit and behavioral acceptance remain
unproven. `accepted` and `capability_accepted` remain false.

Focused CPU integration executes two actual tiny synthetic source forwards shared
by all60 blocks, all120 real affine fits, exclusive serialization/readback,
physical1933 consolidation, native full60 load and all120 MLP calls. Additional
checks cover interrupted partial resume, source/activation tampering and honest
missing coverage. These tests use explicit synthetic FP8 fixtures, not original
Comfy checkpoint weights or actual encoded images.
