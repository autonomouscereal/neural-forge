# Standalone plain-QIMG native backbone

`c16_qimg_native_stack.py` loads a complete source-space folded backbone through
the actual accepted native QIMG modules. It retains the real image/text streams,
joint attention, source eager RoPE, additive mask, six-way residual modulation,
both biased GELU MLPs per block, shared time embedding, input projections and
output normalization/projection. The module is frozen and exposes no training,
sampler, VAE, ImageEdit conditioning, host chart or Chimera child interface.

The existing seven-file consumer freeze V1 is unchanged. The new wrapper pins
its four executable consumer files to their accepted hashes before construction.
It reuses `_Block`, `_Time`, `_Linear` and `_RMSNorm` from that implementation;
there is no generic residual-width surrogate or Comfy dependency.

## Exact recovered source contract

The original parent header was recovered as `qimg_parent_header.raw.json`,
234040 bytes, SHA256
`bd90f4d6af8d235cb3b46ccf5f426abb351995e28e852b04fc678a541d87158e`.
The source-only installed `model.py` snapshot is 26510 bytes, SHA256
`14c805af8da13d31094c2c704e413cc282ef43f4afe17f77de9070bdca301f28`.
These files are in `owner_fold_policy_oct06/remaining_donor_metadata`. Recovery
receipt SHA256 is
`e366882545b8b6b6f196818f39971ba55a8c8b80c883ebbb26f48babc7e85888`.
No model bodies were read by the wrapper's authoring lane.

The exact nine remaining source tensor names are:

- `img_in.weight`, `img_in.bias`
- `txt_norm.weight`, `txt_in.weight`, `txt_in.bias`
- `norm_out.linear.weight`, `norm_out.linear.bias`
- `proj_out.weight`, `proj_out.bias`

They are all FP8 E4M3 storage in the original header. Production width is 3072,
text context width 3584, head width 128 and axes 16/56/56. The actual source
patch order is channel then the two local patch axes, with text rows before
image rows in joint attention. Output modulation splits **scale, shift** and
uses the source `addcmul` order. Position generation preserves the source BF16
`linspace` boundary even for asymmetric grids; replacing it with F32 `arange`
can change large positions.

The runtime physically owns `60*32 + 4 + 9 = 1933` parameter tensors. It shares
one time module across all 60 blocks and computes time and rotation once per
forward. External per-block fold files still contain their own four time
tensors; their physical file storage is distinct from deduplicated runtime
accounting. Full keep remains functionally valid. MLP compression ratios are
reported diagnostics, with no fidelity, update, cosine or compression veto.

## Versioned folded-stack reader

```python
from c16_qimg_native_stack import load_qimg_native_stack

model = load_qimg_native_stack(fold_stack_path, exact_fold_stack_sha256,
                               device="cpu")
prediction = model(model_input, context, sigma, text_mask=additive_bf16_mask)
report = model.runtime_report()
```

The caller must make the exact accepted consumer `src` and the immutable NF
tools available on `PYTHONPATH`. There is no dependency install or service call.

The input schema is the fold driver's
`neural_forge.qimg_folded_stack.v1`. The reader requires explicit complete
ordered 0..59 requested/completed/required coverage, no missing blocks,
`full_stack_complete:true`, `status:complete`, false acceptance/capability flags,
the current owner policy, externally pinned capture/policy references, one exact
fold receipt per block and the original capture's 13-key global source record.
A directory name or a successful status flag cannot establish completeness.

The explicit Capture V4 adapter uses `CaptureIndex` to bind root/journal/source
metadata once, `load_block_inputs(..., load_forwards=False)` to check each actual
source block/time file, and the unchanged native joint `load_candidate` to check
candidate/selection bytes. It then constructs the existing strict 36-key native
component manifest solely from those verified tensors and references. It does
not fabricate a legacy single-block capture receipt. Activation payloads were
checked by the fold driver and are not reread during native coefficient loading.
That scope is recorded in the runtime report.

The complete driver receipt must include `functional_validation_complete:true`,
no error and its physical `standalone_state`. `readback_consolidated_state` reads
the actual saved 1933-key safetensors, verifies its complete tensor/file hashes,
shape/storage islands, finite coefficients and equality to the already verified
components/globals, then strictly copies those bytes into the real native
modules' owned allocations. It checks module bytes after loading. A missing tensor, duplicate layer,
wrong hash/shape/storage, changed frozen source tensor, mismatched shared time or
actual nonfinite coefficient/activation remains a functional failure. No files
are deleted or historical receipts rewritten by this reader.

## Forward and validation scope

Inputs are complete finite BF16 `[1,16,1,H,W]` plain model inputs with even H/W,
BF16 `[1,T,3584]` context, native F32 `[1]` sigma in `[0,1]`, and an optional
nonpositive BF16 additive text mask with a valid row. Total joint context is
bounded to the existing 2048-token capture contract. External autocast, training
and global dtype recasts are refused. The returned prediction has the model
input shape. `return_taps=True` returns complete per-block real native taps plus
global patch/input/time/rotation/normalization/projection/prediction tensors.

`test_c16_qimg_native_stack.py` provides `tiny_native_fixture()` for a synthetic
60-block source baseline: width/head 32, axes 8/12/12, source MLP inner 128 and
context width 32. Its actual native forward is suitable for the coordinator's
real fitter/driver integration fixture. The direct folded fixture tests exercise
all 120 MLP calls, one time call, exact source patch/inverse order, complete
parameter/storage accounting, component-loop parity, both evolving streams,
late-block MLP effect, complete native saved-state readback, source identity and
nonfinite/input/coverage failures. They read no original model checkpoint and
are not production execution, source-oracle parity or capability evidence.

The authoring Windows Python has no Torch, pytest or safetensors. Syntax checks
are local; actual Torch integration is dispatched by the coordinator through the
already available isolated CPU image. Authentic all60 source capture/folds,
1933-key physical export/readback and pinned source-forward parity still require
the real execution lane. A loaded standalone backbone remains unevaluated and
uninstalled into the all-source child.
