# Complete native QIMG joint-block replay

`composite/tools/c16_qimg_joint_replay.py` is a source-block validator, not a
sampler, generic `adaln6` conversion, installer, or all-source child claim. It
does not run a fold or select a candidate. It consumes a previously measured,
hash-pinned dual-stream fold and its original image-disjoint capture receipts.

## Native execution, not a substitute block

The validator instantiates the exact pinned r5 `QwenImageTransformerBlock`,
`FeedForward`, `QwenTimestepProjEmbeddings`, `EmbedND`, and `ops.manual_cast`.
It pins the six capture source files plus the actual `ops.py`, `common_dit.py`,
and attention implementation. Already imported Comfy modules from another tree
refuse. The captured Torch build must match. The complete immutable runtime
image and its remaining dependency closure must still be attested by the caller's
launcher; these source-file checks are not a newly invented whole-image digest.

Only each native FeedForward's inner dimension changes for the folded block.
All attention, Q/K norms, separate stream modulation, and global time tensors
must remain byte-identical to the selected source. First MLP rows and their biases
must match the retained source channel indices verbatim. Refit downprojections
and biases remain BF16. The exact candidate file/tensor hashes, shape/dtype
coverage, sorted unique indices, bias-inclusive count floor, and stored update
veto are independently checked. `load_state_dict(..., strict=True, assign=True)`
preserves mixed original-FP8/refit-BF16 storage; every loaded coefficient is
checked afterward. There is no global cast to FP8 or new quantization.

The native block receives the unchanged captured full `img_in`, `txt_in`,
`temb`, `rotation_used`, and additive text mask. Copies protect the original
capture tensors from mutation. All text and image rows participate in native
joint attention, with the original text-first order. Actual centered image
coordinates, native text coordinates, complete 2x2 latent patch grid, rotary
shape and generated-to-used cast are checked. No row sampling or replacement
coordinates are permitted. Finite additive biases and legitimate negative
infinity masking are preserved, not converted into another mask convention.

The source's output order is **text, image**. Its image residual uses native
`_apply_gate` / `torch.addcmul`, whereas one text residual uses addition and
multiplication; the validator leaves both paths in the original source class.
Reference-image, timestep-zero, control, patch and replacement-block paths are
outside this plain source-capture contract.

## Baseline before any folded execution

The caller must explicitly supply a finite, nonnegative `--max-source-rel-l2`.
There is no numerical default. For **every complete captured forward**, the
original block is checked against both actual captured stream outputs and its
attention outputs, MLP inputs/outputs, and modulation outputs. Native global time
and position embeddings are independently recomputed and checked against the
captured `temb` and generated rotation table. The block itself still receives
the captured values, not those recomputed substitutes.

Every baseline tensor uses the caller's explicit source tolerance. A failed
baseline stops before construction/evaluation of the folded block. Its failure
receipt reports the baseline separately and makes no folded-error claim.

Only after all baseline forwards pass is the folded block evaluated on the same
complete parent block inputs. The report contains per-forward fit/held scores
for both streams over all context rows and valid rows, plus pooled valid-row
scores for each stream and split. Masked text rows remain in the attention
context and the all-row diagnostic; they do not inflate valid-row scores.

There is **no folded full-block acceptance gate** in this validator. Its report
does not reuse an MLP-error threshold as a new behavioral or joint-block gate,
subtract baseline error, choose widths, retry candidates, or use held-out rows
to fit anything. Acceptance requires separate preregistered policy and eventual
behavioral review. Replaying one block on captured parent inputs does not prove
error propagation through a full 60-block child.

## Invocation and resources

```text
python c16_qimg_joint_replay.py
  --captures <capture_receipt.json> [<additional capture_receipt.json> ...]
  --fold-receipt <fold_receipt.json>
  --fold-receipt-sha256 <exact reviewed SHA256>
  --comfy-root /opt/ComfyUI --device cpu
  --max-source-rel-l2 <explicit preregistered baseline tolerance>
  --out <new JSON file in an existing directory>
```

The only written artifact is the new report, opened in exclusive-create mode.
Existing reports, source files, captures, and folded weights are not overwritten
or cleaned up. A numerical baseline failure prints diagnostics without creating
a success report. A disk failure can leave a partial report; it is preserved,
never silently reused, and is not a successful validation.

Source/folded single blocks are evaluated sequentially, not full 20B parents.
Complete capture files are verified and replayed one forward at a time; pooled
output scoring retains the output/target rows. The existing capture count and
row bounds remain enforced. Budget several GiB of host memory for actual-width
data and pooled float64 diagnostics. `--device cuda:0` does not allocate a GPU;
the caller must use the tracked allocator and a pinned image with read-only
source/capture/fold mounts. CPU and GPU kernels can differ: the explicit original
baseline is mandatory on the chosen runtime, never waived by fixture success.

The 2026-09-07 CPU-only check of immutable image `735020f27cb59da7ca4ea62284154ad97fcbc1a765fa8ffc6c0abd9a2d283c0a`
found a dependency limitation **before native block construction**: importing
`comfy_kitchen` unconditionally imports its Triton backend, whose autotuner raises
`0 active drivers` when no GPU is exposed. The installed import path has no
pre-import backend-disable switch; the public disable API is reached only after
the failing import. No environment, package or kernel patch was applied. Thus
CPU contract fixtures pass, but this image's native no-GPU fixture is not claimed
to pass. A reviewed tracked GPU test remains required for actual native mechanics.

## Test scope

Local synthetic fixtures test source-first refusal, every baseline tap,
full-context contracts, mask semantics, mixed-storage/candidate lineage,
per-stream pooled reporting, and no overwrite. Their cross-stream test double
is deliberately not a source-faithfulness claim.

An optional test enabled with `QIMG_COMFY_ROOT=/opt/ComfyUI` exercises the actual
pinned native block with tiny synthetic mixed-FP8/BF16 weights, separate image
and text widths, full context, and an additive negative-infinity mask. That
checks the native API/loading/forward mechanics only. It is not real capture
parity, image quality, sampler, VAE/interface, paired-chart, or full-child proof.
For a separately authorized tracked CUDA test, also set
`QIMG_NATIVE_TEST_DEVICE=cuda:0`. This test neither allocates a GPU nor repairs
the runtime dependency limitation.
