# Measured QIMG dual-stream MLP folding

`composite/tools/c16_qimg_fold.py` consumes complete authentic QIMG capture
receipts. It emits a **measured candidate**, not a native-installed container,
accepted media result, or proof of the all-source child. There is no training,
optimizer, input preparation, model download, cleanup or allocator integration.
GPU execution, if chosen later by root, requires its own tracked allocation.

## Inputs and preregistration

Provide one or more `capture_receipt.json` files from the pinned source adapter.
Each referenced safetensors file must remain next to its receipt. All 24 capture
tensors and all 36 frozen block/time tensors are verified by complete file hash,
complete tensor hash coverage, shape and dtype. All receipts must share the same
frozen source block/time file identity and source geometry. Parent identity remains
the recorded parent header plus these selected tensors; no full parent hash is
invented. Containment, duplicate-forward and image/prompt linkage checks fail closed.

The current all-fit keyboard canary is refused **before tensor payload loading**.
Fit and held-out image hashes must be disjoint across every supplied receipt,
including different names and sigma batches. Image byte identity does not establish
semantic independence of visually equivalent images or independence from teacher
pretraining; dataset review remains separate. Each forward's native joint context
was captured intact; the fold uses token-local MLP taps, not sampled-attention replay.

The caller must register a policy before looking at candidate holdout results,
then pass its exact SHA256. Chronological preregistration is externally reviewed;
the tool verifies pinned bytes, not a self-asserted timestamp. No numeric acceptance
threshold is supplied by the tool. Policy structure, with non-executable placeholders:

```json
{
  "schema": "neural_forge.qimg_fold_policy.v1",
  "policy_id": "<reviewed identity>",
  "preregistered_before_scoring": true,
  "gate_scope": "pooled_and_each_heldout_forward",
  "streams": {
    "img": {
      "keep": "<explicit positive integer <=3071>",
      "max_source_replay_rel_l2": "<preregistered number>",
      "max_heldout_rel_l2": "<preregistered number>",
      "min_heldout_cos": "<preregistered number in [-1,1]>"
    },
    "txt": {
      "keep": "<separately chosen positive integer <=3071>",
      "max_source_replay_rel_l2": "<preregistered number>",
      "max_heldout_rel_l2": "<preregistered number>",
      "min_heldout_cos": "<preregistered number in [-1,1]>"
    }
  }
}
```

Unknown policy criteria are refused rather than silently ignored. Source replay
is gated on every fit and held-out forward. Candidate error is gated both pooled
and on **every** held-out forward, separately for image and text. These numerical
gates are not a substitute for the user's behavioral acceptance.

## Count and row constraints

The unchanged safety floor requires `fit_rows >= 2 * keep` independently for both
streams. Text fit/holdout counts include **valid context rows only**; masked rows
cannot inflate support. Source replay still checks every captured text row.
Keep 3071 needs at least 6142 fit rows in that stream. A smaller separately chosen
text keep is a stronger compression attempt, not evidence equivalent to keep 3071.
Rank-starved pivoting refuses; it never silently lowers the requested keep.

For width 3072 / parent inner 12288, each biased GELU MLP has 75,512,832 parameters.
Keep 3072 has 18,880,512 parameters and fails the fourfold floor because the retained
output bias remains counted. Keep 3071 has 18,874,367 parameters and qualifies by
count. The whole block/time ratio is reported separately and is not called 4x.

## Source replay and closed-form mechanics

1. Load the original selected source tensors in their stored FP8 format, then cast
   each MLP projection/bias to the captured BF16 compute dtype. Replay the exact
   `Linear -> GELU(approximate="tanh") -> Linear` boundaries on each complete
   stream's captured MLP input tensor. Compare with the actual captured output.
   Both streams must pass before either reaches pivoting or fitting. Differences
   between CPU/GPU or Torch kernels remain visible as source replay error.
2. Accumulate a float64 feature Gram from FIT-only BF16 source GELU activations.
   Reuse `mediafold.linalg.pivoted_columns_gram`; first-layer rows **and biases**
   are retained verbatim in FP8, using byte-preserving gathers. No SVD replaces
   the channels and no further FP8 quantization is performed.
3. Augment kept features with a constant-one column. Anchor both the second
   projection and its output bias to the source's BF16-cast coefficients, and
   call the unchanged `law_abiding_solve` lambda ladder / `update_scale <= 1`
   veto. Targets are the **captured** MLP outputs, never recomputed substitutes.
   The refit weight and bias are stored in BF16; their stored update norm is
   checked against the same veto after serialization precision conversion.
4. Score the candidate against real captured outputs on all valid fit/held rows.
   Held-out rows never affect pivoting, the ridge ladder choice, or coefficients.
   Source replay and fold residual are separate fields; no subtraction hides
   baseline error and no held-out result triggers an automatic retry or keep change.

This is **MLP replay only**, not full joint-attention block replay. All original
attention, normalization, modulation and global time tensors are retained exactly.
Source positions, masks and sigma/noise/cast evidence remain in the referenced,
hash-verified capture files. No source coordinates or paired charts are fabricated.

## Output and execution

```text
python c16_qimg_fold.py --captures <capture_receipt.json> [<another_receipt.json> ...]
  --policy <preregistered_policy.json> --policy-sha256 <exact SHA256>
  --out <new directory> --validate-only
```

`--validate-only` verifies identities, complete captures, counts and row support;
it does not run source replay or fitting. Remove it for the reviewed folding unit.
Default `--device cpu` requires no allocation. Optional `--device cuda:0` is only
an execution parameter: this tool does not allocate a GPU or authorize its use.
Torch thread limits should be bound by the caller's runtime environment.

Memory is bounded to 32 receipt batches, 1024 forwards and 131072 retained rows
per stream. The full-width float64 Gram is about 1.13 GiB per stream; existing
pivoting clones it and retains its factor. Captured rows, BF16 features and the
existing affine solve add several GiB at larger row counts. Root must inspect the
row preflight and budget host RAM; these are not 24 GiB GPU residency claims.
The existing solver runs on CPU in float64. Per-forward BF16 MLP work alone uses
the selected execution device. No full 20B parent is loaded by the fold tool.

Only a passing numerical candidate writes:

- `folded_source_block_and_time.safetensors`: original source names, narrowed
  image/text MLPs, all other source tensors unchanged.
- `selection.safetensors`: two index vectors, explicitly excluded from parameters.
- `fold_receipt.json`: exact policy/source/implementation identities, baseline
  replay, fit/held metrics, per-forward groups, solve trail and counted storage.

The mixed original-FP8 / refit-BF16 storage is intentional. Globally loading it as
FP8 would change the fitted result and is not supported. A native QIMG dual-stream
runtime with these actual widths, full joint replay, paired host/source charts,
and the eventual image interface is still required. Generic `adaln6` is not used.

A numerical, source-replay or stored-update refusal writes an exclusive JSON
receipt beside the requested output directory, then prints the same diagnostics
and exits with code 2. For `--out /run-output/fold`, the default refusal file is
`/run-output/fold.rejected.json`; `--rejection-receipt <new file>` can name another
separate destination. The receipt preserves the original status and measured
scores, caller policy/digest, capture/frozen-source identities and implementation
digests, with `candidate_written: false`. It is flushed to disk before stdout is
used, so lost container logs do not erase the refusal. No candidate directory is
created. Other validation/runtime exceptions retain their existing behavior and
are not classified as numerical refusals.

Both destinations must be new and separate, outside capture and implementation
source directories, and must not cover or replace input files. These checks run
before large capture loading/fitting and again before writing. Existing files,
directories and dangling symlinks are refused; no outputs are reused, overwritten
or cleaned up. Success and `--validate-only` create no refusal file or its parents.
If a write/reread fails after a new candidate directory was created, available
files remain and its receipt says `incomplete_output`; no cleanup is attempted.
Outputs are reread and checked against all stored tensor hashes before a
measured-candidate receipt is finalized. This diagnostic correction changes no
selection, refit, runtime arithmetic or preregistered numerical threshold.

## Validation scope

Unit tests use explicitly marked tiny synthetic FP8/BF16 fixtures. They check both
streams, bias counts, stored coefficient preservation, source replay refusal before
solving, row floors, image leakage, caller policy integrity, rejection/no overwrite,
and unchanged fitted bytes when only held-out source rows change. No fixture is
source proof, and implementation/testing alone does not execute a real fold.
