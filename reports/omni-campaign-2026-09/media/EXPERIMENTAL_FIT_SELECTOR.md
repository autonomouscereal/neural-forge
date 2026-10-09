# Experimental FIT-only output-aware channel selection

Recipe ID: `fit-centered-native-bf16-parent-output-norm-v1`.

This optional research module is not imported by `c16_qimg_fold.py`, exported by
`mediafold.__init__`, or enabled by a production policy. It loads no models,
captures, benchmarks or checkpoints and writes no artifacts. Its tests use only
small synthetic CPU tensors. No actual-QIMG improvement, accepted fold, retained
intelligence, multimodal capability or performance improvement is established.

The existing QIMG selector pivots the uncentered hidden-feature Gram. Two
synthetic counterexamples show why that objective can choose a high-energy
channel whose source readout weight is zero, or a constant channel that an
existing affine bias can represent. This is an objective limitation, not proof
of a defect causing the measured QIMG error.

The fixed proposal first translates every FIT row relative to the first row.
It computes the mean of these differences in an FP64 chunked first pass and
accumulates their centered Gram in a second pass. Constant columns therefore
produce exact zeros even when summing their large absolute values would round
the mean. This also avoids subtracting two large raw moments for nearly constant
features. Translation or Gram arithmetic overflow refuses selection. It then
pivots `D @ centered_Gram @ D`, where `D[j,j]` is the Euclidean norm of column
`j` of the native BF16 parent readout across every output. The existing
`pivoted_columns_gram` primitive chooses indices. Only those original indices
are returned, sorted into source order. Hidden features and source parameters
are neither normalized nor mixed. Chunk size is an implementation batching
parameter, not an alternative selector recipe.

Inputs must be nonempty, finite, strided CPU matrices with FP16, BF16, FP32 or
FP64 features. FP8 source coefficients are not valid activation inputs. The parent readout
must already have BF16 dtype matching the source runtime; the function does not
silently reinterpret an FP8 or FP32 readout. The caller must independently prove
authentic source replay and FIT-only provenance. A tensor alone cannot prove
those facts. All-zero output weights, constant features and insufficient rank
refuse selection rather than filling the result with arbitrary channels.

After selection, a Cholesky check certifies independent chosen columns. Each
squared Cholesky diagonal must exceed the fixed threshold
`max(1e-24, eps64 * max(FIT_rows, source_channels) * max(weighted_Gram_diagonal))`.
This adds a conservative rank refusal for roundoff residues that can survive
the existing primitive's absolute `1e-24` stopping floor. Very small global
readout scaling can trigger that inherited absolute floor. No universal scale
invariance is claimed. BF16 rounding and chunk accumulation can affect tied or
nearly tied pivots; a future experiment must pin the implementation and chunk
size before data collection. Gram storage is O(source_channels squared) in
FP64; the existing pivot primitive also allocates a Gram copy and a
source_channels-by-keep factor. The rank certificate uses keep-by-keep matrices.
Chunking bounds temporary row matrices, not the total memory needed by the Gram
and pivoting. No measured QIMG throughput or memory reduction is claimed.

For a future experimental fitter, the actual affine design must remain the
unweighted, uncentered original selected features plus the existing bias column.
The prior must remain the corresponding native parent readout columns and
parent bias. Retain the existing minimum of two FIT rows per retained channel,
lambda ladder, parent anchor, update-scale
veto, stored BF16 update check, source replay checks, per-forward and pooled
numerical gates, and bias-inclusive parameter count with the unchanged 4x
requirement. This module grants no admission and changes none of those gates.

Output-weight norms ignore cross-channel output cancellation. Moving a large
constant into the affine bias can also violate the unchanged update veto.
Consequently, the proposal is a heuristic and is not guaranteed to improve
FIT error, numerical admission or capability retention. No held-out-informed
parameter search is part of this recipe.

The previously observed QIMG holdouts are development data. Before any new
acceptance claim, preregister the recipe, hashes, source runtime, captures,
split identities, parameter accounting and unchanged gates, then collect fresh
image-disjoint data without using those new held-out outcomes to select or
tune the recipe. Folding numerical checks remain distinct from later capability
evaluation of a complete assembled child.

Run the isolated CPU tests with the existing Neural Forge environment:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
& .venv/Scripts/python.exe -B -m pytest reports/omni-campaign-2026-09/media/test_experimental_fit_selector.py -q -p no:cacheprovider
```
