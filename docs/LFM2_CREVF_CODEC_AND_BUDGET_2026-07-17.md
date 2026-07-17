# CREVF binary codec and exact-budget foundation

Date: 2026-07-17

Status: implemented serialization/planning foundation; no model-behavior claim

## Purpose

The Causal Residual-Entropy Value Fold (CREVF) needs to make allocation
decisions against bytes that will actually exist. Idealized
`number_of_values * bit_width` estimates omit scales, mixed-width indexes,
exact anchors, entropy tables, headers, hashes, tokenizer/config files, and
persistent runtime state and temporary expansion during local entropy decoding.

This first production slice supplies the deterministic binary and accounting
contract. It does not estimate causal importance, calculate a Hessian,
construct residual correctors, run a parent model, train, or assert
intelligence retention.

Implemented files:

- `src/neural_forge/hf_lfm2_causal_residual_entropy_value_fold.py`
- `src/neural_forge/hf_lfm2_crevf_budget.py`
- `tests/test_hf_lfm2_causal_residual_entropy_value_fold.py`
- `tests/test_hf_lfm2_crevf_budget.py`

## Tensor representation

A quantized tensor record contains:

1. a compact fixed binary header;
2. uint64 shape dimensions;
3. canonical NFC UTF-8 tensor name and role;
4. three-bit group-width codes, where `0..4` means `4..8` bits;
5. one finite positive little-endian FP16 scale per group;
6. a per-tensor canonical fixed-DEFLATE envelope around the packed code bytes;
7. an optional sparse exact-BF16 outlier segment;
8. an optional run-encoded exact-BF16 anchor segment; and
9. a SHA-256 digest of every preceding tensor byte.

Group size is an integer multiple of eight. This makes every group code stream
byte-aligned at all supported widths. The final group is padded, and padded
positions must contain the symmetric zero code. A decoder rejects nonzero bit
padding and any noncanonical final-group padding.

### Exact BF16 data

Exact values never pass through JSON or base64.

- Sparse outliers are encoded as a count followed by minimal ULEB128 index
  deltas and raw little-endian BF16 words.
- Anchors are encoded as sorted, disjoint, nonadjacent runs. Each run contains
  a minimal ULEB128 start delta, a minimal ULEB128 length, and contiguous raw
  BF16 words.
- Exact-only records use one full-tensor anchor run. This is intended for small
  norms and control tensors that must not be quantized.

Outliers and anchors cannot overlap. All exact indices must be in range.
Before a group scale or base code is calculated, exact outlier and anchor
positions are replaced by zero. This prevents one protected extreme value from
destroying the resolution of every ordinary value in its group. Protected
positions must also carry the width-specific symmetric zero base code; both
encoder and decoder reject any other representation.

### Measured candidate construction

`iter_crevf_quantized_candidates()` now emits fully encoded policy candidates,
so selection pays actual outlier delta indexes/BF16 words, scales, width
selectors, padded codes, entropy payloads, headers, and hashes. Its axes are:

- row-aligned group size;
- integer width;
- stable per-group top-k absolute-magnitude outlier peeling;
- clipping ratio;
- rounding threshold offset; and
- an optional nonnegative per-weight importance diagonal from sealed cached
  evidence.

Clipping/rounding is selected by weighted reconstruction MSE inside each group.
This is a zero-backprop candidate generator, not a behavioral scorer. The
planner still requires separately sealed causal loss units.

For the audited LFM2 source, hidden size is 1024 and FFN size is 2560.
Therefore 32, 64, 128, and 256 all preserve row boundaries; 256 is not a
cross-row error for this parent. The equal-coverage local diagnostic at
`reports/lfm2_quantizer_geometry_20260717/report-v2.json` covered 124 tensors
and 229,668,608 of 229,693,184 values (99.9893%) at every one of those group
sizes. At group size 256, best-clip relative RMSE for exact top-k 0/2/4 was:

- 8 bit: 0.004229 / 0.003502 / 0.003286;
- 7 bit: 0.008452 / 0.007084 / 0.006604;
- 6 bit: 0.017156 / 0.014186 / 0.013431;
- 5 bit: 0.033883 / 0.029113 / 0.027161; and
- 4 bit: 0.066239 / 0.056525 / 0.053228.

These values prove that clipping and peeling improve the weight-error
diagnostic and suggest an 8/7/6/5-bit search ladder around 50/55/60/65. They do
not prove coherence or intelligence retention. Group size, width, and top-k
must be chosen by cached causal error per actual encoded byte, not RMSE alone.

The schema/planner boundary deliberately leaves room for later binary option
families: OSAQ-style absorbed additive outlier suppression, AAAC-style tiny
per-layer scalar codebooks, TurboBoA-style closed-form attention compensation,
CoreQ-style successive-rounding/beam candidates, and byte-beneficial SERQ
low-rank saliency residuals. Those record types are not implemented in this
slice. When added, their codebooks/factors/correctors must be authenticated
binary segments, included in persistent/transient accounting, and evaluated
without training from the existing cached evidence.

## Canonical entropy coding

Each quantized tensor has its own raw-DEFLATE stream. The one permitted encoder
configuration is:

- level 9;
- raw stream (`wbits=-15`);
- memory level 9; and
- fixed Huffman strategy (`Z_FIXED`).

The compact envelope declares decoded and encoded byte counts. Tensor geometry
determines the expected decoded count before decompression begins. The decoder
uses that count as a hard output bound, rejects unconsumed/trailing data, and
re-encodes decoded bytes with the fixed configuration. Valid but noncanonical
alternatives are rejected.

Fixed Huffman codes do not standardize the compressor's LZ77 match choices.
The envelope therefore seals the four-component zlib runtime profile and
rejects a different runtime before canonical re-encoding. Production packaging
must pin that exact zlib profile; a future codec profile may vendor a fully
specified encoder instead.

Fixed-DEFLATE keeps entropy work inside the standard C runtime instead of a
Python bit loop. The group quantizer, variable-width bit packer/parser, and
runtime dequantizer are width-batched and chunked. On the local CPU fixture,
one million four-bit values completed build/encode/decode/dequantize in roughly
0.01/0.06/0.06/0.02 seconds. This is an implementation benchmark, not a
model-quality result.

## Authentication layers

Corruption is rejected at three independent levels:

1. tensor SHA-256;
2. artifact-record SHA-256; and
3. whole-artifact SHA-256.

The decoder also checks all fixed versions and flags, exact segment lengths,
record ordering, unique tensor names, canonical UTF-8, minimal ULEB128,
canonical fixed-DEFLATE bytes, decoded allocation bounds, code ranges, overlay
zero codes, scale validity, shape arithmetic, and the absence of unclaimed
trailing bytes.

## Packed runtime

The runtime keeps:

- packed three-bit group widths;
- entropy-coded group-code segments;
- FP16 scales;
- binary sparse-outlier bytes; and
- binary exact-anchor bytes.

One tensor's bit-packed code stream is decoded only when that tensor is
dequantized. The decoded code bytes are local transient state and are discarded
after reconstruction; all tensors' decoded code streams are never required to
co-reside. Dense BF16 values are likewise materialized only by `dequantize()`.
Runtime buffers are registered as nonpersistent buffers. `CrevfPackedTensor`
and `CrevfPackedRuntime` own zero `Parameter` objects, and their state
dictionaries do not create a hidden checkpoint copy.

Four counts intentionally remain independent:

- physical checkpoint bytes count entropy-coded bytes, every binary
  header/envelope/hash, and caller-declared shared checkpoint files;
- persistent encoded bytes count the encoded tensor buffers that stay resident;
- decoded full-state bytes report the counterfactual size if every entropy
  stream were decoded together, but are a diagnostic and are not a residency
  requirement; and
- decoded-code transient bytes are reported per tensor and conservatively
  summed by layer-like path, exposing the maximum layer-local decode demand.

The decoder checks tensor, total decoded-full-state, and layer-transient
ceilings from authenticated geometry before DEFLATE inflation. A default
128-MiB per-tensor decoded-code allocation policy also fails closed before
inflation. DEFLATE flush output is bounded.

## Exact 50%, 55%, 60%, and 65% accounting

`CrevfReductionTarget` supports only:

- 500,000 ppm reduction;
- 550,000 ppm reduction;
- 600,000 ppm reduction; and
- 650,000 ppm reduction.

This deliberately rejects small-compression operating points. Exact 50% is the
starting line for repaired all-eight-bit candidates; 55%, 60%, and 65% are the
successively harder targets.

For a caller-sealed dense checkpoint reference `D`, the ceiling is:

```text
floor(D * (1,000,000 - reduction_ppm) / 1,000,000)
```

`account_crevf_artifact()` authenticates the artifact before reporting:

- artifact physical bytes;
- caller-declared shared checkpoint bytes;
- total checkpoint physical bytes;
- persistent encoded bytes;
- decoded full-state diagnostic bytes;
- maximum tensor and layer decoded-code transient bytes; and
- exact headroom.

It fails closed by default if the physical checkpoint exceeds the ceiling.

The caller must pass the exact byte count for config, tokenizer, manifest, and
any other checkpoint files stored outside the CREVF artifact. A later exporter
must converge the final manifest size in the same way the existing packed
value-control exporter does.

All budgets can be sealed independently:

- `must_beat_physical_bytes` converts a comparison artifact into a strict
  `candidate_bytes < comparison_bytes` gate; and
- `max_persistent_encoded_bytes` caps resident encoded buffers plus declared
  shared persistent state;
- `max_decoded_full_state_bytes` optionally caps the full-decoded diagnostic
  without making it a residency requirement; and
- `max_layer_transient_bytes` caps the conservative sum of decoded code bytes
  for any one layer-like tensor group.

The live uniform-eight-bit reference currently occupies 214,133,416 physical
bytes from a 464,128,028-byte source. Its first implementation reported
231,490,240 persistent bytes; the corrected rotary-exact build reports
231,490,112. Both retain every decoded code stream. CREVF does not require
that simultaneous decoded residency. Its physical reduction is 53.8633%.
With the same source reference, the CREVF physical ceilings are:

- 50%: 232,064,014 bytes;
- 55%: 208,857,612 bytes;
- 60%: 185,651,211 bytes; and
- 65%: 162,444,809 bytes.

The exact-50 ceiling leaves 17,930,598 physical bytes above the all-eight-bit
artifact. That is the explicit repair budget for causally selected exact
outliers, writer-row anchors, and later residual correctors. Exact 55/60/65
instead require removing another 5,275,804 / 28,482,205 / 51,688,607 bytes
relative to all-eight-bit before paying for repair.

At exact 50%, production plans should seal 232,064,014 as the physical and
persistent upper bound, report the full-decoded diagnostic separately, and
seal a measured layer-transient limit. At 55/60/65 they should additionally
pass `must_beat_physical_bytes=214133416`. Entropy may reduce physical and
persistent bytes but may not conceal decoded transient demand.

## Multiple-choice byte planner

The planner receives tensor-site options as complete authenticated tensor
blobs. It derives option size from:

```text
record envelope bytes + actual tensor blob bytes
```

Callers cannot substitute an idealized byte estimate. The artifact header and
whole-artifact hash are fixed bytes counted exactly once. Shared checkpoint
bytes are counted before any site option is selected.

Every site contributes exactly one option. Option loss is a nonnegative
integer supplied by a later sealed causal/Hessian scorer. The exact objective
order is:

1. minimum total loss;
2. minimum physical checkpoint bytes;
3. minimum persistent encoded runtime bytes;
4. minimum full-decoded diagnostic bytes; and
5. lexicographically smallest option-ID sequence.

The solver maintains a deterministic Pareto frontier over actual encoded
bytes. With a persistent ceiling it maintains the exact joint physical /
persistent / loss frontier. With decoded-full or layer-transient ceilings it
adds those resources to an exact layer-grouped frontier, resets the transient
dimension only after a complete layer-like group, and then continues the
global frontier. It can therefore choose a different feasible repair mix
instead of selecting an invalid optimum and rejecting afterward. If any
frontier exceeds its explicit safety limit, it fails closed instead of
silently truncating the search or returning an approximate plan.

Frontier states use shared parent pointers instead of copying the complete
option history into every candidate. A separate provisional-state ceiling is
enforced during candidate generation, before Pareto pruning, so an option
cross-product cannot exhaust memory before the normal frontier limit is
checked. A conservative cumulative accepted-node upper bound separately limits
the ancestor chains retained by surviving parent pointers.

### Hard role floors

Loss scores and caller-provided role labels cannot bypass these constraints:

- query and key projection sites require every quantized group to be at least
  seven bits;
- norm sites require exact-only BF16 records; and
- control-anchor sites require exact-only BF16 records.

An exact-only record also satisfies a query/key floor.

The planner derives a second floor from the canonical tensor identity. It
recognizes direct Q/K names and fused `qkv`, `wqkv`, `c_attn`, and `in_proj`
forms. A declared role that understates the tensor-identity floor is rejected
before planning. Path inference treats tokenized `norm`, `layernorm`,
`rmsnorm`, `ln_1`, and `ln_f` forms as normalization, without falsely
classifying modality names such as `surface_normal_head`.

For cross-family or cross-modality adapters, `CrevfSite.semantic_kind` is the
primary sealed catalog contract. It supports typed query, key, fused-QK,
normalization, control-anchor, and other semantics. A role or filename may
raise this floor but cannot lower it.

If no site option satisfies its floor, or the minimum floor-compliant records
exceed the target, planning stops with a budget error.

## Integration path

The next CREVF components should build on this contract rather than create a
second serializer:

1. generate candidate site options at exact 50% first, then 55/60/65;
2. enumerate measured row-aligned group sizes 32, 64, 128, and 256;
3. enumerate per-group top-k exact outlier peeling and clipping/rounding
   policies; encode every candidate so real delta-index, BF16 value, scale,
   padding, entropy, header, and hash bytes are charged;
4. attach sealed cached-causal and Hessian-derived integer loss units;
5. add low-rank or sparse residual-corrector records as additional binary site
   options;
6. assemble and fresh-reload the selected artifact;
7. verify all four byte budgets; and
8. run cached parent-relative whole-response behavior gates without training.

Passing this codec/planner suite proves serialization, authentication,
role-floor enforcement, and byte arithmetic only. Coherence and intelligence
retention remain unproven until an assembled candidate passes the established
behavior evaluation.

## First assembled candidate contract

The first assembled validation child is `CREVF-HJ8` at exact 50% physical
reduction:

- preserve full LFM2 depth and logical topology as an explicit value-fold
  validation control;
- solve eight-bit main codes with cached Hessian-weighted successive rounding;
- round Q/K and V/O as coupled attention alternatives and score the exact
  nonlinear attention output;
- refresh candidate inputs with child-only quantized-prefix execution while
  reusing the sealed parent traces;
- keep norms, convolution kernels, tied vocabulary control rows, and other
  typed control state exact initially; and
- spend the remaining `17,930,598` physical bytes on the exact residual
  weights or compact writer corrections with the greatest cached causal loss
  reduction per actual encoded byte.

Promotion requires at least `10/11` parent-correct tasks, `8/8`
parent-demonstrated capability groups, at least `0.90` semantic
intelligence/coherence retention, and no new blank, generation-limit, or
repetition regression. Exact token agreement is diagnostic only.

This full-topology candidate validates the correction machinery; it is not the
final arbitrary-architecture folding proof. The same solver and packed repair
records must then be composed with the canonical structural interaction beam,
and logical topology/unique learned values must be reported separately from
physical checkpoint bytes.

The zero-backpropagation method stack is based on current post-training
quantization work: attention-joint error compensation, closed-form
parent/child mismatch correction, stable low-curvature outlier suppression,
activation-aware codebooks, saliency-selected single-matrix residual writers,
and dynamic grouping. Neuron Forge implements and tests its own
architecture-adapted primitives; external repositories are research
references, not vendored runtime dependencies.
