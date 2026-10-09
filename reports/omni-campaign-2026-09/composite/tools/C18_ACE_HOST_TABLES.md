# ACE music-code tables for the S3 native host

The isolated writer emits the existing C16 C18 input: one plain F32 safetensors
file containing `embed.weight` and `lm_head.weight`, each `[65536,5120]`, and a
hash-bound JSON sidecar. It does not install a child or fit a chart. Source and
host files remain unchanged. Training eligibility is zero.

The completed ACE LM is 2048-wide; the completed native S3 host is 5120-wide.
Two independent, explicitly directed zero-bias charts are required:

* `host_to_source.weight A [2048,5120]`: `source_final_hidden = host_final_hidden @ A.T`.
* `source_to_host.weight B [5120,2048]`: `host_embedding = source_embedding @ B.T`.

For source embedding rows `E`, host input rows are `E @ B.T`, and host output
rows are `E @ A`. The latter reproduces logits under the declared host-to-source
map because `host_hidden @ A.T @ E.T = host_hidden @ (E @ A).T`.
The source tied table does not tie these two rectangular host projections. A
tied host is supported only when both resulting tables are byte-identical.

Only source IDs `[151669,215669)` map to native IDs `[252416,316416)`. The remaining
1536 native rows `[316416,317952)` are exact positive zero in both tables. No
excluded named source codes or nonexistent code65535 receive synthetic rows.

`--bindings` is strict JSON with exactly ten path/bytes/SHA256 bindings:
`ace_component_manifest`, `host_bundle_receipt`, `host_binding_reader`, `host_config`,
`host_index`, `host_runtime_contract`, `paired_capture_receipt`, `paired_capture_rows`,
`charts`, and `chart_receipt`. The chart receipt schema
is `chimera.ace_host_directed_charts.development.v1`; it contains exact `bindings`
for the other nine files, the two equations above in `directions`,
`bias_policy:"zero_no_bias_tensors"`, `numerical_policy:"diagnostic_only"`,
`pairing:"exact_record_and_position"`, `fit_split:"FIT"`, and
`synthetic_fixture:false`. These descriptors are identity obligations, not
evidence that a capture or fit occurred. A real producer must independently
prove the observations and chart fit; this writer cannot turn assertions into
an observed source execution receipt. The exact adopted host guard
`c17_source_binding.py` SHA256
`b86d86a24e2bfedb229c071fff3ed89cc75d27aeaca6ed61a87db8af46a571d5`
joins the completed S3 bundle receipt to saved config/index/runtime, ledger,
native state and shard inventory. Merely pinning three unrelated files cannot
pass this guard.

The paired capture schema is `chimera.ace_host_paired_capture.development.v1`.
It joins source/host/bundle and capture-row file pins, rejects fixture claims,
requires unique exact FIT record/position pairs and exact source/native codec ID
mapping. Its four F32 arrays are `source_embedding_rows`, `host_embedding_rows`,
`source_final_hidden_rows`, `host_final_hidden_rows`. They are separately typed
boundaries, not a symmetric inverse or an interchange of embeddings and hidden
states. Source embedding rows are compared against the actual protected ACE
table. Capture/chart tensor hashes and independent through-origin fit direction,
row-count and row-file provenance are verified. These structural checks do not
prove that a producer observed real host/source activations; the source-capture
producer remains a distinct missing implementation.

The CLI calls the existing pinned `ace_fold.load_component` reader. Its existing
runtime/code/complete portable proof/tokenizer/source tensor/protected table and
28 folded coefficient guards remain intact. The host descriptor must be the
actual native S3 checkpoint, not stock HF. Geometry, complete ACE inventory and
source/host codec ledger are checked before source model loading. Charts are
exact F32 CPU matrices; malformed geometry, missing/extra/bias tensors, changed
bound files or nonfinite coefficients/outputs fail. Finite fidelity/compression
values never refuse a valid table.

The new table is streamed in <=1024-row chunks (default256), avoiding a full
2.5GiB table allocation. Both output tensor hashes use the existing C16
`TensorView.sha256` domain: safetensors dtype name + compact JSON shape + bytes.
Every serialized output byte is read back, and unused-row zero bytes are checked.
The table path and sidecar use exclusive creation; the sidecar commits last.
A failed write retains its partial output and requires a fresh path. It never
overwrites or removes a prior artifact. A committed sidecar does not by itself
make tables eligible for production installation, behavior acceptance or training.
The CLI also rejects output beneath the donor portable run, the sealed reader,
or any bound metadata/capture/chart source directory before model loading.

Example after real paired captures and charts exist, with the matching existing
ACE runtime/code and appropriate freshly admitted CPU RAM/disk:

```text
python c18_ace_host_tables.py --bindings actual_c18_bindings.json \
  --ace-reader-dir /path/to/pinned/ace_packet --out /fresh/tables.safetensors --threads 2
```

The actual native S3 and complete ACE descriptors can be parsed now. The 24 ACE
synthetic FIT records establish source IDs, but no saved matching S3 host music
targets or genuine ACE-to-host chart producer has been identified. K3 C17 taps
and diffusion C16 affine charts are incompatible substitutes. Therefore full
real-source table publication and child installation are still blocked by the
explicit paired source/host contract and its observed zero-bias charts. No GPU
allocation, model capability evaluation, benchmark content, source mutation,
queue edit or optimizer is part of this implementation milestone.
