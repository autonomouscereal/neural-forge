# Accepted K3 native folded-boundary capture

`c17_capture_k3.py` observes **one already accepted MoE layer** per invocation.
It never fits, trains, calls parent experts, rebuilds the upstream prefix, or
changes the accepted run's resume context. It uses the receipt-pinned native
`attention_half` and `ffn_half`, preserving full sequence lengths, zero-based
positions, KDA state resets, MLA, BlockAttnRes, original chunks and compute dtype.
The CPU implementation tests are synthetic mechanics, not model intelligence.

## Preconditions and bounded canary

1. Let the current native window finish. Independently pin the accepted receipt,
   snapshot manifest, selected body, all ancestor/current FFNs and done markers.
   Preserve immutable selected snapshots **before another window can prune them**.
   Read-only mounts are required; this tool does not copy, freeze or delete inputs.
2. Choose a layer whose *starting folded* and *ending teacher + folded* snapshots
   still exist in the frozen manifest. L2 needs `before_L02_folded` and both
   `before_L03` streams. L3 needs `before_L03_folded` and both `before_L04` streams.
   Do not assume either pair survived the keep-two policy. Missing inputs refuse.
3. Use the accepted GPU class, image/software, CUDA device enumeration, dtype,
   thread count, backend flags and chunk sizes. The accepted runtime/context is
   compared exactly, not migrated. The allocator environment is operational;
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can be retained. All GPU work
   must use the tracked allocator; no untracked host execution.
4. Mount the exact native source read-only, adding these two new capture modules
   (`c17_capture_k3.py`, its helper `c17_capture_host.py`) separately from native
   source identity. Mount only `config.json` and `model.safetensors.index.json`
   for the parent metadata directory: **parent model shards are unnecessary**.
   Mount original corpus files, accepted artifacts and snapshots read-only; mount
   one fresh output directory writable. This document does not allocate or launch.
5. In the reviewed tracked runtime, execute:

   ```text
   python /code/composite/tools/c17_capture_k3.py \
     --capture-plan /inputs/c17-k3-plan.json --plan-sha256 <independently-pinned-full-SHA> \
     --out /run-output/c17-k3-L03
   ```

   Use a fresh output path. There is no resume/overwrite flag. A failure may leave
   a partial output, but never an accepted capture receipt. No parent shard hashes
   are manufactured: body/FFNs are tied to their accepted verbatim/full-file hashes.

## Explicit plan (all file paths absolute)

Every ordinary file reference is `{"path":"/mounted/file","sha256":"<64 lower hex>"}`.
The plan itself must be independently SHA-pinned by the command. Example shape:

```json
{
  "schema": "neural-forge.c17-k3-capture-plan.v1",
  "fixture": false,
  "layer": 3,
  "accepted_receipt": {"path": "/inputs/accepted.json", "sha256": "..."},
  "model_metadata_directory": "/parent-metadata",
  "input_ids": {"path": "/inputs/k3_input_ids.safetensors", "sha256": "..."},
  "review_input_ids": [
    {"path": "/inputs/<original-review-file-1>.safetensors", "sha256": "..."},
    {"path": "/inputs/<original-review-file-2>.safetensors", "sha256": "..."}
  ],
  "evidence": {"path": "/inputs/evidence.json", "sha256": "..."},
  "boundary_map": {"path": "/inputs/boundary_map.json", "sha256": "..."},
  "body": {"path": "/body/L03/body.safetensors", "sha256": "..."},
  "accepted_layers": {
    "1": {"ffn": {"path": "/fold/dense/L01/ffn.safetensors", "sha256": "..."}, "done": {"path": "/fold/dense/L01/done.json", "sha256": "..."}},
    "2": {"ffn": {"path": "/fold/dense/L02/ffn.safetensors", "sha256": "..."}, "done": {"path": "/fold/dense/L02/done.json", "sha256": "..."}},
    "3": {"ffn": {"path": "/fold/dense/L03/ffn.safetensors", "sha256": "..."}, "done": {"path": "/fold/dense/L03/done.json", "sha256": "..."}}
  },
  "snapshot_manifest": {"path": "/inputs/snapshots_manifest.json", "sha256": "..."},
  "start_folded": {"path": "/snapshots/before_L03_folded.safetensors", "manifest_key": "<exact original manifest key>"},
  "end_teacher": {"path": "/snapshots/before_L04.safetensors", "manifest_key": "<exact original manifest key>"},
  "end_folded": {"path": "/snapshots/before_L04_folded.safetensors", "manifest_key": "<exact original manifest key>"},
  "max_capture_bytes": 68719476736
}
```

Do not copy placeholder hashes/paths. Review files must be the **complete original
ordered list**, with original basenames (part of the native corpus hash), including
any K3-native review file. The S1 boundary map must bind the exact evidence-ID file.
Original fit/held positions alone are captured; review/padding still participate in
full native replay and prefix hashes. The budget above is an illustrative 64 GiB
output ceiling, not a measured resource requirement or allocation recommendation.
The tool computes a conservative F32 output projection and refuses an insufficient
ceiling; it never silently reduces the pair set. Paired tensors remain on CPU;
plan host RAM for that projected capture plus original/residual snapshots and native
working state. No giant teacher-prefix GPU alias or parent expert weights are loaded.

## Evidence and next step

The final `capture_receipt.json` requires exact folded prefix **and block-stack**
FP32-byte hashes against the accepted ending snapshot. Starting and ending snapshot
files are full-hashed against the frozen manifest with stat stability, source and
complete FFN-lineage checks. Ending teacher snapshot prefix must match the accepted
teacher prefix, explicitly `teacher_reference_verified=true` and
`teacher_forward_replayed=false`. This is not a teacher-forward rerun.

`Lxx/k3_boundaries.safetensors` contains nine actual boundaries and ordered I64
evidence/host/K3 position/group vectors. In particular, `ffn_input` is the actual
normalized input and `ffn_output` is the direct native folded+shared write, **not**
a residual difference. `ffn_read` (pre-normalization BlockAttnRes result) and
`post_attention` (carried accumulator) are deliberately different names.

These artifacts satisfy the K3 observation side of C10's
`neural-forge.c17-native-chart-boundaries.v1` contract. They do **not** constitute
new charts. A subsequent reviewed producer must join actual host observations at
identical ordered positions, then provide evidence for `C_out: host.ffn_input ->
k3.ffn_input` and `C_in: k3.ffn_output -> host.ffn_output`, without relabeling old
input-only charts or silently changing the chart algebra. Attention/residual charts
and all-source held capability remain separate uncompleted gates.

`--synthetic-fixture-root` permits only explicit tiny CPU synthetic models, caps
sequence volume, confines data/output to that root and marks the receipt `fixture`.
C10 production rejects that receipt. It is not a production fallback.
