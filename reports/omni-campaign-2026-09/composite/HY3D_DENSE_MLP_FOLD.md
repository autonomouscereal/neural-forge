# HY3D block0: frozen native dense-MLP folding

This is a component converter, not an installed all-modality child, a ShapeVAE
absorption, a mesh decoder, or a mesh-quality result. It consumes actual sealed
`c16_capture_hy3d.py --block 0` evidence and the reviewed native parent. There is
no production fixture mode, gradient descent, teacher optimization, seed search,
held-informed retry, host feature-band allocation, or companion decoder output.

## Preregistered experiment

`tools/hy3d_fold_policy_sep07.json` was committed in `e5d75651` before any real
HY3D capture/scoring. Its canonical LF SHA256 is
`4670285e6ae57082c6c564d8ea52947a2e0230729905df20fca7e479c7a06b99`.
The CLI requires a separately supplied exact raw-file SHA256; a declaration in
JSON alone cannot establish historical preregistration. Preserve the committed
policy and launch record; do not change the policy after seeing held scores.

One complete native context per object: exactly two fit and two held objects.
Object bytes, canonical exact triangle geometry, and original conditioning-image
bytes must all be distinct. Sample order is retained, including interleaved
fit/held rows. Every context contains all 4,097 tokens, including native time
token0, the complete 1,370-row image context, actual sigma, native input latent,
and all captured attention/MLP writes. No rows are shuffled or dropped.

The engineering checks are pooled **and each held object**, on both MLP output
and final full-block output: relative L2 <= 0.30 and mean row cosine >= 0.95.
These reuse QIMG's preregistered engineering bounds; they are not a HY3D mesh
quality claim. Unlike QIMG's source replay tolerance, **every HY3D baseline
boundary must be literally F16 byte-exact**, including signed zeros. There is
no source replay tolerance or post-result acceptance adjustment.

## Source and fit math

1. `c16_replay_hy3d.load_capture` validates the pinned source identity, complete
   capture contract, run/combined/per-sample receipt joins, file hashes, original
   mesh/image/split/seed/geometry identities, actual sigma, and exact source
   surface-to-anchor ownership. Per-sample self/cross writes are joined rather
   than invented from combined MLP-only evidence.
2. Load only the original native block0, time embedder and input embedder from
   the exact F16 safetensors checkpoint. Replay the unmodified native classes
   on every fit and held object. This preserves native self-attention then
   DINO cross-attention, three norms, QKV/KV packing, SDPA policy, time/input
   embedding and residual grouping. MoE, later dense blocks, skipped context,
   guessed state and external autocast are refused by this converter.
3. After **all** baseline boundaries pass, obtain fit-only native features with
   `GELU(fc1(x), approximate="none")`. Keep B=1/full-sequence GEMM grouping and
   native F16 bias/casts. QIMG's BF16/tanh-approximate GELU evaluator is not used.
   Only F64 Gram accumulation is chunked (512 rows).
4. Reuse unchanged `mediafold.linalg.pivoted_columns_gram` for verbatim channel
   selection. Retain selected fc1 weight/bias bytes. Rank exhaustion rejects;
   it does not silently pick fewer channels or backfill with random channels.
5. Use all ordered FIT rows in the unchanged `law_abiding_solve`, with an affine
   ones column and prior consisting of selected native fc2 columns plus the
   native output bias. The fixed prior-anchored lambda ladder chooses on FIT
   error only. THE LAW's update-norm <= prior-norm veto and fallback to the prior
   remain unchanged; no K3 alternative fitting rule is used.
6. Cast fitted fc2 and its bias to F16, then independently recompute the stored
   affine update norm and reject a ratio > 1 or any nonfinite value. F64 fit
   scores are reported separately from full native F16 runtime replay scores.
7. Substitute only the two linears inside the original MLP class. Replay every
   full context; all non-MLP writes/time/input must still be byte-exact. Apply the
   frozen pooled and per-held-object gates once. Restore original linears even
   if replay fails. No attention, skip, time, embedding or source tensor changes.

Both linear biases count. Native width 2,048, inner 8,192:

| MLP | Parameters | F16 tensor bytes |
| --- | ---: | ---: |
| Original biased MLP | 33,564,672 | 67,129,344 |
| Keep 2,047 | 8,388,607 | 16,777,214 |

Keeping 2,048 fails the bias-inclusive fourfold floor. The 4x claim is for the
MLP component only, **not** the whole block, model, runtime memory, or dense child.
Time/input/attention source tensors are not omitted from the replay lineage.

## Executable entry point (launch remains coordinator-owned)

Use a previously validated runtime with the pinned native source and genuine
dependencies. Mount source, checkpoint and captures read-only. GPU replay must
use the same native F16/SDPA behavior as capture; a kernel/dtype mismatch is a
baseline rejection, not permission to relax the check.

```bash
python composite/tools/c16_fold_hy3d.py \
  --capture-run /captures/hy3d-block0 \
  --parent-checkpoint /models/hunyuan_3d_v2.1.safetensors \
  --source-root /native/ComfyUI-Hunyuan3d-2-1 \
  --policy composite/tools/hy3d_fold_policy_sep07.json \
  --policy-sha256 4670285e6ae57082c6c564d8ea52947a2e0230729905df20fca7e479c7a06b99 \
  --output /results/hy3d-block0-fold-once \
  --device cuda:0 --cpu-threads 4
```

Default device is CPU; specifying CUDA does not launch or allocate a container.
The solver and selection remain F64 CPU operations. Root must budget host RAM
and threads as well as native replay GPU memory: the 8,192² Gram alone is
512 MiB, and full captures, fit features, design/targets, source tensors and
solver temporaries also remain resident. This tool does not load the complete
7.36 GB parent, conditioning model or ShapeVAE, and does not hash the full parent.

Success writes only three small/necessary artifacts:

- `folded_mlp.safetensors`: precisely `fc1.weight`, `fc1.bias`, `fc2.weight`,
  `fc2.bias`, all F16 and bias-inclusive counted.
- `selection.safetensors`: retained original channel indices, for auditing;
  these indices are not additional inference parameters.
- `fold_receipt.json`: `c16_replay_hy3d.FOLD_SCHEMA` compatible receipt with
  numeric policy verdict, exact capture/policy/source/selected-tensor hashes,
  source code/tool hashes, runtime precision, ordered object lineage, source
  replay and separate F64-fit/F16-runtime scores. `folded_child_proven=false`.

Failure preserves `rejection.json` (numerical/rank/source replay/THE LAW failure)
with no candidate artifact. Unexpected runtime/output failures preserve
`incomplete.json` and any partial files without a final accepted receipt. Existing
directories are never overwritten, resumed, cleaned up or deleted by this tool.
Invalid source/policy/capture inputs fail before creating an output directory.

Before publication, recheck all tracked capture/policy/tool hashes, native
source identity/dependencies, parent file stamp, and actual selected native
block/time/input tensor hashes. Read back written candidate bytes before the
final receipt. The existing standalone replay tool can then independently load
and evaluate the emitted candidate; installation/charts remain separate work.

## Validation scope

CPU tests use explicitly labeled synthetic contexts and small frozen weights;
they test split isolation, exact source-baseline veto, full-context preservation,
native GELU, bias accounting, unchanged law/fallback, stored-F16 update veto,
artifact compatibility and durable rejection. The optional
`HY3D_NATIVE_TEST_SOURCE` test runs the actual pinned native block class on tiny
synthetic weights. Neither proves actual parent fidelity or any 3D/child ability.

The real four-object parent capture and its once-only fold remain outstanding
until the coordinator supplies the actual sealed capture and authorizes launch.
