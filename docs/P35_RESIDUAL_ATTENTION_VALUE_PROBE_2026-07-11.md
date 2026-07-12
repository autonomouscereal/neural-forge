# P35 residual-attention V -> O four-control probe

`experiments/run_qwen_residual_attention_value_probe.py` is the executable
boundary test for physical prototype redirect `L19/G4 -> L21/G6` on the
preserved Qwen3-4B attention-p35 / MLP-p36 checkpoint.  The CLI is hard-wired
to that redirect and rejects a runtime unless it has 36 logical layers, eight
KV groups, and 280 live attention prototypes.

It does not save or commit a checkpoint.  It constructs the 279-prototype
redirect in memory, evaluates four controls, and restores the 280-prototype
incumbent exactly:

1. uncompensated redirect;
2. O-only ridge using the redirected child's original `A @ V`;
3. globally fitted V-only;
4. V installed, cached child traces recollected, then O refitted from the new
   real child `A @ V`.

Only forced calibration traces enter either ridge solve.  Held-out prompts are
token-row disjoint and are never passed to a fitter.  Calibration and held-out
sets are both scored on forced and greedy cached lanes under identical parent
references, parent-token drivers, horizons, and tolerances for all controls.
Every post-V trace proves that the exact prefill/incremental `v_proj` input
history multiplied by the materialized V projection reproduces the post-update
V cache.

Each V calibration is bound to the live tentative registry twice.  The builder
requires the child trace's materialized V weight and bias hashes to equal the
supplied registry at that layer; the global fitter rechecks every affected
layer and writes those hashes into its report.  Installation refuses if any
materialized V projection changed after fitting.  Around every parent,
uncompensated, O-only, V-only, and V-then-O trace collection, the runner also
captures the full content-addressed Q/K/V/O registry hash plus a run-local
storage/version hash over every registered runtime tensor, and requires both
snapshots to remain identical before and after collection.

Parent V-cache reconstruction is deliberately retained as a diagnostic
subproblem, not the promotion objective.  The V-only control is always emitted
even when it is harmful.  The real decision evidence is held-out residual,
cache-transition, and action behavior after the installed V is consumed, plus
the separately refitted V -> O residual contribution.  The executable does not
silently retarget V in response to oracle results.

The inherited block-sparse O solve is valid for this exact lineage because the
survivor is executed once in each affected layer (`L2/G1`, `L19/G4`, and
`L21/G6`).  The runner fail-closes if a survivor is assigned to multiple groups
in one layer.  Such a topology needs a separately tested solver that sums all
scaled group contexts for the shared O slice and uses the layer residual target
only once; duplicating that target per group is forbidden.

## Deliberate scope limits

This executable is fail-closed to the prepared Qwen3 codebook path and
identical raw token IDs.  The value-fit module does not implement a tokenizer
chart or cross-tokenizer alignment, and it does not independently hash the
entire Q/K/O/upstream runtime.  The executable supplies the latter transaction
binding around every collection; callers outside this runner must provide an
equivalent full-runtime snapshot contract.  These limits are emitted in the
public fit/report contract and must not be described as architecture- or
tokenizer-universal evidence.

## Exact command

Run from the separate residual-attention checkout after copying the new source,
test, prompt, and documentation files there.  This command is documented only;
it has not been launched by the implementation task.

```bash
MODEL=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
P35=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child
OUT=/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711/p35_L19G4_to_L21G6_v_o_four_control_h8.json

CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd \
PYTHONPATH="$PWD/src" python experiments/run_qwen_residual_attention_value_probe.py \
  --codebook-checkpoint "$P35" \
  --tokenizer "$MODEL" \
  --calibration-prompts-json experiments/p35_residual_attention_v_calibration_prompts_20260711.json \
  --heldout-prompts-json experiments/p35_residual_attention_v_heldout_prompts_20260711.json \
  --output "$OUT" \
  --device cuda:0 \
  --dtype float16 \
  --horizon 8 \
  --ridge 1e-6 \
  --max-prompts-per-split 2 \
  --max-prompt-tokens 256 \
  --max-affected-sites 16 \
  --max-affected-layers 8 \
  --max-capture-bytes 536870912 \
  --max-projection-bytes 134217728 \
  --local-files-only
```

With the RTX 8000 UUID in `CUDA_VISIBLE_DEVICES`, the process-local device is
`cuda:0`.  Use physical GPU UUIDs on this host: numeric CUDA ordinals have not
matched `nvidia-smi` indices reliably.  Keep the process argument consistent
with the visibility mapping.  The current global venv wheel does not support
P40 `sm_61`; the plain command above deliberately targets a supported RTX
device.

## Required report evidence

The output JSON includes:

- source checkpoint manifest hash and post-rollback parameter report;
- parent/proposed registry hashes, prototype counts, and exact physical byte
  savings;
- canonical/physical/permutation-aware affected assignment coordinates;
- disjoint calibration/held-out prompt hashes and fixed driver token IDs;
- V calibration, fit, installed-slice, materialized-layer, and rollback hashes;
- independent O-only and post-V O-fit reports;
- forced and greedy damage/gates for both prompt splits and every control;
- trace hashes proving full-prefix/cache equivalence and `U @ Wv.T` V-cache
  reconstruction;
- per-layer calibration-time materialized V weight/bias hashes and pre/post
  trace-collection registry/runtime snapshot bindings;
- exact tentative-state and incumbent-state rollback flags;
- `checkpoint_saved=false`, `fold_committed=false`, and
  `compaction_certified=false`.

This is a causal boundary experiment, not a compaction result.  A favorable
held-out V -> O result still requires a committed smaller checkpoint,
save/reload identity, broad behavior/choice/generation evaluation, and
parent-free replay before promotion.
