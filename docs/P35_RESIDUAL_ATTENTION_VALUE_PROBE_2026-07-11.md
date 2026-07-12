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
In strict validation mode, every accepted post-V trace checks whether the exact
prefill/incremental `v_proj` input history multiplied by the materialized V
projection reproduces the post-update V cache within declared tolerances. A
record-only run should retain the same error without making it the functional
success criterion.

Each V calibration is bound to the live tentative registry twice.  The builder
requires the child trace's materialized V weight and bias hashes to equal the
supplied registry at that layer; the global fitter rechecks every affected
layer and writes those hashes into its report.  Installation refuses if any
materialized V projection changed after fitting.  Around every parent,
uncompensated, O-only, V-only, and V-then-O trace collection, the runner also
captures the full content-addressed Q/K/V/O registry hash plus a run-local
storage/version hash over every registered runtime tensor, and requires both
snapshots to remain identical before and after collection.

Parent V-cache reconstruction and the existing attention/residual/cache damage
reports are deliberately retained as diagnostics. They cannot rank controls.
The V-only control is always emitted even when it is harmful. Control selection
uses calibration functional output only: fixed parent-action top-1 agreement,
parent-action likelihood/top-k/rank, then independent greedy rollout retention
and degeneration. Selection fails closed unless every forced calibration
driver is the parent's actual top-1 action. Held-out function is evaluated and
ordered separately, names no winner, and cannot feed a fit or selection. The
executable does not silently retarget V in response to diagnostic results.

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

## Completed live attempts: all failed closed before a report

Three live executions exercised progressively more of the runner, but none
completed its declared numerical identity checks. They produced no JSON
report, saved no checkpoint, and committed no fold:

1. The initial horizon-8 FP16 run stopped during the O-only held-out
   collection. Cached-versus-full-prefix maximum logit drift was `0.0466309`,
   exceeding the declared logit absolute tolerance `0.04`.
2. A horizon-8 FP16 rerun with logit absolute tolerance `0.06` reached V
   installation, then stopped because exact V reconstruction maximum error was
   `0.0360547`, exceeding the declared context absolute tolerance `0.008`.
3. The strict horizon-4 FP32 P40 run reached post-V recollection and the
   `v_then_o` stage, then stopped because cached-versus-full-prefix maximum
   logit drift was `0.0097847`, exceeding the declared logit absolute
   tolerance `0.005`. The top-1 check occurred earlier and did not trigger.

The FP32 failure log is
`/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711/p35_L19G4_to_L21G6_v_o_four_control_h4_fp32_p40.log`.
There is no corresponding result JSON.

These are fail-closed numerical-contract results, not method scores. They do
not establish that raw V, O-only, or V-then-O helped or harmed held-out
behavior, because no run completed and no variant metrics were emitted. The
FP16 failures especially must not be reported as algorithmic falsification;
the strict FP32 attempt also failed an execution-equivalence gate before a
complete report could be certified.

Cached/full-prefix and activation-reconstruction drift should therefore be
available in both strict validation and record-only diagnostic modes. Strict
mode can certify the collector contract; record-only mode should retain the
same measurements while allowing functional held-out comparisons to finish.
Internal numerical closeness is not itself the compression objective. This
policy does not retroactively create variant scores for the three incomplete
runs above.

## Completed record-mode result: tensor fit was a false positive

The FP32 horizon-4 record-mode rerun completed and emitted:

`runs/residual-attention-20260711/p35_L19G4_to_L21G6_v_o_four_control_h4_fp32_record_p40_gpu2.json`

Its report SHA-256 is
`db5308e77e2863133d82c56d9671e02dee8e45838693a9b174257eb8ff13a15f`.
No checkpoint was saved, the tentative fold was rolled back exactly, and all
structural, finite-value, cache-layout, projection/hash, and lineage checks
remained fail-closed. Only finite internal closeness mismatches were recorded
instead of aborting the behavioral experiment.

The result is a direct warning against using internal numerical agreement as
a compaction gate:

- The raw V calibration fit reduced its local RMSE from `0.658301` to
  `0.015534` and its value-cache diagnostic from about `1.99728` to `0.02509`.
- That fit was ill-conditioned (`7.7937e10`) and moved the fitted anchors by
  `257.986x` relative to their starting scale.
- On held-out prompts, raw V raised functional risk from `0.841695` to
  `7.153297`, lost parent top-1 behavior on every forced trace, damaged top-k
  actions on every trace, and made all six scored greedy paths diverge.
- V followed by the global O fit was worse: held-out functional risk
  `8.785512`, forced mismatch `1.0`, top-k damage `1.0`, and all six greedy
  paths diverged.
- O-only also lost held-out top-1 on every forced trace and diverged greedily.

All 48 V-projection-to-cache/hash audits passed, and cached decoding retained
`480/480` top-1 agreement with full-prefix decoding. Three of 120 traces
recorded finite cached/full-prefix numerical drift above the strict tolerance;
all three were V-then-O calibration traces. Those three diagnostic mismatches
would previously have hidden the much more important result: a numerically
excellent local V fit was behaviorally catastrophic out of sample.

Raw cache regression and global shared-O regression are therefore rejected as
repairs for this redirect. Their matrices may still be used as initializers,
but neither tensor fit nor parent-weight resemblance may select or certify a
child. Selection must use fixed parent actions and independent child rollouts,
followed by held-out and broad suite evaluation.

## Exact command

Run from the separate residual-attention checkout after copying the new source,
test, prompt, and documentation files there. This reproduces the initial FP16
numerical contract; it is not a known-successful command, as documented above.

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
  --logit-rtol 0.01 \
  --logit-atol 0.04 \
  --context-rtol 0.008 \
  --context-atol 0.008 \
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
- function-only calibration reports and ranking for every control, with the
  sole named winner sourced from calibration;
- separate held-out function reports and ordering with no held-out winner;
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
