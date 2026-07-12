# P35 residual-targeted value-basis probe (2026-07-11)

`experiments/run_qwen_residual_attention_value_basis_probe.py` is the separate
executable test of the residual-targeted V hypothesis. It preserves and does
not call or modify `run_qwen_residual_attention_value_probe.py`, whose raw
parent-V-cache control remains available for direct comparison.

The new CLI is hard-wired to the preserved Qwen3-4B p35 frontier and physical
prototype redirect:

```text
removed:  L19/G4, source-flat-index 156
survivor: L21/G6, source-flat-index 174
shared executable survivor sites after redirect:
  L2/G1
  L19/G4
  L21/G6
```

It rejects exact mode unless the incumbent has 36 layers, eight KV groups,
and 280 attention prototypes, the proposed registry has the expected three
survivor consumers, and the survivor provenance is 174.

## Four matched controls

All controls use the same token-row-disjoint calibration and held-out prompt
files, parent-token drivers, horizon, cached forced lane, cached greedy lane,
and numeric tolerances:

1. `uncompensated`: the 279-prototype redirect only;
2. `o_only`: block-sparse survivor O fitted from the original tentative child
   contexts;
3. `basis_v_only`: a bounded survivor-174/removed-156 V combination fitted in
   residual-contribution space;
4. `basis_v_then_o`: basis V installed, all actual child cached traces
   recollected, and survivor O refitted block-sparsely from the new `A @ V`.

Only forced calibration traces enter either fit. Every control also receives a
function-only calibration report and a separate held-out function report.
Calibration function alone ranks controls and names the winner; selection
fails closed unless all forced drivers are actual parent top-1 actions.
Held-out forced and greedy traces are evaluation-only, produce an ordering but
no named winner, and cannot feed fitting or selection. Existing
attention/residual/cache diagnostics remain in the report but are excluded from
the selection key.

## Basis policy

The two raw V directions are captured from the incumbent before tentative
compaction removes source 156. The coefficient policy is intentionally small
and explicit:

```text
                   prior    lower    upper
survivor 174        1.0      0.5      1.5
removed 156         0.0     -0.5      0.5
```

For every prompt/action at every one of the three shared sites, the fitter
passes both V basis directions through the tentative candidate's fixed Q/K
attention, logical-site V scale, current materialized O block, O scale, and
canonical-to-physical group permutation. The target is the incumbent parent's
isolated selected-group residual contribution. The finite active-set ridge
materializes the selected mixture directly into the existing survivor V;
there are no runtime coefficients, adapters, gradients, optimizer state, or
new parameters.

The O-only fit is discarded before the alternating control. After basis V is
installed, the runner recollects the real child attention contexts and fits a
new block-sparse O. It never reuses the stale pre-V O solve.

## Transaction and evidence boundary

Both V basis entries are captured before tentative construction. Every parent
O block is then captured from the still-installed incumbent with a
`ParentOutputBlockBasisEntry` before tentative installation. The calibration
builder requires all V entries and every parent O capture to name the same
full incumbent registry hash; it also verifies source dtype round trips,
canonical/physical site coordinates, source prototype provenance, and
immutable payload hashes. Every trace collection is surrounded by:

- the full content-addressed Q/K/V/O registry snapshot;
- a run-local identity/version snapshot over every registered runtime tensor.

Both state snapshots must remain identical before and after collection. The
current strict validation path also requires post-V traces to match the
installed materialized V hash and checks `U @ Wv.T + bias` against the
post-update V cache within declared tolerances. That numerical reconstruction
check validates instrumentation; it is not the functional compression
objective. A record-only functional run should retain its measured drift and
continue to held-out behavior instead of using internal closeness as a veto.

Nested cleanup order is mandatory:

```text
restore tentative O exactly
  -> roll back basis V exactly
    -> restore incumbent p35 registry exactly
```

The V rollback handle refuses to run while O remains changed, so this order is
also enforced by adversarial tests. The output report records both tentative
reset and incumbent rollback results. It always emits:

```text
checkpoint_saved = false
fold_committed = false
compaction_certified = false
```

## Documented launch command

This command is prepared but has not been deployed or launched by the
implementation task. It reuses the existing disjoint p35 calibration and
held-out prompt JSON files.

```bash
ROOT=/home/cereal/model-surgery-lab/neural-forge-src
MODEL=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
P35=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child
OUT=/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711/p35_L19G4_to_L21G6_residual_value_basis_four_control_h8.json

CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd \
PYTHONPATH="$ROOT/src:$ROOT/experiments" \
python "$ROOT/experiments/run_qwen_residual_attention_value_basis_probe.py" \
  --codebook-checkpoint "$P35" \
  --tokenizer "$MODEL" \
  --calibration-prompts-json "$ROOT/experiments/p35_residual_attention_v_calibration_prompts_20260711.json" \
  --heldout-prompts-json "$ROOT/experiments/p35_residual_attention_v_heldout_prompts_20260711.json" \
  --output "$OUT" \
  --device cuda:0 \
  --dtype float16 \
  --horizon 8 \
  --basis-ridge 1e-4 \
  --o-ridge 1e-6 \
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

The equivalent fail-closed background wrapper is
`experiments/remote_launch_p35_residual_attention_value_basis_probe_20260711.sh`.
It checks paths, refuses to overwrite an existing result/log/PID, resolves a
physical GPU UUID, and refuses a GPU already using at least 4 GiB.

## Interpretation boundary

A favorable held-out `basis_v_then_o` result would demonstrate a real
zero-gradient residual-aware attention improvement for this one removed
prototype. It would not yet certify a compact child. Promotion still requires
installing the accepted transaction into a standalone 279-prototype child,
save/reload identity, broad parent-free behavior/choice/generation evaluation,
and continued compaction toward the requested target size.
