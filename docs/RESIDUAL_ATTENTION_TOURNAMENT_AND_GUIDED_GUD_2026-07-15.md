# Residual-attention tournament result and guided-GUD continuation

Date: 2026-07-15

This note extends, rather than replaces, the evidence catalog and execution
plan in `CURRENT_RESULT_AND_NEXT_EXECUTION_2026-07-11.md`.  The project remains
one continuous Neuron Forge effort: compile a parent into an exact-size,
standalone child and judge success by learned output behavior, not numerical
weight similarity.

## Independently authenticated tournament result

The three-arm residual-attention candidate tournament completed at:

`/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_guided_candidate_tournament_portable_v3_20260713`

The run is a valid candidate-only experiment on the exact
`1,410,567,168`-parameter child.  All three arms started from their exact
preregistered states, returned to compiled state
`39c57212e804280b67e9706fc2935a038c41c0f8491ab950af6b45f4a21a606e`,
and made zero sealed forwards.  The sparse-site arm performed exactly 64
post-optimizer mask verifications; the two unmasked arms performed zero.

Authenticated roots:

- preregistration: `5a603e0c90f5db2d1a891bbf59a8cbb6116d19e98c00b11b62ed7cacd1aa1fed`;
- evaluation contract: `0f99a02f9711ac8633c763caa6b01cbb2bbcfb6e3fc91de44edda8f517f2e587`;
- candidate frontier: `cb0068bb4d1a646d1368600394b9e276ca15ada12606ce803f16a25e9b25b937`;
- winner selection: `b432e6bd3c6ef8434ff8745361e95cc7baae502a51f2d434fc150de38d85e263`;
- run report: `86d47fa3a4e65d9c22efa6e8188bf85eba3c2d312c0dd969c67e82fd5446a902`.

All three retained packs independently loaded with `weights_only=True` and
reproduced their finite 26-factor v3 runtime and portable identities.  No pack
contains a parent, native-child checkpoint, optimizer state, or sealed result.

The winner is the step-40 trajectory at continuation step 12:

| Candidate metric | Same-arm step 0 | Winner step 12 | Change |
|---|---:|---:|---:|
| action agreement | 47/192 | 49/192 | +2 |
| positional token match | 48/192 | 48/192 | 0 |
| LCP tokens | 27/192 | 27/192 | 0 |
| rollouts with LCP at least 4 | 0/24 | 0/24 | 0 |
| exact rollouts | 0/24 | 0/24 | 0 |
| repeat excess | 0.4114583 | 0.3958333 | improved |
| adjacent-repeat excess | 0.5654762 | 0.5535714 | improved |
| parent-action mean rank | 3172.60 | 1378.01 | improved |

The winner genuinely Pareto-dominates its same-arm step zero on the
authenticated output metrics.  It is nevertheless not coherent: it acquired
no longer trajectory, no LCP-at-least-four rollout, and no exact rollout.  The
model-free publication gate is therefore false.  Its decision SHA-256 is
`8175d498bf6bbfe0498cfc1b250fd422143c98f570a68053b5928dd547c9c6b9`.
Sealed data remains unconsumed, and no publication preregistration or child
checkpoint is authorized.

This is small, real residual-attention progress.  It is not a successful
standalone aggressive compaction.  The strongest coherent result remains the
preserved p35 child at `3,995,570,176` parameters; the aggressive
`1,410,567,168` child remains incoherent.

## Evidence defect found and closed for future runs

The completed tournament used frontier schema v1.  It retained the top three
factor packs and final arm receipts, but did not persist the complete metrics
preimage for every non-retained callback.  Those missing callback metrics
cannot be reconstructed honestly after the fact.

Future tournaments use frontier/run/winner schema v2.  Each callback is first
written as pending and then finalized with its metrics hash, rank, retention
disposition, pack-creation decision, eviction identities, and terminal replay
verification.  At most three factor packs remain on disk.  Failure reports bind
the last durable frontier, including a pending callback if the final write
failed.

Publication no longer trusts a literal Pareto flag.  It reconstructs the exact
three-arm by twelve-checkpoint lattice, replays top-three retention, locates the
winner and its same-arm step zero, recomputes Pareto admission from output
metrics, and binds that decision into the publication semantic contract.  A
false admission blocks sealed access.

## Next continuation: attention-aware complete GUD circuit

The QKVO tournament improved first-action ranking and repetition but did not
propagate behavior through a generated trajectory.  The next bounded method
keeps the best retained v3 QKVO state active and adds a separate exact-zero
Gate/Up/Down residual product.  It does not reset compilation or discard the
attention gain.

The v4 research state will contain the existing 26 v3 factors plus six
`guided_gud.{gate,up,down}_{left,right}` tensors: 32 tensors and
`33,660,928` temporary trainable parameters in total.  Guided GUD contributes
`4,792,320`; after a successful bake the standalone child remains exactly
`1,410,567,168` parameters with no adapter or sidecar.

For 30 layers at rank 8, the guided right rows are:

- Gate: `[30, 8, 1536]`;
- Up: `[30, 8, 1536]`;
- Down: `[30, 8, 5120]`.

The right bases are behavior-guided.  For each projection, the audit uses the
real downstream output Jacobian and a role-local input VJP:

`input_sensitivity = D @ W_base + (D @ L_inherited) @ R_inherited`

This preserves the complete Gate/Up/nonlinear-product/Down circuit and includes
all downstream attention, causal context, later layers, and output loss.  It
avoids conflating Gate and Up through a shared-input gradient hook.  Three
preregistered basis arms will compare output-Jacobian, balanced, and
Procrustes-aligned hybrid bases, with an equal-rank random control.

Training stages are fixed before candidate evaluation:

1. steps 1-8: guided GUD left factors only;
2. steps 9-24: guided GUD left and right factors;
3. steps 25-64: joint GUD plus the preserved QKVO factors.

The recovered QKVO state stays frozen through step 24.  Gate and Up share one
coupled layer mask; Down may use its own mask.  All-site and masked arms are
compared rather than assuming masking helps.  A separable deterministic tensor
fingerprint and explicitly bounded sketch buffers keep the audit below the
256 MiB auxiliary-memory contract.

No sealed evaluation occurs unless a post-step-zero candidate independently
passes Pareto admission, obtains at least one rollout with LCP at least four,
and preserves repetition.  Only then may the existing bake, fresh-reload, exact
parameter-count, and broad behavior gates run.
