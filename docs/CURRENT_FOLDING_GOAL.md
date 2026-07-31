# Neural Forge current folding goal

Updated: 2026-07-30

## Objective

Build one automatic compiler that folds one or more parent models into a
complete blank child of a requested learned-parameter count.  The long-term
compiler must discover and bridge architecture, tokenizer, and modality
differences dynamically.  The immediate proof target is local LFM2 text-to-text
folding at exact 4x learned-parameter compaction, because a coherent child must
be demonstrated before broad adapter coverage is useful.

The child must be standalone and dense.  Folding may use sealed cached parent
evidence and deterministic analytic compilation.  It may not obtain the result
through gradient training, distillation epochs, optimizers, pruning,
quantization, sparsity, parameter sharing, a parent runtime sidecar, or a
hidden donor model.  Those techniques can be studied later, but they do not
prove this folding objective.

Success is judged primarily by complete decoded responses and manual
intelligence review.  The working quality target is roughly 90% retained useful
intelligence, not exact wording or exact token identity.  Numeric measurements
diagnose the compiler; they are not a substitute for reading the child.
Safety is reported separately from folding quality. An evaluator output-cap hit
is censored/incomplete rather than failed; EOS, cap, loop, and runtime-error
observations remain distinct.

## Canonical exact-4x budget

`57,420,144` learned parameters is THE canonical exact-4x target.  It is
`4.0002195745102975x` learned-parameter compaction against the
`229,693,184`-parameter parent, and every new run, plan, option bank, target
synthesis, and receipt must use it.  `57,405,368`, `57,422,440`, and
`56,984,048` are historical or alternative budgets from earlier frontier work;
they stay readable in old artifacts and must not be used for new runs.

## Current evidence

- Exact parameter compaction is proven structurally: complete dense LFM2 blank
  children at `57,420,144` parameters versus a `229,693,184`-parameter parent
  provide `4.0002195745102975x` learned-parameter compaction.
- Counting independent k32 `C_in` and `C_out` routes leaves an exact physical
  dense budget of `53,552,368`.  The old authenticated H816 option bank reaches
  `53,552,304` and is short by 64 parameters, so it cannot be reused as-is.
  Exact structural witnesses exist when layer 13 retains its 1,024-channel
  ShortConv state/FIR algebra and only its output writer is factorized; for
  example writer-rank 31 with layer-5 MLP width 765 reaches the exact dense
  target while retaining all six attention layers at Q16/KV8/D64.  This is a
  feasibility witness, not a hardcoded selection policy.  Layout accounting and
  blank materialization already support factored writers; the production
  adapter/target guards and cached risk option bank still need the path enabled.
- The counted-state planner currently reserves the right arithmetic but final
  production has not yet packaged observed route tensors with the child or
  called the counted-state verifier after fresh reload.  A dense-only artifact
  must not be reported as the complete `57,420,144`-parameter result.
- Useful intelligence at exact 4x is not yet proven.  The strongest reviewed
  exact-4x candidates remain `0/16` coherent on the complete manual task set.
- The A9 same-body k32 residual-OMP route retains 3 useful and 7 partial
  folding-quality observations after counting every bridge coefficient, at
  `1.361447x` effective compaction. Nine outputs reached EOS and seven are
  censored at the evaluator cap. A11's split output route reaches 3 useful and
  9 partial observations at `1.346018x`; its safety observations are tracked
  separately. These prove a major token-interface fold can preserve coherent
  behavior, while reasoning, retrieval, and autoregressive trajectory quality
  remain incomplete before body compression.
- The authenticated A12 four-lane teacher-forced audit now isolates the token
  routes over 365 aligned donor-token states without a parent forward or new
  trace.  The same-run exact/exact control retains `362/365` cached-parent
  winners.  With the exact parent table as readout, folded continuation inputs
  retain `316/365`, folded prompt inputs retain `323/365`, and both folded
  retain `294/365`.  With the independent folded `C_out`, the corresponding
  counts are `332/365`, `299/365`, `308/365`, and `289/365`.  These are causal
  diagnostics rather than an intelligence score: vocabulary coverage,
  semantic-time execution, and tokenizer round-trip are working, while
  context/position-sensitive `C_in` drift and a smaller separable `C_out` loss
  remain.  One shared support/global map is therefore not sufficient.
- Local component fits can be strong while the whole child trajectory fails.
  A0025 measured small local action errors but endpoint errors as high as
  `14.37`, showing that independently fitted components land in incompatible
  evolving residual coordinates.
- The old global residual correction physically changed zero target
  coordinates.  Output-only unrestricted correction improved supported-logit
  error and repetition/EOS behavior but still produced `0/16` coherent
  responses.
- A0011 did not test the global-map hypothesis: it failed before compilation on
  legal FP32 fused-kernel replay drift of `1.60933e-05`.  It is an
  infrastructure-only outcome and will not be rerun.
- The real donor tokenizer has `64,402` contiguous defined IDs inside the
  model's `65,536` physical row capacity.  The remaining `1,134` rows are
  undefined and must never be scored.  The strict target bridge supplies
  `4,096` injective exact anchors.
- Exact raw ByteLevel token-string segmentation maps every one of the `64,402`
  defined donor IDs into a nonempty sequence of target-vocabulary rows with
  zero failures and a maximum code length of 15.  Standalone Unicode decoding
  is lossy for 665 byte-fragment rows, so singleton Unicode transport is not a
  valid compiler contract.
- Reusing the installed hard-anchor table while reconstructing all defined
  donor logits already preserves `293/365` (`80.27%`) cached parent argmax
  decisions.  All 293 direct-anchor parent winners survive and all 72
  multi-token-code winners fail; cached prompt/forced-token occurrences still
  retain `0.806` mean embedding cosine and median `1.0`.  This isolates the
  remaining token-interface gap to multi-code readout/reconstruction, not
  missing tokenizer coverage or uniformly bad prompt embeddings.
- A counted rank-64 global factorization fits the same interface budget and
  improves average embedding cosine, but drops cached parent argmax retention
  to `15.6%`.  Plain unanchored low-rank token-interface maps are closed.

## Closed paths

Do not spend another full run on:

- one global residual map;
- one static target-to-donor vocabulary row lookup;
- singleton Unicode decoding as a donor-token transport contract;
- a plain unanchored global low-rank token-interface factorization;
- output-head-only correction of an already incoherent exact-4x body;
- tiny or marginal compaction as the main target;
- exact-token matching as the intelligence verdict;
- re-generating parent evidence that the sealed cache already contains.  This
  bans repeating a capture whose batch content address or corpus fingerprint is
  already sealed.  It does *not* ban new evidence on states no sealed cache
  covers: bounded sealed on-policy extension rounds that run only the child,
  collect its visited prefixes, and evaluate the parent once per round on those
  prefixes are permitted under
  [docs/ON_POLICY_EVIDENCE_EXTENSION_2026-07-31.md](ON_POLICY_EVIDENCE_EXTENSION_2026-07-31.md);
- generic Llama integration before the LFM2 folding mechanism works.

Component-only experiments remain valid diagnostics, but the claimed result
must still be one complete aggressively compacted child.

## Active mechanism

1. Separate semantic time from external spelling.  Prompts are donor-tokenized
   internally; each donor action reconstructs one semantic embedding and
   advances the body exactly once; generated donor IDs are decoded as one
   complete string and only then retokenized by the target tokenizer.  A donor
   token that spells as several target tokens must never advance attention,
   RoPE, cache, or ShortConv once per spelling token.
2. Compile raw ByteLevel target-row codes once for all defined donor IDs,
   direct anchors first.  Bind them to the child's existing compact table with
   parameterless integer routing.  Authenticate the live table once per
   bounded decode session, not once per token, and exclude undefined physical
   capacity rows.
3. Test the already-installed hard-anchor table first on the same-body control.
   Its 4,096 exact anchors fix every float row of the only 4,096-by-1,024
   table, so there is no continuous solve freedom while all anchors remain
   protected.  If whole responses remain weak, compare checkpointed constrained
   semantic solves that protect 3,072 and then 2,048 behavior-priority anchors
   while using the remaining rows to repair multi-code donor behavior.  Do not
   run another global map or unanchored low-rank interface.
4. Reconcile the body with moving boundary-local encoder/decoder charts,
   attention-aware macrocell transport, a forward analytic sweep, and a reverse
   physical reclosure sweep.  Do not collapse these into one global chart.
5. Reuse the sealed parent cache and pre-causal child capsules.  Persist
   content-addressed phase checkpoints after structural assembly, each accepted
   layer/boundary, body reclosure, final norm, token bridge, publication, and
   manual evaluation.
6. Resolve numeric roles independently: source storage, source/child forward,
   accumulation, factorization/solve, target publication, and real-runtime
   verification.  Promote immutable evidence for analytic work, cast once at
   publication, and never treat ordinary cross-kernel roundoff as a folding
   failure.
7. Emit durable structured progress, cache hit/miss reasons, timings, tensor
   shapes/hashes, dtype/device policy, memory, solver diagnostics, physical
   changes, and explicit failure context throughout every run.

## Next proof sequence

1. Finish the authenticated high-cap replay only for A9/A11 responses censored
   by the old evaluator cap.  Reuse the saved children; do not refold.
2. Convert A12's position/context causal damage into independent residual-OMP
   support searches for `C_in` and `C_out`.  Keep fit prompts disjoint from the
   manual-review prompts and locally reclose only affected token routes.
3. Publish and manually review that counted same-body route.  Its purpose is to
   establish the strongest token bridge before body compression, not to claim
   the final result.
4. Reserve both continuous route tensors inside the fixed total
   `57,420,144`-parameter budget.  With two k32 tensors this leaves
   `53,552,368` parameters for the physical dense target.  Enable authenticated
   factored ShortConv output writers, add their cached output-action risk to the
   option bank, then dynamically select and materialize an exact heterogeneous
   LFM2 topology at that count.
5. Fold the complete body with boundary-local encoder/decoder charts,
   coordinated residual attention, state-aware ShortConv, semantic MLP
   transport, a forward analytic sweep, and reverse physical reclosure.  Reuse
   the existing cached component evidence and phase checkpoints.
6. Freshly reload the complete standalone child and retain uncapped/censored
   metadata plus complete decoded outputs for manual intelligence review.
   Authenticate the physically observed dense state and both packaged routes,
   then require counted-state verification to total exactly `57,420,144`.
7. Iterate causal component allocation at the same aggressive total target;
   do not retreat to marginal compaction.  Only after coherent LFM2 proof,
   broaden architecture/modality discovery, token/feature bridges, and target
   materializers behind the same compiler ABI.
