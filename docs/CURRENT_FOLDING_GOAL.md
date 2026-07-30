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

## Current evidence

- Exact parameter compaction is proven structurally: complete dense LFM2 blank
  children at `57,420,144` parameters versus a `229,693,184`-parameter parent
  provide `4.0002195745102975x` learned-parameter compaction.
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
- repeated parent trace generation when the sealed cache already contains the
  required evidence;
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

1. Complete raw ByteLevel span evidence and parameterless semantic-time runtime.
2. Reuse the same-body 4096-row hard-anchor checkpoint and complete the held-out
   256-donor-token manual response review without a new solve.
3. If needed, compare checkpointed 3,072/1,024 and 2,048/2,048 protected/free
   analytic table solves, followed only by a cached target-versus-competitor
   margin repair in the winning table's free-row space.  Retain decoded whole
   responses for manual selection.
4. Short cached exact-4x moving-gauge/body reclosure probes with physical
   target-dtype changes.
5. One checkpointed exact-4x full fold using the semantic bridge plus moving local
   reclosure.
6. Fresh standalone reload and complete manual intelligence review.
7. Only after coherent LFM2 proof: broaden architecture/modality discovery,
   token/feature bridges, and target materializers behind the same compiler ABI.
