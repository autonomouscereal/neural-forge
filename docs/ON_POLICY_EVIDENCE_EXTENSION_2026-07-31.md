# Sealed On-Policy Evidence Extension

Updated: 2026-07-31

## Motivation

Every closed-form solve in the LFM2 lane fits parent evidence that was captured
on *teacher-forced* trajectories: a prompt is prefilled, then a fixed
parent-greedy continuation is forced token by token, and the parent's exact
circuit endpoints at those states are sealed into the content-addressed trace
cache.  The folded child is then fitted against that cache.

The child leaves that manifold on its very first free-running token.  Nothing
in the current pipeline constrains parent-child agreement on the states the
child actually visits.  The authenticated A12 four-lane teacher-forced audit
made the consequence explicit: with the exact parent table as readout, folded
continuation inputs retain `316/365` cached-parent winners and folded prompt
inputs retain `323/365`, yet the same children remain `0/16` coherent on the
complete manual task set.  Cached teacher-forced fit does not predict
autoregressive coherence, and no amount of re-solving on the same sealed corpus
can close a gap the corpus does not contain.

The fold-only fix is not more training and not a different loss.  It is more
*evidence*, taken where the evidence is missing: run the child, collect the
prefixes it actually visits, evaluate the parent once on those prefixes, seal
that as additional evidence, and re-solve the closed-form writers on the union
distribution.

## Contract

An on-policy evidence extension round is implemented by
`src/neural_forge/hf_lfm2_on_policy_evidence_round.py` and is bound by the
following contract.  A round that cannot satisfy every clause fails closed; it
never degrades to a warning.

1. **Bounded.** Rounds are numbered from `0` and are capped by an explicit
   `max_rounds` (default `2`).  A round declares its index, its bound, and the
   sealed manifests of every previous round.  `round_index >= max_rounds` is a
   budget error.

2. **Child-only rollouts.** The child is freshly reloaded from its published
   standalone artifact directory and is the only model executed while
   trajectories are collected.  Generation is deterministic greedy argmax
   (`do_sample=False`, `num_beams=1`, fixed seed, deterministic algorithms);
   sampled trajectories are rejected because they are not reproducible from the
   sealed manifest.  The round installs a forward counter on every parent
   submodule for the rollout phase and fails closed if the parent executes even
   once.

3. **Content-addressed child-visited prefixes.** Rollout prefixes are decoded to
   exact UTF-8, content-addressed by SHA-256, and deduplicated
   (`hf_on_policy_probe`).  Rows are then independently re-tokenized by the
   parent tokenizer, so a child whose vocabulary differs from the parent's is
   handled by exact decoded-text identity rather than by assuming a shared token
   space.  A trajectory whose parent re-tokenization shifts the prompt boundary
   is rejected and recorded, never silently reinterpreted.

4. **One batched parent evaluation per round.** Each surviving trajectory
   becomes one `build_lfm2_circuit_trace_cache` batch: the prefill is the parent
   tokenization of the seed prompt, and the forced-decode chain is the parent
   tokenization of the child's own greedy continuation.  Capturing that batch
   evaluates the parent at every child-visited prefix of the trajectory in a
   single pass.  `build_lfm2_circuit_trace_cache` is invoked exactly once per
   round, over exactly those batches, into a new sealed extension cache
   directory.  That is the single sanctioned parent evaluation of the round.

5. **This is evidence extension, not trace regeneration.** Any batch whose
   content address already appears in the sealed base cache or in a previous
   round's extension cache is dropped before the capture.  If nothing new
   survives, the round fails closed rather than re-deriving sealed evidence.
   The extension corpus fingerprint must differ from every prior corpus
   fingerprint.  Regenerating evidence that is already sealed remains a closed
   path.

6. **Comparable capture policy.** The extension cache must be captured under the
   same selection policy, implementation identity, layer topology, and execution
   identity as the sealed base cache, and must carry the same parent identity
   fingerprint.  Otherwise base and extension evidence could not be consumed by
   one closed-form solve.

7. **No gradients, no training.** The whole round runs with gradients disabled.
   No optimizer is constructed, no backward is taken, no parameter is updated.
   The round snapshots per-parameter `requires_grad` flags, gradient identities,
   and gradient values for both models and fails closed if any of them change.
   Parent state, child state, and the published child artifact inventory are
   re-fingerprinted afterwards and must be identical.

8. **Sealed manifest.** Each round publishes one canonical-JSON, self-sealed
   manifest (`neural-forge-lfm2-on-policy-evidence-round-v1`) recording: the
   child artifact inventory and its digest, the parent identity fingerprint,
   round index and bound, the prompt-corpus hash, the manual-review
   disjointness proof, the rollout contract, the extension cache content
   identity and its ordered batch input hashes, and counters proving zero
   gradients, zero optimizer steps, zero backward calls, zero parent forwards
   during the rollout, and exactly one parent capture pass per batch.  The
   loader is fail-closed on noncanonical bytes, a broken self-seal, an
   unexpected schema, a missing or mismatched extension cache, or a nonzero
   forbidden counter.

## Consumption rule

Extension evidence is ordinary sealed parent evidence.  It may be consumed by
**any** closed-form solve in the lane — token bridge, boundary chart, residual
attention, ShortConv, MLP transport, reclosure, output writer — exactly like the
base cache.  There is no separate "on-policy loss"; there is one union
distribution of sealed parent evidence and the same closed-form writers.

`resolve_lfm2_on_policy_evidence_references(base_cache_directory,
round_manifest_paths)` returns the ordered
`(base_cache_path, *extension_cache_paths)` tuple a solver should consume, after
authenticating that every round shares the base cache's parent, that round
indices are contiguous from `0`, that no declared bound is violated, and that no
cache path is double-counted.  `open_lfm2_on_policy_evidence_caches` opens the
same ordered set as verified `Lfm2CircuitTraceCache` handles.

Two rules survive unchanged:

- **Behavioral review prompts stay disjoint.** The round requires a
  manual-review prompt manifest (a sealed digest manifest, a bare digest list,
  or a review-prompt document) and fails closed if any fit prompt hashes into
  it.  Both the exact and whitespace-stripped digests are compared, so a
  cosmetic edit cannot smuggle a reserved behavioral prompt into the fit corpus.
  The disjointness proof — manual manifest SHA-256, digest-set SHA-256,
  comparison mode, overlap count, verified flag — is part of the sealed round
  manifest.
- **Manual intelligence review still decides.** On-policy extension improves the
  evidence a closed-form solve can fit; it does not grade the child.  The
  verdict remains complete decoded responses read by a human against the
  reserved review set.

## What this is not

- It is not training.  No optimizer, no backward, no parameter update, no
  distillation epoch.
- It is not a parent runtime sidecar.  The parent is executed offline, once per
  round, and never ships with the child.
- It is not trace regeneration.  Evidence already present in a sealed cache is
  dropped before capture, and a duplicated corpus fingerprint is an error.
- It is not unbounded.  Rounds are capped, batches are capped, prompts are
  capped, and every bound is recorded in the sealed manifest.

## Status

The round driver, its fail-closed loader, and its reference resolver are
implemented and tested against a real two-layer `Lfm2ForCausalLM`
(`tests/test_hf_lfm2_on_policy_evidence_round.py`).  Wiring the round into the
production catalog runner is a separate wave; the required runner changes are
recorded in `NOTES-1C.md` at the repository root.
