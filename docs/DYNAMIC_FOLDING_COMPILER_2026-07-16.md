# Dynamic Folding Compiler

Date: 2026-07-16

## Objective

Neural Forge must accept one or more loaded local parent models, a common
model-family-agnostic orchestration request, cached parent evidence, and
behavioral evaluation contracts. The common schema still names the requested
family and modalities and dispatches only inside a registered capability
envelope; it is not unseen-architecture support. The system must automatically
choose a target adapter,
materialize the exact blank child, select the fold stack, inspect, align, fold,
evaluate, roll back, publish, freshly reload, and certify the child without
caller-built architecture surgery or intermediate manual decisions.

The child is successful only when it is standalone, smaller in unique learned
state and serialized/runtime inference state, created from a
compiler-authenticated seed-independent zero blank, changed by an effectful
fold with zero observed training operations under the guarded/audited
execution model, and accepted by one lane-specific free-running behavior
manifest before and after fresh reload. Numerical weight matching, local RMSE,
PPL, first-token agreement, or exact wording are diagnostics rather than
promotion gates.

## Automatic run contract

The production caller supplies only loaded local parent model(s), an exact
target family/modality/size objective, a registered portfolio ID when more
than one capability overlaps, immutable cached evidence, and a new output
root. The system then performs this sequence without per-layer human surgery:

1. inspect every parent into the common circuit IR and authenticate live state;
2. select one registered capability, target adapter, fold stack, and bounded
   candidate portfolio;
3. construct and independently audit an exact-size child whose learned
   Parameters are all zero-valued, plus its structural runtime buffers;
4. plan complete circuit ownership and reject unsupported paths before
   mutation;
5. execute evidence-bound fold transactions with exact rollback and no
   optimizer, backward call, or parent forward;
6. run the bound lane-specific whole-output behavior/retention gate on the
   in-memory child;
7. save a standalone child, reload and gate it from staging, move it to the
   retained destination, then independently reload and gate that exact path;
8. authenticate the complete artifact/provenance tree, commit the ledger, and
   retain at most one selected accepted child (first accepted by default;
   highest-scoring only when exhaustive search is requested).

Adding a new family or modality is registration work, not a new orchestration
script: provide a circuit inspector, exact target synthesizer,
compiler-reviewed structural-buffer policy, evidence/cache verifier, fold
operator/stack, standalone local saver/loader, behavior gate, and bounded
portfolio. Unknown families still fail closed; the common compiler does not
pretend it can infer the semantics of an unseen arbitrary Python model.

## Compiler layers

The implementation keeps three graph types separate.

1. `dynamic_circuit_ir.py` describes the executable architecture: circuit
   nodes, residual order, token and modality interfaces, cache/recurrent state,
   learned-tensor ownership, tying, and exact storage.
2. Existing parent trace caches and canonical execution graphs describe
   immutable evidence and parent-relative behavioral targets.
3. `dynamic_fold_planner.py` and `dynamic_fold_executor.py` describe proposed
   transactions, current-state novelty, installation, acceptance, rollback,
   evidence refresh, and provenance.

`hf_canonical_graph.py` remains a trace-target graph. It is not reused as an
architecture graph.

## Implemented compiler spine

- `dynamic_model_inspector.py`
  - Deterministically selects a topology adapter.
  - Converts GPT-2, Llama, Qwen3, Gemma2/3, and LFM2 into the causal-circuit IR.
  - Gives every named learned tensor exactly one semantic owner.
  - Represents KV cache and LFM FIR history as recurrent state edges.
  - Fails closed on unsupported or ambiguous architecture.

- `dynamic_fold_registry.py`
  - Declares selectable, experimental, closed, deferred, and training-only
    methods.
  - Seeds the settled negative lanes from the 2026 experiment catalog.
  - Uses family, lane, configuration, and exact-candidate fingerprints so a
    renamed runner or changed seed is not treated as a new method.
  - Maintains an append-only hash-chained experiment ledger.
  - Production uses a durable single-writer ledger that atomically rewrites and
    fsyncs every append, detects a stale on-disk chain, and records
    compensating supersession entries rather than erasing provisional results.

- `dynamic_fold_planner.py`
  - Aligns whole causal circuits instead of independent layer tensors.
  - Assigns every target parameter path to exactly one transaction.
  - Uses ordered depth spans and explicit capability gaps.
  - Rejects concrete children that do not equal the exact requested budget.
  - Filters prior exact attempts and curated closed lanes before execution.
  - Centrally enforces source/target circuit kinds, coupled roles, standalone
    closure, and exact evidence-channel manifests.

- `dynamic_fold_budget.py`
  - Solves exact multi-choice child topology budgets.
  - Counts shared learned resources once.
  - Enforces dependencies and conflicts.
  - Fails closed instead of approximating when the bounded state ceiling is
    exceeded.

- `dynamic_fold_executor.py`
  - Rebinds every action to the current child state and evidence epoch.
  - Installs through a registered operator, invokes an independent evaluator,
    commits accepted candidates, and restores rejected candidates exactly.
  - Refreshes evidence only after acceptance.
  - Freezes every parent and child Parameter during callbacks, executes under
    `no_grad`, records guarded optimizer/backward attempts, and requires all
    gradients to remain absent.
  - Includes an exact-copy control operator that preserves Parameter objects,
    gradients, tying, and full-state identity.
  - Binds live parent checkpoint fingerprints to plans and evidence.
  - Restores parent and child state after every non-terminal or rejected run.
  - Snapshots and restores module/global hook registries so hidden forward or
    backward hooks cannot survive a rejected or mutating callback.
  - Treats the in-process training-API guard as best-effort; arbitrary operator
    plugins still require implementation audit or isolated execution, while
    parameter freezing prevents pre-bound backward aliases from writing model
    gradients.
  - Requires initializers to be non-mutating and every accepted production run
    to contain an effectful registered action; learned state may change only
    inside a planned transaction.
  - Requires action and final evidence to retain the plan's authenticated root,
    and requires the expected catalog method to account for changed paths.
  - Fingerprints registered operator implementations and statically audits
    their source for optimizer, backward, autograd, or gradient-enabling paths.
    The runtime guard also intercepts `torch.enable_grad` and
    `torch.set_grad_enabled(True)`.
  - Binds the actual class-owned `prepare` implementation, rejects
    instance-level overrides, and separately fingerprints, audits, and
    reauthenticates every returned transaction's `commit` and `rollback`
    implementations immediately before use.
  - Canonically hashes concrete evidence payload contents, binds each payload
    to its root/action/current child state, and rechecks immutability after
    prepare, evaluate, commit, rollback, refresh, and final evaluation.
  - Observes every live parent module during execution and observes the child
    root during the final behavior gate. Production rejects any parent forward
    or a gate that merely touches an irrelevant leaf module.

- `dynamic_fold_snapshot.py`
  - Keeps the original in-memory rollback backend for small models.
  - Adds bounded disk-backed parent/child snapshots for large transactions,
    streaming CPU chunks instead of retaining whole-model CPU clones.
  - Fsyncs canonical manifests and tensor files, verifies hashes, shapes,
    dtypes, and sizes during restore, and removes only verified owned snapshot
    directories after success, rejection, or exceptions.

- `dynamic_fold_compiler.py`
  - Exposes the complete automatic system entrypoint:
    target synthesis -> post-synthesis fold-stack selection -> compile ->
    whole-output gate -> atomic publication -> fresh reload -> behavior
    certificate.
  - Runs publication-bound transactions against a staged hash-chained ledger
    and merges their records only after the standalone artifact passes fresh
    reload. Save/reload infrastructure failures remain retryable; behavioral
    rejection closes the exact lane; later pruning or recorder failure adds
    compensating `SUPERSEDED` records.
  - Requires positive learned-parameter compaction in the production path;
    same-size exact-copy remains available only as an explicit low-level
    control.

- `dynamic_system.py`
  - Holds target and folding-stack adapters in a persistent catalog.
  - Selects a compatible stack after independently synthesizing the target.
  - Removes per-run family dispatch from the public workflow: family support
    is registered once instead of supplied as custom surgery for every model.

- `dynamic_target_synthesis.py`
  - Accepts an exact target family, modality set, parameter/byte constraints,
    tokenizer identity, seed, and adapter preference.
  - Selects a registered target adapter automatically.
  - Re-inspects the materialized blank independently and fails closed on
    family, modality, topology, exact-budget, or byte-budget mismatch.
  - Rejects parent/child model, module, Parameter, buffer, and backing-storage
    aliases, then zeroes every unique child Parameter under compiler-owned
    `no_grad`.
  - Recursively audits non-registration attributes, containers, closures, and
    slots so hidden tensors/modules and parent-storage aliases cannot evade
    accounting. Distinct views sharing one backing storage are rejected.
  - Proves dense materialization, exact zero learned state, absent gradients,
    stable parameter/buffer layouts, and unchanged runtime buffers before
    folding. Adapter blank claims are not trusted.
  - Treats every nonzero runtime buffer as untrusted unless an exact
    compiler-reviewed structural policy authenticates its concrete
    implementation, path, role, shape, value contract, and adapter/runtime
    binding. This prevents quantized/custom learned weights from being
    smuggled into a supposedly blank child as buffers.
  - Derives the topology seal centrally from the request, adapter ID/version,
    independently inspected graph, learned-state layout, and buffer state.
  - Accounts for serialized inference state and persistent buffers separately
    from the learned-parameter budget.

- `dynamic_fold_artifact.py`
  - Saves into an atomic staging directory.
  - Rejects parent/optimizer/adapter/sidecar-like artifact files.
  - Freshly reloads the standalone child, verifies graph/state/budget
    equivalence, re-runs the behavioral gate with frozen Parameters, and emits
    a sealed certificate.
  - Binds folding-time and reload-time optimizer/backward/autograd counters,
    file count, total bytes, and the complete artifact-tree digest. The search
    layer re-hashes that tree before promotion.
  - Artifact schema v2 embeds canonical plan, run, child-graph, and
    parent-graph provenance documents plus the complete target-synthesis
    document and binds their file/tree hashes and semantic fingerprints into
    the certificate.
  - Binds the complete behavior-gate manifest and saver/loader implementation
    plus resource fingerprints.
  - Challenges the loader with an empty sibling directory before the real
    reload, so a loader that simply ignores its path cannot certify, and
    reauthenticates provenance after every callback.
  - After staging validation, renames to the retained destination and performs
    a second disjoint load, complete state/topology/policy audit, root-model
    behavior gate, and provenance check from that exact final path. No
    certificate is written if the retained-path proof fails.
  - Observes parent forwards across target/stack preparation, save, load, and
    both reload evaluations and binds the zero counts into the certificate.
  - Loader and arbitrary callback implementations remain an explicit external
    audit boundary; an in-process path challenge is not filesystem sandboxing.
  - A registered evidence object that exposes only an opaque `verify()` plus
    seal remains a provider-trust boundary. Built-in production caches instead
    expose concrete canonical payloads that the compiler content-hashes.

- `dynamic_behavior_gate.py`
  - Reuses cached parent outputs and performs zero new parent forwards.
  - Accepts coherent paraphrases without requiring token or text identity.
  - Rejects new empty responses, new fragments, EOS-margin collapse, missing
    cases, normalized task-score regressions, new generation-limit saturation,
    and per-output repeated-ngram regressions.
  - Applies task retention per case as well as in aggregate, so a gain on one
    prompt cannot hide loss of a different parent-correct prompt.
  - Seals gate identity/version, thresholds, coverage policy, cases,
    splits/categories, cached parent outputs, baseline identities, and
    generator/scorer code and resources into one manifest shared by the
    in-memory and fresh-reload evaluators.
  - Recomputes callable implementation/resource identities instead of trusting
    a caller-supplied fingerprint, and requires an observed root-model forward
    before any production acceptance.
  - Production coverage rejects zero scored cases, zero parent-correct cases,
    and missing required held-out/category/split-by-category coverage.

- `dynamic_search.py`
  - Runs bounded registered candidate portfolios for one exact requested
    target.
  - Continues only after a candidate-specific rejection and stops on
    infrastructure or fatal failures.
  - Accepts only a concrete synthesis -> compilation -> fresh-reload
    certificate chain bound to the exact candidate, adapter, size, state,
    destination, zero-training counters, and on-disk child tree.
  - Retains at most one selected certified artifact and prunes only
    destinations proven to be owned by the current search. The default stops
    after the first acceptance; exhaustive mode may select the highest score.
  - Requires a compiler-authenticated zero blank and topology seal, nonmutating
    bootstrap, source-rooted evidence, source-bound operator audit, an
    effectful expected-method transaction, and positive learned and serialized
    state compaction where live source accounting is available.
  - Requires zero parent forwards in preparation, execution, and publication;
    root child execution in memory, at staging reload, and at retained-path
    reload; exact target/policy/audit provenance; and complete runtime-state
    compaction including nonpersistent buffers.
  - Adds compensating ledger supersession when a provisional artifact is
    pruned, replaced, or fails after publication.

- `dynamic_production_catalog.py`
  - Provides the generic assembly API for target adapters, stack adapters,
    capability envelopes, and deterministic candidate portfolios, then
    exposes one neutral parents + objective + destination facade after that
    registration-time wiring. The real proof runners currently assemble their
    lane bundles explicitly; this is not yet an out-of-the-box unified
    LFM2+Qwen3+NEO catalog.
  - Preflights source family, target family, modality set, tokenizer, exact
    parameter range, pinned adapter, and candidate count before model surgery.
  - Scores authenticated fresh-reload certificate metrics, never untrusted
    adapter self-reports.
  - Fails closed outside explicitly registered capabilities; this is an
    extension mechanism, not a claim that unknown architectures already work.

- `dynamic_lfm2_targets.py`
  - Constructs the smallest legal LFM target stages first.
  - Keeps attention exact during initial MLP stages.
  - Removes MLP channels in geometric cumulative steps.
  - Removes complete GQA groups one at a time only after safer stages pass.

- `dynamic_lfm2_adapter.py`
  - Selects the first open or an exact requested LFM budget.
  - Materializes the exact blank child locally from the loaded parent class.
  - Binds target selection to the exact parent checkpoint, tokenizer, cache,
    topology, and novelty ledger.
  - Exposes cached LFM evidence and the existing closed-loop layer transaction
    through the generic compiler interfaces.
  - Registers a real LFM2 target-synthesis adapter, so an exact LFM target can
    be requested without constructing its config or blank model manually.

- `dynamic_lfm2_system.py`
  - Registers the real cached-evidence LFM bridge as a persistent stack
    adapter.
  - Authenticates the synthesized payload, exact budget, tokenizer, parent
    graph/state, blank state, topology, cache fingerprint, and live child
    before returning a compiler stack.
  - Closes the exact target lane only after the non-mutating final behavior
    gate, atomic save, independent local reload, exact state/topology check,
    and fresh-reload behavior gate all succeed.

- `dynamic_lfm2_behavior.py`
  - Seals one-time parent responses to the exact parent state, trace-cache key,
    tokenizer, prompt packet, and generation protocol.
  - Uses category-aware numeric, code, factual, reasoning, safety, summary,
    creative, and model-surgery validators instead of exact wording.
  - Rejects per-case semantic loss, new blanks/fragments, new truncation, and
    repetition while allowing coherent paraphrases.

- `dynamic_qwen_adapter.py`
  - Treats the previously authenticated attention-p35 / neuron-p36 cached
    codebook evidence as one globally coupled model-level fold, rather than 36
    independent layer edits. Current artifact-v2 acceptance still depends on
    the concrete run's retained-path certificate.
  - Can fit the existing codebook math once or load an authenticated standalone
    checkpoint as immutable cached fold evidence.
  - Synthesizes a differently seeded blank codebook topology, overwrites every
    learned target parameter transactionally, proves seed erasure, and performs
    zero parent/child evidence forwards during replay.
  - Registers target and stack adapters, exact path/budget/topology/tokenizer
    checks, and canonical standalone checkpoint save/reload callbacks.

- `dynamic_qwen_compensated_adapter.py`
  - Replays an authenticated dense-Qwen-to-ragged-Qwen compensated transaction
    into a differently initialized exact-size blank.
  - Treats the multi-anchor MLP change as one model-level transaction, binds it
    to cached parent/candidate evaluation and a reviewed output packet, and
    requires direct-local tokenizer/config/state identity.

- `dynamic_neo_adapter.py` and `dynamic_neo_fold.py`
  - Inspect and synthesize the real NEO early-fusion image+text architecture,
    including learned patch convolution, dense image projection, spatial
    attention, decoder attention, and SwiGLU ownership.
  - Build a sealed image-conditioned calibration cache once, apply a
    model-level 6144 -> 6143 fold without training, and gate the reloaded child
    on all 256 raw-image probes with exact-output equality diagnostic only.

## LFM correction

The rejected 225,957,632-parameter LFM experiment pre-compressed every layer to
14Q/7KV and 2528 FFN before its first layer refinement. A local transaction
could not restore coherence while all remaining layers were already
provisionally damaged.

The new first stage is:

```text
229,693,184 -> 229,650,176
16Q/8KV remains exact
2560 -> 2559 FFN in every layer
43,008 learned parameters removed
zero gradients
```

If coherent, the automatic ladder continues through cumulative MLP reductions
of 1, 2, 4, 8, 16, and 32 channels before attempting the discrete 16Q/8KV to
14Q/7KV attention step.

The existing LFM residual-attention runner now accepts arbitrary legal target
head and FFN dimensions, uses direct local JSON/safetensor loading, and
performs automatic intermediate gates without new per-layer human decisions.
Curated lanes may consume previously reviewed, hash-bound cached evidence, and
final manual inspection remains supplemental to the binding automatic gate.

## Acceptance policy

Intermediate transactions are automatic. An accepted candidate must:

- contain fewer unique learned parameters and less complete runtime tensor
  state than its declared parent set;
- bind to the current child state and immutable evidence;
- own every tensor it changes;
- use zero optimizer steps and backward calls;
- preserve exact rollback on rejection;
- pass current-child free-running behavior and degeneration gates;
- prove observed candidate-model execution during the binding in-memory and
  fresh-reload behavior callbacks;
- refresh downstream child evidence after acceptance;
- provide real parameter compaction in the production facade; same-size
  exact-copy is retained only as an explicit low-level control;
- survive atomic save, direct-local fresh reload, complete artifact-tree
  reauthentication, and the whole-output behavior gate a second time.

Manual review does not make per-layer or intermediate surgery decisions; the
automatic gate is binding and may stop before any artifact exists. A curated
lane may require an already reviewed, hash-bound evidence packet as preflight,
as the Qwen compensated lane does, and final output inspection remains
supplemental. The lane-specific gates score their declared correctness,
relevance, reasoning/code validity, and degeneration contracts. A correct
paraphrase is acceptable when the lane's scorer permits it.

## Proven LFM mechanical frontier and strict rejection

Eleven MLP-only targets through FFN width 2368 passed the earlier automatic
trajectory, exact-budget, save/reload, zero-gradient, and parent-free replay
checks. The deepest mechanically gated target removes 8,257,536 learned
parameters (3.595029%) without training. These are mechanical/trajectory
results, not production behavior certifications.

The hardened width-2552 run removes 344,064 parameters (0.149793%). Its
trajectory sub-gate passes with teacher-forced top-1 `0.9625`, free-run token
match `0.8125`, and mean residual cosine `0.998978597287314`. The strict
28-case whole-output gate rejects it for a semantic regression on
`manual_chat_00` and a new generation-limit saturation on
`recovery_code_00238`. Manual inspection also finds all eight recovery-code
answers nonfunctional, four empty manual answers, and several contradictory or
truncated long-form responses. The pipeline correctly publishes no artifact.

LFM2 therefore has no production-accepted fold-only child yet. Width 2552 is
the best current strict failure-boundary candidate, while widths 2544 and below
remain deeper research evidence. See `DYNAMIC_LFM2_RESULTS_2026-07-16.md`.

## Implemented scope and honest boundary

The production facade is automatic only inside a registered capability
envelope. IR inspection support alone is not executable folding support.
Registered real lanes currently cover:

| Source | Target | Modalities | Target policy |
|---|---|---|---|
| LFM2 | LFM2 | text | progressive legal FFN/GQA stages, including exact 2552 |
| dense Qwen3 | codebook or compensated ragged Qwen3 | text | authenticated cached codebook/p35 and compensated-v10 lanes |
| NEO1.5 | NEO1.5 | image + text | cached image-conditioned 6144 -> 6143 lane and bounded repair portfolio |

This does not yet mean literal arbitrary family, arbitrary target size, or
arbitrary modality. Unknown architectures, cross-tokenizer transport,
cross-family executable operators, audio/video circuits, and unconstrained
multi-parent synthesis remain unsupported until an inspector, target
allocator, fold recipe, cached evidence contract, and behavior oracle are
registered and proven. The catalog fails closed instead of pretending those
combinations work.

The next scale milestones are:

1. turn each successful curated lane into a parameterized family allocator and
   cache-derived candidate recipe across every legal target budget;
2. add tokenizer-interface bridge operators and prove them on held-out
   free-running behavior;
3. add cross-family hidden/residual basis transport behind the same
   transactional and fresh-reload gates;
4. register native audio and video circuit inspectors and modality-specific
   behavioral oracles;
5. keep training disabled unless a separate future project explicitly changes
   the scientific objective.

## Native-modality proof target

The first non-text proof is deliberately a learned vision-language compaction,
not byte serialization. The local `Paranioar/NEO1_5-2B-SFT` parent has
2,896,618,496 parameters and a proven raw-RGB baseline of 239/256 overall and
59/64 held out across color, count, spatial, and OCR prompts.

Its native path is:

```text
raw RGB
-> learned 3->1024 patch convolution
-> learned 1024->2048 dense image projection
-> image-token replacement in a 40-layer Qwen3 decoder
-> text answer
```

The smallest clean target keeps the vision tower and attention exact while
reducing every decoder SwiGLU from 6144 to 6143. One coupled neuron owns a gate
row, up row, and down column, so the exact reduction is:

```text
3 * 2048 * 40 = 245,760 parameters
2,896,618,496 -> 2,896,372,736
```

The NEO inspector, local target adapter, sealed 256-image parent behavior
cache, image-conditioned calibration trace cache, coupled SwiGLU transaction,
per-split/category behavior gate, direct-local sharded loader, and standalone
fresh-reload publisher are implemented. The 4-anchor/candidate-8 lane has
already retained all 180 calibration and all 59 held-out parent-correct cases
in memory. Candidate selection remains heldout-blind: cached calibration
statistics choose the fold, while all 256 images, including 64 held out,
decide promotion. Hardened standalone certification remains pending until the
active artifact-v2 replay completes.
