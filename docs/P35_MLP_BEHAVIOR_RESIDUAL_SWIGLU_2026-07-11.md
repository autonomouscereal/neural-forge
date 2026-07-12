# P35 MLP behavior-residual SwiGLU v3 continuation

This is the MLP counterpart to the successful assignment-local Q/K/V/O
experiment. It is not a restart, an activation-reconstruction retry, or a claim
that p35 is already repaired.

## The child-sized idea

The MLP-p35 codebook removed one real depth prototype for every one of Qwen's
9,728 canonical SwiGLU neurons. Static medoid assignment then redirected the
missing execution to a surviving real prototype. That saved substantial
storage, but it changed the nonlinear function at many logical layers and the
full p35 MLP result retained only 245/400 parent-passing behavior items and
175/388 parent-passing strict generations.

Instead of forcing a survivor to look numerically like the removed neuron, add
a few new native nonlinear paths in parallel:

```text
folded output + down_r(silu(gate_r(input)) * up_r(input))
```

`gate_r` and `up_r` are deterministic random features. `down_r` begins at exact
zero. Installation therefore produces bit-identical folded-child output while
giving output loss a useful first gradient. Only these three residual tensors
can train. There is no API for parent weights, parent activations, hidden-state
cosine, MLP-output reconstruction, or residual-stream matching.

The tensors are ordinary SwiGLU neuron weights. Concatenating gate/up rows and
down columns proves mathematical equivalence for an already dense target MLP.
That helper deliberately rejects a codebook view: materializing all 9,728 base
neurons would duplicate a complete layer and erase the claimed saving. Compact
publication is future work and must place the residual neurons in rows already
reserved inside the target's fixed intermediate width, then pass exact
configuration, accounting, save, and reload gates.

## Why layer 1 is the first target

The preserved MLP-p35 checkpoint was inspected read-only. Its immutable
assignment and source-provenance hashes are:

- assignments: `e4b2399b55b36039634c34a3ddcbf379380e9f27dfd407847dcd0b9452a87583`
- prototype sources: `3aabaed9bb8f7e71d8c76be7cce91110324102c6608344de75f9834dc269353a`

The 9,728 redirected assignments are distributed across all 36 layers. The
largest counts are:

| Logical layer | Redirected neurons |
| ---: | ---: |
| 1 | 2,122 |
| 3 | 1,264 |
| 2 | 897 |
| 35 | 339 |
| 21 | 284 |

Layer 1 is therefore the smallest single-site experiment that attacks the
largest measured part of the uniform p35 error. It is not assumed safe merely
because later-layer one-neuron folds were safe; the output gate decides.

## V1/V2 evidence and the exact v3 continuation

V1 completed and is preserved at
`runs/mlp-behavior-residual-20260711/p35_mlp_L1_behavior_residual_swiglu_w64_h4_s32_fp16_p40_pair.json`.
Its file SHA-256 is
`ec6402bd793578487a7fe20b167096c1cbc46382921c792e7ea5a324b8c43f8e`
and its internal report SHA-256 is
`b422c1b5cd87d00781011de25bf57e2b654a2236eb57cafa82f83f0f04d063d3`.

That run proved the residual trains useful output behavior: calibration selected
step 8, heldout parent-action top-1 improved from 0.609375 to 0.625, mean action
rank improved from 9.046875 to 3.28125, token matches improved from 8/32 to
9/32, and mean rollout prefix improved from 1.0 to 1.125. It was correctly
rejected because exact heldout rollouts regressed from 2/8 to 1/8. The defect
was the winner rule, not missing factors or a restart-worthy operator failure:
V1 selected only by one-step action top-1/rank and never consulted calibration
free rollouts.

Those eight V1 evaluation prompts are now method-development data because their
outcomes motivated V2. They cannot be reused as independent heldout evidence.
V2 therefore retained the V1 residual operator, width, initialization,
output-only loss, 32 updates, fit/calibration fixtures, tokenizer bridge,
parent/child lineage, rollback, and JSON-only artifact policy, but uses a newly
sealed eight-prompt heldout fixture with new IDs/text and zero overlap with all
V1 fit, calibration, and evaluation rows. Its fixture and manifest hashes are
verified before either model was loaded.

V2 completed at
`runs/mlp-behavior-residual-20260711/p35_mlp_L1_behavior_residual_swiglu_w64_h4_s32_rolloutsel_v2_fp16_p40_pair.json`.
Its file SHA-256 is
`1ef820c5e889f0678e8240adc9fd8e9c7e3cd14a09ed1d4c032625293f176443`
and internal report SHA-256 is
`5784d8e676432ae5ddccc4efe10d0b2c14b2e12609b1f6f8e24a9ad11d7bb32d`.
The Pareto selector correctly rejected every trained checkpoint and locked step
0. Calibration began at 2/6 exact rollouts, 11/24 token matches, mean LCP
1.6667, and action top-1 0.7083. Step 8 improved action top-1 to 0.7292 but
regressed tokens to 9/24 and LCP to 1.5; steps 16 and 32 were also ineligible.
The new V2 evaluation was therefore unchanged from baseline. This is useful
negative evidence: all V2 gradient updates still targeted prefixes generated
once by the blank residual/baseline child, while the trained residual's own
rollout distribution moved.

V3 makes one method-development change: DAgger-style fit-only student-prefix
refresh after updates 8, 16, and 24. Each refresh greedily rolls the current
residual over the next two fit prompts in V2's preserved batch order, asks the
frozen parent for output targets on those new prefixes, and binds the prompt,
driver, target, and batch domains by hash. Calibration never supplies a
training batch. The same manual
Adam moment tensors persist through all four phases with reset count zero.
Because V2's evaluation was inspected, V3 uses another pre-model sealed
eight-prompt evaluation fixture excluding every V1 and V2 ID/text/token row.

- parent: unmodified Qwen3-4B-Base on a separate device;
- child: preserved MLP-p35 / attention-p36 codebook;
- site: logical MLP layer 1 only;
- residual width: 64 native SwiGLU neurons;
- dtype: FP16 (two-byte learned tensors);
- fit/calibration/fresh-heldout fixture counts: bounded disjoint 8/6/8
  functional-output prompts;
- rollout horizon: 4;
- updates: 32, micro-batch 2;
- calibration checks: steps 0, 8, 16, and 32, preserving V1's candidate set;
- gate/up learning rate: 0.001;
- down learning rate: 0.002;
- gradient norm bound: 1;
- fit phases: four eight-update windows preserving V2's exact 32-batch order
  and microbatch composition, with only current-student drivers/parent targets
  refreshed after updates 8/16/24;
- targets: fixed parent top-1 actions and parent output distributions on both
  parent-greedy and current-student-greedy fit prefixes;
- selection: require Pareto non-regression in calibration exact free-rollout
  matches, token matches, and mean longest-common-prefix; among eligible states,
  lexicographically maximize those rollout metrics, then maximize parent-action
  top-1 and minimize action rank; continuous loss cannot promote;
- heldout: first model access only after the calibration winner is locked;
- artifact: one JSON report; no checkpoint or factor pack;
- termination: exact removal of the residual wrapper and exact p35 topology
  restoration, whether the experiment succeeds or fails.

Both full parent and p35 are live concurrently, normally on two GPUs. This is
deliberate: it avoids storing another full checkpoint and permits parent action
targets on child-generated prefixes without pretending that cached aggregate
scores are logits. The child's persisted `parent_token_rows` maps compact token
IDs into the raw parent's padded vocabulary and projects parent logits back into
the child's exact vocabulary. That temporary evaluation bridge is audited and
is not part of a published child.

Exact mode also binds the p35 manifest, model tensor-file hash, state-key hash,
tokenizer contract and live vocabulary, the raw parent revision, and all three
content-addressed raw-parent shards. Topology hashes alone are insufficient.

Fresh heldout remains inaccessible during all checkpoint comparisons. Each free
rollout callback closes over the six calibration prompts and their already
collected parent traces. The callback carries the exact calibration token-row
domain and disjoint fresh-heldout domain hashes, records its split and candidate
state hash on every call, and is rejected if it changes the residual state.
After the winner tensors are reloaded, the complete action evaluation and every
per-ID prompt/parent/child rollout hash must reproduce before the runner records
the lock event and permits the first fresh-heldout model forward.

## V3 compute estimate (not yet launched)

V2 and V3 retain the same five calibration action/free-rollout evaluations and
32 backward updates. V3 adds three fit refreshes. Each refreshes the next two
prompts at horizon 4 through the current child (8 cached calls) plus the same 8
forced parent target calls. Exact added cost is therefore 24 child + 24 parent
= 48 cached forward-call units. This is not a FLOP count or wall-time
multiplier.
The 32-update schedule remains intentionally mixed: 16 static parent-greedy
anchor updates, four student-prefix updates from the initial exact-zero child,
and 12 current-student-prefix updates collected immediately before their
eight-update window. Every refreshed row is consumed; no refresh work is built
and discarded.
Model loads, parameter count, and expected peak parameter memory are unchanged.
The V3 wrapper is prepared but has not been launched. No checkpoint, factor
pack, optimizer state, or model copy is written; marginal disk use remains a
sub-megabyte JSON report. The wrapper fail-closes on the exact runner,
operator, trace, batching, and loss-source SHA-256 values, and the report
records the same content-addressed source lineage at runtime.

Scientific risks are explicit: preserved Adam moments may be stale when the
fit distribution changes; eight prompts may overfit their trajectories;
refresh timing and prompt category are coupled to V2's deterministic order;
the static-anchor/current-student mixture is intentional but remains a category
confound; and forced parent targets may be brittle on off-parent student
trajectories. The fresh heldout and unchanged Pareto calibration lock decide
whether the change is useful rather than assuming it is.

## Exact parameter accounting

Each bias-free Qwen3-4B SwiGLU neuron owns three 2,560-element vectors:

```text
gate row + up row + down column = 3 * 2,560 = 7,680 parameters
```

The p36-to-p35 MLP transition removed 9,728 such depth prototypes:

```text
9,728 * 7,680 = 74,711,040 removed parameters
```

One width-64 residual at one logical layer adds:

```text
64 * 7,680 = 491,520 parameters
```

Net relative to the unfolded p36 codebook MLP is therefore:

```text
74,711,040 - 491,520 = 74,219,520 parameters saved
```

Layer 1's 2,122 redirects represent a 16,296,960-parameter removed-prototype
budget. Even when the complete residual is charged only against that selected
slice, 15,805,440 parameters remain saved. The report keeps this codebook delta
separate from vocabulary-row savings and from raw-parent compression.

## Promotion meaning

Passing the 8/6/8 gate would prove only that this residual operator can recover
heldout output behavior for the dominant p35 MLP site. It would justify:

1. a broad p35 parent-pass regression using the established 1,000 behavior,
   1,000 generation, 250 choice, and PPL suite;
2. residual-width competition under the same disjoint gate;
3. adding the next measured layer only after refreshing evidence;
4. packing the small winning tensors, followed by a native bake test;
5. transferring the operator into progressive 1.4B, 600M, and 134.5M target
   architectures.

Factor-pack persistence and compact native-Qwen publication are not implemented
by this bounded runner. Its factor state exists only in memory for a caller that
provides an artifact sink, and the CLI intentionally emits JSON only.

It would not make the p35 codebook a 25x child. MLP width/depth, attention,
vocabulary, and target architecture still have to be reduced together.

## Files

- reusable operator: `src/neural_forge/hf_behavior_residual_swiglu.py`
- bounded runner: `experiments/run_qwen_p35_mlp_behavior_residual.py`
- sealed fresh heldout: `experiments/p35_mlp_behavior_residual_v3_sealed_heldout_prompts_20260711.json`
- heldout seal/provenance: `experiments/p35_mlp_behavior_residual_v3_sealed_heldout_manifest_20260711.json`
- prepared two-P40 wrapper: `experiments/remote_run_p35_mlp_behavior_residual_width64_p40_pair_20260711.sh`
- operator tests: `tests/test_hf_behavior_residual_swiglu.py`
- runner tests: `tests/test_run_qwen_p35_mlp_behavior_residual.py`
- seal integrity tests: `tests/test_p35_mlp_behavior_residual_v2_sealed_heldout.py`, `tests/test_p35_mlp_behavior_residual_v3_sealed_heldout.py`
