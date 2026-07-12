# Exact-Target Native Progressive Compiler Bridge

## Outcome

Neural Forge now has one explicit, executable contract connecting an
exact-budget blank child to the target-native dense GQA Q/K/V/O residual bake
path. The bridge is implemented in
`src/neural_forge/hf_progressive_target_bridge.py`.

It does not restart the project and it does not replace either incumbent. It
reuses:

- the exact-size, multi-parent blank-child compiler;
- the blank-checkpoint/bootstrap machinery;
- the architecture-neutral dense GQA QKVO residual operator; and
- the native dense bake already proven equivalent on its tested Qwen slice.

The new bridge fails before installation unless the live child already has the
declared exact unique-parameter count. It allows temporary output-trained QKVO
factors, requires an external gate containing both held-out action evidence and
free-running rollouts, and then bakes the factors into ordinary target
`nn.Linear` weights. After baking it requires:

- the exact original child parameter count;
- the original state-key topology and structural fingerprint;
- plain dense Q/K/V/O modules in every target layer;
- zero retained adapter registries or projection wrappers;
- zero published adapter parameters or adapter FLOPs; and
- zero parent runtime references or source-sized basis sidecars.

This is an in-memory publication certificate. The normal checkpoint
save/reload and complete artifact-tree byte gate are still required after it.

## Concrete progressive target plan

The first exact plan is no longer expressed as the ambiguous round numbers
`1.4B`, `600M`, and `134M`.

| Step | Native target | Exact unique parameters | One-step ratio | Tokenizer policy |
|---|---:|---:|---:|---|
| Source | Qwen3-4B-Base | `4,022,468,096` | - | Qwen |
| 1 | Qwen3, h1536/i5120/L30/Q16/KV4/d128 | `1,410,567,168` | `2.851667x` | preserve |
| 2 | Qwen3, h1024/i3072/L24/Q8/KV2/d128 | `600,628,224` | `2.348485x` | preserve |
| 3 | SmolLM2/Llama, h576/i1536/L30/Q9/KV3/d64 | `134,515,008` | `4.465139x` | canonical byte transition |

The final ratio is `29.903489x`. Every stage is a real target-sized dense
model, not a source-width model disguised by a low-rank runtime sidecar.

### Exact stage decomposition

| Component | Qwen3 1.410B | Qwen3 600.628M | SmolLM2 134.515M |
|---|---:|---:|---:|
| Token embedding | `233,373,696` | `155,582,464` | `28,311,552` |
| Unique output head | `233,373,696` | `155,582,464` | `0` (tied) |
| All decoder layers | `943,818,240` | `289,462,272` | `106,202,880` |
| Final norm | `1,536` | `1,024` | `576` |
| Total | `1,410,567,168` | `600,628,224` | `134,515,008` |

The final `134,515,008` count is specifically:

```text
tied embedding          = 49,152 * 576                  =  28,311,552
attention / layer       = 2 * 576 * ((9*64) + (3*64))  =     884,736
gated MLP / layer       = 3 * 576 * 1,536              =   2,654,208
decoder norms / layer   = 2 * 576                      =       1,152
one complete layer                                           3,540,096
30 decoder layers       = 30 * 3,540,096               = 106,202,880
final norm              = 576                          =         576
total                    = 28,311,552 + 106,202,880 + 576
                                                        = 134,515,008
```

SmolLM2 does not store Q/K head-norm vectors. The Qwen stages do, adding
`2 * head_dim = 256` parameters per layer. The planner makes this distinction
explicit instead of accidentally applying the Qwen formula to the Llama
endpoint.

## How QKVO fits without changing a target budget

At rank 8, one temporary physical-group site contains:

| Target | Temporary QKVO factor parameters | Published delta after bake |
|---|---:|---:|
| Qwen3 1.410B | `59,392` | `0` |
| Qwen3 600.628M | `43,008` | `0` |
| SmolLM2 134.515M | `22,528` | `0` |

The temporary count is intentionally reported while the experiment is live.
It is not hidden inside the child compression ratio. Publication computes each
target-local `U @ R`, adds it to the selected dense slice, replaces the
wrappers with fresh native linear modules, and removes the factor registry.
The child then returns to its exact pre-installation parameter count.

## Compiler sequence

1. Materialize the seeded blank target at the exact native shape. Use the
   existing Qwen target builder for the two same-tokenizer targets and the
   existing blank checkpoint path for the exact SmolLM2 shape.
2. Compile simultaneous parent canonical evidence with
   `compile_hf_parents_to_blank_child`. That compiler already proves that the
   target count and topology remain exact and marks capability as unproven.
3. Call `install_parent_compiler_result_qkvo_bridge` for selected target
   physical groups. The bridge rejects a failed bootstrap or a count mismatch.
4. Optimize only the temporary QKVO factors against output actions/logits. The
   dense child remains frozen; internal weight, QKV, activation, attention, or
   residual matching is not a publication objective.
5. Evaluate on disjoint held-out actions and free-running rollouts. A local
   numeric metric cannot create the required publication receipt.
6. If the external behavior gate rejects, call `rollback`; the original native
   target modules and state are restored exactly.
7. If it approves, call `bake_for_publication`. The bridge certifies exact
   native topology and zero adapter sidecar.
8. Save once, reload into a fresh model, run
   `certify_native_dense_gqa_target`, and then apply the existing broad behavior
   and complete artifact-tree gates before promotion.
9. The promoted output becomes the behavioral parent for the next blank target
   stage. Failure at any stage retains the previous promoted frontier.

## Audit of the existing compression primitives

### Directly reusable

- `hf_parent_compiler.py` already accepts simultaneous canonical parent
  evidence, enforces the child's exact unique-parameter budget, fingerprints
  the carrier/parents/child, and performs zero-gradient blank bootstrap.
- `hf_blank_checkpoint.py` already checks shape and exact parameter budget
  before publication and verifies a parent-free reload.
- `hf_blank_bootstrap.py` already operates on complete attention and gated-MLP
  target circuits with held-out transaction rollback. It is the current
  architecture transport/bootstrap layer, not a proof of retained behavior.
- `hf_dense_gqa_qkvo_residual_adapter.py` is target-shaped and corrects Q/K/V
  before Q/K normalization, RoPE, and cache append, and O after physical-group
  attention context. Its native bake is the correct publication mechanism.

### Useful intermediate representations, not final native publication forms

- `hf_budget_ir.py` factorizes vocabulary operators and ties MLP parameters
  across logical layers. Its compact form has non-native shared/factorized
  runtime modules, while materialization expands back to the source dense
  parameter count. It is useful for search and diagnosis but cannot by itself
  certify the exact standalone native endpoint.
- `hf_factorized_vocab.py` has exact rank and storage accounting. Its factors
  may guide token-row selection, but a dense target publication must embody
  the chosen vocabulary width/tie policy in the target architecture rather
  than retain a factorized head sidecar.
- `hf_depth_basis_ir.py` and the neuron-codebook machinery preserve execution
  depth through basis registries, assignments, indices, and scales. They are
  valuable proposal generators and proven boundary frontiers, but their
  auxiliary storage must be counted and they are not the native dense
  progressive targets above.

## The remaining high-leverage missing operator

The QKVO bridge resolves the attention-side publication boundary, but attention
is not the dominant 25x budget. The exact decompositions show that MLP, width,
depth, and vocabulary dominate.

The next compiler operator should mirror this QKVO transaction for a complete
target-native gated MLP:

- target-local gate/up corrections before SwiGLU;
- target-local down corrections after the real post-SwiGLU features;
- uneven neuron/site allocation from the existing sensitivity, codebook, and
  multi-anchor evidence;
- output/rollout-only optimization and promotion;
- dense `nn.Linear` bake into the already narrow target gate/up/down matrices;
  and
- zero registry, index, scale, or source-width basis storage after bake.

Depth changes must likewise be expressed by the target's actual layer count.
The existing span planner and ordered residual composition provide proposals,
but only parent-relative held-out behavior can promote a `36 -> 30`, `30 ->
24`, or cross-tokenizer `24 -> 30` compilation.

## Proven versus unproven

Structurally proven locally:

- all three closed-form target counts, including the exact `134,515,008`
  decomposition;
- strict monotone stage accounting and the `29.903489x` endpoint;
- target geometry derivation from an explicit supported decoder layout;
- rejection before adapter installation on a budget or topology mismatch;
- exact rollback of a rejected temporary correction;
- behavior-receipt requirement before bake;
- live-to-baked output equivalence in the focused integration test;
- exact native parameter count, state keys, and structure after bake; and
- no published QKVO wrapper, registry, adapter parameter, adapter FLOP, parent
  reference, or source-sized QKVO sidecar.

Not proven by this local bridge:

- that any of the 1.410B, 600.628M, or 134.515M blank targets retains useful
  parent behavior;
- that the existing one-site p35 QKVO win generalizes to many target sites or
  to the smaller architectures;
- a coherent zero-recovery 25x child;
- arbitrary cross-tokenizer, multi-parent, or multimodal retention; or
- complete serialized artifact bytes before an actual save/reload audit.

Those remain behavioral experiments, not consequences of parameter
accounting.
