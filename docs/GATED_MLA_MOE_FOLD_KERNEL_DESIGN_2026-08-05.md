# Gated-MLA / Stable-LatentMoE Fold Kernel Design

Updated: 2026-08-05
Status: implementation contract for the last two named chimera-omni kernel
gaps.  Written to the standard of
`docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`, which this document
extends rather than replaces: every shared convention there (sealed evidence,
frozen fit/selection/validation partitions, condition-capped ridge, BF16
intermediate replay, the 1.0 no-branch floor, contract constants versus fold
coordinates) is in force here and is not restated.

## Problem

After `NOTES-W12C.md` four of chimera-omni's five sequence-mixer families fold
as planned, ledgered transactions.  Two named gaps remain, and the planner
reports both by name
(`dynamic_fold_planner.py:99-109`, `NOTES-W12C.md` §3):

| declared family | capability the planner asks for |
|---|---|
| `compressed_latent_attention` (`gated_mla`) | `compressed_latent_kv_attention_layer_operator` |
| `stable_latent_moe_router` / `stable_latent_moe_expert` | `stable_latent_moe_channel_mixer_operator` |

They are the two families of the *k3-aligned* reference configuration
(`chimera-omni/src/chimera_omni/config.py:249-290`), so until both close, the
architecture chimera-omni actually targets cannot be folded at all: a layer
whose channel mixer has no owning transaction cannot be closed by folding its
sequence mixer alone.

## Ground truth: what the two modules actually are

Both statements below are read off the executable
(`chimera-omni/src/chimera_omni/modeling/layers.py`), not off the K3 report.

### Gated MLA -- `layers.py:231-306`

```text
query path (compressed when mla_query_latent_size > 0):
    q = q_up(RMSNorm(q_down(x)))            layers.py:275-279
    else  q = q_proj(x)
    q is [B, S, heads, Dq]  with  Dq = Dk + Dx

key/value path:
    c        = kv_down(x)                   [B, S, L + Dx]      layers.py:281
    latent   = c[..., :L]                                        layers.py:282
    kv       = kv_up(RMSNorm(latent))       [B, S, heads*(Dk+Dv)] layers.py:283
    k_base, v = split(kv, [Dk, Dv])                              layers.py:289
    k_extra  = c[..., L:]  broadcast over every head             layers.py:290-294
    k        = cat([k_base, k_extra], -1)   [B, S, heads, Dq]

score / write:
    scores = q k^T / sqrt(Dq),  full causal mask, fp32          layers.py:301-303
    y      = softmax(scores) v                                   layers.py:304
    delta  = out( sigmoid(gate(x)) * y )                         layers.py:306
```

Four consequences the fold must respect.

1. **This is NoPE, not DeepSeek MLA.**  There is no rotary anywhere in the
   architecture (`architecture_manifest.json:17`), and `mla_extra_head_dim`
   is *not* a decoupled RoPE channel set.  It is a **shared, head-independent
   key block** taken straight off `kv_down`'s output, broadcast to every head.
   Assuming DeepSeek semantics here would invent a rotation the module does
   not have.
2. **The key side factors through the latent exactly.**  For every head,
   `k_base_h(s) = W_k_h n(latent_s)`, so
   `q_base_h(t)^T k_base_h(s) = (W_k_h^T q_base_h(t))^T n(latent_s)`.  The
   per-head key contribution to any score therefore has rank at most `L`,
   independently of `Dk`.  This is the MLA absorption identity and it is exact.
3. **The shared block adds a rank-`Dx` term to every head's score**:
   `q_extra_h(t) . c_extra(s)`.  It is the same `c_extra` for all heads, so it
   is a head-independent bilinear term, and its share of the score energy is
   measurable exactly from cached rows.
4. **The gate is channel-wise and pre-writer**, exactly like KDA's
   (`layers.py:187-190` vs `:306`), so the writer solve is endpoint-coupled
   against the *gated* context.

### Stable LatentMoE -- `layers.py:362-445`

```text
flat        = x.reshape(-1, H)                                   layers.py:430
raw_scores  = sigmoid(router(flat))                              layers.py:396-397, 431
indices     = topk(raw_scores + router_bias, k)                  layers.py:403
weights     = raw_scores.gather(indices) / sum(...).clamp_min(1e-6)  layers.py:404-405
latent      = latent_down(flat)                                  layers.py:434
routed[r]  += weights[r, slot] * expert_j(latent[r])   for each expert j
                                                                 layers.py:436-441
out         = latent_up(routed_norm(routed))                     layers.py:442
out        += shared_i(flat)   for each shared expert i           layers.py:443-444
```

with each expert a `FeedForward`
(`layers.py:331-359`): one fused `up: [2*intermediate, in]`, chunked in two,
one activation, one `down`.  The activation is `silu(gate) * up` or, under
`activation_kind = "situ"`,
`beta_g tanh(gate/beta_g) * sigmoid(gate) * beta_u tanh(up/beta_u)`.

Five consequences.

1. **The routing is recomputable exactly** from the captured `router` output
   plus the carried `router_bias` buffer.  `router_bias` enters *selection
   only*; the mixture weights are the unbiased sigmoid scores renormalized
   over the selected set.
2. **`router_bias` is a non-gradient buffer, not a learned tensor.**  It is not
   in `named_parameters`, so it is not in the IR, so no fold transaction owns
   it.  It is therefore a **contract constant**: the fold requires the child's
   buffer to equal the parent's and refuses otherwise, rather than silently
   folding a child whose selection boundary is somewhere else.
3. **Routing is sparse, so each expert's solve consumes only its own rows.**
   With `N` token rows, `E` experts and `K` experts per token, expert `j` sees
   `n_j` rows summing to `N*K`.  A fold that ignores this fits an expert on
   rows it never sees.  The row budget is therefore a *fail-closed condition*,
   not a diagnostic: an expert whose fit-partition row count falls under the
   sealed policy floor makes the whole channel mixer refuse, naming the expert
   and the count.
4. **The residual write is one value.**  `latent_up(...) + sum_i shared_i(...)`
   is written into the residual stream exactly once, so the whole module is
   **one composite operator node**, by the same rule that made CSA/HCA one node
   (`KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md` §4, `NOTES-W12C.md` §1).
5. **The latent path carries no normalization before the experts** (unlike
   MLA's `kv_norm`); `routed_norm` acts on the accumulated routed value.  A
   latent-width change is therefore a plain change of basis on the expert input
   and on the accumulator, and the norm is refit against the accumulator the
   child actually computes.

## Section 1 -- Gated MLA: coordinates, gates, recipe

### 1.1 Fold coordinates

| coordinate | verdict | why |
|---|---|---|
| `query_heads` (= kv heads; MLA is not grouped) | contraction coordinate | barycentric complete head assignment, exactly the GQA pattern (`hf_causal_gqa_attention_fold.py:274-302`) |
| `key_head_dim` `Dk` | contraction coordinate | energy-ranked channel selection on the merged head targets; NoPE, so there are no rotation pairs to preserve |
| `value_head_dim` `Dv` | contraction coordinate | same, ranked through the writer and the gate |
| `kv_latent_size` `L` | contraction coordinate | POD truncation of the observed latent rows; the tail is an exact singular-value statement |
| `query_latent_size` `Lq` | contraction coordinate *when present* | same; `Lq == 0` on both sides selects the dense-query variant |
| `extra_head_dim` `Dx` | **contract constant** | it is the shared/per-head split of the score itself; no evidence anywhere says what the parent would have computed under a different split |
| causal mask | **contract constant** | full causal, no window; a window here would be a different operator class |

### 1.2 Evidence and the gates that bind it

Every row comes from a sealed module boundary of the one parent forward
(`dynamic_native_circuit_evidence.py`), never from a parent call:

| field | boundary |
|---|---|
| `block_input`, `attention_norm_output` | input / output of the branch norm |
| `query_latent_rows`, `normalized_query_latent_rows`, `query_rows` | `q_down`, `q_norm`, `q_up` outputs (or `q_proj` output) |
| `compressed_kv_rows`, `normalized_latent_rows`, `kv_rows` | `kv_down`, `kv_norm`, `kv_up` outputs |
| `gate_rows` | `gate` output (**pre**-sigmoid) |
| `attention_delta` | the mixer module's own output |

Derived rows -- named as derived in the receipt -- are the key/value split, the
broadcast shared key block, the scores, the probabilities, the sigmoid gate and
the context.  Two gates bind them:

* **`latent_key_value_readout_identity`** recomputes `kv_up(kv_norm(latent))`
  from the captured normalized latent rows and the carried `kv_up` weight and
  compares against the captured `kv_up` output.  Exact in exact arithmetic;
  the parent runs fp32 so the measured number is fp32 round-off.
* **`latent_kv_attention_reconstruction`** rebuilds
  `out(sigmoid(gate) * softmax(scores) v)` from the derived rows plus the
  carried `out` weight and compares against the **captured** `attention_delta`.
  This is the KDA-unroll analogue: a wrong split, a wrong broadcast of the
  shared key block, a wrong mask or a wrong gate cannot survive it.

Both run **once, before the ridge ladder**, so an inconsistent cache fails
closed with the kernel's own measurement rather than as "no good ridge".

### 1.3 Ordered closed-form recipe

1. Contract constants; contraction-only geometry; evidence gates above.
2. Barycentric complete head assignment; per-head key and value channel
   selection by energy on the merged targets.
3. KV latent basis: identity when `L' == L`, otherwise the leading `L'` right
   singular vectors of the observed latent rows on the **fit** partition.  The
   discarded singular energy is reported exactly.
4. `kv_down` ridge from child rows to `[latent_p U | c_extra]`; `kv_norm` gain
   fitted diagonally against `n(latent_p) U`; `kv_up` ridge from the child's
   **replayed** normalized latent to the merged/selected key and value targets.
   Replaying the child's own RMSNorm rather than fitting through it is the
   GQA kernel's pre-RoPE discipline transplanted.
5. Query path, same shape: `q_down` ridge, `q_norm` diagonal fit, `q_up` ridge
   to `[q_base selected | q_extra]`; or one `q_proj` ridge in the dense variant.
6. **Installed-probability replay**: child scores from the installed query and
   key paths under the child's own causal mask.  Probability overlap against
   the parent's merged probabilities is a sealed policy floor.
7. Gate ridge to the parent's **pre-sigmoid** gate rows (no inverse-sigmoid is
   needed: the captured boundary is already the pre-activation), with the
   saturated-row fraction reported.
8. Writer: endpoint-coupled ridge from the replayed gated context to the branch
   delta, chart-projected when the residual width changes.  MLA needs no score
   chart -- unlike CSA's full-width summary score, every MLA score is an inner
   product in *head* coordinates, and every consumer fit reads child-width rows.
9. BF16 replay of every intermediate, then the 1.0 no-branch floor.

## Section 2 -- Stable LatentMoE: coordinates, gates, recipe

### 2.1 Fold coordinates

| coordinate | verdict |
|---|---|
| `moe_latent_size` `Lm` | contraction coordinate (POD basis on observed latent rows) |
| `moe_routed_intermediate_size` `Ri` | contraction coordinate (functional dictionary selection per expert) |
| `intermediate_size` `I` (shared experts) | contraction coordinate (same) |
| `moe_num_experts` `E` | **contract constant** -- changing it changes the router's output width and therefore which rows exist |
| `moe_experts_per_token` `K` | **contract constant** -- it changes the mixture, not a width |
| `moe_shared_experts` `S` | **contract constant** -- a different number of full-width writers is a different operator |
| `activation_kind` + SiTU betas | **contract constant** -- a different nonlinearity is a kind translation |
| `router_bias` | **contract constant** (a non-gradient buffer; see ground truth 2) |

### 2.2 Which evidence rows each expert's solve consumes

This is the question sparse routing forces, and the answer is the design's
fail-closed rule.

Let `rows_j` be the token rows the parent routed to expert `j`, recovered
exactly from the captured router output plus the carried bias.  Expert `j`'s
three solves consume exactly `rows_j & fit`:

* gate ridge and up ridge: child latent rows at `rows_j & fit` against the
  parent's captured expert gate/up rows;
* dictionary selection: the realizable feature columns on the same rows;
* down ridge: the selected features against the parent's captured expert output
  rows, projected into the child latent basis.

The expected count is `N*K/E`.  The sealed policy carries an absolute floor
(`minimum_expert_fit_rows`) and a relative floor
(`minimum_expert_row_fraction` of the expected count); an expert under either
one makes the whole fold refuse, naming the expert, its count and the floor.
It is not a warning: an expert fitted on a handful of rows is an interpolation
of noise that the endpoint gate cannot see, because it is only ever exercised
on those same rare rows.

### 2.3 Evidence and the gates that bind it

| field | boundary |
|---|---|
| `router_logit_rows` | `router` output (**pre**-sigmoid) |
| `latent_rows` | `latent_down` output |
| `expert_input_rows[j]`, `expert_output_rows[j]` | input / output of `routed_experts.j` -- captured only for the rows the parent routed there |
| `expert_gate_rows[j]`, `expert_up_rows[j]` | `routed_experts.j.up` output, chunked in two |
| `routed_rows`, `normalized_routed_rows` | input / output of `routed_norm` |
| `routed_delta` | `latent_up` output |
| `shared_gate_rows[i]`, `shared_up_rows[i]`, `shared_delta_rows[i]` | `shared_experts.i.up` output and `shared_experts.i` output |
| `feed_forward_delta` | the channel mixer's own output |

Two gates:

* **`stable_latent_moe_routing_identity`** recomputes the routing and rebuilds
  the accumulator `sum_j scatter(w_j * expert_output_j)` from the captured
  per-expert outputs, and compares it against the captured `routed_norm`
  *input*.  It also checks each recovered `|rows_j|` against the captured row
  count of expert `j` -- an integer equality, so a mis-recovered assignment
  fails before any arithmetic tolerance is consulted.
* **`stable_latent_moe_channel_split`** checks
  `routed_delta + sum_i shared_delta_i == feed_forward_delta` on three
  independently captured values.  This is the CSA split gate for the channel
  axis.

### 2.4 Ordered closed-form recipe

1. Both gates above; the expert row budget; the contract constants.
2. Router ridge from child rows to the parent's pre-sigmoid router logits, then
   the child's own routing recomputed with the carried bias.  Top-`K`
   set agreement against the parent is a sealed policy floor.
3. Latent basis `U` (identity when `Lm' == Lm`), `latent_down` ridge.
4. Per expert: gate/up ridges, deterministic functional-column selection,
   down ridge -- all on `rows_j & fit`, all in the child latent basis.
5. Child accumulator replay with the child's routing, `routed_norm` diagonal
   fit, endpoint-coupled `latent_up` ridge.
6. Shared experts: the same dictionary fit at full width, each against its own
   captured delta.
7. Bounded alternation (2 rounds) between the routed writer and the shared
   writers against `delta - other`, every sub-step an exact ridge, applied only
   if the unregularized objective did not rise.
8. BF16 replay of the whole mixer -- router, top-`K`, experts, norm, both
   writers -- then the 1.0 no-branch floor.

**Fit coordinates for a 4x child.**  Every writer here (`latent_up`, each
shared `down`) maps into the residual, and every one of them is fitted directly
against the *chart-projected* target, so unlike the CSA kernel there is no
parent-coordinate sub-branch fit to transport afterwards.  Every consumer
(`router`, `latent_down`, each `up`) reads child-width rows, so it is
width-agnostic by construction.

### 2.5 Which axes contract for a 4x child

The exact-4x pair used in the proof contracts **only the residual width**;
every operator axis is identity, exactly as `NOTES-W12C.md` recorded for the
four-family crossing.  The reason is arithmetic and is stated with the proof:
with no `[hidden, hidden]` tensor anywhere in this trunk (no CSA/HCA) and every
latent width pinned as an explicit constant, the unique parameter count is
exactly affine in the residual width, `total(H) = a*H + g`, so
`total(H_p) = 4 total(H_c)` has the closed solution `H_p = 4 H_c + 3g/a`
whenever `a` divides `3g`.

## Section 3 -- IR mapping

| chimera block | CircuitKind | node count | key attributes |
|---|---|---|---|
| `gated_mla` | `ATTENTION` | 1 (unchanged) | `projection_style=compressed_latent_kv`, `key_head_dim`, `value_head_dim`, `extra_head_dim`, `kv_latent_size`, `query_latent_size`, `full_rank_output_gate` |
| Stable LatentMoE | `GATED_MLP` | **1 composite** (was 6 + experts) | `channel_mixer=stable_latent_moe`, `composite_operator`, `composite_sub_branches`, `composite_residual_writes=1`, `expert_count`, `experts_per_token`, `shared_expert_count`, `expert_latent_size`, `routed_intermediate_size`, `intermediate_size`, `router_score_activation`, `routed_weight_normalization`, `router_bias_is_non_gradient_buffer` |

`gated_mla` needed no inspector change: its node already declares every
attribute the kernel reads, and it already makes one ordered residual write.

The MoE channel mixer did.  It was emitted as a router node, a latent-down
node, one node per routed expert, a routed-combine `RESIDUAL_ADD`, a norm, a
latent-up node, one node per shared expert and a combine `RESIDUAL_ADD` -- a
decomposition that encodes *two* residual adds the model does not perform and
that no family-neutral branch contract can accept (one norm, one operator, one
ordered write; `dynamic_fold_planner.py:1026-1104`).  It is now one composite
`GATED_MLP` node owning every tensor under `blocks.N.ffn`, with the sub-branch
structure moved onto attributes -- the same resolution `NOTES-W12C.md` §1
applied to CSA/HCA, for the same reason.

`_CHIMERA_STRUCTURALLY_COVERABLE_MIXERS` grows to include `gated_mla`, so
`kernel_gap_block_kinds` is `""` on a k3-aligned graph.

## Section 4 -- Planner, registry, method ids

Two new method ids on two new `EXPERIMENTAL` family lanes:

```text
causal_latent_kv_attention_layer            <- compressed_latent_attention
causal_stable_latent_moe_channel_mixer_layer <- stable_latent_moe_channel_mixer
```

Neither shares a lane with anything registered, so a closure recorded against
one branch operator cannot silently close another.

Per family, what may contract and what may not:

| family | contraction coordinates | contract constants |
|---|---|---|
| `compressed_latent_attention` | `query_heads`, `key_head_dim`, `value_head_dim`, `kv_latent_size` (+ `query_latent_size`, checked in the family arm because it may legitimately be zero on both sides) | `extra_head_dim`, `layer_type`, `sliding_window` |
| `stable_latent_moe_channel_mixer` | `intermediate_size`, `expert_latent_size`, `routed_intermediate_size` | `expert_count`, `experts_per_token`, `shared_expert_count`, `activation`, `situ_gate_beta`, `situ_up_beta`, `router_score_activation`, `routed_weight_normalization` |

The channel-mixer branch arm of `_family_neutral_causal_layer_contract` gains a
third case, checked before the fused-gate-up and split-SwiGLU cases: a MoE
parent and a dense child (or the reverse) is a kind translation and keeps its
gap, because no registered operator crosses between two channel-mixer classes
in one transaction.

### `gated_mla` does not get two method ids

The dense-query (`mla_query_latent_size == 0`) and compressed-query variants
are the *same* `GatedMultiHeadLatentAttention` class under one config constant,
exactly the csa/hca situation.  One executable runs both, so the latent-query
size is a sealed contract *shape* of the crossing -- both sides must be zero or
both positive, and when positive it is a contraction coordinate -- not a second
method id.

## Section 5 -- Fail-closed conditions

Each aborts before any child mutation, inside the existing snapshot
transaction.

1. MLA: the readout identity or the delta reconstruction breaches its
   tolerance (evidence inconsistency).
2. MLA: `extra_head_dim` changes, the query-latent variant changes, a head or
   channel axis expands, or the declared IR geometry disagrees with the live
   module.
3. MLA: replayed probability overlap with the parent below the sealed floor.
4. MoE: the routing identity fails, or a recovered expert row set disagrees
   with the captured expert row count.
5. MoE: the routed/shared split does not reconstruct the sealed write.
6. MoE: any expert under the absolute or relative fit-row floor.
7. MoE: `router_bias` differs between parent and child; `E`, `K`, `S`, the
   activation or a SiTU beta differs.
8. MoE: child top-`K` routing agreement with the parent below the sealed floor.
9. Both: non-finite tensors, overlapping partitions, and the standing rule --
   a validation endpoint at or above the no-branch baseline after BF16 replay.

## Section 6 -- Trade-offs

* **One composite IR node for the MoE (chosen)** versus keeping the expert-wise
  decomposition (rejected): the decomposition is more legible but encodes
  residual writes the module does not perform and cannot satisfy the
  family-neutral branch contract.  The sub-branch structure survives as
  attributes, which is what it is: internal structure of one operator.
* **Experts fitted on the parent's row assignment (chosen)** versus the child's
  own recomputed assignment (rejected): the captured expert evidence *belongs*
  to the parent's assignment; fitting against the child's would compare rows
  the parent never sent to that expert with outputs it never produced.  The
  disagreement between the two assignments is measured and gated
  (`router_top_k_agreement`), not assumed away.
* **`router_bias` as a contract constant (chosen)** versus transporting it
  inside the transaction (rejected): it is a buffer, not a parameter, so it is
  outside the snapshot/rollback envelope every operator here relies on.
  Refusing an unequal buffer is honest; silently folding across a different
  selection boundary is not.
* **An activation-parameterized dictionary fit (chosen)** versus reusing
  `fit_functional_swiglu` (rejected): that helper hardcodes SiLU, and the
  k3-aligned reference runs SiTU.  Reusing it would have restricted the closure
  to half the configurations the family actually has.
* **Channel-aligned selection plus a POD latent basis (chosen)** versus a full
  balanced rotation of the head axis (rejected): the same verdict as the KDA
  design, for the same reason -- the gate and the writer are channel-wise, so a
  rotation conjugates them into tensors the module cannot store.

## Section 7 -- What this does not claim

* No trained chimera parent at scale is folded here; the proof is planning,
  transaction, rollback and audit, plus top-1 agreement against a toy-trained
  parent.
* The exact-4x pair contracts only the residual width.  Head merging, latent
  truncation and expert-dictionary contraction are implemented and unit-tested,
  but the trunk-scale proof does not exercise them.
* One global chart, not a per-boundary chart set: unchanged from W11B, and
  unchanged by this wave.
* The dense (non-MoE) SiTU channel mixer is still refused by the fused
  gate/up lane, which admits SiLU only.  That is an existing lane's envelope,
  not a chimera family gap, and it is deliberately not widened here.
