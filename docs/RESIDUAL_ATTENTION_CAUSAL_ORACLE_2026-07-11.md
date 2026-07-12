# Residual-attention causal oracle

The residual-attention scorer tells us that a candidate trajectory is damaged.
This oracle identifies which recurrent attention state is causal by replaying
the same token path and temporarily borrowing one dense-parent signal.

It runs six fresh-cache controls:

| Control | Parent signal substituted at one logical Qwen layer |
|---|---|
| `none` | Nothing; folded-child baseline |
| `k_cache` | Selected post-RoPE GQA K-cache groups |
| `v_cache` | Selected GQA V-cache groups |
| `attention_delta` | Full `o_proj` residual write |
| `kv_cache` | Selected K and V cache groups |
| `kv_cache_and_attention_delta` | Redundancy/consistency control: K, V, and the full residual write |

`kv_cache_and_attention_delta` is not counted as independent causal evidence.
Because the full parent `o_proj` return replaces the target layer's attention
write, it causally dominates any K/V effect on that write during the same step.
Its purpose is to prove that the cache and delta mechanisms compose without
changing the already delta-clamped output. K, V, delta, and K+V are the four
independent intervention comparisons.

The official candidate `DynamicCache.update` always runs first. The oracle then
replaces selected entries in the returned and stored cache, so sequence length,
cache metadata, Qwen query normalization, RoPE, GQA expansion, masking, softmax,
and all non-target layers retain their real execution semantics. Delta
intervention replaces only the transient `o_proj` return. The subsequent Qwen
residual addition is observed and checked.

Before replay, a hashed attention-coordinate contract must match exactly. It
binds the model, decoder-layer, attention, and rotary module types; actual layer
count and selected layer index; hidden/head/KV/head dimensions; RoPE parameters;
maximum positions; sliding-window and layer-type configuration; Q/K norm types
and epsilons; residual norm types and epsilons; attention bias/dropout/scaling;
and projection-bias presence. A same-shaped candidate with different RoPE or
cache-position semantics is rejected before any raw parent tensor is injected.

The dense Qwen3-4B checkpoint exposes 151,936 output rows while the compact
codebook child exposes 151,669 logical tokenizer rows. Before loading the dense
parent, the runner reads the child's persisted `parent_token_rows` tensor. Child
prompt and forced-token IDs are mapped into dense-parent IDs for capture; parent
logits and token paths are then projected back into the child chart. If a dense
greedy action lands on an unmapped padded row, capture fails closed. The chart,
both token paths, projected logit hashes, and recomputed trace byte counts are
identity-bound. The count includes the persisted int64 vocabulary chart
(1,213,352 bytes for 151,669 child rows). Candidate replay independently verifies its persisted
`parent_token_rows` buffer against that chart.

Interpretation is deliberately causal:

- Strong K recovery isolates address/attention-score geometry or recurrent K.
- Strong V recovery isolates remembered value content.
- K+V recovery stronger than either alone identifies a coupled cache failure.
- Delta recovery with weak K/V recovery means the final attention write is bad
  even when the isolated selected cache groups are repaired.
- Weak recovery for every single-layer control means the damage is distributed,
  starts before that layer, lies in the MLP/logit path, or requires a joint
  multi-layer intervention.

`forced-cache-recurrent` consumes the parent trace's selected tokens for every
control. This holds action feedback fixed while allowing cache error to recur,
so full-horizon recovery ratios are comparable. `autoregressive` consumes the
candidate's greedy token and stops at the first action divergence. Continuing
with parent K/V after divergence would silently use evidence from the wrong
prefix and is forbidden.

## Completed strict FP32 v2 localization

The completed evidence is narrower than the all-layer/all-group command below.
It uses one code prompt, physical site `L19/G4`, horizon four,
`forced-cache-recurrent` feedback, and separate comparisons of p35 and p34 to
the same dense parent. All six controls completed the full horizon; the suite
marked the causal comparison valid, left candidate state unchanged, wrote no
checkpoint, and mutated no checkpoint.

| Child | Intervention | Top-1 agreement | Centered-logit RMSE | Parent-to-child KL | KL recovery vs baseline |
|---|---|---:|---:|---:|---:|
| p35 | none | 4/4 | 0.820874 | 0.075446 | — |
| p35 | parent K | 4/4 | 0.820723 | 0.076711 | -1.68% |
| p35 | parent V | 4/4 | 0.815979 | 0.071226 | +5.59% |
| p35 | parent K+V | 4/4 | 0.814941 | 0.071470 | +5.27% |
| p35 | parent residual write | 4/4 | 0.810114 | 0.069604 | +7.74% |
| p34 | none | 4/4 | 1.029378 | 0.155213 | — |
| p34 | parent K | 4/4 | 1.066804 | 0.189554 | -22.13% |
| p34 | parent V | 3/4 | 1.034352 | 0.190618 | -22.81% |
| p34 | parent K+V | 3/4 | 1.064639 | 0.225014 | -44.97% |
| p34 | parent residual write | 4/4 | 0.952474 | 0.115764 | +25.42% |

The `K+V+residual-write` replay is omitted from the table because v2 labels it
a redundancy control: once the complete target-layer residual write is
substituted, it is bit-identical in these aggregate metrics to residual-write
alone. The independent p34 result is therefore specific: raw parent K, V, or
K+V worsened this path, while the full parent residual write improved KL by
`25.42%`. This motivates residual-contribution fitting at this site; it does
not prove the same intervention wins on other prompts, layers, groups, or
closed-loop autoregressive trajectories.

Evidence files:

- `runs/residual-attention-20260711/p35_layer19_group4_h4_forced_cache_oracle_fp32_v2.json`
- `runs/residual-attention-20260711/p34_layer19_group4_h4_forced_cache_oracle_fp32_v2.json`

## p34 command

Run from the isolated residual-attention checkout, not the active unified-run
checkout. The parent is released before the p34 child is loaded.

```bash
PYTHONPATH="$PWD/src" python experiments/run_qwen_residual_attention_oracle.py \
  --parent-model /home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --candidate-codebook-checkpoint /mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa34_gpu6_r1_20260711/compact_child \
  --tokenizer /home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --prompt "Complete the Python function: def add(a, b):" \
  --layer-index 35 \
  --group-indices 0,1,2,3,4,5,6,7 \
  --horizon 8 \
  --lane greedy \
  --feedback forced-cache-recurrent \
  --device cuda:0 \
  --dtype float32 \
  --local-files-only \
  --output /mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711/p34_layer35_allgroups_oracle.json
```

FP32 strict mode is the evidence-grade default for the causal table above. An
FP16 diagnostic run may use declared tolerances such as `--logit-rtol 0.01
--logit-atol 0.04`. For functional screening, cached/full-prefix or activation
numerical exceedances should also be retainable as record-only diagnostics
rather than automatic candidate vetoes; structural coordinate mapping, cache
shape, hook restoration, and model/trace identity checks remain separate. A
record-only functional result must not be relabeled as a strict causal-oracle
result.

Repeat over damaged layers and groups. Group-local runs localize a bad prototype
assignment; all-group runs test the whole layer. A separate `--feedback
autoregressive` run reports the first closed-loop action divergence but does not
claim recovery after it.

## 158M child boundary

Raw transplantation is valid only when parent and child share:

- tokenizer ID semantics or an exact bound vocabulary chart;
- hidden width and residual coordinate system;
- layer index;
- Q/K/V head counts, GQA grouping, and head dimension;
- Transformers attention backend and `DynamicCache` layout.

The current runner intentionally rejects a 158M child that changes any of those
coordinates. That rejection does not say cross-architecture residual-attention
folding is impossible. It says a raw tensor swap would not be evidence. Before
running this oracle on such a child, add and separately certify three adapters:

1. a token/path bridge that proves the driven child tokens represent the same
   prefixes;
2. a post-RoPE per-head K/V map into the child cache geometry;
3. a parent-residual-to-child-residual delta map.

The adapted oracle must compare adapter reconstruction controls first and bind
the adapter hashes into the trace identity. Until that exists, p34/p35 Qwen
codebook children are the valid causal test bed; the 158M child remains a later
mapped-intervention experiment.

## Evidence and mutation contract

- Parent traces and every full-vocabulary logit row are detached to bounded CPU
  storage.
- Model config plus every state tensor byte is SHA-256 hashed in bounded chunks
  before and after capture/replay.
- The parent attention-coordinate contract and vocabulary-chart bytes are bound
  into the trace identity and revalidated against the candidate runtime.
- Tokenizer vocabulary, special IDs, model class, Transformers version,
  attention backend, cache class, layer topology, prompt, trajectory, and trace
  tensors are identity-bound.
- Hooks, train/eval modes, and mutable config flags are restored exactly.
- Inference mode is mandatory; there is no optimizer, gradient, checkpoint
  write, or permanent model mutation.

Implementation:

- `src/neural_forge/hf_residual_attention_oracle.py`
- `experiments/run_qwen_residual_attention_oracle.py`
- `tests/test_hf_residual_attention_oracle.py`
