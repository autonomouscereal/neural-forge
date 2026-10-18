# S4 — safetensors IO, the two exact-import loaders, and the two receipts

Branch `claude/omni-import` in `C:/Users/cereal/Documents/ai training/chimera-omni`.
Spec: `../r4-chimera-import-lane/CHIMERA_IMPORT_LANE_SPEC_2026-09-01.md` sections 1.12,
1.16, 1.17 and the R-Q / R-K acceptance receipts.

Everything below is a measurement. No threshold in this document is a gate.

---

## 1. What ran, and what it measured

### R-Q — Qwen3.5-0.8B, whole model, 8-prompt pack

`scripts/import_receipt_qwen35.py`, lab venv on the AI server, torch 2.11+cu128 /
transformers 5.12.1. Source: `/home/cereal/models/Qwen3.5-0.8B`
(`model.safetensors-00001-of-00001.safetensors` sha256 `04b1c301231dd422…`, 1,746,942,600 B;
`config.json` `b90b86f35c8e6925…`; `tokenizer.json` `5f9e4d4901a92b99…`). Prompt pack: 8
fixed texts (prose, Python, Chinese, Cyrillic, a number run, SQL, a long sentence)
tokenised by `ChimeraTokenizer.from_qwen`, 13/24/26/9/66/23/14/35 tokens,
pack sha256 `7028db47e85ebfc3…`. 210 predicted positions in total.

| column | max abs logit diff | argmax agreement | max abs hidden diff, layers 0-3 |
|---|---|---|---|
| CPU fp32, torch fallbacks, **recurrent** delta rule | **3.19e-05** | **210/210 = 100%** | **7.75e-07** |
| GPU6 (Quadro RTX 8000) bf16, fla chunked kernel | 5.00e-01 | 209/210 = 99.52% | 1.56e-02 |

Reference logit absolute max on the same prompts: 28.16 (fp32) / 28.25 (bf16), so the fp32
residual is ~1.1e-6 relative. Per-layer, fp32, on the 66-token prompt: layer 0 (kda)
2.98e-07, layer 1 (kda) 4.77e-07, layer 2 (kda) 5.96e-07, layer 3 (gated_gqa) 7.75e-07 —
the error grows linearly with depth, which is what pure fp32 rounding looks like.

The bf16 column is bf16 arithmetic noise (8 mantissa bits, ~0.4% relative) plus the
chunked-vs-recurrent difference; it is reported, not gated. The one argmax disagreement is
on prompt 1 (the Python snippet), 1 position of 24.

Parameter accounting from the import receipt: 752,427,584 trunk parameters written from
Qwen tensors, 4,096 band rows (4,194,304 band parameters, tied head), 770,441,792 total in
the Chimera model; skipped and enumerated with shapes and hashes: `mtp.*` 15 tensors /
20,452,864 parameters, `model.visual.*` 153 tensors / 100,592,896 parameters, `other` 0.
Every source tensor is either consumed exactly once or listed under `skipped` — the test
`test_receipt_accounts_for_every_source_tensor` asserts the partition.

Checkpoint round trip (spec R-Q step 5), bf16 on GPU6: `write_safetensors` → 1 shard,
1,540,891,136 B, `tied_weights = {"lm_head.weight": "embed.weight"}` → `load_chimera_checkpoint`
→ `state_sha256` identical (`1877a83e7ee2527f…` both sides) and the logits
`torch.equal`. 30.1 s.

Receipt: `receipts/qwen35_0p8b_import_receipt.json` (also
`/home/cereal/omni-campaign/import-lane/s4/qwen35_0p8b_import_receipt.json`).

### R-K3 — Kimi K3 layer 0 (KDA + dense SiTU MLP + AttnRes)

`scripts/import_receipt_k3_layer.py --layer 0`. Reference = `KimiDecoderLayer` from the
checkpoint's own `modeling_kimi_linear.py`, `load_state_dict` reporting **zero missing and
zero unexpected keys**, driven through `_forward_attn_residual` with
`block_residual = zeros(B*T, 0, H)`. Chimera side = `import_k3_layer_state` driven through
the layer-0 slice of `_forward_trunk`. Input: seeded `randn[1, 64, 7168]`, seed 0.

| column | layer output max abs diff | relative L2 | mixer output max abs diff | recurrent state |
|---|---|---|---|---|
| CPU fp32 (torch stand-ins for the 3 Triton pieces) | **3.47e-08** | **9.72e-07** | 3.54e-08 | rel L2 **1.40e-05** as-is |
| GPU6 bf16 (fla `chunk_kda`) | 4.88e-04 | 6.10e-03 | 2.44e-04 | rel L2 7.91e-03 **transposed** |

Reference output rms 0.00408, absolute max 0.0898.

The recurrent-state row is the interesting one. `KimiDeltaAttention` calls `chunk_kda(...,
transpose_state_layout=True)`, so the Triton kernel hands back `[B, H, V, K]` while Chimera
holds `[B, H, K, V]`; the naive torch path does not transpose. The script measures **both**
orientations every time instead of assuming one, and the numbers pick the orientation out
unambiguously: on GPU the transposed comparison is 7.9e-03 and the as-is comparison is 1.41
(i.e. noise), on CPU it is the other way round (1.40e-05 as-is, 1.41 transposed).

`block_residual`: reference pushed 1 block, Chimera pushed 1, and both are bit-identical to
the layer input (`pushed_input_unchanged: true`).

Read: 23 tensors, 2,341,213,184 B. First pull off `/mnt/ai_data` took 93 s; the staged copy
under `/home/cereal/omni-campaign/import-lane/s4/k3_layer0_staged` reads in 8.5 s (262 MB/s).

Receipt: `receipts/k3_layer0_import_receipt.json`.

---

## 2. Three things the checkpoints do that the spec did not say

### (a) fla scales the query by `1/sqrt(head_dim)` and Chimera did not

`fla.ops.kda.naive.naive_recurrent_kda` does `q = q.repeat_interleave(G, dim=2) * scale`
with `scale = K ** -0.5`, and `transformers`' `torch_chunk_gated_delta_rule` /
`torch_recurrent_gated_delta_rule` both do `query = query * (1 / query.shape[-1] ** 0.5)`.
`KimiDeltaAttention` had no such scale.

It looks absorbable — the output goes straight into an RMS norm, and RMS norm is
scale-invariant — but that norm has an epsilon, so `rmsnorm(c·o)` is **not**
`rmsnorm(o)` when `c² · mean(o²)` is comparable to eps. At K3's head geometry
`c² = 1/128` and `eps = 1e-5`, which is not negligible.

New switch `kda_query_scale: Literal["none", "inv_sqrt_key_dim"] = "none"`
(`src/chimera_omni/config.py:75`, used at
`src/chimera_omni/modeling/layers.py:206-220` and `:300-305`). Default is the old behaviour;
both import profiles set `"inv_sqrt_key_dim"`. Chimera uses `float(K) ** -0.5` (fla's
spelling); transformers writes `1 / K ** 0.5`, which differs by one ulp at K=128 and is
recorded rather than papered over.

### (b) `KimiMLAAttention` builds its two latent norms with the class default eps

`self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank)` and
`self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)` — no `eps` argument, so they keep
`KimiRMSNorm`'s default `1e-6` while every other norm in the model uses
`config.rms_norm_eps = 1e-5`. Chimera used `rms_norm_eps` for both.

New field `mla_latent_norm_eps: float = 0.0` (0 → `rms_norm_eps`), read by
`GatedMultiHeadLatentAttention`; `k3_chimera_config` sets `1e-6`. This does not affect the
layer-0 receipt (KDA), but it would have made every MLA layer wrong.

### (c) K3 ships `A_log` zero-padded from `num_heads` to `head_dim`

On disk, `layers.0.self_attn.A_log` is F32 `[128]`, while `b_proj.weight` is `[96, 7168]`
and `dt_bias` is `[12288] = 96 × 128`. Entries `[96:]` are **exactly 0.0**. fla reads it as
`A_log.view(H, 1)` with `H = 96`, so the padded tensor cannot be used as-is (that is what
the first run of the receipt script failed on, and what llama.cpp's converter is doing when
it writes `-exp(A_log)[:n_head]`).

`trim_padded_a_log` (`src/chimera_omni/importing/kimi_k3.py:139`) trims to `num_heads`,
**refuses** a non-zero tail rather than guessing, and records the trim in
`receipt["source_adjustments"]`. The receipt script applies the same trim before
`load_state_dict` on the reference layer, which is why the reference loads with zero
missing / zero unexpected keys.

---

## 3. Files

New:
- `src/chimera_omni/importing/safetensors_io.py` — sharded writer, offset-sorted reader,
  `load_chimera_checkpoint`, `state_sha256`, `TensorView` (numpy + json; torch lazy).
- `src/chimera_omni/importing/_common.py` — read-once/record/shape-checked-write plumbing.
- `src/chimera_omni/importing/qwen35.py` — `import_qwen35_state`, `load_qwen35`.
- `src/chimera_omni/importing/kimi_k3.py` — `k3_layer_tensor_names`,
  `import_k3_layer_state`, `load_k3_layer`, `import_k3_model_level_state`,
  `trim_padded_a_log`, `dequant_mxfp4`, `load_k3_moe_tensors`.
- `scripts/import_receipt_qwen35.py`, `scripts/import_receipt_k3_layer.py`.
- `receipts/qwen35_0p8b_import_receipt.json`, `receipts/k3_layer0_import_receipt.json`.
- Tests: `test_safetensors_io.py`, `test_import_tensor_plans.py`,
  `test_import_qwen35_synthetic_torch_optional.py`,
  `test_import_k3_synthetic_torch_optional.py`,
  `test_checkpoint_round_trip_torch_optional.py`.

Changed: `config.py` (two new fields, both defaulting to current behaviour),
`modeling/layers.py` (query scale, latent norm eps), `importing/profiles.py`,
`importing/__init__.py`, `architecture_manifest.json` (regenerated; the diff is exactly the
two new keys).

Server suite: **332 passed** (`python -m pytest --tb=short -p no:warnings tests`, 16.87 s),
of which **42** are the five S4 test files (2.49 s).

---

## 4. What is still owed

- The synthetic tests prove the loaders against transcriptions of the reference code; the
  receipts prove them against the real modules. Neither proves the **MLA** layer against
  the real K3 module — only against the in-test transcription. `--layer 3` was not run
  (the MLA layer lives in shard 4-5 and was not staged).
- The 4B and the 27B-Uncensored Qwen receipts were not run. The 27B still blocks on spec Q1
  (`output_gate_type: "swish"`).
- The routed-expert path (`load_k3_moe_tensors`) is written but never executed; only
  `dequant_mxfp4` is unit-tested, against a hand-computed nibble pattern.
- No GGUF export (`export/`, spec 1.14) exists.
- The bf16 columns are honest bf16 noise, not an import claim. A bf16 exactness claim would
  need the `1 + w` norm folding to move into the forward pass.

---

## 5. Superseded in places — read `S4_BLOCKER_FIXES_2026-09-02.md`

A review of this stage found four blockers, all since fixed on the same branch. Three of
them change what this document says:

- `state_sha256` and `proof.h3_hybrid_av_flow.hash_state_dict` are **not** comparable
  (different dtype tokens); the claim that they were has been removed from the code.
- `hash_state_dict` crashed on any `block_attnres` model (0-dim `prefix_logit_bias`), so
  the joint-AV proof driver was broken from S1 until this fix.
- The K3 and Qwen loaders now refuse the switches their receipts depend on
  (`mla_latent_norm_eps`, `kda_query_scale`, `kda_qk_norm_kind`) instead of accepting a
  wrong one silently.

The suite is **343 passed** after the fixes, and both receipts were re-measured: every
number in section 1 and 2 reproduced. Two figures quoted here are corrected there — the
331 s K3 reference was Triton autotune (1.478 s warm), and `chimera_state_sha256` is a
per-run value because 16 parameters have no Qwen source and are randomly initialised.

> **Superseded 2026-09-02 (S4b, commit 39f6530):** the MLA layer HAS now been measured against the real
> `KimiMLAAttention` on layer 3 — see `IMPORT_LANE_STATUS.md` §S4b and `receipts/k3_layer3_mla_receipt.json`
> (fp32 mixer max abs diff 8.94e-08; every projection and norm bit-identical).
