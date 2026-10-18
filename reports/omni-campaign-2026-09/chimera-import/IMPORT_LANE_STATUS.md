# Chimera Import Lane — Status (S1–S5)

Date: 2026-09-02. Repo: `C:/Users/cereal/Documents/ai training/chimera-omni`, branch `claude/omni-import` (HEAD `f86fe45`).
`main` is untouched at `c1ca0b2`; nothing pushed, nothing rebased. Spec being implemented:
`reports/omni-campaign-2026-09/r4-chimera-import-lane/CHIMERA_IMPORT_LANE_SPEC_2026-09-01.md`.
All server work ran on `ai` (192.168.50.222) through server-manager v2, lab venv `/home/cereal/model-surgery-lab/venv`
(torch 2.11.0+cu128, transformers 5.12.1, fla 0.5.2, gguf 0.19.0, Python 3.12.3). GPU work used GPU6 (Quadro RTX 8000)
with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6` — without `PCI_BUS_ID`, torch index 6 is a Tesla P40.

## 0. One-screen summary

| Stage | Commits | Suite (server) | Review 1 | Review 2 | Headline receipt |
|---|---|---|---|---|---|
| S1 core switches (1.1/1.2/1.3/1.5/1.7/1.13) | `4be3b0a` | 164 passed | pass | — | defaults bit-identical vs `c1ca0b2` (logits/cached-logits/loss sha equal; only +11/+9 zero `prefix_logit_bias` scalars) |
| S2 gated GQA + per-layer widths (1.6/1.4) | `59a5e6d` | 187 passed | pass | — | Qwen3_5Attention transcription max abs diff 1.19e-07 at 0.8B geometry; reviewer ran the real HF class: bit-equal at small geometry, 1.19e-06 at 0.8B |
| S3 ledger/tokens/profiles (1.8/1.9/1.10/1.11) | `fbb7ae7`, `9ef3bc1`, `d5771b3` | 290 passed, 0 skipped | fix (3 blockers) | pass | ledger 248320→252416 / 163840→167936; band mask masked mass 0.0 with text logits identical; control ids now unmapped (248040..248069: aligned 30→4) |
| S4 safetensors IO + loaders + receipts (1.12/1.16/1.17) | `773c36f`, `605dbe9` | 343 passed | fix (4 blockers) | pass | R-Q Qwen3.5-0.8B cpu/fp32 max abs logit diff 3.19e-05, argmax 210/210; R-K3 layer 0 cpu/fp32 max abs diff 3.47e-08, rel-L2 9.7e-07, 0 missing / 0 unexpected keys |
| S5 GGUF export (1.14) | `8f9e2f9`, `c474128`, `f86fe45` | 376 passed | fix (2 blockers) | pass | qwen35 GGUF (320 tensors, sha `fcb61be2…`) served by llama.cpp b10068: 3/3 continuations byte-identical to HF bf16, 48/48 greedy tokens, max top-1 prob gap 0.0292 |

Test count ladder: 164 → 187 → 290 → 343 → 376 on the server; local workstation (Python 3.14, no torch) 201 passed / 78 skipped at HEAD.
`tests/test_forbidden_dependencies.py` green at every stage; no CLIP/SigLIP/QFormer/Whisper/ImageBind/VisionEncoder strings under
`src/chimera_omni`; only secret-shaped line in the tree is `data/postgres.py:99 os.environ.get("CHIMERA_PGPASSWORD", "")`.

Commit ladder (`git log --oneline c1ca0b2..claude/omni-import`):
```
f86fe45 omni-import S5 fix: the receipts re-run at c474128, with the dtype the disk actually holds
c474128 omni-import S5 fix: the nested local attention CSA/HCA hides, and a dtype that reads the disk
8f9e2f9 omni-import S5: the GGUF export lane, and a qwen35 file llama.cpp answers identically
605dbe9 omni-import S4 fix: refuse the switches the receipts ride on, unbreak the freeze hash
773c36f omni-import S4: safetensors IO, the Qwen3.5 and K3 exact-import loaders, and the two receipts
d5771b3 omni-import S3 fix: rope_type from the layout the theta came from, control ids unmapped, segment width from the spec
9ef3bc1 omni-import S3: read the real Qwen rope table, prove the profiles against the checkpoints
fbb7ae7 omni-import S3: fixed-band ledger, tokenizer/bridge, native volume type, import profiles
59a5e6d omni-import S2: gated-GQA block kind and per-layer intermediate sizes
4be3b0a omni-import S1: core import switches (KDA grouped keys, decay/gate modes, SiTU inf, AttnRes identity, tied embeddings)
```

---

## 1. S1 — core import switches (commit `4be3b0a`)

**What landed.** `config.py`: `attnres_init`, `kda_num_heads`/`kda_num_key_heads`, `kda_decay_kind`, `kda_output_gate`,
`kda_qk_norm_kind`, `kda_qk_norm_eps`, `kda_conv_bias`, `tie_word_embeddings`; SiTU betas accept `+inf`; `to_manifest` emits
every dataclass field (inf serialised as `"inf"`); new `ChimeraConfig.from_manifest`. `modeling/layers.py`: KDA grouped keys
(q/k sized by key heads, `repeat_interleave` to value heads in HF order), fla-style l2 norm, Qwen GDN decay
`-exp(A_log)*softplus(a+dt_bias)`, silu output gate, conv bias switch; `DepthAttentionResidual.prefix_logit_bias` (0.0 = today,
40.0 = identity); `FeedForward._activate` treats infinite beta as no cap. `modeling/chimera.py:163` ties `lm_head` to `embed`.
`architecture_manifest.json` regenerated (purely additive, 14 keys).

**Files.** `src/chimera_omni/config.py`, `src/chimera_omni/modeling/layers.py`, `src/chimera_omni/modeling/chimera.py`,
`architecture_manifest.json`, `scripts/behaviour_receipt_s1.py`, six new test files.

**Tests added (46).** `test_config_import_switches.py` (20), `test_kda_grouped_keys_torch_optional.py` (4),
`test_kda_decay_and_gate_modes_torch_optional.py` (7), `test_situ_inf_torch_optional.py` (4),
`test_attnres_identity_torch_optional.py` (7), `test_tied_embeddings_torch_optional.py` (4).

**Receipts.**
- `receipt_s1_before.json` (tree `c1ca0b2`): k3_aligned logits sha `e1a6bb2d…`, cached `f1c71fbc…`, loss 9.827261924743652, params 2797324; small logits `75dc7855…`, cached `666a6c7f…`, loss 9.858162879943848, params 2588164.
- `receipt_s1_after.json` (tree `4be3b0a`): identical logits/cached/loss hashes; params 2797335 / 2588173.
- `receipt_s1_compare.json`: `behaviour_preserved: true`, `changed_shared_state_dict_entries: []`, `removed_state_dict_entries: []`, added = the 11/9 zero `*.prefix_logit_bias` scalars only.
- `pytest_s1_full.log` "164 passed in 9.54s"; `pytest_s1_new_tests.log` "46 passed in 2.66s".

**Review 1: pass.** Reviewer re-ran the suite from a fresh archive (byte-identity of the tested tree proved by sha256 vs `git show`),
checked the math against the real `modeling_qwen3_5.py` (norm-before-gate at lines 193-202; GDN decay at 518; repeat_interleave
at 519-521) and `fla/modules/l2norm.py`. Non-blocking notes: (A) `from_manifest` silently defaults optional fields absent from a
manifest; (B) tied `lm_head` still allocates a full `nn.Linear` before discarding it (5.2 GB fp32 at 27B); (C) identity-AttnRes
exactness is a ratio bound (~7e9), not the spec's absolute 1e9; (D) bf16 rounding differs from Qwen's fused RMSNormGated by ~1 ulp
per KDA layer; (E) `rsqrt(sum+eps)` vs fla's `1/sqrt` may differ by 1 ulp; (F) pre-4be3b0a `torch.save` artifacts now fail strict
`load_state_dict` (missing `prefix_logit_bias`); (G) receipts were written to `/home/cereal/omni-campaign/chimera-import-s1/`, outside the sync dir.

**Gaps still open from S1.**
- `situ(inf,inf)` is NOT bit-equal to `F.silu` (maxdiff 4.77e-07 fp32); a Qwen import must pick one activation path for its exact receipt (S4 picked `activation_kind='silu'`).
- `attnres_init='identity'` makes each AttnRes the identity on `prefix_sum` but the trunk push/reset rule (`chimera.py:514-525`) still drops the pre-push accumulation; only `residual_mode='none'` is a plain residual, which is what the Qwen profile uses. Needs a Cyril ruling if identity is meant to reproduce a Qwen residual stream.
- `kda_qk_norm_kind` default stays `torch_normalize` (spec Q2 flip deferred).

---

## 2. S2 — gated GQA and per-layer intermediate sizes (commit `59a5e6d`)

**What landed.** `BlockKind` gains `gated_gqa`; `gqa_rope_dim=0`, `gqa_rope_theta=1e7`, `gqa_qk_norm=True`, `gqa_output_gate=True`;
`layers.py` `rotate_half` (non-interleaved) and `GatedGQAAttention` (separate `q_proj`/`gate_proj` = de-interleaved Qwen fused
matrix, multiplicative q/k RMSNorm storing 1+w, partial RoPE over leading `gqa_rope_dim` channels, positions = valid-token ranks,
`[B,T,T]` mask API, sigmoid output gate); `ChimeraBlock._masked_attention` covers the cache path and the bidirectional media
refinement branch. `intermediate_sizes: tuple[int,...] = ()` with `resolved_intermediate_size(layer)`.

**Files.** `config.py`, `modeling/layers.py`, `architecture.py`, `architecture_manifest.json`, `scripts/behaviour_receipt_s2.py`,
`tests/test_gated_gqa_torch_optional.py` (9), `tests/test_per_layer_intermediate_sizes_torch_optional.py` (4),
`tests/test_config_import_switches.py` (+2 tests, +8 rejection cases).

**Receipts.**
- `receipt_s2.json`: Qwen transcription max_abs_diff 1.1920928955078125e-07, max_rel 0.0343, expected rms 0.0856 (H=1024, 8q/2kv, head_dim 256, 64 rotary dims, theta 1e7); `gqa_rope_dim=0` + no norm + no gate == `LocalCausalAttention(window=0)` bit_equal true (max 0.0) at num_kv_heads 2 and 8; uniform `intermediate_sizes` list == scalar bit_equal true; per-layer params 2797335 → 2649879 for widths [384,192,384,192,384]; RoPE global-shift max_abs_diff 1.19e-07 / 1.04e-07 / 2.83e-07 / 2.77e-06 / 1.87e-04 at shifts 1/16/256/4096/262144.
- `compare_s1_vs_s2.json`: `behaviour_preserved: true`, parameter delta 0, logits sha unchanged from S1 (`e1a6bb2d…`, `75dc7855…`).
- `receipt_s1_style_after_s2.json`: full per-tensor sha table from the S2 tree.
- `pytest_s2_full.log` "187 passed in 12.22s"; `pytest_s2_new_tests.log` "43 passed in 2.43s".

**Review 1: pass.** Reviewer independently re-derived default preservation against the S1 tree (logits/chunked-cache/state shapes/params
all equal) and — new evidence — ran transformers' real `Qwen3_5Attention` + `Qwen3_5TextRotaryEmbedding` vs `GatedGQAAttention`:
bit_equal TRUE (max 0.0) at H=256/head_dim 64/rope 16; max_abs 1.19e-06 (mean 1.8e-07, rms 1.098) at the 0.8B geometry — pure fp32
GEMM reassociation from the split `[q|gate]` matrix. Adversarial probes (batch independence 0.0; left padding 5.96e-08; padded chunked
cached decoding 7.15e-07; full-width rope shift 7.45e-08) held. Non-blocking: (A) narrative said "172 before" — real S1 count was 164;
(B) `gqa_output_gate` is a bool and cannot express the 27B's `output_gate_type: "swish"` (see spec Q1); (C) `gqa_rope_theta` alone
cannot express yarn/linear rope — later closed in S3 by refusing scaled rope; (D) exactness bound is ~1e-6 *relative*; (E) invalid
rows are non-zero when the module is called directly (ChimeraBlock zeroes them); (F) mask-diagonal validity fallback is inferred, not
enforced; (G) `gqa_rope_theta=inf` serialises as `Infinity`; (H) manifest `position_policy` text still says NoPE while `sequence_mixing` lists `gated_gqa`.

**Gaps still open from S2.**
- RoPE cos/sin table is fp32; ~2e-4 absolute error at 262,144 context. Long-context exact receipts need fp64 angles or HF's cached table.
- 3-axis mrope media positions are not representable (text collapses to 1-D RoPE; Chimera is encoder-free, so not imported).
- Export re-interleave direction was later implemented in S5 (`q_gate_deinterleave_permutation` proved inverse both ways).
- `intermediate_sizes` non-uniform children are not blocked from claiming qwen35 exportability at config level (S5 export reads one width).
- Qwen transcription is duplicated between the test and `behaviour_receipt_s2.py`.

---

## 3. S3 — fixed-band ledger, tokenizer/bridge, volume type, import profiles (commits `fbb7ae7`, `9ef3bc1`, `d5771b3`)

**What landed.** `src/chimera_omni/importing/{profiles,tokenizer,bridge}.py`, `native_segments.py` (volume modality, feature_dim 4096),
fixed-band ledger (Qwen 248320 → 252416, K3 163840 → 167936; 4096-row band of control/media/grounding/sync/tool ids), band logit mask,
`JsonBpeTokenizer` (Qwen tokenizer.json) and `TikTokenBpeTokenizer` (K3 tiktoken.model), `token_alignment` + `embedding_bridge`,
profiles `qwen35_0_8b` / `kimi_k3` / `chimera_omni_27b` with geometry read from the checkpoints' own `config.json`.

**Review 1: fix (3 blockers), all fixed in `d5771b3` and re-reviewed: pass.**
1. `profiles.py` `_qwen_rope`: flat-layout `rope_scaling={'rope_type':'yarn'}` bypassed the scaled-rope guard. Now `rope_type` comes from the same layout as the theta (flat, `rope_scaling['rope_type']`, legacy `type`), and a config carrying both `rope_parameters` and a non-empty `rope_scaling` is refused. Verified none of the three Qwen checkpoints carries both (`text rope keys ['rope_parameters']`, `rope_scaling: None`).
2. `tokenizer.py`/`bridge.py`: control ids were bridged by the byte spelling of their name (`token_bytes(248053) = b'<|vision_start|>'` → K3 pieces `[27,91,7386,9485,91,29]`). Now both readers carry a declared special-id set (Qwen: `added_tokens` table, 26 ids 248044..248069; K3: 256 reserved ids); `token_bytes` returns `b''`; `token_alignment` reports `special_tokens`. Window 248040..248069: aligned 30/unmapped 0 → aligned 4/unmapped 26. Note: the review's stated root cause (the `KeyError` fallback) was wrong — ASCII marker names decode through the byte alphabet; the declared set is the load-bearing fix and a test pins both facts.
3. `native_segments.build_native_segment_packet` ignored `spec.feature_dim` (hard 3072); volume packet now declares 4096 = what `volume_to_native_patches` emits; six 3072 modalities unchanged.
Also closed: on-disk anti-decay test for `QWEN35_FAMILY_DEFAULTS` across all three Qwen checkpoints, folded duplicate volume branch in `canonical_positions`, `source_config_scope` stamped beside `source_config_sha256` (Qwen hashes whole config.json; K3 hashes `text_config` only).

**Tests added.** `test_ledger_fixed_band.py` (21), `test_ledger_band_logit_mask_torch_optional.py` (4), `test_trunk_feature_dim_torch_optional.py` (4),
`test_native_volume_torch_optional.py` (12), `test_tokenizer_and_bridge.py` (16), `test_import_profiles.py` (30), `test_import_profiles_on_disk.py` (9).
Suite 187 → 278 → 290 (0 skipped on the server because the Qwen/K3 assets are present).

**Receipts.**
- `receipt_s3.json` (sha256 `918ab9b018f320dd9b3a9322656362264e03c9d56da9e9fd573cfd094e28cff4`, 8701 B; server copy `/home/cereal/omni-campaign/chimera-omni-import/receipts/receipt_s3.json`): ledger 248320→252416 (54320 rows the proportional policy would relabel) / 163840→167936 (35840); band softmax mass 1.511770248413086 → 0.0 with text logits bit-identical; volume `[1,1,33,35,37]` patch 16 → 27 tokens × 4096, max_abs 0.0; typed flow AV bit-identical, volume row zero; Qwen tok 248070 ids in a 248320 range, K3 163840, both round-trip on the fixture texts; bridge aligned 43 / single 42 / multi 1 / unmapped 0 / special 0 / mean pieces 1.0233, rows written 43, zero rows 248277; geometry read from `/home/cereal/models/Qwen3.5-0.8B/config.json` (H 1024, L 24, vocab 248320, rope 64), `/mnt/ai_data/models/moonshotai/Kimi-K3/config.json` (H 7168, L 93, vocab 163840), `/home/cereal/models/Qwen3.8-27B-Uncensored/config.json` (H 5120, L 64, gate `sigmoid_assumed_declared_swish`).
- `receipt_s3_before_blocker_fix.json` (sha `dea90859…`, 8536 B) — the pre-fix receipt; field-by-field diff to the new one is exactly four added keys + the label, every number unchanged.
- `receipt_s3_transcribed_geometry_backup.json` — the earlier pre-on-disk-fix receipt.
- `pytest_s3fix_full.log` "290 passed in 19.20s"; `pytest_s3fix_s3_files.log` "96 passed in 7.19s"; `pytest_s3fix_blocker_tests.log` "12 passed in 3.65s"; `probe_blockers_s3fix.{py,log}` (pre/post numbers for all three blockers on the real assets).
- Reports: `S3_LEDGER_TOKENS_2026-09-01.md`, `S3_BLOCKER_FIXES_2026-09-02.md`.

**Review 2: pass.** Reviewer reconstructed the pre-fix tree and ran the new tests against it — all six blocker tests FAIL on `9ef3bc1` and pass on `d5771b3`. Non-blocking: `_json_added_token_ids` returns an empty set when `added_tokens` is absent (silent revert to blocker-2 behaviour); `_added_token_ids` swallows all exceptions; tokenizer.json is parsed twice.

**Gaps still open from S3.**
- `token_alignment(full_vocab=True)` over the 248,320-row Qwen vocabulary still not run (now safe to run).
- Qwen's NFC normalizer: `decode(encode(t)) != t` for combining-mark text; receipt round-trip booleans are true only for the four fixture texts. Should become a rate.
- `embedding_bridge` `source_row_rms` reads a zero table in the receipt (meaningless statistic).
- `from_qwen` treats every `added_tokens` id as no-byte, not only `"special": true` ones (fail-safe direction, a choice).
- Spec Q1 unresolved: 27B declares `text_config.output_gate_type='swish'`; transformers 5.12.1 applies `sigmoid(gate)` unconditionally (`modeling_qwen3_5.py:714`); `chimera_omni_27b_profile` raises unless `allow_unverified_output_gate=True`.

---

## 4. S4 — safetensors IO, the Qwen3.5 and K3 exact-import loaders, the two receipts (commits `773c36f`, `605dbe9`)

**What landed.** `importing/safetensors_io.py` (sharded writer/reader, `save_checkpoint`/`load_chimera_checkpoint`, tie detection,
`state_sha256`), `importing/qwen35.py` (`import_qwen35_state`, `_one_plus` folding on norm/q_norm/k_norm, fused `[q|gate]` split,
refuses non-Qwen switches), `importing/kimi_k3.py` (`import_k3_layer_state`, `_check_k3_switches` keyed on block kind, A_log trim
128→96, `load_k3_moe_tensors` written but never executed), `scripts/import_receipt_qwen35.py`, `scripts/import_receipt_k3_layer.py`,
`proof/h3_hybrid_av_flow.py` freeze-hash fix, `KDAState` layout doc (`[B,H,K,V]` vs fla's `[B,H,V,K]`).

**Review 1: fix (4 blockers), all fixed in `605dbe9` and re-reviewed: pass.**
1. `_check_k3_switches` never checked `mla_latent_norm_eps` or `kda_qk_norm_kind`; a gated_mla import with eps resolving to 1e-5 instead of `KimiRMSNorm`'s default 1e-6 returned a receipt that read as exact. Now keyed on block kind: `kda` additionally requires `kda_qk_norm_kind='fla'`; `gated_mla` requires `resolved_mla_latent_norm_eps() == 1e-6` (verified at `modeling_kimi_linear.py:226-230/368/383`).
2. `import_qwen35_state` did not validate `kda_query_scale='inv_sqrt_key_dim'` / `kda_qk_norm_kind='fla'`, the two switches the fp32 receipt rides on. Now refused (`qwen35.py:96-107`).
3. `hash_state_dict` did `tensor.view(uint8)` on 0-dim tensors → `RuntimeError` on every model since S1 added the scalar `prefix_logit_bias`; the joint-AV proof driver died at its first freeze hash. Fixed with `reshape(-1)` (matches `safetensors_io._as_view`) and proved by running the driver end to end (`--steps 2`: freeze hashes `6f4bf8df…` / `74b4820b…` written; `passed: false` is expected for a 2-step smoke).
4. `state_sha256` docstring falsely claimed the same convention as `hash_state_dict` (safetensors `'F32'` vs `str(torch.dtype)` tokens). Claim deleted rather than unified — unifying would invalidate every recorded freeze hash in the campaign; a test locks that the two digests are stable and unequal.
Also: `restore_fast_paths` makes the R-Q transformers patch reversible; receipt records `round_trip_column`; checkpoint tie key is `(data_ptr, storage_offset, shape, stride, dtype)`.

**Tests added.** `test_safetensors_io.py`, `test_checkpoint_round_trip_torch_optional.py`, `test_import_qwen35_synthetic_torch_optional.py`,
`test_import_k3_synthetic_torch_optional.py` (+3 in the fix), `test_state_hash_conventions_torch_optional.py` (5, new),
`test_import_receipt_script_fast_paths_torch_optional.py` (2, new). Suite 290 → 332 → 343.

**Receipts (numbers).**
- R-Q `receipts/qwen35_0p8b_import_receipt.json` (Qwen3.5-0.8B, 8 prompts, 210 argmax positions, `--layers 4`): cpu/float32 max_abs_logit_diff **3.191828727722168e-05**, argmax_agreement 1.0, layer-0..3 hidden max 7.748603820800781e-07; cuda:0/bfloat16 (GPU6, fla fast paths on) max 0.5, argmax 0.9952 (209/210), hidden 0.015625; round trip (bf16 column) `state_sha256_before == after == 5d81a211b450c8f9af896ac03961897202dcd97abdfb8110650d5e4078cf2906`, `logits_bit_identical true`, `tied_weights {lm_head.weight: embed.weight}`, 1 shard, 30.49 s; parameter counts trunk_imported 752,427,584 / band 4,194,304 / model_total 770,441,792 / skipped_mtp 20,452,864 / skipped_media 100,592,896.
- R-K3 `receipts/k3_layer0_import_receipt.json` (Kimi K3 layer 0 = KDA + dense MLP, 23 tensors / 2,341,213,184 B read from the staged shard at 272 MB/s, batch 1 × 64, seed 0): cpu/float32 layer_output max_abs_diff **3.4691765904426575e-08**, mean 2.74e-09, relative_l2 9.716125855522246e-07, mixer_output max 3.54e-08 / rel-L2 2.83e-06, recurrent state as-is rel-L2 1.4026e-05; cuda:0/bfloat16 layer max 4.8828125e-04 / rel-L2 6.0975e-03, mixer 2.44e-04 / 8.82e-03, recurrent state matches only transposed (`reference_transposed` max 1.478e-03; as-is rel-L2 1.4077); `reference_load_state_dict missing_keys [] unexpected_keys []`; `source_adjustments: ['A_log trimmed from 128 to 96 entries (checkpoint zero-pads to head_dim)']`. Reference = the checkpoint's own `KimiDecoderLayer` built from `modeling_kimi_linear.py`.
- `pytest_s4fix_full_605dbe9.log` "343 passed in 20.55s"; `pytest_s4fix_new_tests_605dbe9.log` "53 passed in 3.23s"; `qwen_receipt_s4fix.log`, `k3_receipt_s4fix.log`, `k3_receipt_cpu_s4fix.log`; server `/home/cereal/omni-campaign/import-lane/s4-fix/h3_smoke/report.json`.
- Reports: `S4_IO_AND_RECEIPTS_2026-09-02.md`, `S4_BLOCKER_FIXES_2026-09-02.md`.

**Corrections to earlier S4 narrative (found by re-running).** The "~2900x faster than the fla reference" figure was a cold Triton
autotune artefact (warm: reference 1.478 s vs Chimera 0.108 s). `chimera_state_sha256` in the receipt is a per-run value (16
parameters — `native_media_bridge.*`, `flow_head`, `grounding_head`, … — are randomly initialised and unseeded), so it moved
`1877a83e…` → `5d81a211…` between runs with byte-identical imported tensors; what the round trip proves is invariance under write→read.

**Review 2: pass.** Non-blocking: (a) the gated_mla eps guard tests the *resolved* value, so `rms_norm_eps=1e-6, mla_latent_norm_eps=0.0` passes while norm1/norm2/o_norm silently run at 1e-6 instead of K3's 1e-5 — require the field be set; (b) `kda_qk_norm_eps` unguarded on both loaders; (c) both digests share the receipt key name `state_sha256`; (d) `fast_paths_restored_from_earlier_column` never true in a real run (only unit-tested); (f) S4 report line references drifted (kimi_k3.py:123 → :130-168, qwen35.py:81 → :96-107).

**Gaps still open from S4.**
- **MLA path never measured.** `--layer 3` against the real `KimiMLAAttention` was not run (its shard was not staged). `mla_latent_norm_eps=1e-6` is enforced but is a source-reading claim.
- `kda_qk_norm_kind='fla'` requirement on the K3 loader is likewise source-reading (`modeling_kimi_linear.py:610-627` → `chunk_kda(use_qk_l2norm_in_kernel=True)`); no run with `torch_normalize` on real weights to size the 1e-12 vs 1e-6 difference.
- Qwen round trip measured on the bf16 column only (labelled, not fixed).
- `load_k3_moe_tensors` (896 MXFP4 experts, layer 1 in shard 2) has never executed against a routed-expert tensor.
- Qwen3.5-4B and Qwen3.8-27B-Uncensored receipts not run; 27B still blocks on spec Q1 (`output_gate_type='swish'`).
- Proof-driver evidence is a 2-step smoke, not the 4000-step joint-AV proof.
- Recurrent-state layout `[B,H,K,V]` vs fla `[B,H,V,K]` is documented on `KDAState` only; any cache interop or Triton swap must transpose.

---

## 5. S5 — GGUF export lane (commits `8f9e2f9`, `c474128`, `f86fe45`)

**What landed.** `src/chimera_omni/export/gguf_names.py` (`chimera_parameter_names`, `ExportPlan.covers()`, qwen35 and kimi-k3
name maps, `reorder_v_permutation`, `q_gate_deinterleave_permutation`, `require_qwen35_exportable` / `require_kimi_k3_exportable`),
`src/chimera_omni/export/gguf_writer.py` (`build_export_tensors`, `write_chimera_gguf`, `_cast`, `float16_narrowing_stats`,
`_gguf_package_version`), `scripts/export_gguf_qwen35.py` (`--chimera-commit`), `scripts/gguf_roundtrip_qwen35.py`,
`safetensors_io.checkpoint_dtypes()` (header-only shard dtype census), `tests/test_gguf_name_maps.py` (45 incl. every-block-kind
enumeration test), `tests/test_safetensors_io.py` (+1).

**Review 1: fix (2 blockers), fixed in `c474128`, receipts re-run in `f86fe45`, re-reviewed: pass.**
1. `gguf_names.py:186` enumerated a csa/hca mixer four names short (the nested `mixer.local.{q_proj,k_proj,v_proj,out}` of `SparseLatentAttention`), so `ExportPlan.covers()` — the guarantee against dropped parameters — was defeated for those block kinds. Fixed via a shared `_local_attention_names(prefix)`; `test_the_parameter_enumeration_matches_a_real_state_dict` now binds the enumeration to a real `state_dict()` for `every_block_kind_config` (kda+local+csa+hca+gated_mla+gated_gqa, MTP head, conv biases). Its first run caught a second real bug in the test helper (`blocks.4.mixer.out.weight (64,64) vs (64,48)` — the KDA/MLA shape coincidence the K3-shaped config was hiding).
2. `export_gguf_qwen35.py` recorded `dtype_on_disk` = the *load* dtype (`torch.float32`) while the shards are BF16. Now `source_checkpoint.dtypes_on_disk {"BF16": 390}` beside `load_dtype "torch.float32"`.
Also made measured rather than assumed: `float16_cast` stats, `gguf_version` from `importlib.metadata` (0.19.0), `chimera_commit` from `--chimera-commit`, bos-absence note.

**Receipts (numbers).**
- `receipts/qwen35_0p8b_gguf_export.json`: arch `qwen35`, 320 tensors (F16 187 / F32 133), 17 omitted (13 media + 3 aux + tied `lm_head`), file `qwen35-0p8b-chimera-f16-s5fix.gguf` 1,525,255,104 B sha256 `fcb61be2db105b7f89ece090173dbbacdf75a1dbc4ffaaf02157b23d6d9657ef`; source state sha `03d88620140214a9fdd937acb8cfc1e38a619e1d94c3f01602af9bd26f8795e9`; 252,416 tokens (last 4,096 = CONTROL band `<|co:control:0|>` … `<|co:tool:383|>`), 247,587 merges, `tokenizer.ggml.pre=qwen2`; KV: block_count 24, embedding 1024, ffn 3584, heads 8/2, key/value_length 256, rope base 1e7, rope dims 64, sections [11,11,10,0], ssm conv 4 / state 128 / groups 16 / inner 2048, full_attention_interval 4; `float16_cast {tensors_narrowed 187, elements_narrowed 756,088,832, flushed_to_zero 15,889, subnormal 3,023,581}` (worst: `token_embd.weight` 309 zeroed / 650,896 subnormal; `blk.0.attn_qkv.weight` 14,345 zeroed); `gguf_version 0.19.0`; `chimera_commit c474128771da60691fcf1fb2c9de67780e0c8a35`. Tensor payload (name+type+shape+bytes over all 320) hashes identically on the S5 and S5-fix files: `dfa922a4a6a56c57b40b2758bc3557c6d8928d3645b0f06b05ce5ebc3d9632cb` (the +32 B whole-file delta is the commit string replacing "unknown").
- `receipts/qwen35_0p8b_gguf_roundtrip.json` (llama.cpp `ghcr.io/ggml-org/llama.cpp:server-cuda` b10068 `571d0d540` on GPU6, `-c 2048 -ngl 99`, vs HF bf16 reference, same token-id prompts both sides): prompts 3, `tokenizer_agreement 1.0`, `identical_text 3`, `mean_token_agreement 1.0`, `mean_matching_prefix 48.0`, `top5_id_set_agrees 3`, `top5_ids_agree_in_order 1`, `max_top1_probability_gap 0.02915334106749279` — every number reproduced the original S5 run exactly.
- `pytest_s5fix_full.log` "376 passed, 1 warning in 16.28s" (warning = pre-existing requires_grad-scalar UserWarning); `pytest_s5fix_gguf.log` "45 passed in 2.49s"; `export_s5fix.log`, `roundtrip_s5fix.log`, `llama_server_s5fix.log`.
- Reports: `S5_GGUF_EXPORT_2026-09-02.md`, `S5_BLOCKER_FIXES_2026-09-02.md`.

**Review 2: pass.** Verified receipts are byte-identical to the server copies (export sha `08170840…`, roundtrip `97de0e30…`), probe container removed, GPU6 at 5 MiB. Non-blocking: `float16_cast.note` ("bf16 widens exactly, so these are the only values changed") is true for this artifact but will be wrong for an fp32-native source — replace with a measured `values_changed`; stats allocate ~1-1.3 GB transient on `token_embd`; `_gguf_package_version` reports the distribution, not the imported module; `GGUF_PACKAGE_VERSION_VERIFIED` and measured version are never compared.

**Gaps still open from S5.**
- **Peak RAM ~3.5x fp32 parameter bytes** (`build_export_tensors` materialises all fp32 numpy + `GGUFWriter` holds f16 copies with `use_temp_file=False`). OK at 0.8B (~7 GB); impossible at 4B (~56 GB) and 27B. Streaming rewrite required before those exports.
- kimi-k3 lane is plan-and-tests only: no K3-derived child exported, no llama.cpp load of a kimi-k3 file, MoE half of 1.14(b) (`ffn_gate_inp`, `exp_probs_b`, `ffn_*_exps`, shared experts) absent; KV writes `expert_count 0` / `leading_dense_block_count == block_count` (spec Q8 assumption).
- Companion `chimera-media` GGUF (D5b) not written; the 17 omitted parameters are only enumerated with reasons.
- `tokenizer.ggml.pre='qwen2'` measured on three ASCII prompts; Qwen3.5 splits on `[\p{L}\p{M}]+` vs llama.cpp qwen2's `\p{L}+` — first divergence would be combining-mark text.
- f16 cast has counts but no error metric (no max/mean relative error, no logit-impact analysis beyond the 3-prompt round trip).
- Six llama.cpp "special false but token is control" warnings (`<|fim_prefix|>` etc.) unchecked against a stock conversion.
- `context_length` written as 262,144; probe ran at `-c 2048` with 48-token continuations.
- Only the 0.8B exported; r=3 `reorder_v` for the 27B covered by synthetic unit tests only.
- Spec 1.15 (H3 catalog ruling) untouched.

---

## 6. Consolidated open items, ordered by what they block

1. **Spec Q1 — 27B `output_gate_type='swish'`.** Blocks every Qwen3.8-27B-Uncensored exactness claim (profile, import receipt, export). Needs a ruling (does transformers ≥5.15 mean `silu(gate)`?) and then a gate-kind switch on `GatedGQAAttention` (S2 layer) — `gqa_output_gate` is a bool today.
2. **K3 MLA path never measured** (`--layer 3` vs real `KimiMLAAttention`; requires staging shard for layer 3). Precondition enforced at 1e-6 but not proven numerically.
3. **K3 MoE import** (`load_k3_moe_tensors`, layer 1, 896 MXFP4 experts in shard 2) never executed; kimi-k3 GGUF MoE tensor set unwritten.
4. **GGUF streaming rewrite** (peak RAM) before the 4B / 27B exports.
5. **Qwen3.5-4B receipt** (R-Q at 4B) — not run; would exercise `kda_num_key_heads` grouping beyond r=4.
6. `token_alignment(full_vocab=True)` over the 248,320-row Qwen vocabulary; NFC round-trip as a rate; `source_row_rms` cleanup.
7. AttnRes identity vs the trunk push/reset rule (S1 gap #2) — Cyril ruling if identity is meant to be a Qwen residual stream.
8. `chimera-media` companion GGUF (D5b), spec 1.15 H3 catalog ruling, `chimera`-arch fallback (Q8).
9. Hardening list from reviews: `from_manifest` should require every known key; tied `lm_head` should skip the throwaway `nn.Linear` allocation (5.2 GB at 27B); gated_mla eps guard should require the field be *set*; guard `kda_qk_norm_eps`; rename one of the two `state_sha256` receipt keys; `_json_added_token_ids` empty-set fallback should fail loud; manifest `position_policy` text vs `gated_gqa`; `math.isfinite` on raw float config fields.

---

## 7. Exact commands to reproduce each receipt

Sync the branch to the server first (from bash on the workstation):
```
git -C "/c/Users/cereal/Documents/ai training/chimera-omni" archive --format=tar -o "$TEMP/omni-import.tar" claude/omni-import
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --upload "$TEMP/omni-import.tar" "/home/cereal/omni-campaign/chimera-omni-import/omni-import.tar"
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script 'mkdir -p /home/cereal/omni-campaign/chimera-omni-import/src && cd /home/cereal/omni-campaign/chimera-omni-import && rm -rf src/* && tar -xf omni-import.tar -C src && cd src && timeout 900 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -x -q tests 2>&1 | tail -n 30'
```
Below, `PY=/home/cereal/model-surgery-lab/venv/bin/python`, cwd = the extracted tree, `PYTHONPATH=src`. On `[SSH BUSY]` sleep 20 s and retry (≤10×); keep `--timeout` ≤ 280 and wrap remote commands in `timeout` (launch the long receipts with `nohup … &` and poll their log). Never delete on the server; never walk `/mnt/ai_data` recursively. Set `MSYS_NO_PATHCONV=1` in the bash wrapper so a leading `VAR=/abs/path` argument is not rewritten.

**Full suite (every stage)**
```
timeout 900 $PY -m pytest -o addopts="" -q -p no:warnings -p no:cacheprovider tests
```

**S1 defaults receipt** (`scripts/behaviour_receipt_s1.py`; CPU fp32; run once per tree — `git archive c1ca0b2` for BEFORE, the branch for AFTER)
```
CUDA_VISIBLE_DEVICES= PYTHONPATH=<before_tree>/src $PY scripts/behaviour_receipt_s1.py --out receipt_s1_before.json --label c1ca0b2
CUDA_VISIBLE_DEVICES= PYTHONPATH=<after_tree>/src  $PY scripts/behaviour_receipt_s1.py --out receipt_s1_after.json  --label <commit>
$PY scripts/behaviour_receipt_s1.py --compare receipt_s1_before.json receipt_s1_after.json --out receipt_s1_compare.json
```
Expect `behaviour_preserved: true`, logits sha `e1a6bb2d…` (k3_aligned) / `75dc7855…` (small), loss 9.827261924743652 / 9.858162879943848. (S2 re-used the same script against `receipt_s1_after.json` → `compare_s1_vs_s2.json`.)

**S2 receipt** (`scripts/behaviour_receipt_s2.py`, CPU)
```
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY scripts/behaviour_receipt_s2.py --out receipt_s2.json --label <commit>
```
Expect `qwen_attention_transcription.max_abs_diff 1.1920928955078125e-07`, local-attention equivalence `bit_equal true`.

**S3 receipt** (`scripts/behaviour_receipt_s3.py`, CPU, real assets)
```
CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY scripts/behaviour_receipt_s3.py --out receipts/receipt_s3.json --label s3_fix \
  --qwen-dir /home/cereal/models/Qwen3.5-0.8B --k3-dir /mnt/ai_data/models/moonshotai/Kimi-K3 \
  --qwen-27b-dir /home/cereal/models/Qwen3.8-27B-Uncensored
sha256sum receipts/receipt_s3.json   # 918ab9b018f320dd9b3a9322656362264e03c9d56da9e9fd573cfd094e28cff4
```
Blocker probe: `PYTHONPATH=src $PY probe_blockers_s3fix.py` (copy in the report dir).

**S4 R-Q Qwen3.5-0.8B import receipt** (`scripts/import_receipt_qwen35.py`; GPU6 bf16 column then CPU fp32; `--layers 4`, `--max-tokens 128`; round trip runs on the first column)
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src timeout 3000 $PY scripts/import_receipt_qwen35.py \
  --model-dir /home/cereal/models/Qwen3.5-0.8B --column cuda:0/bfloat16 --column cpu/float32 \
  --round-trip-dir /home/cereal/omni-campaign/import-lane/s4-fix/qwen35_0p8b_chimera_bf16 \
  --out receipts/qwen35_0p8b_import_receipt.json
```
Expect cpu/float32 `max_abs_logit_diff 3.191828727722168e-05`, argmax 210/210; round trip `state_sha256_equal true`, `logits_bit_identical true`. (The CUDA column reads as `cuda:0` inside the process because of `CUDA_VISIBLE_DEVICES=6`; the script's default columns are `cuda:6/bfloat16, cpu/float32` for a process that sees all GPUs.)

**S4 R-K3 layer-0 receipt** (`scripts/import_receipt_k3_layer.py`; staged shard under `/home/cereal/omni-campaign/import-lane/s4/k3_layer0_staged`, K3 modeling source at `/home/cereal/omni-campaign/k3/k3_src`)
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src timeout 3000 $PY scripts/import_receipt_k3_layer.py --layer 0 \
  --k3-dir /mnt/ai_data/models/moonshotai/Kimi-K3 --k3-src /home/cereal/omni-campaign/k3/k3_src \
  --stage-dir /home/cereal/omni-campaign/import-lane/s4/k3_layer0_staged \
  --column cuda:0/bfloat16 --column cpu/float32 --batch 1 --length 64 --seed 0 \
  --out receipts/k3_layer0_import_receipt.json
```
Expect cpu/float32 `layer_output.max_abs_diff 3.4691765904426575e-08`, `relative_l2 9.716e-07`, `missing_keys [] unexpected_keys []`, `A_log trimmed from 128 to 96`. The S4-fix run produced the CPU column separately (`k3_layer0_import_receipt_cpu.json`) and merged; a single invocation with both `--column` flags is equivalent. First-time Triton autotune of the fla reference takes ~5 min; warm it is ~1.5 s.

**S4 proof-driver smoke (blocker 3 evidence)**
```
timeout 260 env PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 $PY src/chimera_omni/proof/h3_hybrid_av_flow.py \
  --steps 2 --batch-size 2 --sample-steps 2 --eval-count 2 --output-dir /home/cereal/omni-campaign/import-lane/s4-fix/h3_smoke \
  --source-commit 605dbe96cf38134b2423840d3ea5d548c01c3f6e --source-tree-dirty false
```
Expect `report.json` with both freeze `state_sha256` values written and `passed: false` (2 steps cannot meet the learning gates; no crash is the claim). `--source-commit` must be the full 40-char hash.

**S5 GGUF export** (CPU; ~7 GB peak at 0.8B)
```
CUDA_VISIBLE_DEVICES= PYTHONPATH=src timeout 1200 $PY scripts/export_gguf_qwen35.py \
  --checkpoint /home/cereal/omni-campaign/import-lane/s4-fix/qwen35_0p8b_chimera_bf16 \
  --tokenizer-dir /home/cereal/models/Qwen3.5-0.8B \
  --out /home/cereal/omni-campaign/import-lane/s5/out/qwen35-0p8b-chimera-f16-s5fix.gguf \
  --file-type f16 --pre qwen2 --chimera-commit c474128771da60691fcf1fb2c9de67780e0c8a35 \
  --receipt receipts/qwen35_0p8b_gguf_export.json
```
Expect 320 tensors / 17 omitted, sha256 `fcb61be2db105b7f89ece090173dbbacdf75a1dbc4ffaaf02157b23d6d9657ef`, `dtypes_on_disk {"BF16": 390}`, `flushed_to_zero 15889`.

**S5 llama.cpp round trip** (fresh container on GPU6; stop and remove it afterwards — it is ours)
```
docker run -d --name omni_gguf_probe_fix --gpus device=GPU-f5a09535-fc85-a027-50f7-734b344747dd \
  -v /home/cereal/omni-campaign/import-lane/s5/out:/models:ro -p 127.0.0.1:28089:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda -m /models/qwen35-0p8b-chimera-f16-s5fix.gguf -c 2048 -ngl 99 --host 0.0.0.0 --port 8080
# wait for "model loaded" in `docker logs omni_gguf_probe_fix`; /props reports build_info b10068-571d0d540
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src timeout 900 $PY scripts/gguf_roundtrip_qwen35.py \
  --model-dir /home/cereal/models/Qwen3.5-0.8B --server http://127.0.0.1:28089 \
  --export-receipt receipts/qwen35_0p8b_gguf_export.json --n-predict 48 --device cuda:0 --dtype bfloat16 \
  --out receipts/qwen35_0p8b_gguf_roundtrip.json
docker stop omni_gguf_probe_fix && docker rm omni_gguf_probe_fix
```
Expect `identical_text 3`, `mean_matching_prefix 48.0`, `tokenizer_agreement 1.0`, `max_top1_probability_gap 0.02915334106749279`. (The original S5 run used `-p 127.0.0.1:28088` and `-np 1`; the fix run used 28089 without `-np`, so `n_slots = 4` — sequential greedy requests, comparison unaffected.)

---

## 8. Server state left behind (nothing pre-existing deleted or stopped)

- `/home/cereal/omni-campaign/chimera-omni-import/` — sanctioned sync dir: `src/`, `src_committed/`, `src_review_s4fix/`, `review_s1/`, `review-s3/{base,head,*.tar,probe*.json}`, `receipts/`, probe/treehash helpers, tars.
- `/home/cereal/omni-campaign/chimera-import-s1/` (S1 receipts + a pre-change tree, ~55 MB tar), `/home/cereal/omni-campaign/chimera-import-s2/`, `/home/cereal/omni-campaign/chimera-review-s2/` (reviewer probes).
- `/home/cereal/omni-campaign/import-lane/s4/` (original receipts, `k3_layer0_staged/` ~2.3 GB shard, `qwen35_0p8b_chimera_bf16/` 1.5 GB), `import-lane/s4-fix/` (re-run receipts, logs, `h3_smoke/` ~25 MB, second 1.5 GB bf16 checkpoint).
- `/home/cereal/omni-campaign/import-lane/s5/` (~3.2 GB: tars, extracted trees, two 1.53 GB GGUFs, receipts, logs).
- `/home/cereal/omni-campaign/k3/k3_src` — K3 modeling source used by the K3 receipt.
- Probe containers `omni_gguf_probe` / `omni_gguf_probe_fix` were stopped and removed; GPU6 at 5 MiB after each stage. GPU0's 21.5 GB llama-server was never touched.

---

## 9. What Codex should review

1. **The math in S1/S2 that every later stage rides on** (reviewed by reading, then proven numerically only through the 0.8B and K3-layer-0 receipts): grouped-key `repeat_interleave` ordering (`layers.py:291-294` vs `modeling_qwen3_5.py:519-521`), `(1+w)` folding set in `importing/qwen35.py:58` `_one_plus` (norm/norm1/norm2/q_norm/k_norm; `linear_attn.norm` left plain), `rotate_half` non-interleaved pairing, GDN decay `-exp(A_log)*softplus(a+dt_bias)`, fla-style l2norm with 1e-6 vs `F.normalize` 1e-12, AttnRes identity bias placement (`layers.py:590`).
2. **The K3 switch preconditions that are enforced but unmeasured**: `mla_latent_norm_eps=1e-6` (`modeling_kimi_linear.py:226-230/368/383`), `kda_qk_norm_kind='fla'` (`:610-627`), and the resolved-vs-set eps bypass (S4 review 2 item a). Best closed by staging layer 3's shard and running `import_receipt_k3_layer.py --layer 3`.
3. **Spec Q1** — the `output_gate_type='swish'` reading for Qwen3.8-27B-Uncensored; what transformers ≥5.15 does with it; whether a gate-kind switch belongs on `GatedGQAAttention`.
4. **The receipt scripts themselves** for tautology: `behaviour_receipt_s3.py:156-165` (typed-flow expected recomputed inline — reviewer confirmed non-tautological), `import_receipt_qwen35.py` `disable_fast_paths`/`restore_fast_paths` ordering, `import_receipt_k3_layer.py` `transpose_state_layout` handling on CPU vs GPU.
5. **GGUF export correctness beyond the 0.8B**: `reorder_v_permutation` at r=3 (27B), `float16_cast.note` premise for fp32-native sources, `tokenizer.ggml.pre=qwen2` vs Qwen3.5's `[\p{L}\p{M}]+` pre-tokenizer, the six CONTROL-type warnings, kimi-k3 KV `expert_count 0` assumption (Q8).
6. **The two incomparable `state_sha256` digests** (`safetensors_io.state_sha256` vs `proof.hash_state_dict`) — confirm that leaving them unequal (to preserve recorded freeze hashes) is the right call, and whether one receipt key should be renamed.
7. **Tokenizer/bridge policy choices** flagged as choices not derivations: all `added_tokens` ids treated as no-byte; `_json_added_token_ids` empty-set fallback; NFC round-trip reported as a fixture boolean.
8. **Rule compliance spot-checks**: `git log -1 main` still `c1ca0b2`; `git branch -r` shows no `claude/omni-import`; `tests/test_forbidden_dependencies.py`; `grep -rn "CLIP\|SigLIP\|QFormer\|Whisper\|ImageBind\|VisionEncoder" src/chimera_omni` empty.

## S4b — K3 MLA mixer receipt against the real module (2026-09-02, commit 39f6530)

Closes the S4 gap "`--layer 3` against the real `KimiMLAAttention` was not run". New script
`scripts/import_receipt_k3_mla.py` (mixer-only staging of layer 3's 14 attention/norm/AttnRes
tensors from the staged shard; reference = `KimiMLAAttention(config.text_config, layer_idx=3)`
from the checkpoint's own `modeling_kimi_linear.py`, eager, additive causal mask; Chimera =
`import_k3_layer_state` -> `ChimeraBlock.attention_output`; seeded `randn[1,64,7168]` fed pre-norm).
Receipt `receipts/k3_layer3_mla_receipt.json` (sha256 badf77da…); 11 new tests; suite 387 passed.

| quantity (cpu/float32) | max abs diff | relative L2 |
|---|---|---|
| input_layernorm / post_attention_layernorm | 0.0 | 0.0 |
| q_latent, q_states, compressed_kv, kv_states, gate_projection | 0.0 | 0.0 |
| **mixer output** | **8.94e-08** | **1.87e-07** |
| attention-input AttnRes / FFN-input AttnRes (real depth stack, 2 blocks) | 4.77e-07 | 4.5e-08 / 5.0e-08 |

The fp32 residual is entirely downstream of the mapping: the reference scales scores by
`q_head_dim ** -0.5` and Chimera divides by `sqrt(query_head_dim)` (one ulp apart), plus
einsum-vs-matmul reduction order and the gate's layout. `mla_latent_norm_eps = 1e-6` and the
nope‖rope concatenation order hold against the real module. bf16 column (GPU6): mixer relative L2
6.6e-03, cosine 0.99998 — bf16 arithmetic, not an import claim. Out of scope by construction: the
routed FFN of layer 3 (Chimera is dense; it enters only through the C14 expert-union fold).

## Q1 resolved (2026-09-02)

Closes open item 1 of section 6 as a *reading*, not as an exactness claim.

transformers `main` and the tagged `v5.15.0` release both apply
`attn_output * torch.sigmoid(gate)` and **never read `output_gate_type`** anywhere in the
model file — `modeling_qwen3_5.py` line 774 on `main`, line 698 on v5.15.0. The key is absent from
`configuration_qwen3_5.py` as well (`Qwen3_5Config` / `Qwen3_5TextConfig` / `Qwen3_5VisionConfig`
on `main`), so nothing constructs a different gate from it either. The
Qwen3.8-27B-Uncensored `output_gate_type: "swish"` flag is therefore **inert under every HF
version**: no released runtime ever branches on it, and a 27B checkpoint loaded by
transformers runs the sigmoid gate regardless of what its `config.json` declares.

Checked three ways, not taken on report: `raw.githubusercontent.com/huggingface/transformers`
at `main` and at tag `v5.15.0` for `models/qwen3_5/{modeling,configuration}_qwen3_5.py`
(`output_gate_type` absent from all four files; the gate line is
`attn_output = attn_output * torch.sigmoid(gate)`), and the transformers **5.12.1** installed in
`/home/cereal/model-surgery-lab/venv` on the AI server —
`grep -rn output_gate_type .../transformers/models/qwen3_5/` returns nothing and the gate sits at
`modeling_qwen3_5.py:714`, which is the line the code docstrings already cite.

What that does and does not settle:

- It removes the *disagreement* risk. There is no HF version under which the source runs
  `silu(gate)` while Chimera runs `sigmoid(gate)`, so the earlier framing ("exact under THIS
  runtime, inexact under the one that wrote it") no longer has a second runtime in it.
- It does **not** prove the weights were trained under the sigmoid gate. The flag being
  unread says nothing about the trainer that produced the checkpoint. So the refusal stays
  on: `chimera_omni_27b_profile(config)` still raises by default, and only
  `allow_unverified_output_gate=True` builds the config.
- No gate-kind switch is added to `GatedGQAAttention`. `gqa_output_gate` stays a bool until
  something other than an unread config key argues otherwise.

Where the observation lives in the code, as one constant so the three places cannot drift:
`chimera_omni.importing.profiles.QWEN_OUTPUT_GATE_Q1_OBSERVATION`.

```
Q1 resolved 2026-09-02: transformers main and v5.15.0 both apply attn_output * torch.sigmoid(gate)
and never read output_gate_type (modeling_qwen3_5.py line 774 / 698), so the Qwen3.8-27B
output_gate_type: "swish" flag is inert under every HF version
```

1. The refusal message quotes it, so anyone who hits the wall gets the finding with the wall
   (`profiles.py` `_qwen_sigmoid_output_gate`).
2. `allow_unverified_output_gate=True` stamps it verbatim into
   `metadata["attention_output_gate_observation"]`, which `ChimeraConfig.to_manifest()` carries
   into every import and export receipt built on that config, next to the existing
   `attention_output_gate = "sigmoid_assumed_declared_swish"`.
3. The 27B **planning** profile (no checkpoint to refuse) appends it to
   `metadata["unresolved"]`, which still contains `swish`.

Guarded by `tests/test_import_profiles.py::test_the_output_gate_refusal_and_its_receipt_carry_the_q1_reading`.

### Merge-readiness fences landed with it (commit `5a87018`, branch `claude/omni-import`)

- **GGUF writer size fence.** `build_export_tensors` materialises every tensor as fp32 numpy
  while `GGUFWriter` holds its own copy until the drain, so the writer refuses a plan over
  `EXPORT_PARAMETER_FENCE = 2_000_000_000` elements unless the caller passes `allow_large=True`
  (`--allow-large` on `scripts/export_gguf_qwen35.py`). The estimate and the waiver are both in
  `receipt["scalability"]`. Measured: the S5-verified 0.8B export is 756,587,328 elements /
  4.23 GiB peak and passes untouched; the 27B profile is 26,937,941,504 elements / 150.53 GiB
  and is refused. This is open item 4 fenced, not fixed — the streaming rewrite is still owed.
  Regression-checked by re-running the real 0.8B export on the fenced tree with the S5-fix
  inputs (same checkpoint, same `--chimera-commit c474128…`), writing to
  `/home/cereal/omni-campaign/import-lane/s5b/`: **sha256
  `fcb61be2db105b7f89ece090173dbbacdf75a1dbc4ffaaf02157b23d6d9657ef`, 1,525,255,104 B, 320
  tensors — byte-identical to `qwen35-0p8b-chimera-f16-s5fix.gguf`.** The receipt gained
  exactly two keys (`export_status`, `scalability`); the only other differences are
  run-dependent (`generated_utc`, `path`, `write_seconds`, `source_checkpoint.load_seconds`).
- **kimi-k3 labelled unproven.** `KIMI_K3_EXPORT_STATUS` = `plan-and-tests-only; no real
  K3-derived child exported; MoE half unmapped (dense-only by design)`, carried on the
  `ExportPlan`, written into `receipt["export_status"]` and into the file's own
  `chimera.export_status` KV. The qwen35 plan's status is empty — that lane has been through a
  real checkpoint and a real llama.cpp build.
