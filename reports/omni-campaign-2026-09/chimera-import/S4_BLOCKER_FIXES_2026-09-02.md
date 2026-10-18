# S4 blocker fixes — IO and receipts

Branch `claude/omni-import`, on top of `773c36f` ("omni-import S4: safetensors IO, the
Qwen3.5 and K3 exact-import loaders, and the two receipts").

Four review blockers, all fixed in place, plus six smaller items from the same review.
Both acceptance receipts were re-run on the AI server afterwards and every measured
number reproduced.

---

## 1. Blockers

### B1 — the K3 switch guard never checked the MLA layer's own precondition

`_check_k3_switches` demanded the KDA switches for every layer and checked nothing that
belongs to a `gated_mla` layer, so importing layer 3 with `mla_latent_norm_eps=0`
(= "use `rms_norm_eps`", 1e-5) resolved the two latent norms to the wrong epsilon,
raised nothing, and returned a receipt with an empty `source_adjustments` and
`imported == block_total` — an inexact block that reads as an exact one.

`KimiRMSNorm.__init__` defaults to `eps=1e-6` (`modeling_kimi_linear.py:226-230`) and
`q_a_layernorm` / `kv_a_layernorm` are built without an eps argument (lines 368, 383),
while every other norm in the model is handed `config.rms_norm_eps` (1e-5).

The guard is now keyed on the block kind (`importing/kimi_k3.py:123`):

* `kda` — decay kind, output gate, conv bias, key/value head equality, `kda_query_scale`,
  and now `kda_qk_norm_kind == "fla"` (`chunk_kda` is called with
  `use_qk_l2norm_in_kernel=True`, whose l2norm clamps at 1e-6; `F.normalize` clamps at
  1e-12);
* `gated_mla` — `resolved_mla_latent_norm_eps() == 1e-6`, named after `KimiRMSNorm`'s
  default;
* anything else refuses.

This also settles review item (f): a `gated_mla` layer contains no KDA, so the KDA
switches no longer gate it.

### B2 — the Qwen loader did not check the two switches its receipt depends on

`import_qwen35_state` validated `residual_mode`, `kda_decay_kind`, `kda_output_gate`,
`kda_conv_bias` and `activation_kind`, but not `kda_query_scale` or `kda_qk_norm_kind` —
the two the fp32 column actually rides on. `modeling_qwen3_5.py:332-341` l2-normalises
q/k with eps 1e-6 and then scales the query by `1 / key_dim**0.5`. Both are now refused
if unset (`importing/qwen35.py:81`), and the module docstring names the load-bearing
switches instead of only claiming "nothing is approximated".

### B3 — `hash_state_dict` crashed on every model built since S1

`proof/h3_hybrid_av_flow.py:485` did `tensor.view(torch.uint8)` on the unflattened
tensor. S1 gave every `DepthAttentionResidual` a 0-dim `prefix_logit_bias`, and
`view(uint8)` refuses 0-dim tensors, so `save_checkpoint` — and with it the joint-AV
proof driver — died at the first freeze hash. It is now
`tensor.reshape(-1).view(torch.uint8)`, matching `safetensors_io._as_view`. For a
contiguous tensor the flattened bytes are the same bytes, so freeze hashes recorded by
earlier runs of that proof are unchanged.

### B4 — the "same convention" claim was false

`state_sha256`'s docstring said a proof freeze hash and a checkpoint hash were
comparable. They are not: `state_sha256` feeds the safetensors dtype token (`"F32"`) into
the digest, `hash_state_dict` feeds `str(tensor.dtype)` (`"torch.float32"`), so the two
digests of one identical state can never agree.

Fixed by deleting the claim, not by unifying the digests: changing the proof lane's
encoding would make every already-recorded freeze hash in this campaign
non-reproducible. Both docstrings now name the difference, and
`tests/test_state_hash_conventions_torch_optional.py` asserts the two digests are stable
and unequal, so the claim cannot drift back in.

---

## 2. Smaller items from the same review

* **(a)** `disable_fast_paths` in `scripts/import_receipt_qwen35.py` mutated the
  transformers module globals irreversibly. A GPU column that followed a CPU column would
  have run the torch fallbacks and recorded `fast_path_available: false` as a property of
  the machine. There is now a `restore_fast_paths`; every column starts from the
  unpatched module and records whether it had to undo a previous one. The column also
  carries `fast_path_note`, spelling out that `is_fast_path_available` is transformers'
  all-four-pieces flag and that `false` does **not** mean the fla kernel was unused.
* **(b)** The write-read round trip runs on one column only. The receipt now records
  `round_trip_column` (`"cuda:0/bfloat16"`), so "logits bit-identical" is attributable.
* **(c)** `safetensors_io._as_view` keyed tie detection on the storage base pointer, so
  two different views of one allocation would have been written once and aliased — silent
  corruption. The key is now `(data_ptr, storage_offset, shape, stride, dtype)`. No
  Chimera parameter is such a view today; the regression test builds one deliberately.
* **(d)** The recurrent-state layout is documented on `KDAState`: Chimera returns
  `[B, H, K, V]`, `chunk_kda` with `transpose_state_layout=True` returns `[B, H, V, K]`,
  and the receipt measures both.
* **(g)** `test_sharding_splits_at_the_requested_byte_budget` now actually asserts the
  claim its comment makes (every shard either holds one tensor or fits the budget).
* **(h)** The report-directory copies of `pytest_s4_full.log` / `pytest_s4_new_tests.log`
  were 21 B and 19 B; they are now the full server logs.

---

## 3. Server evidence (lab venv, `/home/cereal/omni-campaign/chimera-omni-import/src`)

```
$ timeout 900 /home/cereal/model-surgery-lab/venv/bin/python -m pytest --tb=short -p no:warnings tests
343 passed in 17.91s

$ ... the six S4 files plus the two new ones
53 passed in 3.13s
```

332 -> 343: 11 new tests (3 K3 refusal/kind-keying, 1 tie-alias, 5 state-hash, 2
fast-path restore; the two Qwen refusals extend an existing test).

**R-Q re-run** (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6`), same 8 prompts /
210 positions:

| column | max abs logit diff | argmax | hidden (layers 0-3) |
|---|---|---|---|
| `cpu/float32` | 3.191828727722168e-05 | 210/210 | 7.748603820800781e-07 |
| `cuda:0/bfloat16` | 0.5 | 209/210 | 0.015625 |

Round trip on `cuda:0/bfloat16`: `state_sha256_before == state_sha256_after`,
`logits_bit_identical: true`, `tied_weights {lm_head.weight: embed.weight}` — the tie
still collapses under the stricter tie key.

**R-K3 re-run**, layer 0, reference = the checkpoint's own `KimiDecoderLayer`
(`missing_keys: []`, `unexpected_keys: []`):

| column | layer output max abs | relative L2 | recurrent state |
|---|---|---|---|
| `cpu/float32` | 3.4691765904426575e-08 | 9.716125855522246e-07 | as-is rel L2 1.40e-05 |
| `cuda:0/bfloat16` | 0.00048828125 | 0.006097518020615974 | transposed rel L2 7.91e-03 |

Every measured field in both receipts is identical to the `773c36f` run. The only diffs
are timestamps, timings, the new `round_trip_column` / `fast_path_*` fields, and
`chimera_state_sha256` — see below.

---

## 4. What this run corrects in the S4 report

* **The 331 s K3 reference was Triton autotune, not the kernel.** With a warm cache the
  same reference layer runs in **1.478 s** against Chimera's 0.108 s. The earlier
  "~2900x" figure was an artefact of a cold autotune and should not be repeated.
* **`chimera_state_sha256` is per-run, not a property of the checkpoint.** The Qwen import
  leaves 16 parameters randomly initialised (`native_media_bridge.*`, `flow_head`,
  `grounding_head`, ...) and the receipt script does not seed torch, so the whole-model
  hash moved between the two runs (`1877a83e…` -> `5d81a211…`) while every imported
  tensor stayed identical. What the round trip proves is that the hash is unchanged **by
  the write-read**, which is the claim that matters; it is not a checkpoint fingerprint.

## 5. Still owed (unchanged from S4)

`--layer 3` against the real `KimiMLAAttention` was still not run, so
`mla_latent_norm_eps=1e-6` remains derived from reading the checkpoint's source — now at
least it is *enforced* rather than silently assumed. The 4B / 27B receipts, the routed
expert path, and the GGUF writer are untouched.

> **Superseded 2026-09-02 (S4b, commit 39f6530):** the MLA layer HAS now been measured against the real
> `KimiMLAAttention` on layer 3 — see `IMPORT_LANE_STATUS.md` §S4b and `receipts/k3_layer3_mla_receipt.json`
> (fp32 mixer max abs diff 8.94e-08; every projection and norm bit-identical).
