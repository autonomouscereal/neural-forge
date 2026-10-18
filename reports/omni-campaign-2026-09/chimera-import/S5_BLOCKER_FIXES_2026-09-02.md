# S5 blocker fixes — GGUF export

Branch `claude/omni-import`, on top of `8f9e2f9` ("omni-import S5: the GGUF export lane,
and a qwen35 file llama.cpp answers identically"). Code fix commit
`c474128771da60691fcf1fb2c9de67780e0c8a35`.

Two review blockers, both fixed in place, plus four smaller receipt-truth items from the
same review. The export and the round trip were re-run on the AI server afterwards; every
measured number reproduced, and the new file's tensor payload is byte-identical to the S5
file (only a metadata string changed).

---

## 1. Blockers

### B1 — CSA/HCA blocks hid a whole attention module from the enumeration

`gguf_names._mixer_names` enumerated a `csa`/`hca` mixer as `summary_q`, `summary_k`,
`summary_v` and `out`. `SparseLatentAttention.__init__` (`modeling/layers.py:440`) also
builds `self.local = LocalCausalAttention(cfg)`, so the real `state_dict` carries four
more weights per block: `blocks.{i}.mixer.local.{q_proj,k_proj,v_proj,out}.weight`.

`chimera_parameter_names()` is the oracle `ExportPlan.covers()` compares a plan against.
Four names missing from the oracle means a plan that dropped those four weights would
have been accepted as complete — the stage's central guarantee, "refuse a plan that drops
a parameter", was defeated for those two block kinds. Neither export reaches it today
(both `require_*_exportable` reject `csa`/`hca` first), but the D5b `chimera-media`
companion writer and the `chimera`-arch fallback of spec Q8 would have walked into it.

Fixed at `src/chimera_omni/export/gguf_names.py:186` — the four names now come from a
shared `_local_attention_names(prefix)` helper, used both for a `local` block's mixer and
for the `mixer.local.` submodule a sparse-latent block mounts.

The enumeration is now bound to a real `state_dict` for **every** block kind, not only
the two the architectures use: `every_block_kind_config()` in
`tests/test_gguf_name_maps.py` carries one block of each of `kda`, `local`, `csa`, `hca`,
`gated_mla`, `gated_gqa`, plus an MTP head, conv biases, a latent-free MLA query path and
an ungated GQA — branches no previous config exercised. Building that config found a
second, smaller error: the test helper `_shape_for` gave a `gated_mla` block the KDA
`mixer.out` shape, which happens to be right for the K3-shaped config
(`kda_heads * kda_value_head_dim == num_heads * mla_value_head_dim`) and wrong in
general. Fixed in the helper.

### B2 — `dtype_on_disk` reported the dtype the loader asked for

`scripts/export_gguf_qwen35.py:106` wrote

```python
"dtype_on_disk": load_receipt.get("dtype")
```

and `load_chimera_checkpoint` sets that field to `str(dtype)` — the dtype the model was
loaded *into* (`importing/safetensors_io.py:551`), which this script sets to
`torch.float32`. So the committed receipt claimed `"dtype_on_disk": "torch.float32"`
while every stored tensor in the S4 checkpoint is BF16:

```
[export 06:13:29] checkpoint dtypes on disk: {'BF16': 390}
```

That receipt is the only artifact that prices the bf16 -> f16 narrowing, and it asserted
the wrong source dtype.

Fixed with a new `checkpoint_dtypes(directory)` in
`src/chimera_omni/importing/safetensors_io.py:292` — it reads the shard JSON headers only
(one small read per shard, the payload is never touched) and returns the tensor count per
safetensors dtype. The export receipt now carries

```json
"dtypes_on_disk": {"BF16": 390},
"load_dtype": "torch.float32"
```

and `load_chimera_checkpoint`'s own receipt gained `dtypes_on_disk` and `load_dtype`
beside the (now commented) `dtype`.

---

## 2. Receipt-truth items from the same review

* **The f16 cast now has a number.** `float16_narrowing_stats()`
  (`export/gguf_writer.py`) counts, per tensor, the nonzero values the cast flushes to
  zero and the values that survive only as f16 subnormals. Overflow already raised; these
  are the quiet losses. For the 0.8B file:

  ```
  "float16_cast": {
    "file_type": "f16", "tensors_narrowed": 187, "elements_narrowed": 756088832,
    "flushed_to_zero": 15889, "subnormal": 3023581
  }
  ```

  2.1e-5 of the narrowed values became zero and 4.0e-3 became subnormal. The largest
  single contributor to the zeros is `blk.0.attn_qkv.weight` (14,345 of 15,889);
  `token_embd.weight` contributes 309 zeros and 650,896 subnormals out of 258M values.
  The source is bf16 widened to fp32 (exact), so these counts are the whole difference
  between the checkpoint value and the written value. Closes gap 5 of the S5 report.
* **`gguf_version` is a measurement.** 0.19.0 exposes no `__version__`, so the old
  `getattr(gguf, "__version__", "unknown")` wrote "unknown" into every receipt next to a
  hardcoded "verified 0.19.0" string. It now reads `importlib.metadata.version("gguf")`
  and records `0.19.0`.
* **`chimera_commit` is passed in.** The server runs a `git archive` with no `.git`, so
  `repo_commit()` returned "unknown". `--chimera-commit` was added and the receipt now
  records `c474128771da60691fcf1fb2c9de67780e0c8a35`.
* **The missing bos id is stated.** Qwen's `tokenizer_config.json` has `bos_token: null`,
  so no `tokenizer.ggml.bos_token_id` is written and llama.cpp falls back to its gpt2
  default (`/props` reports `bos_token: ","`). `add_bos_token` is written `False`, so
  generation is unaffected; the receipt notes now say so instead of leaving a `null`
  field to be interpreted.

Not done: the peak-RAM streaming rewrite (review item 1). The writer still materialises
the fp32 plan and the f16 copies at once, which is fine at 0.8B (~7 GB) and will not run
at 4B or 27B. It needs `use_temp_file=True` plus dropping each built tensor after
`add_tensor`, and it should be measured when the 4B export is ordered.

---

## 3. What was re-run on the server

Lab venv `/home/cereal/model-surgery-lab/venv`, tree extracted from
`git archive claude/omni-import` at `c474128` into
`/home/cereal/omni-campaign/import-lane/s5/src_fix`.

**Tests** (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6`):

```
376 passed, 1 warning in 16.28s                  # pytest_s5fix_full.log  (was 372 at 8f9e2f9)
============================== 45 passed in 2.49s ==============================
                                                 # pytest_s5fix_gguf.log
tests/test_gguf_name_maps.py::test_a_sparse_latent_block_enumerates_its_nested_local_attention PASSED
tests/test_gguf_name_maps.py::test_float16_narrowing_counts_the_values_the_cast_loses PASSED
tests/test_gguf_name_maps.py::test_the_parameter_enumeration_matches_a_real_state_dict[qwen35_shaped_config] PASSED
tests/test_gguf_name_maps.py::test_the_parameter_enumeration_matches_a_real_state_dict[kimi_k3_shaped_config] PASSED
tests/test_gguf_name_maps.py::test_the_parameter_enumeration_matches_a_real_state_dict[every_block_kind_config] PASSED
tests/test_safetensors_io.py::test_checkpoint_dtypes_reports_the_shards_not_a_loader_argument PASSED
```

The one warning is the pre-existing `requires_grad` scalar warning in
`test_checkpoint_round_trip_torch_optional.py:129`.

**Export** (CPU, written as `qwen35-0p8b-chimera-f16-s5fix.gguf` beside the S5 file rather
than over it):

```
[export 06:12:43] loading /home/cereal/omni-campaign/import-lane/s4-fix/qwen35_0p8b_chimera_bf16 in float32
[export 06:13:12] loaded in 29.06s; state sha256 03d88620140214a9fdd937acb8cfc1e38a619e1d94c3f01602af9bd26f8795e9
[export 06:13:12] 252416 tokens, 247587 merges
[export 06:13:29] checkpoint dtypes on disk: {'BF16': 390}
{"bytes": 1525255104,
 "sha256": "fcb61be2db105b7f89ece090173dbbacdf75a1dbc4ffaaf02157b23d6d9657ef",
 "tensors": 320, "omitted": 17, "tokens": 252416}
```

The file is 32 bytes larger than the S5 file and has a different whole-file sha256 because
`chimera.commit` went from "unknown" to the 40-character hash. The **tensor payload is
identical**, checked by hashing name + ggml type + shape + bytes of all 320 tensors
through `gguf.GGUFReader` on both files:

```
S5     : ('dfa922a4a6a56c57b40b2758bc3557c6d8928d3645b0f06b05ce5ebc3d9632cb', 320)
S5 fix : ('dfa922a4a6a56c57b40b2758bc3557c6d8928d3645b0f06b05ce5ebc3d9632cb', 320)
tensor payload identical: True
```

**Round trip** (new container `omni_gguf_probe_fix`, GPU6 Quadro RTX 8000,
`127.0.0.1:28089`, `-c 2048 -ngl 99`; `/props` `build_info b10068-571d0d540`):

```
0.00.522.343 I srv    load_model: loading model '/models/qwen35-0p8b-chimera-f16-s5fix.gguf'
0.53.384.500 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 2048, kv_unified = 'true'
0.53.413.047 I srv  llama_server: model loaded

{"prompts": 3, "tokenizer_agreement": 1.0, "identical_text": 3,
 "mean_token_agreement": 1.0, "mean_matching_prefix": 48.0,
 "top5_ids_agree_in_order": 1, "top5_id_set_agrees": 3,
 "max_top1_probability_gap": 0.02915334106749279}
```

Every number reproduces the S5 run exactly, including the 0.029 top-1 probability gap and
the three identical continuations. The container was stopped and removed;
`docker ps -a --filter name=omni_gguf_probe_fix` is empty and GPU6 is back to 5 MiB.

---

## 4. Files

Repo (`chimera-omni`, branch `claude/omni-import`):

* `src/chimera_omni/export/gguf_names.py` — `_local_attention_names`, csa/hca fix
* `src/chimera_omni/export/gguf_writer.py` — `float16_narrowing_stats`,
  `_gguf_package_version`, the `float16_cast` receipt block, the bos note
* `src/chimera_omni/importing/safetensors_io.py` — `checkpoint_dtypes`, plus
  `dtypes_on_disk` / `load_dtype` in the load receipt
* `scripts/export_gguf_qwen35.py` — true `dtypes_on_disk`, `--chimera-commit`
* `tests/test_gguf_name_maps.py` — `every_block_kind_config`, two new tests, the MLA
  `mixer.out` shape
* `tests/test_safetensors_io.py` — the `checkpoint_dtypes` test
* `receipts/qwen35_0p8b_gguf_export.json`, `receipts/qwen35_0p8b_gguf_roundtrip.json` —
  replaced by the re-run (the S5 versions stay in git history at `8f9e2f9`)

Server, `/home/cereal/omni-campaign/import-lane/s5/` (nothing deleted): `src_fix/`,
`omni-import-s5fix.tar`, and in `out/`: `qwen35-0p8b-chimera-f16-s5fix.gguf` (1.53 GB),
`qwen35_0p8b_gguf_export_s5fix.json`, `qwen35_0p8b_gguf_roundtrip_s5fix.json`,
`export_s5fix.log`, `roundtrip_s5fix.log`, `llama_server_s5fix.log`,
`pytest_s5fix_full.log`, `pytest_s5fix_gguf.log`. The S5 artifacts sit untouched beside
them.
