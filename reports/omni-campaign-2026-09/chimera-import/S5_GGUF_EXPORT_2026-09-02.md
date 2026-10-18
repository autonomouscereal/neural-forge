# S5 — GGUF export lane (spec 1.14), and the Qwen3.5-0.8B round trip

> **Superseded in part on 2026-09-02** by `S5_BLOCKER_FIXES_2026-09-02.md` (commit `c474128`): the CSA/HCA enumeration blocker, the false `dtype_on_disk` field, and four receipt-truth items. Both receipts here were re-run; the GGUF's tensor payload is unchanged.

Date: 2026-09-02. Commit `8f9e2f9edb689f7bd3df263bfb50c4c5bb3996c5` on branch `claude/omni-import` in `C:/Users/cereal/Documents/ai training/chimera-omni`.
Scope: spec section 1.14 (`export/gguf_names.py`, `export/gguf_writer.py`, `tests/test_gguf_name_maps.py`)
plus the ordered receipt: export the S4-imported Qwen3.5-0.8B Chimera child to a `qwen35` GGUF,
load it in llama.cpp on GPU6, run three fixed prompts through it and through the HF model.

---

## 1. The headline number

The GGUF written from the Chimera child generates **the same text as the HF Qwen3.5-0.8B model**
on all three prompts: 48/48 greedy tokens identical, on every prompt, and the llama.cpp tokenizer
reproduces the HF token ids exactly.

```
{
  "prompts": 3,
  "tokenizer_agreement": 1.0,
  "identical_text": 3,
  "mean_token_agreement": 1.0,
  "mean_matching_prefix": 48.0,
  "top5_ids_agree_in_order": 1,
  "top5_id_set_agrees": 3,
  "max_top1_probability_gap": 0.02915334106749279
}
```

Side by side (full text in `receipts/qwen35_0p8b_gguf_roundtrip.json`):

| prompt | HF Qwen3.5-0.8B (bf16, GPU) | llama.cpp GGUF (f16, GPU6) |
|---|---|---|
| `The capital of France is` | ` Paris.\nThe capital of France is Paris.\n…` | identical |
| `def fibonacci(n): … return` | ` fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10))…` | identical |
| `A receipt is the artifact; … In practice that means` | ` that the prediction is not a measurement.\n…` | identical |

Generation used the **same token id list on both sides** (llama-server accepts an array prompt),
so a tokenizer difference cannot be mistaken for a model difference. Tokenizer agreement is
reported separately and is 3/3.

`top5_ids_agree_in_order = 1` is not a disagreement about candidates: the top-5 **sets** match on
3/3, and the order differs only where the bf16 reference produces *exact* probability ties
(prompt 0: ids 198 and 25 both at 0.057566; prompt 1: ids 15336 and 307 both at 0.060710) that
the two runtimes break differently. The largest top-1 probability gap is 0.029 (0.6527 → 0.6236),
which is the f16-vs-bf16 and kernel difference, not a mapping error.

---

## 2. What was built

### `src/chimera_omni/export/gguf_names.py` — the plan, no third-party imports

Stdlib + `..config` only, so the mapping is auditable without `gguf`, without `torch` and without
a checkpoint.

* `reorder_v_permutation(Hk, r, head_dim)` — llama.cpp's `_reorder_v_heads` as an explicit index
  permutation (identity at `r == 1`, which is the 0.8B and 4B case; the tests exercise `r = 3`).
* `q_gate_interleave_permutation` / `q_gate_deinterleave_permutation` — the export and import
  directions of Qwen's fused per-head `[q|gate]` matrix, proved inverse in both directions.
* `chimera_parameter_names(cfg)` — the `state_dict()` keys of a child, derived from the config
  alone; a torch-optional test asserts it against a real `ChimeraOmniModel`, names **and shapes**.
* `qwen35_export_plan(cfg)` / `kimi_k3_export_plan(cfg)` — `ExportPlan(arch, tensors, omitted, kv)`.
  `ExportPlan.covers()` refuses a plan that drops, double-reads or invents a parameter, so the
  media bridge, the media head and the aux heads are **named as omitted with a reason**
  (companion `chimera-media` GGUF, spec 1.14) rather than silently dropped.
* `require_qwen35_exportable` / `require_kimi_k3_exportable` — every precondition of spec 1.14(a)
  and 1.14(b) asserted, never coerced.

### `src/chimera_omni/export/gguf_writer.py` — the only module that imports `gguf`

* `build_export_tensors(state, cfg, arch=…)` returns every GGUF tensor as fp32 numpy plus the plan
  and the per-parameter read counts, with no file and no `gguf` needed. This is what makes the
  transform tests real instead of shape assertions.
* `write_chimera_gguf(...)` → receipt with the architecture, the full KV table, every tensor with
  shape/dtype/transform/sources, the omissions with reasons, and the **sha256 of the file**.
* `qwen_gguf_vocab(model_dir, cfg)` builds the token table: Qwen's `tokenizer.json` vocab and
  added tokens, `[PAD*]` placeholders for the 250 embedding rows Qwen pads above the tokenizer
  file, and the 4,096 ledger band rows as CONTROL tokens spelled `<|co:band:index|>` — the same
  spelling `ChimeraTokenizer.band_marker` decodes to. 252,416 tokens, one per `token_embd` row.
* Dtype policy matches llama.cpp's converters: rank ≥ 2 takes the file type, norms / per-head
  scalars / biases stay F32. An f16 cast that would produce a non-finite value raises.

Two conventions written down because getting them wrong is silent:

* **The `(1 + w)` norms are already folded in.** llama.cpp's Qwen converter adds 1 to every
  `*norm.weight` except `ssm_norm`; a Chimera child imported from Qwen already stores `1 + w`, so
  the export copies verbatim and must not add 1 again.
* **`ssm_a` is `-exp(decay_log_scale)`**, not the parameter itself.

### `tests/test_gguf_name_maps.py` — 29 tests

Coverage (every parameter accounted for exactly once, media/aux named as omitted), the two
permutations and their inverses, the transforms run end-to-end on a synthetic numpy state
(concatenation order, the `reorder_v` of the V rows/columns/scalars, the q/gate interleave, the
gate-then-up FFN split, the res-score fusion `norm.weight * query`, the 4-D K3 conv shape, the
MLA `kv_up` split into a transposed `attn_k_b` and a plain `attn_v_b`), 20 precondition refusals,
both KV tables, plus two optional-dependency tests: the parameter enumeration against a real
`state_dict()` (torch) and a written file read back byte-exactly with `gguf.GGUFReader` (gguf).

### Scripts

* `scripts/export_gguf_qwen35.py` — checkpoint → GGUF + export receipt.
* `scripts/gguf_roundtrip_qwen35.py` — the three-prompt probe against a running llama-server.

---

## 3. What actually ran on the server

Environment: `/home/cereal/model-surgery-lab/venv` (torch 2.11+cu128, transformers 5.12.1,
**gguf 0.19.0 already installed** — `MODEL_ARCH.QWEN35 = 34`, `MODEL_ARCH.KIMI_K3 = 143`).
Runtime: `ghcr.io/ggml-org/llama.cpp:server-cuda`, digest `sha256:de9b8ee9e154…`, build
**b10068 (`571d0d540`)**; `qwen35` is present in its `libllama.so.0.0.10068`.

**Export** (`out/export.log`), CPU only:

```
[export 05:36:15] loading /home/cereal/omni-campaign/import-lane/s4-fix/qwen35_0p8b_chimera_bf16 in float32
[export 05:36:46] loaded in 31.17s; state sha256 03d88620140214a9fdd937acb8cfc1e38a619e1d94c3f01602af9bd26f8795e9
[export 05:36:47] 252416 tokens, 247587 merges
[export 05:37:13] wrote .../qwen35_0p8b_gguf_export.json
{
  "gguf": "/home/cereal/omni-campaign/import-lane/s5/out/qwen35-0p8b-chimera-f16.gguf",
  "bytes": 1525255072,
  "sha256": "ee297772e2fd4924f6726b68227cfaf415f441022258c327cea02762813e34ae",
  "tensors": 320,
  "omitted": 17,
  "tokens": 252416
}
```

KV table written (from the receipt):

```
general.architecture qwen35   block_count 24   context_length 262144
embedding_length 1024   feed_forward_length 3584   full_attention_interval 4
attention.head_count 8   head_count_kv 2   key_length 256   value_length 256
attention.layer_norm_rms_epsilon 1e-06
rope.freq_base 1e7   rope.dimension_count 64   rope.dimension_sections [11, 11, 10, 0]
ssm.conv_kernel 4   ssm.state_size 128   ssm.group_count 16   ssm.time_step_rank 16   ssm.inner_size 2048
vocab_size 252416
```

**Load** (container `omni_gguf_probe`, GPU6 `GPU-f5a09535-…` = Quadro RTX 8000, port 127.0.0.1:28088):

```
0.01.504.405 I srv    load_model: loading model '/models/qwen35-0p8b-chimera-f16.gguf'
0.54.225.638 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 2048, kv_unified = 'false'
0.54.255.884 I srv  llama_server: model loaded
0.54.255.912 I srv  llama_server: listening on http://0.0.0.0:8080
```

`/props` reports `build_info b10068-571d0d540`, `model_ftype F16`.
The container was **stopped and removed** after the probe; GPU6 is back to 5 MiB.

**Tests** — full suite in the lab venv (torch, gguf and tokenizers all present):

```
........................................................................ [ 19%]
........................................................................ [ 38%]
........................................................................ [ 58%]
........................................................................ [ 77%]
........................................................................ [ 96%]
............                                                             [100%]
```

Re-run against the exact committed tree (`git archive claude/omni-import` at `8f9e2f9`):

```
372 passed, 1 warning in 17.62s          # pytest_s5_full_8f9e2f9.log
tests/test_gguf_name_maps.py .............................  [100%]
============================== 29 passed in 2.27s ==============================
```

372 passed, 0 failed, 0 skipped (`pytest_s5_full_8f9e2f9.log`, `pytest_s5_gguf_8f9e2f9.log`; the
one warning is the pre-existing `requires_grad` scalar warning in
`test_checkpoint_round_trip_torch_optional.py`). Locally on `C:/Python314/python.exe` (no torch,
no gguf) the same suite is green with the expected skips.

---

## 4. Files

Repo (`chimera-omni`, branch `claude/omni-import`):

* `src/chimera_omni/export/__init__.py`
* `src/chimera_omni/export/gguf_names.py`
* `src/chimera_omni/export/gguf_writer.py`
* `scripts/export_gguf_qwen35.py`
* `scripts/gguf_roundtrip_qwen35.py`
* `tests/test_gguf_name_maps.py`
* `receipts/qwen35_0p8b_gguf_export.json`
* `receipts/qwen35_0p8b_gguf_roundtrip.json`

This lane: `qwen35_0p8b_gguf_export.json`, `qwen35_0p8b_gguf_roundtrip.json`, `export.log`,
`roundtrip.log`, `llama_server.log`, `pytest_s5_full.log`, `pytest_s5_full_8f9e2f9.log`,
`pytest_s5_gguf_8f9e2f9.log`, this report.

Server (nothing deleted): `/home/cereal/omni-campaign/import-lane/s5/` holds `omni-import-s5.tar`,
`omni-import-s5b.tar`, `omni-s5-scripts{,2}.tar`, `omni-import-s5-commit.tar`, `src/`, `src_full/`, `src_commit/` and `out/` (the 1.53 GB GGUF,
both receipts, three logs).

---

## 5. What this does NOT prove

* **The `kimi-k3` path has never written a real checkpoint.** Its plan, preconditions and every
  transform are tested on a synthetic numpy state and a tiny file is written and read back with
  `gguf.GGUFReader`, but no K3-derived Chimera child exists yet to export, and no llama.cpp build
  has been asked to load a `kimi-k3` file produced here. The MoE half of spec 1.14(b)
  (`ffn_*_exps`, `ffn_gate_inp`, `exp_probs_b`, the shared experts) is **not implemented at all**:
  Chimera has no MoE feed forward, so there is nothing to map.
* **The companion `chimera-media` GGUF of spec 1.14 is not written.** The 17 omitted parameters
  (media bridge, media head, flow/grounding/sync) are enumerated with reasons in the receipt and
  nothing else.
* **`tokenizer.ggml.pre = "qwen2"` is still an assumption** (spec Q9). It is now an assumption
  with a measurement behind it — the llama.cpp tokenizer reproduced the HF ids exactly on all
  three prompts — but three prompts is three prompts. Qwen3.5's `tokenizer.json` splits on
  `[\p{L}\p{M}]+` where llama.cpp's `qwen2` regex splits on `\p{L}+`, so a text with a combining
  mark is where they would first differ, and no such text was tested.
* **The f16 cast was not measured against the bf16 source.** The checkpoint is bf16 on disk, was
  widened to fp32 (exact) and cast to f16 at write; bf16 values below the f16 normal range become
  subnormal or zero. No count of changed values is in the receipt.
* **llama.cpp warns on six tokens.** `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`,
  `<|fim_pad|>`, `<|repo_name|>` and `<|file_sep|>` are `"special": false` in Qwen's
  `added_tokens`, so the writer typed them USER_DEFINED and llama.cpp overrode them to CONTROL
  with "this is probably a bug in the model". That mirrors what the `tokenizer.json` says; whether
  llama.cpp's own Qwen converter emits them as CONTROL instead was not checked.
* **`context_length` is written as 262,144** (the profile's `max_sequence_length`). The probe ran
  at `-c 2048`. Nothing here says the exported file is correct at long context — and S2 already
  recorded that Chimera's fp32 RoPE table carries ~1.9e-4 absolute error at position 262,144.
* The receipt's `n_predict` is 48 tokens. Agreement over 48 greedy tokens on three prompts is not
  a claim about agreement at 2,048.
