# K3 STREAMED HARNESS v3 — lane `k3`, Chimera Omni fold campaign (2026-09-02)

Four tools plus their test rig.  Nothing here edits neural-forge or chimera-omni.
Server copy: `/home/cereal/omni-campaign/k3/harness/` — byte-identical, checked by
sha256 on both sides.
Runtime: `/home/cereal/model-surgery-lab/venv/bin/python` (torch 2.11.0+cu128,
transformers 5.12.1, fla 0.5.2, compressed_tensors 0.18.0).

| file | what it is |
|---|---|
| `k3_common.py` | shard reads (offset-sorted, 64 MiB buffered), MXFP4 dequant + the compressed_tensors cross-check, `K3Index`, K3's RMSNorm / SiTU / `_apply_attn_res`, safetensors metadata, receipts |
| `k3_layer.py` | one decoder layer without the model: KDA (fla **and** pure-torch backends), gated MLA, latent MoE with packed experts, the form-E dense swap-in, `_forward_attn_residual`, `LayerWeights.device_bytes` |
| `k3_stream.py` | **(a)** the streamed per-layer calibration forward: caches, snapshots, resume, timing |
| `k3_receipt.py` | **(b)** the exactness receipt for one layer against the repo's own modules |
| `c14_union_coreset.py` | **(c)** the expert-union channel coreset: two-stage mass-allocated pivoted selection + Gram-form law-abiding ridge |
| `k3_review_stream.py` | **(d)** the review runner: teacher-forced R-proxy at any N, greedy decode once every MoE layer is folded, stamped baselines |
| `k3_shapes.py` | header-only shape census (no tensor data read) |
| `k3_synth.py` | builds a real, tiny K3-shaped checkpoint to test against (two of them: block size 2 and block size 3), every RMSNorm gain a **distinct** random vector |
| `gram_memory_probe.py` | measures the C14 ridge's peak RSS, pre-fix code path vs shipped, same problem, two processes |
| `test_k3_harness.py` | 30 tests, all green on the server |

## Receipts in this directory

Every claim below is backed by a JSON receipt **that is here**, not only on the
server.  (In v2 four of them existed only under `/home/cereal/omni-campaign/k3/ship/`
and a reader of this directory could not check the headline claims.)

| receipt | what it holds |
|---|---|
| `k3_harness_test_receipt.json` | the 30-test run: 30 passed, 0 skipped, 0 failed (v3.1, on GPU6) |
| `k3_layer1_shapes.json` | header-only census of the REAL Kimi-K3 layer 1 (the `A_log` finding) |
| `k3_layer3_shapes.json` | header-only census of the real layer 3 |
| `k3_synth_bs3_L3_exactness_receipt.json` | whole layer vs the repo module, block size 3, **snapshot** branch |
| `k3_synth_bs3_L5_exactness_receipt.json` | the same at layer 5, **`carried = prefix_sum + attn_out`**, 2 blocks on the stack |
| `k3_synth_v3_L01_c14_receipt.json` | C14 at three rungs, two-stage, form E + form M |
| `k3_synth_v3_L01_c14_stage1only_receipt.json` | the stage-1-only comparison at the same rung |
| `k3_gram_memory_probe_receipt.json` | ridge peak RSS, measured, two processes on one problem |
| `k3_synth_L0_gpu_keydim_receipt.json` | RTX 3090: fla-vs-torch KDA, the gated-norm order, the key-dim `A_log` refusal |
| `k3_devicefix_negative_control_receipt.json` | the v3.1 NEGATIVE control: `t13f` run against the pre-fix `device_bytes` / `device_resident_bytes`, FAILED, with the all-zero byte count in the traceback |

`superseded-pre-v3/` holds the v1 receipts, kept only so the v1 claim block can
be traced; they were produced by code that no longer exists here.  The pre-fix
code the v3.1 negative control ran against is on the server at
`/home/cereal/omni-campaign/k3/backup-pre-devicefix-20260902-073021/` (sha256
identical to the v3 files, checked both sides before the fix was uploaded).

## Entry points

```bash
cd /home/cereal/omni-campaign/k3/harness
V=/home/cereal/model-surgery-lab/venv/bin/python

# tests (builds its own synthetic checkpoints; touches nothing on the share)
$V test_k3_harness.py --work-dir /home/cereal/omni-campaign/k3/testwork5 \
                      --src-dir  /home/cereal/omni-campaign/k3/k3_src

# header-only shape census of a real layer (a few MB of read)
$V k3_shapes.py --model-dir /mnt/ai_data/models/moonshotai/Kimi-K3 --layer 1 --pattern self_attn

# (a) streamed calibration pass, staging the non-expert body to local disk
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582 \
$V k3_stream.py --model-dir /mnt/ai_data/models/moonshotai/Kimi-K3 \
    --layers 0-92 --random-tokens 65536 --tokens-tag 65536 \
    --cache-dir /home/cereal/omni-campaign/k3/cache \
    --body-dir  /home/cereal/omni-campaign/k3/body \
    --device cuda:0 --kda-backend fla --cache-set minimal --snapshot-every 12 --resume

# (b) exactness receipt for one layer  (--reference-moe materialises every expert: 59.2 GB)
$V k3_receipt.py --model-dir /mnt/ai_data/models/moonshotai/Kimi-K3 \
    --src-dir /home/cereal/omni-campaign/k3/k3_src --layer 1 --tokens 256 \
    --device cuda:0 --kda-backend fla --reference-moe \
    --out /home/cereal/omni-campaign/k3/k3_layer1_exactness_receipt.json

# (c) C14 coreset for one layer, all four rungs.  --experts-device DEFAULTS TO cpu:
#     the 896 packed experts are 15.7 GB and holding them on a 25.3 GB card is what
#     made this command die in the allocator before v3.
$V c14_union_coreset.py --model-dir /mnt/ai_data/models/moonshotai/Kimi-K3 --layer 1 \
    --cache /home/cereal/omni-campaign/k3/cache/T65536/L01/tensors.safetensors \
    --out-dir /home/cereal/omni-campaign/k3/folds \
    --rungs 65536,32768,17408,8704 --k-pool-factor 1 --device cuda:0 --experts-device cpu

# (d) review: R-proxy against a STAMPED parent baseline, then decode once every
#     MoE layer is folded.  The first command refuses to run if any layer is
#     folded; the second refuses if the baseline it is handed was itself a child.
$V k3_review_stream.py --model-dir /mnt/ai_data/models/moonshotai/Kimi-K3 --mode proxy \
    --case-file cases.json --emit-baseline parent_logits.safetensors --out review_parent.json
$V k3_review_stream.py --model-dir /mnt/ai_data/models/moonshotai/Kimi-K3 --mode proxy \
    --case-file cases.json --dense-dir /home/cereal/omni-campaign/k3/folds --rung 17408 \
    --baseline parent_logits.safetensors --out review_child.json
$V k3_review_stream.py --model-dir /mnt/ai_data/models/moonshotai/Kimi-K3 --mode decode \
    --case-file cases.json --dense-dir /home/cereal/omni-campaign/k3/folds --rung 17408 \
    --max-new-tokens 256 --out review_decode.json
```

## What the harness found while being built

1. **The shipped Kimi-K3 checkpoint cannot be loaded by its own modeling code.**
   `language_model.model.layers.1.self_attn.A_log` is `F32[128]` (header-only
   read, `k3_layer1_shapes.json`), while `KimiDeltaAttention.__init__` declares
   `torch.empty(num_heads)` = `[96]` and fla's gate does `A_log.view(H, 1)`.
   `b_proj` is `[96, 7168]` and `dt_bias` is `[12288] = 96 x 128`, so 96/128 are
   not simply swapped — 128 is `head_dim`.  Reproduced on a fixture
   (`k3_synth.py --a-log-axis key_dim`): the repo raises
   `size mismatch for A_log: copying a param with shape torch.Size([64]) ... current model is torch.Size([2])`.
   The harness resolves the axis by shape and, for a key-dim `A_log`, precomputes
   the decay and calls `chunk_kda(..., use_gate_in_kernel=False)`.
   **Which broadcast is semantically correct is NOT settled here — decision line
   D-a2b for Cyril; it needs a behavioural receipt on real weights.**
2. **K3's two MLA latent norms run at eps = 1e-6, not `config.rms_norm_eps` = 1e-5.**
   `q_a_layernorm` / `kv_a_layernorm` are built as bare `KimiRMSNorm(rank)`.
   Using the config value moved the MLA output by relL2 6.06e-2 **at fp32** —
   caught because the test compares at fp32, where arithmetic cannot hide algebra.
3. `NORM_BEFORE_GATE = True` is measured, not assumed: against fla's own
   `FusedRMSNormGated` at fp32, norm-then-gate scores relL2 **8.57e-8** and
   gate-then-norm **0.837** (`k3_synth_L0_gpu_keydim_receipt.json`, re-measured
   on the v3 fixture).
4. transformers 5.12.1 moved `OutputRecorder` out of `transformers.utils.generic`;
   `k3_receipt._transformers_compat()` re-exports the same class and records the
   substitution in every receipt.

## Measured on the server (2026-09-02, v3)

* **30/30 tests pass, 0 skipped** — `k3_harness_test_receipt.json`
  (`30 passed, 0 skipped, 0 failed`, sha256 `6963dcd46a0fdf45…`).  The v3.1 run
  was pinned to the RTX 8000 with `CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=6`, so the receipt's `devices` block lists ONE card
  (`sm75 Quadro RTX 8000, 47.6 GB`) where the v3 receipt listed all seven —
  that is the pin, not a missing GPU.
* Whole layer vs `KimiDecoderLayer._forward_attn_residual`, **both** residual
  branches, on a block-size-3 fixture: `k3_synth_bs3_L3_exactness_receipt.json`
  (snapshot layer, 1 block on the stack, prefix_out relL2 **6.57e-3** at bf16)
  and `k3_synth_bs3_L5_exactness_receipt.json` (`carried = prefix_sum + attn_out`,
  **2** blocks, **5.20e-3**).  Block stack bit-exact and MoE relL2 **0.0** /
  router top-k agreement **1.0** on both.  `t10b` puts the same two branches at
  fp32 relL2 **1.78e-7** (snapshot) and **1.32e-7** (carried).
* Ridge memory, measured not derived — `k3_gram_memory_probe_receipt.json`, two
  processes on the same K = 8,192 / out = 3,584 problem: **7.461 GB → 3.814 GB
  peak RSS, 1.956x**, with an identical answer (same lambda 0.01, same
  `update_scale` 0.99255182, same solution norm 20.16837292).
* **The device budget now prices what actually lands on the device** (`t13d`).
  At K = 65,536 the transient-only estimate the previous version used is
  0.82 GB; the honest one is **24.161 GB**, a **29.5x** under-pricing.  On a
  25.3 GB 3090 the default budget is 22.77 GB, so the old number was ADMITTED
  and the new one is REFUSED, with `layer_weights_on_device` / `expert_bank` /
  `form_m_lift` named in the refusal.  A real K3 MoE layer's packed experts are
  **14,797,504,512 B** plus **924,844,032 B** of E8M0 scales.
* **A device alias no longer zeroes the resident estimate** (`t13f`, v3.1).
  On the fixture layer the byte count is **122,832 B** whether it is asked for
  as `cpu`, `cpu:0`, `cuda` or `cuda:0`; before the fix the aliased spellings
  answered **0** for every group.  With the layer on the RTX 8000 the priced
  peak is **333,776 B** against a transient-only **38,912 B**, and
  `default_device_budget_gb` reads **42.803 GB** off the card (0.9 x 47.6)
  through either spelling.
* **The fixture can now tell its norm gains apart** (`t05b`).  Swapping
  `self_attention_res_norm` with `mlp_res_norm` moves the layer by relL2
  **0.287**; swapping `input_layernorm` with `post_attention_layernorm`, by
  **0.250**; flattening `routed_expert_norm` to ones, by **0.0646**.  Every one
  of those was **bit-exact zero** before, because every gain was `torch.ones`.
* **A folded run can no longer be written or read as a parent baseline**
  (`t16c`): the emit refuses before any weight is read, the emitted file carries
  a `__metadata__` stamp (`is_parent`, `layers_folded`, `fold_rung_requested`),
  the read side refuses a stamped child, and the consuming receipt records
  `baseline_path` + `baseline_sha256` + the baseline's fold census.
* The C14 ridge reports `veto_binding`.  On the random-weight fixture it is
  `True` at every rung with `chosen_is_largest_lambda: True` and monotone
  `check_sse` — i.e. the receipt says out loud that the installed map is the
  prior, shrunk, and the `update_scale` of 0.0094–0.0157 is the veto's doing,
  not a fit.
* Segmented streaming: layers 0..m then m+1..n with `--resume` reproduces the
  single pass's `final_prefix_sha256` and the merged receipt keeps every
  segment's per-layer read record.
* MXFP4 dequant is **bit-exact** against `compressed_tensors.compressors.nvfp4.helpers.unpack_fp4_from_uint8`.
* Latent MoE vs the repo's `KimiSparseMoeBlock`: relL2 **0.0**, cos **1.0**,
  router top-k set agreement **1.0**.
* KDA on an RTX 3090 (`k3_synth_L0_gpu_keydim_receipt.json`, re-measured on the
  v3 fixture): fla `chunk_kda` vs the pure-torch recurrence relL2 **6.23e-3** (bf16).
* Streamed pass == resident pass; streamed proxy == resident proxy; incremental
  decode (conv cache + recurrent state + MLA K/V) lands on the full-pass logits.

## What the round-2 review's last blocker was, and what changed (v3.1)

**`torch.device('cuda') != torch.device('cuda:0')`, and every byte count in the
lane compared devices with `==`.**  A tensor moved to `cuda` reports `cuda:0`,
so `LayerWeights.device_bytes(device)` and `CoresetFold.device_resident_bytes`
counted ZERO bytes for a layer that was entirely resident whenever the caller
spelled the device without an index — which is exactly what `--device cuda` on
the command line does.  That put the v3 fix above (the 29.5x resident
under-pricing) straight back in place, silently: an estimate that under-counts
does not raise, it ADMITS the rung and lets the allocator do the raising later.
The same alias exists on the host (`cpu:0` vs `cpu`; a tensor built on `cpu:0`
reports `cpu`).

Fixed in `k3_common.py` with two functions used everywhere a device is compared
or stored: `canonical_device` (a `torch.device` with the index spelled out —
`cuda` with no index resolves to the CURRENT device, not device 0; `cpu`/`meta`
have any written index dropped, because tensors on those never carry one) and
`same_device`.  Call sites changed: `LayerWeights.device_bytes`
(`k3_layer.py:118`), `CoresetFold.__init__` and `CoresetFold.device_resident_bytes`
(`c14_union_coreset.py:618`, `:663`), `default_device_budget_gb`
(`c14_union_coreset.py:302` — it also read `device.index or 0`, i.e. card 0 for
an unindexed `cuda`, which sizes the budget off the wrong card on a mixed box),
and the `--device` resolution in `c14_union_coreset.py`, `k3_stream.py`,
`k3_review_stream.py`, `k3_receipt.py`.  Those four receipts now record the
RESOLVED device in `device` / `experts_device` and keep the literal flag in
`device_requested` / `experts_device_requested`.

`t13f` is the regression test.  Its host arm needs no GPU and fails on the
pre-fix code — negative control run on a pre-fix copy of the harness with only
the new test and the new helpers dropped in:
`AssertionError: ({'attention': 0, 'residual': 0, 'body': 0, 'experts_packed': 0, 'total': 0}, {... 'total': 122832})`.
It also asserts `raw_equality_hits_on_the_alias == 0`, so the test cannot go
vacuous on a future torch that starts indexing CPU tensors, and its CUDA arm
repeats the whole check where the bytes actually cost VRAM.

## What the second adversarial review caught, and what changed (v3)

1. **`device_memory_estimate` priced ~4% of what landed on `--device`, and the
   docstring claimed the opposite.**  `main()` called
   `load_layer(index, layer, args.device, args.device)`, which puts all 896
   PACKED experts on the compute device — 14.80 GB of `weight_packed` plus
   0.93 GB of E8M0 scales — and then `CoresetFold` moved the calibration cache
   there, `ExpertBank` the kept rows, `score` its held-out buffers and
   `--form-m` a [65536, 7168] fp32 lift.  None of it appeared in the estimate,
   which returned ~0.9 GB.  The README's own K = 65,536 entry point was
   therefore told the rung cost 0.9 GB, admitted by a 22.77 GB budget, and died
   in the allocator `enforce_budget` exists to prevent.  Fixed three ways:
   `LayerWeights.device_bytes` MEASURES what is resident per device;
   `device_memory_estimate` takes a `resident` breakdown and adds it to the
   transient peak (with `expert_bank_bytes` / `score_buffer_bytes` /
   `form_m_bytes` as named, testable arithmetic); and `--experts-device` now
   defaults to `cpu`, so the packed experts stay on the host and only the rows a
   rung keeps are moved.  The printed `per_rung_device_estimate` is the same
   object the guard enforces, so the table and the refusal cannot disagree.
   `t13d` checks that the old number is admitted and the new one refused on a
   25.3 GB card, and that the refusal names the resident terms; `t13e` checks the
   placement.
2. **Every RMSNorm gain in both fixtures was `torch.ones`, so no test could
   detect which norm the harness reads.**  K3 has six distinct gains per layer
   and `StreamedLayer.forward` wires four of them by name; with unit gains you
   could swap `self_attention_res_norm` with `mlp_res_norm`, or `input_layernorm`
   with `post_attention_layernorm`, or point `moe_forward` at the wrong norm, and
   t05/t08/t09/t10/t10b/t12/t12b/t17 and both bs3 receipts still reported
   bit-exact block stacks and fp32 relL2 ~1e-7 — the repo module was loaded from
   the same all-ones state dict.  Fixed: `k3_synth.norm_gain` gives every gain a
   distinct `N(1, 0.1)` vector, and `t05b` asserts the gains are pairwise
   distinct, none is a unit vector, and each of nine mutations MOVES the layer
   (0.287 / 0.250 / 0.212 / 0.114 / 0.111 / 0.0646 / 0.0024 on the gains that sit
   on the hidden stream; 8.9e-5 on `q_a_layernorm`, which acts inside a rank-24
   bottleneck — for that one the claim is only that the harness reads it, and at
   fp32 any nonzero movement is that proof).  **Every number in this README was
   re-measured against the new fixtures.**
3. **`--emit-baseline` would write a FOLDED run out as a "parent", and the
   consuming run recorded nothing about what it compared against.**  One
   forgotten flag — copy the child command line, leave `--dense-dir`/`--rung` on —
   and `parent_logits.safetensors` is a rung-65,536 child;
   `_check_requested_folds` is satisfied (folds WERE found), the next run scores
   the 17,408 child against it and reports a far better child than the truth,
   with nothing in `review_child.json` to betray it.  Fixed: `baseline_provenance`
   runs beside the decode refusal, before any weight read.  Emitting refuses while
   any layer is folded; the emitted safetensors carries a `__metadata__` stamp;
   `--baseline` reads that stamp back, records `baseline_path`,
   `baseline_sha256`, `baseline_layers_folded` and `baseline_is_stamped` in the
   receipt, and refuses a folded baseline.  `--allow-folded-baseline` is the only
   way through and is recorded either way; an unstamped baseline is reported as
   unstamped rather than assumed innocent.  `--emit-baseline` in `--mode decode`,
   which silently wrote nothing, is now refused.  `t16c`.
4. **The deliverable directory did not contain the receipts its own README's
   headline claims rested on, and the test counts were wrong.**  Four receipts
   (`k3_synth_bs3_L3/L5_exactness_receipt.json`,
   `k3_gram_memory_probe_receipt.json`, `k3_layer3_shapes.json`) existed only on
   the server; the README said "23 tests" and "the 21/21 line" while the shipped
   receipt said 25.  Fixed: all nine receipts (ten since v3.1) are in this
   directory and listed above, every count matches the shipped receipt (**29**
   at v3, **30** since v3.1 added `t13f`), and this file is re-synced so the
   local and server copies hash-match.

Also in v3: `process_peak_rss_record` renames `measured_peak_rss_gigabytes` to
`process_peak_rss_so_far_gigabytes` with a note, because `VmHWM` is a
process-lifetime high-water mark and rungs run largest-first in one process, so
every rung after the first inherited the first one's peak (0.570 → 0.572 → 0.572
in this run is one measurement and two echoes); and `scan_fold_dir` parses the
layer index as however many digits it has instead of `name[1:3]`, which silently
dropped any three-digit layer.

## What the earlier adversarial review caught, and what changed (v2)

1. **The whole-layer receipt never ran the residual branch the real model takes.**
   `k3_synth.py` defaulted `--attn-res-block-size` to 1 and the test harness did
   not forward the flag, so `layer_idx % block_size == 0` held for every fixture
   layer and `StreamedLayer.forward` always took the snapshot branch.  The line
   `carried = prefix_sum + attn_out` — which **85 of K3's 93 real layers**
   execute, since `attn_res_block_size = 12` — had zero reference evidence.
   Fixed: the default is now 2, the test forwards every shape knob, and a second
   6-layer / block-size-3 fixture runs the whole-layer comparison on **both** a
   snapshot layer (idx 3, 1 block on the stack) and a non-snapshot layer
   (idx 5, **2** blocks).  `blocks_before()` replaces the `layer_idx // B`
   snapshot count, which was only correct at `B = 1`.
2. **`--mode proxy` was not streamed.**  `ReviewModel.load()` read all 93 layers
   and held them.  An unfolded K3 MoE layer is ~15.7 GB packed, so 92 of them is
   ~1.44 TB against 251 GiB — i.e. the parent baseline (`--emit-baseline`, no
   folds) and every N below "all MoE layers folded" could not run at all, which
   is most of what the instrument is for.  Fixed: `ReviewModel.stream_proxy`
   holds one layer at a time and carries only the per-case residual pair; the
   loop is layer-major so each layer is still read once per invocation, not once
   per case.  `t12b` checks the streamed answer equals the resident one.
3. **The decode refusal was unreachable.**  It ran after `load()`, so on the real
   model the process OOMed during the load the guard existed to prevent.  Fixed:
   `ReviewModel.plan()` computes the fold/MoE census from the index alone and the
   refusal fires before any weight read; `t16` asserts nothing logged a load.
4. **The C14 refit's memory was understated ~3x and the top rung was unrunnable.**
   `gram + lam * torch.eye(n)` materialised a dense fp64 identity plus a summed
   copy; three Grams and three cross matrices were accumulated; and all nine
   ladder solutions were held at once.  Fixed: clone + in-place diagonal add,
   two accumulated Grams with the full-fit system derived by addition (the splits
   are disjoint), and a three-pass ladder that keeps only scalars until the one
   solution it returns.

Also in v2: `veto_binding` / `chosen_is_largest_lambda` /
`check_sse_monotone_decreasing_in_lambda` in every ridge receipt, so a
no-signal fold (the answer is the prior, shrunk) cannot be read as a healthy one;
contiguous-block holdout (`--holdout blocks`, the new default) instead of
`arange % 3`; a continuation-only agreement alongside the all-positions one;
`mla_forward` refuses a chunked prefill rather than running silently
non-causally; `import_k3_modeling` refuses a `--src-dir` inside the model
directory; `k3_stream` writes an end-of-window snapshot so the *next*
`--layers` range can resume and a resumed pass carries the earlier segments'
per-layer records forward instead of overwriting them; `read_keys` reports exact
bytes as well as rounded GB; and `parameter_counts` reports the E8M0 scale
elements it excludes.

## Honest limits

* Every number above is from the **synthetic** checkpoints (random weights) except
  the shape census.  The C14 quality numbers on them (held-out y_lat relL2
  0.8817 / 0.8962 / 0.9215 at 4x / 8x / 16x) mean nothing — random experts have no
  coreset structure.  What is verified there is the *mechanics*: verbatim rows,
  mass allocation, the lambda ladder, the `update_scale <= 1` veto, exact
  parameter accounting and monotonicity in width.  The stage-1-only comparison at
  the same rung (`k3_synth_v3_L01_c14_stage1only_receipt.json`: 0.8932 against the
  two-stage 0.8962) is likewise mechanics, not evidence about real K3.
* No real K3 layer has been run through this harness yet: layer 1 costs 16.99 GB
  at the measured 5.3 MB/s CIFS rate.  Re-measure the share before budgeting.
* The C14 HOST memory table is **arithmetic** at real K3 widths.  It is anchored
  by one real measurement (K = 8,192 / out = 3,584: 3.814 GB peak, 1.956x better
  than the pre-fix path) and the estimator is checked against the table by
  `t13c`, but K = 32,768 (estimated 38.1 GB) and K = 65,536 (estimated 145.0 GB)
  have **not** been run.  The estimator is also not calibrated against that one
  anchor: for the K = 8,192 problem it predicts 3.087 GB against a measured
  3.814 GB, and nothing asserts the gap.  Measure 32,768 before attempting 65,536.
* `gram_memory_estimate` still omits the prior copies (~4.7 GB unpriced at
  K = 65,536 against a 145.0 GB estimate).  That is a HOST term; the v3 fix was
  to the DEVICE budget and did not touch it.
* Stage-2 selection at the top two rungs is still impractical — the `[pool, k]`
  fp64 Cholesky factor is 68.7 GB at pool 131,072 / k 65,536 and the pivot loop is
  65,536 Python iterations.  `--k-pool-factor 1` (stage-1 only) is the policy
  there, and the two-stage-vs-stage-1 comparison R6 asks for at K = 32,768 is
  still unpriced.
* `--mode decode` on the real model needs every MoE layer folded; the runner
  refuses otherwise rather than pretending, and refuses before reading.
* The key-dim `A_log` broadcast (D-a2b) still has **no independent reference**.
  `t06`'s key-dim comparison is against a restatement of our own formula and the
  GPU receipt compares our-fla against our-torch; the 30/30 line does not cover
  which broadcast is semantically correct.
* `chunk_kda` has still only been exercised at head_dim 64 / 2 heads / 128 tokens,
  never at K3's 96 x 128 geometry, and the `cu_seqlens` varlen path is written but
  never executed.  No tokenizer: the calibration corpus is random ids or a
  caller-supplied `input_ids` safetensors.
* `k3_layer.py` still caches `router_topk_w` at bf16 while the fold's target
  `y_lat` is fp32; that ~0.4% relative error propagates into the selection mass
  and the prior scale, and is absorbed by the ridge rather than measured.
* `c14.score()` evaluates the child with an fp32 `w2p` while the installed dense
  path casts it to the activation dtype; `stored_vs_scored_dtype` says so, but
  the held-out relL2 in the receipt is not the installed child's number.
* `k3_stream --resume` overwrites `input_ids_sha256` on a resumed pass even
  though the embed step is skipped, so a resumed receipt's token provenance can
  disagree with the cached activations.  Nothing validates the snapshot against
  the ids.
* Form M's `W2` is still in the latent space (approximating `routed_expert_norm`
  in the merged form is not implemented); form E is the scoring form per D-b2.
