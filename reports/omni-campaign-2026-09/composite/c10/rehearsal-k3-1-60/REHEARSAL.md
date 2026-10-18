# C10 partial-K3 union-install rehearsal (M4, 2026-10-10)

Author: Claude (M4 lane). Server `ai`; `$OC` = `/home/cereal/omni-campaign`; rehearsal dir `R = $OC/composite/c10/rehearsal-k3-1-60` (mirrored here without the safetensors). Every number below is a DIAGNOSTIC (owner policy 2026-10-06), except the structural, load, identity and size checks.

**Status (2026-10-10 ~21:15Z):** this is a PREFIX rehearsal. Trunk layers 0-26 were solved at the budgeted width and persisted. They carry K3 MoE 1-38 (16 one-K3 layers and 11 two-K3 layers). Trunk 27-41 (K3 39-60) and the K3-free trunk 42-63 were NOT run.

The run was stopped on purpose after layer 26 (`R/stop_after_26.sh`, SIGTERM at 20:12:46Z after L26.json existed). At the measured pace on the shared box (about 50-75 min per 3-layer group), layers 27-41 needed about 5 more hours and 42-63 about 3 more. No child was written. The load / identity / forward check ran on a labelled HYBRID child: union layers 0-26 plus the trunk FFN zero-padded on layers 27-63 (section 4).

Resume command (same inputs, verified by `partial_inputs.json`): `cd R && STOP=63 setsid nohup sh launch.sh --resume > logs/c10_rehearsal_resume2.log 2>&1 < /dev/null &`. It needs CUDA in a NEW process, which is currently broken (see below).

**Host blocker found during the run:** at about 12:31Z dpkg upgraded the NVIDIA user-space driver to 580.178.04 (`/var/log/dpkg.log`) while the loaded kernel module is still 580.173.02.
- Since then `nvidia-smi` prints `Failed to initialize NVML: Driver/library version mismatch`, and every NEW process gets `torch.cuda.is_available() == False`.
- Processes that already held a CUDA context (this rehearsal, K3) kept running.
- No GPU job (this resume, the full run, c17 42-63, media captures) can start until the module is reloaded or the box is rebooted. That is an owner/root action; I did not attempt it.

## 1. What was rehearsed

C10 (run-book S5) installs K3 FFN rows into the S3 trunk's MLPs. Each trunk layer k gets `ffn.up` = `[gate_q | gate_k3]`, `[up_q | s * up_k3]` (the trunk's 4,352 Qwen rows verbatim plus 12,288 K3 rows chosen by a union coreset), and `ffn.down` = a prior-anchored ridge refit against `MLP_parent(x) + sum_l FFN_k3,l(T x + b) U^T` on the trunk's own FFN-input rows. The trunk is S3 native; the parent targets come from PROBE.S0; the charts are C17 ridge L00-L41.

| input | value |
|---|---|
| trunk | `/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native` (14,864,623,104 header params, Q = 4,352, chimera_commit ef1ffb9) |
| parent targets | `$OC/composite/s0/q38-27bu-chimera-bf16-sep04` |
| K3 | MoE layers 1-60 via `R/k3_dense_1_60/Lxx/ffn.safetensors`, read-only symlinks to `$OC/k3-run-oct09-claude/fold/dense/Lxx/ffn.safetensors`; dense L00 excluded (the c17 alignment skips it) |
| K3 config | `/mnt/lexar/models/moonshotai/Kimi-K3/config.json` |
| charts | `$OC/composite/c17/charts` L00-L41, arm `ridge`, `--chart-bias-ceiling 0` (refusal off; recorded) |
| alignment | `R/alignment_override_k3_1_60.json` (sha `28e19245...6c6d`): trunk 0-41 = the c17 `alignment.json` rows (sha `4006c179...8c8f`); trunk 42-63 = [] (18 trunk layers take two K3 layers, 24 take one, 22 take none) |
| evidence | `$OC/composite/corpus/omni-evidence-27b-v1.json` (manifest `8a3422d8...4870`): 24 x 4,096 tokens, 73,728 fit / 24,576 held rows per layer |
| `--k3-channels` | 12,288 (the run-book value; it is 250 below the E4 maximum of 12,538 that still fits the five media units) |
| size gate | `--parent-parameter-count 27781427952` (ledger PARENT-Q38-27BU). Without it the tool would use the S0 headers (27,699,193,344) |
| tree | `CHIMERA_SRC=$OC/import/rq27b/tree_ef1ffb9/src` (the tool's default is tree_c5e6052) |
| tool | `$OC/composite/tools/c10_union_install.py` 1.3.0, sha `eb766ab389b0b18b1c9c8d7793305c9efe514802946aab885969052f603e952c` |
| launch | `R/launch.sh` (sha `fb3db725...4634`): `STOP=<last layer> sh launch.sh --overwrite` (fresh) or `--resume`; GPU4 via `CUDA_VISIBLE_DEVICES=4`, nice 19, OMP/MKL/OPENBLAS 8, `--chunk 8192`, `--selection-device cuda:0 --solve-device cuda:0` |

Disk rule: the written child would need 54.4 GB plus the 32.7 GB partial beside it, about 87 GB on root, and root had 114-139 GB free while K3 L61-92 still writes about 40 GB. So the child is NOT written. The run stops after layer 63 (`--stop-after-layer 63`, exit 3) and the child the partial defines is checked by `c10_virtual_child_check.py` (section 4). The partial files for layers 42-63 are symlinked to `/mnt/lexar/omni-campaign/c10-rehearsal-k3-1-60-partial-overflow/` (11.2 GB, inside the 15 GB Lexar allowance). `R/watchdog.sh` SIGTERMs the run if root drops below 62 GB or MemAvailable below 16 GB; the persisted layers survive a TERM and `--resume` continues.

## 2. Tool changes made during the rehearsal (c10 1.1.0 -> 1.3.0)

| version | change | why (measured) |
|---|---|---|
| 1.1.1 | `StreamedEngine.block()` keeps the constructor's fp32 RoPE `inv_freq` across the dtype move | Under tree_c5e6052 and tree_11adb32 (no `GatedGQAAttention._apply` guard) 1.1.0 rounded the RoPE table to bf16. The red test on c5e6052 failed with `assert torch.bfloat16 == torch.float32`. Under ef1ffb9 the `_apply` guard already kept it fp32, so the rehearsal inputs were never affected; this protects the tool's default tree. |
| 1.2.0 | `--selection-device`: `pivoted_columns_gram_on()` runs c14's greedy pivoted Cholesky with the factor on the GPU (same pivots, same 1e-30 stop, same -1 exclusion); `phase_seconds` per layer | Attempt 1 (stock 1.1.1) spent >45 min of layer 0 in c14's Python gemv loop on about one core. Scratch benchmark: the host c14 took 52.15 s at n=6,000/k=3,000 vs 23.06 s on cuda, with identical selection. At n=23,552/k=12,288 cuda took 114 s, extrapolated host about 57 min (single K3 layer) and about 115 min (two K3 layers). |
| 1.3.0 | `--solve-device`: the 19-rung fp64 ridge ladder (c14.law_abiding_solve_gram) runs on the GPU when it fits, otherwise on the host (recorded). Gram-defect reports always use a host copy | Attempt 2 (1.2.0, `--chunk 1024`) layer 0 took 2,869 s: pool Gram 1,112 s, ridge 1,185 s (host fp64 BLAS at 8 threads), check Gram 312 s, selection 111 s (pivot 27.7 s). In the same window K3 L76/L77 took 2,914 s and 2,861 s, against 1,416-1,750 s for L73-L75. Attempt 3 (1.3.0, `--chunk 8192`) layer 0 took 491 s: pool Gram 87, ridge 212 on cuda, check Gram 27, selection 64 (pivot 22). K3 L79/L80 ran at 1,547/1,538 s alongside it. |

`--chunk 8192` is a flag, not a code change. It cuts the host fp64 accumulation 8x, because the host adds one 16,384^2 block per chunk.

Tests: `test_c10_union_install.py` gained `test_streamed_engine_keeps_fp32_rope_tables_under_bf16`, `test_device_pivot_matches_c14_pivot`, `test_selection_records_the_pivot_and_phase_seconds`, `test_solve_device_cpu_is_recorded` and `test_cuda_selection_and_solve_match_the_host_child` (CUDA only). The last one requires the same kept channels and lambda on every layer, and weights within rtol 1e-5. Server results (tree ef1ffb9):
- CPU suite: `27 passed, 1 skipped in 83.01s` (c10 plus the vcheck suite, CUDA test skipped)
- CUDA subset on GPU0: `3 passed, 22 deselected in 13.15s`

## 3. Per-layer union receipts (diagnostics)

Full table: `R/rehearsal_layers.json` (from `R/rehearsal_summary.py`). Raw records: `R/child.partial/Lxx.json`.

| trunk k | K3 layers | pool | held union | held trunk | held prior | K3E held | free-solve upd | lambda | update_scale | veto | out-bias E | in-shift | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 23552 | 0.1623 | 0.6413 | 0.3759 | 0.6252 | 1.13 | 0.1 | 0.309 | T | 0.879 | 0.452 | 491 |
| 1 | 2 | 23552 | 0.2193 | 0.8793 | 0.6055 | 0.8621 | 1.19 | 0.1 | 0.370 | T | 0.875 | 0.743 | 574 |
| 2 | 3,4 | 47104 | 0.1327 | 0.9495 | 0.7003 | 0.9472 | 1.90 | 0.01 | 0.961 | T | 0.668 | 0.333 | 792 |
| 3 | 5 | 23552 | 0.1762 | 0.8802 | 0.7581 | 0.8385 | 1.70 | 0.01 | 0.427 | T | 0.088 | 0.272 | 544 |
| 4 | 6,7 | 47104 | 0.2330 | 0.9935 | 0.8395 | 0.9883 | 4.13 | 0.01 | 0.861 | T | 0.170 | 0.304 | 1167 |
| 5 | 8 | 23552 | 0.2124 | 0.9948 | 0.7090 | 0.9908 | 2.88 | 0.01 | 0.730 | T | 0.081 | 0.310 | 509 |
| 6 | 9,10 | 47104 | 0.3257 | 0.9921 | 0.9772 | 0.9865 | 6.25 | 0.1 | 0.893 | T | 0.043 | 0.007 | 771 |
| 7 | 11 | 23552 | 0.2255 | 0.9887 | 0.9104 | 0.9785 | 2.99 | 0.1 | 0.448 | T | 0.004 | 0.239 | 591 |
| 8 | 12 | 23552 | 0.1317 | 0.9921 | 0.7684 | 0.9844 | 2.21 | 0.001 | 0.530 | T | 0.096 | 0.696 | 475 |
| 9 | 13,14 | 47104 | 0.2545 | 0.9579 | 0.9199 | 0.9191 | 3.23 | 0.1 | 0.850 | T | 0.101 | 0.179 | 1157 |
| 10 | 15 | 23552 | 0.2316 | 0.8996 | 0.8645 | 0.7765 | 1.78 | 0.01 | 0.768 | T | 0.055 | 0.136 | 486 |
| 11 | 16,17 | 47104 | 0.2684 | 0.9658 | 0.9424 | 0.9266 | 2.96 | 0.1 | 0.531 | T | 0.157 | 0.014 | 1124 |
| 12 | 18 | 23552 | 0.2407 | 0.9160 | 0.8426 | 0.7767 | 1.62 | 0.01 | 0.667 | T | 0.049 | 0.009 | 505 |
| 13 | 19,20 | 47104 | 0.2368 | 0.9929 | 0.9622 | 0.9808 | 4.10 | 0.1 | 0.575 | T | 0.079 | 0.012 | 840 |
| 14 | 21 | 23552 | 0.2009 | 0.9813 | 0.8859 | 0.9538 | 2.41 | 0.01 | 0.617 | T | 0.019 | 0.026 | 580 |
| 15 | 22,23 | 47104 | 0.1712 | 0.9927 | 0.7945 | 0.9815 | 2.86 | 0.01 | 0.866 | T | 0.080 | 0.017 | 1254 |
| 16 | 24 | 23552 | 0.2745 | 0.7736 | 0.7663 | 0.3215 | 10.78 | 0.1 | 0.841 | T | 0.327 | 0.005 | 449 |
| 17 | 25 | 23552 | 0.2272 | 0.7054 | 0.6913 | 0.3781 | 7.36 | 0.1 | 0.982 | T | 0.274 | 0.004 | 573 |
| 18 | 26,27 | 47104 | 0.2607 | 0.7397 | 0.7243 | 0.6278 | 6.40 | 1 | 0.469 | T | 0.508 | 0.002 | 1213 |
| 19 | 28 | 23552 | 0.2772 | 0.7196 | 0.7072 | 0.4524 | 6.69 | 1 | 0.493 | T | 0.341 | 0.002 | 592 |
| 20 | 29,30 | 47104 | 0.2780 | 0.9143 | 0.8581 | 0.8419 | 4.88 | 1 | 0.624 | T | 0.549 | 0.003 | 1147 |
| 21 | 31 | 23552 | 0.2327 | 0.9119 | 0.8481 | 0.7485 | 2.85 | 0.1 | 0.854 | T | 0.149 | 0.004 | 465 |
| 22 | 32,33 | 47104 | 0.2166 | 0.9420 | 0.8944 | 0.8555 | 2.67 | 0.1 | 0.822 | T | 0.325 | 0.002 | 789 |
| 23 | 34 | 23552 | 0.2404 | 0.9522 | 0.8923 | 0.8523 | 2.46 | 0.1 | 0.748 | T | 0.129 | 0.003 | 1487 |
| 24 | 35 | 23552 | 0.1766 | 0.9827 | 0.7888 | 0.9376 | 2.29 | 0.1 | 0.645 | T | 0.063 | 0.003 | 1380 |
| 25 | 36,37 | 47104 | 0.3586 | 0.8279 | 0.8208 | 0.2959 | 15.38 | 1 | 0.522 | T | 1.612 | 0.003 | 2927 |
| 26 | 38 | 23552 | 0.3269 | 0.8136 | 0.7898 | 0.3509 | 9.35 | 1 | 0.537 | T | 0.686 | 0.002 | 1332 |

Column meanings:
- `held union`: relL2 of the shipped bf16 union FFN against Y on the 24,576 held rows.
- `held trunk`: the unchanged S3 FFN against the same Y.
- `K3E`: `||K3 term|| / ||Y||` on held rows.
- `veto`: ridge veto (the free solve wanted `update_scale > 1`) OR K3 energy below 1e-6.

Findings (all diagnostics):
- **Veto binds on all 27 layers**, every time through the ridge veto: the free solve wanted `update_scale` 1.13-15.38 against the prior (median 2.88). No layer is "function absent". The cast weights land at update_scale 0.309-0.982 at lambda 1e-3 .. 1. By the run-book S5 wording, "veto binds in most layers" triggers the frontier rule (delete + rebuild with more paired rows, a smaller `--k3-channels`, or the C13 mixer rung). That rule is the owner's call and does not gate this rehearsal.
- **The K3 term dominates the targets.**
  - `||sum FFN_k3(T x + b) U^T|| / ||Y||` on held rows is 0.296-0.991 (mean 0.784). It is >= 0.90 on 12 layers (trunk 2, 4-9, 11, 13-15, 24).
  - So Y is mostly the charted K3 write, not the parent MLP. The unchanged S3 FFN scores held relL2 0.64-0.99 against Y (mean 0.900).
  - The union FFN scores 0.132-0.359 (mean 0.233). It beats trunk-only on all 27 layers, and beats the no-repair prior (mean 0.801) on all of them.
- **Chart biases are not small.**
  - The excluded output bias C_in_bias carries energy fraction 0.004-1.612 of Y (median 0.149). It is above 0.1 on 16 of 27 layers and above 1.0 on trunk 25.
  - The dropped input-bias shift is 0.002-0.743 (median 0.012), and above 0.1 on 10 layers (trunk 0-5, 7-10).
  - With the default `--chart-bias-ceiling 0.1` the tool would have refused at layer 0. `--chart-bias-ceiling 0` was used as instructed, and the numbers are recorded per layer.
- SiTU -> SiLU per-channel scalar cost: relL2 at most 0.196 (any source, any layer).
- The coreset keeps almost only routed channels: shared-expert channels get 0-77 of the 12,288 per layer (median 4).
- Cost (1.3.0, `--chunk 8192`, shared box, not a quiet-machine measurement):
  - One-K3 layers: median 558 s (449-1,487).
  - Two-K3 layers: median 1,147 s (771-2,927).
  - Per-phase medians, one-K3: ridge 212 s on cuda, pool Gram 113, selection 73, scores 42, targets 47, check Gram 34, bias+SiTU 33.
  - Per-phase medians, two-K3: pool Gram 355, selection 267 (host Schur + host-gathered GPU pivot), ridge 212.
  - Each 3-layer group also pays about 18 min of streamed KDA capture + advance over 98,304 tokens.
  - Peak RSS 130.4 GB (`/usr/bin/time` max RSS 127,348,952 kB). It is driven by the 51,456-wide fp64 pool Gram plus its Schur complement on two-K3 layers.

## 4. Load / identity check (`c10_virtual_child_check.py` 1.1.0)

The check reconstructs the child from the trunk headers plus the partial FFNs without writing it:
- every partial file hashes like its record;
- the Qwen gate/up rows are byte-identical to the trunk's, and the zero-pad rows are exact zeros;
- the header count = trunk - trunk FFN + partial FFN;
- `ChimeraOmniModel(child_cfg)` on meta names exactly those 1,011 tensors with those shapes;
- a layer-major streamed bf16 forward of the S3 trunk and the child on the 8 R-Q prompts (`tree_ef1ffb9/scripts/import_receipt_qwen35.PROMPT_PACK`, tokenized with the S3 HF dir) reports finiteness, top-1 agreement, KL(trunk||child) and per-layer residual relL2.

Synthetic proof (test `test_virtual_child_equals_the_written_child`): the virtual child's logits are `torch.equal` to `ChimeraOmniModel` on the child that c10 `--resume` writes. The virtual accounting also equals oc_seal on that written child.

Real-trunk smoke (`/home/cereal/c10-rehearsal-oct10/vcheck_smoke.py`, child := trunk, GPU0): top-1 1.0, KL 0.0, residual relL2 0.0, finite, 322 s, peak CUDA 7.26 GB.

Results on the real S3 trunk. Both checks used child header 26,944,218,624 = c10's arithmetic, with headroom 837,209,328 vs 27,781,427,952 and 245,772,199 after the five media units (max `--k3-channels` with the units = 12,538). The meta `ChimeraOmniModel(child_cfg)` matched all 1,011 tensors (`strict_load_equivalent true`).

| check | union layers | device/dtype | identity rows | logits finite | top-1 vs S3 trunk | KL(trunk‖child) mean | residual relL2 vs trunk | receipt |
|---|---|---|---|---|---|---|---|---|
| hybrid L00-02 | 0-2 (trunk on 3-63) | cuda bf16 (GPU0, before the NVML break) | 3/3 verbatim + exact pads | yes | 117/210 = 0.557 | 0.816 | 0.70 @L2, 0.93 @L3, 0.54 @L63 | `R/virtual_child_check_hybrid_L00-02.json` sha `9834f177...a93` (938 s) |
| hybrid L00-26 | 0-26 (trunk on 27-63) | cpu fp32 (no CUDA for new processes) | 27/27 verbatim + exact pads | yes | 7/210 = 0.033 | 5.748 | 0.70 @L2, 0.99 @L3, 9.00 @L8, 5.53 @L26, 1.28 @L63 | `R/virtual_child_check_hybrid_L00-26_cpu_fp32.json` sha `8bcf78b5...954f` (2,975 s, max RSS 79 GB) |

What this says (diagnostic, not a gate):
- The union child loads at the uniform 16,640 width, carries the trunk's rows byte for byte, sits under the parent budget, and produces finite text logits.
- But installing the C10 union FFNs on trunk 0-26 moves the residual stream far from the trunk's: `||x_child - x_trunk|| / ||x_trunk||` reaches 9.0 at layer 8. The child's last-token predictions on all 8 R-Q prompts collapse onto punctuation/space ids (11, 13, 220, 506), where the trunk predicts 198/220/271/727/1358/13120.
- This is what the target definition asks for: Y = parent MLP + the charted K3 FFN output, and the K3 part carries 30-99 % of Y's energy. Whether the composite should add the K3 write at full scale is an owner/run-book question (S5 design), not a fit failure.

## 5. Repeat for the full K3 (MoE 1-92) when it lands

Template: `R/launch_full_k3_1_92.sh`. It is the same command with `--k3-dense $F/k3_dense_1_92 --k3-layers 1-92`, no `--alignment` (the c17 `alignment.json` covers the 92 MoE layers exactly once), and `--out $OC/composite/c10/full-k3-1-92/child --receipt $OC/composite/s5/union_receipt.json`.

Pre-flight (read-only):

```
OC=/home/cereal/omni-campaign
for l in $(seq 61 92); do test -f $OC/k3-run-oct09-claude/fold/dense/L$l/done.json || echo "missing K3 L$l"; done
python3 -c "import json;r=json.load(open('$OC/composite/c17/charts/receipt.json'));print(len(r['layers_present']),r['layers_invalid'])"   # want 64, []
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4                                            # < 2000 MiB
df -h / ; grep MemAvailable /proc/meminfo                                                                      # root >= 33 GB + 60 GB floor; >= 140 GB RAM free
```

Per-layer pass (detached; resumable):

```
cd $OC/composite/c10 && mkdir -p full-k3-1-92/logs && cp rehearsal-k3-1-60/launch_full_k3_1_92.sh full-k3-1-92/ && \
  STOP=63 setsid nohup sh full-k3-1-92/launch_full_k3_1_92.sh --overwrite > full-k3-1-92/logs/c10_full.log 2>&1 < /dev/null &
# after any interruption:  STOP=63 setsid nohup sh full-k3-1-92/launch_full_k3_1_92.sh --resume > full-k3-1-92/logs/c10_full_resume.log 2>&1 < /dev/null &
```

Load / identity check on the partial (GPU0 or GPU4 < 2000 MiB; about 16 min, about 65 GB host RAM): the same `vcheck_launch.sh` with `R` pointed at `full-k3-1-92`, and without `--pad-missing-with-trunk`.

Child write (only when the `--out` filesystem has about 54.4 GB more beside the 32.7 GB partial above the 60 GB floor): `sh full-k3-1-92/launch_full_k3_1_92.sh --resume`, with `STOP` unset. The run then reloads all 64 layers, advances the evidence through them, writes `child/`, re-checks the header count against `--parent-parameter-count`, removes the partial and writes `union_receipt.json` plus the `$OC/composite/s5/` copy.

Expected header count: 14,864,623,104 + 64 x 3 x 5,120 x 12,288 = 26,944,218,624.
- Headroom vs 27,781,427,952: 837,209,328.
- Headroom after the five media units (591,437,129): 245,772,199.

Cost forecast for the full 1-92 run (64 K3-carrying trunk layers: 28 two-K3, 36 one-K3), using the rehearsal medians:
- Layers: 36 x 558 + 28 x 1,147 s = about 14.5 h.
- Streamed capture + advance: 22 groups x about 18 min = about 6.6 h.
- Initial hashing: about 12 min.
- Total about 21 h on the box as loaded today. Unmeasured on a quiet box.
- Peak host RSS about 130 GB and GPU about 17 GB on a 3090.

Two cheaper routes, neither implemented:
- c10 cannot reuse the rehearsal's layers 0-26 for the 1-92 run. Their per-layer inputs are identical (same trunk, parent, evidence, charts L00-26, K3 L01-38, k3_channels, chunk, group size). But `partial_inputs.json` seals the whole run's k3_layers / alignment / charts, so `--resume` refuses. A per-layer adoption mode would save about 9 h.
- Holding the two-K3 Gram (21.2 GB fp64) on GPU6 (RTX 8000, 48 GB, free since K3 exited 17:25:55Z) would remove most of the host Gram/Schur time.

## 6. Not verified

- The full rehearsal (trunk 27-63) and the full c10 child: not run (planned stop, then the NVML mismatch). No `union_receipt.json` exists; only the per-layer `child.partial/Lxx.json` records. Layers 0-26 are persisted at `R/child.partial/` (27 x 511,181,048 B on root). The L42-63 overflow symlinks point at an empty `/mnt/lexar/omni-campaign/c10-rehearsal-k3-1-60-partial-overflow/`.
- The child's real load through `load_chimera_checkpoint`: only proven on the synthetic fixture (bit-exact vs the virtual child). On the real trunk only the virtual/hybrid check exists.
- Numerics:
  - The cuda pivot picked the same columns as c14 on a 6,000/3,000 random Gram and on the 4-layer fixture. On the real 23,552/47,104 pools it was not compared to the host path; the host path would take about 1-2 h per layer.
  - The cuda ridge was checked against the host only on the fixture (same lambda, weights within rtol 1e-5).
  - Layer 0 under 1.2.0/chunk 1024 vs 1.3.0/chunk 8192: same kept channels, held relL2 0.162345 vs 0.162347, different bytes.
- Timing figures are from a shared, loaded box (load average 24-38; other lanes' s7 jobs, pg_dump, K3). They are not quiet-machine measurements. The cost forecast is arithmetic from the medians.
- K3 slowdown attribution: L76/L77 took 2,914/2,861 s while my attempts 1-2 (host fp64 BLAS) ran alongside other lanes and a pg_dump. L79-L83 took 1,403-1,775 s alongside 1.3.0. I did not isolate my share.
- The 8 R-Q prompts are tokenized with the S3 HF tokenizer (tokenizer.json in `S3-C2K4352-sep07`). Ids above 248,320 never occur. That the native vocab maps these ids 1:1 is taken from the S3 exactness receipts, not re-proven here.
- The hybrid checks are hybrids. Their top-1/KL describe "union 0-k + trunk elsewhere", not the c10 child.
