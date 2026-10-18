# S7 readiness review, 2026-10-10 (adversarial pass over lanes M1-M5)

Reviewer: Claude (adversarial reviewer subagent). Server `ai`, `$OC=/home/cereal/omni-campaign`. Live server checks ran 21:31Z-22:10Z (server `date -u`). Owner policy applied: finite fit numbers are diagnostics only. Structural, load, identity, provenance and non-finite checks, plus the per-component >=4x rule, are functional.

## 0. Bottom line

- **The composite child does not exist yet.** Nothing in M1-M5 installs into a real trunk, and no lane claims it did.
- **Since the lane reports, K3 L61-92 has landed.** `k3_dense_pass exit=0 wall_seconds=75153 end=2026-10-10T17:25:55Z`. Receipt `k3-run-oct09-claude/receipts/k3_dense_pass_receipt.json` sha `8948bab7feea…`, which matches its sidecar. `layers_folded` has 92 entries (1..92). The end-of-stack held numbers are diagnostics: top-1 0.6921, KL 0.9996, NLL 2.157 folded vs 1.261 teacher, 23,180 positions.
- **C17 charts for trunk layers 42-63 are running now, and that run is not from any M lane.** Pids 289649/50/51 started at 17:49:38Z with `--threads 10` each and nice 10, so 30 BLAS threads in total. At 21:50Z, 14 of 22 layers are present (L42-L54 and L56). My estimate for the remaining 8 is about 2-3 h.
- **There are three hard blockers before any S7 command can run:**
  1. **Server GPU stack.** `nvidia-smi` prints `Failed to initialize NVML: Driver/library version mismatch` (NVML 580.178, kernel module 580.173.02). Torch reports `cuda_available False 0`. The fix needs root (reload the module or reboot), and it should wait until the c17 run above exits.
  2. **Interface set of record is undecided.** Four interface/C11 row sets exist, and the media charts are valid only under M1's `interface_latent`.
  3. **The media charts are bound to the S3 prefix.** C10 rewrites the FFNs in layers 0-31, so the host rows (x_31) have to be recaptured on the S5 child and the charts refitted.
- **No hidden fit-number gate was found in any new M1-M5 tool.** All their refusals are structural, sha or non-finite checks. Two existing numeric gates sit on the S7 path and conflict with the wording of the owner policy (section 3.2). Neither binds today.

## 1. Readiness table

S7 here means installing into the S5 (C10) child. "Unit" is the structural container.

| Item | Unit status | S7 status | Exact blocker / recorded limitation |
|---|---|---|---|
| **ACE** `ace/ace_dit_L12` | ready | **blocked** | Blockers: B1 GPU, B2 interface set, B3 host rows and charts recapture on S5. Limitations: parent context is seeded gaussian; trunk reads a fixed prompt; procedural clips. Unit facts: MLP 37,748,736 -> 9,437,184 (0.25); time gate 1.18e-07; charts 5120x2048, finite, sha `ae084950…` verified. |
| **H3** `h3/h3_dit_L25_curve6` | ready (needs c16 >= 1.2.0 and tree_ef1ffb9) | **blocked** | Blockers: B1, B2, B3. Limitations: no tool emits the `h3_thw_v1` source positions the unit's runtime contract requires; text (type 0) rows are read through an AV-only chart; write strength not measured; 24 fps assumed; video VAE round trip 9.91 dB. Unit facts: MLP 231,211,008 -> 57,802,752 (0.25); curve6 gate 1.3167e-07 at tol 1e-5; the 4 `H3_FP32_PARAMETERS` are F32 in the container. Both block files are symlinks into Codex's `composite/media/h3-native-curve6-sep07/`; mtimes are 2026-09-07, so nothing was written there. |
| **QIMG** `qimg/qimg_mmdit_L30_img` | ready | **blocked** | Blockers: B1, B2, B3. Limitations: **the parent is `qwen_image_fp8_e4m3fn.safetensors`, an FP8 distribution** (the M3 time-reference receipt names `/parent/qwen_image_fp8_e4m3fn.safetensors`; the unit records `source_dtype F8_E4M3/BF16 widened exactly to BF16`). Nothing was quantized by these lanes, but the source is a quantized checkpoint, not the BF16 original. Also: patch packing order was read from source, not hooked; text stream dropped; whole block 0.686. Unit facts: MLP 75,512,832 -> 18,874,367 (0.24995); FP32 time reference sha `d1ef9940…` matches the file; gate 1.168e-07. |
| **HY3D** `hy3d/hy3d_dit_L10` | ready-with-recorded-limitation | **blocked** | Blockers: B1, B2, B3, plus **D3 (Cyril): `attn2` cross-attention dropped**. **Contradiction between lanes:** M3's runtime contract `hy3d_time_token_row` (one type-7 time row first in each volume segment, paired with parent block-10 input row 0) is NOT what M1 fitted. `s7_pairs_parent.py` line 400 records `block_input[:, 1:] (row 0 = the time token, dropped)`, and `s7_host_rows.py` has no time-row emission. So the installed HY3D block runs without its time token: the E4 limitation is back, not fixed. Other limitations: 4 objects at 1 sigma; held forward cos 0.405. Unit facts: MLP -> 8,388,607 (0.24992); container F16. |
| **QVL32** `qvl32/qvl32_vit_L13` | ready | **blocked** | Blockers: B1, B2/D2 (C11 set of record), B3. Limitations: C11 rows are 2% (A) and 17% (B) of the random S0 `type_embed[6]` norm; 0.5/0.5 normalisation assumed; held forward cos 0.363; generic images, not OCR images. Unit facts: `type_ids [6]`; MLP width 4304 read from the header; MLP -> 2,479,027 (0.24985). |
| **C10** (S5 K3 union) | tool 1.3.0 tested | **blocked** | Blockers: c17 L42-63 must finish (running); B1 GPU; root disk is fine now (217 G free, peak need ~87 G). **D4 (Cyril):** the 0-26 hybrid collapses to top-1 7/210 vs the S3 trunk (KL 5.75; residual 9.0x the trunk norm at L8); the veto binds 27/27. **Must pass `--chart-bias-ceiling 0`**: the default 0.1 is a numeric refusal and would refuse at layer 0 (16/27 layers have bias energy above 0.1, max 1.612). |
| **C15** (5 latent interface rows) | built twice | **ready-with-recorded-limitation once D1 is decided** | **D1:** M1 `interface_latent` vs M2 `interface`. Same bands, different rows (feature_proj relL2 2.3-4.7 between them). The charts are valid only under `interface_latent`. All 13 C15 row/chart files I scanned are finite F32. Real S3 trunk type_affine preconditions are 0 non-zero (no refusal). Limitations: the "through interface" PSNR was not decoded; the H3 packer needs band offsets; bridge row norm is 71.55 against 0.93 for text embeddings (left alone). `COMPOSITE_S6_INPUT_FORMATS.md` still shows the raw `music_ace/linear` example and no latent rung, so the doc is stale. |
| **C11** (OCR rows) | 4 candidate sets | **blocked on D2** | Candidate sets: M1 `interface_latent/c11` (charts bound to this one), M2 `interface_c11`, M3 `c11/qvl32_patch_merger` (version A), M3 `c11/q38u_own_visual` (version B). All use band f[640,2176) e[384,385) type 6, and all files are finite. The band itself is the R0 proposal, not an owner decision. |
| **C18** (music_codes band) | built | **ready-with-recorded-limitation** | Table [65,536 x 5,120] F32, finite, sha `d9cd751b…`. 64,000 rows written and 1,536 left zero (ACE codes [151669,215669) -> band rows [252416,316416)). Installer preconditions verified (width, zero target rows, untied head, sidecar 2/2). Limitations: it is a weak extrapolation (text-token fit R² ~3%, held cos 0.139/0.204), and the x7.9/x7.5 spread correction is a heuristic. |
| **S3 review re-seal** (M5) | done | n/a | `-0003-review` / `-0004-review` sealed. Package `VERIFIED` (`7a8b5152…`). The parent count of 15 now comes from the E2 assessment. No human has read either sheet. |

Owner decisions needed before S7: **D1** interface set; **D2** C11 set and band; **D3** HY3D `attn2` drop; **D4** the C10 target design given the 7/210 collapse; **D5** whether the c16 1e-5 time-reference refusal and the c10 bias-ceiling refusal stay as gates (section 3.2); **D6** whether the media ">=4x" rule is MLP-only (section 3.3); **D7** release GPU6 (RTX 8000) for the S5 host-row capture (section 4, step 5).

## 2. Verification performed

### 2.1 sha256 parity: workstation mirror vs server, 47 lane files

Every file the lanes listed matches between the local mirror and the server: the M1/M2/M3/M4 tools and tests, `S7_UNITS_GAP_TABLE.json` (`793159f7…`), `COMPOSITE_S6_INPUT_FORMATS.md` (`50b6f513…`), the C15/C11/C18 json files, the 5 unit JSONs (except ACE, see below), the M4 rehearsal receipts, `s3_reseal_oc02_m5.sh`, `REVIEW_QUEUE.md` (`ed6b0fad…`) and `SCORECARD.md` (`b34d2b19…`).

Discrepancies:
1. **The local ACE unit JSON is stale.** Local `composite/s7/units/ace/ace_dit_L12/charted_block.json` is `b41ab438…`; the server copy is `48238025…`, which is the sha that M1's summary and config check name. The local mirror has the pre-M1 version.
2. **Path mismatch for `S7_M1_CHARTS_SUMMARY.json`.** The sha is the same (`d96d19c6…`), but the server has it at `composite/s7/units/` and the local mirror at `composite/s7/m1/`. `S7_M1_CONFIG_CHECK.json` (`9aabedb0…`) is the same: `units/` on the server, `m1/` locally.
3. `test_c16_install_media.py` (`70f54c90…`) has no sha in M3's report. Local and server match.

Also checked:
- Per-unit chart sidecars: all 5 `chart_receipt.json.sha256` files match.
- `charts.safetensors` sha equals `charted_block.json charts.sha256` for all 5 units.
- Dry-run manifests: M1 `scratch_dryrun_m1/out_M1` `5bf118f2…` and M3 `m3/scratch_dryrun5120/out_M1w` `574a25b2…` both match their sidecars.

### 2.2 Tests, re-run on the server

All runs were CPU only (`CUDA_VISIBLE_DEVICES=`), under `nice -n 19` with OMP/MKL/OPENBLAS=4. Each suite used the env its own docstring names: `CHIMERA_OMNI_SRC` plus `PYTHONPATH=<tree>/src`. Logs are in `/home/cereal/s7-review-oct10-claude/tests{,2}.log`.

```
test_s7_media_charts.py   [ef1ffb9]   10 passed in 5.65s
test_c15_interface_rows.py           11 passed in 10.87s
test_c18_music_band.py                5 passed in 6.40s
test_c16_h3_curve6.py     [ef1ffb9]   7 passed in 7.79s   (C16_V110=composite/tools/.bak-m3-20261010/c16_install_media.py)
test_c16_install_media.py [2ec15a8]  25 passed in 22.05s
test_c16_install_media.py [ef1ffb9]  25 passed in 25.67s
test_s7_unit_build.py     [ef1ffb9]   8 passed in 35.50s
test_m3_c11_qvl_rows.py   [ef1ffb9]   3 passed in 6.56s
test_c10_union_install.py            24 passed, 1 skipped in 215.70s (0:03:35)
test_c10_virtual_child_check.py       3 passed in 63.39s (0:01:03)
ledger (unittest discover)           Ran 104 tests in 2.955s / OK (skipped=1)
```

My first pass used the wrong env (`CHIMERA_SRC` only), so `chimera_omni` was not importable. That pass gave collection errors and 4 + 2 env failures in `test_s7_unit_build` / `test_m3_c11_qvl_rows`, and `8 passed, 2 skipped` for `test_s7_media_charts`. Everything is green under the documented env. The suites depend on undocumented-at-call-site env, so S7 drivers must export `CHIMERA_OMNI_SRC` and `PYTHONPATH` explicitly.

### 2.3 Receipts opened and claims checked

**Units** (my script `review_units.py`; every tensor loaded and checked with `torch.isfinite`):
- 0 non-finite tensors across 5 `charted_block.safetensors`, 5 `charts.safetensors`, 5 split-ridge diagnostics, 4 time references and the ACE host128 reference.
- Container dtypes: ACE F32; H3 F32x4 + BF16x11; QIMG BF16; HY3D F16; QVL32 BF16; charts F32. There are no F8/I8/U8 tensors in any container (the QIMG FP8 point is about the source, see the table).
- Chart widths are all 5120: ACE 2048, H3 5376, QIMG 3072, HY3D 2048, QVL32 1152.
- **MLP-ratio arithmetic re-derived from the source widths:**
  - ACE: 3·2048·6144 = 37,748,736
  - H3: 3·5376·14336 = 231,211,008
  - QIMG: 2·3072·12288 + biases = 75,512,832
  - HY3D: 2·2048·8192 + biases = 33,564,672
  - QVL32: 2·1152·4304 + biases = 9,921,872
  - All five ratios are ≤ 0.25, matching each unit's `component_4x`.
- **Installed sum:** 102,811,904 + 267,889,176 + 154,985,727 + 46,156,031 + 19,594,291 = **591,437,129**. This equals M1's config check and the M1 manifest's `added_parameters`.

**Lane M1 / M3 dry runs.**
- M1 `out_M1` manifest: `charted_dtype F32`, `quantization none`, `oc_seal_header_parameters 639,572,441`, `equality_asserted true`.
- M3 `out_M1w` ran `--charted-dtype match`, but only ACE and HY3D had real charts.
- **Gap:** no dry run has combined all five real charts with `--charted-dtype match`. Step 7 in section 4 closes it.

**Lane M2.**
- `c15_interface_receipt.json`:
  - `installer_check.bands`: `overlaps []`, `duplicate_type_ids {}`, tensor_sha verified per entry.
  - `simulated_install`: preconditions show 0 non-zero type-affine scales on the real S3 bridge, and finite bridge/head.
  - `trunk_bridge` records the S3 tensor shas.
- `music_codes_band.safetensors.json`: band [252416,317952), `band_rows_nonzero_before 0/0`, installer verdict "preconditions hold".
- My finite/dtype scan (`finite_scan.py`) covered the C18 table and maps, 6 C15 + 1 C11 composed rows (M2), 5 M2 charts, 6+1 `interface_latent` rows and 5 `b15` maps (M1), and both M3 C11 row files. **All finite F32, none quantized.**

**Lane M3.**
- QIMG FP32 reference: sha `d1ef9940…`, which equals the unit's recorded `time_reference.sha256`. The receipt `c642e9fc…` matches.
- H3 unit: `h3_curve6_time_gate` rel_l2 1.3167e-07, `verified true`. `h3_fp32_parameters_source_dtypes` is all F32.

**Lane M4.**
- `rehearsal_layers.json` recomputed:
  - 27 layers, veto 27/27, union beats trunk 27/27
  - held relL2 union 0.132-0.359 (mean 0.233)
  - trunk-only mean 0.900; prior mean 0.801
  - K3 held energy 0.296-0.991 (mean 0.784)
  - update_scale 0.309-0.982
  - bias energy above 0.1 on 16/27 layers (max 1.612)
  - 0 non-finite
- Both virtual-check receipts: `structural_load` 1011 tensors, `missing/unexpected/shape_mismatched []`, `equals_c10_arithmetic true`, `child_header_parameters 26,944,218,624`, `all_logits_finite true`.
- `child.partial/L00-L26.safetensors` scanned: 54 BF16 tensors, 0 non-finite. L27-L63 are dangling symlinks into Lexar paths that were never written; M4 states this.
- No router or expert tensors appear in the 1011-tensor state, so **no MoE/router leakage**. K3 enters only as dense FFN rows.

**Lane M5.**
- `oc_checkpoint.py --root $OC list` -> `OC02-F-Q38-27BU-C2K4352 fold partially_capable 0.522729 8 8 8 1`.
- `verify` -> `checkpoints 13  records verified 34  rows matched 34  deep=no`.
- `ledger.jsonl` has 34 rows, sha `bdd6cc80…c457c81`, which matches M5.
- Record shas match: `-0003-review` `aa8b9a55…`, `-0004-review` `8400aab3…`.
- `oc_package.py verify --package …` -> `files 17  digest 7a8b5152…  VERIFIED`.
- The `-0004` record carries `parent_useful_or_partial_source assessment-PARENT…(15/15), assessment_sha256 31e1af2689dc3b8e`, `package_sha256 7a8b5152…` and `smoke_verdict pending`.
- **Provenance gap (pre-existing, not from M5):** `checkpoint.parent_provenance_sha256 None`.

## 3. Adversarial findings

### 3.1 Writes into forbidden directories

- **Codex dirs: none found.**
  - `find $OC/runtime -newermt 2026-10-10` returned nothing.
  - The same search over `/mnt/lexar/omni-campaign/{qimg-*,hy3d-*,ace-*}` returned nothing.
  - `composite/media/` (Codex's H3 container) has no changes since 2026-09-07.
- **K3 dirs: written after K3 exited, by a lane NOT among M1-M5** ("claude disk executor 2026-10-10", receipts in `$OC/composite/receipts/disk/oct10/`):
  - `k3-run-oct09-claude/fold/snapshots/REMOVED_before_L61.json` (17:50:49Z). Deleted the root copies of `before_L61{,_folded}.safetensors`. The record names two surviving copies (Lexar oct08, NAS window49-60) at the same shas.
  - `before_L84{,_folded}.safetensors.MOVED.json` (19:32:45Z). Moved to `/mnt/ai_data/omni-campaign/k3-run-oct09-claude-archive/`, with NAS `sha256sum -c` 2/2 OK.
  - `$OC/k3-run-oct09-claude-body.MOVED.json` (19:32:46Z). Moved 24.6 GB `L61..` `body.safetensors` to the NAS.
  - `fold/dense` (76 G), `fold/resid` (20 G) and `receipts/` are intact. The c17 links `k3_rows/L61..L92 -> fold/resid/Lxx/held_rows.safetensors` resolve.
  - This breaks the letter of "never touch it or its dirs". It happened after `exit=0`, with hash-verified surviving copies. Recorded for Cyril; not reverted.

### 3.2 Numeric gates that conflict with the owner policy

No new gate exists in the M1-M5 tools: grepping every `raise SystemExit`/`Refusal` site finds only structural, sha, shape, count or non-finite conditions. Two numeric gates remain on the S7 path:
1. **`c16_install_media.py` 1.2.0, `verify_time_reference`** (`DEFAULT_TIME_REFERENCE_REL_TOL = 1.0e-5`, lines 788-835). It raises `Refusal` when rel_l2 > tol. The owner policy names exactly this case ("a bf16 time reference at rel_l2 2.04e-3 is recorded, not a blocker"). M3 avoided it by building an FP32 reference rather than relaxing the gate. It does not bind today (ACE 1.18e-07, H3 1.32e-07, QIMG 1.17e-07). `s7_m1_config_check.py` uses the same function with `--qimg-time-tol 2.1e-3`.
2. **`c10_union_install.py` `--chart-bias-ceiling` default 0.1** (line 1489, `REFUSED layer k`). It would refuse at layer 0 on the real charts. M4 ran with `0`. The S5 command in section 4 passes `0` explicitly.

Two more items are wording only, not gates:
- `c10`'s receipt text `frontier_rule` says "delete this child" when the veto binds in most layers. The code only records it.
- The K3 receipt's G4 "calibrated"/"REFUSED" labels are record-only.

### 3.3 Structure

- **No component under 4x is claimed as 4x.** Each media `--component` is the block MLP at ≤ 0.25 (re-derived above). Recorded for D6:
  - The rule is MLP-only. Attention, time and norm tensors of the five source blocks ride at about 1x.
  - Of 591,437,129 installed params: 96,981,937 are 4x MLP children, 140,260,736 are new chart params, and about 354.2 M are source-block non-MLP params at about 1x.
  - Whole-block ratios are 0.50-0.69.
  - The manifest registers with the 3-field *asserted* form (parent MLP count from container accounting), not one derived from the source headers. That needs explicit `--passed-through` lines at S7 registration.
- **C10 accounting:** `qwen.mlp` = 3·5120·4352·64 (verbatim gate/up rows; down is jointly refit) and `k3.ffn` = 3·5120·12288·64. This is a valid ≥4x split at the column level.
- **Provenance:** all 5 unit JSONs and chart receipts name `charts.pairs.parent_pairs` (`pairs/<unit>/parent_capture.safetensors`) without a sha. The sha is in the sidecar `parent_capture.json` (`file_sha256`), so it is recoverable but not bound in the chart receipt. `paired_rows_sha256` is bound.

### 3.4 CPU and resource discipline

- The c17 42-63 run violates the CPU rule: 3 x 10 threads at nice 10. The earlier `c17_renice_20261010.json` (03:29Z) shows the same tool starved K3 once. It is harmless to K3 now, because K3 has exited.
- The K3 L61-92 wall averaged 2,348 s/layer (75,153 s / 32), against a ~1,500 s baseline. M4 reports ~2,900 s/layer during overlap. Contention did happen; which lane caused it is not attributable from the receipts.
- M1 used GPU4 for 35 s while C10's claim on it existed (self-reported in `m1_ace_pairs.json`).

## 4. S7 command sequence

K3 L61-92 has landed. The steps below are written to be run in order; none of them ran in this pass.

```sh
OC=/home/cereal/omni-campaign; PY=/home/cereal/model-surgery-lab/venv/bin/python
EF=$OC/import/rq27b/tree_ef1ffb9
ENV="CHIMERA_OMNI_SRC=$EF PYTHONPATH=$EF/src CHIMERA_SRC=$EF/src MEDIAFOLD_ROOT=$OC/media OC_LEDGER_ROOT=$OC/ledger OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8"
S3N=/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native
```

**Step 0: owner and root (blocking).**
- (a) Wait for c17 pids 289645-289651 to exit.
- (b) Root reloads the nvidia module or reboots, then confirm:
  ```sh
  nvidia-smi
  $PY -c "import torch;print(torch.cuda.is_available())"   # want True
  ```
- (c) Cyril answers D1-D7.

**Step 1: c17 charts 0-63 present.** The receipt is rebuilt from every `L*/layer.json`.
```sh
python3 -c "import json;r=json.load(open('$OC/composite/c17/charts/receipt.json'));print(len(r['layers_present']),r['layers_invalid'])"   # want 64 []
```

**Step 2: S5 / C10 full K3 1-92.** Follow M4 `REHEARSAL.md` §5; the template keeps `--chart-bias-ceiling 0` and `--k3-channels 12288`.
```sh
for l in $(seq 61 92); do test -f $OC/k3-run-oct09-claude/fold/dense/L$l/done.json || echo missing L$l; done
cd $OC/composite/c10 && mkdir -p full-k3-1-92/logs && cp rehearsal-k3-1-60/launch_full_k3_1_92.sh full-k3-1-92/
GPU=4 STOP=63 setsid nohup sh full-k3-1-92/launch_full_k3_1_92.sh --overwrite > full-k3-1-92/logs/c10_full.log 2>&1 < /dev/null &
```
- Then run `vcheck_launch.sh` with `R=full-k3-1-92` and without `--pad-missing-with-trunk`.
- Then write the child:
  ```sh
  GPU=4 setsid nohup sh full-k3-1-92/launch_full_k3_1_92.sh --resume > full-k3-1-92/logs/c10_write.log 2>&1 < /dev/null &
  ```
- Expected: header 26,944,218,624; `$OC/composite/s5/union_receipt.json`.

**Step 3: register the S5 child (lineage id for c16).** For example `OC02-F-Q38-27BU-C2K4352K3U` via `ledger/oc_checkpoint.py register … --confirm`, after a dry run. The exact argv is not drafted here; follow `composite/s3/register/s3_register_*` as the template.

**Step 4: freeze the S3-bound charts.**
```sh
cp -a $OC/composite/s7/units $OC/composite/s7/units.s3charts-20261010
```
This keeps the H3 symlinks.

**Step 5: host rows on the S5 prefix plus refit, per unit.** `<IF>` / `<C11>` are the D1/D2 choices. The S5 prefix 0..31 is estimated at about 27 GB BF16 (7.48 B S3-prefix params + 32·3·5120·12288 = 6.04 B), which is more than a 3090 holds. This step needs GPU6 (RTX 8000 48 GB) released by Cyril (D7), or a streamed host-row mode, which does not exist.
```sh
for u in ace_dit_L12 h3_dit_L25_curve6 qimg_mmdit_L30_img hy3d_dit_L10 qvl32_vit_L13; do
  env $ENV CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<card> OMP_NUM_THREADS=2 nice -n 19 $PY composite/tools/s7_host_rows.py \
    --unit $u --model $OC/composite/c10/full-k3-1-92/child --tokenizer $(dirname $S3N)/tokenizer/tokenizer.json \
    --interface <IF> --interface-c11 <C11> --out $OC/composite/s7/tap_rows_s5 --device cuda:0
  X=""; [ $u = qvl32_vit_L13 ] && X="--type-ids 6"
  env $ENV OMP_NUM_THREADS=6 nice -n 19 $PY composite/tools/s7_chart_fit.py --unit $u --tap-rows $OC/composite/s7/tap_rows_s5 $X
done
```
The HY3D time-token row must be added to `s7_pairs_parent.py` / `s7_host_rows.py` before this step, or the D3 record should state that HY3D runs without it.

**Step 6: config check against S5.**
```sh
env $ENV nice -n 19 $PY composite/tools/s7_m1_config_check.py --tap-rows $OC/composite/s7/tap_rows_s5 --out $OC/composite/s7/S7_CONFIG_CHECK_S5.json
```

**Step 7: all-real `match` dry run (closes gap 2.3).**
```sh
env $ENV nice -n 19 $PY composite/tools/s7_dryrun.py --hidden 5120 --rows 1024 --charted-dtype match --run-name S5ALL \
  --scratch $OC/composite/s7/scratch_dryrun_s5 --units $OC/composite/s7/units/{ace/ace_dit_L12,h3/h3_dit_L25_curve6,qimg/qimg_mmdit_L30_img,hy3d/hy3d_dit_L10,qvl32/qvl32_vit_L13}
```

**Step 8: S7 install.** Output goes to root, about 55 GB; Lexar is not allowed (15 GB cap).
```sh
env $ENV nice -n 19 $PY composite/tools/c16_install_media.py \
  --child $OC/composite/c10/full-k3-1-92/child --out $OC/composite/s7/OC02-F-Q38-27BU-OMNI1 \
  --c15 <IF> --c11 <C11> --c18 $OC/composite/s7/c18/music_codes_band.safetensors \
  --blocks $OC/composite/s7/units/ace/ace_dit_L12 $OC/composite/s7/units/h3/h3_dit_L25_curve6 \
           $OC/composite/s7/units/qimg/qimg_mmdit_L30_img $OC/composite/s7/units/hy3d/hy3d_dit_L10 \
           $OC/composite/s7/units/qvl32/qvl32_vit_L13 \
  --pair-by depth-fraction --arm insert --charted-dtype match --chimera-src $EF \
  --checkpoint-id OC02-F-Q38-27BU-OMNI1 --parent-checkpoint-id <S5 id from step 3> \
  --accounting-parent-directory /home/cereal/models/Qwen3.8-27B-Uncensored --parent-parameter-count 27781427952 \
  --min-free-gib 60
```
- `--chimera-src` must be ef1ffb9: c16's default tree_2ec15a8 has no `h3_curve6`.
- Leave `--time-reference-rel-tol` at its default only if Cyril keeps D5 as is.

**Step 9: register.** Dry run, then `--confirm`, using the manifest's `register_spec` lines plus explicit `--passed-through` entries for the media non-MLP tensors, the charts and the C15/C18 writes (D6).

**Step 10: S8 review.**
- `run_omni_frozen_review.py` on the composite (2x3090 bf16, the same split style as the S3 child; that GPU0+GPU4 pair has never been tested for P2P corruption).
- `oc_assess.py --parent-dossier composite/s0/review/oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json --bank calibration/banks/lfm2_semantic_whole_response_prompts_v1.json`.
- `oc_package.py build`, then record-review.
- Four decoded media artefacts (one per VAE companion), for the human read.

## 5. Budget

### Parameters (accounting parent PARENT-Q38-27BU = 27,781,427,952)

| Item | Params | Running total | Source |
|---|---|---|---|
| S3 native trunk (vocab 317,952) | 14,864,623,104 | 14,864,623,104 | REHEARSAL.md §5, vcheck |
| C10 K3 rows 64·3·5120·12288 | 12,079,595,520 | 26,944,218,624 | vcheck `child_header_parameters`, `equals_c10_arithmetic true` |
| ACE unit (incl. charts 20,973,568) | 102,811,904 | | unit JSON |
| H3 unit (charts 55,055,616) | 267,889,176 | | unit JSON |
| QIMG unit (charts 31,460,352) | 154,985,727 | | unit JSON |
| HY3D unit (charts 20,973,568) | 46,156,031 | | unit JSON |
| QVL32 unit (charts 11,797,632) | 19,594,291 | | unit JSON |
| Media subtotal | 591,437,129 | 27,535,655,753 | M1 config check = manifest `added_parameters` |
| C15/C11 rows, C18 band | 0 new (written into existing `native_media_bridge`/`head` tensors and band rows [252416,317952)) | 27,535,655,753 | M2 receipts; **not proven by a real install** |
| **Headroom** | **245,772,199** | | 27,781,427,952 − 27,535,655,753 |
| Max `--k3-channels` with all 5 units | 12,538 | | 245,772,199 / (3·5120·64 = 983,040) = 250 channels |

### Root disk (217 G free at 21:31Z; 60 G floor)

| Step | Peak use | Free after (est.) |
|---|---|---|
| c17 42-63 remainder | ~3 G | ~214 G |
| C10 per-layer partial + child write | 32.7 + 54.4 = 87.1 G (partial removed after write) | ~160 G |
| tap_rows_s5 (as for S3) | ~6.4 G | ~153 G |
| S7 output (27.54 B, mostly BF16) | ~55 G | ~98 G |

Lexar: 95 G free, 15 G cap, not used. NAS: archival only.

### Time (estimates, loaded box, unmeasured on a quiet one)

| Step | Estimate |
|---|---|
| c17 remainder | ~2-3 h |
| C10 full 1-92 | ~21 h (M4); ~12 h with a per-layer adoption mode that does not exist |
| Host rows | ~45 min on a 3090 per M1 (but S5 needs GPU6) |
| Chart refits | ~1.5 h CPU (M1) |
| c16 install | not measured at full size |
| S8 review | not measured for this child |

## 6. What I could not verify

- **No GPU path at all.** CUDA is unavailable server-wide, so no GPU-marked test or capture was re-run. The 3 CUDA-subset c10 tests (M4: `3 passed in 13.15s` on GPU0) were not re-run.
- No real c16 install into S3 or S5. C15/C18 "0 new params" and the C11 type-embed swamping were not tested on a written child.
- The H3 `h3_thw_v1` source-position path and the HY3D time-token row have never been executed.
- The S5 prefix memory (~27 GB) is arithmetic, not measured.
- I did not re-derive M1's fit numbers or M2's PSNR/ladder numbers; I opened the receipts and checked structure, sha and finiteness only.
- Who launched the c17 42-63 run and the oct10 disk executor: both are outside M1-M5, and their receipts name only "claude".
- Whether the QIMG FP8 parent was a deliberate campaign choice (bf16 `qwen_image_2.1` exists but is a different model version).
- Codex component identity `a0535820` for the Qwen3-VL file (as in M3).
- The local mirror was not refreshed: the stale ACE unit JSON and the unmirrored M5 registry/package files are left as found.

## 7. Files from this review

| File | sha256 |
|---|---|
| `S7_READINESS_2026-10-10.md` (this file; local mirror and `$OC/composite/receipts/`) | see the upload parity line in the reviewer's return |
| `/home/cereal/s7-review-oct10-claude/` (my scratch: `review_units.py`, `finite_scan.py`, `run_tests*.sh`, `tests.log`, `tests2.log`, `review_units.json`, `finite_scan.log`) | scratch, not mirrored |

No lane artefact was modified, renamed or deleted.
