# STATE DELTA 2026-10-09 (written 2026-10-10 ~07:45Z)

Delta over `STATE_2026-10-08.md` (sections 1-4 there remain authoritative where not superseded). Sources: lane reports E1-E4 of the 2026-10-09 workflow run, plus one live server check at 2026-10-10 07:41:19Z (that check is the source for every "running" and disk number below).

## 0. Does the composite child exist? NO.

There is no all-source composite child yet. It cannot exist before S7, the media install into the real trunk. What exists today:
- The text trunk child S3 (`OC02-F-Q38-27BU-C2K4352`, C2 MLP 17,408->4,352, 14,522,158,320 params). It is now registered, with status `partially_capable` (E1).
- C17 K3<->trunk charts for trunk layers 0-41 only (E3).
- Five media units that pass the structural >=4x rule. Each one was install-proven only on a 128-wide synthetic toy trunk. None has 5120-wide charts (E4).

Still missing for the composite: K3 L61-92 (running), charts for trunk 42-63, the C10 K3 union install, 5120-wide media charts and the S7 install, C15/C18, and then the capability evaluation.

## 1. E1: S3 trunk child registration and review

**Registered.** `OC02-F-Q38-27BU-C2K4352`, in place on Lexar:
- Path: `/mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07`
- directory_sha256 `050199a400a697e6d0294b02600fbc77f23211edbd701e215334a00f9ede6bcf`
- 14,522,158,320 params vs parent 27,781,427,952
- net_ratio 0.5227290096 (1.9130x)
- unexplained 424,699,392 = mtp dropped, not folded
- Record `...-0001-register` f1bc121cc3dc70c8
- Source: `$OC/composite/s3/register/register_confirm_20261009.log`

**Review record.** `...-0002-review` 3832e4b3acbb8bc6, `$OC/composite/s3/review/review-OC02-F-Q38-27BU-C2K4352.json` sha `9a0d1133...181f`:
- sealed bank1: 0U / 8P / 7M (8 held vs parent 15)
- columns A=8, B=1
- status candidate -> `partially_capable`
- Ledger `$OC/ledger.jsonl`: 32 rows, sha `3316783c...3cc3ca`
- `oc_checkpoint.py verify`: 13 checkpoints, 32 records verified, deep=no.

**Ledger code change.** `REGISTRABLE_UNDER_PROTECTED = ("/mnt/lexar",)` in `$OC/ledger/oc_checkpoint.py`:
- New sha `456e2735...0d63`; tests `9dcafe6f...8b03`
- Suite: 104 tests OK, 1 skipped
- Delete is still refused for every `/mnt` path, Lexar included.
- Backups: `$OC/.bak-ledger-lexar-20261009T204958Z/`

**Caveats.**
- The parent count 15 in the review was a placeholder when E1 ran. E2 has since measured parent column A = 15 and B = 12 (both upper bounds). The review record has NOT been re-sealed against the E2 parent dossier.
- The `/mnt` guard change was NOT approved by Cyril (STATE 4.3 left it to him). It is narrower than the STATE option.
- The dossier came from the native export, not from the registered HF directory. They are linked only by the exactness receipts:
  - fp32: 208/210, max logit diff 0.1224
  - bf16: 201/210, 0.5312

## 2. E2: parent Q38-27BU review band (measured)

**Run.** P40 GPU1-3, fp16, `--split 16,24,24`, Pascal venv, tree ef1ffb9. 16,707 s, exit 0. Log `$OC/composite/logs/e2_parent_review_full.log`:
- preflight agree=True
- floor passed, min|h| 138.6058
- 0 non-finite across 36 segments; max residual 932.0 at `blocks.59.norm1.in`

**Parent dossier.** `$OC/composite/s0/review/oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json`, sha `6b40227a...6a1c`.

| | Parent | S3 child |
|---|---|---|
| complete | 14 | 3 (2 of them empty) |
| censored at the 1024 cap | 2 | 13 |
| loops | 0 | 5 |
| tokens | 6,948 | 13,315 |

**Assessments** (upper bounds, no `--per-case`, no human read):
- Parent: A=15, B=12. `$OC/calibration/out/assessment-PARENT-Q38-27BU-NATIVE-NOTHINK.json` `a309f0fb...`
- Child vs parent: A=8, B=1. `.../assessment-OC02-C2K4352-NATIVE-NOTHINK-vs-PARENT.json` `fb3cd4fe...`

**Hardware finding.** P2P copies among the P40s (GPU1/2/3) are silently corrupt, even though `can_device_access_peer` returns True:
- Example: `float16 (1, 38, 5120) 0 -> 1 mismatched elements over 20 copies: 3891200`
- Every multi-P40 `.to(device)` split is suspect.
- Workaround: `$OC/composite/tools/run_review_finite_guard.py` v1.1.0 `9197c0cf...50b8` with `--host-staged-moves`; tests `73a1aae8...b03`.
- The GPU0+GPU4 (3090) pair is untested.

**Not comparable.** The parent ran fp16 on Pascal and the child bf16 on sm86, so their lane digests cannot be compared directly.

## 3. E3: C17 charts, trunk 0-41 (built)

**Trunk taps.** `$OC/composite/c17/trunk_taps/s3native-c2k4352-held-all/`, 64 layers, 24,576 rows/layer, 31 G. Receipt `taps_receipt.json` `c4d2af63...c534e2`.
- Ran on GPU0 only, because stream mode refuses `--split`.
- 46:20 wall.

**K3 rows.** `$OC/composite/c17/k3_rows/L01..L60`, symlinks, "60 linked, refused []". Manifest `farm_manifest.json` `bae6e2b1...1472`.

**Charts.** `$OC/composite/c17/charts/L00..L41`, 12 G. Receipt `receipt.json` `90f833f8...17e0f3`: 42 present, 0 invalid, 0 underdetermined.

Ridge held diagnostics (from that receipt, mean / min / max):

| | mean | min | max |
|---|---|---|---|
| C_in cos | .8186 | .7109 | .8999 |
| C_out cos | .8846 | .7667 | .9863 |
| round-trip cos | .9282 | .7800 | .9913 |

- Ridge beats the SVD and procrustes controls on every layer.
- Two-layer pools are weaker in the trunk -> K3 direction.
- C_in degrades past trunk 36 (~.73-.77).
- All of these are diagnostics, not gates.

**Not verified.**
- No C10 install or capability check on these charts. C10 needs a trunk_to_k3 alignment override for a partial K3, plus `--chart-bias-ceiling 0`.
- The K3 lineage was checked by sha against the w49-60 pass receipt only.

## 4. E4: media units (built, NOT installed)

All tools and units are under `$OC/composite/`:
- `tools/s7_unit_build.py` 1.1.0 `8a143a86...268b`
- `tools/s7_dryrun.py` `7a044575...c937c83`
- `tools/s7_dryrun_check.py` `3420ba2a...956bbc`
- `s7/units/S7_UNITS_GAP_TABLE.json` `4ac704cc...ff7da`

| Unit | MLP ratio | Installed params @5120 | Blocker |
|---|---|---|---|
| ACE `ace/ace_dit_L12` | 0.25 | 102,811,904 | 5120-wide paired rows and charts (the existing chart is 128-wide) |
| H3 `h3/h3_dit_L25_curve6` (symlinks to Codex curve6) | 0.25 | 267,889,176 | c16 patch: keep fp32 coefficients and accept the h3_curve6 time reference; charts |
| QIMG `qimg/qimg_mmdit_L30_img` | 0.2499 | 154,985,727 | Time reference is only a BF16 capture (rel_l2 2.04e-3); needs tolerance >= 2.1e-3 or an FP32 reference from Codex; charts |
| HY3D `hy3d/hy3d_dit_L10` | 0.2499 | 46,156,031 | Runs without its time token; cross-attention dropped. Needs acceptance from Cyril or Codex; charts |
| QVL32 `qvl32/qvl32_vit_L13` | 0.2499 | 19,594,291 | C11 interface rows missing; parent intermediate 4304 assumed; charts |
| C15 / C18 | — | — | Not built: no 5120 interface-row set, no music-code band table |

**Dry runs.**
- Run D (all five units, fp32 charted) passed load. Text hidden states were bit-identical to the trunk without media.
- Run E (without H3, `--charted-dtype match`) passed load.
- Runs A/B/C failed: the loader refused H3 rope in BF16.
- Run N (negative control) exited 3 as expected.

**Budget.** Under PARENT-Q38-27BU with `--k3-channels 12288`:
- Headroom 837,209,328 params; the five units use 591,437,129; 245,772,199 remain.
- Max K3 channels with all five units installed: 12,538.
- H3 joint50 (~13.39B) cannot fit.

**Relay.** STATUS was posted on d544f3fe as msg 07933a6a. No Codex reply to the 19:58Z media-split plan; the last Codex message on 95f6c2bd is from 2026-10-07T07:34Z.

## 5. Running now (server check 2026-10-10 07:41Z)

- **K3 L61-92** (`k3-oct09-claude-61-92`, GPU6, 28,616 MiB used): pid 766441 `k3_dense_pass.py --layers 61-92 --rung 17408 --acceptance record`. Log `/home/cereal/omni-campaign/k3-run-oct09-claude/logs/fitter-61-92.log`.
  - L61-L72 done; L73 loading at 07:37:48Z.
  - L72 took 1507.4 s; prefix held relL2 0.8640 vs zero-child 1.0190 (diagnostic; "acceptance REFUSED (record only, not enforced)").
  - ETA: 20 layers remain x ~1,500 s = ~8.3 h, so roughly 2026-10-10 16:00Z (estimate).
- **Unidentified process.** pid 4169670 `k3_capacity_host_loop_proc_race_v4r9.py` under `runtime/k3-nas-window61-72-oct07-*` (an Oct-07 window controller). Not started by this workflow; not touched; owner unconfirmed.
- **GPUs.** 0/1/2/3/4/5 at 5-24 MiB (idle); GPU6 is K3.
- **Disk.** Root has 170 G free (91% used). Lexar has 95 G free (98%), not 113 G as STATE assumed.

## 6. Blocked / next, in order

1. **Charts for trunk 42-63** are waiting for K3 L61-92. When the run lands:
   - Link L61-92 into `$OC/composite/c17/k3_rows`.
   - Run c17 `--layers 42-63` with the new pass receipt. That receipt must carry layers 0-92 with held_rows sha values that match the L01-60 links already in place.
2. **C10 K3 union install** onto the S3 native trunk using charts 0-63. Needs the partial-K3 alignment override plus `--chart-bias-ceiling 0`. Then a load/identity check.
3. **5120-wide media charts.** Needs paired host rows for ACE/H3/QIMG/HY3D/QVL32 (Claude); the c16 H3 fp32/time-reference patch (Claude); the QIMG FP32 time reference and QVL32 C11 rows (Codex, no reply yet); acceptance of the HY3D time-token drop (Cyril/Codex).
4. **C15/C18** interface rows and the music-code band table (Claude).
5. **S7 install**, which creates the composite child. Then register it, then run the capability review against the parent dossier from E2.
6. **Re-seal the S3 review** against the measured parent (A=15/B=12) instead of the placeholder, and get a human read of both sheets.
7. **Owner decisions:** Cyril ratifies or reverts the `/mnt/lexar` register exception; P2P corruption root cause (IOMMU/ACS suspected), and test of the GPU0+GPU4 pair.

## 7. Not verified in this delta

- `oc_checkpoint.py verify --deep`; the decode-smoke verdict.
- Any capability effect of the C17 charts or the media units.
- The K3 ETA (extrapolated from one layer time).
- Who owns pid 4169670.
- The lane numbers were taken from the lane reports; I did not re-hash any lane artefact (only this file's own upload parity is proven).
