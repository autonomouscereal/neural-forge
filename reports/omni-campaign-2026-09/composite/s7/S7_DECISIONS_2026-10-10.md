# S7 engineering decisions, 2026-10-10

Written by the PREP-A lane (Claude) on 2026-10-10 at about 22:10Z, server `ai`, `$OC=/home/cereal/omni-campaign`.

These answer the questions D1, D2, D3, D5, D6 and D7 from `S7_READINESS_2026-10-10.md` (section 1). They are engineering decisions made by the campaign under the owner's standing authority:

- **Owner policy (Cyril, 2026-10-06):** finite numeric fit values are diagnostics. They are never fail or discard gates. Structural, load, identity, provenance and non-finite checks stay functional, and so does the structural rule of at least 4x per component. The all-source child is assembled first and its capability is evaluated afterwards. There is no training. Choosing a design parameter from measurements counts as design, not as a gate.
- **Cyril, earlier:** "[Cyril quote withheld]". So the child has to work, with coherent text at minimum.

Cyril has not reviewed these decisions one by one and can overrule any of them. Nothing is deleted by them. Every alternative set stays on disk.

**D4 is not decided here.** D4 is the C10 target design after the 0-26 hybrid collapsed to top-1 7/210 against the S3 trunk. It stays open. Under Cyril's "the child must work" instruction, a union that collapses the trunk is not an acceptable final design.

## D1: interface set of record is M1 `interface_latent`

- **Set of record:** `$OC/composite/s7/interface_latent/`
  - `companions.json` sha256 `929083f864be12a25c160632575900fc5dadfe74e46dd6c18e20744651b6d60d`
  - `interface_receipt.json` sidecar `8ec22aded96a78a68263743cd1af5276651bad76b14a28de426387d12cbfdd04`
- **Rows** (name, source, type, feature band):

  | Name | Source | Type | Feature band |
  |---|---|---|---|
  | `music_ace` | ACE | 4 | f[0,384) |
  | `video_h3` | H3 | 2 | f[384,480) |
  | `audio_h3` | H3 | 3 | f[480,512) |
  | `image_qimg` | QIMG | 1 | f[512,576) |
  | `volume_hy3d` | HY3D | 7 | f[576,640) |

- **Reason:** the five media charts (`composite/s7/units/*/charts.safetensors`) were fitted on host rows that were read through this set. The charts are valid only under the rows they were fitted against. Choosing another set would mean refitting every chart.
- **M2 `interface/`** (`companions.json` `66c0ee1c…20df`) is kept as an alternative and is not deleted. It has the same bands and different rows (feature_proj relL2 between the two sets is 2.3-4.7).
- **Consequence:** step 5 of S7 (host rows on the S5 prefix plus chart refit) and step 8 (the c16 install) pass `--interface $OC/composite/s7/interface_latent` and `--c15 $OC/composite/s7/interface_latent`.

## D2: C11 set of record is M1 `interface_latent/c11`

- **Set of record:** `$OC/composite/s7/interface_latent/c11/`
  - `companions.json` sha256 `1181a1bd1567ad9e6a200819c4e9b58003a9d5895db2b6a93f5e4e528a4ace44`
  - Entry `ocr_qvl32`, source QVL32
- **Band:** feature f[640,2176), embed e[384,385), type 6.
- **Reason:** the same as D1. The QVL32 chart was fitted through these rows.
- **Kept as alternatives, not deleted:**
  - M2 `interface_c11/` (`b475a6c7…fb6d`)
  - M3 `c11/qvl32_patch_merger` (version A)
  - M3 `c11/q38u_own_visual` (version B)
- **Recorded limitation, not a gate:** the C11 rows are 2% (A) and 17% (B) of the random S0 `type_embed[6]` norm.
- **Consequence:** pass `--interface-c11 $OC/composite/s7/interface_latent/c11` / `--c11 $OC/composite/s7/interface_latent/c11`.

## D3: HY3D conditioning goes through the joint native sequence; `attn2` is not carried; the unit must carry its time token

- **Unit:** `units/hy3d/hy3d_dit_L10/charted_block.json` (sha256 `c5ae82e27963e043f47a9669e3a352fca38161ac43b4ee89622b0b8eb7cb3ab1`).
- **Design.** Chimera is encoder-free. Conditioning such as the image, the text prompt or the other modalities reaches the charted HY3D block as rows of the one joint native sequence, and the block self-attends to them. So the source block's `attn2` cross-attention, which reads DINO image context, has no input to read and is not carried. Its dropped tensors are recorded in the unit's `provenance.dropped_tensors`:
  - `attn2.{to_q,to_k,to_v,q_norm,k_norm,out_proj.weight,out_proj.bias}`
  - `norm2.{weight,bias}`
- **The time token is a requirement, not optional.** HY3D has no adaLN. Its timestep enters as ROW 0 of the block input. The unit's `runtime_requirements.hy3d_time_token_row` is one type-7 time row first in each volume segment, paired with the parent block-10 input row 0.
- **Defect.** Today's charts were fitted without that row: `s7_pairs_parent.py` line 400 records `block_input[:, 1:] (row 0 = the time token, dropped)`, and `s7_host_rows.py` emits no time row. So the HY3D unit as fitted would run without its time token.
- **Fix (PREP-B):**
  - Emit the time-token row in pairs and host rows.
  - Refit the HY3D charts with it.
  - Re-verify the unit (structure, sha, finiteness) before S7 step 5.
- Until PREP-B lands, the HY3D unit is NOT S7-ready.

## D5: two numeric thresholds become diagnostics, not refusals

Both thresholds now record their numbers instead of refusing. Each keeps an opt-in strict mode. Tool changes are in the same PREP-A lane, backups are under `$OC/.bak-policy-20261010T220722Z/`, and sha256 values are in the lane report.

**1. c16 time-reference tolerance** (`c16_install_media.py` 1.2.0 -> **1.3.0**)

- `verify_time_reference` measures rel_l2 against `--time-reference-rel-tol` (default 1e-5).
- Above the tolerance, the placement's `time_reference` record gets `verified: false`, `policy: diagnostic` and a `warning`. The manifest gains a `warnings` list and a `policy` record. The block is installed, not refused.
- `--strict-time-reference` restores the 1.2.0 refusal. The same applies to `--convert-ace-run`.
- These still refuse with or without the flag:
  - a missing reference on an adaLN or h3_curve6 block
  - a non-finite or all-zero reference or time path
  - an h3_curve6 shape mismatch
  - every sha, shape and structural check
- A verified report is key-for-key the 1.2.0 report.

**2. c10 chart-bias ceiling** (`c10_union_install.py` 1.3.0 -> **1.4.0**)

- `--chart-bias-ceiling` now defaults to 0 (disabled). The receipt records:
  - per layer: the dropped input-bias shift and the output-bias energy, `ceiling_enforced` and `exceeds_reference_0_1` (the old 0.1 default kept as a reference line)
  - at the top level: `chart_bias_policy` with `layers_above_reference`
- An explicit `--chart-bias-ceiling X > 0` is the opt-in strict mode.
- Every structural and non-finite refusal in c10 is unchanged.
- **Resume caveat:** c10's partial identity includes `tool_version`. A 1.3.0 partial does not `--resume` under 1.4.0. The full K3 1-92 run starts fresh, so it is not affected.

`s7_m1_config_check.py --qimg-time-tol 2.1e-3` keeps working. The function now returns a report instead of raising, and the check copies `verified` from it.

## D6: the media >=4x rule applies to the MLP only

- The structural >=4x rule for each media unit is applied to the block MLP: the `--component` is `<source>.dit.block.mlp`, and its ratio is at most 0.25.
- This matches the trunk C2 fold, where attention passes through and only the MLP is folded.
- The media blocks' attention, time and norm tensors, the charts, and the C15/C18 writes are registered as `--passed-through` lines at S7 registration. They are never registered as `--component`.
- **Recorded numbers** (S7 readiness section 3.3, re-derived from the source widths):

  | Unit | Parent MLP params | Child MLP params |
  |---|---|---|
  | ACE | 37,748,736 | 9,437,184 |
  | H3 | 231,211,008 | 57,802,752 |
  | QIMG | 75,512,832 | 18,874,367 |
  | HY3D | 33,564,672 | 8,388,607 |
  | QVL32 | 9,921,872 | 2,479,027 |

- Of the 591,437,129 installed media params:
  - 96,981,937 are 4x MLP children
  - 140,260,736 are new chart params
  - about 354.2 M are source-block non-MLP params at about 1x
- Whole-block ratios are 0.50-0.69, recorded and not claimed as 4x.

## D7: GPU6 is available for the S5 host-row capture

- GPU6 (Quadro RTX 8000, 46 GB) is available to this workflow for S7 step 5, the host rows on the S5 prefix.
- The S5 prefix 0..31 is estimated at about 27 GB BF16. That is arithmetic, not a measurement, and it does not fit a 3090.
- **Rules that apply:**
  - Check `nvidia-smi` first, and never start on a card that is already using more than 2000 MiB.
  - Record the allocation in `$OC/composite/receipts/allocator/<name>.json`, or launch through `hermes.py run`.
  - GPU5 stays reserved for the ICE H3 roster.
  - Never use P2P between the P40s (GPUs 1-3).

## Status after these decisions

- **D1, D2, D5, D6 and D7:** closed for S7.
- **D3:** decided, but the code fix is in PREP-B.
- **D4:** open.
- **Hard blockers still standing:**
  - c17 charts for trunk layers 42-63 (running)
  - the C10 target design (D4)
  - the S5 host-row recapture and chart refit (steps 2-5)
