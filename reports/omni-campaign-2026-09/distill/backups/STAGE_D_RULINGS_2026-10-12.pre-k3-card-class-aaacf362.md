# Stage D rulings and decisions, 2026-10-12

Status: in force from 2026-10-12. This record binds `distill/DISTILL_DESIGN_2026-10-12.md` (sha256 prefix b34fc429).

## Cyril's words (direct, interactive, 2026-10-12)

1. "[Cyril quote withheld]"
2. "[Cyril quote withheld]"
3. Earlier the same day: "[Cyril quote withheld]" and "[Cyril quote withheld]".

Prior rulings that still hold:
- 2026-08-28: the H3 US-licence restriction "[Cyril quote withheld]".
- 2026-08-19: omnicoder is dead; its weights are deletable, and evidence is kept.
- 2026-10-06: finite fit numbers are diagnostics. Chimera has priority on all GPUs.
- 2026-09-02: K3 folds as the all-expert, all-layer whole.

## What these rulings settle

| Item | Decision | Basis |
|---|---|---|
| Go for training (§2 precondition 2) | **GO, standing.** The S8 package is still produced and delivered for Cyril to read, but training does not wait on his read. G0 (structural) and G1 (text identity, install leak) stay *functional* checks, because they detect plumbing faults, not capability. G2-G4 readings are recorded and acted on by the rules below, never as automatic refusals. | rulings 1, 3 |
| Per-stage promotion | A stage starts from the previous stage's clone when bank v1 content ≥ the previous stage, artifact-tolerant, and no modality regressed in content. Otherwise it starts from the earlier clone (CP R2) and the regression is reported. Cyril's read is asynchronous. | ruling 1 |
| H3 lane (G4, D3) | **Included.** The licence basis is Cyril's rulings of 2026-08-28 and 2026-10-12. H3 teacher caches and D3 H3 distillation are authorized. | rulings 2, prior |
| U9 clone location | **Lexar** (`/mnt/lexar`, already registrable), after the omnicoder blobs on Lexar are removed with their evidence kept. If Lexar cannot hold the set, the NAS subtree `/mnt/ai_data/omni-campaign/clones` is added to `REGISTRABLE_UNDER_PROTECTED` with a test (option a). | ruling 1, prior |
| Transient scratch | Allowed on root, and on `/mnt/lexar/omni-campaign/scratch/`. That directory is never registered and is removed with plain `rm` after use. R6 binds registered and campaign artifacts, not declared scratch. | ruling 1 |
| Engine (G5-1) | Build and parity-test T(ii), the streamed bf16 fwd+bwd trainer on the 3090s. Use it if it passes its gradient parity arm and P2 measures a rate of at least 151 tok/s. Otherwise use the measured split cell. | best model + time |
| D-R5-1 (slow engine) | **No depth cut.** Keep full-depth LoRA, accept the longer wall clock, and cut tokens only if the rate is below 60 tok/s. | best model |
| K3 budget (D-R5-6, G5-3) | **2.0M tokens in four passes** (the full budget). λ_K3 starts at 0.3, using the G3 rule. K3 is never skipped. | best model, 2026-09-02 ruling |
| H3 target count (G5-4) | 2,048 targets. Drop to 1,024 only if P4 shows more than 20 s per forward. | best model |
| Vision-in visual-token cap (G5-5) | The parent processor's default max pixels, applied to the P10-measured sizes. | best model |
| On-policy sampler (D-R5-3, G5-6) | Upside path: option (ii), a Q8_0 proxy of the student on GPU6. Planning path: option (i), a Q4_K_M proxy on the P40s, recorded as a transient sampler exception to R3. R3's "Q4 exactly once" still governs the export product. Fraction 0.25 after 20M off-policy tokens. | best model |
| Self-gen sampler quant (U7) | The parent's Q4_K_M on the P40s. Logits are always rescored in bf16. | hardware |
| HY3D time token (PREP-B) | Already fixed in the S7 refit (s7_chart_fit 1.2.0, time row). | S7 receipts |
| Delta registration | Build `oc_checkpoint.py --delta-base` (adapters + trained subsets + the pinned base digest), with tests. | needed for disk |
| Final product | Stage T (curated finetune), then a single Q4 export that fits one 3090 at 256k and the P40s. | standing order |

## Later rulings (dated, appended)

- **2026-10-13T14:26:34Z, ruling `cyril-h3-thermal-withdrawn-2026-10-13T14:26:34Z`.** Cyril to Codex, verbatim: "[Cyril quote withheld]". Source: Codex root thread `01a1326f-4f44-7ef0-a2fa-2a64a706fedc`, rollout `rollout-2026-10-12T17-25-18-01a1326f-…-2a64a706fedc.jsonl` line 763. Effect: the GPU6 H3 thermal denial is withdrawn; H3 teacher caches run on GPU6 under the 2,048-target rule above. Recorded 2026-10-14 by Claude in `$OC/distill/queues/gpu6_h3_thermal_ruling.txt` (first line = the ruling id, read by `queue_gpu6.sh` and `run_media_a_wave2.sh`).
- **2026-10-14, ruling `k3-b2-rebaseline-fp32-production-geometry-2026-10-14`** (Claude, lane gpu6, under the standing GO; not Cyril's words). The K3 cache's B2 parity reference changes from the fold engine's bf16 one-call review dump to an fp32 pass of the same frozen `k3_teacher_cache` at the same production geometry, exported at the dump's own (row, position)s (`k3_teacher_cache` 1.2.4 `export-reference`). `PARITY_THRESHOLDS`, the parity code and the run path are unchanged, and B2 still gates the K3 cache; bf16-vs-fold and fp32-vs-fold are recorded as diagnostics. Basis: `k3_b2_geometry_diag` 1.0.2 verdict `bf16_batch_numerics` (receipts in `$OC/distill/parity/k3cache-2026-10-14-b2-geometry-diag/`, verdict sha256 `e1ce01c3…`). The one-call geometry reproduces the fold bit-exactly at L00-L02; the row-group geometry differs at L02 on 3 of 98,797 real positions by one bf16 ulp, all caused by expert GEMMs whose row group holds a single token (M = 1). Each bf16 geometry is 2.94e-2 relL2 from fp32. Owner policy 2026-10-06 (finite fit numbers are diagnostics) applies to the reference choice. Identity was proven bit-exact only at L00-L02 (three KDA layers; no MLA layer, no block push after L00): see the r2 correction below. Recorded in `$OC/distill/queues/k3_b2_rebaseline_ruling.txt` (first line = the ruling id, read by `queue_gpu6.sh` 1.2.0).
- **2026-10-14, ruling `k3-b2-rebaseline-fp32-production-geometry-2026-10-14-r2`** (Claude, lane gpu6, under the standing GO; not Cyril's words; written before any r2 result). Corrects r1's "identity is proven bit-exact" (the evidence covered L00-L02, all KDA) and adds a fold anchor, because bf16-vs-fp32 of one tool is a self-comparison. B2 now ALSO needs the fold-identity gate `k3_b2_geometry_diag` 1.1.0 `--mode identity` (frozen at queue launch) over the fold's own 143,360-position corpus on GPU6: (1) at every layer L00-L92 the single-call prefix sha256 equals the fold receipt's teacher sha; (2) at every MLA layer, every block push, L1, L2, L61 and L92 the frozen tool's production `CachePass.layer` (65,536-token row groups, the K3 passes' geometry) runs on identical inputs within bf16 batch numerics of the single call (<= 1% of real positions differ, relL2 <= 1e-3, cos_min >= 0.9999, block count equal, pushed slot bit-equal, earlier slots untouched). Either failure is an identity / structural failure: functional, B2 FAILS, the fp32 pass and the K3 cache do not start. Thresholds of the parity gate are unchanged. Read by `queue_gpu6.sh` 1.3.0 (`b2_ok` requires the PASS receipt by the frozen K3 tool's sha over 93 of 93 layers); r1 kept as `queues/k3_b2_rebaseline_ruling.r1-bab355d3.txt`.
- **2026-10-14 (~19:45Z), ruling `cyril-claude-leads-all-lanes-2026-10-14`.** Cyril to Claude, direct and interactive, verbatim: "[Cyril quote withheld]" The same day, also direct to Claude:
  - "[Cyril quote withheld]"
  - "[Cyril quote withheld]"

  Effect:
  - Claude leads every Chimera Omni lane, including the lanes the 2026-10-14 reconciled plan §7.1 gave Codex: L-DATA, L-SPEECH-DATA, L-MEDIA-DATA, L-FOLD and L-NATIVE.
  - Codex works in parallel under Claude's direction whenever it has usage. It takes tasks from `$OC/distill/queues/CODEX_TASKS.md` and reports through the files of record and Relay.
  - GPU and large-RAM launches go only through Claude's lane queues and locks.
  - LibriSpeech and other older open speech corpora are approved for training.
  - Cyril's own recordings are first-class training data. They are kept local, with consent recorded as owner.
