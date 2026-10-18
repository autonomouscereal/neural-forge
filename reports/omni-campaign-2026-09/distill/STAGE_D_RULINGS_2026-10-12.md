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
- **2026-10-15 (~02:00Z), ruling `k3-card-class-equivalence-2026-10-15`** (Claude, lane k3plan, under the standing GO; not Cyril's words; written before any card-class or GPU6 1.3.0 result). Basis: owner policy 2026-10-06 (finite fit numbers are diagnostics; Chimera has priority on all GPUs), the K3 ruling of 2026-10-12 (2.0M tokens, never skipped), and `distill/K3_ACCELERATION_PLAN_2026-10-14.md`. Recorded in `$OC/distill/queues/k3_card_class_ruling.txt` (first line = the ruling id, read by `k3_multicard.sh`).
  1. **Same card class, other tool build (GPU6, k3_teacher_cache 1.3.0).** 1.3.0 seals K3 passes on GPU6 only after `k3-v130-gate` (queue_gpu6.sh >= 1.4.0) shows BIT-IDENTITY with the B2-proven 1.2.4 on GPU6 itself (every prefix and AttnRes block slot after every layer of windows w1 L84-85, w2p L83 at T = 4,096, w4 L84-85 in four row groups, hashed by k3_cache_parity) and its +1-ulp negative control DIFFERs. Any DIFFER or an incomplete gate keeps 1.2.4. No tolerance applies here: 1.3.0 claims exact numerics on the same card and is held to it.
  2. **Other card classes.** A K3 pass may run on a card class other than GPU6's (Quadro RTX 8000, sm_75, lab venv torch 2.11) only after that class passes the equivalence check, run by the very tool sha the passes on that class will use: a full 93-layer bf16 review pass over the B2 review ids (`k3_review_input_ids` + `k3_review_native_input_ids`) at the B2 geometry (`--no-trim`, group tokens 32,768, kda torch, row chunks 1, token chunk 1024) on a card of that class. **Functional (refuses the class):** the run seals (finite, structural), and `parity` - the frozen 1.2.4's code and `PARITY_THRESHOLDS`, unchanged - against the B2 fp32 reference export passes: the same bar GPU6's own bf16 pass met in B2. **Diagnostic (recorded, never a refusal):** the same `parity` read against GPU6's bf16 review pass (exported at the fold dump's positions) and against the fold dump. Bit-equality across card classes is not expected and not required (bf16 GEMM accumulation order, tensor cores, the SDPA kernel). A FAIL refuses that class for K3; it never loosens a threshold.
  3. **One pass, one geometry.** Each K3 pass runs start to finish on one card class, one tool sha and one group-token count, recorded in `<pass dir>.geometry` at its first start; a resume that differs is refused (the routed MoE's bf16 numbers depend on the GEMM row count). An out-of-memory may only lower 1.3.0's `--moe-group-batch` (numerics-neutral). Passes of different classes may be mixed in the 2.0M cache; each pass's tool, class and gate receipt are recorded in `$OC/distill/queues/k3_pass_tools.json`.
  4. **Out of scope.** The P40s (sm_61): an MLA layer at T = 4,096 does not fit 24 GB with the math SDPA they run (lane k3accel, 2026-10-14). GPU5 is never used. Which cards run K3 versus D2 is a scheduling decision (`$Q/k3_multicard.go`), not part of this ruling.

### Claude-lead decisions, 2026-10-15 (Claude as lead under `cyril-claude-leads-all-lanes-2026-10-14` and the standing GO; NOT Cyril's words)

- **2026-10-15 (~02:10Z), Claude-lead decision `d2-codex-heldout-exclusion-2026-10-15`** (not Cyril's words). Codex's latest recovery split is canonical. That is the set of rows Codex admitted to validation and test in its native-text-v3 (short), v4 (long) and v7 (executed tools) packs, as listed by `proof_outputs/chimera_text_jobs_oct13_v8` (sha `dfbfaa4c…`). v8 is the newest manifest and supersedes v1-v4 and v7. All of v1-v8 carry the same held-out sources, so the `_v1` manifest names the same split. Its 5,055 short and long ids equal `corpus-schema-v2` validation + test (the audit's input) and the split of `normalized.jsonl` (`168c4dc9…`). The other 336 executed-tool ids are in no D2 input. Every one of these 5,391 rows is EXCLUDED from all D2 inputs, together with the BFCL-62 rows and their twins (A2 3(a) and item 10 unchanged). D4 replay and stage T inherit the exclusion.
  - Form: on the registry this is an all-terms span mask over each held-out document (3,178,013 positions, 3,095,968 of them trained today). Row exclusion was measured and refused, because it leaves 6.13M positions, below the 18.85M draw.
  - Residual: in 3,311 shared packed rows the held-out text stays attention context for later documents, although no loss term targets it. Reports of Codex val/test after D2 say so.
  - Refill: the lost trained tokens (about 2.96M inside the draw) are refilled from train-split rows in this order: verified tool-call self-gen (L-TOOL raised to 3.0M, sampling stopped by 10-15 22:00Z), then self-gen, then Codex native train rows not yet in D2. The cached undrawn registry rows are the fallback. D2 never waits on a refill source.
  - Supersedes A2 DRAFT item 5's sentence that Codex's val/test "is not a held-out measure for D2+".
  - Receipts: `research/data/d2-audit-2026-10-15/d2_holdout_eos_audit.json` (sha `19d47bac…`) and `research/data/d2-decisions-2026-10-15/d2_decision_measure.json` (sha `5fef0bb5…`), both under `/mnt/lexar/omni-campaign/`.
  - Machine-readable record: `$Q/D2_DECISIONS_2026-10-15.json` (sha `c8d47f22…`). L-TRAIN applies it before D2's first step and writes `distill/corpus/d2_masks_v2/D2_DECISIONS_APPLIED.json`.
- **2026-10-15 (~02:10Z), Claude-lead decision `d2-eos-mask-truncated-2026-10-15`** (not Cyril's words). Every D2 document flagged as truncated gets CE, EOS and UL weight 0 on the three positions that predict its end-of-turn sequence `<|im_end|>`, newline, `<|endoftext|>`. The flag is quality_scorer 1.0.4 class 06 (unclosed think), 08 (cut mid-sentence) or 14 (cut mid-word). That covers 6,936 documents, all ending in exactly those three ids, and 6,831 of their targets are trained today. Q38 KL is kept, because at a mid-word cut the teacher predicts the continuation, not a stop. Self-gen rows that hit the sampling limit (122) end on a non-EOS id and carry no such target. Receipts and record are as above.
- **2026-10-15 (~03:45Z), Claude-lead decision `d2-clean-repack-2026-10-15`** (lane d2clean, under `cyril-claude-leads-all-lanes-2026-10-14` and the standing GO; NOT Cyril's words). It upgrades the span mask of `d2-codex-heldout-exclusion-2026-10-15` to a context-clean registry. Plan: `distill/D2_CLEAN_REPACK_PLAN_2026-10-15.md`.
  - **Built:** `distill/corpus/text_clean_v1` (d2_clean_repack 1.0.1, manifest_sha256 `5f25c817…`, `verify` OK, review screen 0 hits).
    - The 3,386 sealed rows that hold any token of the 5,061 X documents are excluded. X = the 5,055 held-out documents plus the 6 BFCL-62 twins that are not held-out; the BFCL-62 sources are already among the held-out.
    - Their 22,970 non-X documents (12,349,164 tokens) are re-packed whole into 3,015 rows.
    - The 1,497 kept rows stay byte-identical and keep their sealed Q38 caches. Their 1,660,520 positions owned by re-packed or X documents get every term masked.
    - Every non-held-out document is trained once, whole. No held-out token is an input or a target.
    - A2 3(b), A2 3(c) and D2-X2 are carried by document.
    - The sealed `distill/corpus/text` is unchanged.
  - **Teacher cache:** the frozen qtext_teacher_cache 1.0.5 (`f2465a11…`, the sealed caches' tool) with the sealed flags, run on the P40s with `--linear-attn-kernel torch` and `--micro-batch 2`. fla does not compile for sm_61.
    - The card-class bar was written after the measurement and follows owner policy 2026-10-06. Finite and structural results are functional; distances are diagnostics.
    - On 8 sealed rows against an fp32 reference, the P40 cache is as close as the sealed 3090 cache: KL mean 0.0116 vs 0.0109, top-1 0.989 vs 0.988.
    - The kernel is recorded in each cache's config.
  - **Use rule:**
    - d2_plan uses the clean registry only if `d2_clean_repack.py state` exits 0 when the D2 data config is written: built, verified, and 12 of 12 caches sealed. Otherwise D2-X1's span mask applies and D2 does not wait.
    - In clean mode, the registry sources take `clean_exclude_rows.json`, `clean_all_mask.json` and `clean_gt_mask.json`, and the 12 `text-clean-*` files are added as registry sources. The applied receipt names the mode.
    - The clean version is the preferred registry for D4 replay and stage T. This replaces the "not adopted / open option" entry `repack_and_new_registry_cache` of `D2_DECISIONS_2026-10-15.json`, which stays unedited.
  - **Job:** `$T/d2clean_recache.py` 1.0.1, one runner per P40 (`$Q/d2clean.go`). It yields to the D1 bank v1 review and to every v1 RAM wait. ETA (EST) 10-16 about 17:00-21:00Z.


### Cyril's words and approved realignment, 2026-10-17 (direct, interactive, Claude session 753087a5)

- **`cyril-full-distill-no-lora-2026-10-17`.** His words: "[Cyril quote withheld]".
  - Every Chimera Omni stage trains **all weights**.
  - The student is initialized from the folds (`OC02-D-Q38-27BU-OMNI1D1.clone-pre-D2`).
  - Teachers are the **original parents**, with outputs cached once:
    - Kimi K3 is primary (his 08-04 "99% kimi k3");
    - Qwen 3.8 27B Uncensored gets a small weight;
    - the media teachers come later.
  - **Superseded:** the LoRA D2 (`d2_launch.sh` 1.3.2, `streamed_kd_trainer.py` 1.3.4 LoRA sites). It was stopped 2026-10-17T20:19Z and will not be resumed.
- **`cyril-no-small-model-data-2026-10-17`.** His words: "[Cyril quote withheld]".
  - Training text is our own corpus plus frontier-model trajectories harvested from the PostgreSQL memory.
  - Teachers only score it.
  - Kimi K3 cannot free-run here (1.37 TB of packed experts is more than RAM), so it is logit-only. Qwen 3.8 Flash Next is not on disk.
- **`cyril-drop-selfgen-refill-2026-10-17`** (approved by Cyril, 2026-10-17). These sources are removed from every Chimera Omni training mix:
  - `selfgen` (1,145,085 tokens, Qwen 3.8 27B Q4 self-generated);
  - `tool-selfgen` (runs 1+2, 3,002,570 tokens; run 3 partial; Qwen 3.8 27B Q4);
  - `refill-codex-v7-train` and `refill-codex-v8-train` (scripted, templated assistant turns).
  - Their files and caches stay on disk as evidence only. They are not deleted and not trained.
  - `tool-bfcl-native` stays as data.
- **`cyril-k3-on-3090s-2026-10-17`** (approved by Cyril, 2026-10-17). His words: "[Cyril quote withheld]".
  - `k3_multicard.sh` 1.1.0 (`6e12e3a4…`, in `$Q` and `$T`) carries the provisional-bf16 branch of `queue_gpu6.sh` 1.5.0.
  - It runs on GPU0 only, because host RAM fits one 3090 pass beside GPU6. GPU4 is kept for the full-parameter trainer probe.
  - Its card-class verdict goes to `$Q/k3_cardclass_provisional/`, its passes to `teacher_cache/k3/eb35851b2496a628-provisional-bf16/`, and its records to `$Q/k3_pass_tools.provisional.json`.
- **`k3-b2-bf16-numerics-provisional-2026-10-17`.** Approved by Cyril 15:27Z ("[Cyril quote withheld]"). The text is in `$Q/k3_b2_numerics_ruling.txt`.
- **Plan of record:** `reports/omni-campaign-2026-09/REALIGNMENT_2026-10-17.md`.
