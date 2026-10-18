# Chimera v2 folding program: ranked hypotheses and 4B-ladder protocol (2026-10-14)

**Status:** design and plan. Nothing here has been run. No GPU, lock, queue, checkpoint or v1 artifact is touched by this document. It is the Claude research lane that extends Codex's L-FOLD audit (`C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/OBJECTIVE_HISTORY_AND_RECOVERY_BACKLOG.md`) and does not edit Codex files.

**Inputs this design rests on** (all read; paths given; numbers are the ones those files print):
- Fold state: S3 receipt `/mnt/lexar/omni-campaign/s3-fold-sep07-codex/S3-C2K4352-sep07/qf-build-receipt.json` (median held relL2 0.556; 47/64 layers below cos 0.86), `CHIMERA_OMNI_STEP1_RESULTS.md` section 3.3 (OMNI1 = S5 at 210/210 R-Q; text damage is in the trunk fold), `composite/K3_RESULTS_2026-10-08.md`, `composite/c10/design_sweep/VERDICT.md` (K3 union: no held-out text gain), evidence manifests `omni-evidence-v1` / `omni-evidence-27bu-v1` (98,304 tokens, 196 rows, zero `<|im_start|>` tokens).
- Codex audit: p35 QKVO endpoint residual 5/8 -> 8/8 exact 4-token rollouts (the one positive recipe); action-only MLP selection made continuations worse; 0/24 exact-8 in aggressive trials; `mtp_heads=0`; `folding_retention.py` (commit `67b32b54`) at `C:/Users/cereal/Documents/ai training/.agent-worktrees/neural-forge-a495ffdd2f/best/experiments/chimera_recovery/folding_retention.py` (CLI: `--contract --rollouts --out`).
- Literature (2025-2026, cited inline): ShortOPD 2607.13124, Thinking Machines OPD 2025-10-27, MoE->dense 2605.28207, SlimQwen 2605.08738, Puzzle (ICML 2025), GISP 2510.18030, LaCo (ACL 2026), GenDistill 2603.26556, TIP 2604.14084, Sparse Logit Sampling 2503.16870, self-generated calibration (ICLR 2025, Ji et al.), SSGR 2511.18864, REAP 2510.13999, KL mixing 2605.16826, Lightning OPD 2604.13010.
- Data audit: `/mnt/lexar/omni-campaign/research/data-audit/full/audit4_report.json` (46.6% of D2 tokens in hard-defect classes; one ~3,000-character cap behind classes 06 and 08).
- Tools (relative to `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/`): `qwen-fold/qf_build.py` (axes C1-C9; `--mlp-select {evidence,reader-metric}`, `--mlp-group`, `--no-lambda-split`, `--carry-mtp`/`--drop-mtp`, `--dtype`, `--review-script`), `qwen-fold/qf_mlp.py::narrow_mlp_site` (target = pristine parent block at the child's own input; ridge anchored to the parent's truncated columns; `reader_metric` at line 282, never run on a real parent), `qwen-fold/qf_evidence_build.py` (48-character contamination screen; bank v1/v2 + probes only), `calibration/oc_run_frozen_bank.py` + `oc_assess.py` + `wavea_review.sh`, `distill/tools/streamed_kd_trainer.py` (P40 gradient-parity receipt on the 0.8B), p35 runner manifests under `.agent-worktrees/neural-forge-a495ffdd2f/best/experiments/p35_*` and `best/src/neural_forge/hf_target_native_behavior_distillation.py`.

Evidence marks: **(m)** printed and read by a Claude lane; **(cx)** Codex-reported, not recomputed; **EST** estimate; **UNVERIFIED** nobody on the Claude side saw it printed.

---

## 1. Problem

OMNI1's text intelligence was lost in one step: all 64 Qwen MLPs cut 4x at once by a per-layer ridge fit on 98k out-of-template tokens, with no objective that looks at the composed model's output or at generated text. Every later stage (K3 union, media charts, D1) is also a local fit. The literature that keeps capability after >=4x per-component compression uses the local fit only as initialization, then global logit KD, then on-policy distillation; the published failure mode of skipping that (repeated suffixes, running to cap, empty answers) is S8's exactly. v2 needs a fold recipe proven on the cheap 4B ladder, graded behaviour-first, before it is spent on a 27B.

## 2. Design

### 2.1 The v2 fold recipe (what we are trying to prove)

```
parent --(A) calibration: chat-templated, parent-generated, screened
       --(B) selection + allocation: output-weighted coreset, non-uniform width at fixed total
       --(C) local fit = initialization only (C2 ridge, group schedule, lambda by downstream criterion)
       --(D) multi-token endpoint residual: rank-r QKVO/MLP residual fitted on H-token continuations
       --(E) global correction: forward-KL top-k KD (off-policy), then ShortOPD-style on-policy KD
       --(F) grade: frozen review + coherence + free-run 5/8/16/32 + semantic/tool outcomes
```

Every arrow is a hypothesis in section 3 with its own matched experiment. A and B are near-free and run first; D is Cyril's multi-token objective; E is the published repair; F is the only promotion gate.

### 2.2 Components and responsibilities

| Component | Exists? | Responsibility in v2 | Owner |
|---|---|---|---|
| `qf_evidence_build.py` | yes | Add `--source chat-selfgen`: parent-generated, chat-templated (thinking-off) responses on train-pool prompts; extend the contamination screen to R-Q 210, probe64, smoke packs, toolcall bank, Codex 274 + val/test, S8 media prompts | Claude research |
| `qf_mlp.py` | yes | (B) `reader-metric` selection run on a real parent for the first time; new `--mlp-weight down-colnorm`; (C) `--lambda-criterion {internal,next-layer,logit}`; `--no-parent-anchor`. All new flags default to today's behaviour | Claude research |
| `qf_build.py` | yes | `--mlp-keep-plan <json>` (per-layer keep counts at fixed total) for non-uniform allocation; `--carry-mtp` instead of `--drop-mtp` | Claude research |
| Endpoint residual runner (p35 lineage) | yes (QKVO rank 8, H=4) | Generalise to H in {4,8,16}, add the `down_proj` site, child-own-prefix rollouts, parent top-64 targets; emit a receipt `folding_retention.py` accepts | Claude research, on Codex's p35 |
| `streamed_kd_trainer.py` | yes (1.3.x, pinned) | Off-policy forward-KL KD on the ladder child (LoRA r32 on trunk MLPs); used as a research copy with its sha in the receipt | L-TRAIN engine |
| OPD driver (new, ~300 lines) | no | Student samples (HF), loop detector (ShortOPD: periods 1-10, 32-token anchor, >=90% agreement, >=3 cycles, >=64 tokens), EMA budget 256 -> 1024, teacher scores sampled tokens only, JSD 0.5 + 0.3 forward-KL | Claude research; D2's on-policy term shares the design |
| `folding_retention.py` | yes | Free-running 5/8/16/32 exact prefix, LCP, empty/cap/repetition; a block in every grade | Codex tool, read-only |
| `oc_run_frozen_bank.py` + `wavea_review.sh` | yes | Bank v2 15-case frozen review, column B coherence | existing |
| Semantic scorer (new, thin) | no | Exact answer on math/ARC rows with extractable answers; executed tests on code rows; toolcall-bank outcome; correct alternate wording allowed (Codex P0) | Claude research |
| Experiment ledger | files, optional PG index (section 4) | One receipt JSON per child, sha-pinned; table indexes them for paired queries | Claude research |

### 2.3 Data flow for one experiment cell

1. **Evidence**: `qf_evidence_build.py` -> sealed manifest `/mnt/lexar/omni-campaign/research/fold-v2/evidence/<name>.json` (sha printed; screen hit count printed, must be 0).
2. **Fold**: `qf_build.py --parent <OC01 4B> --evidence <manifest> --mlp-keep <k> [axes]` -> child dir + `qf-build-receipt.json` (per-layer held relL2/cos, lambda, update scale).
3. **Residual / correction** (cells that have it): runner writes `residual-receipt.json` / `kd-receipt.json` with token counts, teacher sha, student snapshot sha, horizon per step, on-policy coverage, fallback count (Codex P1 receipt list).
4. **Grade**: `oc_run_frozen_bank.py` (bank v2) -> `wavea_review.sh` -> `oc_assess.py`; `folding_retention.py --contract <c> --rollouts <r> --out <o>`; semantic scorer. All blocks land in `grade.json`.
5. **Pair**: every cell is paired against the C2K2304 baseline on the same bank, prompts and decode settings; the receipt stores the baseline sha it was paired with.

### 2.4 Grading contract (behaviour-first, fixed before any run)

| Block | Instrument | Reported as | Promotion weight |
|---|---|---|---|
| Frozen review | bank v2, 15 cases, `wavea_review.sh` | pass count, column B | primary |
| Coherence | column B + `folding_retention.py` empty/cap/repetition rates | counts | primary |
| Free-running continuation | `folding_retention.py` exact prefix at 5/8/16/32, LCP, 64 prompts | counts | primary (ranks ties) |
| Semantic / tool outcomes | math/ARC exact answer, code tests executed, toolcall-bank outcome | correct / total | primary where the cell claims it |
| Token match, held NLL, relL2, cos | receipts | diagnostics only | none (held NLL is inverted on this family: parent 4.290, S3 3.756, S5 3.432 (m)) |

Success for a cell = better than C2K2304 on at least two primary blocks and worse on none. "Folding works at 4x" is claimed only when a 4x-per-component child reaches the X03/X08 band (15/15, column B 12-13) and its retention@8 is within the parent-vs-parent-reload noise floor of the parent (that floor is measured first; UNVERIFIED today).

Held-out provenance: no cell fits, selects, calibrates or trains on bank v1/v2, R-Q 210, probe64, smoke packs, S8 media prompts, toolcall bank, or Codex's 274 / val / test. The screen runs on every evidence manifest, every KD/OPD prompt set and every generation pool before the cell starts.

Matched budgets: all 4B children keep the same total MLP channel count as C2K2304 (layer count x 2,304; layer count read from the parent config at build time). Non-uniform cells redistribute, never add. Residual cells report their parameter delta (<1% of the child) and the >=4x rule is judged on the component after the residual is merged.

## 3. Ranked hypotheses

Ranked by expected gain in behaviour per GPU-hour on the 4B ladder, then by how directly each attacks the measured OMNI1 failure.

| # | Hypothesis | Mechanism | Expected effect on intelligence density | Cost | Risk |
|---|---|---|---|---|---|
| H1 | **Local fit is initialization; a global correction stage is mandatory** (forward-KL top-k KD, then ShortOPD-style OPD) | Per-layer ridge minimises local error; errors compound across 64 layers (LaCo; LFM2 "locality is the failure"). Global logit KD aligns the composed output; OPD trains on the child's own prefixes, where loops and empties live. ShortOPD: pruned 5.71 -> KD 30.52 -> ShortOPD 48.46 vs teacher 75.17 at 250M rollout tokens; Thinking Machines: 9-30x less compute than SFT | Largest. This is the published gap between our pipeline and every recipe that keeps >=73% after a 4-9x cut | KD: 5-20M tokens on the 4B (3090 EST 4-10 h); OPD: student rollouts are the bottleneck; P40 only for scoring | On-policy from a Q4/Q8 proxy drifts from the student (Lightning OPD teacher-consistency); long OPD without forward-KL mix collapses entropy (2605.16826); LoRA trails full FT by ~6% in OPD |
| H2 | **Multi-token endpoint residual as the fold objective** (Cyril's ask) | After the local fit, fit a rank-r residual on QKVO and/or `down_proj` to minimise cross-entropy over the parent's next H tokens with the child rolled forward on its own prefix (H = 4, 8, 16). p35 did this at H=4 rank 8 on QKVO: 5/8 -> 8/8 exact-4, 56/64 -> 64/64 held top-1 | Direct repair of trajectory error, not point error, at tiny parameter cost. Expected to move retention@8/16 more than any closed-form change | P40-feasible at 4B once teacher continuations are cached (EST 4-8 h per cell) | p35 positive was 8 prompts; aggressive variants gave 0/24 exact-8; action-only MLP selection got worse. Horizon must be tied to a budget controller, not just raised |
| H3 | **Calibration in the deployment distribution**: chat-templated, thinking-off, parent-generated responses, screened | Both manifests have zero chat tokens (m); the fold never saw an assistant-header position; S8 fails at that boundary 3/16. Ji et al. ICLR 2025: self-generated calibration up to +2.68%, can outweigh the pruning method; SSGR +10-13% on pruned R1-Distill | Medium-large for empties and early EOS; small for arithmetic | Lowest of all: 0.1-0.4M generated tokens, 1-3 h on a P40 | Parent generations are over-confident; a 50/50 mix with screened registry rows hedges that. Must be built from carve `train` rows only |
| H4 | **Selection that looks at outputs, and non-uniform width at fixed total** | C2 picks channels spanning activation space and ignores `down_proj` column norm and downstream effect; `reader-metric` exists but never ran on a real parent. Per-layer held relL2 spans 0.10-0.67 (m): uniform 4x spends the same on L63 (0.104) as on L36-45 (0.60-0.67). Puzzle MIP, SVD-LLM V2, Nemotron Elastic all beat uniform | Medium; best per unit cost after H3. Also answers whether "4x per component" must be per layer | CPU + one P40 capture pass; a build is the normal 1.8 h-on-3090 (P40 EST 5-9 h) | Allocation needs a ruling from Cyril (per layer vs per component total). Reader-metric can over-fit the 2,048 positions it scores |
| H5 | **Ridge choices: lambda by downstream criterion, no parent anchor, output-weighted error** | lambda is chosen on the fit corpus (1.0 on 51 layers, 0.1 on 13); the anchor pulls toward the parent's truncated columns; unweighted squared error ignores which directions the next layer reads | Small-medium; cheap; may move held cos by 0.02-0.05 (EST) | CPU-only re-solve on captured rows (minutes per layer) | Indefinite Gram at weak lambda (15/64 layers in C10); keep the lambda split |
| H6 | **Drift budget: fold 8 layers at a time with a short KD between blocks** | LFM2: all-layers-at-once 4x dies (0/15), three components at a time survive; S3 is the all-at-once case. Progressive beats one-shot at equal tokens (SlimQwen) | Medium; interacts with H1 | 8 builds + 8 short KDs, EST 2-3x a single build | Wall clock; ordering (front-to-back vs sensitivity order) is an open choice |
| H7 | **Retain and distil the MTP head** (`--carry-mtp`, MTP-KD auxiliary) | Head exists (120.6M on the 4B, 424.7M on the 27B; dropped today). SlimQwen MTP-KD: +1 MMLU, +2 EvalPlus, -1.9 GSM8K; doubles spec-decode acceptance at depth 4 | Small, mixed; a multi-token auxiliary and a free speculative head | Low once H1's trainer takes a second loss term | Random-init heads are not a fix (Codex P2); teacher-forced target; does not replace H1/H2 |
| H8 | **K3 path: DO-ACP expert selection + concatenation + forward-KL KD, or logit-only fusion; never an additive weight union** | C10's union at w=0.03 gave 110/210 vs 108/210 (noise) for 12.08B params (44% of OMNI1). MoE->dense 2605.28207: keep K experts, concatenate, forward-KL (43.41 vs reverse 37.17), +6.3 pp over dense pruning at 4B tokens; REAP: merging loses routing control irreducibly | Large for the 27B budget (frees ~12B params for media or width); zero for the 4B ladder | Selection on the GPU6 queue; KD cost as H1 | Cross-tokenizer target for K3 as teacher is UNVERIFIED in our "top-32 mapped + tail"; BPM 2607.22334 / SimCT 2605.07711 are the fixes if it leaks mass |
| H9 | **Token-level selection in the KD loss** (TIP: low student entropy, high divergence; RS-KD sampled logits) | Trains on <10-20% of positions at near-full quality; unbiased logit sampling at <10% overhead | Small gain, large cost saving for H1 | Free (both distributions are already in the trainer) | TIP vs EGAD disagree on teacher entropy; test, do not assume |

Rejected or parked, with reason:
- **Additive weight union with other-base models** (C10 as built): no held-out gain across four designs; REAP/MoBE explain why. Keep union only for same-base lane checkpoints (TSV/SoCE).
- **Depth pruning / ReplaceMe as the main lever**: cheap and safe but does not test the 4x-per-component rule; an extra lever after H1-H4 settle.
- **Full-parameter OPD at 27B**: not feasible on this estate; the 27B gets LoRA/partial-layer OPD only.
- **PPO/GRPO recovery**: ShortOPD reports near zero (0.23 / 1.59 on GSM8K); not worth a cell.

## 4. Schema (experiment ledger)

Files are the record: one `grade.json` + `qf-build-receipt.json` per child under `/mnt/lexar/omni-campaign/research/fold-v2/cells/<cell>/`. The table below is only an index for paired queries (which cell beat which baseline on which block). It would live in the already-deployed `agent_memory` PostgreSQL (`192.168.50.222:25490`, vault key `agent_memory_pg`), raw `asyncpg`/`psycopg2`, parameterised SQL, no ORM. It is additive and optional: if Cyril would rather not add a research table to that database, drop this section and query the JSON files with `jq`; nothing else depends on it.

```sql
CREATE TABLE IF NOT EXISTS fold_v2_cells (
    cell_id            TEXT PRIMARY KEY,                 -- e.g. 'X1b-calib-chat-4B'
    created_utc        TIMESTAMPTZ NOT NULL DEFAULT now(),
    parent_code        TEXT NOT NULL,                    -- 'PARENT-Q35-4B' | 'PARENT-Q35-0.8B'
    hypothesis         TEXT NOT NULL,                    -- 'H1'..'H9'
    recipe_json        JSONB NOT NULL,                   -- qf_build argv + residual/KD config
    evidence_sha256    TEXT NOT NULL,
    child_sha256       TEXT,
    baseline_cell_id   TEXT REFERENCES fold_v2_cells(cell_id),
    param_total        BIGINT NOT NULL,
    mlp_channels_total BIGINT NOT NULL,                  -- matched-budget check
    receipt_path       TEXT NOT NULL,                    -- /mnt/lexar/... qf-build-receipt.json
    grade_path         TEXT,                             -- /mnt/lexar/... grade.json
    screen_hits        INTEGER NOT NULL DEFAULT 0,       -- must be 0 to be promotable
    gpu_card           TEXT,
    gpu_hours          NUMERIC(8,2),
    status             TEXT NOT NULL DEFAULT 'planned'
        CHECK (status IN ('planned','built','graded','promoted','rejected'))
);

CREATE TABLE IF NOT EXISTS fold_v2_grades (
    cell_id        TEXT NOT NULL REFERENCES fold_v2_cells(cell_id),
    block          TEXT NOT NULL,     -- 'review'|'coherence'|'retention'|'semantic'|'diagnostic'
    metric         TEXT NOT NULL,     -- 'bank_pass'|'col_b'|'exact5'|'exact8'|'exact16'|'exact32'|'lcp'|'empty'|'capped'|'repeat'|'math_exact'|'code_pass'|'tool_ok'|'held_rell2_median'|...
    value          NUMERIC NOT NULL,
    denominator    NUMERIC,
    source_path    TEXT NOT NULL,     -- the file the number was read from
    PRIMARY KEY (cell_id, block, metric)
);

CREATE INDEX IF NOT EXISTS fold_v2_grades_metric_idx ON fold_v2_grades (metric, value DESC);
```

Insert path: a ~40-line `fold_v2_ledger.py` that reads `grade.json` and runs `INSERT ... ON CONFLICT (cell_id, block, metric) DO UPDATE` with the DSN from `os.environ.get("AGENT_MEMORY_PG_DSN", "")`, resolved by the caller from the vault (vault key: `agent_memory_pg`). No secret in code or in this file.

## 5. Trade-offs

| Choice | Rejected alternative | Why |
|---|---|---|
| Prove the recipe on OC01 4B / 0.8B first; 27B only after a winning cell | Try the fix on OMNI1 directly | v1 cards are committed (D1 GPU0+4, K3 GPU6, self-gen GPU1+2) and must not slow; a 4B C2 build is 1.8 h on a 3090 vs days at 27B; Codex P2 asks for same-parent matched comparisons before any refold |
| Teacher caches first, then child-only passes on the P40 | Teacher and student co-resident on one card | 4B fp32 parent + child = 37 GB > 24 GB P40; the P40 has no bf16, so fp16 is the only co-resident option and Qwen activations may overflow in fp16 (UNVERIFIED on this parent). Caching the parent once in fp16 with a non-finite check, then training the child alone in fp32, is the boring safe route |
| OPD driver as a new ~300-line script | Extend `streamed_kd_trainer.py` | The trainer is the L-TRAIN engine under version pins (`TOOL_PINS.json`); a research lane must not drift it. The driver imports the trainer's `bucket_terms`/top-k loss helpers instead of copying them |
| Forward-KL mix in OPD (JSD 0.5 + 0.3 forward term) | Pure reverse KL (Thinking Machines) | 2605.16826: pure reverse KL on long sequences collapses entropy and inflates length; our failure is already length/loop |
| Short, loop-truncated horizons (256 -> 1024) | Fixed 2048-8192 rollouts | ShortOPD: same score at 29% of rollout tokens (250M vs 869M; 8.5 h vs 35.9 h) |
| Bank v2 + `folding_retention.py` + semantic scorer as the only gate | Held NLL / token match | GenDistill 2603.26556: a model within 0.2 pp on log-likelihood was 20.8 pp behind when generating; our held NLL is inverted (m); Cyril's rule |
| Parent-generated calibration mixed 50/50 with screened registry rows | 100% parent-generated | Hedges over-confidence; both halves are sha-pinned so arm X1b vs X1c separates them |
| DO-ACP concatenation for K3 at 27B | Keep the C10 additive union | The union showed noise-level gain for 12B params; concatenation + forward-KL has a published +6.3 pp at 4B tokens |
| Optional PG index, JSON files primary | PG as the only record | Campaign convention is receipts on `/mnt/lexar`; a DB outage must not block a lane |
| Research copies of `qf_*` with new flags defaulting to old behaviour | Fork the builder | One builder, bit-identical defaults, the 21 existing self-tests stay green; no second source of truth |

## 6. First three experiments (concrete)

All three run on the 4B ladder, paired against the existing C2K2304 baseline (bank 5/15, column B 2 (m)). Server: venv `/home/cereal/model-surgery-lab/venv/bin/python`; outputs under `/mnt/lexar/omni-campaign/research/fold-v2/`; nothing written to root (70 GB free, floor 60). Every GPU step: a `p40.claims` line + `flock $OC/distill/queues/gpu3.lock`, MemAvailable >= 40 GiB, `nice 15`, `CUDA_VISIBLE_DEVICES=3`. Never GPU0/1/2/4/5/6. All P40 hours are EST: no P40 build of a 4B child has been timed; the 3090 figure is 1.8 h, and the P40 (fp32 only, ~12 TFLOPS) is taken as 3-5x slower. If a 3090 window opens after the D1 seal (about 10-15 22:00Z) the same commands run there at about a third of the time, but only through the owning queue.

### X1. Calibration distribution (H3; cheapest; tests the empty-answer hypothesis)

- **Question**: does the same C2 fold, fitted on chat-templated, parent-generated, thinking-off responses, remove the empties/early-EOS and raise retention at identical keep count?
- **Arms** (all `--mlp-keep 2304 --mlp-group 3`, same lambda split, parent `PARENT-Q35-4B`):
  - X1a = baseline: the existing C2K2304 on `omni-evidence-v1.json`, re-graded with the new retention and semantic blocks (no rebuild).
  - X1b = 98,304 tokens, 100% parent-generated chat turns.
  - X1c = 98,304 tokens, 50% X1b rows + 50% screened registry rows re-rendered through the chat template.
  - X1d (only if X1b or X1c wins) = 4x tokens (393,216) of the winning mix; tests the quantity question.
- **Inputs**: 2,000 instructions sampled from carve `train` rows only (`datasets/registry-walk-2026-09-02.json` with the carve in `$OC/distill/corpus/`), screened against every eval set (hit count printed, must be 0). Parent in HF, fp16 on the P40 with a non-finite check per batch; `enable_thinking=False`; `max_new_tokens 768`; temperature 0.7 (as self-gen); keep only generations that end with `<|im_end|>` and pass the ShortOPD loop detector.
- **Tools**: `qf_evidence_build.py --source chat-selfgen --prompts <jsonl> --generations <jsonl>` (new path: chat template in row rendering + extended screen); `qf_evidence_verify.py` unchanged; `qf_build.py`; `oc_run_frozen_bank.py`, `wavea_review.sh`, `oc_assess.py`; `folding_retention.py`; `semantic_score.py` v0 (math/ARC exact + toolcall-bank outcome; code execution deferred to v1 of the scorer).
- **Card + hours (EST)**: generation 2,000 x ~400 tokens = 0.8M tokens at 150-250 tok/s aggregate (8-16 streams, fp16) = 1-1.5 h. Evidence build: CPU, minutes. Each 4B build on the P40: 5-9 h (fp32 weights 18.6 GB resident; capture rows stream to CPU). Grading: bank 15 + retention 64 prompts, ~1 h per child. Total for X1b + X1c: 14-22 P40 hours; X1d adds one build.
- **Success**: a chat-calibrated child beats X1a on >=2 primary blocks and loses on none; concretely empties 0 and capped <= X1a on the bank, retention@8 >= X1a + 4 of 64, bank >= 7/15. If X1b and X1c both lose, H3 is closed for the 4B and the calibration explanation for the 3/16 empties is recorded as falsified.
- **Not touched**: v1 evidence manifests, OMNI1, any lock other than `gpu3.lock`.

### X2. Multi-token endpoint residual on C2K2304 (H2; Cyril's multi-token objective)

- **Question**: does a rank-8 residual on QKVO (p35 recipe) and/or `down_proj`, fitted to the parent's next H tokens under the child's own prefix, raise free-running retention at 8/16/32 without hurting the review, at <1% added parameters?
- **Arms** (base = X1a, or X1's winner if graded first):
  - X2a QKVO rank 8, H=4 (p35 replication at 64 prompts instead of 8).
  - X2b QKVO rank 8, H=8.
  - X2c QKVO + `down_proj` rank 8, H=8 (the MLP site is where the damage is).
  - X2d = X2b with a ShortOPD-style horizon controller: start H=4, grow to 16 while the exact-prefix EMA rises, shrink on loop detection. This is the adaptive form of the multi-token objective.
- **Objective**: cross-entropy on the parent's H continuation tokens + 0.1 x forward-KL on the parent's top-64+tail at those positions (p35's output-distribution term), with the child teacher-forced on its own previously sampled prefix (prefix sampled greedily from the current child at each of 4 outer steps). Only residual parameters train; folded weights are frozen. Receipt records teacher sha, student snapshot sha, horizon per step, prompt count, parameter delta.
- **Inputs**: 1,024 fit prompts + 256 internal-check prompts from carve `train`, screened; parent continuations and top-64 logits cached once.
- **Tools**: new `research/fold-v2/tools/endpoint_residual.py`, built from the p35 runner lineage (`best/experiments/p35_*` manifests, `hf_target_native_behavior_distillation.py`) with `--sites qkvo,down --rank 8 --horizon N --controller`; teacher cache through the X1 generation path; grading as X1.
- **Card + hours (EST)**: teacher cache 1,280 prompts x (prefix + H + top-64): ~1 h. Residual fit: child 4B fp32 (18.6 GB) + rank-8 adapters + activations at batch 2, 4 outer steps x 1,024 prompts x H <= 16: 2-4 h per arm. Grading 1 h per arm. X2a-X2c: 12-16 P40 hours; X2d about 5 h more.
- **Success**: exact-8 >= baseline + 6 of 64 and exact-16 >= baseline + 3; bank and column B not below baseline; empties/caps not up. If X2a does not reproduce p35's direction at 64 prompts, record it and stop the QKVO line; X2c still runs.
- **Gate**: none; nothing shared is touched. Flag: new code in the loss path; the receipt must pass `folding_retention.py`'s teacher-forcing rejection (it refuses teacher-forced rollouts by design), and the unit tests in section 7 step 8 run before any GPU time.

### X3. Global correction after the local fit (H1; the published repair)

- **Question**: from the same C2K2304 start, how much of the review/retention gap closes with (i) 5M tokens of off-policy forward-KL top-64 KD, and (ii) the same plus 0.5M tokens of ShortOPD-style on-policy KD, LoRA r32 on trunk MLPs only?
- **Arms**: X3a KD-only; X3b KD then OPD; X3c OPD-only (same 0.5M on-policy tokens, no KD warm-up) to see whether off-policy warm-up is needed at all. Each arm also grades the retention block at its midpoint so the curve is visible.
- **Inputs**: KD documents from carve `train` after the data-audit masks (classes 06/08 cut endings masked for EOS and CE; class 03 hex prefix / split trailer stripped), screened. OPD prompts: 2,000 from the same pool, weighted toward the classes where C2K2304 empties or loops in its own X1a grade (never from a bank).
- **Objective**: KD = forward KL on parent top-64+tail with a 0.1 CE floor (the D2 shape), completion-only mask. OPD = per-token loss on the student's sampled tokens: JSD alpha 0.5 over teacher top-100 + tail, importance weights clipped at 2.0, rollout temperature 0.8, loop detector and EMA budget 256 -> 1024 as in ShortOPD, plus a 0.3 forward-KL term (2605.16826). The teacher scores sampled tokens only.
- **Tools**: `streamed_kd_trainer.py` (research copy, sha in the receipt) for X3a; new `research/fold-v2/tools/opd_driver.py` (HF sampler for the student, loop detector, budget EMA, teacher scoring, LoRA step via the trainer's loss helpers); grading as X1.
- **Card + hours (EST)**: the P40 is the wrong card for 4B training (no bf16, ~12 TFLOPS fp32): 5M KD tokens = ~5.5 h of teacher cache at ~250 tok/s, then 10-20 h of LoRA steps. So: **run X3 first on the 0.8B ladder on the P40** (PARENT-Q35-0.8B; a C2 4x child built in X1's style, ~2 h; KD 3-5 h; OPD 4-8 h; `streamed_kd_trainer.py` already has a P40 gradient-parity receipt on the 0.8B) and **queue the 4B version for a 3090 window after the D1 seal** (EST 6-10 h for all three arms). The 0.8B result is a direction signal; the 4B result is the one that counts.
- **Success**: X3b beats C2K2304 by >= 4 bank cases with column B >= 6, retention@16 up, empties 0, caps halved; and X3b > X3a on >= 2 primary blocks (otherwise OPD is not earning its cost at this scale). If X3c is within noise of X3b, the KD warm-up leaves the recipe.
- **Gate (for the caller)**: the 3090 window belongs to v1; it is requested through L-TRAIN's queue, never taken. Until then X3-4B is `planned`, not scheduled.

### Alongside, at no GPU cost

- H5 ridge re-solves on the rows X1 captures (CPU, minutes per layer): lambda by next-layer criterion, no parent anchor, down-colnorm weighting. Results choose the X1d/X4 build flags.
- H4 reader-metric census on the 4B (one P40 capture of 2,048 positions, ~20 min) plus a per-layer sensitivity table from the S3 receipt's per-layer relL2, to draft the first non-uniform keep plan at fixed total (pending the ruling in section 7 step 1).
- No further literature survey until X1-X3 report; the mapping from paper to cell is this file.

## 7. Plan (ordered)

1. **Ruling request to Cyril** (no GPU): is ">=4x per component" judged per layer or on the component's total? It decides whether H4's non-uniform allocation is admissible. Record the answer verbatim as an amendment in `distill/STAGE_D_RULINGS_2026-10-12.md`.
2. **Lane registration** (CPU): add `L-FOLD-RESEARCH` (owner Claude, `transferable`, GPU3 via `p40.claims` only) to `$OC/distill/queues/LANES.json`; post the five-line hand-off on Relay thread `64abe991` so Codex's L-FOLD knows this lane consumes its retention tool read-only and shares findings in `research/fold-v2/X*_RESULTS.md`.
3. **Screen extension** (CPU): edit `qwen-fold/qf_evidence_build.py` (screen against R-Q 210, probe64, smoke packs, toolcall bank, Codex 274/val/test, S8 media prompts; add `--source chat-selfgen`); extend `qwen-fold/qf_selftest.py` with a chat-row atomicity test and a template-token-present test (21 -> 23). Upload to `/home/cereal/omni-campaign/qwen-fold/` only after the local self-test passes; keep the previous file beside it with a `.pre-v2` suffix.
4. **Grading harness** (CPU): `research/fold-v2/tools/grade_cell.py` wrapping `oc_run_frozen_bank.py` -> `wavea_review.sh` -> `oc_assess.py`, `folding_retention.py`, and `semantic_score.py` v0; writes `grade.json`. First measurement: the parent-vs-parent-reload retention noise floor (one P40 hour), recorded in `X1_RESULTS.md`.
5. **X1a re-grade** (P40, ~1 h): the existing C2K2304 through the full harness; the paired baseline for everything after.
6. **X1b / X1c** (P40, 14-22 h): generate, build evidence, build children, grade. `research/fold-v2/X1_RESULTS.md` with every printed number and sha. Time the first build and rewrite every EST in this file from it.
7. **H5 / H4 CPU work** while the X1 builds run, on the rows step 6 captured.
8. **X2a-X2d** (P40, 12-21 h): `endpoint_residual.py` with its unit tests (horizon bookkeeping, prefix-sampling identity, teacher sha in receipt, retention-contract compatibility) before any GPU time; results in `X2_RESULTS.md`.
9. **X3 on the 0.8B** (P40, 10-15 h); **X3 on the 4B** queued for a 3090 window through L-TRAIN's queue after the D1 seal. `opd_driver.py` lands with tests for the loop detector, budget EMA, importance-weight clip and receipt fields.
10. **Decision point**: if X3b (4B), or X1 + X2 together, reach the X03/X08 band, write `research/fold-v2/FOLD_RECIPE_v2.md` and the 27B cost table (K3 via H8; media from the freed budget). If not, run H6 (8-layer blocks with KD between) before anything else.
11. **Ledger**: `fold_v2_ledger.py` + the section 4 DDL, only after Cyril says the `agent_memory` database may hold a research table.

Files to create (local `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/research/fold-v2/`, mirrored to `/mnt/lexar/omni-campaign/research/fold-v2/`): `tools/grade_cell.py`, `tools/semantic_score.py`, `tools/endpoint_residual.py`, `tools/opd_driver.py`, `tools/fold_v2_ledger.py`, `tools/test_*.py`, `X1_RESULTS.md`, `X2_RESULTS.md`, `X3_RESULTS.md`, `cells/<cell>/{qf-build-receipt.json,grade.json}`. Files to edit: `qwen-fold/qf_evidence_build.py`, `qwen-fold/qf_selftest.py`, `qwen-fold/qf_mlp.py` (H5 flags behind defaults that reproduce today's behaviour bit-for-bit), `qwen-fold/qf_build.py` (`--mlp-keep-plan`).

## 8. Risks

| Risk | Impact | Mitigation / gate |
|---|---|---|
| **P40 fp16 overflow on Qwen activations** during teacher caching | silent garbage targets | Non-finite check per batch, abort on first hit, fall back to fp32 with CPU offload (slower) |
| **P40 build time for a 4B child is unmeasured** | X1 could take 2x the EST | Time the first build; if > 12 h, move builds to the post-D1 3090 window and keep the P40 for caches and grading |
| **Interference with v1 cards** | slows D1 / K3 / self-gen | GPU3 only, via `p40.claims` + `gpu3.lock`; MemAvailable gate 40 GiB; `nice 15`; no writes under `$OC/distill/` except `LANES.json` and the claims line; outputs on `/mnt/lexar` only |
| **Eval leakage** through parent generations or OPD prompts | invalid promotion | Extended screen on every prompt pool; hit count printed in every receipt and must be 0; self-gen's 58 BFCL prompts (a known leak elsewhere) are excluded by source |
| **OPD proxy drift** (quantized sampler vs student) | OPD gains vanish | Ladder cells sample from the HF student itself; no GGUF proxy below 27B |
| **New code in the loss path** (`endpoint_residual.py`, `opd_driver.py`) | wrong objective reported as a result | Unit tests before any GPU run; receipts carry teacher/student shas; `folding_retention.py`'s teacher-forcing rejection is the external check |
| **`qf_*` edits drift the sealed v1 builders** | S3/S5 receipts no longer reproducible | New flags default to today's behaviour; the 21 existing self-tests stay green; `.pre-v2` copies kept; upload only after the local self-test |
| **Ruling dependency** (4x per layer vs per component) | H4 cells wasted | Ask first (step 1); H4 builds wait, nothing else does |
| **Adding a table to `agent_memory`** | touches a production database | Needs Cyril's say-so; JSON files remain the record either way; DDL is `IF NOT EXISTS`, additive only |
| **Overlap with Codex L-FOLD** | duplicated work | Hand-off on `64abe991` before step 3; this lane reads Codex tools and writes only its own files |

No backup is needed: no v1 artifact, checkpoint, cache or database row is modified. Items that need explicit approval before they happen: (1) the 3090 window for X3-4B, (2) the `agent_memory` research table, (3) the per-layer vs per-component 4x ruling.
