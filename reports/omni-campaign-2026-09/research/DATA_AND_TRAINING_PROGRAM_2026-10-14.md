# Data and training-efficiency program for Chimera Omni (v1 stage T and v2 folds), 2026-10-14

**Written:** 2026-10-14, about 23:40Z, by the Claude research lane (design only; nothing was launched, changed or deleted on the server). It builds on four inputs from the same workflow run: the fold-state read, the fold literature survey, the data audit (`/mnt/lexar/omni-campaign/research/data-audit/`) and the data/training literature survey. It extends Codex's L-DATA corpus repair and L-FOLD audit (`STAGE_D_RECONCILED_PLAN_2026-10-14.md` section 7) and does not edit Codex files.

**Evidence marks.** (m) = printed by a command or read in a receipt, path named. EST = arithmetic from (m) numbers. UNVERIFIED = not seen printed.

**Server read for this document** (`ssh_client.py --server ai`, read-only; `date -u` printed `Wed Oct 14 11:22:20 PM UTC 2026`, m):
- GPU0/GPU4: D1 (pid 125746, 9.1 GiB each; `claude-d1-hardened.service` active). GPU1/GPU2: tool-call llama-servers (pids 4054571 / 4058904, 18.1 GiB, 74-95 % util). GPU3: pid 619185, 16.7 GiB, 0 % util = `k3_cache_parity.py digest` of the K3-acceleration lane (claim line `2026-10-14T23:18:14Z GPU3 lane=k3accel owner=claude job=k3-v130-parity` in `/home/cereal/omni-campaign/distill/queues/p40.claims`, m). GPU5 idle. GPU6 100 % (K3 B2, pid 306480).
- MemAvailable **73.2 GiB** (76,791,484 kB), down from the 120-148 GiB read earlier today. `/` 70 G free (97 %); `/mnt/lexar` 487 G free.
- `$OC` is `/home/cereal/omni-campaign` (root disk); `/mnt/lexar/omni-campaign/research/` holds `data-audit/` and `s3sum.py` only.

**Things named by the task that do not exist yet (checked locally, m):** `distill/SPEECH_LANE_PLAN_v2_2026-10-14.md` and its intake tools are not on disk. The only intake artifacts are the June quarantine tree (`C:/Users/cereal/Documents/ai training/dataset_intake_2026_06_26/README.md`, "UNVERIFIED_DO_NOT_TRAIN") and `manual_review_batches/`. Section 2.1 therefore defines the intake contract this stage consumes, so whichever workflow lands the versioned intake can target it.

---

## 1. Problem

D2 will spend 20.0M tokens of which about 46.6 % fall in hard-defect classes (3,000-character cap cuts, unclosed `<think>`, flattened transcripts, pipeline self-reference; `data-audit/full/audit4_report.json`), and nothing in the pipeline scores, repairs or verifies a row before it is sealed into a teacher cache. The owner wants more and better data (from about 94k rows) and more held-out behaviour per training token. This document designs (1) a quality stage for the versioned intake, (2) a growth plan ordered by verified value with throughput arithmetic for the spare cards, (3) token-efficiency changes as candidate prereg amendments, and (4) the bang-per-token metric, with the first pilots.

Nothing here touches v1's running D1, the sealed D2 caches, or the prereg frozen body. The items that could enter v1 (section 4.1) are flagged for the caller to gate.

---

## 2. Design

### 2.0 Components and data flow

```
NAS registry (640 shards)      parent self-gen (P40/3090 llama-server)      Codex L-DATA repairs
        |                                 |                                       |
        v                                 v                                       v
  $OC/distill/intake/<batch>/   <-- versioned, immutable batch: manifest.json + rows.jsonl
        |
        v
  corpus_quality.py (CPU, nice 15): screen -> classify -> dedup -> score -> verify -> plan-repairs -> promote
        |                      \
        |                       -> repair prompts (sealed selfgen prompt schema) -> selfgen_sampler.py run -> new batch
        v
  rows.clean.jsonl + masks.json + report.json (sha-sealed files: the trainer's contract)
        |                                                   |
        v                                                   v
  PostgreSQL omni_corpus (row ledger, verdicts,       distill_corpus_build.py -> qtext_teacher_cache.py
  lineage, gen runs, bang-per-token)                  (unchanged; consume rows.clean.jsonl)
```

Files stay the training contract: the trainer (1.3.4), `distill_corpus_build.py`, the sealer and every receipt already pin sha-named files. PostgreSQL is the system of record for row-level verdicts, lineage and metrics, because a row's status today lives in four unjoined places (registry walk JSON, `carve.json`, Codex `normalized.jsonl`, D2 `documents.jsonl`) and every "which rows are broken and why" question needed a four-pass script.

### 2.1 Intake contract (what the quality stage consumes)

A batch is a directory `$OC/distill/intake/<batch_id>/`, `batch_id = <date>-<source>-<seq>` (for example `2026-10-16-selfgen-think-0001`), containing:

- `manifest.json`, schema `omni.intake.batch.v1`: `batch_id`, `source_kind` (`registry_shard | selfgen | toolcall_selfgen | codex_repair | external`), producer tool name + version + sha256, `rows_sha256`, row and token counts, licence/provenance note, the eval-screen receipt sha it was built under (null if unscreened), and `parent_batch_ids` for repairs.
- `rows.jsonl`: the `staging/rows.jsonl` normalised schema that `distill_corpus_build.py stage` already writes (`curated_shards.normalize_row` field order, newlines preserved, no 8,192-character cut), plus `rid`, `dataset_id`, `route`, and optional `messages` (chat turns with roles) and `tools` (schemas) for native rows.
- Optional sidecars: `tests/` for code rows, `answers.jsonl` for math/ARC, media paths for multimodal rows.

A batch is immutable once its manifest is written. The quality stage never edits it; it writes a sibling `quality/<stage_version>/`.

### 2.2 The quality stage: `distill/tools/corpus_quality.py` (new, CPU only)

Runs at nice 15 with `CUDA_VISIBLE_DEVICES=` (conftest rule X-T1). Sub-commands in order; each writes a receipt and is idempotent per `(batch_id, stage_version)`.

| Step | What it does | Reuses (imported by path, sha pinned in the receipt) | Output |
|---|---|---|---|
| `screen` | Contamination screen against the full review surface: bank v1/v2 + haystacks, probe/smoke packs, R-Q 210, probe64, needle sets, export parity, extension/paraphrase, S8 media prompts, `omni_toolcall_bank_v1`, Codex's 274 tasks and 62 BFCL rids, Codex recovery val/test rids. Exact, 13-gram, short-item containment, literal codes, media sha. | `distill_corpus_build.build_review_index` (`NGRAM = 13`, `:151`, m); adds the four sources the fold-state read lists as missing (R-Q, smoke, toolcall bank, Codex splits) | `screen.jsonl`, `screen_receipt.json` with every source sha. A hit is a hard reject. |
| `classify` | Deterministic defect classes from the audit, promoted from `/mnt/lexar/omni-campaign/research/data-audit/data_audit4.py` with tests: 01 BFCL-gt, 02 teacher-JSON wrapper, 03 hex-flattened / split-label / newline-collapsed, 04 structured-tool minified, 05 ARC answer-only, 06 unclosed think, 07 empty, 08 cap-cut at ~3,000 chars, 09 web boilerplate / adult-spam keywords, 10 omnicoder self-reference, 11 templated synthetic family, 12 memory-trace, 13 mojibake, 14 mid-word cut. One class per row in the audit's priority order. The three false-positive detectors (degenerate repetition, refusal, tool parse_bad) are not promoted. | audit4 code | `classes.jsonl` |
| `dedup` | Exact (instruction+response sha), response-only sha, MinHash word-5-gram / 128 perm / 16x8 LSH at Jaccard >= 0.8 (the audit's parameters), computed against the **whole promoted ledger** (`dedup_signature` table), not only within the batch. Representative = fewest defect classes, then longest closed response. | audit2 MinHash code | `clusters.jsonl` |
| `score` | Value score in [0,1]: (a) template-family penalty (class 11; the top-25 templates cover 46 % of rows), (b) judge rating 1-5 from the parent (greedy, 64 tokens, thinking off; FineVision-style per-turn rubric: correct / complete / on-task / well-formed) through an idle llama-server, (c) closure (ends on EOS, closed think, no cap ride), (d) route scarcity weight (closed reasoning 150 rows, verified code 634, verified math 515). Stored, never a gate by itself. | `toolcall_selfgen` judge pattern (`CHECK_SAMPLING`, verdict per journal row) | `scores.jsonl` |
| `verify` | Checkable domains only. Math: extracted final answer vs `answers.jsonl`, or vs the majority of k parent samples (section 2.3). Code: run the row's tests in `toolcall_selfgen.execute_sandbox`, 30 s, no network. Tool rows: parse, offered function, required/enum/unknown parameters, no text after calls, exactly the 1.1.0 acceptance. ARC: exact grid. Failure = reject, not mask. | `toolcall_selfgen` validators + sandbox | `verify.jsonl` |
| `plan-repairs` | Class 03: deterministic (strip hex prefix + split-label trailer, restore newlines by re-reading the source shard when available; else reject). Classes 06/08/14: **repair prompts** in the sealed self-gen prompt record format (`distill/corpus/selfgen/prompts.jsonl`): the parent continues the row from its cut point, thinking off for 08/14 (`max_new_tokens` 1,024), thinking on for 06 (6,144). | `selfgen_sampler.py` 1.2.0 consumes the prompt file unchanged | `repair_prompts.jsonl` -> a new intake batch with `parent_batch_ids` |
| `promote` | Applies the decision table; writes `rows.clean.jsonl`, `masks.json` (per-row spans in the trainer's 1.3.4 `gt_mask` / `all_mask` file format), `report.json`; inserts into PostgreSQL in one transaction per batch. | `d2_mask_build.py` format | `rows.clean.jsonl`, `masks.json`, `report.json` |

**Decision table (`promote`):**

| Condition | Action |
|---|---|
| screen hit; class 07 (empty); class 13 (mojibake); verify failed; licence missing | **reject** (never leaves the batch) |
| class 03 with the source shard available | **repair-deterministic**, then re-classify |
| class 06 / 08 / 14 | **repair-with-parent** (new batch). Until the repair lands: keep with `trunc_mask` on the last 2 real positions (CE, EOS, UL off; KL kept) |
| class 02 (teacher-JSON wrapper), 10 (self-reference) | **reject for stage T**; for D2 keep with `gt_mask` (02 is already in A2 3(c)) |
| class 05 (ARC answer-only) | keep for verified / RL lanes only; **exclude from KD** (458 docs, 1.90M tokens, 9.5 % of D2) |
| class 11 (templated family) | keep at most 8 rows per cluster, ranked by score |
| dedup non-representative | reject |
| value score below the bottom-tail threshold (set from the P0 distribution; start at 0.35) | reject |
| everything else | keep |

These are the audit's repair levers 1-5 in order of tokens recovered (3.64M + 1.69M cut endings; 1.43M flattened; 1.08M self-reference; 1.16M in-D2 near-duplicates), and each is checkable on the existing D2 documents without a model (pilot P0).

### 2.3 Generation lanes (growth), all through llama-server on the parent's Q4_K_M

Every lane: the parent's own Q4_K_M re-export (`76c1e4c6…`, byte-identical across the self-gen and tool-call lanes; `TOOLCALL_LANE_PLAN` section 4b, m), served by `selfgen_sampler.py serve` on an allowed card (<= 2,000 MiB in use at launch), **sampling only**; every kept token is rescored in bf16 by `qtext_teacher_cache.py build-text` (1.0.6). Rejection sampling keeps only verified rows. Each lane writes an intake batch, so it goes through the same quality stage. The third-party GGUF at `/home/cereal/models/gguf/Qwen3.8-27B-Uncensored-Q4_K_M.gguf` is never used (not tied to the parent digest).

| Lane | Prompts from | Acceptance (hard) | Why in this order |
|---|---|---|---|
| G1 execution-verified tool use | `toolcall_selfgen.py` 1.1.0 plan generator, extended with edge-case turns (wrong units, stale ids, rate limits, partial failures) and k = 2 samples per item keeping only passing ones | the 1.1.0 acceptance + zero-call content rule | 883 calls with 0 missing required params today (audit section 6, m); the lane is built and running |
| G2 verified math with closed reasoning | the 515 registry math rows with extractable answers, the 9,145 proxy-math rows, the 2,382 unclosed-think prompts; thinking on, 6,144-token budget; k = 4 | closed `</think>`, extractable answer, equals the reference where one exists, else the k-sample majority at >= 3/4 | closed traces are the scarcest high-value class (150 rows); OpenThoughts' samples-per-question result (data-lit row 2) |
| G3 verified code with tests | the 634 code rows with multi-line tests + parent-written tests for code prompts (tests first, solution second) | tests pass in the sandbox, < 30 s, no network | checkable; 332 rows in D2 today |
| G4 cut-row repair | section 2.2 plan-repairs (about 8.1k `v9_base` rows + 2,382 unclosed think) | ends on EOS within budget; judge >= 3/5; no screen hit | recovers 5.3M D2 tokens that may be teaching mid-thought stops (hypothesis, UNVERIFIED until P0) |
| G5 instruction diversity, Magpie-style with personas | the parent prompted with its template up to the user slot; the 20 personas x 6 styles of toolcall_selfgen; thinking off | judge >= 4/5, dedup, screen | replaces the 46 % of rows on 25 templates |
| G6 long-context tasks | ProLong-style 40/30/30 QA / RAG / summarisation over same-route concatenations via `distill_corpus_build.py long` logic | answers with token spans; needle recall by construction | the long rung is 31 x 32k rows; low priority for v1 |
| G7 multimodal QA grounded in the media parents | image QA through the parent's `model.visual` tower (the streamed VL teacher, tool (i), `DISTILL_DESIGN` section 3) on the 4,063 image + 2,469 OCR registry rows; audio via the D3e TTS/ASR pair (Codex L-SPEECH-DATA owns the data) | the answer cites a value/region that a second parent pass on the same image confirms | needed for v2 vision grounding; blocked on tool (i); the VL tower's P40 rate is UNVERIFIED |

**Throughput (m and EST):**

| Card / engine | Measured | Per day |
|---|---|---|
| 2 x P40, llama-server 4 slots x 8,192, tool-call lane incl. rejections, judge and simulated backends | 37.3-42 accepted tok/s (`TOOLCALL_LANE_PLAN` section 5, m); 20.2 tok/s in the first minutes after a restart | **3.2-3.6M accepted tok/day** |
| 2 x P40, rendered rate in the smoke | 34.86 tok/s (m) | 3.0M |
| 1 x P40 single stream at 4k | 11.84 tok/s decode (`DISTILL_DESIGN` section 3, m) | 1.0M |
| 1 x RTX 3090, Q4_K_M 27B, 8 slots | **UNVERIFIED**; EST 90-120 tok/s batched | EST 8-10M; needs `selfgen_sampler.py probe` (0.3 h) before any plan relies on it |
| bf16 rescore on a 3090 | 696.19 tok/s (m) | 60M |
| Judge pass on a P40 (prefill 1-3k + 64 decode) | UNVERIFIED; EST 3-5 s per row | EST 17-29k rows/day per card |

**Windows that exist without slowing v1** (`STAGE_D_RECONCILED_PLAN` section 8, EST): GPU1+GPU2 after the tool-call run ends (~10-15 03:00Z) are promised to speech C-s1..s4 for 10-15 -> 10-16, then free. GPU3 is shared through `p40.claims` and is held by the K3-acceleration parity job right now. GPU0+GPU4 are free from about 10-16 06:00Z (after C1/C2/C3) until D2 starts (~10-22 22:00Z) **unless K3-M1 takes them for K3 passes**; K3 is the critical path and wins. GPU5 never.

**Growth ladder** (accepted tokens after verification; tool-call acceptance 83 % of attempts (495/598, audit section 6, m); math/code/think k-sample yield UNVERIFIED, EST 40-60 %):

| Milestone | New verified tokens | Cards | Wall (EST) |
|---|---|---|---|
| M1, by the D2 start (~10-22) | 6-8M: G1 +1.5M, G2 1.5M, G3 0.5M, G4 2-3M, G5 1M | P40 pair after speech (3 days x 3.3M = 10M gross) | 3-4 days |
| M2, by the stage-T start (~10-31) | 25M cumulative | P40 pair + GPU3 when free; 3090s only if K3-M1 leaves them | 7-9 days |
| M3, v2 fold corpus | 60-80M verified + the cleaned registry (~30M after rejects) = **~100M tokens / ~250k rows** | includes 3090 windows after D4 and during T | 25-30 days of P40-pair time, or 8-10 days with both 3090s |

Row target: from 91,984 registry rows (75,335 text) to about **250k rows**, the first 60k new rows from G1-G4 (verified). Past M2 the limit is judge throughput, not sampling, so the judge moves to a 3090 when one is free.

### 2.4 Token-efficiency changes for D2 / stage T (summary; section 4 gives the amendment text)

Both literature surveys and the S8 failure classes (3/16 empty, 13/16 cap rides) point at teacher-forced KD on defective rows as the weak link. In order of expected gain per token: (A) stop supervising cut endings; (B) make the 1.15M on-policy tokens mandatory, with loop truncation and forward-KL mixing; (C) exclude the 1.90M answer-only ARC tokens from KD; (D) stage-T data = the quality stage's output, not the raw 19,669 "clean" rows (31.6 % end mid-word, audit section 4); (E) likely-first curriculum (RSR) for the damaged student; (F) TIP token selection and RS-KD sampled logits as 4B-ladder pilots only; (G) MTP-KD as a v2 fold auxiliary.

### 2.5 Bang-per-token (BPT): the metric

Every stage seal already produces bank v1 content passes (15, artifact-tolerant), R-Q top-1 vs parent (210), `folding_retention.py` exact 5/8/16/32 continuation rates, `omni_toolcall_bank_v1` passes (185), Codex's 274-task supplementary, and from D2 onward rung snapshots at 5M / 12M / 20M / end. BPT is defined on those:

- Behaviour vector at checkpoint c: `B(c) = [bank_v1_content/15, rq_top1/210, ret_exact8, ret_exact16, toolcall_pass/185, codex274_pass/274]`, each in [0,1], **reported separately**; the composite is the unweighted mean, printed beside the components, never a gate.
- `BPT(c_i -> c_j) = (B(c_j) - B(c_i)) / ((tokens_j - tokens_i) / 1e6)` per component: behaviour gained per million training tokens between two rungs of one run.
- Source attribution on the 4B ladder only: paired runs at equal tokens with one source removed or replaced; per-source BPT is the paired difference. On the 27B it is read from rung-to-rung slopes only (no ablation runs on v1).
- Stored in `bpt_measurement` with the eval-set shas and the receipt sha, so a number without a receipt cannot enter the table.
- Teacher-forced numbers (held NLL, KL) are stored as `is_diagnostic` rows and never enter B (owner rule 2026-10-06; GenDistill's log-likelihood inversion, fold-lit row 11; the campaign's own held-NLL inversion parent 4.290 > S3 3.756 > S5 3.432).

---

## 3. Schema (PostgreSQL, raw DDL)

Database `omni_corpus` on the existing `agent-memory-postgres` container at `192.168.50.222:25490` (same server; new database and role; credential **vault key: `omni_corpus_pg`**, created with `credman.py set` before any code reads it). Access through `psycopg2` with `os.environ.get("OMNI_CORPUS_PG_DSN", "")`; the launcher injects the DSN from the vault and never writes it to a file. Raw parameterised SQL only.

```sql
CREATE TABLE intake_batch (
    batch_id            text PRIMARY KEY,
    source_kind         text NOT NULL CHECK (source_kind IN ('registry_shard','selfgen','toolcall_selfgen','codex_repair','external')),
    producer_tool       text NOT NULL,
    producer_version    text NOT NULL,
    producer_sha256     text NOT NULL,
    manifest_sha256     text NOT NULL,
    rows_sha256         text NOT NULL,
    row_count           integer NOT NULL,
    token_count         bigint NOT NULL,
    licence_note        text,
    screen_receipt_sha  text,
    parent_batch_ids    text[] NOT NULL DEFAULT '{}',
    received_at         timestamptz NOT NULL DEFAULT now(),
    status              text NOT NULL DEFAULT 'received' CHECK (status IN ('received','screened','promoted','rejected','superseded'))
);

CREATE TABLE corpus_row (
    row_uid             bigserial PRIMARY KEY,
    batch_id            text NOT NULL REFERENCES intake_batch(batch_id),
    rid                 text NOT NULL,
    dataset_id          text NOT NULL,
    route               text NOT NULL,
    row_sha256          text NOT NULL,
    response_sha256     text NOT NULL,
    prompt_tokens       integer,
    response_tokens     integer,
    lang                text,
    has_tools           boolean NOT NULL DEFAULT false,
    ends_on_eos         boolean,
    think_closed        boolean,
    UNIQUE (batch_id, rid)
);
CREATE INDEX corpus_row_sha_idx ON corpus_row (row_sha256);
CREATE INDEX corpus_row_resp_sha_idx ON corpus_row (response_sha256);

CREATE TABLE dedup_signature (
    row_uid             bigint PRIMARY KEY REFERENCES corpus_row(row_uid),
    minhash             bytea NOT NULL,          -- 128 x uint32, little endian
    lsh_bands           integer[] NOT NULL       -- 16 band hashes
);
CREATE INDEX dedup_lsh_gin ON dedup_signature USING gin (lsh_bands);

CREATE TABLE dedup_cluster (
    cluster_id          bigint NOT NULL,
    row_uid             bigint NOT NULL REFERENCES corpus_row(row_uid),
    jaccard_to_rep      real,
    is_representative   boolean NOT NULL DEFAULT false,
    PRIMARY KEY (cluster_id, row_uid)
);

CREATE TABLE row_verdict (
    row_uid             bigint NOT NULL REFERENCES corpus_row(row_uid),
    stage_version       text NOT NULL,           -- corpus_quality TOOL_VERSION
    defect_class        smallint,                -- 1..15 per the audit; NULL = none
    screen_hit          boolean NOT NULL DEFAULT false,
    verify_kind         text,                    -- math | code | tool | arc | none
    verify_passed       boolean,
    judge_score         smallint,                -- 1..5; NULL = not judged
    value_score         real,                    -- [0,1]
    action              text NOT NULL CHECK (action IN ('keep','mask','repair','reject')),
    mask_kind           text,                    -- trunc | gt | all
    evidence            jsonb NOT NULL DEFAULT '{}'::jsonb,
    receipt_sha256      text NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (row_uid, stage_version)
);
CREATE INDEX row_verdict_action_idx ON row_verdict (stage_version, action);

CREATE TABLE repair_job (
    job_id              bigserial PRIMARY KEY,
    row_uid             bigint NOT NULL REFERENCES corpus_row(row_uid),
    kind                text NOT NULL CHECK (kind IN ('deterministic_03','continue_08','continue_14','rethink_06','resample_math','resample_code')),
    prompt_sha256       text NOT NULL,
    gen_run_id          text,
    out_batch_id        text REFERENCES intake_batch(batch_id),
    out_rid             text,
    status              text NOT NULL DEFAULT 'planned' CHECK (status IN ('planned','sampled','accepted','rejected')),
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gen_run (
    gen_run_id          text PRIMARY KEY,        -- e.g. run-q38-q4km-2026-10-16-think
    lane                text NOT NULL,           -- G1..G7 | repair
    tool_name           text NOT NULL,
    tool_version        text NOT NULL,
    gguf_sha256         text NOT NULL,
    cards               text[] NOT NULL,
    started_at          timestamptz NOT NULL,
    ended_at            timestamptz,
    attempts            integer NOT NULL DEFAULT 0,
    accepted_rows       integer NOT NULL DEFAULT 0,
    accepted_tokens     bigint NOT NULL DEFAULT 0,
    rate_tok_s          real,
    progress_json_sha   text,
    manifest_sha256     text
);

CREATE TABLE bpt_measurement (
    id                  bigserial PRIMARY KEY,
    checkpoint_id       text NOT NULL,           -- ledger id + rung, e.g. OC02-D-Q38-27BU-OMNI1D2@12M
    run_id              text NOT NULL,
    tokens_seen         bigint NOT NULL,
    component           text NOT NULL,           -- bank_v1_content | rq_top1 | ret_exact8 | ret_exact16 | toolcall | codex274 | composite | diag_nll | diag_kl
    value               double precision NOT NULL,
    eval_set_sha256     text NOT NULL,
    receipt_sha256      text NOT NULL,
    is_diagnostic       boolean NOT NULL DEFAULT false,
    measured_at         timestamptz NOT NULL DEFAULT now(),
    UNIQUE (checkpoint_id, component, eval_set_sha256)
);
```

BPT between two rungs is a query, not a stored column:

```sql
SELECT a.component,
       (b.value - a.value) / ((b.tokens_seen - a.tokens_seen) / 1e6) AS bpt_per_mtok
FROM bpt_measurement a
JOIN bpt_measurement b
  ON a.run_id = b.run_id AND a.component = b.component AND a.eval_set_sha256 = b.eval_set_sha256
WHERE a.checkpoint_id = %s AND b.checkpoint_id = %s AND NOT a.is_diagnostic;
```

---

## 4. Candidate prereg amendments (text; for the caller to gate; none in force)

### 4.1 For D2 (only if accepted before D2's first weight update; otherwise they move to stage T)

**A. Truncation mask (`trunc_mask`).** On the 5,164 class-08 and 1,600 class-06 D2 documents (and the 392 class-14/07), CE, EOS and UL are 0 on the final two real positions of the document and on any separator target that follows it; Q38 KL stays. Mechanism: the 1.3.4 `gt_mask` file, built by `d2_mask_build.py` from `documents.jsonl` x `audit4_report.json` doc ids. Cost 0.5 h CPU; no cache change. **Touches v1's D2 data config: gate it.** Reason: 2,376 of 2,382 unclosed-think rows end between 2,900 and 3,000 characters (audit section 3, m); whether the packer places an EOS target after them is UNVERIFIED and pilot P0 prints it first.

**B. ARC answer-only KD exclusion.** `all_mask` on the 458 class-05 documents (1.90M tokens, 9.5 % of D2), the A2 3(b) mechanism. Reason: grid answers without reasoning at 9.5 % of the budget is poor spend; the grids stay available to verified lanes.

**C. On-policy, mandatory.** Replace the prereg D2 clause "if no sampler is ready … the 1.15M come from the off-policy registry" with: the on-policy 1.15M are required; the sampler is the Q8_0 proxy on GPU6 (ruling option ii) or the Q4_K_M proxy on the P40s (option i), rebuilt at every rung; rollouts use the ShortOPD loop detector (periods 1-10 over the last 512 tokens, >= 90 % agreement on a 32-token anchor, >= 3 cycles, >= 64 tokens) with a 1,024-token starting budget; divergence = generalised JSD beta 0.5 (already in the trainer: `onpolicy_divergence = "jsd"`, `streamed_kd_trainer.py:995`, m) plus a forward-KL term at weight 0.5 on the same buckets; prompts from the S8 failure classes' prompt families, every eval set screened. Reason: fold-lit row 1 and data-lit row 1 (ShortOPD 5.71 -> 48.46 where off-policy KD reached 30.52; Thinking Machines 9-30x less compute than SFT), and S8's loops and cap rides are the failure mode on-policy training repairs.

### 4.2 For stage T (a stage-T preregistration, not a D amendment)

**D. Data = `rows.clean.jsonl`**, not the raw 19,669 Codex "clean" train rows. Codex's split and quarantine are adopted in full (reconciled plan section 3.3 item 6), then the quality stage runs on top: reject 02/07/10/13, repair 03/06/08/14, cap class 11 at 8 per cluster, drop non-representative duplicates. Codex val/test (5,055 rows) stay held out; the near-duplicates of train inside val/test (185 / 153 at Jaccard >= 0.8, audit section 4) are removed from **train**, not from val/test.

**E. Likely-first curriculum.** Order epoch 1 by the student's RSR score (mean token rank / mean NLL under the D4 student; one forward pass over 42M tokens at the 696 tok/s class rate = ~17 h on a 3090, EST) ascending; hard rows in epoch 2. Reason: capacity-dependent selection (data-lit row 4): a low-capacity-behaving student gains early from likely data and turns repetitive on unlikely data. Pilot P4 first; the transfer to a damaged 27B is UNVERIFIED.

**F. Mix and loss.** Curated : distill-replay 3:1 with the KL-to-D4 regulariser (design section 7, Cyril's values) unchanged; CE on assistant positions only for native rows (A2 item 2 pattern); LM CE floor 0.1 kept (SlimQwen: KD + LM beats KD alone); on-policy slice >= 10 % of stage-T tokens with the 4.1 C settings.

**G. Verified-lane upweighting.** G1-G3 rows (execution/answer verified) carry data-config `weight` 1.5 vs 1.0 for judged-only rows, capped so verified rows are <= 35 % of an epoch. The `weight` field exists today (`load_data_config` docstring, m).

### 4.3 For the v2 fold campaign (research lane; no prereg yet)

**H. MTP-KD auxiliary.** Keep the parent's MTP head (424,699,392 parameters) through the fold instead of `--drop-mtp`; add beta x [MTP-LM + MTP-KD], beta 0.3 -> 0.1 (SlimQwen Table 3), to the post-fold KD. **I. Chat-templated, parent-self-generated calibration** (thinking off) replaces the 98k-token raw-row set (both evidence manifests hold zero `<|im_start|>` tokens, fold-state read section 1). **J. TIP and RS-KD pilots** at equal cache bytes against top-64 + tail.

---

## 5. Trade-offs

| Choice | Rejected alternative | Why |
|---|---|---|
| Files stay the training contract; PostgreSQL indexes verdicts and metrics | Trainer reads rows from PostgreSQL | The trainer, sealer and every receipt pin sha-named files; moving the contract would re-open 1.3.4 and the cache seals for no training benefit. PostgreSQL answers the "which rows, why" questions that took four audit scripts. |
| Same `agent-memory-postgres` container, new database `omni_corpus` | A new Postgres container | Already running, backed up by `agent-memory-backup`; one more database costs nothing. A memory-container restart would stall intake inserts; inserts are per-batch transactions with retry. |
| `selfgen_sampler.py` / `toolcall_selfgen.py` on the parent Q4_K_M for every lane | A new generation framework or an external API | Both tools have receipts, tests, resume, and the EOS/EOG handling the rescorer needs; one byte-identical GGUF across lanes. Q4 sampling drift from bf16 is accepted because every kept token is rescored in bf16. |
| Hard verifiers before any judge | Judge-only filtering | Executable checks are free and unbiased; the judge is the parent and costs 3-5 s per row. Judge scores rank, they do not gate, except for G5 where nothing else is checkable. |
| Repair cut rows by parent continuation | Re-pull the ~8.1k `v9_base` rows from the source | Re-pull is better and is Codex L-DATA's lever 1; continuation is the fallback. Both land as batches; the deterministic one wins dedup representative. |
| Likely-first curriculum only after a 4B pilot | Apply it to stage T directly | The published result is on 1.5B-8B students; a wrong ordering costs an 87-201 h stage. |
| On-policy mandatory in D2, gated | Keep the prereg fallback | The fallback silently converts the one on-policy slice into more off-policy KD; both surveys rank OPD first. It touches v1, so it is a gated amendment. |
| BPT per component, no composite gate | One scalar "intelligence density" | No 2025-2026 source defines such a metric; the owner's rule is behaviour-first and never token-match gated. |
| Plan throughput on the P40 pair | Plan on the 3090s | The 3090 Q4 serving rate is UNVERIFIED and the 3090s belong to K3-M1 / D2 if those need them. |

---

## 6. Plan (ordered; CPU steps need no card; every GPU step goes through `p40.claims` + `gpu<N>.lock` and the MemAvailable gate)

1. **Database and vault key** (needs approval: touches the production memory container). `credman.py set omni_corpus_pg "<pw>"`; `CREATE DATABASE omni_corpus; CREATE ROLE omni_corpus LOGIN ...;` run `distill/tools/omni_corpus_schema.sql` (new; section 3). Receipt `distill/receipts/omni_corpus_schema_<date>.json`.
2. **`distill/tools/corpus_quality.py` 1.0.0 + `test_corpus_quality.py`** (CPU). Section 2.2 sub-commands; imports `distill_corpus_build.build_review_index`, `toolcall_selfgen` validators/sandbox and the audit4 detectors by path, shas pinned in every receipt. Register in `TOOL_PINS.json` (card X-T3).
3. **P0 pilot: dry run on the sealed D2 documents** (`$OC/distill/corpus/text/documents.jsonl`, sha `ce6ff058…`). CPU, nice 15, ~2 h EST. Success: class counts within 1 % of `audit4_report.json`; 0 Claude-bank hits (the audit found 0); the trunc-mask position list is produced; **prints whether a packed EOS target follows a cut document** (settles 4.1 A). Output `/mnt/lexar/omni-campaign/research/quality/p0/` + PostgreSQL rows.
4. **Write 4.1 A/B/C as a DRAFT A3 block** under `## Amendments` of `distill/PREREGISTRATION_OMNI_STAGE_D_v1.md` (local + `$OC/distill/` mirror); `check-prereg` must still print `c24d21e0…1370`, 15,339. **Gated: affects v1 D2.**
5. **P1 pilot: cut-row repair, 200 class-08 `v9_base` rows.** Prompts from `corpus_quality.py plan-repairs`; `selfgen_sampler.py run` on **GPU3** after the k3accel parity job releases its claim (claim line + lock; MemAvailable >= 40 GiB). 200 x ~800 new tokens = 160k tokens at 12-18 tok/s = **2.5-4 h** EST. Success: >= 90 % end on EOS within 1,024 tokens; judge >= 3/5 on >= 85 %; 0 screen hits. Also measures the single-P40 batched rate that replaces the EST in section 2.3. Rescore later (0.1 h on a 3090).
6. **P2 pilot: closed reasoning traces, 500 prompts (G2).** GPU1+GPU2 after the tool-call run releases them and the speech C-s claims end (~10-16), or GPU3. Thinking on, 6,144 budget, k = 2 first: 500 x 2 x ~2,500 = 2.5M gross tokens at ~40 tok/s = **17 h** on the P40 pair (k = 4 is 35 h; decide from the k = 2 yield). Success: >= 60 % of prompts yield a closed trace with an extractable answer; on the 515-row referenced subset, accuracy >= 80 %. The yield sets the G2 budget.
7. **P3 pilot: verified code, 300 rows (G3)**, same cards: 300 x 2 x ~1,200 = 0.7M tokens, ~5 h EST. Success: >= 50 % pass their tests; `execute_sandbox` has no network; 0 screen hits.
8. **G1 extension:** `toolcall_selfgen.py` 1.2.0 adds edge-case categories and `--samples-per-item 2`; tests; a second 1.5M-token run after the first finalises. P40 pair, ~11 h at 37 tok/s (m rate). The 274-task screen is applied post hoc as in the reconciled plan section 3.3 item 1.
9. **P4 pilot: bang-per-token on the 4B ladder.** Parent `PARENT-Q35-4B`, child `C2K2304` (bank 5/15, column B 2, paired review exists). Two KD runs at **2M tokens each**, identical row plans except the source: (a) the current D2 mix sampled down; (b) the quality-stage `rows.clean` mix with the trunc/ARC masks. `streamed_kd_trainer.py` 1.3.4. Card: a 3090 in the 10-16 -> 10-22 window if K3-M1 leaves one (~1 h per run at the 4B rate, UNVERIFIED), else GPU3 fp32 (P40 parity receipt exists for the 0.8B; the 4B P40 rate is unmeasured, EST 4-8 h per run). Score: frozen 15-case review, coherence, `folding_retention.py` 5/8/16/32, toolcall bank; store in `bpt_measurement`. Success: (b) >= (a) on bank content and exact-8 retention with no new failure class; otherwise the rules are re-read before they touch stage T.
10. **BPT hooks at stage seals:** `distill/tools/bpt_record.py` (new, CPU) reads the seal package (bank, R-Q, retention, toolcall bank, 274) and inserts `bpt_measurement` rows with receipt shas; run at the D1 seal (baseline), every D2 rung, D3, D4, every T rung.
11. **Stage-T preregistration** `distill/PREREGISTRATION_OMNI_STAGE_T_v1.md` (new) with 4.2 D-G, written after P4 and before T's first step; frozen-body pin via `stage2_training_receipt.py`.
12. **Growth to M2 / M3** per section 2.3: each run an intake batch, each batch through the quality stage, each promoted set rescored and sealed through the existing `distill_corpus_build.py` -> `qtext_teacher_cache.py` path.
13. **Relay note to Codex (L-DATA / L-FOLD)** in the five-line format of the reconciled plan section 7.3: the quality stage adopts its quarantine labels and 274-task screen; asks for the `v9_base` re-pull (lever 1) and the class-03 source shards; shares P0/P1 receipts. Sent by the parent session, not by this document.

**Files to create** (under `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/distill/`, mirrored to `$OC/distill/`): `tools/corpus_quality.py`, `tools/test_corpus_quality.py`, `tools/omni_corpus_schema.sql`, `tools/bpt_record.py`, `tools/test_bpt_record.py`, `intake/README.md` (the section 2.1 contract), `PREREGISTRATION_OMNI_STAGE_T_v1.md`; pilot receipts under `/mnt/lexar/omni-campaign/research/quality/<pilot>/`.

---

## 7. Risks

| Risk | Effect | Mitigation / gate |
|---|---|---|
| **Step 1 touches the production `agent-memory-postgres` container** (new database + role) | A wrong grant or a heavy insert could affect agent memory | Needs approval. The role has rights on `omni_corpus` only; inserts are batched per intake batch; no DDL on `agent_memory`. Take an `agent-memory-backup` checkpoint first. |
| **4.1 A/B/C change v1's D2 data config** | D2 is preregistered; an unamended change breaks the pin | Written only as a DRAFT amendment, gated by the caller; applied at load through 1.3.4 masks, so sealed caches are untouched. |
| P40s are promised to speech (C-s1..s4) and GPU3 to the k3accel lane | P1-P3 slip | GPU steps claim through `p40.claims` + lock and wait; nothing pre-empts. |
| MemAvailable is 73 GiB now, not 120-148 GiB | The 40 GiB background gate holds; the K3 pass gate (>= 70 GiB) is at the line | CPU steps run at nice 15 and under 16 GiB; sampler/judge servers use GPU memory, not host. |
| The parent repairs its own truncated rows (self-distillation amplifies parent quirks) | Style collapse on `v9_base` rows | Verified lanes outrank judged-only rows; Codex's source re-pull supersedes continuations in dedup. |
| Judge, sampler and teacher are all the parent | No independent signal for G5 | G5 capped (1M at M1) and ranked below verified lanes; a second judge (K3 through its cache, or `Qwen3.6-27B-Q4_K_M.gguf` present at `/home/cereal/models/gguf/`, m) is a stage-T option, cost UNVERIFIED. |
| 3090 serving rate and 4B P40 training rate are UNVERIFIED | Growth dates are wrong | Both are measured inside P1 and P4 before any plan relies on them. |
| Eval leakage through generated prompts (the self-gen cache already holds 58 BFCL prompts, audit section 5) | Held-out claims weaken | The screen in step 2 includes the four missing sources; every batch carries its screen receipt sha; an unscreened batch cannot be promoted. |
| Root at 97 % (70 G, floor 60) | A pilot writing to root trips the floor | Pilot outputs go to `/mnt/lexar/omni-campaign/research/`; GGUFs stay in declared scratch and are deleted by the drivers. |
| Owner rule: PostgreSQL only, no file backups of the ledger | A lost database loses verdict history | Every verdict row carries its receipt sha, so the ledger is reconstructible from receipts; the database lives in the memory container's existing volume. |

**Needs approval before any step runs:** step 1 (production container), step 4 (v1 prereg amendment), any claim on GPU0/GPU4 (K3-M1 has priority), and the Codex re-pull ask (step 13).

**What this document does not verify:** the 3090 Q4 serving rate; the yield of k-sample math/code/think generation; whether a packed EOS target follows cut documents (P0 prints it); the P40 time for a 4B KD run; the cost of a second judge; the live tool-call `progress.json` (not found at depth 6 under `/`; the run dir is deeper or named differently); and Codex val/test held-out status for stage T beyond the audit's near-duplicate counts.
