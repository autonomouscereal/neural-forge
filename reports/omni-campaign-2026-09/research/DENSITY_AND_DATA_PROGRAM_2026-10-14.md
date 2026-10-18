# Intelligence-density and data program for Chimera Omni (standing background program), 2026-10-14

**Written:** 2026-10-15 about 01:20Z (dated -10-14 to match the session that ordered it), by the Claude research lane. This is a standing program. It runs in the background behind v1 and never takes a card, RAM or disk that v1 needs.

**Owner's order (Cyril, 2026-10-14, verbatim):** "[Cyril quote withheld]"

**Evidence marks.** (m) = printed by a command or read in a receipt during this session, with the path. EST = arithmetic on (m) numbers. UNVERIFIED = nobody has printed it.

**Parent documents (this program supersedes neither; it orders and runs them):**

| File (under `reports/omni-campaign-2026-09/research/`) | sha256 prefix | Role |
|---|---|---|
| `FOLDING_PROGRAM_v2_2026-10-14.md` | `e51591f12bffc9be` | Fold hypotheses H1-H9, experiments X1-X3 |
| `DATA_AND_TRAINING_PROGRAM_2026-10-14.md` | `6904c9185d7c5107` | Quality stage, growth plan, token efficiency, bang-per-token metric |
| (critique, returned in-session, not filed) | — | Ten required fixes; adopted in section 3 below |

Where this file and a parent disagree, **this file wins**. It already carries the critique's fixes: no token-match gate, no artifact gate, sealed bank v1, pooled paired tests, no duplication of Codex's lanes, RAM gate at K3's floor.

---

## 1. The three tracks

| Track | Question it answers | Feeds v1 | Feeds v2 |
|---|---|---|---|
| **A. Fold** | Which fold recipe keeps the most capability per parameter at ≥4x? | Recovery dosage for stage T (track A's X3 results) | The v2 fold recipe (`FOLD_RECIPE_v2.md`, written only after the decision point) |
| **B. Data** | Which rows are worth training on, and how do we make more of the best kind? | Masks and exclusions for D2 and T (as amendments, through L-TRAIN and L-DATA) | A verified, versioned, larger corpus |
| **C. Training efficiency** | How much held-out behaviour does each training token buy? | Objective choices for T (KD vs on-policy mix, token budgets) | The default distillation recipe after any fold |

All three share one grading harness (`research/fold/tools/fold_eval.py`) and one rule set (section 3).

---

## 2. Who owns what (Claude leads every track; Codex contributes)

**Rewritten 2026-10-15 (~02:15Z) by Claude as lead.** The first version of this section split the work between Codex and Claude along the reconciled plan (`STAGE_D_RECONCILED_PLAN_2026-10-14.md` §7.1) and predated Cyril's ruling `cyril-claude-leads-all-lanes-2026-10-14` (2026-10-14 ~19:45Z, verbatim in `distill/STAGE_D_RULINGS_2026-10-12.md`): "[Cyril quote withheld]" The earlier text is kept in `research/backups/DENSITY_AND_DATA_PROGRAM_2026-10-14.md.pre-lead-division-0710666d` and on the server beside the `$OC/research/` copy.

**Claude leads every track of this program**, including the fold audit and backlog, corpus repair and splits, and verification. Claude decides, signs off and runs. Codex contributes under Claude's direction whenever it has usage:
- It takes work only from `$OC/distill/queues/CODEX_TASKS.md` (local copy `distill/CODEX_TASKS.md`).
- It reports through the files of record and Relay.
- It starts nothing on a GPU or with large RAM except through Claude's lane queues and locks.
- Its open fold hypotheses are welcome in section 11 (one row each). Claude reads section 11 before choosing the next fold experiment.

| Area | Lead (decides, signs off, runs) | Codex's contribution |
|---|---|---|
| Folding audit, `folding_retention.py`, folding backlog (P0-P2 in `reviews/2026-10-13-folding/OBJECTIVE_HISTORY_AND_RECOVERY_BACKLOG.md`) | **Claude (L-FOLD)** | Backlog items as CODEX_TASKS entries (T12, T13); hypotheses in section 11. `folding_retention.py`, `evaluation.py` and `corpus.py` stay vendored unchanged in `research/fold/tools/vendor/`, sha-checked on every run (`VENDOR_PINS.json`); retention stays a diagnostic row, never a gate |
| Corpus repair, splits, quarantine, rid lists | **Claude (L-DATA)** | Repair and verification batches as CODEX_TASKS entries (T8-T10). Claude signs off every promote / repair / exclude decision; the split decisions in force are `d2-codex-heldout-exclusion-2026-10-15` and `d2-eos-mask-truncated-2026-10-15` (`$Q/D2_DECISIONS_2026-10-15.json`) |
| Verification: independent math / code / tool-call checks, receipt re-checks, verifier agreement on synthetic batches | **Claude** | Independent re-runs when a CODEX_TASKS entry assigns them; their results are inputs, Claude decides |
| Behaviour-graded fold experiments on ladder parents (X1-X3, H4, H8) | **Claude (L-RESEARCH)** | Receipt review on request. X2c (A5, Codex backlog P2) no longer waits for a Codex "go": Claude decides after reading Codex's section-11 row |
| Data-quality scoring and synthetic-data pilots (`quality_scorer.py`, `vmath_pilot.py`) | **Claude (L-RESEARCH)** | Spot checks on assignment; Claude promotes intake batches |
| D1/D2/K3/T training, prereg amendments | **Claude (L-D1, L-TRAIN, L-K3)** | Unchanged: research findings reach them only as written amendments |

**Files are the record; Relay is only the notification.**

| Record | Path | Writer |
|---|---|---|
| This program | `research/DENSITY_AND_DATA_PROGRAM_2026-10-14.md` (+ `$OC/research/` copy) | Claude; Codex appends to section 11 only |
| Codex's work queue | `$OC/distill/queues/CODEX_TASKS.md` | Claude writes the tasks; Codex records progress in them |
| Fold receipts | `/mnt/lexar/omni-campaign/research/fold/<exp>/` (`SUMMARY.json`, `SUMMARY.md`, per-checkpoint grades) | Claude |
| Data receipts | `/mnt/lexar/omni-campaign/research/data/<run>/` (`receipt.json`, `summary.json`) | Claude |
| Tools | `research/fold/tools/`, `research/tools/` (local), mirrored byte-identical to `/mnt/lexar/omni-campaign/research/{fold/tools,tools}/` | Claude |
| Codex evidence | `reviews/2026-10-13-folding/RECOVERY_STATUS_*.md`, `proof_outputs/` | Codex (inputs to Claude's decisions) |
| Next state | A new dated `research/PROGRAM_STATUS_<date>.md` per update; this file is not rewritten except section 11 and dated addenda | Claude |

**Lane:** **L-RESEARCH** is registered in `$Q/LANES.json` (2026-10-15T01:58:47Z; owner claude; lock paths `p40.claims` + `gpu3.lock`; transferable false). The x3 queue is supervised by `x3_v1_yield_watchdog.py`; see the 2026-10-15 addendum at the end of this file.

---

## 3. Rules every track obeys

### 3.1 Grading (owner rules, memory `folding-behavior-first-grading.md`, `omni-fold-numbers-are-diagnostics.md`)

1. **Primary grade: content.** It counts correct answers on held tasks with their own graders, and it tolerates formatting. The pool in `fold_eval.py` pack `16ef286e…` (m): 24 arithmetic, 12 code (48 cases), 8 strict JSON, 61 single and 57 parallel BFCL tool calls = **162 items**. Every miss is classed as an **artifact miss** (empty, capped, looping) or a **content miss**.
2. **Never a gate, grade or target:** token match in any form. This covers free-running exact prefix at 5/8/16/32, R-Q teacher-forced top-1, relL2/cos, and held NLL. Held NLL is inverted on this family (parent 4.290, S3 3.756, S5 3.432, m, fold doc). These are printed as diagnostics and used as tie-breaks only.
3. **Artifacts never count against intelligence.** Empty, cap and loop counts are reported in their own column. A cell is never promoted or killed on them.
4. **Bank v1 (sealed)** answers are written out verbatim for Cyril's frozen review; nothing auto-scores them. The bank v1 file `0ee6bf14…` holds 16 prompts, not the 15 the docs say (m). Bank v2 is a DRAFT and is only an extra block.
5. **Paired statistics.** Candidate vs baseline: lost/gained discordant counts on the 162-item pool, exact McNemar, pooled and per block (`fold_eval.py pair`, m). Small banks alone (7/15 vs 5/15 is p = 0.5) are never evidence.
6. **≥4x per component** to call anything a fold. Nothing below 4x is tested. Where 4x holds, probe 8x.
7. **Fold and recovery are reported apart.** The fold-only child is graded as the fold result. KD and on-policy tokens are recovery cost.
8. **Contamination screen** before any training file is used. Sources are the full review surface: bank v1/v2, R-Q 210, probe64, smoke packs, toolcall bank, Codex val/test and 274 tasks, S8 media prompts. Every receipt prints its hit count. The x3 build dropped 2,885 training documents this way (2,840 vs retention prompts, 45 vs Codex tasks, m).

### 3.2 Resources (v1 first)

| Rule | Value | Source |
|---|---|---|
| Cards | GPU3 via `p40.claims` (+ `gpu3.lock` if present); GPU1/GPU2 only after L-TOOL releases them; never GPU0/4/6 while D1/K3 own them; never GPU5 | Reconciled plan §7.1, §7.4 |
| Start gate | MemAvailable ≥ 70 GiB (K3 identity-gate floor) + the job's own measured peak: 80 GiB for x3 (two 0.8B models, 5.3 GiB each on CPU, m), 70 + 9.5 GiB per llama-server for vmath (RssAnon 9,044,744 / 9,315,580 kB measured, m) | Critique item 7 |
| While running | vmath yields when MemAvailable stays under 70 GiB on two reads; x3 has no running RAM yield (see section 4.1 risk) | `vmath_queue.sh` 1.0.1, `x3_queue.sh` 1.0.0 |
| Priority | Every research job is preemptible: `kill -TERM <pid>` or a `YIELD`/`STOP_QUEUE` file | Queue headers |
| CPU | nice 15, `CUDA_VISIBLE_DEVICES=`; anything > 16 GiB RSS only in a posted window | §7.4.3-4 |
| Disk | Outputs to `/mnt/lexar/omni-campaign/research/` (347 G free at 01:14Z, m); `/` is 70 G free, floor 60, so only small docs go to `$OC/research/` | `df -h`, m |
| Precision on P40 | fp32 compute; P40 FP16 is about 1/64 of FP32 rate (NVIDIA spec, not measured here) | Critique item 8 |

---

## 4. What is running now (server read 2026-10-15 01:14-01:15Z, m)

| Job | pid | Card / lock | State | Receipts / logs | ETA |
|---|---|---|---|---|---|
| **X3 ladder pilot** (`x3_queue.sh` 1.0.0) | 772203 (queue), 781060 (`qf_build_p40.py`, VmRSS 9.35 GB) | GPU3 claimed 00:46:19Z; card 13.9 GiB, 80 % util | Building the C2-style 4x baseline of Qwen3.5-0.8B (3,584 → 896 per MLP). Parity printed "logits bit-identical"; L000-L017 done by about 01:13Z; per-layer fit relL2 0.40-0.58 (diagnostic) | `/mnt/lexar/omni-campaign/research/fold/x3-08b/{queue.log,logs/build.log}` | Build: about 1.5 min/layer EST (18 layers in about 27 min). Whole ladder (2 evals + 3 arms × 4 rungs + grades): **UNVERIFIED**; the earlier 8-12 h guess puts SUMMARY at about 10-15 09:00-13:00Z |
| **vmath synthetic-math pilot** (`vmath_queue.sh` 1.0.1) | 836018 | Waiting; wants GPU1/GPU2 | `GATE WAIT 0 usable card(s)`: GPU1/GPU2 held by L-TOOL; MemAvailable 68.3-75.3 GiB vs 79.5 gate (m, `logs/queue.log`) | `/mnt/lexar/omni-campaign/research/data/vmath-pilot-2026-10-15/` | Starts after the tool-call lane releases (its ETA about 10-15 02:30-03:00Z, UNVERIFIED), if RAM allows. No acceptance rate yet |
| Quality scorer 1.0.4 over the registry | — (done) | CPU | 91,984 rows, 3.6 min CPU, peak RSS 1.92 GB (m) | `/mnt/lexar/omni-campaign/research/data/quality-v1.0.4-registry-2026-10-15/{receipt,summary}.json` | Done |
| D2 holdout/EOS audit | — (done) | CPU | See 4.2 | `/mnt/lexar/omni-campaign/research/data/d2-audit-2026-10-15/d2_holdout_eos_audit.json` (sha `19d47bac…`) | Done |

Other lanes at the same read (m): D1 on GPU0+GPU4 (pid 125746, unit active). L-TOOL llama-servers on GPU1/GPU2 (93-96 %). K3 B2 identity gate running since 19:51:18Z on GPU6 (pid 306480). GPU5 idle and untouched.

### 4.1 Risk to v1 from the running research job (open)

MemAvailable read **68.3 GiB at 01:07:52Z** (vmath log) and 74.3 GiB at 01:15:15Z. That is around K3's 70 GiB identity-gate floor while the x3 build holds 9.35 GB RSS. The B2 identity gate had already started (19:51:18Z) before the x3 build, so it is not waiting on that floor now. But x3 has no running RAM yield. **Rule from now:** if any L-K3 or L-D1 gate prints a RAM wait while x3 runs, preempt x3 with `kill -TERM 772203`. The trainer checkpoints, the claim is released within about 3 min, and the job re-queues. Track A's next queue version (`x3_queue.sh` 1.1.0) adds the same two-read 70 GiB yield vmath already has. That is the first engineering item in section 6.

### 4.2 Findings already produced that v1 must act on (not this lane's to fix)

| # | Finding | Evidence | Who decides |
|---|---|---|---|
| F1 | **D2 trains on all of Codex's recovery val and test rows.** 2,506 val + 2,549 test join to registry rows by exact `id`; all are carve `train` and in D2: 3,178,014 tokens (15.9 % of D2). 4,860 documents (3,100,292 tokens) are trained with CE + KL; 195 are excluded or fully masked | `d2_holdout_eos_audit.json` (sha `19d47bac…`) | Codex (L-DATA: which split is canonical) and Claude L-TRAIN (A2 amendment). **Must be settled before D2's first step**; D2 is not running yet |
| F2 | End-of-turn is trained after cut rows: all 6,936 cut D2 documents (classes 06/08/14) end `<\|im_end\|>\n<\|endoftext\|>`; 6,831 of those targets are trained, none covered by `gt_mask`. D2 teaches "stop right after a mid-word cut" | same receipt | L-TRAIN (mask EOS on cut rows, amendment A) |
| F3 | 71.2 % of D2 tokens get a non-keep advisory action (reject 5.11M, repair 3.68M, exclude-from-T 2.94M, ARC KD-excluded 1.39M, hex deterministic repair 1.12M) | `quality-v1.0.4-registry-2026-10-15/summary.json` | L-DATA |
| F4 | 6,916 near-duplicate clusters, 22,660 non-representative members (1.58M D2 tokens) | `clusters.jsonl` | L-DATA |
| F5 | "515 math rows with reference answers" was wrong: strict reading gives 91 usable referenced rows (85 train) | scorer 1.0.4 receipt | L-DATA; changes vmath's design (only 59 referenced prompts) |
| F6 | The 0.8B parent's cross-entropy on re-rendered D2 documents is 4.0-5.4 nats/token, with positions checked aligned | x3 `prepare_data.log` | Data track: corroborates F3 (the corpus is far from the parent's distribution) |

---

## 5. Tools (all tested; mirrored byte-identical to `/mnt/lexar`)

| Tool | sha256 prefix | Tests (m) |
|---|---|---|
| `fold/tools/fold_common.py` | `9e442bd4657d` | `test_fold_tools.py` 17/17 on server CPU |
| `fold/tools/fold_eval.py` (prepare / run / grade / pair) | `b27a3c963dce` | same |
| `fold/tools/x3_ladder.py` | `695f51cf8c45` | same (incl. STOP-file resume) |
| `fold/tools/x3_summary.py` (kill rule: an arm is closed when its content slope across 0 / 0.25 / 0.5 / 1 / 2M tokens is not positive, `:74`) | `9ecfcb24eb1f` | same |
| `fold/tools/x3_queue.sh` 1.0.0 | `fb43ee62c2ba` | `test_x3_queue.sh` 10/10 |
| `fold/tools/qf_build_p40.py` (runs the unchanged fold builder with P40-unsupported kernels swapped for plain torch) | `8cdfc8cb7279` | live build in progress (parity bit-identical, m) |
| `tools/quality_scorer.py` 1.0.4 | `ac43b0e2e750` | pytest 57 passed on server |
| `tools/vmath_pilot.py` 1.0.2 | `c63e3435e972` | same |
| `tools/vmath_queue.sh` 1.0.1 | `75d299701273` | `test_vmath_queue.sh` 15/15 against stub servers |
| `tools/d2_holdout_eos_audit.py` | `48143ca6ac3f` | produced F1/F2 |

---

## 6. Next experiments, in order

The order is cheap and v1-relevant first, then v2. Each item says what has to be true before it starts.

### Track A: fold

| # | Experiment | Starts when | Cost (EST, UNVERIFIED on P40) | Success test (content pool, paired) | Kill / stop |
|---|---|---|---|---|---|
| A0 | `x3_queue.sh` 1.1.0: running RAM yield (two reads < 70 GiB → SIGTERM, re-queue) | Now (CPU edit + `test_x3_queue.sh`), deployed at the next re-queue, never by killing a live build | 1 h | Tests pass | — |
| A1 | **X3 on 0.8B (running):** KD rungs 0.25/0.5/1/2M; OPD rungs; OPD from KD-1M (+0.25/0.5/1M) | Running | 8-12 h guess | A rung's content beats the 4x baseline at exact McNemar p ≤ 0.05 pooled, and no block (arith/code/JSON/tool) is net-negative by > 2 items | Arm closed if its content slope is not positive (`x3_summary.py:74`). 0.8B shows direction only |
| A2 | **X1b calibration on 4B:** rebuild C2K2304 from 98,304 tokens of parent-generated, chat-templated, thinking-off turns (prompts from carve `train` only), plus **one disjoint-draw C2K2304 replicate** as the calibration noise floor | A1 summary written; GPU3 free; RAM gate | 14-22 P40 h | X1b beats the original **and** the replicate on content, with the replicate-vs-original difference as the floor | No gain over the floor → calibration is not the lever; drop X1c/X1d |
| A3 | **X1 chat-rendered registry arm** (100 % registry rows rendered through the chat template, no self-gen) | Only if A2 wins | 7-11 P40 h | Separates "template" from "self-generated" | — |
| A4 | **H4 sensitivity map on 4B (CPU-heavy, GPU optional):** fold one layer at a time at 4x, measure content and output KL on a 64-item slice; then the compliant non-uniform form: 4x everywhere + 8x on insensitive layers (lower total) | After A2; Cyril's ruling on per-layer vs per-component 4x (asked, open) | one short eval per layer (4B layer count read from config at build time), UNVERIFIED | 4x+8x child matches the uniform-4x child on content at fewer parameters | Builder needs a per-layer `intermediate_size` and export path; GGUF per-layer `n_ff` for qwen3_5 UNVERIFIED |
| A5 | **X2c multi-token endpoint residual** (QKVO + `down_proj`, rank 8, H = 8, child on its own prefix; CE on the parent continuation + 0.1 forward-KL top-64) | **Only after Codex agrees on Relay** (it is Codex backlog P2) and A1 is done | 6-10 P40 h | Content up vs the fold-only child, judged after merge; retention is diagnostic only | The July regime (H16 train / H8 eval) gave 0/24 exact-8 (m, Codex audit). If X2c is flat, Cyril's multi-token objective lives in OPD (A1/A6), not in residuals |
| A6 | **X3 on 4B** (the A1 winner's arm only, at its best two rungs) | Post-D1 3090 window via L-TRAIN's queue (needs approval); never on GPU0/4/6 while v1 uses them | 4B KD/OPD on a 3090, UNVERIFIED | Same as A1 | — |
| A7 | **H8 K3 fusion design** (DO-ACP expert concatenation or logit-only fusion, never additive union: C10 gave 110/210 vs 108/210 for 12.08B params, m) | Design only until K3 is sealed (10-22 → 10-23 EST) | CPU | Written design + a cross-tokenizer mass-loss measurement | — |

### Track B: data

| # | Experiment | Starts when | Success test | Feeds |
|---|---|---|---|---|
| B1 | **vmath pilot (queued):** 500 problems × k = 2, thinking on, 6,144 new tokens, 1.2M-token cap; accept on reference match (59) or two-sample agreement (441, weaker) | L-TOOL releases GPU1/2; RAM gate | Acceptance rate printed in `verify/summary.json`. **Then** a 200-row sample independently checked by Codex's P0 verifier (math executed where possible). Promote only if referenced rows show ≥ 90 % verifier agreement | Intake batch for L-DATA |
| B2 | **Bang-per-token pilot on 0.8B** (uses the A1 harness): same token budget, three corpora: (i) D2 as sealed, (ii) D2 minus scorer `reject` + F1 holdouts + F2 EOS masks, (iii) (ii) + B1 verified rows | After A1 releases GPU3 | (ii) ≥ (i) on content at equal tokens → the scorer's advisory actions are worth applying to T; (iii) > (ii) → synthesis is worth scaling | Amendment proposal to L-TRAIN for T |
| B3 | **Repair pilot:** parent continuations for the 3.68M "repair" tokens (cut-off rows), judged by the scorer's closure check | After B1 shows the guest-serving path works on real cards | ≥ 80 % of repaired rows pass closure + Codex's checks | Intake batch |
| B4 | **Scale the winner** (math → code with sandbox-executed tests → tool calls with Codex's executed outcomes) | B2 says synthesis helps | Each domain batch is promoted by L-DATA | Corpus growth beyond about 94k rows |
| B5 | **PostgreSQL row ledger** (`omni_corpus`), raw SQL via asyncpg/psycopg2, DSN from `os.environ.get("AGENT_MEMORY_PG_DSN", "")`, vault key `agent_memory_pg` | Deferred per critique; only if Codex wants it for L-DATA | — | — |

### Track C: training efficiency

C does not get its own runs; it reads A1, A6 and B2:

| Question | Answered by | Decision it informs |
|---|---|---|
| Content gained per million recovery tokens, KD vs OPD vs KD→OPD | A1 rungs (slope per arm) | Stage T's objective mix and the dose for post-fold recovery in v2 |
| Does on-policy add anything on top of KD at equal tokens? | A1 arm 3 vs arm 1 at matched totals | Whether L-TRAIN's on-policy work (Codex P1 "receipt actual on-policy coverage") is worth its rescoring cost (about 10 h per 25M tokens on a 3090, EST) |
| Content per token from cleaned vs raw vs synthetic data | B2 | Which corpus T trains on |

**Bang-per-token metric (reported for every arm):** net content items gained over the starting checkpoint on the 162-item pool, divided by trained tokens in millions, with the exact McNemar p beside it. Artifact counts and retention are printed next to it but never enter it.

---

## 7. Decision rules

1. **Promote a cell** (it becomes the new baseline for its track): content gain at exact McNemar p ≤ 0.05 on the pooled 162 items, no block net-negative by more than 2 items, a contamination screen of 0 hits, and bank v1 answers written out for Cyril. Retention, NLL, relL2 and artifact counts never block or force a promotion.
2. **Kill an arm**: content slope across its rungs is not positive (A1), or it fails to beat the calibration noise floor (A2), or it loses to its own fold-only starting point after merge (A5).
3. **Stop a job immediately** if any v1 lane prints a wait caused by it: card, lock, or RAM. Research jobs never hold a card a v1 queue asks for.
4. **Escalate to Cyril** only for: the per-layer vs per-component 4x ruling (A4); a 3090 window for A6; anything that would change a v1 artifact.
5. **Nothing from this program enters v1 directly.** v1 changes travel as written amendment proposals to L-TRAIN (prereg amendments + `check-prereg`) and as advisory files to L-DATA.
6. **Retire a hypothesis in writing**: every killed arm gets a line in the next `PROGRAM_STATUS_<date>.md` with its receipt path and sha, so it is not re-run.

---

## 8. How results feed v1 stage T and Chimera v2

**v1 (only through the owning lanes, never directly):**

| Result | Path into v1 | Deadline |
|---|---|---|
| F1 holdout leak | L-DATA picks the canonical split; L-TRAIN writes the exclusion into A2 | Before D2's first step (D2 starts 10-22 → 10-23 EST) |
| F2 EOS after cut rows | L-TRAIN masks the EOS target on classes 06/08/14 (amendment A) | Before D2 |
| F3/F4 scorer actions, B2 verdict | Advisory masks/exclusions for T's corpus, signed by L-DATA | Before T (after D4) |
| A1/A6 dosage | KD-vs-OPD mix and token budget for T's recovery phase | Before T |
| B1/B3/B4 verified rows | Intake batches promoted by L-DATA into T's corpus | Before T's corpus is sealed |

**Chimera v2:**
- **Fold recipe v2** = the promoted cells of A2 (calibration), A4 (allocation), A5 (endpoint residual, if it survives), and A1/A6 (recovery as a required stage, not optional), written as `research/FOLD_RECIPE_v2.md` only after the decision point. It carries a ladder-to-27B transfer plan; 0.8B and 4B results are never claimed as 27B results.
- **K3 path** = A7's design, tested when K3 is sealed.
- **Data** = the verified, versioned corpus from track B, with the bang-per-token table deciding what grows.

---

## 9. Approvals and open questions

| Item | Who | Status |
|---|---|---|
| Per-layer vs per-component "≥4x" (decides A4) | Cyril | Open |
| 3090 window after D1 seal for A6 | Cyril / L-TRAIN queue | Open |
| `LANES.json` entry for L-FOLD-RESEARCH | Parent session (lane owner rule) | Open |
| F1 canonical split + A2 exclusion | Codex L-DATA + Claude L-TRAIN | Open, before D2 |
| X2c go (Codex backlog P2) | Codex | Asked on Relay with this file |
| PostgreSQL row ledger | Codex (L-DATA) | Deferred |

---

## 10. UNVERIFIED

- Every P40 duration beyond the x3 build's per-layer rate (eval, KD/OPD steps, 4B build).
- The X3 SUMMARY ETA (10-15 09:00-13:00Z).
- The L-TOOL release time for GPU1/GPU2 (about 02:30-03:00Z per that lane).
- How well the tool-call parser fits the 0.8B's real output.
- The live SIGTERM preemption path of `x3_queue.sh` (only the trainer's STOP-file path was tested).
- Whether x3's 9.35 GB RSS has delayed any K3 or D1 step (no wait line seen in `gpu6.status` tail at 01:15Z, m).
- GGUF per-layer `n_ff` support for qwen3_5 (A4).
- Cross-tokenizer mass loss for K3 fusion (A7).

---

## 11. Codex's open fold hypotheses (Codex appends here)

*Codex: please add your open hypotheses from L-FOLD here, one row each. Give the id from your backlog, the claim, the cheapest test, and whether you want this lane to run it on a ladder parent. Claude reads this section before choosing the next experiment and will not run a Codex item without your row saying "run".*

| Id | Hypothesis | Cheapest test | Run by | Codex note |
|---|---|---|---|---|
| (Codex P2, from the audit) | Endpoint-aware residual / rollout-informed folding beats local coreset/ridge on the same parent, child, data and suite, with fold and recovery costs reported apart | A5 (X2c) on the 4B ladder | Claude, after Codex's go | |
| (Codex P2) | Genuine MTP shifted targets/heads help recovery once the baseline is recoverable | Isolated ablation after A1 | TBD | |
| | | | | |

---

## Evidence index

- Server read 2026-10-15 01:14:16-01:15:15Z via `ssh_client.py --server ai` (read-only): `ps`, `p40.claims` tail, `nvidia-smi`, `/proc/meminfo`, `df -h`, `x3-08b/{queue.log,logs/*}`, `vmath-pilot-2026-10-15/logs/queue.log`, `gpu6.status` tail, `VmRSS` of pid 781060.
- `STAGE_D_RECONCILED_PLAN_2026-10-14.md` §7.1-7.5, §8 (lane ownership, Relay five-line rule, critical path).
- `reviews/2026-10-13-folding/OBJECTIVE_HISTORY_AND_RECOVERY_BACKLOG.md` lines 72-81 (Codex P0-P2).
- `research/fold/tools/x3_summary.py:74` (kill rule), `fold_eval.py:346,367` (McNemar pairing).
- Data receipts: `/mnt/lexar/omni-campaign/research/data/quality-v1.0.4-registry-2026-10-15/`, `d2-audit-2026-10-15/d2_holdout_eos_audit.json`.

---

## Addendum 2026-10-15 (~02:15Z, Claude as lead)

1. **Lead ruling applied.** Section 2 is rewritten (Claude leads every track; Codex contributes through `CODEX_TASKS.md` and section 11). The section 9 rows change as follows:
   - "`LANES.json` entry for L-FOLD-RESEARCH": done, as **L-RESEARCH**.
   - "F1 canonical split + A2 exclusion": decided, ruling `d2-codex-heldout-exclusion-2026-10-15`.
   - F2: decided, ruling `d2-eos-mask-truncated-2026-10-15`. Both are recorded in `$Q/D2_DECISIONS_2026-10-15.json` (sha `c8d47f22…`), and L-TRAIN applies them before D2's first step.
   - "X2c go (Codex)": Claude decides after reading section 11.
2. **Section 4.1 risk closed by a watchdog, not an in-queue yield (A0 replaced).** `x3_v1_yield_watchdog.py` 1.0.0 runs detached as pid 1457503 (m). On any of these pressure signals it writes its own `STOP_QUEUE` and `V1_YIELD.json` and sends SIGTERM to the x3 queue and its stages:
   - a gpu6 `wait_ram` line;
   - D1, post_d1, d1_seal or d2 waiting on MemAvailable;
   - a gpu6 job in WAIT while MemAvailable < 75 GiB;
   - swap > 90 % used while MemAvailable < 90 GiB;
   - the D1 seal's B1 step waiting for, or holding, the P40 review cards.

   It relaunches the queue only after MemAvailable ≥ 110 GiB has held for 600 s with no pressure and the seal review is off GPU3. The full key list is in its docstring. It yielded the pilot at 01:59:24Z on `swap_full` (swap 99.9 % used, MemAvailable 85.2 GiB, m). `train-kd` saved its resume point at 93,826 tokens, step 11 (m). Before the yield, eval-parent and eval-c2k896 were done; c2k896 graded 0/162 content, with 158 artifact misses and 4 ungraded (m, `eval/c2k896/grade.json` `1ba45b64…`).
3. **Claim bug found and fixed.** x3_queue.sh 1.0.0's claim line said "claim released within ~3 min". `d1_seal_lib.claim_state` (the seal queue's `p40-free`) and `lanes.py` read any "released" word as a release, so `p40-free --gpus 3` printed `P40 FREE` at 01:34Z while pid 772203 held GPU3 (m).
   - **x3_queue.sh 1.1.0** writes `CLAIM` / `RELEASE` lines, creates and holds `$Q/gpu3.lock` together with the claim, parses other lanes' `RELEASE` lines, and starts no stage once a TERM is pending. 1.0.0 ran the next stage to its end when the TERM landed during a CPU grade.
   - 1.1.0 was installed by rename. 1.0.0 is kept as `x3_queue.sh.1.0.0-fb43ee62`, the inode the running queue had open.
   - The watchdog restated the running 1.0.0 claim as a clean `CLAIM` line and held `gpu3.lock` for it until the queue released.
4. **Section 3.2 "Cards"**: x3 1.1.0 always takes `gpu3.lock` while it holds GPU3; it is no longer "if present".
5. **Expected consequence (EST).** With swap full and MemAvailable at 80-91 GiB (m, 01:19-02:00Z), the 110 GiB resume gate is not met. The pilot stays paused until host RAM frees, and from the D1 seal's B1 step until the bank v1 review releases GPU3. A1's SUMMARY ETA (10-15 09:00-13:00Z) no longer holds; the new ETA is UNVERIFIED.

| Tool (new or changed, mirrored to `/mnt/lexar/omni-campaign/research/...`) | sha256 prefix | Tests (m, server CPU, 2026-10-15) |
|---|---|---|
| `fold/tools/x3_queue.sh` 1.1.0 | `0c43d3006180` | `test_x3_queue.sh` (`739f3b22f4e9`) 18/18 |
| `fold/tools/x3_v1_yield_watchdog.py` 1.0.0 | `978d1d7c7646` | `test_x3_v1_yield_watchdog.sh` (`43c382353386`) 51/51: every key positive and negative, and yield and resume end to end against the real 1.1.0 queue with a stub stage |
| `tools/d2_decision_measure.py` 1.0.0 | `fb0c246db2a7` | produced `d2-decisions-2026-10-15/d2_decision_measure.json` (`5fef0bb5…`) |
