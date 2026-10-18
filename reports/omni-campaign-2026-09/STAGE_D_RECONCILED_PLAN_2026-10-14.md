# Chimera Omni stage D: reconciled Claude + Codex plan, 2026-10-14

**Written:** 2026-10-14, about 19:00Z, by Claude. It merges about 20 Codex recovery reports (Relay thread `64abe991`, 2026-10-13 14:08Z to 2026-10-14 00:17Z), Codex's artifacts, and six decision lanes (X01-X06). Those lanes re-ran Codex's tools and read the server read-only. The result is one plan for recovering OMNI1 through D1-D4, then T, then Q4.

**Previous state:** `STATE_2026-10-14_WAVE3.md` (about 18:00Z). This document does not replace it. It is the plan; WAVE3 is the state.

**Evidence marks**
- **(m)** = I or a lane saw it printed, with the command or file named.
- **(cx)** = Codex reported it and nobody re-computed it. Treat it as UNVERIFIED.
- **EST** = estimate.
- **UNVERIFIED** = not seen printed by anyone on the Claude side.

**Server writes made for this document:** this file only (uploaded to `$OC/distill/`). The server reads were read-only, at 18:53:17Z and about 18:58Z. I launched, stopped and changed nothing.

**Live at 18:53:17Z (m)**

| Work | Card(s) | State |
|---|---|---|
| Hardened D1 (`claude-d1-hardened.service`, MainPID 125746) | GPU0, GPU4 | ActiveState=active, NRestarts=0, 9,106 MiB per card. Last line: `L03 (block 3, gated_gqa) imported advance 178.5s` at 18:52:40Z, i.e. still replaying the L00-L10 prefix. rss 79.8 GiB, cgroup 79.9/128 GiB, oom_kill 0. |
| K3 B2 bf16 review pass (pid 4606) | GPU6 | `L69 KDA … compute 108.4s` at 18:51:20Z. |
| Tool-call self-gen | GPU1, GPU2 | 159 accepted / 34 rejected, 311,836 of 1,500,000 tokens, 40.49 tok/s, eta_hours 8.15 (18:51:46Z). |

- GPU3 has no compute process. The K3 acceleration profile on it has ended.
- GPU5 is unused.
- MemAvailable is 126,234,732 kB (about 120 GiB).
- `/` is at 97% with 70G free; `/mnt/lexar` has 624G free.

---

## 1. What Codex built that Claude adopts or merges (with credit)

Codex worked from 10-13 14:08Z through 10-14 while Claude had no usage. Cyril's relayed instruction (Codex 043544ed) was "[Cyril quote withheld]". The items below go into the plan. Each one names Codex's artifact and the Claude-side check behind it.

**Source paths**
- Codex branch: `codex/work/codex/chimera-training-recovery-oct13-eae1b72e2c3f`, HEAD `81f54988`, in the worktree `C:/Users/cereal/Documents/ai training/.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f`.
- Codex ledger: `C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/RECOVERY_STATUS_2026-10-13.md`.
- Remote prep: `$OC/distill/codex-preparation-01a1326f-20261013/`.
- Every Codex artifact is **read-only** for Claude. Claude copies what it needs, keeping the source sha.

### 1.1 Already in use: D1 data, D1 launch policy and the D1 prefix itself

| Codex work | Where | How Claude uses it | Check |
|---|---|---|---|
| **D1 source audit**: the old rows [0,977) prefix holds 3,010 docs / 1,498,741 tokens that are incompatible with the recovery splits. The original carve has 0 violations. | `proof_outputs/chimera_d1_source_span_audit_oct13_v2/report.json` | Basis of prereg amendment A1 | X04 re-ran Codex's auditor (sha `72028ff6…`) on the same 5 inputs. It printed 3010 / 1498741, test 673/371547, val 680/351492, and `source_spans.jsonl` `7e72a281…` identical (m) |
| **D1 clean successor** (917 fit / 62 held rows, 3,752,131 fit tokens) | `$OC/distill/codex-preparation-01a1326f-20261013/d1-source-successor-v2/`: ids `74f79199…`, manifest `a488e75b…` | A1, and the live hardened D1 argv | X01 ran `sha256sum` on the server (m). The trainer log prints `ids 979x4096 (fit 917 rows, held 62)` (m) |
| **D1 rungs L00-L08** (Codex run v35, trainer 1.2.8) and **L09-L10** (c36-r2, 1.2.9) | `/mnt/lexar/omni-campaign/distill/d1/OMNI1D1-rungs-canonical/` (SHA256SUMS `6cfb7ef9…`) | **The canonical prefix of Claude's hardened D1.** 11 of the 64 trained layers are Codex's work. | X01: PROVENANCE.json names v35/c36. L00.json sha `d23cbd31…` equals Codex's own first-rung receipt. Both argv use successor-v2 ids, `--held-rows 62`, lr 2e-5, 32,768 tokens/step, 2 epochs, `--no-pin` (m) |
| D1 resource policy: no K3 cache dependency; `--no-pin` | Codex `resource_policy.json` | A1 section 2 | A1 text |
| Pre-D1 clone | `/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1` | D1 base | Digest `8edf50c4…` (cx). It was not re-hashed on the Claude side (UNVERIFIED) |

### 1.2 Adopt now: data and tool calling

| Codex work | Where | Claude adoption |
|---|---|---|
| **Native BFCL re-render**: 628 BFCL v4 rows now carry structured `tool_calls` (506 train / 60 val / 62 test). They are packed with the **parent's own chat template**. | `corpus.py recover_bfcl`/`adapt_bfcl_schema`, `training.py pack_native_chat`. Packed train rows: `$OC/distill/codex-preparation-01a1326f-20261013/native-text-v3-train/assistant_text.safetensors` | The 506 train rows replace the 761 JSON-in-prose BFCL docs in D2 (§3.3). X05 printed `rows 506 tokens 420725 assistant 49213 max len 2109` (m). Template: Codex `template_sha256` `924d3fcc` is the canonical-JSON digest of the parent `chat_template.jinja` (file sha `c3cf9e34`), the same template Claude's self-gen uses (X05 `fmt.py`, m). **No re-render is needed.** |
| **Execution-backed tool seed**: 504 episodes / 896 calls, split 168 train / 168 val / 168 test | `proof_outputs/chimera_tool_seed_oct13_v2/report.json`. Train cache `3ed3f303…`, sealed 2026-10-14T00:35:00Z, 113,026 tokens, 26,634 positions, all finite (X05, m) | **Test 168:** held-out tool probe at every stage seal. **Val 168:** stage-T dev set. **Train 168:** format fixture and stage-T format slice, not D2 KD data (§2.4). |
| **Held-out evaluation set**: 274 tasks (62 BFCL + 168 tool-seed test + 44 operator-written text) | `proof_outputs/chimera_recovery_eval_combined_oct13_v1/manifest.json` | **Supplementary** result beside Claude's banks at every stage seal (§3.4). The 212 non-BFCL tasks have no source group in Claude's corpus (X04, m). The 62 BFCL tasks are valid only once their 69 source rows are excluded from D2 (§3.3). |
| **Recovery splits and quarantine** (19,669 clean train; 5,055 val/test; quarantine for CURRENT_TARGET rewrites and flattened tool records) | `$OC/distill/corpus/text/documents.jsonl` (`ce6ff058…`) joined with Codex's `normalized.jsonl` | **D2:** masks the ground-truth terms only on rewrite and flattened-tool docs (§3.3). **Stage T:** adopts the split and quarantine in full. X04 extended the join to all 35,985 D2 docs; missing_join 0; clean train 19,669 equals Codex's count exactly (m). |
| **Speech preparation**: Qwen3-ASR-0.6B + Qwen3-TTS-0.6B teachers; 201/208 exact admission; replay index; H3 audio 40 Hz clock (800 samples/frame); audio VAE round trip; LibriSpeech ASR-leak repair (884 rows) | `$OC/distill/codex-preparation-01a1326f-20261013/speech-teachers-v1`; `reviews/2026-10-13-folding/recovery-preparation/speech-*`; `chimera_natural_speech_repair_oct13_v3` | Seed data, judge models and parity reference for the D3e speech lane (§5). The 0.6B LFS files match Codex's pins (codex-artifacts reader, m). |
| **Text capacity proofs** (mb4 / mb8 / long singleton) and **video top-64 teacher parity** (792/792 positions; LSE diff 0.0) | `recovery-preparation/qtext-capacity-v4/proof.json` (`d5b60e4d…`, m); `video-k64-runtime-v1/validation.json` (`d0bcdeab…`, m) | Capacity inputs for j6 and later caches. The GiB figures are (cx). |
| **qtext execution-shape finding**: identical inputs give 1,201-1,204 / 1,218 top-1 matches when batch or padding changes; singleton repeats are bitwise | `recovery-preparation/qtext-shape-audit-v1/REVIEW.md` | Recorded as a known property of bf16 teachers, the same class as K3 B2 (§6). **The sealed caches stay valid:** each is one bf16 rounding path of the parent, not a wrong one. |
| **BF16 top-64 mass > 1** (up to 1.0033815; 20/792 above 1.001) | Codex a3a8e5d1 (cx) | Claude's trainer does not reject such rows. `bucket_terms` clamps the tail at 0: `p_tail = (1.0 - p.sum(dim=-1)).clamp(min=0.0)` (`distill/tools/streamed_kd_trainer.py:1144`, m). The excess (≤ 0.34%) stays in the teacher buckets. Card D2-T5 adds a test that pins this behaviour. Neither clamping nor rescaling of stored values is adopted. |
| **Folding audit and `folding_retention.py`** (free-running 5/8/16/32-token agreement, prefix length, EOS / empty / cap / repetition rates; rejects teacher-forced inputs) | `reviews/2026-10-13-folding/OBJECTIVE_HISTORY_AND_RECOVERY_BACKLOG.md`; commit `67b32b54` | A **diagnostic** row at every stage seal (D1..D4), never a gate (owner policy 2026-10-06). Codex keeps the folding lane (§7). |
| **K3 B2 bisect**: first divergence at L2 | `$OC/distill/parity/k3cache-2026-10-13-bisect-gpu6/bisect_sha_compare.json` | Closed by Claude's mechanism receipts (§6). |
| **Allocator orphan reconciliation**, docker-inspect timeouts failing closed, directory-only bind mounts | Codex allocator `c871cbd3` | Kept as Codex's launcher. It must also honour Claude's queue locks (§7.4). |

### 1.3 Merge later, ported onto Claude's engines (Codex specifies, Claude integrates)

| Codex work | Merge target |
|---|---|
| **Structured D2 v2 patch** `streamed_supervision_patch`: assistant-only target masks, raw qtext `verify_shard` at load, row-plan sha in data identity, sampler rewind on non-finite steps (`d1dafa08…`). It is built on trainer 1.3.2 (`77091f8a…`) and refuses the live 1.3.3 (`ReadinessError streamed trainer baseline moved`, X03, m). | Port into `streamed_kd_trainer` **1.3.4** (card D2-T1). Do not adopt the one-replica runner: it refuses K3, on-policy and self-gen sources, and would double D2 wall time (X03). |
| **D3 findings**: D3a `chart_out` owned but unused by the stated loss; D3b fixture timing; D3c 64,000 written vs 65,536 allocated rows; D3d needs aligned native-vocabulary answers | Checks in `d3_hybrid_module` 1.1.0 (card D3-T2). Claude's module already uses `ACE_CODES = 64_000` (`d3_hybrid_module.py:122`, `:1040`, m). Whether the 1,536 reserved rows stay bit-frozen is UNVERIFIED; a test is in D3-T2. |
| **HY3D**: `posterior.mode` recovered 7/27 oracle points, while seeded `posterior.sample` recovered 125/125. Native time is `t = 1 - sigma`. | Card C-V1: check whether Claude's sealed HY3D cache and the D3 HY3D lane used `mode` or `sample`, and which time convention. UNVERIFIED either way. |
| **H3**: measured 40 Hz audio frame grouping; 36 real frames + 1 zero slot; 96 px output `[24,7,6,6]`; Comfy namespace-package fix `b6cd80a9` | Inputs to the H3 production cache and to the type-3 chart amendment (A3). |
| **Image-edit batches** (32) and **temporal video** sources (24) | Optional D3 lanes under A3. Data owner: Codex. Engine: `d3_hybrid_module`. |
| **D4 guards**: distinct optimizer owners per bank; refuses a base that already contains D2 (no double D2 apply); LR 0.3 × producer LR; a lane with fewer than 300 distinct items uses each once and never duplicates ("do not manufacture 300 distinct pairs from 32 caches"); audio CE before speech flow | D4 driver checks (card D4-T1). |
| **NativeControlRowAdapter** (17 zero control rows 248320-248336 in embed and head) | Parked. An optional post-T lane, owned by Codex (§2.5). |

---

## 2. What Claude keeps, and why

1. **Hardened D1, untouched** (X01, keep-claude). It resumes from Codex's own L00-L10 on the same successor ids and recipe, so the lineage is not mixed (m, X01). Relaunching under Codex's allocator would cost 3-4 h of prefix replay per collision. Codex's R6 lost its optimizer authority when the K3 B2 job and a Claude pytest started (Codex R6 result doc). A second D1 lineage is retired.
2. **K3 cache tool frozen at `2aae4dc3`, and queue 1.3.0** (X06, keep-claude). The L2 divergence is explained (§6). Changing the call shape would void B2 and the identity gate, and buy nothing.
3. **D2 engine `streamed_kd_trainer` (two-replica DP, mixture, K3/self-gen/on-policy/long sources) and every sealed cache.** The main D2 lane keeps full-sequence Q38/K3 KL on every stored position. The caches are sealed unmasked (`mask_source "none (every position below the row length)"`, X03, m). The student is being recovered toward its own parent everywhere, and masking would throw away 24% (4.83M) of paid positions (X03 scan: assistant fraction 0.7586, m). Codex's own amendment B keeps "the original replay lane" unchanged.
4. **Claude's tool-call self-gen as the D2 tool core.** It has 542 parent-written functions, 48 domains, 8 behaviour categories and executed sandbox tools, with content judging. Codex's 168 train rows are scripted, cover 2 tools and teach a "Recorded value: […]" dialect. Codex's own report calls them "Operator-authored synthetic format coverage" (X05, m).
5. **Claude's text and tool path through the parent Qwen template tokens.** The native control band [248320, 248337) is zero in OMNI1 and in the clone (X02 row read, script sha `0459b337…`, m). It is unreachable in Claude's deliverable for three reasons: corpus refusal of ids ≥ 248,320, GGUF truncation to 248,320 rows (`chimera_gguf_export.py:23-24`), and the p11 band mask. The template tool rows 248058/248059/248066/248067/248046 are non-zero and trained. Codex's v29 native packets (max label 248328) are not D2/T data.
6. **Claude's preregistered evaluation stays the gate**: frozen eval split `626093ca` (1,948 rows), bank v1/v2, R-Q, P11, `omni_toolcall_bank_v1`. Amendment A1 adopted Codex's split for D1 data only, and Codex's audit calls the overlap "prospective … not retroactive proof".
7. **`d3_hybrid_module` stays the D3 engine of record.** Codex's native trainer supplies objectives and parity references, ported into it (§1.3, §5).

---

## 3. Reconciled data plan

### 3.1 Corpus

| Source | Rows / tokens | D2 role | Masks |
|---|---|---|---|
| Registry text, 20 sealed qtext shards (`text-00000..00019`) | 4,883 rows × 4,096 = 20,000,768 positions (X03, m) | Main off-policy lane, 18.85M draw | KD on every stored position. 69 rows excluded (§3.3). Ground-truth terms off on rewrite and flattened-tool docs. All terms off on JSON-in-prose BFCL docs. |
| K3 rows (4 passes) | 1,991,482 real Qwen tokens | K3 lane (prereg) | Unchanged |
| Self-gen rescore (sealed `fe1f0784…`) | 1,008,982 tokens | Unchanged | Already completion-only (`store_mask` = `target_mask` shifted, X03, m) |
| On-policy (prereg option) | 1.15M | Unchanged | JSD β 0.5 |
| Long | 31 × 32,768 | Long rung | Unchanged |
| **New: tool-call self-gen** (`rescore/toolcall-q38-2026-10-14.safetensors`) | 1.5M rendered (target), about 23% supervised (smoke ratio, toolcall plan §3) | Structured tool lane | `assistant_only` from `target_mask`; `w_ul = 0` |
| **New: Codex native BFCL train** (copied, §3.3) | 506 rows / 420,725 tokens / 49,213 assistant (X05, m) | Structured tool lane | `assistant_only` from Codex's `target_mask`; `w_ul = 0` |

**Effect on the D2 total.** Forward tokens rise by about 1.9M, from 23.0M short to about 24.9M short. That is about +2.3 h at the D2 rates (EST; X03 gave +1.8 h per extra 1.5M). The registry draw stays at 18.85M, now from the 19,718,144 positions left after the 69-row exclusion (X04, m). The rungs keep their meaning: 5M / 12M / 20M / end of short, then long. The new sources are mixed in by position throughout, so every rung sees tool rows.

### 3.2 Holdouts: what is held out from what

| Set | Held out from D2? | Use |
|---|---|---|
| Claude eval split (1,948 rows, `626093ca`) | Yes, "never_in any training set" | **Gate** |
| bank v1/v2, R-Q, P11, `omni_toolcall_bank_v1` | Yes | **Gate** |
| Codex 168 tool-seed test + 44 operator text tasks | Yes. They have no group in Claude's corpus (X04, m). Self-gen function names share no exact name with them (X05, m). | Supplementary, every stage seal |
| Codex 62 BFCL tasks | **Only after** excluding their 69 source rows from D2 and from D4 replay | Supplementary |
| Codex text val/test inventory (2,506 + 2,549 rows) | **No.** All are inside Claude's sealed D2 text by registry rid (X05: "test rows 2549 rid in D2 2549", m) | Not a held-out measure for any D2+ checkpoint. **It is held out for stage T.** |
| BFCL as a whole | Never a held-out measure for a D2-trained checkpoint, except the 62 above (TOOLCALL plan §2) | — |

### 3.3 Tool calling: Codex's 504 execution-backed episodes plus Claude's parent self-gen

1. **Claude self-gen: the core.** It keeps running on GPU1+GPU2. Then `finalize` → `export` → bf16 `qtext_teacher_cache build-text` rescore.
   - The `finalize` screen gains Codex's 274 tasks (`model_inputs.jsonl`) as an extra 13-gram contamination source, with the receipt saved beside `report_codex.json`.
   - This is applied post hoc, so the running process is not touched (X05 action 4).
2. **Codex native BFCL, 506 train rows: the human-written-schema half.**
   - Copied read-only: select the 506 BFCL train `row_ids` from Codex's `native-text-v3-train/assistant_text.safetensors` (input_ids, lengths, target_mask).
   - Written to `$OC/distill/toolcall/bfcl-native/bfcl-native-506.safetensors`, plus `rows.json` pinning the Codex corpus sha `8d815f32…` and the row ids.
   - Then a fresh Q38 cache.
3. **The 761 JSON-in-prose BFCL docs in the sealed registry** (`manual_cleanup_batch_000166_bfcl_v4_function_calls_ready_append_current`, 293,384 tokens) are handled by masking, not row deletion:
   - **Exclude whole rows:** the 69 packed rows that hold the 62 docs behind Codex's BFCL test tasks (282,624 positions, X04, m). This removes even attention exposure, so BFCL-62 stays a valid secondary held-out.
   - **All-terms mask (KD, CE, EOS, UL = 0)** on the token spans of the remaining 699 BFCL docs. Each span is located by `doc_index`. The rows stay, because other docs share them (dropping rows would cost 69-94% of rows, X04).
4. **Codex's 504 episodes, used by split:**
   - **test 168** is a held-out probe at every stage seal (supplementary);
   - **val 168** is the stage-T dev set;
   - **train 168** (cache `3ed3f303…`, sealed) is a format fixture and, at most, a small stage-T format slice. It is not D2 KD data.
5. **Ground-truth-term mask (X04 b)** on the 2,412 CURRENT_TARGET-rewrite docs and the 5,791 flattened-tool docs (union 3,263,593 tokens, X04, m): `w_ce`, EOS and UL are set to 0 there; Q38/K3 KL stays. Where an all-terms mask and this mask overlap, the all-terms mask wins.
6. **Stage T** (ground truth): it trains on the 19,669 clean rows plus structured tool rows (self-gen conversations, BFCL-native train, optionally the 168 seed-train rows). It never trains on the 5,055 val/test rows, applies Codex's quarantine in full, and uses the val 168 for early stopping.

### 3.4 Caches: valid, recapture, new

**Valid, no recapture (X03/X04/X05/X02, m):**
- all 20 qtext text shards;
- qlong;
- self-gen rescore;
- the K3 caches once sealed;
- QIMG TE/VAE/velocity, ACE DiT, ACE LM codes, HY3D (subject to card C-V1), QVL32 ViT + ref;
- Codex's 168-row tool cache (fixture).

None covers rows ≥ 248,320, and the masks and exclusions are applied at load time, not in the caches.

**Recapture:** none.

**New captures**

| Card | Content | Card / host | Hours |
|---|---|---|---|
| C1 | Tool-call self-gen bf16 rescore, `qtext_teacher_cache build-text` | GPU0 after the D1 seal, or GPU6 between K3 passes | about 0.6 h on a 3090 (EST, 1.5M at 696.19 tok/s) |
| C2 | BFCL-native 506 rows, same tool | Same slot as C1 | about 0.2 h (EST, 420,725 at 696.19 tok/s) |
| C3 | H3 velocity + block-25 pairs (2,048) and j6 vision/video answers, **moved off GPU6** (WAVE3 §6 option 2) | GPU0/GPU4 in the D1-seal → D2 gap | H3 about 3.3 h (design); j6 UNVERIFIED |
| C-s0..s6 | Speech data (§5) | GPU3 now; GPU1/2 after self-gen ends | ≤ 12 h TTS + about 6 h (EST) |
| C-V1 | HY3D `mode` vs `sample` and the time-convention check | CPU | 0.5 h |

---

## 4. Trainer and objective per stage

### D1: unchanged from prereg + A1 (running)
- Objective, sites, optimiser and data as in A1. The hardened run resumes from canonical L00-L10.
- Rungs: one per layer, L11..L63, about 23-25 min each (WAVE3 §4, EST).
- New diagnostics at the D1 seal: Codex's `folding_retention.py` free-run agreement table beside bank v1 and R-Q.
- CPU tests anywhere under `distill/tools/` must hide CUDA by mechanism. Card X-T1: `conftest.py` sets `CUDA_VISIBLE_DEVICES=''` unless `OC_TEST_ALLOW_GPU=1`, and asserts `torch.cuda.is_available()` is False. Background: the 16:26:08Z pytest that put contexts on all GPUs was Claude's (X01, agent-memory event, m).

### D2: `streamed_kd_trainer` 1.3.4 (new) + amendment A2
**Trainer 1.3.4 (card D2-T1, CPU only, about 0.5-1 day EST)**
- Ported from Codex's `streamed_supervision_patch`:
  - per-source `supervision: assistant_only`, using the corpus `target_mask`;
  - raw qtext `verify_shard` on every shard at load;
  - row-plan sha in the data identity;
  - sampler-cursor rewind on a non-finite step (today the step counter advances and rows are lost, X03).
- Per-source `w_ul` override (0 for `tool_native`).
- The UL candidate scan resets at `doc_index` boundaries (today it crosses packed docs, `streamed_kd_trainer.py:1279-1304`).
- Per-source `exclude_rows` and two position masks, both in the data identity:
  - `gt_mask` turns off CE, EOS and UL;
  - `all_mask` turns off every term.
- Receipts record the supervised-position count per step and per term. The denominator stays the frozen global one: micro × (L−1) × accum × world (`:3187`).
- Tests:
  - the existing suite;
  - an all-ones `assistant_only` source is bitwise equal to the legacy path;
  - an all-zero `all_mask` gives zero gradient;
  - `exclude_rows` is honoured across resume;
  - CPU grad-parity arm;
  - the BF16 mass > 1 tail clamp (D2-T5).
- **Two replicas are kept.**

**Losses per source**

| Source | Q38 KL | K3 KL | CE 0.1 | EOS 0.1 | UL 0.1 | JSD |
|---|---|---|---|---|---|---|
| Registry text (not masked) | 1.0, all positions | — | yes | yes | yes | — |
| Registry, rewrite / flattened-tool docs | 1.0 | — | **0** | **0** | **0** | — |
| Registry, BFCL prose docs (699) | **0** | — | **0** | **0** | **0** | — |
| Registry, 69 BFCL-62 rows | excluded | | | | | |
| K3 rows | λ·a | λ·(1−a) aligned | yes | yes | yes | — |
| Self-gen 1.0M | 1.0 on stored positions | — | yes | yes | yes | — |
| `tool_native` self-gen + BFCL-native | 1.0, assistant positions only | — | assistant only | assistant only | **0** | — |
| On-policy | — | — | — | — | — | β 0.5 |
| Long | 1.0 | — | yes | yes | yes | — |

Tool results, prompts and schemas get **no** loss in the tool lane. This is mandatory: the model must not learn to generate tool outputs.

**Optional A/B pilot (card D2-P1).** Run it in the D1-seal → D2 idle window on GPU0+GPU4: two 1M-token text-only pilots, full-sequence vs assistant-masked, on the same row plan. If B clearly wins on bank v1 content, R-Q top-1 and held assistant-position KL, write A3 before the 5M rung. Otherwise A2 stands. Cost: about 2 × 1.2 h plus evals (EST).

### D3: `d3_hybrid_module` 1.0.2 → 1.1.0 + amendment A3 (before D3)
- **Order:** ACE → QIMG → HY3D → H3 (D3b) → D3c music → D3d vision/video → **D3e-in → D3e-out (speech)** → [optional: image edit, temporal video].
- **1.1.0 (card D3-T2):**
  - lanes `speech_in_ce` and `speech_out_flow`, with loss parity against Codex's `speech_flow.py`/`audio_cache.py` on 3 replay tensors (fp32, rel ≤ 1e-6);
  - a declared D3a host-write target and objective (Codex's D3a finding);
  - a test that rows 64,000..65,535 of the C18 band stay bit-frozen;
  - the HY3D posterior/time-convention fix if C-V1 finds one;
  - 20% H3 AV replay in D3e-out.
- **One position chart for H3 type-3 rows** across D3b and D3e. Either Codex's measured 40 Hz chart `(text_count + i, 0, c)` for audio-only rows (with the D3b AV caches packed to the matching chart) or zeros in both lanes. This is decided in A3 with a test that the rotary angles are equal.
- Gates and the stop-spending line are as in the prereg. For speech they are as in the speech proposal §4.3.

### D4: prereg design + Codex guards (card D4-T1)
- Replay 3M Q38 text (2.5M short + 0.5M long), 1M K3, and per-modality pairs. Any modality with fewer than 300 distinct items uses each once, with no duplication. The 69 BFCL-62 rows stay excluded.
- Speech replay: 300 speech-in + 300 speech-out pairs (speech proposal §4.2).
- LR 0.3 × each producer's LR; AdamW (0.9, 0.95, 1e-8, 0).
- Then the fp32 merge, with the K3 gain folded once.
- Guards from Codex:
  - distinct optimizer owners for D2 r64, K3 r32 + gain and the shared D3 r16;
  - refuse a base that already contains D2;
  - audio CE before speech flow;
  - receipts read nested `cache_sha256` identities.

### Amendment A2 (text to add under `## Amendments` before D2's first step, then `check-prereg` must print body sha `c24d21e0…1370`, length 15,339)

```
### A2, 2026-10-1x: D2 supervision, tool lane, evaluation hygiene, K3 compute dtype rule

Status: written before D2's first weight update. Inputs: Codex recovery amendment B (successor
prereg sha 7d2b8601..., lines 131-163), Codex D1/recovery audits, Claude lanes X03-X06 (2026-10-14).

1. Main D2 lane unchanged. Registry text, K3, self-gen (fe1f0784), on-policy and long keep the
   section 3 D2 losses on every stored position. Denominator unchanged (global positions per step).
2. Structured tool lane (new D2 sources, supervision=assistant_only):
   (a) tool-call self-gen, rescore/toolcall-q38-2026-10-14 (sealed cache sha <C1>);
   (b) Codex native BFCL train, 506 rows copied to toolcall/bfcl-native/bfcl-native-506.safetensors
       (source corpus 8d815f32..., row ids pinned; sealed cache sha <C2>).
   Loss only on assistant target positions (target_mask): Q38 KL 1.0, CE 0.1, EOS 0.1; unlikelihood
   weight 0 on these sources. Prompt, schema and tool-result positions carry no loss.
3. Registry exclusions and masks (applied at load; the sealed caches are unchanged):
   (a) exclude_rows: the 69 rows holding the 62 source docs of Codex's BFCL held-out tasks
       (rid list sha <X>), also excluded from D4 replay;
   (b) all-terms mask on the token spans of the other 699 BFCL v4 JSON-in-prose docs;
   (c) ground-truth mask (CE, EOS, UL = 0; KL kept) on the 2,412 CURRENT_TARGET-rewrite and
       5,791 flattened-tool docs (union 3,263,593 tokens) per Codex's quarantine labels.
   The off-policy registry draw stays 18.85M tokens; the short total becomes about 24.9M forward
   tokens (exact count printed by the trainer census and recorded). Rungs: 5M / 12M / 20M / end
   of short, then long.
4. Trainer: streamed_kd_trainer >= 1.3.4 (assistant_only sources, per-source w_ul, doc-boundary UL,
   exclude_rows, gt/all masks, raw shard verification, row-plan identity, non-finite rewind).
   Receipts record supervised positions per term per step.
5. Evaluation. Section 4/5 gates are unchanged. Supplementary, recorded every stage seal: Codex's
   sealed 274-task set (combined_oct13_v1); its 62 BFCL tasks count only with 3(a) in force.
   Codex's text val/test inventory is inside D2 and is not a held-out measure for D2+.
6. K3 compute dtype (pre-stated before the B2 verdict). If the B2 fold-identity gate passes and the
   bf16-vs-fp32 parity fails only on target_logprob p99, the four K3 passes run with
   --compute-dtype fp32 under the same frozen tool, geometry and token ranges. If identity fails,
   stop and diagnose. No gate is loosened.
7. Not covered here: D3/D4 changes (speech lanes, H3 type-3 chart, D3a host-write target, optional
   image-edit / temporal-video lanes, D4 guards) need A3, written before D3.
```

Item 6 is in A2 only if `--compute-dtype` is not already a "tool patch version" change under prereg §6. Recording the rule before the result costs nothing.

---

## 5. Speech: where it enters, teachers, schedule

**Where it enters**
- **D3e**, option B of `distill/SPEECH_LANE_PROPOSAL_2026-10-14.md`. It trains the existing H3 audio channel (C15 type 3, 40 Hz, 32 kHz stereo VAE). There is no structural change, so the running D1 lineage is not forked.
- Order: D3e-in (answer CE + optional ASR KD) → D3e-out (rectified-flow x0 MSE, 4 σ draws) → 20% H3 AV replay.
- **Stage T** adds conversation-style speech afterwards.
- A structural speech unit (option D) happens only if D3e-out is still noise at 25%, or speech-in WER stays above about 30% (decision S-R3).

**Teachers and judges**

| Role | Model | Source / state |
|---|---|---|
| Data voice | Qwen3-TTS-12Hz-**1.7B**-CustomVoice rev `0c0e3051`, 9 voices | Codex-downloaded (`codex-speech-1p7-comparison-modality-v1/parents`). CPU canary: 3 of 4 voices, then the 600 s CPU budget ran out (`complete=false`, m). The P40 rate is unmeasured. |
| Admission | Qwen3-ASR-**0.6B** rev `5eb14417` | Codex teacher pins match the server files (m) |
| Held-set judge | Qwen3-ASR-**1.7B** | Both 0.6B and 1.7B were transcript-exact on the canary sentence (n = 1, m). Pin one repository: on-disk `7278e1e7` vs `-hf bcd2b5b7`. |
| Optional third judge | Fluid Speech Mini (CC BY 4.0 per Dot, memory 1003054) | CPU evaluation only, UNVERIFIED |
| Reference only, never a teacher | Fish Audio Drama3 | Its licence forbids training on its outputs |

**Seed data from Codex**
- 192/16 TTS-0.6B rows with 186/15 exact (cx).
- Replay index `3c99cfb7…` and the 31 held rows.
- The 884 leak-repaired LibriSpeech ASR rows, if Cyril grants S-R2.
- Interface rule (Codex): foreign speech codec ids never enter Chimera's vocabulary or the H3 band.

**Schedule (all EST)**

| Step | Owner | When | Hours |
|---|---|---|---|
| E1 Pascal speech venv (on Lexar); E2 `speech_corpus_build`; E3 `speech_teacher_cache`; E5 A3 speech text | Codex (data, CPU) with Claude review, or Claude if Codex has no usage | now → 10-16 | about 9-10 h of CPU engineering |
| E4 `d3_hybrid_module` 1.1.0 speech lanes + parity | Claude | before D3 | about 5 h |
| C-s0 P40 rate probe (16 utterances) | lane owner via GPU3 lock | after E1 | about 1 h |
| C-s1..s4 texts, TTS (≤ 12 h wall cap, N ≈ 3,243 at the 3090 RTF, probably fewer on P40), admission, H3 latents | GPU3, then GPU1-3 after self-gen ends (~03:00Z 10-15) | 10-15 → 10-16 | ≤ 12 h + about 6 h |
| D3e training | Claude (D3 lane) | inside D3, after D3d | 21.0 h upside / 48.3 h planning (synthetic, N = 4,000); 25.6 / 58.8 h with both LibriSpeech routes; scales by 5.2 / 12.1 h per 1,000 utterances |
| D4 speech replay | Claude | D4 | about 0.5 h |

Two items wait on Cyril: **S-R2** (admit natural LibriSpeech despite the 2025-2026 information rule) and S-R3 (later). S-R1 (go for D3e) is covered by the standing GO and is recorded in A3.

---

## 6. The K3 path

1. **B2 failure of 2026-10-13, explained.** p99 1.2499981 > 1.0; median 8.09e-6; top-1 0.97353; non-finite 0 (`parity_b2_receipt.json`, X06, m). Codex's bisect put the first unequal layer at L2. Claude's paired L2 replay settled the cause:
   - router logits are bit-identical across M (`b2_geometry_diag.json` `82532ff0…`: 0 rows changed, max diff 0.0);
   - 5 of 2,293,760 expert rows change, each where a group gives an expert M = 1, at one bf16 ulp (`b2_geometry_mechanism.json` `c4c3ed81…`);
   - at L2, group vs single relL2 is 2.399e-6, against 0.029358 for bf16 vs fp32;
   - verdict `bf16_batch_numerics`, ratio 8.17e-5 (`b2_geometry_verdict.json` `e1ce01c3…`).

   Codex's "candidate cause" is therefore closed: there is no algorithm or call-shape change. The p99 tail is a one-ulp seed amplified through about 90 bf16 layers; that amplification is inferred, not measured layer by layer.
2. **Running now.** Queue 1.3.0. The bf16 review pass was at L69/93 at 18:51:20Z, about 2.3 min per layer, so it ends about 19:45-20:00Z (EST). Then the fold-identity gate (about 10-13 h, gated on MemAvailable ≥ 70 GiB), the fp32 reference pass (about 3.5-4 h), then export and parity. **B2 verdict about 2026-10-15 10:00-13:00Z (EST).**
3. **Decision card ready (A2 item 6).** Identity PASS with a p99-only fail → fp32 K3 passes. The fp32 pass's printed per-layer times set the new duration, which is UNVERIFIED and probably longer than bf16. Identity FAIL → structural; stop.
4. **Duration.** 4 passes × 45-50 h (gpu6 lane EST) puts the K3 seal at **2026-10-22 22:00Z to 10-23 21:00Z**. This is the critical path, about 7 days behind D1.
5. **Acceleration workflow.** `distill/K3_ACCELERATION_PLAN_2026-10-14.md` **does not exist yet** (checked at 19:00Z). The other workflow's profiler `distill/tools/k3_accel_profile.py` 1.0.0 (frozen-tool copy with stage timers; identity proven by `compare` on prefix digests) and its driver `run_k3accel_profile_gpu3_2026-10-14.sh` are deployed. Outputs are in `/mnt/lexar/omni-campaign/scratch/k3accel/{bench,logs,pyspy,runs}` (last write 12:42 local = 18:42Z), and GPU3 is now idle. When the plan lands, its measured per-pass time replaces item 4.
6. **More cards after the D1 seal** (WAVE3 §6 option 1): the passes are independent token ranges. GPU0/GPU4 can run K3 passes from about 10-15 22:00Z, after a B2-equivalent identity check on the 3090 class. Card K3-M1; its cost is the per-card identity check (about 10-13 h on GPU6 basis, EST). Decide after the acceleration plan, because the two compete for the same cards as C3 (H3/j6).

---

## 7. Working split with Codex

**How Cyril works.** He alternates: when one agent runs out of usage he works with the other. Either agent may therefore be absent for hours. Lanes must survive that, and **no lane may depend on an agent being awake to stay safe**. Queues run unattended on the server and hold their own locks. Agents enqueue work, read status and review; they do not hold cards themselves.

### 7.1 Lane ownership

| Lane | Owner | Scope | Server resources |
|---|---|---|---|
| L-D1 | Claude | Hardened D1 to seal and register | GPU0+GPU4 (`gpu0.lock`, `gpu4.lock`), up to 128 GiB RAM cgroup |
| L-K3 | Claude | `queue_gpu6.sh` 1.3.0: B2, K3 passes, probe, H3, j6 | GPU6 (`gpu6` queue), K3 RAM gate |
| L-TOOL | Claude | Tool self-gen, finalize, rescore (C1/C2), BFCL-native copy, `toolcall_bank` | GPU1+GPU2 (`p40.claims`) until done |
| L-TRAIN | Claude | Trainer 1.3.4, D2/D3/D4 drivers, prereg A2/A3, ledger registration | GPU0/4/6 only via queues after the D1 seal |
| L-K3ACC | Claude (the other workflow) | K3 acceleration plan | GPU3 only when claimed in `p40.claims` |
| L-DATA | **Codex** | Corpus repair, splits/quarantine, BFCL and tool-seed maintenance, rid lists (BFCL-62 rid list for A2 3a) | CPU only, nice 15, `CUDA_VISIBLE_DEVICES=` |
| L-SPEECH-DATA | **Codex** | E1-E3, E5 and the C-s data steps (proposal §4.1) | P40 only through a `p40.claims` line and the GPU3 lock; MemAvailable gate ≥ 40 GiB |
| L-MEDIA-DATA | **Codex** | H3/HY3D/image-edit/temporal-video adapters as specs and tests for `d3_hybrid_module` | CPU |
| L-FOLD | **Codex** | Folding audit, `folding_retention.py`, folding backlog; may cite Claude's K3 L2 receipts | CPU; GPU only by request to the card's owner |
| L-NATIVE | **Codex** (parked) | Native control rows, v29 entrypoint; only if Cyril wants the native torch protocol as a deliverable | none until after T |

When one agent is out of usage, the other may **review, enqueue and read** in any lane. It may **take over** a lane only by the takeover rule in §7.4.

### 7.2 Single sources of truth (files are the record; Relay is the notification)

| What | File | Who writes |
|---|---|---|
| Data, losses, rungs, gates | `distill/PREREGISTRATION_OMNI_STAGE_D_v1.md` (frozen body + dated amendments; local copy and `$OC/distill/` mirror) | Either agent, as an amendment only, then `check-prereg` |
| Rulings from Cyril | `distill/STAGE_D_RULINGS_2026-10-12.md` | Either agent, quoting Cyril verbatim |
| Current state | The newest `STATE_<date>_<tag>.md` in `reports/omni-campaign-2026-09/` and `$OC/distill/` | Whoever is working writes the **next** file; older ones are never edited |
| Plan and split | This file | Updated as `…_RECONCILED_PLAN_<date>.md` revisions |
| Lane registry (new, card X-T2) | `$OC/distill/queues/LANES.json`: lane, owner, lock path, status, last receipt path+sha, `transferable` flag, heartbeat UTC | Lane owner only; a takeover writes a takeover receipt |
| Deployed tool pins (new, card X-T3) | `$OC/distill/tools/TOOL_PINS.json`: name, version, sha, superseded path | Whoever deploys; Codex's tests read pins from here instead of hard-coding (today the 1.3.3 trainer and qtext 1.0.6 fail 14 Codex tests on pin drift, m) |
| Codex evidence | `reviews/2026-10-13-folding/RECOVERY_STATUS_*.md` and `proof_outputs/` | Codex; Claude reads and copies |

### 7.3 Hand-offs on Relay (exact commands)

```powershell
$S = "C:/Users/cereal/.agents/skills/coordinate-peer-agents/scripts/session.py"
# Claude, at the start of an attended session (consumes peer-mail cursors; only the attended parent session runs it):
python $S --agent claude sync
# Claude answering a Codex message (thread 64abe991 is the Codex->Claude recovery thread):
python $S --agent claude reply --thread-id 64abe991-3f45-4576-8617-94566f1cf21e --reply-to <message-id> --body-file C:/path/msg.md
# Claude starting a new topic (Codex's next attended prompt receives it as peer mail):
python $S --agent claude say --body-file C:/path/msg.md
# Full text of a truncated message:
python $S --agent claude show --thread-id <T> --message-id <M>
# Delivery check that needs no model usage on the far side:
python $S --agent claude say --receipt-only --body "Routing check"
# Codex uses the same commands with --agent codex.
```

**Every hand-off message carries five lines:**
1. lane;
2. state, with the newest receipt path and sha;
3. what is running (pid or unit, card, lock held);
4. the next command, exactly;
5. what must not be touched.

**Rules**
- A dispatcher receipt or "model adapter unavailable" means the peer is out of usage. It is not a refusal, and it is not a broken Relay (Codex 49aed260 position, adopted).
- Keep sending updates while the peer is out; it reads them on return.
- Channel posts are awareness only.
- Sub-agents and workflow lanes never run `sync`; it would consume the parent's mail.

### 7.4 GPU and RAM launches on the AI server: only through the owning lane, with locks

1. A GPU process starts only from the lane's queue or launcher, which takes `flock` on `$OC/distill/queues/gpu<N>.lock` (P40s: also a `p40.claims` line) and checks MemAvailable against its own gate. **It waits; it never competes.**
2. Codex's allocator (`d1_allocated_root.py`, `remote_runtime.py`) takes only its own lease flock today. It does not read Claude's locks (X01, `d1_allocated_root.py:72-139`, m). **Ask:** before admitting any job, take `flock(LOCK_SH|LOCK_NB)` on the target `gpu<N>.lock` and refuse if it is held. Codex edits its own code; Claude does not.
3. CPU jobs that use more than 16 GiB of RAM (for example Codex's 80 GiB control-row retry, which needs MemAvailable ≥ 193,273,528,320 B) run only in a **posted window**. Claude posts one when D1 is registered (about 10-15 20:00-22:00Z) and outside a K3 pass's RAM gate.
4. Every CPU test runs with `CUDA_VISIBLE_DEVICES=` (conftest, card X-T1).
5. **Takeover of a lane** (needed because Cyril alternates). Allowed only when (a) Cyril says so, or (b) the lane's `LANES.json` entry is `transferable` and its heartbeat is older than 6 h. Before acting, the agent:
   1. runs `sync`;
   2. reads the newest STATE and `LANES.json`;
   3. checks the live process (`systemctl`, `docker ps`, `nvidia-smi`, `pgrep`) and **never relaunches anything that may still be alive** (the D1 r5/r6 lesson);
   4. writes a takeover receipt to `LANES.json`;
   5. posts the five-line hand-off on Relay.
6. **Emergency stop** (host MemAvailable below 8 GiB, or thermal abort): either agent may stop the offending process. It writes a stop receipt and posts it on Relay at once.

### 7.5 Relay messages still owed to Codex (the parent session sends them; not sent by this document)

| To / thread | Content |
|---|---|
| reply to 6f551d6b and a3a8e5d1 on `64abe991` | The K3 L2 cause is closed (receipts `c4c3ed81`, `82532ff0`, `e1ce01c3`); no paired replay needed |
| reply on `64abe991` (or `a64c07cf`) | D1 division accepted as in Codex's draft V2. The pytest leak was Claude's, and the conftest fix is coming. The lock request (§7.4.2). The posted-window rule for the 80 GiB retry. Ask for the V2 delivery receipt. |
| reply to 633e0927 / b0600726 on `64abe991` | This plan's §1-§3, with the asks: the BFCL-62 rid list; whether D2 v2 is re-based on 1.3.4 or left to Claude's port; `TOOL_PINS.json` adoption; cleanup of superseded remote dirs (qimg v2, ace v4, geometry v1-v4) **after Cyril's go-ahead** (`/` at 97%) |
| reply to 435d8838 | Answers its open questions: corpus owner (L-DATA Codex / L-TRAIN Claude), manifests (§3), tool schema source (self-gen `schemas-2026-10-14` + BFCL-native), cache status (§3.4), running jobs (header), media owners (§7.1) |

---

## 8. Updated critical path

All ETAs are EST from printed per-layer times, except where noted.

| # | Event | ETA (UTC) | Gate / note |
|---|---|---|---|
| 1 | K3 B2 bf16 review pass ends | 2026-10-14 19:45-20:00 | L69 at 18:51:20Z, about 2.3 min/layer (m) |
| 2 | D1 prefix replay ends; training from L11 | 10-14 about 20:45 | L03 at 18:52:40Z; 6 KDA × about 1,100 s + 1 GQA × about 180 s remaining |
| 3 | Tool self-gen reaches 1.5M | 10-15 about 03:00 | eta_hours 8.15 at 18:51:46Z (m); then finalize/export about 0.5 h |
| 4 | B2 verdict (identity + fp32 parity) | 10-15 10:00-13:00 | Identity gate waits for MemAvailable ≥ 70 GiB |
| 5 | Speech data on P40s (if E1-E3 land) | 10-15 → 10-16 | ≤ 12 h TTS cap + about 6 h |
| 6 | D1 last rung | 10-15 17:00-19:00 | 53 layers × 23-25 min |
| 7 | **D1 registered** (`OMNI1D1`, clone-pre-D2) | **10-15 20:00-22:00** | seal about 3.1 h |
| 8 | P2 rate cell (GPU0); C1+C2 rescore (about 0.8 h); C3 H3 (about 3.3 h) + j6 on GPU0/4 | 10-15 22:00 → 10-16 about 06:00 | Moved off GPU6 (WAVE3 §6 option 2) |
| 9 | Trainer 1.3.4, A2 written, `check-prereg` OK; optional D2 A/B pilot | by 10-17 | CPU; pilot about 2.4 h on GPU0+4 |
| 10 | **K3 sealed** (bf16, 4 × 45-50 h) | **10-22 22:00 → 10-23 21:00** | Critical path. Changes if fp32 K3 is chosen at #4 (longer, UNVERIFIED) or if the acceleration plan / extra cards (K3-M1) land (shorter) |
| 11 | D2 (about 24.9M short + long + seal) | Upside: 10-22 22:00 → 10-24 about 10:30 (36.5 h). Planning: 10-23 21:00 → 10-26 about 17:00 (68.1 h) | Design 34.5 / 66.1 h + about 2 h for the tool lane |
| 12 | D3 incl. D3e speech | Upside → 10-26 06:40-14:00 (44.1-51.6 h). Planning → 10-30 08:00 to 10-31 01:20 (87.1-104.3 h) | D3 23.1-30.6 / 38.8-56.0 h + D3e 21.0 / 48.3 h (synthetic N = 4,000) |
| 13 | D4 merged `OMNI1D4` | Upside → 10-27 06:20-13:40 (23.7 h). Planning → 10-31 15:00 to 11-01 08:10 (30.8 h) | Needs ≥ 60 G free on `/` (70 G now) |
| 14 | Stage T, then Q4 export | T 87-201 h (design) after D4; Cyril decides the T start | Export keeps `--ledger-band-logit-mask` (risk R6) |

**Biggest levers on the date**
1. K3 duration: acceleration plan, extra cards, and the fp32 choice.
2. Whether D2 runs at the upside rate (P2 verdict on 10-15).
3. D3e speech size N.

**Card list for the next 48 h**

| Card | Owner | Hours (EST) | Needs |
|---|---|---|---|
| X-T1 conftest CUDA hiding | Claude | 0.25 | — |
| X-T2 `LANES.json` + takeover receipt format | Claude, Codex reviews | 1 | — |
| X-T3 `TOOL_PINS.json` | Claude; Codex switches its tests | 0.5 + Codex time | — |
| D2-T1 trainer 1.3.4 (+ D2-T5 tail-clamp test) | Claude | 4-8 | Codex's patch copied read-only |
| D2-M1 `exclude_rows` / `gt_mask` / `all_mask` files from `documents.jsonl` × `normalized.jsonl` (X04 scratch scripts) | Claude | 0.5 | BFCL-62 rid list from Codex for the cross-check |
| D2-M2 BFCL-native 506 copy + `rows.json` | Claude | 0.5 | — |
| A2 into the prereg + `check-prereg` | Claude | 0.5 | Cache shas from C1/C2 |
| C1/C2 rescores | L-TOOL | 0.8 | D1 seal or a GPU6 gap |
| C-V1 HY3D mode/time check | Claude | 0.5 | — |
| E1-E3 speech data tooling | Codex | 8 | GPU3 claim for C-s0 |
| Relay replies (§7.5) | Claude parent session | 0.5 | — |

---

## What I could not verify
- Every (cx) number above, in particular the qtext capacity GiB figures, the 186/192 speech split and the clone digest `8edf50c4…`.
- That `d1_trunk_recovery` 1.3.1 is numerically identical to Codex's 1.2.8/1.2.9 (WAVE3 asserts it).
- The ETAs in §8, which are extrapolated. K3 production pass time, the fp32 pass time, P40 speech rates and j6 time are all unmeasured.
- Whether the D2 loader can exclude rows from a sealed corpus today. It cannot without 1.3.4: `row_ids = sorted(cache.index)` (X04, m).
- The K3 acceleration plan (not written yet).
- Whether Codex's V2 D1 acknowledgement has been delivered on Relay. It existed as a local draft at 18:31Z (X01).

## Evidence index
- **Lanes X01-X06:** server reads at about 18:28-18:31Z, re-runs of Codex tools in scratchpad `x04/`, `/tmp/claude_x02` and `/tmp/claude_x03_asst_frac.py`.
- **This document's server reads:** `date -u` printed `Wed Oct 14 06:53:17 PM UTC 2026`. Also D1 `runtime-log.txt` tail, `systemctl show claude-d1-hardened.service`, `$Q/logs/gpu6/b2_pass.log`, `$Q/gpu6.status`, toolcall `progress.json`, `nvidia-smi --query-compute-apps`, `/proc/meminfo`, `df -h`, and a `find` for k3accel paths.
- **Local reads:**
  - `STATE_2026-10-14_WAVE3.md`
  - `distill/PREREGISTRATION_OMNI_STAGE_D_v1.md` (sections 2, 3, A1)
  - `distill/SPEECH_LANE_PROPOSAL_2026-10-14.md`
  - `distill/TOOLCALL_LANE_PLAN_2026-10-14.md` §3
  - `distill/DISTILL_DESIGN_2026-10-12.md:170-172`
  - `distill/tools/streamed_kd_trainer.py:1128-1156`
  - `distill/tools/d3_hybrid_module.py:47-52,111,122`
  - `C:/Users/cereal/.agents/skills/agent-comms/SKILL.md`
- **Codex message digest:** `scratchpad/relay/digest/bodies_all.txt` (20 bodies, 91,980 bytes).
