# Tool-calling lane: review, census, distillation set and held-out bank (2026-10-14)

**Owner:** Claude (lane `toolcall`). **Why:** Cyril, 2026-10-13 14:02Z to Codex: "[Cyril quote withheld]". Standing GO 2026-10-12 ("[Cyril quote withheld]").

**Times** are UTC. Server logs print UTC-6. Every number below was printed by a command on the server; the receipt or log is named beside it.

---

## 1. Codex's 168-row tool-training cache: review

**Where it is**
- Cache: `$OC/distill/codex-preparation-01a1326f-20261013/qtext-portable-v8/out/cache/q38-recovery-01a1326f-oct13/3ed3f303f8b8c6dcbe4db21bb7a0cb8a4de764f71a3055763eab311eead2b133/`. Sealed `2026-10-14T00:35:00Z`.
- Corpus: `.../native-text-v7-train/assistant_text.safetensors`, sha `3ed3f303…`. Manifest sha `31260fd3…`, schema `chimera_recovery.assistant_supervision.v1`.

**Format**
- The cache is the standard qtext layout: top-64 bf16 logprobs, int32 ids and lse, one shard.
- Size: 168 rows × 817 positions; 113,026 forward tokens; 26,634 stored (assistant) positions.
- Tensors: `input_ids`, `lengths`, `assistant_mask`, `target_mask`.

**Teacher**
- Parent Qwen3.8-27B-U bf16, fp32 head, run through Codex's portable `tps_probe.py` (sha `449eecc4…`) on one 3090.
- 187.13 forward tok/s.
- Parent digest `7e0fb8f0…` equal before and after.
- `all_finite` and `all_structural_ok` true.
- top-1 logprob mean −0.0387; top-64 mass mean 0.9991.

**Content**
- Every row is native format: `<tools>` system block, `<tool_call><function=…>`, `<tool_response>`.
- The data is narrow and scripted:
  - 2 tools only (`local_text` 84 rows, `local_arithmetic` 84 rows).
  - 2 distinct system prompts.
  - 6 scenarios × 28: `schema_upper`, `call_add`, `result_words`, `retry_division`, `parallel_text`, `multiturn_arithmetic`.
  - User turns are templated: "Episode <hex>; scenario X. …".
  - Assistant turns are scripted, not sampled from the parent. Final answers read "Recorded value: [13106]". In `schema_upper` the assistant echoes the JSON schema before the call.
  - Calls per row: 84 rows with 1, 84 rows with 2.
- Correct where I checked: rows 0, 1 and 5 decoded. Arithmetic matches; the calls are schema-valid.

**Provenance and licence**
- Self-made by Codex; no external data.
- Its own manifest says `production_training_ready: false`.

**Contamination**
- Screened with `distill_corpus_build.build_review_index`: 508,323 13-grams; banks v1/v2, haystacks, probe/smoke, R-Q, probe64, needles, parity, extension and paraphrase. Result: 0 / 168 rows hit.
- Against the corpus eval split: 1,948 items, 13-gram. Result: 0 / 168 rows hit.
- Receipt: `$OC/distill/toolcall/analysis/report_codex.json`.

**Verdict**
- Do not use it as D2 training data. It teaches a templated, two-tool dialect (for example "Recorded value: [...]") that is not the parent's behaviour, and it is small: 26,634 KD positions, 0.13% of D2.
- Keep it as a format smoke fixture: it proves that a native-format corpus loads through the qtext cache path.
- The set built below supersedes it.

## 2. Tool-calling content of the D2 text set

Corpus: `$OC/distill/corpus/text`, manifest `c42e508a…`, 20,000,768 tokens, 35,985 docs. Census receipt: `$OC/distill/toolcall/analysis/report_d2.json`. Every document was decoded with the parent tokenizer.

**Native format: none**
- `<tool_call>` (248058), `</tool_call>` (248059), `<tool_response>` (248066) and `</tool_response>` (248067) each occur **0** times in the 20M ids.
- No document carries a tools system block. Each document is rendered `[user: instruction, assistant: response]`.

**Route `tool`:** 2,789,308 tokens, 7,966 docs (14.0% of tokens).

**Docs with JSON-in-prose tool structure:** 3,929 docs, 2,203,285 tokens.
- `{"tool_calls": …}`: 3,145 docs, 1,898,135 tokens.
- `"name"…"arguments"`: 784 docs, 305,150 tokens.
- By route: tool 1,712,245 tokens; code 268,119; web_knowledge 201,244; text 35,443.
- These are mostly omnicoder pipeline artefacts:
  - `v9_base_validated_28025`: `apply_patch` calls against `G:/omnicoder_0.1.0/...` paths.
  - `erdos_broad_domain_code_tool_adjacent_*`: "audit" JSON with `tool_plan`, `checks` and `dedupe_scope` blocks that cite NAS registry paths.

**BFCL v4 ground truth is inside D2:** `manual_cleanup_batch_000166_bfcl_v4_function_calls_ready_append_current`.
- D2 text: 761 docs, 293,384 tokens.
- Self-gen prompts: 58.
- Eval split: 25.
- Registry: 1,122 rows.
- **Consequence:** BFCL cannot be used as a held-out tool-calling measure for any checkpoint trained on D2. The held-out bank in §4 uses only functions written for this campaign.

**Registry as a whole** (`staging/rows.jsonl`, 91,984 rows):
- 12,688 rows have JSON tool structure (tool 5,428, code 4,467, text 1,687, web_knowledge 1,106).
- None are native format.

**Conclusion:** D2, as built, would teach the student to emit JSON tool calls in prose, and nothing in it teaches the parent's own `<tool_call><function=…>` turns, tool-response reading, parallel calls, recovery from errors, or declining when no tool fits. The parent does all of these natively. Self-gen from the parent is the only local source.

## 3. The tool-use distillation set

**Tool:** `distill/tools/toolcall_selfgen.py` **1.1.0** (sha `02070a87…`). Tests are in `test_toolcall_selfgen.py` (sha `bc2c939e…`; server: 18 passed, `logs/tests_v110.log`). 1.0.4 copies kept at `$OC/distill/tools/_superseded/toolcall_selfgen_1.0.4_2026-10-14/`. Version history, all within 2026-10-14:

| Version | Change | What ran on it |
|---|---|---|
| 1.0.0 | — | Never ran against the model. Its HTTP 4xx handling retired workers. |
| 1.0.1 | 4xx = per-sample reject | — |
| 1.0.2 | bank-run parallel over both servers | bank baseline; first 27 schema toolsets (under the stricter library-wide name rule) |
| 1.0.3 | Names unique within a toolset only. Library-wide uniqueness had refused most 2nd/3rd toolsets of a domain on name collisions alone (44 drops in 25 toolsets). | remaining 117 toolsets, plan, smoke |
| 1.0.4 | Backend simulator caps raised from 420 tokens / 2,000 chars to 900 / 4,000. Smoke: 4 of 18 attempt rejections were truncated list/catalog backends, which would have under-represented search/list tools. | production run (fresh run dir; the 2-min 1.0.3 start is kept at `run-…-2026-10-14.v103-aborted-backend-caps/`) |
| 1.1.0 | Review fixes. (a) Zero-call content rule `zero-call-content-v1`: a no_tool / missing_tool row must not claim a lookup or action (multilingual patterns) and the parent as greedy judge must find no offered function that serves the request and no claimed lookup/record facts. Production row tc-00030 had been accepted saying "he verificado el estado actual de la reserva en el sistema" with an invented status. (b) Sandbox: OverflowError (huge-int `calculate`, `date_add` past 9999) no longer escapes `execute_sandbox`; any non-`_Reject` exception in an attempt rejects that attempt with its traceback (run stops past 25 per session); a dying worker is logged and stops the others. | production run resumed in the same run dir 2026-10-14 ~17:05Z; old zero-call rows re-checked on resume |

Drivers:
- `run_toolcall_export_2026-10-14.sh`: re-exports the parent's own Q4_K_M.
- `run_toolcall_selfgen_2026-10-14.sh`: serve, probe, bank, schemas, plan, smoke, run, finalize, export, teardown.

**Sampler**
- The parent's Q4_K_M from the campaign exporter, re-run (U7, "the parent's Q4_K_M on the P40s"), into declared scratch `/mnt/lexar/omni-campaign/scratch/toolcall-q38-2026-10-14/`.
- The self-gen Q4 (`76c1e4c6…`) was deleted after its run.
- `/home/cereal/models/gguf/Qwen3.8-27B-Uncensored-Q4_K_M.gguf` (`4c5e2db0…`) is a third-party conversion (`JonathanColetti/Qwen3.8-27B-Uncensored-GGUF@45d0fc0a`, per its `.verified.json`), not tied to the parent digest. It is not used.
- Served on GPU1 and GPU2 by `selfgen_sampler.py serve`, 4 slots × 8,192 ctx each. GPU3 is left free. Claim line is in `queues/p40.claims`.

**Teacher signal**
- Every token is rescored later in bf16 by `qtext_teacher_cache build-text`, as with the existing self-gen. The sampler only chooses which sequences exist.

**Render**
- The parent's own `chat_template.jinja` (sha `c3cf9e34…`) through `apply_chat_template(messages, tools=…, enable_thinking=False)`.
- Calls use Qwen3-Coder XML (`<tool_call>\n<function=name>\n<parameter=p>\nvalue\n</parameter>…`).
- Tool results go inside a `user` turn as `<tool_response>…</tool_response>`.
- Thinking is off, to match the D2 render and the S8 review.
- `target_mask` covers every assistant turn after its empty think block, including `<|im_end|>`.
- Refusals: an assistant span that does not start and end on token boundaries, and any re-render that does not reproduce its journaled sha.

**Who writes what**

| Part | Written by |
|---|---|
| Tool schemas | The parent, `schemas`: 48 training domains × 3 toolsets, 3-5 functions each, validated against a JSON-schema subset. Plus 4 sandbox toolsets of 8 executable tools. |
| User requests and follow-ups | The parent as user-writer. 20 personas × 6 styles; 84% English, 16% across 7 other languages. |
| Tool backends | Sandbox tools (calculate, convert_units, date_add, days_between, currency_convert, text_statistics, split_bill, percent_change) are executed in Python with exact results. All other tools are simulated by the parent as a JSON backend. |
| Assistant turns (the training target) | The parent: T 0.7, top-p 0.8, top-k 20, Qwen's non-thinking setting. |

**Categories** (plan weights): single 0.20, parallel 0.17, sequential 0.14, multiturn 0.10, error 0.10 (not found, rate limited, unavailable, permission denied, invalid argument, injected on the first call), clarify 0.09, no_tool 0.10, missing_tool 0.10.

**Acceptance (rejection sampling, up to 3 attempts per plan item)**
- Every call must:
  - parse;
  - name an offered function;
  - pass schema validation (types, enums, bounds, required, no unknown parameters);
  - have no text after the calls.
- Every turn must end on EOS.
- The behaviour must match the category: a parallel turn has ≥ 2 calls; sequential has ≥ 2 calling turns; no_tool and missing_tool make 0 calls; clarify asks a question before any call and calls after the answer; multiturn calls again for the follow-up.
- When a sandbox tool returned a result, the final answer must state one of its computed values. Both number conventions are accepted; echoed inputs do not count.
- 1.1.0, no_tool and missing_tool only (`CONTENT_RULES = zero-call-content-v1`): making 0 calls is not enough.
  - The answer must not match `CLAIM_PATTERNS` (first-person "I checked / booked / sent …", "according to the system", in all 8 plan languages; a clause about the functions/tools themselves is exempt).
  - The parent as judge (greedy, 64 tokens, `CHECK_SAMPLING`, outside the locked `SAMPLING`) must answer `fits: false` (no offered function serves the request; catches user-writer drift) and `claims: false` (no claimed lookup/action, no user-record facts). No verdict = reject.
  - The verdict is stored per journal row (`content_check`). On resume, accepted zero-call rows without a current verdict are re-checked: a pass is re-journaled with its verdict; a fail is journaled `reopened` and continues from its next attempt with its earlier rejections. The journal stays append-only; the last record of an index wins. `finalize` refuses unresolved reopened rows and zero-call rows without a verdict, and records the rule in `manifest.content_rule`.
  - The run's token target counts only rows that pass, so it generates past the old cut to cover dropped rows (no target change, no new run dir).

**Size**
- 1.5M rendered tokens (`--token-target 1500000`), cut in plan order.
- The supervised (KD) share was 23% in the smoke (§4b); it is printed by `finalize`.

**Contamination screen** (`finalize`), over user, assistant, call-argument and tool-response text and tool descriptions, but not the template's fixed scaffolding:
- The full campaign review surface (`build_review_index`).
- The corpus eval split (1,948 rows).
- Every string of the toolcall bank.
- Duplicate first-user messages are dropped.

**Held out**
- Training never offers a bank function (asserted in `plan`).
- Synthesized tools whose name or description touches a bank family are refused (`heldout_touch`; the regexes are listed in the manifest).

**Outputs** (all small, on root under `$OC/distill/toolcall/`):

| Output | Path |
|---|---|
| Schemas | `schemas-2026-10-14/{schemas.jsonl, manifest.json, journal.jsonl}` |
| Plan | `plan-2026-10-14.jsonl` (+ `.manifest.json`) |
| Smoke | `smoke-q38-q4km-2026-10-14/` (12k tokens, finalized and exported; gate ≥ 20% acceptance) |
| Production run | `run-q38-q4km-2026-10-14/{run_config.json, journal.jsonl, progress.json, conversations.jsonl, samples.jsonl, manifest.json}` |
| Rescoring corpus | `rescore/toolcall-q38-2026-10-14.safetensors` (+ `.rows.json`): `input_ids` int32, `lengths`, `target_mask`; route `tool_native` |

**How it enters D2**
1. **Rescore** (not yet run; GPU0 belongs to D1 r4 until it exits):

   ```
   qtext_teacher_cache.py build-text --corpus rescore/toolcall-q38-2026-10-14.safetensors ...
   ```

   This is the same command the self-gen rescoring used. About 1.5M forward tokens at 696.19 tok/s ≈ 0.6 h on GPU0, or on GPU6 between K3 passes.
2. **Add the sealed cache** to the D2 `--data-config` as a Q38 source, weight 1.0.
3. **Proposal, to be recorded as a prereg amendment by the bookkeeping lane (Step 1b):**
   - D2 adds ≈ 0.35M `tool_native` KD positions (23% of 1.5M rendered tokens at the smoke ratio; the exact count comes from `finalize`). An equal number of registry off-policy tokens is removed from the JSON-in-prose `tool` route, in the `v9_base_validated_28025` and `erdos_*_tool_adjacent` "audit" rows.
   - The 23.0M D2 total is unchanged.
   - The 761 BFCL documents stay; they are already in the sealed corpus. Bank results are never compared against BFCL.

**Stage T:** the same conversations are SFT rows, with the sampled assistant turns as ground truth. The design §7 says "Stage T also owns … the tool/code routes", so this set is that route's native-format core.

## 4. The held-out tool-calling bank

- **File:** `$OC/calibration/banks/omni_toolcall_bank_v1.json`, built by `toolcall_selfgen.py bank-build`.
  - It is deterministic and sealed (`bank_sha256`).
  - It is never overwritten with different bytes.
  - It is a new file. `bank_v1` and `bank_v2_draft` are untouched, so the sealed corpus screen's source hashes stay valid.
- **Functions:**
  - 5 held-out sandbox calculators: loan_payment, geo_distance, bmi_calculate, timezone_convert, compound_interest.
  - 8 held-out API families: get_flight_status, refill_prescription, book_meeting_room, track_parcel, search_recipes, create_support_ticket, get_air_quality, reserve_campsite.
  - None is offered in training.
- **185 cases, 8 lanes:**

  | Lane | Cases | What it tests |
  |---|---|---|
  | exec_single | 40 | |
  | exec_parallel | 20 | Two calls required in one turn |
  | api_select | 40 | |
  | multi_turn | 15 | Follow-up after a completed call |
  | error_recovery | 15 | INVALID_ARGUMENT with allowed values; a corrected call is expected |
  | no_tool | 20 | |
  | missing_tool | 20 | |
  | clarify | 15 | A required value is missing; a question is expected and no call |

- **Ground truth:** by construction (templated values).
- **Grading** (`bank-grade`):
  - Only the first assistant turn is graded.
  - Every expected call is matched once by name and argument values: numbers within 1e-6 relative or 0.011 absolute; strings case- and space-insensitive; arrays as sets.
  - Free-text required arguments (`summary`) must be present.
  - No extra calls and no schema problems.
  - no_call lanes need no `<tool_call>` and a non-empty answer.
  - clarify also needs a question mark.
- **Engine-agnostic:** `bank-grade` reads a `{id, response}` JSONL from any engine.
  - `bank-run` produces it from a llama-server, greedy, 768 tokens. That covers the parent Q4 baseline now and every Q4 export later.
  - For the bf16 Chimera checkpoints (D1-D4, T), the review runner must render each case with the same template call (`messages`, `tools`, thinking off) and write the same JSONL. That hook is not built.
- **Built:** 2026-10-14 15:13:08Z.
  - Seal `bank_sha256` 7d7a039c23f7c903…; file sha256 `de0e00037870698714f79b397a8e7dbb8dfee9743ee0c9ee915ade4a245b2a8d`.
  - 185 cases: exec_single 40, exec_parallel 20, api_select 40, multi_turn 15, error_recovery 15, no_tool 20, missing_tool 20, clarify 15.
- **Parent baseline** (the parent's Q4_K_M, sha `76c1e4c6…`, llama-server greedy, 2 P40s; 15:13:09-15:23:29Z):
  - **174/185**: exec_single 40/40, exec_parallel 20/20, api_select 40/40, multi_turn 15/15, error_recovery 15/15, no_tool 20/20, missing_tool 16/20, clarify 8/15.
  - Receipt: `$OC/distill/toolcall/bank-runs/parent-q4km.grade.json` (receipt sha `da4ee8b4…`). Responses: `parent-q4km.responses.jsonl` (`04d3f9b4…`).
  - I read all 11 failures. 10 are real model behaviour:
    - It called functions that were not offered: `play_music`, `get_bank_balance`, `web_search`.
    - It used `search_recipes` for a car-rental request.
    - It invented missing values instead of asking (a 2024 date, a principal of 100000, NYC/LA coordinates, a placeholder e-mail).
  - 1 is a rule artifact: `tcb-clarify-006` asked for both missing details as a numbered list without a "?".
  - The "?" rule stays as sealed, so every stage is graded by the same rule.
  - The parent is near ceiling on the six call lanes. The lanes that discriminate are missing_tool and clarify, plus every lane for a damaged student.
- **Proposed D-stage gate** (proposal, for the amendment):
  - Report the toolcall bank lanes at every stage seal beside the parent-Q4 baseline and OMNI1.
  - From D2 onward: total passes ≥ the previous stage, and no lane below OMNI1. These are diagnostics under the 2026-10-06 rule; any parse failure class that appears is functional.

## 4b. Measured so far (2026-10-14)

**Sampler**
- Re-export Q4 sha `76c1e4c604798362b62aae90cce1b616aa379981359fefe05e9968f96e9d638c`. This is **byte-identical** to the self-gen lane's Q4, so the export is reproducible and both self-gen sets come from the same GGUF.
- The parent digest was unchanged (export-check receipt `receipts/export_check_Q38-27BU-parent_toolcall.json`); the bf16 GGUF was deleted from scratch by export-check.
- Servers:
  - GPU1 pid 4054571, healthy after 183.4 s.
  - GPU2 pid 4058904, healthy after 9.0 s.
  - n_ctx 32,768, 4 slots × 8,192.
- EOG probe: both servers end on [248044, 248046, 248063, 248064, 248065], equal to expected (`receipts/eog_probe_toolcall.json`).

**Schema library** (`schemas-2026-10-14/manifest.json`, `d0a79632…`)
- 148 toolsets: 144 parent-synthesized over 48 domains plus 4 sandbox.
- 542 functions.
- Sealed 16:17:25Z.

**Plan** (`plan-2026-10-14.jsonl`, sha `78a6dad1…`): 2,400 items.

| Category | Items |
|---|---|
| single | 513 |
| parallel | 402 |
| sequential | 310 |
| no_tool | 261 |
| multiturn | 240 |
| missing_tool | 234 |
| error | 225 |
| clarify | 215 |

**Smoke** (1.0.3, 16:17:30-16:28:27Z)
- 12 of 15 items accepted:
  - accepted: single 3, parallel 3, missing_tool 2, no_tool 2, sequential 1, multiturn 1;
  - rejected after 3 attempts: no_tool 1, clarify 2.
- Attempt rejections:
  - "called a function where none applies" 3;
  - truncated backend 4;
  - clarify "called before the missing detail was given" 3;
  - "no call for a tool request" 1;
  - "no turn with two or more calls" 1.
- Finalize: 7 conversations in the plan-order cut, 13,536 tokens, **3,164 supervised (23%)**, 0 screened out. Manifest `bed6ed39…`.
- Export: 7 × 2,871. Accepted rendered rate 34.86 tok/s on two P40s (`smoke-…/progress.json`).

**Supervised share:** at 23%, 1.5M rendered tokens give ≈ 0.35M KD positions, not the ≈ 0.5M estimated in §3. The tools system blocks are long. The exact figure comes from `finalize`.

## 5. Runtime

- **Export:** 14:24:56Z to about 15:09Z.
  - bf16 GGUF 54,657,734,336 B (deleted by export-check). Q4 16,810,714,816 B, sha `76c1e4c6…`.
  - Log: `$OC/distill/toolcall/logs/export.log`.
- **Production run:** `toolcall_selfgen.py` 1.0.4, started 2026-10-14 16:30:52Z.
  - Driver log: `$OC/distill/toolcall/logs/driver.log`. Live progress: `$OC/distill/toolcall/run-q38-q4km-2026-10-14/progress.json`.
  - At 16:41:36Z: 13 accepted / 4 rejected, 24,044 tokens, 37.34 accepted tok/s, `eta_hours` 10.98.
  - At 16:42Z: 19 journaled, 15 accepted, 27,359 tokens, 5,772 supervised (21%).
  - **Stopped 17:02:25Z for the 1.1.0 review fixes** (SIGTERM to driver 21620 then run 21884; servers left up). 42 journal rows, all parse, 62,494 accepted tokens; in flight and lost (re-run on resume): tc-00011/37/39/44/45. Backups: `run-…/journal.v104-stopped-2026-10-14.jsonl.bak` (sha `6a78d328…`), `progress.v104-stopped-2026-10-14.json.bak`, `logs/driver.v104-production-stopped.log`.
  - Read-only `content-audit` before relaunch (`toolcall/audits/content_audit_v110_*.jsonl` + `.summary.json`): production 2/7 accepted zero-call rows fail (tc-00030 pattern "he verificado"; tc-00018 Italian order-specific refund question, judge `fits`), smoke 1/4 (tc-00004 "Ich habe Ihre Details geprüft"; smoke is not a training set). Bank rebuild under 1.1.0 is byte-identical to the sealed bank (`de0e0003…`).
  - **Resumed 17:07:33Z on 1.1.0** (same driver, same run dir, appended to `driver.log`): "content rule zero-call-content-v1 applied to 7 earlier rows: 2 reopened"; "42 journaled (31 accepted, 59,749 tokens), 2360 pending".
  - tc-00030 re-ran from attempt 1 and was accepted (judge `{"fits": false, "claims": false}`, no pattern; it now asks for the confirmation code instead of inventing a status). tc-00018 ended `rejected` after attempts 1-2 (content again, then a call where none applies). At 17:13Z no accepted zero-call row lacked a current verdict.
  - At 17:16:03Z: 36 accepted / 11 rejected, 69,508 tokens, session rate 20.15 tok/s, `eta_hours` 19.72; 0 internal failures logged. The rate covers only the first 9 minutes after the restart (in-flight items restarted from scratch), so the ETA is not settled; the 1.0.4 session measured 37-42 tok/s.
- **Expected:** run end between ≈ 2026-10-15 03:00Z (at the 1.0.4 rate) and ≈ 13:00Z (at the first post-restart reading), then `finalize` (screen) and `export`. The driver then stops both servers through their receipts, writes the RELEASE line to `queues/p40.claims`, and deletes the Q4 from scratch.
- **On any failure** nothing is torn down. Rerun the driver to resume: the journal, the plan-order cut and the run config lock make it idempotent.
- **Rejections so far:** the parent most often calls a function where none applies (no_tool / missing_tool; 9 of 18 attempt rejections). It also calls before a missing detail is given (clarify), and it occasionally calls a function that is not offered.
  - Rejection sampling keeps only the correct behaviour.
  - The KD target is still the parent's full distribution at each position.
  - The 0.1 LM-CE floor on the sampled turn is the only term that pulls toward the kept behaviour.
- **Next steps (not done here):**
  1. bf16 rescoring of `rescore/toolcall-q38-2026-10-14.safetensors` with `qtext_teacher_cache build-text` on GPU0 (after D1) or GPU6 (between K3/H3 jobs, under the MemAvailable gate).
  2. Add the sealed cache to the D2 `--data-config`.
  3. Bank responses from each D-stage checkpoint, which needs the review-runner hook in §4.
