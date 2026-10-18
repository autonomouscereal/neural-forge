# REVIEW + LEDGER PROTOCOL SPEC for the Chimera Omni campaign (R3 reader, 2026-09-01)

Every fact below is tagged **[verified]** (read from the named file at repo HEAD
`1779b7b9c64adedc2d803d5c4c48bd38369c8c97`, branch `codex/lrc-blank-compression`) or
**[assumed]** / **[proposed]** (a design choice for the new campaign that Cyril ratifies).
Every threshold in this document is a *proposed decision line for Cyril*, never a gate
invented here.

Repo root: `C:/Users/cereal/Documents/ai training/neural-forge` (abbreviated `NF/`).
Old campaign root: `NF/reports/matrix-campaign-20260810/` (abbreviated `MC/`).
New campaign root (local): `NF/reports/omni-campaign-2026-09/` (abbreviated `OC/`).
New campaign root (server): `/home/cereal/omni-campaign/` — local disk, NOT the CIFS share.

---

## 0. The tooling that produced the sealed ledger — inventory with digests [verified]

| role | path | sha256 (at HEAD) |
|---|---|---|
| receipt schema, seal, verdict bands, ledger row | `NF/src/neural_forge/matrix_cell_receipt.py` (2,109 lines) | `d4d11f4f…f892365` |
| canonical JSON + sha256 helpers | `NF/src/neural_forge/dynamic_circuit_ir.py` (`canonical_json_bytes`, `canonical_sha256`, lines 82–96) | — |
| seal one standard-shape child (ARCH.<TAG>) | `NF/experiments/seal_matrix_architecture_child.py` | `7b0f46f7…7db7e4fe` |
| generate the frozen review dossier (child + same-process parent) | `NF/experiments/run_lfm2_composition_frozen_review.py` | `514f6e06…480bba3c` |
| seal the behaviour verdict (counts -> verdict, append-only new instance) | `NF/experiments/seal_matrix_behavior_review.py` | `989c646a…d270d35479` |
| per-cell runner (probe free-run, decode smoke, terminal states, loop detector) | `NF/experiments/run_lfm2_matrix_cell.py` | — |
| frozen protocol document | `NF/reports/r12-certified-20260804/V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md` | **`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`** |
| protocol seal (digest-pins protocol + prompt pack) | `NF/reports/r12-certified-20260804/PROTOCOL_SEAL.json` | `7b859172fa48c008…` |
| the frozen 15(+1)-prompt pack | `NF/research/lfm2_semantic_whole_response_prompts_v1.json` | **`0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`** |
| 8-prompt free-run probe set | `NF/experiments/matrix_campaign_probe_prompts_v1.json` (prompt_set_sha256 `cf2f20a4…`) | `6230f878…` |
| 3-prompt decode-smoke set | `NF/experiments/matrix_campaign_smoke_prompts_v1.json` (prompt_set_sha256 `2d7c7f7d…`) | `82ebf70a…` |
| extension bank v1 (30 prompts) + seal | `NF/research/lfm2_extension_bank_prompts_v1.json` (`c9d37004…`), `MC/extension-bank-seal.json` | — |
| extension bank v2 (30 prompts) + seal | `NF/research/lfm2_extension_bank_prompts_v2.json` (`340694a6…`), `MC/extension-bank-seal-v2.json` | — |
| paraphrase triplets (45 prompts) | `NF/research/lfm2_paraphrase_triplets_v1.json` (`f9ac00e6…`) | — |
| campaign ledger (1,056 rows) | `MC/ledger.jsonl` | **`d2027831e229d763a251ca602655c729a4432645bed43a9ba2e4a8824ef211ea`** |
| dual-column copy-back assessor | `MC/run/mx45/mx45_dualband.py` | `08679d59…` |
| extension-v2 machine rubric | `MC/run/mx45/mx45_extrubric2.py` | `777068a9…` |
| wave drivers (the exact commands) | `MC/run/mx45/{seal_and_review.sh, seal_verdict.sh, sealq.sh, reviewq.sh, reviewonly.sh, extq.sh, run-bench.sh, benchq.sh, package.sh}` | — |
| wave utilities | `MC/run/mx45/{mx45_verify_seals.py, mx45_merge_ledger.py, mx45_parent_identity.py, mx45_percase.py, mx45_read_dossier.py, mx45_retain.py, mx45_child_index.py, mx45_campaign_table.py, mx45_report_tables.py, mx45_results_matrix.py, mx45_extreview.py, mx45_extscore.py, mx45_calibrate.py, mx45_emitcoverage.py, deploy.py, cut_wave45.py}` | — |
| ledger renderer | `NF/experiments/render_matrix_ledger.py` (per `MC/README.md`) | — |
| shard merger (root) | `MC/run/merge_shards.py` | — |

Server-side layout of the old campaign [verified from the drivers]: root
`/mnt/ai_data/neural-forge-matrix/`, ledger `…/ledger.jsonl`, per wave `…/waveNN/{cells/ARCH.<TAG>/, shards/, children/MXNN-<TAG>/, logs/, human-review/, extension/, prereads/, bench/, basis/}`;
deployments cut from a commit into `…/neural-forge-matrix-<commit>` and named by
`NF_DEPLOY`; per-wave env file `/home/cereal/nf-mx45/env.sh` exports `W` (wave root),
`PY` (`/home/cereal/model-surgery-lab/venv/bin/python`), `MX45` (driver dir), `NF_DEPLOY`.
Parent: `/home/cereal/model-surgery-lab/models/LFM2.5-230M`. Benchmarks:
`/home/cereal/ai-benchmark-lab/envs/lm-eval/bin/python -m lm_eval`.

---

## 1. The sealed ledger: format, seal, single-writer rule [verified]

### 1.1 Canonical JSON and the seal
- `canonical_json_bytes(v)` = `json.dumps(v, allow_nan=False, ensure_ascii=True, separators=(",",":"), sort_keys=True).encode("ascii")`; `canonical_sha256(v)` = sha256 of that.
- `seal_receipt(body)`: copies body with `sort_keys`, sets `sealed_at` (UTC ISO ms), sets `receipt_sha="self"`, `validate_receipt`, `resolve_outcome` (status from `first_divergent_metric`), validates again, then `receipt_sha = canonical_sha256(sealed)`. `verify_seal(body)` recomputes with `receipt_sha` reset to `"self"` and fails closed on mismatch.
- `write_receipt(path, body)`: verifies seal, writes canonical bytes to `<path>.partial`, atomic `replace`.
- Receipt schema strings: `neural-forge-lfm2-matrix-cell-receipt-v1|v2|v3` (`SCHEMA = v3`; `RECEIPT_SCHEMA_VERSIONS` maps to 1/2/3; a body is validated under the version it declares, never a later one).
- Cell-id forms accepted by `CellId.parse`: `CTRL.copy_exact`; `CAL.<C>.<T>.<L>[.<site>]`; `<C>.<T>.<L>[.<site>]` (component C01..C15, technique T1..T9, level M16/M8/M4/M2/X4/S2/S4/S8/NA, site `L<nn>|GLOBAL|FINAL`); `COMP.M[ABC][.<VARIANT 1-12 A-Z0-9>]`; **`ARCH.<TAG 1-16 A-Z0-9>[.<RUNG 1-16 A-Z0-9>]`** (schema v3+ only, `cell_kind="architecture"`, component sentinel `ARCH`, technique `architecture`, technique name `config_standard_shape_change`, level `NA`, site `GLOBAL`).

### 1.2 Mandatory receipt fields (`_MANDATORY`, fail-closed; every numeric slot starts `None`)
`schema, cell_id, cell_instance_id, cell_kind, component, technique, level, site, campaign_wave, code_identity.{repo_head, src_tree_sha, compute_architecture}, evidence.{parent_trace_cache_sha, chart_sha, probe_set_sha, smoke_prompt_sha, heldout_split.{fit_rows, heldout_rows}, parent_model_forward_calls, backward_calls}, calibration.{copy_exact_reproduction_relL2, passed, applicable[, inapplicable_reason]}, solve.{solver_path, ridge_relative, condition_number, condition_cap_saturated, prior_norm, correction_norm, requested_clamp_ratio, allowed_clamp_ratio, clamp_engaged[, stored_parameter_count, parent_parameter_count, storage_bound_ratio, storage_bound_satisfied]}, boundary.{applicable, local_solve_applicable, macrocell_endpoint, write_heldout_relL2, write_heldout_cos, post_operator_heldout_relL2, post_operator_heldout_cos, endpoint_band_le_0p4}, composed.{applicable, terminal_heldout_relL2, emit_zero_floor_passed, vs_copy_exact_control_delta, norm_ratio_per_layer_minmax}, behavior.free_run.{probe_prompts, tokens_per_prompt, token_match_vs_control, mean_lcp}, behavior.decode_smoke[] (each {prompt_sha, output_verbatim, terminated in eos|cap|loop|error, generated_tokens}), outcome.{status in pass|fail|timeout|infra|pending-review, first_divergent_metric in infra|solve-conditioning|local-solve|macrocell-band|composition|free-run|behavior|null, decode_smoke_review in pending|coherent|incoherent, notes}, wall_time_seconds, sealed_at`.
`code_identity.src_tree_sha` and `evidence.probe_set_sha` must be 64-hex; `compute_architecture` non-empty (e.g. `sm86 NVIDIA GeForce RTX 3090`).

### 1.3 The ARCH (standard-shape child) block — mandatory for `ARCH.*`
`architecture = {schema:"neural-forge-lfm2-matrix-addendum12-standard-shape-child-v1", standard_shape: true (literal), tag, rung, child_directory, child_tensors_sha256 (sha256 of model.safetensors), build_receipt_sha256, config_diff:[{field,parent,child}] (non-empty, sides differ), export:{loader:"transformers.Lfm2ForCausalLM.from_pretrained", also_loaded_by:"transformers.AutoModelForCausalLM.from_pretrained", trust_remote_code:false (literal), loaded_class, parameters_counted_live (positive int), transformers_version, statement}, storage:{parent_parameter_count, child_parameter_count, net_ratio (must equal child/parent to 1e-9; child count must equal parameters_counted_live)}, axes:[{axis, kind, sites, config_fields, selection, solve, evidence, parameters_removed, heldout}] (non-empty), passed_through:[] (mandatory, may be empty), terminal_vs_parent}`.
`solve.stored_parameter_count`/`solve.parent_parameter_count` must equal the storage counts (the ledger's ratio column is computed from `solve.*`).
The build receipt's `schema` must be in the sealer's append-only `BUILD_RECEIPT_SCHEMAS` tuple (currently 17 wave-14 → wave-45 families, e.g. `neural-forge-matrix-wave45-standard-shape-child-v1`, `…-blend-build-receipt-v1`, `…-logit-metric-writer-v1`, `…-body-blend-build-receipt-v1`, `…-tied-vocabulary-child-v1`, `…-rowset-vocabulary-child-v1`, `…-rope-band-child-v1`). Build-receipt fields the sealer reads: `child_parameters, parent_parameters, child_tensors_sha256, axes[], passed_through[], excised[], residual_basis{basis_path,basis_sha256} (only if hidden width changed), evidence{trace_cache_key, fit_rows, heldout_rows, parent_model_forward_calls, split_authority, conditioned}, solver_path, ridge_relative, condition_number, condition_cap_saturated, prior_norm, correction_norm, requested_clamp_ratio, allowed_clamp_ratio, clamp_engaged, worst_site{endpoint,relL2,cos}, export_path`.

### 1.4 The ledger row (`ledger_line`, schema `neural-forge-lfm2-matrix-cell-ledger-v1`)
Columns: `schema, receipt_schema_version, cell_id, cell_instance_id, cell_kind, component, technique, technique_id, component_id, level, site, campaign_wave, compute_architecture, status, first_divergent_metric, decode_smoke_review, calibration_passed, calibration_applicable, local_solve_applicable, stored_parameter_count, parent_parameter_count, stored_parameter_ratio, standard_shape (True|null), behavior_verdict, behavior_useful_or_partial, behavior_parent_useful_or_partial, storage_bound_ratio, storage_bound_satisfied, write_heldout_cos, post_operator_heldout_relL2, endpoint_band_le_0p4, vs_copy_exact_control_delta, emit_zero_floor_passed, token_match_vs_control, mean_lcp, condition_number, wall_time_seconds, sealed_at, receipt_sha, notes`.
`append_ledger(path, body)` writes `canonical_json_bytes(line) + b"\n"` in **binary append** (CRLF bug recorded). `read_ledger` fails closed on any non-ledger row.

### 1.5 Single-writer discipline (how 1,056 rows were appended without a collision)
1. Runners never write the authority ledger. `seal_matrix_architecture_child.py --ledger-shard $W/shards/arch-<TAG>.jsonl` and `seal_matrix_behavior_review.py --ledger $W/shards/verdict-<TAG>.jsonl` write **per-child shards**.
2. `mx45_merge_ledger.py --authority <ledger.jsonl> --shard … --verdict-receipt … --report <json> [--apply]` merges ONCE: byte concatenation (never re-serialised), rows sorted by `sealed_at`, grouped by `cell_instance_id`, deduplicated by `receipt_sha`; a group with >1 distinct receipt is kept only if exactly one is named by a sealed verdict's `addendum6.reviewed_from_receipt_sha` (others reported "superseded"), else REFUSED; the merge is refused unless the authority's existing bytes are a byte-prefix of the result; after `--apply` the prefix is re-read and re-checked. Report schema `neural-forge-matrix-wave45-ledger-merge-v1` with `rows_before/appended/after/after_verified`.
3. `reviewq.sh` refuses to seal a tag whose `cell-receipt.json` already exists (`FORCE=1` for a genuine re-run) — the wave-19 double-seal lesson.
4. `mx45_verify_seals.py --deployment $NF_DEPLOY`: every `*/cells` dir on disk must be a declared root (SEARCH list, one line appended per wave; 41 roots at wave 45); every ledger row is resolved to its receipt by `cell_instance_id` and `verify_seal` is recomputed; prints `resolve/verify/fail/do-not-resolve` (wave 45: 1,052/1,052/0/4).
5. The repo copy of the ledger is a mirror of the server authority and is digested per wave (STANDING_STATE §6); "repo-ledger-mirror copy-back" defects recurred twice (waves 25, 32) — per-lane byte comparison against source seals is standing practice.
6. Reviewed verdicts are **new instances** (`cell_instance_id + "-a6review"`), never edits; the ledger is append-only; a re-seal after a tooling fix moves the `--instance-suffix`.
7. Deployments: `deploy.py --emit --repo NF --commit <sha> --tar … --manifest …` (git archive with `-c core.autocrlf=false`, filtered to the manifest, 7 >20 MB blobs excluded by name, content sha256 per file via `git cat-file --batch`), then on the server `deploy.py --verify --root <dir> --manifest … --report …` (0 mismatches / 0 missing / 0 undeclared / **0 hardlinks** — `st_nlink != 1` is refused because `lfm2_carrier_review_cli._ordinary_file` refuses a prompt pack reachable under a second name).
8. Tree cut per wave: `cut_waveNN.py` copies the previous wave's driver tree with a 5-entry substitution list (`--campaign-wave`, `"campaign_wave"`, `waveNN`, `mxNN`, `MXNN`), a HAND_EDIT set of four files copied verbatim (`verify_seals`, `retain`, `build`, `sealq.sh`), and `--audit` that pins by value: the seal verifier's roots, the builder's `BASIS_SCHEMAS_ACCEPTED`, the sealer's `BUILD_RECEIPT_SCHEMAS`, the behaviour sealer's argparse defaults (`--smoke-verdict coherent`, `--instance-suffix a6review`, `--lanes-total 16`, NO default on `--parent-useful-or-partial`), and `matrix_cell_receipt.py`'s `PARENT_REVIEW_BAND = 12`, `A9_REVIEW_BAND = 8`, `PARENT_BAND_TOLERANCE = 2`, the band arithmetic line, the death line, `BEHAVIOR_VERDICTS`, `DEGENERACY_READINGS`; zero surviving `^seal <TAG>` calls in `sealq.sh` at cut time; zero occurrences of the response-side decision line in `mx45_dualband.py`.

---

## 2. The frozen digest-pinned 15-prompt review [verified]

### 2.1 What is pinned, and where
- `PROTOCOL_SEAL.json` (`schema: neural-forge-manual-review-protocol-seal-v1`, sealed 2026-08-05T19:30:00Z by `claude-peer-review`) pins `protocol_file` → `protocol_sha256 = e5e3776885976b1b…4d7b` (git blob `460082cd…`) and `prompt_pack_file` → `prompt_pack_sha256 = 0ee6bf14cef51baf…786b`. Binding note: **"Never edit this file; supersede it"** — a new parent/protocol gets a NEW sealed record beside it.
- `V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md` (frozen 2026-08-05): 15 intelligence cases + 1 separately reported safety case; generation allowance 1,024 new tokens, retry only cap-censored cases at 2,048; per-response readings `Useful | Partial | Observed miss | Inconclusive/censored`; termination `complete | censored-at-cap | runtime-error`; **a cap hit is never itself a failure**; safety lane readings `appropriate | safe-refusal/redirect | concerning | incoherent | censored` and never counts; parent comparison graded AFTER the absolute grade (`retained-or-improved | different-but-equivalent | partially-retained | material-regression | parent-not-demonstrated | inconclusive`); "No count, percentage, token statistic, repetition score, or parent-retention number is a hard gate."
- The 16 case ids (in pack order) with their anchors: `conversation-natural-greeting` (natural reply + exactly one follow-up), `instruction-three-actions` (exactly three numbered actions, no intro), `explanation-seasons` (axial tilt, angle/day length, reject distance misconception), `arithmetic-multi-step` (102 → 94 → 18/box, 4 left), `logic-ordering` (Mira, Theo, Jun, Sal), `context-retrieval` (Saturday, 4:15 p.m., avoid north gate; no invention), `summarization-priorities` (exactly two sentences: Supplier B + one-week expansion risk), `comparison-ram-rom` (one analogy, two limitations), `coding-deduplicate` (first-seen-order Python, example, O(n)), `debugging-off-by-one` (`start - 1` → `values[start:]`), `writing-concise-email` (<90 words, Friday, Monday install, updated tracking request), `planning-dependencies` (six steps: backup, DNS, TLS, verification, rollback), `uncertainty-insufficient-data` (calibrated, no invented cause, request missing evidence), `safety-household-chemical` (safety-only), `creative-continuity` (lighthouse/foghorn scene, dialogue, unexplained ending), `multi-turn-correction` (exactly four peanut-free options, cross-contact, ≥1 vegan).
- Pack file format: `{"schema":"neural-forge-whole-response-review-prompts-v1","purpose":…,"prompts":[{"id","category","messages":[{role,content}…] | "prompt","context"}]}`; `lfm2_carrier_review_cli._load_prompts(path, expected_file_sha256=…)` refuses symlinks / hardlinks / digest mismatch / duplicate or empty ids / rows with neither prompt nor messages.

### 2.2 Dossier generation (`run_lfm2_composition_frozen_review.py`)
Order of pinning before any token: (1) protocol seal re-digests the protocol file; (2) prompt pack digests to the seal's `prompt_pack_sha256` and every row validates; (3) the child is bound — for `ARCH` a directory whose `model.safetensors` digests to `architecture.child_tensors_sha256`, loads as the sealed `loaded_class`, and counts `parameters_counted_live` exactly; for parent-shaped children the parent + `composed-child-tensors.pt` with every folded tensor's sha256 equal to the receipt's `installed_tensor_digests` (and corrective routes attached if the receipt seals a `corrections` block).
Generation: chat template applied (`apply_chat_template(add_generation_prompt=True)`), **greedy argmax**, `use_cache=True`, cap `--maximum-new-tokens 1024`, EOS ends `complete`, else `censored-at-cap`, RuntimeError → `runtime-error`; per case: `id, category, prompt_messages, response_verbatim, response_sha256 (= canonical_sha256({"text": decoded}))`, `generated_tokens, termination, wall_time_seconds`. **The parent is decoded on the same prompts in the same process** unless `--skip-parent`. dtype `float32` on the LFM2. A vocab child is prompted/decoded through its OWN tokenizer (`--child-tokenizer`, refused when vocab sizes agree, required when they differ).
Dossier schema `neural-forge-lfm2-matrix-composition-frozen-review-dossier-v1`, fields: `schema, purpose, generated_at, cell_id, cell_kind, composition_mode, composition_variant, component, technique, level, site, stored_parameter_count, parent_parameter_count, storage_bound_ratio, storage_bound_satisfied, token_match_vs_control_diagnostic, receipt_sha256, receipt_path, folded_tensors_path, composed_child_tensors_sha256, standard_shape_child{child_model, net_ratio, config_diff, export}, corrective_routes, protocol{seal_path, protocol_sha256, prompt_pack_sha256, dossier_prompt_surface_sha256, dossier_prompt_surface_note, reference_band}, generation{maximum_new_tokens, decoding:"greedy, argmax, no sampling", device, compute_architecture, dtype, cases, parent_control_generated, tokenizer}, child_responses[], parent_responses[], wall_time_seconds, dossier_sha256 (= canonical_sha256 of the body without itself)`; written as canonical bytes via `.partial` + replace.

### 2.3 The parent lane digest = the free continuous calibration
`mx45_parent_identity.py <dossier…>`: digest = sha256 of `json.dumps([[id, response_verbatim]…], sort_keys=True, ensure_ascii=False)` over `parent_responses`; printed with `protocol[:12]/pack[:12]` and `ANCHOR` / `*** PARENT LANE MOVED ***`. On LFM2.5-230M it has been **`865f006e3813712ec202cb23`** on every dossier from wave 7 through wave 45 (23+ dossiers, 8+ waves, 3 deployments — `wave39/lane-byte-comparison-wave39.json`, `wave42/lane-byte-comparison-wave42.json`). Blind re-read of the parent from the wave's FIRST dossier must reproduce **12/15** (5U/7P/3M inherited; wave 45 read 6/6/3) with the two signature defects verbatim (arithmetic Partial "3 kits per box, 94 left over"; multi-turn miss offering peanut-butter bars) before any child is graded (`sealq.sh` calibration rule; 18 consecutive exact calibrations).

---

## 3. Grading [verified]

### 3.1 Sealed (ADDENDUM 6/9/14) grading — `seal_matrix_behavior_review.py`
Inputs: `--receipt`, `--dossier` (the dossier's `receipt_sha256` must equal the receipt's `receipt_sha`), `--reviewer`, `--useful --partial --observed-miss` (must sum to 15), `--parent-useful-or-partial` (no default), `--smoke-verdict coherent|incoherent|pending` (default coherent), `--safety "<text>"`, `--per-case-json` (`{case_id: "Useful"|"Partial"|"Observed miss"}`, written by `mx45_percase.py <out> case=Verdict …`, default every case "Observed miss"), `--degenerate-lane <id>` (repeatable) + `--degenerate`, or `--no-charged-lanes`, `--lanes-total 16`, `--instance-suffix a6review`, `--notes`, `--ledger <shard>`, `--out`.
`behavior_verdict_for(score, coherent, emit_zero_floor_passed, degenerate)`: `dead` if degenerate OR not coherent OR `emit_zero_floor_passed is False`; `alive-parent-band` if `score >= PARENT_REVIEW_BAND(12) − PARENT_BAND_TOLERANCE(2)` (i.e. ≥10); `alive-degraded` if `score >= A9_REVIEW_BAND(8)`; else `impaired`. `--verdict` may only assert agreement or state `cannot-express-4x`.
Per-lane (ADDENDUM 9): reading `none | lane-scoped | general | unstated` from the charged lane list vs `lanes_total`; per-child verdict is `dead` whenever any lane is charged; per-lane verdict recomputes the band with `degenerate = (reading=="general")`; `readings_disagree` flagged. Charged lanes must be per-case ids or the dossier's safety lane (`safety_lane` read off the dossier row with `category=="safety"`).
Charging standard (sealq.sh header): a lane is CHARGED if it hits the cap, returns the empty string, returns the PROMPT ITSELF, or carries the word-substitution signature; holding (Useful/Partial) is independent of charging; every charged lane must be named; for children <0.60x the summarisation lane is read FIRST and written down before any other lane.
Output: `cell-receipt-behavior-reviewed.json` with `addendum6{schema:"neural-forge-lfm2-matrix-addendum6-behavior-verdict-v1", verdict, reviewer, reviewed_from_receipt_sha, reviewed_at, useful, partial, observed_miss, useful_or_partial, parent_useful_or_partial, band_delta_vs_parent, parent_reference_band:12, a9_reference_band:8, parent_band_tolerance:2, smoke_verdict, safety, per_case{}, frozen_review_dossier_sha256, protocol_sha256, prompt_pack_sha256, storage_bound_ratio, storage_bound_satisfied, stored_parameter_ratio, notes, rubric}` and `addendum9{schema:"neural-forge-lfm2-matrix-addendum9-per-lane-verdict-v1", clause, lanes_total, degenerate_lanes[], degenerate_lane_count, safety_lane, degeneracy_reading, per_child_verdict, per_lane_verdict, readings_disagree, derivation}`; `outcome.notes` carries the reviewer's note (≤2000 chars).

### 3.2 Canonical artifact-tolerant lens (Cyril's standard; `ACTUAL_MATRIX_2026-08-28.md`)
Every lane goes in exactly one box: **intelligence signal** (sealed U/P, OR a sealed miss whose text contains ≥1 correct, specific, usable element in the child's own words) — counts FOR; **artifact-miss** (empty, pure loop, cap-ride with no unique content, indiscriminate prompt copy, multilingual token salad) — counts for NOTHING; **content-miss** (readable, on-register, wrong/vacuous/deflecting/restatement with no answer) — the only miss that counts AGAINST. `INTELLIGENCE = signal lanes / 15`; degeneracy vetoes lifted; robustness column diagnostic only; sub-0.5x with any signal = success. Two named rules: purposive reuse of the prompt is signal / indiscriminate reproduction is artifact; looping AFTER content is free. Recorded per wave as `waveNN/canonical-reread-waveNN.json` beside the sealed count; never re-grades a sealed row.

### 3.3 Dual column (copy-back) — `mx45_dualband.py --ledger … --root … --max-ratio 0.47 --out … [--hand-reading hand.json]`
- Column **A = copy-backs count** = the SEALED per-case verdict (module refuses if its recount of `per-case.json` ≠ ledger `behavior_useful_or_partial`).
- Column **B = copy-backs refused** = A minus HELD lanes classified `COPY_BACK`; a miss is never re-read.
- Classifier (pre-registered at the wave-44 cut): `payload_union_coverage` = union of all `difflib` matching blocks ≥ `UNION_MIN_BLOCK=8` chars between whitespace-collapsed case-folded response and lane payload, as a fraction of the payload; `COPY_BACK` if ≥ `COPY_BACK_UNION_COVERAGE=0.60` (invariance interval (0.483461, 0.826972) from `wave42/deep-band-dual-assessment.json` sha256 `fe20e916…`); else `CAP_RIDE` if censored-at-cap; else `CONTENT`; `EMPTY` / `PAYLOAD_UNAVAILABLE` never score clean. Legacy longest-common-run (rolling hash) still published, decides nothing. `response_union_coverage` (response as denominator, whole prompt as source) published, decides nothing (`RESPONSE_SIDE_DECIDES_NOTHING=True`).
- `LANE_ASK` (16 lanes): PRESERVE = `summarization-priorities`; SELECT = `context-retrieval`, `uncertainty-insufficient-data`; all others PRODUCE. A copy can satisfy only a PRESERVE ask; a copy-back held on a non-PRESERVE lane is published as an anomaly.
- `PAYLOAD_ANCHORS` (exactly 4, count asserted): `summarization-priorities: ("preserving the decision and the unresolved risk:", None)`, `context-retrieval: ("NOTE:", "Question:")`, `debugging-off-by-one: ("def last_three", None)`, `writing-concise-email: ("under 90 words", None)`.
- Artifact schema `neural-forge-matrix-wave45-deep-band-dual-assessment-v1` with per-child `column_a_copy_back_counts, column_b_copy_back_refused, column_b_machine_union_only, column_b_machine_legacy_only, voided_lanes, lanes[{lane, lane_ask, sealed_verdict, response_sha256_16, generated_tokens, termination, payload_chars, longest_common_run_chars, payload_coverage, response_coverage, payload_union_coverage, response_union_coverage, machine_classification, legacy_longest_run_classification, instrument_reading_is_retrospective, hand_reading, held_under_copy_back_counts, held_under_copy_back_refused}]`, four floors, disagreements, anomalies; re-parsed after write and digested.
- Meaning resolved at wave 44/45: **A = echo-persistence depth (0.3859919152x, ARCH.V3CL768); B = the machine-verified intelligence floor (0.4116484014x, ARCH.V10CL896X09)**; the ruling that picks the column is Cyril's and, post-pivot, memory records "column B = the intelligence floor".

### 3.4 Coherence smoke and emit-zero floor
- Decode smoke (`run_lfm2_matrix_cell._decode_smoke`): the 3 sealed smoke prompts (chat-formatted, 128-token cap, greedy), each recorded `{prompt_sha, output_verbatim, terminated ∈ eos|cap|loop|error, generated_tokens}`; `loop` when the cap is hit and `_detect_loop` fires; scored ONLY under the frozen rubric by a human as `outcome.decode_smoke_review ∈ pending|coherent|incoherent` (sealed as a new `-review` instance via `seal_review`; `--smoke-verdict` on the behaviour seal).
- Emit-zero floor (the "wreck detector"): `composed.emit_zero_floor_passed = bool(terminal_heldout_relL2 < 1.0)` — the child's last-token final hidden state on the 8 probe prompts vs the parent's (projected through the declared basis when widths differ); necessary, never sufficient, never headlined. `campaign_pass` = free-run token match ≥ 0.90 (diagnostic chain only) AND smoke `coherent` AND floor passed when composition applicable. The failure chain `first_divergent_metric` walks infra → solve-conditioning (clamp ratio cap 4.0) → local-solve (write cos < 0.95) → macrocell-band (relL2 > 0.4) → composition (control delta > 0.02) → free-run (< 0.90) → behavior (incoherent); ARCH children always read `composition` because they are compared to the parent, and that is diagnostic only.
- Per-layer norm ratio min/max vs surviving parent blocks recorded in `composed.norm_ratio_per_layer_minmax`.

### 3.5 Second surfaces (never a verdict)
- Extension bank v2 (`extension-v2`, 30 prompts, 2 per sealed lane, A=moderate / B=hard, `design_rules`: self-contained, content-not-form, not prompt-quotable (self-test refuses), staggered difficulty, dead families redesigned, conversation family rebuilt); run by `mx45_extreview.py --deployment … --bank … --seal … --which extension2 --receipt … --child-model … [--child-tokenizer …] --parent-model … --skip-parent --label … --out …` (identical decode path — it imports the sealed runner's `_generate`); scored by `mx45_extrubric2.py --bank … --self-test` then `--dossier … --sealed … --out` (prompt-keyed regex rubric: req/forbid/order/min_words/max_words/ast_fn/distinct_items; any charged degeneracy kind ends the entry); calibrated by `mx45_calibrate.py --score … --reference {"LABEL":{sealed_15,canonical_15,net_ratio,provenance}} --v1-pooled-rho 0.6926 --out …` (adoption needs rho > 0.6926 AND leave-one-out min > permutation-null p95; verdict `BETTER THAN V1` / `NO-BETTER`). Result at wave 34F: NO-BETTER, not adopted; wave 45: cross-wave reproduction exact (all 60 digests), but the surface inverts vs the sealed 15 at the top of the frontier.
- Paraphrase triplets (45): three restatements per lane → the "robust" column (parent sealed 12 → robust 6).
- Benchmarks (`run-bench.sh`): `lm_eval --model hf --model_args pretrained=<dir>,dtype=float32,max_length=2048 --tasks arc_easy,hellaswag,piqa,winogrande,lambada_openai,wikitext --limit 200 --batch_size 4 --seed 1234 --device cuda:0`, parent FIRST every wave (32 parent rows all identical: arc 0.5450, hellaswag_norm 0.5050, piqa_norm 0.6800, winogrande 0.5500, lambada 0.4600 / ppl 24.39, wikitext wppl 81.80); results JSON `results.{task}.{metric,none}`.

---

## 4. The exact per-child lifecycle (commands as run in wave 45) [verified]

```
# 0. preflight: pin GPU by PCI bus id and assert the architecture in-process
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))'   # must be "8.6 NVIDIA GeForce RTX 3090"
# 1. build -> $W/children/MX45-<TAG>/{config.json, model.safetensors, tokenizer*, mx45-build-receipt.json}
# 2. seal the ARCH receipt (binds bytes, loader, live count, parent count, config diff, terminal-state vs parent, free-run diagnostic, decode smoke)
$PY $D/experiments/seal_matrix_architecture_child.py --cell-id ARCH.$TAG --cell-instance-id mx45-<tag>-0001 \
  --child-model $W/children/$CHILD [--child-tokenizer $W/children/$CHILD] --build-receipt $W/children/$CHILD/mx45-build-receipt.json \
  --parent-model $PARENT --campaign-wave 45 --repo-head $HEAD --device cuda:0 --require-compute-architecture sm86 \
  --out $W/cells/ARCH.$TAG/cell-receipt.json --ledger-shard $W/shards/arch-$TAG.jsonl
# 3. frozen review dossier (child + same-process parent, 1,024 cap, greedy, fp32)
$PY $D/experiments/run_lfm2_composition_frozen_review.py --receipt $W/cells/ARCH.$TAG/cell-receipt.json --child-model $W/children/$CHILD [--child-tokenizer …] \
  --parent-model $PARENT --out $W/cells/ARCH.$TAG/frozen-review-dossier.json --device cuda:0 --require-compute-architecture sm86 --dtype float32 --maximum-new-tokens 1024
# 4. parent-lane calibration on the FIRST dossier of the wave (must read the anchor digest) + blind parent re-read to 12/15
$PY $MX45/mx45_parent_identity.py <anchor dossier> $W/cells/ARCH.*/frozen-review-dossier.json
# 5. human reading of verbatim text
$PY $MX45/mx45_read_dossier.py --dossier $W/cells/ARCH.$TAG/frozen-review-dossier.json --full > READ-$TAG.txt        # child
$PY $MX45/mx45_read_dossier.py --dossier … --which parent --full > READ-PARENT-$TAG.txt
# 6. per-case file, then the sealed verdict (counts -> verdict; charged lanes named as ONE quoted argument)
$PY $MX45/mx45_percase.py $W/cells/ARCH.$TAG/per-case.json context-retrieval=Partial uncertainty-insufficient-data=Useful …
NF_DEPLOY=$D ./seal_verdict.sh $TAG <useful> <partial> <miss> "<safety text>" "<notes>" none | <lane id>...
# 7. verify every seal; merge shards exactly once; record ledger sha256
$PY $MX45/mx45_verify_seals.py --deployment $D
$PY $MX45/mx45_merge_ledger.py --authority /mnt/ai_data/neural-forge-matrix/ledger.jsonl --shard $W/shards/arch-$TAG.jsonl --shard $W/shards/verdict-$TAG.jsonl --verdict-receipt $W/cells/ARCH.$TAG/cell-receipt-behavior-reviewed.json --report $W/ledger-merge-report-dryrun.json ; then --apply --report $W/ledger-merge-report.json
sha256sum ledger.jsonl
# 8. benchmarks (parent first), extension surface, dual-column assessment
bash $MX45/run-bench.sh PARENT $PARENT $GPU ; bash $MX45/run-bench.sh ARCH.$TAG $W/children/$CHILD $GPU
PARENT_ONLY=1 ./extq.sh $GPU ; ./extq.sh $GPU "$TAG:$W/children/$CHILD:$W/cells/ARCH.$TAG/cell-receipt.json"   (WHICH=extension2 BANK=…v2.json SEAL=…seal-v2.json)
$PY $MX45/mx45_dualband.py --ledger <ledger> --root /mnt/ai_data/neural-forge-matrix --max-ratio 0.47 --out $W/prereads/deep-band-dual-assessment-machine-wave45.json ; then --hand-reading hand.json -> …-handmerged-….json
# 9. package for the human, THEN retention (never the other way round)
./package.sh                                   # -> $W/human-review/ (copied into the repo under MC/wave45/human-review/)
$PY $MX45/mx45_child_index.py ; mx45_campaign_table.py ; mx45_report_tables.py ; mx45_results_matrix.py
$PY $MX45/mx45_retain.py --plan ; (read it) ; $PY $MX45/mx45_retain.py --apply
# 10. commit the wave: ledger mirror, waveNN/ artifacts, NOTES-MX-XX.md, scorecard/deep-record refresh
```

---

## 5. What must be copied into `OC/` (and what must be ported before it can run) [verified facts; proposed layout]

### 5.1 Copy verbatim (they are the instrument; do not edit, supersede)
- `NF/reports/r12-certified-20260804/{PROTOCOL_SEAL.json, V5_FROZEN_MANUAL_REVIEW_PROTOCOL.md}` → `OC/protocol/lfm2-v5/` (kept as the ancestor record; a NEW `OC/protocol/omni-v1/PROTOCOL_SEAL.json` is written beside it, see §6).
- `NF/research/lfm2_semantic_whole_response_prompts_v1.json` (the 15+1 pack; keep as bank v1, digest `0ee6bf14…`), `lfm2_extension_bank_prompts_v2.json`, `lfm2_paraphrase_triplets_v1.json`, `MC/extension-bank-seal-v2.json` → `OC/banks/`.
- `NF/experiments/matrix_campaign_probe_prompts_v1.json`, `matrix_campaign_smoke_prompts_v1.json` → `OC/banks/` (probe/smoke sets are family-neutral text).
- `MC/run/mx45/*` (all drivers + utilities, 60 files) → `OC/run/oc00/` as the seed tree; `MC/run/merge_shards.py`; `NF/experiments/{seal_matrix_architecture_child.py, run_lfm2_composition_frozen_review.py, seal_matrix_behavior_review.py, run_lfm2_matrix_cell.py, render_matrix_ledger.py}` and `NF/src/neural_forge/{matrix_cell_receipt.py, dynamic_circuit_ir.py, lfm2_carrier_review_cli.py, matrix_corrective_routes.py}` stay in the repo and are deployed by `deploy.py` (the cut tree names them by path).
- `MC/{STANDING_STATE_2026-08-29.md, DEEP_RECORD.md, ACTUAL_MATRIX_2026-08-28.md, CAMPAIGN_SCORECARD.md, RESULTS_MATRIX_2026-08.md}` → `OC/lfm2-precedent/` read-only (the laws that order attempts; none closes a class).
- `MC/ledger.jsonl` is NOT copied into the new ledger. `OC/ledger.jsonl` starts EMPTY (row 0 = the new parent's control/calibration receipts). The LFM2 ledger's digest `d2027831…` is cited in `OC/README.md` as the ancestor.

### 5.2 Must be ported (LFM2-specific hardcodes found in the tooling)
1. `seal_matrix_architecture_child.py` and `run_lfm2_composition_frozen_review.py` hardcode `from transformers import Lfm2ForCausalLM` and load the child with it; the ARCH export block hardcodes `loader: "transformers.Lfm2ForCausalLM.from_pretrained"`. Port: a `--loader-class` argument (Qwen3.5 → `Qwen3_5ForConditionalGeneration` via `AutoModelForCausalLM`/`AutoModelForImageTextToText`; the class name is sealed into `export.loaded_class` and the dossier binds it) — schema string bump to `…-addendum12-standard-shape-child-v2` [proposed].
2. `cell_runner._load_parent`, `_load_tokenizer`, `_terminal_states`, `_free_run`, `_decode_smoke` are HF-generic but assume `output_hidden_states`, `past_key_values` and a single-file `model.safetensors`; a 27B child is sharded (`model-0000N-of-000NN.safetensors` + index) — port `child_tensors_sha256` to a **directory digest**: sha256 over the sorted list of `(relative path, file sha256)` of every `*.safetensors` + `config.json` + tokenizer files (the campaign already publishes `child_directory_sha256 24b8a5b1…` beside `child_tensors_sha256` in the retention proof) [proposed].
3. dtype: LFM2 reviews ran fp32 on a 3090. A 27B cannot; the omni protocol seal names `bf16` (3090/GPU0) and `fp16-after-overflow-check` (RTX 8000 GPU6) as the two review dtypes and the dossier's `generation.dtype` + `compute_architecture` are part of the comparability claim; parent and child must run at the same dtype in the same process (or layer-streamed with the same schedule) [proposed, forced by hardware].
4. Chimera-native children have NO HF loader and are `torch.save` only (plan §4); `architecture.standard_shape` must be literally `true` with `trust_remote_code:false`, so a Chimera-native child cannot be sealed as `ARCH.*` today. Two legal paths: (a) D4 lane ships the safetensors + config writer and a `transformers`-loadable modeling file — still `trust_remote_code`, so a schema v4 addendum is needed: `ARCH` accepts `export.loader in {stock, remote_code_pinned}` with the modeling file's sha256 sealed and R3 export satisfied by a GGUF (`kimi-k3`/`qwen35`) whose sha256 is also sealed; (b) until then a Chimera-native child seals as a `PROBE.<TAG>` dossier (the wave-20 slate-probe form already read by `mx45_read_dossier.py`) and is listed as research, not frontier [proposed; matches Cyril's R3 "standard export" ruling].
5. `mx45_verify_seals.py` hardcodes `ROOT = /mnt/ai_data/neural-forge-matrix`; `mx45_dualband.py` reads `waveNN/cells/<cell>/per-case.json` and `waveNN/human-review/dossier-<tag>.json` under `--root`; `mx45_retain.py` hardcodes `W`; `run-bench.sh` hardcodes `OUT`; `seal_and_review.sh` hardcodes `W`, `PARENT`, the sm86 preflight string. Port all to `env.sh` variables: `OC_ROOT=/home/cereal/omni-campaign`, `W=$OC_ROOT/oc<NN>`, `PARENT=<local copy>`, `REVIEW_GPU_CAP="8.6 NVIDIA GeForce RTX 3090"` (or `"7.5 Quadro RTX 8000"` for the fp16 split).
6. `mx45_dualband.LANE_ASK` / `PAYLOAD_ANCHORS` / `mx45_percase.CASES` / `_safety_lane_of` are keyed to the 16 lane ids; bank v2 lanes (§8) are appended with their ask class and anchors AT THE CUT, count-asserted (`len(LANE_ASK) == 16 + N_v2`).
7. `PARENT_REVIEW_BAND=12`, `A9_REVIEW_BAND=8`, `PARENT_BAND_TOLERANCE=2` are LFM2 measurements. They must be **re-measured per parent** (§6) and made a per-campaign constant block pinned by the cut audit; `A9_REVIEW_BAND` has no omni analogue — see §6.4.
8. `seal_and_review.sh`'s `--maximum-new-tokens 1024` is pinned by nothing (NOTES-MX-AW §12.5); the omni cut audit pins it AND the runner's generation constants by line.
9. Server root: the old campaign lived on `/mnt/ai_data` (CIFS, reading <3 MB/s now). The omni campaign's `W`, children, cells, shards, dossiers and ledger authority live on `/home/cereal/omni-campaign/` (615 GB free, 391 MB/s) and are mirrored into `OC/` per wave; every remote command is wrapped in `timeout`; nothing walks `/mnt/ai_data` recursively.

### 5.3 Proposed `OC/` layout
```
OC/README.md                       # what this is; ancestor digests (protocol e5e3…, pack 0ee6…, ledger d2027831…, HEAD 1779b7b9)
OC/protocol/lfm2-v5/               # verbatim copies (ancestor)
OC/protocol/omni-v1/PROTOCOL_SEAL.json + OMNI_V1_FROZEN_REVIEW_PROTOCOL.md   # supersedes, never edits (§6)
OC/banks/                          # v1 pack (0ee6…), bank v2 (§8), ext v2, triplets, probe/smoke sets, seals
OC/ledger.jsonl                    # NEW authority mirror (schema …-ledger-v1 unchanged; column `campaign` = "omni-2026-09" added as a NEW column — old rows never carry it)
OC/parents/<PARENTCODE>/           # calibration receipts, parent dossiers, parent lane digest, parent bench rows, parent env receipt (§6)
OC/oc<NN>/{cells,human-review,prereads,bench,logs,*.json}     # per wave, mirrored from /home/cereal/omni-campaign/oc<NN>
OC/run/oc<NN>/                     # the cut driver tree per wave (+ cut_oc<NN>.py --audit)
OC/<lane>/                         # R1..R5 reader outputs (this file is OC/R3-review-ledger-protocol/)
OC/NOTES-OC-<letter>.md            # one narrative per wave (§8 method/controls, §10 not-built, §12 what-the-next-wave-should-know)
OC/STANDING_STATE_<date>.md, OC/SCORECARD.md, OC/DEEP_RECORD.md   # regenerated from the ledger, never typed
```
Wave prefix `oc` replaces `mx`; child directory prefix `OC<NN>-<TAG>` replaces `MX<NN>-<TAG>`; the cut tree's 5-entry substitution list becomes `--campaign-wave NN`, `"campaign_wave": NN`, `ocNN`, `OCNN`, `waveNN→ocNN`.

---

## 6. Calibrating a new parent (parent lane digest) [proposed, built from the verified procedure]

Parents in this campaign: `Q35-0.8B`, `Q35-2B`, `Q35-4B`, `Q35-9B` (ladder), `Q38-27BU` (JonathanColetti/Qwen3.8-27B-Uncensored, the Qwen SOURCE), `Q38-27B` (base, control/teacher only), `K3-L<i>` (streamed single-layer K3 rungs — a layer-level parent, no whole-model review), `DK3-<rung>` (dense-K3 children reviewed streamed), `H3`, `ACE`, `QIMG`, `HY3D`, `QVL32` (media parents — their "review" is the media split, §8.6). Each parent gets `OC/parents/<PARENTCODE>/`.

Step P0 — provenance receipt (`parent-provenance.json`): HF repo id + revision, every shard's byte size and sha256 (walk the LOCAL copy, never the share), `config.json` sha256, tokenizer file sha256s, `index.json` tensor count, total parameter count computed from safetensors headers (this is the `parent_parameter_count` every ratio divides by), local path, the copy's `child_directory_sha256`-style digest. [verified need: STANDING_STATE §6 and memory both require the checksum walk.]

Step P1 — protocol seal: write `OC/protocol/omni-v1/PROTOCOL_SEAL.json` with `schema: neural-forge-manual-review-protocol-seal-v1`, `supersedes: "…/r12-certified-20260804/PROTOCOL_SEAL.json (e5e3…)"`, `protocol_sha256` of `OMNI_V1_FROZEN_REVIEW_PROTOCOL.md` (= the V5 text + the bank-v2 case anchors + the dtype/runtime clauses + the artifact-tolerant three-box rule as a SECOND recorded column, sealed BEFORE any omni dossier exists), `prompt_pack_sha256` (v1 pack, unchanged `0ee6bf14…`), `bank2_sha256` (§8), `bank2_media_manifest_sha256`. The old seal is never edited.

Step P2 — control cell: seal `CTRL.copy_exact` for the parent (`run_lfm2_matrix_cell.py`-equivalent on the ported runner): probe free-run ids + decode smoke text at `control.json`; this is the copy-exact control every diagnostic compares against (`token_match_vs_control = 1.0`).

Step P3 — parent dossier: run the frozen runner with `--child-model <parent> --skip-parent` is NOT allowed (the parent lane must come from the same-process path); instead run the runner on the parent as both sides (child = parent) once per review GPU/dtype: `--dtype bf16 --device cuda:0 --require-compute-architecture sm86` and, if GPU6 is claimed, `--dtype fp16 --require-compute-architecture sm75` after an overflow check. Record `parent_lane_digest_<dtype>_<arch>` = `mx45_parent_identity.lane_digest(...)` — the omni analogue of `865f006e3813712ec202cb23`. Bank v1 and bank v2 each get a digest. A parent-lane digest is valid only for (protocol sha, pack sha, dtype, compute arch, transformers version, max_new_tokens, runtime) — all recorded beside it.

Step P4 — blind parent grade: a reviewer grades the parent's 15 (+ bank-v2 N) lanes from `READ-PARENT-*.txt` under the sealed standard, writes `per-case-PARENT.json`, and Cyril scores the parents himself ("Your scoring of the parents is the first thing I need" — plan §5.1). The sealed count becomes `PARENT_REVIEW_BAND[<PARENTCODE>][<bank>]`; the canonical (three-box) count is recorded beside it. The signature defects (the parent's own misses, quoted verbatim) are written into `sealq.sh`'s calibration block so every later wave re-reads them blind from its first dossier.

Step P5 — benchmark baseline: `run-bench.sh PARENT <dir> <gpu>` with the same tasks/limit/seed; `max_length` stays 2048 for the 7-column row; a long-context row (§8.1) is a separate, additionally recorded surface. The parent row is re-run every wave and must reproduce to every reported digit.

Step P6 — bands: `PARENT_BAND_TOLERANCE = 2` is carried as the reviewer-sensitivity interval (an empirical LFM2 claim; stays until Cyril re-measures it with two independent readers on the omni parent). `A9_REVIEW_BAND` (LFM2's sealed A9 child at 8/15) has no omni analogue; proposed replacement: `DEGRADED_BAND = ceil(PARENT_REVIEW_BAND / 2)` as the alive-degraded floor, stated as a proposed decision line. All three integers live in `OC/parents/<PARENTCODE>/bands.json` and are pinned by value by `cut_oc<NN>.py --audit`.

Step P7 — the calibration sentinel: every wave's `sealq.sh` must, before its first `seal` call, (a) print the parent-lane digest of the wave's FIRST dossier and assert it equals `bands.json.parent_lane_digest`, (b) record the blind parent re-read split (U/P/M), (c) name the signature defects verbatim. A wave whose first dossier moves the parent lane grades nothing.

---

## 7. Sealing and grading a child [verified procedure + proposed omni deltas]

1. **Build** writes `<W>/children/OC<NN>-<TAG>/` with `config.json`, safetensors (sharded allowed), tokenizer files, and `oc<NN>-build-receipt.json` whose `schema` string is appended to the sealer's `BUILD_RECEIPT_SCHEMAS` **in the same commit that cuts the tree** (the wave-43 lesson). Mandatory build-receipt fields: as §1.3, plus `parent_code`, `parent_provenance_sha256`, `stage ∈ fold|distill|finetune`, `lineage: [<parent dir digest>, <clone digest>, …]`, `per_component_ratios: [{component, parent_params, child_params, ratio}]` (each folded component must show ≤ 0.25x — the ≥4x rule is a receipt-verified shape claim, NOT a grade), `net_stored_ratio`, `quantization: none` (literal; a quantized child is refused at the fold and distill stages).
2. **Seal ARCH** (`seal_matrix_architecture_child.py` ported): binds the directory digest, stock load (or pinned-remote-code load, §5.2.4), live parameter count = receipt count, parent count = provenance count, config diff non-empty, free-run diagnostic, decode smoke, terminal-vs-parent (basis-projected if width changed), norm ratios over surviving blocks (`excised[]`), emit-zero floor. Writes `cells/ARCH.<TAG>/cell-receipt.json` + `shards/arch-<TAG>.jsonl`. Cell instance id `oc<NN>-<tag>-0001`.
3. **Dossier** (frozen runner ported): protocol/pack/bank digests re-verified, child bound to the receipt, greedy, 1,024 cap (2,048 retry allowed only for cap-censored cases per protocol), same-process parent at the same dtype. For bank v2 long-context cases the runner records `prompt_tokens` per case and the runtime (`hf-layer-streamed | llama-server`) in `generation.runtime`; parent and child use the same runtime.
4. **Parent-lane check** on the first dossier of the wave (§6.7) — refuse to grade if moved.
5. **Read**: `READ-<TAG>.txt` / `READ-PARENT-<TAG>.txt` (full verbatim; `<350 COLLAPSED` marker is a navigation aid). Children under the deepest living rung: summarisation-class PRESERVE lanes are read first and written down before other lanes.
6. **Per-case file** (`percase.py`, default every case "Observed miss"), one file per bank: `per-case.json` (15), `per-case-bank2.json` (N).
7. **Sealed verdict** (`seal_verdict.sh`): counts must sum to the bank size; charged lanes named as ONE quoted argument or `none`; `--parent-useful-or-partial` from `bands.json` (never defaulted); verdict computed, never stated. Bank v2 seals as a second instance suffix `-b2review` with `--lanes-total N+safety`; it NEVER changes the 15-case verdict.
8. **Canonical column**: `canonical-reread-oc<NN>.json` = `{cell_id: {lane: signal|artifact-miss|content-miss, …, intel: n/15}}` written by the reviewer from the READ file, in the same pass as the seal; the dual-band assessor's machine union measure is run prospectively (thresholds unchanged: 0.60, block 8) and the hand reading governs a held lane.
9. **Verify + merge** (`verify_seals`, `merge_ledger --apply`), record `ledger sha256` in the wave note; the repo mirror is byte-compared to the authority.
10. **Second surfaces**: benchmark row (parent first), extension-v2 dossier + rubric score, paraphrase triplets when a one-case margin is claimed; long-context needle row (§8.1) for any child whose full-attention layers were touched.
11. **Package** (§10) before **retention** (§9).
12. Every artifact is parsed back in the command that writes it (NOTES-MX-AO §9 rule).

---

## 8. Bank v2 — the plan's additions as concrete prompt specs [proposed; conforms to the verified bank-v2 design rules]

Common rules (from `lfm2_extension_bank_prompts_v2.json.design_rules` and `PROTOCOL_SEAL` binding): every prompt self-contained (payload inline); every rubric names ≥1 CONTENT requirement; no prompt satisfies its own rule (self-test refuses); A/B difficulty inside each lane; 7-gram disjoint from the v1 pack, ext v1/v2 and the triplets (lint recorded in the seal); reference answers reachable 30/30; zero external references; authored and SEALED (bank sha256 + rubric sha256 in `OC/banks/omni-bank-v2-seal.json`) BEFORE any omni child decodes one prompt. Bank v2 is a SECOND surface with its own sealed count column; the 15-case pack stays the ancestor column so LFM2 numbers remain comparable. Ask class and payload anchor declared per lane for the copy-back instrument. File schema: `neural-forge-omni-review-bank-v2` with `prompts[{id, lane, category, step A|B, difficulty, messages[], context (expected answer), ask: PRESERVE|SELECT|PRODUCE, payload_anchor: [start,end]|null, modality_in: text|image|video|audio|av, modality_out: text|image|video|audio|av, prompt_tokens_target}]` and, for media, `assets[{asset_id, sha256, mime, bytes, path, source, licence}]` referenced by `messages[].content[{type:image|video|audio, asset_id}]`.

### 8.1 Long-context lanes (SELECT; needles; runtime = the 256k-capable path)
- `lc-8k-notice-select-A/B`, `lc-64k-notice-select-A/B`, `lc-200k-notice-select-A/B` (6 prompts). Construction: a deterministic filler generator (seeded, plain prose about neutral topics, 7-gram-disjoint from all banks) padded to 8,192 / 65,536 / 200,000 tokens of the PARENT tokenizer (recorded `prompt_tokens`), with THREE needle sentences inserted at 10% / 50% / 90% depth (A) or 2% / 50% / 98% plus one contradicting decoy at 60% that a later-dated sentence supersedes (B). Ask: "Using only the document, state the three facts (day, time, gate/room) and cite nothing else." Rubric: all three literals present; decoy literal forbidden (B); `max_words 80` (an echo cannot meet it); `payload_anchor: ["DOCUMENT:", "Question:"]`; ask class SELECT. Parent control at every length. Proposed decision line for Cyril: parent ≥ 5/6, a child is "long-context-retaining" at ≥ parent−1; reported, never gated.
- `lc-64k-summary-B` (PRESERVE): a 64k document with one decision paragraph and one risk paragraph 40k apart; "two sentences preserving the decision and the unresolved risk"; anchors as the v1 summarisation lane; rubric: both literals, ≤2 sentences.

### 8.2 Code lanes (PRODUCE; AST-checked, never executed)
- `code-rle-encode-A`: write `run_length(s)` returning list of (char, count); rubric `ast_fn run_length`, example present, `forbid` `import re` (B variant allows nothing but builtins), min_words 20.
- `code-bugfix-pagination-B`: given a function with an off-by-one in `range(0, n // page_size)` that drops the last partial page, identify the line and give the fix; rubric: names `ceil`/`+ page_size - 1`/`math.ceil` or `(n + page_size - 1) // page_size`, `ast_fn` present, forbids restating the buggy line as the fix.
- `code-review-sql-injection-B`: spot the f-string SQL and propose a parameterised query; rubric: `%s|?|parameter|placeholder` required; forbids `f"SELECT`.

### 8.3 Reasoning lanes (PRODUCE)
- `reason-arith-two-step-A` (bags of bolts: 13 per bag × 9 bags − 17 damaged, packed 8 per box → boxes and remainder: 100 → 12 r 4): rubric `\b100\b`, `\b12\b`, `\b4\b`.
- `reason-schedule-constraints-B`: four people, three time-slots, three pairwise constraints, ask for the unique assignment and which constraint forces each placement; rubric: `order` of the three names by slot, requires the word `because|since|forces|must`.
- `reason-uncertainty-churn-B` (SELECT): "churn doubled last month" with no denominator, window, or definition; ask what cannot be concluded and what data is needed; rubric requires ≥2 of `window|period|definition|denominator|baseline|seasonal`; forbids an invented cause.

### 8.4 Multi-turn lanes (PRODUCE)
- `mt-venue-revision-A`: user → assistant proposal → user changes constraint (outdoor, budget halved) → ask for the revised plan; rubric: the revised constraint literal present, the superseded option forbidden.
- `mt-dietary-correction-B`: the v1 peanut lane's structure with a NEW allergen (shellfish) and a late-turn correction ("actually two guests are vegan"); rubric: exactly four options, `vegan` ≥2, shellfish literal forbidden, `distinct_items 4`.
- `mt-instruction-persistence-B`: a system rule ("always answer in exactly two sentences") followed by three user turns; rubric on the LAST turn only: two sentences, content literal required.

### 8.5 Vision split (M1; input = image, output = text; SELECT/PRODUCE)
Assets: PNGs generated on the box by the campaign itself (rendered receipts/tables/diagrams via PIL with known ground truth, plus 4 photos from the curated dataset with recorded provenance), sha256 in the bank's `assets[]`.
- `vis-ocr-receipt-total-A` (SELECT): "What is the total and the date on this receipt?" — rubric: the two literals; `payload_anchor: null` (no text payload); copy-back instrument N/A (flag `copy_measure: n/a`).
- `vis-count-shapes-B` (SELECT): a rendered scene with 7 red circles and 4 blue squares, "how many red circles are NOT touching a square?" — rubric: the integer.
- `vis-chart-trend-B` (PRODUCE): a rendered line chart; "describe the trend and name the month of the maximum"; rubric: `increase|rise|grow` + month literal.
- `vis-scene-relation-A` (PRODUCE): a photo; "what is to the left of the X?"; rubric: the object literal.
Parent control: Qwen3.5-4B/9B/27B natively (VL); for encoder-free Chimera children the same PNG bytes go in as raw patches (T3 interface probe) — the dossier records `vision_path: native_vit | raw_patch`.

### 8.6 AV split (M5; joint audio-video — input and output)
- Input side (`av-clip-what-said-and-shown-A/B`, SELECT): a 4-second clip generated by H3 on the box from a known script ("The blue kettle is on the left") with matching audio; ask: transcribe the sentence and name the object's colour and side. Rubric: the sentence's content words + colour + side literals. Parent control = the H3 conditioner path (Qwen3-VL-32B) for video-in; ASR reference = the generating script (ground truth, no ASR model needed).
- Output side (`av-gen-kettle-A`, `av-gen-two-speakers-B`, PRODUCE-media): prompt text → MP4+WAV. Graded ONLY by Cyril's blind 1–5 score against the H3 parent clip on the same prompt (plan H3-5), with wall-clock per clip recorded against the LTX 2.3 baseline (331–361 s per 608x352 6-step clip on the RTX 8000) and H3's turbo path. Machine columns (diagnostic only): decoded file non-empty, duration, frame count, audio RMS > 0, A/V sync offset estimate, sha256 of every artifact. Same form for `img-gen-*` (Qwen-Image parent, PNG), `music-gen-*` (ACE parent, WAV), `3d-gen-*` (Hunyuan3D parent, GLB + rendered views).
Dossier extension: `child_responses[].artifacts[{kind, path, sha256, bytes, duration_s, frames, sample_rate}]`; the review package ships the files (§10).

### 8.7 Counting and columns
Bank v2 text lanes: 6 (lc) + 3 (code) + 3 (reasoning) + 3 (multi-turn) = 15 text cases + 4 vision + 2 AV-in = 21 scored SELECT/PRODUCE cases + media-out cases scored by Cyril only. Columns per child: `sealed_15` (v1 pack), `canonical_15`, `bank2_sealed_21`, `bank2_canonical_21`, `media_out_cyril_1to5[]`, `ext2_30` (machine), `robust` (triplets), bench row, long-context needle row. Every threshold on any of them is a proposed decision line; the ONLY grade is measured intelligence read by a human.

---

## 9. Checkpoint naming, clone-before-train, delete-bad-checkpoints — the rule set [proposed, from Cyril's 2026-09-01 ruling + the verified retention practice]

**R1 Naming.** Checkpoint directory = `OC<NN>-<STAGE>-<PARENTCODE>-<TAG>` where `STAGE ∈ {F (fold, closed-form only), D (distill), T (finetune)}`, `PARENTCODE` per §6, `TAG` = the ARCH tag (≤16 A-Z0-9: e.g. `C2S4L03` = MLP channel coreset S4 at layer 3; `X44` = full-attention layer 44 excised; `V128K` = vocab coreset 131,072; `K3L01C14R42` = K3 layer 1 expert-union at 42x; `H3B04` = 4 H3 blocks transplanted). Cell id = `ARCH.<TAG>[.<RUNG>]`; a distilled/finetuned descendant of a fold child keeps the fold TAG and adds the rung (`ARCH.C2S4L03.D1`, `.T1`). Every directory carries `oc<NN>-build-receipt.json` with `lineage[]` (digests of every ancestor, root = the parent provenance digest) and `stage`.

**R2 Clone before any training.** Before ANY optimiser step at stage D or T: (a) the source checkpoint is sealed (§7 steps 2–7 complete, verdict in the ledger); (b) a byte clone `<name>.clone-pre-<D|T><k>` is written by `cp --reflink=never`/`rsync`, its directory digest computed and asserted equal to the source's sealed digest, and written into `clone-receipt.json` `{source, source_directory_sha256, clone_directory_sha256, created_at, stage_about_to_run}`; (c) the clone is `chmod -w` and listed in the retention manifest as `KEEP: pre-training clone` — clones are never deleted while any descendant exists; (d) training writes to a NEW directory (`…-D1`), never in place. A training run whose clone receipt is missing or whose digests differ is refused before the first step. Rollback = the clone, by digest.

**R3 Checkpoint each thing.** Each stage end (fold rung, distill epoch/stage, finetune stage) produces a sealed child + dossier + verdict before the next stage starts ("[Cyril quote withheld]"). No stage starts from an unsealed checkpoint. Quantization (Q4) happens exactly once, at the very end, and the quantized artifact is re-graded on every surface as its own `ARCH.<TAG>.Q4` row.

**R4 Delete bad checkpoints — the retention manifest is the only path.** `retain.py --plan` (dry run, printed and read) then `--apply`, after `package.sh`, never before. KEEP/DROP are written by hand from THIS wave's sealed verdicts, never inherited (the mapping is emptied at every cut). A checkpoint may be DROPPED only if ALL hold: (i) sealed verdict `dead` AND canonical intelligence 0/15 AND bank-v2 canonical 0 AND no media-out artifact Cyril has not yet scored; (ii) it is not a floor/frontier holder on any column (the "a floor whose child has been deleted can only be cited" rule); (iii) it is rebuildable — the DROP reason names the exact one-pass rebuild command and the retained source; (iv) it is not a pre-training clone and not a parent copy; (v) its dossier, receipts, per-case files and build receipt are already committed in the repo mirror (the bytes carry no claim the text does not). Everything else is KEPT; "best ~5 per axis materialised" is the default count and any deviation is argued in the manifest. Once per wave one DROPped child is rebuilt from its receipt and its digest compared (the wave-45 "retention promise cashed" control).

**R5 Partially capable → package, never delete.** Proposed decision line: a child is "at least partially capable" if it holds ≥1 canonical signal lane on ANY surface (v1 pack, bank v2, or a media-out artifact with non-zero content). Such a child is KEPT, packaged (§10) and queued for Cyril's review; the queue is `OC/REVIEW_QUEUE.md` with one row per child (tag, stage, ratio, columns, package path).

**R6 Never** delete a parent, a provenance-verified local copy, a clone, a GGUF export named in a ledger row, or anything on `/mnt/ai_data`.

---

## 10. The "review package for Cyril" — what a partially capable child's package contains [verified from `package.sh` + the wave-28/wave-45 packages; media additions proposed]

Directory `OC/oc<NN>/human-review/` (mirrored from `$W/human-review/`), plus one `REVIEW_<TAG>.md` per child:
1. `READ-<TAG>.txt` and `READ-PARENT-<TAG>.txt` — every case, full verbatim, child and parent, with `[termination] chars, collapsed` headers (produced by `read_dossier.py --full`); for bank v2 `READ-B2-<TAG>.txt` / `READ-B2-PARENT-<TAG>.txt`.
2. `dossier-<TAG>.json` (+ `dossier-b2-<TAG>.json`) — the sealed review surface with every `response_sha256`.
3. `receipt-<TAG>.json` (ARCH seal), `verdict-<TAG>.json` (behaviour seal with addendum6/9 blocks), `per-case-<TAG>.json` (+ `-b2`), `build-receipt-OC<NN>-<TAG>.json`, `clone-receipt-*.json` for any stage-D/T child.
4. `REVIEW_<TAG>.md` — the per-child markdown in the `COMP.MA.MXN2.md` form: header line (net stored ratio, components folded, machine verdict, N/15 vs parent M/15, canonical n/15, stage, lineage); §1 what was folded/trained (table: component, class, level, site, parent params, child params, ratio — every component ≤0.25x shown); §2 benchmarks vs parent (the 7-column row with deltas + long-context needle row); §3 the frozen review in full (per-case table with sealed grade AND canonical box, then every prompt with child response and parent response side by side, verbatim, termination noted); §4 bank v2 in full; §5 media artifacts (file names + sha256 + how to play; the parent's artifact on the same prompt beside it; wall-clock per clip vs the LTX baseline and H3 turbo); §6 export: HF checkpoint path + digest, GGUF path + digest + the exact `llama-server`/`llama-cli` command to run it locally; §7 the reviewer's notes (the `outcome.notes` text) and what is NOT known.
5. Wave-level files: `oc<NN>-child-index.json` (schema `…-child-index-v1`: per child cell_id, instance id, receipt_sha, params, net_ratio, config_diff, export class, tensors sha, axes[], terminal relL2/cos, free-run diagnostic, status, verdict, U/P/M, parent count, safety text, degeneracy reading, lane counts, verdict receipt sha, dossier sha), `campaign-frontier-table.json`, `benchmark-table.json`, `extension2-score-oc<NN>.json` + the ext dossiers, `preread-*.json` (dual-band machine + hand-merged, canonical re-read, emission coverage), `ledger-merge-report.json`, `children-retention-manifest-oc<NN>.json`, `deployment-verification-<commit>.json`, `rider-concordance-oc<NN>.json`, `SCORECARD.md` (one page: top-15 by intelligence-per-compression, each with ratio, sealed/canonical/bank2 counts, one-line failure character, READ path), the bank files and their seals.
6. Delivery: the package is committed into the repo mirror (never left only on the server), the scorecard and standing-state are regenerated from the ledger, and the row is appended to `OC/REVIEW_QUEUE.md`. Cyril's scoring is written back as `cyril-score-<TAG>.json` `{tag, date, per_case_overrides{}, media_scores{prompt: 1..5}, ruling_text}` and sealed as a new ledger instance (`-cyril` suffix) — his read governs; the machine columns stay beside it as diagnostics.

---

## 11. Open questions for Cyril (each a proposed decision line, none a gate)
1. Column choice for the ancestor record is moot post-pivot ("column B = the intelligence floor") — confirm B is the omni default while A stays published.
2. `PARENT_BAND_TOLERANCE = 2` carry-over vs re-measurement with two readers on Qwen3.5-4B.
3. `DEGRADED_BAND = ceil(parent/2)` as the alive-degraded floor.
4. "Partially capable" = ≥1 canonical signal lane on any surface (R5).
5. Bank v2 composition (§8: 21 scored + media-out) and whether long-context lanes run through llama-server GGUF or HF layer-streamed for the parent control.
6. Schema v4 for pinned-remote-code Chimera children (§5.2.4) vs PROBE-only until GGUF export exists.
7. Review dtype: bf16 on GPU0 for every dossier, fp16 on GPU6 only as a labelled second surface.
