# Run 0 held-out text CE drift, 2026-11-05

Side session for the Omni lead. This was read-only on the live run: nothing in `run/`, `maint/`, the autopilot or the mixer was touched.
- **Server work:** two capped CPU passes, both single-threaded at nice 19 with OOM score +500. They were wdelta.py (147 s, ionice idle) and heldout_by_type.py (seconds).
- **Review:** an independent adversarial review checked every claim. Its corrections are folded in and marked "(review)".

## Answer

**The rise is not overfitting.** It is concentrated in the part of the held-out set that the run has stopped seeing, and part of that it is forbidden to train on.

The 57 held-out texts come from the sealed r1 registry corpus and fall into four groups:
- **22 agent-pipeline texts:** Omnicoder "You are a teacher model..." prompts, the generated "2026 pass-N" notes, and agent workflow and tool artifacts.
- **13 natural-prose texts.**
- **13 public texts** in chat form or inside pipeline wrappers.
- **9 texts of other kinds:** reasoning traces and synthetic haystack records.

What moved, measured against the 16 checkpoints of the 600-1100 plateau with checkpoint-to-checkpoint noise:
- **Agent-pipeline types:** up +0.17 to +0.33 (z 2.4-3.2). They carry all of the rise against the plateau.
- **Natural prose:** at its plateau level (+0.01, z 0.1).
- **Haystack:** flat (z 0.0).
- **Reasoning:** improved (-0.14).

Against the best stretch (850-1100), natural prose is +0.09 (z 3.2). It gave back the gain it made while pdf-corpus-docs and fineweb-edu-docs carried CE. v36's Gutenberg stories stopped that.

The registry and agent-pipeline exposure left the mix in steps:
- k3, about 6% of positions, was cut to 1% at 950 and switched off at v26 (1110).
- The authorship allowlist v8 cut the teacher-cached registry text (text_clean_v1) from 5.8M to 1.1M trainable positions at v24 (1050).
- The natural-document sources left at v24.
- v32 (1200) filtered public-text-chat from 140.9M to 6.0M trainable positions.

**Training text CE did not fall through learning.** It dropped in one step at v24 (2.4 -> 1.0 between steps 1050 and 1057), stayed flat at 0.9-1.1, then jumped to 1.7 when stories entered at 1301. The growing gap is a change in what training CE measures.

**Two amplifiers, both measured, effect sizes unknown:**
- **A constant LR.** WSD with `--steps 1000000` never decays. Trunk FFN motion is mostly step noise. The tied embedding/head table drifts toward the current mix.
- **Head re-serving (review).** Every manifest adoption that changes a bucket restarts its sources' seeded shuffle from the top. Since 1050, at least 28% of rank 0's text and tool rows were re-serves of records already served; puzzle-chat is at 52%.

**A separate symptom, (g) format balance.** In v39, 74% of target turns that follow an empty think block are tool calls. The think probe opens exactly that block, and at 1350 it gets tool calls in 7 of 12 replies.

**Ruled out as the cause:**
- grad-norm spikes;
- loss-weight changes (proven unchanged);
- the trainer code (the onset ran on the same trainer as the flat stretch).

**Not significant:** media share. It was flat over the onset; interference could not be tested without a GPU.

**Proposals:**
- **1450, trainer (1400 stays mix-fair alone):** `run/TRUNK_LR_SCALE` 0.5 -> 0.25. It is the existing lever, it survives a resume, and writing 0.5 rolls it back.
- **Next trainer slot:** carry source cursors across manifest adoptions (code to build).
- **Eval:** report held-out by type, plus a train-distribution held-out.
- **Data (the mix monitor's and the lead's call):** a judged natural-prose CE source, and a fix for the think/tool format coupling.

**Escalations:** the 80/20 media/text split and the architecture are Cyril's call. They are flagged, not proposed.

## 1. What moved, and when

### 1.1 The onset, with honest noise

Run-0 noise moves all 57 texts together at each checkpoint, so per-text standard errors understate it (review). For example, the single step 1145->1146 moved prose -0.035 and the pipeline prompts +0.066. The robust comparison is between long-run means of "all", which is the mean of all 57 texts:

| Stretch | Mean "all" | Note |
|---|---|---|
| 850-1100 (6 checkpoints) | 3.499 | checkpoint sd 0.033 |
| 1250-1350 (4 checkpoints) | 3.607 | **+0.108, z 3.2** |

- **Paired across texts:** 1100 -> 1350 is +0.170 (53/57 up) and 600 -> 1100 is -0.023 ± 0.037. Those standard errors are too small, for the reason above.
- **Earlier swings were as large:** 50-step moves before 1100 ranged from -0.094 (807->850) to +0.064 (719->750). The first rising windows, +0.042 (1110->1145) and +0.048 (1150->1200), cannot be told apart from them. The onset lies somewhere in 1100-1200.

### 1.2 By text type

The r1 rows carry no source field, so `heldout_by_type.py` classifies each text by its opening. After the review's corrections, the generated notes are told apart from public text inside a "Domain: ..." wrapper, and public code chat and lyrics have their own class.

Per type: the mean over the 600-1100 checkpoints (with the checkpoint sd), the mean over 1250-1350, z, and the paired 1100->1350 delta:

| Type | n | 600-1100 | 1250-1350 | z | 850-1100 -> late (z) | 1100->1350 |
|---|---|---|---|---|---|---|
| pipeline teacher prompts ("You are a teacher model ... Omnicoder") | 8 | 1.808 (sd 0.140) | 2.137 | **+2.4** | +0.185 (+2.3) | +0.297 |
| synthetic "2026 pass-N" notes | 7 | 4.011 (0.102) | 4.266 | **+2.5** | +0.168 (+2.2) | +0.189 |
| agent / tool / registry artifacts (patches, workflow briefs, grounding, Codex wrappers) | 7 | 3.435 (0.052) | 3.604 | **+3.2** | +0.153 (+5.3) | +0.209 |
| natural web/doc prose | 13 | 3.985 (0.073) | 3.996 | +0.1 | **+0.092 (+3.2)** | +0.148 |
| public text inside a "Domain:" wrapper | 6 | 3.536 (0.054) | 3.553 | +0.3 | +0.068 (+1.6) | +0.129 |
| public code chat and lyrics | 3 | 3.265 (0.040) | 3.331 | +1.6 | +0.085 (+5.3) | +0.108 |
| reasoning traces | 4 | 2.449 (0.103) | 2.305 | -1.4 | -0.041 (-1.2) | -0.032 |
| haystack (synthetic "Record N: ...") | 9 | 4.544 (0.137) | 4.546 | 0.0 | +0.081 (+0.7) | +0.179 |

What each comparison says:
- **Against the 600-1100 plateau,** the three agent-pipeline rows (22 texts) account for essentially all of the rise in "all" (+0.098 of +0.096).
- **Against 850-1100,** they account for about half. Natural prose accounts for about 17%, haystack 10%, and the wrapped and public text the rest.
- **The pipeline prompts and haystack are near-duplicate families (review),** so their "8/8" and "8/9" agreement counts are each about one sample. The z-scores above use checkpoint noise, not text counts.

Paired deltas are consistent with this:
- **Natural prose:** -0.18 (13/13) over 600->1100, +0.15 (13/13) over 1100->1350, +0.005 over 1300->1350 (after v36).
- **The pipeline types** rose in every interval, including 600->1100 (+0.31 to +0.33).

### 1.3 Training CE follows the manifests

`text_ce_text_kd`, mean per 50 steps:

| Steps | 701 | 1001 | **1051** | 1151 | 1201 | 1251 | **1301** | 1351 |
|---|---|---|---|---|---|---|---|---|
| CE | 2.45 | 2.16 | **1.04** | 0.90 | 1.02 | 0.89 | **1.71** | 1.51 |

The step at v24 happens within 7 steps (review: 2.4 -> 1.0 between 1050 and 1057). After it, text_kd was mostly puzzles, chess puzzles and chat. At v36 it became mostly stories and reasoning traces. These are level shifts from composition, not learning.

## 2. Ranked causes

### Rank 1: what the held-out set contains, and which of it the mix still covers (hypothesis d plus the mix). Strong evidence.

Registry, agent-pipeline and document exposure by manifest:

| Manifest (adopted at step) | k3 registry rows | text_clean_v1 registry text (trainable positions) | Natural-document CE sources | public-text-chat (trainable positions) | tool |
|---|---|---|---|---|---|
| v19 (807) | ~6% of positions, share 0.08 | 5.84M | pdf-corpus-docs (~14% of positions) | 168M | 11% |
| v21 (950) | share 0.01 | 5.84M | fineweb-edu-docs + pdf-corpus-docs | 168M | 11% |
| v22 (1000) | 0.01 | 5.84M | pdf-corpus-docs | 168M | 11% |
| **v24 (1050)** | 0.001, 62k positions | **1.12M** (allowlist v8) | **none** | 143M | **22.7%** |
| **v26 (1110)** | **off** | 1.12M | none | 141M | 22.7% |
| v28 (1145) | off | 1.12M | none; long-docs-8k at 2%, realised ~0 | 141M | 22.7% |
| **v32 (1200)** | off | 1.06M | long-docs-8k realised ~7.5-8% of tokens | **6.0M** | 22.7% (one source) |
| v36 (1300) | off | 0.93M | Gutenberg stories (13 sources), reasoning-chat | 6.0M | 22.7% |
| v39 (1350) | off | 0.93M | as v36, plus deepmath | 5.7M | 22.7% |

Sources: the manifests, plus the mix monitor's `mixmon/epoch_report.txt` for the position shares.

**What lines up with what:**
- **Agent-pipeline types (22 texts):** their counterpart in training is registry text. It shrank in steps: k3 at 950 and 1110, the authorship allowlist at 1050. These texts rise through every interval from 600 and are now 0.15-0.33 above the plateau.
  - Much of it is text the authorship guard keeps out by rule (Omnicoder prompts, Codex wrappers, generated notes).
  - Its rise is expected forgetting. It is not a regression of the model the run is building.
- **Natural prose:** improved during 807-1100 while pdf/fineweb docs carried CE, gave that back after v24 and v32 removed them, and was flat once v36's stories arrived.
  - **Counter-evidence (review):** long-docs-8k realised about 8% over 1201-1300 and prose still rose. So "similar data in the mix" is not sufficient by itself; 8k-token Wikipedia rows are 1-2% of rows.
- **Reasoning:** improves when reasoning-chat is in the mix (v36, v39).
- **Haystack:** no robust change.

**Caveats (review):**
- **v24 at first helped:** 1050->1100 was -0.062, plausibly from dropping pdf-corpus-docs, which the source audit rated 97% bad. The damage is the slower loss of what those sources also covered.
- **The q38 and k3 KL metrics are means over all supervised positions** (v2_trainer.py:1331). They fell from 0.16-0.23 (700-950) to about 0.02 (1051+) and 0.004 (1351). That shows fewer teacher positions and/or better agreement, so it is not on its own evidence of less KD. The trainable-position counts in the table are the evidence of less supply.
- **v32 (1200) is a second trigger alongside v24.** It cut public-text-chat 23x and left one tool source.

**Why not calibration or eval code (review alternatives):**
- **Sharpening the output distribution** on a narrow mix would raise CE on everything outside it, haystack and prose included. Instead the rise is concentrated in the registry and agent-pipeline types. The direct test is a per-checkpoint temperature fit on the held-out logits; that needs one CPU eval.
- **The evals load the deployed code tree.** An eval-code change would move every type together, and no jump shows at deploy boundaries (1145->1146 +0.004; 1200->1228 -0.015). The direct test is to re-run holdout_ce on step 1100 with today's tree.

### Rank 2: the constant LR (hypothesis b). Moderate evidence; it amplifies Rank 1.

`--schedule wsd --steps 1000000` with no `--cooldown-steps`, so every group stays at its peak:
- trunk family: 5e-5 × trunk_lr_scale 0.5 = 2.5e-5;
- new modules: 8e-4;
- mixers and gates: `--mixer-lr 1e-3`, relative.

The 468 halving touched only the trunk family (FFN matrices, the text rows of the tied table, norms; stage_factors / parameter_groups).

Weight motion (`wdelta.py`; ‖ΔW‖/‖W‖ between checkpoints):

| Group | Params | 1000->1300 | 1300->1350 | Per-step rate ratio, 50 vs 300 steps (√6 = 2.45 for a random walk) |
|---|---|---|---|---|
| trunk FFN | 881M | 3.84% | 1.51% | 2.36 |
| tied table (embed/head) | 510M | 4.60% | 1.31% | 1.71 |
| mixers | 637M | 1.79% | 0.60% | 2.02 |
| residual gates | 3M | 1.16% | 0.32% | 1.65 |
| media bridge / x0 head | 40M | 43.9% | 18.7% | 2.56 |

Reading it:
- **Trunk FFN motion is mostly step noise.** The table moves more directionally, toward the current token mix, which is what moves CE on text the run no longer sees.
- **Caveat (review):** the 300-step window spans five manifests. Changes of direction between manifests produce the same ratio pattern, so the drift/noise split is indicative only.
- **The mixers move at about their configured relative rate,** around 8e-4 per step under a random-walk model. That is less than the FFNs and the table, so a lower mixer or new-module LR is not the lever.
- **The 468 precedent:** after the trunk halving, held-out went from 4.87 to 3.49 within 130 steps. It is confounded by the end of warm-up and the flow ramp. Read it as "held-out is LR-sensitive", not as an effect size.

### Rank 3: head re-serving at manifest adoptions (hypothesis a, reframed; found by the review). The mechanism is proven; its effect on held-out is unmeasured.

How it happens:
- `builder.load(..., buckets=adopted["same"])` restores data state only for buckets a new manifest leaves unchanged.
- A changed bucket starts fresh: every source's `_Cursor` restarts at epoch 0, position 0 of the same seeded permutation (run0_data.py:415-424). The same head records are served again.
- text_kd changed at all six adoptions since v24.
- `served` counters reset exactly at those boundaries (public-text-chat: 279 -> 0 at 1110; 925 -> 0 at 1300).

Re-serving in steps 1051-1357 (rank 0, counted only across restarts with an unchanged record set, so a lower bound):

| Source | Rows served | Distinct, at most | Re-served, at least |
|---|---|---|---|
| puzzle-chat@v2 | 1,699 | 823 | 52% |
| think-text-chat@v1 | 382 | 208 | 46% |
| chess-puzzle-chat@v1 | 1,144 | 687 | 40% |
| public-tool-chat@v3 | 2,312 | 1,585 | 31% |
| public-text-chat@v5 | 2,238 | 1,587 | 29% |
| **text_kd + tool, all sources** | **10,667** | | **≥ 28%** (≤ 40% if filtered record sets kept their order) |

What this means:
- The head of each source has been seen 2-6 times, and the rest of the source never.
- This is repetition that whole-source epoch counts hide. The review corrected my original epoch script: `served` is restored from the checkpoint on a resume, and the rolled-back re-runs of 1201-1229 had been counted.
- By whole-source epochs, only 1.8% of text_kd rows since 1100 come from sources past 1 epoch, and the mix monitor's pass estimates agree (public-text-chat 0.26, puzzle 0.54).
- It plausibly feeds the low, flat training CE and a narrower fit. Its effect on held-out cannot be separated from Rank 1 with the data at hand.

**A side finding for the trainer owner (mix monitor):** the EpochLedger is not saved and restarts at every launch, so the 7-day caps have never bound. At the v39 pace, three groups would run 4.7-4.9 epochs per 7 days against the CE cut of 4.

### Rank 4: media share (hypothesis c). Untested; no change at the onset.

- **Shares were flat until 1300.** Media was 0.538-0.566 of positions over 851-1200 and 0.52 over 1201-1300. It fell to 0.478 after 1300 (review).
- **Media text CE kept improving.** Speech went 4.09 -> 3.69 and image_understand held at about 2.2.
- **How the 80/20 counts:** in positions (mix monitor), media is about 58%, text_kd about 20%, tool about 20% and long 3-5%. So "80/20" means media plus tool against text_kd. Tool is 43% of supervised target turns.
- **Interference** would need per-bucket gradient cosines on a GPU.

### (g) Mode/format balance: explains the think-mode tool calls (strong evidence; not held-out CE)

**How the think probe works.** `text_probe.py --think` places Qwen's empty think block `<think>\n\n</think>\n\n` after the forced decision, then samples.

**What the data teaches.** The mix monitor's `think_tool_scan.py` reads the v39 target turns through the trainer's RecordFilter (`mixmon/think_tool_v39.txt`):

| Turn type | Share of all target turns |
|---|---|
| empty think -> tool call | 20.5% |
| empty think -> answer | 7.4% |
| no think -> tool call | 19.3% |
| no think -> answer | 49.5% |
| reasoning -> answer | 2.6% |
| reasoning -> tool call | 0.8% |

- **74% of turns after an empty think block are tool calls.** Almost all come from public-tool-chat@v3, which is 43% of target turns. Its v32 rebuild gave half its rows an empty think prefix.
- **The data never has a tool call without a tool list, and never nests `<think>`.**

**What the model does.** Think-probe tool calls per 12 replies:

| Step | 950-1200 | 1228 | 1250 | 1260 | 1300 | 1350 |
|---|---|---|---|---|---|---|
| Tool calls | 0-2 | 1 | 3 | 3 | 2 | 7 |

The plain text probe gives 0/12 at every step from 1150. So the empty think block has become a cue for "call a tool".
- **The rate matches the data:** 7 of 12 is about the data's 74% conditional rate.
- **The rise lines up with the manifests.** It begins after v32 brought in public-tool-chat@v3, and jumps after v36, which added swe-windows (empty think -> tool call, 100%) and the terminal reasoning -> tool traces.
- **The malformed parts are model errors.** Invented tools ("rainwaterfall") and nested think tags appear in no training target.

### Ruled out

- **Grad-norm spikes.** 797-806 and 1088-1105 reached up to 6.9e6; every step is clipped to 1.0, and step 1079 was non-finite and skipped. 1100->1110 was +0.005 on all 57 texts (+0.016 on the 48 non-haystack). The rise continued with no spikes after 1105.
- **(e) Code changes.** The onset ran on the same trainer as the flat 950-1100 stretch: the installed v2_trainer.py in `chimera-omni-v2.prev-7e8e9f8` has mtime 11-02 20:48Z (e1f607e), before the 950 relaunch. c8aa0bd went live at 1146, 3eab232 (b06d417 context author, SDPA long rows) at 1200, the DFT spectrum (equal to stft within 3.7e-6) at 1228, and 84b3e58 (OOM order only) after it. None precedes the onset. The eval-code caveat is under Rank 1.
- **(f) Loss weights.** teacher_weights are identical in v19-v39. `--media-ce` and `--teacher-weights` are in training_identity, and every resume since 268 was accepted, so the weights never changed. flow_weight has been 1.0 since step 601.

### The degenerate replies are a separate symptom

"23*23=66" loops, "The sky is blue because it is blue." and "The capital of Australia is the capital of Australia." come from a weak text LM: held-out prose CE is 4.0 nats/token, and answer-style supervision is thin (49.5% of target turns are plain answers, mostly short chat and puzzle turns). Held-out CE does not measure them, and neither does text_stopped (0.67). Only the think-mode tool calls, under (g), have a direct fix.

## 3. Proposals

One change per checkpoint, so each effect can be attributed. 1400 stays mix-fair alone.

### P1 (trainer, 1450): trunk LR scale 0.5 -> 0.25

- **What:** write `run/TRUNK_LR_SCALE` = `{"scale": 0.25, "why": "RUN0_HELDOUT_DRIFT_2026-11-05 P1"}` before the relaunch after the 1450 checkpoint stop.
- **Why this lever:** take_trunk_lr_scale keeps the scale in the state, reads it at a start, and it survives resumes without touching training_identity.
- **Why not the obvious alternatives:** `--lr`, `--mixer-lr`, `--new-module-lr`, `--media-ce`, `--teacher-weights`, `--cooldown-steps`, and `--steps` together with a cooldown are all in training_identity. A resume after changing any of them refuses ("start a new run directory").
- **Expected effect:**
  - Trunk FFN and table step noise and drift rate halve.
  - The forgetting rate of text outside the mix should roughly halve, from about +0.035 to about +0.015 per 50 steps on the agent-pipeline and prose types. A one-time drop of a few hundredths is plausible, not promised.
- **Cost:** slower adaptation to new mixes and media.
- **Watch at 1500 and 1550, against 1450:**
  - held-out by type;
  - per-bucket text CE;
  - speech and image text CE;
  - x0_mse;
  - decision and good-view probes.
- **Rollback:** write `{"scale": 0.5}` at the next stop.
- **Floor:** alarm-min-lr-scale is 0.125, so the top-10 alarm can still halve the scale once more.

### P5 (trainer code, the next trainer slot after P1): carry source cursors across manifest adoptions

- **What:** when a bucket changes, restore each source's `_Cursor` and stats if the source id and its record-set fingerprint (`records`, `rank_records`, filter counts) are unchanged. Restart only sources that are new or changed.
- **Flag:** behind a flag that defaults off and is bit-exact when off, as mix-fair was done. A test should prove that an adopted manifest with one changed source resumes the other sources' cursors.
- **Coordinate with the mix monitor:** mix-fair's credit reset sits on the same adoption path.
- **Expected effect:** the ≥28% re-served head rows become fresh records. Sources reach rows they never have.
- **Rollback:** the flag off.

### P2 (eval; the eval owner's call): split the gauge

- **What:** report held-out by type, as in section 1.2 (`heldout_by_type.py` with `levels=`). Gate on natural prose, wrapped public text, reasoning and haystack, plus a new sealed train-distribution held-out (unseen records of public-text-chat, the stories, reasoning-chat and public-tool-chat).
- **Keep the 22 agent-pipeline texts** as an exclusion probe that is expected to rise.
- **Rollback:** none needed; it adds columns.

### P3 (data; the mix monitor's and the lead's call)

- **Natural prose:** a judged natural web-prose CE source back in text_kd. A candidate is the FineWeb v5 GOOD parts from the K3 judge run.
  - **Evidence:** prose gave back its 850-1100 gain (+0.09, z 3.2) after the doc sources left, and was flat once stories came in.
  - **Limits that apply:** quality over quantity, judge first, never Claude- or Codex-authored targets.
  - **Rollback:** a manifest bump.
- **Think/tool coupling (g); the mix monitor offered to build it:**
  - **First:** render most public-tool-chat tool turns without the think prefix, and add empty think -> answer turns until answers clearly outnumber tool calls after an empty think block.
  - **Second:** add tool-less think-mode negatives ("no tools available -> answer directly").
  - **Low priority:** capping the terminal reasoning -> tool traces (0.8% of turns).
  - **Check:** think-probe tool calls back to 0-2 of 12 within 50-100 steps.
  - **Rollback:** a manifest bump.

### P4 (trainer code; build it, use it only for a release): a state-based cooldown

- **What:** a `run/COOLDOWN` operator file, `{from_step, steps, min_ratio}`, kept in the state like trunk_lr_scale, decaying both families linearly.
- **Why it needs building:** `--cooldown-steps` is in the identity, so today a decay cannot be applied to this run.
- **How to use it:** cut a release candidate, evaluate, then resume the main line from the pre-cooldown checkpoint.
- **Expected effect:** the usual one-time WSD drop, of unknown size here. It is not a drift fix.

### Tests that would settle the open alternatives (the lead's call)

- **Eval code (CPU, eval owner):** re-run holdout_ce on step 1100 with today's code tree.
- **Calibration:** a per-checkpoint temperature fit on held-out logits at 1100 and 1350. If the rise disappears at the best temperature, it is calibration, not content.
- **Direct test of Rank 1 (GPU; costs main-line steps):** a 50-step branch from 1350 with a judged prose source restored, measuring natural prose by type.

### Not recommended

- **Lowering `--mixer-lr` or `--new-module-lr`:** the mixers move at their configured rate and less than the trunk and table.
- **Changing loss weights:** they never changed.
- **Tighter whole-source epoch caps:** whole-source repetition is low. The repetition is at the head, which is P5.

## Escalations (Cyril's call; flagged, not decided)

- **80/20 media/text split:** no evidence at the onset. Media shares were flat until 1300. Interference is untested; there is no GPU.
- **Architecture:** nothing here points at the block layout.

## Method and data

- **Copied read-only to the workstation:**
  - metrics.jsonl (1,499 rows; the last row per step is used);
  - events.jsonl;
  - evals/holdout_*.json, trend.tsv, text_*.json and think_*.json;
  - manifests v19, v21, v22, v24, v26, v28, v32, v36 and v39;
  - the run-0 argv (run/trainer_config.json).
- **Server passes:**
  - `ops/run0/drift/wdelta.py 1000,1300,1350`: capped.sh MemoryHigh 6G, 1 thread, nice 19, ionice -c3, oom_score_adj 500; output `run0/scratch-drift/wdelta.json`.
  - `ops/run0/drift/heldout_by_type.py`: it rebuilds holdout_ce.pick() over the sealed r1 texts.jsonl (eval_split 24, codex_recovery_heldout 24, bank_v2_haystack 9; minimum 1500 characters) and checks the counts against each holdout json.
- **From the mix monitor (Relay 11-05, 05:00Z):** position shares, pass estimates and think/tool turn types (`mixmon/epoch_report.txt`, `mixmon/think_tool_v39.txt`; scripts on claude/run0-mix-monitor-20261104).
- **Adversarial review:** scratch scripts are in the session scratchpad `review/` folder (ho.py, ep2.py, seg.py, man2.py, met.py, media.py).
