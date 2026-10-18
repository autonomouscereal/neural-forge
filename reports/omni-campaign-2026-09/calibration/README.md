# Omni campaign — parent calibration lane

The instrument the omni parents (and later, the fold/distill/finetune children)
are read with. It is the LFM2 matrix campaign's review surface, ported to
`qwen3_5`-class checkpoints and to sharded, bf16, multimodal parents.

**Nothing in this lane grades anything.** It generates verbatim text, puts each
lane in a form box, publishes two columns, and hands Cyril a sheet. Every
threshold is a *proposed decision line*, never a gate.

## Files

| file | what it is |
|---|---|
| `oc_calib.py` | the instrument: canonical bytes, bank loading + refusals, the copy measure, the form boxes, greedy decode, the coherence smoke, the emit-zero floor |
| `oc_run_frozen_bank.py` | runs a bank on one model and writes a dossier (schema `neural-forge-omni-parent-calibration-dossier-v1`) |
| `oc_assess.py` | both dual-band columns + Cyril's markdown scoring sheet |
| `oc_bank_v2_draft.py` | authors, self-tests and digests the **DRAFT** bank v2 |
| `oc_testcards.py` | five deterministic PNG vision cards with exact ground truth (stdlib only) |
| `oc_parent_calibration.py` | the orchestrator: decides whether the parent landed, then runs everything |
| `oc_kernel_probe.py` | prints the v2 linear-attention kernel report for one checkpoint, on the CPU, in about a second; grades nothing, refuses nothing |
| `test_oc_calibration.py` | 77 tests; instrument tests need no torch, torch tests skip without it, and a skip is reported as a skip and never counted as a pass. **76 passed / 1 skipped / 0 failed** on the server, **71 passed / 6 skipped / 0 failed** on the workstation (2026-09-02) |
| `banks/` | the sealed v1 pack, probe and smoke sets (verbatim copies), and the draft bank v2 + its five cards |
| `run_r6.sh` | the server pass that answers what r5 could not: the thinking-off surface, the long-context rungs, the bank-v2 text lanes, and the repeat |
| `run_r7.sh` | the three rungs r6 could not finish (64k, 200k, the REPEAT); refuses to start on a card that is not free, and prints both `lane_digest_validity` tuples plus **REPRODUCTION / NEW ANCHOR** before it prints the digests |
| `out/KERNEL-FIELD-CORRECTION-2026-09-02.md` | why the kernel field in every pre-r7 dossier is not comparable with the current one |

## Digests

| artifact | sha256 |
|---|---|
| `banks/lfm2_semantic_whole_response_prompts_v1.json` | `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b` (the seal's) |
| `banks/matrix_campaign_probe_prompts_v1.json` | `6230f878be9287a9abf4ad129d586fc22346867897858b9f1a4f0b75d1f82f2e` |
| `banks/matrix_campaign_smoke_prompts_v1.json` | `82ebf70a1606ec8a57c4daa31218fe1d376a91f211246c78716615ca66d2ceca` |
| `banks/omni_review_bank_v2_draft.json` | see `banks/omni_review_bank_v2_draft.json.sha256` |

Anchors, after r7:

| lane digest | what it is | validity tuple |
|---|---|---|
| `3a13002f1df1…` | bank v1, template default, 16 lanes | r5 (`9e8dfc05…`, incomplete tuple) **and** r7 REPEAT (`743f2e9c…`, full v2 tuple) — byte-identical, no lane moved |
| `bc1080a6eb24…` | bank v1, `--thinking off` | r6 (`aa75b45b…`, v1 kernel semantics) |
| `c7b355361aef…` | bank v2 64k, `--thinking off` | r7 (`fcad8f89…`, full v2 tuple) |
| `14c070394316…` | bank v2 200k — both lanes `out-of-memory`, counted as nothing | r7 (`b1412ded…`, full v2 tuple) |

**Compare a lane digest only against a dossier whose validity tuple is complete
and v2.** `oc-dossier-Q35-4B-REPEAT.json` is the one to hold the parent to.

## Commands

Tests (workstation — torch tests skip):

```powershell
& "C:/Python314/python.exe" "C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/calibration/test_oc_calibration.py"
```

Tests (server — everything runs):

```bash
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script '
cd /home/cereal/omni-campaign/calibration
timeout 280 /home/cereal/model-surgery-lab/venv/bin/python test_oc_calibration.py
'
```

Re-author + self-test the draft bank v2 (deterministic; the digest must not move):

```powershell
& "C:/Python314/python.exe" oc_bank_v2_draft.py --out banks/omni_review_bank_v2_draft.json --assets-dir banks/assets --v1-pack "C:/Users/cereal/Documents/ai training/neural-forge/research/lfm2_semantic_whole_response_prompts_v1.json" --self-test
```

Which linear-attention kernels a checkpoint actually binds (CPU, ~1 s, no card):

```bash
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script '
cd /home/cereal/omni-campaign/calibration
timeout 250 /home/cereal/model-surgery-lab/venv/bin/python oc_kernel_probe.py   --model-dir /home/cereal/models/Qwen3.5-4B --device cpu --dtype bfloat16   --out out/kernel-probe-v2-Q35-4B.json
'
```

Full parent calibration on the server (bank v1, GPU0, bf16):

```bash
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script '
cd /home/cereal/omni-campaign/calibration
nohup timeout 3000 /home/cereal/model-surgery-lab/venv/bin/python oc_parent_calibration.py \
  --model-dir /home/cereal/models/Qwen3.5-4B --label Q35-4B \
  --download-log /home/cereal/hf-downloads/dl_qwen35_4b_now.log \
  --work /home/cereal/omni-campaign/calibration/out \
  --device cuda:0 --require-compute-architecture sm86 --dtype bfloat16 \
  --bank-v1 banks/lfm2_semantic_whole_response_prompts_v1.json \
  --probe-prompts banks/matrix_campaign_probe_prompts_v1.json \
  --smoke-prompts banks/matrix_campaign_smoke_prompts_v1.json \
  > out/run-Q35-4B.log 2>&1 &
'
```

Add the draft bank's text lanes as a second pass with
`--bank-v2 banks/omni_review_bank_v2_draft.json`. `--text-only` drops ten of the
forty entries and names every one of them in the dossier, under **two different
keys because they are two different admissions**: seven cases whose *prompt*
this runner cannot build (`cases_skipped_unsupported_modality` — five vision,
two AV-in) and three whose *answer* is a media artifact
(`cases_skipped_not_runnable` — `av-gen-kettle-A`, `img-gen-shelf-B`,
`music-gen-two-bar-A`). Those three carry a plain text prompt and the default
`modality_in: text`, so the modality filter alone used to keep them and a text
LLM was asked to "generate a four-second clip"; `test_text_only_over_bank_v2_
keeps_exactly_the_thirty_text_cases` pins the selection at 30. Naming a
`runnable: false` case in `--lanes` is now refused outright.

**The nine long-context lanes are in that 30 and they have never decoded.** Pass
`--bank-v2-lanes` to choose a subset, as the commands below do.

Attach Cyril's reading and get the real columns:

```bash
/home/cereal/model-surgery-lab/venv/bin/python oc_assess.py \
  --dossier out/oc-dossier-Q35-4B.json \
  --bank banks/lfm2_semantic_whole_response_prompts_v1.json \
  --per-case out/per-case-Q35-4B.json \
  --out-json out/assessment-Q35-4B.json \
  --out-markdown out/SCORING-SHEET-Q35-4B.md
```

`per-case-<TAG>.json` is `{"<lane id>": "Useful"|"Partial"|"Observed miss"|"Inconclusive/censored", ...}`
and it must name **every** scored lane — a file that names ten of fifteen is
refused, because the five it does not name would be counted as not-held and that
is a miss the instrument would have invented. Without a per-case file both
columns are published as **upper bounds** and are labelled so.

## What the two columns mean

* **Column A — copy-back counts.** A lane that carries the asked-for facts holds
  the case however it got them. On LFM2 this is echo persistence.
* **Column B — copy-back refused.** Column A minus held lanes that returned the
  prompt's own payload. On LFM2 this is the machine-verified intelligence floor.

**Column B is always column A minus `voided_lanes`, in both branches.** Without a
reading, the column-A bound subtracts only `EMPTY`, `LOOP` and `CAP_RIDE` — the
boxes a machine can prove hold nothing. An ECHO lane stays *inside* column A,
because "this lane is a copy" is a threshold call and the instrument's own
caution says that call may not transfer to a capable parent. An earlier version
dropped ECHO from column A as well, which printed `A 11/15, B 11/15` beside a
list of four voided lanes and produced a "bound" that a legal reading of the same
dossier exceeded by four. That is fixed and pinned by
`test_an_echo_lane_stays_inside_the_column_a_upper_bound`.

The copy measure is `payload_union_coverage >= 0.60` over common runs of >= 8
characters — the LFM2 numbers, unretuned, so an omni column is comparable with an
LFM2 column. Whether that threshold *transfers* to a 4B-class parent is
unmeasured and is a decision for Cyril.

## What has actually been run (2026-09-01)

Server mirror: `/home/cereal/omni-campaign/calibration/`. Artifacts in `out/`
(byte-identical copies pulled back into this directory; digests below are the
same on both sides).

**Qwen3.5-0.8B, GPU0 (RTX 3090, sm86), bf16, frozen 15-case bank + safety lane,
1,024-token cap, 368.6 s.**

| | |
|---|---|
| loaded class | `Qwen3_5ForConditionalGeneration`, `trust_remote_code=False` |
| parameters (live) | 852,985,920 |
| model directory sha256 | `392cee346d9b119b861b3f84bfadcab287db5fa4649fa9013671b90f09055389` (10 files) |
| dossier sha256 | `445efcec5780e7531b030f55360016a69dcc60c0ec057025567f0e6766bd7555` |
| lane digest | `a96e949e8f671fff8d49465aaeac437147f323b47c7237f15b77b806b8952b18` |
| emit-zero floor | passed (min ‖h‖ 119.98, min logit sd 2.157, 6 distinct argmax over 8 probes) |
| decode-path preflight | manual loop == greedy `generate` (**agree**) |
| terminations | 16/16 `complete`; no cap-ride, no loop, no runtime error |
| columns (upper bounds, no reading yet) | A 11/15, B 11/15 |

**The lane digest reproduces.** A second independent run of the same command
(`Q35-0.8B-REPEAT`, 16/16 lanes, same card, same dtype) produced
`a96e949e8f671fff8d49465aaeac437147f323b47c7237f15b77b806b8952b18` again, with
zero lanes moved by `response_sha256`. That is the omni analogue of the LFM2
parent-lane anchor `865f006e…`, established for Q35-0.8B on the tuple recorded in
`lane_digest_validity` (bank `0ee6bf14…`, surface `0b429e4c…`, bfloat16,
`sm86 NVIDIA GeForce RTX 3090`, transformers 5.12.1, 1,024 tokens,
hf-transformers, manual argmax loop). Two runs is two runs, not twenty-three
dossiers across eight waves — but the anchor exists and a moved lane is now
detectable.

**Qwen3.5-4B — provenance only.** Directory digest
`94cb094ac11062b0c44c4527b0be380ad81d249382055ced95e367c5ecf589a2` over 11 files,
**4,539,265,536** parameters, loads as `Qwen3_5ForConditionalGeneration` with
`trust_remote_code=False` in 4.8 s. It has **not** been decoded: GPU0 is held by
another process (`/app/llama-server`, 21.5 GiB of 24), GPU4 and GPU6 are not
claimed, the P40s are sm61 and this torch build refuses them, and CPU is not an
option — `fla.ops.gated_delta_rule.chunk_gated_delta_rule` is a Triton CUDA
kernel, so Qwen3.5 cannot forward on CPU at all. The command above runs
unchanged the moment a >= 10 GiB sm86 card is free.

**Draft bank v2, six text lanes on the 0.8B.** `--text-only` dropped
`vis-count-shapes-B` and named it in the dossier; dossier sha256
`3e9b353c30da9195f25cb55bbc4f81b5c914c2a5e69f70a209ce6864e6d704f5`. Rubrics
fired as designed: `mt-dietary-correction-B` produced 9 items against the asked
4, `know-refuse-fabrication-B` invented an answer instead of declining.

### The one instrument finding worth Cyril's attention

Four of fifteen lanes on the 0.8B were called **ECHO** by the LFM2-calibrated
copy measure (`explanation-seasons` 0.624, `logic-ordering` 0.675,
`writing-concise-email` 0.707, `debugging-off-by-one` 0.926). All four are
PRODUCE lanes, and wave 40's own precedent says a copy can satisfy a PRESERVE ask
and *neither a SELECT nor a PRODUCE one*. So either the model handed the payload
back, or 0.60 — set on a 230M parent that could not restate anything at length —
over-reports on a parent that legitimately quotes the material it was asked to
work on (a correct off-by-one fix **contains** the function it fixes).

**The threshold was not moved.** The four lanes are published as instrument
cautions in the assessment JSON and at the top of the scoring sheet, with their
coverage numbers, so the reading decides. Proposed decision line for Cyril: either
re-cut the `debugging-off-by-one` / `writing-concise-email` payload anchors for a
capable parent, or promote the response-side denominator on PRODUCE lanes — both
are anchor/denominator changes, not threshold changes, and both must be
registered before the pass that uses them.

## Qwen3.5-4B, decoded (2026-09-02)

GPU0 came free and the 4B decoded. **The frozen 15-case bank ran end to end on
the real parent**, which is what the whole lane was built for.

| | |
|---|---|
| model | `/home/cereal/models/Qwen3.5-4B`, dir sha256 `94cb094ac11062b0c44c4527b0be380ad81d249382055ced95e367c5ecf589a2` (11 files) |
| loaded class | `Qwen3_5ForConditionalGeneration`, `trust_remote_code=False`, **4,539,265,536** parameters |
| surface | GPU0 (RTX 3090, sm86), bf16, 1,024-token cap, transformers 5.12.1, template default |
| dossier sha256 | `9e8dfc0580050d557703d97cfacac61e2429a1bbde4e6f78b0ee3b732cb491a7` |
| lane digest | `3a13002f1df108679d401af65ce4201c3db1396421603e9c79edb90a027a2e37` |
| emit-zero floor | **passed** (min ‖h‖ 146.688, min logit sd 1.6936, 7 distinct argmax over 8 probes) |
| decode-path preflight | manual loop == greedy `generate` (**agree**) |
| terminations | **13/16 `censored-at-cap`, 3 `complete`**; smoke 3/3 cap |
| wall time | 3,515.6 s |

### The finding that matters more than the digest

**Thirteen of sixteen lanes never wrote an answer.** They wrote a plan. Case 1
verbatim, first line: `Thinking Process:` — then `1. **Analyze the Request:**`,
drafts, refinements, and the 1,024-token cap. Cases 2 and 3 are the same shape.

The cause is not the model. Qwen3.5's chat template ends on a conditional: if
`enable_thinking` is defined and false it emits a **closed** think block, and
otherwise — *which includes not passing it at all* — it emits an **open** one. So
the V5 protocol's "1,024 new tokens", a number set on LFM2.5-230M, which has no
reasoning mode, buys chain-of-thought on this parent and the answer arrives after
the budget is gone.

Three things follow, and none of them is a grade:

1. **A cap-ride here is still not a failure.** The protocol says so and the
   artifact-tolerant lens says so: a cap-ride without unique content counts
   NOTHING, neither for nor against. What it does mean is that the reading has
   *nothing to read*, so the surface — not the model — has to change.
2. **The switch is now explicit, and it is recorded.** `--thinking
   {template-default,off}`. `template-default` remains the default, because
   silently flipping a review surface is how two incomparable reviews get filed
   under one name. The dossier carries `thinking_report`, which records whether
   the setting actually reached the template rather than whether it was asked
   for.
3. **It is in `lane_digest_validity`**, beside the dtype, for the same reason the
   dtype is: two reviews on two settings are two surfaces, not one surface
   measured twice.

**Proposed decision lines for Cyril** (neither is taken here): review the omni
parents with thinking **on** and a cap large enough to reach the answer — V5
already permits a retry at 2,048 for cap-censored cases, and this parent looks
like it needs more than that — or with thinking **off** at the V5 cap, which
measures the answer and not the deliberation. Whichever is chosen, parent and
child must be decoded on the same setting.

### The second finding: the kernel path is part of the surface

*(Corrected 2026-09-02. The first version of this section, and the field it
described, are refuted below by this lane's own artifacts. Kept visible rather
than rewritten away, because the refutation is the useful part.)*

A Qwen3.5 parent is a hybrid: three quarters of its layers are Gated DeltaNet,
and each such layer binds **four** primitives that transformers resolves silently
at import time — either to an installed kernel or to its own torch
implementation. The two are the same function in exact arithmetic and **not** the
same function in bf16: they accumulate the recurrence in a different order, so a
greedy argmax can land on the other side of a near-tie and one lane's text moves.
That is a surface variable and it belongs beside the dtype.

**What the first version claimed, and why it was wrong.** It read the transformers
log line `The fast path is not available … Falling back to torch implementation`
and concluded that `fla` was absent and the whole recurrence ran in torch, then
encoded that as the rule `fla AND causal-conv1d, else torch-fallback`. Three
artifacts refute it:

* `modeling_qwen3_5.py:217-219` prints that warning off `is_fast_path_available`,
  which is `all()` of **four** callables — while `:420-423` binds those four
  **independently**, each with its own `or torch_…` fallback;
* `out/r6-lc64k-Q35-4B.log:137-152` is a live traceback through
  `modeling_qwen3_5.py:535 → self.chunk_gated_delta_rule →
  fla/ops/gated_delta_rule/chunk.py:567` — fla's Triton chunked kernel
  **executing** on the box whose every r6 log carries that warning on line 50;
* `pip_install.log:86` plus the package mtimes put `fla-core 0.5.2` /
  `flash-linear-attention 0.5.2` on this venv at **2026-09-01 14:25:13 −0600**,
  ten hours *before* r5 started at `00:44:18`.

So on both r5 and r6 the recurrence ran on Triton and the convolution ran in
torch, at the same time, and the field would have written `torch-fallback` into
the validity tuple of both. A field added to make a moved lane **attributable**
was supplying a false attribution.

**What it records now (`kernel_report_schema: neural-forge-omni-kernel-path-v2`).**
`kernel_path` is probed off the loaded model, one entry per primitive:

```
chunk=fla-0.5.2; recurrent=fla-0.5.2; conv-prefill=torch; conv-decode=torch
```

with `bindings` (provider, resolved callable, version, per slot),
`gated_delta_layers_probed`, `bindings_uniform_across_layers`,
`transformers_fast_path_flag` (the `all()` flag the warning is printed off — not
the path), and `packages_present` as the inventory it always was. It is recorded,
never enforced: which kernel path the campaign standardises on is Cyril's call.

**Correction row — the delivered dossiers carry v1 semantics.** The four r6
dossiers (`…-NOTHINK`, `…-b2-…`, `…-lc8k-…`, `…-lc32k-…`) were written by the
inventory rule. In those files `linear_attention_kernels.kernel_path:
"fla-0.5.2"` means *fla present, causal_conv1d absent* — an inventory string, not
a measured path — and they carry no `kernel_report_schema`, no `bindings` and no
`gated_delta_layers_probed`. **A v1 string and a v2 string are not comparable.**
Treat any v1 dossier's kernel element as unmeasured; the next run on an unchanged
box will legitimately show that element moving while nothing about the surface
moved. `out/KERNEL-FIELD-CORRECTION-2026-09-02.md` says the same thing next to
the artifacts, and `lane_digest_validity.linear_attention_kernel_report_schema`
now travels with the string so the two can never be confused again.

**Note on the r5 dossier:** it was produced *before* those fields existed, so
`9e8dfc05…` carries neither the kernel element nor the thinking element. Its
template surface (`template-default`) is established from the template; its
kernel surface is established from the install timestamps above and **not** from
the r5 log, which captured no stderr at all (`grep -c "Loading weights\|UserWarning"
out/run-r5-Q35-4B.log` → `0`).

## r6 — the pass that answers what r5 could not

`./run_r6.sh /home/cereal/models/Qwen3.5-4B Q35-4B cuda:0`, cheapest-and-most-
informative first, each rung its own process so an out-of-memory at the top costs
nothing below it:

1. bank v1 `--thinking off` — the second surface, and the one that produces
   answers to read;
2. the bank-v2 **8k** long-context rung — these nine lanes had never decoded;
3. the twenty-one short bank-v2 text lanes;
4. 32k -> 64k -> 200k, in that order;
5. bank v1 on the template default again — the REPEAT, which is what turns
   `3a13002f...` from a number into an anchor. Last, because it is the most
   expensive arm here and the least urgent.

## r6 results — the two surfaces, side by side (2026-09-02)

Same bank, same parent, same card, same dtype. **One template variable
apart *that was recorded*** — r5's kernel element was never measured (see the
correction above), so read this as the two surfaces as they were run, not as a
clean single-variable ablation.

| | template default (r5) | `--thinking off` (r6) |
|---|---|---|
| dossier sha256 | `9e8dfc05…` | `aa75b45b…` |
| lane digest | `3a13002f…` | `bc1080a6…` |
| terminations | 13 `censored-at-cap`, 3 `complete` | **16 `complete`** |
| wall time | 3,515.6 s | **642.7 s** |
| form boxes (15 scored) | 12 ECHO, 3 CONTENT | 2 ECHO, **13 CONTENT** |
| form boxes (all 16 incl. the charged-not-scored safety lane) | 13 ECHO, 3 CONTENT | 2 ECHO, 14 CONTENT |
| column A (bound) | 15/15 | 15/15 |
| column B (bound) | **3/15** | **13/15** |

Column B is column A minus the ECHO lanes, and on the template default **twelve
of fifteen lanes were called ECHO**, at union coverage 0.63 to 1.00.

That is the third finding, and it is about the instrument rather than the model:
**the LFM2 copy measure does not survive contact with a reasoning trace.** A
think block that opens by restating the task —

> `1. **Analyze the Request:** * User's state: "I had a long day."`

— is, to a measure that counts shared runs of >= 8 characters against the prompt
payload, indistinguishable from handing the prompt back. The threshold was **not
moved**; it is the LFM2 0.60, unretuned, so an omni column stays comparable with
an LFM2 column. What moved is the understanding of when it applies: on a
reasoning surface it reports the deliberation, not the answer. On the same
parent with thinking off, only two lanes read ECHO — `logic-ordering` (0.643)
and `debugging-off-by-one` (0.926), the second being the caution already on
record from the 0.8B, where a correct off-by-one fix *contains* the function it
fixes.

### Bank v2 draft, first decode on a capable parent

* **21 short text lanes** — 20 `complete`, 1 cap-ride; dossier `09fdc0a6…`.
* **long-context 8k** — dossier `76f39ba1…`. All three decoded. The haystacks
  regenerated and matched their pinned digests, and the needles were found:
  `lc-8k-notice-select-A` returned *Thursday / 6:40 a.m. / north gate*, which is
  the pinned needle exactly; `lc-8k-notice-select-B` handled the superseded
  decoy, naming the second bulletin and identifying the cancelled one;
  `lc-8k-crossref-two-facts-B` joined both facts (*twilight crew*, *granary
  loft*). Machine observations — **the grade is Cyril's.**
* **long-context 32k** — dossier `38bf7be6…`, one case, 41,563 prompt tokens,
  13 tokens out: *"Desmond Uhl holds the master key for the granary."*
* **A bank-calibration number worth having:** the 8k lanes tokenise to
  **10,180-10,246** real Qwen tokens against a nominal target of 8,192, and the
  32k lane to **41,563** against 32,768. The generator's
  `tokens_per_sentence_nominal: 21` under-counts by about 25% on this tokenizer.
  The lane names are nominal; the dossier records the true count. Whether to
  re-cut the rungs to land on the real numbers is a decision for Cyril.

### The envelope, measured

| rung | real prompt tokens | result |
|---|---|---|
| 8k | ~10,200 | decoded |
| 32k | 41,563 | decoded |
| 64k | ~82,000 | **CUDA out of memory** (23.18 GiB in use of 23.56) |
| 200k | — | **CUDA out of memory** (tried to allocate 3.95 GiB) |

> **Superseded for the 64k rung by r7 (below).** Those two OOMs were taken with
> the ungated preflight holding a full KV cache across a second prefill of the
> same prompt. With that cache released, 64k decodes: 3/3 `complete` at 84,100
> real tokens in ~23 s each. Only the 200k row survives as an envelope, at
> ~258,800 tokens. The table is left as it was recorded; read the r7 section for
> what the runner actually reaches.

**This bounds the runner, not the model.** The same card class already ran
250,940 tokens of the 27B GGUF under llama.cpp at 4-bit KV (plan §11). What does
not fit is *bf16 weights + unquantised KV + a full prefill through stock
transformers on one 24 GiB card*. The options are a bigger card, a chunked
prefill, or the `llama-server` runtime the R3 port note already anticipates
(`generation.runtime: hf-layer-streamed | llama-server`) — a decision for Cyril,
and one the long-context half of bank v2 cannot be read without.

**r6 recorded that OOM as a traceback and no dossier**, because the decode-path
preflight — two prefills of the longest prompt, ungated — died before any review
token existed. Both holes are now closed and pinned by
`test_the_runner_guards_the_preflight_and_the_decode_against_oom`: the preflight
records `agree: null` with its reason instead of dying, and a case that will not
fit is recorded with termination `out-of-memory`, which `classify_form` boxes as
an **artifact** — it counts nothing, because the model was never asked. Re-run
the two rungs with `run_r7.sh` to convert the crash into a measurement.

### The kernel path did NOT move under us — and the first version of this note said it did

*(Corrected 2026-09-02.)* This section used to read: "`fla` 0.5.2 was installed on
this box for the K3 lane **between** the r5 and r6 runs … the inventory changed
and the effective path did not … r5 and r6 are therefore on the same kernel path
and remain comparable — which is a fact, not an assumption." Every clause of that
is wrong in a different way, and the two that matter are:

* **The install did not happen between the runs.** `pip_install.log:86` and the
  `fla_core-0.5.2.dist-info/RECORD` mtime both date it `2026-09-01 14:25:13
  −0600`; r5 began `2026-09-02 00:44:18`. `fla` was present for *both* runs.
* **The effective path was never `torch-fallback`.** The r6 traceback shows fla's
  Triton chunked kernel running with `causal_conv1d` absent (see the corrected
  section above).

**What can be said now.** r5's kernel path was **not recorded** — the field did
not exist yet, and r5's log kept no stderr — so r5 and r6 are *not established by
their artifacts* as one kernel surface. They are established as one by the
install timestamps and by the fact that no package changed between them, which is
weaker evidence than an artifact and is labelled as such. **The r5→r6 delta is
therefore at least two variables** (`--thinking`, plus an unrecorded kernel
element), and the 13-cap-rides / 3,516 s versus 16-completions / 643 s table
below should be read as *the two surfaces as they were run*, not as a
single-variable ablation of `--thinking`. The single-variable claim is available
cheaply — two v2 dossiers on today's box, one per thinking mode — and has not
been paid for yet.

**Consequence for the anchor.** The r7 REPEAT is taken on a v2 box against
`3a13002f…`, whose kernel element was never measured. So r7 no longer prints
`ANCHOR HOLDS` / `LANE MOVED` alone: it prints both `lane_digest_validity` tuples
side by side, states in one line whether the comparison is a **REPRODUCTION** (a
v2 dossier on both sides, every validity element equal) or a **NEW ANCHOR**
(anything unmeasured or unequal on the r5 side, which is the case today), and
only then reports whether the digests match. A digest match against an unmeasured
tuple is luck, not a reproduction, and it is now labelled as such.

## r7 — the three rungs r6 could not finish (2026-09-02, after the fixes)

`./run_r7.sh /home/cereal/models/Qwen3.5-4B Q35-4B cuda:0`, GPU0 free
(24,101 MiB), same parent, same dtype, same card. The three rungs r6 lost, run
against the corrected runner. **All three now produce artifacts**, and two of
them produce a result r6 could not reach at all.

### 64k decoded. It never OOMed — the diagnostic did.

| lane | prompt tokens | out | termination | wall |
|---|---|---|---|---|
| `lc-64k-notice-select-A` | 84,106 | 26 | `complete` | 23.4 s |
| `lc-64k-notice-select-B` | 84,057 | 14 | `complete` | 22.2 s |
| `lc-64k-summary-B` | 84,105 | 48 | `complete` | 24.8 s |

dossier `fcad8f8906a6e9fd…`, lane digest `c7b35536…`, decode-path preflight
`agree: True`, emit-zero floor passed.

**This rung was never out of the machine's reach.** r6 recorded it as a
`torch.OutOfMemoryError` because the decode-path preflight decodes the first case
twice — manual loop, then `model.generate` — and *held the manual loop's KV cache
across the second prefill*. Two 84k caches plus 9 GiB of weights do not fit in
24 GiB; one does, with room to decode. Dropping the forward output before
`generate` (`oc_run_frozen_bank.decode_path_agreement`) is the whole difference
between "this runner does not reach 64k on one 3090" and the table above. The
old claim was a property of the diagnostic, published as a property of the
runner.

The answers are needle-selects and they land (machine observation, **not** a
grade):

> `lc-64k-notice-select-A` → "The audit is rescheduled for **Monday at 3:05
> p.m.** Auditors must sign in at the **archive vault**."
> `lc-64k-notice-select-B` → "Tuesday, 7:15 a.m., Piotr Nadeau"

### 200k is a real envelope, and it is now a dossier row instead of a traceback

Both lanes raised `torch.OutOfMemoryError` at **258,837** and **258,739** prompt
tokens, 1.3 s apart, and *both were recorded*: dossier `b1412ded70838efb…`,
termination `out-of-memory`, `generated_tokens: 0`, boxed `EMPTY` /
`is_artifact_box: true` with the reason "the prompt did not fit in memory on this
device, so the model was never asked". The three-box column reads
`artifact_miss 0 / content_miss 0 / read_required 2` — it counts **nothing**, as
the lens requires.

That the SECOND lane got its own measurement is the point of the other cache fix:
before it, the first lane's retained cache would have decided the second lane's
result, and the envelope would have been a property of the running order.

**The rung names are still wrong and are now wrong by more.** The "64k" lanes
tokenise to ~84,100 real Qwen tokens and the "200k" lanes to ~258,800. The
dossier records the true count; the lane ids and `prompt_tokens_target` do not,
and re-cutting them would move the draft bank's digest. Unchanged, and listed
below.

### The REPEAT reproduced the lane digest exactly — and is still not a reproduction

```
lane digest run 1: 3a13002f1df108679d401af65ce4201c3db1396421603e9c79edb90a027a2e37
lane digest run 2: 3a13002f1df108679d401af65ce4201c3db1396421603e9c79edb90a027a2e37
lanes moved: none
comparison kind: NEW ANCHOR
   because: unmeasured on one side: chat_template_thinking,
            chat_template_thinking_took_effect, linear_attention_kernel_path,
            linear_attention_kernel_report_schema
   because: kernel report schema None/'neural-forge-omni-kernel-path-v2'
            is not v2 on both sides
DIGESTS MATCH, but this is a NEW ANCHOR, not a reproduction: the two runs are
not established as one surface, so a match here is not evidence that the parent
held.
```

All sixteen responses are byte-identical to r5's, 1,024-token cap-rides included
(dossier `743f2e9c34f39ef1…`). **`3a13002f…` is now two runs.** But r5's dossier
carries no kernel element and no thinking element, so the *comparison* still has
unmeasured variables on one side and `run_r7.sh` says so before it says anything
else. `oc-dossier-Q35-4B-REPEAT.json` is the anchor to compare against from here:
it is the one with a complete v2 validity tuple.

**What the match is evidence for.** A pure-torch Gated-DeltaNet recurrence and
fla's Triton one differ in bf16 accumulation order; sixteen lanes of greedy
decoding, thirteen of them running the full 1,024 steps, landing byte-identical
across the two runs is hard to obtain if the two runs were on different kernels.
Taken with the install timestamps (`fla` 0.5.2 present since 2026-09-01
14:25:13, ten hours before r5), it is good evidence that r5 ran the same
`chunk=fla-0.5.2 … conv-prefill=torch` path r7 measured. It is evidence, not a
record: the r5 artifact still does not say.

### The kernel binding, measured directly

`oc_kernel_probe.py` loads the checkpoint on the CPU and prints the v2 report in
about a second. On this box, over **24** Gated-DeltaNet layers
(`out/kernel-probe-v2-Q35-4B.json`):

| slot | provider | resolved callable |
|---|---|---|
| `chunk` | **fla 0.5.2** | `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule` |
| `recurrent` | **fla 0.5.2** | `fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule` |
| `conv-prefill` | torch | *(unbound: transformers branches to its own convolution)* |
| `conv-decode` | torch | `…modeling_qwen3_5.torch_causal_conv1d_update` |

`transformers_fast_path_flag: false` — which is why the warning prints — and
`bindings_uniform_across_layers: true`. The refuted v1 rule would have labelled
this exact state `torch-fallback`; the probe publishes that string as
`inventory_label_v1_refuted` so the two can be lined up.

## Still open

* **The r5→r6 two-surface table is not a single-variable ablation.** r5's kernel
  element was never recorded, so that comparison has an unmeasured variable in
  it. The r7 REPEAT closes it for the *template-default* side (a full v2 tuple
  now exists at `3a13002f…`); the `--thinking off` side is still a v1 dossier.
  One re-run of `--thinking off` on this box would make it a clean one-variable
  table, and it has not been paid for.
* **Every dossier written before 2026-09-02 04:00 carries v1 kernel semantics**
  (`out/KERNEL-FIELD-CORRECTION-2026-09-02.md`). A v1 string and a v2 string are
  not comparable; check
  `lane_digest_validity.linear_attention_kernel_report_schema` on both sides
  before reading a kernel difference as a surface difference.
* **200k does not fit and 64k now does.** `lc-200k-*` OOMs at ~258,800 tokens in
  bf16 with an unquantised KV cache on one 24 GiB card. That is this runner's
  envelope, recorded as two artifact-boxed rows that count nothing. Reaching it
  needs a bigger card, a chunked prefill, or the `llama-server` runtime the R3
  port note anticipates — Cyril's call.
* **The long-context rungs are mis-named against the real tokenizer**: "8k" is
  ~10.2k tokens, "32k" is 41,563, "64k" is ~84,100, "200k" is ~258,800. The
  dossiers record the true counts. Re-cutting the lanes would move the draft
  bank's digest, so they are left as they are pending Cyril.
* **No vision or AV lane has ever decoded.** The five vision cards exist, are
  byte-reproducible and are digest-pinned in the bank; the runner refuses them by
  name because it has no processor or asset loader. That refusal is deliberate
  and tested — decoding the text half of a vision prompt and filing it as a
  vision result is the failure it prevents — but the gap is real.
* **Nothing here is graded.** Every column is an upper bound until Cyril attaches
  a per-case reading.

## The three-box canonical column

`signal` / `artifact-miss` / `content-miss`. A machine can only fill the last two,
and only after a reader has called a lane a miss. Loops, echo, cap-rides and a
missing EOS never count against intelligence; only a content-miss does.
