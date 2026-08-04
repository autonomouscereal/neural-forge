# Cross-family fold at real geometry — llama H576 into blank qwen3 H432 (2026-08-03/04)

Runner: `experiments/run_cross_family_real_scale_fold.py`
(`sha256 e997810230698490f7aef27cc19e8055088377ef33a1958a087bea7d9851fa7f`,
unchanged across all four runs below).  Everything below was produced on the
workstation, CPU only, `torch 2.13.0+cpu` / `transformers 5.13.0`, six threads,
no network, no server, `HF_HUB_OFFLINE=1`.

Provenance: the runner and its schema test have two authors.  A second agent
process worked the same wave concurrently on this workstation and edited both
files after they were created; those edits were kept, including the
`rope_theta_kilo` correction (SmolLM2-135M really does use RoPE base 100000).
That process publishes to `reports/cross-family-real-20260804/` and authored
`NOTES-W7A.md`; this directory and the addendum at the end of that file are the
other half.  The two report roots share nothing but the runner and the CPU.
249 MB total here, of which 245 MB is three folded children and their
tokenizers.

Four runs live beside this file.  The three that completed each seal a manifest
and each verifies:

```
python experiments/run_cross_family_real_scale_fold.py \
    --report-root reports/cross-family-real-20260803/<run> --verify-only
# {"verified": true, "files": 15, "problems": []}
```

The fourth is a run that failed closed on purpose.  It has no manifest — a
manifest is only written after the behavior stage — so `--verify-only` refuses
it with `No sealed manifest`, which is the intended shape of a refused run: a
sealed `fold-failure.json`, no `fold.json`, no child on disk.

| run | parent | folded depth | status |
|---|---|---|---|
| `L6-prefix-fold` | exact-subspace rank 432 | 6 of 30 layers | complete, decoded |
| `L2-exact-subspace-parent` | exact-subspace rank 432 | 2 of 30 layers | complete, decoded |
| `L2-full-rank-parent-control` | full-rank | 2 of 30 layers | complete, decoded |
| `tiny-fail-closed-control` | full-rank, tiny geometry | refused at layer 0 | **failed closed, rolled back to blank** |

---

## Read this before the numbers

**The parent is not pretrained.**  The task assumed SmolLM2-135M was in the
local Hugging Face cache.  It is in the *AI server's* cache, which this wave is
forbidden to touch, and the local-files-only load fails:

```
HuggingFaceTB/SmolLM2-135M FAIL OSError We couldn't connect to
'https://huggingface.co' to load the files, and couldn't find them in the
cached files.
```

There is no pretrained causal LM anywhere on this workstation.  So the parent is
a **seeded random initialization at SmolLM2-135M's exact geometry** — H576, 30
layers (a 6-layer prefix folded here), 9 query heads over 3 key/value heads,
head_dim 64, intermediate 1536, RoPE base 100000, RMSNorm epsilon 1e-5.  Every
receipt carries `parent.pretrained: false` and
`parent.weight_source: "seeded-random-init"`.

What that costs, stated up front so nothing below can be misread:

* The expected outcome — "degraded but structured" text — **was not achievable
  and did not happen.**  A random parent has no linguistic structure to degrade.
  Both parent and child emit degenerate repetition loops.
* The only behavioral question these runs can answer is *does the child
  reproduce the parent*, not *is the child any good*.  The answer to the first
  question is **no, not at this contraction and depth**, and that answer is
  reported honestly below rather than hidden behind the mechanism result.
* Everything about the **mechanism** — planning, chart compilation, the three
  operator lanes, the transaction/rollback discipline, the vocabulary bridge at
  50257 -> 16384 rows, the zero-parent-forward and zero-training counters — is
  a real measurement at real geometry and does not depend on the parent being
  pretrained.

The tokenizers are real.  `sshleifer/tiny-gpt2` ships the complete 50257-row
GPT-2 ByteLevel-BPE tokenizer; the child's vocabulary is a real 16384-row
ByteLevel-BPE **subset compiled by `prune_hf_tokenizer`** (16127 retained
merges, complete byte coverage, complete merge dependency chains), and the fold
installs its rows through a sealed **pair** bridge
(`bridge_kind: sealed-independent-tokenizer-pair-bridge`,
`artifact_sha256 8f7038fb...`, `profile aa880368...`), not through the strict
donor-row lane.  This is the documented fallback: the pair bridge is
source <-> source-subset, because no second independent real ByteLevel
tokenizer of useful size is available offline.  The subset relation is used a
second time, for *evaluation alignment only* — it gives an exact row map for
comparing a 50257-row head to a 16384-row one — and never for installing a
weight.

---

## Configuration

| | parent | child |
|---|---|---|
| family | llama | qwen3 |
| residual width | 576 | 432 (0.75x) |
| layers folded | 6 (prefix of 30) | 6 |
| query / key-value heads | 9 / 3 | 6 / 3 |
| head_dim | 64 | 64 (must match: the planner compares `rotary_dim`) |
| MLP intermediate | 1536 | 1152 |
| per-head Q/K norm | none (identity) | RMSNorm, fitted by the coupled kernel |
| vocabulary | 50257 | 16384, untied embedding and head |
| parameters | 79,137,216 | 26,106,096 |

Evidence: 48 calibration prompts from
`reports/kernel-stage2-20260801/calibration_prompts_64.json` (indices 0..47),
tokenized by the **child** tokenizer into 128-token right-padded windows; the
parent reads the donor preimage of exactly those rows, so both models see the
same text at the same positions.  6144 positions, **4797 valid**, 2399 fit /
2398 validation, 1347 padded.  **One parent forward** for the whole run.
Held-out: prompts 48..63, 80-token unpadded windows, never seen by any fit.

---

## Mechanism: what the executor actually did

`L6-prefix-fold`, `fold.json`:

* Plan **ready, zero gaps**, 14 actions:
  `token_interface_bridge` x1, `copy_exact` x1 (the parameter-free rotary
  global), `causal_gqa_rope_residual_attention_layer` x6,
  `causal_swiglu_residual_mlp_layer` x6.
  `partitions_every_learned_child_tensor: true` — every learned child tensor is
  owned by exactly one transaction.
* **14 attempts, 14 accepted**, status `complete`; 69 changed state paths, all
  unique; `completed_every_learned_child_tensor: true`.
* `observed_parent_root_forward_calls: 0`,
  `observed_parent_module_forward_calls: 0`, and an independent
  `register_forward_hook` counter on the parent also read **0** during
  execution.  The only parent forward in the run is the single evidence
  capture.
* `optimizer_constructions: 0`, `optimizer_steps: 0`, `backward_calls: 0`,
  `autograd_grad_calls: 0`, `pruning_operations: 0`,
  `quantization_operations: 0`,
  `executed_fold_static_zero_training_audit_passed: true`,
  `operator_static_zero_training_audit_passed: true` — the static audit covers
  the locally composed interface operator too.
* `child_recapture_calls: 12` — one live sequential recapture of the child per
  branch action, which is the discipline that keeps the branch targets in child
  coordinates instead of parent ones.
* Rollback is exercised by a fourth sealed run,
  `tiny-fail-closed-control` — tiny geometry (H128 -> H96), full-rank parent,
  and deliberately starved evidence (192 fit rows against 192 target MLP
  neurons).  The gated-MLP branch of layer 0 reaches endpoint relative RMSE
  **1.84994**, above the no-branch baseline of 1.0, and the operator refuses it:

  ```
  Cross-width causal gated_mlp branch operator endpoint relative RMSE
  1.8499435936767656 is at or above the no-branch baseline bound 1.0 at
  boundary 'layer001.attention_in'.
  ```

  `fold-failure.json` seals `child_state_rolled_back_to_blank: true` — the
  branch that failed and every branch accepted before it are undone, so a
  partially folded child never survives.  That run has no `fold.json`,
  no `child/` and no decoded output, which is the correct outcome.

The token interface installed **16384 of 16384 target rows** (0 uncovered) from
the pair codes, plus the fitted final-norm gain at
`final_norm_relative_rmse = 7.88e-07`.  Zero uncovered rows is expected here and
is a property of this tokenizer pair, not of the operator: a subset target makes
almost every pair code a byte-exact anchor.  The masked-row policy
(`leave-uncovered-target-rows-at-incumbent-blank`) is still enforced and still
sealed; it simply had nothing to leave alone.

---

## Numbers

### Boundary charts (13 boundaries, `charts.json`)

| | exact-subspace parent (L6) | full-rank parent (L2) |
|---|---|---|
| cycle defect `c_b`, median | 1.265e-07 | 0.1258 |
| norm-commutation row fluctuation, median | 1.302e-15 | 5.514e-03 |
| held-out retained energy fraction, min / median | 0.999999999999978 / 0.999999999999981 | 0.8955 / 0.9751 |
| absorption-demand unabsorbed fraction | 0 | 0 |

The exact-subspace parent is the stage-1 control
`tests/test_cross_width_causal_fold_execution.py` already relies on, lifted to
real geometry: residual-facing weights confined to one 432-dimensional subspace,
RMSNorm gains set to one.  A chart that finds that subspace discards nothing, so
every endpoint number under it is a statement about the branch kernels rather
than about how much a matrix can be compressed.

The full-rank control is the interesting surprise: a *randomly initialized*
576-wide residual still keeps **97.5% of its held-out energy in 432 dimensions**
(min 89.6%).  The balanced controllability/observability POD is finding real
anisotropy induced by RMSNorm and residual accumulation, not just cutting a
quarter of an isotropic ball.  Stage 2 measured 0.9926 median held-out retention
on the *pretrained* SmolLM2-135M, so the ordering is
pretrained > random-full-rank, as it should be.

### Per-branch held-out endpoint error, `L6-prefix-fold`

The no-branch baseline is 1.0 by construction (a zero write predicts nothing);
the operator refuses anything at or above it.

| layer | attention endpoint RMSE | q-norm gain RMSE | k-norm gain RMSE | gated-MLP endpoint RMSE | upstream error energy ratio (attn / mlp) |
|---|---|---|---|---|---|
| 0 | 0.12203 | 0.1933 | 0.0882 | 0.35884 | 7.63e-17 / 2.72e-03 |
| 1 | 0.21766 | 0.2381 | 0.1711 | 0.51020 | 1.04e-02 / 2.86e-02 |
| 2 | 0.26770 | 0.2010 | 0.1221 | 0.58604 | 1.01e-02 / 3.68e-02 |
| 3 | 0.31316 | 0.1999 | 0.1030 | 0.62474 | 9.05e-03 / 3.35e-02 |
| 4 | 0.33960 | 0.1868 | 0.0982 | 0.65387 | 7.46e-03 / 2.95e-02 |
| 5 | 0.35979 | 0.1928 | 0.0960 | 0.68615 | 6.13e-03 / 2.69e-02 |

Two things in that table matter more than the values.

1. **The composition did not run away.**  I expected it to.  A tiny-geometry
   rehearsal (H128 -> H96) showed the upstream error energy ratio multiplying by
   about ten per residual-write boundary, which would have tripped the
   fail-closed bound within a handful of layers.  At real geometry it rises to
   3.7e-02 at layer 2 and then **falls** — 3.35e-02, 2.95e-02, 2.69e-02.  The
   sequential recapture plus the per-boundary chart keeps the accumulated
   discrepancy bounded rather than compounding.  Depth was not the wall here.
2. **The gated-MLP branch is the weak one**, monotonically, at every depth —
   0.359 rising to 0.686 — while attention stays roughly half of it.  Stage 2
   saw the same ordering on the real pretrained parent (0.0759 attention against
   0.6657 MLP at the first boundary).  Whatever limits this fold is the
   1536 -> 1152 SwiGLU contraction, not the residual crossing and not the family
   change.

### Same-depth A/B: what the residual crossing costs

Both arms are two layers, identical in every other respect.

| | exact-subspace parent | full-rank parent |
|---|---|---|
| attention endpoint RMSE, L0 / L1 | 0.12592 / 0.22731 | 0.24126 / 0.30588 |
| gated-MLP endpoint RMSE, L0 / L1 | 0.37622 / 0.52896 | 0.49903 / 0.69183 |
| interface final-norm RMSE | 6.77e-06 | 2.74e-03 |
| teacher-forced logit relative RMSE | 0.5656 | 0.8771 |
| teacher-forced top-1 agreement | 23.05% | 4.84% |

The delta between the columns is exactly the price of the 2.5% of residual
energy the chart cannot keep when the parent is not confined.  It roughly
doubles every branch error and takes top-1 agreement from 23% to 5%.

### Whole-model teacher-forced agreement, held-out prompts

| run | logit relative RMSE | centered logit relative RMSE | top-1 agreement |
|---|---|---|---|
| `L6-prefix-fold` | 0.6773 | 0.6773 | 16.25% |
| `L2-exact-subspace-parent` | 0.5656 | 0.5656 | 23.05% |
| `L2-full-rank-parent-control` | 0.8771 | 0.8771 | 4.84% |

1280 held-out rows each.  The child's logits are compared against the parent's
restricted to the child's 16384 rows, on the same text at the same positions.

**This is the honest headline and it is a negative one.**  A 0.68 relative RMSE
against the teacher's own logits, with 16% top-1 agreement, is not behavior
transfer.  The child is a complete, finite, runnable model that was installed
entirely by closed-form transactions from one parent forward — and it does not
do what the parent does.

---

## Decoded continuations — complete, verbatim

Greedy argmax, no sampling, no temperature, no beams; 32 new tokens per prompt;
16 held-out prompts (corpus indices 48..63) that no fit ever saw.  Three
decodes per prompt:

* **child** — the folded qwen3 child over its own 16384-row vocabulary;
* **parent (rows)** — the parent, argmax restricted to the child's row set, which
  is the only token-for-token comparable decode;
* **parent (free)** — the parent over its full 50257 rows, for reference.

The complete objects, including token ids and the per-token surface forms, are
in `L6-prefix-fold/behavior.json` under `continuations`.  Nothing is summarized
away here.

### Character of the outputs

Both models collapse into repetition within two or three tokens.  The parent's
own free decode is `ignition ignition ignition ...`, `CENT CENT CENT ...`,
`Section Section Section ...`.  The child's is `spokes spokes spokes ...`,
`himself Berlin himself Berlin ...`, `fix Kh fix Kh ...`.  That is exactly what
an untrained decoder-only transformer does under greedy decoding: a small number
of embedding rows dominate the logit and the argmax locks onto them.

So the character of the child's text is **degenerate in the same way the
parent's is, but on a different attractor.**  The child sometimes cycles with
period two or three (`himself Berlin`, `fix Kh`, `duration voltage`) where the
parent cycles with period one, which is a real difference in the shape of the
collapse and not a sign of more structure.

Greedy prefix agreement between child and parent-restricted decodes: median 0
tokens of 32, max 13, min 0.  Fifteen of sixteen prompts diverge on the first
token.  The one that agrees for 13 tokens agrees on a repetition loop, so it
should be read as two models falling into the same attractor rather than as
transfer.

The full set follows.

#### held-out prompt 0

```
prompt tail  : ...xious people, whose genuine recollections are equally inconsistent. Good practice is to treat inconsistency as
child        : surv spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokes spokesumsumsumsumsumsumsumsumsumsumsums
parent (rows): customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers customers
parent (free): ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition ignition
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 1

```
prompt tail  : ...roof about a component used by everything else removes an entire class of concern from the components above it
child        : himself Berlin himself Berlin himself Berlin himself Berlin himself Berlin himself Berlin himself Berlin himself Berlin himself Berlin himself Berlin himself Berlin himself Berlin himself Berlin himself Bart himself Bart himself Bart
parent (rows): 1973 1973 1973 1973 1973 1973 1973ptions 1973ptions 1973ptions 1973ptions 1973ptions 1973ptions 1973ptions 1973ptions 1973ptions 1973ptions 1973ptions 1973ptions 1973ptions
parent (free):CENT 1973 1973 1973 1973CENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENTCENT
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 2

```
prompt tail  : ...occurrence expensive was that the new channel crossed land that had been developed in the intervening decades.
child        : duration voltage duration voltagemospetry duration voltagemospetry trafficetry trafficetry traffic voltage traffic03FOarp03FOarp voltage traffic03FOarp voltage trafficetry03
parent (rows): interface interface puzzle interface interface interface interfaceMain interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface interface
parent (free): interface SATSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSectionSection
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 3

```
prompt tail  : ... metric in common use rewards reducing overhead but says nothing about whether the computation was useful. Fac
child        : ey tweets fixGiven fixGiven fix toile fix KhseaGiven ey fix toile fix Kh fix Kh fix Kh fix Kh fix fix fix fix lips fix lips fix lips
parent (rows):� stellar stellar stellar fix stellar fix belt fix stellar fix stellar fix stellar fix belt fix interface fix interface fix interface fix interface fix interfacecher lipscher lips lips lips
parent (free): Winner negro negro negroMade stellar stellar stellar stellar Winner belt fix stellar WinnerMade stellar WinnerMade stellar WinnerMade collaborators collaborators collaborators collaborators lipsoenix WinnerMade negroMadeMade
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 4

```
prompt tail  : ...ugh, when to stop. Apprenticeship transmits that kind of knowledge efficiently and everything else inefficient
child        : publeased special publeased special publeased special publeased special publeased special puboerkerkerkerk tacticso tacticsvision specialTrump tacticsvision tactics supplyvision
parent (rows): elder commun sensors communANT Wisconsincare Wisconsincare Wisconsincare Wisconsincare Wisconsincare Wisconsincare elder commun elder communANT WisconsinVPVPVPVPVPVPVPVPVP
parent (free): Crowd endorsementExec Crowd endorsement Crowd Crowd Crowd Crowd Crowd Crowd Crowd Crowd Crowd Crowd commun Crowd commun Crowd commun Crowd prevailed wound Crowd prevailedpots Crowd prevailedpots SG historicalpots
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 5

```
prompt tail  : ...imentally reduced stated hostility but did not change policy preferences. That combination suggests the misper
child        : practically steep steep steep steep steep steep steep steep steep steep steep steep steep steep steep steep steep steep steep steep steep names loadedero steep names loadedero steep 9pop
parent (rows): steepagonrane steepagonrane steepagonrane steepagonrane steepagonraneraneraneraneraneraneraneraneraneraneraneraneraneraneraneraneranerane
parent (free): pigment split Cab attRot Montgomery attRot Montgomery attRot Montgomery attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot attRot
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 6

```
prompt tail  : ...le from the surface, they were routinely destroyed by bottom trawling before anyone knew they existed. Mapping
child        : Final PDF Final PDF Final PDF Final PDF Final PDF Arsenal statement statement statement statement statement statement statement statement statement statement statement statement statement statement statement statement statement statement statement statement statement
parent (rows): attempts Th equ quarterback attempts Th equ quarterback attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts
parent (free): attemptsHig attemptsHigHigHigHigHigHigHigHigHigHigHigHigHigHigHigHig attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts attempts
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 7

```
prompt tail  : ...han positional ones but cost space and encourage sloppiness about what the schema actually is. The decision is
child        : accounts x accounts x accounts holding king king fortune Following king fortune Following king fortune Following king toler Commander king toler Commander Laur holding Commander Laur holding Season cour number holding Season
parent (rows): fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fitting fittingLetarityarityarityarityarityarityarityarityarityarity
parent (free): fitting fitting fitting fitting fitting fitting fitting fitting unsu fitting unsu unsu unsu unsu unsu unsu unsu unsuarity fitting unsuarity fitting unsuarity fitting unsu unsu reconsider posture unsu reconsider
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 8

```
prompt tail  : ...rrors. Progress felt slow to the singers and was visible to nobody until it was suddenly complete. The passage
child        : 130 130"? 130"? 130"? 130"? 130"? 130 millenn 130 millenn 130 millenn 130 millenn 130 millenn 130 millenn 130 millenn 130 millenn 130 millenn 130 millenn 130
parent (rows):ussionussionussionussionussionussionussionussionussionussionussionussionussionussionussionussionussionussionussionussionussionussionussion <<ussion <<ussion <<ussion << facility <<
parent (free): collisionsussionThinkussionThinkThinkThinkThinkussion collisionsussion ported collisionsussion ported collisionsussion ported collisionsussion ported collisionsussion ported ported ported ported ported ported ported ported ported
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 9

```
prompt tail  : ...etwork. Breaking that loop requires changing several things at once, which is difficult under budgets that are
child        : Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Sex BureauMainMainMainMainMainMainMainMainMainMainMain
parent (rows): Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn subsid subsid subsid subsid subsid subsid subsid subsid subsid subsid subsid subsid subsid subsid subsid subsid subsid subsid Turn
parent (free): Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn Turn subsid subsid subsid subsid subsid subsid subsid subsid subsid subsidWER Turn Turn Turn subsid subsid subsidWER Turn
greedy prefix agreement: 13 of 32 tokens
```

#### held-out prompt 10

```
prompt tail  : ...acturers increasingly implement charge limits and thermal throttling to manage this, sometimes without telling
child        :)-claimclaimclaimclaimclaimclaim originally swimmingFinallyclaimFinallyclaimFinallyFinallyclaimFinally originally PunFinallyFinallyFinally originallyFinally originallyFinally originallyFinallyFinallyFinallyFinallyFinally
parent (rows): originallyFinally Brady originallyFinally Brady phr 42 originallyFinally originallyFinally Brady phr phr phr phr 42Finally originally standard originally standard originallyFinally phr phr phr grading originally standard originally
parent (free): 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465 465
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 11

```
prompt tail  : ...on. Two of the hands share a distinctive abbreviation for a common word, which points to shared training. None
child        : 15spacespace 15 special default researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher researcher
parent (rows):alianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalian
parent (free):alian Kialianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalianalian
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 12

```
prompt tail  : ...properties the abstraction actually guarantees rather than which ones it appears to. Most production incidents
child        : placing moves BTC less "[ flatoni flatoni flatoni flatoni flatoni flatoni flatoni flat this this this this this this this this this this this this
parent (rows): practices bring drives practices bring drivesirm drivesirm drivesirm drivesirm drivesirm drivesirm drivesirm drives Large drives Large drives Large drives Large Large Large Large Large Large
parent (free):Ble fetish drives combatants drives combatants drivesCENT combatants drivesCENTBle tofu tofu tofuBle tofuBleBleBleBleBleCENTCENTCENTCENTCENTCENT discriminatingBle tofuCENT
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 13

```
prompt tail  : ...d slowly since. Forecasting the dispersion accurately requires knowing the eruption column height in near real
child        : increase overt (); (); (); (); incentive (); rather increase overt incentive increase overt incentive increase overt incentive� overtamy overtamy overtamy overt overt overt overt overt overt
parent (rows):aronaronaronaronaronaronaronaronaron nodePTaronaronaronaronaronaronaronaron nodePT cy nopher27 cy nopher27 cy n cy
parent (free):pricedpricedpricedpricedpricedpricedpricedpricedpricedpricedpricedpricedpricedpricedERNERNERNERNERNERNERNERNERNERNERNERNERNERNERNERNERNERN
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 14

```
prompt tail  : ...g is not aesthetic fussiness. The result is indistinguishable from the surrounding fabric, which was the point
child        : existence plates plates les rendering existence plates les rendering existence les les les rendering existence existence existence existence les les les les les les les les les les les les les les
parent (rows): meals meals meals adverseuminumchens meals adverseuminumchens meals adverseuminumuminumuminumuminumuminumuminumchens meals Stephenchens mealszbuminumchens meals GMuminumchens mealsourge
parent (free): meals Hebdo meals Hebdo lenses raftYN uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising uprising
greedy prefix agreement: 0 of 32 tokens
```

#### held-out prompt 15

```
prompt tail  : ...uired a substantially more complex observation operator and a great deal of work on bias correction. It is one
child        : wield Ratheridden mainstream wieldiddencowiddenentialsentialsentialsentialsiddenentialsiddenentialsiddenentialsiddenentialsiddenentialsiddenentialsiddenentialsiddenentialsiddeniddeniddenidden
parent (rows):words combowords combowords combowords combowords combosoftsoftsoftsoftsoftsoftsoftsoftums combo combo combowordsums combowordsums combowords combowords Sept
parent (free): Werewolf xxmmmmmm FurnNever combo FurnNever FurnNever combo Furnmmm Furn Furnmmm Furnmmm FurnmmmFac FurnmmmFac FurnmmmFac FurnmmmFac Furn
greedy prefix agreement: 0 of 32 tokens
```

---

## What limits these numbers

* **The parent is not pretrained.**  Repeated because it is the single largest
  caveat.  Restore `HuggingFaceTB/SmolLM2-135M` to the workstation cache and
  every number above should be re-measured; the chart retention should rise
  (stage 2 measured 0.9926 median held-out retention on the real checkpoint
  against 0.9751 here for a random full-rank parent), and only then does the
  decoded text become readable evidence about anything.
* **Six layers of thirty.**  A bounded prefix, chosen because a 6-layer run
  costs 57 minutes of contended CPU and a 30-layer run would not fit the
  budget.  The prefix is honest about what it is (`bounded_layer_prefix: true`,
  `folded_layer_prefix: 6`, `full_depth: 30`) but it does not prove the
  composition survives thirty boundaries.  What it does show is that the error
  accumulation *stopped growing* by layer 2, which is the opposite of the
  tiny-geometry rehearsal and the first evidence that depth may not be the wall.
* **The reverse reclosure sweep never ran.**  Every branch receipt carries
  `reverse_reclosure_sweep_executed: false`.  The design prescribes exactly one
  reverse sweep — charts frozen, writers re-solved back to front against
  terminal-anchored targets — for the residual error this fold leaves on the
  table, and nothing in `src` executes it.  The gated-MLP endpoint numbers are
  the most likely beneficiary.
* **2399 fit rows against 1152 target MLP neurons.**  A ratio of 2.08, which is
  the same neighbourhood stage 2 worked in (2357 fit rows, intermediate 1152).
  It is enough to be determined but not comfortable; the fit/validation gap in
  the MLP branch should be read with that in mind.  A tiny-geometry run with the
  ratio at 1.0 failed the fail-closed bound outright, so this is a real
  sensitivity.
* **One head geometry.**  `head_dim` must match the parent exactly or the
  planner gaps the crossing on the rotary node, and dense qwen3 refuses
  `num_key_value_heads == 1` or `== num_attention_heads`.  With a 9/3 parent
  that leaves very little room; 6/3 was chosen, so the key/value width is held
  and only the query heads contract.
* **The interface operator is not in `src`.**  The planner emits one
  three-tensor interface action for a simultaneous family+width crossing and no
  promoted operator can execute it.  The runner composes one locally out of
  public pieces.  See `NOTES-W7A.md`.
* **Contended wall clock.**  A second agent process was running its own fold on
  the same six threads for part of the `L6-prefix-fold` run, and available RAM
  reached 860 MB.  The timings in `fold.json`
  (`chart_compile 1195 s`, `execute 2059 s`) are upper bounds, not clean
  measurements.  The two-layer arms ran with the box mostly idle
  (`chart_compile 63-73 s`, `execute 235-243 s`).

## What this does and does not establish

Established, at real geometry, by artifacts that verify:

* A llama parent and a blank qwen3 child differing in **family**, **residual
  width** (576 -> 432), **head count** (9 -> 6 query heads), **MLP width**
  (1536 -> 1152), **Q/K normalization** (identity -> per-head RMS, fitted) and
  **vocabulary** (50257 -> 16384, sealed pair bridge) can be planned gap-free and
  installed completely by fourteen closed-form transactions in one executor run.
* From **one** parent forward, with **zero** parent forwards during execution and
  zero optimizer steps, backward calls, gradient calls, pruning or quantization
  operations, under a static source audit that also covers the locally composed
  operator.
* The per-boundary residual charts are numerically exact under the
  exact-subspace control (cycle defect 1.3e-07, norm-commutation 1.3e-15) and
  still retain 97.5% of held-out energy against a full-rank random parent.
* Accumulated cross-boundary error rose for two layers and then declined, so a
  six-boundary composition is stable in a way the tiny-scale rehearsal predicted
  it would not be.

Not established, and not claimed:

* **Behavior transfer.**  0.68 teacher-forced logit relative RMSE and 16% top-1
  agreement is a failure to reproduce the parent, and the greedy decodes confirm
  it — median zero tokens of agreement.
* **Anything about intelligence.**  The parent has none to transfer.
* **Depth beyond six.**
* **That the vocabulary crossing is hard.**  A subset target makes almost every
  pair code a byte-exact anchor, so the pair bridge covered every row.  A
  genuinely unrelated tokenizer would leave rows uncovered and exercise the
  masked-row policy this run only sealed.

## Reproduction

```bash
# headline run, 6-layer prefix, exact-subspace parent
HF_HUB_OFFLINE=1 python experiments/run_cross_family_real_scale_fold.py \
    --report-root reports/cross-family-real-20260803/L6-prefix-fold \
    --parent-geometry smollm2-135m --parent-residual-rank 432 --layers 6 \
    --calibration-prompts-count 48 --holdout-prompts-count 16 \
    --evidence-window 128 --holdout-window 80 --decode-new-tokens 32

# same-depth A/B
HF_HUB_OFFLINE=1 python experiments/run_cross_family_real_scale_fold.py \
    --report-root reports/cross-family-real-20260803/L2-exact-subspace-parent \
    --parent-geometry smollm2-135m --parent-residual-rank 432 --layers 2 \
    --calibration-prompts-count 48 --holdout-prompts-count 16 \
    --evidence-window 128 --holdout-window 80 --decode-new-tokens 32

HF_HUB_OFFLINE=1 python experiments/run_cross_family_real_scale_fold.py \
    --report-root reports/cross-family-real-20260803/L2-full-rank-parent-control \
    --parent-geometry smollm2-135m --layers 2 \
    --calibration-prompts-count 48 --holdout-prompts-count 16 \
    --evidence-window 128 --holdout-window 80 --decode-new-tokens 32
```

The runner is resumable: each stage seals a receipt whose `input_sha256` chains
the stages above it, and a stage is reused only when nothing it depends on has
moved.  The folded child is saved under `<run>/child` and reloaded against its
sealed state fingerprint, so the behavior stage can be re-run without refolding.

Input corpus:
`reports/kernel-stage2-20260801/calibration_prompts_64.json`
(`sha256 45852453b3bbdcf91ecb9ca785d4e3aa340138b7766aad6e0d2855ce28bf3791`).

Schema test: `tests/test_cross_family_real_scale_fold_seal.py` (pins the receipt
seal, the stage lineage, the configuration identity and the report verification;
it does not execute a fold).
