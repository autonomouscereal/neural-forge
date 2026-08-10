# NOTES-MX-E — matrix campaign, the rest of wave 1

Wave MX-E. The remaining 27 cells of wave 1 (design §4.1 rows 4–30), run against
the fresh evidence base `2dec87ac…` at the pre-registered 5e-3 calibration
tolerance, on both RTX 3090s with per-runner ledger shards merged afterwards.

Repo baseline HEAD `c59be064`, branch `codex/lrc-blank-compression`. The frozen
collector, the compatibility policy and Codex's live-edit files
(`dynamic_fold_diagnostics`, `dynamic_model_inspector`, the closed-loop adapter,
the `stage2_*` files and their tests) were **not** touched. No `git add`, no
commit.

**Verdicts up front:**

| deliverable | verdict | one line |
|---|---|---|
| 27 cells, wave 1 rows 4–30 | **ALL SEALED** | 3 PASS, 20 FAIL, 4 INFRA; wall clock 18:03→18:32 UTC on two GPUs |
| the campaign's first passes | **C06.T1.M16.L02, C06.T7.M16.L02, C07.T1.M16.L02** | attention readers at L2 clear every link; free-run 0.977 / 0.920 / 0.941 |
| T5 vs T1 at the same three sites | **a behavioural tie that T5 wins on storage** | mean free-run 0.714 vs 0.721 — inside the noise; T5 stores 983,040, T1 stores 1,966,080 against a dense 1,048,576 |
| the S8 macrocell band | **0 of 18 where it was actually measured** | the rendered ledger's "3/26 (12%)" is an artifact; all three agreements come from cells whose band is a sentinel or a category error |
| one harness defect, fixed | **`_flatten_rows` had the attention axis order backwards** | it blocked all 11 attention cells; the fix is proven inert on every non-attention cell, field for field |
| one harness gap, reported not fixed | **the calibration arm assumes a single 2-D linear map** | that is why C03, C04, C10 and C14 are `infra` — four cells void, not four null results |

---

## 1. Before anything ran: the deployment check

`NOTES-MX-D` §5's lesson is that a receipt can be self-consistent and still lie
about which kernel ran. So the deployment was digested against the repo before a
single cell started, and this time the check covered **more than the eight files
MX-D synced** — because the wave-1 cells about to run were the attention ones,
and the attention lane has files MX-D never had to look at.

| file | deployed | repo @ `c59be064` | verdict |
|---|---|---|---|
| `matrix_cell_receipt.py` | `4f72ca3b0a817cb3` | `4f72ca3b0a817cb3` | identical |
| `matrix_technique_registry.py` | `c2b533774ac6d990` | same | identical |
| `matrix_technique_body_routes.py` | `055082c8a081317c` | same | identical |
| `matrix_technique_lowrank_writers.py` | `ab53ef253b35644b` | same | identical |
| `matrix_technique_relation_matching.py` | `8281e44123b39970` | same | identical |
| `run_lfm2_matrix_cell.py` | `f72f5511fb326fff` | same | identical (then patched, §3) |
| `render_matrix_ledger.py` | `13715e6883322cc3` | same | identical |
| `seal_matrix_cell_review.py` | `08fe52edc834ee0b` | same | identical |
| `matrix_campaign_probe_prompts_v1.json` | `6230f878be9287a9` | same | identical |
| `matrix_campaign_smoke_prompts_v1.json` | `82ebf70a1606ec8a` | same | identical |
| **`hf_lfm2_direct_local.py`** | `a7eb4fce441a5388` | `c0ddc19de3bbd7ab` | **differs** — analysed below |
| **`hf_lfm2_matrix_attention_replay.py`** | **absent** | `beb27e8d712a64d5` | **missing** — analysed below |

Neither divergence was synced, and both decisions are recorded rather than
assumed:

* **`hf_lfm2_direct_local.py` is exactly the `dee1c952` version** the deployment
  was built from, and the delta to HEAD is commit `2f205aa2`'s *additive* work:
  one new schema constant, one new public
  `build_lfm2_direct_local_runtime_alias_manifest`, and a refactor that splits
  `_lfm2_direct_local_state_alias_abi` into a `_from_entries` helper it then
  calls with the same argument. Five hunks, at lines 42 / 248 / 300 / 1833 /
  1835. The runner imports exactly two names from this module,
  `load_lfm2_direct_local_model` and `load_lfm2_direct_local_tokenizer`, and
  neither is touched. Syncing it would have changed `code_identity.src_tree_sha`
  for no behavioural difference and split wave 1's provenance for nothing.
* **`hf_lfm2_matrix_attention_replay.py` is imported by nothing the runner
  reaches.** Its only importer in the tree is its own test file; the runner's
  `discover_technique_modules` scans for `matrix_technique_*` and never sees it.
  Deploying it would have moved the tree digest without moving a line of
  executed code.

So the wave-1 cells ran the kernels the repo says they ran, and the two
divergences are named here rather than left for the next wave to rediscover.

---

## 2. The GPU that was not the GPU

The two-runner plan was to use both RTX 3090s — `nvidia-smi` indices 0 and 4.
Batch B was launched with `CUDA_VISIBLE_DEVICES=4` and its first five cells came
back `infra` in eight minutes. The logs said why:

```
Tesla P40 with CUDA capability sm_61 is not compatible with the current
PyTorch installation.  The current PyTorch install supports CUDA capabilities
sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
```

CUDA's default device ordering is `FASTEST_FIRST`, not `nvidia-smi`'s PCI order,
and on this host index 4 resolves to a P40. Measured directly:

```
CUDA_VISIBLE_DEVICES=4 default    -> Tesla P40 (6, 1)
CUDA_VISIBLE_DEVICES=4 PCI_BUS_ID -> NVIDIA GeForce RTX 3090 (8, 6)
```

This matters more than a wasted eight minutes. `NOTES-MX-C` §4.4's portability
wall is precisely that A9's fp32 OMP candidate search does not reproduce across
architectures; half of wave 1 measured on sm_61 would not have been comparable
to cells 1–3 measured on sm_86, and the receipts would not have said so, because
nothing in the schema records the device architecture.

**All five P40 rows were quarantined, not merged.** They are `infra` receipts
that died in evidence loading before any arithmetic, so nothing numeric was
lost. They sit in `/home/cereal/nf-mxe/discarded-p40/` with their shard, and no
campaign ledger ever saw them.

The batch driver now **refuses to start** on anything else:

```
CAP=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY -c '...')
if [ "$CAP" != "8.6 NVIDIA GeForce RTX 3090" ]; then
  echo "PREFLIGHT REFUSED: GPU $GPU resolves to '$CAP', not the SM86 RTX 3090" >&2
  exit 3
fi
```

Worth registering as a campaign rule: **the receipt schema has no field for the
compute architecture the cell ran on**, and it is the one input to a matrix cell
that the sealed identities do not cover.

---

## 3. The harness defect that blocked every attention cell

With the right GPU, batch B still failed at the same place:

```
CELL FAILED RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 16384 but got size 64 for tensor number 1 in the list.
```

### 3.1 What it was

`run_lfm2_matrix_cell.py::_flatten_rows` declared, in its own docstring:

> Attention endpoints are captured as ``(batch, tokens, heads, head_dim)``

They are not. The frozen collector writes them in the axis order the attention
op works in. Read straight out of the sealed cache at layer 2:

| endpoint | prefill | decode step |
|---|---|---|
| `attention.q_linear` | `[1, 16, 256, 64]` | `[1, 16, 1, 64]` |
| `attention.k_linear` | `[1, 8, 256, 64]` | `[1, 8, 1, 64]` |
| `attention.v` | `[1, 8, 256, 64]` | `[1, 8, 1, 64]` |
| `attention.context` | `[1, 16, 256, 64]` | `[1, 16, 1, 64]` |
| `attention.o`, everything else | `[1, 256, 1024]` | `[1, 1, 1024]` |

Axis 1 is heads (16 query, 8 key/value — GQA makes this unambiguous) and axis 2
is tokens. Folding the *last two* axes therefore turned the prefill batch into
`(16, 16384)` and each decode step into `(16, 64)`, and the per-endpoint
concatenation across phases raised before a number was computed.

The crash was the lucky part. Had the phases happened to agree in width, the
rows would have been head-major spans of concatenated tokens — silently
misaligned against every 3-D endpoint and against the parent weight the
calibration arm multiplies by — and the cells would have produced plausible,
wrong numbers.

### 3.2 The fix, and why it is provably the right one

`experiments/run_lfm2_matrix_cell.py:309` — transpose the head axis back behind
the token axis before folding:

```python
if tensor.ndim == 4:
    tensor = tensor.permute(0, 2, 1, 3)
```

Four 4-D endpoints exist in `_LAYOUT`'s reach and all four are per-head
attention, so the branch is exactly scoped.

The layout is not asserted from a convention; it is derived from the operation
that wrote it. HF computes `q_proj(x).view(B, T, H, D).transpose(1, 2)`, so the
inverse of the seal is `permute(0, 2, 1, 3).reshape(B*T, H*D)`, and that is what
makes the calibration arm's `rows(operator.input) @ q_proj.T` comparable to
`rows(attention.q_linear)` *positionally*. Five new tests in
`tests/test_run_lfm2_matrix_cell.py` (16 → 20), one of which is that derivation
run end to end, one of which pins the real capture shapes and the 264-row
per-batch count, one of which pins that GQA endpoints keep their narrower 512
width, and one of which pins that 3-D endpoints are untouched.

The pre-existing test asserted the wrong layout in its own comment and was
rewritten rather than deleted.

**The calibration gate is the empirical proof.** A wrong permutation cannot
produce a small number here: after the fix every attention cell's copy_exact
calibration lands on the same bf16 quantization floor as the conv and MLP cells
— `0.001662` at C06, `0.002094` at C07, `0.002282` at C08, `0.001653` at C09,
against a tolerance of 5e-3 and a floor of ~2⁻⁹.

### 3.3 The fix is inert everywhere else — measured, not argued

`C05.T1.M16.L00` was run once before the patch and once after, same evidence
cache, same knobs, same GPU. All **97 receipt fields** compared:

```
volatile diffs (expected): cell_instance_id, code_identity.src_tree_sha,
                           receipt_sha, sealed_at, wall_time_seconds
SUBSTANTIVE DIFFS: 0
```

Every boundary, composed, behavioural and solve number is bit-identical. So
cells 1–3 and batch A (pre-patch tree) are directly comparable to batch B
(post-patch tree), and the two `src_tree_sha` values in the ledger are a
provenance record, not a confound. Both runs are kept: the sealed instance is
`cell-receipt.json` (`9414632eb139`) and the post-patch control is
`cell-receipt.post-patch-rerun.json` (`73acee21d14c`) with its log, in
`reports/matrix-campaign-20260810/cells/C05.T1.M16.L00/`. The re-run was
deliberately **not** appended to the campaign ledger — it is a harness control,
not a cell instance. The pre-patch deployment file is backed up at
`/mnt/ai_data/neural-forge-matrix/backup/deployment-before-mxe-20260810T180150Z/run_lfm2_matrix_cell.py.pre-mxe`
(`f72f5511fb326fff`).

Local suite: `tests/test_run_lfm2_matrix_cell.py` 20 passed; with
`test_run_lfm2_matrix_cell_fresh_evidence.py`, `test_matrix_cell_receipt.py` and
`test_matrix_technique_registry.py`, **87 passed**. `ruff` reports the same two
pre-existing unused imports it reports at HEAD and nothing new.

---

## 4. The results

27 cells. Boundary/composed numbers are diagnostic; `free-run` and the review
are the only fields §5 rule 1 may read.

| # | cell | component | technique | site | write cos | macrocell | vs control | free-run | smoke | link | **status** |
|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| 4 | `C13.T5.M16.L00` | mlp_down | T5 | L00 | 0.99182 | 0.1097 | 0.08937 | 0.332 | coherent | composition | fail |
| 5 | `C13.T5.M16.L05` | mlp_down | T5 | L05 | 0.99336 | 0.0407 | 0.05593 | 0.502 | coherent | composition | fail |
| 6 | `C13.T5.M16.L13` | mlp_down | T5 | L13 | 0.99633 | 0.0327 | 0.04819 | 0.561 | coherent | composition | fail |
| 7 | `C11.T5.M16.L00` | mlp_gate | T5 | L00 | 0.99232 | *n/m* | 0.09837 | 0.400 | coherent | composition | fail |
| 8 | `C12.T5.M16.L00` | mlp_up | T5 | L00 | 0.98642 | *n/m* | 0.19453 | 0.307 | coherent | composition | fail |
| 9 | `C03.T5.M16.L00` | conv_in_proj | T5 | L00 | — | — | — | — | — | infra | **infra** |
| 10 | `C08.T5.M16.L02` | attn_v | T5 | L02 | 0.99084 | *n/m* | 0.03812 | 0.578 | coherent | composition | fail |
| 11 | `C09.T5.M16.L02` | attn_o | T5 | L02 | 0.99952 | 0.0164 | 0.02002 | 0.709 | coherent | composition | fail |
| 12 | `C05.T1.M16.L00` | conv_out_proj | T1 | L00 | 0.99842 | 0.0395 | 0.04367 | 0.686 | coherent | composition | fail |
| 13 | `C05.T1.M16.L05` | conv_out_proj | T1 | L05 | 0.99684 | 0.0239 | 0.03870 | 0.744 | coherent | composition | fail |
| 14 | `C05.T1.M16.L13` | conv_out_proj | T1 | L13 | 0.99636 | 0.0155 | 0.03391 | 0.734 | coherent | composition | fail |
| 15 | `C13.T1.M16.L00` | mlp_down | T1 | L00 | 0.99730 | 0.0460 | 0.06677 | 0.535 | coherent | composition | fail |
| 16 | `C13.T1.M16.L05` | mlp_down | T1 | L05 | 0.99485 | 0.0350 | 0.06287 | 0.383 | coherent | composition | fail |
| 17 | `C06.T1.M16.L02` | attn_q | T1 | L02 | **1.00000** | *inv* | **0.00124** | **0.977** | coherent | — | **PASS** |
| 18 | `C07.T1.M16.L02` | attn_k | T1 | L02 | 0.99991 | *n/m* | 0.00743 | **0.941** | coherent | — | **PASS** |
| 19 | `C08.T1.M16.L02` | attn_v | T1 | L02 | 0.99921 | *n/m* | 0.01539 | 0.738 | coherent | free-run | fail |
| 20 | `C09.T1.M16.L02` | attn_o | T1 | L02 | 0.99989 | 0.0072 | 0.01322 | 0.820 | coherent | free-run | fail |
| 21 | `C04.T1.M16.L00` | conv_kernel | T1 | L00 | — | — | — | — | — | infra | **infra** |
| 22 | `C05.T8.M16.L00` | conv_out_proj | T8 | L00 | 0.99865 | 0.0411 | 0.05430 | 0.523 | coherent | composition | fail |
| 23 | `C13.T8.M16.L00` | mlp_down | T8 | L00 | 0.98130 | 0.1266 | 0.17910 | 0.199 | coherent | composition | fail |
| 24 | `C09.T8.M16.L02` | attn_o | T8 | L02 | 0.99955 | 0.0155 | 0.02102 | 0.678 | coherent | composition | fail |
| 25 | `C06.T7.M16.L02` | attn_q (QK) | T7 | L02 | 0.99881 | *inv* | 0.01122 | **0.920** | coherent | — | **PASS** |
| 26 | `C06.T7.M16.L08` | attn_q (QK) | T7 | L08 | 0.99851 | *inv* | 0.00942 | 0.846 | coherent | free-run | fail |
| 27 | `C05.T4.M16.L13` | conv_out_proj | T4 r31 | L13 | 0.80268 | 0.1121 | 0.19786 | 0.225 | coherent | local-solve | fail |
| 28 | `C09.T4.M16.L02` | attn_o | T4 r31 | L02 | 0.89995 | 0.2395 | 0.31799 | 0.049 | **incoherent** | local-solve | fail |
| 29 | `C10.T1.M16.L02` | attn_per_head | T1 | L02 | — | — | — | — | — | infra | **infra** |
| 30 | `C14.T1.NA.L00` | norms | T1 | L00 | — | — | — | — | — | infra | **infra** |

*n/m* = macrocell **not measured**: the component's write is narrower or wider
than the residual, so the runner cannot compose it and records the sentinel
`residual.post_layer#not-composable-from-write` with a hardcoded relL2 0.0 /
cos 1.0 / band `true`. *inv* = the macrocell is dimensionally composable and
mechanically meaningless (§4.4). Every calibration relL2 landed in
`0.000638 – 0.002282` against the 5e-3 tolerance; the only exception is cell 29,
which measured **4.4955** and was refused (§4.5).

### 4.1 The T5-vs-T1 verdict, called out explicitly

Design §4.1 rows 12–14 exist to answer one question against rows 1–3: *does the
sparse-OMP body route beat plain dense ridge at the same component, same level,
same three sites?* Same evidence, same split, same knobs, same GPU.

| site | T5 free-run | T1 free-run | Δ | T5 vs-control | T1 vs-control | T5 stored | T1 stored |
|---|---:|---:|---:|---:|---:|---:|---:|
| L00 | 0.662 | **0.686** | −0.024 | 0.04355 | 0.04367 | 983,040 | 1,966,080 |
| L05 | **0.791** | 0.744 | +0.047 | 0.04048 | 0.03870 | 983,040 | 1,966,080 |
| L13 | 0.688 | **0.734** | −0.046 | 0.02948 | 0.03391 | 983,040 | 1,966,080 |
| mean | 0.714 | 0.721 | −0.007 | 0.03784 | 0.03876 | 983,040 | 1,966,080 |

**On behaviour it is a tie.** T1 wins two sites, T5 wins one, the mean gap is
0.007 on a metric whose site-to-site spread within a single technique is 0.13.
Neither technique passes; all six cells fail at the same chain link, at composed
drifts that differ by less than 12% of each other. The route's extra machinery —
support selection, the coupled write solve, four accepted Jacobi sweeps — buys
nothing measurable over a POD basis and one ridge solve at this component.

**On storage it is not close, and the direction is the opposite of the naive
reading.** The parent `conv.out_proj` is 1024×1024 = 1,048,576 parameters. T5 at
M16 stores 983,040 — a real 6% saving. T1 at M16 stores 1,966,080, because its
stored form is a `1024×960` basis *plus* a `960×1024` solved factor: **1.88× the
dense parent.** T1's "contraction" at a marginal level is a rank restriction,
not a compression, exactly the trap `NOTES-MX-B` §4.3 documented for T4 and §5.2
noted for T7. The same applies to every T1 cell in this wave (C13's T1 stores
8,601,600 against a 2,621,440 parent — 3.3×).

So the honest verdict is: **at `conv_out_proj` M16, T5 matches dense ridge
behaviourally and is the only one of the two that is a contraction at all.**
That is a weaker claim than "the route wins" and a stronger one than the
free-run column alone supports.

The same ordering appears wherever all three body techniques were run at one
site, and T8 is consistently last:

| site | T1 | T5 | T8 |
|---|---:|---:|---:|
| C05 @ L00 | **0.686** | 0.662 | 0.523 |
| C13 @ L00 | **0.535** | 0.332 | 0.199 |
| C09 @ L02 | **0.820** | 0.709 | 0.678 |

T8's scale-free Gram objective — the design's "invariance bet", §1.2's
highest-value unknown — is the **worst** technique at all three sites it was
tried, including the `conv_out_proj` wall it was chosen to cover. It is not
inert (the receipts confirm `gram_truncation_active` and the row weighting), it
is simply a worse objective for this task than reproducing the values.

### 4.2 The three passes, and what they are not

`C06.T1.M16.L02`, `C07.T1.M16.L02`, `C06.T7.M16.L02` are the first cells in the
campaign to satisfy §5 rule 1: free-run ≥ 0.90 against the copy_exact control,
three decode-smoke outputs reviewed coherent, emit-zero floor passed.

`C06.T1.M16.L02` is the strongest cell in wave 1 by a wide margin — write cosine
`1.00000`, composed drift `0.00124` against a 0.02 gate (16× inside it),
free-run 0.977 — and **two of its three smoke outputs are byte-identical to the
copy_exact control's**, including the control's own "for your convenience"
wording slip. `C07.T1` reproduces the control's dinner-plan output byte for byte
as well.

Three things these passes do **not** license.

1. **They are not compressions.** C06.T1 and C07.T1 store 1.88× and 1.41× their
   parent tensors. C06.T7 stores 1,044,480 against 1,048,576 — a **0.4%**
   parameter saving, exactly as `NOTES-MX-B` §5.2 warned; T7's contraction is in
   the score-kernel rank (`head_dim` 60 of 64), and it only becomes a storage
   win if the harness stores the factored form, which it does not.
2. **They are one site each.** §5 rule 2 needs ≥2 of 3 sites. The rendered ledger
   marks C06/T1 and C07/T1 `works to M16` off a single site; the second site is
   a wave-2 obligation, not a result. And T7's own second site *fails*: the same
   technique at the same level at L08 keeps composition (drift 0.0094, better
   than L02's 0.0112) and loses behaviour, free-run 0.846.
3. **They say nothing about composing.** §5 rule 6 is explicit, and cell 26 is
   the object lesson inside one technique.

### 4.3 T4 at the goal document's own witness rank

Cells 27 and 28 were run in the rank-knob form `NOTES-MX-B` §4.3 requires: at
M16 a factored 1024×1024 writer would *store* 1,966,080 against a dense
1,048,576 and the kernel fails closed, so both ran at `writer_rank: 31` — the
rank of the feasibility witness in `CURRENT_FOLDING_GOAL`. The M16 in those cell
ids is a nominal level label; the operative contraction is the rank, and the
receipts seal `writer_rank: 31`, `requested_writer_rank: 31`,
`maximum_shrinking_rank: 511`, `writer_rank_clamped: false`.

The parameter claim reproduces exactly: 63,488 stored against 1,048,576, ratio
**0.0605**, at both sites, with `weight_energy_outside_evidence_range` at 4e-14
(the evidence spans the writer's row space, so the evidence weighting gave up
nothing).

**The fit does not.** Write cosine falls to **0.803** at C05/L13 and **0.900** at
C09/L02, both under the 0.95 gate, so both cells stop at chain link 3 —
`local-solve`. The witness is a *layout* feasibility result and this is the first
time it has been asked to be a *fit* result on real sealed evidence at depth.
It is not one.

C09.T4 is also the campaign's **only `incoherent` body cell**: its one-pan dinner
output degenerates into "protein, carbs, and carbs, and carbs," repeated some
forty times, and the runner's termination classifier correctly labelled it
`loop` rather than `cap` — so the censoring rule does not apply and this is a
real failure. C05.T4 at the same rank degrades without collapsing (its circuit
breaker becomes "a safety valve … when it's too hot or too low") and stays
coherent. Same technique, same rank, same geometry, different site, different
side of the coherence line.

### 4.4 The S8 band predicted behaviour 0 times out of 18

The rendered ledger reports `Agreement 3/26 (12%)`. **That number should not be
read.** Every one of its three agreements comes from a cell whose macrocell
reading is not a measurement:

| class | cells | what the band reading is | behaved |
|---|---:|---|---:|
| **measured** | 18 | `residual.post_layer`, composed from a write that really is added into the residual (C05, C09, C13) | **0** |
| sentinel | 5 | write width ≠ residual width (C07, C08, C11, C12); the runner records relL2 0.0 / cos 1.0 / band `true` by fiat | 0 |
| category error | 3 | C06: `q_linear`'s width happens to equal the residual's, so the runner composes `residual.post_layer + (q̂ − q)` — dimensionally valid, mechanically meaningless, since q does not enter the residual additively but through a softmax | 0 |
| none | 4 | the `infra` cells never reached the boundary stage | 0 |

The three passes are two category-error cells and one sentinel cell. **On the 18
cells where the band was genuinely measured, it is 0 for 18.** With MX-D's three
C05 cells already inside that count, the campaign's pre-registered §5 rule-4
tally now reads: *in band, did not behave — 18; out of band — 0.* The band has
still never been observed to fail, and it has still never predicted anything.

The C06 case is the one to fix before wave 2, because it is the only one that
produces a plausible-looking number instead of an obvious sentinel: 0.0066 at
C06.T1 and 0.1157/0.1517 at C06.T7 look like band readings and are not.

### 4.5 Four `infra` cells: one gap, four faces

Cells 9, 21, 29 and 30 are void — "cell void, not evidence", per the chain's
first link. They are void for one reason, and it is worth stating once rather
than four times.

**The runner's generalized `published_child` calibration gate assumes the
component is a single 2-D linear map from `input_endpoint` to `write_endpoint`**
(`run_lfm2_matrix_cell.py:1648`, `reproduced = held_input @ parent_weight.T`).
That holds for C05–C09 and C11–C13. It fails four different ways elsewhere:

| cell | component | what happened | why |
|---|---|---|---|
| 9 | `conv_in_proj` | `RuntimeError: size of tensor a (3072) must match tensor b (1024)` | `_LAYOUT["C03"]["write"]` is `None` and `component_spec` falls back to `layout["write"] or layout["macrocell"]` (`matrix_technique_registry.py:322`), so a 3072-wide in_proj output is compared against the 1024-wide residual. **The evidence exists** — the cache carries `short_conv.b`, `.c` and `.x`, whose stack *is* the in_proj output — but the layout has no name for the stacked endpoint. |
| 21 | `conv_kernel` | `RuntimeError: Expected size for first two dimensions of batch2 tensor to be [3, 1024] but got [3, 1]` | the FIR kernel is `(1024, 1, 3)`; `parent_weight.T` is not a linear map. T1's own kernel would also have refused it — `contraction_axis == "kernel"` is not in `("output", "input")`. |
| 29 | `attn_per_head` | calibration relL2 **4.4955** vs the 5e-3 tolerance → refused | C10 owns four tensors; the gate takes `weight_keys[0]` (`q_proj`) and compares `operator.input @ q_proj.T` against `operator.output`. Shapes agree, so it computes a number, and the number is meaningless. This is the cleanest demonstration of the gap: it did not crash, it *measured the wrong thing and was correctly refused for it*. |
| 30 | `norms` | `RuntimeError: size of tensor a (1584) must match tensor b (1024)` | the RMSNorm gain is a 1-D `(1024,)` vector; `held_input @ gain.T` collapses 1584×1024 rows to a 1584-vector. The C14 kernel's own `predict` normalizes before scaling; the generic gate cannot know that. |

**Deliberately not fixed.** Cell 9's fix requires *inventing* a stacked endpoint
and choosing the B/C/x block order — a design decision that, silently wrong,
would produce a confident and false cell. Cells 21/29/30 need the gate to ask
the technique for a parent-calibration predictor instead of assuming a matmul.
Both are harness-lane design changes, not bug fixes, and neither should be made
mid-wave by the lane running the cells. The receipts carry the exact error text;
that is the wave-1 result for those four rows.

N10 (standalone norm refit) and the first-ever C04/C10 measurements therefore
remain **unmeasured**, and the reason is the harness, not the technique.

### 4.6 Smaller things the receipts said

* **The free-run ordering is not the composed ordering.** C09.T5 (drift 0.02002)
  and C09.T8 (0.02102) are 5% apart on composition and 3 points apart on
  free-run, while C13.T5.L00 (0.08937) and C13.T8.L00 (0.17910) are 2× apart on
  composition and 13 points apart on free-run. The composed-terminal metric
  orders cells the same way free-run does at a given site, and does not order
  them across sites.
* **C09.T5 landed at 0.02002 against a 0.02 gate** — the closest miss in the
  campaign, four decimal places from a different chain link.
* **Condition numbers on the T1 arm are enormous and never saturate the cap.**
  `240,109,510` at C06 and C07 (identical to the digit — both solve against the
  same `operator.input` design), `963,110` at C09, `2,417,441` at C13/L05. All
  finite, all `condition_cap_saturated: false`, and C06.T1 is simultaneously the
  best-behaving cell in wave 1. Link 2 of the chain never fired in 27 cells.
* **T7's alternating lane confirmed its own finding on real evidence.**
  `alternating_improved_on_closed_form: false` at both sites, with the iteration
  actually run and rejected (`iterations_accepted: 0` out of a cap of 8) —
  `NOTES-MX-B` §2.3 predicted this from synthetic data and it holds on the
  sealed cache. Score-kernel relative error 0.0076 fit / 0.0081 held out,
  Procrustes orthogonality deviation ~1.7e-14, partner `partner_is_copy_exact:
  true`.
* **`C13.T5/T1` at L05 calibrate at 0.000638**, a third of the other sites'
  floor. The bf16 quantization floor is endpoint-dependent, which is one more
  reason the 5e-3 tolerance is a ceiling over a measured band and not a target.

---

## 5. Ledger, races, and the merge

Two runners appending to one ledger is a real race, not a theoretical one:
`append_ledger` opens the file `"ab"` and issues **two** `write()` calls — the
canonical row, then `b"\n"`. `O_APPEND` makes each atomic; it does not make the
pair atomic. Two concurrent runners can interleave a newline into another
runner's row.

So each GPU batch wrote its own shard and a **single serialized appender**
merged them (`reports/matrix-campaign-20260810/run/merge_shards.py`):

* shard lines are byte-concatenated, never re-serialized, so a merge cannot
  normalize a sealed row;
* rows are ordered by `sealed_at` with a byte tie-break, so the merge is a total
  order and reproducible;
* the merge **refuses** unless the authority's existing bytes are a prefix of
  the result, and it re-verifies the prefix after writing.

```
rows 14 -> 41 (+27)
prefix verified byte-for-byte
```

Then 23 reviews, sealed from one process, 41 → 64. Repo mirror and server
authority agree at `ef1b271a8c7a5c1f`, and the pre-wave mirror is a byte prefix
of it. All 61 receipts in the repo re-verify their seals.

---

## 6. Backups and the destructive-change check

| what | where |
|---|---|
| ledger before the wave | `/mnt/ai_data/neural-forge-matrix/backup/ledger.jsonl.before-mxe-20260810T180150Z` (`d1574143bef80811`, 14 rows) |
| ledger before the shard merge | `…/backup/ledger.jsonl.before-mxe-merge-*` |
| ledger before the 23 review appends | `…/backup/ledger.jsonl.before-mxe-reviews-*` |
| the deployed runner, pre-patch | `…/backup/deployment-before-mxe-20260810T180150Z/run_lfm2_matrix_cell.py.pre-mxe` (`f72f5511fb326fff`) |
| repo mirror ledger + LEDGER.md, pre-wave | scratch (`ledger.jsonl.before-mxe` `d1574143bef80811`, `LEDGER.md.before-mxe` `11c3275732fb94f7`) |
| the five quarantined P40 receipts and their shard | `/home/cereal/nf-mxe/discarded-p40/` |

| check | result |
|---|---|
| ledger rows | 14 → 41 (cells) → 64 (reviews); **no row rewritten**, prefix verified after every append |
| mirror vs authority | byte-identical, `ef1b271a8c7a5c1f` |
| receipt seals re-verified in the repo | **61 / 61**, no failures |
| sealed evidence base `2dec87ac…` | opened read-only; 17 per-(site, component) caches were built through the runner's authenticated selective session, which hashes every batch and calls `session.verify()` before the payload is written, and 10 further cell runs reused a cache only after re-deriving its payload digest against the recorded binding |
| P40 rows in any campaign ledger | **zero** |
| `/` free | 14 GB → 9.2 GB (the 17 evidence caches, ~250 MB each, under `/home/cereal/nf-mxe/`) |

---

## 7. Verification

```
# local, .venv (python 3.11.9)
python -m pytest tests/test_run_lfm2_matrix_cell.py -q
# -> 20 passed        (16 + 4 new; one pre-existing test rewritten)

python -m pytest tests/test_run_lfm2_matrix_cell.py \
                 tests/test_run_lfm2_matrix_cell_fresh_evidence.py \
                 tests/test_matrix_cell_receipt.py \
                 tests/test_matrix_technique_registry.py -q
# -> 87 passed

python -m ruff check experiments/run_lfm2_matrix_cell.py tests/test_run_lfm2_matrix_cell.py
# -> 2 errors, both pre-existing at HEAD (verified by linting the HEAD blobs)
```

Server: the twelve-file deployment digest, the pre/post-patch inertness re-run
(97 fields, 0 substantive diffs), 27 cell runs, the shard merge, 23 review seals,
the ledger render, and the free-space and read-only-evidence checks.

**Not verified:** no cell at M8, M4 or X4; no second site for C06/C07 T1, so
neither pass has cleared §5 rule 2; no T2/T3/T6/T9 cell; no knob other than the
registered defaults except `writer_rank: 31` on the two T4 cells; C03, C04, C10
and C14 have no evidence at all. The three passes are single-site, single-level
results on a single component family.

## 8. Paths

* wave-1 cells 4–30: `reports/matrix-campaign-20260810/cells/<cell-id>/`
  (`cell-receipt.json`, `cell-receipt-reviewed.json`, `cell-diagnostics.json`,
  `cell-<cell-id>.log`) — 27 new directories. `C05.T1.M16.L00` ran before the
  batch driver existed and streamed to stdout, so it carries the post-patch
  control run's log instead of its own.
* ledger + rendered report: `reports/matrix-campaign-20260810/{ledger.jsonl,LEDGER.md}`
  (mirrors `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`)
* run inputs: `reports/matrix-campaign-20260810/run/{mxe-knobs.json,mxe-knobs-t4-rank31.json,run-mxe-batch.sh,merge_shards.py}`
* harness patch: `experiments/run_lfm2_matrix_cell.py:309` + five tests in
  `tests/test_run_lfm2_matrix_cell.py`
* campaign store: `/mnt/ai_data/neural-forge-matrix/cells/`, run logs
  `/home/cereal/nf-mxe/logs/cell-*.log`, backups
  `/mnt/ai_data/neural-forge-matrix/backup/` (see §6)
