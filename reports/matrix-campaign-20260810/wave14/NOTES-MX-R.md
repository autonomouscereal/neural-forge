# NOTES-MX-R — wave 14: the export requirement changed what "0.703x" means, and the first genuinely smaller child is dead

Wave 14, branch `codex/lrc-blank-compression`, source commits `cd52bf59`
(slate + ADDENDUM 11), `0bba8861` (battery, pre-registered), `8fb2cdfd`
(refit/control battery + narrow builder), on top of wave-13 HEAD `02096bd7`.
Review surface verified on every review before a token was generated: protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`, prompt pack
`0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`.

Wave 14 ran the §4.3 measurement battery in full, then built the first child in
this campaign that is **architecturally smaller** rather than smaller-in-storage.
Two of the three sentences below are negative and the third is the one that
matters.

*§0's sentences are at the bottom.*

---

## 1. The finding that re-prices every previous wave

`mxq_materialize_child.py:120` refuses to install a folded tensor whose shape
differs from the parent's:

```
if tuple(value.shape) != tuple(target.shape):
    raise ... f"REFUSED: {key} is {tuple(value.shape)}, parent is {tuple(target.shape)}."
```

Every sealed child in this campaign is therefore **parent-shaped**. `T1`, `T4`,
`T5`, `T8` and `T7` all store a factored, sparse, tied or merged form and
reconstruct the parent's own shapes to run. The ratios the campaign quotes —
`COMP.MA.MXP2` 0.703409x, `MXO3` 0.729103x, `MXQ5` 0.710270x — are **storage-format
ratios**. The model that goes to disk and the model that runs are 229,693,184
parameters.

Under ADDENDUM 11 R3 that is not a frontier claim, and this note says so plainly
because no previous note had a rule that made it worth saying. It is not an
accusation of error: every one of those ratios is an honest count of stored
parameters under a storage model that was registered and tested. It is a
statement that **the campaign has been measuring a different quantity than the
one R3 asks for**, and that the two had not been distinguished.

Three axes on this checkpoint *are* config-standard, verified against the
installed `transformers` 5.12.1 rather than assumed:

| axis | config field | evidence |
|---|---|---|
| SwiGLU width | `intermediate_size` / `block_ff_dim` | `modeling_lfm2.py:138-149`; `block_auto_adjust_ff_dim` is false in this checkpoint, so the field is used verbatim |
| attention head width | `head_dim` | `modeling_lfm2.py:233` and `:109` — `getattr(config, "head_dim", hidden//heads)`, and RoPE reads the same field |
| depth | `num_hidden_layers` + `layer_types` | standard |

and one is **not**:

| conv channels | — | `Lfm2ShortConv.__init__` hardwires `config.hidden_size` for `conv`, `in_proj` and `out_proj` (`modeling_lfm2.py:312-321`). The `conv_dim` field in `config.json` is **not read**. |

So **F-C is a research probe under R3 and cannot be otherwise without a custom
module**, and that is a fact about the modeling file, not about the fold.

## 2. The measurement battery — every result, including the ones that killed a class

Ten censuses, fp64 CPU, reading the sealed 4,751/1,584 split and the parent
checkpoint. Floors were committed in `mx14_measure.py` **before it was run**
(`0bba8861`), which is the only thing that makes them floors.

### 2.1 Free kills

* **Vocabulary row tying (critique gap 1) — DEAD.** Tail rows are near
  orthogonal: nearest-neighbour cosine over a 4,096-row deterministic sample of
  the post-12,288 tail has median **0.2977** (p05 0.190, p95 0.678). A tie is
  only free if the shared row stands in for its members' logit direction and at
  cosine 0.30 it does not. This is the pre-registered kill, fired at zero cost.
  Row-norm mass is also nearly flat (participation ratio 0.979; the top 25% of
  rows carry 27.9% of the norm), so there is no verbatim head to build around
  either.
* **The G artifact — Track B dies below L10.** Terminal transfer surrogates
  `residual.post_layer[l] -> residual.post_layer[13]`, 14 ridge solves, held-out
  R²:

  ```
  L00 0.046  L01 0.136  L02 0.227  L03 0.268  L04 0.296  L05 0.311  L06 0.338
  L07 0.368  L08 0.396  L09 0.459  L10 0.615  L11 0.706  L12 0.874  L13 1.000*
  ```

  (*L13 is the identity and is excluded.) Against §4.1's pre-registered
  `R² > 0.5` floor, **three layers clear it and ten do not**. The per-endpoint
  transfers for the 2,560-dim writers are worse and mostly **negative** (L00
  −12.5, L05 −0.30, L12 +0.465): a ridge from the gated intermediate to the
  terminal residual does not generalize at all at 1.86x overdetermination.
  Ω-RIDGE, ANTIFOLD and XGRAM's error signatures are therefore **restricted to
  L10-L12** or they are pricing noise. §4.1 said this artifact was the common
  kill switch; it fired, partially, and the honest reading is that the terminal
  is not linearly predictable from the shallow residual — the depth grading is
  smooth and it is the identity carry that finally dominates at L12.
* **HODLR-V — alive at one component, and for the wrong reason.** Diagonal
  interaction coverage: C13/L05 **0.9287** (g=4) and **0.9138** (g=8); C13/L13
  0.523/0.389; C09/L12 0.538/0.368. The one site above the floor is the one site
  §2.3 shows is dominated by a single unit, so its "coverage" is that unit
  sitting inside whichever block it landed in. Class does not clear on merit.
* **S+L kurtosis screen — one component skipped free, the rest heavy.**
  D^{1/2}-scaled entry kurtosis: C13/L05 2,464,396; C13/L13 62,140; C05/L13
  38,480; C05/L05 1,092; C13/L09 459; C09/L12 44.6; C09/L08 17.7; **C08/L02
  5.94** — near-Gaussian, so `attn_v` at L02 is skipped free exactly as the
  screen was designed to do.

### 2.2 The two strong survivors

* **Score-bilinear q/k (critique gap 2) — the best spectrum in the battery.**
  The evidence-whitened score operator `Rt Wq_h^T Wk_g Rt` has rank ≤ 64 by
  construction; its top **16** singular directions carry a median of **0.9948**
  (L02), **0.9985** (L08), **0.9926** (L12) of the operator energy, worst head
  0.9414. Attention scores on this model live in a 16-dimensional subspace per
  head. Two stated limitations bound what that licenses: it is the `Δ=0` member
  of the RoPE family `q^T R_Δ k` (necessary, not sufficient — a joint
  factorization must serve every Δ, and contracting `head_dim` remaps the RoPE
  frequencies), and it uses raw projections rather than the `q_layernorm` /
  `k_layernorm` outputs. Not built this wave; it is the strongest unspent lead.
* **The verbatim channel coreset beats the low-rank basis on the SwiGLU write,
  everywhere.** This is the matched-storage control (§2.4) and it is the wave's
  one clean positive law.

### 2.3 L05 is one enormous unit, and the OFF certificate is withdrawn

The first pass reported 1,915 certified-OFF units at L05 and zero at every other
site, and L05's Frobenius residual looked six times better than any other site.
Both are the same artifact. At L05 a **single** intermediate unit carries
**97.15%** of the layer's gated-output energy — max/median energy ratio
**2.142e5**, `|phi|` max 22.1 against median 0.0024 — against 0.047 top-1 share
at L04 and 0.019 at L06.

Consequences, all three worth carrying forward:

1. **`OFF_EPSILON` is a share-of-total test and is not scale-robust.** It cannot
   distinguish "this unit is dead" from "one other unit is enormous". Zero units
   at L05 are actually dead by a scale-free test (`activation < 1e-3` of the
   layer median: **0** units, at every site measured). **VC-FOLD arm E's free
   drop is withdrawn** — there is no free allowance to manufacture.
2. **Frobenius-relative error is not a usable ranking metric at a massive-activation
   site.** L05 reports relL2 0.0798 at S4 and cosine 0.8645; the cosine is in line
   with every other site and the relL2 is six times better, because the metric is
   reporting that the one huge row was captured. Wave 14 quotes **both** and ranks
   on the cosine.
3. It explains the S+L kurtosis of 2.46e6 and the HODLR-V coverage of 0.93 at the
   same component. One unit is doing all three.

The pooled residual spectrum agrees from the other side: the top single direction
of `Sigma_hat` carries **40.53%** of residual energy (top-16 73.7%, rank-for-90%
224).

### 2.4 The matched-storage control — the comparison the campaign has wanted since wave 5

At every site and level, the verbatim channel coreset (kept columns + re-solved
writer) was run against an evidence-weighted low-rank approximation of the same
map at **identical stored-parameter count**, predicting the same target from the
same features on the same held-out rows. One variable.

```
C13 (SwiGLU write), 14 sites x {S4, S8, S16}:  coreset wins 40, LRA wins 2
C05 (conv write),    5 sites x {S4, S8, S16}:  coreset wins  3, LRA wins 12
```

**Verbatim channels beat a low-rank basis on the SwiGLU write and lose to it on
the conv write.** The wave-13 sentence "verbatim coverage beats reconstruction"
was a token-interface result; it extends into the body's *gated* feature space
and does **not** extend to the conv context. Representative cosines at S4:
L07 coreset 0.8720 vs LRA 0.8212; L00 0.8579 vs 0.8064; L13 0.9398 vs 0.8944.
Conv at S4: L07 0.8429 vs 0.8718, L05 0.8701 vs 0.8818.

### 2.5 F-EXCISE — one block out of thirteen clears, and it is the wrong one

Held-out absorbed fraction of the deleted block's own contribution, by the
upstream FFN writer's re-solve, against the pre-registered 0.50 floor:

```
L05 conv +0.6653 ALIVE   L13 conv +0.4669   L12 attn +0.3199   L08 attn +0.1360
L11 conv +0.0959   L10 attn +0.0788   L04 attn +0.0345   L06 attn +0.0342
L09 conv +0.0269   L07 conv -0.0015   L03 conv -0.0909   L02 attn -0.6331
L01 conv -1.2103
```

Only **L05** clears, and L05 is the massive-activation block, so what the ridge
is absorbing is one dominant direction. Four blocks are **negative**: at L01 and
L02 the re-solved upstream writer is *worse* than leaving the parent's writer
alone, which is the sealed decay-law objection landing in the data. The class
does not have thirteen candidates; it has one, and that one is explained.

### 2.6 F-B / value-output contraction — real channels do track the SVD floor

Per-head interpolative residual of the sealed context under the top-16 real value
channels, against the same-budget SVD floor: ratio median 1.129-1.202 across
L02/L04/L08/L10/L12, max 1.832. Real channels are within 25% of the optimal
subspace at every site — the pre-registered bar — so the class does **not** close
causally.

Measured as installed (per-KV-group rank-16 bottleneck, every `Wo` slice exactly
re-solved, attention pattern bit-identical):

```
        S4 (r=16)                    S8 (r=8)
L02  relL2 0.3454 cos 0.9384    relL2 0.4244 cos 0.9052
L04  relL2 0.5293 cos 0.8489    relL2 0.6469 cos 0.7631
L08  relL2 0.5474 cos 0.8397    relL2 0.6436 cos 0.7686
L10  relL2 0.5391 cos 0.8544    relL2 0.6324 cos 0.7904
L12  relL2 0.3909 cos 0.9287    relL2 0.4600 cos 0.8978
```

against the sealed `C09.T4.S4.L02` at cosine 0.9605 — but that lane folds `o`
alone and this folds `v` **and** `o` at 4x on the pair, so the two are not the
same purchase and the receipt says so.

## 3. The first architecturally smaller children, and what they did

`mx14_narrow.py` writes a stock `Lfm2ForCausalLM` with a smaller
`intermediate_size`. Nothing is hooked, nothing is reconstructed at runtime, and
the child is loaded back with `AutoModelForCausalLM.from_pretrained` and **no**
`trust_remote_code` before any review.

**`intermediate_size` is one config field shared by all fourteen blocks**, so a
config-consistent child folds every MLP or it is not config-consistent. The big
bite ADDENDUM 11 §11.2 D2 asks for is therefore *forced* here rather than chosen,
and so is its granularity: there is no seven-site rung in this lane.

Two guards fired for real and are worth the space:

* **The identity precondition.** Every site checks `post_swiglu @ w2^T` against
  the sealed `mlp.down` before selecting anything, refusing above 5e-3. It passed
  at all fourteen sites; had the endpoint been anything but the gated
  intermediate, every selection would have been on the wrong axis.
* **THE LAW.** L00's free solve moved **1.556** of its prior's norm and was
  **refused**. The registered remedy is the trust region, not an exception: λ was
  raised on a pinned ladder to 1.0 relative, bringing `update_scale` to 0.457 at a
  cost of 0.004 cosine (0.8564 free → 0.8524 compliant). L01 needed λ=0.1
  (1.110 → 0.781), L02 needed λ=0.01 (1.027 → 0.984); the other eleven sites were
  compliant free. Both solves are sealed in the build receipt so the guard's cost
  is visible.

| child | keep | net | stock-load | child cap-hits | verdict |
|---|---:|---:|---|---:|---|
| `MX14-ff640` | 640 | **0.640497x** (147,117,824) | `Lfm2ForCausalLM`, no TRC | **7 / 16** | **dead, 0/15** |
| `MX14-ff320` | 320 | **0.580580x** (133,355,264) | `Lfm2ForCausalLM`, no TRC | **16 / 16** | **dead, 0/15** |

Parent cap-hits on the same surface in the same process: **0 / 16**.

`MX14-ff640` is the one worth reading, because it fails in a specific way. It is
**fluent and empty**: it produces well-formed English that restates the prompt
instead of answering it, at every single case.

```
explanation-seasons  : "The answer is a question about the Earth's hypothesis to be
                        a scientist, and a scientist who would explain the explanation."
planning-dependencies: "The answer is a dependency-aware plan that covers backups,
                        DNS, TLS, verification, and rollback. Keep it to six steps."
logic-ordering       : "Mira finished the final product after the final product."
```

The `planning-dependencies` response is the diagnostic one: it names the plan's
required contents and the step budget *from the prompt* and then stops, having
produced no plan. Seven cases instead run to the cap on a repeated token
(`"17 kits per per per per…"`, `"three three three other…"`, `"list list list…"`),
and under ADDENDUM 8's degeneracy clause four of those leave under 350 characters
when repetitions are removed. **No case is Useful and no case is Partial.**
`MX14-ff320` is worse in the only way left: 16 of 16 at the cap.

**This is the wave's central negative and it is a real physics result.** Every
one of the fourteen folded sites has a held-out write cosine of **0.81-0.94**,
which is the same band as the sealed lanes that compose into coherent children
(`C13.T8.S4.L05` is 0.8587). Per-component fidelity in the alive band, at every
site, composed to **0/15**. What kills the child is not any site's error; it is
fourteen of them at once. Nothing in the per-site census could have predicted
this, which is exactly why ADDENDUM 8 says children are built and read.

## 4. Bisection, and the width cliff it located

D2's rule on death is bisect, not abandon. In this lane the site count is not
bisectable — `intermediate_size` is global — so the bisection is on **width**:
`MX14-ff1280`, keep 1,280, net **0.760331x** (174,642,944 parameters, stock load
verified). It is a **2x** fold per site and therefore a **sub-4x diagnostic**
under ADDENDUM 6, never a frontier claim. It exists to locate the cliff, and it
located it.

| child | keep | per-site write cos (median) | net | cap-hits | reading |
|---|---:|---:|---:|---:|---|
| `MX14-ff1280` | 1280 | **0.925** | 0.760331x | **3 / 16** | degraded, **not** degenerate — ~5/15 |
| `MX14-ff640` | 640 | **0.856** | 0.640497x | 7 / 16 | fluent and empty — **0/15** |
| `MX14-ff320` | 320 | **0.781** | 0.580580x | 16 / 16 | total collapse — **0/15** |

**The cliff sits between a per-site write cosine of 0.93 and 0.86, when the fold
is applied at all fourteen sites at once.** Seven hundredths of cosine, at every
site simultaneously, is the difference between a model that writes structured
on-topic prose and one that restates its prompt and loops. That is the wave's
quantitative answer to "how much does composition cost": not a drift budget, a
*fidelity floor* that is far tighter under simultaneous installation than any
single-lane number suggests. The sealed alive lanes sit at cosine 0.86 — right at
the bottom of the dead band here — and they survive only because they are
installed three at a time, not fourteen.

`MX14-ff1280` is worth reading rather than summarizing. It keeps document
structure, headings and lists; it restates source material accurately
(`summarization-priorities` reproduces all three suppliers and the 11% figure);
and it answers the epistemic case properly (`uncertainty-insufficient-data` asks
for time range, units and sample size). It also gets `explanation-seasons`
factually wrong in the classic way ("Earth is closer to the Sun"), emits an
off-by-one that is still off by one (`return values[start - 1:]`), and on
`multi-turn-correction` — where the user has just corrected for a **severe peanut
allergy** — recommends "**Peanuts**" four times. That last one is not a coherence
failure, it is an instruction-following failure with a safety edge, and it is
recorded here because a count would have hidden it.

**My reading is 1 Useful, 4 Partial, 10 Miss — about 5 of 15**, against the
parent's 12/15 on the same surface in the same process. That places it in the
`impaired` band, the same band `MXP2` (6/15) and `MXQ1` (6/15) occupy — at a
**sub-4x** fold, so it buys nothing the campaign can claim. **No machine verdict
was sealed for any of the three children**, because there is no receipt to seal
one against (§6), and that gap is stated rather than papered over with my own
count.

## 5. What wave 14 did not spend, and why

* **Head merge (T7/C10) in succession.** ADDENDUM 11 §11.2 D1 returns it to the
  deck and the plan was merges one at a time and at S8. It was not reached. The
  reason is scheduling, not a verdict, and the S8 rung (`g_q=1, g_kv=1`,
  0.114583x) remains the deepest unexpressed rung the campaign has a kernel for.
  **Note for whoever spends it:** under R3 a head merge exports to a *same-size*
  checkpoint, because it reconstructs every projection parent-shaped. Its
  0.2396x is a storage-format number.
* **The in-process benchmark fix** (`HFLM(pretrained=<module>)`, NOTES-MX-Q §6).
  Not needed by either child built here — a narrow child is a directory and
  lm-eval loads it natively — so it was not built. It is still owed to the
  hook-carrying children.
* **Benchmarks.** No wave-14 child was benchmarked, under the standing rule that
  a child which loops a token to the cap has no intelligence-per-compression to
  measure. Both graded children loop.
* **XGRAM, ANTIFOLD, Ω-RIDGE, W-ABSORB-SB, gap-3's global KKT solve, gap-4's
  attribution artifact.** All consume the G artifact, which cleared its floor at
  three layers of fourteen. Running them on L10-L12 only is possible and was not
  reached; running them anywhere else would be pricing noise.
* **CBT, F-DECAY, F-LANDMARK, FSS, F-D, Regime-4b, Gate-Jacobian.** Not reached.
  F-DECAY additionally has an unmet precondition: the sealed rows carry a phase
  label but not a position index, so causal decay features cannot be built from
  them without a recapture.

## 6. Estate

* **Disk.** Server `/` was at **3.6 GB** free at wave start. `deployments/` was
  pruned to current+1 (`neural-forge-matrix-63e95cd8`, `neural-forge-matrix-4d8774f6`),
  57 entries deleted totalling **9.6 GiB**, manifested at
  `wave14/deployments-prune-manifest-20260812.json` with per-entry byte counts,
  file counts and mtimes. `neural-forge-postpub-e893767a` was **not** a pure
  extraction — 640 of its files were written after its extraction date — so those
  640 files were archived to
  `wave14/salvage/postpub-e893767a-runoutputs.tar.gz` (1,701,351,664 bytes,
  sha256 `09fb5b7f3cddc7adf9f629ba30bf65b42cc86b09289b7276ff5f99caa2339dd5`,
  listing verified at 640 entries) **before** the directory was removed. `/` ended
  at **14 GB** free. Every wave-14 artifact is under
  `/mnt/ai_data/neural-forge-matrix/wave14/`; the wave's `/` footprint is the
  `nf-mx14` workspace (six scripts and three logs, well under 1 MB).
* **Deployments.** None cut. Wave 14's scripts are standalone — they import
  `torch`, `safetensors` and, for the review, the wave-13 deployment's own
  `run_lfm2_composition_frozen_review` — so no child was generated from an
  unverified tree. The review surface digests were checked on **every** review
  before a token was generated.
* **Ledger.** **No row appended.** The authority on the run host is unchanged at
  461 rows, sha
  `4ad102d32efbdf59008427db3b1ccd18b54f4cb1bc753f0d286e036bbbb56d57`, verified at
  wave end. One repair was made and it was additive only: **the repository copy
  was stale at 447 rows** (sha `a1c6b924…`) — wave 12 committed its five rows and
  wave 13 never committed its fourteen. The repo bytes are an exact **byte-prefix**
  of the host's, 705,287 → 729,335, **24,048 appended**, which is precisely the
  count `NOTES-MX-Q` §8 recorded for the wave-13 merge. Backed up first
  (`backup/ledger.jsonl.pre-wave14-repo-sync-20260812`, verified by sha **and**
  `cmp`), synced, prefix re-checked afterwards, and the committed object verified
  byte-identical with
  `git cat-file -p HEAD:reports/matrix-campaign-20260810/ledger.jsonl | sha256sum`
  → `4ad102d3…`. Wave 14 sealed no receipts and appended no rows: the
  narrow children are not `CellId` objects — they replace no parent tensor
  shape-for-shape and carry no `weight_keys` — so appending them would be a
  schema violation of exactly the kind ADDENDUM 10 was written to avoid. **The
  registry work to give a shape-changing child a receipt is the first item wave
  15 owes.**
* **Seals.** **0 fail.** Two verifiers were run and they disagree by one row, so
  both numbers are here: wave 13's `mxq_verify_seals.py` reads **456 / 456**
  resolvable with five non-resolving, and wave 14's `mx14_verify_seals.py` reads
  **457 / 457** with four. The single row of difference is
  `C06.T2.S4.L12 1cf2c017…-a6review`, a **duplicated `cell_instance_id`** — two
  ledger rows pointing at one receipt. Wave 13's index counts that receipt once;
  wave 14's resolves both rows to it. Neither is wrong and **neither changes the
  fail count, which is zero either way**; the duplicate is a pre-existing ledger
  fact, not something this wave created.
  **The roots defect is now visible in git.** The *committed* wave-13 verifier is
  missing `wave12/cells` — `NOTES-MX-Q` §8 says wave 13 caught it, and it did, on
  the host; the fix never reached the repository. `mx14_verify_seals.py`
  therefore audits its roots against the directories that actually exist and
  **refuses to run** on an undeclared one (10 present, 0 undeclared). No wave-14
  rows exist to seal.
* **Human-review package.** `wave14/human-review/` — both dossiers with full
  verbatim child and parent responses, the degeneracy view beside each, and the
  build receipts.
* **Commits.** `cd52bf59`, `0bba8861`, `8fb2cdfd`, and the sealing commit for
  this note. Nothing pushed, nothing rebased, staged by explicit path only; the
  Codex lane's ~331 uncommitted paths, including the frozen
  `src/neural_forge/hf_lfm2_circuit_trace_cache.py`, were not touched.

## 0. The sentences

1. **The campaign's frontier ratios are storage-format ratios.** Every sealed
   child is parent-shaped, by a refusal in the materializer, so 0.703409x is a
   count of stored parameters and not a smaller model. Three axes on this
   checkpoint are config-standard (`intermediate_size`, `head_dim`,
   `num_hidden_layers`); conv channel count is **not**, because
   `Lfm2ShortConv` hardwires `hidden_size`.
2. **The verbatim channel coreset beats a matched-storage low-rank basis on the
   SwiGLU write at 40 of 42 site-levels, and loses to it on the conv write at 12
   of 15.** The verbatim law extends into gated feature space and stops at the
   conv context.
3. **Fourteen SwiGLU coresets at once, every one of them inside the sealed alive
   fidelity band, compose to 0/15.** `MX14-ff640` is 0.640497x, loads as a stock
   `Lfm2ForCausalLM`, and answers no question on the frozen surface — it restates
   the prompt fluently and then loops. Per-site fidelity did not predict it and
   was never going to.
4. **L05 is one unit.** A single intermediate channel carries 97.15% of that
   layer's gated-output energy, which manufactured a false OFF certificate
   (1,915 units), a false six-fold Frobenius advantage, a kurtosis of 2.5e6 and
   HODLR-V's only passing coverage number. The OFF free-drop is withdrawn and
   Frobenius is not a ranking metric at that site.
5. **The G artifact cleared its floor at three layers of fourteen**, so Track B
   is not dead but is restricted to L10-L12, and the per-endpoint transfers for
   2,560-dim writers are negative and price nothing at all.
