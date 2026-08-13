# NOTES-MX-Y — wave 21: the cliff has a slope, the lambda split trades cases, and the worst reader is not the variable

Wave 21, branch `codex/lrc-blank-compression`, source commits `591b400e` (the
wave-21 builder: the max-min basis objective, the per-endpoint read census, the
free endpoint mode) and `f336b04d` (two defects the self-test found before a GPU
saw the objective), plus the sealing commit for this note. One deployment built,
sealed and reviewed every child: `neural-forge-matrix-f336b04d`. Protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b` and prompt
pack `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b` verified
digest-true in the deployment on every dossier.

**Three calibrations passed before anything was graded.** The fresh conditioned
capture reproduces the sealed collector to waves 15, 19 and 20's numbers exactly
(L00 relL2 3.367e-3 cos 0.9999944, L05 1.647e-2 / 0.9997207, L13 2.034e-2 /
0.9997842). The parent's sixteen frozen-review responses digest to
**`865f006e3813712ec202cb23`** on all five wave-21 dossiers. And the parent's
benchmark row reproduces the sealed reference in all seven columns (0.5450 /
0.5050 / 0.6800 / 0.5500 / 0.4600, lambada ppl 24.39, wikitext 81.80) — the
**eighth** wave to re-run it and the eighth to match, which the consolidated
report now states as a campaign-level fact rather than a per-wave one.

*§0's sentences are at the bottom.*

---

## 1. The free measurement, and the number nobody had written down

Wave 19's law and wave 20's bound are both about **retained reader-projected
residual energy**, and every number either wave quoted is **pooled** over the 29
endpoints where a stock LFM2 reads the stream. A pooled number is a mean. Nobody
had ever printed the per-endpoint read retention, because the census records the
per-endpoint moment for the **raw** residual only — the currency wave 20 proved
is the wrong one.

One capture pass, no build:

| sealed basis | pooled read (hld) | worst endpoint | where | spread | endpoints under 0.90 |
|---|---:|---:|---|---:|---:|
| `E-reader-pca-896` | 0.9627 | 0.9440 | `B01.ffn` | 0.0307 | **0 of 29** |
| `E-reader-pca-768` | 0.9193 | **0.8801** | `B01.ffn` | 0.0651 | **4 of 29** |
| `E-reader-pca-736` | 0.9075 | 0.8632 | `B01.ffn` | 0.0739 | 9 of 29 |
| `E-reader-pca-640` | 0.8697 | 0.8084 | `B01.ffn` | 0.1032 | 26 of 29 |
| `E-reader-pca-512` | 0.8124 | 0.7268 | `B01.ffn` | 0.1458 | 29 of 29 |

**The pooled number does hide readers, and the living 3/15 child has four of
them under the floor.** `ARCH.W768` sits at 0.9193 pooled with `B00.ffn` at
0.8961, `B01.op` at 0.8905, `B02.ffn` at 0.8833 and `B01.ffn` at **0.8801** —
below the 0.8953 at which `ARCH.W704` is 0/15 and below the 0.8967 at which
`ARCH.W768AW100` is 0/15. The gradient is **depth-ordered and stable**: blocks 0
to 3 are the badly-retained ones at every width, blocks 6 to 9 the best, and the
argmin is `B01.ffn` at all five widths.

So the hypothesis was not falsified by its own free measurement, which is the
opposite of what happened to wave 20, and the build was warranted.

## 2. The attack, and what it cost

`--basis reader-maxmin` maximises `min_i tr(E^T M_i E)` over the 29 trace-one
endpoint reader moments instead of their mean. On the convex hull of rank-k
projections the objective is concave, so by Sion's minimax theorem it equals
`min_w [sum of the top k eigenvalues of sum_i w_i M_i]` over the endpoint
simplex — convex in `w`, with the vector of per-endpoint retentions as its
subgradient. The iteration is therefore one eigensolve and one exponentiated-
gradient step per round, needs no capture of its own, and **at uniform weights
returns the plain reader-PCA basis bit for bit**, which the self-test pins with
`torch.equal`.

It does exactly what it was built to do:

| at width 768 | plain reader-PCA | reader-maxmin |
|---|---:|---:|
| worst endpoint, read, held-out | 0.8801 | **0.8997** |
| endpoints under the 0.90 floor | **4 of 29** | **1 of 29** |
| pooled read, held-out | 0.9193 | 0.9137 |
| directions exchanged | 0 | **39.44 of 768** |
| final-norm relative logit SSE | 0.1752 | **0.1522** |
| review | **3/15** | **0/15** |

**The max-min child reconstructs the parent's own logits better than the living
child does and every one of its sixteen lanes returns "I am a" loops.** The
summarisation lane — which survives at 896, at 768 plain, at 768 with the lambda
split, at 768 stale, and in both 896 compositions — does not survive this one.
The weight the iteration moved is exactly where the measurement said to move it
(`B02.ffn` 0.129, `B01.ffn` 0.117, `B00.ffn` 0.084, `OUT.final` 0.067), so this
is not a failure to optimise. It is the objective being wrong.

**And the benchmark agrees, which the last two waves' negatives did not.**
`MM768` scores **0.0000** on lambada with a perplexity of **23,120** against the
plain child's 8,177 — the worst distribution row of any child ever benchmarked
in this campaign — and is below the plain child on arc_easy as well. Wave 20's
`W768AW25` was the awkward case where the accuracy columns preferred the 0/15
child; wave 21's negative is unambiguous on both instruments.

**So the floor is not the binding constraint.** Five objectives have now been
tried against the pooled reader moment's own top eigenvectors — case-weighted at
two betas, evidence-PCA, reader-generalised, and now max-min over readers — and
**every one of them loses**, at exchanges from 17 to 231 directions, whether it
raises the minimum, lowers it, or moves the currency. The two constructions that
keep the pooled subspace exactly are the only ones that read 3/15.

## 3. The band, measured: the cliff is a slope

`ARCH.W736`, 0.689220x, retained read energy **0.9075** pooled, **1/15**.

| width | net | retained read (hld) | review |
|---:|---:|---:|---:|
| 896 | 0.859023x | 0.9627 | **4/15** |
| 768 | 0.722610x | 0.9193 | **3/15** |
| **736** | **0.689220x** | **0.9075** | **1/15** |
| 704 | 0.656116x | 0.8953 | 0/15 |
| 640 | 0.590763x | 0.8697 | 0/15 |
| 512 | 0.463481x | 0.8124 | 0/15 |

**The cliff is 32 dimensions wide and it is not a cliff.** Wave 19 called the
768→640 collapse "a step, not a slope" and bracketed it at 128 wide; wave 20
halved that to 64; the rung inside it reads **1/15**, so the sequence 4 → 3 → 1 →
0 is monotone and graded, and the last interval that could still hide a step is
736 to 704 — 32 dimensions and 0.0122 of retained read energy.

What `W736` keeps is one case and it is worth quoting, because it is the only
thing between this child and the empty string:

> Good catch, I am planning for 8 people. I am planning for 4.

against `ARCH.W704`, which returns the empty string in every lane. The
summarisation lane, held by every living width child before it, is gone: `W736`
names both suppliers and the 11% figure and never reaches a decision.

## 4. The lambda split, composed both ways, and it is not monotone

Wave 20 left the writers' internal fit-side lambda split un-composed with
anything. Wave 21 composed it upward and sideways, and the answer is different in
all three places:

| where | control | with the split | delta |
|---|---:|---:|---:|
| 768 (wave 20) | `W768` 3/15 | `W768LS` **3/15**, different cases | **0**, +1 capability |
| 896 | `W896` 4/15 | `W896LS` **3/15** | **−1**, +1 capability |
| 896 + block 13 excised | `W896X13` 1/15 | `W896LSX13` **2/15** | **+1** |
| 736 (the cliff) | `W736` 1/15 | `W736LS` **0/15** | **−1** |

**The capability it buys is the same one every time, and it is the axis's own
signature zero.** At 768 it recovers `context-retrieval`; at 896 and in the
composition it recovers `planning-dependencies` — the other case the attribution
artifact names as a width-axis zero. In the composition the recovery is legible
side by side, because the two children are the same 187,214,592 parameters and
their summarisation lanes are **byte-identical**:

* `W896X13` (wave 19), planning lane: "Identify the backend and target backend"
  ten times, naming nothing from the task.
* `W896LSX13` (wave 21), same lane: "Identify the new domain and targeted backup
  / Identify the backup / Identify the TLS / Identify verification / Identify
  rollback / Implement backup / Implement TLS / Implement verification" — the
  parent's own dependency order, missing only DNS.

So the split is **not a score improvement and not a score regression**; it is a
re-allocation whose sign depends on what the child had left to lose. At the
living width it costs a case, at the cliff it costs the last one, and inside a
composition — where the control had one case — it pays.

## 5. Benchmarks

ADDENDUM 8 invocation unchanged (limit 200, seed 1234, batch 4, max_length 2048,
float32, sm86):

| model | net | review | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `W896` (wave 19) | 0.859023 | **4/15** | 0.3550 | 0.4000 | 0.5650 | 0.4900 | 0.1450 | 638.72 | 500.08 |
| `W896LS` | 0.859023 | **3/15** | 0.3650 | 0.4050 | 0.5650 | 0.5100 | 0.1400 | **622.20** | **487.20** |
| `W896X13` (wave 19) | 0.815064 | 1/15 | 0.3750 | 0.4000 | 0.5200 | 0.4950 | 0.1500 | 1671.03 | 923.07 |
| `W896LSX13` | 0.815064 | **2/15** | 0.3600 | 0.4100 | 0.5200 | 0.4850 | 0.1400 | 1675.13 | 915.75 |
| `W768` (wave 19) | 0.722610 | **3/15** | 0.3100 | 0.4300 | 0.5200 | 0.4850 | 0.0300 | 8176.90 | 1279.46 |
| `MM768` | 0.722610 | **0/15** | 0.2900 | 0.3850 | 0.5000 | 0.5150 | **0.0000** | **23120.09** | 1888.09 |
| `W736` | 0.689220 | **1/15** | 0.2850 | 0.4050 | 0.5150 | 0.4800 | 0.0150 | 14146.56 | 1787.96 |
| `W736LS` | 0.689220 | 0/15 | 0.3050 | 0.4000 | 0.5050 | 0.4650 | 0.0200 | 13488.58 | 1815.87 |

**The review and the distribution disagree for the third wave running, and this
time on the lambda split.** `W896LS` is better than the sealed 4/15 child on
`arc_easy`, `hellaswag`, `winogrande`, lambada perplexity **and** wikitext — six
of seven columns — and holds one fewer review case. Under behavioural gating the
split costs a case at 896; a wave reporting the perplexity column would have
called it a free win.

## 6. Estate

* **Commits.** `591b400e` (the max-min objective, the per-endpoint read census,
  the free endpoint mode, the one-line schema addition in the ADDENDUM 12
  sealer), `f336b04d` (the two self-test repairs), plus the sealing commit for
  this note. Nothing pushed, nothing rebased, staged by explicit path only; the
  Codex lane's uncommitted paths, including the frozen
  `src/neural_forge/hf_lfm2_circuit_trace_cache.py`, were not touched.
* **Self-test.** `mx21_selftest.py` **24/24**, no model loaded — waves 19 and
  20's seventeen plus seven of this wave's own, including that step zero of the
  max-min iteration is `torch.equal` to the plain reader-PCA basis, that the
  selected basis never loses to it on the minimum, that it strictly wins when
  the endpoints disagree, and that it **cannot** beat the pooled basis pooled.
* **Two defects the self-test caught before a GPU ran.** Step zero was not
  exact — `pooled_moment` accumulates trace-one blocks in endpoint order and
  divides by the count, and a uniformly-weighted `einsum` differs from that by
  4.0e-14, which moves eigenvectors — and the strict-gain check was running on
  two random Wishart matrices, which inherit the same tilt, agree about which
  directions matter, and leave the objective nothing to arbitrate. Both are
  fixed in `f336b04d` and both are in the commit message.
* **Ledger.** **636 → 646**, sha
  `080ac0e8ef8608af9250b4301c625175fc0ad53edb40e0221369c1a80cdfb8c8`. Backed up
  first (`backup/ledger.jsonl.pre-wave21-20260813T111051`, verified by sha **and**
  `cmp`), merged exactly once behind `wave21/.MERGED_ONCE`, byte-prefix verified
  against the backup afterwards and again in the repository copy. Ten rows: 5
  ARCH run rows and 5 ADDENDUM 6 verdict rows, no superseded rows.
* **Seals.** `mx21_verify_seals.py --deployment <deployment>`: **17 roots
  present, 0 undeclared; 646 rows, 642 resolve, 642 verify, 0 fail, 4 do not
  resolve** — the same four pre-existing non-resolvers waves 14–20 named. The
  verifier was **excluded from the tree rename by name** and extended by one
  line, which is the fourth warning that file has issued and the second time it
  was obeyed as written.
* **The deployment was verified file by file for the first time.** It was cut as
  a hardlink copy of wave 20's plus a 930 KB delta, then checked against a
  manifest of the commit's own blob digests: **3,568 files, 0 content
  mismatches**, 714 declared absent (`tests/` and the repository-root notes and
  scripts, which no deployment has ever carried) and 7 files over 20 MB excluded
  by name. Every previous deployment's completeness was an assumption.
* **The deployment directory ends at the sealing commit and the receipts do
  not, and the difference is seven driver files.** Every child in this wave was
  built, sealed and reviewed from `neural-forge-matrix-f336b04d` and every
  receipt says so in `repo_head`. Four more driver files were written during the
  wave (`mx21_campaign_table.py`, `mx21_report_tables.py`, `queues/benchq.sh`,
  `queues/reviewonly.sh`) and three were finalised (`sealq.sh`, `package.sh`,
  `mx21_retain.py`), so the directory was re-cut to `d12c76db` afterwards and
  re-verified: **3,571 files, 0 content mismatches, 0 hardlinks**. `git diff
  f336b04d d12c76db -- src experiments` is **empty** — the code that sealed
  every receipt is byte-identical in both commits, and the whole drift is in
  scripts that write tables and copy files.
* **And the hardlinks broke the review, correctly.** The frozen review refused
  every child with `manual prompt artifact must be one ordinary file`: the guard
  is `st_nlink != 1`, and it is right, because a prompt pack reachable under a
  second name is a pack something else can rewrite. The links were broken, the
  deployment re-verified, and the four already-sealed receipts were **reviewed
  rather than re-sealed** — `reviewonly.sh` exists so that a re-run of step two
  does not put a second ARCH row in the shard, which is wave 19's double-seal.
* **Disk.** Server `/` began and ended at 12 GB free. Deployments end at
  current+1 (`f336b04d` and `1c0c7e3a`), with `e7d068da` pruned and named in
  `wave21/deployments-prune-manifest-wave21.json`. Five children were built and
  **all five are kept**: each is one arm of a controlled comparison whose other
  arm is sealed in an earlier wave, and the manifest says so per child.
* **Human-review package, in the repository.**
  `reports/matrix-campaign-20260810/wave21/human-review/` — 45 files.
* **And wave 15's package is committed at last.** The campaign's own frontier
  child, `ARCH.CL896`, had **no dossier rendering anywhere in the repository**:
  wave 15 built its package and left it on the server, and every wave since
  quoted that child's score without shipping its text. The 74-file package is
  now in `wave15/human-review/`, its parent lane digests to the same
  `865f006e3813712e`, and the consolidated report's scorecard links to it.

## 7. What this wave did not spend

* **No rung between 736 and 704.** The last interval that could hide a step is
  32 dimensions wide.
* **The max-min objective was not tried at 896.** The free measurement prices
  it (0.9440 → 0.9530 on the worst endpoint, 24.7 directions exchanged) and §2
  says what to expect; it was not built, because the width where the controls
  are is 768.
* **The lambda split was not tried on the OTHER composition axis.** It pays
  inside the excision composition and was not tried with the head merge, where
  wave 19's `W896KV6JS` sits at 2/15.
* **Nothing was tried between the pooled reader moment and the objectives that
  failed.** Five have now lost. §2's honest reading is that the search space
  around the pooled eigenvectors is not merely narrow, it is a point.

## 0. The sentences

1. **The pooled retention number hides readers, and the living child has four
   under the floor.** `ARCH.W768` reads 3/15 at 0.9193 pooled with `B01.ffn` at
   **0.8801** — below the retention at which three separate children are 0/15.
2. **Raising the worst reader does not buy behaviour; it costs everything.**
   The max-min basis lifts the worst endpoint from 0.8801 to 0.8997, cuts the
   endpoints under the floor from four to one, reconstructs the parent's logits
   **better** than the living child (relative logit SSE 0.1522 against 0.1752),
   and reads **0/15** against 3/15.
3. **Five objectives have now lost to the pooled reader moment's own top
   eigenvectors**, at exchanges from 17 to 231 of 768 directions, in every
   direction the objective can be moved. The two constructions that keep the
   subspace exactly are the only ones that read 3/15.
4. **The 0.90–0.92 band is measured and the cliff is a slope.** `ARCH.W736` at
   0.9075 retained read energy is **1/15**, so the width ladder reads 4 → 3 → 1
   → 0 and the last interval that could hide a step is **32 dimensions**.
5. **The deepest scoring child in the campaign is still wave 15's**, by 0.0008
   of net ratio and three review cases: `ARCH.CL896` at 0.688431x and 4/15,
   against wave 21's `ARCH.W736` at 0.689220x and 1/15.
6. **The writers' lambda split is a case trade whose sign depends on what is
   left to lose**: 0 at 768, **−1** at 896, **−1** at the cliff, and **+1**
   inside a composition, buying the same capability every time.
7. **A capability the split recovers survives a composition step.**
   `W896LSX13` and wave 19's `W896X13` are the same 187,214,592 parameters with
   byte-identical summarisation lanes, and the whole difference is a planning
   lane that names backup, TLS, verification and rollback in the parent's own
   order where the control names nothing from the task.
8. **The review and the distribution disagree for the third wave running.**
   `W896LS` beats the sealed 4/15 child on six of seven benchmark columns and
   holds one fewer review case.
9. **The max-min child's benchmark row agrees with its review, which is new.**
   Lambada accuracy **0.0000** and perplexity **23,120** — the worst
   distribution row in the campaign — where wave 20's negative was better than
   the living child on four accuracy tasks.
10. **The deployment's completeness was an assumption until this wave.** 3,568
    files verified against the commit's own blob digests, **0 content
    mismatches**, 714 absent by declaration and 7 excluded by name.
11. **The hardlink trick that made the deployment cheap broke the review, and
    the guard that caught it is right.** A prompt pack with `st_nlink > 1` is a
    pack a second name can rewrite, and the review's whole authority is that the
    pack it read is the pack the seal names.
12. **The campaign's frontier child had no dossier in the repository.** Wave 15
    left its package on the server; `ARCH.CL896`'s 74-file package is committed
    here, four waves after the child became the thing every note compares to.
