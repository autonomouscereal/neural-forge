# NOTES-MX-W — wave 19: the last standard axis, and the first variable that is not per-site quality

Wave 19, branch `codex/lrc-blank-compression`, source commits `a69cfcc0` (the
wave-19 builder: the `hidden_size` axis, the shared residual basis, the identity
gate, the per-case attribution artifact, and the ADDENDUM 12 sealer's
basis-projected terminal comparison), `6cb1266a` and `e7d068da` (two repairs to
the fold's own guard, both found by the identity gate refusing itself), plus the
sealing commit for this note. One deployment built and sealed every child:
`neural-forge-matrix-e7d068da`. Protocol
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b` and prompt
pack `0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b` verified
on every extraction and every review.

**Three calibrations passed before anything was graded.** The fresh conditioned
capture reproduces the sealed collector's rows to wave 15's numbers exactly (L00
relL2 3.367e-3 cos 0.9999944, L05 1.647e-2 / 0.9997207, L13 2.034e-2 /
0.9997842). The parent's sixteen frozen-review responses digest to
**`865f006e3813712ec202cb23`** on **all seven** wave-19 dossiers. And the parent's
benchmark row reproduces the sealed reference in all seven columns (0.5450 /
0.5050 / 0.6800 / 0.5500 / 0.4600, lambada ppl 24.39, wikitext 81.80).

*§0's sentences are at the bottom.*

---

## 1. The axis, and the config that makes it legal

`hidden_size` is the only standard config field that reaches the tied
embedding — 65,536 × 1024 = 67,108,864 parameters, **29.2% of the model** — and
it is the wall `NOTES-MX-V` §5 named: 0.5x needs the 162.58M body to fall 3.4x
while no other axis moves the embedding at all. Narrowing the residual moves
both at once. The child's parameter count is

    32*H^2 + 191541*H + 768

which returns **229,693,184 at H = 1024** — the parent, exactly — and the
exported children match it to the parameter at every rung.

**`head_dim` is PINNED at 64 and written explicitly.** `Lfm2Attention` reads
`getattr(config, "head_dim", hidden_size // num_attention_heads)`, so leaving it
out at `hidden_size` 768 builds every attention block at head_dim **48**: a
different RoPE frequency ladder and a different head width, which is wave 15's
`--head-dim` axis, sealed **dead** (`ARCH.HD16`, 0/15). Pinning it costs 1.5
points of net ratio at 768 (0.722610x against 0.707202x) and buys the parent's
own frequency ladder, the parent's own `q_layernorm`/`k_layernorm`, and an axis
that moves one thing. The legal config, stated in full:

    hidden_size = block_dim = conv_dim = H     head_dim = 64 (explicit)
    num_attention_heads = num_heads = 16       num_key_value_heads = 8
    intermediate_size = block_ff_dim = 2560    layer_types, rope_parameters unchanged

`block_dim` and `conv_dim` are this checkpoint's own restatements of the residual
width; `Lfm2Config` declares neither and the modeling code reads neither, but a
config that says 1024 in two fields and 768 in the one that is read is not
config-consistent under R3, so they move too.

## 2. The census, which was free, and which does not decide anything

One pass over the corpus, the residual's second moment at all **29**
residual-write endpoints (14 blocks × {`operator_norm`, `ffn_norm`} plus
`embedding_norm`), each endpoint equalised to trace one. Cumulative energy in
the top-k directions:

| k | reader.fit | reader.hld | raw.fit | raw.hld |
|---:|---:|---:|---:|---:|
| 960 | 0.9874 | 0.9895 | 0.9945 | 0.9954 |
| **896** | 0.9729 | 0.9769 | 0.9881 | 0.9899 |
| 832 | 0.9568 | 0.9627 | 0.9809 | 0.9836 |
| **768** | 0.9389 | 0.9466 | 0.9729 | 0.9763 |
| 704 | 0.9191 | 0.9284 | 0.9639 | 0.9681 |
| **640** | 0.8973 | 0.9081 | 0.9539 | 0.9587 |
| 576 | 0.8732 | 0.8853 | 0.9427 | 0.9480 |
| **512** | 0.8465 | 0.8596 | 0.9301 | 0.9357 |
| 256 | 0.7004 | 0.7142 | 0.8565 | 0.8614 |

`reader` is the residual as it is READ — RMS-normed and multiplied by that
endpoint's own gain, the only form any consumer in this architecture ever sees;
`raw` is the residual as it is carried. **The spectrum is flat.** There is no
knee: giving up a quarter of the directions costs 6% of the read energy and
half of them costs 15%. The basis was written as a sealed artifact per width
(`wave19/basis/E-<mode>-<H>.safetensors`, digest in every receipt built against
it) and the *measured* retained fraction on the held-out rows — the honest
number, since the basis is fit on the fit rows — is **0.9627** at 896, **0.9193**
at 768, 0.8697 at 640 and 0.8124 at 512.

Per the anti-proxy rule the ladder ran regardless, and §4 is the reason that rule
exists: the census would have predicted the wrong cliff.

## 3. The construction, and an identity gate that is a byte comparison

One shared orthonormal `E` (1024 × H). A *per-boundary* chart — what wave 2's
`hf_causal_cross_width_charts` compiles, excavated before any of this was
written — is **not expressible in a stock child**: the residual is a running sum
and there is nowhere to put a chart transition, so the shared basis is forced by
R3 rather than chosen for convenience.

With `r_c = E^T r`, and because **the lift preserves the norm** (`||E r_c|| =
||r_c||`), the parent's RMSNorm at the lifted row and the child's at the row
itself differ by exactly `sqrt(1024/H)`. Every piece is then closed form and
exact against the local target:

* **readers** (q/k/v, conv `in_proj`, w1/w3): `W_c = sqrt(1024/H) · W diag(g) E diag(1/g_c)`;
* **the norm gauge**: `g_c = diag(E^T diag(g) E)`, the transported gain's own
  diagonal — a convex combination of the parent's gains, so it cannot leave
  their range;
* **writers** (attention `out_proj`, conv `out_proj`, w2, the embedding rows):
  `E^T W`, the L2-optimal write into the retained subspace, then re-solved
  against the child's own installed activations under THE LAW;
* **the tied head**: the tied matrix is `W_emb E` in both roles, so the final
  gain has nowhere to live but the final norm's diagonal, and it is solved there
  in closed form from one H × H Hadamard system
  (`[(E^T C E) ∘ (Z^T Z)] g = Σ z_i ∘ (Q n_i)`, `C = W_emb^T W_emb`) with no
  65,536-column target ever materialised;
* **the conv** is the one genuinely lossy piece and says so: `Lfm2ShortConv`
  gates channelwise (`Bx = B * x`, `y = C * conv_out`), so no rotation of its
  internal channel space is representable and its H channels are a
  pivoted-Cholesky **coreset** of the parent's 1024, conv taps verbatim.

**The identity gate is a byte comparison and it passes.** At `--hidden-size 1024
--basis identity`, **132 of 132 tensors are bit-identical to the parent's**, the
loader counts 229,693,184 parameters = 1.000000x, and the only config difference
is `head_dim: null -> 64`, which is the value the fallback computes anyway. It is
provable rather than lucky: at `E = I` the reader fold is `W diag(g) diag(1/g)`,
both factors are bf16, their product is exact in fp64 (eight mantissa bits times
eight fit in fifty-three), and dividing that exact product by its own exact
factor returns the other factor exactly.

**The gate refused itself twice, and both refusals were the guard being wrong.**
The first was a floor on the divisor: this parent has RMSNorm channels it has
switched off — `layers.1.ffn_norm` carries **9.2e-7** and `layers.0.ffn_norm`
**1.3e-6** — and at the identity the transported gain inherits them, so a floor
fires on the one case where the division is provably exact. The second was the
replacement guard measuring `||W_c||` against `||W diag(g)||` instead of `||W||`,
which reads `1/mean|g| ≈ 17` at the identity. The guard that survives is
`||W_c|| / (sqrt(1024/H) ||W||)`, exactly 1.0 at the identity by construction,
measured at every site.

## 4. The ladder — and per-site quality is not the variable

| child | H | net | review | per-lane | conv write cos (med/min) | attn | mlp | max drift | min parent energy retained |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `ARCH.W896` | 896 | 0.859023x | **4/15** | impaired (9/16) | 0.9944 / 0.9820 | 0.9998 | 0.9999 | 0.267 | 0.9705 |
| `ARCH.W768` | 768 | 0.722610x | **3/15** | impaired (9/16) | 0.9904 / 0.9652 | 0.9998 | 0.9999 | 0.375 | 0.9346 |
| `ARCH.W640` | 640 | 0.590763x | 0/15 | dead (16/16) | 0.9902 / 0.9523 | 0.9999 | 0.9999 | 0.466 | 0.8883 |
| `ARCH.W512` | 512 | 0.463481x | 0/15 | dead (16/16) | 0.9916 / 0.9505 | 0.9999 | 0.9999 | 0.557 | 0.8258 |

**This is the wave's finding and it is not the one the deck expected.** Wave 14
located a cliff *in per-site write cosine*, between 0.93 and 0.86, and every wave
since has read per-site quality as the thing that predicts a child. On this axis
**per-site quality does not move**: at 512 the attention writes still land at
cos 0.9999, the MLP writes at 0.9999 and the conv writes at 0.9916 median
against their own local targets — every one of them *better* than the 896 child's
conv — and the model emits the empty string in eight of sixteen lanes and fills
the cap with quotation marks in the other eight. Two things do move, monotonically
and together: the fraction of the parent's residual the subspace can hold
(0.9705 → 0.8258) and the drift the child's stream accumulates against `E^T r`
(0.267 → 0.557).

The honest reading: **a fold can be locally excellent everywhere and globally
dead**, and on this axis the controlling variable is how much of the stream you
kept, not how well you wrote into what is left. The census's flat spectrum is
what that looks like from the other side — there was never a subspace to find,
only a price to pay per direction.

**The cliff is between 768 and 640**, located by the bisection rung, and it is
sharp: 3/15 to 0/15 across 128 dimensions and 13 points of net ratio, with the
retained energy moving only 0.9193 → 0.8697. 640 is *deader* than 512 in one
respect worth recording — fifteen of its sixteen lanes return the **empty
string**, where 512 at least fills eight lanes with punctuation.

**What survives is one lane, and it survives everywhere.** The summarisation
lane returns the decision, the risk and the mitigation correctly at 896, at 768
(with the supplier's name corrupted to "Supper B"), and in **both** compositions
— four different children, the same two sentences. It is the campaign's
most-held case (§7).

## 5. The compositions: the second axis is still fatal

| child | recipe | net | review |
|---|---|---:|---:|
| `ARCH.W896` | width alone | 0.859023x | **4/15** |
| `ARCH.W896KV6JS` | width + 16→12 heads, joint solve, λ on an internal split | 0.841047x | 2/15 |
| `ARCH.W896X13` | width + block 13 excised (the census's best single, 13/15 solo) | 0.815064x | 1/15 |

The merge composes on the narrowed block without any new machinery — its census,
its gauge and its solve read the child's own rows and are width-agnostic — and
its per-site numbers are healthy (relL2 0.26–0.29, cos 0.956–0.965 against
prior-only 0.92–0.94). It still costs two review cases. The excision of block 13
— the *best* single in the whole census — costs three. Wave 18's law survives its
first contact with a new axis: **the first composition step is survivable and the
second is fatal**, and the width axis is not an exception to it.

There is no mask defect to asterisk here. A width child is **instantiated from a
config** whose `layer_types` already omits the excised block, so wave 18's
defect — a live layer list shifting out from under a config one — cannot arise.

## 6. The quad, built because the rule said to build one

`ARCH.Q01050913`, blocks 1+5+9+13, the max-spread quad, **0.789916x, 0/15**, all
sixteen lanes charged. Seven fill the cap; the greeting lane returns 1,024
colons; the summarisation lane echoes the prompt and degenerates into "Supplier
B is significantly more than than than percent more than". The excision ladder
is now complete and monotone: **13/15 single → 7/15 pair → 3/15 triple → 0/15
quad**, which is the additive-interference rule the scaling document states, run
to its end.

## 7. The free probes: one built, one still correctly gated

**The per-case attribution artifact is built** (`mx19_percase_attribution.py`,
`wave19/percase-attribution.json`), from 63 sealed per-case readings across waves
15–19 and their build receipts. It needs no GPU and no evidence.

* **The review is 89.6% a ladder.** A single strict difficulty order predicts
  **0.8963** of the (child, case) cells. The campaign's 15-case score is mostly
  one number wide — which is why per-child scores compare at all — with about a
  tenth of the matrix carrying real per-axis structure.
* **The difficulty order**, campaign-wide hold rate:
  summarization-priorities **0.44**, uncertainty-insufficient-data 0.43,
  writing-concise-email 0.41, instruction-three-actions 0.35, context-retrieval
  0.32, planning-dependencies 0.27, conversation-natural-greeting 0.25,
  creative-continuity 0.21, comparison-ram-rom 0.14, debugging-off-by-one 0.13,
  multi-turn-correction 0.10, arithmetic-multi-step 0.08, coding-deduplicate
  0.05, logic-ordering 0.03, explanation-seasons **0.02**.
* **The tenth that is not the ladder is per-axis and it is legible.** The head
  merge holds debugging-off-by-one at **0.43** against a campaign 0.13 and
  creative-continuity at 0.43 against 0.21, and gives up planning-dependencies
  (0.14 against 0.27). Excision holds instruction-three-actions at **0.59**
  against 0.35 and planning-dependencies at 0.52 against 0.27. The width axis
  holds summarisation (0.50, and 1.00 in both compositions) and takes
  context-retrieval and planning-dependencies to **zero**.

**XGRAM retrospective calibration was not run, and the reason is a sealed
measurement rather than a budget.** Both it and the gap-4 attribution consume the
G artifact — the terminal-transfer surrogates `residual.post_layer[l] ->
residual.post_layer[13]` — and wave 14 published its held-out R²: `L00 0.046,
L01 0.136, L02 0.227, L03 0.268, L04 0.296, L05 0.311, L06 0.338, L07 0.368,
L08 0.396, L09 0.459, L10 0.615, L11 0.706, L12 0.874`. Against the
pre-registered `R² > 0.5` floor, **three layers of thirteen clear it**, and the
per-endpoint transfers for the 2,560-dim writers are mostly negative. XGRAM's
error signatures are therefore restricted to L10–L12 or they are, in wave 14's
own words, pricing noise — and a width fold touches all fourteen layers at once,
so there is no version of this wave's children XGRAM could have priced. The
probe is not blocked on time. It is blocked on a precondition that has already
been measured and failed, and the honest thing is to say so with the numbers
rather than run it and publish a correlation nobody can use.

## 8. Benchmarks

Every non-looping child, ADDENDUM 8 invocation unchanged (limit 200, seed 1234,
batch 4, max_length 2048, float32, sm86):

| model | net | review | arc_e | hella_n | piqa_n | wino | lambada | lam ppl | wiki wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000 | 12/15 | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `W896` | 0.859023 | **4/15** | 0.3550 | 0.4000 | 0.5650 | 0.4900 | 0.1450 | 638.72 | 500.08 |
| `W896KV6JS` | 0.841047 | 2/15 | 0.3150 | 0.3850 | 0.5200 | 0.5150 | 0.0650 | 2821.11 | 1161.46 |
| `W896X13` | 0.815064 | 1/15 | 0.3750 | 0.4000 | 0.5200 | 0.4950 | 0.1500 | 1671.03 | 923.07 |
| `W768` | 0.722610 | **3/15** | 0.3100 | 0.4300 | 0.5200 | 0.4850 | 0.0300 | 8176.90 | 1279.46 |

`W640`, `W512` and the quad were not benchmarked, under the standing rule that a
child which returns the empty string or loops a token to the cap in every lane
has no intelligence-per-compression to measure.

The review and the distribution disagree again, and in a new place: `W896X13`
has the best `arc_easy` of any child in the wave (0.3750, above the 896 child it
is built from) and the **worst** review score but one. A single held case is not
a distribution.

## 9. The frontier, and the distance to 0.5x

**Both readings, because they differ this wave.**

* **Strict (deepest child at ≥ 4/15).** Wave 16's `ARCH.W1536X13` **stands** at
  0.769439x, 4/15. `ARCH.W896` matches its score at a shallower 0.859023x.
* **Deepest alive (deepest child with a non-zero score).** It **moves** to
  `ARCH.W768`, **0.722610x at 3/15** — 12.0M parameters below wave 18's deepest
  non-zero child (`T010713`, 0.842437x, 3/15) and 10.8M below wave 16's frontier.

**Distance to 0.5x is now 51.1M parameters** (0.5x = 114,846,592; `W768` =
165,978,624), and for the first time the campaign has crossed the line at all:
`ARCH.W512` is **106,458,368 parameters, 0.463481x** — 8.4M *below* half the
parent — and it is 0/15. The wall `NOTES-MX-V` §5 described is gone as an
arithmetic obstacle and remains entirely as a behavioural one.

**The measured reason it stops** is one notch more specific than any wave before
it could state. It is not per-site install damage: every write at 512 is cos
0.99+ against its own local target. It is that below ~0.92 of the read energy —
between 768 and 640 — the residual stream stops carrying enough of itself for the
blocks downstream to find anything in it, and the model's first token becomes the
end of the sequence.

## 10. Estate

* **Commits.** `a69cfcc0` (the builder, the basis, the identity gate, the
  attribution artifact, the sealer's basis-projected terminal comparison),
  `6cb1266a` and `e7d068da` (the two guard repairs), plus the sealing commit for
  this note. Nothing pushed, nothing rebased, staged by explicit path only; the
  Codex lane's 324 uncommitted paths, including the frozen
  `src/neural_forge/hf_lfm2_circuit_trace_cache.py`, were not touched.
* **Self-test.** `mx19_selftest.py` **10/10**, no model loaded: the transported
  gain is the parent's at `E = I` to zero absolute delta, the reader fold at
  `E = I` returns the parent's weight with equal bf16 bits, the reader fold is
  exact against the local target to **8.1e-7** (the residual is RMSNorm's own
  eps, which the two normalisations divide by different means), the lift
  preserves the row norm to 4.0e-16, `E^T W` beats no least-squares alternative
  by more than 2.6e-26, the gram-side coreset is the row-side coreset index for
  index, the Hadamard normal equations reproduce a brute-force least squares to
  1.3e-15, and the parameter arithmetic returns 229,693,184 at H = 1024.
* **Ledger.** **606 → 620**, sha
  `20784fbf6b8c268667ab090e203f1d24f6cb92b9f4e9d452fbb0f81a5c0c4c2d`. Backed up
  first (`backup/ledger.jsonl.pre-wave19-20260813T074310`, verified by sha **and**
  `cmp`), merged exactly once behind `wave19/.MERGED_ONCE`, byte-prefix verified
  against the backup afterwards and again in the repository copy. Fourteen rows:
  7 ARCH run rows and 7 ADDENDUM 6 verdict rows.
* **One superseded row, dropped by the guard.** The quad's review was queued on
  both GPUs by mistake and sealed twice, leaving two ARCH rows for one child.
  The merge resolved it exactly as wave 15 designed: the row the sealed verdict
  names in `addendum6.reviewed_from_receipt_sha` (`2bd6c2d4aa05`) was kept and
  `c5d4822213af` was dropped and named in the merge report.
* **Seals.** `mx19_verify_seals.py --deployment <deployment>`: **15 roots
  present, 0 undeclared; 620 rows, 616 resolve, 616 verify, 0 fail, 4 do not
  resolve** — the same four pre-existing non-resolvers waves 14–18 named. The
  verifier was extended by **one added line**, not by rewriting `wave18` to
  `wave19`, which is the mistake wave 18's audit caught.
* **Disk.** Server `/` began and ended at 13 GB free. Deployments end at
  current+1 — `e7d068da` (wave 19) and `b372d47c` (wave 18) — with `a69cfcc0`
  and `6cb1266a` pruned. Eight children were built; **five are kept** (`W896`,
  `W768`, `W896KV6JS`, `W896X13`, `W512`) and three are deleted and named with
  their sealed tensor digests in
  `wave19/children-retention-manifest-wave19.json`, freeing 1.03 GiB. `W512` is
  kept at 0/15 deliberately: it is the campaign's only child under half the
  parent and a landmark you have deleted is a number in a note. `MX19-IDENT` is
  deleted *because* its gate passed — 132 of 132 tensors bit-identical means
  keeping it keeps the parent twice.
* **Human-review package, in the repository.**
  `reports/matrix-campaign-20260810/wave19/human-review/` — 60 files: every
  dossier, every child and parent response rendered verbatim, every run receipt,
  every verdict receipt, every per-case reading, every build receipt, the six
  basis censuses, the benchmark table, the parent-lane identity check and the
  ledger merge report.

## 11. What the wave did not spend, and one thing it left standing

* **The writer re-solve is mildly harmful at some sites and was not fixed.** At
  every MLP site the prior-only held-out cosine is **1.0000** and the solved one
  is 0.9999; at `L01`'s conv the prior is 0.9867 and the solve returns 0.9820.
  `law_abiding_solve` takes the first λ the trust region admits, which is the
  smallest, and wave 18 paid five review cases to learn that this is exactly the
  overfit an internal fit-side split fixes (`law_abiding_solve_split`, already
  in this file, wired only into the value path). Wiring it into the width
  writers is a one-line change and a rebuilt ladder; it was not spent this wave
  because the effect is 1e-4 in cosine against a conv coreset that costs 5e-3
  and a drift that costs 0.3.
* **The `evidence-pca` basis was written and never built against.** The
  alternative basis at 768 is sealed (`E-evidence-pca-768.safetensors`,
  retaining 0.9636 of the raw energy against reader-PCA's 0.9522, and 0.9122 of
  the read energy against 0.9193). The rule was to build it only if a width
  landed marginal; 768 landed at 3/15 and 640 at 0/15, which is a cliff and not
  a margin, so the basis-sensitivity question is unanswered and cheap.
* **`--arm stale` was never graded.** The closed-form fold with no conditioned
  re-solve is a real control on this axis — the prior *is* the L2-optimal
  projection — and it would price the conditioned solve exactly. Not built.
* **No rung between 768 and 640 was built.** The cliff is located to a 128-wide
  interval and could be halved again at 704 (0.656116x) for one build and one
  review.

## 0. The sentences

1. **The last standard axis is built, and the identity of its fold is provable
   in bytes.** At width 1024 with the identity basis, **132 of 132 exported
   tensors are bit-identical to the parent's**, because the norm gauge
   `g_c = diag(E^T diag(g) E)` makes the reader fold `W diag(g) diag(1/g)` and a
   bf16 product divided by its own bf16 factor is the other factor exactly.
2. **The residual stream has no low-rank structure to exploit.** Pooled over all
   29 endpoints, the top 768 of 1024 directions hold **0.9389** of the energy a
   reader sees and the top 512 hold 0.8465. There is no knee; there is a price
   per direction.
3. **`hidden_size` 896 and 768 both live: 4/15 at 0.859023x and 3/15 at
   0.722610x.** Every other fold class in this campaign that reached 0.72x was
   zero.
4. **Per-site install quality is not the variable.** At 512 the attention and
   MLP writes land at cos **0.9999** against their own local targets and the conv
   writes at 0.9916 median — better than the living 896 child's conv — and the
   model returns the empty string. What moves with the score is the retained
   residual energy (0.9705 → 0.8258) and the accumulated drift (0.267 → 0.557).
5. **The cliff is between 768 and 640, and 640 is deader than 512.** Fifteen of
   sixteen lanes at 640 return the empty string; 512 at least fills eight of them
   with quotation marks.
6. **The campaign is under 0.5x for the first time, and it is 0/15.**
   `ARCH.W512` is 106,458,368 parameters, **0.463481x**, 8.4M below half the
   parent. The wall is gone as arithmetic and is entirely behavioural.
7. **The second axis is still fatal, on a new axis.** Width plus the head merge
   is 2/15 and width plus the census's best single excision is 1/15, from a 4/15
   width child and a 13/15 excision.
8. **The excision ladder is complete and monotone: 13 → 7 → 3 → 0.** The
   max-spread quad at 0.789916x is 0/15 with all sixteen lanes charged.
9. **The review is 89.6% one number wide.** A single difficulty order predicts
   0.8963 of 63 children × 15 cases; the tenth that it misses is per-axis and
   legible — the merge keeps debugging, excision keeps instruction-following,
   width keeps summarisation and takes context-retrieval to zero.
10. **The frontier moves on one reading and not the other, and both are stated.**
    Deepest child with a non-zero score: `ARCH.W768`, **0.722610x at 3/15**.
    Deepest child at ≥4/15: still wave 16's `ARCH.W1536X13`, 0.769439x.
