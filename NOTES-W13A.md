# NOTES-W13A — r12 exact-4x first-divergence diagnosis (cache only)

Wave 13A, branch `codex/lrc-blank-compression`. R13 plan item 5, cache-only
half: the first-divergence diagnosis of the sealed, certified, 0/15 r12 child
`a0023-exact4x-h816-mgv3-r12` (57,420,144 counted parameters, exact 4.0002x of
LFM2.5-230M's 229,693,184).

**Zero refolding, zero training, zero parent forward calls.** Every sealed
ledger in this wave carries `parent_forward_calls: 0`, `training_steps: 0`,
`backward_calls: 0`, `optimizer_steps: 0`, `model_mutations: 0`, and
`numeric_acceptance_decision: null`. Nothing under the sealed r12 run directory
was written, moved, or deleted (verified: `find $RUN -newermt "2026-08-04
21:00"` is empty after the whole wave).

**Read §6 first if you only want the verdict.** The short version: the R13
causal story's headline mechanism — moderate per-boundary error compounding
multiplicatively through 14 pretrained boundaries — is **contradicted by the
measurement**. The child leaves the parent's decision region at **layer 0**,
and layers 1–9 add essentially nothing because there is nothing left to lose.

---

## 1. What was run, and exactly how

Tools: Codex's `60eb928b` ("Authenticate padded LM-head audit domain") and
`9bc0d735` ("Add authenticated exact4x causal folding diagnostic").

**The stated deployment was wrong and cost a detour.** The task pointed at
`/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a`
(HEAD `8369f42e`). `8369f42e` is an *ancestor* of both diagnostics commits —
`git merge-base --is-ancestor 9bc0d735 8369f42e` is false — so that deployment
does not contain `audit_lfm2_exact4x_causal_diagnostic.py` at all and carries
the pre-`60eb928b` first-divergence runner (50,509 bytes vs 62,072). The
runtime that does have both is
`/home/cereal/model-surgery-lab/deployments/neural-forge-exact4x-9bc0d735`
(its git HEAD reads `7b48e20` with Codex's diagnostics as working-tree
overlay). All four files there are byte-identical to local `9bc0d735` and to
local `HEAD b9e37374`:

```
2ee18fb3dbe540e2860b8d0cc2aca707048baa3e37eee3ade0a9913776ce0108  experiments/audit_lfm2_exact4x_first_divergence.py
1c5a505ab72aced405b8080b844b10b7954456ef7e4e4ba4189e75a5d334bb2f  experiments/audit_lfm2_exact4x_causal_diagnostic.py
33d0fdd5fcb3b54505cc29a14649cc3377ff058339399667987bf9efadd3c963  src/neural_forge/hf_lfm2_exact4x_first_divergence.py
c7fccb11a961712789493caf47ae567dc7ada6bd260f83b3ae7d329984dc7789  src/neural_forge/hf_lfm2_exact4x_causal_diagnostic.py
```

Nothing in that deployment was edited. The only mtime change under it is
`.git/index`, touched by a read-only `git -C … status --short`.

### Sealed inputs (all discovered from the run itself, none guessed)

| input | value |
|---|---|
| child package dir | `…/a0023-exact4x-h816-mgv3-r12/published-child/child` (`certificate.standalone_child_directory: "child"`) |
| `--child-package-sha256` | `19f5e7d26619d129520154c78fbec4d1f32b7c5a3a08254ea2a64d5788aa71db` |
| `--trace-cache` | `…/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95` |
| `--basis-phase-cache-root` | `…/neural_forge_phase_cache_v1/lfm2-circuit-balanced-residual-basis-v1` |
| `--basis-phase-key-sha256` | `ce0593c4762a8ed586d2c16741f1ed02df694c432a8233416585f4d16ebb49f4` |
| `--basis-phase-hmac-key-file` | `/home/cereal/model-surgery-lab/keys/neural-forge-balanced-basis-phase-v1.key` |
| `--basis-phase-hmac-key-id` | `lfm2-circuit-balanced-residual-basis-v1` |

The three cache paths and both HMAC arguments come verbatim from
`deployments/nf-launch-integrated-carrier-exact4x-r12.sh` (read only). The
phase-key digest comes from Codex's own sealed
`neural_forge_exact4x_first_divergence_20260804/…-layer0-v3/request.json`.

**Source/target widths are discovered, not asserted.** The runner reads the
residual source width from the trace `layer_topology` (all 14 layers agree:
`hidden_size: 1024`) and the target width from the child receipt (`816`); the
`--source-width` / `--target-hidden-size` flags only *assert* those. So the
carrier geometry is 1024 → 816 residual with the 4x taken globally on parameter
count (FFN intermediate 2560 → 321, the 8x contraction the R13 plan flags), not
4x on residual width. Layer kinds alternate
`short_conv, short_conv, full_attention, short_conv, full_attention, …`; layers
2/4/6/8/10/12 are `full_attention`.

### GPU

`nvidia-smi` at launch: cuda:0 and cuda:4 are RTX 3090s (r12's pair), 1/2/3/5
are Tesla P40s (sm_61 — this torch build refuses them; Codex's first causal
smoke died exactly there with `cuDNN version 91900 is not compatible with
devices with SM < 7.5`), and index 6 is a Quadro RTX 8000 (sm_75, 46 GB, only a
260 MiB ComfyUI resident). Codex's sealed v3 probe used the RTX 8000, so both
lanes pin it by UUID and expose nothing else:

```
export CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd   # cuda:0 = Quadro RTX 8000
```

Both launchers preflight the device class before `nohup`. The causal runner
fails closed if its runtime device string differs from the sealed v3 request,
so the two lanes had to agree.

### Commands

Launchers are committed with the outputs at
`/home/cereal/model-surgery-lab/artifacts/neural_forge_r12_diagnosis_v1/` and
copied into `reports/r12-diagnosis-20260804/`.

```bash
# lane 1 — first divergence, all 14 focus layers
python "C:/Users/cereal/.claude/skills/server-manager/ssh_client.py" --server ai --script '
bash /home/cereal/model-surgery-lab/artifacts/neural_forge_r12_diagnosis_v1/nf-r12-diagnosis-first-divergence.sh'

#   which runs, under PYTHONPATH=$RUNTIME/src:$RUNTIME and the venv at
#   /home/cereal/model-surgery-lab/venv/bin/python:
#     experiments/audit_lfm2_exact4x_first_divergence.py \
#       --child-directory  …/published-child/child \
#       --child-package-sha256 19f5e7d2… --trace-cache …/76ff3cbc… \
#       --basis-phase-cache-root …/lfm2-circuit-balanced-residual-basis-v1 \
#       --basis-phase-key-sha256 ce0593c4… \
#       --basis-phase-hmac-key-file …/neural-forge-balanced-basis-phase-v1.key \
#       --basis-phase-hmac-key-id lfm2-circuit-balanced-residual-basis-v1 \
#       --output-directory …/neural_forge_r12_diagnosis_v1/first-divergence \
#       --runtime-device cuda:0 --runtime-dtype float32 \
#       --source-width 1024 --target-hidden-size 816 \
#       --focus-layer-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13 \
#       --maximum-selected-parent-bytes 536870912 \
#       --maximum-donor-rows-per-score-chunk 4096

# lane 2 — causal diagnostic, once per operator layer
bash …/nf-r12-diagnosis-causal.sh 0
bash …/nf-r12-diagnosis-causal.sh 12
#   → audit_lfm2_exact4x_causal_diagnostic.py … \
#       --first-divergence-directory …/first-divergence \
#       --first-divergence-ledger-sha256 a4fa6e6886a4aaf95d34b47c272597a1eab72df5da964cb0e7cbf3efe0d51bd8 \
#       --operator-layer-index {0,12} --operator-kind auto \
#       --bootstrap-resample-count 4096 --bootstrap-confidence-level 0.95
```

Codex's v3 probe ran `--focus-layer-indices 0`. This one runs all 14. That
changes nothing about the state-path profile — `focus_layers_limit_branch_
diagnostics_only: true`, every layer's `residual.post_operator` /
`residual.post_layer` is observed either way — but it is what produces the
per-layer **branch** (`operator.output`, `mlp.down`) rows in §3, which turn out
to carry the decisive evidence. Selected parent payload grows 51,368,000 →
65,425,472 bytes per record, still inside the 512 MiB read ceiling, so the
ceiling did not have to move.

Wall clock, measured from the hash-chained progress journals (which start
*after* preflight, so model load, route relocation and — for the causal lanes —
the re-authentication of all 256 v3 batch results are excluded; that preflight
is roughly 4–5 min per lane):

| lane | journal events | span (`preflight/request-authenticated` → `run-complete`) |
|---|---|---|
| first divergence | 770 | 0:07:12 (256 records, 1 child forward each) |
| causal layer-00 | 777 | 0:04:55 (2,048 child forwards = 8 coalitions x 256) |
| causal layer-12 | 777 | 0:03:58 (2,048 child forwards) |

All three lanes together are well under an hour of GPU time.

### Sealed identities produced

| artifact | digest |
|---|---|
| first-divergence request | `e04d43fda67d7158fdb7857d8e0ab5e2a1c2c944fb91b902227a7688efa486f0` |
| first-divergence ledger | `a4fa6e6886a4aaf95d34b47c272597a1eab72df5da964cb0e7cbf3efe0d51bd8` |
| causal layer-00 ledger | `b08e6b799e17c0cc73f2c2fd9ec1a292968a53d92ff10ef95e51b606cf045872` |
| causal layer-12 ledger | `a9620832ac30998ae9c0a24d0c8cdff60509315f2d3d1a51c9aa68d6b5bec4eb` |

Re-derived independently after fetching, from the canonical-JSON bodies, by
`reports/r12-diagnosis-20260804/verify_seals.py` — all 15 checks `OK`,
including that both causal `v3_binding`s pin the fetched first-divergence
ledger/request/child/trace, and that every operation audit is cache-only.

---

## 2. Per-layer divergence profile

256 records (192 fit / 64 heldout), 9 phases each (prefill + 8 forced decode
steps) → 1,728 fit and 576 heldout observed states per boundary. Metric is
`relative_l2 = ||child − encode(parent)|| / ||encode(parent)||` in float64,
against the sealed balanced-basis chart `df7f6e88…`. **`relative_l2 = 1.0` is
the "emit zero" baseline** — per R13 item 2, the same bound the kernel lane
already enforces (`hf_causal_kind_translation.py:1437`, not re-verified here)
and the LFM2 lane lacks.

Heldout means (fit agrees to ≲1%). `post_operator` rows for L02–L09 are elided
for width — they are in `first-divergence-profile.txt`, and each elided value
is implied by the following row's incremental delta:

| boundary | mean relL2 | incr. delta | mean cos | norm ratio | irreducible* |
|---|---|---|---|---|---|
| `c_in` | **0.000000** | — | 1.0000 | 1.000 | 0.0000 |
| L00 post_operator | 0.8358 | +0.8358 | 0.6151 | 0.920 | 0.7739 |
| L00 post_layer | **1.1594** | +0.3237 | 0.3512 | 1.036 | 0.9313 |
| L01 post_operator | 1.2917 | +0.1322 | 0.4330 | 1.353 | 0.8940 |
| L01 post_layer | 1.0912 | −0.2004 | 0.5079 | 1.175 | 0.8490 |
| L02 post_layer | 1.3334 | −0.1967 | 0.3245 | 1.257 | 0.9417 |
| L03 post_layer | 1.3251 | −0.0668 | 0.3342 | 1.259 | 0.9365 |
| L04 post_layer | 1.4266 | −0.1784 | 0.3006 | 1.356 | 0.9496 |
| L05 post_layer | 1.4485 | −0.0859 | 0.3007 | 1.387 | 0.9483 |
| L06 post_layer | 1.7380 | −0.1079 | 0.2562 | 1.698 | 0.9626 |
| L07 post_layer | 1.4984 | −0.1621 | 0.2968 | 1.448 | 0.9499 |
| L08 post_layer | 1.6085 | −0.1051 | 0.2838 | 1.574 | 0.9544 |
| L09 post_layer | 1.4423 | −0.0606 | 0.3036 | 1.384 | 0.9491 |
| L10 post_operator | **3.0529** | **+1.6105** | 0.1954 | 3.081 | 0.9761 |
| L10 post_layer | 2.2822 | −0.7707 | 0.1982 | 2.252 | 0.9739 |
| L11 post_layer | 1.8738 | −0.1564 | 0.2265 | 1.819 | 0.9670 |
| L12 post_operator | **8.6308** | **+6.7570** | 0.0273 | 8.593 | 0.9830 |
| L12 post_layer | 6.4902 | −2.1406 | 0.0576 | 6.461 | 0.9776 |
| L13 post_operator | 5.8285 | −0.6617 | 0.0577 | 5.790 | 0.9737 |
| L13 post_layer (terminal) | **4.4728** | −1.3557 | 0.1768 | 4.527 | 0.9513 |

\* `irreducible = mean sqrt(1 − cos²)`: the relative-L2 error that would remain
after an **oracle per-vector rescale**. It is the part of the divergence no
gauge, norm, or scale correction can remove. Derived in
`first-divergence-scale-direction-split.txt`.

Read that table in three pieces.

1. **`c_in` is exact.** `relative_l2 = 0.000000`, cosine `1.000000`, over all
   2,304 states. The C_in carrier route reproduces the chart-projected parent
   embedding bit-for-bit. The input interface is not a contributor.
2. **Layer 0 alone crosses the baseline.** One pretrained boundary takes the
   residual from 0.0 to 1.159 — already worse than emitting zeros — and
   knocks cosine to 0.351. Its irreducible component is 0.931: even with a
   perfect per-vector rescale, layer 0's output is 93% wrong in direction.
   The R13 story quotes "0.13–0.55 at layer 0" for real pretrained parents.
   r12's is 1.16.
3. **Depth 1–9 is a plateau, not a ramp.** 17 of the 27 incremental deltas
   after layer 0's operator boundary are *negative*, and the nine layers L1–L9
   add a net **+0.283** relL2 in total (heldout; +0.287 fit) — an average of
   +0.031 per layer. Mean relL2 wanders between 1.09 and 1.85 across those nine
   layers. Irreducible sits at 0.94–0.96 the whole way:
   direction is already gone, so there is nothing further to lose. The two real
   jumps — L10 (+1.61) and L12 (+6.76) — are both `full_attention` layers, and
   both are almost pure **norm blow-up**: norm ratio 3.08 and 8.59 while
   irreducible only creeps 0.949 → 0.983.

### Branch writes: the fold fails at the first solve

The all-14 focus set gives the per-layer branch rows. Heldout:

| branch | mean relL2 | mean cos | norm ratio | irreducible |
|---|---|---|---|---|
| L00 `operator.output` | 1.0987 | 0.1711 | 0.651 | 0.9757 |
| L00 `mlp.down` | 0.9445 | 0.4014 | 0.573 | 0.9055 |
| L01 `operator.output` | 4.7466 | 0.0791 | 4.689 | 0.9892 |
| L02–L09 (all) | 1.29–3.38 | −0.034…0.074 | 0.77–3.26 | 0.993–0.999 |
| L10 `operator.output` | 7.3355 | 0.0589 | 7.319 | 0.9948 |
| L12 `operator.output` | **18.3425** | −0.0246 | **18.286** | 0.9814 |
| L13 `mlp.down` | 2.6699 | 0.3629 | 2.827 | 0.9167 |

Layer 0's own short-conv write already has cosine 0.171 against the projected
parent write — 97.6% irreducible — and its MLP write 0.401. From layer 1 on,
every branch write has |cos| ≤ 0.12. **The closed-form solves did not produce
approximations of the parent's branch functions at any depth, including the
first one.** They also systematically under-write at layer 0 (norm ratio 0.57,
0.65) and over-write everywhere after (up to 18.3x).

---

## 3. Where the child crosses vocab decision boundaries

From `first-divergence-decision-boundary.txt` (derived over all 256 sealed
batch checkpoints; source `analyze_first_divergence_batches.py`).

- **Parent-winner top-1 agreement: 15/1,728 fit (0.868%), 3/576 heldout
  (0.521%).** The child's C_out top-1 matches the parent's on well under 1% of
  states.
- **First discrete mismatch: batch 0, `prefill`, `phase_index 0`** — the
  earliest observable state of the earliest record. 249 of 256 records first
  disagree at their prefill state. **0 of 256 records agree at every phase.**
- **Rank of the parent's winner inside the child's 64,402-row donor score
  vector**: median 6,032.5, mean 18,143.7, p05 12.2, p95 62,769.8, max 64,397.
  Rank 1 in 18 of 2,304 states; rank ≤ 100 in 312.
- **Margin**: the cached parent's own winner margin averages +1.414 (median
  +1.031). The folded child's margin *for that same token* averages **−1.088**
  and is **negative in 2,286 of 2,304 states**.
- **The padded LM head is exonerated.** 65,536 physical rows vs 64,402 defined
  semantic rows; `undefined_physical_top1_count: 0` in both splits and maximum
  undefined softmax mass 3.9e-05 (fit) / 2.3e-05 (heldout). The 1,134 padding
  rows are not where the probability is going. This is precisely what
  `60eb928b` was written to authenticate, and it comes back clean.
- **No time trend.** Terminal residual relL2 by decode phase index (heldout):
  5.04, 4.43, 4.64, 4.40, 4.56, 4.27, 4.51, 4.21, 4.20 — flat-to-decreasing,
  and prefill is the *worst* phase. Under teacher forcing there is no
  within-trajectory drift term to find; the divergence is entirely depth-shaped.

---

## 4. Causal diagnostic, layer 0 (`short_conv`)

Groups: **C** = restore `residual.pre` into `model.layers.0`, **S** = restore
the short-conv write at `model.layers.0.conv`, **M** = restore the MLP write at
`model.layers.0.feed_forward.w2`. "Restore" installs the chart-projected parent
activation at that site and lets the rest of the child run. Full trajectory
interventions, 8 coalitions x 256 records = 2,048 child forwards, 0 parent
forwards.

Heldout coalition means:

| coalition | terminal relL2 | defined-vocab KL | parent-winner margin |
|---|---|---|---|
| `{}` unmodified child | 4.4728 | 7.6950 | −1.1090 |
| `{C}` | **4.4728** | **7.6950** | **−1.1090** |
| `{S}` | 34.1981 | 7.3246 | −0.9468 |
| `{M}` | 38.1189 | 7.8475 | −0.9412 |
| `{S,M}` | 41.2957 | 7.5889 | −0.8018 |
| `{C,S,M}` | 41.2957 | 7.5889 | −0.8018 |

Three things fall out.

1. **C is an exact null player.** Every C-containing coalition equals its
   C-free counterpart to the last printed digit, and every C-containing
   Harsanyi dividend is exactly `0.000000` in both splits and all three
   metrics. The bootstrap Shapley for C is `0.000000 [0.000000, 0.000000]`.
   That is the causal counterpart of `c_in` measuring 0.0.
2. **Restoring the parent's layer-0 writes makes the terminal state 7.6x–9.2x
   WORSE.** If the child's layers 1–13 were a lossy approximation of the
   parent's, feeding them the parent's true layer-0 output would move the
   terminal state *toward* the parent. It moves it 30 relative units away.
   Heldout bootstrap Shapley on terminal-error *reduction* (4,096 resamples,
   record-clustered, 95%): S = **−16.451 [−17.466, −15.502]**, M = **−20.372
   [−21.195, −19.498]**. Tight, signed, and unambiguous.
3. **The two branch errors are strongly sub-additive, not multiplicative.**
   Harsanyi order-2 `{S,M}` dividend on terminal error: **+26.549** heldout
   (+25.238 fit). Individually the harms are −29.73 and −33.65; jointly only
   −36.82 instead of −63.37. Order-3 `{C,S,M}` is exactly 0. On the
   parent-winner margin the `{S,M}` interaction is −0.023 heldout / +0.018 fit
   — i.e. additive to within noise.

On the *output* metrics layer 0 does behave like the place where signal is
still recoverable, but barely: restoring both writes moves defined-vocab KL
7.695 → 7.589 (−1.4%) and the parent-winner margin −1.109 → −0.802. Still
deeply negative. Layer 0 is necessary and nowhere near sufficient.

### Measured end-to-end amplification

This is the one number that *supports* the "downstream amplification factor
nothing measures" half of the R13 story, and it is worth carrying into r13.
Restoring S zeroes the layer-0 error and the terminal error rises by at least
`34.198 − 4.473 = 29.73` relative units (triangle-inequality lower bound). In
absolute chart units, using the heldout mean projected-parent norms
(0.8448 at L0 post_layer, 5.7773 at the terminal):

```
layer-0 absolute residual error   = 1.159 x 0.845 =   0.98
terminal absolute deviation (>=)  = 29.73 x 5.777 = 171.7
end-to-end gain (>=)              = 175x over 13 layers  =  1.49x per layer
```

The child's stack multiplies a unit layer-0 perturbation by ~175x by the
terminal boundary. That gain applies to **corrections as much as to errors** —
which is exactly why item 6's back-to-front reclosure has to be judged on the
composed metric, not on writer-local fits.

---

## 5. Causal diagnostic, layer 12 (`full_attention`, the largest amplifier)

Groups: **C** = restore `residual.pre` into `model.layers.12`, **A** = restore
the attention write at `model.layers.12.self_attn` (consumer `out_proj`),
**M** = restore the MLP write at `model.layers.12.feed_forward.w2`.

Heldout coalition means:

| coalition | terminal relL2 | defined-vocab KL | parent-winner margin |
|---|---|---|---|
| `{}` unmodified child | 4.4728 | 7.6950 | −1.1090 |
| `{C}` | **141.4586** | 8.2759 | −1.5344 |
| `{A}` | 10.5278 | 7.9284 | −1.2695 |
| `{C,A}` | 29.2926 | 8.0295 | −1.3951 |
| `{M}` | 5.0111 | 7.6804 | −1.0960 |
| `{C,M}` | 141.6773 | 8.2734 | −1.5321 |
| `{A,M}` | 12.7281 | 7.8120 | −1.1757 |
| `{C,A,M}` | 19.7824 | 7.7236 | −1.1491 |

- **Handing layer 12 the parent's actual residual state multiplies terminal
  error by 31.6x** (4.47 → 141.46). By depth 12 the child's residual stream is
  so far out of gauge with the parent's — norm ratio 1.82 entering the layer,
  cosine 0.227 — that parent-consistent input is *actively catastrophic*.
  Bootstrap Shapley for C on terminal-error reduction: **−73.919 [−76.292,
  −71.577]**.
- **A largely exists to compensate C.** Order-2 `{C,A}` Harsanyi dividend
  **+118.221** heldout (+119.071 fit), against order-1 `{C}` −136.986 and
  `{A}` −6.055. Order-3 `{C,A,M}` +11.391. The layer-12 attention write is
  fitted to the child's own broken residual, not to a parent-like one.
- **Nothing at layer 12 recovers output quality.** Every intervention leaves
  defined-vocab KL and the parent-winner margin flat or worse, including the
  grand coalition (KL 7.695 → 7.724, margin −1.109 → −1.149). Contrast layer 0,
  where the grand coalition at least improved both. That is a clean depth
  signature: what recoverable signal exists is at the *front* of the stack.

---

## 6. Verdict — does the R13 causal story hold?

**At which depth does r12's child leave the parent's decision region?**

**Layer 0.** In plain language: the child gets the embedding exactly right,
and then the very first pretrained boundary destroys it. After one layer the
residual is already further from the parent's than a vector of zeros would be
(relative L2 1.159 vs the 1.0 zero baseline), 93% of that error survives any
possible rescaling, and the layer's own short-conv write has cosine 0.171
against what it was supposed to reproduce. The output side agrees: the child's
top-1 token disagrees with the parent's on the very first prefill state of the
very first record, in 249 of 256 records, and in no record does it agree at
every phase. There is no "wall at layer N" to find, because the wall is the
first layer. A depth-ladder probe at 3/6/10 would have measured a corpse three
times.

**Does the profile match the R13 causal story (multiplicative depth
compounding), or contradict it?**

**It contradicts the headline mechanism and rescues one sub-claim.**

Contradicted:

1. *"Moderate per-boundary error."* r12's layer-0 boundary error is 1.16
   relative L2, not the 0.13–0.55 the story cites. There is no small-error
   regime to compound. The premise fails at the first data point.
2. *"Depth compounds it."* Layers 1–9 are a plateau (1.09–1.85) that adds a
   net +0.283 relL2 across all nine, with 17 of 27 post-layer-0 incremental
   deltas negative. Irreducible directional error is pinned at 0.94–0.96 across
   that whole span. Compounding needs headroom; after layer 0 there is none.
3. *Most decisively:* compounding presumes the child is trying to track the
   parent and drifting. It is not. Restoring the parent's true activations
   makes the terminal state 7.6x worse at layer 0 and **31.6x worse** at layer
   12. The child is an internally self-consistent function that has been solved
   against its own wrong intermediate states; the parent's trajectory is not
   even a low-error operating point for it. "Error propagating through the
   parent's pretrained dynamics" is the wrong frame — those dynamics are not in
   the child.
4. *The 16x approximate vocab bridge is not where it surfaced.* Zero states
   put top-1 in the 1,134 undefined padding rows; undefined softmax mass
   ≤ 3.9e-05. Whatever the bridge costs elsewhere, the collapse is upstream
   of it and visible in the residual stream at layer 0.
5. *Exposure bias is not measurable here.* Terminal error is flat across the 8
   forced decode steps and prefill is the worst phase. Under teacher forcing
   there is no drift term. That does not refute the on-policy argument — this
   evidence simply cannot speak to it, because the child never ran free here.

Supported, and worth keeping:

6. *"Downstream amplification factors nothing measures"* is real and now
   measured: **≥175x end-to-end, ~1.49x per layer**, concentrated in the
   `full_attention` layers (L10 +1.61, L12 +6.76 incremental relL2; L12's
   attention write alone has norm ratio 18.3). But it amplifies magnitude, not
   information — irreducible directional error barely moves across those same
   spikes (0.949 → 0.983). Removing the amplification would leave a child that
   is uniformly ~95% wrong in direction instead of ~95% wrong and 8x too big.

Also confirmed exactly as the story says: the acceptance machinery could not
have caught this, and **item 2's absolute terminal floor would have.** Terminal
relative L2 hits the `1.0` no-branch bound at layer 0's own install. A
fail-closed terminal floor evaluated per layer would have aborted r12 after one
boundary instead of after 95.2 hours and a publication.

### What this reprioritises for r13

- **Item 2 (absolute composition-aware terminal floor) is now the highest-value
  change, not the second.** It is the only thing here that would have stopped
  the run, and it would have stopped it at layer 0.
- **Item 4 (cap contraction by measured functional width) and item 7 (EIV ridge
  + margin row weights) are attacking the right term** — per-boundary solve
  quality — and should be judged at **depth 1**, on layer 0 alone, before any
  multi-layer arm. If layer 0 cannot be solved below relL2 ~0.3 with an
  irreducible component well under 0.5, no downstream change matters.
- **Items 1 and 6 (on-policy evidence, reverse reclosure) are second-order for
  this failure**, and item 6 in particular has to fight a measured ~175x
  forward gain. Neither is refuted; both are simply unjudgeable until a
  boundary solve exists worth composing.
- **Item 3 (exact vocab interface) is not the fix for r12's collapse.** The
  padded-domain audit is clean. Keep the arm to isolate bridge cost, but do not
  expect it to move the intelligence verdict.
- **Item 5's own success criterion — "a measured wall depth for LFM2" — is
  answered: depth 1.** The go/no-go it was meant to gate should therefore be
  read as: do not commit a 14-layer run until a *single-layer* fold of this
  parent at this compaction produces a boundary with cosine well above 0.17.

---

## 7. Artifacts

Remote (new root, nothing else written):
`/home/cereal/model-surgery-lab/artifacts/neural_forge_r12_diagnosis_v1/`
- `first-divergence/` — `request.json`, `first-divergence-ledger.json`,
  `complete.json`, `progress.jsonl`, `batches/` (256 HMAC-sealed checkpoints)
- `causal-diagnostic/layer-00/`, `causal-diagnostic/layer-12/` — `request.json`,
  `causal-ledger.json`, `complete.json`, `progress.jsonl`, `batches/`,
  `features/` (per-record safetensors), `live-feature-consumer-oracles.json`
- `nf-r12-diagnosis-first-divergence.sh`, `nf-r12-diagnosis-causal.sh`
- derived: `first-divergence-profile.txt`,
  `first-divergence-decision-boundary.txt`,
  `first-divergence-scale-direction-split.txt`,
  `causal-layer-00-summary.txt`, `causal-layer-12-summary.txt`, and their JSONs
- 4.2 GB total, dominated by `features/` and the batch checkpoints

Local: `reports/r12-diagnosis-20260804/` (2.4 MB — every ledger, request,
completion, progress journal and derived report; the per-record batch/feature
payloads stay on the server). `verify_seals.py` there re-derives all four
sealed digests from canonical JSON and cross-checks the causal `v3_binding`s
against the fetched first-divergence ledger.

## 8. Not done / caveats

- **Only layers 0 and 12 have a causal decomposition.** Each layer costs ~9 min
  end to end. Layer 10 (the other `full_attention` amplifier) and a mid-stack
  `short_conv` (e.g. layer 5) are the obvious next two — `bash
  nf-r12-diagnosis-causal.sh 10` and `… 5`, under 20 min for the pair.
- **`--transported-feature-lineage-json` was not supplied**, so both ledgers
  carry `transported_feature_oracle_available: false` and
  `live_feature_oracles_evaluated: false`. That lane needs an artifact r12 did
  not produce; the Shapley/Harsanyi evidence above does not depend on it.
- **Everything here is teacher-forced**, from the sealed parent trace cache.
  It cannot measure exposure bias, and it does not attempt to.
- **Numeric only.** Per
  `docs/BEHAVIOR_EVIDENCE_DECISION_POLICY_2026-07-23.md`, every ledger carries
  `numeric_acceptance_decision: null` and the manual whole-response review
  remains the authority. Nothing here overturns or replaces the 0/15 verdict —
  it explains it.
- **Not committed.** No `git add`, no commit, no push, per the wave brief.
