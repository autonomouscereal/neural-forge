# C10 D4 verdict: the S5 K3 union target is `a003` = `--union-target trunk/additive:0.03`

Judged 2026-10-11 from `RESULTS.json` (sha256 `04d1ad13bfe4723a866ded1008c9100bcf90eb3c8c28258f73e46758c9a4a911`),
`DESIGN.md` (sha256 `5715f05b4b37494eb02ae56dc1aeac25788cf0962f12647b0aa7f07e27866828`), `eval/virtual_check_p1..p4.json`
and `sweep/sweep_receipt_L*.json`. Every number is a diagnostic (owner policy 2026-10-06). This is a design choice made
from measurements. It is not a gate, and it discards nothing.

## Decision

- **Chosen:** `a003`, spec `trunk/additive:0.03`. The target is `Y = 1.0 * FFN_trunk(x) + 0.03 * TERM_k3(x)`, with
  the prior `[trunk_down | 0.03 * prior_k3]`.
  - Fixed settings, the same as the sweep: K3 dense L1-92 with the c17 92->64 alignment, c17 charts L00-L63,
    `--k3-channels 12288` and `--chart-bias-ceiling 0`.
  - Real install: `c10_union_install.py` 1.5.0, closed loop, adding `--union-target trunk/additive:0.03`.
- **Runner-up:** `g01`, spec `trunk/rms:0.1` (mean w = 0.081).
- **Rejected despite the lowest held NLL:** `a01` and `b01` (see below).
- **Plain statement:** K3 adds **no measurable held-out text benefit at the fold stage**.
  - The held-NLL gains are in-sample.
  - On the clean R-Q prompts, the K3-carrying children agree with the 27B parent on 110/210 positions. The S3 trunk
    agrees on 108/210, which is within noise.
  - `a003` is chosen because it is the largest K3 contribution that keeps the trunk coherent. The K3 channels have to
    be present in the S5 child for later distillation. It is not chosen because K3 improves the text.

## The numbers behind it

Reference values:

| reference | held NLL | top-1 vs parent (held) | top-1 vs parent (R-Q) | KL(· ‖ parent) held |
|---|---|---|---|---|
| S3 trunk | 3.756 (3.760 on the 3090 parts) | 0.630 | 0.514 (108/210) | 1.981 |
| 27B parent | 4.290 | 1 | 1 | 0 |

Per-variant values:

| variant | K3 target energy | R-Q top-1 vs trunk | R-Q top-1 vs parent | held NLL (dNLL vs trunk) | held KL vs trunk | held KL vs parent | held top-1 vs parent | max norm ratio (layer) | final relL2 | veto binds |
|---|---|---|---|---|---|---|---|---|---|---|
| a0 (control) | 0.000 | 0.967 (203/210) | 0.510 | 3.751 (-0.004) | 0.023 | 1.972 | 0.632 | 1.001 (L63) | 0.048 | 64/64* |
| **a003** | **0.124** | **0.857 (180/210)** | **0.524** | **3.414 (-0.341)** | **1.111** | **2.623** | **0.543** | **1.166 (L10)** | **0.407** | **7/64** |
| g01 | 0.099 | 0.890 (187/210) | 0.524 | 3.660 (-0.101) | 0.444 | 2.109 | 0.617 | 1.061 (L41) | 0.270 | 8/64 |
| a01 | 0.294 | 0.643 (135/210) | 0.490 | 2.371 (-1.384) | 2.685 | 4.157 | 0.473 | 1.816 (L9) | 1.023 | 8/64 |
| b01 | 0.314 | 0.610 (128/210) | 0.467 | 2.435 (-1.321) | 2.749 | 4.232 | 0.467 | 1.763 (L9) | 0.997 | 8/64 |
| g03 | 0.286 | 0.690 (145/210) | 0.490 | 3.156 (-0.605) | 1.586 | 3.096 | 0.545 | 1.218 (L41) | 0.582 | 17/64 |
| legacy (old target) | 0.762 | 0.000 (0/210) | 0.000 | 12.339 (+8.584) | 11.255 | 11.775 | 0.001 | 27.281 (L7) | 3.455 | 64/64 |

\* In `a0` the veto binds trivially: with w = 0 the union adds nothing, so the trunk prior wins in every layer.
`a03`, `b025` and `b05` collapse (R-Q top-1 vs trunk 0.048-0.067, held NLL 7.3-10.4) and are not considered.

## Why `a003`

### 1. It wins under both readings of the brief

- **The pre-registered rule** (DESIGN.md, written before any result) gives a coherent set of {a0, a003, g01}.
  `a003` carries the most K3 of the three (energy 0.124 against 0.099 for `g01`, a gap larger than the 0.01 tie band).
- **The judge's rule** asks for the lowest held NLL among coherent variants. Within that same set, `a003` is also
  the lowest: 3.414, against 3.660 for `g01` and 3.751 for `a0`.

### 2. Why not `a01` or `b01`

These two have the lowest held NLL of any variant (2.37 and 2.44, which is below the trunk and far below the 27B
parent's 4.29). They are rejected because that NLL is not evidence of a better child:

- **The held rows are contaminated.** The c17 charts were fitted on trunk taps of these same six held rows
  (`trunk_taps/s3native-c2k4352-held-all`, 24,576 rows per layer). So the K3 term on those rows is in-sample.
- **They read better than the parent while moving away from it.** An NLL nearly 2 nats below the 27B parent, on text
  the parent is supposed to model better, while held KL to the parent doubles (1.98 to 4.16) and held top-1 agreement
  with the parent falls (0.630 to 0.473). That is the signature of the K3 path fitting these particular rows, not of
  better language modelling. The same tension appears, much smaller, in `a003`: its held KL to the parent is 2.62
  against the trunk's 1.98.
- **On the clean R-Q prompts they are slightly worse than the trunk** at matching the parent: 0.490 and 0.467,
  against 0.514.
- **They fail two of the three coherence conditions.** R-Q top-1 vs trunk is 0.64 / 0.61 against the 0.80 line, and
  the residual norm reaches 1.82x / 1.76x at L9. Both numbers are open loop. The legacy row shows that open-loop
  residual growth survives closing the loop (open loop 0/210, closed-loop rehearsal 7/210), so a 1.8x growth at L9
  is a real risk of trunk collapse in the closed-loop install. Cyril's requirement is that the child works and
  produces coherent text, so a design that trades that away for an in-sample NLL is not acceptable.

### 3. `a003` keeps the trunk intact

- 180/210 R-Q top-1 agreement with the trunk.
- R-Q KL to the trunk 0.100.
- Residual norm at most 1.166x (L10), ending at 1.03x.
- The veto binds in only 7 of 64 layers, so the union fit actually wins in 57.
- No layer was refused, and all logits are finite.

### 4. Out-of-sample evidence is a tie

The clean signal is R-Q agreement with the parent: a003 110/210, g01 110/210, trunk 108/210, a0 107/210.
A difference of two positions is within noise. That is why the verdict says K3 adds no measurable benefit at the
fold stage.

## Runner-up: `g01` (trunk/rms:0.1)

`g01` is the safer design and carries about 20% less K3 (energy 0.099 vs 0.124). It keeps the trunk much closer:

| | g01 | a003 |
|---|---|---|
| R-Q top-1 vs trunk | 187/210 | 180/210 |
| held KL vs trunk | 0.444 | 1.111 |
| max norm ratio | 1.061 | 1.166 |
| held top-1 vs parent | 0.617 | 0.543 |
| R-Q agreement with parent | 0.524 | 0.524 |

`g01` is the design to switch to if the closed-loop `a003` install measures below the coherence line. That switch
would be a design re-choice, not a refusal. `a003`'s margin over the 0.80 line on R-Q is only 0.057, and
closed-loop compounding is unmeasured.

## What this means for the composite

1. **D4 is answered.** The S5 child is the S3 trunk with a K3 L1-92 union installed at about 12% of the target's
   energy.
   - Structure is unchanged by the choice of target: 12,288 K3 columns per layer and a header of 26,944,218,624
     parameters.
   - The column-level >=4x accounting is unchanged (`qwen.mlp` 3·5120·4352·64 plus `k3.ffn` 3·5120·12288·64).
   - The media budget is unchanged (12,288 <= 12,538).
   - Only the refit `down` values differ, not the structure.
2. **The child's text capability will be the trunk's.** The K3 source is present as weights, but it is attenuated
   about 33x relative to the charted term. The charted K3 sum is about 18.5x the trunk MLP output (L05: 3822 vs 206),
   so it cannot enter at native scale without collapsing the trunk. K3 capability therefore has to come out later,
   in distillation or capability evaluation. It does not come from the fold. The scale mismatch is a property of the
   c17 charts. A later chart refit with output-norm matching is the place to recover K3 strength, not a larger `w`.
3. **S7 sequencing is unchanged.** The media charts must still be recaptured and refitted on the S5 prefix once the
   real C10 child exists (S7_READINESS §2, step 5).

## Blockers before the real closed-loop install (to fix, not gates)

1. **The single-target solver still raises on an indefinite Gram at weak ridge rungs.** The sweep hit this in 15 of 64
   layers (rel 1e-6 to 1e-4). The rung-skip in 1.5.0 exists only on the sweep path. Port it to the single-target
   path, with a test, before launching. Otherwise the S5 install will most likely stop at the first such layer.
2. **Disk.**
   - Root has about 91 GB free with the sweep's `sweep/variants` (102 GB) and `sweep/shared` (21 GB) still present.
   - The C10 full install peaks at about 87 GB (S7_READINESS §5) and the floor is 60 GB.
   - The sweep owner has to archive or prune the non-chosen variant partials before launch. The open-loop `a003`
     partials are not reusable by a closed-loop install. Nothing was deleted by this verdict.
3. **Design checkpoint during the install.** Run `c10_virtual_child_check.py` on the closed-loop child after L0-21.
   If R-Q top-1 vs trunk falls below 0.80 or the residual norm exceeds 1.5x, re-choose `g01`. This is a design
   re-choice from a measurement, not a discard.

## Not verified

- **No closed-loop install for `a003` or any other variant.** The pick rests on open-loop fits evaluated as virtual
  children.
- **No held-out K3 measurement.** All six held rows are in the c17 chart-fit set. A K3 benefit can only be claimed
  after an evaluation on fresh rows that were never used for c17 or C10 fitting, and none exist yet.
- **No decoded text.** Coherence here means top-1 agreement, KL and NLL. Greedy decodes of the R-Q prompts from the
  real S5 child have to be read before the child is called working.
- **The parent held-NLL anomaly is unresolved.** The parent reads 4.290 against 3.756 for the trunk on the held rows,
  but better on R-Q.
- **Cross-card bf16 noise** of about 0.005 nats between the RTX 8000 and 3090 parts.
