# Wave 12 (MX-P) human-review package — Cyril scores these himself

Three files. Each child file carries its recipe, its net storage, its corrective
routes (if any) disclosed as **added** storage, **both** degeneracy readings, the
machine verdict as an annotation, the benchmark row, and the FULL verbatim text
of all sixteen frozen-review cases paired with the parent's. Nothing is truncated
and nothing is summarized.

| file | what it is | net stored | review | per-child | per-lane |
|---|---|---:|---:|---|---|
| `PARENT.md` | the unfolded parent, same process, same prompts | 1.000000x | 12/15 | — | — |
| `COMP.MA.MXP2.md` | **M6** — the frontier body plus the *min-drift* third lane | **0.703409x** | 6/15 | dead | **impaired** |
| `COMP.MA.MXP1.md` | **M2** — `COMP.MA.MXO4`'s exact fold set plus four corrective routes | 0.712544x | 7/15 | dead | **impaired** |

## The two questions wave 12 puts to you

**1. Does the frontier move to 0.703409x, or does it stay at 0.729103x?**

ADDENDUM 8 §8.2 says a child is a success when every component stores at 4x and
the verdict is `alive-parent-band`, `alive-degraded` **or `impaired`** — "a 6/15
or 7/15 child whose output is coherent, on-prompt language is a successful
fold-init" — and §8.3 abolished the 8/15 floor and defined the frontier as *the
deepest net stored ratio at which output is still coherent*. Under those rules,
plus ADDENDUM 9 §9.1's per-lane reading, **`COMP.MA.MXP2` at 0.703409x is the
deepest coherent child the campaign has produced** and the frontier moves.

Under the stricter reading — the 8/15 alive band the drift budget is calibrated
on — neither wave-12 child qualifies and `COMP.MA.MXO3` at **0.729103x, 8/15,
alive-degraded** remains the frontier. Both numbers are true statements about
different lines. **The rules as written say the frontier moved; the band wave 11
reported it against says it did not.** You decide which one the campaign reports.

One caveat stated plainly so it cannot be read as a thumb on the scale: wave-11's
children seal `unstated` per-lane readings (that machinery did not exist yet), so
the calculator falls back to their per-child verdicts. Their per-lane readings
were **not** re-derived by this wave, because re-reading earlier children in the
wave that wants the frontier is exactly the move the campaign's rules exist to
prevent. It is recorded as a wave-13 item.

**2. Is closed-form repair worth its storage?**

`COMP.MA.MXP1` is `COMP.MA.MXO4`'s **exact** ten-component fold set — 5/15, dead,
one lane past wave 11's frontier — plus four additive corrective routes the
parent does not have. Repair moved it **5/15 → 7/15** and its drift 0.540899 →
0.517480. That is a real, positive result for a mechanism wave 11 could not even
express.

It is also strictly dominated. The same wave's *recipe* change — swapping the
third lane from `L07` at T8/S4 to `L13` at T5/S4, chosen by measurement instead
of by depth order — bought **twice the drift for one one-hundred-sixty-second of
the storage**:

```
M2 corrective routes   0.011146 drift per million stored parameters
M6 lane shopping       3.961    drift per million stored parameters    ~355x
```

The most interesting single number in the package is in `COMP.MA.MXP1.md` §1b:
the route at **L13**, a block that was never folded, explained **0.0%** of that
block's residual gap on rows it never saw, returned a held-out gap of
**1.014878** — larger than the gap it was correcting — and made the composed
drift **worse**. Repair only has purchase where a fold's own error is still
linear in the block's input, and that purchase decays fast with depth: 28.0% at
L05, 10.4% at L06, 10.0% at L07, 0.0% at L13.

## What did not work, so you can weigh it against what did

The first M2 run died four minutes into its second route with the frozen
collector's own refusal — `Layer 5 residual endpoints do not reproduce its
output` — because a tensor added to the feed_forward *module's* output is not
inside `mlp.down`, and the collector verifies `layer_output == post_operator +
mlp.down` on every layer. The collector was right and it is frozen; the write
moved one module deeper. The refused run is sealed as an `infra` row and its log
is kept. It is in `NOTES-MX-P.md` §4.1 with its numbers.

## Benchmarks

lm-eval, `--limit 200 --seed 1234 --batch_size 4`, `max_length=2048`, float32,
one PCI-pinned sm86 RTX 3090; partial evaluations by construction.

| model | net | arc_easy | hellaswag_n | piqa_n | wino | lambada | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000x | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `COMP.MA.MXO3` (w11) | 0.729103x | 0.4300 | 0.4300 | 0.5900 | 0.5450 | 0.1650 | 527.91 | 190.15 |
| **`COMP.MA.MXP2`** | **0.703409x** | 0.4250 | 0.4250 | **0.5850** | **0.5400** | **0.1450** | **857.20** | **231.96** |
| `COMP.MA.MXO4` (w11) | 0.703396x | 0.4200 | 0.4350 | 0.5600 | 0.5250 | 0.1000 | 1477.86 | 257.83 |

**At identical storage and identical component count, the shopped lane beats the
autopilot lane on five of seven columns and halves lambada perplexity.** That is
the M6 result stated continuously, and it did not cost a parameter.

`COMP.MA.MXP1` was not benchmarked: its corrective routes are forward hooks, not
`state_dict` entries, so the materializer cannot write an HF child that carries
them without custom modeling code. Stated as a gap, not smoothed over — see
`NOTES-MX-P.md` §6.
