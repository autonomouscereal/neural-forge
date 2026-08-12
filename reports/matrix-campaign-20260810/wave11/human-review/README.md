# Wave 11 (MX-O) human-review package — Cyril scores these himself

Four files. Each child file carries its recipe, its net storage, **both**
degeneracy readings, the machine verdict as an annotation, the benchmark row,
and the FULL verbatim text of all sixteen frozen-review cases paired with the
parent's. Nothing is truncated and nothing is summarized.

| file | what it is | net stored | review | per-child | per-lane |
|---|---|---:|---:|---|---|
| `PARENT.md` | the unfolded parent, same process, same prompts | 1.000000x | 12/15 | — | — |
| `COMP.MA.MXO3.md` | **the wave's frontier** — MXN4 plus one SwiGLU lane at L06 | **0.729103x** | 8/15 | dead | **alive-degraded** |
| `COMP.MA.MXO4.md` | one lane deeper (L07); where the ladder stops | 0.703396x | 5/15 | dead | dead |
| `COMP.MA.MXO2.md` | **the 0.5x target set**, solved independently | **0.497739x** | 0/15 | dead | dead |

**The question wave 11 puts to you.** `COMP.MA.MXO3` is the deepest coherent
child the campaign has produced and the first one below A9's own published
0.734512x. It is 8/15 with three charged lanes: two ride the 1024-token cap
(one of them, `debugging-off-by-one`, still leaves a real if incorrect function
when the repetitions are stripped) and one, `context-retrieval`, emits nothing.
Under ADDENDUM 9 §9.1 — your ruling on MXN4 — that reads **alive-degraded** and
the frontier moves from 0.754811x to 0.729103x. Under the per-child reading it
is dead and the frontier does not move. `COMP.MA.MXO4` is the control that says
where it stops regardless: one more lane and the content goes hollow — "do a
quick task with a small item, such as a small item" — while the structure
survives.

**What did not work, so you can weigh it against what did.** The alternating
joint co-solve (ADDENDUM 9's new mechanism) improved five of six components on
their own objective and made the composed child **worse**; the cross-layer
shared basis lost to independent truncation in every layer of every component
class at equal storage, even at a 2.98x rank uplift. Neither is in these files,
because neither produced a child worth your time. Both are in `NOTES-MX-O.md`
§4b and §5 with their numbers.

**Benchmarks** (lm-eval, `--limit 200 --seed 1234`, float32, `max_length=2048`,
one PCI-pinned sm86 RTX 3090; partial evaluations by construction):

| model | net | arc_easy | hellaswag_n | piqa_n | wino | lambada | lambada ppl | wikitext wppl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| parent | 1.000000x | 0.5450 | 0.5050 | 0.6800 | 0.5500 | 0.4600 | 24.39 | 81.80 |
| `COMP.MA.MXN4` (wave 10) | 0.754811x | 0.4250 | 0.4500 | 0.6000 | 0.5250 | 0.1700 | 370.09 | 168.70 |
| **`COMP.MA.MXO3`** | **0.729103x** | **0.4300** | 0.4300 | 0.5900 | **0.5450** | 0.1650 | 527.91 | 190.15 |
| `COMP.MA.MXO4` | 0.703396x | 0.4200 | 0.4350 | 0.5600 | 0.5250 | 0.1000 | 1477.86 | 257.83 |

`COMP.MA.MXO3` stores **0.0257x less of the parent than MXN4** and is **above**
it on `arc_easy` and `winogrande` while below on the rest.
