# wave 13 (MX-Q) — index

The last mechanism class on the matrix (**M5**, the Procrustes head merge) and
the closing report of the fold-only campaign.

| file | what it is |
|---|---|
| `NOTES-MX-Q.md` | the wave note: the directive that changed the rules, the kernel, the price, the children, the estate |
| `cells/` | one directory per sealed child: receipt, behaviour-reviewed receipt, diagnostics, dossier, state digest |
| `probes/` | `headmerge-vs-swiglu.json` (M5's lane price against the MXO3 base, with the SwiGLU reference that calibrates it) and `deep-ladder.json` (the greedy ladder past MXP2) |
| `human-review/` | Cyril's package: full verbatim responses, parent-paired, machine verdict as an annotation |
| `run/mxq/` (in `../run/mxq/`) | the wave's tooling |

**The closing report is not here.** It is `docs/PHASE1_FOLD_ONLY_FRONTIER_2026-08-12.md`,
because it is a statement about the program rather than about one wave, and
`docs/CURRENT_FOLDING_GOAL.md` carries a dated pointer to it.

## Source and surface

* Kernel commit `ebc16da4`, planner commit `63e95cd8`, deployment
  `neural-forge-matrix-63e95cd8`.
* Review surface verified on extraction **before any child was generated**:
  protocol `e5e37768…`, seal `7b859172…`, prompt pack `0ee6bf14…`.
* Knob file `e355e4e7bb8a3058…`, byte-identical to waves 7–12.
* Every heavy artifact under `/mnt/ai_data/neural-forge-matrix/wave13/`.

## The one-line result

M5 is **expressible at 4x** (0.239583x per attention block, correcting
`NOTES-MX-P` §8's 0.21875x), **exact at its identity gate** (≤ 8.7e-15), and
**8–13× more expensive per parameter than a SwiGLU lane** — and the stacked
child that carries it, `COMP.MA.MXQ4` at 0.682581x, is the deepest child the
campaign has built and reads **0/15**. The price was a heuristic; the 0/15 is
the measurement.
