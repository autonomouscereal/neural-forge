# Wave 12 (MX-P) artifacts

* `NOTES-MX-P.md` — the wave, its arithmetic, and its five sentences.
* `probes/lane-shopping.json` — M6: sixteen `(lane, technique, level)` variants
  priced against one fixed base, with the two sealed numbers that validate the
  probe (the base reproduces `COMP.MA.MXO3`'s body drift; `L07.T8.S4` reproduces
  `COMP.MA.MXO4`'s sealed marginal) and the base-restoration check after every
  candidate.
* `cells/COMP.MA.MXP*/` — sealed receipts, diagnostics, composed-child tensor
  digests, frozen-review dossiers, and — for a child that carries them — the
  ADDENDUM 10 corrective-route manifest.
* `human-review/` — Cyril scores these himself: one file per child plus
  `PARENT.md`, every response verbatim and unedited, both degeneracy readings
  printed, the machine verdict as an annotation.
* `ledger.jsonl.pre-merge-backup` — the authoritative ledger's bytes before the
  wave-12 merge, verified byte-identical to the original by sha256 before the
  merge ran.
* `pruned-tensors-manifest.json` — what retention deleted, with digests, sizes
  and the recipe each set is re-derivable from.

Deployments this wave ran under, and why there is more than one:

| commit | used for |
|---|---|
| `4d8774f6` | ADDENDUM 10 + the corrective class; generated `COMP.MA.MXP2` and the first `COMP.MA.MXP1` attempt |
| `dcfe22ff` | the sealed per-lane verdict (grading only, briefly) |
| `f753b9e7` | the corrective write moved into `mlp.down`; generated every rung after, and sealed every verdict |
