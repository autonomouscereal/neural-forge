# Correction: the kernel field in the dossiers in this directory

**Issued 2026-09-02, beside the artifacts it is about.** Read this before
comparing the `linear_attention_kernels` block or the
`lane_digest_validity.linear_attention_kernel_path` element of any dossier in
`out/` against a dossier produced by the current code.

## What is wrong with the delivered artifacts

Four dossiers here carry the field:

| dossier | `kernel_path` as written | what it actually meant |
|---|---|---|
| `oc-dossier-Q35-4B-NOTHINK.json` | `fla-0.5.2` | fla importable, `causal_conv1d` not |
| `oc-dossier-b2-Q35-4B.json` | `fla-0.5.2` | same |
| `oc-dossier-lc8k-Q35-4B.json` | `fla-0.5.2` | same |
| `oc-dossier-lc32k-Q35-4B.json` | `fla-0.5.2` | same |

`oc-dossier-Q35-4B.json` (r5) predates the field entirely and carries no kernel
element at all.

Those strings are **package inventory**, not a measured path, and they were
produced by a rule that is refuted:

> `fla` AND `causal-conv1d` present ⇒ name them; otherwise ⇒ `torch-fallback`.

The rule was inferred from the transformers log line *"The fast path is not
available … Falling back to torch implementation"*. That line is printed off
`is_fast_path_available`, which is `all()` of **four** callables
(`modeling_qwen3_5.py:217-219`, `:425-429`), and the four are bound
**independently** (`:420-423`), each with its own `or torch_…` fallback. With
`fla` present and `causal_conv1d` absent — the state of this box for every run in
this directory — the warning fires, the convolution runs in torch, and **fla's
Triton chunked Gated-DeltaNet kernel still runs**.

The proof is in this directory: `r6-lc64k-Q35-4B.log` lines 137-152 are a
traceback through

```
transformers/models/qwen3_5/modeling_qwen3_5.py:535  self.chunk_gated_delta_rule(
fla/ops/gated_delta_rule/chunk.py:567                chunk_gated_delta_rule
```

taken while line 50 of that same log carries the fallback warning.

## What the current code writes instead

`oc_calib.kernel_path_report(model)` probes the loaded model, one entry per
primitive, and stamps the report with

```
kernel_report_schema: neural-forge-omni-kernel-path-v2
```

which `oc_run_frozen_bank.py` carries into
`lane_digest_validity.linear_attention_kernel_report_schema`. On this box a v2
dossier reads

```
chunk=fla-0.5.2; recurrent=fla-0.5.2; conv-prefill=torch; conv-decode=torch
```

## The rule for a reader

* **A v1 string and a v2 string are not comparable.** The same unchanged box
  yields `fla-0.5.2` under v1 and the four-slot string under v2.
* Therefore: when the next run on this box shows this validity element moving,
  **that is the definition moving, not the surface.** Check
  `linear_attention_kernel_report_schema` on both sides first; if either side is
  absent or `v1`, the kernel element of that comparison is **unmeasured** and the
  lane digests are being compared across an unrecorded variable.
* `packages_present` in a v1 dossier is still true and still useful — it is the
  inventory, and the inventory did not change across any run in this directory
  (`fla-core 0.5.2` / `flash-linear-attention 0.5.2` installed
  `2026-09-01 14:25:13 −0600`, before the earliest run here at `00:44:18` on
  2026-09-02; `causal_conv1d` absent throughout).

No artifact was edited to produce this note. The dossiers are left exactly as
they were written; this file is the correction.
