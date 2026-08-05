# NOTES-W10A — chimera-omni is now a covered platform, at exact 4x

Report: `reports/chimera-omni-fold-readiness-20260804/README.md`
Coverage matrix + kernel interface expectations:
`docs/CHIMERA_OMNI_FOLD_COVERAGE_2026-08-04.md`
Target: `C:/Users/cereal/Documents/ai training/chimera-omni` (read-only, not vendored)

## What this proved

The folding platform reaches the target platform. A real
`chimera_omni.modeling.ChimeraOmniModel` — built through chimera-omni's own
config API, trained on chimera-omni's own tiny-proof objective until it passes
chimera-omni's own criterion — compiles into the canonical IR, and a child with
**exactly** a quarter of its unique parameters was built from it with zero
gradient steps.

* 1,104,128 → 276,032 unique parameters. `residual_from_exact = 0`. All 63
  tensors individually exactly 4x. No quantization, no sparsity, no storage
  sharing, no sidecar.
* Fold-only guard clean: 0 optimizer steps, 0 backward calls, 0 autograd-grad
  calls, parent state unchanged.
* 73.3% next-token top-1 agreement with the parent (blank child of the same
  size: 0.8%); 91.3% of the blank → parent gap closed on the native-media
  objective; real nonempty PNG/WAV/MP4 emissions from all three models on an
  identical stimulus.

## The one architectural thing that mattered

`residual_mode = "block_attnres"` is a **block** scheme, not a residual
decoration. At every depth where `layer_index % attnres_block_size == 0`,
`_forward_trunk` hands the running prefix sum to the depth-source list and sets
`prefix_sum = None`; the attention output then *becomes* the new accumulator.
**There is no base term at those depths.**

I modelled it as an ordinary `base + delta` write first. That was wrong, and it
was wrong in the direction that matters: it would have put a residual edge in
the graph the model never executes, and any kernel reading that graph would
have folded against an addition that is not there. The IR now emits
`layer.NNN.operator_residual` with a single `delta` input and
`residual_accumulator_reset = True` at checkpoint depths. Two edges disappeared
(71 → 69) and the graph became true.

Same category of detail, second instance: the trunk passes the **live**
`completed_blocks` list into every depth selector and keeps appending to it
afterwards. A forward hook that stores `args` by reference reads whatever the
list grew into by the end of the forward, not what that call saw. The capture
snapshots the container.

## Why the exact 4x is exact

Pin `attention_head_dim` (so `num_heads * attention_head_dim` stops depending on
`hidden_size`) and pin `intermediate_size`, and **every** chimera parameter
block becomes exactly linear in `hidden_size` — embeddings, heads, norms, depth
selectors, Q/K/V/O, the fused gate/up, the media bridge, the emission head, the
aux heads. Halving twice divides the unique count by four with nothing left
over. That is an architecture property, not an accounting trick, and it is why
the residual width is the single compaction axis here.

## What the planner did, and why leaving it alone was right

`plan.json`: `planner_ready = false`, gaps
`cross_family_closed_loop_layer_operator`, `global_interface_projection`,
`global_structural_semantic_alignment`, `tokenizer_fingerprint`.

`_family_neutral_causal_layer_contract` requires both graphs to declare
`inspector == "hf-causal-decoder"` and `position_policy == "rope"`. chimera
declares `chimera-omni-native-decoder` and `"nope"` — it has no RoPE anywhere.
`CausalGQAAttentionBranchOperator` calls `build_decoder_layout()` and reads
`HFCausalCircuitEvidence`, so a planned action would have failed at execution
instead of at planning. **Do not relax that gate to let chimera through.** The
numeric fold in this run is a driver-local closed-form transaction built from
`hf_closed_form_fit`, and it is recorded as such — it has no ledger record and
no `DynamicFoldExecutor` run receipt.

## Inspector notes

`ChimeraOmniCircuitInspector`, additive block in
`src/neural_forge/dynamic_model_inspector.py`, priority 245, registered.

* It compiles **all five** sequence-mixing families and **both** channel
  mixers. `kda` → `RECURRENT_MIXER` with a recurrent `EdgeKind.STATE` self-edge
  (it really is a bounded gated delta recurrence with an explicit
  `[B,H,Dk,Dv]` state). `csa`/`hca` split into two `ATTENTION` nodes plus a
  combine, because `SparseLatentAttention` nests a whole `LocalCausalAttention`
  inside the module that also owns the summary projections. `gated_mla` →
  `ATTENTION` with `projection_style = "compressed_latent_kv"`.
  `stable_latent_moe` → router/latent-down/experts/routed-norm/latent-up/
  shared-experts/combine.
* The boundary projection is one `MODALITY_INPUT` node owning **one** tensor
  with **six** typed ports. That pairing is the encoder-free claim.
* The emission head is the repository's **second** real `MODALITY_OUTPUT`
  emitter (points3d's quantized centroid was the first) and it reads the same
  final-normed rows the text head reads — siblings, not a chain.
* Unknown families fail closed naming the family; a block whose live module
  owns a tensor its declaration does not predict fails closed naming both.

Two things worth knowing before touching it:

1. `ChimeraOmniModel` stores its config at `cfg`, not `config`, so
   `_config_payload` cannot seal it. `_ChimeraOmniLayoutView` renames exactly
   that one attribute against the same module object and the finished graph is
   re-emitted under the real model class, so the view never appears in the IR.
2. `chimera`'s `music` and `tts` lanes are not canonical IR modalities. They
   are raw-waveform segment lanes identical in construction to `audio`, so they
   enter as canonical `audio` with the chimera lane name preserved on the
   interface id and the representation. The mapping is in graph metadata under
   `chimera_media_lane_canonicalization`; it is not applied silently.

## Honest negative in the report

Aggregate holdout says the folded child closes 91.3% of the gap. The four
single-sample per-modality renders disagree in one place: **audio 1.504 for the
folded child against 1.183 for the blank child** — the fold is worse than blank
on that one sample. Video (0.175 vs the parent's 0.173) and music (0.772 vs
1.407) are strongly in favour. One sample per modality is not a measurement and
the aggregate is, but the row stays in `behavior.json` and in the README.

Text cross-entropy barely moves (parent 7.253, folded 7.204, blank 7.266)
because the tiny-proof corpus is almost all placeholder tokens. Text is not a
discriminating axis on this objective and nothing should be read into it.

## Next

1. **Kernels for `kda` / `csa` / `hca` / `gated_mla`.** §4 of the coverage doc
   is the contract: what the IR already hands a kernel, what a
   `FoldMethodSpec` + `_bundle_kind_contract` branch has to declare, and the
   numerical traps (KDA's L2-normalized q/k and its bounded decay; the
   compressed-summary branch's `1 + ceil(seq/stride)` sequence axis; MLA's
   `kv_latent_size + extra_head_dim` split point).
2. **A registered chimera fold method** so this stops being driver code and
   starts producing ledger records.
3. **Per-boundary charts for non-HF trunks.** Everything downstream of
   `ResidualBoundaryChartSet` is gated on `HFCausalCircuitEvidence`. Until
   there is an evidence bundle for a non-HF parent, chimera gets one global
   chart.
4. **Depth reduction.** Only width moved here. `num_layers` needs
   `ordered_residual_span_compose`, which is `standalone=False`.

## Files

| path | what |
|---|---|
| `src/neural_forge/dynamic_model_inspector.py` | `ChimeraOmniCircuitInspector` + helpers (additive block), registry entry, `__all__` |
| `experiments/run_chimera_omni_toy_fold.py` | the proof driver |
| `docs/CHIMERA_OMNI_FOLD_COVERAGE_2026-08-04.md` | coverage matrix, gaps, kernel interface expectations |
| `reports/chimera-omni-fold-readiness-20260804/` | sealed report + emitted artifacts |
| `tests/test_dynamic_chimera_omni_inspector.py` | 17 inspector cases (own file, to avoid merge skew) |
| `tests/test_run_chimera_omni_toy_fold.py` | 16 runner + sealed-report re-verification cases |
