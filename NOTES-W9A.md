# NOTES-W9A — the first real multimodal fold: SmolVLM2-256M-Video-Instruct

Wave W9A folded a **real, pretrained, downloaded multimodal checkpoint** for the
first time.  Every multimodal proof in this tree before it — NEO native vision,
the native audio / video / points3d lanes — ran against a parent the test that
folded it had just synthesized.  The parent here is
`C:/Users/cereal/Documents/ai training/models/SmolVLM2-256M-Video-Instruct`,
`model.safetensors` 1,025,998,224 bytes,
`sha256 dcc8f4c00d30d70bb3028a1bfc373e1b1972e25919f8788183cae4ae72431ce2`
as sealed in every run receipt.

Two artifacts:

* `src/neural_forge/dynamic_model_inspector.py` — additive
  `SmolVlmCircuitInspector` (plus `_SmolVlmTextDecoderView`,
  `_smolvlm_module_paths`, and the `SMOLVLM_*` constants), registered in
  `default_inspector_registry()` at priority 250.
* `experiments/run_smolvlm_real_multimodal_fold.py` — the sealed driver, and
  `reports/smolvlm-real-fold-20260804/` its output.

Everything that stopped inside `src` rather than being worked around silently is
written below with the exact symbol and the exact message.

---

## 1. Inspector design

### 1.1 It is a wrapper class, not a marker gate

`supports()` is two conditions and no marker key:

```python
getattr(config, "model_type", None) == "smolvlm"
and _qualified_class(model) == (
    "transformers.models.smolvlm.modeling_smolvlm.SmolVLMForConditionalGeneration"
)
```

A `SimpleNamespace(config=SimpleNamespace(model_type="smolvlm"))` is refused by
`supports()` and, if `inspect()` is called on it directly, by a
`DynamicModelInspectionError`.  That is the same shape
`NeoNativeVisionCircuitInspector` uses — a real structural gate on a real
runtime class — rather than the config-marker pattern the native audio / video /
points3d lanes use for their synthesized parents.

### 1.2 The vision tower and connector are MODALITY_INPUT-side operators

Seventy-eight nodes cover everything between raw normalized RGB crop pixels and
decoder-width image-token embeddings:

| node | kind | owns |
|---|---|---|
| `input.image.patch_embedding` | `MODALITY_INPUT` | `patch_embedding.{weight,bias}` |
| `input.image.position_embedding` | `POSITION_ENCODING` | `position_embedding.weight` |
| `input.image.encoder.NNN.attention_norm` | `NORM` | `layer_norm1.{weight,bias}` |
| `input.image.encoder.NNN.attention` | `ATTENTION` | `self_attn.{q,k,v,out}_proj.{weight,bias}` |
| `input.image.encoder.NNN.attention_residual` | `RESIDUAL_ADD` | — |
| `input.image.encoder.NNN.mlp_norm` | `NORM` | `layer_norm2.{weight,bias}` |
| `input.image.encoder.NNN.mlp` | `DENSE_MLP` | `mlp.fc{1,2}.{weight,bias}` |
| `input.image.encoder.NNN.mlp_residual` | `RESIDUAL_ADD` | — |
| `input.image.post_layernorm` | `NORM` | `post_layernorm.{weight,bias}` |
| `input.image.pixel_shuffle` | `INTERFACE_PROJECTION` | — (parameter free) |
| `input.image.modality_projection` | `INTERFACE_PROJECTION` | `connector.modality_projection.proj.weight` |
| `input.image_text_embedding_fusion` | `INTERFACE_PROJECTION` | — (parameter free) |

Every learned tensor has exactly one owner: `_GraphBuilder._module_parameters`
raises on a second claim, and `_GraphBuilder.finish` raises unless the captured
set equals the model's parameter set exactly.  On the real checkpoint the graph
accounts for **256,484,928** parameters, which is `named_parameters` to the
element.

Every vision port carries `modality="image"`.  The vision encoder layers are
emitted with `layer_index=None` and a `vision_layer_index` *attribute*, because
`graph.layer_indices` is the decoder depth axis the planner iterates; giving the
tower a layer index would make the planner try to fold vision layers as decoder
layers.

### 1.3 Fail-closed on unexpected vision-parameter placement

After the tower and connector nodes are emitted the inspector computes the set
of captured parameter paths under `model.vision_model.` and `model.connector.`
and compares it to the set the model actually has.  Any difference raises

```
SmolVLM's vision tower and connector own learned parameters outside their
proven front-end operators; unowned=[...], foreign=[...].
```

`tests/test_dynamic_smolvlm_inspector.py` exercises that with a
`register_parameter("unmodelled_gain", ...)` on the tower, plus four more
fail-closed variants: a biased modality projection, a re-strided patch stem, a
truncated encoder depth, and a non-llama `text_config.model_type`.

### 1.4 The decoder is not re-derived — it is *inspected*

`_SmolVlmTextDecoderView` renames `model.text_model` to `model` and keeps
`lm_head`, against the **same module objects**.  `HFCausalDecoderInspector`
then proves that decoder with its own `build_decoder_layout` contract, and
`SmolVlmCircuitInspector` transcribes the proven subgraph into the composite
graph with

* node ids prefixed `decoder.`,
* paths rebased by identity lookup (`_smolvlm_module_paths` indexes every child
  module by `id()`), not by string surgery, so a rebase that does not resolve to
  a real registered module raises instead of silently inventing a path,
* `role_prefix` recovered from the proven roles, so `_module_parameters` re-runs
  the ownership check on the composite side too.

The composite metadata records `decoder_inspector: "hf-causal-decoder"` and
`decoder_graph_fingerprint`, so the decoder half of a SmolVLM graph is
traceable to the exact graph the shared inspector emitted.  If the decoder fails
its contract the composite inspection fails closed with
`SmolVLM's text decoder failed the shared causal-decoder contract: ...`.

The alternative — factoring `HFCausalDecoderInspector.inspect`'s ~420-line body
into a shared emitter — would have been a behaviour-preserving refactor of an
existing class, and the task scope is additive.  Running the real inspector and
transcribing its output is strictly stronger anyway: it cannot drift.

### 1.5 Video is recorded, not invented

SmolVLM2-Video samples video into image frames and pushes each frame through the
same tower; the checkpoint owns no video-specific learned tensor.  The inspector
therefore emits **one image lane** and records, in metadata:

```
video_lane: "video enters as sampled image frames through the same vision
tower; this checkpoint owns no separate video front end, so no video modality
interface is emitted"
```

plus `video_frames_share_this_operator: true` on the patch-embedding node.  A
`MODALITY_VIDEO` interface here would be a claim the weights do not support.

---

## 2. Environment blockers — the real processor cannot be constructed

The task asked for the probe images to go through "the real processor".  They do
go through the real *image* path.  `SmolVLMProcessor` itself cannot be
instantiated in this environment, for three separate reasons, none of them
Neural Forge's.

**2.1 `AutoProcessor` refuses the checkpoint outright.**

```
ValueError: Could not load any image processor class for
.../SmolVLM2-256M-Video-Instruct. The model configuration resolves to the
following image processor classes: pil: SmolVLMImageProcessorPil,
torchvision: SmolVLMImageProcessor. None of these classes could be imported.
Missing optional dependencies: torchvision.
```

`torchvision` is not installed and installing a build matched to
`torch 2.13.0+cpu` into the shared repository venv is not a change this task
should make.

**2.2 The PIL backend is gated on torchvision but does not use it.**

```
ImportError: image_processing_pil_smolvlm requires the Torchvision library but
it was not found in your environment.
```

`transformers/models/smolvlm/image_processing_pil_smolvlm.py` imports
`math`, `numpy`, `...image_processing_backends.PilBackend`,
`...image_processing_utils.BatchFeature`, `...image_transforms`,
`...image_utils`, `...processing_utils` and `...utils` — no torchvision
anywhere.  The requirement is attached by the package's lazy-module import
structure, and

```python
importlib.import_module(
    "transformers.models.smolvlm.image_processing_pil_smolvlm"
).SmolVLMImageProcessorPil.from_pretrained(directory, local_files_only=True)
```

constructs and runs correctly, `backend == "pil"`.  That is what the driver
does.

**2.3 `SmolVLMProcessor.__init__` has two more video-side gates.**

```
ImportError: Package `num2words` is required to run SmolVLM processor.
Install it with `pip install num2words`.
```

`num2words` is used at exactly one site in `processing_smolvlm.py`
(`expand_text_with_video_tokens`, `frame_count=num2words(num_frames)`); the
constructor gate is unconditional.  With `num2words` present the constructor
still fails:

```
TypeError: Received a SmolVLMVideoProcessor for argument video_processor, but a
BaseVideoProcessor was expected.
```

`ProcessorMixin.check_argument_for_proper_class` resolves `BaseVideoProcessor`
off the `transformers` root, where it is a torchvision placeholder, so a real
`SmolVLMVideoProcessor` (which imports fine by module path) fails the isinstance
test against it.

**What the driver does instead.**  `RealSmolVlmImagePath` reproduces the image
branch of `SmolVLMProcessor.__call__` from the same components, call for call:
the real `SmolVLMImageProcessorPil`, the real
`transformers.models.smolvlm.processing_smolvlm.get_image_prompt_string`, and
the real tokenizer.  It is sealed in every `evidence.json` under
`image_processor`, including the class name, the backend, the expansion callable
and `do_image_splitting: false`.  The video branch is not used.

`do_image_splitting` is turned **off** deliberately: with splitting on, a 64x64
probe becomes 17 crops of 512x512 and 1088 image tokens per prompt, which is a
1141-token window before any text.  With it off each probe is one crop and
exactly 64 image tokens, an 81-token window, and the full-depth parent still
answers correctly (`" The square is red."`), so nothing about the probe's
readability was traded away.

---

## 3. What `src` could not do, and what happened instead

### 3.1 One-forward evidence refuses `inputs_embeds`, and it is right to

```python
if "input_ids" not in inputs or inputs.get("inputs_embeds") is not None:
    raise HFCausalCircuitEvidenceError(
        "Full-model folding evidence requires input_ids, not inputs_embeds."
    )
```

(`src/neural_forge/hf_causal_circuit_evidence.py:657`)

An evidence bundle whose token rows are not token rows cannot be audited.  But
SmolVLM's image rows *are* token rows: `<image>` (id 49190) is a real vocabulary
row, and the composite replaces those rows' embeddings with connector output
before the first decoder layer.  `ImageTokenMergeHook` performs exactly that
substitution — through the model's **own** `SmolVLMModel.inputs_merger` — as a
forward hook on `text_model.embed_tokens`, and fails closed on any batch other
than the one it was sealed against:

```
The image merge hook was reached with token ids it was not sealed against;
image conditioning cannot be claimed for them.
```

`collect_hf_transport_trace` hooks the decoder blocks, the final norm and the
head; it does not hook `embed_tokens`, so there is no hook-ordering interaction.
The receipts count the merge (`parent_image_merge_calls: 1`) beside the forward
(`parent_decoder_forward_calls: 1`) and the tower
(`parent_vision_tower_forward_calls: 1`) so the "one forward" claim is stated at
the granularity it is true at: **one decoder forward**, preceded by one vision
tower pass whose output that forward consumes.

A `src` fix would be a variant capture that accepts `input_ids` *plus* a sealed
`(position_mask, replacement_rows)` pair and seals the replacement digest into
the evidence — i.e. modality-conditioned token rows as a first-class evidence
shape.  That touches `HFCausalCircuitEvidence.__slots__`, its seal, and
`collect_hf_transport_trace`'s input normalization.

### 3.2 The planner cannot own the connector, so the driver carries it

`DynamicFoldPlanner` only emits cross-width branch and chart-projected interface
actions when **both** graphs report `inspector == "hf-causal-decoder"`
(`_cross_width_boundary_chart_crossing`, `_family_neutral_causal_layer_contract`
in `src/neural_forge/dynamic_fold_planner.py`).  A `smolvlm` composite graph
does not, by construction, so the fold is planned on the two *decoder* graphs
and the composite graphs are sealed alongside as structural evidence
(`fold.json` -> `graphs.parent_composite_fingerprint` /
`child_composite_fingerprint` / `*_composite_metadata`).

That leaves the vision tower and the connector outside any fold action:

* **Vision tower** — geometry is unchanged, so all 197 tensors / 86,433,024
  elements are copied and then re-compared tensor by tensor
  (`torch.equal`) before the carry receipt is written.  This is `copy_exact`
  semantics executed by the driver rather than by `ExactCopyOperator`.
* **Connector** — cannot be carried unchanged.  `modality_projection.proj` is
  `Linear(12288 -> 576)` and it writes into the residual stream at *the same
  boundary the token embedding writes*.  The only projection consistent with the
  embedding rows the interface operator installs is the same interface-locked
  chart, applied to the connector's output coordinates:
  `W_child = (A0^T W_parent)`, computed as
  `embedding_chart.fold_consumer_weight(W_parent.T).T`.

  That is checked, not asserted: after the carry the driver runs the *child's*
  own vision tower and connector on the probe crops and compares the result to
  the chart encoding of the parent's image features.  Measured
  **2.01e-07** relative RMSE (float32 round-off) on both the L3 and L6 runs.

A `src` fix would be a `ModalityInterfaceProjectionOperator` for a
`MODALITY_INPUT`-side producer that writes an interface-locked residual
boundary, plus a planner clause that lets a composite graph carry a
decoder-planned chart set into its modality lane.  Both are real additions, not
rewrites; neither is in this task's ownership.

### 3.3 A llama child reports no fitted Q/K gains

`CausalGQAAttentionBranchOperator` classifies the target's per-head
normalization through `_target_query_key_norm`, which returns `"identity"` for a
llama child and omits `query_norm_gain_relative_rmse` /
`key_norm_gain_relative_rmse` from the transaction report.  The cross-family
runner reads those keys unconditionally because its child is qwen3.  The driver
here records whichever of them the operator emitted.  Not a defect — a
documented consequence of folding llama into llama instead of llama into qwen3,
and it removes one source of freedom the cross-family run had.

---

## 4. Budget: why a bounded layer prefix

The evidence bundle stores thirteen per-token tensors per layer.  At this run's
48 x 96 = 4608 positions that is ~34.6 KB per position per layer:

| decoder layers folded | evidence bundle | chart compile | execute | wall clock |
|---|---|---|---|---|
| 1 | ~0.16 GB | 46 s (3 boundaries) | 121 s (2 branches) | 3.8 min |
| 3 | ~0.48 GB | 84 s (7 boundaries) | 338 s (6 branches) | 8.0 min |
| 6 | ~0.96 GB | 60 s (13 boundaries) | **abandoned, see below** | — |
| **30 (full depth)** | **~4.8 GB** | ~20 min (61 boundaries) | ~60 min (60 branches) | **~1.5 h** |

The workstation reported **6.5 GB free** when this wave started (31.9 GB total,
79% loaded by other work).  A 4.8 GB evidence bundle plus the chart solves plus
three fp32 SmolVLM copies does not fit, and a fold that fails closed at any
layer rolls the child back to blank — an hour and a half spent for no child and
no decode.  So the runs here are bounded prefixes, `--layers` is sealed into
every receipt as `folded_layer_prefix` / `full_depth` / `bounded_layer_prefix`,
and **no claim is made about full-depth behaviour**.

The prior art matters here: `reports/cross-family-real-20260804-pretrained`
found that a *real pretrained* SmolLM2-135M parent fails closed at layer 3's
attention branch (endpoint relative RMSE 1.092 against the 1.0 no-branch bound)
where a random parent at the same depth does not.  SmolVLM2's decoder is
SmolLM2-shaped, so the same wall was the expected outcome here.  The three
layers that did run all folded (worst branch 0.947 against the 1.0 bound), and
the run that would have shown whether layer 3 refuses did not survive the
workstation — see 4.1.  **No claim is made about where this checkpoint's depth
wall is.**

### 4.1 The six-layer arm was abandoned twice, both times on resources

A `--layers 6` arm was attempted twice at the same 48 x 96 evidence corpus.  It
reached the executor both times and produced **no** fold refusal.

* **Attempt 1**, sharing the workstation with the three-layer arm, died inside
  the executor's own post-acceptance guard:

  ```
  smolvlm multimodal fold failed closed: Accepted evidence refresh evidence
  payload integrity verification failed.
  ```

  `_assert_fold_evidence_payload_unchanged` wraps `BaseException` from
  `_fold_evidence_payload_details`, which streams every evidence tensor through
  `tensor.reshape(-1).view(torch.uint8).numpy()`.  An allocation failure there
  is reported as an integrity failure.  Nothing mutated the evidence — the guard
  could not re-read it.

* **Attempt 2**, alone on the machine, got past the charts (13 boundaries,
  60 s) and the carry and was executing branches when free disk on `C:` fell
  from 14.0 GB to 1.8 GB in about twelve minutes.  The consumer was the run's
  own committed memory driving Windows pagefile expansion: killing PID 24968
  returned free disk to 8.0 GB within five seconds.  It was terminated
  deliberately, because filling the system volume would have broken concurrent
  work that has nothing to do with this wave.  The partial `L6-prefix/`
  directory was removed rather than committed half sealed.

So the six-layer depth is **untested**, not refused.  What the two attempts do
establish is that the ceiling on this workstation is a *committed memory*
ceiling around a one-gigabyte evidence bundle, well before the ~4.8 GB a
full-depth run would need — which is a stronger argument for shrinking the
evidence representation than for finding a bigger box.

Evidence sizing is not free either.  A first bounded run at 6 text prompts + 3
probes (~410 fit rows) failed closed at **layer 0's** attention branch:

```
Cross-width causal attention branch operator endpoint relative RMSE
1.1904438462717335 is at or above the no-branch baseline bound 1.0 at
boundary 'layer000.mlp_in'.
```

At 36 text prompts + 12 probes (2252 fit rows against 1152 target MLP neurons,
a 1.95x ratio, matching the cross-family run's 2.06x) the same layer folds at
0.545.  Starved evidence is indistinguishable from a hard boundary if you only
run it once; that refusal is a measurement of the corpus, not of the parent.

---

## 5. What the runs found

Numbers, per-branch tables and the complete verbatim decodes are in
`reports/smolvlm-real-fold-20260804/README.md`.  The three findings worth
carrying forward:

1. **A mixed text+image residual is *more* compressible into 432 of 576
   coordinates than a text-only one.**  Held-out retained energy median 0.983
   (min 0.981) across the L3 boundaries, against 0.972 (min 0.957) for the
   text-only SmolLM2-135M run at the same depth and width.  The image rows are
   64 consecutive connector-driven positions and they do not fight the text
   rows for the basis.

2. **The folded child tracks its parent better on image-conditioned rows than
   on text-only rows.**  L3 teacher-forced logit relative RMSE **0.332** on
   image-conditioned held-out positions against **0.634** on text-only ones, and
   top-1 agreement **50.4%** against a 45.4% modal floor (image, +5.0 points)
   versus **25.4%** against a 33.4% modal floor (text, -8.0 points).  The image
   lane is the one where the fold buys something above the trivial floor.  The
   most likely mechanism is mechanical rather than semantic: the connector is
   carried through the *same* chart as the embedding at machine precision, so
   image positions enter the child's residual stream almost exactly where the
   chart says they should, while text positions enter through 49,280 chart-
   projected embedding rows and a fitted final-norm gain.

3. **A layer prefix of a VLM is not a VLM, and the decodes say so loudly.**  The
   3-layer parent prefix emits `,   and,   and,` forever on every prompt,
   image-conditioned or not; the folded child emits scattered rare tokens.
   Greedy prefix agreement is 0 of 24 tokens on every held-out prompt in both
   lanes.  The full-depth unfolded parent, decoded on the same prompts and
   printed beside them, answers `" The square is red."`.  That gap is the
   honest size of what a bounded-prefix multimodal fold has proven so far:
   the *mechanism* end to end on a real multimodal checkpoint, and none of the
   behaviour.

---

## 6. Environment changes

`num2words 0.5.14` was installed into the repository venv while diagnosing
section 2.3 and **removed again** once it was established that
`SmolVLMProcessor` still cannot be constructed with it present.  No other
package was added, removed or upgraded.  No file outside this task's ownership
was modified.
