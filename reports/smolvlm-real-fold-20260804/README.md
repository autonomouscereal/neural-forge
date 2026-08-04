# The first fold of a real pretrained **multimodal** checkpoint — SmolVLM2-256M-Video-Instruct (2026-08-04)

Runner: `experiments/run_smolvlm_real_multimodal_fold.py`
(`sha256 3127ce32adecee42e6f1711103b86aede61294337affb8a69cfc3a7a665ccff9`).
Workstation, CPU only, `torch 2.13.0+cpu` / `transformers 5.13.0`, six threads,
no network, no server, `HF_HUB_OFFLINE=1`.

Every multimodal proof in this tree before today folded a parent that the test
folding it had just synthesized — NEO native vision, the native audio, video and
points3d lanes.  The parent here was downloaded:

```
C:/Users/cereal/Documents/ai training/models/SmolVLM2-256M-Video-Instruct
  config.json         sha256 756cb7d37d7659c9d5a71c563f9d2530f9dd674ab7e909d8d0dd5b1a2d3e8317
  model.safetensors   sha256 dcc8f4c00d30d70bb3028a1bfc373e1b1972e25919f8788183cae4ae72431ce2
                      1,025,998,224 bytes, loaded as float32
  tokenizer.json      sha256 5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a
  preprocessor_config.json  sha256 149e315d9410368e5491455bb06e0f763426e9e56cca731c13b24404a29b6374
```

Its composition is verified field by field — `model_type`, `architectures`,
`scale_factor`, `image_token_id`, `vocab_size`, `tie_word_embeddings`, nine
`text_config` fields and four `vision_config` fields — before any compute
happens, and the weight digest is sealed into the run identity.

**What this run is.**  A SigLIP-shaped `smolvlm_vision` tower and a
pixel-shuffle connector are carried onto a blank child; the llama text decoder
is folded across a residual width contraction **H576 -> H432** with the same
sealed per-boundary chart set and the same two promoted causal branch kernels
`reports/cross-family-real-20260804-pretrained` used; and the folded child —
a working `SmolVLMForConditionalGeneration` — is then handed real RGB images and
asked to continue.

**What this run is not.**  It is a **bounded three-layer prefix** of the
checkpoint's thirty decoder layers, and a layer prefix of a VLM is not a VLM.
The decoded continuations below say that far more loudly than any caveat can:
the three-layer parent prefix emits `,   and,   and,` forever, and the child
folded from it emits scattered rare tokens.  The **full-depth unfolded parent**
is decoded on the same prompts and printed beside them so the size of that gap
is on the page rather than left to the reader.

---

## Layout

```
reports/smolvlm-real-fold-20260804/
  README.md                  this file
  L3-prefix/                 the run
    configuration.json       checkpoint identity, geometry, corpus, probe plan
    probe.json               the 16 deterministic probes and their digests
    probes/*.png             the probe images themselves, 64x64 RGB
    evidence.json            one decoder forward, mixed text + image rows
    charts.json              the 7 sealed residual boundary charts
    carry.json               vision tower copy + chart-projected connector
    fold.json                plan, executor run, per-branch numbers
    behavior.json            teacher-forced numbers + every decode, verbatim
    child/                   the folded multimodal child, loadable
    manifest.json            content digest of everything above
```

Verify:

```
python experiments/run_smolvlm_real_multimodal_fold.py \
    --report-root reports/smolvlm-real-fold-20260804/L3-prefix --verify-only
# {"verified": true, "files": 29, "problems": []}
```

The child is a real loadable multimodal model, not a state dict.  Reloaded from
`L3-prefix/child` with `AutoModelForImageTextToText.from_pretrained` it reports
`SmolVLMForConditionalGeneration`, `text_config.hidden_size 432`,
`num_hidden_layers 3`, inspects as `smolvlm-native-vision` over 100 nodes and
140,294,352 parameters, and takes an image-conditioned forward
(`pixel_values` + `pixel_attention_mask` + `input_ids`) whose argmax at the last
position is `'aughed'` — the first token of the `color_012` continuation
recorded below.

---

## Read this before the numbers

**1. The fold is planned on the decoder, and the vision half is carried.**
`DynamicFoldPlanner` only emits cross-width branch and chart-projected interface
actions when both graphs report `inspector == "hf-causal-decoder"`.  A `smolvlm`
composite graph does not, so the plan is over the two *decoder* graphs and the
composite graphs are sealed beside it as structural evidence
(`fold.json -> graphs`).  The tower is copied tensor for tensor; the connector
is projected through the same interface-locked chart as the embedding rows.
Neither is a fold action.  `NOTES-W9A.md` section 3.2 says exactly what a `src`
fix would have to add.

**2. A layer prefix of a pretrained model is not a pretrained language model.**
`--layers 3` keeps the first three of thirty decoder layers and then applies the
real `model.norm` and the real `lm_head` to whatever those three produce.  Every
weight is pretrained; the object as a whole was never trained to emit a
distribution from layer three.  These runs answer *does the child reproduce this
parent, on real structured anisotropic multimodal weights* — and they do **not**
answer *can a folded multimodal child see*.  The depth actually folded is sealed
in every receipt as `folded_layer_prefix` / `full_depth` / `bounded_layer_prefix`.

**3. Full depth does not fit this workstation's budget, and neither did six
layers.**  The evidence bundle is ~34.6 KB per position per layer; at 4608
positions that is ~4.8 GB for thirty layers, against 6.5 GB free when the wave
started, plus ~20 min of chart compile and ~60 min of execution — and a fold
that fails closed at any layer rolls the child back to blank, so the whole hour
and a half buys nothing.

A `--layers 6` arm was attempted twice and abandoned both times **on resources,
not on a fold refusal**: once inside the executor's post-acceptance evidence
re-read under memory pressure, once when the run's own committed memory drove
Windows pagefile expansion and took free disk on `C:` from 14.0 GB to 1.8 GB in
twelve minutes (killing the process returned it to 8.0 GB in five seconds).  Its
partial directory was deleted rather than committed half sealed.  So **the depth
wall for this checkpoint is not located by this report**; the prior text-only
run put SmolLM2-135M's at layer 3, and all three layers here folded.  Full
detail in `NOTES-W9A.md` section 4.1.

**4. The real `SmolVLMProcessor` cannot be constructed here.**  `transformers`
5.13.0 gates *both* SmolVLM image processors behind a torchvision backend, and
the PIL one does not use torchvision at all; `SmolVLMProcessor.__init__`
additionally needs `num2words` (video-only) and isinstance-checks the video
processor against a torchvision placeholder class.  The driver reproduces the
**image branch** of `SmolVLMProcessor.__call__` from its own components — the
real `SmolVLMImageProcessorPil` (`backend == "pil"`), the real
`transformers.models.smolvlm.processing_smolvlm.get_image_prompt_string`, and
the real tokenizer — and seals all of that in `evidence.json -> image_processor`.
Verbatim refusals are in `NOTES-W9A.md` section 2.

**5. Starved evidence is not a depth wall.**  A first bounded attempt at 6 text
prompts and 3 probes (~410 fit rows) failed closed at **layer 0**:

```
Cross-width causal attention branch operator endpoint relative RMSE
1.1904438462717335 is at or above the no-branch baseline bound 1.0
at boundary 'layer000.mlp_in'.
```

At 36 text prompts and 12 probes (2252 fit rows against 1152 target MLP
neurons, a 1.95x ratio matching the cross-family run's 2.06x) the same layer
folds at 0.545.  Every number below is from the second corpus.

---

## Configuration

| | parent | child |
|---|---|---|
| composite | `SmolVLMForConditionalGeneration` | `SmolVLMForConditionalGeneration` |
| vision tower | `smolvlm_vision`, H768, 12 layers, patch 16, image 512 | identical, copied |
| connector | `Linear(12288 -> 576)`, bias free | `Linear(12288 -> 432)`, chart projected |
| decoder family | llama | llama |
| decoder residual width | 576 | **432** (0.75x) |
| decoder layers folded | 3 (prefix of 30) | 3 |
| query / key-value heads | 9 / 3 | 6 / 3 |
| head_dim | 64 | 64 (must match: the planner compares `rotary_dim`) |
| MLP intermediate | 1536 | 1152 |
| per-head Q/K norm | none (identity) | none (identity) |
| vocabulary | 49,280, untied | 49,280, untied — **the same tokenizer** |
| composite parameters | 160,902,336 (3-layer prefix) | 140,294,352 |
| decoder parameters | 67,391,424 | 48,552,912 |

Because parent and child share one tokenizer, the token interface is the
**strict row-identity path**: `CrossWidthTokenHeadInterfaceOperator` from `src`
executes the planner's single `token_interface_bridge` action directly, and no
local composition of the cross-family pair-code interface is needed.

**Evidence.**  One decoder forward over 48 windows of 96 tokens:

| | rows | valid positions |
|---|---|---|
| text-only (`calibration_prompts_64.json`, indices 0..35) | 36 | 3,456 |
| image-conditioned (probes 0..11) | 12 | ~1,048 |
| **total** | **48** | **4,504** valid, 2,252 fit / 2,252 validation, 104 padded |

768 of those positions are `<image>` (id 49190) — 64 per image row — and they
carry the **real** vision tower and connector output, substituted by the model's
own `SmolVLMModel.inputs_merger`.  `parent_decoder_forward_calls: 1`,
`parent_vision_tower_forward_calls: 1`, `parent_image_merge_calls: 1`.

**Probes.**  Sixteen 64x64 RGB images built by integer-only deterministic PIL
drawing — no random source is consulted, so the same count always yields the
same images with the same digests.  Four categories x four instances:
`color` (which color is the square), `count` (how many circles),
`spatial` (red square relative to blue), `split` (top half's color).
Probes 0..11 are calibration; **12..15 are held out** and no fit ever saw them.

**Held out.**  Text prompts 36..43 at 64-token windows, and probes 12..15.
Neither appears in any fit.

---

## Mechanism: what the executor did

`L3-prefix/fold.json`:

* Plan **ready, zero gaps**, 8 actions: `token_interface_bridge` x1,
  `copy_exact` x1 (the parameter-free rotary global),
  `causal_gqa_rope_residual_attention_layer` x3,
  `causal_swiglu_residual_mlp_layer` x3.
  `partitions_every_learned_child_decoder_tensor: true`.
* **8 attempts, 8 accepted**, status `complete`; 30 changed state paths, all
  unique; `completed_every_learned_child_decoder_tensor: true`.
* `observed_parent_root_forward_calls: 0`,
  `observed_parent_module_forward_calls: 0`, and an independent
  `register_forward_hook` counter on the parent decoder also read **0** during
  execution.  The only parent decoder forward in the run is the single evidence
  capture.
* `optimizer_constructions: 0`, `optimizer_steps: 0`, `backward_calls: 0`,
  `autograd_grad_calls: 0`, `pruning_operations: 0`,
  `quantization_operations: 0`,
  `executed_fold_static_zero_training_audit_passed: true`,
  `operator_static_zero_training_audit_passed: true`.
* `child_recapture_calls: 6` — one live sequential recapture of the child per
  branch action — and `child_image_merge_calls: 6`, i.e. **every** child
  recapture was itself image-conditioned.

Timings (`seconds` in `fold.json`): vision tower 13 s, evidence capture 17 s,
chart compile 84 s, carry 0.6 s, execute 338 s.

### The carry

| | method | size | check |
|---|---|---|---|
| vision tower | `copy_exact` | 197 tensors, 86,433,024 elements | every tensor re-compared with `torch.equal` after the copy |
| connector | chart projection at `layer000.attention_in` | 5,308,416 elements | see below |

The connector writes into the residual stream at the same boundary the token
embedding writes, so the only projection consistent with the embedding rows the
interface operator installs is that same interface-locked chart applied to the
connector's output coordinates.  That is checked rather than asserted: after the
carry the **child's own** vision tower and connector are run on the probe crops
and compared against the chart encoding of the parent's image features.

```
child_image_feature_vs_chart_encoded_parent_relative_rmse = 2.0134e-07
```

Float32 round-off.  The child sees the same image, in the child's coordinates.

---

## Numbers

### Boundary charts

Seven boundaries, two of them interface locked
(`layer000.attention_in`, `output.final_norm_in`).

| | this run (text + image) | `cross-family-real-20260804-pretrained` L3 (text only) |
|---|---|---|
| boundaries | 7 | 7 |
| cycle defect, median | 0.1060 | 0.1354 |
| norm-commutation row fluctuation, median | 0.0119 | 0.00764 |
| fit retained energy, min / median | 0.9875 / 0.9884 | — |
| **held-out retained energy, min / median** | **0.9812 / 0.9833** | 0.9571 / 0.9716 |
| absorption-demand unabsorbed fraction, median | 0 | 0 |

**A mixed text+image residual is *more* compressible into 432 of 576
coordinates than a text-only one**, not less: held-out retained energy median
0.983 against 0.972, and the worst boundary 0.981 against 0.957.  The 64
consecutive connector-driven positions per image row do not fight the text rows
for the basis.

### Per-branch held-out endpoint error

The no-branch baseline is 1.0 by construction — a zero write predicts nothing —
so a value at or above 1.0 makes the operator refuse the branch.

| layer | branch | held-out endpoint relative RMSE | upstream error energy ratio |
|---|---|---|---|
| 0 | attention | **0.5452** | 2.30e-11 |
| 0 | gated MLP | 0.5883 | 8.80e-04 |
| 1 | attention | 0.7299 | 1.04e-02 |
| 1 | gated MLP | 0.8681 | 6.56e-02 |
| 2 | attention | 0.9465 | 1.14e-01 |
| 2 | gated MLP | 0.5822 | 1.03e-02 |

Attention: min 0.5452, median 0.7299, max 0.9465.
Gated MLP: min 0.5822, median 0.5883, max 0.8681.

For comparison, the text-only pretrained SmolLM2-135M run at the same depth and
width read 0.131 / 0.760 / 0.980 on attention.  This parent's first layer is
harder (0.545 against 0.131) and its third is not worse (0.947 against 0.980).
Two things differ at once — a different checkpoint and a mixed-modality
corpus — so this is a difference, not an attribution.

### Token interface

One `token_interface_bridge` action, `chart_projected_token_head_interface`,
`shared_tokenizer_identity: true`, installing
`model.embed_tokens.weight`, `lm_head.weight` and `model.norm.weight`
from the two interface-locked charts.
`final_norm_relative_rmse = 0.4627`.

### Whole-model teacher-forced agreement, held out

Parent and child read the same text — and, on the image lane, the same
image — at the same positions.  Both vocabularies are the same 49,280 rows, so
no restriction or alignment is involved.

| lane | rows | logit relative RMSE | centered logit relative RMSE | top-1 agreement | parent distinct top-1 | parent modal floor | margin |
|---|---|---|---|---|---|---|---|
| text-only | 512 | 0.6335 | 0.6994 | 25.39% | 42 | 33.40% (`Ġ`) | **-8.01 points** |
| **image-conditioned** | 359 | **0.3320** | 0.4304 | **50.42%** | 19 | 45.40% (`Ġ`) | **+5.02 points** |

The modal floor is what a child that emitted the parent's single most frequent
argmax token everywhere and nothing else would score; the margin is what the
fold actually bought over that floor.

**The folded child tracks its parent roughly twice as well on image-conditioned
positions as on text-only ones** — 0.332 against 0.634 logit relative RMSE —
and the image lane is the only one that clears its trivial floor.  The most
likely mechanism is mechanical rather than semantic: the connector is carried
through the *same* chart as the embedding at machine precision (2e-07), so image
positions enter the child's residual stream almost exactly where the chart says
they should, while text positions enter through 49,280 chart-projected embedding
rows and a fitted final-norm gain (0.4627 relative RMSE on that gain alone).
This is a statement about the interface, not about seeing.

---

## Decoded continuations — complete, verbatim

Greedy argmax, no sampling, no temperature, no beams; 24 new tokens per prompt;
the image crop supplied once on the prefill exactly as the composite's own
generation path supplies it.  Three decodes per prompt:

* **child** — the folded 3-layer H432 multimodal child;
* **parent (L3)** — the same 3-layer prefix the child was folded from, which is
  the only token-for-token comparable decode;
* **parent (L30)** — the untruncated checkpoint, decoded on the identical
  prompt, present only to show how far a 3-layer prefix is from the model.

Complete objects, including token ids and per-token surface forms, are in
`L3-prefix/behavior.json` under `text_continuations` and `image_continuations`.

Greedy prefix agreement between child and parent (L3): **0 of 24 tokens on
every one of the twelve held-out prompts**, in both lanes.

### Image-conditioned held-out prompts

The four probes below are `L3-prefix/probes/*.png`; each was drawn by integer
arithmetic and never seen by any fit.

#### probe `color_012` — a yellow square on white, "What color is the square? Answer with one word."
```
child      :aughed affili ( the Southeasternemeteriesrokes adul D,aughed affiliash outbre ( the Southeastern so,aughed affiliberspace
parent (L3):   and,   and,   and,   and,   and,   and,
parent (L30): Yellow<end_of_utterance>
Assistant: Yellow<end_of_utterance>
Assistant: Yellow<end_of_utterance>
Assistant: Yellow<end_of_utterance>
Assistant:
```

#### probe `count_013` — four red circles, "How many red circles are shown? Answer with one digit."
```
child      :akespe affili (  thisolly,thelesson:naments  the    what, affilierm, eankar and
parent (L3):   and,   and,   and,   and,   and,   and,
parent (L30): 4<end_of_utterance>
Assistant: 4<end_of_utterance>
Assistant: 4<end_of_utterance>
Assistant: 4<end_of_utterance>
```

#### probe `spatial_014` — red square below blue square, "Where is the red square relative to the blue square? Answer left, right, above, or below."
```
child      :aughed diastembourgs,elivery and can't strugg.  " the        what, affili
parent (L3):   and,   and,   and,   and,   and,   and,
parent (L30): The red square is above the blue square.<end_of_utterance>
Assistant: The red square is above the blue square.<end_of_utterance>
```

The probe places red at y=48 and blue at y=16 with y increasing downward, so the
expected answer is `below` and the **full-depth parent is wrong** here.  That is
the checkpoint's behaviour on a 64x64 synthetic upscaled to 512x512, reported as
measured; nothing in this run tuned it and nothing here is entitled to hide it.

#### probe `split_015` — yellow over purple, "What color is the top half of the image? Answer with one word."
```
child      : Do, affiliizabeth Head or other type s Aviv (  the    iriam or other than just/__,
parent (L3):   and,   and,   and,   and,   and,   and,
parent (L30): Yellow<end_of_utterance>
Assistant: Yellow<end_of_utterance>
Assistant: Yellow<end_of_utterance>
Assistant: Yellow<end_of_utterance>
Assistant:
```

### Text-only held-out prompts

#### held-out text prompt 0
```
prompt tail : ...authors had documented the combination in an appendix, so nothing had been concealed. What
child       : kind        umb recapanation - (    what,lez, the
parent (L3) :,   and,   and,   and,   and,   and,   and
parent (L30): was the effect of the combination? The effect was that the two categories were now separated by a margin of 10
```

#### held-out text prompt 1
```
prompt tail : ...reas individual attention is already at its practical limit. Teams that get this right end
child       :,          iriam ( iriam ( iriam ( iriam ( iriam
parent (L3) :   and,   and,  and,   and,   and,   and,
parent (L30): up with a better understanding of the situation and a better response to the situation.

The postmortem is a
```

#### held-out text prompt 2
```
prompt tail : ...th impeccable papers. Players tend to care about this less than dealers do. The instrument
child       : pathlibs to the,         (      ( iriam (
parent (L3) :.   and,   and,   and,   and,   and,   and
parent (L30): is a rare and valuable piece of music, and it's a testament to the skill of the maker.

The
```

#### held-out text prompt 3
```
prompt tail : ..., which sounds small until it is integrated over a summer afternoon. Mounting arrangements
child       :,ueblracuse (        (    what,ippings dolph anduebl necessit comprom.
parent (L3) :.   and,   and,   and,   and,   and,   and
parent (L30): are also quite small, and the solar panels are not very tall. The solar panels are not very tall, but they
```

#### held-out text prompt 4
```
prompt tail : ...ded. The safest approach is to characterise the existing behaviour with tests written from
child       :osaic        what  what  what,lez, I G,angar Aztes I<0xEF>
parent (L3) :   and,   and,   and,   and,   and,   and,
parent (L30): scratch, and then write the replacement in a way that is consistent with the existing code.

In the case of
```

#### held-out text prompt 5
```
prompt tail : ...the midpoint. The last resident left in a year nobody recorded precisely. What remains is
child       : the same,                what,rovers A,
parent (L3) : a.   and,   and,   and,   and,   and,
parent (L30): a small village, with a few houses, a church, a school, and a few shops. The village is now
```

#### held-out text prompt 6
```
prompt tail : ...s is the reason that agricultural and clinical use cannot be treated as separate problems.
child       : the       then ortiumm I theories a, iriam ( iriam ( iriam
parent (L3) :   and,   and,  and,   and,   and,  and,
parent (L30): The resistance cassette can be selected for in the laboratory, and the resistance cassette can be selected for in the environment
```

#### held-out text prompt 7
```
prompt tail : ...n fragments. Everything else in her job could tolerate interruption; the writing could not
child       : arynge,xietyenfranch,      iriam (   , I Gesticposiumre,
parent (L3) :.   and,   and,   and,   and,   and,   and
parent (L30):.
She was a writer, but she was not a writer. She was a writer, but she was not a
```

---

## What these decodes are and are not evidence of

**They are the first decoded output ever taken from a folded child of a real
pretrained multimodal checkpoint, on real image-conditioned prompts.**  The
image crops went through the real image processor, the real vision tower and the
real connector; the connector was carried into the child's coordinates at 2e-07;
the child is a loadable `SmolVLMForConditionalGeneration` that runs
`pixel_values` end to end.  The pipeline is real from PNG to token.

**They are not evidence that a folded child can see.**  Neither the child nor
the parent it was folded from can write a sentence at three layers, so
"reproduces its parent's continuation" is a claim about a degenerate target.
The one place a signal survives at this depth is teacher forcing, where the
image lane clears its modal floor by 5.02 points and the text lane misses its
own by 8.01 — and the honest reading of that is the interface arithmetic quoted
above, not visual competence.

Getting behaviour out of this requires depth, and depth requires either more RAM
than this workstation has free or an evidence bundle that does not hold thirteen
per-token tensors per layer at once.  Both are named in `NOTES-W9A.md`.
