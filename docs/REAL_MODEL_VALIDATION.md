# Real Model Validation

The toy dense tests prove tensor surgery invariants. They do not prove that a
real checkpoint still behaves. This validation path runs small public
checkpoints through behavior probes before and after surgery.

## Default Run

```bash
python experiments/run_real_model_validation.py \
  --run-dir runs/real-model-validation \
  --device cuda:0 \
  --dtype float16
```

Defaults:

- text base: `Qwen/Qwen3-0.6B-Base`
- text tuned parent: `Qwen/Qwen3-0.6B`
- VLM sanity parent: `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`

The run writes:

- `reports/text_base.json`
- `reports/text_tuned.json`
- `reports/text_merged_alpha_050.json`
- `reports/text_merged_ffn_099.json` by default, unless `--shrink-ratio` is changed
- `reports/vision_probe.json`
- `reports/vision_chain.json` when the optional composite chain is run
- `reports/summary.json`

## What Counts As Evidence

Minimum useful evidence:

- parent models load and generate;
- merged model loads and generates;
- FFN-resized model loads and generates;
- perplexity and choice probes do not collapse to nonsense;
- `hf-gate` passes for any candidate being called safe;
- VLM probe produces grounded text for synthetic images.

This is not a claim that a heavily frankensteined model is production-good
without recovery training. It is a behavioral smoke test that distinguishes
actual model survival from merely making tensor dimensions line up.

## Current Safe Scope

Supported real-checkpoint surgery:

- same-shape Hugging Face causal LM checkpoint averaging;
- task-arithmetic Hugging Face merges with alpha sweeps;
- Qwen/Gemma/Llama-style gated FFN intermediate resize by folding or copying
  channels;
- calibration-aware gated FFN repair using ridge least squares on parent FFN
  inputs/outputs;
- lightweight causal LM behavior probes;
- report gates and FFN resize sweeps that reject unsafe candidates;
- small VLM image-description/OCR sanity probes.
- transitional VLM-observation -> merged-text-model chains for grounded behavior
  checks before native multimodal training exists.
- native early-fusion sandbox training via `native-vlm-demo`, using raw image
  patches plus text tokens in one transformer.
- merged-parent raw-patch vision graft training via `hf-vision-graft`, using an
  existing HF causal LM parent and no external vision encoder.

Not yet supported:

- direct cross-architecture merge of Qwen + Gemma + LFM into one valid decoder;
- untrained native vision grafting that magically understands pixels;
- aggressive compression without calibration/recovery training.

The safe next step is improving the recovery path: sequential activation repair
after FFN folding, larger real multimodal datasets for the graft path, and
behavior gates that score both text retention and grounded multimodal answers.
