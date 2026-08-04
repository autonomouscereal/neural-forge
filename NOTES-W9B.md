# NOTES-W9B — ACE-Step 5Hz audio-token LM readiness

Report: `reports/acestep-lm-readiness-20260804/README.md`
Model: `C:/Users/cereal/Documents/ai training/models/acestep-5Hz-lm-1.7B`

## What this proved

The ACE-Step audio-token LM is foldable by the machinery we already have. It
needed no new inspector arm and no weakened guard — only an honest causal-LM
wrapper, because the checkpoint publishes a **bare `Qwen3Model` body**.

## The one thing that mattered

The checkpoint has no `lm_head` tensor and no `model.` key prefix. `Qwen3Model`
is not a causal LM, so `hf_arch_layout` refuses it — correctly. Since
`tie_word_embeddings` is true, the head is fully determined: `logits = h @ E^T`.
Wrapping the body in a `Qwen3ForCausalLM` whose `lm_head.weight` **is** the
body's `embed_tokens.weight` makes the existing inspector and the existing
one-forward evidence path work unchanged, and the equality is bitwise
(`torch.equal` → True), not approximate.

Added `wrap_bare_tied_causal_body()` to `dynamic_model_inspector.py` as a
clearly separated additive block. It is a **wrapper adapter, not an inspector
arm**: no registry change, no guard change. It refuses an untied body rather
than inventing head weights, and builds its shell on the meta device so
wrapping 1.85B parameters copies nothing.

Tests: `tests/test_dynamic_bare_tied_causal_body.py` (10 cases, own file to
avoid merge skew with the concurrent SmolVLM inspector work).

## Correction to the task brief

The brief said "64,000-code 5Hz audio codebook". It is **65,535 codes**, and it
starts at row **151,669**, not `217,204 − 64,000 = 153,204`:

```
text + specials  0 … 151,668   (151,669 rows, incl. 26 added specials)
audio codes      151,669 … 217,203   (65,535 rows)  code = row - 151669
151,669 + 65,535 = 217,204 = vocab_size   (exact)
```

Anything that keys off the audio boundary must use **151,669**.

Also: HEAD was `60eb928b`, not the `59f94a83` in the brief (an ancestor).

## Fetch: do not use SFTP for multi-GB files here

`sftp.get` on the 3.7GB file fails closed with a bogus `[Errno 2] No such file`
(paramiko unbounded prefetch). Bounding the prefetch to 64 requests fixes the
error but drops throughput to **0.17 MB/s**. Streaming over one `dd` exec
channel ran at **37.4 MB/s**. `scratchpad/stream_safetensors.py` has the recipe
(resumable, chunk-aligned, sha-verified). Also note local disk was only 15.5 GB
free, so the tar→extract round trip was skipped in favour of per-file transfer
with per-file digests.

## Lane status

| lane | verdict |
|---|---|
| GQA contraction 16q/8kv → 8q/4kv | applies (`gqa_group` contract True on all 28 layers) |
| Cross-width H2048 → H1024 | applies but **chart-gated**; no chart set exists yet |
| Pair/strict vocab bridge over 217,204 rows | applies as-is; audio rows are ordinary donor rows |

Layer 0 emits `causal_gqa_rope_residual_attention_layer` and
`causal_swiglu_residual_mlp_layer`. That is the planner emitting operators — **no
numeric fold was executed and nothing was scored.**

## Next

1. Build an authenticated chart set for this parent to unblock cross-width.
2. Re-capture evidence in fp32 on a host with real RAM headroom; this box had
   ~4 GB commit free, so everything here is bf16.
3. A real tokenizer-semantics fingerprint (the graph currently uses the
   safetensors sha256 as a stand-in).
