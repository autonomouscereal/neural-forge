# ACE-Step 5Hz audio-token LM — fold readiness audit

**Date:** 2026-08-04 · **Branch:** `codex/lrc-blank-compression` · **HEAD at audit:**
`60eb928b` (the brief named `59f94a83`, which is an ancestor of this HEAD)
**Runtime:** local CPU, `neural-forge/.venv` (torch 2.13.0+cpu, transformers 5.13.0)
**Model:** `C:/Users/cereal/Documents/ai training/models/acestep-5Hz-lm-1.7B`

Verdict: **the audio modality's real parent is foldable by the existing machinery.**
It needed no new inspector arm and no relaxed guard — only an honest causal-LM
wrapper around a checkpoint that ships as a bare decoder body.

---

## 1. Fetch verification

| item | value |
|---|---|
| source | `ai:/home/cereal/comfyui/models/ACE-Step/acestep-5Hz-lm-1.7B` |
| transport | server-manager v2 client (`ssh_client.py --server ai`), never raw `ssh` |
| files | 9 of 9, each sha256-verified against a digest taken on the server *before* transfer |
| `model.safetensors` | 3,708,521,528 B, sha256 `f161689da73e5ecefa28ff780d51c2d92a00f056d021d7933c779ed5c6cd7db8` — **matches the expected digest exactly** |
| server scratch | `/tmp/acestep-fetch-20260804` removed; source dir untouched |

**Deviation from the brief, and why.** The brief specified tar → download → extract.
I built the dereferenced tar on the server and hashed it, then abandoned that route:

- Local free disk was **15.5 GB**. Tar + extract peaks at ~7.5 GB; direct per-file
  transfer peaks at 3.75 GB.
- `sftp.get` on the 3.7 GB tar failed closed with a spurious `[Errno 2] No such file`
  (paramiko's unbounded prefetch). Bounding the prefetch window to 64 requests fixed
  the error but collapsed throughput to **0.17 MB/s** (ETA 284 min).
- Streaming the file over a single `dd` exec channel ran at **37.4 MB/s** (77.8 s).

Per-file hashing is strictly stronger verification than one tar digest, so integrity
was not traded away. Fetch scripts:
`scratchpad/fetch_acestep.py`, `scratchpad/stream_safetensors.py`.

---

## 2. Architecture facts (`architecture.json`)

Confirmed against `config.json` **and** the safetensors header (310 tensors):

- `model_type: qwen3`, `architectures: ["Qwen3Model"]` — a **bare decoder body**.
- Checkpoint top-level prefixes are `embed_tokens`, `layers`, `norm`. There is **no
  `model.` prefix and no `lm_head` key**. `get_output_embeddings()` returns `None`.
- H2048 · 28 layers · 16 query heads / 8 KV heads · head_dim 128 · intermediate 6144.
- `tie_word_embeddings: true`, bf16 storage, rope_theta 1e6, `sliding_window: null`,
  all 28 `layer_types` are `full_attention`.
- Unique parameters: **1,854,243,840**.

### How the runtime gets logits

Tied embeddings mean the head *is* the input embedding: `logits = h @ E^T`. The
honest wrapper is a `Qwen3ForCausalLM` whose `lm_head.weight` is the same tensor
object as `embed_tokens.weight`. Verified on the real weights:

```
torch.equal(hidden @ embed_tokens.weight.T, wrapper(input_ids).logits)  ->  True
max abs delta = 0.0            # bitwise, same dtype, not "close enough"
lm_head.weight is embed_tokens.weight -> True   (same data_ptr, no copy)
```

The wrapper shell is built on the meta device, so wrapping a 1.85B-parameter body
allocates **no second copy** of the weights.

---

## 3. Inspector outcome (`inspector.json`)

| target | inspector selected | outcome |
|---|---|---|
| bare `Qwen3Model` | `hf-causal-decoder` | **fails closed (correct)** |
| tied causal-LM wrapper | `hf-causal-decoder` | **inspected cleanly** |

The bare body is refused by `hf_arch_layout`:

> `Unsupported qwen3 decoder layout: model must be exactly
> transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM, got
> transformers.models.qwen3.modeling_qwen3.Qwen3Model.`

**This guard is correct and was left intact.** A headless body is not a causal LM;
teaching the inspector to accept one would let any bare body masquerade as a
foldable parent. So no inspector arm was added.

The wrapper compiles to a complete graph:

- family `qwen3`, 28 layers, **172 nodes / 282 edges**
- logical parameters 2,299,077,632 · unique 1,854,243,840
- difference **444,833,792 = 217,204 × 2048**, exactly one embedding matrix — the
  tie is *accounted*, not hidden
- covers every parameter path; `lm_head.weight` and `model.embed_tokens.weight`
  share one storage group
- interfaces: `text.tokens.input`, `text.logits.output`
- graph fingerprint `e93ec9a669763d2efa0695a49738da5bc648efc3cacb2d4a611eed0232a82c28`

### What was added to `src`

One additive block in `dynamic_model_inspector.py`: `wrap_bare_tied_causal_body()`.
It is a **wrapper adapter, not an inspector arm** — the registry, every existing
inspector, and every layout guard are untouched. It fails closed on: missing config,
unsupported `model_type`, a body that is already a causal LM, `tie_word_embeddings`
not exactly `True` (an untied body has no recoverable head — refuse rather than
invent one), a pre-existing `lm_head` or output embeddings, a missing or
wrong-shaped `embed_tokens.weight`, and any parameter left on the meta device.

---

## 4. Audio-token boundary map (`vocab-boundary.json`)

**The brief's "64,000-code codebook" is wrong.** Measured from `added_tokens.json`:

| region | rows | count |
|---|---|---|
| text + base specials | `0 … 151,642` | 151,643 |
| added text specials (`<\|endoftext\|>`, `<\|im_start\|>`, `<think>`, …) | `151,643 … 151,668` | 26 |
| **audio codebook** `<\|audio_code_N\|>` | **`151,669 … 217,203`** | **65,535** |

- Boundary row: **151,669**. `id < 151669` is text; `id >= 151669` is an audio code.
- `code = row - 151669`, dense and ordered: `<|audio_code_0|>` → 151,669,
  `<|audio_code_65534|>` → 217,203.
- Partition is exact: `151,669 + 65,535 = 217,204 = vocab_size`.

So the codebook is 65,535 codes (2^16 − 1), not 64,000, and it starts at 151,669,
not at `vocab_size − 64,000 = 153,204`.

---

## 5. Evidence capture outcome (`evidence.json`)

`capture_hf_causal_circuit_evidence` on the tied wrapper, over **deterministic
mixed text/audio sequences** (no RNG — fixed modular arithmetic):

- row 0: text and audio alternating every token
- row 1: 8 text tokens then 16 contiguous audio codes (the shape ACE-Step actually sees)
- batch 2 × sequence 24 = **28 audio tokens and 20 text tokens**

Result: **SEALED.**

| field | value |
|---|---|
| cache key | `ea2ad3ec54c2e9013e5e795ef8c588e3bdb9f29894fb4308d1d41409d01655e0` |
| parent forward calls | **1** |
| layers captured | 28 |
| tensor manifest entries | 430 |
| logits | `[2, 24, 217204]` |
| fit/validation disjoint | True |
| `verify()` after sealing | ok |
| tamper check | mutating one logit **is rejected** by `verify()` |

The seal was proven live, not assumed: perturbing `logits[0,0,0]` after sealing made
`verify()` raise. Audio-codebook rows flow through the same embedding and the same
tied head as text rows — no audio-specific path is involved.

---

## 6. Lane applicability matrix (`lane-matrix.json`)

Every row below was **executed** against the graph compiled from the real
checkpoint, using the planner's own contract helpers — not argued from prose.

| lane | predicate run | result |
|---|---|---|
| **GQA contraction** 16q/8kv → 8q/4kv | `_coupled_role_contract("gqa_group", …)` on all 28 real attention nodes | **APPLIES** — `True` |
| **Cross-width chart** H2048 → H1024 | `_cross_width_boundary_chart_crossing(parent, child)` against a real contracted child (H1024, 8q/4kv, 28L) | **APPLIES, chart-gated** — `True` |
| **Pair/strict vocab bridge** over 217,204 rows | `_token_interface_tying_contract` + `IdentityVocabularyProjector.fold_consumer_weight` on real donor rows incl. audio rows 151,669 / 151,670 / 184,437 / 217,203 | **APPLIES AS-IS** — transport exact |

Enabling facts, all read off the real graph: `inspector = hf-causal-decoder`,
`position_policy = rope`, `hidden_size = 2048`, `sliding_window = None`, and all 28
layers uniform in `(query_heads, key_value_heads, layer_type)`.

### Branch methods emitted for one layer

`_family_neutral_causal_layer_contract` on **layer 0**, parent → contracted child,
emits both ordered causal branches:

| branch | method |
|---|---|
| attention | `causal_gqa_rope_residual_attention_layer` |
| MLP | `causal_swiglu_residual_mlp_layer` |

Source and target node triples are
`attention_norm → attention → attention_residual` and
`mlp_norm → mlp → mlp_residual` on both sides.

**Scope honesty:** this is the planner *emitting* the branch operators for one real
layer. It is **not** a numerically executed fold. Cross-width operators additionally
require an authenticated per-boundary chart set and fail closed without one; no
chart set was built, and no fold was executed or scored. Width *expansion* is
outside the v1 envelope — the balanced basis is a contraction construction.

---

## 7. Blockers and honest gaps

1. **No numeric fold was executed.** Lane applicability is proven at the
   planner/contract level plus one executed vocabulary-transport primitive. An
   end-to-end branch fold needs a sealed chart set and the fold executor.
2. **Cross-width is chart-gated.** H2048 → H1024 is admissible but inert until an
   authenticated chart set exists for this parent.
3. **bf16 throughout.** The workstation had ~3.9 GB available RAM and only ~4.2 GB
   of commit headroom against a 34 GB machine, so fp32 (≈7.4 GB for 1.85B params)
   was not affordable. The model was loaded mmap-backed in its native bf16, which is
   also its storage dtype. Captured evidence is therefore bf16; fold fitting that
   wants fp32 residuals will need to re-capture on a larger host.
4. **The wrapper is a runtime reconstruction, not a shipped artifact.** It is exact
   (bitwise) and allocation-free, but ACE-Step's own inference stack was not
   consulted — the tie is inferred from `tie_word_embeddings: true` plus the absence
   of any head tensor, which admits exactly one honest reading.
5. **Tokenizer fingerprint is a stand-in.** The graph was fingerprinted with the
   safetensors sha256 rather than a real tokenizer-semantics fingerprint.
6. **Untouched by design:** r12 (PID 3946726), every other report directory, and all
   pre-existing inspector arms and guards.

---

## Artifacts

| file | contents |
|---|---|
| `architecture.json` | config + safetensors-header facts, tied-wrapper equivalence proof |
| `inspector.json` | bare-body refusal (with cause) and tied-wrapper graph accounting |
| `evidence.json` | sealed one-forward evidence over mixed text/audio tokens, tamper check |
| `vocab-boundary.json` | exact text/audio row partition |
| `lane-matrix.json` | executed lane predicates and layer-0 branch method emission |
