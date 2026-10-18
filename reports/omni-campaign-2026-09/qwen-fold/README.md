# qwen-fold — the Wave A fold harness for `qwen3_5`

Lane root (workstation): `reports/omni-campaign-2026-09/qwen-fold/`
Lane root (server): `/home/cereal/omni-campaign/qwen-fold/`
Runtime: `/home/cereal/model-surgery-lab/venv/bin/python` (transformers 5.12.1, torch 2.11.0+cu128)

Every fold class here is a **single transaction** that produces a **standard
`qwen3_5` HF checkpoint** plus a tensor-shape receipt. Nothing in this lane
grades a child: the receipt's numbers are diagnostics and proposed decision
lines for Cyril, and the only grade is the frozen behavioural review run by the
review lane.

## Modules

| File | Class | What it does |
|---|---|---|
| `qf_arch.py` | — | The only module that knows qwen3_5 names: loaders, layer kinds, the two RMSNorm gain conventions (`1+w` vs `w`), plane layout, closed-form parameter count, config surgery, the mask-dispatch assertion. |
| `qf_corpus.py` | — | Sealed evidence manifest (`neural-forge-omni-evidence-v1`), re-sealed on read; fit/held split; internal check split. Manifests are written through a temporary and `os.replace`d into place, so a killed write can never leave a truncated seal where the evidence was. |
| `qf_evidence_build.py` | — | **Builds** the manifest: real Qwen ids out of the curated master shard registry (`chimera_omni.data.curated_shards`) over text/code/math/tool/reasoning, contamination-screened against the frozen review surface at 48-char containment, packed **row-atomically** into `--batches` x `--sequence-length` (a row that does not fit waits for the next batch and the remainder is eos-padded, so the fit/held split is a DOCUMENT split), sealed by `qf_corpus.write_manifest` and receipted. Refuses to overwrite an existing manifest without `--force`, and `--force` backs the old one up to `evidence/.superseded-<UTC>/` first. |
| `qf_evidence_verify.py` | — | Verifies a BUILT manifest: seal, shapes, splits, check-row arithmetic, id range, tamper refusal, ragged refusal, the tokenizer gate both ways, that no batch begins mid-row (read off the manifest, not off the builder's report), and that the build receipt is about that file. |
| `qf_harness.py` | — | Model + hooks. Captures a module's input, the **pristine** module's output on that same input (the conditioned-local target), the live output, and Gram-only accumulation for the 27B. `harness_parity` proves determinism, hook placement and logit exactness. |
| `qf_solve.py` | — | Closed-form algebra only. Ridge under THE LAW (row and Gram forms), pivoted coresets, whitened low-rank contraction, partial-RoPE plane algebra, the per-plane joint relation solve, the gated writer solve, CG re-fit of a shared value projection. |
| `qf_vocab.py` | **C1** | Vocabulary coreset: row deletion, merge-closure-preserving keep set, remapped tokenizer + sidecars, hard-id remap. The byte alphabet is named from the canonical ByteLevel `bytes_to_unicode` table, **not** by scanning for one-character tokens (a real Qwen table has thousands of those that are not byte tokens). |
| `qf_mlp.py` | **C2** | MLP channel coreset: verbatim gate/up rows, ridge-refit `down_proj`, `--mlp-group` sites per capture, one global `intermediate_size`. One corpus pass per group, shared by its sites; the schedule that ran is in the axis report (`schedule.corpus_passes`). |
| `qf_excise.py` | **C3/C4** | Layer excision with `layer_types`/`num_hidden_layers` re-derivation, `layer_idx` re-indexing, and the above-diagonal attention-mass assertion. |
| `qf_gdn.py` | **C5** | GDN value-head fold — head-count arm and head-dimension arm. The merged head's **beta** row is fitted, not averaged: the delta rule consumes `sigmoid(b·x)` (modeling 466/516), so an energy-weighted mean of the parents' LOGIT rows is Jensen-biased (3.70x understatement on the suite's negative control). `_refit_beta` fits against `sum_j w_j sigmoid(b_j·x)` through the logit and installs only where it beats that row on the evidence — the same guard shape as the decay row. Both refits are ridges over the same evidence rows, so the layer forms `X^T X` and `[X|1]^T [X|1]` **once** (`_refit_grams`) and every child head reuses them through `prior_anchored_gram`; only the `O(N h)` right-hand sides stay in the head loop. |
| `qf_attn.py` | **C6/C7** | NoPE-dimension contraction; kv-head merge (kv-only and full-group arms). C6 moves `head_dim`, which moves the softmax temperature (`scaling = head_dim ** -0.5`), so `sqrt(child_head_dim/parent_head_dim)` is folded into the `q_norm` gain — exact, closed-form, and checked by `attention_gate_probe` + `logit_scale_ratio` in the suite. |
| `qf_termgain.py` | **C9** | Terminal gain re-solve, Hadamard normal equations, exact identity gate. |
| `qf_export.py` | — | Sharded export, per-tensor and directory digests, MTP carry-through, stock re-load verification, the `convert_hf_to_gguf` hook and the frozen-review runner hook. |
| `qf_build.py` | — | The CLI. Writes `qf-build-receipt.json`. Refuses evidence whose `tokenizer_sha256` is not the parent's own tokenizer.json (`--allow-tokenizer-mismatch` records the exception): the ids of one Qwen manifest are legal for any parent with a wide enough table, so without this check a 27B build on 4B evidence completes silently with every ridge fitted on a stream the parent decodes as different text. Structural axes **may** be combined in one call (each is recorded separately); `--vocab-keep` **may not** be combined with anything and that is enforced, because a vocabulary fold renames every row and the sealed stream is in the parent's id space. A vocabulary child writes `evidence-remapped.json` beside itself for the next build in the chain. `--carry-mtp` is refused when any axis moved a shape the MTP block reads. |
| `qf_selftest.py` | — | 21 self-tests on a tiny random-init `Qwen3_5ForCausalLM`, including the evidence builder's row atomicity (with the flat-stream packer as a negative control), its contamination screen and per-pack liveness, the manifest overwrite/atomicity guards, and the consumption-time tokenizer gate. |
| `qf_evidence.py` | — | Prints the lane's evidence block **from** `selftest-results.json`. No number or digest in a claim about this lane may be one this script cannot print. |

## Commands

```bash
# self-tests (CPU, ~1 minute)
cd /home/cereal/omni-campaign/qwen-fold
/home/cereal/model-surgery-lab/venv/bin/python qf_selftest.py \
    --work /home/cereal/omni-campaign/qwen-fold/.work \
    --json-out /home/cereal/omni-campaign/qwen-fold/selftest-results.json

# the evidence block, generated FROM that artifact instead of written by hand
/home/cereal/model-surgery-lab/venv/bin/python qf_evidence.py \
    --results /home/cereal/omni-campaign/qwen-fold/selftest-results.json

# build the sealed evidence manifest (bounded share reads; ~2 min)
/home/cereal/model-surgery-lab/venv/bin/python qf_evidence_build.py \
    --out /home/cereal/omni-campaign/qwen-fold/evidence/omni-evidence-v1.json
#   ... add --force to supersede an existing manifest (backed up first)

# verify the manifest that Wave A is gated on
/home/cereal/model-surgery-lab/venv/bin/python qf_evidence_verify.py \
    --manifest /home/cereal/omni-campaign/qwen-fold/evidence/omni-evidence-v1.json \
    --parent /home/cereal/models/Qwen3.5-4B \
    --other-parent /home/cereal/models/Qwen3.8-27B-Uncensored

# one child
/home/cereal/model-surgery-lab/venv/bin/python qf_build.py \
    --parent /home/cereal/models/Qwen3.5-4B \
    --out    /home/cereal/omni-campaign/qwen-fold/children/QF-C2K2304 \
    --evidence /home/cereal/omni-campaign/qwen-fold/evidence/omni-evidence-v1.json \
    --tag C2K2304 --parent-code Q35-4B --device cuda:0 --dtype bfloat16 \
    --mlp-keep 2304

# via the queue (adds the GPU preflight and the reproduction check)
./bodyq.sh C2K2304 --mlp-keep 2304

# a C1 child and its chain: the vocabulary axis is EXCLUSIVE, so the second
# fold is a second build against the first child and the manifest it wrote
/home/cereal/model-surgery-lab/venv/bin/python qf_build.py --parent <PARENT> \
    --out children/QF-V4X --evidence evidence/omni-evidence-v1.json \
    --tag V4X --vocab-keep 62080
/home/cereal/model-surgery-lab/venv/bin/python qf_build.py \
    --parent children/QF-V4X --evidence children/QF-V4X/evidence-remapped.json \
    --out children/QF-V4X-C2 --tag V4XC2 --mlp-keep 2304
```

## Rung table for the 4B and the 27B

| Class | Flag | 4B rung | 27B rung | Component fold |
|---|---|---|---|---|
| C1 | `--vocab-keep` | 248,320 → 62,080 | default OFF at 27B (D11) | 4x |
| C2 | `--mlp-keep` | 9,216 → 2,304 / 1,152 | 17,408 → 4,352 / 2,176 | 4x / 8x |
| C3 | `--excise <full layer>` | one of 8 | one of 16 | −12.5% / −6.25% KV |
| C4 | `--excise <gdn layer>` | one of 24 | one of 48 | — |
| C5-A | `--gdn-value-heads/--gdn-key-heads` | 32/16 → 16/8 → 8/4 | 48/16 → 24/8 → 12/4 | 2x / 4x |
| C5-B | `--gdn-*-head-dim` | 128 → 32 | 128 → 32 | 4x |
| C6 | `--nope-keep` | 192 → 48 (head_dim 256 → 112) | same | 4x on NoPE |
| C7 | `--kv-heads [--kv-arm]` | 4 → 2 → 1 | 4 → 3 → 2 → 1 | KV cache /2 |
| C9 | `--terminal-gain` | — | — | repair, 0 parameters |

Every number above is a rung to build, not a threshold to pass.

## Known gaps

* C6's query frame is shared across a kv group. That is not a concession: the
  group's total score error is `sum_h ||Cq_h^{1/2}(M-I)Ck^{1/2}||_F^2` for one
  shared `M`, which is the same objective written in the SUMMED metric, and for
  a fixed `Pk` the per-head optimum `(Pk Ck Pk^T)^-1 Pk Ck` is head-independent.
  The axis solves that optimum separately and records the gap against what it
  installed (`1.1e-16` on the fixture) plus the per-head cost spread
  (`own_metric_relerror_installed` 0.682–0.723 over 8 heads), so a reader can
  check the claim instead of trusting it.
* `_refit_grams` forms two Grams per layer, not one: `X^T X` is the top-left
  block of `[X|1]^T [X|1]` in exact arithmetic but not bit-for-bit (the wider
  GEMM blocks differently — measured `4.547473508864641e-13` max abs difference
  at `N=768, h=128`, `0.0` at three other shapes), and slicing would have moved
  a shipped number. `test_c5_refit_gram_hoist` asserts the hoisted form returns
  the beta row, the decay row and the decay bias at max abs difference `0.0`
  from the row form it replaced, and that a whole-model fold makes 6 Gram
  formations for 6 layers where the pre-fix code made 24 for 12 child heads.
* The evidence manifest is 4,096-token rows only. No registry row is anywhere
  near 32k tokens (longest ~12.6k characters), so the plan's long-context
  evidence corpus still has to be BUILT; `--sequence-length` is a flag with no
  source of long rows behind it.
* `omni-evidence-v1.json` is evidence for the **Qwen3.5** parents only. The
  0.8B and 4B tokenizer.json files are byte-identical; the
  `Qwen3.8-27B-Uncensored` on this box is a different file
  (`06b95093…` vs `5f9e4d49…`), so the 27B lane needs its own manifest. The
  builder records this per parent (`tokenizer.sibling_parents`) and `qf_build.py`
  now refuses the mismatch at consumption time.
* The tokenizer defines 248,070 rows (248,044 base BPE + 26 added) inside
  248,320 physical = **250** dead rows, not the 276 PORT_SPEC 0.3 and the R5
  digest quote. 276 is 250 + those 26 added tokens, i.e. what
  `range(config.vocab_size) - base_bpe_ids` returns; a `Stage2DonorVocabulary`
  or C1 port following that derivation would delete every added token —
  eos/bos 248,044 and the vision/image/video control tokens included. The
  measurement and the hazard are both in the build receipt
  (`tokenizer.vocabulary_arithmetic`).
* No width fold (C8) and no reader-metric selection census beyond
  `qf_mlp.reader_metric`; C8 was not in this lane's scope.
* `convert_hf_to_gguf` is invoked, not vendored — there is no llama.cpp checkout
  on the server yet, so the hook currently returns the exact command and does
  not run. Point `--gguf` at a checkout to close this.
* The frozen-review hook shells out to the review lane's runner; this lane never
  scores anything.
* C7's relation solve runs on gain-scaled **pre-norm** rows; `k_norm`'s per-token
  RMS division is not in the solve. The merged key's RMS shift is measured
  (`key_rms_ratio`, 0.709 on the fixture) and reported as `rms_frame_mismatch`,
  and it is deliberately **not** installed — `k_norm`'s gain is shared by every
  query head reading that key. Unrepaired approximation.
* **Zero-gain norm coordinates.** C7 solves on gain-scaled rows and divides the
  gain (`1 + weight`, which is what `Qwen3_5RMSNorm.forward` multiplies by) back
  out before installing. Qwen3.5-4B **layer 27's `q_norm` has one coordinate —
  index 1 — whose bf16 weight is exactly `-1.0`**, so its effective gain is
  exactly `0`; no other full-attention layer of that parent has one (the next
  smallest is `0.00390625`). Coordinate 1 is the real half of rotary plane 1, so
  the scaled row is zero while the plane is not, and the complex coefficient the
  relation solve applies to the plane puts a non-zero value back into that row —
  which the division then turned into `±Inf`. `q_norm`'s own per-token RMS
  spread that to NaN across the whole head, and the gated writer stream, its
  Gram, and the ridge derived from that Gram's diagonal mean were all NaN. This
  is what killed the first K2KV build at layer 27 with `linalg.cholesky: the
  leading minor of order 1 is not positive-definite`. `qf_attn.unfold_gain` now
  installs the **parent's own raw row** at such a coordinate: the score is
  unchanged (the norm multiplies it by zero either way) and the per-token RMS the
  norm divides the head by is the parent's. Recorded per layer as
  `zero_gain_coordinates`; the log line names the coordinates. Pinned by
  `qf_selftest.test_c7_zero_gain_coordinate`, which plants the same gain at the
  same geometry on the fixture and asserts the naked division is non-finite and
  `unfold_gain` is not.
* **Degenerate Grams are named, not guessed at.** `prior_anchored_gram` raises
  `qf_solve.GramNotPositiveDefinite` with the offending channels, the ridge, and
  whether the defect is a dead channel or a non-finite one; `law_abiding_solve`,
  `law_abiding_solve_gram` and `law_abiding_solve_split` fall back per rung to
  `eigen_anchored_gram` (symmetric eigendecomposition with an explicit
  eigenvalue floor, `EIGEN_FLOOR_REL`) and record what that cost as
  `solver_fallback` in the guard, i.e. in the receipt. The fallback **refuses**
  a Gram carrying a non-finite entry and a Gram with no positive eigenvalue,
  because neither has an answer that is not invented; the refusal that
  propagates is the Cholesky one with the fallback's reason appended.
  `qf_attn._resolve_writer` drops non-finite captured rows from the normal
  equations, counts them into `stream_repair`, and refuses above
  `NONFINITE_ROW_CEILING`. Nothing is clamped or imputed. Note the negative
  result pinned by `test_gram_refusal_and_eigen_fallback`: a merely DEAD channel
  cannot raise this error, because `A^T A` is PSD and `lam` is positive — only
  an all-zero stream or a non-finite one can, which is what made the row defect
  the only remaining explanation.
* `wavea_queue.sh` takes `QUEUE_TAGS="K2KV K2FG T9 V62080"` to run a subset of
  the `QUEUE` array. It filters only — array order, flags, sealed
  `--recipe-command`, and every guard are unchanged, and an unknown tag is a
  refusal. It exists so a wave that stopped part-way is resumable without
  editing the array that seals the recipes.
* The attention probe (`attention_gate_probe` + `logit_scale_ratio`) is run in
  the suite on the **first** full-attention layer only, because that is the one
  whose input is provably the parent's byte for byte. Later layers see a
  different input and their probabilities are not directly comparable.
* The self-tests run on a random-init model. They prove **mechanics** (indices,
  shapes, config re-derivation, gain conventions, identity gates, digests) and
  they say nothing about how much intelligence any fold retains — a random
  parent has none to retain. Fold quality is measured only on a real parent, by
  the frozen review.
