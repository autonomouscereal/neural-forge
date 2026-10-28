# Additional folding tests required before scaling (the list Cyril offered to help with)

Everything below is written against the estate as digested (READ 1–5). The record proved closed-form folding on exactly one model (LFM2.5-230M; machine-verified intelligence floor 0.4116484014x, ARCH.V10CL896X09, 1/15; best 14/15 at 0.794570x, ARCH.V19), and STANDING_STATE §7.4 says "No wave has measured what any of this transfers to." Chimera Omni needs six proofs that do not exist. Each is designed as the smallest build that produces a go/no-go under the standing rules (closed-form fold stage, behavior-first artifact-tolerant grading, build-don't-predict, standard-checkpoint export, dense, quantization last).

Hardware assumed throughout (READ 4 snapshot 2026-09-01): GPU0 RTX 3090 24 GB idle, GPU4 RTX 3090 occupied by `comfyui_h3_gpu4` (17.7 GB; Cyril's ruling: wait for H3 workflows, kill queued, then take it), GPU6 Quadro RTX 8000 48 GB idle, GPU1/2/3/5 Tesla P40 24 GB sm_61 idle (fp32 only), 251 GiB RAM, 88 CPUs; `model-surgery-lab/venv` (torch 2.11.0+cu128, transformers 5.12.1) and `venv-pascal-cu118-torch260` for P40s. No llama.cpp source, no gguf-py, no nvcc on the host.

## Execution order (risk retired per hour, highest first)

| Order | Test | Risk it retires | GPU-h | Wall clock | New code | Blocking prerequisite |
|---|---|---|---|---|---|---|
| 1 | T6-A: 256k Q4 hybrid inference smoke on P40 and 3090 | The end-goal envelope (27B hybrid, Q4, 256k, 24 GB) may not exist on this hardware at all | ~5 | 1 day | none (existing GGUF) | llama.cpp binary that loads on sm_61 |
| 2 | T4: fold-then-retrain recovery pricing | Whether a fold is worth anything as an initializer; prices the whole program | 12–16 | 2–3 days | none (stage-2 trainer ae4b8fb5 exists) | Cyril's column A/B floor ruling; GPU0 + GPU6 |
| 3 | T1: cross-model residual bridge | Whether "fold model A's block into model B's stream" works closed-form at all (the program's core novelty) | 4–6 | 2 days | cross-tokenizer pairing + bridge fitter (~1 day) | none (both models on the box) |
| 4 | T6-B: export toolchain + K3-shaped all-dense GGUF load | Whether Chimera's own architecture has any runtime/export path | ~4 | 1 day | tiny K3-shaped writer | llama.cpp checkout + gguf-py; ~18 GB HF hybrid download |
| 5 | T5: 27B-scale feasibility and costing | Whether a fold matrix at 27B is affordable on an RTX 8000 + 251 GiB RAM | 10–15 | 3–4 days | layer-streaming capture harness (~2 days) | Qwen/Qwen3.8-27B bf16 safetensors (~54 GB) download |
| 6 | T2: MoE-to-dense absorption on a small MoE | Whether any closed-form MoE-to-dense initializer exists (K3's only weight-side entry) | 10–15 | 3 days | expert-union coreset solver | OLMoE-1B-7B (~14 GB) and optionally DeepSeek-V2-Lite (~32 GB) downloads |
| 7 | T3: modality-token interface fold | Whether media-model knowledge can enter an encoder-free trunk closed-form | ~4 | 1–2 days | probe harness | CIFAR-10 download |

Total: roughly 50–65 GPU-hours and 6–8 days of engineering, runnable in about one week with GPU0, GPU6 and the P40s in parallel. T6-A and T4 alone answer the two questions that decide whether the rest is worth running.

---

## T6-A. 256k Q4 inference smoke on a hybrid-attention 27B (P40 and 3090)

**Why first.** The end goal is "Q4, fits an RTX 3090 and the P40s, 256k context." READ 5's arithmetic says a Qwen3.8-27B-class 3:1 GDN:full hybrid at Q4_K_M (15.4 GiB) with q8_0 KV (8.5 GiB at 262,144 tokens) plus 1.5–2 GiB compute buffers is ~26 GiB: it does not fit 24 GiB. It fits only with q4_0 KV (4.5 GiB, derived total ~21.9 GiB) or IQ4_XS weights (borderline). None of this is measured on this box. If the envelope fails here, the target must be re-specified before any folding is scaled.

**Asset already on the box.** `/mnt/ai_data/models/lmstudio/lmstudio-community/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf` (15.41 GiB, arch qwen35, 64 blocks, 24 heads/4 KV, ctx 262144, dense). It is the same GATED_DELTA_NET + gated-attention hybrid class as Qwen3.8-27B, so it measures the right kernels. Note: using a Q4 file here is an inference-envelope test, not a fold; the quantization-last rule is untouched.

**Protocol.**
1. Kernel availability on sm_61: run `ghcr.io/ggml-org/llama.cpp:server-cuda` (version 10068, on the box) with `CUDA_VISIBLE_DEVICES` pointed at one P40. If it reports "no kernel image is available", the image was built without sm_61 (CUDA 13 dropped Pascal; READ 5). Fallback: build llama.cpp from source in an `nvidia/cuda:12.8-devel` container with `CMAKE_CUDA_ARCHITECTURES=61;75;86` (20–40 min). This single fact is worth knowing before anything else.
2. Fit ladder on one P40 and on GPU0: `-c 262144 -fa on` with (a) `-ctk q8_0 -ctv q8_0`, (b) `-ctk q4_0 -ctv q4_0`. Never mix K/V types (default CUDA FA kernels support only matched pairs; mixed pairs silently fall back to CPU, issue #24485). Record peak VRAM from `nvidia-smi` and the server's own buffer report.
3. Speed at depth: prefill wall-time to 4k/32k/128k/256k and decode t/s at each depth. The CUDA `gated_delta_net.cu` kernel has "TODO: Add chunked kernel" (sequential prefill); READ 5 extrapolates 15–30 min for a 256k prefill on P40. Measure it.
4. Quality at depth: needle-in-a-haystack at 32k/64k/128k/256k for both KV types; the 15-case frozen review bank through llama-server (canonical artifact-tolerant lens). Reference for the q4_0-vs-q8_0 delta: the 2-GPU split (GPU6 + GPU0) at q8_0.
5. Repeat step 2–4 on GPU0 (RTX 3090) for the 3090 target.

**Cost.** ~5 GPU-hours (mostly prefill waiting), ~1 day wall. No new code.

**Go/no-go.** GO if a 27B hybrid at Q4 loads on sm_61, holds 262,144 context in 24 GiB with q4_0 (or q8_0) KV, passes needle ≥90% at 256k, and decodes ≥3 t/s at 256k on P40. NO-GO on any of those means the 256k-on-24 GB target requires the Chimera KDA:MLA layout (READ 5: 2.39 GiB KV at q8_0 for 64 layers, 4x smaller) — which then depends on T6-B — or a smaller context budget, and Cyril re-specs before the fold campaign scales.

**Honest limits.** This tests text only. No runtime on the box can do media *output* at Q4 (see the infeasibility list at the end).

---

## T4. Fold-then-retrain recovery proof (the stage 2 that was never run)

**Why second.** Cyril's staged plan is compression + distillation + retraining recovers the 90%, and the record's own position is "a fold is an initializer, not a product" (PHASE1 §6), "recovery belongs to RETRAINING, not decoding" (wave 35; 20 decode settings × 300 lanes, 0 recoveries). Yet no retraining of any fold child has ever been measured. Everything needed exists: STAGE2_FIRST_MEASUREMENT_DESIGN (4 arms), trainer ae4b8fb5 (138 tests), 26 sealed corpus shards (541M of 1.12B teacher self-gen tokens), the frozen 15-case instrument and 7-column benchmark row. The gate was closed by Cyril on 2026-08-10 when stage 1 was reopened; the 2026-09-01 pivot makes it the highest-value unrun experiment in the estate, because it prices every later stage.

**Arms (all identical recipe: KL to LFM2.5-230M parent logits + LM cross-entropy on the sealed shards, same optimizer, same schedule, same token budget, checkpoints at 25M/50M/100M/200M tokens).**
- A1: fold-init deep child ARCH.V10CL896X09 (0.4116484014x, 94,552,832 params, 1/15 both lenses) — or ARCH.V3CL768 (0.3859919152x) if Cyril rules column A. This arm is the program's "deep fold + recover" claim.
- A2: random-init at byte-identical topology to A1. This is the control that decides whether folding buys anything as an initializer.
- A3: fold-init shallow child ARCH.V12CL1792 (0.618813x, 7/15 robust frontier). Measures recovery as a function of fold depth: does the deep fold reach the shallow child's level, and at what token cost.
- A4 (optional, cheap): ARCH.V19 (0.794570x, 14/15) — does retraining push a near-parent child above the parent band (12/15)?

**Instrument.** At every checkpoint: frozen 15-case review (sealed and canonical lens, per-lane byte comparison), 7-column benchmark row (arc_e/hella/piqa/wino/lambada/lam-ppl/wiki-ppl), union copy measure (pre-registered 0.60) so copy-back cannot count as recovery, and the 39-restatement robustness column. Cyril scores the 15-case outputs himself at the 100M and 200M checkpoints (his P26 rule).

**Cost (derived).** Student fwd+bwd ≈ 6 × 94.5M = 0.57 GFLOP/token; teacher forward ≈ 2 × 230M = 0.46 GFLOP/token; ≈1.03 GFLOP/token → 200M tokens ≈ 2.1e17 FLOP ≈ 2.3 h per arm on a 3090 at ~25 TFLOPS effective (assumption: ~35% MFU at this size), plus ~1 h of checkpoint evals → ~3.5 h/arm; 3 mandatory arms ≈ 11 GPU-h, 4 arms ≈ 14 GPU-h. A1 and A2 on GPU0 and GPU6 in parallel; A3 follows. P40 fallback is fp32-only (~12 h/arm). The stage-2 design's own estimate for its 4 full arms at 1.12B tokens was 145–150 GPU-h; this is the 200M-token minimal version.

**Go/no-go.**
- GO ("fold is a valuable initializer"): A1 leads A2 by ≥3 cases (the instrument's resolution: gap-1 orderings invert 34.5%) at every checkpoint ≤100M tokens AND on ≥4 of 7 benchmark columns, and A1 reaches Cyril's scored bar (8/15, MXN4 baseline) within 200M tokens.
- NO-GO: A2 matches A1 within resolution by 100M tokens. Then folding contributes nothing that 100M tokens of distillation does not; Chimera Omni's fold stage collapses to "select topology, distill from teachers" (the arXiv 2605.28207 path), and the closed-form matrix at 27B should not be run.
- The number the program needs regardless of verdict: cases recovered per 1M tokens and per GPU-hour, and the token count at which A1 re-enters the parent band. Scaled to 27B (derived: a 0.5x child + 27B teacher ≈ 1.35e11 FLOP/token; 200M tokens ≈ 2.7e19 FLOP ≈ 7 days on GPU0 + GPU4 + GPU6 at ~40 TFLOPS combined) this is the first honest price tag for the recovery stage.

**Assumptions.** (i) The 230M result transfers in sign, not magnitude (nothing has been re-measured on a second model). (ii) The sealed shards are still intact after the 2026-08-19/22 cleanups (verify hashes first; "verify artifacts, not exit codes").

---

## T1. Cross-model residual-space alignment (fold A's block into B's stream)

**What it must decide.** Chimera Omni folds several models into one trunk. Every donor pair in the program has different widths, tokenizers and mixers (Qwen 27B d=5120/V=248,320; K3 d=7168/V=163,840; H3 and the media models are latent-space DiTs). No wave ever put a foreign block into a stream. The estate's own laws are hostile: locality bounds every gradient-free per-component objective (law 33), reconstruction never predicts behaviour (laws 1, 6, 49), the second composition step is fatal on five axes (law 27). The literature is split: Orthogonal Procrustes wins for steering transfer (Atlas-Alignment, 2510.27413v2) but a cos-0.97 linear bridge gave zero functional transfer Pythia-160M→410M (2606.03280). So this must be built, not argued.

**Smallest decisive pair (both on the box).**
- Host H = LFM2.5-230M (d=1024, 14 blocks; fully instrumented: parent band 12/15 digest 865f006e, excision census, benchmark row identical across 32 re-runs).
- Donor D = Qwen3-0.6B-Base (`~/.cache/huggingface/hub`, 1.2 GB; d=1024, 28 layers, GQA + RoPE + SwiGLU — assumption on the exact config, verify from config.json). Same width makes rung 1 a square bridge; different family, tokenizer and mixer makes it the honest cross-model case.
- Rung 2 (rectangular, the program's real case): D = Qwen3-4B-Base (d=2560, 7.6 GB on the box).

**Protocol.**
1. Evidence: 2M tokens of the campaign's sealed teacher self-gen shards, run through both models; pair residual states at positions where both tokenizations end a token at the same character offset (common boundaries); hold out 10% for λ selection (law: two-sided solves need their own validation split; trust-region λ was wrong by 3–5 orders).
2. Bridges (closed-form only, all counted as model parameters per Cyril's P34 ruling — a 1024² pair adds 2.1M params, 0.9% of the host): (i) Orthogonal Procrustes R = UVᵀ from SVD(XᵀY); (ii) ridge (XᵀX+λI)⁻¹XᵀY; (iii) ridge with RMS-gauge/mean matching. Fit B_in on (host layer-i input, donor layer-j input) and B_out on (donor layer-j output, host layer-i output). Choose j by held-out R² of B_in (census used for *ordering* only, never as a gate) and also at matched relative depth.
3. Host sites: L05 (excision control 6/15), L09 (10/15), L13 (13/15, above parent — a free site where a transplant can only show harm or positive transfer). Re-derive `config.layer_types` on any structural change (harness law 54).
4. Arms per site: (a) replace host block i with B_in→donor block j→B_out; (b) insert the bridged donor block as an extra block (host keeps all 14); (c) excision-only control (known from the census); (d) identity bridge (no map); (e) random-orthogonal bridge.
5. Grading: frozen 15-case review (canonical lens), 7-column benchmark row, union copy measure. Positive-transfer probe: a 20-prompt extension bank on the lanes that die first in the ladder (coding 0.05, arithmetic 0.08, logic 0.03 hold rates) where Qwen3-0.6B-Base is stronger than LFM2 — exact-answer scored plus Cyril's read. Prior: no positive transfer; the decisive comparison is (a) vs (c).

**Cost.** Captures: minutes on GPU0. Solves: 1024² or 2560×1024 Gram/cross moments in fp64, seconds. Reviews: ~15 children × ~5 min. Total ~4–6 GPU-hours; ~1 day to write the cross-tokenizer pairing and bridge fitter (the fold/review/seal harness exists in neural-forge).

**Go/no-go.**
- GO: at ≥2 of 3 sites, the fitted-bridge transplant (a) beats the excision control (c) by ≥3 cases while identity (d) and random-orthogonal (e) bridges do not (the bridge is carrying the donor's computation), and insert (b) holds within 2 cases of the parent. Then proceed to rung 2 (rectangular) with the same gate.
- NO-GO: (a) ≤ (c) at every site with every bridge. Foreign blocks cannot be inserted without gradients; the "fold multiple models into one" mechanism is dead closed-form, and donors can only enter Chimera Omni as teachers (distillation) — a different program from the one described. Partial: (b) holds but (a) fails = foreign blocks are inert, still no-go for absorption.

**Assumption stated.** The program's pairs are all rectangular and cross-tokenizer; rung 1 passing is necessary, not sufficient.

---

## T6-B. Export toolchain and a K3-shaped all-dense GGUF load

**What it must decide.** The chimera-omni repo has no GGUF, safetensors or HF adapter path (OMNICODER_EXPORT_MAP: GGUF "Not Imported"; the only serialization is `torch.save`). Cyril's ruling: "[Cyril quote withheld]" (gguf/llama.cpp). Two facts decide whether Chimera's architecture is exportable at all: llama.cpp merged Kimi-K3 (PR #26185, 2026-08-15, CPU+CUDA) and GATED_DELTA_NET with KDA mode (PR #19504). Chimera's k3_aligned profile is KDA + gated MLA + Block AttnRes + SiTU (K3's config also says `hidden_act situ`), but dense FFN at every layer; K3 has `first_k_dense_replace 1`, so llama.cpp's kimi_k3 graph must already build a dense FFN for layer 0. Whether it accepts *all* layers dense is unverified.

**Protocol.**
1. Install a llama.cpp checkout and `gguf-py` into `model-surgery-lab/venv` (READ 4: `pip show gguf` is empty in every venv). Build CUDA 12.8 binaries with arches 61/75/86 (reuse from T6-A).
2. Conversion path on a real hybrid: download one small HF-format GDN hybrid (Qwen3.5-9B, 32 layers with 8 full-attention — the model whose q4_0-KV BLEU 1.000 result is cited in issue #21385; ~18 GB bf16; assumption: the smaller 3.5 siblings are equally usable). `convert_hf_to_gguf.py` → `llama-quantize Q4_K_M` → load on a P40 → 15-case review parity against the bf16 HF model on GPU6 (the fold-then-export-then-quantize chain the program will use, exercised end to end).
3. Chimera path: write a K3-shaped *all-dense* synthetic checkpoint from `k3_aligned_test_config()` shapes into K3 tensor naming (a small writer in the repo), convert, and load in llama.cpp on CPU. Diagnose exactly which of KDA, gated MLA (shared extra NoPE channels, sigmoid gate), AttnRes, SiTU, and dense-every-layer are accepted.

**Cost.** ~4 GPU-hours; 1 day. Downloads: ~18 GB.

**Go/no-go.** GO if (2) round-trips with review parity and (3) loads with all-dense layers: Chimera Omni has a standard export path with no new llama.cpp architecture. Partial: (3) fails on AttnRes or all-dense → either a llama.cpp arch patch (weeks) or the block pattern must be constrained to what kimi_k3/qwen35 already implement — a decision Cyril should make before the trunk topology is fixed. NO-GO on (2) means the fold-export-quantize chain itself is broken and everything downstream waits.

**Honest note.** This gives text export only. Native media bridge/head, x0 flow sampling and the typed ledger have no llama.cpp path.

---

## T5. 27B-scale feasibility: can closed-form solves and evidence capture run here at all?

**Source problem first.** Cyril's named source is "Qwen 3.8 27B". It is not on the box in any foldable form: only `Qwen3.6-27B-Q4_K_M.gguf`, and a 170 MB `.part` of a Qwen3.8-27B GGUF. A Q4 GGUF source is quantization-first (forbidden; and dequantized Q4 weights cannot be refit faithfully). The first action item is downloading `Qwen/Qwen3.8-27B` safetensors (~54 GB bf16; assumption on availability and size).

**Memory math (derived from the Qwen3.8-27B config in READ 5: hidden 5120, 64 layers = 48 GDN + 16 gated full attention, GDN 16 k-heads/48 v-heads × 128, full 24 Q / 4 KV × 256, FFN 17,408, vocab 248,320; per-layer projection counts are approximate):**

| Tensor class | Size | Count | Params | bf16 | fp32 |
|---|---|---|---|---|---|
| embed_tokens (and lm_head if untied — assumption: untied) | 248,320 × 5120 | 2 | 2.54B | 5.1 GB | 10.2 GB |
| FFN gate/up/down | 5120 × 17,408 | 64 × 3 | 17.1B | 34.2 GB | 68.5 GB |
| GDN layer (q,k,v,z,o + small) | ~115M/layer | 48 | ~5.5B | 11 GB | 22 GB |
| Full-attention layer (q,k,v,gate,o) | ~105M/layer | 16 | ~1.7B | 3.4 GB | 6.7 GB |
| Total | | | ~26.9B | ~54 GB | ~108 GB |

Consequences: FFN is ~70% of the body (matches law "MLP ≈ two-thirds of body"); the untied interface is ~9.4% (tied ~4.7%) — larger than READ 2's c=12 derivation (3.2–6.2%) because V=248,320 and untied; the vocab-coreset axis (the LFM2 champion, 14/15 at 0.79x) is therefore worth up to ~7% at 27B, at the known +15.8% sequence-length cost, which shrinks effective 256k context. 48 of 64 layers are GDN: the record has **zero** fold coverage on linear-attention layers (LFM2's conv is "LFM2-specific", law 55), so the only transferable axes into GDN blocks are whole-block excision (the census method the record calls its most scale-transferable data) and value-head merge (law 52: 16→12 held at 8/15 with a joint solve; 16→8 dead with any solve).

Full-model bf16 (54 GB) does not fit the RTX 8000 (48 GB). Capture must stream layers from RAM; review decode must use a two-GPU split (GPU6 + GPU0 = 72 GB) or CPU.

**Protocol.**
1. Layer-streaming capture harness: hold one layer on GPU6 at a time (≤0.55 GB), keep the current activation set for 1M evidence tokens on GPU6 (1M × 5120 × bf16 = 10.5 GB), accumulate per-site fp64 moments only (input Gram 5120² = 210 MB; SwiGLU intermediate Gram 17,408² = 2.4 GB; cross 17,408×5120 = 0.7 GB; reader moments for the width law). Never store all 64 boundaries (672 GB).
2. Time one full capture (derived: 2 × 27e9 × 1e6 = 5.4e16 FLOP ≈ 20–45 min on GPU6 at 20–45 TFLOPS effective; PCIe streaming of 54 GB ≈ seconds). Record wall time, peak GPU and RAM.
3. One fold per transferable axis, timed: (a) FFN channel coreset at 4x on 3 non-adjacent sites (Gram-pivoted Cholesky QR on the 17,408² Gram — equivalent to the campaign's pivoted QR on the data matrix for column selection — then ridge refit of down_proj; solve minutes); record per-site write cosine (law 46: 0.93 alive / 0.86 dead on LFM2 — ordering information only, never a gate); (b) vocab coreset deletion at K chosen by ≥81% occurrence coverage on the evidence (law 15); (c) one interior-late block excision with `layer_types` re-derived and an assertion of zero attention mass above the diagonal (law 54).
4. Price each candidate on the real base (law 32) and decode the 15-case review on the GPU6+GPU0 split; record minutes per candidate.

**Cost.** ~54 GB download; ~2 days harness; 10–15 GPU-hours for the timing runs.

**Go/no-go.** GO if capture ≤2 h and fold+price+review ≤3 h per candidate: a 100-candidate 27B matrix costs ~100–300 GPU-h on GPU6 (1–3 weeks), affordable. NO-GO if a candidate costs >6 h: the matrix runs on a proxy (Qwen3.5-9B hybrid, same layer mix) and only the final composition is built at 27B. Either way this test produces the first 27B-specific behavioural numbers (three FFN sites at 4x, one vocab rung, one excision) and shows whether the LFM2 laws hold their sign on a second, hybrid, model.

**Assumptions.** Untied embeddings; the record's "price on the real base" law scales linearly; the P40s are unusable for any 27B bf16 work (no bf16; fp32 = 108 GB).

---

## T2. Expert-to-dense absorption on a small MoE (closed-form)

**K3 reality first (READ 4/5 facts).** Kimi K3 is not dense: 2.8T total / 104B active, 896 experts top-16 + 2 shared, 92 MoE layers, 1,560,936,091,448 bytes MXFP4 on a CIFS share. Expert mass ≈ 896 × 3 × 3584 × 3072 ≈ 29.6B params per layer, 2.72T total. Folding that into a ~15B dense FFN is ~180x. The only published MoE→fully-dense conversion (arXiv 2605.28207) needs ~4B tokens of distillation; the training-free tools (MergeMoE, ExFold, Sub-MoE, PuzzleMoE) top out at 50% expert reduction, and REAP shows pruning beats merging there. On this box, evidence-based folding of K3 is infeasible: a forward pass needs the whole 1.45 TiB reachable per token (RAM + VRAM = 435 GB; a Q2_K GGUF is ~700 GB and the root disk has 615 GB free), so it would run from CIFS at minutes per token. What *is* feasible: data-free weight statistics per shard on CPU (expert SVDs, cluster structure; I/O-bound, a few hours per full read), and loading K3's non-expert body (attention + KDA + norms + embeddings, derived ≈ 28–30B params, ~56–60 GB bf16) into RAM as a speculative donor for T1's rectangular rung. Recommendation to put to Cyril: K3 enters Chimera Omni as architecture (KDA + gated MLA + AttnRes + SiTU, already adopted) unless T2 and T1 both pass.

**Small-MoE test that decides whether any closed-form MoE→dense initializer exists.**
- Rung A: OLMoE-1B-7B (assumption: d=2048, 16 layers, 64 experts top-8, expert intermediate 1024; ~14 GB bf16 download; fits GPU0). Used by HARC.
- Rung B (K3-shaped analog: MLA + shared experts + fine-grained routed experts): DeepSeek-V2-Lite (assumption: 15.7B total, 27 layers, 64 routed + 2 shared, top-6, MLA kv_lora_rank 512; ~32 GB bf16 → GPU6). Used by arXiv 2605.28207.

**Protocol (per rung).**
1. Calibrate the instrument on the new parent: frozen 15-case bank and 7-column benchmark row, parent run three times to a stable digest (as the campaign did for LFM2).
2. Capture on 1M tokens: router logits, per-expert usage f_j, per-expert input second moments, and MoE-layer (X, Y) pairs.
3. Dense targets at 2x/4x/8x of MoE-FFN storage (OLMoE: dense I = 32,768 / 16,384 / 8,192 versus 64 × 1024 expert channels; 4x is Cyril's per-component minimum).
4. Methods, all closed-form: (M1) usage-ranked prune-and-concatenate with mean-gate scaling (the REAP-style baseline); (M2) MergeMoE frequency-weighted output-space merge with least-squares T1; (M3) ExFold ridge scalar folding of dropped experts into kept ones; (M4) Sub-MoE joint SVD truncated to one dense FFN; (M5) the estate's own champion mechanism transferred: treat the union of all experts' channels (65,536 for OLMoE) as one gate-weighted SwiGLU, run stream-selected pivoted-QR coreset on the routed, gate-weighted activation matrix (law 48: verbatim coreset beat low-rank 40/42 on SwiGLU writes), ridge-refit the down projection. Router calibration (HARC closed form) applied where a partial router survives; the final dense child has no router.
5. Two composition arms: all layers from original-MoE evidence, and sequential with evidence refreshed from the partially-dense model (Cyril's P21 method; the estate's "evidence-refresh" was ordered three times and never run).
6. Controls: random-init dense of the same shape; FFN zeroed (how much the FFN carries).

**Cost.** Rung A: capture ~15 min; the 65,536² Gram is 34 GB fp64 (RAM) or 17 GB fp32 (GPU6); pivoted Cholesky for k=16,384 is minutes on GPU6; ~5 methods × 3 rungs × 2 arms = 30 children + reviews ≈ 8–12 GPU-h. Rung B adds ~6 GPU-h.

**Go/no-go.** GO ("a closed-form MoE→dense initializer exists"): a 4x dense child holds ≥1 content lane on the frozen bank (union copy measure applied; artifacts discounted per Cyril's 2026-08-28 standard), beats random-init (expected 0/15), and the best merge/coreset method beats the prune-concatenate baseline (M1) by ≥3 cases — otherwise merging adds nothing over pruning, which is REAP's finding. NO-GO: every method at 4x reads 0/15 or artifacts only. Even a GO at 4x–8x on a 7B MoE is no license for 180x on K3; it licenses only using the mechanism on a MoE donor small enough to be evidence-captured here, and feeds T4 (fold-init versus random-init, now on a MoE-derived dense child).

---

## T3. Modality-token interface folding (image/audio knowledge into a text trunk's token space)

**What the record already implies.** Chimera is encoder-free: raw patches (default feature_dim 3072 = 1·3·32·32) → linear bridge → trunk; raw waveform windows for audio. Every image/audio/3D model on the box is a latent-space model with a VAE (Qwen-Image 20B, H3 20.1B with video/audio VAEs, LTX 2.3 22B, Hunyuan3D 2.1, ACE-Step on an Oobleck VAE). Their weights operate on latents the raw-signal trunk never sees, so "folding their knowledge" closed-form means either folding VAE encoders (encoders, excluded by the architecture) or transplanting DiT blocks into the trunk (T1's rectangular, cross-modal case). The v4 proof shows the media path *trains* (3,185,932 params, 4,000 steps, 2,614.5 s on GPU6, 13/13 gates). The decisive question is therefore narrow: can a closed-form interface give a frozen text trunk any media capability, and does donor knowledge move through it?

**Protocol.**
1. Data: CIFAR-10 (32×32×3 = exactly Chimera's 3072 raw feature; download required) with captions "a photo of a {class}"; audio: 10-keyword spotting from the cached `librispeech_asr` (labels derived from transcripts) using Chimera's raw 1536-sample windows.
2. Trunk: LFM2.5-230M (instrumented; d=1024).
3. Arm E (encoder-free, raw): ridge W (3072→1024, or 4 × 768 patches) from raw pixels to the trunk's caption-token embedding centroid; image tokens prepended; prompt "This is a photo of a"; LM-scored class continuation, top-1 on the 10k test split.
4. Arm D (donor features, evidence-time only): SmolVLM2-256M-Video-Instruct's vision tower features (on the box, 984 MB) through the same ridge — measures whether donor knowledge moves through a linear interface at all.
5. Arm D′ (encoder-free-compliant absorption): transplant the donor's patch-embed + first k vision blocks into the trunk as extra blocks via T1 bridges, so no separate encoder module remains; same probe. This is a gray zone under the encoder-free rule — flag for Cyril's ruling.
6. Controls: random W, no-image prior. Text retention: 15-case review (E/D are additive at input and cannot change text behaviour; D′ must be measured).

**Cost.** ~3–4 GPU-hours on GPU0; 1–2 days including the probe harness.

**Go/no-go.** GO for closed-form raw interface: arm E ≥30% top-1 (3× chance). GO for donor-knowledge transfer: D ≥60% and D′ ≥0.8 × D with text review within 2 cases of parent. Honest priors: E lands near chance (a linear map from raw pixels to semantics is weak, and the LM readout weakens it further); D succeeds; D′ is bounded by T1. Expected program consequence: media-model weights are not closed-form donors for an encoder-free trunk; the media path is training, with the media models as *teachers* (ACE-Step already accepted as a music specialist; Qwen-Image qualified; LTX 2.3 rejected; H3 quarantined pending the license ruling) — which is what the v4 result already says. That is a useful negative: it tells Cyril where folding applies (text trunk) and where it does not.

---

## What the hardware and the record make infeasible (stated plainly)

1. **Kimi K3 as an evidence-folded source.** 1.45 TiB MXFP4 on CIFS versus 435 GB RAM+VRAM and 615 GB free local disk; no vllm; llama.cpp K3 support exists upstream but a ~700 GB Q2_K file cannot be hosted locally. Only data-free statistics and the ~56–60 GB non-expert body are reachable. 180x expert→dense compression has no method, published or in the estate (the record's FFN cliff is 2.86x per site; all-14-sites 4x died).
2. **Qwen3.8-27B** is not on the box in a foldable form; Qwen3.6-27B exists only as Q4 GGUF (quantization-first). A ~54 GB bf16 download is the gating action.
3. **MiniMax H3** exists only as ComfyUI repacks with no HF config, on VAE latents, with the repo still fail-closed on license; its realistic roles are architecture (the joint AV flow design, already absorbed) and, if Cyril's internal-use ruling is applied to the catalog, teacher data.
4. **No runtime exists for media output at Q4 on a P40 or a 3090.** llama.cpp does text; its mtmd path does encoder-based media *input*; nothing runs the native bridge/head or x0 flow sampling. P40 is fp32-only, so torch inference of a 27B is not an alternative (108 GB). The "all modalities in and out on a 24 GB card" end goal requires a new inference engine regardless of how folding goes.
5. **256k has never been trained in Chimera** (longest sequence 128 tokens; NoPE has no cap but length generalization is the pending M5 curriculum). Folding does not create long-context ability; among the named sources only Qwen3.8-27B (262,144 native) and K3 (1M) have it.
6. **Multi-donor composition** is predicted dead closed-form by the most-replicated law in the record (first composition step survivable, second fatal on five axes; ~3 cases per independently solved component). The only recorded remedy is retraining between steps — which is why T4 runs before any multi-donor fold is attempted.
7. **GDN/KDA layers have zero fold coverage** in the record; 48 of Qwen3.8-27B's 64 layers and 69 of K3's 93 are linear-attention. Excision and value-head merge are the only axes with any transferable evidence.

## What Cyril needs to supply

- Downloads to `/mnt/ai_data/models`: `Qwen/Qwen3.8-27B` safetensors (~54 GB, T5), one small HF GDN hybrid such as Qwen3.5-9B (~18 GB, T6-B/T5 proxy), OLMoE-1B-7B (~14 GB) and optionally DeepSeek-V2-Lite (~32 GB) for T2, CIFAR-10 for T3.
- Rulings: column A (0.3859919152x, echo persistence) versus column B (0.4116484014x, machine-verified) as the T4 fold-init arm; whether transplanting an encoder's blocks into the trunk (T3 arm D′) is permitted under encoder-free; whether K3 is architecture-only for this phase; whether the H3 internal-use license ruling is to be applied to the catalog.
- GPU4: kill `comfyui_h3_gpu4` once its H3 workflows finish (his 2026-08-28 ruling) so both 3090s are available for T4.
- Manual scoring of the 15-case outputs at the T4 100M/200M checkpoints and of the first 27B children from T5 (his P26 rule).
- Confirmation that the 26 sealed stage-2 corpus shards and the neural-forge ledger survived the 2026-08-19/22 cleanups (hash check before T4 starts).