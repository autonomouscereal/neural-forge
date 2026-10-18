# OMNI1 Stage D: the distillation design (synthesis, 2026-10-12, revised after adversarial review)

Status: **design only.** Nothing here has been launched. No GPU was touched and nothing was written on the server.
Base: the judges' unanimous winner, **memory-first** ([J1], [J2]). Grafts are taken from capability-first and pipeline-first, and every fatal flaw the judges named is removed (see the fix list at the end of §1).
Revision: every finding of the adversarial review of 2026-10-12 was checked against the local docs and code. The accepted ones are applied in place. The rejected ones, and the reason for each, are in the **Review disposition** appendix at the end.

**Source key.** Every number carries a tag saying where it came from and how solid it is.

| Tag | Source |
|---|---|
| [TPS] | `distill/TPS_TABLE_2026-09-02.md` + receipts `$OC/distill/tps/*.json` |
| [TPR] | `distill/TPS_PROBE_RESULTS_2026-09-02.md` |
| [TPP] | `distill/tps_probe.py` (the streamed engine's code) |
| [CAMP] | `CAMPAIGN.md` (§1.8, §4.2, §4.3) |
| [CP] | `CHECKPOINT_POLICY.md` (R1-R10, open decisions 1-6) |
| [OCK] | `ledger/oc_checkpoint.py` (`PROTECTED_PREFIXES`, `REGISTRABLE_UNDER_PROTECTED`) |
| [R5] | `phase0-reader-digests/4_R5_training.md` (D-R5-1 … D-R5-11) |
| [R6] | `phase0-reader-digests/5_R6_k3h3.md` |
| [K3R] | `composite/K3_RESULTS_2026-10-08.md` |
| [VER] | `composite/c10/design_sweep/VERDICT.md` + `s5_metrics_full.json` |
| [C10] | `composite/tools/c10_union_install.py` (union target, SiTU→SiLU lift) |
| [CHT] | `chimera-omni/src/chimera_omni/modeling/charted.py` |
| [S7R] | `S7_READINESS_2026-10-10.md` |
| [SD] | `STATE_DELTA_2026-10-09.md` |
| [EXP] | `export/EXPORT_LANE_SPEC.md` (§2, §3.3.3, §4.1, run table) |
| [WALK] | `datasets/registry-walk-2026-09-02.json` |
| [RUN] | `COMPOSITE_STEP1_RUNBOOK.md` (S2, S7, S8, S9 rows) |
| [BANK] | `calibration/banks/omni_review_bank_v2_draft.json` |
| [SRV] | live server read, 2026-10-12 22:00Z, via the workflow brief |
| [MF] / [CF] / [PF] | the memory-first, capability-first and pipeline-first designs |
| [J1] / [J2] | the two judges |
| [REV] | the adversarial review of this design, 2026-10-12 |

Suffixes: **(m)** measured receipt, **(d)** derived (the arithmetic is shown), **(a)** assumed (it must be receipted before any budget binds).

---

## 1. The plan in one paragraph

This design starts where step 1 ends. Step 1 is what Cyril listed:
- refit the five media blocks onto the trunk+K3 (S5) child;
- assemble and register OMNI1 = `OC02-F-Q38-27BU-OMNI1` (27,535,655,753 params (d) [VER]/[SRV]);
- run the S8 capability tests;
- write the byte-for-byte copy `OMNI1.clone-pre-D1`. **Where that copy can live is an open owner decision (U9):** the registry refuses any clone destination under `/mnt` except `/mnt/lexar` [OCK]/[CP R6], Lexar has ≈ 40 GB left after OMNI1, and root cannot hold the full clone set (§4.1).

S0 starts only after the S8 package has been delivered. Before Cyril gives a recorded go, only **measurement and teacher caches** run: S0 probes, plus the sealed teacher caches of stage C. None of them changes a weight (the two training-rate probes run at lr = 0), so they run while Cyril reads S8 ([PF] graft).

After the go, four training stages run. Each ends in a registered **delta** checkpoint, a byte clone and a review package:
- **D1** repairs the 4x MLP coreset damage per layer. It is prefix-fed (Cyril's K3 rule) and trains the full 4,352-wide trunk FFN against the Qwen3.8-27B-U parent, on both 3090s. The frozen K3 term is **excluded from the D1 loss** and kept in the child's prefix advance, so D1 cannot learn to cancel K3.
- **D2** is end-to-end LoRA KD on **text + K3 only**. Signals: Qwen top-64 logits, cross-tokenizer K3 logits (always used; never skipped), EOS, an unlikelihood term, an on-policy GKD tail if Cyril rules a sampler for it, and a 32k long-context rung. Every text loss is computed over vocab rows [0, 248,320) only, so the C18 music band and the ledger bands are never pushed down.
- **D3** distills the media units on a new hybrid module (qwen3_5 layers + Chimera unit + interface rows): the five media units against cached teacher velocities, the C18 music-code band against the ACE LM, and **vision-in and video-in answer KD** (moved here from D2, because the QVL32 unit and the C11 rows are Chimera-only tensors with no host in a qwen3_5-layout student). Lanes run in the order ACE, Qwen-Image, Hunyuan3D, H3; H3 is licence-gated.
- **D4** is a consolidation replay. It folds the learned K3 per-column gain into the K3 down columns once, in fp32, and materialises the one merged dense checkpoint `OMNI1D4.clone-pre-T1` that stage T (the curated finetune) starts from.

**Engine.** The streamed engine that exists today [TPP] is forward-only and text-only. The plan therefore budgets every student stage on the **one measured 27B training cell** (RTX 8000 + 3090 split, fp16 LoRA, 151.24 tok/s at L = 2,048 (m) [TPS]) until two new tools pass parity arms in S0: (i) a streamed vision-language teacher and (ii) a streamed fwd+bwd trainer. If both pass and P2 measures a rate, the streamed bf16 engine on both 3090s becomes the upside path. In the planning path the student uses GPU6 + GPU4; in the upside path it uses the 3090s only. The P40s run teachers and samplers only. GPU5 is never used. Q4 happens exactly once, after stage T.

Wall clock (d, §11):
- **Planning path (split cell): ≈ 164 h after Cyril's go (≈ 6.8 days)**, plus ≈ 28 h of probes and caches that are hidden inside his S8 read if it takes that long. **≈ 192 h, or 8.0 days, if they are not hidden.** These use the upper bound of the unmeasured vision-in token count; the lower bound is ≈ 147 h after the go.
- **Upside path (streamed engine passes S0): ≈ 100 h after the go (≈ 4.2 days); ≈ 128 h (5.3 days) with the pre-go phase.**
- Both assume the clone set goes to a registrable NAS subtree (U9 option a). Option b (root) saves ≈ 3 h.

**Judge-named fatal flaws, and how this design removes them:**
1. NAS overflow: 7 × 55 GB = 385 GB against 345 GB free [J1, J2]. Fixed with delta checkpoints and only two merged materialisations (§4.1). The review then showed the registry refuses NAS clones at all [REV A1/OCK], so the location of the ≈ 201 GB permanent set is now an explicit owner decision (U9) with both options priced.
2. A G0 that demanded bit-exact equality after folding the 0.03 into bf16. Fixed: the 0.03 is already inside the jointly refit down projection [VER]/[C10]. There is no runtime scalar, so G0 checks a tensor-exact projection with no gain tensor; the D2 per-column gain starts at 1.0 and is folded once, at the D4 merge (§2 G0).
3. The D0 K3 pass overlapping D1's 84 GB pinned host buffers. Fixed: the K3 pass is serialised into stage C, before D1, and is split into four ≤ 0.5M-token passes so its attention-residual state (≈ 72 GB per pass) fits the host (§3).
4. A fallback rate that read scaler warm-up as a permanent 7/13 skip rate. Fixed: S0 P7 makes the reading at lr = 0; 81 tok/s is kept only if skips persist, and then it is a proposal for Cyril (D-R5-1), not an automatic cut.
5. No audio-in or video-in lane. Fixed: video-in answer KD in D3d; audio-in is surfaced as a teacher gap with a stage-T path (§5.7).
6. From the runners-up, not carried over: skipping K3 on a bits-per-byte probe, re-initialising from S5 (which un-composites OMNI1), type-gated adapters, and a frozen-trunk-only media stage.

---

## 2. Go/no-go from S8: decisions for Cyril, not automatic fails

**Precondition.** No gradient step that changes a weight runs until two things are true:
1. The S8 package for `OC02-F-Q38-27BU-OMNI1` has been delivered (owner rule). S0 starts only after this.
2. Cyril has recorded an explicit go as a sealed `cyril-score`/ruling record [CP R8].

**S8 correction needed before S8 launches** (step 1, not this design): the runbook S8 row assigns the companion VAE decodes and GLB export to "P40s (GPU2/3/5)" [RUN S8]. GPU5 is reserved for another project, so S8 must use GPU1-3.

Each line below is a **proposed reading** with a recommendation. Cyril's per-case read decides. Everything is graded artifact-tolerant: loops, echo, missing EOS and cap-rides do not count, and only content misses do. Finite fit numbers (relL2, KL, NLL, cos, PSNR) are printed beside every gate and are never a refusal.

| Gate | What S8 (or S0) must show | Reference numbers | Proposed reading → recommendation |
|---|---|---|---|
| **G0 structural** (wiring, not capability) | OMNI1 registered and `MANIFEST` verifies; loads in Chimera; one R-Q forward is finite; `clone-pre-D1` exists at the location Cyril picked (U9) and its digest equals the OMNI1 digest; S0-P1 reads every OMNI1 text tensor into the qwen3_5 layout **tensor-exact** through a zero-copy name/slice map: uniform FFN width 16,640 (the fused Chimera `[gate\|up]` sliced; 4,352 trunk + 12,288 K3 channels, which c10 lifted from SiTU to SiLU into the same FFN [C10]), **no gain tensor**. The 0.03 is already inside the jointly refit down columns [VER]. Forward parity (argmax agreement on R-Q 210) is reported as a diagnostic. | — | Any failure → recommend fixing step 1 (S7 install / clone) before any D spend. This is a plumbing fault, not a capability verdict. |
| **G1 text identity** ([CF] graft) | R-Q top-1, OMNI1 vs S5, on text-only inputs; count of new failure classes on bank v1. | Expected ≈ 1.0: media blocks write an exact zero on rows they do not own (m, code) [CHT]; bands are masked. | Below about 0.99, or loops appearing on lanes where S5 had none → the install leaks into the text path. Recommend diagnosing S7 before D1. |
| **G2 text start point** | Bank v1 content lanes, column A, artifact-tolerant (`a`); R-Q top-1 vs parent. | S3 trunk A = 8, parent A = 15 (m) [SD/brief]. S5 R-Q vs parent 110/210 = 0.524 (m) [VER]; the trunk-vs-parent noise floor is 108/210 (m) [VER]. | `a ≥ 6` → recommend GO. `3 ≤ a ≤ 5` → first run S8-lite on the clone: K3 down columns zeroed and media interface rows zeroed, bank v1 on two surfaces, ≈ 2.7 h (d: 2 × 15 × 1,024 tok / 3.2 tok/s). Zeroing the K3 columns is **not** the S3 trunk, because the trunk's down columns were refit jointly with K3 [VER]; the S3 trunk dossier is read beside it. If S8-lite recovers ≥ 2 lanes, the install caused the loss: recommend back to S5/S7. Otherwise GO. `a ≤ 2` → recommend NO-GO pending Cyril's read. |
| **G3 K3** | Bank v1 and R-Q with the K3 columns on and zeroed. The S0 per-route K3-vs-Q38 bits-per-byte probe ([CF] graft) informs λ_K3 and the route mix. **It never skips K3.** | K3 adds no measurable held-out text benefit at the fold stage (m) [VER]. | K3 is distilled in every outcome (owner ruling). Proposed λ_K3 start 0.3 (D-R5-6 [R5], Cyril's). If K3-on costs ≥ 2 content lanes, the proposal is: start D2 with the K3 gain frozen at 1.0 and λ_K3 = 0.1, unfreeze at the 12M rung if bank v1 has not regressed. Cyril rules. |
| **G4 per media modality** (H3, QIMG, HY3D, ACE, QVL32), each lane independent | S8 decoded a **structurally valid** artifact: `audio_rms > 0`, non-constant pixels, mesh faces > 0, frames > 0, vision answers present, and QVL32 features non-zero (image tokens reach the block). | Noise content is the expected S8 state (m) [S7R]. | Valid but noisy → GO for that lane. That is what D3 exists to fix. Undecodable → that lane returns to the S7 install fix and the others proceed. **H3** also needs the D12 licence flip **and** the H3 US-licence stop cleared, otherwise it is skipped and recorded. **HY3D** also needs PREP-B (the time token) fixed, or a recorded ruling to run without it ([CF]/[J2] graft). |
| **G5 budget** (from S0: **proposals, each needing a recorded ruling**; nothing is cut automatically) | S0 probe receipts, packaged with the S8 read. | — | The S0 package carries one proposal per item, each answered by a recorded ruling before the stage it touches: (1) **engine**: split cell (planning) or streamed bf16 if tools (i) and (ii) pass parity and P2 has a rate; (2) **D-R5-1** [R5, "Cyril ratifies"]: if the chosen engine is < 100 tok/s, or P7 shows fp16 skips persisting past warm-up (81 tok/s), the options are LoRA on the last 24 layers or fewer D2 tokens; (3) **K3 budget, D-R5-6**: 2.0M tokens in four passes (≈ 20.6 h) or 0.5M in one (≈ 5.1 h), priced from P3; (4) **H3 target count** if P4 > 10 s/forward (e.g. 1,024 instead of 2,048); (5) **vision-in visual-token cap** from P10's measured image sizes; (6) **on-policy sampler** (D-R5-3, R3; §4 D2). Until a ruling is recorded, the plan carries the §11 planning numbers. |

**Promotion between stages.** Each stage starts from the previous stage's clone only if all three hold against its own pre-stage clone:
- bank v1 content lanes are ≥;
- bank v2 vision and media content is ≥;
- Cyril has read the package.

Otherwise the next stage starts from the earlier clone [CP R2], and nothing is lost.

**Stop-spending line** ([CF] graft): a media modality whose decodes are still noise after 25% of its D3 budget is paused and reported, not continued.

---

## 3. Teacher-output cache plan (stage C, sealed, content-addressed, resumable)

Every teacher output is written once to `teacher_cache/<teacher>/<corpus_sha>/` as shards with a `manifest.json`. The manifest records:
- the teacher's integrity check before and after (a full directory digest for every teacher except K3; for K3 see its row);
- the corpus sha;
- the shard read order;
- a finite check per shard.

No teacher runs inside a training loop ([PF] graft). The one exception is on-policy scoring in D2, which is itself written as a cache. Eval splits and every frozen-bank, probe and smoke item are carved out by the contamination screen **before** any cache is built. The fold evidence corpus is in-sample for the charts and is never used for held claims [VER]. Every text-logit cache stores rows [0, 248,320) only.

| Cache | Parent and engine | Content | Size (d) | Disk | Card | Wall (arithmetic) |
|---|---|---|---|---|---|---|
| Q-text | Qwen3.8-27B-U bf16, layer-streamed mb8 [TPS]; one streamer only while K3 runs (host-RAM rule below). The vision and video rows need **new tool (i)**, the streamed VL teacher (`model.visual` → `get_rope_index` → `inputs_embeds`), because the existing engine builds a `Qwen3_5TextModel` only and drops `model.visual` [TPP] | Top-64 bf16 logprobs + int32 ids + lse, 384 B/token, stored at text positions only [CAMP §4.2]. Compute tokens: 20M registry text + 2.0M K3-row anchors + vision-in (1.6M text + **1.7M-8.9M visual**, P10 measures; the 1.6M was a text-only count [WALK]) + 1.15M video-in (incl. 8 × 256 patch tokens per clip) = **26.45M-33.65M** | stored 23.7M × 384 B = 9.10 GB (20 + 2.0 + 1.6 + 0.10 text-position tokens) | root | GPU0 | 26.45M-33.65M / 696.19 tok/s (m) [TPS] = **10.6-13.4 h** on one card. The vision-tower rate is assumed equal to the text rate (a) |
| Q-long | same, mb1 | 1.0M tokens at L = 32,768: 31 synthetic documents of concatenated same-route rows plus needle and retrieval tasks. The registry has 0 rows ≥ 32,768 (m) [WALK] | 0.38 GB | root | GPU0 | 1.0M / 171.05 = 5,846 s = **1.62 h**. 171.05 tok/s was measured at L = 4,096 mb1 [TPS]; its use at 32k is (a) |
| Self-gen text | Qwen3.8-27B-U Q4_K_M GGUF on llama-server, one per P40, no P2P. Sampling only; logits are rescored in bf16. The GGUF is the **parent's**, not a campaign checkpoint; D-R5-2 [R5] proposes Q8 for sampling, which does not fit a 24 GB P40 (≈ 29 GB), so the sampler quant is a ruling (U7) | 1.0M sampled tokens | < 10 MB ids. Transient: parent bf16 GGUF ≈ 55 GB + Q4 ≈ 17 GB on **root** (never under `/mnt`, where R6 makes any file permanent [CP R6]); both deleted after sampling | root | GPU1-3 | Export (per-tensor convert + quantize) 0.5-1 h CPU (a) [CF], done in S0 for P5. Sampling 1.0M / (3 × 11.84 tok/s, the P40 proxy decode @4k (m) [EXP]) = 28,153 s = **7.8 h** |
| Self-gen rescoring | Qwen bf16 streamed | Top-64 on the 1.0M self-gen tokens | 0.38 GB | root | GPU0 | 1.0M / 696.19 = 1,436 s = **0.4 h**, after the sampling finishes |
| K3 | Kimi K3, teacher-forced, one layer resident with packed experts on device [R6] | Top-32 ids + logprobs + lse, 196 B/token, over 2.0M dual-tokenized tokens (boundary_map), run as **4 passes of ≤ 0.5M tokens** | 0.39 GB. Host state per pass: `prefix_sum` + up to 8 AttnRes block snapshots (`attn_res_block_size` 12 over 93 layers) + the working `h` = 10 × 0.5M × 7,168 × 2 B = **71.7 GB** (d) [R6 a.4, RUN S2]. At 2M tokens in one pass it would be ≈ 287 GB, more than the box | root | GPU6 | Per pass per layer: read 16.99 GB / 241-308 MB/s (m) [R6] = 55-70 s; dequant 30 s (a); compute 0.5M × 18 experts × 33.03M × 2 FLOP = 5.95e14 at 6 TFLOPS (a, bf16 on sm75 with no bf16 tensor cores [J1]) = 99 s → 184-199 s × 93 × 4 passes = 68,448-74,028 s = **19.0-20.6 h** (d). If Cyril rules 0.5M total (D-R5-6): one pass, **≈ 5.1 h**. Integrity: per-shard sha256 from the fold receipt + size/mtime before and after, plus one full sha256 hashed in line during pass 1 (no extra read). A standalone full digest would cost 1.4-1.8 h per read of the 1.56 TB copy |
| K3 route probe ([CF] graft) | same pass | Bits-per-byte per route vs the Q-text cache on the same byte spans | inside K3 | root | GPU6 | inside the pass. **Informs λ_K3 and the route mix only** |
| H3 velocity | H3 fl2va DiT, bf16 or a guarded fp16 cast | (x0, seed, σ, guided v) at 512 clips × 4 σ = 2,048 targets, 2 s @24 fps 320×512 ≈ 2,020 latent tokens [MF]; plus block-25 in/out pairs | ≤ 60 GB total for all media caches (a) [MF] | root | GPU6 after K3 | 2,048 × 2 forwards (CFG) × 2.9 s (d [MF]: 8.6e13 FLOP at 30 TFLOPS (a)) = 11,878 s = **3.3 h**. **Only after D12 + the US-licence stop are cleared** |
| QIMG text-encoder embeddings | `qwen_2.5_vl_7b_fp8_scaled`, 9,384,670,680 B (m) [R6] | Prompt + negative embeddings for the 2,048 targets, precomputed in their own pass | < 1 GB (a) | root | GPU4 | ≈ **0.2 h** (a) |
| QIMG velocity | Qwen-Image **FP8 E4M3 storage with a per-layer cast to bf16** (ComfyUI `manual_cast` path). A full bf16 upcast is 20.4B × 2 B ≈ 40.9 GB and does not fit a 3090. The targets are FP8-model predictions, which is recorded [CF] | 2,048 targets at 512² ≈ 1,024 latent tokens, guided | (in the 60 GB) | root | GPU4; GPU6 after ViT if P4 shows the 20.43 GB FP8 weights + activations exceed 22 GB | 2,048 × 2 × ≈ 2 s (a [CF]: 44 TFLOP at 30 TFLOPS) = 8,192 s = **2.3 h** |
| HY3D trajectories | Hunyuan3D 2.1 fp32 | 256 shapes × 30 steps = 7,680 (x_t, σ, guided v) pairs from curated object images, plus ShapeVAE reference meshes | (in the 60 GB) | root | GPU1-3 after self-gen | 7,680 × 2 (CFG) × 5.4 s (d [MF]: 2.7e13 FLOP at 5 TFLOPS fp32 (a)) / 3 P40s = 27,648 s = **7.7 h** |
| ACE DiT + LM | ACE-Step 1.5 DiT; LM `qwen_1.7b_ace15` | DiT: 1,024 clips × 4 σ guided v on Oobleck latents. LM: top-64 over the 64k audio codes on 2M self-generated code tokens | (in the 60 GB) + 0.77 GB | root | GPU6 after H3 | DiT 0.35 h × 2 for CFG = 0.7 h (d from [MF] (a)). LM self-gen 0.6 h + scoring 0.4 h [MF] (a). ≈ **1.7 h** |
| QVL32 ViT features | Qwen3-VL-32B ViT (block 13 in/out) from `qwen3vl_32b_minimax_h3_bf16` | Feature pairs for the QVL32 unit | < 5 GB (a) | root | GPU6 | < **1 h** (a) [MF] |
| Vision-/video-in answers | The parent's own `model.visual` tower in front of the streamed LLM (tool (i)) | Inside Q-text | (inside Q-text) | root | GPU0 | inside Q-text. Used by D3d, not D2 |

**Schedule and host-RAM rule** (fixes [J1] flaw 3 and [REV A5]):
- Stage C runs **after S0 and before D1**: GPU6 runs K3, then H3, then ACE, then ViT; the P40s run self-gen, then HY3D; GPU0 runs Q-text, Q-long and self-gen rescoring; GPU4 runs the QIMG text-encoder pass, then QIMG.
- K3 host footprint per pass: 71.7 GB state + 25-58 GB RSS = 97-130 GB (d; the fold pass peaked at a 94.5 GB cgroup at T = 65,536 (m) [R6/brief]). P3 records RSS with all 8 AttnRes blocks resident before this binds.
- One Qwen streamer measured 56.1 GB RSS (m) [TPS]. K3 + one streamer = 153-186 GB, inside the 226 GiB (242.7 GB) rule. K3 + two streamers would reach 242 GB, at the line, so **only one Qwen streamer runs beside K3**. If P3 or monitoring shows more, the Qwen caches run first on both 3090s (≈ 7.5 h) and K3 runs after them: C becomes 7.5 + 20.6 = **28.1 h**.
- **K3 must finish before D1 starts**, because D1 pins 84 GB of host buffers.
- The tails of the HY3D, H3, ACE and ViT passes may overlap D1, since they use the P40s and GPU6 with small host RAM. In the planning path GPU6 must be free when D2 starts; its tail (≈ 6 h) ends inside D1 (11.4 h).

**Stage C critical path:** ≈ **20.6 h** (K3, d). GPU0 chain 12.6-15.5 h; GPU4 2.5 h; P40 chain 15.5 h; GPU6 tail 20.6 + 3.3 + 1.7 + 1 = 26.6 h, which overruns into D1 harmlessly. If Cyril rules a 0.5M K3 budget, C ≈ 15.5 h (GPU0 chain) or ≈ 10 h with both 3090s streaming once K3 has finished.

**Root disk for caches:**
- text ≈ 10.3 GB: Q-text 9.10 + Q-long 0.38 + self-gen rescoring 0.38 + D2 on-policy rescoring (1.15M × 384 B) 0.44, all (d);
- K3 0.4 GB;
- media ≤ 60 GB (a);
- parent GGUFs 72 GB transient, deleted after self-gen ends (C + 7.8 h), before the GPU6 media caches start (C + 20.6 h).

**Free root must be re-read after S5 exits and before C** (§10 U1).

---

## 4. Stages

Common to every training stage:
- Pre-registration `PREREGISTRATION_OMNI_STAGE_D_v1.md`, whose sha is pinned in every `training-receipt.json` (`stage2_training_receipt.py`, reused verbatim [CAMP §1.8]).
- A rolling safety checkpoint every 30 min with optimizer state, keeping 2, on root.
- All CPU jobs nice 15. GPU5 is never touched.
- The 1.3× overhead on training walls is an assumption [MF].
- P2P is used nowhere: the P40s corrupt silently, and GPU0↔GPU4 P2P is untested [brief]. All reductions are host-staged.
- Every text loss (D1 diagnostic KL, D2, D3d, D4) and every on-policy sample is computed over vocab rows **[0, 248,320)** only. Rows [248,320, 252,416) are the ledger bands and [252,416, 317,952) the C18 music band (64,000 written rows) [RUN]/[S7R]; S8 masks them with `--ledger-band-logit-mask`, and a full-vocab KL would push them down through the hidden states.

**Engines and planning rates:**
- **Planning engine:** the measured split cell, RTX 8000 + 3090 fp16 LoRA (GPU6 + GPU4), **151.24 tok/s (m)** at L = 2,048 [TPS]; the 4,096 shape was never measured. Its 6 of 13 skipped steps were exactly 6 halvings from 65,536 down to the final scale 1,024, i.e. GradScaler warm-up [J1 reading of the receipt]. S0 P7 confirms this at lr = 0. 81 tok/s (151 × 7/13) applies only if skips persist, and then it goes to Cyril as a D-R5-1 proposal (G5).
- **Upside engine:** streamed bf16 fwd+bwd, **174 tok/s per 3090 (d)** = 696.19 forward-only (m) [TPS] / 4 (fwd, recompute, 2× bwd (a)) [MF]. **It does not exist yet.** The existing streamed engine loads each layer with `load_state_dict(..., assign=False)` into two reused buffers, so the weights are gone before any backward, and `--mode` has no streamed-train option [TPP]. New tool (ii) adds per-layer boundary checkpoints with weights reloaded in backward; its parity arm checks gradients against a full load on Qwen3.5-0.8B, in the style of `stream-verify` [TPS]. Only after that passes does S0 P2 measure a rate.
- **Media hybrid module** (D3): qwen3_5 layers + the Chimera unit at depth d + interface rows + flow head. New tool work, parity-checked in S0/C (risk #6). Its rates are the [MF] derived rates, scaled from whichever text engine is chosen.

**Ledger id pattern** [CP R1/R2]: `OC02-D-Q38-27BU-OMNI1D<n>` for the delta, then `OMNI1D<n>.clone-pre-D<n+1>` for its byte clone. The final stage is `OC02-D-Q38-27BU-OMNI1D4` (merged) → `OMNI1D4.clone-pre-T1`. Where each lives is §4.1.

Delta registration needs a new `oc_checkpoint.py --delta-base` mode: adapters + trained subsets + the pinned base digest. **This is an owner decision and new tool work** ([CF]/[PF]/[J1]/[J2] graft).

| Stage | Goal | Teacher signal | Trainable params | Data | Card + memory budget | Wall-clock (arithmetic) | Checkpoint / clone / ledger | Success measure (capability; fit numbers are diagnostics) |
|---|---|---|---|---|---|---|---|---|
| **S0 probes** (no weight changes; after S8 delivery, pre-go) | Turn every derived rate into a receipt and build the missing engines. **P1** zero-copy name/slice-mapping reader: OMNI1 Chimera → qwen3_5 layout, tensor-exact, uniform FFN 16,640, no gain tensor, no materialised copy (the Chimera FFN is one fused `[gate\|up]` tensor [R6]). **T(i)** streamed VL teacher + parity arm vs a full load. **T(ii)** streamed fwd+bwd + gradient parity arm on Qwen3.5-0.8B. **P2** T(ii) rate cell, L4096 mb8 and L32768 mb1, one 3090, **lr = 0** (fwd + bwd + scaler check), only after T(ii) passes. **P3** K3 pure forward, 2 layers at T = 65,536, read/dequant/compute split out, **RSS with all 8 AttnRes blocks resident**. **P4** s/forward and peak VRAM: H3 bf16 vs fp16 cast (GPU6); QIMG FP8 + per-layer cast on a 3090, plus its text encoder; HY3D; ACE DiT and LM; ViT. **P5** llama-server Q4_K_M of the parent on a P40, `-np 4` (needs the parent GGUF export, done here on root). **P6** NAS write MB/s, measured on the clone writes themselves; no test file is written under `/mnt`, where R6 would keep it forever. **P7** 100-step GradScaler check at `init_scale 2^10` on the split cell **with OMNI1's K3 term present**, **lr = 0** ([CF]/[J2] graft; it reads static overflow, not overflow after weights move). **P8** header read of `qwen3vl_32b_minimax_h3_bf16`: is there an lm_head and how many layers ([CF] graft). **P9** dry-run tensor map of the per-tensor GGUF converter on the P1 reader. **P10** measured byte sizes of the media sidecars **and image dimensions → visual-token counts**. **P11** OMNI1 256k needle baseline: teacher-forced needle read at 32k / 128k / 256k × 2 depths with `run_omni_frozen_review.py --stream --teacher-forced` [RUN] extended with exact sequence-chunked FFNs and projections (a stock layer at L = 262,144 materialises 262,144 × 16,640 × 2 B ≈ 8.7 GB per FFN tensor, ≈ 17.4 GB for the fused gate\|up), with a parity arm vs the unchunked path at 4k | None. Parents are opened read-only and checked before and after | **0 updated.** P2 and P7 run fwd + bwd at lr = 0 and keep only receipts | Random ids (P2, P7); R-Q 210 (P1); 65,536 dual-tokenized tokens (P3); 2 clips, 2 images, 1 image-to-3D, 2 music clips (P4); needle documents (P11) | P1: CPU + GPU0, ≤ 18 GB (measured streamed peak 17.7 GB [TPS]). P2: GPU0, cap 22 GB. P3: GPU6, 16.6-17.0 GB layer (m) [R6] + 0.94 GB hidden. P4: GPU6 H3 40.2 GB of 46; QIMG on GPU4; P40s fp32. P7: GPU6 + GPU4 at 37.9 + 21.1 GB (m) [TPS]. P11: GPU0 + GPU4 | ≈ **7 h**, probes parallel across cards. P1 2-3 h CPU (a) [MF]. P3 ≤ 2 × 1,512 s = 0.84 h upper bound (fold-pass rate (m) [R6]). P11: 2 × 32,768 / 171.05 + 2 × (131,072 + 262,144) / 43 (a, quarter rate) = 18,672 s = 5.2 h on one card → **2.6 h** on two. T(i)/T(ii) parity arms ≤ 1 h each. Tool build time is not counted | No checkpoint. Receipts go under `$OC/distill/tps/` and become new TPS_TABLE rows | Every probe has a receipt with tok/s or s/forward, peak VRAM per card and host RSS. P1 is tensor-exact. T(i), T(ii) and P11 pass their parity arms or are reported failed. All budgets are recomputed from these receipts and handed to Cyril as G5 proposals |
| **C caches** (pre-go) | §3 | §3 | 0 | §3 | §3 | ≈ **20.6 h** critical (K3) | Sealed cache manifests, not checkpoints. Their shas go into the D receipts | Manifests finite; parent integrity checks equal before and after |
| **D1 per-layer trunk recovery** | Win back the coreset loss: per-layer held relL2 median 0.556, worst L39 0.674 (m) [brief/S5]. Train each layer's 4,352-wide trunk FFN to reproduce the parent layer output from **the child's own prefix**. The K3 columns stay frozen **and outside the loss** | Qwen3.8-27B-U layer i. Target y_p(i) = f_p,i(y_p(i−1)). Prediction = attention_c,i + FFN_trunk,c,i on h_c(i), **without the K3 term**: the K3 term is exactly the child's deviation from the parent (held KL to parent 2.62 vs 1.98 for the trunk [VER]), so including it would teach the trunk −0.03·TERM_k3. Loss = per-token relL2 + (1 − cos). The child's prefix advance keeps the K3 term. K3-term energy relative to the trunk FFN output is reported before and after D1. Frozen-head logit KL vs the parent after L63, rows [0, 248,320), is a diagnostic only | Trunk FFN gate/up/down 64 × 3 × 5,120 × 4,352 = 4,278,190,080 (matches the receipt [brief]) + post-attn RMSNorm 64 × 5,120 = 327,680 → **4,278,517,760**, one layer at a time. Frozen: attention/KDA, K3 columns, embed, head, media | First 4M tokens of the C registry text set, packed 4,096 | **GPU0 + GPU4 data-parallel** within each layer (host-reduced grads). Per card ≈ 7 GB (d) [MF]: frozen layer 0.61 + trainable state 66.85M × 18 B = 1.20 + parent layer 0.76 + mb4×4096 activations ≈ 4 GB. Host: two pinned 4M × 5,120 × 2 B = 41.9 GB buffers = 84 GB. **No K3 pass overlaps** | Per layer: parent fwd 4M / (2 × 44,544) = 44.9 s (44,544 = 64 × 696.19 (d)); train 2 epochs = 8M / (2 × 14,848) = 269.4 s (14,848 = 44,544 / 3 (a)); child advance 44.9 s → 359.2 s × 64 = 22,989 s = 6.39 h × 1.3 = **8.3 h**. If DP fails: 16.6 h on one card [MF]. Seal: delta 8.56 GB (Lexar) + clone to the U9 location 0.36 h (13.1 MB/s (a)); review bank v1 × 2 surfaces 2.7 h → ≈ 3.1 h. **D1 ≈ 11.4 h** | Rung file per layer (64 × 0.134 GB = 8.6 GB root, released once the delta is registered) → delta `OC02-D-Q38-27BU-OMNI1D1` (parent `OMNI1.clone-pre-D1`) → `OMNI1D1.clone-pre-D2` → R7 package READ-PARENT / READ-OMNI1 / READ-D1 | Bank v1 content lanes (both surfaces, artifact-tolerant) ≥ OMNI1's S8 count; R-Q top-1 vs parent ≥ S8 |
| **D2 end-to-end LoRA KD, text + K3** | Fit the text stack to the teachers' distributions; pull K3 knowledge in through its own branch; teach EOS and non-repetition; keep 256k behaviour. Vision-in and video-in moved to D3d | All over rows [0, 248,320). (1) Q38 top-64 forward KL + tail bucket, w 1.0. (2) K3 top-32 KL on aligned boundaries only: K3→Qwen by identical byte string, unmatched mass to the tail, λ_K3 per G3 (proposed start 0.3) with a 0.5/0.5 Q38 anchor on K3 rows ([CF] graft). (3) 32k long rung. (4) LM-CE floor 0.1 + **EOS supervision + unlikelihood on repeated 3-grams** ([CF] graft; training aids, not grading). (5) **On-policy GKD**, proposed fraction 0.25 after 20M off-policy tokens (D-R5-3 [R5], Cyril's), only with a ruled sampler: (i) a Q4 GGUF proxy of the student as a recorded exception to R3 (Q4 exactly once), (ii) a Q8_0 proxy (≈ 29 GB, GPU6 only, so upside path only), or (iii) fraction 0, the 1.15M tokens then coming from the off-policy registry | LoRA r64 on trunk FFN gate/up/down: 64 × 64 × 3 × (5,120 + 4,352) = **116,391,936** (d). LoRA r64 on KDA/GDN in and out and gated-GQA v/o: **≤ 123,731,968** (d: the measured 400,556,032 site set [TPS] minus its FFN share at width 17,408, 64 × 64 × 3 × 22,528 = 276,824,064; that remainder still includes the q/k sites D2 freezes, so it is an upper bound). K3 down-proj LoRA r32: 64 × 32 × (12,288 + 5,120) = 35,651,584. K3 per-column gain 64 × 12,288 = 786,432 fp32, **init 1.0** (the 0.03 is already in the down values [VER]). **Total ≤ 276,561,920 ≈ 277M; P2 prints the exact count.** Frozen: **GQA q/k, RoPE, KDA decay**, embed, head, all media tensors | 18.85M registry off-policy + 1.15M on-policy (or off-policy, per the ruling) + 1.0M self-gen + 2.0M K3 rows = **23.0M short**, + 1.0M long. Rungs pre-registered at 5M / 12M / 20M / 23.0M; the long rung runs last | **Planning:** split cell on GPU6 + GPU4, base fp16, host RSS 52.1 GB (m) [TPS]. **Upside:** GPU0 + GPU4 bf16 DP, each streaming one layer from a shared page cache of OMNI1 (on Lexar) + the D1 delta; per card ≈ 18 GB (d), anchored on the measured 17.7 GB mb8 forward [TPS]; head chunked to the 248,320 text rows. Host ≈ page cache 54 + boundary activations 2 × 21.5 + Adam 277M × 12 B = 3.3 + grads 1.1 ≈ **101 GB** | **Planning:** 23.0M / 151.24 = 42.24 h; long 1.0M / 75.6 (a: half rate at 32k) = 3.67 h; 45.91 × 1.3 = **59.7 h**. **Upside:** 23.0M / 348 = 18.36 h; long 1.0M / 86 (a: quarter rate at 32k) = 3.23 h; 21.59 × 1.3 = **28.1 h**. On-policy (only option i), off the critical path: per-tensor export of the composed base + D1 + D2-5M rung to GGUF with no merged HF copy (bf16 ≈ 54 GB + Q4 ≈ 17 GB, transient on root; 1.5 h CPU (a)), sample 1.15M / (3 × 11.84) = 9.0 h on P40s, score 1.15M / 696.19 = 0.46 h on GPU0. Seal: delta ≤ 1.11 GB + clone 0.1 h; review bank v1 2.7 h; bank v2 text/vision regression read ≈ 1 h (a); needle read with the P11 engine 2.6 h (a) → 6.4 h. **D2 ≈ 66.1 h planning, ≈ 34.5 h upside** | Adapter rungs ≈ 1.1 GB at 5M/12M → delta `OMNI1D2` (LoRA + gains, base digests pinned) → `OMNI1D2.clone-pre-D3` → package with the long-context table | Bank v1 content ≥ D1; R-Q vs parent ≥ D1; bank v2 vision lanes not regressed; 256k needle read ≥ OMNI1's P11 baseline (same engine, same needles); K3-columns-zeroed ablation reported both ways (not equal to the S3 trunk). Free-run match/lcp on the 64-prompt probe every 1M tokens = diagnostic |
| **D3 media + vision-in distillation** | Turn the five charted blocks into working generators inside the D2 trunk, and teach vision-in / video-in. **D3a** per-unit refit on host rows recaptured on the D2 child (replacing the stale S3-prefix rows [brief]). **D3b** guided-velocity KD, backprop only through layers d..63 above each unit, x_d cached. **D3c** C18 music-code band by KL to the ACE LM. **D3d** vision-in and video-in answer KD through the QVL32 unit + C11 rows (moved from D2: charted blocks write only on rows they own and are Chimera-only [CHT]) | C caches: H3, QIMG, HY3D, ACE DiT and LM, ViT features, and the vision/video answer caches inside Q-text (§3). D3d text losses over rows [0, 248,320) | D3a: ACE 102,811,904 + H3 267,889,176 + QIMG 154,985,727 + HY3D 46,156,031 + QVL32 19,594,291 = **591,437,129** [brief]. D3b: + interface-out rows C15 f[0,640) ≈ 3.3M + LoRA r16 on d..63 ≈ 86M [MF] (d). D3c: band rows [252,416, 317,952) in embed + head, 2 × 65,536 × 5,120 = 671,088,640, sparse-row lazy Adam on host + LoRA r16 all layers. D3d: QVL32 unit 19,594,291 + C11 rows 1,536 × 5,120 = 7,864,320 + the shared LoRA r16. Frozen: text vocab, layers below each unit (D3b), interface-in rows during D3b | Curated sidecars: video 1,336 (512 clips), image 4,228 (2,048 + 3D conditioning), music 2,768 (1,024 clips) [WALK]; audio/tts/speech paired into the H3 audio channel; 220 + 128 local proof rows. D3d: 1.6M text + 1.7M-8.9M visual + 1.15M video = **4.45M-11.65M** tokens (P10 pins it). **LTX-rollout v9_base rows are excluded** until their manifests are inspected; bank v2 media prompts excluded | D3a: one 3090, the largest unit is H3 at 267.9M × 18 B = 4.8 GB + rows ≈ 7 GB. D3b: GPU0 runs ACE then HY3D; GPU4 runs QIMG then H3. ≤ 22 GB per card: streamed d..63 ≈ 8 GB + unit with optimizer ≤ 4.8 GB + head < 1 GB. **x_d recaptured in chunks of ≤ 20 GB per card** (H3 x_d = 2,100 × 5,120 × 2 B = 21.5 MB/sample [MF]). D3c and D3d: full depth (the C11 rows sit at the input), both 3090s streamed; band optimizer 671M × 12 B = 8.1 GB host | **Upside:** recapture 11.1M tok / 1,392 tok/s (d [MF]) = 1.1 h; D3a ≈ 2 h (a) [MF]; D3b at 330 tok/s/card (d [MF], ≈ 34/64 depth): GPU0 = ACE 0.32M + HY3D 1,024 × 4,160 = 4.26M → 4.58M / 330 = 3.86 h; GPU4 = QIMG 2.2M + H3 4.3M = 6.5M / 330 = **5.47 h** (3.86 h governs if H3 is skipped); D3c 2M / 348 = 1.6 h; D3d 4.45M-11.65M / 348 = 3.55-9.30 h. (1.1 + 2 + 5.47 + 1.6 + 3.55…9.30) × 1.3 = **17.8-25.3 h**. **Planning:** D3b 11.08M / 284.7 (151.24 × 64/34 (a)) = 10.81 h; D3c 2M / 151.24 = 3.67 h; D3d 8.17-21.40 h → (1.1 + 2 + 10.81 + 3.67 + 8.17…21.40) × 1.3 = **33.5-50.7 h**. Seal: deltas 0.11 h, decodes 1 h, text bank v1 2.7 h, bank v2 vision 1 h, side-by-side assembly 0.5 h (a) → 5.3 h. **D3 ≈ 23.1-30.6 h upside, 38.8-56.0 h planning** | Per-modality rung (0.04-0.54 GB) → delta `OMNI1D3` (≈ 3.5 GB) with each unit's ratios carried (R2b) → `OMNI1D3.clone-pre-D4` | Decoded artifacts per modality **beside the parent's output on the same prompt** for Cyril's blind 1-5 ([CF] graft); content no worse than OMNI1's S8 on any modality; bank v2 vision cards answer image content; video-in answers present; text bank v1 ≥ D2. Structural checks: rms, pixels, faces, frames. Velocity MSE/cos/PSNR = diagnostics |
| **D4 consolidation + handoff** ([CF] graft) | Undo cross-stage forgetting (text ↔ media ↔ K3 on shared LoRA); materialise the ONE dense merged checkpoint; prove export-readiness | Replay from the existing caches only: 3M Q38 text (incl. 0.5M long), 1M K3-row, and 300 pairs each for vision, ACE, QIMG, HY3D and H3. Text losses over rows [0, 248,320) | All D2 + D3 trainables at LR × 0.3. Then an exact fp32 merge of the LoRA into the trunk, and the K3 per-column gains folded into the K3 down columns **once, here, in fp32, cast once** (the only fold point) | Replay slices; the eval split and banks stay unseen | Engine as in D2/D3. Merge: CPU streaming from Lexar base + 4 deltas, write to root (≥ 60 GB free; c16 `--min-free-gib 60`). H checks: GPU6 Chimera load; a P40 load of the truncated text GGUF | **Upside:** text 3M + K3 1M = 4M / 348 = 3.2 h; media 300 × (1,074 + 4,160 + 2,100 + 313) = 2.29M / 660 = 0.96 h; 4.16 × 1.3 = 5.4 h. **Planning:** 4M / 151.24 = 7.35 h; 2.29M / 284.7 = 2.23 h; × 1.3 = 12.5 h. Merge ≈ 3 h (a [CF]). Seal under U9 option a: register on root 0.15 h + clone to the NAS subtree 55.1e9 / 13.1e6 = 1.17 h write + 1.17 h digest + `verify --deep` 1.17 h ≈ 3.7 h (option b ≈ 0.5 h). Full S8-equivalent review 6-8 h [RUN via CF] → 7 h; 256k needle read 2.6 h (a); H (convert map, truncated GGUF, Chimera load) 2 h (a). **D4 ≈ 23.7 h upside, ≈ 30.8 h planning** | `OC02-D-Q38-27BU-OMNI1D4` (merged, 55.1 GB on root, lineage through D1-D3, `oc_package.py build/verify`) → `OMNI1D4.clone-pre-T1` (byte copy at the U9 location, digest equal, chmod a-w) | All side by side (parent / OMNI1 / D4): bank v1/v2, R-Q, vision cards n/5, video-in, media artifacts, K3 lane, 256k needle read vs P11. Cyril decides whether stage T starts |

### 4.1 Disk arithmetic, per disk, permanent vs transient (fixes [J1]/[J2] flaw 1 and [REV A1/A2])

**Free space** [SRV], read while S5 (≈ 54 GB, on root) was still being written: root 157 GB, NAS 345 GB, Lexar 95 GB (≈ 40 GB after OMNI1). **All three are re-read with `df` after S5 exits** (U1).

**What the registry allows** [OCK]/[CP R6]: `register` and `clone` refuse any directory under `/mnt` except `/mnt/lexar`; `/mnt/ai_data` (the NAS) is refused for both. R6 also says nothing under `/mnt` is ever deleted, so anything written to the NAS is permanent. Transient files therefore never go to the NAS.

**Permanent set** (registered; clones are never deleted while a descendant exists [CP R2]):

| Item | Size (d) | Where |
|---|---|---|
| `OMNI1.clone-pre-D1` (step 1, S9) | 55.1 GB | U9 |
| D1 delta + clone | 2 × 8.56 = 17.12 GB | Lexar if the post-OMNI1 `df` shows ≥ 40 GB; else with the clones |
| D2 delta + clone | 2 × ≤ 1.11 = 2.22 GB | same |
| D3 delta + clone | 2 × 3.5 = 7.0 GB | same |
| D4 replay delta + clone | 2 × ≈ 4.6 = 9.2 GB | same |
| D4 merged | 55.1 GB | root |
| `OMNI1D4.clone-pre-T1` | 55.1 GB | U9 |
| **Total** | **≈ 201 GB** (165.3 full-size + 35.5 deltas) | |

**U9, Cyril picks one before S9** (step 1 needs it for `clone-pre-D1`):
- **(a)** Add the exact subtree `/mnt/ai_data/omni-campaign/clones` to `REGISTRABLE_UNDER_PROTECTED`, with a test (delete stays refused under `/mnt`). NAS then holds 55.1 + 55.1 + the delta clones ≈ 128 GB of 345 GB. Cost: CIFS digests at ≈ 12.5 MiB/s, about 70 min per 55 GB, twice per clone, plus `verify --deep` (priced in D4 and in S9: ≈ 3.5 h per full-size clone).
- **(b)** Free root by archiving the S3/S5/sweep working directories to the NAS as unregistered mirrors (permanent there under R6). Root must then hold ≈ 185 GB at the D4 peak (165.3 full-size + 10.3 text caches + ≈ 9 safety checkpoints).

**Transient set** (root only, deleted after use):

| Item | Size (d) | Lifetime |
|---|---|---|
| Parent bf16 GGUF + Q4 for self-gen | 55 + 17 = 72 GB | S0 → end of self-gen sampling (C + 7.8 h) |
| Text caches + K3 cache | 10.3 + 0.4 GB | C → D4 |
| Media caches incl. QIMG TE embeddings | ≤ 60 GB (a) | C → D3; archived before the D4 merge |
| D1 rung files | 8.6 GB | D1 → delta registration |
| D3 x_d chunks | ≤ 40 GB | D3 |
| On-policy student GGUFs (only if Cyril rules option i) | 54 + 17 = 71 GB | D2 5M rung → end of sampling |
| Rolling safety checkpoints (2 kept, LoRA + Adam) | ≈ 9 GB | per stage |
| P1 projection, merged D1/D2 copies | 0 | never written: P1 is a zero-copy reader; the engines compose base + deltas per layer; GGUF conversion is per tensor |

**Root peaks, option (a)** (d): C ≈ 83 GB (GGUFs + text caches); D2 ≈ 80 GB, or ≈ 151 GB with the on-policy GGUFs of option (i), at the edge of the 157 GB read; D3 ≈ 120 GB (caches 71 + x_d 40 + safety 9); D4 ≈ 75 GB (D4 merged 55 + text caches 10 + safety 9). Option (b) adds the 55.1 GB `clone-pre-D1` from S9 onward and the second 55.1 GB clone at D4.

**Without delta registration:** 9 full copies × 55.1 = 496 GB, which no disk here holds. Then Cyril must free NAS archive space and pick (a), or allow hardlink clones ([CP] open decision 6; ext4 root only, since the NAS is CIFS and Lexar is exFAT; R2 says a hardlink is not a pre-training clone).

---

## 5. Per modality

### 5.1 Text + K3

**Starting state:**
- S5 R-Q top-1 vs trunk 181/210; vs parent 110/210 = 0.524.
- Held NLL 3.432 vs trunk 3.756 vs parent 4.290. These are in-sample for the charts (m) [VER].
- The K3 charted term is ≈ 18.5× the trunk MLP output, attenuated ≈ 33× by the 0.03 scale (m) [VER]. The 0.03 is not a runtime scalar: c10 refit the whole down projection `[h × (4,352 + 12,288)]` against `Y = FFN_trunk + 0.03·TERM_k3` with the prior `[trunk_down | 0.03·prior_k3]` [VER]/[C10].
- K3 end-of-stack held top-1 0.692; NLL 2.157 vs teacher 1.261 (m) [K3R].

**Path:**
- D1 recovers the trunk FFN per layer against the parent with the K3 term outside the loss and the K3 columns frozen, so K3 survives D1 intact.
- In D2, K3 enters through three things together: its own down-projection LoRA, a per-column gain initialised at 1.0 that can lift the attenuation, and cross-tokenizer KL on aligned boundaries anchored 0.5/0.5 to Q38. The gain is folded into the K3 down columns once, at the D4 merge.
- The K3 route probe informs λ_K3 and the routes; K3 is never skipped (owner ruling, [J2]).
- Free-run decode from K3 is infeasible: 92 × 14.8 GB packed experts = 1.37 TB > RAM (m) [R6]. So K3 is a teacher-forced logit teacher only.

**Grading:** bank v1/v2 content lanes, R-Q vs parent, and the K3-columns-zeroed ablation read both ways. Zeroing the K3 columns does not give back the S3 trunk (its down columns were refit too), so the S3 trunk dossier is read beside it.

### 5.2 MiniMax H3 (video + audio out)

**Gates:** D12 licence flip **and** the H3 US-licence stop must both be cleared. If not, H3 is skipped and recorded, and the block stays at its fold state.

**Teacher:** the bf16 DiT, 40,225,724,176 B, fits only GPU6 (m) [brief/R6]. An fp16 cast with a finite guard is tried first (P4). TE embeddings are cached separately, because the TE (51.5 GB) and the DiT do not fit GPU6 together [CF].

**Known caps, carried and named:**
- `h3_thw_v1` positions have never executed;
- text is read through an AV-only chart;
- the video VAE round trip is **9.91 dB** (m) [brief]. That floor caps any AV result, so the C15 video band is re-measured on real curated clips first [CF].

**Lane:** last in the order. 2,048 targets (or the count Cyril rules from P4), 3.3 h of teacher time, 4.3M D3b tokens.

### 5.3 Qwen-Image (image out)

**Teacher:** the FP8 E4M3 distribution, 20,430,635,136 B (m) [brief], kept in FP8 storage with a per-layer bf16 cast; a full bf16 upcast (≈ 40.9 GB) does not fit a 3090. Its text encoder `qwen_2.5_vl_7b_fp8_scaled` (9,384,670,680 B (m) [R6]) runs in its own pass first, so the guided targets have cached prompt and negative embeddings. The targets are FP8-model predictions, and that is recorded. GPU6 is the fallback if P4 shows a 3090 cannot hold it.

**Starting state:** the whole-block fit is 0.686 (m) [brief].

**Lane:** D3 lane 2. 2,048 guided targets at 512²; 2.2M D3b tokens.

**Decision line:** if the first 25% of the budget still decodes to noise in Cyril's read, the lane is paused and reported [CF].

### 5.4 Hunyuan3D 2.1 (3D out)

**Teacher:** the 7,365,943,290 B checkpoint + ShapeVAE (m) [brief], fp32 on the P40s.

**Distillation data:** the teacher's own sampling trajectories, because no mesh corpus exists.

**Conditioning:** attn2 is dropped by D3, so the student receives the conditioning image through its vision-in path.

**Prerequisite:** PREP-B (the time token) or a recorded ruling. The current held forward cos is 0.405 (m) [brief].

**Lane:** D3 lane 3. 7,680 pairs cached; 4.26M D3b tokens. Graded on GLB + 4 views beside the parent; Chamfer is a diagnostic.

### 5.5 ACE-Step 1.5 (music out)

**Lane:** first, because it is the cheapest teacher.

**Teachers:**
- the DiT, 4,787,825,604 B, for guided velocity on Oobleck latents;
- the LM `qwen_1.7b_ace15`, 3,708,523,360 B (m) [brief], a plain causal LM over 64k codes. It is the KL teacher for the C18 band (D3c).

**Starting state:** C15 audio 23.09 dB vs a 32.78 dB VAE round trip; C18 text-fit R² ≈ 3%, held cos 0.139/0.204 (m) [brief]. 64,000 of the 65,536 band rows are written (ACE codes [151,669, 215,669) → band rows [252,416, 316,416)) (m) [S7R].

**Grading:** music.wav beside the parent's for Cyril's blind 1-5, plus `audio_rms`.

### 5.6 Qwen3-VL-32B (vision in)

**Teachers:**
- answer KD from the trunk parent's own `model.visual` tower (same tokenizer, same 5,120 width), cached in C by tool (i) and trained in **D3d** on the hybrid module (the QVL32 unit and the C11 rows are Chimera-only, so a qwen3_5-layout student cannot host them);
- QVL32 feature KD against ViT block 13 in D3a, at weight 0.1.

**Token count:** the 1.6M vision-in figure counted text only [WALK]. The image + OCR rows (≈ 6.8k [WALK raw routes]) at 256-1,300 visual tokens each add 1.7M-8.9M tokens. P10 measures the image sizes; a visual-token cap is a G5 proposal.

**Open question:** S0 P8 reads the header of `qwen3vl_32b_minimax_h3_bf16`. At ≈ 25.7B params it may be a truncated text encoder with no lm_head [CF]. If it is a full VLM, its answers become a second answer teacher at weight 0.3. Otherwise it is a ViT feature source only.

**Starting state:** QVL32 cos 0.363; C11 rows at only 2%/17% of the type-embed norm (m) [brief].

### 5.7 Video-in and audio-in (gap the judges named, [J2])

**Video-in:** D3d adds 1.15M tokens of answer KD: 512 clips × (8 frames × 256 patch tokens + about 200 text) = 1,151,488 (d). The parent sees the frames through its own vision tower (tool (i)), and the student sees them frame by frame through the QVL32 unit + C11 rows.

**Audio-in:** **none of the seven sources is an audio-understanding teacher.** ACE and H3 generate audio; the H3 TE is vision-language. Two options for Cyril:
- (a) Supervised audio-in in stage T, through the C15 audio interface-in rows. Data: speech 1,378, audio 1,784 and sound 598 registry rows [WALK], and the LibriSpeech 267 / LJSpeech 257 local transcripts [brief].
- (b) Add an audio-understanding teacher as an eighth source. **That is his decision, and it is not planned here.**

---

## 6. 256k context plan

**Facts:**
- OMNI1 is 48 KDA layers (constant-size state) + 16 gated-GQA layers. This is derived from full_attention_interval 4 [brief].
- The Qwen3.6-27B proxy, with the same 3:1 layout, hit **12/12 needles to 260,712 tokens** on a 3090 with q4_0 KV, at an own peak of 21,522 MiB (m) [EXP §3.3.3].
- The curated set has 0 rows ≥ 32,768 tokens (m) [WALK].
- S8 runs the bank v2 long-context lanes at 8k / 32k / 64k only ("64k lane ~1 h") [RUN S8]. The bank also holds `lc-200k` lanes [BANK], but no step runs them. **So S8 produces no 256k number**; S0 P11 creates the baseline.
- The only measured streamed rates are at L = 4,096 (171.05 mb1, 696.19 mb8) [TPS]. A stock layer at L = 262,144 materialises ≈ 8.7 GB per FFN tensor (≈ 17.4 GB fused gate|up) and runs out of memory on a 3090.
- No card here can train at 256k.

So this plan protects long context rather than training it.

1. **Frozen in every D stage:** GQA q_proj/k_proj, RoPE/partial-RoPE parameters and the KDA decay parameters. LoRA touches only GQA v/o, the KDA in/out projections and the FFNs [MF].
2. **D1** changes only per-token FFN maps, which carry no position dependence.
3. **D2 32k rung:** 1.0M tokens, KL to parent logits computed at L = 32,768 over rows [0, 248,320). Teacher 1.62 h on one card (a: the 4k rate). Student 3.23 h at an assumed 86 tok/s on 2 cards (upside) or 3.67 h at 75.6 tok/s (planning).
4. **Needle reads** with the P11 engine (teacher-forced, exact sequence-chunked FFNs and projections): OMNI1 in S0 (the **baseline**), then the D2 and D4 seals. 32k / 128k / 256k × 2 depths = 851,968 tokens: 65,536 / 171.05 + 786,432 / 43 (a) = 5.2 h on one card, **2.6 h on two**.
5. **Final word:** `envelope_bench.py` on the single Q4 after stage T: `-c 262144 -fa on -ctk q4_0 -ctv q4_0 -np 1` on a 3090. 5 points to 260,712 tokens, 12 needles, about 3,005.6 s per sweep (m on the proxy) [CF/EXP]. Reported beside the P11 baseline and a P11-engine read of the stage-T bf16 child; Cyril reads it (free-run Q4 and teacher-forced bf16 are different measurements).
6. K3's native 1M-position capability is **not claimed**. K3 enters only through position-agnostic FFN columns.

---

## 7. Hand-off to the curated-dataset finetune (stage T)

**Input:** `OMNI1D4.clone-pre-T1`, a merged dense bf16 Chimera checkpoint (K3 gain folded at the D4 merge) with its qwen3_5-layout reader proven in S0 P1 and the D4 H checks.

**Data:** the curated master registry:
- 631 entries / 94,318 rows / 640 shards; 42,145,686 tokens over 50 routes (m) [WALK];
- `ready_untrained` 52,416. The "trained_*" labels refer to the dead omnicoder, so every row is untrained for Chimera [brief].

**Preparation:**
- Copy the needed rows to local disk first. The text payload is 539,954,956 B: at 12.5 MiB/s (m) that is ≈ 41 s. Media sidecar bytes are measured in S0 P10.
- Carve out the eval split and the banks once, and reuse the exact carve-out from D.
- Note for Cyril: the same rows were used **teacher-forced** in D and are used **with ground truth** in T. Held claims use only the carved split.

**Training plan:**
- Mix curated : distill-replay 3:1, ≤ 2 epochs, with a KL regularizer to the D4 checkpoint (D-R5-9 [R5]; Cyril's values). Text losses over rows [0, 248,320); band rows only through their own media lanes.
- Same engine as D2. 2 epochs of 42.1M tokens = 84.3M tokens: planning 84.3M / 151.24 × 1.3 = 201 h; upside 84.3M / 348 × 1.3 = 87.5 h. This is not counted in §11.
- Stage T also owns: audio-in (§5.7a), tts/speech, the tool/code routes, and the 151 rows ≥ 4k / 2 rows ≥ 8k of real long-context data [WALK].

**Output:** T1 → `.clone-pre-Q4` → the Q4 export (§8). It is registered per R2/R3.

---

## 8. Q4 export: one artifact, after stage T, for one 3090 and the P40s

Quantization runs **exactly once**, on the sealed stage-T child, never on a D checkpoint. The result is its own `.Q4` ledger row [CP R3]. **One** Q4_K_M text artifact serves both the 3090 and the P40s; there is no second quant. The only possible exception is the on-policy sampling proxy in D2 option (i), which exists only if Cyril rules it as a recorded R3 exception and is never registered as an export.

### Text / long-context: llama.cpp GGUF, qwen3_5 architecture

**Layout:** 64 layers, uniform FFN 16,640 (4,352 trunk + 12,288 K3; the K3 gain was folded at the D4 merge, so there is nothing to fold here), untied head, **vocab truncated to the 248,320 text rows**. The ledger and C18 bands live in the torch package below; llama-server has no band mask short of a 69,632-entry `logit_bias`.

**Steps:**
1. Per-tensor conversion through the P1 reader to a bf16 GGUF on root (≈ 52 GB transient; the final export GGUF is a ledger artifact and is never deleted once registered [CP R6]).
2. `llama-imatrix` over 2M screened curated tokens. This needs the separate-root imatrix build, so that the R3-joined llama-server sha does not change (export gap 2 [EXP/CF]).
3. `llama-quantize` to Q4_K_M.

**Size arithmetic** (d; S5 header 26,944,218,624 params [brief]):

| Item | Arithmetic | Size |
|---|---|---|
| Embed + head (truncated) | 2 × 248,320 × 5,120 = 2,542,796,800 params, at Q6_K 6.5625 bpw | 2.09 GB |
| Body | 26,944,218,624 − 2 × 317,952 × 5,120 = 23,688,390,144 params, at Q4_K_M ≈ 4.85 bpw (a, typical) | 14.36 GB |
| **Q4_K_M total** | | **≈ 16.45 GB = 15,686 MiB** |

Truncation saves 69,632 × 5,120 × 2 × 6.5625 / 8 ≈ 0.58 GB against the full-vocab file, and makes the logits buffer equal to the proxy's.

**Fit on one 3090 at 262,144 with q4_0/q4_0 KV:**
- Proxy own peak 21,522 MiB with 15,736 MiB of weights (m) [EXP §3.3.3, CF], so KV + compute = 5,786 MiB.
- OMNI1-T: 15,686 + 5,786 = **21,472 MiB of 24,576**, leaving ≈ 3,100 MiB of headroom (d).
- Same attention geometry assumed (a). FFN 16,640 < 17,408, so the compute buffers do not grow.

**Fit on a P40 at 262,144 with q4_0 KV:**
- The proxy loaded and ran at **21,409 / 24,576 MiB** on GPU1 (m) [EXP §4.1].
- OMNI1-T: 21,409 − 15,736 + 15,686 = **21,359 MiB**, about 3,200 MiB of headroom (d).
- The P40 q4_0 needle sweep was 3 of 4 points at last record [EXP]. With q8_0 KV it loads only up to 233,472 tokens (m) [EXP].
- A 256k prefill on a P40 takes hours: 44-66 tok/s (m) [CF/EXP]. The 3090 is the 256k serving card, and the P40s serve shorter contexts.

### Media and vision-in: "Chimera-Q4" torch package

llama.cpp cannot host the mid-stack charted blocks, the bridge/head or the VAEs (export gap 4 [CF/EXP]). The package uses **the same Q4 weight grid** ([CF]): trunk linears are read from the one GGUF.

**Dequantization:** `gguf-py` dequantizes in CPU numpy; dequantizing a 16 GB trunk on every denoise step is not workable. The package needs a **CUDA K-quant dequant kernel** (or the ggml-cuda backend through a binding), per layer on the fly. This is new tool work.

**Weights:** GGUF grid ≈ 16.45 GB + media blocks 591M × 2 B = 1.18 GB + band rows (embed + head) 69,632 × 5,120 × 2 × 2 B = 1.43 GB + bridge/head interface ≈ 0.18 GB, bf16 → **≈ 19.2 GB** on a 3090 (d). Media runs at ≤ 16k context, so KV is small.

**VAE decoders:** fp16, loaded one at a time **after the trunk is moved to host**. Sizes: H3 video 5.2 GB, audio 0.6, QIMG 0.25, ACE 0.34, ShapeVAE 0.66 [brief]. The H3 video VAE does not fit beside the trunk (19.2 + 5.2 = 24.4 GB), hence the swap or a tiled decode.

**P40 media:** unmeasured, and the owner's spec says the export fits the P40s, so it is **not dropped here**; it is a decision for Cyril (U7). Pascal (sm61) has no bf16 arithmetic: the bf16 media blocks and band rows would run as fp32 (2.37 + 2.86 GB) or as a guarded fp16 cast, and the K-quant dequant kernel must be built for sm61.

**Grading:**
- bank v1/v2 vs the HF bf16 T child;
- Cyril's blind media 1-5, with seconds per clip recorded;
- the 256k envelope with needles;
- decode tok/s on a 3090 and a P40;
- KV type recorded, because q4_0 KV moved 10 of 15 answers in the 0.8B parity test [MF/EXP].

**Ledger:** GGUF digest, the torch-package digest and the exact llama-server command [CP R3].

---

## 9. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | The streamed bf16 fwd+bwd engine does not exist; its 174 tok/s/card is derived; nothing > 230M has been trained here [MF]/[TPP] | Planning numbers use the measured 151.24 split cell. T(ii) + gradient parity arm + P2 before the streamed path is proposed. D-R5-1 proposal to Cyril if the chosen engine is < 100 tok/s |
| 2 | fp16 on Turing with OMNI1's K3 term (18.5× pre-attenuation) may overflow beyond warm-up [J1]; P7 at lr = 0 sees only static overflow | P7 runs on OMNI1 itself; the planning split cell logs skip counts from step 1 of D2 and pauses for a ruling if they persist |
| 3 | K3 pure-forward cost unmeasured. The fold pass was 1,288-2,348 s/layer including fit [R6]. Attention-residual state is ≈ 143 GB per 1M tokens | S0 P3 with all 8 blocks resident; ≤ 0.5M tokens per pass; budget 2.0M (4 passes) or 0.5M is Cyril's (D-R5-6). K3 is still distilled at any size |
| 4 | Cross-tokenizer KL may transfer little (mass lost on code/math tokens; no fold-stage held benefit [VER]) | Route probe informs λ and routes; Q38 anchor; K3-zeroed ablation reported both ways; never a skip |
| 5 | qwen3_5 projection of OMNI1 not tensor-exact → no training path exists [brief] | S0 P1 is G0. If it fails, D stops and Cyril decides: build the Chimera-native path or not |
| 6 | D3 needs a new hybrid module (qwen3_5 layers + unit at depth d + C11/C15 rows + flow head) and it now also hosts vision-in/video-in; the S7 contract defects carry into it | Build and parity-check it in S0/C. Per-lane G4 prerequisites; stop-spending line |
| 7 | H3 licence (D12, US stop); H3 nearly fills GPU6 | Lane last and skippable; fp16 guard with bf16 fallback |
| 8 | Disk: registry refuses NAS clones; root cannot hold the ≈ 165 GB full-size set plus caches; NAS dropped 766 → 345 GB between 10-08 and 10-12 [SRV]; delta mode is new tool work | U9 ruling before S9; deltas on Lexar; transients on root only; `df` after S5 and before C, D2 and D4 |
| 9 | Host RAM: K3 pass (97-130 GB) vs D1 (84 GB) vs streamers (56 GB each); other tenants (fold workflow, Codex pid 4169670) | K3 serialized into C with one streamer beside it; 226 GiB rule; tenant claims checked via the relay before C/D1 |
| 10 | Shared LoRA + media training regresses text | Per-stage bank v1 reads; promotion rule; D4 replay |
| 11 | Media capacity: one charted block per source may not reach DiT-quality output | Report honestly; Cyril decides on more capacity. Type-gated adapters are **not** planned, because they are not dense [J2] |
| 12 | Contamination: the registry is used in D (teacher-forced) and in T (ground truth); the fold corpus is in-sample | Carve-out before C; held claims only on the carved split |
| 13 | Media teacher s/forward all FLOP-derived; could be 2-5× off [MF] | S0 P4 before any budget binds; G5 proposals |
| 14 | Q4 headroom (≈ 3.1 GB on a 3090, ≈ 3.2 GB on a P40) is proxy-derived | Measured envelope on the real artifact; a different quant would be a new R3 ruling, not a silent second export |
| 15 | On-policy sampling has no R3-clean sampler (a Q4 proxy breaks "Q4 exactly once") | Cyril rules option (i), (ii) or (iii) (D-R5-3); rescoring always in bf16 |
| 16 | Shared-box contention slowed K3 1.7× during fold [CF] | 1.3× overhead already in; contended walls are reported, not hidden |
| 17 | Vision-in tokens unmeasured (1.7M-8.9M visual); tool (i) does not exist | P10 + G5 cap proposal; T(i) parity arm; planning uses the upper bound |
| 18 | The 256k engine (P11) is new: sequence chunking must stay exact | Parity arm vs the unchunked path at 4k; rates at 128k/256k are (a) until P11 runs |
| 19 | Qwen-Image FP8 + per-layer cast may still exceed a 3090 | P4 measures peak; GPU6 fallback |

---

## 10. Assumptions and unknowns to resolve first

**Resolve before C or D1:**

| # | Unknown | How it is resolved |
|---|---|---|
| U1 | Root, NAS and Lexar free space after S5 and OMNI1 finish (157 / 345 / ≈ 40 GB were read mid-write [SRV]) | One read-only `df` after S5 exits and before C; again before D2 and D4. If root is short: media caches move off root only by archiving S3/S5 working dirs per the cleanup rules (NAS writes are permanent) |
| U2 | **Owner decision:** `oc_checkpoint --delta-base` mode | Cyril. If declined, see §4.1 "without delta registration" |
| U4 | S0 P1-P11 rates, T(i), T(ii) parity | Every "(a)/(d)" rate above |
| U5 | P8: is `qwen3vl_32b_minimax_h3_bf16` a full VLM | Header read |
| U6 | Media sidecar resolvability, byte sizes, image dimensions; LTX-rollout rows | S0 P10 + per-entry manifest inspection |
| U7 | Cyril's rulings | D12 (H3 licence) + the H3 US-licence stop; D3 (HY3D attn2 drop) + PREP-B; D-R5-1 (rate floor and its cut); D-R5-2 (parent sampler quant: Q8 proposed, does not fit a P40; Q4_K_M planned); D-R5-3 (on-policy fraction and sampler, §4 D2 options i-iii); D-R5-6 (K3 token budget and λ_K3); H3 target count; vision-in visual-token cap; audio-in option (§5.7); **P40 media** in the export (§8) |
| U8 | GPU4/GPU6 and host RAM released by the fold workflow and the Codex k3_capacity process (pid 4169670, owner unconfirmed) | Claims confirmed through the relay |
| U9 | **Owner decision, needed by step 1 (S9):** where the clone set lives. (a) registrable NAS subtree `/mnt/ai_data/omni-campaign/clones` + test, or (b) free root by archiving S3/S5/sweep working dirs | Cyril, before S9. §4.1 prices both |
| U10 | Runbook S8 row names GPU5 for VAE decodes | Corrected to GPU1-3 before S8 launches (§2) |

**Assumptions carried** (all marked (a) above):
- OMNI1 is assembled as specified: 27,535,655,753 params, 64 layers (48 KDA + 16 gated-GQA), hidden 5,120, vocab 317,952 (text 248,320 + ledger bands 4,096 + music band 65,536), untied head, on Lexar.
- Training multipliers: 3× forward for D1, 4× for streamed fwd+bwd.
- 1.3× overhead.
- K3: 6 TFLOPS, a 30 s dequant per layer per pass, 10 resident [T, 7,168] bf16 state tensors.
- Media teachers: 30 TFLOPS on RTX 8000/3090, 5 TFLOPS fp32 on the P40.
- NAS write = 13.1 MB/s; Lexar write fast enough that delta seals stay inside the priced 0.36 h.
- The parent vision tower runs at the text-streamer rate.
- Streamed rates at L = 32,768: 171.05 tok/s teacher (measured at 4k only); 43 tok/s/card student; 75.6 tok/s for the split cell.
- Needle reads at 128k/256k: 43 tok/s per card.
- The split cell at L = 4,096 runs at its L = 2,048 rate; D3b on the split cell runs at 151.24 × 64/34.
- Self-gen on P40s matches the proxy's 11.84 tok/s.
- The 3.2 tok/s HF bf16 review rate is from the brief; no local receipt was found.

**Out of scope for this document** (relayed user request): fixing the oversized Suricata log and pausing the agentic IT stack container are server writes. This design task was read-only, so they were not done here and belong to the orchestrating session.

---

## 11. Total wall-clock

Planning path = the measured split cell for every student stage, U9 option (a), and the upper bound of the vision-in token count. Upside path = the streamed bf16 engine after T(ii) passes and P2 measures it.

| Phase | Planning path | Upside path | Basis |
|---|---|---|---|
| S0 probes | 7 h | 7 h | §4 (P11 2.6 h on both 3090s; P1 2-3 h CPU) |
| C caches | 20.6 h (28.1 h if K3 and a streamer cannot share host RAM) | same | §3, K3 critical: 4 passes × 93 layers × 184-199 s |
| *Pre-go subtotal* | *27.6 h; hidden if Cyril's S8 read takes at least that long* | *27.6 h* | |
| D1 | 11.4 h (19.7 h single-card) | 11.4 h | 8.3 train + 3.1 seal |
| D2 (text + K3) | 66.1 h | 34.5 h | 59.7 / 28.1 train + 6.4 seal |
| D3 (media + vision-in) | 56.0 h (38.8 h at the vision lower bound) | 30.6 h (23.1 h) | 50.7 / 25.3 train + 5.3 seal |
| D4 | 30.8 h | 23.7 h | replay 12.5 / 5.4 + merge 3 + seal 3.7 + review 7 + needles 2.6 + H 2 |
| **After-go total** | **≈ 164.3 h ≈ 6.8 days** (147.1 h at the vision lower bound) | **≈ 100.2 h ≈ 4.2 days** (92.7 h) | |
| **Nominal incl. pre-go** | **≈ 191.9 h ≈ 8.0 days** | **≈ 127.8 h ≈ 5.3 days** | |
| U9 option (b) (clones on root) | −3.2 h (D4 seal 0.5 h) | −3.2 h | §4.1 |
| K3 budget 0.5M (if Cyril rules D-R5-6) | C ≈ 15.5 h (pre-go ≈ 22.5 h); D2 short tokens −1.5M (−3.6 h planning, −1.6 h upside, d with × 1.3) | same C | §3 |
| Persistent fp16 skips (81 tok/s) | D-R5-1 proposal to Cyril, not an automatic cut and not a longer run | — | [R5] |

Not counted:
- Cyril's review time at G0-G5 and after D1, D2, D3 and D4;
- step-1 extra for U9 option (a): writing `clone-pre-D1` to the NAS ≈ 3.5 h instead of ≈ 0.3 h on root [RUN S9];
- tool build time for P1, T(i), T(ii), P11, the hybrid module, the delta mode, the per-tensor GGUF converter and the CUDA K-quant dequant;
- stage T (≈ 201 h planning, ≈ 87.5 h upside, derived, §7);
- the final Q4: convert + imatrix ≈ 2 h (a), 256k envelope ≈ 3 h (a; 3,005.6 s measured sweep on the proxy), grading ≈ 3 h.

**GPU-hours (d), excluding review passes:**
- Planning path: RTX 8000 ≈ 150 (C: K3 20.6, H3 3.3, ACE/ViT 2.7; split-cell training D2 59.7 + D3 50.7 + D4 12.5; S0 ≈ 2). RTX 3090 ≈ 185 (GPU4 shares the 122.9 h of split-cell training; C: GPU0 chain 15.5 + QIMG 2.5; S0 ≈ 12; D1 2 × 8.3; needle reads 3 × 5.2).
- Upside path: RTX 3090 ≈ 180 (S0 12, C 18, D1 16.6, D2 2 × 28.1, D3 2 × 25.3, D4 2 × 5.4, needles 15.6). RTX 8000 ≈ 29 (C 26.6, S0 ≈ 2).
- P40: ≈ 48 (self-gen 3 × 7.8 = 23.4, HY3D 3 × 7.7 = 23.1, S0 P4/P5 ≈ 1), + 27.0 if Cyril rules on-policy option (i) (3 × 9.0).
- GPU5: 0.

---

## Appendix: Review disposition

Each finding of [REV] was checked against the files it cites. "Applied" means the change is in the body above.

| Finding | Disposition | Where / reason |
|---|---|---|
| A1 clones cannot go on the NAS; root cannot hold them | **Applied.** Confirmed in `oc_checkpoint.py` (`PROTECTED_PREFIXES`, `REGISTRABLE_UNDER_PROTECTED = ("/mnt/lexar",)`), CP R6 and runbook S9. My count of the permanent set is ≈ 201 GB (it includes the D4 replay delta pair) | §1, §4.1, U9, §11 |
| A2 transient writes missing | **Applied, with one change:** the review's "write straight to NAS scratch" is rejected because CP R6 makes anything under `/mnt` permanent; transients go to root. P1 is a zero-copy reader and GGUF conversion is per tensor | §3, §4.1, S0 P1/P9 |
| A3 streamed engine is forward-only, text-only | **Applied.** Confirmed in `tps_probe.py` (`Qwen3_5TextModel` only, `model.visual` set to None, `assign=False` into two reused buffers, no streamed-train mode). T(i) and T(ii) added; planning path moved to the split cell | §1, §4, S0, §11 |
| A4 D2 media trainables have no host on a qwen3_5 student | **Applied** for the QVL32 unit, the C11 rows and vision-/video-in KD (moved to D3d). **Rejected in part:** the claim that the qwen3_5 layout cannot cover the K3 columns is wrong, because `c10_union_install.py` lifts the K3 SiTU channels to SiLU into one FFN with `down [h x (Q + k3)]`, so a uniform 16,640-wide FFN holds them (P1 proves it); D-R5-5 does not say otherwise | §1, §4 D2/D3, §5.6, §5.7 |
| A5 K3 host RAM off by ≈ 10× | **Applied.** Confirmed by R6 a.4 and the runbook S2 row (`prefix_sum` + up to 8 snapshots, ≈ 8.5 GB per 65,536-token stream by layer 84). Four ≤ 0.5M passes, 19.0-20.6 h, P3 RSS with 8 blocks; 0.5M total offered under D-R5-6 | §3, §9, §11 |
| A6 Qwen-Image does not fit a 3090 as written | **Applied.** FP8 storage + per-layer cast, GPU6 fallback, text encoder `qwen_2.5_vl_7b_fp8_scaled` (9,384,670,680 B, R6) in its own pass and in P4 | §3, §5.3, S0 P4 |
| A7 256k gate has no baseline; check runs out of memory | **Applied.** Confirmed: S8 prices lc-8k/32k/64k only; `lc-200k` lanes exist in the bank unrun; 171.05 tok/s was measured at L = 4,096. Baseline placed in S0 (P11) rather than S8, because S8 is a step-1 runbook row | §6, S0 P11, D2/D4 |
| A8 U3 already answered; gain design wrong | **Applied.** VERDICT: target `Y = 1.0·FFN_trunk + 0.03·TERM_k3`, prior `[trunk_down \| 0.03·prior_k3]`, down refit. U3 removed; G0 has no gain tensor; gain init 1.0; one fold point (D4 merge); zeroed-K3 ablation described as not the S3 trunk | §1, §2 G0/G2/G3, §4, §5.1, §8 |
| B1 D1 trains the trunk to erase K3 | **Applied.** K3 term excluded from the D1 prediction, kept in the prefix advance, energy reported before and after | §1, §4 D1 |
| B2 D2 losses must mask the bands | **Applied.** All text losses and on-policy sampling over rows [0, 248,320); band layout confirmed in the runbook and S7_READINESS (64,000 written C18 rows) | §1, §3, §4, §5.5 |
| B3 K3 digest cost not counted | **Applied.** Receipt per-shard sha256 + size/mtime before/after, one full digest hashed in line on pass 1; standalone cost stated | §3 |
| B4 media Q4 package underspecified | **Applied.** CUDA K-quant dequant, sm61 bf16 handling, P40 media moved to U7 | §8, U7 |
| B5 vision-in tokens low by 2-6× | **Applied.** Walk token counts are text-only; image + OCR raw-route rows ≈ 6.8k; P10 measures dimensions; Q-text and D3d carry the 1.7M-8.9M range | §3, §4 D3, §5.6 |
| B6 D2 trainables ≈ 300M, not 464M | **Applied, refined:** FFN LoRA 116,391,936 confirmed; with QVL32/C11 moved to D3 the D2 upper bound is 276,561,920; delta ≤ 1.11 GB, Adam 3.3 GB | §4 D2 |
| C1 G5 makes Cyril's decisions | **Applied.** D-R5-1 is "[proposed] … Cyril ratifies" (R5) and "Cyril's to rule on" (TPR). G5 is now a proposal list with recorded rulings | §2 G5, §4 |
| C2 P2/P7 take optimizer steps pre-go | **Applied.** Both at lr = 0; S0 starts after the S8 package. Noted that lr = 0 measures static overflow only | §1, §2, S0, §9 #2 |
| C3 three quantizations vs R3 "exactly once" | **Applied.** One Q4_K_M; IQ4_XS removed; on-policy sampler is a ruling (options i-iii); text GGUF truncated to 248,320 rows (≈ 0.58 GB saved) | §4 D2, §8, U7 |
| C4 S8 uses GPU5 | **Applied.** Confirmed in the runbook S8 row; correction recorded as U10 | §2, U10 |
| "Checked and correct" list | No action; these numbers are kept | — |
| Not raised by the review, found while applying it | D-R5-2 proposes a Q8 parent sampler, which does not fit a P40; added to U7. The 5 GB NAS test write in P6 would be permanent under R6; P6 now measures the clone writes instead | §3, S0 P6, U7 |
