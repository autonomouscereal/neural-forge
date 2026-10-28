# Run 0 code survey: arch-v2 branch (2026-10-22)

Read-only scout. Scope: what the code on `claude/arch-v2-20261021` already does for run 0 (plant Qwen3.5-2B, ablate, then distill on all modalities on 4 P40s), and exactly what is missing.

**Where things are**
- **Worktree:** `C:/Users/cereal/Documents/ai training/.agent-worktrees/chimera-omni-5d4d4e1dcd/workers/claude/arch-v2-20261021`
  - HEAD is `8d8f064`. The tree is clean: no uncommitted work.
- **Server code copies:** `/mnt/lexar/omni-campaign/scratch/arch-v2-trainer/code/chimera-omni-<sha>/`
  - Copies exist for 35606ee, 65eccd8, d0ed6e1 and 8d8f064.
  - `$OC/chimera-omni-v2`, the path the lane queue expects, does **not** exist.

Evidence labels: **[V]** = I read or ran it this session. **[R]** = a report or receipt says so and I did not re-run it.

---

## 0. Bottom line

1. **The trunk import is close.**
   - The exact Qwen3.5 importer exists and fits the Qwen3.5 key layout. I checked the layout against the on-disk 4B: `model.language_model.*` (426 tensors), `model.visual.*` (297), `mtp.*` (15).
   - The co2-2b geometry is the Qwen3.5-2B text geometry.
   - **But Qwen3.5-2B is not on disk** [V]. Only 0.8B and 4B are in `/home/cereal/models`, plus 9B on the NAS.
   - **The trainer refuses any init that is not fold-built** (`V2_INIT_RULE`). A plant path has to be written.
2. **Ablation: nothing exists.** No shrink/perturb, re-init or noise tool is anywhere in `src/`.
3. **Marker tokens: the control band has 8 tokens and room for 504.**
   - Today a media span is `<|media|> header-text <|/hdr|> <|raw|>xN <|/media|>`. The modality is named only by the header text.
   - There are no per-modality marker tokens and no output-decision token.
   - The trainer cannot supervise a decision token on a generation sample: an x0 pass refuses causal text labels.
4. **"Real sizes, all modalities, KD, 4 P40s, days" is the largest gap.**
   - The trainer's only media source is the toy proxy npz. Its images are 64x64, which is **2x2 rows of 32 px: exactly the toy Cyril rejected**. Speech is 0.64 s windows and volumes are 16^3.
   - There is no real-media loader.
   - Generation batches have no token budget.
   - Media has no KD (REPA is not implemented).
   - The context is 4,096 tokens.
   - The two-card resume diverged in the last smoke run (an open bug).
5. **Measured 2B mechanics** [V, probe = random weights, lr 0, toy media]:

   | Setup | Time per step | Tokens per step | Throughput | Peak memory |
   |---|---|---|---|---|
   | 1 card | ~60 s | 9.1k | ~150 tok/s | 20.7 GB |
   | 2 cards | ~78 s | 18.2k | ~230 tok/s | 20.7 GB |

   - On 2 cards the host all-reduce takes 18-20 s per step.
   - With `--accum 4` on 4 cards I project roughly 35-47M tokens/day. This is arithmetic, not a measurement.

---

## 1. The co2-2b profile (`src/chimera_omni/config.py`)

**Builder:** `chimera_omni_v2_2b_config()`, registered as `V2_PROFILES["co2-2b"]`. Values [V]:

| Field | Value |
|---|---|
| layers | 24 = 18 GDN (`kda` with `kda_decay_kind="gdn"`) + 6 `gated_gqa`, pattern `(kda,kda,kda,gated_gqa)` |
| width / FFN | 2048 / 6144, dense SiLU-GLU |
| GQA | 8 q / 2 kv x 256 head dim; partial RoPE 64 dims, theta 1e7; mRoPE section (11,11,10), interleaved; QK-norm; sigmoid output gate |
| GDN | 16 v / 16 qk heads x 128; SiLU output gate; fla q/k norm; query scale 1/sqrt(128); no conv bias; conv kernel 4 |
| depth mixing | `block_attnres`, block size 4, `attnres_init="zero"` (uniform mix) |
| vocab | 248,320 text + 512 control = 248,832, tied, fixed-band ledger |
| raw boundary | `native_feature_dim=3072`, `native_type_conditioning="none"`, `native_unit_norm="rms"` (A6b), head GLU 2048 -> 4096 -> 3072 |
| refinement | `native_bidirectional_refinement=True` on `gated_gqa` only |
| | `native_reverse_media_scan=False` (A8 off) |
| | `native_x0_skip_rms=0` (skip off) |
| other | `mtp_heads=0`, `aux_heads=()`, `max_sequence_length=262144`, rms eps 1e-6 |
| metadata | `init_source` = "fold: K3 dense fold ...", `init_rule` = `V2_INIT_RULE` = "fold-built (no direct placement of any existing model's weights)" |

**Parameter count**
- 1,923,165,809 as built by the trainer [V, D3 log].
- The spec's 1,923,133,041 is 32,768 lower. The difference is the A6b `amplitude_proj` (16 x 2048).

**Spec drift to know about**
- **Video tubelets.** The spec section 4 table says a video row is 1 frame of 32x32. The code (`V2_VIDEO_TUBELET=2`, `V2_VIDEO_PATCH=(32,16)`) cuts 2 frames x 3 x 32 x 16 per row.
- **A8 and the x0 skip are both off** in the profile. They were "tried and not adopted" (spec 14.9), yet the spec section 5 still says "A8, required before S3".

---

## 2. The qwen35 importer (`src/chimera_omni/importing/qwen35.py` + `importing/profiles.py`)

**Function:** `load_qwen35(model_dir, cfg=None)`
- Reads `config.json` and builds `qwen35_chimera_config(hf_config)`, then `import_qwen35_state`.
- It does an exact rewrite of every text-trunk tensor [V, code read].

**What it maps**
- **Embedding:** `embed_tokens` is copied into rows `[0, 248320)`. The 512 band rows are zero-initialised.
- **LM head:** a tied head is checked to be tied. An untied `lm_head` is copied, and its band rows are zeroed.
- **Norms:** `(1 + w)` norms (input, post-attention, final, q/k norms) are folded into a plain weight in fp32.
- **MLP:** `gate_proj` and `up_proj` are concatenated into `ffn.up`; `down_proj` maps to `ffn.down`.
- **Linear attention (GDN):**
  - `in_proj_qkv` is split into q, k and v.
  - `conv1d` is split into three depthwise convolutions.
  - `in_proj_z` -> output gate, `in_proj_b` -> write strength, `in_proj_a` -> decay down.
  - `dt_bias`, `A_log` and `out_proj` map directly.
  - The shared `linear_attn.norm` row is broadcast to one row per head.
- **Full attention:**
  - The fused `q_proj` is de-interleaved into `q_proj` and `gate_proj` (sigmoid gate).
  - `k_proj`, `v_proj` and `o_proj` map directly.
- **Refusals** (raised as errors):
  - a scaled or yarn RoPE;
  - MoE;
  - an activation other than SiLU;
  - a non-sigmoid output gate, unless `allow_unverified_output_gate` is set;
  - any GDN switch that would not be exact.

**What it skips**
- `mtp.*` (one MTP layer plus fc and norms on the 4B) and `model.visual.*` (the ViT) are **not loaded**.
- Both are listed in the receipt with their shapes and hashes.
- `mtp_num_hidden_layers` is stamped into the metadata as dropped.
- Everything else on the student side keeps its random init and is listed in the receipt as "random init (no Qwen source)": the media bridge and head, and the AttnRes pseudo-queries and norms.

**Does it handle a Qwen3.5-2B checkpoint?** Structurally yes:
- the key prefixes and the `layer_types`/`rope_parameters` parsing match the on-disk 4B [V];
- the 2B is the same family.

Three blockers:

1. **The checkpoint is absent** [V]. The spec also says its licence is UNVERIFIED (ARCHITECTURE_V2 section 3).
2. **The importer refuses `residual_mode != "none"`**, and co2-2b is `block_attnres`.
   - Workaround: import into the Qwen-shaped config (residual "none"), then `load_state_dict(strict=False)` into a co2-2b model.
   - `tests/test_v2_architecture_torch_optional.py::test_uniform_attnres_init_reproduces_the_plain_residual_and_identity_does_not` pins this. The zero-init AttnRes model reproduces the plain residual to within 1e-5 x the logit scale, and the only missing keys are `*residual*` [V, code read; the test passes].
   - The missing keys in that path will be `attention_residual.*`, `output_attention_residual.*`, `native_media_bridge.*` and `native_media_head.*`.
   - The mRoPE section is exact for text, because t = h = w.
3. **The trainer refuses the result** (see section 6). `build_base_model` only accepts:
   - a fold receipt (`_fold_receipt_problem` rejects any copy or placement method);
   - a trainer checkpoint whose lineage is a fold;
   - `--init random` for proxy profiles only.

   `load_fold_base` also requires `metadata.init_rule == V2_INIT_RULE`.

**Exactness receipt tool:** `scripts/import_receipt_qwen35.py`
- It compares HF `Qwen3_5ForConditionalGeneration` with the import: max logit difference, argmax agreement and per-layer hidden states over a fixed prompt pack.
- I did not re-run it here.

---

## 3. Raw boundary, x0 head, control band, markers

**Boundary:** `NativeContinuousMediaBridge`, untyped branch (`modeling/chimera.py:27`).
- `feature_proj(raw / rms)` + `amplitude_proj(Fourier(log-rms))` + `position_proj(thwr)` + `position_fourier_proj(Fourier(thwr))` + `noise_proj(Fourier(sigma))`, then RMSNorm.
- The result is added onto each `<|raw|>` slot.
- Type ids are validated but never used in computation.

**x0 head:** `UnifiedNativeMediaEmissionHead`
- RMSNorm, then GLU MLP (2048 -> 4096 -> 3072). There is no type affine.
- `_x0_skip` (EDM-style c_skip/c_out) exists but is off in co2-2b.

**Control band:** `v2_layout.V2_CONTROL_TOKENS`, at ids `248320 + index` for co2-2b [V]:

| Id | Token |
|---|---|
| 248320 | `<|pad|>` |
| 248321 | `<|eos|>` |
| 248322 | `<|media|>` |
| 248323 | `<|/hdr|>` |
| 248324 | `<|/media|>` |
| 248325 | `<|raw|>` |
| 248326 | `<|user|>` |
| 248327 | `<|assistant|>` |

- Ids 248328-248831 (504 rows) are unused.
- **Modality is said only in the header text.** The proof headers are `"image 64x64"`, `"video 12.5fps 64x64 4f"`, `"audio 38400hz mono"` or the kind headers `"speech 38400hz mono"` / `"music 38400hz mono"`, `"volume 16x16x16 sdf pbr"`, plus `"draft 1/{f} "` for drafts.
- **Qwen's own special tokens** (ids 248044-248069) include `<|im_start|>`/`<|im_end|>`, `<|vision_start|>`/`<|vision_end|>`, `<|image_pad|>`, `<|video_pad|>`, `<tool_call>` and `<think>` [V, tokenizer.json]. There are no audio or 3D tokens.
- Qwen ids 248070-248319 are unused table rows.

**Chat-format inconsistency (real).**
- The teacher-cached text corpora are Qwen-chat-templated token ids (`<|im_start|>` and so on).
- The trainer's media samples use the band's `<|user|>` / `<|assistant|>` / `<|eos|>`.
- So the model learns two turn conventions.

---

## 4. Media unit formats, and whether a media set is refined jointly

**Formats** (`native_media.py`; `V2_*` constants) [V]:

| Modality | Unit (3,072 values) | Rows |
|---|---|---|
| Image / OCR | 32x32x3 raster patch, padded to a multiple of 32 | 512^2 = 256 rows, 1024^2 = 1,024 |
| Video | 2 frames x 3 x 32 x 16 tubelet at the clip's fps; positions on the physical clock | 448x256 at 12.5 fps ≈ 1,400 rows/s (spec) |
| Audio / music / TTS | 3,072 samples of one channel at 38.4 kHz = 80 ms | 12.5 rows/s per channel |
| 3D | 8^3 voxels x 6 channels (SDF truncated / R / G / B / roughness / metallic), from `sdf_volume_to_raw` | 64^3 = 512 rows, 128^3 = 4,096 |

- **Draft:** `native_draft(packet, f)` re-cuts the same media at 1/f resolution.
- **Positions:** `native_rope_grid` puts rows on (tick or z, row, column), interleaved with text in mRoPE.
- **Signal level:** one shared signal RMS (`V2_SIGNAL_RMS=0.5`) is reached by a per-dataset gain.

**Joint or unit by unit: jointly within one span, not across spans.**
- A generation sample ends in ONE open media span: `V2Sample.media(open_suffix=True)` must end the sequence.
- `_native_refinement_layout` makes the trailing contiguous run of media rows the refined suffix. Every row in it attends bidirectionally to every other row and to the prefix, in the 6 GQA layers.
- The 18 GDN layers stay causal (raster order) inside the span, because A8 (`native_reverse_media_scan`) is off in co2-2b.
- Sampling (`v2_refine.sample_suffix`) runs 16 Euler x0 steps over all rows of the span together, so it is "chunk output with refinement passes" at the level of the whole span.
- `draft_then_refine` chains coarse-to-fine passes: draft factors such as (4, 2), then 1.
- **Not supported in v2:**
  - joint refinement of two spans, for example video + its audio. Header tokens between spans break suffix contiguity, and `V2Sample` allows one open span.
  - a span that mixes modalities: `NativeMediaPacket` has one `kind`, and the header names one kind.
- The August AV proof (`proof/h3_hybrid_av_flow.py`) did joint AV by concatenating the video and audio rows into one suffix with joint per-type noise levels. It used the legacy typed K3 toy config, not v2.

---

## 5. Loss and objective code

**Text** (`training/v2_trainer.py chunked_text_objective`):
- The head runs on supervised positions only, in chunks of 256 positions, over the full 248,832-row softmax.
- Terms:
  - CE with weight 0.1 on cached text and 1.0 on media answers;
  - **top-K bucket forward KL** against each same-vocabulary teacher cache: q38-27bu and flash-next-strata-iq3s, both top-64 [V manifests], weight 1.0 each;
  - **K3 cross-tokenizer bucket KL**: `XtokTeacher`, weight 0.3;
  - unlikelihood (3-gram, 8 candidates, 0.1, kept inside one document).
- The bucket KL is exact over top-K plus one tail bucket. That is a top-n target, not top-1.
- **No top-n agreement metric is logged.** For example, student top-10 against teacher top-10 overlap: Cyril's stated metric.

**Media generation** (`media/training.py x0_media_objective`):
- Rectified flow with `x_t = (1 - sigma) x0 + sigma eps` and the type-free level schedule (`level_flow_noise_levels`).
- The model predicts **x0**. The loss is **x0-space MSE** with type-balanced reduction.
- Raw-audio multi-resolution STFT, mel, level and boundary terms are SNR-weighted toward low sigma.
- **There is no velocity-space loss weighting** (TOKENIZER_FREE_STUDY 3e lever), **no image perceptual/LPIPS term** and no REPA.
- Draft-then-refine is trained: with `--draft-factor 2`, half of the generation samples are a 1/2 draft and half are a refine pass with a noised true draft as context.

**Media understanding:** CE on the answer text after a clean raw span.

---

## 6. The T2 trainer (`src/chimera_omni/training/v2_trainer.py`, 1,648 lines)

**Entry points:** `python -m chimera_omni.training.v2_trainer`
- `memory-plan --profile`
- `train --profile --init DIR --tokenizer tok.json --data-config data.json --run-dir RUN --devices cuda:0,cuda:1,...`
- `probe` (random weights, lr 0, nothing saved)

**Data interfaces** (`chimera_omni.v2_trainer.data.v1` JSON)
- **`text`:** a list of `{name, corpus, teachers:{name: cache_dir}, k3:{xtok, boundary_map?, unverified_ok?}, weight, supervision}`.
  - `TokenCorpus` loads a safetensors corpus **fully into RAM** (`input_ids` [rows, len] plus lengths, target_mask and doc_index).
  - `TopKTeacherCache` memory-maps sealed `omni-distill-teacher-cache-manifest-v1` shards and checks the corpus sha and ids on every row read.
  - A teacher-supervised source uses only the rows its caches hold.
- **`media`:** only `{"kind": "proof_npz", "path": .../v2proxy_all_v1.npz}`, the toy proxy data (`ProofMediaSource`):
  - COCO crops at 96x64, 4 classes;
  - LibriSpeech 0.64 s windows;
  - MAESTRO and ACE band music;
  - ModelNet10 SDF at 16^3, 4 classes.

  There is **no other media kind**. A run with no media is refused ("every step must carry every modality").

**Mixing**
- MIXTURE_V1 token shares: text 45, image 20, video 8, speech 15, music 7, 3D 5.
- Every micro-step carries every modality in both directions: understanding plus generation, and edit.
- `MixtureAccountant` tops up the neediest modality.

**Losses:** see section 5. Each micro-step runs one understanding pass and one generation pass. The x0 pass refuses causal labels, so the two are separate forwards.

**Precision:** fp32 weights, gradients and compute. There is no autocast. That is correct for P40s: fp16 runs at 1/64 rate and there is no bf16. Teacher logprobs are stored in bf16 and read as float.

**Memory switches**
- Non-reentrant per-layer activation checkpointing (`use_gradient_checkpointing`).
- Chunked GDN scan, chunk 64, pure torch.
- SDPA for GQA. Whether the memory-efficient kernel is chosen on sm_61 with a boolean mask is **unverified**.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`.

**Optimizer memory**
- Default **Adafactor**: no momentum, factored row/column statistics of about 2 MB for the whole 2B, row-blocked updates for the 248,832 x 2,048 table.
- `adamw-offload`: 15.4 GB pinned host memory per replica, so about 62 GB for 4 ranks. Not compatible with Strata's RAM headroom.
- `adamw` on the card: refused for the 2B because it does not fit.

**Multi-GPU**
- One process per device, spawned. Gradients are averaged through host memory: device -> pinned buffer -> gloo all-reduce -> device. There is no P2P anywhere.
- Every rank builds **every** micro-batch (world x accum) on its main thread and trains its own share. Building is redundant per rank and does not scale to real decoding.
- Replicas are checked identical (state sha256) at every checkpoint.

**Checkpoint and resume**
- A checkpoint holds:
  - chimera safetensors fp32 (~7.7 GB for the 2B);
  - `optimizer.pt`;
  - every RNG state and the builder and mixture state;
  - `COMPLETE.json` with a sha256 for every file.
- Writes are atomic (temporary directory, then rename). `--keep 3`.
- Resume: `--resume auto` takes the newest complete checkpoint.
- Stopping: a `STOP` file, SIGTERM or SIGINT checkpoints and exits. `--stop-after N` stops after step N.
- `training_identity` includes `world` (the card count). **Changing the number of cards blocks resume**, even though the docstring says the cards may change.

**omni_lab**
- `--omni-lab-cmd server` runs `omni_lab.py log-run start / metric (every --lab-every steps, in the background) / end` through the intake venv.
- It also writes `STEP {json}` log lines and `steps.jsonl` for the ingester.
- It writes no `run_model` or `run_dataset` links.

**Not in the trainer**
- a plant or ablate init;
- marker or decision tokens;
- real-media loaders;
- token-budgeted generation batches;
- long context beyond `--seq-len` (default 4096; the text caches cover 4,096 positions, plus one `long-00000` corpus with a q38 cache);
- media KD and REPA;
- a top-n metric;
- periodic sample dumps;
- an exclusion guard on corpora;
- a K3-dominant weighting. The defaults are q38 1.0, strata 1.0, K3 0.3. The 2026-10-17 binding ruling says K3 is primary and Q38 is small. K3 xtok covered only 8 of 20 steps in smoke A [V].

### 6a. What the 2026-10-22 smoke runs showed

All runs are under `/mnt/lexar/omni-campaign/scratch/arch-v2-trainer/runs/`. Each attempt had pre-registered rules (omni_lab reviews 38296 / 38306 / 38321). Mechanics only; no behaviour claims.

**Attempt 1: `smoke-20261022T113351Z`, wrapper run 38297, GPU1.**
- **A** (38298): co2-fold-proxy from the c21 fold child, 20 steps, seq 2048. Finite. Peak 3.6 GB, about 5 s per step.
- **B** (38300 / 38301): resume worked.
- **C** (38302): two replicas on one card crashed in torch's non-reentrant checkpoint when run as threads.
- **D** (38303): the 2B probe ran out of memory in the Adafactor step. That was fixed in d0ed6e1.
- **E** (38304): the 2B probe with adamw-offload passed. 20.6 GB peak, host RSS 19.6 GB, optimizer 4-18 s.
- The graded rules failed:
  - R1, because C has no steps;
  - R5;
  - R6, because K3 was > 0 on only 8 of 20 steps.

**Attempt 2: `smoke2-...114923Z`, wrapper 38307, GPU1+2.**
- C, C2 and D passed:
  - D (38311) is the 2B on 1 card at 4,096: 59-61 s per step, 9.1k tokens, about 150 tok/s, peak 20.7 GB.
  - C2 is the threaded two-card proxy.
- D2 (38313), the threaded two-card 2B, crashed. That led to the process-per-card design in 8d8f064.

**Attempt 3: `smoke3-...120845Z`, wrapper 38322, GPU1+2.**
- **D3 (38325) PASS:** the 2B on two processes. Per step: 78-81 s, 18.2k tokens, about 230 tok/s, all-reduce 17.7-20.5 s, optimizer 0.6 s, peak 20.7 GB, host RSS 10.9 GB per rank.
- **C3a (38323):** the proxy stopped at step 3 and its checkpoint shas agreed.
- **C3b (38324) FAIL:** after the resume, the ranks diverged. The trainer refused the step-6 checkpoint with "replicas diverged (state sha256 per rank d5ec80f3… vs 161ce7c0…)", and the run ended with rc 2.
  - Graded R4 (resume) and R5 (identical replicas) **FAIL**.
  - **The root cause is not found, and no fix is committed after 8d8f064.**
  - The test `test_two_ranks_through_host_memory_take_the_step_of_one_rank_accumulating_both` also fails on CPU, on a tolerance: `blocks.1.attention_residual.norm.weight` max diff 5.7e-5, 1.6% of entries > 1e-5 [V].

### 6b. Test suite (server CPU, code copy 8d8f064, torch 2.6.0 Pascal venv) [V]

- 729 tests collected, **10 failed**. The rest passed or skipped; the final tally line was cut from my output tail.
- **Real failures:**
  - `test_chunked_gdn_scan_torch_optional::test_cached_decode_continues_exactly_after_a_chunked_prefill`: logit diff 2.19e-5 against a tolerance of 2e-5. fp32 noise.
  - `test_v2_trainer_torch_optional::test_two_ranks_through_host_memory_...`: see above.
- **Environment failures (8):** the `*_proof_artifacts` tests need `proof_outputs/` and the reports, which the server code copy does not include (FileNotFoundError).
- Every other v2 test passed: architecture, mixed sequence, mRoPE, all modalities, round 5 generation, trainer resume reproducibility, gradient checkpointing equivalence, every-modality micro-step, and the remaining tests.

---

## 7. MISSING, by run-0 step

### (1) Plant Qwen 2B weights

1. **Get the checkpoint.** `Qwen/Qwen3.5-2B` is not on disk. Download it to Lexar and check its licence. Then assert that `qwen35_chimera_config(config.json)` matches co2-2b field by field: layers, width, FFN, heads, head dims, rope, `tie_word_embeddings`. The 2B values are web-sourced only [W].
2. **Write a plant tool.** It does not exist. Steps:
   - `load_qwen35` (residual "none", fp32);
   - build `chimera_omni_v2_2b_config()` and `load_state_dict(strict=False)`;
   - assert the missing keys are exactly `*attention_residual*`, `native_media_bridge.*` and `native_media_head.*`, and that nothing is unexpected;
   - write a chimera checkpoint plus a plant receipt: source shas, consumed and skipped tensors (MTP and visual kept as side records), the logit check against HF on the prompt pack, and the AttnRes-uniform equivalence check.
3. **Trainer acceptance.**
   - `build_base_model` and `load_fold_base` only take fold receipts and check `init_rule == V2_INIT_RULE`.
   - They need a `plant` lineage kind (or a run-0 profile such as `co2-2b-plant` with its own `init_source`/`init_rule`).
   - This reverses the 2026-10-21 "no direct placement" rule baked into config.py, the trainer docstring and the lane preflight. Cyril's 2026-10-22 order supersedes it for run 0, and it should be recorded as a ruling.
4. **Optional:** MTP is dropped and not supported (`mtp_heads=0`; the spec called it a training auxiliary only). The visual tower is correctly discarded (encoder-free).

### (2) Ablate "[Cyril quote withheld]"

- There is no tool. It needs a recipe choice and a receipt. Candidates:
  - shrink-and-perturb, `w <- lambda*w + sigma*rms(w)*eps`;
  - partial re-init of a fraction of FFN rows or heads;
  - noise scaled by tensor RMS.
- Decisions are needed on:
  - which tensors are exempt (embedding/head table, norms, the new media bridge and head, AttnRes);
  - lambda and sigma;
  - the seed.
- Write per-tensor RMS before and after plus a sha256 into the receipt. Checkpoint and clone the planted base first, as the standing rule requires.

### (3) Per-modality marker tokens and an output-decision token

1. **Add the tokens.** Append them to `V2_CONTROL_TOKENS`; appending keeps ids 248320-248327 stable.
   - Markers: `<|text|>`, `<|image|>`, `<|audio|>`, `<|video|>`, `<|3d|>`. Decide whether speech and music share `<|audio|>`.
   - Output decisions, for example `<|out:text|>`, `<|out:image|>`, `<|out:audio|>`, `<|out:video|>`, `<|out:3d|>`, `<|out:edit|>`, with combinations written as a sequence plus a terminator.
   - Initialise the new rows from related Qwen rows (for example the mean of `<|vision_start|>` or label words) rather than zero. The table is tied, so a zero row is a zero input.
2. **Use them in the layout.** `V2Sample.media` must emit the modality marker in place of, or next to, `<|media|>`. The header keeps size, fps and draft text.
3. **Supervise the decision token.**
   - For generation samples the x0 pass cannot carry labels, so each generation sample needs a twin causal sample: request + `<|assistant|>` + decision token(s) supervised. That is a BatchBuilder and `micro_step` change.
   - For text-only replies, `<|out:text|>` must be inserted into the pre-tokenized corpora. That shifts positions against the teacher caches, so it needs a position-remapping view: insert the token, mark the cache position -1, shift the rest.
4. **Unify the chat format.** Either Qwen's `<|im_start|>role` everywhere, or band roles everywhere, but not both.
5. **Joint concepts** (the same concept as image, text and audio in one context) need multi-span samples and a data record format that groups modalities. Today each understanding turn holds one span.

### (4) All modalities at real sizes with distillation on 4 P40s for days

**Data**
- A real-media source class: a manifest of canonical records with media path, caption or transcript, provenance and split. It must:
  - decode at real sizes (images 256²-1024², aspect-bucketed to multiples of 32);
  - resample audio to 38.4 kHz (48 kHz -> 38.4 kHz polyphase);
  - cut video at 12.5 fps into 2-frame tubelets;
  - voxelize 3D to SDF at 64³ or more;
  - apply the dataset gain to `V2_SIGNAL_RMS`;
  - run in multi-process prefetch.
- `ProofMediaSource` is the only implementation today.
- **Shard batch building per rank.** Every rank builds all world x accum micro-batches today, so decoding cost multiplies by the world size.
- Grow the text data config from the smoke's 2 corpora to the full cached set. About 40 corpora have top-64 q38 and Strata caches [V].
- **Add an exclusion guard.** There is no code check against:
  - `selfgen`/`tool-selfgen`;
  - `d2_refill/codex-native-v7/v8` (scripted, and Codex output);
  - `toolcall-q38-*`;
  - Claude or Codex outputs;
  - the raw Strata/Sprite spool.

  The data config is the only gate.

**Batching and memory**
- **Generation passes need a token budget.** `--gen-rows 32` caps samples, not tokens. A real 1024² target is 1,024 rows (2,048 with its draft as context), times 32 samples, in one left-collated forward. That will not fit: the D3 peak is already 20.7 of 24 GB at 9k tokens. Add max-tokens per pass, split the backward, and bucket by length.
- **Context at 50k minimum** (256k capable) is not trainable as built:
  - at 50k tokens, fp32 per-layer checkpoints alone are about 12 GB (30 blocks x 50k x 2048 x 4 B), on top of 15.4 GB of weights and gradients;
  - it needs activation offload to host or fp16/bf16 storage of checkpoints;
  - verify the SDPA memory-efficient kernel on sm_61 with the refinement mask, because the explicit path would build a [heads, T, T] fp32 score matrix;
  - long text rows with teacher signal also need caches beyond 4,096 positions (only `long-00000` exists).

**Distillation**
- Text KD exists (top-64 bucket KL, q38 + Strata + K3 xtok).
- Missing:
  - the K3-primary weighting, per the binding 10-17 ruling, and wider K3 coverage;
  - a top-10+ agreement metric, and optionally a top-n ranking target;
  - media KD: REPA to QVL32 (cache at `teacher_cache/qvl32_vit/qvl32_vit_l13`, pre/post-merge UNVERIFIED), and teacher-generated media pairs as data (Qwen-Image 2.1, H3, TTS).
- Optional TOKENIZER_FREE_STUDY 3e levers: velocity-space loss weighting and an LPIPS image term.

**Joint media refinement**
- Allow one open suffix to hold several spans, or a mixed AV packet on the shared 80 ms clock, so that video plus audio (and image sets) refine together.
- Optionally turn on A8 (reverse GDN scan) so the 18 GDN layers are not raster-causal inside the span. It is off and unproven at proxy scale.

**Multi-GPU robustness**
- Fix or mitigate the C3 resume divergence before a multi-day run. Options:
  - find the rank-local state after resume;
  - broadcast rank 0's weights and optimizer state after load;
  - add a periodic sha check that does more than refuse.
- Drop `world` from `training_identity`, or document it, so a lost card does not strand the run.

**Throughput (4 P40s)**
- Measured: about 150 tok/s on 1 card, about 230 tok/s on 2 cards. The all-reduce is 23-25% of the step and moves 7.7 GB fp32 per step through gloo.
- Ways to cut it:
  - raise `--accum` (for example 4);
  - reduce in fp16 on the host;
  - overlap the reduction with backward.
- Projection, arithmetic only, with toy media and no real decoding: about 430 tok/s (~37M tokens/day) at accum 1, and about 550 tok/s (~47M tokens/day) at accum 4.
- Host RAM: about 11 GB RSS plus about 7.7 GB pinned gradient buffer per rank, so roughly 75 GB for 4 ranks. 131 GB was free at 18:40Z, and Strata offloads to RAM.

**Lanes and locks**
- `scripts/arch_v2_2b_queue.sh` only takes `gpu1.lock`/`gpu2.lock`. Its preflight requires a passed proxy proof and a fold init, so it will refuse run 0.
- Needed:
  - a run-0 queue on GPU1, GPU2, GPU3 and GPU5 (`lanes.py claim-run --lane L-TRAIN --gpus 1,2,3,5`, with p40.claims lines);
  - deployment of the code to the path the queue uses.
- **GPU5 is claimed hourly by the media-judge L-EXP job** (claims at :35 each hour through 18:40Z [V p40.claims]). That job must be moved or paused.
- GPU3 was last held by the strata-qsa engine lane (released at 15:40Z).

**Operations for a multi-day run**
- Checkpoints are about 7.7 GB fp32 each. Keep 3 on Lexar.
- There is no periodic sample dump (text answers or decoded media) for watching behaviour. Cyril's pass/fail is behaviour, so even with testing skipped a dump every N steps is the cheapest monitor.

---

## 8. Files cited

- Config, layout and import:
  - `src/chimera_omni/config.py` (`_V2_COMMON`, `chimera_omni_v2_2b_config`, `V2_INIT_RULE`);
  - `src/chimera_omni/v2_layout.py`;
  - `src/chimera_omni/v2_refine.py`;
  - `src/chimera_omni/native_media.py`;
  - `src/chimera_omni/importing/{qwen35.py, profiles.py}`.
- Model and loss:
  - `src/chimera_omni/modeling/chimera.py` (bridge, head, `_native_refinement_layout`, forward);
  - `src/chimera_omni/modeling/layers.py` (`use_chunked_gdn_scan`, `use_sdpa_attention`, `_with_reverse_media_scan`);
  - `src/chimera_omni/media/training.py` (`x0_media_objective`, the audio losses).
- Trainer: `src/chimera_omni/training/v2_trainer.py`.
- Scripts: `scripts/arch_v2_2b_{lane.py,queue.sh}` and `scripts/import_receipt_qwen35.py`.
- Server:
  - `/mnt/lexar/omni-campaign/scratch/arch-v2-trainer/{SMOKE_RULES_v1*.json, smoke_job*.sh, grade_smoke*.py, runs/smoke{,2,3}-*}`;
  - `/home/cereal/models/Qwen3.5-4B` (key layout reference);
  - the teacher caches under `/mnt/lexar/omni-campaign/teacher_cache/qtext/`.
- omni_lab: runs 38296-38325 (smoke), 38060-38240 (proxy proof evals, all pre-registered FAIL).
