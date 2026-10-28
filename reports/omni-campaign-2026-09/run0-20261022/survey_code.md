# Run 0 code survey: arch-v2 branch plus the three run-0 branches (refreshed 2026-10-28)

Read-only scout. Scope: what the code already does for run 0 (plant Qwen3.5-2B, ablate, then distill on every modality on 4 P40s for days), and exactly what is still missing.

The 2026-10-22 13:00 version of this survey is kept at `backups/survey_code.20261022.md` (sha256 `93fd52c7…`). It covered `claude/arch-v2-20261021` only. **It is out of date in one important way:** about an hour after it was written, three run-0 branches were committed (2026-10-22 14:02-15:11 MDT) that implement most of what it listed as missing. None of them was merged, integrated or run on a GPU.

Evidence labels: **[V]** = read or run this session (2026-10-28). **[R]** = a report, commit message or receipt says so and I did not re-run it.

---

## 0. Bottom line

1. **Code state** [V]
   - `claude/arch-v2-20261021` is still at `8d8f064`, clean. Nothing is newer on it.
   - Three branches each fork from `8d8f064`, and **none is merged** into another or into arch-v2:

     | Branch | Head | Area | Size |
     |---|---|---|---|
     | `claude/run0-a-20261022` | `4a53d3a` | A: plant, ablate, marker and decision tokens, item-level flow schedule, velocity-weighted loss, GQA kernels | 14 files, +3,177 |
     | `claude/run0-b-20261022` | `96ed2b9` (4 commits) | B: real-size media decoding, records and provenance guard, KD position remap, shards and manifest builder, intake sync | 8 files, +3,708 |
     | `claude/run0-c-20261022` | `e52fa93` | C: 4-P40 trainer (plant lineage, per-rank data, the resume-divergence fix, guards, watch, dumps, supervisor, queue on GPU1/2/3/5) | 6 files, +3,996 |

   - The file sets do not overlap. `git merge-tree` of A+C and of A+B is clean, and B and C touch disjoint files [V].
   - I overlaid A+B+C into a scratch tree and ran the run-0 tests (section 7). **There are two real cross-area failures,** both in the tokenizer argument that A's layout API (`turn_open` / `special_token_id`) takes:
     - B's realize path passes a byte tokenizer stub, and A raises `KeyError '<|im_start|>'`;
     - C's `run0_dump` passes an encode callable, and A raises `TypeError`. Every sample dump item becomes an error.
2. **Nothing of run 0 has run** [V, server 2026-10-28]
   - **Qwen3.5-2B is still not on disk.** Only `/home/cereal/models/Qwen3.5-0.8B` and `Qwen3.5-4B` exist. There is no plant, no ablate output and no run-0 init anywhere.
   - `/home/cereal/omni-campaign/run0/data` holds 19 GB of parts across 10 sources. **There is no `run0_data_v1.json` manifest:** `final_steps.sh` never produced one, so the trainer's `--data-config` has nothing to read. A 450 MB `data-smoke` set does have a manifest.
   - omni_lab has no run-0, plant or co2-2b row after id 38325.
3. **Host and disk numbers that the code's memory plan collides with** [V, 2026-10-28]
   - `MemAvailable` is 29 GiB of 251 (Strata is resident).
   - The trainer's measured footprint is about 10.9 GB RSS per rank. That includes a full fp32 **pinned** gradient mirror (7.7 GB per rank), because `HostAllReduce` allocates one host buffer per parameter.
   - **4 ranks need about 44 GB of RSS. That does not fit in 29 GiB today.**
   - Run-0 C's own `--mem-resume-gib` default is 75.
   - The NAS `/mnt/ai_data` is **100% full** (1.0 MB free), Lexar has 381 GB free and `/` has 113 GB free.
   - A run-0 checkpoint is about 7.7 GB of fp32 weights plus the optimizer and state. Run 0 must checkpoint to Lexar, never to the NAS.
4. **What still blocks a 10-day run.** Each item is detailed in section 8.
   - Merge A+B+C and fix the `<|im_start|>` contract break.
   - Download Qwen3.5-2B and verify its geometry against `co2-2b-run0`.
   - Run the S0-S4 smoke ladder that C defines. Cyril said to skip testing, but one smoke is the only way to know the run will finish before spending 10 days.
   - Finish the data manifest.
   - Cut the host-RAM footprint per rank, or free RAM.
   - Free GPU5: the hourly media judge still claims it, most recently at 01:35Z on 10-28.
   - Note the throughput: about 230 tok/s measured on 2 cards. A 4-card projection is about 35-47M tokens/day (arithmetic only).

---

## 1. The co2-2b profile (`src/chimera_omni/config.py`, arch-v2) [V]

| Field | Value |
|---|---|
| layers | 24 = 18 GDN (`kda` with `kda_decay_kind="gdn"`) + 6 `gated_gqa`, pattern `(kda,kda,kda,gated_gqa)` |
| width / FFN | 2048 / 6144, dense SiLU-GLU |
| GQA | 8 q / 2 kv heads x 256; partial RoPE 64 dims, theta 1e7; mRoPE section (11,11,10) interleaved; QK-norm; sigmoid output gate |
| GDN | 16 v / 16 qk heads x 128; SiLU output gate; fla q/k norm; query scale 1/sqrt(128); no conv bias; conv 4 |
| depth mixing | `block_attnres`, block size 4, `attnres_init="zero"` (a uniform mix, which equals the plain residual under pre-norm up to RMSNorm eps; pinned by `test_uniform_attnres_init_reproduces_the_plain_residual_and_identity_does_not`) |
| vocab | 248,320 Qwen rows + 512 control band = 248,832, tied, fixed-band ledger |
| raw boundary | 3,072 values per row, `native_type_conditioning="none"`, `native_unit_norm="rms"` (A6b); x0 head = RMSNorm -> GLU 2048->4096->3072 |
| refinement | `native_bidirectional_refinement=True` on the GQA layers only; A8 reverse GDN scan off; `native_x0_skip_rms=0` |
| other | `mtp_heads=0`, `aux_heads=()`, `max_sequence_length=262,144` |

- **Parameter count:** 1,923,165,809 as built, against the spec's 1,923,133,041. The difference is A6b's `amplitude_proj`.
- **arch-v2 metadata** carries `init_rule = V2_INIT_RULE` ("fold-built, no direct placement"). The arch-v2 trainer refuses any init that is not a fold receipt (`_fold_receipt_problem`, `load_fold_base`).
- **Run-0 A adds `co2-2b-run0`.** It has the same geometry and differs only in metadata: `PLANT_RULING = cyril-run0-plant-qwen-2026-10-22` and `V2_INIT_RULE_RUN0`. Run-0 C's trainer accepts that lineage.

## 2. The Qwen3.5 importer (`importing/qwen35.py`, `importing/profiles.py`) [V]

**What it maps.** The rewrite is exact, in fp32:
- `embed_tokens` -> rows `[0, 248320)`. A tied head is checked. Band rows are zero in the importer; run-0 A re-seeds them.
- The `(1 + w)` norms are folded into a plain weight.
- `gate_proj` and `up_proj` are concatenated into `ffn.up`.
- GDN: `in_proj_qkv` is split into q, k and v, and `conv1d` is split into three depthwise convolutions. Then `in_proj_z` -> output gate, `in_proj_b` -> write strength, `in_proj_a` -> decay down, and `dt_bias`, `A_log` and `out_proj` map directly. The shared gated-norm row is broadcast to one row per head.
- Full attention: the fused `q_proj` is de-interleaved into query and sigmoid gate; k, v and o map directly; the q/k norms are folded.

**Refusals.** It refuses:
- a yarn or scaled RoPE;
- MoE;
- an activation other than SiLU;
- a non-sigmoid gate (unless the opt-in is set);
- any GDN switch that would not be exact;
- `residual_mode != "none"`.

**What it skips.**
- `mtp.*` and `model.visual.*` are not loaded. Each is listed in the receipt with its shapes and hashes. That is correct for an encoder-free end model.
- `mtp_num_hidden_layers` is stamped into the metadata as dropped.
- The media bridge, the x0 head and the AttnRes parameters keep their random init and are listed as such.
- **mRoPE:** the source `mrope_section` is only stamped. Text uses t = h = w, so the text import is exact.

**Qwen3.5-2B.**
- Structurally it is handled: same family and key layout as the on-disk 4B (`model.language_model.*` 426 tensors, `model.visual.*` 297, `mtp.*` 15 [R, previous survey]).
- The checkpoint is absent. The 2B shape values in `config.py` are web-sourced only.
- Run-0 A adds `qwen35_source_shapes`, `qwen35_geometry_problems` and `check_qwen35_header`. These check shape, dtype and unmapped tensors from the safetensors header without loading weights, and `plant_qwen35_2b.py --check-only` drives them. Running that against the downloaded 2B is the first action.

**Plant (run-0 A, `importing/run0_plant.py`, 842 lines)** [V, code read; R for the measurements]
1. The exact import runs into the Qwen-shaped config (residual "none").
2. It is then loaded into `co2-2b-run0` with Block AttnRes at zero init.
3. Only the names in `PLANT_NEW_PARAMETER_PATTERNS` may be missing: the AttnRes queries and norms, the bridge and the x0 head.
4. The last linear of the x0 head is zeroed.
5. Each used control row is seeded from a named Qwen row plus 0.02 x rms noise. Spare rows get the mean text row plus noise.
6. `plant_function_drift` measures drift against the Qwen function.
7. `plant_qwen35_2b.py --reference` runs the S1 comparison against HF. `--read N` runs a greedy read pack in the donor chat template.

## 3. Ablation (run-0 A) [V, code read]

**Tool:** `scripts/ablate_run0.py` with `run0_plant` ablation.
- Defaults: `--exempt-blocks 7` (the bottom 29% are untouched), `--shrink 0.9` on `ffn.down` above them, then `--sigma 0.05`: `ffn.up` and `ffn.down` get N(0, (0.05 rms(w))^2).
- Mixers, norms, the table, AttnRes, the bridge and the x0 head are never touched.
- `--rollback` is a milder variant: lambda 0.95 on blocks 12-23, no noise.
- It writes a probe and a collapsed-head scan into the receipt. These are receipts, not gates.
- arch-v2 alone has no ablation code.

## 4. Raw boundary, x0 head, control band, markers and decision token

**Boundary (arch-v2)** [V]
- `feature_proj(raw / rms)` + `amplitude_proj(Fourier(log-rms))` + `position_proj` and `position_fourier_proj` (thwr) + `noise_proj(Fourier(sigma))`, then RMSNorm.
- The result is added onto each `<|raw|>` slot.
- Type ids are validated but never used in computation.

**Control band**

| Branch | Tokens | Ids |
|---|---|---|
| arch-v2 | 8 tokens: `<|pad|> <|eos|> <|media|> <|/hdr|> <|/media|> <|raw|> <|user|> <|assistant|>` | 248320-248327 |
| run-0 A | adds 9 more, without moving ids 0-7: `<|text|> <|image|> <|audio|> <|video|> <|3d|> <|out|> <|/out|> <|edit|> <|mask|>` | 248328-248336 |

The remaining 495 rows are spare.

**Run-0 A layout API**
- `V2Sample.marker(modality)` emits the modality marker.
- `V2Sample.media(..., modality=...)` opens a span with its marker instead of `<|media|>`. `None` keeps the legacy opener.
- `V2Sample.decision(outputs, edit=)` emits `<|out|> [<|edit|>] <|m1|> <|m2|> … <|/out|>`, and every token is a CE target at weight 1.0.
- `turn_open` / `turn_close` use the donor's `<|im_start|>` format, with ids resolved from the tokenizer. This fixes the arch-v2 inconsistency of two turn conventions: band roles in the media samples and im_start in the corpora.
- `loss_mask` flows through `collate`.
- **This covers Cyril's ask:** one marker per modality on inputs and outputs, and an output-decision group at the start of the reply that names text, image, video, audio, edit or combinations.
- **Speech, music and sound share `<|audio|>`.** The kind (speech or music) is said in the header text.

**Decision-token supervision on generation samples.** The x0 pass cannot carry causal labels. Run-0 B's samples therefore include a **twin** causal sample for the request plus decision (`KINDS` includes `twin`), and C trains it with decision CE 1.0 [V, code read].

## 5. Media units and joint refinement

**Units (arch-v2 `native_media.py`)** [V]

| Modality | Unit (3,072 values) |
|---|---|
| Image / OCR | 32x32x3 patch; sides padded to multiples of 32; positions normalised per grid |
| Video | 2-frame tubelet of 3 x 32 x 16 at the clip's fps |
| Audio | 3,072 samples of one channel at 38.4 kHz = 80 ms |
| 3D | 8^3 x 6 (truncated SDF, RGB, roughness, metallic) |

There is one shared signal RMS of 0.5, set per dataset by a gain. `native_draft` re-cuts the same media at 1/f resolution for draft-then-refine.

**Real sizes**
- **arch-v2 trains toys only.** The trainer's only media source, `ProofMediaSource`, reads `v2proxy_all_v1.npz`:
  - 64x64 images = 2x2 rows, exactly what Cyril rejected;
  - 4-frame 64x64 video;
  - 0.64 s audio;
  - 16^3 volumes;
  - class-label answers.
- **Run-0 B** (`training/media_sources.py`, `run0_data.py`, `run0_samples.py`) decodes at the plan's real sizes [V, code read]:
  - image understanding: long side ≤ 768 (COCO 640x480 = 300 rows);
  - image generation: five 512-long-side aspect buckets of 192-384 rows;
  - video at 12.5 fps, 24 frames: understanding 256x448 = 2,688 rows, generation 192x320 = 1,440 rows;
  - audio: polyphase resampling to 38.4 kHz;
  - 3D: a 64^3 x 6 volume = 512 rows.

  Decoding is CPU only, through ffmpeg or PIL with a per-read timeout. Pre-decoded `media.bin` blobs are sha256-checked.

**Joint vs unit-by-unit**
- In both branches, one generation sample ends in **one** open media span.
- Inside that span, every row is refined jointly: bidirectional attention in the 6 GQA layers, while the 18 GDN layers stay raster-causal because A8 is off.
- Sampling runs 16 Euler x0 steps over all rows together, which is "chunks with refinement passes" at the span level.
- Run-0 A adds `item_flow_noise_levels`: **one sigma per item** over its whole suffix, with a resolution shift of x max(1, sqrt(rows/256)). Every row of an image or clip therefore moves on one schedule and is refined together.
- **Still not supported in either branch:**
  - joint refinement of two spans in one set, such as video plus its audio, or an image set;
  - a mixed AV packet.

  `V2Sample.media` still refuses anything after an open suffix. The August AV proof (`proof/h3_hybrid_av_flow.py`) did joint AV only on the legacy typed toy config.

## 6. Losses, KD and the trainers

**Text KD (arch-v2, kept in C)** [V]
- The vocabulary head runs on supervised positions only, 256 positions per chunk.
- Terms:
  - CE (0.1 on cached text, 1.0 on media answers);
  - **top-K bucket forward KL** against each same-vocabulary cache (q38-27bu and Strata IQ3_S, top-64);
  - the K3 cross-tokenizer bucket KL (`XtokTeacher`, 0.3);
  - unlikelihood, kept inside one document.
- The top-K plus tail bucket is a top-n target, never top-1.
- **Run-0 C adds:**
  - `topn_agreement`: per-teacher top-10 overlap and captured top-10 mass, logged and used by a fixed holdout alarm;
  - `--teacher-weights` (a per-position rule, so K3 can be 1.0 where it exists, per D1);
  - `loss_mask` gating;
  - B's `text_remap`, which inserts tokens into a cached corpus and remaps the cache positions so the KD stays aligned.

**Media generation**
- arch-v2: x0 prediction with an x0-space MSE (type-balanced), plus raw-audio STFT, mel, level and boundary terms.
- Run-0 A adds:
  - `flow_loss` with **velocity-space weighting** w = 1/max(sigma, 0.2)^2, normalised to E[w] = 1 (the TOKENIZER_FREE_STUDY 3e lever);
  - logit-normal times.
- C exposes them as `--flow-weighting velocity` and `--sigma-sampler`. **The defaults are `none` and `uniform`**, so they must be set on the launch line.
- **Still missing:** an LPIPS or perceptual image term, media KD or REPA to QVL32, and a mel loss for speech beyond the existing audio terms.

**arch-v2 T2 trainer** (`python -m chimera_omni.training.v2_trainer {memory-plan,train,probe}`) [V]
- fp32 everywhere; no autocast, which is right for P40s.
- Non-reentrant per-layer checkpointing, a chunked GDN scan (chunk 64, pure torch), SDPA for GQA.
- Optimizer: Adafactor with factored second moments, no momentum and row-blocked updates (about 2 MB of state). `adamw-offload` would need 15.4 GB of host RAM per rank, which does not fit today.
- Multi-GPU: one process per card, gloo all-reduce through pinned host buffers, no P2P.
- Checkpoints: atomic, sha256 per file, every RNG state, `--resume auto`, STOP file and SIGTERM handling.
- omni_lab: `log-run` start, metric and end, plus `steps.jsonl`.

**Run-0 C trainer** (3,523 lines; on top of T2) [R: commit message and code read]
- `--init` accepts the plant -> ablate -> align receipt chain.
- `--data-config` takes a `chimera_omni.run0.data.v1` manifest read through B's `build_sources`, with one batcher **per rank**. In T2 every rank built every micro-batch.
- Token budgets per pass: `--understand-tokens 8192`, `--gen-tokens 4096`, `--long-lane-tokens`.
- Optimizer details: decoupled weight decay on trunk matrices only, `--new-module-lr` for the bridge, x0 head, AttnRes and control rows, and a WSD schedule.
- **The C-1 resume fix:** every checkpoint `torch.load` maps to the CPU. Before the fix, every rank loaded rank 0's optimizer state onto rank 0's card and copied it card to card, which is the 38324 divergence and P2P. After the fix:
  - rank 0 broadcasts the model and optimizer state after a load;
  - shas must agree before step 1;
  - a 3-step gradient-slice check runs;
  - each checkpoint gets a `replica_sha.json` with repair from rank 0;
  - `--allow-world-change` is added.
- Guards:
  - non-finite steps are skipped and alerted, and the run halts with a checkpoint of untouched weights;
  - divergence quarantines the checkpoint;
  - memory and disk floors pause the run;
  - exit codes 0/2/3/4/1.
- Operations:
  - `run0_dump.py` periodic sample dumps (behaviour, not numbers);
  - `run0_watch.py` (alarms, `status.json`, omni_lab `run_event` / `run_dataset` / `run_model` through raw SQL);
  - `run0_supervisor.py`;
  - `scripts/run0_queue.sh`, which uses `lanes.py claim-run` on GPU1/2/3/5, refuses 0/4/6, checks ECC and the deployed sha, and has `--smoke` (S3, 100 steps).
- **Unchanged from T2:**
  - `--seq-len 4096` default;
  - `HostAllReduce` still mirrors every parameter in its own fp32 pinned buffer (7.7 GB per rank);
  - no activation offload;
  - Adafactor without momentum.

### 6a. What the 2026-10-22 smoke runs showed [V: logs and omni_lab rows re-read 2026-10-28]

All runs are under `/mnt/lexar/omni-campaign/scratch/arch-v2-trainer/runs/`, with pre-registered rules (reviews 38296, 38306, 38321). They tested mechanics only, with random weights for the 2B and toy media.

| Run | Setup | Result |
|---|---|---|
| 38303 D | 2B probe | Out of memory in the Adafactor step. Fixed in d0ed6e1 by row-blocked updates. |
| 38304 E | 2B probe, adamw-offload, 1 card | Passed. 64-78 s per step, optimizer 4-18 s, host RSS 19.6 GB. |
| 38311 D | 2B, 1 card, seq 4096, Adafactor | ~60 s per step, 9.1k tokens, **~150 tok/s**, peak 20.7 GB of 24. |
| 38313 D2 | threaded two-card 2B | Crashed: torch non-reentrant checkpoint `SavedTensorHooks` assert. Led to one process per card in 8d8f064. |
| 38325 D3 | 2B, 2 processes (GPU1+2) | 78-81 s per step, 18.2k tokens, **~230 tok/s**, all-reduce 17.7-20.5 s (23-25% of the step), optimizer 0.6 s, peak 20.7 GB, host RSS 10.9 GB per rank. |
| 38323 / 38324 C3 | proxy, 2 processes, stop at 3 then resume to 6 | **The replicas diverged after the resume** (state sha256 `d5ec80f3…` vs `161ce7c0…`), rc 2. C-1 above is the fix, on run0-c only and never re-run on GPUs. |

- The peak of 20.7 GB at about 9k tokens per micro-step leaves about 3 GB of margin on a 24 GB card.
- C's `--understand-tokens 8192` / `--gen-tokens 4096` budgets keep a pass near the measured point. Real 1,440-2,688-row video items must fit inside `--gen-tokens` with their context, or go to their own pass.

## 7. Test suite status [V]

**arch-v2 `8d8f064`**, on the server Pascal venv (torch 2.6.0, CPU), run over 10 v2-relevant files:
- `test_v2_architecture`, `test_v2_trainer`, `test_v2_all_modalities`, `test_v2_fold_proxy_profile`, `test_v2_mrope_positions`, `test_v2_round5_generation`, `test_chunked_gdn_scan`, `test_import_qwen35_synthetic`, `test_config_import_switches` and `test_import_profiles`.
- Everything passed except **2 failures, both fp32 tolerance edges:**
  - `test_cached_decode_continues_exactly_after_a_chunked_prefill`: 2.19e-5 against a tolerance of 2e-5;
  - `test_two_ranks_through_host_memory_take_the_step_of_one_rank_accumulating_both`: `blocks.1.attention_residual.norm.weight` max diff 5.7e-5, 1.6% of entries > 1e-5.
- The previous survey's full-suite run on 10-22 counted 729 collected and 10 failed: these two, plus 8 `*_proof_artifacts` tests that need `proof_outputs/`, which the server code copy lacks.

**Run-0 A+B+C overlaid** (scratch tree, local Python 3.11, torch 2.13 CPU):
- I ran `test_run0_arch`, `test_run0_data`, `test_run0_trainer` and the v2 trainer, architecture and all-modalities files. **2 tests failed. No other FAILED or ERROR line appeared.** I did not capture pytest's final count line.
- **Both failures are real cross-area contract breaks.** A's layout API (`V2Sample.turn_open` -> `special_token_id`) expects a tokenizer object or a token-to-id mapping, and B and C call it differently:
  1. `test_run0_data_torch_optional.py::test_realize_with_the_real_layout_api_emits_the_grammar`
     - B's `realize` passes a byte tokenizer stub with no `<|im_start|>`.
     - A raises `KeyError("the tokenizer has no special token '<|im_start|>'")`, but B's test expects A-2 to resolve it.
     - This affects the proxy/byte path only. The Qwen tokenizer has `<|im_start|>`.
  2. `test_run0_trainer_torch_optional.py::test_dumps_write_readable_files_split_across_ranks`
     - Every dump item fails with `TypeError: enc must be a tokenizers.Tokenizer, an HF tokenizer or a token -> id mapping`.
     - C's `run0_dump` passes a plain encode callable where A wants a tokenizer.
     - Merged as-is, the periodic sample dumps (the behaviour monitor for the run) would write errors instead of samples.
  - Neither failure is visible on its own branch. They show that the three areas were never run together.
- C's commit message reports "full suite 744 passed, 21 skipped, 0 failed" on C's own branch alone, without A and B [R].

---

## 8. MISSING, by run-0 step (2026-10-28)

### (1) Plant Qwen 2B weights

1. **Download `Qwen/Qwen3.5-2B` to Lexar** (Apache-2.0, not gated [R, plan §0]), never to the full NAS.
2. Run `plant_qwen35_2b.py --check-only`: the header-only geometry check against `co2-2b-run0`. If the actual config disagrees with the web-sourced shapes (heads, head dims, `tie_word_embeddings`, rope), the profile changes first.
3. Run `plant_qwen35_2b.py` (writes the plant layout and receipt), then `--reference --read N`: drift against HF plus greedy reads. That is behaviour, not a gate number.
4. **Code: merge A+B+C into one branch** (no conflicts by merge-tree), then fix the tokenizer-argument contract that both integration failures break:
   - **C's `run0_dump`** must pass the real tokenizer object, not an encode callable. Otherwise every dump item errors.
   - **B's proxy/byte path** needs a tokenizer that has `<|im_start|>`, or A needs a band-role fallback.
   - Then re-run the three run-0 test files on the merged tree.
5. Deploy one commit-pinned code copy. `run0_queue.sh` checks the deployed sha.

### (2) Ablate

- The code exists (`ablate_run0.py`: exempt 7, shrink 0.9, sigma 0.05).
- Missing: running it on the real plant. Checkpoint and clone the plant first, per the standing rule. Read the probe and the collapsed-head scan in the receipt before training.
- Optionally run A-7 `align_bridge_stage0.py` (stage-0 image bridge alignment).

### (3) Marker tokens and the output-decision token

- **Done in code** (A layout API, B twin samples, C decision CE).
- Missing:
  - the merge;
  - confirming the seeded control rows in the plant receipt;
  - confirming that B's text corpora get `<|text|>` and `<|out|> <|text|> <|/out|>` inserted through `text_remap` with cache positions remapped. Check `verify-kd` on the real caches after the manifest is built.
- **Not supported:** a sample that outputs two raw modalities at once, such as video plus audio. The decision can name both, but generation still ends in one open span (see the joint refinement gap below).

### (4) All modalities at real sizes with distillation on 4 P40s for days

**Data**
- B decodes real sizes, but **no run-0 manifest exists** (`run0/data/run0_data_v*.json` is absent). Finish `final_steps.sh` (manifest -> per-source verify -> verify-kd -> intake register -> summary) on the merged code.
- The 19 GB of parts cover COCO caption/t2i, Cyril's speech, the H3 video set, HY3D voxel64 and the intake image-edit, t2i, music, Strata-text and video sources.
- **Gaps are named in the plan, not closed by code:**
  - real sound effects and paired AV (downloads approved under D-audio, none on disk);
  - video-understanding text.

**Joint media sets**
- Allow one open suffix to hold several spans (video + audio on the shared 80 ms clock, image sets), with one sigma per set.
- Today it is one span per sample, in both branches.
- Optional: A8 reverse GDN scan, so the 18 GDN layers see the whole span.

**Media KD**
- No REPA or QVL32 feature loss, and no LPIPS. Media distillation is data-only: teacher-generated pairs from Qwen-Image 2.1, H3, ACE and TTS land as records.

**Long context**
- `--seq-len 4096` default. The text caches cover 4,096 positions, plus the `long-00000` corpus.
- 50k-token training is not feasible as built: fp32 per-layer checkpoints alone are about 12 GB at 50k on top of 15.4 GB of weights and gradients.
- It needs activation offload to host (which conflicts with today's 29 GiB free) or fp16 storage of checkpoints, plus long teacher caches.
- Run 0 can only be 4k-trained. The 256k capability comes from the parents' rope, not from training.

**Host RAM (blocks 4 ranks today)**
- About 10.9 GB RSS per rank x 4 = about 44 GB, against 29 GiB `MemAvailable` [V].
- **Code lever:** `HostAllReduce` allocates a full fp32 pinned mirror of every parameter per rank (7.7 GB). Two fixes:
  - a bucketed reuse buffer (for example 256 MB): reduce bucket by bucket, copy back, reuse;
  - or one shared `/dev/shm` reduce buffer: all ranks are on one host.

  Either cuts about 7.5 GB per rank, to roughly 3.5 GB RSS each (about 14 GB for 4), and may shorten the 18-20 s all-reduce.
- **Ops lever:** Cyril's item 2. Move Strata MoE experts to disk-backed reads to free host RAM. That is Strata-side, not this repo.
- Set C's `--mem-floor-gib` and `--mem-resume-gib` (default 75) to values that the host can actually reach. Otherwise the supervisor pauses forever.

**Throughput and completion**
- Measured: about 150 tok/s on 1 card, about 230 tok/s on 2.
- 4-card projection (arithmetic): about 430-550 tok/s, or about 37-47M tokens/day. 10 days is about 0.4-0.5B tokens, minus real-media decode time and the per-rank all-reduce, which grows with 4 ranks.
- To be sure the run will complete before committing 10 days, the minimum is:
  - run S3 (`run0_queue.sh --smoke`, 100 steps on 4 cards);
  - include one forced resume, which re-proves the C-1 fix that has never run on GPUs;
  - check `replica_sha.json` agreement;
  - check peak VRAM below about 22.5 GB with real video items;
  - check host RSS with 4 ranks;
  - check disk headroom for 3 kept checkpoints of about 8-15 GB on Lexar.

**GPU5**
- Still claimed every hour by the media-judge L-EXP job, through 01:35Z on 10-28 [V p40.claims].
- D2 says to move the judge to Strata, or pause `media-judge.timer`, before launch. `run0_queue.sh` claims through `lanes.py`, so a held lock refuses rather than steals.

**Checkpoints**
- fp32 safetensors of about 7.7 GB plus the optimizer and state per checkpoint, keep 3: about 25-45 GB on Lexar (381 GB free).
- C's disk floor must point at Lexar. **Never the NAS (0 bytes free).**

---

## 9. Files cited

- **arch-v2 worktree:** `C:/Users/cereal/Documents/ai training/.agent-worktrees/chimera-omni-5d4d4e1dcd/workers/claude/arch-v2-20261021`
  - `src/chimera_omni/{config.py, v2_layout.py, v2_refine.py, native_media.py}`
  - `importing/{qwen35.py, profiles.py}`
  - `modeling/{chimera.py, layers.py}`
  - `media/training.py`
  - `training/v2_trainer.py`
  - `scripts/arch_v2_2b_{lane.py,queue.sh}`
- **run-0 worktrees** (same parent, `workers/claude/run0-{a,b,c}-20261022`):
  - A: `importing/run0_plant.py`, `scripts/{plant_qwen35_2b,ablate_run0,align_bridge_stage0,attn_kernel_check}.py`, `tests/test_run0_arch_torch_optional.py`
  - B: `training/{media_sources,run0_data,run0_samples,text_remap}.py`, `scripts/{run0_build_manifest,run0_intake_sync}.py`, `run0_stage_data.sh`, `tests/test_run0_data_torch_optional.py`
  - C: `training/{v2_trainer,run0_dump,run0_watch}.py`, `scripts/{run0_queue.sh,run0_supervisor.py}`, `tests/test_run0_trainer_torch_optional.py`
- **Integration scratch tree:** `C:/Users/cereal/AppData/Local/Temp/claude/C--Users-cereal/0c189f21-3de3-4c40-b42c-db54c9d84462/scratchpad/run0-int`
- **Plan:** `RUN0_BUILD_PLAN.md` in this directory (with the decisions addendum `cyril-run0-decisions-2026-10-22`).
- **Server:**
  - `/mnt/lexar/omni-campaign/scratch/arch-v2-trainer/{SMOKE_RULES_v1*.json, smoke_job*.sh, runs/smoke{,2,3}-*}`
  - `/home/cereal/omni-campaign/run0/{data, data-smoke, code-b*, final_steps.sh}`
  - `/home/cereal/omni-campaign/distill/queues/p40.claims`
  - omni_lab runs 38296-38325
