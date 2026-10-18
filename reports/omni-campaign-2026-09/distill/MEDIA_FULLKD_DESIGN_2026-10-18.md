# Chimera Omni media distillation, full-parameter: design and status (2026-10-18, Claude session 753087a5)

**Cyril's words this design answers:**
- 2026-10-17: "[Cyril quote withheld]"; "[Cyril quote withheld]".
- 2026-10-18: "[Cyril quote withheld]".
- 2026-10-18: "[Cyril quote withheld]".
- Standing: one model, any input, any output, the most intelligence per GB.

## 1. What already existed (and was reused, not rebuilt)

**`d3_hybrid_module.py` 1.1.1** is the D3 engine of record. It holds:
- the hybrid model: OMNI1's 64 qwen3_5 text layers, plus the five Chimera charted units after text layer 31, plus the native media bridge and the flow head;
- the objectives: guided-velocity flow, speech-in CE, speech-out flow, band KL, vision answers;
- Codex's native-batch binders, with every ReadyObjectiveAdapter guard;
- parity, synthetic and rate proofs.

It had no run driver, and its trainable sets were LoRA r16 + slice deltas.

**Codex's sealed native batches**, already in the student's feature layout (`native_band.v1`) and adopted 2026-10-14:

| set | schema | items |
|---|---|---|
| `native-ace-full-v5` (ACE-Step 1.5 guided velocity, code-conditioned) | native_flow_cache.v1, route music_generate | 2,764 |
| `native-qimg-full-v3` (Qwen-Image guided velocity) | native_flow_cache.v1, image_generate | 1,126 |
| `native-h3-v1` (H3 joint AV) | native_flow_cache.v1, joint_av | 32 (parity set) |
| `native-speech-flow-v2` (speech out on the H3 audio channel) | speech_flow_cache.v1 | 744 |
| `native-speech-ce-v2` + `native-audio-ce-v2` (speech in, answer CE) | audio_cache.v1 | 380 |
| `geometry-flow-cpu-v5/native-batches` (HY3D, image-conditioned) | recovery_geometry_flow_cache.v1 | 32 |
| `native-edit-v1` (Qwen-Image-Edit) | (to read) | (to read) |

The speech texts and recordings are Cyril's own data.

**Claude's sealed teacher caches** (Lexar `teacher_cache/`): ACE DiT velocity, ACE LM codes, HY3D trajectories, QIMG TE/VAE/velocity, QVL32 ViT L13 (+ ref), and `qtext_vl` (Q38 answers on vision rows).

## 2. Stage M1 (running): every weight of the media modules, trunk frozen

- **Trainer:** `distill/tools/media_full_kd_trainer.py` 1.1.0. Queue: `media_m1_queue.sh`, on GPU0.
- **Trainables:**
  - every unit parameter, except the H3 curve table and the RoPE inverse frequencies, which the source fixes (as in D3);
  - every flow-head parameter.
  - Total 622,973,233 in 110 fp32 tensors.
- **Optimizer:** the full trainer's streamed Adafactor.
- **Why the trunk is frozen here:** full-KD run 1 trains the trunk on GPU4, and one tensor is never trained in two places. The media tensors are disjoint from run 1's 851 text tensors, so the two results join exactly; M2 then trains everything together.
- **The step (layer-major groups):**
  1. Each trunk layer is loaded once per group and applied to every sample. A group is about 30k tokens, about 60 samples.
  2. The units are recomputed with grad at the end.
  3. Boundaries stay on the GPU.
  4. Per sample: `loss_fn(prediction)` (flow) or `head_loss(normed, context)` (speech), plus an optional unit read scope (D3e's H3 prompt-only scope).
- **Proofs:**
  - `selftest`, on a tiny child made by the real importer: group gradients are bit-identical to d3's per-sample `d3b_forward_backward` (67 tensors) and to d3's `speech_lane_step` (speech in and out). The loss falls.
  - `probe`, on real OMNI1D1 weights at lr 0: every route gets finite gradients; peak 5.5 GiB; RSS ≤ 16 GB; about 280 s per group of trunk streaming. That cost is why groups are large.
- **Data config:** `/mnt/lexar/omni-campaign/distill/media_m1/media_data_v2.json`, with fractions ACE 0.30, QIMG 0.30, H3 0.05, speech out 0.20, speech in 0.15. Every tensor file is checked against its receipt sha. 16 held items per route are never trained.
- **Export:** `out/export/media_modules.safetensors`, in Chimera names (`blocks.<b>.charted.*`, `native_media_head.*`).

## 3. Routes still to add (each is a binder + its d3 objective; M1 and M2 share them)

| route | data | d3 objective | what it needs |
|---|---|---|---|
| H3 production (2,048 targets) | the 512-item H3 corpus | D3b guided velocity | an ided items file (`items.jsonl` has no top-level `id`); `stage_c_media_a h3-encode / h3-text / h3-velocity` in the ComfyUI r5 image (GPU); Codex's `h3_native` conversion (CPU docker) |
| HY3D geometry | `recovery_geometry_flow_cache.v1` (32) | D3b with image conditioning | a binder for the geometry bands (`geometry_feature [576, 640)`, `image_feature [640, 2176)`) |
| vision in (D3d) | `qtext_vl` answers + QVL32 ViT L13 | `d3d_head_loss` + `UnitFeatureKD` | full-depth backward (the C11 bridge rows train); `check_vision_alignment` |
| music codes (D3c) | ACE LM codes (top-64 over the 64,000 written C18 rows) | `band_kl` | full-depth backward to the band embed rows; band head rows; reserved rows bit-frozen |
| image edit | `native-edit-v1` | D3b | read its schema |

Full-depth groups (D3c, D3d) keep boundaries for all 64 layers, so they use smaller groups or `--offload cpu`.

## 4. Stage M2: everything trains together (after run 1 exports)

**One process, one master.**
- The text master is run 1's final weights, held in host RAM (`HostWeightStore.load`).
- The media tensors are M1's export.
- The P1 reader of clone-pre-D1 + p1_plan supplies structure, the bridge and the band extras.

**The engine is `streamed_full_kd_trainer.FullParamEngine`, unchanged.** HybridD3's media methods are bound to it: `hybrid.engine = full_engine`, and the final norm is the engine's.

**Two kinds of step, interleaved by token fraction:**
1. **Text** (`forward_backward`: K3 / Q38 + Flash Next mixture / CE). Exactly run 1's step.
2. **Media groups.** Layer-major forward 0..L-1, then backward L-1..0. On each layer:
   - every sample's recompute + backward accumulates fp32 parameter gradients;
   - ONE fused Adafactor update for that layer follows (the multi-sample form of `FullParamEngine._layer_backward`);
   - the units are recomputed between layer d and layer d-1;
   - the embedding-level gradient trains the embedding rows of the step (text rows sparse, music band rows) and the bridge.

**Trainables:** every text weight, every unit, the bridge, the flow head, and the music band rows. All of it, no adapters.

**Memory (3090, 24 GB):**
- text micro-batch 6 × 4,096;
- media groups of 8-12k tokens with boundaries offloaded to the host;
- units on the GPU only during media steps.

**Export:** the Chimera child. Text tensors map back through the P1 name map (text layer t ↔ `blocks.<text_blocks[t]>`); media tensors keep their Chimera names. The base is a copy of the clone's other tensors. That is the one new full checkpoint the stage writes.

**Gates** (behaviour first, per [[folding-behavior-first-grading]]):
- the held media items per route;
- the bank v1 text review;
- tool calling (BFCL native held set);
- speech in/out held rows.

Finite fit numbers are diagnostics, never gates.

## 5. Decisions recorded here

- **Q38 + Flash Next on one row** is a mixture cache (`mix_teacher_cache.py`, weights 0.25 / 0.75), gradient-exact for the weighted KL sum. The text objective weights are unchanged.
- **The flow head is x0-parametrised and the objective is guided velocity** (the 2026-08-28 proof; the prereg D3b lane loss).
- **"operator_authored" data in Codex's records is Cyril's own data.**
