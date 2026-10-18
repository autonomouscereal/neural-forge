# Chimera Omni step 1 results: the all-source composite (OMNI1)

**The all-source composite child exists: `OC02-F-Q38-27BU-OMNI1` (registered, cloned to `OMNI1.clone-pre-D1`).** It has 27,535,655,753 parameters and holds pieces of all seven sources. It loads and runs, but it is not yet intelligent. On text it runs a 4x-folded Qwen trunk that gets simple arithmetic wrong and loops on most prompts. Its media blocks produce finite output that is no better than the noisy input. Stage D exists to fix both, and it has started (section 7).

Written 2026-10-14 by Claude. It answers Cyril's question to Codex of 2026-10-13: "[Cyril quote withheld]" (Codex root thread `01a1326f`, rollout `rollout-2026-10-12T17-25-18-01a1326f-4f44-7ef0-a2fa-2a64a706fedc.jsonl`).

**Rules for this report:**
- Every number comes from a receipt that I read. The receipt is cited as `[R#]`; the index with sha256s is in section 9.
- Every number is a diagnostic (owner policy 2026-10-06). Nobody has read the text cases yet, so every count is a machine ceiling, not a grade.
- `$OC` = `/home/cereal/omni-campaign`.

---

## 1. What exists

| Item | Value | Receipt |
|---|---|---|
| Checkpoint | `OC02-F-Q38-27BU-OMNI1` at `/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1/`: 14 shards, 1,120 tensors (1,116 BF16 + 4 F32), 55,073,531,261 bytes in 18 files | R1, R3, R5 |
| Parameters | 27,535,655,753 = the S5 child's 26,944,218,624 + 591,437,129 charted-block parameters (asserted equal) | R1, R3 |
| Against the Qwen parent | Parent 27,781,427,952. Net ratio 0.99115; headroom 245,772,199 below the parent | R1, R2, R3 |
| Registration | Record `OC02-F-Q38-27BU-OMNI1-0001-register`, sealed 2026-10-13T05:01:45.445Z, receipt_sha `3f38c5cf771fad29…`, status `candidate`, directory sha256 `8edf50c4…1264`. Ledger sha256 after the append: `716581e6…69cc`, `exit 0` | R2, R4 |
| Structural check | `s7_omni1_check` 1.0.0: "OMNI1 loads strict; identity vs S5 and accounting hold; text and every charted block's media forward finite; every charted block writes on its own segment and only there" | R3 |
| Pre-D clone | `OC02-F-Q38-27BU-OMNI1.clone-pre-D1`. Copy mode, 18 files, 55,073,531,261 bytes, sealed 2026-10-13T20:02:58.841Z, clone directory sha256 equal to the source's (`8edf50c4…`). Made by Codex | R5 |
| Ledger | 39 rows (`wc -l`, 2026-10-14). Row 36 is the OMNI1 register; row 39 is the clone | `sed -n 36p/39p ledger.jsonl`, 2026-10-14 |

Layout: 69 blocks = 64 text-trunk blocks plus 5 charted media blocks at positions 32-36. The text trunk is bit-identical to S5 in all 64 blocks (`identity_trunk_vs_s5.bit_identical = 64/64`) [R3].

## 2. What each of the seven sources contributes

Source: `composite_manifest.json` `sources_all` (count 7) and `components` [R1].

| # | Source | What is inside OMNI1 | Parameters | Fold ratio |
|---|---|---|---|---|
| 1 | Qwen3.8-27B-Uncensored (trunk parent) | All 64 text layers. Each FFN is folded from 17,408 to 4,352 channels (S3). Attention and GDN mixers, embeddings and head are passed through. MTP is dropped, not folded (424,699,392). | FFN 64 x 66,846,720 = 4,278,190,080 (parent 64 x 267,386,880 = 17,112,760,320) | **0.25 per layer** (4x) |
| 2 | Kimi K3 | Dense K3 layers 1-92, lifted through the c17 ridge charts into 12,288 added rows per trunk layer. The union FFN is 16,640 wide = 4,352 Qwen + 12,288 K3. Design `a003` = trunk/additive 0.03. | 12,079,595,520 added (64 x 3 x 5,120 x 12,288) | K3's own fold: per MoE layer 29,784,936,832 to 370,675,200 (`moe_net_ratio` 80.353), from `K3_RESULTS_2026-10-08.md` line 25 [R18]. I did not re-read that receipt. |
| 3 | ACE-Step 1.5 (DiT + LM) | Block 32 = `ace_dit_L12` (C16 charted 4x block, insert arm). C15 interface rows (type 4). **ACE-LM**: the C18 music-code band, rows [252,416, 317,952) of embed and head. | Block +102,811,904; C15 2,631,680 written; C18 671,088,640 written in place | MLP 37,748,736 to 9,437,184 = **0.25**; whole block 0.550 |
| 4 | MiniMax H3 | Block 33 = `h3_dit_L25_curve6`; C15 rows (types 2 and 3) | Block +267,889,176; C15 993,280 + 337,920 | MLP 231,211,008 to 57,802,752 = **0.25**; whole block 0.551 |
| 5 | Qwen-Image | Block 35 = `qimg_mmdit_L30_img` (image stream only); C15 rows (type 1) | Block +154,985,727; C15 665,600 | MLP 75,512,832 to 18,874,367 = **0.2499**; whole block 0.686 |
| 6 | Hunyuan3D 2.1 | Block 34 = `hy3d_dit_L10`; C15 rows (type 7) | Block +46,156,031; C15 665,600 | MLP 33,564,672 to 8,388,607 = **0.2499**; whole block 0.500 |
| 7 | Qwen3-VL-32B (ViT) | Block 36 = `qvl32_vit_L13`; C11 rows (type 6) | Block +19,594,291; C11 7,879,680 | MLP 9,921,872 to 2,479,027 = **0.2499**; whole block 0.512 |

**What this means in practice:**
- Each media source contributes **one folded block** plus interface rows. Its VAE and the rest of its network stay outside the checkpoint.
- The S8 media tests below therefore measure one block's denoising emission through the trunk. They are not end-to-end generation.
- All five components meet the 4x rule (`meets_4x: true`) [R1].

## 3. Text capability (S8, bank v1, 16 cases)

**Setup** [R6, R8]:
- Bank `lfm2_semantic_whole_response_prompts_v1` (`0ee6bf14…`).
- Thinking off; greedy decoding; 1,024-token cap; `--ledger-band-logit-mask` on.
- bf16 on a Quadro RTX 8000 (head) plus 2x RTX 3090, split 6/33/30.
- `run_omni_frozen_review` 1.2.0. Dossier generated 2026-10-13T08:47:59Z, wall time 6,177 s.
- Preflights passed: decode path `agree=True`; emit-zero floor passed (min |h| 137.39, min logit sd 2.102).

### 3.1 Case by case

Columns: tokens / termination / machine form box. The parent and S3 boxes come from their own assessments on the same runner [R9, R10]. "S5 proxy" is S5's fp32 teacher-forced argmax agreement with OMNI1's own output stream [R12].

| Case | Parent | S3 (trunk fold) | OMNI1 | S5 proxy |
|---|---|---|---|---|
| conversation-natural-greeting | 28 / complete / CONTENT | 0 / complete / EMPTY | **0 / complete / EMPTY** | 0/0, predicts stop |
| instruction-three-actions | 119 / complete / CONTENT | 1024 / cap / CAP_RIDE | 1024 / cap / CAP_RIDE | 1023/1024 |
| explanation-seasons | 822 / complete / CONTENT | 1024 / cap / CAP_RIDE | 1024 / cap / CAP_RIDE, loop | 1022/1024 |
| arithmetic-multi-step | 385 / complete / ECHO | 1024 / cap / ECHO | 1024 / cap / ECHO | 1024/1024 |
| logic-ordering | 468 / complete / ECHO | 1024 / cap / ECHO | 1024 / cap / CAP_RIDE | 1023/1024 |
| context-retrieval | 60 / complete / CONTENT | 3 / complete / CONTENT | **0 / complete / EMPTY** | 0/0, predicts stop |
| summarization-priorities | 59 / complete / CONTENT | 1024 / cap / ECHO | 1024 / cap / ECHO | 1023/1024 |
| comparison-ram-rom | 388 / complete / CONTENT | 1024 / cap / ECHO | 1024 / cap / ECHO | 1019/1024 |
| coding-deduplicate | 255 / complete / CONTENT | 1024 / cap / ECHO | 1024 / cap / CAP_RIDE, loop | 1024/1024 |
| debugging-off-by-one | 186 / complete / ECHO | 1024 / cap / CAP_RIDE | 1024 / cap / CAP_RIDE, loop | 1023/1024 |
| writing-concise-email | 79 / complete / CONTENT | 1024 / cap / CAP_RIDE | 1024 / cap / ECHO, loop | 1022/1024 |
| planning-dependencies | 1024 / cap / CONTENT | 1024 / cap / CAP_RIDE | 1024 / cap / CAP_RIDE, loop | 1024/1024 |
| uncertainty-insufficient-data | 1024 / cap / CONTENT | 1024 / cap / ECHO | 1024 / cap / ECHO, loop | 1023/1024 |
| safety-household-chemical | 954 / complete / CONTENT | 1024 / cap / ECHO | 1024 / cap / CAP_RIDE, loop | 1024/1024 |
| creative-continuity | 377 / complete / CONTENT | 1024 / cap / ECHO | 1024 / cap / CAP_RIDE, loop | 1021/1024 |
| multi-turn-correction | 720 / complete / CONTENT | 0 / complete / EMPTY | **0 / complete / EMPTY** | 0/0, predicts stop |

**OMNI1 totals** [R6]:
- 3 empty answers. The model emitted end-of-turn immediately on all three.
- 13 runs to the 1,024-token cap.
- 8 cases with `loop_detected = true` (counted from the dossier; the state doc's "9" is wrong).

**Machine ceilings, columns A/B** (all upper bounds, of 15 scored lanes) [R8, R9, R10]:

| Model | Column A | Column B |
|---|---|---|
| Parent | 15 | 12 |
| S3 | 8 | 1 |
| OMNI1 | 5 | 0 |

The sealed S8 record says `alive-degraded` as a ceiling only; `useful` is 0 [R7].

### 3.2 Four cases, quoted from the dossier [R6]

**arithmetic-multi-step.** The task is 17 kits/h x 6 h, 8 fail, pack into 5 boxes. OMNI1 gets the first product wrong and stays consistent with it:

> `$$17 \times 6 = 118$$` … `$$118 - 8 = 116$$` … "28 complete kits go in each box, and 4 remaining kits are left over."

It then repeats "Let's check the answer. $$28 \times 5 + 4 = 116$$" until the cap. The correct values are 102, 94, and 18 per box with 4 left over. The parent writes "= 102 \text{ kits}" and "= 94 \text{ kits}" [R11]. S3 also writes `17 \times 6 = 118` [R13], so this miss comes from the trunk fold, not from the media install.

**debugging-off-by-one.** OMNI1 starts correctly and then collapses into a repeated word:

> "The bug in the function is that the slice `values[start - 1:]` is off off off off off off …"

The parent names the fix (`values[start:]`) in 186 tokens [R11].

**safety-household-chemical.** OMNI1 misreads the prompt as a puzzle and loops:

> "The riddle is a classic puzzle. Let me decode the clues: - "I accidentally mixed two household cleaners" → "I accidentally mixed two household cleaners" → …"

The parent gives ventilation and stop steps [R11]. This is the safety lane. It is charged but never machine-scored, and it is still unread [R7].

**context-retrieval.** OMNI1 returns an empty string. It ends the turn at once on a note-plus-question prompt. The parent answers with Saturday, 4:15 p.m., and the north gate [R11].

The most coherent OMNI1 text is the summarization case. Its first sentence restates the supplier comparison correctly, then the model keeps going to the cap.

### 3.3 Where the text damage comes from

- **OMNI1 equals S5 on text.**
  - R-Q (8 prompts, 210 positions): OMNI1 vs S5 top-1 = 210/210, and the max absolute logit difference on the base vocab is 0.0 [R3].
  - S5's fp32 CPU teacher-forced proxy on OMNI1's own S8 streams agrees at **1019-1024 of 1024** positions on all 13 non-empty cases.
  - On all 3 empty cases S5 also predicts the stop right away [R12, R14; `proxy_exit=0 2026-10-13T11:33:26Z`].
  - So the media install did not cause the text failures. They were already present in S5.
- **The damage is in the folded trunk.**
  - R-Q top-1 against the parent: S3 108/210 (0.514), S5 110/210 (0.524), OMNI1 110/210 (0.524) [R3, R15].
  - S3 already shows the same failure classes as OMNI1: 2 empty answers, 13 cap rides, and the 118 [R10, R13].
- **Held NLL** (24,576 held positions of `omni-evidence-27bu-v1`, bf16, vcheck) [R15, R16]:

  | Model | Held NLL | Held top-1 vs parent |
  |---|---|---|
  | Parent | 4.290 | 1.0 |
  | S3 trunk | 3.756 | 0.630 |
  | S5 | 3.432 | 0.543 |

  OMNI1 was not measured separately; its text logits equal S5's (above).
- **Caution: the held NLL does not match the behaviour.** The folded children score a *lower* NLL than the parent on this held split. Free-run behaviour shows the reverse. I have no explanation for this (UNVERIFIED cause; the set is the fold's own evidence corpus), so it is not evidence of capability.
- **Music band.** Unmasked, OMNI1's C18 music-code rows win the full-vocab argmax at **146 of 210** R-Q positions (max band logit 32.75) [R3]. This is why every text decode and every export must keep `--ledger-band-logit-mask`, or the model emits music codes. No text loss in D2-D4 trains these rows down (preregistration section 2.1).

## 4. Media out (S8 media forward and decode)

**How the test works** [R17]:
- `s8_media_decode` 1.0.0, 2026-10-13T06:42:32Z, ComfyUI VAEs on an RTX 3090.
- Each stimulus is noised to sigma (x_t). OMNI1's emission is decoded as an x0 estimate at two points:
  - `omni_final`: at the final hidden state.
  - `omni_site`: right after the charted block, block 36.
- The parent denoiser's x0 is decoded beside it where it was captured.
- "vs noisy input" asks whether the x0 is closer to the clean target than x_t itself was.
- Structural check (prereg G4, functional): every OMNI1 decode is finite. Audio rms > 0, images non-constant, H3 has 13 frames, and HY3D has faces > 0 at sigma 0.25.

| Modality, metric | Sigma | x_t (noisy input) | OMNI1 final | OMNI1 site | Parent x0 | OMNI1 final minus noisy input |
|---|---|---|---|---|---|---|
| **ACE** audio, SNR dB vs clean (192,000 samples at 48 kHz; omni rms 0.30/0.26/0.26) | 0.30 | 7.69 | 3.45 | -2.78 | 6.72 | **-4.24 dB** |
| | 0.50 | 2.16 | -0.58 | -7.09 | 2.99 | **-2.74 dB** |
| | 0.75 | -0.55 | -1.51 | -8.08 | -0.48 | **-0.96 dB** |
| **Qwen-Image**, PSNR dB vs clean (512x512) | 0.15 | 27.98 | 28.35 | 28.33 | 38.53 | **+0.36 dB** |
| | 0.35 | 19.02 | 19.41 | 19.41 | 33.54 | **+0.39 dB** |
| | 0.65 | 12.38 | 12.74 | 12.78 | 29.17 | **+0.35 dB** |
| | 0.90 | 9.38 | 9.55 | 9.68 | 23.39 | **+0.18 dB** |
| **H3 video**, PSNR dB vs clean (13 frames, 64x64) | step 0 | 9.66 | 5.28 | 6.12 | not captured | **-4.37 dB** |
| | step 1 | 8.24 | 4.44 | 6.04 | not captured | **-3.80 dB** |
| | step 2 | 8.34 | 4.81 | 6.41 | not captured | **-3.53 dB** |
| **H3 audio**, SNR dB vs clean (17,600 samples; omni rms 0.44-0.74 vs clean 0.081) | step 0 | -0.63 | -14.82 | -16.16 | not captured | **-14.18 dB** |
| | step 1 | -4.52 | -18.81 | -19.49 | not captured | **-14.29 dB** |
| | step 2 | -9.78 | -19.29 | -19.83 | not captured | **-9.51 dB** |
| **HY3D** mesh, IoU vs clean (257^3 grid, threshold 0.6; clean mesh 290,872 v / 648,378 f) | 0.25 | 0.074 (187,524 f) | 0.045 (73,918 f) | 0.007 (15,352 f) | not captured | **-0.029** |
| | 0.50 | 0.000 (empty) | 0.000 (empty) | 0.000 (empty) | 0.370 (598,530 f) | 0 |
| | 0.75 | 0.000 (empty) | 0.000 (empty) | 0.000 (empty) | not captured | 0 |

**Reading the table:**
- ACE, H3 and HY3D are worse than the noisy input at every sigma where a comparison is possible.
- Qwen-Image is 0.18-0.39 dB better than the noisy input, against the parent's 10.5-16.8 dB.
- HY3D produces an empty mesh at sigma 0.50 and 0.75, which is structural only at those sigmas; x_t is also empty there.

**Instrument caution.** The ACE stimulus carries the clean latent inside its conditioning features (`context_note`, [R19]). The parent ACE x0 nevertheless beats the noisy input only at sigma 0.50 and 0.75 (+0.83 and +0.07 dB).

## 5. Vision in (QVL, S8)

**Setup** [R20]:
- The question is "Describe this image." with 448-pixel images, the C11 type-6 rows and a 48-token cap.
- Greedy decoding; bands masked.
- Preflight: cached vs recompute argmax equal (step-1 logit max abs diff 0.125).

| Case | OMNI1 answer (verbatim start; every case hit 48 tokens, `stopped=False`) |
|---|---|
| Aquarium gallery photo | "The prompt asks for a description of the image, but the image provided is a text string. The text string is a sequence of words, which are tokens. …" |
| Aquarium, features zeroed | "The image is a placeholder. It is a generic, abstract representation of a generic, abstract representation. …" (loop) |
| Travel-bag photo | "The image is a text block describing a concept of "describe this image" in a specific format. …" |
| Travel bag, features zeroed | the same placeholder loop as the zeroed aquarium |
| No image | "The image is a **2x2 grid** of four squares. Each square is divided into a 3x3 grid of cells. …" (made up) |

**Flags** [R20]: `answers_differ_between_images = true` and `image_answer_equals_zeroed_answer = [false, false]`. The image rows reach the stack and change the answer, but the content is not grounded. This is the expected S8 state that D3d is meant to fix.

## 6. Honest weaknesses

1. **The text intelligence is gone, and the cause is the trunk fold, not the omni assembly.**
   - Arithmetic is wrong (17x6 = 118).
   - 3 of 16 cases return nothing.
   - 13 of 16 ride the cap, and 8 of them loop.
   - The parent ends 14 of 16 on its own, below the cap.
   - Column B (the machine intelligence floor) is 0 for OMNI1, against 12 for the parent [R6, R8, R9].
2. **No media modality generates.** Every OMNI1 x0 is finite, but none is meaningfully better than the noisy input (section 4). Each modality is a single folded block, not its parent network.
3. **Vision answers are not grounded** (section 5).
4. **The music band dominates the unmasked head** (146/210). The mask is mandatory everywhere until D3c/D4 address it.
5. **The held NLL is lower than the parent's** while behaviour is far worse. Treat that diagnostic as uninformative until it is explained.
6. **Nobody has read anything yet.** The bank counts are machine ceilings; the safety lane is unread. Cyril's per-case read against `SCORING-SHEET-OC02-OMNI1-NATIVE-NOTHINK-vs-PARENT-s8.md` governs [R7].
7. **Hardware and dtype differ between the columns.** Parent: fp16 on P40s. S3: bf16 on 3090s. OMNI1: bf16 on RTX 8000 + 3090s. These are recorded as diagnostics [R7].

**Preregistration pin.** `stage2_training_receipt.py check-prereg` was run on the server on 2026-10-14 after amendment A1. It printed `PREREG OK`, `preregistration_sha256` `c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370`, `preregistration_byte_length` 15339, and document sha256 `4e5f4b20…f19e`. The same check on Codex's successor document printed the same pin.

## 7. Stage D plan and status (2026-10-14 ~14:21Z)

**The plan is unchanged** (DISTILL_DESIGN; PREREGISTRATION_OMNI_STAGE_D_v1 + amendment A1):
- **D1** per-layer trunk recovery, then **D2** text + K3 LoRA KD, then **D3** media and vision-in distillation, then **D4** consolidation with one merge, then **stage T**, then a single Q4 export.
- **D1** is aimed at the folded-trunk damage in section 3.
- **D2** at reasoning and arithmetic, using K3 and Q38 targets.
- **D3a-d** at the media and vision results in sections 4-5.
- **D3c** at the music band.

**Status:**

| Track | State | Evidence |
|---|---|---|
| D1 | Running: attempt r4, Codex's (container `codex-d1-native-01a10f78-oct14-c37-recovery-r4`, Up 34 minutes at 14:21Z, GPU0+GPU4 at 8,926/8,917 MiB). It is re-running the 11 finished layers: the log line "L00 (block 0, kda) resumed advance 1040.4s" is at 14:12:56Z. New training starts at L11, which the state doc estimates at about 16:30Z today; it estimates the finish at about 2026-10-15 13:00-16:00Z if no restart. ETAs are UNVERIFIED. | `/mnt/lexar/omni-campaign/distill/d1/run-OMNI1D1-native-01a10f78-20261014-c37-recovery-r4/runtime-log.txt`, `runtime-status.json` (`completed:false`, started 13:49:09Z) |
| D1 data and gate | Recorded as **amendment A1** (successor dataset 979 rows, 917/62, sha `74f79199…`; gate = no live K3 + MemAvailable ≥ 100 GiB + `--no-pin`). The old launcher `queues/d1_launch.sh` is now 1.1.1-superseded and refuses every weight-moving run. | `distill/PREREGISTRATION_OMNI_STAGE_D_v1.md` sha `4e5f4b20…`; `distill/queues/d1_launch.status` |
| Teacher caches, sealed | qtext rescore and groups, qlong, QIMG, ACE DiT and LM codes, HY3D, QVL32 ViT | `STATE_2026-10-14_CODEX_HANDOFF.md` section 4 (not re-read for this report) |
| K3 2.0M cache | Not started: B2 parity failed (p99 gate only); diagnosis is next on GPU6 (plan Step 2) | state doc B2 |
| H3 production caches | Unblocked by the thermal ruling file written today (`distill/queues/gpu6_h3_thermal_ruling.txt`, line 1 `cyril-h3-thermal-withdrawn-2026-10-13T14:26:34Z`). Not yet run. | this lane |
| Vision-in/video-in answer cache (j6) | Not started: the gate file was never written; the box-frame fix comes first (plan Step 3) | state doc B3 |
| D2 prerequisites | Sealed D1, sealed K3 2.0M, and a P2 engine verdict (P2 ran out of memory) | state doc section 5 |

**Cards at 14:21Z:**
- GPU0 and GPU4 hold D1 r4.
- GPU1, 2, 3, 5 and 6 show 5 MiB each.
- Host: 157 GiB available.
- Disk free: `/` 71 G, `/mnt/lexar` 646 G.

## 8. Could not verify

- K3's own fold ratio (80.353) comes from `K3_RESULTS_2026-10-08.md`. I did not re-open its receipt.
- The cause of the held NLL inversion (section 3.3).
- D1 r4's lease validity and the ETAs. I read only the runtime log and status, without touching the container or the leases.
- Section 7's teacher-cache list is carried from the state doc, not re-read.
- No human read exists for any S8 lane.

## 9. Receipt index

| Ref | Path | sha256 |
|---|---|---|
| R1 | `/mnt/lexar/omni-campaign/omni1/OC02-F-Q38-27BU-OMNI1/composite_manifest.json` | `95297685584bc21648dcda27df08ac0f8fb060bc7ebd185ac2ba7860a41be176` |
| R2 | `$OC/registry/OC02-F-Q38-27BU-OMNI1/OC02-F-Q38-27BU-OMNI1-0001-register.json` | `bf3369e5721ac5d8a700093686f124732fe567ffaf98fe8f37ec1529a8ecd4a6` |
| R3 | `$OC/composite/s7/omni1/omni1_check.json` | `9ad8b01fb59498c0dfa82acb9e40f651f51c410055033eb7387c413a991d4662` |
| R4 | `$OC/composite/s7/omni1/logs/register_confirm.log` | `47498340a6af6d0f5eaea325919f3f7810435343b8a0101711dd55446bb7e41c` |
| R5 | `/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1.clone-receipt.json` | `7737e7050ab181d743d019beded2c86a03e50ef8330c02bf28439b9ce02a7162` |
| R6 | `$OC/composite/s8/review/oc-dossier-OC02-OMNI1-NATIVE-NOTHINK.json` | `88a432c74405fe071934078274c1690c789c4400f133d777def6ddf19612d174` |
| R7 | `$OC/composite/s8/review/review-OC02-F-Q38-27BU-OMNI1-s8.json` | `4e72b77c0c229412ddd99506df728605e6f46e9856bfa61dee1f40d5f1be523b` |
| R8 | `$OC/calibration/out/assessment-OC02-OMNI1-NATIVE-NOTHINK-vs-PARENT-s8.json` | `ec3000f5ba3a6fb14655782e6487bd6e75b4ea879607969b6e80464fb8f6b367` |
| R9 | `$OC/calibration/out/assessment-PARENT-Q38-27BU-NATIVE-NOTHINK.json` | `a309f0fb50b00fb2b8b183e9a903ec5017d713342333ed5053f6502c8fc2da5f` |
| R10 | `$OC/calibration/out/assessment-OC02-C2K4352-NATIVE-NOTHINK.json` | `fb3cd4fe75c3f3988aacc15565cfa9ba0b25d59f804efcaae55710946a0fc3f9` |
| R11 | `reports/omni-campaign-2026-09/composite/s0/review/oc-dossier-PARENT-Q38-27BU-NATIVE-NOTHINK.json` | `6b40227a3562e7a0d36563f48b92d99af6a26a60d2603bf21b92f6b23ee56a1c` |
| R12 | `$OC/composite/s8/review/oc-dossier-OC02-C2K4352K3U-PROXY-vs-OMNI1-CPU-FP32.json` | `5f4d906ebe0ce4cbdf296ba9d3f2143063883c870443d422adb2889ab3a2b74f` |
| R13 | `$OC/calibration/out/SCORING-SHEET-OC02-C2K4352-NATIVE-NOTHINK-vs-PARENT-m5.md` (S3's arithmetic case, line 342) | `0e267fb7c406ce2a254da199e6d2bf6aff1516eca5e8b2885cd3eed809d19ccb` |
| R14 | `$OC/composite/s8/logs/s5_proxy.out` | `2bd9f86f861e7f416d4843b8b32991f2da2a02b3960934193d0dd4d243aaed51` |
| R15 | `reports/omni-campaign-2026-09/composite/c10/S5-C2K4352K3U/s5_metrics_full.json` | `7e875818779ebe16727ac8c259694ab8879edb16dd1defe9acbf1cc17ad86bb3` |
| R16 | `reports/omni-campaign-2026-09/composite/c10/S5-C2K4352K3U/vcheck_full_partial.json` | `895af92401962b2ca0d2eaf449e800064d371efe0891e1a9a06473e4fb0b6d33` |
| R17 | `$OC/composite/s8/media_out/s8_media_decode.ace-qimg-h3-hy3d.json` | `d17692523f378928231596a596128ec4462e85ebd05a87d9b4f19a10f3a110f8` |
| R18 | `reports/omni-campaign-2026-09/composite/K3_RESULTS_2026-10-08.md` | `686d8e1c4399e9da008c50ab9dca5650926f51038dc05a235bfb16f2f6949c91` |
| R19 | `$OC/composite/s8/media_out/s8_media_forward.json` | `f48f7a3956e553e96ac2fbb0e58fde3d827e418432c5cacf8c21c6b683b53cc9` |
| R20 | `$OC/composite/s8/media_out/qvl/answers.json` | `6bb78caa7474cfe487c3afc27a26b923d0aa96cf23f0391682fa7920f40ebe98` |

Server and local copies of R6, R7, R17, R19 and R20 have equal sha256 (checked 2026-10-14). R1-R5, R8-R10, R12-R14 were read from a server tarball (`/mnt/lexar/omni-campaign/scratch/claude-step1-evidence/ev.tgz`, sha256 `e7ae7710…16bf4`).
