# Speech lane: review of Codex's speech work and a proposal for speech in/out in Chimera Omni

**Written:** 2026-10-14, about 14:40Z, by Claude (plan Step 9 in `STATE_2026-10-14_CODEX_HANDOFF.md`).
**Scope:** review plus proposal. No GPU was used, and nothing on the server was launched, stopped, changed or deleted. I made three read-only SSH snapshots (14:12:27Z and two after it) and read local receipts and memory.
**Marks:** (m) = measured, with the receipt named; (d) = derived by arithmetic shown here; (a) = assumption; UNVERIFIED = I did not see it printed.

---

## 1. What Cyril asked for (his words)

Codex root thread `01a1326f`, rollout `…2a64a706fedc`, from the forensics scratch `scratchpad/cx/2a64a706fedc.txt`:

- 2026-10-13 17:38:44Z: "[Cyril quote withheld]"
- 17:44:46Z: "[Cyril quote withheld]"
- 17:56:50Z, answering Codex's question about which Dot conversation: "[Cyril quote withheld]"

What Dot actually saved (PostgreSQL memory, space `chatgpt/dot/2e2c422acb8c81a38dc352a722f0f06e`):

| Memory id | Who | Content |
|---|---|---|
| 892443 | Cyril | "[Cyril quote withheld]" (with a screenshot) |
| 892447 | Dot | Added **Fish Audio Drama 3**. Dot noted that its licence forbids using the model or its outputs to train other speech or foundation models. |
| 892685 | Cyril | "use a qwen model" |
| 892754 | Dot | Added **NVIDIA PersonaPlex 7B v1**, a full-duplex model that listens and speaks at the same time. Dot downloaded nothing. |
| 1002920 | Cyril, 2026-10-14 | "[Cyril quote withheld]". The screenshot was not captured and no reply was saved. **The model is unidentified (UNVERIFIED).** |

Earlier research in memory (Codex, June-July 2026) covers Baichuan-Audio, Fun-Audio-Chat, Step-Audio / Step-Audio 2, Qwen3-Omni, PersonaPlex and LongCat-Audio-Codec. The 2026-09-01 fold plan lists "Qwen3-ASR / Qwen3-TTS 0.6B … fold whole" (memory 999097).

**How I read "bidirectional".** Codex took it to mean bidirectional attention in the ASR encoder (Codex 23:44Z: "bidirectional attention in the ASR audio encoder, while the text and speech decoders remain causal"). Cyril's examples point elsewhere: PersonaPlex (full duplex) and an ASR+TTS pair. I read it as **speech in and speech out**. With his "use a qwen model", that points to the Qwen3-ASR + Qwen3-TTS pair. This is my reading, not his words.

---

## 2. What Codex built, what each part is, and what was proven

### 2.1 Downloaded parent models (all Apache-2.0 per Codex and Dot; I did not re-read the licences)

| Model | Path on the server | Bytes (m) | Revision | Architecture (m, config.json) |
|---|---|---|---|---|
| Qwen3-ASR-0.6B | `$OC/distill/codex-preparation-01a1326f-20261013/speech-teachers-v1/Qwen3-ASR-0.6B` | model.safetensors 1,876,091,704 | `5eb14417…` | Same family as the 1.7B below |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | `…/speech-teachers-v1/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 1,811,626,576 + speech_tokenizer 682,293,092 | `85e237c1…` | Same family as the 1.7B below |
| Qwen3-ASR-1.7B | `$OC/distill/codex-speech-1p7-comparison-modality-v1/parents/Qwen3-ASR-1.7B` | 4,220,320,824 + 478,200,688 | `7278e1e7…` | `Qwen3ASRForConditionalGeneration`. Audio encoder: 24 layers, d_model 1,024, 16 heads, FFN 4,096, 128 mel bins, non-causal (`is_encoder_decoder false`), downsample width 480. Text decoder: Qwen3, 28 layers, hidden 2,048, vocab **151,936** (not Chimera's 248,320) |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | `…/parents/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 3,833,402,552 + speech_tokenizer 682,293,092 | `0c0e3051…` | Talker: Qwen3, 28 layers, hidden 2,048, vocab 3,072. Code predictor: 5 layers, 16 code groups × 2,048. Codec `qwen3_tts_tokenizer_12hz`: 24 kHz mono, 12.5 Hz, 16 valid quantizers × 2,048, causal conv. **9 voices:** serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee, eric, dylan. 12 language/dialect ids |

The 1.7B pair download log printed `required_bytes 9223333259, complete true, all_source_bytes_verified true`.

**Revision split.** Cloud thread `01a10f78` later "selected" `Qwen3-ASR-1.7B-hf` rev `bcd2b5b7…` (memory 1001821, 2026-10-14 12:41Z). That is a different repository and revision from the copy on disk. The `-hf` variant runs on stock transformers. The Pascal venv has transformers 5.12.1, and the on-disk copy needs the `qwen_asr` package. One of the two has to be pinned (§5, E1).

### 2.2 Speech trees on the server (sizes from `du -sh`, 14:12Z)

| Tree | Size | What it is | State |
|---|---|---|---|
| `codex-speech-1p7-comparison-modality-v1` | 8.6 G | The 1.7B parents plus a CPU canary | Comparison only, no admission. **ASR:** 1.7B and 0.6B were both 7/7 verbatim-exact on **one sentence** ("After the storm, the gardener moved the clay pots beneath the porch."), 4 original and 3 regenerated waveforms; mean 13.11 s vs 5.27 s per clip on CPU fp32 (m, `poll-asr-v2.json`). **TTS 1.7B on CPU:** 3 of 4 voices took 56.8 / 65.8 / 67.8 s for 4.16 / 4.72 / 4.80 s of audio, then exit 1 at the 10-minute CPU limit (m, `poll-tts-v4.json`, `complete false`). Codex's own canary scripts are in the local worktree |
| `codex-speech-preparation-modality-01a1326f-v1` | 1.6 G | Admission inputs (201 sources) and `clean-speech-cache-v1` (201 H3 audio-VAE encodes, `sealed true`, `native_admission_granted false`) | The `native-ce` phase **failed**: `ValueError: audio chart requires actual full-duration, full-coverage stereo encoder evidence` (`logs/native-ce.log`). `clean-audio-cache-v1` is empty. The chain was superseded by `codex-native-audio-consumer-modality-v2` |
| `codex-joint-audio-preparation-modality-01a1326f-v1` | 749 M | Joint audio source preparation: 352 inventory rows → 225 candidates → **194 train admitted** (8 procedural non-speech fixtures + 186 speech), sealed; chain exit 0 | Data preparation only; `training_started false` |
| `codex-native-audio-consumer-modality-v2` | 8.4 M | Replay index of the final audio/speech batches | `passed true`, `sealed true`, `model_forward_performed false`, `production_training_ready false`. `native-speech-flow-v2/` cannot be read by my user (Permission denied) |
| `codex-preparation-01a1326f-20261013/speech-*` | teachers 4.1 G, runtimes 877 M, campaign-bf16-v2 85 M, others < 10 M | The 0.6B pair; two venvs (torch 2.11.0+cu128, transformers 4.57.3, qwen_tts 0.1.1); the GPU4 TTS/ASR campaign | See §2.3 |
| `codex-speech-tests-modality-01a1326f-v1..v4` | 12 M | pytest scratch | — |

### 2.3 What was measured

- **TTS-0.6B on GPU4 (RTX 3090, bf16, batch 1):** 192 training utterances, mean 18.79 s each (median 18.47 s), 3,608.3 s of compute for 896.2 s of audio, **RTF ≈ 4.0** (m, `recovery-preparation/speech-gpu-campaign-bf16-v2/tts-192.json`, sha `96911d1a…`). Plus 16 held utterances.
- **ASR-0.6B (bf16, 3090):** 0.82 s per utterance (m, `asr-192.json`, 39 rows in the file I read).
  - Final admission: **201 of 208** sources were exact after case/punctuation normalisation, 7 quarantined (m, `preparation.json`: `independent_asr_exact 201`, `quarantined 7`).
  - Per Codex's messages, that is 186/192 train and 15/16 held. I did not see the 186 printed.
- **Corpus shape (m, `tts-192.json`):**
  - 192 rows = **48 distinct sentences × 4 voices** (Aiden, Ryan, Serena, Vivian).
  - Mean 10.85 words; mean 4.67 s (min 2.96, max 20.4); English only; all synthetic.
- **H3 audio VAE** (`minimax_h3_audio_vae_fp32.safetensors`, 605,254,808 B, in `/home/cereal/comfyui/models/vae/`):
  - Speech resampled to 32 kHz stereo gives **800 samples per latent frame = 40 Hz**. Raw latents are `[1,32,2,T]`, packed channel-major as `[2T,32]`. Encodes repeat bit-exactly.
  - On one 4.175 s utterance: encode 10.1 s and decode 18.8 s on CPU; waveform **SNR 10.85 dB** (m, `h3-audio-proof-v3/probe.json`, sha `d7fcbe06…`).
  - ASR transcribed the original **and** the H3-decoded audio exactly ("The small wooden boat is moving slowly across the lake.") (m, `h3-decoded-speech-asr-v1/result.json`, sha `3ce05af4…`).
  - So the H3 latent space keeps the words of one sentence. That is n = 1.
- **Replay index** (m, `speech-replay-final-v1/actual/replay-index.json`, sha `3c99cfb7…`):
  - 938 entries:
    - 744 `speech_generate` / `supervised_speech_rectified_flow` (186 × σ ∈ {0.05, 0.3, 0.7, 1.0});
    - 186 `speech_understand` / `authored_audio_assistant_ce`;
    - 8 `audio_understand`.
  - 31 held rows. Mean sequence length 411.0 (CE) and 405.5 (flow) positions; mean 184.3 latent frames per channel (≈ 368.7 audio rows).
  - Supervised units: 2,833 (CE) and 8,776,960 (flow).
- **Natural speech** (m, `chimera_natural_speech_repair_oct13_v3/manifest.json`):
  - 1,763 LibriSpeech train-clean-100 rows: 1,394 train / 176 validation / 193 test.
  - 884 ASR prompts had the transcript leaked into the prompt; Codex stripped it.
  - **0 admitted.** The listed blocker: "Original source dates need explicit 2025–2026 evidence."
  - The registry also holds 361 laion-orpheus TTS rows and 3,108 SonicMaster audio rows.
- **ASR raw-input consumer v20** (Codex cloud workspace, `NATIVE_SPEECH_ENGINEERING_REPORT_V20.json`, sha `d9bc0f3a…`):
  - A CPU review of one LibriSpeech clip: PCM round trip byte-equal, labels exact. 0 rows admitted.
  - It records **OMNI1's audio input ABI**: `audio_h3`, set `c15`, type_id **3**, latent width **32**, feature band **[480, 512)**, emission band **[224, 256)**.

### 2.4 The objectives Codex designed (candidate amendments, not in the canonical prereg)

Source: `recovery-preparation/speech-preregistration-v1/PREREGISTRATION_SPEECH_RECOVERY_2026-10-13.md` (sha `cf676e8d…`). Both stages sit after D3d vision/video and before D4.

**`D3d_audio_ce` (understanding)**
- Clean H3 latents at σ = 0 go into feature band 480:512.
- Loss: assistant-answer cross-entropy over rows [0, 248,320).
- The H3 unit reads only prompt text and the clean audio, never the answer.
- Trainable: H3 unit, type-3 bridge columns, type-embed row 3, shared r16 LoRA. lr 2e-5 / 1e-4.

**`D3d_speech_flow` (speaking)**
- Condition: "Generate speech in the synthetic {speaker} voice, saying exactly:\n{transcript}". Noisy type-3 rows follow the assistant prefix.
- Noise: x_t = (1−σ)·x0 + σ·ε; only x_t enters the feature band, and x0 supervises emission band 224:256.
- Loss: velocity-form MSE (x_t − x̂0)/σ vs ε − x0.
- σ grid {0.05, 0.3, 0.7, 1.0}. 32-step Euler sampling for review.
- Trainable: H3 unit, type-3 bridge and emission slices, shared r16 LoRA.
- Stop at 25% of budget for a listening + ASR review.

**A new H3 rotary chart for audio-only rows:** (prompt_text_count + i, 0, c), where c is the stereo index.

### 2.5 Quality verdict

**Good:**
- Hashes everywhere, held splits kept out of the optimiser, independent exact-ASR admission.
- The real H3 VAE was used and measured; the 40 Hz audio rate was measured rather than assumed.
- The leakage repair is correct, and the Fish licence problem was caught.
- The x0-endpoint semantics match Claude's `d3_hybrid_module` 1.0.2 (`--flow-param x0`; "FLOW TARGETS ARE x0"), so the two objectives can share an emission band.

**Not good enough to train on as it stands:**
1. **Nothing has touched Chimera.**
   - `model_forward_performed false` and `production_training_ready false` on every speech receipt.
   - No speech capability exists or has been measured, in or out.
2. **The corpus is a smoke set.**
   - 48 sentences × 4 voices, about 15 minutes of audio, English only, synthetic only.
   - The 1.7B comparison used one sentence.
   - At this size, speech-in would at best learn those 48 sentences, and speech-out would learn 4 voices saying 48 sentences.
3. **There are two D3 engines.**
   - The speech objectives are written for Codex's `experiments/chimera_recovery` native trainer (`speech_flow.py` `a7633b7d…`, `audio_cache.py` `9ce755d1…`, `h3_audio_chart.py`).
   - The D3 engine of record is Claude's `distill/tools/d3_hybrid_module.py` 1.0.2, which already carries type-3 bridge and emission deltas.
4. **The chart conflicts.**
   - `d3_hybrid_module` and the S7 check feed H3 source positions as zeros (`h3_zero_positions`; "the h3_thw_v1 positions have never run").
   - Codex's speech chart feeds (prompt_text_count + i, 0, c).
   - If one H3 unit is trained under both, D3b-H3 and the speech lanes teach it two position codes for the same type-3 rows.
5. **The amendments are not canonical.**
   - They exist only in Codex's successor prereg copy, the same pattern as B6 in the state document.
   - The canonical `PREREGISTRATION_OMNI_STAGE_D_v1.md` still reads "(none)".
6. **"Bidirectional" was not delivered as Cyril probably meant it.** No speech-to-speech model was fetched. The ASR+TTS pair still covers both directions, which is fine.
7. **All natural speech was blocked** on a date rule that is a research-information rule, not a corpus rule. This needs Cyril's word (decision S-R2).
8. **Root disk.** About 14 G of speech parents and venvs sit on root (4.1 + 8.6 + 0.9 G), against a 60 G floor and 71 G free.

---

## 3. How speech can enter OMNI1: the options

OMNI1 already has one audio channel: the H3 unit `h3_dit_L25_curve6` (267,889,176 params in D3a), inserted after trunk layer 31 (register record), with C15 type 3 (`audio_h3`, 32-wide latents, 40 Hz, 32 kHz stereo VAE).
- Its bridge was fitted at 24.13 dB against a 36.34 dB VAE round trip (`c15_real_v3`, STATE_2026-10-08).
- ACE owns music.
- Any new modality *unit* changes the OMNI1 structure. D1 is running on `OMNI1.clone-pre-D1` now, so a structural change would fork the whole D lineage.

| Option | What it is | Verdict |
|---|---|---|
| **A. Fold a speech unit now** (C16 4x coreset of one Qwen3-ASR encoder block + new C15 type; TTS codec band) | The step-1 recipe again | **Reject for stage D.** One block of a 24-layer encoder cannot do ASR: QVL32's single ViT block gave ungrounded vision at S8 (cos 0.363). It forks the base under a running D1, and it needs new capture, coreset, install, s7 check and registration before any training |
| **B. Train the existing H3 audio channel in D3 (D3e)** with teacher-generated data: Qwen3-TTS-1.7B makes the waveforms, Qwen3-ASR checks them, the H3 VAE turns them into latents | Codex's two objectives, moved onto `d3_hybrid_module` and scaled up | **Recommended.** No structural change; consistent with the H3 audio channel; the data path is already proven (n = 1 round trip); cheap teachers on idle P40s; off the critical path |
| **C. Stage T only** (design §5.7a) | Supervised audio-in during the curated finetune | Keep as the second pass (conversation-style speech), not the first: stage T is ~87-201 h and starts after D4 |
| **D. Structural speech unit after D4** (whole ASR encoder as an input unit + a C18-style speech-code band trained by KL to the TTS talker) | The architecture that actually gives strong speech, as in Qwen-Audio or Qwen3-Omni | **Hold as a decision line** if B misses its bar (§6). Sized in §7 |

---

## 4. Proposal: D3e speech lane on the H3 audio channel (option B)

### 4.1 Teachers and data (stage C-speech, P40s, now)

| Step | Tool / model | Card | Output (on Lexar, never root) | Time |
|---|---|---|---|---|
| C-s0 Runtime probe | New Pascal speech venv (E1). TTS-1.7B fp32 and ASR-0.6B/1.7B fp32, 16 utterances at batch 1 and batch 8; H3 VAE encode on a P40 | GPU1 | `p40_speech_rates.json`: s/utt, peak VRAM, RSS | ≈ 1 h |
| C-s1 Transcripts | `speech_corpus_build` (E2). 4,000 new speakable texts from curated answer rows (5-40 words, 10% long at 15-30 s) through the existing contamination screen, plus Codex's 201 admitted rows as seed | CPU | `speech/corpus/<sha>/texts.jsonl` | ≈ 0.5 h |
| C-s2 Synthesis | Qwen3-TTS-12Hz-1.7B-CustomVoice rev `0c0e3051`, all 9 voices round-robin, English. Other languages only if Cyril wants them | GPU1-3, one process per card | 24 kHz WAV + receipts. 4,000 × 4.67 s = 18,680 s of audio (d) → 0.90 GB | **Capped at 12 h wall.** At the measured 0.6B 3090 RTF of 4.0, 4,000 utterances on one card = 20.8 h (d). The P40 1.7B fp32 rate is UNVERIFIED; the count N comes from C-s0: N = 3 × 43,200 s / t_utt |
| C-s3 Admission | Qwen3-ASR-0.6B. Exact after case/punctuation → admit; else quarantine (Codex's rule) | GPU1 after its share of C-s2 | `admission.jsonl` | ≈ 1-3 h (a: 3× the 3090's 0.82 s) |
| C-s4 Latents | H3 audio VAE encode, 32 kHz stereo, full coverage, 1 in 20 re-encoded for the bitwise check | GPU2-3 | x0 `[2T,32]` fp32 only, ≈ 10,240 B per audio second → 0.19 GB (d). **Do not store** Codex's 6,144-wide band tensors: x_t is rebuilt from the sealed seed | UNVERIFIED (CPU measured 10.1 s per 4.2 s clip) |
| C-s5 ASR soft targets (optional, recommended) | Qwen3-ASR-1.7B teacher-forced on admitted audio + transcript; top-32 logprobs mapped Qwen3 vocab (151,936) → Chimera rows [0, 248,320) by identical byte string (K3's `xtok` rule) | GPU1 | ≈ 0.12M positions × 196 B ≈ 24 MB (d) | ≈ 1.5 h (a) |
| C-s6 Natural speech (only after ruling S-R2) | LibriSpeech train-clean-100: 1,394 train / 176 val / 193 test rows, using Codex's leak-repaired prompts; H3 encode as C-s4 | GPU2-3 | as C-s4 | ≈ 1-2 h (a) |

**Judge independence.**
- ASR-**0.6B** admits training rows.
- ASR-**1.7B** scores the held evaluation.
- Neither model grades data it also selected.

**Held set:**
- Codex's 15 speech + 16 non-speech held rows;
- 5% of the new synthetic texts, split by text group so the same sentence never crosses splits;
- LibriSpeech test (193), if S-R2 allows it.

### 4.2 Training (D3e, on the D3 engine, after D3d)

**Order:** ACE → QIMG → HY3D → H3 (D3b) → D3c music → D3d vision/video → **D3e-in → D3e-out** → D4.

Speech-out trains the same H3 unit and emission band that D3b-H3 trained, so D3e-out mixes in **20% H3 AV replay** from the D3b cache to protect video+audio generation.

| Lane | Objective (Codex's, ported) | Trainable | Positions (d) |
|---|---|---|---|
| D3e-in | `authored_audio_assistant_ce`: CE weight 1.0 on answer tokens, rows [0, 248,320); ASR KD weight 0.5 where C-s5 exists; σ = 0 clean latents in band 480:512; H3 reads prompt + audio only | H3 unit 267.9M, type-3 bridge cols + type-embed row 3, shared r16 LoRA | synthetic 4,000 × 411.0 = 1.64M; LibriSpeech 1,394 × ≈ 1,091 = 1.52M (a: 12.7 s mean); non-speech 8 × 189 ≈ 0. **≈ 3.17M** |
| D3e-out | `supervised_speech_rectified_flow`: x_t = (1−σ)x0 + σε in, x0 MSE on emission 224:256 in Codex's velocity form; 4 σ per utterance | H3 unit, type-3 bridge + emission slices, shared r16 LoRA | 4,000 × 4 × 405.5 = **6.49M**, + 20% H3 replay |

**Totals (d, rates from the design):**
- 1.64 + 1.52 + 6.49 = **9.65M** speech positions. The 20% H3 replay adds 6.49 / 0.8 − 6.49 = 1.62M, for **11.27M** in all.
- **Upside** (348 tok/s, both 3090s, full depth): 11.27M / 348 = 32,385 s = 9.0 h, × 1.3 = **11.7 h** (10.0 h without replay).
- **Planning** (151.24 tok/s, split cell): 74,518 s = 20.7 h, × 1.3 = **26.9 h** (23.0 h without replay).
- Without LibriSpeech (S-R2 refused), subtract 1.52M: about 9.8 h upside / 22.5 h planning.
- **Review at the 25% points:** 32-step Euler on 100 held transcripts ≈ 100 × 32 × 410 / 696 tok/s ≈ 0.5 h, plus H3 decode 18.8 s per clip on CPU ≈ 0.5 h, plus ASR-1.7B WER: about **1.2 h** per review (d).
- **D4 addition** (Codex's replay rule, scaled): 300 speech-in rows + 300 speech-out pairs ≈ 0.25M positions ≈ 0.2 h upside (d).
- Optimiser, warm-up, skip guard and 30-min checkpoints follow Codex's amendment, which matches D3 (lr 2e-5 unit / 1e-4 LoRA, AdamW (0.9, 0.95), clip 1.0, seed 20261013).

### 4.3 Gates (policy 2026-10-06: fit numbers are diagnostics)

**Functional (stop the lane):**
- non-finite loss or gradient;
- writes outside type-3 rows;
- held sampling decodes silent or constant audio (`audio_rms` = 0);
- empty transcripts;
- text identity: R-Q top-1 against the pre-D3e clone below about 0.99 on text-only inputs.

**Diagnostic (recorded, read by Cyril):**
- speech-in WER (ASR-1.7B reference transcript vs the model's answer) on held synthetic and LibriSpeech test;
- speech-out intelligibility = ASR-1.7B WER on decoded samples;
- Cyril's blind listen 1-5 beside the Qwen3-TTS original;
- the baseline is the D3-entry clone on the same items (expected ≈ 100% WER).

**Stop-spending line (design §2):** at 25% of D3e-out, if every held decode is noise (WER ≥ 90%), pause the lane and report. That trips decision S-R3 (option D).

**Promotion (rulings):** bank v1 content and all earlier modalities are no worse than D3d; otherwise D4 starts from the D3d clone and the speech regression is reported.

### 4.4 Prerequisites before the first D3e step
1. **One H3 position chart for type-3 rows across D3b and D3e**, recorded as an amendment.
   - My recommendation: Codex's measured 40 Hz chart `(text_count + i, 0, c)` for audio-only rows in both lanes, provided the D3b H3 AV caches are packed with the matching video-relative chart. Otherwise use the zero plan in both.
   - Test: same type-3 rows get the same rotary angles in both lane builders.
2. **`d3_hybrid_module` 1.1.0** with lanes `speech_in_ce` and `speech_out_flow`.
   - Parity test: loss equal to Codex's `speech_flow.py` / `audio_cache.py` on 3 replay tensors (fp32, rel ≤ 1e-6).
   - Codex's files are copied into `distill/tools/` with their source sha, never edited in Codex's worktree.
3. **Canonical prereg amendment** (below `## Amendments`, frozen body sha `c24d21e0…1370` unchanged) recording the D3e lanes, data shas, budgets and gates. Then run `stage2_training_receipt.py check-prereg`.

---

## 5. Engineering (CPU, Claude, can start now)

| # | Work | Hours (a) |
|---|---|---|
| E1 | Pascal speech venv at `/mnt/lexar/omni-campaign/runtimes/speech-pascal/`: torch 2.6.0+cu118 (the existing Pascal venv `/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260` has transformers 5.12.1 and **no `qwen_tts`**, m), qwen_tts 0.1.1, transformers 4.57.3 (Codex's TTS venv is torch 2.11.0+cu128, which I expect not to run on sm_61), qwen_asr. Pin the ASR repository: on-disk `7278e1e7` vs `-hf bcd2b5b7` | 1-2 |
| E2 | `speech_corpus_build.py` with tests: text selection, contamination screen reuse, group split, Codex seed rows | 2 |
| E3 | `speech_teacher_cache.py` with tests: `tts-gen`, `asr-admit`, `h3-encode`, `asr-topk` workers; sealed manifests; resumable; one card per process; MemAvailable ≥ 40 GiB gate; a `p40.claims` line per card; no P2P | 5 |
| E4 | `d3_hybrid_module` 1.1.0: two lanes, chart switch, parity against Codex's loss, H3 replay mixing, tests | 5 |
| E5 | Prereg amendment, rulings lines, `check-prereg` | 0.5 |

**Wall clock:**
- About 1.5 days of CPU engineering in parallel with Steps 1-5.
- P40 data generation of ≤ 12 h + ≈ 4 h, finishing long before D3. D3 needs D1 (≈ 2026-10-15 13-16Z at best) plus D2 (28-66 h).
- **Speech adds ≈ 11.7 h (upside) / 26.9 h (planning) to D3**, including the H3 replay and before the ≈ 1.2 h reviews, and ≈ 0.2 h to D4. It is not on the D2 critical path.

**Resources:**
- Host RAM: TTS/ASR processes are < 16 GB each (a). Three of them stay well inside the 251 GiB with D1 (≤ 128 GiB) and a K3 pass (≈ 72 GB); each gates on MemAvailable.
- Disk: everything goes to Lexar (646 G free); about 5 G in total.
- Copy the 1.7B pair and ASR-0.6B to `/mnt/lexar/omni-campaign/teachers/speech/` with a sha check. Leave Codex's root copies in place until the Step 7 go-ahead.
- GPU5 is not used. GPU0, GPU4 and GPU6 are not touched before D3.

---

## 6. Decisions for Cyril

| Id | Question | Recommendation |
|---|---|---|
| S-R1 | Speech in/out through the existing H3 audio channel as D3e (option B), with Qwen3-TTS-1.7B data and the Qwen3-ASR judges | **Go.** I treat it as covered by the standing GO ("[Cyril quote withheld]") and record it as an amendment unless he objects |
| S-R2 | Admit natural speech (LibriSpeech 2015, LJSpeech) despite the "2025-2026 information" rule that Codex applied to corpora | Admit. The rule is about research facts; the curated registry already trains on older rows. **His word is needed** |
| S-R3 | If D3e-out is still noise at 25%, or held speech-in WER is above about 30% after D3e: build the structural speech unit (option D) as an OMNI2 install after D4, before stage T | Decide when the numbers exist |
| S-R4 | PersonaPlex 7B (full duplex) and other speech-to-speech models | Not a stage-D teacher. Candidate data generator for spoken dialogue in stage T. Licence not re-read (UNVERIFIED) |
| S-R5 | The model in his 2026-10-14 Dot screenshot ("this might be a great speech model") | Ask him which model it is: it was not saved |
| — | Fish Audio Drama 3 | Not used. His own "use a qwen model" followed it in the same Dot thread, and its licence forbids training on its outputs |

---

## 7. Option D sizing (for S-R3; not planned)

**Speech-in unit**
- The whole Qwen3-ASR-1.7B audio encoder, 24 layers × (4 × 1,024² + 2 × 1,024 × 4,096) = **301,989,888 params** (d, excluding the conv front end and biases), installed whole as a non-causal input unit (not a 4x coreset).
- A new C15 type with a 1,024 → 5,120 bridge (5.2M params) and a 128-bin mel front end.
- Teacher: answer KD from the ASR decoder via `xtok`.

**Speech-out band**
- A C18-style band of 2,048 code-group-0 rows in embed + head: 2 × 2,048 × 5,120 = 21.0M (d).
- The 5-layer code predictor, about 5 × (4 × 1,024² + 3 × 1,024 × 3,072) = 68.2M (d).
- The 12 Hz codec decoder as the audio decoder (682 MB fp32 file).
- Trained by KL to the Qwen3-TTS talker's top-64, the way D3c trains the music band against the ACE LM.

**Cost:** a new capture, install, s7 check and registration (OMNI2), new stage-C caches, and a vocab-layout change (the ledger band offsets move). Roughly a step-1-sized job. It is only worth it if option B fails its bar.

---

## 8. What I could not verify

- Any P40 rate for TTS-1.7B, ASR or the H3 VAE (none measured).
- Whether the cu128 torch 2.11 speech venv runs on sm_61 (expected not).
- LibriSpeech mean duration (12.7 s assumed).
- The 186/192 train split count (only the 201/208 total was printed to me).
- The contents of `native-speech-flow-v2/` (permission denied).
- Every licence text.
- The model in the 2026-10-14 Dot screenshot.
- Any Chimera forward with audio: none exists.

**Evidence:**
- `reviews/2026-10-13-folding/recovery-preparation/{speech-*,h3-audio-proof-v3,h3-decoded-speech-asr-v1,speech-replay-final-v1}`
- `.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-image-edit-d4-oct13-3e25f2da3b32/proof_outputs/{chimera_speech_17b_comparison_oct13_v1,chimera_natural_speech_repair_oct13_v3}`
- `.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-trainer-readiness-oct13-3e25f2da3b32/proof_outputs/speech_native_prepare_v1`
- `.agent-worktrees/chimera-omni-5d4d4e1dcd/workers/codex/native-unified-modalities-oct14-7e9d6a43744c/receipts/native_speech_cpu_consumer_v20_v1/final_v2`
- Memory ids 892443, 892447, 892685, 892754, 1001821, 1002920
- Server `du`/`ls`/config reads at 14:12:27Z and later
