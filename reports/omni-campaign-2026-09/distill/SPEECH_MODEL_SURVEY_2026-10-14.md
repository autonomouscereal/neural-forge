# Speech model survey for Chimera Omni: ASR, TTS and speech-to-speech teachers (open weights, 2025-2026)

**Written:** 2026-10-14 by Claude, as a research pass for the D3e speech lane (`SPEECH_LANE_PROPOSAL_2026-10-14.md`, `STAGE_D_RECONCILED_PLAN_2026-10-14.md` §5).
**Scope:** research only. No GPU was used, nothing was downloaded to the server, and nothing on the server was touched. All model facts come from Hugging Face model cards and API metadata, live leaderboard data, and licence files, read on 2026-10-14.
**Marks:** (m) = read from a primary source (leaderboard data file, HF API, model card or licence text), with the source named; (d) = derived by arithmetic shown here; (a) = assumption; (s) = secondary source (a blog or aggregator), weaker; UNVERIFIED = not seen in a primary source.
**Why this pass:** Cyril, 2026-10-14: "[Cyril quote withheld]"

---

## 0. Bottom line

1. **ASR / speech-in teachers.**
   - Keep **Qwen3-ASR-1.7B** (already on disk).
   - Add **Cohere Transcribe 03-2026** (2B, Apache-2.0) as the independent judge. It is a Conformer encoder-decoder trained from scratch, so it is not a Qwen relative, and it has the best LibriSpeech numbers of any open model on the board.
   - Add **IBM Granite Speech 3.3 2B** (Apache-2.0, 5.36 avg WER, #3 open-weight on the current board) as a third, cheap English checker.
   - All three fit a P40 in fp32.
   - For *answers* to spoken questions, the best teacher is our own text teacher run on the gold transcript, not an audio LLM. On VoiceBench, ASR + strong-LLM cascades beat most end-to-end speech LLMs (§2.3).
2. **TTS / speech-out teachers.**
   - The highest human-preference open-weight TTS models of 2026 are Fish Audio Drama 3 (Oct 2026, Elo 1242), Breeze TTS 2 (Aug 2026, 1220), Fish S2 Pro (1116) and Higgs TTS 3. **All four licences forbid, or restrict to non-commercial use, using their outputs to train another speech or generative model.**
   - Use permissive teachers instead:
     - **Qwen3-TTS-12Hz-1.7B** (Apache-2.0, on disk): CustomVoice for the fixed voices, Base for cloning;
     - **VoxCPM2** (Apache-2.0, Apr 2026, 2.3B, 30 languages, 48 kHz output, the best speaker similarity among permissive cloners);
     - **MOSS-TTS v1.5** (Apache-2.0, May 2026, 8.5B, 31 languages) for multilingual and long-form text, or **Fun-CosyVoice3-0.5B** (Apache-2.0) as the cheap P40 option.
   - On the objective Open TTS Leaderboard (2026-10-09) these sit within about 0.1-0.3 WER points of the restricted models.
3. **Speech-to-speech / omni teachers (stage T conversation speech and duplex behaviour).**
   - **Qwen3-Omni-30B-A3B-Instruct** (Apache-2.0, VoiceBench 85.5, speech out in 10 languages; needs int4 to fit one card).
   - **MiniCPM-o 4.5** (Apache-2.0, Feb 2026, 9.4B, full duplex, 19 GB bf16, fits a 3090).
   - For turn-taking timing only: **NVIDIA PersonaPlex-7B** or **NemotronLabs VoiceChat 11B** (Aug 2026, OpenMDW-1.1, permissive, full duplex with tool calling). Their reasoning is weak: VoiceChat scores 58.1 on VoiceBench.
   - **Nemotron 3 Nano Omni 30B-A3B** (Apr 2026) is the best open speech-*understanding* model (VoiceBench 89.39), but it outputs text only and needs 66 GB in bf16.
4. **Fold verdict.**
   - **No speech-out unit should be folded.** The H3 audio channel already is Chimera's audio output path. A second codec/talker unit (Qwen3-TTS talker plus its 12 Hz codec) would make two competing audio output representations.
   - **One speech-in unit is worth folding: the whole Qwen3-ASR-1.7B audio encoder** (24 layers, d 1,024, non-causal, from the Qwen3-Omni AuT lineage, Apache-2.0, 52 languages), installed as an input unit with its own C15 type, as the vision encoders were.
   - It should be the whole encoder, not one block, and it should come *after* D4 (proposal option D). It is the trigger if D3e-in misses its bar.
   - No end-to-end S2S model is foldable as a unit: its speech ability lives in its own LLM backbone, and Chimera's trunk replaces that (§6.4).
5. **Cyril's own recordings** go in as (i) a cloning reference for an owner voice through the Apache-licensed cloners (Qwen3-TTS-Base, VoxCPM2), (ii) real-speech held evaluation, and (iii) natural speech-in training rows, labelled by two-ASR agreement. They stay local, `consent=owner` (§7).

---

## 1. Sources and how they were read (all 2026-10-14)

| Source | What it gave | Date of data | How read |
|---|---|---|---|
| Open ASR Leaderboard results, `hf-audio/open-asr-leaderboard-results/english_short_latest.csv` | 63 rows: avg WER, RTFx, licence, size, per-set WER | dataset `lastModified` 2026-10-09T11:54Z (m) | Downloaded the CSV and sorted it myself |
| Open ASR Leaderboard snapshot of 2026-05-06 (benchmarklist.com, Appen blog) | Older ranking: Granite Speech 4.1 2B #1 at 5.33 | May 2026 (s) | WebFetch. Only used to show that the board's metric changed |
| **Open TTS Leaderboard** `hf-audio/open_tts_leaderboard`, launched 2026-09-30 (blog `huggingface.co/blog/open-tts-leaderboard`) | WER on Seed-TTS + CV3-Eval (transcribed by Qwen3-ASR-1.7B-hf), RTFx on an H200, SIM (WavLM) for cloning, 9 languages | version "09-10-2026" (m) | Called the Space's Gradio API `/_refresh_leaderboard` directly: English and all-languages, with and without cloning |
| Artificial Analysis Speech Arena, provider-voice board (`artificialanalysis.ai/text-to-speech/leaderboard`) | Human-preference Elo, 97 models, 17 of them open-weight | releases up to Oct 2026 (s) | WebFetch (JS page summarised by a model). Corroborated by orcarouter.ai (Breeze 1,215, Aug 2026) and digitalapplied.com (2026-09-25 snapshot) |
| VoiceBench leaderboard (`matthewcym.github.io/VoiceBench/`) | Spoken-instruction QA, 47 systems | repo README commit 2026-09-10 (m) | WebFetch |
| HF API `api/models/<id>` for 126 repos | createdAt, lastModified, licence tag, safetensors param count, repo bytes, gating, sha | live (m) | Script; raw JSON saved (§11) |
| Model cards (README) for 43 repos and 8 licence files | Architecture, languages, benchmarks, runtime, licence clauses | live (m) | curl, then grep and reading |

**Two board caveats.**
- The Open ASR Leaderboard changed its test sets during 2026. It now uses "Cleaned" AMI/Earnings22/GigaSpeech/VoxPopuli, URGENT2024 and Voice Arena Monsoon. For example, Granite Speech 4.1 2B was 5.33 on 2026-05-06 and is 6.42 now. **Compare numbers only within one snapshot.** All ASR numbers below come from the 2026-10-09 file.
- The Open TTS Leaderboard scores WER with Qwen3-ASR. That may favour Qwen-family TTS slightly (a). SIM uses WavLM.

---

## 2. ASR / speech understanding (open weights, 2025-2026)

### 2.1 Open ASR Leaderboard, English short-form, data of 2026-10-09 (m)

Overall rank is among all 63 rows; proprietary APIs hold ranks 1-5, 7, 10-12, 14-15, 22, 25 and 31. Lower avg WER is better. RTFx is the leaderboard's throughput figure (higher is faster).

| Overall rank | Model (HF repo) | Avg WER | LS clean / other | RTFx | Params (m, HF API) | Langs | Licence | Released (HF createdAt) |
|---:|---|---:|---|---:|---:|---:|---|---|
| 6 | `microsoft/Phi-4-multimodal-instruct` | **5.02** | 1.35 / 3.45 | 163 | 5.57B | 8 | MIT | 2025-02-24 |
| 8 | `ibm-granite/granite-speech-3.3-8b` | 5.26 | 1.11 / 2.52 | 263 | 8.65B | 5 | Apache-2.0 | 2025-04 (rev 3.3.2, 2025-06-19) |
| 9 | `ibm-granite/granite-speech-3.3-2b` | 5.36 | 1.21 / 2.85 | 509 | 3.01B | 5 | Apache-2.0 | 2025-04-28 |
| 13 | `Qwen/Qwen3-ASR-1.7B-hf` | 5.75 | 1.26 / 2.94 | 836 | 2.04B (`-hf`); 2.35B original | 52 (30 + 22 Chinese dialects) | Apache-2.0 | 2026-01-28 (orig.), 2026-06-26 (`-hf`) |
| 16 | `HojoAI/Hojo-ASR-V1` (Qwen3-Omni encoder + Qwen3-4B) | 6.07 | 1.30 / 3.24 | 73 | 5.18B | 2 | Apache-2.0 | 2026-05-24 |
| 17 | `TheStageAI/thewhisper-large-v3-turbo` | 6.07 | 1.41 / 3.20 | 1,184 | 0.81B | 28 | CC-BY-4.0 | 2025-10-27 |
| 18 | `bosonai/higgs-audio-v3-stt` (Whisper enc + Qwen3 dec) | 6.19 | 1.08 / 2.61 | 110 | 2.68B | 1 | Apache-2.0 | 2026-03-25 |
| 19 | `CohereLabs/cohere-transcribe-03-2026` | 6.26 | **0.97 / 2.05** | 924 | 2.07B | 14 | Apache-2.0 (gated: contact share) | 2026-03-24 |
| 20 | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | 6.29 | 1.66 / 4.16 | 387 | 0.91B | 50 | Apache-2.0 | 2026-05-19 |
| 23 | `nvidia/canary-qwen-2.5b` | 6.37 | 1.23 / 2.63 | 882 | 2.56B | en | CC-BY-4.0 | 2025-06-26 |
| 24 / 32 | `nvidia/parakeet-tdt-0.6b-v3` (fast / std) | 6.38 / 6.51 | 1.46 / 3.08 | 13,305 | 0.63B | 26 | CC-BY-4.0 | 2025-08-04 |
| 26 / 35 | `ibm-granite/granite-speech-5.0-470m-turboctc` (-nc / Apache) | 6.41 / 6.84 | 1.45 / 2.67 | ~21,000 | 0.47B | en | CC-BY-NC-SA / Apache-2.0 | 2026-08-04 |
| 27 | `ibm-granite/granite-speech-4.1-2b` | 6.42 | 1.03 / 2.19 | 547 | 2.31B | 6 | Apache-2.0 | 2026-04-16 |
| 34 | `Qwen/Qwen3-ASR-0.6B-hf` | 6.60 | 1.70 / 4.01 | 750 | 0.78B | 52 | Apache-2.0 | 2026-06-26 |
| 36 | `nvidia/nemotron-speech-streaming-en-0.6b` | 6.89 | 1.89 / 4.35 | 1,114 | 0.62B | en | NVIDIA Open Model License | 2025-12-17 |
| 37 | `mistralai/Voxtral-Small-24B-2507` | 6.90 | 1.23 / 2.80 | 101 | 24.3B | 8 | Apache-2.0 | 2025-07-01 |
| 47 | `microsoft/VibeVoice-ASR-HF` (60-min single pass, diarised) | 8.23 | 1.56 / 4.95 | 218 | 8.33B | 50 | MIT | 2026-01/03 |
| 51 | `facebook/omniASR-LLM-7B-v2` | 8.78 | 1.35 / 3.25 | 130 | 7.8B | 1,676 | Apache-2.0 | UNVERIFIED (HF API 401) |
| 52 | `mistralai/Voxtral-Mini-4B-Realtime-2602` (streaming) | 9.01 | 1.62 / 4.94 | 103 | 4.43B | 13 | Apache-2.0 | 2026-01-21 |
| 55 | `nvidia/nemotron-3.5-asr-streaming-0.6b` | 10.51 | 2.83 / 6.79 | 1,348 | 0.64B | 40 | OpenMDW-1.1 | 2026-05-15 |
| 59 | `google/gemma-4-E4B-it` (general omni LLM) | 12.66 | 3.85 / 9.97 | 161 | 8.0B | 140 | Apache-2.0 | 2026-03-02 |

**Not on the board but relevant (m, model cards):**
- `XiaomiMiMo/MiMo-V2.5-ASR`: 7.6B, MIT, 2026-04-23. Mandarin/English, dialects, code-switching. Its card claims a lead on AMI.
- `microsoft/VibeVoice-ASR-Streaming-7B`: 2026-09-02, MIT.
- `bosonai/higgs-audio-v3-8b-stt-v2`: 8.9B, Apache-2.0, Whisper-large-v3 encoder + Qwen3-8B. Card: LS clean 1.25 / other 2.38.
- `FunAudioLLM/Fun-ASR-Nano-2512`: Apache-2.0. Board avg 7.76.
- `zai-org/GLM-ASR-Nano-2512`: MIT, 2.26B.

### 2.2 Reading of the ASR field
- In the 2026-10-09 data, Phi-4-multimodal (2025) and Granite Speech 3.3 (2025) still lead the open-weight English board. Qwen3-ASR-1.7B is the best open model **with broad language coverage** (52). Cohere Transcribe has the best LibriSpeech numbers of any open model (0.97 / 2.05) but only 14 languages, and it needs the language specified up front.
- **2026 newcomers that matter for us:**
  - Qwen3-ASR (Jan);
  - Cohere Transcribe (Mar);
  - Granite Speech 4.1 / 5.0 (Apr / Aug);
  - Higgs Audio v3 STT (Mar-Apr);
  - MOSS-Transcribe-Diarize (May; 0.9B, 50 languages, with diarisation);
  - VibeVoice-ASR (Jan / Sep; 60-minute single pass with speakers and timestamps);
  - Voxtral Realtime (Jan);
  - Nemotron 3.5 ASR streaming (May).
- **Self-preference bias.** Cloud thread 01a10f78 picked `Qwen3-ASR-1.7B-hf` (rev `bcd2b5b7`). The Open TTS Leaderboard uses the same model as its WER judge. If our admission ASR and our TTS teacher are both Qwen-family, a Qwen-on-Qwen WER can be optimistic (a). Cohere Transcribe is architecturally independent and is the best second judge.

### 2.3 Speech understanding beyond transcription (VoiceBench, spoken-instruction QA, page read 2026-10-14 (m))

| VoiceBench overall | System | Weights | Note |
|---:|---|---|---|
| 89.39 | NVIDIA Nemotron 3 Nano Omni 30B-A3B | open | Apr 2026, text output only, NVIDIA Open Model Agreement, 33.0B params, 66 GB bf16 |
| 88.80 | Qwen3-Omni-30B-A3B-Thinking | open | text output only |
| 87.80 | Whisper-v3-large + GPT-4o (cascade) | closed | |
| 85.50 | Qwen3-Omni-30B-A3B-Instruct | open | speech out |
| 79.23 | Parakeet-TDT-0.6B-v2 + Qwen3-8B (cascade) | open | |
| 76.91 | Kimi-Audio-7B-Instruct | open | |
| 71.23 | MiniCPM-o (older 2.6 entry) | open | |
| 64.32 | Phi-4-multimodal | open | best open ASR, weak at answering |
| 58.10 | Nemotron 3 VoiceChat (V1) | open | full duplex |
| 29.51 | Moshi | open | |

Two further rows, BR-Voice-Reasoner (90.46) and LFG-3 (89.88), are listed as open, but their HF repos are tiny community uploads (`brgroup/BR-Voice-Reasoner`, `glenn2/LFG-3`, 3 and 5 likes). I did not check them and do not recommend them.

**What this means for D3e-in.** A good ASR feeding a strong text LLM is about as good as the best end-to-end audio LLMs. A plain cascade (0.6B Parakeet + Qwen3-8B) beats Kimi-Audio. Chimera's trunk is far stronger than Qwen3-8B. So the right answer target for a spoken question is **Chimera's own text teacher answering the gold transcript**. The ASR teachers supply transcripts, admission checks and optional ASR soft targets. That is what the proposal's C-s5 already does (a).

---

## 3. TTS / expressive speech generation and voice cloning

### 3.1 Open TTS Leaderboard (objective), version 09-10-2026 (m, Gradio API)

**English, no cloning:** average WER over Seed-TTS-en + CV3-en; RTFx on an H200.

| Rank | Model | WER | RTFx | Size (B) | Licence | Langs | Training on outputs allowed? |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | `hexgrad/Kokoro-82M` | 1.46 | 71 | 0.08 | Apache-2.0 | 8 | yes (no cloning) |
| 3 | `Supertone/supertonic-3` | 1.54 | 47 | 0.10 | OpenRAIL-M | 31 | use-restrictions only (a) |
| 4 | `fishaudio/s2-pro` | 1.58 | 0.4 | 4.95 | Fish Audio Research | 83 | **No** (§9) |
| 5 | `k2-fsa/OmniVoice` | 1.61 | 38 | 0.81 | CC-BY-NC (weights) | 646 | NC only |
| 6 | `kyutai/pocket-tts` | 1.62 | 9 | 0.10 | CC-BY-4.0 | 6 | yes |
| 7 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (RL) | 1.68 | 6 | 0.11* | Apache-2.0 (HF tag; the board says CC-BY-4.0) | 9 | yes |
| 8 | `ResembleAI/chatterbox` | 1.70 | 2 | 0.80 | MIT | en (+23 multilingual variant) | yes |
| 9 | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.71 | 13 | 1.92 | Apache-2.0 | 10 | yes |
| 10 | `BreezeBlue/Breeze-TTS-2` | 1.72 | 3 | 3.47 | BreezeBlue research/NC | 2 | **Non-commercial derivative only** |
| 11 | `LiquidAI/LFM2.5-Audio-1.5B` | 1.77 | 2 | 1.53 | LFM Open v1.0 ($10M revenue cap) | 1 | yes under the cap |
| 12 | `microsoft/VibeVoice-Realtime-0.5B` | 1.82 | 3 | 1.36 | MIT | 10 | yes |
| 13 | `nvidia/magpie_tts_multilingual_357m` | 2.04 | 13 | 0.31 | NVIDIA Open Model License | 9 | yes (a) |
| 17 | `IndexTeam/IndexTTS-2.5` | 2.23 | 2 | 1.67 | bilibili | 5 | only into "non-commercial AI models" |
| 19 | `OpenMOSS-Team/MOSS-TTS` | 2.33 | 3 | 8.49 | Apache-2.0 | 20 (v1.5: 31) | yes |
| 20 | `tencent/AuK` | 2.48 | 7 | 1.53 | MIT | 2 | yes |
| 21 | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (talker) | 2.54 | 11 | 35.3 | Apache-2.0 | 10 out | yes |
| 23 | `mistralai/Voxtral-4B-TTS-2603` | 2.71 | 47 | 4.0 | CC-BY-NC-4.0 | 9 | NC only |
| 24 | `bosonai/higgs-tts-3-4b` | 3.23 | 54 | 4.65 | Boson research/NC | 102 | **No** |
| 25 | `openbmb/VoxCPM2` | 3.38 | 2 | 2.38 | Apache-2.0 | 30 | yes |
| 28 | `sesame/csm-1b` | 3.71 | 8 | 1.77 | Apache-2.0 | en | yes |
| 30-31 | `vibevoice/VibeVoice-1.5B-hf` / `-7B-hf` | 5.76 / 6.02 | 23 / 15 | 2.7 / 9.3 | MIT | 2 | yes |

\* The board's "size" for CosyVoice3 counts the LM only. The repo is 9.75 GB (m, HF API).

**English, with voice cloning (reference audio given):** WER and SIM ×100.

| Rank | Model | WER | SIM | Licence |
|---:|---|---:|---:|---|
| 1 | `bosonai/higgs-tts-3-4b` | 1.59 | 64.7 | research/NC, **no training** |
| 2 | `k2-fsa/OmniVoice` | 1.69 | 72.0 | CC-BY-NC |
| 3 | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.81 | 69.3 | Apache-2.0 |
| 4 | `openbmb/VoxCPM2` | 1.82 | 70.9 | Apache-2.0 |
| 5 | `fishaudio/s2-pro` | 1.82 | 62.0 | research, **no training** |
| 6 | `kyutai/pocket-tts` | 1.90 | 46.7 | CC-BY-4.0 |
| 7 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 1.96 | 58.3 | Apache-2.0 |
| 8 | `tencent/AuK` | 2.07 | **75.1** | MIT |
| 10 | `OpenMOSS-Team/MOSS-TTS` | 2.15 | 64.9 | Apache-2.0 |
| 12 | `BreezeBlue/Breeze-TTS-2` | 2.29 | 65.5 | research/NC |

**All 9 languages** (en, zh, fr, es, de, it, ja, ko, ru):

| Rank | Model | Without cloning: avg WER/CER | With cloning: WER/CER, SIM |
|---:|---|---:|---|
| 1 | OmniVoice | 3.25 | 3.52, 74.8 |
| 2 | fishaudio/s2-pro | 3.68 | 3.98, 67.9 |
| 3 | Qwen3-TTS-1.7B (CustomVoice / Base) | 4.11 | 3.92, 72.9 |
| 4 | MOSS-TTS | 4.30 | 5.04, 70.6 |
| 5 | VoxCPM2 | 4.36 | 4.18, **75.2** |
| — | higgs-tts-3-4b | 8.07 | 3.74, 69.1 |

### 3.2 Human preference: Artificial Analysis Speech Arena, provider voices (s, read 2026-10-14)

Open-weight entries only (17 of 97). Overall leader: ElevenLabs v4 Turbo, 1,328.

| Overall rank | Model | Elo | Released | Licence for training on outputs |
|---:|---|---:|---|---|
| 10 | Fish Audio Drama 3 Preview (`fishaudio/drama-3`) | 1,242 | Oct 2026 | **Forbidden** |
| 11 | Breeze TTS 2 (`BreezeBlue/Breeze-TTS-2`) | 1,220 | Aug 2026 | Non-commercial only |
| 35 | Fish Audio S2 Pro | 1,116 | Mar 2026 | **Forbidden** |
| 41 | Step-Audio-EditX (`stepfun-ai/Step-Audio-EditX`, 3.53B) | 1,096 | Feb 2026 | Apache-2.0 (GitHub licence API; HF repo untagged) |
| 45 | Voxtral TTS (`mistralai/Voxtral-4B-TTS-2603`) | 1,088 | Mar 2026 | CC-BY-NC-4.0 |
| 55 | NVIDIA Magpie-Multilingual 357M | 1,064 | Feb 2026 | NVIDIA Open Model License |
| 56 | Kokoro 82M v1.0 | 1,061 | Jan 2025 | Apache-2.0 |
| 65 | Maya1 (3.3B) | 1,046 | 2025 | Apache-2.0 |
| 67 | OpenAudio S1 Mini | 1,043 | 2025 | CC-BY-NC-SA |
| 69 | Higgs Audio V3 TTS (`bosonai/higgs-tts-3-4b`) | 1,037 | Jun 2026 | **Forbidden** |
| 76 | Chatterbox | 1,025 | May 2025 | MIT |
| 82 | VibeVoice 1.5B | 956 | Aug 2025 | MIT |

Neither Qwen3-TTS open weights nor VoxCPM2 nor MOSS-TTS nor CosyVoice3 is on this arena. The "Qwen3 TTS" row (931, Jan 2026) is listed there as not open-weight, i.e. Alibaba's hosted voice, so it says nothing firm about the open 1.7B checkpoint (a).

**The licence trap, stated plainly.** The four open models with the best human preference in 2026 (Drama 3, Breeze 2, S2 Pro, Higgs 3) all reserve training use. Permissive models are close on intelligibility (WER within about 0.1-0.3) and lead on speaker similarity (VoxCPM2 75.2, AuK 75.1). Their human-preference standing is unmeasured or lower. Cyril has said "[Cyril quote withheld]". The clauses are recorded in §9 so that call is made knowingly. My recommendation stays with the permissive set, because the gap we can measure is small.

### 3.3 Card facts for the TTS shortlist (m)

- **Qwen3-TTS-12Hz (Jan 2026).**
  - 0.6B / 1.7B in three kinds: Base (3 s clone), CustomVoice (9 voices + instruct), VoiceDesign (voice from a text description).
  - 10 languages; 12.5 Hz codec with 16 codebooks; 24 kHz mono; streaming.
  - On disk: 1.7B CustomVoice rev `0c0e3051`. Base (`fd4b2543`, 4.54 GB) and VoiceDesign (`5ecdb673`, 4.52 GB) are not yet downloaded.
- **VoxCPM2 (openbmb, 2026-04-03, 2.29B).**
  - Tokenizer-free diffusion-autoregressive model; 30 languages plus 9 Chinese dialects; **48 kHz output**.
  - Voice design, controllable cloning, transcript-continuation cloning.
  - Card: RTF about 0.3 on an RTX 4090. "Apache-2.0 … free for commercial use."
- **MOSS-TTS v1.5 (OpenMOSS, 2026-05-25, 8.49B).** 31 languages, zero-shot cloning, long-form, token-level duration control, pinyin/IPA, `[pause Xs]` markers, Apache-2.0. Siblings: MOSS-TTSD v1.0 (dialogue, 8.36B), MOSS-TTS-Realtime (2.33B).
- **Fun-CosyVoice3-0.5B-2512 (2025-12-11).** 9 languages, 18+ Chinese dialects, instruct control (emotion/speed/volume), Apache-2.0. Its Seed-TTS numbers are cited in Qwen3-Omni's card: test-zh 0.71 / test-en 1.45.
- **tencent AuK (2026-09-09, 1.5B, MIT).**
  - One model for zero-shot TTS, instruct TTS, content editing (replace/insert words), pitch/speed/volume edits, paralinguistic editing, enhancement and separation. English and Chinese.
  - AuK-Flash is the 4-step distilled variant.
  - Useful later for **editing** teacher waveforms instead of regenerating them.
- **Fish Audio Drama 3 (2026-10-01).**
  - 35.9B MoE with 3B active per token, FP8 experts, 41.5 GB repo; int4 build `fishaudio/drama-3-int4` is 24.0 GB.
  - Up to 16 speakers per text, about 83 languages.
  - Fish Audio Community License v1.0 (2026-10-13).
  - Tech report arXiv 2610.18010 (not read).
  - Reference and evaluation only.

---

## 4. End-to-end speech-to-speech / omni (speech in and out)

| Model (repo) | Released (m) | Params (m) | Speech out | Langs (speech) | Licence | Evidence of quality (m unless marked) |
|---|---|---:|---|---|---|---|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | 2025-09-20 | 35.26B (30B-A3B thinker + talker) | yes, streaming | in 19 / out 10 | Apache-2.0 | VoiceBench 85.50; LS 1.22/2.48 (card); Seed-TTS en WER 1.39 (card); Open TTS board en 2.54 |
| `openbmb/MiniCPM-o-4_5` | 2026-02-03 | 9.37B (SigLIP2 + Whisper-medium + CosyVoice2 + Qwen3-8B) | yes, **full duplex**, voice cloning by audio system prompt | en, zh | Apache-2.0 | Card: LS 1.4/2.8; Seed-TTS en 2.38 (SIM 64.9); LongTTS-en 3.37 (Qwen3-Omni 17.33); bf16 19.0 GB, int4 11.0 GB |
| `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` | 2026-07-29 (card: release 2026-08-03) | 11.1B (FastConformer + Nemotron Nano v2 9B + TTS decoder) | yes, full duplex, **tool calling** | en | OpenMDW-1.1 | FullDuplexBench: turn-taking latency 448 ms, interruption TOR 1.0; VoiceBench 58.10. About 550k h of training audio |
| `nvidia/personaplex-7b-v1` | 2026-01-15 | 8.37B (Moshi-based) | yes, full duplex, persona text prompt + voice prompt | en | NVIDIA Open Model License (+CC-BY-4.0) | FDB interruption TOR 0.95, latency 0.24 s; turn-taking TOR 0.908 |
| `kyutai/personaplex-rl-seamless` | 2026-06-02 | 8.37B | yes | en | CC-BY-NC-4.0 delta + NVIDIA OML | RL for interactivity; no numbers on the card |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | 2026-04-20 | 33.0B | **no** (text out) | — | NVIDIA Open Model Agreement | VoiceBench **89.39**, best open |
| `moonshotai/Kimi-Audio-7B-Instruct` | 2025-04-25 | 9.77B | yes | en, zh | MIT (Qwen2.5 parts Apache) | VoiceBench 76.91 |
| `stepfun-ai/Step-Audio-2-mini` | 2025-08-28 | 8.32B | yes | en, zh | Apache-2.0 | card benchmarks are in images (not transcribed) |
| `stepfun-ai/Step-Audio-R1.1` | 2026-01-14 | 33.5B | "thinking while speaking" (card); speech out in the open weights UNVERIFIED | — | Apache-2.0 | card claims top rank on an AA audio benchmark (images) |
| `XiaomiMiMo/MiMo-Audio-7B-Instruct` | 2025-09-18 | 8.02B | yes | en, zh | MIT | — |
| `FunAudioLLM/Fun-Audio-Chat-8B` | 2025-12-23 | 9.45B | yes | en, zh | Apache-2.0 | card: "~24 GB for inference" |
| `tencent/Covo-Audio-Chat` | 2026-03-16 | 8.40B | yes (an FD variant exists) | — | **academic use only** | — |
| `LiquidAI/LFM2.5-Audio-1.5B` | 2025-12-18 | 1.47B | yes (Mimi-compatible codes) | en (+JP variant 2026-05) | LFM Open v1.0 | Open TTS board en WER 1.77 (rank 11) |
| `inclusionAI/Ming-flash-omni-2.0` | 2026-02-10 | 104B (100B-A6B) | yes (speech + audio + music) | — | MIT | too large for our cards |
| `meituan-longcat/LongCat-Flash-Omni` | 2025-10-23 | 561B | yes | — | MIT | too large |
| `kyutai/glm-4-voice-of-reason-9b` | 2026-08-04 | 9.54B | yes | — | GLM-4-Voice licence | GSM8K 0.706 spoken |
| Qwen3.5-Omni | announced 2026-03-30 (s) | 30B-A3B (s) | yes | 74 in / 29 out (s) | — | **No open weights found**: HF API returned 401 for `Qwen/Qwen3.5-Omni` and `Qwen/Qwen3.5-Omni-Light`; tech report arXiv 2604.15804 lists only a demo |
| MiniMax speech (Speech 2.6 / 2.8) | 2025-10 / 2026-02 | — | — | — | API only (s) | No open-weight MiniMax speech model exists. H3 is our joint audio-video source, not a speech model |

---

## 5. Hardware fit (our cards)

**Rules used (a: standard hardware facts).**
- P40 (Pascal sm_61): no bf16, and FP16 throughput is a small fraction of FP32, so **run fp32**: 4 B/param, ≤ about 5B params on a 24 GB card with room for activations. No FlashAttention-2.
- RTX 8000 (Turing sm_75, 46 GB): fp16 tensor cores, bf16 not native, no FlashAttention-2. fp32 up to about 10B or fp16 up to about 20B.
- 3090 (Ampere): bf16 and FlashAttention-2 are fine; up to about 10B in bf16.
- FP8 and NVFP4 checkpoints need sm_89+, so they run on none of our cards.

Weight memory = params × bytes (d). It excludes activations and KV cache.

| Model | bf16 / fp32 weights (d) | P40 (fp32) | 3090 (bf16) | RTX 8000 | Role |
|---|---|---|---|---|---|
| Qwen3-ASR-1.7B (orig 2.35B / -hf 2.04B) | 4.7 / 9.4 GB | **yes** | yes | yes | ASR KD + held judge |
| Qwen3-ASR-0.6B | 1.9 / 3.8 GB | yes | yes | yes | admission (Codex: 0.82 s/utt on a 3090) |
| Cohere Transcribe (2.07B) | 4.1 / 8.3 GB | **yes** | yes | yes | independent judge |
| Granite Speech 3.3 2B (3.01B) | 6.0 / 12.0 GB | yes | yes | yes | third checker (en, fr, de, es, pt) |
| Granite Speech 3.3 8B (8.65B) | 17.3 / 34.6 GB | no | yes | yes (fp32) | best Apache English ASR |
| Phi-4-multimodal (5.57B) | 11.1 / 22.3 GB | marginal | yes | yes | best open avg WER |
| Qwen3-TTS-1.7B (1.92B + 0.17B codec) | 4.2 / 8.4 GB | **yes** | yes | yes | TTS teacher (Codex's 0.6B: RTF 4.0 on a 3090, batch 1) |
| VoxCPM2 (2.29B) | 4.6 / 9.2 GB | yes (speed UNVERIFIED) | yes | yes | TTS / clone teacher |
| Fun-CosyVoice3-0.5B | ~1-2 / ~3-4 GB (a) | yes | yes | yes | cheap TTS teacher |
| MOSS-TTS v1.5 (8.49B) | 17.0 / 34 GB | no | yes | yes (fp32) | multilingual / long-form teacher |
| tencent AuK (1.53B) | 3.1 / 6.1 GB | yes | yes | yes | editing teacher |
| Fish Drama 3 (35.9B, FP8 experts) | 41.5 GB on disk; int4 24.0 GB | no | no (FP8; int4 24.0 GB is over capacity) | int4 maybe (UNVERIFIED) | reference only |
| Qwen3-Omni-30B-A3B-Instruct | card minimum 78.85 GB bf16; int4 about 20.3 GB (MiniCPM card) | no | int4 yes | int4 yes (kernel support UNVERIFIED) | S2S teacher |
| MiniCPM-o 4.5 (9.37B) | 19.0 GB measured bf16; 11.0 GB int4 | no | **yes** | fp16 yes | duplex S2S teacher |
| NemotronLabs VoiceChat 11B | 22.2 GB (d) | no | tight | fp16 yes | duplex timing teacher |
| PersonaPlex-7B (8.37B) | 16.7 GB (d) | no | yes | fp16 yes | duplex timing teacher |
| Nemotron 3 Nano Omni (33.0B) | 66 GB bf16 | no | no | no (only split across RTX 8000 + 3090, UNVERIFIED) | speech-in answer reference |

**Current availability** (from the task brief):
- 3090s are busy with D1 until about 2026-10-15 22:00Z;
- the RTX 8000 is on the K3 queue for days;
- GPU1/2 are on tool-call self-generation until about 2026-10-15 03:00Z;
- GPU3 is shared through `$OC/distill/queues/p40.claims`.

**The P40-only teacher set is therefore what can run now:** Qwen3-ASR (0.6B and 1.7B), Cohere Transcribe, Granite Speech 3.3 2B, Qwen3-TTS-1.7B, VoxCPM2, CosyVoice3, AuK. Omni and S2S teachers wait for a 3090.

---

## 6. Recommendations

### 6.1 ASR / speech-in teachers (top 3)

1. **Qwen3-ASR-1.7B**. Pin one revision: `-hf` `bcd2b5b7` runs on stock transformers; the original `7278e1e7` is on disk.
   - Roles: ASR soft-target KD (C-s5) and held-set WER judge.
   - Why: 52 languages, Apache-2.0, 5.75 avg WER (#4 open on the 2026-10-09 board, #1 among open models with more than 8 languages).
   - Its encoder is also the fold candidate (§6.4).
2. **Cohere Transcribe 03-2026** (`CohereLabs/cohere-transcribe-03-2026`, 2.07B, Apache-2.0, sha `d552dadc`).
   - Roles: independent admission and held judge. It breaks the Qwen-judges-Qwen loop.
   - Why: best open LibriSpeech numbers (0.97 / 2.05), so it is the right judge for the LibriSpeech held rows. 14 languages; the language must be given.
   - Gated: needs an HF login and contact sharing. Cyril or a token has to accept once.
3. **Granite Speech 3.3 2B** (`ibm-granite/granite-speech-3.3-2b`, 3.01B, Apache-2.0, sha `4ac2f02f`).
   - Role: third checker for majority-vote labels on Cyril's own recordings and on natural speech.
   - Why: 5.36 avg WER (#3 open) and it fits a P40.
   - Upgrade path: Granite 3.3 8B or Phi-4-multimodal on a 3090 when one is free.
- **Answer teacher for speech-in rows:** the text teacher already used in D1-D3, run on the gold or admitted transcript (§2.3 evidence).
  - Optional, for paralinguistic and audio-event questions only: Qwen3-Omni-30B-A3B-Thinking or Nemotron 3 Nano Omni, run later on a 3090 in int4.

### 6.2 TTS / speech-out teachers (top 3, permissive only)

Targets are waveforms → resample to 32 kHz stereo → H3 audio VAE → x0 latents, as in the proposal's C-s4. A teacher that produces 24 kHz or 48 kHz audio is fine because the H3 VAE resamples. VoxCPM2's 48 kHz output loses nothing on the way down (a).

1. **Qwen3-TTS-12Hz-1.7B.**
   - CustomVoice (on disk) for the fixed synthetic voices.
   - Also download **Base** (`fd4b2543`, 4.54 GB) for cloning Cyril's voice, and optionally VoiceDesign (`5ecdb673`, 4.52 GB) to create many distinct voices from text so speech-out does not overfit 9 voices.
   - Standing: Open TTS board en 1.71, multilingual #3, cloning #3 (SIM 69.3).
2. **VoxCPM2** (`openbmb/VoxCPM2`, 2.29B, sha `32279eff`, 4.96 GB).
   - Best permissive speaker similarity (75.2 multilingual, cloned); 30 languages; 48 kHz; voice design.
   - Use it as the second voice family so Chimera does not learn one vendor's prosody.
3. **MOSS-TTS v1.5** (`OpenMOSS-Team/MOSS-TTS-v1.5`, 8.49B, sha `cdd3b911`, 17.0 GB) once a 3090 is free: multilingual and long-form (the 15-30 s tail in the proposal's length mix).
   - On P40s only: **Fun-CosyVoice3-0.5B-2512** (sha `29e01c4e`, 9.75 GB repo).
- **Helper:** tencent **AuK** (MIT) to *edit* admitted clips (word swaps, pace, paralinguistics), which gives paired data cheaply.
- **Admission rule stays:** an independent ASR must transcribe each clip exactly. With two TTS families, admit a clip when Qwen3-ASR-0.6B *and* Cohere Transcribe both pass (a).
- **Keep out of training:** Fish Drama 3, Fish S2 Pro, Higgs TTS 3, Breeze TTS 2. IndexTTS-2.5, Voxtral TTS and OmniVoice are non-commercial only. Drama 3 can serve as a **listening reference** for the 25% review if Cyril wants an A/B (it is research use).

### 6.3 Speech-to-speech teachers (stage T and later)

1. **Qwen3-Omni-30B-A3B-Instruct** (Apache-2.0, sha `26291f79`, 70.5 GB bf16 repo; int4 for one card).
   - Role: generates spoken *answers* to spoken questions (thinker + talker) for conversation-style pairs.
   - Its voices are fixed; the multilingual output covers 10 languages.
2. **MiniCPM-o 4.5** (Apache-2.0, sha `503e7542`, 20.05 GB).
   - Role: full-duplex and turn-taking behaviour, and long speech (LongTTS-en 3.37 against Qwen3-Omni's 17.33).
   - Fits one 3090 in bf16.
3. **NemotronLabs VoiceChat 11B** (OpenMDW-1.1, sha `443794ea`, 44.4 GB fp32 on disk) **or PersonaPlex-7B** (NVIDIA OML, gated).
   - Role: duplex timing labels only (barge-in, backchannel, latency).
   - Their content quality is low (VoiceChat VoiceBench 58.1), so never use their words as targets.

### 6.4 Is any of these good enough to FOLD as a source unit?

**Speech-out: no.**
- Chimera already has an audio output channel: the H3 unit `h3_dit_L25_curve6` with C15 type 3, 40 Hz latents, 32 kHz stereo VAE. Its bridge is fitted at 24.13 dB, and a decoded H3 round trip kept the words of an utterance exactly (n = 1).
- Folding a TTS talker (Qwen3-TTS 1.7B talker + 12.5 Hz 16-codebook codec, or VoxCPM2) would add a **second audio output representation** with its own codec. The trunk would then have to learn which of two audio heads to drive, and video+audio generation would no longer share the speech path.
- Distilling TTS waveforms into H3 latents (D3e-out) gives the same capability without changing the structure.
- The best open TTS models are also the training-restricted ones (§3.2), and a fold is training use.

**Speech-in: one candidate, and only as a whole encoder, after D4.**
- The **Qwen3-ASR-1.7B audio encoder** is the right unit. Measured from config.json: 24 layers, d_model 1,024, 16 heads, FFN 4,096, 128 mel bins, non-causal (§2.1 of the proposal).
- Why this encoder:
  - It is small (about 0.3B (d) = 24 × (4·1,024² + 2·1,024·4,096) ≈ 302M, excluding conv stem and projection), so it fits beside the trunk in the Q4 3090 budget.
  - It is Apache-2.0.
  - It covers 52 languages and dialects.
  - It descends from the Qwen3-Omni AuT encoder, so the trunk's Qwen lineage has seen this representation style (a).
  - It sits behind the #4 open English ASR (5.75).
- What it would replace: the speech-in path through H3 latents. H3 latents are tuned for reconstruction (SNR 10.85 dB on one clip), not for phonetic content. That is the main risk to D3e-in.
- **How:** a new C15 type id (`audio_asr`, 16 kHz mono mel → encoder → projection into the trunk), whole encoder, the same recipe as the vision encoder units. It is not a C16 single-block coreset, which the proposal rejected (QVL32 single-block S8 cos 0.363).
- **When:** after D4. Doing it during D1 forks the D lineage.
- **Trigger:** D3e-in held WER, transcript-to-answer agreement, or spoken-QA accuracy below the bar set in the proposal (§6 there).
- Alternatives considered and rejected:
  - Cohere Transcribe's Conformer: 14 languages, trained from scratch, no Qwen lineage.
  - Granite or Phi-4 encoders: fewer languages, and their accuracy comes partly from their LLM decoders.

**Whole S2S models: no.**
- Qwen3-Omni, MiniCPM-o, VoiceChat and PersonaPlex each tie speech generation to their own LLM backbone through a talker conditioned on that backbone's hidden states.
- Chimera's trunk (Qwen3.8-27B lineage, vocab 248,320) replaces the backbone, so a folded talker would be conditioned on the wrong hidden-state distribution.
- Use them as data teachers only.

---

## 7. Cyril's own recordings ("transcordings") and a growing corpus

- **Store:**
  - raw files on the NAS (`/mnt/ai_data/…/speech/owner/` or `\\192.168.50.250\ai\…`);
  - the working copy on `/mnt/lexar/omni-campaign/speech/owner/`;
  - never root, never uploaded anywhere.
- **Manifest row:** `consent=owner`, `speaker=cyril`, sha256, duration, sample rate, source device. The manifest is append-only, so his ongoing additions just add rows (raw PostgreSQL if it goes into a table; no ORM).
- **Labels:** run Qwen3-ASR-1.7B, Cohere Transcribe and Granite 3.3 2B.
  - All three agree after normalisation → auto-admit.
  - Two of three agree → admit with a flag.
  - Otherwise → hold for Cyril's correction.
  - No single model's transcript becomes truth.
- **Uses, in order:**
  1. Real-speech **held evaluation** for D3e-in. Speech-in currently has 0 natural rows admitted.
  2. **Natural speech-in training rows.**
  3. A **cloning reference** for an owner voice through Qwen3-TTS-Base / VoxCPM2. Both are Apache-2.0, and the use is self-consented, so the no-impersonation clauses in other licences are not engaged.
  4. Stage T conversation pairs.
- **Size guide (a):** 10-30 minutes of clean speech is enough for a cloning reference and a held set. Training value grows with hours.
- **Split rule:** split by recording session, so no session crosses train and held.

---

## 8. Model register (HF API, read 2026-10-14; size = sum of repo files)

| Repo | sha | Repo size | Params | Licence tag |
|---|---|---:|---:|---|
| Qwen/Qwen3-ASR-1.7B | 7278e1e7 | 4.70 GB | 2.35B | apache-2.0 |
| Qwen/Qwen3-ASR-1.7B-hf | bcd2b5b7 | 4.09 GB | 2.04B | apache-2.0 |
| Qwen/Qwen3-ASR-0.6B | 5eb14417 | 1.88 GB | 0.94B | apache-2.0 |
| CohereLabs/cohere-transcribe-03-2026 | d552dadc | 4.13 GB | 2.07B | apache-2.0 (gated) |
| ibm-granite/granite-speech-3.3-2b | 4ac2f02f | 11.78 GB | 3.01B | apache-2.0 |
| ibm-granite/granite-speech-3.3-8b | 5670dcfa | 17.44 GB | 8.65B | apache-2.0 |
| microsoft/Phi-4-multimodal-instruct | 93f923e1 | 12.88 GB | 5.57B | mit |
| Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | 0c0e3051 | 4.52 GB | 1.92B | apache-2.0 |
| Qwen/Qwen3-TTS-12Hz-1.7B-Base | fd4b2543 | 4.54 GB | 1.93B | apache-2.0 |
| Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | 5ecdb673 | 4.52 GB | 1.92B | apache-2.0 |
| openbmb/VoxCPM2 | 32279eff | 4.96 GB | 2.29B | apache-2.0 |
| OpenMOSS-Team/MOSS-TTS-v1.5 | cdd3b911 | 17.00 GB | 8.49B | apache-2.0 |
| FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | 29e01c4e | 9.75 GB | — | apache-2.0 |
| tencent/AuK | 790742b7 | 6.76 GB | — | mit |
| ResembleAI/chatterbox | 5bb1f6ee | 13.87 GB | — | mit |
| fishaudio/drama-3 | e0a9218a | 41.50 GB | 35.9B (BF16 + F8_E4M3) | fish-audio-community-license (card text) / fish-audio-research-license (tag) |
| BreezeBlue/Breeze-TTS-2 | 3e28c515 | 7.68 GB | 3.47B | breezeblue-research-and-non-commercial |
| bosonai/higgs-tts-3-4b | 239f63fb | 9.32 GB | 4.65B | boson-higgs-tts-3-research-and-non-commercial |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | 26291f79 | 70.53 GB | 35.26B | apache-2.0 |
| openbmb/MiniCPM-o-4_5 | 503e7542 | 20.05 GB | 9.37B | apache-2.0 |
| nvidia/NVIDIA-NemotronLabs-VoiceChat-11B | 443794ea | 44.40 GB | 11.10B | openmdw-1.1 |
| nvidia/personaplex-7b-v1 | fdaf4090 | 17.14 GB | 8.37B | nvidia-open-model-license (gated) |
| nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 | e5e99324 | 66.06 GB | 33.02B | nvidia-open-model-agreement |

**Download budget for the recommended additions (d):**
- P40-runnable set: Qwen3-TTS-Base + VoiceDesign 9.06 GB, VoxCPM2 4.96 GB, Cohere Transcribe 4.13 GB, Granite 3.3 2B 11.78 GB, CosyVoice3 9.75 GB, AuK 6.76 GB = **46.4 GB**.
- Later 3090 set: MOSS-TTS v1.5 17.0 GB, MiniCPM-o 4.5 20.1 GB, an int4 Qwen3-Omni of about 20 GB (a) = **about 57 GB**.
- All of it goes to `/mnt/lexar/omni-campaign/` (600+ GB free). Root stays untouched (about 70 GB free, 60 GB floor).

---

## 9. Licence register (training use is the column that matters)

| Model | Licence | Training another model on its outputs | Evidence |
|---|---|---|---|
| Qwen3-ASR, Qwen3-TTS, Qwen3-Omni, VoxCPM2, MOSS-TTS, CosyVoice3, Granite Speech, Cohere Transcribe, MiniCPM-o 4.5, Step-Audio-2-mini, Step-Audio-EditX, Fun-Audio-Chat | Apache-2.0 | Allowed | HF tags (m); EditX from the GitHub licence API (m) |
| AuK, Chatterbox, Phi-4-mm, VibeVoice, Kimi-Audio, MiMo-Audio, Ming | MIT | Allowed | HF tags (m) |
| Parakeet / Canary, Kyutai pocket-tts | CC-BY-4.0 | Allowed with attribution | HF tags (m) |
| NemotronLabs VoiceChat, Nemotron 3.5 ASR | OpenMDW-1.1 | Allowed (permissive model licence) | licence file (m) |
| PersonaPlex, Magpie, Nemotron-3-Nano-Omni | NVIDIA Open Model License / Agreement | Card says "ready for commercial use"; full text not read | UNVERIFIED for output-training clauses |
| LFM2.5-Audio | LFM Open v1.0 | Allowed below $10M annual revenue | licence §5 (m) |
| IndexTTS-2.5 | bilibili Model Use License | "may not Use … to improve any AI model, except … non-commercial AI models" | licence (m) |
| OmniVoice, Voxtral 4B TTS, personaplex-rl-seamless | CC-BY-NC(-4.0) | Non-commercial only | card / tag (m) |
| Breeze TTS 2 | BreezeBlue Research & NC | Bars training "any non-BreezeBlue … model, except for creating a Derivative Model solely for a Research Purpose or Non-Commercial Purpose"; a distilled model counts as a Derivative Model | licence §1.3, §(b) (m) |
| Fish S2 Pro | Fish Audio Research (2026-03-07) | "will not use … any output … to create or improve any foundational generative AI model" | licence (m) |
| **Fish Drama 3** | Fish Audio Community v1.0 (2026-10-13) | "will not use … any output … to create, train, distill, or improve any foundational generative AI model or any speech, audio, or voice generation model" | licence (m) |
| Higgs TTS 3 | Boson Research & NC | "use … any output … to create, train, fine-tune, distill, or improve any … speech model" is prohibited | licence §IV(b)(i) (m) |
| Covo-Audio-Chat | Tencent custom | "only for academic purposes" | licence (m) |

---

## 10. What I could not verify

- **Artificial Analysis Elo values** come from a model-summarised fetch of a JavaScript page. Two other sources broadly agree (Breeze about 1,215-1,220, S2 Pro about 1,116-1,125), but I could not read the raw table.
- **Qwen3.5-Omni open weights:** none found (HF 401 on two guessed ids). Secondary sites disagree about a "Light" variant.
- **Speed of any model on a P40 or RTX 8000:** nothing measured. All fits in §5 are arithmetic on weight sizes, and int4 kernel support on Turing is unchecked.
- **NVIDIA Open Model License / Agreement:** full text not read for output-training terms.
- **Step-Audio-R1.1:** whether its open weights emit speech is unconfirmed (card benchmarks are images).
- **Cohere Transcribe:** card read through WebFetch because the repo is gated.
- **PersonaPlex and personaplex-rl-seamless:** cards read through WebFetch for the same reason.
- **Unread primary sources:** the Fish Drama 3 tech report (arXiv 2610.18010), the AuK paper (2609.08936) and the VoiceBench per-model provenance.
- **`facebook/omniASR-LLM-7B-v2`:** the HF API returned 401, so release date and size come only from the leaderboard CSV.
- **The two top-VoiceBench "open" community entries** (BR-Voice-Reasoner, LFG-3): not examined.
- **Qwen-judge bias:** possible favouritism toward Qwen TTS in the Open TTS Leaderboard WER is an assumption, not measured.

## 11. Evidence saved (local, for re-checking)

Folder `distill/speech-survey-2026-10-14/`:

| File | Bytes | sha256 (first 16) |
|---|---:|---|
| open_asr_leaderboard_english_short_2026-10-09.csv | 16,241 | 8d2758fa41fe1444 |
| open_tts_leaderboard_lb_en_clone0_v09-10-2026.json | 59,355 | 1cb636d576cb4ec1 |
| open_tts_leaderboard_lb_en_clone1_v09-10-2026.json | 79,353 | 035ff905b7b5a36c |
| open_tts_leaderboard_lb_all_clone0_v09-10-2026.json | 24,410 | 455bc98df7197ea5 |
| open_tts_leaderboard_lb_all_clone1_v09-10-2026.json | 14,271 | 3a3a64f97a7db9f4 |
| hf_model_metadata_a.json (96 repos) | 30,826 | 797ebc3de78524a1 |
| hf_model_metadata_b.json (31 repos) | 9,700 | 3a53f8c0b0acffb0 |
| voicebench_readme_2026-09-10.md | 32,424 | c0997e70144f0793 |
| licences/fishaudio_drama-3_LICENSE.txt | 19,685 | 733a32442087acbd |
| licences/fishaudio_s2-pro_LICENSE.md | 10,360 | aa7d9206e9d71059 |
| licences/BreezeBlue_Breeze-TTS-2_LICENSE.txt | 18,719 | f158b88cd5147392 |
| licences/bosonai_higgs-tts-3-4b_LICENSE.txt | 32,184 | 0382189330b87c38 |
| licences/IndexTeam_IndexTTS-2.5_LICENSE.txt | 10,553 | cc7da9ea0f8a97ef |
| licences/LiquidAI_LFM2.5-Audio_LICENSE.txt | 10,575 | c138fe99cb339429 |
| licences/tencent_Covo-Audio-Chat_LICENSE.txt | 1,695 | a2dec3fcff4b214e |
| licences/nvidia_NemotronLabs-VoiceChat-11B_LICENSE.txt | 2,704 | c55ed9a3dd7c5df1 |

**Primary URLs:**
- `huggingface.co/spaces/hf-audio/open_asr_leaderboard` (data `datasets/hf-audio/open-asr-leaderboard-results`, 2026-10-09)
- `huggingface.co/spaces/hf-audio/open_tts_leaderboard` (version 09-10-2026) and `huggingface.co/blog/open-tts-leaderboard` (2026-09-30)
- `artificialanalysis.ai/text-to-speech/leaderboard` (read 2026-10-14)
- `matthewcym.github.io/VoiceBench/`
- the model cards at `huggingface.co/<repo>` for every repo named above (read 2026-10-14)

**Secondary:**
- `orcarouter.ai/blog/breeze-tts-2-tops-open-weights-speech-arena` (Aug 2026)
- `digitalapplied.com/blog/best-text-to-speech-models-september-2026-ranked-priced` (2026-09-24/28)
- `appen.com/blog/hugging-face-open-llm-leaderboard` (May 2026)
- `benchmarklist.com/benchmarks/open_asr_leaderboard/` (2026-05-06)
