#!/usr/bin/env python3
"""speech_teacher_smoke.py -- load-and-run smoke for the staged speech teachers (Claude, 2026-10-15).

Design of record: distill/SPEECH_LANE_PLAN_v2_2026-10-14.md section 2.1 (teachers) and section 5 step 6 (the C-s0 rate
probe this precedes).  Proves each staged teacher loads from /mnt/lexar/models/<org>/<name>/ inside its runtime and does
its job on a handful of inputs; it is NOT the C-s0 rate probe (C-s0 runs on a P40 with 16 utterances; this smoke runs
where a card or CPU is free, and CPU timings say nothing about P40 rates).

Subcommands
    pick-librispeech  (any python3)   choose N test-clean utterances by sha256("chimera-speech-smoke-v1:" + utt_id),
                                      lowest first, among those with 8-30 reference words -> asr manifest JSON
    asr               (speech-asr)    --model qwen3-asr-1.7b | qwen3-asr-0.6b | granite-4.1-2b | granite-3.3-2b over a
                                      manifest; prints transcript, reference and WER per row and the pooled WER
    tts               (speech-qwentts | speech-cosyvoice3)
                                      --model qwen3-tts-customvoice | qwen3-tts-voicedesign | qwen3-tts-base | cosyvoice3;
                                      the fixed SENTENCES -> one wav each + an asr manifest whose reference is the text.
                                      cosyvoice3 is zero-shot only (no built-in voices): --clone-ref is required.
    voxcpm2 is REFUSED (1.1.0): its loader imports pydantic (Cyril's no-ORM / no-Pydantic rule); granite-3.3-2b stays
    loadable for comparison only (superseded as judge B by granite-4.1-2b).
    merge-manifests   (any python3)   concatenate tts manifests for one round-trip asr pass
Every run writes a result JSON: model dir + receipt revision, device, dtype, threads, load seconds, per-row seconds,
audio seconds, real-time factor, peak RSS (CPU) or torch peak allocated (CUDA).  WER = word edit distance / reference
words after one normaliser (lowercase, curly quotes folded, hyphens to spaces, keep [a-z0-9'], collapse spaces); no
number normalisation, so "20" against "twenty" counts as an error.  Outputs of this smoke are smoke-only: they never
enter a corpus (the LibriSpeech rows are test-clean, a held bank in the holdout registry v2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

TOOL_VERSION = "1.1.0"
# 1.1.0 (2026-10-15, teacher fix): + granite-4.1-2b (judge B; no LoRA), + cosyvoice3 (family 2, Fun-CosyVoice3-0.5B-2512,
#        RL LLM weights by default), voxcpm2 refused, pick-librispeech names the split it read (was always test-clean).
# 1.0.1: Granite loads base + LoRA from the local dir (1.0.0 went to the hub offline)
MODELS = Path("/mnt/lexar/models")
MODEL_DIRS = {
    "qwen3-asr-1.7b": MODELS / "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": MODELS / "Qwen/Qwen3-ASR-0.6B",
    "granite-4.1-2b": MODELS / "ibm-granite/granite-speech-4.1-2b",
    "granite-3.3-2b": MODELS / "ibm-granite/granite-speech-3.3-2b",
    "qwen3-tts-customvoice": MODELS / "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "qwen3-tts-voicedesign": MODELS / "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "qwen3-tts-base": MODELS / "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "cosyvoice3": MODELS / "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
}
REFUSED = {"voxcpm2": "openbmb/VoxCPM2 is unusable under the no-Pydantic rule (voxcpm 2.0.3 imports pydantic); its runtime "
                      "speech-voxcpm was removed 2026-10-15"}
CV3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"  # CosyVoice3 zero-shot prompt-text form (card, en usage)
# Fixed smoke texts: numbers spelled out so the WER does not measure number formatting.
SENTENCES = [
    "The river rose overnight, so the ferry left an hour later than usual.",
    "Please set the oven to two hundred degrees and wait ten more minutes.",
    "She painted the old fence blue, then sat on the porch to watch it dry.",
    "Our meeting moved to Thursday because the projector was broken again.",
]
CUSTOM_SPEAKERS = ["ryan", "aiden", "serena", "vivian"]  # the four Codex already ran in English (2026-10-13 canary)
DESIGN_INSTRUCTS = [
    "A calm middle-aged man with a deep, warm voice, speaking slowly and clearly.",
    "A cheerful young woman with a bright voice, speaking at a brisk pace.",
    "An elderly woman with a soft, slightly husky voice, gentle and unhurried.",
    "A young man with a crisp, energetic newsreader delivery.",
]
GRANITE_SYSTEM = ("Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\n"
                  "You are Granite, developed by IBM. You are a helpful AI assistant")  # the model card's prompt
GRANITE_USER = "<|audio|>can you transcribe the speech into a written format?"   # also the 4.1 card's raw-ASR prompt


def normalise(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def receipt_revision(model_dir: Path) -> str | None:
    path = model_dir / ".speech_fetch_receipt.json"
    return json.loads(path.read_text())["revision"] if path.is_file() else None


def peak_memory(device: str) -> dict:
    out = {"peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)}
    if device.startswith("cuda"):
        import torch
        out["cuda_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        out["cuda_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    return out


def cmd_pick(args: argparse.Namespace) -> int:
    root = Path(args.librispeech)
    rows = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            if 8 <= len(text.split()) <= 30:
                key = hashlib.sha256(f"chimera-speech-smoke-v1:{utt}".encode()).hexdigest()
                rows.append((key, utt, text, trans.parent / f"{utt}.flac"))
    rows.sort()
    manifest = [{"id": utt, "audio": str(path), "reference": text, "source": f"librispeech/{root.name}",
                 "licence": "CC BY 4.0", "pick_key": key} for key, utt, text, path in rows[: args.n]]
    Path(args.out).write_text(json.dumps(manifest, indent=1) + "\n")
    for row in manifest:
        print(f"{row['id']}: {row['reference']}")
    return 0


def load_audio(path: str, sr: int = 16000):
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def cmd_asr(args: argparse.Namespace) -> int:
    import torch
    torch.set_num_threads(args.threads)
    model_dir = MODEL_DIRS[args.model]
    manifest = json.loads(Path(args.manifest).read_text())
    device = args.device
    extra: dict = {}
    started = time.time()
    if args.model.startswith("qwen3-asr"):
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(str(model_dir), dtype=torch.float32, device_map=device,
                                              max_inference_batch_size=1, max_new_tokens=256)

        def transcribe(wav):
            return model.transcribe(audio=(wav, 16000), language="English")[0].text
    else:
        from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration
        processor = AutoProcessor.from_pretrained(str(model_dir))
        tokenizer = processor.tokenizer
        if args.model == "granite-3.3-2b":
            # 3.3 ships its LoRA (adapter_config.json, base_model_name_or_path = the hub id of the same repo). The
            # Auto* loaders see the adapter file and redirect the BASE load to that hub id, which fails offline (1.0.0).
            # The concrete class with an explicit adapter path loads base AND adapter from the local directory.
            model, info = GraniteSpeechForConditionalGeneration.from_pretrained(
                str(model_dir), dtype=torch.float32, adapter_kwargs={"_adapter_model_path": str(model_dir)},
                output_loading_info=True)
            extra["lora_adapter_loaded"] = bool(getattr(model, "_hf_peft_config_loaded", False))
            if not extra["lora_adapter_loaded"]:
                raise RuntimeError("granite LoRA adapter not loaded")
            chat = [{"role": "system", "content": GRANITE_SYSTEM}, {"role": "user", "content": GRANITE_USER}]
        else:
            # 4.1: config has_lora_adapter = false, no adapter file; its chat template has no system role
            model, info = GraniteSpeechForConditionalGeneration.from_pretrained(
                str(model_dir), dtype=torch.float32, output_loading_info=True)
            chat = [{"role": "user", "content": GRANITE_USER}]
        extra["loading_info"] = {k: sorted(map(str, v))[:20] for k, v in info.items() if v}
        if info.get("missing_keys"):
            raise RuntimeError(f"granite weights left uninitialised: {sorted(info['missing_keys'])[:5]}")
        model = model.to(device).eval()
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        def transcribe(wav):
            inputs = processor(prompt, torch.from_numpy(wav).unsqueeze(0), device=device, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
            new = out[0, inputs["input_ids"].shape[-1]:].unsqueeze(0)
            return tokenizer.batch_decode(new, add_special_tokens=False, skip_special_tokens=True)[0]
    load_seconds = time.time() - started
    rows, edits, words = [], 0, 0
    for item in manifest:
        wav = load_audio(item["audio"])
        t0 = time.time()
        text = transcribe(wav).strip()
        seconds = time.time() - t0
        ref, hyp = normalise(item["reference"]), normalise(text)
        dist = edit_distance(ref, hyp)
        edits, words = edits + dist, words + len(ref)
        audio_seconds = len(wav) / 16000
        rows.append({"id": item["id"], "reference": item["reference"], "transcript": text,
                     "word_edits": dist, "reference_words": len(ref), "wer": round(dist / max(1, len(ref)), 4),
                     "audio_seconds": round(audio_seconds, 2), "seconds": round(seconds, 2),
                     "rtf": round(seconds / audio_seconds, 3)})
        print(f"[{args.model}] {item['id']}  WER {dist}/{len(ref)} = {dist / max(1, len(ref)):.3f}  "
              f"({seconds:.1f} s for {audio_seconds:.1f} s audio)\n   REF: {item['reference']}\n   HYP: {text}", flush=True)
    result = {"schema": "omni-campaign.speech-teacher-smoke.asr.v1", "tool_version": TOOL_VERSION,
              "model": args.model, "model_dir": str(model_dir), "revision": receipt_revision(model_dir),
              "device": device, "dtype": "float32", "threads": args.threads, "load_seconds": round(load_seconds, 1),
              "manifest": args.manifest, "rows": rows, "pooled_wer": round(edits / max(1, words), 4),
              "word_edits": edits, "reference_words": words, **extra, **peak_memory(device),
              "torch": torch.__version__, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(f"[{args.model}] pooled WER {edits}/{words} = {result['pooled_wer']:.4f}; load {load_seconds:.0f} s; "
          f"peak {peak_memory(device)}", flush=True)
    return 0


def cmd_tts(args: argparse.Namespace) -> int:
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(args.threads)
    model_dir = MODEL_DIRS[args.model]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    ref = json.loads(Path(args.clone_ref).read_text())[0] if args.clone_ref else None
    started = time.time()
    if args.model == "cosyvoice3":
        if ref is None:
            raise SystemExit("cosyvoice3 needs --clone-ref (it has no built-in voices)")
        from cosyvoice.cli.cosyvoice import CosyVoice3
        model = CosyVoice3(str(model_dir), load_trt=False, load_vllm=False, fp16=False)
        llm_weights = "llm.rl.pt" if args.cv3_llm == "rl" else "llm.pt"
        if args.cv3_llm == "rl":  # the leaderboard / card row is the RL LLM; CosyVoice3() always loads llm.pt first
            state = torch.load(str(model_dir / llm_weights), map_location=model.model.device, weights_only=True)
            model.model.llm.load_state_dict(state, strict=True)
            model.model.llm.to(model.model.device).eval()
        sample_rate = model.sample_rate
        prompt_text = CV3_PROMPT_PREFIX + ref["reference"]

        def synth(i, text):
            chunks = [out["tts_speech"] for out in model.inference_zero_shot(text, prompt_text, ref["audio"], stream=False)]
            wav = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()
            return wav, {"clone_ref": ref["id"], "clone_ref_source": ref["source"], "llm_weights": llm_weights}
    else:
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(str(model_dir), dtype=torch.float32, device_map=device)
        sample_rate = None

        def synth(i, text):
            if args.model == "qwen3-tts-customvoice":
                wavs, sr = model.generate_custom_voice(text=text, speaker=CUSTOM_SPEAKERS[i], language="English")
                cond = {"speaker": CUSTOM_SPEAKERS[i]}
            elif args.model == "qwen3-tts-voicedesign":
                wavs, sr = model.generate_voice_design(text=text, instruct=DESIGN_INSTRUCTS[i], language="English")
                cond = {"instruct": DESIGN_INSTRUCTS[i]}
            else:
                wavs, sr = model.generate_voice_clone(text=text, language="English", ref_audio=ref["audio"],
                                                      ref_text=ref["reference"])
                cond = {"clone_ref": ref["id"], "clone_ref_source": ref["source"]}
            cond["sample_rate"] = sr
            return wavs[0], cond
    load_seconds = time.time() - started
    rows, manifest = [], []
    for i, text in enumerate(SENTENCES):
        t0 = time.time()
        wav, cond = synth(i, text)
        seconds = time.time() - t0
        sr = cond.get("sample_rate", sample_rate)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        path = out_dir / f"{args.model}-{i}.wav"
        sf.write(str(path), wav, sr)
        audio_seconds = len(wav) / sr
        peak = float(np.abs(wav).max()) if wav.size else 0.0
        rows.append({"index": i, "text": text, "wav": str(path), "sample_rate": sr,
                     "audio_seconds": round(audio_seconds, 2), "seconds": round(seconds, 2),
                     "rtf": round(seconds / max(audio_seconds, 1e-6), 3), "peak_abs": round(peak, 4),
                     "finite": bool(np.isfinite(wav).all()), "condition": cond,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest.append({"id": f"{args.model}-{i}", "audio": str(path), "reference": text,
                         "source": f"smoke-tts:{args.model}"})
        print(f"[{args.model}] {i}: {audio_seconds:.2f} s audio at {sr} Hz in {seconds:.1f} s "
              f"(RTF {seconds / max(audio_seconds, 1e-6):.2f}), peak {peak:.3f}  {cond}", flush=True)
    result = {"schema": "omni-campaign.speech-teacher-smoke.tts.v1", "tool_version": TOOL_VERSION,
              "model": args.model, "model_dir": str(model_dir), "revision": receipt_revision(model_dir),
              "device": device, "dtype": "float32", "threads": args.threads, "load_seconds": round(load_seconds, 1),
              "rows": rows, **peak_memory(device), "torch": torch.__version__,
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / f"{args.model}.result.json").write_text(json.dumps(result, indent=1) + "\n")
    (out_dir / f"{args.model}.asr_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[{args.model}] load {load_seconds:.0f} s; peak {peak_memory(device)}", flush=True)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    rows = []
    for path in args.manifests:
        if Path(path).is_file():
            rows.extend(json.loads(Path(path).read_text()))
        else:
            print(f"missing (skipped): {path}", file=sys.stderr)
    Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{len(rows)} rows -> {args.out}")
    return 0 if rows else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick-librispeech")
    p.add_argument("--librispeech", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--out", required=True)
    a = sub.add_parser("asr")
    a.add_argument("--model", required=True, choices=[m for m in MODEL_DIRS if "asr" in m or "granite" in m])
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--threads", type=int, default=16)
    t = sub.add_parser("tts")
    t.add_argument("--model", required=True,
                   choices=[m for m in MODEL_DIRS if "tts" in m or "cosyvoice" in m] + sorted(REFUSED))
    t.add_argument("--out-dir", required=True)
    t.add_argument("--clone-ref", help="asr manifest whose first row is the clone reference (qwen3-tts-base, cosyvoice3)")
    t.add_argument("--cv3-llm", choices=["rl", "base"], default="rl", help="cosyvoice3 LLM weights: llm.rl.pt or llm.pt")
    t.add_argument("--device", default="cpu")
    t.add_argument("--threads", type=int, default=16)
    m = sub.add_parser("merge-manifests")
    m.add_argument("manifests", nargs="+")
    m.add_argument("--out", required=True)
    args = parser.parse_args()
    if getattr(args, "model", None) in REFUSED:
        print(f"refused: {REFUSED[args.model]}", file=sys.stderr)
        return 2
    if args.cmd in ("asr", "tts") and args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("CPU smoke must run with CUDA_VISIBLE_DEVICES=\"\"", file=sys.stderr)
        return 2
    return {"pick-librispeech": cmd_pick, "asr": cmd_asr, "tts": cmd_tts, "merge-manifests": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
