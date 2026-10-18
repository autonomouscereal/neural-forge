OC=/home/cereal/omni-campaign; R=$OC/distill/speech/teachers; RT=/mnt/lexar/omni-campaign/runtimes
for d in Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-ASR-0.6B Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign ibm-granite/granite-speech-3.3-2b openbmb/VoxCPM2 hexgrad/Kokoro-82M; do
  python3 $OC/distill/tools/speech/speech_fetch.py verify --dest /mnt/lexar/models/$d | tail -1
done > $R/fetch/verify_2026-10-15.txt 2>&1
cat $R/fetch/verify_2026-10-15.txt
CUDA_VISIBLE_DEVICES="" $RT/speech-torch/bin/python -I -c "import torch;print('arch_flags', torch._C._cuda_getArchFlags())"
for v in speech-asr speech-qwentts speech-voxcpm; do echo "$v pydantic: $(grep -i -c '^pydantic' $R/runtime/$v.freeze.txt) gradio: $(grep -i -c '^gradio' $R/runtime/$v.freeze.txt)"; done
python3 - <<'PY'
import json, glob, hashlib, os, datetime, pathlib
R = pathlib.Path("/home/cereal/omni-campaign/distill/speech/teachers")
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
models = []
for f in sorted(glob.glob(str(R / "fetch/*__*.json"))):
    d = json.load(open(f))
    if "repo" not in d: continue
    models.append({"repo": d["repo"], "revision": d["revision"], "dest": d["dest"], "licence": d["licence"]["card_license"],
                   "licence_files": d["licence"]["files"], "gated": d["gated"], "files": d["file_count"], "bytes": d["total_bytes"],
                   "downloaded_bytes": d["downloaded_bytes"], "seeded_from": d["seed"], "complete": d["complete"],
                   "receipt": f, "receipt_sha256": sha(f)})
smoke = {}
for f in sorted(glob.glob(str(R / "smoke/*.json")) + glob.glob(str(R / "smoke/tts/*.result.json"))):
    d = json.load(open(f))
    if not isinstance(d, dict) or "model" not in d: continue
    key = pathlib.Path(f).name
    row = {"model": d["model"], "revision": d["revision"], "tool_version": d["tool_version"], "device": d["device"], "threads": d["threads"],
           "load_seconds": d["load_seconds"], "peak_rss_gib": d["peak_rss_gib"], "file": f, "sha256": sha(f)}
    if "pooled_wer" in d:
        row.update(pooled_wer=d["pooled_wer"], word_edits=d["word_edits"], reference_words=d["reference_words"],
                   rtf=[r["rtf"] for r in d["rows"]], lora_adapter_loaded=d.get("lora_adapter_loaded"))
    else:
        row.update(rtf=[r["rtf"] for r in d["rows"]], audio_seconds=[r["audio_seconds"] for r in d["rows"]],
                   sample_rate=d["rows"][0]["sample_rate"], all_finite=all(r["finite"] for r in d["rows"]))
    smoke[key] = row
ls = json.load(open(R / "fetch/librispeech__test-clean.json"))
out = {"schema": "omni-campaign.speech-stage-teachers.v1", "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "by": "claude", "plan": "distill/SPEECH_LANE_PLAN_v2_2026-10-14.md section 2.4 + section 3 (E1)",
       "layout": "/mnt/lexar/models/<org>/<name>/ (stage-teachers order); plan v2 said models/speech/<org>__<name>@<rev8>/ -- superseded",
       "models": models, "librispeech_test_clean": ls,
       "runtimes": {v: {"path": f"/mnt/lexar/omni-campaign/runtimes/{v}", "freeze": str(R / f"runtime/{v}.freeze.txt"),
                         "freeze_sha256": sha(R / f"runtime/{v}.freeze.txt"), "import": json.load(open(R / f"runtime/{v}.import.json"))}
                    for v in ["speech-asr", "speech-qwentts", "speech-voxcpm"]},
       "runtime_base": {"path": "/mnt/lexar/omni-campaign/runtimes/speech-torch", "freeze_sha256": sha(R / "runtime/speech-torch.freeze.txt"),
                        "shared_by": "zz_speech_torch_base.pth in each overlay venv"},
       "smoke": smoke,
       "not_done": ["Cohere Transcribe (gated; needs Cyril's HF gate acceptance + vault key hf_token)",
                    "GPU smoke: GPU3 held by k3accel then fold-research x3-08b at launch; CPU only, so no P40 rate or VRAM figure",
                    "Kokoro-82M runtime (weights only; fallback)", "root parent deletion (needs Cyril's go-ahead)"]}
p = R / "STAGE_TEACHERS_2026-10-15.json"
p.write_text(json.dumps(out, indent=1) + "\n")
print(p, sha(p), len(models), "models", len(smoke), "smoke results")
PY
df -h /mnt/lexar / | tail -2; du -sh --apparent-size /mnt/lexar/models/Qwen /mnt/lexar/models/ibm-granite /mnt/lexar/models/openbmb /mnt/lexar/models/hexgrad; du -sh $R
