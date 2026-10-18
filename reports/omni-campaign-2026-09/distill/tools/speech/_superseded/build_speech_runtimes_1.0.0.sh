#!/usr/bin/env bash
# build_speech_runtimes.sh -- E1 of SPEECH_LANE_PLAN_v2 (section 3), Claude 2026-10-15.
# Three Pascal (sm_61) venvs instead of the plan's one, because the packages pin incompatible transformers:
#   speech-asr     qwen-asr 0.0.6 (pins transformers==4.57.6) + Granite Speech 3.3 (transformers>=4.52, peft)
#   speech-qwentts qwen-tts 0.1.1 (pins transformers==4.57.3)
#   speech-voxcpm  voxcpm 2.0.3 (its model code imports pydantic for config classes; isolated here so no other
#                  speech runtime carries it; our own code never imports it)
# All on torch 2.6.0+cu118 / torchaudio 2.6.0+cu118 (the build venv-pascal-cu118-torch260 proves on the P40s), installed
# ONCE in speech-torch and shared by a .pth line appended to each overlay venv (own site-packages first, base second).
# /mnt/lexar is exFAT (no symlinks, 1 MiB directories): venv's lib64 -> lib symlink is pre-empted by a real lib64
# directory, and one shared torch avoids three copies of ~15k torch files.
# Lean installs: the speech packages go in with --no-deps and only their inference imports are added (no gradio,
# flask, vllm, flash-attn, modelscope, funasr, datasets).  CPU only here: CUDA_VISIBLE_DEVICES="".
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" PIP_DISABLE_PIP_VERSION_CHECK=1
RT=/mnt/lexar/omni-campaign/runtimes
export TMPDIR=$RT/.tmp PIP_CACHE_DIR=$RT/.pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
OUT=/home/cereal/omni-campaign/distill/speech/teachers/runtime
mkdir -p "$OUT"
TORCH_IDX=https://download.pytorch.org/whl/cu118
rc_all=0

mkbase() {
  local v=$RT/speech-torch
  mkdir -p "$v/lib64"
  [ -x "$v/bin/python" ] || /usr/bin/python3 -m venv --copies "$v" || return 1
  "$v/bin/python" -m pip install -q --upgrade "pip<27" || return 1
  "$v/bin/python" -m pip install -q --index-url $TORCH_IDX torch==2.6.0+cu118 torchaudio==2.6.0+cu118 || return 1
  "$v/bin/python" -m pip install -q "numpy<2.5" || return 1
  "$v/bin/python" -m pip freeze > "$OUT/speech-torch.freeze.txt"
}

mkvenv() {  # name -- overlay venv on the shared torch base
  local v=$RT/$1
  mkdir -p "$v/lib64"
  [ -x "$v/bin/python" ] || /usr/bin/python3 -m venv --copies "$v" || return 1
  "$v/bin/python" -m pip install -q --upgrade "pip<27" || return 1
  echo "$RT/speech-torch/lib/python3.12/site-packages" > "$v/lib/python3.12/site-packages/zz_speech_torch_base.pth"
  printf 'torch==2.6.0+cu118
torchaudio==2.6.0+cu118
' > "$v/constraints.txt"
}

finish() {  # name import-snippet
  local v=$RT/$1
  "$v/bin/python" -m pip freeze > "$OUT/$1.freeze.txt"
  "$v/bin/python" -m pip check > "$OUT/$1.pipcheck.txt" 2>&1
  timeout 600 "$v/bin/python" -I -c "$2" > "$OUT/$1.import.json" 2> "$OUT/$1.import.err"
  local rc=$?
  echo "== $1 import rc=$rc"; cat "$OUT/$1.import.json"; tail -3 "$OUT/$1.import.err"
  [ $rc -ne 0 ] && rc_all=1
}

echo "=== $(date -u +%FT%TZ) speech-torch (shared base)"
mkbase || { echo "base build FAILED"; echo 1 > "$OUT/exitcode"; exit 1; }

HDR='import json, sys, torch, transformers; r={"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda, "arch_list": torch.cuda.get_arch_list(), "transformers": transformers.__version__, "cuda_visible": torch.cuda.is_available()};'

echo "=== $(date -u +%FT%TZ) speech-asr"
if mkvenv speech-asr; then
  P=$RT/speech-asr/bin/python
  $P -m pip install -q -c $RT/speech-asr/constraints.txt transformers==4.57.6 accelerate==1.12.0 peft librosa soundfile \
      scipy numpy nagisa==0.2.11 soynlp==0.0.493 pytz && \
  $P -m pip install -q --no-deps qwen-asr==0.0.6 || rc_all=1
  finish speech-asr "$HDR"'
from qwen_asr import Qwen3ASRModel
from transformers import GraniteSpeechForConditionalGeneration, AutoProcessor
import peft, librosa, soundfile
r.update(qwen_asr="Qwen3ASRModel", granite="GraniteSpeechForConditionalGeneration", peft=peft.__version__, import_passed=True)
print(json.dumps(r))'
else rc_all=1; fi

echo "=== $(date -u +%FT%TZ) speech-qwentts"
if mkvenv speech-qwentts; then
  P=$RT/speech-qwentts/bin/python
  $P -m pip install -q -c $RT/speech-qwentts/constraints.txt transformers==4.57.3 accelerate==1.12.0 librosa soundfile \
      sox onnxruntime einops numpy && \
  $P -m pip install -q --no-deps qwen-tts==0.1.1 || rc_all=1
  finish speech-qwentts "$HDR"'
from qwen_tts import Qwen3TTSModel
import onnxruntime
r.update(qwen_tts="Qwen3TTSModel", onnxruntime=onnxruntime.__version__, import_passed=True)
print(json.dumps(r))'
else rc_all=1; fi

echo "=== $(date -u +%FT%TZ) speech-voxcpm"
if mkvenv speech-voxcpm; then
  P=$RT/speech-voxcpm/bin/python
  $P -m pip install -q -c $RT/speech-voxcpm/constraints.txt transformers==4.57.3 einops pydantic librosa soundfile inflect \
      addict wetext tqdm simplejson sortedcontainers safetensors huggingface_hub argbind regex pyyaml numpy && \
  $P -m pip install -q --no-deps voxcpm==2.0.3 || rc_all=1
  finish speech-voxcpm "$HDR"'
from voxcpm import VoxCPM
r.update(voxcpm="VoxCPM", import_passed=True)
print(json.dumps(r))'
else rc_all=1; fi

du -sh --apparent-size $RT/speech-* $RT/.pip-cache; du -sh $RT/speech-* 2>/dev/null
echo "=== BUILD DONE $(date -u +%FT%TZ) rc_all=$rc_all"
echo $rc_all > "$OUT/exitcode"
