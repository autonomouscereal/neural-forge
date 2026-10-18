#!/usr/bin/env bash
# build_speech_runtimes.sh 1.1.0 -- E1 of SPEECH_LANE_PLAN_v2 (section 3), Claude 2026-10-15.
# 1.1.0 (2026-10-15, teacher fix): speech-voxcpm DROPPED (it installed pydantic 2.14.1 for voxcpm 2.0.3 -- Cyril's rule:
#   no ORM / Pydantic / SQLAlchemy in any environment we build; removed by remove_voxcpm_runtime.sh).  New runtime
#   speech-cosyvoice3 for speech-out family 2 (Fun-CosyVoice3-0.5B-2512).  Every runtime now passes a FORBIDDEN gate
#   twice: on the resolver's dry-run report BEFORE anything is installed, and on pip freeze + import probes AFTER.
#   SPEECH_RT_ONLY="name ..." builds only the named runtimes (the others are not touched).
#   1.0.0 is kept as tools/speech/_superseded/build_speech_runtimes_1.0.0.sh.
# Runtimes (Pascal sm_61, all on the shared torch 2.6.0+cu118 base):
#   speech-asr        qwen-asr 0.0.6 (pins transformers==4.57.6) + Granite Speech 4.1 / 3.3 (transformers>=4.52, peft)
#   speech-qwentts    qwen-tts 0.1.1 (pins transformers==4.57.3)
#   speech-cosyvoice3 CosyVoice source @074ca6dc (Apache-2.0) + Matcha-TTS @dd9105b3 (MIT), lean inference deps only,
#                     transformers==4.51.3 (the upstream pin).  Upstream requirements.txt also lists pydantic, fastapi,
#                     gradio, uvicorn, deepspeed, tensorrt, vllm: server / webui / training only -- the library code
#                     under cosyvoice/ and matcha/ never imports pydantic (grep, 2026-10-15) and none of them is installed.
# torch/torchaudio are installed ONCE in speech-torch and shared by a .pth line appended to each overlay venv (own
# site-packages first, base second).  /mnt/lexar is exFAT (no symlinks, 1 MiB directories): venv's lib64 -> lib symlink
# is pre-empted by a real lib64 directory, and one shared torch avoids three copies of ~15k torch files.
# Lean installs: the speech packages go in with --no-deps and only their inference imports are added (no gradio,
# flask, vllm, flash-attn, funasr, datasets, fastapi).  CPU only here: CUDA_VISIBLE_DEVICES="".
set -uo pipefail
export CUDA_VISIBLE_DEVICES="" PIP_DISABLE_PIP_VERSION_CHECK=1
VERSION=1.1.0
RT=/mnt/lexar/omni-campaign/runtimes
export TMPDIR=$RT/.tmp PIP_CACHE_DIR=$RT/.pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
OUT=/home/cereal/omni-campaign/distill/speech/teachers/runtime
mkdir -p "$OUT"
TORCH_IDX=https://download.pytorch.org/whl/cu118
ONLY=" ${SPEECH_RT_ONLY:-speech-asr speech-qwentts speech-cosyvoice3} "
# Cyril's rule, as distribution names (PEP 503 normalised): ORM / Pydantic / SQLAlchemy and their usual carriers.
FORBIDDEN='^(pydantic|pydantic-core|pydantic-settings|pydantic-extra-types|sqlalchemy|sqlalchemy-utils|sqlmodel|alembic|peewee|pony|tortoise-orm|django|mongoengine|odmantic|beanie|ormar|piccolo|fastapi|gradio)$'
rc_all=0

want() { [[ "$ONLY" == *" $1 "* ]]; }

names_forbidden() {  # stdin: one distribution name per line -> prints the forbidden ones (normalised)
  tr 'A-Z_.' 'a-z--' | sed 's/[[:space:]]*$//' | grep -E "$FORBIDDEN" | sort -u
}

mkbase() {
  local v=$RT/speech-torch
  mkdir -p "$v/lib64"
  [ -x "$v/bin/python" ] || /usr/bin/python3 -m venv --copies "$v" || return 1
  "$v/bin/python" -m pip install -q --upgrade "pip<27" || return 1
  "$v/bin/python" -m pip install -q --index-url $TORCH_IDX torch==2.6.0+cu118 torchaudio==2.6.0+cu118 || return 1
  "$v/bin/python" -m pip install -q "numpy<2.5" || return 1
  "$v/bin/python" -m pip freeze > "$OUT/speech-torch.freeze.txt"
  local bad
  bad=$(sed -E 's/[=@ ].*//' "$OUT/speech-torch.freeze.txt" | names_forbidden)
  echo "== speech-torch freeze forbidden: ${bad:-none}"
  [ -z "$bad" ]
}

mkvenv() {  # name -- overlay venv on the shared torch base
  local v=$RT/$1
  mkdir -p "$v/lib64"
  [ -x "$v/bin/python" ] || /usr/bin/python3 -m venv --copies "$v" || return 1
  "$v/bin/python" -m pip install -q --upgrade "pip<27" || return 1
  echo "$RT/speech-torch/lib/python3.12/site-packages" > "$v/lib/python3.12/site-packages/zz_speech_torch_base.pth"
  printf 'torch==2.6.0+cu118\ntorchaudio==2.6.0+cu118\nnumpy<2.5\n' > "$v/constraints.txt"
}

resolve_gate() {  # name pip-args... -- dry-run resolve of the FULL dependency tree in a fresh scratch venv (only the
  # shared torch base visible, itself gated in mkbase); refuses before anything is installed into the real runtime
  local name=$1; shift
  local S=$RT/.resolve-scratch rep=$OUT/$name.resolve.json
  rm -rf "$S"; mkdir -p "$S/lib64"
  /usr/bin/python3 -m venv --copies "$S" && "$S/bin/python" -m pip install -q --upgrade "pip<27" || { echo "== scratch venv FAILED"; return 1; }
  echo "$RT/speech-torch/lib/python3.12/site-packages" > "$S/lib/python3.12/site-packages/zz_speech_torch_base.pth"
  "$S/bin/python" -m pip install -q --dry-run --report "$rep" -c "$RT/$name/constraints.txt" "$@"       > "$OUT/$name.resolve.log" 2>&1 || { echo "== $name resolve FAILED"; tail -5 "$OUT/$name.resolve.log"; rm -rf "$S"; return 1; }
  rm -rf "$S"
  local bad n
  bad=$(/usr/bin/python3 -c 'import json,sys; [print(i["metadata"]["name"]) for i in json.load(open(sys.argv[1]))["install"]]' "$rep" | names_forbidden)
  n=$(/usr/bin/python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["install"]))' "$rep")
  echo "== $name resolve: $n distributions on top of the torch base; forbidden: ${bad:-none}"
  [ -z "$bad" ]
}

finish() {  # name import-snippet
  local v=$RT/$1
  "$v/bin/python" -m pip freeze > "$OUT/$1.freeze.txt"
  "$v/bin/python" -m pip check > "$OUT/$1.pipcheck.txt" 2>&1
  local bad
  bad=$(sed -E 's/[=@ ].*//' "$OUT/$1.freeze.txt" | names_forbidden)
  echo "== $1 freeze forbidden: ${bad:-none}"
  [ -n "$bad" ] && rc_all=1
  timeout 900 "$v/bin/python" -I -c "$2" > "$OUT/$1.import.json" 2> "$OUT/$1.import.err"
  local rc=$?
  echo "== $1 import rc=$rc"; cat "$OUT/$1.import.json"; tail -3 "$OUT/$1.import.err"
  [ $rc -ne 0 ] && rc_all=1
}

echo "=== $(date -u +%FT%TZ) build_speech_runtimes $VERSION; building:$ONLY"
echo "=== $(date -u +%FT%TZ) speech-torch (shared base)"
mkbase || { echo "base build FAILED"; echo 1 > "$OUT/exitcode"; exit 1; }

# Every import probe also proves the rule inside the interpreter (covers .pth-chained site-packages too).
HDR='import json, sys, importlib.util as iu, torch, transformers; r={"python": sys.version.split()[0], "torch": torch.__version__, "cuda_build": torch.version.cuda, "arch_list": torch.cuda.get_arch_list(), "transformers": transformers.__version__, "cuda_visible": torch.cuda.is_available()};
bad=[m for m in ("pydantic", "pydantic_core", "sqlalchemy", "sqlmodel", "peewee", "pony", "tortoise", "django", "mongoengine", "fastapi", "gradio") if iu.find_spec(m) is not None]
r["forbidden_importable"]=bad
assert not bad, bad;'

if want speech-asr; then
  echo "=== $(date -u +%FT%TZ) speech-asr"
  if mkvenv speech-asr; then
    P=$RT/speech-asr/bin/python
    ASR_DEPS="transformers==4.57.6 accelerate==1.12.0 peft librosa soundfile scipy numpy nagisa==0.2.11 soynlp==0.0.493 pytz"
    resolve_gate speech-asr $ASR_DEPS && \
    $P -m pip install -q -c $RT/speech-asr/constraints.txt $ASR_DEPS && \
    $P -m pip install -q --no-deps qwen-asr==0.0.6 || rc_all=1
    finish speech-asr "$HDR"'
from qwen_asr import Qwen3ASRModel
from transformers import GraniteSpeechForConditionalGeneration, AutoProcessor
import peft, librosa, soundfile
r.update(qwen_asr="Qwen3ASRModel", granite="GraniteSpeechForConditionalGeneration", peft=peft.__version__, import_passed=True)
print(json.dumps(r))'
  else rc_all=1; fi
fi

if want speech-qwentts; then
  echo "=== $(date -u +%FT%TZ) speech-qwentts"
  if mkvenv speech-qwentts; then
    P=$RT/speech-qwentts/bin/python
    QT_DEPS="transformers==4.57.3 accelerate==1.12.0 librosa soundfile sox onnxruntime einops numpy"
    resolve_gate speech-qwentts $QT_DEPS && \
    $P -m pip install -q -c $RT/speech-qwentts/constraints.txt $QT_DEPS && \
    $P -m pip install -q --no-deps qwen-tts==0.1.1 || rc_all=1
    finish speech-qwentts "$HDR"'
from qwen_tts import Qwen3TTSModel
import onnxruntime
r.update(qwen_tts="Qwen3TTSModel", onnxruntime=onnxruntime.__version__, import_passed=True)
print(json.dumps(r))'
  else rc_all=1; fi
fi

if want speech-cosyvoice3; then
  echo "=== $(date -u +%FT%TZ) speech-cosyvoice3"
  CV_SHA=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc        # github.com/FunAudioLLM/CosyVoice (now QwenAudio/CosyVoice) main, 2026-05-25
  CV_TGZ_SHA256=10c587f0853db27667a64a0a2dc0788b1c782cdab223682691db78702082a550
  MT_SHA=dd9105b34bf2be2230f4aa1e4769fb586a3c824e        # the third_party/Matcha-TTS submodule commit at CV_SHA
  MT_TGZ_SHA256=59c43afae79110e4057eaa43f45b5558b556435da8fc85eab83f0c819ba3358d
  if mkvenv speech-cosyvoice3; then
    V=$RT/speech-cosyvoice3
    P=$V/bin/python
    SRC=$V/src
    mkdir -p "$SRC"
    ok=1
    for spec in "QwenAudio/CosyVoice:$CV_SHA:$CV_TGZ_SHA256:cosyvoice" "shivammehta25/Matcha-TTS:$MT_SHA:$MT_TGZ_SHA256:matcha"; do
      IFS=: read -r repo sha want_sha pkg <<< "$spec"
      tgz=$SRC/$(basename $repo)-$sha.tar.gz
      [ -f "$tgz" ] || curl -sSL --retry 3 --max-time 600 -o "$tgz" "https://codeload.github.com/$repo/tar.gz/$sha"
      got=$(sha256sum "$tgz" | cut -d' ' -f1)
      if [ "$got" != "$want_sha" ]; then echo "== $repo tarball sha256 $got != $want_sha: refusing"; ok=0; continue; fi
      # only the python package and the licence: the trees also hold symlinks exFAT cannot store, none of them needed
      tar -xzf "$tgz" -C "$SRC" "$(basename $repo)-$sha/$pkg" "$(basename $repo)-$sha/LICENSE" || ok=0
      echo "$SRC/$(basename $repo)-$sha" >> "$V/lib/python3.12/site-packages/aa_speech_cosyvoice3_src.pth.tmp"
    done
    sort -u "$V/lib/python3.12/site-packages/aa_speech_cosyvoice3_src.pth.tmp" > "$V/lib/python3.12/site-packages/aa_speech_cosyvoice3_src.pth"
    rm -f "$V/lib/python3.12/site-packages/aa_speech_cosyvoice3_src.pth.tmp"
    # The inference import closure of cosyvoice.cli.cosyvoice.CosyVoice3 + cosyvoice3.yaml (hyperpyyaml resolves every
    # !name/!new tag, so cosyvoice.dataset.processor -> pyarrow, pyworld, whisper and matcha.utils -> hydra, rich, gdown,
    # matplotlib, wget, lightning are imported at load even though inference never calls them).
    CV_DEPS="transformers==4.51.3 HyperPyYAML==1.2.3 onnxruntime inflect wetext x-transformers==2.11.24 conformer==0.3.2
             diffusers==0.29.0 lightning==2.2.4 omegaconf==2.3.0 hydra-core==1.3.2 modelscope librosa scipy soundfile pyarrow
             pyworld tiktoken more-itertools numba regex tqdm einops rich gdown matplotlib wget"
    # openai-whisper: upstream pins 20231117, whose setup.py needs pkg_resources (gone from current setuptools, build
    # failed 2026-10-15T03:16Z); 20250625 builds and keeps the log_mel_spectrogram + tokenizer API CosyVoice uses.
    # It declares triton>=2 (torch 2.6 ships triton 3.2): installed --no-deps, its runtime deps are in CV_DEPS
    [ $ok -eq 1 ] && resolve_gate speech-cosyvoice3 $CV_DEPS && \
    $P -m pip install -q -c $V/constraints.txt $CV_DEPS && \
    $P -m pip install -q --no-deps openai-whisper==20250625 || rc_all=1
    $P -m pip show openai-whisper 2>/dev/null | grep -E '^(Version|Requires):' | sed 's/^/== openai-whisper declared /'
    finish speech-cosyvoice3 "$HDR"'
from cosyvoice.cli.cosyvoice import CosyVoice3
import matcha, whisper, onnxruntime, hyperpyyaml, diffusers
r.update(cosyvoice="CosyVoice3", cosyvoice_src=__import__("cosyvoice").__file__, matcha_src=matcha.__file__,
         onnxruntime=onnxruntime.__version__, diffusers=diffusers.__version__, import_passed=True)
print(json.dumps(r))'
  else rc_all=1; fi
fi

du -sh --apparent-size $RT/speech-* $RT/.pip-cache 2>/dev/null; du -sh $RT/speech-* 2>/dev/null
echo "=== BUILD DONE $(date -u +%FT%TZ) rc_all=$rc_all"
echo $rc_all > "$OUT/exitcode"
