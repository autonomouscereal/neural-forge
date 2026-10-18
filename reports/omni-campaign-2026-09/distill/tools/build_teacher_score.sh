#!/bin/bash
# build_teacher_score.sh -- compile teacher_score.cpp against the export lane's libllama (llama.cpp b96806d9, CUDA 12.8,
# sm 61/75/86).  Host g++; the binary lands next to llama-server in $LLAMA/bin with a DT_RPATH (--disable-new-dtags: a RUNPATH is not searched for libggml.so's own libggml-cpu/cuda deps) to it, so it needs no
# LD_LIBRARY_PATH.  A receipt with the source and binary sha256 is written beside it.
set -euo pipefail
OC=/home/cereal/omni-campaign
T=$OC/distill/tools
LLAMA=$OC/export/llamacpp
SRC=$T/teacher_score.cpp
OUT=$LLAMA/bin/teacher-score
COMMIT=$(git -C "$LLAMA/llama.cpp" rev-parse HEAD)
g++ -O3 -march=native -std=c++17 -pthread -DTS_LLAMA_COMMIT="\"$COMMIT\"" \
  -I"$LLAMA/llama.cpp/include" -I"$LLAMA/llama.cpp/ggml/include" \
  "$SRC" -o "$OUT.tmp" -L"$LLAMA/bin" -lllama -lggml -lggml-base -Wl,--disable-new-dtags,-rpath,"$LLAMA/bin"
mv -f "$OUT.tmp" "$OUT"
python3 - "$SRC" "$OUT" "$COMMIT" <<'EOF'
import hashlib, json, sys, datetime
src, out, commit = sys.argv[1:4]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"tool": "teacher_score", "source": src, "source_sha256": h(src), "binary": out, "binary_sha256": h(out),
           "llama_cpp_commit": commit, "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
          open(out + ".receipt.json", "w"), indent=1)
print(open(out + ".receipt.json").read())
EOF
