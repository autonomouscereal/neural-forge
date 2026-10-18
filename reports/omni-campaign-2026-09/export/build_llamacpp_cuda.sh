#!/usr/bin/env bash
# Build llama.cpp with CUDA 12.8 for sm_61 (P40) + sm_75 (RTX 8000) + sm_86 (RTX 3090)
# inside an nvidia/cuda devel container, because the host has no nvcc.
#
# Produces:
#   <ROOT>/bin/{llama-server,llama-cli,llama-quantize,llama-gguf,llama-bench}
#   <ROOT>/llama.cpp/                      (source tree; convert_hf_to_gguf.py lives here)
#   <ROOT>/build-receipt.json              (commit, arches, per-step wall, binary sha256s)
#   <ROOT>/logs/build-<stamp>.log
#
# Every step is timed and appended to <ROOT>/logs/steps.jsonl. Re-running is safe: an
# existing source tree is reused (git fetch/checkout), an existing build dir is reused by
# cmake. Nothing outside <ROOT> is written. No container other than the one this script
# starts is touched.
#
# Usage:
#   ./build_llamacpp_cuda.sh                              # defaults
#   ./build_llamacpp_cuda.sh --arch "61;75;86" --jobs 48
#   ./build_llamacpp_cuda.sh --ref b7000                  # pin a llama.cpp tag/commit
set -euo pipefail

ROOT="/home/cereal/omni-campaign/export/llamacpp"
IMAGE="nvidia/cuda:12.8.1-devel-ubuntu22.04"
ARCHES="61;75;86"
JOBS="$(nproc)"
REPO="https://github.com/ggml-org/llama.cpp.git"
REF=""
VENV_PY="/home/cereal/model-surgery-lab/venv/bin/python"
# The lab venv is SHARED with other lanes and pip takes no lock on site-packages: two
# concurrent installs can leave half-written packages that surface later as somebody
# else's ImportError. gguf-py is therefore NOT installed by default -- export_hf_to_gguf.py
# picks up <ROOT>/llama.cpp/gguf-py via PYTHONPATH, which touches no shared state and
# guarantees the gguf-py of the same commit. Pass --venv-gguf to opt back in.
INSTALL_GGUF_PY=0
CONTAINER_NAME="oc-llamacpp-build-$$"

while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --arch) ARCHES="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --venv-python) VENV_PY="$2"; shift 2 ;;
    --no-venv-gguf) INSTALL_GGUF_PY=0; shift ;;
    --venv-gguf) INSTALL_GGUF_PY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${ROOT}/logs"
STEPS="${LOG_DIR}/steps.jsonl"
mkdir -p "${ROOT}" "${LOG_DIR}" "${ROOT}/bin"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

step() {
  # step <name> <timeout_s> <command...>
  local name="$1"; shift
  local tmo="$1"; shift
  local t0 t1 rc
  t0=$(date +%s)
  log "STEP ${name} (timeout ${tmo}s)"
  set +e
  timeout "${tmo}" "$@"
  rc=$?
  set -e
  t1=$(date +%s)
  printf '{"step":"%s","rc":%d,"wall_s":%d,"stamp":"%s"}\n' \
    "${name}" "${rc}" "$((t1 - t0))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STEPS}"
  log "STEP ${name} rc=${rc} wall=$((t1 - t0))s"
  return ${rc}
}

log "root=${ROOT} image=${IMAGE} arches=${ARCHES} jobs=${JOBS} ref=${REF:-<default-branch>}"

# ---------------------------------------------------------------------------- image
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  step docker_pull 3000 docker pull "${IMAGE}"
else
  log "image already present: ${IMAGE}"
  printf '{"step":"docker_pull","rc":0,"wall_s":0,"stamp":"%s","note":"image already present"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STEPS}"
fi

# ---------------------------------------------------------------------------- source
if [ ! -d "${ROOT}/llama.cpp/.git" ]; then
  step git_clone 900 git clone "${REPO}" "${ROOT}/llama.cpp"
else
  log "source tree present, fetching"
  step git_fetch 900 git -C "${ROOT}/llama.cpp" fetch --all --tags --prune
fi
if [ -n "${REF}" ]; then
  step git_checkout 120 git -C "${ROOT}/llama.cpp" checkout --detach "${REF}"
fi
COMMIT="$(git -C "${ROOT}/llama.cpp" rev-parse HEAD)"
DESCRIBE="$(git -C "${ROOT}/llama.cpp" describe --tags --always 2>/dev/null || echo unknown)"
log "llama.cpp commit=${COMMIT} describe=${DESCRIBE}"

# ---------------------------------------------------------------------------- build
BUILD_SCRIPT="${ROOT}/_container_build.sh"
cat > "${BUILD_SCRIPT}" <<'CONTAINER_EOF'
#!/usr/bin/env bash
set -euo pipefail
ARCHES="$1"; JOBS="$2"; HOST_UID="$3"; HOST_GID="$4"
export DEBIAN_FRONTEND=noninteractive
echo "== apt =="
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential cmake git ccache libcurl4-openssl-dev pkg-config ca-certificates
echo "== versions =="
cmake --version | head -1
gcc --version | head -1
nvcc --version | tail -2

# The devel image has no libcuda.so.1 (that lib belongs to the DRIVER, not the toolkit).
# ggml-cuda.so needs it at link time for cuMem*/cuDevice* (the VMM allocator path), so the
# final executable link fails with "undefined reference to cuMemCreate" unless the driver
# stub is on the linker path. Two belts: the real driver if the container was started with
# --gpus, otherwise the toolkit's stub with an explicit SONAME symlink.
STUBS=/usr/local/cuda/lib64/stubs
if [ -f "${STUBS}/libcuda.so" ] && [ ! -e "${STUBS}/libcuda.so.1" ]; then
  ln -sf "${STUBS}/libcuda.so" "${STUBS}/libcuda.so.1"
fi
export LIBRARY_PATH="${STUBS}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${STUBS}:${LD_LIBRARY_PATH:-}"
echo "== driver libs visible to the linker =="
ls -l "${STUBS}" 2>/dev/null | head -5
ldconfig -p 2>/dev/null | grep -c libcuda || true

echo "== configure =="
cd /work/llama.cpp
cmake -B /work/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCHES}" \
  -DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${STUBS} -Wl,-rpath-link,${STUBS}" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${STUBS} -Wl,-rpath-link,${STUBS}" \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON
echo "== build =="
cmake --build /work/build --config Release -j "${JOBS}" \
  --target llama-server llama-cli llama-quantize llama-gguf llama-bench
echo "== collect =="
# The 2026 llama.cpp layout ships THIN launchers (~18 KB) plus libllama-<tool>-impl.so,
# libllama-common.so.0, libllama.so, libggml*.so.<ver>. A glob of "*.so" misses every
# versioned library and the binaries then die with
# "error while loading shared libraries: libllama-common.so.0". Copy the whole build/bin
# tree, preserving symlinks (cp -a), and sweep any stray versioned .so elsewhere.
mkdir -p /work/bin
if [ -d /work/build/bin ]; then
  cp -a /work/build/bin/. /work/bin/
  echo "copied build/bin -> bin"
fi
find /work/build -path /work/build/bin -prune -o -type f -name "*.so*" -print 2>/dev/null | \
  while read -r lib; do
    base="$(basename "${lib}")"
    [ -e "/work/bin/${base}" ] || { cp -f "${lib}" /work/bin/; echo "swept ${base}"; }
  done
for binary in llama-server llama-cli llama-quantize llama-gguf llama-bench; do
  if [ ! -x "/work/bin/${binary}" ]; then
    found="$(find /work/build -maxdepth 4 -type f -name "${binary}" | head -1)"
    [ -n "${found}" ] && cp -f "${found}" /work/bin/ && echo "collected ${binary} (fallback)"
  fi
done
# The HOST has the NVIDIA driver but no CUDA toolkit (no nvcc, no libcudart.so.12), so the
# binaries must carry the CUDA runtime with them. libcuda.so.1 is deliberately NOT bundled:
# that one belongs to the driver and must come from the host at runtime.
echo "== bundle CUDA runtime libs =="
SEARCH_DIRS="/usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib /usr/lib/x86_64-linux-gnu"
for dir in ${SEARCH_DIRS}; do
  [ -d "${dir}" ] || continue
  # libnccl lives outside /usr/local/cuda in the devel image; recent llama.cpp links it
  for pattern in libcudart libcublas libcublasLt libnccl; do
    cp -a "${dir}"/${pattern}.so* /work/bin/ 2>/dev/null || true
  done
done
# whatever is still unresolved, hunted by name across the same dirs (two passes: a bundled
# lib can pull in another)
for pass in 1 2; do
  for binary in /work/bin/llama-server /work/bin/llama-cli /work/bin/llama-quantize; do
    [ -x "${binary}" ] || continue
    LD_LIBRARY_PATH=/work/bin ldd "${binary}" 2>/dev/null | awk '/not found/ {print $1}'
  done | sort -u | while read -r missing; do
    case "${missing}" in
      libcuda.so*) echo "skipping driver lib ${missing} (the host provides it)"; continue ;;
    esac
    hit="$(find ${SEARCH_DIRS} -name "${missing}" 2>/dev/null | head -1)"
    if [ -n "${hit}" ]; then cp -aL "${hit}" "/work/bin/${missing}"; echo "bundled ${missing}"; fi
  done
done

echo "== in-container smoke (LD_LIBRARY_PATH=/work/bin) =="
LD_LIBRARY_PATH=/work/bin /work/bin/llama-cli --version 2>&1 | head -3 || true
LD_LIBRARY_PATH=/work/bin /work/bin/llama-quantize --help 2>&1 | head -3 || true
chown -R "${HOST_UID}:${HOST_GID}" /work
echo "== done =="
CONTAINER_EOF
chmod +x "${BUILD_SCRIPT}"

# --gpus all makes the real libcuda.so.1 available inside the container (the NVIDIA
# runtime mounts it and runs ldconfig). No GPU memory is allocated by a compile/link; if
# the flag is refused, the stub path inside the container script covers the link anyway.
BUILD_RC=0
if ! step docker_build 21600 \
    docker run --rm --name "${CONTAINER_NAME}" \
      --gpus all \
      -v "${ROOT}:/work" \
      -w /work \
      "${IMAGE}" \
      bash /work/_container_build.sh "${ARCHES}" "${JOBS}" "$(id -u)" "$(id -g)"; then
  BUILD_RC=1
  log "docker_build reported a failure; continuing to collect whatever was produced"
fi

# ---------------------------------------------------------------------------- smoke
# NO PIPES HERE. `cmd | head` records the exit status of `head`, and this build already
# produced three receipt rows reading `smoke_llama_quantize_help rc=0` for a binary whose
# `llama-cli --version` was rc=127 -- a green step over a binary that could not start.
# Output goes to a file; the receipt block below asserts on its CONTENT.
export LD_LIBRARY_PATH="${ROOT}/bin:${LD_LIBRARY_PATH:-}"
QUANT_HELP_LOG="${LOG_DIR}/smoke-llama-quantize-help.txt"
CLI_VERSION_LOG="${LOG_DIR}/smoke-llama-cli-version.txt"
: > "${QUANT_HELP_LOG}"
: > "${CLI_VERSION_LOG}"

step_capture() {
  # step_capture <name> <timeout_s> <outfile> <command...>  -- rc of the COMMAND, no pipe
  local name="$1"; shift
  local tmo="$1"; shift
  local outfile="$1"; shift
  local t0 t1 rc
  t0=$(date +%s)
  log "STEP ${name} (timeout ${tmo}s) -> ${outfile}"
  set +e
  timeout "${tmo}" "$@" > "${outfile}" 2>&1
  rc=$?
  set -e
  t1=$(date +%s)
  printf '{"step":"%s","rc":%d,"wall_s":%d,"stamp":"%s","output_file":"%s"}\n' \
    "${name}" "${rc}" "$((t1 - t0))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${outfile}" >> "${STEPS}"
  log "STEP ${name} rc=${rc} wall=$((t1 - t0))s"
  head -3 "${outfile}" || true
  return ${rc}
}

if [ -x "${ROOT}/bin/llama-cli" ]; then
  step_capture smoke_llama_cli_version 120 "${CLI_VERSION_LOG}" \
    "${ROOT}/bin/llama-cli" --version || true
fi
if [ -x "${ROOT}/bin/llama-quantize" ]; then
  step_capture smoke_llama_quantize_help 120 "${QUANT_HELP_LOG}" \
    "${ROOT}/bin/llama-quantize" --help || true
fi

# ---------------------------------------------------------------------------- gguf-py
# Default: touch nothing. The lab venv is shared and concurrently written by other lanes.
GGUF_STATUS="not_installed_use_PYTHONPATH=${ROOT}/llama.cpp/gguf-py"
if [ "${INSTALL_GGUF_PY}" = "1" ] && [ -x "${VENV_PY}" ]; then
  log "WARNING: --venv-gguf given; installing into the SHARED lab venv ${VENV_PY}"
  if step pip_install_gguf_py 900 "${VENV_PY}" -m pip install --disable-pip-version-check \
      "${ROOT}/llama.cpp/gguf-py"; then
    GGUF_STATUS="pip"
  else
    GGUF_STATUS="pip_failed_use_pythonpath"
  fi
  if "${VENV_PY}" -c "import gguf" >/dev/null 2>&1; then
    log "gguf importable in the lab venv"
  else
    GGUF_STATUS="not_importable_use_PYTHONPATH=${ROOT}/llama.cpp/gguf-py"
    log "WARNING: gguf not importable; export_hf_to_gguf.py must be given --gguf-py-path"
  fi
fi

# ---------------------------------------------------------------------------- receipt
OC_VENV_PY="${VENV_PY}" \
"${VENV_PY:-python3}" - "$ROOT" "$COMMIT" "$DESCRIBE" "$ARCHES" "$IMAGE" "$JOBS" "$GGUF_STATUS" "$BUILD_RC" <<'PY'
import json, os, subprocess, sys
sys.path.insert(0, "/home/cereal/omni-campaign/export")
from llama_probe import sha256_file, write_receipt, utc_now, run

root, commit, describe, arches, image, jobs, gguf_status, build_rc = sys.argv[1:9]
bin_dir = os.path.join(root, "bin")
binaries = {}
for name in sorted(os.listdir(bin_dir)) if os.path.isdir(bin_dir) else []:
    path = os.path.join(bin_dir, name)
    if os.path.isfile(path):
        binaries[name] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}

steps = []
steps_path = os.path.join(root, "logs", "steps.jsonl")
if os.path.exists(steps_path):
    with open(steps_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    steps.append(json.loads(line))
                except ValueError:
                    pass

env = dict(os.environ, LD_LIBRARY_PATH=bin_dir)
rc, out, err = run([os.path.join(bin_dir, "llama-cli"), "--version"], timeout=60, env=env)
version_text = (out + err).strip().splitlines()[:6]

# artifact check, not exit-code check: does the produced binary actually resolve?
ldd_rc, ldd_out, ldd_err = run(["ldd", os.path.join(bin_dir, "llama-server")], timeout=60, env=env)
unresolved = [line.strip() for line in (ldd_out + ldd_err).splitlines() if "not found" in line]

# CONTENT assertions. rc alone is worthless here: llama-quantize --help exits non-zero on
# some builds, and a binary that cannot start still gives rc=0 through a pipe. What matters
# is whether the quant types this lane converts to are actually offered by THIS binary.
quant_rc, quant_out, quant_err = run(
    [os.path.join(bin_dir, "llama-quantize"), "--help"], timeout=120, env=env
)
quant_help = quant_out + quant_err
server_rc, server_out, server_err = run(
    [os.path.join(bin_dir, "llama-server"), "--help"], timeout=120, env=env
)
server_help = server_out + server_err


def assert_contains(text, needle):
    line = next((row.strip() for row in text.splitlines() if needle in row), None)
    return {"needle": needle, "present": line is not None, "line": (line or "")[:200]}


smoke_assertions = {
    "llama_cli_version_rc": rc,
    "llama_cli_version_ok": rc == 0 and any("version" in line.lower() for line in version_text),
    "llama_quantize_help_rc": quant_rc,
    "llama_quantize_help_bytes": len(quant_help),
    "llama_quantize_offers": [
        assert_contains(quant_help, "Q4_K_M"),
        assert_contains(quant_help, "IQ4_XS"),
    ],
    "llama_server_flags": [
        assert_contains(server_help, "--flash-attn"),
        assert_contains(server_help, "--cache-type-k"),
    ],
}
smoke_assertions["all_present"] = all(
    item["present"]
    for item in smoke_assertions["llama_quantize_offers"] + smoke_assertions["llama_server_flags"]
) and bool(smoke_assertions["llama_cli_version_ok"])

# Shared-venv state, recorded rather than assumed: which gguf (if any) is importable there,
# what pip reports, and whether the venv is still internally consistent.
venv_python = os.environ.get("OC_VENV_PY", "")
venv_state = {"venv_python": venv_python}
if venv_python and os.path.exists(venv_python):
    _, freeze_out, _ = run([venv_python, "-m", "pip", "freeze"], timeout=180)
    venv_state["gguf_freeze_lines"] = [
        line.strip() for line in freeze_out.splitlines() if "gguf" in line.lower()
    ]
    check_rc, check_out, check_err = run([venv_python, "-m", "pip", "check"], timeout=180)
    venv_state["pip_check_rc"] = check_rc
    venv_state["pip_check_output"] = (check_out + check_err).strip().splitlines()[-8:]

receipt = {
    "kind": "llamacpp_cuda_build",
    "created_at": utc_now(),
    "root": root,
    "image": image,
    "cuda_architectures": arches,
    "jobs": jobs,
    "llama_cpp_commit": commit,
    "llama_cpp_describe": describe,
    "binaries": binaries,
    "llama_cli_version_lines": version_text,
    "llama_cli_version_rc": rc,
    "llama_server_unresolved_libs": unresolved,
    "runtime_env": {"LD_LIBRARY_PATH": bin_dir},
    "smoke_assertions": smoke_assertions,
    "shared_venv_state": venv_state,
    "gguf_py_status": gguf_status,
    "gguf_py_pythonpath": os.path.join(root, "llama.cpp", "gguf-py"),
    "build_rc": int(build_rc),
    "expected_binaries": ["llama-server", "llama-cli", "llama-quantize", "llama-gguf", "llama-bench"],
    "missing_binaries": [
        name for name in ("llama-server", "llama-cli", "llama-quantize", "llama-gguf", "llama-bench")
        if name not in binaries
    ],
    "steps": steps,
}
path = os.path.join(root, "build-receipt.json")
digest = write_receipt(path, receipt)
print("build receipt: %s sha256=%s" % (path, digest))
print("binaries: %s" % ", ".join(sorted(binaries)))
print("llama-cli --version rc=%s: %s" % (rc, version_text[:2]))
for item in smoke_assertions["llama_quantize_offers"] + smoke_assertions["llama_server_flags"]:
    print("  assert %-16s present=%-5s :: %s" % (item["needle"], item["present"], item["line"][:90]))
if not smoke_assertions["all_present"]:
    print("SMOKE ASSERTION FAILED: the built binaries do not offer everything this lane needs")
if unresolved:
    print("UNRESOLVED SHARED LIBS for llama-server: %s" % unresolved)
PY

log "BUILD LANE COMPLETE root=${ROOT}"
