# EXPORT + ENVELOPE LANE — Chimera Omni campaign, 2026-09-01

Lane root (workstation): `C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/export/`
Lane root (server): `/home/cereal/omni-campaign/export/`

Everything in this lane **measures and records**. No file here contains a pass/fail gate.
Every number written into a receipt is a reported measurement; the decision lines are
Cyril's (`plan §10`, `§11`). Standing rules honoured: quantization happens once at the very
end and is re-graded after; token match is a diagnostic, never a grade; artifacts are
verified by hash, never by exit code; no container this lane did not start is ever stopped;
every remote command is wrapped in `timeout`.

---

## 0. Files

| File | What it is |
|---|---|
| `llama_probe.py` | Shared library: canonical-JSON receipts, sha256 + checkpoint directory digests, llama-server HTTP client (greedy), llama.cpp **log signal parser**, `nvidia-smi` peak sampler, needle-in-a-haystack builder + scorer, `docker logs` head/tail reader. Stdlib only. |
| `build_llamacpp_cuda.sh` | (a) CUDA 12.8 llama.cpp build for `sm_61;sm_75;sm_86` inside `nvidia/cuda:12.8.1-devel-ubuntu22.04` (the host has no nvcc). Emits `llama-server/llama-cli/llama-quantize/llama-gguf/llama-bench` + `build-receipt.json` + installs `gguf-py` into the lab venv. |
| `export_hf_to_gguf.py` | (b) HF checkpoint -> `convert_hf_to_gguf.py` -> `llama-quantize` Q4_K_M / IQ4_XS, with a source-directory digest and a per-stage export receipt. |
| `parity_bank.json` | (b) `EXPORT-PB1`: 15 prompts, export-fidelity instrument only. **Deliberately disjoint from the frozen 15-case review and bank v2** — those surfaces must never be trained on or run through an ad-hoc harness. |
| `parity_check.py` | (b) Runs the bank through llama-server (GGUF) and through the source HF checkpoint (transformers), byte-identical prompts, greedy on both. Emits `parity-<tag>.json` + `READ-<tag>.txt` with every answer verbatim. |
| `envelope_bench.py` | (c) The 256k envelope protocol: `-c 262144 -fa on -np 1`, matched `q8_0/q8_0` or `q4_0/q4_0` KV, FA-kernel evidence from the load banner, peak VRAM, prefill wall at 4k/32k/128k/256k, decode tok/s, needle at 32k/64k/128k/256k x depth. |
| `p40probe_check.py` | (d) **Lifecycle** read-only inspection of the running P40 envelope container + short-context sanity completion. Contains no container lifecycle call; two tests (a literal grep and an AST walk that also catches `docker restart` and non-constant arguments) prove it. It is *not* resource read-only: without `--skip-completion` it occupies a slot on a server another lane owns. |
| `fa_negative_control.py` | (e) The control that licenses or voids the R3 inference: starts this build with `-fa off -ctk q4_0 -ctv q4_0`, then with `-fa on`, and records what happened. `llama_probe.load_r3_control()` reads its receipt, so every FA reading in the lane moves with the measurement instead of with an assumption. |
| `fixtures/Qwen3.5-0.8B.config.json` | Byte-identical copy of the real checkpoint config (sha256 `b90b86f3…`, verified against the server) so the nested-config parsing is pinned against the thing the lane actually converted, not a synthetic flat config. |
| `kv_fit_search.py` | (c) The follow-up the q8_0 result forced: bisects the largest `-c` a (model, KV type, card) triple actually LOADS at, by launching the server at each candidate and recording the verbatim allocator lines. Host runner only; stops only its own server. |
| `wait_free_vram.sh` | (c) Waits for a card to have N MiB free, then `exec`s the command after `--`. Written because the 3090 sweep died at load against another lane's resident 9,218 MiB job, and that process is not ours to stop. Starts nothing if the card never frees. |
| `rerun_parity_iq4xs.sh` | (b) Re-measures the IQ4_XS parity receipts under the flags §2.3 documents, runs the bank twice against one server as a determinism control, and stops the one unit it started. |
| `compare_parity_passes.py` | (b) Row-for-row diff of two parity receipts taken with identical arguments. Reports; sets no threshold. |
| `test_export_lane.py` | 145 stdlib unit tests over the parser, receipts, needle, calibration, context headroom, the KV-fit bisect, export plan, parity comparison, the FA evidence provenance chain, container-state resolution, and the container-safety invariants. |

---

## 1. (a) The CUDA 12.8 build recipe

The host has no `nvcc` (verified: `which nvcc` -> nothing). The build therefore runs inside
`nvidia/cuda:12.8.1-devel-ubuntu22.04` with the work tree bind-mounted, and the artefacts
are `chown`ed back to the calling uid before the container exits.

Architectures, and why all three:

| SM | Card on this box | Needed for |
|---|---|---|
| `61` | Tesla P40 x4 (GPU1/2/3/5) | the 24 GB Q4 envelope target; no bf16, fp16 at 1/64 rate |
| `75` | Quadro RTX 8000 46 GB (GPU6) | K3 streamed layers, H3, large-KV runs |
| `86` | RTX 3090 x2 (GPU0/GPU4) | primary bf16 review + envelope |

CMake flags used:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;75;86"
-DCMAKE_CUDA_FLAGS="-Wno-deprecated-gpu-targets"
-DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"
-DCMAKE_SHARED_LINKER_FLAGS=<same>
-DGGML_NATIVE=OFF -DLLAMA_CURL=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF
```

`GGML_NATIVE=OFF` because the binaries must run on this heterogeneous box under whatever
CPU the container reports; `LLAMA_CURL=ON` keeps `-hf` working; tests are off because the
build is already three CUDA architectures wide. nvcc warns that sm_61 offline compilation
is deprecated in a future CUDA release — recorded, suppressed, still produced.

### 1.1 Four things that actually broke, and the fixes (all four are in the script)

Each one produced a green-looking step and a broken artifact, which is why the script
smoke-runs the binary and records `ldd` output in the receipt rather than trusting `rc=0`.

1. **`undefined reference to cuMemCreate` at the final link.** `libcuda.so.1` belongs to the
   *driver*, not the toolkit, and a plain `docker run` of a devel image has neither. Fix:
   run the build container with `--gpus all` (the NVIDIA runtime mounts the real driver
   lib) **and** symlink + link-path the toolkit stub at `/usr/local/cuda/lib64/stubs`, so
   either route works.
2. **Thin launchers with no libraries.** The 2026 layout ships ~18 KB `llama-server` /
   `llama-cli` launchers plus `libllama-<tool>-impl.so`, `libllama-common.so.0`,
   `libggml-cuda.so.0.22.0`. A `find -name "*.so"` collect misses every **versioned**
   library, and the binaries then die with
   `error while loading shared libraries: libllama-common.so.0`. Fix: `cp -a build/bin/.`
   (whole tree, symlinks preserved) plus a `*.so*` sweep.
3. **No CUDA runtime on the host.** The host has the driver but no toolkit, so
   `libcudart.so.12` / `libcublas` / `libcublasLt` must travel with the binaries. Fix: the
   script bundles them from the container into `llamacpp/bin`. `libcuda.so*` is explicitly
   **not** bundled — that one must come from the host driver at run time.
4. **`libnccl.so.2` not found.** Recent llama.cpp links NCCL, which lives in
   `/usr/lib/x86_64-linux-gnu`, outside `/usr/local/cuda`. Fix: the bundling sweep searches
   both trees and runs twice (a bundled lib can pull in another).

Result: `llamacpp/bin` is ~1.3 GB (binaries + ggml + the bundled CUDA runtime) and runs on
the bare host with `LD_LIBRARY_PATH=/home/cereal/omni-campaign/export/llamacpp/bin`.

### 1.2 Measured build (2026-09-02, this box)

**Hashes are quoted from the receipt field, never re-typed.** Every number below with a
`receipt:` prefix names the field it came from; read the field, not this table, if the two
ever disagree. (They did once: an earlier revision of this spec published a build-receipt
sha256 that matched no artifact anywhere, while the export receipt's `build_receipt_sha256`
had the real one. Copy-pasted hashes are how that happens, so they are cited by reference
now and re-verified with `sha256sum` before publication.)

| Fact | Value |
|---|---|
| Image | `nvidia/cuda:12.8.1-devel-ubuntu22.04` (9.34 GB, pulled) |
| llama.cpp | commit `b96806d96061049a5b574269b049bf6241d63d46`, describe **`b10752`** (receipt: `llama_cpp_commit` / `llama_cpp_describe`) |
| Toolchain in container | cmake 3.22.1, gcc 11.4.0, nvcc release 12.8 V12.8.93 |
| `git clone` | 55 s |
| `docker_build` (3 arches, `-j44`) | 402 s to the link failure; 341 s for the fixed full build; 35-39 s incremental (receipt: `steps[]`) |
| Binaries | `llama-server llama-cli llama-quantize llama-gguf llama-bench` + 37 libraries, all sha256'd (receipt: `binaries`) |
| `llama-cli --version` on the host | `version: 0.3.0-dev (build 0, commit unknown) / built with GNU 11.4.0 for Linux x86_64` (receipt: `llama_cli_version_lines`) |
| `ldd llama-server` unresolved | **none** (receipt: `llama_server_unresolved_libs`) |
| `llama-quantize` offers | `15 or Q4_K : alias for Q4_K_M`, `30 or IQ4_XS : 4.25 bpw non-linear quantization` — asserted by CONTENT (receipt: `smoke_assertions.llama_quantize_offers`) |
| `llama-server` flags | `-fa, --flash-attn [on\|off\|auto]`, `-ctk/--cache-type-k` — asserted by CONTENT (receipt: `smoke_assertions.llama_server_flags`) |
| `llama-quantize --help` exit code | **rc 1** — this binary exits non-zero on `--help`, which is exactly why the smoke check now asserts on output text (receipt: `smoke_assertions.llama_quantize_help_rc`) |
| gguf-py | NOT installed by this script any more; used from `llamacpp/llama.cpp/gguf-py` via PYTHONPATH (receipt: `gguf_py_status`, `gguf_py_pythonpath`) |
| Shared lab venv state | recorded read-only: `gguf @ file:///…/llama.cpp/gguf-py` from the earlier install, `pip check` rc 0 "No broken requirements found." (receipt: `shared_venv_state`) |
| Build receipt | `llamacpp/build-receipt.json`, sha256 verified with `sha256sum` on the server **after** this rewrite: `4a9554feae73a0db10ee356a2232355300f1b1b765d93cbeec51edbeea657607`. The previous receipt is kept at `build-receipt.json.bak-20260902T013109Z`. |

#### 1.2.1 A green step over a binary that could not start

The smoke check used to be `bash -c "'…/llama-quantize' --help 2>&1 | head -40"`. That
records the exit status of **`head`**, and the outer script's `set -o pipefail` does not
reach inside a nested `bash -c`. The receipt's own ledger showed the consequence three
times (00:17:37, 00:19:30, 00:25:52): `smoke_llama_cli_version` rc **127** — the binary
could not start at all — sitting next to `smoke_llama_quantize_help` rc **0** on the same
broken tree. Three green rows for a dead binary, in the lane whose doctrine is "verify
artifacts, never exit codes".

The fix is in `build_llamacpp_cuda.sh`: `step_capture` runs the command with **no pipe**
and redirects to a file, so the recorded rc is the command's (it is now honestly `1`), and
the receipt block asserts on the *content* of that output — that `Q4_K_M` and `IQ4_XS` are
actually offered and that `--flash-attn` and `--cache-type-k` actually exist — with the
matching line kept verbatim in `smoke_assertions`.

#### 1.2.2 gguf-py no longer touches the shared venv

The old default pip-installed `gguf-py` into `/home/cereal/model-surgery-lab/venv`, which
other lanes use and which was being written concurrently by another install (fla +
compressed-tensors). pip takes no lock on `site-packages`. The default is now
`--no-venv-gguf` behaviour (`--venv-gguf` opts back in, with a warning), and
`export_hf_to_gguf.py` puts `<root>/llama.cpp/gguf-py` on `PYTHONPATH` itself — which also
guarantees the converter runs against the gguf-py of the same commit. The venv's state is
still *recorded* (`pip freeze | grep gguf`, `pip check`) so the earlier install is visible:
it reports as `gguf @ file:///…/gguf-py` with **no version string**, which is why the
"gguf 0.19.0" in the previous summary had nothing behind it.

Run it:

```bash
# on the server, detached (the build is long; the SSH lease must not hold it)
cd /home/cereal/omni-campaign/export
setsid nohup ./build_llamacpp_cuda.sh --arch "61;75;86" --jobs 44 \
    > llamacpp/logs/build-driver.log 2>&1 < /dev/null &

# progress
tail -f /home/cereal/omni-campaign/export/llamacpp/logs/build-driver.log
cat     /home/cereal/omni-campaign/export/llamacpp/logs/steps.jsonl
```

Outputs: `llamacpp/bin/*`, `llamacpp/llama.cpp/` (source, incl. `convert_hf_to_gguf.py`),
`llamacpp/build-receipt.json` (llama.cpp commit, arch list, per-step wall, **sha256 of every
binary**, `llama-cli --version` lines, gguf-py install status).

`gguf-py`: not installed anywhere by default. `export_hf_to_gguf.py` prepends
`<root>/llama.cpp/gguf-py` to `PYTHONPATH` automatically (`--gguf-py-path` overrides), and
the receipt records the path it used in `gguf_py_path_used`.

---

## 2. (b) HF -> GGUF -> Q4_K_M / IQ4_XS, with parity

```bash
VENV=/home/cereal/model-surgery-lab/venv/bin/python
cd /home/cereal/omni-campaign/export

# 1. convert + quantize (dry-run first prints the exact commands and touches nothing)
$VENV export_hf_to_gguf.py --hf-dir /home/cereal/models/Qwen3.5-0.8B \
    --out-dir /home/cereal/omni-campaign/export/gguf --tag Q35-0.8B-stock \
    --outtype bf16 --quants Q4_K_M,IQ4_XS --dry-run
$VENV export_hf_to_gguf.py --hf-dir /home/cereal/models/Qwen3.5-0.8B \
    --out-dir /home/cereal/omni-campaign/export/gguf --tag Q35-0.8B-stock \
    --outtype bf16 --quants Q4_K_M,IQ4_XS

# 2. serve the Q4 (our build, GPU4 by UUID; CUDA_VISIBLE_DEVICES by index is unsafe here)
export LD_LIBRARY_PATH=/home/cereal/omni-campaign/export/llamacpp/bin:$LD_LIBRARY_PATH
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582 \
  /home/cereal/omni-campaign/export/llamacpp/bin/llama-server \
  -m /home/cereal/omni-campaign/export/gguf/Q35-0.8B-stock.Q4_K_M.gguf \
  -c 32768 -fa on -np 1 -ngl 999 --host 127.0.0.1 --port 18090 \
  > /home/cereal/omni-campaign/export/logs/q35-08b-q4.log 2>&1 &

# 3. 15-prompt parity, GGUF vs the HF source it came from
$VENV parity_check.py --tag Q35-0.8B-Q4KM \
    --server-url http://127.0.0.1:18090 \
    --gguf-path /home/cereal/omni-campaign/export/gguf/Q35-0.8B-stock.Q4_K_M.gguf \
    --hf-dir /home/cereal/models/Qwen3.5-0.8B --hf-device cuda:0 --hf-dtype bfloat16 \
    --out-dir /home/cereal/omni-campaign/export/parity
```

Reading the parity result: `exact_matches` at Q4 is expected to be **well under 15/15**.
A quantized model diverging in wording is a normal artifact, not a defect, and this estate
does not grade on token match. What the receipt is for is (i) proving the export runs at
all, (ii) showing WHERE it diverges (`char_prefix` per prompt), and (iii) giving a human
both answers side by side in `READ-<tag>.txt`. `--skip-hf` runs the GGUF side alone (no
torch needed); `--reference` compares a later GGUF against an earlier HF run.

### 2.1 Measured export (2026-09-02): Qwen3.5-0.8B, the real thing

`/home/cereal/models/Qwen3.5-0.8B` converts and quantizes with **no patches to llama.cpp** —
the `qwen35` path in commit `b96806d9` takes this architecture as written:

| Stage | Wall | Output | sha256 (16) |
|---|---|---|---|
| `convert_hf_to_gguf.py --outtype bf16` | 20.14 s | 1.56 GB `Q35-08B-stock.bf16.gguf` | `693897b38e938c69` |
| `llama-quantize … Q4_K_M 44` | 7.66 s | 0.54 GB `Q35-08B-stock.Q4_K_M.gguf` | `7c6004cc50be9739` |
| `llama-quantize … IQ4_XS 44` | 11.64 s | 0.52 GB `Q35-08B-stock.IQ4_XS.gguf` (no imatrix) | `328fd4c2a04ab97d` |

Receipt: `gguf/export-receipt-Q35-08B-stock.json`, re-generated after the architecture fix
below, sha256 `d2b78107ad4515841d584b6255e313386a3031c1b40ac89be2a7ab85843493c5` (the
pre-fix receipt is kept as `.bak-20260902T013415Z`; the three GGUF hashes are unchanged
because the artifacts were reused, not rebuilt).

#### 2.1.1 The receipt used to be full of nulls on exactly this family

`read_hf_arch` read `config.get("hidden_size")` and friends from the **top level**.
`Qwen3_5ForConditionalGeneration` keeps the text tower under `text_config`, so the shipped
receipt recorded `hidden_size: null, num_hidden_layers: null, vocab_size: null,
max_position_embeddings: null, layer_types_len: null` — no usable architecture record for a
lineage join — while every architectural fact in the lane summary was hand-read narrative
with nothing behind it. Worse, `tie_word_embeddings` *was* read, from the top level, so a
model where the two levels disagreed would have gotten a silently wrong value.

`read_hf_arch` now walks `text_config` / `llm_config` / `thinker_config` /
`language_config` / `decoder_config` / `llm_cfg`, and records **where each field came
from**, the top-level values alongside, the raw `config.json` sha256, and a loud
`warnings` list (also echoed to stderr and copied to the receipt's top level as
`hf_config_warnings`) whenever a required field still resolves to `None`. Measured on the
real checkpoint:

```
hf arch: Qwen3_5ForConditionalGeneration (qwen3_5) layers=24 hidden=1024 vocab=248320
         ctx=262144  [sub-configs: text_config]
```

with `head_dim 256`, `attn_output_gate true`, `full_attention_interval 4`,
`num_key_value_heads 2`, `layer_types_counts {full_attention: 6, linear_attention: 18}`,
`config_sha256 b90b86f35c8e6925…`, `warnings []` — all now *in the receipt*.
`test_read_hf_arch_pinned_to_the_real_checkpoint_config` pins this against a byte-identical
copy of the server's `config.json` in `fixtures/`, the same way the log tests are pinned
against real log text. The old test built a flat synthetic config, asserted the values it
had just written, and is the reason a receipt full of nulls shipped with 54 green tests.

### 2.2 Measured parity (EXPORT-PB1, Q4_K_M vs the bf16 HF source)

GGUF on GPU4 through our own `llama-server`; HF through transformers 5.12.1 bf16 on the same
card; identical prompt text; greedy on both.

```
{"compared": 15, "exact_matches": 2, "normalized_matches": 2,
 "mean_char_prefix": 43.93, "gguf_errors": 0, "prompts": 15}
```

Read it the way the estate reads folds. PB1-01 "Compute 47 * 23": **both sides answer
` 1081`** and then diverge in the continuation (the GGUF starts a step-by-step, the HF side
starts a `</think>` block). That is an exact content match with zero exact string match —
which is the whole reason token match is never a grade here. PB1-14 and one other match
exactly; everything else diverges after a mean of 44 characters. Verbatim pairs for all 15
are in `parity/READ-Q35-08B-Q4KM.txt`.

Notes on the two quant targets: `Q4_K_M` is the envelope-arithmetic default from plan §2
(15.4 GB at 27B). `IQ4_XS` is smaller but is normally produced **with an importance
matrix**; this script quantizes without one unless `llama-quantize` is given `--imatrix`
via a future flag, so the IQ4_XS row must be read as "IQ4_XS, no imatrix" wherever it
appears.

### 2.3 Second pass, 2026-09-02: the IQ4_XS artifact was finally served

Gap 9 of the previous revision said the IQ4_XS file "converted and hashed; only Q4_K_M was
loaded and benched". It has now been loaded, on GPU5 (Tesla P40), through our own
`llama-server` at `-c 8192 -fa on -ctk q4_0 -ctv q4_0`, healthy in 5-6 s, and run against
the same EXPORT-PB1 bank with `--skip-hf --reference parity/parity-Q35-08B-Q4KM.json` (the
HF answers were already measured; re-running the bf16 side would have cost a card and
changed nothing).

| artifact | compared | exact | normalized | mean char prefix | receipt sha256 |
|---|---|---|---|---|---|
| `Q35-08B-stock.Q4_K_M.gguf` | 15 | 2 | 2 | 43.93 | `b6d40ce1ec68b7b9…` |
| `Q35-08B-stock.IQ4_XS.gguf` | 15 | 1 | 2 | 27.40 | `9a55493c7a33860a…` (re-run) |

The IQ4_XS receipt was **re-measured on 2026-09-02** so the shipped artifact carries the
HF-column provenance added in §5.3 D4 rather than only future runs carrying it. The bytes
of the first run are kept as `parity-Q35-08B-IQ4XS.08f8106a.json`. The re-run reproduces it
**exactly — 0 of 15 GGUF answers differ** — under the same flags, so the numbers above are
unchanged and now come with `hf_side_source: reference`, an `hf_reference` block naming
`parity-Q35-08B-Q4KM.json` by sha256 `b6d40ce1…` with its `hf_dir`
(`/home/cereal/models/Qwen3.5-0.8B`), `bfloat16` and `AutoModelForCausalLM(dtype)`, and
`hf_source: "reference"` on all 15 rows.

Read as content, not string match: on PB1-01 (`Compute 47 * 23`) **both** quantizations
answer ` 1081` and then diverge in the continuation, exactly as the Q4_K_M row did against
bf16. IQ4_XS diverges from the bf16 reference sooner on average (27.4 vs 43.9 characters),
which is the expected direction for a smaller quant produced **without an importance
matrix** — one bank, one run, reported as a measurement.

#### 2.3.2 What the re-run measured on the way: KV quantization moves the answers

The re-run was first made at `-c 4096 -fa on` with **f16** KV, not the `-c 8192` +
`q4_0/q4_0` of §2.3, and it changed **10 of 15 answers** — PB1-01 still opens ` 1081` and
then continues completely differently, PB1-02 goes from ` 13:40 - 09:15 = 4:25` to
` 4 hours 25 minutes`, PB1-13's date extraction reorders. Under greedy decode
(`temperature 0`, `top_k 1`, fixed seed) on the same GGUF file. So:

| control | rows differing (of 15) |
|---|---|
| two passes against the SAME running server | **0** |
| two separate launches, same flags (q4_0 KV) | **0** |
| the earlier run 2 h before, same flags (q4_0 KV) | **0** |
| same file, same prompts, **f16 KV instead of q4_0** | **10** |

Serving is reproducible; the KV cache type is part of the measurement. The f16 pass is kept
as `parity-Q35-08B-IQ4XS.f16kv-3ab44297.json` (+ `-REPEAT`, `-CHAT`) and the canonical
receipts are the q4_0 ones. This is why `parity-<tag>.json` now records `server_props`
(`build_info`, `n_ctx`, `total_slots`, `chat_template`) — and why the KV types, which
`/props` does **not** expose, are a named gap: `parity_check.py` attaches to a server it
did not launch, so it cannot witness the flags the way `envelope_bench.py` does. On this
build `build_info` reads `b0-unknown`, so it does not identify the binary either.

#### 2.3.1 The chat-template path changed an answer

`parity_check.py --chat` (gap 10: implemented, never exercised) now has a run:
`parity/parity-Q35-08B-IQ4XS-CHAT.json` (sha256 `a181b7961036ba2a…` after the re-run;
the first run's bytes are kept as `parity-Q35-08B-IQ4XS-CHAT.42781720.json`),
`chat_template_applied: true`, 15 rows. **The `1381` reproduces**: the re-run under the
documented flags answers `1381` to PB1-01 again. The rendering is the real Qwen one —

```
<|im_start|>user
Compute 47 * 23. Give only the number.
Answer:<|im_end|>
<|im_start|>assistant
<think>

</think>

```

— and under it the same IQ4_XS file that answered ` 1081` from the raw prompt answers
**`1381`**. Same server, same weights, same greedy settings; only the prompt rendering
changed. That is a content miss, not a termination artifact, and it matters because the
review runner renders with `apply_chat_template(add_generation_prompt=True)`: an export
graded through one rendering is not automatically the same model through the other.
It is one prompt in one run and is reported as such, not as a verdict on either path.

Note on `compared: 0` in that receipt: with `--skip-hf` and no `--reference` there is no
second side to compare against, so the row-level comparison count is legitimately zero
while all 15 GGUF answers are recorded verbatim. A chat-rendered reference would have to be
produced by an HF run under the same rendering; that run was not made.

Commands (both were run; the second needs `--hf-dir` even with `--skip-hf`, because the
chat rendering uses that tokenizer):

```bash
$VENV parity_check.py --tag Q35-08B-IQ4XS --server-url http://127.0.0.1:18097 \
    --gguf-path gguf/Q35-08B-stock.IQ4_XS.gguf --skip-hf \
    --reference parity/parity-Q35-08B-Q4KM.json --out-dir parity
$VENV parity_check.py --tag Q35-08B-IQ4XS-CHAT --server-url http://127.0.0.1:18097 \
    --gguf-path gguf/Q35-08B-stock.IQ4_XS.gguf --skip-hf --chat \
    --hf-dir /home/cereal/models/Qwen3.5-0.8B --out-dir parity
```

---

---

## 3. (c) The 256k envelope protocol

One command per (model, card, KV pair). Two matched KV pairs, never mixed:

```bash
VENV=/home/cereal/model-surgery-lab/venv/bin/python
cd /home/cereal/omni-campaign/export

# q4_0/q4_0 KV on GPU4 (RTX 3090), our own container, full sweep
$VENV envelope_bench.py --tag OC-Q4KM-gpu4-q4kv \
  --model /home/cereal/models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
  --runner host --gpu-uuid GPU-2ce6a95c-7b1f-644a-3808-d781732a0582 --gpu-index 4 \
  --kv q4_0 --ctx 262144 --port 18093 \
  --prefill-points 4096,32768,131072,262144 \
  --needle-points 32768,65536,131072,262144 --needle-depths 0.1,0.5,0.9

# q8_0/q8_0 KV, same card (this is the pair that does NOT fit a 27B Q4_K_M at 256k per
# plan §2 arithmetic — running it is how we find out, not predicting it)
$VENV envelope_bench.py --tag OC-Q4KM-gpu4-q8kv ... --kv q8_0

# attach mode: bench a server someone else started; starts and stops NOTHING
$VENV envelope_bench.py --tag P40-Q36-27B-q4kv --attach http://127.0.0.1:18080 \
  --container-name p40_256k --gpu-index 1 --kv q4_0 --ctx 262144 \
  --prefill-points 4096,32768 --needle-points 32768 --needle-depths 0.1,0.9
```

What it records per run:

* `config` — ctx, `type_k`/`type_v` (matched pair), `-fa` request, `-np 1`, runner, GPU uuid.
* `log_analysis` — parsed from the **head** of the server log (the FA/KV/offload banner
  prints once at load and then scrolls away behind per-request progress lines):
  `flash_attn_state` (`enabled|disabled|auto|unknown`), `kv_type_k/v`, KV buffer MiB per
  CUDA device, `offloaded N/M layers`, `n_ctx`, and **every** fallback / "not supported" /
  CUDA-error line verbatim under `fallback_evidence`.
* `requested_flags_source` — **where the "requested" envelope came from**, one of
  `own_launch_command` / `docker_inspect` / `cli_defaults_unverified`. Only the first two
  are observations of the server that is answering. In the third case (`--attach` with no
  `--container-name`, a renamed container, docker unavailable) the flags are our own CLI
  defaults, and `flash_attn_evidence()` **discards them**: attaching to a peer's
  `-fa off -ctk f16` server can no longer produce a receipt asserting a q4_0
  flash-attention run. The basis list then starts with `R0 launch flags NOT OBSERVED`.
* `r3_control` + `flash_attn_evidence.r3_premise` — see §5.2. R3 is only usable when a
  negative control on this build has actually been run.
* `gpu_sampling` — `peak_by_index` is **whole-card** `nvidia-smi` MiB (it includes every
  other tenant of the card), and `peak_by_pid` / `own_peak_memory_used_mib` are the
  **measured per-process** peak from `--query-compute-apps`, with the server PID we started
  marked `tracked_as_ours`. The run's own share is measured, never subtracted.
* per context point: calibrated prompt length (via the server's own `/tokenize`), prefill
  wall clock and tok/s with `cache_prompt=false`, decode tok/s over the cached prefix with
  `ignore_eos` on (the calibration prompt contains the needle *and* its question, so
  without it the 4k point stops at EOS after 9 tokens and the 32k point after 63, and the
  two tok/s figures are not comparable; `--no-decode-ignore-eos` restores the old
  behaviour, and `decode_ignore_eos` is recorded per point).
* per needle: depth fraction, insert line, prompt tokens, hit/miss by exact code
  containment, and **the answer verbatim**.

Cost, so the sweep is scheduled rather than discovered: a 256k prefill on a P40 at the
measured ~150-250 tok/s is 17-29 minutes, and each needle depth is another full prefill
(`cache_prompt=false`). Four context points x three depths at 256k is hours. Use
`--max-wall-s` to bound a run — it writes `stopped_early`, `partial: true` and
`complete_sweep: false`, and keeps the `.partial.json` (see §5.3 D2) — and launch long
sweeps detached:

```bash
systemd-run --user --unit=oc-envelope-gpu4-q4kv \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
  $VENV /home/cereal/omni-campaign/export/envelope_bench.py --tag ... --max-wall-s 21600
```

(The estate's launch lesson: `launch_detached.py` children died silently at ~10 min inside
an SSH session scope, twice. `systemd-run --user` or `setsid nohup` only.)

### 3.1 Measured envelope run (2026-09-02, re-run): 0.8B Q4_K_M, GPU4 (RTX 3090), q4_0/q4_0 KV

Made with **our** binaries rather than the ghcr image, bound to `127.0.0.1`,
`-c 262144 -fa on -np 1 -ngl 999 -ctk q4_0 -ctv q4_0`. The server reported itself
**listening at 2.312 s on its own log clock** (`log_analysis.listening_at_log_s`); the
`server_wait.wait_s` figure of 3.05 s is when our health poll first saw it and is an upper
bound, not a load time. (The first revision of this spec quoted 20.05 s — that was the old
5-second poll granularity being read as if it were the model load time.)

| Context | Prefill wall | Prefill tok/s | Decode tok/s (64 tok, `ignore_eos`) | Needle |
|---|---|---|---|---|
| 4,096 | 0.28 s | 14,851 | 298.3 | — |
| 32,768 | 2.30 s | 15,606 | 259.9 | depth 0.10 **HIT**, depth 0.90 **HIT** |

* VRAM on GPU4, **measured two ways**: whole-card peak **2,501 / 24,576 MiB**, and this
  run's own peak **2,478 MiB**, measured per-PID
  (`nvidia-smi --query-compute-apps`, pid 3958070, our `llama-server`). Everything else on
  the card during the run accounted for ~23 MiB. The earlier revision subtracted a
  remembered 630 MiB of another lane's process and reported "~1.87 GB own share": that
  process was **not** resident during this window, the number was arithmetic rather than
  measurement, and it was ~600 MB low. Per-PID sampling now makes the subtraction
  unnecessary.
* FA reading: *"flash-attention path ran: quantized KV was accepted at load, which this
  build refuses without flash_attn (negative control 38e80ce8b6941c48 — the control that
  was current when this run was made; under the §5.3 D3 join it no longer licenses R3,
  because it records no `llama_server_sha256`, and a re-run of this sweep would cite
  19f2eba1 instead)"*, basis
  `R1 launch flag -fa=on (source=own_launch_command)` +
  `R3 KV types q4_0/q4_0 quantized and the server reached ready=True [premise verified]`.
  See §5.2 for how the premise stopped being an assumption.
* `kv_buffer_mib_total` is `None` — this build prints no KV-buffer line (see §5.1); VRAM
  comes from `nvidia-smi` instead.
* Needle answers carry the expected repetition artifact
  (`'MB3-9052 Answer: MB3-9052 Answer: …'` at depth 0.10; depth 0.90 answers
  `'MB3-9052 <think> Thinking Process: …'`). Under artifact-tolerant grading both are
  **retrieval hits with a termination artifact**, and the receipt keeps the strings
  verbatim so a human can see exactly that.
* Decode is now measured over the same 64 tokens at both context points, so 298.3 vs 259.9
  is a like-for-like comparison in the expected direction. The previous 147.7-vs-258.3 pair
  was 9 tokens against 63 and should not have been printed side by side.
* Teardown is receipted (`teardown {"stopped": true, "pid": 3958070}`) because the run did
  not pass `--keep-server`. The earlier run used `--keep-server`, so its shutdown happened
  outside any receipt; `--keep-server` now writes a warning into the receipt saying so.

Receipt: `envelope/envelope-Q35-08B-Q4KM-gpu4-q4kv-v2.json` sha256
`7b2b90a49df381165ff88ff64f7e9f80bf852669b2ee63e5a3cd94ca20d6fe6c`. The pre-fix run is
kept as `envelope-Q35-08B-Q4KM-gpu4-q4kv.json` (sha256 `810da40026fdd3bb…`) — its numbers
stand as measured, but read its VRAM and decode rows with the two caveats above.
The 27B x 256k rows on the P40 and the 3090 belong to the sibling runs already in
`needle_gpu0_3090_q4kv.jsonl` / `needle_gpu1_p40_q4kv.jsonl`; this lane's contribution is
the protocol + receipt format they can be re-run under.

### 3.2 The 262,144-token point could not have worked, and now can

The sibling lane's own needle log carries these rows against a `-c 262144` server:

```
{"label": "gpu1_p40_q4kv", "depth_target": 250000, "pos": 0.25, "error": "<HTTPError 400: 'Bad Request'>"}
{"label": "gpu0_3090_q4kv", "depth_target": 250000, "pos": 0.25, "error": "<HTTPError 400: 'Bad Request'>"}
```

`envelope_bench.py` had the same defect. `calibrate_line_count` aimed at `target_tokens`
with a symmetric 2% tolerance, so the 262,144-token point would build a prompt of up to
267,387 tokens and then ask the server to generate 64 more on top of it. llama-server
refuses that outright, urllib raises, and the exception took the whole sweep with it — the
marquee measurement of this lane would have come back as an exception every single time,
and nobody had run it at 27B yet to find out.

Fixed in two places, both pinned by tests:

* `llama_probe.context_point_headroom(n_ctx, decode_tokens)` computes the reserve
  (`max(decode_tokens, needle n_predict)` plus a 64-token safety margin) and the prompt
  ceiling. `calibrate_line_count(..., max_tokens=ceiling)` now treats overshoot as never
  inside tolerance, converges from below, and runs a final shrink pass, returning
  `capped` / `over_cap` / `requested_target_tokens` so the receipt shows the cap it hit.
  At `-c 262144` with 64 decode tokens the ceiling is **262,016**.
* `measure_context_point` records `context_headroom` and `capped_by_ctx`, and every request
  goes through `_request_or_error`, so a refusal is recorded as data (`error`,
  `error_body`) and the sweep continues to the next point instead of dying at this one.

`TestContextHeadroom` drives the full 262,144 point against a fake server that refuses
exactly like llama-server does (`prompt + n_predict > n_ctx` raises `HTTPError 400`) and
asserts zero refusals with the ceiling in place and at least one without it.

#### 3.2.1 The fix, proven against a real llama-server

Waiting hours for the 27B 262,144-token point is not the only way to test this. The same
shape — a requested context point equal to the served context — reproduces at any size, so
it was run at `-c 8192` on GPU2 with the 0.8B Q4_K_M, our binary, in under three minutes:

```
context point 8192 tokens (code MB3-9052) ...
    NOTE: 8192-token point capped to 8012 tokens by -c 8192 (reserve 128 for decode)
    needle depth=0.10 tokens=8012 hit=True
    needle depth=0.90 tokens=8012 hit=True
  prefill 1.9s (4530.5 tok/s), decode 112.2 tok/s
```

No `400`, both needles retrieved at the top of the context, and the cap is stated in the
receipt (`capped_by_ctx: true`, `context_headroom.reserve_tokens: 128`). Receipt
`envelope/envelope-HEADROOM-PROOF-Q35-08B-gpu2.json` sha256
`5c38e80abbc23ba5f7784bd8fbf9dbc588369b7cc20286f8945dbaef3bea3d1c`. The depth-0.90 answer
carries the familiar repetition artifact (`MB3-9052 Answer: MB3-9052 Answer: ...`) and is
scored a hit on content, as this estate scores it.

The same command shape at 27B is what the in-flight runs will produce at 262,016 tokens:

```bash
$VENV envelope_bench.py --tag HEADROOM-PROOF-Q35-08B-gpu2 \
  --model gguf/Q35-08B-stock.Q4_K_M.gguf --runner host \
  --gpu-uuid GPU-a92b0dfe-... --gpu-index 2 --kv q4_0 --ctx 8192 --port 18098 \
  --prefill-points 4096,8192 --needle-points 8192 --needle-depths 0.1,0.9
```

### 3.3 Measured, 2026-09-02: the 27B at 256k through the protocol

Every container the previous revision leaned on is gone (section 4), so these runs use our
own binaries on cards that were free, launched as `systemd-run --user` units.

| run | card | KV | result |
|---|---|---|---|
| `OC-Q36-27B-Q4KM-gpu3-q8kv` | Tesla P40 24 GB | q8_0/q8_0 | **did not load at `-c 262144`**; the weights fit and the compute buffers did not (verbatim `cudaMalloc failed` for 1,360.28 MiB). Receipt `d7499931fceb200c…`. Its own-peak figure of 23,942 MiB is **not usable**: the same run's whole-card peak on that card is 15,245 MiB, and per-process used cannot exceed whole-card used at one instant, so one of the two nvidia-smi series missed the ramp. `GpuSampler` now records `whole_card_peak_max_mib` and sets `own_peak_exceeds_whole_card_peak` with a note when the pair is impossible; it reports the inconsistency and repairs neither number. |
| `Q36-27B-Q4KM-q8kv-p40gpu3` (KV fit search) | Tesla P40 24 GB | q8_0/q8_0 | **loads at `-c 233472`, fails at `-c 237568`**; 8 launches, 2,041 s. Receipt `71c59ecac6732659…` |
| `OC-Q36-27B-Q4KM-gpu1-q4kv` | Tesla P40 24 GB | q4_0/q4_0 | in flight at the time of writing; 3 of 4 context points measured |
| `OC-Q36-27B-Q4KM-gpu0-q4kv` and `-v2` | RTX 3090 | q4_0/q4_0 | **lost the card to another lane twice** (3.3.2) |
| `OC-Q36-27B-Q4KM-gpu0-q4kv-v3` | RTX 3090 | q4_0/q4_0 | **COMPLETE: all 5 context points to 260,712 tokens, 12/12 needles hit, own peak 21,522 MiB** (3.3.3). Receipt `6626c991fbaa148e…` |

The q8_0 failure is worth quoting verbatim, because it is not a KV-size arithmetic problem:

```
1.06.706.993 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1360.28 MiB on device 0: cudaMalloc failed: out of memory
1.06.707.006 E graph_reserve: failed to allocate compute buffers
1.06.719.902 E llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
```

The weights and the q8_0 KV both allocated; the graph compute buffers were what would not.
Plan section 2 predicted the q8_0 pair would not fit a 24 GB card at 256k, and it does not —
but the ceiling it *does* reach was unknown until `kv_fit_search.py` bisected it:
**233,472 tokens with q8_0/q8_0 KV on 24 GB**, 89% of the 262,144 target, bracketed to
4,096 tokens by eight real launches (a failing attempt costs about 7 s; the boundary pair
was `-c 237568` failing and `-c 233472` loading healthy in 9.2 s).

One anomaly in that receipt, left visible rather than smoothed: the first attempt
(`-c 262144`) reports `wait_s` 1,709.62 against a requested `--load-timeout` of 420 while
another lane held the host CPUs with a 128k prefill. The poll loop is wall-clock bounded,
so the overrun is real and unexplained. `load_timeout_s` and `wait_exceeded_load_timeout`
are now recorded per attempt so the next occurrence is visible in the receipt instead of
being reconstructed from file timestamps. Per-attempt `own_peak_memory_used_mib` is also
recorded now: the run-level `own_peak_memory_used_mib` of 107,384 MiB is the SUM over eight
sequential servers and is nobody's footprint.

#### 3.3.3 The full 256k envelope at 27B, completed (RTX 3090, q4_0/q4_0 KV)

`envelope/envelope-OC-Q36-27B-Q4KM-gpu0-q4kv-v3.json`, sha256
`6626c991fbaa148e24b918c92202efad8e53704efd39677f330d3cb26acbe947`, `partial: false`,
3,005.6 s wall. (This receipt predates the completeness fields added in §5.3; it has no
`complete_sweep` key. Its five points, its non-null decode leg at every point and its
receipted teardown are what make it complete. Runs from here on carry `complete_sweep`.) Qwen3.6-27B Q4_K_M (16.5 GB), our own `llama-server`,
`-c 262144 -fa on -np 1 -ngl 999 -ctk q4_0 -ctv q4_0`, bound to loopback, GPU0 by UUID.
This is the run gap 7 said had never been made.

| context point | calibrated prompt | prefill wall | prefill tok/s | decode tok/s (64 tok) | needles (0.1 / 0.5 / 0.9) |
|---|---|---|---|---|---|
| 4,096 | 4,093 | 3.2 s | 1,279.1 | 38.95 | — |
| 32,768 | 32,764 | 28.8 s | 1,161.2 | 32.76 | HIT HIT HIT |
| 65,536 | 65,524 | 69.4 s | 948.5 | 22.26 | HIT HIT HIT |
| 131,072 | 131,073 | 166.9 s | 787.5 | 20.58 | HIT HIT HIT |
| **262,144 (capped to 260,712)** | 260,712 | **466.8 s** | **559.6** | **12.42** | **HIT HIT HIT** |

* **12 of 12 needles retrieved**, including three at 260,712 tokens, answers verbatim in the
  receipt (`<think> </think> VD5-3094` and so on — terminated cleanly, no cap-ride).
* **VRAM**: whole-card peak 24,121 / 24,576 MiB (three of another lane's ComfyUI processes
  were resident during the run, 1.3–2.6 GB each); **this run's own measured per-PID peak was
  21,522 MiB**. A 27B Q4_K_M at 262,144 context with q4_0/q4_0 KV therefore fits a 24 GB
  card with roughly 3 GB of headroom — measured, not computed.
* **FA reading**: *flash-attention path ran: quantized KV was accepted at load, which this
  build refuses without flash_attn*, basis `R1 launch flag -fa=on (source=own_launch_command)`
  plus `R3 ... [premise verified]` against negative control `19f2eba1ee87fa29`. No fallback,
  "not supported" or CUDA-error line anywhere in the log.
* **Teardown receipted** (`{"pid": 1485403, "stopped": true}`); the card was left free.
* The 262,144 row exists **because** of the headroom fix (§3.2): the point was capped to
  260,712 tokens and served. Without the cap this is the row that comes back as
  `HTTPError 400`, which is precisely what happened to the sibling lane at 250k.

Decode falls 38.95 → 12.42 tok/s and prefill 1,279 → 560 tok/s from 4k to 260k, i.e. the
256k envelope costs about 3.1x in decode and 2.3x in prefill against a short context on this
card. Those are reported measurements for Cyril's envelope decision, not thresholds.

#### 3.3.1 The 27B q4_0 sweep on a P40, in flight

`envelope/envelope-OC-Q36-27B-Q4KM-gpu1-q4kv.partial.json`, our binaries,
`-c 262144 -fa on -np 1 -ngl 999 -ctk q4_0 -ctv q4_0`, 21,405 MiB resident on GPU1:

| context point | calibrated prompt | prefill wall | prefill tok/s | decode tok/s (64 tok) | needles |
|---|---|---|---|---|---|
| 4,096 | 4,093 | 16.7 s | 245.5 | 11.84 | — |
| 32,768 | 32,764 | 176.4 s | 188.0 | 8.88 | 2 (depths 0.1, 0.9) |
| 131,072 | 131,072 | 2,961.7 s | 44.3 | 4.25 | — |
| 262,144 | capped to <= 262,016 by 3.2 | in flight | | | |

The 131k prefill rate (44.3 tok/s) is roughly half the sibling lane's 98.1 tok/s at 139k on
the same card. That difference is host contention, not the card: the KV fit search and two
other lanes were running on the box in that window. A rate measured while the box is shared
is the rate that run got, and the receipt records the wall clock either way.

#### 3.3.2 Losing a shared card, twice, and what was done about it

The 3090 run died at load both times, and neither death was a defect in the bench:

```
0.50.569.568 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 15088.32 MiB on device 0: cudaMalloc failed: out of memory
```

GPU0 was holding another lane's `oc_run_frozen_bank.py --model-dir /home/cereal/models/Qwen3.5-4B`
(9,218 MiB). That process is not ours to stop, so `wait_free_vram.sh` waits for the card
instead. It waited 2,194 s, saw 24,099 MiB free, launched — and a peer job took the card
back within 4 s, so the second attempt failed too. The third launch
(`oc-env-27b-gpu0-q4kv-v3`) wraps the wait in a retry loop of up to 12 attempts at a
21,000 MiB bar, because on a shared card a single wait-then-launch loses a race it cannot
see coming. Every failed launch still writes a receipt naming, per PID, the tenant it
collided with.

```bash
# what is running now (both detached, both write partial receipts as they go)
systemctl --user list-units "oc-*" --all
tail -f /home/cereal/omni-campaign/export/logs/oc-env-27b-gpu1-q4kv.log
tail -f /home/cereal/omni-campaign/export/logs/oc-env-27b-gpu0-q4kv-v3.log
```


**Binding.** `envelope_bench.py` binds `127.0.0.1` by default (`--bind` overrides).
llama-server has no auth and says so in its own log ("CORS is set to allow all origins
('*') and no API key is set / this can be a security risk"), and this box also fronts
Keycloak, mailcow and the SOC dashboard, so a multi-hour detached sweep must not listen on
every interface. The docker runner publishes as `-p 127.0.0.1:<port>:<port>` and keeps
`--host 0.0.0.0` *inside* the container netns, because a container-internal loopback bind
is unreachable through a published port.

**Container safety.** `envelope_bench.py` stops only a container it started itself, whose
name it generated (`oc-env-<tag>-<port>`), and it refuses to start one whose name already
exists. `--attach` mode issues no `docker run`/`rm` at all. `p40probe_check.py` has no
container lifecycle call anywhere in it — enforced by two tests: a literal grep, and an AST
walk that also catches `docker restart`, `docker update` and a verb or name built from a
variable. The AST detector is itself checked against `envelope_bench.py`, where it must (and
does) find the `docker rm -f`.

---

## 4. (d) The running P40 envelope container

The plan text calls it `p40probe`; **the container actually running is `p40_256k`**
(`ghcr.io/ggml-org/llama.cpp:server-cuda`, port 18080, GPU1 = Tesla P40). `p40probe_check.py`
tries both names and takes `--container` as an override.

```bash
VENV=/home/cereal/model-surgery-lab/venv/bin/python
$VENV /home/cereal/omni-campaign/export/p40probe_check.py --tag P40-256K --port 18080
$VENV /home/cereal/omni-campaign/export/p40probe_check.py --tag GPU0-3090 \
      --container gpu0_256k --port 18081
# queue behind an in-flight long-context request instead of skipping:
$VENV .../p40probe_check.py --queue-anyway --request-timeout 3600
```

### 4.1 Measured (2026-09-02): `p40_256k`, Tesla P40, Qwen3.6-27B Q4_K_M at 256k

State: running, `/health` ok, `n_ctx` 262144 (log **and** props), `kv_unified = 'false'`,
21,409 / 24,576 MiB on GPU1. Launch command read off the container:
`-fa on -ctk q4_0 -ctv q4_0 -c 262144 -np 1 -ngl 999`. FA reading: *flash-attention path
ran* (basis R1 + R3, no refusal line anywhere in the log).

Short-context sanity, greedy, `cache_prompt=false`, verbatim:

| id | prompt | answer | stop | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|
| SC-1 | `17 + 26`, number only | `' 43\n\nWhat is the name of the largest island in the world?\nAnswer: Greenland\n\nWhat is the'` | limit | 36.4 | 14.14 |
| SC-2 | which drawer holds the spare key | `'\n\n<think>\n\n</think>\n\ndrawer 7'` | **eos** | 87.1 | 15.23 |
| SC-3 | largest planet, one word | `' Jupiter\n\nWhat is the name of the planet that is the 3rd planet from the sun?\nAnswer: 1\n\nWhat is the name of the'` | limit | 45.1 | 13.97 |

All three answers are **content-correct**; two ride the token cap and continue with
self-generated questions (a termination artifact, which never counts against content in
this estate). Decode 14.0-15.2 tok/s matches the plan's recorded 14.1 tok/s for this card
and model. Receipt: `probe/probe-P40-256K-SANITY.json` sha256 `e7a501f3f74ce337…`.
An earlier run of the same script, while the sibling lane's 140k needle request held the
slot, correctly reported `slot_busy=True` and **skipped** rather than queueing
(`probe/probe-P40-256K.json`).

It reports: container state / image / cmd / GPU device request, load-banner analysis,
`nvidia-smi` per index, `/health`, `/props` (`n_ctx` is nested in this build — the helper
unwraps it), `/slots` busy state, and — only when the server has finished loading and no
slot is mid-request — three short-context sanity completions with timings and verbatim
answers. Default behaviour when a slot is busy is to SKIP and say so, because the sibling
lane's needle run owns that server.

`read_only: true` in these receipts means **lifecycle** read-only, and the receipt now says
so in `read_only_scope` / `note`: nothing is stopped, killed, restarted or removed, but
unless `--skip-completion` is passed the check does occupy a slot and GPU time on a server
another lane owns.

### 4.2 Re-run after the evidence-chain fix

`probe/probe-P40-256K-V2.json` (sha256
`7622bfcf8176f2d177339ce6fa67d0dbbb3afc202b605c1525b6c7828aef5bc4`) is the same container
re-inspected with `--skip-completion`, so it took nothing from the peer lane. It carries the
new fields: `requested_flags_source: docker_inspect`, the licensed R3 premise, and
`r3_control_scope_note` recording that the control was measured on *our* build, not on the
`ghcr.io` image this container runs. The two earlier probe receipts (`probe-P40-256K.json`,
`probe-P40-256K-SANITY.json`) stand as measured.

### 4.3 2026-09-02: the container is gone, and the receipt now says so

The task for (d) is "check the running p40probe container (`docker logs p40probe`) and, if
it has finished loading, run the short-context sanity through its port 18080". Measured
this session:

```
$ docker logs p40probe
Error response from daemon: No such container: p40probe
$ docker logs p40_256k
Error response from daemon: No such container: p40_256k
$ docker inspect p40probe --format "{{.State.Status}}"
error: no such object: p40probe
$ docker ps -a --format "{{.Names}}" | grep -Ei "p40|256k|probe|llama"
(no output)
```

Both `p40_256k` and `gpu0_256k` were **removed** — not stopped — some time between the
sibling lane's last needle row (`needle_gpu1_p40_q4kv.jsonl`, 22:03 on 2026-09-01) and this
session. Every GPU on the box was idle at the start of this pass, and nothing this lane did
removed them.

`p40probe_check.py` used to report that state as `container_running: false` with
`inspect: {"error": "container not running"}`, which reads like a stopped container with a
log sitting there to inspect. It now resolves three states — running, exited-but-present,
removed — via `docker ps` plus `docker ps -a`, records `container_exists` and
`container_status_line`, reads the log whenever the container **exists** (a stopped
container still carries its image, launch command, device request and exit state, all of
which are envelope evidence), and when the object is absent writes
`container_absent_evidence` with the daemon's verbatim refusal plus:

> the container named 'p40_256k' is not present on this host in any state (running or
> exited), so there is no log, no launch command and no server to send a sanity completion
> to. Nothing was started to replace it: that decision belongs to whoever owns the card.

Receipt: `probe/probe-P40-256K-ABSENT-20260902.json` sha256
`df3a1597c9baaafe82a086cdf9786ba67a55798fe54fc92e4c108692bd759d2e`
(`completions_skipped` names the absence, not a load failure).

`TestProbeReadOnly.test_resolve_container_separates_gone_from_stopped` pins all three
states, including that an explicitly named container is never silently swapped for a
candidate that happens to exist.

**What replaced the measurement.** The short-context sanity that container existed to
provide (Qwen3.6-27B Q4_K_M on a Tesla P40) is covered by the 4,096-token point of
`OC-Q36-27B-Q4KM-gpu1-q4kv` (section 3.3.1): same model, same card class, our own binary,
prefill 245.5 tok/s and decode 11.8 tok/s, with the answers recorded verbatim in the
receipt. The earlier `probe-P40-256K-SANITY.json` rows (14.0–15.2 tok/s decode) stand as
measured against the container that is now gone.

---

## 5. Receipts

Every script writes canonical JSON (sorted keys, compact separators, atomic replace) and
prints the receipt's own sha256.

| Receipt | Written by | Joins to |
|---|---|---|
| `llamacpp/build-receipt.json` | build script | every export receipt (`build_receipt_sha256`, `llama_cpp_commit`) |
| `gguf/export-receipt-<tag>.json` | `export_hf_to_gguf.py` | the child's fold/train receipt via `source_directory_digest.directory_sha256` |
| `parity/parity-<tag>.json` + `READ-<tag>.txt` | `parity_check.py` | the export receipt via `gguf_sha256` |
| `envelope/envelope-<tag>.json` | `envelope_bench.py` | the export receipt via `model_sha256` |
| `probe/probe-<tag>.json` | `p40probe_check.py` | — |
| `fa-control/fa-negative-control.json` | `fa_negative_control.py` | every envelope/probe receipt via `r3_control.receipt_sha256` |

**Hashes in prose are cited by receipt field, and re-verified with `sha256sum` on the
server before publication.** An earlier revision of this document published a
build-receipt sha256 (`ce529235…`) that matched no artifact on either machine, while the
export receipt's own `build_receipt_sha256` had the correct one. A hash nobody can
reproduce is worse than no hash, so: quote the field, then check it.

The checkpoint directory digest is the same construction the R3 ledger protocol uses:
sha256 over the sorted `(relative path, file sha256)` lines, so an export row can be joined
to the fold/distill/finetune lineage by digest rather than by filename.

---

## 5.1 The log-format finding that changed the FA assertion

The task asked to "assert the FA kernel ran (parse logs for CPU fallback)". Measured on
this box, **the classic banner does not exist any more**. `p40_256k`'s entire log is 185
lines and the load section is:

```
0.00.402.341 I srv    load_model: loading model '/models/Qwen3.6-27B-Q4_K_M.gguf'
0.55.659.681 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 262144, kv_unified = 'false'
0.55.687.709 I srv  llama_server: model loaded
0.55.687.716 I srv  llama_server: listening on http://0.0.0.0:18080
```

No `flash_attn =`, no `type_k =`, no `KV buffer size`, no `offloaded N/M layers`. **Our own
build behaves identically** — the 174-line `envelope/server-Q35-08B-Q4KM-gpu4-q4kv.log` has
exactly one informative load line, the same `n_ctx_slot / kv_unified` one. So this is the
2026 llama.cpp logging format, not a container artifact. So the
parser keeps the old patterns (older builds and the host-runner logs still print some of
them) and the evidence chain was rebuilt on four readings, R1-R4, documented in
`flash_attn_evidence()`: the launch flags, **with provenance** (R1), a banner line if the
build prints one (R2), quantized-KV acceptance (R3), and any refusal/fallback/CUDA-error
line (R4). `TestCompressed2026LogFormat` pins all of this against the real log text.

`n_ctx_slot` is not `n_ctx`. With `-np N` the 2026 server prints only the per-slot value,
so the parser now records `n_ctx_slot`, `n_slots`, `n_ctx_from_slots` and an `n_ctx_source`
saying which of them `n_ctx` came from. At `-np 1` this changes nothing; on any attached
server with parallel slots the old parser would have reported ctx/N as the context.

## 5.2 R3 stopped being an assumption: the negative control

R3 — "llama.cpp refuses quantized KV without flash_attn, so a server that loaded with q4_0
KV ran the FA path" — was carried over from older llama.cpp behaviour and had never been
exercised against commit `b96806d9`. That is the same build that turned out to print no FA
banner at all, so "it refuses" was exactly the kind of assumption this build had already
broken once. Worse, the refusal string the parser hunted for (`cache quantization
requires`) existed only in a synthetic test log written by hand.

`fa_negative_control.py` went and looked, on a free P40 (GPU2), with our own binary and the
0.8B Q4 GGUF at `-c 4096`:

| trial | flags | result |
|---|---|---|
| `neg-fa-off` | `-fa off -ctk q4_0 -ctv q4_0` | **refused**: process exit code 1 after 3.04 s, never healthy |
| `pos-fa-on` | `-fa on -ctk q4_0 -ctv q4_0` | loaded, healthy in 3.01 s |

Verbatim, from `fa-control/fa-control-build-neg-fa-off.log`:

```
0.01.024.676 E llama_init_from_model: quantized V cache requires flash_attn to be enabled
0.01.851.021 E cmn  common_init_: failed to create context with model '...Q35-08B-stock.Q4_K_M.gguf'
0.01.856.608 E srv  llama_server: exiting due to model loading error
```

**The premise holds — and the parser was wrong.** The real message is
`quantized V cache requires flash_attn to be enabled`; the pattern the lane was matching
(`cache quantization requires`) would **not** have fired on it, so a genuine refusal would
have been recorded as "no refusal seen". Both spellings are now in `_FALLBACK_PATTERNS`,
and `TestRealRefusalText` pins the parser against this verbatim line instead of a
hand-written one.

Run it (about 8 seconds, needs one free card and nothing else):

```bash
VENV=/home/cereal/model-surgery-lab/venv/bin/python
$VENV /home/cereal/omni-campaign/export/fa_negative_control.py \
  --model /home/cereal/omni-campaign/export/gguf/Q35-08B-stock.Q4_K_M.gguf \
  --gpu-uuid GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4 --gpu-index 2 \
  --kv q4_0 --ctx 4096 --port 18097 --trial-timeout 90 \
  --out-dir /home/cereal/omni-campaign/export/fa-control
```

Mechanically: `load_r3_control()` reads
`fa-control/fa-negative-control.json` (sha256
`19f2eba1ee87fa2925ac3e233537a6062f691b43bb012a2325ce3a4dbca89557` — the bytes the 27B
envelope receipt cites, now shipped under both that name and the immutable
`fa-negative-control.19f2eba1.json`; the earlier `38e80ce8...` control, which recorded no
`llama_server_sha256` and therefore licenses nothing under the join in §5.3, is kept for
provenance as `fa-negative-control.38e80ce8.json`) and
`flash_attn_evidence()` reports `r3_premise` as `verified` / `void` /
`unverified_on_this_build`. With no control receipt the reading is *"quantized KV was
accepted at load, but the R3 premise is UNVERIFIED on this build"* — never "flash attention
ran". If a future build loads anyway with `-fa off`, the control records
`premise_holds: false` and every FA reading in the lane downgrades to R1 automatically,
with `"R3 IS VOID on this build"` in the text. The control is per-build: the probe receipt
for `p40_256k` (a `ghcr.io` image, not our binary) carries an explicit
`r3_control_scope_note` saying the premise is *inherited* there, not measured.

## 5.3 Four defects found by adversarial review, and what they cost

All four were in shipped code, all four are fixed in place, and each one has a test that
fails against the old behaviour.

**D1 — a refused context point destroyed the receipt it was supposed to protect.**
`measure_context_point()` returns early when the prefill request is refused, so one bad
point cannot cost the points after it. The early return set `error`, `needle` and the
prefill keys — and nothing about decode. `finish()`'s summary loop then indexed
`point["decode_tokens_per_s"]` unconditionally. Sequence: the complete receipt is written
at the top of `finish()`, the `KeyError` fires at the bottom, `main()`'s `except` catches
it and calls `finish()` a **second** time with `partial=True` and
`error="KeyError('decode_tokens_per_s')"` — a correct sweep overwritten by a receipt
blaming a Python bug, the `.partial.json` left undeleted, and a traceback on exit. Fix:
the early return writes `decode: null`, `decode_tokens_per_s: null` and
`decode_tokens_requested`, and the summary prints refused points as refusals instead of
indexing keys they never had. `TestRefusedPointSurvivesTheReceipt` drives `finish()` with
a refused point and asserts the receipt still says `partial: false` and still carries the
real HTTP error.

**D2 — a budget-truncated sweep claimed to be complete.** The `--max-wall-s` deadline
check sets `stopped_early` and breaks the point loop; control then fell through to
`finish(..., partial=False)`, which **also deletes** `envelope-<tag>.partial.json`. A
3-of-4-point run killed by its budget was therefore byte-indistinguishable, by the field
§3.3.3 cites as the completeness marker, from the finished 3090 run — and the intermediate
receipt that would have shown the truncation was removed. Fix: `sweep_is_partial(receipt)`
is the single decision (`stopped_early` or `error` ⇒ partial), the partial file survives,
and the receipt now carries three explicit completeness fields:

| field | meaning |
|---|---|
| `partial` | the process did not run the sweep to the end (crash, interrupt, or budget) |
| `complete_sweep` | every requested context point was measured and nothing was cut short |
| `points_requested` / `points_measured` / `points_refused` | the counts behind that verdict |

`partial: false` alone is no longer the completeness marker: **read `complete_sweep`.**
A sweep with a refused point is `complete_sweep: true` with a non-empty `points_refused`
— it ran to the end and the refusal is the measurement.

**D3 — a control with no binary hash licensed every binary at its commit.**
`_join_control_to_build()` counted only the fields present on *both* sides, so a control
carrying `llama_cpp_commit` and no `llama_server_sha256` matched on one field and returned
`matches: true`. The control this lane shipped first (`38e80ce8...`) is exactly that shape.
A commit is not an identity: the same commit built twice — different `--arch`, different
CUDA flags, or the separate `llama-imatrix` build that gap 2 defers *because it changes the
binary* — produces different bytes and inherits a licence it never earned. Fix: the join
requires `llama_server_sha256` on both sides (`REQUIRED_JOIN_FIELD`); a field the build has
and the control lacks is now recorded in `fields_missing_on_control` and drops `matches` to
`null`, which `flash_attn_evidence()` already reads as *"could NOT be joined"* and refuses
to promote. The join also reports `fields_compared` and `required_field_matched`, so a
reader can see what was actually checked. The 27B marquee receipt is unaffected: its
control matched on **both** fields and still does.

**D4 — a borrowed HF column read exactly like a measured one.** `--reference` copies the
reference receipt's `hf` answers into the new receipt. It recorded nothing about where
they came from, and because `--skip-hf` was used, `hf_dir` / `hf_dtype` / `hf_loader` were
`null` beside a fully populated `hf` column and a `compared: 15` summary. Nobody could tell
which file supplied the reference, which checkpoint or dtype produced it, or that it was
borrowed at all — and `write_receipt` replaces receipts in place, so re-running the
reference would leave the borrowing receipt describing bytes that no longer exist. That is
the duplicated-contract decay the R3 build-join was built to prevent, left open on the one
receipt that depends on it. Fix: the receipt now carries

* `hf_side_source`: `measured` | `reference` | `absent`,
* `hf_reference`: the reference `path`, its `sha256`, `tag`, `created_at`, `bank_id` /
  `bank_sha256`, `hf_dir`, `hf_dtype`, `hf_loader`, `chat_template_applied`, `gguf_path` /
  `gguf_sha256`, `rows_available` / `rows_borrowed`, plus `bank_matches_this_run` and
  `chat_rendering_matches_this_run` (both warn on stderr when false),
* per row, `hf_source: "reference"` and `hf_reference_sha256`,
* and a `HF SIDE: BORROWED from <path> (sha256=...)` banner at the top of
  `READ-<tag>.txt`, with each HF block headed `--- HF (transformers, reference) ---`.

A reference that itself borrowed its column (no `hf_dir` of its own) gets an explicit
`warning` in `hf_reference`.

Two further findings from the same review, not blockers but fixed here: `GpuSampler` now
flags an own-peak that exceeds the whole-card peak (§3.3, receipt `d7499931…`), and the
KV-fit test's fake sampler keys `peak_by_pid` with `str(pid)` like the real one, so the
`.get(str(pid))` fallback every real run depends on is actually exercised.

**Test count: 129 -> 145**, stdlib only, green on the workstation and on the server in the
lab venv.

## 6. Honest gaps

1. **"The FA kernel ran" is still not directly observable.** llama.cpp emits no such line,
   and this build emits no banner either. What the lane now has is R1 with provenance, R3
   with a measured premise (§5.2), and R4. If a stronger assertion is wanted the next step
   is a `-v` / `GGML_SCHED_DEBUG=2` run whose graph placement is parsed — designed, not
   built.
1b. **`kv_buffer_mib_total` is unavailable on this build**, so the KV footprint cannot be
   read off the log. VRAM is `nvidia-smi`: whole-card peak *and* per-PID peak, both
   recorded. The per-PID number is what a footprint conclusion should use.
2. **IQ4_XS without an imatrix.** No `--imatrix` plumbing yet; any IQ4_XS row must be read
   with that caveat — including the §2.3 parity rows. `llama-imatrix` is not among the
   binaries this lane built (`llama-server/cli/quantize/gguf/bench` only) and the build
   tree's `build/` directory is gone, so adding it means a fresh full build. That build was
   deliberately NOT run this session: it would overwrite `llamacpp/bin` and change the
   `llama-server` sha256 that the R3 negative control is joined to, while two envelope
   sweeps were mid-flight against exactly that binary. Do it into a separate root.
3. **Parity's HF side needs the model to fit.** At 27B it does not fit one card;
   `--hf-device-map sequential` is passed through but is unexercised. Parity is proven at
   the 0.8B/4B rungs first.
4. **The bench does not know about media output.** Plan §8's "trunk GGUF + companion media
   GGUF + driver" (D5b) has no runtime anywhere; this lane covers the text trunk only.
5. **Needle prompts are synthetic filler**, not curated text, and are not screened against
   the contamination banks (they contain no bank content by construction, but that is an
   argument, not a receipt).
6. `--gpu-index` only drives the nvidia-smi peak sampling; GPU selection is by
   `--gpu-uuid` because `CUDA_VISIBLE_DEVICES=6` selects a P40 on this host (index order
   differs from PCI order without `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
7. ~~The 27B x 256k sweep has not been run through `envelope_bench.py`.~~ **CLOSED on the
   3090** (§3.3.3: complete receipt, 12/12 needles, 260,712-token point served). Still open:
   the same sweep on a **P40**, which was in flight when this was written with 3 of 4
   context points measured (§3.3.1) — its 262k prefill runs at roughly 44–66 tok/s, so the
   256k rows on that card take hours. Read
   `envelope-OC-Q36-27B-Q4KM-gpu1-q4kv.partial.json` for where it got to.
8. **`--runner docker` is untested.** The measured path is `--runner host` with the built
   binaries. The docker path's `--gpus device=<uuid>` + volume mount are written but never
   exercised, because starting more containers on a box whose GPUs other lanes hold is the
   more intrusive choice.
9. ~~The IQ4_XS artifact has never been served.~~ **CLOSED 2026-09-02** — served on GPU5
   and run through EXPORT-PB1 against the recorded bf16 answers (§2.3).
10. `parity_check.py --chat` **has now been exercised on the GGUF side** (§2.3.1) and it
   found something: the chat rendering changed PB1-01 from ` 1081` to `1381` on IQ4_XS.
   What is still missing is a chat-rendered HF reference to compare against, so the chat
   run has answers but no second side (`compared: 0`).
18. **A parity receipt still cannot witness the serving configuration.** `parity_check.py`
   attaches to a server it did not launch, and `/props` exposes neither the KV cache types
   nor the `-fa` flag — and on this build `build_info` reads `b0-unknown`, so it does not
   even identify the binary. §2.3.2 measured what that costs: the same GGUF under f16 KV
   instead of q4_0 changes 10 of 15 greedy answers. Everything else reproduced exactly
   (0/15 across passes, launches and a two-hour gap), so the receipt is reproducible
   *given* the configuration and silent *about* it. The fix is to let `parity_check.py`
   own the launch the way `envelope_bench.py` does, or to pass and hash the launch command;
   neither was done here, and the flags are recorded in this document instead.
19. **The 27B P40 sweep still runs the pre-fix `envelope_bench.py`.** `oc-env-27b-gpu1-q4kv`
   was launched before §5.3 D1/D2 landed and its Python has the old module in memory, so
   if its `--max-wall-s 25200` budget cuts it short it will still write `partial: false`
   with no `complete_sweep` key. Count its points against `points` in the receipt, not the
   flag. It was deliberately left alone rather than restarted.

11a. ~~The two IQ4_XS parity receipts borrow their HF column with no provenance.~~
   **CLOSED 2026-09-02** — `parity_check.py` now records `hf_side_source`, the full
   `hf_reference` block (path, sha256, tag, hf_dir, dtype, loader, bank match, chat-render
   match) and a per-row `hf_source`, and both IQ4_XS receipts were **re-run** through it so
   the shipped artifacts carry it rather than only future ones (§5.3 D4).

11. **The shipped parity receipt predates two changes.** It was measured before
   `served_model_matches_gguf_path` existed (the served model and `--gguf-path` did match in
   that run, checked by hand) and before the needle/decode changes, which do not touch
   parity at all. It was not re-run, because re-running it costs a card and adds nothing to
   the fixes under review.
12. **The R3 control is per-build and was run on ours.** `p40_256k` / `gpu0_256k` run
   `ghcr.io/ggml-org/llama.cpp:server-cuda`, a binary this lane did not build. Their receipts
   carry `r3_control_scope_note` saying the premise is inherited there. Making it local is
   one `fa_negative_control.py --llamacpp-root` run against that image — not done, because it
   would mean starting another container on a card another lane holds.
13. **`fa_negative_control.py` was run at `-c 4096` on a P40 with a 0.8B model.** It answers
   a load-time acceptance question, which is architecture- and size-independent as far as
   this build is concerned, but it was not repeated per card or per model.

14. ~~The headroom fix is proven at 8k, not yet at 262k.~~ **CLOSED** — proven at both:
   8,012 tokens on the 0.8B (§3.2.1) and 260,712 tokens on the 27B at `-c 262144`
   (§3.3.3), three needle hits at the capped point, no refusal in either.
15. **One unexplained timing.** The KV-fit search's `-c 262144` attempt waited 1,709 s
   against a 420 s `--load-timeout` on a wall-clock-bounded poll loop (§3.3). The receipt
   now records the budget and the overrun flag; the cause is not known.
17. **A detached run shows nothing in its log until it flushes.** `oc-env-27b-gpu1-q4kv`
   was launched without `PYTHONUNBUFFERED=1`, so its driver log stayed empty for hours
   while the run was healthy; the observability channel for it is
   `envelope-<tag>.partial.json`, which is rewritten after every context point. Later units
   set `--setenv=PYTHONUNBUFFERED=1`, and any future launch should.
16. **`wait_free_vram.sh` cannot win a race it cannot see.** It samples free VRAM, and a
   peer can take the card between the sample and the allocation — measured, 4 s. The retry
   loop makes that survivable, not impossible.
