# S3 C2K4352 tracked continuation — 2026-09-07

This is a fresh folding-only child, not training, a parent baseline rerun, or
an accepted all-modality result. Root owns allocation, image build, immutable
source staging, and the separate frozen review. Preserve existing S3/K3/Claude
outputs. Do not overwrite a child or reuse a failed output directory.

## Runtime

The base tag `local/omni-k3-fold:sep07-e2e615be` was observed resolving to
`sha256:d4ef16f161e51df2bd51df62c1aaa4ea7d9fc65fefcbaef8382027a527ec3ba6`.
Immediately verify that mapping before building `Dockerfile.s3-fold` with
`--pull=false`; record the resulting image ID and use that ID in the allocator.
The local image ID is not a registry manifest digest. The Dockerfile installs
the pinned Transformers 5.12.1 **default dependency closure, no extras**, runs
`pip check`, and checks the existing Torch/CUDA, numpy, safetensors pins.

No FLA, causal-conv1d, kernels, accelerate, Pillow, torchvision, audio, video,
or serving packages are required for this fold. Strictly hiding those optional
packages on the Linux host still allowed both Qwen3.5 stock model classes,
meta construction of the actual VL parent, and its offline Qwen2Tokenizer.
The image itself must still pass the included build-time import test.
The stock GDN torch fallback is intentional and must not change mid-run.

Set `OMP_NUM_THREADS=22`, `MKL_NUM_THREADS=22`, `OPENBLAS_NUM_THREADS=22`,
`TOKENIZERS_PARALLELISM=false`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and `PYTHONDONTWRITEBYTECODE=1` from the start.
The S3 parser has **no `--threads` option**; the environment configures Torch.
The runtime preflight refuses a different Torch thread count. Twenty-two threads
permit K3's other 22-thread CPU work; they are not a CPU affinity reservation.

Mounted containers run as operator UID/GID, which has no passwd entry in the
minimal image. Set `USER=codex-operator`, `LOGNAME=codex-operator`, `HOME=/tmp`,
and `TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor` from startup. A real non-root
import check exposed `getpwuid(): uid not found: 1000`; a later duplicate
Torch cache-registration error was its secondary symptom, not a model failure.

## Inputs and mounts

Use only the controller's structured **directory** mounts. Source trees must
be immutable reviewed archives, not an active shared checkout. All paths need
canonical/no-symlink validation through installed `remote_runtime.validate_mounts`.
Do not mount files individually, and do not use banned `/run/...` targets.

| Source directory | Container directory | Access |
| --- | --- | --- |
| Newly staged reviewed NF source (complete campaign lanes included) | `/code/neural-forge` | read-only |
| `/home/cereal/models/Qwen3.8-27B-Uncensored` | `/parent` | read-only |
| Dedicated staged directory containing only the exact evidence JSON | `/inputs/evidence` | read-only |
| New dedicated S3 output parent directory | `/run-output` | writable |

Stage the existing evidence, not a regeneration:
`/home/cereal/omni-campaign/composite/corpus/omni-evidence-27b-v1.json`.
Its internal seal is
`8a3422d89d245b79a152ffb1de44d077d39ec5574dc063bf4e8482a5107b4870`;
its tokenizer is
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.
Do not mount K3 output or Claude's old snapshots into S3.

## Preflight and command inside the allocated container

With the pinned environment above, first run:

```sh
python /code/neural-forge/reports/omni-campaign-2026-09/composite/tools/preflight_s3_tracked.py \
  --parent /parent \
  --evidence /inputs/evidence/omni-evidence-27b-v1.json \
  --out /run-output/S3-C2K4352-sep07 \
  --check-runtime --check-memory
```

This reads metadata and safetensors headers only, verifies the evidence seal,
tokenizer, unique batch IDs, valid token IDs, splits, source hashes, pinned
runtime, and fresh output. The header/stat inventory is explicitly **not a
full parent digest**. The production builder's full digest and both rechecks
remain enabled: three reads of roughly 55.6 GB are the existing integrity cost.

Exact fold command (no review script, no GGUF conversion, no resident parent):

```sh
python /code/neural-forge/reports/omni-campaign-2026-09/composite/tools/qf_stream27.py \
  --streamed --parent /parent \
  --evidence /inputs/evidence/omni-evidence-27b-v1.json \
  --out /run-output/S3-C2K4352-sep07 \
  --tag S3-C2K4352-sep07 --parent-code Q38-27BU \
  --device cuda:0 --dtype bfloat16 \
  --mlp-keep 4352 --mlp-group 3 --solve-form gram --drop-mtp --full-sequence-preflight \
  --notes 'Fresh tracked S3 fold; 22 CPU threads; GPU4 mapped to cuda:0; no training; frozen review remains separate.'
```

Use GPU4's RTX3090 through the allocator, not a host CUDA ordinal in the
container. Do not add `--resident-check`: the BF16 parent exceeds 24 GiB.
The full-sequence preflight runs one actual evidence row through all layers
and the last-token head before allocating the parity Grams. It records finite
outputs and observed CUDA allocated/reserved peaks; it is not quality acceptance
or a host parity-peak guarantee. Leave ordinary streamed parity enabled. It checks determinism, hook placement,
and full-trunk tap parity, not behavioural acceptance or a rerun review baseline.

## Resource assessment and acceptance limits

Observed parent: hidden 5120, intermediate 17408, 64 text layers (48 linear,
16 full attention), untied vocab 248320, plus the unchanged vision tower.
The corpus has 24 rows of 4096 tokens. Its routes are text/code/math/tool/
reasoning; that is not native image/video/audio or long-context validation.

The stock model has 27,356,728,560 parameters including vision. C2 removes
12,834,570,240 parameters and leaves 14,522,158,320: about 27.05 GiB BF16.
MTP omissions are separately counted from indexed headers and receive **no
fold-ratio credit**. The nested VL text config must declare MTP zero; the S3
patch fixes that prior export mismatch.

GPU forwarding uses BF16 streamed layers; Gram formation, selection, and the
prior-anchored FP64 ladder run on CPU. GPU fixed embedding+head is 4.74 GiB;
one full-attention FP32 score matrix at length4096 is 1.50 GiB, before softmax,
layer buffers, state copies, and allocator temporaries. An RTX3090 is a plausible
fit, not a measured VRAM proof: validate actual full-sequence mechanics in the
tracked GPU before committing to a long run, and retain peak-memory evidence.

Reserve **128 GiB host RAM** for S3 admission and keep K3's separate headroom.
Major CPU components: group-of-3 FP64 MLP weights 5.98 GiB, inner/check Grams+
RHS 17.53 GiB, BF16 XY stash 5.63 GiB, residual 0.94 GiB, final folded cache 7.97 GiB.
Independent review estimates roughly **100.30 GiB nominal anonymous tensor
high-water** in parity, before allocator/MKL workspace and mapped pages.
The 128 GiB admission check is not a cgroup reservation or a measured peak.
Parity temporarily retains two group captures and compares against a row-form
design, so `--solve-form gram` does not eliminate its transient row allocation.
Export assembles the full BF16 parent (~50.96 GiB) on CPU, then releases it
before the **FP32 child reload (~54.10 GiB)**. Folded cache remains live during
export. Components are not peak RSS guarantees; monitor system headroom and
do not start another full resident review/export concurrently.

Run frozen review only **after** the child's sealed receipt exists. Existing
`--review-script` calls the review before writing that receipt. Root owns the
offline review adapter and cached-parent-baseline reuse. A BF16 child is itself
larger than GPU4 memory, so review needs an already validated streaming/offload
path or a separately allocated larger GPU; do not assume resident GPU4 fits.
No C2 proxy metric or successful export establishes behavioural acceptance.

## Validation

The new pure CPU preflight/MTP tests pass locally. The wider existing streamed
suite needs Linux with Transformers **5.12.1**: Windows currently has 5.13.0,
which removed `_update_linear_attn_mask`, and its `/mnt/ai_data` path semantics
also differ. The pinned Linux 5.12.1 API was independently checked present.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -o addopts='' -p no:cacheprovider \
  reports/omni-campaign-2026-09/composite/tools/test_qf_s3_preflight.py \
  reports/omni-campaign-2026-09/composite/tools/test_qf_stream27.py -q
```

No source integrity checks, fold kernels, selection laws, thresholds, parent
weights, or existing outputs were changed by this preparation.
