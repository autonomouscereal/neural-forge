# Chimera Omni recovery preparation — 2026-10-13

`training.py` prepares intelligence and tool supervision and checks native media
contracts. It has no training or GPU-launch command. Corpus records use
`recovery_corpus.v1`; media routes fail closed without actual digest-pinned assets.

## Text and tools

Run with the parent's locally available tokenizer and CPU Torch/safetensors:

```bash
CUDA_VISIBLE_DEVICES='' python experiments/chimera_recovery/training.py prepare-text \
  --corpus /path/to/recovery.train.jsonl \
  --tokenizer /home/cereal/models/Qwen3.8-27B-Uncensored \
  --out /path/to/new/packed --max-length 4096
```

Every row is one complete native conversation or a terminal assistant tool
action. Tool schemas, JSON arguments, parallel result links, and message ownership
are validated. Original call IDs remain in receipts. When the parent template
omits IDs, the packer follows that authentic template and preserves IDs as audited
metadata. All assistant bodies, including tool calls and closing EOS, may be
supervised; system/user/tool results and role headers cannot be supervised.
Selected earlier assistant turns can be masked explicitly. Oversized rows refuse
instead of truncating a call/result pair. Media cannot be discarded into a text
shard. The shard contains `input_ids`, `lengths`, and bool `assistant_mask`.

The existing teacher-cache builder accepts this shard's `input_ids` and
`lengths`. Its build-text command loads the teacher; it is a later runtime action,
not part of this CPU preparation. Once that new cache is sealed, bind it:

```bash
python experiments/chimera_recovery/training.py bind-text-cache \
  --packed-manifest /path/to/new/packed/manifest.json \
  --cache /path/to/sealed/new/teacher_cache \
  --teacher-sha256 <audited-immutable-teacher-tree-sha256> \
  --out /path/to/new/recovery_data_config.json
```

The binder refuses unsealed, structurally invalid, wrong-corpus, wrong-vocabulary,
or changed-teacher caches. It emits an existing-trainer-compatible text source
with explicit `supervision: assistant_only`. Production `QTextCache` still checks
actual teacher shard contents when loading the source.

The minimal `streamed_supervision.patch` is against exact source SHA256
`77091f8ab1581c1edac3e3f1c1e5d727d60f8fb8465745c03a6eb6b12f0567da`,
verified against the live 2026-10-13 14:18:17 UTC inventory. Apply only to an
isolated candidate of that source, after the dated recovery amendment:

```bash
python experiments/chimera_recovery/training.py patch-streamed \
  --source /path/to/audited/streamed_kd_trainer.py \
  --out /path/to/new/streamed_supervision.patch
```

The patch validates corpus mask shape/dtype and padding. Token t's objective is
gated by assistant ownership of token t+1. This gates CE, EOS, Qwen KD, K3 KD,
unlikelihood, and JSD. The corpus and supervision policy participate in exact
data/resume identity. Unmasked legacy sources preserve their existing objective.
No D1 source or frozen preregistration body is edited by this package.

## Native media and readiness

`validate_native_batch` checks actual bridge names and shapes: features
`[B,M,6144]`, type/indices/validity `[B,M]`, positions `[B,M,4]`, detached targets,
nonnegative media loss weights, finite noisy inputs and sigma, and H3 packed
source positions `[B,S,3]`. It detects direct target/input aliasing; media
sidecar validation additionally detects identical input/target bytes and explicitly
declared clean-target leakage. Those checks cannot prove a cache adapter has no
semantic leakage; its extraction recipe still requires independent review.

`native_velocity_loss` is executable type-balanced MSE over each modality's
emission band in velocity `(x_t-x0)/sigma`. Explicit `x0` predictions convert
differentiably; parent raw targets require the audited per-type sign. This is
velocity matching as frozen D3b requires. The existing D3 x0 endpoint-MSE default
is a different objective and remains blocked unless selected in a dated amendment.
This helper alone does not wire a production D3 driver or create media teacher
caches. `optimizer_ownership` checks the actual trainable delta bank, including
functional tensors outside frozen module trees, against optimizer parameters.

`flow_velocity_loss` exports the existing D3 `flow_x0_loss` Python signature and
uses `bands[type]['emission']`. Noisy bridge inputs are remapped explicitly from
`bands[type]['feature']` before x0 conversion. `ReadyObjectiveAdapter` binds
actual tensors to D3b's `loss_fn(prediction)` callback; it requires a reviewed
`chimera_recovery.native_flow_cache.v1` receipt with sealed/finite/structural,
guided-teacher, and input/target-separation attestations, teacher/cache/corpus/
amendment hashes, declared student/teacher parameterizations, and layouts.
Production cache adapters must emit that receipt; a fake receipt is not evidence.

The candidate D3 patch adds a real optional callback to its existing lr=0 rate
entry point, selected by `--flow-objective guided_velocity --objective-adapter
/path/to/training.py`. It preserves existing x0 rate/synthetic defaults. Rate
targets remain random, labeled fixtures; this flag does not create a production
teacher cache or production D3 run. The baseline documents `--flow-param`, but
that flag does not exist in its CLI parser. Python `param='raw'` still selects
raw-head parameterization, and its old loss still scores converted x0 endpoints.

```bash
python experiments/chimera_recovery/training.py patch-d3-objective \
  --source /path/to/audited/d3_hybrid_module.py \
  --out /path/to/new/d3_velocity_objective.patch
```

The D3 source baseline is SHA256
`540fca2d9d976da336c8d24ae5369c5a303ff37367cae54a9bcfc762a7e95197`.
The CPU test compares adapter loss and gradients against independent sigma-
scaled endpoint algebra, including distinct input/output channel allocations.

All semantic routes have explicit dispatch; unknown routes never become text.
Vision/video/audio understanding require real conditioning and answer caches.
Image/edit, audio, music, joint AV, and volume generation require their native
cache adapters and teacher targets. H3 synchronized AV and HY3D genuine geometry
remain corpus requirements. No active thermal prerequisite is imposed here.

```bash
python experiments/chimera_recovery/training.py cpu-dryrun --out /path/to/contract.json
python experiments/chimera_recovery/training.py readiness \
  --campaign /path/to/distill --corpus /path/to/recovery.train.jsonl \
  --out /path/to/readiness.json
CUDA_VISIBLE_DEVICES='' CHIMERA_STREAMED_BASELINE=/path/to/audited/streamed_kd_trainer.py \
  python -m pytest tests/test_chimera_recovery_training.py -q
```

Readiness reports preparation capabilities separately from production blockers
and returns exit 3 while production is unverified. `cpu-dryrun` describes contracts
and explicitly performs no model forward/backward. Torch tests exercise numerical
losses, gradients, ownership, masks, and the actual patched `build_batch` extracted
from the supplied baseline, without importing model classes or loading weights.
Tokenizer doubles in tests are labeled doubles; an actual-parent run is needed
for native tokenizer evidence. No synthetic fixture establishes model intelligence.
