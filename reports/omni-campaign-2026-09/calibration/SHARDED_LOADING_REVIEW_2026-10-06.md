# Stock-HF frozen text review: explicit resident sharding

The existing runner loaded a whole checkpoint into CPU RAM before calling
`model.to(device)`. That route cannot load the 14,522,158,320-parameter BF16
S3 checkpoint safely when only about 20.9 GiB of host RAM is available. This
change adds optional direct stock-Transformers placement. It establishes
loader mechanics and guards, not S3 behavioral retention or GPU fit.

## Contract

`oc_run_frozen_bank.py` accepts optional `--device-map` with exactly `auto`,
`balanced`, `balanced_low_0`, or `sequential`. Repeat `--max-memory` for `cpu`
and every CUDA index visible to the process, using integer `MiB` or `GiB`
budgets. Indices are relative to `CUDA_VISIBLE_DEVICES`. A zero limit excludes
a device; zero-budget GPUs are not queried with `mem_get_info`.

For two visible GPUs, the placement arguments can be:

```text
--device cuda:0 --device-map balanced
--max-memory 0=20GiB --max-memory 1=20GiB --max-memory cpu=0GiB
```

These arguments do not reserve GPUs. Allocation must be coordinated before
launch. The loader rejects duplicate/malformed limits, unknown indices,
missing visible devices, budgets exceeding currently free CUDA memory or
available physical RAM, an excluded `--device`, and use with `--tiny-random`.
Budgets control weight placement; KV cache, activations and runtime overhead
still need headroom. No device-map file or arbitrary code loader is accepted.

Mapped loading passes the same requested dtype and `trust_remote_code=False`
directly to `from_pretrained`, together with the strategy and byte budgets.
It avoids the whole-model `.to(...)` call. Actual resident parameter devices
are checked against positive budgets, and disk/meta placement is refused.
Mixed CPU/GPU offload with meta parameters is outside this resident-only
option and fails explicitly; no disk offload directory is supplied.

Input IDs for the preflight, emit-zero diagnostic, review and coherence smoke
use the actual input embedding device. Export metadata records the actual
device map, its source, the strategy, limits, input device, all used device
architectures and floating parameter dtype counts. Stock Transformers 5.12.1
does not retain `hf_device_map` on the tested CPU-only result; in that case
the runner measures `named_parameters()` devices directly. It never invents
an intended map. FP32 norms and buffers are not forcibly cast or rejected.
Mapped placement, dtype counts and all used compute architectures also enter
the dossier's lane-validity tuple. With these new options omitted, the old
loader call, whole-model move, generation settings and validity tuple remain.

## Existing evaluation scope and staged recipes

Frozen v1 contains 16 text prompts (15 scored cases plus charged safety).
Its file SHA256 is
`0ee6bf14cef51bafc26990db44e5038771426c620c0381285c0cd5571e92786b`.
Its existing greedy/cache review cap is 1,024 tokens. The protocol SHA256 is
`e5e3776885976b1be5a8469323f7083d40c11900cca66b862eb0da41517a4d7b`.
No bank, hash, stop rule, chat template or generation cap changed here.

The existing small mechanics checks are the eight-token manual/generate
comparison, emit-zero diagnostic, and three disjoint 128-token coherence
prompts. The smoke file hash is
`82ebf70a1606ec8a57c4daa31218fe1d376a91f211246c78716615ca66d2ceca`.
They are diagnostics, with coherence read by a human.

`CAMPAIGN.md` section 4 names an existing five-case bank-v2 text side
measurement: `code-rle-encode-A,reason-arith-two-step-A,lc-8k-notice-select-A,lc-8k-notice-select-B,mt-dietary-correction-B`,
using `--thinking off --text-only`. Its full fold package requires v1 on both
thinking surfaces, the bank-v2 text lanes and a matching parent comparison.
Bank v2 has 30 runnable text cases; its vision and AV/media specifications are
draft and excluded by this runner. No automatic intelligence pass threshold
is introduced. The existing `oc_assess.py` requires human per-case readings.

The copied local Q35-4B scoring sheet is a different parent from S3/Q38 and
contains ungraded upper-bound columns. It cannot establish S3 retention.
Matching Q38 parent outputs and exact bank/surface identities must be verified
before a retention comparison. The composite S8 stage separately requires
actual vision and decoded image/video/audio/music/volume/mesh evidence.

## Historical S3 review-hook mismatch

The existing S3 `qf-build-receipt.json` records a dry-run review hook naming
`run_omni_frozen_review.py` with `--receipt`, `--child-model` and
`--parent-model`. Current `composite/tools/run_omni_frozen_review.py` instead
requires `--model` and a `chimera_omni.checkpoint.v1` native checkpoint.
The S3 artifact is stock HF `Qwen3_5ForConditionalGeneration`; those interfaces
are incompatible. The receipt says `ran:false`, reason `dry-run`. This work
does not rewrite that historical receipt or claim the hook was run. The
stock-HF calibration runner is the applicable existing text review surface.

## Measured validation

Tests were written before implementation. On unmodified integration commit
`e3e72abe3c0374129ca51b305fd2144ad2ae9a0f`, the new contract suite had
32 failures, 1 pass and 2 dependency skips in 0.45 seconds. The existing
calibration suite had 71 passes and 6 Torch skips in 16.19 seconds.

Final existing local CPU Torch/HF runtime: combined calibration and loader
suites had **116 passes, 1 skip in 22.08 seconds**. The single skip is the
mapped real-HF test because local Accelerate is absent. The established
calibration suite alone had 77 passes in 15.61 seconds.

Final existing server runtime: Torch 2.11.0+cu128, Transformers 5.12.1,
Accelerate 1.14.0; **40 passes in 8.96 seconds**. Tests used
`CUDA_VISIBLE_DEVICES=''`, offline HF flags and one CPU thread. Peak process
RSS was 871.3828125 MiB. Each real fixture had 2,016 parameters, 8,064 raw
weight bytes and 10,888 serialized safetensors bytes across ten shards.
Both default and sequential CPU loading reproduced pre-save FP32 logits
bit-exactly (`rtol=0`, `atol=0`), with unchanged parameter counts and dtype.
No trained checkpoint, GPU allocation, shared service or source was changed
on the server. Fixture location:
`/tmp/omni-oct06-sharded-loader-ea427a575db8/pytest-final/`.

Validated source SHA256:

- `oc_run_frozen_bank.py`: `dc14af0c3442d656fa41868c0551a5e68a47dcb1f8676d7be88ad13e25ba541e`
- `test_oc_sharded_loader.py`: `c716d0967b40b67ae972c80d3356c6e39f216407d98c347a134e5f39a3067951`

Peer review found two issues which are addressed: mapped validity must include
every used device architecture, and tests must cover actual meta/unbudgeted
parameters and legitimate FP32 norm accounting. The CPU fixture specifies zero
budgets for any visible GPU and was executed with GPUs hidden.

Large S3 BF16 loading, multi-GPU free-run behavior, readable text quality,
parent retention and multimodal capability remain unmeasured by this change.
