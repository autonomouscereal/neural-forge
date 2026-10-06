# Shared-forward QIMG stack capture

`c16_qimg_stack_capture.py` adds a runnable multi-block capture route without
editing the frozen existing writers. It reuses the actual pinned Comfy source
loader and `QimgBlockTap`, verifies the real encoded image/context manifests,
and loads the full parent exactly once per capture attempt. One source forward
supplies every requested block. Actual taps are cloned by the existing source
hooks, then streamed into independent safetensors files and released block by
block. There is no fidelity, cosine, update or compression acceptance threshold.
Actual source/shape/load/identity and nonfinite activation failures still stop
the capture. No services, allocator, model training or capability evaluation
are operated by this program.

## Commands

First use a small real multi-block subset, with the actual verified source and
encoded manifest paths supplied by the coordinator:

```text
python -B c16_qimg_stack_capture.py --weights <verified-original-QIMG.safetensors> --inputs <actual-encoded-batch1/capture_inputs.json> --blocks 0,30,59 --out <new-small-stack-capture> --comfy-root <verified-Comfy-root> --device cuda:0 --max-tokens 2048 --max-forwards 32 --validate-only
```

Removing `--validate-only` launches actual source capture. The validation path
loads and verifies the bounded encoded tensor files and model headers/code;
it does not load the diffusion parent or run a forward. A full requested capture
uses `--blocks 0-59` and up to four actual encoded manifest paths after `--inputs`.
Each manifest retains its own encoding receipt, image/prompt identity, sigma,
token IDs and valid text mask. The tool does not create or substitute encodings.

Resume requires a pinned previous canonical manifest and a **new output root**:

```text
python -B c16_qimg_stack_capture.py --weights <same-verified-source> --inputs <same-actual-input-manifests> --blocks <same-block-list> --out <new-resume-root> --comfy-root <same-verified-Comfy-root> --device cuda:0 --resume-manifest <previous-stack_capture_manifest.json> --resume-sha256 <exact-previous-SHA256>
```

Code/input/source/stat/device/request identity must match. Every referenced
artifact is checked by full file and tensor hashes, shapes, dtypes and finite
values; shared journals bind actual runtime/source coefficient digests, sample
metadata and common prediction identity. Final block receipt views are also
reverified. Resume ancestry is bounded to eight attempts. Repeated inherited files
are checked once per resume preflight, rather than repeatedly scanning the same
body for every ancestor. If the previous capture already completed, this path
verifies it and emits a new reference manifest without reloading or replaying
the parent. A changed source/code/runtime needs a separate explicit run, not a
silent resume rewrite.

## Artifacts and coverage

The new versioned schemas are:

- `neural_forge.qimg_stack_capture.v1`: immutable per-forward snapshots and final
  `stack_capture_manifest.json`, exact request identity, ancestry and coverage.
- `neural_forge.qimg_stack_source_forward.v1`: one complete shared source pass,
  unique pass identity, shared prediction hash and every requested block payload.
- `neural_forge.qimg_stack_block_capture.v1`: each block's own final receipt,
  exact frozen coefficients and completed-forward references with explicit gaps.

Every forward's canonical journal is published only after all its requested
block payloads were written and read back. A disk/source failure preserves
already written bytes, partial snapshots and a final partial manifest; an
incomplete forward is not counted. Resume reruns that whole forward rather
than mixing different source passes and pretending they were one execution.
Each snapshot and final receipt is a new file published with an exclusive
atomic hard link after staged bytes are flushed. Staged files are preserved;
they are not completion markers. Sudden-loss directory durability is not
promised. Missing canonical markers cannot establish completion.

Coverage distinguishes a complete requested subset from all contiguous 60
parent blocks. Even all forward journals present cannot make the manifest
complete when final source/input identity checks fail. Capability, fold and
installation claims stay false. A partial root may contain useful captured
bytes, but does not establish complete intended coverage.

Before completion, the bounded input parser rechecks upstream image hashes,
encoder headers, manifests, encoding receipts and actual encoded tensor bytes;
helper and Comfy source pins must also remain stable. This does not rerun any
encoding or source model forward.

Each selected block preserves its source block plus the four original global
time tensors for downstream per-block work. `source/global_source.safetensors`
also preserves the four time tensors and the actual nine remaining global
input/context/output tensors. All copied source coefficients are compared to
the loaded parent's exact storage bytes. Shared time copies are duplicate
storage, not separate model parameters or compression credit. The complete
parent raw-file digest is not newly claimed; emitted source subsets have full
file/tensor digests, with parent header/stat identity checked around execution.

These new receipts intentionally are not relabeled legacy single-block capture
receipts. The next stack-fold reader must consume the actual artifact references,
join blocks/input groups and preserve declared coverage. This tool does not
implement full-stack folding, native QIMG inference, host charts or Chimera
checkpoint installation.

## Resources and tests

The output estimate conservatively sums the existing single-block estimate
across requested blocks and batches, including duplicated common fields. It
requires at least another 4 GiB free and never deletes files to make space.
One source pass holds simultaneous CPU-cloned taps; streaming releases them
after serialization, but does not remove the initial simultaneous-tap peak.
Actual peak RSS and initialized CUDA peaks are recorded. The coordinator still
must inspect actual memory/disk/GPU resources before a real run. No all60 memory
or speed claim follows from tiny fixtures.

`test_c16_qimg_stack_capture.py` exercises actual Torch hooks and safetensors
I/O. Tiny simultaneous captures are compared byte-for-byte to separate
single-block captures. Tests cover one shared source call, all60 tiny block
coverage, source/emitter failures and hook cleanup, finite guards, canonical
publication, payload tampering, explicit partial coverage, immutable resume
and complete-resume without parent loading. Driver tests explicitly mock the
source/header/class identity and resource observations; they are not actual
Comfy model, real encoding or production resource evidence.

```text
python -B -m pytest -q -p no:cacheprovider test_c16_qimg_stack_capture.py
```
