`preflight_k3_continuation.py` prepares the next native L05..L12 window from the
specific accepted L01..L04 prefix replayed by C17. It reads files and prints JSON.
It uses the standard library, never imports Torch, and never writes inputs,
creates an output directory, inspects a live GPU, allocates a card, or launches a
process. The existing `preflight_k3_tracked.py` fresh 0–1 plan remains unchanged.

The production anchors are the existing capture receipt SHA256
`853c6ed70bf8b8a4845da7597425ef7061585f5b8abe528509faf23aa39b3d8b`,
capture plan SHA256
`0500111bbf2e334cf4e9b65017795997c4654ca87f69935313e54e59ce213542`,
and dated digest receipt SHA256
`58bc54f0c635ec78f53a4575155d249ebd10bde4ea302f65872bebe253d7a91f`.
There is no CLI switch to replace these anchors or enable synthetic evidence.
Fixture tests patch anchors in their isolated Python module, then hash real
fixture bytes through the same verifier.

Before invocation, the root prepares the following existing evidence. The
planner does not perform this preparation.

- `--accepted`: immutable accepted output root containing `fold/dense/L01..L04`
  FFNs and done markers, and both `fold/snapshots/before_L05*.safetensors`.
- `--sealed`: the original accepted `k3_dense_pass_receipt.json` and
  `snapshots_manifest.json`, matching their pins in the capture plan.
- `--output`: a **separate already-prepared continuation copy**. Its `/run-output`
  layout must contain byte-identical prefix FFNs/done markers, dual L05
  snapshots, `fold/snapshots/snapshots_manifest.json`, and
  `receipts/k3_dense_pass_receipt.json`. Hard links or symlinks resolving to the
  accepted inputs are refused. Any partial receipt, later layer artifacts, or
  later snapshot is refused. Preserve the existing L00 MLP, native bodies and
  model-level artifacts; a late native window does not regenerate those.
- `--parent`: the exact physical source mapped to the original `/parent` identity.
  Config/index hashes, every index shard basename and all 96 shard size/mtime
  stamps must match the accepted resume context. All 98 unique digest entries
  must exist at the same size, and must cover all 96 shards. The identities of
  the two ancillary digest entries are taken from that pinned receipt, not
  guessed. Config and index are separately hash-verified. This is dated full
  verification plus current metadata checks, **not a new full shard hash pass**.
- `--nf`: the exact seven native tool/lane files bound by the accepted context.
- `--corpus`: the three exact files with their original order/basenames.
- `--capture`, `--capture-plan`, `--digest`: the pinned files above.
- `--runtime`, `--runtime-sha256`, `--image`: a fresh root-collected observation
  of the runtime in the inspected immutable image. The observation is JSON with
  this schema (ellipsis below means copy the complete measured objects):

```json
{
  "schema": "neural-forge.k3-runtime-observation.v1",
  "observed_utc": "2026-10-06T06:01:00+00:00",
  "image_id": "sha256:<64 lowercase hex digits from inspected image>",
  "gpu_uuid": "GPU-f5a09535-fc85-a027-50f7-734b344747dd",
  "runtime": {"...": "complete measured native runtime object"},
  "software": {"...": "complete measured native software object"}
}
```

The observation must be collected from the root's actual controlled runtime,
not populated by copying historical affirmative metadata. Use the existing
native/C17 runtime and software identity routines for those object formats.
Measure them with the exact `python` executable and image entrypoint used by the
printed command, not another interpreter that happens to report similar JSON.
It must match every accepted context field: original Quadro RTX 8000 `sm75`,
47.6 GB, `cuda:0` for device/experts, CPU residuals, Torch KDA, numerical flags,
22 threads, Torch `2.11.0+cu128`, CUDA `12.8`, package and source hashes.
The exact original GPU UUID is required. A 3090, another card, an unknown field,
or a software/dtype change fails closed. The observation's own file SHA256 is
supplied separately by the root. Its timestamp must be no older than five
minutes at both the beginning and end of preflight. This freshness limit is an
operational guard, not a change to any folding acceptance threshold. The planner
checks the caller's pinned observation; it does not authenticate a live GPU.

Example host invocation (all paths must already exist):

```sh
python -B preflight_k3_continuation.py \
  --nf /reviewed/neural-forge --parent /exact/k3-source --corpus /exact/corpus \
  --accepted /readonly/accepted-output --sealed /readonly/sealed-receipts \
  --output /new/prepared-continuation --capture /proof/capture_receipt.json \
  --capture-plan /proof/c17_k3_l3_l4_capture_sep07.json \
  --digest /proof/k3_lexar_digest_verify.json \
  --runtime /proof/fresh-runtime.json --runtime-sha256 <observed-file-sha256> \
  --image sha256:<inspected-image-id>
```

Only after all checks pass does stdout contain a command with `--resume
--layers 5-12 --rung 17408 --acceptance gate`, one writable continuation mount,
and the original runtime settings. On refusal it exits 2 with no plan. The root
must keep the observed runtime and read-only mounts unchanged, acquire resources
through the existing allocator, and review the resulting native receipt before
any later window. The listed later windows (13–24 through 85–92) are conditional
planning information; this tool supports only the current captured prefix.

The planner establishes continuity with an exact local source. It does not
resolve the unproved upstream Kimi K3 repository/revision, substitute another
Kimi checkpoint, admit C17 native charts, prove all 92 MoE layers completed, or
prove the whole multimodal dense child. The first native G4 measurement writes a
calibration baseline; that calibration alone is not independent whole-child
behavioral evidence. Native acceptance constants and source files are unchanged.
