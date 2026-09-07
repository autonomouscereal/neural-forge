# Sealed broad QIMG capture bridge

`tools/c16_qimg_broad_capture.py` consumes four explicit `(generation plan,
plan SHA256, actual parent receipt, receipt SHA256)` pairs, in batch order. It
matches every plan to the committed `qimg_parent_broad_sep07_batch1..4.json`
content, without retyping prompts, seeds, splits or image hash lists. The caller
pins actual archived plan bytes; semantic comparison permits serialization
differences, including the reviewed LF/CRLF difference, not changed plan content.

All four actual receipts must be complete, source-pinned native parent runs with
all eight original entries, in original order. Each original PNG is hashed and
verified as one untransformed RGB 512x512 image. IDs, prompts, PNG hashes and
calibration seeds must be globally unique; all 24 fit/8 held images must remain
disjoint. One or more caller-pinned old capture receipts are mandatory exclusions
and must cover at least the eight previous development images. Add the earlier
canary receipt as an exclusion too if it is within the retained old-image set.
The script does not silently invent a claim that all historical files were
searched: it records the precise exclusion receipts and their image hashes.

Calibration sigmas are exactly `.15, .35, .65, .9`. Each explicit int63 noise seed
is derived deterministically from a versioned SHA256 rule over generation-plan
SHA256, original ID and generation seed; the result and rule are stored before
encoding. They are calibration noise draws, not sampler trajectories or an
alternative image selection policy. The bridge writes four plans and one sealed
bundle into a fresh directory. It does not copy image files or resave PNG bytes.

Example interface (substitute actual paths and independently verified hashes):

```text
python c16_qimg_broad_capture.py seal
  --batch PLAN1 PLAN1_SHA RECEIPT1 RECEIPT1_SHA
  --batch PLAN2 PLAN2_SHA RECEIPT2 RECEIPT2_SHA
  --batch PLAN3 PLAN3_SHA RECEIPT3 RECEIPT3_SHA
  --batch PLAN4 PLAN4_SHA RECEIPT4 RECEIPT4_SHA
  --exclude-capture OLD_RECEIPT OLD_RECEIPT_SHA
  --out NEW_SEALED_DIRECTORY
```

The printed bundle digest is the review boundary. Seal under the same immutable
source tree and mounted absolute image/receipt/plan paths that the execution
container will see. The bundle pins the runner and its six existing tool source
files. A source change or plan change requires a fresh reviewed seal, never an
in-place rewrite. Teacher mount paths may differ from their original generation
paths: header digest, header/file byte counts and explicit header-only scope
must remain exact; both original and current path provenance remain in receipts.
No full 20GB diffusion parent digest is invented or recomputed by this bridge.

After review, the coordinator starts the existing pinned image through its own
tracked GPU allocator and invokes:

```text
python c16_qimg_broad_capture.py run
  --bundle SEALED_BUNDLE --bundle-sha256 REVIEWED_BUNDLE_SHA
  --weights DIFFUSION_WEIGHTS --vae VAE_WEIGHTS --text-encoder TEXT_WEIGHTS
  --comfy-root /opt/ComfyUI --threads 2 --out NEW_RUN_DIRECTORY
```

The runner has **no** Docker, GPU allocation, service, download or cleanup API.
It calls existing production tools with `subprocess.run(..., check=True)` and no
shell. The enclosing image/device/lease identity remains the coordinator's
explicit external obligation. It executes four encoding subprocesses in order,
audits complete image/prompt/seed/tensor linkage, full tensor identities, actual
FP32 sigmas and actual boolean validity masks, then requires **at least 6,142
valid fit text rows** before launching any full parent capture subprocess.
No length estimate, padding count or held text row can satisfy this fit floor.

Then it executes four source-capture subprocesses in order, with block30,
max-forwards32 and max-tokens2048 unchanged. It streams the existing full-tensor
identity and joint-context validators one capture at a time, checks actual
encoded-model-input BF16 casts, token IDs/validity, sigma, source frozen bytes,
complete image/text context, source receipt/encoding linkage and all 128 forward
identities. Actual image row totals must be 98,304 fit +32,768 held =131,072,
exactly the existing converter cap. Text totals also retain the 131,072 cap.
No Gram construction, solver, fold score, threshold update or numerical
acceptance occurs. The current rejected fold remains rejected.

Outputs are the original production encoding/capture artifacts plus one compact
run receipt; failure emits one incomplete receipt and preserves all outputs.
Existing output directories refuse before any subprocess. No automatic retry,
resume, deletion or overwrite is implemented. A failure after some stages needs
coordinator review; the runner does not selectively regenerate successful data.

CPU tests use synthetic PNGs/receipts and mocked native subprocesses; these are
not actual parent evidence. They cover all-batch coverage, original byte/prompt/
split/seed preservation, old-image leakage, altered metadata, plan sealing,
actual validity-mask row counts and F32 sigma precision, sequential stage order,
capture-to-encoding linkage, row floor/cap rejection and preserved failures.
Existing fold/joint/capture suites remain the owners of full native tensor math
validation. The actual broad runtime must still pass pinned Linux validation,
seal the real completed receipts and execute the production sequence.
