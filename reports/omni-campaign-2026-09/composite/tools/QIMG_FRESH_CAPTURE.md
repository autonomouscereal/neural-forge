# Fresh QIMG capture-only entry point

`c16_qimg_fresh_capture.py` provides an explicit opt-in path for new frozen
calibration plans. The existing `c16_qimg_broad_capture.py` command retains its
September 7 plan contract. Neither fresh capture nor its receipts fit, accept,
install or claim a folded model.

The new path requires a raw-pinned `neural_forge.qimg_fresh_capture_contract.v1`
and `neural_forge.qimg_fresh_capture_request.v1`. The contract references a new
recipe revision containing the actual raw source hashes, including all fresh
helpers, native capture/encoding/generation helpers and mediafold initializer/
solver. Bind the final checkout bytes and commit before external freeze; Git
LF archive hashes and a Windows raw checkout hash are different domains.

`c16_qimg_fresh_metadata.py` is a byte-preserved dependency from the independently
reviewed v2 metadata proposal, SHA256
`46a65ddf5c968083e8ebe01c1fdbf0d2ba6ad3e9684294acd752dd4b2d3fe946`.
Its permanently false launch flags remain unchanged. Its metadata completeness
is not launch authority. The distinct staged API in
`c16_qimg_fresh_preregistration.py` reuses its strict parsing and metadata checks.

The stages are:

1. `pregeneration`: four pinned eight-image plans with six FIT/two held each,
   ordered family assignments, complete historical/family/frozen/hidden/gold/
   media exclusion QA, and an external freeze binding contract, recipe, plans
   and assignments. Generation and actual capture-start fields must be null.
   This command checks prerequisites; it does not generate images.
2. `seal`: complete generated image identities, later external generation and
   postgeneration QA, plus all four native parent receipts. It verifies each
   original 512x512 RGB PNG, native teacher source/runtime and deterministic
   calibration seed, then exclusively writes four encoding plans and a fresh
   sealed bundle. Actual capture-start evidence must still be null.
3. `run`: an independently pinned coordinator authorization, the exact sealed
   bundle and explicit runtime paths. It executes the existing four encoding
   and four source-capture commands with unchanged audits. Each dispatch
   revalidates all source/recipe/authority/image commitments and authorization.
4. `admit`: externally pinned actual start linked to the run, plus a complete
   run receipt. It re-audits the actual tensor receipts, frozen source bytes,
   128 unique forwards and unchanged row floors/caps. This is capture receipt
   admission only; acceptance remains false.

All stage commands except `run` take:

```text
--contract-path PATH --contract-sha SHA
--request-path PATH --request-sha SHA
--code-root NF_ROOT --evidence-root EVIDENCE_ROOT
--anchor-path OWNER_ANCHOR --anchor-sha SHA
```

`seal` additionally requires four ordered `--parent-receipt RELATIVE_JSON SHA`
arguments, explicit `--exclude-capture RELATIVE_JSON SHA` arguments and a fresh
`--out`. `admit` requires `--run-receipt-path PATH --run-receipt-sha SHA` and a
postcapture request/anchor. Keep the sealed precapture request and source anchor
immutable; write separate postcapture files.

`run` takes:

```text
--bundle-path PATH --bundle-sha SHA --out NEW_DIRECTORY
--weights PATH --vae PATH --text-encoder PATH --comfy-root PATH --threads 1_TO_22
--runtime-authorization-path PATH --runtime-authorization-sha SHA
--runtime-anchor-path PATH --runtime-anchor-sha SHA
```

The runtime authorization schema is
`neural_forge.qimg_capture_runtime_authorization.v1`. It binds the exact bundle,
contract, approved interval, runtime image digest and physical GPU UUID, with
action `encode_and_capture_only`. The interval is at most one hour.

Create its separately supplied
`neural_forge.qimg_capture_runtime_authority_anchor.v1` **after sealing**. It binds
the immutable source-authority anchor hash, contract hash and authorization hash.
Putting a bundle-bound authorization hash into the source anchor embedded by
that same bundle creates a circular hash dependency and is unsupported.

The coordinator independently establishes both anchors' origin/currentness,
truthful QA/freeze/generation/start records, actual runtime image/GPU state,
resource interruption/allocation permission and hard stop/restoration. The code
verifies supplied byte commitments; it does not authenticate authority, operate
providers/containers or inspect GPU state. It checks the authorization before
dispatch but cannot terminate a long subprocess when the interval expires.
Enforce the hard stop externally. Admission validates the recorded historical
authorization window rather than requiring it to be current later.

The native source/runtime/PNG/seed/encoding/capture audits are shared with the
legacy path. Gates remain 24 FIT/8 held images, four fixed sigmas, all 128 forwards,
actual FIT text rows at least 6142 and at most 131072 rows per stream. Source
replay/held/update/compression policy is not changed by capture. A separately
reviewed fitter and accepted-artifact writer are required for any fold artifact.

Existing outputs are never overwritten, retried or cleaned automatically.
Incomplete run output receives the shared durable failure receipt. This entry
point neither selects a media width nor omits a source.
