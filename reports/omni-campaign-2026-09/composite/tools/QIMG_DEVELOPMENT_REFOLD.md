# Nonaccepting QIMG development refold runner

`c16_qimg_development_refold.py` is a separate experimental tool. It does not
change or patch the production `c16_qimg_fold.py`, policies, selector primitive
or solver. It makes the optional centered/output-weighted selector callable
through actual source BF16 replay, FIT refitting and native BF16 diagnostics.
No real-model improvement or retained capability is established by its synthetic
CPU tests. This tool does not reserve/schedule a compute window or pause services.
An explicit CUDA invocation creates CUDA tensors and uses device memory.

The data-free tensor interface is:

```python
diagnostics = c16_qimg_development_refold.fold_loaded(loaded_data, policy, device="cpu")
```

It reads supplied tensors and code hashes, not capture/model files. The caller
must establish loaded-data provenance. The file CLI instead uses the original
hash/source/runtime/image-split capture loader and exact policy-byte loader.
Both streams and all captured source forwards replay before either stream is
selected or fitted. Only FIT valid rows enter the fixed selector and existing
parent-anchored affine solver. Native BF16 candidate replay then scores pooled
and every development held forward. Original row minimum, bias-inclusive 4x
parameter count, lambda ladder, update veto and stored BF16 update check remain.
Numerical gates cannot be weakened beyond source relL2 .005, development held
relL2 .30 and cosine .95. A stronger caller gate still grants no acceptance.
Non-finite native source/candidate arithmetic or score/solver diagnostics refuse
before a gate comparison. Finite FP64 coefficients that overflow BF16 storage
produce a tagged refusal with no raw NaN/Inf; an uncomputed update is recorded as
null with its classification, never substituted by zero or called a pass.

Every returned or persisted receipt, including a numerical pass, has
`accepted=false`, `acceptance_enabled=false`, `candidate_written=false` and
`fresh_data_acceptance_validated=false`. The API returns diagnostic metadata and
hashes only. Ephemeral candidate weights are used for native scoring and are
discarded; neither candidate nor selection tensors are returned or written.
There is no accepted output path, installer, promotion, training action, fresh
override or public fixture bypass. Historical observed holdouts are development
data. The policy's old preregistration assertion is identified separately from
the new selector's unestablished chronology/freshness.

Example interface for a future owner-controlled diagnostic, after verifying
resources and pinning exact local policy bytes (the placeholders are not paths
known to exist):

```powershell
& <existing-python> -B c16_qimg_development_refold.py `
  --development-only --captures <verified-capture-receipt-1> <verified-capture-receipt-2> `
  --policy <unchanged-broad-policy> --policy-sha256 <raw-working-file-sha256> `
  --report <new-separate-research-directory>/development_refold.json --device cpu
```

`--validate-only` loads and verifies tensor payloads and applies the row/count
gates, but does not replay or fit; it is not a cheap metadata-only check.
`--device cuda:0` supports the original replay device interface but does not
reserve/schedule a GPU window or authorize interrupting another job. Gram/pivot/ridge work
remains FP64 CPU. Model-scale RAM, disk, runtime and any GPU allocation must be
established before execution; no new compute window was requested or created.

Reports must be new JSON files outside input/capture and code directories.
Existing files and overlaps refuse before tensor loading. Completion and caught
refusal receipts use exclusive creation, flush and fsync before CLI stdout.
The CLI prints only status/provenance and receipt hash. A numerical refusal exits
2; successful development completion exits 0 but remains nonaccepting. Process
termination, disk failure or exhausted resources can prevent a durable receipt;
no mid-solve resume checkpoint is provided or invented.

Receipt provenance pins this tool, the optional selector and imported capture,
production-fold/preflight and solver code. Policy and capture hashes retain
their original domains. No full-parent payload rehash, full joint-block replay,
all-60-block QIMG fold, image-generation retention or assembled Omni child is
claimed by a token-local MLP diagnostic.

Fresh acceptance remains blocked: existing schemas attest capture hashes and
image-disjoint splits but do not establish the new recipe's prior registration
time, exclusion of previously observed development identities, or authority for
a fresh holdout. A production acceptance implementation needs a reviewed pinned
recipe/source/policy manifest, owner-established fresh image identities and
chronology, immutable new captures, unchanged gates, durable accepted-artifact
identity and a separate reviewed integration path. This tool deliberately has
no mode pretending those contracts are complete. Later capability evaluation
belongs after complete folding/assembly, as the owner requested.

Run only small synthetic CPU regressions now:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
& .venv/Scripts/python.exe -B -m pytest reports/omni-campaign-2026-09/composite/tools/test_c16_qimg_development_refold.py -q -p no:cacheprovider
```
