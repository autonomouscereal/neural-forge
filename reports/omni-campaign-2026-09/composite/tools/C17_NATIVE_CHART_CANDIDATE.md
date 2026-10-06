# Native paired read/write chart candidate

`c17_fit_native_charts.py` takes real `c17_capture_host.py` and
`c17_capture_k3.py` observations and fits two **different** directed maps:

- `C_out`: actual host `ffn_input` to actual folded K3 `ffn_input`.
- `C_in`: actual folded K3 `ffn_output` to actual host `ffn_output`.

It does not reuse legacy prenorm rows or fit an input-space map and relabel it
as a write map. Original S1 byte-paired positions, ordering, batch IDs and outer
fit/held evidence groups must agree exactly. A multi-K3-layer pool repeats the
same host rows once per source layer, with unchanged groups and equal row weight.

Both maps are explicitly fitted **through the origin**. C10 cannot faithfully
represent learned affine offsets in its bias-free child, so `C_out_bias` and
`C_in_bias` are fixed zero during fitting, never learned and then dropped. This
removes only the chart-offset discrepancy; it does not establish that the
downstream installed computation preserves native behavior, or that separately
fitted maps are inverse/chart-law preserving.

## Execution (CPU, no model load or installation)

After actual captures exist and their provenance has been independently reviewed:

```text
python /code/composite/tools/c17_fit_native_charts.py \
  --plan /inputs/c17-native-chart-plan.json --plan-sha256 <full SHA256> \
  --out /run-output/c17-native-chart-candidate
```

Every input reference is an absolute path plus independently pinned full SHA256.
Output must be fresh; no overwrite/resume/cleanup is performed. The plan shape is:

```json
{
  "schema": "neural-forge.c17-native-chart-plan.v1",
  "fixture": false,
  "host_layer": 0,
  "host_capture": {"path": "/host-capture/capture_receipt.json", "sha256": "..."},
  "host_import_receipt": {"path": "/inputs/actual-host-import.json", "sha256": "..."},
  "k3_captures": [
    {
      "layer": 3,
      "capture": {"path": "/k3-capture/capture_receipt.json", "sha256": "..."},
      "accepted_receipt": {"path": "/inputs/accepted-k3.json", "sha256": "..."}
    }
  ],
  "evidence": {"path": "/inputs/evidence.json", "sha256": "..."},
  "boundary_map": {"path": "/inputs/boundary_map.json", "sha256": "..."},
  "seed": 20260907,
  "threads": 22,
  "chunk": 512,
  "max_working_bytes": 17179869184
}
```

Placeholders above are not runnable evidence. Select host/source layers from the
reviewed alignment; the tool does not invent a complete 92:64 alignment. Source
layers must be a sorted unique list. All selected source captures must have the
same accepted native resume context. Window captures require the selected layer's
exact folded-prefix evidence as well as successful final-window verification.

The host gate accepts an actual `chimera_omni.import.qwen35.v1` receipt whose
checkpoint config/index identities match the host capture, with matched shard
coverage and no output-gate waiver. A preflight is not an import gate. This
proves source/import identity only: **post-S3 host behavioral acceptance remains
unavailable until the actual post-S3 host and its separate behavior gate exist**.
The current S3 job is not claimed complete by this tool.

The FP64 ridge objective, relative lambda scale and nine-rung ladder reuse the
reviewed `mediafold.linalg` convention. Lambdas are chosen independently per map
on an inner partition of the original **fit evidence groups**. Original outer
held groups never select a rung. Each map is then refitted on all original fit
rows. At least two original fit groups are required. Gram/RHS accumulation and
diagnostics are row-chunked; no complete FP64 row-copy pool is constructed. The
solve refuses a Cholesky failure instead of quietly increasing lambda or falling
back to an identity/prior. Stored F32 coefficients are measured separately from
the FP64 solution, with fit/held relL2, mean/p05 cosine, R² and held zero baseline.
These are diagnostics, **not invented intelligence thresholds**.

Both `refit_regime` and `inner_fit_regime` distinguish unique original host-pair
row identities from pooled observations. For `C_out`, repeating the same host
design against multiple K3 layers increases observation count, not the design
row bound. For `C_in`, source-layer/host-pair identities may add design rows, but
are still only a rank upper bound. Numerical rank is not measured. The legacy
`underdetermined` diagnostic is `true` when the row bound proves a shortage and
`null` otherwise; enough row identities never establish full column rank.
Neither these counts nor their interpretation changes fitting, lambda selection,
stored coefficients, scores, or candidate-only acceptance restrictions.

`max_working_bytes` bounds a conservative numerical-workspace estimate; it excludes
OS pages backing the mapped capture files and is not an RSS or memory reservation.
Budget those captured rows separately. All source files are full-hashed with stat
stability. Actual remote execution must still use the supported tracked runtime;
this source helper never starts one.

## Candidate boundary and remaining acceptance work

The only output is `Lxx/ridge.safetensors` plus `Lxx/candidate_layer.json`. The
candidate includes C10's full `neural-forge.c17-native-chart-boundaries.v1`
domain/codomain provenance, input hashes, group splits, numerical trail and
explicit `production_install_eligible=false`.

**No `layer.json` is emitted**. C10 also refuses every production-native chart
before chart I/O: copying or renaming candidate metadata, removing restrictions,
or asserting affirmative eligibility cannot promote it. The explicit marked,
tiny CPU fixture route remains available only for mechanics tests.

There is no current production admission route. Before implementing one, a
reviewed acceptance policy and actual installed-computation evaluator must bind
these exact map bytes, resulting post-S3 host identity, accepted K3 source
weights and untouched held evidence. `native_chart_boundaries` remains a
provenance-only validator available to this fitter, not an acceptance gate.
Attention/residual transfer, every other source/modality, and a functioning dense
all-source child's capability demonstration remain uncompleted gates.

`--synthetic-fixture-root` is only for explicit tiny CPU numerical/provenance
tests, with every input/output confined to that root and `fixture=true` retained
in the candidate. Such tests are not evidence of model capability.
