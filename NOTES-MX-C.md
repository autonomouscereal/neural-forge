# NOTES-MX-C — matrix campaign, evidence recapture and 0b recalibration

Wave MX-C. Executes the coordinator's decision on `NOTES-MX-A.md` §5.3: neither
extend the trace-cache compatibility policy nor re-point the campaign at a
different corpus, but **recapture the campaign's parent evidence under the
current collector identity** (the w12e precedent) and **recalibrate cell 0b from
artifact bit-identity to behavioral agreement**.

Repo baseline HEAD `a31747d8`, branch `codex/lrc-blank-compression`. The frozen
collector (`hf_lfm2_circuit_trace_cache.py`) and the compatibility policy
(`hf_lfm2_circuit_trace_cache_compatibility.py`) were **not touched**. No
`git add`, no commit.

**Verdicts up front:**

| deliverable | verdict | one line |
|---|---|---|
| fresh matrix evidence capture | **SEALED and readable at HEAD** | 24 batches, cache key `2dec87ac…`, identical parent / corpus / capture-selection / execution identity to the sealed base; only `implementation_sha256` moves, which was the point |
| cell 0b `CAL.C01.T5.X4`, recalibrated | **FAIL — campaign STILL STOPS** | the route now runs to completion and publishes a gradeable dossier, but reproduces only **2/16** of the sealed A9 responses (`token_match_vs_control 0.125`, `mean_lcp 0.327`) against the pre-registered 0.90 |
| wave-1 cells 1–3 | **NOT RUN** | design §4.1: the calibration cell decides whether the campaign may start, and it did not pass |

The failure moved. It is no longer "the harness cannot open the evidence" or
"the route dies before it can report". It is now a single, named, measured
mechanism fact, in §4.4.

---

## 1. The fresh capture

`/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence/`

| field | value |
|---|---|
| trace cache key | `2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff` |
| manifest | `evidence-manifest.json`, `manifest_sha256 219a63c43df129d2…` |
| schema | `neural-forge-lfm2-matrix-campaign-evidence-manifest-v1` |
| batches | 24 (the first 24 rows of the sealed corpus, in sealed order) |
| split | 18 fit / 6 heldout, `whole-record-seeded-sha256-ranking`, seed `lfm2-circuit-balanced-v1`, validation fraction 0.25 |
| rows | 264/batch (256 sampled prefill tokens + 8 forced decode steps) → **4,752 fit / 1,584 heldout** |
| size / wall | 8.0 GB, 326 s on one RTX 3090 |
| `implementation_sha256` | **`d8af5f9d…`** (collector `23e47f4c…`, canonical contract `789c6a6f…`, torch 2.11.0+cu128) |

Four identities are **bit-identical to the sealed evidence base** `76ff3cbc…`,
and that is what makes this a recapture rather than a new corpus:

| identity | sealed `76ff3cbc…` | fresh `2dec87ac…` | |
|---|---|---|---|
| `parent.identity_sha256` | `96a6779d…` | `96a6779d…` | same |
| `capture.selection_sha256` | `f229e5f3…` | `f229e5f3…` | same |
| `execution.execution_sha256` | `3d077b94…` | `3d077b94…` | same |
| per-batch `input_sha256` (first 24) | — | — | **all 24 identical** |
| `implementation_sha256` | `eb3bddeb…` | `d8af5f9d…` | **moved — deliberately** |

The parent identity match is the load-bearing one: `96a6779d…` is the
**bfloat16** parent, which is both what the sealed evidence base was captured
from and what the A9 route pins with `--expected-parent-identity-sha256`. A
float32 capture would have had a different parent identity and the A9 route
would have refused it. That is why the capture is bf16, on an SM86 card
(RTX 3090, `CUDA_VISIBLE_DEVICES=0`), never on the P40s.

The capture selection is reconstructed field-by-field from the sealed manifest
and the driver **refuses** unless the reconstruction reproduces the sealed
`selection_sha256`; `sample_last_tokens`/`max_decode_steps` were not narrowed,
so it does. `_attn_implementation` is forced to `eager` before any forward,
exactly as `prepare_lfm2_tokenizer_bridge_evidence.py:1280` does — without that
the collector refuses causal-circuit evidence outright, which is how the first
attempt failed.

Driver: `/home/cereal/nf-mxc/capture_matrix_evidence.py` (deliberately outside
the deployment checkout, the w12e placement). Report:
`/home/cereal/nf-mxc/capture-report.json`.

### 1.1 The split had to get its own authority

A fresh cache has no balanced-basis phase bound to it, and the matrix campaign
never fits the exact-4x basis those phases exist to serve — so
`load_site_evidence`'s single authority was unreachable, and it fails closed
rather than inventing a split (correctly).

The split is therefore computed by the **same rule** the balanced-basis
machinery seals — `whole-record-seeded-sha256-ranking` over the cache's own
record ids — and sealed into the capture manifest. The reimplementation is not
trusted on my say-so: `--self-test-only` builds synthetic records, runs the
estate's own `hf_lfm2_balanced_circuit_basis._split_records` against them, and
refuses to capture anything if the two disagree. It agrees
(`agrees_with_estate_implementation: true`, recorded in the manifest).

### 1.2 Readability verified through the harness's own loader

Not "the file opens" — the actual `load_site_evidence` path, at three real
sites, with the split manifest as authority
(`/home/cereal/nf-mxc/probe_calibration_floor.py`, run against a 4-batch pilot
capture built the same way):

```
split authority evidence-manifest.json (3 fit / 1 heldout batches, e80af835e665)
C05 conv_out_proj  short_conv.context -> operator.output
  L00  calibration relL2 0.001661   mean cos 0.9999988
  L05  calibration relL2 0.001650   mean cos 0.9999988
  L13  calibration relL2 0.001894   mean cos 0.9999987
split_authority = matrix-campaign-capture-manifest
implementation_sha256 = d8af5f9d…
```

Identical to the last digit whether the parent is loaded float32 or bfloat16 —
the checkpoint's `conv.out_proj` is bfloat16-valued either way — so the residual
is purely the **bf16 quantization of the stored activation rows**, not a dtype
mismatch between the cell and its evidence.

**Consequence the campaign has to register:** the runner's calibration gate
defaults to a 1e-3 tolerance, which sits *below* this floor and would refuse
every body cell for a reason that is storage, not harness. The pre-registered
value for body cells on this evidence base is **5e-3** — 2.6x the measured
floor, still three decades below anything a genuine row-selection or endpoint
error produces (those are O(1) with cosine far below 0.9999). It is written
down with its measurement in
`/mnt/ai_data/neural-forge-matrix/c05-t5-m16-knobs.json` and is a per-cell knob,
not a code default.

The pilot capture was deleted after the measurement (1.4 GB reclaimed on a
filesystem with 15 GB free); it was my own throwaway, not sealed estate data.

---

## 2. Harness changes

All in `experiments/run_lfm2_matrix_cell.py`. Deployed copy re-verified
byte-identical to the repo copy (`sha256 f72f5511fb326fff…`), previous copy
backed up first (§6).

| where | what |
|---|---|
| `experiments/run_lfm2_matrix_cell.py:323` | `SPLIT_MANIFEST_SCHEMA` |
| `experiments/run_lfm2_matrix_cell.py:326` | `_authenticated_split_manifest` — the second split authority |
| `experiments/run_lfm2_matrix_cell.py:423,461` | `load_site_evidence` takes `evidence_split_manifest`; supplying both authorities is refused before anything is opened |
| `experiments/run_lfm2_matrix_cell.py:627` | the site-evidence binding records `split_authority`, `evidence_split_manifest_sha256`, `implementation_sha256` |
| `experiments/run_lfm2_matrix_cell.py:911` | `route_solve_block` extracted from `_a9_token_route` (§4.2 — this is the bug) |
| `experiments/run_lfm2_matrix_cell.py:1157` | `ROUTE_CALIBRATION_MODES` and the two pre-registered readings of 0b's gate |
| `experiments/run_lfm2_matrix_cell.py:1350` | `--evidence-split-manifest` |
| `tests/test_run_lfm2_matrix_cell_fresh_evidence.py` | 13 new tests, 9 of them refusals |

`_authenticated_split_manifest` enforces exactly what
`hf_lfm2_exact4x_first_divergence.authoritative_split_by_batch` enforces: the
manifest re-seals to its own recorded digest, names this exact cache and this
exact parent, carries one row per batch, agrees with every batch reference's
`input_sha256`, and uses both splits. A cached site-evidence payload built under
a different split manifest is also refused, because reusing it would silently
relabel held-out rows as fit. Nothing here weakens the original path: the
balanced-basis authority is untouched and still the default.

The `behavioral-agreement` calibration mode is a **knob**
(`route_calibration_mode`), and `artifact-bit-identity` — §5.2's original
registration — remains the default. In behavioral mode the artifact comparison
is still computed and sealed in full; it is demoted to diagnostic, never
dropped, because it is the evidence of *what* moved.

---

## 3. Proposed ADDENDUM 2 (text only — I did not edit the design doc)

> ## ADDENDUM 2 — 2026-08-10: cell 0b's calibration criterion
>
> §4.1 enters cell 0b to prove the harness can drive the one known behavioral
> success. `NOTES-MX-A` §5.2 registered its tolerance as bit-identity of the
> compiled route artifact. That registration is now known to be unsatisfiable at
> HEAD **through no fault of the harness**, and §5 forbids editing a rule, so it
> is superseded here rather than amended in place.
>
> The cause is recorded, not asserted. A9's *numeric* configuration is fully
> reachable again on an SM86 card: every one of the eleven dtype roles in
> `neural-forge-lfm2-sparse-semantic-numeric-dtype-contract-v1` — including
> `forward_dtype: bfloat16` — reproduces the sealed contract exactly. What does
> not reproduce is the contract's **digest**, because the runtime dtype
> resolver's own receipt is hashed into it and that receipt's schema moved from
> the resolver A9 ran under to `…-runtime-dtype-resolution-v2`. A provenance
> format change therefore changes every downstream phase key, nothing hits the
> sealed codebook phase, and the 4,096-row residual-OMP support search
> recomputes from scratch.
>
> **Cell 0b's criterion is therefore behavioral, and the campaign's own
> tolerance applies unchanged.** The gate at chain link 1 becomes the
> infrastructure question only — did the harness drive the mechanism to a
> complete, sealed route artifact with a gradeable whole-response dossier? The
> acceptance question is answered where §5 rule 1 says every acceptance question
> is answered, at chain link 6: `behavior.free_run.token_match_vs_control` of
> the recompiled route's decoded responses against the **sealed A9 dossier's**
> decoded responses, at the design's own **≥ 0.90**. The artifact-identity
> comparison is still computed and sealed on every 0b instance as a diagnostic,
> so a reader can always see exactly which identities moved.
>
> This is a *weaker* calibration than bit-identity and is recorded as such. It
> is not a waiver: nothing is pinned, no compatibility list is extended, and a
> 0b that fails this criterion stops the campaign exactly as §4.1 requires.

The threshold is **0.90**, not a number fitted to what I measured. Proposing
0.125 as the new bar because 0.125 is what came back would be the goalpost move
the whole pre-registration discipline exists to prevent.

---

## 4. Cell 0b, second reading

Run: `CUDA_VISIBLE_DEVICES=0` (RTX 3090, SM86), main venv (torch 2.11.0+cu128),
`--dtype float32` for the cell shell, the sealed A9 launch parameters otherwise
unchanged, `route_calibration_mode: behavioral-agreement`.
Knobs `/mnt/ai_data/neural-forge-matrix/cal-c01-t5-x4-knobs-mxc2.json`, whose
`_deviations` field names all four departures from the sealed A9 launch.

Receipt `4018355521b93a30…`, instance `5b8e6ee3692a4dcc`, **status `fail`,
`first_divergent_metric: free-run`**, 76.9 s (second launch; every phase was a
cache hit from the first, which ran 32 min 13 s).

| | |
|---|---:|
| `calibration.passed` (infrastructure) | **true** |
| `calibration.artifact_bit_identical` | false |
| `behavior.free_run.token_match_vs_control` | **0.125** (2/16 exact whole responses) |
| `behavior.free_run.mean_lcp` | **0.327** |
| pre-registered threshold | **0.90** |

### 4.1 What the run proves that the first one could not

The first 0b run died at the cached-hidden diagnostic with
`Lfm2CircuitTraceCorruptionError`. This one produced
`cached-hidden-diagnostic.json` against the **fresh capture** — 6,335 rows,
top-1 5,286 (0.834 overall; 0.9987 on rows whose parent winner is a direct
anchor), MRR 0.852 — plus `run-receipt.json`, both codebook references, and the
16-prompt `whole-response-dossier.json`. The harness drives the A9 mechanism
end to end. That question is settled.

Two of the six sealed artifact identities now **match A9 exactly**:
`candidate_support_math_abi_sha256 8cb1feff…` and
`coefficient_solve_math_abi_sha256 48bbbf46…`. The OMP math ABI is the same on
the 3090 under torch 2.11 as it was on the P40 under torch 2.6. The four that
differ are the codebook artifact and its phase keys.

### 4.2 One latent harness defect, found the expensive way

The first launch compiled the entire route — 32 minutes, the full 4,096-row OMP
support search, the diagnostic, all sixteen 256-token generations — and then
**died at `seal_receipt`**:

```
neural_forge.dynamic_circuit_ir.DynamicCircuitIRError: IR payload is not canonical JSON
ValueError: Out of range float values are not JSON compliant: inf
```

`_a9_token_route` reported `"allowed_clamp_ratio": float("inf")`, and
`seal_receipt` canonicalizes with `allow_nan=False`. No receipt, no ledger row,
nothing to show for the compile. It had never fired because no route had ever
run to completion. Fixed to `0.0` with `clamp_engaged: False` — the estate's own
convention for a technique with no clamp lane (`blank_technique_solve`, and the
T5 body-route kernel) — and the block is now `route_solve_block`, a pure
function, so `test_a_completed_route_solve_block_is_sealable` walks a real blank
receipt through the real sealer without a GPU-hour. **The same defect is still
live elsewhere; see §5.**

### 4.3 The numbers, prompt by prompt

`difflib` character ratio, a diagnostic proxy and explicitly *not* the frozen
manual whole-response review:

| | mean |
|---|---:|
| sealed A9 child ~ parent | 0.119 |
| recompiled child ~ parent | 0.107 |
| **recompiled child ~ sealed A9 child** | **0.503** |

Exact whole-response matches vs sealed A9: `context-retrieval` and
`summarization-priorities` (the two shortest responses, 102 and 155 characters).
Highest near-misses: `arithmetic-multi-step` 0.938, `debugging-off-by-one`
0.833, `coding-deduplicate` 0.707. Lowest: `multi-turn-correction` 0.079,
`creative-continuity` 0.107.

Read that table carefully. The sealed A9 child is itself only 0.119 similar to
the parent — A9's "10–12/16" was a *manual intelligence* verdict, never textual
reproduction — and the two children are **four to five times more like each
other than either is like the parent**. This is not a broken mechanism
producing noise. It is the same mechanism compiled twice, landing in the same
behavioral neighbourhood and not on the same point. 0.125 is nowhere near
chance (chance for exact 256-token equality is ~0) and nowhere near 0.90.

### 4.4 Root cause, and it is now a single fact

The compiled codebook differs in **content**, not merely in digest:
`base-codebook-reference.json` shows `candidate_content_sha256` and
`tensor_content.content_sha256` both differing, while
`continuous_coefficient_count` (1,933,888), `omp_orthogonality_tolerance` and
`support_selection` are identical, and `child_state_sha256 0a0550400e0dcc71` is
**identical** — the child body checkpoint never moved; only the route did.

So: identical dtypes, identical math ABI, identical body, different supports.
The residual-OMP candidate search runs on the GPU in float32
(`--selection-device cuda:0 --selection-dtype float32`) over 4,096 anchors with
near-ties resolved by `|residual cosine|`. Pascal and Ampere float32 matmul
kernels do not agree to the last ulp, one tie flips one atom, and every
subsequent step of that row's support diverges. **A9's compiled artifact is
reproducible only on the hardware that compiled it** — a P40 under torch 2.6 —
and that combination is exactly the one that cannot open any parent trace cache
at HEAD (`NOTES-MX-A` §5.3's venv table). There is still no configuration in
which both constraints hold; what has changed is that we now know *why*, and it
is one sentence instead of three suspects.

### 4.5 What I did not do

* Did not lower the threshold to fit the measurement.
* Did not re-run 0b on a P40 to chase bit-identity. That would reproduce the
  first run's death: the Pascal venv's implementation identity `689fe5b0…`
  cannot open the fresh cache either, and pointing the diagnostic back at
  `dd9efbe2…` to dodge that is the audit-waive the coordinator ruled out.
* Did not seal a decode-smoke review. `resolve_outcome` makes a cell with a
  crossed chain link `fail` regardless of the review verdict, so a review on
  this receipt could not change anything and would only add a row implying it
  might have.
* Did not run wave-1 cells 1–3. Design §4.1: if calibration fails, nothing else
  runs. The evidence loader, the split authority and the C05 calibration arm are
  all proven at all three of those sites (§1.2), so cells 1–3 are ready to
  launch the moment the gate opens — but the gate is the gate.

### 4.6 The decision this hands back

1. **Accept 0b as calibrated by the sealed A9 artifacts** and drop the cell to a
   documentation check. §5.3's option 3, now with a measured reason: the
   mechanism demonstrably runs under the harness and lands in A9's behavioral
   neighbourhood; only the support search is non-portable.
2. **Re-register 0b against a portable route.** `--selection-device cpu` makes
   the candidate search hardware-independent, so a *future* 0b would be
   reproducible across machines. It does not reproduce A9 (which ran on cuda),
   so it is a new calibration anchor rather than a reproduction of the old one,
   and it costs one route compile to establish.
3. **Have the frozen manual review judge the recompiled child's 16 responses**
   against A9's rubric. If it retains 10–12/16 the way A9 did, the mechanism is
   calibrated in the sense that actually matters and 0.503 textual agreement is
   the wrong instrument. This is the cheapest of the three and the only one that
   uses the campaign's own authoritative behavioral judge; the dossier is
   already sealed and waiting.

My recommendation is 3, then 1 — and *not* 2 until 3 has been read, because
option 2 spends a compile to answer a question option 3 may already answer.

---

## 5. Findings that are not about this wave

* **The `inf`-in-the-receipt defect is still live in three places** in the
  harness lane's registry, and each one is a wave-1 cell that will burn its GPU
  time and then fail to seal, exactly as §4.2 did:
  * `src/neural_forge/matrix_technique_registry.py:801` —
    `allowed_clamp_ratio` is `float("inf")` whenever the T1 correction report
    carries no allowed clamp ratio. That is wave-1 cells 12–21 and 29.
  * `src/neural_forge/matrix_technique_registry.py:879` — the C14 norm kernel
    sets `allowed_clamp_ratio: float("inf")` *unconditionally*. Wave-1 cell 30
    cannot seal, guaranteed.
  * `src/neural_forge/matrix_technique_registry.py:785` — `condition_number`
    becomes `float("inf")` when the solver reports none. This one is **not** a
    one-line fix: line 786 uses `not math.isfinite(condition_value)` as the
    definition of `condition_cap_saturated`, so replacing the sentinel changes
    the chain's second link. It needs its owner, not a guess from me, which is
    why I reported it instead of patching it.
* **A concurrent process left `src/neural_forge/hf_lfm2_direct_local.py`
  corrupted in the working tree.** At 10:29 local it was rewritten with a
  literal truncation marker at line 762 —
  `except…5826 tokens truncated…ut_embeddings()` — which is a `SyntaxError`, and
  since `neural_forge/__init__.py` imports it transitively, **every test in the
  repo currently fails to collect.** The file was clean at the start of this
  session and its diff against HEAD is ~200 lines of somebody's real in-progress
  work (a new `RUNTIME_TENSOR_ALIAS_MANIFEST_SCHEMA`, a
  `_lfm2_direct_local_state_alias_abi_from_entries` refactor), so I did **not**
  restore it from HEAD — that would have destroyed uncommitted work that is not
  mine. It needs its author. My own verification was run in a clean scratch copy
  of `src`/`tests`/`experiments` with that one file taken from HEAD (§7); the
  server deployment's copy is intact, which is why every server run in this
  wave loaded the parent without trouble.
* The route runner emits `SyntaxWarning: invalid escape sequence '\|'` from
  `experiments/render_matrix_ledger.py:117` on every render. Cosmetic, but it is
  a real deprecation and will be an error in a future Python.

---

## 6. Backups and the destructive-change check

Inventories taken before anything shared was touched, under
`/mnt/ai_data/neural-forge-matrix/backup/` (suffix `-20260810T152245Z`):
`phase-cache-inventory-before-mxc`, `sealed-bridge-evidence-inventory-before-mxc`,
`ledger.jsonl.before-mxc`, and
`deployment-before-mxc-20260810T152245Z/experiments/run_lfm2_matrix_cell.py`
(the pre-edit deployed runner, `1511325bce1662d8…`).

| check | result |
|---|---|
| phase-cache paths before / after | 88,930 → 130,597 |
| phase-cache paths **removed** | **0** |
| sealed bridge evidence (path + size) before / after | **identical** |
| `find -newermt` over `lfm2_tokenizer_bridge_evidence_20260717` | empty |
| `find -newermt` over `neural_forge_sparse_semantic_bridge_20260730` (sealed A9) | empty |
| `find -newermt` over `lfm2_complete_causal_evidence_20260720_a0003` | empty |
| ledger | append-only; 6 rows → 7, no row rewritten |

The 41,667 new phase-cache paths are the recomputed OMP supports and coefficient
chunks from the 0b route, published into the content-addressed cache under new
keys. `/` finished at 15 GB free.

---

## 7. Verification

```
# clean scratch tree (repo src/tests/experiments with the one corrupted file
# taken from HEAD -- see §5), python 3.11.9
python -m pytest tests/ -q -k matrix \
  --ignore=tests/test_hf_lfm2_direct_local.py \
  --ignore=tests/test_hf_lfm2_matrix_local_isolation_backend.py
# -> 257 passed, 1 skipped

python -m pytest tests/test_run_lfm2_matrix_cell_fresh_evidence.py -q
# -> 13 passed        (this wave's new tests, 9 of them refusals)

python -m pytest tests/test_run_lfm2_layer0_representability_study.py -q
# -> 4 passed        (the pattern this harness generalizes is not regressed)

python -m ruff check experiments/run_lfm2_matrix_cell.py \
                     tests/test_run_lfm2_matrix_cell_fresh_evidence.py
# -> 1 pre-existing error (unused CONTROL_CELL_ID import, present at a31747d8);
#    nothing new
```

The two ignored files are the concurrent author's own tests and fail only
because the scratch tree pins their module to HEAD.

Server runs, all under
`/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-dee1c952`:
the split-rule self-test, the 4-batch pilot capture, the three-site readability
and calibration-floor probe, the 24-batch capture, both 0b launches, and the
ledger render. Deployed runner re-verified byte-identical to the repo copy
(`f72f5511fb326fff…`) after the change.

**Not verified:** no body cell has run against the fresh capture. The evidence
loader, the split authority, and the C05 calibration arm are proven at L00/L05/
L13 on real sealed rows (§1.2), but no T5 solve, boundary score, composed
forward or free-run has executed on this evidence base — that is wave-1 cell 1,
and the gate is closed.

## 8. Paths

* fresh evidence: `/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810/evidence/` (`2dec87ac…/`, `evidence-manifest.json`)
* 0b route run: `…/neural_forge_matrix_campaign_20260810/CAL.C01.T5.X4-route-mxc2/` (first, seal-crashed launch preserved at `…-route-mxc/`)
* 0b receipt: `/mnt/ai_data/neural-forge-matrix/cells/CAL.C01.T5.X4-mxc/cell-receipt.json`
* knobs: `/mnt/ai_data/neural-forge-matrix/cal-c01-t5-x4-knobs-mxc2.json`, `…/c05-t5-m16-knobs.json`
* ledger + report: `/mnt/ai_data/neural-forge-matrix/ledger.jsonl`, `…/report/LEDGER.md`
* drivers: `/home/cereal/nf-mxc/{capture_matrix_evidence.py, probe_calibration_floor.py, compare_dossiers.py}` and their logs
* backups: `/mnt/ai_data/neural-forge-matrix/backup/`
