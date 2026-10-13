# Future folding recipe integration audit, 2026-10-13

Recovery of the existing registered model remains first. This is a source-only
audit of the actual 2026 campaign mirror and two private source correction
candidates. No fold, refold, weight loading, replacement checkpoint or training
launch occurred. Local source findings require exact remote source hash
agreement before they describe the live builder.

## Concrete receipt and CLI defects

`receipt_q27_streamed.py:906–944` computes parity maxima without requiring
finite base logits or hidden taps. NaN arithmetic can leave aggregate maxima
at zero because Python's `max(0.0, nan)` retains the first argument. Its band
counter counts only finite values, so a NaN or positive-infinity band entry can
look like the intended negative-infinity mask. At918–929 it silently skips a
missing requested tap, allowing incomplete hidden measurements to appear to
have zero worst error. These defects affect the measurement receipt; they are
not evidence that a particular model output actually contained NaNs or that
an existing checkpoint was built incorrectly.

`folding_recipe_candidates.py receipt` adds early prompt/shape/count checks,
exact requested tap coverage, finite base logits/hidden taps, and an exact
`-Inf` check on the import-only interface band before maxima or argmax. It
preserves the comparison arithmetic and resource policy for valid inputs. It
does not reinterpret import parity as a task score or impose a new numerical
acceptance tolerance.

`s7_unit_build.py:716–717` documents HY3D `--parent` as an optional time-token
reference source. Dispatch735–736 ignores it, while `build_hy3d` at520/546 uses
the block source for its time token and optional reference comparison. The
`unit` candidate passes an explicit `time_reference_parent`; the block's
`--src` tensors remain the source of the block, while the time reference uses
the requested parent, checks width, and records its full file SHA256. When
`--parent` is absent behavior is preserved. Review this declared two-source
contract before generating a new unit; existing units are unaffected.

Pinned original byte hashes:

| Source | SHA256 |
|---|---|
| `composite/tools/receipt_q27_streamed.py` | `c7fb1b0245d556c3af304ca6aeb9d78d9c294d39c682bcf93576e43488017257` |
| `composite/tools/s7_unit_build.py` | `36df14de22b363e665d130fc5d6619ceb40e5c57ba98ec4c5ac22cfd72624c49` |

The generator refuses a changed source baseline or an existing output file.
It writes a separate candidate, unified patch and receipt; it never applies a
patch to the campaign source or executes the recipe.

```powershell
C:/Python314/python.exe -B -m unittest discover -s tests -p test_chimera_folding_recipe_candidates.py
C:/Python314/python.exe -B experiments/chimera_recovery/folding_recipe_candidates.py receipt --source 'C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/composite/tools/receipt_q27_streamed.py' --out 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_folding_recipe_candidates_oct13/receipt_q27_streamed_candidate.py'
C:/Python314/python.exe -B experiments/chimera_recovery/folding_recipe_candidates.py unit --source 'C:/Users/cereal/Documents/ai training/neural-forge/reports/omni-campaign-2026-09/composite/tools/s7_unit_build.py' --out 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_folding_recipe_candidates_oct13/s7_unit_build_candidate.py'
```

Seven focused tests include a mocked CLI dispatch that proves the requested
HY3D parent reaches the time-reference argument, and five CPU Torch regressions
for valid complete parity, missing taps, non-finite observations, malformed
band masks and broadcast/missing-prompt geometry. Set
`CHIMERA_H3_CAMPAIGN_SOURCE=/home/cereal/omni-campaign` on the supported server
CPU runtime to bind the source mirror. Those tests do not invoke a builder or
load any model weights.

## Integration limits to preserve as explicit future recipe work

`qf_stream27.py:1316–1336` implements the C2 MLP intermediate-channel coreset
fold only. It selects gate/up rows and refits a full hidden-width down writer
(1017/883–889). Other layer tensors are loaded unchanged (281–302). It has no
complete-layer or output-owner partition recipe. The standard global
`intermediate_size` update requires uniform coverage of every layer
(1122–1155); partial-site runs remain diagnostic. A future complete-layer or
owner-map design needs an explicit compiler/config/evaluator contract rather
than another flag on this existing C2 path.

The parent/staged-checkpoint checks at124–200 are substantial: normalized
config equality, full checkpoint/index hashes and safe shard paths. They are
rechecked before child assembly and after export at1439/1446. The entry's
builder record at1500 adds a name rather than an executable source manifest
for itself and imported `qf_*` recipe modules. Future recipe identity should
pin those actual module bytes, solve flags, schedule, ownership map and
calibration/holdout source groups along with the already pinned weight source.

The Gram branch scores the fp64 writer at1031–1046 before installation narrows
it to model dtype at887. Preserve that solve diagnostic, then measure actual
installed outputs separately. Streamed causal-mask evidence remains `None`
at1383; resident comparison results are recorded without a veto. Add explicit
mask dispatch and post-cast installed-output tests for new recipes. The
existing streamed import receipt requires matching width, depth, vocabulary
and layer-kind schedule at269; a dimension-changing full-layer fold needs a
different evaluator contract. Resource-success exit at1247 is not task
acceptance.

Unit conversion also has a narrower meaning than full parent preservation.
The four-times component check at `s7_unit_build.py:97` applies to MLP parameter
counts; whole-block ratio is recorded only. QIMG parent receipt acceptance and
time-pass flags at347–368 are copied into evidence rather than vetoing unit
conversion. Missing host-width charts are expected intermediate containers.
QIMG drops the joint text partner and 3D RoPE at384; HY3D drops DINO
cross-attention while preserving its time-row contract at504/570; QVL drops
2D RoPE and learned position embeddings at698. Those are declared structural
choices and should be compared on actual content/conditioning tasks after
recovery, without claiming that a finite installed unit preserves them.

`c16_install_media.py:823–886` intentionally treats time-reference numerical
mismatch as diagnostic under the recorded 2026-10-06 owner policy; strict
mode is available. Preserve the warning and exact reference hashes when
reviewing old units. Installer completion, compression accounting, import
parity, frozen task metrics and native decode/content review remain separate
evidence. None establishes the others.

The review-hook ordering at `qf_stream27.py:1473/1508` remains unresolved: the
hook receives the receipt path before this entry writes the receipt. Its
delegated helper must be read before asserting a failure or proposing a
change. No defect is claimed for that ordering from this audit alone.
