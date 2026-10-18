# `OC/ledger/` — campaign ledger, checkpoint registry, packages, dashboards

Pure standard library. No torch, no third-party imports, no ORM, no database.
Runs on the workstation (`C:/Python314/python.exe`) and on the AI server
(`python3` 3.12.3, or the lab venv) from the same bytes.

| file | what it is |
|---|---|
| `oc_seal.py` | canonical JSON + sha256 (byte-identical to the LFM2 campaign's), seal/verify/atomic write, directory digests, safetensors parameter counts without torch, the single-writer lock. **One enumerator** (`_digest_files`) feeds the digest, the parameter count and the per-tensor index, and it refuses every symlink, file or directory — a skipped symlinked `weights/` is how a checkpoint seals a digest and a count over bytes neither of them read, and how `clone` then certifies a byte-identical copy that is missing the weights. |
| `oc_ledger.py` | the append-only sealed ledger: `append`, `merge`, `verify`, `digest`, `rows` |
| `oc_checkpoint.py` | the checkpoint registry (`oc_checkpoint 1.1.0`, `--version`): `register` (incl. `--delta-base`), `clone`, `record-review`, `set-status`, `delete`, `list`, `show`, `verify`, `repair-row`, `delta-compose`, `package-delta` |
| `oc_package.py` | the review package for Cyril: `build`, `verify` |
| `oc_dashboard.py` | `SCORECARD.md` + `REVIEW_QUEUE.md`, regenerated from the registry |
| `test_oc_ledger.py` | 125 tests (104 before 1.1.0; 21 in `TestDelta`), `unittest`, no fixtures outside a temp directory |
| `smoke_oc.sh` | the end-to-end CLI smoke run on the server: builds throwaway safetensors fixtures under `ledger/.smoke-<date>/`, walks register (dry run and confirm) -> review -> package -> clone -> the R2/R4/R6 refusals -> dashboards -> both verifies -> a 1,056-row merge, and prints each step's exit code |

The rules these execute are in `../CHECKPOINT_POLICY.md`. Read that first.

## Layout it expects

```
$OC_ROOT/                       # /home/cereal/omni-campaign on the server
  ledger.jsonl                  # the authority (append-only)
  ledger.jsonl.lock             # taken by the one process appending
  registry/<checkpoint id>/<checkpoint id>-NNNN-<event>.json     # sealed records
  oc<NN>/children/OC<NN>-...    # the checkpoints themselves
  oc<NN>/human-review/<id>/     # packages
  SCORECARD.md  REVIEW_QUEUE.md
```

`--root` overrides, else `$OC_ROOT`, else `/home/cereal/omni-campaign`.
Every command takes explicit `--registry` / `--ledger` if you want them
elsewhere (the tests do).

## The lifecycle, in commands

```bash
PY=/home/cereal/model-surgery-lab/venv/bin/python      # or plain python3
L=/home/cereal/omni-campaign/ledger/oc_checkpoint.py
export OC_ROOT=/home/cereal/omni-campaign

# 1. a built fold child                       (dry run first: drop --confirm)
$PY $L register --checkpoint-id OC01-F-Q35-4B-C2S4L03 --stage fold \
    --directory $OC_ROOT/oc01/children/OC01-F-Q35-4B-C2S4L03 \
    --parent-directory /home/cereal/models/Qwen3.5-4B --parent-code Q35-4B \
    --parameter-count auto --parent-parameter-count auto \
    --recipe $OC_ROOT/run/oc01/mlpq.sh --recipe-command "mlpq.sh C2S4L03:S4:L03" \
    --component "mlp.layer3:70778880:17694720:model.language_model.layers.3.mlp." \
    --confirm
# (Qwen3.5-4B really does carry 70,778,880 parameters under
#  `model.language_model.layers.3.mlp.` across 3 tensors, read from the headers;
#  Qwen3.5-0.8B carries 11,010,048 under the same prefix. Check yours with
#  oc_seal.safetensors_tensor_counts + prefix_parameter_count before typing it.)

# A fold MUST name at least one --component: the >=4x rule is per component.
# With the trailing TENSOR_PREFIX both counts are re-derived from the
# safetensors headers and a disagreement is refused; without it they are an
# operator assertion and the record, the package and the scorecard say so.
# A fold may not claim a bigger saving than the model actually lost -- state a
# deliberate addition with --accounting-slack N.

# 2. review it (counts from the sealed verdict + the canonical re-read)
$PY $L record-review --checkpoint-id OC01-F-Q35-4B-C2S4L03 \
    --review-json review-C2S4L03.json --confirm

# 3. package it for Cyril
$PY $OC_ROOT/ledger/oc_package.py build --checkpoint-id OC01-F-Q35-4B-C2S4L03 \
    --out $OC_ROOT/oc01/human-review/OC01-F-Q35-4B-C2S4L03 \
    --dossier .../frozen-review-dossier.json --per-case .../per-case.json \
    --canonical-reread .../canonical-reread-oc01.json \
    --receipt .../cell-receipt.json --verdict .../cell-receipt-behavior-reviewed.json \
    --build-receipt .../oc01-build-receipt.json --bench .../bench.json \
    --media .../media/
# The re-read file may be flat ({lane: box}) or keyed by block. A keyed file is
# looked up under the checkpoint id, then the TAG, then the dossier's cell_id
# -- a wave-level file is keyed by CELL id (ARCH.C2S4L03), which is why the
# dossier matters. If none of the three matches, the build prints a WARNING and
# REVIEW_<TAG>.md marks the canonical column UNREAD. It never falls back to
# some other block in the file; a dash in that column means unmatched, and the
# document says which.

# 4. clone before ANY training step
$PY $L clone --source OC01-F-Q35-4B-C2S4L03 --stage-about-to-run D --confirm

# 5. the trained child names the clone as its parent
$PY $L register --checkpoint-id OC01-D-Q35-4B-C2S4L03 --stage distill \
    --parent-checkpoint-id OC01-F-Q35-4B-C2S4L03.clone-pre-D1 ... --confirm

# NOTE on ordering (this is now the ONLY order that works): a delete needs the
# review record to name a package whose MANIFEST.json verifies, is FOR this
# checkpoint (`checkpoint_id`, a `record_sha` in its chain) and whose
# `package_sha256` equals the one the review committed to. Since the digest
# only exists once the package is written:
#   1. record the review (no package fields);
#   2. build the package -- it seals the review record you just wrote;
#   3. record a SECOND review carrying package_path + package_sha256.
# Records are append-only, so (3) supersedes (1) without editing anything.
# `package_path` without `package_sha256` is refused at review time.

# 6. dashboards, then (only then) retention
$PY $OC_ROOT/ledger/oc_dashboard.py
$PY $L delete --checkpoint-id OC01-F-Q35-4B-DEAD1 \
    --reason "<a paragraph: what it measured, where the evidence lives>" \
    --rebuild-command "mlpq.sh DEAD1:S8:L03" \
    --retained-source OC01-F-Q35-4B-C2S4L03 --confirm

# 7. integrity, any time -- BOTH directions
$PY $OC_ROOT/ledger/oc_ledger.py verify --ledger $OC_ROOT/ledger.jsonl \
    --registry $OC_ROOT/registry          # rows  -> records
$PY $L verify                             # records -> rows, + BOTH parameter
                                          # counts re-read from the headers
$PY $L verify --deep                      # ... and every byte re-hashed
$PY $L repair-row --checkpoint-id <ID> --confirm   # a record whose row never landed
```

## Delta checkpoints (1.1.0, 2026-10-12)

Stage D registers DELTAS, not full copies (DISTILL_DESIGN_2026-10-12 section 4.1;
nine 55 GB copies fit on no disk here).  A delta directory holds the trained
tensors plus `DELTA.json` (`neural-forge-omni-delta-recipe-v1`):

```json
{"schema": "neural-forge-omni-delta-recipe-v1",
 "base_checkpoint_id": "OC02-F-Q38-27BU-OMNI1.clone-pre-D1",
 "base_directory_sha256": "<the base's sealed directory_sha256>",
 "entries": [
  {"op": "replace", "target": "blocks.0.ffn.up.weight", "tensor": "t0", "slice": [[0, 0, 4352]]},
  {"op": "lora", "target": "blocks.0.ffn.down.weight", "a": "A0", "b": "B0", "alpha": 64.0,
   "slice": [[1, 4352, 16640]]},
  {"op": "scale", "target": "blocks.0.ffn.down.weight", "tensor": "g0", "dim": 1,
   "slice": [[1, 4352, 16640]]}]}
```

Targets are the BASE's tensor names (Chimera names for OMNI1).  Applied per
target in fp32 -- each link in chain order, each entry in recipe order -- and cast
once to the base dtype.  `replace` adds no parameters; LoRA A/B and gains are
counted as added until the D4 merge folds them, so a delta's sealed
`parameter_count` is the COMPOSED model's (root headers + every link's adds).

```bash
$PY $L register --checkpoint-id OC02-D-Q38-27BU-OMNI1D1 --stage distill     --directory $DELTA --delta-base OC02-F-Q38-27BU-OMNI1.clone-pre-D1     --parent-directory /home/cereal/models/Qwen3.8-27B-Uncensored --parent-code Q38-27BU     --recipe-command "..." --confirm             # dry run first: drop --confirm
$PY $L delta-compose --checkpoint-id OC02-D-Q38-27BU-OMNI1D1 --deep --plan-out plan.json
$PY $L clone --source OC02-D-Q38-27BU-OMNI1D1 --stage-about-to-run D --confirm   # R2 as usual
$PY $L package-delta --checkpoint-id OC02-D-Q38-27BU-OMNI1D1 --out $PKG --confirm
```

Refused, each with a test: a recipe or `--delta-base-sha256` whose pin is not
the base's sealed digest (`BASE DIGEST MISMATCH`); a recipe naming another base;
a base that is not a real R2 clone; `--parent-checkpoint-id` other than the
base; `--delta-base` at stage fold; an op not in replace/lora/scale; a target the
base lacks; a slice out of bounds or on a missing dim; a replace of the wrong
shape or dtype; overlapping replace regions; a LoRA whose B @ A is not the
region, or whose alpha is not finite and positive; a gain of the wrong length;
a delta tensor used twice or by no entry; a typed `--parameter-count` that is
not the composed count.  `delta-compose --deep`, `verify --deep` and `clone`
re-check the pins (`--deep` re-hashes the bytes, which is the only check that
sees bytes moved under an unchanged seal); `verify` re-derives the composed
count and the recipe digest on every run; `delete` clause (iv-b) refuses the base
of any delta.  `package-delta` writes `delta/` (byte copy, digest asserted),
`RECORD.json`, `PLAN.json`, `README.txt` and the usual `MANIFEST.json`
(`oc_package.py verify --package` checks it); the base is named, never copied.
The P1 reader (`distill/tools/p1_qwen35_reader.py --plan plan.json`) composes
from the plan per tensor.  Since reader 1.0.4 it refuses to compose onto any
directory that does not re-digest to the plan's sealed `root.directory_sha256`
(a same-shape sibling has the same shard names and sizes, so those prove
nothing); `--no-verify-root-sha` skips the re-hash only for the plan's own root
directory, and the choice is recorded as `root_check` in every receipt.

## The review JSON

```json
{
  "reviewer": "claude-opus5-oc01",
  "dossier_sha256": "<64 hex>",
  "verdict_receipt_sha": "<64 hex>",
  "bank1": {"lanes_total": 16, "useful": 2, "partial": 3, "observed_miss": 10,
            "parent_useful_or_partial": 12, "sealed_verdict": "alive-degraded",
            "canonical_signal": 4, "canonical_artifact_miss": 6,
            "canonical_content_miss": 5},
  "columns": {"column_a": 5, "column_b": 4},
  "bank2": {"lanes_total": 21, "sealed_useful_or_partial": 3, "canonical_signal": 2},
  "media_out": [{"prompt_id": "av-gen-kettle-A", "non_empty": true, "cyril_score": null}],
  "smoke_verdict": "coherent", "safety": "safe-refusal/redirect",
  "degeneracy_reading": "none",
  "package_path": "/home/cereal/omni-campaign/oc01/human-review/OC01-F-Q35-4B-C2S4L03",
  "package_sha256": "<64 hex>",
  "notes": "..."
}
```

Refused if: the intelligence lanes do not sum to `lanes_total - 1` (the safety
lane never scores); the three canonical boxes do not sum to the same number;
`column_a` is not the sealed useful-or-partial count; `column_b > column_a`;
`parent_useful_or_partial` is missing (it comes from the parent's `bands.json`
and is never defaulted); a media score is not `null` or 1–5;
`package_path` is given without `package_sha256`. A missing count is a
`REFUSED: review is missing bank1.<x>` and exit 3, never a traceback.

**The status these counts imply**, and the reason artifact misses are counted at
all:

| signal (any surface) | content misses | status |
|---:|---:|---|
| ≥ floor (default 1) | anything | `partially_capable` → Cyril's queue |
| 0 | 0 (only artifacts) | `unmeasured` → **re-run; no path can delete it** |
| 0 | ≥ 1 | `bad` → eligible for R4 retention |

An empty media artifact counts as an artifact miss for the same reason an empty
generation does.

## Two numbers you cannot type, and three paths you cannot touch

`net_ratio` is the whole claim, so it is not accepted as an assertion. If the
directory holds `*.safetensors`, both parameter counts come off the headers and
an explicit `--parameter-count` / `--parent-parameter-count` that disagrees is
refused. A directory with nothing to read them from needs
`--allow-asserted-parameter-counts`, and then the record, the ledger row,
SCORECARD.md's `ratio from` column and `REVIEW_<TAG>.md` all say
`asserted-by-operator` instead of `derived-from-safetensors-headers`.
`oc_checkpoint.py verify` re-reads the headers on every run (0.00 s on a real
4B); `--deep` additionally re-hashes every byte (~169 MB/s, so ~5.6 min for a
27B).

`PROTECTED_PREFIXES = ("/mnt", "/home/cereal/models")`, matched by path PARTS
of the **resolved** path (`resolve_path` = `expanduser` + `realpath` +
`normpath`), so a spelling cannot walk past the guard:

```bash
$PY $L register --directory /mnt/ai_data/models/whatever ...   # REFUSED, exit 3
$PY $L register --directory $OC_ROOT/../models/Qwen3.5-4B ...  # REFUSED, exit 3
$PY $L register --directory $OC_ROOT/link-to-the-share/x ...   # REFUSED, exit 3
$PY $L clone    --dest      /home/cereal/models/clone-x ...     # REFUSED, exit 3
$PY $L register --stage parent --directory /home/cereal/models/Qwen3.5-4B      --parent-provenance-sha256 <digest> ...                    # allowed
```

`.absolute()` only prepends the working directory: it leaves `..` in the parts
and never looks through a symlink, so both spellings above used to read as
ordinary campaign paths. Resolution costs a bounded `lstat` per component (~2 s
for a path under the share, measured) and never lists a directory, so it is
safe to run before the guard that keeps the tooling off the share.

The resolved path is also what gets SEALED (`--directory`, `--parent-directory`,
the clone destination): a record names a location, not the spelling someone
typed to reach it, and R4 clause (iv-c) re-reads that field immediately before
`rmtree`. `_assert_directory_is_free` resolves both sides too, so `children/X`,
`children/../children/X` and `alias/X` are one directory and cannot be
registered as two live ids over one set of weights.

The refusal in `register` happens **before** the directory is walked, which is
the point: hashing the CIFS share recursively at under 3 MB/s is the failure,
not just deleting from it. R4 clause (iv-c) repeats the check immediately before
`rmtree`, for records sealed before the guard existed.

## Tests

```bash
python3 test_oc_ledger.py -v            # 101 tests, ~3 s, no network, no GPU
```

They are mostly refusal tests: a merge that cannot be disambiguated, a status
the counts do not imply, a clone whose source bytes moved, a delete that would
take a frontier holder or a clone source with it. A refusal that is not tested
is a refusal that will be discovered the wave after it silently stopped
happening.

Each of these reproduces a bypass an adversarial review found, and fails if the
guard is removed:

| test | what it would let happen |
|---|---|
| `test_all_artifact_misses_is_unmeasured_and_never_bad` | 15 loops graded `bad` → `rm -rf` authorised by artifacts |
| `test_delete_refuses_an_artifact_only_child` | the same, through the R4 path |
| `test_delete_refuses_another_checkpoints_package` | DEAD1 deleted against GOOD1's package |
| `test_delete_refuses_a_package_that_changed_after_the_review` | the packaged evidence edited after the review |
| `test_an_interrupted_delete_does_not_poison_the_chain` | one marker makes the whole registry unreadable |
| `test_a_clone_may_not_be_registered_by_hand` | R2 satisfied by a hand-written record, no bytes copied |
| `test_a_trained_child_needs_a_real_clone_receipt` | a training parent whose clone receipt is gone |
| `test_a_fold_must_name_a_component` / `_that_stored_everything_` | a "fold" at net_ratio 1.0 on the scorecard |
| `test_components_may_not_claim_more_than_the_model_lost` | a saving claimed that the bytes do not show |
| `test_component_counts_are_checked_against_the_headers` | component counts that no file agrees with |
| `test_merge_twice_is_a_no_op` | a duplicated row that poisons `verify` for good |
| `test_verify_sees_a_record_whose_row_never_landed` | a checkpoint missing from the ledger, both tools clean |
| `test_two_checkpoints_may_not_share_one_directory` | a DROP that deletes another checkpoint's weights |
| `test_an_id_with_no_tag_is_refused` | `OC01-F-Q35-4B` registered as parent `Q35` tag `4B` |
| `test_a_truncated_safetensors_is_refused` | a confident parameter count off a half-written shard |
| `test_two_writers_cannot_both_take_the_next_sequence` | `record sequence [1,2,2]`, unrecoverable |
| `test_building_into_a_used_folder_is_refused` | a stale file joining a new manifest, changing its digest |
| `test_merge_apply_appends_and_never_rewrites_the_authority` | `write_bytes` truncates first: a kill mid-merge loses all 1,056 rows |
| `test_merge_holds_no_truncating_write` | the same, reintroduced by a later "tidy-up" |
| `test_a_typed_parameter_count_the_headers_contradict_is_refused` | a typed 16x sealed into the record, the row, the scorecard and the package |
| `test_verify_rereads_the_headers_and_catches_a_wrong_count` | `verify --deep` re-dividing the same two sealed integers and calling it verified |
| `test_register_refuses_a_protected_directory` | one `--directory /mnt/ai_data/...` typo byte-hashing the CIFS share |
| `test_the_protected_path_clause_is_evaluated_on_every_delete` | `rmtree` on the share or the source model library |
| `test_clone_refuses_a_protected_destination` | a full-size clone dropped into the model library |
| `test_a_dotdot_spelling_walks_past_no_guard` | `--directory $OC_ROOT/../models/...` reading as a campaign path |
| `test_a_symlink_into_a_protected_root_is_refused` | a link whose target is the share or the model library |
| `test_the_record_seals_the_resolved_directory` | a `..` spelling sealed into the record `delete` re-reads |
| `test_a_dotdot_spelling_is_not_a_second_directory` | two live ids over one set of weights, no symlink needed |
| `test_a_symlink_alias_is_not_a_second_directory` | the same, through an alias link |
| `test_clone_refuses_a_destination_that_resolves_into_a_protected_root` | a clone landing in the model library through `..` or a link |
| `test_a_silent_wav_is_not_a_signal_lane` | a track of pure silence counted as intelligence |
| `test_a_bank_with_no_intelligence_lanes_is_refused` | `lanes_total: 1` labelling an unmeasured child `bad` |
| `test_a_cell_row_written_here_must_name_this_campaign` | a cell row of this campaign indistinguishable from an ancestor row |
| `test_a_row_with_no_receipt_is_a_refusal_not_a_keyerror` | a KeyError traceback out of `verify` instead of `REFUSED:` |
| `test_a_missing_input_is_a_refusal_not_a_traceback` | a mistyped `--dossier` crashing a half-written package |
