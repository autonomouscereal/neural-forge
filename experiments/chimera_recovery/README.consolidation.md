# D4 preparation, 2026-10-13

`consolidation.py` produces a pinned preparation receipt and checks a supplied replay/optimizer/state/merge job. It does not implement the missing joint D4 engine or sequential checkpoint importer. Every result retains `production_ready: false`; a valid descriptor is not a native parity, full-integrity cache or registered-child receipt.

The frozen campaign rules remain 2.5M short Q38 tokens, 0.5M long Q38 tokens, 1M K3 tokens, and 300 native input/target pairs in each of vision, ACE, QIMG, HY3D and H3. Every prior D2/D3 base learning rate is multiplied by 0.3. Whole rows cannot silently replace those nominal token budgets: the selector records any overshoot and a production implementation must bind a dated budget amendment. No repetition or padding fills a shortfall.

## Commands

Run from the repository root:

```text
python -B -m experiments.chimera_recovery.consolidation --preregistration <campaign.md> --inventory <inventory.json> --text-jobs <text-jobs-manifest.json> --admission <media-admission-manifest.json> --stage-source <train_stages.py> --stage-source <other-actual-source.py> --native-manifest qimg=<native-qimg-manifest.json> --native-manifest ace=<native-ace-manifest.json> --progress <captured-progress.json> --modality-jobs <modality-jobs.json> --out <new-preparation-directory>
python -B -m experiments.chimera_recovery.consolidation --validate-job <pinned-validation-plan.json>
python -B -m pytest -q tests/test_chimera_recovery_consolidation.py
```

Preparation preserves prior outputs. `--supersedes <prior-report.json>` pins an older receipt when a new report corrects or updates its inventory.

## Validation plan contract

Use schema `chimera_recovery.d4_validation_plan.v1`. `sources` must contain exactly these file references, each with `path` and the actual lowercase `sha256`: `preregistration`, `replay_catalog`, `optimizer_groups`, `stage_states`, `merge_operations`, `tensor_inventory`, `owner_regions`, `initial_shared_bank`. The command rehashes every file and separately rehashes the actual checkpoint/shared-bank files referenced by each stage state. Paths are relative to the containing descriptor or absolute. Only UTF-8 JSON/JSONL descriptors are interpreted; state files are hashed, never executed or loaded into a model.

The replay catalog contains stable row IDs, lane, train split, `held: false`, `recovery_group_id`, sealed target/corpus/teacher/shard pins and actual `real_tokens` for text. Media `pair_id` is the canonical JSON SHA of `input_sha256`, `target_sha256` and `conditioning_sha256`, each binding actual native tensor content. Different sigma samples can be distinct pairs when those genuine inputs/targets differ; a new row ID cannot conceal the same pair. A target hash equal to its input is refused. `heldout_groups` is checked throughout the catalog, including unused suffixes. A production cache preflight must additionally prove target provenance, noise/conditioning semantics and every referenced tensor's bytes; catalog labels do not supply that proof.

Optimizer groups contain a declared physical `parameter_id`/`numel` census, `family`, `previous_base_lr` and `lr`. The validator requires each of the nine D2/D3 families exactly once per physical parameter and checks base LR × 0.3, rather than multiplying an already decayed terminal rate. Runtime tensor identities and aliases still need direct verification.

Stage states follow D2, aggregate D3a, D3b ACE/QIMG/HY3D/H3, D3c, D3d in that order. Each has `stage_role`, `state_file`, `state_sha256`, `parent_state_sha256`, and explicit `semantics`. D3 stages carry `shared_lora_parent_sha256`, `shared_lora_state_sha256` and `shared_lora_semantics`; full replacement also supplies `shared_lora_file`. D3a carries the unchanged initial bank. D2's r64 bank cannot substitute for the D3 r16 bank. Independently initialized sibling banks or a sum of complete A/B states fail this contract. These fields require an actual importer/compose receipt before they can be populated for production.

`tensor_inventory` contains actual tensor `shapes`, canonical `aliases` and `protected_vocab_targets`. `owner_regions` maps every optimizer family to actual native target/axis/start/stop regions. Operations specify kind, family `bank`, target region, `compute_dtype: float32`, `cast: defer_until_target_complete`, and a unique site-specific `adapter_state_id` for LoRA. A complete plan must cover exactly the declared owned regions, including vectors and higher-dimensional unit tensors. Protected embed/head operations write only active music rows [252416,316416), preserving the text vocabulary, ledger and reserved 1536 rows. Native replacements cannot overwrite an earlier merge write. K3 down-column LoRA precedes one gain operation on columns [4352,16640); a later LoRA or native write is rejected. This checks operation contracts, not the numerical A/B/scatter/scale implementation.

## Observed availability and gaps

The 2026-10-13 17:32:39Z snapshot and current v3 admission establish the following preparation inventory:

| Lane | Pinned availability | Remaining evidence |
| --- | --- | --- |
| Short Q38 | 10,778,424 prepared input tokens | Fresh sealed targets and replay row catalog |
| Recovery long | 1,474,677 input tokens, maximum train context 21,701 | These are not the original 32k rung; bind intended long-context targets |
| Original 32k | 1,000,609 reported corpus tokens | Current admitted row groups and full target integrity |
| K3 | Preregistered 1,991,482 Qwen tokens / 1,867,899 aligned pairs | Actual sealed replay catalog and unchanged byte alignment |
| QIMG | 1,126 sealed-manifest batches / 1,121 admitted groups | Native tensor traversal and exact 300-pair selection |
| ACE | 2,764 sealed-manifest batches / 691 admitted groups | Native tensor traversal and exact 300-pair selection |
| Vision | 2,294 source-admitted items | ViT features alone do not provide aligned assistant-answer KD |
| ACE LM | 838 observed teacher shards, unsealed | Full traversal, final held bits, native code conditioning |
| HY3D | 146 observed teacher shards, unsealed; 256 original source items | Authentic geometry/conditioning adapter and actual sealed pair inventory |
| H3 | 512 original unrelated AV pairs excluded; 8 new train source items | Synchronized joint teacher targets and actual native pair inventory |

The original/recovery/source counts above are not teacher-cache seals. Source item and connected-group counts are not native pair counts. Multiple genuine sigma targets may expand a source item into several distinct input/target pairs, so 256 HY3D items do not alone prove a 300-pair shortfall. The prepared text plan contains 21,373 train rows and 9,634,142 assistant-supervised tokens; this does not establish model performance.

Current stage runners build a fresh hybrid and r16 bank from the immutable configured base. Same-stage exact resume does not import earlier stages. D3b exports unit additive deltas, emission scatter deltas and a complete LoRA bank; D3a/c/d output full learned state checkpoints with compose-required receipts. A sequential importer and union forward/backward/optimizer/exact-resume driver remain necessary. Preserve every current registered parent and clone while implementing and testing that handoff. Numerical final-merge parity and the registered D4 child/pre-T clone are subsequent runtime evidence, not preparation artifacts.
