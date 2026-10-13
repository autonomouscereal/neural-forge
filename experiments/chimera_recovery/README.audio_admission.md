# Global audio source admission, 2026-10-13

`audio_admission.py` seals source eligibility against the complete normalized
registry, pinned frozen evaluation, prior media admission and its cross-route
exclusions. It preserves every original source row and family split, verifies
actual PCM16 and groundtruth bytes against the original audio preflight, and
propagates held/evaluation exclusions through connected source, family, asset
and full-task identities. Generic observation instructions alone do not merge
different waveforms. Native H3 encoding, chart amendment, batch and driver
validation remain independent requirements; the output always says
`native_train_ready:false`.

The actual final procedural run traversed 91,984 normalized rows, 1,948 frozen
evaluation rows, 17,203 prior media entries, 21,081 cross-route exclusions and
all 144 procedural source rows. It reverified all 216 generated assets. Its
24 audio rows preserve eight train, eight validation and eight test rows;
only the eight train rows passed source admission. This small tranche covers
authored non-speech tones, chirps and noise bursts, not speech recognition.

The sealed manifest uses `recovery_audio_admission.v1`. Required verified flags
are `complete_source_traversal`, `frozen_exclusions_verified`,
`connected_groups_verified` and `sealed`. Every flag is backed by actual
source hashes and traversal counts. `outputs.admission` and
`outputs.train_admitted` pin JSONL hashes and counts; the manifest has its own
canonical `manifest_sha256` excluding that field. A packer also pins the
manifest file hash. `train_admitted` and `source_train_eligible` describe source
eligibility only. Their rows require split `train` and no blockers.

Row `id` and `item_id` are identical. `source_row_sha256` hashes the unchanged
whole source row. `authored_answer_sha256` hashes the JSON encoding of its
assistant content string, using sorted-key compact UTF-8 JSON; the separate
`groundtruth_answer_sha256` hashes its parsed groundtruth object and matches
the original preflight. `asset_sha256` identifies the original waveform,
with `asset_path` retained. Connected grouping is explicit through
`recovery_group_id`, `source_group_id` and `connected_source_groups`.

Executable preparation from the root workspace uses an exact JSON map
`{name:{path,sha256}}` for `source`, `source_inventory`, `source_report`,
`source_preflight`, `normalized`, `corpus_manifest`, `eval_rows`,
`frozen_exclusions` and `prior_admission`:

```powershell
python -m experiments.chimera_recovery.audio_admission `
  --pins 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_audio_admission_pins_oct13_v1.json' `
  --asset-root 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_media_seed_oct13_v2/assets' `
  --output 'C:/Users/cereal/Documents/ai training/proof_outputs/chimera_audio_source_admission_oct13_v2'
```

The prior manifest and all referenced row files, original registry files and
corpus/evaluation source pins must agree. New output directories preserve old
receipts. Tests reject byte tampering, incomplete traversal, resealed wrong
row counts, same-family/different-asset holdouts, cross-route held assets,
frozen spoken targets and nonminimal/nonzero/source-altering padding.

`validate_tail_derivation` independently verifies the real H3 speech input
derivation: all original PCM survives, the tail consists solely of the minimal
number of zero samples needed for a 25ms boundary, and both file hashes and
recorded counts match. The measured diagnostic preserved 99,840 mono PCM16
samples at24kHz (4.16s), appended exactly360 zero samples and encoded the
100,200-sample derived file (4.175s). The actual H3 proof produced167 latent
frames with no internal pad/trim. This derivation audit grants no production
or tensor admission.
