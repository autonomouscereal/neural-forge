# Chimera import lane — S3 blocker fixes, 2026-09-02

Branch `claude/omni-import`, commit `omni-import S3 fix: ...` on top of `9ef3bc1`.
Companion to `S3_LEDGER_TOKENS_2026-09-01.md`; the review of that stage raised three
blockers, all fixed in place and measured on the AI server.

Everything below was run in `/home/cereal/model-surgery-lab/venv` (torch 2.11.0+cu128,
Python 3.12.3) against the real Qwen3.5-0.8B / Qwen3.5-4B / Qwen3.8-27B-Uncensored /
Kimi-K3 assets.

## Blocker 1 — a scaled rope could walk through the scaled-rope refusal

`profiles.py::_qwen_rope` backfilled `rope_theta` and `partial_rotary_factor` from the
flat `text_config` keys when there was no nested `rope_parameters` table, but read
`rope_type` **only** from that nested table. In the pre-5.x layout the fallback exists
to serve — flat theta and rotary factor, with the scaled variant declared in its own
`rope_scaling` table — the guard therefore saw `rope_type = "default"` and imported a
yarn rope as a plain `positions * inv_freq` table. `test_qwen_profile_refuses_a_scaled_rope`
only ever exercised the nested path.

Measured on the real 0.8B config, with its rope flattened into the old layout:

```
pre-fix rope_type seen by the guard: default -> scaled rope accepted
post-fix: ValueError unsupported rope_type 'yarn': Chimera's gated-GQA block builds the plain positions*inv_freq table, so a scaled...
post-fix both-layouts: ValueError config carries both rope_parameters and a non-empty rope_scaling table; the builder reads rope_parameters and ...
```

Fix: `rope_type` now comes out of the same layout the theta came from (flat
`rope_type`, else `rope_scaling["rope_type"]`, else the legacy `rope_scaling["type"]`),
and a config that carries **both** layouts is refused rather than having half of it
silently ignored. No checkpoint on the box carries both — verified, all three declare
`text_config.rope_parameters` and no `rope_scaling` at either level — so the new
refusal costs nothing on the real assets.

New tests: `test_qwen_profile_refuses_a_scaled_rope_declared_in_the_flat_layout`
(yarn / legacy `type` / dynamic, plus an explicitly unscaled table that still builds),
`test_qwen_profile_refuses_a_rope_scaling_table_it_would_never_read`.

## Blocker 2 — control ids were bridged by the byte spelling of their own name

`JsonBpeTokenizer.token_bytes` decoded an added token through the GPT-2 byte alphabet.
For an **ASCII** marker name every character is in that alphabet, so the decode
succeeded and returned the name as if it were text; `token_alignment` then re-encoded
that name in the destination vocabulary and counted the row as ALIGNED, and
`embedding_bridge` wrote it as the mean of those destination rows. The same held for
K3's 256 reserved ids, whose tiktoken bytes are the literal name.

Measured on `/home/cereal/models/Qwen3.5-0.8B` and the K3 checkpoint:

```
pre-fix  token_bytes(248053) = b'<|vision_start|>'
post-fix token_bytes(248053) = b''  special=True
pre-fix  destination pieces for that name: [27, 91, 7386, 9485, 91, 29]
pre-fix  k3 token_bytes(163584) = b'[BOS]'
post-fix k3 token_bytes(163584) = b''
ids 248040..248069  pre-fix  aligned=30 unmapped=0
ids 248040..248069  post-fix aligned=4 unmapped=26
corpus with markers: aligned=3 unmapped=2 special=2
marker ids in pairs: [248045] -> [False]
```

(The `except KeyError` fallback the review named was a red herring: it only fires for
non-ASCII markers. The load-bearing fix is the declared id set.)

Fix: `BaseTokenizer` gained `special_ids` / `is_special_id()`, and `token_bytes`
returns `b""` for them. `ChimeraTokenizer.from_qwen` reads the ids from the
`added_tokens` table in `tokenizer.json` (the file is the authority; the `tokenizers`
binding's `get_added_tokens_decoder()` is only a fallback), and `from_k3` declares the
256 reserved ids above the merge ranks. `token_alignment` now reports a
`special_tokens` count next to `unmapped_tokens` instead of absorbing those rows into
`aligned_tokens`; `embedding_bridge` leaves their rows at zero, which is what its
docstring already promised.

This is the defect that would have corrupted the `full_vocab=True` sweep the S3
closeout still lists as owed: ~26 Qwen control rows and 256 K3 reserved ids.

New tests: `test_an_added_token_piece_never_reports_the_bytes_of_its_own_name`,
`test_a_control_id_is_counted_unmapped_and_its_bridged_row_stays_zero`,
`test_qwen_control_ids_are_reported_unmapped_rather_than_bridged` (real asset),
`test_k3_reserved_ids_carry_no_bytes_only_a_name` (real asset).

## Blocker 3 — the segment packet declared a width the patchifier never emits

`build_native_segment_packet` ignored `spec.feature_dim` and kept a hard `3072`
default. Every pre-volume modality is 3,072 wide so nothing had caught it, but the
volume spec is 4,096:

```
pre-fix  volume packet feature_dim = 3072 (hard default)
post-fix volume packet feature_dim = 4096, patchifier emits 4096, segments 27/27
other modalities unchanged: {'image': 3072, 'video': 3072, 'audio': 3072, 'music': 3072, 'tts': 3072, 'ocr': 3072, 'volume': 4096}
```

Fix: the default is `None`, resolved to `spec.feature_dim`; an explicit argument still
wins. Behaviour-identical for the six 3,072 modalities.

New tests: `test_the_segment_packet_declares_the_width_its_modality_actually_emits`,
`test_the_volume_packet_width_matches_the_patchifier`.

## Also in this commit

- `QWEN35_FAMILY_DEFAULTS` — the *assumed* half of the 27B planning profile — now has
  its own on-disk anti-decay test across all three Qwen checkpoints
  (`test_the_qwen3_5_family_defaults_still_match_every_checkpoint`), reading rope
  fields where transformers reads them. Previously only the verified half
  (`QWEN38_27B_VERIFIED_GEOMETRY`) was checked against disk.
- The duplicated `"volume"` branch in `canonical_positions` is folded into the
  character-identical `video`/`ocr` branch, so the two cannot drift apart.
- Both profiles stamp `source_config_scope` next to `source_config_sha256`, because
  the Qwen digest covers the whole `config.json` and the K3 digest covers `text_config`
  only (or the transcribed constant when no file is given).

## Verification

```
290 passed in 19.20s     # full suite, 0 skipped, every asset reachable
 96 passed in 7.19s      # the seven S3 test files, verbose
 12 passed in 3.65s      # the ten new blocker/anti-decay tests
```

Logs: `pytest_s3fix_full.log`, `pytest_s3fix_s3_files.log`,
`pytest_s3fix_blocker_tests.log`. The before/after measurement script and its output
are `probe_blockers_s3fix.py` / `probe_blockers_s3fix.log`.

Receipt `receipt_s3.json` regenerated from the three checkpoints' own `config.json`
(sha256 `918ab9b018f320dd9b3a9322656362264e03c9d56da9e9fd573cfd094e28cff4`, 8,701
bytes). The previous receipt is kept as `receipt_s3_before_blocker_fix.json` (sha256
`dea908595fd0c13d2ff59b54b7a5950e9d2e77df7bbaa4bc07cfbc55fcdaf83b`). A field-by-field
diff of the two receipts is **four added keys and the label** — three
`source_config_scope` stamps and `alignment.special_tokens = 0` — and nothing else:
every ledger, mask, volume, typed-flow, tokenizer and bridge number is unchanged, so
the fixes did not move the measured path.

`special_tokens = 0` there is correct: the four fixture texts contain no markers. The
non-zero case is in the probe log above (`special=2` on a corpus with chat markers).

## Still open after this commit

- Spec Q1 (27B `output_gate_type: "swish"` vs transformers 5.12.1's unconditional
  sigmoid) is untouched and still blocks any 27B exactness claim.
- `token_alignment(full_vocab=True)` over all 248,320 Qwen rows has still never been
  run; it is now safe to run, which it was not before this commit.
- The Qwen tokenizer normalizes NFC, so `decode(encode(t)) != t` for a combining-mark
  document. The receipt's `round_trip_exact: true` is true for the four fixture texts
  only; it is faithful to Qwen (the ids are what Qwen itself emits) but it is a
  fixture-selected boolean, not a guarantee. Left as-is; recorded here.
- `embedding_bridge`'s `source_row_rms` reads a table the receipt passes as zeros, so
  that statistic can never mean anything. Left as-is.
- Nothing in S3 loads a weight: profile exactness is still config-level only.
