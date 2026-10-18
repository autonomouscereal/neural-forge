# Chimera import lane — S3 (ledger and tokens) closeout, 2026-09-01

Branch `claude/omni-import` in `C:/Users/cereal/Documents/ai training/chimera-omni`.
Spec: `../r4-chimera-import-lane/CHIMERA_IMPORT_LANE_SPEC_2026-09-01.md` sections 1.8-1.11.

> **Superseded in part by `S3_BLOCKER_FIXES_2026-09-02.md`.** Review of this stage
> found three defects in the code described below (a scaled rope that could pass the
> scaled-rope refusal through the flat layout, control ids bridged by the byte
> spelling of their own name, and a segment packet declaring 3,072 for a 4,096-wide
> volume). They are fixed on the same branch; the suite is now 290 passed and the
> receipt sha256 is `918ab9b0...`, not the `dea90859...` quoted below (kept as
> `receipt_s3_before_blocker_fix.json`).

## What S3 delivers

| Spec item | Where |
|---|---|
| 1.8 fixed-band ledger | `src/chimera_omni/ledger.py` (`DEFAULT_FIXED_BANDS`, `build_fixed_band_ledger`, `normalize_bands`), `config.py` (`ledger_kind` / `ledger_base_vocab_size` / `ledger_bands`, `ledger()`, `ledger_band_begin()`), `architecture.py` (manifest uses `cfg.ledger()`), `modeling/chimera.py` (`ledger_band_logit_mask=False`) |
| 1.9 tokenizer + bridge | `src/chimera_omni/importing/tokenizer.py`, `importing/bridge.py` |
| 1.10 native volume type id 7 | `native_media.py`, `native_segments.py`, `media/training.py` (`FLOW_TYPE_SHIFTS`, `typed_flow_noise_levels`), `eval/gates.py` |
| 1.11 trunk feature width | `src/chimera_omni/importing/profiles.py` (`TRUNK_FEATURE_DIM = 6144`, `qwen35_chimera_config`, `k3_chimera_config`, `chimera_omni_27b_profile`) |

Every new config field defaults to current behaviour: `ledger_kind="proportional"`,
`ledger_band_logit_mask=False`, `native_feature_dim=3072` (only the profiles move it
to 6,144), `REQUIRED_MODALITIES` unchanged (volume joins `MEDIA_` /
`GENERATIVE_MEDIA_MODALITIES` only).

## The defect this closeout fixed

`qwen35_chimera_config()` read `rope_theta` and `partial_rotary_factor` as flat
`text_config` keys. **No Qwen checkpoint on the server declares them that way.**
All three carry a nested `text_config["rope_parameters"]`, and the 0.8B and 4B have
no flat `partial_rotary_factor` at all:

```
FAIL /home/cereal/models/Qwen3.5-0.8B/config.json  ValueError config is missing the required field 'partial_rotary_factor'
FAIL /home/cereal/models/Qwen3.5-4B/config.json    ValueError config is missing the required field 'partial_rotary_factor'
```

The builder that is supposed to select every switch for an exact import could not
read a single real checkpoint. It passed its unit tests because those tests build
from a transcribed dict.

Fixed in `profiles.py::_qwen_rope`, which reads the nested table first (flat keys
stay as a fallback for planning dicts) exactly where transformers 5.12.1 reads it —
`modeling_qwen3_5.py:105` (`rope_type`), `:134` (`rope_theta`), `:135`
(`partial_rotary_factor`), `:137` (`dim = int(head_dim * partial_rotary_factor)`,
truncation, not rounding). Also added: refusal of a non-`default` `rope_type`, of a
routed (`num_experts`) or non-`silu` Qwen trunk, mRoPE/`rope_type` metadata stamps,
and a `dropped_source_mtp_layers` stamp for the MTP stack the trunk import drops.

New guard against the same class of rot: `tests/test_import_profiles_on_disk.py`
builds all three profiles from the checkpoints' own `config.json` and asserts the
transcribed constants (`K3_TEXT_CONFIG`, `QWEN38_27B_VERIFIED_GEOMETRY`) still match
what is on disk; `test_the_k3_split_pattern_is_still_verbatim_from_the_checkpoint`
parses `tokenization_kimi.py` and compares its `pat_str` to the copy in
`importing/tokenizer.py`. All skip on a box without the checkpoints.

## The 27B output gate (spec Q1) — decision needed from Cyril

`/home/cereal/models/Qwen3.8-27B-Uncensored/config.json` declares
`text_config.output_gate_type: "swish"`. transformers 5.12.1 has **no such key** in
`models/qwen3_5/` and applies `attn_output * torch.sigmoid(gate)` unconditionally
(`modeling_qwen3_5.py:714`), so under the lab runtime the checkpoint runs with a
sigmoid gate; the checkpoint was written by transformers 5.15.0, which may not.

Behaviour now: `chimera_omni_27b_profile(real_config)` raises by default and builds
with `allow_unverified_output_gate=True`, stamping
`metadata["attention_output_gate"] = "sigmoid_assumed_declared_swish"`. If 5.15.0
really uses `silu(gate)`, the gated-GQA block needs a gate-kind switch (an S2-layer
change) before the 27B import can claim exactness.

## Verification (AI server, `/home/cereal/model-surgery-lab/venv`, CPU)

```
278 passed in 13.61s          # full suite, no skips (all assets reachable)
 84 passed in 5.03s           # the S3 test files, verbose
  7 passed in 0.20s           # the on-disk profile + pat_str tests
```

Receipt `receipt_s3.json` (sha256 `dea908595fd0c13d2ff59b54b7a5950e9d2e77df7bbaa4bc07cfbc55fcdaf83b`,
8,536 bytes), built from the real `config.json` of all three checkpoints:

- fixed-band ledger: Qwen `248,320 + 4,096 = 252,416`, K3 `163,840 + 4,096 = 167,936`;
  the proportional policy would have relabelled 54,320 Qwen / 35,840 K3 trained rows.
  Band cost `41,943,040` params untied at H=5120, `4,194,304` tied at H=1024.
- band logit mask: text logits bit-identical with the mask on, band softmax mass
  `1.5118 -> 0.0`.
- volume: `[1,1,33,35,37]` at patch 16 -> 27 tokens x 4,096 features, round trip
  `max_abs_diff 0.0`; lossless again at the 6,144 trunk width.
- typed flow: AV levels bit-identical to the pre-generalisation code; volume (type 7)
  now shifted at 3.0, types 0/1/6 stay at sigma 0.
- tokenizers: Qwen `tokenizer.json` has 248,070 ids inside a 248,320 text range
  (Qwen pads the table); K3 tiktoken 163,584 ranks + 256 reserved = 163,840. Both
  round-trip the fixture corpus exactly. Qwen->K3 bridge on the corpus: 43 aligned
  tokens, 42 single-piece, 1 multi-piece, 0 unmapped, byte round trip exact.
- profiles: 0.8B (H 1024, L 24, rope 64/1e7, kda 16/16, tied), K3 (H 7168, L 93,
  decay rank 128, AttnRes block 12, SiTU 4/25), 27B (H 5120, L 64, kda 16/48, untied).

Logs: `pytest_s3_full_after_ondisk_fix.log`, `pytest_s3_new_tests_after_ondisk_fix.log`.
The pre-fix receipt is kept as `receipt_s3_transcribed_geometry_backup.json`.

## Known gaps carried forward

- K3's `kda_decay_rank = linear_attn_config["head_dim"] = 128` is now verified
  against `modeling_kimi_linear.py:523` (`f_a_proj = Linear(hidden_size, head_dim)`),
  not assumed.
- `FLOW_TYPE_SHIFTS[7] = 3.0` for volume is the spec's assumed value; unswept.
- The bridge maps a foreign row to the mean of its destination rows and leaves
  unaligned rows at zero — no receipt claims those rows are usable.
- No safetensors IO (1.12), GGUF writer (1.14) or loaders (1.16/1.17) yet; nothing
  in S3 loads a weight.
