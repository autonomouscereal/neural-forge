# NOTES-W12D — the r12 on-policy round fail-closed: cached-decode admissibility

Wave 12D, branch `codex/lrc-blank-compression`.
Root-cause and fix for the r12 on-policy evidence round that failed closed with

```
on-policy round failed closed: decode.001 cached logits differ from
use_cache=False full-prefix logits (maximum error 1.90625).
```

Investigated against local HEAD `3f71c6ac` (HEAD moved to `b9e37374` mid-wave;
neither touches the two files in play). Server runtime
`/home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a`
at `8369f42e`, which is byte-identical to HEAD for every file named here.

**Read §5 first if you only want the relaunch.** The decode gate was not the
only thing blocking this round.

---

## 1. Symptom and what actually failed

Round root
`/home/cereal/model-surgery-lab/artifacts/neural_forge_on_policy_rounds_v1/a0023-exact4x-h816-mgv3-r12`,
`round.log`. The raise site is `_compare_logits`
(`src/neural_forge/hf_lfm2_circuit_trace_cache.py:3363`) during the **parent**
capture, at phase `decode.001` of the **first** capture batch.

Reproduced exactly, twice, on the server from the sealed generation cache
(scratch only, nothing in the round dir written):
`/home/cereal/nf-round-debug/repro_round.py` → same message, same `1.90625`.
Only three comparisons ran before the raise (`cached prefill`, `decode.000`,
`decode.001`), which is what pins the failure to batch 0.

The message is slightly misleading and cost time: `maximum error 1.90625` is
the **global** max, and at that element the tolerance is `1.0 + 0.2*8.3125 =
2.6625`, i.e. that element passes. `torch.allclose` failed on **190 other
elements** of the 65536-wide row whose `|full|` is small enough that the
`atol=1.0` floor bites. If anyone touches that message, report the violating
element, not the max.

### The failing prefix

Fit prompt 0, and the child's own continuation re-tokenized to donor space:

```
prompt : 'Good morning. I just moved to a new city and do not know anyone yet.
          Respond warmly and ask one question that would actually help.'
forced : [2, 2, 2, 2, 2, 816, 2, 1000]
tokens : <|endoftext|> x5, 'om', <|endoftext|>, 'ost'
```

The child is the degenerate r12 child. It emits `<|endoftext|>` in a loop.
Generation does not stop on it because the child tokenizer's `eos_token_id` is
**7** (`<|im_end|>`), not 2 — so the probe forces all 8 tokens past the
collapse. The bridge round-trip is faithful (target 2 → donor 2, both
`<|endoftext|>`); nothing is mis-mapped. No out-of-vocabulary rows, no unk
flood: donor ids in the failing batch span 523–8564 against a 65536 vocab.

---

## 2. Root cause — genuine bf16 cached/full-prefix divergence (hypothesis b)

`src/neural_forge/hf_lfm2_circuit_trace_cache.py:3747-3779` compares the
one-token KV-cached decode logits against a `use_cache=False` full-prefix
recompute. On these prefixes the two are genuinely unequal **at the parent's
own runtime precision**. Nothing is miscomputed; the gate is correct to refuse.

Measured on the server (`/home/cereal/nf-round-debug/isolate.py`), same prompt,
same parent, eager + cuda:0:

| run | decode.000 | decode.001 | decode.002 | violators @ .001 |
|---|---|---|---|---|
| exact batch 0, **bf16** | 0.140625 | **1.90625** | 1.09375 | **190** |
| exact batch 0, **fp32** | 2.4e-05 | 3.0e-05 | 4.4e-05 | 0 |
| same prompt, non-degenerate 8-token continuation, bf16 | 0.1875 | 0.15625 | 0.2266 | 0 |
| same prompt + 8x `<|endoftext|>`, bf16 | 0.140625 | **1.90625** | 1.09375 | **190** |

fp32 collapses the error by ~5 orders of magnitude, so this is precision, not
cache logic. The synthetic 8x-control-token run reproduces it with no
truncation anywhere.

### Why `decode.001` and not `decode.000`

Per-layer probe (`/home/cereal/nf-round-debug/layerprobe.py`). LFM2.5-230M is
`['conv','conv','full_attention','conv','full_attention', ...]`, 14 layers.

- Layers 0–4 diverge by 0.002–0.01 in both steps. Attention layers 2 and 4 are
  *not* where this starts.
- At `decode.000` the first control token produces a **massive activation**:
  `max|h|` jumps to **36** at layer 5 (conv) and stays there. Cached-vs-full
  diverges by 0.75 absolute there — but only **2%** relative, so the final
  RMSNorm + LM head give a harmless logit error of 0.14. `decode.000` passes.
- That 0.75 absolute error is now **inside the conv/KV cache state**.
- At `decode.001` the hidden state collapses back to O(1) (`max|h|` 0.67–2.1)
  while the inherited absolute error stays 0.06–0.29. Relative error at layer
  13 is **0.43**. The final norm divides by the now-small norm, so 43% relative
  error lands directly on the logits → `max|Δlogit| = 1.90625`.

So: a massive-activation control token writes a bf16-rounded state into the
cache, the next step normalizes against a much smaller vector, and the stored
rounding is amplified into the logits. `decode.002+` recovers.

### Blast radius

Full survey of all 119 capture batches (`/home/cereal/nf-round-debug/survey.py`,
`survey.json`): **4 of 119 fail**, indices `[0, 1, 109, 112]`, every one at
`decode.001`, every one a run of control tokens (`<|endoftext|>` id 2 or
`<|startoftext|>` id 1):

```
0   forced [2,2,2,2,2,816,2,1000]      max 1.90625  violations 190
1   forced [1,1,1,1,1,2038,11440,11440] max 2.0      violations 179
109 forced [1,1,1,1,1,1,1,1]            max 1.21875  violations 1
112 forced [1,1,1,1,1,1,1,1]            max 1.9375   violations 201  (also decode.002, 2.512)
```

For contrast the sealed base cache's own 2304 recorded comparisons never
exceeded **0.5625** (`p99 = 0.359`). The child-visited corpus is a genuinely
different regime, which is why the base capture passed the identical gate.

Control tokens are **necessary but not sufficient**: 15 batches contain them,
only 4 fail. Max error among the 104 batches with no control token is 0.703
with zero violations. A token-space blacklist would therefore over-exclude 11
sound trajectories and still not be provably sufficient — which is why the fix
measures the gate instead of guessing at its trigger.

---

## 3. Ruled out, with evidence

**(a) Truncation bug — ruled out.** `--truncate-forced-decode-to-capture-budget`
slices `forced_ids = forced_ids[:max_decode_steps]`
(`hf_lfm2_on_policy_evidence_round.py:870`) and nothing else; prompt ids and
masks are untouched, and each capture batch is a single unpadded row
(`_CaptureBatch.as_model_batch`), so cached and no-cache paths are built from
the same `prefix_ids`/`prefix_mask` inside the capture and cannot disagree on
the effective sequence. Empirically: **batch 0 is not truncated**
(`truncated: False`, 8 forced tokens = the budget), and across all 119 batches
**17 are truncated and none of them fail** (`truncated AND failing = []`).

**(c) dtype/device drift — ruled out as drift, confirmed as the amplifier.**
The launcher passes `--parent-dtype float32`, but
`_load_parent_matching_fingerprint`
(`experiments/run_r12_on_policy_round.py:153-190`) silently walks
`float32 → bfloat16 → float16` until one reproduces the sealed parent
fingerprint. Measured on the server:

```
float32   identity 2ab3d000...  MATCH False
bfloat16  identity 96a6779d...  MATCH True   <- sealed base cache parent
float16   identity 3689d6ae...  MATCH False
```

So the round runs in **bf16**, and so did the base capture — the sealed
identity *is* the bf16 one, and the sealed tolerances (`rtol=0.2, atol=1.0`)
are bf16-scale. Execution identity matches exactly
(`3d077b94...`, eager, cuda, fast-short-conv false/false). There is no drift to
pin; bf16 is the sealed configuration. Do **not** "fix" this by forcing fp32:
that changes the parent identity and retires the base cache.

---

## 4. The fix

`src/neural_forge/hf_lfm2_on_policy_evidence_round.py` only. **The collector was
deliberately left byte-identical** — see §4.2, this is the important part.

### 4.1 Exclude-and-record, not uncached replay

Chosen over "replay failing rows with `use_cache=False`". The sealed capture
selection asserts `forced_decode_use_cache: true` for **every** phase, and
`require_matching_capture_selection` (`:1811-1819`) pins the round's selection
digest to the base cache's `f229e5f3...`. Sealing a mixed cached/uncached cache
under that selection would make the sealed manifest state something false about
part of its own contents, and base + extension would stop being the homogeneous
corpus a closed-form writer is entitled to assume. Excluding keeps every sealed
phase genuinely cached-verified; the refusals are recorded with their measured
error rather than silently dropped.

Added:

- `screen_lfm2_cached_decode_admissibility(...)` + `_screen_one_batch` — replays
  exactly the capture's forwards and calls the collector's own
  `_compare_logits`, returning one verdict per batch instead of raising. No
  hooks, no retained traces, nothing written, training flags restored.
- `_screen_cached_decode_admissibility(...)` — drops refused trajectories,
  records each with `first_inadmissible_phase`, `max_abs_error`,
  `elementwise_tolerance_violations`, forced token ids and truncation flag, and
  refuses the round outright if fewer than
  `min_cached_decode_admissible_batches` survive.
- `Lfm2OnPolicyRolloutConfig.screen_cached_decode_admissibility` (default
  **False**) and `min_cached_decode_admissible_batches` (default 1). The screen
  is **opt-in**: left off, this round behaves exactly as before and still fails
  closed, so the refusal stays the default answer.
- Manifest: `capture.cached_decode_admissibility` block; and
  `parent.executed_only_inside_sealed_capture` is now **computed**, because a
  screened round really does run the parent outside the sealed capture. A
  screened round reports it `False` plus
  `executed_in_cached_decode_admissibility_screen: true` and
  `screen_parent_forward_calls`. It does not keep asserting something untrue.

Digest safety: the new rollout keys are added to the rollout body **only when
the screen is enabled**, the same conditional-additive pattern the bridged
request keys use. An unscreened round keeps its exact `rollout_sha256`,
`request_sha256` and round directory, so already sealed rounds still reload.

### 4.2 Why the fix is NOT in the collector — do not undo this

I first put the screen in `hf_lfm2_circuit_trace_cache.py` as an additive
function. That is wrong, and the test suite caught it.

`implementation_identity()` (`hf_lfm2_circuit_trace_cache.py:844-871`) hashes
**the collector's own source file**:

```
collector_source_sha256 = sha256(hf_lfm2_circuit_trace_cache.py)
```

and `run_lfm2_on_policy_evidence_round` requires the extension cache and the
base cache to agree on `implementation_sha256` (`:2541-2547`). Measured:

```
deployment collector       16271526...  -> implementation 732ca9ce...
+ my additive function     175fda28...  -> implementation 00a7f07b...
```

So **any** edit to that file — even a pure addition that changes no behaviour —
retires every cache sealed before it. Three collector tests fail immediately on
literal digests. The screen therefore lives in the round module and imports the
collector's private helpers (`_compare_logits`, `_forward`, `_output_logits`,
`_cpu_tensor`, `_sample_sequence_tensor`, `_move_to_device`, `_output_field`,
`_resolve_input_device`), which is the pattern `_prepare_batches` already used.
`test_the_screen_must_not_live_in_the_collector` pins this.

### 4.3 Verification

Run on the server venv (torch is not installed on the Windows runtime), against
a staged tree whose collector sha256 is `16271526...`, i.e. byte-identical to
the deployment:

```
tests/test_hf_lfm2_on_policy_evidence_round.py      23 passed   (16 pre-existing + 7 new)
tests/ORIG_test_round.py (pre-change file)          16 passed   (backward compatible)
tests/test_hf_lfm2_circuit_trace_cache.py           49 passed, 1 skipped  (baseline)
ruff check (both files)                             clean
```

End-to-end on the real r12 data, screen enabled
(`/home/cereal/nf-round-debug/prove_screen.py`):

```
SCREEN over 119 real r12 batches: admissible 115 excluded 4
excluded batch indices: [0, 1, 109, 112]
  batch 0  decode.001 max_abs_error 1.90625 violations 190/65536 bfloat16 truncated False
  batch 1  decode.001 max_abs_error 2.0     violations 179/65536 bfloat16 truncated False
  batch 109 decode.001 max_abs_error 1.21875 violations 1/65536  bfloat16 truncated False
  batch 112 decode.001 max_abs_error 1.9375  violations 201/65536 bfloat16 truncated False
total screen parent forwards: 2100
```

Identical to the independent survey. The screen costs one extra parent forward
pass per candidate batch (2 + 2·decode_steps each); for this round that is 2100
forwards, a couple of minutes on one 3090.

---

## 5. The round is still blocked — second, independent defect

**Fixing the decode gate is necessary but not sufficient. Do not relaunch
expecting a sealed round.**

With the screen enabled the capture now completes, and the round fails at the
*next* clause (`/home/cereal/nf-round-debug/repro_screened.py`, exit 0, log
`screened.log`):

```
Lfm2OnPolicyEvidenceRoundIntegrityError: Extension and base caches disagree on
implementation; their evidence cannot be consumed by one closed-form solve.
```

Field-by-field against the sealed base manifest:

| field | sealed base | postpub runtime | |
|---|---|---|---|
| `torch_version` | 2.11.0+cu128 | same | SAME |
| `transformers_version` | 5.12.1 | same | SAME |
| `transformers_lfm2_source_sha256` | 2ec8be8a… | same | SAME |
| `safetensors_version` | 0.8.0 | same | SAME |
| `collector_source_sha256` | 71e99ee9… | 16271526… | **DIFF** |
| `canonical_contract_source_sha256` | 5956f909… | 789c6a6f… | **DIFF** |

The environment is identical. Only neural-forge's own sources drifted. Git
archaeology on the file contents:

- base cache collector content == `42d97c35` (2026-07-17, *Implement severe
  CPCF blank-target folding*)
- base cache canonical content == `19be391c` (2026-07-10, *Add fail-closed LFM2
  short-conv transplants*)
- postpub collector content == `b799a1a7` (2026-07-31, *Shard child replay
  across identical devices*)
- postpub canonical content == `a98d2063` (2026-07-29, *Inspect sealed
  heterogeneous LFM2 blanks*)

The base cache is 2026-07-17 evidence; the collector has moved twice since. The
launcher header explains the round *must* run from the postpub runtime (it
needs `ab3ec99a` and `09825ab2`), and that runtime cannot reproduce the sealed
implementation identity. That is a real conflict and it is a **coordinator
decision**, not mine. The three options, none of them free:

1. Recapture the base cache under the current implementation identity — most
   expensive, and it retires the sealed 256-batch corpus and everything keyed
   to it.
2. Run the round from a runtime pinned to the 2026-07-17 collector + 2026-07-10
   canonical — but then it lacks the bridged-round preimage support the round
   needs, so this probably needs a cherry-pick, which changes the collector
   digest again unless the cherry-pick avoids those two files.
3. Deliberately relax the implementation clause with a recorded, reviewed
   delta. This is an evidence-integrity decision and needs an explicit argument
   that the collector delta between `42d97c35` and `b799a1a7` cannot change
   captured tensor values.

I did not touch this. It was not the reported failure and picking one of these
is a judgement call about what the evidence means.

---

## 6. Relaunch procedure

Nothing has been relaunched. The round directory
`.../a0023-exact4x-h816-mgv3-r12/round-000` and its sibling
`round-000.failed-truncation-001539` are untouched; the only writes I made are
under `/home/cereal/nf-round-debug/`.

### 6.1 Coordinator must commit first

The launcher refuses a dirty or mismatched runtime
(`nf-launch-r12-round-refold.sh`, exit 41):

```bash
actual_source_commit=$(git -C "$RUNTIME" rev-parse HEAD)
[ "$actual_source_commit" != "$SOURCE_COMMIT" ] && exit 41
[ -n "$(git -C "$RUNTIME" status --porcelain)" ] && exit 41
```

Commit `src/neural_forge/hf_lfm2_on_policy_evidence_round.py` and
`tests/test_hf_lfm2_on_policy_evidence_round.py` (and this file), then update
the runtime from the bundle so its HEAD matches and the tree is clean.

### 6.2 One wrapper change is still required (NOT applied — outside my ownership)

`experiments/run_r12_on_policy_round.py` builds the rollout config and has no
flag for the screen, so the screen cannot be turned on from the launcher.
Needed, in `run()` at the `rollout = Lfm2OnPolicyRolloutConfig(...)` call:

```python
        screen_cached_decode_admissibility=(
            args.screen_cached_decode_admissibility
        ),
        min_cached_decode_admissible_batches=(
            args.min_cached_decode_admissible_batches
        ),
```

and in the parser:

```python
    parser.add_argument(
        "--screen-cached-decode-admissibility", action="store_true"
    )
    parser.add_argument(
        "--min-cached-decode-admissible-batches", type=int, default=1
    )
```

Also worth surfacing in the runner receipt: copy
`manifest["capture"]["cached_decode_admissibility"]` into the receipt body so
the exclusions are visible without opening the round manifest.

### 6.3 Launcher change

In the `round)` stage of
`/home/cereal/model-surgery-lab/deployments/nf-launch-r12-round-refold.sh`, add
after `--truncate-forced-decode-to-capture-budget`:

```
    --screen-cached-decode-admissibility \
    --min-cached-decode-admissible-batches 100 \
```

`100` is a deliberate floor: this round has 119 candidates and 115 admissible,
so anything much below ~100 would let a much more degenerate child through
quietly. Set it to whatever fraction of the corpus you are willing to seal.

The existing `if [ -e "$ROUND_ROOT/round-000" ]` guard (exit 40) will refuse
until the failed `round-000` is moved aside, the same way
`round-000.failed-truncation-001539` was. Move it, do not delete it — it holds
the generation cache that made this diagnosis cheap.

### 6.4 Resolve §5 before running

`SOURCE_COMMIT=<new commit> ./nf-launch-r12-round-refold.sh round` will now get
past `decode.001` and stop on the implementation clause instead. Decide §5
first, or the relaunch just buys a different fail-closed message.

### 6.5 Deployment copy

`src/neural_forge/hf_lfm2_on_policy_evidence_round.py` was copied into
`neural-forge-postpub-e893767a` with the original preserved at
`hf_lfm2_on_policy_evidence_round.py.w12d-backup`. **This leaves the runtime
checkout dirty, so the launcher will exit 41 until it is committed there.** To
revert:

```bash
cd /home/cereal/model-surgery-lab/deployments/neural-forge-postpub-e893767a
git checkout -- src/neural_forge/hf_lfm2_on_policy_evidence_round.py
rm -f src/neural_forge/hf_lfm2_on_policy_evidence_round.py.w12d-backup
```

The collector was **not** copied and must stay as it is (§4.2).

---

## 7. Scratch artifacts (server, safe to delete)

All under `/home/cereal/nf-round-debug/`:

| file | what it proves |
|---|---|
| `dtype_probe.py` | which parent dtype reproduces the sealed fingerprint |
| `repro_round.py` / `repro-round/` | exact reproduction; `capture-batches.json`, `compare-trace.json` |
| `isolate.py` | bf16 vs fp32, degenerate vs non-degenerate, synthetic repetition |
| `survey.py` / `survey.json` | all 119 batches, per-phase error and violations |
| `layerprobe.py` | per-layer divergence and the massive-activation amplification |
| `prove_screen.py` | the screen's verdicts on the real corpus |
| `repro_screened.py` / `screened.log` | end-to-end with the screen; exposes §5 |
| `localtest/` | staged tree used to run the suites on the server venv |
