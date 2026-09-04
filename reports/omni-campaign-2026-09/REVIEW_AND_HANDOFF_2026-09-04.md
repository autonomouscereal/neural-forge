# September 4 review and continuation

## Course verdict

The September all-source, dense Chimera Omni campaign is the correct continuation.
The LFM2 matrix is historical evidence, not the immediate work queue. Step 1 is
**not complete**: an exact Qwen import and separate source/component receipts do
not constitute the requested folded, all-source child.

The governing order, reconfirmed in PostgreSQL human prompts 609454 (September 1)
and 619622 (September 2), is:

1. Fold the requested sources into one dense Chimera child.
2. Show its source/modality ledger and actual whole-model behavior.
3. Preserve the checkpoint and clone before distillation.
4. Distill, demonstrate results, then clone and fine-tune, demonstrating results again.

Q4 is a final deployment step, not folding credit. The latest deployment target is
24 GB including P40, with 256k context; neither has been demonstrated for OMNI1.
No distillation or training was started in this continuation.

## What was reviewed, and limits

- 403 core PostgreSQL human-prompt rows across eight project sessions, plus 19
  ancillary project prompts. Automated heartbeat and subagent dispatches were
  excluded from human counts. Origin prompts, later corrections, and the three
  relevant Claude transcripts were compared; no substantive Claude user prompt
  was missing from PostgreSQL. Historical encoding damage was not invented away.
- The entire current Codex task, through September 4: 78 paginated turns and 74
  user messages. These overlap PostgreSQL counts; they are not additive unique users.
- All 297 messages (420,675 body characters) in this task's exact Relay thread,
  including 295 historical messages from August 4–13 and two September 4 items.
  Thread `b3f96e0c-d2bb-40fc-9a9e-ac4a0e902625` belongs to root context
  `a4db65edea82b93d5205bf525b6172164c13babd026a78ed06cf48762dd99d12`.
  Of 88 Claude messages, 87 were unattended dispatches; the sole attended message
  was an estate-audit response, not an acceptance or verification of Neural Forge
  implementation. Relay ACKs must not be called Claude code review.
- Matrix setup/waves 1–45: inventory of 48 notes files, their summary/orientation
  sections, relevant full scorecards/addenda, current standing state, frontier,
  and commit history through `1779b7b9`. All 1,056 ledger rows were parsed;
  SHA-256 independently matches
  `d2027831e229d763a251ca602655c729a4432645bed43a9ba2e4a8824ef211ea`.
  The frontier contains 358 reviewed whole-model children.
- Chimera changes through `5b35906`, import/export code and tests, R-Q27B and K3 MLA
  receipts, September campaign/runbook/stages, three composite tools, corpus
  receipts, and a current read-only server inventory.

This is not a claim that every historical tool-output byte, every dossier paragraph,
or every one of the 1,052 reportedly resolving matrix seals was independently read
or recomputed. Newer Claude work is established from transcripts, code and receipts,
not from the stale private Relay thread. No unrelated private thread was consumed.

## What Claude achieved

The matrix demonstrated useful component folds and physically smaller executable
children, while exposing that component wins do not compose predictably. User
addenda explicitly relaxed the immediate objective from 90% quality at 0.25x to
retained intelligence and a coherent 0.5x milestone. Those changes were authorized,
not automatically goal drift. The final 0.4116484014x child retained only one narrow
non-copy lane; the deeper 0.3859919152x child is explicitly an echo-persistence
diagnostic. Neither establishes a general folding ceiling or large-model transfer.

The September work established real Qwen27B import accounting and short text-path
agreement, a K3 MLA mixer receipt, and paired Qwen/K3 evidence. It did not establish
that K3/media intelligence has transferred to a compressed composite child.

## Corrections to older summaries and scheduling

- R-Q27B is now real: 210/210 short-prompt argmax positions over eight prompts;
  maximum logit difference `3.0326843e-4`, CPU fp32. The fold plan's claim that
  every layer is inside a `1e-5` relative band is incorrect. Prompt 4/L63 records
  `0.0126647949 / 9.29210949 = 1.3629623e-3`. This is not a bf16, serialized
  roundtrip, decode, long-context or compression receipt.
- Actual whole-reference peak RSS is 247,780,716,544 bytes (230.76 GiB), not the
  runbook's 203-GiB estimate. That expensive reference run must not be co-scheduled
  using the obsolete estimate. Import-only S0 is a different operation whose peak
  must be measured separately.
- Corpus v2 has 24 x 4,096 = 98,304 evidence positions, 18 fit/6 held rows,
  69,743 K3 fit tokens, 23,186 held tokens, and 22,769 held cross-tokenizer boundary
  pairs. Separate review files add 45,056 positions and have zero fit masks.
  Evidence residual estimates based on 65,536 positions require a 1.5x correction.
- Current parent registry: `PARENT-Q38-27BU`, 27,781,427,952 header-derived
  parameters, directory SHA `8f6735a4510e65f014f0dea8efdbaa3e5a05407c370c976de2d148266d60667a`.
  September 4 deep verification: one checkpoint/record/ledger row matched.
- Server at review: no composite process, GPUs idle, 232 GiB RAM available,
  185 GB free on root (90% used), Lexar mounted at `/mnt/lexar` with about 708 GB
  free. The September 3 Lexar/K3 staging instruction supersedes treating repeated
  CIFS reads as the only route. The Lexar copy now contains all 96 shards
  (1,560,936,091,448 bytes); its log completed September 3 at 18:09:54Z. Inspection
  of the copy script shows **size-only** verification, not SHA verification. Preserve
  this substantial completed work; verify against authoritative digests before
  attributing its bytes to the source. Do not recopy the entire model blindly.

## Safe implementation continuation

Before modifications, campaign code and top-level runbook documents were saved in
Git commit `2e393e860692410c6672855115fd4cca9655a4e8`, branch
`codex/sep04-composite-baseline`. This used a separate index: the existing branch,
main index, unrelated dirty files, and model artifacts were untouched.

Two actionable problems were fixed before a real 27B continuation:

- S0 could delete its own parent with `--src P --out P --overwrite`. It now rejects
  resolved path overlap, including ancestry/aliases, protects comparison inputs and
  executable source, validates existing layout/options before replacing outputs,
  and records actual executable-package/tool hashes. Dirty executable checkout
  files are refused; an archive COMMIT is labeled as a declaration, not proof.
- `qf_stream27 --src-dir` previously compared only architecture config. It now
  hashes the actual index and every consumed weight shard, refuses substituted
  same-config weights before engine construction, records source identity, and
  rechecks it before assembly and after export. This adds bounded-memory full-file
  reads; it is not a no-I/O provenance shortcut. An independent review caught and
  closed a further ambiguity: HF assembly prefers a single `model.safetensors`,
  while the streaming reader prefers an index. Mixed layouts selecting different
  files are now refused. Three identity checks read roughly 324 GB for separately
  staged 54-GB weights (162 GB when source and parent are the same resolved path).

Tests: six source-identity regressions passed locally, followed by three further
regressions included in the server rerun. The full QF suite passes
on the established AI-server runtime: **19 passed in 26.96 seconds**. The local
full suite exposes a pre-existing runtime incompatibility (`_update_linear_attn_mask`
missing in the local Transformers version) plus a Linux-share-path assumption;
it is not reported as passing locally. Server test deployment is isolated under
`/home/cereal/omni-campaign/composite/codex-sep04-safety`; Claude's original server
tool files were not overwritten.

S0 local validation: **24 passed, 2 skipped in 157.93 seconds**; the skips require
real 0.8B assets absent locally. Symlink tests ran and passed. Full server validation:
**26 passed in 197.37 seconds**, including those real-asset cases. Combined with QF,
all 45 server tests pass. The subsequent parent deep-hash check continued beyond
the SSH command's 300-second wrapper limit; its original process was inspected,
not duplicated. Real-run results are recorded below. No model cleanup was
executed during this review; root disk pressure remains a planning constraint.

### S2P memory prerequisite completed

`k3_common.apply_attn_res` now slices tokens before concatenating snapshots or
upcasting. The existing five-positional API defaults to 1,024-token chunks;
`token_chunk_size=None` retains the original expression. Invalid sizes and
mismatched token/hidden dimensions are refused. Empty input, zero snapshots,
noncontiguous inputs and dtype promotion are preserved.

The independent pre-change expression matches bit-for-bit in the fp32/bf16 CPU
fixtures: **63 tests passed locally**, and **63 passed on the server in 7.42 s**.
The allocation-bound test observes token chunks 1,024 / 1,024 / 3 before cat/upcast.
At T=98,304, nine snapshot/prefix entries and H=7,168, one fp32 stack is 23.625 GiB
unchunked versus 252 MiB per default chunk (shape arithmetic, not a measured RSS
claim). The bound applies to inference/no_grad; autograd can retain intermediates.
GPU numerical behavior and total real-layer memory are not yet measured.

This makes **108 focused server tests passing** across QF, S0 and this prerequisite.
It does not complete the K3 dual-stream harness. Deployment is isolated under
`composite/codex-sep04-safety/k3-memory`; the original server K3 lane is unchanged.

### Real S0 continuation

The existing parent passed deep verification: one checkpoint, one record and one
ledger row matched, `deep=yes`. At 20:03:50Z the guarded CPU import was started
against that parent, with the music_codes:65536 band, bf16 output, and no overwrite.
Output: `/home/cereal/omni-campaign/composite/s0/q38-27bu-chimera-bf16-sep04`.
Launcher PID 346331; Python worker PID 346333. At 20:29:49Z the importer wrote its
receipt: 27,699,193,344 header-derived parameters, 1,011 tensors and 14 bf16 shards.
Peak RSS reported inside the importer is 178,717,544,448 bytes (166.44 GiB).
State SHA: `3fc569a683111ef447ef287af241ae42311a2ac5fa38522659c9b0fae734509e`.
Receipt SHA: `79c6b7f1eae3db24be7c5edd50533f4cdf48169ec18805d69e9b2d988c73d9ec`.

The first independent verification attempt stopped at its successful-process-exit
guard: the receipt was already written, but Python was still in disk sleep during
teardown. Only stdin/stdout/stderr remained open; model memory had been released.
That refusal was not counted as a pass. The process was left intact and exited
with status **0**. `/usr/bin/time -v` measured **33m37.74s**, versus 1,556.61 seconds
through receipt generation (642.45s import, 526.8s writing). The delay is included
in the elapsed process time; no duplicate import was launched.

Independent serialization verification then **passed** in 443.74 seconds with
4,329,029,632-byte peak RSS (4.03 GiB). It reproduced the pre-write full-state SHA,
checked all 14 shard file hashes and 1,011 individual tensor hashes, reconciled
the exact 27,699,193,344 header elements, required BF16 throughout, and checked
actual zero bytes in both untied ledger-band tables. It also checked unique
header keys, contiguous offsets, exact payload lengths and index/name agreement.
The selected server runtime was confirmed to have `sys.flags.optimize == 0` and
`PYTHONOPTIMIZE` unset; assertion gates were not intentionally disabled.

Only five compact evidence/source files are retained locally under
`composite/receipts/sep04/` (not the model weights):

- `import_receipt.json` and its SHA-256 sidecar.
- `s0_serialization_verification.json`, SHA
  `7b6356e1c786bffb8437a7476d4bce6a82b9819d2d87dbd571e5b9c08c7b5796`.
- `s0_import.time.log`, SHA
  `b0c0692bc954caa98a885357855e6959eee5fb91d7d20da83e6b80729660668d`.
- `verify_s0_serialization.py`, independently reviewed implementation, SHA
  `1f912500cffe6705fb58e341a93fb6b559d5db15bd9c00f21ac81d5ff53ea347`.
  Run with ordinary Python, never `-O`/`-OO`; it refuses to replace its result.

**Built and serialization-verified is not behavior-verified.** No new whole-model
forward, bf16 parity, split execution, compression, long-context or media-transfer
claim is made. This unregistered import probe is not OMNI1, and its zero music rows
do not demonstrate music capability.

Retention: preserve this one new probe, existing parents, the completed K3 Lexar
copy, code and the needed compact receipts. No existing model/result files were
removed. Root has about 129 GB free after writing S0. A broader cleanup still needs
a path-specific keep/delete manifest; the old runbook deletion commands are not
an approved current cleanup plan.

## Exact remaining construction work

- S0: use the built, serialization-verified bf16 27B import with its declared music
  band; complete bf16/split execution and streamed parity before sealing PROBE.S0.
  Do not rerun the import. It is not a fold or the all-source child.
- S2P: implement the missing full-K3 dual-stream pass. Fit on folded-prefix inputs
  against teacher targets, keep both residual pairs, bind snapshots to input IDs,
  test uninterrupted/resumed equivalence, and use the tested chunked AttnRes. Then measure real L01
  at the current corpus size before launching all 92 MoE layers.
- Complete streamed27B exactness and the family-neutral readable review runner.
- Fold the Qwen trunk, complete K3 charts/union, fold and install actual media
  components, and judge the single child with visible text/media plus source
  ablations. A source's name or imported rows alone is not functional transfer.
- Register/review/package the all-source child and clone it before any training.

The old CL768 LFM ladder is closed. Do not restart Wave46 or revive old August
runtime-receipt/cleanup tasks as a substitute for this handoff.
