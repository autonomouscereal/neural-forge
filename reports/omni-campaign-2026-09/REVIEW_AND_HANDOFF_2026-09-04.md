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
  September 4 non-deep verification: one checkpoint/record/ledger row matched.
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
real 0.8B assets absent locally. Symlink tests ran and passed. Full server validation
and real-run results will be appended below as they land. No model cleanup was
executed during this review; root disk pressure remains a planning constraint.

## Exact remaining construction work

- S0: create the bf1627B Chimera import with the declared music band; independently
  verify serialization and bf16/split execution. This remains PROBE.S0, not a fold.
- S2P: implement the missing full-K3 dual-stream pass. Fit on folded-prefix inputs
  against teacher targets, keep both residual pairs, bind snapshots to input IDs,
  test uninterrupted/resumed equivalence, and chunk AttnRes. Then measure real L01
  at the current corpus size before launching all 92 MoE layers.
- Complete streamed27B exactness and the family-neutral readable review runner.
- Fold the Qwen trunk, complete K3 charts/union, fold and install actual media
  components, and judge the single child with visible text/media plus source
  ablations. A source's name or imported rows alone is not functional transfer.
- Register/review/package the all-source child and clone it before any training.

The old CL768 LFM ladder is closed. Do not restart Wave46 or revive old August
runtime-receipt/cleanup tasks as a substitute for this handoff.
