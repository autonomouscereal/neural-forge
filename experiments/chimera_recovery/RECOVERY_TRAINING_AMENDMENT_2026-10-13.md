# Chimera Omni recovery preparation amendment — 2026-10-13

User direction: prepare recovery training for tool calling, intelligence and all
modalities first; optimize folding after preparation. This amendment is a
candidate for the stage-D preregistration. Append it below `## Amendments`
before an affected production step. The frozen 15,339-byte body with SHA-256
`c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370`
must remain byte-identical. Record both the frozen-body digest and the amended
whole-document digest in every run identity and initial receipt.

## A. Corpus integrity and semantic coverage

Keep the sealed D1 input unchanged. For subsequent recovery passes, prepare
`recovery_corpus.v1` rows from the verified registry staging snapshot and
read-only PostgreSQL memory archives. Bind the exact ready-row manifest,
source file digests, original shard digests, source identifiers, collection
dates and split assignment in each cache/run receipt. Unknown original author
dates remain unknown; collection dates are not original author dates.

Tool supervision must retain actual schemas, function names, JSON arguments,
ordered assistant calls and matched results. A final assistant call without a
result may teach call construction and must be labeled as outcome-unobserved.
Do not invent executed tool results. Unambiguous source name-based references
may be canonicalized to deterministic call IDs with that transformation named
in provenance. BFCL acceptable argument alternatives may be selected only from
the recorded target alternatives and validated against the recorded schema.
Missing schemas, context, ambiguous links and redacted targets are quarantine
conditions. Preserve multi-turn, parallel, failure and retry evidence only
when the source explicitly demonstrates it.

Related sessions, source documents, exact and near duplicates belong to one
split. Exclude frozen evaluation sources and prompts before cache creation.
Recompute grouping across merged tranches rather than trusting each tranche's
split independently. Reserve fresh held-out tasks for arithmetic, reasoning,
code, instruction following, tool schemas/calls/results/retries/parallel and
each modality. Never report a coverage label as a behavioral result.

## B. Assistant supervision and teacher caches

The added structured-chat recovery lane uses the actual parent chat template,
thinking off, and authentic `tools` input. Preserve tokens for user/schema/tool
context as input but supervise assistant-owned targets only. Shift ownership
to the next-token target: `assistant_mask[:, 1:]`, intersecting real lengths.
Prompt, tool-result and padding targets have zero CE, EOS, Qwen KD, K3 KD,
unlikelihood and JSD weights. Keep text predictions within `[0, 248320)`.
Never train the music/ledger bands on text examples.

Caches for this lane must be generated from its new packed token IDs and bound
to their SHA-256; flattened-text caches cannot be reused by label or shape.
Apply the existing D2 trainable sites and optimizer contract. The initial
structured-chat pass is one deterministic pass over accepted training rows
(seed 20261013), with its actual token budget calculated and sealed after
packing. No upsampling may fabricate coverage of absent capabilities. Replay
the existing disjoint text/long/K3 sets in D4 under their original contract.

## C. Modality repair requirements

Image generation and editing need prompt/conditioning/target provenance and
genuine target images or sealed teacher trajectories; editing additionally
needs explicit source/target roles and a task instruction. Vision and video
understanding require real input media and grounded answer supervision.
Audio/music and speech understanding must have genuine audio references and
appropriate targets; route names alone do not supply them. Joint AV requires
video and audio from the same source interval with timestamps and alignment.
The current H3 deterministic assignment of unrelated audio is excluded from
synchronized AV training. Existing independent video/audio reconstruction data
may be evaluated separately and must not be claimed as synchronized AV.

3D supervision needs geometry or a sealed geometry-teacher target, meaningful
object conditioning, coordinate/normalization metadata and target provenance.
Charts and arbitrary icons without geometry are excluded from claims of 3D
recovery. A teacher-conditioned image-to-3D sample is distinct from a ground
truth geometry sample; record that distinction.

For denoising, clean targets are available to the loss only. They cannot be
copied into ordinary conditioning features. Legitimate inpainting/editing
context is explicitly role/mask limited. D3b keeps the preregistered
guided-velocity MSE objective: production adapters must explicitly select
`flow_param=velocity`; endpoint/x0 smoke tests do not establish this objective.
Require native/hybrid forward and gradient parity, decoder contracts, finite
targets, actual adapter/driver availability and parameter census before the
affected route trains. Unready routes have executable prerequisite jobs and
explicit blockers, never a `ready` verdict from metadata-only checks.

## D. Receipts, execution and promotion

D1 production launcher passes `--preregistration` into the trainer itself.
The trainer validates before opening checkpoints, includes the observed
document pin in the resume identity, and seals the initial receipt before the
first weight update. Existing resource allocator, disk/RAM floors, clone
protection and exact optimizer resume remain required.

Preparation and CPU contract tests do not launch recovery training or prove
model recovery. Keep original checkpoints and active queues untouched while
preparing private candidates. Publish a route-by-route readiness report with
verified data/caches, code versions, loss objectives, trainable sites, held-out
evaluation and remaining external prerequisites. Preserve stage promotion's
existing content/no-regression rule. Folding optimization starts after this
training preparation is complete.
