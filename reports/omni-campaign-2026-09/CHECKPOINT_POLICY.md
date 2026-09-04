# CHECKPOINT POLICY — Chimera Omni campaign (`omni-2026-09`)

Status: **the rules Cyril gave, written down.** Every threshold in here that is
not one of his sentences is a *proposed decision line for him*, marked as such.
Nothing in this file is a gate invented by a tool.

The ruling this document exists to execute, verbatim (2026-09-01, ninth ruling
of the session):

> [Cyril quote withheld]

and the standing order it sits inside:

> *"fold to the ceiling, checkpoint and clone, distill from the parents for all
> modalities, checkpoint and clone, finetune on the curated dataset; delete bad
> checkpoints; package partially-capable ones for your review."*

The tooling that enforces this lives in `ledger/`:
`oc_seal.py`, `oc_ledger.py`, `oc_checkpoint.py`, `oc_package.py`,
`oc_dashboard.py`, tests in `test_oc_ledger.py`. **The registry is the only path
that may delete a checkpoint.** There is no other supported way.

---

## R0. What a checkpoint is

A directory of weights (safetensors, sharded allowed), its config and tokenizer
files, and an append-only chain of **sealed records** in the registry. The
record is the artifact; the ledger row is the index into it. A checkpoint that
is not in the registry does not exist for any purpose in this campaign — it
cannot be reviewed, cloned, packaged, cited as a floor, or deleted.

Storage is counted in **parameters**, read from the safetensors headers on disk,
never from a config's claim about itself. `net_ratio = child parameters /
parent parameters` is arithmetic and is never asserted over the counts. Every
header's tensor byte-spans are checked against the file's real size before a
count is taken off it, so a truncated shard is a refusal and not a confident
number.

**The bytes decide, and the record says so.** When the directory holds
`*.safetensors`, both counts are ALWAYS read from the headers, and an explicit
`--parameter-count` / `--parent-parameter-count` that disagrees is a refusal —
the same treatment a `--component` TENSOR_PREFIX already got. A typed count
survives only where there is nothing to read it from, and only behind
`--allow-asserted-parameter-counts`; the record, the ledger row, SCORECARD.md's
`ratio from` column and `REVIEW_<TAG>.md`'s ratio line then all carry
`asserted-by-operator` instead of `derived-from-safetensors-headers`, so a
reader can tell a measured 16x from a typed one without leaving the page.

`oc_checkpoint.py verify` RE-DERIVES both counts from the headers on every run,
not just under `--deep`: re-dividing two sealed integers can never disagree with
the sealed quotient, and the header read costs 0.00 s on a real 4B while `--deep`
spends minutes re-hashing bytes.

The directory digest is sha256 over the sorted `[[path, file sha256], …]` list
of every file under the checkpoint, with **six** exclusions and no others:
the file suffixes `*.partial` (a crash-truncated atomic write), `*.pyc` and
`*.lock`, and the directories `__pycache__/`, `.git/` and `.ipynb_checkpoints/`
(a model directory obtained with `git clone` carries a `.git` tree whose pack
files change on every fetch without a weight changing). That is the one place
the digest is not literally every byte in the tree, and it is stated here
rather than left to be discovered.

**Symlinks are refused, files and directories alike.** A symlinked directory is
not skipped — it is a hard refusal, because skipping it is how a checkpoint
seals a `directory_sha256` and a `parameter_count` that cover none of the bytes
under it, and how `clone` then copies the same visible subset, re-digests it,
finds the two digests equal and writes a receipt asserting a byte-identical
clone that is missing the weights. If shards live elsewhere, copy them in or
register the target directory. The digest, the parameter count and the
per-tensor index all walk through one enumerator (`oc_seal._digest_files`), so
they cannot disagree about which files are in a checkpoint.

## R1. Naming

```
OC<NN>-<STAGE>-<PARENTCODE>-<TAG>      STAGE ∈ {F fold, D distill, T finetune}
PARENT-<PARENTCODE>                    a registered parent
<source>.clone-pre-<D|T><k>            a pre-training clone
```

`TAG` is 1–16 characters of `A-Z0-9` and carries no dash, which is what lets a
parent code keep its own (`Q35-0.8B`, `K3-L01`, `DK3-42`). The split is never
greedy: either `--parent-code` is given and the id must literally start with
`OC<NN>-<L>-<code>-`, or the parent code must be one of the codes below. An id
that is a parent code with no tag (`OC01-F-Q35-4B`) is REFUSED rather than read
as parent `Q35` tag `4B`. Examples: `C2S4L03`
(MLP channel coreset, level S4, layer 3), `X44` (full-attention layer 44
excised), `V128K` (vocab coreset at 131,072), `K3L01C14R42` (K3 layer 1
expert-union at 42x), `H3B04` (4 H3 DiT blocks transplanted). A distilled or
finetuned descendant of a fold child keeps the fold TAG and adds a rung
(`ARCH.C2S4L03.D1`).

Parent codes: `Q35-0.8B`, `Q35-2B`, `Q35-4B`, `Q35-9B`, `Q38-27BU` (the Qwen
SOURCE), `Q38-27B` (base, control/teacher only), `K3-L<i>`, `DK3-<rung>`, `H3`,
`ACE`, `QIMG`, `HY3D`, `QVL32`.

Every record carries `stage`, `parent_code`, `lineage[]` rooted at the parent's
provenance digest, the recipe digest, and the directory digest.

## R2. Clone before any training. No exceptions.

Before **any** optimiser step at stage D or T:

1. the source checkpoint is **sealed and reviewed** — its verdict is in the
   ledger. `oc_checkpoint.py clone` refuses an unreviewed source; the override
   `--allow-unreviewed-source` exists so that an exception is a recorded
   argument and not a silent default;
2. a **byte clone** is written (`--mode copy`, real bytes, no reflink), its
   directory digest is recomputed and **asserted equal** to the source's sealed
   digest, and a sealed `clone-receipt.json` records
   `{source, source_directory_sha256, clone_directory_sha256, created_at,
   stage_about_to_run}`;
3. the clone is `chmod -w` and registered with status `cloned`. **Clones are
   never deleted while any descendant exists**;
4. training writes to a **new** directory, never in place. A stage-D or stage-T
   registration whose `--parent-checkpoint-id` is not a clone is refused.

**A clone record can only be produced by `clone`.** `register --stage clone` is
refused outright: it would have sealed `status=cloned` with no byte copy, no
digest assertion and no receipt, and R2 would then be satisfied by a name. And
the check on a training parent is not the word `clone` in its record — the
registration re-opens that clone's sealed receipt and requires the chain to open
with a `clone` event, the receipt to seal to the digest the record names, to be
*for* that clone, and to carry `source_directory_sha256 ==
clone_directory_sha256`. The default clone destination is `$OC_ROOT/clones/`,
never beside the source, so cloning a registered parent cannot drop a full-size
copy into the model library.

Rollback is the clone, by digest. A training run whose clone receipt is missing
or whose digests differ is refused before the first step.

*Hardlink clones (`--mode hardlink --allow-hardlink`) are supported for cheap
snapshots but are NOT a pre-training clone: they share inodes, so a trainer that
writes in place destroys both copies at once. The clone record says which mode
was used.*

## R2b. A fold must name what it folded

The **≥4x rule is per component**, so a fold that names no component has not
been checked against it at all. `register --stage fold` therefore requires at
least one `--component NAME:PARENT_PARAMS:CHILD_PARAMS[:TENSOR_PREFIX]`, and:

* with a `TENSOR_PREFIX`, both counts are **re-derived** from the parent's and
  the child's safetensors headers and a disagreement is a refusal. Without one
  they stay an operator assertion, and the record, the review package and the
  scorecard all say `asserted` rather than `derived`;
* a fold whose child is **not smaller** than its parent is refused (that is a
  fold that stored everything, and it used to register clean at net_ratio 1.0);
* the components may not **claim a bigger saving than the model actually lost**:
  `Σ(parent − child)` over the components is bounded by
  `parent_total − child_total`, both read from the headers. A build that
  legitimately adds parameters elsewhere states the difference with
  `--accounting-slack N`, which lands in the sealed record as an argued
  exception instead of a silent tolerance.

Two checkpoints may not share, or nest inside, one directory: R4(iv-b) follows
parent and clone links only, so a second id on the same bytes would let a DROP
of one delete the other's weights.

## R3. Checkpoint each thing

Each stage end — a fold rung, a distill stage, a finetune stage — produces a
sealed child **plus dossier plus verdict** before the next stage starts. No
stage starts from an unsealed checkpoint. Quantization is forbidden at the fold
and distill stages (the registry refuses `--quantization` anything but `none`
there); Q4 happens exactly once, at the very end, and the quantized artifact is
re-graded on every surface as its own rung.

## R4. Delete bad checkpoints — the registry is the only path

`oc_checkpoint.py delete` prints all **fourteen** conditions with PASS/REFUSE and
does nothing without `--confirm`. The whole command — conditions, `rmtree` and
the sealed delete record — runs under this checkpoint's chain lock, and the
delete record and its ledger row are BUILT before anything is unlinked, so every
way the commit can refuse refuses with the weights still there. A checkpoint may
be deleted **only if all** hold:

| # | condition |
|---|---|
| i-a | current status is `bad` (`unmeasured` is a re-run, never a delete) |
| i-b | a review record exists (the bytes carry no claim the text does not) |
| i-c | canonical signal lanes = **0 on every surface** (bank v1, bank v2, media-out) |
| i-d | no non-empty media artifact is awaiting Cyril's blind 1–5 |
| i-e | at least **one canonical CONTENT miss** was measured. A loop, a cap-ride, an empty generation or a missing EOS measures nothing either way and may never authorise a delete |
| ii | it is not a floor/frontier holder — *a floor whose child has been deleted is a floor that can only be cited and not checked* |
| iii-a | the DROP names an exact one-pass rebuild command **and** the retained source it rebuilds from |
| iii-b | that retained source is a registered checkpoint or a path that exists |
| iii-c | the reason is argued, not asserted (≥ 60 characters; in practice a paragraph) |
| iv | it is not a clone and not a parent |
| iv-b | nothing in the registry descends from it (no child names it as parent or clone source) |
| iv-c | its directory is not under a protected prefix — `/mnt` (the CIFS share) or `/home/cereal/models` (the source model library). `register` already refuses these, but a record sealed before that guard existed still names its directory, and this clause is the last stop before `rmtree` |
| v | its evidence is already **packaged**, the package's `MANIFEST.json` verifies, **and the package is this checkpoint's**: `manifest.checkpoint_id` is this id, `manifest.record_sha` is a receipt in this chain, and `manifest.package_sha256` equals the `package_sha256` the review committed to. (`verify_manifest` alone is self-referential — it recomputes the digest from the manifest's own file list, so any well-formed package, including another checkpoint's, verifies) |
| vi | the directory on disk still digests to what the sealed record describes |

What is deleted is the **bytes**. The record chain, the ledger rows, the
dossier, the per-case files, the receipts and the package survive, and the
delete record carries the rebuild command and the retained source. Once per
wave one dropped child is rebuilt from its receipt and its digest compared
(the LFM2 campaign's "retention promise cashed" control).

Default retention is roughly the best five per axis; any deviation is argued in
the record's reason, in prose, not implied by a count.

## R5. "At least partially capable" → package, never delete

**Proposed decision line for Cyril:** a child is *at least partially capable* if
it holds **≥ 1 canonical signal lane on ANY surface** — the frozen 15-case pack,
bank v2, or a media-out artifact with non-zero content. The floor is the
`--partially-capable-floor` argument, so re-ruling it is one flag and not an
edit.

Canonical (three-box) reading, Cyril's standard, carried unchanged from
`ACTUAL_MATRIX_2026-08-28.md`: every lane goes in exactly one box —
**intelligence signal** (sealed Useful/Partial, or a sealed miss whose text
contains ≥1 correct, specific, usable element in the child's own words) counts
FOR; **artifact-miss** (empty, pure loop, cap-ride with no unique content,
indiscriminate prompt copy, token salad) counts for NOTHING; **content-miss**
(readable, on-register, wrong / vacuous / deflecting / restatement) is the only
miss that counts AGAINST. Loops, echo, cap-rides and a missing EOS never count
against intelligence.

Such a child is KEPT, packaged (R7) and queued in `REVIEW_QUEUE.md`.

The same lens decides the other side of the line: **zero signal is only a grade
if a content miss was measured.** Zero signal with nothing but artifact misses
is `unmeasured` (R8) — a re-run, never a delete.

## R6. Never delete

A parent; a provenance-verified local copy of a source model; a clone; a GGUF
export named in a ledger row; anything on `/mnt/ai_data` or anywhere else under
`/mnt`; anything in the source model library `/home/cereal/models`.

The registry refuses the first three by rule. The path rules are enforced too,
not left as an instruction: `PROTECTED_PREFIXES = ("/mnt", "/home/cereal/models")`
in `oc_checkpoint.py`, checked by path PARTS (so `/mnt/ai_data_scratch` is under
`/mnt` and is refused, while `/home/cereal/omni-campaignX` is not under
`/home/cereal/omni-campaign` and is not) at three points:

* `register` refuses a `--directory` under one, **before** `directory_digest`
  walks it — one typo used to make register and `verify --deep` byte-hash the
  CIFS share at under 3 MB/s;
* `clone` refuses a destination under one;
* `delete` carries it as R4 clause (iv-c), which is the last stop before
  `rmtree`: a record sealed before the guard existed still names its directory.

The single exception: `--stage parent` may be registered under
`/home/cereal/models`, because that is exactly what a registered parent is, and
`delete` refuses stage `parent` outright (clause iv) so nothing under the library
is reachable by an unlink. The share has no exception.

A GGUF export named in a ledger row is still outside the registry's reach and
remains a standing instruction to every operator.

## R7. The review package

Built by `oc_package.py build` for one checkpoint, into
`OC/oc<NN>/human-review/<checkpoint id>/` and mirrored into the repo (never left
only on the server). It contains the sealed record chain and its ledger rows,
the dossiers, receipts, verdict, per-case files, build receipt, clone receipts,
the recipe, `READ-*.txt` (child and parent, every case, full verbatim), decoded
media with machine diagnostics, `REVIEW_<TAG>.md` (the seven-section form), a
one-page `README.md`, and a `MANIFEST.json` digesting every file plus a
`package_sha256`.

Cyril's scoring comes back as `cyril-score-<TAG>.json`
`{tag, date, per_case_overrides, media_scores, ruling_text}` and is sealed as a
new record with `--cyril-ruling`. **His read governs**; the machine columns stay
beside it as diagnostics.

## R8. Statuses, and who decides them

| status | meaning | set by |
|---|---|---|
| `candidate` | registered, not yet reviewed | registration |
| `partially_capable` | ≥1 canonical signal lane on any surface (R5) | computed from the review counts |
| `unmeasured` | 0 canonical signal **and** 0 content misses — every miss was an artifact, so nothing was measured. Not deletable by any path; re-run it | computed from the review counts |
| `bad` | 0 canonical signal on every surface **with at least one content miss** measured | computed from the review counts |
| `cloned` | a frozen pre-training clone | the clone command |

`unmeasured` is the artifact-tolerant rule made structural. "No signal" and "no
measurement" are not the same finding, and the second one must never authorise
an irreversible `rm -rf`: a child that looped on every lane was not graded, it
was not run properly.

The status is **computed**, for the same reason the LFM2 behaviour verdict is:
the grade is a judgement of text, but the line the grade falls on must not move
after someone sees which side it landed on. `--status` may only assert
agreement. The single exception is Cyril's own ruling, which overrides the
counts and is recorded with his words in `--cyril-ruling`.

## R9. The ledger

One append-only authority, `OC/ledger.jsonl`, carrying two row schemas:
`neural-forge-lfm2-matrix-cell-ledger-v1` (ARCH children and behaviour verdicts
from the ported LFM2 sealers) and `neural-forge-omni-checkpoint-ledger-v1` (this
registry's register / clone / review / status / delete events). Both carry
`campaign = "omni-2026-09"`; the ancestor campaign's 1,056 rows (sha256
`d2027831e229d763a251ca602655c729a4432645bed43a9ba2e4a8824ef211ea`) are **not**
copied in — they are cited.

Rules: rows are canonical JSON bytes plus `\n`, appended in binary under an
exclusive writer lock; rows are never rewritten (a correction is a new record
instance); an append or a merge onto a file that does not end in a newline is
refused (it would splice two rows onto one line); a merge is **idempotent** —
a row whose `(instance, receipt_sha)` is already in the authority is skipped and
counted, and a shard row that reuses an authority instance under a *different*
receipt is refused — and after `--apply` the file is re-read and its first
`len(before)` bytes must still be the bytes that were there; `verify` re-reads
every row and refuses one that is not canonical bytes, which is how a repo
mirror that drifted from the server authority is caught.

The ledger is checked in **both directions**. `oc_ledger.py verify --registry`
walks rows and resolves them to records; `oc_checkpoint.py verify` walks records
and asserts each one has its row. A record whose append died (lock timeout,
Ctrl-C, ENOSPC) is an orphan only the second direction sees, and
`oc_checkpoint.py repair-row` re-derives the missing row from the sealed record
— it repairs the index, never the evidence.

## R9b. The registry's own housekeeping

Nothing that is not a sealed record may sit where a chain is read. A chain is
exactly the files matching `<id>-NNNN-<event>.json` in `registry/<id>/`; an
in-flight delete marker lives in `registry/.pending/`, a chain lock in
`registry/.locks/`, and both are ignored by every reader. (An interrupted delete
used to leave a marker inside the record directory, where `glob("*.json")`
matched it, and every later read raised — for the whole registry, not just that
checkpoint.) A mutating command takes its checkpoint's chain lock, re-reads the
head inside it and refuses if another writer moved it, so two commands cannot
both claim the next sequence number.

`verify` REPORTS a surviving `.pending` marker as a problem: an interrupted
delete means the bytes it names may be half-gone.

## R10. Two dashboards, regenerated, never typed

`oc_dashboard.py` writes `SCORECARD.md` (every checkpoint by stage, ordered by
canonical signal per unit of ratio) and `REVIEW_QUEUE.md` (what is waiting on
Cyril, what media is unscored, what is bad and awaiting retention, and which
clones exist). Both are outputs. Neither is an input to anything.

---

## Open decisions for Cyril (each a proposed line, none a gate)

1. `partially_capable` floor = **≥1 canonical signal lane on any surface**.
2. Column **B** (copy-backs refused) as the default intelligence column, with A
   always published beside it.
3. `PARENT_BAND_TOLERANCE = 2` carried from LFM2 vs re-measured with two readers
   on Qwen3.5-4B; `DEGRADED_BAND = ceil(parent band / 2)`.
4. Default retention count (~best five per axis) and whether a deviation needs
   his sign-off or only an argued reason in the manifest.
5. Whether an unreviewed source may ever be cloned
   (`--allow-unreviewed-source`), or whether that override should be removed.
6. Whether hardlink clones are allowed at all outside pre-training snapshots.
