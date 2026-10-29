# Inventory snapshot, 2026-10-22-foldrec (the Neural Forge folding record)

The complete per-arm record of the folding work (May to October 2026). Extraction agents of the Omni Lab review
(Claude session 0c189f21) each read one group of sources: Neural Forge notes, docs, reports and receipts, git
history and code, agent_memory events (Claude and Codex prompts, tool output, stops), Codex rollouts, Claude Code
transcripts, workflow task outputs and AI-server run directories. Each agent wrote one shard (schema `foldrec-v1`):
every fold arm, child or attempt it found, with path + lines, a commit or an event id for each record, plus
findings, techniques and Cyril's verbatim directives. Negatives, closed paths and failed launches are kept as
records. Checker passes then re-read the sources adversarially and stamped a `check` object
(`confirmed` / `corrected` / `added` / `unverifiable`) on the records they verified.

The folder name is the start date. The first 23 shards were written on 2026-10-22, and later passes (gap1,
gap2, gap3 and re-reads) wrote into the same folder until 2026-10-28T22:35Z (`extracted_at`). The last file
change was at 2026-10-28T23:18Z. `ingest/foldrec.py` reads every `*.json` here
(`omni_lab.py ingest foldrec [--dry-run]`). The ingester never edits these files. A newer extraction goes in a
new dated folder.

**Only this README and `SHA256SUMS` are in git.** The 86 shard files (56,706,381 bytes) are not committed. Two
things keep the ingested bytes recoverable: `fold_shard.src_sha256` in the database holds each file's sha256 at
ingest time (identical to the table below), and `fold_arm.record` keeps every raw shard record verbatim. Check
the folder with `sha256sum -c SHA256SUMS`.

## Ingested into production omni_lab, 2026-10-28

| Batch | When (UTC) | Ingester | Result |
|---|---|---|---|
| 272 | 23:26:42 to 23:44:05, 1,043.7 s | `ingest/foldrec.py` @ bef42a0 | 141,007 rows written: experiment 7,878, fold_arm 7,895, fold_component 12,163, fold_behaviour 6,600, fold_diagnostic 38,600, fold_source 32,865, experiment_alias 7,886, experiment_link 2,755, fold_finding 2,947 (+ 5,324 experiment refs, 3,653 sources), fold_shard 86, fold_coverage 6,149, technique 5,760 new + 1 updated, directive 444 new + **1 overwritten (bug, see below)** |
| 273 | 23:54:50 to 00:20:44, 1,554.6 s | `ingest/foldrec.py` @ 44bff5d | 123 rows written, all of them directives: 9 inserted, 112 verbatims corrected, 2 stale rows deleted. Every other table: 0 inserted, 0 updated. This is the idempotency proof. |

The shard files were byte-identical (all 86 sha256s) from the first dry run (23:20Z) to the end of batch 273.
Backup taken before the ingest: `omni_lab_20261028T231600Z` (Lexar `/mnt/lexar/omni-campaign/backups/omni_lab/`,
sha256 `47b5e094578c811f90e6d39e26d686b498f1f8d8f8216a3056cac069ee76925a`, restore test exact at 158,909 rows).

**Bug found and fixed in batch 272.** `ingest/directives.py` and foldrec both key a quote as `ev:<id>:<sha8 of quote>`.
foldrec split Cyril's own `dense....` on the dots, so it did not recognise directive 93 (`ev:605703:0f1939e0`,
batch 24) as the same words and overwrote it. Row 93 was put back by a parameterized UPDATE, using the values in
the backup above. Commit 44bff5d changes three things. A key another ingester holds is skipped as `key-taken`. A
quote is matched whole before it is split on `...`, which gives 112 foldrec rows Cyril's exact dots back. foldrec's
own directive rows that no shard wants any more are deleted (2 in batch 273), unless something references them.

After batch 273:

- 7,895 arms. 0 duplicate keys. There are 16 case-only key pairs (32 arms, e.g. `a0011` / `A0011`), and they stay
  separate because dedup is by exact key.
- 2,951 arms were described by more than one shard. 1,441 of them carry kept disagreements in
  `fold_arm.conflicts`. Where shards disagree on a number or on verdict_class, the column is left NULL (verdict_class
  419, ratio_net 405, axes 202, ratio_body 2).
- 17 arms attach to experiments that other ingesters wrote: 11 foldhist, 6 proof_eval. All 43 foldhist
  experiments are linked: 11 with an attached arm, 42 through `experiment_link`. That is 1,785 link rows, plus
  131 finding references.
- 451 directive rows came from foldrec. Of the 1,346 shard directives, 293 merged under the same key, 311 are
  contained in a longer quote, 137 were already present in another row, 51 were key-taken, 57 were not found in
  their source, 24 had an unreadable source, 20 had no source and 2 were not a user prompt. No two directive rows
  in the table hold identical verbatim text.
- Orphans: 0 component, behaviour, diagnostic, source or coverage rows without an arm. 2,067 arms have no
  component, and 4 finding experiment keys resolve to nothing.
- `omni_lab.py db check`: schema 7, all 7 migrations ok.

## Checker stats (from the shards)

| Records | Total | With a `check` | confirmed | corrected | added | unverifiable |
|---|--:|--:|--:|--:|--:|--:|
| experiments | 13,773 | 4,129 | 2,493 | 1,546 | 87 | 3 |
| findings | 2,947 | 1,194 | 1,072 | 119 | 1 | 2 |
| directives | 1,346 | 869 | 713 | 59 | 87 | 10 |
| techniques | 1,467 | - | - | - | - | - |

`corrected` means the checker re-read the source and changed the record. The checker's note on the record
says what changed. Ten shards also carry a shard-level checker block (`checker`,
`checker_note`, `checker_previous` or `checked_at`): docs-aug-campaign, docs-june-july-early, docs-lfm2-july,
gap1-mx-wave-artifacts-unopened, gap1-mx-wave-notes-p-w, gap1-run0-lateoct-chimera-repo-misc,
gap2-oct13-recovery-prereg-amendments-geometry, gap3-claude-753087a5-38b5797d-oct17-18-sessions, mem-matrix-1 and
server-surgery-lab. Six shards have no record-level checks at all: gap2-calibration-out-parent-baselines,
gap2-composite-c10-detail, gap2-composite-s7-s8-sep-receipts, gap2-partial-md-completions,
gap2-recovery-oct13-oct15-evidence and gap2-reports-machine-artifacts-and-hidden-dirs (467 experiments, 163
findings, all extracted 2026-10-22T17Z).

Caveats carried from the shards:

- 42 soft problems. The enum values are outside `foldrec-v1`: `child_exists_now` = `no` (15), `n/a` (16) or
  `superseded` (2); `behaviour.scorer` = `unknown` (7); `objective_target` = `logits` (1). There is also one date
  that is a month only (`omnicoder-20B`: `2026-05`). Each of these columns is NULL and the raw value stays in
  `fold_arm.record`.
- 827 arms have no exact date, so `experiment.date` is NULL. `date_text` keeps what the source says.
- An arm's behaviour row is the score as the source wrote it. Only 11 behaviour rows are scored by Cyril
  (scorer `cyril`). Most are scored by `claude` (2,775) or `machine` (2,006). Cyril's own words about an arm are
  in `fold_arm.cyril` (449 arms).
- Coverage: 5,997 paths that arms cite were not declared in any shard's `sources_read`. Shards often declared a
  directory or a glob rather than each file.

## Shards

Totals: 86 files, 13,773 experiment records, 2,947 findings, 1,467 technique entries, 1,346 directives, 5,128
sources read, 1,021 sources not read (each with a reason). The check columns count records with a `check` object,
by status. The sha256 values are those of the files here, LF line endings, and they equal `fold_shard.src_sha256`.

| Shard file | extracted_at | Experiments | Findings | Techniques | Directives | Sources read | Not read | Experiment checks | Finding checks | Directive checks | Bytes | sha256 |
|---|---|--:|--:|--:|--:|--:|--:|---|---|---|--:|---|
| `codex-reviews-misc.json` | 2026-10-28T11:23:13Z | 60 | 21 | 15 | 11 | 39 | 13 | 46 (37 confirmed, 9 corrected) | 15 (15 confirmed) | 11 (9 confirmed, 2 corrected) | 202,004 | `f3fd4ba5a73f4c7d33156d432e58c4f155f47eb3496ca721b03136eca40a7ab8` |
| `cyril-scores.json` | 2026-10-28T11:20:40+00:00 | 44 | 21 | 6 | 180 | 50 | 7 | 44 (27 confirmed, 17 corrected) | 21 (17 confirmed, 4 corrected) | 180 (89 confirmed, 87 added, 4 corrected) | 393,052 | `e6f58e44e8c5748ae0d87a827ab6f92b8be238318cf9f83989c3bd6ac87a4ef4` |
| `docs-aug-campaign.json` | 2026-10-28T06:36:57.462899+00:00 | 544 | 283 | 124 | 105 | 29 | 3 | 44 (27 confirmed, 16 corrected, 1 unverifiable) | 6 (6 confirmed) | 8 (8 confirmed) | 1,285,085 | `9c2d9b3207682c90aa9ed0d376727dbc505092ae243ab81c48b6a6b26477f222` |
| `docs-crosswidth-kernel-modality.json` | 2026-10-28T06:21:38Z | 69 | 39 | 26 | 14 | 77 | 9 | 26 (25 confirmed, 1 corrected) | 18 (18 confirmed) | 14 (14 confirmed) | 363,020 | `1d29c46f8d681035f79e536706f4448d49cbc7090e8803b2f566bbedc1ef98d1` |
| `docs-june-july-early.json` | 2026-10-28T06:10:30+00:00 | 419 | 76 | 66 | 48 | 24 | 3 | 97 (65 confirmed, 32 corrected) | 27 (27 confirmed) | 48 (48 confirmed) | 1,115,141 | `538fdfe823ad3b93f20ee83a57a1d44dc5b9c588f4823df0056ef48a8f62fbd2` |
| `docs-lfm2-july.json` | 2026-10-28T06:09:58+00:00 | 192 | 53 | 31 | 22 | 38 | 2 | 192 (163 confirmed, 29 corrected) | 31 (30 confirmed, 1 corrected) | 22 (22 confirmed) | 618,969 | `cd4009e1619f71c851ad34ac425708223c64db649ecc1d893c48a75ca3877ce3` |
| `docs-residual-attention.json` | 2026-10-28T06:20:36+00:00 | 155 | 18 | 14 | 5 | 35 | 3 | 26 (15 confirmed, 11 corrected) | - | - | 400,982 | `064be81909c89ca9a4063582bb32ffe65d7567cebe5deb6fc15037522375e346` |
| `gap1-child-deletion-cleanup-records.json` | 2026-10-28T13:00:46+00:00 | 1,493 | 172 | 0 | 10 | 36 | 5 | 341 (202 confirmed, 139 corrected) | - | - | 8,233,834 | `4f76d6e4e76fb41adfeed427427c1e153df94d5a45fa47a5730342abe4d84a50` |
| `gap1-claude-0c18-workflows-tasks.json` | 2026-10-22T20:45:03+00:00 | 47 | 22 | 5 | 10 | 77 | 18 | 39 (23 confirmed, 16 corrected) | 15 (14 confirmed, 1 corrected) | 10 (10 confirmed) | 202,817 | `5cb19721a33abe5bb90f195d2d363058244570a687be2bcbcbc2510b823bfa55` |
| `gap1-claude-a763-workflows-subagents.json` | 2026-10-22T20:52:32Z | 0 | 21 | 48 | 0 | 343 | 38 | - | 21 (20 confirmed, 1 corrected) | - | 221,894 | `210d50a4f47824fded1647b89172171785fac956c269f940b1b5a00481aafa72` |
| `gap1-codex-oct-workspace-a.json` | 2026-10-22T16:17:34.148519+00:00 | 46 | 10 | 11 | 6 | 34 | 5 | 18 (13 confirmed, 5 corrected) | 5 (4 confirmed, 1 corrected) | 6 (4 unverifiable, 2 confirmed) | 168,267 | `3bf44fa0f91e27868a73f95a614d65282873411da30d385dd3f5cc7cb0bb6891` |
| `gap1-codex-oct-workspace-b-and-claude-tasks.json` | 2026-10-22T16:26:25Z | 81 | 21 | 14 | 23 | 51 | 8 | 28 (17 confirmed, 11 corrected) | 11 (10 confirmed, 1 corrected) | 23 (23 confirmed) | 292,677 | `4901445f3d4d313d36306a47b5d8e7bb7c681872236d915f167bf8c79caa011c` |
| `gap1-codex-rollouts-aug-oct.json` | 2026-10-22T21:07:23+00:00 | 26 | 15 | 9 | 10 | 6 | 7 | 26 (21 confirmed, 5 corrected) | 15 (14 confirmed, 1 corrected) | 10 (9 confirmed, 1 corrected) | 135,542 | `74065393d5418cd0f2540c200b3893be328696a89445f5591b7a3980f68d1e59` |
| `gap1-codex-rollouts-jun-jul.json` | 2026-10-22T20:50:32+00:00 | 35 | 12 | 6 | 9 | 9 | 4 | 34 (31 confirmed, 3 corrected) | 9 (9 confirmed) | 9 (9 confirmed) | 120,725 | `bcbe1961074466cae755ea49dfeccaa519fd9e01c31200df02b13af08e8c6b57` |
| `gap1-codex-sep07-lexar.json` | 2026-10-22T16:22:11+00:00 | 30 | 13 | 9 | 3 | 23 | 4 | 11 (10 confirmed, 1 corrected) | 13 (12 confirmed, 1 corrected) | 3 (2 corrected, 1 confirmed) | 123,309 | `2cbfedf0facf70b063e548408e9b97beb00e0fa18bcadcda6b7dcd1207d507e7` |
| `gap1-cyril-prompts-uncited.json` | 2026-10-28T12:46:02+00:00 | 8 | 9 | 8 | 23 | 20 | 4 | 8 (5 corrected, 3 confirmed) | 9 (7 confirmed, 2 corrected) | 23 (23 confirmed) | 71,444 | `4848452996e5644db3cf570220e9af853467b55e923c26db0fda8d7475c39099` |
| `gap1-d1-recovery-attempts-k3b2-audit.json` | 2026-10-28T12:49:32+00:00 | 44 | 7 | 5 | 1 | 23 | 7 | 27 (14 confirmed, 13 corrected) | 7 (5 confirmed, 2 corrected) | 1 (1 corrected) | 248,616 | `4f352fbc99dae00caceb68946433d4896e00dc6d81119674058e77e1da43f380` |
| `gap1-lexar-largefoldopt-oct10-scratch.json` | 2026-10-28T12:46:51.153220+00:00 | 43 | 10 | 7 | 4 | 22 | 5 | 40 (36 confirmed, 4 corrected) | 10 (6 confirmed, 4 corrected) | 4 (3 unverifiable, 1 confirmed) | 165,025 | `3bc41648655e4e8465a18df30377ec6701e3dbb7093420f52982ff9227ed1f4a` |
| `gap1-mx-human-review-dossiers.json` | 2026-10-22T16:16:40+00:00 | 35 | 30 | 10 | 16 | 50 | 2 | 35 (34 confirmed, 1 corrected) | 30 (26 confirmed, 4 corrected) | 16 (14 confirmed, 2 corrected) | 431,930 | `3d22d63659facdb49e71e35f8dfff7a8457744e660139d169d8317cb6fb87093` |
| `gap1-mx-probes-arma-ledger-snapshots.json` | 2026-10-28T12:53:56Z | 110 | 37 | 17 | 4 | 117 | 3 | 110 (83 corrected, 27 confirmed) | 23 (21 confirmed, 2 corrected) | 4 (4 confirmed) | 498,368 | `6658e2837e440190a8c07059002e2e63c84985599c9db622ccf0435a01e0aa12` |
| `gap1-mx-wave-artifacts-unopened.json` | 2026-10-22T20:46:37Z | 371 | 82 | 8 | 1 | 323 | 272 | 18 (14 confirmed, 4 corrected) | 7 (4 corrected, 3 confirmed) | 1 (1 corrected) | 2,683,194 | `8906fe82e5a7a097978cc513956d82a18dcc287d55de82d8b4e618cee82474c1` |
| `gap1-mx-wave-notes-i-o.json` | 2026-10-22T16:14:37+00:00 | 210 | 58 | 15 | 10 | 16 | 3 | 84 (71 corrected, 13 confirmed) | 6 (6 confirmed) | 10 (6 corrected, 3 confirmed, 1 unverifiable) | 728,027 | `3ef0f6e97ea6318e010e134337d72995a7d1b55c1269ce7c6bbb08b6f1cf78e4` |
| `gap1-mx-wave-notes-p-w.json` | 2026-10-22T16:17:53+00:00 | 229 | 88 | 9 | 14 | 18 | 3 | 23 (18 confirmed, 5 corrected) | - | 1 (1 unverifiable) | 754,364 | `f584726c8800d687ab22f786d64dc3e46cb02a1a9a0e92e2bf72837740880f8b` |
| `gap1-nf-branch-only-docs.json` | 2026-10-22T20:47:44+00:00 | 27 | 25 | 14 | 12 | 140 | 3 | 19 (15 confirmed, 4 corrected) | 20 (20 confirmed) | 12 (9 confirmed, 3 corrected) | 173,475 | `0b58da7df8f65d0e1414c2d49b737ae9d20bcdf920dee815255d54f662f142f9` |
| `gap1-nf-local-staging-runs-codextmp.json` | 2026-10-22T20:58:24.164864+00:00 | 154 | 12 | 8 | 9 | 539 | 7 | 41 (32 confirmed, 9 corrected) | 11 (10 confirmed, 1 corrected) | 9 (9 confirmed) | 641,275 | `3b77fd045590c802047e0818f3caea719cd2717f08dfd58a8de0d3ec99d0c783` |
| `gap1-omni-media-fold-lane.json` | 2026-10-22T16:24:50+00:00 | 123 | 17 | 11 | 9 | 48 | 4 | 27 (15 confirmed, 12 corrected) | 12 (12 confirmed) | 9 (9 confirmed) | 546,847 | `2e76c10531111cf44cd0686cb3cd88c97c69b9773ac84033c26ec18d3c329990` |
| `gap1-omni-oc02-sheets-phase0.json` | 2026-10-22T16:20:40Z | 14 | 24 | 10 | 10 | 36 | 5 | 14 (8 confirmed, 6 corrected) | 24 (23 confirmed, 1 corrected) | 10 (10 confirmed) | 167,429 | `7c7233e3478e0bc7dca62eac79d107959d9ba0b632ef6fa296f9aa47d5e5fc90` |
| `gap1-omni-qwenfold-oc01.json` | 2026-10-22T16:22:54Z | 42 | 16 | 10 | 6 | 27 | 5 | 21 (16 corrected, 5 confirmed) | 11 (6 confirmed, 5 corrected) | 6 (6 confirmed) | 239,162 | `eefff8aacacfc600257eacc05b59727e55f78d168398a3917aedb70666ab2deb` |
| `gap1-omni-residual-docs.json` | 2026-10-22T20:42:27Z | 38 | 22 | 9 | 11 | 50 | 5 | 33 (27 confirmed, 6 corrected) | 12 (10 confirmed, 2 corrected) | 11 (8 confirmed, 3 corrected) | 178,997 | `f4bb87ae48786e39316d59d1def023e89a009be8ee83058315f10502ccb693ea` |
| `gap1-omnilab-db-foldhist-reconcile.json` | 2026-10-28T13:01:57Z | 1,871 | 21 | 3 | 5 | 15 | 5 | 23 (12 corrected, 11 confirmed) | 13 (13 confirmed) | 5 (4 confirmed, 1 corrected) | 3,462,346 | `062e279f6938275caaa6d6698c0e5efa8015cf2bdeda94e78f671155bdfc38bd` |
| `gap1-pg-toolstream-jul-aug-residual.json` | 2026-10-28T13:01:16.458968+00:00 | 269 | 15 | 2 | 0 | 6 | 4 | 269 (223 corrected, 29 confirmed, 17 added) | 14 (8 confirmed, 4 corrected, 1 unverifiable, 1 added) | - | 1,070,495 | `dbacef2260747477edb6a15a6268128c84344a1d4751a3c7bb4787e667aabd0e` |
| `gap1-run0-lateoct-chimera-repo-misc.json` | 2026-10-28T12:49:02Z | 107 | 32 | 14 | 37 | 52 | 6 | 45 (23 confirmed, 22 corrected) | 6 (5 confirmed, 1 corrected) | - | 350,042 | `6c624fb80dfb94e6a8600149ec49938c439ddd1492d6fb2c6598130ce1daf786` |
| `gap1-srv-omni-k3-attempt-dirs.json` | 2026-10-22T20:48:53+00:00 | 56 | 8 | 3 | 0 | 16 | 5 | 24 (23 confirmed, 1 corrected) | 8 (7 confirmed, 1 corrected) | - | 231,967 | `50f63bf4e34af1b8ebb13c27c8214d0ca1dbffdd259d669705826d34dc2aa1cf` |
| `gap1-srv-omni-native-lane-dirs.json` | 2026-10-22T20:56:16.656192+00:00 | 115 | 9 | 3 | 1 | 115 | 7 | 34 (19 confirmed, 15 corrected) | - | 1 (1 corrected) | 331,394 | `1e2419136416d74e76249931e48a7fbc3d39ec5599c1019f5b325cdc52e7d69d` |
| `gap1-surgerylab-july-failed-launches.json` | 2026-10-28T12:58:15Z | 45 | 23 | 6 | 4 | 28 | 6 | 45 (38 confirmed, 6 corrected, 1 unverifiable) | 23 (19 confirmed, 3 corrected, 1 unverifiable) | 4 (4 confirmed) | 235,327 | `9afeb19c2ae58ad77271524e176345d9f5aab42382e41c627bdc470ab274a294` |
| `gap2-calibration-out-parent-baselines.json` | 2026-10-22T17:23:43Z | 29 | 17 | 15 | 9 | 85 | 6 | - | - | - | 147,719 | `ea94834c42b2461efcba88f017bb11d9da9ea670994871093253d3677267903a` |
| `gap2-codex-oct06-08-child-archive-retirement-records.json` | 2026-10-28T20:10:09Z | 29 | 10 | 2 | 5 | 57 | 5 | 15 (12 confirmed, 3 corrected) | 7 (6 confirmed, 1 corrected) | 5 (5 confirmed) | 163,427 | `d98d5b68df6bc09a9f818942089a31568ae927fa640a083f6d5d5d7c97203bc9` |
| `gap2-codex-task4-oct07-fold-reviews-negatives.json` | 2026-10-28T20:02:45Z | 55 | 41 | 15 | 1 | 93 | 16 | 55 (52 confirmed, 3 corrected) | 41 (38 confirmed, 3 corrected) | 1 (1 confirmed) | 332,530 | `7af93bfa72ba4ec2fe97ca2615c212e5ca6edc20703251883838d633a3248751` |
| `gap2-codex-task4-oct08-native-resumption-k3-lanes.json` | 2026-10-28T20:17:08.246594+00:00 | 79 | 8 | 3 | 13 | 16 | 9 | 19 (16 confirmed, 3 corrected) | 6 (5 confirmed, 1 corrected) | 13 (13 confirmed) | 239,395 | `5afcf1b3b91fada8e01d44ec2b8852c44bd74a07631dbfd2bbd469f037b8b61c` |
| `gap2-composite-c10-detail.json` | 2026-10-22T17:17:44+00:00 | 120 | 23 | 8 | 3 | 97 | 4 | - | - | - | 742,543 | `d5acfc10aad7b70a9f0b383849528d85c6ad96b718c7646a34972261d2f6efc4` |
| `gap2-composite-s7-s8-sep-receipts.json` | 2026-10-22T17:21:00Z | 63 | 14 | 12 | 5 | 240 | 8 | - | - | - | 251,711 | `3fdef064e54479b843617bd15bc63e8bb9262d8fe1f43d4cfe9c03979dc43a1f` |
| `gap2-lateoct-rulings-k3ratify-nascleanup-events.json` | 2026-10-28T19:57:53Z | 26 | 13 | 4 | 4 | 16 | 4 | 26 (13 corrected, 13 confirmed) | 13 (11 confirmed, 2 corrected) | 4 (4 confirmed) | 177,460 | `a0c0ae4e56d281ea71a87950cd0c4cb17f764a59bb37ae28138a0e092c18df5a` |
| `gap2-mx-orphan-rebuild-arms-and-planned-cells.json` | 2026-10-28T14:36:36Z | 40 | 20 | 5 | 1 | 67 | 6 | 40 (37 confirmed, 3 corrected) | 15 (12 confirmed, 3 corrected) | 1 (1 confirmed) | 209,791 | `dd8352a9fb1cdbafcfca63161789755c67cce61761feb7fb659a63725575e890` |
| `gap2-notes-w13h-w13u-and-thin-w13-attempts.json` | 2026-10-28T19:53:44+00:00 | 60 | 27 | 10 | 4 | 16 | 2 | 60 (48 confirmed, 12 corrected) | 12 (11 confirmed, 1 corrected) | 4 (4 confirmed) | 261,191 | `51e853ae231fee0c6882f5bc5f391ca318fddbd1640e98d8657d44c6b3042a27` |
| `gap2-oct13-recovery-prereg-amendments-geometry.json` | 2026-10-28T19:55:56+00:00 | 48 | 14 | 9 | 4 | 126 | 10 | 48 (47 confirmed, 1 corrected) | 14 (13 confirmed, 1 corrected) | 4 (4 confirmed) | 245,347 | `c2da1862db34c86ff7299ce4f9b20c059d4e0c194b60c206daaefe159d78e02c` |
| `gap2-partial-md-completions.json` | 2026-10-22T17:25:04Z | 123 | 77 | 24 | 92 | 49 | 6 | - | - | - | 396,606 | `4721219423c994947d4201b35b401d308d773d3e35b13d37788db8eaa49ce46d` |
| `gap2-pg-codex-tooloutput-metrics-jun26-30-jul16-31.json` | 2026-10-28T14:35:45+00:00 | 31 | 5 | 5 | 1 | 4 | 3 | 31 (23 confirmed, 8 corrected) | 5 (3 confirmed, 2 corrected) | 1 (1 confirmed) | 605,890 | `a5f25db214e2d316aa2854031a798e83fd51f43b894786083cb36013243539b0` |
| `gap2-recovery-oct13-oct15-evidence.json` | 2026-10-22T17:18:56Z | 9 | 12 | 9 | 3 | 293 | 102 | - | - | - | 145,230 | `862d598af7f63c48233a036d36a1ebd13cdf2c98ee93d6df09719dd46232c950` |
| `gap2-reports-machine-artifacts-and-hidden-dirs.json` | 2026-10-22T17:23:52+00:00 | 123 | 20 | 16 | 0 | 84 | 5 | - | - | - | 452,051 | `f2665223b7d20d02543ef55e73f26aa9334a6d669d8d7a96a7827bd43ae85dee` |
| `gap2-srv-chimera-proof-avflow-h3validation-and-reenumerate.json` | 2026-10-28T20:25:33Z | 78 | 7 | 2 | 0 | 35 | 9 | 28 (21 confirmed, 7 corrected) | 5 (4 confirmed, 1 corrected) | - | 208,974 | `3398792ea621a15b49a3a528629acb26c47e14f7fd3b64feae17bd60edeb622b` |
| `gap3-claude-0c18-oct08-21-uncited-task-outputs.json` | 2026-10-28T22:31:33Z | 155 | 12 | 6 | 3 | 71 | 0 | 125 (107 confirmed, 18 corrected) | - | - | 476,508 | `70fe9ab372916b47d17843076a80a16fee2405de56f5dd2266f1322477858f91` |
| `gap3-claude-753087a5-38b5797d-oct17-18-sessions.json` | 2026-10-28T22:34:49Z | 42 | 19 | 10 | 22 | 16 | 3 | 27 (19 confirmed, 8 corrected) | 7 (5 confirmed, 2 corrected) | 22 (21 confirmed, 1 corrected) | 190,150 | `6987e1f03141835b72d8daec2e768f3c7f984c12fa0b26916ed3381b2d8056a2` |
| `gap3-codex-docs-orphan-ids-and-misc-leftovers.json` | 2026-10-28T22:34:52+00:00 | 11 | 11 | 0 | 2 | 34 | 10 | 11 (11 confirmed) | 11 (9 confirmed, 2 corrected) | 2 (2 confirmed) | 65,573 | `28e1a5b3e6e9687d3a6a664eef2ac83bf39c515f192cabd60ad62bf746d0f9a9` |
| `gap3-lateoct-commits-rulings-backups-k3ratify-residual.json` | 2026-10-28T22:27:34Z | 6 | 34 | 5 | 6 | 25 | 4 | 6 (5 confirmed, 1 corrected) | 33 (33 confirmed) | 6 (5 confirmed, 1 corrected) | 79,675 | `85b4adb436a69eed2e27b34a488630a54497a8840cf28bc23692cb0df2f6b63d` |
| `gap3-oct13-recovery-proof-outputs-live-metadata.json` | 2026-10-28T22:28:00.540863+00:00 | 19 | 11 | 5 | 3 | 24 | 24 | 19 (14 corrected, 5 confirmed) | 11 (8 confirmed, 3 corrected) | 3 (3 confirmed) | 215,415 | `395c878e2420e3e9cd174f292fa844d8cfe87a7ddc615ff3e643fe6baee974ea` |
| `gap3-pg-claude-uncited-dense-stops-and-tasknotifs.json` | 2026-10-28T22:27:24Z | 5 | 2 | 0 | 0 | 123 | 1 | 5 (3 confirmed, 2 corrected) | 2 (1 confirmed, 1 corrected) | - | 78,606 | `ffc7bcaded19e05c21b3d90b0539957ae9657052192a4ed5668274aa4b87272d` |
| `gap3-pg-codex-oct-executor-prompts-dot-and-junjul-stops.json` | 2026-10-28T22:29:54+00:00 | 20 | 14 | 8 | 12 | 6 | 3 | 20 (16 confirmed, 4 corrected) | 10 (9 confirmed, 1 corrected) | 12 (12 confirmed) | 100,888 | `2e85334e4aab6305dee157a10bc5926a5f05659ce0d1e084effdc83ecb8f553c` |
| `gap3-srv-distill-tps-probes-and-codex-modality-control-dirs.json` | 2026-10-28T22:35:05Z | 148 | 12 | 8 | 8 | 322 | 127 | 41 (28 confirmed, 13 corrected) | 9 (9 confirmed) | - | 630,480 | `56ebc2f2df326f295b9f128f15c277586b2cabe8ef41244196546bf694cf6f45` |
| `gap3-srv-omnicampaign-k3-media-qf-composite-residual-logs.json` | 2026-10-28T22:27:00+00:00 | 30 | 25 | 7 | 1 | 38 | 5 | 24 (23 confirmed, 1 corrected) | 21 (20 confirmed, 1 corrected) | 1 (1 confirmed) | 150,490 | `69bfb960064772210d36c2abee39c9c6679f16ad689988e70fff6ea482170965` |
| `k3-folds.json` | 2026-10-28T11:26:32Z | 57 | 31 | 10 | 23 | 106 | 16 | 46 (43 confirmed, 3 corrected) | 31 (30 confirmed, 1 corrected) | 23 (20 confirmed, 3 corrected) | 358,189 | `52b7d4c1021e9fd3dc1ffacb6d0aa408caf6ff6399d0f5d23afdee38c6d66b0c` |
| `mem-matrix-1.json` | 2026-10-28T06:14:03+00:00 | 207 | 78 | 66 | 33 | 3 | 2 | 94 (56 confirmed, 38 corrected) | 34 (34 confirmed) | 33 (17 corrected, 15 confirmed, 1 unverifiable) | 604,618 | `ad1b33049c1a1ee13a455b72f59c08bc0b58ceafbef6d3c62b6f22e1195a35d4` |
| `mem-matrix-2.json` | 2026-10-28T06:22:49+00:00 | 67 | 60 | 24 | 7 | 4 | 2 | 56 (37 corrected, 19 confirmed) | 10 (8 confirmed, 2 corrected) | 7 (7 confirmed) | 277,437 | `069318ea904f2d970eeb2455cd44c5bf1c3d346441d9ab66fdd87a60158c60e5` |
| `mem-matrix-3.json` | 2026-10-28T06:17:11+00:00 | 60 | 48 | 23 | 5 | 7 | 1 | 60 (46 confirmed, 14 corrected) | 48 (46 confirmed, 2 corrected) | 5 (4 confirmed, 1 corrected) | 232,216 | `e633e0e18c8aa7b805d35353f661977c8b4c3053817475202a6d91b901e07f3b` |
| `mem-other.json` | 2026-10-28T06:31:57+00:00 | 115 | 46 | 55 | 47 | 32 | 3 | 18 (10 confirmed, 8 corrected) | 4 (3 confirmed, 1 corrected) | - | 316,854 | `28ed39e09685f21fc60b775ea1b69abec2f36485436dd0798d3d3808f3f1d64d` |
| `nf-git-and-code.json` | 2026-10-28T06:39:44Z | 264 | 65 | 43 | 19 | 15 | 6 | 34 (25 confirmed, 9 corrected) | 1 (1 corrected) | - | 563,517 | `e921b145285816418a15ad7af5164ce5c9be5397261a349ac7158d2304d80b03` |
| `notes-early.json` | 2026-10-28T06:00:32.371556+00:00 | 24 | 25 | 12 | 0 | 16 | 5 | 21 (12 confirmed, 9 corrected) | 14 (7 confirmed, 7 corrected) | - | 104,402 | `b76947db9bdcc58e9c7799d7671b86cf4e1c72d42b0691f9202bc9a9dc673d63` |
| `notes-mx-a-h.json` | 2026-10-28T06:06:15+00:00 | 118 | 71 | 14 | 7 | 9 | 2 | 35 (22 corrected, 13 confirmed) | 15 (8 corrected, 7 confirmed) | 7 (5 confirmed, 2 corrected) | 464,000 | `c0bf22254ee28607948bb7111cc02457ce6d534d3bd69906576f77e3ac337ee5` |
| `notes-mx-al-aw.json` | 2026-10-28T06:01:31+00:00 | 110 | 97 | 19 | 4 | 13 | 2 | 51 (41 confirmed, 10 corrected) | 33 (33 confirmed) | 4 (4 confirmed) | 497,023 | `1d07510637e59ee2c94ffa2d9908902b000c1d5963f7dd5cba029310d013bb6b` |
| `notes-mx-xyz-aa-ak.json` | 2026-10-28T06:16:53+00:00 | 233 | 95 | 22 | 9 | 16 | 3 | 82 (79 confirmed, 3 corrected) | 6 (5 confirmed, 1 corrected) | 9 (8 confirmed, 1 corrected) | 954,775 | `70da856153a67b2b9a5f5f5c4dc311017ba026ba409ba7c9cd91223e94b7128f` |
| `notes-s2-neo.json` | 2026-10-28T05:59:12.171096+00:00 | 26 | 32 | 12 | 3 | 11 | 2 | 16 (14 confirmed, 2 corrected) | 10 (10 confirmed) | 3 (3 confirmed) | 117,933 | `0c6710b99b870ebfd4da2513b30c6c6aaa488285abb99b2440d0c98a9ce88d16` |
| `notes-w10-w12.json` | 2026-10-28T06:03:59+00:00 | 38 | 48 | 19 | 3 | 13 | 3 | 38 (34 corrected, 4 confirmed) | 48 (47 confirmed, 1 corrected) | 3 (3 confirmed) | 196,037 | `0499335d90dfe10e50618fe5b0d110e14b7675364861a000033a6c9bea936171` |
| `notes-w13a-k.json` | 2026-10-28T06:06:30+00:00 | 52 | 19 | 7 | 5 | 12 | 2 | 31 (23 confirmed, 8 corrected) | 13 (13 confirmed) | 5 (5 confirmed) | 205,107 | `068e9d8772429d8afd694c221e859bc9283797e44abc9a9edf67c6bd74c2d51b` |
| `notes-w13l-v.json` | 2026-10-28T06:01:09+00:00 | 22 | 35 | 10 | 4 | 15 | 4 | 16 (13 confirmed, 3 corrected) | 35 (33 confirmed, 2 corrected) | 4 (4 confirmed) | 121,066 | `0d242ad4cf0be77d92fffe8d397e69083b00edaa1a828c82b955d7feb0bc97db` |
| `notes-w3-w9.json` | 2026-10-28T05:59:59+00:00 | 67 | 38 | 27 | 13 | 18 | 7 | 67 (57 confirmed, 10 corrected) | 38 (36 confirmed, 2 corrected) | 13 (13 confirmed) | 294,417 | `0470b90b58c88eafd8b8ad0c29b7fcede1577773191d4154269b3cbe7f99cfcd` |
| `omni-fold-sep-oct.json` | 2026-10-28T11:21:10Z | 105 | 25 | 20 | 22 | 44 | 19 | 69 (57 confirmed, 12 corrected) | 22 (21 confirmed, 1 corrected) | 22 (21 confirmed, 1 corrected) | 385,747 | `9b5b04512f121d18041f97c0aaaf98b2889ee6d6bc9cc90be28645e9b4d7e4f1` |
| `pg-aug-a.json` | 2026-10-28T11:09:50+00:00 | 436 | 37 | 26 | 42 | 5 | 5 | 35 (28 confirmed, 7 corrected) | - | - | 945,451 | `8076c53c5c77d51bc62107ef528f4c3b18df5062ad52739d2da41911c40974f1` |
| `pg-aug-b.json` | 2026-10-28T11:03:43.427753+00:00 | 208 | 54 | 86 | 22 | 14 | 4 | 115 (83 confirmed, 32 corrected) | 23 (22 confirmed, 1 corrected) | 22 (22 confirmed) | 775,446 | `1f976b61307e58fff52794ec6d29a4ef1eb278fa62cc6d2c4ce9a2c4346aa49a` |
| `pg-jul-a.json` | 2026-10-28T11:09:02+00:00 | 95 | 18 | 42 | 33 | 5 | 5 | 23 (16 confirmed, 7 corrected) | 8 (7 confirmed, 1 corrected) | - | 262,364 | `ca3ef6c7e3840e95f228aed2b979157cb49b4a0dd5633e7df636963074f2a671` |
| `pg-jul-b.json` | 2026-10-28T11:12:24.833011+00:00 | 221 | 35 | 35 | 33 | 11 | 5 | 84 (57 confirmed, 26 corrected, 1 unverifiable) | 13 (13 confirmed) | 33 (32 confirmed, 1 corrected) | 598,291 | `65cc618bc799f150c10d0af1870c98d8c8fd70466c1d8b1f279e805fce2248d9` |
| `pg-may-jun.json` | 2026-10-28T06:57:18+00:00 | 266 | 30 | 52 | 30 | 10 | 6 | 76 (42 corrected, 34 confirmed) | 8 (8 confirmed) | - | 1,003,633 | `a2c42d149367afbcd723fc286b4bfa3cd04d85855835b3526f7326a0d0aa6471` |
| `pg-oct.json` | 2026-10-28T11:04:26.330179+00:00 | 79 | 26 | 13 | 62 | 16 | 7 | 62 (48 confirmed, 14 corrected) | 26 (22 confirmed, 4 corrected) | 62 (62 confirmed) | 419,974 | `3e87f4170f26355596ca970d2fcdbb64120e49f18e89d6e012c256f61de16c38` |
| `pg-sep.json` | 2026-10-28T12:30:00+00:00 | 85 | 23 | 15 | 16 | 31 | 7 | 85 (74 confirmed, 11 corrected) | 23 (19 confirmed, 4 corrected) | 16 (14 confirmed, 2 corrected) | 404,043 | `cf2af5af3f5d0b5b2325e3255a16e2a6ab04173a7cee7ef68a82a194dbe3f710` |
| `reports-aug.json` | 2026-10-28T06:27:36+00:00 | 648 | 58 | 23 | 21 | 28 | 12 | 225 (219 corrected, 6 confirmed) | 10 (10 confirmed) | - | 3,884,169 | `4a8df112034da544f4257e9a7f61f9e17363dd0d57a2eeb053ddb302c4f6c3f8` |
| `reports-july.json` | 2026-10-28T06:32:25Z | 178 | 24 | 20 | 20 | 150 | 8 | 33 (24 confirmed, 9 corrected) | 10 (9 confirmed, 1 corrected) | 20 (19 confirmed, 1 corrected) | 749,322 | `6d628bf61e09e544c92b6220bae127b74462a5cefb9a62a969f2f9d53fb7b8c5` |
| `server-matrix-runs.json` | 2026-10-28T11:07:05Z | 1,231 | 28 | 17 | 28 | 41 | 10 | 110 (70 added, 27 confirmed, 13 corrected) | 28 (25 confirmed, 3 corrected) | 28 (27 confirmed, 1 corrected) | 7,301,964 | `49c3f5693dd747b6c02780605a3bd0a0f6b7befab449a9a0ee48f6d6618ec05b` |
| `server-surgery-lab.json` | 2026-10-28T11:37:38+00:00 | 260 | 11 | 11 | 0 | 14 | 11 | 35 (18 confirmed, 17 corrected) | 8 (4 confirmed, 4 corrected) | - | 1,013,694 | `f06b2115336c5d01f810cf3d5a8cae02adfb52d6ca95ba0c641882f6e6efd837` |
