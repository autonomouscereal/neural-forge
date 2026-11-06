# Neural Forge folding record: database export, 2026-11-06

This folder is the public copy of the work database behind Neural Forge. Every fold
experiment from 2026-06-26 to 2026-10-29 is a row here, across LFM2, Qwen3/3.5, the Q38-27B
trunk, Kimi K3 and the media parents. Each row carries its techniques, ratios, diagnostics,
behaviour reads, verdicts and findings, and the source that every claim cites.

- Source: the `omni_lab` PostgreSQL 16.13 database, read with the read-only role at
  2026-11-06 19:04Z (`manifest.json` has the exact query per table).
- Format: one gzip CSV per table (`COPY ... WITH (FORMAT csv, HEADER true)`), plus the seven
  forward-only migrations that define the schema (`schema/`). Six are byte-identical to the
  applied migrations, whose sha256 values are in `data/schema_migration.csv.gz`. The seventh,
  `0006_deletions.sql`, has one quote withheld in its header comment, so its hash no longer
  matches its row. The SQL itself is unchanged.
- Integrity: `SHA256SUMS` covers every file here except itself. A reload into a throwaway
  database reproduced every table's row count exactly (2026-11-06, after the quotes were
  withheld).
- Cyril's words are withheld. Every quote that matches something he wrote (his directives and
  rulings in `omni_lab`, his prompts in agent memory) reads `[Cyril quote withheld]`.
  `manifest.json` → `quotes_withheld` counts the changed cells per column.
- Size: about 17 MB compressed.

## Load it

```bash
createdb nf_foldrec
cd db/omni_lab_foldrec_2026-11-06
psql -d nf_foldrec -f load.sql
```

`load.sql` creates the schema and then drops the foreign keys, because this is a subset of
the database (see below), so some ids point at tables that are not shipped. After that it
loads every table and prints three counts. Raw SQL only: no ORM.

## What is in it

| Table | Rows | What a row is |
|---|--:|---|
| `fold_arm` | 7,895 | One fold arm: parent, child, ratio, objective, training used, verdict class and text, a note on Cyril's reaction where there is one (his words withheld), sources |
| `fold_component` | 12,163 | One folded component of an arm: kind, layers, technique, ratio, rank or k |
| `fold_behaviour` | 6,674 | One behaviour read: suite, who scored it (claude / codex / machine / cyril / mixed / none), useful, partial, miss |
| `fold_diagnostic` | 38,600 | One numeric diagnostic (top-1/top-k, LCP, perplexity, drift, write cosine, ...) with its denominator and split |
| `fold_finding` | 2,947 | One finding, with status `proven` / `supported` / `open` / `closed_path` / `superseded` / `refuted` |
| `fold_finding_source`, `fold_source` | 3,653 / 33,013 | The file, line range, commit or memory event behind each finding and each arm |
| `fold_finding_experiment` | 5,324 | Which arms a finding rests on |
| `fold_coverage`, `fold_shard` | 6,149 / 87 | What each extraction shard read and did not read (the record's own gaps) |
| `experiment`, `experiment_alias`, `experiment_link` | 7,928 / 7,886 / 2,755 | Experiment registry, the other names each arm had, and lineage between arms |
| `technique` | 5,805 | Technique vocabulary, with code location and first date |
| `model`, `model_lineage`, `model_deletion` | 1,921 / 53 / 242 | Model metadata (no weights), parent/child lineage, and the record of every checkpoint deleted in the 2026-10-21 cleanup with its rebuild recipe |
| `run`, `run_model`, `run_device`, `run_event`, `metric` | 950 / 29 / 15 / 112 / 2,562 | Neural Forge runs and every run an experiment points at |
| `ledger_event` | 42 | Mirror of the campaign checkpoint ledger |
| `project`, `schema_migration` | 7 / 7 | Project keys and the applied migrations |

Start with `fold_arm` joined to `fold_behaviour` (`experiment_id`), and with `fold_finding`
filtered on `status`. The record's own reading guide is
`reports/omni-campaign-2026-09/FOLDING_RECORD_2026-10-22.md` on the `claude/omni-campaign`
branch, which was built from this database.

## What was left out, and why

- **No weights, tensors or caches.** The database stores paths and digests only.
- **`directive`, `ruling`, `document`**: the verbatim instruction log covers every project,
  including private ones.
- **Cyril's verbatim words, everywhere else.** `fold_arm.cyril` keeps the context, the date
  and the event id of each reaction; the words are withheld. A `fold_source.quote` or
  `fold_finding_source.quote` that cites one of his prompts is withheld whole. Paraphrases
  written by the agents are kept.
- **`eval_*`, `test_case`, `test_suite`, `suite_member`, `grader`, `test`**: the Chimera/Sprite
  evaluation banks and their outputs. They are not Neural Forge, and some banks are private.
- **`dataset`, `dataset_version`, `run_dataset`**: corpus manifests (102 MB) for the training
  lanes.
- **`host`, `gpu`, `env`, `tool`, `tool_version`, `ingest_batch`, `lane`, `arch_*`,
  `model_tensor_group`**: machine inventory and ingest bookkeeping.
- Runs of other projects, unless a fold experiment points at them.

Before publishing, every file was scanned for the 137 vault credentials and for token, key and
password patterns. The only match is the string `lmstudio`, LM Studio's default placeholder
API key, which appears as a model-path word in three notes.

## Reading the numbers

- `ratio_net` / `ratio_body` are child/parent (0.25 = 4x smaller).
- "x/15" counts are from the frozen 15-prompt review, and they are upper bounds: re-read with
  three restatements per lane, the parent drops from 12 to 6 (finding `MX33.F1`).
- `verdict_class = 'works'` never means "4x with retained intelligence" by itself. Small
  shrinks and mechanism-only passes are in that class too.
- Under the 2026-10-22 rule, no number passes or fails a fold. An arm that was dropped on a
  number is *unreviewed*, not failed (`reports/omni-campaign-2026-09/JUDGEMENT_AUDIT_2026-10-29.md`).
