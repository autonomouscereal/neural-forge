-- Load the Neural Forge folding-record export into an EMPTY PostgreSQL 16 database.
--   createdb nf_foldrec
--   cd db/omni_lab_foldrec_2026-11-06 && psql -d nf_foldrec -f load.sql
-- Raw SQL only. Needs gzip on PATH (\copy ... FROM PROGRAM runs on the client).
-- The export is a subset of omni_lab (see README.md), so rows can point at tables that are
-- not shipped (host, gpu, env, tool_version, ingest_batch, lane, eval_*). The script
-- therefore drops every foreign-key constraint after creating the schema, then loads.
\set ON_ERROR_STOP on
BEGIN;
\i schema/0001_core.sql
\i schema/0002_registry.sql
\i schema/0003_knowledge.sql
\i schema/0004_views.sql
\i schema/0005_readonly_grants.sql
\i schema/0006_deletions.sql
\i schema/0007_foldrec.sql

-- omni_lab_db.migrate() creates this table itself; same definition
CREATE TABLE IF NOT EXISTS schema_migration (
    version     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    applied_by  TEXT NOT NULL DEFAULT current_user,
    tool        TEXT
);

DO $$
DECLARE r record;
BEGIN
    FOR r IN SELECT conrelid::regclass AS tbl, conname
             FROM pg_constraint
             WHERE contype = 'f' AND connamespace = 'public'::regnamespace
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', r.tbl, r.conname);
    END LOOP;
END $$;

-- 0001 seeds project rows; the export carries the full table, so start the loaded tables empty
TRUNCATE project, schema_migration, technique, model, model_lineage, model_deletion, run, run_model,
         run_device, run_event, metric, ledger_event, experiment, experiment_alias, experiment_link,
         fold_shard, fold_arm, fold_component, fold_behaviour, fold_diagnostic, fold_source, fold_finding,
         fold_finding_experiment, fold_finding_source, fold_coverage;

\copy project FROM PROGRAM 'gzip -dc data/project.csv.gz' WITH (FORMAT csv, HEADER true)
\copy schema_migration FROM PROGRAM 'gzip -dc data/schema_migration.csv.gz' WITH (FORMAT csv, HEADER true)
\copy technique FROM PROGRAM 'gzip -dc data/technique.csv.gz' WITH (FORMAT csv, HEADER true)
\copy model FROM PROGRAM 'gzip -dc data/model.csv.gz' WITH (FORMAT csv, HEADER true)
\copy model_lineage FROM PROGRAM 'gzip -dc data/model_lineage.csv.gz' WITH (FORMAT csv, HEADER true)
\copy model_deletion FROM PROGRAM 'gzip -dc data/model_deletion.csv.gz' WITH (FORMAT csv, HEADER true)
\copy run FROM PROGRAM 'gzip -dc data/run.csv.gz' WITH (FORMAT csv, HEADER true)
\copy run_model FROM PROGRAM 'gzip -dc data/run_model.csv.gz' WITH (FORMAT csv, HEADER true)
\copy run_device FROM PROGRAM 'gzip -dc data/run_device.csv.gz' WITH (FORMAT csv, HEADER true)
\copy run_event FROM PROGRAM 'gzip -dc data/run_event.csv.gz' WITH (FORMAT csv, HEADER true)
\copy metric FROM PROGRAM 'gzip -dc data/metric.csv.gz' WITH (FORMAT csv, HEADER true)
\copy ledger_event FROM PROGRAM 'gzip -dc data/ledger_event.csv.gz' WITH (FORMAT csv, HEADER true)
\copy experiment FROM PROGRAM 'gzip -dc data/experiment.csv.gz' WITH (FORMAT csv, HEADER true)
\copy experiment_alias FROM PROGRAM 'gzip -dc data/experiment_alias.csv.gz' WITH (FORMAT csv, HEADER true)
\copy experiment_link FROM PROGRAM 'gzip -dc data/experiment_link.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_shard FROM PROGRAM 'gzip -dc data/fold_shard.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_arm FROM PROGRAM 'gzip -dc data/fold_arm.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_component FROM PROGRAM 'gzip -dc data/fold_component.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_behaviour FROM PROGRAM 'gzip -dc data/fold_behaviour.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_diagnostic FROM PROGRAM 'gzip -dc data/fold_diagnostic.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_source FROM PROGRAM 'gzip -dc data/fold_source.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_finding FROM PROGRAM 'gzip -dc data/fold_finding.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_finding_experiment FROM PROGRAM 'gzip -dc data/fold_finding_experiment.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_finding_source FROM PROGRAM 'gzip -dc data/fold_finding_source.csv.gz' WITH (FORMAT csv, HEADER true)
\copy fold_coverage FROM PROGRAM 'gzip -dc data/fold_coverage.csv.gz' WITH (FORMAT csv, HEADER true)
COMMIT;

SELECT 'fold_arm' AS t, count(*) FROM fold_arm
UNION ALL SELECT 'fold_finding', count(*) FROM fold_finding
UNION ALL SELECT 'experiment', count(*) FROM experiment;
