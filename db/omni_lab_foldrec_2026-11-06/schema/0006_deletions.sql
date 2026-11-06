-- omni_lab 0006_deletions -- record every weight directory removed from disk (cleanup order, Cyril 2026-10-21:
-- "[Cyril quote withheld]").
--
-- model gains the deletion facts; model_deletion is the ledger-equivalent record of each removed unit
-- (one row per removed directory, holding its digests, recipe archive and the preflight that cleared it).
-- status 'deleting' marks a row whose directory is being removed: it becomes 'deleted' only after the
-- path is verified absent, so a failed rm never leaves the database claiming the bytes are gone.
-- Ingesters never change a 'deleting' or 'deleted' status (omni_lab_db.upsert_model, ingest/model_scan.py).

ALTER TABLE model DROP CONSTRAINT IF EXISTS model_status_check;
ALTER TABLE model ADD CONSTRAINT model_status_check
    CHECK (status IN ('present','deleting','deleted','missing','unknown'));
ALTER TABLE model ADD COLUMN deleted_at          TIMESTAMPTZ,
                  ADD COLUMN delete_reason       TEXT,
                  ADD COLUMN rebuild_recipe_path TEXT;

CREATE TABLE model_deletion (
    id                    BIGSERIAL PRIMARY KEY,
    model_id              BIGINT NOT NULL REFERENCES model(id),       -- the unit row (path of the removed directory)
    duplicate_of_model_id BIGINT REFERENCES model(id),                -- byte copies: the row that still holds the bytes
    host_id               BIGINT NOT NULL REFERENCES host(id),
    unit_path             TEXT NOT NULL,
    plan_wave             TEXT NOT NULL,                              -- W1-NAS, W2-HOME, W3-LEXAR, W4-WS, W5
    plan_index            INTEGER,                                    -- index into cleanup_plan.json server_delete
    status                TEXT NOT NULL DEFAULT 'recorded'
                          CHECK (status IN ('recorded','deleting','deleted','failed','aborted')),
    size_bytes_du         BIGINT NOT NULL,
    weight_bytes          BIGINT,
    file_count            INTEGER,
    layout                TEXT,
    parameter_count       BIGINT,
    digest_mode           TEXT NOT NULL CHECK (digest_mode IN ('full-sha256','index+headers+listing','git-blob')),
    listing_sha256        TEXT NOT NULL,                              -- sha256 over sorted relpath\tsize\tdigest lines
    file_digests          JSONB NOT NULL,                             -- {relpath: {size, mtime, sha256 | header_sha256}}
    recipe_archive_path   TEXT NOT NULL,
    recipe_archive_sha256 TEXT,                                       -- sha256 of the archive's MANIFEST.tsv
    recipe_files          JSONB NOT NULL,                             -- [relpath, ...] copied into the archive
    affected_model_ids    BIGINT[] NOT NULL DEFAULT '{}',             -- every model row inside the unit, marked too
    producing_run_ids     BIGINT[],
    ledger_checkpoint_id  TEXT,
    ledger_receipt_sha    TEXT,
    reason                TEXT NOT NULL,
    owner_order           TEXT NOT NULL,                              -- Cyril's words, verbatim
    preflight             JSONB NOT NULL,                             -- open files, references, data checks at delete time
    df_before             JSONB,
    df_after              JSONB,
    error                 TEXT,
    recorded_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    executed_at           TIMESTAMPTZ,
    verified_absent_at    TIMESTAMPTZ,
    UNIQUE (host_id, unit_path)
);
CREATE INDEX model_deletion_model ON model_deletion(model_id);
CREATE INDEX model_deletion_wave ON model_deletion(plan_wave, status);

INSERT INTO host (name, detail) VALUES ('workstation', '{"os": "Windows 11 Pro", "drives": ["C:", "D:", "E:"]}'::jsonb)
    ON CONFLICT (name) DO NOTHING;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'omni_lab_ro') THEN
        GRANT SELECT ON model_deletion TO omni_lab_ro;
    END IF;
END
$$;
