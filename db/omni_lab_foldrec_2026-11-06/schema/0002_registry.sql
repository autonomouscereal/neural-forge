-- omni_lab 0002_registry -- Phase B: lanes, ledger mirror, datasets.
-- The ledger ($OC/ledger.jsonl + registry/) stays the checkpoint authority; ledger_event mirrors it.

CREATE TABLE lane (                                  -- from $OC/distill/queues/LANES.json
    id          BIGSERIAL PRIMARY KEY,
    project     TEXT NOT NULL REFERENCES project(key),
    key         TEXT NOT NULL UNIQUE,                -- 'L-D1','L-K3','L-TOOL',...
    owner       TEXT,                                -- 'claude' | 'codex' | 'cyril'
    state       TEXT,
    lock_path   TEXT,
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
ALTER TABLE run ADD COLUMN lane_id BIGINT REFERENCES lane(id);
CREATE INDEX run_lane ON run(lane_id);

CREATE TABLE ledger_event (                          -- one row per ledger.jsonl line
    id              BIGSERIAL PRIMARY KEY,
    model_id        BIGINT NOT NULL REFERENCES model(id),
    checkpoint_id   TEXT NOT NULL,
    record_seq      INTEGER NOT NULL,
    event           TEXT NOT NULL,                   -- register | review | clone | set-status | delete
    stage           TEXT,
    sealed_at       TIMESTAMPTZ,
    sealed_verdict  TEXT,
    column_a        INTEGER,
    column_b        INTEGER,
    net_ratio       NUMERIC,
    record          JSONB NOT NULL,                  -- full registry JSON
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    UNIQUE (checkpoint_id, record_seq)
);
CREATE INDEX ledger_event_model ON ledger_event(model_id);

CREATE TABLE dataset (
    id           BIGSERIAL PRIMARY KEY,
    project      TEXT NOT NULL REFERENCES project(key),
    key          TEXT NOT NULL UNIQUE,               -- 'teacher_cache/qtext/flash-next-strata-iq3s','corpus/intake'
    kind         TEXT NOT NULL CHECK (kind IN ('corpus','teacher-cache','bank','taskset','probe','media-stimuli','training-config','other')),
    description  TEXT,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
CREATE TABLE dataset_version (
    id          BIGSERIAL PRIMARY KEY,
    dataset_id  BIGINT NOT NULL REFERENCES dataset(id),
    version     TEXT NOT NULL,                       -- content sha, corpus_version id, or 'v1'
    path        TEXT,
    host_id     BIGINT REFERENCES host(id),
    manifest    JSONB NOT NULL DEFAULT '{}'::jsonb,
    splits      JSONB NOT NULL DEFAULT '{}'::jsonb,  -- {'train':n,'holdout':n}
    item_count  BIGINT,
    size_bytes  BIGINT,
    sealed      BOOLEAN NOT NULL DEFAULT false,      -- false for provisional caches (e.g. K3 bf16)
    created_at  TIMESTAMPTZ,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    UNIQUE (dataset_id, version)
);
CREATE TABLE run_dataset (
    run_id              BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    dataset_version_id  BIGINT NOT NULL REFERENCES dataset_version(id),
    role                TEXT NOT NULL DEFAULT 'input',   -- 'input','teacher','holdout','output'
    detail              JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (run_id, dataset_version_id, role)
);
