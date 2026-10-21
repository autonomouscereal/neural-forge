-- omni_lab 0003_knowledge -- owner directives, rulings, documents, and architecture conformance.
-- Architecture conformance is measured from TENSORS (model_tensor_group), not from config flags.

CREATE TABLE directive (
    id                BIGSERIAL PRIMARY KEY,
    directive_key     TEXT NOT NULL UNIQUE,          -- 'ev:506681' | 'cc:7d20c05f:1006' | 'doc:<sha16>:<line>'
    said_at           TIMESTAMPTZ NOT NULL,
    said_at_verified  BOOLEAN NOT NULL DEFAULT true, -- false for dot-sync rows
    author            TEXT NOT NULL DEFAULT 'cyril',
    channel           TEXT NOT NULL,                 -- 'codex','claude-code','dot','transcript','doc'
    source_event_id   BIGINT,                        -- agent_memory_events.id (other database; no FK)
    verbatim          TEXT NOT NULL,
    topics            TEXT[] NOT NULL DEFAULT '{}',  -- 'architecture','moe','encoders','k3','fold','training',...
    summary           TEXT,
    supersedes_id     BIGINT REFERENCES directive(id),
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
CREATE INDEX directive_topics_gin ON directive USING gin (topics);
CREATE INDEX directive_said_at ON directive(said_at);

CREATE TABLE ruling (
    id            BIGSERIAL PRIMARY KEY,
    ruling_key    TEXT NOT NULL UNIQUE,              -- 'OWNER-RULING-2026-10-12-STAGE-D-GO'
    decided_at    TIMESTAMPTZ NOT NULL,
    who           TEXT NOT NULL,                     -- 'cyril' | 'claude-lead' | 'codex'
    authority     TEXT NOT NULL CHECK (authority IN ('owner','lead','agent')),
    verbatim      TEXT,
    decision      TEXT NOT NULL,
    scope         TEXT[] NOT NULL DEFAULT '{}',
    directive_id  BIGINT REFERENCES directive(id),
    attachments   JSONB NOT NULL DEFAULT '[]'::jsonb,-- [{path, sha256}]
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);

CREATE TABLE document (
    id           BIGSERIAL PRIMARY KEY,
    sha256       TEXT NOT NULL UNIQUE,
    path         TEXT NOT NULL,                      -- canonical copy
    other_paths  TEXT[] NOT NULL DEFAULT '{}',
    host         TEXT NOT NULL,
    title        TEXT,
    doc_date     DATE,
    date_source  TEXT CHECK (date_source IN ('text','filename','folder','mtime')),
    doc_type     TEXT,                               -- state,plan,design,ruling,receipt,report,review,runbook,dataset-manifest
    arm          TEXT,                               -- fold,distill,media,speech,k3,arch,eval,data,infra
    status       TEXT NOT NULL DEFAULT 'current' CHECK (status IN ('current','superseded','backup','scratch')),
    summary      TEXT,
    size_bytes   BIGINT,
    rank         INTEGER,
    run_id       BIGINT REFERENCES run(id),
    model_id     BIGINT REFERENCES model(id),
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
CREATE INDEX document_type_arm ON document(doc_type, arm, doc_date DESC);

-- ---------- architecture conformance ----------
CREATE TABLE model_tensor_group (                    -- tensor census per model, grouped by name prefix and layer span
    model_id          BIGINT NOT NULL REFERENCES model(id) ON DELETE CASCADE,
    prefix            TEXT NOT NULL,                 -- 'model.layers.*.mlp', 'native_media_bridge', 'lm_head'
    layer_lo          INTEGER NOT NULL DEFAULT -1,   -- -1 = not layer-indexed
    layer_hi          INTEGER NOT NULL DEFAULT -1,
    n_tensors         INTEGER NOT NULL,
    params            BIGINT,
    dtype             TEXT,
    modality_scope    TEXT NOT NULL DEFAULT 'unknown'
                      CHECK (modality_scope IN ('shared','text','image','audio','video','3d','music','unknown')),
    source_component  TEXT,                          -- 'k3','ace.dit.block12.mlp','qwen3.8 trunk'
    detail            JSONB NOT NULL DEFAULT '{}'::jsonb,   -- e.g. channel partition {"k3":[4352,16640]}
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    PRIMARY KEY (model_id, prefix, layer_lo)
);
CREATE TABLE arch_requirement (                      -- R1..Rn from Cyril's directives (intended_arch.json)
    id             TEXT PRIMARY KEY,                 -- 'R3'
    text           TEXT NOT NULL,
    directive_ids  BIGINT[] NOT NULL DEFAULT '{}',   -- directive.id rows that state it
    source_event_ids BIGINT[] NOT NULL DEFAULT '{}', -- agent_memory_events ids, kept even before directives load
    detail         JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE arch_check (
    model_id        BIGINT NOT NULL REFERENCES model(id) ON DELETE CASCADE,
    requirement_id  TEXT NOT NULL REFERENCES arch_requirement(id),
    status          TEXT NOT NULL CHECK (status IN ('pass','fail','partial','unknown')),
    evidence        TEXT,
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    PRIMARY KEY (model_id, requirement_id)
);
