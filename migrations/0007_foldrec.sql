-- omni_lab 0007_foldrec -- the complete Neural Forge folding record (inventory/<date>-foldrec/*.json, schema foldrec-v1,
-- loaded by ingest/foldrec.py).  Additive: new tables and views, new nullable columns on technique, and a wider
-- evidence_level CHECK on experiment.  No existing row is changed.
--
-- Shape
--   * one fold ARM = one experiment row (key = the source's own id) + one fold_arm row with every per-arm field.
--     An arm whose key is already an experiment (e.g. a foldhist entry) attaches its fold_arm to that row; the
--     foldhist columns stay as foldhist wrote them.
--   * per-arm lists are child tables keyed (experiment_id, ordinal): fold_component, fold_behaviour,
--     fold_diagnostic, fold_source.  shards[] says which shard file(s) gave each entry.
--   * when two shards disagree on a field of the same arm, every variant is kept in fold_arm.conflicts with
--     its shard and sources; a disagreeing NUMBER (and verdict_class) is left NULL in its column, never picked.
--   * experiment_alias holds every other id of an arm; experiment_link relates arms to the earlier summary rows
--     (foldhist 'NF-02', 'MX-4X', ...): same_as / part_of / rerun_of / summarizes.
--   * fold_finding (+ evidence links), technique extensions, fold_shard + fold_coverage (what each shard read).
-- Text is stored exactly as the shard wrote it; numeric columns are parsed copies, NULL when not parseable.

-- ---------- existing tables ----------
ALTER TABLE experiment DROP CONSTRAINT IF EXISTS experiment_evidence_level_check;
ALTER TABLE experiment ADD CONSTRAINT experiment_evidence_level_check
    CHECK (evidence_level IN ('receipt','report','notes','memory','transcript','commit_msg','codex','unverified'));

ALTER TABLE technique ADD COLUMN code_location   TEXT,              -- 'src/neural_forge/x.py:fn @36e29997'
                      ADD COLUMN first_date      DATE,
                      ADD COLUMN first_date_text TEXT,
                      ADD COLUMN status_text     TEXT,              -- as the record states it
                      ADD COLUMN detail          JSONB NOT NULL DEFAULT '{}'::jsonb;  -- foldrec variants per shard

-- ---------- shards and what they read ----------
CREATE TABLE fold_shard (
    shard              TEXT PRIMARY KEY,                 -- the shard's own name ('notes-early')
    schema             TEXT NOT NULL,                    -- 'foldrec-v1'
    extracted_at       TIMESTAMPTZ,
    extracted_at_text  TEXT,
    n_experiments      INTEGER NOT NULL DEFAULT 0,
    n_findings         INTEGER NOT NULL DEFAULT 0,
    n_techniques       INTEGER NOT NULL DEFAULT 0,
    n_directives       INTEGER NOT NULL DEFAULT 0,
    n_sources_read     INTEGER NOT NULL DEFAULT 0,
    n_sources_not_read INTEGER NOT NULL DEFAULT 0,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
CREATE TABLE fold_coverage (
    shard    TEXT NOT NULL REFERENCES fold_shard(shard) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,                           -- position in sources_read, then sources_not_read
    state    TEXT NOT NULL CHECK (state IN ('read','not_read')),
    path     TEXT NOT NULL,
    lines    TEXT,                                       -- 'all' | 'a-b' as written
    note     TEXT,                                       -- note (read) or reason (not_read)
    PRIMARY KEY (shard, ordinal)
);
CREATE INDEX fold_coverage_path ON fold_coverage(path);

-- ---------- one row per fold arm ----------
CREATE TABLE fold_arm (
    experiment_id      BIGINT PRIMARY KEY REFERENCES experiment(id) ON DELETE CASCADE,
    arm_key            TEXT NOT NULL UNIQUE,             -- the canonical foldrec key (= experiment.key)
    title              TEXT,
    program            TEXT CHECK (program IN ('early-surgery','neural-forge','residual-attention','lfm2-dynamic','cpcf',
                                               'sparse-semantic','cross-width','cross-family','kernel','stage1','stage2',
                                               'matrix-campaign','chimera-omni-toy','chimera-omni','k3','media','density',
                                               'other')),
    wave               TEXT,
    owner              TEXT CHECK (owner IN ('claude','codex','both','unknown')),
    date_text          TEXT,
    parent_name        TEXT,
    parent_params      TEXT,
    child_ref          TEXT,
    child_params       TEXT,
    child_exists_now   TEXT CHECK (child_exists_now IN ('yes','deleted','unknown')),
    axes               INTEGER,
    ratio_body         NUMERIC,                          -- child/parent body (0.25 = 4x)
    ratio_net          NUMERIC,
    ratio_text         TEXT,
    objective_fit      TEXT CHECK (objective_fit IN ('closed_form','ridge','coreset','low_rank','eigen','pca','omp',
                                                     'gradient','kd','on_policy','search','other')),
    objective_target   TEXT CHECK (objective_target IN ('activation','output_write','top1','topk','kl','multi_token',
                                                        'endpoint_residual','energy','attention_pattern','other')),
    objective_k        TEXT,
    objective_horizon  TEXT,
    objective_text     TEXT,
    training_used      TEXT CHECK (training_used IN ('none','recovery_after_fold','gradient_fit_of_fold','unknown')),
    training_text      TEXT,
    verdict_class      TEXT CHECK (verdict_class IN ('works','partial','fails','mechanics_only','design_only','unmeasured',
                                                     'closed_path','superseded')),
    verdict_text       TEXT,
    cyril              TEXT,                             -- his verbatim reaction / score about this arm (+ ev id)
    evidence_level     TEXT CHECK (evidence_level IN ('receipt','report','notes','memory','transcript','commit_msg')),
    shards             TEXT[] NOT NULL DEFAULT '{}',     -- every shard that described this arm
    member_keys        TEXT[] NOT NULL DEFAULT '{}',     -- every shard key merged into this arm
    run_ids            BIGINT[] NOT NULL DEFAULT '{}',   -- run rows whose receipt_path / run_dir a source or child_ref names
    conflicts          JSONB NOT NULL DEFAULT '{}'::jsonb,   -- field -> [{value, shard, key, sources}] when shards disagree
    record             JSONB NOT NULL,                   -- [{shard, record}] every raw shard record, verbatim
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
CREATE INDEX fold_arm_program ON fold_arm(program, verdict_class);

CREATE TABLE fold_component (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   BIGINT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('hidden_width','mlp_width','depth_excision','depth_merge','attn_q','attn_k',
                                                  'attn_v','attn_o','attn_qkvo','heads','head_dim','conv','embed','vocab',
                                                  'lm_head','experts_to_dense','cross_width','cross_family',
                                                  'kind_translation','modality_unit','whole_model','other')),
    layers          TEXT,
    technique_id    BIGINT REFERENCES technique(id),
    technique_text  TEXT,                                -- the name exactly as the source writes it
    ratio           NUMERIC,
    ratio_text      TEXT,
    rank_or_k       TEXT,
    notes           TEXT,
    shards          TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, ordinal)
);
CREATE INDEX fold_component_technique ON fold_component(technique_id, kind);

CREATE TABLE fold_behaviour (
    id                 BIGSERIAL PRIMARY KEY,
    experiment_id      BIGINT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    ordinal            INTEGER NOT NULL,
    suite              TEXT,
    scorer             TEXT CHECK (scorer IN ('cyril','claude','codex','machine','mixed','none')),
    score              TEXT,                             -- '12/15' as written
    score_value        NUMERIC,                          -- parsed from score 'a/b' (a), else NULL
    score_denominator  NUMERIC,                          -- parsed from score 'a/b' (b), else NULL
    useful             INTEGER,
    partial            INTEGER,
    miss               INTEGER,
    total              INTEGER,
    column_b           TEXT,
    parent_score       TEXT,
    text               TEXT,
    shards             TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, ordinal)
);

CREATE TABLE fold_diagnostic (
    id             BIGSERIAL PRIMARY KEY,
    experiment_id  BIGINT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    ordinal        INTEGER NOT NULL,
    name           TEXT NOT NULL,                        -- tf_top1 | top5 | kl | relL2 | ... | other
    value_text     TEXT NOT NULL,                        -- exactly as written
    value_num      DOUBLE PRECISION,                     -- when value_text is one number / 'a/b' / 'n%'
    value_unit     TEXT,                                 -- '%', 'fraction', 'x', 'dB' (how value_num was read)
    horizon        TEXT,
    denominator    TEXT,
    split          TEXT,
    note           TEXT,
    shards         TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, ordinal)
);
CREATE INDEX fold_diagnostic_name ON fold_diagnostic(name);

CREATE TABLE fold_source (
    id             BIGSERIAL PRIMARY KEY,
    experiment_id  BIGINT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    ordinal        INTEGER NOT NULL,
    path           TEXT,
    lines          TEXT,
    commit_ref     TEXT,
    ev             TEXT,                                 -- agent_memory event id as written
    ev_id          BIGINT,                               -- parsed agent_memory_events.id (other database; no FK)
    quote          TEXT,
    shards         TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (experiment_id, ordinal)
);
CREATE INDEX fold_source_path ON fold_source(path);

-- ---------- aliases and links between experiments ----------
CREATE TABLE experiment_alias (
    alias          TEXT NOT NULL,
    experiment_id  BIGINT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    source         TEXT NOT NULL DEFAULT 'foldrec',      -- writer of the row (ingest_batch.source)
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    PRIMARY KEY (alias, experiment_id)                   -- one alias may name several arms (a summary id)
);
CREATE INDEX experiment_alias_experiment ON experiment_alias(experiment_id);

CREATE TABLE experiment_link (
    from_id   BIGINT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    to_id     BIGINT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    relation  TEXT NOT NULL CHECK (relation IN ('same_as','part_of','rerun_of','summarizes')),
    source    TEXT NOT NULL DEFAULT 'foldrec',
    detail    JSONB NOT NULL DEFAULT '{}'::jsonb,        -- {via: alias, arms_sharing_alias: n}
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    PRIMARY KEY (from_id, to_id, relation),
    CHECK (from_id <> to_id)
);
CREATE INDEX experiment_link_to ON experiment_link(to_id);

-- ---------- findings ----------
CREATE TABLE fold_finding (
    id           BIGSERIAL PRIMARY KEY,
    finding_key  TEXT NOT NULL UNIQUE,
    statement    TEXT NOT NULL,                          -- verbatim or near-verbatim, as the shard wrote it
    status       TEXT CHECK (status IN ('proven','supported','refuted','closed_path','open','superseded')),
    scope        TEXT,
    shards       TEXT[] NOT NULL DEFAULT '{}',
    conflicts    JSONB NOT NULL DEFAULT '{}'::jsonb,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
CREATE TABLE fold_finding_experiment (                   -- evidence link finding -> arm
    finding_id      BIGINT NOT NULL REFERENCES fold_finding(id) ON DELETE CASCADE,
    experiment_key  TEXT NOT NULL,                       -- as written (kept when it resolves to nothing yet)
    experiment_id   BIGINT REFERENCES experiment(id) ON DELETE SET NULL,
    PRIMARY KEY (finding_id, experiment_key)
);
CREATE TABLE fold_finding_source (
    id          BIGSERIAL PRIMARY KEY,
    finding_id  BIGINT NOT NULL REFERENCES fold_finding(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    path        TEXT,
    lines       TEXT,
    commit_ref  TEXT,
    ev          TEXT,
    quote       TEXT,
    shards      TEXT[] NOT NULL DEFAULT '{}',
    UNIQUE (finding_id, ordinal)
);

-- ---------- views ----------
-- One row per fold arm.  behaviour_* is the first behaviour row (n_behaviour says how many there are).
CREATE VIEW v_fold_arms AS
SELECT fa.arm_key AS key,
       e.date,
       fa.date_text,
       fa.program,
       fa.wave,
       fa.owner,
       fa.title,
       fa.parent_name AS parent,
       fa.child_ref,
       (SELECT string_agg(fc.kind || COALESCE('[' || fc.layers || ']', '')
                          || COALESCE(' ' || COALESCE(t.name, fc.technique_text), '')
                          || COALESCE(' r=' || COALESCE(fc.ratio::text, fc.ratio_text), '')
                          || COALESCE(' k=' || fc.rank_or_k, ''), '; ' ORDER BY fc.ordinal)
          FROM fold_component fc LEFT JOIN technique t ON t.id = fc.technique_id
         WHERE fc.experiment_id = e.id) AS components,
       (SELECT string_agg(DISTINCT COALESCE(t.name, fc.technique_text), ' | ')
          FROM fold_component fc LEFT JOIN technique t ON t.id = fc.technique_id
         WHERE fc.experiment_id = e.id) AS techniques,
       fa.axes,
       fa.ratio_body,
       fa.ratio_net,
       fa.ratio_text,
       NULLIF(concat_ws(' ', fa.objective_fit, '-> ' || fa.objective_target, 'k=' || fa.objective_k,
                        'h=' || fa.objective_horizon), '') AS objective,
       fa.objective_text,
       fa.training_used,
       b.suite AS behaviour_suite,
       b.score AS behaviour_score,
       b.scorer,
       b.parent_score,
       b.column_b,
       (SELECT count(*) FROM fold_behaviour fb WHERE fb.experiment_id = e.id) AS n_behaviour,
       (SELECT string_agg(fd.name || '=' || fd.value_text || COALESCE('@' || fd.horizon, '')
                          || COALESCE(' (' || fd.split || ')', ''), '; ' ORDER BY fd.ordinal)
          FROM fold_diagnostic fd WHERE fd.experiment_id = e.id) AS diagnostics,
       fa.verdict_class,
       fa.verdict_text,
       fa.cyril,
       fa.evidence_level,
       (SELECT count(*) FROM fold_source fs WHERE fs.experiment_id = e.id) AS n_sources,
       (SELECT string_agg(a.alias, ', ' ORDER BY a.alias) FROM experiment_alias a WHERE a.experiment_id = e.id) AS aliases,
       (SELECT string_agg(o.key || ' ' || l.relation, ', ' ORDER BY o.key)
          FROM experiment_link l JOIN experiment o ON o.id = l.from_id WHERE l.to_id = e.id) AS linked_from,
       array_to_string(fa.shards, ', ') AS shards,
       fa.conflicts <> '{}'::jsonb AS has_conflicts,
       e.id AS experiment_id
FROM fold_arm fa
JOIN experiment e ON e.id = fa.experiment_id
LEFT JOIN LATERAL (SELECT fb.suite, fb.score, fb.scorer, fb.parent_score, fb.column_b FROM fold_behaviour fb
                   WHERE fb.experiment_id = e.id ORDER BY fb.ordinal LIMIT 1) b ON true;

-- Per technique x component kind.  A ratio is child/parent, so ">= 4x" is ratio <= 0.25 (component ratio, else
-- the arm's body ratio).  Behaviour fraction = score a/b, else useful/total; scorer 'none' rows are ignored.
CREATE VIEW v_technique_scoreboard AS
WITH comp AS (
    SELECT COALESCE(t.name, fc.technique_text, '(none)') AS technique, fc.technique_id, fc.kind, fc.experiment_id,
           COALESCE(fc.ratio, fa.ratio_body) AS ratio, fa.parent_name, fa.verdict_class
    FROM fold_component fc
    JOIN fold_arm fa ON fa.experiment_id = fc.experiment_id
    LEFT JOIN technique t ON t.id = fc.technique_id
), beh AS (
    SELECT DISTINCT ON (experiment_id) experiment_id, score, useful, total, frac
    FROM (SELECT fb.experiment_id, fb.score, fb.useful, fb.total, fb.ordinal,
                 COALESCE(fb.score_value / NULLIF(fb.score_denominator, 0), fb.useful::numeric / NULLIF(fb.total, 0)) AS frac
          FROM fold_behaviour fb WHERE fb.scorer IS DISTINCT FROM 'none') x
    WHERE frac IS NOT NULL
    ORDER BY experiment_id, frac DESC, ordinal
)
SELECT c.technique,
       c.kind,
       count(DISTINCT c.experiment_id) AS arms,
       count(DISTINCT c.experiment_id) FILTER (WHERE b.frac IS NOT NULL) AS arms_with_behaviour,
       round(max(b.frac), 4) AS best_behaviour,
       (array_agg(COALESCE(b.score, b.useful || '/' || b.total) || ' (' || e.key || ')'
                  ORDER BY b.frac DESC NULLS LAST, e.key))[1] AS best_behaviour_text,
       COALESCE(bool_or(c.ratio <= 0.25), false) AS tested_at_4x,
       count(DISTINCT c.experiment_id) FILTER (WHERE c.ratio <= 0.25) AS arms_at_4x,
       round(max(b.frac) FILTER (WHERE c.ratio <= 0.25), 4) AS best_behaviour_at_4x,
       min(c.ratio) AS min_ratio,
       string_agg(DISTINCT c.parent_name, ' | ') AS scales_tested,
       count(DISTINCT c.experiment_id) FILTER (WHERE c.verdict_class = 'works') AS works,
       count(DISTINCT c.experiment_id) FILTER (WHERE c.verdict_class = 'partial') AS partial,
       count(DISTINCT c.experiment_id) FILTER (WHERE c.verdict_class = 'fails') AS fails,
       count(DISTINCT c.experiment_id) FILTER (WHERE c.verdict_class = 'closed_path') AS closed_path,
       count(DISTINCT c.experiment_id) FILTER (WHERE c.verdict_class IN ('mechanics_only','design_only','unmeasured'))
           AS unproven,
       max(t.status_text) AS status
FROM comp c
JOIN experiment e ON e.id = c.experiment_id
LEFT JOIN beh b ON b.experiment_id = c.experiment_id
LEFT JOIN technique t ON t.id = c.technique_id
GROUP BY c.technique, c.kind;

CREATE VIEW v_findings AS
SELECT f.finding_key AS key,
       f.status,
       f.statement,
       f.scope,
       (SELECT count(*) FROM fold_finding_experiment fe WHERE fe.finding_id = f.id) AS n_experiments,
       (SELECT count(*) FROM fold_finding_experiment fe WHERE fe.finding_id = f.id AND fe.experiment_id IS NULL)
           AS n_unresolved,
       (SELECT string_agg(fe.experiment_key, ', ' ORDER BY fe.experiment_key)
          FROM fold_finding_experiment fe WHERE fe.finding_id = f.id) AS experiment_keys,
       (SELECT count(*) FROM fold_finding_source s WHERE s.finding_id = f.id) AS n_sources,
       (SELECT s.path || COALESCE(':' || s.lines, '') FROM fold_finding_source s WHERE s.finding_id = f.id
         ORDER BY s.ordinal LIMIT 1) AS first_source,
       array_to_string(f.shards, ', ') AS shards,
       f.id AS finding_id
FROM fold_finding f;

-- Per source path: which shards read it (and which lines), which skipped it and why, how many arms cite it.
CREATE VIEW v_fold_coverage AS
WITH paths AS (
    SELECT path FROM fold_coverage
    UNION
    SELECT path FROM fold_source WHERE path IS NOT NULL
), cited AS (
    SELECT path, count(DISTINCT experiment_id) AS n_arms FROM fold_source WHERE path IS NOT NULL GROUP BY path
)
SELECT p.path,
       (SELECT string_agg(DISTINCT fc.shard, ', ') FROM fold_coverage fc WHERE fc.path = p.path AND fc.state = 'read')
           AS shards_read,
       (SELECT string_agg(fc.shard || COALESCE(':' || fc.lines, ''), ', ' ORDER BY fc.shard, fc.ordinal)
          FROM fold_coverage fc WHERE fc.path = p.path AND fc.state = 'read') AS lines_read,
       (SELECT string_agg(DISTINCT fc.shard, ', ') FROM fold_coverage fc WHERE fc.path = p.path AND fc.state = 'not_read')
           AS shards_not_read,
       (SELECT string_agg(fc.shard || ': ' || COALESCE(fc.note, '?'), ' | ' ORDER BY fc.shard, fc.ordinal)
          FROM fold_coverage fc WHERE fc.path = p.path AND fc.state = 'not_read') AS not_read_reasons,
       COALESCE(c.n_arms, 0) AS arms_citing,
       NOT EXISTS (SELECT 1 FROM fold_coverage fc WHERE fc.path = p.path AND fc.state = 'read')
           AND COALESCE(c.n_arms, 0) > 0 AS cited_not_declared_read
FROM paths p
LEFT JOIN cited c ON c.path = p.path;

-- ---------- grants (0005's default privileges cover these; explicit as in 0006) ----------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'omni_lab_ro') THEN
        GRANT SELECT ON fold_shard, fold_coverage, fold_arm, fold_component, fold_behaviour, fold_diagnostic,
                        fold_source, experiment_alias, experiment_link, fold_finding, fold_finding_experiment,
                        fold_finding_source, v_fold_arms, v_technique_scoreboard, v_findings, v_fold_coverage
            TO omni_lab_ro;
    END IF;
END
$$;
