-- omni_lab 0001_core -- PostgreSQL 16, raw SQL, no extensions required.
-- Phase A of the Omni Lab design: everything the eval runner and the
-- first ingesters (hosts_gpus, oc_ledger models, banks, dossiers,
-- student_tests, toolcall_bankruns, fullkd/media receipts, foldhist) write.
--
-- Conventions
--   * surrogate BIGSERIAL ids, natural keys UNIQUE (ingesters use ON CONFLICT)
--   * provenance on every domain row: src_path, src_sha256, src_host, ingest_batch_id
--   * a blank cell in a result pivot is the ABSENCE of an eval_result row, never a NULL grade
--   * schema_migration is created by the migration runner, not here

-- ---------- provenance ----------
CREATE TABLE ingest_batch (
    id             BIGSERIAL PRIMARY KEY,
    source         TEXT NOT NULL,                    -- 'oc_ledger','dossiers','runner',...
    parser         TEXT,                             -- module path, e.g. ingest/dossiers.py
    parser_sha256  TEXT,
    host           TEXT NOT NULL,                    -- 'ai' | 'workstation'
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at       TIMESTAMPTZ,
    status         TEXT NOT NULL DEFAULT 'running'
                   CHECK (status IN ('running','ok','failed','partial')),
    rows_seen      INTEGER NOT NULL DEFAULT 0,
    rows_written   INTEGER NOT NULL DEFAULT 0,
    rows_skipped   INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    argv           JSONB,
    file_shas      TEXT[] NOT NULL DEFAULT '{}'      -- sha256 of every source file this batch consumed
);
CREATE INDEX ingest_batch_source ON ingest_batch(source, started_at DESC);
CREATE INDEX ingest_batch_file_shas ON ingest_batch USING gin (file_shas);

-- ---------- projects (arms of the program) ----------
CREATE TABLE project (
    key          TEXT PRIMARY KEY,                   -- readable FK target
    name         TEXT NOT NULL,
    description  TEXT
);
INSERT INTO project (key, name, description) VALUES
  ('chimera-omni', 'Chimera Omni',  'OMNI1 composite fold, stage D distillation (D1/D2/full-KD/M1/M2), chimera-omni architecture repo'),
  ('neural-forge', 'Neural Forge',  'fold compiler, LFM2 exact-4x proof, matrix campaign, 4B ladder, density program'),
  ('k3',           'Kimi K3 lane',  'K3 teacher caches, K3 charting into OMNI1'),
  ('speech',       'Speech lane',   'speech data, speech-out routes'),
  ('media',        'Media lane',    'image/video/audio/3D/music units and routes (M1/M2)'),
  ('corpus',       'Corpus/intake', 'canonical corpus, intake, teacher-cache datasets'),
  ('lab',          'Omni Lab',      'this database, its ingesters and runner');

-- ---------- hardware / software environment ----------
CREATE TABLE host (
    id      BIGSERIAL PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE,                    -- 'ai' (192.168.50.222), 'workstation'
    detail  JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE gpu (
    id          BIGSERIAL PRIMARY KEY,
    host_id     BIGINT NOT NULL REFERENCES host(id),
    gpu_index   INTEGER NOT NULL,                    -- nvidia-smi index at last sighting (PCI order)
    uuid        TEXT NOT NULL,                       -- GPU-UUID; the join key, never the index
    name        TEXT NOT NULL,                       -- 'NVIDIA GeForce RTX 3090'
    sm          TEXT,                                -- compute capability '8.6','6.1','7.5'
    vram_mib    INTEGER,
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (host_id, uuid)
);
CREATE INDEX gpu_host_index ON gpu(host_id, gpu_index);

CREATE TABLE env (                                   -- software stack, deduplicated by content hash
    id               BIGSERIAL PRIMARY KEY,
    env_sha256       TEXT NOT NULL UNIQUE,           -- sha256 of canonical JSON of every column below
    python           TEXT,
    torch            TEXT,
    cuda             TEXT,
    driver           TEXT,
    transformers     TEXT,
    fla              TEXT,
    triton           TEXT,
    llama_cpp_build  TEXT,
    code_commit      TEXT,                           -- chimera-omni tree / CHIMERA_OMNI_SRC / repo HEAD
    env_vars         JSONB NOT NULL DEFAULT '{}'::jsonb,   -- CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER, PYTORCH_CUDA_ALLOC_CONF
    packages         JSONB NOT NULL DEFAULT '{}'::jsonb,
    first_seen       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- tools (scripts that produce results) ----------
CREATE TABLE tool (
    id    BIGSERIAL PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE                       -- 'oc_calib','student_test','toolcall_selfgen',...
);
CREATE TABLE tool_version (
    id          BIGSERIAL PRIMARY KEY,
    tool_id     BIGINT NOT NULL REFERENCES tool(id),
    version     TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    path        TEXT,
    pinned      BOOLEAN NOT NULL DEFAULT false,      -- current TOOL_PINS row
    superseded  BOOLEAN NOT NULL DEFAULT false,
    pinned_at   TIMESTAMPTZ,
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    UNIQUE (tool_id, sha256)
);

-- ---------- models / lineage ----------
CREATE TABLE model (
    id                    BIGSERIAL PRIMARY KEY,
    project               TEXT NOT NULL REFERENCES project(key),
    model_key             TEXT NOT NULL UNIQUE,      -- ledger checkpoint_id, else 'path:<digest16>'
    ledger_checkpoint_id  TEXT UNIQUE,               -- NULL until registered in $OC/ledger.jsonl
    role                  TEXT NOT NULL CHECK (role IN ('parent','fold-child','clone','delta','student','media-unit',
                                                        'teacher','export','composite','intermediate','other')),
    layout                TEXT NOT NULL,             -- 'hf','chimera.checkpoint.v1','qwen3_5','gguf','safetensors-raw'
    family                TEXT,                      -- 'Qwen3.8-27B-U','Kimi-K3','LFM2.5-230M',...
    stage                 TEXT,                      -- 'S3','S5','OMNI1','D1','fullkd-run1','M1',...
    path                  TEXT,
    host_id               BIGINT REFERENCES host(id),
    directory_sha256      TEXT,                      -- oc_seal directory digest when known
    parameter_count       BIGINT,
    size_bytes            BIGINT,
    dtype                 TEXT,
    config                JSONB NOT NULL DEFAULT '{}'::jsonb,   -- config.json / manifest essentials
    sealed_verdict        TEXT,
    status                TEXT NOT NULL DEFAULT 'present' CHECK (status IN ('present','deleted','missing','unknown')),
    created_at            TIMESTAMPTZ,
    notes                 TEXT,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
CREATE INDEX model_family_stage ON model(family, stage);
CREATE TABLE model_lineage (
    child_id   BIGINT NOT NULL REFERENCES model(id),
    parent_id  BIGINT NOT NULL REFERENCES model(id),
    relation   TEXT NOT NULL CHECK (relation IN ('fold','clone','delta-of','distill','merge','export','absorb','chart','source')),
    ordinal    INTEGER NOT NULL DEFAULT 0,           -- order among multi-parent composites (OMNI1 has 7 sources)
    detail     JSONB NOT NULL DEFAULT '{}'::jsonb,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    PRIMARY KEY (child_id, parent_id, relation),
    CHECK (child_id <> parent_id)
);
CREATE INDEX model_lineage_parent ON model_lineage(parent_id);

-- ---------- runs: fold, distill, cache, gen, eval, probe, build, export, review ----------
CREATE TABLE run (
    id               BIGSERIAL PRIMARY KEY,
    run_key          TEXT NOT NULL UNIQUE,           -- '<kind>:<lane-or-tool>:<start ts>:<short sha>' or receipt-derived
    project          TEXT NOT NULL REFERENCES project(key),
    kind             TEXT NOT NULL CHECK (kind IN ('fold','distill','cache','gen','eval','probe','build','export','review','smoke','other')),
    name             TEXT NOT NULL,                  -- human label: 'fullkd run1 seg3'
    model_id         BIGINT REFERENCES model(id),    -- model acted on / produced / under test
    tool_version_id  BIGINT REFERENCES tool_version(id),
    host_id          BIGINT REFERENCES host(id),
    env_id           BIGINT REFERENCES env(id),
    status           TEXT NOT NULL CHECK (status IN ('queued','running','pass','fail','refused','stopped','skipped','superseded','unknown')),
    started_at       TIMESTAMPTZ,
    ended_at         TIMESTAMPTZ,
    wall_seconds     DOUBLE PRECISION,
    exit_code        INTEGER,
    config           JSONB NOT NULL DEFAULT '{}'::jsonb,  -- argv, dtype, engine, batch, seeds, lr, objective, segments...
    config_sha256    TEXT,                           -- sha256 of canonical JSON of config (computed in code)
    receipt_path     TEXT,
    receipt_sha256   TEXT,
    log_path         TEXT,
    run_dir          TEXT,
    parent_run_id    BIGINT REFERENCES run(id),      -- segment -> parent run, restart -> original
    owner            TEXT,                           -- 'claude' | 'codex' | 'cyril'
    notes            TEXT,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
CREATE INDEX run_kind_started ON run(kind, started_at DESC);
CREATE INDEX run_model_id_idx ON run(model_id);
CREATE INDEX run_config_sha ON run(config_sha256);
CREATE INDEX run_config_gin ON run USING gin (config);

CREATE TABLE run_device (                            -- placement actually used
    run_id        BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    gpu_id        BIGINT NOT NULL REFERENCES gpu(id),
    role          TEXT NOT NULL DEFAULT 'compute',   -- 'compute','linear-attn','full-attn','split-A','split-B','server'
    kernel_path   TEXT,                              -- 'fla-triton-sm86','torch-fallback','llama.cpp-cuda'
    kernels       JSONB NOT NULL DEFAULT '{}'::jsonb,-- full receipt kernels dict
    layers        INT4RANGE,
    mem_mib_peak  INTEGER,
    detail        JSONB NOT NULL DEFAULT '{}'::jsonb,-- e.g. {"mapping":"index-from-status"} for backfill
    PRIMARY KEY (run_id, gpu_id, role)
);
CREATE TABLE run_model (                             -- extra model relations (baselines, teachers, outputs)
    run_id    BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    model_id  BIGINT NOT NULL REFERENCES model(id),
    role      TEXT NOT NULL,                         -- 'teacher','baseline','output','base','delta'
    PRIMARY KEY (run_id, model_id, role)
);
CREATE TABLE metric (                                -- every scalar: curves (step>0) and summaries (step=0)
    run_id       BIGINT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    name         TEXT NOT NULL,                      -- 'loss','loss/speech_out','tok_s','column_a','content_pass'
    step         BIGINT NOT NULL DEFAULT 0,
    value        DOUBLE PRECISION NOT NULL,
    denominator  DOUBLE PRECISION,                   -- 'x/15' style summaries keep the denominator
    ts           TIMESTAMPTZ,
    extra        JSONB,
    PRIMARY KEY (run_id, name, step)
);
CREATE TABLE run_event (                             -- parsed status-file lines
    id         BIGSERIAL PRIMARY KEY,
    run_id     BIGINT REFERENCES run(id) ON DELETE CASCADE,
    ts         TIMESTAMPTZ,
    state      TEXT NOT NULL,                        -- START|PASS|FAIL|STOP|REFUSED|SKIP|exit
    line       TEXT NOT NULL,
    src_path TEXT NOT NULL, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    UNIQUE (src_path, line)
);

-- ---------- test suites / cases (cases are global, suites link to them) ----------
CREATE TABLE test_suite (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL,                   -- 'bank_v1','toolcall_v1','probe','smoke','codex_combined_oct13',...
    version         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'sealed' CHECK (status IN ('draft','sealed','retired')),
    modality_in     TEXT[] NOT NULL DEFAULT '{text}',
    modality_out    TEXT[] NOT NULL DEFAULT '{text}',
    default_grader  TEXT,                            -- grader.name, NULL = human-graded
    bank_sha256     TEXT,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    UNIQUE (name, version)
);
CREATE TABLE test_case (
    id              BIGSERIAL PRIMARY KEY,
    case_id         TEXT NOT NULL,                   -- stable id from the bank ('bank-v1-07', task id)
    content_sha256  TEXT NOT NULL,                   -- sha256 of canonical JSON {prompt, tools, media}; an edited prompt is a new row
    category        TEXT,
    lane            TEXT,
    tags            TEXT[] NOT NULL DEFAULT '{}',
    modality_in     TEXT[] NOT NULL DEFAULT '{text}',
    modality_out    TEXT[] NOT NULL DEFAULT '{text}',
    prompt          JSONB NOT NULL,                  -- messages[] exactly as the bank stores them
    tools           JSONB,
    media           JSONB,                           -- stimuli refs {kind, path, sha256}
    reference       JSONB,                           -- expected / oracle / rubric
    grader          TEXT,                            -- overrides suite default
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    UNIQUE (case_id, content_sha256)
);
CREATE INDEX test_case_case_id ON test_case(case_id);
CREATE TABLE suite_member (
    suite_id  BIGINT NOT NULL REFERENCES test_suite(id) ON DELETE CASCADE,
    case_pk   BIGINT NOT NULL REFERENCES test_case(id),
    ordinal   INTEGER NOT NULL,                      -- position in the bank file (dossiers map positionally)
    PRIMARY KEY (suite_id, case_pk),
    UNIQUE (suite_id, ordinal)
);
CREATE INDEX suite_member_case ON suite_member(case_pk);

CREATE TABLE grader (
    id               BIGSERIAL PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,           -- 'toolcall_selfgen.grade_case','oc_calib.classify_form','human:cyril',...
    kind             TEXT NOT NULL CHECK (kind IN ('machine','human')),
    tool_version_id  BIGINT REFERENCES tool_version(id),
    scale            TEXT NOT NULL,                  -- 'pass/fail','correct/partial/miss/none','0-1','count'
    description      TEXT
);

-- A named, re-runnable test definition: "test1 + configs".
CREATE TABLE test (
    id             BIGSERIAL PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    suite_name     TEXT NOT NULL,
    suite_version  TEXT,                             -- NULL = newest version at run time
    engine         TEXT NOT NULL,                    -- 'auto','hf','chimera','qwen35-student','gguf'
    gen            JSONB NOT NULL DEFAULT '{}'::jsonb,   -- cap, thinking, temperature, seed, dtype, split...
    devices        INTEGER[],                        -- preferred card indexes; run resolves to gpu uuids
    case_filter    TEXT[],                           -- NULL = whole suite
    notes          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- eval runs / results / grades ----------
CREATE TABLE eval_run (
    run_id                BIGINT PRIMARY KEY REFERENCES run(id) ON DELETE CASCADE,
    model_id              BIGINT NOT NULL REFERENCES model(id),
    suite_id              BIGINT NOT NULL REFERENCES test_suite(id),
    test_id               BIGINT REFERENCES test(id),
    engine                TEXT NOT NULL,             -- 'hf','chimera','qwen35-student','gguf'
    gen                   JSONB NOT NULL DEFAULT '{}'::jsonb,  -- cap, temperature, top_p, seed, eos, extra_stop_token_ids, ledger band mask, split, dtype
    tokenizer_sha256      TEXT,
    chat_template_sha256  TEXT,
    bank_file_sha256      TEXT,
    thinking_requested    TEXT,                      -- 'template-default','off','on'
    thinking_took_effect  BOOLEAN,
    case_filter           TEXT[],
    label                 TEXT,                      -- dossier label e.g. 'OC02-OMNI1-NATIVE-NOTHINK'
    baseline_run_id       BIGINT REFERENCES run(id)
);
CREATE INDEX eval_run_model_suite ON eval_run(model_id, suite_id);
CREATE INDEX eval_run_test ON eval_run(test_id);

CREATE TABLE eval_result (
    id              BIGSERIAL PRIMARY KEY,
    eval_run_id     BIGINT NOT NULL REFERENCES eval_run(run_id) ON DELETE CASCADE,
    case_pk         BIGINT NOT NULL REFERENCES test_case(id),
    output_text     TEXT,
    output_tokens   INTEGER,
    termination     TEXT CHECK (termination IN ('eos','cap','stop','loop','nan','error','timeout',
                                                'unsupported-modality','other')),
    form            TEXT,                            -- oc_calib.classify_form
    artifact_flags  TEXT[] NOT NULL DEFAULT '{}',    -- 'loop','echo','think-leak','empty','tool-parse-fail'
    finite          BOOLEAN,
    latency_ms      INTEGER,
    tok_s           DOUBLE PRECISION,
    raw             JSONB,                           -- parsed tool calls, logits summary, ...
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    UNIQUE (eval_run_id, case_pk)
);
CREATE INDEX eval_result_case ON eval_result(case_pk);

CREATE TABLE eval_grade (
    eval_result_id  BIGINT NOT NULL REFERENCES eval_result(id) ON DELETE CASCADE,
    grader_id       BIGINT NOT NULL REFERENCES grader(id),
    grade           TEXT NOT NULL,                   -- 'correct','partial','miss','none','pass','fail'
    score           DOUBLE PRECISION,
    why             TEXT,
    detail          JSONB,
    graded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id),
    PRIMARY KEY (eval_result_id, grader_id)
);

CREATE TABLE eval_artifact (                         -- non-text outputs: image/audio/video/mesh/latent
    id               BIGSERIAL PRIMARY KEY,
    eval_result_id   BIGINT NOT NULL REFERENCES eval_result(id) ON DELETE CASCADE,
    kind             TEXT NOT NULL CHECK (kind IN ('image','audio','video','mesh','latent','text','other')),
    path             TEXT NOT NULL,
    sha256           TEXT,
    bytes            BIGINT,
    duration_s       DOUBLE PRECISION,
    width            INTEGER,
    height           INTEGER,
    sample_rate      INTEGER,
    metrics          JSONB NOT NULL DEFAULT '{}'::jsonb,   -- psnr_db, clip score, finite, ...
    UNIQUE (eval_result_id, path)
);

-- ---------- experiments: fold / arch / distill / data / kernel ----------
CREATE TABLE technique (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,               -- 'C2 MLP coreset','vocab K','token-bridge dual route'
    family       TEXT,                               -- 'fold','distill','arch','kernel','data'
    description  TEXT
);
CREATE TABLE experiment (
    id                BIGSERIAL PRIMARY KEY,
    project           TEXT NOT NULL REFERENCES project(key),
    key               TEXT NOT NULL UNIQUE,          -- 'NF-10','MX-4X','S3-C2K4352','X3-08B',...
    kind              TEXT NOT NULL CHECK (kind IN ('fold','arch','distill','data','kernel','other')),
    technique_id      BIGINT REFERENCES technique(id),
    parent_id         BIGINT REFERENCES model(id),
    child_id          BIGINT REFERENCES model(id),
    run_id            BIGINT REFERENCES run(id),
    ratio_body        NUMERIC,                       -- child/parent params (body)
    ratio_component   NUMERIC,
    params            JSONB NOT NULL DEFAULT '{}'::jsonb,   -- rank, ratio, calib tokens, objective, steps
    budget            JSONB NOT NULL DEFAULT '{}'::jsonb,   -- tokens, steps, gpu-hours
    behaviour_legacy  TEXT,                          -- '0/15', '8/15 B1' as written in the source
    verdict           TEXT,
    evidence_level    TEXT NOT NULL CHECK (evidence_level IN ('receipt','report','memory','codex','unverified')),
    results_path      TEXT,
    date              DATE,
    detail            JSONB NOT NULL DEFAULT '{}'::jsonb,
    src_path TEXT, src_sha256 TEXT, src_host TEXT, ingest_batch_id BIGINT REFERENCES ingest_batch(id)
);
CREATE INDEX experiment_kind_technique ON experiment(kind, technique_id);
CREATE TABLE experiment_eval (
    experiment_id  BIGINT NOT NULL REFERENCES experiment(id) ON DELETE CASCADE,
    eval_run_id    BIGINT NOT NULL REFERENCES eval_run(run_id),
    role           TEXT NOT NULL CHECK (role IN ('child','parent','baseline','other')),
    PRIMARY KEY (experiment_id, eval_run_id)
);
