-- omni_lab 0004_views -- read surfaces. All long-form; the CLI pivots.
-- Later migrations change a view with CREATE OR REPLACE VIEW (or DROP + CREATE when columns change).

-- Effective grade per result: newest human grade wins over newest machine grade.
CREATE VIEW v_effective_grade AS
SELECT res.id AS eval_result_id,
       COALESCE(gh.grade, gm.grade) AS grade,
       COALESCE(gh.score, gm.score) AS score,
       gh.grade AS human_grade,
       gm.grade AS machine_grade
FROM eval_result res
LEFT JOIN LATERAL (SELECT g.grade, g.score FROM eval_grade g JOIN grader gr ON gr.id = g.grader_id
                   WHERE g.eval_result_id = res.id AND gr.kind = 'human'
                   ORDER BY g.graded_at DESC LIMIT 1) gh ON true
LEFT JOIN LATERAL (SELECT g.grade, g.score FROM eval_grade g JOIN grader gr ON gr.id = g.grader_id
                   WHERE g.eval_result_id = res.id AND gr.kind = 'machine'
                   ORDER BY g.graded_at DESC LIMIT 1) gm ON true;

-- Per-case long form. A missing (eval_run, case) pair has no row: that is the blank cell.
CREATE VIEW v_eval_matrix AS
SELECT er.run_id        AS eval_run_id,
       r.name           AS run_name,
       t.name           AS test,
       r.started_at,
       m.model_key,
       m.stage,
       er.engine,
       ts.name          AS suite,
       ts.version       AS suite_version,
       tc.id            AS case_pk,
       tc.case_id,
       tc.content_sha256,
       tc.category,
       res.termination,
       res.artifact_flags,
       res.latency_ms,
       eg.grade,
       eg.score,
       eg.human_grade,
       eg.machine_grade,
       res.id           AS eval_result_id
FROM eval_run er
JOIN run r          ON r.id = er.run_id
JOIN model m        ON m.id = er.model_id
JOIN test_suite ts  ON ts.id = er.suite_id
LEFT JOIN test t    ON t.id = er.test_id
JOIN eval_result res ON res.eval_run_id = er.run_id
JOIN test_case tc   ON tc.id = res.case_pk
JOIN v_effective_grade eg ON eg.eval_result_id = res.id;

-- One row per eval run: "test | configs | results" in one line.
CREATE VIEW v_results AS
SELECT er.run_id AS eval_run_id,
       t.name AS test,
       m.model_key,
       m.stage,
       ts.name || '@' || ts.version AS suite,
       er.engine,
       r.config->>'dtype' AS dtype,
       (SELECT string_agg(g.gpu_index::text || ':' || g.name || COALESCE('/' || rd.kernel_path, ''), ', ' ORDER BY g.gpu_index)
          FROM run_device rd JOIN gpu g ON g.id = rd.gpu_id WHERE rd.run_id = r.id) AS devices,
       CASE WHEN e.id IS NULL THEN NULL
            ELSE concat_ws(' ', 'torch=' || e.torch, 'tf=' || e.transformers, 'fla=' || e.fla,
                           'llama=' || e.llama_cpp_build, 'code=' || left(e.code_commit, 10)) END AS env,
       er.gen->>'cap' AS cap,
       er.thinking_requested,
       er.thinking_took_effect,
       count(res.id) FILTER (WHERE eg.grade = 'correct') AS correct,
       count(res.id) FILTER (WHERE eg.grade = 'partial') AS partial,
       count(res.id) FILTER (WHERE eg.grade = 'miss')    AS miss,
       count(res.id) FILTER (WHERE eg.grade = 'none')    AS "none",
       count(res.id) FILTER (WHERE eg.grade = 'pass')    AS pass,
       count(res.id) FILTER (WHERE eg.grade = 'fail')    AS fail,
       count(res.id) FILTER (WHERE res.id IS NOT NULL AND eg.grade IS NULL) AS ungraded,
       count(res.id) FILTER (WHERE res.termination = 'unsupported-modality') AS unsupported,
       count(res.id) AS n_results,
       (SELECT count(*) FROM suite_member sm WHERE sm.suite_id = er.suite_id) AS n_cases,
       round(count(res.id)::numeric / NULLIF((SELECT count(*) FROM suite_member sm WHERE sm.suite_id = er.suite_id), 0), 3) AS coverage,
       r.status,
       r.started_at,
       r.config_sha256
FROM eval_run er
JOIN run r         ON r.id = er.run_id
JOIN model m       ON m.id = er.model_id
JOIN test_suite ts ON ts.id = er.suite_id
LEFT JOIN test t   ON t.id = er.test_id
LEFT JOIN env e    ON e.id = r.env_id
LEFT JOIN eval_result res ON res.eval_run_id = er.run_id
LEFT JOIN v_effective_grade eg ON eg.eval_result_id = res.id
GROUP BY er.run_id, t.name, m.model_key, m.stage, ts.name, ts.version, er.engine, r.id, e.id;

CREATE VIEW v_run_summary AS
SELECT r.id, r.run_key, r.kind, r.name, r.status, r.owner,
       r.project, l.key AS lane,
       m.model_key, m.stage,
       tl.name AS tool, tv.version AS tool_version, left(tv.sha256, 12) AS tool_sha,
       r.started_at, r.ended_at,
       COALESCE(r.wall_seconds, EXTRACT(EPOCH FROM (r.ended_at - r.started_at))) AS wall_seconds,
       r.exit_code,
       (SELECT string_agg(g.gpu_index::text || ':' || g.name || COALESCE('/' || rd.kernel_path, ''), ', ' ORDER BY g.gpu_index)
          FROM run_device rd JOIN gpu g ON g.id = rd.gpu_id WHERE rd.run_id = r.id) AS devices,
       r.config->>'dtype'  AS dtype,
       r.config->>'engine' AS engine,
       r.config_sha256,
       (SELECT count(*) FROM eval_result e WHERE e.eval_run_id = r.id) AS n_results,
       (SELECT count(*) FROM metric mt WHERE mt.run_id = r.id) AS n_metrics,
       (SELECT mt.value FROM metric mt WHERE mt.run_id = r.id AND mt.name = 'loss' ORDER BY mt.step DESC LIMIT 1) AS last_loss,
       r.receipt_path, r.log_path
FROM run r
LEFT JOIN lane l          ON l.id = r.lane_id
LEFT JOIN model m         ON m.id = r.model_id
LEFT JOIN tool_version tv ON tv.id = r.tool_version_id
LEFT JOIN tool tl         ON tl.id = tv.tool_id;

CREATE VIEW v_latest_models AS
SELECT m.id, m.model_key, m.ledger_checkpoint_id, m.project, m.family, m.role, m.stage, m.layout,
       m.parameter_count, m.path, m.status, m.sealed_verdict, m.created_at,
       (SELECT string_agg(pm.model_key || ' (' || ml.relation || ')', ', ' ORDER BY ml.ordinal)
          FROM model_lineage ml JOIN model pm ON pm.id = ml.parent_id WHERE ml.child_id = m.id) AS parents,
       (SELECT max(r.started_at) FROM eval_run er JOIN run r ON r.id = er.run_id WHERE er.model_id = m.id) AS last_eval_at,
       (SELECT count(*) FROM model_lineage ml WHERE ml.parent_id = m.id) AS n_children
FROM model m
WHERE m.status = 'present'
ORDER BY m.family, m.role, m.created_at DESC NULLS LAST;

-- Which modalities each model actually handled, measured from results (not config flags).
CREATE VIEW v_model_modality_coverage AS
WITH per_case AS (
    SELECT er.model_id, 'in'::text AS direction, mod.modality, r.started_at, res.termination, eg.grade, eg.score
    FROM eval_run er
    JOIN run r ON r.id = er.run_id
    JOIN eval_result res ON res.eval_run_id = er.run_id
    JOIN test_case tc ON tc.id = res.case_pk
    JOIN v_effective_grade eg ON eg.eval_result_id = res.id
    CROSS JOIN LATERAL unnest(tc.modality_in) AS mod(modality)
    UNION ALL
    SELECT er.model_id, 'out'::text, mod.modality, r.started_at, res.termination, eg.grade, eg.score
    FROM eval_run er
    JOIN run r ON r.id = er.run_id
    JOIN eval_result res ON res.eval_run_id = er.run_id
    JOIN test_case tc ON tc.id = res.case_pk
    JOIN v_effective_grade eg ON eg.eval_result_id = res.id
    CROSS JOIN LATERAL unnest(tc.modality_out) AS mod(modality)
)
SELECT m.model_key, m.stage, pc.direction, pc.modality,
       count(*) AS n_results,
       count(*) FILTER (WHERE pc.grade IN ('correct','pass') OR pc.score >= 1) AS n_pass,
       count(*) FILTER (WHERE pc.termination = 'unsupported-modality') AS n_unsupported,
       count(*) FILTER (WHERE pc.grade IS NULL) AS n_ungraded,
       round((count(*) FILTER (WHERE pc.grade IN ('correct','pass') OR pc.score >= 1))::numeric / count(*), 3) AS pass_rate,
       max(pc.started_at) AS last_eval_at
FROM per_case pc
JOIN model m ON m.id = pc.model_id
GROUP BY m.model_key, m.stage, pc.direction, pc.modality;

-- Architecture conformance: requirement checks plus the tensor groups that are NOT shared across modalities.
CREATE VIEW v_arch_conformance AS
SELECT m.model_key, m.stage, ar.id AS requirement, ar.text AS requirement_text,
       ac.status, ac.evidence, ac.checked_at,
       (SELECT count(*) FROM model_tensor_group g WHERE g.model_id = m.id AND g.modality_scope NOT IN ('shared','unknown'))
           AS n_modality_specific_groups
FROM model m
CROSS JOIN arch_requirement ar
LEFT JOIN arch_check ac ON ac.model_id = m.id AND ac.requirement_id = ar.id
WHERE m.status = 'present';

CREATE VIEW v_directives_by_topic AS
SELECT t.topic, d.said_at, d.said_at_verified, d.channel, d.directive_key, d.verbatim, d.summary, d.supersedes_id
FROM directive d, LATERAL unnest(d.topics) AS t(topic)
ORDER BY t.topic, d.said_at;
