-- omni_lab 0005_readonly_grants -- SELECT-only access for omni_lab_ro (compare, sql, DBeaver/psql browsing).
-- The role is created by bootstrap/bootstrap_db.py; this migration is a no-op where it does not exist.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'omni_lab_ro') THEN
        EXECUTE format('GRANT CONNECT ON DATABASE %I TO omni_lab_ro', current_database());
        GRANT USAGE ON SCHEMA public TO omni_lab_ro;
        GRANT SELECT ON ALL TABLES IN SCHEMA public TO omni_lab_ro;
        -- tables and views created later by the owning role (omni_lab) inherit SELECT
        ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO omni_lab_ro;
    END IF;
END
$$;
