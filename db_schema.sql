-- =====================================================
-- Tracking Live — PostgreSQL Schema
-- Run automatically by Docker on first startup via
-- /docker-entrypoint-initdb.d/schema.sql
-- =====================================================

-- One row per production run (created on server start or /api/stats/reset)
CREATE TABLE IF NOT EXISTS sessions (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at         TIMESTAMPTZ,
    checkpoint_id    TEXT,
    camera_source    TEXT,
    total            INT         NOT NULL DEFAULT 0,
    ok_count         INT         NOT NULL DEFAULT 0,
    nok_no_barcode   INT         NOT NULL DEFAULT 0,  -- barcode missing, date present (or irrelevant)
    nok_no_date      INT         NOT NULL DEFAULT 0,  -- date missing, barcode present
    nok_both         INT         NOT NULL DEFAULT 0   -- both barcode and date missing
);

-- Periodic KPI snapshots for trend charts on the dashboard
CREATE TABLE IF NOT EXISTS stats_snapshots (
    id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id     UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    captured_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    total          INT         NOT NULL DEFAULT 0,
    ok_count       INT         NOT NULL DEFAULT 0,
    nok_no_barcode INT         NOT NULL DEFAULT 0,
    nok_no_date    INT         NOT NULL DEFAULT 0,
    nok_both       INT         NOT NULL DEFAULT 0,
    nok_rate       NUMERIC(5,2) NOT NULL DEFAULT 0.0  -- % of NOK
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_snapshots_session  ON stats_snapshots(session_id);
