-- ──────────────── SESSIONS ────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at         TIMESTAMPTZ,
    checkpoint_id    TEXT,
    camera_source    TEXT,
    total            INT         NOT NULL DEFAULT 0,
    ok_count         INT         NOT NULL DEFAULT 0,
    nok_no_barcode   INT         NOT NULL DEFAULT 0,
    nok_no_date      INT         NOT NULL DEFAULT 0,
    nok_both         INT         NOT NULL DEFAULT 0
);

-- ──────────────── IMAGES ────────────────
CREATE TABLE IF NOT EXISTS images (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID        NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    packet_number   INT         NOT NULL,
    defect_type     TEXT        NOT NULL,  -- 'no_barcode', 'no_date', 'both', 'anomaly'
    image_path      TEXT        NOT NULL,
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(session_id, packet_number)
);

-- ──────────────── INDEXES ────────────────
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_images_session_id ON images(session_id);
CREATE INDEX IF NOT EXISTS idx_images_captured_at ON images(captured_at);