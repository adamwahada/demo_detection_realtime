CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    started_at      TEXT,
    ended_at        TEXT,
    checkpoint_id   TEXT DEFAULT '',
    camera_source   TEXT DEFAULT '',
    total           INTEGER DEFAULT 0,
    ok_count        INTEGER DEFAULT 0,
    nok_no_barcode  INTEGER DEFAULT 0,
    nok_no_date     INTEGER DEFAULT 0,
    nok_anomaly     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS defective_packets (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(id),
    packet_num      INTEGER NOT NULL,
    defect_type     TEXT NOT NULL,
    crossed_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_defective_session ON defective_packets (session_id);
