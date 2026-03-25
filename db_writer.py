"""Background DB writer for demo_detection_realtime.

All writes happen on a dedicated queue-backed thread so detector/video threads
never block on database I/O.
"""

import os
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime

try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

from db_config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    STATS_ENABLED_DEFAULT,
    WRITE_QUEUE_MAXSIZE,
)

_SQLITE_PATH = os.path.join(os.path.dirname(__file__), "data", "tracking_demo.db")

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    started_at      TEXT,
    ended_at        TEXT,
    checkpoint_id   TEXT DEFAULT '',
    camera_source   TEXT DEFAULT '',
    total           INTEGER DEFAULT 0,
    ok_count        INTEGER DEFAULT 0,
    nok_no_barcode  INTEGER DEFAULT 0,
    nok_no_date     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stats_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    captured_at     TEXT,
    total           INTEGER DEFAULT 0,
    ok_count        INTEGER DEFAULT 0,
    nok_no_barcode  INTEGER DEFAULT 0,
    nok_no_date     INTEGER DEFAULT 0,
    nok_rate        REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_snapshots_session ON stats_snapshots (session_id);
"""


def _ts():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


class DBWriter:
    def __init__(self):
        self.write_queue = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._active = STATS_ENABLED_DEFAULT
        self._current_session_id = None

        self._backend = self._detect_backend()
        self._available = self._backend != "none"

        self._sqlite_conn = None
        self._sqlite_lock = threading.Lock()
        self._pg_conn = None

        if self._backend == "sqlite":
            self._init_sqlite()

    def _detect_backend(self):
        if _PSYCOPG2_AVAILABLE:
            try:
                conn = psycopg2.connect(
                    host=DB_HOST,
                    port=DB_PORT,
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    connect_timeout=2,
                )
                conn.close()
                print("[DBWriter] Backend: PostgreSQL")
                return "postgres"
            except Exception:
                print("[DBWriter] PostgreSQL unreachable; using SQLite fallback.")
        else:
            print("[DBWriter] psycopg2 not installed; using SQLite fallback.")
        return "sqlite"

    def _init_sqlite(self):
        os.makedirs(os.path.dirname(_SQLITE_PATH), exist_ok=True)
        conn = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SQLITE_SCHEMA)
        conn.commit()
        self._sqlite_conn = conn
        print(f"[DBWriter] SQLite database: {_SQLITE_PATH}")

    def start(self):
        if not self._available or (self._thread and self._thread.is_alive()):
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="DBWriter")
        self._thread.start()
        print("[DBWriter] Writer thread started.")

    def stop(self):
        self._stop_event.set()
        try:
            self.write_queue.put_nowait({"type": "stop"})
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._sqlite_conn:
            try:
                self._sqlite_conn.close()
            except Exception:
                pass
        if self._pg_conn:
            try:
                self._pg_conn.close()
            except Exception:
                pass

    def set_active(self, active):
        with self._lock:
            self._active = bool(active)

    @property
    def is_active(self):
        with self._lock:
            return self._active

    @property
    def current_session_id(self):
        with self._lock:
            return self._current_session_id

    @property
    def backend(self):
        return self._backend

    def open_session(self, checkpoint_id="", camera_source=""):
        sid = str(uuid.uuid4())
        with self._lock:
            self._current_session_id = sid
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO sessions (id, started_at, checkpoint_id, camera_source) VALUES (%s, %s, %s, %s)",
                            (sid, _ts(), checkpoint_id, camera_source),
                        )
                    conn.commit()
            except Exception as e:
                print(f"[DBWriter] open_session PG error: {e}")
                self._pg_conn = None
        else:
            with self._sqlite_lock:
                self._sqlite_conn.execute(
                    "INSERT INTO sessions (id, started_at, checkpoint_id, camera_source) VALUES (?, ?, ?, ?)",
                    (sid, _ts(), checkpoint_id, camera_source),
                )
                self._sqlite_conn.commit()
        return sid

    def close_session(self, session_id, totals=None):
        if not self._available or not session_id:
            return
        totals = totals or {}
        params = (
            _ts(),
            totals.get("total", 0),
            totals.get("ok_count", 0),
            totals.get("nok_no_barcode", 0),
            totals.get("nok_no_date", 0),
            session_id,
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE sessions SET ended_at=%s, total=%s, ok_count=%s, nok_no_barcode=%s, nok_no_date=%s WHERE id=%s",
                            params,
                        )
                    conn.commit()
            except Exception as e:
                print(f"[DBWriter] close_session PG error: {e}")
                self._pg_conn = None
        else:
            with self._sqlite_lock:
                self._sqlite_conn.execute(
                    "UPDATE sessions SET ended_at=?, total=?, ok_count=?, nok_no_barcode=?, nok_no_date=? WHERE id=?",
                    params,
                )
                self._sqlite_conn.commit()
        with self._lock:
            if self._current_session_id == session_id:
                self._current_session_id = None

    def get_session_kpis(self, session_id):
        if not self._available or not session_id:
            return {}
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return {}
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
                    row = cur.fetchone()
                    return dict(row) if row else {}
            except Exception as e:
                print(f"[DBWriter] get_session_kpis error: {e}")
                self._pg_conn = None
                return {}
        with self._sqlite_lock:
            cur = self._sqlite_conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else {}

    def list_sessions(self, limit=50):
        if not self._available:
            return []
        sql = (
            "SELECT id, started_at, ended_at, checkpoint_id, camera_source, total, ok_count, "
            "nok_no_barcode, nok_no_date FROM sessions ORDER BY started_at DESC LIMIT "
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return []
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql + "%s", (limit,))
                    return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"[DBWriter] list_sessions error: {e}")
                self._pg_conn = None
                return []
        with self._sqlite_lock:
            cur = self._sqlite_conn.execute(sql + "?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def list_snapshots(self, session_id, limit=500):
        if not self._available or not session_id:
            return []
        sql = (
            "SELECT id, session_id, captured_at, total, ok_count, nok_no_barcode, nok_no_date, nok_rate "
            "FROM stats_snapshots WHERE session_id = "
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return []
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql + "%s ORDER BY captured_at ASC LIMIT %s", (session_id, limit))
                    return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"[DBWriter] list_snapshots error: {e}")
                self._pg_conn = None
                return []
        with self._sqlite_lock:
            cur = self._sqlite_conn.execute(sql + "? ORDER BY captured_at ASC LIMIT ?", (session_id, limit))
            return [dict(r) for r in cur.fetchall()]

    def _run(self):
        while not self._stop_event.is_set():
            try:
                event = self.write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event.get("type") == "stop":
                break

            if not self.is_active:
                self.write_queue.task_done()
                continue

            try:
                if event.get("type") == "snapshot":
                    self._write_snapshot(event)
            except Exception as e:
                print(f"[DBWriter] Event write error: {e}")
                self._pg_conn = None

            self.write_queue.task_done()

    def _write_snapshot(self, ev):
        total = ev.get("total", 0)
        nok = total - ev.get("ok_count", 0)
        rate = round(nok / total * 100, 2) if total > 0 else 0.0
        params = (
            ev.get("session_id"),
            _ts(),
            total,
            ev.get("ok_count", 0),
            ev.get("nok_no_barcode", 0),
            ev.get("nok_no_date", 0),
            rate,
        )
        sql = (
            "INSERT INTO stats_snapshots (session_id, captured_at, total, ok_count, "
            "nok_no_barcode, nok_no_date, nok_rate) VALUES "
        )
        if self._backend == "postgres":
            conn = self._get_pg_conn()
            if not conn:
                return
            with conn.cursor() as cur:
                cur.execute(sql + "(%s,%s,%s,%s,%s,%s,%s)", params)
            conn.commit()
        else:
            with self._sqlite_lock:
                self._sqlite_conn.execute(sql + "(?,?,?,?,?,?,?)", params)
                self._sqlite_conn.commit()

    def _get_pg_conn(self):
        if not _PSYCOPG2_AVAILABLE:
            return None
        if self._pg_conn is not None:
            try:
                if self._pg_conn.closed == 0:
                    return self._pg_conn
            except Exception:
                pass
        for _ in range(3):
            try:
                self._pg_conn = psycopg2.connect(
                    host=DB_HOST,
                    port=DB_PORT,
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    connect_timeout=3,
                )
                self._pg_conn.autocommit = False
                return self._pg_conn
            except Exception:
                time.sleep(1.0)
        self._pg_conn = None
        return None
