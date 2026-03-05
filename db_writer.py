"""
DBWriter — Thread 4 (background, fully decoupled from detection).

Pulls events from a non-blocking queue and writes to the database.
Never blocks the detection loop.

Backend priority:
  1. PostgreSQL  — if psycopg2 is installed AND Docker container is reachable
  2. SQLite      — automatic fallback, zero install, file at ./data/tracking.db

Events understood:
  {"type": "snapshot", "session_id", "total", "ok_count",
   "nok_no_barcode", "nok_no_date", "nok_both"}

  {"type": "stop"}   — signals the writer thread to exit cleanly
"""

import os
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime

# ──────────────────────────────────────────────
# Optional PostgreSQL support
# ──────────────────────────────────────────────
try:
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

from db_config import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    STATS_ENABLED_DEFAULT, SNAPSHOT_EVERY_N_PACKETS,
    WRITE_QUEUE_MAXSIZE,
)

# SQLite fallback path
_SQLITE_PATH = os.path.join(os.path.dirname(__file__), "data", "tracking.db")

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
    nok_no_date     INTEGER DEFAULT 0,
    nok_both        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stats_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    captured_at     TEXT,
    total           INTEGER DEFAULT 0,
    ok_count        INTEGER DEFAULT 0,
    nok_no_barcode  INTEGER DEFAULT 0,
    nok_no_date     INTEGER DEFAULT 0,
    nok_both        INTEGER DEFAULT 0,
    nok_rate        REAL    DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_snapshots_session ON stats_snapshots (session_id);
"""


def _ts() -> str:
    """Local datetime string — simple format e.g. 2026-03-05T09:00:46"""
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


class DBWriter:
    """
    Background writer thread — auto-selects PostgreSQL or SQLite.

    Usage:
        writer = DBWriter()
        writer.start()
        sid = writer.open_session(checkpoint_id="...", camera_source="cam0")
        writer.write_queue.put_nowait({"type": "packet", ...})
        writer.close_session(sid, totals={...})
        writer.stop()
    """

    def __init__(self):
        self.write_queue  = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)
        self._thread      = None
        self._stop_event  = threading.Event()
        self._lock        = threading.Lock()
        self._active      = STATS_ENABLED_DEFAULT
        self._current_session_id = None

        # Determine backend
        self._backend     = self._detect_backend()
        self._available   = self._backend != "none"

        # SQLite: one shared connection (WAL mode allows concurrent reads)
        self._sqlite_conn = None
        self._sqlite_lock = threading.Lock()

        # PostgreSQL connection (writer thread + Flask query calls)
        self._pg_conn     = None

        if self._backend == "sqlite":
            self._init_sqlite()

    # ──────────────────────────────────────────────
    # Backend detection
    # ──────────────────────────────────────────────

    def _detect_backend(self) -> str:
        if _PSYCOPG2_AVAILABLE:
            try:
                conn = psycopg2.connect(
                    host=DB_HOST, port=DB_PORT,
                    dbname=DB_NAME, user=DB_USER,
                    password=DB_PASSWORD,
                    connect_timeout=2,
                )
                conn.close()
                print("[DBWriter] Backend: PostgreSQL")
                return "postgres"
            except Exception:
                print("[DBWriter] PostgreSQL unreachable — falling back to SQLite.")
        else:
            print("[DBWriter] psycopg2 not installed — using SQLite fallback.")
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

    # ──────────────────────────────────────────────
    # Public control
    # ──────────────────────────────────────────────

    def start(self):
        """Start the background writer thread."""
        if not self._available:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="DBWriter"
        )
        self._thread.start()
        print("[DBWriter] Writer thread started.")

    def stop(self):
        """Signal the writer thread to stop and wait for it."""
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
        print("[DBWriter] Writer thread stopped.")

    def set_active(self, active: bool):
        with self._lock:
            self._active = active
        print(f"[DBWriter] Stats recording {'ENABLED' if active else 'DISABLED'}.")

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def current_session_id(self):
        with self._lock:
            return self._current_session_id

    # ──────────────────────────────────────────────
    # Session management
    # ──────────────────────────────────────────────

    def open_session(self, checkpoint_id: str = "", camera_source: str = "") -> str:
        sid = str(uuid.uuid4())
        with self._lock:
            self._current_session_id = sid
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO sessions (id, started_at, checkpoint_id, camera_source) "
                            "VALUES (%s, %s, %s, %s)",
                            (sid, _ts(), checkpoint_id, camera_source)
                        )
                    conn.commit()
            except Exception as e:
                print(f"[DBWriter] open_session PG error: {e}")
                self._pg_conn = None
        elif self._backend == "sqlite":
            with self._sqlite_lock:
                self._sqlite_conn.execute(
                    "INSERT INTO sessions (id, started_at, checkpoint_id, camera_source) "
                    "VALUES (?, ?, ?, ?)",
                    (sid, _ts(), checkpoint_id, camera_source)
                )
                self._sqlite_conn.commit()
        print(f"[DBWriter] Session opened: {sid[:8]}... (backend={self._backend})")
        return sid

    def close_session(self, session_id: str, totals: dict = None):
        if not self._available or not session_id:
            return
        totals = totals or {}
        params = (
            _ts(),
            totals.get("total", 0),
            totals.get("ok_count", 0),
            totals.get("nok_no_barcode", 0),
            totals.get("nok_no_date", 0),
            totals.get("nok_both", 0),
            session_id,
        )
        sql_pg = ("UPDATE sessions SET ended_at=%s, total=%s, ok_count=%s, "
                  "nok_no_barcode=%s, nok_no_date=%s, nok_both=%s WHERE id=%s")
        sql_sl = ("UPDATE sessions SET ended_at=?, total=?, ok_count=?, "
                  "nok_no_barcode=?, nok_no_date=?, nok_both=? WHERE id=?")
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(sql_pg, params)
                    conn.commit()
            except Exception as e:
                print(f"[DBWriter] close_session PG error: {e}")
                self._pg_conn = None
        elif self._backend == "sqlite":
            with self._sqlite_lock:
                self._sqlite_conn.execute(sql_sl, params)
                self._sqlite_conn.commit()
        print(f"[DBWriter] Session closed: {session_id[:8]}...")

    # ──────────────────────────────────────────────
    # KPI queries (called by Flask endpoints)
    # ──────────────────────────────────────────────

    def get_session_kpis(self, session_id: str) -> dict:
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
        else:
            with self._sqlite_lock:
                cur = self._sqlite_conn.execute(
                    "SELECT * FROM sessions WHERE id = ?", (session_id,)
                )
                row = cur.fetchone()
                return dict(row) if row else {}

    def list_sessions(self, limit: int = 50) -> list:
        if not self._available:
            return []
        sql = ("SELECT id, started_at, ended_at, checkpoint_id, camera_source, "
               "total, ok_count, nok_no_barcode, nok_no_date, nok_both "
               "FROM sessions ORDER BY started_at DESC LIMIT ")
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
        else:
            with self._sqlite_lock:
                cur = self._sqlite_conn.execute(sql + "?", (limit,))
                return [dict(r) for r in cur.fetchall()]

    def export_csv(self, session_id: str = None) -> str:
        """Export session summary row(s) as CSV."""
        header = "id,started_at,ended_at,total,ok_count,nok_no_barcode,nok_no_date,nok_both,nok_rate"
        if session_id:
            sessions = [self.get_session_kpis(session_id)]
        else:
            sessions = self.list_sessions(limit=1000)
        sessions = [s for s in sessions if s]
        if not sessions:
            return header + "\n"
        lines = [header]
        for s in sessions:
            total = s.get('total', 0)
            ok    = s.get('ok_count', 0)
            nok   = total - ok
            rate  = round(nok / total * 100, 2) if total > 0 else 0.0
            lines.append(
                f"{s.get('id','')},{s.get('started_at','')},{s.get('ended_at','')}"
                f",{total},{ok},{s.get('nok_no_barcode',0)}"
                f",{s.get('nok_no_date',0)},{s.get('nok_both',0)},{rate}"
            )
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # Background thread
    # ──────────────────────────────────────────────

    def _run(self):
        print("[DBWriter] Writer loop started.")
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
                if event["type"] == "snapshot":
                    self._write_snapshot(event)
            except Exception as e:
                print(f"[DBWriter] Event write error: {e}")
                self._pg_conn = None

            self.write_queue.task_done()

        print("[DBWriter] Writer loop exited.")

    def _write_snapshot(self, ev: dict):
        total = ev.get("total", 0)
        nok   = total - ev.get("ok_count", 0)
        rate  = round(nok / total * 100, 2) if total > 0 else 0.0
        sid   = ev.get("session_id")
        params = (
            sid, _ts(), total,
            ev.get("ok_count", 0),
            ev.get("nok_no_barcode", 0),
            ev.get("nok_no_date", 0),
            ev.get("nok_both", 0),
            rate,
        )
        sql = ("INSERT INTO stats_snapshots "
               "(session_id, captured_at, total, ok_count, "
               "nok_no_barcode, nok_no_date, nok_both, nok_rate) "
               "VALUES ")
        if self._backend == "postgres":
            conn = self._get_pg_conn()
            if not conn:
                return
            with conn.cursor() as cur:
                cur.execute(sql + "(%s,%s,%s,%s,%s,%s,%s,%s)", params)
            conn.commit()
        else:
            with self._sqlite_lock:
                self._sqlite_conn.execute(sql + "(?,?,?,?,?,?,?,?)", params)
                self._sqlite_conn.commit()

    # ──────────────────────────────────────────────
    # PostgreSQL connection management
    # ──────────────────────────────────────────────

    def _get_pg_conn(self):
        """Return an open psycopg2 connection, reconnecting if needed."""
        if not _PSYCOPG2_AVAILABLE:
            return None
        if self._pg_conn is not None:
            try:
                if self._pg_conn.closed == 0:
                    return self._pg_conn
            except Exception:
                pass
        for attempt in range(3):
            try:
                self._pg_conn = psycopg2.connect(
                    host=DB_HOST, port=DB_PORT,
                    dbname=DB_NAME, user=DB_USER,
                    password=DB_PASSWORD,
                    connect_timeout=3,
                )
                self._pg_conn.autocommit = False
                return self._pg_conn
            except Exception as e:
                if attempt == 0:
                    print(f"[DBWriter] PG connect failed: {e}")
                time.sleep(1.0)
        self._pg_conn = None
        return None
