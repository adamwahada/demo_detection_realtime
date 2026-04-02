import os
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2.pool import ThreadedConnectionPool
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

_TUNIS_TZ = ZoneInfo("Africa/Tunis")
_SQLITE_PATH = os.path.join(os.path.dirname(__file__), "data", "tracking_demo.db")

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    group_id        TEXT DEFAULT '',
    shift_id        TEXT DEFAULT '',
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
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    packet_num      INTEGER NOT NULL,
    defect_type     TEXT NOT NULL,
    crossed_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_defective_session ON defective_packets (session_id);

CREATE TABLE IF NOT EXISTS shifts (
    id              TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'recurring',
    start_time      TEXT NOT NULL,
    end_time        TEXT NOT NULL,
    session_date    TEXT,
    days_of_week    TEXT NOT NULL DEFAULT '[]',
    camera_source   TEXT DEFAULT '0',
    checkpoint_id   TEXT NOT NULL DEFAULT 'tracking',
    active          INTEGER DEFAULT 1,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shift_variants (
    id          TEXT PRIMARY KEY,
    shift_id    TEXT NOT NULL REFERENCES shifts(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    active      INTEGER,
    start_time  TEXT,
    end_time    TEXT,
    start_date  TEXT NOT NULL,
    end_date    TEXT NOT NULL,
    days_of_week TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _ts():
    return datetime.now(_TUNIS_TZ).replace(tzinfo=None).isoformat(timespec='seconds')


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
        self._sqlite_read_conn = None          # dedicated read connection (WAL allows concurrent reads)
        self._sqlite_read_lock = threading.Lock()
        self._pg_pool = None   # ThreadedConnectionPool (postgres only)

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
                # Auto-create missing tables in PostgreSQL
                self._ensure_pg_tables(conn)
                conn.close()
                print("[DBWriter] Backend: PostgreSQL")
                return "postgres"
            except Exception:
                print("[DBWriter] PostgreSQL unreachable; using SQLite fallback.")
        else:
            print("[DBWriter] psycopg2 not installed; using SQLite fallback.")
        return "sqlite"

    @staticmethod
    def _ensure_pg_tables(conn):
        """Create tables in PostgreSQL if they don't exist yet."""
        ddl = """
        CREATE TABLE IF NOT EXISTS sessions (
            id              TEXT PRIMARY KEY,
            group_id        TEXT DEFAULT '',
            shift_id        TEXT DEFAULT '',
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
        CREATE TABLE IF NOT EXISTS shifts (
            id              TEXT PRIMARY KEY,
            label           TEXT NOT NULL,
            type            TEXT NOT NULL DEFAULT 'recurring',
            start_time      TEXT NOT NULL,
            end_time        TEXT NOT NULL,
            start_date      TEXT,
            end_date        TEXT,
            session_date    TEXT,
            days_of_week    TEXT NOT NULL DEFAULT '[]',
            camera_source   TEXT DEFAULT '0',
            checkpoint_id   TEXT NOT NULL DEFAULT 'tracking',
            active          INTEGER DEFAULT 1,
            created_at      TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shift_variants (
            id          TEXT PRIMARY KEY,
            shift_id    TEXT NOT NULL,
            kind        TEXT NOT NULL,
            active      INTEGER,
            start_time  TEXT,
            end_time    TEXT,
            start_date  TEXT NOT NULL,
            end_date    TEXT NOT NULL,
            days_of_week TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );
        """
        alter_stmts = [
            "ALTER TABLE shifts ADD COLUMN IF NOT EXISTS camera_source TEXT DEFAULT '0'",
            "ALTER TABLE shifts ADD COLUMN IF NOT EXISTS checkpoint_id TEXT NOT NULL DEFAULT 'tracking'",
            "ALTER TABLE shifts ADD COLUMN IF NOT EXISTS type TEXT DEFAULT 'recurring'",
            "ALTER TABLE shifts ADD COLUMN IF NOT EXISTS session_date TEXT",
            "ALTER TABLE shifts ADD COLUMN IF NOT EXISTS start_date TEXT",
            "ALTER TABLE shifts ADD COLUMN IF NOT EXISTS end_date TEXT",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS group_id TEXT DEFAULT ''",
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS shift_id TEXT DEFAULT ''",
            "CREATE INDEX IF NOT EXISTS idx_shifts_type ON shifts (type)",
            "CREATE INDEX IF NOT EXISTS idx_shift_variants_shift_id ON shift_variants (shift_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_group_id ON sessions (group_id)",
        ]
        try:
            with conn.cursor() as cur:
                # Prevent indefinite hangs if another connection holds a lock
                cur.execute("SET LOCAL statement_timeout = '5s'")
                cur.execute(ddl)
                for stmt in alter_stmts:
                    cur.execute(stmt)
            conn.commit()
        except Exception as e:
            print(f"[DBWriter] _ensure_pg_tables error: {e}")
            conn.rollback()

    def _init_sqlite(self):
        os.makedirs(os.path.dirname(_SQLITE_PATH), exist_ok=True)
        conn = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SQLITE_SCHEMA)
        # Migrate existing databases: add missing columns if not present
        for alter in [
            "ALTER TABLE sessions ADD COLUMN nok_anomaly INTEGER DEFAULT 0",
            "ALTER TABLE sessions ADD COLUMN group_id TEXT DEFAULT ''",
            "ALTER TABLE sessions ADD COLUMN shift_id TEXT DEFAULT ''",
            "ALTER TABLE shifts ADD COLUMN camera_source TEXT DEFAULT '0'",
            "ALTER TABLE shifts ADD COLUMN checkpoint_id TEXT NOT NULL DEFAULT 'tracking'",
            "ALTER TABLE shifts ADD COLUMN type TEXT DEFAULT 'recurring'",
            "ALTER TABLE shifts ADD COLUMN session_date TEXT",
            "ALTER TABLE shifts ADD COLUMN start_date TEXT",
            "ALTER TABLE shifts ADD COLUMN end_date TEXT",
            "CREATE INDEX IF NOT EXISTS idx_shifts_type ON shifts (type)",
            "CREATE INDEX IF NOT EXISTS idx_shift_variants_shift_id ON shift_variants (shift_id)",
        ]:
            try:
                conn.execute(alter)
            except Exception:
                pass  # column already exists
        conn.commit()
        self._sqlite_conn = conn
        # Dedicated read-only connection — WAL allows concurrent reads without blocking writes
        self._sqlite_read_conn = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
        self._sqlite_read_conn.row_factory = sqlite3.Row
        self._sqlite_read_conn.execute("PRAGMA journal_mode=WAL")
        self._sqlite_read_conn.execute("PRAGMA query_only=ON")
        print(f"[DBWriter] SQLite database: {_SQLITE_PATH}")
        print(f"[DBWriter] SQLite read connection opened")
        self._close_zombie_sessions_sqlite()

    def _close_zombie_sessions_sqlite(self):
        """Mark sessions that were never closed (server crash / kill) as interrupted."""
        try:
            with self._sqlite_lock:
                self._sqlite_conn.execute(
                    "UPDATE sessions SET ended_at = ? WHERE ended_at IS NULL",
                    ("interrupted:" + _ts(),),
                )
                self._sqlite_conn.commit()
        except Exception as e:
            print(f"[DBWriter] zombie-session cleanup error: {e}")

    def close_zombie_sessions_pg(self, conn):
        """Mark sessions that were never closed (server crash / kill) as interrupted."""
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET ended_at = %s WHERE ended_at IS NULL",
                    ("interrupted:" + _ts(),),
                )
            conn.commit()
        except Exception as e:
            print(f"[DBWriter] zombie-session cleanup (PG) error: {e}")

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
        if self._sqlite_read_conn:
            try:
                self._sqlite_read_conn.close()
            except Exception:
                pass
        if self._pg_pool:
            try:
                self._pg_pool.closeall()
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

    def _sqlite_read(self, sql, params=()):
        """Execute a SELECT on the dedicated read connection (never blocks writes)."""
        with self._sqlite_read_lock:
            cur = self._sqlite_read_conn.execute(sql, params)
            return cur.fetchall()

    def open_session(self, checkpoint_id="", camera_source="", group_id="", shift_id=""):
        sid = str(uuid.uuid4())
        with self._lock:
            self._current_session_id = sid
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if conn:
                    try:
                        self.close_zombie_sessions_pg(conn)
                    except Exception:
                        pass
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO sessions (id, group_id, shift_id, started_at, checkpoint_id, camera_source) VALUES (%s, %s, %s, %s, %s, %s)",
                            (sid, group_id, shift_id, _ts(), checkpoint_id, camera_source),
                        )
                    conn.commit()
            except Exception as e:
                print(f"[DBWriter] open_session PG error: {e}")
            finally:
                self._release_pg_conn(conn)
        else:
            with self._sqlite_lock:
                self._sqlite_conn.execute(
                    "INSERT INTO sessions (id, group_id, shift_id, started_at, checkpoint_id, camera_source) VALUES (?, ?, ?, ?, ?, ?)",
                    (sid, group_id, shift_id, _ts(), checkpoint_id, camera_source),
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
            totals.get("nok_anomaly", 0),
            session_id,
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE sessions SET ended_at=%s, total=%s, ok_count=%s, nok_no_barcode=%s, nok_no_date=%s, nok_anomaly=%s WHERE id=%s",
                            params,
                        )
                    conn.commit()
            except Exception as e:
                print(f"[DBWriter] close_session PG error: {e}")
            finally:
                self._release_pg_conn(conn)
        else:
            with self._sqlite_lock:
                self._sqlite_conn.execute(
                    "UPDATE sessions SET ended_at=?, total=?, ok_count=?, nok_no_barcode=?, nok_no_date=?, nok_anomaly=? WHERE id=?",
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
                return {}
            finally:
                self._release_pg_conn(conn)
        rows = self._sqlite_read("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return dict(rows[0]) if rows else {}

    def list_sessions(self, limit=50):
        if not self._available:
            return []
        sql = (
            "SELECT id, group_id, shift_id, started_at, ended_at, checkpoint_id, camera_source, total, ok_count, "
            "nok_no_barcode, nok_no_date, nok_anomaly FROM sessions ORDER BY started_at DESC LIMIT "
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
                return []
            finally:
                self._release_pg_conn(conn)
        rows = self._sqlite_read(sql + "?", (limit,))
        return [dict(r) for r in rows]

    def list_crossings(self, session_id, limit=5000):
        if not self._available or not session_id:
            return []
        sql = (
            "SELECT id, session_id, packet_num, defect_type, crossed_at "
            "FROM defective_packets WHERE session_id = "
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return []
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql + "%s ORDER BY packet_num ASC LIMIT %s", (session_id, limit))
                    return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"[DBWriter] list_crossings error: {e}")
                return []
            finally:
                self._release_pg_conn(conn)
        rows = self._sqlite_read(sql + "? ORDER BY packet_num ASC LIMIT ?", (session_id, limit))
        return [dict(r) for r in rows]

    def list_grouped_sessions(self, limit=50):
        """Return sessions merged by group_id.

        Sessions that share a group_id (started by the same toggleRecording call)
        are collapsed into one logical row.  The merged row sums numeric counters
        across all pipelines and exposes an 'sessions' sub-list with the raw rows.
        Sessions with no group_id (legacy / imported) are returned individually.
        """
        rows = self.list_sessions(limit=limit * 4)  # fetch more to cover all pipelines per group
        from collections import OrderedDict
        groups: OrderedDict = OrderedDict()
        for r in rows:
            gid = r.get("group_id") or r["id"]  # no group → treat as own group
            if gid not in groups:
                groups[gid] = {
                    "id": gid,
                    "shift_id": r.get("shift_id", ""),
                    "started_at": r["started_at"],
                    "ended_at": r.get("ended_at"),
                    "total": 0,
                    "ok_count": 0,
                    "nok_no_barcode": 0,
                    "nok_no_date": 0,
                    "nok_anomaly": 0,
                    "checkpoint_ids": [],
                    "session_ids": [],
                    "sessions": [],
                }
            g = groups[gid]
            g["total"] += r.get("total") or 0
            g["ok_count"] += r.get("ok_count") or 0
            g["nok_no_barcode"] += r.get("nok_no_barcode") or 0
            g["nok_no_date"] += r.get("nok_no_date") or 0
            g["nok_anomaly"] += r.get("nok_anomaly") or 0
            cp = r.get("checkpoint_id", "")
            if cp and cp not in g["checkpoint_ids"]:
                g["checkpoint_ids"].append(cp)
            g["session_ids"].append(r["id"])
            g["sessions"].append(r)
            # earliest start / latest end
            if r["started_at"] and (not g["started_at"] or r["started_at"] < g["started_at"]):
                g["started_at"] = r["started_at"]
            if r.get("ended_at") and (not g["ended_at"] or r["ended_at"] > g["ended_at"]):
                g["ended_at"] = r["ended_at"]
        return list(groups.values())[:limit]

    def list_crossings_for_group(self, group_id, limit=5000):
        """Return crossings across all sessions in a group."""
        if not self._available or not group_id:
            return []
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return []
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT dp.id, dp.session_id, dp.packet_num, dp.defect_type, dp.crossed_at "
                        "FROM defective_packets dp "
                        "JOIN sessions s ON s.id = dp.session_id "
                        "WHERE s.group_id = %s "
                        "ORDER BY dp.crossed_at ASC LIMIT %s",
                        (group_id, limit),
                    )
                    return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"[DBWriter] list_crossings_for_group error: {e}")
                return []
            finally:
                self._release_pg_conn(conn)
        rows = self._sqlite_read(
            "SELECT dp.id, dp.session_id, dp.packet_num, dp.defect_type, dp.crossed_at "
            "FROM defective_packets dp "
            "JOIN sessions s ON s.id = dp.session_id "
            "WHERE s.group_id = ? "
            "ORDER BY dp.crossed_at ASC LIMIT ?",
            (group_id, limit),
        )
        return [dict(r) for r in rows]

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
                if event.get("type") == "session_update":
                    self._update_session_live(event)
                elif event.get("type") == "crossing":
                    self._write_crossing(event)
            except Exception as e:
                print(f"[DBWriter] Event write error: {e}")

            self.write_queue.task_done()

    def _update_session_live(self, ev):
        """UPDATE the sessions row with latest running totals (called every N packets).

        Dashboard sees live numbers; crash leaves last-known state instead of zeroes.
        """
        params = (
            ev.get("total", 0),
            ev.get("ok_count", 0),
            ev.get("nok_no_barcode", 0),
            ev.get("nok_no_date", 0),
            ev.get("nok_anomaly", 0),
            ev.get("session_id"),
        )
        if self._backend == "postgres":
            conn = self._get_pg_conn()
            if not conn:
                return
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE sessions SET total=%s, ok_count=%s, nok_no_barcode=%s, "
                        "nok_no_date=%s, nok_anomaly=%s WHERE id=%s",
                        params,
                    )
                conn.commit()
            finally:
                self._release_pg_conn(conn)
        else:
            with self._sqlite_lock:
                self._sqlite_conn.execute(
                    "UPDATE sessions SET total=?, ok_count=?, nok_no_barcode=?, "
                    "nok_no_date=?, nok_anomaly=? WHERE id=?",
                    params,
                )
                self._sqlite_conn.commit()

    def _write_crossing(self, ev):
        params = (
            ev.get("session_id"),
            ev.get("packet_num", 0),
            ev.get("defect_type", "unknown"),
            ev.get("crossed_at", _ts()),
        )
        sql = (
            "INSERT INTO defective_packets (session_id, packet_num, "
            "defect_type, crossed_at) VALUES "
        )
        if self._backend == "postgres":
            conn = self._get_pg_conn()
            if not conn:
                return
            try:
                with conn.cursor() as cur:
                    cur.execute(sql + "(%s,%s,%s,%s)", params)
                conn.commit()
            finally:
                self._release_pg_conn(conn)
        else:
            with self._sqlite_lock:
                self._sqlite_conn.execute(sql + "(?,?,?,?)", params)
                self._sqlite_conn.commit()

    def _get_pg_conn(self):
        """Get a connection from the pool (postgres) or return None."""
        if not _PSYCOPG2_AVAILABLE:
            return None
        if self._pg_pool is None:
            try:
                self._pg_pool = ThreadedConnectionPool(
                    minconn=1, maxconn=5,
                    host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
                    user=DB_USER, password=DB_PASSWORD,
                    connect_timeout=3,
                )
            except Exception as e:
                print(f"[DBWriter] PG pool creation failed: {e}")
                return None
        try:
            conn = self._pg_pool.getconn()
            conn.autocommit = False
            return conn
        except Exception as e:
            print(f"[DBWriter] PG pool.getconn() failed: {e}")
            return None

    def _release_pg_conn(self, conn):
        """Return a connection to the pool."""
        if self._pg_pool and conn:
            try:
                self._pg_pool.putconn(conn)
            except Exception:
                pass

    # ═══════════════════════════════════════════════
    # SHIFTS CRUD  (direct calls, not queue-based)
    # ═══════════════════════════════════════════════

    def _attach_variants(self, shifts):
        """Attach a 'variants' list to each shift dict in-place."""
        all_variants = self.get_all_variants()
        by_shift = {}
        for v in all_variants:
            by_shift.setdefault(v["shift_id"], []).append(v)
        for s in shifts:
            s["variants"] = by_shift.get(s["id"], [])
        return shifts

    def get_all_variants(self):
        """Return all shift_variants rows."""
        if not self._available:
            return []
        sql = ("SELECT id, shift_id, kind, active, start_time, end_time, "
               "start_date, end_date, days_of_week, created_at "
               "FROM shift_variants ORDER BY created_at")
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return []
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"[DBWriter] get_all_variants error: {e}")
                return []
            finally:
                self._release_pg_conn(conn)
        rows = self._sqlite_read(sql)
        return [dict(r) for r in rows]

    def get_variants_for_shift(self, shift_id):
        """Return all shift_variants rows for a specific shift."""
        if not self._available or not shift_id:
            return []
        sql = ("SELECT id, shift_id, kind, active, start_time, end_time, "
               "start_date, end_date, days_of_week, created_at "
               "FROM shift_variants WHERE shift_id = ")
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return []
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql + "%s ORDER BY created_at", (shift_id,))
                    return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"[DBWriter] get_variants_for_shift error: {e}")
                return []
            finally:
                self._release_pg_conn(conn)
        rows = self._sqlite_read(sql + "? ORDER BY created_at", (shift_id,))
        return [dict(r) for r in rows]

    def get_all_shifts(self):
        """Return all recurring shifts ordered by start_time, each with a 'variants' list."""
        if not self._available:
            return []
        sql = (
            "SELECT id, label, type, start_time, end_time, start_date, end_date, session_date, days_of_week, "
            "camera_source, checkpoint_id, active, created_at "
            "FROM shifts WHERE type = 'recurring' OR type IS NULL ORDER BY start_time"
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return []
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    shifts = [dict(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"[DBWriter] get_all_shifts error: {e}")
                return []
            finally:
                self._release_pg_conn(conn)
        else:
            rows = self._sqlite_read(sql)
            shifts = [dict(r) for r in rows]
        return self._attach_variants(shifts)

    def get_shift(self, shift_id):
        """Return a single shift by id, or None."""
        if not self._available or not shift_id:
            return None
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return None
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM shifts WHERE id = %s", (shift_id,))
                    row = cur.fetchone()
                    return dict(row) if row else None
            except Exception as e:
                print(f"[DBWriter] get_shift error: {e}")
                return None
            finally:
                self._release_pg_conn(conn)
        rows = self._sqlite_read("SELECT * FROM shifts WHERE id = ?", (shift_id,))
        return dict(rows[0]) if rows else None

    def insert_shift(self, shift):
        """Insert a new recurring shift row."""
        if not self._available:
            return False
        params = (
            shift["id"], shift["label"], "recurring", shift["start_time"],
            shift["end_time"], shift.get("start_date"), shift.get("end_date"),
            None, shift["days_of_week"],
            shift.get("camera_source", "0"),
            shift.get("checkpoint_id", "tracking"),
            shift.get("active", 1),
            shift["created_at"],
        )
        sql = (
            "INSERT INTO shifts (id, label, type, start_time, end_time, start_date, end_date, session_date, days_of_week, "
            "camera_source, checkpoint_id, active, created_at) VALUES "
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute(sql + "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", params)
                conn.commit()
                return True
            except Exception as e:
                print(f"[DBWriter] insert_shift error: {e}")
                return False
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute(sql + "(?,?,?,?,?,?,?,?,?,?,?,?,?)", params)
            self._sqlite_conn.commit()
        return True

    def update_shift(self, shift_id, fields):
        """Update specific fields of a shift. `fields` is a dict of column->value."""
        if not self._available or not shift_id or not fields:
            return False
        allowed = {"label", "start_time", "end_time", "start_date", "end_date",
                   "days_of_week", "camera_source", "checkpoint_id", "active"}
        cols = {k: v for k, v in fields.items() if k in allowed}
        if not cols:
            return False
        ph = "%s" if self._backend == "postgres" else "?"
        set_clause = ", ".join(f"{k} = {ph}" for k in cols)
        values = list(cols.values()) + [shift_id]
        sql = f"UPDATE shifts SET {set_clause} WHERE id = {ph}"
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute(sql, values)
                conn.commit()
                return True
            except Exception as e:
                print(f"[DBWriter] update_shift error: {e}")
                return False
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute(sql, values)
            self._sqlite_conn.commit()
        return True

    def delete_shift(self, shift_id):
        """Delete a shift by id."""
        if not self._available or not shift_id:
            return False
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM shifts WHERE id = %s", (shift_id,))
                conn.commit()
                return True
            except Exception as e:
                print(f"[DBWriter] delete_shift error: {e}")
                return False
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute("DELETE FROM shifts WHERE id = ?", (shift_id,))
            self._sqlite_conn.commit()
        return True

    def toggle_shift(self, shift_id):
        """Flip active 0<->1. Returns new active value or None on failure."""
        if not self._available or not shift_id:
            return None
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE shifts SET active = CASE WHEN active = 1 THEN 0 ELSE 1 END "
                        "WHERE id = %s RETURNING active", (shift_id,)
                    )
                    row = cur.fetchone()
                conn.commit()
                return row[0] if row else None
            except Exception as e:
                print(f"[DBWriter] toggle_shift error: {e}")
                return None
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute(
                "UPDATE shifts SET active = CASE WHEN active = 1 THEN 0 ELSE 1 END WHERE id = ?",
                (shift_id,),
            )
            self._sqlite_conn.commit()
            cur = self._sqlite_conn.execute("SELECT active FROM shifts WHERE id = ?", (shift_id,))
            row = cur.fetchone()
            return row["active"] if row else None

    # ═══════════════════════════════════════════════
    # SHIFT VARIANTS CRUD
    # ═══════════════════════════════════════════════

    def insert_variant(self, variant):
        """Insert a shift_variant row. Returns the inserted dict or None."""
        if not self._available:
            return None
        params = (
            variant["id"], variant["shift_id"], variant["kind"],
            variant.get("active"), variant.get("start_time"), variant.get("end_time"),
            variant["start_date"], variant["end_date"],
            variant["days_of_week"], variant["created_at"],
        )
        sql = (
            "INSERT INTO shift_variants (id, shift_id, kind, active, start_time, end_time, "
            "start_date, end_date, days_of_week, created_at) VALUES "
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cur.execute(sql + "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", params)
                conn.commit()
                return variant
            except Exception as e:
                print(f"[DBWriter] insert_variant error: {e}")
                return None
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute(sql + "(?,?,?,?,?,?,?,?,?,?)", params)
            self._sqlite_conn.commit()
        return variant

    def update_variant(self, variant_id, fields):
        """Update specific fields of a shift_variant."""
        if not self._available or not variant_id or not fields:
            return False
        allowed = {"kind", "active", "start_time", "end_time", "start_date", "end_date", "days_of_week"}
        cols = {k: v for k, v in fields.items() if k in allowed}
        if not cols:
            return False
        ph = "%s" if self._backend == "postgres" else "?"
        set_clause = ", ".join(f"{k} = {ph}" for k in cols)
        values = list(cols.values()) + [variant_id]
        sql = f"UPDATE shift_variants SET {set_clause} WHERE id = {ph}"
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute(sql, values)
                conn.commit()
                return True
            except Exception as e:
                print(f"[DBWriter] update_variant error: {e}")
                return False
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute(sql, values)
            self._sqlite_conn.commit()
        return True

    def delete_variant(self, variant_id):
        """Delete a shift_variant by id."""
        if not self._available or not variant_id:
            return False
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM shift_variants WHERE id = %s", (variant_id,))
                conn.commit()
                return True
            except Exception as e:
                print(f"[DBWriter] delete_variant error: {e}")
                return False
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute("DELETE FROM shift_variants WHERE id = ?", (variant_id,))
            self._sqlite_conn.commit()
        return True

    def delete_variants_for_shift(self, shift_id):
        """Delete all shift_variants for a given shift (used on shift delete)."""
        if not self._available or not shift_id:
            return False
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM shift_variants WHERE shift_id = %s", (shift_id,))
                conn.commit()
                return True
            except Exception as e:
                print(f"[DBWriter] delete_variants_for_shift error: {e}")
                return False
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute("DELETE FROM shift_variants WHERE shift_id = ?", (shift_id,))
            self._sqlite_conn.commit()
        return True

    # ── One-off sessions (stored in unified shifts table, type='one_off') ────

    def get_all_one_off_sessions(self):
        """Return all one-off sessions from the unified shifts table."""
        if not self._available:
            return []
        sql = (
            "SELECT id, label, session_date AS date, start_time, end_time, "
            "camera_source, checkpoint_id, created_at "
            "FROM shifts WHERE type = 'one_off' ORDER BY session_date, start_time"
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return []
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    return [dict(r) for r in cur.fetchall()]
            except Exception as e:
                print(f"[DBWriter] get_all_one_off_sessions error: {e}")
                return []
            finally:
                self._release_pg_conn(conn)
        rows = self._sqlite_read(sql)
        return [dict(r) for r in rows]

    def insert_one_off_session(self, session):
        """Insert a one-off session into the unified shifts table."""
        if not self._available:
            return None
        params = (
            session["id"], session["label"], "one_off",
            session["start_time"], session["end_time"],
            session["date"],  # → session_date
            "[]",             # days_of_week unused for one_off
            session.get("camera_source", "0"),
            session.get("checkpoint_id", "tracking"),
            1,
            session["created_at"],
        )
        sql = (
            "INSERT INTO shifts "
            "(id, label, type, start_time, end_time, session_date, days_of_week, "
            "camera_source, checkpoint_id, active, created_at) VALUES "
        )
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cur.execute(sql + "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", params)
                conn.commit()
                return session
            except Exception as e:
                print(f"[DBWriter] insert_one_off_session error: {e}")
                return None
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute(sql + "(?,?,?,?,?,?,?,?,?,?,?)", params)
            self._sqlite_conn.commit()
        return session

    def update_one_off_session(self, session_id, fields):
        """Update start_time and/or end_time of a one-off session."""
        if not self._available or not session_id or not fields:
            return False
        allowed = {"start_time", "end_time"}
        cols = {k: v for k, v in fields.items() if k in allowed}
        if not cols:
            return False
        ph = "%s" if self._backend == "postgres" else "?"
        set_clause = ", ".join(f"{k} = {ph}" for k in cols)
        values = list(cols.values()) + [session_id]
        sql = f"UPDATE shifts SET {set_clause} WHERE id = {ph} AND type = 'one_off'"
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute(sql, values)
                conn.commit()
                return True
            except Exception as e:
                print(f"[DBWriter] update_one_off_session error: {e}")
                return False
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute(sql, values)
            self._sqlite_conn.commit()
        return True

    def delete_one_off_session(self, session_id):
        """Delete a one-off session from the unified shifts table."""
        if not self._available or not session_id:
            return False
        if self._backend == "postgres":
            try:
                conn = self._get_pg_conn()
                if not conn:
                    return False
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM shifts WHERE id = %s AND type = 'one_off'", (session_id,))
                conn.commit()
                return True
            except Exception as e:
                print(f"[DBWriter] delete_one_off_session error: {e}")
                return False
            finally:
                self._release_pg_conn(conn)
        with self._sqlite_lock:
            self._sqlite_conn.execute("DELETE FROM shifts WHERE id = ? AND type = 'one_off'", (session_id,))
            self._sqlite_conn.commit()
        return True
