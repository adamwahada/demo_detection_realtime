import queue
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
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    STATS_ENABLED_DEFAULT, WRITE_QUEUE_MAXSIZE,
)


def _ts() -> str:
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


class DBWriter:
    """
    Minimalistic background writer — PostgreSQL only.
    Handles only 'sessions' and 'images' tables.
    """

    def __init__(self):
        self.write_queue  = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)
        self._thread      = None
        self._stop_event  = threading.Event()
        self._lock        = threading.Lock()
        self._active      = STATS_ENABLED_DEFAULT
        self._current_session_id = None
        self._available   = _PSYCOPG2_AVAILABLE
        self._pg_conn     = None

    # ──────────────────────────────
    # Public control
    # ──────────────────────────────

    def start(self):
        if not self._available:
            print("[DBWriter] PostgreSQL not available — writer disabled.")
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

    # ──────────────────────────────
    # Session management
    # ──────────────────────────────

    def open_session(self, checkpoint_id: str = "", camera_source: str = "") -> str:
        sid = str(uuid.uuid4())
        with self._lock:
            self._current_session_id = sid
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
        print(f"[DBWriter] Session opened: {sid[:8]}...")
        return sid

    def close_session(self, session_id: str, totals: dict = None):
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
        sql = ("UPDATE sessions SET ended_at=%s, total=%s, ok_count=%s, "
               "nok_no_barcode=%s, nok_no_date=%s, nok_both=%s WHERE id=%s")
        try:
            conn = self._get_pg_conn()
            if conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                conn.commit()
        except Exception as e:
            print(f"[DBWriter] close_session PG error: {e}")
            self._pg_conn = None
        print(f"[DBWriter] Session closed: {session_id[:8]}...")

    # ──────────────────────────────
    # Image handling
    # ──────────────────────────────

    def record_image(self, session_id: str, packet_number: int, defect_type: str, image_path: str):
        if not self._available or not session_id:
            return
        try:
            conn = self._get_pg_conn()
            if conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO images (session_id, packet_number, defect_type, image_path) "
                        "VALUES (%s, %s, %s, %s)",
                        (session_id, packet_number, defect_type, image_path)
                    )
                conn.commit()
        except Exception as e:
            print(f"[DBWriter] record_image error: {e}")
            self._pg_conn = None

    def get_session_images(self, session_id: str) -> list:
        if not self._available or not session_id:
            return []
        try:
            conn = self._get_pg_conn()
            if not conn:
                return []
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, packet_number, defect_type, image_path, captured_at "
                    "FROM images WHERE session_id = %s ORDER BY captured_at",
                    (session_id,)
                )
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[DBWriter] get_session_images error: {e}")
            self._pg_conn = None
            return []

    # ──────────────────────────────
    # Session KPI / daily stats
    # ──────────────────────────────

    def get_session_kpis(self, session_id: str) -> dict:
        if not self._available or not session_id:
            return {}
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

    def list_sessions(self, limit: int = 50) -> list:
        if not self._available:
            return []
        sql = ("SELECT id, started_at, ended_at, checkpoint_id, camera_source, "
               "total, ok_count, nok_no_barcode, nok_no_date, nok_both "
               "FROM sessions ORDER BY started_at DESC LIMIT %s")
        try:
            conn = self._get_pg_conn()
            if not conn:
                return []
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (limit,))
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[DBWriter] list_sessions error: {e}")
            self._pg_conn = None
            return []

    def export_csv(self, session_id: str = None) -> str:
        header = "id,started_at,ended_at,total,ok_count,nok_no_barcode,nok_no_date,nok_both,nok_rate"
        sessions = [self.get_session_kpis(session_id)] if session_id else self.list_sessions(limit=1000)
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
                f",{total},{ok},{s.get('nok_no_barcode',0)},{s.get('nok_no_date',0)},{s.get('nok_both',0)},{rate}"
            )
        return "\n".join(lines)

    # ──────────────────────────────
    # Background thread
    # ──────────────────────────────

    def _run(self):
        print("[DBWriter] Writer loop started.")
        while not self._stop_event.is_set():
            try:
                event = self.write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if event.get("type") == "stop":
                break

            self.write_queue.task_done()
        print("[DBWriter] Writer loop exited.")

    # ──────────────────────────────
    # PostgreSQL connection
    # ──────────────────────────────

    def _get_pg_conn(self):
        if not _PSYCOPG2_AVAILABLE:
            return None
        if self._pg_conn and getattr(self._pg_conn, "closed", 1) == 0:
            return self._pg_conn
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