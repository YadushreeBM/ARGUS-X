"""SQLite persistence for sessions, camera identities and recovery events.

Connections use a bounded busy timeout and are always closed. WAL mode is
configured during initialization only, never on every application connection.
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent.parent / "argus_x.db"
BUSY_TIMEOUT_MS = 15000


def connect(db_path=DB_PATH):
    """Create a configured connection. Caller is responsible for closing it."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_connection(db_path=DB_PATH):
    """Connection context that commits on success and always closes."""
    conn = connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize(db_path=DB_PATH):
    """Create schema using SQLite's normal journal mode and a busy timeout.

    We intentionally do not switch journal mode here: Streamlit can rerun this
    function while the live worker is writing, and changing journal mode needs
    an exclusive lock. The busy timeout handles ordinary short write contention.
    """
    with db_connection(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(session_id TEXT PRIMARY KEY,source_name TEXT,created_at TEXT,frames INTEGER,fps REAL,summary_json TEXT);
        CREATE TABLE IF NOT EXISTS identities(session_id TEXT,identity_id INTEGER,class_name TEXT,state TEXT,first_frame INTEGER,last_frame INTEGER,hits INTEGER,recovery_count INTEGER,PRIMARY KEY(session_id,identity_id));
        CREATE TABLE IF NOT EXISTS recovery_events(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,frame INTEGER,identity_id INTEGER,event TEXT,confidence REAL,detail TEXT);
        CREATE TABLE IF NOT EXISTS camera_sessions(session_id TEXT,camera_id TEXT,video_path TEXT,frames INTEGER,summary_json TEXT,PRIMARY KEY(session_id,camera_id));
        CREATE TABLE IF NOT EXISTS global_identity_events(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,timestamp REAL,camera_id TEXT,local_track_id INTEGER,global_id TEXT,event TEXT,confidence REAL);
        """)


def save_session(session_id, source_name, frames, fps, tracker, db_path=DB_PATH):
    initialize(db_path)
    summary = tracker.summary()
    with db_connection(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO sessions VALUES(?,?,?,?,?,?)", (session_id, source_name, datetime.now(timezone.utc).isoformat(), frames, fps, json.dumps(summary)))
        for t in tracker.tracks.values():
            conn.execute("INSERT OR REPLACE INTO identities VALUES(?,?,?,?,?,?,?,?)", (session_id, t.identity_id, t.class_name, t.state, t.first_frame, t.last_frame, t.hits, t.recovery_count))
        conn.execute("DELETE FROM recovery_events WHERE session_id=?", (session_id,))
        conn.executemany("INSERT INTO recovery_events(session_id,frame,identity_id,event,confidence,detail) VALUES(?,?,?,?,?,?)", [(session_id, e["frame"], e["identity_id"], e["event"], e["confidence"], e["detail"]) for e in tracker.events])
    return summary


def save_multi_session(result, source_name="multi-camera session", db_path=DB_PATH):
    initialize(db_path)
    with db_connection(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO sessions VALUES(?,?,?,?,?,?)", (result["session_id"], source_name, datetime.now(timezone.utc).isoformat(), sum(c["frames"] for c in result["cameras"]), result["fps"], json.dumps({"global_identities": result["global_identities"], "camera_count": len(result["cameras"])})))
        for c in result["cameras"]:
            conn.execute("INSERT OR REPLACE INTO camera_sessions VALUES(?,?,?,?,?)", (result["session_id"], c["camera_id"], c["video"], c["frames"], json.dumps(c["summary"])))
        conn.executemany("INSERT INTO global_identity_events(session_id,timestamp,camera_id,local_track_id,global_id,event,confidence) VALUES(?,?,?,?,?,?,?)", [(result["session_id"], e["timestamp"], e["camera_id"], e["local_track_id"], e["global_id"], e["event"], e["confidence"]) for e in result["global_events"]])
