"""Historique local des sessions (SQLite). Aucune image n'est jamais enregistrée."""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing

from .config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started REAL NOT NULL,
    ended REAL,
    profile TEXT NOT NULL,
    duration_s REAL DEFAULT 0,
    present_s REAL DEFAULT 0,
    distracted_s REAL DEFAULT 0,
    avg_vigilance REAL DEFAULT 100,
    min_vigilance REAL DEFAULT 100,
    blinks INTEGER DEFAULT 0,
    yawns INTEGER DEFAULT 0,
    alarms INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    kind TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


def _connect(path=None) -> sqlite3.Connection:
    path = path or DB_PATH
    if path != ":memory:":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


class SessionRecorder:
    """Enregistre une session ; toutes les erreurs disque sont silencieuses."""

    def __init__(self, profile: str, path=None):
        self.path = path
        self.session_id: int | None = None
        try:
            with closing(_connect(path)) as conn, conn:
                cur = conn.execute("INSERT INTO sessions (started, profile) VALUES (?, ?)",
                                   (time.time(), profile))
                self.session_id = cur.lastrowid
        except sqlite3.Error:
            pass

    def add_events(self, alerts) -> None:
        if self.session_id is None or not alerts:
            return
        now = time.time()
        try:
            with closing(_connect(self.path)) as conn, conn:
                conn.executemany(
                    "INSERT INTO events (session_id, t, kind, level, message) VALUES (?,?,?,?,?)",
                    [(self.session_id, now, a.kind, a.level, a.message) for a in alerts],
                )
        except sqlite3.Error:
            pass

    def close(self, analyzer, duration_s: float) -> None:
        if self.session_id is None:
            return
        try:
            with closing(_connect(self.path)) as conn, conn:
                if duration_s < 10:  # session trop courte pour être utile
                    conn.execute("DELETE FROM events WHERE session_id=?", (self.session_id,))
                    conn.execute("DELETE FROM sessions WHERE id=?", (self.session_id,))
                else:
                    conn.execute(
                        """UPDATE sessions SET ended=?, duration_s=?, present_s=?, distracted_s=?,
                           avg_vigilance=?, min_vigilance=?, blinks=?, yawns=?, alarms=?
                           WHERE id=?""",
                        (time.time(), duration_s, analyzer.present_s, analyzer.distracted_s,
                         analyzer.mean_vigilance, analyzer.min_vigilance, analyzer.total_blinks,
                         analyzer.total_yawns, analyzer.total_alarms, self.session_id),
                    )
        except sqlite3.Error:
            pass
        self.session_id = None


def recent_sessions(limit: int = 20, path=None) -> list[dict]:
    try:
        with closing(_connect(path)) as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE ended IS NOT NULL ORDER BY started DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error:
        return []
