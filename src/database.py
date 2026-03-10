"""
database.py - SQLite database layer for storing ticket predictions
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/tickets.db")


def get_db_path() -> str:
    return os.getenv("DB_PATH", DB_PATH)


@contextmanager
def get_connection(db_path: str = None):
    """Context manager for database connections."""
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent write performance
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_db(db_path: str = None):
    """Initialize the database schema."""
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tickets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_text TEXT    NOT NULL,
                department  TEXT    NOT NULL,
                priority    TEXT    NOT NULL,
                confidence  REAL,
                latency_ms  REAL,
                source      TEXT    DEFAULT 'api',
                timestamp   TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_department ON tickets(department);
            CREATE INDEX IF NOT EXISTS idx_priority   ON tickets(priority);
            CREATE INDEX IF NOT EXISTS idx_timestamp  ON tickets(timestamp);

            CREATE TABLE IF NOT EXISTS request_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint    TEXT    NOT NULL,
                status_code INTEGER,
                latency_ms  REAL,
                timestamp   TEXT    NOT NULL
            );
        """)
    logger.info(f"Database initialized at: {db_path or get_db_path()}")


def insert_ticket(
    ticket_text: str,
    department: str,
    priority: str,
    confidence: float = None,
    latency_ms: float = None,
    source: str = "api",
    db_path: str = None,
) -> int:
    """Insert a classified ticket and return its ID."""
    timestamp = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO tickets
               (ticket_text, department, priority, confidence, latency_ms, source, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ticket_text, department, priority, confidence, latency_ms, source, timestamp),
        )
        ticket_id = cursor.lastrowid
    logger.info(f"Inserted ticket id={ticket_id} -> {department} [{priority}]")
    return ticket_id


def log_request(endpoint: str, status_code: int, latency_ms: float, db_path: str = None):
    """Log an API request for monitoring."""
    timestamp = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO request_log (endpoint, status_code, latency_ms, timestamp) VALUES (?, ?, ?, ?)",
            (endpoint, status_code, latency_ms, timestamp),
        )


def get_tickets(
    limit: int = 50,
    department: str = None,
    priority: str = None,
    db_path: str = None,
) -> List[Dict[str, Any]]:
    """Retrieve tickets with optional filtering."""
    query = "SELECT * FROM tickets WHERE 1=1"
    params = []

    if department:
        query += " AND department = ?"
        params.append(department)
    if priority:
        query += " AND priority = ?"
        params.append(priority)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_stats(db_path: str = None) -> Dict[str, Any]:
    """Return aggregate statistics from the database."""
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        by_dept = conn.execute(
            "SELECT department, COUNT(*) as count FROM tickets GROUP BY department ORDER BY count DESC"
        ).fetchall()
        by_priority = conn.execute(
            "SELECT priority, COUNT(*) as count FROM tickets GROUP BY priority"
        ).fetchall()
        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) FROM tickets WHERE latency_ms IS NOT NULL"
        ).fetchone()[0]

    return {
        "total_tickets": total,
        "by_department": [dict(r) for r in by_dept],
        "by_priority": [dict(r) for r in by_priority],
        "avg_latency_ms": round(avg_latency, 2) if avg_latency else None,
    }
