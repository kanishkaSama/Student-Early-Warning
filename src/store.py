"""SQLite storage for predictions and generated messages.

Tables
------
predictions(code_module, code_presentation, id_student, cutoff_day, risk_score,
            at_risk, reasons_json, final_result, explain_method, model_name, created_at)
            primary key: (code_module, code_presentation, id_student, cutoff_day)
messages(id, code_module, code_presentation, id_student, cutoff_day, message,
         source, note, created_at)

Only pseudonymous student IDs are stored - no names or contact details.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    code_module TEXT NOT NULL,
    code_presentation TEXT NOT NULL,
    id_student INTEGER NOT NULL,
    cutoff_day INTEGER NOT NULL,
    risk_score REAL NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    at_risk INTEGER NOT NULL CHECK (at_risk IN (0, 1)),
    reasons_json TEXT NOT NULL,
    final_result TEXT,
    explain_method TEXT,
    model_name TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (code_module, code_presentation, id_student, cutoff_day)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code_module TEXT NOT NULL,
    code_presentation TEXT NOT NULL,
    id_student INTEGER NOT NULL,
    cutoff_day INTEGER NOT NULL,
    message TEXT NOT NULL,
    source TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_student
    ON messages (code_module, code_presentation, id_student, cutoff_day);
"""

PREDICTION_COLUMNS = ["code_module", "code_presentation", "id_student", "cutoff_day",
                      "risk_score", "at_risk", "reasons_json", "final_result",
                      "explain_method", "model_name"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA_SQL)
    return connection


def reset(db_path: str | Path) -> None:
    """Remove old predictions and messages (they belong to an older model run)."""
    with closing(connect(db_path)) as connection, connection:
        connection.execute("DELETE FROM predictions")
        connection.execute("DELETE FROM messages")


def save_predictions(db_path: str | Path, predictions: pd.DataFrame) -> int:
    rows = predictions[PREDICTION_COLUMNS].copy()
    rows["reasons_json"] = rows["reasons_json"].map(
        lambda v: v if isinstance(v, str) else json.dumps(v))
    rows["created_at"] = _now()
    records = [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
                     for v in rec) for rec in rows.itertuples(index=False)]
    placeholders = ", ".join("?" * len(rows.columns))
    with closing(connect(db_path)) as connection, connection:
        connection.executemany(
            f"INSERT OR REPLACE INTO predictions ({', '.join(rows.columns)}) VALUES ({placeholders})",
            records)
    return len(records)


def load_predictions(db_path: str | Path) -> pd.DataFrame:
    with closing(connect(db_path)) as connection:
        frame = pd.read_sql_query("SELECT * FROM predictions", connection)
    frame["reasons"] = frame["reasons_json"].map(json.loads)
    return frame


def save_message(db_path: str | Path, key: dict, text: str, source: str, note: str = "") -> None:
    with closing(connect(db_path)) as connection, connection:
        connection.execute(
            "INSERT INTO messages (code_module, code_presentation, id_student, cutoff_day, "
            "message, source, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (key["code_module"], key["code_presentation"], int(key["id_student"]),
             int(key["cutoff_day"]), text, source, note, _now()))


def latest_message(db_path: str | Path, key: dict) -> dict | None:
    with closing(connect(db_path)) as connection:
        row = connection.execute(
            "SELECT message, source, note, created_at FROM messages WHERE code_module = ? "
            "AND code_presentation = ? AND id_student = ? AND cutoff_day = ? "
            "ORDER BY id DESC LIMIT 1",
            (key["code_module"], key["code_presentation"], int(key["id_student"]),
             int(key["cutoff_day"]))).fetchone()
    if row is None:
        return None
    return dict(zip(["message", "source", "note", "created_at"], row))
