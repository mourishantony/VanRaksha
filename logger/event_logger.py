"""SQLite event logger for VanRaksha AI."""
from __future__ import annotations

import csv
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path


class EventLogger:
    """Persist and query detection events in SQLite."""

    def __init__(self, db_path: str):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_zone TEXT,
                    species TEXT,
                    confidence REAL,
                    threat_score REAL,
                    alert_level TEXT,
                    risk_label TEXT,
                    bbox_x1 INTEGER, bbox_y1 INTEGER,
                    bbox_x2 INTEGER, bbox_y2 INTEGER,
                    sms_sent INTEGER DEFAULT 0,
                    voice_played INTEGER DEFAULT 0
                )
                """
            )
            # Add risk_label column to existing databases (migration)
            try:
                self.conn.execute("ALTER TABLE events ADD COLUMN risk_label TEXT")
            except Exception:
                pass  # Column already exists
            self.conn.commit()

    def log_event(self, event: dict) -> int:
        """Insert event and return inserted row id."""
        payload = {
            "timestamp": event.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "camera_zone": event.get("camera_zone"),
            "species": event.get("species", "unknown"),
            "confidence": event.get("confidence", 0.0),
            "threat_score": event.get("threat_score", 0.0),
            "alert_level": event.get("alert_level", "SAFE"),
            "risk_label": event.get("risk_label", ""),
            "bbox_x1": event.get("bbox", [0, 0, 0, 0])[0],
            "bbox_y1": event.get("bbox", [0, 0, 0, 0])[1],
            "bbox_x2": event.get("bbox", [0, 0, 0, 0])[2],
            "bbox_y2": event.get("bbox", [0, 0, 0, 0])[3],
            "sms_sent": int(bool(event.get("sms_sent", False))),
            "voice_played": int(bool(event.get("voice_played", False))),
        }
        with self.lock:
            cur = self.conn.execute(
                """
                INSERT INTO events (
                    timestamp, camera_zone, species, confidence, threat_score, alert_level, risk_label,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, sms_sent, voice_played
                ) VALUES (
                    :timestamp, :camera_zone, :species, :confidence, :threat_score, :alert_level, :risk_label,
                    :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2, :sms_sent, :voice_played
                )
                """,
                payload,
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Return latest events ordered newest-first."""
        with self.lock:
            rows = self.conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """Return aggregate event statistics."""
        with self.lock:
            total_events = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            by_species_rows = self.conn.execute("SELECT species, COUNT(*) as c FROM events GROUP BY species").fetchall()
            by_alert_rows = self.conn.execute("SELECT alert_level, COUNT(*) as c FROM events GROUP BY alert_level").fetchall()
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            last_24h = self.conn.execute("SELECT COUNT(*) FROM events WHERE timestamp >= ?", (cutoff,)).fetchone()[0]

        return {
            "total_events": total_events,
            "by_species": {r["species"]: r["c"] for r in by_species_rows},
            "by_alert_level": {r["alert_level"]: r["c"] for r in by_alert_rows},
            "last_24h": last_24h,
        }

    def export_csv(self, filepath: str, days: int = 30):
        """Export recent events to CSV file."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE timestamp >= ? ORDER BY id DESC", (cutoff,)
            ).fetchall()

        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
                "id", "timestamp", "camera_zone", "species", "confidence", "threat_score", "alert_level",
                "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "sms_sent", "voice_played"
            ])
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = EventLogger("logger/vanraksha.db")
    row_id = logger.log_event({"species": "tiger", "threat_score": 90, "alert_level": "CRITICAL"})
    logging.getLogger(__name__).info("Event logger self-test: %s %s", row_id, logger.get_stats())
