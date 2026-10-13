"""Bounded JSONL mission records. Only explicit event fields reach the log."""
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys


class _Handler(RotatingFileHandler):
    def handleError(self, record):
        raise OSError("Event log write failed.")


class EventJournal:
    def __init__(self, directory, max_bytes=1024 * 1024, backups=3):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.handler = _Handler(directory / "events.jsonl", maxBytes=max_bytes,
                                backupCount=backups, encoding="utf-8")
        self.failed = False

    def __call__(self, event, **fields):
        if self.failed:
            return
        packet = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        try:
            line = json.dumps(packet, separators=(",", ":"), allow_nan=False)
            self.handler.emit(logging.makeLogRecord({"msg": line, "args": ()}))
        except (ValueError, OSError):
            self.failed = True
            print("Waypoint event log unavailable; check storage before further trials.", file=sys.stderr)

    def close(self):
        self.handler.close()
