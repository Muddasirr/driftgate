"""On-disk response cache.

Re-running an unchanged suite should cost nothing and return identical results,
so responses are stored in SQLite keyed by a hash of everything that affects the
output (endpoint, model, messages, tools, sampling params, repeat index).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from driftgate.results import Completion

DEFAULT_CACHE_PATH = Path(".driftgate") / "cache.sqlite"


class ResponseCache:
    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY,"
            " completion TEXT NOT NULL,"
            " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self._db.commit()

    @staticmethod
    def key(parts: dict[str, Any]) -> str:
        canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Completion | None:
        row = self._db.execute("SELECT completion FROM responses WHERE key = ?", (key,)).fetchone()
        return Completion.model_validate_json(row[0]) if row else None

    def put(self, key: str, completion: Completion) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO responses (key, completion) VALUES (?, ?)",
            (key, completion.model_dump_json()),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
