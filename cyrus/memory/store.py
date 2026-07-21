"""Local SQLite FTS5 memory with provenance and hard-deletion controls."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class MemoryError(RuntimeError):
    """Raised when memory provenance or storage invariants fail."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists() and self.path.is_symlink():
            raise MemoryError("memory database may not be a symbolic link")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                license TEXT NOT NULL,
                language TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'semantic',
                indexed_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                record_id UNINDEXED,
                text,
                source UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT NOT NULL,
                record_id TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def index_approved(self, data_root: str | Path) -> dict[str, Any]:
        root = Path(data_root)
        manifest_root = root / "manifests"
        indexed = 0
        unchanged = 0
        if not manifest_root.exists():
            raise MemoryError(f"manifest directory not found: {manifest_root}")
        with self.connection:
            for manifest_path in sorted(manifest_root.glob("*.json")):
                if manifest_path.is_symlink():
                    raise MemoryError(f"manifest may not be a symbolic link: {manifest_path}")
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise MemoryError(f"invalid manifest: {manifest_path}") from exc
                if manifest.get("state") != "approved":
                    continue
                record_id = manifest.get("record_id")
                if not isinstance(record_id, str) or not re.fullmatch(r"[a-f0-9]{64}", record_id):
                    raise MemoryError(f"invalid record id in {manifest_path}")
                text_path = root / "approved" / f"{record_id}.txt"
                if not text_path.is_file() or text_path.is_symlink():
                    raise MemoryError(f"approved text missing or unsafe: {record_id}")
                payload = text_path.read_bytes()
                if _sha256(payload) != manifest.get("normalized_sha256"):
                    raise MemoryError(f"approved text hash mismatch: {record_id}")
                text = payload.decode("utf-8")
                existing = self.connection.execute(
                    "SELECT content_sha256 FROM records WHERE record_id = ?", (record_id,)
                ).fetchone()
                if existing is not None and existing["content_sha256"] == manifest["normalized_sha256"]:
                    unchanged += 1
                    continue
                self.connection.execute("DELETE FROM memory_fts WHERE record_id = ?", (record_id,))
                self.connection.execute(
                    """
                    INSERT INTO records (
                        record_id, text, source, content_sha256, license, language, kind, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'semantic', ?)
                    ON CONFLICT(record_id) DO UPDATE SET
                        text = excluded.text,
                        source = excluded.source,
                        content_sha256 = excluded.content_sha256,
                        license = excluded.license,
                        language = excluded.language,
                        indexed_at = excluded.indexed_at
                    """,
                    (
                        record_id,
                        text,
                        manifest["source"],
                        manifest["normalized_sha256"],
                        manifest["license"],
                        manifest.get("language", "und"),
                        _utc_now(),
                    ),
                )
                self.connection.execute(
                    "INSERT INTO memory_fts (record_id, text, source) VALUES (?, ?, ?)",
                    (record_id, text, manifest["source"]),
                )
                self.connection.execute(
                    "INSERT INTO audit_events (event, record_id, created_at) VALUES ('indexed', ?, ?)",
                    (record_id, _utc_now()),
                )
                indexed += 1
        return {"indexed": indexed, "unchanged": unchanged, "total": self.count()}

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)[:16]
        if not terms:
            raise MemoryError("search query contains no searchable words")
        return " OR ".join(f'"{term}"' for term in terms)

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if not 1 <= limit <= 50:
            raise MemoryError("search limit must be between 1 and 50")
        fts_query = self._fts_query(query)
        rows = self.connection.execute(
            """
            SELECT
                f.record_id,
                r.source,
                r.content_sha256,
                r.license,
                r.language,
                snippet(memory_fts, 1, '[', ']', ' … ', 24) AS excerpt,
                bm25(memory_fts) AS score
            FROM memory_fts AS f
            JOIN records AS r ON r.record_id = f.record_id
            WHERE memory_fts MATCH ?
            ORDER BY score ASC, f.record_id ASC
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_records(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT record_id, source, content_sha256, license, language, kind, indexed_at
            FROM records ORDER BY indexed_at DESC, record_id ASC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def forget(self, record_id: str) -> bool:
        if not re.fullmatch(r"[a-f0-9]{64}", record_id):
            raise MemoryError("record id must be a full lowercase SHA-256 value")
        with self.connection:
            existing = self.connection.execute(
                "SELECT 1 FROM records WHERE record_id = ?", (record_id,)
            ).fetchone()
            if existing is None:
                return False
            self.connection.execute("DELETE FROM memory_fts WHERE record_id = ?", (record_id,))
            self.connection.execute("DELETE FROM records WHERE record_id = ?", (record_id,))
            self.connection.execute(
                "INSERT INTO audit_events (event, record_id, created_at) VALUES ('forgotten', ?, ?)",
                (record_id, _utc_now()),
            )
        remaining = self.connection.execute(
            "SELECT 1 FROM memory_fts WHERE record_id = ?", (record_id,)
        ).fetchone()
        if remaining is not None:
            raise MemoryError("deletion verification failed")
        return True

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])

