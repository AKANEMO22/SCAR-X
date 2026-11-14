"""Bounded SQLite caches; append-only evidence/audit, explicit evidence status."""

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class _DeferredConnection:
    """Defer nested connection contexts to one outer feedback transaction.

    Used only while Store.lock is held. SQLite connections normally commit at
    every nested `with connection`, so an ordinary outer context is insufficient.
    """
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Store:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(
            self.directory / "hybrid.sqlite", check_same_thread=False
        )
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL; PRAGMA cache_size=-4096; PRAGMA mmap_size=0;
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, created REAL NOT NULL,
                actor TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, stream TEXT NOT NULL,
                start REAL NOT NULL, end REAL NOT NULL, text TEXT NOT NULL, source TEXT NOT NULL,
                modality TEXT NOT NULL, status TEXT NOT NULL, public_topic TEXT,
                created REAL NOT NULL, digest TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(id UNINDEXED,text);
            CREATE TABLE IF NOT EXISTS objects(kind TEXT NOT NULL,id TEXT NOT NULL,payload TEXT NOT NULL,
                PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS checkpoints(id TEXT PRIMARY KEY,created REAL NOT NULL,
                digest TEXT NOT NULL,payload BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS evidence_stream_time ON evidence(stream,start,end);
        """)
        self.db.commit()

    def event(self, kind, payload, actor="runtime"):
        if actor not in (
            "runtime",
            "user",
            "orchestrator",
            "heuristic",
            "neural_policy",
            "verifier",
            "vlm",
            "asr",
        ):
            raise ValueError("Unknown actor")
        text = encode(payload)
        if len(text.encode()) > 131072:
            raise ValueError("Audit event too large")
        with self.lock, self.db:
            cursor = self.db.execute(
                "INSERT INTO events(created,actor,kind,payload) VALUES(?,?,?,?)",
                (time.time(), actor, kind, text),
            )
            return cursor.lastrowid

    def events(self, after=0, limit=100):
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?",
                (int(after), max(1, min(int(limit), 200))),
            ).fetchall()
        return [dict(row, payload=json.loads(row["payload"])) for row in rows]

    def put(self, kind, identifier, value):
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO objects VALUES(?,?,?)",
                (kind, identifier, encode(value)),
            )

    def get(self, kind, identifier):
        with self.lock:
            row = self.db.execute(
                "SELECT payload FROM objects WHERE kind=? AND id=?", (kind, identifier)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def atomic_feedback(self, request_id, fingerprint, callback):
        if (not isinstance(request_id, str) or not 1 <= len(request_id) <= 128
                or any(not (c.isalnum() or c in '-_.') for c in request_id)
                or not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(c not in '0123456789abcdef' for c in fingerprint)):
            raise ValueError('Invalid feedback retry identity')
        key = hashlib.sha256(request_id.encode()).hexdigest()
        with self.lock:
            cached = self.get('feedback_receipt', key)
            if cached is not None:
                if cached.get('fingerprint') != fingerprint:
                    raise ValueError('Feedback retry identity was reused with different content')
                return cached['response']
            connection = self.db
            if isinstance(connection, _DeferredConnection) or connection.in_transaction:
                raise RuntimeError('Feedback needs its own outer transaction')
            self.db = _DeferredConnection(connection)
            try:
                with connection:
                    result = callback()
                    self.put('feedback_receipt', key,
                             {'fingerprint': fingerprint, 'response': result})
                return result
            finally:
                self.db = connection

    def list(self, kind, limit=200):
        with self.lock:
            rows = self.db.execute(
                "SELECT payload FROM objects WHERE kind=? ORDER BY rowid DESC LIMIT ?",
                (kind, min(max(1, int(limit)), 1000)),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def observe(
        self,
        text,
        *,
        stream="local",
        start=0.0,
        end=0.0,
        source,
        modality="text",
        status="observed",
        public_topic=None,
    ):
        if status not in (
            "observed",
            "human_statement",
            "supported",
            "verified",
            "contradicted",
        ):
            raise ValueError("Unknown evidence status")
        # Verified is never accepted from a free-text caller or a checkbox.
        if status == "verified":
            raise ValueError(
                "Verification requires a separately recorded verifier result"
            )
        if (
            not isinstance(text, str)
            or not 0 < len(text) <= 16000
            or not 0 <= start <= end
        ):
            raise ValueError("Invalid timestamped evidence")
        item = dict(
            id=uuid.uuid4().hex,
            stream=stream,
            start=float(start),
            end=float(end),
            text=text,
            source=str(source),
            modality=modality,
            status=status,
            public_topic=public_topic,
            created=time.time(),
            digest=hashlib.sha256(text.encode()).hexdigest(),
        )
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO evidence VALUES(:id,:stream,:start,:end,:text,:source,:modality,:status,:public_topic,:created,:digest)",
                item,
            )
            self.db.execute("INSERT INTO evidence_fts VALUES(?,?)", (item["id"], text))
        self.event(
            "evidence_observed",
            {k: v for k, v in item.items() if k != "text"} | {"text": text[:2048]},
        )
        return item

    def search(self, query, stream=None, limit=12, before=None):
        import re

        if not isinstance(query, str) or len(query) > 6000:
            raise ValueError("Search query exceeds bounded 6000-character working request")
        words = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
        if not words:
            return []
        if len(words) <= 16:
            match = " OR ".join('"' + word + '"' for word in words)
        else:
            # Full bounded request, independent of labels or source contents.
            terms = list(dict.fromkeys(word.lower() for word in words))
            nodes = ['"' + word + '"' for word in terms]
            while len(nodes) > 1:
                nodes = ["(" + nodes[i] + " OR " + nodes[i + 1] + ")"
                         if i + 1 < len(nodes) else nodes[i]
                         for i in range(0, len(nodes), 2)]
            match = nodes[0]
        sql = "SELECT e.* FROM evidence_fts f JOIN evidence e ON e.id=f.id WHERE evidence_fts MATCH ?"
        params = [match]
        if stream:
            sql += " AND e.stream=?"
            params.append(stream)
        if before is not None:
            sql += " AND e.end<=?"
            params.append(float(before))
        sql += " ORDER BY bm25(evidence_fts),e.start LIMIT ?"
        params.append(max(1, min(limit, 32)))
        with self.lock:
            return [dict(row) for row in self.db.execute(sql, params)]

    def evidence(self, identifier):
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM evidence WHERE id=?", (identifier,)
            ).fetchone()
        return dict(row) if row else None

    def timeline(self, stream, limit=12, before=None, descending=False, modality=None):
        sql = "SELECT * FROM evidence WHERE stream=?"
        params = [stream]
        if before is not None:
            sql += " AND end<=?"
            params.append(float(before))
        if modality is not None:
            sql += " AND modality=?"
            params.append(modality)
        sql += " ORDER BY start " + ("DESC" if descending else "ASC") + " LIMIT ?"
        params.append(min(12, max(1, int(limit))))
        with self.lock:
            return [dict(row) for row in self.db.execute(sql, params)]

    def save_snapshot(self, payload, pinned=False, learning_event=None):
        if not isinstance(payload, bytes) or len(payload) > 16_000_000:
            raise ValueError("Snapshot exceeds bounded local-learning state")
        identifier = uuid.uuid4().hex
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO checkpoints VALUES(?,?,?,?)",
                (identifier, time.time(), hashlib.sha256(payload).hexdigest(), payload),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO objects VALUES(?,?,?)",
                ("runtime", "active_checkpoint", encode({"id": identifier})),
            )
            if pinned:
                self.db.execute(
                    "INSERT INTO objects VALUES(?,?,?)",
                    ("checkpoint_pin", identifier, encode({"id": identifier})),
                )
            self.db.execute(
                "DELETE FROM checkpoints WHERE id NOT IN (SELECT id FROM checkpoints ORDER BY created DESC LIMIT 32) AND id NOT IN (SELECT id FROM objects WHERE kind='checkpoint_pin')"
            )
            self.db.execute(
                "INSERT INTO events(created,actor,kind,payload) VALUES(?,?,?,?)",
                (
                    time.time(),
                    "runtime",
                    "checkpoint_saved",
                    encode({"id": identifier, "bytes": len(payload), "pinned": pinned}),
                ),
            )
            if learning_event is not None:
                self.db.execute(
                    "INSERT INTO events(created,actor,kind,payload) VALUES(?,?,?,?)",
                    (
                        time.time(),
                        "neural_policy",
                        "local_learning_update",
                        encode(learning_event),
                    ),
                )
        return identifier

    def snapshot(self, identifier):
        with self.lock:
            row = self.db.execute(
                "SELECT digest,payload FROM checkpoints WHERE id=?", (identifier,)
            ).fetchone()
        if row is None or hashlib.sha256(row["payload"]).hexdigest() != row["digest"]:
            raise ValueError("Missing/corrupt checkpoint")
        return row["payload"]

    def close(self):
        with self.lock:
            self.db.close()
