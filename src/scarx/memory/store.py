import hashlib
import json
from pathlib import Path
import re
import sqlite3
from contextlib import nullcontext


class EpisodeStore:
    """Append-only source text with bounded SQLite page cache and FTS5 search."""
    def __init__(self, path, *, check_same_thread=True):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path,check_same_thread=check_same_thread)
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            PRAGMA cache_size=-8192;
            PRAGMA mmap_size=0;
            CREATE TABLE IF NOT EXISTS episodes(
              id INTEGER PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL,
              kind TEXT NOT NULL, verified INTEGER NOT NULL, sha256 TEXT NOT NULL,
              created TEXT NOT NULL DEFAULT(strftime('%Y-%m-%dT%H:%M:%fZ','now')));
            CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(text,content='episodes',content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
              INSERT INTO search(rowid,text) VALUES(new.id,new.text);
            END;
        ''')

    def observe(self, text, source, *, kind="document", verified=False, commit=True):
        if not text or not source or len(text.encode("utf-8")) > 1_000_000:
            raise ValueError("Text/source required; each episode must be <= 1 MB UTF-8")
        if kind not in ("document","correction","generated","test_result"):
            raise ValueError("Invalid episode kind")
        if kind == "generated" and verified:
            raise ValueError("Unreviewed generated output cannot be a verified lesson")
        with (self.db if commit else nullcontext()):
            cur = self.db.execute("INSERT INTO episodes(text,source,kind,verified,sha256) VALUES(?,?,?,?,?)",
                                  (text,source,kind,int(verified),hashlib.sha256(text.encode()).hexdigest()))
        return cur.lastrowid

    def get(self, episode_id):
        self.db.row_factory = sqlite3.Row
        row = self.db.execute("SELECT * FROM episodes WHERE id=?",(episode_id,)).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return dict(row)

    def search(self, query, limit=5):
        if not 1 <= limit <= 20:
            raise ValueError("Search limit 1..20")
        terms = re.findall(r"\w+",query,flags=re.UNICODE)[:16]
        if not terms:
            return []
        fts_query = " OR ".join('"'+t.replace('"','""')+'"' for t in terms)
        self.db.row_factory = sqlite3.Row
        return [dict(r) for r in self.db.execute(
            "SELECT e.id,substr(e.text,1,4096) AS text,e.source,e.verified,e.created FROM search s JOIN episodes e ON e.id=s.rowid WHERE search MATCH ? ORDER BY rank LIMIT ?",
            (fts_query,limit))]

    def lessons(self, limit=32):
        if not 1 <= limit <= 128:
            raise ValueError("Replay limit 1..128")
        self.db.row_factory = sqlite3.Row
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM episodes WHERE verified=1 AND kind IN ('correction','test_result') ORDER BY id DESC LIMIT ?",(limit,))]

    def close(self):
        self.db.close()

