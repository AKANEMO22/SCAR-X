"""SCAR-X Round 12: SSD Long-Term Replay Store Architecture.

Provides an offline, idle-time replay store designed for local SSD persistence:
- Strict binary/JSON schema with format versioning.
- SHA-256 checksum integrity verification.
- Atomic file writes (write-to-temp then atomic replace) to prevent corruption.
- Hard capacity bounds with FIFO / Reservoir retention.
- Snapshot rollback and resumable state.
- Separation of concerns:
    * External memory: durable storage of validated training episodes.
    * Replay rehearsal: staged reactivation during idle compute cycles.
    * Synaptic retention: persistent parameter consolidation in ALIF weights.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


class SSDReplayStore:
    """Durable, checksummed, atomic-write SSD replay repository."""

    SCHEMA_VERSION = 1

    def __init__(self, store_dir: Path | str, capacity: int = 1024):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.db_file = self.store_dir / "replay_store.json"
        self.snapshot_dir = self.store_dir / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        self.entries = []
        self._load_or_initialize()

    def _load_or_initialize(self):
        if not self.db_file.exists():
            self.entries = []
            self.commit()
            return

        with open(self.db_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify schema version
        if data.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"Incompatible schema version: {data.get('schema_version')}")

        # Verify checksum
        stored_checksum = data.get("checksum", "")
        payload_str = json.dumps(data.get("entries", []), sort_keys=True)
        computed_checksum = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        if stored_checksum != computed_checksum:
            raise ValueError(f"Corrupted replay store: checksum mismatch ({stored_checksum} != {computed_checksum})")

        self.entries = data.get("entries", [])

    def commit(self):
        """Atomic write to SSD using write-then-rename."""
        payload_str = json.dumps(self.entries, sort_keys=True)
        checksum = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        doc = {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "count": len(self.entries),
            "checksum": checksum,
            "entries": self.entries,
        }

        temp_file = self.store_dir / f"replay_store.tmp.{os.getpid()}"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replace
        temp_file.replace(self.db_file)

    def add_episode(self, sequence: np.ndarray, target: int, metadata: Optional[dict] = None) -> bool:
        """Add an episode to the SSD store under capacity constraints."""
        seq_arr = np.asarray(sequence, dtype=np.float32)
        entry = {
            "id": len(self.entries) + 1,
            "shape": list(seq_arr.shape),
            "data": seq_arr.tolist(),
            "target": int(target),
            "metadata": metadata or {},
        }

        if len(self.entries) >= self.capacity:
            # FIFO eviction when full
            self.entries.pop(0)

        self.entries.append(entry)
        self.commit()
        return True

    def get_replay_batch(self, batch_size: int, seed: int = 42) -> List[Tuple[np.ndarray, int]]:
        """Sample a batch of episodes for idle-time replay rehearsal."""
        if not self.entries:
            return []
        rng = np.random.default_rng(seed)
        n = min(batch_size, len(self.entries))
        indices = rng.choice(len(self.entries), size=n, replace=False)

        batch = []
        for idx in indices:
            e = self.entries[idx]
            arr = np.array(e["data"], dtype=np.float32)
            batch.append((arr, e["target"]))
        return batch

    def create_snapshot(self, snapshot_name: str) -> Path:
        """Create a versioned rollback snapshot on SSD."""
        snap_path = self.snapshot_dir / f"{snapshot_name}.json"
        shutil.copy2(self.db_file, snap_path)
        return snap_path

    def rollback(self, snapshot_name: str) -> bool:
        """Rollback repository to a named snapshot."""
        snap_path = self.snapshot_dir / f"{snapshot_name}.json"
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap_path}")

        shutil.copy2(snap_path, self.db_file)
        self._load_or_initialize()
        return True
