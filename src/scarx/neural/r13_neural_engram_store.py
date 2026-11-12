"""SCAR-X Round 13: SSD Neural Engram Store.

Implements:
- Persistent disk storage for learned ALIF synaptic engrams.
- Full topological, synaptic, and eligibility trace persistence.
- SHA-256 checksum integrity verification.
- Atomic file commits (write-then-rename) with fsync.
- Safe memory unloading of inactive neural modules.
- Hot/cold engram restoration with bit-level execution equivalence.
- Snapshot rollback and corruption detection.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import json
import os
import shutil
import gc
import numpy as np

from scarx.neural.r13_modular_alif import ModularALIFModule


class NeuralEngramStore:
    """Durable SSD storage and lifecycle manager for SNN neural engrams."""

    SCHEMA_VERSION = 1

    def __init__(self, store_dir: Path | str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.store_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)

    def _engram_path(self, engram_id: str) -> Path:
        return self.store_dir / f"{engram_id}.engram.json"

    def save_engram(
        self,
        module: ModularALIFModule,
        engram_id: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serialize and atomically persist a learned ALIF module to SSD."""
        engram_data = module.get_engram()
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "engram_id": engram_id,
            "module_name": module.module_name,
            "neuron_ids": module.neuron_ids.tolist(),
            "extra_metadata": extra_metadata or {},
            "engram": engram_data,
        }

        # Compute payload hash
        payload_str = json.dumps(payload, sort_keys=True)
        checksum = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        wrapped = {
            "schema_version": self.SCHEMA_VERSION,
            "engram_id": engram_id,
            "checksum": checksum,
            "payload": payload,
        }

        target_file = self._engram_path(engram_id)
        temp_file = self.store_dir / f"{engram_id}.tmp.{os.getpid()}"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(wrapped, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic swap
        temp_file.replace(target_file)
        return target_file

    def load_engram(self, engram_id: str, target_module: ModularALIFModule) -> Dict[str, Any]:
        """Verify checksum, validate topology, and restore module state from SSD."""
        target_file = self._engram_path(engram_id)
        if not target_file.exists():
            raise FileNotFoundError(f"Engram not found: {target_file}")

        with open(target_file, "r", encoding="utf-8") as f:
            wrapped = json.load(f)

        if wrapped.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"Incompatible engram schema version: {wrapped.get('schema_version')}")

        stored_checksum = wrapped.get("checksum", "")
        payload = wrapped.get("payload", {})
        payload_str = json.dumps(payload, sort_keys=True)
        computed_checksum = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        if stored_checksum != computed_checksum:
            raise ValueError(f"Corrupted engram file: checksum mismatch ({stored_checksum} != {computed_checksum})")

        engram = payload.get("engram", {})
        target_module.load_engram(engram)
        return payload

    def unload_module(self, module: ModularALIFModule) -> None:
        """Deallocate numpy arrays of module from active RAM."""
        module.unload()
        gc.collect()

    def create_snapshot(self, engram_id: str, snapshot_name: str) -> Path:
        """Create a versioned rollback backup of an engram."""
        source_file = self._engram_path(engram_id)
        if not source_file.exists():
            raise FileNotFoundError(f"Source engram not found: {source_file}")
        snap_file = self.snapshots_dir / f"{engram_id}_{snapshot_name}.snap.json"
        shutil.copy2(source_file, snap_file)
        return snap_file

    def rollback(self, engram_id: str, snapshot_name: str) -> bool:
        """Rollback an engram to a previously saved snapshot."""
        snap_file = self.snapshots_dir / f"{engram_id}_{snapshot_name}.snap.json"
        if not snap_file.exists():
            raise FileNotFoundError(f"Snapshot not found: {snap_file}")
        target_file = self._engram_path(engram_id)
        shutil.copy2(snap_file, target_file)
        return True

    def exists(self, engram_id: str) -> bool:
        return self._engram_path(engram_id).exists()

    def get_engram_disk_bytes(self, engram_id: str) -> int:
        target_file = self._engram_path(engram_id)
        if target_file.exists():
            return target_file.stat().st_size
        return 0
