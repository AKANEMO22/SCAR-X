"""SCAR-X Round 12: Reservoir Replay Buffer for Continual Memory.

Implements:
- Algorithm R (Waterman / Vitter 1985) for task-boundary-free continual streaming.
- Ultra-compact footprint (Capacity M in {16, 64}, <100 KB RAM).
- Zero task-ID or task metadata storage.
- Deterministic seeded RNG.
- Strict protection against future example leakage.
"""
from typing import Optional, Tuple
import numpy as np


class ReservoirReplayBuffer:
    """Reservoir Replay Buffer with fixed capacity and uniform sampling over past stream."""

    def __init__(self, capacity: int, seed: int = 140011):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.buffer_x = []
        self.buffer_y = []
        self.total_seen = 0

    def __len__(self) -> int:
        return len(self.buffer_x)

    @property
    def memory_bytes(self) -> int:
        """Estimate buffer memory footprint in bytes."""
        x_bytes = sum(x.nbytes for x in self.buffer_x)
        y_bytes = len(self.buffer_y) * 8
        return x_bytes + y_bytes

    def sample_replay(self) -> Optional[Tuple[np.ndarray, int]]:
        """Sample one past example uniformly from the current reservoir.
        
        Returns None if buffer is empty.
        Must be called BEFORE inserting the new incoming example if replay
        should be selected from strictly prior history.
        """
        if len(self.buffer_x) == 0:
            return None
        idx = int(self.rng.integers(0, len(self.buffer_x)))
        return self.buffer_x[idx].copy(), int(self.buffer_y[idx])

    def insert(self, x: np.ndarray, y: int) -> bool:
        """Insert a new incoming example using Algorithm R.
        
        Returns True if item was accepted into reservoir, False if discarded.
        """
        self.total_seen += 1
        x_arr = np.asarray(x, dtype=np.float32)
        y_val = int(y)

        if len(self.buffer_x) < self.capacity:
            self.buffer_x.append(x_arr.copy())
            self.buffer_y.append(y_val)
            return True
        else:
            # Draw j uniformly from [0, total_seen - 1]
            j = int(self.rng.integers(0, self.total_seen))
            if j < self.capacity:
                self.buffer_x[j] = x_arr.copy()
                self.buffer_y[j] = y_val
                return True
            return False

    def get_contents_copy(self):
        """Return deep copy of current buffer contents for auditing."""
        return [x.copy() for x in self.buffer_x], list(self.buffer_y)
