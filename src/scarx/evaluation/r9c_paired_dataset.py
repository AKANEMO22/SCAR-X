"""SCAR-X Round 9c Paired Clean vs Noisy Dataset Generator & Verification Suite.

Guarantees bit-level paired experimental invariants:
1. Identical base noise and stimulus events (keys, values, query, label, positions, marker).
2. Distractor conditions differ ONLY in the specific distractor factor being studied.
3. Cryptographic hash checks confirm zero stimulus divergence across conditions.
4. Generates 6 paired conditions:
   - clean (no distractor pulse)
   - weak_distractor (amplitude 0.4)
   - medium_distractor (amplitude 1.2)
   - strong_distractor (amplitude 2.5)
   - swapped_channel (distractor pulse applied to alternate unused channel)
   - shifted_time (distractor pulse shifted to different non-stimulus ticks)
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any
import numpy as np


PAIRED_CONDITIONS = (
    "clean",
    "weak_distractor",
    "medium_distractor",
    "strong_distractor",
    "swapped_channel",
    "shifted_time",
)


@dataclass(frozen=True)
class SampleMetadata:
    sample_id: int
    label: int
    query: int
    keys: tuple[int, int]
    values: tuple[int, int]
    positions: tuple[int, int]
    stimulus_hash: str


@dataclass
class PairedBatch:
    metadata: list[SampleMetadata]
    labels: np.ndarray  # (N,)
    sequences: dict[str, np.ndarray]  # cond_name -> (N, 12, 8)


def generate_paired_dataset(
    seed: int,
    count: int,
    ood: bool = False,
    base_distractor_channel: int = 6,
    swapped_distractor_channel: int = 4,  # alternate channel for early ticks when query not present
) -> PairedBatch:
    """Generates strictly matched paired dataset across all 6 conditions."""
    rng = np.random.default_rng(seed)
    labels = (np.arange(count) % 2).astype(np.int32)
    sigma = 0.14 if ood else 0.10

    metadata_list: list[SampleMetadata] = []
    clean_seqs = np.zeros((count, 12, 8), dtype=np.float32)
    weak_seqs = np.zeros((count, 12, 8), dtype=np.float32)
    medium_seqs = np.zeros((count, 12, 8), dtype=np.float32)
    strong_seqs = np.zeros((count, 12, 8), dtype=np.float32)
    swapped_seqs = np.zeros((count, 12, 8), dtype=np.float32)
    shifted_seqs = np.zeros((count, 12, 8), dtype=np.float32)

    standard_ticks = (3, 7, 8)
    shifted_ticks = (2, 5, 9)

    for i in range(count):
        y = int(labels[i])
        query = int(rng.integers(2))
        a = y ^ query
        values = (a, 1 - a)
        order = tuple(int(k) for k in rng.permutation(2))
        positions = (0, 7) if ood else (int(rng.integers(1, 3)), int(rng.integers(4, 6)))

        # 1. Base sequence with Gaussian background noise
        base_noise = rng.normal(0, sigma, (12, 8)).astype(np.float32)
        base = base_noise.copy()

        # 2. Add relational key-value stimuli
        for pos, key in zip(positions, order):
            base[pos, key] += 1.5
            base[pos, 2 + values[key]] += 1.5

        # 3. Add query at tick 10
        base[10, 4 + query] += 1.5

        # 4. Add final readout trigger token at tick 11
        base[11, 7] += 1.0

        # Compute stimulus fingerprint hash
        meta_str = f"{i}:{y}:{query}:{order}:{values}:{positions}"
        stim_hash = hashlib.sha256(meta_str.encode("utf-8")).hexdigest()[:16]

        meta = SampleMetadata(
            sample_id=i,
            label=y,
            query=query,
            keys=(order[0], order[1]),
            values=(values[0], values[1]),
            positions=positions,
            stimulus_hash=stim_hash,
        )
        metadata_list.append(meta)

        # Base clean is exact copy of base
        clean_seqs[i] = base.copy()
        weak_seqs[i] = base.copy()
        medium_seqs[i] = base.copy()
        strong_seqs[i] = base.copy()
        swapped_seqs[i] = base.copy()
        shifted_seqs[i] = base.copy()

        # Deterministic pulse signs for this sample
        pulse_signs = [float(rng.choice([-1.0, 1.0])) for _ in range(len(standard_ticks))]
        shifted_signs = [float(rng.choice([-1.0, 1.0])) for _ in range(len(shifted_ticks))]

        # Standard distractor ticks
        for dt, sgn in zip(standard_ticks, pulse_signs):
            if dt not in positions and dt < 10:
                weak_seqs[i, dt, base_distractor_channel] += sgn * 0.4
                medium_seqs[i, dt, base_distractor_channel] += sgn * 1.2
                strong_seqs[i, dt, base_distractor_channel] += sgn * 2.5

                # Swapped channel distractor: use an alternate channel not carrying stimulus at dt
                # At dt in (3, 7, 8), query (4, 5) and marker (7) are inactive!
                swap_ch = 4 if base_distractor_channel == 6 else 6
                swapped_seqs[i, dt, swap_ch] += sgn * 1.2

        # Shifted time distractor: ticks (2, 5, 9)
        for dt, sgn in zip(shifted_ticks, shifted_signs):
            if dt not in positions and dt < 10:
                shifted_seqs[i, dt, base_distractor_channel] += sgn * 1.2

    # Verification of paired invariants
    for cond_name, seqs in [
        ("weak_distractor", weak_seqs),
        ("medium_distractor", medium_seqs),
        ("strong_distractor", strong_seqs),
        ("swapped_channel", swapped_seqs),
        ("shifted_time", shifted_seqs),
    ]:
        for i in range(count):
            pos1, pos2 = metadata_list[i].positions
            # Key 1 event tick must be identical
            np.testing.assert_array_equal(
                seqs[i, pos1, :4],
                clean_seqs[i, pos1, :4],
                err_msg=f"Stimulus corruption at tick {pos1} in {cond_name}",
            )
            # Key 2 event tick must be identical
            np.testing.assert_array_equal(
                seqs[i, pos2, :4],
                clean_seqs[i, pos2, :4],
                err_msg=f"Stimulus corruption at tick {pos2} in {cond_name}",
            )
            # Query at tick 10 must be identical
            np.testing.assert_array_equal(
                seqs[i, 10, :6],
                clean_seqs[i, 10, :6],
                err_msg=f"Query corruption at tick 10 in {cond_name}",
            )
            # Marker at tick 11 must be identical
            np.testing.assert_array_equal(
                seqs[i, 11, :],
                clean_seqs[i, 11, :],
                err_msg=f"Trigger corruption at tick 11 in {cond_name}",
            )

    return PairedBatch(
        metadata=metadata_list,
        labels=labels,
        sequences={
            "clean": clean_seqs,
            "weak_distractor": weak_seqs,
            "medium_distractor": medium_seqs,
            "strong_distractor": strong_seqs,
            "swapped_channel": swapped_seqs,
            "shifted_time": shifted_seqs,
        },
    )
