"""CUDA execution of the existing ALIF equations, with original FP32 arrays.

No optimizer/autograd or backward path. Host initialization/checkpoint encoding
is unchanged. This module is imported only for explicit GPU mode.
"""

from ..neural.core import SpikingCore
import os
from pathlib import Path

_DLL_HANDLES = []


def cupy_backend():
    # Reuse the pinned cuBLAS DLLs shipped with the native VLM instead of
    # downloading a second >500MB copy. NVRTC/cudart come from the small wheels.
    if os.name == "nt" and not _DLL_HANDLES:
        directory = (
            Path(__file__).resolve().parents[3]
            / "local/hybrid-v1/runtime/llama.cpp-b12360-cuda12.4"
        )
        if not directory.is_dir():
            raise RuntimeError("Install the pinned native CUDA runtime before GPU ALIF")
        _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
        import ctypes

        # cuda-pathfinder uses native LoadLibrary search, which does not consult
        # Python's added DLL directories. Preloading exact pinned paths lets its
        # documented already-loaded check reuse these libraries without PATH edits.
        for name in ("cublasLt64_12.dll", "cublas64_12.dll"):
            _DLL_HANDLES.append(ctypes.WinDLL(str(directory / name)))
    os.environ.setdefault(
        "CUPY_CACHE_DIR",
        str(Path(__file__).resolve().parents[3] / "local/hybrid-v1/cupy-cache"),
    )
    import cupy

    return cupy


class CudaSpikingCore(SpikingCore):
    def __init__(self, reference, xp):
        self.xp = xp
        self.config = reference.config
        self.updates = reference.updates
        self.reward_mean = reference.reward_mean
        for name, value in reference.arrays().items():
            setattr(self, name, xp.asarray(value))

    def reset_state(self):
        c, xp = self.config, self.xp
        self.voltage = xp.zeros(c.hidden, xp.float32)
        self.adaptation = xp.zeros(c.hidden, xp.float32)
        self.spikes = xp.zeros(c.hidden, xp.float32)
        self.pre = xp.zeros(c.inputs + c.hidden, xp.float32)
        self.post = xp.zeros(c.hidden, xp.float32)
        self.features = xp.zeros(c.hidden, xp.float32)
        start = (self.updates * c.plastic_rows) % c.hidden
        self.rows = (xp.arange(c.plastic_rows, dtype=xp.int32) + start) % c.hidden
        self.elig_in = xp.zeros((c.plastic_rows, c.inputs), xp.float32)
        self.elig_rec = xp.zeros((c.plastic_rows, c.fan_in), xp.float32)

    def step(self, x, track=False):
        if track:
            raise ValueError(
                "Hybrid eligibility is managed by LocalALIF, not core STDP"
            )
        c, xp = self.config, self.xp
        old = self.spikes.copy()
        current = self.w_in @ x + xp.sum(self.w_rec * old[self.indices], axis=1)
        self.voltage = c.leak * self.voltage + current
        limit = c.threshold + c.adaptation_strength * self.adaptation
        self.spikes = (self.voltage >= limit).astype(xp.float32)
        self.voltage -= self.spikes * limit
        self.adaptation = c.adaptation_decay * self.adaptation + self.spikes
        self.features = (
            c.feature_decay * self.features + (1 - c.feature_decay) * self.spikes
        )
        # Hybrid computes its masked readout once after the sequence. The old
        # core's per-tick probability return is unused here; avoid extra launches.
        return None

    def probabilities(self):
        xp = self.xp
        logits = self.w_out @ self.features + self.bias
        logits -= xp.max(logits)
        p = xp.exp(logits)
        return p / p.sum()
