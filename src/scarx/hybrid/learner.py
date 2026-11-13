"""Hybrid-only H128 ALIF, local forward eligibility and one-step output credit.

Forward/trace equations derive from audited R8 MatchedLocalCredit and H128 Own
(main 37749041; pinned peer607ffca). New encoder and two masked output heads
require new experiments: prior toy-task scores are NOT transferred here.
No torch, autodiff, temporal backward pass or cross-neuron Jacobian recursion.
"""

from dataclasses import asdict
import hashlib
import io
import json
import re
import zipfile
import time
import numpy as np
from ..neural.core import Config, SpikingCore
from ..evaluation.readout_eligibility import spike_slope

ACTIONS = ("review", "search", "verify", "ask", "defer", "consolidate")
MODES = ("learned", "readout_only", "frozen", "shuffled")


def features(text, numeric=()):
    """Versioned fixed signed unigram/bigram hashing; no target/task-ID features."""
    tokens = re.findall(r"[^\W_]+", str(text).lower(), re.UNICODE)[:128]
    vector = np.zeros(64, np.float32)
    for term in tokens + [a + " " + b for a, b in zip(tokens, tokens[1:])]:
        digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % 48
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = np.linalg.norm(vector[:48])
    if norm:
        vector[:48] *= 2.0 / norm
    for i, value in enumerate(tuple(numeric)[:16]):
        vector[48 + i] = np.clip(float(value), -1, 1)
    if not np.isfinite(vector).all():
        raise ValueError("Nonfinite feature")
    # Multiple causal ticks, not a vocabulary/sequence-length claim.
    return np.repeat(vector[None, :], 8, axis=0)


class LocalALIF:
    def __init__(self, seed=260113, mode="learned"):
        if mode not in MODES:
            raise ValueError("Unknown learning condition")
        self.seed = int(seed)
        self.mode = mode
        self.core = SpikingCore(
            Config(
                inputs=64,
                hidden=128,
                outputs=8,
                fan_in=8,
                plastic_rows=12,
                seed=self.seed,
            )
        )
        shape = (128, 72)
        self.e_v = np.zeros(shape, np.float32)
        self.e_a = np.zeros(shape, np.float32)
        self.e_f = np.zeros(shape, np.float32)
        self.syn_trace = np.zeros(shape, np.float32)
        self.syn_energy = np.full(shape, 1e-4, np.float32)
        self.importance = np.zeros(shape, np.float32)
        self.f_mean = np.zeros(128, np.float32)
        self.f_var = np.full(128, 0.01, np.float32)
        self.gamma = np.linspace(0.25, 0.92, 128, dtype=np.float32)[:, None]
        self.rng = np.random.default_rng(self.seed + 35_100_000)
        self.signal_rng = np.random.default_rng(self.seed + 35_100_000)
        self.updates = 0
        self.hidden_updates = 0
        self.ticks = 0
        self.last_step_epoch = (
            0.0  # Ephemeral measurement, not learning/checkpoint state.
        )
        self.replay_seen = 0
        self.replay = []
        self.reward_mean = 0.0
        self.device = "cpu"
        self.gpu_device = 0
        self.xp = np

    def to_device(self, device, gpu_device=0):
        if device not in ("cpu", "gpu"):
            raise ValueError("Expected cpu/gpu learner device")
        if device == self.device:
            if device == "gpu" and gpu_device != self.gpu_device:
                raise ValueError("Use checkpoint reload to change CUDA device")
            return self
        if device == "cpu":
            payload = self.dumps()
            restored, _ = type(self).loads(payload)
            self.__dict__.update(restored.__dict__)
            return self
        from .cuda_alif import CudaSpikingCore, cupy_backend

        cp = cupy_backend()

        cp.cuda.Device(gpu_device).use()
        core = CudaSpikingCore(self.core, cp)
        names = [k for k in self.arrays() if not k.startswith("core.")]
        for name in names:
            setattr(self, name, cp.asarray(getattr(self, name)))
        self.core, self.xp = core, cp
        self.device, self.gpu_device = "gpu", gpu_device
        cp.cuda.get_current_stream().synchronize()
        return self

    def host(self, value):
        return self.xp.asnumpy(value) if self.device == "gpu" else value

    def active_spikes(self):
        return int(self.xp.count_nonzero(self.core.spikes))

    def arrays(self):
        return {"core." + k: v for k, v in self.core.arrays().items()} | {
            k: getattr(self, k)
            for k in (
                "e_v",
                "e_a",
                "e_f",
                "syn_trace",
                "syn_energy",
                "importance",
                "f_mean",
                "f_var",
                "gamma",
            )
        }

    def step(self, row, track=True):
        xp = self.xp
        c = self.core.config
        row = xp.asarray(row, xp.float32)
        if row.shape != (64,) or not xp.isfinite(row).all():
            raise ValueError("Invalid feature row")
        if track:
            old = self.core.spikes
            theta = c.threshold + c.adaptation_strength * self.core.adaptation
            u = (
                c.leak * self.core.voltage
                + self.core.w_in @ row
                + xp.sum(self.core.w_rec * old[self.core.indices], axis=1)
            )
            if self.device == "cpu":
                psi = spike_slope(u - theta)
            else:
                raw_slope = (u - theta) / 0.4
                probability = 1 / (1 + xp.exp(-xp.clip(raw_slope, -20, 20)))
                psi = (
                    probability * (1 - probability) / 0.4 * (xp.abs(raw_slope) < 20)
                ).astype(xp.float32)
            pre = xp.concatenate(
                (xp.broadcast_to(row, (128, 64)), old[self.core.indices]), axis=1
            )
            u_e = c.leak * self.e_v + pre
            a_old = self.e_a.copy()
            z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)
        self.core.step(row, track=False)
        if track:
            self.e_v[:] = (
                u_e
                - theta[:, None] * z_e
                - self.core.spikes[:, None] * c.adaptation_strength * a_old
            )
            self.e_a[:] = c.adaptation_decay * a_old + z_e
            z_e[:, 64:] *= 1.0 if xp.max(xp.abs(row)) > 0.6 else 0.15
            self.e_f[:] = self.gamma * self.e_f + (1.0 - self.gamma) * z_e
        self.ticks += 1
        self.last_step_epoch = time.time()

    def _forward_native(self, sequence, head="action", track=False):
        xp = self.xp
        if self.device == "gpu":
            xp.cuda.Device(self.gpu_device).use()
        seq = xp.asarray(sequence, xp.float32)
        if (
            seq.ndim != 2
            or seq.shape[1] != 64
            or not 1 <= len(seq) <= 256
            or not xp.isfinite(seq).all()
        ):
            raise ValueError("Invalid local sequence")
        if head not in ("action", "relevance"):
            raise ValueError("Unknown readout head")
        self.core.reset_state()
        self.e_v.fill(0)
        self.e_a.fill(0)
        self.e_f.fill(0)
        for row in seq:
            self.step(row, track)
        scale = xp.maximum(xp.sqrt(self.f_var), 0.05)
        raw = (self.core.features - self.f_mean) / scale
        standardized = xp.clip(raw, -5, 5)
        indices = xp.arange(2, 8) if head == "action" else xp.arange(2)
        logits = self.core.w_out[indices] @ standardized + self.core.bias[indices]
        logits -= logits.max()
        p = xp.exp(logits)
        p /= p.sum()
        return p.astype(xp.float32), standardized, raw, scale, indices

    def forward(self, sequence, head="action", track=False):
        return tuple(self.host(v) for v in self._forward_native(sequence, head, track))

    def decide(self, sequence, allowed):
        if not allowed or any(a not in ACTIONS for a in allowed):
            raise ValueError("No valid action")
        p, *_ = self.forward(sequence)
        index = max((ACTIONS.index(a) for a in allowed), key=lambda i: float(p[i]))
        return ACTIONS[index], dict(zip(ACTIONS, map(float, p)))

    def relevance(self, query, evidence):
        sequence = features(
            query + " [event] " + evidence["text"],
            (min(len(evidence["text"]) / 4000, 1.0),),
        )
        return float(self.forward(sequence, head="relevance")[0][1])

    def learn(
        self,
        sequence,
        target,
        *,
        head="action",
        reward=1.0,
        evidence=None,
        remember=True,
    ):
        if (
            not isinstance(evidence, dict)
            or evidence.get("kind") not in ("human_feedback", "machine_verifier")
            or not evidence.get("id")
        ):
            raise ValueError(
                "Learning needs identified human feedback or machine verification"
            )
        if not np.isfinite(reward) or not -1 <= reward <= 1:
            raise ValueError("Invalid reward")
        xp = self.xp
        p, f, raw, scale, indices = self._forward_native(sequence, head, track=True)
        if type(target) is not int or not 0 <= target < len(indices):
            raise ValueError("Invalid action/label")
        loss = -float(xp.log(xp.maximum(p[target].astype(xp.float64), 1e-12)))
        before = {
            k: v.copy() for k, v in self.core.arrays().items() if k.startswith("w_")
        }
        if self.mode != "frozen":
            error = -p.copy()
            error[target] += 1.0
            error *= float(reward)
            # Pre-update readout feedback; no reverse traversal through recurrence.
            signal = (self.core.w_out[indices].T @ error) * (xp.abs(raw) < 5) / scale
            self.core.w_out[indices] += xp.clip(
                0.04 * error[:, None] * f[None, :], -0.025, 0.025
            )
            self.core.bias[indices] += xp.clip(0.04 * error, -0.025, 0.025)
            if self.mode in ("learned", "shuffled"):
                if self.mode == "shuffled":
                    signal = signal[xp.asarray(self.signal_rng.permutation(128))]
                direction = signal[:, None] * self.e_f
                self.syn_trace[:] = 0.9 * self.syn_trace + 0.1 * direction
                self.syn_energy[:] = 0.99 * self.syn_energy + 0.01 * direction**2
                delta = xp.clip(
                    0.008 * self.syn_trace / (xp.sqrt(self.syn_energy) + 1e-4),
                    -0.01,
                    0.01,
                )
                self.core.w_in += delta[:, :64]
                self.core.w_rec += delta[:, 64:]
                self.hidden_updates += 1
            for name in ("w_in", "w_rec", "w_out", "bias"):
                xp.clip(
                    getattr(self.core, name), -3.0, 3.0, out=getattr(self.core, name)
                )
            self.updates += 1
            self.core.updates += 1
        if not all(xp.isfinite(a).all() for a in self.arrays().values()):
            raise FloatingPointError("Nonfinite local update")
        # Statistics use only the currently observed experience, never future calibration.
        if self.mode != "frozen":
            self.f_mean[:] = 0.99 * self.f_mean + 0.01 * self.core.features
            self.f_var[:] = (
                0.99 * self.f_var + 0.01 * (self.core.features - self.f_mean) ** 2
            )
        if remember and self.mode != "frozen":
            item = {
                "sequence": np.asarray(sequence, np.float32).tolist(),
                "target": target,
                "head": head,
                "reward": float(reward),
                "evidence": evidence,
            }
            self.replay_seen += 1
            if len(self.replay) < 64:
                self.replay.append(item)
            else:
                index = int(self.rng.integers(self.replay_seen))
                if index < 64:
                    self.replay[index] = item
        return {
            "loss": loss,
            "updates": self.updates,
            "hidden_updates": self.hidden_updates,
            "weight_delta": {
                k: float(xp.linalg.norm(getattr(self.core, k) - v))
                for k, v in before.items()
            },
            "accepted": True,
            "mode": self.mode,
            "reward": float(reward),
            "evidence": evidence,
        }

    def consolidate(self):
        if not self.replay:
            return None
        row = self.replay[int(self.rng.integers(len(self.replay)))]
        return self.learn(
            row["sequence"],
            row["target"],
            head=row["head"],
            reward=row["reward"],
            evidence=row["evidence"],
            remember=False,
        )

    def memory_bytes(self):
        return sum(a.nbytes for a in self.arrays().values()) + len(
            json.dumps(self.replay).encode()
        )

    def dumps(self, controller=None):
        meta = dict(
            format="scarx-hybrid-alif-v2",
            encoder="signed-hash64-v1",
            seed=self.seed,
            mode=self.mode,
            updates=self.updates,
            hidden_updates=self.hidden_updates,
            ticks=self.ticks,
            core_updates=self.core.updates,
            reward_mean=self.reward_mean,
            replay_seen=self.replay_seen,
            replay=self.replay,
            rng=self.rng.bit_generator.state,
            signal_rng=self.signal_rng.bit_generator.state,
            controller=controller or {},
        )
        values = {k: self.host(v) for k, v in self.arrays().items()} | {
            "metadata": np.frombuffer(
                json.dumps(meta, allow_nan=False).encode(), dtype=np.uint8
            )
        }
        output = io.BytesIO()
        np.savez(output, **values)  # ZIP_STORED: no model compression.
        return output.getvalue()

    @classmethod
    def loads(cls, payload, device="cpu", gpu_device=0):
        if not isinstance(payload, bytes) or len(payload) > 16_000_000:
            raise ValueError("Snapshot byte bound")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]
            if (
                len(names) != len(set(names))
                or sum(i.file_size for i in infos) > 16_000_000
            ):
                raise ValueError("Duplicate/oversized snapshot members")
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            metadata = data["metadata"]
            if (
                metadata.dtype != np.uint8
                or metadata.ndim != 1
                or metadata.size > 2_000_000
            ):
                raise ValueError("Metadata type/bound")
            meta = json.loads(metadata.tobytes())
            if (
                meta.get("format")
                not in ("scarx-hybrid-alif-v1", "scarx-hybrid-alif-v2")
                or meta.get("encoder") != "signed-hash64-v1"
            ):
                raise ValueError("Incompatible checkpoint")
            if (
                meta.get("format") == "scarx-hybrid-alif-v1"
                and meta.get("mode") == "shuffled"
                and meta.get("updates", 0) > 0
            ):
                raise ValueError(
                    "Legacy shuffled RNG streams cannot be migrated with exact continuation"
                )
            model = cls(meta["seed"], meta["mode"])
            expected = model.arrays()
            if set(data.files) != set(expected) | {"metadata"}:
                raise ValueError("Unexpected/missing snapshot arrays")
            for key, target in expected.items():
                value = data[key]
                if (
                    value.shape != target.shape
                    or value.dtype != target.dtype
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("Invalid array " + key)
                target[:] = value
            if np.any(model.core.indices < 0) or np.any(model.core.indices >= 128):
                raise ValueError("Invalid topology indices")
            if (
                np.any(model.core.rows < 0)
                or np.any(model.core.rows >= 128)
                or np.any(model.f_var < 0)
                or np.any(model.syn_energy <= 0)
                or np.any(model.importance < 0)
            ):
                raise ValueError("Invalid local state bounds")
            for name in ("updates", "hidden_updates", "ticks", "replay_seen"):
                if type(meta[name]) is not int or meta[name] < 0:
                    raise ValueError("Invalid counter")
                setattr(model, name, meta[name])
            if type(meta["core_updates"]) is not int or meta["core_updates"] < 0:
                raise ValueError("Invalid core counter")
            model.core.updates = meta["core_updates"]
            model.reward_mean = float(meta["reward_mean"])
            if not np.isfinite(model.reward_mean):
                raise ValueError("Invalid reward baseline")
            if not isinstance(meta["replay"], list) or len(meta["replay"]) > 64:
                raise ValueError("Invalid replay bound")
            for row in meta["replay"]:
                seq = np.asarray(row["sequence"], np.float32)
                if (
                    seq.ndim != 2
                    or seq.shape[1] != 64
                    or not 1 <= len(seq) <= 256
                    or not np.isfinite(seq).all()
                ):
                    raise ValueError("Invalid replay sequence")
                if (
                    row["head"] not in ("action", "relevance")
                    or type(row["target"]) is not int
                    or not 0 <= row["target"] < (6 if row["head"] == "action" else 2)
                ):
                    raise ValueError("Invalid replay label")
                if (
                    not np.isfinite(row["reward"])
                    or not -1 <= row["reward"] <= 1
                    or row["evidence"].get("kind")
                    not in ("human_feedback", "machine_verifier")
                ):
                    raise ValueError("Invalid replay evidence")
            model.replay = meta["replay"]
            model.rng.bit_generator.state = meta["rng"]
            if "signal_rng" in meta:
                model.signal_rng.bit_generator.state = meta["signal_rng"]
            if not isinstance(meta["controller"], dict):
                raise ValueError("Invalid controller state")
            model.to_device(device, gpu_device)
            return model, meta["controller"]
