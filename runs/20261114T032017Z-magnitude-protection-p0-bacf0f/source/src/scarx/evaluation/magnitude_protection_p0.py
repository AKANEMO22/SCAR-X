"""P0-only reused magnitude/anchor rule, never imported into serving."""

import hashlib
import io
import json
import zipfile
import numpy as np
from scarx.hybrid.learner import LocalALIF

FORMAT = "scarx-magnitude-protection-prototype-v1"
CONSTANTS = {"importance_scale": 5.0, "anchor_scale": .005,
             "importance_rate": .001, "importance_cap": 20.0,
             "anchor_period": 64, "anchor_rate": .02,
             "delta_clip": .025, "weight_clip": 3.0}


class MagnitudeProtectionP0:
    def __init__(self, seed=260113, mode="learned", device="gpu"):
        self.base = LocalALIF(seed, mode).to_device(device)
        self.output_importance = self.base.xp.zeros((8, 128), self.base.xp.float32)
        self.bias_importance = self.base.xp.zeros(8, self.base.xp.float32)

    def learn(self, sequence, target, **kwargs):
        model, xp = self.base, self.base.xp
        before = {k:getattr(model.core, k).copy() for k in ("w_in", "w_rec", "w_out", "bias")}
        result = model.learn(sequence, target, **kwargs)  # Original local credit only.
        if model.mode == "frozen":
            return {**result, "algorithm": FORMAT, "P0_only": True}
        groups = (
            ("w_in", "stable_in", model.importance[:, :64]),
            ("w_rec", "stable_rec", model.importance[:, 64:]),
            ("w_out", "stable_out", self.output_importance),
            ("bias", "stable_bias", self.bias_importance),
        )
        protection_delta = {}
        for key, stable_name, omega in groups:
            weight = getattr(model.core, key)
            delta = weight - before[key]
            active = delta != 0
            # Retain readout/head masks and exact zero-update groups. Magnitude
            # protection is a local regularizer, not newly invented credit.
            protected = xp.where(active,
                delta / (1 + CONSTANTS["importance_scale"] * omega)
                - CONSTANTS["anchor_scale"] * omega *
                    (before[key] - getattr(model.core, stable_name)), 0)
            weight[:] = xp.clip(before[key] + xp.clip(
                protected, -CONSTANTS["delta_clip"], CONSTANTS["delta_clip"]),
                -CONSTANTS["weight_clip"], CONSTANTS["weight_clip"])
            magnitude = xp.abs(delta)
            omega[:] = xp.minimum(CONSTANTS["importance_cap"], omega
                + CONSTANTS["importance_rate"] * magnitude /
                    (magnitude.mean() + 1e-6))
            protection_delta[key] = float(xp.linalg.norm(weight - before[key]))
        if model.updates % CONSTANTS["anchor_period"] == 0:
            for key, stable, _ in groups:
                anchor = getattr(model.core, stable)
                anchor[:] += CONSTANTS["anchor_rate"] * (getattr(model.core, key) - anchor)
        if not all(xp.isfinite(a).all() for a in (self.output_importance, self.bias_importance, *model.arrays().values())):
            raise FloatingPointError("Nonfinite protection prototype state")
        return {**result, "weight_delta": protection_delta, "algorithm": FORMAT,
                "P0_only": True, "importance_is_causal_skill_measure": False}

    def consolidate(self):
        if not self.base.replay:
            return None
        row = self.base.replay[int(self.base.rng.integers(len(self.base.replay)))]
        return self.learn(row["sequence"], row["target"], head=row["head"],
                          reward=row["reward"], evidence=row["evidence"], remember=False)

    def dumps(self):
        base = self.base.dumps()
        extra = io.BytesIO()
        np.savez(extra, output_importance=self.base.host(self.output_importance),
                 bias_importance=self.base.host(self.bias_importance))
        payload = extra.getvalue()
        meta = {"format": FORMAT, "constants": CONSTANTS,
                "base_sha256": hashlib.sha256(base).hexdigest(),
                "extra_sha256": hashlib.sha256(payload).hexdigest()}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, raw in (("metadata.json", json.dumps(meta, sort_keys=True).encode()),
                              ("local-alif.npz", base), ("protection.npz", payload)):
                item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_STORED
                archive.writestr(item, raw)
        raw = stream.getvalue()
        if len(raw) > 16_000_000:
            raise ValueError("Protection checkpoint exceeds bound")
        return raw

    @classmethod
    def loads(cls, raw, device="gpu"):
        if not isinstance(raw, bytes) or len(raw) > 16_000_000:
            raise ValueError("Invalid prototype checkpoint")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (len(entries) != 3 or {e.filename for e in entries} != {
                    "metadata.json", "local-alif.npz", "protection.npz"}
                    or any(e.compress_type != zipfile.ZIP_STORED for e in entries)
                    or sum(e.file_size for e in entries) > 16_000_000
                    or archive.getinfo("metadata.json").file_size > 4096
                    or archive.getinfo("protection.npz").file_size > 100_000):
                raise ValueError("Unexpected prototype archive")
            meta = json.loads(archive.read("metadata.json"))
            base, extra = archive.read("local-alif.npz"), archive.read("protection.npz")
        if (set(meta) != {"format", "constants", "base_sha256", "extra_sha256"}
                or meta["format"] != FORMAT or meta["constants"] != CONSTANTS
                or hashlib.sha256(base).hexdigest() != meta["base_sha256"]
                or hashlib.sha256(extra).hexdigest() != meta["extra_sha256"]):
            raise ValueError("Incompatible algorithm/checksum")
        with zipfile.ZipFile(io.BytesIO(extra)) as archive:
            entries = archive.infolist()
            if (len(entries) != 2 or {e.filename for e in entries} != {
                    "output_importance.npy", "bias_importance.npy"}
                    or any(e.compress_type != zipfile.ZIP_STORED for e in entries)
                    or sum(e.file_size for e in entries) > 100_000):
                raise ValueError("Invalid extra importance archive")
        values = {}
        with np.load(io.BytesIO(extra), allow_pickle=False) as data:
            for name, shape in (("output_importance", (8, 128)), ("bias_importance", (8,))):
                value = data[name]
                if (value.dtype != np.float32 or value.shape != shape
                        or not np.isfinite(value).all() or np.any(value < 0)
                        or np.any(value > CONSTANTS["importance_cap"])):
                    raise ValueError("Invalid importance state")
                values[name] = value.copy()
        base_model, controller = LocalALIF.loads(base, device="cpu")
        if controller or np.any(base_model.importance > CONSTANTS["importance_cap"]):
            raise ValueError("Unexpected prototype/controller state")
        prototype = cls.__new__(cls)
        prototype.base = base_model.to_device(device)
        for name, value in values.items():
            setattr(prototype, name, prototype.base.xp.asarray(value))
        return prototype
