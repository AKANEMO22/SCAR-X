"""Version-bound research shadow policy; no serving integration or new equation."""

import hashlib
import io
import json
import zipfile
import numpy as np
from scarx.hybrid.learner import LocalALIF
from .hybrid_incident_encoder import observation_sequence, FORMAT as ENCODER

FORMAT = "scarx-recovery-shadow-v1"
ACTION_ORDER = ("show_schema", "inspect_status", "retry", "clarify", "defer", "finish")
MAX_BYTES = 16_000_000


class RecoveryPolicy:
    def __init__(self, seed=260113, mode="frozen", device="gpu", gpu_device=0):
        self.model = LocalALIF(seed, mode).to_device(device, gpu_device)

    def propose(self, question, observations, *, decision_epoch, allowed=ACTION_ORDER):
        if (not isinstance(allowed, (list, tuple)) or not allowed
                or len(set(allowed)) != len(allowed)
                or any(v not in ACTION_ORDER for v in allowed)):
            raise ValueError("Expected explicit permitted research action set")
        sequence = observation_sequence(question, observations, decision_epoch=decision_epoch)
        probabilities = self.model.forward(sequence)[0]
        target = max((ACTION_ORDER.index(v) for v in allowed),
                     key=lambda index: float(probabilities[index]))
        return {"action": ACTION_ORDER[target],
                "probabilities": dict(zip(ACTION_ORDER, map(float, probabilities))),
                "allowed": list(allowed), "single_action_forced": len(allowed) == 1,
                "policy_qualified": False, "executed": False,
                "encoder": ENCODER, "input_shape": list(sequence.shape),
                "input_sha256": hashlib.sha256(sequence.tobytes()).hexdigest(),
                "model_updates": self.model.updates, "neural_device": self.model.device}

    def dumps(self):
        payload = self.model.dumps()
        metadata = {"format": FORMAT, "encoder": ENCODER,
                    "action_order": list(ACTION_ORDER),
                    "seed": self.model.seed, "mode": self.model.mode,
                    "payload_sha256": hashlib.sha256(payload).hexdigest()}
        raw = json.dumps(metadata, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode()
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in (("metadata.json", raw), ("local-alif.npz", payload)):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                archive.writestr(info, data)
        result = stream.getvalue()
        if len(result) > MAX_BYTES:
            raise ValueError("Recovery checkpoint exceeds budget")
        return result

    @classmethod
    def loads(cls, raw, device="gpu", gpu_device=0):
        if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
            raise ValueError("Invalid recovery checkpoint bytes")
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                if (len(entries) != 2 or {v.filename for v in entries} !=
                        {"metadata.json", "local-alif.npz"}
                        or any(v.compress_type != zipfile.ZIP_STORED for v in entries)
                        or sum(v.file_size for v in entries) > MAX_BYTES
                        or archive.getinfo("metadata.json").file_size > 4096):
                    raise ValueError("Unexpected recovery checkpoint archive")
                meta = json.loads(archive.read("metadata.json"))
                payload = archive.read("local-alif.npz")
        except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
            raise ValueError("Corrupt recovery checkpoint") from exc
        if (not isinstance(meta, dict)
                or set(meta) != {"format", "encoder", "action_order", "seed", "mode", "payload_sha256"}
                or meta["format"] != FORMAT or meta["encoder"] != ENCODER
                or meta["action_order"] != list(ACTION_ORDER)
                or type(meta["seed"]) is not int or meta["seed"] < 0
                or meta["mode"] not in ("learned", "readout_only", "frozen", "shuffled")
                or hashlib.sha256(payload).hexdigest() != meta["payload_sha256"]):
            raise ValueError("Incompatible recovery policy metadata")
        # Validate all outer metadata before creating CUDA arrays. Inner reader
        # already rejects duplicate/missing arrays and incompatible dtype/state.
        model, controller = LocalALIF.loads(payload, device="cpu")
        if controller or model.seed != meta["seed"] or model.mode != meta["mode"]:
            raise ValueError("Recovery outer/inner identity mismatch")
        instance = cls.__new__(cls)
        instance.model = model.to_device(device, gpu_device)
        return instance
