"""SCAR-X Round 9b research-model snapshots.

Explicit caller-supplied model template only, never import classes from payload.
Supports FactorialMTSGEProp and R9bBFCascade models.
Enforces:
1. Exact schema, bounds, and SHA-256 checksums per array and metadata.
2. Data-only format: zip containing NPY array files and JSON metadata (no pickle).
3. Max byte bound 2MB.
4. Pre-allocation verification of array header, dtype, and shapes.
5. Atomic save with replacement and clean rollback.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np

FORMAT = "scarx-research-r9b-snapshot-v1"
MAX_BYTES = 2_000_000

MTSG_ARRAYS = {
    "e_v", "e_a", "e_f", "syn_trace", "syn_energy",
    "f_mean", "f_var", "w_in_0", "w_rec_0", "gamma_e_vec",
}
MTSG_SETTINGS = {
    "decay_mode", "gating_mode", "reset_mode", "control_mode",
    "sensory_threshold", "lr_hid", "clip_hid", "lr_out", "clip_out",
    "calibrated", "episodes_learned", "hidden_updates", "total_ticks",
}

BF_ARRAYS = {
    "u1_in", "u2_in", "u1_rec", "u2_rec", "u1_out", "u2_out",
    "bias", "w_in_0", "w_rec_0", "e_v", "e_a", "e_f",
    "syn_trace", "syn_energy", "f_mean", "f_var", "gamma_e_vec",
}
BF_SETTINGS = {
    "consolidation_mode", "kappa1", "kappa2", "sensory_threshold",
    "calibrated", "episodes_learned", "hidden_updates",
}


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_controller(value, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError("Controller nesting bound exceeded")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for child in value:
            validate_controller(child, depth + 1)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for child in value.values():
            validate_controller(child, depth + 1)
        return
    raise ValueError("Controller must use lossless JSON types with string keys")


def metadata_loads(raw: bytes) -> dict:
    if len(raw) > 65536:
        raise ValueError("Metadata byte bound exceeded")

    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate JSON key in metadata")
            result[k] = v
        return result

    val = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if not isinstance(val, dict):
        raise ValueError("Metadata object required")
    return val


def layout(model) -> tuple[str, set[str], dict[str, np.ndarray]]:
    name = type(model).__name__
    if name == "FactorialMTSGEProp":
        arrays = MTSG_ARRAYS
        settings = MTSG_SETTINGS
    elif name == "R9bBFCascade":
        arrays = BF_ARRAYS
        settings = BF_SETTINGS
    else:
        raise ValueError(f"Unsupported model type for R9b snapshot: {name}")

    c = model.config
    if (c.hidden, c.inputs, c.outputs, c.fan_in, c.plastic_rows, c.neuron) != (48, 8, 2, 8, 12, "alif"):
        raise ValueError("Snapshot scope strictly H48 D8 O2 K8 R12 ALIF only")

    all_arrays = {"core__" + k: v for k, v in model.core.arrays().items()}
    all_arrays.update({"model__" + k: getattr(model, k) for k in arrays})
    return name, settings, all_arrays


def validate_settings(kind: str, settings: dict, core_settings: dict) -> None:
    expected = MTSG_SETTINGS if kind == "FactorialMTSGEProp" else BF_SETTINGS
    if set(settings) != expected:
        raise ValueError(f"Settings schema mismatch for {kind}")
    for k, v in settings.items():
        if k == "calibrated":
            if type(v) is not bool:
                raise ValueError("Boolean setting required for calibrated")
        elif k in ("episodes_learned", "hidden_updates", "total_ticks"):
            if type(v) is not int or not 0 <= v < 2**63:
                raise ValueError(f"Bounded integer required for {k}")
        elif k in ("sensory_threshold", "lr_hid", "clip_hid", "lr_out", "clip_out", "kappa1", "kappa2"):
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100:
                raise ValueError(f"Finite bounded float required for {k}")
        elif k == "decay_mode" and v not in ("uniform_fast", "uniform_mid", "uniform_slow", "heterogeneous"):
            raise ValueError(f"Unknown decay_mode {v}")
        elif k == "gating_mode" and v not in ("none", "recurrent_only", "soft_sensory"):
            raise ValueError(f"Unknown gating_mode {v}")
        elif k == "reset_mode" and v not in ("own_reset", "stopped_reset"):
            raise ValueError(f"Unknown reset_mode {v}")
        elif k == "control_mode" and v not in ("learned", "shuffled_signal", "zero_signal", "frozen_hidden", "readout_only"):
            raise ValueError(f"Unknown control_mode {v}")
        elif k == "consolidation_mode" and v not in ("none", "full", "readout_only", "hidden_only", "selective"):
            raise ValueError(f"Unknown consolidation_mode {v}")

    if set(core_settings) != {"updates", "reward_mean"}:
        raise ValueError("Invalid core_settings schema")
    if type(core_settings["updates"]) is not int or not 0 <= core_settings["updates"] < 2**63:
        raise ValueError("Core updates bound")
    if type(core_settings["reward_mean"]) not in (int, float) or not math.isfinite(core_settings["reward_mean"]):
        raise ValueError("Core reward_mean nonfinite")


def dumps(model, controller=None) -> bytes:
    kind, names, arrays = layout(model)
    settings = {k: getattr(model, k) for k in names}
    core_settings = {"updates": model.core.updates, "reward_mean": model.core.reward_mean}
    validate_settings(kind, settings, core_settings)
    validate_controller(controller)
    if controller is not None and (not isinstance(controller, dict) or len(encoded(controller)) > 16384):
        raise ValueError("Bounded controller object required")

    members = {}
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype not in (np.dtype("float32"), np.dtype("int32")) or not np.isfinite(array).all():
            raise ValueError(f"Invalid snapshot array: {name}")
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        members[name + ".npy"] = stream.getvalue()

    rng = model.signal_rng.bit_generator.state if hasattr(model, "signal_rng") else None
    meta = {
        "format": FORMAT,
        "kind": kind,
        "config": asdict(model.config),
        "settings": settings,
        "core_settings": core_settings,
        "rng": rng,
        "controller": controller,
        "sha256": {k: digest(raw) for k, raw in members.items()},
    }
    meta["metadata_sha256"] = digest(encoded(meta))
    members["metadata.json"] = encoded(meta)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(zipfile.ZipInfo(name), raw)

    payload = out.getvalue()
    if len(payload) > MAX_BYTES:
        raise ValueError("Snapshot payload exceeded max bound")
    return payload


def loads(template, payload: bytes) -> tuple:
    kind, _, expected = layout(template)
    if not isinstance(payload, bytes) or len(payload) > MAX_BYTES:
        raise ValueError("Snapshot size bound")

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            fields = {k + ".npy" for k in expected} | {"metadata.json"}
            if len(names) != len(fields) or set(names) != fields or sum(i.file_size for i in archive.infolist()) > MAX_BYTES:
                raise ValueError("Snapshot member/bounds mismatch")
            if archive.getinfo("metadata.json").file_size > 65536:
                raise ValueError("Metadata byte bound")

            meta = metadata_loads(archive.read("metadata.json"))
            check = meta.pop("metadata_sha256")
            if set(meta) != {"format", "kind", "config", "settings", "core_settings", "rng", "controller", "sha256"} or digest(encoded(meta)) != check:
                raise ValueError("Metadata checksum mismatch")
            if meta["format"] != FORMAT or meta["kind"] != kind or meta["config"] != asdict(template.config):
                raise ValueError("Template/kind/config mismatch")

            validate_settings(kind, meta["settings"], meta["core_settings"])
            validate_controller(meta["controller"])

            if set(meta["sha256"]) != {k + ".npy" for k in expected}:
                raise ValueError("Array checksum fields mismatch")

            # Check declared header before np.load allocation
            for name, prior in expected.items():
                fn = name + ".npy"
                with archive.open(fn) as stream:
                    ver = np.lib.format.read_magic(stream)
                    if ver == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=8192)
                    elif ver == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=8192)
                    else:
                        raise ValueError("Unsupported NPY header")
                    end = stream.tell()
                if shape != prior.shape or dtype != prior.dtype or fortran or dtype.hasobject:
                    raise ValueError("Declared array shape/dtype rejected")
                if end + math.prod(shape) * dtype.itemsize != archive.getinfo(fn).file_size:
                    raise ValueError("NPY body/header size mismatch")

            arrays = {}
            for name in expected:
                raw = archive.read(name + ".npy")
                if digest(raw) != meta["sha256"][name + ".npy"]:
                    raise ValueError("Array checksum mismatch")
                arr = np.load(io.BytesIO(raw), allow_pickle=False)
                if not np.isfinite(arr).all():
                    raise ValueError("Non-finite array in snapshot")
                arrays[name] = arr

        result = copy.deepcopy(template)
        if hasattr(result, "signal_rng"):
            if not isinstance(meta["rng"], dict) or meta["rng"].get("bit_generator") != "PCG64":
                raise ValueError("PCG64 RNG state required")
            result.signal_rng.bit_generator.state = meta["rng"]

        for name, arr in arrays.items():
            scope, key = name.split("__", 1)
            target_obj = result.core if scope == "core" else result
            getattr(target_obj, key)[:] = arr

        for k, v in meta["settings"].items():
            setattr(result, k, v)
        for k, v in meta["core_settings"].items():
            setattr(result.core, k, v)

        if kind == "R9bBFCascade":
            result.sync_active_weights_to_core()

        return result, copy.deepcopy(meta["controller"])
    except (KeyError, TypeError, OverflowError, zipfile.BadZipFile, UnicodeError) as exc:
        raise ValueError("Malformed R9b research snapshot") from exc


def save(model, path, controller=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dumps(model, controller)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load(template, path) -> tuple:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Snapshot file exceeds bound")
    return loads(template, path.read_bytes())
