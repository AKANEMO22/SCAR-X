"""Isolated FP32 ALIF temporal-hint candidate; never imported by serving."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from scarx.neural.core import Config, SpikingCore

SEEDS = (261161, 261167, 261173)
TICKS = 32
COLS = (0, 1, 3, 4)
ARMS = ("label", "teacher", "shuffled", "readout")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cuda():
    """Reuse installed exact native DLL location, no installs/service changes."""
    import ctypes
    runtime = ROOT.parent / "scarx-hybrid-20261113/local/hybrid-v1/runtime/llama.cpp-b12360-cuda12.4"
    if not runtime.is_dir():
        raise FileNotFoundError(runtime)
    handles = [os.add_dll_directory(str(runtime))]
    handles += [ctypes.WinDLL(str(runtime / name)) for name in ("cublasLt64_12.dll", "cublas64_12.dll")]
    os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "local/p1b-20261116/cupy-cache"))
    import cupy as cp
    cp.cuda.Device(0).use()
    return cp, handles


def cases(seed, split, cycle=1):
    """No class target in input; target is ordered short A->B coincidence."""
    spec = {
        "qual": ((1, 2, 4, 6), (18, 22), 3, (.991, 1.009), 11000),
        "train": ((1, 2, 4, 5), (16, 20), 12, (.995, 1.005), 22000),
        "dev": ((1, 2, 5, 6), (18, 22), 8, (.993, 1.007), 33000),
        "test": ((1, 2, 6, 7), (17, 21, 23), 8, (.990, 1.010), 44000),
    }
    gaps, clocks, repetitions, amp_range, salt = spec[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, salt, cycle]))
    values = []
    e_col, k_col = (0, 1) if cycle == 1 else (3, 4)
    for order in (1, -1):
        for gap in gaps:
            for clock in clocks:
                for rep in range(repetitions):
                    amplitude = float(rng.uniform(*amp_range))
                    if split == "test":
                        amplitude = float(rng.uniform(.990, .992) if rep % 2 == 0 else rng.uniform(1.008, 1.010))
                    sequence = np.zeros((TICKS, 64), np.float32)
                    sequence[clock - gap, e_col] = sequence[clock, e_col] = amplitude
                    sequence[clock - gap, k_col] = order * amplitude
                    sequence[clock, k_col] = -order * amplitude
                    target_order = 1 if cycle == 1 else -1
                    label = int(order == target_order and gap <= 2)
                    values.append({"id": f"c{cycle}-{split}-{order}-{gap}-{clock}-{rep}",
                                   "sequence": sequence, "label": label, "gap": gap,
                                   "clock": clock, "order": order, "amplitude": amplitude,
                                   "split": split, "cycle": cycle})
    return [values[i] for i in rng.permutation(len(values))]


def old_cases():
    out = []
    for clock in (16, 20, 24):
        for amplitude in (.18, .22, .98, 1.02):
            seq = np.zeros((TICKS, 64), np.float32)
            seq[clock, 2] = amplitude
            out.append({"id": f"old-{clock}-{amplitude}", "sequence": seq,
                        "label": int(amplitude > .5)})
    return out


class ALIF:
    def __init__(self, seed, sign=1, kind="student", xp=np, cycle=1):
        self.seed, self.sign, self.kind, self.xp, self.cycle = seed, sign, kind, xp, cycle
        h = 128 if kind == "teacher" else 64
        core = SpikingCore(Config(inputs=64, hidden=h, outputs=2, fan_in=8, plastic_rows=8, seed=seed))
        rng = np.random.default_rng(seed + 500)
        for row in range(8):
            core.indices[row] = np.arange(8)
        if h == 128:
            for row in range(8, 64):
                core.indices[row] = rng.choice(64, 8, replace=False)
        core.w_in.fill(0);core.w_rec.fill(0);core.w_out.fill(0);core.bias.fill(0)
        core.w_in[:8, 2] = .9
        start = 64 if kind == "teacher" else 8
        if kind in ("teacher", "witness"):
            e_col, k_col = (0, 1) if cycle == 1 else (3, 4)
            polarity = -1 if cycle == 1 else 1
            core.w_in[start:start+28, e_col] = .358
            core.w_in[start:start+28, k_col] = polarity * .158
            core.w_in[start+28:start+42, e_col] = .45
            core.w_in[start+28:start+42, k_col] = .45
            core.w_in[start+42:start+56, e_col] = .45
            core.w_in[start+42:start+56, k_col] = -.45
            core.w_out[1, start:start+28] = 3 / 28
        else:
            core.w_in[8:64, 0] = rng.uniform(.118, .122, 56)
            core.w_in[8:64, 1] = rng.uniform(-.002, .002, 56)
            core.w_out[1, 8:36] = sign * 3 / 28
        core.bias[1] = -.32
        for name in ("in", "rec", "out"):
            getattr(core, "stable_"+name)[:] = getattr(core, "w_"+name)
        core.stable_bias[:] = core.bias
        if xp is not np:
            from scarx.hybrid.cuda_alif import CudaSpikingCore
            core = CudaSpikingCore(core, xp)
        self.core = core
        self.ev = xp.zeros((h, 4), xp.float32);self.ea = self.ev.copy();self.ef = self.ev.copy()
        self.mom = self.ev.copy();self.energy = xp.full((h, 4), 1e-4, xp.float32)
        self.old_W = xp.zeros((2, h), xp.float32);self.old_W[1, :8] = 3/8
        self.old_bias = xp.asarray([0., -.32], xp.float32)
        self.updates = self.hidden_updates = self.forwards = 0
        self.rng = np.random.default_rng(seed + 900)

    def host(self, value):
        return self.xp.asnumpy(value) if self.xp is not np else np.asarray(value)

    def forward(self, sequence, track=False, hint=None, old=False, tape=False):
        xp, c = self.xp, self.core.config
        seq = xp.asarray(sequence, xp.float32)
        targets = xp.asarray(hint, xp.float32) if hint is not None else xp.zeros((TICKS, 56), xp.float32)
        self.core.reset_state();self.ev.fill(0);self.ea.fill(0);self.ef.fill(0)
        bank = xp.zeros_like(self.ef);history = [];spikes = []
        for t, row in enumerate(seq):
            if track:
                theta = c.threshold + c.adaptation_strength * self.core.adaptation
                u = c.leak*self.core.voltage + self.core.w_in@row + xp.sum(self.core.w_rec*self.core.spikes[self.core.indices], axis=1)
                raw_slope = (u-theta)/.4
                prob = 1/(1+xp.exp(-xp.clip(raw_slope, -20, 20)))
                psi = prob*(1-prob)/.4*(xp.abs(raw_slope)<20)
                ue = c.leak*self.ev + row[xp.asarray(COLS)][None, :]
                a_old = self.ea.copy()
                ze = psi[:, None]*(ue-c.adaptation_strength*a_old)
            self.core.step(row, track=False)
            if xp is not np:
                self.core.probabilities()  # Match CPU core.step's unused output computation.
            q = xp.clip(self.core.features/.1, -5, 5)
            if track:
                self.ev[:] = ue-theta[:, None]*ze-self.core.spikes[:, None]*c.adaptation_strength*a_old
                self.ea[:] = c.adaptation_decay*a_old+ze
                self.ef[:] = c.feature_decay*self.ef+(1-c.feature_decay)*ze
                modulation = xp.zeros(c.hidden, xp.float32)
                modulation[8:64] = (targets[t] - q[8:64])/.1*(xp.abs(self.core.features[8:64]/.1)<5)
                bank += modulation[:, None]*self.ef/TICKS
            if tape:
                history.append(self.host(q).copy());spikes.append(self.host(self.core.spikes).copy())
        weights, bias = (self.old_W, self.old_bias) if old else (self.core.w_out, self.core.bias)
        logits = weights@q+bias;logits -= logits.max()
        probability = xp.exp(logits);probability /= probability.sum()
        self.forwards += 1
        return self.host(probability), bank, self.host(q), history, spikes

    def proposal(self, label, probability, hint_bank, arm):
        xp = self.xp
        error = -xp.asarray(probability, xp.float32);error[label] += 1
        raw = self.core.features/.1
        signal = (self.core.w_out.T@error)*(xp.abs(raw)<5)/.1
        direction = signal[:, None]*self.ef
        if arm in ("teacher", "shuffled"):
            direction += hint_bank
        direction[:8] = 0
        moment = .9*self.mom+.1*direction
        energy = .99*self.energy+.01*direction**2
        step = xp.clip(.008*moment/(xp.sqrt(energy)+1e-4), -.01, .01)
        before = self.core.w_in[:, xp.asarray(COLS)]
        after = xp.clip(before+step, -3, 3)
        after[:8] = before[:8]
        return direction, moment, energy, after-before, error

    def learn(self, sequence, label, hint, arm):
        xp = self.xp
        p, bank, q, _, _ = self.forward(sequence, track=True, hint=hint)
        direction, moment, energy, delta, error = self.proposal(label, p, bank, arm)
        if arm != "readout":
            self.mom[:] = moment;self.energy[:] = energy
            self.core.w_in[:, xp.asarray(COLS)] += delta
            self.hidden_updates += 1
        W_delta = xp.clip(.04*error[:, None]*xp.asarray(q)[None, :], -.025, .025)
        W_delta[:, :8] = 0
        self.core.w_out += W_delta;xp.clip(self.core.w_out, -3, 3, out=self.core.w_out)
        self.core.bias += xp.clip(.04*error, -.025, .025)
        xp.clip(self.core.bias, -3, 3, out=self.core.bias)
        self.updates += 1;self.core.updates += 1
        return {"loss": -float(np.log(max(float(p[label]), 1e-12))),
                "before_correct": int(np.argmax(p)) == label,
                "hidden_delta_L2": float(xp.linalg.norm(delta)) if arm != "readout" else 0.,
                "direction_L2": float(xp.linalg.norm(direction)),
                "hint_bank_L2": float(xp.linalg.norm(bank)), "updates": self.updates,
                "direction_array": self.host(direction).copy(), "hint_bank_array": self.host(bank).copy(),
                "probabilities_before": p.copy()}

    def dumps(self):
        arrays = {"core_"+k: self.host(v) for k, v in self.core.arrays().items()}
        arrays.update({k:self.host(getattr(self,k)) for k in ("ev","ea","ef","mom","energy","old_W","old_bias")})
        meta = {"format":"scope-p1b-v1","seed":self.seed,"sign":self.sign,"kind":self.kind,"cycle":self.cycle,
                "config":asdict(self.core.config),"updates":self.updates,"hidden_updates":self.hidden_updates,
                "forwards":self.forwards,"rng":self.rng.bit_generator.state}
        arrays["metadata"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        out=io.BytesIO();np.savez(out,**arrays);return out.getvalue()

    @classmethod
    def loads(cls, raw, xp=np):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)) or sum(i.file_size for i in archive.infolist())>4_000_000:
                raise ValueError("Checkpoint bounds/duplicates")
        with np.load(io.BytesIO(raw),allow_pickle=False) as data:
            meta=json.loads(data["metadata"].tobytes())
            if meta["format"]!="scope-p1b-v1":raise ValueError("format")
            obj=cls(meta["seed"],meta["sign"],meta["kind"],xp,meta["cycle"])
            if asdict(obj.core.config)!=meta["config"]:raise ValueError("config")
            expected={"core_"+k for k in obj.core.arrays()}|{"ev","ea","ef","mom","energy","old_W","old_bias","metadata"}
            if set(data.files)!=expected:raise ValueError("members")
            for name in expected-{"metadata"}:
                target=getattr(obj.core,name[5:]) if name.startswith("core_") else getattr(obj,name)
                value=data[name]
                if value.shape!=target.shape or value.dtype!=target.dtype or not np.isfinite(value).all():raise ValueError("shape/dtype")
                target[:]=xp.asarray(value)
            for key in ("updates","hidden_updates","forwards"):setattr(obj,key,meta[key])
            obj.core.updates=obj.updates;obj.rng.bit_generator.state=meta["rng"]
            return obj
