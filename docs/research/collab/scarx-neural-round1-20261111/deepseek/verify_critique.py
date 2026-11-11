"""Verify the reviewer's corrections against my own published artifacts.

Claims under test:
V1. My published script does NOT print 0.0299/0.1931 or 0.0264/0.1463.
V2. Decay 0.25 is not the 0.96 row: decay0.25 ~0.69-0.76 on XOR, spike-sum 0.96.
V3. parity3: decay 0.25 ~0.58 (below 65), spike-sum ~0.83-0.88, bank5 ~0.91.
V4. On the hard path the ridge probe reads 0.69-0.72 but the ONLINE head
    ('direct') sits at ~0.50 while training - the head never learns.
V5. Mean training loss sits near log(2)=0.693, so advantage = -loss - baseline
    is near zero throughout: the scalar rule is starved, not merely wrong.
V6. On sim_xor the firing rates are ~0.06 hard vs ~0.21 Bernoulli (3.4x), not
    the 6.5x I reported from a different stimulus.
"""
import re
from pathlib import Path

import numpy as np

from scarx.neural.core import Config

HERE = Path(__file__).resolve().parent
CFG = dict(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12,
           leak=.65, threshold=.5, feature_decay=.25, adaptation_strength=.02)
DECAYS = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
S = 1.5


def cfg(seed=41001, **kw):
    return Config(seed=seed, **{**CFG, **kw})


def sim_xor(seed, count):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = rng.integers(0, 2, count).astype(np.int32)
    for i in range(count):
        a = int(rng.integers(2)); b = a ^ int(y[i])
        t = int(rng.integers(1, 4))
        x[i, t, 0] += S * a
        x[i, t, 1] += S * b
        x[i, 10, 7] += S
    return x, y


def parity3(seed, count):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (count, 12, 8)).astype(np.float32)
    y = np.zeros(count, np.int32)
    for i in range(count):
        bits = [int(rng.integers(2)) for _ in range(3)]
        t = int(rng.integers(1, 4))
        for c, b in enumerate(bits):
            x[i, t, c] += S * b
        x[i, 10, 7] += S
        y[i] = bits[0] ^ bits[1] ^ bits[2]
    return x, y


def weights(seed):
    c = cfg(seed)
    g = np.random.default_rng(c.seed)
    idx = np.stack([g.choice(c.hidden, c.fan_in, replace=False) for _ in range(c.hidden)]).astype(np.int32)
    w_in = g.normal(.15, .45, (c.hidden, c.inputs)).astype(np.float32)
    w_rec = g.normal(0, .12 / np.sqrt(c.fan_in), (c.hidden, c.fan_in)).astype(np.float32)
    return c, idx, w_in, w_rec


def train(seq, mode, seed=901, w=None):
    c, idx, w_in, w_rec = w or weights(seed)
    x = np.asarray(seq, np.float32)
    b, t, _ = x.shape
    v = np.zeros((b, c.hidden), np.float32); a = v.copy(); z = v.copy()
    rng = np.random.default_rng(901)
    out, rates = [], []
    for s in range(t):
        v = c.leak * v + x[:, s] @ w_in.T + np.sum(w_rec[None, :, :] * z[:, idx], axis=2)
        lim = c.threshold + c.adaptation_strength * a
        p = 1 / (1 + np.exp(-np.clip((v - lim) / .4, -20, 20)))
        z = (v >= lim).astype(np.float32) if mode == "hard" else \
            (rng.random((b, c.hidden)) < p).astype(np.float32)
        v = v - z * lim
        a = c.adaptation_decay * a + z
        out.append(z.copy()); rates.append(z.mean())
    return np.stack(out, 1), float(np.mean(rates))


def integ(tr, d):
    n, t, h = tr.shape
    f = np.zeros((n, h), np.float32)
    for s in range(t):
        f = d * f + (1 - d) * tr[:, s]
    return f


def ridge(train_f, train_y, test_f, test_y):
    """Same estimator as tasks21.ridge_probe: z-scored features, ridge 0.01."""
    mean = train_f.mean(0); scale = train_f.std(0) + 1e-4
    A = np.column_stack(((train_f - mean) / scale, np.ones(len(train_f))))
    B = np.column_stack(((test_f - mean) / scale, np.ones(len(test_f))))
    tgt = np.eye(2)[train_y]
    w = np.linalg.solve(A.T @ A + .01 * np.eye(A.shape[1]), A.T @ tgt)
    return float(np.mean(np.argmax(B @ w, 1) == test_y))


# ------------------------------------------------------------------ V1
def v1():
    print("=" * 76)
    print("V1  does my published sampler_diagnosis.py print 0.0299 / 0.1931?")
    print("=" * 76)
    src = (HERE / "sampler_diagnosis.py").read_text(encoding="utf-8")
    for needle in ("0.0299", "0.1931", "0.0264", "0.1463", "6.5"):
        print(f"  the string {needle!r:10s} appears in the script: {needle in src}")
    printed = re.findall(r'f"[^"]*\{(\w+)[^"]*"', src)
    print(f"  the script prints mean logit / p / |logit|<1 / p>0.95 / p<0.05 only.")
    print("  -> it computes RATE and VARIANCE nowhere.")
    print("  VERDICT: the reviewer is right. Those four numbers are not reproducible")
    print("  from the artifact I published.")

    # recompute them the way the reviewer did: on the diagnostic stimulus
    c, idx, w_in, w_rec = weights(41001)
    rng = np.random.default_rng(7)
    x = rng.normal(0, .08, (300, 12, 8)).astype(np.float32)
    for i in range(300):
        x[i, 2, int(rng.integers(3))] += 1.5
    for mode in ("hard", "bernoulli"):
        _, rate = train(x, mode, w=(c, idx, w_in, w_rec))
        print(f"  diagnostic stimulus, {mode:10s} rate = {rate:.4f}")


# --------------------------------------------------------------- V2/V3
def v2v3():
    print("\n" + "=" * 76)
    print("V2/V3  which readout row is decay 0.25?")
    print("=" * 76)
    for name, gen in (("sim_xor", sim_xor), ("parity3", parity3)):
        print(f"\n  {name}")
        print(f"  {'seed':>9} {'decay0.25':>10} {'spike_sum':>10} {'bank5':>8} {'decay0.95':>10}")
        for gs in (31001, 31009):
            fx, fy = gen(gs, 400)
            tx, ty = gen(gs + 900, 400)
            core, idx, w_in, w_rec = weights(41001)
            tr = train(fx, "hard", w=(core, idx, w_in, w_rec))[0]
            tt = train(tx, "hard", w=(core, idx, w_in, w_rec))[0]
            d25 = ridge(integ(tr, .25), fy, integ(tt, .25), ty)
            ssum = ridge(tr.sum(1), fy, tt.sum(1), ty)
            bank = ridge(np.concatenate([integ(tr, d) for d in (.1, .25, .5, .75, .9)], 1),
                         fy,
                         np.concatenate([integ(tt, d) for d in (.1, .25, .5, .75, .9)], 1), ty)
            d95 = ridge(integ(tr, .95), fy, integ(tt, .95), ty)
            print(f"  {gs:>9} {d25:10.4f} {ssum:10.4f} {bank:8.4f} {d95:10.4f}")


# ------------------------------------------------------------------ V6
def v6():
    print("\n" + "=" * 76)
    print("V6  firing rate on sim_xor itself (not the diagnostic stimulus)")
    print("=" * 76)
    core, idx, w_in, w_rec = weights(41001)
    fx, _ = sim_xor(31001, 400)
    for mode in ("hard", "bernoulli"):
        _, rate = train(fx, mode, w=(core, idx, w_in, w_rec))
        print(f"  {mode:10s} rate = {rate:.4f}")


if __name__ == "__main__":
    v1(); v2v3(); v6()
