"""R7 round 27 — what is the IRREDUCIBLE forgetting floor? Solo vs joint per task.

THE QUESTION THAT DECIDES WHETHER THE GATE IS REACHABLE
-------------------------------------------------------
Everything so far measures forgetting as

    forgetting_j = R[j][j] - R[T-1][j]
                 = (solo accuracy) - (accuracy after all tasks are in the head)

That difference contains two separable parts:

    IRREDUCIBLE : solo_j - joint_j     the price of co-representing all three tasks
    EXCESS      : joint_j - final_j    what sequential updating actually destroys

Round 6 measured the joint ceiling at 0.8620 mean, and round 18 at 0.828-0.887 per seed. If
the irreducible part alone is at or above 3pp, then **no readout-side mechanism can ever meet
the mission's 3pp gate**, and interleaving is already near its limit. If it is well under 3pp,
there is room and the remaining gap is a mechanism problem.

This is the decisive decomposition and it has not been measured. Rounds 6 and 18 measured
joint accuracy but never compared it to SOLO accuracy on the same seeds.

DESIGN (15 seeds, same set as rounds 24/26 so results compose directly)
    solo_j    head trained on task j alone, 1600 epochs (the sequential per-task budget)
    joint     head trained on the UNION, 4800 epochs = 1600 x 3 so each task gets equal
              exposure and the total budget matches the sequential arms exactly
    final_j   taken from round 26's interleaved_tail R matrices (already stored)

    irreducible_j = solo_j - joint_j
    excess_j      = joint_j - final_j
    forgetting_j  = irreducible_j + excess_j     (identity, checked in code)

REPORTED
    per task and per seed, plus the loss-based aggregate the mission's gate means:
    mean(max(irreducible,0)) and mean(max(excess,0))

FALSIFIER / DISCRIMINATION
    If mean irreducible >= 3pp  -> the gate is UNREACHABLE by any readout-side mechanism on
       this suite. That is a measured bound on the protocol, not a mechanism failure.
    If mean irreducible is small but excess is large -> mechanism problem, keep working.
    If excess is already ~0 -> interleaving has reached the joint limit and further
       readout work cannot help.

No hidden-weight learning; the core stays frozen. Nothing is tuned on these numbers.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r7_secs as S  # noqa: E402

SEEDS = tuple(240001 + 7 * k for k in range(15))
ORDER = ("order_short", "parity3", "sim_xor")
N_TR, N_TE = 384, 256
EPOCHS = 800
SOLO_EPOCHS = 2 * EPOCHS          # 1600, the sequential per-task budget
JOINT_EPOCHS = 3 * SOLO_EPOCHS    # 4800, total budget matched to the sequential arms


def sim_xor(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        a = int(rng.integers(2)); b = int(rng.integers(2)); y[i] = a ^ b
        t = int(rng.integers(1, 4))
        x[i, t, 0] += 1.5 * a; x[i, t, 1] += 1.5 * b; x[i, 10, 7] += 1.5
    return x, y


def parity3(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        bits = [int(rng.integers(2)) for _ in range(3)]
        y[i] = bits[0] ^ bits[1] ^ bits[2]
        t = int(rng.integers(1, 4))
        for k, bb in enumerate(bits):
            x[i, t, k] += 1.5 * bb
        x[i, 10, 7] += 1.5
    return x, y


def order_short(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .08, (n, 12, 8)).astype(np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        y[i] = int(rng.integers(2))
        first, second = (4, 5) if y[i] == 0 else (5, 4)
        p = int(rng.integers(1, 4))
        x[i, p, first] += 1.5
        x[i, p + 2, second] += 1.5
        x[i, 10, 7] += 1.5
    return x, y


GENS = {"sim_xor": sim_xor, "parity3": parity3, "order_short": order_short}


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def feats(core, X):
    out = []
    for i in range(len(X)):
        core.reset_state()
        tot = np.zeros(core.config.hidden, np.float32)
        for t in range(12):
            core.step(X[i, t], track=False)
            tot += core.spikes
        out.append(tot)
    return np.stack(out)


class Head:
    __slots__ = ("mu", "sd", "w", "b")

    def __init__(self, d):
        self.mu = None; self.sd = None
        self.w = np.zeros((2, d)); self.b = np.zeros(2)

    def z(self, F):
        if self.mu is None:
            self.mu = F.mean(0); self.sd = np.maximum(F.std(0), 0.05)
        return (F - self.mu) / self.sd

    def step(self, A, y, lr=.5):
        p = softmax(A @ self.w.T + self.b)
        e = -p; e[np.arange(len(y)), y] += 1
        self.w += lr * (e.T @ A) / len(y)
        self.b += lr * e.mean(0)

    def acc(self, A, y):
        return float(np.mean(np.argmax(A @ self.w.T + self.b, 1) == y))


def main():
    t0 = time.time()
    print("=" * 100)
    print("R7 ROUND 27 — irreducible vs excess forgetting: is the 3pp gate reachable at all?")
    print("=" * 100)
    print(f"  solo budget {SOLO_EPOCHS}/task | joint budget {JOINT_EPOCHS} total "
          f"(= 3 x {SOLO_EPOCHS}, budget-matched)")
    print(f"  forgetting = irreducible + excess  (identity checked in code)\n")

    cells = []
    for k, seed in enumerate(SEEDS):
        core = S.SECSCore(S.cfg(seed), mode="scalar", seed=seed)
        Ftr, ytr, Fte, yte = {}, {}, {}, {}
        for tk in ORDER:
            trx, try_ = GENS[tk](seed + 200_000, N_TR)
            tex, tey = GENS[tk](seed + 300_000, N_TE)
            Ftr[tk] = feats(core, trx); ytr[tk] = try_
            Fte[tk] = feats(core, tex); yte[tk] = tey

        # ---- SOLO: one head per task, fit only on that task
        solo = {}
        for tk in ORDER:
            h = Head(Ftr[tk].shape[1])
            A = h.z(Ftr[tk]); y = ytr[tk]
            for _ in range(SOLO_EPOCHS):
                h.step(A, y)
            solo[tk] = h.acc(h.z(Fte[tk]), yte[tk])

        # ---- JOINT: one head on the union, budget-matched
        Fj = np.concatenate([Ftr[tk] for tk in ORDER])
        yj = np.concatenate([ytr[tk] for tk in ORDER])
        hj = Head(Fj.shape[1])
        Aj = hj.z(Fj)
        rng = np.random.default_rng(seed + 7)
        n = len(yj)
        half = max(1, n // 2)
        for _ in range(JOINT_EPOCHS):
            idx = rng.choice(n, half, replace=False)
            hj.step(Aj[idx], yj[idx])
        joint = {tk: hj.acc(hj.z(Fte[tk]), yte[tk]) for tk in ORDER}

        cells.append({"seed": seed, "solo": solo, "joint": joint})
        print(f"  {k+1:2d}/15 {seed}  solo "
              + " ".join(f"{solo[t]:.3f}" for t in ORDER)
              + "  joint " + " ".join(f"{joint[t]:.3f}" for t in ORDER), flush=True)

    # ---- attach round 26's interleaved_tail finals so the identity can be checked
    r26 = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7" \
        / "rmatrix-both-aggregations.json"
    finals = {}
    if r26.exists():
        d26 = json.loads(r26.read_text(encoding="utf-8"))
        for c in d26["cells"]:
            R = np.array(c["interleaved_tail"]["R"], float)
            finals[c["seed"]] = {tk: float(R[-1, j]) for j, tk in enumerate(ORDER)}

    print("\n" + "=" * 100)
    print("PER-SEED DECOMPOSITION (interleaved_tail as the achieved arm)")
    print("=" * 100)
    print(f"  {'seed':>8} {'task':>12} {'solo':>7} {'joint':>7} {'final':>7} "
          f"{'irred':>7} {'excess':>8} {'total':>7}")
    rows = []
    for c in cells:
        seed = c["seed"]
        for tk in ORDER:
            so, jo = c["solo"][tk], c["joint"][tk]
            fi = finals.get(seed, {}).get(tk)
            irr = (so - jo) * 100
            exc = None if fi is None else (jo - fi) * 100
            tot = None if fi is None else (so - fi) * 100
            rows.append({"seed": seed, "task": tk, "solo": so, "joint": jo,
                         "final": fi, "irreducible_pp": irr, "excess_pp": exc,
                         "total_pp": tot})
            ftxt = f"{fi:7.4f}" if fi is not None else f"{'--':>7}"
            etxt = f"{exc:+8.2f}" if exc is not None else f"{'--':>8}"
            ttxt = f"{tot:+7.2f}" if tot is not None else f"{'--':>7}"
            print(f"  {seed:>8} {tk:>12} {so:7.4f} {jo:7.4f} {ftxt} "
                  f"{irr:+7.2f} {etxt} {ttxt}")

    have = [r for r in rows if r["excess_pp"] is not None]
    print("\n" + "=" * 100)
    print("AGGREGATES — loss-based, the mission's reading")
    print("=" * 100)
    irr = np.array([r["irreducible_pp"] for r in rows])
    exc = np.array([r["excess_pp"] for r in have])
    tot = np.array([r["total_pp"] for r in have])
    for name, v in (("irreducible (solo-joint)", irr),
                    ("excess (joint-final)", exc),
                    ("total (solo-final)", tot)):
        print(f"  {name:26s} mean {v.mean():+7.2f}pp   "
              f"loss-based mean {np.mean(np.maximum(v, 0)):6.2f}pp   "
              f"max {v.max():+7.2f}   min {v.min():+7.2f}")

    ident = float(np.max(np.abs(tot - (irr[:len(tot)] + exc))))
    print(f"\n  identity check max|total - (irr+excess)| = {ident:.2e}  "
          f"{'OK' if ident < 1e-6 else 'MISMATCH'}")

    print("\n" + "=" * 100)
    print("THE GATE QUESTION")
    print("=" * 100)
    irr_loss = float(np.mean(np.maximum(irr, 0)))
    exc_loss = float(np.mean(np.maximum(exc, 0)))
    tot_loss = float(np.mean(np.maximum(tot, 0)))
    print(f"  loss-based irreducible forgetting : {irr_loss:6.2f}pp per task")
    print(f"  loss-based excess forgetting      : {exc_loss:6.2f}pp per task")
    print(f"  loss-based total (what we measure): {tot_loss:6.2f}pp per task")
    print(f"  mission mean-forgetting gate      :   3.00pp")

    if irr_loss >= 3.0:
        print(f"\n  => UNREACHABLE. The irreducible floor alone is {irr_loss:.2f}pp, at or")
        print(f"     above the 3pp gate. No readout-side mechanism can pass it.")
        verdict = "gate_unreachable_irreducible"
    elif exc_loss <= 1.0:
        print(f"\n  => SATURATED. The irreducible floor is {irr_loss:.2f}pp and the excess")
        print(f"     is only {exc_loss:.2f}pp. Interleaving has essentially reached the")
        print(f"     joint limit; further readout work cannot help.")
        verdict = "saturated"
    else:
        room = exc_loss
        print(f"\n  => ROOM REMAINS. Irreducible {irr_loss:.2f}pp + excess {exc_loss:.2f}pp.")
        print(f"     The excess ({room:.2f}pp) is what a better mechanism could still remove.")
        verdict = "room_remains"

    out = Path(__file__).resolve().parents[1] / "artifacts" / "deepseek-r7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "irreducible-floor.json").write_text(json.dumps({
        "seeds": list(SEEDS), "order": list(ORDER),
        "solo_epochs": SOLO_EPOCHS, "joint_epochs": JOINT_EPOCHS,
        "rows": rows, "verdict": verdict,
        "loss_based": {"irreducible_pp": irr_loss, "excess_pp": exc_loss,
                       "total_pp": tot_loss},
        "identity_max_abs_err": ident,
        "note": "excess uses round 26's interleaved_tail finals; identity "
                "forgetting = irreducible + excess holds by construction",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  written: irreducible-floor.json   wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
