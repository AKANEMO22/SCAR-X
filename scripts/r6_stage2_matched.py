"""Round 6A Stage 2 — the matched-update contrast: unique data vs repetition.

THE decisive experiment of the round.

Hold total updates fixed at U = 16384. Vary how many of them are DISTINCT:

    arm             n_unique   epochs   total updates
    repeated-256       256        64        16384
    repeated-1024     1024        16        16384
    repeated-4096     4096         4        16384
    unique-16384     16384         1        16384

If unique-16384 does not beat repeated-256 at equal updates, the scaling gain in
Stage 1 is repetition, not new knowledge — and the brief says that must not be called
learning more.

Also runs a pure repetition arm at n=1024 with increasing epochs, to test H2 alone
(more updates, no new data) and to look for a plateau.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r6_scaling as S  # noqa: E402
from scarx.evaluation.runs import new_run, write_json  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

QT = G.QUERY_TICK
N_HELD, N_PAIRS, CAL_SIZE = 1024, 256, 256
U = 16384                       # the matched update budget


def build(seed, n_unique):
    trx, try_, tvA, tvB, tq, _ = G.independent_query_set_v1(seed + S.TRAIN_OFFSET, n_unique)
    tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + S.HELD_OFFSET, N_HELD)
    ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(
        seed + S.PAIRED_OFFSET, N_PAIRS)
    f_tr = S.capture_features(seed, trx, QT)
    f_te = S.capture_features(seed, tex, QT)
    f_pt = S.capture_features(seed, ptx, QT)
    cal = f_tr[:CAL_SIZE]
    mu, sd = cal.mean(0), np.maximum(cal.std(0), 1e-6)
    f = lambda a: (a - mu) / sd           # noqa: E731
    return dict(f_tr=f(f_tr), f_te=f(f_te), f_pt=f(f_pt),
                y_tr=try_, y_te=tey, vA=evA, vB=evB, q=eq,
                pty=pty, pvA=pvA, pvB=pvB, pflips=flips,
                digests={"train": G.dataset_sha256(trx, try_) if n_unique >= 256 else None,
                         "held": G.dataset_sha256(tex, tey)})


def run():
    root = Path(__file__).resolve().parents[1]
    proto = root / "experiments/collab/scarx-neural-round1-20261111/deepseek-r6/protocol-v1.md"
    psha = hashlib.sha256(proto.read_bytes()).hexdigest() if proto.exists() else "PROTOCOL_PENDING"
    run = new_run(root, "r6-matched-update-v1", {
        "protocol_sha256": psha, "seeds": list(S.DEV_SEEDS),
        "matched_updates": U,
        "arms": ["repeated-256", "repeated-1024", "repeated-4096", "unique-16384"],
        "epoch_scan": {"n_unique": 1024, "epochs": [1, 2, 4, 8, 16]}})

    rows = []
    t0 = time.time()
    with ResourceMonitor(interval=.1) as mon:
        for seed in S.DEV_SEEDS:
            rng = np.random.default_rng(seed + 909)

            # ---- matched-update arms, all at U = 16384 updates
            for n in (256, 1024, 4096, 16384):
                d = build(seed, n)
                epochs = U // n
                assert epochs * n == U
                order = rng.permutation(n)
                h = S.train_head(d["f_tr"], d["y_tr"], seed, order, epochs=epochs)
                from scarx.evaluation.tasks21 import ridge_probe
                ridge_acc = ridge_probe(d["f_tr"], d["y_tr"], d["f_te"], d["y_te"])
                rep = S.full_report(h, d["f_te"], d["y_te"], d["vA"], d["vB"], d["q"],
                                    d["f_pt"], d["pty"], d["pvA"], d["pvB"], d["pflips"],
                                    d["f_tr"], d["y_tr"], ridge_acc)
                rep.update({"seed": seed, "arm": f"repeated-{n}", "n_unique": n,
                            "epochs": epochs, "total_updates": epochs * n,
                            "condition": "repeated" if n < 16384 else "unique"})
                rows.append(rep)
                mon.check()
                print(f"  seed{seed} repeated-{n:<6d} ep{epochs:3d} "
                      f"direct {rep['direct']:.4f} bothA {rep['both_A']:.4f}", flush=True)

            # ---- epoch scan at fixed n = 1024 (H2 in isolation)
            d = build(seed, 1024)
            base_order = rng.permutation(1024)
            for ep in (1, 2, 4, 8, 16):
                h = S.train_head(d["f_tr"], d["y_tr"], seed, base_order, epochs=ep)
                from scarx.evaluation.tasks21 import ridge_probe
                ridge_acc = ridge_probe(d["f_tr"], d["y_tr"], d["f_te"], d["y_te"])
                rep = S.full_report(h, d["f_te"], d["y_te"], d["vA"], d["vB"], d["q"],
                                    d["f_pt"], d["pty"], d["pvA"], d["pvB"], d["pflips"],
                                    d["f_tr"], d["y_tr"], ridge_acc)
                rep.update({"seed": seed, "arm": f"epochs-{ep}", "n_unique": 1024,
                            "epochs": ep, "total_updates": ep * 1024,
                            "condition": "epoch_scan"})
                rows.append(rep)
                mon.check()
            print(f"  seed{seed} epoch scan done", flush=True)

    wall = time.time() - t0
    write_json(run / "result.json",
               {"protocol_sha256": psha, "phase": "stage2_matched_update",
                "notes": {"matched": f"every arm performs exactly {U} updates",
                          "unique_arm": "16384 distinct episodes, 1 epoch"},
                "trials": rows, "resources": mon.report(), "wall_seconds": wall})

    print("\n" + "=" * 92)
    print(f"MATCHED-UPDATE CONTRAST — all arms perform exactly {U} updates")
    print("=" * 92)
    print(f"  {'arm':16s} {'epochs':>7} {'direct':>8} {'train':>8} {'both_A':>8} "
          f"{'needB':>8} {'ridge':>8} {'clip%':>7}")
    for n in (256, 1024, 4096, 16384):
        sub = [r for r in rows if r["arm"] == f"repeated-{n}"]
        m = lambda k: float(np.nanmean([r[k] for r in sub]))   # noqa: E731
        print(f"  {'repeated-'+str(n):16s} {sub[0]['epochs']:>7} {m('direct'):8.4f} "
              f"{m('train'):8.4f} {m('both_A'):8.4f} {m('acc_needB'):8.4f} "
              f"{m('ridge'):8.4f} {m('clip_fraction')*100:7.2f}")

    print("\n" + "=" * 92)
    print("EPOCH SCAN at n=1024 — more updates, NO new data (H2 in isolation)")
    print("=" * 92)
    print(f"  {'epochs':>7} {'updates':>9} {'direct':>8} {'train':>8} {'both_A':>8} "
          f"{'gap':>8}")
    for ep in (1, 2, 4, 8, 16):
        sub = [r for r in rows if r["arm"] == f"epochs-{ep}"]
        m = lambda k: float(np.nanmean([r[k] for r in sub]))   # noqa: E731
        print(f"  {ep:>7} {ep*1024:>9} {m('direct'):8.4f} {m('train'):8.4f} "
              f"{m('both_A'):8.4f} {m('ridge_gap'):8.4f}")

    print(f"\n  wall {wall:.1f}s | peak commit "
          f"{mon.report()['peak_commit_or_vms_bytes']:,} B")
    print(f"RESULT {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
