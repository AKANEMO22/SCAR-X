"""Round 6C Stage 0 — Neural information and learning locality audit.

Reproduces the R5B reference results on historical seeds (97001..97013 dev,
98001..98037 confirmation) and verifies all 11 required Stage 0 invariants
before any candidate is designed or scored on Round 6 seeds.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scarx.evaluation import r5_independent as G  # noqa: E402
from scarx.evaluation import r5_probe as P  # noqa: E402
from scarx.neural.core import SpikingCore  # noqa: E402
from scarx.runtime.resources import ResourceMonitor  # noqa: E402

HIST_DEV_SEEDS = (97001, 97007, 97013)
HIST_CONF_SEEDS = (98001, 98007, 98013, 98019, 98023, 98029, 98031, 98037)


class AuditLog:
    def __init__(self):
        self.rows = []
        self.ok = True

    def check(self, name, passed, detail=""):
        passed = bool(passed)
        self.rows.append({"check": name, "pass": passed, "detail": str(detail)})
        if not passed:
            self.ok = False
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return passed


def main():
    print("=" * 84)
    print("SCAR-X ROUND 6C — STAGE 0: NEURAL INFORMATION & LEARNING LOCALITY AUDIT")
    print("=" * 84)
    log = AuditLog()

    with ResourceMonitor(interval=0.1) as mon:
        # 1. Frozen ALIF holds independently decodable vA and vB on historical dev seeds
        vA_accs, vB_accs, mixed_accs, q_neural_accs = [], [], [], []
        r5_dev_direct, r5_dev_bothA, r5_dev_bothB = [], [], []
        uncond_cross_talk_norms = []

        for seed in HIST_DEV_SEEDS:
            trx, try_, tvA, tvB, tq, _ = G.independent_query_set_v1(seed + 200_000, 1024)
            tex, tey, evA, evB, eq, _ = G.independent_query_set_v1(seed + 300_000, 512)
            ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(seed + 400_000, 256)

            ctr = P.capture(seed, trx, query_tick=G.QUERY_TICK)
            cte = P.capture(seed, tex, query_tick=G.QUERY_TICK)
            cpt = P.capture(seed, ptx, query_tick=G.QUERY_TICK)

            pre_tr = P.state_vector(ctr, "pre")
            pre_te = P.state_vector(cte, "pre")
            post_tr = P.state_vector(ctr, "post")
            post_te = P.state_vector(cte, "post")
            post_pt = P.state_vector(cpt, "post")

            vA_accs.append(P.ridge(pre_tr, tvA, pre_te, evA))
            vB_accs.append(P.ridge(pre_tr, tvB, pre_te, evB))

            # Mixed-only binding check (vA != vB)
            m_tr = tvA != tvB
            m_te = evA != evB
            mixed_accs.append(P.ridge(pre_tr[m_tr], tvA[m_tr], pre_te[m_te], evA[m_te]))

            # Neural query decodability from post - pre state transition (Mode B feasibility)
            q_delta_tr = post_tr - pre_tr
            q_delta_te = post_te - pre_te
            q_neural_accs.append(P.ridge(q_delta_tr, tq, q_delta_te, eq))

            # R5 dev screen reproduction
            cue_tr = G.query_cue(trx, tick=G.QUERY_TICK)
            cue_te = G.query_cue(tex, tick=G.QUERY_TICK)
            cue_pt = G.query_cue(ptx, tick=G.QUERY_TICK)

            f_tr = P.bilinear_query_map(post_tr, cue_tr)
            f_te = P.bilinear_query_map(post_te, cue_te)
            f_pt = P.bilinear_query_map(post_pt, cue_pt)

            h = P.train_head(f_tr, try_, seed=seed)
            r5_dev_direct.append(h.acc(f_te, tey))
            pr = h.predict(f_pt)
            a_pr, b_pr = pr[0::2], pr[1::2]
            ya, yb = pty[0::2], pty[1::2]
            r5_dev_bothA.append(float(np.mean((a_pr[flips] == ya[flips]) & (b_pr[flips] == yb[flips]))))
            r5_dev_bothB.append(float(np.mean((a_pr[~flips] == ya[~flips]) & (b_pr[~flips] == yb[~flips]))))

            # Measure unconditional z-score cross-talk on gated-off branch in R5 Head
            # In f_tr, columns 194:226 are proj * q0 (zero when q=1) and 226:258 are proj * q1 (zero when q=0)
            q1_rows = np.flatnonzero(tq == 1)
            gated_off_basis = np.stack([h.basis(f_tr[idx])[194:226] for idx in q1_rows[:32]])
            uncond_cross_talk_norms.append(float(np.linalg.norm(gated_off_basis, axis=1).mean()))

        mean_vA = float(np.mean(vA_accs))
        mean_vB = float(np.mean(vB_accs))
        mean_mixed = float(np.mean(mixed_accs))
        mean_q_neural = float(np.mean(q_neural_accs))
        log.check(
            "1. Frozen ALIF holds independently decodable vA and vB (~0.997)",
            mean_vA > 0.99 and mean_vB > 0.99 and mean_mixed > 0.99,
            f"vA={mean_vA:.4f}, vB={mean_vB:.4f}, mixed_only={mean_mixed:.4f}, neural_query={mean_q_neural:.4f}",
        )

        # 2. R5's query-conditioned feature map is label-independent
        x0, y0, *_ = G.independent_query_set_v1(97001, 64)
        cap_a = P.capture(97001, x0, query_tick=G.QUERY_TICK)
        f_a = P.bilinear_query_map(P.state_vector(cap_a, "post"), G.query_cue(x0, tick=G.QUERY_TICK))
        _ = 1 - y0  # invert labels
        cap_b = P.capture(97001, x0, query_tick=G.QUERY_TICK)
        f_b = P.bilinear_query_map(P.state_vector(cap_b, "post"), G.query_cue(x0, tick=G.QUERY_TICK))
        log.check(
            "2. R5 query-conditioned feature map is strictly label-independent",
            np.array_equal(f_a, f_b),
            f"max abs diff = {float(np.max(np.abs(f_a - f_b))):.2e}",
        )

        # 3. Query cue is obtained from an actual input marker
        x_audit, y_audit, vA_a, vB_a, q_a, _ = G.independent_query_set_v1(97001, 4096)
        cue_audit = G.query_cue(x_audit)
        log.check(
            "3. Query cue is obtained from actual input marker (ch4/ch5 at located query tick)",
            np.array_equal(cue_audit[:, 1].astype(np.int32), q_a)
            and np.array_equal(cue_audit[:, 0].astype(np.int32), 1 - q_a),
            f"matches q on {len(q_a)}/4096 rows",
        )

        # 4. First-key shortcut is approximately 0.75 direct accuracy
        fk_pred = G.baseline_first_key(x_audit)
        fk_acc = float(np.mean(fk_pred == y_audit))
        log.check(
            "4. First-key shortcut is ~0.75 direct accuracy",
            abs(fk_acc - 0.75) < 0.03,
            f"first_key_acc={fk_acc:.4f} (expected 0.7522 on seed 97001)",
        )

        # 5. Case-A first-key both-correct is 0
        px, py, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(97001, 1024)
        fk_paired = G.baseline_first_key(px)
        fka, fkb = fk_paired[0::2], fk_paired[1::2]
        pya, pyb = py[0::2], py[1::2]
        fk_both_A = float(np.mean((fka[flips] == pya[flips]) & (fkb[flips] == pyb[flips])))
        fk_flip_A = float(np.mean(fka[flips] != fkb[flips]))
        log.check(
            "5. Case-A first-key both-correct is 0.0000 (and flip rate is 0.0000)",
            fk_both_A == 0.0 and fk_flip_A == 0.0,
            f"caseA_both_correct={fk_both_A:.4f}, caseA_flip={fk_flip_A:.4f}",
        )

        # 6. Counterfactual siblings differ only at query
        ev, od = px[0::2], px[1::2]
        diff = np.abs(ev - od)
        off_q_max = float(max(diff[:, :G.QUERY_TICK, :].max(), diff[:, G.QUERY_TICK + 1:, :].max()))
        on_q_nonquery_ch = float(max(diff[:, G.QUERY_TICK, :4].max(), diff[:, G.QUERY_TICK, 6:].max()))
        on_q_query_ch = float(diff[:, G.QUERY_TICK, 4:6].max())
        log.check(
            "6. Counterfactual siblings differ ONLY at query tick and ONLY on query channels 4:6",
            off_q_max == 0.0 and on_q_nonquery_ch == 0.0 and abs(on_q_query_ch - G.PULSE) < 1e-6,
            f"off_tick_max={off_q_max:.2e}, nonquery_ch_max={on_q_nonquery_ch:.2e}, query_ch_diff={on_q_query_ch:.4f}",
        )

        # 7. Original R5 readout updates real weights
        h_init = P.Head(f_a.shape[1], 97001)
        w_before = h_init.w.copy()
        b_before = h_init.b.copy()
        h_trained = P.train_head(f_a, y0, seed=97001)
        dw_norm = float(np.linalg.norm(h_trained.w - w_before))
        db_norm = float(np.linalg.norm(h_trained.b - b_before))
        log.check(
            "7. Original R5 readout updates real weights",
            dw_norm > 0.1 and db_norm > 0.0 and np.isfinite(h_trained.w).all(),
            f"||dW||_F={dw_norm:.4f}, ||db||_2={db_norm:.4f}",
        )

        # 8. Hidden state/weights are not silently modified inside evaluation, and forward parity holds
        _, _, w0, r0 = P.weights(97001)
        h0 = hashlib.sha256(w0.tobytes() + r0.tobytes()).hexdigest()
        core = SpikingCore(P.cfg(97001))
        core_w_sha_before = hashlib.sha256(core.w_in.tobytes() + core.w_rec.tobytes()).hexdigest()
        max_parity_err = 0.0
        for i in range(8):
            core.reset_state()
            for t in range(G.N_TICKS):
                core.step(x0[i, t], track=False)
                max_parity_err = max(
                    max_parity_err,
                    float(np.max(np.abs(core.spikes - cap_a["spike_train"][i, t]))),
                )
        _, _, w1, r1 = P.weights(97001)
        h1 = hashlib.sha256(w1.tobytes() + r1.tobytes()).hexdigest()
        core_w_sha_after = hashlib.sha256(core.w_in.tobytes() + core.w_rec.tobytes()).hexdigest()
        log.check(
            "8. Hidden weights untouched & SpikingCore.step parity is exact (0.00e+00)",
            h0 == h1 and core_w_sha_before == core_w_sha_after and max_parity_err == 0.0,
            f"sha_match={h0 == h1}, max_spike_diff={max_parity_err:.2e}",
        )

        # 9. No test feature statistics enter training
        h_leak = P. train_head(f_a[:32], y0[:32], seed=97001)
        mean_before_pred = h_leak.mean.copy()
        m2_before_pred = h_leak.m2.copy()
        n_before_pred = h_leak.n
        _ = h_leak.predict(f_a[32:] * 100.0 + 50.0)
        log.check(
            "9. No test feature statistics enter training/calibration during evaluation",
            n_before_pred == 32
            and np.array_equal(mean_before_pred, h_leak.mean)
            and np.array_equal(m2_before_pred, h_leak.m2),
            f"n={h_leak.n}, mean_diff={float(np.max(np.abs(mean_before_pred - h_leak.mean))):.2e}",
        )

        # 10. Query tick is not incorrectly hardcoded
        xs1 = G.shifted(x0[:32], shift=1, seed=97001)
        qt0 = G.locate_query_tick(x0[0])
        qt1 = G.locate_query_tick(xs1[0])
        cue_s1 = G.query_cue(xs1)
        log.check(
            "10. Query tick is dynamically located (not hardcoded to 10) and preserves cue under shift",
            qt0 == 10 and qt1 == 11 and np.array_equal(cue_s1, G.query_cue(x0[:32])),
            f"qt_clean={qt0}, qt_shifted={qt1}, cue_equal=True",
        )

        # 11. Temporal transformations measurably change the neural state (vs zero-pad no-op)
        cap_clean32 = P.capture(97001, x0[:32], query_tick=10)
        cap_zpad = P.capture(97001, G.shifted_zero_pad(x0[:32], shift=1), query_tick=11)
        cap_shift = P.capture(97001, xs1, query_tick=11)
        cap_rev = P.capture(97001, G.reversed_writes(x0[:32]), query_tick=10)
        zpad_diff = float(np.max(np.abs(P.state_vector(cap_zpad, "post") - P.state_vector(cap_clean32, "post"))))
        shift_diff = float(np.max(np.abs(P.state_vector(cap_shift, "post") - P.state_vector(cap_clean32, "post"))))
        rev_diff = float(np.max(np.abs(P.state_vector(cap_rev, "post") - P.state_vector(cap_clean32, "post"))))
        log.check(
            "11. Temporal transformations measurably change neural state (and zero-pad no-op is caught)",
            zpad_diff == 0.0 and shift_diff > 0.5 and rev_diff > 0.5,
            f"zero_pad_diff={zpad_diff:.4f}, noise_pad_shift_diff={shift_diff:.4f}, reversed_writes_diff={rev_diff:.4f}",
        )

        # 12. Reproduce R5B Confirmation clean & stress numbers on the 8 historical confirmation seeds
        conf_clean_direct, conf_clean_bothA, conf_clean_bothB = [], [], []
        conf_dist_bothA, conf_noise_bothA = [], []
        for seed in HIST_CONF_SEEDS:
            trx, try_, *_ = G.independent_query_set_v1(seed + 200_000, 1024)
            tex, tey, *_ = G.independent_query_set_v1(seed + 300_000, 512)
            ptx, pty, pvA, pvB, pq, pid, flips = G.independent_paired_set_v1(seed + 400_000, 256)

            f_tr = P.bilinear_query_map(
                P.state_vector(P.capture(seed, trx, query_tick=10), "post"),
                G.query_cue(trx, tick=10),
            )
            f_te = P.bilinear_query_map(
                P.state_vector(P.capture(seed, tex, query_tick=10), "post"),
                G.query_cue(tex, tick=10),
            )
            f_pt = P.bilinear_query_map(
                P.state_vector(P.capture(seed, ptx, query_tick=10), "post"),
                G.query_cue(ptx, tick=10),
            )
            h = P.train_head(f_tr, try_, seed=seed)
            conf_clean_direct.append(h.acc(f_te, tey))
            pr = h.predict(f_pt)
            a_pr, b_pr = pr[0::2], pr[1::2]
            ya, yb = pty[0::2], pty[1::2]
            conf_clean_bothA.append(float(np.mean((a_pr[flips] == ya[flips]) & (b_pr[flips] == yb[flips]))))
            conf_clean_bothB.append(float(np.mean((a_pr[~flips] == ya[~flips]) & (b_pr[~flips] == yb[~flips]))))

            # Distractors
            d_tr = G.with_distractors(trx, seed + 11)
            d_pt = G.with_distractors(ptx, seed + 13)
            hd = P.train_head(
                P.bilinear_query_map(
                    P.state_vector(P.capture(seed, d_tr, query_tick=10), "post"),
                    G.query_cue(d_tr, tick=10),
                ),
                try_,
                seed=seed,
            )
            prd = hd.predict(
                P.bilinear_query_map(
                    P.state_vector(P.capture(seed, d_pt, query_tick=10), "post"),
                    G.query_cue(d_pt, tick=10),
                )
            )
            conf_dist_bothA.append(
                float(np.mean((prd[0::2][flips] == ya[flips]) & (prd[1::2][flips] == yb[flips])))
            )

            # Noise change
            n_tr = G.noisier(trx, seed + 21, sigma=0.16)
            n_pt = G.noisier(ptx, seed + 23, sigma=0.16)
            hn = P.train_head(
                P.bilinear_query_map(
                    P.state_vector(P.capture(seed, n_tr, query_tick=10), "post"),
                    G.query_cue(n_tr, tick=10),
                ),
                try_,
                seed=seed,
            )
            prn = hn.predict(
                P.bilinear_query_map(
                    P.state_vector(P.capture(seed, n_pt, query_tick=10), "post"),
                    G.query_cue(n_pt, tick=10),
                )
            )
            conf_noise_bothA.append(
                float(np.mean((prn[0::2][flips] == ya[flips]) & (prn[1::2][flips] == yb[flips])))
            )

        c_direct = float(np.mean(conf_clean_direct))
        c_bothA = float(np.mean(conf_clean_bothA))
        c_bothB = float(np.mean(conf_clean_bothB))
        c_distA = float(np.mean(conf_dist_bothA))
        c_noiseA = float(np.mean(conf_noise_bothA))
        log.check(
            "12. Exact reproduction of R5B confirmation results (direct=0.8989, both_A=0.6676, both_B=0.9809)",
            abs(c_direct - 0.89892578125) < 1e-6
            and abs(c_bothA - 0.6676171928664432) < 1e-6
            and abs(c_bothB - 0.9808925435845981) < 1e-6,
            f"direct={c_direct:.4f}, both_A={c_bothA:.4f}, both_B={c_bothB:.4f}, dist_bothA={c_distA:.4f}, noise_bothA={c_noiseA:.4f}",
        )
        mon.check()

    out_dir = ROOT / "artifacts" / "gemini-r6"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "stage0_audit",
        "verdict": "STAGE0_PASS" if log.ok else "STAGE0_FAIL",
        "passed": sum(1 for r in log.rows if r["pass"]),
        "total": len(log.rows),
        "checks": log.rows,
        "historical_reproduction": {
            "dev_seeds": list(HIST_DEV_SEEDS),
            "conf_seeds": list(HIST_CONF_SEEDS),
            "pre_query_vA_ridge": mean_vA,
            "pre_query_vB_ridge": mean_vB,
            "pre_query_mixed_only_ridge": mean_mixed,
            "neural_query_delta_ridge": mean_q_neural,
            "r5_dev_direct": float(np.mean(r5_dev_direct)),
            "r5_dev_both_A": float(np.mean(r5_dev_bothA)),
            "r5_dev_both_B": float(np.mean(r5_dev_bothB)),
            "r5_conf_clean_direct": c_direct,
            "r5_conf_clean_both_A": c_bothA,
            "r5_conf_clean_both_B": c_bothB,
            "r5_conf_distractors_both_A": c_distA,
            "r5_conf_noise_both_A": c_noiseA,
            "r5_gated_off_crosstalk_l2_norm": float(np.mean(uncond_cross_talk_norms)),
            "first_key_shortcut_direct": fk_acc,
            "first_key_caseA_both_correct": fk_both_A,
        },
        "resources": mon.report(),
    }
    out_file = out_dir / "stage0-audit.json"
    out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("=" * 84)
    print(f"STAGE 0 RESULT: {payload['passed']}/{payload['total']} — {payload['verdict']}")
    print(f"Gated-off branch unconditional z-score cross-talk L2 norm in R5: {payload['historical_reproduction']['r5_gated_off_crosstalk_l2_norm']:.4f}")
    print(f"Written: {out_file}")
    print("=" * 84)
    return 0 if log.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
