"""Phase D: Single grounded improvement - Damped Reset E-prop (alpha_reset = 0.5)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scarx.neural.core import Config
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.evaluation.r8_benchmark_integrity import (
    generate_strict_binding,
    generate_xor_representation,
    generate_distractor_binding,
)
from scarx.evaluation.r10_bottleneck_analysis import evaluate_model_standard


class DampedResetEPropALIF(IndependentLearningALIF):
    """E-prop with Damped Reset (Bellec et al. 2020 Supp Note 2).

    Interpolates between detached reset (eprop_matched, alpha=0) and full reset (own, alpha=1).
    alpha_reset = 0.5 preserves event boundary damping without causing explosive reset instability.
    """
    def __init__(self, config, alpha_reset=0.5, condition="learned_mtsg"):
        super().__init__(config, method="eprop_matched", condition=condition)
        self.alpha_reset = float(alpha_reset)

    def step(self, x, track=True):
        c = self.config
        row = np.asarray(x, dtype=np.float32)
        if track:
            old_z = self.core.spikes
            theta = c.threshold + c.adaptation_strength * self.core.adaptation
            i_in = self.core.w_in @ row
            i_rec = np.sum(self.core.w_rec * old_z[self.core.indices], axis=1)
            u = (c.leak * self.core.voltage + i_rec) + i_in
            psi = 1.0 / (1.0 + 10.0 * np.abs(u - theta)) ** 2 # surrogate derivative
            pre = np.concatenate((np.broadcast_to(row, (c.hidden, c.inputs)), old_z[self.core.indices]), axis=1)
            u_e = c.leak * self.e_v + pre
            a_old = self.e_a.copy()
            z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)

        self.core.step(row, track=False)

        if track:
            # Damped reset: interpolate between alpha=0 (eprop_matched) and alpha=1 (own)
            self.e_v[:] = u_e - (self.alpha_reset * theta[:, None] * z_e) - self.core.spikes[:, None] * c.adaptation_strength * a_old
            self.e_a[:] = c.adaptation_decay * a_old + z_e
            if self.gate_mode == "activity":
                gate = 1.0 if float(np.max(np.abs(row))) > self.sensory_threshold else 0.15
                z_e[:, c.inputs:] *= gate
            self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e

        self.total_ticks += 1
        return self.core.features


def evaluate_task(task_name, gen_func):
    seeds = (140011, 140017, 140023)
    print(f"\n=== EVALUATING TASK: {task_name} ===")

    for alpha in [0.0, 0.5, 1.0]:
        directs = []
        probes = []
        for seed in seeds:
            train_x, train_y = gen_func(seed + 200, 1024)
            refit_x, refit_y = gen_func(seed + 300, 256)
            id_x, id_y = gen_func(seed + 400, 512)
            ood_x, ood_y = gen_func(seed + 500, 512, ood=True)
            data = {"train": (train_x, train_y), "refit": (refit_x, refit_y), "id": (id_x, id_y), "ood": (ood_x, ood_y)}

            cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
            model = DampedResetEPropALIF(cfg, alpha_reset=alpha)
            model.calibrate(train_x[:32])
            for ep in range(1024):
                idx = ep % len(train_x)
                model.learn_episode(train_x[idx], int(train_y[idx]))
            res = evaluate_model_standard(model, data)
            directs.append(res["id_direct"])
            probes.append(res["id_probe"])

        label = "eprop_matched (alpha=0.0)" if alpha == 0.0 else "damped_eprop (alpha=0.5)" if alpha == 0.5 else "own_reset (alpha=1.0)"
        print(f"[{label:26s}] Mean Direct: {np.mean(directs)*100:5.2f}% | Mean Probe: {np.mean(probes)*100:5.2f}% | Per Seed: {[round(d*100, 2) for d in directs]}")


if __name__ == "__main__":
    evaluate_task("Strict Binding", generate_strict_binding)
    evaluate_task("Delayed XOR", generate_xor_representation)
    evaluate_task("Distractor Binding", generate_distractor_binding)
