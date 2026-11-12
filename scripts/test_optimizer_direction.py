"""Micro-experiment comparing Coordinate-wise RMS vs Direction-Preserving Global Norm optimizer."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scarx.neural.core import Config
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.evaluation.r8_benchmark_integrity import generate_strict_binding
from scarx.evaluation.r10_bottleneck_analysis import evaluate_model_standard


def train_with_optimizers(method="braintrace", seed=140011, episodes=1024):
    cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
    train_x, train_y = generate_strict_binding(seed + 200, 1024)
    refit_x, refit_y = generate_strict_binding(seed + 300, 256)
    id_x, id_y = generate_strict_binding(seed + 400, 512)
    ood_x, ood_y = generate_strict_binding(seed + 500, 512, ood=True)
    data = {
        "train": (train_x, train_y),
        "refit": (refit_x, refit_y),
        "id": (id_x, id_y),
        "ood": (ood_x, ood_y),
    }

    # Baseline Coordinate-wise RMS
    m_base = IndependentLearningALIF(cfg, method=method)
    m_base.calibrate(train_x[:32])
    for ep in range(episodes):
        idx = ep % len(train_x)
        m_base.learn_episode(train_x[idx], int(train_y[idx]))
    res_base = evaluate_model_standard(m_base, data)

    # Direction-Preserving Global Norm Optimizer
    # We subclass or create a custom stepping model
    class GlobalNormALIF(IndependentLearningALIF):
        def learn_episode(self, sequence, target, stat_rate=0.01, record_diagnostics=False):
            seq = np.asarray(sequence, dtype=np.float32)
            self.reset_episode()
            for row in seq:
                self.step(row, track=True)

            f = self.core.features
            self.f_mean = (1.0 - stat_rate) * self.f_mean + stat_rate * f
            self.f_var = (1.0 - stat_rate) * self.f_var + stat_rate * (f - self.f_mean) ** 2
            f_hat, raw, scale = self.standardized_features(f)

            logits = self.core.w_out @ f_hat + self.core.bias
            logits -= np.max(logits)
            p = np.exp(logits)
            p /= np.sum(p)
            loss = -float(np.log(max(float(p[int(target)]), 1e-12)))

            err = -p.copy()
            err[int(target)] += 1.0

            pre_update_signal = (self.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale

            # Readout update
            grad_out = err[:, None] * f_hat[None, :]
            self.core.w_out += np.clip(0.04 * grad_out, -0.025, 0.025)
            self.core.bias += np.clip(0.04 * err, -0.025, 0.025)
            np.clip(self.core.w_out, -3.5, 3.5, out=self.core.w_out)
            np.clip(self.core.bias, -3.5, 3.5, out=self.core.bias)

            # Hidden update with Direction-Preserving Global Norm
            signal = pre_update_signal
            direction = signal[:, None] * self.e_f

            # Momentum
            self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
            # Global Frobenius norm
            gnorm = float(np.linalg.norm(self.syn_trace)) + 1e-6
            # Target effective learning step with matched magnitude
            target_scale = min(0.008, 0.008 / max(1.0, gnorm / 0.5))
            step = (target_scale / gnorm) * self.syn_trace

            d = self.config.inputs
            self.core.w_in += step[:, :d]
            self.core.w_rec += step[:, d:]
            np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
            np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)
            self.hidden_updates += 1
            self.core.updates += 1
            self.episodes_learned += 1
            return loss, None

    m_gn = GlobalNormALIF(cfg, method=method)
    m_gn.calibrate(train_x[:32])
    for ep in range(episodes):
        idx = ep % len(train_x)
        m_gn.learn_episode(train_x[idx], int(train_y[idx]))
    res_gn = evaluate_model_standard(m_gn, data)

    print(f"Method: {method} Seed: {seed}")
    print(f"  Baseline (Coordinate RMS): Direct={res_base['id_direct']*100:.2f}% | Probe={res_base['id_probe']*100:.2f}% | OOD={res_base['ood_direct']*100:.2f}%")
    print(f"  Direction-Preserving GN  : Direct={res_gn['id_direct']*100:.2f}% | Probe={res_gn['id_probe']*100:.2f}% | OOD={res_gn['ood_direct']*100:.2f}%")
    print(f"  Delta: Direct = {res_gn['id_direct']*100 - res_base['id_direct']*100:+.2f} pp | Probe = {res_gn['id_probe']*100 - res_base['id_probe']*100:+.2f} pp")


if __name__ == "__main__":
    train_with_optimizers("braintrace", 140011)
    train_with_optimizers("braintrace", 140017)
    train_with_optimizers("braintrace", 140023)
