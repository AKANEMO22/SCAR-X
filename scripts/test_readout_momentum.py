"""Test whether adding momentum to the readout update resolves the Readout Bottleneck (H3)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scarx.neural.core import Config
from scarx.neural.r10_independent_replication import IndependentLearningALIF
from scarx.evaluation.r8_benchmark_integrity import generate_strict_binding
from scarx.evaluation.r10_bottleneck_analysis import evaluate_model_standard


class MomentumReadoutALIF(IndependentLearningALIF):
    def __init__(self, config, method="braintrace", condition="learned_mtsg"):
        super().__init__(config, method=method, condition=condition)
        self.w_out_trace = np.zeros_like(self.core.w_out)
        self.b_out_trace = np.zeros_like(self.core.bias)

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

        # Pre-update credit signal
        pre_update_signal = (self.core.w_out.T @ err) * (np.abs(raw) < 5.0) / scale

        # Momentum-stabilized Readout update
        grad_out = err[:, None] * f_hat[None, :]
        self.w_out_trace = 0.90 * self.w_out_trace + 0.10 * grad_out
        self.b_out_trace = 0.90 * self.b_out_trace + 0.10 * err

        self.core.w_out += np.clip(0.04 * self.w_out_trace, -0.025, 0.025)
        self.core.bias += np.clip(0.04 * self.b_out_trace, -0.025, 0.025)
        np.clip(self.core.w_out, -3.5, 3.5, out=self.core.w_out)
        np.clip(self.core.bias, -3.5, 3.5, out=self.core.bias)

        perm = self.signal_rng.permutation(self.config.hidden)

        # Hidden update (standard)
        signal = pre_update_signal
        if self.condition == "shuffled_signal":
            signal = signal[perm]
        direction = signal[:, None] * self.e_f

        self.syn_trace = 0.90 * self.syn_trace + 0.10 * direction
        self.syn_energy = 0.99 * self.syn_energy + 0.01 * (direction * direction)
        rms = np.sqrt(self.syn_energy) + 1e-4
        norm_step = self.syn_trace / rms

        lr_hid = 0.008
        clip_hid = 0.010
        step = np.clip(lr_hid * norm_step, -clip_hid, clip_hid)

        d = self.config.inputs
        self.core.w_in += step[:, :d]
        self.core.w_rec += step[:, d:]
        np.clip(self.core.w_in, -3.0, 3.0, out=self.core.w_in)
        np.clip(self.core.w_rec, -3.0, 3.0, out=self.core.w_rec)
        self.hidden_updates += 1

        self.core.updates += 1
        self.episodes_learned += 1
        return loss, None


def test_methods(method="braintrace"):
    for seed in (140011, 140017, 140023):
        train_x, train_y = generate_strict_binding(seed + 200, 1024)
        refit_x, refit_y = generate_strict_binding(seed + 300, 256)
        id_x, id_y = generate_strict_binding(seed + 400, 512)
        ood_x, ood_y = generate_strict_binding(seed + 500, 512, ood=True)
        data = {"train": (train_x, train_y), "refit": (refit_x, refit_y), "id": (id_x, id_y), "ood": (ood_x, ood_y)}

        cfg = Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=seed)
        m_base = IndependentLearningALIF(cfg, method=method)
        m_base.calibrate(train_x[:32])
        for ep in range(1024):
            idx = ep % len(train_x)
            m_base.learn_episode(train_x[idx], int(train_y[idx]))
        r_base = evaluate_model_standard(m_base, data)

        m_mom = MomentumReadoutALIF(cfg, method=method)
        m_mom.calibrate(train_x[:32])
        for ep in range(1024):
            idx = ep % len(train_x)
            m_mom.learn_episode(train_x[idx], int(train_y[idx]))
        r_mom = evaluate_model_standard(m_mom, data)

        print(f"[{method}] Seed: {seed}")
        print(f"  Base Direct: {r_base['id_direct']*100:.2f}% | Probe: {r_base['id_probe']*100:.2f}% | Gap: {r_base['id_probe']*100 - r_base['id_direct']*100:+.2f} pp")
        print(f"  Mom  Direct: {r_mom['id_direct']*100:.2f}% | Probe: {r_mom['id_probe']*100:.2f}% | Gap: {r_mom['id_probe']*100 - r_mom['id_direct']*100:+.2f} pp")
        print(f"  Delta Direct: {r_mom['id_direct']*100 - r_base['id_direct']*100:+.2f} pp")


if __name__ == "__main__":
    print("Testing BrainTrace with Momentum Readout:")
    test_methods("braintrace")
    print("\nTesting E-prop matched with Momentum Readout:")
    test_methods("eprop_matched")
