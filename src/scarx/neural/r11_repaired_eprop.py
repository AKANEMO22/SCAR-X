"""Repaired Damped Reset E-Prop with verified endpoint parity (P0.A).

Implements:
- DampedResetEPropALIF with exact SCAR-X logistic surrogate spike_slope.
- Exact endpoint parity:
  alpha_reset = 0.0 == eprop_matched (Bellec et al. 2020)
  alpha_reset = 1.0 == own_reset (SCAR-X R8 baseline)
"""
from __future__ import annotations

import copy
import numpy as np
from ..neural.core import Config, SpikingCore
from ..evaluation.readout_eligibility import spike_slope
from .r10_independent_replication import IndependentLearningALIF


class DampedResetEPropALIF(IndependentLearningALIF):
    """E-prop with Damped Spike Reset using exact SCAR-X logistic surrogate derivative.

    Equations:
    u_e = leak * e_v + pre
    z_e = spike_slope(u - theta) * (u_e - beta * e_a)
    e_v = u_e - (alpha_reset * theta * z_e) - spikes * beta * e_a
    e_a = rho * e_a + z_e
    e_f = gamma_e * e_f + (1 - gamma_e) * z_e_gated

    Endpoints:
    - alpha_reset = 0.0: Exactly reproduces eprop_matched (detached reset spike).
    - alpha_reset = 1.0: Exactly reproduces own_reset (full reset derivative).
    - alpha_reset = 0.5: Damped reset e-prop (Bellec et al. 2020 Supp Note 2).
    """

    def __init__(
        self,
        config: Config,
        alpha_reset: float = 0.5,
        condition: str = "learned_mtsg",
        sensory_threshold: float = 0.6,
    ):
        super().__init__(config, method="eprop_matched", condition=condition, sensory_threshold=sensory_threshold)
        self.alpha_reset = float(alpha_reset)
        # Ensure multi-timescale eligibility decay vector matches baseline
        self.trace_kind = "multi"
        self.gate_mode = "activity"
        self.gamma_e_vec = np.linspace(0.25, 0.92, config.hidden, dtype=np.float32)[:, None]

    def step(self, x: np.ndarray, track: bool = True) -> np.ndarray:
        c = self.config
        row = np.asarray(x, dtype=np.float32)
        if row.shape != (c.inputs,) or not np.isfinite(row).all():
            raise ValueError("Invalid input vector")

        if track:
            old_z = self.core.spikes
            theta = c.threshold + c.adaptation_strength * self.core.adaptation
            i_in = self.core.w_in @ row
            i_rec = np.sum(self.core.w_rec * old_z[self.core.indices], axis=1)
            u = (c.leak * self.core.voltage + i_rec) + i_in
            psi = spike_slope(u - theta)
            pre = np.concatenate(
                (np.broadcast_to(row, (c.hidden, c.inputs)), old_z[self.core.indices]),
                axis=1,
            )
            u_e = c.leak * self.e_v + pre
            a_old = self.e_a.copy()
            z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)

        # Single unchanged neural core forward
        self.core.step(row, track=False)

        if track:
            # Interpolated reset derivative: alpha_reset * (theta * z_e)
            reset_spike_term = self.alpha_reset * theta[:, None] * z_e
            adaptive_term = self.core.spikes[:, None] * c.adaptation_strength * a_old
            self.e_v[:] = u_e - reset_spike_term - adaptive_term

            self.e_a[:] = c.adaptation_decay * a_old + z_e
            if self.gate_mode == "activity":
                gate = 1.0 if float(np.max(np.abs(row))) > self.sensory_threshold else 0.15
                z_e[:, c.inputs:] *= gate
            self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e

        self.total_ticks += 1
        return self.core.features
