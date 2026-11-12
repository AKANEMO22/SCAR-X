"""Independent evaluation and diagnostic suite for SCAR-X Round 10.

Implements:
1. Standard evaluation battery: ID direct, ID probe, OOD direct, OOD probe, causal revert drop.
2. Full Surrogate Recurrent Oracle: exact whole-sequence recurrent sensitivity control.
3. Diagnostic suite for Hypotheses H1-H4:
   - H1: Feedback signal distortion & alignment with true feature gradient.
   - H2: Optimizer distortion (raw direction vs actual applied update).
   - H3: Readout decoding gap (probe vs direct head).
   - H4: Gradient quality drift across training trajectory.
"""
from __future__ import annotations

import copy
import numpy as np
from ..neural.core import Config, SpikingCore
from .readout_eligibility import spike_slope
from ..neural.r10_independent_replication import IndependentLearningALIF


def refit_probe(
    train_f: np.ndarray, train_y: np.ndarray, test_f: np.ndarray
) -> np.ndarray:
    """Independent ridge regression linear probe on frozen features."""
    mean = train_f.mean(axis=0)
    scale = train_f.std(axis=0) + 1e-4
    a = np.column_stack(((train_f - mean) / scale, np.ones(len(train_f))))
    b = np.column_stack(((test_f - mean) / scale, np.ones(len(test_f))))
    # Ridge solve with lambda = 0.01
    reg = 0.01 * np.eye(a.shape[1], dtype=np.float32)
    weights = np.linalg.solve(a.T @ a + reg, a.T @ np.eye(2, dtype=np.float32)[train_y])
    return np.argmax(b @ weights, axis=1).astype(np.int32)


def evaluate_model_standard(
    model: IndependentLearningALIF, data: dict
) -> dict:
    """Run independent evaluation pass on ID, OOD, and counterfactual weight revert."""
    refit_x, refit_y = data["refit"][0], data["refit"][1]
    refit_f = model.features_many(refit_x)

    results = {}
    for split in ("id", "ood"):
        x, y = data[split][0], data[split][1]
        f = model.features_many(x)

        scale = np.maximum(np.sqrt(model.f_var), 0.05).astype(np.float32)
        f_hat = np.clip((f - model.f_mean) / scale, -5.0, 5.0).astype(np.float32)
        direct_pred = np.argmax(f_hat @ model.core.w_out.T + model.core.bias, axis=1).astype(np.int32)
        probe_pred = refit_probe(refit_f, refit_y, f)

        results[f"{split}_direct"] = float(np.mean(direct_pred == y))
        results[f"{split}_probe"] = float(np.mean(probe_pred == y))

    # Weight revert counterfactuals
    id_x, id_y = data["id"][0], data["id"][1]
    clone = copy.deepcopy(model)
    clone.revert_hidden_weights(revert_in=True, revert_rec=True)
    f_revert = clone.features_many(id_x)
    scale = np.maximum(np.sqrt(clone.f_var), 0.05).astype(np.float32)
    f_hat_rev = np.clip((f_revert - clone.f_mean) / scale, -5.0, 5.0).astype(np.float32)
    rev_direct = np.argmax(f_hat_rev @ clone.core.w_out.T + clone.core.bias, axis=1).astype(np.int32)
    results["revert_both_direct"] = float(np.mean(rev_direct == id_y))
    results["direct_revert_drop"] = float(results["id_direct"] - results["revert_both_direct"])

    return results


def full_surrogate_recurrent_sensitivity(
    core: SpikingCore, sequence: np.ndarray
) -> np.ndarray:
    """Full forward-recurrent surrogate sensitivity of features to all hidden weights.

    Propagates sensitivity across both within-neuron dynamics and recurrent spikes.
    Exact control baseline for gradient direction checking.
    Output shape: (hidden, total_params) where total_params = hidden * (inputs + fan_in).
    """
    c = core.config
    seq = np.asarray(sequence, dtype=np.float32)
    m = copy.deepcopy(core)
    m.reset_state()
    h = c.hidden
    k_tot = c.inputs + c.fan_in
    pcount = h * k_tot

    ev = np.zeros((h, pcount), dtype=np.float64)
    ea = np.zeros((h, pcount), dtype=np.float64)
    ez = np.zeros((h, pcount), dtype=np.float64)
    ef = np.zeros((h, pcount), dtype=np.float64)

    for row in seq:
        old_z = m.spikes
        theta = c.threshold + c.adaptation_strength * m.adaptation
        i_in = m.w_in @ row
        i_rec = np.sum(m.w_rec * old_z[m.indices], axis=1)
        u = (c.leak * m.voltage + i_rec) + i_in
        psi = spike_slope(u - theta).astype(np.float64)

        pre = np.concatenate((np.broadcast_to(row, (h, c.inputs)), old_z[m.indices]), axis=1)
        injection = np.zeros((h, pcount), dtype=np.float64)
        for j in range(h):
            injection[j, j * k_tot : (j + 1) * k_tot] = pre[j]

        # Recurrent cross-neuron propagation
        rec_propagation = np.einsum("hk,hkp->hp", m.w_rec.astype(np.float64), ez[m.indices])
        ue = c.leak * ev + rec_propagation + injection
        ze = psi[:, None] * (ue - c.adaptation_strength * ea)

        m.step(row, track=False)
        ev = ue - theta[:, None] * ze - m.spikes[:, None] * c.adaptation_strength * ea
        ea = c.adaptation_decay * ea + ze
        ef = c.feature_decay * ef + (1.0 - c.feature_decay) * ze
        ez = ze

    return ef


def compute_gradient_cosine(
    local_direction: np.ndarray, reference_vector: np.ndarray
) -> dict:
    """Compute cosine, sign agreement, and norm ratio between local and full vectors."""
    a = np.asarray(local_direction, dtype=np.float64).ravel()
    b = np.asarray(reference_vector, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na * nb == 0:
        return {"cosine": 0.0, "sign_agreement": 0.5, "norm_ratio": 0.0}
    cos = float(np.dot(a, b) / (na * nb))
    sign_agree = float(np.mean(np.sign(a) == np.sign(b)))
    return {"cosine": cos, "sign_agreement": sign_agree, "norm_ratio": float(na / nb)}
