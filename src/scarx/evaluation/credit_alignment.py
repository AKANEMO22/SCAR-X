"""Read-only credit alignment: does the local direction actually reduce loss?

Continues audit38, which measured direction NORMS/COSINES only and explicitly
refused to call large norms useful credit. Here we apply the SAME direction the
real rule would apply (mean over endpoints, as `ReadoutEligibility.finish` does)
to a disposable copy, using the checkpoint's own Adam/clip, and measure the real
teacher-forced NLL change against norm-matched controls.

No neuraltraining, optimizer, headrefit, newlearner, Torch or serving import.
Source checkpoint objects are never mutated: every application happens on a
deep copy and every protected field is asserted unchanged.
"""
import copy
import hashlib
import numpy as np
from .word_training_audit import (
    potential_direction, prefix_forward, protected,
)


def case_direction(model, prompt, gold):
    """Mean potential direction over endpoints - exactly what finish() would apply."""
    rows = prefix_forward(model, prompt, gold[:-1], measure_credit=True)
    if len(rows) != len(gold):
        raise ValueError('Credit direction/endpoint alignment')
    directions = [potential_direction(model, row, target) for row, target in zip(rows, gold)]
    stacked = np.stack(directions)
    direction = stacked.mean(0).astype(np.float32)
    if not np.isfinite(direction).all():
        raise ValueError('Nonfinite case direction')
    return direction


class Applier:
    """Apply arbitrary directions to a disposable copy, restoring exactly each time."""

    def __init__(self, model):
        if model.learner.condition != 'correct':
            raise ValueError('Alignment measured ONLY on correct sources')
        self.worker = copy.deepcopy(model)
        learner = self.worker.learner
        self.shape = learner.bank.shape
        self.pristine = (self.worker.core.w_in.copy(), self.worker.core.w_rec.copy(),
                         learner.moment.copy(), learner.variance.copy(), learner.hidden_updates)
        self.calls = 0

    def _restore(self):
        core, learner = self.worker.core, self.worker.learner
        core.w_in[:] = self.pristine[0]
        core.w_rec[:] = self.pristine[1]
        learner.moment[:] = self.pristine[2]
        learner.variance[:] = self.pristine[3]
        learner.hidden_updates = self.pristine[4]

    def measure(self, prompt, target, direction=None):
        """Teacher-forced NLL after optionally applying one direction. None = zero-op."""
        self._restore()
        before = self.worker.core.w_in.copy(), self.worker.core.w_rec.copy()
        if direction is not None:
            if direction.shape != self.shape or not np.isfinite(direction).all():
                raise ValueError('Bounded finite applied direction')
            self.worker.learner._apply_direction(np.asarray(direction, np.float32))
        metrics = self.worker.teacher_metrics(prompt, target)
        delta = [(before[0] - self.worker.core.w_in), (before[1] - self.worker.core.w_rec)]
        applied = float(np.sqrt(sum(float(np.sum(d * d, dtype=np.float64)) for d in delta)))
        self.calls += 1
        return float(metrics['nll']), applied


def norm_matched(direction, rng):
    """Gaussian control rescaled to the same PRE-Adam norm.

    NOTE: this control does NOT match the POST-Adam applied magnitude, because
    Adam normalises elementwise and the real direction is sparse while a Gaussian
    is dense. Use `permuted` as the primary control; this one is secondary and
    both applied norms are reported so the mismatch stays visible.
    """
    raw = rng.standard_normal(direction.shape).astype(np.float32)
    norm = float(np.linalg.norm(raw.astype(np.float64)))
    target = float(np.linalg.norm(direction.astype(np.float64)))
    if norm < 1e-12 or target < 1e-12:
        raise ValueError('Degenerate control norm')
    return (raw * np.float32(target / norm)).astype(np.float32)


def permuted(direction, rng):
    """Primary control: the direction's own values in a shuffled position order.

    Preserves the exact value multiset, the exact sparsity pattern size, the norm
    and therefore the exact Adam/clip behaviour and applied magnitude, while
    destroying which synapse receives which credit. This isolates alignment from
    mere energy: if the local rule is informative it must beat its own shuffle.
    """
    flat = np.asarray(direction, np.float32).ravel().copy()
    if not np.isfinite(flat).all():
        raise ValueError('Bounded finite permuted control')
    return rng.permutation(flat).reshape(np.asarray(direction).shape).astype(np.float32)


def measure_case(applier, prompt, target, direction, rng, draws):
    """One case: NLL before, with direction, signflip, zero-op and matched controls."""
    base, _ = applier.measure(prompt, target, None)
    with_direction, applied_direction = applier.measure(prompt, target, direction)
    with_flip, applied_flip = applier.measure(prompt, target, -direction)
    zero, applied_zero = applier.measure(prompt, target, None)
    randoms = []
    applied_randoms = []
    perms = []
    applied_perms = []
    for _ in range(draws):
        control = norm_matched(direction, rng)
        value, applied = applier.measure(prompt, target, control)
        randoms.append(value)
        applied_randoms.append(applied)
        control = permuted(direction, rng)
        value, applied = applier.measure(prompt, target, control)
        perms.append(value)
        applied_perms.append(applied)
    if zero != base or applied_zero != 0.0:
        raise ValueError('Zero-op measurement not deterministic')
    return {
        'nll_before': base,
        'nll_direction': with_direction,
        'nll_signflip': with_flip,
        'delta_direction': with_direction - base,
        'delta_signflip': with_flip - base,
        'delta_random_mean': float(np.mean([r - base for r in randoms])),
        'delta_random_values': [float(r - base) for r in randoms],
        'delta_permuted_mean': float(np.mean([r - base for r in perms])),
        'delta_permuted_values': [float(r - base) for r in perms],
        'gain_vs_random': float(np.mean(randoms) - with_direction),
        'gain_vs_permuted': float(np.mean(perms) - with_direction),
        'gain_vs_signflip': float(with_flip - with_direction),
        'aligned': bool(with_direction < base),
        'direction_norm_pre_adam': float(np.linalg.norm(direction.astype(np.float64))),
        'applied_norm_direction': applied_direction,
        'applied_norm_signflip': applied_flip,
        'applied_norm_random_mean': float(np.mean(applied_randoms)),
        'applied_norm_permuted_mean': float(np.mean(applied_perms)),
    }


def heldout_direction(applier, train_cases, directions):
    """Direction accumulated over TRAIN cases only, applied once, scored on dev."""
    if len(train_cases) != len(directions) or not train_cases:
        raise ValueError('Heldout direction alignment')
    combined = np.stack(directions).mean(0).astype(np.float32)
    if combined.shape != applier.shape or not np.isfinite(combined).all():
        raise ValueError('Heldout direction shape')
    return combined
