"""Transformed e-prop adapter; one unchanged SCAR-X ALIF forward, no autograd.

Reference: Bellec et al. (2020), doi:10.1038/s41467-020-17236-y.
Upstream IGITUGraz/eligibility_propagation, pinned in UPSTREAM_SHA below.
This is NOT the original EligALIF forward or its reported task performance.

Equations are adapted from the e-prop team's BSD-3-Clause reference:
Copyright 2019-2020, the e-prop team: Guillaume Bellec, Franz Scherr,
Anand Subramoney, Elias Hajek, Darjan Salaj, Robert Legenstein, Wolfgang Maass,
Institute for theoretical computer science, TU Graz, Austria.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that: (1) source redistributions retain this notice,
conditions and disclaimer; (2) binary redistributions reproduce them in their
documentation/materials; (3) copyright holder/contributor names may not endorse
derived products without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from .r8_matched_credit import MatchedLocalCredit
from .readout_eligibility import spike_slope

UPSTREAM_URL = 'https://github.com/IGITUGraz/eligibility_propagation'
UPSTREAM_SHA = 'efd02e6879c01cda3fa9a7838e8e2fd08163c16e'
VARIANTS = ('eprop_matched', 'eprop_aligned')
ARMS = ('multi_gated', 'multi_gated_shuffle', 'multi_gated_readout_only', 'multi_gated_frozen')


def published_spike_slope(margin, threshold, dampening=.3):
    """Upstream models.py:339, dt=1, n_refractory=0; fixture ONLY.

    Quality adapters both keep the existing SCAR-X logistic surrogate.
    """
    m = np.asarray(margin, dtype=np.float64)
    t = np.asarray(threshold, dtype=np.float64)
    if not np.isfinite(m).all() or not np.isfinite(t).all() or np.any(t <= 0) or dampening != .3:
        raise ValueError('Finite margins and positive threshold; fixed reference dampening')
    return dampening / t * np.maximum(0., 1. - np.abs(m / t))


def upstream_trace_scan(pre, psi, leak, rho, beta, decay_out=None):
    """NumPy form of pinned models.py:351-366,392 (no refractory/diagonal mask).

    pre[T,H,P] and psi[T,H] are explicitly supplied numerical fixtures. Return
    raw e, epsilon_v, epsilon_a BEFORE current spike, optionally readout-filtered e.
    Upstream reset amplitude is constant: its stopped-reset-spike graph has no
    adaptive-threshold reset path. This helper is NOT the SCAR-X learner step.
    """
    p = np.asarray(pre, np.float64); s = np.asarray(psi, np.float64)
    if p.ndim != 3 or s.shape != p.shape[:2] or not 1 <= p.shape[0] <= 256 or not 1 <= p.shape[1] <= 48 or not 1 <= p.shape[2] <= 16:
        raise ValueError('Bounded T,H,P fixture shape')
    if not np.isfinite(p).all() or not np.isfinite(s).all() or not all(np.isfinite(v) for v in (leak, rho, beta)) or not 0 <= leak < 1 or not 0 <= rho < 1 or beta < 0:
        raise ValueError('Finite trace inputs/constants')
    if decay_out is not None and (not np.isfinite(decay_out) or not 0 <= decay_out < 1):
        raise ValueError('Readout decay range')
    ev = np.zeros(p.shape[1:], np.float64); ea = ev.copy(); ef = ev.copy()
    raws = []; vs = []; aas = []; fs = []
    for presyn, slope in zip(p, s):
        ev = leak * ev + presyn
        # Upstream epsilon_a(t) contains all adaptation changes before z(t).
        vs.append(ev.copy()); aas.append(ea.copy())
        e = slope[:, None] * (ev - beta * ea)
        ea = rho * ea + e
        ef = e if decay_out is None else decay_out * ef + (1 - decay_out) * e
        raws.append(e.copy()); fs.append(ef.copy())
    return tuple(np.stack(x) for x in (raws, vs, aas, fs))


class LiteratureEPropALIF(MatchedLocalCredit):
    """Source-adapted e-prop with reset-spike detached, adaptive theta retained.

    Only own-neuron sensitivities are propagated. Presynaptic recurrent spikes
    are values, never differentiated through other neurons/time. Pre-update
    W_out.T@error, readout/statistics and optimizer are inherited unchanged.
    """
    def __init__(self, config, variant='eprop_matched', arm='multi_gated'):
        if variant not in VARIANTS or arm not in ARMS:
            raise ValueError('Only preregistered e-prop variants/signals')
        if (config.hidden, config.inputs, config.outputs, config.fan_in, config.plastic_rows, config.neuron) != (48, 8, 2, 8, 12, 'alif'):
            raise ValueError('Fixed H48 D8 O2 K8 R12 ALIF research scope')
        super().__init__(config, arm)
        self.eprop_variant = variant
        if variant == 'eprop_aligned':
            self.trace_kind = 'slow'; self.gate_mode = 'none'
            self.gamma_e_vec.fill(config.feature_decay)

    def step(self, x, track=True):
        c = self.config; row = np.asarray(x, dtype=np.float32)
        if row.shape != (c.inputs,) or not np.isfinite(row).all():
            raise ValueError('Invalid input vector')
        if track:
            old_z = self.core.spikes
            theta = c.threshold + c.adaptation_strength * self.core.adaptation
            i_in = self.core.w_in @ row
            i_rec = np.sum(self.core.w_rec * old_z[self.core.indices], axis=1)
            # Keep pinned R8 eligibility grouping; core.step itself is unchanged.
            u = (c.leak * self.core.voltage + i_rec) + i_in
            psi = spike_slope(u - theta)
            pre = np.concatenate((np.broadcast_to(row, (c.hidden, c.inputs)), old_z[self.core.indices]), axis=1)
            u_e = c.leak * self.e_v + pre
            a_old = self.e_a.copy()
            z_e = psi[:, None] * (u_e - c.adaptation_strength * a_old)
        self.core.step(row, track=False)
        if track:
            # Detach reset spike, NOT theta(a): d(v_after)=du-z*beta*da.
            # Upstream constant-threshold reset would have no second term.
            self.e_v[:] = u_e - self.core.spikes[:, None] * c.adaptation_strength * a_old
            self.e_a[:] = c.adaptation_decay * a_old + z_e
            if self.gate_mode == 'activity':
                gate = 1.0 if float(np.max(np.abs(row))) > self.sensory_threshold else .15
                z_e[:, c.inputs:] *= gate
            self.e_f[:] = self.gamma_e_vec * self.e_f + (1.0 - self.gamma_e_vec) * z_e
            if not all(np.isfinite(getattr(self, n)).all() for n in ('e_v', 'e_a', 'e_f')):
                raise ValueError('Nonfinite e-prop eligibility')
        self.total_ticks += 1
        return self.core.features

    @classmethod
    def loads(cls, payload):
        """Bounded shared research adapter; schema integration owned by orchestrator."""
        import io, zipfile
        from .forensic.r8_snapshot import metadata, loads, MAX_BYTES
        from ..neural.core import Config
        if not isinstance(payload, bytes) or len(payload) > MAX_BYTES:
            raise ValueError('Snapshot bound')
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                if len(z.namelist()) != len(set(z.namelist())) or sum(i.file_size for i in z.infolist()) > MAX_BYTES or z.getinfo('metadata.json').file_size > 65536:
                    raise ValueError('Snapshot member bound')
                meta = metadata(z.read('metadata.json'))
            config = Config(**meta['config']); settings = meta['settings']
            conditions = {'learned_mtsg': 'multi_gated', 'shuffled_signal': 'multi_gated_shuffle', 'readout_only': 'multi_gated_readout_only', 'frozen_hidden': 'multi_gated_frozen'}
            return loads(cls(config, settings['eprop_variant'], conditions[settings['condition']]), payload)[0]
        except (KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise ValueError('Malformed e-prop snapshot') from exc
