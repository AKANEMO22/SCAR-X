import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scarx.neural.core import Config
from scarx.evaluation.literature_eprop import (
    LiteratureEPropALIF as EProp, published_spike_slope, upstream_trace_scan, VARIANTS,
)
from scarx.evaluation.r8_matched_credit import MatchedLocalCredit
from scarx.evaluation.r8_stopped_reset import StoppedResetCredit
from scarx.evaluation.readout_eligibility import spike_slope


def config(beta=.08):
    return Config(inputs=8, outputs=2, hidden=48, fan_in=8, plastic_rows=12, seed=6291, adaptation_strength=beta)


def fixture():
    # Synthetic technical fixture, not a task dataset or DEV score.
    return np.random.default_rng(6292).uniform(-.15, 1.1, (4, 19, 8)).astype(np.float32)


def test_execute_pinned_upstream_trace_function_without_tensorflow():
    """Execute the actual unmodified AST method with a NumPy primitive shim.

    This verifies its hardcoded recurrence, not TF/autodiff/BPTT parity.
    """
    root = Path(__file__).resolve().parents[1]
    path = root.parent.parent / 'references/eligibility_propagation/Figure_3_and_S7_e_prop_tutorials/models.py'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '53360882ef676442fac80311d0a48769cfa1e979ff68bb87e2f095dcb90a3c3b'
    tree = ast.parse(raw)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EligALIF')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_eligibility_traces')

    def scan(fn, elems, initializer):
        count = len(next(iter(elems.values()))) if isinstance(elems, dict) else len(elems)
        state = initializer; results = []
        for i in range(count):
            row = {k: v[i] for k, v in elems.items()} if isinstance(elems, dict) else elems[i]
            state = fn(state, row); results.append(np.array(state, copy=True))
        return np.stack(results)

    tf = SimpleNamespace(shape=np.shape, transpose=lambda x,perm:np.transpose(x,axes=perm), maximum=np.maximum,
        where=np.where, ones=np.ones, zeros_like=np.zeros_like, ones_like=np.ones_like,
        concat=lambda xs, axis: np.concatenate([np.asarray(x) for x in xs], axis=axis),
        scan=scan, int32=np.int32, abs=np.abs, eye=np.eye)
    scope = {'tf': tf, 'shift_by_one_time_step': lambda x: np.concatenate((np.zeros_like(x[:, :1]), x[:, :-1]), axis=1)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    # Upstream stores membrane decay as a neuron-width tensor, not a scalar.
    obj = SimpleNamespace(decay_b=.95, beta=.08, _decay=np.full(4,.85), n_refractory=0, dampening_factor=.3, thr=.65)
    rng = np.random.default_rng(6293)
    presyn = rng.normal(0, .3, (1, 9, 3)); scaled = rng.uniform(-1.4, 1.4, (1, 9, 4))
    zpost = rng.integers(2, size=(1, 9, 4)).astype(np.float64)
    actual, ev, ea, psi = scope['compute_eligibility_traces'](obj, scaled, presyn, zpost, False)
    pre = np.broadcast_to(presyn[0, :, None, :], (9, 4, 3))
    reference = upstream_trace_scan(pre, psi[0], .85, .95, .08)
    for tensor, expected in zip((actual, ev, ea), reference[:3]):
        np.testing.assert_allclose(tensor[0].transpose(0, 2, 1), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(psi[0], published_spike_slope(scaled[0] * .65, .65), rtol=1e-14, atol=1e-14)


def test_adaptive_threshold_reset_jacobian_finite_difference():
    # The fixed hard spike controls reset; only adaptation's spike copy has
    # the prescribed local surrogate. No reverse traversal or learned dataset.
    leak, rho, beta, threshold, weight, pre = .85, .95, .08, .65, .4, .7
    v, a = .8, 1.1
    u = leak * v + weight * pre; theta = threshold + beta * a
    z = float(u >= theta); psi = float(spike_slope(np.array(u - theta)))
    assert z == 1

    def detached_graph(vp, ap, wp):
        up = leak * vp + wp * pre; tp = threshold + beta * ap
        z_local = z + psi * ((up - tp) - (u - theta))
        return np.array([up - z * tp, rho * ap + z_local])

    h = 1e-6
    jac = np.column_stack(((detached_graph(v + h, a, weight) - detached_graph(v - h, a, weight)) / (2*h),
                          (detached_graph(v, a + h, weight) - detached_graph(v, a - h, weight)) / (2*h)))
    injection = (detached_graph(v, a, weight + h) - detached_graph(v, a, weight - h)) / (2*h)
    np.testing.assert_allclose(jac, [[leak, -beta*z], [psi*leak, rho-beta*psi]], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(injection, [pre, psi*pre], rtol=1e-8, atol=1e-9)
    previous = np.array([.23, -.31])
    ue = leak*previous[0] + pre; ze = psi*(ue-beta*previous[1])
    np.testing.assert_allclose(jac@previous+injection, [ue-z*beta*previous[1], rho*previous[1]+ze], atol=1e-9)


@pytest.mark.parametrize('variant', VARIANTS)
def test_scalar_recurrence_preserves_forward_and_one_call(variant, monkeypatch):
    m = EProp(config(), variant); baseline = MatchedLocalCredit(config())
    calls = []; original = m.core.step
    def counted(*args, **kwargs):
        calls.append(1); return original(*args, **kwargs)
    monkeypatch.setattr(m.core, 'step', counted)
    ev = np.zeros((48, 16), np.float64); ea = ev.copy(); ef = ev.copy()
    for row in fixture()[0]:
        c = m.config; oldz = m.core.spikes.copy(); olda = m.core.adaptation.copy()
        theta = c.threshold + c.adaptation_strength*olda
        u = (c.leak*m.core.voltage + np.sum(m.core.w_rec*oldz[m.core.indices], axis=1)) + m.core.w_in@row
        psi = spike_slope(u-theta).astype(np.float64)
        pre = np.concatenate((np.broadcast_to(row, (48, 8)), oldz[m.core.indices]), axis=1)
        m.step(row); baseline.step(row)
        for key in m.core.arrays():
            np.testing.assert_array_equal(m.core.arrays()[key], baseline.core.arrays()[key])
        ue = .85*ev+pre; ze = psi[:, None]*(ue-.08*ea)
        ev = ue-m.core.spikes[:, None]*.08*ea; ea = .95*ea+ze
        if variant == 'eprop_matched':ze[:, 8:] *= 1. if np.max(np.abs(row)) > .6 else .15
        gamma = m.gamma_e_vec.astype(np.float64); ef = gamma*ef+(1-gamma)*ze
        for name, expected in [('e_v', ev), ('e_a', ea), ('e_f', ef)]:
            np.testing.assert_allclose(getattr(m, name), expected, rtol=1e-4, atol=5e-6)
    assert len(calls) == len(fixture()[0])


def test_beta_zero_reproduces_stopped_baseline_exactly():
    a = EProp(config(0)); b = StoppedResetCredit(config(0))
    xs = fixture(); a.calibrate(xs[:1]); b.calibrate(xs[:1])
    for i, x in enumerate(xs):
        assert a.learn_episode(x, i % 2) == b.learn_episode(x, i % 2)
        for key in a.core.arrays():np.testing.assert_array_equal(a.core.arrays()[key], b.core.arrays()[key])
        for key in a.state_array_names:np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        assert a.signal_rng.bit_generator.state == b.signal_rng.bit_generator.state
        assert a.core.updates == b.core.updates and a.total_ticks == b.total_ticks


def test_reset_path_distinct_from_own_and_stopped_with_adaptation():
    models = [EProp(config()), MatchedLocalCredit(config()), StoppedResetCredit(config())]
    for row in fixture()[0]:
        for m in models:m.step(row)
    assert not np.array_equal(models[0].e_v, models[1].e_v)
    assert not np.array_equal(models[0].e_v, models[2].e_v)
    assert not np.array_equal(models[0].e_f, models[2].e_f)


def test_optimizer_readout_unchanged_at_identical_trace(monkeypatch):
    # Copy only eligibility after each real unchanged forward. The inherited
    # learner must then perform bitwise identical readout and hidden updates.
    a = EProp(config()); b = MatchedLocalCredit(config()); a.calibrated=b.calibrated=True
    original = b.step
    def same_trace(row, track=True):
        features = original(row, track)
        if track:
            b.e_v[:] = a.e_v; b.e_a[:] = a.e_a; b.e_f[:] = a.e_f
        return features
    monkeypatch.setattr(b, 'step', same_trace)
    # a is run first, so b's last trace is the same endpoint used by a.
    x = fixture()[0]
    loss_a = a.learn_episode(x, 1); loss_b = b.learn_episode(x, 1)
    assert loss_a == loss_b
    for key in a.core.arrays():np.testing.assert_array_equal(a.core.arrays()[key], b.core.arrays()[key])
    for key in ('syn_trace', 'syn_energy', 'f_mean', 'f_var'):
        np.testing.assert_array_equal(getattr(a,key), getattr(b,key))
    assert a.signal_rng.bit_generator.state == b.signal_rng.bit_generator.state


def test_no_extra_array_state_and_aligned_filter():
    a = EProp(config(), 'eprop_aligned'); b = MatchedLocalCredit(config())
    assert set(vars(a)) == set(vars(b)) | {'eprop_variant'}
    assert a.memory_bytes() == b.memory_bytes()
    np.testing.assert_array_equal(a.gamma_e_vec, np.full((48,1), config().feature_decay, np.float32))
    assert a.gate_mode == 'none'
    slope = published_spike_slope(np.array([-1.3,-.65,0,.65,1.3]), .65)
    np.testing.assert_allclose(slope, [0,0,.3/.65,0,0], atol=1e-15)


@pytest.mark.parametrize('variant', ['other', None, 'upstream_triangular'])
def test_unregistered_variant_rejected(variant):
    with pytest.raises(ValueError):EProp(config(), variant)
