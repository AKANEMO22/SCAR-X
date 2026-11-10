import numpy as np
import pytest

from scarx.evaluation.readout_eligibility import ReadoutEligibility
from scarx.evaluation.slow_trace import SlowTraceEligibility, g1_config, order_sequences


def calibrated(mode, condition='correct', seed=3):
    model = SlowTraceEligibility(g1_config(seed), condition, mode)
    values, _labels = order_sequences(71003, 128)
    model.calibrate(values)
    return model


def test_fast_forward_matches_the_existing_readout_eligibility():
    config = g1_config(5)
    left = SlowTraceEligibility(config, 'correct', 'fast')
    right = ReadoutEligibility(config, 'correct')
    values, _labels = order_sequences(71002, 4)
    for row in values:
        np.testing.assert_array_equal(left.features(row), right.features(row))


def test_prediction_and_frozen_learning_do_not_change_hidden_weights():
    model = calibrated('slow', 'frozen')
    before = model.core.w_in.copy()
    values, labels = order_sequences(71002, 1)
    assert model.predict(values[0]) in (0, 1)
    model.learn(values[0], int(labels[0]))
    np.testing.assert_array_equal(model.core.w_in, before)


def test_failed_update_restores_weights_trace_and_rng():
    model = calibrated('slow', 'correct')
    values, labels = order_sequences(71002, 1)
    before = model.dumps()
    weights = model.core.w_in.copy()
    slow = model.slow.copy()
    rng = model.signal_rng.bit_generator.state

    def fail():
        model.core.w_in += 1
        model.slow += 1
        raise RuntimeError('injected')

    model.finish = fail
    with pytest.raises(RuntimeError, match='injected'):
        model.learn(values[0], int(labels[0]))
    np.testing.assert_array_equal(model.core.w_in, weights)
    np.testing.assert_array_equal(model.slow, slow)
    assert model.signal_rng.bit_generator.state == rng
    assert model.dumps() == before


def test_memory_accounting_includes_the_slow_arrays():
    base = ReadoutEligibility(g1_config(3), 'correct')
    model = SlowTraceEligibility(g1_config(3), 'correct', 'slow')
    extra = model.slow.nbytes + model.e_s.nbytes
    assert extra == 3264
    assert model.memory_bytes() == base.memory_bytes() + extra


def test_loader_rejects_duplicate_members_and_non_float32_without_mutation():
    import io
    import json
    import zipfile
    model = calibrated('slow', 'correct', seed=9)
    saved = model.dumps()
    before = model.core.w_in.copy()
    with zipfile.ZipFile(io.BytesIO(saved)) as archive:
        blobs = {name: archive.read(name) for name in archive.namelist()}
    duplicate = io.BytesIO()
    with zipfile.ZipFile(duplicate, 'w') as archive:
        for name, payload in blobs.items():
            archive.writestr(name, payload)
        archive.writestr('extra.npz', blobs['extra.npz'])
    with pytest.raises(ValueError, match='members'):
        SlowTraceEligibility.loads(duplicate.getvalue())
    with np.load(io.BytesIO(blobs['extra.npz']), allow_pickle=False) as arrays:
        slow = arrays['slow'].astype(np.float64)
        eligibility = arrays['e_s'].astype(np.float64)
    slow[0] = np.nextafter(np.float64(0.25), np.float64(1))
    extra = io.BytesIO()
    np.savez(extra, slow=slow, e_s=eligibility)
    meta = json.loads(blobs['metadata.json'])
    meta.pop('metadata_sha256')
    meta['extra_sha256'] = __import__('hashlib').sha256(extra.getvalue()).hexdigest()
    meta['metadata_sha256'] = __import__('hashlib').sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    foreign = io.BytesIO()
    with zipfile.ZipFile(foreign, 'w') as archive:
        archive.writestr('base.zip', blobs['base.zip'])
        archive.writestr('extra.npz', extra.getvalue())
        archive.writestr('metadata.json', json.dumps(meta))
    with pytest.raises(ValueError, match='dtype'):
        SlowTraceEligibility.loads(foreign.getvalue())
    np.testing.assert_array_equal(model.core.w_in, before)


def test_checkpoint_roundtrip_and_foreign_format_rejected():
    model = calibrated('slow', 'correct', seed=9)
    values, _labels = order_sequences(71004, 2)
    model.slow[:] = 0.25
    model.e_s[:] = 0.125
    saved = model.dumps()
    restored = SlowTraceEligibility.loads(saved)
    np.testing.assert_array_equal(restored.slow, model.slow)
    np.testing.assert_array_equal(restored.e_s, model.e_s)
    np.testing.assert_array_equal(restored.core.w_in, model.core.w_in)
    assert restored.predict(values[0]) == model.predict(values[0])
    first = SlowTraceEligibility.loads(saved)
    second = SlowTraceEligibility.loads(saved)
    train, labels = order_sequences(71002, 1)
    assert first.learn(train[0], int(labels[0])) == second.learn(train[0], int(labels[0]))
    foreign = ReadoutEligibility(g1_config(9), 'correct').dumps()
    with pytest.raises(ValueError, match='members'):
        SlowTraceEligibility.loads(foreign)


def _replace_extra(saved, extra):
    import io, json, zipfile, hashlib
    with zipfile.ZipFile(io.BytesIO(saved)) as archive:
        blobs = {name: archive.read(name) for name in archive.namelist()}
    meta = json.loads(blobs['metadata.json']); meta.pop('metadata_sha256')
    meta['extra_sha256'] = hashlib.sha256(extra).hexdigest()
    meta['metadata_sha256'] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    result = io.BytesIO()
    with zipfile.ZipFile(result, 'w') as archive:
        archive.writestr('base.zip', blobs['base.zip'])
        archive.writestr('extra.npz', extra)
        archive.writestr('metadata.json', json.dumps(meta))
    return result.getvalue()


def test_loader_rejects_inner_duplicate_members():
    import io, zipfile, warnings
    model = SlowTraceEligibility(g1_config(9)); saved = model.dumps()
    with zipfile.ZipFile(io.BytesIO(saved)) as archive: extra = archive.read('extra.npz')
    with zipfile.ZipFile(io.BytesIO(extra)) as archive: arrays = {n: archive.read(n) for n in archive.namelist()}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in arrays.items(): archive.writestr(name, raw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); archive.writestr('slow.npy', arrays['slow.npy'])
    with pytest.raises(ValueError, match='extra members'):
        SlowTraceEligibility.loads(_replace_extra(saved, stream.getvalue()))
    assert model.dumps() == saved


def test_loader_checks_declared_shape_before_array_allocation(monkeypatch):
    import io, zipfile
    model = SlowTraceEligibility(g1_config(9)); saved = model.dumps()
    with zipfile.ZipFile(io.BytesIO(saved)) as archive: extra = archive.read('extra.npz')
    with zipfile.ZipFile(io.BytesIO(extra)) as archive: es = archive.read('e_s.npy')
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': '<f4', 'fortran_order': False, 'shape': (1_000_000_000,)})
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('slow.npy', header.getvalue()); archive.writestr('e_s.npy', es)
    malformed = stream.getvalue(); original = np.load
    def guarded_load(file, *args, **kwargs):
        if isinstance(file, io.BytesIO) and file.getvalue() == malformed:
            raise AssertionError('Malformed extra must be rejected before np.load can allocate')
        return original(file, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded_load)
    with pytest.raises(ValueError, match='extra shape'):
        SlowTraceEligibility.loads(_replace_extra(saved, malformed))
    assert model.dumps() == saved
