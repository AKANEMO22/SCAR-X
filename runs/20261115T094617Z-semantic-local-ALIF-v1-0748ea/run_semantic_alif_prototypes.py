"""One registered20-stage,3-seed local ALIF experiment with hard persistent quota."""
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import subprocess, sys, time, uuid
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.evaluation.semantic_prototype_alif import SemanticALIF, ARMS
from scarx.hybrid.store import Store, encode
from prepare_semantic_alif_features import controls, bm25


class Budget:
    def __init__(self, run):
        self.run, self.started = run, time.monotonic()
        self.counts = {'forward': 0, 'update': 0}
        self.last_resource = 0.
        self.resources = []
        self.file = (run / 'attempts.jsonl').open('x', encoding='utf-8')

    def charge(self, kind):
        if time.monotonic() - self.started > 1200:
            raise TimeoutError('20minute global wall quota')
        cap = {'forward': 40000, 'update': 4500}[kind]
        if self.counts[kind] >= cap:
            raise RuntimeError(kind + ' quota exhausted')
        self.counts[kind] += 1
        self.file.write(encode({'kind': kind, 'number': self.counts[kind], 'epoch': time.time()}) + '\n')
        self.file.flush()
        if time.monotonic() - self.last_resource > 5:
            self.resource()

    def resource(self):
        memory = psutil.Process().memory_info()
        free = int(subprocess.check_output([str(Path(os.environ['WINDIR']) / 'System32/nvidia-smi.exe'),
            '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW).strip()) * 1024**2
        row = {'epoch': time.time(), 'RSS': memory.rss,
               'private_commit': getattr(memory, 'private', None), 'GPU_free': free}
        self.resources.append(row)
        with (self.run / 'resources.jsonl').open('a', encoding='utf-8') as f:
            f.write(encode(row) + '\n')
        self.last_resource = time.monotonic()
        if memory.rss > 2_000_000_000 or free < 512 * 1024**2:
            raise MemoryError('Research worker2GB / dedicated reserve512MiB guard')


def predict(model, x):
    values = []
    for row in x:
        p, _ = model.forward(row)
        values.append(p)
    return np.stack(values)


def schedule(train, labels, seed):
    rng = np.random.default_rng(seed + 730)
    pools = {c: list(rng.permutation(train[labels[train] == c])) for c in range(6)}
    positions = [0] * 6
    reservoir, seen, stages, unique = [], 0, [], set()
    for stage in range(20):
        c = stage % 6
        current = []
        for _ in range(12):
            index = int(pools[c][positions[c] % len(pools[c])])
            positions[c] += 1
            current.append(index)
            unique.add(index)
            seen += 1
            if len(reservoir) < 64:
                reservoir.append(index)
            else:
                j = int(rng.integers(seen))
                if j < 64:
                    reservoir[j] = index
        replay = [int(reservoir[int(rng.integers(len(reservoir)))]) for _ in range(12)]
        stages.append({'stage': stage, 'current': current, 'replay_samples': replay,
                       'controller': {'stage': stage, 'replay': reservoir.copy(), 'replay_seen': seen,
                                      'rng': rng.bit_generator.state, 'unique_seen': sorted(unique)}})
    return stages


def bootstrap(difference):
    # All selected normalized TEST strings are unique (verified below), so each
    # source-text cluster has one row; resample balanced intents and seeds.
    rng = np.random.default_rng(260216)
    values = []
    for _ in range(10000):
        seeds = rng.integers(difference.shape[0], size=difference.shape[0])
        tasks = np.concatenate([c * 40 + rng.integers(40, size=40) for c in range(6)])
        values.append(difference[np.ix_(seeds, tasks)].mean())
    return [float(v) for v in np.quantile(values, [.025, .975])]


def preflight(budget, run):
    checks = []
    for seed in (260113, 260137):
        for arm in ('prototype', 'readout_only', 'shuffled'):
            cpu = SemanticALIF(seed, arm, budget, 'cpu')
            gpu = SemanticALIF(seed, arm, budget, 'gpu')
            for step in range(2):
                vector = np.eye(64, dtype=np.float32)[step]
                a, b = cpu.learn(vector, step), gpu.learn(vector, step)
                if abs(a['loss'] - b['loss']) > 3e-5:
                    raise ValueError('CPU/GPU P0 loss mismatch')
                for key, value in cpu.model.arrays().items():
                    np.testing.assert_allclose(value, gpu.model.host(gpu.model.arrays()[key]), rtol=3e-5, atol=3e-5)
                np.testing.assert_array_equal(cpu.counts, gpu.counts)
            payload = gpu.dumps({'replay': [0, 1], 'stage': 0, 'RNG': gpu.signal_rng.bit_generator.state})
            clone, controller = SemanticALIF.loads(payload, budget, 'gpu')
            assert clone.dumps(controller) == payload
            vector = np.eye(64, dtype=np.float32)[3]
            a, b = gpu.learn(vector, 3), clone.learn(vector, 3)
            assert a == b and clone.dumps() == gpu.dumps()
            rolled, _ = SemanticALIF.loads(payload, budget, 'gpu')
            p, _ = rolled.forward(vector)
            rolled2, _ = SemanticALIF.loads(payload, budget, 'gpu')
            q, _ = rolled2.forward(vector)
            np.testing.assert_array_equal(p, q)
            assert rolled.dumps() == rolled2.dumps()
            assert gpu.model.device == 'gpu' and gpu.model.core.w_in.dtype == np.float32
            assert type(gpu.model.core.w_in).__module__.startswith('cupy')
            assert np.array_equal(gpu.model.host(gpu.model.core.w_rec), gpu.initial_recurrent)
            (run / f'P0-{seed}-{arm}.npz').write_bytes(payload)
            checks.append({'seed': seed, 'arm': arm, 'numeric_parity_fixture': True,
                           'same_GPU_resume_and_rollback_exact': True})
    if budget.counts['update'] > 40 or budget.counts['forward'] > 200:
        raise RuntimeError('P0 subquota exceeded')
    return checks


def main():
    feature_dir = Path((ROOT / 'local/current-semantic-features.txt').read_text(encoding='utf-8').strip())
    gate = json.loads((feature_dir / 'result.json').read_text(encoding='utf-8'))
    if not gate['headroom_gate_pass'] or gate['TEST_scored']:
        raise ValueError('Prerequisite failed or TEST already used')
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-local-ALIF-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    with (ROOT / 'local/semantic-local-ALIF-v1-execution.json').open('x', encoding='utf-8') as f:
        json.dump({'run': str(run)}, f)
    (ROOT / 'local/current-semantic-local-ALIF.txt').write_text(str(run), encoding='utf-8')
    artifact = ROOT / 'artifacts' / run.name
    artifact.mkdir()
    rows = json.loads((feature_dir / 'data.json').read_text(encoding='utf-8'))
    with np.load(feature_dir / 'features.npz', allow_pickle=False) as f:
        x, raw, labels, train, dev = (f[n].copy() for n in ('x', 'raw', 'labels', 'train', 'dev'))
    test = np.array([i for i, r in enumerate(rows) if r['split'] == 'TEST'])
    assert len(test) == 240 and np.array_equal(labels[test], np.repeat(np.arange(6), 40))
    import re
    assert len({re.sub(r'\W+', ' ', rows[i]['text'].casefold()).strip() for i in test}) == 240
    seeds = (260113, 260137, 260171)
    plans = {seed: schedule(train, labels, seed) for seed in seeds}
    (run / 'schedules.json').write_text(encode(plans), encoding='utf-8')
    sources = ('scripts/run_semantic_alif_prototypes.py', 'src/scarx/evaluation/semantic_prototype_alif.py',
               'scripts/prepare_semantic_alif_features.py', 'experiments/semantic-alif-local-prototypes-v1.md')
    manifest = {'source_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sources},
        'features_run': feature_dir.name, 'features_sha256': gate['features_sha256'], 'seeds': seeds,
        'arms': ARMS, 'max_updates': 4500, 'max_forwards': 40000, 'max_seconds': 1200,
        'primary_control_from_DEV': gate['strongest_registered_DEV'],
        'gates': {'acquisition': .65, 'gain': .05, 'mean_forgetting': .03, 'worst_forgetting': .05,
                  'CI_lower_must_exceed': 0}, 'test_scored_once_final_only': True,
        'scope': 'six-intent module; common frozen pretrained encoder; not full assistant/RAG superiority'}
    (run / 'manifest.json').write_text(encode(manifest), encoding='utf-8')
    for name in sources:
        (run / Path(name).name).write_bytes((ROOT / name).read_bytes())
    budget = Budget(run)
    trace = Store(ROOT / 'local/hybrid-v1')
    report = {'completed': False, 'failures': [], 'main_or_live_changed': False,
              'research_goal_achieved': False, 'whole_assistant_superiority': False}
    curves, final, reset, comparator, all_losses = {}, {}, {}, {}, []
    try:
        report['P0_checks'] = preflight(budget, run)
        (run / 'P0.json').write_text(encode({'checks': report['P0_checks'], 'counts': budget.counts.copy()}), encoding='utf-8')
        trace.event('work_trace.activity', {'title': 'ALIF prototype P0 parity/save-load/rollback passed',
            'run_id': run.name, 'counts': budget.counts.copy(), 'quality_claim': False}, 'orchestrator')
        for seed in seeds:
            final[seed], reset[seed], comparator[seed], curves[seed] = {}, {}, {}, {}
            for arm in ARMS:
                model = SemanticALIF(seed, arm, budget)
                curve = []
                frozen_dev = predict(model, x[dev]) if arm == 'frozen' else None
                for stage in plans[seed]:
                    losses = []
                    if arm != 'frozen':
                        for kind, indices in (('new', stage['current']), ('replay', stage['replay_samples'])):
                            for index in indices:
                                result = model.learn(x[index], int(labels[index]))
                                all_losses.append({'seed': seed, 'arm': arm, 'stage': stage['stage'],
                                    'sample_index': index, 'id': rows[index]['id'], 'kind': kind, **result})
                                losses.append(result['loss'])
                        (artifact / f'{seed}-{arm}-stage{stage["stage"]:02d}.npz').write_bytes(model.dumps(stage['controller']))
                    probabilities = frozen_dev if frozen_dev is not None else predict(model, x[dev])
                    prediction = probabilities.argmax(axis=1)
                    scores = [float(np.mean(prediction[labels[dev] == c] == c)) for c in range(6)]
                    curve.append(scores)
                    np.save(run / f'DEV-{seed}-{arm}-stage{stage["stage"]:02d}.npy', probabilities, allow_pickle=False)
                    trace.event('work_trace.activity', {'title': 'Học ALIF và kiểm DEV thật', 'run_id': run.name,
                        'seed': seed, 'arm': arm, 'stage': stage['stage'] + 1, 'stages': 20,
                        'loss': float(np.mean(losses)) if losses else None, 'updates': model.updates,
                        'hidden_updates': model.hidden_updates, 'DEV_per_intent': scores,
                        'DEV_accuracy': float(np.mean(scores)), 'active_neurons_last_tick': model.spikes,
                        'weight_delta_L2_sum': sum(r['weight_delta_L2'] for r in all_losses[-24:]) if losses else 0,
                        'resources': budget.resources[-1], 'device': model.model.device,
                        'actual_DEV_examples': [{'question': rows[int(i)]['text'], 'model_intent': int(p),
                            'reference_intent': int(labels[i])} for i, p in zip(dev[:12], prediction[:12])],
                        'frozen_DEV_cached_exact': arm == 'frozen'}, 'orchestrator')
                curves[seed][arm] = curve
                before = model.dumps(plans[seed][-1]['controller'])
                # TEST is not used for fitting, stream, replay or arm selection.
                final[seed][arm] = predict(model, x[test])
                if arm != 'frozen':
                    clone, _ = SemanticALIF.loads(before, budget)
                    np.testing.assert_array_equal(final[seed][arm], predict(clone, x[test]))
                    clone.reset_hidden()
                    reset[seed][arm] = predict(clone, x[test])
                    # Exact next-update continuation on TRAIN clones only.
                    a, _ = SemanticALIF.loads(before, budget)
                    b, _ = SemanticALIF.loads(before, budget)
                    index = plans[seed][-1]['current'][-1]
                    assert a.learn(x[index], int(labels[index])) == b.learn(x[index], int(labels[index]))
                    assert a.dumps() == b.dumps()
                    report.setdefault('cold_resume_checks', []).append({'seed': seed, 'arm': arm, 'exact': True})
                if sum(p.stat().st_size for p in artifact.glob('*')) > 100_000_000:
                    raise MemoryError('100MB artifacts cap')
            seen = np.array(plans[seed][-1]['controller']['unique_seen'])
            assert set(seen) <= set(train)
            for space, matrix in (('full1024', raw), ('PCA64', x)):
                comparator[seed].update({space + '/' + k: v for k, v in controls(matrix[seen], labels[seen], matrix[test]).items()})
            comparator[seed]['BM25'] = bm25([rows[i]['text'] for i in seen], labels[seen], [rows[i]['text'] for i in test])
        correctness = {arm: np.stack([final[s][arm].argmax(axis=1) == labels[test] for s in seeds]) for arm in ARMS}
        comparisons = {}
        for name in comparator[seeds[0]]:
            control = np.stack([comparator[s][name] == labels[test] for s in seeds])
            diff = correctness['prototype'].astype(float) - control
            comparisons[name] = {'control_accuracy': float(control.mean()), 'gain': float(diff.mean()),
                                 'CI95': bootstrap(diff)}
        for name in ('readout_only', 'frozen', 'shuffled'):
            diff = correctness['prototype'].astype(float) - correctness[name]
            comparisons[name] = {'control_accuracy': float(correctness[name].mean()), 'gain': float(diff.mean()),
                                 'CI95': bootstrap(diff)}
        difference = correctness['prototype'].astype(float) - np.stack([reset[s]['prototype'].argmax(axis=1) == labels[test] for s in seeds])
        comparisons['hidden_reset'] = {'control_accuracy': float((correctness['prototype'] - difference).mean()),
                                      'gain': float(difference.mean()), 'CI95': bootstrap(difference)}
        forgetting = {}
        for arm in ARMS:
            by_seed = []
            for seed in seeds:
                curve = np.array(curves[seed][arm])
                forgotten = np.array([max(0, curve[c:, c].max() - curve[-1, c]) for c in range(6)])
                by_seed.append({'seed': seed, 'mean': float(forgotten.mean()), 'worst': float(forgotten.max()),
                                'per_intent': forgotten.tolist(), 'final_DEV_accuracy': float(curve[-1].mean())})
            forgetting[arm] = by_seed
        acquired = float(correctness['prototype'].mean()) >= .65
        retained = all(r['mean'] <= .03 + 1e-9 and r['worst'] <= .05 + 1e-9 for r in forgetting['prototype'])
        gains = all(r['gain'] >= .05 and r['CI95'][0] > 0 for r in comparisons.values())
        if time.monotonic() - budget.started > 1200:
            raise TimeoutError('Global wall quota including final statistics')
        budget.resource()
        report.update(completed=True, verdict='MODULE_GATES_PASS_NOT_WHOLE_ASSISTANT' if acquired and retained and gains else 'LOCAL_ALIF_MODULE_GATES_NOT_MET',
            TEST_accuracy={a: float(v.mean()) for a, v in correctness.items()}, comparisons=comparisons,
            DEV_forgetting20=forgetting, acquisition_gate=acquired, retention_gate=retained,
            all_control_and_attribution_gain_gate=gains, research_goal_achieved=False,
            module_gates_pass=acquired and retained and gains,
            unique_TRAIN_seen={str(s): len(plans[s][-1]['controller']['unique_seen']) for s in seeds})
    except Exception as exc:
        report['failures'].append(type(exc).__name__ + ': ' + str(exc))
        report['verdict'] = 'SEMANTIC_LOCAL_ALIF_STOPPED_NOT_PASS'
    finally:
        (run / 'losses.jsonl').write_text(''.join(encode(r) + '\n' for r in all_losses), encoding='utf-8')
        (run / 'curves.json').write_text(encode(curves), encoding='utf-8')
        arrays = {f'{seed}_{arm}': p for seed, by_arm in final.items() for arm, p in by_arm.items()}
        arrays.update({f'{seed}_reset_{arm}': p for seed, by_arm in reset.items() for arm, p in by_arm.items()})
        arrays.update({f'{seed}_control_{name.replace("/", "_")}': p for seed, names in comparator.items() for name, p in names.items()})
        np.savez(run / 'predictions.npz', **arrays)
        report.update(charged_counts=budget.counts, elapsed_seconds=time.monotonic() - budget.started,
            peak_sampled_worker_RSS=max((r['RSS'] for r in budget.resources), default=None),
            min_sampled_GPU_free=min((r['GPU_free'] for r in budget.resources), default=None),
            artifacts=str(artifact.relative_to(ROOT)))
        (run / 'result.json').write_text(encode(report), encoding='utf-8')
        trace.event('work_trace.activity', {'title': report['verdict'], 'run_id': run.name, **report}, 'orchestrator')
        trace.close()
        budget.file.close()
    print(encode({'run': str(run), **report}), flush=True)
    if not report['completed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
