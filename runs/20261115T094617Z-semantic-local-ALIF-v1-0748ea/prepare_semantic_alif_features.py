"""TRAIN-only PCA and fixed DEV baselines; never select or score TEST here."""
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, re, sys, uuid
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.hybrid.store import Store, encode


def normalized(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), np.float32(1e-8))


def controls(train_x, train_y, query_x):
    similarity = query_x @ train_x.T
    nearest = np.argsort(-similarity, axis=1, kind='stable')
    result = {'1NN': train_y[nearest[:, 0]],
              '5NN': np.array([np.argmax(np.bincount(train_y[ids[:5]], minlength=6)) for ids in nearest])}
    centroids = normalized(np.stack([train_x[train_y == i].mean(axis=0) for i in range(6)]))
    result['centroid'] = np.argmax(query_x @ centroids.T, axis=1)
    # Fixed regularization, no DEV search. Primal/dual same ridge hypothesis.
    dual = np.linalg.solve(train_x @ train_x.T + np.eye(len(train_x), dtype=np.float32),
                           np.eye(6, dtype=np.float32)[train_y])
    result['ridge_alpha1'] = np.argmax(query_x @ train_x.T @ dual, axis=1)
    return result


def bm25(train_texts, train_y, query_texts):
    tokens = lambda text: re.findall(r'[^\W_]+', text.casefold(), re.UNICODE)
    docs = [Counter(tokens(text)) for text in train_texts]
    lengths = np.array([sum(d.values()) for d in docs], dtype=np.float32)
    average = lengths.mean()
    df = Counter(t for d in docs for t in d)
    idf = {t: np.log(1 + (len(docs) - n + .5) / (n + .5)) for t, n in df.items()}
    result = []
    for query in query_texts:
        scores = np.zeros(len(docs), dtype=np.float32)
        for term in set(tokens(query)):
            tf = np.array([d[term] for d in docs], dtype=np.float32)
            scores += idf.get(term, 0) * tf * 2.5 / (tf + 1.5 * (.25 + .75 * lengths / average))
        result.append(train_y[int(np.argmax(scores))])
    return np.asarray(result)


def main():
    encoder = Path((ROOT / 'local/current-semantic-encoder.txt').read_text(encoding='utf-8').strip())
    receipt = json.loads((encoder / 'result.json').read_text(encoding='utf-8'))
    if not receipt['completed']:
        raise ValueError('Encoder prerequisite failed; no features/learning allowed')
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-feature-gate-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    with (ROOT / 'local/semantic-feature-gate-v1-execution.json').open('x', encoding='utf-8') as f:
        json.dump({'run': str(run)}, f)
    rows = json.loads((encoder / 'data.json').read_text(encoding='utf-8'))
    raw = np.load(encoder / 'vectors.npy', allow_pickle=False)
    assert raw.dtype == np.float32 and raw.shape == (600, 1024) and np.isfinite(raw).all()
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'TRAIN'])
    dev = np.array([i for i, r in enumerate(rows) if r['split'] == 'DEV'])
    labels = np.array([r['label'] for r in rows], dtype=np.int32)
    raw = normalized(raw)
    mean = raw[train].mean(axis=0)
    _, singular, basis = np.linalg.svd(raw[train] - mean, full_matrices=False)
    basis = basis[:64].T.copy()
    x = normalized((raw - mean) @ basis)
    predictions = {}
    for name, matrix in (('full1024', raw), ('PCA64', x)):
        for control, value in controls(matrix[train], labels[train], matrix[dev]).items():
            predictions[name + '/' + control] = value
    predictions['BM25'] = bm25([rows[i]['text'] for i in train], labels[train], [rows[i]['text'] for i in dev])
    scores = {k: float(np.mean(v == labels[dev])) for k, v in predictions.items()}
    strongest = max(scores, key=scores.get)
    result = {'encoder_run': encoder.name, 'DEV_scores': scores, 'strongest_registered_DEV': strongest,
              'headroom_gate_pass': scores[strongest] <= .95, 'TEST_scored': False,
              'SNN_updates': 0, 'full_system_superiority': False,
              'PCA_fit_split': 'TRAIN only240', 'PCA_retained_variance_fraction':
                  float(np.sum(singular[:64] ** 2) / np.sum(singular ** 2)),
              'codec': 'train-centered-nativeQwen1024-PCA64-unitFP32-v1'}
    np.savez(run / 'features.npz', raw=raw, x=x, mean=mean, basis=basis,
             labels=labels, train=train, dev=dev, **{'prediction_' + k.replace('/', '_'): v for k, v in predictions.items()})
    (run / 'data.json').write_bytes((encoder / 'data.json').read_bytes())
    result['features_sha256'] = hashlib.sha256((run / 'features.npz').read_bytes()).hexdigest()
    (run / 'result.json').write_text(encode(result), encoding='utf-8')
    (run / 'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT / 'local/current-semantic-features.txt').write_text(str(run), encoding='utf-8')
    trace = Store(ROOT / 'local/hybrid-v1')
    trace.event('work_trace.activity', {'title': 'Đối chứng DEV trước khi học ALIF', 'run_id': run.name, **result}, 'orchestrator')
    trace.close()
    print(encode({'run': str(run), **result}), flush=True)


if __name__ == '__main__':
    main()
