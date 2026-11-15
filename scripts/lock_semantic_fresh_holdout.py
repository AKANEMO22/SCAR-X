"""Reserve unseen source utterances before any future confirmation inference."""
from datetime import datetime, timezone
import csv, hashlib, io, json, random, re, sys, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.hybrid.store import Store, encode


def main():
    original = Path((ROOT / 'local/current-semantic-ALIF-data.txt').read_text(encoding='utf-8').strip())
    receipt = json.loads((original / 'result.json').read_text(encoding='utf-8'))
    raw_dir = Path(receipt['raw_source_directory'])
    raw = (raw_dir / 'banking_data/train.csv').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == receipt['sources']['banking_data/train.csv']['sha256']
    normalize = lambda text: re.sub(r'\W+', ' ', text.casefold()).strip()
    previous = json.loads((original / 'selected-data.json').read_text(encoding='utf-8'))
    seen = {normalize(r['text']) for r in previous}
    source_rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    selected, available = [], {}
    for label, intent in enumerate(receipt['labels']):
        pool, unique = [], set()
        for index, row in enumerate(source_rows):
            key = normalize(row['text'])
            if row['category'] == intent and key not in seen and key not in unique:
                pool.append((index, row))
                unique.add(key)
        available[intent] = len(pool)
        if len(pool) < 40:
            raise ValueError('Insufficient fresh utterances; do not change category/sample target')
        random.Random(280216 + label).shuffle(pool)
        for index, (source_index, row) in enumerate(pool[:40]):
            key = normalize(row['text'])
            assert key not in seen
            selected.append({'id': f'FRESH-{label}-{index}', 'split': 'FRESH_HELDOUT',
                'source_split': 'official_train_previously_unused', 'source_row_index': source_index,
                'label': label, 'intent': intent, 'text': row['text'].strip()})
            seen.add(key)
    assert len(selected) == 240 and len({normalize(r['text']) for r in selected}) == 240
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-semantic-fresh-holdout-lock-v1-' + uuid.uuid4().hex[:6])
    run.mkdir()
    with (ROOT / 'local/semantic-fresh-holdout-lock-v1-execution.json').open('x', encoding='utf-8') as f:
        json.dump({'run': str(run)}, f)
    raw_selected = encode(selected).encode()
    (run / 'locked-rows.json').write_bytes(raw_selected)
    report = {'verdict': 'FRESH_HUMAN_UTTERANCES_LOCKED_BEFORE_MODEL_INFERENCE',
        'source_pin': receipt['source_commit'], 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'prior_selected_run': original.name, 'prior_selected_excluded': len(previous),
        'fresh_rows': 240, 'fresh_normalized_unique': 240, 'available_by_intent_before_selection': available,
        'seed': 280216, 'per_intent': 40, 'locked_sha256': hashlib.sha256(raw_selected).hexdigest(),
        'label_or_category_selection_on_scores': False, 'model_or_learning_calls': 0,
        'not_official_Banking77_test': True, 'pretrained_backbone_contamination_not_ruled_out': True,
        'not_allowed_in_TRAIN_DEV_or_parameter_selection': True, 'source_license': receipt['source_license'],
        'attribution': receipt['attribution'], 'requires_qualified_frozen_protocol_before_inference': True}
    (run / 'result.json').write_text(encode(report), encoding='utf-8')
    (run / 'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT / 'local/current-semantic-fresh-holdout.txt').write_text(str(run), encoding='utf-8')
    trace = Store(ROOT / 'local/hybrid-v1')
    trace.event('work_trace.activity', {'title': 'Khóa tập kiểm thử mới, chưa gọi model', 'run_id': run.name,
        **report}, 'orchestrator')
    trace.close()
    print(encode({'run': str(run), **report}), flush=True)


if __name__ == '__main__':
    main()
