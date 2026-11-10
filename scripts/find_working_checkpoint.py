"""Which checkpoint still answers correctly? Concrete restoration evidence.

Round 14 established that the SHIPPED weights answer all four sample prompts
correctly while the current live session does not. The live database holds 108
checkpoints, so the question becomes concrete and answerable:

    which is the most recent checkpoint that still answers correctly?

This reads each checkpoint read-only and scores the four shipped prompts from a
reset state (the reproducible convention). Nothing is written to the session.
The goal is to give the user a concrete restore target rather than a vague claim.
"""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.neural.unified import UnifiedCore  # noqa: E402

EXPECT = {
    'hi': 'Hello.',
    'chào bạn': 'Chào bạn.',
    'bạn là ai': 'Tôi là scar-x.',
    'tạm biệt': 'Tạm biệt bạn.',
}

DB = ROOT / 'local' / 'generation1' / 'generation1.sqlite'
db = sqlite3.connect(f'file:{DB.as_posix()}?mode=ro', uri=True)
rows = db.execute('SELECT id, created, action, sha, payload FROM checkpoints ORDER BY created').fetchall()
print(f'checkpoints: {len(rows)}')


def score(payload):
    """Checkpoint rows store a full session archive, not a bare model blob.

    The row starts with 'PK' and contains 'model.zip', so it must be opened the
    same way Generation._unpack does. Calling UnifiedCore.loads directly on the
    whole payload always fails with 'Invalid generation checkpoint', which is what
    produced an empty score table on the first attempt.
    """
    import hashlib
    import io
    import zipfile
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = set(archive.namelist())
            if 'model.zip' in names:
                data = archive.read('model.zip')
                meta = json.loads(archive.read('session.json')) if 'session.json' in names else {}
                if meta.get('sha256') and hashlib.sha256(data).hexdigest() != meta['sha256']:
                    return None, 'checksum'
            elif 'metadata.json' in names or 'core.npz' in names:
                data = payload
            else:
                return None, f'members:{sorted(names)[:3]}'
        model = UnifiedCore.loads(data)
    except Exception as exc:
        return None, f'load:{type(exc).__name__}'
    model.core.reset_state()
    model.last_active[:] = model.ticks
    good = 0
    texts = {}
    for prompt, expect in EXPECT.items():
        try:
            text = model.generate(prompt)['text']
        except Exception:
            text = ''
        texts[prompt] = text
        norm = ' '.join(text.split()).strip(' .!?').lower()
        if norm == ' '.join(expect.split()).strip(' .!?').lower():
            good += 1
    return good, texts


results = []
for cid, created, action, sha, payload in rows:
    good, info = score(payload)
    results.append({'id': cid[:12], 'created': created, 'action': action,
                    'sha': sha[:16] if sha else None, 'good': good,
                    'texts': info if isinstance(info, dict) else None,
                    'error': info if isinstance(info, str) else None})

scored = [r for r in results if r['good'] is not None]
print(f'scored: {len(scored)}')
dist = {}
for r in scored:
    dist[r['good']] = dist.get(r['good'], 0) + 1
print('distribution of correct answers out of 4:', dict(sorted(dist.items(), reverse=True)))
print()

perfect = [r for r in scored if r['good'] == 4]
partial = [r for r in scored if r['good'] == 3]
print(f'checkpoints with 4/4 correct : {len(perfect)}')
print(f'checkpoints with 3/4 correct : {len(partial)}')

if perfect:
    last = perfect[-1]
    print(f"\nMOST RECENT PERFECT checkpoint:")
    print(f"  id      : {last['id']}")
    print(f"  created : {last['created']}")
    print(f"  action  : {last['action']}")
    print(f"  sha     : {last['sha']}")
    print(f"  answers :")
    for p, t in last['texts'].items():
        print(f"     {p!r:12s} -> {t!r}")
else:
    print('\nno checkpoint in this session answers all four correctly')

latest = scored[-1] if scored else None
if latest:
    print(f"\nCURRENT (most recent) checkpoint: {latest['id']} good={latest['good']}/4")
    if latest['texts']:
        for p, t in latest['texts'].items():
            print(f"     {p!r:12s} -> {t!r}")

out = ROOT / 'local' / 'checkpoint-quality.json'
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'\nwrote {out}')
