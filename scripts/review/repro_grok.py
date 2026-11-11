"""Read-only reproduction of grok's slow-trace claims. Writes only to scratch output."""
import json, os, sys, tempfile, subprocess
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.dont_write_bytecode = True
G = r'C:\Users\hachimi\Downloads\model agent\scar-x-grok-round1'
sys.path.insert(0, G + r'\src'); sys.path.insert(0, G + r'\scripts')
import numpy as np
import grok_slow_trace_screen as s
from scarx.evaluation.slow_trace import SlowTraceEligibility, g1_config, order_sequences
from scarx.evaluation.tasks21 import ridge_probe

train, ytr = order_sequences(71002, 64)
dev, ydev = order_sequences(71004, 64)
conf, yconf = order_sequences(71005, 64)
gm = SlowTraceEligibility(g1_config(71001), 'frozen', 'slow')
ft, st = s.probe(gm, train); fd, sd = s.probe(gm, dev)
out = {'probe_fast': ridge_probe(ft, ytr, fd, ydev), 'probe_slow': ridge_probe(st, ytr, sd, ydev)}
# overlap check between train/dev/confirm
h = lambda a: {r.tobytes() for r in a}
out['overlap_train_conf'] = len(h(train) & h(conf)); out['overlap_dev_conf'] = len(h(dev) & h(conf))
rows = []
for seed in s.CONFIRM_SEEDS:
    r = {'seed': seed}
    for name, mode, cond in s.ARMS:
        m = s.train_one(seed, mode, cond, train, ytr)
        r[name], _ = s.score_arm(m, conf, yconf)
    rows.append(r)
out['means'] = {n: float(np.mean([r[n] for r in rows])) for n, _, _ in s.ARMS}
out['zero_eq_frozen_all_seeds'] = all(r['slow-zero'] == r['slow-frozen'] for r in rows)
out['gain_vs_frozen'] = s.interval([r['slow-correct'] - r['slow-frozen'] for r in rows])
out['gain_vs_fast'] = s.interval([r['slow-correct'] - r['fast-correct'] for r in rows])
# cold resume: dump trained model, load in a fresh process, next learn step must match warm
m = s.train_one(71037, 'slow', 'correct', train, ytr)
blob = m.dumps()
tmp = tempfile.mkdtemp(); open(tmp + '/m.bin', 'wb').write(blob)
np.save(tmp + '/x.npy', train[0])
code = (f"import sys,os;sys.dont_write_bytecode=True;sys.path.insert(0,r'{G}\\src');import numpy as np,hashlib;"
        "from scarx.evaluation.slow_trace import SlowTraceEligibility as S;"
        f"m=S.loads(open(r'{tmp}\\m.bin','rb').read());x=np.load(r'{tmp}\\x.npy');"
        "r=m.learn(x,int(sys.argv[1]));print(repr(r));print(hashlib.sha256(m.dumps()).hexdigest())")
cold = subprocess.run([sys.executable, '-c', code, str(int(ytr[0]))], capture_output=True, text=True)
warm = S = SlowTraceEligibility.loads(blob)
wr = warm.learn(train[0], int(ytr[0]))
import hashlib
out['cold_stdout'] = cold.stdout.strip().splitlines(); out['cold_stderr'] = cold.stderr[-300:]
out['warm_result'] = repr(wr); out['warm_dump_sha'] = hashlib.sha256(warm.dumps()).hexdigest()
print(json.dumps(out, indent=1, default=str))
