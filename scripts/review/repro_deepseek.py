"""Read-only reproduction of deepseek pair_binding claims (imports its code; writes nothing in its tree)."""
import os, sys, json, hashlib, subprocess, tempfile
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.dont_write_bytecode = True
D = r'C:\Users\hachimi\Downloads\model agent\scar-x-deepseek-round1'
sys.path.insert(0, D + r'\src')
import numpy as np
from scarx.evaluation.pair_binding import order2_features, order3_binding, order3_features, pair_binding
from scarx.evaluation.tasks21 import ridge_probe, representation
from scarx.neural.core import Config
from scarx.neural.plasticity import PlasticCore

out = {}
# positive: strict_binding needs order-3
fx, fy = order3_binding(31001, 400); tx, ty = order3_binding(31901, 400)
out['strict_o1'] = ridge_probe(fx.reshape(len(fx), -1), fy, tx.reshape(len(tx), -1), ty)
out['strict_o2'] = ridge_probe(order2_features(fx), fy, order2_features(tx), ty)
out['strict_o3'] = ridge_probe(order3_features(fx), fy, order3_features(tx), ty)
fx, fy = pair_binding(31001, 500); tx, ty = pair_binding(31901, 500)
out['pair_o1'] = ridge_probe(fx.reshape(len(fx), -1), fy, tx.reshape(len(tx), -1), ty)
out['pair_o2'] = ridge_probe(order2_features(fx), fy, order2_features(tx), ty)
# negative control: score vs frozen on pair_binding, 1 seed, lr .025, 2048 train (as protocol)
def cfg(seed, lr):
    return Config(inputs=8, hidden=48, outputs=2, fan_in=8, plastic_rows=12, seed=seed, leak=.65, threshold=.5,
                  feature_decay=.25, adaptation_strength=.02, readout_lr=lr, hidden_lr=.002, anchor_strength=0.)
res = {}
for arm in ('score', 'frozen'):
    m = PlasticCore(cfg(41001, .025), arm)
    x, y = pair_binding(41001 + 6_300_000, 2048)
    for s_, l in zip(x, y):
        m.forward(s_, track=True); m.learn(int(l))
    tx, ty = pair_binding(41001 + 6_500_000, 256)
    res[arm] = float(np.mean(m.predict_many(tx, 903) == ty))
    if arm == 'score':
        blob = m.dumps(); sm = m
out['direct_seed41001_lr025'] = res
# cold resume
tmp = tempfile.mkdtemp(); open(tmp + r'\m.npz', 'wb').write(blob)
np.save(tmp + r'\x.npy', x[0])
code = (f"import sys;sys.dont_write_bytecode=True;sys.path.insert(0,r'{D}\\src');import numpy as np,hashlib;"
        "from scarx.neural.plasticity import PlasticCore as P;"
        f"m=P.loads(open(r'{tmp}\\m.npz','rb').read());m.forward(np.load(r'{tmp}\\x.npy'),track=True);m.learn(int(sys.argv[1]));"
        "print(json.dumps({k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in m.arrays().items()}) if False else "
        "__import__('json').dumps({k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in m.arrays().items()}))")
cold = subprocess.run([sys.executable, '-c', code, str(int(y[0]))], capture_output=True, text=True)
w = PlasticCore.loads(blob); w.forward(x[0], track=True); w.learn(int(y[0]))
warm = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in w.arrays().items()}
try:
    out['cold_resume_match'] = json.loads(cold.stdout.strip().splitlines()[-1]) == warm
except Exception:
    out['cold_resume_match'] = None; out['cold_err'] = cold.stderr[-400:]
print(json.dumps(out, indent=1))
