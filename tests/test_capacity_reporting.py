"""Real-data regressions: budget pairing and capacity-label interpretation."""
import csv,json
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]/'runs/20261112T132737Z-capacity-dev-v1-160b6e'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def test_export_uses_same_height_and_budget_frozen_probe():
    rows=[read(p) for p in (RUN/'jobs').glob('*/result.json')]+[read(p) for p in (RUN/'reused-h48').glob('*/capacity-result.json')]
    index={(r['height'],r['arm'],r['task'],r['seed'],r['budget']):r for r in rows}
    with (RUN/'per-seed-results.csv').open(encoding='utf-8',newline='') as f:table=list(csv.DictReader(f))
    assert len(table)==252
    for r in table:
        h=int(r['height']);s=int(r['seed']);cell=index[(h,r['arm'],r['task'],s,r['budget'])]
        frozen=index[(h,'readout_only',r['task'],s,r['budget'])]
        assert float(r['probe_gain_vs_frozen'])==cell['metrics']['id_probe']-frozen['metrics']['id_probe']
        assert float(r['causal_direct_drop'])==cell['metrics']['id_direct']-cell['metrics']['revert_both_direct']

def test_capacity_only_label_does_not_mean_absence_of_hidden_learning():
    d=read(RUN/'audit.json');independent=read(RUN/'independent-audit-v1.json')
    assert d['integrity']==independent['integrity']=='PASS'
    assert d['decision']=='CAPACITY_ONLY_SIGNAL' and independent['capacity_signal_pairs']==[]
    g=d['gates']['128']['own']
    assert g['direct_improved_families']==2 and g['hidden_advantage_improved_families']==1
    assert all(v['causal_direct_drop']>.03 for t,v in g['by_task'].items() if t!='order')
    for t in ('strict_binding','xor','distractor_binding'):
        assert d['means']['128']['own'][t]['id_probe']>d['means']['128']['readout_only'][t]['id_probe']
    assert d['fresh_confirmation'] is False and d['p2_tested'] is False and d['production_ready'] is False

def test_xor_extra_advantage_gate_has_reported_frozen_baseline_ceiling():
    d=read(RUN/'audit.json');m=d['means']
    previous=m['48']['own']['xor']['id_probe']-m['48']['readout_only']['xor']['id_probe']
    maximum=1-m['128']['readout_only']['xor']['id_probe']
    assert previous>maximum
    assert maximum-previous<0 and maximum-previous<.02
    # This limitation is disclosed, never converted post hoc into a gate PASS.
    assert d['gates']['128']['own']['by_task']['xor']['hidden_advantage_growth']<.02
