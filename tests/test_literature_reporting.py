"""Regression for budget leakage in derived CSV, using immutable actual results."""
import csv,json
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]/'runs/20261112T080533Z-literature-reference-dev-v1-839e07'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def test_exported_probe_gain_uses_same_budget_readout():
    rows=[read(p) for p in (RUN/'jobs').glob('*/result.json')]
    index={(r['method'],r['task'],r['seed'],r['budget'],r['shuffled']):r for r in rows}
    with (RUN/'per-seed-results-v2.csv').open(encoding='utf-8',newline='') as stream:table=list(csv.DictReader(stream))
    assert len(table)==132
    for r in table:
        key=(r['method'],r['task'],int(r['seed']),r['budget'],r['shuffled']=='True')
        cell=index[key];baseline=index[('readout_only',r['task'],int(r['seed']),r['budget'],False)]
        assert float(r['probe_gain'])==cell['metrics']['id_probe']-baseline['metrics']['id_probe']
    # Frozen hidden features yield identical refit-probe baselines in BOTH
    # budgets: the old lookup was imprecise but changed no numerical gain.
    controls=[r for r in rows if r['method']=='readout_only' and r['budget']=='equal_cpu']
    assert all(r['metrics']['id_probe']==index[('readout_only',r['task'],r['seed'],'equal_updates',False)]['metrics']['id_probe'] for r in controls)
    assert any(r['metrics']['id_direct']!=index[('readout_only',r['task'],r['seed'],'equal_updates',False)]['metrics']['id_direct'] for r in controls)

def test_csv_reporting_fix_did_not_change_quality_metrics_or_decision():
    a=read(RUN/'audit-v1.json');b=read(RUN/'audit-v2.json')
    assert {k:v for k,v in a.items() if k!='audit_wall_seconds'}=={k:v for k,v in b.items() if k!='audit_wall_seconds'}
    assert b['integrity']=='PASS' and b['decision']=='REFERENCE_MIXED'
    assert b['fresh_confirmation'] is False and b['new_training_updates']==0
