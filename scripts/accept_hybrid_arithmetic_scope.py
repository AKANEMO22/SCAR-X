"""Actual isolated controller path, declared review fixture, zero model calls.

Never modifies the registered live store. This is label-authorization acceptance,
not an answer-quality benchmark or proof of neural learning.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.config import HybridConfig
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.learner import LocalALIF
from scarx.hybrid.store import Store
from scarx.hybrid.arithmetic import VERIFIER_ID


class Monitor:
    def sample(self):
        rss=psutil.Process().memory_info().rss
        return {'rss_bytes':rss,'conservative_ram_bytes':rss}
    check=sample


class DeclaredReflectionFixture:
    def __init__(self):self.calls=0
    def generate(self,*args,**kwargs):
        self.calls+=1
        return {'text':json.dumps({'question':'','reason':'','public_topic_id':None})}


def main():
    run=ROOT/'local/arithmetic-review-fixture'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True)
    store=Store(run/'private-data');backend=DeclaredReflectionFixture()
    engine=HybridEngine(store,backend,Monitor(),HybridConfig(execution_mode='gpu'))
    engine.model,_=LocalALIF.loads(engine.model.dumps(),device='gpu',gpu_device=0)
    initial=engine.model.dumps();rows=[]
    repro=json.loads((ROOT/'docs/release/evidence/RC12_ARITHMETIC_TASK_REPRO_20261115.json').read_text(encoding='utf-8'))
    cases=[(r['question'],r['answer']) for r in repro['rows']]+[('3-1','2'),('03-01','2')]
    def reject_learning(*args,**kwargs):
        raise AssertionError('Unverifiable task entered the learning path')
    engine._guarded_learn=reject_learning
    try:
        for i,(question,answer) in enumerate(cases):
            engine.record_agent_answer('owned-scope-fixture-'+str(i),question,answer,
                source='isolated-controller-fixture-not-native-VLM-output')
            engine.state['cognition']['last_activity']=time.time()-45
            engine.state['last_presence']=time.time()-45;engine.state['last_review']=0
            result=engine.review_once()
            assert result['status']=='reviewed_unverified' and result['weights_updated'] is False
            assert engine.model.dumps()==initial and engine.model.updates==0
            rows.append({'question':question,'fixture_answer':answer,'review_status':result['status'],'weights_updated':False})
        checkpoint=engine.save(pinned=True)
        restored,state=LocalALIF.loads(store.snapshot(checkpoint),device='gpu',gpu_device=0)
        assert restored.dumps()==initial and state['review']['completed']==len(cases)
        report={'verdict':'ACTUAL_CONTROLLER_REVIEW_REJECTS_INCIDENTAL_AND_AMBIGUOUS_ARITHMETIC_LABELS',
            'verifier_id':VERIFIER_ID,'backend':'declared reflection JSON fixture, not native VLM generation',
            'model_device':engine.model.device,'cases':rows,'actual_native_VLM_calls':0,
            'SNN_forward_or_learning_calls':0,'isolated_neural_arrays_RNG_replay_unchanged':True,
            'cold_restore_exact_model_and_review_records':True,'live_user_state_modified':False,
            'source_sha256':{},'scientific_or_production_PASS':False}
        import hashlib
        for path in (Path(__file__),ROOT/'src/scarx/hybrid/arithmetic.py',ROOT/'src/scarx/hybrid/review.py',ROOT/'src/scarx/hybrid/engine.py'):
            report['source_sha256'][path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        (run/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'private_run':str(run),**report},ensure_ascii=False),flush=True)
    finally:store.close()


if __name__=='__main__':main()
