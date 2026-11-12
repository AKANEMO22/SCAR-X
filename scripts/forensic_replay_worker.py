"""Run pinned historical code with output-only redirection and audit taps.

No source file, neuron equation, data seed, update arithmetic or gate is changed.
Extra predictions/identities are saved alongside the original replay output.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time

for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=('r6d-matrix','r6d-controls','r6b'),required=True)
    args=parser.parse_args();source=Path(args.source).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False);(out/'arrays').mkdir()
    audit=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(source/'src'));sys.path.insert(0,str(source/'scripts'))
    import numpy as np
    from scarx.runtime.limits import apply_memory_limit
    from scarx.runtime.resources import ResourceMonitor
    ref=load('forensic_independent_evaluator',audit/'src/scarx/evaluation/forensic/reference_evaluator.py')
    cap=apply_memory_limit(2_000_000_000)
    records=[];heads=[];builds=[];applications=[]
    def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def record_arrays(kind,meta,**arrays):
        identifier=f'{len(records):04d}-{kind}'
        path=out/'arrays'/(identifier+'.npz')
        np.savez_compressed(path,**arrays)
        item={'id':identifier,**meta,'arrays':str(path.relative_to(out)),
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        records.append(item)
        with (out/'observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
        return item
    def input_validation(x,y,a,b,q,pid,flips,qt):
        try:return {'counterfactual':ref.validate_pair_inputs(x,y,a,b,q,pid,flips,expected_tick=qt),'violation':None}
        except ref.AuditViolation as exc:
            relaxed=ref.validate_pair_inputs(x,y,a,b,q,pid,flips,expected_tick=qt,common_noise=False)
            return {'counterfactual':relaxed,'violation':str(exc)}
    started=time.perf_counter()
    with ResourceMonitor(interval=.05) as monitor:
        if args.phase.startswith('r6d'):
            from scarx.evaluation import r6d_audit as A,r5_independent as G
            original_train=A.train_head;original_evaluate=A.evaluate;original_build=A.build_cell
            original_apply=A.apply_features;original_paired=A.paired_metrics;original_predict=A.Head.predict
            stress_inputs={};held_inputs={}
            def build(seed,n,scaler,held=1024,pairs=256):
                d=original_build(seed,n,scaler,held,pairs)
                p=G.independent_paired_set_v1(seed+A.PAIRED_OFFSET,pairs)
                validation=input_validation(d['ptx'],d['pty'],d['pvA'],d['pvB'],p[4],p[5],d['pflips'],10)
                obs=ref.oracle_events(d['tex'])
                if not np.array_equal(obs['labels'],d['y_te']):raise ref.AuditViolation('held oracle mismatch')
                held_inputs.clear();held_inputs[ref.tensor_digest(d['f_te'])]={'oracle':obs,'seed':seed,'digests':d['digests'],
                    'training_label_digest':ref.tensor_digest(d['y_tr'])}
                builds.append({'seed':seed,'n_unique':n,'convention':scaler.convention,'digests':d['digests'],
                    'train_raw_feature_digest':None,'scaled_train_digest':ref.tensor_digest(d['f_tr']),
                    'scaled_held_digest':ref.tensor_digest(d['f_te']),'scaled_pair_digest':ref.tensor_digest(d['f_pt']),
                    'validation':validation,'actual_scaler_digest':scaler.digest()})
                d['_audit_pair_q']=p[4];d['_audit_pair_id']=p[5]
                return d
            def train(f,y,seed,order,epochs=1):
                identity={'seed':seed,'train_features':ref.tensor_digest(f),'training_labels':ref.tensor_digest(y),
                    'order':ref.tensor_digest(np.asarray(order)),'epochs':epochs,'updates':len(order)*epochs}
                initial=A.Head(f.shape[1],seed)
                identity['head_initial']=initial.digest()
                h=original_train(f,y,seed,order,epochs)
                identity['head_final']=h.digest();h._audit_identity=identity
                record_arrays('trained-head',identity,training_labels=y,order=np.asarray(order),weights=h.w,bias=h.b)
                heads.append(identity);monitor.check();return h
            def evaluate(h,d,scaler=None,query_tick=10):
                original=original_evaluate(h,d,scaler,query_tick)
                pp=h.predict(d['f_pt']);pred=h.predict(d['f_te'])
                independent=ref.pair_metrics(pp,d['pty'],d['pvA'],d['pvB'],d['_audit_pair_q'],d['_audit_pair_id'],d['pflips'])
                record_arrays('r6d-evaluate',{'source_metrics':original,'independent_pair_metrics':independent,
                    'seed':d['seed'],'n_unique':d['n_unique'],'head_identity':h._audit_identity,
                    'digests':d['digests'],'direct_independent':float(np.mean(pred==d['y_te']))},
                    pair_prediction=pp,pair_label=d['pty'],vA=d['pvA'],vB=d['pvB'],q=d['_audit_pair_q'],pair_id=d['_audit_pair_id'],flips=d['pflips'],
                    held_prediction=pred,held_label=d['y_te'])
                return original
            pending={}
            def apply(seed,x,scaler,query_tick=10):
                f=original_apply(seed,x,scaler,query_tick)
                obs=ref.oracle_events(x)
                info={'seed':seed,'query_tick':query_tick,'oracle_ticks':np.unique(obs['query_ticks']).tolist(),
                    'input_digest':ref.tensor_digest(x),'scaled_feature_digest':ref.tensor_digest(f),
                    'scaler_digest':scaler.digest(),'head_identity':heads[-1] if heads else None}
                if len(x)==1024:
                    expected=G.independent_query_set_v1(seed+A.HELD_OFFSET,1024)
                    if not np.array_equal(obs['labels'],expected[1]):raise ref.AuditViolation('actual stress held-out oracle changed labels')
                    info['all_actual_held_oracle_labels_equal']=True
                    stress_inputs.clear();stress_inputs[info['scaled_feature_digest']]={'metadata':info,'oracle':obs}
                if len(x)==512:
                    p=G.independent_paired_set_v1(seed+A.PAIRED_OFFSET,256)
                    info.update(input_validation(x,p[1],p[2],p[3],p[4],p[5],p[6],query_tick))
                    pending.clear();pending.update({'x':x.copy(),'f':f,'metadata':info,'pair':p})
                applications.append(info);return f
            def predict(h,f):
                pred=original_predict(h,f)
                if len(f)==1024:
                    held=held_inputs.get(ref.tensor_digest(f))
                    if held:
                        obs=held['oracle']
                        identity=h._audit_identity if hasattr(h,'_audit_identity') else None
                        record_arrays('held-prediction',{'seed':held['seed'],'direct_independent':float(np.mean(pred==obs['labels'])),
                            'head_identity':identity,'training_labels_are_original':bool(identity and identity['training_labels']==held['training_label_digest']),
                            'data_digests':held['digests']},prediction=pred,label=obs['labels'],vA=obs['vA'],vB=obs['vB'],q=obs['q'])
                    context=stress_inputs.pop(ref.tensor_digest(f),None)
                    if context:
                        obs=context['oracle']
                        record_arrays('r6d-stress-held',{'direct_independent':float(np.mean(pred==obs['labels'])),**context['metadata']},
                            prediction=pred,label=obs['labels'],vA=obs['vA'],vB=obs['vB'],q=obs['q'])
                return pred
            def paired(pred,y,a,b,flips):
                original=original_paired(pred,y,a,b,flips)
                if pending and len(pred)==len(pending['x']):
                    p=pending['pair'];ind=ref.pair_metrics(pred,y,a,b,p[4],p[5],flips)
                    record_arrays('r6d-stress',{'independent_pair_metrics':ind,'source_metrics':original,**pending['metadata']},
                        prediction=pred,label=y,vA=a,vB=b,q=p[4],pair_id=p[5],flips=flips,input=pending['x'])
                    pending.clear()
                return original
            A.build_cell=build;A.train_head=train;A.evaluate=evaluate;A.apply_features=apply;A.paired_metrics=paired;A.Head.predict=predict
            filename='r6d_matrix.py' if args.phase=='r6d-matrix' else 'r6d_controls.py'
            runner=load('pinned_historical_runner',source/'scripts'/filename)
            def new_output(*unused,**unused_kw):
                p=out/'original-output';p.mkdir(exist_ok=False);return p
            runner.new_run=new_output
            sys.argv=[str(source/'scripts'/filename),'confirm']
            runner.run()
        else:
            runner=load('pinned_historical_runner',source/'scripts/grok_r6_causal.py')
            duplicates=[]
            try:ref.unique_top_level_definitions((source/'scripts/grok_r6_causal.py').read_text())
            except ref.AuditViolation as exc:duplicates.append(str(exc))
            gen=runner.gen;original_fit=runner.fit_predict;original_pair=runner.pair_stats;original_features=runner.features
            feature_context={}
            def features(seed,x,tick,projection=1234):
                f=original_features(seed,x,tick,projection)
                obs=ref.oracle_events(x)
                feature_context.clear();feature_context.update({'seed':seed,'query_tick':tick,'projection':projection,'input':x.copy(),
                    'oracle':obs,'features_digest':ref.tensor_digest(f)})
                return f
            def fit(cls,f,y,test,seed):
                predictions,accuracy=original_fit(cls,f,y,test,seed)
                if len(test)==512 and feature_context.get('features_digest')==ref.tensor_digest(test):
                    x=feature_context['input'];obs=feature_context['oracle'];pid=np.repeat(np.arange(len(x)//2),2)
                    flips=obs['vA'][::2]!=obs['vB'][::2]
                    # A 512-row IID set is not necessarily a sibling set. Identify
                    # pairs by the oracle query ordering plus input contract.
                    try:
                        validation=input_validation(x,obs['labels'],obs['vA'],obs['vB'],obs['q'],pid,flips,feature_context['query_tick'])
                    except ref.AuditViolation:return predictions,accuracy
                    ind=ref.pair_metrics(predictions,obs['labels'],obs['vA'],obs['vB'],obs['q'],pid,flips)
                    record_arrays('r6b-head',{'seed':seed,'class_module':cls.__module__,'class_name':cls.__name__,
                        'train_features_digest':ref.tensor_digest(f),'train_labels_digest':ref.tensor_digest(y),
                        'test_features_digest':ref.tensor_digest(test),'projection':feature_context['projection'],
                        'validation':validation,'independent_pair_metrics':ind},prediction=predictions,label=obs['labels'],vA=obs['vA'],vB=obs['vB'],q=obs['q'],pair_id=pid,flips=flips)
                monitor.check();return predictions,accuracy
            original_ridge=runner.probe.ridge_predict
            def ridge(f,y,test,alpha=.01):
                pred=original_ridge(f,y,test,alpha)
                x=feature_context['input'];obs=feature_context['oracle'];pid=np.repeat(np.arange(len(x)//2),2);flips=obs['vA'][::2]!=obs['vB'][::2]
                ind=ref.pair_metrics(pred,obs['labels'],obs['vA'],obs['vB'],obs['q'],pid,flips)
                record_arrays('r6b-ridge',{'seed':feature_context['seed'],'train_features_digest':ref.tensor_digest(f),
                    'test_features_digest':ref.tensor_digest(test),'projection':feature_context['projection'],'independent_pair_metrics':ind},
                    prediction=pred,label=obs['labels'],vA=obs['vA'],vB=obs['vB'],q=obs['q'],pair_id=pid,flips=flips)
                return pred
            runner.features=features;runner.fit_predict=fit;runner.probe.ridge_predict=ridge
            runner.OUT=out/'original-output';runner.main()
            write('effective_bindings.json',{'duplicates':duplicates,'screen_seed_line':runner.screen_seed.__code__.co_firstlineno,
                'pair_stats_line':original_pair.__code__.co_firstlineno,'fit_predict_line':original_fit.__code__.co_firstlineno,
                'candidate_module':runner.NoClip.__module__})
        monitor.check()
    write('audit-instrumentation.json',{'run_class':'HISTORICAL_REPRODUCTION','phase':args.phase,
        'source_head':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'output_only_redirect':True,'equations_seeds_updates_unchanged':True,'heads':heads,'builds':builds,'stress_applications':applications,
        'observations':records,'resources':monitor.report(),'cap':cap,'seconds':time.perf_counter()-started})
    print(json.dumps({'phase':args.phase,'output':str(out),'observations':len(records),'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
