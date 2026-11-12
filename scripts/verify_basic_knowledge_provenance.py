"""Recheck frozen AGY output/source lineage without calling any generator.

Private CLI streams/invocations are read locally only. Public output contains
hashes, model/status attestations and counts, never raw stream or auth content.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
from collections import Counter,defaultdict
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scarx.evaluation.knowledge_factory import (CORE,UNIFIED,SERVER,APP,extract_facts,verify_candidate,
    trusted_split,REQUIRED_FIELDS,inference_input,QUERY_VERIFIER_REVISION)
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor
from public_research_mirror import publish


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(raw):return hashlib.sha256(raw).hexdigest()
def candidate_hash(row):return sha(json.dumps(row,sort_keys=True,ensure_ascii=False).encode())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--public-only',action='store_true');args=parser.parse_args()
    parent=read(ROOT/'artifacts/basic-knowledge-factory-v1.json')
    repaired=read(ROOT/'artifacts/basic-knowledge-verified-v1.json')
    ledger_path=ROOT/repaired['source_ledger'];pool_path=ROOT/repaired['data']
    if sha(ledger_path.read_bytes())!=repaired['source_ledger_sha256'] or sha(pool_path.read_bytes())!=repaired['sha256']:
        raise ValueError('Frozen ledger/pool changed')
    ledger=read(ledger_path);facts=ledger['facts'];pool=read(pool_path)['items'];lookup={r['id']:r for r in facts}
    batches=parent['batches']+[repaired['repair_batch']]
    if len(batches)!=5 or len(pool)!=320:raise ValueError('Registered five calls /320records expected')
    cap=apply_memory_limit();issues=[];source_checks=[];batch_checks=[];raw_index=defaultdict(list)
    contract=(ROOT/'data/contracts/ANTIGRAVITY_BASIC_KNOWLEDGE_V1.md').read_text(encoding='utf-8')
    rejected=read(ROOT/parent['quarantine']);missing={(row['fact_id'],row['variant']) for row in rejected['missing']}
    run=new_run(ROOT,'basic-knowledge-public-provenance-review-v1' if args.public_only else 'basic-knowledge-provenance-review-v1',{'display_title':'AGY frozen dataset · origin/source revalidation · no calls/train',
        'observability_kind':'data_validation','verifier_revision':QUERY_VERIFIER_REVISION,'model_scope':'Existing CLI evidence/pool only. No new AGY/Qwen/student inference or final quality scoring.'},
        data_description='Existing public AST facts, frozen actual AGY candidates; private CLI evidence hashed locally only.',
        data_files=[repaired['data'],repaired['source_ledger'],'experiments/basic-knowledge-factory-v1.md'])
    with ResourceMonitor(interval=.02) as monitor:
        sources={}
        for name in (CORE,UNIFIED,SERVER,APP):
            path=ledger_path.parent/'sources'/name;actual=path.read_bytes()
            expected=subprocess.check_output(['git','show',lookup['config_hidden']['source_commit']+':'+name],cwd=ROOT)
            ok=actual==expected;source_checks.append({'path':name,'sha256':sha(actual),'git_blob_content_exact':ok})
            if not ok:issues.append({'reason':'PINNED_SOURCE_BYTES','path':name})
            sources[name]=actual.decode('utf-8')
        extracted=extract_facts(sources)
        if extracted!=facts:issues.append({'reason':'FACT_LEDGER_NOT_REPRODUCED_FROM_SOURCE'})
        for receipt in batches:
            index=receipt['batch'];candidate_run=parent['run'] if index<=4 else repaired['run']
            path=ROOT/candidate_run/f'candidates-batch-{index}.json';values=read(path)['items']
            observed_hash=candidate_hash(values)
            if observed_hash!=receipt['candidate_sha256']:issues.append({'reason':'BATCH_CANDIDATE_HASH','batch':index})
            for row in values:raw_index[candidate_hash(row)].append(index)
            selected=facts[(index-1)*10:index*10] if index<=4 else [f for f in facts if f['id'] in sorted({key[0] for key in missing})[:10]]
            publicfacts=[{k:f[k] for k in ('id','scope','field','value','value_type','maximum','known_domain')} for f in selected]
            expected_prompt=contract+'\nPUBLIC VERIFIED FACTS:\n'+json.dumps(publicfacts,ensure_ascii=False)
            if index==5:
                note={'missing_slots':[{'fact_id':f,'variant':v} for f,v in sorted(missing)],
                    'important':'For API questions avoid Config, UnifiedCore AND the substring configuration; ask only about API subject. Avoid any spelled-out numeric filler such as one/một or negation. Existing verifier remains unchanged. Return all8variants per suppliedfact; only original missing slots can replace rejected candidates.'}
                expected_prompt+='\nSOURCE-VALIDATION REPAIR NOTES (no modelquality scores):\n'+json.dumps(note,ensure_ascii=False)
            public_payload_ok=sha(expected_prompt.encode())==receipt['prompt_sha256']
            if not public_payload_ok:issues.append({'reason':'PROMPT_NOT_REPRODUCED_FROM_PUBLIC_CONTRACT_FACTS','batch':index})
            model_ok=receipt['model_requested']==receipt['model_reported']=='gemini-3.8-flash-high' and receipt['effort_requested']=='high'
            if not model_ok or receipt['error']:issues.append({'reason':'MODEL_OR_BATCH_STATUS','batch':index})
            if args.public_only:
                batch_checks.append({'batch':index,'candidate_records':len(values),'candidate_sha256':observed_hash,
                    'model_metadata_matches_recorded_receipt':model_ok,'origin_attestation_pass':None,
                    'submitted_prompt_hash_reproduced_from_public_sources':public_payload_ok,
                    'origin_evidence_scope':'Public receipt and candidate hash only; original private CLI stream not independently reattested in this mode',
                    'raw_private_content_published':False})
                monitor.check();continue
            private=(ROOT/receipt['invocation_private']).resolve()
            if not private.is_relative_to(ROOT/'local/basic-knowledge-factory'):raise ValueError('Private evidence boundary')
            invocation=read(private);argv=invocation['argv'];stream=private.parent/'events.jsonl';raw=stream.read_bytes()
            if len(raw)>12_000_000:raise ValueError('Historical stream bound')
            stream_ok=sha(raw)==receipt['stream_sha256'];prompt_ok=sha(argv[argv.index('--print')+1].encode())==receipt['prompt_sha256']
            inits=[];results=[];tools=[]
            for line in raw.decode('utf-8-sig').splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('event')=='init':inits.append(event['init'].get('model'))
                if event.get('event')=='result':results.append(event['result'])
                step=event.get('step_update',{})
                if step.get('step_type')=='tool':tools.append(step.get('tool_name'))
            result=results[-1] if results else {};structured=result.get('structured_output')
            if structured is None and 'response' in result:structured=json.loads(result['response'])
            origin_ok=(stream_ok and prompt_ok and public_payload_ok and bool(inits) and inits[-1]=='gemini-3.8-flash-high'
                and result.get('status')=='SUCCESS' and structured is not None and structured.get('items')==values
                and all(name=='finish' for name in tools))
            if not origin_ok:issues.append({'reason':'CLI_ORIGIN_ATTESTATION','batch':index})
            batch_checks.append({'batch':index,'candidate_records':len(values),'candidate_sha256':observed_hash,
                'private_stream_sha256_matches':stream_ok,'private_prompt_sha256_matches':prompt_ok,
                'submitted_prompt_hash_reproduced_from_public_sources':public_payload_ok,
                'original_stream_reported_model':inits[-1] if inits else None,'original_stream_status':result.get('status'),
                'structured_items_match_public_candidates':structured is not None and structured.get('items')==values,
                'logged_tool_names':sorted(set(tools)),'origin_attestation_pass':origin_ok,'raw_private_content_published':False})
            monitor.check()
        raw_origin_missing=[]
        for row in pool:
            variant=int(row['id'].rsplit('_p',1)[1]);fact=lookup[row['fact_id']]
            raw={k:row[k] for k in ('fact_id','language','prompt','canonical_answer')};raw['variant']=variant
            if candidate_hash(raw)!=row['raw_candidate_sha256'] or row['raw_candidate_sha256'] not in raw_index:
                raw_origin_missing.append(row['id'])
            reason=verify_candidate(raw,fact)
            if reason:issues.append({'reason':reason,'id':row['id']})
            if not set(REQUIRED_FIELDS)<=row.keys() or row['source_hash']!=fact['source_hash'] or row['source_id']!=fact['source_id']:
                issues.append({'reason':'SOURCE_METADATA','id':row['id']})
            if row['split']!=trusted_split(fact,variant) or row['automatic_learning_allowed'] is not False:
                issues.append({'reason':'SPLIT_OR_LEARNING_PERMISSION','id':row['id']})
            if set(inference_input(row))!={'prompt'}:issues.append({'reason':'LABEL_OR_SOURCE_IN_INFERENCE_INPUT','id':row['id']})
        if raw_origin_missing:issues.append({'reason':'ACCEPTED_ROW_NOT_IN_ACTUAL_CANDIDATES','ids':raw_origin_missing})
        # Derive all memberships only from accepted rows; no final inference.
        split_checks={}
        for split in sorted({row['split'] for row in pool}):
            actual=[json.loads(line) for line in (pool_path.parent/(split+'.jsonl')).read_text(encoding='utf-8').splitlines() if line]
            expected=[row for row in pool if row['split']==split]
            split_checks[split]=actual==expected
            if actual!=expected:issues.append({'reason':'SPLIT_FILE_POOL_MISMATCH','split':split})
        monitor.check()
    compatibility=read(ROOT/'artifacts/basic-knowledge-compatibility-v1.json')
    decision='PUBLIC_SOURCE_LINEAGE_VERIFIED_ORIGIN_NOT_REATTESTED_CODEC_BLOCKED' if args.public_only else 'SOURCE_AND_ORIGIN_VERIFIED_CODEC_BLOCKED'
    result={'decision':decision if not issues else 'PROVENANCE_REVIEW_FAILED','public_only_mode':args.public_only,
        'source_pool':repaired['data'],'source_pool_sha256':repaired['sha256'],'records':len(pool),'known_facts':32,'outside_domain_facts':8,
        'actual_historical_generation_calls':5,'new_generation_calls':0,'historical_cli_version':parent['cli_version'],
        'historical_executable_sha256':parent['executable_sha256'],'historical_model':'gemini-3.8-flash-high','historical_effort':'high',
        'source_checks':source_checks,'facts_reproduced_from_ast':extracted==facts,'batch_checks':batch_checks,
        'accepted_raw_candidate_origin_matches':len(pool)-len(raw_origin_missing),'split_file_matches':split_checks,
        'split_counts':dict(Counter(row['split'] for row in pool)),'issues':issues,'resources':monitor.report(),'cap':cap,
        'private_streams_or_invocations_published':False,'historical_cost_usd_reported':None,
        'compatibility_decision_preserved':compatibility['decision'],'automatic_learning_allowed':False,'student_training':False,
        'final_quality_inference_performed':False,'production_ready':False}
    receipt_name='basic-knowledge-public-provenance-review-v1.json' if args.public_only else 'basic-knowledge-provenance-review-v1.json'
    write_json(run/'result.json',result);write_json(ROOT/'artifacts'/receipt_name,{'run':run.relative_to(ROOT).as_posix(),**result})
    (run/'report.md').write_text('# Frozen basic knowledge origin/source review\n\nNo new generation or learning. Private evidence was checked locally; public outputs contain hashes/counts only.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    publish(run)
    print(json.dumps({'run':run.name,'decision':result['decision'],'records':len(pool),'issues':issues,'private_content_published':False}))
    return int(bool(issues))


if __name__=='__main__':raise SystemExit(main())
