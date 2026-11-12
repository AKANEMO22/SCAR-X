"""Data/codec audit only. Does not instantiate or train a neural model."""
from pathlib import Path
import os,sys,json,hashlib,io,zipfile,types
from collections import Counter,defaultdict
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.neural.language import tokenize,render
from scarx.neural.unified import UnifiedCore
from scarx.evaluation.runs import new_run,write_json
from public_research_mirror import publish


def compatibility(rows,vocabulary,encoding):
    existing=set(vocabulary);required=set();surface=[];too_long=[];oov=[];conflicts=[];vectors={}
    proxy=types.SimpleNamespace(context_size=128,encoding=encoding)
    for row in rows:
        answer=row['canonical_answer'];tokens=tokenize(answer);required.update(tokens)
        normalized=lambda text:' '.join(text.casefold().split())
        reconstructed=render(tokens)
        if normalized(reconstructed)!=normalized(answer):surface.append({'id':row['id'],'canonical':answer,'rendered':reconstructed,'tokens':tokens})
        if len(tokens)>24:too_long.append(row['id'])
        if not set(tokens)<=existing:oov.append(row['id'])
        # Public G1 context function on a data-only proxy: no core or weights.
        vector=UnifiedCore.context(proxy,row['prompt']);fingerprint=hashlib.sha256(vector.tobytes()).hexdigest()
        target=' '.join(tokens)
        if fingerprint in vectors and vectors[fingerprint][1]!=target:conflicts.append({'ids':[vectors[fingerprint][0],row['id']]})
        else:vectors[fingerprint]=(row['id'],target)
    return {'rows':len(rows),'existing_public_vocabulary':len(existing),'required_answer_tokens':len(required),'union_vocabulary_size':len(existing|required),'within_supported_256_cap':len(existing|required)<=256,'out_of_existing_checkpoint_vocabulary':len(oov),'oov_row_ids':oov,'over_24_target_tokens':too_long,'surface_roundtrip_mismatches':surface,'conflicting_exact_context_collisions':conflicts,'neural_instances_created':0,'neural_inference_performed':False,'note':'Token-sequence capacity, rawcheckpoint OOV, and exact rendered text are distinct. Spaced decimals can be token-equivalent but fail exact surface output. No codec changed.'}


def main():
    info=json.loads((root/'artifacts/basic-knowledge-verified-v1.json').read_text(encoding='utf-8'));pool=root/info['data'];assert hashlib.sha256(pool.read_bytes()).hexdigest()==info['sha256'];rows=json.loads(pool.read_text(encoding='utf-8'))['items']
    release=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'));raw=(root/release['checkpoint']).read_bytes();assert hashlib.sha256(raw).hexdigest()==release['sha256']
    with zipfile.ZipFile(io.BytesIO(raw)) as z:meta=json.loads(z.read('metadata.json'))
    run=new_run(root,'basic-knowledge-compatibility-v1',{'protocol':'experiments/basic-knowledge-factory-v1.md','display_title':'Verified data · vocabulary/EOS/decimal/context audit · no training','observability_kind':'data_validation','model_scope':'Existing publiccodec on data-only proxy, no model instance or live checkpoint. No surface normalization fixes or neural qualityscores.','stage':2},data_description='ActualFlashHigh verifiedcandidates and publicrelease vocabulary/encoding only.',data_files=[info['data']])
    audit=compatibility(rows,meta['vocabulary'],meta.get('encoding',{'kind':'legacy'}));counts=dict(Counter(r['split'] for r in rows));result={'run':run.relative_to(root).as_posix(),'decision':'DATASET_NOT_LEARNABLE_UNDER_CURRENT_CODEC' if not audit['within_supported_256_cap'] or audit['out_of_existing_checkpoint_vocabulary'] or audit['surface_roundtrip_mismatches'] or audit['conflicting_exact_context_collisions'] else 'DATASET_CODEC_COMPATIBLE_NOT_TRAINING_AUTHORIZATION','source_pool':info['data'],'source_sha256':info['sha256'],'verified_rows':len(rows),'facts':len({r['fact_id'] for r in rows}),'split_counts':counts,'audit':audit,'separate_blockers':{'raw_checkpoint_lacks_tokens':bool(audit['out_of_existing_checkpoint_vocabulary']),'exact_surface_roundtrip_fails':bool(audit['surface_roundtrip_mismatches']),'unsupported_capacity':not audit['within_supported_256_cap']},'automatic_learning_allowed':False,'main_training':False,'pilot_run':False,'quality_score_computed':False,'decoder_or_encoder_changed':False}
    write_json(run/'result.json',result);write_json(root/'artifacts/basic-knowledge-compatibility-v1.json',result)
    (run/'report.md').write_text('# Basic knowledge data/codec compatibility\n\nNo neuralqualityscores. Within256cap does not mean rawcheckpoint can emit newtokens. Exactsurface and token-equivalent numeric outputs differ. No codecmigration/fix implemented.\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');publish(run)
    print(json.dumps({'decision':result['decision'],'verified_rows':len(rows),'split_counts':counts,'union_vocabulary':audit['union_vocabulary_size'],'raw_checkpoint_oov_rows':audit['out_of_existing_checkpoint_vocabulary'],'surface_mismatches':len(audit['surface_roundtrip_mismatches']),'context_conflicts':len(audit['conflicting_exact_context_collisions'])}))


if __name__=='__main__':main()
