"""Encode the agent's complete public128row semantic review BEFORE training."""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.direct_word_data import validate,split_reviewed,TARGETS
from scarx.neural.language import tokenize,render
from scarx.app.observatory import archived_single_prompt_references
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.work_trace import activity


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    generation=json.loads((root/'artifacts/antigravity-direct-word-prompts.json').read_text(encoding='utf-8'));repair=json.loads((root/'artifacts/antigravity-direct-word-repair.json').read_text(encoding='utf-8'))
    raw=(root/generation['data']).read_bytes();fixed=(root/repair['data']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==generation['sha256'] and hashlib.sha256(fixed).hexdigest()==repair['sha256']
    items=validate(json.loads(raw)['items']);replacement=json.loads(fixed)['replacement'];assert replacement['id']=='vi_greet_11' and replacement['language']=='vi' and replacement['prompt']=='Xin chào bạn trợ lý, chúc bạn một ngày làm việc thật hiệu quả và vui vẻ nhé!'
    corrected=[replacement if r['id']=='vi_greet_11' else r for r in items];validate(corrected);rows=split_reviewed(corrected)
    # Eachrow was individually read in this session, not automatically certified
    # by schema or teacher confidence. Store that decision and specific caveats.
    caveats={'vi_greet_09':'Greeting conveyed by pleasureatmeeting, no explicit chào; valid greeting paraphrase.',
        'en_greet_14':'Greeting conveyed by daywish, no explicit hello; legitimate closedgreeting but potentiallycontext-dependent.',
        'vi_identity_07':'Yes/no form asking personalname; canonicalnameanswer is appropriate though not unique.',
        'en_identity_08':'Yes/no namequestion; naming oneself is acceptable, style target not unique.',
        'vi_bye_13':'Goodbye plus healthwish, still unambiguous farewell.',
        'vi_greet_11':'Original mixedlanguage quarantined; independentlyread all-VietnameseAGYrepair accepted.'}
    old=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'));historical={ ' '.join(tokenize(p)) for f in old['families'] for key in ('train','development','acceptance') for p in f[key]}
    historical.update(' '.join(tokenize(p)) for p in archived_single_prompt_references(root))
    for row in rows:row.update(review_decision='accepted_explicit_style_example',review_caveat=caveats.get(row['id']),historical_prompt_overlap=' '.join(tokenize(row['prompt'])) in historical)
    assert len([r for r in rows if r['split']=='train'])==len([r for r in rows if r['split']=='development'])==64
    assert len({' '.join(tokenize(r['prompt'])) for r in rows})==128
    for target in TARGETS.values():assert tokenize(render(tokenize(target)))==tokenize(target) and len(tokenize(target))<=7
    run=new_run(root,'direct-word-semantic-review-v1',{'protocol':'experiments/direct-word-bridge-v1.md','display_title':'Review128AGYprompts ·1languagefix, chưa học',
        'observability_kind':'data_review','model_scope':'All128publicrows read; projectstylelabels not factualtruth. No neuralqualityselection/training.',
        'items':128,'all_rows_read':True,'quarantined_original_rows':1,'reviewed_repairs':1},
        data_description='All publicgeneratedprompts and one languagecorrection; fixed8train8dev perfamily BEFOREtraining.',
        data_files=[generation['data'],repair['data'],'experiments/direct-word-bridge-v1-data-amendment.md','experiments/direct-word-bridge-v1-language-repair.md'])
    candidate=root/'data/candidates'/run.name;candidate.mkdir(parents=True)
    payload={'format':'scarx-direct-word-style-dataset-v1','items':rows,'source_generation':generation['run'],'source_sha256':generation['sha256'],'source_repair':repair['run'],'repair_sha256':repair['sha256'],
        'target_version':'ownedstyle36A-noapostrophe','target_source':'Projectcanonical8style labels, not AGYanswers/facts.','split_seed':36101,'all_rows_read':True,'automatic_learning_allowed':False,
        'semantic_review_is_factual_verification':False,'historical_prompt_overlap_count':sum(r['historical_prompt_overlap'] for r in rows),'caveats':caveats,
        'quarantined_original':next(r for r in items if r['id']=='vi_greet_11')}
    path=candidate/'reviewed.json';write_json(path,payload);receipt={'run':run.relative_to(root).as_posix(),'source_run':generation['run'],'repair_run':repair['run'],
        'data':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decision':'DIRECT_WORD_STYLE_DATA_REVIEWED','items':128,
        'train':64,'development':64,'all_rows_read':True,'historical_prompt_overlap_count':payload['historical_prompt_overlap_count'],'review_caveats':len(caveats),
        'original_quarantined_rows':1,'reviewed_repair_rows':1,'automatic_learning_allowed':False,'teacher_answers_used':False,'trained':False,'validation_pending':False}
    (run/'report.md').write_text('# Publicprompt semanticreview\n\nAll128read beforetraining. One mixedlanguagecandidate quarantined and oneactualAGYrepair read/accepted. Targets ownedstyles, notfacts. Historicaloverlaps/caveats retained.\n\n'+json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    write_json(run/'result.json',receipt);write_json(root/'artifacts/direct-word-data-reviewed.json',receipt)
    activity(run,'Đã review128câu, khóa64train/64dev','Một mixedlanguagegốc được giữ/quarantine, sửaAGY riêng đãđọc.6caveats ghi rõ. Nhãncanonicalstyle riêng, khôngteachertruth hoặc chọn theo modelscore.',evidence={'reviewed_data':receipt['data'],'sha256':receipt['sha256'],'historical_overlap_count':receipt['historical_prompt_overlap_count'],'train_dev':[64,64]})
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
