"""Validate generated records; never equate authored rubric with factual proof."""
from pathlib import Path
import json,re,hashlib,sys,unicodedata
from collections import Counter,defaultdict
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.neural.language import tokenize
from scarx.learning.verifiers import parse_integer_expression,integer_answer

def canonical(text):return ' '.join(tokenize(text)).strip(' .!?')
def template(text):
    text=canonical(text)
    text=re.sub(r'\b\d+\b','<n>',text)
    text=re.sub(r'\b(an|bình|chi|dũng)\b','<person>',text)
    return text

def main():
    pointer=json.loads((root/'artifacts/antigravity-dataset-latest.json').read_text(encoding='utf-8'))
    directory=(root/pointer['candidate_directory']).resolve()
    if not directory.is_relative_to((root/'data/candidates').resolve()):raise ValueError('Unexpected dataset path')
    rows=[]
    for file in sorted(directory.glob('batch-*.jsonl')):
        rows.extend(json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip())
    finals=set()
    base=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
    known_train={canonical(p) for f in base['families'] for p in f['train']}
    finals.update(canonical(p) for f in base['families'] for p in f['acceptance'])
    finals.update(canonical(p) for p,_ in json.loads((root/'data/dialogue/generation1-confirmation-next.json').read_text(encoding='utf-8'))['prompts'])
    old=json.loads((root/'data/dialogue/pilot-v2-test.json').read_text(encoding='utf-8'))
    finals.update(canonical(p) for values in old.values() for p in values)
    by_input=defaultdict(list);by_shape=defaultdict(set);by_template=defaultdict(set)
    for row in rows:
        by_input[canonical(row.get('input',''))].append(row)
        by_shape[template(row.get('input',''))].add(row.get('split'))
        by_template[(row.get('family_id'),row.get('template_id'))].add(row.get('split'))
    approved=[];quarantine=[];ids=set();vocabulary=set()
    for original in rows:
        row=dict(original);reasons=[]
        for key in ('id','family_id','template_id','skill','input','target','split','source','confidence'):
            if not isinstance(row.get(key),str) or not row[key].strip():reasons.append('missing_or_invalid_'+key)
        if reasons:quarantine.append({'record':row,'reasons':reasons});continue
        if row['id'] in ids:reasons.append('duplicate_id')
        ids.add(row['id'])
        if row['split'] not in ('train','development'):reasons.append('invalid_split')
        if not 1<=len(row['input'])<=2048 or not 1<=len(tokenize(row['target']))<=12:reasons.append('size_limit')
        if row['source']!='synthetic:antigravity' or row['confidence']!='authored-not-independently-verified':reasons.append('unsupported_provenance_claim')
        key=canonical(row['input']);same=by_input[key]
        if len({canonical(r['target']) for r in same})>1:reasons.append('conflicting_labels_same_input')
        if len({r['split'] for r in same})>1:reasons.append('exact_cross_split_leakage')
        if key in finals:reasons.append('historical_confirmation_overlap')
        if row['split']=='development' and key in known_train:reasons.append('development_seen_in_previous_training')
        if len(by_template[(row['family_id'],row['template_id'])])>1:reasons.append('template_id_cross_split')
        if len(by_shape[template(row['input'])])>1:reasons.append('surface_template_cross_split')
        verification=row.get('verification',{});independent=False
        if not isinstance(verification,dict):reasons.append('invalid_verification')
        elif verification.get('kind')=='integer_arithmetic':
            operands=verification.get('operands');operator=verification.get('operator')
            match=parse_integer_expression(row['input'])
            if not match or not isinstance(operands,list) or len(operands)!=2 or any(type(v) is not int or not 0<=v<=9 for v in operands) or operator not in ('+','-'):
                reasons.append('arithmetic_input_not_independently_parseable')
            else:
                a,b=operands;expected=a+b if operator=='+' else a-b
                if [match[0],match[2]]!=operands or match[1]!=operator or row['target'].strip()!=str(expected) or verification.get('expected_integer')!=expected or expected<0:
                    reasons.append('arithmetic_mismatch')
                else:independent=True
        elif verification.get('kind')=='rubric':
            if not isinstance(verification.get('criterion'),str) or not verification['criterion'].strip():reasons.append('missing_rubric')
        else:reasons.append('unknown_verifier')
        row['validation']={'independently_verified':independent,'automatic_learning_allowed':independent,
            'basis':'operands parsed from input and recomputed without eval' if independent else 'authored rubric; human review required'}
        if reasons:quarantine.append({'record':row,'reasons':sorted(set(reasons))})
        else:
            if any(canonical(r['input'])==key for r in approved):
                quarantine.append({'record':row,'reasons':['duplicate_input_same_split']});continue
            approved.append(row);vocabulary.update(tokenize(row['target']))
    if len(vocabulary)>200:raise ValueError('Accepted targetvocabulary exceeds pilot200 limit')
    for split,name in (('train','train.jsonl'),('development','development.jsonl')):
        (directory/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in approved if r['split']==split),encoding='utf-8')
    # Interleave experiences. Family/template IDs are evaluator metadata, never
    # supplied as hidden task cues to the neural input encoder.
    train=[r for r in approved if r['split']=='train'];order=np.random.default_rng(2093).permutation(len(train))
    (directory/'experience-stream.jsonl').write_text(''.join(json.dumps(train[int(i)],ensure_ascii=False)+'\n' for i in order),encoding='utf-8')
    (directory/'quarantine.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in quarantine),encoding='utf-8')
    counts=Counter(reason for r in quarantine for reason in r['reasons'])
    result={'raw_examples':len(rows),'accepted_examples':len(approved),'train':sum(r['split']=='train' for r in approved),
        'development':sum(r['split']=='development' for r in approved),'quarantined':len(quarantine),'quarantine_reasons':dict(counts),
        'independently_verified_arithmetic':sum(r['validation']['independently_verified'] for r in approved),
        'automatic_learning_train_examples':sum(r['split']=='train' and r['validation']['automatic_learning_allowed'] for r in approved),
        'target_vocabulary':len(vocabulary),'trained':False,'stream_shuffle_seed':2093,'candidate_directory':str(directory.relative_to(root)),
        'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.jsonl')}}
    (directory/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/'artifacts/antigravity-dataset-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
