"""Repair template scheduling before training, reuse raw tool outputs, no rerun."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import copy
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.hybrid.store import encode


def main():
    original=Path((ROOT/'local/current-verified-repair-data.txt').read_text(encoding='utf-8').strip())
    manifest=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
    rows=json.loads((original/'verified-data.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((original/'verified-data.json').read_bytes()).hexdigest()==json.loads((original/'result.json').read_text(encoding='utf-8'))['data_sha256']
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-repair-data-qualified-v2-'+uuid.uuid4().hex[:6]);run.mkdir()
    counts=defaultdict(Counter);times=defaultdict(set);words=defaultdict(set)
    for row in rows:
        i=int(row['id'].rsplit('-',1)[1]);split=row['split'];templates=manifest['templates'][split]
        template_index=(i//4)%len(templates)
        request=templates[template_index].format(time=row['requested_time'])
        row['original_prompt_sha256']=hashlib.sha256(encode(row['messages']).encode()).hexdigest()
        public=json.loads(row['messages'][1]['content']);public['request']=request
        row['messages'][1]['content']=encode(public);row['user_request']=request
        row['template_index']=template_index
        counts[split][str(template_index)+':'+row['error_kind']]+=1
        times[split].add(row['requested_time']);words[split].add(templates[template_index])
        assert row['independent_effect_proof']['goal_verified']
        assert row['before_state_sha256']!=row['after_state_sha256']
        assert json.loads(row['target'])['parameters']['time']==row['requested_time']
        assert row['original_actual_result']['exception'] and row['corrected_actual_result']['exception'] is None
        assert 'session_token' not in encode(row['messages'])
    for split in counts:
        assert len(set(counts[split].values()))==1
        for other in counts:
            if other!=split:assert not(times[split]&times[other]) and not(words[split]&words[other])
    (run/'verified-data.json').write_text(encode(rows),encoding='utf-8')
    report={'verdict':'SOURCE_VERIFIED_REPAIRS_BALANCED_PROMPT_VIEW_NOT_MODEL_QUALITY_PASS',
        'parent_run':original.name,'parent_data_sha256':hashlib.sha256((original/'verified-data.json').read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256((run/'verified-data.json').read_bytes()).hexdigest(),
        'rows':56,'splits':{s:sum(r['split']==s for r in rows) for s in counts},
        'balanced_template_error_cross_counts':{s:dict(c) for s,c in counts.items()},
        'wording_and_requested_time_disjoint':True,'no_new_tool_or_model_calls':True,
        'parent_template_error_confound_preserved':True,'proofs_are_executable_teacher_not_VLM':True,
        'scientific_or_production_PASS':False}
    (run/'result.json').write_text(encode(report),encoding='utf-8')
    (run/'source.py').write_bytes(Path(__file__).read_bytes())
    (ROOT/'local/current-qualified-repair-data.txt').write_text(str(run),encoding='utf-8')
    print(encode({'run':str(run),**report}),flush=True)


if __name__=='__main__':main()
