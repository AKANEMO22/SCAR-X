"""Read-only canonical-Hermes receipt audit, preserve RC15 followup failure."""
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
QUERY='việt nam có bao nhiêu tỉnh thành 2026'


def turn_summary(turn):
    hits,errors=[],[]
    for row in turn['tool_results']:
        raw=row['result']
        try:value=json.loads(raw) if isinstance(raw,str) else raw
        except ValueError:
            # The actual no-progress guard can return plain text. Preserve it
            # as an error receipt; never count it as successful search evidence.
            errors.append({'non_JSON_tool_receipt':str(raw)[:800]})
            continue
        if not isinstance(value,dict):
            errors.append({'non_object_tool_receipt':str(raw)[:800]})
            continue
        if row.get('is_error') or value.get('success') is not True:
            errors.append(value.get('error','tool-error'))
        else:hits.extend(value.get('data',{}).get('web',[]))
    return {'question':turn['query'],'actual_tool_names':[c['name'] for c in turn['tool_calls']],
        'actual_search_queries':[c.get('args',{}).get('query') for c in turn['tool_calls']],
        'returned_public_results':len(hits),'result_urls':[h['url'] for h in hits],
        'tool_errors':errors,'native_answer':turn.get('completion',{}).get('text'),
        'outcome':turn.get('completion',{}).get('outcome'),'seconds':turn.get('seconds')}


def main():
    previous=ROOT/'local/dated-public-web-acceptance/20261115T121526948447Z'
    current=Path((ROOT/'local/current-dated-public-web-acceptance.txt').read_text(encoding='utf-8').strip())
    assert current!=previous
    old=json.loads((previous/'result.json').read_text(encoding='utf-8'))
    new=json.loads((current/'result.json').read_text(encoding='utf-8'))
    assert old['completed'] and new['completed']
    assert [r['query'] for r in new['turns']]==[QUERY,'là']
    turns=[turn_summary(r) for r in new['turns']]
    old_turns=[turn_summary(r) for r in old['turns']]
    assert old_turns[0]['actual_search_queries']==[QUERY]
    assert len(old_turns[1]['actual_search_queries'])==3 and old_turns[1]['actual_search_queries'][0]!=QUERY
    for turn in turns:
        assert turn['actual_tool_names']==['web_search']
        assert turn['actual_search_queries']==[QUERY]
        assert turn['returned_public_results']>0 and not turn['tool_errors'] and turn['outcome']=='completed'
        assert all(url.startswith(('https://','http://')) for url in turn['result_urls'])
        assert 'stopped retrying' not in (turn['native_answer'] or '').casefold()
    # The reference fact was checked separately from actual model output. Never
    # create a training label from text merely because the model emitted34.
    report={'verdict':'ACTUAL_HERMES_DATED_QUERY_AND_SHORT_FOLLOWUP_ROUTE_VERIFIED',
        'canonical_Hermes':True,'actual_SearXNG':True,'turns':turns,
        'prior_RC15':{'first_question_search_succeeded':True,'short_followup_failed':True,
            'turns':old_turns,'raw_receipt_sha256':hashlib.sha256((previous/'result.json').read_bytes()).hexdigest()},
        'current_receipt_sha256':hashlib.sha256((current/'result.json').read_bytes()).hexdigest(),
        'no_local_file_or_paid_extractor_calls':True,'new_training':False,'SNN_policy_learned_this_route':False,
        'answer_substituted_by_runtime':False,'answer_breakdown_and_source_citation_not_demonstrated':True,
        'general_tool_use_or_neural_quality_PASS':False,
        'reference_fact_checked_separately':{
            'as_of':'2026-11-15','provincial_level_units_total':34,
            'current_breakdown':{'provinces':27,'centrally_governed_cities':7},
            'sources':[
                'https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/4/nq30_nqthanhlaptpdongnai.pdf',
                'https://vqh.hanoi.gov.vn/index.php?language=vi&nv=news&op=tin-lien-ket%2Fxu-huong-phat-trien-do-thi-thuc-hien-mo-hinh-chinh-quyen-dia-phuong-2-cap-3342.html'],
            'older_2025_28_provinces_6_cities_is_not_current_after_2026_04_30':True,
            'not_given_as_expected_answer_to_native_model':True,'not_a_learning_label':True}}
    path=ROOT/'docs/release/evidence/ACTUAL_HERMES_DATED_WEB_20261115.json'
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
