"""Independent receipt/trace rescore, no image inference or training."""
import hashlib,json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    current=Path((ROOT/'local/current-Hermes-image-question-acceptance.txt').read_text(encoding='utf-8').strip())
    report=json.loads((current/'result.json').read_text(encoding='utf-8'))
    failed=ROOT/'local/image-question-acceptance/20261115T124143239268Z'
    old=json.loads((failed/'result.json').read_text(encoding='utf-8'))
    assert report['completed'] and old['completed']
    assert report['image_sha256']==old['image_sha256']
    expected_query='câu này làm  ntn giải thích chi tiết'
    assert report['turns'][0]['query']==old['turns'][0]['query']==expected_query
    assert all(t['completion']['outcome']=='completed' and not t['tool_calls'] for t in report['turns'])
    answer=report['turns'][0]['completion']['text']
    previous_answer=old['turns'][0]['completion']['text']
    assert previous_answer.startswith('Đáp án: **A.')
    assert answer.startswith('Đáp án: **C. output$distPlot <- renderPlot(')
    assert 'Shiny for R' in answer and 'https://shiny.posit.co/' in answer
    assert 'đặt vùng hiển thị đồ thị trong UI' in answer
    assert 'không bắt buộc phải đợi người dùng tương tác' in answer
    greeting=report['turns'][1]['completion']['text']
    assert 'chào' in greeting.casefold() and 'Đáp án:' not in greeting and 'renderPlot' not in greeting
    database=ROOT/'local/hybrid-v1/hybrid.sqlite'
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as c:
        events=[{'id':r[0],'kind':r[1],'payload':json.loads(r[2])} for r in c.execute(
            "SELECT id,kind,payload FROM events WHERE id>? AND kind IN ('image_question_reading','image_question_grounded_answer') ORDER BY id",
            (report['trace_start_cursor'],))]
    readings=[e for e in events if e['kind']=='image_question_reading' and e['payload'].get('grounded_answer',{}).get('content')==answer]
    grounded=[e for e in events if e['kind']=='image_question_grounded_answer' and e['payload'].get('answer')==answer]
    assert len(readings)==len(grounded)==1
    reading=readings[0]['payload'];decision=grounded[0]['payload']
    assert reading['grounded_answer']['choice_id']==decision['choice_id']=='C'
    assert reading['grounded_answer']['option_chosen_by']=='frozen GPU VLM'
    assert reading['grounded_answer']['native_freeform_explanation_replaced'] is True
    assert 'runtime rendering' in decision['producer']
    assert not reading['new_training'] and reading['network_requests']==0
    assert all(token in reading['visible_text'] for token in ('plotOutput','renderPlot','output$distPlot','D.'))
    expected_usage={key:sum((u or {}).get(key,0) for u in (reading.get('transcription_usage'),reading['grounded_answer'].get('selection_usage')))
                    for key in ('prompt_tokens','completion_tokens','total_tokens')}
    assert decision['usage']==expected_usage
    assert expected_usage['completion_tokens']==report['turns'][0]['completion']['usage']['output']
    output={
        'verdict':'EXACT_HERMES_IMAGE_CASE_AND_SOURCE_GROUNDED_RESPONSE_VERIFIED_NOT_GENERAL_VISION_PASS',
        'image_sha256':report['image_sha256'],'private_image_uploaded_to_GitHub':False,
        'reported_question':expected_query,'actual_answer':answer,
        'actual_followup_greeting':greeting,'current_raw_receipt_sha256':hashlib.sha256((current/'result.json').read_bytes()).hexdigest(),
        'prior_RC17_failed_answer':previous_answer,'prior_raw_receipt_sha256':hashlib.sha256((failed/'result.json').read_bytes()).hexdigest(),
        'prior_failure_preserved':True,'local_GPU_transcription':reading['visible_text'],
        'native_selected_option':'C','model_calls_for_image_task':2,
        'explanation_producer':decision['producer'],'native_freeform_explanation_replaced':True,
        'reference_ids':reading['reference_ids'],'reference_sources':reading['reference_sources'],
        'usage_counts_native_transcription_and_selection_not_rendered_words':expected_usage,
        'actual_Hermes_tools_called':0,'network_requests_for_attachment':0,'new_training':False,
        'SNN_learned_Shiny_or_general_vision_improvement':False,
        'reference_card_scope':'One manually sourced R Shiny API card; not a general documentation retrieval service',
        'actual_scope':'One reported image plus one greeting regression; limited product acceptance, not heldout benchmark',
        'script_admission_failure_before_this_run':{'reason':'Hermes refused retained-input path; upload must use profile staging cache',
            'guard_preserved':True,'fixed_by':'copy exact same pixels to fresh owned staging file; no bypass'},
        'native_probe_failures_preserved':True,
    }
    destination=ROOT/'docs/release/evidence/ACTUAL_HERMES_IMAGE_QUESTION_20261115.json'
    with destination.open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps(output,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
