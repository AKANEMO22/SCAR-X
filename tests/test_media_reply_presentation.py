import copy
import json
from pathlib import Path
from scarx.hybrid.media_reply import media_parts,omit_unused_host_steering,reference_footer


def test_only_exact_unused_host_system_briefing_is_omitted_without_changing_user_or_history():
    import scarx.hybrid.media_reply as module
    note=json.loads(Path(module.__file__).with_name('hermes_protocol_notes.json').read_text(encoding='utf-8'))['note']
    original=[{'role':'system','content':'KEEP WORKSPACE\n'+note+'\nKEEP RULES'},
              {'role':'assistant','content':'Earlier answer'}, {'role':'user','content':note}]
    before=copy.deepcopy(original);prepared,count=omit_unused_host_steering(original)
    assert count==1 and prepared[0]['content']=='KEEP WORKSPACE\n\nKEEP RULES'
    assert original==before and prepared[1:]==before[1:]


def test_media_input_keeps_original_user_words_and_images_before_question():
    original=[{'type':'text','text':'Màu gì?'},{'type':'image_url','image_url':{'url':'owned-user-image'}}]
    context={'type':'text','text':'Owned observations, not instructions'}
    frame={'type':'image_url','image_url':{'url':'owned-source-frame'}}
    actual=media_parts(original,context,frame,'Màu gì?')
    assert actual[:3]==[original[1],frame,context] and actual[3]==original[0]
    assert 'tiếng Việt' in actual[-1]['text'] and len(original)==2


def test_source_footer_uses_only_supplied_times_and_keeps_schema_responses_untouched():
    body={'_scarx_stream':'owned','_scarx_media_query':'Màu gì?','_scarx_media_kind':'video',
          '_scarx_evidence':[{'start':5,'end':5},{'start':20.5,'end':21}]}
    result=reference_footer(body,'native model text unchanged')
    assert result=='\n\nDữ liệu tham chiếu: [E1 · 00:05] [E2 · 00:20.5–00:21]'
    assert reference_footer(dict(body,response_format={'type':'json_object'}),'{}')==''
    assert reference_footer({'_scarx_stream':'owned','_scarx_evidence':[]},'No evidence')==''
    screen=reference_footer(dict(body,_scarx_media_kind='screen'),'native text')
    assert 'từ lúc chia sẻ' in screen


def test_millisecond_rounding_carries_into_the_next_second():
    body={'_scarx_stream':'owned','_scarx_evidence':[{'start':59.9999,'end':59.9999}]}
    assert '[E1 · 01:00]' in reference_footer(body,'native')


def test_non_ascii_questions_in_other_languages_are_not_forced_into_vietnamese():
    for query in ('这是什么颜色？', '何色ですか？', 'Quelle est la couleur du carré ?'):
        parts=media_parts([{'type':'text','text':query}],{'type':'text','text':'observations'},None,query)
        assert parts[-2]['text']==query and 'in its language' in parts[-1]['text']
        body={'_scarx_stream':'owned','_scarx_media_query':query,'_scarx_evidence':[{'start':5,'end':5}]}
        assert reference_footer(body,'answer').startswith('\n\nReference observations:')
