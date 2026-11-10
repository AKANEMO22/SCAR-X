from tui_gateway.controller_questions import standalone_clarify


def test_standalone_question_emits_correlated_terminal_event_after_answer():
    events=[]
    def block(event,sid,payload,timeout):
        assert event=='clarify.request' and sid=='qa' and payload['standalone']
        payload['request_id']='wire-id';return '2'
    answer=standalone_clarify('qa','1+1?',None,block,lambda *a:events.append(a),300)
    assert answer=='2'
    assert events==[('clarify.resolved','qa',{'request_id':'wire-id','user_response':'2','question':'1+1?','choices':None,'standalone':True,'source':'standalone learning controller'})]


def test_cancelled_standalone_question_settles_without_fabricated_answer():
    events=[]
    def block(event,sid,payload,timeout):payload['request_id']='cancelled';return ''
    assert standalone_clarify('qa','unknown',None,block,lambda *a:events.append(a),1)==''
    assert events[0][2]['user_response']=='' and events[0][2]['request_id']=='cancelled'
