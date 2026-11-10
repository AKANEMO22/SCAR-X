"""Complete standalone human questions without waiting for an LLM tool turn."""


def standalone_clarify(session_id,question,choices,block,emit,timeout):
    payload={'question':question,'choices':choices,'standalone':True}
    answer=block('clarify.request',session_id,payload,timeout=timeout)
    # block() writes its wire request_id into this payload. The renderer needs
    # a matching terminal event because no LLM tool.complete follows an idle
    # controller question. The answer here is user input, not a model result.
    if payload.get('request_id'):
        emit('clarify.resolved',session_id,{'request_id':payload['request_id'],
            'user_response':answer if isinstance(answer,str) else '',
            'question':question,'choices':choices,'standalone':True,
            'source':'standalone learning controller'})
    return answer
