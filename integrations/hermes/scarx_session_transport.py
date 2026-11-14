"""Physical Hermes transcript identity for SCAR-X local media bindings.

Generic routing affinity can name a host conversation scope. Video attachments
are authorized against the transcript IDs exposed to the desktop, so preserve
the actual main agent session on this single registered local route.
"""

from urllib.parse import urlsplit
import re
import unicodedata


def social_greeting_only(messages):
    if not isinstance(messages,list) or not all(isinstance(m,dict) for m in messages):return False
    latest=next((i for i in range(len(messages)-1,-1,-1) if messages[i].get('role')=='user'),None)
    if latest is None or latest!=len(messages)-1:return False
    content=messages[latest].get('content')
    if not isinstance(content,str) or len(content)>160:return False
    plain=''.join(c for c in unicodedata.normalize('NFKD',content.casefold()) if not unicodedata.combining(c)).replace('đ','d')
    return bool(re.fullmatch(r'\s*(?:xin\s+chao(?:\s+ban)?|chao(?:\s+ban)?|hi|hello|hey)[!.,\s]*(?:(?:tra loi ngan(?: gon)?(?: bang tieng viet)?(?: nhe)?|reply briefly(?: in english)?)[!.,\s]*)?',plain))


def omit_unused_steering_example(messages):
    """A social turn has no tool result at which Hermes can deliver a steer.

    Remove only the host's exact generated briefing from system-role messages;
    never search/replace control-frame lookalikes in user text, history or tools.
    Real tool tasks keep the original steering protocol unchanged.
    """
    from agent.prompt_builder import STEER_CHANNEL_NOTE
    changed=False;result=[]
    for message in messages:
        content=message.get('content')
        if message.get('role')=='system' and isinstance(content,str) and content.count(STEER_CHANNEL_NOTE)==1:
            message=dict(message,content=content.replace(STEER_CHANNEL_NOTE,''));changed=True
        result.append(message)
    return result if changed else messages


def bind_transcript_session(agent, kwargs):
    url=urlsplit(str(getattr(agent,'base_url','') or ''))
    session=getattr(agent,'session_id',None)
    try:port=url.port
    except ValueError:return kwargs
    if (getattr(agent,'model',None)!='scarx-hybrid' or url.scheme!='http'
            or url.hostname not in ('127.0.0.1','localhost') or port!=8775
            or url.path.rstrip('/')!='/v1' or not isinstance(session,str)
            or not 1<=len(session)<=160 or session=='local'
            or any(ord(c)<32 or ord(c)>126 for c in session)):
        return kwargs
    headers=dict(kwargs.get('extra_headers') or {})
    # Case-insensitive replacement avoids duplicate conflicting HTTP headers.
    headers={k:v for k,v in headers.items() if k.lower() not in ('x-scarx-session','x-scarx-submission')}
    headers['X-SCARX-Session']=session
    # The canonical executor supplies the actual admitted request, including
    # queued/retried inputs. No extraction from messages or global session state.
    from gateway.session_submission import current_submission_id
    submission=current_submission_id()
    if isinstance(submission,str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,256}',submission):
        headers={k:v for k,v in headers.items() if k.lower()!='x-scarx-submission'}
        headers['X-SCARX-Submission']=submission
    prepared=dict(kwargs,extra_headers=headers)
    if kwargs.get('tool_choice') in (None,'auto') and social_greeting_only(kwargs.get('messages')):
        # No generated answer or prompt replacement: narrow host tool exposure
        # for a purely social turn, restoring normal tools on the next task.
        prepared.pop('tools',None);prepared.pop('tool_choice',None)
        prepared['messages']=omit_unused_steering_example(kwargs['messages'])
    return prepared
