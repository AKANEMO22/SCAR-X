"""Media request presentation and source references, never neural learning."""

import hashlib
import json
from pathlib import Path
import re


def _vietnamese_query(query):
    # A narrow presentation hint, not a general language classifier. Other
    # scripts and languages retain the instruction to use the user's language.
    return bool(re.search(r'[ăâđêôơưĂÂĐÊÔƠƯ\u1ea0-\u1ef9]|\b(?:màu|gì|tôi|bạn|nào|hãy)\b', query, re.IGNORECASE))


def omit_unused_host_steering(messages):
    data=json.loads(Path(__file__).with_name('hermes_protocol_notes.json').read_text(encoding='utf-8'))
    note=data['note']
    if hashlib.sha256(note.encode()).hexdigest()!=data['sha256']:raise ValueError('Pinned host note integrity mismatch')
    result=[];omitted=0
    for message in messages:
        content=message.get('content')
        if message.get('role')=='system' and isinstance(content,str) and content.count(note)==1:
            message=dict(message,content=content.replace(note,''));omitted+=1
        result.append(message)
    return result,omitted


def media_parts(original,observations,frame,query):
    images=[part for part in original if part.get('type')=='image_url']
    texts=[part for part in original if part.get('type')!='image_url']
    language=('Trả lời bằng tiếng Việt, ngắn gọn; dùng bằng chứng đã cho và không chép lại định dạng nội bộ.'
              if _vietnamese_query(query) else 'Answer the user question briefly in its language using the supplied evidence. Do not repeat internal control markers.')
    return images+([frame] if frame else [])+[observations]+texts+[{'type':'text','text':language}]


def reference_footer(body,answer):
    if not body.get('_scarx_stream') or body.get('response_format') is not None or not answer.strip():return ''
    evidence=body.get('_scarx_evidence') or []
    if not evidence:return ''
    query=body.get('_scarx_media_query','')
    vietnamese=_vietnamese_query(query)
    def stamp(value):
        whole,millis=divmod(round(float(value)*1000),1000)
        clock=f'{whole//60:02d}:{whole%60:02d}'
        return clock+('.'+f'{millis:03d}'.rstrip('0') if millis else '')
    refs=[]
    for i,item in enumerate(evidence[:4],1):
        start,end=stamp(item['start']),stamp(item['end'])
        refs.append(f'[E{i} · {start}'+(f'–{end}' if end!=start else '')+']')
    label='Dữ liệu tham chiếu' if vietnamese else 'Reference observations'
    if body.get('_scarx_media_kind')=='screen':label+= ' (mốc từ lúc chia sẻ)' if vietnamese else ' (since sharing started)'
    # This footer states which sources were supplied, not that their claims or
    # the generated answer were independently verified. Keep it separately logged.
    return '\n\n'+label+': '+' '.join(refs)
