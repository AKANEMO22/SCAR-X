"""Public-web turn routing, not neural learning or a fabricated search result."""

import json
import re
import unicodedata
from .tool_routing import task_user_index, text_content

# This route is a SearXNG search request. An extraction provider can be a
# separate commercial service; do not automatically escalate this search to it.
# Other turns retain the original Hermes tool set, including explicit page reads.
WEB_TOOLS = {'web_search'}


def plain(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold())
                   if not unicodedata.combining(c)).replace('đ', 'd').strip()


def followup(text):
    return bool(re.fullmatch(
        r'(?:tim(?: kiem)?(?: lai| tiep)?(?: (?:tren )?(?:web|mang))?(?: di| giup| giup toi)?|'
        r'tra(?: cuu)?(?: lai)?(?: di)?|search(?: it| again)?(?: (?:the )?(?:web|online))?(?: please)?|'
        r'look it up)[.!?\s]*', plain(text)))


def public_web_scope(messages, tools, requested=None):
    if requested not in (None, 'auto') or not tools:
        return None
    if not any(t.get('function', {}).get('name') == 'web_search' for t in tools):
        return None
    latest = task_user_index(messages)
    if latest is None:
        return None
    source = latest
    text = text_content(messages[source].get('content', ''))
    # Resolve only a short chain of bare search follow-ups. Never skip a newer
    # substantive local/private request to reuse an older public topic.
    for _ in range(4):
        if not followup(text):
            break
        source = next((i for i in range(source-1, -1, -1)
                       if messages[i].get('role') == 'user'), None)
        if source is None:
            return None
        text = text_content(messages[source].get('content', ''))
    if followup(text) or not 3 <= len(text) <= 600:
        return None
    value = plain(text)
    if value in {'chao ban', 'xin chao', 'hi', 'hello', 'ok', 'cam on', 'thanks'}:
        return None
    if re.search(r"\b(?:(?:khong|dung)(?: can| duoc)? (?:tim(?: kiem)?(?! thay| duoc)|tra cuu)|"
                 r"(?:do not|dont|don't|never) (?:search|look up))\b", value):
        return None
    # Do not turn local evidence, paths or control-frame references into an
    # automatic remote query. Explicit web intent still cannot export these.
    if (re.search(r'(?:[a-z]:[\\/]|/(?:users|home|etc)/|[\w.+-]+@[\w.-]+\.[a-z]{2,})', value)
            or re.search(r'\b(?:mat khau|password|api[ _-]?key|credential|secret|private|'
                         r'so dien thoai|dia chi nha|ten toi|video nay|clip nay|anh nay|'
                         r'man hinh|this video|this image|screen)\b', value)
            or any(marker in value for marker in ('[prior context', '[context compaction', '[out-of-band'))):
        return None
    explicit = bool(re.search(r'\b(?:tim(?: kiem)? (?:tren )?(?:web|mang)|tra(?: cuu)? (?:tren )?(?:web|mang)|'
                              r'search (?:the )?(?:web|online)|look up online)\b', value))
    local = bool(re.search(r'\b(?:workspace|folder|file|thu muc|project|du an|repo|repository|'
                           r'terminal|debug|fix|sua loi|viet code)\b', value))
    fresh = bool(re.search(r'\b(?:moi nhat|hien nay|hien tai|hom nay|latest|most recent|'
                           r'current (?:episode|version|release|price|weather|news))\b', value)
                 or re.search(r'\b(?:ra|da ra|ra toi|ra den)\b.{0,50}\b(?:tap|season|seasion|chap|chuong)\b', value))
    if not explicit and (local or not (fresh or source != latest)):
        return None
    if explicit and not re.sub(r'\b(?:tim|kiem|tra|cuu|tren|web|mang|search|the|online|look|up|di)\b', '', value).strip(' .!?'):
        return None
    calls = [call for message in messages[latest+1:] if message.get('role') == 'assistant'
             for call in message.get('tool_calls') or []]
    return {'query': text, 'source_user_index': source, 'latest_user_index': latest,
            'followup_resolved': source != latest, 'explicit_web': explicit,
            'web_calls': sum(c.get('function', {}).get('name') in WEB_TOOLS for c in calls),
            'search_already_requested': any(c.get('function', {}).get('name') == 'web_search' for c in calls)}


def web_prelude(messages, tools, requested=None):
    scope = public_web_scope(messages, tools, requested)
    if not scope or scope['search_already_requested']:
        return None
    return {'name': 'web_search', 'arguments': json.dumps({'query': scope['query']}, ensure_ascii=False),
            'event': {k: v for k, v in scope.items() if k not in ('web_calls', 'search_already_requested')}
            | {'selector': 'runtime public-web first-tool route; not learned SNN policy',
               'query_from_user_only': True, 'model_answer_not_generated_here': True}}
