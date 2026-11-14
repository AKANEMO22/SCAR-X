"""Causally bind supplied web-tool snippets to an answer; never truth labels."""

import hashlib
import ipaddress
import json
import re
import time
import unicodedata
from urllib.parse import urlsplit

from .store import encode
from .tool_routing import task_user_index, text_content


def task_user_message(messages):
    index = task_user_index(messages)
    return messages[index] if index is not None else next(m for m in reversed(messages) if m.get('role') == 'user')


def task_interpretation(question):
    """Explicit colloquial query context, not a rewritten user row or a fact."""
    normalized = ''.join(c for c in unicodedata.normalize('NFKD', question.casefold())
                         if not unicodedata.combining(c)).replace('đ', 'd')
    expanded = question
    if re.search(r'\b(?:tap|chuong|chap|phan|mua|gia|co|duoc|la)\s+bn\b', normalized):
        expanded = re.sub(r'\bbn\b', 'bao nhiêu', question)
    quantity = (expanded != question
                or bool(re.search(r'\b(?:bao nhieu|how many|which episode|what episode)\b', normalized))
                or bool(re.search(r'\bmấy\b', question.casefold())))
    return {'format': 'scarx-public-task-context-v1', 'original_question': question[:1000],
            'expanded_question': expanded[:1000], 'question_is_excerpt': len(question) > 1000,
            'requested_value': 'quantity_or_index' if quantity else 'answer_the_original_question',
            'interpretation_is_not_factual_evidence': True}


def _source_url(value):
    if not isinstance(value, str) or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ('https', 'http') or not parsed.hostname
                or parsed.username or parsed.password or parsed.port not in (None, 80, 443)
                or parsed.hostname == 'localhost' or parsed.hostname.endswith(('.local', '.internal'))):
            return None
        try:
            if not ipaddress.ip_address(parsed.hostname).is_global:
                return None
        except ValueError:
            pass
    except ValueError:
        return None
    return value


def supplied_web_sources(messages):
    """Only current-turn tool results paired to actual web_search call IDs.

    Client-supplied tool messages are not an independent factual verifier. No
    URL is fetched here, no prior-turn or unpaired tool text is promoted.
    """
    latest = task_user_index(messages)
    if latest is None:
        return [], {'paired_results': 0, 'partial': False}
    calls = {}
    for message in messages[latest + 1:]:
        if message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            if not isinstance(call, dict):
                continue
            function = call.get('function') or {}
            if not isinstance(function, dict):
                continue
            identifier = call.get('id')
            if function.get('name') != 'web_search' or not isinstance(identifier, str) or not 1 <= len(identifier) <= 256:
                continue
            try:
                args = json.loads(function.get('arguments', '{}'))
            except (ValueError, TypeError):
                continue
            query = args.get('query') if isinstance(args, dict) else None
            if isinstance(query, str) and 0 < len(query) <= 1000:
                calls[identifier] = query
    entries, seen, paired = [], set(), 0
    partial = False
    remaining_bytes = 36_000
    # Retain newest responses first; retain source ranking within a response.
    for message in reversed(messages[latest + 1:]):
        call = message.get('tool_call_id')
        if message.get('role') != 'tool' or not isinstance(call, str) or call not in calls:
            continue
        paired += 1
        raw = text_content(message.get('content', ''))
        if len(raw) > 250_000:
            partial = True
            continue
        try:
            result = json.loads(raw)
            if not isinstance(result, dict) or result.get('success') is not True:
                partial = True
                continue
            hits = result.get('data', {}).get('web', [])
            if not isinstance(hits, list):
                raise ValueError('Not a source list')
        except (ValueError, TypeError, AttributeError):
            partial = True
            continue
        for hit in hits[:20]:
            if not isinstance(hit, dict):
                partial = True
                continue
            url = _source_url(hit.get('url'))
            title, description = hit.get('title', ''), hit.get('description', '')
            if not url or not isinstance(title, str) or not isinstance(description, str):
                partial = True
                continue
            if (call, url) in seen:
                continue
            seen.add((call, url))
            text = (title[:300] + '\n' + description[:2400]).strip()
            if len(title) > 300 or len(description) > 2400:
                partial = True
            if not text:
                continue
            if len(entries) >= 12:
                partial = True
                continue
            entry = {'tool_call_id': call, 'query': calls[call], 'source': url, 'text': text}
            size = len(encode(entry).encode())
            if size > remaining_bytes:
                partial = True
                continue
            entries.append(entry)
            remaining_bytes -= size
        if len(hits) > 20:
            partial = True
    return entries, {'paired_results': paired, 'partial': partial,
                     'kind': 'supplied search snippets, not full-page or factual verification'}


def persist_web_sources(store, entries, session):
    """Atomic evidence/FTS/provenance journal; idempotent same supplied result."""
    scope = hashlib.sha256(str(session).encode()).hexdigest()[:24]
    observations = []
    with store.lock, store.db:
        for entry in entries:
            content_digest = hashlib.sha256(entry['text'].encode()).hexdigest()
            # A host may supply a shorter view after context compression. Keep
            # each view immutable instead of overwriting it or rejecting chat.
            key = encode([scope, entry['tool_call_id'], entry['query'], entry['source'], content_digest])
            identifier = hashlib.sha256(key.encode()).hexdigest()[:32]
            old = store.db.execute('SELECT * FROM evidence WHERE id=?', (identifier,)).fetchone()
            if old is not None:
                item = dict(old)
                if (item['text'] != entry['text'] or item['source'] != entry['source']
                        or item['stream'] != 'web-turn:' + scope or item['status'] != 'observed'
                        or item['modality'] != 'web_search_snippet'):
                    raise ValueError('Web tool result identity changed across retry')
            else:
                item = {'id': identifier, 'stream': 'web-turn:' + scope, 'start': 0.0, 'end': 0.0,
                        'text': entry['text'], 'source': entry['source'], 'modality': 'web_search_snippet',
                        'status': 'observed', 'public_topic': None, 'created': time.time(),
                        'digest': content_digest}
                store.db.execute('INSERT INTO evidence VALUES(:id,:stream,:start,:end,:text,:source,:modality,:status,:public_topic,:created,:digest)', item)
                store.db.execute('INSERT INTO evidence_fts VALUES(?,?)', (identifier, item['text']))
                provenance = {'evidence_id': identifier, 'tool_call_id': entry['tool_call_id'],
                              'query': entry['query'], 'factual_verification': False,
                              'content_view_sha256': content_digest,
                              'timebase': 'gateway observation time, not claimed source publication time'}
                store.db.execute('INSERT INTO objects VALUES(?,?,?)', ('web_source_provenance', identifier, encode(provenance)))
                store.db.execute('INSERT INTO events(created,actor,kind,payload) VALUES(?,?,?,?)',
                    (time.time(), 'runtime', 'evidence_observed', encode({**item, 'text': item['text'][:2048],
                                                                      'factual_verification': False})))
            observations.append(item)
    return observations
