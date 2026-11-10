"""Probe live chat quality with explicit UTF-8, no shell encoding in the path."""
import json
import sys
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE = 'http://127.0.0.1:8765/v1/chat/completions'
PROMPTS = ['hi', 'hello', 'chào bạn', 'bạn là ai', 'bạn tên gì', 'cảm ơn', 'tạm biệt', 'chào']


def ask(prompt):
    body = json.dumps({'model': 'scarx-generation1',
                       'messages': [{'role': 'user', 'content': prompt}]},
                      ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(BASE, body, {'Content-Type': 'application/json; charset=utf-8'})
    data = json.load(urllib.request.urlopen(req, timeout=90))
    content = data['choices'][0]['message']['content']
    return content, data['choices'][0].get('finish_reason'), data.get('choices', [{}])[0]


def cycles(words):
    """Longest repeated adjacent block length, 0 if none."""
    for size in range(1, max(1, len(words) // 2) + 1):
        for start in range(len(words) - 2 * size + 1):
            if words[start:start + size] == words[start + size:start + 2 * size]:
                return size
    return 0


for prompt in PROMPTS:
    text, finish, raw = ask(prompt)
    words = text.split()
    print(f'--- {prompt!r}')
    print(f'    out      = {text!r}')
    print(f'    finish   = {finish}')
    print(f'    words    = {len(words)}')
    print(f'    cycle_ln = {cycles(words)}')
    routed = text.startswith('[Hỏi để học]')
    print(f'    routed   = {routed}')
