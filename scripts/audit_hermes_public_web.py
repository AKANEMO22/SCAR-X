"""Audit already completed canonical web turns, without searching/inference."""

import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = Path((ROOT/'local/current-public-web-acceptance.txt').read_text().strip())
    receipt = json.loads((run/'result.json').read_text(encoding='utf-8'))
    assert receipt['completed'] and len(receipt['turns']) == 2
    expected = ('jojo ra tập mới nhất là tập bn', 'tìm kiếm đi')
    turns = []
    for query, turn in zip(expected, receipt['turns']):
        assert turn['query'] == query and turn['completion']['outcome'] == 'completed'
        assert [t['name'] for t in turn['tool_calls']] == ['web_search']
        assert turn['tool_calls'][0]['args']['query'] == expected[0]
        assert len(turn['tool_results']) == 1 and not turn['tool_results'][0]['is_error']
        raw = turn['tool_results'][0]['result']
        result = json.loads(raw) if isinstance(raw, str) else raw
        assert result['success'] is True
        hits = result['data']['web']
        assert hits and all(h['url'].startswith(('https://', 'http://')) for h in hits)
        answer = turn['completion']['text']
        turns.append({'query': query, 'executed_tool': 'web_search',
                      'actual_search_query': expected[0], 'returned_results': len(hits),
                      'result_urls': [h['url'] for h in hits], 'native_answer': answer,
                      'native_answer_is_not_independently_fact_verified': True,
                      'episode_number_after_episode_word_present': bool(re.search(r'\b(?:tập|episode|ep)\s*(?:số\s*)?\d+', answer, re.IGNORECASE))})
    report = {'verdict': 'ACTUAL_HERMES_WEB_SEARCH_AND_USER_TOPIC_FOLLOWUP_VERIFIED',
              'canonical_Hermes_chat': True, 'turns': turns,
              'local_file_or_paid_extractor_tool_calls': 0,
              'reported_false_capability_refusal_repeated': any('chỉ có thể hỗ trợ' in t['native_answer'].lower() for t in turns),
              'remaining_answer_quality_issue': 'Both outputs provide series/launch date instead of the requested current episode number; search routing success is not answer-quality success',
              'new_search_or_inference_calls_in_audit': 0,
              'raw_receipts_sha256': {name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ('result.json', 'events.json')},
              'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('src/scarx/hybrid/web_routing.py', 'src/scarx/hybrid/gateway.py')},
              'research_or_production_PASS': False}
    path = ROOT/'docs/release/evidence/ACTUAL_HERMES_WEB_SEARCH_20261115.json'
    with path.open('x', encoding='utf-8') as handle:json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
