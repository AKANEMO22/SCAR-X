"""Audit an already completed, owned-clip Hermes acceptance without inference."""

import hashlib
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = Path((ROOT / 'local/current-media-presentation-activation.txt').read_text().strip())
    completion = json.loads((run / 'actual-Hermes-result.json').read_text(encoding='utf-8'))
    state = json.loads((run / 'state-preserved.json').read_text(encoding='utf-8'))
    assert completion['completion']['outcome'] == 'completed'
    assert completion['tool_events'] == 0
    assert state['full_saved_arrays_equal'] and state['learned_metadata_RNG_replay_equal']
    delivered = completion['completion']['text']
    database = ROOT / 'local/hybrid-v1/hybrid.sqlite'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute("SELECT id,payload FROM events WHERE kind='answer' ORDER BY id DESC").fetchall()
    matches = [(identifier, json.loads(raw)) for identifier, raw in rows
               if json.loads(raw).get('answer') == delivered]
    assert len(matches) == 1, 'Acceptance answer must identify exactly one stored event'
    identifier, answer = matches[0]
    native, footer = answer['native_answer'], answer['runtime_reference_footer']
    assert delivered == native + footer
    assert native == 'Hình đầu tiên là một hình vuông màu đỏ.'
    assert footer == '\n\nDữ liệu tham chiếu: [E1 · 00:05]'
    assert [(e['start'], e['end']) for e in answer['evidence']] == [(5.0, 5.0)]
    suite = ET.parse(ROOT / 'local/media-presentation-final-release-20261114/python.xml').getroot().find('testsuite')
    assert suite.attrib['tests'] == '263'
    assert all(suite.attrib[key] == '0' for key in ('errors', 'failures', 'skipped'))
    result = {
        'verdict': 'KNOWN_CLIP_HERMES_NATIVE_VI_ANSWER_WITH_SEPARATE_RUNTIME_SOURCE_FOOTER',
        'actual_canonical_Hermes_completion': True,
        'acceptance_event_id': identifier,
        'native_model_answer': native,
        'runtime_supplied_source_footer': footer,
        'source_time_seconds': 5.0,
        'Hermes_tool_events': completion['tool_events'],
        'checkpoint_arrays_RNG_replay_preserved_at_activation': True,
        'live_synaptic_updates_at_activation': state['updates'],
        'Python_release_checks_passed': 263,
        'new_LLM_ASR_calls_in_this_readonly_audit': 0,
        'weights_or_learning_rule_changed': False,
        'known_fixture_only_not_general_video_quality_PASS': True,
        'footer_is_not_a_native_learned_citation_or_truth_verification': True,
        'research_or_production_PASS': False,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('src/scarx/hybrid/gateway.py',
                                       'src/scarx/hybrid/media_reply.py',
                                       'src/scarx/hybrid/hermes_protocol_notes.json')},
        'receipt_sha256': {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                           for name in ('actual-Hermes-result.json', 'actual-Hermes-events.json',
                                        'state-preserved.json')},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
