"""Revalidate old owned observations against current source; no neural calls."""

import ast
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scarx.hybrid.engine import HybridEngine
from scarx.hybrid.learner import ACTIONS
from scarx.evaluation.recovery_policy import ACTION_ORDER


def source_method(path, owner, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    method = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(method, include_attributes=False)


def main():
    run = ROOT / 'runs/20261113T232337Z-incident-information-p0-2d5a24'
    module_spec = importlib.util.spec_from_file_location('_recorded_information_verify', ROOT / 'scripts/verify_hybrid_incident_information.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    original_audit = module.verify(run)
    assert original_audit['verdict'] == 'FINITE_INPUT_COLLISION_VERIFIED'
    archived = run / 'source/src/scarx/hybrid/engine.py'
    current = ROOT / 'src/scarx/hybrid/engine.py'
    method_matches = {}
    for name in ('goal_sequence', 'observe_tool_experience'):
        before, after = source_method(archived, 'HybridEngine', name), source_method(current, 'HybridEngine', name)
        method_matches[name] = before == after
        assert method_matches[name]
    records = json.loads((run / 'result.json').read_text(encoding='utf-8'))['records']
    verified = []
    for record in records:
        old = np.load(run / (record['case_id'] + '-current.npy'), allow_pickle=False)
        # Use the presence flag actually recorded in the original numeric input;
        # no current user's presence, question, files or model are consulted.
        context = SimpleNamespace(present=lambda: bool(old[0, 54]))
        goal = {'question': record['goal_question'], 'reason': record['goal_reason'],
                'public_topic_id': None, **record['goal_flags']}
        actual = HybridEngine.goal_sequence(context, goal)
        np.testing.assert_array_equal(actual, old)
        verified.append({'case_id': record['case_id'], 'matrix_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                         'probe_read_ok': record['actual_probe']['read_ok'],
                         'probe_reported_not_found': record['actual_probe']['reported_not_found']})
    assert len({r['matrix_sha256'] for r in verified}) == 1
    report = {'verdict': 'CURRENT_RUNTIME_STILL_DROPS_RECORDED_PROBE_DISTINCTIONS_BEFORE_SNN',
              'current_source_git_sha': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'current_release_spec_version': json.loads((ROOT/'release/hybrid/payload-files.json').read_text(encoding='utf-8'))['version'],
              'original_run': run.name, 'original_archived_source_and_raw_data_verified': True,
              'current_methods_AST_equal_to_archived': method_matches,
              'reencoded_existing_owned_cases': verified,
              'current_input_shape': [8, 64], 'distinct_actual_probe_outcomes': 3,
              'distinct_current_neural_inputs': 1,
              'same_checkpoint_cannot_distinguish_these_cases_from_this_input_alone': True,
              'not_a_model_accuracy_bound_for_coarse_six_actions': True,
              'production_action_order': list(ACTIONS),
              'research_shadow_action_order': list(ACTION_ORDER),
              'same_six_slots_do_not_authorize_weight_transfer': True,
              'new_model_initializations_forward_backward_or_learning_calls': 0,
              'new_dataset_or_relabeling': False,
              'current_engine_sha256': hashlib.sha256(current.read_bytes()).hexdigest(),
              'scientific_or_production_PASS': False}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    payload = json.dumps(main(), indent=2)
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(payload + '\n')
    print(payload)
