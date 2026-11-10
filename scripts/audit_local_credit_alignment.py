"""Registered39 readonly audit: is the local credit direction actually aligned?

Applies the SAME direction the real rule applies (mean over endpoints) to a
disposable copy using the checkpoint's own Adam/clip, and compares the real
teacher-forced NLL change against norm-matched random and sign-flipped controls.
No neuraltraining/optimizer/headrefit/newlearner/Torch/serving import.
"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import sys, json, time, hashlib
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
import numpy as np
from direct_words_common import source_data
from scarx.evaluation.word_training_audit import load_reference, protected
from scarx.evaluation.credit_alignment import (
    Applier, case_direction, measure_case, heldout_direction, permuted,
)
from scarx.evaluation.direct_word_data import FAMILIES
from scarx.evaluation.phase21 import interval
from scarx.evaluation.runs import new_run, write_json
from scarx.runtime.experiment_status import progress
from scarx.runtime.work_trace import activity, append
from scarx.runtime.resources import ResourceMonitor
from scarx.runtime.limits import apply_memory_limit

SOURCE = 'runs/20261110T062226Z-direct-word-bridge-v1-a6a908'
DRAWS = 16
CONTROL_SEED = 39039


def aggregate(rows):
    """Macro over 8 families for every registered quantity."""
    def macro(key):
        values = []
        for family in FAMILIES:
            group = [r[key] for r in rows if r['family'] == family and r.get(key) is not None]
            if group:
                values.append(float(np.mean(group)))
        return float(np.mean(values)) if values else None

    result = {
        'cases': len(rows),
        'nll_before': macro('nll_before'),
        'delta_direction': macro('delta_direction'),
        'delta_signflip': macro('delta_signflip'),
        'delta_random_mean': macro('delta_random_mean'),
        'delta_permuted_mean': macro('delta_permuted_mean'),
        'gain_vs_random': macro('gain_vs_random'),
        'gain_vs_permuted': macro('gain_vs_permuted'),
        'gain_vs_signflip': macro('gain_vs_signflip'),
        'alignment_fraction': float(np.mean([r['aligned'] for r in rows])),
        'direction_norm_pre_adam': macro('direction_norm_pre_adam'),
        'applied_norm_direction': macro('applied_norm_direction'),
        'applied_norm_random_mean': macro('applied_norm_random_mean'),
        'applied_norm_permuted_mean': macro('applied_norm_permuted_mean'),
    }
    for control in ('random', 'permuted'):
        d = result[f'applied_norm_direction']; c = result[f'applied_norm_{control}_mean']
        result[f'norm_match_ratio_{control}'] = (c / d if d else None)
    return result


def flags(dev, gain_random_interval, gain_flip_interval, gain_permuted_interval):
    """Pre-registered descriptive flags; never project quality gates."""
    excludes_zero = lambda ci: ci is not None and (ci['lower95'] > 0 or ci['upper95'] < 0)
    return {
        'LOCAL_DIRECTION_ALIGNED': bool(dev['alignment_fraction'] > .5
                                        and excludes_zero(gain_flip_interval)
                                        and gain_flip_interval['lower95'] > 0),
        'DIRECTION_UNINFORMATIVE': bool(not excludes_zero(gain_permuted_interval)),
        'DIRECTION_CONFLICT_CONFIRMED': bool(excludes_zero(gain_flip_interval)
                                             and gain_flip_interval['upper95'] < 0),
        'GAUSSIAN_CONTROL_OVERSHOOTS': bool(dev['norm_match_ratio_random'] is not None
                                            and dev['norm_match_ratio_random'] > 1.5),
    }


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    cap = apply_memory_limit(); began = time.perf_counter()
    receipt, train, dev, vocab, files = source_data()
    source = json.loads((root / SOURCE / 'result.json').read_text(encoding='utf-8'))
    assert source['decision'] == 'STOP_DIRECT_WORD_BRIDGE_V1' and len(source['trials']) == 15
    for r in source['trials']:
        files.extend([r['checkpoint'], SOURCE + f"/word-{r['seed']}-{r['method']}.json"])

    run = new_run(root, 'local-credit-alignment-audit-v1', {
        'protocol': 'experiments/local-credit-alignment-audit-v1.md',
        'display_title': 'READ ONLY39 · tín hiệu học cục bộ có đúng hướng?',
        'model_scope': '15existing36checkpoints. Applies own-neuron local direction to a DISPOSABLE copy; compares real NLL change vs norm-matched random/signflip. No training/optimizer/headfit/Transformer.',
        'source_models': 15, 'cases_per_source': 128, 'neural_training': False, 'neural_update_steps': 0,
        'control_draws': DRAWS, 'source_run': SOURCE},
        data_description='Pinned reviewed actual AGY128 questions / 8 owned styles, no new labels/teacher/Qwen/private inputs. Existing 15 model source weights read only; all applications on deep copies.',
        data_files=files + [SOURCE + '/result.json'])
    (run / 'report.md').write_text(
        '# Readonly local credit alignment audit39\n\nDiagnostics in progress, no neural training.\n'
        'Applied directions are disposable-copy counterfactuals, NOT training or free answers.\n',
        encoding='utf-8')
    write_json(run / 'answer-source.json', {
        'source_run': SOURCE,
        'scope': 'Archived36 free word outputs; original file timestamps. NOT the alignment counterfactual answers.'})
    activity(run, 'Bắt đầu39: tín hiệu cục bộ có giảm loss thật?',
             'Áp chính hướng local (trung bình endpoint, Adam/clip của checkpoint) lên bản sao dùng-một-lần, đo ΔNLL teacher-forced thật so với control cùng norm (random/signflip). Không huấn luyện.',
             status='started',
             evidence={'protocol': 'experiments/local-credit-alignment-audit-v1.md',
                       'neural_updates': 0, 'source_models': 15, 'control_draws': DRAWS})

    records = []; table = []; forward_calls = 0; applied_calls = 0
    with ResourceMonitor(interval=.1) as monitor:
        for seed in (36313, 36319, 36341):
            trial = next(r for r in source['trials'] if r['seed'] == seed and r['method'] == 'correct')
            path = root / trial['checkpoint']; payload = path.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == trial['sha256']
            m = load_reference(payload); before = protected(m)
            applier = Applier(m); rng = np.random.default_rng(CONTROL_SEED + seed)
            rows = []; train_dirs = []
            for index, case in enumerate(train + dev, 1):
                gold = m.target_tokens(case['target'])
                direction = case_direction(m, case['prompt'], gold); forward_calls += 1
                if np.linalg.norm(direction.astype(np.float64)) > 0:
                    result = measure_case(applier, case['prompt'], case['target'], direction, rng, DRAWS)
                    applied_calls += 1 + 1 + 1 + DRAWS
                else:
                    result = None
                if case['split'] == 'train' and direction is not None:
                    train_dirs.append((case, direction))
                row = {'id': case['id'], 'split': case['split'], 'family': case['family'],
                       'prompt': case['prompt'], 'target': case['target'],
                       'direction_is_zero': result is None,
                       'scope': 'Disposable-copy counterfactual, NOT training and NOT a free answer.'}
                row.update(result or {k: None for k in (
                    'nll_before', 'nll_direction', 'nll_signflip', 'delta_direction',
                    'delta_signflip', 'delta_random_mean', 'delta_random_values',
                    'delta_permuted_mean', 'delta_permuted_values',
                    'gain_vs_random', 'gain_vs_permuted', 'gain_vs_signflip', 'aligned',
                    'direction_norm_pre_adam', 'applied_norm_direction',
                    'applied_norm_signflip', 'applied_norm_random_mean',
                    'applied_norm_permuted_mean')})
                rows.append(row)
                if index in (1, 65):
                    append(run, 'answer', {
                        'scope': f'READ ONLY39 counterfactual · seed{seed}/{case["split"]} · NOT MAIN',
                        'case_id': case['id'], 'prompt': case['prompt'], 'target': case['target'],
                        'generated': ('zero direction: nothing applied' if result is None else
                                      f'delta_direction={result["delta_direction"]:.6f} vs random={result["delta_random_mean"]:.6f}'),
                        'neural_ticks': 0, 'seconds': 0.0, 'active': 48, 'allocated': 48,
                        'checkpoint': trial['checkpoint'],
                        'input_scope': 'Fixed36 checkpoint; local direction applied to disposable copy only. No learning, no free answer.'})
                if index % 32 == 0:
                    progress(run, f'READ ONLY39 seed{seed} {index}/128', len(records), 15,
                             metrics={'seed': seed, 'cases': index, 'neural_updates': 0,
                                      'seconds': time.perf_counter() - began})
                    monitor.check()
                    if time.perf_counter() - began > 300:
                        raise TimeoutError('Registered39 audit300s')

            # Held-out test: accumulate over TRAIN only, apply once, score on dev.
            dev_rows = [r for r in rows if r['split'] == 'development' and not r['direction_is_zero']]
            heldout = None
            if train_dirs and dev_rows:
                combined = heldout_direction(applier, [c for c, _ in train_dirs], [d for _, d in train_dirs])
                if np.linalg.norm(combined.astype(np.float64)) > 0:
                    deltas = []
                    for row in dev_rows:
                        base, _ = applier.measure(row['prompt'], row['target'], None)
                        after, applied = applier.measure(row['prompt'], row['target'], combined)
                        deltas.append({'id': row['id'], 'family': row['family'],
                                       'nll_before': base, 'nll_after': after,
                                       'delta': after - base, 'applied_norm': applied})
                        applied_calls += 2
                    heldout = {'train_cases': len(train_dirs), 'dev_cases': len(deltas),
                               'dev_mean_delta': float(np.mean([d['delta'] for d in deltas])),
                               'dev_improved_fraction': float(np.mean([d['delta'] < 0 for d in deltas])),
                               'rows': deltas,
                               'scope': 'Train-aggregated direction applied ONCE; dev labels used ONLY for scoring.'}

            assert protected(m) == before and m.core.updates == trial['word_updates']
            summary = {split: aggregate([r for r in rows if r['split'] == split])
                       for split in ('train', 'development')}
            record = {'seed': seed, 'method': 'correct', 'summary': summary, 'heldout': heldout,
                      'source_checkpoint': trial['checkpoint'], 'source_sha256': trial['sha256'],
                      'source_protected_unchanged': True, 'neural_updates': 0,
                      'control_draws': DRAWS, 'control_seed': CONTROL_SEED + seed}
            write_json(run / f'{seed}.alignment.json', {'record': record, 'cases': rows})
            records.append(record)
            for split in ('train', 'development'):
                v = summary[split]
                table.append({'seed': seed, 'split': split,
                              **{k: v[k] for k in ('cases', 'nll_before', 'delta_direction',
                                                   'delta_signflip', 'delta_random_mean',
                                                   'delta_permuted_mean', 'gain_vs_random',
                                                   'gain_vs_permuted', 'gain_vs_signflip',
                                                   'alignment_fraction',
                                                   'norm_match_ratio_random',
                                                   'norm_match_ratio_permuted')}})
            progress(run, f'READ ONLY39 seed{seed} done, no updates', len(records), 15,
                     metrics={'neural_updates': 0,
                              'dev_alignment': summary['development']['alignment_fraction'],
                              'dev_gain_vs_random': summary['development']['gain_vs_random']})

        assert 'torch' not in sys.modules
        for r in source['trials']:
            assert hashlib.sha256((root / r['checkpoint']).read_bytes()).hexdigest() == r['sha256']

        dev_gain_random = [r['summary']['development']['gain_vs_random'] for r in records]
        dev_gain_permuted = [r['summary']['development']['gain_vs_permuted'] for r in records]
        dev_gain_flip = [r['summary']['development']['gain_vs_signflip'] for r in records]
        random_interval = interval(dev_gain_random)
        permuted_interval = interval(dev_gain_permuted)
        flip_interval = interval(dev_gain_flip)
        dev_macro = {k: float(np.mean([r['summary']['development'][k] for r in records]))
                     for k in ('alignment_fraction', 'gain_vs_random', 'gain_vs_permuted',
                               'gain_vs_signflip', 'delta_direction', 'delta_random_mean',
                               'delta_permuted_mean', 'delta_signflip',
                               'norm_match_ratio_random', 'norm_match_ratio_permuted')}
        dev_macro['cases'] = int(sum(r['summary']['development']['cases'] for r in records))
        heldout_deltas = [r['heldout']['dev_mean_delta'] for r in records if r['heldout']]
        result = {
            'run': run.relative_to(root).as_posix(), 'source_run': SOURCE,
            'decision': 'LOCAL_CREDIT_ALIGNMENT_DIAGNOSTIC_ONLY',
            'alignment_rows': table, 'records': records,
            'dev_macro': dev_macro,
            'dev_gain_vs_random_interval': random_interval,
            'dev_gain_vs_permuted_interval': permuted_interval,
            'dev_gain_vs_signflip_interval': flip_interval,
            'heldout': {'seeds': len(heldout_deltas),
                        'mean_dev_delta': float(np.mean(heldout_deltas)) if heldout_deltas else None,
                        'interval': interval(heldout_deltas) if len(heldout_deltas) > 1 else None},
            'flags': flags(dev_macro, random_interval, flip_interval, permuted_interval),
            'model_sources': 15, 'public_cases_per_source': 128,
            'actual_forward_segments': forward_calls, 'applied_counterfactual_evaluations': applied_calls,
            'neural_training': False, 'neural_update_steps': 0, 'torch_imported': False,
            'deployed': False, 'main_weights_changed': False, 'normal_chat_or_vision_proven': False,
            'resources': monitor.report(), 'cap': cap, 'seconds': time.perf_counter() - began,
            'limits': [
                '15existing checkpoints / 3 paired seeds only; CIs are exploratory, not 15 independent seeds.',
                'Single-step local linear probe of ONE applied update; repeated application NOT tested.',
                'Direction is one-step readout-error x own-neuron eligibility; no gradient/backward computed or approximated.',
                'Applied to a disposable deep copy; source checkpoints and main weights unchanged.',
                'No newlearner/objective/epoch/LR/grid from this audit; any credit-rule change needs its own protocol and approval.',
                'Existing STOP35/36/37 and P1/P2 flags unchanged; this diagnoses a mechanism, it does not certify language ability.']}
        (run / 'report.md').write_text(
            '# Local credit alignment audit39\n\nNo neural training. Directions applied to disposable copies only.\n\n'
            + json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2) + '\n', encoding='utf-8')
        write_json(run / 'result.json', result)
        write_json(root / 'artifacts/local-credit-alignment.json', result)
        activity(run, 'Hoàn tất39: đo hướng tín hiệu học cục bộ',
                 f"3seeds×128cases×2split. Alignment dev={dev_macro['alignment_fraction']:.4f}, gain_vs_permuted={dev_macro['gain_vs_permuted']:+.6f} (CI{permuted_interval['lower95']:+.6f}..{permuted_interval['upper95']:+.6f}), gain_vs_signflip={dev_macro['gain_vs_signflip']:+.6f}. Flags={result['flags']}. Source checkpoints unchanged, 0 neural updates.",
                 evidence={'decision': result['decision'], 'neural_updates': 0,
                           'flags': result['flags'], 'applied_evaluations': applied_calls})
        print(json.dumps({'run': result['run'], 'dev_macro': dev_macro, 'flags': result['flags'],
                          'heldout': result['heldout'], 'resources': result['resources']},
                         ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
