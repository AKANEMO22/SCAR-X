"""One-command product acceptance suite measuring the 12 required metrics.

Run:
    python scripts/acceptance_suite.py                # baseline only
    python scripts/acceptance_suite.py --teach        # baseline + taught state + restart

Every run happens on a discarded copy; the live session is never opened for
writing, and its runtime SHA is sampled before and after only to show it moved
for its own reasons (the idle loop), not because of this suite.

No learning rule, topology, signal, guard threshold or architecture is changed.
"""
import argparse
import os
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Bound this CLI's numerical workers before importing Generation/NumPy. This
# changes processresource use only, never the user's persistent environment.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.path.insert(0, str(ROOT / 'src'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scarx.app.generation import Generation, normalized  # noqa: E402
from scarx.evaluation.product_acceptance import (  # noqa: E402
    CASES, HELDOUT, copy_session, guard_report, measure, summarise, weights_sha256,canonical_reply,resources,
)

LIVE = ROOT / 'local' / 'generation1'

# Product teaching set: only prompts whose expected answer the product should give.
TEACH_SET = [
    ('bạn là ai', 'Tôi là SCAR-X.'),
    ('bạn tên gì', 'Tôi là SCAR-X.'),
    ('chào bạn', 'Chào bạn.'),
    ('cảm ơn', 'Không có gì.'),
    ('tạm biệt', 'Tạm biệt bạn.'),
    ('bạn khỏe không', 'Tôi đang hoạt động.'),
]


def live_sha():
    try:
        db = sqlite3.connect(f'file:{(LIVE / "generation1.sqlite").as_posix()}?mode=ro', uri=True)
        row = db.execute('SELECT sha FROM runtime LIMIT 1').fetchone()
        db.close()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def heldout(app,metric_version='product-canonical-v2'):
    if metric_version!='legacy-keywords-v1':
        references={'chào':'Chào bạn.','scar-x':'Tôi là SCAR-X.','cảm ơn':'Không có gì.','tạm biệt':'Tạm biệt bạn.'}
        rows=measure(app,[(p,e,(e,),references[e]) for p,e in HELDOUT],metric_version=metric_version)
        return summarise(rows)['free_answer_correct'],rows
    rows = []
    for prompt, expect in HELDOUT:
        text = app.model.generate(prompt)['text']
        rows.append({'prompt': prompt, 'expect': expect, 'output': text,
                     'correct': expect in ' '.join(text.lower().split())})
    return sum(r['correct'] for r in rows) / max(1, len(rows)), rows


def snapshot(app,metric_version='product-canonical-v2'):
    rows = measure(app, CASES,metric_version=metric_version)
    g_rows = guard_report(app, '__acceptance__')
    h_acc, h_rows = heldout(app,metric_version)
    return {
        'weights_sha256': weights_sha256(app.model),
        'summary': summarise(rows), 'rows': rows,
        'guard': g_rows, 'guard_passing': sum(g['pass'] for g in g_rows),
        'guard_total': len(g_rows),
        'heldout_acc': h_acc, 'heldout_rows': h_rows,
        'correct_prompts': sorted(r['prompt'] for r in rows if r['correct']),
        'rss_bytes': app.process.memory_info().rss,
        'vms_bytes': getattr(app.process.memory_info(), 'vms', None),
        'metric_version':metric_version,'paraphrase_scope':'Six reused developmentphrases, NOT unseenfinaltest',
        'resource_measurement':resources(app),
    }


def teach_sequence(app, lessons):
    results = []
    for prompt, answer in lessons:
        before_text=canonical_reply(app.model,prompt)['text'];before=Generation._pair_verdict(before_text,answer)
        rec = {'prompt': prompt, 'answer': answer, 'accepted': False,'before_generated':before_text,
            'baseline_target_complete':before['complete_ok'],'baseline_target_content':before['content_ok']}
        try:
            out = app.teach({'prompt': prompt, 'answer': answer,
                             'repeats': 16, 'confirmation': True})
            rec['accepted'] = True
            m = out.get('learning_metrics', {})
            rec['guard_lost'] = m.get('guard_lost')
            rec['loss_mean'] = m.get('loss_mean')
        except Exception as exc:
            rec['error'] = f'{type(exc).__name__}'
            rec['guard_lost'] = getattr(exc, 'metrics', {}).get('guard_lost') \
                if hasattr(exc, 'metrics') else None
        after_text=canonical_reply(app.model,prompt)['text'];after=Generation._pair_verdict(after_text,answer)
        rec['after_generated']=after_text;rec['target_content_after_attempt']=after['content_ok']
        rec['new_content_acquired']=not before['content_ok'] and after['content_ok']
        rec['target_complete_after_attempt']=after['complete_ok']
        rec['newly_acquired']=not before['complete_ok'] and after['complete_ok']
        rec['accepted_without_acquisition']=rec['accepted'] and not rec['newly_acquired']
        results.append(rec)
    return results


def run(do_teach, do_restart,metric_version='product-canonical-v2'):
    work = copy_session(LIVE)
    out = {}
    try:
        app = Generation(work, background=False)
        try:
            base = snapshot(app,metric_version)
            out['baseline'] = base
            if do_teach:
                acquisition = teach_sequence(app, TEACH_SET)
                for lesson in acquisition:
                    final=Generation._pair_verdict(canonical_reply(app.model,lesson['prompt'])['text'],lesson['answer'])
                    lesson['target_complete_final']=final['complete_ok'];lesson['target_content_final']=final['content_ok']
                taught = snapshot(app,metric_version)
                out['acquisition'] = acquisition
                out['taught'] = taught
                out['newly_lost'] = sorted(set(base['correct_prompts']) - set(taught['correct_prompts']))
                out['newly_gained'] = sorted(set(taught['correct_prompts']) - set(base['correct_prompts']))
                out['weights_changed'] = base['weights_sha256'] != taught['weights_sha256']
        finally:
            app.close()

        if do_restart:
            time.sleep(0.2)
            reopened = Generation(work, background=False)
            try:
                out['after_restart'] = snapshot(reopened,metric_version)
            finally:
                reopened.close()
    finally:
        cleanup=work.parent.resolve()
        if cleanup.parent!=Path(tempfile.gettempdir()).resolve() or not cleanup.name.startswith('scarx-acc-') or work.name!='session':raise ValueError('Refusing cleanup outside owned acceptance scratchdirectory')
        shutil.rmtree(cleanup, ignore_errors=True)
    return out


def metres(out):
    """Map to the 12 required metrics, each explicitly measured or explicitly absent."""
    base = out['baseline']
    taught = out.get('taught')
    restart = out.get('after_restart')
    acq = out.get('acquisition', [])
    m = {}
    m['1_free_answer_accuracy'] = base['summary']['free_answer_correct']
    m['2_heldout_generalization'] = None
    m['2b_reused_development_paraphrases'] = base['heldout_acc']
    m['2_evaluation_scope']='Six previously inspected developmentphrases; unseenfinalgeneralization NOT measured'
    if acq:
        eligible=[a for a in acq if a.get('baseline_target_complete') is False]
        m['3_learning_acquisition'] = sum(a.get('newly_acquired',False) for a in eligible)/len(eligible) if eligible else None
        m['3b_acquisition_denominator']=len(eligible)
        m['3c_final_new_targets_retained']=sum(a.get('target_complete_final',False) for a in eligible)/len(eligible) if eligible else None
        content_eligible=[a for a in acq if a.get('baseline_target_content') is False]
        m['3d_content_acquisition']=sum(a.get('new_content_acquired',False) for a in content_eligible)/len(content_eligible) if content_eligible else None
        m['3e_content_acquisition_denominator']=len(content_eligible)
        m['3_acquisition_scope']='Primary: newlycompletecanonicalreply, not broadskill; content-onlychange reportedseparately'
        m['5_guard_acceptance'] = sum(a['accepted'] for a in acq) / len(acq)
        m['3_legacy_accepted_as_acquisition']=m['5_guard_acceptance']
        m['6_guard_regression'] = len(out.get('newly_lost', []))
    else:
        m['3_learning_acquisition'] = None
        m['5_guard_acceptance'] = None
        m['6_guard_regression'] = None
    m['4_retention'] = (len(set(base['correct_prompts']) & set(taught['correct_prompts']))
                        / len(base['correct_prompts'])) if taught and base['correct_prompts'] else None
    m['7_guard_passing'] = f"{base['guard_passing']}/{base['guard_total']}"
    m['8_premature_stop'] = base['summary']['premature_stop']
    m['9_repetition'] = base['summary']['repetition']
    m['10_rss_bytes'] = base['rss_bytes']
    m['10_vms_bytes'] = base['vms_bytes']
    m['10_under_2gb'] = base.get('resource_measurement',{}).get('under_limit')
    m['10_commit_bytes']=base.get('resource_measurement',{}).get('commit_bytes')
    m['10_peak_rss_bytes']=base.get('resource_measurement',{}).get('peak_rss_bytes')
    m['10_peak_commit_bytes']=base.get('resource_measurement',{}).get('peak_commit_bytes')
    m['11_restart_persistence'] = (
        # Compare WEIGHTS, which must persist. Text comparison is unreliable
        # because the recurrent working state legitimately keeps moving, so a
        # text mismatch is not evidence of lost learning.
        taught['weights_sha256'] == restart['weights_sha256']
        if (taught and restart) else None)
    m['11b_restart_text_identical'] = (
        [r['text'] for r in taught['rows']] == [r['text'] for r in restart['rows']]
        if (taught and restart) else None)
    m['12_weights_proven'] = 'weights_sha256 (not the updates counter)'
    m['metric_version']=base.get('metric_version','legacy-keywords-v1')
    m['thread_environment']={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teach', action='store_true', help='also run the teaching sequence')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--metric-version',choices=['product-canonical-v2','legacy-keywords-v1'],default='product-canonical-v2')
    args = ap.parse_args()

    before = live_sha()
    out = run(args.teach, args.teach,args.metric_version)

    print('=' * 68)
    print('SCAR-X PRODUCT ACCEPTANCE')
    print('=' * 68)
    base = out['baseline']
    print(f"\nweights_sha256 : {base['weights_sha256']}")
    print(f"{'prompt':16s} {'correct':8s} {'complete':9s} output")
    for r in base['rows']:
        print(f"{r['prompt']:16s} {str(r['correct']):8s} {str(r['complete']):9s} {r['text'][:38]!r}")

    if args.teach:
        print(f"\nteaching sequence ({len(TEACH_SET)} lessons):")
        for a in out['acquisition']:
            state = 'ACCEPTED' if a['accepted'] else 'rejected'
            extra = '' if a['accepted'] else f"  guard_lost={a.get('guard_lost')}"
            print(f"  {state:9s} {a['prompt']!r:20s} -> {a['answer']!r:24s}{extra}")
        print(f"\nnewly_lost   : {out['newly_lost']}")
        print(f"newly_gained : {out['newly_gained']}")
        print(f"weights changed: {out['weights_changed']}")
        if 'after_restart' in out:
            weights_same = out['taught']['weights_sha256'] == out['after_restart']['weights_sha256']
            texts_same = ([r['text'] for r in out['taught']['rows']]
                          == [r['text'] for r in out['after_restart']['rows']])
            print(f"weights identical after cold restart: {weights_same}")
            print(f"texts identical after cold restart  : {texts_same}"
                  + ('' if texts_same else '  (working state drifted, not a loss)'))

    print('\n' + '=' * 68)
    print('12 REQUIRED METRICS')
    print('=' * 68)
    for k, v in metres(out).items():
        if isinstance(v, float):
            print(f'  {k:28s}: {v:.4f}')
        else:
            print(f'  {k:28s}: {v}')

    after = live_sha()
    print(f'\nlive runtime sha before/after : {str(before)[:16]} / {str(after)[:16]}')
    print('  (runtime payload churns because the idle loop keeps writing state;')
    print('   that is why weights_sha256, not this value, is the weight proof.)')

    if args.json_out:
        payload = {'metrics': metres(out), **out,
                   'live_sha_before': before, 'live_sha_after': after,
                   'scope': 'Product acceptance on a discarded copy. No rule, '
                            'topology, signal or guard-threshold change.'}
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                       encoding='utf-8')
        print(f'wrote {args.json_out}')


if __name__ == '__main__':
    main()
