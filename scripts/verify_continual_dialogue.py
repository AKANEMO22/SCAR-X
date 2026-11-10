"""One persistent experience stream, human-answer fixture and idle local learning.

No epochs/task resets or deployment of the generated corpus. Test-only isolated
session. Arithmetic labels recomputed; authored rubric data stays untrained.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,uuid,threading
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.app.generation import Generation
from scarx.runtime.limits import apply_memory_limit
from scarx.learning.verifiers import parse_integer_expression,integer_answer
from scarx.evaluation.runs import new_run,write_json
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
cap=apply_memory_limit()
run=new_run(root,'continuous-dialogue-acceptance',{'protocol':'experiments/dialogue-online-acceptance.md','single_persistent_learner':True,'epochs':0,'private_session_used':False},data_description='Explicitly authored1+1 human-answer fixture plus first8 machine-verified Antigravity train records; no rubric labels learned')
app=Generation(root/'local'/('continuous-qa-'+uuid.uuid4().hex[:8]),background=False)
try:
    original=app.model.core.fingerprint();model_id=id(app.model)
    response=app.chat('1+1 bằng mấy?',session_id='qa-human')
    q=response['learning_question'];before_updates=app.model.core.updates
    feedback=app.answer_question({'session_id':'qa-human','question_id':q['id'],'answer':'2'})
    for _ in range(140):app.idle_once()
    human_result=app.model.generate('1+1 bằng mấy?')['text']
    print(json.dumps({'phase':'human_feedback','accepted':app.dialogue.accepted_updates,'rejected':app.dialogue.rejected_updates,'response':human_result},ensure_ascii=False),flush=True)
    pointer=json.loads((root/'artifacts/antigravity-dataset-latest.json').read_text(encoding='utf-8'))
    dataset=root/pointer['candidate_directory']
    rows=[json.loads(l) for l in (dataset/'experience-stream.jsonl').read_text(encoding='utf-8').splitlines()]
    records=[r for r in rows if r['validation']['automatic_learning_allowed']][:8]
    experiences=[]
    for row in records:
        expression=parse_integer_expression(row['input']);assert expression is not None and str(integer_answer(expression))==row['target']
        before=app.model.generate(row['input'])['text']
        app.dialogue.enqueue('qa-data',row['input'],row['target'],'Antigravity arithmetic; parsed and recomputed: '+row['id'])
        for _ in range(2):app._online_microstep('verified_experience_stream')
        experiences.append({'id':row['id'],'before':before,'target':row['target'],'after':app.model.generate(row['input'])['text']})
    for _ in range(80):app.idle_once()
    saved=app.export();clone=Generation(root/'local'/('continuous-resume-'+uuid.uuid4().hex[:8]),background=False)
    try:
        clone._unpack(saved)
        assert app._pack()==clone._pack()
        for _ in range(3):app.idle_once();clone.idle_once()
        for key,array in app.model.core.arrays().items():np.testing.assert_array_equal(array,clone.model.core.arrays()[key])
        assert app.model.ticks==clone.model.ticks and app.rng.bit_generator.state==clone.rng.bit_generator.state
        assert app.dialogue.accepted_updates==clone.dialogue.accepted_updates and app.dialogue.rejected_updates==clone.dialogue.rejected_updates
        # New DB checkpoints/question records intentionally receive new unique
        # operation IDs. They are not neural state or algorithmic randomness.
    finally:clone.close()
    result={'human_question':q['question'],'human_answer':'2','arithmetic_answer_checked':feedback['independently_verified'],
      'human_example_after_idle':human_result,'single_learner_object_preserved':id(app.model)==model_id,
      'weights_changed_from_explicit_labels':app.model.core.fingerprint()!=original,
      'updates':app.model.core.updates-before_updates,'accepted_microsteps':app.dialogue.accepted_updates,'rejected_microsteps':app.dialogue.rejected_updates,
      'idle_reviews':app.dialogue.review_count,'lessons':len(app.dialogue.lessons),'pending_questions':sum(q['status']=='open' for q in app.dialogue.questions),
      'generated_data_stream_examples':experiences,'epochs':0,'task_id_given_to_neural_core':False,
      'snapshot_exact_restore':True,'neural_rng_learning_continuation_equal':True,'new_operation_ids_expected_unique':True,
      'ram_bytes':app.process.memory_info().rss,'resources':app.monitor.report(),'cap':cap,
      'general_arithmetic_or_retention_success':False,'production_weights_replaced':False,'rubric_labels_trained':False}
    write_json(run/'result.json',result);write_json(root/'artifacts/continuous-dialogue-acceptance.json',{'run':run.relative_to(root).as_posix(),**result})
    (run/'report.md').write_text('# Continuous dialogue engineering acceptance\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nOne learner throughout; labeled updates and idle review, no fullnetwork retraining/task resets. Small developmentguard may reject useful updates and cannot establish global retention. Arithmetic sample recall is not general arithmetic. Generated rubric labels not trained.\n',encoding='utf-8')
    print('RESULT '+str(run),flush=True)
finally:app.close()
