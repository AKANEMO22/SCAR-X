from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from scarx.evaluation.rgb_grounding import different_class_pairing
from scarx.evaluation.runs import new_run,write_json

primary=json.loads((root/'artifacts/rgb-object-grounding-latest.json').read_text(encoding='utf-8'));source=root/primary['run']
run=new_run(root,'rgb-object-grounding-negative-audit',{'protocol':'experiments/rgb-object-grounding-negative-audit.md','source_run':primary['run'],'training_updates':0},data_description='Existing public development prediction vectors; additive correction of mostlysameclass shiftcontrol')
rows=[]
for row in primary['trials']:
    raw=json.loads((source/f"{row['condition']}-{row['seed']}.json").read_text(encoding='utf-8'));labels=np.array(raw['targets']);p=np.array(raw['predictions'])
    old_same=float(np.mean(labels[np.roll(np.arange(len(labels)),1)]==labels));mapping=different_class_pairing(labels)
    wrong=float(np.mean(p[mapping]==labels));item={**row,'original_mismatch_preserved_class_fraction':old_same,
      'corrected_mismatch_preserved_class_fraction':float(np.mean(labels[mapping]==labels)),
      'corrected_mismatched_accuracy':wrong,'corrected_gain_over_mismatched':row['accuracy']-wrong,
      'corrected_informative':row['accuracy']>=.30 and row['gain_over_blank']>=.10 and row['accuracy']-wrong>=.10}
    rows.append(item);write_json(run/f"{row['condition']}-{row['seed']}.json",{'summary':item,'crossclass_pairing':mapping.tolist(),'corrected_mismatched_predictions':p[mapping].tolist()})
summary={c:{k:float(np.mean([r[k] for r in rows if r['condition']==c])) for k in ('accuracy','blank_accuracy','corrected_mismatched_accuracy','corrected_gain_over_mismatched')} for c in ('raw','normalized')}
result={'run':run.relative_to(root).as_posix(),'source_run':primary['run'],'summary':summary,'trials':rows,
 'metric_correction':'Original roll1 matched150/160sameclasses. Crossclassmappingnowchanges160/160; no training/split/thresholdchanges.',
 'image_grounding_gate':False,'normal_photo_qa_proven':False,'deployed':False,'decision':'PIVOT_VISUAL_REPRESENTATION_NO_SCALE'}
write_json(run/'result.json',result);write_json(root/'artifacts/rgb-object-grounding-audit.json',result)
(run/'report.md').write_text('# RGB negative pairing audit\n\n'+json.dumps(result,indent=2)+'\n\nAll classifiers remain below pre-registered30%accuracy; no natural vision or captioning success. Raw metric error retained and corrected additively.\n',encoding='utf-8')
print(json.dumps({'run':str(run),'summary':summary,'image_grounding_gate':False}),flush=True)
