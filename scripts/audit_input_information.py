"""Measure information preserved before the SNN; no live/private data."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,io,base64,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
import numpy as np
from PIL import Image
from scarx.neural.unified import UnifiedCore
from scarx.app.pilots import PilotLab
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit
from scarx.runtime.resources import ResourceMonitor


def pairs(seed,count):
    rng=np.random.default_rng(seed);out=[]
    for i in range(count):
        a,b=map(int,rng.choice(np.arange(1,1000),2,replace=False))
        if i%2:
            prefix='Bắt đầu. ';clauses=[f'Mã mới là {a}. ',f'Mã mới là {b}. '];suffix='Kết thúc. Mã mới nhất là gì?'
        else:
            prefix='Start. ';clauses=[f'The new code is {a}. ',f'The new code is {b}. '];suffix='End. What is the latest code?'
        out.append({'id':f'{seed}-{i}','language':'vi' if i%2 else 'en',
          'left':prefix+''.join(clauses)+suffix,'right':prefix+''.join(reversed(clauses))+suffix,
          'left_target':str(b),'right_target':str(a)})
    return out


def color_fixture(rgb):
    mask=np.zeros((8,8),bool);mask[1:7,2]=True;mask[1:7,5]=True;mask[1,2:6]=True;mask[6,2:6]=True
    pixels=np.zeros((8,8,3),np.uint8);pixels[mask]=rgb
    image=Image.fromarray(pixels).resize((64,64),Image.Resampling.NEAREST)
    stream=io.BytesIO();image.save(stream,format='PNG');data=stream.getvalue()
    return 'data:image/png;base64,'+base64.b64encode(data).decode(),data


def main():
    cap=apply_memory_limit();run=new_run(root,'p1-input-information-audit-v1',{'protocol':'experiments/p1-input-information-audit-v1.md','data_seeds':[9107,19107],'checkpoint_seeds':[17,29,43],'updates':0},data_description='Authored last-write temporal counterexamples and synthetic same-luminance color glyphs; no personal input')
    released=json.loads((root/'artifacts/generation1-release.json').read_text(encoding='utf-8'));reference=UnifiedCore.loads((root/released['checkpoint']).read_bytes())
    second=root/'artifacts/generation1/20261109T000625Z-generation1-joint-bd6c92/seed-29.checkpoint.zip'
    models=[('public17',reference),('public29',UnifiedCore.loads(second.read_bytes())),('fresh43',UnifiedCore(reference.vocabulary,43))]
    started=time.monotonic();rows=[]
    with ResourceMonitor(interval=.1) as monitor:
        for split,seed in (('development',9107),('confirmation',19107)):
            for item in pairs(seed,64):
                left=reference.context(item['left']);right=reference.context(item['right'])
                row={**item,'split':split,'exact_context_equal':bool(np.array_equal(left,right)),'context_l2_difference':float(np.linalg.norm(left-right))}
                if len(rows)<8 or split=='confirmation' and int(item['id'].split('-')[-1])<8:
                    predictions={}
                    for name,m in models:
                        outputs=[]
                        for field in ('left','right'):
                            clone=UnifiedCore.loads(m.dumps());clone.core.reset_state();clone.last_active[:]=clone.ticks
                            outputs.append(clone.generate(item[field],max_words=4)['text'])
                        predictions[name]={'left':outputs[0],'right':outputs[1],'same':outputs[0]==outputs[1]}
                    row['predictions']=predictions
                rows.append(row);monitor.check()
                if time.monotonic()-started>60:raise TimeoutError('Registered60s audit budget')
        red,red_png=color_fixture((255,40,40));green,green_png=color_fixture((0,178,0))
        a,sha_a=PilotLab.pixels(red);b,sha_b=PilotLab.pixels(green)
        (run/'red.png').write_bytes(red_png);(run/'green.png').write_bytes(green_png)
        vision={'different_rgb_png':red_png!=green_png,'different_input_hash':sha_a!=sha_b,
          'pixel_arrays_identical':bool(np.array_equal(a,b)),'pixel_max_difference':float(np.max(np.abs(a-b))),
          'current_output_classes':int(reference.core.config.outputs-len(reference.vocabulary)),
          'red_prediction':reference.image(a)['digit'],'green_prediction':UnifiedCore.loads((root/released['checkpoint']).read_bytes()).image(b)['digit'],
          'scope':'digit-supported synthetic inputs with distinguishable colors, not normal-photo recognition'}
        write_json(run/'text-pairs.json',rows)
        result={'run':run.relative_to(root).as_posix(),'protocol':'experiments/p1-input-information-audit-v1.md',
          'text_pairs':len(rows),'exact_context_collisions':sum(r['exact_context_equal'] for r in rows),
          'max_context_difference':max(r['context_l2_difference'] for r in rows),
          'same_state_output_equal':all(v['same'] for r in rows for v in r.get('predictions',{}).values()),
          'vision':vision,'resources':monitor.report(),'cap':cap,'training_updates':0,'deployed':False,
          'interpretation':'If counterexamples collide before the core, fix input information preservation before scaling/reward tuning. Does not by itself establish learned representation.'}
        write_json(run/'result.json',result);write_json(root/'artifacts/p1-input-information-audit.json',result)
        (run/'report.md').write_text('# P1 input information audit\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nRaw text pairs and synthetic color PNGs are stored beside this report. Same-state information bounds do not assert global model accuracy. No live/private model changed.\n',encoding='utf-8')
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
