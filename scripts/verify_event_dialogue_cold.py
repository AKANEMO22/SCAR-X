"""Additive cold-process check; original experiment receipts stay unchanged."""
from pathlib import Path
import sys,json,subprocess,os,argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.event_dialogue import EventDialogue
from scarx.evaluation.runs import new_run,write_json
from scarx.runtime.limits import apply_memory_limit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path);parser.add_argument('--data',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    apply_memory_limit()
    if args.checkpoint:
        examples=[json.loads(l) for l in args.data.read_text(encoding='utf-8').splitlines()];examples=[r for r in examples if r['split']=='development']
        model=EventDialogue.loads(args.checkpoint.read_bytes());write_json(args.out,model.score(examples));return
    source=json.loads((root/'artifacts/event-dialogue-latest.json').read_text(encoding='utf-8'));data=root/source['data']/'experiences.jsonl'
    run=new_run(root,'event-dialogue-cold-audit',{'source_run':source['run'],'new_training':False},data_description='Same public development inputs/checkpoints, independent new Python process per checkpoint')
    outputs=[]
    for row in source['trials']:
        output=run/f"{row['condition']}-{row['seed']}.json"
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(root/row['checkpoint']),'--data',str(data),'--out',str(output)],check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        actual=json.loads(output.read_text(encoding='utf-8'));expected=row['stages'][-1]['development']
        assert actual==expected,(row['condition'],row['seed'])
        outputs.append({'condition':row['condition'],'seed':row['seed'],'cold_process_equal':True})
    result={'run':run.relative_to(root).as_posix(),'source_run':source['run'],'checks':outputs,'all_exact':True,'new_training':False,'normal_chat_or_vision_proven':False}
    write_json(run/'result.json',result);write_json(root/'artifacts/event-dialogue-cold-check.json',result)
    (run/'report.md').write_text('# Event dialogue cold-process audit\n\nEvery final development response, NLL and counter matches a fresh Python process. This is persistence evidence, not language quality.\n',encoding='utf-8')
    print(json.dumps({'run':str(run),'cold_processes':len(outputs),'all_exact':True}),flush=True)


if __name__=='__main__':main()
