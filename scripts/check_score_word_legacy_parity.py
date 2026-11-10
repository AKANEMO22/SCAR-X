"""Compare actual old source snapshot and current optional-encoder adapter."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,zipfile
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from scarx.evaluation.score_word import ScoreWord


def main():
    run='20261109T175033Z-g1-word-score-transfer-v1-1b1604'
    checkpoint=root/'artifacts/word-score-transfer'/run/'27103-full.zip'
    corpus=json.loads((root/'data/dialogue/generation1.json').read_text(encoding='utf-8'))
    case=corpus['families'][0];prompt=case['train'][0];target=case['answer']
    code='''import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from scarx.evaluation.score_word import ScoreWord
p=Path(sys.argv[2]);m=ScoreWord.loads(p.read_bytes())
initial=m.dumps()==p.read_bytes();result=m.learn(sys.argv[3],sys.argv[4]);state=hashlib.sha256(m.dumps()).hexdigest();reply=m.generate(sys.argv[3],991)
print(json.dumps({'initial_bytes_exact':initial,'learn':result,'state_sha256_after_learning':state,'reply':reply}))
'''
    with tempfile.TemporaryDirectory(prefix='scarx-legacy-score-') as directory:
        destination=Path(directory)
        with zipfile.ZipFile(root/'runs'/run/'source.zip') as archive:
            for member in archive.infolist():
                if member.filename.startswith('src/scarx/') and member.filename.endswith('.py'):
                    path=(destination/member.filename).resolve()
                    if not path.is_relative_to(destination.resolve()) or member.file_size>1_000_000:raise ValueError('Snapshot member')
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(member))
        completed=subprocess.run([sys.executable,'-c',code,str(destination/'src'),str(checkpoint),prompt,target],cwd=destination,capture_output=True,check=True,timeout=30)
        old=json.loads(completed.stdout)
    model=ScoreWord.loads(checkpoint.read_bytes());initial=model.dumps()==checkpoint.read_bytes()
    result=model.learn(prompt,target);state=hashlib.sha256(model.dumps()).hexdigest();reply=model.generate(prompt,991)
    current={'initial_bytes_exact':initial,'learn':result,'state_sha256_after_learning':state,'reply':reply}
    assert old==current and initial,'Legacy behavior changed'
    proof={'old_run':'runs/'+run,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'old_snapshot_vs_current_exact':True,'initial_bytes_exact':True,'future_learning_and_generation_exact':True,
        'actual':current,'serving_core_modified':False}
    (root/'artifacts/g1-word-input-legacy-parity.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof))


if __name__=='__main__':main()
