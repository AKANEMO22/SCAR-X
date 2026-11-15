"""Generate the approved v2 runner from the immutable v1 experiment source.

No CUDA/model/process execution. Verify the complete per-seed loop is unchanged.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/20261115T054242Z-neural-outcome-bridge-p1a-v1-f39bc6'


def generate():
    manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    original=(BASE/'source/scripts/run_hybrid_neural_outcome_bridge_p1a.py').read_bytes()
    assert hashlib.sha256(original).hexdigest()==manifest['source_sha256']['scripts/run_hybrid_neural_outcome_bridge_p1a.py']
    text=original.decode('utf-8').replace('\r\n','\n')
    def replace(before,after):
        nonlocal text
        if text.count(before)!=1:raise ValueError('V2 patch anchor is not unique: '+before[:80])
        text=text.replace(before,after,1)
    replace('import psutil\n','import psutil\nimport platform\n')
    replace('from scarx.evaluation.recovery_policy import RecoveryPolicy\n',
        'from scarx.evaluation.recovery_policy import RecoveryPolicy\nfrom scarx.evaluation.bridge_effect_guard import StickyEffectGuard\nfrom scarx.evaluation.bridge_environment_metrics import sample_physical_gpu\n')
    for a,b in [('-neural-outcome-bridge-p1a-v1-','-neural-outcome-bridge-p1a-v2-'),
                ('neural-outcome-bridge-p1a-v1-execution.json','neural-outcome-bridge-p1a-v2-execution.json'),
                ('current-neural-outcome-bridge-p1a.txt','current-neural-outcome-bridge-p1a-v2.txt'),
                ('neural-outcome-bridge-p1a-v1-budget.jsonl','neural-outcome-bridge-p1a-v2-budget.jsonl'),
                ("names=['scripts/run_hybrid_neural_outcome_bridge_p1a.py'","names=['scripts/run_hybrid_neural_outcome_bridge_p1a_v2.py'"),
                ("'experiments/hybrid-neural-action-outcome-bridge-p1a-v1.md'","'experiments/hybrid-neural-bridge-guard-confirmation-v2.md'")]:replace(a,b)
    replace("'src/scarx/hybrid/learner.py','src/scarx/hybrid/cuda_alif.py','src/scarx/neural/core.py']",
        "'src/scarx/hybrid/learner.py','src/scarx/hybrid/cuda_alif.py','src/scarx/neural/core.py',\n        'src/scarx/evaluation/bridge_effect_guard.py','src/scarx/evaluation/bridge_environment_metrics.py',\n        'scripts/prepare_neural_bridge_confirmation_v2.py','tests/test_bridge_effect_guard.py']")
    replace("    live_before=P0.live_learned_state()\n",'''    live_before=P0.live_learned_state()
    system=Path(os.environ['WINDIR'])/'System32'
    external={'platform':Path(platform.__file__).resolve(),'cmd':(system/'cmd.exe').resolve(),
              'gpu':(system/'nvidia-smi.exe').resolve(),
              'monitor':(ROOT/'src/scarx/evaluation/bridge_environment_metrics.py').resolve()}
    pins={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in external.items()}
    json_write(run/'environment-pins.json',pins)
    prior=ROOT/'runs/20261115T054242Z-neural-outcome-bridge-p1a-v1-f39bc6'
    previous={'run':prior.name,'forwards':18,'dispatches':10,
        'result_sha256':hashlib.sha256((prior/'result.json').read_bytes()).hexdigest(),
        'ledger_sha256':hashlib.sha256((prior/'attempt-ledger.jsonl').read_bytes()).hexdigest()}
''')
    replace("'source_sha256':hashes,'started_epoch':started,'seeds':list(SEEDS),'max_forwards':24,",
        "'confirmation_of':previous,'environment_pins':pins,\n        'source_sha256':hashes,'started_epoch':started,'seeds':list(SEEDS),'max_forwards':24,")
    replace("'live_before':live_before,'approval':'User answer duyệt on15/11/2026, isolated P1A only',",
        "'live_before':live_before,'approval':'User approved confirmation v2 on15/11/2026; additional quota, no reset of v1',")
    before='''    def guard(event,args):
        if event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.posix_spawn'):
            blocked.append(event);raise PermissionError('No network/process/model-service calls in P1A')
        if event=='import' and str(args[0]).split('.')[0]=='torch':
            blocked.append('torch_import');raise PermissionError('No autograd control in P1A')
    sys.addaudithook(guard)
'''
    after='''    effect_guard=StickyEffectGuard(
        pins['platform']['path'],pins['platform']['sha256'],pins['cmd']['path'],pins['cmd']['sha256'],
        pins['monitor']['path'],pins['monitor']['sha256'],pins['gpu']['path'],pins['gpu']['sha256'])
    blocked=effect_guard.violations
    def guard(event,args):
        if event=='import' and str(args[0]).split('.')[0]=='torch':
            blocked.append({'event':'torch_import'});raise PermissionError('No autograd control in P1A')
        effect_guard.hook(event,args)
    sys.addaudithook(guard)
'''
    replace(before,after)
    replace('''    def resources():
        info=psutil.Process().memory_info()
''','''    def resources():
        effect_guard.check() # Sticky even if an SDK swallowed its PermissionError.
        info=psutil.Process().memory_info()
''')
    replace('''        free,total=(map(int,cp.cuda.runtime.memGetInfo()) if cp is not None else (None,None))
        memory.append({'epoch':time.time(),'peak_RSS_bytes':rss,'peak_private_commit_bytes':private,
                       'physical_cuda_free_bytes':free,'physical_cuda_total_bytes':total})
''','''        available,available_total=(map(int,cp.cuda.runtime.memGetInfo()) if cp is not None else (None,None))
        physical=sample_physical_gpu(pins['gpu']['path'],pins['gpu']['sha256'])
        effect_guard.check()
        free,total=physical['free_bytes'],physical['total_bytes']
        memory.append({'epoch':time.time(),'peak_RSS_bytes':rss,'peak_private_commit_bytes':private,
                       'physical_cuda_free_bytes':free,'physical_cuda_total_bytes':total,
                       'physical_gpu_source':physical['source'],
                       'cuda_runtime_available_bytes':available,'cuda_runtime_total_bytes':available_total})
''')
    replace("    preflight=ROOT/'local/neural-bridge-preflight-final-20261115.xml'", "    preflight=ROOT/'local/neural-bridge-v2-preflight-20261115.xml'")
    replace("        cp=cupy_backend();cp.cuda.Device(0).use();resources()\n",'''        cp=cupy_backend();effect_guard.check()
        cp.cuda.Device(0).use();effect_guard.check();resources()
        if budget.records():raise ValueError('Fresh v2 ledger was already consumed before E0')
        json_write(run/'environment-preflight.json',{'passed':True,'forward_attempts':0,'dispatch_attempts':0,
            'before_ALIF_initialization':True,'dedicated_gpu_sample':memory[-1],
            'allowed_queries':effect_guard.allowed_queries,'declined_sdk_probes':effect_guard.declined_sdk_probes})
        report['environment_preflight_passed']=True
        P0.publish(run,'P1A-v2 environment gate passed','Pinned SDK metadata handling and dedicated GPU query passed before ALIF initialization.')
''')
    replace("        json_write(run/'resources.json',memory)\n",'''        json_write(run/'resources.json',memory)
        json_write(run/'guard-events.json',{'violations':effect_guard.violations,
            'allowed_queries':effect_guard.allowed_queries,'declined_sdk_probes':effect_guard.declined_sdk_probes,
            'gpu_queries':effect_guard.gpu_queries,'ver_queries_declined':effect_guard.ver_queries})
''')
    replace("report.update(elapsed_seconds=time.time()-started,blocked_attempts=blocked,",
        "report.update(elapsed_seconds=time.time()-started,blocked_attempts=blocked,\n            environment_preflight_passed=report.get('environment_preflight_passed',False),")
    replace("min([r['physical_cuda_free_bytes'] for r in memory if r['physical_cuda_free_bytes'] is not None]+[2**63-1])",
        "min([r['physical_cuda_free_bytes'] for r in memory if r['physical_cuda_free_bytes'] is not None],default=None)")
    compile(text,'approved-p1a-v2-candidate.py','exec')
    def loop(source):
        tree=ast.parse(source)
        return next(n for n in ast.walk(tree) if isinstance(n,ast.For)
                    and isinstance(n.target,ast.Name) and n.target.id=='seed'
                    and isinstance(n.iter,ast.Name) and n.iter.id=='SEEDS')
    assert ast.dump(loop(original),include_attributes=False)==ast.dump(loop(text),include_attributes=False)
    target=ROOT/'scripts/run_hybrid_neural_outcome_bridge_p1a_v2.py'
    if target.exists():
        if target.read_bytes()!=text.encode('utf-8'):raise ValueError('Existing candidate differs; refusing overwrite')
    else:
        with target.open('xb') as f:f.write(text.encode('utf-8'))
    print(json.dumps({'generated':str(target),'per_seed_loop_AST_identical_to_v1':True,
        'v1_source_sha256':hashlib.sha256(original).hexdigest(),'v2_source_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'model_or_tool_calls':0}))


if __name__=='__main__':generate()
