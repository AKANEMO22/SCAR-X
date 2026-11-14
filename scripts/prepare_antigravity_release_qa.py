"""Sanitized release QA handoff for user-requested Antigravity CLI testing.

Only public code/tests and prior public P0 source/audits enter the copy. Never
copy deployment config, user chat, models, private state or raw runtime logs.
"""

from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    job=ROOT/'local/Antigravity-release-QA'/stamp;job.mkdir(parents=True)
    target=Path(tempfile.mkdtemp(prefix='scarx-public-release-QA-')).resolve()
    assert target.parent==Path(tempfile.gettempdir()).resolve() and target.name.startswith('scarx-public-release-QA-')
    tests=['test_hybrid_learning.py','test_hybrid_gateway.py','test_hybrid_modes.py','test_hybrid_cognition.py',
           'test_hybrid_incidents.py','test_hybrid_review.py','test_hybrid_resources.py','test_hybrid_device_budget.py',
           'test_hybrid_maintenance.py','test_hybrid_tool_routing.py','test_hybrid_cuda.py','test_hermes_session_transport.py',
           'test_hybrid_rc_package.py','test_full_query_retrieval_shadow.py','test_verifier_capability_shadow.py',
           'test_asr_profile_resolver.py','test_hermes_resource_roots.py',
           'test_hybrid_frame_snapshot.py','test_video_request_ownership.py','test_hermes_admission_scope.py',
           'test_media_reply_presentation.py','test_hybrid_screen_audio.py','test_hybrid_pipeline_audit.py',
           'test_hybrid_feedback_transaction.py','test_hybrid_web_routing.py','test_hybrid_web_evidence.py']
    files=list((ROOT/'src/scarx').rglob('*.py'))
    files += [ROOT/'src/scarx/hybrid/web.html',ROOT/'src/scarx/hybrid/hermes_protocol_notes.json',
              ROOT/'run_hybrid.py',ROOT/'pyproject.toml']
    files += [ROOT/'tests'/name for name in tests]
    files += [ROOT/'tests'/name for name in ('hybrid_trace.test.cjs','hybrid_composer.test.cjs','hybrid_question_http.test.cjs')]
    files += [p for p in (ROOT/'integrations/hermes').rglob('*') if p.is_file() and p.suffix in ('.py','.js','.json','.yaml','.patch') and p.name!='settings.json' and '__pycache__' not in p.parts]
    files += list((ROOT/'release/hybrid').glob('*'))
    files += [ROOT/'scripts'/name for name in ('package_hybrid_rc.py','verify_hybrid_rc_candidate.py',
        'hybrid_service.py','hybrid_screen_worker.py','run_native_searxng.py','install_hybrid_hermes.py',
        'install_hybrid_provider.py','install_hermes_session_transport.py','install_hermes_observer_envelope.py',
        'test_hybrid_question_submission.py','hybrid_pipeline_audit.py')]
    for run,candidate in [('20261114T101526Z-full-query-retrieval-p0-v1-7a9ee0','candidate-store.py'),
                          ('20261114T105211Z-verifier-capability-p0-v1-8e205a','candidate-engine.py')]:
        folder=ROOT/'runs'/run
        files += [folder/candidate,folder/'independent-data-audit-v1.json']
        files += [p for p in (folder/'source').rglob('*') if p.is_file()]
    files += [ROOT/'docs/release/evidence/ARITHMETIC_TASK_SCOPE_FINAL_SOURCE_20261115.json']
    dll=ROOT/'build/scarx_cpu.dll'
    if dll.is_file():files.append(dll)
    hashes={}
    for path in sorted(set(files)):
        if not path.is_file():continue
        rel=path.relative_to(ROOT)
        assert not any(p in ('local','data','artifacts','.git','__pycache__') for p in rel.parts)
        assert path.stat().st_size<2_000_000
        out=target/rel;out.parent.mkdir(parents=True,exist_ok=True);raw=path.read_bytes();out.write_bytes(raw)
        hashes[rel.as_posix()]=hashlib.sha256(raw).hexdigest()
    (target/'public-source-manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    config={'python':str(ROOT/'local/hybrid-v1/runtime-env/Scripts/python.exe'),
            'node':str(Path(os.environ['LOCALAPPDATA'])/'hermes/tools/node-26.12.1-win32-x64/node.exe'),
            'pytest_files':['tests/'+name for name in tests]}
    (target/'qa-config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    driver='''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;cfg=json.loads((root/'qa-config.json').read_text());manifest=json.loads((root/'public-source-manifest.json').read_text())
def unchanged():
 return all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in manifest.items())
assert unchanged();out=root/'qa-results';out.mkdir(exist_ok=True);env=dict(os.environ,PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1')
commands=[('python',[cfg['python'],'-m','pytest',*cfg['pytest_files'],'-q','--junitxml='+str(out/'python.xml')]),
 ('node',[cfg['node'],'--test','tests/hybrid_trace.test.cjs','tests/hybrid_composer.test.cjs'])]
for case in ('chat','debug','lost-receipt'):
 commands.append(('feedback-'+case,[cfg['python'],'scripts/test_hybrid_question_submission.py','--case',case]))
results=[]
for name,command in commands:
 started=time.monotonic();r=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
 (out/(name+'.stdout.log')).write_text(r.stdout,encoding='utf-8');(out/(name+'.stderr.log')).write_text(r.stderr,encoding='utf-8')
 results.append({'name':name,'exit_code':r.returncode,'seconds':time.monotonic()-started,'stdout_tail':r.stdout[-1200:],'stderr_tail':r.stderr[-800:]})
assert unchanged(),'QA modified immutable public source'
report={'results':results,'public_source_unchanged':True,'live_gateway_or_userdata_modified':False,'LLM_ASR_calls':0,'isolated_ALIF_CPU_CUDA_fixture_updates':True}
(out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
raise SystemExit(0 if all(row['exit_code']==0 for row in results) else 1)
'''
    (target/'run_release_checks.py').write_text(driver,encoding='utf-8')
    prompt=f'''User requested full release QA, not a new research campaign. Work ONLY in this sanitized public-code copy: {target}.
First execute exactly this allowlisted diagnostic command once: & '{config['python']}' '{target/'run_release_checks.py'}'. It runs the whole listed release runtime/CPU/CUDA/persistence/gateway/host-route/resource/ASR-contract/packaging suite, Node UI tests and 3 actual loopback feedback transport scenarios on private fixtures. It starts NO live LLM/ASR services and modifies NO live user state. If permission or sandbox refuses, report precisely, never bypass it.
Then read qa-results/result.json and relevant source/tests. Audit for concrete release bugs, especially greeting control-marker echo, stopped-screen history and cross-session/media focus, long query budgets, tool-loop observation/permission, checkpoint and code rollback. Inspect tests for gaps. You may run ONE short additional regression on a disposable fixture inside this directory if needed. Do not change source/tests/thresholds to make them pass. Write qa-results/antigravity-review.md with reproducible findings, severity, path+line, expected/actual, minimal repair and limits; final response should state tests actually executed and findings. Do not modify code, original checkout, Hermes, model workers, main or R13. No web/CLI data generation, downloads, new model/research, training benchmarks, eight-hour tests or app UI automation. The only CUDA learning is bounded existing parity fixtures, not a research result.
Known accepted facts: SNN advantage over same LLM+RAG and lifelong skill retention remain unproven. Synthetic/engineering fixtures are not model quality PASS. User cancelled the long8h; never claim it passed. Current code is a candidate snapshot including an already-unit-tested fix: omit the exact unused Hermes STEER_CHANNEL_NOTE only for pure social turns; other tasks retain it. Real captured-window test has silent synthetic video, no speaker/mic audio. Models keep original precision and ALIF stays local. The source manifest must stay unchanged. Do not call things implemented unless verified. Finish within7 minutes; no repeated retries if a tool is unavailable.'''
    (job/'prompt.txt').write_text(prompt,encoding='utf-8')
    args=[str(Path(os.environ['LOCALAPPDATA'])/'agy/bin/agy.exe'),'--model','gemini-3.8-flash-high','--effort','high',
          '--mode','auto','--sandbox','--print-timeout','480s','--output-format','stream-json',
          '--log-file',str(job/'cli.log'),'--print',prompt]
    with (job/'events.jsonl').open('wb') as out,(job/'stderr.log').open('wb') as err:
        process=subprocess.Popen(args,cwd=target,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    receipt={'job_directory':str(job),'sanitized_copy':str(target),'public_files':len(hashes),
             'pid':process.pid,'created':psutil.Process(process.pid).create_time(),
             'model':'gemini-3.8-flash-high','effort':'high','private_data_sent':False,
             'scope':'release functional QA; no new research or live mutations','started_epoch':time.time()}
    (job/'job.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (ROOT/'local/current-Antigravity-release-QA.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
