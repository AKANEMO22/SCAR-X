"""Explicit code-only Hybrid RC; never invoke the legacy Generation1 packager."""

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'local','runs','data','artifacts','.git','tests','__pycache__','.venv','venv','cache'}
SUFFIXES={'.py','.js','.json','.yaml','.html'}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def validate_payload_path(name):
    p=PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or p.is_absolute()
            or any(part in ('','..','.') or part in FORBIDDEN or part.startswith('.') for part in name.split('/'))
            or p.suffix not in SUFFIXES):raise ValueError('Forbidden release payload path: '+name)
    if name in ('run_hybrid.py','integrations/hermes/scarx_session_transport.py','integrations/hermes/session_submission.py'):return
    if not name.startswith(('src/scarx/','scripts/','integrations/hermes/hybrid_plugin/')):
        raise ValueError('Payload outside runtime allowlist roots')
    if p.name in ('settings.json','config.json','auth.json','gateway_state.json'):
        raise ValueError('Generated/private config is not source payload')


def zip_members(archive):
    infos=archive.infolist();names=[i.filename for i in infos]
    if len(names)!=len(set(n.casefold() for n in names)):raise ValueError('Duplicate/case-aliased ZIP member')
    for info in infos:
        name=info.filename;p=PurePosixPath(name)
        if ('\\' in name or ':' in name or p.is_absolute() or any(part in ('','.','..') for part in name.split('/')) or '\x00' in name
                or (info.external_attr>>16)&0o170000==0o120000 or info.file_size>2_000_000):
            raise ValueError('Unsafe ZIP member')
    if sum(info.file_size for info in infos)>20_000_000:raise ValueError('Expected code-only archive')
    return infos


def source_sha(root):
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()


def predecessor_manifest(root,spec):
    pin=spec.get('predecessor')
    if pin is None:return None
    path=(root/pin['archive']).resolve()
    if not path.is_relative_to((root/'artifacts/releases').resolve()) or path.suffix!='.zip':
        raise ValueError('Predecessor archive must stay in the release directory')
    if digest(path.read_bytes())!=pin['sha256']:raise ValueError('Pinned predecessor archive checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        infos=zip_members(archive)
        previous=json.loads(archive.read('MANIFEST.json'))
        if (previous.get('format')!='scarx-hybrid-local-rc-v1'
                or previous.get('model_id')!='scarx-hybrid'
                or previous.get('version')!=pin['version']
                or previous.get('source_git_sha')!=pin['source_git_sha']):
            raise ValueError('Pinned predecessor identity mismatch')
        if {i.filename for i in infos}!=set(previous['files'])|{'MANIFEST.json'}:
            raise ValueError('Predecessor members differ from its manifest')
        for name,entry in previous['files'].items():
            raw=archive.read(name)
            if len(raw)!=entry['bytes'] or digest(raw)!=entry['sha256']:
                raise ValueError('Predecessor payload integrity mismatch')
    return previous


def overlay_candidates(root):
    specs=[('src/scarx/hybrid/store.py','runs/20261114T101526Z-full-query-retrieval-p0-v1-7a9ee0/candidate-store.py',
            'runs/20261114T101526Z-full-query-retrieval-p0-v1-7a9ee0/independent-data-audit-v1.json',
            'integrations/hermes/full_query_retrieval_patch.py',
            'SHADOW_SQLITE_QUERY_COVERAGE_AND_RESULTS_DATA_VERIFIED_NOT_LIVE_PROMOTION'),
           ('src/scarx/hybrid/engine.py','runs/20261114T105211Z-verifier-capability-p0-v1-8e205a/candidate-engine.py',
            'runs/20261114T105211Z-verifier-capability-p0-v1-8e205a/independent-data-audit-v1.json',
            'integrations/hermes/verifier_capability_guard_patch.py',
            'SHADOW_CONTROLLER_MASK_DELTA_VERIFIED_NOT_SEMANTIC_OR_NEURAL_PASS')]
    changed={};provenance={}
    for target,candidate,audit,recipe,verdict in specs:
        report=json.loads((root/audit).read_text(encoding='utf-8'))
        if report.get('verdict')!=verdict:raise ValueError('P0 data audit not verified')
        raw=(root/candidate).read_bytes();compile(raw,target,'exec')
        scope={'__name__':'_scarx_release_exact_patch'}
        recipe_raw=(root/recipe).read_bytes()
        exec(compile(recipe_raw,recipe,'exec'),scope)
        baseline=(root/Path(audit).parent/'source'/target).read_bytes()
        if scope['patch_bytes'](baseline)!=raw:
            raise ValueError('Candidate differs from the qualified exact-source P0 recipe')
        legacy = raw
        followup_recipe = root/'integrations/hermes/atomic_feedback_patch.py'
        followup_audit = root/'docs/release/evidence/ARITHMETIC_TASK_SCOPE_SOURCE_20261115.json'
        followup = None
        if followup_recipe.exists() or followup_audit.exists():
            if not followup_recipe.is_file() or not followup_audit.is_file():
                raise ValueError('Incomplete atomic feedback qualification')
            qualification=json.loads(followup_audit.read_text(encoding='utf-8'))
            recipe_bytes=followup_recipe.read_bytes()
            if (qualification['verdict']!='FEEDBACK_AND_WEB_CONTEXT_SOURCE_REGRESSIONS_VERIFIED_NOT_NEURAL_QUALITY_PASS'
                    or qualification['recipe_sha256']!=digest(recipe_bytes)
                    or qualification['before_sha256'][target]!=digest(legacy)):
                raise ValueError('Atomic feedback qualification/recipe mismatch')
            namespace={'__name__':'_atomic_feedback_release_recipe'}
            exec(compile(recipe_bytes,str(followup_recipe),'exec'),namespace)
            raw=namespace['patch_bytes'](target,legacy)
            if digest(raw)!=qualification['after_sha256'][target]:
                raise ValueError('Atomic feedback result differs from tested source')
            followup={'recipe':'integrations/hermes/atomic_feedback_patch.py',
                      'audit':'docs/release/evidence/ARITHMETIC_TASK_SCOPE_SOURCE_20261115.json',
                      'recipe_sha256':digest(recipe_bytes),'audit_sha256':digest(followup_audit.read_bytes()),
                      'legacy_qualified_candidate_sha256':digest(legacy)}
        if (root/target).read_bytes() not in (baseline,legacy,raw):
            raise ValueError('Active source differs from both qualified before/after states')
        changed[target]=raw
        provenance[target]={'candidate_sha256':digest(raw),'audit_sha256':digest((root/audit).read_bytes()),
                            'recipe_sha256':digest(recipe_raw),
                            'audit_reference':audit,'qualified_scope':'isolated P0; live release acceptance pending'}
        if followup:provenance[target]['post_RC8_atomic_feedback']=followup
    return changed,provenance


def wrapper(action):
    script=r'''param([switch]$Check)
$ErrorActionPreference = 'Stop'
$settingsPath = Join-Path $env:LOCALAPPDATA 'hermes\plugins\scarx-hybrid\dashboard\settings.json'
if (-not (Test-Path -LiteralPath $settingsPath)) { throw 'Existing SCAR-X Hybrid registration is required.' }
$settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
$python = [string]$settings.python
if (-not (Test-Path -LiteralPath $python)) { throw 'Registered runtime Python is missing.' }
'''
    if action in ('start','stop'):
        script+="$targetRoot = Split-Path -Parent ([string]$settings.entrypoint)\n"
        script+="& $python (Join-Path $targetRoot 'scripts\\hybrid_service.py') "+action+"\n"
    else:
        flag={'install':'','rollback':' --rollback','verify':' --verify-only'}[action]
        script+="if ($Check) { & $python (Join-Path $PSScriptRoot 'manage.py') --package $PSScriptRoot --check } else { & $python (Join-Path $PSScriptRoot 'manage.py') --package $PSScriptRoot"+flag+" }\n"
    script+="if ($LASTEXITCODE -ne 0) { throw 'SCAR-X command failed. Keep the console details and backup; do not delete user data.' }\n"
    return script.encode('utf-8')


def build(root,output_dir):
    spec=json.loads((root/'release/hybrid/payload-files.json').read_text(encoding='utf-8'))
    files=spec['files']
    if len(files)!=len(set(n.casefold() for n in files)):raise ValueError('Duplicate allowlist path')
    for name in files:validate_payload_path(name)
    if not {'scripts/hybrid_screen_worker.py','scripts/run_native_searxng.py','run_hybrid.py'}<=set(files):
        raise ValueError('Required Hybrid helper missing')
    overlays,provenance=overlay_candidates(root)
    predecessor=predecessor_manifest(root,spec)
    before={};blobs={}
    for name in files:
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Source link/path escape')
        original=path.read_bytes()
        before[name]=(predecessor['files'].get('payload/'+name,{}).get('sha256')
                      if predecessor else digest(original))
        blobs['payload/'+name]=overlays.get(name,original)
    top={'manage.py':root/'release/hybrid/manage.py','README.md':root/'release/hybrid/README.md',
         'HERMES_MEDIA_ADMISSION.patch':root/'release/hybrid/HERMES_MEDIA_ADMISSION.patch',
         'KNOWN_LIMITATIONS.md':root/'release/hybrid/KNOWN_LIMITATIONS.md',
         'ASSETS.json':root/'release/hybrid/ASSETS.json','DEPENDENCIES.json':root/'release/hybrid/DEPENDENCIES.json'}
    for name,path in top.items():blobs[name]=path.read_bytes()
    for name,action in [('Install-Hermes.ps1','install'),('Start-SCARX.ps1','start'),('Stop-SCARX.ps1','stop'),('Rollback-SCARX.ps1','rollback'),('Verify-Package.ps1','verify')]:
        blobs[name]=wrapper(action)
    commit=source_sha(root)
    manifest={'format':'scarx-hybrid-local-rc-v1','version':spec['version'],'source_git_sha':commit,
        'model_id':'scarx-hybrid','deployment':spec['deployment'],'release_state':'prepared_candidate_not_live_qualified',
        'files':{name:{'sha256':digest(raw),'bytes':len(raw)} for name,raw in sorted(blobs.items())},
        'target_before_sha256':before,'reviewed_overlays':provenance,
        'predecessor':spec.get('predecessor'),
        'private_state_models_venv_cache_datasets_or_research_runs_included':False,
        'running_test_must_finish_before_install':True}
    blobs['MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    output_dir.mkdir(parents=True,exist_ok=True)
    target=output_dir/f"SCAR-X-Hybrid-{spec['version']}+{commit[:12]}.zip"
    if target.exists():raise FileExistsError('Do not overwrite a release artifact')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_STORED) as archive:
        for name,raw in sorted(blobs.items()):
            info=zipfile.ZipInfo(name,(2026,11,14,0,0,0));info.compress_type=zipfile.ZIP_STORED
            info.external_attr=0o100644<<16;archive.writestr(info,raw)
    with zipfile.ZipFile(target) as archive:zip_members(archive)
    hashvalue=digest(target.read_bytes())
    target.with_suffix('.zip.sha256').write_text(hashvalue+'  '+target.name+'\n',encoding='ascii')
    return target,manifest,hashvalue


def verify_extracted_imports(package,python):
    code='''import sys,json,pathlib
src=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(src))
import scarx.hybrid.gateway,scarx.runtime.maintenance
loaded={n:str(m.__file__) for n,m in sys.modules.items() if n=='scarx' or n.startswith('scarx.')}
assert all(pathlib.Path(p).resolve().is_relative_to(src) for p in loaded.values())
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
print(json.dumps({'all_SCARX_imports_from_payload':True,'torch_imported':False,'modules':sorted(loaded)}))
'''
    return subprocess.run([str(python),'-I','-B','-c',code,str(package/'payload/src')],capture_output=True,text=True,encoding='utf-8',timeout=30)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/releases');args=p.parse_args()
    target,manifest,sha256=build(ROOT,args.output)
    with tempfile.TemporaryDirectory(prefix='scarx-hybrid-rc-verify-') as temporary:
        extracted=Path(temporary)
        if extracted.resolve().parent!=Path(tempfile.gettempdir()).resolve() or not extracted.name.startswith('scarx-hybrid-rc-verify-'):
            raise ValueError('Unexpected owned verification directory')
        with zipfile.ZipFile(target) as archive:
            for info in zip_members(archive):
                destination=extracted/info.filename;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(archive.read(info))
        # Verify with the trusted source helper, not unverified extracted code.
        spec=importlib.util.spec_from_file_location('_hybrid_rc_verify_manager',ROOT/'release/hybrid/manage.py')
        manager=importlib.util.module_from_spec(spec)
        old_flag=sys.dont_write_bytecode;sys.dont_write_bytecode=True
        try:spec.loader.exec_module(manager)
        finally:sys.dont_write_bytecode=old_flag
        manager.verify_package(extracted)
        result=verify_extracted_imports(extracted,ROOT/'local/hybrid-v1/runtime-env/Scripts/python.exe')
        if result.returncode:raise RuntimeError('Extracted import closure failed: '+result.stderr[-1500:])
        verification=json.loads(result.stdout)
    receipt={'artifact':str(target),'sha256':sha256,'bytes':target.stat().st_size,'payload_files':len(manifest['target_before_sha256']),
        'source_git_sha':manifest['source_git_sha'],'verification':verification,
        'status':'CODE_PACKAGE_VERIFIED_NOT_INSTALLED_OR_LIVE_ACCEPTED'}
    target.with_suffix('.verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
