"""Public runtime pins only; no model/venv/private-state bytes in RC artifact."""

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scarx.runtime.asr_profiles import PROFILES


def file_pin(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(2<<20):digest.update(block)
    return {'bytes':path.stat().st_size,'sha256':digest.hexdigest()}


def main():
    directory=ROOT/'release/hybrid';data=ROOT/'local/hybrid-v1'
    integrity=json.loads((data/'native-integrity.json').read_text(encoding='utf-8'))
    installation={}
    for name,entry in integrity['exports'].items():
        installation['local/hybrid-v1/models/native/'+name]={'bytes':entry['bytes'],'sha256':entry['sha256']}
    small=PROFILES['openai/whisper-small']
    installation['local/hybrid-v1/'+small.relative_path]={'bytes':small.native_bytes,'sha256':small.native_sha256}
    for relative in ['local/hybrid-v1/native-integrity.json','local/hybrid-v1/original-reference.json',
                     'local/hybrid-v1/candidates/whisper-small/native-storage-receipt.json']:
        installation[relative]=file_pin(ROOT/relative)
    for folder,executable in [('runtime/llama.cpp-b12360-cuda12.4','llama-server.exe'),
                              ('runtime/whisper.cpp-b5454-cuda12.4/Release','whisper-cli.exe')]:
        native=data/folder
        for path in sorted([native/executable,*native.glob('*.dll')]):
            installation[path.relative_to(ROOT).as_posix()]=file_pin(path)
    home=Path(os.environ['LOCALAPPDATA'])/'hermes'
    needed=['hermes-agent/agent/agent_init.py','hermes-agent/agent/conversation_compression.py',
            'hermes-agent/agent/context_compressor.py','hermes-agent/tools/tool_search.py',
            'hermes-agent/tools/file_operations.py','hermes-agent/tools/file_operations_read_contract.py',
            'desktop-plugins/scarx-hybrid/plugin.js','plugins/scarx-hybrid/__init__.py',
            'plugins/scarx-hybrid/learning_hooks.py','plugins/scarx-hybrid/dashboard/plugin_api.py',
            'plugins/scarx-hybrid/dashboard/manifest.json']
    hermes={name:file_pin(home/name) for name in needed}
    desktop=home/'hermes-agent/apps/desktop/release/win-unpacked/resources/app.asar.unpacked/dist'
    for path in sorted(desktop.rglob('*')):
        if path.is_file() and path.suffix in ('.js','.css','.html'):
            hermes[path.relative_to(home).as_posix()]=file_pin(path)
    system={}
    for command in ('ffmpeg','ffprobe'):
        found=shutil.which(command)
        if not found:raise RuntimeError('Required media tool absent: '+command)
        system[command]=file_pin(Path(found))
        system[command]['version']=subprocess.check_output([found,'-version'],text=True,encoding='utf-8',errors='replace',timeout=10).splitlines()[0]
    assets={'format':'scarx-existing-hybrid-asset-pins-v1','assets_embedded':False,
        'model_identity':{'name':'Qwen/Qwen3.5-0.8B','revision':'2fc06364715b967f1860aea9cf38778875588b17',
                          'ASR':small.identity(),'ALIF':'H128 FP32 local learning'},
        'runtime_releases':{'llama.cpp':'b12360 CUDA12.4','whisper.cpp':'b5454 CUDA12.4',
                            'SearXNG_commit':'219b63b207c98b85471de5abab773e76432f1961'},
        'model_export_pins_origin':'existing tensor-audited native-integrity export pins + pinned ASR registry; actual bytes rechecked on install',
        'installation_files':installation,'hermes_integration_files':hermes,'system_tools':system,
        'runtime_paths_required':['local/hybrid-v1/runtime-env/Scripts/python.exe',
                                  'local/searxng/env/Scripts/python.exe','local/searxng/settings.yml','local/searxng/installation.json']}
    (directory/'ASSETS.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    packages=['numpy','psutil','Pillow','PyYAML','cupy-cuda12x','windows-capture','PyAudioWPatch',
              'opencv-python','cuda-pathfinder','nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12','cuda-toolkit']
    versions={'format':'scarx-existing-runtime-versions-v1','Python':'.'.join(map(str,sys.version_info[:3])),
              'packages':{name:metadata.version(name) for name in packages},
              'reuse_existing_environment_only':True,'venv_embedded':False}
    (directory/'DEPENDENCIES.json').write_text(json.dumps(versions,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'native_asset_pins':len(installation),'Hermes_code_pins':len(hermes),
                      'runtime_package_versions':len(packages),'private_configs_or_state_copied':False}))


if __name__=='__main__':main()
