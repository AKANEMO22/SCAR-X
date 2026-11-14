"""Actual Hermes hook envelope + dispatcher on a disposable owned project.

Three failed reads and one optional readonly project probe. No model, network,
live gateway mutation, fabricated tool result or learning success claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('hook_source',type=Path);args=p.parse_args()
    directory=args.directory.resolve()
    if not directory.is_relative_to(ROOT/'local') or directory.exists():raise ValueError('Expected new owned local directory')
    directory.mkdir(parents=True);home=directory/'home';project=directory/'project'
    home.mkdir();project.mkdir();source=Path(os.environ['LOCALAPPDATA'])/'hermes/hermes-agent'
    sys.path.insert(0,str(source));os.environ.update(HERMES_HOME=str(home),TERMINAL_CWD=str(project));os.chdir(project)
    from hermes_yaml import safe_dump
    (home/'config.yaml').write_text(safe_dump({'plugins':{'enabled':['observer-contract-fixture']},
        'terminal':{'backend':'local'},'tools':{'tool_search':{'enabled':'off'}}}),encoding='utf-8')
    plugin=home/'plugins/observer-contract-fixture';plugin.mkdir(parents=True)
    (plugin/'plugin.yaml').write_text('name: observer-contract-fixture\nversion: 1.0.0\ndescription: Owned contract check\n',encoding='utf-8')
    raw=args.hook_source.resolve().read_bytes();(plugin/'learning_hooks.py').write_bytes(raw)
    output=directory/'observations.jsonl';os.environ['SCARX_CONTRACT_OUTPUT']=str(output)
    (plugin/'__init__.py').write_text('''import importlib.util,json,os
from pathlib import Path

def register(ctx):
 spec=importlib.util.spec_from_file_location('owned_actual_observer',Path(__file__).with_name('learning_hooks.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 def record(value):
  with open(os.environ['SCARX_CONTRACT_OUTPUT'],'a',encoding='utf-8') as f:f.write(json.dumps(value)+'\\n')
 module.register_observer(ctx,record)
''',encoding='utf-8')
    (project/'README.md').write_text('Owned inventory application. Documentation port 9237.\n',encoding='utf-8')
    from hermes_cli.plugins import discover_plugins
    from hermes_cli.lifecycle import invoke_hook
    from model_tools import get_tool_definitions,handle_function_call
    from agent.api_request_hooks import ApiRequestHooksMixin
    discover_plugins(force=True);get_tool_definitions(quiet_mode=True)
    envelope=ApiRequestHooksMixin()._api_request_payload_for_hook({'model':'scarx-hybrid',
        'tools':[{'type':'function','function':{'name':'read_file'}}]})
    assert envelope['method']=='POST' and envelope['body']['tools'][0]['function']['name']=='read_file'
    calls=[]
    for i in range(3):
        invoke_hook('pre_api_request',model='scarx-hybrid',base_url='http://127.0.0.1:8775/v1',
            session_id='owned-contract',turn_id='one-turn',task_id='owned-contract-task',
            request=envelope,user_message='Read the current project README')
        result=handle_function_call('read_file',{'path':str(project/'wrong/README.md')},
            session_id='owned-contract',turn_id='one-turn',task_id='owned-contract-task',tool_call_id='read-'+str(i))
        decoded=json.loads(result);assert decoded.get('not_found') is True
        calls.append({'result':decoded,'call_id':'read-'+str(i)})
    observations=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    assert len(observations)==3
    probes=[o['probe'] for o in observations if o.get('probe')]
    if probes:
        assert len(probes)==1 and probes[0]['read_ok'] and probes[0]['original_path_differs']
        assert probes[0]['source_name']=='README.md' and probes[0]['semantic_resolution_verified'] is False
    report={'source':'actual installed Hermes ApiRequestHooksMixin envelope, plugin discovery and dispatcher',
        'hook_sha256':hashlib.sha256(raw).hexdigest(),'actual_failed_read_calls':3,
        'public_post_tool_observations':len(observations),'readonly_project_probes':len(probes),
        'probe_read_ok':bool(probes and probes[0]['read_ok']),'semantic_task_resolution_verified':False,
        'new_neural_VLM_or_web_calls':0,'live_gateway_or_private_data_modified':False,
        'learning_success_claimed':False}
    (directory/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (directory/'actual-returns.json').write_text(json.dumps(calls,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
