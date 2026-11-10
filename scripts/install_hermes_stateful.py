"""Reversible, surgical Hermes adapter; never modify SCAR-X neural code.

User explicitly requested adapting Hermes to SCAR-X. Snapshot every original
target, write a reviewable diff, refuse changed anchors/concurrent edits.
"""
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import tempfile

root=Path(__file__).resolve().parents[1]
hermes=Path(os.environ['LOCALAPPDATA'])/'hermes/hermes-agent'
patches={
 'tui_gateway/compute_host_bridge.py':[
  ('if kind in {"clarify.request", "clarify.expire"}:',
   'if kind in {"clarify.request", "clarify.expire", "clarify.resolved"}:')],
 'tui_gateway/agent_callbacks.py':[
  ('def _agent_cbs(sid: str) -> dict:\n',
   '''def _agent_cbs(sid: str) -> dict:
    def _question_callback(q, c, multi_select=False, questions=None, standalone=False):
        if standalone:
            from tui_gateway.controller_questions import standalone_clarify
            return standalone_clarify(sid, q, c, _block, _emit, _clarify_timeout_seconds())
        return _clarify_block(sid, q, c, multi_select=multi_select, questions=questions)
'''),
  ('        "clarify_callback": lambda q, c, multi_select=False, questions=None: (\n            _clarify_block(sid, q, c, multi_select=multi_select, questions=questions)),\n',
   '        "clarify_callback": _question_callback,\n')],
 'apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/input-requests.ts':[
  ("  if (event.type === 'clarify.expire') {\n",
   '''  if (event.type === 'clarify.resolved') {
    if (!sessionId) return true
    const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
    const request = $clarifyRequests.get()[sessionId]
    if (!requestId || (request && request.requestId !== requestId)) return true
    // clarify.respond may acknowledge and clear the store before this event.
    // Retain the correlated metadata in the terminal event for that ordering.
    const question = request?.question ?? (payload?.standalone === true && typeof payload?.question === 'string' ? payload.question : '')
    if (!question) return true
    const answer = typeof payload?.user_response === 'string' ? payload.user_response : ''
    upsertToolCall(sessionId, {
      tool_id: requestId, name: 'clarify',
      args: { question, choices: request?.choices ?? payload?.choices ?? [] },
      result: { question, user_response: answer, source: 'standalone controller' }
    }, 'complete', event.type, occurredAt)
    clearClarifyRequest(requestId, sessionId)
    updateSessionState(sessionId, state => ({
      ...state, needsInput: false, awaitingResponse: false,
      streamId: state.busy ? state.streamId : null
    }))
    return true
  }

  if (event.type === 'clarify.expire') {
''')],
 'hermes_cli/config_providers.py':[
  ('    "discover_models", "extra_body", "extra_headers", "capabilities", "ssl_ca_cert", "ssl_verify"}',
   '    "discover_models", "extra_body", "extra_headers", "capabilities", "ssl_ca_cert", "ssl_verify",\n    "interaction_mode", "max_input_characters", "proactive_questions"}'),
  ('        ("discover_models", lambda v: isinstance(v, (bool, str))),\n',
   '        ("discover_models", lambda v: isinstance(v, (bool, str))),\n        ("interaction_mode", lambda v: v in ("agent", "stateful_chat")),\n        ("max_input_characters", lambda v: type(v) is int and v > 0),\n        ("proactive_questions", lambda v: type(v) is bool),\n'),
  ('        "ssl_ca_cert", "ssl_verify"):\n',
   '        "ssl_ca_cert", "ssl_verify", "interaction_mode", "max_input_characters", "proactive_questions"):\n')],
 'agent/agent_init.py':[
  ('    _load_tools(agent, enabled_toolsets, disabled_toolsets)\n',
   "    from hermes_cli.config import load_config_readonly as _stateful_load_config\n    from agent.stateful_chat import configure_stateful_chat\n    _stateful_chat = configure_stateful_chat(agent, _stateful_load_config())\n    if not _stateful_chat:\n        _load_tools(agent, enabled_toolsets, disabled_toolsets)\n"),
  ('    _init_memory(agent, _agent_cfg, skip_memory, platform)\n','    _init_memory(agent, _agent_cfg, skip_memory or _stateful_chat, platform)\n'),
  ('    _enforce_minimum_context(agent)\n',
   '    if _stateful_chat:\n        agent.context_compressor._stateful_input_limit_characters = agent._stateful_chat_max_input_characters\n    _enforce_minimum_context(agent)\n'),
  ('    if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH and not _allow_lmstudio_explicit_below_floor:\n',
   '    _stateful_tool_free = getattr(agent, "_stateful_chat_backend", False) and not getattr(agent, "tools", None)\n    if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH and not (_allow_lmstudio_explicit_below_floor or _stateful_tool_free):\n'),
  ('def _inject_context_engine_tools(agent):\n','def _inject_context_engine_tools(agent):\n    if getattr(agent, "_stateful_chat_backend", False):\n        return\n')],
 'agent/system_prompt.py':[
  ('    # Model context window scales the context-file caps; stable per conversation.\n',
   '    if getattr(agent, "_stateful_chat_backend", False):\n        return {"stable": "", "context": "", "volatile": ""}\n    # Model context window scales the context-file caps; stable per conversation.\n')],
 'agent/context_breakdown.py':[
  ('    used = max(0, getattr(compressor, "last_prompt_tokens", 0) or 0)\n',
   '    input_limit = getattr(compressor, "_stateful_input_limit_characters", None)\n    if input_limit is not None:\n        return {"context_mode": "backend_state", "input_limit_characters": input_limit,\n                "context_max": 0, "context_used": 0, "context_percent": 0, "context_source": "backend_state", "context_estimated": False, "request_load": getattr(compressor, "_stateful_request_load", None)}\n    used = max(0, getattr(compressor, "last_prompt_tokens", 0) or 0)\n'),
  ('    messages = messages or []\n',
   '    if getattr(agent, "_stateful_chat_backend", False):\n        from agent.stateful_chat import stateful_workload_display\n        load = stateful_workload_display(agent)\n        return {"categories": [], "estimated_total": 0, "model": agent.model,\n                **context_usage_fields(agent.context_compressor), "request_load": load}\n    messages = messages or []\n')],
 'agent/turn_api_request.py':[
  ('    api_kwargs = apply_vision_request(agent, api_kwargs)\n',
   '    api_kwargs = apply_vision_request(agent, api_kwargs)\n    if getattr(agent, "_stateful_chat_backend", False):\n        extra = dict(api_kwargs.get("extra_body") or {})\n        extra["scarx_request_id"] = agent._stateful_request_id\n        extra["scarx_session_id"] = agent.session_id\n        extra["scarx_question_surface"] = getattr(agent, "_stateful_question_surface", "inline")\n        api_kwargs["extra_body"] = extra\n')],
 'apps/desktop/src/types/hermes.ts':[
  ('export interface UsageStats {\n',
   "export interface RequestLoad {\n  phase?: string\n  modality?: string\n  input_characters: number\n  input_utf8_bytes: number\n  image_count: number\n  neural_image_pixels?: number\n  prompt_features_nonzero?: number\n  neural_input_channels?: number\n  neural_input_buffer_bytes?: number\n  active_neurons?: number\n  allocated_neurons?: number\n  max_neurons?: number\n  output_connected_neurons?: number\n  neural_arrays_bytes?: number\n  process_rss_bytes?: number\n  source?: string\n  allocation_reason?: string\n}\n\nexport interface UsageStats {\n  context_mode?: 'backend_state'\n  input_limit_characters?: number\n  request_load?: RequestLoad | null\n"),
  ('export interface ContextBreakdown {\n',
   "export interface ContextBreakdown {\n  context_mode?: 'backend_state'\n  input_limit_characters?: number\n  request_load?: RequestLoad | null\n")],
 'apps/desktop/src/lib/statusbar.tsx':[
  ('export function usageContextLabel(usage: UsageStats): string {\n',
   "export function usageContextLabel(usage: UsageStats): string {\n  if (usage.context_mode === 'backend_state') {\n    const load = usage.request_load\n    return load ? `Nạp ${load.input_characters} ký tự${load.image_count ? ` + ${load.image_count} ảnh` : ''}${load.active_neurons ? ` · ${load.active_neurons} neuron` : ''}` : 'Chưa có lượt nạp'\n  }\n"),
  ('export function contextBarLabel(usage: UsageStats): string {\n',
   "export function contextBarLabel(usage: UsageStats): string {\n  if (usage.context_mode === 'backend_state') return ''\n")],
 'apps/desktop/src/app/shell/hooks/use-statusbar-items.tsx':[
  ('            context_source: contextBreakdown.context_source,\n',
   '            context_source: contextBreakdown.context_source,\n            context_mode: contextBreakdown.context_mode,\n            input_limit_characters: contextBreakdown.input_limit_characters,\n            request_load: contextBreakdown.request_load,\n')],
 'apps/desktop/src/app/shell/context-usage-panel.tsx':[
  ('  const segmentTotal = categories.reduce',
   '''  if (usage.context_mode === 'backend_state') {
    return (
      <div className="flex w-72 flex-col gap-3 p-3 text-[0.75rem]" data-slot="stateful-memory-panel">
        <p className="font-medium text-foreground">Thông tin nạp cho lượt xử lý</p>
        <p>{usage.request_load ? `${usage.request_load.input_characters} / ${usage.input_limit_characters} ký tự · ${usage.request_load.input_utf8_bytes} byte UTF-8 · ${usage.request_load.image_count} ảnh` : 'Chưa có lượt nạp trong phiên này.'}</p>
        {usage.request_load?.active_neurons != null && <p>Neuron đang tính: {usage.request_load.active_neurons} · đã cấp: {usage.request_load.allocated_neurons} · trần: {usage.request_load.max_neurons}</p>}
        {usage.request_load?.neural_input_buffer_bytes != null && <p>Vector đầu vào: {usage.request_load.neural_input_channels} kênh / {usage.request_load.neural_input_buffer_bytes} byte. Prompt có {usage.request_load.prompt_features_nonzero} kênh khác 0.</p>}
        {usage.request_load?.neural_image_pixels != null && <p>Ảnh vào lõi: {usage.request_load.neural_image_pixels} pixel.</p>}
        <p className="text-muted-foreground">Lượt nạp gần nhất; tự đổi theo đầu vào. Không gửi lại lịch sử, không nạp tools hoặc truy xuất. Trọng số core vẫn nằm trong RAM.</p>
        <p className="text-muted-foreground">Neuron mới cần học; cấp nhiều hơn chưa chứng minh trả lời tốt hơn. Dung lượng trí nhớ và kỹ năng: chưa đo.</p>
      </div>
    )
  }

  const segmentTotal = categories.reduce''')],
 'agent/title_generator.py':[
  ('    if provider in {"custom", "local", "llama-cpp"} and host in {"127.0.0.1", "localhost", "::1"}:\n',
   '    if (provider in {"custom", "local", "llama-cpp"} or provider.startswith("custom:")) and host in {"127.0.0.1", "localhost", "::1"}:\n')],
 'agent/turn_request_assembly.py':[
  ('    api_messages, effective_system = build_api_messages(\n',
   '    if getattr(agent, "_stateful_chat_backend", False):\n        from copy import deepcopy\n        api_messages = [{"role": "user", "content": deepcopy(agent._stateful_chat_raw_user_message)}]\n        approx_tokens = estimate_messages_tokens_rough(api_messages)\n        return AssembledRequest("fallthrough", api_messages, [], None, None, approx_tokens, approx_tokens, approx_tokens * 4)\n\n    api_messages, effective_system = build_api_messages(\n')],
 'agent/conversation_loop.py':[
  ('    if moa_config is None:\n',
   '    if getattr(agent, "_stateful_chat_backend", False):\n        from agent.stateful_chat import prepare_stateful_chat_turn\n        if moa_config:\n            raise ValueError("stateful_chat does not support MoA orchestration")\n        user_message = prepare_stateful_chat_turn(agent, user_message)\n        conversation_history = None\n        system_message = None\n    if moa_config is None:\n')]
}
updates={};diffs=[]
baseline={};known_hashes={}
for receipt_path in sorted((root/'local/hermes-stateful-backups').glob('*/receipt.json')):
    for relative,record in json.loads(receipt_path.read_text(encoding='utf-8'))['files'].items():
        known_hashes.setdefault(relative,set()).add(record['after'])
        if relative not in baseline and record['before'] is not None:baseline[relative]=(receipt_path.parent/relative).read_text(encoding='utf-8')
for relative,replacements in patches.items():
    path=hermes/relative;original=path.read_bytes();text=original.decode('utf-8')
    if relative in baseline:
        if hashlib.sha256(original).hexdigest() not in known_hashes[relative]:raise ValueError('Hermes file edited since our snapshot; review before replacement: '+relative)
        updated=baseline[relative]
    else:updated=text
    for old,new in replacements:
        if new in updated:continue
        if updated.count(old)!=1:raise ValueError(f'Unsupported/changed Hermes source anchor: {relative}')
        updated=updated.replace(old,new,1)
    if updated!=text:
        if path.suffix=='.py':compile(updated,str(path),'exec')
        updates[relative]=(original,updated.encode('utf-8'))
        diffs.extend(difflib.unified_diff(text.splitlines(True),updated.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
for relative,source in [('agent/stateful_chat.py','stateful_chat.py'),('agent/stateful_questions.py','stateful_questions.py'),('tui_gateway/controller_questions.py','controller_questions.py'),('tests/tui_gateway/test_controller_questions.py','test_controller_questions.py'),('tests/agent/test_stateful_chat.py','test_stateful_chat.py'),('tests/agent/test_stateful_questions.py','test_stateful_questions.py'),('apps/desktop/src/app/shell/stateful-memory-panel.test.tsx','stateful-memory-panel.test.tsx'),('apps/desktop/src/app/session/hooks/use-message-stream/gateway-event/input-requests.standalone.test.ts','controller-question-lifecycle.test.ts')]:
    path=hermes/relative;data=(root/'integrations/hermes'/source).read_bytes()
    original=path.read_bytes() if path.exists() else None
    if original!=data:
        if original is not None:
            known=[]
            for receipt in (root/'local/hermes-stateful-backups').glob('*/receipt.json'):
                known.append(json.loads(receipt.read_text(encoding='utf-8')).get('files',{}).get(relative,{}).get('after'))
            if hashlib.sha256(original).hexdigest() not in known:raise ValueError('Existing helper differs; review before replacement: '+relative)
        updates[relative]=(original,data)
        diffs.extend(difflib.unified_diff(original.decode().splitlines(True) if original else [],data.decode().splitlines(True),fromfile='a/'+relative if original else '/dev/null',tofile='b/'+relative))
def write_full_patch():
    originals={}
    for receipt in sorted((root/'local/hermes-stateful-backups').glob('*/receipt.json')):
        for relative,record in json.loads(receipt.read_text(encoding='utf-8'))['files'].items():
            if relative not in originals:
                originals[relative]=(receipt.parent/relative).read_text(encoding='utf-8') if record['before'] is not None else ''
    full=[]
    for relative,original in originals.items():
        current=(hermes/relative).read_text(encoding='utf-8')
        full.extend(difflib.unified_diff(original.splitlines(True),current.splitlines(True),fromfile='a/'+relative if original else '/dev/null',tofile='b/'+relative))
    (root/'integrations/hermes/stateful-chat.patch').write_text(''.join(full),encoding='utf-8')

if not updates:
    write_full_patch();print(json.dumps({'already_installed':True}));raise SystemExit(0)
backup=root/'local/hermes-stateful-backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');backup.mkdir(parents=True)
receipt={'hermes_root':str(hermes),'files':{}}
for relative,(original,data) in updates.items():
    if original is not None:
        copy=backup/relative;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(original)
    receipt['files'][relative]={'before':hashlib.sha256(original).hexdigest() if original is not None else None,'after':hashlib.sha256(data).hexdigest()}
(root/'integrations/hermes/stateful-chat.patch').write_text(''.join(diffs),encoding='utf-8')
for relative,(original,data) in updates.items():
    path=hermes/relative
    if (path.read_bytes() if path.exists() else None)!=original:raise RuntimeError('Concurrent edit: '+relative)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.stateful.tmp')
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
(backup/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
write_full_patch()
print(json.dumps({'updated_files':list(updates),'backup':str(backup),'neural_core_modified':False},indent=2))
