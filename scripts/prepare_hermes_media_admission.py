"""Exact-anchor Hermes transport extension for the real video-upload consumer.

Prepare public source candidates and byte backups, never touch live source in
prepare mode. Apply is a CAS overlay after inspection; no config/DB changes.
"""

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
HERMES=Path(os.environ['LOCALAPPDATA'])/'hermes/hermes-agent'


def replace(raw,before,after):
    text=raw.decode('utf-8').replace('\r\n','\n')
    if after in text:return raw
    if text.count(before)!=1:raise ValueError('Hermes source anchor changed: '+before[:100])
    output=text.replace(before,after,1).encode()
    return output.replace(b'\n',b'\r\n') if b'\r\n' in raw else output


def prepare():
    run=ROOT/'local/hermes-media-admission'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run.mkdir(parents=True)
    changes={
      'gateway/session_ingress.py':('''        with await _execution_scope(authority, home):
            yield''','''        with await _execution_scope(authority, home):
            from gateway.session_submission import submission_scope
            with submission_scope(row.get('request_id')):
                yield'''),
      'apps/desktop/src/app/chat/composer/contrib.ts':('''export interface ComposerDraft {
  text: string
  attachments?: ComposerAttachment[]
}''','''export interface ComposerDraft {
  text: string
  attachments?: ComposerAttachment[]
  /** Called on this exact send after session resolution and before admission. */
  prepareSession?: (identity: {
    runtimeSessionId: string; storedSessionId: null | string; submissionId: string; text: string
  }) => Promise<void> | void
}'''),
      'apps/desktop/src/app/chat/composer/hooks/use-middleware-submit.ts':('''      return onSubmitProp(draft.text, { ...options, attachments: draft.attachments })''','''      return onSubmitProp(draft.text, {
        ...options, attachments: draft.attachments,
        ...(draft.prepareSession ? {
          onSessionPrepared: async identity => {
            await options?.onSessionPrepared?.(identity)
            await draft.prepareSession?.(identity)
          }
        } : {})
      })'''),
      'apps/desktop/src/app/session/hooks/use-prompt-actions/utils.ts':('''  onComposerScopeAssigned?: (scope: string) => void''','''  onComposerScopeAssigned?: (scope: string) => void
  /** Request metadata prepared for this exact resolved admission, never global focus. */
  onSessionPrepared?: (identity: {
    runtimeSessionId: string; storedSessionId: null | string; submissionId: string; text: string
  }) => Promise<void> | void'''),
      'apps/desktop/src/app/session/hooks/use-prompt-actions/submit.ts':('''                const params: Record<string, unknown> = { ...prepared.params, session_id: liveId }

                try {''','''                const params: Record<string, unknown> = { ...prepared.params, session_id: liveId }
                await options?.onSessionPrepared?.({
                  runtimeSessionId: liveId, storedSessionId: recoverStoredSessionId,
                  submissionId, text
                })
                const metadataDrift = sessionDriftReason()
                if (metadataDrift) { throw new SessionRecoveryAborted(metadataDrift, liveId) }

                try {'''),
      'apps/desktop/src/sdk/composer.ts':('''export const composerHost = {''','''export const composerHost = {
  /** Composer metadata can wait for the exact resolved session/admission. */
  preparedSessionVersion: 1 as const,'''),
    }
    # File payloads are generic Hermes metadata boundaries, not new model tools.
    mapping={name:replace((HERMES/name).read_bytes(),before,after) for name,(before,after) in changes.items()}
    mapping['gateway/session_submission.py']=(ROOT/'integrations/hermes/session_submission.py').read_bytes()
    mapping['agent/scarx_session_transport.py']=(ROOT/'integrations/hermes/scarx_session_transport.py').read_bytes()
    records={}
    for name,raw in mapping.items():
        target=HERMES/name;before=target.read_bytes() if target.is_file() else None
        if name.endswith('.py'):compile(raw,name,'exec')
        if before is not None:
            backup=run/'before'/name;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(before)
        candidate=run/'candidate'/name;candidate.parent.mkdir(parents=True,exist_ok=True);candidate.write_bytes(raw)
        records[name]={'before_sha256':hashlib.sha256(before).hexdigest() if before else None,'after_sha256':hashlib.sha256(raw).hexdigest()}
    (run/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (ROOT/'local/current-hermes-media-admission.txt').write_text(str(run),encoding='utf-8')
    print(json.dumps({'prepared_directory':str(run),'source_files':len(records),'live_source_applied':False}))


def apply(run):
    run=run.resolve()
    if not run.is_relative_to(ROOT/'local/hermes-media-admission'):raise ValueError('Unexpected prepared source location')
    records=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    changes=[]
    for name,info in records.items():
        path=(HERMES/name).resolve();candidate=(run/'candidate'/name).read_bytes()
        if not path.is_relative_to(HERMES):raise ValueError('Hermes source path escape')
        if hashlib.sha256(candidate).hexdigest()!=info['after_sha256']:raise ValueError('Candidate checksum changed')
        before=path.read_bytes() if path.is_file() else None
        if before==candidate:continue
        if (hashlib.sha256(before).hexdigest() if before else None)!=info['before_sha256']:raise ValueError('Concurrent Hermes source edit: '+name)
        changes.append((path,before,candidate))
    for path,before,candidate in changes:
        fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.media-admission-',suffix='.tmp')
        try:
            with os.fdopen(fd,'wb') as out:out.write(candidate);out.flush();os.fsync(out.fileno())
            if (path.read_bytes() if path.is_file() else None)!=before:raise ValueError('Concurrent source edit during apply')
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
    with (run/'source-applied.json').open('x',encoding='utf-8') as out:json.dump({'files_changed':len(changes),'configuration_or_database_modified':False,'live_process_reloaded':False},out,indent=2)
    print(json.dumps({'source_files_applied':len(changes),'source_backup':str(run/'before'),'process_reload_required':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',type=Path);args=parser.parse_args()
    apply(args.apply) if args.apply else prepare()
