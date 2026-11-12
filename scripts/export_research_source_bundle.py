"""Reconstruct exact executed source bytes from pinned Git objects + EOL hash.

Writes a NEW bundle only; never edits any checkout, learner or prior evidence.
Handles mixed Windows CRLF/new LF files without relaxing historical checksums.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def reconstruct(repo,commit,path,expected):
    raw=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+path])
    lf=raw.replace(b'\r\n',b'\n')
    for mode,value in (('git_blob',raw),('LF',lf),('CRLF',lf.replace(b'\n',b'\r\n'))):
        if sha(value)==expected:return mode,value
    raise ValueError('Pinned Git object cannot reconstruct executed body: '+path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=(ROOT/a.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned run required')
    m=json.loads((run/'manifest.json').read_text(encoding='utf-8'));out=run/'source-byte-bundle';out.mkdir(exist_ok=False)
    receipts={}
    for path,h in m['source_sha256'].items():
        name=Path(path)
        if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe source path')
        mode,raw=reconstruct(ROOT,m['source_commit'],path,h);dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        receipts['source/'+path]={'sha256':h,'eol_reconstruction':mode,'git_commit':m['source_commit']}
    peer=ROOT.parent/'review-gemini-r8-607ffca'
    for path,h in m['peer_sha256'].items():
        mode,raw=reconstruct(peer,m['peer_sha'],path,h);dest=out/'peer607'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        receipts['peer607/'+path]={'sha256':h,'eol_reconstruction':mode,'git_commit':m['peer_sha']}
    receipt={'files':receipts,'parent_manifest_sha256':sha((run/'manifest.json').read_bytes()),'new_training_updates':0,
       'scope':'Exact source bytes; overlay only into a NEW isolated reproduction checkout at recorded commits. Never overwrite main/live/source of completed runs.'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(str(out))
if __name__=='__main__':main()
