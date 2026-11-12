"""Exact executed capacity source bytes from Git+hash, no checkout mutation."""
from pathlib import Path
import argparse,json
from export_research_source_bundle import reconstruct,sha
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=(ROOT/a.run).resolve()
    if run.parent!=ROOT/'runs':raise ValueError('Owned run only')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'));out=run/'source-byte-bundle';out.mkdir(exist_ok=False);receipts={}
    for path,h in manifest['source_lock']['files'].items():
        name=Path(path)
        if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe source path')
        mode,raw=reconstruct(ROOT,manifest['source_commit'],path,h);dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        receipts['source/'+path]={'sha256':h,'git_commit':manifest['source_commit'],'eol_reconstruction':mode}
    peer=ROOT.parent/'review-gemini-r8-607ffca'
    for path,h in manifest['source_lock']['peer'].items():
        mode,raw=reconstruct(peer,manifest['peer_sha'],path,h);dest=out/'peer607'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        receipts['peer607/'+path]={'sha256':h,'git_commit':manifest['peer_sha'],'eol_reconstruction':mode}
    with (out/'receipt.json').open('x',encoding='utf-8') as f:json.dump({'files':receipts,'parent_manifest_sha256':sha((run/'manifest.json').read_bytes()),
       'scope':'Overlay ONLY into a NEW isolated reproduction checkout; never mutate main/live/completed-source checkout','new_training_updates':0},f,indent=2)
    print(str(out))
if __name__=='__main__':main()
