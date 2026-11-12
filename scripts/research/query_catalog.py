"""Query research results without loading models, contacting APIs or training."""
from pathlib import Path
import argparse,hashlib,json,sys
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('list','search','show','verify'));p.add_argument('query',nargs='?',default='');a=p.parse_args()
    data=json.loads((ROOT/'docs/research/experiment-catalog.json').read_text(encoding='utf-8'));entries=data['entries']
    if a.action=='verify':
        count=0
        for entry in entries:
            for path,expected in entry['report_sha256'].items():
                file=(ROOT/path).resolve()
                if not file.is_relative_to(ROOT) or hashlib.sha256(file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:raise ValueError('Missing/changed report: '+path)
                count+=1
        archive=json.loads((ROOT/'docs/research/archive/source-index.json').read_text(encoding='utf-8'))
        for item in archive['files']:
            if hashlib.sha256((ROOT/item['local_path']).read_bytes()).hexdigest()!=item['selected_source']['sha256']:raise ValueError('Historical source copy changed')
        print(json.dumps({'catalog_entries':len(entries),'report_references_verified':count,'immutable_source_copies':len(archive['files']),'new_model_calls':0}));return
    if a.action=='show':
        matches=[v for v in entries if v['id'].casefold()==a.query.casefold()]
        if not matches:raise SystemExit('Unknown experiment ID')
        print(json.dumps(matches[0],ensure_ascii=False,indent=2));return
    if a.action=='search':entries=[v for v in entries if a.query.casefold() in json.dumps(v,ensure_ascii=False).casefold()]
    for v in entries:print(v['id']+' | '+v['status']+' | '+v['audit_level']+' | '+v['title'])
if __name__=='__main__':main()
