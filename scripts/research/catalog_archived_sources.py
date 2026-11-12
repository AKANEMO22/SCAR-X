"""Import immutable historical REPORTS/protocols/metric receipts, never peer code.

One readable copy per owner/original path; all versioned source SHAs remain
indexed. Original bytes are preserved, including retracted/unverified claims.
The canonical ledger supplies the assessment; originals are not authority.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2]
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source-index',required=True);a=p.parse_args()
    entries=json.loads(Path(a.source_index).read_text(encoding='utf-8'));base=ROOT/'docs/research/archive'
    if (base/'source-index.json').exists():raise ValueError('Do not overwrite the historical source index')
    versions={};branches=[]
    for entry in entries:
        record={**entry,'evidence_status':'AUTHOR_RECORD_NOT_AUTOMATICALLY_VERIFIED','source_urls':{}}
        for path in dict.fromkeys(entry['reports']+entry['protocols']+entry['metrics']):
            raw=git('show',entry['sha']+':'+path)
            # Preserve larger artifacts through the immutable Git commit link.
            url='https://github.com/AKANEMO22/SCAR-X/blob/'+entry['sha']+'/'+path
            record['source_urls'][path]=url
            if len(raw)>5_000_000:continue
            key=(entry['owner'],path);item={'commit':entry['sha'],'branch':entry['branch'],'sha256':sha(raw),'bytes':len(raw),'source_url':url}
            variants=versions.setdefault(key,[])
            if not any(v['sha256']==item['sha256'] for v in variants):variants.append(item)
        branches.append(record)
    files=[]
    for (owner,path),variants in sorted(versions.items()):
        # Input branch list is chronological within each owner. Latest recorded
        # source is the readable copy, not a claim that its conclusions are valid.
        chosen=variants[-1];dest=base/owner/'original'/path;dest.parent.mkdir(parents=True,exist_ok=True)
        raw=git('show',chosen['commit']+':'+path);dest.write_bytes(raw)
        if sha(dest.read_bytes())!=chosen['sha256']:raise ValueError('Historical report copy mismatch')
        files.append({'owner':owner,'original_path':path,'local_path':str(dest.relative_to(ROOT)).replace('\\','/'),
                      'selected_source':chosen,'versions':variants})
    base.mkdir(parents=True,exist_ok=True)
    index={'date':'2026-11-13','scope':'Historical reports/protocols/receipts only; unverified/retracted claims preserved, no peer learner imported',
        'active_R13_1':'IN_PROGRESS_EXCLUDED; no active files read/copied/changed, only ancestor f4272ba historical R13 source',
        'files':files,'branches':branches}
    (base/'source-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for owner in sorted({v['owner'] for v in files}):
        rows=['# '+owner+' — nguồn lịch sử','',
          'Các bản gốc giữ nguyên cả claim sai/bị rút lại. Đọc [sổ thí nghiệm](../../EXPERIMENT_LEDGER.md) để biết kết luận đã kiểm. Không copy learner vào serving.','',
          '| Tệp | Source SHA | SHA256 nội dung |','|---|---|---|']
        for item in files:
            if item['owner']!=owner:continue
            label=item['original_path'];relative='original/'+label;source=item['selected_source']
            rows.append(f"| [{label}]({relative}) | [{source['commit'][:12]}]({source['source_url']}) | `{source['sha256']}` |")
        (base/owner/'README.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print(json.dumps({'historical_files':len(files),'source_branches':len(branches),'bytes':sum(v['selected_source']['bytes'] for v in files)}))
if __name__=='__main__':main()
