"""Small user-shareable supplement; preserves the original delivered package."""
from pathlib import Path
import hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'docs/research/forensic-r6-20261111'
    names=['RAW_EVENT_LABEL_ADDENDUM.md','RAW_EVENT_ORACLE_FOLLOWUP_PROTOCOL.md',
           'RAW_EVENT_ORACLE_FOLLOWUP_ERRATUM.md','RAW_EVENT_DELIVERY_VERIFICATION.json']
    files=[folder/name for name in names]
    files+=[ROOT/'src/scarx/evaluation/forensic/event_oracle.py',ROOT/'tests/test_forensic_event_oracle.py',
            ROOT/'scripts/forensic_gemini_event_audit.py',Path(__file__),
            ROOT/'artifacts/forensic-r6/independent-tests-with-events.xml',ROOT/'artifacts/forensic-r6/gemini-events-receipt.json']
    for run in ROOT.glob('runs/*forensic-gemini-events-v1*'):files.extend(p for p in run.rglob('*') if p.is_file())
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    records={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    if any('local' in Path(name).parts or '.git' in Path(name).parts for name in records):raise ValueError('Private content path refused')
    manifest={'source_commit':commit,'parent_delivery_commit':'24453bdcfbd61f9fa36804b98e6f4d23af180182',
        'parent_delivery_zip_sha256':'3268cc1d00e148e0c0915d4eccfbeca6976618275fa1f0ec7c56c104e9572d59',
        'neural_forward_calls':0,'learning_updates':0,'network_upload':False,'files':records}
    target=ROOT/'artifacts/forensic-r6/SCARX_RAW_EVENT_ADDENDUM_2026-11-11.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('ADDENDUM_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        for name in records:archive.write(ROOT/name,name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(name)).hexdigest()==digest for name,digest in records.items())
    print(json.dumps({'package':str(target),'source_commit':commit,'bytes':target.stat().st_size,
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
